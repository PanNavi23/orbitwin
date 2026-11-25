#!/usr/bin/env python3
"""Deterministic GmatConsole test double. Never use for scientific validation."""

from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime
from pathlib import Path

# orbitwin.gmat_runner.is_test_double reads this line: the double says what it is.
ORBITWIN_GMAT_TEST_DOUBLE = True

CREATE_REPORT = re.compile(r"Create\s+ReportFile\s+(\w+)\s*;", re.IGNORECASE)
# The optional GMAT keyword, and the value up to the last ;, as gmat_parser._ASSIGN_RE
# reads them on a line's code part; GMAT also runs a line with no ; at all.
ASSIGN = re.compile(r"^\s*(?:GMAT\s+)?(\w+)\.(\w+)\s*=\s*(.*?)\s*;?\s*$", re.IGNORECASE)
GOAL = re.compile(r"\.Elapsed(Secs|Days)\s*=\s*([-+0-9.eE]+)", re.IGNORECASE)


#: GMAT ends a line at CR, LF or CRLF only; str.splitlines also splits at VT, FF
#: and \x1c-\x1e, which GMAT reads as text.
LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")


def _code(line: str) -> str:
    """gmat_parser._code_part, copied: this double imports nothing from orbitwin.

    tests/unit/test_script_line_model.py holds the two to the same answers.
    """
    nul = line.find("\x00")
    if nul >= 0:
        line = line[:nul]
    percent = line.find("%")
    if percent < 0:
        return line
    quote = line.find("'")
    if 0 <= quote < percent < line.find("'", quote + 1):
        return line
    return line[:percent]


def gmat_text(raw: str) -> str:
    """The script as GMAT reads it: up to a NUL, CR/LF lines, comments dropped.

    This double's own `% FAKE_GMAT_...` directives are kept wherever they sit, after
    code or a NUL included, each on a line of its own.
    """
    read, _, unread = raw.partition("\x00")
    lines = []
    for line in LINE.findall(read):
        body = line.rstrip("\r\n")
        lines.append(_code(body))
        if "% FAKE_GMAT" in body:
            lines.append(body[body.index("% FAKE_GMAT"):])
    lines += [
        line[line.index("% FAKE_GMAT"):].rstrip("\r\n").split("\x00", 1)[0]
        for line in LINE.findall(unread)
        if "% FAKE_GMAT" in line
    ]
    return "\n".join(lines) + "\n"


ALIGNMENT = re.compile(
    r"^\s*Propagate\s+\w+\((?P<early>\w+)\)\s*\{\s*(?P=early)\.(?:\w+ModJulian\s*=|Elapsed)"
)


def alignment_of(text: str) -> tuple[str, str, float] | None:
    """(early, late, seconds) when the first mission command flies one craft to the other's epoch."""
    mission = [
        line for line in text.split("BeginMissionSequence", 1)[-1].splitlines()[1:]
        if line.strip() and not line.strip().startswith(
            ("%", "Toggle", "PenUp", "PenDown", "MarkPoint", "ClearPlot"))
    ]
    match = ALIGNMENT.match(mission[0]) if mission else None
    if match is None:
        return None
    resources = text.split("BeginMissionSequence", 1)[0]
    fields: dict[str, dict[str, str]] = {}
    for line in resources.splitlines():
        assign = ASSIGN.match(line)
        if assign:
            fields.setdefault(assign.group(1), {})[assign.group(2)] = assign.group(3).strip("'\"")
    craft = re.findall(r"^\s*Create\s+Spacecraft\s+(\w+)\s*;", resources, re.MULTILINE)
    early = match.group("early")
    late = next(name for name in craft if name != early)
    offset_s = (_epochs(fields[late])["A1"] - _epochs(fields[early])["A1"]) * 86400.0
    if "ModJulian" not in mission[0] and offset_s <= 1.0e-6:
        return None  # one craft for a while, as the coast run flies its target
    return early, late, offset_s


#: A propagator call flying two craft at once: Prop(A, B).
JOINT_CALL = re.compile(r"\b\w+\(\s*\w+\s*,\s*\w+[^)]*\)")


def joint_apart(text: str) -> bool:
    """R2026a aborts a joint Propagate of craft on different epochs with no alignment first."""
    if alignment_of(text) is not None:
        return False
    resources, _, mission = text.partition("BeginMissionSequence")
    if not any(
        line.strip().startswith("Propagate") and JOINT_CALL.search(line)
        for line in mission.splitlines()
    ):
        return False
    fields: dict[str, dict[str, str]] = {}
    for line in resources.splitlines():
        assign = ASSIGN.match(line)
        if assign:
            fields.setdefault(assign.group(1), {})[assign.group(2)] = assign.group(3).strip("'\"")
    craft = re.findall(r"^\s*Create\s+Spacecraft\s+(\w+)\s*;", resources, re.MULTILINE)
    try:
        a1 = [_epochs(fields.get(name, {}))["A1"] for name in craft]
    except (Unsupported, ValueError):
        return False
    return len(a1) == 2 and abs(a1[1] - a1[0]) * 86400.0 > 1.0e-6


def _mission_lines(text: str) -> list[str]:
    """The mission sequence without its alignment, whose rows are written apart."""
    lines = text.split("BeginMissionSequence", 1)[-1].splitlines()
    if alignment_of(text) is None:
        return lines
    first = next(index for index, line in enumerate(lines) if line.strip().startswith("Propagate"))
    return lines[:first] + lines[first + 1 :]


def mission_end_s(text: str) -> float:
    """Where the script's Propagate commands stop, summed over its segments."""
    total = 0.0
    for line in _mission_lines(text):
        if line.strip().startswith("Propagate"):
            match = GOAL.search(line)
            if match:
                total += float(match.group(2)) * (86400.0 if match.group(1).lower() == "days" else 1.0)
    return total


PROBE_REPORT = re.compile(r"^\s*Report\s+OrbitWinStateProbe\s+(?P<params>[^;]+);", re.MULTILINE)
#: GMAT's Earth; the solver's 398600.4418 differs, as it does against real GMAT.
MU_KM3_S2 = 398600.4415
MJD_ZERO = datetime(1941, 1, 5, 12)
#: TAI - UTC from 1999; enough for the templates this double is fed.
LEAP_SECONDS = (
    (datetime(1999, 1, 1), 32.0),
    (datetime(2006, 1, 1), 33.0),
    (datetime(2009, 1, 1), 34.0),
    (datetime(2012, 7, 1), 35.0),
    (datetime(2015, 7, 1), 36.0),
    (datetime(2017, 1, 1), 37.0),
)
A1_MINUS_TAI_S = 0.0343817
TT_MINUS_TAI_S = 32.184


class Unsupported(Exception):
    """Something only real GMAT can convert."""


def _leap_s(mjd: float) -> float:
    instant = MJD_ZERO.timestamp() + mjd * 86400.0
    usable = [leap for start, leap in LEAP_SECONDS if start.timestamp() <= instant]
    if not usable:
        raise Unsupported(f"the test double has no leap seconds before 1999 (MJD {mjd})")
    return usable[-1]


def _epochs(fields: dict[str, str]) -> dict[str, float]:
    """ModJulian epochs in each scale from Epoch and DateFormat, TDB taken as TT."""
    text = fields.get("Epoch", "21545")
    form = fields.get("DateFormat", "TAIModJulian")
    if form.endswith("Gregorian"):
        scale = form.removesuffix("Gregorian")
        moment = datetime.strptime(text, "%d %b %Y %H:%M:%S.%f")
        mjd = (moment - MJD_ZERO).total_seconds() / 86400.0
    elif form.endswith("ModJulian"):
        scale = form.removesuffix("ModJulian")
        mjd = float(text)
    else:
        raise Unsupported(f"DateFormat {form}")
    if scale == "UTC":
        tai = mjd + _leap_s(mjd) / 86400.0
    elif scale in ("TAI", "A1", "TT"):
        tai = mjd - {"TAI": 0.0, "A1": A1_MINUS_TAI_S, "TT": TT_MINUS_TAI_S}[scale] / 86400.0
    else:
        raise Unsupported(f"DateFormat {form}")
    utc = tai - _leap_s(tai) / 86400.0
    tt = tai + TT_MINUS_TAI_S / 86400.0
    return {"A1": tai + A1_MINUS_TAI_S / 86400.0, "UTC": utc, "TAI": tai, "TT": tt, "TDB": tt}


#: Orbit-state and mission-only epoch fields the double does not model beyond
#: its type's six; real GMAT refuses or reads them otherwise (gmat_parser.STATE_TYPE_FIELDS).
_UNMODELLED = re.compile(
    r"(SMA|ECC|INC|RAAN|AOP|TA|X|Y|Z|VX|VY|VZ|MA|EA|HA|RadPer|RadApo|RMAG|RA|DEC|VMAG|AZI|FPA"
    r"|RAV|DECV|MLONG|TLONG|SemilatusRectum|(Planetodetic|Equinoctial|AltEquinoctial"
    r"|ModEquinoctial|Delaunay|Incoming|Outgoing|Brouwer)\w+"
    r"|(A1|TAI|UTC|TT|TDB)(ModJulian|Gregorian)|CurrA1MJD|Element[1-6]|Cartesian(V?[XYZ]))",
    re.IGNORECASE,  # GMAT refuses a wrongly cased field ("ta", "Sma")
)
_SIX = {
    "Cartesian": {"X", "Y", "Z", "VX", "VY", "VZ"},
    "Keplerian": {"SMA", "ECC", "INC", "RAAN", "AOP", "TA"},
}


_GOVERNING = ("AnomalyType", "DisplayStateType", "CoordinateSystem", "DateFormat", "Epoch")
_GOVERNING_LOWER = {kept.lower() for kept in _GOVERNING}


def _state(name: str, fields: dict[str, str]) -> list[float]:
    cased = sorted(
        field for field in fields if field.lower() in _GOVERNING_LOWER and field not in _GOVERNING
    )
    if cased:
        raise Unsupported(f"{name}.{cased[0]} in the wrong case")
    system = fields.get("CoordinateSystem", "EarthMJ2000Eq")
    kind = fields.get("DisplayStateType", "Cartesian")
    if system != "EarthMJ2000Eq":
        raise Unsupported(f"{name}.CoordinateSystem = {system}")
    if kind not in _SIX:
        raise Unsupported(f"{name}.DisplayStateType = {kind}")
    other = sorted(
        field for field in fields if _UNMODELLED.fullmatch(field) and field not in _SIX[kind]
    )
    if other:
        raise Unsupported(f"{name}.{other[0]} beside DisplayStateType = {kind}")
    if kind == "Cartesian":
        return [float(fields[key]) for key in ("X", "Y", "Z", "VX", "VY", "VZ")]
    # GMAT reads the six by AnomalyType; MA put them 5.6e5 km away.
    if fields.get("AnomalyType", "TA") != "TA":
        raise Unsupported(f"{name}.AnomalyType = {fields['AnomalyType']}")
    sma, ecc = float(fields["SMA"]), float(fields["ECC"])
    inc, raan, aop, ta = (math.radians(float(fields[key])) for key in ("INC", "RAAN", "AOP", "TA"))
    p = sma * (1.0 - ecc * ecc)
    radius = p / (1.0 + ecc * math.cos(ta))
    speed = math.sqrt(MU_KM3_S2 / p)
    r_pf = (radius * math.cos(ta), radius * math.sin(ta))
    v_pf = (-speed * math.sin(ta), speed * (ecc + math.cos(ta)))
    co, so, ci, si, cw, sw = (
        math.cos(raan), math.sin(raan), math.cos(inc), math.sin(inc), math.cos(aop), math.sin(aop)
    )
    rotation = (
        (co * cw - so * sw * ci, -co * sw - so * cw * ci),
        (so * cw + co * sw * ci, -so * sw + co * cw * ci),
        (sw * si, cw * si),
    )
    position = [row[0] * r_pf[0] + row[1] * r_pf[1] for row in rotation]
    velocity = [row[0] * v_pf[0] + row[1] * v_pf[1] for row in rotation]
    return position + velocity


def answer_state_probe(text: str, out: Path) -> int:
    """Write the probe report for EarthMJ2000Eq craft; anything else needs real GMAT."""
    request = PROBE_REPORT.search(text)
    assert request is not None
    if "% FAKE_GMAT_PROBE_FAIL" in text:
        print("ERROR!!!!!! ---- forced fake probe failure", file=sys.stderr)
        return 3
    resources = text.split("BeginMissionSequence", 1)[0]
    # GMAT reads an included file in place.
    resources = re.sub(
        r"^\s*#Include\s+'([^']+)'.*$",
        lambda match: Path(match.group(1)).read_text(encoding="utf-8"),
        resources,
        flags=re.MULTILINE,
    )
    fields: dict[str, dict[str, str]] = {}
    at: dict[tuple[str, str], int] = {}
    for index, line in enumerate(resources.splitlines()):
        match = ASSIGN.match(line)
        if match:
            fields.setdefault(match.group(1), {})[match.group(2)] = match.group(3).strip("'\"")
            at[(match.group(1), match.group(2))] = index
    # GMAT reads Epoch in the DateFormat set when that line runs; R2026a put it 37 s off.
    early = sorted(
        name for (name, field), index in at.items()
        if field == "Epoch" and at.get((name, "DateFormat"), -1) > index
    )
    if early:
        print(f"ERROR!!!!!! ---- fake GmatConsole cannot convert {early[0]}.Epoch before its "
              "DateFormat", file=sys.stderr)
        return 1
    # ASSIGN does not read Epoch.<format>; without this the default epoch went out.
    scaled = re.search(
        r"^\s*(?:GMAT\s+)?(\w+)\.Epoch\.\w+\s*=", resources, re.MULTILINE | re.IGNORECASE
    )
    if scaled:
        print(f"ERROR!!!!!! ---- fake GmatConsole cannot convert {scaled.group(1)}.Epoch.<format>",
              file=sys.stderr)
        return 1
    filename = re.search(r"OrbitWinStateProbe\.Filename\s*=\s*'([^']+)'", resources)
    assert filename is not None
    parameters = request.group("params").split()
    values: list[float] = []
    try:
        for parameter in parameters:
            craft, _, quantity = parameter.partition(".")
            if quantity.endswith("ModJulian"):
                values.append(_epochs(fields.get(craft, {}))[quantity.removesuffix("ModJulian")])
            else:
                axis = ("X", "Y", "Z", "VX", "VY", "VZ").index(quantity.split(".")[-1])
                values.append(_state(craft, fields.get(craft, {}))[axis])
    except Unsupported as exc:
        print(f"ERROR!!!!!! ---- fake GmatConsole cannot convert {exc}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    # As GMAT does: every live file writer clears or writes its Filename, wherever it points.
    writers = r"^\s*Create\s+(?:ReportFile|EphemerisFile|EclipseLocator|ContactLocator)\s+(\w+)\s*;"
    for name in re.findall(writers, resources, re.MULTILINE):
        target = fields.get(name, {}).get("Filename")
        if name != "OrbitWinStateProbe" and target is not None:
            path = Path(target)
            (path if path.is_absolute() else out / path).write_text("", encoding="utf-8")
    (out / filename.group(1)).write_text(
        " ".join(parameters) + "\n" + " ".join(f"{value:.17g}" for value in values) + "\n",
        encoding="utf-8",
    )
    print("*** GMAT Integration test (Console version) successful! ***")
    return 0


def answer_alignment_probe(text: str, out: Path) -> int:
    """Rows of the alignment alone. Unlike GMAT the early craft does not move, so its
    state at t = 0 is its initial one; only the A1 epochs advance."""
    resources = text.split("BeginMissionSequence", 1)[0]
    fields: dict[str, dict[str, str]] = {}
    for line in resources.splitlines():
        match = ASSIGN.match(line)
        if match:
            fields.setdefault(match.group(1), {})[match.group(2)] = match.group(3).strip("'\"")
    aligned = alignment_of(text)
    if aligned is None:
        print("ERROR!!!!!! ---- fake alignment probe without an alignment", file=sys.stderr)
        return 1
    early, late, offset_s = aligned
    filename = fields["OrbitWinAlignmentProbe"]["Filename"]
    headers = [item.strip() for item in fields["OrbitWinAlignmentProbe"]["Add"].strip("{}").split(",")]
    rows = []
    try:
        for k in range(FROZEN_ROWS + 1):
            s_since = offset_s * k / FROZEN_ROWS
            row = []
            for header in headers:
                craft, _, quantity = header.partition(".")
                if quantity == "A1ModJulian":
                    own = s_since if craft == early else offset_s
                    row.append(_epochs(fields[craft])["A1"] + (own - (offset_s if craft == late else 0.0)) / 86400.0)
                else:
                    axis = ("X", "Y", "Z", "VX", "VY", "VZ").index(quantity.split(".")[-1])
                    row.append(_state(craft, fields[craft])[axis])
            rows.append(row)
    except Unsupported as exc:
        print(f"ERROR!!!!!! ---- fake GmatConsole cannot convert {exc}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    (out / filename).write_text(
        " ".join(headers) + "\n"
        + "\n".join(" ".join(f"{value:.17g}" for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )
    print("*** GMAT Integration test (Console version) successful! ***")
    return 0


# A command label is allowed, as GMAT and gmat_parser allow it.
MANEUVER = re.compile(r"^\s*Maneuver(?:\s+'[^']*')?\s+(\w+)\s*\(", re.IGNORECASE)
#: R2026a's aliases for Element1-3 (Burn.cpp); measured: the later assignment wins.
BURN_ALIASES = {"V": "Element1", "N": "Element2", "B": "Element3"}
#: diagnostic.BURN_RESOURCE_NAME.
BURN_REPORT = "OrbitWinBurnCheck"
#: Both craft drift along y at this speed whenever the script maneuvers, so
#: the pursuer's r x v is never zero and a VNB or LVLH burn has a frame. The
#: drift is common to both, so every relative quantity is as before.
DRIFT_KM_S = 7.5
#: A row this long before each burn carries the track's own velocity.
BEFORE_BURN_S = 1.0e-3
#: Rows an alignment writes before t = 0.
FROZEN_ROWS = 4


def maneuvers(text: str) -> list[tuple[float, str]]:
    """(epoch, ImpulsiveBurn) for each Maneuver: the Propagate goals before it, summed."""
    found, total = [], 0.0
    for line in _mission_lines(text):
        stripped = line.strip()
        if stripped.startswith("Propagate"):
            match = GOAL.search(line)
            if match:
                total += float(match.group(2)) * (86400.0 if match.group(1).lower() == "days" else 1.0)
        else:
            maneuver = MANEUVER.match(line)
            if maneuver:
                found.append((total, maneuver.group(1)))
    return found


def report_points(text: str) -> list[tuple[float, int | None, bool]]:
    """(epoch, the Maneuver it is next to, after it) for each burn-report Report command."""
    commands = [
        line.strip() for line in _mission_lines(text)
        if line.strip() not in ("", ";") and not line.strip().startswith("%")
    ]
    points: list[tuple[float, int | None, bool]] = []
    total, count, after = 0.0, 0, False
    for position, line in enumerate(commands):
        if line.startswith("Propagate"):
            match = GOAL.search(line)
            if match:
                total += float(match.group(2)) * (86400.0 if match.group(1).lower() == "days" else 1.0)
            after = False
        elif MANEUVER.match(line):
            count += 1
            after = True
        elif re.match(rf"Report\s+{BURN_REPORT}\b", line):
            following = commands[position + 1] if position + 1 < len(commands) else ""
            if after:
                points.append((total, count - 1, True))
            else:
                points.append((total, count if MANEUVER.match(following) else None, False))
    return points


def _unit(a: list[float]) -> list[float]:
    norm = sum(x * x for x in a) ** 0.5
    return [x / norm for x in a]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def to_inertial(
    components: list[float], frame: tuple[str, str], r: list[float], v: list[float]
) -> list[float]:
    """Script components in (CoordinateSystem, Axes) on the state r, v, as GMAT reads them."""
    system, axes = (name.lower() for name in frame)
    if system != "local":
        return list(components)
    if axes == "vnb":
        first = _unit(v)
        second = _unit(_cross(r, v))
        basis = [first, second, _cross(first, second)]
    else:  # LVLH
        first = _unit(r)
        third = _unit(_cross(r, v))
        basis = [first, _cross(third, first), third]
    return [sum(components[k] * basis[k][axis] for k in range(3)) for axis in range(3)]


def pre_burn_velocity(
    components: list[float], frame: tuple[str, str], r: list[float], after: list[float]
) -> list[float]:
    """The velocity a burn of `components` turns into `after`, its frame on the pre-burn state."""
    before = list(after)
    for _ in range(60):
        impulse = to_inertial(components, frame, r, before)
        before = [after[k] - impulse[k] for k in range(3)]
    return before


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--startup_file", "-s")
    parser.add_argument("--logfile", "-l")
    parser.add_argument("--verbose")
    parser.add_argument("--run", "-r")
    parser.add_argument("--version", "-v", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def output_path(startup: str | None) -> Path:
    if startup:
        for line in Path(startup).read_text(encoding="utf-8").splitlines():
            if line.strip().upper().startswith("OUTPUT_PATH"):
                return Path(line.split("=", 1)[1].strip())
    return Path.cwd()


TIDE_FILE = re.compile(r"^\s*\w+\.GravityField\.\w+\.TideFile\s*=\s*'([^']*)'\s*;", re.MULTILINE)


def missing_tide_file(text: str, script_dir: Path) -> str | None:
    """A TideFile GMAT would not find: absolute as written, else beside the script.

    GMAT also tries <BODY>_POT_PATH, which the double does not model.
    """
    for name in TIDE_FILE.findall(text.split("BeginMissionSequence", 1)[0]):
        # GravityField adds .tide to a name with no "." in it.
        path = Path(name.replace("\\", "/") + ("" if "." in name else ".tide"))
        if name and not (path if path.is_absolute() else script_dir / path).is_file():
            return name
    return None


def main() -> int:
    args = parse_args()
    if args.version:
        print("Fake GmatConsole R2026a-test-double")
        return 0
    if not args.run:
        print("missing --run", file=sys.stderr)
        return 2
    text = gmat_text(Path(args.run).read_text(encoding="utf-8-sig"))
    missing = missing_tide_file(text, Path(args.run).resolve().parent)
    if missing is not None:
        # R2026a's words, raised while the script is read, probe or not.
        print(f'**** ERROR **** ODEModel Exception Thrown: The file name "{missing}" does not '
              "exist in line:", file=sys.stderr)
        return 1
    if PROBE_REPORT.search(text):
        return answer_state_probe(text, output_path(args.startup_file))
    if "Create ReportFile OrbitWinAlignmentProbe;" in text:
        return answer_alignment_probe(text, output_path(args.startup_file))
    if "% FAKE_GMAT_FAIL" in text:
        print("ERROR!!!!!! ---- forced fake failure", file=sys.stderr)
        return 3
    if joint_apart(text):
        print("Command Exception: Spacecraft epochs do not match in a Propagator used in the "
              "Propagate command")
        return 1
    # A miss of a known size: the target sits this far off the pursuer's track.
    miss = re.search(r"^% FAKE_GMAT_MISS_KM ([0-9.]+)", text, re.MULTILINE)
    miss_km = float(miss.group(1)) if miss else 0.0
    # The target's epoch falling this far behind per row, as unsynchronized
    # propagators do.
    drift = re.search(r"^% FAKE_GMAT_EPOCH_DRIFT_S ([0-9.]+)", text, re.MULTILINE)
    drift_s = float(drift.group(1)) if drift else 0.0
    # GMAT writes its steps and a last row where Propagate stops; a run that
    # aborts at this epoch (a drag model below 100 km) keeps what it wrote.
    abort = re.search(r"^% FAKE_GMAT_ABORT_AT_S ([0-9.]+)", text, re.MULTILINE)
    end_s = mission_end_s(text)
    aborted = abort is not None and end_s > float(abort.group(1))
    if aborted:
        end_s = float(abort.group(1))
    # By default the pursuer reaches the target at 10 s and stays; a flyby
    # passes through it at T s, written every STEP s.
    flyby = re.search(r"^% FAKE_GMAT_FLYBY ([0-9.]+) ([0-9.]+)", text, re.MULTILINE)
    if flyby:
        closest_s, step_s = float(flyby.group(1)), float(flyby.group(2))
        grid = [index * step_s for index in range(int(end_s / step_s) + 1)]
    else:
        closest_s, grid = 10.0, [0.0, 10.0, 20.0]
    times = sorted({*(t for t in grid if t < end_s), end_s})
    # GMAT writes one row where each Propagate ends, before the Maneuver, and
    # none after it. The track below is the one flown after every burn, so that
    # row carries the track's velocity less the impulse, found on its own
    # pre-burn state; a row just before it carries the track's own.
    fields_of: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        match = ASSIGN.match(line)
        if match:
            # As R2026a reads a burn: V, N and B are Element1-3, the later assignment wins.
            field = BURN_ALIASES.get(match.group(2), match.group(2))
            fields_of.setdefault(match.group(1), {})[field] = match.group(3)
    burns = []
    flown = []
    every = []
    for epoch, name in maneuvers(text):
        spec = fields_of.get(name, {})
        components = [float(spec.get(f"Element{k}", "0")) for k in (1, 2, 3)]
        frame = (spec.get("CoordinateSystem", "Local"), spec.get("Axes", "VNB"))
        if 0.0 < epoch < end_s:
            burns.append((epoch, components, frame))
        every.append((components, frame))
        if epoch < end_s or not aborted:
            flown.append((epoch, components, frame))
    # A lone burn at t = 0 is flown too, and needs r x v as much as any other.
    drift = DRIFT_KM_S if burns or flown else 0.0
    times = sorted({*times, *(t for t, _, _ in burns), *(t - BEFORE_BURN_S for t, _, _ in burns)})
    reports = CREATE_REPORT.findall(text)
    fields: dict[str, dict[str, str]] = {name: {} for name in reports}
    for line in text.splitlines():
        match = ASSIGN.match(line)
        if match and match.group(1) in fields:
            fields[match.group(1)][match.group(2)] = match.group(3)
    out = output_path(args.startup_file)
    out.mkdir(parents=True, exist_ok=True)
    # Epochs as the probe reads them, for the A1 columns; an alignment writes
    # FROZEN_ROWS rows first, the late craft waiting at its epoch with ElapsedSecs 0.
    craft_fields: dict[str, dict[str, str]] = {}
    for line in text.split("BeginMissionSequence", 1)[0].splitlines():
        match = ASSIGN.match(line)
        if match:
            craft_fields.setdefault(match.group(1), {})[match.group(2)] = match.group(3).strip("'\"")
    aligned = alignment_of(text)
    for name in reports:
        if fields[name].get("WriteReport", "true").strip("'\" ").lower() in ("false", "off"):
            continue  # R2026a writes no file for it, not even a header
        filename = fields[name].get("Filename", f"'{name}.txt'").strip("'\"")
        add = fields[name].get("Add", "{}")
        headers = [item.strip() for item in add.strip("{}").split(",") if item.strip()]
        path = out / filename
        if name == "OrbitWinDiagnostic":
            # Build deterministic rows from the actual requested header names.
            # The pursuer comes first in the sibling's columns.
            pursuer_name = headers[0].split(".")[0]
            rows = []
            schedule = [(time_s, False) for time_s in times]
            if aligned is not None and aligned[2] > 0.0:
                schedule = [
                    (-aligned[2] * (1.0 - k / FROZEN_ROWS), True) for k in range(FROZEN_ROWS)
                ] + schedule
            for index, (time_s, frozen) in enumerate(schedule):
                # Through the alignment's rows the early craft moves and the late one
                # waits where t = 0 puts it.
                pursuer_early = aligned is not None and aligned[0] == pursuer_name
                track_s = time_s if not frozen or pursuer_early else 0.0
                target_s = time_s if not frozen or not pursuer_early else 0.0
                pursuer_x = 7000.0 + closest_s - (track_s if flyby else min(track_s, closest_s))
                # Once it stops, so does its velocity: Hermite over a long gap
                # otherwise swings the pursuer through the target.
                pursuer_vx = -1.0 if flyby or track_s <= closest_s else 0.0
                pursuer_v = [pursuer_vx, drift, 0.0]
                for epoch, components, frame in burns:
                    if not frozen and time_s == epoch:
                        r = [pursuer_x, drift * time_s, 0.0]
                        pursuer_v = pre_burn_velocity(components, frame, r, pursuer_v)
                values = {
                    ("p", "X"): pursuer_x, ("p", "Y"): drift * track_s, ("p", "Z"): 0.0,
                    ("p", "VX"): pursuer_v[0], ("p", "VY"): pursuer_v[1], ("p", "VZ"): pursuer_v[2],
                    ("t", "X"): 7000.0, ("t", "Y"): miss_km + drift * target_s, ("t", "Z"): 0.0,
                    ("t", "VX"): 0.0, ("t", "VY"): drift, ("t", "VZ"): 0.0,
                }
                row = []
                for header in headers:
                    suffix = header.split(".")[-1]
                    owner = header.split(".")[0]
                    pursuer = owner == pursuer_name
                    # Each craft's ElapsedSecs counts from its own epoch.
                    own_s = time_s
                    if aligned is not None and aligned[2] > 0.0:
                        own_s = time_s + aligned[2] if owner == aligned[0] else max(time_s, 0.0)
                    if not pursuer:
                        own_s -= drift_s * index
                    if suffix == "ElapsedSecs":
                        row.append(own_s)
                    elif suffix == "Altitude":
                        # Geodetic on the equator, where the fake flies, is r - R.
                        r = (values[("p", "X")] ** 2 + values[("p", "Y")] ** 2) ** 0.5
                        row.append(r - 6378.1363)
                    elif suffix == "A1ModJulian":
                        row.append(_epochs(craft_fields.get(owner, {}))["A1"] + own_s / 86400.0)
                    elif ".EarthMJ2000Eq." in header:
                        row.append(values.get(("p" if pursuer else "t", suffix), 0.0))
                    else:
                        row.append(0.0)
                rows.append(row)
        elif name == BURN_REPORT:
            # The Report commands around each Maneuver: the pursuer before and after.
            # The columns the Report command asks for, in its order.
            request = re.search(rf"^\s*Report\s+{BURN_REPORT}\s+([^;]+);", text, re.MULTILINE)
            assert request is not None
            asked = request.group(1).split()
            owner = asked[0].split(".")[0]
            rows = []
            # Each Report command where the mission runs it, until an abort.
            for epoch, index, after in report_points(text):
                if aborted and epoch >= end_s:
                    break
                pursuer_x = 7000.0 + closest_s - (epoch if flyby else min(epoch, closest_s))
                track_v = [-1.0 if flyby or epoch <= closest_s else 0.0, drift, 0.0]
                r = [pursuer_x, drift * epoch, 0.0]
                own_s = epoch
                if aligned is not None and aligned[2] > 0.0 and owner == aligned[0]:
                    own_s = epoch + aligned[2]
                a1 = _epochs(craft_fields.get(owner, {}))["A1"] + own_s / 86400.0
                v = track_v
                if index is not None and not after:
                    v = pre_burn_velocity(*every[index], r, track_v)
                row = []
                for column in asked:
                    craft, _, quantity = column.partition(".")
                    state = {"A1ModJulian": a1, **dict(zip(
                        (f"EarthMJ2000Eq.{axis}" for axis in ("X", "Y", "Z", "VX", "VY", "VZ")),
                        (*r, *v), strict=True,
                    ))}
                    if craft != owner or quantity not in state:
                        print(f"ERROR!!!!!! ---- fake GmatConsole cannot report {column}",
                              file=sys.stderr)
                        return 1
                    row.append(state[quantity])
                rows.append(row)
        else:
            # GMAT writes every ReportFile at the same steps, alignment rows included.
            frozen_count = FROZEN_ROWS if aligned is not None and aligned[2] > 0.0 else 0
            rows = [[float(k) for _ in headers] for k in range(frozen_count + len(times))]
        # The sibling asks for Precision 17, so its A1 epochs come back bit-exact.
        digits = 17 if name in ("OrbitWinDiagnostic", BURN_REPORT) else 16
        headed = fields[name].get("WriteHeaders", "true").strip("'\"").lower() not in ("false", "off")
        path.write_text(
            (" ".join(headers) + "\n" if headed else "")
            + "\n".join(" ".join(f"{value:.{digits}g}" for value in row) for row in rows)
            + "\n",
            encoding="utf-8",
        )
    if args.logfile:
        Path(args.logfile).parent.mkdir(parents=True, exist_ok=True)
        # R2026a's record shape, written by a run that still exits 0.
        logged = "".join(
            f"{index}: {args.run}: **** ERROR **** {record}\n   \" 1: the line\"\n\n"
            for index, record in enumerate(
                re.findall(r"^% FAKE_GMAT_LOG_ERROR (.+)$", text, re.MULTILINE), 1)
        )
        Path(args.logfile).write_text(logged + "Fake GMAT run successful\n", encoding="utf-8")
    if aborted:
        print("Console Application Execution Failed: Moderator::RunMission failed")
        return 1
    print("*** GMAT Integration test (Console version) successful! ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
