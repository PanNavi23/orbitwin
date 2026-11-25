#!/usr/bin/env python3
"""Check the rows of the report that is actually going to be uploaded.

The organizer's validator reads the completion time off the last row, not off
the first row inside the sphere. Measured 2026-09-05: a submitted run whose
report had Δr = 2.394 km at 5982.99 s but ran on to 4TA was scored with
Tteam = 23815.43 s, and the 22.2266-point time term went to zero. Every other
gate in this repository had passed.

So the last thing to look at before uploading is the file itself:

    python scripts/check_report_rows.py <certify output>/submission/ODC_2026_Team54.txt \
        --config <the day's config> --certificate <certify output>/submission/certification.json

The report's name is the template's with its TeamXX placeholder numbered from
policy.team_number (ODC_2026_Team_XX.txt gives ODC_2026_Team_54.txt); certify
prints it.
The two files give the script, the time column, the frozen rows, T_max, the
coefficients and the total delta-v, and the report and script must be the
bytes the certificate lists. When certify ran the console copy of a script
GmatConsole refused for its plots (gmat.console_copy), the copy is rebuilt
from the script and proven again, and must be the one every exact run ran;
the last row must then match T_team or the check exits 2. By hand, --horizon-s is the day config's
policy.horizon_s, not certification.json's horizon_s, which is on the solver's
clock. A hit passes when its last row is
inside the sphere; under the ranking rules a miss passes when its last row is
at T_max.

It reads GMAT's report, computes the relative distance on every row from the two
spacecraft's positions, and reports what a validator would see -- the last row,
the first row inside the sphere, and whether they are the same row. Optionally
scores it under announced coefficients.

Nothing here re-derives the trajectory, because the point is to check the
artifact rather than the intention behind it.

The report is split the way its ReportFile wrote it: --script reads Delimiter,
WriteHeaders, FixedWidth and the Add list from the script, or pass them with
--delimiter, --write-headers, --fixed-width and --columns. With neither, the
report is read in GMAT's default format.

By hand, without --certificate (the GUI path of docs/RUNBOOK.md step 7),
--script also binds the report to the script, and a mismatch exits 2: the
last row must end the script's Propagate goals, each burn must show in the
rows as its impulse and no other jump may, and the file must have the name the
script's official ReportFile writes. Σ|Δv| is then the script's; a
--total-dv that differs is refused.

A report with no position of the target (the organizer's workshop style) shows
no distance. With --certificate it passes on certify's diagnostic flight, with a
WARNING saying so; by hand it exits 7, which only the --certificate form clears.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, assert_never

import numpy as np

from orbitwin.argtypes import positive_finite as _positive_finite
from orbitwin.errors import OrbitWinError
from orbitwin.models import ReportFileSpec
from orbitwin.pinned import refuse_unpinned
from orbitwin.report_format import (
    DEFAULT_REPORT_FORMAT,
    ReportFormat,
    is_epoch_parameter,
    is_text_parameter,
    position_frames,
    read_rows,
    report_format,
)
from orbitwin.rules import (
    DistanceTerm,
    InterceptionRules,
    interception_rules,
    rules_name,
)

if TYPE_CHECKING:
    from orbitwin.config import OrbitWinConfig
    from orbitwin.gmat_parser import MissionSchedule, ScheduledManeuver


def read_report(
    path: str, fmt: ReportFormat = DEFAULT_REPORT_FORMAT, names: list[str] | None = None
) -> tuple[list[str], np.ndarray]:
    """Return the header names and the numeric rows of a GMAT ReportFile.

    Without headers the names are `names`, the ReportFile's Add list.
    """
    try:
        header, data = read_rows(path, fmt, names)
    except (OrbitWinError, OSError) as exc:
        raise SystemExit(f"{path}: {exc}" if isinstance(exc, OSError) else str(exc)) from None
    return list(header), data


def position_columns(
    header: list[str], count: int, spacecraft: str, other: str | None,
    other_frame: str | None = None,
) -> list[int]:
    """Indices of X, Y, Z for one spacecraft, by header name where possible.

    Any frame both craft are in will do: a distance is the same in every
    frame at one epoch (R2026a: EarthFixed and EarthMJ2000Eq agree to 2e-11
    km). EarthMJ2000Eq first, then by name. With `other` None, the other
    craft's columns came by flag: this craft must be in their frame, which is
    EarthMJ2000Eq when their names do not say.
    """
    missing = spacecraft
    if header and len(header) == count:
        own = position_frames(header, spacecraft)
        if other is None:
            shared = {other_frame or "EarthMJ2000Eq"} & set(own)
        else:
            theirs = position_frames(header, other)
            shared = set(own) & set(theirs)
            # Name the craft whose columns are absent, not the one asked about.
            missing = other if own and not theirs else spacecraft
        frames = sorted(shared, key=_frame_order)
        if frames:
            return list(own[frames[0]])
    raise SystemExit(
        f"cannot locate {missing} X/Y/Z columns by name; header has {len(header)} "
        f"names for {count} columns. Pass --pursuer-cols / --target-cols."
    )


def flagged_frame(header: list[str], count: int, indices: list[int]) -> str | None:
    """The frame the named columns at `indices` share, or None."""
    if len(header) != count or not all(0 <= i < count for i in indices):
        return None
    frames = set()
    crafts = set()
    for index, axis in zip(indices, "XYZ", strict=True):
        parts = header[index].split(".")
        if parts[-1] != axis or len(parts) not in (2, 3):
            return None
        frames.add(parts[1] if len(parts) == 3 else "EarthMJ2000Eq")
        crafts.add(parts[0])
    return frames.pop() if len(frames) == 1 and len(crafts) == 1 else None


def _frame_order(frame: str) -> tuple[bool, str]:
    return frame != "EarthMJ2000Eq", frame


def report_spec(script: str, report: str, name: str | None) -> ReportFileSpec:
    """The ReportFile in `script` that wrote `report`."""
    # Here, not at the top: a check without --script never imports the parser.
    try:
        from orbitwin.gmat_parser import report_file_specs
    except ImportError as exc:
        raise SystemExit(f"--script needs the project's venv to parse the script: {exc}") from None

    try:
        specs = report_file_specs(script)
    except (OrbitWinError, OSError) as exc:
        raise SystemExit(f"{script}: {exc}") from None
    if name is not None:
        chosen = [spec for spec in specs if spec.name == name]
    elif len(specs) == 1:
        chosen = list(specs)
    else:
        chosen = [spec for spec in specs if Path(spec.filename).name == Path(report).name]
    if len(chosen) != 1:
        raise SystemExit(
            f"{script}: cannot tell which of the ReportFiles {[spec.name for spec in specs]} "
            f"wrote {report}; pass --report-file"
        )
    if not chosen[0].write_report:
        raise SystemExit(f"{script}: {chosen[0].name}.WriteReport is false; GMAT writes no file")
    return chosen[0]


#: How far --total-dv may be from the script's Σ|Δv|. A unit slip is a factor of 1000.
TOTAL_DV_TOLERANCE_KM_S = 1.0e-6
#: The row after a burn, flown back to the burn under point mass and J2, against
#: the row before: R2026a's other forces left 3.1e-7 km/s and 6.6e-7 km over
#: its 1 s step on the certified R1, R6 and R8 reports. Allowed: a floor and an
#: unmodelled acceleration over the step, plus the digits Precision prints.
BURN_JUMP_FLOOR_KM_S = 1.0e-5
BURN_GAP_FLOOR_KM = 1.0e-4
UNMODELLED_ACCELERATION_KM_S2 = 1.0e-5
#: Intervals within this many of a burn give the pull of the forces left out there.
NEIGHBOURS = 3
#: Times their largest pull; at twice it R3-R5's drag near 100 km took 0.37 of the allowance.
PULL_MARGIN = 3.0
#: Longer steps are not scanned: point mass and J2 drift too far over them.
SCANNED_STEP_S = 600.0


def _rule_set_or_none(name: object) -> InterceptionRules | None:
    """The interception rule set a record names, or None for a name no rule set has."""
    try:
        return interception_rules(name) if isinstance(name, str) else None
    except ValueError:
        return None


def _refuse(message: str) -> NoReturn:
    """Exit 2: the report and the script given with it disagree."""
    print(f"check_report_rows.py: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _file_name(filename: str) -> str:
    """A ReportFile Filename's last component, by either separator, as GMAT on Windows."""
    return re.split(r"[\\/]", filename.strip().strip("'\""))[-1]


def _hold_to_official(script: str, report: str, spec: ReportFileSpec) -> None:
    """`spec` must be the report the organizer scores, and `report` named as it writes it."""
    try:
        from orbitwin.gmat_parser import report_file_specs
        from orbitwin.script_renderer import official_report_of
    except ImportError as exc:
        raise SystemExit(f"--script needs the project's venv to parse the script: {exc}") from None
    official = official_report_of(report_file_specs(script))
    if official is None or official.name != spec.name:
        _refuse(f"{spec.name} is not {script}'s official report"
                f"{'' if official is None else ', ' + official.name}, the one uploaded")
    expected = _file_name(official.filename)
    if Path(report).name != expected:
        _refuse(f"{report} is named {Path(report).name!r}, and {script}'s official report "
                f"{official.name} writes {expected!r}: not this script's report, or renamed")


def _pursuer_state_columns(header: list[str], craft: str) -> tuple[list[int], list[int]] | None:
    """EarthMJ2000Eq X, Y, Z and VX, VY, VZ of `craft`, or None without all six."""
    def find(axis: str) -> int | None:
        for name in (f"{craft}.EarthMJ2000Eq.{axis}", f"{craft}.{axis}"):
            if name in header:
                return header.index(name)
        return None

    found = [find(axis) for axis in ("X", "Y", "Z", "VX", "VY", "VZ")]
    if any(index is None for index in found):
        return None
    indices = [int(index) for index in found if index is not None]
    return indices[:3], indices[3:]


def _fly_back(
    r_km: np.ndarray, v_km_s: np.ndarray, dt_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each row's (r, v) flown back its dt_s under point mass and J2 about J2000's z (RK4)."""
    from orbitwin.pursuer_model import EGM96_J2, EGM96_RADIUS_KM

    mu = 398600.4418

    def acceleration(r: np.ndarray) -> np.ndarray:
        r2 = np.sum(r * r, axis=1, keepdims=True)
        zz = 5.0 * r[:, 2:3] ** 2 / r2
        k = 1.5 * EGM96_J2 * mu * EGM96_RADIUS_KM**2 / r2**2.5
        return np.asarray(-mu * r / r2**1.5 + k * r * np.hstack([zz - 1.0, zz - 1.0, zz - 3.0]))

    steps = max(2, math.ceil(float(np.max(dt_s, initial=0.0)) / 0.5))
    h = -np.asarray(dt_s, dtype=float)[:, None] / steps
    r, v = r_km.astype(float), v_km_s.astype(float)
    for _ in range(steps):
        a1 = acceleration(r)
        a2 = acceleration(r + 0.5 * h * v)
        a3 = acceleration(r + 0.5 * h * (v + 0.5 * h * a1))
        a4 = acceleration(r + h * (v + 0.5 * h * a2))
        r, v = (r + h * v + h * h / 6.0 * (a1 + a2 + a3),
                v + h / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4))
    return r, v


def _neighbour_rate(rate: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The largest finite rate within NEIGHBOURS intervals of each of `rows`, itself left out."""
    best = np.zeros(len(rows))
    for shift in [*range(-NEIGHBOURS, 0), *range(1, NEIGHBOURS + 1)]:
        at = rows + shift
        inside = (at >= 0) & (at < len(rate))
        found = np.where(inside, rate[np.clip(at, 0, len(rate) - 1)], np.nan)
        best = np.fmax(best, np.nan_to_num(found, nan=0.0))
    return best


def _digit(values: np.ndarray, precision: int | None) -> float:
    """The last digit Precision prints of the largest of `values`; 0 at full precision."""
    largest = float(np.max(np.abs(values))) if values.size else 0.0
    if precision is None or largest <= 0.0:
        return 0.0
    return 10.0 ** (math.floor(math.log10(largest)) - precision + 1)


def bind_to_script(
    args: argparse.Namespace, schedule: MissionSchedule, header: list[str], full: np.ndarray,
    time_index: int, waiting: tuple[str, int] | None, precision: int | None,
) -> None:
    """Refuse a report its script did not write: its end, its burns and its alignment.

    The last row must be the script's Propagate goals summed from the
    alignment's end, and each burn must show in the rows as its impulse.
    R2026a writes no row after a Maneuver (burn_rows.py): the row at the burn
    epoch is the pre-burn state and the next is one step on, so that row is
    flown back to the epoch and the velocity difference is the impulse.
    --binding warn turns a refusal over the burns into a WARNING line.
    """
    from orbitwin.burn_rows import BURN_EPOCH_TOLERANCE_S
    from orbitwin.errors import DynamicsError
    from orbitwin.frames import script_components_to_inertial
    from orbitwin.models import StateVector, as_vector3

    script = args.script
    warned: list[str] = []

    def burns_differ(message: str) -> None:
        if args.binding != "warn":
            _refuse(message)
        warned.append(message)
        UNBOUND.append(message)
        print(f"WARNING: {message}")

    if len(header) != full.shape[1]:
        _refuse(f"{args.report}: its columns are not named, so it cannot be held to {script}")
    strangers = sorted({m.spacecraft for m in schedule.maneuvers} - {args.pursuer})
    if strangers:
        _refuse(f"{script} maneuvers {strangers}, not the pursuer {args.pursuer}")
    # The aligning craft's clock runs through the alignment; the waiting one's reads 0 there.
    own = f"{schedule.alignment_craft}.ElapsedSecs"
    clocks = [*([header.index(own)] if own in header else []), time_index,
              *(i for i, name in enumerate(header) if name.endswith(".ElapsedSecs")),
              *(i for i, name in enumerate(header) if is_epoch_parameter(name))]
    clock = next((i for i in clocks if is_epoch_parameter(header[i])), None)
    if clock is None:
        _refuse(f"{args.report} has no ElapsedSecs or epoch column to hold to {script}'s "
                "Propagate goals")
    kind = header[clock].rsplit(".", 1)[-1]
    seconds = full[:, clock] * (1.0 if kind == "ElapsedSecs" else 86400.0)
    if schedule.alignment_craft is None and waiting is not None:
        _refuse(f"{args.report} holds {waiting[0]} still for {waiting[1]} rows, as an epoch "
                f"alignment does, and {script} has none")
    if schedule.alignment_craft is not None and waiting is not None and (
            waiting[0] == schedule.alignment_craft):
        _refuse(f"{args.report} holds {waiting[0]} still, and {script}'s alignment flies it")
    if schedule.alignment_craft is not None and waiting is None:
        _refuse(f"{script} starts with an epoch alignment flying {schedule.alignment_craft}, "
                f"and {args.report} holds no craft still as its rows do: they were cut, or it "
                "is another script's report")
    # The alignment's end: the row where the waiting craft starts, else the first.
    start = waiting[1] if schedule.alignment_craft is not None and waiting is not None else 0
    q_t = _digit(seconds, precision)
    tolerance_s = max(BURN_EPOCH_TOLERANCE_S, 2.0 * q_t)
    if schedule.alignment_craft is not None and own in header:
        # The alignment flies its craft from its own epoch, ElapsedSecs 0, on every row.
        flown = full[:, header.index(own)]
        q_own = max(BURN_EPOCH_TOLERANCE_S, 2.0 * _digit(flown, precision))
        if not abs(float(flown[0])) <= q_own:
            _refuse(f"{args.report} starts at {float(flown[0]):.6f} on {own}, and {script}'s "
                    f"epoch alignment flies {schedule.alignment_craft} from 0: its first rows "
                    "were cut")
        if schedule.max_step_s is not None and start > 0:
            steps = np.diff(flown[:start + 1])
            cut = np.flatnonzero(steps > schedule.max_step_s + q_own)
            if cut.size:
                j = int(cut[0])
                # Not a burn check: --binding warn does not touch it. A cut alignment is
                # not this script's flight.
                _refuse(f"{args.report}: rows missing in the epoch alignment between "
                        f"t = {float(flown[j]):.6f} and t = {float(flown[j + 1]):.6f} on {own}, "
                        f"{float(steps[j]):g} s apart where {script}'s propagators step at most "
                        f"{schedule.max_step_s:g} s: rows were cut")
    origin_s = float(seconds[start])
    # Head-start burns sit in the alignment's rows, on its craft's clock from row 0.
    early = [m for m in schedule.maneuvers if m.head_start]
    on_own_clock = (schedule.alignment_craft is not None
                    and header[clock].startswith(f"{schedule.alignment_craft}."))
    if early and not on_own_clock:
        burns_differ(f"{args.report}: {header[clock]} is not {schedule.alignment_craft}'s "
                     f"clock, so {script}'s {len(early)} head-start burn(s) cannot be placed "
                     "in the alignment's rows")
    if on_own_clock:
        judge_boundary(args, schedule, float(seconds[0]), origin_s)
    expected_s = origin_s + schedule.end_s
    shown = f"{header[clock]} ({'s' if kind == 'ElapsedSecs' else 'in seconds'})"
    if not abs(float(seconds[-1]) - expected_s) <= tolerance_s:
        _refuse(f"{args.report} ends at {float(seconds[-1]):.6f} on {shown}, and {script}'s "
                f"Propagate goals end at {expected_s:.6f} ({origin_s:.6f} + {schedule.end_s:.6f}"
                f" s): not this script's report")
    # A burn right after the alignment: with MaxStep 60 s R2026a ended the alignment on
    # two rows, the pre-burn state and then the post-burn one, and the second is `start`.
    first = max(start - 1, 0)
    spans: list[tuple[ScheduledManeuver, int, int, str]] = []
    for number, maneuver in enumerate(schedule.maneuvers, 1):
        if maneuver.head_start and not on_own_clock:
            continue
        epoch_s = (float(seconds[0]) if maneuver.head_start else origin_s) + maneuver.time_s
        label = f"burn {number} ({maneuver.burn}, t = {epoch_s:.6f} on {header[clock]})"
        lo, hi = (0, start) if maneuver.head_start else (first, len(full))
        at = lo + np.flatnonzero(np.abs(seconds[lo:hi] - epoch_s) <= tolerance_s)
        if at.size == 0:
            burns_differ(f"{args.report} has no row at {label}: not this script's report")
            continue
        if int(at[-1]) + 1 >= len(full):
            burns_differ(f"{args.report} ends at {label}: not this script's report")
            continue
        spans.append((maneuver, int(at[0]), int(at[-1]) + 1, label))
    # GMAT steps at most MaxStep (a fixed step at InitialStepSize): a longer step is rows
    # the report does not hold, and a burn there would not show.
    # A pursuer that flies the alignment on its own clock is scanned through it too: a
    # head-start burn the script lacks shows there.
    scan = 0 if on_own_clock and schedule.alignment_craft == args.pursuer else start
    rows = np.arange(scan, len(full) - 1)
    dt_s = seconds[rows + 1] - seconds[rows]
    if np.any(dt_s < -tolerance_s):
        _refuse(f"{args.report}: {header[clock]} goes back after the alignment")
    if schedule.max_step_s is not None:
        longest_s = schedule.max_step_s + max(BURN_EPOCH_TOLERANCE_S, 2.0 * q_t)
        cut = np.flatnonzero(dt_s > longest_s)
        if cut.size:
            j = int(cut[0])
            burns_differ(f"{args.report}: rows missing between t = {seconds[rows[j]]:.6f} and "
                         f"t = {seconds[rows[j] + 1]:.6f} on {header[clock]}, "
                         f"{float(dt_s[j]):g} s apart where {script}'s propagators step at most "
                         f"{schedule.max_step_s:g} s: not this script's report, or rows were cut")
    state = _pursuer_state_columns(header, args.pursuer)
    if state is None:
        what = (f"{args.report} has no EarthMJ2000Eq position and velocity of {args.pursuer}, "
                f"so its burns are not checked against {script}: only its end and a row at "
                "each burn epoch bind it, and 6b.7's by-eye comparison is the only binding of "
                "the burns")
        # Never implied by --binding warn: a report that cannot show a burn is said so.
        if not args.accept_end_time_binding_only:
            _refuse(f"{what}. Pass --accept-end-time-binding-only to go on so (docs/RUNBOOK.md "
                    "step 6b.6)")
        UNBOUND.append(what)
        print(f"WARNING: {what}")
        return
    r_cols, v_cols = state
    r_km, v_km_s = full[:, r_cols], full[:, v_cols]
    q_r, q_v = _digit(r_km, precision), _digit(v_km_s, precision)
    # Interval j runs from row j to row j + 1; each is flown back and compared.
    scanned = dt_s <= SCANNED_STEP_S
    back_r = np.full((len(rows), 3), np.nan)
    back_v = np.full((len(rows), 3), np.nan)
    if np.any(scanned):
        back_r[scanned], back_v[scanned] = _fly_back(
            r_km[rows[scanned] + 1], v_km_s[rows[scanned] + 1], dt_s[scanned])
    miss_v = np.linalg.norm(back_v - v_km_s[rows], axis=1)
    miss_r = np.linalg.norm(back_r - r_km[rows], axis=1)
    # What the printed digits leave: of each value, and of each row's time, by |v| and |a|.
    speed = np.linalg.norm(v_km_s, axis=1)
    pull_g = 398600.4418 / np.sum(r_km * r_km, axis=1)
    noise_v = 4.0 * q_v + 2.0 * q_t * pull_g[rows]
    noise_r = 4.0 * (q_r + q_v * dt_s) + 2.0 * q_t * speed[rows]
    # What the forces left out pull per second here; drag near 100 km is 4e-5 km/s^2 (R3).
    rate = np.where(scanned & (dt_s >= 1.0e-3),
                    np.maximum(miss_v - noise_v, 0.0) / np.maximum(dt_s, 1.0e-3), np.nan)
    burned = np.zeros(len(rows), dtype=bool)
    for _, pre, post, _ in spans:
        burned[max(pre - scan, 0):post - scan] = True
    rate[burned] = np.nan
    lengths = 0
    for maneuver, pre, post, label in spans:
        step_s = float(seconds[post] - seconds[pre])
        if not step_s <= SCANNED_STEP_S:
            burns_differ(f"{args.report}: the row after {label} is {step_s:g} s on, over the "
                         f"{SCANNED_STEP_S:g} s point mass and J2 are trusted for")
            continue
        r_back, v_back = (x[0] for x in _fly_back(r_km[post:post + 1], v_km_s[post:post + 1],
                                                  np.array([step_s])))
        jump = v_back - v_km_s[pre]
        claimed = np.asarray(maneuver.components_km_s, dtype=float)
        try:
            inertial = script_components_to_inertial(
                claimed, StateVector(as_vector3(r_km[pre]), as_vector3(v_km_s[pre])),
                *maneuver.frame)
            off_km_s = float(np.linalg.norm(jump - inertial))
            how = "as a vector"
        except DynamicsError:
            lengths += 1
            off_km_s = abs(float(np.linalg.norm(jump)) - float(np.linalg.norm(claimed)))
            how = f"in length ({'/'.join(maneuver.frame)} is not built here)"
        near = np.array([pre - scan - 1, post - scan])
        pull = UNMODELLED_ACCELERATION_KM_S2 + PULL_MARGIN * float(
            np.max(_neighbour_rate(rate, near)))
        allowed_km_s = (BURN_JUMP_FLOOR_KM_S + 4.0 * q_v + 2.0 * q_t * float(pull_g[pre])
                        + pull * step_s)
        allowed_km = (BURN_GAP_FLOOR_KM + 4.0 * (q_r + q_v * step_s)
                      + 2.0 * q_t * float(speed[pre]) + 0.5 * pull * step_s**2)
        gap_km = float(np.linalg.norm(r_back - r_km[pre]))
        if not gap_km <= allowed_km:
            burns_differ(f"{args.report}: the row after {label} is {gap_km:.3g} km from where "
                         f"the row at it leads (allowed {allowed_km:.3g} km): the rows do not "
                         "continue")
        elif not off_km_s <= allowed_km_s:
            burns_differ(f"{args.report}: the velocity jumps {float(np.linalg.norm(jump)):.9g} "
                         f"km/s at {label}, and {script} burns "
                         f"{float(np.linalg.norm(claimed)):.9g} km/s there; {off_km_s:.3g} km/s "
                         f"apart {how}, allowed {allowed_km_s:.3g}: not this script's report")
    # Anywhere else the rows must follow from each other: a burn the script lacks shows here.
    check = np.flatnonzero(scanned & ~burned)
    pulls = UNMODELLED_ACCELERATION_KM_S2 + PULL_MARGIN * _neighbour_rate(rate, check)
    allowed_v = BURN_JUMP_FLOOR_KM_S + noise_v[check] + pulls * dt_s[check]
    allowed_r = BURN_GAP_FLOOR_KM + noise_r[check] + 0.5 * pulls * dt_s[check]**2
    bad = check[(miss_v[check] > allowed_v) | (miss_r[check] > allowed_r)]
    if bad.size:
        j = int(bad[0])
        burns_differ(f"{args.report}: the rows jump {miss_v[j]:.9g} km/s and {miss_r[j]:.3g} km "
                     f"between data rows {rows[j] + 1} and {rows[j] + 2} (t = "
                     f"{seconds[rows[j]]:.6f} to {seconds[rows[j] + 1]:.6f} on "
                     f"{header[clock]}), where {script} has no burn: not this script's report")
    if warned:
        print(f"WARNING: --binding warn: {len(warned)} burn check(s) above failed and did not "
              "refuse; 6b.7's by-eye comparison is now mandatory")
        return
    count = len(schedule.maneuvers)
    only = f" ({lengths} in magnitude only, its frame not built here)" if lengths else ""
    print(f"script        : {script}: the rows end at its Propagate goals and hold its "
          f"{count} burn{'' if count == 1 else 's'}{only} and no other")


#: GMAT's Earth: EquatorialRadius (EGM96) and WGS-84's flattening, as Earth.Altitude
#: takes them (pursuer_model.WGS84_FLATTENING).
EARTH_EQUATORIAL_KM = 6378.1363
EARTH_FLATTENING = 1.0 / 298.257223563
#: Within this of 0 a height computed from the position decides nothing (exit 6): from
#: EarthMJ2000Eq or Ec it leaves out precession and nutation (54.6 m measured, 62.7 m at
#: Precision 6); EarthFixed was within 0.3 m of GMAT's. The printed digits add to both.
UNDECIDED_KM = {"EarthFixed": 0.001, "EarthMJ2000Eq": 0.1, "EarthICRF": 0.1,
                "EarthMJ2000Ec": 0.1}
#: J2000's obliquity, which GMAT's EarthMJ2000Ec is tilted by from EarthMJ2000Eq.
OBLIQUITY_J2000_RAD = math.radians(23.4392911)

#: The outcomes of §4 note 1(b) that are neither a pass nor a disqualification. Each is
#: its own exit status and the last line, after any other verdict that passed.
SURFACE_UNJUDGED_EXIT = 4
SURFACE_TEMPLATE_EXIT = 5
SURFACE_UNDECIDED_EXIT = 6
SURFACE: list[tuple[int, str]] = []


def _km(value: float) -> str:
    """A height in km that shows its sign and size however small: -1e-13 is not -0.000000."""
    return f"{value:.6f}" if abs(value) >= 1.0e-3 or value == 0.0 else f"{value:.3e}"


def geodetic_altitude_km(r_km: np.ndarray) -> np.ndarray:
    """Each row's height above the ellipsoid, from Earth-centred X, Y, Z about the pole.

    In EarthMJ2000Eq: rotation about the pole leaves the height alone, and the
    declination differs from the true-of-date latitude only by precession and
    nutation since J2000, about 0.15 deg in 2026: up to about 55 m of height
    (54.6 m against GMAT's), nothing at the equator or the poles. EarthFixed has
    the true pole. GMAT's Earth.Altitude, when the report has it, is used instead.
    The latitude is iterated to well under a millimetre.
    """
    e2 = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)
    p = np.hypot(r_km[:, 0], r_km[:, 1])
    z = r_km[:, 2]
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(6):
        n = EARTH_EQUATORIAL_KM / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * n * np.sin(lat), p)
    sin, cos = np.sin(lat), np.cos(lat)
    # Good at the poles too, where p / cos(lat) is not.
    height: np.ndarray = p * cos + z * sin - EARTH_EQUATORIAL_KM * np.sqrt(1.0 - e2 * sin**2)
    return height


def ecliptic_to_equatorial(r_km: np.ndarray) -> np.ndarray:
    """EarthMJ2000Ec X, Y, Z in EarthMJ2000Eq: a turn about X by the obliquity."""
    c, s = math.cos(OBLIQUITY_J2000_RAD), math.sin(OBLIQUITY_J2000_RAD)
    x, y, z = r_km[:, 0], r_km[:, 1], r_km[:, 2]
    return np.column_stack([x, c * y - s * z, s * y + c * z])


def judge_surface(args: argparse.Namespace, header: list[str], flown: np.ndarray,
                  position: list[int], until: str, precision: int | None) -> None:
    """§4 note 1(b) on every row up to the interception (a miss: the last row).

    The organizer judges GMAT's Earth.Altitude < 0: the report's column when it has
    one (also written as the short form, {pursuer}.Altitude), else the height from
    the pursuer's position. Below the surface on a row our burns flew disqualifies,
    exit 2. Only on the rows the template fixes, the frozen rows and the one at t = 0
    (row 1 without an alignment), it is the template's initial state: exit 5, ask the
    organizer. A height computed within
    UNDECIDED_KM of 0 decides nothing: exit 6. No altitude and no position it can
    use: exit 4. Neither 4, 5 nor 6 is a pass.
    """
    rules = interception_rules(args.rules)
    if rules.earth_crossing != "disqualify-before-interception":
        return
    clause = rules.earth_crossing_clause
    named = [f"{args.pursuer}.Earth.Altitude", f"{args.pursuer}.Altitude"]
    column = next((name for name in named if name in header), None)
    band_km = 0.0
    if column is not None:
        altitude = flown[:, header.index(column)]
        source = column
    else:
        frame = flagged_frame(header, len(header), position) if header and position else None
        if frame not in UNDECIDED_KM:
            why = (f"the report has no {named[0]} or {named[1]}, and "
                   + ("no named position of " + args.pursuer if frame is None
                      else f"{args.pursuer}'s position is in {frame}, whose z is not the pole"))
            print(f"surface       : not judged ({clause}): {why}")
            args.judged_surface = False
            SURFACE.append((SURFACE_UNJUDGED_EXIT, (
                f"SURFACE NOT JUDGED ({clause}): {why}. Check {args.pursuer}'s lowest altitude "
                "by eye in GMAT up to the interception (step 6b.7), or ask the organizer")))
            return
        r_km = flown[:, position]
        if frame == "EarthMJ2000Ec":
            r_km = ecliptic_to_equatorial(r_km)
        altitude = geodetic_altitude_km(r_km)
        source = f"{args.pursuer}'s {frame} position, WGS-84"
        band_km = UNDECIDED_KM[frame] + 2.0 * _digit(flown[:, position], precision)
    bad = np.flatnonzero(~np.isfinite(altitude))
    if bad.size:
        raise SystemExit(f"{args.report}: data row {int(bad[0]) + 1} gives {args.pursuer} an "
                         f"altitude that is not a finite number ({source})")
    args.judged_surface = True
    # The rows the template fixes before any burn of ours: the alignment's and the first at
    # t = 0, the state it leaves the pursuer in; without an alignment, row 1.
    fixed = min(args.frozen_rows + 1, len(altitude))
    low = int(np.argmin(altitude))
    print(f"surface       : {args.pursuer} at least {_km(float(altitude[low]))} km up to "
          f"{until} (data row {low + 1}; {source})")
    ours = altitude[fixed:]
    if ours.size:
        row = fixed + int(np.argmin(ours))
        if altitude[row] < -band_km:
            _disqualified(f"{args.pursuer} is {_km(float(altitude[row]))} km below the surface "
                          f"on data row {row + 1}, before the interception ({clause}; "
                          f"{source})")
    head = int(np.argmin(altitude[:fixed]))
    if altitude[head] < band_km:
        SURFACE.append((SURFACE_TEMPLATE_EXIT, (
            f"SURFACE: the template's initial state is below the surface ({clause}): "
            f"{args.pursuer} at {_km(float(altitude[head]))} km on data row {head + 1}, before "
            f"any burn of ours ({source}); ask the organizer")))
        return
    if ours.size and abs(float(altitude[row])) <= band_km:
        SURFACE.append((SURFACE_UNDECIDED_EXIT, (
            f"SURFACE UNDECIDED ({clause}): {args.pursuer} at {_km(float(altitude[row]))} km on "
            f"data row {row + 1}, within the {band_km * 1000.0:.3g} m that {source} may be off "
            f"by; GMAT's {named[0]} decides. Check it by eye (step 6b.7), or ask the "
            "organizer")))


def _disqualified(message: str) -> NoReturn:
    """Exit 2: by the rules the plan scores nothing (§4 note 1(a), (c))."""
    print(f"DISQUALIFIED: {message}")
    raise SystemExit(2)


def judge_rules(args: argparse.Namespace, schedule: MissionSchedule | None) -> None:
    """§4 note 1 on every plan that gives the burns: the script (by hand, or the one a
    certificate lists), the certificate's or verification's candidate, and the total.

    The rulebook's limits, not our safe margins (certify's): an impulse over the limit
    plus ε_v, or a total over the cap, exits 2 where the rules disqualify; a burn under
    min_separation_s after the one before costs penalty_points, as does an impulse over
    the limit under rules that deduct for it, once per maneuver.
    """
    rules = interception_rules(args.rules)
    if schedule is None and args.script is not None and args.certificate is not None:
        try:
            from orbitwin.gmat_parser import mission_schedule

            schedule = mission_schedule(args.script)
        except (ImportError, OrbitWinError, OSError) as exc:
            print(f"note: {args.script}'s burns not judged: {exc}", file=sys.stderr)
    #: Per plan: (time, |Δv|, side). Side 0 is a head-start burn on the early craft's clock
    #: while the lead is unknown; the pair across the alignment's end is then judge_boundary's.
    plans: list[tuple[str, list[tuple[float, float, int]]]] = []
    config = getattr(args, "day_config", None)
    if config is not None and not config.policy.head_start_burns:
        # The day turned the head start off (--no-head-start-burns): nothing may burn in it.
        early = ([f"{args.script} line {m.line_index + 1}" for m in schedule.maneuvers
                  if m.head_start] if schedule is not None else [])
        if args.recorded is not None and isinstance(args.recorded[1], dict):
            early += [f"the {args.recorded[0]}'s burn at t = {b['time_s']!r} s"
                      for b in args.recorded[1].get("burns") or []
                      if isinstance(b, dict) and isinstance(b.get("time_s"), int | float)
                      and b["time_s"] < 0.0]
        if early:
            _refuse(f"{args.config} turns policy.head_start_burns off, and this plan burns "
                    f"before the later craft's epoch: {', '.join(early)}. Not this day's upload")
    if schedule is not None:
        lead = getattr(args, "certified_lead_s", None)
        plans.append((str(args.script), [
            (m.time_s - (lead or 0.0) if m.head_start else m.time_s,
             math.hypot(*m.components_km_s), int(not m.head_start or lead is not None))
            for m in schedule.maneuvers]))
    totals: list[tuple[str, float]] = []
    if args.recorded is not None and isinstance(args.recorded[1], dict):
        kind, candidate = args.recorded
        burns = candidate.get("burns")
        if isinstance(burns, list) and all(
                isinstance(b, dict) and isinstance(b.get("time_s"), int | float)
                and isinstance(b.get("dv_inertial_km_s"), list) for b in burns):
            plans.append((f"the {kind}'s candidate", [
                (float(b["time_s"]), math.hypot(*map(float, b["dv_inertial_km_s"])), 1)
                for b in burns]))
        if isinstance(candidate.get("total_dv_km_s"), int | float):
            totals.append((f"the {kind}'s total_dv_km_s", float(candidate["total_dv_km_s"])))
    unit = _dv_unit(args)
    if args.total_dv is not None and unit is not None and args.recorded is None:
        totals.append((args.total_dv_from, args.total_dv / (1000.0 if unit == "m/s" else 1.0)))
    args.judged = [source for source, _ in plans] + [source for source, _ in totals]
    limit_km_s = rules.impulse_limit_km_s + rules.impulse_epsilon_km_s
    spacing = rules.separation_clause
    worst = 0
    args.script_violating = None
    for source, burns in plans:
        ordered = sorted(burns, key=lambda burn: (burn[2], burn[0]))
        violating: set[int] = set()
        for k, (time_s, size_km_s, side) in enumerate(ordered):
            if size_km_s > limit_km_s:
                said = (f"{source}: burn {k + 1} at t = {time_s:.3f} s is "
                        f"{size_km_s * 1000.0:.6f} m/s, over the {args.rules} rules' "
                        f"{rules.impulse_limit_km_s * 1000.0:g} m/s per impulse"
                        f"{' plus ε_v' if rules.impulse_epsilon_km_s else ''} "
                        f"({rules.impulse_clause})")
                if rules.over_impulse_limit == "disqualify":
                    _disqualified(said)
                print(f"WARNING: {said}")
                violating.add(k)
            if k and side == ordered[k - 1][2] and (
                    time_s - ordered[k - 1][0] < rules.min_separation_s - 1.0e-9):
                print(f"WARNING: {source}: burn {k + 1} at t = {time_s:.3f} s is "
                      f"{time_s - ordered[k - 1][0]:.3f} s after the one before, under the "
                      f"{rules.min_separation_s:g} s the rules ask ({spacing})")
                violating.add(k)
        totals.append((source, sum(size for _, size, _ in burns)))
        worst = max(worst, len(violating))
        if source == str(args.script) and schedule is not None:
            args.script_violating = violating
    args.worst_violations = worst
    cap_km_s = rules.total_dv_limit_km_s
    for source, total_km_s in totals:
        if cap_km_s is not None and total_km_s > cap_km_s:
            # 12 digits: 3.000000001 km/s must not read as the cap itself.
            _disqualified(f"{source}: Σ|Δv| is {total_km_s:.12g} km/s, over the {args.rules} "
                          f"rules' {cap_km_s:.12g} km/s cap, which has no tolerance "
                          f"({rules.total_clause})")
    if worst:
        args.penalty_points = rules.penalty_points * worst
        print(f"WARNING: {worst} violating maneuver{'' if worst == 1 else 's'}: "
              f"-{args.penalty_points:g} points in the score printed below ({spacing})")


def judge_boundary(args: argparse.Namespace, schedule: MissionSchedule, epoch_s: float,
                   origin_s: float) -> None:
    """§4 note 1(d) across the alignment's end, which judge_rules leaves to the report
    when nothing gave it the head start's length: epoch_s is the early craft's epoch and
    origin_s the alignment's end on the report's clock."""
    violating = getattr(args, "script_violating", None)
    early = [m for m in schedule.maneuvers if m.head_start]
    later = [m for m in schedule.maneuvers if not m.head_start]
    if violating is None or not early or not later or getattr(args, "certified_lead_s", None):
        return
    rules = interception_rules(args.rules)
    last = max(epoch_s + m.time_s for m in early)
    gap_s = origin_s + min(m.time_s for m in later) - last
    if gap_s >= rules.min_separation_s - 1.0e-9:
        return
    spacing = rules.separation_clause
    print(f"WARNING: {args.script}: the first burn after the alignment is {gap_s:.3f} s after "
          f"the last head-start burn, under the {rules.min_separation_s:g} s the rules ask "
          f"({spacing})")
    violating.add(len(early))
    worst = max(args.worst_violations, len(violating))
    if worst > args.worst_violations:
        args.worst_violations = worst
        args.penalty_points = rules.penalty_points * worst
        print(f"WARNING: {worst} violating maneuver{'' if worst == 1 else 's'}: "
              f"-{args.penalty_points:g} points in the score printed below ({spacing})")


def _said_judged(args: argparse.Namespace) -> None:
    """What of §4 note 1 was judged: the burns' clauses, and (b) on the rows."""
    rules = interception_rules(args.rules)
    clauses = rules.burn_clauses
    said = (f"{clauses} judged on {', '.join(args.judged)}" if args.judged
            else f"{clauses} not judged: no --script, certificate or verification gives the "
                 "burns")
    if args.judged_surface is not None:
        said += ("; (b) judged on the rows" if args.judged_surface
                 else "; (b) not judged: no altitude or usable position of the pursuer")
    print(f"  ({said})")


def _penalized(points: float, args: argparse.Namespace) -> float:
    """A score less §4 note 1's deductions, floored at zero (§5)."""
    return max(0.0, points - float(args.penalty_points))


def _schedule(script: str) -> MissionSchedule:
    try:
        from orbitwin.gmat_parser import mission_schedule
    except ImportError as exc:
        raise SystemExit(f"--script needs the project's venv to parse the script: {exc}") from None
    try:
        return mission_schedule(script)
    except (OrbitWinError, OSError) as exc:
        raise SystemExit(f"{script}: {exc}") from None


def _dv_unit(args: argparse.Namespace) -> str | None:
    """The unit Cv is in: the config's, else the one the rules fix, else unknown."""
    config = args.day_config
    if config is not None and config.score is not None:
        return str(config.score.delta_v_unit)
    units = interception_rules(args.rules).score_units
    return None if units is None else units[1]


def total_from_script(args: argparse.Namespace, schedule: MissionSchedule) -> None:
    """Σ|Δv| from the script's Element1-3 (km/s); a --total-dv that disagrees exits 2."""
    total_km_s = schedule.total_dv_km_s
    unit = _dv_unit(args)
    given = args.total_dv
    if unit is None:
        # Cv's unit is unknown: a given total must be the script's in one of the two.
        if given is not None and not any(
                abs(given - total_km_s * scale) <= TOTAL_DV_TOLERANCE_KM_S * scale
                for scale in (1.0, 1000.0)):
            _refuse(f"--total-dv {given!r} is neither {total_km_s:.9g} km/s nor "
                    f"{total_km_s * 1000.0:.9g} m/s, the Σ|Δv| of {args.script}'s burns")
        if given is None:
            print(f"note: {args.script}'s burns sum to {total_km_s:.9g} km/s; nothing gives the "
                  "unit of Cv, so pass --total-dv in it to score", file=sys.stderr)
        return
    scale = 1000.0 if unit == "m/s" else 1.0
    if given is not None and not abs(given - total_km_s * scale) <= TOTAL_DV_TOLERANCE_KM_S * scale:
        _refuse(f"--total-dv is {given!r} {unit} and {args.script}'s burns sum to "
                f"{total_km_s * scale:.9g} {unit} ({total_km_s:.9g} km/s), the unit Cv is in. "
                "Leave --total-dv out: the script gives it")
    args.total_dv = total_km_s * scale
    args.total_dv_from = f"the burns of --script {args.script}"


#: How far from 0 seconds since t = 0 may read on its first row. GMAT wrote
#: exactly 0 there on every certified report checked (R1, R6, R7, R8).
T0_TOLERANCE_S = 1.0e-6


def waiting_craft(
    full: np.ndarray, positions: dict[str, list[int]], header: list[str],
    precision: int | None = None,
) -> tuple[str, int] | None:
    """The craft an epoch alignment holds still, and for how many rows before t = 0.

    The later craft waits at its epoch while the earlier one flies to it, so
    its X, Y and Z repeat bit for bit. Measured on our certified R6 and R8
    reports: it first moves on row 1801 with certification.json's
    frozen_rows = 1800, the other craft on row 1; without an alignment both
    move on row 1. A craft's own epoch column must hold still with it: at a
    coarse Precision a slow first segment repeats the positions too (R1 at
    Precision 6, for one row). A craft with no epoch column must hold for more.

    t = 0 is the first row at the later epoch, as certify counts frozen_rows
    (report_parser.report_clock). At MaxStep 60 s and more R2026a may end the
    alignment on two rows at one instant, its own last row and the next
    Propagate's first (post-burn when a burn follows at once); the second still
    holds the waiting craft, so it is counted at t = 0 too.
    """
    first_move: dict[str, int] = {}
    for craft, columns in positions.items():
        own = [i for i, name in enumerate(header) if name.startswith(f"{craft}.")
               and is_epoch_parameter(name)]
        moved = np.flatnonzero(np.any(full[:, [*columns, *own]] != full[0, [*columns, *own]],
                                      axis=1))
        first_move[craft] = int(moved[0]) if moved.size else len(full)
        if not own and first_move[craft] <= 2:
            first_move[craft] = 1
    waiting = [craft for craft, row in first_move.items() if row > 1]
    if len(waiting) != 1:
        return None
    rows = first_move[waiting[0]] - 1
    early = next(iter(craft for craft in positions if craft != waiting[0]), None)
    clocks = []
    for index, name in enumerate(header):
        if early is not None and name.startswith(f"{early}.") and is_epoch_parameter(name):
            seconds = full[:, index] * (1.0 if name.endswith("ElapsedSecs") else 86400.0)
            clocks.append((max(T0_TOLERANCE_S, 2.0 * _digit(seconds, precision)), seconds))
    if clocks:
        # Back to the first of the rows at the instant the alignment ends, on the
        # finest clock; one that does not tell the alignment's own steps apart says nothing.
        at_once, seconds = min(clocks, key=lambda clock: clock[0])
        first = rows
        while first > 0 and abs(seconds[first] - seconds[first - 1]) <= at_once:
            first -= 1
        if first > 0:
            rows = first
    return waiting[0], rows


ORIGIN_HINT = (
    "pass --time-column naming the time-origin craft's ElapsedSecs, the craft with the "
    "earlier epoch under the ranking rules (clock.time_origin_craft in certification.json)"
)


def elapsed_tolerance_s(full: np.ndarray, column: int, precision: int | None) -> float:
    """1 µs, or the last digit the ReportFile's Precision prints the column to, if coarser."""
    if precision is None:
        return T0_TOLERANCE_S
    largest = float(np.max(np.abs(full[:, column]))) if full.size else 0.0
    if largest <= 0.0:
        return T0_TOLERANCE_S
    return max(T0_TOLERANCE_S, 10.0 ** (math.floor(math.log10(largest)) - precision + 1))


def require_finite_times(report: str, header: list[str], full: np.ndarray) -> None:
    """A time candidate that is not finite, named before any choice between them."""
    if len(header) != full.shape[1]:
        return
    for index, name in enumerate(header):
        if not (name.endswith(".ElapsedSecs") or is_epoch_parameter(name)):
            continue
        bad = np.flatnonzero(~np.isfinite(full[:, index]))
        if bad.size:
            raise SystemExit(
                f"{report}: data row {int(bad[0]) + 1} holds a {name} that is not finite"
            )


def earliest_elapsed_column(
    report: str, header: list[str], full: np.ndarray, pursuer: str, target: str,
    precision: int | None = None,
) -> tuple[str, bool] | None:
    """The ElapsedSecs of the craft with the earlier epoch, and whether the craft
    started apart; None without both craft's.

    Each counts from its own craft's epoch, so the earlier craft's runs ahead by
    the head start on every row, through an alignment and after it. With one
    start they agree, and the pursuer's is taken.
    """
    names = [f"{craft}.ElapsedSecs" for craft in (pursuer, target)]
    if len(header) != full.shape[1] or not all(name in header for name in names):
        return None
    first, second = header.index(names[0]), header.index(names[1])
    lead = full[:, first] - full[:, second]
    tolerance = max(elapsed_tolerance_s(full, first, precision),
                    elapsed_tolerance_s(full, second, precision))
    apart = bool(np.any(np.abs(lead) > tolerance))
    if np.all(lead >= -tolerance):
        return names[0], apart
    if np.all(lead <= tolerance):
        return names[1], apart
    raise SystemExit(
        f"{report}: neither {names[0]} nor {names[1]} runs ahead of the other on every row, "
        "so the report does not say which craft started first"
    )


def default_time_column(
    report: str, header: list[str], data: np.ndarray, precision: int | None = None
) -> int:
    """The time column by name: the ElapsedSecs, else the one numeric epoch.

    Several that disagree, as both craft's ElapsedSecs do after an epoch
    alignment, or none at all, are refused: which is T_team is the config's
    time origin (policy.mission_time_origin), not the column order.
    """
    seconds = [index for index, name in enumerate(header) if name.endswith(".ElapsedSecs")]
    candidates = seconds or [index for index, name in enumerate(header) if is_epoch_parameter(name)]
    if not candidates:
        raise SystemExit(f"{report}: no ElapsedSecs or numeric epoch column; pass --time-column")
    first = candidates[0]
    # Only epochs of one kind can disagree by an alignment; A1 and UTC always differ.
    kind = header[first].rsplit(".", 1)[-1]
    # In seconds: ElapsedDays and a ModJulian count days. A ModJulian's last
    # digit is 0.9 us at Precision 16, an alignment's offset seconds or more.
    scale = 1.0 if kind == "ElapsedSecs" else 86400.0
    for other in candidates[1:]:
        if header[other].rsplit(".", 1)[-1] != kind:
            continue
        tolerance_s = (
            max(elapsed_tolerance_s(data, first, precision),
                elapsed_tolerance_s(data, other, precision))
            if kind == "ElapsedSecs" else 1.0e-3
        )
        if np.any(np.abs(data[:, other] - data[:, first]) * scale > tolerance_s):
            if kind.endswith("ModJulian"):
                # One instant for both craft once the alignment's rows are skipped.
                hint = "pass --frozen-rows from certification.json, or --time-column"
            else:
                hint = ORIGIN_HINT if seconds else "pass --time-column"
            raise SystemExit(
                f"{report}: {header[first]} and {header[other]} differ, as after an epoch "
                f"alignment; {hint}"
            )
    return first


def _gmat_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered in {"true", "on"}:
        return True
    if lowered in {"false", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"not a GMAT boolean: {text!r}")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"not a positive integer: {text!r}")
    return value


def _one_character(text: str) -> str:
    if len(text) != 1:
        raise argparse.ArgumentTypeError(f"one character, as it appears in the file: {text!r}")
    return text


#: The hand check's exit for a report carrying no target position: it cannot read the
#: distance off it, and step 7's --certificate form can (docs/RUNBOOK.md step 7).
TARGET_NOT_IN_REPORT_EXIT = 7
#: The line main prints last for it.
NO_TARGET: list[str] = []
#: What step 7 --certificate prints, and exits 0, on such a report.
NO_TARGET_WARNING = (
    "WARNING: the official report carries no {target} position; d_min and the interception "
    "were proven on certify's diagnostic flight of the same script, not on this report"
)

#: What this run's binding left unchecked, warned past: main then exits UNBOUND_EXIT.
UNBOUND: list[str] = []
#: A pass that holds only with the by-eye comparison of docs/RUNBOOK.md step 6b.7 done.
UNBOUND_EXIT = 3
UNBOUND_LINE = ("WARNING: burns not bound by this check: the 6b.7 by-eye comparison is "
                "mandatory")


def main() -> int:
    """_main, with a binding warned past repeated as the last line and exit 3 for a pass,
    and an undecided §4 note 1(b) (exit 4, 5 or 6) last of all."""
    UNBOUND.clear()
    SURFACE.clear()
    NO_TARGET.clear()
    code = _main()
    if UNBOUND:
        print(UNBOUND_LINE)
        code = UNBOUND_EXIT if code == 0 else code
    if code == TARGET_NOT_IN_REPORT_EXIT:
        print(NO_TARGET[0])
        return code
    if SURFACE and code in (0, UNBOUND_EXIT):
        # Last: neither a pass nor a refusal, a person decides (docs/RUNBOOK.md step 7).
        surface, line = SURFACE[0]
        print(line)
        return surface
    return code


def _main() -> int:
    # No abbreviations: --cr would otherwise stand for --cr-km and drop the unit.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("report")
    parser.add_argument("--pursuer", help="default: the config's pursuer_name, else SC_Human")
    parser.add_argument("--target", help="default: the config's target_name, else SC_Alien")
    parser.add_argument(
        "--rules", type=rules_name,
        help="rulebook whose interception sphere is checked: without --config, default ranking; "
        "with one, the config's, which must be ranking unless this names it",
    )
    parser.add_argument(
        "--config", help="the day's config (scripts/ranking_day_config.py): rules, craft, T_max, "
        "time origin and coefficients, so none is copied by hand",
    )
    parser.add_argument(
        "--certificate", help="certify's certification.json: frozen rows, total delta-v, the "
        "script beside it, and T_team to compare",
    )
    parser.add_argument(
        "--verification", help="for the report the GUI wrote, which --certificate refuses: the "
        "<run>/diagnostic/verification.json certify wrote, held to the config's coefficients",
    )
    parser.add_argument("--radius-km", type=_positive_finite,
                        help="override the rulebook's sphere")
    parser.add_argument("--pursuer-cols", help="comma-separated X,Y,Z column indices (0-based)")
    parser.add_argument("--target-cols", help="comma-separated X,Y,Z column indices (0-based)")
    parser.add_argument("--script", help="the script that wrote the report; its ReportFile "
                        "decides how the report is split")
    parser.add_argument("--report-file", help="which ReportFile in --script, if it has several")
    parser.add_argument("--delimiter", type=_one_character,
                        help="the delimiter character in the file (a script's '\\t' writes '\\')")
    parser.add_argument("--write-headers", type=_gmat_bool)
    parser.add_argument("--fixed-width", type=_gmat_bool)
    parser.add_argument("--column-width", type=_positive_int)
    parser.add_argument("--columns", help="comma-separated Add list, for a report without headers")
    parser.add_argument("--kt", type=float)
    parser.add_argument("--ct", type=float)
    parser.add_argument("--kv", type=float)
    parser.add_argument("--cv", type=float)
    parser.add_argument("--cr-km", type=_positive_finite,
                        help="C_r in km, the ranking rules' distance term (2026/11/04)")
    parser.add_argument("--total-dv", type=float, help="Σ|Δv| in the unit Cv uses; with "
                        "--script and no certificate the script's, and refused if it differs")
    parser.add_argument(
        "--binding", choices=("enforce", "warn"), default="enforce",
        help="with --script and no certificate: warn turns a refusal over the burns into a "
        "WARNING line, never one over the end, the name or the total; 6b.7's by-eye "
        "comparison is then mandatory (docs/RUNBOOK.md step 7)",
    )
    parser.add_argument(
        "--accept-end-time-binding-only", action="store_true",
        help="go on with a report that has no EarthMJ2000Eq velocity of the pursuer, whose "
        "burns cannot be checked (docs/RUNBOOK.md step 6b.6)",
    )
    parser.add_argument(
        "--frozen-rows", type=int,
        help="leading rows an epoch alignment wrote before t = 0, as certification.json "
        "records them; skipped",
    )
    parser.add_argument(
        "--time-column", help="header of the column read as time (default: the ElapsedSecs, "
        "found by name); after an epoch alignment, the origin craft's"
    )
    parser.add_argument(
        "--horizon-s", type=_positive_finite,
        help="T_max as announced, the day config's policy.horizon_s, on the --time-column "
        "clock; a scored miss passes only if its last row is there",
    )
    args = parser.parse_args()
    refuse_unpinned()
    cr_by_hand = args.cr_km is not None
    args.cr_used = False
    # By hand: a certificate binds its report and script by SHA-256 instead.
    schedule = _schedule(args.script) if args.script and args.certificate is None else None
    if schedule is None and (args.binding != "enforce" or args.accept_end_time_binding_only):
        parser.error("--binding and --accept-end-time-binding-only are for a report held to "
                     "--script by hand, without --certificate")
    certified_t_team = _fill_from_day_files(args, parser)
    if schedule is not None:
        total_from_script(args, schedule)
    judge_rules(args, schedule)
    if cr_by_hand and not interception_rules(args.rules).distance_term.takes_cr:
        parser.error(f"--cr-km is the ranking rules' C_r; the {args.rules} rules' distance term "
                     "has none")

    fields = (args.delimiter, args.write_headers, args.fixed_width, args.column_width, args.columns)
    if args.script is not None and any(value is not None for value in fields):
        parser.error("--script gives the ReportFile format; do not also pass its fields")
    if args.report_file is not None and args.script is None:
        parser.error("--report-file names a ReportFile in --script")
    names: list[str] | None
    precision: int | None = None
    spec: ReportFileSpec | None = None
    if args.script is not None:
        spec = report_spec(args.script, args.report, args.report_file)
        fmt, names = report_format(spec), list(spec.parameters)
        precision = spec.precision
        if spec.append:
            print(
                f"warning: {spec.name}.AppendToExistingFile is true; every run of the script "
                "adds its rows to the file, so it must hold one run only",
                file=sys.stderr,
            )
    else:
        fmt = ReportFormat(
            write_headers=DEFAULT_REPORT_FORMAT.write_headers
            if args.write_headers is None else args.write_headers,
            delimiter=args.delimiter or DEFAULT_REPORT_FORMAT.delimiter,
            fixed_width=DEFAULT_REPORT_FORMAT.fixed_width
            if args.fixed_width is None else args.fixed_width,
            column_width=args.column_width or DEFAULT_REPORT_FORMAT.column_width,
        )
        names = (
            [name.strip() for name in args.columns.strip().strip("{}").split(",")]
            if args.columns else None
        )
    radius_km = args.radius_km
    if radius_km is None:
        radius_km = interception_rules(args.rules).intercept_radius_km

    header, data = read_report(args.report, fmt, names)
    columns = data.shape[1]
    if not 0 <= args.frozen_rows < data.shape[0]:
        raise SystemExit(f"--frozen-rows {args.frozen_rows} leaves no row of {data.shape[0]}")
    full, data = data, data[args.frozen_rows:]
    require_finite_times(args.report, header, full)
    if len(header) == columns and not args.target_cols and not position_frames(header, args.target):
        return _without_target(args, header, full, data, precision, certified_t_team, schedule,
                               spec)

    def flagged(flag: str) -> list[int]:
        try:
            indices = [int(value) for value in flag.split(",")]
        except ValueError:
            raise SystemExit(f"column flags take three integers, got {flag!r}") from None
        if len(indices) != 3:
            raise SystemExit("column flags take exactly three indices")
        if not all(-columns <= i < columns for i in indices):
            raise SystemExit(f"column flags {flag} fall outside the {columns} values of a row")
        indices = [i % columns for i in indices]
        if len(header) == columns:
            # Named, they must be one craft's X, Y and Z in one frame.
            names = [header[i] for i in indices]
            if flagged_frame(header, columns, indices) is None:
                raise SystemExit(f"column flags {flag} name {names}, not one craft's X, Y and Z")
        return indices

    def resolve(flag: str | None, spacecraft: str, other: str, other_flag: str | None) -> list[int]:
        if flag:
            return flagged(flag)
        if other_flag:
            frame = flagged_frame(header, columns, flagged(other_flag))
            return position_columns(header, columns, spacecraft, None, frame)
        return position_columns(header, columns, spacecraft, other)

    columns_of = {
        args.pursuer: resolve(args.pursuer_cols, args.pursuer, args.target, args.target_cols),
        args.target: resolve(args.target_cols, args.target, args.pursuer, args.pursuer_cols),
    }
    if columns_of[args.pursuer] == columns_of[args.target]:
        raise SystemExit("the two craft resolve to the same columns")
    if len(header) == columns:
        # Named, the two sides must be two craft in one frame, flags or not.
        sides = [columns_of[args.pursuer], columns_of[args.target]]
        owners = [{header[i].split(".")[0] for i in side} for side in sides]
        frames = [flagged_frame(header, columns, side) for side in sides]
        if owners[0] == owners[1]:
            raise SystemExit(f"both craft's columns are {sorted(owners[0])[0]}'s")
        if frames[0] != frames[1]:
            raise SystemExit(
                f"the two craft's positions are in {frames[0]} and {frames[1]}; a distance "
                "needs one frame"
            )
    pursuer = data[:, columns_of[args.pursuer]]
    target = data[:, columns_of[args.target]]
    waiting = waiting_craft(full, columns_of, header if len(header) == columns else [],
                            precision)
    if args.same_epoch_claimed:
        seconds = [k for k, name in enumerate(header)
                   if len(header) == columns and name.endswith(".ElapsedSecs")]
        apart = len(seconds) > 1 and not np.allclose(full[:, seconds[0]], full[:, seconds[1]],
                                                     rtol=0.0, atol=1.0e-6)
        if waiting is not None or apart:
            raise SystemExit(
                f"{args.report}: the config's craft share an epoch (it sets no "
                "policy.mission_time_origin, or earliest without "
                "policy.closest_approach_window), but this report holds an epoch alignment; "
                "it is another day's report, or pass --time-column"
            )
    named = len(header) == columns
    ranking = interception_rules(args.rules).time_origin == "earliest"
    # Under the ranking rules T_team counts from the earlier epoch with or without a
    # config: both craft's ElapsedSecs always say which, one column only sometimes.
    earliest = (
        earliest_elapsed_column(args.report, header, full, args.pursuer, args.target, precision)
        if (args.earliest_origin or ranking) and named else None
    )
    present = [f"{c}.ElapsedSecs" for c in (args.pursuer, args.target)
               if named and f"{c}.ElapsedSecs" in header]
    if (ranking and named and waiting is None and len(present) == 1 and earliest is None
            and args.certified_origin is None and not args.certified_one_start
            and (args.earliest_origin or args.time_column == present[0])):
        # One clock read as T_team and no alignment rows: it is T_team only if the craft
        # start together, which only the day's config and script, or a certificate, show.
        # Found by name and unnamed, it is not scored, and the note below says so.
        reason = _one_start_shown(args)
        if reason is not None:
            raise SystemExit(
                f"{args.report}: the report carries only {present[0]} and no epoch "
                f"alignment's rows, so it does not say whether that craft started first; "
                f"{reason}. Pass --certificate, or --config and --script for the day"
            )
        earliest = (present[0], False)
    if args.earliest_origin and named and earliest is None:
        if waiting is not None:
            early = args.target if waiting[0] == args.pursuer else args.pursuer
            if f"{early}.ElapsedSecs" not in header:
                raise SystemExit(
                    f"{args.report}: an epoch alignment holds {waiting[0]} still, so {early} "
                    f"started first and its ElapsedSecs is T_team, but the report has no "
                    f"{early}.ElapsedSecs; T_team is clock.t_team_s in certification.json"
                )
            earliest = (f"{early}.ElapsedSecs", True)
        elif args.certified_one_start and len(present) == 1:
            # One start, by the certificate: either craft's is T_team.
            earliest = (present[0], False)
        elif args.certified_origin is not None and args.certified_origin in present:
            # The certificate names the craft that started first from the probed epochs.
            earliest = (args.certified_origin, True)
        else:
            # A single column cannot show a head start that left no rows behind.
            raise SystemExit(
                f"{args.report}: the config's craft start apart (it sets "
                f"policy.closest_approach_window), and the report carries "
                f"{' and '.join(present) or 'no ElapsedSecs'}, which does not say which craft "
                "started first. Pass the --certificate that names it; by hand, read T_team "
                "from clock.t_team_s in certification.json"
            )
    if earliest is not None:
        column, apart = earliest
        if apart and args.certified_origin is not None and args.certified_origin != column:
            raise SystemExit(
                f"{args.report}: {column} runs ahead of the other craft's ElapsedSecs, so its "
                "craft started first and it is T_team under the earliest origin, but the "
                f"certificate's clock.time_origin_craft gives {args.certified_origin}"
            )
        if args.time_column is None and args.earliest_origin:
            args.time_column = column
            said = ("runs ahead of the other's" if apart and len(present) == 2
                    else "agrees with the other's" if len(present) == 2
                    else "is the one in the report")
            print(f"note: {column} is T_team: the config's time origin is the craft that started "
                  f"first, and its ElapsedSecs {said}", file=sys.stderr)
        elif apart and args.time_column is not None and args.time_column != column:
            given = ("the certificate's clock.time_origin_craft gives"
                     if args.time_column == args.certified_origin else "--time-column is")
            raise SystemExit(
                f"{args.report}: {column} runs ahead of the other craft's ElapsedSecs, so its "
                f"craft started first and it is T_team under the {args.rules} rules (the "
                f"earlier epoch), but {given} {args.time_column}"
            )
    if args.time_column is not None:
        if args.time_column not in header:
            raise SystemExit(f"no column {args.time_column!r}; the header is {header}")
        time_index = header.index(args.time_column)
        if time_index >= columns:
            raise SystemExit(
                f"{args.report}: {args.time_column} is name {time_index + 1} of {len(header)}, "
                f"but a row has {columns} values"
            )
    elif named:
        time_index = default_time_column(args.report, header, data, precision)
    else:
        # No names: column 0, as before, and never scored.
        time_index = 0
    if (named and waiting is not None and interception_rules(args.rules).time_origin == "earliest"
            and header[time_index] == f"{waiting[0]}.ElapsedSecs"):
        # The waiting craft started later: its ElapsedSecs is T_team short by the head start.
        raise SystemExit(
            f"{args.report}: an epoch alignment holds {waiting[0]} still, so it started later "
            f"and {header[time_index]} is short of T_team by the head start; under the "
            f"{args.rules} rules T_team counts from the earlier craft's epoch"
        )
    if waiting is not None and args.frozen_rows < waiting[1]:
        # The two craft are at two epochs there: only an inertial frame keeps the distance.
        names = [header[i] for i in columns_of[args.pursuer]] if len(header) == columns else []
        frame = names[0].split(".")[1] if names and len(names[0].split(".")) == 3 else (
            "EarthMJ2000Eq" if names else None
        )
        if frame != "EarthMJ2000Eq":
            raise SystemExit(
                f"{args.report}: rows {args.frozen_rows + 1}-{waiting[1]} are an epoch "
                f"alignment's, where the craft are at two epochs, and positions in "
                f"{frame or 'unnamed columns'} do not give their distance there; pass "
                f"--frozen-rows {waiting[1]}, or report EarthMJ2000Eq"
            )
    frozen_doubt = (waiting is None and args.frozen_rows > 0) or (
        waiting is not None and waiting[1] != args.frozen_rows
    )
    if waiting is None and args.frozen_rows > 0:
        print(
            f"note: --frozen-rows skips {args.frozen_rows} rows, but no craft holds still in "
            "the report as an epoch alignment's rows do; check it against certification.json",
            file=sys.stderr,
        )
    if waiting is not None and waiting[1] != args.frozen_rows:
        print(
            f"note: {waiting[0]} holds still for {waiting[1]} rows before it moves, as an epoch "
            f"alignment writes them; --frozen-rows is {args.frozen_rows}",
            file=sys.stderr,
        )
    times = data[:, time_index]
    what = header[time_index] if named else f"column {time_index}"
    if named and is_text_parameter(header[time_index]):
        raise SystemExit(
            f"{args.report}: the time column {what} is text, not a number; "
            "pass --time-column naming an ElapsedSecs"
        )
    # A NaN fails every comparison: a NaN distance is never inside, a NaN time
    # never goes back.
    bad = np.flatnonzero(~np.isfinite(np.column_stack([times, pursuer, target])).all(axis=1))
    if bad.size:
        raise SystemExit(
            f"{args.report}: data row {args.frozen_rows + int(bad[0]) + 1} holds a time or "
            f"position that is not finite (time column {what})"
        )
    # A second run starts its epochs over. Judged on an epoch column: the
    # first column may be anything, an Altitude say.
    if not named:
        # Nothing says which column is an epoch.
        epochs = []
        print(
            f"note: {len(header)} column names for {columns} values a row, so a report of two "
            "runs is not looked for; pass --script or --columns",
            file=sys.stderr,
        )
    elif is_epoch_parameter(header[time_index]):
        epochs = [time_index]
    else:
        epochs = [index for index, name in enumerate(header) if is_epoch_parameter(name)]
    if named and not epochs:
        print(
            "note: no column is a numeric epoch, so a report of two runs is not looked for",
            file=sys.stderr,
        )
    # T_team is the time-origin craft's ElapsedSecs.
    # Chosen by default it is that only with no alignment, and then reads 0 at t = 0.
    why_not = None
    if not named:
        why_not = "no column names say which is time; pass --script or --columns"
    elif not header[time_index].endswith(".ElapsedSecs"):
        has_seconds = any(name.endswith(".ElapsedSecs") for name in header)
        why_not = f"the time column {header[time_index]} is not an ElapsedSecs; " + (
            ORIGIN_HINT if has_seconds
            else "the report has none, so T_team is clock.t_team_s in certification.json"
        )
    elif args.time_column is None:
        # Found by name it may still be the wrong craft's: a Toggle can keep an
        # alignment's rows out of the report altogether.
        why_not = f"{header[time_index]} was found by name; {ORIGIN_HINT}"
        if waiting is not None:
            why_not += f"; an epoch alignment holds {waiting[0]} still"
        elif abs(times[0]) > T0_TOLERANCE_S:
            why_not += (
                f"; it reads {times[0]:g} s on the first row kept, not 0, so check "
                "--frozen-rows against certification.json"
            )
    if why_not is not None and getattr(args, "console_copied", False):
        # Its exact runs ran the copy, not the script: the last row is what ties them.
        print(f"VERDICT: the certificate's T_team cannot be compared, which a console copy's "
              f"report needs: {why_not}")
        return 2
    unit = " s" if named and header[time_index].endswith(".ElapsedSecs") else ""
    if why_not is not None and named:
        print(f"note: not scored: {why_not}", file=sys.stderr)
    elif waiting is not None and named and earliest is None:
        print(
            f"note: an epoch alignment held {waiting[0]} still; {header[time_index]} is T_team "
            "only if its craft is the time origin, the one that started first under the ranking "
            "rules (clock.time_origin_craft in certification.json)",
            file=sys.stderr,
        )
    if epochs:
        _one_run(args, header, data, epochs[0], precision)
    if schedule is not None:
        assert spec is not None
        _hold_to_official(args.script, args.report, spec)
        bind_to_script(args, schedule, header, full, time_index, waiting, precision)
    distance = np.linalg.norm(pursuer - target, axis=1)
    # Under from_origin d_min counts the alignment's rows too, as score_candidate and
    # certify's replayed score do (Scenario.alignment_minimum_distance_km).
    alignment_min_km = None
    if args.from_origin and args.earliest_origin and args.frozen_rows > 0:
        frames_held = {flagged_frame(header, columns, columns_of[craft])
                       for craft in (args.pursuer, args.target)} if named else {None}
        if frames_held != {"EarthMJ2000Eq"}:
            raise SystemExit(
                f"{args.report}: the config's d_min window is from_origin, which counts the "
                f"{args.frozen_rows} rows --frozen-rows skips, and positions not in "
                "EarthMJ2000Eq do not give the distance there, where the craft are at two epochs"
            )
        skipped = full[:args.frozen_rows]
        alignment_min_km = float(np.linalg.norm(
            skipped[:, columns_of[args.pursuer]] - skipped[:, columns_of[args.target]], axis=1
        ).min())

    inside = np.flatnonzero(distance <= radius_km)
    closest = int(distance.argmin())
    last = len(times) - 1

    print(f"file          : {args.report}")
    shown = header[time_index] if len(header) == columns else f"column {time_index}"
    print(f"time column   : {shown}")
    print(f"rows          : {len(times)}   columns: {columns}   skipped: {args.frozen_rows}")
    print(f"first row     : t = {times[0]:.6f}{unit}")
    print(f"LAST row      : t = {times[last]:.6f}{unit}   dr = {distance[last]:.6f} km")
    print(f"closest row   : t = {times[closest]:.6f}{unit}   dr = {distance[closest]:.6f} km")
    print(f"rows within {radius_km:g} km: {inside.size}")
    # §4 note 1(b) on every row the mission flew up to the entry, or to the end for a miss.
    judged_rows = args.frozen_rows + (int(inside[0]) + 1 if inside.size else len(times))
    judge_surface(args, header if named else [], full[:judged_rows], columns_of[args.pursuer],
                  "the first row inside" if inside.size else "the last row", precision)
    print()

    if inside.size == 0:
        code = _miss_verdict(args, times, distance, closest, last, unit, why_not,
                             _horizon_tolerance(args.horizon_s, precision), frozen_doubt,
                             alignment_min_km)
        if cr_by_hand and not args.cr_used:
            print("C_r not used: the miss was not priced, which needs --kt, --ct, --kv, --cv "
                  "and --total-dv beside --cr-km, and a last row at T_max")
        return _against_certificate(code, certified_t_team, times[last], why_not,
                                    _horizon_tolerance(args.horizon_s, precision))

    entry = int(inside[0])
    print(f"first row inside : t = {times[entry]:.6f}{unit}   dr = {distance[entry]:.6f} km")
    print("rows around it:")
    for k in range(max(0, entry - 2), min(len(times), entry + 3)):
        mark = "  <-- inside" if distance[k] <= radius_km else ""
        print(f"    t = {times[k]:14.6f}   dr = {distance[k]:12.6f} km{mark}")
    print()

    last_inside = distance[last] <= radius_km
    tolerance = _horizon_tolerance(args.horizon_s, precision)
    if args.horizon_s is not None and why_not is not None:
        print(f"note: the entry was not checked against T_max: {why_not}", file=sys.stderr)
    elif args.horizon_s is not None and times[entry] > args.horizon_s + tolerance:
        print(f"VERDICT: the first row inside is at {times[entry]:.6f}{unit}, after T_max = "
              f"{args.horizon_s:.6f} s. An entry after T_max is not an interception.")
        if cr_by_hand:
            print("C_r not used: nothing was priced")
        return 1
    shallow = last_inside and _shallow_last_row(args, radius_km, float(distance[last]))
    if last_inside:
        # A refused shallow row has its VERDICT above; this is then only a note.
        print("note: the LAST row is inside the sphere." if shallow
              else "VERDICT: the LAST row is inside the sphere.")
        print("A validator reading the completion time off the last row sees the interception.")
    else:
        gap = times[last] - times[entry]
        print("VERDICT: the last row is NOT inside the sphere.")
        gap_unit = "s" if unit else f"(in {header[time_index] if named else f'column {time_index}'})"
        print(f"A validator reading the last row sees Tteam = {times[last]:.6f} {gap_unit}, "
              "which is")
        print(f"{gap:.6f} {gap_unit} after the entry at {times[entry]:.6f} {gap_unit}. "
              "Measured 2026-09-05: that")
        print("cost 22.2266 points. Set policy.mission_end_policy = 'interception' and re-render.")

    if cr_by_hand:
        print("C_r not used: a hit's distance term is 50 at any C_r")
    scoring = None not in (args.kt, args.ct, args.kv, args.cv, args.total_dv)
    if scoring and why_not is not None:
        print()
        print(f"score not computed: {why_not}")
    elif scoring:
        dv_term = _logistic(25.0, args.kv * (args.total_dv - args.cv))
        print()
        for label, t_team in (("validator reads last row", times[last]),
                              ("rule 4 first entry", times[entry])):
            time_term = _logistic(25.0, args.kt * (t_team - args.ct))
            print(f"  {label:<26} Tteam = {t_team:11.4f} s -> "
                  f"score {_penalized(50.0 + time_term + dv_term, args):.4f}")
        _said_judged(args)
        read_last = _penalized(
            50.0 + _logistic(25.0, args.kt * (times[last] - args.ct)) + dv_term, args)
        if last_inside and not shallow and not _hold_to_certified_score(args, read_last):
            return 1

    return _against_certificate(0 if last_inside and not shallow else 1, certified_t_team,
                                times[last], why_not, tolerance)


def _one_run(args: argparse.Namespace, header: list[str], data: np.ndarray, epoch: int,
             precision: int | None) -> None:
    """Refuse a report whose epoch goes back: a second run starts its epochs over."""
    order = data[:, epoch]
    # R2026a ended R8's alignment at MaxStep 2700 on two rows 3e-7 s apart, the
    # second earlier; a second run goes back to its start.
    per_s = 1.0 if header[epoch].endswith("ElapsedSecs") else 86400.0
    jitter = max(T0_TOLERANCE_S, 2.0 * _digit(order * per_s, precision)) / per_s
    back = np.flatnonzero(np.diff(order) < -jitter)
    if back.size:
        k = int(back[0])
        raise SystemExit(
            f"{args.report}: {header[epoch]} goes back from {order[k]:.6f} to {order[k + 1]:.6f} at "
            f"data row {args.frozen_rows + k + 2}, so the report holds more than one run "
            "(AppendToExistingFile, or SolverIterations = All)"
        )


def _same_printed(printed: float, value: float, precision: int | None) -> bool:
    """Whether ``printed`` is ``value`` written to ``precision`` significant digits and read
    back, within half a printed digit or 2 ulp (to a nanosecond in a second's worth of
    digits without a precision). R8's 5911.5072750952095 prints as 5911.507275095209."""
    if precision is None:
        return abs(printed - value) <= 1.0e-9 * max(1.0, abs(value))
    if float(f"{value:.{precision}g}") == printed:
        return True
    half = 0.5 * 10.0 ** (math.floor(math.log10(abs(value))) - precision + 1) if value else 0.0
    return abs(printed - value) <= max(half, 2.0 * math.ulp(value)) * (1.0 + 1.0e-9)


def _without_target(
    args: argparse.Namespace, header: list[str], full: np.ndarray, data: np.ndarray,
    precision: int | None, certified_t_team: float | None, schedule: MissionSchedule | None,
    spec: ReportFileSpec | None,
) -> int:
    """A report with no position of the target, as the organizer's workshop template writes.

    No distance can be read off it. By hand it is bound to --script and then exits
    TARGET_NOT_IN_REPORT_EXIT. With --certificate, certify proved the interception
    and d_min on its diagnostic flight of the same script, and recorded so
    (official_report_rows); the report, bound by SHA-256, must end at the certified
    T_team to its printed digits, and the score is the certificate's evidence priced
    under the config.
    """
    report, target = args.report, args.target
    seconds = [f"{craft}.ElapsedSecs" for craft in (args.pursuer, args.target)
               if f"{craft}.ElapsedSecs" in header]
    column = args.time_column or args.certified_origin
    if column is None and len(seconds) == 2:
        found = earliest_elapsed_column(report, header, full, args.pursuer, args.target, precision)
        column = found[0] if found is not None else None
    elif column is None and seconds and (args.certified_one_start or args.certificate is None):
        column = seconds[0]
    if column is not None and column not in header:
        raise SystemExit(f"{report}: no column {column!r}, the time-origin craft's ElapsedSecs; "
                         f"the header is {header}")
    epochs = [i for i, name in enumerate(header) if is_epoch_parameter(name)]
    time_index = header.index(column) if column is not None else (epochs[0] if epochs else None)
    frames = position_frames(header, args.pursuer)
    own = list(frames[min(frames, key=_frame_order)]) if frames else []
    waiting = waiting_craft(full, {args.pursuer: own}, header, precision) if own else None
    if time_index is not None:
        _one_run(args, header, data, time_index, precision)
    if args.certificate is None:
        if schedule is not None:
            assert spec is not None
            if time_index is None:
                _refuse(f"{report} has no ElapsedSecs or epoch column to hold to {args.script}")
            _hold_to_official(args.script, report, spec)
            bind_to_script(args, schedule, header, full, time_index, waiting, precision)
        NO_TARGET.append(
            f"VERDICT: NOT JUDGED BY HAND: {report} carries no {target} position, so no distance, "
            "interception or d_min can be read from it. Only step 7's --certificate form on the "
            "report certify wrote can pass it (docs/RUNBOOK.md step 7)")
        return TARGET_NOT_IN_REPORT_EXIT
    certificate = args.certificate_body
    record = certificate.get("official_report_rows")
    if not isinstance(record, dict) or record.get("carries_target_position") is not False:
        _refuse(f"{report} carries no {target} position, and the certificate does not record "
                "that certify proved it on its diagnostic flight (official_report_rows): certify "
                "again")
    candidate = certificate.get("candidate") or {}
    parity = certificate.get("final_script_parity") or {}
    hit = candidate.get("successful") is True
    if hit != (parity.get("first_entry_time_s") is not None):
        _refuse("the certificate's candidate and its final replay disagree on the interception")
    if column is None or time_index is None:
        raise SystemExit(f"{report}: no ElapsedSecs of the craft T_team counts from "
                         f"({args.certified_origin or ' or '.join(seconds) or 'none'}), so its "
                         "last row cannot be held to the certificate's T_team")
    times = data[:, time_index]
    if not np.all(np.isfinite(times)):
        raise SystemExit(f"{report}: a {column} that is not finite")
    last_s = float(times[-1])
    print(f"file          : {report}")
    print(f"time column   : {column}")
    print(f"rows          : {len(times)}   columns: {len(header)}   skipped: {args.frozen_rows}")
    print(f"first row     : t = {float(times[0]):.6f} s")
    print(f"LAST row      : t = {last_s:.6f} s   dr = not in this report (no {target} position)")
    judge_surface(args, header, full, own, "the last row", precision)
    print()
    rules = interception_rules(args.rules)
    assert certified_t_team is not None
    if hit:
        if not _same_printed(last_s, certified_t_team, precision):
            print(f"VERDICT: the last row is at {last_s!r} s and the certificate's T_team is "
                  f"{certified_t_team!r} s: they differ in the printed digits, so this is not "
                  "the report certify measured.")
            return 1
        print(f"The last row is the certificate's T_team = {certified_t_team:.6f} s, a hit "
              "certify's diagnostic flight stopped inside the sphere.")
        event = (candidate.get("metadata") or {}).get("gmat_event") or {}
        depth = event.get("stop_depth_km")
        if args.stop_depth_gate is not None and isinstance(depth, int | float):
            setting_km, tolerance, allowed = args.stop_depth_gate
            asked_km = min(setting_km, 0.5 * rules.intercept_radius_km)
            print(f"depth         : {float(depth) * 1000.0:.4g} m inside on certify's "
                  "diagnostic flight")
            if float(depth) < asked_km * (1.0 - tolerance) and not allowed:
                print(f"VERDICT: short of the config's policy.stop_depth_km = "
                      f"{asked_km * 1000.0:.4g} m, and policy.allow_shallow_stop_submission is "
                      "not set.")
                return 1
    else:
        # The distance priced must be the final replay's closest approach, or under
        # from_origin the alignment's when closer: nothing else gives one here.
        scored, flown = record.get("scored_distance_km"), parity.get("minimum_distance_km")
        if not (isinstance(scored, int | float) and isinstance(flown, int | float)
                and math.isfinite(scored) and math.isfinite(flown)
                and scored <= flown * (1.0 + 1.0e-12)
                and (args.from_origin or scored >= flown * (1.0 - 1.0e-12))):
            _refuse(f"the certificate's official_report_rows.scored_distance_km {scored!r} is "
                    f"not its final replay's minimum_distance_km {flown!r}"
                    + (", or the alignment's closer one" if args.from_origin else ""))
        tolerance = _horizon_tolerance(args.horizon_s, precision)
        if args.horizon_s is None or not abs(last_s - args.horizon_s) <= tolerance or not (
                abs(certified_t_team - args.horizon_s) <= tolerance):
            print(f"VERDICT: a miss must run to T_max = {args.horizon_s} s, and the last row is "
                  f"at {last_s:.6f} s (certificate's T_team {certified_t_team:.6f} s).")
            return 1
        print(f"The last row is at T_max = {args.horizon_s:.6f} s: a miss, its d_min "
              f"{float(record.get('scored_distance_km', math.nan)):.6f} km on certify's "
              "diagnostic flight.")
    if None not in (args.kt, args.ct, args.kv, args.cv, args.total_dv) and (
            not rules.distance_term.takes_cr or args.cr_km is not None):
        distance_points = (50.0 if hit else distance_term(
            float(record.get("scored_distance_km", math.inf)), rules, args.cr_km))
        args.cr_used = not hit
        t_team = last_s if hit else float(args.horizon_s)
        time_term = _logistic(25.0, args.kt * (t_team - args.ct))
        dv_term = _logistic(25.0, args.kv * (args.total_dv - args.cv))
        score = _penalized(distance_points + time_term + dv_term, args)
        print()
        # The lines step 7's other verdicts print, read by orbitwin day.
        if hit:
            print(f"  {'validator reads last row':<26} Tteam = {t_team:11.4f} s -> "
                  f"score {score:.4f}")
        else:
            deducted = f" - {args.penalty_points:g}" if args.penalty_points else ""
            print(f"  scored miss: distance {distance_points:.4f} + time {time_term:.4f} + "
                  f"delta-v {dv_term:.4f}{deducted} = {score:.4f}")
        print("  (the distance term from certify's diagnostic flight)")
        _said_judged(args)
        if not _hold_to_certified_score(args, score):
            return 1
    print(NO_TARGET_WARNING.format(target=target))
    return 0


#: How far a score computed here may sit from the certificate's, both of the printed rows.
SCORE_TOLERANCE = 1.0e-6


def _hold_to_certified_score(args: argparse.Namespace, score: float) -> bool:
    """False, with a VERDICT, when the certificate prices the report's rows otherwise."""
    record = (getattr(args, "certificate_body", None) or {}).get("official_report_rows")
    certified = record.get("score") if isinstance(record, dict) else None
    if not isinstance(certified, int | float):
        return True
    if abs(score - float(certified)) <= SCORE_TOLERANCE:
        print(f"The score matches the certificate's {float(certified):.4f}, priced on these rows.")
        return True
    print(f"VERDICT: the score here is {score:.6f} and the certificate's, priced on these rows, "
          f"is {float(certified):.6f}: not the report or coefficients certify priced.")
    return False


#: How far inside a GUI report's last row must be, for what reading and rounding the
#: printed digits can move it: the rules ask only that it be inside (§3).
RULE_MARGIN_KM = 1.0e-6


def _shallow_last_row(args: argparse.Namespace, radius_km: float, last_km: float) -> bool:
    """True when the last row is not deep enough: certify's stop depth for its own report
    (--certificate), the rules' sphere less RULE_MARGIN_KM for the organizer's GUI report."""
    depth_km = radius_km - last_km
    print(f"depth         : the last row is {depth_km * 1000.0:.4g} m inside")
    if args.stop_depth_gate is None:
        return False
    setting_km, tolerance, allowed = args.stop_depth_gate
    # Half the radius in use, as verify caps it.
    asked_km = min(setting_km, 0.5 * radius_km)
    if depth_km >= asked_km * (1.0 - tolerance):
        return False
    if args.certificate is None:
        # The stop depth is our margin for their GMAT's drift, not an upload criterion.
        if depth_km < RULE_MARGIN_KM:
            print(f"VERDICT: the last row is {depth_km * 1.0e6:.4g} mm inside, under the "
                  f"{RULE_MARGIN_KM * 1.0e6:g} mm that reading the printed digits can move it: "
                  "not shown inside the sphere.")
            return True
        print(f"WARNING: the last row is {depth_km * 1000.0:.4g} m inside, short of the "
              f"config's policy.stop_depth_km = {asked_km * 1000.0:.4g} m that certify flew "
              "ours to: this GMAT ended the script shallower. The rules ask only that the last "
              "row be inside, so this passes.")
        return False
    print(f"WARNING: short of the config's policy.stop_depth_km = {asked_km * 1000.0:.4g} m: "
          "a GMAT reading other data files (space weather, EOP) can fly this as a miss.")
    if allowed:
        print("policy.allow_shallow_stop_submission is set, so this passes as certify did.")
        return False
    print("VERDICT: certify refuses such a hit without policy.allow_shallow_stop_submission.")
    return True


def _against_certificate(code: int, t_team: float | None, last_s: float, why_not: str | None,
                         tolerance: float) -> int:
    """The report must end where certify measured T_team, or it is not the one certified."""
    if code != 0 or t_team is None:
        return code
    if why_not is not None:
        print(f"note: the certificate's T_team {t_team:.6f} s was not compared: {why_not}",
              file=sys.stderr)
        return code
    if not abs(last_s - t_team) <= tolerance:
        print(f"VERDICT: the last row is at {last_s:.6f} s and the certificate's T_team is "
              f"{t_team:.6f} s: this is not the report certify measured.")
        return 1
    print(f"The last row matches the certificate's T_team = {t_team:.6f} s.")
    return code


def _check_console_copy(certificate: dict, script: Path, script_sha256: str,
                        parser: argparse.ArgumentParser, folder: Path) -> None:
    """Hold a certificate whose runs ran the console copy to the script it lists.

    certify ran the copy because GmatConsole refused the script for its plots
    alone, and packaged the copy's report. The copy is rebuilt here from the
    script and proven again; its hash must be the one every exact run ran. The
    script's plot lines, which only the GUI reads, must be those of
    official_original.script beside the certificate, as certify required.
    """
    import re

    from orbitwin.console_copy import assert_plot_lines_unchanged, build_console_copy, proven_copy
    from orbitwin.gmat_parser import GmatScriptDocument

    record = certificate["console_copy"]
    sha = re.compile(r"[0-9a-f]{64}")
    if not (
        isinstance(record, dict)
        and all(isinstance(record.get(key), str) and sha.fullmatch(record[key])
                for key in ("submitted_sha256", "copy_sha256"))
        and isinstance(record.get("removed"), list) and record["removed"]
        and all(isinstance(item, str) for item in record["removed"])
        and isinstance(record.get("deleted_lines"), list)
        and all(type(item) is int and item > 0 for item in record["deleted_lines"])
    ):
        parser.error("the certificate's console_copy is malformed; certify writes it")
    if record["submitted_sha256"] != script_sha256:
        parser.error("the certificate's console_copy is not of the script it certifies")
    runs = certificate.get("exact_runs")
    if not (isinstance(runs, list) and runs and all(
        isinstance(run, dict) and run.get("script_sha256") == record["copy_sha256"]
        for run in runs
    )):
        parser.error("the certificate's exact runs did not all run its console_copy")
    try:
        document = GmatScriptDocument.read(script)
        built = build_console_copy(document)
        proven = None if built is None else proven_copy(document, built, "the console copy")
    except (OrbitWinError, OSError, ValueError) as exc:
        parser.error(f"the console copy of {script.name} cannot be proven again: {exc}")
    if proven is None or (proven.copy_sha256, list(proven.removed), list(proven.deleted_lines)) != (
        record["copy_sha256"], record["removed"], record["deleted_lines"]
    ):
        parser.error(f"the console copy rebuilt from {script.name} is not the one the "
                     "certificate's runs ran")
    official = folder / "official_original.script"
    listed = certificate.get("submission_files")
    original = certificate.get("original_script_sha256")
    if not (official.is_file() and isinstance(listed, dict) and isinstance(original, str)
            and listed.get(official.name) == original == _sha256(official)):
        parser.error(f"{official} is missing or not the official script the certificate "
                     "names; the plot lines cannot be held to it")
    try:
        assert_plot_lines_unchanged(GmatScriptDocument.read(official), document)
    except (OrbitWinError, OSError, ValueError) as exc:
        parser.error(f"{exc}")
    print(f"certificate: GmatConsole refused {script.name} for its plots alone; the report is "
          f"that of its console copy less {', '.join(record['removed'])}, rebuilt and proven "
          "again here", file=sys.stderr)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one_start_shown(args: argparse.Namespace) -> str | None:
    """None when the day's config and script show the craft start together, else why not.

    Without a console the parser reads only epochs written as one text, and
    refuses two: so a parse here is a proof of one start.
    """
    if args.day_config is None or args.script is None:
        return "no --config and --script to show the craft start together"
    from orbitwin.errors import OrbitWinError
    from orbitwin.gmat_parser import parse_gmat_script

    try:
        scenario, _ = parse_gmat_script(args.script, args.day_config)
    except (OrbitWinError, OSError) as exc:
        return f"{args.script} does not show it: {exc}"
    if scenario.alignment is not None and scenario.alignment.offset_s > 0.0:
        return f"in {args.script} the craft start apart"
    return None


def _fill_from_day_files(args: argparse.Namespace, parser: argparse.ArgumentParser) -> float | None:
    """Fill what --config, --certificate and --verification know; a flag given by hand
    must agree with each that gives it, or exit 2.

    Returns the certificate's T_team, to compare with what the report shows.
    """
    import json

    args.same_epoch_claimed = False
    #: (certificate or verification, its candidate) for judge_rules, else None.
    args.recorded = None
    #: Points §4 note 1 deducts from the score printed: judge_rules.
    args.penalty_points = 0.0
    #: What judge_rules read the burns from; empty when nothing gives them.
    args.judged = []
    #: Who gave args.total_dv, for judge_rules' message.
    args.total_dv_from = "--total-dv"
    #: Whether judge_surface judged note 1(b) on the rows; None when the rules do not ask.
    args.judged_surface = None
    #: (stop depth km, tolerance, opt-in) when the config stops a hit inside, else None.
    args.stop_depth_gate = None
    if args.certificate is not None and args.config is None:
        parser.error("--certificate needs --config: the time column and T_max are the day's")
    if args.verification is not None and (args.config is None or args.certificate is not None):
        parser.error("--verification goes with --config and without --certificate, which "
                     "carries the same score block")
    config = None
    if args.config is not None:
        from orbitwin.config import load_config, rulebook_refusal
        from orbitwin.errors import ConfigurationError

        try:
            config = load_config(args.config)
        except (OSError, ConfigurationError) as exc:
            parser.error(f"--config: {exc}")
        # Also a hand --rules other than the config's (_hold_to_config for the rest).
        refusal = rulebook_refusal(config, args.config, args.rules)
        if refusal is not None:
            parser.error(f"--config: {refusal}")
        args.rules = config.policy.rules
        if config.policy.mission_end_policy != "horizon":
            # Here, not at the top: report_parser imports scipy (see the numpy-only test).
            from orbitwin.report_parser import STOP_DEPTH_TOLERANCE

            args.stop_depth_gate = (
                config.policy.stop_depth_km, STOP_DEPTH_TOLERANCE,
                config.policy.allow_shallow_stop_submission,
            )
        # The defaults below, before the time column is named after one of them.
        _hold_to_config("--pursuer", args.pursuer, config.pursuer_name, parser)
        _hold_to_config("--target", args.target, config.target_name, parser)
        _hold_to_config("--horizon-s", args.horizon_s, config.policy.horizon_s, parser)
        args.pursuer = args.pursuer or config.pursuer_name or "SC_Human"
        args.target = args.target or config.target_name or "SC_Alien"
        if args.horizon_s is None:
            args.horizon_s = config.policy.horizon_s
        origin = config.policy.mission_time_origin
        args.earliest_origin = origin == "earliest"
        args.from_origin = config.policy.closest_approach_window == "from_origin"
        if args.time_column is None:
            # The parser refuses craft on two epochs without an origin and a d_min
            # window, so a config with neither flies them on one; the report is
            # held to that below.
            args.same_epoch_claimed = origin is None or (
                origin == "earliest" and config.policy.closest_approach_window is None
            )
        if origin != "earliest":
            craft = args.target if origin == "target" else args.pursuer
            _hold_to_config("--time-column", args.time_column, f"{craft}.ElapsedSecs", parser)
            args.time_column = f"{craft}.ElapsedSecs"
        score = config.score
        if score is not None:
            for name in ("kt", "ct", "kv", "cv", "cr_km"):
                flag = f"--{name.replace('_', '-')}"
                _hold_to_config(flag, getattr(args, name), getattr(score, name), parser)
                setattr(args, name, getattr(score, name))
    args.rules = args.rules or "ranking"
    args.pursuer = args.pursuer or "SC_Human"
    args.target = args.target or "SC_Alien"
    if config is None:
        args.earliest_origin = False
        args.from_origin = False
    args.certified_origin = None
    args.certified_one_start = False
    args.certified_lead_s = None
    args.certificate_body = None
    args.day_config = config
    if args.certificate is None:
        if args.verification is not None:
            assert config is not None
            try:
                verification = json.loads(Path(args.verification).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                parser.error(f"--verification: {exc}")
            if not isinstance(verification, dict):
                parser.error("--verification is not a JSON object; verify writes one")
            _check_verified_script(verification, args, parser)
            args.recorded = ("verification", verification.get("candidate"))
            _check_score_block(verification, config, args, parser, "verification")
            _check_radius(args, parser, "verification")
            if "frozen_rows" in verification:
                _hold_frozen_rows(args, verification["frozen_rows"], parser, "verification")
        elif config is not None:
            print("note: no --certificate or --verification, so the coefficients certify "
                  "priced with were not checked against the config's", file=sys.stderr)
        args.frozen_rows = args.frozen_rows or 0
        return None
    assert config is not None
    path = Path(args.certificate)
    try:
        certificate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(f"--certificate: {exc}")
    if not isinstance(certificate, dict):
        parser.error("--certificate is not a JSON object; certify writes one")
    args.recorded = ("certificate", certificate.get("candidate"))
    args.certificate_body = certificate
    # Only an absent key takes a default: [] or 0 where an object belongs is refused.
    for key in ("clock", "rule_set", "candidate", "final_script_parity"):
        if key in certificate and not isinstance(certificate[key], dict):
            parser.error(f"the certificate's {key} is not an object; certify writes one")
    clock = certificate.get("clock") or {}
    made_under = (certificate.get("rule_set") or {}).get("name")
    for key, value in (("clock.t_team_s", clock.get("t_team_s")),
                       ("rule_set.name", made_under),
                       ("submission_files", certificate.get("submission_files")),
                       ("submission_script_name", certificate.get("submission_script_name"))):
        if value is None:
            parser.error(f"--certificate has no {key}; certify writes it, so this is not one")
    if "mission_time_origin" not in clock:
        parser.error("--certificate has no clock.mission_time_origin; certify writes it")
    if (config.policy.mission_time_origin == "earliest"
            and (clock["mission_time_origin"] != "earliest" or "time_origin_craft" not in clock)):
        parser.error(f"the certificate's clock.mission_time_origin is "
                     f"{clock['mission_time_origin']!r}: it was written before T_team counted "
                     "from the earlier craft's epoch (organizer, 2026-11-04). Certify again "
                     "with this config")
    if clock["mission_time_origin"] != config.policy.mission_time_origin:
        parser.error(f"the certificate's clock.mission_time_origin is "
                     f"{clock['mission_time_origin']!r} and the config's "
                     f"{config.policy.mission_time_origin!r}: not the same day")
    if _rule_set_or_none(made_under) is not interception_rules(args.rules):
        parser.error(f"the certificate was made under the {made_under} rules, not {args.rules}")
    # A rulebook revised under one name scores differently: 2026/11/04 changed the distance term.
    source = interception_rules(args.rules).source
    made_from = certificate["rule_set"].get("sha256")
    if made_from != source.sha256:
        parser.error(
            f"the certificate was made under {certificate['rule_set'].get('document')!r} "
            f"({certificate['rule_set'].get('header_date')}, SHA-256 {made_from}), and the "
            f"{args.rules} rules are now {source.document!r} ({source.header_date}, SHA-256 "
            f"{source.sha256}): certify again under the current rules"
        )
    _check_score_block(certificate, config, args, parser, "certificate")
    _check_radius(args, parser, "certificate")
    if args.earliest_origin:
        craft = clock.get("time_origin_craft")
        if craft not in (args.pursuer, args.target):
            parser.error(f"the certificate's clock.time_origin_craft is {craft!r}, not "
                         f"{args.pursuer} or {args.target}; certify writes it")
        if "alignment_offset_s" not in clock:
            parser.error("--certificate has no clock.alignment_offset_s; certify writes it")
        recorded = clock.get("head_start_burns")
        if (recorded is not None and config is not None
                and recorded != config.policy.head_start_burns):
            parser.error(f"the certificate was certified with policy.head_start_burns "
                         f"{recorded}, and --config has {config.policy.head_start_burns}: "
                         "certify again under the day's config")
        if not clock["alignment_offset_s"]:
            # One start: either craft's ElapsedSecs is the clock.
            args.certified_one_start = True
        else:
            args.certified_origin = f"{craft}.ElapsedSecs"
            if craft == args.pursuer:
                # The head start's length: a head-start burn's time less it is the solver's.
                args.certified_lead_s = float(clock["alignment_offset_s"])
            if args.time_column is not None and args.time_column != args.certified_origin:
                parser.error(f"--time-column is {args.time_column} and the certificate's "
                             f"clock.time_origin_craft gives {args.certified_origin}: leave "
                             "--time-column out, the certificate gives it")
    # The report and the script must be the bytes certify wrote, not merely files beside it.
    # By content: certify lists files by their path in the submission folder.
    listed = certificate["submission_files"]
    if not (isinstance(listed, dict) and all(isinstance(v, str) for v in listed.values())):
        parser.error("the certificate's submission_files is malformed; certify writes it")
    hashes = set(listed.values())
    report = Path(args.report)
    if not report.is_file():
        parser.error(f"{report}: no such report")
    if _sha256(report) not in hashes:
        parser.error(f"{report.name} is not a report this certificate lists (SHA-256 differs)")
    listed_name = certificate.get("submission_script_name")
    if not (isinstance(listed_name, str) and Path(listed_name).name == listed_name):
        parser.error(f"the certificate's submission_script_name {listed_name!r} is not a file "
                     "name")
    name = listed_name
    if name not in listed:
        parser.error(f"the certificate's submission_script_name {name!r} is not a file it lists")
    # The script's own entry, not any file listed (official_original.script is one).
    script_sha256 = listed[name]
    final = certificate.get("final_script_sha256")
    if final is not None and final != script_sha256:
        parser.error("the certificate's final_script_sha256 is not its submission script's")
    fields = (args.delimiter, args.write_headers, args.fixed_width, args.column_width,
              args.columns)
    flagged = any(value is not None for value in fields)
    if args.script is None:
        # The script beside the certificate, held to its entry even when format flags
        # (--delimiter, ...) mean it is not read.
        beside = path.parent / name
        if not beside.is_file():
            # main() takes --script or the format flags, never both.
            instead = (" instead of the format flags, which it gives" if flagged else "")
            parser.error(f"{beside} is missing: put the certificate's script back beside it, "
                         f"or pass a copy of it with --script{instead}")
        if _sha256(beside) != script_sha256:
            parser.error(f"{beside} differs from the script this certificate lists")
        if not flagged:
            args.script = str(beside)
    elif not Path(args.script).is_file():
        parser.error(f"--script {args.script}: no such file")
    elif _sha256(Path(args.script)) != script_sha256:
        parser.error(f"{args.script} differs from the script this certificate lists")
    args.console_copied = "console_copy" in certificate
    if args.console_copied:
        _check_console_copy(certificate, Path(args.script) if args.script else path.parent / name,
                            script_sha256, parser, path.parent)
    else:
        # Without a console copy every exact run ran the script itself: a certificate
        # stripped of its console_copy says otherwise and is refused.
        runs = certificate.get("exact_runs")
        if not (isinstance(runs, list) and runs and all(
            isinstance(run, dict) and run.get("script_sha256") == script_sha256 for run in runs
        )):
            parser.error("the certificate's exact runs did not all run the script it lists, "
                         "and it records no console_copy they ran instead")
    _hold_frozen_rows(args, certificate.get("frozen_rows", 0), parser, "certificate")
    # T_max on the day's clock is the solver's plus the time origin's offset.
    metadata = (certificate.get("candidate") or {}).get("metadata", {})
    if not isinstance(metadata, dict):
        parser.error("the certificate's candidate.metadata is not an object; certify writes one")
    offset_s = _finite(clock.get("time_origin_offset_s", 0.0), "clock.time_origin_offset_s",
                       parser)
    t_team_s = _finite(clock["t_team_s"], "clock.t_team_s", parser)
    solver_horizon = metadata.get("horizon_s")
    if solver_horizon is not None and args.horizon_s is not None:
        day_horizon = _finite(solver_horizon, "candidate.metadata.horizon_s", parser) + offset_s
        if not abs(day_horizon - args.horizon_s) <= 1.0e-3:
            parser.error(f"T_max is {args.horizon_s:g} s here and {day_horizon:g} s in the "
                         "certificate: the config is not the one certify ran")
    total_km_s = (certificate.get("candidate") or {}).get("total_dv_km_s")
    if total_km_s is not None and config.score is not None:
        unit = config.score.delta_v_unit
        scale = 1000.0 if unit == "m/s" else 1.0
        total = _finite(total_km_s, "candidate.total_dv_km_s", parser) * scale
        if args.total_dv is not None and not (
                abs(args.total_dv - total) <= TOTAL_DV_TOLERANCE_KM_S * scale):
            parser.error(f"--total-dv is {args.total_dv!r} {unit} and the certificate's "
                         f"candidate.total_dv_km_s is {total:.9g} {unit}: leave --total-dv "
                         "out, the certificate gives it")
        if args.total_dv is None:
            args.total_dv_from = "the certificate's candidate.total_dv_km_s"
        args.total_dv = total
    replayed = (certificate.get("final_script_parity") or {}).get("replayed_score")
    if replayed is not None:
        replayed_score = _finite(replayed, "final_script_parity.replayed_score", parser)
        print(f"certificate: replayed score {replayed_score:.4f}, T_team {t_team_s:.6f} s",
              file=sys.stderr)
    return t_team_s


_COEFFICIENT_FLAGS = ("kt", "ct", "kv", "cv", "cr_km")


def _hold_to_config(flag: str, given: object, configured: object,
                    parser: argparse.ArgumentParser) -> None:
    """A flag given by hand must be what the config says, where it says anything: exit 2."""
    if given is None or configured is None:
        return
    if isinstance(given, float) and isinstance(configured, int | float):
        same = math.isclose(given, float(configured), rel_tol=1.0e-12, abs_tol=0.0)
    else:
        same = given == configured
    if not same:
        parser.error(f"{flag} is {given!r} and the config's is {configured!r}: leave {flag} "
                     "out, the config gives it")


def _hold_frozen_rows(args: argparse.Namespace, frozen: object,
                      parser: argparse.ArgumentParser, kind: str) -> None:
    """--frozen-rows from the certificate or verification; a hand value that differs exits 2."""
    if isinstance(frozen, bool) or not isinstance(frozen, int) or frozen < 0:
        parser.error(f"the {kind}'s frozen_rows is {frozen!r}, not a row count")
    if args.frozen_rows is not None and args.frozen_rows != frozen:
        parser.error(f"--frozen-rows is {args.frozen_rows} and the {kind}'s frozen_rows is "
                     f"{frozen}: leave --frozen-rows out, the {kind} gives it")
    args.frozen_rows = frozen


def _check_radius(args: argparse.Namespace, parser: argparse.ArgumentParser, kind: str) -> None:
    """A --radius-km other than the rules' sphere, which the certificate or verification was
    judged on, exits 2."""
    sphere_km = interception_rules(args.rules).intercept_radius_km
    if args.radius_km is not None and args.radius_km != sphere_km:
        parser.error(f"--radius-km is {args.radius_km:g} and the {args.rules} rules' sphere, "
                     f"which the {kind} was judged on, is {sphere_km:g} km: leave it out")


def _check_verified_script(verification: dict[str, object], args: argparse.Namespace,
                           parser: argparse.ArgumentParser) -> None:
    """Exit 2 unless --verification is the one GMAT flew the --script file for: one from
    another run (step 4a's, an earlier .refused) passed on its score block alone."""
    if args.script is None:
        parser.error("--verification needs --script, the script it verified")
    if "candidate_script_sha256" not in verification:
        parser.error("--verification has no candidate_script_sha256: it was written before "
                     "verify recorded the script it flew. Certify again")
    flown = verification["candidate_script_sha256"]
    script = Path(args.script)
    if not script.is_file():
        parser.error(f"--script {script}: no such file")
    if _sha256(script) != flown:
        parser.error(f"--verification verified a script with SHA-256 {flown}, and --script "
                     f"{script} has {_sha256(script)}: take the verification.json the "
                     "refusal message names, of the run that wrote this script")


def _check_score_block(record: dict[str, object], config: OrbitWinConfig,
                       args: argparse.Namespace, parser: argparse.ArgumentParser,
                       kind: str) -> None:
    """Exit 2 when the certificate or verification was priced under other coefficients
    than this check's: the config's, or those given by hand.

    R2: certified at C_r 300 and checked at 3000, it printed 23.6716 and 31.5363 and passed.
    A side without coefficients prices nothing, so nothing disagrees.
    """
    from orbitwin.score import SCORE_BLOCK_FIELDS, score_stamp, score_stamp_difference

    made = "certified" if kind == "certificate" else "verified"
    if "score_block" not in record:
        parser.error(f"--{kind} has no score_block: it was written before orbitwin recorded "
                     "the coefficients it priced with. Certify again under the announced "
                     "coefficients")
    recorded = record["score_block"]
    if config.score is not None:
        checking: dict[str, object] = {
            **score_stamp(config.score), **{name: getattr(args, name) for name in _COEFFICIENT_FLAGS}
        }
        names: tuple[str, ...] = SCORE_BLOCK_FIELDS
    else:
        # An unscored config: only what was given by hand prices anything.
        checking = {name: getattr(args, name) for name in _COEFFICIENT_FLAGS
                    if getattr(args, name) is not None}
        names = tuple(checking)
    if not names:
        return
    if recorded is None:
        print(f"note: the {kind} was made without coefficients and carries no score; the "
              "score below is this check's alone", file=sys.stderr)
        return
    difference = score_stamp_difference(recorded, checking, names)
    if difference is not None:
        parser.error(f"{made} under {difference[0]}, checking under {difference[1]}: "
                     "certify again under the announced coefficients")


def _finite(value: object, key: str, parser: argparse.ArgumentParser) -> float:
    """A number the certificate holds, or exit 2: JSON's NaN and Infinity compare as
    nothing, and an integer past a double's range does not convert."""
    from orbitwin.config import finite_number

    number = finite_number(value)
    if number is None:
        shown = str(value) if isinstance(value, int | float) else repr(value)
        parser.error(f"the certificate's {key} is {shown[:40]}, not a finite number")
    return number


#: GMAT writes the last row at the Propagate goal: 6711.667000001762 for R1's
#: 6711.667, by an ElapsedSecs or an ElapsedDays goal, and 5791.380999865942 for R2's.
HORIZON_TOLERANCE_S = 1.0e-3


def _horizon_tolerance(horizon_s: float | None, precision: int | None) -> float:
    """1 ms, or the last digit the ReportFile's Precision prints at T_max if coarser."""
    if horizon_s is None or precision is None:
        return HORIZON_TOLERANCE_S
    digit = 10.0 ** (math.floor(math.log10(horizon_s)) - precision + 1)
    return max(HORIZON_TOLERANCE_S, digit)


def _logistic(points: float, exponent: float) -> float:
    if exponent > 700.0:
        return 0.0
    if exponent < -700.0:
        return points
    return points / (1.0 + math.exp(exponent))


def distance_term(distance_km: float, rules: InterceptionRules, cr_km: float | None) -> float:
    """Δr_min = max(d, 5): 50 e^{-(Δr_min-5)/100}, or 100/(1+e^{((Δr_min-5)/C_r)^2}) since
    the ranking rules' 2026/11/04 revision. Written here, not imported from score.py."""
    excess = max(distance_km, rules.intercept_radius_km) - rules.intercept_radius_km
    match rules.distance_term:
        case DistanceTerm.EXPONENTIAL:
            return 50.0 * math.exp(-excess / 100.0)
        case DistanceTerm.LOGISTIC:
            if cr_km is None:
                raise ValueError("the ranking rules' distance term needs C_r")
            u = excess / cr_km
            return _logistic(100.0, u * u)
        case _:
            assert_never(rules.distance_term)


def _miss_verdict(args: argparse.Namespace, times: np.ndarray, distance: np.ndarray,
                  closest: int, last: int, unit: str, why_not: str | None,
                  tolerance: float, frozen_doubt: bool,
                  alignment_min_km: float | None = None) -> int:
    """No row inside: a failure under the preliminary, an answer under the ranking rules."""
    rules = interception_rules(args.rules)
    # The preliminary submits interceptions only (constraints.interception_required).
    if not rules.miss_ranked or not rules.miss_scored:
        print("VERDICT: no row is inside the sphere. This report does not show an interception.")
        return 1
    print("No row is inside the sphere: this report is a miss.")
    print(f"Under the {args.rules} rules a miss scores on its closest approach "
          f"({distance[closest]:.6f} km on the rows) with T_team = T_max.")
    if distance[closest] <= rules.intercept_radius_km:
        print(f"VERDICT: the closest row is inside the rules' {rules.intercept_radius_km:g} km "
              "sphere; only --radius-km makes this a miss.")
        return 1
    if frozen_doubt:
        print("VERDICT: unchecked. --frozen-rows does not match the rows an epoch alignment "
              "held still; take it from certification.json.")
        return 1
    if args.horizon_s is None:
        print("VERDICT: unchecked. Pass --horizon-s with the announced T_max to check that the "
              "last row is there.")
        return 1
    if why_not is not None:
        print(f"VERDICT: unchecked. The last row cannot be checked against T_max: {why_not}")
        return 1
    off = float(times[last]) - args.horizon_s
    if not abs(off) <= tolerance:
        print(f"VERDICT: the LAST row is at {times[last]:.6f}{unit}, {off:+.6f} s from T_max = "
              f"{args.horizon_s:.6f} s. A miss must run to T_max.")
        return 1
    print(f"VERDICT: the LAST row is at T_max = {args.horizon_s:.6f} s: a scored miss.")
    if None in (args.kt, args.ct, args.kv, args.cv, args.total_dv):
        return 0
    if rules.distance_term.takes_cr and args.cr_km is None:
        print()
        print("  not priced: no C_r. Pass --cr-km, or a --config that has score.cr_km.")
        return 0
    d_min_km = float(distance[closest])
    if alignment_min_km is not None:
        print(f"  d_min under from_origin counts the alignment's rows too: "
              f"{alignment_min_km:.6f} km there")
        d_min_km = min(d_min_km, alignment_min_km)
    distance_points = distance_term(d_min_km, rules, args.cr_km)
    args.cr_used = True
    time_term = _logistic(25.0, args.kt * (args.horizon_s - args.ct))
    dv_term = _logistic(25.0, args.kv * (args.total_dv - args.cv))
    print()
    total = distance_points + time_term + dv_term
    deducted = f" - {args.penalty_points:g}" if args.penalty_points else ""
    print(f"  scored miss: distance {distance_points:.4f} + time {time_term:.4f} + "
          f"delta-v {dv_term:.4f}{deducted} = {_penalized(total, args):.4f}")
    if args.cr_km is not None:
        print(f"  (C_r = {args.cr_km:g} km)")
    _said_judged(args)
    return 0 if _hold_to_certified_score(args, _penalized(total, args)) else 1


if __name__ == "__main__":
    sys.exit(main())
