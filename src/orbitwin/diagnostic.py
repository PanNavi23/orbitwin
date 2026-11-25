from __future__ import annotations

import re
from dataclasses import dataclass

from .burn_rows import BURN_REPORT_QUANTITIES, maneuvers_flown
from .errors import PolicyError
from .gmat_parser import (
    _BEGIN_MISSION_RE,
    _MANEUVER_RE,
    ALLOWED_RESOURCE_TYPES,
    PLOT_TYPES,
    GmatScriptDocument,
    _code_part,
    created_types,
    names_with_file_fields,
    plot_command,
    referenced_resources,
)
from .models import Scenario

#: Subscriber types that draw. GmatConsole cannot create an OpenFrames object
#: at all -- the interpreter refuses the whole script -- and on the organizer's
#: own workshop template a GroundTrack leaves the run reporting success while
#: writing no report at all. None of them contributes to a measurement, and
#: this copy is never submitted, so the private one keeps its ReportFiles and
#: nothing that draws.
GUI_SUBSCRIBER_TYPES = PLOT_TYPES
_CREATE_RE = re.compile(
    r"^\s*(?:GMAT\s+)?Create\s+(?P<type>[A-Za-z_]\w*)\s+(?P<names>[^;]+);", re.IGNORECASE
)
NEUTRALISED_PREFIX = "% [orbitwin diagnostic] "
#: Every double printed so that float() gives back its bits.
DIAGNOSTIC_PRECISION = 17


#: Written only by the Report commands around each Maneuver.
BURN_RESOURCE_NAME = "OrbitWinBurnCheck"


@dataclass(frozen=True, slots=True)
class DiagnosticScript:
    script_bytes: bytes
    report_filename: str
    resource_name: str
    #: Plot objects commented out so the run can be measured at all.
    neutralised: tuple[str, ...] = ()
    #: The pursuer's epoch and state just before and just after each Maneuver,
    #: or None when not asked for or nothing is maneuvered.
    burn_report_filename: str | None = None


def _neutralise_plots(
    document: GmatScriptDocument,
    types: frozenset[str] = GUI_SUBSCRIBER_TYPES,
    keep: frozenset[str] = frozenset(),
) -> tuple[list[str], list[str]]:
    """Comment out every plot object, keeping the ReportFiles.

    Measured against GMAT R2026a. A script declaring an OpenFrames object is
    refused outright -- "The OpenFramesInterface is an unknown object type" --
    because console skips GUI plugins and the library is not shipped with the
    Linux build either. That one is unambiguous.

    The organizer's workshop template shows the quieter half. With OpenFrames
    commented out it runs to completion and prints the success marker, and
    writes no report; commenting the GroundTrack out as well produces the
    report. The same GroundTrack block, verbatim, does not reproduce that in a
    minimal script, so what interacts with what here is not established -- only
    that on their own template a plot subscriber is the difference between a
    measurement and a run that claims success and measures nothing.

    Neutralising all of them is the rule that does not depend on knowing which.
    This copy is private and never submitted, the submitted bytes keep every
    plot the organizer wrote, and none of these objects touches propagation.
    """
    names: list[str] = []
    for line in document.lines:
        match = _CREATE_RE.match(line)
        if match and match.group("type") in types:
            names.extend(
                part
                for part in re.split(r"[,\s]+", match.group("names"))
                if part and part not in keep
            )
    if not names:
        return list(document.lines), []
    touched = re.compile(
        r"^\s*(?:GMAT\s+)?(?:"
        + "|".join(re.escape(name) for name in names)
        + r")\s*\."
    )
    out: list[str] = []
    mission = False
    for line in document.lines:
        code = _code_part(line.rstrip("\r\n"))
        if mission:
            toggled = plot_command(code, document)
            if toggled and not set(toggled) <= set(names):
                raise PolicyError(
                    f"{line.strip()!r} toggles {sorted(set(toggled) - set(names))}, kept, beside "
                    f"{sorted(set(toggled) & set(names))}, commented out; it cannot be either"
                )
            out.append(NEUTRALISED_PREFIX + line if toggled else line)
            continue
        mission = bool(_BEGIN_MISSION_RE.match(code))
        match = _CREATE_RE.match(line)
        # A Create naming one kept object stays whole; the dropped ones' fields go.
        creates_plot = bool(
            match
            and match.group("type") in types
            and not set(re.split(r"[,\s]+", match.group("names").strip())) & keep
        )
        if creates_plot or touched.match(line):
            out.append(NEUTRALISED_PREFIX + line)
        else:
            out.append(line)
    return out, names


def add_diagnostic_report(
    document: GmatScriptDocument,
    scenario: Scenario,
    report_filename: str = "orbitwin_diagnostic.txt",
    *,
    check_burns: bool = False,
) -> DiagnosticScript:
    """The private sibling: a diagnostic ReportFile and no plots.

    With ``check_burns`` it also reports the pursuer on either side of each
    Maneuver, which the diagnostic report cannot show: R2026a writes no row
    after a burn (burn_rows), and verify reads the impulse GMAT flew from these.
    """
    resource_name = "OrbitWinDiagnostic"
    for reserved in (resource_name, *([BURN_RESOURCE_NAME] if check_burns else [])):
        if reserved in document.text:
            raise PolicyError(f"Official script already uses reserved resource name {reserved}")
    # The parser's spelling, which allows a comment or a space before the semicolon.
    begin_index = next(
        (index for index, line in enumerate(document.lines) if _BEGIN_MISSION_RE.match(line)),
        None,
    )
    if begin_index is None:
        raise PolicyError("Cannot insert diagnostic ReportFile without BeginMissionSequence")
    newline = document.encoding.newline
    p = scenario.pursuer.name
    t = scenario.target.name
    # A1 epochs hold the craft together after an alignment, where ElapsedSecs
    # count from two epochs; Precision 17 makes them bit-exact (§4).
    parameters = [
        f"{p}.ElapsedSecs",
        f"{t}.ElapsedSecs",
        f"{p}.A1ModJulian",
        f"{t}.A1ModJulian",
        f"{p}.EarthMJ2000Eq.X",
        f"{p}.EarthMJ2000Eq.Y",
        f"{p}.EarthMJ2000Eq.Z",
        f"{p}.EarthMJ2000Eq.VX",
        f"{p}.EarthMJ2000Eq.VY",
        f"{p}.EarthMJ2000Eq.VZ",
        f"{t}.EarthMJ2000Eq.X",
        f"{t}.EarthMJ2000Eq.Y",
        f"{t}.EarthMJ2000Eq.Z",
        f"{t}.EarthMJ2000Eq.VX",
        f"{t}.EarthMJ2000Eq.VY",
        f"{t}.EarthMJ2000Eq.VZ",
        # Geodetic, as the organizer judges a crossing (§5); a cross-check.
        f"{p}.Earth.Altitude",
    ]
    block = [
        f"% OrbitWin internal diagnostic subscriber; never submit this sibling.{newline}",
        f"Create ReportFile {resource_name};{newline}",
        f"{resource_name}.Filename = '{report_filename}';{newline}",
        f"{resource_name}.Precision = {DIAGNOSTIC_PRECISION};{newline}",
        f"{resource_name}.Add = {{{', '.join(parameters)}}};{newline}",
        f"{resource_name}.WriteHeaders = true;{newline}",
        f"{resource_name}.LeftJustify = On;{newline}",
        f"{resource_name}.ZeroFill = Off;{newline}",
        f"{resource_name}.FixedWidth = true;{newline}",
        f"{resource_name}.ColumnWidth = 26;{newline}",
        f"{resource_name}.WriteReport = true;{newline}",
        f"{resource_name}.AppendToExistingFile = false;{newline}",
        newline,
    ]
    # And what a private run never needs: an EphemerisFile, or a type only
    # policy.extra_resource_types admitted.
    # A craft's hardware stays, or GMAT refuses the craft; so does an admitted
    # resource that names a data file, so every run behind a certificate reads it.
    types = created_types(document)
    others = {kind for kind in types.values() if kind not in ALLOWED_RESOURCE_TYPES}
    keep = referenced_resources(document) | {
        name for name in names_with_file_fields(document) if types.get(name) in others
    }
    lines, neutralised = _neutralise_plots(
        document, GUI_SUBSCRIBER_TYPES | {"EphemerisFile"} | others, frozenset(keep)
    )
    # The Maneuvers verify pairs the rows with, so the two cannot disagree.
    maneuvers = maneuvers_flown(scenario)
    burn_report: str | None = None
    if check_burns and maneuvers:
        stem, dot, suffix = report_filename.rpartition(".")
        burn_report = f"{stem}_burns{dot}{suffix}" if dot else f"{report_filename}_burns"
        if burn_report in {report.filename for report in scenario.report_files}:
            raise PolicyError(f"Official script already writes the reserved report {burn_report}")
        state = " ".join(f"{p}.{quantity}" for quantity in BURN_REPORT_QUANTITIES)
        report = f"Report {BURN_RESOURCE_NAME} {state};{newline}"
        # From the last, so the earlier indices still hold; all are after begin_index.
        for maneuver in reversed(maneuvers):
            index = maneuver.line_index
            line = lines[index] if index is not None else ""
            match = _MANEUVER_RE.match(line.rstrip("\r\n"))
            if index is None or match is None or match.group("burn") != maneuver.name:
                raise PolicyError(
                    f"The script has no `Maneuver {maneuver.name}` where {scenario.source_path} "
                    f"put it (line {index})"
                )
            if not line.endswith(("\n", "\r")):
                line += newline
            lines[index : index + 1] = [report, line, report]
        block += [
            f"Create ReportFile {BURN_RESOURCE_NAME};{newline}",
            f"{BURN_RESOURCE_NAME}.Filename = '{burn_report}';{newline}",
            f"{BURN_RESOURCE_NAME}.Precision = {DIAGNOSTIC_PRECISION};{newline}",
            f"{BURN_RESOURCE_NAME}.WriteHeaders = false;{newline}",
        ]
    lines[begin_index:begin_index] = block
    encoded = "".join(lines).encode("utf-8")
    if document.encoding.has_utf8_bom:
        encoded = b"\xef\xbb\xbf" + encoded
    return DiagnosticScript(
        encoded, report_filename, resource_name, tuple(neutralised), burn_report
    )
