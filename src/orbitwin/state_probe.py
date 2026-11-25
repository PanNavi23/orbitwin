"""Initial epochs and inertial states as GMAT reads them.

A private copy of the organizer's script keeps every resource, neutralises the
plots (GmatConsole refuses OpenFrames) and replaces the mission sequence with
one ``Report`` of each craft before anything propagates. GMAT then converts
whatever the script gives -- EarthFixed, SphericalAZFPA, any DateFormat -- to
EarthMJ2000Eq Cartesian and absolute epochs. We do not convert EarthFixed
ourselves: that needs GMAT's Earth orientation model and its EOP data.

See docs/ASYNC_START_AND_INSERTION.md §1.
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import OrbitWinConfig
from .diagnostic import _neutralise_plots
from .errors import GmatRunError, ParseError
from .gmat_parser import (
    _ASSIGN_RE,
    _BEGIN_MISSION_RE,
    _CREATE_RE,
    PROBE_KEPT_TYPES,
    AlignedStatesNeeded,
    GmatScriptDocument,
    _code_part,
    assert_resources_supported,
    created_types,
    parse_gmat_script,
    referenced_resources,
    split_create_names,
)
from .gmat_runner import run_gmat, said_by_gmat
from .models import (
    AlignedStates,
    EpochAlignment,
    GmatRunResult,
    ProbedCraft,
    ProbedEpoch,
    Scenario,
    StateProbe,
    StateVector,
    spacecraft_binding_sha256,
)
from .report_parser import EPOCH_TOLERANCE_S, parse_numeric_report

PROBE_RESOURCE_NAME = "OrbitWinStateProbe"
PROBE_REPORT_FILENAME = "orbitwin_state_probe.txt"
ALIGNMENT_RESOURCE_NAME = "OrbitWinAlignmentProbe"
ALIGNMENT_REPORT_FILENAME = "orbitwin_alignment_probe.txt"
#: Precision 17 prints every double so that float() gives back the same bits.
PROBE_PRECISION = 17
_EPOCH_SCALES = ("A1", "UTC", "TAI", "TT", "TDB")
_STATE_AXES = ("X", "Y", "Z", "VX", "VY", "VZ")


def _resource_section(
    document: GmatScriptDocument,
) -> tuple[int, list[str], dict[str, dict[str, str]], dict[str, str]]:
    """BeginMissionSequence's line, the declared craft, every object's assignments
    before it and each created name's lower-cased type."""
    begin = next(
        (index for index, line in enumerate(document.lines)
         if _BEGIN_MISSION_RE.match(_code_part(line))),
        None,
    )
    if begin is None:
        raise ParseError("BeginMissionSequence was not found")
    craft: list[str] = []
    assignments: dict[str, dict[str, str]] = {}
    created: dict[str, str] = {}
    for line in document.lines[:begin]:
        code = _code_part(line)
        create = _CREATE_RE.match(code)
        if create:
            # GMAT keeps the first Create of a name and ignores a repeat.
            for name in split_create_names(create.group("names")):
                if name not in created:
                    created[name] = create.group("type").lower()
                    if created[name] == "spacecraft":
                        craft.append(name)
            continue
        assign = _ASSIGN_RE.match(code)
        if assign:
            assignments.setdefault(assign.group("object"), {})[assign.group("field")] = (
                assign.group("value")
            )
    if not craft:
        raise ParseError("The script declares no Spacecraft to probe")
    return begin, craft, assignments, created


def probe_parameters(name: str) -> list[str]:
    return [f"{name}.{scale}ModJulian" for scale in _EPOCH_SCALES] + [
        f"{name}.EarthMJ2000Eq.{axis}" for axis in _STATE_AXES
    ]


def build_probe_script(
    document: GmatScriptDocument, extra_types: Iterable[str] = ()
) -> tuple[bytes, tuple[str, ...]]:
    """The private probe copy and the craft it reports, in declaration order."""
    # Before GmatConsole sees it: an #Include'd ReportFile deleted a file outside the run.
    assert_resources_supported(document, extra_types)
    if PROBE_RESOURCE_NAME in document.text:
        raise ParseError(f"The script already uses the reserved name {PROBE_RESOURCE_NAME}")
    begin, craft, _, _ = _resource_section(document)
    newline = document.encoding.newline
    # Only what places a craft is kept; GMAT refuses a craft naming a dropped
    # tank ("Nonexistent Hardware"). A kept ReportFile deleted a file outside the run.
    dropped = frozenset(created_types(document).values()) - PROBE_KEPT_TYPES
    lines, _ = _neutralise_plots(document, dropped, frozenset(referenced_resources(document)))
    resource = PROBE_RESOURCE_NAME
    block = [
        f"% OrbitWin state probe; private, never submitted.{newline}",
        f"Create ReportFile {resource};{newline}",
        f"{resource}.Filename = '{PROBE_REPORT_FILENAME}';{newline}",
        f"{resource}.Precision = {PROBE_PRECISION};{newline}",
        f"{resource}.WriteHeaders = true;{newline}",
        f"{resource}.AppendToExistingFile = false;{newline}",
        newline,
    ]
    parameters = [parameter for name in craft for parameter in probe_parameters(name)]
    mission = [
        lines[begin],
        f"Report {resource} {' '.join(parameters)};{newline}",
    ]
    text = "".join(lines[:begin] + block + mission)
    return text.encode("utf-8"), tuple(craft)


def parse_probe_report(
    path: str | Path,
    craft: tuple[str, ...],
    fields: Mapping[str, Mapping[str, str]],
    created: Mapping[str, str],
) -> tuple[ProbedCraft, ...]:
    report = parse_numeric_report(path)
    if report.data.shape[0] != 1:
        raise GmatRunError(f"State probe {path} has {report.data.shape[0]} rows, not one")
    index = {header: column for column, header in enumerate(report.headers)}
    probed: list[ProbedCraft] = []
    for name in craft:
        wanted = probe_parameters(name)
        missing = [parameter for parameter in wanted if parameter not in index]
        if missing:
            raise GmatRunError(f"State probe {path} lacks {missing}")
        # parse_numeric_report has already refused NaN and infinity.
        values = [float(report.data[0, index[parameter]]) for parameter in wanted]
        probed.append(
            ProbedCraft(
                name=name,
                epoch=ProbedEpoch(*values[:5]),
                state=StateVector(
                    (values[5], values[6], values[7]), (values[8], values[9], values[10])
                ),
                binding_sha256=spacecraft_binding_sha256(name, fields, created),
            )
        )
    return tuple(probed)


def probe_initial_states(
    script: str | Path,
    gmat_console: str | Path,
    startup_file: str | Path,
    workdir: str | Path,
    *,
    timeout_s: float,
    extra_resource_types: Iterable[str] = (),
) -> StateProbe:
    """Run the probe copy of ``script`` in its own directory and read what GMAT reports."""
    document = GmatScriptDocument.read(script)
    probe_bytes, craft = build_probe_script(document, extra_resource_types)
    _, _, assignments, created = _resource_section(document)
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=False)
    probe_script = root / "state_probe.script"
    probe_script.write_bytes(probe_bytes)
    result = run_gmat(
        gmat_console,
        probe_script,
        root / "run",
        [PROBE_REPORT_FILENAME],
        startup_file=startup_file,
        timeout_s=timeout_s,
        template_dir=Path(script).resolve().parent,
    )
    if not result.success:
        # The run directory is temporary; GMAT's own words go with the exception.
        said = said_by_gmat(result)
        raise GmatRunError(f"The state probe failed: {result.error}{said}")
    return StateProbe(
        craft=parse_probe_report(
            result.reports[PROBE_REPORT_FILENAME], craft, assignments, created
        ),
        run=result,
    )


def build_alignment_probe_script(
    document: GmatScriptDocument, alignment: EpochAlignment, extra_types: Iterable[str] = ()
) -> tuple[bytes, tuple[str, ...]]:
    """A private copy that flies only the alignment, reporting both craft every step.

    It keeps the force models and propagators the alignment needs; the plots,
    the official ReportFiles and the burns are commented out.
    """
    assert_resources_supported(document, extra_types)
    if ALIGNMENT_RESOURCE_NAME in document.text:
        raise ParseError(f"The script already uses the reserved name {ALIGNMENT_RESOURCE_NAME}")
    begin, craft, _, _ = _resource_section(document)
    newline = document.encoding.newline
    kept = PROBE_KEPT_TYPES | {"ForceModel", "Propagator"}
    dropped = frozenset(created_types(document).values()) - kept
    lines, _ = _neutralise_plots(document, dropped, frozenset(referenced_resources(document)))
    resource = ALIGNMENT_RESOURCE_NAME
    parameters = [
        f"{name}.{quantity}"
        for name in craft
        for quantity in ("A1ModJulian", *(f"EarthMJ2000Eq.{axis}" for axis in _STATE_AXES))
    ]
    block = [
        f"% OrbitWin alignment probe; private, never submitted.{newline}",
        f"Create ReportFile {resource};{newline}",
        f"{resource}.Filename = '{ALIGNMENT_REPORT_FILENAME}';{newline}",
        f"{resource}.Precision = {PROBE_PRECISION};{newline}",
        f"{resource}.WriteHeaders = true;{newline}",
        f"{resource}.AppendToExistingFile = false;{newline}",
        f"{resource}.Add = {{{', '.join(parameters)}}};{newline}",
        newline,
    ]
    text = alignment.text if alignment.text.endswith(("\n", "\r")) else alignment.text + newline
    mission = [lines[begin], text]
    return "".join(lines[:begin] + block + mission).encode("utf-8"), tuple(craft)


def probe_alignment(
    script: str | Path,
    alignment: EpochAlignment,
    binding_sha256: str,
    gmat_console: str | Path,
    startup_file: str | Path,
    workdir: str | Path,
    *,
    timeout_s: float,
    extra_resource_types: Iterable[str] = (),
) -> AlignedStates:
    """Fly the alignment alone in GMAT and read both craft at its end, t = 0."""
    document = GmatScriptDocument.read(script)
    probe_bytes, _ = build_alignment_probe_script(document, alignment, extra_resource_types)
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=False)
    probe_script = root / "alignment_probe.script"
    probe_script.write_bytes(probe_bytes)
    result = run_gmat(
        gmat_console, probe_script, root / "run", [ALIGNMENT_REPORT_FILENAME],
        startup_file=startup_file, timeout_s=timeout_s,
        template_dir=Path(script).resolve().parent,
    )
    if not result.success:
        said = said_by_gmat(result)
        raise GmatRunError(f"The alignment probe failed: {result.error}{said}")
    report = parse_numeric_report(result.reports[ALIGNMENT_REPORT_FILENAME])

    def track(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a1 = report.column(f"{name}.A1ModJulian")
        r = np.column_stack([report.column(f"{name}.EarthMJ2000Eq.{axis}") for axis in "XYZ"])
        v = np.column_stack(
            [report.column(f"{name}.EarthMJ2000Eq.{axis}") for axis in ("VX", "VY", "VZ")]
        )
        return a1, r, v

    early_a1, early_r, early_v = track(alignment.early)
    late_a1, late_r, late_v = track(alignment.late)
    gap_s = abs(float(early_a1[-1] - late_a1[-1])) * 86400.0
    if gap_s > EPOCH_TOLERANCE_S:
        raise GmatRunError(
            f"The alignment probe ends {gap_s:.9g} s from {alignment.late}'s epoch; the "
            "alignment does not put the two on one"
        )
    return AlignedStates(
        states={
            alignment.early: StateVector(
                (float(early_r[-1][0]), float(early_r[-1][1]), float(early_r[-1][2])),
                (float(early_v[-1][0]), float(early_v[-1][1]), float(early_v[-1][2])),
            ),
            alignment.late: StateVector(
                (float(late_r[-1][0]), float(late_r[-1][1]), float(late_r[-1][2])),
                (float(late_v[-1][0]), float(late_v[-1][1]), float(late_v[-1][2])),
            ),
        },
        a1_mjd=float(late_a1[-1]),
        early_minimum_radius_km=float(np.linalg.norm(early_r, axis=1).min()),
        minimum_distance_km=float(np.linalg.norm(early_r - late_r, axis=1).min()),
        binding_sha256=binding_sha256,
        run=result,
        early_track=tuple(
            (float((a1 - early_a1[-1]) * 86400.0), *map(float, r), *map(float, v))
            for a1, r, v in zip(early_a1, early_r, early_v, strict=True)
        ),
    )


def _moved(path: str, old_root: str, new_root: str | None) -> str:
    if not path.startswith(old_root):
        return path
    return "" if new_root is None else new_root + path[len(old_root):]


def _relocated(run: GmatRunResult, old_root: str, new_root: str | None) -> GmatRunResult:
    """``run`` with every path under ``old_root`` moved to ``new_root``, or blanked."""
    return dataclasses.replace(
        run,
        workdir=_moved(run.workdir, old_root, new_root),
        script_path=_moved(run.script_path, old_root, new_root),
        stdout_path=_moved(run.stdout_path, old_root, new_root),
        stderr_path=_moved(run.stderr_path, old_root, new_root),
        log_path=None if run.log_path is None else _moved(run.log_path, old_root, new_root),
        reports={key: _moved(value, old_root, new_root) for key, value in run.reports.items()},
        partial_reports={
            key: _moved(value, old_root, new_root) for key, value in run.partial_reports.items()
        },
    )


def _relocated_probe(probe: StateProbe, root: Path, new_root: str | None = None) -> StateProbe:
    aligned = probe.aligned
    if aligned is not None:
        aligned = dataclasses.replace(aligned, run=_relocated(aligned.run, str(root), new_root))
    return dataclasses.replace(
        probe, run=_relocated(probe.run, str(root), new_root), aligned=aligned
    )


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    """The probe run's files, held until the caller has made its output directory."""

    run: GmatRunResult
    scratch_root: str
    files: dict[str, bytes]
    #: The alignment probe as run, paths still under scratch_root.
    aligned: AlignedStates | None = None

    def place(self, scenario: Scenario, destination: str | Path) -> Scenario:
        """Write the files under ``destination`` and point the scenario's record at them."""
        if scenario.state_probe is None:
            raise GmatRunError("The scenario carries no state probe to place")
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=False)
        for relative, data in self.files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        placed = _relocated_probe(
            dataclasses.replace(scenario.state_probe, run=self.run, aligned=self.aligned),
            Path(self.scratch_root), str(root.resolve()),
        )
        return dataclasses.replace(scenario, state_probe=placed)


def load_scenario(
    script: str | Path,
    config: OrbitWinConfig,
    *,
    gmat_console: str | Path | None = None,
    startup_file: str | Path | None = None,
) -> tuple[Scenario, GmatScriptDocument, ProbeEvidence | None]:
    """Parse ``script``, probing it first whenever a console is given.

    The one way in for an entry point that may hold a console: with one the
    initial states and epochs are GMAT's, without one they are our in-process
    reading, which covers EarthMJ2000Eq Keplerian/Cartesian on one epoch text.

    The probe runs in a temporary directory, so a refusal leaves nothing behind.
    Its record carries no paths until the caller places the evidence with
    ``ProbeEvidence.place``; one that never does records no dead path.
    """
    if gmat_console is None:
        scenario, document = parse_gmat_script(script, config)
        return scenario, document, None
    if startup_file is None:
        raise GmatRunError("A state probe needs the startup file that matches the console")
    with tempfile.TemporaryDirectory(prefix="orbitwin-probe-") as scratch:
        root = Path(scratch).resolve() / "state_probe"
        probe = probe_initial_states(
            script, gmat_console, startup_file, root, timeout_s=config.gmat.timeout_s,
            extra_resource_types=config.policy.extra_resource_types,
        )
        try:
            parse_gmat_script(script, config, probe=_relocated_probe(probe, root))
        except AlignedStatesNeeded as needed:
            # Δ > 0: t = 0 is the alignment's end, which only GMAT can say.
            probe = dataclasses.replace(
                probe,
                aligned=probe_alignment(
                    script, needed.alignment, needed.binding_sha256, gmat_console,
                    startup_file, root / "alignment", timeout_s=config.gmat.timeout_s,
                    extra_resource_types=config.policy.extra_resource_types,
                ),
            )
        except ParseError:
            pass  # refused again, with the same words, below
        files = {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
    evidence = ProbeEvidence(probe.run, str(root), files, probe.aligned)
    probe = _relocated_probe(probe, root)
    scenario, document = parse_gmat_script(script, config, probe=probe)
    return scenario, document, evidence


def add_probe_arguments(parser: argparse.ArgumentParser) -> None:
    """The optional console pair for a tool that reads the organizer's script."""
    parser.add_argument(
        "--gmat-console",
        help="with --startup-file: read the initial states through the GMAT state probe",
    )
    parser.add_argument("--startup-file")


def load_scenario_for_tool(
    parser: argparse.ArgumentParser,
    script: str | Path,
    config: OrbitWinConfig,
    gmat_console: str | Path | None,
    startup_file: str | Path | None,
) -> tuple[Scenario, GmatScriptDocument]:
    """load_scenario for a command-line tool, which keeps no probe files."""
    if (gmat_console is None) != (startup_file is None):
        parser.error("--gmat-console and --startup-file must be supplied together")
    scenario, document, _ = load_scenario(
        script, config, gmat_console=gmat_console, startup_file=startup_file
    )
    return scenario, document


__all__ = [
    "PROBE_REPORT_FILENAME",
    "PROBE_RESOURCE_NAME",
    "ProbeEvidence",
    "add_probe_arguments",
    "build_probe_script",
    "load_scenario",
    "load_scenario_for_tool",
    "parse_probe_report",
    "probe_initial_states",
]
