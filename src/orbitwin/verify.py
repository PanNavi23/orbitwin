from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .burn_rows import check_flown_impulses, maneuvers_flown
from .config import SCHEDULE_TOLERANCE_S, OrbitWinConfig, validate_config
from .console_copy import GROUND_TRACK_TYPES, ConsoleCopy, write_console_copy
from .constraints import (
    FORMAL_FRONTIER,
    ROBUST_FRONTIER,
    apply_constraints,
    below_minimum_flown_altitude,
    frontier_of,
    gmat_contradicts_hit,
    gmat_verified_on_record,
    interception_required,
    shallow_stop,
    stop_depth_mismatch,
    validate_candidate,
)
from .diagnostic import DIAGNOSTIC_PRECISION, add_diagnostic_report
from .errors import (
    AltitudeFloorError,
    ConstraintVerdictError,
    DynamicsError,
    GmatRunError,
    OrbitWinError,
    PolicyError,
    StopVerdictError,
)
from .event_detection import EventResult
from .gmat_parser import (
    GmatScriptDocument,
    _code_part,
    created_at,
    created_types,
    parse_gmat_script,
    plot_command,
    report_file_specs,
)
from .gmat_runner import (
    EXIT_STATUS_ERROR,
    NO_FRESH_REPORT_ERROR,
    GmatError,
    file_size_limit_for,
    gmat_errors,
    is_test_double,
    machine_error,
    no_answer,
    run_gmat,
    said_by_gmat,
    unexplained_log_errors,
)
from .io import prepare_empty_output
from .mission_clock import clock_record
from .models import (
    STAGE_SCRIPT_NAME,
    Candidate,
    CandidateStage,
    GmatRunResult,
    ReportFileSpec,
    Scenario,
    StateProbe,
    clock_stamp_mismatch,
    file_sha256,
    stage_rank,
    uses_head_start,
)
from .official_report import (
    last_digit,
    printed_distance_margin_km,
    read_official_rows,
    same_printed_value,
)
from .openframes import OPENFRAMES_TYPES
from .report_parser import (
    STOP_DEPTH_TOLERANCE,
    alignment_row_distance_km,
    evaluate_diagnostic_report,
    parse_numeric_report,
    report_clock,
    stop_depth_used_km,
)
from .rules import InterceptionRules, interception_rules, rule_set_record
from .score import (
    HEAD_START_ALIGNMENT_KEY,
    apply_score,
    score_candidate,
    score_stamp,
    score_stamp_difference,
    scored_distance_km,
)
from .script_renderer import (
    RenderResult,
    assert_final_script_policy,
    assert_team_number_for_submission,
    official_report_of,
    render_candidate,
    team_number_warnings,
    with_added_alignment,
    write_rendered_candidate,
)
from .spad import spad_check
from .state_probe import load_scenario
from .trajectory import state_with_burns_at


def gmat_authority_stage(config: OrbitWinConfig, result: GmatRunResult) -> CandidateStage:
    """Return the tier a successful run under this build can justify.

    A run only reaches the organizer tier when the build has been declared
    organizer-compatible and the executable matches the configured fingerprint.
    validate_config requires that fingerprint alongside the flag, so the tier
    always rests on a named binary rather than on a boolean. The startup file
    is pinned too when configured: it selects the gravity, leap-second and EOP
    data, so it changes the physics as surely as the binary does.

    Anything else is a public-GMAT result, ranked below the organizer's validator
    and build (COMPETITION_POLICY.md, Order of authority; ASSUMPTIONS_AND_GAPS.md,
    Release status meaning).
    """
    if not config.gmat.organizer_compatible:
        return CandidateStage.PUBLIC_GMAT_VERIFIED
    expected = config.gmat.organizer_executable_sha256
    if expected is None or result.executable_sha256.lower() != expected.strip().lower():
        return CandidateStage.PUBLIC_GMAT_VERIFIED
    expected_startup = config.gmat.organizer_startup_sha256
    if expected_startup is not None and (
        result.startup_sha256 is None
        or result.startup_sha256.lower() != expected_startup.strip().lower()
    ):
        return CandidateStage.PUBLIC_GMAT_VERIFIED
    return CandidateStage.ORGANIZER_GMAT_VERIFIED


def verify_candidate_with_gmat(
    original_script: str | Path,
    scenario: Scenario,
    original_document: GmatScriptDocument,
    candidate: Candidate,
    config: OrbitWinConfig,
    gmat_console: str | Path,
    startup_file: str | Path | None,
    output_dir: str | Path,
    *,
    timeout_s: float | None = None,
) -> tuple[Candidate, GmatRunResult, dict[str, Any]]:
    # Callers render first: solve to sort renderable candidates, certify before
    # it creates anything.
    root = Path(output_dir)
    prepare_empty_output(root, "GMAT verification")
    # Recorded before anything is overwritten, and on every path including the
    # successful one. GMAT replaces the intercept epoch and the closest
    # approach, so without this the analytical figures are simply lost -- and
    # the distance between the two is the measurement that says whether the seed
    # layer can be trusted at all.
    # setdefault: a re-verification keeps the plan, not the earlier GMAT flight.
    candidate.metadata.setdefault("analytical_before_gmat", {
        "successful": planned_hit(candidate),
        "intercept_time_s": candidate.intercept_time_s,
        "minimum_distance_km": candidate.minimum_distance_km,
        # The score too. apply_score below overwrites it with one computed from
        # this run's coefficients, so a candidate loaded from a candidates.json
        # written under different ones loses the number it was chosen by
        # without a trace. Recording it does not make the two comparable --
        # GMAT moves the event inputs as well -- but it puts both in the
        # certification evidence, where a reader can see they differ.
        "score": candidate.score,
    })
    # Not team_solution.script. This is a working render under verification and
    # the candidate reaches PUBLIC_GMAT_VERIFIED at best, so carrying the
    # submission name would put files an operator could pick up and upload in
    # every gmat_verification/<id>/ directory. The naming ladder exists to stop
    # exactly that.
    #: Each flight's GmatLog errors, by run id: run_gmat reads them off a run that succeeded.
    logged: dict[str, list[str]] = {}
    #: The scenario of the last flight's script, which its report is read under.
    last_flown: list[Scenario] = []

    def measure(path: str, flown_scenario: Scenario, stop_depth_km: float) -> EventResult:
        last_flown[:] = [flown_scenario]
        return evaluate_diagnostic_report(
            path,
            scenario.pursuer.name,
            scenario.target.name,
            config.policy.intercept_radius_km,
            central_body_radius_km=config.policy.central_body_radius_km,
            maneuvers=maneuvers_flown(flown_scenario),
            alignment=flown_scenario.alignment,
            stop_depth_km=stop_depth_km,
            geodetic_radius_km=flown_scenario.geodetic_radius_km,
        )

    def fly(
        suffix: str, mission_end_policy: str, *, measuring: bool, flown: Candidate = candidate
    ) -> tuple[RenderResult, GmatRunResult, str, EventResult, str]:
        """Render under one mission end, run GMAT on its diagnostic sibling, read it.

        A measuring flight only locates the stop, so one that aborts after
        measuring it (a drag model failing below 100 km) still serves.
        """
        script = root / f"verification_candidate{suffix}.script"
        rendered = write_rendered_candidate(
            script,
            root / f"verification_candidate{suffix}.script.diff.txt",
            original_document,
            scenario,
            flown,
            replace(config, policy=replace(config.policy, mission_end_policy=mission_end_policy)),
        )
        flown_scenario, flown_document = parse_gmat_script(
            script, config, probe=scenario.state_probe
        )
        diagnostic = add_diagnostic_report(
            flown_document, flown_scenario, config.gmat.diagnostic_report_name,
            check_burns=True,
        )
        sibling = root / f"diagnostic_sibling{suffix}.script"
        sibling.write_bytes(diagnostic.script_bytes)
        names = written_reports(flown_scenario)
        # Before the diagnostic report: run_gmat stops at the first missing one.
        if diagnostic.burn_report_filename is not None:
            names.append(diagnostic.burn_report_filename)
        names.append(diagnostic.report_filename)
        run = run_gmat(
            gmat_console,
            sibling,
            root / f"diagnostic-run{suffix.replace('_', '-')}",
            names,
            startup_file=startup_file,
            timeout_s=timeout_s or config.gmat.timeout_s,
            file_size_limit_bytes=file_size_limit_for(flown_scenario),
            template_dir=Path(original_script).resolve().parent,
        )
        if run.log_errors:
            logged[run.run_id] = list(run.log_errors)
        if not run.success and not measuring:
            raise GmatRunError((run.error or "GMAT diagnostic run failed") + said_by_gmat(run))
        path = run.reports.get(diagnostic.report_filename)
        if not run.success:
            path = run.partial_reports.get(diagnostic.report_filename)
        if path is None:
            raise GmatRunError((run.error or "GMAT diagnostic run failed") + said_by_gmat(run))
        maneuvers = maneuvers_flown(flown_scenario)
        if diagnostic.burn_report_filename is not None:
            reports = run.reports if run.success else run.partial_reports
            check_flown_impulses(
                reports.get(diagnostic.burn_report_filename), maneuvers, complete=run.success
            )
        event = measure(path, flown_scenario, config.policy.stop_depth_km)
        if not run.success and event.stop_time_s is None:
            raise GmatRunError((run.error or "GMAT diagnostic run failed") + said_by_gmat(run))
        if run.success:
            report_reaches_mission_end(event, rendered.mission_end_s, path)
        return rendered, run, path, event, file_sha256(script)

    # Measured first, then the submitted script flown as it will be uploaded.
    # The first flight ends past where the search predicted the hit; only one
    # that has not measured where to stop is flown on to T_max, where a miss is
    # scored and a late entry is found.
    candidate.metadata.pop("gmat_event", None)
    # Resolution warnings describe one verification; this one writes its own.
    candidate.warnings = tuple(
        warning for warning in candidate.warnings
        if not warning.startswith((_SPARSE_ENTRY, _SHALLOW_STOP))
    )
    policy = config.policy.mission_end_policy
    rendered, result, diagnostic_path, event, flown_sha256 = fly(
        "", policy, measuring=policy != "horizon",
        flown=measuring_candidate(candidate, scenario, config),
    )
    if (
        event.stop_time_s is None
        and rendered.mission_end_s < scenario.max_time_s - SCHEDULE_TOLERANCE_S
    ):
        _, result, diagnostic_path, event, flown_sha256 = fly(
            "_to_horizon", "horizon", measuring=True
        )
    hit = (
        event.first_entry_time_s is not None
        and event.first_entry_time_s <= scenario.max_time_s + 1.0e-8
    )
    stop_s: float | None = None
    margin_km = 0.0
    if hit and policy != "horizon":
        stop_s = event.stop_time_s
        # The official report prints its rows to Precision digits, and step 7 reads the
        # stop depth off them: deeper by what that rounding can take (R3 T14, Precision 4).
        margin_km = printed_margin_km(scenario, last_flown[0], diagnostic_path, event)
        if margin_km > 0.0 and stop_s is not None:
            deeper = measure(diagnostic_path, last_flown[0],
                             config.policy.stop_depth_km + margin_km).stop_time_s
            stop_s = stop_s if deeper is None else deeper
        if stop_s is None:  # Still inside at T_max: stop at the deepest point.
            stop_s = event.minimum_distance_time_s
        candidate.successful = True
        candidate.intercept_time_s = event.first_entry_time_s
        candidate.metadata["gmat_event"] = {"stop_time_s": stop_s}
        rendered, result, diagnostic_path, event, flown_sha256 = fly(
            "_at_stop", policy, measuring=False
        )
        if event.first_entry_time_s is None:
            raise StopVerdictError(
                f"Stopped at {stop_s!r} s, the submitted script no longer enters"
            )
        if event.distance_at_end_km > config.policy.intercept_radius_km:
            raise StopVerdictError(
                f"Stopped at {stop_s!r} s, the submitted script's last report row is "
                f"{event.distance_at_end_km:.6g} km from the target, outside the "
                f"{config.policy.intercept_radius_km:.6g} km sphere"
            )
    candidate.intercept_time_s = _completion_time_s(event, hit)
    stop_depth_km = (
        config.policy.intercept_radius_km - event.distance_at_end_km if stop_s is not None else None
    )
    candidate.minimum_distance_km, closest_s = _closest_approach(event, hit)
    candidate.distance_at_completion_km = (
        config.policy.intercept_radius_km if hit else event.distance_at_end_km
    )
    candidate.successful = hit
    candidate.gmat_run_id = result.run_id
    candidate.metadata["event_refined"] = True
    if event.minimum_pursuer_radius_km is not None:
        # GMAT replaces the interception epoch and the closest approach; the
        # geocentric radius that gates submission was still the two-body
        # estimate from the search. Under a force model richer than point mass
        # those are not the same number, and this is the one that was flown.
        candidate.metadata["minimum_pursuer_radius_km"] = float(
            event.minimum_pursuer_radius_km
        )
    # GMAT's geodetic Earth.Altitude, the organizer's criterion: recorded, never a gate.
    candidate.metadata["minimum_geodetic_altitude_km"] = event.minimum_geodetic_altitude_km
    required = interception_required(scenario)
    candidate.metadata["scored_miss"] = not candidate.successful and not required
    if candidate.successful or not required:
        candidate.promote(
            gmat_authority_stage(config, result),
            run_id=result.run_id,
            executable_sha256=result.executable_sha256,
            startup_sha256=result.startup_sha256,
            derived_startup_sha256=result.derived_startup_sha256,
            organizer_compatible=config.gmat.organizer_compatible,
        )
    candidate.metadata["gmat_event"] = {
        "first_entry_time_s": event.first_entry_time_s,
        "minimum_distance_km": candidate.minimum_distance_km,
        "minimum_distance_time_s": closest_s,
        # Between rows: a hit's distance, and for a miss certify's check on the burns' rows.
        "interpolated_minimum_distance_km": event.minimum_distance_km,
        "interpolated_minimum_distance_time_s": event.minimum_distance_time_s,
        "distance_at_end_km": event.distance_at_end_km,
        "samples_inside_radius": event.samples_inside_radius,
        "first_entry_segment_dt_s": event.first_entry_segment_dt_s,
        "stop_time_s": stop_s,
        "stop_depth_km": stop_depth_km,
        # The setting it was stopped under; rank and certify compare it.
        "stop_depth_setting_km": config.policy.stop_depth_km,
        # Added to it for the official report's printed digits (printed_margin_km).
        "printed_margin_km": margin_km,
        "minimum_geodetic_altitude_km": event.minimum_geodetic_altitude_km,
        "flown_script_sha256": flown_sha256,
    }
    resolution_warnings: tuple[str, ...] = ()
    if event.entry_is_under_resolved:
        # The report cadence is the propagator's output step, an immutable
        # organizer field. Say how much evidence the crossing actually rests on.
        resolution_warnings = (
            f"{_SPARSE_ENTRY}{event.samples_inside_radius} report sample(s) inside "
            f"the {config.policy.intercept_radius_km:.6g} km sphere "
            f"(segment dt {event.first_entry_segment_dt_s} s); the epoch is interpolated",
        )
    depth_used_km = stop_depth_used_km(
        config.policy.stop_depth_km, config.policy.intercept_radius_km
    )
    if stop_depth_km is not None and stop_depth_km < depth_used_km * (1.0 - STOP_DEPTH_TOLERANCE):
        # The passage never got that deep, so the upload ends at its deepest.
        resolution_warnings += (
            f"{_SHALLOW_STOP}{stop_depth_km * 1000.0:.4g} m "
            f"inside the {config.policy.intercept_radius_km:.6g} km sphere, short of "
            f"policy.stop_depth_km = {depth_used_km * 1000.0:.4g} m",
        )
    if uses_head_start(candidate.burns) and scenario.alignment_minimum_distance_km is not None:
        # Under from_origin d_min counts this flight's alignment rows, not the probe's.
        candidate.metadata[HEAD_START_ALIGNMENT_KEY] = alignment_row_distance_km(
            diagnostic_path, scenario.pursuer.name, scenario.target.name, scenario.alignment
        )
    # Constraints first so any penalties are recorded, then score, so the stored
    # score and the stored penalties describe the same state. GMAT is the
    # authority on intercept time and completion distance, and both are score
    # inputs, so this must run before anything ranks on the result.
    constraint = apply_constraints(candidate, scenario, config, require_success=required)
    apply_score(candidate, scenario, config.score)
    if not constraint.valid:
        raise ConstraintVerdictError(
            "GMAT candidate failed constraints: " + "; ".join(constraint.violations)
        )
    if below_minimum_flown_altitude(candidate, config):
        flown_km = float(candidate.metadata["minimum_pursuer_radius_km"])
        raise AltitudeFloorError(
            f"GMAT flew candidate {candidate.candidate_id} down to "
            f"{flown_km - config.policy.central_body_radius_km:.3f} km altitude, under "
            f"policy.minimum_flown_altitude_km = {config.policy.minimum_flown_altitude_km}"
        )
    candidate.warnings = tuple(sorted(set(candidate.warnings + resolution_warnings)))
    evidence = {
        "candidate": candidate.to_dict(),
        "gmat_run": result.to_dict(),
        "diagnostic_report": diagnostic_path,
        "event": candidate.metadata["gmat_event"],
        # What candidate.score was priced with, on which script; check_report_rows
        # --verification holds the config and --script to them.
        "score_block": score_stamp(config.score),
        "candidate_script_sha256": flown_sha256,
        # GMAT logged these and exited 0; certify refuses any not in BENIGN_LOG_ERRORS.
        "gmat_log_errors": logged,
        # The alignment's rows before t = 0, as certify reads them: check_report_rows
        # --frozen-rows on the GUI path, where no certification.json is written.
        "frozen_rows": report_clock(
            parse_numeric_report(diagnostic_path), scenario.pursuer.name,
            scenario.target.name, scenario.alignment,
        ).frozen_rows,
    }
    (root / "verification.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return candidate, result, evidence


def printed_margin_km(
    scenario: Scenario, flown: Scenario, path: str, event: EventResult
) -> float:
    """How much deeper a hit stops so the official report's printed last row is
    policy.stop_depth_km inside: the distance's rounding at the craft's radius there.
    ``flown`` is the scenario of the script that wrote ``path``."""
    spec = official_report_of(scenario.report_files)
    if spec is None or event.first_entry_time_s is None:
        return 0.0
    report = parse_numeric_report(path)
    times = report_clock(report, scenario.pursuer.name, scenario.target.name,
                         flown.alignment).times_s
    near = times >= event.first_entry_time_s - (event.report_step_s or 0.0)
    radius_km = 0.0
    for craft in (scenario.pursuer.name, scenario.target.name):
        position = np.column_stack([
            report.column(f"{craft}.EarthMJ2000Eq.{axis}", f"{craft}.{axis}") for axis in "XYZ"
        ])
        radius_km = max(radius_km, float(np.linalg.norm(position[near], axis=1).max(initial=0.0)))
    return printed_distance_margin_km(spec, scenario.pursuer.name, scenario.target.name,
                                      radius_km)


#: How far a whole report's last row may fall short of the script's end. GMAT writes a
#: row where each Propagate stops; on R2026a that row was at most 2.5e-6 s short in
#: 122 fresh flights and 7360 archived ones (the review of #187).
REPORT_END_TOLERANCE_S = 1.0e-3


def report_reaches_mission_end(
    event: EventResult, mission_end_s: float, path: str, *,
    precision: int = DIAGNOSTIC_PRECISION,
) -> None:
    """Refuse a complete run whose report stops short of the script's end.

    GMAT writes a row at the end of every Propagate, so a report that ends
    earlier was cut short (a full disk), whatever GmatConsole returned; one row
    missing is a step short. The slack is REPORT_END_TOLERANCE_S plus the rounding
    of the time column at the report's Precision (the diagnostic report's 17 digits).
    """
    end_s = event.end_time_s
    if end_s is None:
        raise GmatRunError(f"{path}: the report has no rows")
    rounding_s = 10.0 ** (1 - precision) * max(abs(mission_end_s), abs(end_s), 1.0)
    if mission_end_s - end_s > REPORT_END_TOLERANCE_S + rounding_s:
        raise GmatRunError(
            f"{path} ends at {end_s!r} s, {mission_end_s - end_s:.6g} s short of the script's "
            f"end at {mission_end_s!r} s (rows up to {event.report_step_s or 0.0:.6g} s apart): "
            "the report was cut short"
        )


def measuring_candidate(
    candidate: Candidate, scenario: Scenario, config: OrbitWinConfig
) -> Candidate:
    """The candidate as the first flight renders it, ending where its stop is measured.

    A straight pass leaves the sphere at most 2 radius / speed after entry, by
    which time it has reached the stop depth or shown it never will. The speed
    is two-body at the predicted entry. Not extended when it is unknown, nor
    after an epoch alignment, whose solver clock the two-body states are not on.
    """
    entry_s = candidate.intercept_time_s
    if not candidate.successful or entry_s is None:
        return candidate
    if scenario.alignment is not None and scenario.alignment.offset_s > 0.0:
        return candidate
    try:
        pursuer = state_with_burns_at(
            scenario.pursuer.initial_state, candidate.burns, entry_s, scenario.mu_km3_s2
        )
        target = state_with_burns_at(scenario.target.initial_state, (), entry_s, scenario.mu_km3_s2)
    except DynamicsError:
        return candidate
    speed_km_s = float(np.linalg.norm(np.asarray(pursuer.v) - np.asarray(target.v)))
    if not math.isfinite(speed_km_s) or speed_km_s <= 0.0:
        return candidate
    return replace(
        candidate, intercept_time_s=entry_s + 2.0 * config.policy.intercept_radius_km / speed_km_s
    )


_SPARSE_ENTRY = "GMAT first entry rests on "
_SHALLOW_STOP = "The submitted script's last report row is only "


def _closest_approach(event: EventResult, hit: bool) -> tuple[float, float | None]:
    """A miss's closest approach and its epoch, on the closest report row: the
    validator read D off the rows (2026-09-05), and the Hermite minimum between
    them is never farther. A hit scores 5 whatever; its passage minimum stays
    interpolated, as the tie-break."""
    if hit:
        return event.minimum_distance_km, event.minimum_distance_time_s
    if event.minimum_row_distance_km is None:
        # Our invariant, not GMAT's verdict: every event verify reads comes off a report.
        raise RuntimeError("A miss's closest approach was read off no report rows")
    return event.minimum_row_distance_km, event.minimum_row_distance_time_s


def _completion_time_s(event: EventResult, hit: bool) -> float | None:
    """When the submitted report says a hit completed: its last row, wherever the
    script stops (2026-09-05: the validator read C off the last row). None for a miss."""
    return event.end_time_s if hit else None


#: GmatConsole R2026a's words for an OpenFrames object it cannot create, read
#: when a record carries no script line (tests/integration/test_gmat_contract.py).
_UNKNOWN_TYPE_RE = re.compile(
    r'\*\*\*\* ERROR \*\*\*\* Interpreter Exception: Cannot create an object "(?P<name>\w+)"\. '
    r'The "(?P<type>\w+)" is an unknown object type '
)
_NO_LHS_RE = re.compile(
    r'\*\*\*\* ERROR \*\*\*\* Interpreter Exception: Cannot find LHS object named "(?P<name>\w+)" '
    r"in line:$"
)
#: The first word of a script line: a Create, or the object whose field it sets.
_LINE_SUBJECT_RE = re.compile(
    r"^\s*(?:GMAT\s+)?(?:Create\s+(?P<type>\w+)\s+(?P<names>[^;]*)|(?P<object>\w+)\s*\.)"
)
#: Distinct errors read at most; the workshop template writes 58.
_ERROR_LINES = 10_000


def written_reports(scenario: Scenario) -> list[str]:
    """The ReportFiles GMAT writes: R2026a writes no file at all for WriteReport = false."""
    return [report.filename for report in scenario.report_files if report.write_report]


#: GMAT's words for a file it could not read, and only those: a data file our install
#: or run folder lacks (R2026a: a CSSI file the startup names, a TideFile beside the
#: template, a kernel) is our fault, not the script's. "not found" alone is a script error.
_MISSING_FILE_RE = re.compile(
    r'\bCannot open\b|\bfile(?: name)? "[^"]*" does not exist|^ERROR loading ', re.MULTILINE
)


class PlotRefusal(NamedTuple):
    """Why GmatConsole refused a script for its plots, and what else GMAT said."""

    #: As main says it, or "".
    reason: str
    #: GMAT's errors the plots do not explain, each once, with the line they name.
    other: tuple[str, ...]
    #: False when GMAT wrote more errors than were read.
    complete: bool = True


def _reported(lines: tuple[str, ...]) -> str:
    if not lines:
        return ""
    more = f" (and {len(lines) - 3} more)" if len(lines) > 3 else ""
    return ". GMAT reported: " + "; ".join(lines[:3]) + more


def _is_openframes_line(source: str, openframes: dict[str, str]) -> bool:
    """Whether a script line GMAT named creates, or sets a field of, an OpenFrames object."""
    match = _LINE_SUBJECT_RE.match(source)
    if match is None:
        return False
    if match["type"] is not None:
        names = [name for name in re.split(r"[,\s]+", match["names"].strip()) if name]
        return bool(names) and all(openframes.get(name) == match["type"] for name in names)
    return match["object"] in openframes


#: R2026a's words for a Toggle, PenUp, ... of an object it could not create.
_TOGGLE_SUBSCRIBER_RE = re.compile(
    r'\*\*\*\* ERROR \*\*\*\* Interpreter Exception: Nonexistent Subscriber "(?P<name>\w+)" '
    r"referenced in the (?:Toggle|PenUp|PenDown)"
)


def _is_toggle_of_openframes(
    error: GmatError, openframes: dict[str, str], script: GmatScriptDocument
) -> bool:
    """A plot command (Toggle, PenUp, ...) of plots alone, refused for an OpenFrames
    object it names."""
    found = _TOGGLE_SUBSCRIBER_RE.match(error.text)
    try:
        toggled = plot_command(_code_part(error.source), script)
    except OrbitWinError:
        return False
    return bool(found and toggled and found["name"] in openframes and found["name"] in toggled)


def _is_openframes_error(
    error: GmatError, openframes: dict[str, str], script: GmatScriptDocument
) -> bool:
    """Whether GMAT reported this error for an OpenFrames object the script creates.

    By the script line GMAT printed after it; without one, by R2026a's words.
    """
    if error.line:
        return _is_openframes_line(error.source, openframes) or _is_toggle_of_openframes(
            error, openframes, script
        )
    unknown = _UNKNOWN_TYPE_RE.match(error.text)
    if unknown:
        return openframes.get(unknown["name"]) == unknown["type"]
    lhs = _NO_LHS_RE.match(error.text)
    return lhs is not None and lhs["name"] in openframes


def _missing_report_after_ground_track(document: GmatScriptDocument, error: str | None) -> str:
    """The GroundTracks created before the ReportFile whose missing report ``error`` names.

    Measured on R2026a: R1 with the workshop's GroundTrack created before its
    ReportFile ran clean and wrote no report; created after it, the report.
    Only a report the script writes counts: one with WriteReport false is none.
    """
    prefix = NO_FRESH_REPORT_ERROR.format("")
    if not (error or "").startswith(prefix):
        return ""
    missing = (error or "")[len(prefix):]
    try:
        specs = report_file_specs(document)
    except OrbitWinError:
        return ""
    owner = next(
        (spec.name for spec in specs if spec.filename == missing and spec.write_report), None
    )
    lines = created_at(document)
    if owner is None or owner not in lines:
        return ""
    types = created_types(document)
    return ", ".join(
        sorted(
            f"{kind} {name}"
            for name, kind in types.items()
            if kind in GROUND_TRACK_TYPES and lines.get(name, len(document.lines)) < lines[owner]
        )
    )


def plot_refusal(script: GmatScriptDocument, result: GmatRunResult) -> PlotRefusal:
    """Why GmatConsole refused ``script`` for its plots, or ""; and GMAT's other errors.

    ``script`` is the document whose bytes GMAT ran, not a path to read again.
    The OpenFrames reason is main's: an exit status from a script creating
    OpenFrames objects. The GroundTrack one is the measured case: the missing
    report is one the script writes, and a GroundTrack is created before its
    ReportFile. The other errors are GMAT's own that are not about an
    OpenFrames object the script creates.
    """
    if result.success:
        return PlotRefusal("", ())
    created = created_types(script)
    openframes = {name: kind for name, kind in created.items() if kind in OPENFRAMES_TYPES}
    said = gmat_errors(result, _ERROR_LINES)
    other = tuple(
        str(error) for error in said.errors
        if error.kind != "summary" and not _is_openframes_error(error, openframes, script)
    )
    status = result.return_code != 0 and result.error == EXIT_STATUS_ERROR.format(
        result.return_code
    )
    if openframes and status:
        named = ", ".join(sorted(f"{kind} {name}" for name, kind in openframes.items()))
        return PlotRefusal(f"GmatConsole cannot create {named} (OpenFrames)", other,
                           said.complete)
    ground = _missing_report_after_ground_track(script, result.error)
    if ground:
        return PlotRefusal(
            f"it creates {ground} before that ReportFile, and on R2026a a GroundTrack "
            "created before its ReportFile left GmatConsole writing no report",
            other, said.complete,
        )
    return PlotRefusal("", other, said.complete)


def refusal_message(
    script: GmatScriptDocument, result: GmatRunResult, failed: str, *, baseline: bool = False,
    kept_as_written: str | None = None,
) -> str:
    """A failed run of ``script``: GMAT's own errors first, then what they mean.

    Plots alone: run it in the GUI, which has the plugin. Anything else GMAT
    rejected, the GUI rejects too, and an unread rest might hold such an error,
    so neither offers it.
    """
    reason, other, complete = plot_refusal(script, result)
    name = script.path.name
    said = ""
    if not complete:
        said = f". GMAT wrote more than {_ERROR_LINES} errors and not all were read"
        said += "; the first follow" if other else ""
    said += _reported(other)
    stop = (
        "The official script may not be edited: stop and take this to the organizer "
        "(docs/RUNBOOK.md, Stop rules)"
        if baseline
        else "If `orbitwin baseline` fails the same way on the official script, stop and "
        "take this to the organizer (docs/RUNBOOK.md, Stop rules); if not, the rendered "
        "edits are at fault"
    )
    also = f"; besides, {reason}" if reason else ""
    if other and any(_MISSING_FILE_RE.search(error) for error in other):
        note = (
            f". GMAT rejects {name} for these. Some name a file GMAT could not open or "
            "find: if it is this machine's (the GMAT install, its startup file or the run "
            "folder), the fault is ours, so fix it and run again (docs/RUNBOOK.md step 3). "
            "If not, the GMAT GUI, which shares GmatConsole's interpreter, would reject the "
            f"script too, so neither runs it as written{also}. {stop}"
        )
    elif other:
        note = (
            f". GMAT rejects {name} for these, and the GMAT GUI, which shares its "
            f"interpreter, would too, so neither runs it as written{also}. {stop}"
        )
    elif reason and not complete:
        note = (
            f". What GMAT said is incomplete, so whether the GMAT GUI would run {name} is "
            f"unknown{also}. {stop}"
        )
    elif reason:
        then = (
            "Solve still runs on its private copy; the zero-burn report needs the GMAT GUI"
            if baseline
            else "Run it in the GMAT GUI and put the report it writes through check_report_rows "
            "(docs/RUNBOOK.md step 6)"
        )
        if kept_as_written is not None and not baseline:
            # No certification.json on this path to say so.
            then += (
                f". Its official report keeps the organizer's name {kept_as_written!r} "
                "(policy.report_filename_as_written): upload it under that exact name"
            )
        note = f". {name}: {reason}. {then}"
    else:
        note = ""
    return (result.error or failed) + said + note


def console_copy_for(
    script: GmatScriptDocument, result: GmatRunResult, failed: str, config: OrbitWinConfig,
    workdir: Path, *, baseline: bool = False, kept_as_written: str | None = None,
) -> tuple[Path, ConsoleCopy]:
    """The proven console copy to run after GmatConsole refused ``script``, or raise.

    Only under gmat.console_copy, and only when GMAT's records say the plots
    and nothing else: every error is one it writes for an OpenFrames object the
    script creates, or the run ended clean and wrote no report after a
    GroundTrack. Otherwise the refusal stands as refusal_message words it, with
    why no copy ran.
    """
    message = refusal_message(script, result, failed, baseline=baseline,
                              kept_as_written=kept_as_written)
    reason, other, complete = plot_refusal(script, result)
    if not (config.gmat.console_copy and reason):
        raise GmatRunError(message)
    if other:
        why = "GMAT reported errors besides the ones the plots explain"
    elif not complete:
        why = f"GMAT wrote more than {_ERROR_LINES} errors, so the plots may not be all"
    else:
        try:
            path, copy = write_console_copy(script.path, workdir)
        except (OrbitWinError, OSError, ValueError) as exc:
            if machine_error(exc) is not None:
                # This machine, not the script: no answer, not the GUI path.
                raise GmatRunError(f"{message}. The console copy could not be written: "
                                   f"{exc}") from exc
            why = str(exc)
        else:
            if copy.submitted_sha256 == hashlib.sha256(script.raw_bytes).hexdigest() == (
                result.script_sha256
            ):
                return path, copy
            path.unlink(missing_ok=True)
            why = (f"{script.path.name} changed: GmatConsole refused "
                   f"{result.script_sha256} and the copy is of {copy.submitted_sha256}")
    raise GmatRunError(f"{message}. No console copy was run (gmat.console_copy): {why}")


def copy_record(path: Path, copy: ConsoleCopy, refused: GmatRunResult) -> dict[str, Any]:
    """What certification.json and baseline.json say of a console copy and why it ran."""
    return {
        "script": str(path),
        **copy.to_dict(),
        "submitted_run": {
            "error": refused.error,
            "return_code": refused.return_code,
            "script_sha256": refused.script_sha256,
            "workdir": refused.workdir,
        },
    }


def copy_failure(
    failed: GmatRunResult | BaseException, label: str, script: Path, copy: Path,
    record: dict[str, Any], *, first: bool,
) -> str:
    """A failed run of the console copy, a result or what run_gmat raised, as the copy's."""
    if isinstance(failed, BaseException):
        said = f"{label}: running it raised {type(failed).__name__}: {failed}"
    else:
        said = (failed.error or label) + _reported(
            tuple(str(error) for error in gmat_errors(failed, _ERROR_LINES).errors
                  if error.kind != "summary")
        )
    shows = (
        "with the plots left out it still fails, so they are not what stops it"
        if first
        else "its first run succeeded, so this run, not the copy, failed"
    )
    return (
        f"{said}. That was a run of {copy.name}, the console copy of {script.name} less "
        f"{', '.join(record['removed'])}, after GmatConsole refused {script.name} for them; "
        f"{shows}"
    )


#: What baseline runs when the official script needs policy.add_epoch_alignment's line.
ALIGNED_BASELINE_NAME = "official_with_alignment.script"


def aligned_baseline_note(ran: str | Path, line: str, *, split: bool) -> str:
    """Why baseline ran ``ran`` and not the official script; ``split``: its first
    Propagate flies each craft on a propagator of its own."""
    as_written = (
        "GMAT runs the official script's split Propagate with both ElapsedSecs equal while "
        "the epochs stay apart, which is not one instant"
        if split
        else "GMAT aborts the official script as written (\"Spacecraft epochs do not match\")"
    )
    return (
        f"baseline ran {Path(ran).resolve()}, the official script with the epoch alignment "
        f"solve and certify add (policy.add_epoch_alignment) as its first mission command, "
        f"{line.strip()!r}: {as_written}. A GMAT GUI run of the zero burn runs this copy "
        "too, from beside the official script's files (docs/RUNBOOK.md step 3)"
    )


def baseline_original_script(
    original_script: str | Path,
    config: OrbitWinConfig,
    gmat_console: str | Path,
    startup_file: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    scenario, document, probe_evidence = load_scenario(
        original_script, config, gmat_console=gmat_console, startup_file=startup_file
    )
    # Before the run, not after: run_gmat writes output_dir/run, so checking
    # afterwards would be checking a directory this call had already filled.
    output = Path(output_dir)
    prepare_empty_output(output, "baseline")
    if probe_evidence is not None:
        scenario = probe_evidence.place(scenario, output / "state_probe")
    # The bytes parsed; run_gmat refuses to run any others.
    before_hash = hashlib.sha256(document.raw_bytes).hexdigest()
    reports = written_reports(scenario)
    # GMAT aborts the script as written; solve and certify add this line, so baseline does.
    added = scenario.alignment if (
        scenario.alignment is not None and scenario.alignment.line_index is None
    ) else None
    ran_path, ran_document, ran_hash = Path(original_script), document, before_hash
    note: str | None = None
    if added is not None:
        ran_path = output / ALIGNED_BASELINE_NAME
        ran_path.write_bytes(with_added_alignment(document, scenario))
        reread, ran_document = parse_gmat_script(ran_path, config, probe=scenario.state_probe)
        if reread.alignment is None or reread.alignment.line_index is None or (
            reread.alignment.text != added.text
        ):  # pragma: no cover - the parser reads back what the renderer wrote
            raise PolicyError(f"{ran_path} does not read back with the alignment {added.text!r}")
        ran_hash = hashlib.sha256(ran_document.raw_bytes).hexdigest()
        split = len(scenario.propagate_segments[0].propagator_names) > 1
        note = aligned_baseline_note(ran_path, added.text, split=split)

    def unchanged(failure: str | None = None, *, ran: bool = True) -> str:
        """The official script's hash, as before the run; refused, with what else went wrong.

        Neither says baseline changed it: run_gmat may have refused before GMAT started.
        """
        also = (f"; the run also failed: {failure}" if ran else f"; baseline also stopped: {failure}"
                ) if failure else ""
        try:
            after = file_sha256(original_script)
        except FileNotFoundError as exc:
            raise PolicyError(f"The official script was moved during baseline: {exc}{also}") from exc
        except OSError as exc:
            raise PolicyError(
                f"Baseline could not re-read the official script: {exc}{also}"
            ) from exc
        if after != before_hash:
            raise PolicyError(f"The official script changed during baseline{also}")
        return after

    def run(script: Path, sha256: str, workdir: str) -> GmatRunResult:
        try:
            return run_gmat(
                gmat_console,
                script,
                output / workdir,
                reports,
                startup_file=startup_file,
                timeout_s=config.gmat.timeout_s,
                template_dir=Path(original_script).resolve().parent,
                expected_sha256=sha256,
            )
        except Exception as exc:
            # A changed official script outranks the run's failure; an interrupt does not wait.
            unchanged(str(exc), ran=False)
            raise

    result = run(ran_path, ran_hash, "run")
    # Custody first; the explanation reads the parsed bytes, not the path again.
    after_hash = unchanged(None if result.success else result.error)
    copied: dict[str, Any] | None = None
    if not result.success:
        try:
            copy_path, proven = console_copy_for(
                ran_document, result, "Original baseline failed", config,
                output / "console_copy", baseline=True,
            )
        except GmatRunError as exc:
            message = str(exc) + (f". The {note}" if note is not None else "")
            raise GmatRunError(message) from None
        copied = copy_record(copy_path, proven, result)
        try:
            result = run(copy_path, proven.copy_sha256, "console-run")
        except Exception as exc:
            raise GmatRunError(copy_failure(
                exc, "Console copy baseline failed", ran_path, copy_path, copied, first=True
            )) from exc
        unchanged(None if result.success else result.error)
        if not result.success:
            raise GmatRunError(copy_failure(
                result, "Console copy baseline failed", ran_path, copy_path, copied, first=True
            ))
        ran_path, ran_hash = copy_path, proven.copy_sha256
    if result.script_sha256 != ran_hash:
        raise GmatRunError(
            f"Baseline ran {result.script_sha256}, not the official script {ran_hash}"
            if added is None and copied is None
            else f"Baseline ran {result.script_sha256}, not {ran_path} {ran_hash}"
        )
    evidence = {
        "original_script": str(Path(original_script).resolve()),
        "original_sha256_before": before_hash,
        "original_sha256_after": after_hash,
        # None: GMAT ran the official script itself.
        "epoch_alignment_added": None if added is None else added.text,
        "epoch_alignment_note": note,
        "ran_script": str(ran_path.resolve()),
        "ran_sha256": ran_hash,
        "scenario": scenario.to_dict(),
        # None: GMAT ran the script above it; else the proven copy less its plots,
        # which ran_script then names (gmat.console_copy).
        "console_copy": copied,
        "run": result.to_dict(),
    }
    (output / "baseline.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return evidence


def _replay_with_diagnostic(
    final_script: str | Path,
    config: OrbitWinConfig,
    gmat_console: str | Path,
    startup_file: str | Path | None,
    workdir: str | Path,
    pursuer_name: str,
    target_name: str,
    probe: StateProbe | None,
    template_dir: Path,
) -> tuple[Any, GmatRunResult]:
    """Replay the *exact final* script with a diagnostic sibling attached.

    The sibling differs from the submitted bytes by two extra ReportFiles, a
    Report command on each side of every Maneuver and, where the script draws,
    by having its plot subscribers commented out.
    Running it lets certification state two things about the script that is
    actually submitted rather than about an earlier render of it: that it still
    enters the sphere, and -- by comparing the official report against the exact
    runs -- that neither change perturbed the propagation.

    That second claim is measured, not assumed. On a script that both runs in
    console and draws, the official report comes out byte-identical between the
    plain run and the sibling: 159265 bytes either way. A script whose plots
    stop console running it at all cannot be compared that way, because there
    is no plain run to compare against; certification then rests on the exact
    reruns, which use the submitted bytes. Under gmat.console_copy they run the
    proven copy less the plots instead, and the comparison is against that.
    """
    scenario, document = parse_gmat_script(final_script, config, probe=probe)
    diagnostic = add_diagnostic_report(
        document, scenario, config.gmat.diagnostic_report_name, check_burns=True
    )
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    sibling = root / "final_diagnostic_sibling.script"
    sibling.write_bytes(diagnostic.script_bytes)
    report_names = written_reports(scenario)
    if diagnostic.burn_report_filename is not None:
        report_names.append(diagnostic.burn_report_filename)
    report_names.append(diagnostic.report_filename)
    result = run_gmat(
        gmat_console,
        sibling,
        root / "run",
        report_names,
        startup_file=startup_file,
        timeout_s=config.gmat.timeout_s,
        file_size_limit_bytes=file_size_limit_for(scenario),
        template_dir=template_dir,
    )
    if not result.success:
        raise GmatRunError(
            (result.error or "Final-script diagnostic replay failed") + said_by_gmat(result))
    maneuvers = maneuvers_flown(scenario)
    if diagnostic.burn_report_filename is not None:
        check_flown_impulses(
            result.reports.get(diagnostic.burn_report_filename), maneuvers, complete=True
        )
    event = evaluate_diagnostic_report(
        result.reports[diagnostic.report_filename],
        pursuer_name,
        target_name,
        config.policy.intercept_radius_km,
        central_body_radius_km=config.policy.central_body_radius_km,
        maneuvers=maneuvers,
        alignment=scenario.alignment,
        geodetic_radius_km=scenario.geodetic_radius_km,
    )
    return event, result


def _row_count(path: str | Path, *, headers: bool) -> int:
    """Data rows of a report: its non-empty lines, less the header GMAT writes.

    Not "lines that are all numbers": a UTCGregorian column is words. Nor a
    header guessed from its tokens: a Variable's column is headed by a bare name.
    """
    lines = [
        line for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return max(0, len(lines) - (1 if headers else 0))


def _check_archive_stamps(
    candidate: Candidate, scenario: Scenario, config: OrbitWinConfig
) -> tuple[str, ...]:
    """Refuse a candidate stamped for another script or horizon; warn if unstamped,
    if it was verified at another stop depth, or searched under another score block.

    rank checks the same stamps; a candidate handed straight to certify did not
    pass through rank. A failed candidate's time term is T_max, and the script
    is what the burns were planned against.
    """
    stamp = candidate.metadata.get("scenario_sha256")
    horizon = candidate.metadata.get("horizon_s")
    mismatch = clock_stamp_mismatch(candidate.metadata, scenario)
    if mismatch is not None:
        raise PolicyError(
            f"Candidate {candidate.candidate_id} was solved on another clock: {mismatch}"
        )
    if stamp and str(stamp) != scenario.source_sha256:
        raise PolicyError(
            f"Candidate {candidate.candidate_id} was produced from a script hashing to {stamp}; "
            f"{scenario.source_path} hashes to {scenario.source_sha256}"
        )
    if horizon is not None and not abs(float(horizon) - scenario.max_time_s) <= 1.0e-6:
        # 4 T_A follows the initial state, which certify reads through the probe.
        hint = (
            "; this 4 T_A is from GMAT's probed state, so solve with --gmat-console as well"
            if config.policy.horizon_s is None and scenario.state_probe is not None
            else ""
        )
        raise PolicyError(
            f"Candidate {candidate.candidate_id} was solved under a horizon of "
            f"{float(horizon):.6f} s and is being certified under {scenario.max_time_s:.6f} s"
            f"{hint}"
        )
    messages: list[str] = []
    missing = [name for name, value in (("scenario_sha256", stamp), ("horizon_s", horizon))
               if not value and value != 0]
    if missing:
        messages.append(
            f"Candidate {candidate.candidate_id} carries no {' or '.join(missing)}, so nothing "
            "shows it was produced from this script and horizon"
        )
    depth = stop_depth_mismatch(candidate, config)
    if depth:
        messages.append(f"{depth}; this certification measures it again")
    if "searched_score" not in candidate.metadata:
        messages.append(
            f"Candidate {candidate.candidate_id} carries no searched_score, so nothing shows "
            "its search aimed at these coefficients; rank says the same"
        )
    else:
        difference = score_stamp_difference(candidate.metadata["searched_score"], config.score)
        if difference is not None:
            messages.append(
                f"Candidate {candidate.candidate_id} was searched under {difference[0]} and is "
                f"certified under {difference[1]}: the search optimised for other coefficients; "
                "search again under these (docs/RUNBOOK.md step 5)"
            )
    for message in messages:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    return tuple(messages)


def _assert_one_set_of_data_files(
    runs: list[GmatRunResult], probe_run: GmatRunResult | None
) -> None:
    """Every run read the same files with the same hashes.

    The probe copy drops the ForceModel, so its files may be a subset of the
    others'. Any other run naming a different set lost or gained one.
    """
    key_sets = {frozenset(run.data_file_sha256) for run in runs}
    common = frozenset.intersection(*key_sets)
    uneven = frozenset.union(*key_sets) - common
    if probe_run is not None:
        uneven |= frozenset(probe_run.data_file_sha256) - common
    read: dict[str, set[str]] = {}
    for run in [*runs, *([probe_run] if probe_run is not None else [])]:
        for key, entry in run.data_file_sha256.items():
            read.setdefault(key, set()).add(json.dumps(entry, sort_keys=True))
    changed = sorted(uneven | {key for key, entries in read.items() if len(entries) > 1})
    if changed:
        raise GmatRunError(
            f"GMAT's data files changed between the runs behind this certificate "
            f"({', '.join(changed)}); certify again once they are settled"
        )


#: Why a certificate's score and d_min come from its diagnostic flight.
NO_TARGET_PROOF = (
    "the official report carries no {target} position; d_min and the interception were proven "
    "on certify's diagnostic flight of the same script, not on this report"
)


def official_rows_record(
    path: str, spec: ReportFileSpec, scenario: Scenario, config: OrbitWinConfig,
    verified: Candidate, *, frozen_rows: int, t_team_s: float, hit: bool,
    completion_s: float | None, diagnostic_distance_km: float | None,
) -> dict[str, Any]:
    """The official report as uploaded, on its printed digits, as step 7 reads it.

    Refuses a report whose printed rows say other than the certificate: a last
    row away from T_team, a hit's last row outside the sphere or short of the
    stop depth, a miss with a row inside. The score is the printed rows': T_team
    off the last row, a miss's d_min off the rows; without the target's
    position, d_min is the diagnostic flight's (NO_TARGET_PROOF).
    """
    pursuer, target = scenario.pursuer.name, scenario.target.name
    apart = scenario.alignment is not None and scenario.alignment.offset_s > 0.0
    origin = scenario.time_origin_craft
    crafts = ([origin] if origin else []) + ([] if apart else [pursuer, target])
    rows = read_official_rows(path, spec, pursuer, target, frozen_rows=frozen_rows,
                              time_columns=[f"{craft}.ElapsedSecs" for craft in crafts])
    radius_km = config.policy.intercept_radius_km
    last_s = rows.last_row_time_s
    if last_s is not None and not (
        same_printed_value(last_s, t_team_s, spec.precision) if hit
        else abs(last_s - t_team_s) <= max(1.0e-3, last_digit(spec.precision, abs(t_team_s)))
    ):
        raise GmatRunError(
            f"The official report's last row is at {last_s!r} s on {rows.time_column}, and "
            f"T_team is {t_team_s!r} s, {t_team_s:.{spec.precision}g} at its Precision "
            f"{spec.precision}: step 7 would refuse it"
        )
    depth_km = None
    if rows.last_row_distance_km is not None and rows.minimum_row_distance_km is not None:
        printed = f"at Precision {spec.precision}, its rows printed to "
        if not hit and rows.minimum_row_distance_km <= radius_km:
            raise GmatRunError(
                f"GMAT flew a miss, and the official report's rows, {printed}"
                f"{rows.minimum_row_distance_km:.6g} km, read as a hit: step 7 would refuse it"
            )
        if hit:
            depth_km = radius_km - rows.last_row_distance_km
            asked_km = stop_depth_used_km(config.policy.stop_depth_km, radius_km)
            if depth_km < 0.0 or (
                config.policy.mission_end_policy != "horizon"
                and depth_km < asked_km * (1.0 - STOP_DEPTH_TOLERANCE)
                and not config.policy.allow_shallow_stop_submission
            ):
                raise PolicyError(
                    f"The official report's last row is {depth_km * 1000.0:.4g} m inside the "
                    f"{radius_km:g} km sphere {printed}{rows.last_row_distance_km:.6g} km, short "
                    f"of policy.stop_depth_km = {asked_km * 1000.0:.4g} m: step 7 would refuse "
                    "it. Ask the organizer for a finer Precision"
                )
    flown = replace(
        verified,
        intercept_time_s=(last_s - scenario.time_origin_offset_s
                          if hit and last_s is not None else completion_s),
        minimum_distance_km=(rows.minimum_row_distance_km
                             if rows.minimum_row_distance_km is not None
                             else diagnostic_distance_km),
        candidate_id="",
    )
    if rows.alignment_minimum_row_distance_km is not None and uses_head_start(verified.burns):
        # Its own alignment rows, as printed, where head-start burns moved it.
        flown.metadata = {**verified.metadata,
                          HEAD_START_ALIGNMENT_KEY: rows.alignment_minimum_row_distance_km}
    priced = scenario
    if (rows.alignment_minimum_row_distance_km is not None
            and scenario.alignment_minimum_distance_km is not None):
        priced = replace(scenario,
                         alignment_minimum_distance_km=rows.alignment_minimum_row_distance_km)
    on_rows = rows.carries_target_position and rows.minimum_row_distance_km is not None
    return {
        **rows.to_dict(),
        "stop_depth_km": depth_km,
        "scored_distance_km": scored_distance_km(flown, priced),
        "score": (score_candidate(flown, priced, config.score).raw_score
                  if config.score is not None else None),
        "proof": ("the official report's rows" if on_rows
                  else NO_TARGET_PROOF.format(target=target) if not rows.carries_target_position
                  else "certify's diagnostic flight of the same script"),
    }


def _logged_errors(
    verification: dict[str, Any], runs: list[GmatRunResult]
) -> dict[str, list[str]]:
    """The errors GMAT logged in verification's flights and in ``runs``, by run id."""
    logged = {str(run_id): list(records)
              for run_id, records in (verification.get("gmat_log_errors") or {}).items()}
    for run in runs:
        if run.log_errors:
            logged[run.run_id] = list(run.log_errors)
    return logged


def planned_hit(candidate: Candidate) -> bool:
    """Whether the archive plans this candidate as a hit.

    solve clears `successful` when its GMAT verification raises (a timeout, a
    crash) and keeps the plan in `analytical_successful`.
    """
    return candidate.successful or candidate.metadata.get("analytical_successful") is True


#: A loss certify takes without --accept-gmat-loss. Across 8284 GMAT-verified
#: rows in this workstation's archives the plan and GMAT's flight differed by at
#: most 0.045 points (2026-11-05).
GMAT_LOSS_TOLERANCE_POINTS = 0.5
#: A loss certify reports: a GMAT re-flight of a verified row is bit-identical, so
#: anything above this there is data or build drift.
GMAT_LOSS_REPORTED_POINTS = 0.05
#: Without coefficients: a miss GMAT flies this much wider is refused.
#: In A3's sensitivity runs (2026-11-04) data files moved a flown row's distance by
#: at most 15 m (SPAD sign) and 9 m (F10.7 x1.5); the pursuer model's plans sat
#: 0.06-0.17 km from GMAT on R1 and R6. A plan flown 0.1 km wider is past both.
GMAT_WIDER_MISS_TOLERANCE_KM = 0.1
#: A row solve verified flies bit-identically: any widening past this is drift.
VERIFIED_WIDER_MISS_REPORTED_KM = 1.0e-6
#: A GMAT entry this much later than the archive's is reported, and refused
#: without coefficients to price it.
LATE_ENTRY_S = 60.0


@dataclass(frozen=True)
class GmatComparison:
    """The archive's claim against GMAT's flight, as certification.json records it."""

    #: What each side is priced on: "completion" (a hit solve verified, and every
    #: hit GMAT flies: the submitted script's last row), "first entry" (a plan GMAT
    #: never flew) or "closest approach" (a miss).
    planned_basis: str
    flown_basis: str
    planned_score: float | None
    #: The submitted script's, as certify replays it.
    flown_score: float | None
    loss_points: float | None
    #: A plan against GMAT's first entry, before the stop depth: information only.
    entry_loss_points: float | None
    planned_entry_s: float | None
    flown_entry_s: float | None
    late_entry_s: float | None
    planned_closest_km: float | None
    flown_closest_km: float | None
    flipped: bool
    refused_without_opt_in: bool
    message: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_with_archive(
    archived: Candidate, verified: Candidate, scenario: Scenario, config: OrbitWinConfig
) -> GmatComparison:
    """How the script certify would submit compares with the archive's claim, and
    whether certify refuses it.

    ``archived`` is the candidate as read, before verification rewrote it.
    ``verified`` carries what the submitted script scores: its completion, stop
    depth included, which certify's replay must reproduce.
    """
    hit = planned_hit(archived)
    archived_event = archived.metadata.get("gmat_event") or {}
    verified_row = gmat_verified_on_record(archived)
    planned_basis = (
        "closest approach" if not hit
        else "completion" if archived_event.get("first_entry_time_s") is not None
        else "first entry"
    )
    flown_basis = "completion" if verified.successful else "closest approach"
    planned_entry_s = (
        archived_event.get("first_entry_time_s") or archived.intercept_time_s if hit else None
    )
    flown_event = verified.metadata["gmat_event"]
    flown_entry_s = flown_event.get("first_entry_time_s")
    flipped = hit and not verified.successful
    late_s = (
        flown_entry_s - planned_entry_s
        if hit and verified.successful and planned_entry_s is not None
        and flown_entry_s is not None and flown_entry_s - planned_entry_s > LATE_ENTRY_S
        else None
    )
    planned_closest = None if hit else archived.minimum_distance_km
    flown_closest = None if verified.successful else verified.minimum_distance_km
    widening = (
        flown_closest - planned_closest
        if planned_closest is not None and flown_closest is not None else None
    )
    planned_score = flown_score = loss = entry_loss = None
    if config.score is not None:
        # Both under this config's coefficients and penalties: the burns are the same.
        count = len(verified.penalties)
        plan = replace(archived, successful=hit, metadata=dict(archived.metadata))
        planned_score = score_candidate(plan, scenario, config.score, penalty_count=count).raw_score
        flown_score = score_candidate(verified, scenario, config.score,
                                      penalty_count=count).raw_score
        loss = planned_score - flown_score
        at_entry = ""
        if planned_basis == "first entry" and verified.successful:
            entering = replace(verified, intercept_time_s=flown_entry_s,
                               metadata=dict(verified.metadata))
            entry_loss = planned_score - score_candidate(
                entering, scenario, config.score, penalty_count=count).raw_score
            at_entry = (f" (at GMAT's first entry, before the stop depth, "
                        f"{planned_score - entry_loss:.3f})")
        refuse = flipped or loss > GMAT_LOSS_TOLERANCE_POINTS
        report = refuse or late_s is not None or loss > GMAT_LOSS_REPORTED_POINTS
        priced = (f" Under this config's coefficients the archive's claim is "
                  f"{planned_score:.3f} points and the script certify would submit "
                  f"{flown_score:.3f}{at_entry}.")
    else:
        wider = widening is not None and widening > GMAT_WIDER_MISS_TOLERANCE_KM
        drifted = (
            widening is not None and verified_row and widening > VERIFIED_WIDER_MISS_REPORTED_KM
        )
        refuse = flipped or wider or late_s is not None
        report = refuse or drifted
        priced = (" No coefficients are set in this config, so neither is priced; certify "
                  f"refuses a hit flown as a miss, an entry over {LATE_ENTRY_S:g} s late, or a "
                  f"miss flown more than {GMAT_WIDER_MISS_TOLERANCE_KM:g} km wider.")
    record = dict(
        planned_basis=planned_basis, flown_basis=flown_basis, planned_score=planned_score,
        flown_score=flown_score, loss_points=loss, entry_loss_points=entry_loss,
        planned_entry_s=planned_entry_s, flown_entry_s=flown_entry_s, late_entry_s=late_s,
        planned_closest_km=planned_closest, flown_closest_km=flown_closest, flipped=flipped,
        refused_without_opt_in=refuse,
    )
    if not report:
        return GmatComparison(**record, message=None)

    def at(value: float | None, unit: str) -> str:
        return "an unrecorded time" if value is None else f"{value:.3f} {unit}"

    radius = f"{config.policy.intercept_radius_km:g} km sphere"
    if hit:
        claim = f"a hit, first inside the {radius} at {at(planned_entry_s, 's')}"
        if planned_basis == "completion":
            claim += f" and completing at {at(archived.intercept_time_s, 's')}"
    else:
        claim = f"a miss, {at(archived.minimum_distance_km, 'km')} from the target at closest"
    if verified.successful:
        flown = (
            f"a hit first inside at {at(flown_entry_s, 's')} and completing at "
            f"{at(verified.intercept_time_s, 's')}"
        )
    else:
        flown = (
            f"a miss, closest approach {at(verified.minimum_distance_km, 'km')} at "
            f"{at(flown_event.get('minimum_distance_time_s'), 's')}"
        )
    failed = archived.metadata.get("gmat_verification_error")
    if verified_row:
        why = (
            " Solve's GMAT verification flew it as the archive says, so GMAT's data files or "
            "build changed between the two flights: compare data_file_sha256 in the two "
            "verification.json files."
        )
    elif failed:
        why = f" Solve's GMAT verification of it failed ({failed}); certify a submittable row."
    else:
        why = " Solve never flew it in GMAT; certify a row rank lists as submittable."
    message = (
        f"Candidate {archived.candidate_id}: the archive holds {claim}. GMAT now flies "
        f"{flown}.{priced}{why}"
    )
    return GmatComparison(**record, message=message)


def _set_aside(root: Path, kind: str = "refused") -> Path:
    """Move a refused certification, or one that got no answer, out of the way, so the
    same --output works again."""
    for index in itertools.count(1):
        kept = root.with_name(f"{root.name}.{kind}" + ("" if index == 1 else f"-{index}"))
        if not os.path.lexists(kept):
            root.rename(kept)
            return kept
    raise AssertionError("unreachable")


def _clause(reason: str, rules: InterceptionRules) -> str:
    """``reason`` from validate_candidate, with the clause it breaks and what that costs."""
    deduct = f"-{rules.penalty_points:g} points per maneuver"
    if "exceeds formal limit" in reason:
        cost = "disqualifies" if rules.over_impulse_limit == "disqualify" else deduct
        return f"{reason} ({rules.impulse_clause}: {cost})"
    if reason.startswith("total delta-v"):
        return f"{reason} ({rules.total_clause or 'policy.total_dv_limit_km_s'}: disqualifies)"
    if " separation " in reason:
        return f"{reason} (the {rules.min_separation_s:g} s gap: {deduct})"
    return reason


def rule_breach(candidate: Candidate, scenario: Scenario, config: OrbitWinConfig) -> str | None:
    """Why certify refuses a candidate that breaks the rules, or None when it breaks none."""
    checked = validate_candidate(candidate, scenario, config)
    if checked.valid:
        return None
    rules = interception_rules(scenario.rules)
    return (
        f"Candidate {candidate.candidate_id} breaks the {rules.name} rules "
        f"({rules.source.header_date}): "
        + "; ".join(_clause(reason, rules) for reason in checked.violations)
        + ". It is on neither frontier, and policy.allow_formal_frontier_submission does not "
        "take it."
    )


def certify_candidate(
    original_script: str | Path,
    candidate: Candidate,
    config: OrbitWinConfig,
    gmat_console: str | Path,
    startup_file: str | Path | None,
    output_dir: str | Path,
    *,
    runs: int | None = None,
    accept_gmat_loss: bool = False,
) -> dict[str, Any]:
    """Certify ``candidate``; a refusal after the output exists moves it to <output>.refused,
    and no answer (gmat_runner.no_answer) to <output>.no-answer."""
    root = Path(output_dir)
    fresh = not root.exists() or (root.is_dir() and not any(root.iterdir()))
    archive_notes: list[str] = []
    try:
        return _certify(original_script, candidate, config, gmat_console, startup_file, root,
                        runs=runs, accept_gmat_loss=accept_gmat_loss,
                        archive_notes=archive_notes)
    except Exception as exc:
        if not (fresh and root.is_dir() and any(root.iterdir())):
            raise
        # Absolute, so the paths work from any directory (docs/RUNBOOK.md step 6).
        kept = Path(os.path.abspath(
            _set_aside(root, "no-answer" if no_answer(exc) else "refused")))
        files = [kept / "staging" / "final.script", kept / "diagnostic" / "verification.json"]
        named = ", ".join(str(path) for path in files if path.is_file())
        note = (
            f"This certification is kept in {kept}" + (f" ({named})" if named else "")
            + f", and {os.path.abspath(root)} is free again; a certify stopped with Ctrl-C is "
            "not moved, so give the next one a fresh --output."
        )
        if archive_notes:
            # Warned at the start, long scrolled away by the time of the plots stop.
            note += " Archive warnings, again: " + "; ".join(archive_notes) + "."
        if isinstance(exc, OrbitWinError):
            text = str(exc)
            refused = type(exc)(f"{text}{'' if text.endswith('.') else '.'} {note}")
            # The CLI names the GUI path's pre-check from it (cli.gui_precheck).
            refused.__dict__["kept_dir"] = kept
            raise refused from exc
        exc.__dict__["kept_dir"] = kept
        exc.add_note(note)
        raise


def _certify(
    original_script: str | Path,
    candidate: Candidate,
    config: OrbitWinConfig,
    gmat_console: str | Path,
    startup_file: str | Path | None,
    output_dir: str | Path,
    *,
    runs: int | None = None,
    accept_gmat_loss: bool = False,
    archive_notes: list[str] | None = None,
) -> dict[str, Any]:
    validate_config(config)  # a config built in code never went through load_config
    # The archive's claim; everything below rewrites the candidate in place.
    archived = copy.deepcopy(candidate)
    # Before anything is written: a refusal later leaves an output directory behind.
    rerun_count = config.gmat.reruns if runs is None else runs
    if rerun_count < 1:
        raise ValueError("Certification requires at least one exact run")
    # In a temporary directory: every refusal below must leave nothing behind.
    scenario, document, probe_evidence = load_scenario(
        original_script, config, gmat_console=gmat_console, startup_file=startup_file
    )
    # The submission path has to apply the same frontier gate as solve(),
    # otherwise certify becomes the way around it: certify is handed a candidate
    # id from candidates.json, and nothing else here would have looked at which
    # frontier it came from.
    if not written_reports(scenario):
        # The competition asks for the modified script *and* its report. With no
        # ReportFile there is nothing to upload and nothing for the byte-parity
        # comparison to compare, so certification would pass on no evidence and
        # hand back a submission directory containing no report at all.
        raise PolicyError(
            "The official script declares no ReportFile GMAT writes. There would be no report "
            "to submit and no bytes to compare, so this cannot be certified."
        )
    assert_team_number_for_submission(scenario, config)
    # Hashed beside the data files: a non-stock .spo flies drag differently (GMT-6588).
    spad_record, spad_notes = spad_check(scenario, document, gmat_console, startup_file)
    for note in spad_notes:
        warnings.warn(note, RuntimeWarning, stacklevel=3)
    report_name_warnings = team_number_warnings(scenario, config)
    for message in report_name_warnings:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    # Before the frontier: a plan that breaks a rule is on neither, and was called legal.
    breach = rule_breach(candidate, scenario, config)
    if breach is not None:
        raise PolicyError(breach)
    frontier = frontier_of(candidate, config)
    if frontier != ROBUST_FRONTIER and not config.policy.allow_formal_frontier_submission:
        raise PolicyError(
            f"Candidate {candidate.candidate_id} is on the {FORMAL_FRONTIER} frontier: legal "
            f"under the rules but outside the production margins of "
            f"{config.policy.dv_safe_limit_km_s:.6g} km/s and "
            f"{config.policy.safe_burn_separation_s:.6g} s. Set "
            "policy.allow_formal_frontier_submission to certify it anyway."
        )
    archive_warnings = _check_archive_stamps(candidate, scenario, config)
    if archive_notes is not None:
        archive_notes.extend(archive_warnings)
    # rank lists this as a blocker without filtering; certify is the gate.
    contradicted = gmat_contradicts_hit(candidate, config)
    if contradicted and not accept_gmat_loss:
        raise PolicyError(
            f"Candidate {candidate.candidate_id}: {contradicted}, or certify it as GMAT flies "
            "it with --accept-gmat-loss"
        )
    if contradicted:
        warnings.warn(f"Candidate {candidate.candidate_id}: {contradicted}; flying it again "
                      "under --accept-gmat-loss", RuntimeWarning, stacklevel=3)
    # Rendered once before anything is created: a refusal here, a burn written
    # in another frame among them, must not leave an output directory behind.
    # Without the archived stop, as verification will first fly it.
    candidate.metadata.pop("gmat_event", None)
    render_candidate(document, scenario, candidate, config)
    root = Path(output_dir)
    prepare_empty_output(root, "certify")
    if probe_evidence is not None:
        scenario = probe_evidence.place(scenario, root / "state_probe")
    verified, diagnostic_run, diagnostic_evidence = verify_candidate_with_gmat(
        original_script,
        scenario,
        document,
        candidate,
        config,
        gmat_console,
        startup_file,
        root / "diagnostic",
    )
    # A scored miss, or a later hit, passes every gate below.
    comparison = compare_with_archive(archived, verified, scenario, config)
    if comparison.refused_without_opt_in and not accept_gmat_loss:
        raise PolicyError(
            f"{comparison.message} To certify it as GMAT flies it, run the same command with "
            "--accept-gmat-loss."
        )
    if comparison.message is not None:
        then = (
            "Under --accept-gmat-loss it will be certified as GMAT flies it if the remaining "
            "checks pass." if comparison.refused_without_opt_in else "Certify goes on."
        )
        warnings.warn(f"{comparison.message} {then}", RuntimeWarning, stacklevel=3)
    if not verified.verified_by_gmat:
        raise GmatRunError("Candidate did not enter the 5 km sphere in GMAT")
    # Measured by this run; the archive's own depth is not trusted.
    shallow = shallow_stop(verified, config)
    if shallow and not config.policy.allow_shallow_stop_submission:
        raise PolicyError(
            f"{shallow}. Set policy.allow_shallow_stop_submission to certify it anyway."
        )
    if shallow:
        warnings.warn(f"{shallow}; certified under policy.allow_shallow_stop_submission",
                      RuntimeWarning, stacklevel=3)

    # Build in staging. The submission directory is created only once every
    # check has passed, so a directory named "submission" cannot contain the
    # leftovers of a certification that failed halfway through.
    staging = root / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    final_script = staging / "final.script"
    final_diff = staging / "final.script.diff.txt"
    write_rendered_candidate(final_script, final_diff, document, scenario, verified, config)
    assert_final_script_policy(original_script, final_script, config, probe=scenario.state_probe)
    flown_sha256 = verified.metadata["gmat_event"]["flown_script_sha256"]
    if file_sha256(final_script) != flown_sha256:
        raise GmatRunError(
            f"The final script {file_sha256(final_script)} is not the script GMAT verified "
            f"({flown_sha256})"
        )

    # The names the final script writes, team number included.
    final_scenario, final_document = parse_gmat_script(
        final_script, config, probe=scenario.state_probe
    )
    submitted_reports = [spec for spec in final_scenario.report_files if spec.write_report]
    exact_runs: list[GmatRunResult] = []
    official_reports = written_reports(final_scenario)

    def run(script: Path, sha256: str, workdir: str) -> GmatRunResult:
        return run_gmat(
            gmat_console,
            script,
            root / workdir,
            official_reports,
            startup_file=startup_file,
            timeout_s=config.gmat.timeout_s,
            file_size_limit_bytes=file_size_limit_for(scenario),
            template_dir=Path(original_script).resolve().parent,
            expected_sha256=sha256,
        )

    # The bytes the policy guard and the flown-hash check approved; under
    # gmat.console_copy, once GmatConsole refuses them for their plots alone,
    # their proven copy less those plots.
    ran_script, ran_sha256 = final_script, flown_sha256
    console: dict[str, Any] | None = None
    for index in range(1, rerun_count + 1):
        label = f"Exact final run {index} failed"
        if console is None:
            result = run(final_script, flown_sha256, f"exact-run-{index:02d}")
            if result.success:
                exact_runs.append(result)
                continue
            if index > 1:
                raise GmatRunError(refusal_message(
                    final_document, result, label,
                    kept_as_written=config.policy.report_filename_as_written,
                ))
            copy_path, proven = console_copy_for(
                final_document, result, label, config, root / "console_copy",
                kept_as_written=config.policy.report_filename_as_written,
            )
            console = copy_record(copy_path, proven, result)
            ran_script, ran_sha256 = copy_path, proven.copy_sha256
        assert console is not None
        try:
            result = run(ran_script, ran_sha256, f"console-exact-run-{index:02d}")
        except Exception as exc:
            raise GmatRunError(copy_failure(exc, label, final_script, ran_script, console,
                                            first=index == 1)) from exc
        if not result.success:
            raise GmatRunError(copy_failure(result, label, final_script, ran_script, console,
                                            first=index == 1))
        exact_runs.append(result)

    # Chain of custody. The run results carry the hash of the script each run
    # actually executed, and nothing was comparing it: certification could say
    # "these bytes intercept" and "these bytes are the submission" without ever
    # linking the two. A console copy is linked to them by its proof.
    final_hash = file_sha256(final_script)
    if console is not None and console["submitted_sha256"] != final_hash:
        raise GmatRunError(
            f"The console copy was proven against {console['submitted_sha256']}, but the "
            f"script being certified is {final_hash}"
        )
    for index, result in enumerate(exact_runs, 1):
        if result.script_sha256 != (final_hash if console is None else ran_sha256):
            raise GmatRunError(
                f"Exact run {index} executed {result.script_sha256} but the script being "
                f"certified is {final_hash}"
                + ("" if console is None else f", run as its console copy {ran_sha256}")
            )

    if config.gmat.compare_report_bytes and official_reports:
        for report_name in official_reports:
            hashes = {result.report_sha256[report_name] for result in exact_runs}
            if len(hashes) != 1:
                raise GmatRunError(
                    f"Official report {report_name} is not byte-reproducible across exact runs: {hashes}"
                )

    # Interception was proven on a sibling of an earlier render. Prove it again
    # against the bytes actually being submitted, and show -- rather than assume
    # -- that the diagnostic subscriber leaves the official report untouched.
    final_event, final_sibling_run = _replay_with_diagnostic(
        final_script,
        config,
        gmat_console,
        startup_file,
        root / "final-parity",
        scenario.pursuer.name,
        scenario.target.name,
        scenario.state_probe,
        Path(original_script).resolve().parent,
    )
    replayed_hit = (
        final_event.first_entry_time_s is not None
        and final_event.first_entry_time_s <= scenario.max_time_s + 1.0e-8
    )
    if verified.successful and not replayed_hit:
        raise GmatRunError("The exact final script does not enter the intercept sphere on replay")
    if replayed_hit and not verified.successful:
        raise GmatRunError("The exact final script enters the sphere on replay; verification measured a miss")
    frozen_rows = report_clock(
        parse_numeric_report(final_sibling_run.reports[config.gmat.diagnostic_report_name]),
        scenario.pursuer.name, scenario.target.name, scenario.alignment,
    ).frozen_rows
    # The sibling comments out the plots, and an official report counts every
    # alignment row too; one more or fewer and the last row is another one.
    # frozen_rows is read off the sibling's diagnostic report; the official one
    # must have the same rows for it to say anything about the upload.
    diagnostic_rows = _row_count(
        final_sibling_run.reports[config.gmat.diagnostic_report_name], headers=True
    )
    # Which runs the official reports below came from.
    ran = "the final script" if console is None else f"its console copy {ran_script.name}"
    arrays = {
        name for name, kind in created_types(document).items() if kind.lower() == "array"
    }
    for spec in submitted_reports:
        report_name = spec.filename
        if any(parameter in arrays for parameter in spec.parameters):
            # A whole Array is written on as many lines as it has rows, so lines
            # are not rows; the byte comparison below still holds the reruns together.
            continue
        rows = [
            _row_count(run.reports[report_name], headers=spec.write_headers)
            for run in (final_sibling_run, exact_runs[-1])
        ]
        if rows[0] != rows[1] or rows[0] != diagnostic_rows or not rows[0]:
            raise GmatRunError(
                f"Official report {report_name} has {rows[1]} rows from {ran} and "
                f"{rows[0]} from its diagnostic sibling, whose diagnostic report has "
                f"{diagnostic_rows}"
            )
    if config.gmat.compare_report_bytes:
        for report_name in official_reports:
            sibling_hash = final_sibling_run.report_sha256.get(report_name)
            exact_hash = exact_runs[-1].report_sha256[report_name]
            if sibling_hash != exact_hash:
                raise GmatRunError(
                    f"The diagnostic sibling changed official report {report_name} against "
                    f"{ran}: {sibling_hash} != {exact_hash}"
                )

    # The certificate has to describe the submitted bytes. They are the bytes
    # verification flew, so the score inputs must come back unchanged.
    replayed = (
        final_event.first_entry_time_s,
        _completion_time_s(final_event, replayed_hit),
        _closest_approach(final_event, replayed_hit)[0],
        # Unscored for a miss, but it still catches a replay read without its burns (#106).
        final_event.minimum_distance_km,
    )
    claimed = (
        verified.metadata["gmat_event"]["first_entry_time_s"],
        verified.intercept_time_s,
        verified.minimum_distance_km,
        verified.metadata["gmat_event"]["interpolated_minimum_distance_km"],
    )
    if any(
        (a is None) != (b is None) or (a is not None and b is not None and abs(a - b) > 1.0e-6)
        for a, b in zip(replayed, claimed, strict=True)
    ):
        raise GmatRunError(
            f"Verification measured entry, completion and closest approach (scored, "
            f"interpolated) {claimed!r}, but "
            f"replaying the exact final script gives {replayed!r}. The certificate would "
            "describe a run other than the submitted bytes."
        )
    replay_score: float | None = None
    diagnostic_score: float | None = None
    if config.score is not None:
        replay_score = diagnostic_score = score_candidate(
            replace(
                verified,
                intercept_time_s=replayed[1],
                minimum_distance_km=replayed[2],
                candidate_id="",
            ),
            scenario,
            config.score,
        ).raw_score
    if verified.successful and final_event.distance_at_end_km > config.policy.intercept_radius_km:
        # The validator takes the completion time from the report's last row
        # (2026-09-05, 22 points).
        reason = (
            "; policy.mission_end_policy is horizon, which flies a hit on to T_max"
            if config.policy.mission_end_policy == "horizon"
            else ""
        )
        raise GmatRunError(
            f"The final script's last report row is {final_event.distance_at_end_km:.6g} km "
            f"from the target, outside the {config.policy.intercept_radius_km:.6g} km sphere; "
            f"a validator reading the last row would see no interception{reason}"
        )
    # T_team on the day's clock, from the earlier epoch (§3); a miss completes at T_max (#99).
    t_team_s = (
        scenario.max_time_s if replayed[1] is None else float(replayed[1])
    ) + scenario.time_origin_offset_s
    # The uploaded report as printed: step 7 reads T_team and a hit's depth off its last row.
    official_spec = official_report_of(final_scenario.report_files)
    official_rows = None
    # The test double writes the row number in every column of an official report.
    if (official_spec is not None and official_spec.filename in exact_runs[-1].reports
            and not is_test_double(gmat_console)):
        official_rows = official_rows_record(
            exact_runs[-1].reports[official_spec.filename], official_spec, scenario, config,
            verified, frozen_rows=frozen_rows, t_team_s=t_team_s, hit=replayed_hit,
            completion_s=replayed[1], diagnostic_distance_km=replayed[2],
        )
    if official_rows is not None:
        replay_score = official_rows["score"]
    # The replay of the bytes being submitted, which for a miss runs to T_max.
    # solve refuses to nominate such a candidate, but certify writes the
    # submission and is reachable by naming the candidate directly.
    radius_km = min(
        final_event.minimum_pursuer_radius_km or float("inf"),
        float(verified.metadata.get("minimum_pursuer_radius_km", float("inf"))),
    )
    if (
        not config.policy.allow_body_intersecting_submission
        and radius_km < config.policy.central_body_radius_km
    ):
        rules = interception_rules(config.policy.rules)
        raise PolicyError(
            f"Candidate {verified.candidate_id} passes inside the "
            f"{config.policy.central_body_radius_km:.6g} km central body before completing "
            f"(minimum {radius_km:.1f} km). "
            + (
                f"The {rules.name} rules disqualify it "
                f"({rules.source.header_date} {rules.earth_crossing_clause})."
                if rules.earth_crossing_clause is not None
                else "Set policy.allow_body_intersecting_submission to certify it anyway."
            )
        )
    floor_km = config.policy.minimum_flown_altitude_km
    if floor_km is not None and radius_km < config.policy.central_body_radius_km + floor_km:
        raise AltitudeFloorError(
            f"Candidate {verified.candidate_id} flies down to "
            f"{radius_km - config.policy.central_body_radius_km:.3f} km altitude, under "
            f"policy.minimum_flown_altitude_km = {floor_km}"
        )
    if verified.score_is_stale:
        raise PolicyError(
            f"Candidate {verified.candidate_id} carries a score computed from inputs it no "
            "longer has; refusing to certify it"
        )
    # A refreshed EOP file moved a probed insertion state by 6-25 m; every run
    # behind one certificate must have read the same data.
    _assert_one_set_of_data_files(
        [diagnostic_run, *exact_runs, final_sibling_run],
        scenario.state_probe.run if scenario.state_probe is not None else None,
    )
    logged = _logged_errors(
        diagnostic_evidence,
        [*exact_runs, final_sibling_run,
         *([scenario.state_probe.run] if scenario.state_probe is not None else [])],
    )
    unexplained = sorted({record for records in logged.values()
                          for record in unexplained_log_errors(records)})
    if unexplained:
        raise GmatRunError(
            f"GmatConsole exited 0 and logged {len(unexplained)} error(s) none of which is "
            f"known harmless (gmat_runner.BENIGN_LOG_ERRORS), in run(s) "
            f"{', '.join(sorted(logged))}: {unexplained[0]}. A run that says it went wrong is "
            "not evidence; read its GmatLog.txt"
        )

    # Everything passed. Certification only confers final authority when the
    # underlying build was the organizer's; otherwise the evidence is recorded
    # but the candidate stays at the public-GMAT tier and the artifact keeps a
    # name that does not overstate it.
    # What this run justifies, not what the candidate arrived carrying.
    # promote() is monotonic, so a candidate that reached the organizer tier
    # somewhere else keeps it -- and would otherwise hand that authority to a
    # certification run made on a different build. Demonstrated: a
    # candidates.json claiming organizer_gmat_verified produced
    # team_solution.script from a public R2026a run with organizer_compatible
    # unset. io.py no longer restores such a claim; this is the same refusal
    # for a candidate that never came from a file.
    conferred = gmat_authority_stage(config, diagnostic_run)
    runs_evidence = {
        "exact_runs": [result.run_id for result in exact_runs],
        "parity_run": final_sibling_run.run_id,
    }
    organizer = stage_rank(conferred) >= stage_rank(CandidateStage.ORGANIZER_GMAT_VERIFIED)
    emitted = conferred
    if console is not None:
        # Certified means GmatConsole ran the submitted bytes themselves.
        verified.stage_evidence["certification_of_console_copy"] = {
            **runs_evidence,
            "submitted_sha256": console["submitted_sha256"],
            "copy_sha256": console["copy_sha256"],
        }
    elif organizer:
        verified.promote(CandidateStage.CERTIFIED, **runs_evidence)
        emitted = CandidateStage.CERTIFIED
    else:
        verified.stage_evidence["certification_without_organizer_build"] = runs_evidence

    submission = root / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    # The name says what this run established, not what the candidate arrived
    # holding. promote() is monotonic on purpose -- evidence obtained elsewhere
    # was still obtained -- so a candidate that reached CERTIFIED somewhere else
    # keeps the stage, and reading the name off it handed a public-GMAT run the
    # submission filename. Gating the promotion was not enough: a candidate
    # already at the top never needed promoting.
    script_name = STAGE_SCRIPT_NAME[emitted]
    shutil.copy2(final_script, submission / script_name)
    if file_sha256(submission / script_name) != final_hash:
        raise GmatRunError(
            f"The submitted copy of {script_name} does not match the bytes that were run"
        )
    shutil.copy2(final_diff, submission / f"{script_name}.diff.txt")
    shutil.copy2(original_script, submission / "official_original.script")

    # Under a console copy these are the copy's: its kept lines are the submitted
    # bytes' own, and the sibling's report above, run off the submitted bytes,
    # is byte for byte the same (gmat.console_copy, docs/COMPETITION_POLICY.md).
    final_run = exact_runs[-1]
    for report_name, source in final_run.reports.items():
        destination = submission / report_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    certification = {
        "rule_set": rule_set_record(config.policy.rules),
        # What replayed_score was priced with; check_report_rows holds the day's config to it.
        "score_block": score_stamp(config.score),
        "candidate": verified.to_dict(),
        "original_script_sha256": file_sha256(original_script),
        "final_script_sha256": final_hash,
        "gmat_data_files": diagnostic_run.data_file_sha256,
        "spad_files": spad_record,
        "spad_warnings": spad_notes,
        "state_probe": scenario.state_probe.to_dict() if scenario.state_probe else None,
        "stage": verified.stage.value,
        # What this run established, which is what the artifact is named for.
        # It can sit below `stage`: a candidate certified on the organizer's
        # build and re-certified here on a public one keeps the higher stage
        # and gets the lower name.
        "conferred_stage": emitted.value,
        "frontier": frontier,
        # The radius the altitude gates above compared: the lower of the final
        # replay's and the verification flight's.
        "flown_minimum_radius_km": radius_km if math.isfinite(radius_km) else None,
        "flown_minimum_altitude_km": (
            radius_km - config.policy.central_body_radius_km
            if math.isfinite(radius_km)
            else None
        ),
        "minimum_flown_altitude_km": floor_km,
        # GMAT's geodetic Earth.Altitude on the final replay's rows, as the
        # organizer judges a crossing. Never below the sphere's figure above.
        "flown_minimum_geodetic_altitude_km": final_event.minimum_geodetic_altitude_km,
        "submission_script_name": script_name,
        "official_report_files": official_reports,
        "report_name_warnings": list(report_name_warnings),
        "report_filename_as_written": config.policy.report_filename_as_written,
        # The solver clock against the one T_team is scored on (§3).
        "clock": {**clock_record(scenario, config.policy), "t_team_s": t_team_s},
        # What the uploaded report's printed rows show; its score is replayed_score.
        "official_report_rows": official_rows,
        # Rows the epoch alignment wrote before t = 0, the same in every report
        # of a run: scripts/check_report_rows.py --frozen-rows skips them.
        "frozen_rows": frozen_rows,
        "archive_warnings": list(archive_warnings),
        # Set only when policy.allow_shallow_stop_submission let a shallow hit through.
        "shallow_stop": shallow,
        "planned_hit_certified_as_miss": comparison.flipped,
        # Set when --accept-gmat-loss let a flight worse than the archive's through.
        "accepted_gmat_loss": (
            comparison.message if comparison.refused_without_opt_in else None
        ),
        # Recorded whatever it found: the prices, the loss, a late entry.
        "gmat_comparison": comparison.to_dict(),
        # Set when --accept-gmat-loss lifted it.
        "lifted_first_flight_contradiction": contradicted or None,
        "diagnostic": diagnostic_evidence,
        "final_script_parity": {
            "first_entry_time_s": final_event.first_entry_time_s,
            "minimum_distance_km": replayed[2],
            "interpolated_minimum_distance_km": final_event.minimum_distance_km,
            "samples_inside_radius": final_event.samples_inside_radius,
            "first_entry_segment_dt_s": final_event.first_entry_segment_dt_s,
            "entry_is_under_resolved": final_event.entry_is_under_resolved,
            "end_time_s": final_event.end_time_s,
            # The uploaded report's printed rows priced, as step 7 prices them.
            "replayed_score": replay_score,
            # The same flight on the diagnostic report's 17 digits.
            "diagnostic_replayed_score": diagnostic_score,
            "sibling_run": final_sibling_run.to_dict(),
        },
        "exact_runs": [result.to_dict() for result in exact_runs],
        # Every error GMAT logged in a run behind this certificate, by run id; each was benign.
        "gmat_log_errors": logged,
        # Absent when GmatConsole ran the submitted bytes themselves; else the
        # proven copy less the plots that every exact run above ran.
        **({} if console is None else {"console_copy": console}),
        **({"stage_reason": "console_copy"} if console is not None and organizer else {}),
        "submission_files": {
            str(path.relative_to(submission)): file_sha256(path)
            for path in sorted(submission.rglob("*"))
            if path.is_file()
        },
    }
    (submission / "certification.json").write_text(
        json.dumps(certification, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return certification
