"""The target's trajectory as GMAT flies it, run once and interpolated.

SC_Alien never maneuvers, so where it is does not depend on anything the
search chooses. One GMAT run with no burns gives its states over [0, T_max]
under the organizer's force model; the search can then aim at those instead
of a two-body estimate that drifts 18 km in 2 h under the ranking model.
See docs/GMAT_IN_THE_LOOP.md, step 1.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import OrbitWinConfig
from .diagnostic import add_diagnostic_report
from .errors import DynamicsError, GmatRunError, ParseError
from .event_detection import _hermite_coefficients
from .gmat_parser import GmatScriptDocument, parse_gmat_script, propagator_calls, script_lines
from .gmat_runner import file_size_limit_for, run_gmat
from .models import EpochAlignment, Scenario, StateVector, as_vector3
from .report_parser import EPOCH_TOLERANCE_S, NumericReport, parse_numeric_report, report_clock
from .script_renderer import write_rendered_candidate
from .search import zero_burn_candidate

#: Rows closer together than this are one epoch written twice, as GMAT does
#: where one Propagate segment ends and the next begins.
_SAME_EPOCH_S = 1.0e-9
_MANEUVER_LINE_RE = re.compile(r"^\s*Maneuver\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Ephemeris:
    times_s: np.ndarray
    positions_km: np.ndarray
    velocities_km_s: np.ndarray

    def state_at(self, time_s: float) -> StateVector:
        """Cubic Hermite on position and velocity between the bracketing rows."""
        times = self.times_s
        if not times[0] - _SAME_EPOCH_S <= time_s <= times[-1] + _SAME_EPOCH_S:
            raise DynamicsError(
                f"t = {time_s!r} s is outside the ephemeris, {times[0]!r} to {times[-1]!r} s"
            )
        index = int(np.clip(np.searchsorted(times, time_s, "right") - 1, 0, len(times) - 2))
        dt_s = float(times[index + 1] - times[index])
        u = (time_s - float(times[index])) / dt_s
        coefficients = _hermite_coefficients(
            self.positions_km[index],
            self.velocities_km_s[index],
            self.positions_km[index + 1],
            self.velocities_km_s[index + 1],
            dt_s,
        )
        position = coefficients @ np.array([1.0, u, u * u, u**3])
        velocity = coefficients @ np.array([0.0, 1.0, 2.0 * u, 3.0 * u * u]) / dt_s
        return StateVector(as_vector3(position), as_vector3(velocity))


def ephemeris_from_report(
    path: str | Path,
    pursuer_name: str,
    target_name: str,
    alignment: EpochAlignment | None = None,
) -> Ephemeris:
    """The target's rows from a diagnostic report, one per epoch, from t = 0."""
    report = parse_numeric_report(path)
    clock = report_clock(report, pursuer_name, target_name, alignment)
    positions, velocities = craft_columns(report, target_name)
    joint = slice(clock.frozen_rows, None)
    return ephemeris_from_rows(
        clock.times_s[joint], positions[joint], velocities[joint], f"{path} for {target_name}"
    )


def _target_only_mission(script: bytes, scenario: Scenario) -> bytes:
    """The coast sibling with each Propagate flying the target alone, on its own
    propagator, for the same time, and the maneuvers left out.

    The pursuer may start on an arc that reaches the ground unpowered, and GMAT
    then fails the whole run (RK89 accuracy, measured), taking the target with it.
    An epoch alignment is kept verbatim: it is what puts the two on one epoch.
    """
    target = scenario.target.name
    # Up to a NUL, as GMAT reads it: a Propagate rewritten here drops the NUL, and
    # the lines after it, which the parser left unchecked, would then be read.
    lines = list(script_lines(script.decode("utf-8").split("\x00", 1)[0]))
    segments = iter(segment for segment in scenario.propagate_segments if not segment.implicit)
    aligned = scenario.alignment is not None
    begun = False
    for index, line in enumerate(lines):
        if not begun:
            begun = bool(re.match(r"^\s*BeginMissionSequence\s*;", line, re.IGNORECASE))
            continue
        ending = line[len(line.rstrip("\r\n")):]
        if re.match(r"^\s*Propagate\b", line, re.IGNORECASE):
            if aligned:
                aligned = False  # the parser reads it only as the first command
                continue
            segment = next(segments)
            props = [prop for prop, craft in propagator_calls(line) if target in craft]
            if len(props) != 1:
                raise ParseError(f"Cannot tell which propagator flies {target} in: {line.strip()}")
            lines[index] = (
                f"Propagate {props[0]}({target}) {{{target}.ElapsedSecs = {segment.duration_s!r}}};"
                f"{ending}"
            )
        elif _MANEUVER_LINE_RE.match(line):
            lines[index] = f"% {line}"
    return "".join(lines).encode("utf-8")


def target_rows_from_report(
    path: str | Path,
    target_name: str,
    alignment: EpochAlignment | None = None,
    *,
    alignment_rows: bool = False,
) -> Ephemeris:
    """The target's rows on the solver clock, from a run that propagated it alone.

    After an alignment that flies the target, t = 0 is the row where its A1
    epoch reaches the pursuer's, which waits there; rows before it are dropped
    unless ``alignment_rows``, which keeps them at negative times. One that
    flies the pursuer leaves the target at its own epoch, ElapsedSecs 0.
    """
    report = parse_numeric_report(path)
    times = report.column(f"{target_name}.ElapsedSecs")
    positions, velocities = craft_columns(report, target_name)
    if alignment is not None and alignment.offset_s > 0.0 and alignment.early == target_name:
        target_a1 = report.column(f"{target_name}.A1ModJulian")
        waiting_a1 = report.column(f"{alignment.late}.A1ModJulian")
        met = np.flatnonzero(np.abs(target_a1 - waiting_a1) * 86400.0 <= EPOCH_TOLERANCE_S)
        if not met.size:
            raise GmatRunError(
                f"{path}: {target_name} never reaches {alignment.late}'s A1 epoch; the "
                "alignment did not end there"
            )
        times = times - times[met[0]]
        keep = slice(0 if alignment_rows else int(met[0]), None)
        times, positions, velocities = times[keep], positions[keep], velocities[keep]
    return ephemeris_from_rows(times, positions, velocities, f"{path} for {target_name}")


def craft_columns(report: NumericReport, name: str) -> tuple[np.ndarray, np.ndarray]:
    """One craft's EarthMJ2000Eq position and velocity rows."""
    positions = np.column_stack(
        [report.column(f"{name}.EarthMJ2000Eq.{axis}") for axis in ("X", "Y", "Z")]
    )
    velocities = np.column_stack(
        [report.column(f"{name}.EarthMJ2000Eq.{axis}") for axis in ("VX", "VY", "VZ")]
    )
    return positions, velocities


def ephemeris_from_rows(
    times: np.ndarray, positions: np.ndarray, velocities: np.ndarray, source: str
) -> Ephemeris:
    """Rows as an Ephemeris, keeping the later of two rows at one epoch.

    The later row is the one after a Maneuver, so the result is valid from a
    burn onward and not across it.
    """
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(positions))
            and np.all(np.isfinite(velocities))):
        raise GmatRunError(f"{source} holds a non-finite value")
    if len(times) < 2:
        raise GmatRunError(f"{source} is not reported at increasing epochs")
    keep = np.append(np.abs(np.diff(times)) > _SAME_EPOCH_S, True)
    times, positions, velocities = times[keep], positions[keep], velocities[keep]
    if len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise GmatRunError(f"{source} is not reported at increasing epochs")
    return Ephemeris(times, positions, velocities)


def target_ephemeris_from_gmat(
    scenario: Scenario,
    document: GmatScriptDocument,
    config: OrbitWinConfig,
    gmat_console: str | Path,
    startup_file: str | Path | None,
    workdir: str | Path,
    *,
    alignment_rows: bool = False,
) -> Ephemeris:
    """Fly the organizer's target alone to T_max, on its own propagator, and read it.

    ``alignment_rows`` keeps what the target flies in the alignment, before t = 0.
    """
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=False)
    # A coast that happens to intercept would otherwise end the mission there.
    to_horizon = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, mission_end_policy="horizon")
    )
    script = root / "coast.script"
    write_rendered_candidate(
        script, root / "coast.script.diff.txt", document, scenario,
        zero_burn_candidate(scenario, to_horizon), to_horizon,
    )
    coast_scenario, coast_document = parse_gmat_script(
        script, to_horizon, probe=scenario.state_probe
    )
    diagnostic = add_diagnostic_report(
        coast_document, coast_scenario, config.gmat.diagnostic_report_name
    )
    sibling = root / "coast_diagnostic.script"
    sibling.write_bytes(_target_only_mission(diagnostic.script_bytes, coast_scenario))
    result = run_gmat(
        gmat_console, sibling, root / "run", [diagnostic.report_filename],
        startup_file=startup_file, timeout_s=config.gmat.timeout_s,
        file_size_limit_bytes=file_size_limit_for(coast_scenario),
        template_dir=document.path.resolve().parent,
    )
    if not result.success:
        raise GmatRunError(result.error or "GMAT coast run failed")
    ephemeris = target_rows_from_report(
        result.reports[diagnostic.report_filename], scenario.target.name,
        coast_scenario.alignment, alignment_rows=alignment_rows,
    )
    if ephemeris.times_s[0] > _SAME_EPOCH_S or (
        ephemeris.times_s[-1] < scenario.max_time_s - 1.0e-6
    ):
        raise GmatRunError(
            f"The coast run reports {scenario.target.name} from {ephemeris.times_s[0]!r} to "
            f"{ephemeris.times_s[-1]!r} s, not over [0, {scenario.max_time_s!r}] s"
        )
    return ephemeris
