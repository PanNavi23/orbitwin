"""Turn a near hit into a hit on GMAT's own residuals.

The search plans in two-body dynamics, and under the ranking force model GMAT
flies those plans tens of km wide; a plan made in pursuer_model.py's model,
0.9 to 2.4 km on R1, R3 and R4. This flies a candidate in GMAT, takes the
miss at the planned arrival and changes the last impulse by a Newton step whose
sensitivity comes from the two-body model; on 2026-10-31 that went 18.565 km,
9.5 m, 6 mm in two corrections. See docs/GMAT_IN_THE_LOOP.md, step 3.

The miss is taken at the planned arrival t_f, not at GMAT's closest approach.
d r(t_f) / d v is what the two-body model differentiates, and it has full rank,
while the closest-approach miss lies in the plane normal to the relative
velocity, so its sensitivity has rank two and leaves the arrival epoch free to
drift -- by the miss over the relative speed, which is seconds at 10 km/s and
minutes on a slow approach. Zeroing the miss at t_f keeps the epoch the score
was planned on. Whether the corrected schedule hits is judged on the closest
approach of its first passage through the sphere, as verification judges it.

Nothing here promotes a candidate. It comes back analytical-stage, and
verify_candidate_with_gmat stays the authority on it.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .burn_rows import (
    BURN_EPOCH_TOLERANCE_S,
    FlownManeuver,
    maneuvers_flown,
    split_at_alignment,
    with_post_burn_rows,
)
from .config import OrbitWinConfig
from .constraints import apply_constraints
from .diagnostic import add_diagnostic_report
from .ephemeris import craft_columns, ephemeris_from_rows
from .errors import DynamicsError, GmatRunError, OrbitWinError, PolicyError
from .event_detection import EventResult, evaluate_hermite_samples
from .frames import inertial_to_script_components
from .gmat_parser import GmatScriptDocument, parse_gmat_script
from .gmat_runner import file_size_limit_for, run_gmat
from .io import prepare_empty_output
from .models import (
    Burn,
    Candidate,
    GmatRunResult,
    Scenario,
    StateVector,
    as_vector3,
    burn_frame,
)
from .orbit import propagate_universal
from .pursuer_model import PursuerModel, pursuer_model_for
from .pursuer_model import fly as fly_in_model
from .report_parser import (
    parse_numeric_report,
    relative_event,
    report_clock,
    stop_depth_used_km,
)
from .score import HEAD_START_ALIGNMENT_KEY, apply_score
from .script_renderer import write_rendered_candidate
from .trajectory import fly_pursuer, head_start_alignment_distance_km

#: How long the private run continues past the planned arrival, so the rows
#: bracket t_f and a closest approach that comes a little late.
ARRIVAL_MARGIN_S = 120.0

#: A Newton step with the two-body sensitivity that leaves more than this
#: fraction of the miss means that sensitivity is wrong for this arc. The probe
#: left 5e-4.
_CONTRACTION = 0.3

#: Central-difference step on the two-body arrival, km/s.
_TWO_BODY_STEP_KM_S = 1.0e-5

#: Perturbation for a GMAT-measured sensitivity, km/s: 2.4 km at t_f over a
#: 2400 s arc against millimetre integrator noise.
_GMAT_STEP_KM_S = 1.0e-3

#: Slack on a safe cap, as in multi_burn.
_CAP_TOLERANCE_KM_S = 1.0e-12

#: The two-impulse solve aims this far inside every cap.
_SOLVER_MARGIN_KM_S = 1.0e-9

#: Inertial and script components of one burn are one vector in two frames.
_FRAME_TOLERANCE_KM_S = 1.0e-9

Schedule = list[tuple[float, np.ndarray]]


def _corrected(schedule: Schedule) -> list[int]:
    """The one or two impulses the correction moves: the last two that carry one.

    An exact-zero slot a pinned schedule keeps (docs/ASYNC_START_AND_INSERTION.md,
    "Step 6 as built") lies nearer arrival than the burn before it and would
    take far more to correct the same miss. With one burn, it and the last slot.
    """
    active = [index for index, (_, impulse) in enumerate(schedule) if np.any(impulse)]
    last = len(schedule) - 1
    if len(active) >= 2:
        return active[-2:]
    if len(active) == 1 and active[0] != last:
        return [active[0], last]
    return list(range(max(0, len(schedule) - 2), len(schedule)))

#: What ends a correction with its rounds kept rather than escaping: our own
#: refusals, a singular solve, and the file system or GmatConsole failing to
#: start. KeyboardInterrupt and SystemExit are not among them.
_RUN_ERRORS = (OrbitWinError, np.linalg.LinAlgError, OSError)


@dataclass(frozen=True, slots=True)
class CorrectionStep:
    """One GMAT round: a flight, or the concurrent runs that measure J."""

    round: int
    #: "flight" or "sensitivity".
    kind: str
    run_ids: tuple[str, ...]
    wall_s: float
    #: Longest GmatConsole process in the round; wall_s adds rendering and reading.
    gmat_s: float
    #: |r_pursuer - r_target| at the planned arrival, from GMAT.
    miss_at_arrival_km: float | None = None
    closest_approach_km: float | None = None
    closest_approach_time_s: float | None = None
    first_entry_time_s: float | None = None
    #: Lowest geocentric radius over the whole flown arc.
    minimum_radius_km: float | None = None
    total_dv_km_s: float | None = None
    largest_impulse_km_s: float | None = None
    #: How the next schedule was made from this flight: "last" or "last-two".
    mode: str | None = None
    #: "two-body" or "gmat".
    sensitivity: str | None = None
    #: Norm of the change to the impulses, km/s.
    change_km_s: float | None = None
    #: The exception that ended the correction in this round, "Type: message":
    #: GMAT failing or refusing, a report too short to read, or a fault in
    #: this code; correction.json carries its traceback.
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    #: True when GMAT's closest approach reached aim_km.
    converged: bool
    #: The schedule that met the aim, scored; None unless converged.
    candidate: Candidate | None
    #: Unless converged, a legal flight for a caller deciding what to try
    #: next: the one with the closest first passage, ties to the smaller t_f
    #: miss, except after the earlier-pass stop, which returns the flight that
    #: hit t_f. Unscored, not successful, never a hit.
    closest_flight: Candidate | None
    closest_approach_km: float | None
    closest_approach_time_s: float | None
    first_entry_time_s: float | None
    #: Why it did not converge.
    reason: str | None
    rounds: int
    gmat_runs: int
    wall_s: float
    log: tuple[CorrectionStep, ...] = field(default_factory=tuple)
    #: Of the exception that ended the correction, if one did.
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "candidate": self.candidate.to_dict() if self.candidate is not None else None,
            "closest_flight": (
                self.closest_flight.to_dict() if self.closest_flight is not None else None
            ),
            "closest_approach_km": self.closest_approach_km,
            "closest_approach_time_s": self.closest_approach_time_s,
            "first_entry_time_s": self.first_entry_time_s,
            "reason": self.reason,
            "rounds": self.rounds,
            "gmat_runs": self.gmat_runs,
            "wall_s": self.wall_s,
            "log": [dataclasses.asdict(step) for step in self.log],
            "traceback": self.traceback,
        }


@dataclass(frozen=True, slots=True)
class _Flight:
    burns: tuple[Burn, ...]
    #: r_pursuer - r_target at t_f, km.
    miss_km: np.ndarray
    event: EventResult
    minimum_radius_km: float
    minimum_radius_time_s: float
    #: The lowest radius where verification looks for the altitude floor.
    floor_radius_km: float
    floor_radius_time_s: float
    run: GmatRunResult


@dataclass(frozen=True, slots=True)
class _Limits:
    burn_cap_km_s: float
    total_cap_km_s: float | None


def correct_with_gmat(
    original_document: GmatScriptDocument,
    scenario: Scenario,
    candidate: Candidate,
    config: OrbitWinConfig,
    gmat_console: str | Path,
    startup_file: str | Path | None,
    workdir: str | Path,
    *,
    aim_km: float = 0.5,
    max_rounds: int = 4,
    arrival_margin_s: float = ARRIVAL_MARGIN_S,
) -> CorrectionResult:
    """Correct `candidate` until GMAT's closest approach is within `aim_km`.

    A round is one GMAT run, or the three (six, with two or more impulses) runs that
    measure the sensitivity when the two-body one stops converging, at most
    config.gmat.workers at a time; `max_rounds` bounds the rounds, `gmat_runs`
    in the result counts runs.

    The last impulse takes the correction. When that would put it over the safe
    cap or the total over its safe limit, the last two share it, least-norm
    with the caps as constraints; when that fails too, the result says why.
    Burn epochs never move. Every flight must stay above the central body over
    the whole arc flown, and one that does not ends the correction. So does any
    OrbitWinError, LinAlgError or OSError in a round, ours or GMAT's: the
    reason names its type, correction.json keeps its traceback, and the rounds
    before are kept. Anything else escapes.

    Only a converged result carries `candidate`. Otherwise `closest_flight` is
    a legal flight to try next, unscored and not successful: see
    CorrectionResult.closest_flight for which.
    """
    # A closest approach at the aim reaches the stop depth; one past it never does.
    deepest_aim_km = config.policy.intercept_radius_km - stop_depth_used_km(
        config.policy.stop_depth_km, config.policy.intercept_radius_km
    )
    if not 0.0 < aim_km <= deepest_aim_km:
        raise ValueError(
            f"aim_km {aim_km!r} must be in (0, {deepest_aim_km!r}] km, the radius less "
            "policy.stop_depth_km"
        )
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    if not (math.isfinite(arrival_margin_s) and arrival_margin_s >= 0.0):
        raise ValueError(f"arrival_margin_s {arrival_margin_s!r} must be finite and >= 0")
    began = time.perf_counter()
    root = Path(workdir)
    prepare_empty_output(root, "GMAT correction")
    policy = config.policy
    limits = _Limits(policy.dv_safe_limit_km_s, policy.total_dv_safe_limit_km_s)
    burns = tuple(sorted(candidate.burns, key=lambda burn: burn.time_s))
    arrival_s = _planned_arrival(candidate)

    def finish(
        converged: bool,
        best: tuple[_Flight, int] | None,
        reason: str | None,
        log: Sequence[CorrectionStep],
        runs: int,
        error: BaseException | None = None,
    ) -> CorrectionResult:
        corrected = None
        if best is not None and arrival_s is not None:
            corrected = _corrected_candidate(
                candidate, best[0], scenario, config, arrival_s, aim_km, converged, best[1]
            )
        flight = best[0] if best is not None else None
        result = CorrectionResult(
            converged=converged,
            candidate=corrected if converged else None,
            closest_flight=None if converged else corrected,
            closest_approach_km=flight.event.minimum_distance_km if flight else None,
            closest_approach_time_s=flight.event.minimum_distance_time_s if flight else None,
            first_entry_time_s=flight.event.first_entry_time_s if flight else None,
            reason=reason,
            rounds=len(log),
            gmat_runs=runs,
            wall_s=time.perf_counter() - began,
            log=tuple(log),
            traceback=_traceback(error) if error else None,
        )
        (root / "correction.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return result

    refusal = _refusal(burns, arrival_s, scenario, config)
    if not candidate.successful:
        # Its arrival is T_max, not an aim; a miss needs an aim offset first.
        refusal = "the candidate is a miss, and only a hit is corrected"
    if refusal is not None or arrival_s is None:
        return finish(False, None, refusal, (), 0)
    planned_in = candidate.metadata.get("pursuer_model")
    model = None if planned_in is None else pursuer_model_for(scenario, str(planned_in))
    try:
        _check_frames(burns, scenario, model)
    except PolicyError as error:
        return finish(False, None, str(error), (), 0)

    private = dataclasses.replace(
        config, policy=dataclasses.replace(policy, mission_end_policy="interception")
    )
    end_s = min(arrival_s + arrival_margin_s, scenario.max_time_s)
    slots = [burn.slot_index for burn in burns]
    schedule: Schedule = [(burn.time_s, np.asarray(burn.dv_inertial_km_s, dtype=float))
                          for burn in burns]
    current = burns

    started_runs = [0]
    counting = threading.Lock()

    def count_run() -> None:
        with counting:
            started_runs[0] += 1

    def fly(flown: tuple[Burn, ...], directory: Path) -> _Flight:
        return _fly(
            original_document, scenario, private, flown, end_s, arrival_s,
            gmat_console, startup_file, directory, count_run,
        )

    def failed(kind: str, started: float, error: Exception) -> CorrectionResult:
        # GMAT aborting (drag below ~100 km stops RK89) or timing out, a report
        # too short to read, or a fault of ours: the round ends the correction
        # and is never the best. The type says which.
        log.append(
            CorrectionStep(
                round=len(log) + 1, kind=kind, run_ids=(),
                wall_s=time.perf_counter() - started, gmat_s=0.0, error=_described(error),
            )
        )
        return finish(
            False, best, f"round {len(log)} ({kind}) failed: {_described(error)}", log,
            started_runs[0], error,
        )

    log: list[CorrectionStep] = []
    best: tuple[_Flight, int] | None = None
    sensitivity = "two-body"
    measured: dict[int, np.ndarray] | None = None
    previous_miss_km: float | None = None
    while True:
        round_number = len(log) + 1
        started = time.perf_counter()
        try:
            flight = fly(current, root / f"round-{round_number:02d}")
        except _RUN_ERRORS as error:
            return failed("flight", started, error)
        miss_km = float(np.linalg.norm(flight.miss_km))
        log.append(
            CorrectionStep(
                round=round_number,
                kind="flight",
                run_ids=(flight.run.run_id,),
                wall_s=time.perf_counter() - started,
                gmat_s=flight.run.duration_s,
                miss_at_arrival_km=miss_km,
                closest_approach_km=flight.event.minimum_distance_km,
                closest_approach_time_s=flight.event.minimum_distance_time_s,
                first_entry_time_s=flight.event.first_entry_time_s,
                minimum_radius_km=flight.minimum_radius_km,
                total_dv_km_s=float(sum(burn.magnitude_km_s for burn in current)),
                largest_impulse_km_s=max(burn.magnitude_km_s for burn in current),
            )
        )
        if flight.minimum_radius_km < policy.central_body_radius_km:
            return finish(
                False, best,
                f"GMAT flies the pursuer to {flight.minimum_radius_km:.3f} km at "
                f"t = {flight.minimum_radius_time_s:.3f} s in round {round_number}, inside the "
                f"{policy.central_body_radius_km:.4f} km central body",
                log, started_runs[0],
            )
        # Verification refuses a flight under the floor; no use correcting one.
        floor_km = policy.minimum_flown_altitude_km
        if floor_km is not None and (
            flight.floor_radius_km < policy.central_body_radius_km + floor_km
        ):
            return finish(
                False, best,
                f"GMAT flies the pursuer to "
                f"{flight.floor_radius_km - policy.central_body_radius_km:.3f} km altitude at "
                f"t = {flight.floor_radius_time_s:.3f} s in round {round_number}, under the "
                f"{floor_km:g} km policy.minimum_flown_altitude_km",
                log, started_runs[0],
            )
        if best is None or (flight.event.minimum_distance_km, miss_km) < (
            best[0].event.minimum_distance_km, float(np.linalg.norm(best[0].miss_km))
        ):
            best = (flight, round_number)
        if flight.event.minimum_distance_km <= aim_km:
            return finish(True, best, None, log, started_runs[0])
        if miss_km <= aim_km:
            # Hit at t_f, but an earlier pass enters first and verification
            # measures that one; correcting t_f further cannot move it.
            return finish(
                False, (flight, round_number),
                f"round {round_number}: the miss at t_f is {miss_km:.6g} km, but GMAT first "
                "enters the sphere at "
                f"{flight.event.first_entry_time_s!r} s on an earlier pass that comes no closer "
                f"than {flight.event.minimum_distance_km:.6g} km at "
                f"{flight.event.minimum_distance_time_s!r} s: the flight already intercepts "
                "there, short of the aim",
                log, started_runs[0],
            )
        if len(log) >= max_rounds:
            return finish(
                False, best,
                f"{max_rounds} round(s) used; the closest flight, round {best[1]}, comes "
                f"within {best[0].event.minimum_distance_km:.6g} km against the "
                f"{aim_km:.6g} km aim",
                log, started_runs[0],
            )
        if previous_miss_km is not None and miss_km > _CONTRACTION * previous_miss_km:
            if sensitivity == "gmat":
                return finish(
                    False, best,
                    f"round {round_number}: the miss went from {previous_miss_km:.6g} to "
                    f"{miss_km:.6g} km with the GMAT-measured sensitivity",
                    log, started_runs[0],
                )
            if len(log) + 2 > max_rounds:
                return finish(
                    False, best,
                    f"round {round_number}: the two-body sensitivity took the miss from "
                    f"{previous_miss_km:.6g} to {miss_km:.6g} km, and measuring GMAT's needs "
                    "two more rounds than remain",
                    log, started_runs[0],
                )
            started = time.perf_counter()
            try:
                measured, probes = _measure_sensitivity(
                    fly, scenario, schedule, slots, burns, flight,
                    root / f"round-{len(log) + 1:02d}-sensitivity", config.gmat.workers, model,
                )
            except _RUN_ERRORS as error:
                return failed("sensitivity", started, error)
            sensitivity = "gmat"
            log.append(
                CorrectionStep(
                    round=len(log) + 1,
                    kind="sensitivity",
                    run_ids=tuple(probe.run.run_id for probe in probes),
                    wall_s=time.perf_counter() - started,
                    gmat_s=max(probe.run.duration_s for probe in probes),
                )
            )
        previous_miss_km = miss_km
        try:
            jacobians = (
                measured
                if measured is not None
                else _two_body_jacobians(scenario, schedule, arrival_s)
            )
            stepped, mode, reason = _step(schedule, flight.miss_km, jacobians, limits)
            following = _express(scenario, stepped, slots, burns, model) if stepped else None
        except _RUN_ERRORS as error:
            return finish(
                False, best, f"the step after round {round_number} failed: {_described(error)}",
                log, started_runs[0], error,
            )
        flight_index = next(
            index for index in range(len(log) - 1, -1, -1) if log[index].kind == "flight"
        )
        log[flight_index] = dataclasses.replace(
            log[flight_index],
            mode=mode,
            sensitivity=sensitivity,
            change_km_s=(
                float(np.sqrt(sum(
                    float(np.sum((new - old) ** 2))
                    for (_, new), (_, old) in zip(stepped, schedule, strict=True)
                )))
                if stepped is not None
                else None
            ),
        )
        if stepped is None or following is None:
            return finish(False, best, f"after round {round_number}: {reason}", log, started_runs[0])
        schedule, current = stepped, following


def _described(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


#: Where this package and the interpreter live, written relative in tracebacks.
_PACKAGE_ROOT = str(Path(__file__).resolve().parents[1]) + os.sep
_PYTHON_ROOTS = sorted({sys.prefix, sys.base_prefix, sys.exec_prefix}, key=len, reverse=True)


def _traceback(error: BaseException) -> str:
    """The traceback with causes, paths relative to the package or the interpreter."""
    text = "".join(traceback.format_exception(error))
    text = text.replace(_PACKAGE_ROOT, "")
    for root in _PYTHON_ROOTS:
        text = text.replace(root + os.sep, "<python>" + os.sep)
    return text


def _planned_arrival(candidate: Candidate) -> float | None:
    """Where the last arc was aimed: the Lambert arrival, else the interception.

    A candidate this module returned keeps its t_f in gmat_correction, since its
    interception is GMAT's first entry, seconds before t_f.
    """
    value = candidate.metadata.get("arrival_time_s")
    if value is None:
        value = candidate.metadata.get("gmat_correction", {}).get("planned_arrival_s")
    if value is None:
        value = candidate.intercept_time_s
    return None if value is None else float(value)


def _refusal(
    burns: tuple[Burn, ...],
    arrival_s: float | None,
    scenario: Scenario,
    config: OrbitWinConfig,
) -> str | None:
    """Why this candidate cannot be corrected, or None."""
    policy = config.policy
    if not burns:
        return "the candidate has no impulse to correct"
    if arrival_s is None or not math.isfinite(arrival_s):
        return "the candidate carries no planned arrival epoch"
    if arrival_s <= burns[-1].time_s:
        return (
            f"the planned arrival {arrival_s:.6g} s is not after the last impulse at "
            f"{burns[-1].time_s:.6g} s"
        )
    if arrival_s > scenario.max_time_s + 1.0e-9:
        return f"the planned arrival {arrival_s:.6g} s is after T_max {scenario.max_time_s:.6g} s"
    for index, burn in enumerate(burns):
        if burn.magnitude_km_s > policy.dv_safe_limit_km_s + _CAP_TOLERANCE_KM_S:
            return (
                f"impulse {index} is {burn.magnitude_km_s:.12g} km/s before any correction, over "
                f"the {policy.dv_safe_limit_km_s:.12g} km/s safe cap"
            )
        if index and burn.time_s - burns[index - 1].time_s < policy.safe_burn_separation_s - 1.0e-9:
            return (
                f"impulses {index - 1} and {index} are {burn.time_s - burns[index - 1].time_s:.12g}"
                f" s apart, under the {policy.safe_burn_separation_s:.12g} s safe separation"
            )
    total = sum(burn.magnitude_km_s for burn in burns)
    if (
        policy.total_dv_safe_limit_km_s is not None
        and total > policy.total_dv_safe_limit_km_s + _CAP_TOLERANCE_KM_S
    ):
        return (
            f"the total {total:.12g} km/s is over the {policy.total_dv_safe_limit_km_s:.12g} km/s "
            "safe limit before any correction"
        )
    return None


def _flown_states(
    scenario: Scenario, schedule: Schedule, model: PursuerModel | None = None
) -> list[StateVector]:
    """Pre-burn state at each impulse: two-body, or in the model the plan was made in."""
    if model is not None:
        times = np.array([time_s for time_s, _ in schedule], dtype=float)
        if np.any(times < 0.0):
            raise DynamicsError("no pursuer model flies a head start; it was planned two-body")
        initial = scenario.pursuer.initial_state
        flown = fly_in_model(
            model, np.asarray(initial.r), np.asarray(initial.v), times,
            np.array([impulse for _, impulse in schedule]).reshape(-1, 3),
            float(times[-1]) if len(times) else 0.0,
        )
        return [
            StateVector(as_vector3(r), as_vector3(v))
            for r, v in zip(flown.r_at_burns, flown.v_before, strict=True)
        ]
    end_s = schedule[-1][0] if schedule else 0.0
    return fly_pursuer(scenario, schedule, end_s)[0]


def _express(
    scenario: Scenario,
    schedule: Schedule,
    slots: Sequence[int],
    original: tuple[Burn, ...],
    model: PursuerModel | None = None,
) -> tuple[Burn, ...]:
    """The schedule as Burns, script components taken on the flown state the
    plan was made in: two-body, or `model`'s.

    That is the convention every search here writes burns in. Burns before the
    first changed impulse are the original objects, bit for bit.
    """
    unchanged = 0
    for (_, impulse), burn in zip(schedule, original, strict=True):
        if not np.array_equal(impulse, np.asarray(burn.dv_inertial_km_s, dtype=float)):
            break
        unchanged += 1
    if unchanged == len(original):
        return original
    states = _flown_states(scenario, schedule, model)
    burns = list(original[:unchanged])
    for index in range(unchanged, len(schedule)):
        time_s, impulse = schedule[index]
        components = inertial_to_script_components(
            impulse, states[index], *burn_frame(scenario, slots[index])
        )
        if abs(float(np.linalg.norm(components)) - float(np.linalg.norm(impulse))) > (
            _FRAME_TOLERANCE_KM_S
        ):
            raise DynamicsError(f"impulse {index} changed length between frames")
        burns.append(
            Burn(
                time_s=time_s,
                dv_inertial_km_s=as_vector3(impulse),
                dv_script_km_s=as_vector3(components),
                slot_index=slots[index],
                frame=burn_frame(scenario, slots[index]),
            )
        )
    return tuple(burns)


def _check_frames(
    burns: tuple[Burn, ...], scenario: Scenario, model: PursuerModel | None = None
) -> None:
    """Refuse a candidate whose burns are not in the convention _express writes,
    on the states of the model it names, or two-body.

    Otherwise the corrected burn would be written in a different convention
    from its neighbours, and GMAT would fly a vector nobody computed.
    """
    schedule = [(burn.time_s, np.asarray(burn.dv_inertial_km_s, dtype=float)) for burn in burns]
    states = _flown_states(scenario, schedule, model)
    for index, (burn, state) in enumerate(zip(burns, states, strict=True)):
        expected = inertial_to_script_components(
            schedule[index][1], state, *burn_frame(scenario, burn.slot_index)
        )
        drift = float(np.max(np.abs(expected - np.asarray(burn.dv_script_km_s, dtype=float))))
        if drift > _FRAME_TOLERANCE_KM_S:
            raise PolicyError(
                f"Burn {index}'s script components differ by {drift:.3g} km/s from its inertial "
                f"vector expressed on the {model.name if model else 'two-body'} flown state; "
                "refusing to correct a candidate whose two frames disagree"
            )


def _two_body_arrival(scenario: Scenario, schedule: Schedule, arrival_s: float) -> np.ndarray:
    return np.asarray(fly_pursuer(scenario, schedule, arrival_s)[1].r, dtype=float)


def _two_body_jacobians(
    scenario: Scenario, schedule: Schedule, arrival_s: float
) -> dict[int, np.ndarray]:
    """d r(t_f) / d v for the impulses _corrected picks, by central differences."""
    jacobians: dict[int, np.ndarray] = {}
    for index in _corrected(schedule):
        columns = []
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = _TWO_BODY_STEP_KM_S
            plus = _replace_impulse(schedule, index, schedule[index][1] + offset)
            minus = _replace_impulse(schedule, index, schedule[index][1] - offset)
            columns.append(
                (_two_body_arrival(scenario, plus, arrival_s)
                 - _two_body_arrival(scenario, minus, arrival_s))
                / (2.0 * _TWO_BODY_STEP_KM_S)
            )
        jacobians[index] = np.column_stack(columns)
    return jacobians


def _replace_impulse(schedule: Schedule, index: int, impulse: np.ndarray) -> Schedule:
    replaced = list(schedule)
    replaced[index] = (schedule[index][0], np.asarray(impulse, dtype=float))
    return replaced


def _measure_sensitivity(
    fly: Callable[[tuple[Burn, ...], Path], _Flight],
    scenario: Scenario,
    schedule: Schedule,
    slots: Sequence[int],
    original: tuple[Burn, ...],
    base: _Flight,
    directory: Path,
    workers: int,
    model: PursuerModel | None = None,
) -> tuple[dict[int, np.ndarray], list[_Flight]]:
    """J from GMAT: perturb each component of the impulses _corrected picks.

    A perturbation that would lengthen its impulse is scaled back to the
    original length, so no probe crosses a cap the schedule respects. Probes
    are written in the frames of the model the plan was made in.
    """
    jobs: list[tuple[int, np.ndarray, tuple[Burn, ...], Path]] = []
    for index in _corrected(schedule):
        impulse = schedule[index][1]
        for axis in range(3):
            trial = impulse.copy()
            trial[axis] += -_GMAT_STEP_KM_S if impulse[axis] > 0.0 else _GMAT_STEP_KM_S
            length = float(np.linalg.norm(impulse))
            if float(np.linalg.norm(trial)) > length > 0.0:
                trial *= length / float(np.linalg.norm(trial))
            perturbed = _replace_impulse(schedule, index, trial)
            jobs.append(
                (index, trial - impulse, _express(scenario, perturbed, slots, original, model),
                 directory / f"burn{index}-axis{axis}")
            )
    directory.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        futures = [pool.submit(fly, burns, path) for _, _, burns, path in jobs]
        flights: list[_Flight] = [future.result() for future in futures]
    jacobians: dict[int, np.ndarray] = {}
    for index in sorted({job[0] for job in jobs}):
        picked = [k for k, job in enumerate(jobs) if job[0] == index]
        offsets = np.column_stack([jobs[k][1] for k in picked])
        changes = np.column_stack([flights[k].miss_km - base.miss_km for k in picked])
        jacobians[index] = changes @ np.linalg.inv(offsets)
    return jacobians, flights


def _within(schedule: Schedule, limits: _Limits) -> str | None:
    """Which safe limit the schedule breaks, or None."""
    for index, (_, impulse) in enumerate(schedule):
        magnitude = float(np.linalg.norm(impulse))
        if magnitude > limits.burn_cap_km_s + _CAP_TOLERANCE_KM_S:
            return (
                f"impulse {index} would be {magnitude:.12g} km/s, over the "
                f"{limits.burn_cap_km_s:.12g} km/s safe cap"
            )
    if limits.total_cap_km_s is not None:
        total = sum(float(np.linalg.norm(impulse)) for _, impulse in schedule)
        if total > limits.total_cap_km_s + _CAP_TOLERANCE_KM_S:
            return (
                f"the total would be {total:.12g} km/s, over the "
                f"{limits.total_cap_km_s:.12g} km/s safe limit"
            )
    return None


def _step(
    schedule: Schedule,
    miss_km: np.ndarray,
    jacobians: dict[int, np.ndarray],
    limits: _Limits,
) -> tuple[Schedule | None, str, str | None]:
    """The next schedule, the mode that made it, and why there is none."""
    chosen = _corrected(schedule)
    last = chosen[-1]
    try:
        change = -np.linalg.solve(jacobians[last], miss_km)
    except np.linalg.LinAlgError:
        broken = "the last impulse's sensitivity is singular"
    else:
        trial = _replace_impulse(schedule, last, schedule[last][1] + change)
        broken_or_none = _within(trial, limits)
        if broken_or_none is None:
            return trial, "last", None
        broken = broken_or_none
    if len(chosen) < 2:
        return None, "last", f"{broken}, and there is no earlier impulse to share the correction"
    shared = _least_norm_two(schedule, miss_km, jacobians[chosen[0]], jacobians[last], limits)
    if shared is None:
        return None, "last-two", (
            f"correcting the last impulse alone: {broken}; and no change to the last two "
            f"within the safe limits removes the {float(np.linalg.norm(miss_km)):.6g} km miss"
        )
    return shared, "last-two", None


def _least_norm_two(
    schedule: Schedule,
    miss_km: np.ndarray,
    jacobian_a: np.ndarray,
    jacobian_b: np.ndarray,
    limits: _Limits,
) -> Schedule | None:
    """Least-norm change to the two impulses _corrected picks that zeroes the linearised miss.

    Six unknowns in m/s, three equations in metres, each impulse inside its cap
    and the total inside its limit, all _SOLVER_MARGIN_KM_S inside. Starts
    from the unconstrained least-norm change and keeps it when it is legal.
    """
    a, b = _corrected(schedule)
    base_a = 1000.0 * schedule[a][1]
    base_b = 1000.0 * schedule[b][1]
    others_m_s = 1000.0 * sum(
        float(np.linalg.norm(impulse)) for k, (_, impulse) in enumerate(schedule) if k not in (a, b)
    )
    matrix = np.hstack([jacobian_a, jacobian_b])
    target_m = -1000.0 * np.asarray(miss_km, dtype=float)
    cap_m_s = 1000.0 * (limits.burn_cap_km_s - _SOLVER_MARGIN_KM_S)

    def schedule_for(x: np.ndarray) -> Schedule:
        stepped = _replace_impulse(schedule, a, (base_a + x[:3]) / 1000.0)
        return _replace_impulse(stepped, b, (base_b + x[3:]) / 1000.0)

    start = np.linalg.pinv(matrix) @ target_m
    if _within(schedule_for(start), limits) is None:
        return schedule_for(start)

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda x: matrix @ x - target_m, "jac": lambda x: matrix},
        {
            "type": "ineq",
            "fun": lambda x: cap_m_s**2 - float(np.sum((base_a + x[:3]) ** 2)),
            "jac": lambda x: np.concatenate([-2.0 * (base_a + x[:3]), np.zeros(3)]),
        },
        {
            "type": "ineq",
            "fun": lambda x: cap_m_s**2 - float(np.sum((base_b + x[3:]) ** 2)),
            "jac": lambda x: np.concatenate([np.zeros(3), -2.0 * (base_b + x[3:])]),
        },
    ]
    if limits.total_cap_km_s is not None:
        total_m_s = 1000.0 * (limits.total_cap_km_s - _SOLVER_MARGIN_KM_S) - others_m_s

        def total_slack(x: np.ndarray) -> float:
            return total_m_s - float(
                np.linalg.norm(base_a + x[:3]) + np.linalg.norm(base_b + x[3:])
            )

        def total_gradient(x: np.ndarray) -> np.ndarray:
            return -np.concatenate([_unit(base_a + x[:3]), _unit(base_b + x[3:])])

        constraints.append({"type": "ineq", "fun": total_slack, "jac": total_gradient})
    solved = minimize(
        lambda x: 0.5 * float(x @ x),
        start,
        jac=lambda x: x,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1.0e-12},
    )
    # Feasibility decides, not solved.success. On two ranking-model cases with the
    # last impulse at the cap along VNB (-0.3, 1, 0) and (1, 1, 0), SLSQP ended
    # with status 8 ("positive directional derivative") on a point 1e-11 m off
    # the equations and inside every cap; the case in the gmat test ends status 0.
    x = np.asarray(solved.x, dtype=float)
    residual_m = float(np.linalg.norm(matrix @ x - target_m))
    if not np.all(np.isfinite(x)) or residual_m > 1.0e-3 + 1.0e-9 * float(
        np.linalg.norm(target_m)
    ):
        return None
    stepped = schedule_for(x)
    return stepped if _within(stepped, limits) is None else None


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / length if length > 0.0 else np.zeros_like(vector)


def _fly(
    document: GmatScriptDocument,
    scenario: Scenario,
    config: OrbitWinConfig,
    burns: tuple[Burn, ...],
    end_s: float,
    arrival_s: float,
    gmat_console: str | Path,
    startup_file: str | Path | None,
    directory: Path,
    on_run: Callable[[], None] = lambda: None,
) -> _Flight:
    """Render `burns` into the organizer's script, fly it to `end_s`, read the result.

    The render is the one verification would make, ending at `end_s` instead of
    the first entry; the diagnostic sibling adds both craft's states.
    """
    directory.mkdir(parents=True, exist_ok=False)
    probe = Candidate(
        burns=burns,
        intercept_time_s=end_s,
        minimum_distance_km=None,
        distance_at_completion_km=None,
        successful=True,
        analytical_endpoint_error_km=None,
        branch="gmat-shoot",
    )
    script = directory / "correction_candidate.script"
    rendered = write_rendered_candidate(
        script, directory / "correction_candidate.script.diff.txt", document, scenario, probe,
        config,
    )
    for burn in burns:
        if abs(rendered.burn_times_s[burn.slot_index] - burn.time_s) > 1.0e-9:
            raise PolicyError(
                f"The render put slot {burn.slot_index} at {rendered.burn_times_s[burn.slot_index]!r}"
                f" s, not the {burn.time_s!r} s being corrected"
            )
    if rendered.mission_end_s < arrival_s:
        raise PolicyError(
            f"The render ends at {rendered.mission_end_s!r} s, before the planned arrival "
            f"{arrival_s!r} s"
        )
    flown_scenario, flown_document = parse_gmat_script(
        script, config, probe=scenario.state_probe
    )
    diagnostic = add_diagnostic_report(
        flown_document, flown_scenario, config.gmat.diagnostic_report_name
    )
    sibling = directory / "diagnostic_sibling.script"
    sibling.write_bytes(diagnostic.script_bytes)
    run = run_gmat(
        gmat_console, sibling, directory / "run", [diagnostic.report_filename],
        startup_file=startup_file, timeout_s=config.gmat.timeout_s, on_started=on_run,
        file_size_limit_bytes=file_size_limit_for(scenario),
        template_dir=document.path.resolve().parent,
    )
    if not run.success:
        raise GmatRunError(run.error or "GMAT correction run failed")
    return _read_flight(
        run.reports[diagnostic.report_filename], scenario, config, burns,
        maneuvers_flown(flown_scenario), arrival_s, run,
    )


def _read_flight(
    path: str | Path,
    scenario: Scenario,
    config: OrbitWinConfig,
    burns: tuple[Burn, ...],
    maneuvers: Sequence[FlownManeuver],
    arrival_s: float,
    run: GmatRunResult,
) -> _Flight:
    report = parse_numeric_report(path)
    clock = report_clock(report, scenario.pursuer.name, scenario.target.name, scenario.alignment)
    # From t = 0; the alignment's rows are checked for altitude below.
    joint = slice(clock.frozen_rows, None)
    pursuer_early = (
        scenario.alignment is not None and scenario.alignment.early == scenario.pursuer.name
    )
    # The pursuer's own rows before t = 0 when it flies the alignment; none otherwise.
    before = slice(0, clock.frozen_rows if pursuer_early else 0)
    times = clock.times_s[joint]
    pursuer_r, pursuer_v = craft_columns(report, scenario.pursuer.name)
    target_r, target_v = craft_columns(report, scenario.target.name)
    early, maneuvers = split_at_alignment(maneuvers)
    before_t, before_r, before_v, _, _ = with_post_burn_rows(
        clock.times_s[before], pursuer_r[before], pursuer_v[before], target_r[before],
        target_v[before], early, str(path),
    )
    if len(early) and not len(before_t):
        raise GmatRunError(f"{path} has no rows of {scenario.pursuer.name}'s head start")
    pursuer_r, pursuer_v = pursuer_r[joint], pursuer_v[joint]
    target_r, target_v = target_r[joint], target_v[joint]
    times, pursuer_r, pursuer_v, target_r, target_v = with_post_burn_rows(
        times, pursuer_r, pursuer_v, target_r, target_v, maneuvers, str(path)
    )
    last_burn_s = max(burn.time_s for burn in burns)
    # From the last burn on; of its two rows the later is after the Maneuver.
    after = times >= last_burn_s - BURN_EPOCH_TOLERANCE_S
    pursuer = ephemeris_from_rows(
        times[after], pursuer_r[after], pursuer_v[after], f"{path} for {scenario.pursuer.name}"
    )
    target = ephemeris_from_rows(
        times[after], target_r[after], target_v[after], f"{path} for {scenario.target.name}"
    )
    miss = pursuer.state_at(arrival_s).r - target.state_at(arrival_s).r
    # As verification measures it: a hit's closest approach is its first passage's.
    event = relative_event(
        times, pursuer_r - target_r, pursuer_v - target_v, config.policy.intercept_radius_km,
        stop_depth_km=config.policy.stop_depth_km,
    )
    # The pursuer's distance from the centre, on the same Hermite interpolant.
    ground = evaluate_hermite_samples(
        np.concatenate((before_t, times)),
        np.concatenate((before_r, pursuer_r)),
        np.concatenate((before_v, pursuer_v)),
        config.policy.central_body_radius_km,
    )
    # Between rows, from an alignment the pursuer flies through the row after the
    # first entry or t_f: where the submitted script stops, which verify then
    # measures to its end.
    end_s = event.first_entry_time_s if event.first_entry_time_s is not None else arrival_s
    cut = min(len(times), int(np.searchsorted(times, end_s, "right")) + 1)
    floor = evaluate_hermite_samples(
        np.concatenate((before_t, times[:cut])),
        np.concatenate((before_r, pursuer_r[:cut])),
        np.concatenate((before_v, pursuer_v[:cut])),
        config.policy.central_body_radius_km,
    )
    return _Flight(
        burns=burns,
        miss_km=np.asarray(miss, dtype=float),
        event=event,
        minimum_radius_km=ground.minimum_distance_km,
        minimum_radius_time_s=ground.minimum_distance_time_s,
        floor_radius_km=floor.minimum_distance_km,
        floor_radius_time_s=float(floor.minimum_distance_time_s),
        run=run,
    )


def _corrected_candidate(
    original: Candidate,
    flight: _Flight,
    scenario: Scenario,
    config: OrbitWinConfig,
    arrival_s: float,
    aim_km: float,
    converged: bool,
    round_number: int,
) -> Candidate:
    """The flown schedule as a Candidate carrying what GMAT measured on it."""
    schedule = [(burn.time_s, np.asarray(burn.dv_inertial_km_s, dtype=float))
                for burn in flight.burns]
    two_body_target = np.asarray(
        propagate_universal(scenario.target.initial_state, arrival_s, scenario.mu_km3_s2).r,
        dtype=float,
    )
    two_body_miss_km = float(
        np.linalg.norm(_two_body_arrival(scenario, schedule, arrival_s) - two_body_target)
    )
    event = flight.event
    metadata: dict[str, Any] = {
        "event_refined": True,
        "minimum_pursuer_radius_km": flight.minimum_radius_km,
        "corrected_from": original.candidate_id,
        "gmat_correction": {
            "planned_arrival_s": arrival_s,
            "aim_km": aim_km,
            "converged": converged,
            "round": round_number,
            "run_id": flight.run.run_id,
            "closest_approach_km": event.minimum_distance_km,
            "closest_approach_time_s": event.minimum_distance_time_s,
            "miss_at_arrival_km": float(np.linalg.norm(flight.miss_km)),
            "two_body_miss_at_arrival_km": two_body_miss_km,
        },
    }
    # The archive stamps certify checks, for the same script and horizon.
    for key in ("scenario_sha256", "horizon_s", "time_origin_offset_s",
                "alignment_minimum_distance_km", "pursuer_model"):
        if key in original.metadata:
            metadata[key] = original.metadata[key]
    window = (
        None if scenario.alignment_minimum_distance_km is None
        else head_start_alignment_distance_km(scenario, flight.burns)
    )
    if window is not None:
        metadata[HEAD_START_ALIGNMENT_KEY] = window
    # Kept only when the original has it: it is part of the candidate id.
    if "arrival_time_s" in original.metadata:
        metadata["arrival_time_s"] = original.metadata["arrival_time_s"]
    metadata["gmat_correction"]["first_entry_time_s"] = event.first_entry_time_s
    if not converged:
        # Measured only to t_f + margin and short of the aim: not a hit to rank.
        metadata["event_refined"] = False
        unconverged = Candidate(
            burns=flight.burns,
            intercept_time_s=None,
            minimum_distance_km=None,
            distance_at_completion_km=None,
            successful=False,
            analytical_endpoint_error_km=two_body_miss_km,
            branch=original.branch,
            metadata=metadata,
        )
        apply_constraints(unconverged, scenario, config)
        return unconverged
    corrected = Candidate(
        burns=flight.burns,
        intercept_time_s=event.first_entry_time_s,
        minimum_distance_km=event.minimum_distance_km,
        distance_at_completion_km=config.policy.intercept_radius_km,
        successful=True,
        analytical_endpoint_error_km=two_body_miss_km,
        branch=original.branch,
        metadata=metadata,
    )
    apply_constraints(corrected, scenario, config)
    apply_score(corrected, scenario, config.score)
    return corrected
