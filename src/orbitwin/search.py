from __future__ import annotations

import itertools
import math
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np
from scipy.stats import qmc

from .config import OrbitWinConfig, SearchConfig
from .constraints import FORMAL_FRONTIER, ROBUST_FRONTIER, apply_constraints, frontier_of
from .errors import OrbitWinError
from .event_detection import evaluate_analytical_trajectory
from .frames import inertial_to_script_components
from .lambert import solve_lambert_all
from .models import (
    Burn,
    Candidate,
    CandidateStage,
    Scenario,
    StateVector,
    as_vector3,
    burn_frame,
    template_maneuvers,
)
from .orbit import propagate_universal
from .pareto import prune_dominated
from .pursuer_model import fly as fly_in_model
from .pursuer_model import pursuer_model_for
from .score import apply_score, measured_minimum_distance_km, ranking_key
from .trajectory import state_with_burns_at


def _candidate_order(
    candidate: Candidate, *, scored: bool, scenario: Scenario | None
) -> tuple[float, float, float, float, str] | tuple[float, float, float, str]:
    """Rank within the search. ``scored`` describes the run, not the candidate.

    See pipeline._sort_key: branching per candidate makes keys of different
    shapes that still compare, silently ordering by unrelated quantities.

    The two shapes in the return type are that hazard written down. They are
    the scored key and the unscored one, five elements against four, and a
    sort that mixed them would compare a delta-v against a score. Nothing
    does, because every caller binds ``scored`` with partial() before the
    sort, so one whole sort gets one shape. Both end in the candidate id,
    which is what makes each of them total.
    """
    if scored:
        # Official ranking: score, then minimum distance, total delta-v,
        # completion time.
        return ranking_key(candidate, scenario=scenario)
    # Without announced coefficients there is no score to rank on, so fall back
    # to the time/delta-v Pareto ordering the archive is built around.
    return (
        candidate.total_dv_km_s,
        candidate.intercept_time_s if candidate.intercept_time_s is not None else math.inf,
        measured_minimum_distance_km(candidate),
        candidate.candidate_id,
    )


def _unmeasured(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop what GMAT measured on one exact script; a variant is another script."""
    return {key: value for key, value in metadata.items() if key != "gmat_event"}


#: Roughly how much coarser than the dense grid the tail beyond dense_max is.
#: Not a divisor that is applied exactly -- see _tail_intervals.
_TAIL_COARSENING = 3
#: Fewest intervals the tail is ever cut into, so that even a two-point dense
#: grid gets a tail with an interior point rather than only its far end.
_MIN_TAIL_INTERVALS = 2


def _tail_intervals(grid_burns: int) -> int:
    """How many intervals the coarse tail beyond ``dense_max`` is cut into.

    This has to nest the same way the dense grid does. ``linspace(a, b, n)`` is
    contained in ``linspace(a, b, m)`` exactly when ``n - 1`` divides ``m - 1``,
    which is why the profiles step by ``n -> 2n - 1``: the interval count
    doubles and every old point survives. The tail is a ``linspace`` over
    ``[dense_max, burn_max]`` on the same two endpoints, so its interval count
    has to double along with the dense one or the tail samples are thrown away
    on the way up the ladder.

    ``one_burn_grid_burns // 3`` did not. 23 // 3 = 7 against 45 // 3 = 15, and
    7 does not divide 15, so standard's eight tail points and deep's sixteen
    shared only the two endpoints -- the same trap the counts themselves used
    to be in. Measured on all three bundled scenarios, 282 of standard's 1687
    (burn epoch, arrival epoch) pairs were absent from deep's 6771, every one
    of them beyond ``dense_max``.

    Doubling exactly on every rung, for every base count, rules out plain
    division: ``(2d) // 3 == 2 * (d // 3)`` fails whenever ``d % 3 == 2``, and
    44 % 3 is 2, so a fix anchored on today's numbers would break on the rung
    above deep. Powers of two cannot fail that way. Take the smallest power of
    two at least ``dense_intervals / 3``: doubling the dense intervals doubles
    it, because if ``p`` is the smallest power of two with ``3p >= d`` then
    ``3(2p) >= 2d`` and ``3p < 2d``, leaving ``2p`` as the smallest for ``2d``.
    The price is that the tail now gets between a third and two thirds as many
    intervals as the dense region rather than exactly a third: 8 for standard's
    22 and 16 for deep's 44, one more than before in each case.

    The floor binds below ``grid_burns = 5``: a dense grid of at most three
    intervals asks for fewer tail intervals than the floor allows, so those
    counts all get two and the doubling identity does not hold there. It is
    kept because it is the floor the previous formula had, and lowering it to
    make an identity tidy would thin the tail of exactly the smallest profiles.
    """
    dense_intervals = max(1, grid_burns - 1)
    intervals = _MIN_TAIL_INTERVALS
    while _TAIL_COARSENING * intervals < dense_intervals:
        intervals *= 2
    return intervals


def _time_samples(scenario: Scenario, config: OrbitWinConfig) -> list[tuple[float, float]]:
    search = config.search
    tmax = scenario.max_time_s
    pairs: set[tuple[float, float]] = set()
    # Preserve organizer-provided timeline landmarks as deterministic seeds.
    cumulative = 0.0
    segment_boundaries = [0.0]
    for segment in scenario.propagate_segments:
        cumulative += segment.duration_s
        segment_boundaries.append(cumulative)
    for burn_index, burn_time in enumerate(segment_boundaries[:-1]):
        for arrival in segment_boundaries[burn_index + 1 :]:
            if (
                burn_time >= config.policy.min_propagate_duration_s
                and arrival - burn_time >= search.min_tof_s
                and arrival <= tmax
            ):
                pairs.add((round(float(burn_time), 9), round(float(arrival), 9)))
    # The whole legal window. A burn later than tmax - min_tof cannot reach any
    # arrival inside the horizon; anything earlier than the segment floor renders
    # a zero-length first Propagate, which hangs GMAT. Between those two bounds
    # every epoch is legal and none is excluded.
    #
    # Reach and density are separated deliberately. Spreading a fixed budget
    # uniformly over the wider window thins the region where solutions actually
    # concentrate: measured on DifferentPlane_Circular, that alone cost 2 of 3
    # candidates under the minimal profile. So the dense region keeps its full
    # grid and the tail is sampled in addition to it, never instead of it.
    burn_min = config.policy.min_propagate_duration_s
    burn_max = max(burn_min, tmax - search.min_tof_s)
    dense_max = max(burn_min, min(burn_max, tmax * search.dense_window_fraction))
    burn_grid = np.linspace(burn_min, dense_max, search.one_burn_grid_burns)
    if burn_max > dense_max:
        # dense_max is already in burn_grid, so the tail contributes its
        # interior points and burn_max.
        tail_intervals = _tail_intervals(search.one_burn_grid_burns)
        burn_grid = np.concatenate(
            (burn_grid, np.linspace(dense_max, burn_max, tail_intervals + 1)[1:])
        )
    for burn_time in burn_grid:
        available = tmax - burn_time
        if available < search.min_tof_s:
            continue
        arrivals = np.linspace(
            burn_time + search.min_tof_s,
            tmax,
            search.one_burn_grid_arrivals,
        )
        for arrival in arrivals:
            pairs.add((round(float(burn_time), 9), round(float(arrival), 9)))

    def _sobol_pairs(low: float, high: float, count: int, seed: int) -> None:
        if count <= 0 or high <= low:
            return
        exponent = math.ceil(math.log2(max(1, count)))
        engine = qmc.Sobol(d=2, scramble=True, seed=seed)
        for u_burn, u_tof in engine.random_base2(exponent)[:count]:
            burn_time = low + float(u_burn) * (high - low)
            remaining = tmax - burn_time - search.min_tof_s
            if remaining <= 0.0:
                continue
            arrival = burn_time + search.min_tof_s + float(u_tof) * remaining
            pairs.add((round(burn_time, 9), round(arrival, 9)))

    # As with the grid: the dense region keeps its full sample budget and the
    # tail is sampled in addition, so widening the reach cannot thin the region
    # where solutions concentrate.
    _sobol_pairs(burn_min, dense_max, search.one_burn_samples, search.seed)
    _sobol_pairs(dense_max, burn_max, search.one_burn_samples // 4, search.seed + 101)
    return sorted(pairs)


def union_seeds(search: SearchConfig) -> tuple[int, ...]:
    """The Sobol seeds one sweep is repeated with, in a fixed order.

    The first is the configured seed itself, so `seed_count = 1` reproduces the
    single-seed run exactly. The rest come from numpy's SeedSequence, which is
    what makes the draws independent rather than merely different: consecutive
    integers would also risk landing on the offsets the search already adds
    internally (+101 here, +17/+4013/+8017 in the two-burn search), which would
    spend a whole sweep re-drawing points we already have.
    """
    if search.seed_count < 1:
        # Coercing to one sweep was silent, and silence here means the operator
        # asked for repeated draws, did not get them, and was told the run
        # succeeded. validate_config refuses this too; the flag path reaches
        # here first.
        raise OrbitWinError(
            f"search.seed_count must be at least 1, not {search.seed_count}"
        )
    if search.seed_count == 1:
        return (search.seed,)
    extra = np.random.SeedSequence(search.seed).generate_state(
        search.seed_count - 1, dtype=np.uint32
    )
    return (search.seed, *(int(value) for value in extra))


def zero_burn_candidate(scenario: Scenario, config: OrbitWinConfig) -> Candidate:
    event = evaluate_analytical_trajectory(
        scenario.pursuer.initial_state,
        scenario.target.initial_state,
        (),
        scenario.max_time_s,
        scenario.mu_km3_s2,
        config.policy.intercept_radius_km,
        max_step_s=config.search.event_step_s,
        # Measured here too. The baseline carries event_refined, so every gate
        # treats it as evaluated -- and flies_through_central_body reads an
        # absent measurement as clear. A start may dive (the parser only refuses
        # one inside the body), and this is the measurement that withholds it.
        central_body_radius_km=config.policy.central_body_radius_km,
    )
    candidate = Candidate(
        burns=(),
        intercept_time_s=event.first_entry_time_s,
        minimum_distance_km=event.minimum_distance_km,
        distance_at_completion_km=(
            config.policy.intercept_radius_km
            if event.first_entry_time_s is not None
            else event.distance_at_end_km
        ),
        successful=event.first_entry_time_s is not None,
        analytical_endpoint_error_km=None,
        branch="zero-burn",
        # Evaluated over the full horizon, so this verdict is a trajectory event.
        metadata={
            "arrival_time_s": scenario.max_time_s,
            "event_refined": True,
            "minimum_pursuer_radius_km": float(event.minimum_pursuer_radius_km)
            if event.minimum_pursuer_radius_km is not None
            else None,
        },
    )
    # The zero-burn baseline went through no constraint check at all, which also
    # left it without a frontier tag. It has no burns so it cannot violate a
    # burn rule, but it should reach the archive by the same path as everything
    # else rather than by omission.
    apply_constraints(candidate, scenario, config)
    apply_score(candidate, scenario, config.score)
    return candidate


class ParallelSearchError(OrbitWinError):
    """A worker process died. Re-running with workers=1 gives the same answer."""

    code = "search"


def resolve_workers(requested: int) -> int:
    """Turn a configured worker count into a usable one.

    Zero or less means "use the machine": one fewer than the CPU count, so a
    long sweep does not take the box away from whatever else is running.

    The useful ceiling is physical cores, not threads. Measured on an i5-14500
    (14 physical, 20 logical) against the 2026-09-05 scenario, once the local
    refinement stopped running its pairs on one core:

        workers   1      4      8      14     19
        seconds   23.39  8.17   6.68   5.75   5.75
        speedup   1.00x  2.86x  3.50x  4.07x  4.07x

    14 and 19 are the same number, so the five extra logical CPUs buy nothing
    and a caller who wants the rest of the box can set the physical count by
    hand. The knee is at 8, which reaches 86% of the ceiling on 57% of the
    machine.

    Before that change the same curve saturated at 3.12x, and 19 logical read
    1.6% *faster* than 14 physical -- which looked like evidence against this
    paragraph and was really the serial section talking.
    """
    if requested > 0:
        return requested
    return max(1, (os.cpu_count() or 1) - 1)


#: What every task in one pool run shares. The pool's initializer sets these
#: once per worker process, rather than pickling the same scenario and config
#: onto every task of the map. Both stay None in the parent, which never runs
#: the initializer and never calls the workers below.
_WORKER_SCENARIO: Scenario | None = None
_WORKER_CONFIG: OrbitWinConfig | None = None


def _init_worker(scenario: Scenario, config: OrbitWinConfig) -> None:
    global _WORKER_SCENARIO, _WORKER_CONFIG
    _WORKER_SCENARIO = scenario
    _WORKER_CONFIG = config


def _worker_context() -> tuple[Scenario, OrbitWinConfig]:
    """What _init_worker left in this process.

    Every pool that submits the workers below passes _init_worker, so this
    never refuses in practice. It is written as a refusal rather than an
    assert because -O strips assert, and a worker that had somehow started
    uninitialised would otherwise report the miss as an AttributeError raised
    from somewhere inside the physics.
    """
    scenario, config = _WORKER_SCENARIO, _WORKER_CONFIG
    if scenario is None or config is None:
        raise ParallelSearchError(
            "A search worker process was started without its initializer, so it "
            "has no scenario to solve. The result is unchanged with "
            "search.workers = 1, which is the fallback to use."
        )
    return scenario, config


def _candidates_for_pair(
    scenario: Scenario,
    config: OrbitWinConfig,
    burn_time: float,
    arrival_time: float,
    limit: float,
) -> list[Candidate]:
    """Every legal one-burn candidate for one (burn epoch, arrival) pair.

    Pure: the same inputs give the same list, which is what lets the sweep be
    split across processes without changing the result.
    """
    frame = burn_frame(scenario, 0)
    found: list[Candidate] = []
    tof = arrival_time - burn_time
    if tof < config.search.min_tof_s:
        return found
    pursuer_at_burn = propagate_universal(
        scenario.pursuer.initial_state, burn_time, scenario.mu_km3_s2
    )
    target_at_arrival = propagate_universal(
        scenario.target.initial_state, arrival_time, scenario.mu_km3_s2
    )
    solutions = solve_lambert_all(
        pursuer_at_burn.r,
        target_at_arrival.r,
        tof,
        scenario.mu_km3_s2,
        max_revolutions=config.search.max_revolutions,
        endpoint_tolerance_km=config.search.endpoint_tolerance_km,
        reference_normal=np.cross(pursuer_at_burn.r, pursuer_at_burn.v),
    )
    for solution in solutions:
        dv = solution.v1 - pursuer_at_burn.v
        magnitude = float(np.linalg.norm(dv))
        if magnitude > limit + 1.0e-12:
            continue
        script_components = inertial_to_script_components(dv, pursuer_at_burn, *frame)
        burn = Burn(
            time_s=burn_time,
            dv_inertial_km_s=as_vector3(dv),
            dv_script_km_s=as_vector3(script_components),
            slot_index=0,
            frame=frame,
        )
        candidate = Candidate(
            burns=(burn,),
            intercept_time_s=arrival_time,
            minimum_distance_km=solution.endpoint_error_km,
            distance_at_completion_km=min(
                config.policy.intercept_radius_km, solution.endpoint_error_km
            ),
            successful=solution.endpoint_error_km <= config.policy.intercept_radius_km,
            analytical_endpoint_error_km=solution.endpoint_error_km,
            branch=solution.branch,
            metadata={
                "burn_time_s": burn_time,
                "arrival_time_s": arrival_time,
                "tof_s": tof,
                "lambert_z": solution.z,
                "lambert_revolution": solution.revolution,
                # Until the refinement pass below runs, `successful` and
                # `minimum_distance_km` describe the Lambert endpoint
                # residual, not a 5 km entry on the actual trajectory.
                "event_refined": False,
            },
        )
        if not apply_constraints(candidate, scenario, config).valid:
            continue
        apply_score(candidate, scenario, config.score)
        found.append(candidate)
    return found


def _event_worker(request: tuple[tuple[Burn, ...], float]) -> object:
    burns, arrival = request
    scenario, config = _worker_context()
    return evaluate_analytical_trajectory(
        scenario.pursuer.initial_state,
        scenario.target.initial_state,
        burns,
        arrival,
        scenario.mu_km3_s2,
        config.policy.intercept_radius_km,
        max_step_s=config.search.event_step_s,
        # The serial path passes this too. A worker that measured something
        # different would make the answer depend on the worker count.
        central_body_radius_km=config.policy.central_body_radius_km,
    )


def evaluate_events(
    scenario: Scenario,
    config: OrbitWinConfig,
    requests: list[tuple[tuple[Burn, ...], float]],
    workers: int,
) -> list[object]:
    """Evaluate one trajectory event per request, in request order.

    This is where the search actually spends its time. On the different-plane
    scenario under standard.yaml the Lambert sweep is 7.4 s of 51.2 s while the
    refinement below it is 43.8 s, so parallelising the sweep alone moves almost
    nothing.

    Each evaluation is independent and pure, and map() yields in input order, so
    the result is the serial list element for element whatever the worker count.
    """
    if not requests:
        return []
    if workers <= 1 or len(requests) < 2:
        return [
            evaluate_analytical_trajectory(
                scenario.pursuer.initial_state,
                scenario.target.initial_state,
                burns,
                arrival,
                scenario.mu_km3_s2,
                config.policy.intercept_radius_km,
                max_step_s=config.search.event_step_s,
                central_body_radius_km=config.policy.central_body_radius_km,
            )
            for burns, arrival in requests
        ]
    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(requests)),
            initializer=_init_worker,
            initargs=(scenario, config),
        ) as pool:
            return list(pool.map(_event_worker, requests))
    except BrokenProcessPool as exc:
        # Do not silently fall back. The answer would be right but the operator
        # would never learn the machine could not sustain the worker count.
        raise ParallelSearchError(
            f"A search worker process died ({exc}). The result is unchanged with "
            "search.workers = 1, which is the fallback to use."
        ) from exc


def refine_events(
    scenario: Scenario,
    config: OrbitWinConfig,
    targets: list[Candidate],
    workers: int,
) -> list[object]:
    """Evaluate each candidate's trajectory event, extending failures to Tmax.

    A one-burn arc is evaluated to its Lambert arrival, which is where the
    interception is meant to happen and is much cheaper than the full horizon.
    That is right for a candidate that intercepts and wrong for one that does
    not, twice over.

    初賽規則 §4 sets Tteam = 4TA when there is no interception, so the distance
    term of the score is Δr(4TA) -- and stopping at the arrival measures it
    somewhere else entirely. The rules also accept an interception at any
    t <= Tmax, so an approach that only closes after the arrival epoch is a
    legal solution the first pass cannot see.

    Only candidates that failed at their arrival pay for the second pass.
    """
    if not targets:
        return []
    events = list(
        evaluate_events(
            scenario,
            config,
            [
                (candidate.burns, float(candidate.metadata["arrival_time_s"]))
                for candidate in targets
            ],
            workers,
        )
    )
    late = [
        index
        for index, event in enumerate(events)
        if event.first_entry_time_s is None  # type: ignore[attr-defined]
        and float(targets[index].metadata["arrival_time_s"]) < scenario.max_time_s - 1.0e-9
    ]
    if late:
        extended = evaluate_events(
            scenario,
            config,
            [(targets[index].burns, scenario.max_time_s) for index in late],
            workers,
        )
        for index, event in zip(late, extended, strict=True):
            events[index] = event
    return events


def record_event(candidate: Candidate, event: object, config: OrbitWinConfig) -> None:
    """Copy a trajectory event onto the candidate it was measured for.

    On success the scored distance is the interception radius exactly: 初賽規則
    §4 sets Δrmin = max(Δr(Tteam), 5) and tint is by definition the first
    instant with Δr <= 5. On failure it is the separation where the evaluation
    ended, which refine_events extends to Tmax so that it is the Δr(4TA) §4 asks
    for. The caller still has to re-score; this only moves the measurements.
    """
    candidate.intercept_time_s = event.first_entry_time_s  # type: ignore[attr-defined]
    candidate.minimum_distance_km = event.minimum_distance_km  # type: ignore[attr-defined]
    candidate.distance_at_completion_km = (
        config.policy.intercept_radius_km
        if event.first_entry_time_s is not None  # type: ignore[attr-defined]
        else event.distance_at_end_km  # type: ignore[attr-defined]
    )
    candidate.successful = event.first_entry_time_s is not None  # type: ignore[attr-defined]
    candidate.metadata["event_refined"] = True
    candidate.metadata["event_samples_inside_radius"] = event.samples_inside_radius  # type: ignore[attr-defined]
    radius_km = event.minimum_pursuer_radius_km  # type: ignore[attr-defined]
    if radius_km is not None:
        candidate.metadata["minimum_pursuer_radius_km"] = float(radius_km)
        if float(radius_km) < config.policy.central_body_radius_km:
            # Point-mass gravity integrates straight through the Earth, so the
            # search will happily buy an interception with a trajectory that
            # passes underground -- a large retrograde impulse from a 7000 km
            # circular orbit puts pericenter near 3300 km. The rules do not
            # forbid it and GMAT will not object, which is exactly why it has
            # to be said out loud rather than left in the archive unmarked.
            depth_km = config.policy.central_body_radius_km - float(radius_km)
            candidate.warnings = tuple(
                sorted(
                    {
                        *candidate.warnings,
                        f"Trajectory passes {depth_km:.1f} km below the central-body "
                        f"radius before completion (minimum {float(radius_km):.1f} km)",
                    }
                )
            )


def drop_post_intercept_burns(
    scenario: Scenario,
    config: OrbitWinConfig,
    candidates: list[Candidate],
    workers: int,
) -> list[Candidate]:
    """Emit a variant of any candidate that still burns after it has arrived.

    初賽規則 §5 eq. 1 sums ‖ΔVk‖ over k = 1..N where N is 總機動次數, the number
    of maneuvers, with no clause excluding one that fires after the
    interception. Read as written, an impulse scheduled past tint is charged in
    full while contributing nothing to an interception that already happened.

    Dropping it cannot move that interception. §4 defines tint as the *first*
    entry into the 5 km sphere, and the trajectory up to a later burn does not
    depend on that burn, so the variant keeps the same tint and the same
    Δrmin of exactly 5. It carries strictly less ΔVteam, so under §5 as written
    it scores at least as well; under the opposite reading -- that only burns up
    to tint count -- both carry the same total and the same score. There is no
    reading under which the variant is worth less on the score.

    §6 is the reason this returns a variant instead of editing in place. The
    tie-break on the closest approach reads the whole trajectory, and after the
    dropped burn that is a different trajectory, so the variant is evaluated on
    its own terms and ranking_key chooses. Deciding here would be guessing at
    which of two genuinely different submissions is better.

    A candidate whose burns all fall after tint yields a variant with no burns
    at all, which is the zero-burn baseline arrived at from the other side. It
    is a legal submission and it is kept.

    Variants are deduplicated by their surviving impulses, because that is what
    the renderer writes: two variants carrying the same burns are the same
    script. Without it, a scenario where the pair closes on its own produces one
    copy per affected candidate, and since they tie exactly on both Pareto axes
    none of them dominates another, so every copy survives pruning. A
    constructed co-planar geometry -- 3 km radial gap, initial phase gap sized
    to close at mid-window -- gave 11 identical zero-burn variants across the
    two searches before this.

    Neither bundled scenario produces the shape, and neither do 40 random
    different-plane geometries: 324 one-burn and 248 two-burn refined
    interceptions, not one with a burn past its own first entry. It takes an
    interception that happens *incidentally*, before the maneuver designed to
    cause it, and random phases pass hundreds of km apart. The constructed
    geometry does produce it, and there the wasted impulses reach 1590.8 m/s --
    the whole budget, spent after the mission had already succeeded.
    """
    variants: list[Candidate] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in candidates:
        intercept = candidate.intercept_time_s
        if intercept is None:
            continue
        kept = tuple(burn for burn in candidate.burns if burn.time_s <= intercept + 1.0e-9)
        if len(kept) == len(candidate.burns):
            continue
        signature = tuple(
            (burn.time_s, burn.dv_inertial_km_s, burn.dv_script_km_s, burn.slot_index)
            for burn in kept
        )
        if signature in seen:
            continue
        seen.add(signature)
        variants.append(
            Candidate(
                burns=kept,
                intercept_time_s=None,
                minimum_distance_km=None,
                distance_at_completion_km=None,
                successful=False,
                analytical_endpoint_error_km=candidate.analytical_endpoint_error_km,
                branch=candidate.branch + "/post-intercept-trimmed",
                metadata={**_unmeasured(candidate.metadata), "trimmed_from": candidate.candidate_id},
            )
        )
    if not variants:
        return []
    events = refine_events(scenario, config, variants, workers)
    trimmed: list[Candidate] = []
    for variant, event in zip(variants, events, strict=True):
        record_event(variant, event, config)
        if not apply_constraints(variant, scenario, config).valid:
            continue
        apply_score(variant, scenario, config.score)
        trimmed.append(variant)
    return trimmed


def _pair_chunk_worker(chunk: tuple[tuple[float, float], ...], limit: float) -> list[Candidate]:
    scenario, config = _worker_context()
    return [
        candidate
        for burn_time, arrival_time in chunk
        for candidate in _candidates_for_pair(scenario, config, burn_time, arrival_time, limit)
    ]


def _evaluate_pairs(
    scenario: Scenario,
    config: OrbitWinConfig,
    pairs: Sequence[tuple[float, float]],
    limit: float,
    workers: int,
) -> list[Candidate]:
    """Every candidate the given (burn, arrival) pairs produce.

    Contiguous chunks collected through map(), which yields in input order, so
    the concatenation is the serial list element for element: any worker count
    produces the identical list. Four chunks per worker keeps the tail balanced
    without making the per-task pickling significant.

    Both callers go through here. The local refinement used to run its own
    pairs in a plain loop while the sweep ran its own across the machine, which
    put 963 of the 2,044 pairs a solve evaluates on one core -- most of the
    serial fraction the worker curve was flattening against.
    """
    candidates: list[Candidate] = []
    if workers > 1 and len(pairs) > workers:
        size = max(1, math.ceil(len(pairs) / (workers * 4)))
        chunks = [tuple(pairs[start : start + size]) for start in range(0, len(pairs), size)]
        try:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker, initargs=(scenario, config)
            ) as pool:
                for produced in pool.map(_pair_chunk_worker, chunks, itertools.repeat(limit)):
                    candidates.extend(produced)
        except BrokenProcessPool as exc:
            raise ParallelSearchError(
                f"A search worker process died ({exc}). The result is unchanged with "
                "search.workers = 1, which is the fallback to use."
            ) from exc
    else:
        for burn_time, arrival_time in pairs:
            candidates.extend(
                _candidates_for_pair(scenario, config, burn_time, arrival_time, limit)
            )
    return candidates


def _prune_per_frontier(
    candidates: list[Candidate], keep: int, config: OrbitWinConfig | None = None
) -> list[Candidate]:
    """Prune each frontier against itself, never against the other.

    A shared archive lets the formal frontier crowd out the robust one. A burn
    at 1.499 km/s buys time that nothing at 1.495 can match, so on a (time,
    delta-v) Pareto front it displaces exactly the candidates this project would
    rather submit. Keeping both whole is the point of having two.
    """
    grouped: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        # Compute when a config is available rather than trusting the tag: a
        # candidate that arrived without one would otherwise be grouped as
        # robust and could displace a genuinely robust point.
        name = (
            frontier_of(candidate, config)
            if config is not None
            else str(candidate.metadata.get("frontier", ROBUST_FRONTIER))
        )
        grouped.setdefault(name, []).append(candidate)
    return [
        candidate
        for name in (ROBUST_FRONTIER, FORMAL_FRONTIER)
        for candidate in prune_dominated(grouped.get(name, []), keep)
    ]


def _refinement_targets(
    candidates: list[Candidate], top: int, *, scored: bool, scenario: Scenario
) -> list[Candidate]:
    """The most competitive subset of each frontier, in a stable order."""
    targets: list[Candidate] = []
    for name in (ROBUST_FRONTIER, FORMAL_FRONTIER):
        group = sorted(
            (
                candidate
                for candidate in candidates
                if str(candidate.metadata.get("frontier", ROBUST_FRONTIER)) == name
            ),
            key=partial(_candidate_order, scored=scored, scenario=scenario),
        )
        targets.extend(group[: min(top, len(group))])
    return targets


def _look_around(
    scenario: Scenario,
    config: OrbitWinConfig,
    candidates: list[Candidate],
    limit: float,
    workers: int = 1,
) -> list[Candidate]:
    """Resample near the (burn, arrival) pairs the sweep already liked.

    The sweep's grid is coarse against this landscape -- 2626 s between burn
    epochs and 1478 s between arrivals on the different-plane scenario -- and
    the basins are narrower than that, so which solutions it finds depends on
    where it happens to land. Measured across five Sobol seeds on that
    scenario, the cheapest candidate the production profile finds ranges from
    543.9 to 648.8 m/s: the headline number is substantially a draw.

    Widening the sweep does not fix it, and that is not a guess. linspace(a, b,
    12) and linspace(a, b, 14) share only their endpoints, so a "denser" grid is
    a different sample set rather than a superset -- measured, going to 14x28
    lost three of the eight successes on the high-plane scenario and took its
    cheapest candidate from 812.3 to 1331.7 m/s. Raising the refinement budget
    to 100 or 200 changed neither number.

    This cannot lose anything, which is the property none of those had: it only
    adds candidates, around pairs that are already in hand. On the seed where
    the sweep stops at 648.8 m/s it finds 504.4 -- cheaper than the production
    seed's own best -- and where it finds nothing it costs about half the sweep
    again.
    """
    top = config.search.local_refine_top
    steps = config.search.local_refine_steps
    if top <= 0 or steps <= 1 or not candidates:
        return []
    horizon = scenario.max_time_s
    grid_burns = max(2, config.search.one_burn_grid_burns)
    grid_arrivals = max(2, config.search.one_burn_grid_arrivals)
    burn_cell = horizon * config.search.dense_window_fraction / (grid_burns - 1)
    arrival_cell = horizon / (grid_arrivals - 1)
    offsets_burn = np.linspace(-burn_cell, burn_cell, steps)
    offsets_arrival = np.linspace(-arrival_cell, arrival_cell, steps)

    seeds = sorted(
        candidates,
        key=partial(_candidate_order, scored=config.score is not None, scenario=scenario),
    )[:top]
    pairs: set[tuple[float, float]] = set()
    for candidate in seeds:
        if not candidate.burns:
            continue
        burn_time = float(min(burn.time_s for burn in candidate.burns))
        arrival = float(candidate.metadata.get("arrival_time_s", horizon))
        for db in offsets_burn:
            for da in offsets_arrival:
                pair_burn = round(burn_time + float(db), 9)
                pair_arrival = round(arrival + float(da), 9)
                if pair_burn < config.policy.min_propagate_duration_s:
                    continue
                if pair_arrival > horizon or pair_arrival - pair_burn < config.search.min_tof_s:
                    continue
                pairs.add((pair_burn, pair_arrival))

    return _evaluate_pairs(scenario, config, sorted(pairs), limit, workers)


def search_one_burn(scenario: Scenario, config: OrbitWinConfig) -> list[Candidate]:
    """Sweep one-burn Lambert transfers across both frontiers.

    The sweep runs at the rule boundary and each candidate is classified
    afterwards. Sweeping twice, once per limit, would repeat every Lambert solve
    for the 0.33% of the delta-v range that separates 1.495 from 1.500 km/s.
    """
    limit = config.policy.dv_limit_km_s
    pairs = _time_samples(scenario, config)
    workers = resolve_workers(config.search.workers)
    candidates = _evaluate_pairs(scenario, config, pairs, limit, workers)

    # Before pruning: a local candidate that dominates one from the sweep
    # should be in the archive rather than arriving after the cut.
    candidates.extend(_look_around(scenario, config, candidates, limit, workers))
    candidates = _prune_per_frontier(candidates, config.search.keep_candidates, config)
    # Refine first-entry time and true minimum for the most competitive subset
    # of each frontier, so the formal one cannot consume the whole budget.
    targets = _refinement_targets(
        candidates, config.search.event_refine_top, scored=config.score is not None,
        scenario=scenario,
    )
    events = refine_events(scenario, config, targets, workers)
    for candidate, event in zip(targets, events, strict=True):
        record_event(candidate, event, config)
        apply_score(candidate, scenario, config.score)
    candidates.extend(drop_post_intercept_burns(scenario, config, targets, workers))
    candidates.sort(
        key=partial(_candidate_order, scored=config.score is not None, scenario=scenario)
    )
    return candidates


def select_candidates(
    candidates: Iterable[Candidate], keep: int, config: OrbitWinConfig | None = None
) -> list[Candidate]:
    """Deduplicate and prune, keeping each frontier whole.

    Pruning the two frontiers together here would undo the separation
    search_one_burn just made: a faster candidate bought with delta-v outside
    the production margin dominates the archive it is not allowed to be
    submitted from.
    """
    unique: dict[str, Candidate] = {}
    for candidate in candidates:
        unique[candidate.candidate_id] = candidate
    selected = _prune_per_frontier(list(unique.values()), keep, config)
    selected.sort(
        key=partial(
            _candidate_order,
            scored=config is not None and config.score is not None,
            # No alignment window here: solve sorts what this keeps with the scenario.
            scenario=None,
        )
    )
    return selected


def slot_assignment_variants(
    scenario: Scenario, config: OrbitWinConfig, candidates: Iterable[Candidate]
) -> list[Candidate]:
    """Emit each candidate under every other legal assignment of burns to slots.

    Which ImpulsiveBurn resource carries an impulse does not change the
    trajectory, but it does change renderability. A late single burn in slot 0
    leaves the remaining slots to be scheduled after it, which can run past
    Tmax; the same burn in a later slot, with the earlier ones carrying an exact
    zero vector, is expressible. Searching only slot 0 therefore discarded legal
    solutions for a reason that has nothing to do with the dynamics.

    Generated after the search has pruned. Every Pareto criterion is
    slot-independent, so the surviving set is the same whichever slots are used,
    and generating variants earlier would only dilute the archive.
    """
    slot_count = len(scenario.burn_slots)
    if slot_count < 2 or not template_maneuvers(scenario):
        # Nothing to reassign when the mission maneuvers nothing yet: a burn
        # goes wherever the Propagate is split, so which resource carries it
        # changes no renderability and the variants would be pure duplication.
        return []
    variants: list[Candidate] = []
    for candidate in candidates:
        ordered = sorted(candidate.burns, key=lambda burn: burn.time_s)
        count = len(ordered)
        if count == 0 or count > slot_count:
            continue
        # A plan made in a pursuer model takes its frames on that model's states.
        model_name = candidate.metadata.get("pursuer_model")
        model = None if model_name is None else pursuer_model_for(scenario, str(model_name))
        model_states = (
            None
            if model is None
            else fly_in_model(
                model,
                np.asarray(scenario.pursuer.initial_state.r),
                np.asarray(scenario.pursuer.initial_state.v),
                np.array([burn.time_s for burn in ordered]),
                np.array([burn.dv_inertial_km_s for burn in ordered]),
                ordered[-1].time_s,
            )
        )
        current = tuple(burn.slot_index for burn in ordered)
        # Slot 0 of a template that maneuvers right after its alignment fires at t = 0.
        pinned = scenario.propagate_segments[0].implicit
        for assignment in itertools.combinations(range(slot_count), count):
            if assignment == current:
                continue
            at_zero = abs(ordered[0].time_s) <= 1.0e-9
            if pinned and (assignment[0] == 0) != at_zero:
                continue
            burns: list[Burn] = []
            for index, (burn, slot_index) in enumerate(zip(ordered, assignment, strict=True)):
                slot = scenario.burn_slots[slot_index]
                # The frame is anchored to the state immediately before this
                # impulse, which for a later burn already includes the earlier
                # ones.
                state = (
                    state_with_burns_at(
                        scenario.pursuer.initial_state,
                        candidate.burns,
                        burn.time_s,
                        scenario.mu_km3_s2,
                        include_burn_at_time=False,
                    )
                    if model_states is None
                    else StateVector(
                        as_vector3(model_states.r_at_burns[index]),
                        as_vector3(model_states.v_before[index]),
                    )
                )
                components = inertial_to_script_components(
                    np.asarray(burn.dv_inertial_km_s, dtype=float),
                    state,
                    slot.coordinate_system,
                    slot.origin,
                    slot.axes,
                )
                burns.append(
                    replace(
                        burn,
                        dv_script_km_s=as_vector3(components),
                        slot_index=slot_index,
                        frame=(slot.coordinate_system, slot.origin, slot.axes),
                    )
                )
            # A variant is a different script that GMAT has not run. It must not
            # inherit the source's verification state: dataclasses.replace is a
            # shallow copy, so without this a variant of a verified candidate
            # would carry verified_by_gmat, its stage and its run id while never
            # having been replayed. stage_evidence and score_components are
            # rebuilt because promote() mutates the former in place, which would
            # otherwise write into the source.
            metadata = _unmeasured(candidate.metadata)
            metadata["slot_assignment"] = list(assignment)
            metadata["derived_from"] = candidate.candidate_id
            variants.append(
                replace(
                    candidate,
                    burns=tuple(burns),
                    metadata=metadata,
                    score_components=dict(candidate.score_components),
                    stage_evidence={},
                    _stage=CandidateStage.ANALYTICAL,
                    gmat_run_id=None,
                    candidate_id="",
                )
            )
    return variants
