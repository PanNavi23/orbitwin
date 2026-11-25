"""Search, correct, verify: step 4 of docs/GMAT_IN_THE_LOOP.md.

solve calls these two around its own search. Neither promotes anything:
verify_candidate_with_gmat stays the only way a candidate becomes verified.
"""

from __future__ import annotations

import dataclasses
import heapq
import math
import os
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .config import OrbitWinConfig
from .constraints import (
    ROBUST_FRONTIER,
    flies_through_central_body,
    frontier_of,
)
from .ephemeris import target_ephemeris_from_gmat
from .errors import ConfigurationError, OrbitWinError
from .gmat_parser import GmatScriptDocument
from .gmat_runner import resume_runs, stop_runs
from .gmat_shoot import CorrectionResult, correct_with_gmat
from .models import Candidate, Scenario, head_start_lead_s, head_start_status, uses_head_start
from .pursuer_model import checked_pursuer_model
from .ranking_search import (
    EphemerisTarget,
    RankingSearchResult,
    RankingSolution,
    best_per_margin,
    build_setup,
    build_starts,
    polish_in_model,
    schedule_signature,
    scored_candidate,
    search_ranking,
)
from .score import scored_closest_approach_km

#: The search's own floor when policy.minimum_flown_altitude_km is unset.
DEFAULT_ALTITUDE_FLOOR_KM = 100.0
#: Solutions kept per margin, the hunt script's default. Per margin, because
#: unmargined ones score better analytically and GMAT flies many under the floor.
KEEP_TOP = 15


def ranking_search_candidates(
    scenario: Scenario,
    document: GmatScriptDocument,
    config: OrbitWinConfig,
    output: Path,
    gmat_console: str | Path | None,
    startup_file: str | Path | None,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Run the ranking search, aimed at GMAT's target track when there is a console.

    Returns the candidates, best first, and a record for the summary. The zero
    burn is left out: solve already has it.
    """
    began = time.perf_counter()
    search = config.search
    target = None
    target_error = None
    if gmat_console is not None:
        try:
            target = EphemerisTarget(
                target_ephemeris_from_gmat(
                    scenario, document, config, gmat_console, startup_file,
                    output / "ranking_search_target",
                )
            )
        except (OrbitWinError, OSError) as exc:
            # The conic still seeds the search; verification judges what it finds.
            target_error = f"{type(exc).__name__}: {exc}"
    floor_km = config.policy.minimum_flown_altitude_km
    if floor_km is None:
        floor_km = DEFAULT_ALTITUDE_FLOOR_KM
    margins = [search.ranking_force_model_margin_km]
    if search.ranking_search_without_margin:
        margins.append(0.0)
    setup = build_setup(
        scenario,
        config,
        target=target,
        altitude_floor_km=floor_km,
        force_model_margins_km=margins,
        correction_margin_km_s=search.ranking_correction_margin_km_s,
        screen_iterations=search.ranking_screen_iterations,
        polish_iterations=search.ranking_polish_iterations,
    )
    starts = build_starts(
        setup, config, range(1, min(search.ranking_max_burns, setup.limits.max_burns) + 1),
        transfer_count=search.ranking_transfer_starts,
        random_count=search.ranking_random_starts,
        seed=search.seed,
    )
    workers = search.workers if search.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    model = None
    model_error = None
    try:
        model = checked_pursuer_model(scenario, document, search.ranking_pursuer_model)
    except ConfigurationError as exc:
        # A model measured on other forces would mislead the search: fly two-body.
        model_error = str(exc)
    two_body = search_ranking(
        scenario, config, setup, starts, workers=workers, polish_top=search.ranking_polish_top
    )
    results: list[tuple[RankingSearchResult, Any]] = []
    head_start = head_start_search(
        scenario, config, target, floor_km, margins, workers
    )
    if head_start is not None:
        results.append((head_start, None))
    if model is not None:
        model_setup = dataclasses.replace(setup, pursuer_model=model)
        results.append((
            polish_in_model(
                scenario, config, model_setup, two_body,
                top=search.ranking_pursuer_model_top, workers=workers,
            ),
            model,
        ))
    results.append((two_body, None))
    candidates = []
    for result, flown_in in results:
        chosen: list[RankingSolution] = []
        searched = [s for s in result.solutions if s is not result.baseline]
        per_margin = [
            solution
            for margin in setup.force_model_margins_km
            for solution in [s for s in searched if s.floor_margin_km == margin][:KEEP_TOP]
        ]
        for solution in [*per_margin, *best_per_margin(result).values()]:
            if solution is result.baseline:
                continue
            if all(schedule_signature(solution) != schedule_signature(kept) for kept in chosen):
                chosen.append(solution)
        for solution in chosen:
            candidate = scored_candidate(
                solution, scenario, config, result.setup.limits, flown_in
            )
            candidate.metadata["force_model_margin_km"] = solution.floor_margin_km
            candidates.append(candidate)
    record = {
        "target": "gmat" if target is not None else "two-body",
        "target_error": target_error,
        "starts": len(starts),
        "solutions": sum(len(result.solutions) for result, _ in results),
        "kept": len(candidates),
        "force_model_margins_km": sorted(set(margins)),
        "altitude_floor_km": floor_km,
        "pursuer_model": model.name if model is not None else "two-body",
        "pursuer_model_error": model_error,
        "head_start": {
            "lead_s": head_start_lead_s(scenario),
            "note": head_start_status(scenario),
            "solutions": None if head_start is None else len(head_start.solutions),
            "kept": sum(uses_head_start(candidate.burns) for candidate in candidates),
        },
        "wall_s": time.perf_counter() - began,
    }
    return candidates, record


def head_start_search(
    scenario: Scenario,
    config: OrbitWinConfig,
    target: Any,
    floor_km: float,
    margins: Sequence[float],
    workers: int,
) -> RankingSearchResult | None:
    """The same search on the head start's clock, kept to schedules that burn in it.

    The search without it runs as before, so what solve offers only grows;
    rank weighs a head-start candidate against it (policy.head_start_min_gain_points).
    """
    if head_start_lead_s(scenario) <= 0.0:
        return None
    search = config.search
    setup = build_setup(
        scenario,
        config,
        target=target,
        altitude_floor_km=floor_km,
        force_model_margins_km=margins,
        correction_margin_km_s=search.ranking_correction_margin_km_s,
        screen_iterations=search.ranking_screen_iterations,
        polish_iterations=search.ranking_polish_iterations,
        head_start=True,
    )
    starts = build_starts(
        setup, config, range(1, min(search.ranking_max_burns, setup.limits.max_burns) + 1),
        transfer_count=search.ranking_transfer_starts,
        random_count=search.ranking_random_starts,
        seed=search.seed,
    )
    result = search_ranking(
        scenario, config, setup, starts, workers=workers, polish_top=search.ranking_polish_top
    )
    lead_s = setup.limits.clock_offset_s
    result.solutions = [s for s in result.solutions if s.times_s and s.times_s[0] < lead_s]
    result.baseline = None
    return result


def correction_choices(
    candidates: Sequence[Candidate], config: OrbitWinConfig
) -> list[Candidate]:
    """The first gmat.correct_top hits in `candidates` inside the production
    margins, and as many again of those flown in a pursuer model."""
    hits = [
        c for c in candidates
        if c.successful and c.burns and frontier_of(c, config) == ROBUST_FRONTIER
    ]
    # Plans flown in a pursuer model, and plans that burn in the head start, have
    # budgets of their own, so they never take a two-body plan's place; without
    # them this is hits[:correct_top].
    used: dict[str, int] = {}
    chosen: list[Candidate] = []
    flies = head_start_flies(candidates, config)
    for candidate in hits:
        group = _budget(candidate)
        if group == "head-start" and not flies:
            continue
        if used.get(group, 0) < config.gmat.correct_top:
            used[group] = used.get(group, 0) + 1
            chosen.append(candidate)
    return chosen


#: Share of policy.head_start_min_gain_points a head-start plan must lead by, two-body,
#: to be flown in GMAT at all: two-body and GMAT scores of R8's plans were 0.02 apart.
HEAD_START_LEAD_SHARE = 0.5


def head_start_flies(candidates: Sequence[Candidate], config: OrbitWinConfig) -> bool:
    """Whether head-start plans get GMAT runs: only with a score that can price the
    gate, and only when the best leads the best without one by HEAD_START_LEAD_SHARE of
    it. Before the coefficients step 4a's net takes none (RUNBOOK step 4a)."""
    if config.score is None:
        return False

    def best(early: bool) -> float:
        return max((c.score for c in candidates if c.score is not None and c.burns
                    and uses_head_start(c.burns) == early), default=-math.inf)

    gate = HEAD_START_LEAD_SHARE * config.policy.head_start_min_gain_points
    return best(True) >= max(best(False), max(
        (c.score for c in candidates if c.score is not None and not c.burns), default=-math.inf
    )) + gate


def _budget(candidate: Candidate) -> str:
    """Which GMAT budget a plan draws on."""
    if "pursuer_model" in candidate.metadata:
        return "model"
    return "head-start" if uses_head_start(candidate.burns) else "two-body"


#: Plans this close fly as one: R5 kept 13 copies of a 13-burn miss up to
#: 6.4e-6 km/s and 3.1 ms apart, and GMAT flew the eight tried 0.53 m apart.
#: 5e-5 km/s in one burn component moves a pass about 1 km by R5's T_max.
SAME_PLAN_TIME_S = 0.1
SAME_PLAN_DV_KM_S = 5.0e-5


def _same_plan(a: Candidate, b: Candidate) -> bool:
    if len(a.burns) != len(b.burns):
        return False
    return all(
        abs(x.time_s - y.time_s) <= SAME_PLAN_TIME_S
        and max(abs(p - q) for p, q in zip(x.dv_inertial_km_s, y.dv_inertial_km_s, strict=True))
        <= SAME_PLAN_DV_KM_S
        for x, y in zip(
            sorted(a.burns, key=lambda burn: burn.time_s),
            sorted(b.burns, key=lambda burn: burn.time_s),
            strict=True,
        )
    )


def miss_choices(
    candidates: Sequence[Candidate], config: OrbitWinConfig, *,
    alignment_minimum_distance_km: float | None,
) -> list[Candidate]:
    """The first gmat.verify_misses misses in `candidates` inside the production
    margins, not through the central body, that score above the zero burn, one
    per plan, and as many again of those flown in a pursuer model.

    `candidates` in rank order. Only the analytic score picks them; GMAT's
    decides (R5: an 88.3 km miss flew 113.7 km, under the zero burn). Without
    a score block they are the closest misses that pass nearer than the zero
    burn, on the distance the term prices (Scenario.alignment_minimum_distance_km
    under from_origin): RUNBOOK step 4a certifies the closest beside it.
    """
    if config.gmat.verify_misses <= 0:
        return []
    if config.score is None:
        # Unscored, d_min and delta-v trade at unknown weights; a miss no closer
        # than the zero burn costs more for no distance.
        def scored_km(candidate: Candidate) -> float:
            return scored_closest_approach_km(candidate, alignment_minimum_distance_km)

        floor_km = min(
            (scored_km(c) for c in candidates if c.branch == "zero-burn"), default=math.inf
        )
        pool = sorted(candidates, key=scored_km)

        def better(candidate: Candidate) -> bool:
            return scored_km(candidate) < floor_km
    else:
        # With no zero burn to submit, a miss only has to score.
        floor = max(
            (c.score for c in candidates if c.branch == "zero-burn" and c.score is not None),
            default=0.0,
        )
        pool = list(candidates)

        def better(candidate: Candidate) -> bool:
            return candidate.score is not None and candidate.score > floor
    chosen: dict[str, list[Candidate]] = {"two-body": [], "head-start": [], "model": []}
    flies = head_start_flies(candidates, config)
    for candidate in pool:
        if not flies and uses_head_start(candidate.burns):
            continue
        if (
            candidate.successful
            or not candidate.burns
            or not better(candidate)
            or frontier_of(candidate, config) != ROBUST_FRONTIER
            # No force model holds a pursuer up; the altitude floor is left to GMAT's flight.
            or (
                not config.policy.allow_body_intersecting_submission
                and flies_through_central_body(candidate, config)
            )
        ):
            continue
        group = chosen[_budget(candidate)]
        # Slot variants and copies from other starts: one GMAT run says it for all.
        if len(group) < config.gmat.verify_misses and not any(
            _same_plan(candidate, kept) for kept in group
        ):
            group.append(candidate)
    return [*chosen["two-body"], *chosen["model"], *chosen["head-start"]]


def correct_one(
    candidate: Candidate,
    document: GmatScriptDocument,
    scenario: Scenario,
    config: OrbitWinConfig,
    gmat_console: str | Path,
    startup_file: str | Path | None,
    output: Path,
) -> CorrectionResult | Exception:
    """One correction, one GmatConsole at a time; what it raised, if it did."""
    # Corrections run gmat.workers at a time, so each one measures J alone.
    serial = dataclasses.replace(config, gmat=dataclasses.replace(config.gmat, workers=1))
    try:
        return correct_with_gmat(
            document, scenario, candidate, serial, gmat_console, startup_file,
            output / "gmat_correction" / candidate.candidate_id,
            aim_km=config.gmat.correct_aim_km, max_rounds=config.gmat.correct_rounds,
        )
    except Exception as exc:  # One failed correction must not lose the others.
        return exc


def correction_record(
    original: Candidate, result: CorrectionResult | Exception
) -> tuple[Candidate | None, dict[str, Any]]:
    """The converged correction of `original`, if any, and the attempt's record."""
    if isinstance(result, Exception):
        return None, {
            "candidate_id": original.candidate_id,
            "converged": False,
            "corrected_id": None,
            "closest_approach_km": None,
            "reason": f"the correction raised {type(result).__name__}: {result}",
            "error": f"{type(result).__name__}: {result}",
            "first_flight_closest_km": None,
            "rounds": None,
            "gmat_runs": None,
            "wall_s": None,
            "traceback": "".join(traceback.format_exception(result)),
        }
    fixed = result.candidate if result.converged else None
    first = result.log[0] if result.log else None
    return fixed, {
        "candidate_id": original.candidate_id,
        "converged": result.converged,
        "corrected_id": fixed.candidate_id if fixed is not None else None,
        "closest_approach_km": result.closest_approach_km,
        # GMAT's flight of the original, unchanged.
        "first_flight_closest_km": (
            first.closest_approach_km
            if first is not None and first.kind == "flight" else None
        ),
        "reason": result.reason,
        # A round that raised ends it with a traceback; a GMAT verdict does not.
        "error": result.reason if result.traceback is not None else None,
        "rounds": result.rounds,
        "gmat_runs": result.gmat_runs,
        "wall_s": result.wall_s,
    }


class GmatQueue:
    """Jobs run at most `workers` at a time, each holding one GmatConsole at a time.

    A waiting job of a lower tier starts before any of a higher one; within a
    tier, in the order added. `done` runs on the calling thread, in completion
    order, and may add jobs.
    """

    def __init__(self, workers: int) -> None:
        if workers < 1:
            raise ValueError(f"workers {workers!r} must be at least 1")
        self.workers = workers
        self._waiting: list[tuple[int, int, Callable[[], Any], Callable[[Any], None] | None]] = []
        self._added = 0

    def add(
        self, tier: int, job: Callable[[], Any], done: Callable[[Any], None] | None = None
    ) -> None:
        heapq.heappush(self._waiting, (tier, self._added, job, done))
        self._added += 1

    def waiting(self, tier: int) -> int:
        """Jobs of `tier` not yet started."""
        return sum(entry[0] == tier for entry in self._waiting)

    def run(self) -> None:
        """Until nothing waits or runs. A job, a `done` or an interrupt that
        raises kills the GmatConsole runs in flight and refuses new ones while
        the running jobs return; the jobs still waiting never start."""
        running: dict[Future[Any], tuple[int, Callable[[Any], None] | None]] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            try:
                while self._waiting or running:
                    while self._waiting and len(running) < self.workers:
                        _, order, job, done = heapq.heappop(self._waiting)
                        running[pool.submit(job)] = (order, done)
                    finished, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in sorted(finished, key=lambda f: running[f][0]):
                        _, done = running.pop(future)
                        value = future.result()
                        if done is not None:
                            done(value)
            except BaseException:
                # Unstopped, a running correction would fly round after round
                # past a Ctrl-C. Runs stay refused until every job has returned,
                # however many more interrupts arrive.
                try:
                    while True:
                        try:
                            stop_runs()
                            pool.shutdown(wait=True)
                            break
                        except BaseException:
                            continue
                finally:
                    resume_runs()
                raise
