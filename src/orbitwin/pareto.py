from __future__ import annotations

import math

from .models import Candidate


def dominates(left: Candidate, right: Candidate, tolerance: float = 1.0e-12) -> bool:
    left_time = left.intercept_time_s if left.intercept_time_s is not None else math.inf
    right_time = right.intercept_time_s if right.intercept_time_s is not None else math.inf
    not_worse = (
        left_time <= right_time + tolerance
        and left.total_dv_km_s <= right.total_dv_km_s + tolerance
    )
    strictly_better = (
        left_time < right_time - tolerance
        or left.total_dv_km_s < right.total_dv_km_s - tolerance
    )
    return not_worse and strictly_better


def pareto_frontier(candidates: list[Candidate]) -> list[Candidate]:
    successful = [candidate for candidate in candidates if candidate.successful]
    successful.sort(
        key=lambda candidate: (
            candidate.intercept_time_s if candidate.intercept_time_s is not None else math.inf,
            candidate.total_dv_km_s,
            candidate.candidate_id,
        )
    )
    frontier: list[Candidate] = []
    best_dv = math.inf
    for candidate in successful:
        if candidate.total_dv_km_s < best_dv - 1.0e-12:
            frontier.append(candidate)
            best_dv = candidate.total_dv_km_s
    return frontier


def prune_dominated(candidates: list[Candidate], keep: int) -> list[Candidate]:
    frontier = pareto_frontier(candidates)
    frontier_ids = {candidate.candidate_id for candidate in frontier}
    remainder = [candidate for candidate in candidates if candidate.candidate_id not in frontier_ids]
    remainder.sort(
        key=lambda candidate: (
            not candidate.successful,
            candidate.total_dv_km_s,
            candidate.intercept_time_s if candidate.intercept_time_s is not None else math.inf,
            candidate.candidate_id,
        )
    )
    return (frontier + remainder)[:keep]
