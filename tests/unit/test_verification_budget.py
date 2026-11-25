from __future__ import annotations

from orbitwin.models import Burn, Candidate
from orbitwin.pipeline import _distinct_trajectories_first

# Which candidates GMAT replays. On the day every run is time on a machine that
# is not ours, so the budget is the scarce thing.


def _candidate(time_s: float, dv: tuple[float, float, float], slot: int, *, hit: bool = True):
    burn = Burn(time_s=time_s, dv_inertial_km_s=dv, dv_script_km_s=dv, slot_index=slot)
    return Candidate(
        burns=(burn,), intercept_time_s=1000.0 if hit else None,
        minimum_distance_km=0.5 if hit else 90.0,
        distance_at_completion_km=5.0 if hit else 90.0, successful=hit,
        analytical_endpoint_error_km=None, branch=f"t{time_s}s{slot}",
        metadata={"event_refined": True},
    )


def test_a_slot_variant_does_not_spend_a_run_the_other_trajectory_needs() -> None:
    # Same impulse at the same epoch in two slots is two scripts and one
    # trajectory. Measured under GMAT, two such variants replayed 1.8e-9 km
    # apart -- 1.8 micrometres, which decides nothing.
    a0 = _candidate(100.0, (0.5, 0.0, 0.0), 0)
    a1 = _candidate(100.0, (0.5, 0.0, 0.0), 1)
    b0 = _candidate(200.0, (0.6, 0.0, 0.0), 0)
    chosen = _distinct_trajectories_first([a0, a1, b0], 2)
    assert [c.candidate_id for c in chosen] == [a0.candidate_id, b0.candidate_id]


def test_the_variants_still_get_whatever_budget_is_left() -> None:
    a0 = _candidate(100.0, (0.5, 0.0, 0.0), 0)
    a1 = _candidate(100.0, (0.5, 0.0, 0.0), 1)
    b0 = _candidate(200.0, (0.6, 0.0, 0.0), 0)
    chosen = _distinct_trajectories_first([a0, a1, b0], 3)
    assert {c.candidate_id for c in chosen} == {a0.candidate_id, a1.candidate_id, b0.candidate_id}
    assert [c.candidate_id for c in chosen] == [a0.candidate_id, b0.candidate_id, a1.candidate_id]


def test_it_does_not_reach_past_the_hits_into_the_misses() -> None:
    """The limit that makes this an improvement rather than a regression.

    A first version reordered across the whole list and spent a run on a
    candidate the analytical layer had already measured as missing, in place of
    a second script for a trajectory that hits. A replay cannot turn the miss
    into a submission; the variant it displaced is a script that could be one.
    """
    a0 = _candidate(100.0, (0.5, 0.0, 0.0), 0)
    a1 = _candidate(100.0, (0.5, 0.0, 0.0), 1)
    miss = _candidate(300.0, (0.0, 0.0, 0.0), 0, hit=False)
    chosen = _distinct_trajectories_first([a0, a1, miss], 2)
    assert [c.candidate_id for c in chosen] == [a0.candidate_id, a1.candidate_id]
    assert all(c.successful for c in chosen)


def test_rank_order_survives_both_passes() -> None:
    ordered = [
        _candidate(100.0, (0.5, 0.0, 0.0), 0),
        _candidate(100.0, (0.5, 0.0, 0.0), 1),
        _candidate(200.0, (0.6, 0.0, 0.0), 0),
        _candidate(200.0, (0.6, 0.0, 0.0), 1),
        _candidate(300.0, (0.7, 0.0, 0.0), 0),
    ]
    chosen = _distinct_trajectories_first(ordered, 5)
    assert [c.candidate_id for c in chosen] == [
        ordered[0].candidate_id, ordered[2].candidate_id, ordered[4].candidate_id,
        ordered[1].candidate_id, ordered[3].candidate_id,
    ]
    # Nothing is dropped when the budget covers everything.
    assert len(chosen) == len(ordered)


def test_an_empty_budget_selects_nothing() -> None:
    assert _distinct_trajectories_first([_candidate(100.0, (0.5, 0.0, 0.0), 0)], 0) == []
