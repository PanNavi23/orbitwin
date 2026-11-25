from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, ScoreConfig, SearchConfig
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Candidate
from orbitwin.score import apply_score
from orbitwin.search import (
    drop_post_intercept_burns,
    record_event,
    refine_events,
    search_one_burn,
)

# 初賽規則 §5 eq. 1 sums the impulse magnitudes over k = 1..N, where N is
# 總機動次數. Nothing in the clause excludes a maneuver that fires after the
# interception, so read as written such a burn is charged in full for nothing.


def _config() -> OrbitWinConfig:
    return dataclasses.replace(
        OrbitWinConfig(),
        search=SearchConfig(
            one_burn_samples=24,
            one_burn_grid_burns=4,
            one_burn_grid_arrivals=8,
            two_burn_enabled=False,
            event_refine_top=3,
            keep_candidates=20,
            workers=1,
        ),
        score=ScoreConfig(kt=0.001, ct=5000.0, kv=0.004, cv=800.0, delta_v_unit="m/s"),
    )


@pytest.fixture(scope="module")
def trimmed_pair(different_plane_script: Path):
    """A wasteful candidate and its trimmed variant, both measured once.

    Every trajectory evaluation here runs to the mission horizon, so the search
    and the two refinements are hoisted out of the tests rather than repeated
    by each of them.

    The wasteful shape is built rather than searched for. Neither bundled
    scenario produces a burn scheduled past its own first entry, so waiting for
    one would be waiting forever; appending an impulse 400 s after a confirmed
    interception produces exactly the shape §5 charges for and §4 says cannot
    matter.
    """
    config = _config()
    scenario, _ = parse_gmat_script(different_plane_script, config)
    found = [
        candidate
        for candidate in search_one_burn(scenario, config)
        if candidate.successful and candidate.metadata.get("event_refined") is True
    ]
    assert found, "the fixture needs at least one confirmed interception"
    base = found[0]
    assert base.intercept_time_s is not None

    late = Burn(
        time_s=base.intercept_time_s + 400.0,
        dv_inertial_km_s=(0.05, -0.03, 0.02),
        dv_script_km_s=(0.05, -0.03, 0.02),
        slot_index=1,
    )
    wasteful = Candidate(
        burns=(*base.burns, late),
        intercept_time_s=None,
        minimum_distance_km=None,
        distance_at_completion_km=None,
        successful=False,
        analytical_endpoint_error_km=base.analytical_endpoint_error_km,
        branch=base.branch,
        metadata={**base.metadata, "gmat_event": {"stop_time_s": 1.0e6}},
    )
    (event,) = refine_events(scenario, config, [wasteful], workers=1)
    record_event(wasteful, event, config)
    apply_score(wasteful, scenario, config.score)

    (trimmed,) = drop_post_intercept_burns(scenario, config, [wasteful], workers=1)
    return scenario, config, base, wasteful, trimmed


def test_a_burn_after_the_interception_is_dropped(trimmed_pair) -> None:
    _scenario, config, base, wasteful, trimmed = trimmed_pair
    assert wasteful.intercept_time_s == pytest.approx(base.intercept_time_s, abs=1e-6), (
        "an impulse after the first entry cannot move the first entry"
    )
    assert len(trimmed.burns) == len(wasteful.burns) - 1
    assert "gmat_event" not in trimmed.metadata, "measured on another script"
    assert trimmed.total_dv_km_s < wasteful.total_dv_km_s
    assert trimmed.intercept_time_s == pytest.approx(wasteful.intercept_time_s, abs=1e-6)
    # §4: on any success Δrmin is exactly the interception radius.
    assert trimmed.distance_at_completion_km == pytest.approx(config.policy.intercept_radius_km)


def test_dropping_it_cannot_lower_the_score(trimmed_pair) -> None:
    _scenario, _config, _base, wasteful, trimmed = trimmed_pair
    assert wasteful.score is not None and trimmed.score is not None
    assert trimmed.score >= wasteful.score
    # The whole difference is the delta-v term: same tint, same Δrmin of 5.
    assert trimmed.score_components["distance_score"] == pytest.approx(
        wasteful.score_components["distance_score"]
    )
    assert trimmed.score_components["time_score"] == pytest.approx(
        wasteful.score_components["time_score"]
    )
    assert trimmed.score_components["delta_v_score"] > wasteful.score_components["delta_v_score"]


def test_the_variant_is_measured_on_its_own_trajectory(trimmed_pair) -> None:
    # §6 breaks ties on the closest approach over the trajectory, and after the
    # dropped burn that is a different trajectory. The variant must not inherit
    # the original's measurements.
    _scenario, _config, _base, wasteful, trimmed = trimmed_pair
    assert trimmed.candidate_id != wasteful.candidate_id
    assert trimmed.metadata["trimmed_from"] == wasteful.candidate_id
    assert trimmed.metadata["event_refined"] is True
    assert trimmed.minimum_distance_km is not None
    assert trimmed.branch.endswith("/post-intercept-trimmed")


def test_a_candidate_that_burns_before_it_arrives_yields_nothing(trimmed_pair) -> None:
    scenario, config, base, _wasteful, _trimmed = trimmed_pair
    assert base.intercept_time_s is not None
    assert all(burn.time_s <= base.intercept_time_s for burn in base.burns)
    assert drop_post_intercept_burns(scenario, config, [base], workers=1) == []


def test_a_failed_candidate_yields_nothing(trimmed_pair) -> None:
    # With no interception there is no "after the interception" to trim, and
    # §4 measures the failure at 4TA where every burn has already fired.
    scenario, config, base, _wasteful, _trimmed = trimmed_pair
    missed = Candidate(
        burns=base.burns,
        intercept_time_s=None,
        minimum_distance_km=120.0,
        distance_at_completion_km=900.0,
        successful=False,
        analytical_endpoint_error_km=None,
        branch=base.branch,
        metadata=dict(base.metadata),
    )
    assert drop_post_intercept_burns(scenario, config, [missed], workers=1) == []
