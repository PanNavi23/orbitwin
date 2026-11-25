from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, ScoreConfig, SearchConfig
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Scenario
from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, orbital_period_s
from orbitwin.two_burn import search_two_burn_global

MU_KM3_S2 = 398600.4418
TARGET_SMA_KM = 7000.0
PURSUER_SMA_KM = 7003.0


def _pair_that_closes_on_its_own(scenario: Scenario) -> Scenario:
    """A geometry where the interception happens before anyone maneuvers.

    初賽規則 §4 defines tint as the *first* entry into the 5 km sphere, so it
    can fall before a maneuver that was scheduled to cause a later one. That is
    what `drop_post_intercept_burns` guards, and random geometries do not
    produce it: 40 random different-plane pairs gave 324 one-burn and 248
    two-burn refined interceptions without a single burn past its own first
    entry, because two random circular orbits pass hundreds of km apart.

    So it is built. Co-planar, 3 km apart in radius, with the initial phase gap
    sized so the faster inner orbit closes it at mid-window. The pair then
    passes inside 5 km on its own.
    """
    period_s = orbital_period_s(TARGET_SMA_KM, MU_KM3_S2)
    horizon_s = 4.0 * period_s
    drift_rad_s = 2.0 * math.pi / period_s - 2.0 * math.pi / orbital_period_s(
        PURSUER_SMA_KM, MU_KM3_S2
    )
    gap_deg = math.degrees(0.5 * drift_rad_s * horizon_s)
    return dataclasses.replace(
        scenario,
        # The target holds the smaller, faster orbit, so it starts behind.
        target=dataclasses.replace(
            scenario.target,
            initial_state=keplerian_to_cartesian(
                KeplerianElements(TARGET_SMA_KM, 0.0, 28.5, 40.0, 0.0, -gap_deg), MU_KM3_S2
            ),
        ),
        pursuer=dataclasses.replace(
            scenario.pursuer,
            initial_state=keplerian_to_cartesian(
                KeplerianElements(PURSUER_SMA_KM, 0.0, 28.5, 40.0, 0.0, 0.0), MU_KM3_S2
            ),
        ),
        max_time_s=horizon_s,
    )


@pytest.fixture(scope="module")
def closing_pair_search(different_plane_script: Path):
    config = OrbitWinConfig(
        search=SearchConfig(
            one_burn_samples=64,
            one_burn_grid_burns=8,
            one_burn_grid_arrivals=12,
            two_burn_enabled=True,
            two_burn_samples=128,
            # A candidate that burns after it has already arrived wastes that
            # impulse, so it ranks badly and only reaches refinement when the
            # refined set is generous. That is also why it is not a threat to
            # the submission -- but §5 still charges for it.
            event_refine_top=30,
            keep_candidates=80,
            workers=1,
        ),
        score=ScoreConfig(kt=0.001, ct=5000.0, kv=0.004, cv=800.0, delta_v_unit="m/s"),
    )
    scenario, _ = parse_gmat_script(different_plane_script, config)
    scenario = _pair_that_closes_on_its_own(scenario)
    return scenario, config, search_two_burn_global(scenario, config)


def _late_burns(candidate) -> list:
    assert candidate.intercept_time_s is not None
    return [burn for burn in candidate.burns if burn.time_s > candidate.intercept_time_s + 1e-9]


@pytest.mark.slow
def test_the_search_itself_reaches_a_burn_after_the_interception(closing_pair_search) -> None:
    # The unit tests build the shape by hand. This one shows the search reaches
    # it, so the trim guards a real path rather than a hypothetical one.
    _scenario, _config, candidates = closing_pair_search
    refined = [
        candidate
        for candidate in candidates
        if candidate.metadata.get("event_refined") and candidate.intercept_time_s is not None
    ]
    assert refined
    assert [candidate for candidate in refined if _late_burns(candidate)]


@pytest.mark.slow
def test_trimming_reaches_a_candidate_the_archive_does_not_already_hold(
    closing_pair_search,
) -> None:
    # When every burn falls after tint the variant is the zero-burn baseline,
    # which the pipeline already carries. The case that adds something is the
    # partial one -- t1 before the interception, t2 after -- where the variant
    # is neither the parent nor the baseline.
    _scenario, _config, candidates = closing_pair_search
    partial = [
        candidate
        for candidate in candidates
        if candidate.intercept_time_s is not None
        and 0 < len(_late_burns(candidate)) < len(candidate.burns)
    ]
    assert partial, "the constructed geometry is supposed to produce a partial trim"
    variants = [c for c in candidates if c.branch.endswith("/post-intercept-trimmed")]
    assert any(variant.burns for variant in variants), (
        "a partial trim must leave a variant that still carries an impulse"
    )


@pytest.mark.slow
def test_identical_variants_are_emitted_once(closing_pair_search) -> None:
    # Candidates whose burns all fall after tint trim to the same empty script.
    # They tie exactly on both Pareto axes, so nothing dominates anything and
    # every copy would survive pruning.
    _scenario, _config, candidates = closing_pair_search
    variants = [c for c in candidates if c.branch.endswith("/post-intercept-trimmed")]
    assert variants
    signatures = {
        tuple((b.time_s, b.dv_inertial_km_s, b.dv_script_km_s, b.slot_index) for b in c.burns)
        for c in variants
    }
    assert len(signatures) == len(variants)
