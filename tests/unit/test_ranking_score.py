from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, ScoreConfig
from orbitwin.constraints import apply_constraints
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Candidate, Scenario
from orbitwin.rules import PRELIMINARY, RANKING
from orbitwin.score import apply_score, score_candidate

SCORE = ScoreConfig(kt=0.003, ct=3000.0, kv=1.0, cv=1.5, delta_v_unit="km/s")
#: The same terms in the units the 2026/11/04 revision fixes, with a C_r.
RANKING_SCORE = ScoreConfig(kt=0.003, ct=3000.0, kv=0.001, cv=1500.0, delta_v_unit="m/s",
                            cr_km=100.0)
SCORES = {"preliminary": SCORE, "ranking": RANKING_SCORE}


def _config(rules: str) -> OrbitWinConfig:
    base = OrbitWinConfig()
    policy = dataclasses.replace(
        base.policy, rules=rules, horizon_s=7200.0, dv_limit_km_s=0.25, dv_safe_limit_km_s=0.2495
    )
    return dataclasses.replace(base, policy=policy, score=SCORES[rules])


@pytest.fixture(scope="module", params=["preliminary", "ranking"])
def parsed(request: pytest.FixtureRequest, workshop_script: Path) -> tuple[Scenario, OrbitWinConfig]:
    config = _config(request.param)
    return parse_gmat_script(workshop_script, config)[0], config


def _miss(minimum_km: float | None, at_completion_km: float) -> Candidate:
    return Candidate(
        burns=(),
        intercept_time_s=None,
        minimum_distance_km=minimum_km,
        distance_at_completion_km=at_completion_km,
        successful=False,
        analytical_endpoint_error_km=None,
        branch="test",
        metadata={"event_refined": True},
    )


def test_the_scenario_records_its_rulebook(parsed: tuple[Scenario, OrbitWinConfig]) -> None:
    scenario, config = parsed
    assert scenario.rules == config.policy.rules


def test_a_near_miss_is_scored_on_its_closest_approach_under_the_ranking_rules(
    parsed: tuple[Scenario, OrbitWinConfig],
) -> None:
    """Eq. (2)-(3): Δr_min = max(min Δr over [0, T_max], 5).

    6 km at closest, 900 km apart at T_max. The preliminary's §4 scores the
    900 km; the ranking rules score the 6 km, on the term as revised 2026/11/04.
    """
    scenario, config = parsed
    distance = score_candidate(_miss(6.0, 900.0), scenario, config.score).distance_score
    expected = (100.0 / (1.0 + math.exp((1.0 / 100.0) ** 2)) if scenario.rules == "ranking"
                else 50.0 * math.exp(-895.0 / 100.0))
    assert distance == pytest.approx(expected, rel=1e-12)


def test_a_hit_scores_the_full_distance_term_either_way(
    parsed: tuple[Scenario, OrbitWinConfig],
) -> None:
    scenario, config = parsed
    hit = dataclasses.replace(
        _miss(0.004, 5.0), intercept_time_s=5000.0, successful=True
    )
    assert score_candidate(hit, scenario, config.score).distance_score == 50.0


def test_an_unmeasured_miss_earns_no_distance_points(
    parsed: tuple[Scenario, OrbitWinConfig],
) -> None:
    scenario, config = parsed
    unknown = _miss(None, None)  # type: ignore[arg-type]
    assert score_candidate(unknown, scenario, config.score).distance_score == 0.0


def test_over_the_per_burn_limit_disqualifies_under_the_ranking_rules(
    parsed: tuple[Scenario, OrbitWinConfig],
) -> None:
    """§4 note 1(a) 不列入排名; the preliminary deducted ten points instead."""
    scenario, config = parsed
    over = dataclasses.replace(
        _miss(6.0, 900.0), burns=(Burn(100.0, (0.26, 0.0, 0.0), (0.26, 0.0, 0.0)),)
    )
    result = apply_constraints(over, scenario, config)
    assert not result.valid
    charged = [penalty for penalty in over.penalties if "formal limit" in penalty]
    assert bool(charged) is (scenario.rules == "preliminary")


def test_moving_the_closest_approach_makes_the_score_stale(
    parsed: tuple[Scenario, OrbitWinConfig],
) -> None:
    scenario, config = parsed
    candidate = _miss(6.0, 900.0)
    apply_score(candidate, scenario, config.score)
    assert not candidate.score_is_stale
    candidate.minimum_distance_km = 50.0
    assert candidate.score_is_stale


def test_the_ceilings_are_the_terms_at_the_ends_of_their_ranges() -> None:
    """kt 0.01 /s, Ct 500 s, kv 2 per km/s, Cv 1.5 km/s, T_max 1200 s, a 3 km/s cap."""
    from orbitwin.config import ScoreConfig
    from orbitwin.score import ceilings

    bound = ceilings(ScoreConfig(kt=0.01, ct=500.0, kv=2.0, cv=1.5, delta_v_unit="km/s"),
                     1200.0, 3.0)
    assert bound.any_answer == pytest.approx(98.64703189745372, abs=1e-12)
    assert bound.any_miss == pytest.approx(73.83712945042085, abs=1e-12)
    # Rewarding a late, expensive answer turns the best end over.
    upside = ceilings(ScoreConfig(kt=-0.01, ct=500.0, kv=-2.0, cv=1.5, delta_v_unit="km/s"),
                      1200.0, 3.0)
    assert upside.any_answer == pytest.approx(50.0 + 25.0 / (1 + math.exp(-7.0))
                                              + 25.0 / (1 + math.exp(-3.0)), abs=1e-12)


def test_the_ceilings_read_delta_v_in_its_announced_unit() -> None:
    from orbitwin.config import ScoreConfig
    from orbitwin.score import ceilings

    # kv < 0 puts the best term at the cap, where the unit matters.
    km = ceilings(ScoreConfig(kt=0.003, ct=4633.105, kv=-1.2, cv=2.0, delta_v_unit="km/s"),
                  5791.381, 3.0)
    m = ceilings(ScoreConfig(kt=0.003, ct=4633.105, kv=-0.0012, cv=2000.0, delta_v_unit="m/s"),
                 5791.381, 3.0)
    assert km.best_dv_term == pytest.approx(25.0 / (1.0 + math.exp(-1.2)), abs=1e-12)
    assert (m.any_answer, m.any_miss) == pytest.approx((km.any_answer, km.any_miss), abs=1e-12)


def _r2_bound(cr_km: float | None = 3000.0):
    from orbitwin.config import ScoreConfig
    from orbitwin.score import ceilings

    # R2 under kt 0.003, Ct 0.8 T_max, kv 1.2 per km/s, Cv 2 km/s, C_r 3000 km.
    return ceilings(ScoreConfig(kt=0.003, ct=4633.105, kv=0.0012, cv=2000.0, delta_v_unit="m/s",
                                cr_km=cr_km), 5791.381, 3.0)


def test_a_far_miss_winner_is_beaten_only_closer_than_it_passed() -> None:
    """R2's zero burn, 4711.155 km off: 7.87 points at C_r 3000 km, and under the
    preliminary's term 1.8e-19, which vanishes once added to the rest."""
    from orbitwin.score import ScoreBreakdown

    for rules, distance in (
        (RANKING, 100.0 / (1.0 + math.exp(((4711.155044962988 - 5.0) / 3000.0) ** 2))),
        (PRELIMINARY, 50.0 * math.exp(-(4711.155044962988 - 5.0) / 100.0)),
    ):
        bound = _r2_bound(3000.0 if rules is RANKING else None)
        winner = ScoreBreakdown(distance, bound.time_term_at_horizon, bound.best_dv_term, 0.0,
                                distance + bound.time_term_at_horizon + bound.best_dv_term)
        assert distance > 0.0
        assert bound.miss_must_be_within_km(winner, rules) == pytest.approx(
            4711.155044962988, rel=1e-9 if rules is RANKING else 1e-12
        )


def test_no_miss_beats_a_good_hit_and_any_beats_a_floored_score() -> None:
    from orbitwin.score import ScoreBreakdown

    bound = _r2_bound()
    hit = ScoreBreakdown(50.0, 24.0, 22.0, 0.0, 96.0)
    assert bound.miss_must_be_within_km(hit, RANKING) is None
    floored = ScoreBreakdown(50.0, 0.0, 0.0, 60.0, 0.0)
    assert bound.miss_must_be_within_km(floored, RANKING) == math.inf


def test_a_penalised_best_leaves_more_room_to_a_miss() -> None:
    from orbitwin.score import ScoreBreakdown

    bound = _r2_bound()
    clean = ScoreBreakdown(30.0, 10.0, 20.0, 0.0, 60.0)
    penalised = ScoreBreakdown(30.0, 10.0, 20.0, 10.0, 50.0)
    near, far = (bound.miss_must_be_within_km(row, RANKING) for row in (clean, penalised))
    assert near is not None and far is not None and far > near
    rest = 30.0 + 10.0 - bound.time_term_at_horizon + 20.0 - bound.best_dv_term - 10.0
    assert far == pytest.approx(5.0 + 3000.0 * math.sqrt(math.log(100.0 / rest - 1.0)))
