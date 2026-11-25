"""score.levers: ∂S/∂T, ∂S/∂ΔV and ∂S/∂d at a row, and each term's headroom.

Every rate is checked against a central difference of score_candidate itself,
so the units and the clock are the ones the score uses, not the ones assumed here.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, ScoreConfig
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Candidate, Scenario
from orbitwin.rules import interception_rules
from orbitwin.score import (
    Levers,
    ScoreBreakdown,
    ceilings,
    distance_term_slope,
    levers,
    on_solver_clock,
    score_candidate,
)

CAP_KM_S = 3.0
#: The ranking profile's safe per-impulse and total limits.
IMPULSE_KM_S = 0.249
BUDGET_KM_S = 2.99
#: The ranking rules since 2026/11/04: ΔV in m/s, and a C_r.
RANK = ScoreConfig(kt=0.003, ct=3000.0, kv=0.0012, cv=1500.0, delta_v_unit="m/s", cr_km=100.0)
#: The same score under the preliminary rules, in km/s and in m/s.
KM = ScoreConfig(kt=0.003, ct=3000.0, kv=1.2, cv=1.5, delta_v_unit="km/s")
M = dataclasses.replace(RANK, cr_km=None)


def _parsed(workshop_script: Path, rules: str) -> Scenario:
    base = OrbitWinConfig()
    policy = dataclasses.replace(base.policy, rules=rules, horizon_s=7200.0,
                                 dv_limit_km_s=0.25, dv_safe_limit_km_s=0.2495)
    parsed = parse_gmat_script(workshop_script, dataclasses.replace(base, policy=policy))[0]
    assert parsed.max_time_s == 7200.0 and parsed.time_origin_offset_s == 0.0
    return parsed


@pytest.fixture(scope="module")
def scenario(workshop_script: Path) -> Scenario:
    return _parsed(workshop_script, "ranking")


@pytest.fixture(scope="module")
def preliminary(workshop_script: Path) -> Scenario:
    return _parsed(workshop_script, "preliminary")


@pytest.fixture(scope="module")
def scenarios(scenario: Scenario, preliminary: Scenario) -> dict[str, Scenario]:
    return {"ranking": scenario, "preliminary": preliminary}


def _row(t_s: float | None, dv_km_s: float, closest_km: float = 4.0) -> Candidate:
    return Candidate(
        burns=(Burn(100.0, (dv_km_s, 0.0, 0.0), (dv_km_s, 0.0, 0.0)),),
        intercept_time_s=t_s,
        minimum_distance_km=closest_km,
        distance_at_completion_km=closest_km,
        successful=t_s is not None,
        analytical_endpoint_error_km=None,
        branch="test",
        metadata={"event_refined": True},
    )


def _levers(row: Candidate, scenario: Scenario, score: ScoreConfig, penalties: int = 0) -> Levers:
    breakdown = score_candidate(row, scenario, score, penalty_count=penalties)
    return _of(row, breakdown, scenario, score)


def _of(row: Candidate, breakdown: ScoreBreakdown, scenario: Scenario,
        score: ScoreConfig) -> Levers:
    return levers(row, breakdown, scenario, score, CAP_KM_S, impulse_km_s=IMPULSE_KM_S,
                  budget_km_s=BUDGET_KM_S)


def _gain(row: Candidate, scenario: Scenario, score: ScoreConfig, field: str, h: float,
          penalties: int = 0) -> float:
    """(S(x - h) - S(x + h)) / 2h of score_candidate, x being `field` of ``row``."""

    def at(step: float) -> float:
        if field == "dv":
            dv = row.total_dv_km_s + step
            moved = dataclasses.replace(row, burns=(Burn(100.0, (dv, 0.0, 0.0), (dv, 0.0, 0.0)),))
        elif field == "distance":
            d = row.minimum_distance_km + step  # type: ignore[operator]
            moved = dataclasses.replace(row, minimum_distance_km=d, distance_at_completion_km=d)
        else:
            moved = dataclasses.replace(row, **{field: getattr(row, field) + step})
        return score_candidate(moved, scenario, score, penalty_count=penalties).raw_score

    return (at(-h) - at(h)) / (2.0 * h)


def _f(score: ScoreConfig, t_team_s: float) -> float:
    """The time term on the announced clock, written out."""
    return 25.0 / (1.0 + math.exp(score.kt * (t_team_s - score.ct)))


def _g(score: ScoreConfig, dv_km_s: float) -> float:
    scale = 1000.0 if score.delta_v_unit == "m/s" else 1.0
    return 25.0 / (1.0 + math.exp(score.kv * (dv_km_s * scale - score.cv)))


def _d(rules: str, distance_km: float, cr_km: float | None) -> float:
    excess = max(distance_km, 5.0) - 5.0
    if rules == "preliminary":
        return 50.0 * math.exp(-excess / 100.0)
    assert cr_km is not None
    return 100.0 / (1.0 + math.exp((excess / cr_km) ** 2))


CASES = [
    pytest.param("ranking", RANK, 0.0, id="ranking"),
    pytest.param("ranking", RANK, 1800.0, id="ranking-offset"),
    pytest.param("ranking", dataclasses.replace(RANK, kt=-0.003, kv=-0.0012), 1800.0,
                 id="ranking-rewarded-offset"),
    pytest.param("preliminary", KM, 0.0, id="km/s"),
    pytest.param("preliminary", KM, 1800.0, id="km/s-offset"),
    pytest.param("preliminary", M, 0.0, id="m/s"),
    pytest.param("preliminary", dataclasses.replace(KM, kt=-0.003, kv=-1.2), 0.0, id="rewarded"),
]


@pytest.mark.parametrize("rules, score, offset_s", CASES)
def test_a_hits_rates_are_score_candidates_own_slopes(
    scenarios: dict[str, Scenario], rules: str, score: ScoreConfig, offset_s: float
) -> None:
    shifted = dataclasses.replace(scenarios[rules], time_origin_offset_s=offset_s)
    row = _row(2500.0, 1.0)
    found = _levers(row, shifted, score)
    assert found.hit and not found.floored and found.points_per_km_closer is None
    assert found.points_per_s_earlier == pytest.approx(
        _gain(row, shifted, score, "intercept_time_s", 0.01), rel=1e-6)
    assert found.points_per_km_s_saved == pytest.approx(
        _gain(row, shifted, score, "dv", 1e-5), rel=1e-6)
    assert found.points_per_s_earlier is not None
    if score.kt <= 0.0:
        # Arriving earlier costs here: no time to buy, so no break-even.
        assert found.break_even_s_per_km_s is None
        assert "No break-even" in str(found.to_dict()["notes"])
    else:
        assert found.break_even_s_per_km_s == pytest.approx(
            found.points_per_km_s_saved / found.points_per_s_earlier, rel=1e-12)


@pytest.mark.parametrize("rules, score, offset_s", CASES)
def test_a_km_s_spent_for_the_break_even_seconds_leaves_the_score_where_it_was(
    scenarios: dict[str, Scenario], rules: str, score: ScoreConfig, offset_s: float
) -> None:
    shifted = dataclasses.replace(scenarios[rules], time_origin_offset_s=offset_s)
    row = _row(2500.0, 1.0)
    found = _levers(row, shifted, score)
    if score.kt <= 0.0:
        return
    assert found.break_even_s_per_km_s is not None
    step = 1e-5
    traded = _row(2500.0 - step * found.break_even_s_per_km_s, 1.0 + step)
    before = score_candidate(row, shifted, score).raw_score
    after = score_candidate(traded, shifted, score).raw_score
    assert abs(after - before) < 1e-3 * step * abs(found.points_per_km_s_saved)


@pytest.mark.parametrize("rules, score, offset_s", CASES)
def test_headroom_is_against_the_best_end_on_the_solver_clock(
    scenarios: dict[str, Scenario], rules: str, score: ScoreConfig, offset_s: float
) -> None:
    shifted = dataclasses.replace(scenarios[rules], time_origin_offset_s=offset_s)
    row = _row(2500.0, 1.0)
    found = _levers(row, shifted, score)
    # T_team runs offset_s ahead of the solver's t: the reachable ends are offset_s and
    # T_max + offset_s on the announced clock.
    best_time = max(_f(score, offset_s), _f(score, 7200.0 + offset_s))
    best_dv = max(_g(score, 0.0), _g(score, CAP_KM_S))
    assert found.time_headroom == pytest.approx(best_time - _f(score, 2500.0 + offset_s),
                                                abs=1e-12)
    assert found.delta_v_headroom == pytest.approx(best_dv - _g(score, 1.0), abs=1e-12)
    assert found.distance_headroom == 0.0 and found.penalty_headroom == 0.0
    assert found.total_headroom == pytest.approx(found.time_headroom + found.delta_v_headroom,
                                                 abs=1e-12)


def test_the_announced_unit_does_not_change_the_answer(preliminary: Scenario) -> None:
    for row in (_row(2500.0, 1.0), _row(None, 0.4, closest_km=30.0)):
        km, m = (_levers(row, preliminary, score) for score in (KM, M))
        assert m.points_per_km_s_saved == pytest.approx(km.points_per_km_s_saved, rel=1e-9)
        assert m.delta_v_headroom == pytest.approx(km.delta_v_headroom, abs=1e-12)


MISSES = [
    pytest.param("preliminary", KM, 30.0, id="exponential-30km"),
    pytest.param("preliminary", KM, 6.0, id="exponential-6km"),
    *(pytest.param("ranking", dataclasses.replace(RANK, cr_km=cr), d, id=f"cr{cr:g}-{d:g}km")
      for cr in (10.0, 100.0, 300.0) for d in (6.0, 30.0, 150.0)),
]


@pytest.mark.parametrize("rules, score, closest_km", MISSES)
def test_a_miss_has_no_time_slope_and_is_worth_its_closest_approach(
    scenarios: dict[str, Scenario], rules: str, score: ScoreConfig, closest_km: float
) -> None:
    scenario = scenarios[rules]
    row = _row(None, 0.4, closest_km=closest_km)
    found = _levers(row, scenario, score)
    assert not found.hit and not found.floored
    assert found.points_per_s_earlier is None and found.break_even_s_per_km_s is None
    # Only becoming a hit moves T_team off T_max.
    assert found.time_headroom == pytest.approx(_f(score, 0.0) - _f(score, 7200.0), abs=1e-12)
    assert found.distance_headroom == pytest.approx(
        50.0 - _d(rules, closest_km, score.cr_km), abs=1e-12)
    assert found.points_per_km_closer == pytest.approx(
        _gain(row, scenario, score, "distance", 1e-3), rel=1e-6, abs=1e-12)
    assert found.points_per_km_s_saved == pytest.approx(
        _gain(row, scenario, score, "dv", 1e-5), rel=1e-6)
    assert "becoming a hit" in str(found.to_dict()["notes"])


@pytest.mark.parametrize("closest_km", [4.0, 5.0])
def test_the_ranking_term_is_flat_at_the_sphere(scenario: Scenario, closest_km: float) -> None:
    """Revised 2026/11/04: no slope at 5 km, where the old term fell 0.5 pt/km."""
    row = _row(None, 0.4, closest_km=closest_km)
    found = _levers(row, scenario, RANK)
    assert found.distance_headroom == 0.0 and found.points_per_km_closer == 0.0
    assert math.copysign(1.0, found.points_per_km_closer) == 1.0
    assert _gain(row, scenario, RANK, "distance", 1e-3) == pytest.approx(0.0, abs=1e-5)


def test_a_preliminary_miss_inside_the_sphere_gains_nothing_by_closing(
    preliminary: Scenario,
) -> None:
    row = _row(None, 0.4, closest_km=4.0)
    found = _levers(row, preliminary, KM)
    assert found.distance_headroom == 0.0
    assert found.points_per_km_closer == 0.0 == _gain(row, preliminary, KM, "distance", 1e-3)


def test_a_miss_rewarded_for_lateness_gains_no_time_by_hitting(scenario: Scenario) -> None:
    late = dataclasses.replace(RANK, kt=-0.003)
    assert _levers(_row(None, 0.4, closest_km=30.0), scenario, late).time_headroom == 0.0


def test_deductions_are_headroom_and_leave_the_rates_alone(scenario: Scenario) -> None:
    row = _row(2500.0, 1.0)
    clean = _levers(row, scenario, RANK)
    charged = _levers(row, scenario, RANK, penalties=2)
    assert not charged.floored and charged.penalty_headroom == 20.0
    assert charged.total_headroom == pytest.approx(clean.total_headroom + 20.0, abs=1e-12)
    for name in ("points_per_s_earlier", "points_per_km_s_saved", "break_even_s_per_km_s"):
        assert getattr(charged, name) == getattr(clean, name)
    assert charged.points_per_s_earlier == pytest.approx(
        _gain(row, scenario, RANK, "intercept_time_s", 0.01, penalties=2), rel=1e-6)


@pytest.mark.parametrize("t_s", [2500.0, None])
def test_a_floored_score_has_no_slope(scenario: Scenario, t_s: float | None) -> None:
    row = _row(t_s, 1.0, closest_km=4.0 if t_s else 30.0)
    breakdown = score_candidate(row, scenario, RANK, penalty_count=10)
    assert breakdown.raw_score == 0.0
    found = _of(row, breakdown, scenario, RANK)
    assert found.floored and found.break_even_s_per_km_s is None
    assert found.points_per_km_s_saved == 0.0 == _gain(row, scenario, RANK, "dv", 1e-5, 10)
    if t_s is None:
        assert found.points_per_s_earlier is None and found.points_per_km_closer == 0.0
    else:
        assert found.points_per_s_earlier == 0.0
        assert _gain(row, scenario, RANK, "intercept_time_s", 0.01, 10) == 0.0
    unfloored = (breakdown.distance_score + breakdown.time_score + breakdown.delta_v_score
                 - breakdown.penalty_score)
    bound = ceilings(RANK, 7200.0, CAP_KM_S)
    assert found.penalty_headroom == 100.0
    assert found.total_headroom == pytest.approx(bound.any_answer - unfloored, abs=1e-12)
    assert "floored" in str(found.to_dict()["notes"])


def test_a_saturated_term_is_flat_and_flat_time_has_no_break_even(scenario: Scenario) -> None:
    # kt (T - Ct) = -1000 and kv (ΔV - Cv) = -5000: both terms are exactly 25.
    steep = dataclasses.replace(RANK, kt=1.0, kv=10.0)
    row = _row(2000.0, 1.0)
    found = _levers(row, scenario, steep)
    assert found.points_per_s_earlier == 0.0 and found.points_per_km_s_saved == 0.0
    assert math.copysign(1.0, found.points_per_s_earlier) == 1.0
    assert found.break_even_s_per_km_s is None
    assert found.time_headroom == 0.0 and found.delta_v_headroom == 0.0
    # Free delta-v pays for itself whatever it buys.
    free = _levers(row, scenario, dataclasses.replace(RANK, kv=10.0))
    assert free.points_per_km_s_saved == 0.0 and free.break_even_s_per_km_s == 0.0
    # Rewarded and saturated: still 0, not -0.0, in the JSON.
    rewarded = _levers(row, scenario, dataclasses.replace(RANK, kt=-1.0, kv=-10.0))
    assert rewarded.points_per_s_earlier is not None
    assert math.copysign(1.0, rewarded.points_per_s_earlier) == 1.0
    assert math.copysign(1.0, rewarded.points_per_km_s_saved) == 1.0


def test_a_nearly_saturated_slope_keeps_its_precision(scenario: Scenario) -> None:
    """At kt (T - Ct) = -40, 1 - F/25 cancels to 0 in floating point; the slope must not."""
    row = _row(3000.0 - 40.0 / RANK.kt, 1.0)
    found = _levers(row, scenario, RANK)
    assert found.points_per_s_earlier == pytest.approx(RANK.kt * 25.0 * math.exp(-40.0),
                                                       rel=1e-9)
    assert found.break_even_s_per_km_s is not None and math.isfinite(found.break_even_s_per_km_s)


@pytest.mark.parametrize("t_s, closest_km, penalties", [
    (2500.0, 4.0, 0), (None, 30.0, 0), (None, math.inf, 0), (None, 1e6, 0), (2500.0, 4.0, 10),
    (1e-9, 4.0, 0),
])
def test_the_dict_is_strict_json(scenario: Scenario, t_s: float | None, closest_km: float,
                                 penalties: int) -> None:
    for score in (RANK, dataclasses.replace(RANK, kt=50.0, kv=1e3, cr_km=1e-3)):
        found = _levers(_row(t_s, 1.0, closest_km=closest_km), scenario, score, penalties)
        json.dumps(found.to_dict(), allow_nan=False)


def test_the_ceilings_name_the_best_time_term_they_add(scenario: Scenario) -> None:
    for score in (RANK, dataclasses.replace(RANK, kt=-0.003)):
        bound = ceilings(on_solver_clock(score, scenario), 7200.0, CAP_KM_S)
        assert bound.best_time_term == pytest.approx(max(_f(score, 0.0), _f(score, 7200.0)),
                                                     abs=1e-12)


def _positive_zeros(value: object) -> bool:
    if isinstance(value, dict):
        return all(_positive_zeros(item) for item in value.values())
    if isinstance(value, float):
        return math.copysign(1.0, value) == 1.0 or value != 0.0
    return True


@pytest.mark.parametrize("rules, score, offset_s", CASES)
def test_spending_is_priced_by_score_candidate_itself(
    scenarios: dict[str, Scenario], rules: str, score: ScoreConfig, offset_s: float
) -> None:
    """A whole impulse is not the derivative times 249 m/s: the logistic bends over it."""
    shifted = dataclasses.replace(scenarios[rules], time_origin_offset_s=offset_s)
    for row in (_row(2500.0, 1.0), _row(None, 0.4, closest_km=30.0)):
        found = _levers(row, shifted, score)
        before = score_candidate(row, shifted, score).raw_score
        for extra, lost in ((IMPULSE_KM_S, found.impulse_points_lost),
                            (BUDGET_KM_S - row.total_dv_km_s, found.remaining_budget_points_lost)):
            after = score_candidate(_row(row.intercept_time_s, row.total_dv_km_s + extra,
                                         closest_km=row.minimum_distance_km or 0.0),
                                    shifted, score).raw_score
            assert lost == pytest.approx(before - after, abs=1e-12)
        assert found.impulse_km_s == IMPULSE_KM_S
        assert found.remaining_budget_km_s == pytest.approx(BUDGET_KM_S - row.total_dv_km_s)


def test_a_whole_impulse_is_not_the_marginal_rate_times_its_size(scenario: Scenario) -> None:
    # ΔV - Cv = -1500 m/s, on the logistic's convex side: the derivative undercounts.
    row = _row(2500.0, 0.0)
    found = _levers(row, scenario, RANK)
    linear = found.points_per_km_s_saved * IMPULSE_KM_S
    assert found.impulse_points_lost > 1.05 * linear


def test_spending_stops_at_the_floor(scenario: Scenario) -> None:
    row = _row(2500.0, 1.0)
    breakdown = score_candidate(row, scenario, RANK, penalty_count=8)
    assert 0.0 < breakdown.raw_score < 10.0
    found = _of(row, breakdown, scenario, RANK)
    after = score_candidate(_row(2500.0, BUDGET_KM_S), scenario, RANK, penalty_count=8).raw_score
    # The rest of the budget would cost 12.5 points of delta-v term; the floor caps it.
    assert after == 0.0
    assert found.remaining_budget_points_lost == breakdown.raw_score
    over = _levers(_row(2500.0, 2.995), scenario, RANK)
    assert over.remaining_budget_km_s == 0.0 and over.remaining_budget_points_lost == 0.0


@pytest.mark.parametrize("score", [
    dataclasses.replace(RANK, kt=-0.003, kv=0.0),
    dataclasses.replace(RANK, kv=-0.0),
    dataclasses.replace(RANK, kt=-0.0, kv=-0.0),
])
def test_no_negative_zero_reaches_the_json(scenario: Scenario, score: ScoreConfig) -> None:
    for row in (_row(3000.0, 1.0, closest_km=1.0), _row(None, 1.5, closest_km=5.0)):
        shown = _levers(row, scenario, score).to_dict()
        assert _positive_zeros(shown), shown
        assert not re.search(r"-0\.0(?!\d)", json.dumps(shown, allow_nan=False))


def test_rewarded_lateness_has_no_break_even(scenario: Scenario) -> None:
    found = _levers(_row(2500.0, 1.0), scenario, dataclasses.replace(RANK, kt=-0.003))
    assert found.points_per_s_earlier is not None and found.points_per_s_earlier < 0.0
    assert found.break_even_s_per_km_s is None


@pytest.mark.parametrize("kt", [0.003, -0.003, 1e-4, 0.05])
def test_a_term_at_its_best_end_has_no_headroom_at_all(scenario: Scenario, kt: float) -> None:
    """any_answer - 50 - best_dv left -3.6e-15 here; the best end itself is stored."""
    score = dataclasses.replace(RANK, kt=kt, ct=3333.3)
    best_end = 0.0 if kt > 0.0 else 7200.0
    found = _levers(_row(best_end, 0.0), scenario, score)
    assert found.time_headroom == 0.0
    assert found.delta_v_headroom == 0.0


def test_a_slope_between_the_cancellation_and_the_guard_is_not_zero(scenario: Scenario) -> None:
    """kt (T - Ct) = -100: past where F (1 - F/25) cancels, well inside the 700 guard."""
    row = _row(3000.0 - 100.0 / RANK.kt, 1.0)
    found = _levers(row, scenario, RANK)
    assert found.points_per_s_earlier == pytest.approx(RANK.kt * 25.0 * math.exp(-100.0),
                                                       rel=1e-9)


def test_a_break_even_past_the_largest_float_is_none(scenario: Scenario) -> None:
    # kt (T - Ct) = -699 with kt 1e-10: a time rate of 7e-313 pt/s, nonzero but tiny.
    score = dataclasses.replace(RANK, kt=1e-10)
    found = _levers(_row(3000.0 - 699.0 / 1e-10, 1.0), scenario, score)
    assert found.points_per_s_earlier is not None and 0.0 < found.points_per_s_earlier < 1e-300
    assert found.points_per_km_s_saved > 1.0
    assert found.break_even_s_per_km_s is None


def test_the_distance_rate_is_taken_where_the_alignment_rows_put_it(scenario: Scenario) -> None:
    """closest_approach_window from_origin: the alignment's 20 km is what is scored."""
    aligned = dataclasses.replace(scenario, alignment_minimum_distance_km=20.0)
    row = _row(None, 0.4, closest_km=30.0)
    found = _levers(row, aligned, RANK)
    assert found.points_per_km_closer == pytest.approx(
        -distance_term_slope(20.0, interception_rules("ranking"), RANK.cr_km), rel=1e-12)
    assert found.points_per_km_closer != pytest.approx(
        -distance_term_slope(30.0, interception_rules("ranking"), RANK.cr_km), rel=1e-3)


def test_a_hit_is_what_completes_before_t_max_not_what_is_flagged(scenario: Scenario) -> None:
    flagged_off = dataclasses.replace(_row(2500.0, 1.0), successful=False)
    found = _levers(flagged_off, scenario, RANK)
    assert found.hit and found.points_per_s_earlier is not None
    assert found.points_per_s_earlier == pytest.approx(
        _gain(flagged_off, scenario, RANK, "intercept_time_s", 0.01), rel=1e-6)


def test_a_score_at_exactly_zero_is_not_floored(scenario: Scenario) -> None:
    """Unfloored at exactly 0, an improvement still raises the score."""
    row = _row(2500.0, 1.0)
    found = _of(row, ScoreBreakdown(50.0, 25.0, 25.0, 100.0, 0.0), scenario, RANK)
    assert not found.floored and found.points_per_s_earlier
    assert found.points_per_km_s_saved > 0.0
