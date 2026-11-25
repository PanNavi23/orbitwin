"""The ranking rules' distance term as revised 2026/11/04: 100/(1 + e^{((Δr_min - 5)/C_r)^2}).

The preliminary keeps 50 e^{-(Δr_min - 5)/100}. The table was computed apart
from this code, to two decimals.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import math
import re
import types
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, ScoreConfig, load_config, validate_config
from orbitwin.errors import ConfigurationError
from orbitwin.ranking_search import _Problem
from orbitwin.rules import PRELIMINARY, RANKING
from orbitwin.score import (
    Ceilings,
    ScoreBreakdown,
    distance_term,
    distance_term_slope,
)

ROOT = Path(__file__).resolve().parents[2]
CRS_KM = (50.0, 100.0, 300.0, 1000.0, 3000.0)
TABLE = {
    25.0: (46.01, 49.00, 49.89, 49.99, 50.00),
    100.0: (2.63, 28.85, 47.50, 49.77, 49.97),
    500.0: (0.00, 0.00, 6.17, 43.90, 49.32),
    1000.0: (0.0, 0.0, 0.0, 27.09, 47.25),
    4711.0: (0.0, 0.0, 0.0, 0.0, 7.87),
}
GRID_KM = (0.0, 4.999, 5.0, 5.001, 6.0, 12.5, 25.0, 88.0, 100.0, 319.4, 500.0, 1000.0,
           4711.0, 1.0e5, 1.0e9)


@pytest.mark.parametrize("distance_km", sorted(TABLE))
def test_the_table(distance_km: float) -> None:
    got = tuple(round(distance_term(distance_km, RANKING, cr), 2) for cr in CRS_KM)
    assert got == pytest.approx(TABLE[distance_km], abs=1e-9)


@pytest.mark.parametrize("cr_km", CRS_KM)
def test_a_hit_scores_50_and_the_top_is_flat(cr_km: float) -> None:
    for inside in (0.0, 1.0, 4.999, 5.0):
        assert distance_term(inside, RANKING, cr_km) == 50.0
        assert distance_term_slope(inside, RANKING, cr_km) == 0.0
    # 49 out to 5 + 0.2 C_r, 25 at 5 + C_r √ln 3.
    assert distance_term(5.0 + 0.2 * cr_km, RANKING, cr_km) == pytest.approx(49.0002, abs=1e-4)
    assert distance_term(5.0 + 0.2 * cr_km * (1.0 - 1e-9), RANKING, cr_km) >= 49.0
    half = 5.0 + cr_km * math.sqrt(math.log(3.0))
    assert distance_term(half, RANKING, cr_km) == pytest.approx(25.0, abs=1e-12)
    assert math.sqrt(math.log(3.0)) == pytest.approx(1.048, abs=5e-4)


@pytest.mark.parametrize("cr_km", CRS_KM)
def test_monotone(cr_km: float) -> None:
    values = [distance_term(d, RANKING, cr_km) for d in GRID_KM]
    assert all(b <= a for a, b in itertools.pairwise(values))
    assert values[-1] == 0.0


@pytest.mark.parametrize(("distance_km", "cr_km"), [
    (6.0, 50.0), (25.0, 50.0), (60.0, 50.0), (100.0, 100.0), (200.0, 300.0), (319.4, 300.0),
    (900.0, 1000.0), (1000.0, 3000.0), (4711.0, 3000.0),
])
def test_the_slope_is_the_derivative(distance_km: float, cr_km: float) -> None:
    step = 1.0e-3
    central = (distance_term(distance_km + step, RANKING, cr_km)
               - distance_term(distance_km - step, RANKING, cr_km)) / (2.0 * step)
    assert distance_term_slope(distance_km, RANKING, cr_km) == pytest.approx(central, rel=1e-6)


def test_far_misses_underflow_to_zero_without_raising() -> None:
    # u^2 = 700 is the guard; e^710 would overflow.
    for u in (26.4, 26.5, 27.0, 1.0e3, 1.0e200):
        assert distance_term(5.0 + u, RANKING, 1.0) >= 0.0
        assert distance_term_slope(5.0 + u, RANKING, 1.0) <= 0.0
    assert distance_term(5.0 + 27.0, RANKING, 1.0) == 0.0
    assert distance_term(math.inf, RANKING, 1.0) == 0.0


@pytest.mark.parametrize("distance_km", GRID_KM)
def test_the_preliminary_term_is_unchanged(distance_km: float) -> None:
    old = 50.0 * math.exp(-(max(distance_km, 5.0) - 5.0) / 100.0)
    assert distance_term(distance_km, PRELIMINARY, None) == old


@pytest.mark.parametrize("cr_km", [None, 0.0, -300.0, math.nan, math.inf])
def test_the_ranking_term_is_never_guessed(cr_km: float | None) -> None:
    with pytest.raises(ConfigurationError, match="C_r"):
        distance_term(100.0, RANKING, cr_km)


def _bound(cr_km: float | None) -> Ceilings:
    return Ceilings(any_answer=95.0, any_miss=70.0, best_dv_term=15.0, time_term_at_horizon=5.0,
                    best_time_term=30.0, cr_km=cr_km)


def _best(rest: float) -> ScoreBreakdown:
    """A best whose distance term is all a miss at the best ends must beat."""
    return ScoreBreakdown(rest, 5.0, 15.0, 0.0, rest + 20.0)


@pytest.mark.parametrize("cr_km", CRS_KM)
@pytest.mark.parametrize("rest", [1.0e-6, 0.01, 1.0, 7.87, 25.0, 46.01, 49.0, 49.999])
def test_the_inversion_round_trips(cr_km: float, rest: float) -> None:
    within = _bound(cr_km).miss_must_be_within_km(_best(rest), RANKING)
    assert within is not None and math.isfinite(within) and within > 5.0
    assert distance_term(within, RANKING, cr_km) == pytest.approx(rest, rel=1e-9)
    assert distance_term(within * (1.0 - 1e-6), RANKING, cr_km) > rest
    assert within == pytest.approx(5.0 + cr_km * math.sqrt(math.log(100.0 / rest - 1.0)))


def test_the_inversion_ends() -> None:
    bound = _bound(300.0)
    assert bound.miss_must_be_within_km(_best(50.0), RANKING) is None
    assert bound.miss_must_be_within_km(_best(60.0), RANKING) is None
    assert bound.miss_must_be_within_km(_best(-0.1), RANKING) == math.inf
    # Floored at zero, any positive score beats it.
    floored = ScoreBreakdown(0.0, 0.0, 0.0, 60.0, 0.0)
    assert bound.miss_must_be_within_km(floored, RANKING) == math.inf
    with pytest.raises(ConfigurationError, match="C_r"):
        _bound(None).miss_must_be_within_km(_best(25.0), RANKING)
    # The preliminary's own inversion stands.
    old = _bound(None).miss_must_be_within_km(_best(25.0), PRELIMINARY)
    assert old == pytest.approx(5.0 + 100.0 * math.log(2.0))


@pytest.mark.parametrize("cr_km", [100.0, 300.0])
def test_a_best_whose_term_underflows_is_only_tied_past_the_edge(cr_km: float) -> None:
    """R2's zero burn at C_r 100 km: its term is exactly 0, and so is any miss's past
    5 + C_r √700, so only a miss inside that edge beats it."""
    assert distance_term(4711.155, RANKING, 100.0) == 0.0
    within = _bound(cr_km).miss_must_be_within_km(_best(0.0), RANKING)
    assert within == pytest.approx(5.0 + cr_km * math.sqrt(700.0), rel=1e-12)
    assert distance_term(within, RANKING, cr_km) > 0.0
    assert distance_term(within * (1.0 + 1e-9), RANKING, cr_km) == 0.0
    # Below the smallest term the formula yields, the same edge.
    assert _bound(cr_km).miss_must_be_within_km(_best(1e-310), RANKING) == within


def test_the_preliminary_inversion_has_its_own_edge() -> None:
    within = _bound(None).miss_must_be_within_km(_best(0.0), PRELIMINARY)
    assert within is not None and math.isfinite(within)
    assert distance_term(within * (1.0 - 1e-9), PRELIMINARY, None) > 0.0


def _search(cr_km: float | None, rules: str = "ranking", radius_km: float = 5.0) -> _Problem:
    problem = _Problem.__new__(_Problem)
    problem.setup = types.SimpleNamespace(  # type: ignore[assignment]
        limits=types.SimpleNamespace(intercept_radius_km=radius_km),
        score=ScoreConfig(kt=0.003, ct=4000.0, kv=0.0012, cv=2000.0, cr_km=cr_km),
        rules=rules,
    )
    return problem


@pytest.mark.parametrize("cr_km", CRS_KM)
def test_the_miss_phase_climbs_the_official_term(cr_km: float) -> None:
    problem = _search(cr_km)
    for miss_km in (5.0, 6.0, 25.0, 88.0, 319.4, 1000.0, 4711.0):
        value, slope = problem._distance_term(miss_km)
        assert value == distance_term(miss_km, RANKING, cr_km)
        assert slope == distance_term_slope(miss_km, RANKING, cr_km)
        step = 1.0e-3
        central = (problem._distance_term(miss_km + step)[0]
                   - problem._distance_term(miss_km - step)[0]) / (2.0 * step)
        if miss_km > 5.0:
            assert slope == pytest.approx(central, rel=1e-6, abs=1e-12)
    # Inside, the score: flat at 50.
    assert problem._distance_term(5.0) == (50.0, 0.0)
    assert problem._distance_term(4.0) == (50.0, 0.0)
    assert problem._distance_term(1.0) == (50.0, 0.0)


@pytest.mark.parametrize("cr_km", [1.0, 100.0])
def test_the_miss_phase_measures_from_the_policys_sphere(cr_km: float) -> None:
    """A 4 km policy sphere left a band from 4 to 5 km with no gradient at all."""
    problem = _search(cr_km, radius_km=4.0)
    for miss_km in (4.2, 4.6, 4.9, 30.0):
        value, slope = problem._distance_term(miss_km)
        assert value == distance_term(miss_km + 1.0, RANKING, cr_km) < 50.0
        assert slope == distance_term_slope(miss_km + 1.0, RANKING, cr_km) < 0.0
    assert problem._distance_term(4.0) == (50.0, 0.0)


def test_the_miss_phase_under_the_preliminary_is_its_term() -> None:
    problem = _search(None, rules="preliminary")
    value, slope = problem._distance_term(105.0)
    assert value == 50.0 * math.exp(-1.0) and slope == -value / 100.0


# -- config


def _ranking(**score: object) -> OrbitWinConfig:
    base = OrbitWinConfig()
    policy = dataclasses.replace(base.policy, rules="ranking", dv_limit_km_s=0.25,
                                 dv_safe_limit_km_s=0.249, total_dv_limit_km_s=3.0,
                                 total_dv_safe_limit_km_s=2.99)
    fields = {"kt": 0.003, "ct": 4000.0, "kv": 0.0012, "cv": 2000.0, "cr_km": 300.0, **score}
    return dataclasses.replace(base, policy=policy, score=ScoreConfig(**fields))  # type: ignore[arg-type]


def test_a_ranking_score_in_m_per_s_with_c_r_is_accepted() -> None:
    validate_config(_ranking())


@pytest.mark.parametrize(
    ("score", "message"),
    [
        ({"cr_km": None}, "no cr_km"),
        ({"cr_km": 0.0}, "cr_km must be positive"),
        ({"cr_km": -300.0}, "cr_km must be positive"),
        ({"cr_km": math.nan}, "score.cr_km must be a finite number"),
        ({"cr_km": math.inf}, "score.cr_km must be a finite number"),
        ({"cr_km": "300"}, "score.cr_km must be a number"),
        ({"delta_v_unit": "km/s", "kv": 1.2, "cv": 2.0}, "fix ΔV_team in m/s"),
        ({"time_unit": "min"}, "Only seconds"),
    ],
)
def test_a_ranking_score_block_fails_closed(score: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigurationError, match=re.escape(message)):
        validate_config(_ranking(**score))


def test_the_preliminary_refuses_c_r() -> None:
    base = OrbitWinConfig(score=ScoreConfig(kt=0.001, ct=5000.0, kv=0.004, cv=800.0))
    validate_config(base)
    with pytest.raises(ConfigurationError, match="has no C_r"):
        validate_config(dataclasses.replace(
            base, score=dataclasses.replace(base.score, cr_km=300.0)  # type: ignore[arg-type]
        ))


def test_a_ranking_file_without_c_r_is_refused(tmp_path: Path) -> None:
    import yaml

    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["score"] = {"kt": 0.003, "ct": 4000.0, "kv": 0.0012, "cv": 2000.0,
                    "time_unit": "s", "delta_v_unit": "m/s"}
    path = tmp_path / "day.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="never guessed"):
        load_config(path)
    raw["score"]["cr_km"] = 300
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    loaded = load_config(path).score
    assert loaded is not None and loaded.cr_km == 300


# -- the independent re-checks compute what score.py computes


def _script(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["check_report_rows", "audit_burn_table"])
def test_the_re_checks_agree_with_score_py(name: str) -> None:
    script = _script(name)
    for distance_km in GRID_KM:
        assert script.distance_term(distance_km, PRELIMINARY, None) == distance_term(
            distance_km, PRELIMINARY, None
        )
        for cr_km in (*CRS_KM, 1.0, 1.0e6):
            assert script.distance_term(distance_km, RANKING, cr_km) == pytest.approx(
                distance_term(distance_km, RANKING, cr_km), rel=1e-12, abs=1e-300
            ), (distance_km, cr_km)
    with pytest.raises(ValueError, match="C_r"):
        script.distance_term(100.0, RANKING, None)


def test_c_r_is_checked_when_the_search_is_set_up(workshop_script: Path) -> None:
    """Not first in a worker process, where it surfaced as a pool failure."""
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.ranking_search import build_setup

    config = dataclasses.replace(_ranking(), policy=dataclasses.replace(
        _ranking().policy, horizon_s=7200.0))
    scenario, _ = parse_gmat_script(workshop_script, config)
    build_setup(scenario, config)
    missing = dataclasses.replace(config, score=dataclasses.replace(config.score, cr_km=None))
    with pytest.raises(ConfigurationError, match="no cr_km"):
        build_setup(scenario, missing)


def test_every_formula_is_priced_by_name() -> None:
    from orbitwin.rules import DistanceTerm

    assert PRELIMINARY.distance_term is DistanceTerm.EXPONENTIAL
    assert RANKING.distance_term is DistanceTerm.LOGISTIC
    # A rule set whose term is no DistanceTerm is refused, not priced as one of the two.
    odd = dataclasses.replace(RANKING, distance_term="logistic")  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        distance_term(100.0, odd, 300.0)


def test_the_rule_set_says_whether_it_takes_c_r() -> None:
    from orbitwin.rules import DistanceTerm

    assert DistanceTerm.LOGISTIC.takes_cr and RANKING.distance_term.takes_cr
    assert not DistanceTerm.EXPONENTIAL.takes_cr and not PRELIMINARY.distance_term.takes_cr


@pytest.mark.parametrize("text", ["0", "-30", "nan", "inf", "-inf", "x", ""])
def test_one_argparse_type_refuses_what_is_not_a_positive_finite_number(text: str) -> None:
    import argparse

    from orbitwin.argtypes import positive_finite

    with pytest.raises(argparse.ArgumentTypeError):
        positive_finite(text)
    assert positive_finite("300") == 300.0 and positive_finite("1e-3") == 1e-3


def test_solve_prints_the_zero_burns_closest_approach_for_step_5() -> None:
    from orbitwin.pipeline import zero_burn_line

    assert zero_burn_line({"zero_burn_closest_approach": None}) is None
    assert zero_burn_line({"zero_burn_closest_approach": {"km": 4711.1554, "measured_by": "gmat"}}) == (
        "the zero burn passes 4711.155 km off (GMAT); "
        "scripts/ranking_day_config.py --zero-burn-dmin-km 4711.155"
    )
    assert "(two-body, not GMAT)" in str(zero_burn_line(
        {"zero_burn_closest_approach": {"km": 30.0, "measured_by": "two-body"}}))


def test_a_zero_burn_gmat_never_flew_is_labelled_two_body() -> None:
    from orbitwin.models import Candidate
    from orbitwin.pipeline import _zero_burn_closest_approach

    zero = Candidate(
        burns=(), intercept_time_s=None, minimum_distance_km=3049.3,
        distance_at_completion_km=3049.3, successful=False, analytical_endpoint_error_km=None,
        branch="zero-burn",
    )
    assert _zero_burn_closest_approach([zero]) == {
        "km": 3049.3, "measured_by": "two-body", "gmat_verification_error": None,
    }
    assert _zero_burn_closest_approach([]) is None


def test_a_zero_burn_gmat_flew_but_then_refused_keeps_gmats_label() -> None:
    """verify writes GMAT's distance, then an altitude or constraint check can still raise."""
    from orbitwin.models import Candidate
    from orbitwin.pipeline import _zero_burn_closest_approach, zero_burn_line

    zero = Candidate(
        burns=(), intercept_time_s=None, minimum_distance_km=3061.2,
        distance_at_completion_km=3061.2, successful=False, analytical_endpoint_error_km=None,
        branch="zero-burn",
        metadata={"gmat_event": {"minimum_distance_km": 3061.2},
                  "gmat_verification_error": "GMAT flew it under the altitude floor"},
    )
    assert not zero.verified_by_gmat
    closest = _zero_burn_closest_approach([zero])
    assert closest == {"km": 3061.2, "measured_by": "gmat",
                       "gmat_verification_error": "GMAT flew it under the altitude floor"}
    line = zero_burn_line({"zero_burn_closest_approach": closest})
    assert line is not None and "(GMAT, though its verification failed)" in line
