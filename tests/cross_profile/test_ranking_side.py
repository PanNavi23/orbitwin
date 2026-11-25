"""One input, judged by the ranking rules, where the duel or the preliminary differ.

The duel's side of each row is in test_duel_side.py, with the duel engine.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, ScoreConfig, load_config
from orbitwin.constraints import interception_required, validate_candidate
from orbitwin.errors import ParseError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Candidate, Scenario
from orbitwin.score import score_candidate


def _ranking(project_root: Path) -> tuple[OrbitWinConfig, Scenario]:
    base = load_config(project_root / "config" / "ranking-rehearsal.yaml")
    config = dataclasses.replace(
        base, policy=dataclasses.replace(base.policy, horizon_s=6711.667)
    )
    script = project_root / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
    return config, parse_gmat_script(script, config)[0]


def _candidate(burns: list[tuple[float, float]], closest_km: float = 3.0) -> Candidate:
    return Candidate(
        burns=tuple(
            Burn(t, (dv, 0.0, 0.0), (dv, 0.0, 0.0), slot_index=i) for i, (t, dv) in enumerate(burns)
        ),
        intercept_time_s=4000.0,
        minimum_distance_km=closest_km,
        distance_at_completion_km=5.0,
        successful=closest_km <= 5.0,
        analytical_endpoint_error_km=None,
        branch="cross-profile",
    )


def test_300_m_s_is_disqualifying_not_a_deduction(project_root: Path) -> None:
    config, scenario = _ranking(project_root)
    result = validate_candidate(_candidate([(100.0, 0.300)]), scenario, config)
    assert not result.valid
    assert any("exceeds formal limit 0.25" in v for v in result.violations)
    assert result.rule_violations == ()  # §4 note 1(a): not ranked, not ten points off


def test_over_3_km_s_in_total_is_disqualifying(project_root: Path) -> None:
    config, scenario = _ranking(project_root)
    burns = [(100.0 + 101.0 * k, 0.249) for k in range(13)]  # 3.237 km/s
    result = validate_candidate(_candidate(burns), scenario, config)
    assert not result.valid
    assert any("total" in v for v in result.violations)


def test_a_second_burn_is_judged_on_its_separation(project_root: Path) -> None:
    config, scenario = _ranking(project_root)
    assert validate_candidate(_candidate([(100.0, 0.2), (201.0, 0.2)]), scenario, config).valid
    close = validate_candidate(_candidate([(100.0, 0.2), (150.0, 0.2)]), scenario, config)
    assert len(close.rule_violations) == 1  # §4 note 1(d): ten points per violating maneuver


def test_the_distance_term_is_full_inside_the_sphere(project_root: Path) -> None:
    _, scenario = _ranking(project_root)
    score = ScoreConfig(kt=0.001, ct=3000.0, kv=0.001, cv=1000.0, delta_v_unit="m/s",
                        cr_km=100.0)
    inside = score_candidate(_candidate([(100.0, 0.2)], closest_km=4.0), scenario, score)
    outside = score_candidate(_candidate([(100.0, 0.2)], closest_km=6.0), scenario, score)
    assert inside.distance_score == 50.0  # eq. (3): Δr_min = max(d_min, 5)
    # eq. (5) as revised 2026/11/04.
    assert outside.distance_score == pytest.approx(100.0 / (1.0 + math.exp(0.01**2)))


def test_a_miss_is_an_answer(project_root: Path) -> None:
    _, scenario = _ranking(project_root)
    assert not interception_required(scenario)


def test_the_preliminary_horizon_is_not_the_ranking_default(project_root: Path) -> None:
    base = load_config(project_root / "config" / "ranking-rehearsal.yaml")
    script = project_root / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
    with pytest.raises(ParseError, match="announced on the day"):
        parse_gmat_script(script, base)


def test_the_preliminary_judges_the_same_burn_differently(workshop_script: Path) -> None:
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    assert interception_required(scenario)
    legal_there = validate_candidate(_candidate([(100.0, 0.300)]), scenario, config)
    assert not any("exceeds formal limit" in v for v in legal_there.violations)
    over = validate_candidate(_candidate([(100.0, 1.6)]), scenario, config)
    assert len(over.rule_violations) == 1  # a deduction there, not a disqualification
