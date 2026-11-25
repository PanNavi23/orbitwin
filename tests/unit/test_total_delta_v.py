from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, validate_config
from orbitwin.constraints import (
    FORMAL_FRONTIER,
    ROBUST_FRONTIER,
    apply_constraints,
    frontier_of,
    validate_candidate,
)
from orbitwin.errors import ConfigurationError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Candidate, Scenario


def _ranking(**overrides: object) -> OrbitWinConfig:
    """The ranking round's limits (2026/10/06 §2, §4 note 1), with our margins."""
    base = OrbitWinConfig()
    fields: dict[str, object] = {
        "dv_limit_km_s": 0.25,
        "dv_safe_limit_km_s": 0.2495,
        "total_dv_limit_km_s": 3.0,
        "total_dv_safe_limit_km_s": 2.995,
    }
    fields.update(overrides)
    return dataclasses.replace(base, policy=dataclasses.replace(base.policy, **fields))


def _burns(magnitudes_km_s: list[float], *, alternate: bool = False) -> Candidate:
    burns = tuple(
        Burn(
            101.0 * index,
            (sign * magnitude, 0.0, 0.0),
            (sign * magnitude, 0.0, 0.0),
        )
        for index, magnitude in enumerate(magnitudes_km_s)
        for sign in [(-1.0 if alternate and index % 2 else 1.0)]
    )
    return Candidate(
        burns=burns,
        intercept_time_s=2000.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.0,
        branch="test",
    )


@pytest.fixture(scope="module")
def scenario(workshop_script: Path) -> Scenario:
    return parse_gmat_script(workshop_script, _ranking())[0]


def test_the_rule_value_itself_is_legal(scenario: Scenario) -> None:
    candidate = _burns([0.25] * 12)
    assert candidate.total_dv_km_s == 3.0
    assert validate_candidate(candidate, scenario, _ranking()).valid


def test_between_the_margin_and_the_rule_is_formal(scenario: Scenario) -> None:
    # Every impulse inside its own margin, so only the total can say formal.
    config = _ranking()
    candidate = _burns([0.2307] * 13)
    assert 2.995 < candidate.total_dv_km_s < 3.0
    result = validate_candidate(candidate, scenario, config)
    assert result.valid
    assert [warning for warning in result.warnings if "total delta-v" in warning]
    assert frontier_of(candidate, config) == FORMAL_FRONTIER


def test_over_the_rule_disqualifies_rather_than_deducts(scenario: Scenario) -> None:
    """§4 note 1(c): 不列入排名. A ten-point penalty would leave it ranked."""
    config = _ranking()
    candidate = _burns([0.25] * 11 + [0.25 + 1.0e-9])
    result = apply_constraints(candidate, scenario, config)
    assert not result.valid
    assert any("total delta-v" in violation for violation in result.violations)
    assert not any("total" in penalty for penalty in candidate.penalties)


def test_the_total_is_the_sum_of_norms(scenario: Scenario) -> None:
    """Opposite impulses cancel in the vector sum, not in the fuel spent (eq. 4)."""
    candidate = _burns([0.2] * 16, alternate=True)
    assert candidate.vector_sum_dv_km_s == pytest.approx(0.0, abs=1e-12)
    assert not validate_candidate(candidate, scenario, _ranking()).valid


def test_inside_the_margin_is_robust(scenario: Scenario) -> None:
    config = _ranking()
    candidate = _burns([0.2495] * 12)
    result = validate_candidate(candidate, scenario, config)
    assert result.valid and not result.warnings
    assert frontier_of(candidate, config) == ROBUST_FRONTIER


def test_no_total_rule_means_no_total_check(scenario: Scenario) -> None:
    config = _ranking(total_dv_limit_km_s=None, total_dv_safe_limit_km_s=None)
    assert validate_candidate(_burns([0.2495] * 40), scenario, config).valid


@pytest.mark.parametrize(
    "limit, safe",
    [(3.0, None), (None, 2.995), (3.0, 3.0001), (math.nan, 2.995), (3.0, -1.0), (3.0, 0.0)],
)
def test_an_unusable_total_limit_is_refused(limit: float | None, safe: float | None) -> None:
    with pytest.raises(ConfigurationError, match="total_dv"):
        validate_config(_ranking(total_dv_limit_km_s=limit, total_dv_safe_limit_km_s=safe))
