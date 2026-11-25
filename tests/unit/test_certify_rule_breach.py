"""certify names the rule a plan breaks, before calling it legal.

It used to check only the frontier, so a 0.26 km/s burn under the ranking rules,
which disqualifies, was refused as "on the formal frontier: legal under the
rules but outside the production margins".
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from conftest import quick_config, ranking_copy, ranking_quick_config, stamped

from orbitwin.errors import PolicyError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Candidate
from orbitwin.verify import certify_candidate, rule_breach


def _plan(*burns: tuple[float, float]) -> Candidate:
    """Burns as (time s, magnitude km/s), with a hit at 4000 s."""
    return Candidate(
        burns=tuple(Burn(t, (dv, 0.0, 0.0), (dv, 0.0, 0.0), i) for i, (t, dv) in enumerate(burns)),
        intercept_time_s=4000.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="test",
        metadata={"arrival_time_s": 4000.0, "event_refined": True},
    )


@pytest.fixture
def ranking(workshop_script: Path, tmp_path: Path):
    config = ranking_quick_config()
    script = ranking_copy(workshop_script, tmp_path)
    scenario, _ = parse_gmat_script(script, config)
    return script, scenario, config


def test_each_ranking_breach_is_named_with_its_clause(ranking) -> None:
    _, scenario, config = ranking
    assert rule_breach(_plan((1000.0, 0.2)), scenario, config) is None
    over = rule_breach(_plan((1000.0, 0.26)), scenario, config)
    assert over is not None and "breaks the ranking rules (2026/11/04)" in over
    assert "exceeds formal limit 0.25 km/s (§4 note 1(a): disqualifies)" in over
    total = rule_breach(_plan(*((1000.0 + 200.0 * i, 0.24) for i in range(13))), scenario, config)
    assert total is not None
    assert "exceeds the 3 km/s limit (§4 note 1(c): disqualifies)" in total
    gap = rule_breach(_plan((1000.0, 0.2), (1099.0, 0.2)), scenario, config)
    assert gap is not None
    assert "is below 100 s (the 100 s gap: -10 points per maneuver)" in gap
    assert "legal" not in over + total + gap


def test_certify_refuses_a_breach_by_its_rule_not_as_legal(
    ranking, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script, _, config = ranking
    plan = stamped(_plan((1000.0, 0.26)), script, config)
    allowed = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, allow_formal_frontier_submission=True)
    )
    for each in (config, allowed):
        with pytest.raises(PolicyError, match=r"§4 note 1\(a\): disqualifies") as refused:
            certify_candidate(script, plan, each, fake_gmat, fake_startup, tmp_path / "c", runs=1)
        assert "legal under the rules" not in str(refused.value)
    assert not (tmp_path / "c").exists()


def test_a_preliminary_breach_names_its_deduction(workshop_script: Path) -> None:
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    over = rule_breach(_plan((1000.0, 1.6)), scenario, config)
    assert over is not None and "breaks the preliminary rules (2026/05/20)" in over
    assert "exceeds formal limit 1.5 km/s (§2, §5: -10 points per maneuver)" in over
