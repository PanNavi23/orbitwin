"""Each rulebook's numbers, against the clause they come from."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
from conftest import organizer_file

from orbitwin.config import OrbitWinConfig, load_config, validate_config
from orbitwin.errors import ConfigurationError
from orbitwin.rules import (
    DUEL,
    INTERCEPTION_RULES,
    PRELIMINARY,
    RANKING,
    DistanceTerm,
    interception_rules,
)


def test_the_preliminary_rules_document_is_the_one_recorded() -> None:
    pdf = organizer_file("rules-folder") / PRELIMINARY.source.document
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == PRELIMINARY.source.sha256


def test_the_preliminary_is_history() -> None:
    assert PRELIMINARY.historical and not RANKING.historical
    assert (PRELIMINARY.impulse_limit_km_s, PRELIMINARY.impulse_epsilon_km_s) == (1.5, 0.0)
    assert PRELIMINARY.over_impulse_limit == "penalty"
    assert (PRELIMINARY.min_separation_s, PRELIMINARY.penalty_points) == (100.0, 10.0)
    assert PRELIMINARY.total_dv_limit_km_s is None and PRELIMINARY.total_clause is None
    assert PRELIMINARY.intercept_radius_km == 5.0
    assert (PRELIMINARY.horizon, PRELIMINARY.miss_scored, PRELIMINARY.miss_distance) == (
        "4*T_A", True, "at-completion"
    )
    assert (PRELIMINARY.earth_crossing, PRELIMINARY.earth_crossing_clause) == ("not-ruled", None)
    assert PRELIMINARY.target_forces == "gravity only" and PRELIMINARY.advance == 8
    assert (PRELIMINARY.distance_term, PRELIMINARY.score_units) == (DistanceTerm.EXPONENTIAL, None)


def test_the_ranking_rules_as_corrected() -> None:
    # 排位賽規則 2026/11/04 修訂版; it replaced ecd26af2...e281ee3 (2026/10/06).
    assert RANKING.source.sha256 == (
        "1ded3ebbaebb658477643c0e05391b7ba2bc040ac30dd18f0297beb4dbbc9e7e"
    )
    assert RANKING.source.header_date == "2026/11/04"
    # §4 eq. (5) and the units sentence below it, as revised.
    assert RANKING.distance_term is DistanceTerm.LOGISTIC
    assert RANKING.score_units == ("s", "m/s")
    assert RANKING.intercept_radius_km == 5.0
    assert RANKING.impulse_limit_km_s == 0.25
    assert RANKING.impulse_epsilon_km_s == pytest.approx(1.0e-5 / 1000.0)
    assert RANKING.over_impulse_limit == "disqualify"
    assert (RANKING.total_dv_limit_km_s, RANKING.min_separation_s) == (3.0, 100.0)
    assert RANKING.penalty_points == 10.0
    assert (RANKING.horizon, RANKING.miss_scored, RANKING.miss_distance) == (
        "announced", True, "closest-approach"
    )
    assert (RANKING.earth_crossing, RANKING.earth_crossing_clause) == (
        "disqualify-before-interception", "§4 note 1(b)"
    )
    assert RANKING.target_forces == "every external force of §7"
    assert RANKING.advance == 4


def test_the_duel_is_a_different_contract() -> None:
    assert DUEL.source.sha256 == (
        "78c09cf72e7deb6ac3fcbc5f061964c9ac2c11f8f3bceb9497e616961ef94df4"
    )
    assert not isinstance(DUEL, type(RANKING))
    assert (DUEL.rounds, DUEL.decision_s, DUEL.impulses_per_round) == (6, 600.0, 1)
    assert (DUEL.over_limit, DUEL.total_dv_limit_km_s) == ("scale", None)
    assert (DUEL.submission_unit, DUEL.submission_min_decimals) == ("m/s", 2)
    assert DUEL.late_submission == (0.0, 0.0, 0.0)
    assert DUEL.outcome_order == ("capture", "earth-crossing", "terminal-semimajor-axis")
    assert DUEL.network_and_generative_ai_banned
    assert DUEL.announced == (
        "r0, v0 and epoch of each craft",
        "safe semimajor-axis interval",
        "impulse limit ΔV_lim",
        "capture radius R_cap",
        "round time T_round",
    )
    # Nothing the day announces is fixed in the rulebook.
    fields = {field.name for field in dataclasses.fields(DUEL)}
    assert not fields & {"impulse_limit_km_s", "capture_radius_km", "round_s", "horizon"}


def test_a_mission_config_cannot_name_the_duel() -> None:
    assert "duel" not in INTERCEPTION_RULES
    with pytest.raises(ValueError, match="not a mission config"):
        interception_rules("duel")
    base = OrbitWinConfig()
    config = dataclasses.replace(base, policy=dataclasses.replace(base.policy, rules="duel"))
    with pytest.raises(ConfigurationError, match="not a mission config"):
        validate_config(config)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("dv_limit_km_s", 0.26, r"disqualify an impulse over 0\.25 km/s"),
        ("total_dv_limit_km_s", 3.1, "disqualify a total over 3 km/s"),
        ("total_dv_limit_km_s", None, "total_dv_limit_km_s is unset"),
        ("min_burn_separation_s", 99.999, "closer than 100 s"),
        ("intercept_radius_km", 5.5, "intercept at 5 km"),
        ("allow_body_intersecting_submission", True, "crosses the Earth before the interception"),
    ],
)
def test_a_ranking_config_stays_inside_the_ranking_rules(
    project_root, field: str, value: object, match: str
) -> None:
    config = load_config(project_root / "config" / "ranking.yaml")
    changes = {field: value}
    if field == "total_dv_limit_km_s" and value is None:
        changes["total_dv_safe_limit_km_s"] = None
    bad = dataclasses.replace(config, policy=dataclasses.replace(config.policy, **changes))
    with pytest.raises(ConfigurationError, match=match):
        validate_config(bad)


def test_a_preliminary_config_stays_inside_its_rules() -> None:
    base = OrbitWinConfig()
    bad = dataclasses.replace(base, policy=dataclasses.replace(base.policy, dv_limit_km_s=1.6))
    with pytest.raises(ConfigurationError, match=r"penalise an impulse over 1\.5 km/s"):
        validate_config(bad)


def test_the_archived_profiles_are_the_preliminary(project_root) -> None:
    for name in ("odc2026.yaml", "odc2026-deep.yaml"):
        assert not (project_root / "config" / name).exists()
        config = load_config(project_root / "config" / "history" / "preliminary" / name)
        assert interception_rules(config.policy.rules) is PRELIMINARY
    for name in ("ranking.yaml", "ranking-rehearsal.yaml"):
        assert interception_rules(load_config(project_root / "config" / name).policy.rules) is RANKING


def test_the_cli_says_when_it_runs_under_the_historical_rules(
    workshop_script, project_root, capsys: pytest.CaptureFixture[str]
) -> None:
    from orbitwin.cli import main

    assert main(["inspect", str(workshop_script), "--rules", "preliminary"]) == 0
    assert "historical rulebook of 2026/05/20" in capsys.readouterr().err
    ranking = project_root / "config" / "ranking-rehearsal.yaml"
    main(["inspect", str(workshop_script), "--config", str(ranking)])
    assert "historical" not in capsys.readouterr().err
