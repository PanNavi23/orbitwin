from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
import yaml

from orbitwin.config import OrbitWinConfig, load_config, validate_config
from orbitwin.errors import ConfigurationError, ParseError
from orbitwin.gmat_parser import parse_gmat_script

# Not profiles: their keys are documentation, and load_config refuses them.
_NOT_PROFILES = {"editable-fields.yaml", "tolerances.yaml"}


def _ranking(**overrides: object) -> OrbitWinConfig:
    base = OrbitWinConfig()
    fields: dict[str, object] = {
        "rules": "ranking",
        "dv_limit_km_s": 0.25,
        "dv_safe_limit_km_s": 0.249,
        "total_dv_limit_km_s": 3.0,
        "total_dv_safe_limit_km_s": 2.99,
    }
    fields.update(overrides)
    return dataclasses.replace(base, policy=dataclasses.replace(base.policy, **fields))


def test_the_rule_values_themselves_are_accepted() -> None:
    validate_config(_ranking())
    validate_config(_ranking(dv_limit_km_s=0.25 + 1.0e-8))
    validate_config(_ranking(total_dv_limit_km_s=3.0))


def test_the_preliminary_per_burn_default_is_refused() -> None:
    """1.5 km/s is six times what §4 note 1(a) allows before disqualifying."""
    config = _ranking()
    policy = dataclasses.replace(
        config.policy,
        dv_limit_km_s=OrbitWinConfig().policy.dv_limit_km_s,
        dv_safe_limit_km_s=OrbitWinConfig().policy.dv_safe_limit_km_s,
    )
    with pytest.raises(ConfigurationError, match=r"policy\.dv_limit_km_s is 1\.5 km/s.*1\(a\)"):
        validate_config(dataclasses.replace(config, policy=policy))


def test_a_per_burn_limit_past_the_epsilon_is_refused() -> None:
    # One ulp past 0.25 km/s + 1e-5 m/s, so any wider epsilon lets it through.
    limit = math.nextafter(0.25 + 1.0e-8, math.inf)
    with pytest.raises(ConfigurationError, match=r"policy\.dv_limit_km_s is 0\.25000001"):
        validate_config(_ranking(dv_limit_km_s=limit))


def test_an_unset_total_is_refused() -> None:
    with pytest.raises(ConfigurationError, match=r"policy\.total_dv_limit_km_s is unset.*1\(c\)"):
        validate_config(_ranking(total_dv_limit_km_s=None, total_dv_safe_limit_km_s=None))


def test_a_total_over_three_is_refused_with_no_tolerance() -> None:
    limit = math.nextafter(3.0, math.inf)
    with pytest.raises(ConfigurationError, match=r"total_dv_limit_km_s is 3\.0000000000000004 km/s"):
        validate_config(_ranking(total_dv_limit_km_s=limit))


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"dv_safe_limit_km_s": 0.2501}, "dv_safe_limit_km_s"),
        ({"total_dv_safe_limit_km_s": 3.0001}, "total_dv_safe_limit_km_s"),
        ({"total_dv_safe_limit_km_s": None}, "total_dv_safe_limit_km_s"),
    ],
)
def test_the_rulebook_independent_checks_still_apply(
    overrides: dict[str, object], field: str
) -> None:
    with pytest.raises(ConfigurationError, match=field):
        validate_config(_ranking(**overrides))


def test_the_ranking_profile_holds_the_limits(project_root: Path, tmp_path: Path) -> None:
    path = project_root / "config" / "ranking.yaml"
    script = project_root / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
    profile = load_config(path)
    policy = profile.policy
    assert policy.rules == "ranking"
    # Ranking §1 names the craft; the open template cannot.
    assert (profile.pursuer_name, profile.target_name) == ("SC_Human", "SC_Alien")
    assert (policy.dv_limit_km_s, policy.dv_safe_limit_km_s) == (0.25, 0.249)
    assert (policy.total_dv_limit_km_s, policy.total_dv_safe_limit_km_s) == (3.0, 2.99)
    assert policy.horizon_s is None
    assert profile.score is None, "the coefficients are announced, not guessed"
    with pytest.raises(ParseError, match=r"policy\.horizon_s"):
        parse_gmat_script(script, profile)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["policy"]["horizon_s"] = 7200
    filled = tmp_path / "ranking.yaml"
    filled.write_text(yaml.safe_dump(raw), encoding="utf-8")
    scenario, _ = parse_gmat_script(script, load_config(filled))
    assert scenario.max_time_s == 7200.0
    assert scenario.rules == "ranking"
    assert (scenario.pursuer.name, scenario.target.name) == ("SC_Human", "SC_Alien")


def test_the_preliminary_profiles_are_unchanged(project_root: Path) -> None:
    validate_config(OrbitWinConfig())
    profiles = sorted(
        path for path in (
            *(project_root / "config").glob("*.yaml"),
            *(project_root / "config" / "history" / "preliminary").glob("*.yaml"),
        )
        if path.name not in _NOT_PROFILES and not path.name.startswith("ranking")
    )
    assert len(profiles) >= 7, [path.name for path in profiles]
    for path in profiles:
        policy = load_config(path).policy
        assert policy.rules == "preliminary", path.name
        assert policy.dv_limit_km_s == 1.5, path.name
        assert policy.total_dv_limit_km_s is None, path.name


def test_every_ranking_profile_holds_the_ranking_limits(project_root: Path) -> None:
    profiles = sorted((project_root / "config").glob("ranking*.yaml"))
    assert len(profiles) >= 2, [path.name for path in profiles]
    for path in profiles:
        policy = yaml.safe_load(path.read_text(encoding="utf-8"))["policy"]
        assert policy["rules"] in {"ranking", "ranking-onsite"}, path.name
        assert policy["dv_limit_km_s"] <= 0.25 and policy["dv_safe_limit_km_s"] < 0.25, path.name
        assert policy["total_dv_limit_km_s"] <= 3.0, path.name
        assert policy["total_dv_safe_limit_km_s"] < 3.0, path.name


def test_every_ranking_profile_runs_gmat_eight_at_once(project_root: Path) -> None:
    # Measured on R1 and R6; config/ranking.yaml has the numbers.
    profiles = sorted((project_root / "config").glob("ranking*.yaml"))
    assert len(profiles) >= 2, [path.name for path in profiles]
    for path in profiles:
        assert load_config(path).gmat.workers == 8, path.name


def test_solve_and_certify_refuse_a_config_built_in_code(
    workshop_script: Path, fake_gmat: Path, fake_startup: Path, tmp_path: Path
) -> None:
    """A config made with dataclasses.replace never passes through load_config."""
    from orbitwin.pipeline import solve_scenario
    from orbitwin.search import zero_burn_candidate
    from orbitwin.verify import certify_candidate

    preliminary = _ranking(dv_limit_km_s=1.5, dv_safe_limit_km_s=1.495, horizon_s=7200.0)
    with pytest.raises(ConfigurationError, match=r"policy\.dv_limit_km_s is 1\.5 km/s"):
        solve_scenario(workshop_script, preliminary, tmp_path / "solve")
    assert not (tmp_path / "solve").exists()

    scenario, _ = parse_gmat_script(workshop_script, _ranking(horizon_s=7200.0))
    candidate = zero_burn_candidate(scenario, _ranking(horizon_s=7200.0))
    with pytest.raises(ConfigurationError, match=r"policy\.dv_limit_km_s is 1\.5 km/s"):
        certify_candidate(workshop_script, candidate, preliminary, fake_gmat, fake_startup,
                          tmp_path / "certify")
    assert not (tmp_path / "certify").exists()
