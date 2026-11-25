from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
import yaml

from orbitwin.config import OrbitWinConfig, load_config, validate_config
from orbitwin.errors import ConfigurationError, ParseError
from orbitwin.gmat_parser import parse_gmat_script

_ALIEN_SMA = "AlienSC.SMA = 9600;"
_ALIEN_ECC = "AlienSC.ECC = 0;"


def _with_alien(workshop_script: Path, tmp_path: Path, sma: str, ecc: str) -> Path:
    text = workshop_script.read_text(encoding="utf-8")
    assert _ALIEN_SMA in text and _ALIEN_ECC in text
    text = text.replace(_ALIEN_SMA, f"AlienSC.SMA = {sma};").replace(
        _ALIEN_ECC, f"AlienSC.ECC = {ecc};"
    )
    path = tmp_path / "alien.script"
    path.write_text(text, encoding="utf-8")
    return path


def _horizon(seconds: float | None) -> OrbitWinConfig:
    base = OrbitWinConfig()
    return dataclasses.replace(base, policy=dataclasses.replace(base.policy, horizon_s=seconds))


def test_an_escaping_target_needs_the_announced_horizon(
    workshop_script: Path, tmp_path: Path
) -> None:
    script = _with_alien(workshop_script, tmp_path, "-14000", "1.5")
    with pytest.raises(ParseError, match=r"policy\.horizon_s"):
        parse_gmat_script(script, _horizon(None))

    scenario, _ = parse_gmat_script(script, _horizon(7200.0))
    assert scenario.max_time_s == 7200.0
    assert scenario.target_period_s is None


@pytest.mark.parametrize("announced_s", [5000.0, 50000.0])
def test_the_announced_horizon_wins_over_four_periods(
    workshop_script: Path, announced_s: float
) -> None:
    # Both sides of 4*T_A = 37444 s, so neither min() nor max() would pass.
    scenario, _ = parse_gmat_script(workshop_script, _horizon(announced_s))
    assert scenario.max_time_s == announced_s
    assert scenario.target_period_s == pytest.approx(9360.905, abs=1e-3)


def test_the_ranking_rules_do_not_fall_back_to_four_periods(workshop_script: Path) -> None:
    """T_max comes after the next perigee, under 1.5 periods: 4*T_A is never it."""
    preliminary, _ = parse_gmat_script(workshop_script, _horizon(None))
    assert preliminary.max_time_s == pytest.approx(4.0 * preliminary.target_period_s)
    ranking = dataclasses.replace(
        _horizon(None), policy=dataclasses.replace(_horizon(None).policy, rules="ranking")
    )
    with pytest.raises(ParseError, match=r"set policy\.horizon_s"):
        parse_gmat_script(workshop_script, ranking)


def test_an_unknown_rulebook_is_refused() -> None:
    config = _horizon(None)
    with pytest.raises(ConfigurationError, match=r"policy\.rules"):
        validate_config(
            dataclasses.replace(config, policy=dataclasses.replace(config.policy, rules="final"))
        )


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_non_finite_horizon_cannot_slip_past_the_parser(
    workshop_script: Path, value: float
) -> None:
    # validate_config refuses it; this is the path that skips validate_config.
    with pytest.raises(ParseError, match="not finite"):
        parse_gmat_script(workshop_script, _horizon(value))


def test_the_circular_target_warning_is_the_preliminarys(
    workshop_script: Path, tmp_path: Path
) -> None:
    script = _with_alien(workshop_script, tmp_path, "10000", "0.05")
    preliminary, _ = parse_gmat_script(script, _horizon(None))
    ranking, _ = parse_gmat_script(script, _horizon(30000.0))
    assert any("circular" in warning for warning in preliminary.warnings)
    assert not any("circular" in warning for warning in ranking.warnings)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, "7200", True, 20.0, 2.0])
def test_an_unusable_horizon_is_refused(value: object) -> None:
    # 20 and 2 are not above the default 20 s final padding.
    with pytest.raises(ConfigurationError, match="horizon_s"):
        validate_config(_horizon(value))  # type: ignore[arg-type]


def test_a_yaml_horizon_is_read_and_kept_a_float(workshop_script: Path, tmp_path: Path) -> None:
    path = tmp_path / "ranking.yaml"
    path.write_text(yaml.safe_dump({"policy": {"horizon_s": 7200}}), encoding="utf-8")
    scenario, _ = parse_gmat_script(workshop_script, load_config(path))
    assert scenario.max_time_s == 7200.0
    assert isinstance(scenario.max_time_s, float)


def test_non_finite_target_elements_are_refused(workshop_script: Path, tmp_path: Path) -> None:
    """A Cartesian target never meets keplerian_to_cartesian's mu check.

    Its elements come from cartesian_to_keplerian, where mu = nan makes them
    nan, and nan fails both the closed and the open test.
    """
    scenario, _ = parse_gmat_script(workshop_script, _horizon(None))
    text = workshop_script.read_text(encoding="utf-8")
    for craft in (scenario.pursuer, scenario.target):
        lines = [f"{craft.name}.DisplayStateType = Keplerian;"] + [
            f"{craft.name}.{key} = {craft.fields[key]};"
            for key in ("SMA", "ECC", "INC", "RAAN", "AOP", "TA")
        ]
        cartesian = [f"{craft.name}.DisplayStateType = Cartesian;"] + [
            f"{craft.name}.{key} = {value!r};"
            for key, value in zip(
                ("X", "Y", "Z", "VX", "VY", "VZ"),
                (*craft.initial_state.r_km, *craft.initial_state.v_km_s),
                strict=True,
            )
        ]
        block = "\n".join(lines)
        assert block in text
        text = text.replace(block, "\n".join(cartesian))
    path = tmp_path / "cartesian.script"
    path.write_text(text, encoding="utf-8")
    assert parse_gmat_script(path, _horizon(None))[0].max_time_s > 0.0

    base = OrbitWinConfig()
    broken = dataclasses.replace(base, policy=dataclasses.replace(base.policy, mu_km3_s2=math.nan))
    with pytest.raises(ParseError, match=r"not finite .* under mu = nan"):
        parse_gmat_script(path, broken)
