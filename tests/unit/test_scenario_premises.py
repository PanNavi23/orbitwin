from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig
from orbitwin.errors import ParseError
from orbitwin.gmat_parser import parse_gmat_script

# Two premises the preliminary hands us, and two different answers to breaking
# them. 說明會簡報 p.8 states the target is circular, so a target that is not
# says the scenario is not the described one -- but the period is just as well
# defined for an ellipse, nothing is ambiguous, and refusing on a guessed
# tolerance would mean producing nothing if the official state carries a little
# numerical eccentricity. A start inside the planet is different in kind: it is
# not a flight, so it is refused. A pericenter inside it is not: an inserted
# craft may start on an arc it has to burn off, so that is said, with when.


def _rewrite(script: Path, tmp_path: Path, replacements: dict[str, str]) -> Path:
    text = script.read_text(encoding="utf-8-sig")
    for old, new in replacements.items():
        assert old in text, old
        text = text.replace(old, new, 1)
    out = tmp_path / "mutated.script"
    out.write_text(text, encoding="utf-8")
    return out


def test_an_eccentric_target_is_reported_and_still_solved(
    different_plane_script: Path, tmp_path: Path
) -> None:
    config = OrbitWinConfig()
    baseline, _ = parse_gmat_script(different_plane_script, config)
    assert not [w for w in baseline.warnings if "circular" in w]

    mutated = _rewrite(different_plane_script, tmp_path, {"AlienSC.ECC = 0;": "AlienSC.ECC = 0.01;"})
    scenario, _ = parse_gmat_script(mutated, config)
    assert [w for w in scenario.warnings if "circular" in w], scenario.warnings
    # Said, not refused: the horizon is still computed and the scenario usable.
    assert scenario.max_time_s > 0.0


def test_the_tolerance_decides_and_is_configurable(
    different_plane_script: Path, tmp_path: Path
) -> None:
    mutated = _rewrite(different_plane_script, tmp_path, {"AlienSC.ECC = 0;": "AlienSC.ECC = 1e-9;"})
    tight = OrbitWinConfig()
    assert tight.policy.target_circular_tolerance > 1.0e-9
    scenario, _ = parse_gmat_script(mutated, tight)
    assert not [w for w in scenario.warnings if "circular" in w], (
        "a competition-day state carrying numerical eccentricity must not be flagged"
    )

    strict = dataclasses.replace(
        tight, policy=dataclasses.replace(tight.policy, target_circular_tolerance=1.0e-12)
    )
    scenario, _ = parse_gmat_script(mutated, strict)
    assert [w for w in scenario.warnings if "circular" in w]


def test_a_start_inside_the_body_is_refused(
    different_plane_script: Path, tmp_path: Path
) -> None:
    mutated = _rewrite(
        different_plane_script, tmp_path, {"AlienSC.SMA = 9000;": "AlienSC.SMA = 6300;"}
    )
    with pytest.raises(ParseError, match="starts 6300 km from the centre, inside"):
        parse_gmat_script(mutated, OrbitWinConfig())


@pytest.mark.parametrize("craft", ["Pursuer", "AlienSC"])
def test_a_start_that_dives_is_said_not_refused(
    different_plane_script: Path, tmp_path: Path, craft: str
) -> None:
    """Apogee 6800 km, pericenter 6300 km, from apogee: above ground, falling."""
    from orbitwin.orbit import first_radius_crossing_s

    sma = "Pursuer.SMA = 7000;" if craft == "Pursuer" else "AlienSC.SMA = 9000;"
    ecc = "Pursuer.ECC = 0;" if craft == "Pursuer" else "AlienSC.ECC = 0;"
    ta = "Pursuer.TA = 300;" if craft == "Pursuer" else "AlienSC.TA = 20;"
    mutated = _rewrite(
        different_plane_script,
        tmp_path,
        {sma: f"{craft}.SMA = 6550;", ecc: f"{craft}.ECC = {500 / 13100!r};", ta: f"{craft}.TA = 180;"},
    )
    config = OrbitWinConfig()
    config = dataclasses.replace(config, policy=dataclasses.replace(config.policy, horizon_s=6000.0))
    scenario, _ = parse_gmat_script(mutated, config)
    spec = scenario.pursuer if craft == "Pursuer" else scenario.target
    crossing_s = first_radius_crossing_s(spec.initial_state, 6378.1363, 6000.0, scenario.mu_km3_s2)
    assert crossing_s is not None and 1978.0 < crossing_s < 1979.0
    (said,) = [w for w in scenario.warnings if "reaches the 6378.14 km central body" in w]
    assert f"t = {crossing_s:.3f} s" in said
    assert ("the zero burn is not one" in said) is (craft == "Pursuer")


def test_a_dive_after_the_horizon_is_not_said(different_plane_script: Path, tmp_path: Path) -> None:
    mutated = _rewrite(
        different_plane_script,
        tmp_path,
        {"Pursuer.SMA = 7000;": "Pursuer.SMA = 6550;",
         "Pursuer.ECC = 0;": f"Pursuer.ECC = {500 / 13100!r};",
         "Pursuer.TA = 300;": "Pursuer.TA = 180;"},
    )
    config = OrbitWinConfig()
    config = dataclasses.replace(config, policy=dataclasses.replace(config.policy, horizon_s=1900.0))
    scenario, _ = parse_gmat_script(mutated, config)
    assert not [w for w in scenario.warnings if "central body at" in w]


def test_a_low_but_real_orbit_is_still_accepted(
    different_plane_script: Path, tmp_path: Path
) -> None:
    # The refusal must not reach a legal low orbit. 6500 km circular is 122 km
    # of altitude: unpleasant, entirely above the surface, and not our call.
    mutated = _rewrite(different_plane_script, tmp_path, {"Pursuer.SMA = 7000;": "Pursuer.SMA = 6500;"})
    scenario, _ = parse_gmat_script(mutated, OrbitWinConfig())
    assert scenario.max_time_s > 0.0


# The central body decides mu, and mu decides the period, the horizon and every
# Lambert arc. The two-body layer carries one mu, from the config, so a script
# about another body is not a harder problem -- it is a different one, answered
# with the wrong constant.


@pytest.mark.parametrize(
    ("what", "find", "replace"),
    [
        ("CentralBody", "FM.CentralBody = Earth;", "FM.CentralBody = Luna;"),
        ("PrimaryBodies", "FM.PrimaryBodies = {Earth};", "FM.PrimaryBodies = {Luna};"),
        ("GravityField", "FM.GravityField.Earth.Degree = 0;", "FM.GravityField.Luna.Degree = 0;"),
    ],
)
def test_a_force_model_about_another_body_is_refused(
    different_plane_script: Path, tmp_path: Path, what: str, find: str, replace: str
) -> None:
    mutated = _rewrite(different_plane_script, tmp_path, {find: replace})
    with pytest.raises(ParseError):
        parse_gmat_script(mutated, OrbitWinConfig())


def test_the_check_is_against_the_configured_body_not_the_word_earth(
    different_plane_script: Path, tmp_path: Path
) -> None:
    # Solving a Luna problem is legitimate as long as the constants agree; what
    # is refused is a script and a config that describe different bodies.
    mutated = _rewrite(
        different_plane_script,
        tmp_path,
        {
            "FM.CentralBody = Earth;": "FM.CentralBody = Luna;",
            "FM.PrimaryBodies = {Earth};": "FM.PrimaryBodies = {Luna};",
            "FM.GravityField.Earth.Degree = 0;": "FM.GravityField.Luna.Degree = 0;",
            "FM.GravityField.Earth.Order = 0;": "FM.GravityField.Luna.Order = 0;",
        },
    )
    lunar = dataclasses.replace(
        OrbitWinConfig(),
        policy=dataclasses.replace(
            OrbitWinConfig().policy,
            central_body_name="Luna",
            mu_km3_s2=4902.800118,
            central_body_radius_km=1737.4,
        ),
    )
    scenario, _ = parse_gmat_script(mutated, lunar)
    assert scenario.mu_km3_s2 == pytest.approx(4902.800118)
    # And the same file against the Earth config is refused.
    with pytest.raises(ParseError, match="Luna"):
        parse_gmat_script(mutated, OrbitWinConfig())


def test_extra_primary_bodies_are_reported_not_refused(
    different_plane_script: Path, tmp_path: Path
) -> None:
    # A third body is a richer model than the seed layer carries, which is the
    # same situation as drag or a gravity harmonic: GMAT is the authority and
    # the seed says what it ignored.
    mutated = _rewrite(
        different_plane_script, tmp_path,
        {"FM.PrimaryBodies = {Earth};": "FM.PrimaryBodies = {Earth, Luna};"},
    )
    scenario, _ = parse_gmat_script(mutated, OrbitWinConfig())
    assert [w for w in scenario.warnings if "Luna" in w], scenario.warnings


def test_under_the_ranking_rules_a_dive_is_named_a_disqualification(
    project_root: Path, tmp_path: Path
) -> None:
    import re

    from conftest import quick_config

    from orbitwin.pipeline import solve_scenario

    text = (project_root / "examples" / "ranking" / "R1_NearPlane_Reachable.script").read_text(
        encoding="utf-8"
    )
    for key, value in {"SMA": "6550", "ECC": repr(500 / 13100), "TA": "180"}.items():
        text = re.sub(rf"^SC_Human\.{key} = .*;$", f"SC_Human.{key} = {value};", text, flags=re.M)
    script = tmp_path / "falling.script"
    script.write_text(text, encoding="utf-8")
    base = quick_config()
    config = dataclasses.replace(
        base,
        pursuer_name="SC_Human", target_name="SC_Alien",
        policy=dataclasses.replace(
            base.policy, rules="ranking", horizon_s=6711.667, dv_limit_km_s=0.25,
            dv_safe_limit_km_s=0.249, total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99,
        ),
    )
    assert "SC_Human.SMA = 6550;" in text
    summary = solve_scenario(script, config, tmp_path / "solve")
    assert summary["best_candidate"] is None
    (said,) = [w for w in summary["warnings"] if "central body before completing" in w]
    assert "The ranking rules disqualify it" in said and "do not forbid" not in said
