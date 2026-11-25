"""The mission clock: T_team, T_max and Ct from the earlier craft's epoch, and T_max on §3.

The organizer answered Team 54 in writing on 2026-11-04 18:17 that the mission
starts at the earlier epoch (「從比較早開始的開始算」). Rules §3 put T_max where
SC_Alien, past perigee, first climbs out to the mission boundary.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import FAKE_STARTUP

from orbitwin.cli import main
from orbitwin.config import (
    OrbitWinConfig,
    PolicyConfig,
    ScoreConfig,
    load_config,
    validate_config,
)
from orbitwin.conic import seconds_to_periapsis
from orbitwin.errors import ConfigurationError, ParseError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.mission_clock import check_horizon
from orbitwin.models import CLEARLY_ECCENTRIC, StateVector
from orbitwin.orbit import propagate_universal
from orbitwin.state_probe import load_scenario

ROOT = Path(__file__).resolve().parents[2]
CROSSING = ROOT / "tests" / "fixtures" / "async" / "aligned_crossing.script"
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
FAKE = ROOT / "scripts" / "fake_gmat_console.py"
STARTUP = FAKE_STARTUP
MU = 398600.4418
#: The fixture: SC_Alien 5400 s before SC_Human.
DELTA_S = 5400.0
ANSWER = "從比較早開始的開始算"


def _config(**policy: object) -> OrbitWinConfig:
    settings: dict[str, object] = {
        "rules": "ranking", "horizon_s": 20000.0, "total_dv_limit_km_s": 3.0,
        "total_dv_safe_limit_km_s": 2.9, "dv_limit_km_s": 0.25, "dv_safe_limit_km_s": 0.249,
        "mission_time_origin": "earliest", "closest_approach_window": "after_alignment",
        # Ours, which certify under the ranking rules writes into the report's name.
        "team_number": 54,
    }
    settings.update(policy)
    return OrbitWinConfig(
        policy=dataclasses.replace(PolicyConfig(), **settings),  # type: ignore[arg-type]
        pursuer_name="SC_Human",
        target_name="SC_Alien",
    )


def _load(config: OrbitWinConfig, script: Path = CROSSING):
    scenario, _, _ = load_scenario(script, config, gmat_console=FAKE, startup_file=STARTUP)
    return scenario


def _pursuer_early(tmp_path: Path) -> Path:
    """The fixture with the epochs swapped, SC_Human 5400 s first."""
    text = CROSSING.read_text(encoding="utf-8")
    for old, new in (
        ("SC_Human.Epoch = '01 Jan 2026 12:00:00.000';", "SC_Human.Epoch = 'X';"),
        ("SC_Alien.Epoch = '01 Jan 2026 10:30:00.000';",
         "SC_Alien.Epoch = '01 Jan 2026 12:00:00.000';"),
        ("SC_Human.Epoch = 'X';", "SC_Human.Epoch = '01 Jan 2026 10:30:00.000';"),
        ("Propagate Prop(SC_Alien) {SC_Alien.UTCModJulian = SC_Human.UTCModJulian};",
         "Propagate Prop(SC_Human) {SC_Human.UTCModJulian = SC_Alien.UTCModJulian};"),
    ):
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "pursuer_early.script"
    path.write_text(text, encoding="utf-8")
    return path


# --- whose epoch ------------------------------------------------------------------------


def test_the_alien_first_is_the_origin() -> None:
    scenario = _load(_config())
    assert scenario.time_origin_craft == "SC_Alien"
    assert scenario.time_origin_offset_s == pytest.approx(DELTA_S, abs=1e-6)
    assert scenario.max_time_s == pytest.approx(20000.0 - DELTA_S, abs=1e-6)
    # What the preliminary's explicit choice of that craft gave.
    named = _load(_config(rules="preliminary", mission_time_origin="target"))
    assert named.time_origin_offset_s == scenario.time_origin_offset_s
    assert named.max_time_s == scenario.max_time_s


def test_the_human_first_is_the_origin(tmp_path: Path) -> None:
    """R8's shape: main's documented `target` read T_team 5400 s short here."""
    path = _pursuer_early(tmp_path)
    scenario = _load(_config(), path)
    assert scenario.time_origin_craft == "SC_Human"
    assert scenario.time_origin_offset_s == pytest.approx(DELTA_S, abs=1e-6)
    assert scenario.max_time_s == pytest.approx(20000.0 - DELTA_S, abs=1e-6)
    late = _load(_config(rules="preliminary", mission_time_origin="target"), path)
    assert late.time_origin_craft == "SC_Alien" and late.time_origin_offset_s == 0.0
    assert late.max_time_s == pytest.approx(20000.0, abs=1e-6)


def test_with_one_start_either_craft_reads_the_same() -> None:
    config = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    for origin in (None, "earliest"):
        policy = dataclasses.replace(config.policy, horizon_s=6711.667, mission_time_origin=origin)
        scenario, _ = parse_gmat_script(R1, dataclasses.replace(config, policy=policy))
        assert scenario.alignment is None
        assert scenario.time_origin_craft == "SC_Human"
        assert (scenario.time_origin_offset_s, scenario.max_time_s) == (0.0, 6711.667)


@pytest.mark.parametrize("origin", ["target", "pursuer"])
def test_the_ranking_rules_refuse_any_other_origin(origin: str, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=f"2026-11-04 18:17 to Team 54, 「{ANSWER}」"):
        validate_config(_config(mission_time_origin=origin))
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"].update(horizon_s=6711.667, mission_time_origin=origin)
    path = tmp_path / "day.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=ANSWER):
        load_config(path)
    # The preliminary rules are unchanged.
    validate_config(_config(rules="preliminary", horizon_s=None, mission_time_origin=origin))


@pytest.mark.parametrize("origin", ["earliest", None])
def test_the_ranking_rules_take_earliest_or_nothing(origin: str | None) -> None:
    validate_config(_config(mission_time_origin=origin))


def test_craft_that_start_apart_without_an_origin_are_refused_with_the_answer() -> None:
    with pytest.raises(ParseError, match=f"'earliest', by .*{ANSWER}"):
        _load(_config(mission_time_origin=None))


# --- T_max against SC_Alien's coast ------------------------------------------------------


def _r1(horizon_s: float, boundary_km: float | None = None, rules: str = "ranking"):
    config = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    policy = dataclasses.replace(
        config.policy, horizon_s=horizon_s, mission_boundary_km=boundary_km,
        mission_time_origin="earliest",
    )
    if rules != "ranking":
        policy = dataclasses.replace(policy, rules=rules)
        config = dataclasses.replace(config, search=dataclasses.replace(
            config.search, ranking_search=False))
    scenario, _ = parse_gmat_script(R1, dataclasses.replace(config, policy=policy))
    return [warning for warning in scenario.warnings if warning.startswith("WARNING: T_max ")]


@pytest.mark.parametrize("horizon_s", [111.86, 4499.0])
def test_a_t_max_before_the_aliens_perigee_is_refused(horizon_s: float) -> None:
    """R1's SC_Alien passes perigee at 4500.000 s two-body, 4500.238 s in GMAT; 111.86 is
    T_max typed in minutes."""
    with pytest.raises(ParseError, match=r"before SC_Alien's perigee at t = 4500\.000 s"):
        _r1(horizon_s)


def test_a_t_max_past_the_perigee_is_taken() -> None:
    assert _r1(4501.0) == []
    assert _r1(6711.667) == []


def test_the_preliminary_rules_do_not_check_t_max_against_a_perigee() -> None:
    assert _r1(111.86, rules="preliminary") == []


@pytest.mark.parametrize(
    ("horizon_s", "off_km"),
    [
        (8511.667, "+9636.437"),  # R6's T_max on R1
        (6771.667, "+337.668"),  # A1's step 5 typo
    ],
)
def test_a_t_max_where_the_alien_is_off_the_boundary_is_warned(
    horizon_s: float, off_km: str
) -> None:
    warnings = _r1(horizon_s, 17000.0)
    assert len(warnings) == 1, warnings
    assert f"{off_km} km from the announced boundary 17000.0 km" in warnings[0]
    # No boundary announced, nothing to compare.
    assert _r1(horizon_s) == []


def test_the_announced_t_max_is_at_the_boundary() -> None:
    # Two-body puts R1's SC_Alien 1.813 km out at 6711.667 s, inside 10 km and the roundings.
    assert _r1(6711.667, 17000.0) == []


#: Seconds from t = 0 to the eccentric SC_Alien's perigee in _eccentric_alien.
PERIGEE_S = 2000.0


def _eccentric_alien(tmp_path: Path) -> Path:
    """The fixture with SC_Alien on an inbound hyperbola, perigee PERIGEE_S after t = 0.
    The fake console's alignment leaves the early craft where it is."""
    radius = 6827.0
    at_perigee = StateVector((radius, 0.0, 0.0), (0.0, math.sqrt(4.0 + 2.0 * MU / radius), 0.0))
    state = propagate_universal(at_perigee, -PERIGEE_S, MU)
    text = CROSSING.read_text(encoding="utf-8")
    for axis, value in zip(("X", "Y", "Z", "VX", "VY", "VZ"), (*state.r, *state.v), strict=True):
        text, count = re.subn(rf"^SC_Alien\.{axis} = .*;$", f"SC_Alien.{axis} = {float(value)!r};",
                              text, flags=re.M)
        assert count == 1
    # The organizer's placeholder, which certify fills with our number.
    old_name = "OfficialReport.Filename = 'OfficialReport.txt';"
    assert old_name in text
    text = text.replace(old_name, "OfficialReport.Filename = 'ODC_2026_TeamXX.txt';")
    path = tmp_path / "eccentric_alien.script"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_perigee_is_on_the_solver_clock_after_an_alignment(tmp_path: Path) -> None:
    """SC_Alien flies the 5400 s alignment and is inbound at t = 0: T_max from its own
    epoch lands on the solver clock 5400 s earlier."""
    script = _eccentric_alien(tmp_path)
    check = _load(_config(), script).horizon_check
    assert check is not None and check.ecc > 1.0
    assert check.perigee_s == pytest.approx(PERIGEE_S, abs=1e-6)
    with pytest.raises(ParseError, match="before SC_Alien's perigee"):
        _load(_config(horizon_s=DELTA_S + PERIGEE_S - 10.0), script)
    taken = _load(_config(horizon_s=DELTA_S + PERIGEE_S + 10.0), script)
    assert taken.time_origin_craft == "SC_Alien" and not taken.horizon_before_perigee_accepted


def test_a_near_circular_alien_is_warned_not_refused() -> None:
    """J2 gives a circular LEO an osculating e up to 3e-3, and so a "perigee" anywhere in a
    period: the fixture's SC_Alien, e = 0.001, would have been refused at a correct T_max.
    Its T_max is then unchecked, so every T_max is warned, a typo too."""
    scenario = _load(_config())
    check = scenario.horizon_check
    assert check is not None and check.ecc < 0.002 and check.near_circular
    assert check.perigee_s is not None and 1000.0 < check.perigee_s < 1100.0
    for horizon_s in (DELTA_S + check.perigee_s - 10.0, 20000.0, DELTA_S + 111.86):
        loaded = _load(_config(horizon_s=horizon_s))
        assert loaded.horizon_check is not None and not loaded.horizon_check.before_perigee
        warned = [w for w in loaded.warnings if w.startswith("WARNING: T_max ")]
        assert len(warned) == 1, loaded.warnings
        assert "near circular (osculating e 0.001" in warned[0]
        assert "the rules' premise" in warned[0] and "Confirm T_max with the organizer" in warned[0]
    # A clearly eccentric SC_Alien is checked, so it is not warned.
    assert not [w for w in _r1_scenario_warnings() if w.startswith("WARNING: T_max ")]


def _r1_scenario_warnings() -> tuple[str, ...]:
    config = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    policy = dataclasses.replace(config.policy, horizon_s=6711.667, mission_time_origin="earliest")
    scenario, _ = parse_gmat_script(R1, dataclasses.replace(config, policy=policy))
    return scenario.warnings


@pytest.mark.parametrize(("ecc", "refused"), [(0.04, False), (0.06, True)])
def test_the_eccentricity_gate(ecc: float, refused: bool) -> None:
    """Either side of CLEARLY_ECCENTRIC, 0.05, an ellipse inbound 100 s before perigee."""
    assert CLEARLY_ECCENTRIC == 0.05
    rp = 7000.0
    a = rp / (1.0 - ecc)
    at_perigee = StateVector((rp, 0.0, 0.0), (0.0, math.sqrt(MU * (2.0 / rp - 1.0 / a)), 0.0))
    inbound = propagate_universal(at_perigee, -100.0, MU)
    check = check_horizon(inbound, 50.0, 50.0, None, MU)
    assert check.perigee_s == pytest.approx(100.0, abs=1e-6)
    assert check.before_perigee is refused and check.before_osculating_perigee


def _hyperbola_at_perigee() -> StateVector:
    radius = 6827.0
    return StateVector((radius, 0.0, 0.0), (0.0, math.sqrt(4.0 + 2.0 * MU / radius), 0.0))


def _ellipse_at_perigee() -> StateVector:
    rp, ra = 7000.0, 20000.0
    a = 0.5 * (rp + ra)
    return StateVector((rp, 0.0, 0.0), (0.0, math.sqrt(MU * (2.0 / rp - 1.0 / a)), 0.0))


@pytest.mark.parametrize("perigee", [_hyperbola_at_perigee, _ellipse_at_perigee])
@pytest.mark.parametrize("before_s", [10.0, 1500.0])
def test_the_check_takes_the_two_body_perigee_when_inbound(perigee, before_s: float) -> None:
    inbound = propagate_universal(perigee(), -before_s, MU)
    assert check_horizon(inbound, 1.0, 1.0, None, MU).perigee_s == pytest.approx(
        before_s, abs=1e-6)
    outbound = propagate_universal(perigee(), before_s, MU)
    assert check_horizon(outbound, 1.0, 1.0, None, MU).perigee_s is None
    assert seconds_to_periapsis(np.asarray(inbound.r), np.asarray(inbound.v), MU) == (
        pytest.approx(before_s, abs=1e-6))


def test_a_circular_alien_has_no_perigee_to_pass() -> None:
    circle = StateVector((7000.0, 0.0, 0.0), (0.0, math.sqrt(MU / 7000.0), -1e-12))
    check = check_horizon(circle, 100.0, 100.0, None, MU)
    assert check.perigee_s is None and not check.before_perigee


def test_an_alien_descending_through_the_boundary_is_off_it() -> None:
    """Past apogee an ellipse comes back down through the radius: §3's crossing climbs."""
    start = propagate_universal(_ellipse_at_perigee(), 10.0, MU)

    def radius(t: float) -> float:
        return float(np.linalg.norm(propagate_universal(start, t, MU).r))

    lo, hi = 8000.0, 12000.0  # a period of 15612 s: apogee near 7800 s
    assert radius(lo) > 17000.0 > radius(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if radius(mid) > 17000.0 else (lo, mid)
    check = check_horizon(start, lo, lo, 17000.0, MU)
    assert abs(check.radius_km - 17000.0) < 1e-3 and check.radial_rate_km_s < 0.0
    assert check.perigee_s is None and check.off_boundary


# --- what inspect prints ----------------------------------------------------------------


def _day_config(tmp_path: Path, **policy: object) -> Path:
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"].update(
        mission_time_origin="earliest", closest_approach_window="after_alignment", **policy
    )
    path = tmp_path / "day.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("swap", [False, True], ids=["alien first", "human first"])
def test_inspect_names_the_origin_craft_the_head_start_and_the_solver_t_max(
    swap: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _pursuer_early(tmp_path) if swap else CROSSING
    early, late = ("SC_Human", "SC_Alien") if swap else ("SC_Alien", "SC_Human")
    config = _day_config(tmp_path, horizon_s=20000.0, mission_boundary_km=17000.0)
    code = main(["inspect", str(script), "--config", str(config), "--gmat-console", str(FAKE),
                 "--startup-file", str(STARTUP)])
    err = capsys.readouterr().err
    assert code == 0, err
    assert (
        f"orbitwin inspect: mission clock: origin earliest is {early}'s epoch; {early} starts "
        f"Δ = 5400.000000 s before {late}. T_team is solver t + 5400.000000 s; T_max 20000.0 s "
        f"from {early}'s epoch is 14600.000000 s on the solver clock, which starts at "
        f"{late}'s epoch"
    ) in err
    assert "orbitwin inspect: T_max check (two-body, solver clock): SC_Alien " in err
    assert "the 17000.0 km boundary: " in err


def test_inspect_with_one_start_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _day_config(tmp_path, horizon_s=6711.667, mission_boundary_km=17000.0)
    code = main(["inspect", str(R1), "--config", str(config)])
    err = capsys.readouterr().err
    assert code == 0, err
    assert ("mission clock: origin earliest; the craft start together, so T_team is the solver clock; "
            "T_max 6711.667 s is 6711.667000 s on it") in err
    assert ("T_max check (two-body, solver clock): SC_Alien passes perigee at t = 4500.000 s "
            "(osculating e 1.07); at T_max, t = 6711.667 s, it is 17001.813 km from the centre, climbing") in err
    assert "at the 17000.0 km boundary: +1.813 km" in err


# --- an announced T_max before the perigee, taken by override ----------------------------


def test_the_refusal_says_to_stop_and_ask_the_organizer() -> None:
    with pytest.raises(ParseError) as refused:
        _r1(111.86)
    message = str(refused.value)
    assert "STOP and ask the organizer whether this T_max is right" in message
    assert "Only after they confirm it in writing" in message
    assert "--accept-announced-horizon-before-perigee-s <that T_max>" in message


def _overridden(horizon_s: float, **policy: object) -> OrbitWinConfig:
    return _config(horizon_s=horizon_s, accept_announced_horizon_before_perigee_s=horizon_s,
                   **policy)


def _r1_policy(horizon_s: float, accepted: float | None):
    config = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    return dataclasses.replace(config, policy=dataclasses.replace(
        config.policy, horizon_s=horizon_s, mission_time_origin="earliest",
        accept_announced_horizon_before_perigee_s=accepted,
    ))


def test_with_the_override_r1_takes_a_t_max_before_the_perigee() -> None:
    scenario, _ = parse_gmat_script(R1, _r1_policy(111.86, 111.86))
    assert scenario.max_time_s == 111.86 and scenario.horizon_before_perigee_accepted
    assert scenario.horizon_check is not None
    assert scenario.horizon_check.perigee_s == pytest.approx(4500.0, abs=1e-3)
    overrides = [w for w in scenario.warnings if w.startswith("OVERRIDE ")]
    assert len(overrides) == 1 and "before SC_Alien's perigee at t = 4500.000 s" in overrides[0]
    # Past the perigee the override changes nothing and records nothing.
    scenario, _ = parse_gmat_script(R1, _r1_policy(6711.667, 6711.667))
    assert not scenario.horizon_before_perigee_accepted
    assert not [w for w in scenario.warnings if w.startswith("OVERRIDE ")]


def test_the_override_covers_only_the_t_max_it_names() -> None:
    """A later typo of T_max does not ride on an override written for another value."""
    with pytest.raises(ConfigurationError, match="accepts only the T_max it names"):
        validate_config(_r1_policy(111.87, 111.86))
    # Built in code past validation, the parser still refuses the other value.
    with pytest.raises(ParseError, match="before SC_Alien's perigee"):
        parse_gmat_script(R1, _r1_policy(111.87, 111.86))


def _before_perigee_s() -> float:
    return DELTA_S + PERIGEE_S - 10.0


def test_inspect_records_the_override(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    horizon_s = _before_perigee_s()
    config = _day_config(tmp_path, horizon_s=horizon_s,
                         accept_announced_horizon_before_perigee_s=horizon_s)
    code = main(["inspect", str(_eccentric_alien(tmp_path)), "--config", str(config),
                 "--gmat-console", str(FAKE), "--startup-file", str(STARTUP)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    record = __import__("json").loads(captured.out)["announced_horizon_before_perigee"]
    assert record["max_time_s"] == pytest.approx(horizon_s - DELTA_S, abs=1e-6)
    assert record["alien_perigee_s"] == pytest.approx(PERIGEE_S, abs=1e-6)
    assert record["alien_perigee_from_origin_s"] == pytest.approx(horizon_s + 10.0, abs=1e-6)
    assert "orbitwin inspect: OVERRIDE policy.accept_announced_horizon_before_perigee_s" in (
        captured.err
    )


def test_solve_rank_and_certify_record_the_override(tmp_path: Path) -> None:
    from conftest import quick_config, stamped

    from orbitwin.pipeline import solve_scenario
    from orbitwin.rank import rank_report
    from orbitwin.search import zero_burn_candidate
    from orbitwin.verify import certify_candidate

    horizon_s = _before_perigee_s()
    config = dataclasses.replace(
        quick_config(), policy=_overridden(horizon_s).policy, pursuer_name="SC_Human",
        target_name="SC_Alien",
        score=ScoreConfig(kt=0.003, ct=6000.0, kv=0.0012, cv=2000.0, delta_v_unit="m/s",
                          cr_km=300.0),
    )
    eccentric = _eccentric_alien(tmp_path)
    summary = solve_scenario(eccentric, config, tmp_path / "solve", gmat_console=FAKE,
                             startup_file=STARTUP, gmat_verify_top=0)
    record = summary["announced_horizon_before_perigee"]
    assert record is not None and record["alien_perigee_from_origin_s"] == pytest.approx(
        horizon_s + 10.0, abs=1e-6
    )
    assert any(w.startswith("OVERRIDE ") for w in summary["warnings"])
    script = tmp_path / "miss.script"
    script.write_text(eccentric.read_text(encoding="utf-8") + "% FAKE_GMAT_MISS_KM 900\n",
                      encoding="utf-8")
    scenario = _load(config, script)
    miss = stamped(zero_burn_candidate(scenario, config), script, config)
    assert rank_report(scenario, [miss], config)["clock"][  # type: ignore[index]
        "announced_horizon_before_perigee"] == record
    certificate = certify_candidate(script, miss, config, FAKE, STARTUP, tmp_path / "certify")
    assert certificate["clock"]["announced_horizon_before_perigee"] == record
    # Without the override none of them gets that far.
    plain = dataclasses.replace(
        config, policy=dataclasses.replace(
            config.policy, accept_announced_horizon_before_perigee_s=None))
    with pytest.raises(ParseError, match="before SC_Alien's perigee"):
        solve_scenario(eccentric, plain, tmp_path / "plain", gmat_console=FAKE,
                       startup_file=STARTUP, gmat_verify_top=0)


def test_the_day_config_writes_the_override_and_keeps_it_to_the_day(tmp_path: Path) -> None:
    import subprocess
    import sys

    def run(*args: str, script: Path = R1) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "ranking_day_config.py"),
                               "--script", str(script), "--team-number", "54",
                               # Fewer cores than the profile's 8 would add a WARNING.
                               "--gmat-workers", "1", *args],
                              capture_output=True, text=True, check=False,
                              env={k: v for k, v in os.environ.items()
                                   if not k.startswith("GMAT_") and k != "BASH_ENV"})

    first = tmp_path / "day.yaml"
    done = run("--output", str(first), "--horizon-s", "111.86",
               "--accept-announced-horizon-before-perigee-s", "111.86")
    assert done.returncode == 0, done.stderr
    assert load_config(first).policy.accept_announced_horizon_before_perigee_s == 111.86
    assert "override    : T_max 111.86 s is taken if inspect finds it before" in done.stdout
    assert "WARNING" not in done.stdout
    other = run("--output", str(tmp_path / "other.yaml"), "--horizon-s", "111.87",
                "--accept-announced-horizon-before-perigee-s", "111.86")
    assert other.returncode == 2 and "names the one announced T_max it accepts" in other.stderr
    # Without it the readback flies R1's SC_Alien and refuses that T_max.
    refused = run("--output", str(tmp_path / "refused.yaml"), "--horizon-s", "111.86")
    assert refused.returncode == 2 and "before SC_Alien's perigee" in refused.stderr
    assert not (tmp_path / "refused.yaml").exists()
    # SC_Alien past its perigee at t = 0: any T_max fits, and no override is written.
    climbing = tmp_path / "climbing.script"
    climbing.write_text(R1.read_text(encoding="utf-8").replace(
        "SC_Alien.TA = 241.21247020102464;", "SC_Alien.TA = 118.78752979897536;"),
        encoding="utf-8")
    plain = run("--output", str(tmp_path / "plain.yaml"), "--horizon-s", "111.86",
                script=climbing)
    assert plain.returncode == 0, plain.stderr
    assert load_config(tmp_path / "plain.yaml").policy.accept_announced_horizon_before_perigee_s \
        is None and "before SC_Alien's perigee" not in plain.stdout
    score = ["--kt-per-s", "0.003", "--ct-s", "90", "--kv", "0.0012", "--cv", "2000",
             "--cr-km", "300", "--delta-v-unit", "m/s"]
    scored = run("--output", str(tmp_path / "scored.yaml"), "--horizon-s", "111.86", *score,
                 "--day-config", str(first))
    assert scored.returncode == 0, scored.stderr
    assert load_config(tmp_path / "scored.yaml").policy.accept_announced_horizon_before_perigee_s \
        == 111.86
    retyped = run("--output", str(tmp_path / "retyped.yaml"), "--horizon-s", "111.86", *score,
                  "--accept-announced-horizon-before-perigee-s", "111.86", "--day-config",
                  str(tmp_path / "plain.yaml"))
    assert retyped.returncode == 2
    assert "--accept-announced-horizon-before-perigee-s 111.86 here, None in" in retyped.stderr
    assert ("Write step 2's config again with --accept-announced-horizon-before-perigee-s, "
            "then this one from it") in retyped.stderr


def test_inspect_prints_a_boundary_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """R6's T_max on R1 puts SC_Alien 9636 km past the boundary; the RUNBOOK reads WARNINGs."""
    config = _day_config(tmp_path, horizon_s=8511.667, mission_boundary_km=17000.0)
    code = main(["inspect", str(R1), "--config", str(config)])
    err = capsys.readouterr().err
    assert code == 0, err
    assert ("orbitwin inspect: WARNING: T_max 8511.667 s puts SC_Alien 26636.437 km from the "
            "centre, +9636.437 km from the announced boundary 17000.0 km") in err
