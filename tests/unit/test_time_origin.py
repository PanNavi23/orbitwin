"""The time-origin evidence: which epoch the template and SC_Alien's track say T_max counts from.

The tracks here are two-body hyperbolas built in the test; production reads
GMAT's coast. The fake console only checks the plumbing: its SC_Alien stands still.
See docs/ASYNC_START_AND_INSERTION.md, "Time-origin evidence".
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import FAKE_STARTUP, quick_config

import orbitwin.time_origin as time_origin_module
from orbitwin.cli import main
from orbitwin.config import OrbitWinConfig, PolicyConfig, load_config
from orbitwin.ephemeris import Ephemeris, target_ephemeris_from_gmat
from orbitwin.errors import ConfigurationError, GmatRunError, ParseError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import StateVector
from orbitwin.orbit import propagate_universal
from orbitwin.pipeline import solve_scenario
from orbitwin.state_probe import load_scenario
from orbitwin.time_origin import (
    assess_time_origin,
    time_origin_evidence,
    written_half_step,
)

ROOT = Path(__file__).resolve().parents[2]
CROSSING = ROOT / "tests" / "fixtures" / "async" / "aligned_crossing.script"
FAKE = ROOT / "scripts" / "fake_gmat_console.py"
STARTUP = FAKE_STARTUP
MU = 398600.4418
#: The fixture: SC_Alien 5400 s before SC_Human, joint goal 6000 s.
DELTA_S = 5400.0
GOAL_S = 6000.0


def _config(**policy: object) -> OrbitWinConfig:
    settings: dict[str, object] = {
        "mission_time_origin": "target", "closest_approach_window": "after_alignment",
    }
    settings.update(policy)
    return OrbitWinConfig(
        policy=dataclasses.replace(PolicyConfig(), **settings),  # type: ignore[arg-type]
        pursuer_name="SC_Human",
        target_name="SC_Alien",
    )


def _load(config: OrbitWinConfig, script: Path = CROSSING):
    scenario, document, _ = load_scenario(script, config, gmat_console=FAKE, startup_file=STARTUP)
    return scenario, document


def _variant(tmp_path: Path, old: str, new: str, source: Path = CROSSING) -> Path:
    text = source.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / f"variant_{len(list(tmp_path.iterdir()))}.script"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


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


def _hyperbola(perigee_s: float) -> StateVector:
    """SC_Alien at perigee: 6827 km, v_inf 2 km/s."""
    radius = 6827.0
    return StateVector((radius, 0.0, 0.0), (0.0, math.sqrt(4.0 + 2.0 * MU / radius), 0.0))


def _track(start_s: float, end_s: float, perigee_s: float = 2000.0, step_s: float = 60.0):
    """SC_Alien on the solver clock, rows every ``step_s`` and one on each end."""
    times = np.unique(np.append(np.arange(start_s, end_s, step_s), end_s))
    states = [propagate_universal(_hyperbola(perigee_s), t - perigee_s, MU) for t in times]
    return Ephemeris(
        times, np.array([s.r for s in states]), np.array([s.v for s in states])
    )


def _radius_at(time_s: float, perigee_s: float = 2000.0) -> float:
    return float(np.linalg.norm(propagate_universal(_hyperbola(perigee_s), time_s - perigee_s, MU).r))


def _rate_at(time_s: float, perigee_s: float = 2000.0) -> float:
    state = propagate_universal(_hyperbola(perigee_s), time_s - perigee_s, MU)
    return float(np.dot(state.r, state.v) / np.linalg.norm(state.r))


_TRACK_LINE = re.compile(
    r"^time origin, track \(t from SC_Human's epoch\): GMAT's SC_Alien passes perigee at "
    r"t = (?P<perigee>\S+) s, (?P<low>\S+) km from the centre; "
    r"at SC_Alien's epoch \+ T_max \(t = (?P<t1>\S+) s\) (?P<r1>\S+) km, (?P<p1>past|before) "
    r"perigee, (?P<c1>climbing|not climbing) (?P<v1>\S+) km/s; "
    r"at SC_Human's epoch \+ T_max \(t = (?P<t2>\S+) s\) (?P<r2>\S+) km, (?P<p2>past|before) "
    r"perigee, (?P<c2>climbing|not climbing) (?P<v2>\S+) km/s$"
)


def _assess(config: OrbitWinConfig, track, script: Path = CROSSING, scenario=None):
    loaded, document = _load(config, script)
    return assess_time_origin(scenario or loaded, document, config.policy, track)


def _by(evidence) -> dict[str, object]:
    return {item.reading: item for item in evidence.readings}


# --- the two readings ------------------------------------------------------------------


def test_template_and_boundary_both_pick_the_targets_epoch() -> None:
    """T_max 11400 s from SC_Alien's epoch is 6000 s after t = 0, where the template's
    joint goal ends and SC_Alien climbs through the boundary."""
    boundary = round(_radius_at(GOAL_S), 3)
    config = _config(horizon_s=GOAL_S + DELTA_S, mission_boundary_km=boundary)
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + DELTA_S))
    readings = _by(evidence)
    assert evidence.verdict == "target"
    assert evidence.reason == "the template and the boundary support it"
    assert readings["target"].template_end_s == pytest.approx(GOAL_S + DELTA_S, abs=1e-6)
    assert readings["pursuer"].template_end_s == pytest.approx(GOAL_S, abs=1e-6)
    assert readings["target"].template_matches and not readings["pursuer"].template_matches
    assert readings["target"].boundary_crossing_s == pytest.approx(GOAL_S + DELTA_S, abs=1e-3)
    assert readings["pursuer"].boundary_matches is False
    assert readings["target"].alien_radius_km == pytest.approx(boundary, abs=1e-3)
    # Hermite between 60 s rows, the cubic's own error near perigee: 6.6 ms and 1.2 m here.
    assert evidence.perigee_s == pytest.approx(2000.0, abs=0.05)
    assert evidence.perigee_radius_km == pytest.approx(6827.0, abs=0.01)
    assert evidence.lines[-1].startswith(
        "time origin verdict: target (SC_Alien's epoch), the template and the boundary support "
        "it; policy.mission_time_origin is target."
    )
    # The lines a person reads, against the conic itself.
    track_line = _TRACK_LINE.match(evidence.lines[2])
    assert track_line is not None, evidence.lines[2]
    assert float(track_line["perigee"]) == pytest.approx(2000.0, abs=0.05)
    assert float(track_line["low"]) == pytest.approx(6827.0, abs=0.01)
    assert (float(track_line["t1"]), float(track_line["t2"])) == (GOAL_S, GOAL_S + DELTA_S)
    for index, time_s in (("1", GOAL_S), ("2", GOAL_S + DELTA_S)):
        assert float(track_line[f"r{index}"]) == pytest.approx(_radius_at(time_s), abs=1e-3)
        assert float(track_line[f"v{index}"]) == pytest.approx(_rate_at(time_s), abs=1e-4)
        assert (track_line[f"p{index}"], track_line[f"c{index}"]) == ("past", "climbing")
    boundary_line = re.match(
        r"^time origin, boundary: past perigee SC_Alien first reaches (\S+) km (\S+) s from "
        r"SC_Alien's epoch and (\S+) s from SC_Human's epoch; T_max 11400\.0 s matches the "
        r"target reading \(tolerance (\S+) s\)$",
        evidence.lines[3],
    )
    assert boundary_line is not None, evidence.lines[3]
    assert float(boundary_line[1]) == boundary
    assert float(boundary_line[2]) == pytest.approx(GOAL_S + DELTA_S, abs=1e-3)
    assert float(boundary_line[3]) == pytest.approx(GOAL_S, abs=1e-3)
    assert float(boundary_line[4]) == pytest.approx(0.5 + 1.0005 / _rate_at(GOAL_S), abs=1e-3)
    # T_max's 0.5 s, the radius's 0.5 m and 1 km over the climb rate, and 1 µs.
    assert readings["target"].boundary_tolerance_s == pytest.approx(
        0.5 + (0.0005 + 1.0) / _rate_at(GOAL_S) + 1.0e-6, abs=1.0e-7
    )


def test_the_same_template_read_from_the_pursuers_epoch() -> None:
    """Announced as 6000 s, T_max runs from SC_Human's epoch, and the setting is flagged."""
    boundary = round(_radius_at(GOAL_S), 3)
    config = _config(horizon_s=GOAL_S, mission_boundary_km=boundary)
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + 60.0))
    readings = _by(evidence)
    assert evidence.verdict == "pursuer"
    assert readings["pursuer"].template_matches and readings["pursuer"].boundary_matches
    # A goal of T_max itself, the later craft's reading, does not say which was meant.
    assert evidence.template_supports is None
    assert evidence.reason == "the boundary supports it"
    # 6000 s from SC_Alien's epoch is 600 s after t = 0, before its 2000 s perigee.
    assert readings["target"].refuted_because == "SC_Alien has not yet passed perigee there"
    assert readings["target"].after_perigee is False
    track_line = _TRACK_LINE.match(evidence.lines[2])
    assert track_line is not None, evidence.lines[2]
    assert (float(track_line["t1"]), float(track_line["t2"])) == (GOAL_S - DELTA_S, GOAL_S)
    assert (track_line["p1"], track_line["c1"]) == ("before", "not climbing")
    assert float(track_line["v1"]) == pytest.approx(_rate_at(GOAL_S - DELTA_S), abs=1e-4)
    assert float(track_line["v1"]) < 0.0
    assert "policy.mission_time_origin is target, which this evidence does not support; a " \
        "person decides, nothing was switched" in evidence.lines[-1]
    # Reported, never applied.
    assert evidence.configured == "target"


@pytest.mark.parametrize(
    ("horizon_s", "supported"), [(GOAL_S + DELTA_S, True), (GOAL_S, False)],
    ids=["from the earlier epoch", "from the later epoch"],
)
def test_earliest_reads_as_the_early_crafts_reading(horizon_s: float, supported: bool) -> None:
    """The organizer's origin is SC_Alien's here; a T_max from SC_Human's epoch is flagged."""
    boundary = round(_radius_at(GOAL_S), 3)
    config = _config(horizon_s=horizon_s, mission_boundary_km=boundary,
                     mission_time_origin="earliest")
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + DELTA_S))
    assert (evidence.configured, evidence.configured_reading) == ("earliest", "target")
    assert evidence.verdict == ("target" if supported else "pursuer")
    assert "count from the earlier epoch, the organizer answered on 2026-11-04" in (
        evidence.lines[0]
    )
    flagged = (
        "policy.mission_time_origin is earliest, the target reading, which this evidence does "
        "not support: T_max may have been entered from the other craft's epoch"
    )
    assert (flagged in evidence.lines[-1]) is not supported, evidence.lines[-1]
    if supported:
        assert "; policy.mission_time_origin is earliest, the target reading. Δ" in (
            evidence.lines[-1]
        )


def test_a_template_goal_unrelated_to_t_max_matches_neither(tmp_path: Path) -> None:
    """The preliminary template ran a fixed 12000 s, unrelated to the horizon."""
    script = _variant(tmp_path, "SC_Human.ElapsedSecs = 6000", "SC_Human.ElapsedSecs = 12000")
    boundary = round(_radius_at(GOAL_S), 3)
    config = _config(horizon_s=GOAL_S + DELTA_S, mission_boundary_km=boundary)
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + DELTA_S), script)
    assert not any(item.template_matches for item in evidence.readings)
    assert "T_max 11400.0 s matches neither" in evidence.lines[1]
    assert evidence.verdict == "target"
    assert evidence.reason == "the boundary supports it"


def test_without_a_boundary_the_template_decides() -> None:
    config = _config(horizon_s=GOAL_S + DELTA_S)
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + DELTA_S))
    assert evidence.verdict == "target"
    assert evidence.reason == "the template supports it"
    assert all(item.boundary_matches is None for item in evidence.readings)
    assert "policy.mission_boundary_km is not set" in evidence.lines[3]


def test_a_reading_before_perigee_is_eliminated(tmp_path: Path) -> None:
    script = _variant(tmp_path, "SC_Human.ElapsedSecs = 6000", "SC_Human.ElapsedSecs = 12000")
    config = _config(horizon_s=GOAL_S)
    evidence = _assess(config, _track(-DELTA_S, GOAL_S), script)
    assert evidence.verdict == "pursuer"
    assert evidence.reason == (
        "by elimination: for the target reading SC_Alien has not yet passed perigee there"
    )


@pytest.mark.parametrize("boundary_km", [7000.0, 1.0e6])
def test_a_boundary_the_track_never_climbs_through_matches_neither(boundary_km: float) -> None:
    """Already past 7000 km at SC_Alien's epoch, and nowhere near 1e6 km by T_max."""
    config = _config(horizon_s=GOAL_S + DELTA_S, mission_boundary_km=boundary_km)
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + DELTA_S, perigee_s=-8000.0))
    assert all(item.boundary_matches is False for item in evidence.readings)
    assert evidence.boundary_tolerance_s is None
    assert f"does not climb through {boundary_km!r} km" in evidence.lines[3]
    # A radius that fits neither reading confirms nothing the template says.
    assert evidence.verdict is None
    assert evidence.reason == "the template supports it, but the boundary matches neither reading"


def test_a_boundary_crossing_far_from_both_readings_overrules_the_template() -> None:
    """The radius announced in another sense, an altitude say: SC_Alien crosses it
    at neither reading's T_max."""
    config = _config(horizon_s=GOAL_S + DELTA_S, mission_boundary_km=round(_radius_at(4000.0), 3))
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + 2.0 * DELTA_S))
    readings = _by(evidence)
    assert readings["target"].template_matches
    assert not any(item.boundary_matches for item in evidence.readings)
    assert evidence.boundary_tolerance_s is not None
    assert evidence.verdict is None
    assert "T_max 11400.0 s matches neither (tolerance" in evidence.lines[3]


def test_a_boundary_inside_the_tolerance_still_vetoes(tmp_path: Path) -> None:
    """Δ 6 s and SC_Alien climbing slowly just past perigee: the boundary's tolerance
    is wider than Δ/2, so it picks nothing, but matching only the other reading, or
    neither, it still says the template's pick does not hold."""
    config = _config(horizon_s=GOAL_S + 6.0)
    scenario, document = _load(config)
    assert scenario.alignment is not None
    close = dataclasses.replace(
        scenario, alignment=dataclasses.replace(scenario.alignment, offset_s=6.0)
    )
    track = _track(-6.0, GOAL_S + 60.0, perigee_s=GOAL_S - 13.0, step_s=10.0)
    # 1 s after T_max from SC_Human's epoch, 7 s after SC_Alien's.
    other = dataclasses.replace(
        config.policy, mission_boundary_km=_radius_at(GOAL_S + 7.0, perigee_s=GOAL_S - 13.0)
    )
    evidence = assess_time_origin(close, document, other, track)
    readings = _by(evidence)
    assert evidence.boundary_tolerance_s is not None and evidence.boundary_tolerance_s > 3.0
    assert readings["target"].template_matches and not readings["pursuer"].template_matches
    assert readings["pursuer"].boundary_matches and not readings["target"].boundary_matches
    assert evidence.verdict is None
    assert evidence.reason == "the template supports it, but the boundary matches the other reading"
    neither = dataclasses.replace(
        config.policy, mission_boundary_km=_radius_at(GOAL_S + 30.0, perigee_s=GOAL_S - 13.0)
    )
    far = assess_time_origin(close, document, neither, track)
    assert not any(item.boundary_matches for item in far.readings)
    assert far.boundary_tolerance_s is not None and 2.0 * far.boundary_tolerance_s >= 6.0
    assert far.reason == "the template supports it, but the boundary matches neither reading"


def test_a_root_with_no_climb_rate_is_not_a_crossing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Radial rows rising to the radius and falling back. Bisection stops a hair short
    of the top, where the rate is still positive; one landing on it exactly has none
    to divide the tolerance by, and is refused."""
    times = np.array([0.0, 10.0, 20.0])
    track = Ephemeris(
        times,
        np.array([[9000.0, 0.0, 0.0], [9010.0, 0.0, 0.0], [9000.0, 0.0, 0.0]]),
        np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
    )
    # The cubic between the first two rows is 9000 + 20u - 10u^2.
    assert time_origin_module._crossing(track, 0.0, 9005.0) == pytest.approx(
        10.0 * (1.0 - 1.0 / math.sqrt(2.0)), abs=1e-9
    )
    monkeypatch.setattr(time_origin_module, "_bisect", lambda *_args: 10.0)
    assert time_origin_module._crossing(track, 0.0, 9010.0) is None


def test_neither_branch_without_a_pick_is_said_as_such(tmp_path: Path) -> None:
    script = _variant(tmp_path, "SC_Human.ElapsedSecs = 6000", "SC_Human.ElapsedSecs = 12000")
    config = _config(horizon_s=GOAL_S + DELTA_S)
    nothing = _assess(config, _track(-DELTA_S, GOAL_S + DELTA_S), script)
    assert not any(item.refuted_because for item in nothing.readings)
    assert nothing.verdict is None
    assert nothing.reason == "neither the template nor SC_Alien's track picks one"
    falling = _assess(config, _track(-DELTA_S, GOAL_S + DELTA_S, perigee_s=20000.0), script)
    assert falling.verdict is None
    assert falling.reason == "neither reading fits SC_Alien's track"


def test_an_alignment_we_add_leaves_the_goal_uncounted(tmp_path: Path) -> None:
    """The organizer's goal was written for a script with no alignment, which GMAT
    cannot run; where it ends against ours is our arithmetic, not their word."""
    script = _variant(
        tmp_path, "Propagate Prop(SC_Alien) {SC_Alien.UTCModJulian = SC_Human.UTCModJulian};\n", ""
    )
    plain = _config(horizon_s=GOAL_S + DELTA_S, add_epoch_alignment=True)
    scenario, _ = _load(plain, script)
    assert scenario.alignment is not None and scenario.alignment.line_index is None
    evidence = _assess(plain, _track(-DELTA_S, GOAL_S + 2.0 * DELTA_S), script)
    assert evidence.template_counted is False
    assert _by(evidence)["target"].template_matches
    assert "Not counted: the template has no alignment" in evidence.lines[1]
    assert evidence.verdict is None
    assert evidence.reason == "neither the template nor SC_Alien's track picks one"
    bounded = _config(
        horizon_s=GOAL_S + DELTA_S, add_epoch_alignment=True,
        mission_boundary_km=round(_radius_at(GOAL_S), 3),
    )
    with_radius = _assess(bounded, _track(-DELTA_S, GOAL_S + 2.0 * DELTA_S), script)
    assert with_radius.verdict == "target"
    assert with_radius.reason == "the boundary supports it"


def test_a_goal_that_never_ends_is_refused(tmp_path: Path) -> None:
    """The parser reads 1e400 as inf; inspect stops, and solve before its output exists."""
    script = _variant(tmp_path, "SC_Human.ElapsedSecs = 6000", "SC_Human.ElapsedSecs = 1e400")
    config = _config(horizon_s=GOAL_S + DELTA_S)
    with pytest.raises(ParseError, match="end at inf"):
        _assess(config, None, script)
    with pytest.raises(ParseError, match="end at inf"):
        solve_scenario(script, config, tmp_path / "solve", gmat_console=FAKE, startup_file=STARTUP)
    assert not (tmp_path / "solve").exists()


def test_template_and_boundary_disagreeing_is_undetermined() -> None:
    """The template ends T_max from SC_Alien's epoch, the boundary from SC_Human's."""
    horizon = GOAL_S + DELTA_S
    config = _config(horizon_s=horizon, mission_boundary_km=round(_radius_at(horizon), 3))
    evidence = _assess(config, _track(-DELTA_S, horizon + 60.0))
    readings = _by(evidence)
    assert readings["target"].template_matches and readings["pursuer"].boundary_matches
    assert evidence.verdict is None
    assert evidence.reason == "the template and the boundary support different readings"
    assert evidence.lines[-1].startswith("time origin verdict: undetermined,")


def test_neither_reading_fitting_the_track_is_undetermined() -> None:
    """SC_Alien still falling at both ends: the T_max announced is not its climb-out."""
    config = _config(horizon_s=GOAL_S + DELTA_S)
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + DELTA_S, perigee_s=20000.0))
    assert all(item.refuted_because for item in evidence.readings)
    assert evidence.verdict is None
    assert evidence.reason == "the template supports it, but SC_Alien has not yet passed " \
        "perigee there"
    assert evidence.outbound_from_start is False
    assert "does not pass perigee within the track" in evidence.lines[2]


def test_an_alien_already_climbing_at_its_epoch_is_past_perigee() -> None:
    config = _config(horizon_s=GOAL_S + DELTA_S)
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + DELTA_S, perigee_s=-8000.0))
    assert evidence.outbound_from_start is True and evidence.perigee_s is None
    assert all(item.after_perigee for item in evidence.readings)
    assert evidence.verdict == "target"
    assert "already climbing at its epoch" in evidence.lines[2]


def test_a_head_start_inside_the_tolerance_cannot_be_told_apart() -> None:
    """Goal and horizon written to whole seconds: 1 s of tolerance, wider than half of Δ."""
    config = _config(horizon_s=GOAL_S + 1.0)
    scenario, document = _load(config)
    assert scenario.alignment is not None
    close = dataclasses.replace(
        scenario, alignment=dataclasses.replace(scenario.alignment, offset_s=0.8)
    )
    evidence = assess_time_origin(close, document, config.policy, _track(-0.8, GOAL_S + 1.0))
    assert all(item.template_matches for item in evidence.readings)
    assert (
        "matches both (not counted: the numbers as written cannot tell readings Δ apart)"
        in evidence.lines[1]
    )
    assert evidence.verdict is None
    # Neither within the tolerance is still said as neither.
    far = _config(horizon_s=9000.0)
    unrelated = assess_time_origin(close, document, far.policy, _track(-0.8, 9000.0))
    assert not any(item.template_matches for item in unrelated.readings)
    assert "matches neither (tolerance" in unrelated.lines[1]


def test_t_max_written_to_the_second_cannot_pick_readings_under_a_second_apart(
    tmp_path: Path,
) -> None:
    """The goal is exact to the millisecond, but a T_max of 6001 s is known only to half
    a second either way: a window of about 1 s, wider than readings 0.9 s apart."""
    script = _variant(tmp_path, "SC_Human.ElapsedSecs = 6000", "SC_Human.ElapsedSecs = 6000.000")
    config = _config(horizon_s=GOAL_S + 1.0)
    scenario, document = _load(config, script)
    assert scenario.alignment is not None
    close = dataclasses.replace(
        scenario, alignment=dataclasses.replace(scenario.alignment, offset_s=0.9)
    )
    evidence = assess_time_origin(close, document, config.policy, _track(-0.9, GOAL_S + 2.0))
    readings = _by(evidence)
    assert readings["target"].template_matches and not readings["pursuer"].template_matches
    assert evidence.template_supports is None
    assert evidence.verdict is None


def test_a_goal_written_coarser_than_the_head_start_cannot_pick(tmp_path: Path) -> None:
    """6.0e3 is written to the hundred seconds. Held to a second it matches only one
    reading 100 s apart, but as written it fits both."""
    script = _variant(tmp_path, "SC_Human.ElapsedSecs = 6000", "SC_Human.ElapsedSecs = 6.0e3")
    config = _config(horizon_s=GOAL_S + 100.0)
    scenario, document = _load(config, script)
    assert scenario.alignment is not None
    close = dataclasses.replace(
        scenario, alignment=dataclasses.replace(scenario.alignment, offset_s=100.0)
    )
    evidence = assess_time_origin(close, document, config.policy, _track(-100.0, GOAL_S + 200.0))
    readings = _by(evidence)
    assert readings["target"].template_matches and not readings["pursuer"].template_matches
    assert evidence.template_supports is None
    assert "the target reading (not counted: the numbers as written" in evidence.lines[1]
    assert evidence.verdict is None


def test_two_goals_on_one_line_take_the_coarser_rounding(tmp_path: Path) -> None:
    script = _variant(
        tmp_path, "{SC_Human.ElapsedSecs = 6000}",
        "{SC_Alien.ElapsedSecs = 6000.0, SC_Human.ElapsedSecs = 6000}",
    )
    config = _config(horizon_s=GOAL_S + DELTA_S)
    evidence = _assess(config, None, script)
    assert evidence.template_tolerance_s == pytest.approx(0.5 + 0.5 + 1.0e-6)


def test_a_boundary_matching_neither_is_named_when_nothing_picks(tmp_path: Path) -> None:
    script = _variant(tmp_path, "SC_Human.ElapsedSecs = 6000", "SC_Human.ElapsedSecs = 12000")
    config = _config(horizon_s=GOAL_S + DELTA_S, mission_boundary_km=round(_radius_at(4000.0), 3))
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + 2.0 * DELTA_S), script)
    assert evidence.verdict is None
    assert evidence.reason == "the boundary matches neither reading, and nothing else picks one"


def test_a_template_of_several_segments_sums_their_goals_and_roundings(tmp_path: Path) -> None:
    """A Maneuver right after the alignment (the implicit segment), a goal in days
    and one in seconds: the end is their sum, the tolerance each one's rounding."""
    burns = "".join(
        f"Create ImpulsiveBurn {name};\n{name}.CoordinateSystem = Local;\n{name}.Origin = Earth;\n"
        f"{name}.Axes = VNB;\n{name}.Element1 = 0;\n{name}.Element2 = 0;\n{name}.Element3 = 0;\n"
        f"{name}.DecrementMass = false;\n{name}.Isp = 300;\n{name}.GravitationalAccel = 9.81;\n\n"
        for name in ("B1", "B2")
    )
    script = _variant(tmp_path, "BeginMissionSequence;", burns + "BeginMissionSequence;")
    script = _variant(
        tmp_path, "Propagate Prop(SC_Alien, SC_Human) {SC_Human.ElapsedSecs = 6000};",
        "Maneuver B1(SC_Human);\n"
        "Propagate Prop(SC_Alien, SC_Human) {SC_Human.ElapsedDays = 0.0347222};\n"
        "Maneuver B2(SC_Human);\n"
        "Propagate Prop(SC_Alien, SC_Human) {SC_Human.ElapsedSecs = 3000.5};",
        source=script,
    )
    goal = 0.0347222 * 86400.0 + 3000.5
    config = _config(horizon_s=round(goal + DELTA_S, 3))
    scenario, document = _load(config, script)
    assert [segment.implicit for segment in scenario.propagate_segments] == [True, False, False]
    evidence = assess_time_origin(scenario, document, config.policy, None, "not flown here")
    assert evidence.template_goal_s == pytest.approx(goal, abs=1e-9)
    # 0.5e-7 d, 0.05 s, T_max's 0.0005 s and 1 µs for Δ.
    assert evidence.template_tolerance_s == pytest.approx(0.00432 + 0.05 + 0.0005 + 1e-6)
    assert _by(evidence)["target"].template_matches
    assert evidence.verdict == "target"


# --- SC_Human first ----------------------------------------------------------------------


def test_with_the_pursuer_first_the_late_target_reads_the_goal_as_is(tmp_path: Path) -> None:
    script = _pursuer_early(tmp_path)
    boundary = round(_radius_at(GOAL_S), 3)
    evidence = _assess(
        _config(horizon_s=GOAL_S, mission_boundary_km=boundary), _track(0.0, GOAL_S + 60.0),
        script,
    )
    readings = _by(evidence)
    assert evidence.early == "SC_Human"
    assert readings["target"].offset_s == 0.0 and readings["pursuer"].offset_s == DELTA_S
    assert evidence.verdict == "target"
    # SC_Alien is the later craft here, so its goal of T_max itself is not counted.
    assert evidence.reason == "the boundary supports it"
    pursuer_first = _assess(
        _config(
            horizon_s=GOAL_S + DELTA_S, mission_boundary_km=boundary, mission_time_origin="pursuer"
        ),
        _track(0.0, GOAL_S + DELTA_S), script,
    )
    assert pursuer_first.verdict == "pursuer"
    assert pursuer_first.reason == "the template and the boundary support it"
    assert "; policy.mission_time_origin is pursuer. Δ 5400.000000 s;" in pursuer_first.lines[-1]


def test_a_goal_of_t_max_itself_does_not_pick_the_later_epoch() -> None:
    """An author who counts from SC_Alien's epoch and writes T_max into the joint goal,
    not knowing it counts from that Propagate, leaves the same template as one who
    counts from SC_Human's. SC_Alien climbing at both ends, nothing else picks."""
    config = _config(horizon_s=GOAL_S)
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + DELTA_S, perigee_s=-3000.0))
    assert _by(evidence)["pursuer"].template_matches
    assert evidence.template_supports is None
    assert not any(item.refuted_because for item in evidence.readings)
    assert evidence.verdict is None
    assert (
        "Not counted: a goal of T_max itself also comes from a template that counts from "
        "SC_Alien's epoch and did not take the head start off" in evidence.lines[1]
    )


def _ellipse(perigee_s: float, time_s: float) -> StateVector:
    """A closed SC_Alien, perigee 7000 km and apogee 30000 km, period about 25042 s."""
    sma = 18500.0
    speed = math.sqrt(MU * (2.0 / 7000.0 - 1.0 / sma))
    return propagate_universal(
        StateVector((7000.0, 0.0, 0.0), (0.0, speed, 0.0)), time_s - perigee_s, MU
    )


def _to_boundary_s(boundary_km: float = 17000.0) -> tuple[float, float]:
    """(time from perigee to the boundary, period) on _ellipse's orbit."""
    sma, ecc = 18500.0, 23000.0 / 37000.0
    eccentric = math.acos((1.0 - boundary_km / sma) / ecc)
    mean_motion = math.sqrt(MU / sma**3)
    return (eccentric - ecc * math.sin(eccentric)) / mean_motion, 2.0 * math.pi / mean_motion


def _ellipse_track(perigee_s: float, start_s: float, end_s: float) -> Ephemeris:
    times = np.unique(np.append(np.arange(start_s, end_s, 20.0), end_s))
    states = [_ellipse(perigee_s, time_s) for time_s in times]
    return Ephemeris(times, np.array([s.r for s in states]), np.array([s.v for s in states]))


def test_a_reading_that_starts_beyond_the_boundary_waits_for_the_next_climb_out() -> None:
    """SC_Human's epoch finds a closed SC_Alien already out past 17000 km and climbing:
    its T_max is the climb out after the next perigee, one period on."""
    to_boundary_s, period_s = _to_boundary_s()
    horizon = round(-5000.0 + period_s + to_boundary_s, 3)
    config = _config(horizon_s=horizon, mission_boundary_km=17000.0, mission_time_origin="pursuer")
    track = _ellipse_track(-5000.0, -DELTA_S, horizon + DELTA_S)
    evidence = _assess(config, track)
    readings = _by(evidence)
    assert readings["pursuer"].boundary_crossing_s == pytest.approx(horizon, abs=1e-2)
    assert readings["target"].boundary_crossing_s == pytest.approx(
        -5000.0 + to_boundary_s + DELTA_S, abs=1e-2
    )
    assert readings["target"].refuted_because == "SC_Alien is not climbing there"
    assert evidence.verdict == "pursuer"
    assert evidence.reason == "the boundary supports it"
    # Two crossings, two climb rates: the line prints the wider tolerance.
    tolerances = [item.boundary_tolerance_s for item in evidence.readings]
    assert tolerances[0] != tolerances[1]
    assert evidence.boundary_tolerance_s == max(tolerances)


def test_a_reading_ten_seconds_before_perigee_has_not_passed_it() -> None:
    config = _config(horizon_s=GOAL_S + DELTA_S)
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + 2.0 * DELTA_S, perigee_s=GOAL_S + 10.0))
    target = _by(evidence)["target"]
    assert target.after_perigee is False
    assert target.refuted_because == "SC_Alien has not yet passed perigee there"


def test_each_reading_finds_its_own_climb_out() -> None:
    """Closed and with a period under Δ, SC_Alien passes perigee and the boundary
    during the alignment too; the later epoch's T_max is its first climb out after it."""
    to_boundary_s, period_s = _to_boundary_s()
    perigee_s = 325.0 - to_boundary_s
    delta_s, horizon = 30000.0, 325.0
    config = _config(horizon_s=horizon, mission_boundary_km=17000.0, mission_time_origin="pursuer")
    scenario, document = _load(config)
    assert scenario.alignment is not None
    far = dataclasses.replace(
        scenario, alignment=dataclasses.replace(scenario.alignment, offset_s=delta_s)
    )
    track = _ellipse_track(perigee_s, -delta_s, horizon + delta_s)
    evidence = assess_time_origin(far, document, config.policy, track)
    readings = _by(evidence)
    assert readings["pursuer"].boundary_crossing_s == pytest.approx(horizon, abs=1e-3)
    assert readings["pursuer"].boundary_matches
    # From SC_Alien's epoch the first climb out is one period earlier.
    assert readings["target"].boundary_crossing_s == pytest.approx(
        horizon - period_s + delta_s, abs=1e-2
    )
    assert evidence.verdict == "pursuer"
    assert evidence.reason == "the boundary supports it"


def test_t_max_from_an_early_pursuer_can_end_before_the_alien_exists(tmp_path: Path) -> None:
    script = _pursuer_early(tmp_path)
    evidence = _assess(_config(horizon_s=3000.0), _track(0.0, 3000.0), script)
    readings = _by(evidence)
    assert readings["pursuer"].refuted_because == (
        "T_max from SC_Human's epoch ends before SC_Alien's epoch, before the two fly together"
    )
    assert readings["pursuer"].alien_radius_km is None
    assert evidence.verdict == "target"
    assert evidence.reason.startswith("by elimination: for the pursuer reading")


def test_t_max_from_an_early_alien_can_end_before_the_two_fly_together(tmp_path: Path) -> None:
    """The mirror case: SC_Alien exists then, but the parser would refuse that origin."""
    script = _variant(tmp_path, "SC_Human.ElapsedSecs = 6000", "SC_Human.ElapsedSecs = 12000")
    boundary = round(_radius_at(-3000.0, perigee_s=-4000.0), 3)
    config = _config(horizon_s=2400.0, mission_boundary_km=boundary, mission_time_origin="pursuer")
    evidence = _assess(config, _track(-DELTA_S, 2400.0 + DELTA_S, perigee_s=-4000.0), script)
    readings = _by(evidence)
    # SC_Alien crosses the radius 2400 s after its epoch, but that is before SC_Human's.
    assert readings["target"].boundary_matches
    assert readings["target"].refuted_because == (
        "T_max from SC_Alien's epoch ends before SC_Human's epoch, before the two fly together"
    )
    assert evidence.verdict is None
    assert evidence.reason.startswith("the boundary supports it, but T_max from SC_Alien's")


def test_an_elimination_is_vetoed_by_a_boundary_matching_neither(tmp_path: Path) -> None:
    script = _variant(tmp_path, "SC_Human.ElapsedSecs = 6000", "SC_Human.ElapsedSecs = 12000")
    config = _config(horizon_s=GOAL_S, mission_boundary_km=round(_radius_at(6300.0), 3))
    evidence = _assess(config, _track(-DELTA_S, GOAL_S + DELTA_S), script)
    assert _by(evidence)["target"].refuted_because is not None
    assert not any(item.boundary_matches for item in evidence.readings)
    assert evidence.verdict is None
    assert evidence.reason == (
        "by elimination: for the target reading SC_Alien has not yet passed perigee there, but "
        "the boundary matches neither reading"
    )


# --- what cannot be read is not read -----------------------------------------------------


def test_a_track_that_does_not_start_at_the_aliens_epoch_is_refused() -> None:
    config = _config(horizon_s=GOAL_S + DELTA_S)
    with pytest.raises(GmatRunError, match="not at its epoch"):
        _assess(config, _track(0.0, GOAL_S + DELTA_S))


def test_a_track_short_of_t_max_is_refused() -> None:
    config = _config(horizon_s=GOAL_S + DELTA_S)
    with pytest.raises(GmatRunError, match="short of T_max"):
        _assess(config, _track(-DELTA_S, GOAL_S + DELTA_S - 10.0))


@pytest.mark.parametrize(
    ("start_s", "end_s", "refused"),
    [
        # 1e-5 s: about 30 A1 steps at its epoch, where GMAT's alignment ends.
        (-DELTA_S + 5.0e-6, GOAL_S + DELTA_S, None),
        (-DELTA_S + 2.0e-5, GOAL_S + DELTA_S, "not at its epoch"),
        (-DELTA_S - 2.0e-5, GOAL_S + DELTA_S, "not at its epoch"),
        # 1 µs short of T_max, GMAT's own stop; further is not T_max.
        (-DELTA_S, GOAL_S + DELTA_S - 5.0e-7, None),
        (-DELTA_S, GOAL_S + DELTA_S - 2.0e-6, "short of T_max"),
    ],
)
def test_the_track_ends_are_held_to_their_tolerances(
    start_s: float, end_s: float, refused: str | None
) -> None:
    config = _config(horizon_s=GOAL_S + DELTA_S)
    if refused is None:
        assert _assess(config, _track(start_s, end_s)).verdict == "target"
    else:
        with pytest.raises(GmatRunError, match=refused):
            _assess(config, _track(start_s, end_s))


def test_an_unreadable_track_leaves_the_template_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A boundary set but not read confirms nothing and vetoes nothing; the line says so."""
    config = _config(horizon_s=GOAL_S + DELTA_S, mission_boundary_km=17000.0)
    scenario, document = _load(config)
    monkeypatch.setattr(
        time_origin_module, "_alien_track", lambda *_args: _track(0.0, GOAL_S + DELTA_S)
    )
    evidence = time_origin_evidence(scenario, document, config, FAKE, STARTUP)
    assert evidence is not None
    assert evidence.track_error is not None and "not at its epoch" in evidence.track_error
    assert evidence.lines[2].startswith("time origin, track: not read:")
    assert evidence.verdict == "target"
    assert evidence.reason == "the template supports it; SC_Alien's track was not read"

    def fails(*_args: object) -> Ephemeris:
        raise GmatRunError("GMAT coast run failed")

    monkeypatch.setattr(time_origin_module, "_alien_track", fails)
    failed = time_origin_evidence(scenario, document, config, FAKE, STARTUP)
    assert failed is not None and failed.track_error == "GmatRunError: GMAT coast run failed"
    assert all(item.alien_radius_km is None for item in failed.readings)


def test_a_t_max_that_ends_before_the_later_epoch_needs_no_track() -> None:
    """Arithmetic alone rules it out, with or without SC_Alien's track."""
    config = _config(horizon_s=2400.0, mission_time_origin="pursuer")
    scenario, document = _load(config)
    evidence = assess_time_origin(scenario, document, config.policy, None, "coast failed")
    assert _by(evidence)["target"].refuted_because == (
        "T_max from SC_Alien's epoch ends before SC_Human's epoch, before the two fly together"
    )
    assert evidence.verdict == "pursuer"
    assert evidence.reason.startswith("by elimination: for the target reading")
    assert evidence.reason.endswith("; SC_Alien's track was not read")


def test_a_t_max_of_exactly_the_head_start_leaves_nothing_to_fly() -> None:
    config = _config(horizon_s=DELTA_S, mission_time_origin="pursuer")
    scenario, document = _load(config)
    evidence = assess_time_origin(scenario, document, config.policy, None, "not flown here")
    assert _by(evidence)["target"].instant_s == 0.0
    assert _by(evidence)["target"].refuted_because is not None


@pytest.mark.parametrize(
    "error", [OSError("No space left on device"), ParseError("refused"), GmatRunError("rc 1")]
)
def test_any_coast_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fails(*_args: object) -> Ephemeris:
        raise error

    monkeypatch.setattr(time_origin_module, "_alien_track", fails)
    config = _config(horizon_s=GOAL_S + DELTA_S)
    scenario, document = _load(config)
    evidence = time_origin_evidence(scenario, document, config, FAKE, STARTUP)
    assert evidence is not None
    assert evidence.track_error == f"{type(error).__name__}: {error}"


def test_a_track_given_with_an_error_is_not_read() -> None:
    config = _config(horizon_s=GOAL_S + DELTA_S, mission_boundary_km=round(_radius_at(4000.0), 3))
    scenario, document = _load(config)
    track = _track(-DELTA_S, GOAL_S + 2.0 * DELTA_S)
    evidence = assess_time_origin(scenario, document, config.policy, track, "coast failed")
    assert all(item.alien_radius_km is None for item in evidence.readings)
    assert all(item.boundary_matches is None for item in evidence.readings)
    assert evidence.lines[2] == "time origin, track: not read: coast failed"


def test_no_track_and_no_reason_still_says_it_was_not_read() -> None:
    config = _config(horizon_s=GOAL_S + DELTA_S)
    scenario, document = _load(config)
    evidence = assess_time_origin(scenario, document, config.policy, None)
    assert evidence.track_error == "no track was given"
    assert evidence.lines[2] == "time origin, track: not read: no track was given"


def test_without_an_announced_t_max_nothing_is_compared() -> None:
    config = _config()
    scenario, document = _load(config)
    evidence = time_origin_evidence(scenario, document, config, FAKE, STARTUP)
    assert evidence is not None
    assert evidence.verdict is None
    assert all(item.template_matches is None for item in evidence.readings)
    assert "policy.horizon_s is not set" in evidence.lines[1]
    assert "T_max not set;" in evidence.lines[-1]
    json.dumps(evidence.to_dict(), allow_nan=False)


def test_craft_that_start_together_have_no_question(workshop_script: Path) -> None:
    config = OrbitWinConfig()
    scenario, document = parse_gmat_script(workshop_script, config)
    assert time_origin_evidence(scenario, document, config, None, None) is None
    with pytest.raises(ParseError, match="start together"):
        assess_time_origin(scenario, document, config.policy, None)


@pytest.mark.parametrize(
    ("text", "unit", "half_s"),
    [
        ("6711.667", 1.0, 5.0e-4),
        ("12000", 1.0, 0.5),
        # Coarser than a second is held to a second: it can only miss a match.
        ("1.2e4", 1.0, 0.5),
        ("0.0776805", 86400.0, 0.5 * 1.0e-7 * 86400.0),
        ("0.0776", 86400.0, 0.5),
    ],
)
def test_written_half_step(text: str, unit: float, half_s: float) -> None:
    assert written_half_step(text, unit, 1.0) == pytest.approx(half_s)


@pytest.mark.parametrize("text", ["abc", "nan", "inf", ""])
def test_a_goal_that_is_not_a_finite_decimal_is_refused(text: str) -> None:
    with pytest.raises(ParseError):
        written_half_step(text, 1.0, 1.0)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (6000.0, "inside the"),
        (6378.1363, "inside the"),
        (float("nan"), "finite"),
        ("17000", "must be a number"),
    ],
)
def test_the_boundary_must_be_a_radius_outside_the_body(value: object, match: str) -> None:
    from orbitwin.config import validate_config

    with pytest.raises(ConfigurationError, match=match):
        validate_config(_config(mission_boundary_km=value))


# --- the plumbing, on the fake console ---------------------------------------------------


def test_the_coast_keeps_the_alignment_rows_on_request(tmp_path: Path) -> None:
    config = _config(horizon_s=GOAL_S + DELTA_S, mission_time_origin="pursuer")
    scenario, document = _load(config)
    kept = target_ephemeris_from_gmat(
        scenario, document, config, FAKE, STARTUP, tmp_path / "kept", alignment_rows=True
    )
    dropped = target_ephemeris_from_gmat(scenario, document, config, FAKE, STARTUP, tmp_path / "d")
    assert kept.times_s[0] == pytest.approx(-DELTA_S, abs=1e-6)
    assert dropped.times_s[0] == 0.0
    assert kept.times_s[-1] == dropped.times_s[-1] == pytest.approx(GOAL_S + DELTA_S, abs=1e-6)


@pytest.mark.parametrize("early", ["SC_Alien", "SC_Human"])
@pytest.mark.parametrize("origin", ["target", "pursuer"])
def test_the_coast_flies_past_t_max_from_the_later_epoch(
    tmp_path: Path, origin: str, early: str
) -> None:
    """With the origin the early craft the solver flies T_max less Δ; the evidence needs
    all of it from SC_Alien's epoch on, and Δ/2 more, where a crossing that picks can lie."""
    script = CROSSING if early == "SC_Alien" else _pursuer_early(tmp_path)
    config = _config(horizon_s=GOAL_S + DELTA_S, mission_time_origin=origin)
    scenario, document = _load(config, script)
    track = time_origin_module._alien_track(
        scenario, document, config, FAKE, STARTUP, tmp_path / "coast"
    )
    assert track.times_s[0] == pytest.approx(-DELTA_S if early == "SC_Alien" else 0.0, abs=1e-6)
    assert track.times_s[-1] == pytest.approx(GOAL_S + 1.5 * DELTA_S, abs=1e-6)


def test_a_crossing_just_after_the_announced_t_max_is_inside_the_track() -> None:
    """Measured on R8: T_max 6711.667 s announced, GMAT's SC_Alien at the boundary
    0.2 ms later, after the end of a coast that stopped at T_max."""
    horizon = GOAL_S + DELTA_S
    boundary = _radius_at(GOAL_S + 2.0e-4)
    config = _config(horizon_s=horizon, mission_boundary_km=boundary)
    stops_at_t_max = _assess(config, _track(-DELTA_S, horizon))
    assert stops_at_t_max.verdict == "target"
    assert _by(stops_at_t_max)["target"].boundary_matches is True
    pursuer_late = _assess(
        _config(horizon_s=GOAL_S, mission_boundary_km=boundary), _track(-DELTA_S, GOAL_S)
    )
    # Ending at T_max cuts the crossing off; Δ more keeps it.
    assert _by(pursuer_late)["pursuer"].boundary_matches is False
    kept = _assess(
        _config(horizon_s=GOAL_S, mission_boundary_km=boundary),
        _track(-DELTA_S, GOAL_S + DELTA_S),
    )
    assert _by(kept)["pursuer"].boundary_matches is True
    assert kept.verdict == "pursuer" and kept.reason == "the boundary supports it"


def _day_config(tmp_path: Path, **policy: object) -> Path:
    raw = {
        "pursuer_name": "SC_Human", "target_name": "SC_Alien",
        "policy": {"mission_time_origin": "target", "closest_approach_window": "after_alignment",
                   **policy},
    }
    path = tmp_path / "day.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    load_config(path)
    return path


def test_inspect_prints_the_evidence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _day_config(tmp_path, horizon_s=GOAL_S + DELTA_S, mission_boundary_km=9000.0)
    assert main([
        "inspect", str(CROSSING), "--config", str(config), "--rules", "preliminary",
        "--gmat-console", str(FAKE), "--startup-file", str(STARTUP),
    ]) == 0
    captured = capsys.readouterr()
    evidence = json.loads(captured.out)["time_origin_evidence"]
    assert evidence["offset_s"] == pytest.approx(DELTA_S, abs=1e-6)
    assert evidence["horizon_s"] == GOAL_S + DELTA_S
    printed = [line for line in captured.err.splitlines() if "time origin" in line]
    assert printed == [f"orbitwin inspect: {line}" for line in evidence["lines"]]
    assert len(printed) == 5
    # The fake's SC_Alien stands still at 7000 km, so it never climbs to 9000 km.
    assert printed[-1].startswith(
        "orbitwin inspect: time origin verdict: undetermined, the template supports it, but "
        "SC_Alien is not climbing there; policy.mission_time_origin is target. Δ 5400.000000 s; "
        "T_max 11400.0 s; target (SC_Alien): template end 11400.0000 s, r(T_max) 7000.000 km;"
    )
    assert evidence["template_counted"] is True
    assert [item["template_matches"] for item in evidence["readings"]] == [True, False]


def test_inspect_of_craft_that_start_together_prints_none(
    workshop_script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect", str(workshop_script), "--rules", "preliminary"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["time_origin_evidence"] is None
    assert "time origin" not in captured.err


@pytest.mark.parametrize(
    ("perigee_s", "origin", "verdict"),
    [
        # SC_Alien past perigee and climbing from its epoch on: only the template speaks.
        (-8000.0, "pursuer", "target"),
        # The same, set as the evidence reads it: nothing to warn about.
        (-8000.0, "target", "target"),
        # The organizer's origin: SC_Alien starts first, so it is the target reading.
        (-8000.0, "earliest", "target"),
        # Still falling at both readings' T_max: the template's pick is ruled out.
        (20000.0, "pursuer", None),
        # No track: the template alone, and a warning that the track was not read.
        (None, "pursuer", "target"),
    ],
)
def test_solve_carries_the_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, perigee_s: float | None, origin: str,
    verdict: str | None,
) -> None:
    base = quick_config()
    config = dataclasses.replace(
        base,
        pursuer_name="SC_Human", target_name="SC_Alien",
        policy=dataclasses.replace(
            base.policy, mission_time_origin=origin, closest_approach_window="after_alignment",
            horizon_s=GOAL_S + DELTA_S,
        ),
    )

    def track(*_args: object) -> Ephemeris:
        if perigee_s is None:
            raise GmatRunError("GMAT coast run failed")
        return _track(-DELTA_S, GOAL_S + 2.0 * DELTA_S, perigee_s=perigee_s)

    monkeypatch.setattr(time_origin_module, "_alien_track", track)
    summary = solve_scenario(
        CROSSING, config, tmp_path / "solve", gmat_console=FAKE, startup_file=STARTUP,
        gmat_verify_top=0,
    )
    evidence = summary["time_origin_evidence"]
    assert evidence["verdict"] == verdict
    # Warned whenever the evidence does not back the origin that was set.
    reading = "target" if origin == "earliest" else origin
    assert (evidence["lines"][-1] in summary["warnings"]) is (verdict != reading)
    unread = "time origin, track: not read: GmatRunError: GMAT coast run failed"
    assert (unread in summary["warnings"]) is (perigee_s is None)
    written = json.loads((tmp_path / "solve" / "summary.json").read_text(encoding="utf-8"))
    assert written["time_origin_evidence"] == json.loads(json.dumps(evidence))


def test_solve_prints_the_evidence_after_its_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        time_origin_module, "_alien_track",
        lambda *_args: _track(-DELTA_S, GOAL_S + 2.0 * DELTA_S, perigee_s=-8000.0),
    )
    search = dataclasses.asdict(quick_config().search)
    raw = {
        "pursuer_name": "SC_Human", "target_name": "SC_Alien", "search": search,
        "policy": {"mission_time_origin": "target", "closest_approach_window": "after_alignment",
                   "horizon_s": GOAL_S + DELTA_S},
    }
    config = tmp_path / "day.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert main([
        "solve", str(CROSSING), "--config", str(config), "--output", str(tmp_path / "solve"),
        "--gmat-console", str(FAKE), "--startup-file", str(STARTUP), "--gmat-verify-top", "0",
        "--rules", "preliminary",
    ]) == 0
    captured = capsys.readouterr()
    evidence = json.loads(captured.out)["time_origin_evidence"]
    printed = [line for line in captured.err.splitlines() if "time origin" in line]
    assert printed == [f"orbitwin solve: {line}" for line in evidence["lines"]]
    assert evidence["verdict"] == "target"


def test_solve_without_an_announced_t_max_warns_once(tmp_path: Path) -> None:
    """Nothing to compare: the verdict is undetermined, which is warned; the track is not
    read for want of a T_max, not for a failure, which is not."""
    base = quick_config()
    config = dataclasses.replace(
        base,
        pursuer_name="SC_Human", target_name="SC_Alien",
        policy=dataclasses.replace(
            base.policy, mission_time_origin="target", closest_approach_window="after_alignment"
        ),
    )
    summary = solve_scenario(
        CROSSING, config, tmp_path / "solve", gmat_console=FAKE, startup_file=STARTUP,
        gmat_verify_top=0,
    )
    evidence = summary["time_origin_evidence"]
    assert evidence["verdict"] is None
    origin_warnings = [line for line in summary["warnings"] if line.startswith("time origin")]
    assert origin_warnings == [evidence["lines"][-1]]
