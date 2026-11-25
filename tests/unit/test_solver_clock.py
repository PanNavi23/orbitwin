"""One solver clock: t = 0 at the alignment's end, T_team from mission_time_origin.

See docs/ASYNC_START_AND_INSERTION.md §3.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from conftest import FAKE_STARTUP, TEAM_NUMBER, with_ranking_report

from orbitwin.config import OrbitWinConfig, PolicyConfig, ScoreConfig
from orbitwin.errors import OrbitWinError, ParseError, PolicyError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import AlignedStates, Candidate, StateVector, clock_stamps
from orbitwin.multi_burn import MultiBurnLimits
from orbitwin.ranking_search import RankingLimits
from orbitwin.score import on_solver_clock, score_candidate
from orbitwin.state_probe import load_scenario

ROOT = Path(__file__).resolve().parents[2]
CROSSING = ROOT / "tests" / "fixtures" / "async" / "aligned_crossing.script"
FAKE = ROOT / "scripts" / "fake_gmat_console.py"
STARTUP = FAKE_STARTUP
SCORE = ScoreConfig(kt=0.003, ct=6000.0, kv=1.2, cv=2.0, delta_v_unit="km/s")
#: SCORE as the ranking rules revised 2026/11/04 take it: m/s, and a C_r.
RANKING_SCORE = ScoreConfig(kt=0.003, ct=6000.0, kv=0.0012, cv=2000.0, delta_v_unit="m/s",
                            cr_km=3000.0)


def _config(**policy: object) -> OrbitWinConfig:
    settings: dict[str, object] = {
        "mission_time_origin": "target", "closest_approach_window": "after_alignment",
    }
    settings.update(policy)
    return OrbitWinConfig(
        policy=dataclasses.replace(PolicyConfig(), **settings),  # type: ignore[arg-type]
        pursuer_name="SC_Human",
        target_name="SC_Alien",
        score=RANKING_SCORE if settings.get("rules") == "ranking" else SCORE,
    )


def _load(config: OrbitWinConfig, script: Path = CROSSING):
    return load_scenario(script, config, gmat_console=FAKE, startup_file=STARTUP)


def _hit(at_s: float) -> Candidate:
    return Candidate(
        burns=(), intercept_time_s=at_s, minimum_distance_km=1.0,
        distance_at_completion_km=5.0, successful=True, analytical_endpoint_error_km=None,
        branch="test",
    )


def test_t_team_runs_from_the_origins_epoch() -> None:
    """SC_Alien, the origin, starts 5400 s before t = 0: a hit at 600 s is T_team 6000 s."""
    early, _, _ = _load(_config())
    late, _, _ = _load(_config(mission_time_origin="pursuer"))
    assert early.time_origin_offset_s == pytest.approx(5400.0, abs=1e-6)
    assert late.time_origin_offset_s == 0.0
    # T_max = 4 T_A from SC_Alien's epoch, of which 5400 s are gone at t = 0.
    assert late.max_time_s - early.max_time_s == pytest.approx(5400.0, abs=1e-6)
    assert on_solver_clock(SCORE, early).ct == pytest.approx(600.0, abs=1e-6)
    assert score_candidate(_hit(600.0), early, SCORE).time_score == pytest.approx(12.5)
    assert score_candidate(_hit(6000.0), late, SCORE).time_score == pytest.approx(12.5)
    # A miss completes at T_max, the same instant whichever clock reads it.
    miss = dataclasses.replace(_hit(0.0), intercept_time_s=None, successful=False)
    assert score_candidate(miss, early, SCORE).time_score == pytest.approx(
        score_candidate(miss, late, SCORE).time_score
    )


def test_from_origin_counts_the_alignments_closest_approach() -> None:
    config = _ranking_config()
    window = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, closest_approach_window="from_origin")
    )
    after, _, _ = _load(config)
    origin, _, _ = _load(window)
    assert after.alignment_minimum_distance_km is None
    assert origin.alignment_minimum_distance_km is not None
    miss = Candidate(
        burns=(), intercept_time_s=None, minimum_distance_km=5000.0,
        distance_at_completion_km=9000.0, successful=False, analytical_endpoint_error_km=None,
        branch="test",
    )
    near = origin.alignment_minimum_distance_km
    assert score_candidate(miss, after, RANKING_SCORE).distance_score < score_candidate(
        miss, origin, RANKING_SCORE
    ).distance_score
    # The fake holds the early craft still, so its alignment comes no nearer than this.
    assert near < 5000.0


def _aligned_probe(scenario, radius_km: float):
    probe = scenario.state_probe
    aligned = probe.aligned
    moved = dict(aligned.states)
    return dataclasses.replace(
        probe, aligned=dataclasses.replace(aligned, states=moved, early_minimum_radius_km=radius_km)
    )


def _pursuer_early(tmp_path: Path) -> Path:
    """The crossing fixture with the epochs swapped, SC_Human 5400 s first."""
    text = CROSSING.read_text(encoding="utf-8")
    text = text.replace("SC_Human.Epoch = '01 Jan 2026 12:00:00.000';", "SC_Human.Epoch = 'X';")
    text = text.replace("SC_Alien.Epoch = '01 Jan 2026 10:30:00.000';",
                        "SC_Alien.Epoch = '01 Jan 2026 12:00:00.000';")
    text = text.replace("SC_Human.Epoch = 'X';", "SC_Human.Epoch = '01 Jan 2026 10:30:00.000';")
    text = text.replace(
        "Propagate Prop(SC_Alien) {SC_Alien.UTCModJulian = SC_Human.UTCModJulian};",
        "Propagate Prop(SC_Human) {SC_Human.UTCModJulian = SC_Alien.UTCModJulian};",
    )
    path = tmp_path / "pursuer_early.script"
    path.write_text(text, encoding="utf-8")
    return path


def test_an_early_pursuer_that_dives_through_the_alignment_is_infeasible(tmp_path: Path) -> None:
    config = _config(mission_time_origin="pursuer", horizon_s=40000.0)
    path = _pursuer_early(tmp_path)
    scenario, _, _ = _load(config, path)
    assert scenario.alignment is not None and scenario.alignment.early == "SC_Human"
    with pytest.raises(ParseError, match="before any burn may be placed: the template is infeasible"):
        parse_gmat_script(path, config, probe=_aligned_probe(scenario, 6300.0))


def test_the_searches_may_start_at_t_zero_only_after_an_alignment(tmp_path: Path) -> None:
    config = _config()
    aligned, _, _ = _load(config)
    assert RankingLimits.from_config(aligned, config).first_burn_floor_s == 0.0
    assert MultiBurnLimits.from_config(aligned, config).first_burn_floor_s == 0.0
    plain, _ = parse_gmat_script(
        ROOT / "examples" / "synthetic" / "NoManeuver_Circular.script",
        OrbitWinConfig(pursuer_name="Pursuer", target_name="AlienSC"),
    )
    assert RankingLimits.from_config(plain, config).first_burn_floor_s > 0.0
    assert MultiBurnLimits.from_config(plain, config).first_burn_floor_s > 0.0


def test_rank_and_certify_refuse_an_archive_on_another_clock(tmp_path: Path) -> None:
    from orbitwin.rank import rank_report
    from orbitwin.verify import _check_archive_stamps

    config = _config()
    scenario, _, _ = _load(config)
    candidate = _hit(600.0)
    candidate.metadata.update(
        scenario_sha256=scenario.source_sha256, horizon_s=scenario.max_time_s,
        time_origin_offset_s=0.0, event_refined=True,
    )
    with pytest.raises(OrbitWinError, match="mission_time_origin differs"):
        rank_report(scenario, [candidate], config)
    with pytest.raises(PolicyError, match="mission_time_origin differs"):
        _check_archive_stamps(candidate, scenario, config)
    candidate.metadata["time_origin_offset_s"] = scenario.time_origin_offset_s
    report = rank_report(scenario, [candidate], config)
    clock = report["clock"]
    assert clock["mission_time_origin"] == "target"  # type: ignore[index]
    assert clock["time_origin_craft"] == "SC_Alien"  # type: ignore[index]
    assert clock["time_origin_offset_s"] == scenario.time_origin_offset_s  # type: ignore[index]


def test_aligned_states_are_recorded_with_the_probe() -> None:
    scenario, _, evidence = _load(_config())
    assert evidence is not None and evidence.aligned is not None
    assert any(name.startswith("alignment/") for name in evidence.files)
    assert isinstance(scenario.state_probe.aligned, AlignedStates)  # type: ignore[union-attr]
    assert isinstance(scenario.target.initial_state, StateVector)


def test_the_ranking_search_moves_ct_once() -> None:
    """Moved in the setup and again in score_candidate, 300 s and 600 s hits tied."""
    from orbitwin.ranking_search import (
        Measurement,
        RankingSolution,
        build_setup,
        scored_candidate,
        solution_key,
    )

    config = _ranking_config()
    scenario, _, _ = _load(config)
    setup = build_setup(scenario, config)
    keys = []
    for hit_s in (300.0, 600.0):
        solution = RankingSolution(
            "x", "p", (10.0,), ((0.249, 0.0, 0.0),),
            Measurement(1.0, hit_s, hit_s, 9.0, 7000.0), 0.249,
        )
        official = scored_candidate(solution, scenario, config, setup.limits).score
        assert official is not None
        # The ordering's score is the official one less its per-burn cost.
        key_score = -solution_key(solution, setup, scenario, config)[0]
        assert key_score == pytest.approx(official - 1.0e-3, abs=1.0e-9)
        keys.append(key_score)
    assert keys[0] > keys[1]
    assert setup.score is config.score and setup.time_origin_offset_s == scenario.time_origin_offset_s


def test_from_origin_is_after_alignment_when_the_origin_is_the_later_craft() -> None:
    late, _, _ = _load(_config(mission_time_origin="pursuer", closest_approach_window="from_origin"))
    assert late.time_origin_offset_s == 0.0
    assert late.alignment_minimum_distance_km is None


def _ranking_config(**policy: object) -> OrbitWinConfig:
    """SC_Alien starts first in the fixture, so the earliest origin is the target's."""
    settings: dict[str, object] = {"mission_time_origin": "earliest"}
    settings.update(policy)
    return _config(rules="ranking", horizon_s=20000.0, total_dv_limit_km_s=3.0,
                   total_dv_safe_limit_km_s=2.9, dv_limit_km_s=0.25, dv_safe_limit_km_s=0.249,
                   team_number=TEAM_NUMBER, **settings)


def test_the_search_s_own_time_terms_add_the_offset_once() -> None:
    """SLSQP's _time_term and the start ranking's _transfer_value, not only the ordering."""
    from orbitwin.ranking_search import Transfer, _Problem, _transfer_value, build_setup
    from orbitwin.score import _logistic_descending

    config = _ranking_config()
    scenario, _, _ = _load(config)
    offset = scenario.time_origin_offset_s
    assert offset == pytest.approx(5400.0, abs=1e-6)
    setup = build_setup(scenario, config)
    expected = _logistic_descending(25.0, RANKING_SCORE.kt * (300.0 + offset - RANKING_SCORE.ct))
    assert _Problem(setup, 1)._time_term(300.0) == pytest.approx(expected)
    transfer = Transfer(10.0, 300.0, (0.0, 0.0, 0.0))
    dv_term = _logistic_descending(25.0, RANKING_SCORE.kv * (0.0 - RANKING_SCORE.cv))
    assert _transfer_value(transfer, 1.0, setup) == pytest.approx(expected + dv_term)


def test_a_certified_miss_completes_at_t_max_on_the_origin_s_clock(tmp_path: Path) -> None:
    from orbitwin.search import zero_burn_candidate
    from orbitwin.verify import certify_candidate

    config = _ranking_config()
    scenario, _, _ = _load(config)
    miss = zero_burn_candidate(scenario, config)
    text = with_ranking_report(CROSSING.read_text(encoding="utf-8")) + "% FAKE_GMAT_MISS_KM 900\n"
    script = tmp_path / "miss.script"
    script.write_text(text, encoding="utf-8")
    scenario, _, _ = _load(config, script)
    miss.metadata.update(scenario_sha256=scenario.source_sha256, horizon_s=scenario.max_time_s)
    miss.metadata.update(clock_stamps(scenario))
    certificate = certify_candidate(script, miss, config, FAKE, STARTUP, tmp_path / "certify")
    clock = certificate["clock"]
    assert (clock["mission_time_origin"], clock["time_origin_craft"]) == ("earliest", "SC_Alien")
    assert clock["t_team_s"] == pytest.approx(scenario.max_time_s + scenario.time_origin_offset_s)
    assert clock["t_team_s"] == pytest.approx(20000.0)


def test_conftest_s_stamped_carries_the_clock_stamps(tmp_path: Path) -> None:
    """stamped() stands in for solve's stamping, which includes clock_stamps."""
    from conftest import stamped

    from orbitwin.search import zero_burn_candidate
    from orbitwin.verify import certify_candidate

    config = _ranking_config()
    script = tmp_path / "miss.script"
    script.write_text(
        with_ranking_report(CROSSING.read_text(encoding="utf-8")) + "% FAKE_GMAT_MISS_KM 900\n",
        encoding="utf-8",
    )
    scenario, _, _ = _load(config, script)
    assert scenario.time_origin_offset_s == pytest.approx(5400.0, abs=1e-6)
    miss = stamped(zero_burn_candidate(scenario, config), script, config)
    certificate = certify_candidate(script, miss, config, FAKE, STARTUP, tmp_path / "certify")
    assert certificate["clock"]["t_team_s"] == pytest.approx(20000.0)


def test_rank_records_the_alignment_probe_and_refuses_another_window() -> None:
    from orbitwin.rank import rank_report
    from orbitwin.verify import _check_archive_stamps

    config = _ranking_config(closest_approach_window="from_origin")
    scenario, _, _ = _load(config)
    candidate = _hit(600.0)
    candidate.metadata.update(
        scenario_sha256=scenario.source_sha256, horizon_s=scenario.max_time_s,
        time_origin_offset_s=scenario.time_origin_offset_s, event_refined=True,
    )
    with pytest.raises(OrbitWinError, match="closest_approach_window differs"):
        rank_report(scenario, [candidate], config)
    with pytest.raises(PolicyError, match="closest_approach_window differs"):
        _check_archive_stamps(candidate, scenario, config)
    candidate.metadata["alignment_minimum_distance_km"] = scenario.alignment_minimum_distance_km
    clock = rank_report(scenario, [candidate], config)["clock"]
    assert clock["alignment_probe_report_sha256"]  # type: ignore[index]


def test_an_early_pursuer_under_the_floor_is_refused(tmp_path: Path) -> None:
    config = _config(mission_time_origin="pursuer", horizon_s=40000.0, minimum_flown_altitude_km=100.0)
    path = _pursuer_early(tmp_path)
    scenario, _, _ = _load(_config(mission_time_origin="pursuer", horizon_s=40000.0), path)
    with pytest.raises(ParseError, match="every candidate would be refused"):
        parse_gmat_script(path, config, probe=_aligned_probe(scenario, 6420.0))


def test_multi_burn_puts_a_first_burn_at_zero_or_past_the_floor() -> None:
    import numpy as np

    from orbitwin.multi_burn import MultiBurnLimits, _unpack

    limits = MultiBurnLimits(
        burn_cap_km_s=0.249, separation_s=101.0, first_burn_floor_s=0.0,
        radius_floor_km=6478.0, horizon_s=1000.0,
    )
    for raw, expected in ((-1.0, 0.0), (2.0e-4, 0.0), (7.0e-4, 1.001e-3), (5.0, 5.0)):
        # render_floor_s defaults to the policy's 1e-3 s plus 1 µs.
        first, *_ = _unpack(np.array([raw, 500.0, 0.0, 0.0, 0.0]), 1, limits)
        assert first == pytest.approx(expected)


def test_the_first_burn_floor_follows_the_policy() -> None:
    import numpy as np

    from orbitwin.multi_burn import _unpack
    from orbitwin.ranking_search import repair, wall_violations

    config = _config(min_propagate_duration_s=0.01)
    scenario, _, _ = _load(config)
    limits = MultiBurnLimits.from_config(scenario, config)
    assert limits.render_floor_s == pytest.approx(0.01 + 1.0e-6)
    first, *_ = _unpack(np.array([0.007, 500.0, 0.0, 0.0, 0.0]), 1, limits)
    assert first == pytest.approx(0.01 + 1.0e-6)
    ranking = RankingLimits.from_config(scenario, config)
    impulses = np.zeros((1, 3))
    for raw, expected in ((0.002, 0.0), (0.007, 0.01 + 1.0e-6), (5.0, 5.0)):
        times, _ = repair([raw], impulses, ranking)
        assert times[0] == pytest.approx(expected)
    assert wall_violations([0.004], impulses, ranking)
    assert not wall_violations([0.0], impulses, ranking)


def test_an_archive_without_clock_stamps_is_named_as_such() -> None:
    from orbitwin.models import clock_stamp_mismatch

    scenario, _, _ = _load(_config())
    assert "carries no clock stamp" in (clock_stamp_mismatch({}, scenario) or "")


def test_a_template_that_maneuvers_after_its_alignment_has_slot_zero_at_zero(
    tmp_path: Path,
) -> None:
    """Every full-slot schedule was refused at render with slot 0 floored at 1 ms."""
    import numpy as np

    from orbitwin.ranking_search import repair, wall_violations
    from orbitwin.script_renderer import render_candidate

    workshop = ROOT / "examples" / "two_burn_rendezvous.script"
    epoch = "AlienSC.Epoch = '15 Mar 2027 12:00:00.000';"
    text = workshop.read_text(encoding="utf-8")
    assert epoch in text
    text = text.replace(epoch, "AlienSC.Epoch = '14 Mar 2027 21:00:00.000';")
    head, _, mission = text.partition("BeginMissionSequence;\n")
    lines = mission.splitlines(keepends=True)
    align = "Propagate DefaultProp(AlienSC) {AlienSC.UTCModJulian = HumanSC.UTCModJulian};\n"
    path = tmp_path / "slotted.script"
    path.write_text(head + "BeginMissionSequence;\n" + align + "".join(lines[1:]), encoding="utf-8")
    config = OrbitWinConfig(policy=dataclasses.replace(
        PolicyConfig(), mission_time_origin="pursuer", closest_approach_window="after_alignment",
        horizon_s=40000.0,
    ))
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    limits = RankingLimits.from_config(scenario, config)
    assert limits.pins_first_burn(2) and not limits.pins_first_burn(1)
    assert limits.first_burn_floor_for(1) == pytest.approx(limits.separation_s)
    impulses = np.full((2, 3), 0.01)
    times, _ = repair([5.0, 300.0], impulses, limits)
    assert times[0] == 0.0
    assert wall_violations([5.0, 300.0], impulses, limits)
    assert not wall_violations(times, impulses, limits)
    from orbitwin.models import Burn

    frame = ("Local", "Earth", "VNB")
    candidate = Candidate(
        burns=tuple(Burn(t, (0.01, 0.0, 0.0), (0.01, 0.0, 0.0), k, frame) for k, t in enumerate(times)),
        intercept_time_s=None, minimum_distance_km=None, distance_at_completion_km=None,
        successful=False, analytical_endpoint_error_km=None, branch="test",
    )
    assert render_candidate(document, scenario, candidate, config).burn_times_s[0] == 0.0

    # Slot 0 alone at t = 0 with slot 1 empty: dropping the empty one moved the burn
    # to slot 1, floored at 101 s, and the schedule was lost (0.52 points on GMAT).
    from orbitwin.ranking_search import as_candidate, build_setup, settle

    setup = build_setup(scenario, config)
    lone = np.array([[0.02, 0.0, 0.0], [0.0, 0.0, 0.0]])
    solution = settle("x", "p", [0.0, 101.0], lone, setup)
    assert solution is not None and solution.times_s[0] == 0.0
    assert solution.burn_count == 2 and solution.active_burn_count == 1
    burns = as_candidate(solution, scenario, config, setup.limits).burns
    assert [burn.slot_index for burn in burns] == [0, 1]
    assert render_candidate(
        document, scenario, dataclasses.replace(candidate, burns=burns, candidate_id=""), config
    ).burn_times_s[0] == 0.0


def test_a_pinned_lone_burn_survives_an_empty_slot_after_the_interception(
    tmp_path: Path,
) -> None:
    """An empty slot left past the entry was trimmed, and the burn at 0 floored to 101 s."""
    import numpy as np

    from orbitwin.errors import PolicyError as Refused
    from orbitwin.lambert import solve_lambert_all
    from orbitwin.ranking_search import as_candidate, build_setup, settle
    from orbitwin.script_renderer import render_candidate

    workshop = ROOT / "examples" / "two_burn_rendezvous.script"
    epoch = "AlienSC.Epoch = '15 Mar 2027 12:00:00.000';"
    text = workshop.read_text(encoding="utf-8")
    assert epoch in text
    text = text.replace(epoch, "AlienSC.Epoch = '14 Mar 2027 21:00:00.000';")
    head, _, mission = text.partition("BeginMissionSequence;\n")
    lines = mission.splitlines(keepends=True)
    align = "Propagate DefaultProp(AlienSC) {AlienSC.UTCModJulian = HumanSC.UTCModJulian};\n"
    path = tmp_path / "slotted.script"
    path.write_text(head + "BeginMissionSequence;\n" + align + "".join(lines[1:]), encoding="utf-8")
    config = OrbitWinConfig(policy=dataclasses.replace(
        PolicyConfig(), mission_time_origin="pursuer", closest_approach_window="after_alignment",
        horizon_s=40000.0,
    ))
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    setup = build_setup(scenario, config)
    r0, v0 = np.asarray(setup.initial.r_km), np.asarray(setup.initial.v_km_s)
    target, _ = setup.target.state(4200.0)
    burn = min(
        (np.asarray(item.v1_km_s) - v0 for item in solve_lambert_all(r0, np.asarray(target), 4200.0, setup.mu)),
        key=np.linalg.norm,
    )
    impulses = np.array([burn, [0.0, 0.0, 0.0]])
    for empty_s in (101.0, 4193.5, 5200.0):
        solution = settle("x", "p", [0.0, empty_s], impulses, setup)
        assert solution is not None and solution.times_s == (0.0, 101.0), empty_s
        assert solution.measurement.intercept_time_s is not None
    # Nor a small burn past the entry, which the trim used to drop with slot 0's place.
    for tail in (1.0e-6, 1.0e-3, 0.05):
        late = np.array([burn, [tail, 0.0, 0.0]])
        solution = settle("x", "p", [0.0, 5200.0], late, setup)
        assert solution is not None and solution.times_s == (0.0, 101.0), tail
        assert solution.active_burn_count == 1 and solution.measurement.intercept_time_s
    # Three slots: the trim keeps a burn flown before the entry where it was.
    from orbitwin.conic import coast

    three = dataclasses.replace(setup, limits=dataclasses.replace(setup.limits, slot_count=3))
    first = np.array([0.01, 0.0, 0.0])
    r1, v1 = coast(r0, v0 + first, 150.0, setup.mu)
    second = min(
        (np.asarray(item.v1_km_s) - np.asarray(v1)
         for item in solve_lambert_all(np.asarray(r1), np.asarray(target), 4050.0, setup.mu)),
        key=np.linalg.norm,
    )
    kept = settle("x", "p", [0.0, 150.0, 5200.0],
                  np.array([first, second, [0.05, 0.0, 0.0]]), three)
    assert kept is not None and kept.times_s == (0.0, 150.0, 251.0)
    assert kept.measurement.intercept_time_s is not None
    # With zero vectors forbidden the renderer refuses what the guard would.
    off = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, allow_zero_vector_unused_slots=False)
    )
    candidate = as_candidate(solution, scenario, off, setup.limits)
    with pytest.raises(Refused, match="exact zero vector"):
        render_candidate(document, scenario, candidate, off)


def test_an_exact_zero_burn_is_refused_at_render_when_the_policy_forbids_it() -> None:
    """The final-script guard refused it; the renderer let it through to stop solve."""
    from orbitwin.models import Burn
    from orbitwin.script_renderer import render_candidate

    config = OrbitWinConfig(
        policy=dataclasses.replace(PolicyConfig(), allow_zero_vector_unused_slots=False),
        pursuer_name="Pursuer", target_name="AlienSC",
    )
    path = ROOT / "examples" / "synthetic" / "NoManeuver_Circular.script"
    scenario, document = parse_gmat_script(path, config)
    frame = ("Local", "Earth", "VNB")
    candidate = Candidate(
        burns=(
            Burn(1000.0, (0.05, 0.0, 0.0), (0.05, 0.0, 0.0), 0, frame),
            Burn(2000.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, frame),
        ),
        intercept_time_s=None, minimum_distance_km=None, distance_at_completion_km=None,
        successful=False, analytical_endpoint_error_km=None, branch="test",
    )
    with pytest.raises(PolicyError, match="exact zero vector"):
        render_candidate(document, scenario, candidate, config)


def test_the_corrector_moves_burns_not_the_empty_slots_after_them() -> None:
    """A zero kept 2 s before the entry needed 143 m/s where the real burn needed 5."""
    import numpy as np

    from orbitwin.gmat_shoot import _corrected

    burn, zero = np.array([0.01, 0.0, 0.0]), np.zeros(3)
    assert _corrected([(0.0, burn), (150.0, burn), (3492.0, zero), (3593.0, zero)]) == [0, 1]
    assert _corrected([(0.0, burn), (150.0, burn), (3492.0, burn)]) == [1, 2]
    assert _corrected([(0.0, burn), (101.0, zero)]) == [0, 1]
    assert _corrected([(0.0, burn), (101.0, zero), (202.0, zero)]) == [0, 2]
    assert _corrected([(50.0, burn)]) == [0]


def test_a_correction_step_moves_only_the_burns() -> None:
    """Jacobians and the step on [burn, burn, zero, zero]: every caller of _corrected."""
    import numpy as np

    from orbitwin import gmat_shoot

    scenario, _, _ = _load(_config())
    small = np.array([0.0, 0.05, 0.0])
    schedule = [(0.0, small), (150.0, small), (3492.0, np.zeros(3)), (3593.0, np.zeros(3))]
    jacobians = gmat_shoot._two_body_jacobians(scenario, schedule, 4200.0)
    assert sorted(jacobians) == [0, 1]
    # A miss the last burn alone cannot remove inside a 0.06 km/s cap, so both share it.
    trial, mode, broken = gmat_shoot._step(
        schedule, np.array([40.0, 0.0, 0.0]), jacobians, gmat_shoot._Limits(0.06, None)
    )
    assert broken is None and trial is not None, broken
    assert mode in ("last", "last-two")
    for index in (2, 3):
        assert not np.any(trial[index][1])

