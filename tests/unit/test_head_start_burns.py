"""Burns before the later craft's epoch: policy.head_start_burns.

COMPETITION_POLICY.md, "Burns in the head start: signed off" (2026-11-08).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from conftest import FAKE_STARTUP

from orbitwin.burn_rows import FlownManeuver, maneuvers_flown
from orbitwin.config import OrbitWinConfig, PolicyConfig, ScoreConfig, SearchConfig
from orbitwin.conic import coast
from orbitwin.constraints import validate_candidate
from orbitwin.errors import DynamicsError, ParseError, PolicyError
from orbitwin.gmat_parser import mission_schedule, parse_gmat_script
from orbitwin.models import (
    Burn,
    Candidate,
    head_start_lead_s,
    head_start_status,
    pursuer_start,
    uses_head_start,
)
from orbitwin.rank import head_start_choice, head_start_line
from orbitwin.ranking_search import RankingLimits, repair, wall_violations
from orbitwin.report_parser import evaluate_diagnostic_report
from orbitwin.score import HEAD_START_ALIGNMENT_KEY, ScoreBreakdown, scored_distance_km
from orbitwin.script_renderer import assert_final_script_policy, render_candidate
from orbitwin.state_probe import load_scenario

ROOT = Path(__file__).resolve().parents[2]
CROSSING = ROOT / "tests" / "fixtures" / "async" / "aligned_crossing.script"
FAKE = ROOT / "scripts" / "fake_gmat_console.py"
STARTUP = FAKE_STARTUP
#: The fixture's head start once SC_Human is moved first: 01:30 before 12:00.
LEAD_S = 5400.0
SCORE = ScoreConfig(kt=0.003, ct=6000.0, kv=0.0012, cv=2000.0, delta_v_unit="m/s", cr_km=3000.0)


def _config(**policy: object) -> OrbitWinConfig:
    settings: dict[str, object] = {
        "rules": "ranking", "mission_time_origin": "earliest",
        "closest_approach_window": "after_alignment", "horizon_s": 12000.0,
        "head_start_burns": True, "head_start_min_gain_points": 0.05,
        "dv_limit_km_s": 0.25, "dv_safe_limit_km_s": 0.249, "total_dv_limit_km_s": 3.0,
        "total_dv_safe_limit_km_s": 2.99, "safe_burn_separation_s": 101.0,
        "final_padding_s": 20.0,
    }
    settings.update(policy)
    return OrbitWinConfig(
        policy=dataclasses.replace(PolicyConfig(), **settings),  # type: ignore[arg-type]
        search=dataclasses.replace(
            SearchConfig(), ranking_search=True, ranking_max_burns=2,
            ranking_transfer_starts=2, ranking_random_starts=2, ranking_polish_top=2, workers=2,
        ),
        pursuer_name="SC_Human", target_name="SC_Alien", score=SCORE,
    )


def _script(tmp_path: Path, *, pursuer_first: bool = True, alignment: str | None = None,
            mission_tail: str = "") -> Path:
    """The crossing fixture, SC_Human 5400 s first unless `pursuer_first` is False."""
    text = CROSSING.read_text(encoding="utf-8")
    if pursuer_first:
        text = text.replace("SC_Human.Epoch = '01 Jan 2026 12:00:00.000';", "SC_Human.Epoch = 'X';")
        text = text.replace("SC_Alien.Epoch = '01 Jan 2026 10:30:00.000';",
                            "SC_Alien.Epoch = '01 Jan 2026 12:00:00.000';")
        text = text.replace("SC_Human.Epoch = 'X';", "SC_Human.Epoch = '01 Jan 2026 10:30:00.000';")
        text = text.replace(
            "Propagate Prop(SC_Alien) {SC_Alien.UTCModJulian = SC_Human.UTCModJulian};",
            alignment or "Propagate Prop(SC_Human) {SC_Human.UTCModJulian = SC_Alien.UTCModJulian};",
        )
    text = text.replace(
        "Propagate Prop(SC_Alien, SC_Human) {SC_Human.ElapsedSecs = 6000};",
        "Propagate Prop(SC_Alien, SC_Human) {SC_Human.ElapsedSecs = 6000};" + mission_tail,
    )
    path = tmp_path / ("pursuer_first.script" if pursuer_first else "target_first.script")
    path.write_text(text, encoding="utf-8")
    return path


def _load(path: Path, config: OrbitWinConfig | None = None):
    return load_scenario(path, config or _config(), gmat_console=FAKE, startup_file=STARTUP)


def _burn(time_s: float, slot: int, frame: tuple[str, str, str]) -> Burn:
    return Burn(time_s, (0.0, 0.01, 0.0), (0.01, 0.0, 0.0), slot, frame)


def _candidate(times: tuple[float, ...], frame: tuple[str, str, str]) -> Candidate:
    return Candidate(
        burns=tuple(_burn(t, k, frame) for k, t in enumerate(times)), intercept_time_s=None,
        minimum_distance_km=100.0, distance_at_completion_km=100.0, successful=False,
        analytical_endpoint_error_km=None, branch="test",
    )


def _rendered(tmp_path: Path, times: tuple[float, ...], config: OrbitWinConfig | None = None):
    path = _script(tmp_path)
    config = config or _config()
    scenario, document, _ = _load(path, config)
    candidate = _candidate(times, scenario.synthesized_burn_frame)
    result = render_candidate(document, scenario, candidate, config)
    generated = tmp_path / "generated.script"
    generated.write_bytes(result.script_bytes)
    return path, scenario, generated, result


def _with_mission(tmp_path: Path, generated: Path, old: str, new: str) -> Path:
    text = generated.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    edited = tmp_path / "edited.script"
    edited.write_text(text.replace(old, new), encoding="utf-8")
    return edited


# -- the scenario ------------------------------------------------------------------


def test_the_head_start_opens_on_an_open_pursuer_first_template(tmp_path: Path) -> None:
    scenario, _, _ = _load(_script(tmp_path))
    assert head_start_lead_s(scenario) == pytest.approx(LEAD_S, abs=1e-6)
    assert scenario.head_start_state is not None
    probed = scenario.state_probe.craft_named("SC_Human")  # type: ignore[union-attr]
    # GMAT's own state at the pursuer's epoch, not a flight back from t = 0.
    assert scenario.head_start_state == probed.state  # type: ignore[union-attr]
    start_s, state = pursuer_start(scenario, [-100.0])
    assert (start_s, state) == (-scenario.head_start_s, scenario.head_start_state)
    assert pursuer_start(scenario, [100.0]) == (0.0, scenario.pursuer.initial_state)


@pytest.mark.parametrize(
    ("change", "note"),
    [
        ({"head_start_burns": False}, None),
        ({"pursuer_first": False}, "SC_Alien starts first"),
        ({"alignment": "Propagate Prop(SC_Human) {SC_Human.ElapsedSecs = 5400};"},
         "ElapsedSecs goal"),
    ],
)
def test_the_head_start_stays_closed_elsewhere(tmp_path: Path, change: dict, note: str | None) -> None:
    policy = {k: v for k, v in change.items() if k == "head_start_burns"}
    shape = {k: v for k, v in change.items() if k != "head_start_burns"}
    config = _config(**policy) if policy else _config()
    if not shape.get("pursuer_first", True):
        config = _config(mission_time_origin="earliest")
    scenario, document, _ = _load(_script(tmp_path, **shape), config)
    assert head_start_lead_s(scenario) == 0.0
    if note is not None:
        assert note in str(scenario.head_start_note)
    # Nor will the renderer write a burn before t = 0 there.
    with pytest.raises(PolicyError, match="head start"):
        render_candidate(document, scenario,
                         _candidate((-100.0,), scenario.synthesized_burn_frame), config)


def test_a_slotted_template_has_no_head_start(tmp_path: Path) -> None:
    config = _config()
    _, _, generated, _ = _rendered(tmp_path, (500.0,))
    # The render maneuvers, so it reads as slotted: nothing may come before its alignment.
    slotted, document, _ = _load(generated, config)
    assert slotted.head_start_s == pytest.approx(LEAD_S, abs=1e-6)
    assert head_start_lead_s(slotted) == 0.0
    assert "maneuvers already" in str(head_start_status(slotted))
    slot = slotted.burn_slots[0]
    early = Burn(-100.0, (0.0, 0.01, 0.0), (0.01, 0.0, 0.0), 0,
                 (slot.coordinate_system, slot.origin, slot.axes))
    candidate = dataclasses.replace(_candidate((500.0,), early.frame or ("", "", "")),
                                    burns=(early,))
    with pytest.raises(PolicyError, match="head start"):
        render_candidate(document, slotted, candidate, config)


# -- render, parse and guard ---------------------------------------------------------


def test_head_start_burns_render_before_the_alignment(tmp_path: Path) -> None:
    path, scenario, generated, result = _rendered(tmp_path, (-3000.0, -1000.0, 500.0))
    mission = generated.read_text(encoding="utf-8").split("BeginMissionSequence;\n")[1].splitlines()
    assert mission[:6] == [
        "Propagate Prop(SC_Human) {SC_Human.ElapsedSecs = 2400.0};",
        "Maneuver ImpulsiveBurn1(SC_Human);",
        "Propagate Prop(SC_Human) {SC_Human.ElapsedSecs = 2000.0};",
        "Maneuver ImpulsiveBurn2(SC_Human);",
        "Propagate Prop(SC_Human) {SC_Human.UTCModJulian = SC_Alien.UTCModJulian};",
        "Propagate Prop(SC_Alien, SC_Human) {SC_Human.ElapsedSecs = 500.0};",
    ]
    assert result.burn_times_s == (-3000.0, -1000.0, 500.0)
    config = _config()
    assert_final_script_policy(path, generated, config, probe=scenario.state_probe)
    parsed, _ = parse_gmat_script(generated, config, probe=scenario.state_probe)
    assert [m.time_s for m in maneuvers_flown(parsed)] == pytest.approx([-3000.0, -1000.0, 500.0])
    schedule = mission_schedule(generated)
    assert schedule.alignment_craft == "SC_Human"
    assert schedule.head_start_s == pytest.approx(4400.0)
    # The last row: the alignment plus the goals after it, never the head start's.
    assert schedule.end_s == pytest.approx(result.mission_end_s)
    assert [(m.time_s, m.head_start) for m in schedule.maneuvers] == [
        (2400.0, True), (4400.0, True), (500.0, False)]
    assert schedule.total_dv_km_s == pytest.approx(0.03)


def test_an_added_alignment_keeps_the_head_start_before_it(tmp_path: Path) -> None:
    text = _script(tmp_path).read_text(encoding="utf-8")
    bare = tmp_path / "no_alignment.script"
    bare.write_text(text.replace(
        "Propagate Prop(SC_Human) {SC_Human.UTCModJulian = SC_Alien.UTCModJulian};\n", ""),
        encoding="utf-8")
    config = _config(add_epoch_alignment=True)
    scenario, document, _ = _load(bare, config)
    assert scenario.alignment is not None and scenario.alignment.line_index is None
    candidate = _candidate((-1000.0, 500.0), scenario.synthesized_burn_frame)
    generated = tmp_path / "generated.script"
    generated.write_bytes(render_candidate(document, scenario, candidate, config).script_bytes)
    mission = generated.read_text(encoding="utf-8").split("BeginMissionSequence;\n")[1].splitlines()
    assert mission[2] == "Propagate Prop(SC_Human) {SC_Human.A1ModJulian = SC_Alien.A1ModJulian};"
    assert_final_script_policy(bare, generated, config, probe=scenario.state_probe)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        # Another command before the alignment.
        ("Maneuver ImpulsiveBurn1(SC_Human);\n",
         "Maneuver ImpulsiveBurn1(SC_Human);\nPropagate Prop(SC_Human) {SC_Human.ElapsedSecs = 1};\n",
         "epoch alignment is read only"),
        # Another shape of Propagate there.
        ("{SC_Human.ElapsedSecs = 2400.0};",
         "{SC_Human.ElapsedSecs = 2400.0, StopTolerance = 1e-7};", "is not `Propagate"),
        # Another propagator there.
        ("Propagate Prop(SC_Human) {SC_Human.ElapsedSecs = 2400.0};",
         "Propagate Other(SC_Human) {SC_Human.ElapsedSecs = 2400.0};", "Other"),
        # A zero-length segment hangs GMAT.
        ("{SC_Human.ElapsedSecs = 2400.0};", "{SC_Human.ElapsedSecs = 0};", "floor"),
        # A last burn at the later epoch: the head start is 5400 s.
        ("{SC_Human.ElapsedSecs = 2000.0};", "{SC_Human.ElapsedSecs = 3000.0};", "strictly before"),
        # The target maneuvered.
        ("Maneuver ImpulsiveBurn1(SC_Human);", "Maneuver ImpulsiveBurn1(SC_Alien);",
         "SC_Alien|Target spacecraft|non-pursuer"),
        # The target flown alone.
        ("Propagate Prop(SC_Human) {SC_Human.ElapsedSecs = 2400.0};",
         "Propagate Prop(SC_Alien) {SC_Alien.ElapsedSecs = 2400.0};", "is not `Propagate"),
    ],
)
def test_the_parser_refuses_anything_else_before_the_alignment(
    tmp_path: Path, old: str, new: str, match: str
) -> None:
    path, scenario, generated, _ = _rendered(tmp_path, (-3000.0, -1000.0, 500.0))
    text = generated.read_text(encoding="utf-8")
    if "Other(" in new:
        text = text.replace("Create Propagator Prop;", "Create Propagator Prop Other;")
        generated.write_text(text, encoding="utf-8")
    edited = _with_mission(tmp_path, generated, old, new)
    with pytest.raises((ParseError, PolicyError), match=match):
        assert_final_script_policy(path, edited, _config(), probe=scenario.state_probe)


def test_the_guard_refuses_head_start_burns_where_the_policy_is_off(tmp_path: Path) -> None:
    path, scenario, generated, _ = _rendered(tmp_path, (-1000.0, 500.0))
    with pytest.raises(ParseError, match=r"policy\.head_start_burns is off"):
        assert_final_script_policy(
            path, generated, _config(head_start_burns=False), probe=scenario.state_probe
        )


def test_the_guard_refuses_head_start_burns_on_a_slotted_original(tmp_path: Path) -> None:
    config = _config()
    _, scenario, slotted, _ = _rendered(tmp_path, (500.0,))
    # Head-start pairs written into a template that maneuvers already.
    edited = _with_mission(
        tmp_path, slotted,
        "Propagate Prop(SC_Human) {SC_Human.UTCModJulian",
        "Propagate Prop(SC_Human) {SC_Human.ElapsedSecs = 100.0};\nManeuver Extra(SC_Human);\n"
        "Propagate Prop(SC_Human) {SC_Human.UTCModJulian",
    )
    text = edited.read_text(encoding="utf-8").replace(
        "BeginMissionSequence;",
        "Create ImpulsiveBurn Extra;\nExtra.CoordinateSystem = Local;\nExtra.Origin = Earth;\n"
        "Extra.Axes = VNB;\nExtra.Element1 = 0.001;\nExtra.Element2 = 0;\nExtra.Element3 = 0;\n"
        "Extra.DecrementMass = false;\nBeginMissionSequence;")
    edited.write_text(text, encoding="utf-8")
    with pytest.raises(PolicyError, match=r"no head\s+start|Create of|adds"):
        assert_final_script_policy(slotted, edited, config, probe=scenario.state_probe)


@pytest.mark.parametrize(
    ("times", "match"),
    [
        ((-LEAD_S, 500.0), "outside the head start"),  # a zero-length first segment
        ((-0.0001, 500.0), "outside the head start"),  # no time left in the alignment
        # A hair under the floor, which the parser would refuse after the render.
        ((-0.0009999995, 500.0), "outside the head start"),
        ((-LEAD_S + 0.0009999995, 500.0), "outside the head start"),
        ((-1000.0, -950.0), "outside the head start"),  # under the separation
        ((-50.0, 0.0), "earlier than"),  # 50 s across the alignment's end
        ((-50.0, 30.0), "earlier than"),
    ],
)
def test_the_renderer_holds_the_head_start_walls(tmp_path: Path, times, match: str) -> None:
    with pytest.raises(PolicyError, match=match):
        _rendered(tmp_path, times)


def test_a_burn_right_after_the_alignment_follows_a_head_start_burn(tmp_path: Path) -> None:
    path, scenario, generated, result = _rendered(tmp_path, (-200.0, 0.0))
    assert result.burn_times_s == (-200.0, 0.0)
    assert_final_script_policy(path, generated, _config(), probe=scenario.state_probe)


# -- the rules' checks ----------------------------------------------------------------


def test_constraints_take_the_head_start_and_the_separation_across_its_end(
    tmp_path: Path,
) -> None:
    config = _config()
    scenario, _, _ = _load(_script(tmp_path), config)
    frame = scenario.synthesized_burn_frame
    assert validate_candidate(_candidate((-3000.0, 500.0), frame), scenario, config).valid
    early = validate_candidate(_candidate((-LEAD_S - 1.0, 500.0), frame), scenario, config)
    assert any("before mission start" in v for v in early.violations)
    close = validate_candidate(_candidate((-50.0, 30.0), frame), scenario, config)
    assert any("separation" in v for v in close.violations)
    closed = dataclasses.replace(scenario, head_start_s=0.0)
    assert not validate_candidate(_candidate((-3000.0, 500.0), frame), closed, config).valid


# -- the search ----------------------------------------------------------------------


def test_the_head_start_walls_keep_burns_off_the_alignments_end(tmp_path: Path) -> None:
    config = _config()
    scenario, _, _ = _load(_script(tmp_path), config)
    limits = RankingLimits.head_start(scenario, config)
    solver = RankingLimits.from_config(scenario, config)
    assert limits.clock_offset_s == pytest.approx(LEAD_S, abs=1e-6)
    assert limits.horizon_s == pytest.approx(solver.horizon_s + LEAD_S)
    assert limits.first_burn_floor_s == solver.render_floor_s
    impulses = np.zeros((1, 3))
    lead = limits.boundary_s
    assert lead is not None
    for inside in (lead - 1.0e-4, lead + 1.0e-4):
        assert any("alignment's end" in v for v in wall_violations([inside], impulses, limits))
        times, _ = repair([inside], impulses, limits)
        assert not wall_violations(times, impulses, limits)
        assert times[0] >= inside
    assert not wall_violations([lead], impulses, limits)
    with pytest.raises(DynamicsError, match="no head start"):
        RankingLimits.head_start(dataclasses.replace(scenario, head_start_s=0.0), config)


def test_the_head_start_search_only_adds(tmp_path: Path) -> None:
    from orbitwin.gmat_loop import ranking_search_candidates

    config = _config()
    path = _script(tmp_path)
    scenario, document, _ = _load(path, config)
    found, record = ranking_search_candidates(scenario, document, config, tmp_path, None, None)
    off = _config(head_start_burns=False)
    closed, closed_document, _ = _load(path, off)
    plain, plain_record = ranking_search_candidates(
        closed, closed_document, off, tmp_path / "off", None, None
    )
    # Every plan the search makes without the head start is still made, unchanged.
    assert [c.candidate_id for c in found if not uses_head_start(c.burns)] == [
        c.candidate_id for c in plain
    ]
    early = [c for c in found if uses_head_start(c.burns)]
    assert early and record["head_start"]["kept"] == len(early)
    assert plain_record["head_start"]["solutions"] is None
    from orbitwin.gmat_shoot import _check_frames

    for candidate in early:
        assert -LEAD_S < candidate.burns[0].time_s < 0.0
        assert candidate.branch.endswith("/head-start")
        render_candidate(document, scenario, candidate, config)
        # Its script components are on the states the correction flies it on.
        _check_frames(candidate.burns, scenario)


# -- d_min under from_origin -------------------------------------------------------------


def test_from_origin_prices_a_head_start_miss_on_its_own_alignment_rows(tmp_path: Path) -> None:
    config = _config(closest_approach_window="from_origin")
    scenario, _, _ = _load(_script(tmp_path), config)
    assert scenario.alignment_minimum_distance_km is not None
    miss = _candidate((-1000.0,), scenario.synthesized_burn_frame)
    miss.minimum_distance_km = 9000.0
    # Unmeasured, the alignment is left out rather than borrowed from the burn-free flight.
    assert scored_distance_km(miss, scenario) == 9000.0
    miss.metadata[HEAD_START_ALIGNMENT_KEY] = 12.5
    assert scored_distance_km(miss, scenario) == 12.5
    plain = _candidate((500.0,), scenario.synthesized_burn_frame)
    plain.minimum_distance_km = 9000.0
    assert scored_distance_km(plain, scenario) == min(9000.0, scenario.alignment_minimum_distance_km)
    after = dataclasses.replace(scenario, alignment_minimum_distance_km=None)
    assert scored_distance_km(miss, after) == 9000.0


# -- the diagnostic report --------------------------------------------------------------


def _report(path: Path, burn_s: float, dv: np.ndarray, *, step_s: float = 10.0) -> None:
    """Two two-body craft: SC_Human from -600 s with a burn at burn_s; SC_Alien waiting to 0."""
    mu = 398600.4418
    human_r, human_v = np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0])
    alien_r, alien_v = np.array([0.0, 7000.0, 0.0]), np.array([-7.5, 0.0, 0.0])
    times = np.arange(-600.0, 600.0 + step_s / 2, step_s)
    rows = []
    epoch = 30000.0
    for t in times:
        if t <= burn_s:
            r, v = coast(human_r, human_v, t + 600.0, mu)
        else:
            r0, v0 = coast(human_r, human_v, burn_s + 600.0, mu)
            r, v = coast(r0, v0 + dv, t - burn_s, mu)
        a, b = coast(alien_r, alien_v, max(t, 0.0), mu)
        rows.append([t + 600.0, max(t, 0.0), epoch + (t + 600.0) / 86400.0,
                     epoch + 600.0 / 86400.0 + max(t, 0.0) / 86400.0,
                     *r, *v, *a, *b, float(np.linalg.norm(r)) - 6378.1363])
    header = ["SC_Human.ElapsedSecs", "SC_Alien.ElapsedSecs", "SC_Human.A1ModJulian",
              "SC_Alien.A1ModJulian",
              *(f"SC_Human.EarthMJ2000Eq.{k}" for k in ("X", "Y", "Z", "VX", "VY", "VZ")),
              *(f"SC_Alien.EarthMJ2000Eq.{k}" for k in ("X", "Y", "Z", "VX", "VY", "VZ")),
              "SC_Human.Earth.Altitude"]
    path.write_text(" ".join(header) + "\n" + "\n".join(
        " ".join(f"{value:.17g}" for value in row) for row in rows) + "\n", encoding="utf-8")


def test_the_diagnostic_reads_a_burn_in_the_alignments_rows(tmp_path: Path) -> None:
    from orbitwin.models import EpochAlignment

    alignment = EpochAlignment("SC_Human", "SC_Alien", "Prop", "epoch-equality", 600.0, "", 0)
    dv = np.array([0.0, -0.3, 0.0])
    path = tmp_path / "diagnostic.txt"
    _report(path, -300.0, dv)
    burn = FlownManeuver(-300.0, (0.0, -0.3, 0.0), ("EarthMJ2000Eq", "Earth", "MJ2000Eq"), "B1", 1)
    event = evaluate_diagnostic_report(
        path, "SC_Human", "SC_Alien", 5.0, central_body_radius_km=6378.1363,
        maneuvers=(burn,), alignment=alignment,
    )
    # The burn lowers the orbit from -300 s on; its post-burn row keeps Hermite off it.
    mu = 398600.4418
    r0, v0 = coast(np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0]), 300.0, mu)
    times = np.linspace(0.0, 900.0, 9001)
    lowest = min(float(np.linalg.norm(coast(r0, v0 + dv, t, mu)[0])) for t in times)
    assert event.minimum_pursuer_radius_km == pytest.approx(lowest, abs=1.0e-3)
    # And it is the joint rows that are measured against the target, from t = 0.
    assert event.end_time_s == pytest.approx(600.0)
    with pytest.raises(Exception, match="no row at the burn epoch"):
        evaluate_diagnostic_report(
            path, "SC_Human", "SC_Alien", 5.0, central_body_radius_km=6378.1363,
            maneuvers=(dataclasses.replace(burn, time_s=-305.0),), alignment=alignment,
        )


# -- rank --------------------------------------------------------------------------------


def _scored(times: tuple[float, ...], score: float) -> tuple[Candidate, ScoreBreakdown]:
    candidate = _candidate(times, ("Local", "Earth", "VNB"))
    return candidate, ScoreBreakdown(50.0, score - 75.0, 25.0, 0.0, score)


@pytest.mark.parametrize(("head_start", "chosen"), [(95.10, "head-start"), (95.06, "without")])
def test_rank_takes_a_head_start_only_by_the_gain(
    tmp_path: Path, head_start: float, chosen: str
) -> None:
    config = _config()
    scenario, _, _ = _load(_script(tmp_path), config)
    early, early_score = _scored((-1000.0, 500.0), head_start)
    late, late_score = _scored((500.0,), 95.04)
    record = head_start_choice(scenario, config, [early, late],
                               {id(early): early_score, id(late): late_score})
    assert record["chosen"] == chosen
    assert record["gain_points"] == pytest.approx(head_start - 95.04)
    assert record["best_without"] == late.candidate_id
    line = head_start_line(record)
    assert ("does not take" in line) == (chosen == "without")
    assert "0.05" in line
    alone = head_start_choice(scenario, config, [early], {id(early): early_score})
    assert alone["chosen"] == "head-start" and alone["best_without"] is None


def test_head_start_plans_draw_on_their_own_gmat_budget(tmp_path: Path) -> None:
    from orbitwin.gmat_loop import correction_choices, head_start_flies, miss_choices

    config = dataclasses.replace(_config(), gmat=dataclasses.replace(
        _config().gmat, correct_top=1, verify_misses=1))
    frame = ("Local", "Earth", "VNB")
    early, late = _candidate((-1000.0, 500.0), frame), _candidate((500.0,), frame)
    for candidate in (early, late):
        candidate.successful = True
    early.score, late.score = 95.10, 95.06
    # A head-start plan ranked first never takes the plan without one off the list.
    assert correction_choices([early, late], config) == [early, late]
    # Half the gate ahead two-body, or it is not flown at all; nor before the coefficients.
    early.score = 95.08
    assert not head_start_flies([early, late], config)
    assert correction_choices([early, late], config) == [late]
    early.score = 95.10
    assert not head_start_flies([early, late], dataclasses.replace(config, score=None))
    for candidate in (early, late):
        candidate.successful = False
    early.score, late.score = 50.1, 50.0
    assert miss_choices([early, late], config, alignment_minimum_distance_km=None) == [late, early]
    early.score = 50.0
    assert miss_choices([early, late], config, alignment_minimum_distance_km=None) == [late]


def test_rank_records_the_margin_when_the_plan_without_wins(tmp_path: Path) -> None:
    config = _config()
    scenario, _, _ = _load(_script(tmp_path), config)
    early, early_score = _scored((-1000.0, 500.0), 95.049)
    late, late_score = _scored((500.0,), 95.0656)
    record = head_start_choice(scenario, config, [late, early],
                               {id(early): early_score, id(late): late_score})
    assert record["chosen"] == "without"
    assert record["gain_points"] == pytest.approx(95.049 - 95.0656)
    assert "does not take" in head_start_line(record)


def test_the_schedule_holds_head_start_segments_to_the_alignment_craft(tmp_path: Path) -> None:
    _, _, generated, _ = _rendered(tmp_path, (-1000.0, 500.0))
    edited = _with_mission(
        tmp_path, generated, "Propagate Prop(SC_Human) {SC_Human.ElapsedSecs = 4400.0};",
        "Propagate Prop(SC_Alien) {SC_Alien.ElapsedSecs = 4400.0};")
    with pytest.raises(ParseError, match="head-start"):
        mission_schedule(edited)
    bad = _with_mission(tmp_path / ".." / tmp_path.name, generated, "ElapsedSecs = 4400.0}",
                        "ElapsedSecs = 4.4.0}")
    with pytest.raises(ParseError):
        mission_schedule(bad)


def test_a_frameless_burn_of_a_head_start_plan_is_refused(tmp_path: Path) -> None:
    config = _config()
    scenario, document, _ = _load(_script(tmp_path), config)
    plan = _candidate((-1000.0, 500.0), scenario.synthesized_burn_frame)
    plan = dataclasses.replace(
        plan, burns=(plan.burns[0], dataclasses.replace(plan.burns[1], frame=None)))
    with pytest.raises(PolicyError, match="records no frame"):
        render_candidate(document, scenario, plan, config)



def test_a_head_start_plan_leaves_t_zero_from_gmats_aligned_state(tmp_path: Path) -> None:
    from orbitwin.orbit import propagate_universal
    from orbitwin.trajectory import fly_pursuer

    scenario, _, _ = _load(_script(tmp_path))
    mu = scenario.mu_km3_s2
    start = scenario.head_start_state
    assert start is not None and scenario.head_start_defect is not None
    coasted = propagate_universal(start, scenario.head_start_s, mu)
    aligned = scenario.pursuer.initial_state
    assert np.allclose(scenario.head_start_defect.r, aligned.r - coasted.r)
    assert np.allclose(scenario.head_start_defect.v, aligned.v - coasted.v)
    impulse = np.array([0.0, 0.02, 0.0])
    _, unburned = fly_pursuer(scenario, [(-1000.0, np.zeros(3))], 0.0)
    _, burned = fly_pursuer(scenario, [(-1000.0, impulse)], 0.0)
    # Unburned it arrives where GMAT's alignment leaves it; burned, plus the burn's effect.
    assert np.allclose(unburned.r, aligned.r) and np.allclose(unburned.v, aligned.v)
    pre = propagate_universal(start, scenario.head_start_s - 1000.0, mu)
    plain = propagate_universal(
        type(pre)(pre.r_km, tuple(pre.v + impulse)), 1000.0, mu)
    assert np.allclose(burned.r, aligned.r + plain.r - coasted.r, atol=1.0e-9)
    assert np.allclose(burned.v, aligned.v + plain.v - coasted.v, atol=1.0e-12)
    # Without a head-start burn nothing changes.
    before, _ = fly_pursuer(scenario, [(100.0, impulse)], 200.0)
    assert np.allclose(before[0].r, propagate_universal(aligned, 100.0, mu).r)


def test_the_head_start_search_differentiates_through_the_jump(tmp_path: Path) -> None:
    from orbitwin import ranking_search as rs

    config = _config()
    scenario, _, _ = _load(_script(tmp_path), config)
    setup = rs.build_setup(scenario, config, head_start=True)
    lead = setup.jump_s
    assert lead == pytest.approx(LEAD_S, abs=1e-6)
    impulses = np.array([[0.03, -0.02, 0.01], [-0.01, 0.04, 0.02]])
    for times in ([lead - 2000.0, lead + 300.0], [lead - 3000.0, lead - 500.0],
                  [lead + 200.0, lead + 900.0]):
        problem = rs._Problem(setup, 2, 0.0)
        x = problem.pack(times, impulses, lead + 3000.0)
        value = problem._evaluate(x)
        flown = rs.fly(times, impulses, setup.initial, setup.mu, setup.jump_at())
        aim = lead + 3000.0
        miss = np.linalg.norm(flown.state(aim)[0] - setup.target.state(aim)[0])
        assert value.miss_km == pytest.approx(float(miss), rel=1e-12)
        step = 1.0e-7
        for i in range(len(x)):
            up, down = x.copy(), x.copy()
            up[i] += step
            down[i] -= step
            high, low = problem._evaluate(up), problem._evaluate(down)
            assert value.miss_gradient[i] == pytest.approx(
                (high.miss_km - low.miss_km) / (2 * step), rel=1e-4, abs=1e-3)
            assert value.floor_gradients[:, i] == pytest.approx(
                (high.floors_km - low.floors_km) / (2 * step), rel=1e-4, abs=1e-2)


def test_rank_warns_when_a_head_start_miss_is_priced_on_its_alignment_pass(
    tmp_path: Path,
) -> None:
    from orbitwin.rank import head_start_window_warning

    config = _config(closest_approach_window="from_origin")
    scenario, _, _ = _load(_script(tmp_path), config)
    miss = _candidate((-1000.0, 500.0), ("Local", "Earth", "VNB"))
    miss.minimum_distance_km = 900.0
    assert head_start_window_warning(miss, scenario) is None  # its pass is not measured
    miss.metadata[HEAD_START_ALIGNMENT_KEY] = 12.0
    said = head_start_window_warning(miss, scenario)
    assert said is not None and "12.000 km" in said and "Q5" in said
    miss.metadata[HEAD_START_ALIGNMENT_KEY] = 950.0
    assert head_start_window_warning(miss, scenario) is None
    plain = _candidate((500.0,), ("Local", "Earth", "VNB"))
    plain.minimum_distance_km = 9000.0
    assert head_start_window_warning(plain, scenario) is None
    after = _config()
    (tmp_path / "after").mkdir()
    closed, _, _ = _load(_script(tmp_path / "after"), after)
    miss.metadata[HEAD_START_ALIGNMENT_KEY] = 12.0
    assert head_start_window_warning(miss, closed) is None


def test_a_head_start_burns_axes_stay_on_the_two_body_arc(tmp_path: Path) -> None:
    """GMAT's rows as the pre-burn state took R9's first-flight miss at arrival from
    33.76 to 39.92 km (review round 2 of #202): the axes stay on the plan's own arc."""
    from orbitwin.orbit import propagate_universal
    from orbitwin.trajectory import fly_pursuer, head_start_reference

    scenario, _, _ = _load(_script(tmp_path))
    grid, _, _ = head_start_reference(scenario)
    assert grid[0] == pytest.approx(-LEAD_S, abs=1e-6) and grid[-1] == 0.0
    start = scenario.head_start_state
    assert start is not None
    before, _ = fly_pursuer(scenario, [(-1000.0, np.array([0.0, 0.02, 0.0]))], 0.0)
    coasted = propagate_universal(start, LEAD_S - 1000.0, scenario.mu_km3_s2)
    assert np.allclose(before[0].r, coasted.r) and np.allclose(before[0].v, coasted.v)


def test_from_origin_reads_a_head_start_pass_on_gmats_rows(tmp_path: Path) -> None:
    from orbitwin.trajectory import head_start_alignment_distance_km, head_start_reference

    config = _config(closest_approach_window="from_origin")
    scenario, _, _ = _load(_script(tmp_path), config)
    grid, rows_r, _ = head_start_reference(scenario)
    target = np.asarray(scenario.target.initial_state.r)
    # A burn of nothing: the probe's own rows, as scenario.alignment_minimum_distance_km.
    nothing = (Burn(-1000.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0, ("Local", "Earth", "VNB")),)
    rows = grid < 0.0
    assert head_start_alignment_distance_km(scenario, nothing) == pytest.approx(
        float(np.linalg.norm(rows_r[rows] - target, axis=1).min()))
