"""The epoch alignment: the first mission command, flying the earlier craft alone.

See docs/ASYNC_START_AND_INSERTION.md §2. The fake console probes these
EarthMJ2000Eq templates; tests/integration/test_epoch_alignment_gmat.py runs
real GMAT on the tutorial's shape.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
from conftest import FAKE_STARTUP

from orbitwin.burn_rows import maneuvers_flown
from orbitwin.config import OrbitWinConfig, PolicyConfig, load_config
from orbitwin.ephemeris import _target_only_mission
from orbitwin.errors import ConfigurationError, ParseError, PolicyError
from orbitwin.models import Burn, Candidate, Scenario
from orbitwin.script_renderer import assert_final_script_policy, render_candidate
from orbitwin.search import zero_burn_candidate
from orbitwin.state_probe import load_scenario

ROOT = Path(__file__).resolve().parents[2]
OPEN = ROOT / "examples" / "synthetic" / "NoManeuver_Circular.script"
WORKSHOP = ROOT / "examples" / "two_burn_rendezvous.script"
FAKE = ROOT / "scripts" / "fake_gmat_console.py"
STARTUP = FAKE_STARTUP
#: AlienSC 15 h before Pursuer, as in the tutorial.
EARLY_EPOCH = "AlienSC.Epoch = '25 Jun 2026 14:00:00.000';"
ALIGN = "Propagate Prop(AlienSC) {AlienSC.A1ModJulian = Pursuer.A1ModJulian};\n"
JOINT = "Propagate Prop(Pursuer, AlienSC) {Pursuer.ElapsedSecs = 30000, AlienSC.ElapsedSecs = 30000};\n"


def _config(**policy: object) -> OrbitWinConfig:
    return OrbitWinConfig(
        policy=dataclasses.replace(PolicyConfig(), **policy),  # type: ignore[arg-type]
        pursuer_name="Pursuer",
        target_name="AlienSC",
    )


def _asynchronous(**policy: object) -> OrbitWinConfig:
    # T_max a day from AlienSC's epoch: 4 T_A is less than its 15 h head start.
    settings: dict[str, object] = {
        "mission_time_origin": "target", "closest_approach_window": "after_alignment",
        "horizon_s": 86400.0,
    }
    settings.update(policy)
    return _config(**settings)


def _script(tmp_path: Path, mission: str, *, early: bool = True, name: str = "t.script") -> Path:
    text = OPEN.read_text(encoding="utf-8")
    resources = text.split("BeginMissionSequence;")[0]
    if early:
        resources = resources.replace("AlienSC.Epoch = '26 Jun 2026 05:00:00.000';", EARLY_EPOCH)
    path = tmp_path / name
    path.write_text(resources + "BeginMissionSequence;\n" + mission, encoding="utf-8")
    return path


def _load(path: Path, config: OrbitWinConfig) -> Scenario:
    scenario, _, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    return scenario


# --- reading the three forms ------------------------------------------------------


@pytest.mark.parametrize("scale", ["A1", "TAI", "UTC", "TT", "TDB"])
def test_the_equality_form_is_an_alignment_and_not_a_segment(tmp_path: Path, scale: str) -> None:
    line = f"Propagate Prop(AlienSC) {{AlienSC.{scale}ModJulian = Pursuer.{scale}ModJulian}};\n"
    scenario = _load(_script(tmp_path, line + JOINT), _asynchronous())
    alignment = scenario.alignment
    assert alignment is not None
    assert (alignment.early, alignment.late, alignment.propagator) == ("AlienSC", "Pursuer", "Prop")
    assert alignment.form == "epoch-equality" and alignment.text == line
    assert alignment.offset_s == pytest.approx(54000.0, abs=1.0e-6)
    assert scenario.epoch_offset_s == pytest.approx(-54000.0, abs=1.0e-6)
    assert [segment.duration_s for segment in scenario.propagate_segments] == [30000.0]


def test_the_alignment_propagator_is_flown_too(tmp_path: Path) -> None:
    """A propagator only the alignment uses still sets the forces GMAT flies."""
    path = _script(tmp_path, ALIGN.replace("Prop(", "Prop2(") + JOINT)
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("BeginMissionSequence;", "Create Propagator Prop2;\nBeginMissionSequence;"),
        encoding="utf-8",
    )
    scenario = _load(path, _asynchronous())
    assert scenario.alignment is not None and scenario.alignment.propagator == "Prop2"
    assert [gap for gap in scenario.two_body_gaps if "Prop2 sets no FM" in gap]


def test_the_value_and_elapsed_forms_are_alignments(tmp_path: Path) -> None:
    pursuer = _load(_script(tmp_path, ALIGN + JOINT), _asynchronous()).pursuer.probed_epoch
    assert pursuer is not None
    for line, form in (
        (f"Propagate Prop(AlienSC) {{AlienSC.UTCModJulian = {pursuer.utc_mjd!r}}};\n", "epoch-value"),
        ("Propagate Prop(AlienSC) {AlienSC.ElapsedSecs = 54000};\n", "elapsed"),
        ("Propagate Prop(AlienSC) {AlienSC.ElapsedDays = 0.625};\n", "elapsed"),
    ):
        scenario = _load(_script(tmp_path, line + JOINT, name=f"{form}.script"), _asynchronous())
        assert scenario.alignment is not None and scenario.alignment.form == form, line
        assert len(scenario.propagate_segments) == 1


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("Propagate Prop(AlienSC) {AlienSC.ElapsedSecs = 53999};\n", r"53999.0 s, but it starts"),
        ("Propagate Prop(AlienSC) {AlienSC.UTCModJulian = 31217.1};\n", "from Pursuer's probed epoch"),
        ("Propagate Prop(AlienSC) {AlienSC.A1ModJulian = Pursuer.UTCModJulian};\n", "one scale"),
        ("Propagate Prop(AlienSC) {AlienSC.A1ModJulian = AlienSC.A1ModJulian};\n", "not the other"),
        ("Propagate Prop(AlienSC) {Pursuer.A1ModJulian = AlienSC.A1ModJulian};\n", "craft it flies"),
        # Measured on R2026a: a goal behind the craft is propagated toward forever.
        ("Propagate Prop(Pursuer) {Pursuer.A1ModJulian = AlienSC.A1ModJulian};\n", "never reaches"),
        ("Propagate Prop(Pursuer) {Pursuer.ElapsedSecs = 54000};\n", "takes the two further apart"),
        ("Propagate Prop(Ghost) {Ghost.ElapsedSecs = 54000};\n", "'Ghost', which is not one of"),
        (
            "Propagate Prop(AlienSC, Pursuer) {AlienSC.A1ModJulian = Pursuer.A1ModJulian};\n",
            "flies one craft",
        ),
        (
            "Propagate Prop(AlienSC) {AlienSC.A1ModJulian = Pursuer.A1ModJulian, "
            "AlienSC.ElapsedSecs = 9};\n",
            "and nothing else",
        ),
    ],
)
def test_an_alignment_that_does_not_end_at_the_later_epoch_is_refused(
    tmp_path: Path, line: str, message: str
) -> None:
    with pytest.raises(ParseError, match=message):
        _load(_script(tmp_path, line + JOINT), _asynchronous())


def test_an_alignment_is_read_only_as_the_first_command(tmp_path: Path) -> None:
    mission = JOINT + ALIGN
    with pytest.raises(ParseError, match="only as the first mission command"):
        _load(_script(tmp_path, mission, early=False), _config())


def test_without_a_console_an_alignment_is_refused(tmp_path: Path) -> None:
    from orbitwin.gmat_parser import parse_gmat_script

    with pytest.raises(ParseError, match=r"epoch alignment.*--gmat-console"):
        parse_gmat_script(_script(tmp_path, ALIGN + JOINT, early=False), _config())


# --- Δ: zero, ambiguous, and apart -------------------------------------------------


def test_one_start_with_an_alignment_is_a_zero_length_alignment(tmp_path: Path) -> None:
    """Bit-equal A1 epochs: GMAT ends the alignment at once (rc 0 on R2026a)."""
    scenario = _load(_script(tmp_path, ALIGN + JOINT, early=False), _config())
    assert scenario.alignment is not None and scenario.alignment.offset_s == 0.0
    # No origin to choose: nothing happens before t = 0.
    assert scenario.alignment.line_index is not None


def test_a_value_alignment_at_one_start_must_be_the_epoch_itself(tmp_path: Path) -> None:
    epoch = _load(_script(tmp_path, ALIGN + JOINT, early=False), _config()).pursuer.probed_epoch
    assert epoch is not None
    exact = f"Propagate Prop(AlienSC) {{AlienSC.A1ModJulian = {epoch.a1_mjd!r}}};\n"
    assert _load(_script(tmp_path, exact + JOINT, early=False, name="e.script"), _config()).alignment
    # 0.3 µs behind: within the tolerance, yet a goal GMAT would chase for ever.
    behind = epoch.a1_mjd - 3.5e-12
    line = f"Propagate Prop(AlienSC) {{AlienSC.A1ModJulian = {behind!r}}};\n"
    with pytest.raises(ParseError, match="exactly when the two start together"):
        _load(_script(tmp_path, line + JOINT, early=False, name="b.script"), _config())


def test_an_alignment_between_epochs_a_microsecond_apart_is_ambiguous(tmp_path: Path) -> None:
    """One A1 double step, 0.31 µs: one start without an alignment, ambiguous with one."""
    text = _script(tmp_path, ALIGN + JOINT, early=False).read_text(encoding="utf-8")
    early = 31217.708
    late = math.nextafter(early, math.inf)
    for craft, epoch in (("Pursuer", late), ("AlienSC", early)):
        text = text.replace(f"{craft}.DateFormat = UTCGregorian;", f"{craft}.DateFormat = A1ModJulian;")
        text = text.replace(f"{craft}.Epoch = '26 Jun 2026 05:00:00.000';", f"{craft}.Epoch = '{epoch!r}';")
    path = tmp_path / "step.script"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ParseError, match="ambiguous"):
        _load(path, _config())
    path.write_text(text.replace(ALIGN, ""), encoding="utf-8")
    assert _load(path, _config()).alignment is None


def test_apart_without_an_alignment_is_refused_and_names_the_permission(tmp_path: Path) -> None:
    with pytest.raises(ParseError, match=r"Spacecraft epochs do not match.*add_epoch_alignment.*Q4"):
        _load(_script(tmp_path, JOINT), _asynchronous())
    split = "Propagate Prop(Pursuer) Prop(AlienSC) {Pursuer.ElapsedSecs = 600, AlienSC.ElapsedSecs = 600};\n"
    with pytest.raises(ParseError, match=r"split Propagate.*add_epoch_alignment"):
        _load(_script(tmp_path, split, name="split.script"), _asynchronous())


def test_apart_needs_the_time_origin_and_the_closest_approach_window(tmp_path: Path) -> None:
    path = _script(tmp_path, ALIGN + JOINT)
    with pytest.raises(
        ParseError, match=r"mission_time_origin and policy.closest_approach_window.*'earliest'"
    ):
        _load(path, _config())
    with pytest.raises(
        ParseError, match=r"set policy\.closest_approach_window\. .*d_min is still asked"
    ):
        _load(path, _config(mission_time_origin="pursuer"))


def test_apart_both_craft_start_where_the_alignment_leaves_them(tmp_path: Path) -> None:
    """t = 0 is the alignment's end; the fake leaves the early craft where it was."""
    path = _script(tmp_path, ALIGN + JOINT)
    scenario = _load(path, _asynchronous())
    aligned = scenario.state_probe.aligned  # type: ignore[union-attr]
    assert aligned is not None
    assert scenario.target.initial_state == aligned.states["AlienSC"]
    assert scenario.pursuer.initial_state == aligned.states["Pursuer"]
    # T_max runs from AlienSC's epoch, 15 h before t = 0.
    own = _load(path, _asynchronous(mission_time_origin="pursuer"))
    assert scenario.time_origin_offset_s == pytest.approx(54000.0, abs=1.0e-6)
    assert own.time_origin_offset_s == 0.0
    assert own.max_time_s - scenario.max_time_s == pytest.approx(54000.0, abs=1.0e-6)
    assert scenario.alignment_minimum_distance_km is None
    window = _load(path, _asynchronous(closest_approach_window="from_origin"))
    assert window.alignment_minimum_distance_km == aligned.minimum_distance_km


def test_a_probe_without_the_alignments_end_is_not_enough(tmp_path: Path) -> None:
    from orbitwin.gmat_parser import AlignedStatesNeeded, parse_gmat_script

    path = _script(tmp_path, ALIGN + JOINT)
    probe = _load(path, _asynchronous()).state_probe
    assert probe is not None
    with pytest.raises(AlignedStatesNeeded, match="where GMAT's own run of the alignment"):
        parse_gmat_script(path, _asynchronous(), probe=dataclasses.replace(probe, aligned=None))
    # Bound to what moves the early craft: another propagator setting is refused.
    path.write_text(
        path.read_text(encoding="utf-8").replace("Prop.MaxStep = 2;", "Prop.MaxStep = 3;"),
        encoding="utf-8",
    )
    with pytest.raises(ParseError, match="alignment probe was flown on a script with another"):
        parse_gmat_script(path, _asynchronous(), probe=probe)


def test_an_offset_that_reaches_t_max_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ParseError, match="it is reached before the two fly together"):
        _load(_script(tmp_path, ALIGN + JOINT), _asynchronous(horizon_s=54000.0))
    # From the later craft's epoch nothing before t = 0 counts.
    late = _load(_script(tmp_path, ALIGN + JOINT, name="late.script"),
                 _asynchronous(horizon_s=54000.0, mission_time_origin="pursuer"))
    assert late.max_time_s == 54000.0


@pytest.mark.parametrize(
    ("field", "value"),
    [("mission_time_origin", "alien"), ("closest_approach_window", "always")],
)
def test_the_two_settings_take_only_their_words(field: str, value: str, tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(f"policy:\n  {field}: {value}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=field):
        load_config(path)


# --- the counting rule -------------------------------------------------------------


def _slotted(tmp_path: Path, *, early: bool, first: str) -> Path:
    """The workshop template with an alignment, then `first` before its first burn."""
    text = WORKSHOP.read_text(encoding="utf-8")
    if early:
        epoch = "AlienSC.Epoch = '15 Mar 2027 12:00:00.000';"
        assert epoch in text
        text = text.replace(epoch, "AlienSC.Epoch = '14 Mar 2027 21:00:00.000';")
    head, _, mission = text.partition("BeginMissionSequence;\n")
    lines = mission.splitlines(keepends=True)
    align = "Propagate DefaultProp(AlienSC) {AlienSC.UTCModJulian = HumanSC.UTCModJulian};\n"
    # P0, M1, P1, M2, P2; without P0 the first burn follows the alignment.
    body = [align, *lines] if first == "propagate" else [align, *lines[1:]]
    path = tmp_path / f"slotted_{first}.script"
    path.write_text(head + "BeginMissionSequence;\n" + "".join(body), encoding="utf-8")
    return path


def test_a_burn_right_after_the_alignment_has_an_implicit_zero_segment(tmp_path: Path) -> None:
    path = _slotted(tmp_path, early=False, first="maneuver")
    scenario = _load(path, OrbitWinConfig())
    assert len(scenario.burn_slots) == 2
    first, *rest = scenario.propagate_segments
    assert first.implicit and first.duration_s == 0.0 and first.duration_spans == ()
    assert [segment.implicit for segment in rest] == [False, False]
    assert [maneuver.time_s for maneuver in maneuvers_flown(scenario)] == [0.0, 3728.5]


def test_the_implicit_segment_fixes_slot_zero_at_t_zero(tmp_path: Path) -> None:
    path = _slotted(tmp_path, early=False, first="maneuver")
    config = OrbitWinConfig()
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    zero = zero_burn_candidate(scenario, config)
    rendered = render_candidate(document, scenario, zero, config)
    assert rendered.burn_times_s[0] == 0.0 and rendered.segment_durations_s[0] == 0.0
    late = Candidate(
        burns=(Burn(50.0, (0.01, 0.0, 0.0), (0.01, 0.0, 0.0), 0, ("Local", "Earth", "VNB")),),
        intercept_time_s=None, minimum_distance_km=None, distance_at_completion_km=None,
        successful=False, analytical_endpoint_error_km=None, branch="test",
    )
    with pytest.raises(PolicyError, match="no Propagate between them to move it"):
        render_candidate(document, scenario, late, config)


# --- the renderer and the policy guard -----------------------------------------------


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _burn_at(time_s: float) -> Candidate:
    return Candidate(
        burns=(Burn(time_s, (0.01, 0.0, 0.0), (0.01, 0.0, 0.0), 0, ("Local", "Earth", "VNB")),),
        intercept_time_s=None, minimum_distance_km=None, distance_at_completion_km=None,
        successful=False, analytical_endpoint_error_km=None, branch="test",
    )


def test_the_renderer_keeps_the_alignment_byte_for_byte(tmp_path: Path) -> None:
    config = _config()
    for path in (
        _script(tmp_path, ALIGN + JOINT, early=False),
        _slotted(tmp_path, early=False, first="propagate"),
        _slotted(tmp_path, early=False, first="maneuver"),
    ):
        local = OrbitWinConfig() if "slotted" in path.name else config
        scenario, document, _ = load_scenario(path, local, gmat_console=FAKE, startup_file=STARTUP)
        assert scenario.alignment is not None and scenario.alignment.line_index is not None
        candidate = zero_burn_candidate(scenario, local) if "slotted" in path.name else _burn_at(600.0)
        rendered = render_candidate(document, scenario, candidate, local)
        out = _write(tmp_path, f"out_{path.name}", rendered.script_bytes)
        mission = out.read_text(encoding="utf-8").split("BeginMissionSequence;\n")[1]
        assert mission.splitlines(keepends=True)[0] == scenario.alignment.text
        assert_final_script_policy(path, out, local, probe=scenario.state_probe)


def test_an_open_template_may_burn_at_t_zero_after_its_alignment(tmp_path: Path) -> None:
    path = _script(tmp_path, ALIGN + JOINT, early=False)
    config = _config()
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    rendered = render_candidate(document, scenario, _burn_at(0.0), config)
    out = _write(tmp_path, "zero.script", rendered.script_bytes)
    mission = out.read_text(encoding="utf-8").split("BeginMissionSequence;\n")[1].splitlines()
    assert mission[0] == ALIGN.strip() and mission[1].startswith("Maneuver ")
    assert_final_script_policy(path, out, config, probe=scenario.state_probe)
    flown = _load(out, config)
    assert flown.propagate_segments[0].implicit
    # Inside the floor that keeps GMAT from hanging, it is still refused.
    with pytest.raises(PolicyError, match="floor"):
        render_candidate(document, scenario, _burn_at(5.0e-4), config)


def test_without_an_alignment_t_zero_is_still_refused(tmp_path: Path) -> None:
    path = _script(tmp_path, JOINT, early=False)
    config = _config()
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    with pytest.raises(PolicyError, match="floor"):
        render_candidate(document, scenario, _burn_at(0.0), config)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda text: text.replace(ALIGN, ""), "drops the epoch alignment"),
        (
            lambda text: text.replace(ALIGN, ALIGN.replace("A1ModJulian", "TAIModJulian")),
            "changes the epoch alignment",
        ),
        (lambda text: text.replace(ALIGN, ALIGN.replace("};", "};  % ours")), "changes the epoch"),
    ],
)
def test_the_guard_refuses_an_alignment_that_moved(tmp_path: Path, edit, message: str) -> None:
    path = _script(tmp_path, ALIGN + JOINT, early=False)
    config = _config()
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    rendered = render_candidate(document, scenario, _burn_at(600.0), config)
    out = tmp_path / "edited.script"
    out.write_text(edit(rendered.script_bytes.decode("utf-8")), encoding="utf-8")
    with pytest.raises(PolicyError, match=message):
        assert_final_script_policy(path, out, config, probe=scenario.state_probe)


def test_the_guard_refuses_an_alignment_the_original_lacks(tmp_path: Path) -> None:
    path = _script(tmp_path, JOINT, early=False)
    config = _config()
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    rendered = render_candidate(document, scenario, _burn_at(600.0), config)
    out = tmp_path / "added.script"
    out.write_text(
        rendered.script_bytes.decode("utf-8").replace(
            "BeginMissionSequence;\n", "BeginMissionSequence;\n" + ALIGN
        ),
        encoding="utf-8",
    )
    with pytest.raises(PolicyError, match="that the original does not need"):
        assert_final_script_policy(path, out, config, probe=scenario.state_probe)


def test_the_guard_refuses_a_slotted_segment_dropped_for_a_burn_at_zero(tmp_path: Path) -> None:
    """Removing a slotted Propagate would otherwise read as the implicit segment."""
    path = _slotted(tmp_path, early=False, first="propagate")
    config = OrbitWinConfig()
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    rendered = render_candidate(document, scenario, zero_burn_candidate(scenario, config), config)
    lines = rendered.script_bytes.decode("utf-8").splitlines(keepends=True)
    begin = lines.index("BeginMissionSequence;\n")
    # Alignment, P0, M1, P1, M2, P2 -> alignment, M1, P1 (+P0's time), M2, P2.
    del lines[begin + 2]
    out = tmp_path / "dropped.script"
    out.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(PolicyError, match="segment"):
        assert_final_script_policy(path, out, config, probe=scenario.state_probe)


def test_add_epoch_alignment_adds_the_a1_form_and_nothing_else(tmp_path: Path) -> None:
    path = _script(tmp_path, JOINT)
    config = _asynchronous(add_epoch_alignment=True)
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    alignment = scenario.alignment
    assert alignment is not None and alignment.line_index is None
    assert alignment.text == ALIGN
    rendered = render_candidate(document, scenario, _burn_at(600.0), config)
    out = _write(tmp_path, "added.script", rendered.script_bytes)
    mission = out.read_text(encoding="utf-8").split("BeginMissionSequence;\n")[1]
    assert mission.startswith(ALIGN)
    assert_final_script_policy(path, out, config, probe=scenario.state_probe)
    with pytest.raises(ParseError, match="add_epoch_alignment"):
        assert_final_script_policy(
            path, out, _asynchronous(), probe=scenario.state_probe
        )


def test_add_epoch_alignment_leaves_a_synchronous_template_alone(tmp_path: Path) -> None:
    path = _script(tmp_path, JOINT, early=False)
    config = _config(add_epoch_alignment=True)
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    assert scenario.alignment is None
    rendered = render_candidate(document, scenario, _burn_at(600.0), config)
    out = _write(tmp_path, "sync.script", rendered.script_bytes)
    assert "ModJulian" not in out.read_text(encoding="utf-8").split("BeginMissionSequence;\n")[1]
    assert_final_script_policy(path, out, config, probe=scenario.state_probe)


def test_add_epoch_alignment_keeps_the_templates_own_alignment(tmp_path: Path) -> None:
    own = "Propagate Prop(AlienSC) {AlienSC.UTCModJulian = Pursuer.UTCModJulian};\n"
    path = _script(tmp_path, own + JOINT)
    config = _asynchronous(add_epoch_alignment=True)
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    assert scenario.alignment is not None and scenario.alignment.line_index is not None
    rendered = render_candidate(document, scenario, _burn_at(600.0), config)
    out = _write(tmp_path, "own.script", rendered.script_bytes)
    mission = out.read_text(encoding="utf-8").split("BeginMissionSequence;\n")[1]
    assert mission.startswith(own)
    assert mission.count("ModJulian") == 2  # the template's one line, both sides
    assert_final_script_policy(path, out, config, probe=scenario.state_probe)


# --- the coast run -----------------------------------------------------------------


def test_the_coast_run_replays_the_alignment_verbatim(tmp_path: Path) -> None:
    path = _slotted(tmp_path, early=False, first="maneuver")
    config = OrbitWinConfig()
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    rendered = render_candidate(document, scenario, zero_burn_candidate(scenario, config), config)
    out = _write(tmp_path, "coast.script", rendered.script_bytes)
    coast = _load(out, config)
    mission = _target_only_mission(rendered.script_bytes, coast).decode("utf-8")
    lines = mission.split("BeginMissionSequence;\n")[1].splitlines()
    assert lines[0] == scenario.alignment.text.strip()  # type: ignore[union-attr]
    assert lines[1].startswith("% Maneuver")
    assert lines[2].startswith("Propagate DefaultProp(AlienSC) {AlienSC.ElapsedSecs = ")


# --- what the guard keeps of each Propagate ------------------------------------------

R3 = ROOT / "examples" / "ranking" / "R3_FastHighPerigee_TwoPropagators.script"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        # Each craft on the other's propagator.
        ("DefaultProp(SC_Human) AlienProp(SC_Alien)", "AlienProp(SC_Human) DefaultProp(SC_Alien)"),
        ("Propagate Synchronized ", "Propagate "),
        ("{SC_Human.ElapsedSecs", "{SC_Alien.ElapsedSecs"),
    ],
)
def test_the_guard_refuses_a_propagate_changed_beyond_its_duration(
    tmp_path: Path, old: str, new: str
) -> None:
    # The parser's own synchronisation rule off, so the shape check alone decides.
    config = OrbitWinConfig(
        policy=dataclasses.replace(
            PolicyConfig(), rules="ranking", horizon_s=5092.815,
            require_synchronized_propagation=False,
        ),
        pursuer_name="SC_Human",
        target_name="SC_Alien",
    )
    scenario, document, _ = load_scenario(R3, config, gmat_console=FAKE, startup_file=STARTUP)
    rendered = render_candidate(document, scenario, zero_burn_candidate(scenario, config), config)
    text = rendered.script_bytes.decode("utf-8")
    assert old in text
    out = tmp_path / "changed.script"
    out.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(PolicyError, match="not the template's own with other durations"):
        assert_final_script_policy(R3, out, config, probe=scenario.state_probe)
    out.write_text(text, encoding="utf-8")
    assert_final_script_policy(R3, out, config, probe=scenario.state_probe)


def test_the_guard_refuses_a_slotted_propagator_swap(tmp_path: Path) -> None:
    config = OrbitWinConfig()
    scenario, document, _ = load_scenario(WORKSHOP, config, gmat_console=FAKE, startup_file=STARTUP)
    rendered = render_candidate(document, scenario, zero_burn_candidate(scenario, config), config)
    text = rendered.script_bytes.decode("utf-8")
    old = "Propagate DefaultProp(HumanSC) DefaultProp(AlienSC) {"
    assert old in text
    out = tmp_path / "swapped.script"
    out.write_text(
        text.replace(old, "Propagate DefaultProp(HumanSC, AlienSC) {"), encoding="utf-8"
    )
    with pytest.raises(PolicyError, match="other than its durations"):
        assert_final_script_policy(WORKSHOP, out, config, probe=scenario.state_probe)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("Propagate Prop(Ghost) {Ghost.A1ModJulian = Pursuer.A1ModJulian};\n", "not one of the"),
        (
            "Propagate NoSuchProp(AlienSC) {AlienSC.A1ModJulian = Pursuer.A1ModJulian};\n",
            r"\['NoSuchProp'\], which the script does not create",
        ),
    ],
)
def test_an_alignment_naming_what_the_script_lacks_is_refused(
    tmp_path: Path, line: str, message: str
) -> None:
    with pytest.raises(ParseError, match=message):
        _load(_script(tmp_path, line + JOINT, early=False), _config())


def test_a_segment_on_an_undeclared_propagator_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ParseError, match="does not create as a Propagator"):
        _load(_script(tmp_path, JOINT.replace("Prop(", "Other("), early=False), _config())


def test_changed_lines_point_past_an_added_alignment(tmp_path: Path) -> None:
    path = _slotted(tmp_path, early=True, first="propagate")
    head, _, mission = path.read_text(encoding="utf-8").partition("BeginMissionSequence;\n")
    path.write_text(head + "BeginMissionSequence;\n" + mission.split("\n", 1)[1], encoding="utf-8")
    config = dataclasses.replace(
        OrbitWinConfig(),
        policy=dataclasses.replace(
            PolicyConfig(), add_epoch_alignment=True, mission_time_origin="pursuer",
            closest_approach_window="after_alignment",
        ),
    )
    scenario, document, _ = load_scenario(path, config, gmat_console=FAKE, startup_file=STARTUP)
    rendered = render_candidate(document, scenario, zero_burn_candidate(scenario, config), config)
    lines = rendered.script_bytes.decode("utf-8").splitlines()
    for index in rendered.changed_lines:
        assert lines[index].startswith(("Propagate DefaultProp(HumanSC", "ImpulsiveBurn")), (
            lines[index]
        )


def test_a_maneuver_line_carrying_another_command_is_refused(tmp_path: Path) -> None:
    """R2026a: exit 1, "Maneuver command is malformed"."""
    path = _slotted(tmp_path, early=False, first="propagate")
    text = path.read_text(encoding="utf-8").replace(
        "Maneuver ImpulsiveBurn1(HumanSC);",
        "Maneuver ImpulsiveBurn1(HumanSC); Propagate DefaultProp(HumanSC) {HumanSC.ElapsedSecs = 5};",
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ParseError, match="Maneuver the parser cannot read"):
        _load(path, OrbitWinConfig())
    path.write_text(
        text.replace(
            "Maneuver ImpulsiveBurn1(HumanSC); Propagate DefaultProp(HumanSC) {HumanSC.ElapsedSecs = 5};",
            "Maneuver ImpulsiveBurn1(HumanSC);  % a comment is fine",
        ),
        encoding="utf-8",
    )
    assert len(_load(path, OrbitWinConfig()).burn_slots) == 2
