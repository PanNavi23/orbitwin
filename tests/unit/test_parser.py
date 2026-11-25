from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, load_config
from orbitwin.errors import ParseError
from orbitwin.gmat_parser import GMAT_RESERVED_RESOURCE_NAMES, parse_gmat_script


def test_workshop_parser_infers_roles_and_timeline(workshop_script: Path) -> None:
    scenario, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    assert scenario.pursuer.name == "HumanSC"
    assert scenario.target.name == "AlienSC"
    assert [slot.name for slot in scenario.burn_slots] == ["ImpulsiveBurn1", "ImpulsiveBurn2"]
    assert [segment.duration_s for segment in scenario.propagate_segments] == [1000.0, 3728.5, 1000.0]
    assert scenario.target_period_s > 9300.0
    assert scenario.max_time_s == pytest.approx(4.0 * scenario.target_period_s)


def test_different_plane_parser_preserves_3d_geometry(different_plane_script: Path) -> None:
    scenario, _ = parse_gmat_script(different_plane_script, OrbitWinConfig())
    assert scenario.pursuer.fields["INC"] == "28"
    assert scenario.target.fields["INC"] == "43"
    assert scenario.pursuer.fields["RAAN"] == "15"
    assert scenario.target.fields["RAAN"] == "65"
    assert scenario.pursuer.initial_state.r_km != scenario.target.initial_state.r_km


def test_include_macro_is_rejected(tmp_path: Path, workshop_script: Path) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    path = tmp_path / "include.script"
    path.write_text("#Include 'hidden.script'\n" + text, encoding="utf-8")
    with pytest.raises(ParseError, match="Include"):
        parse_gmat_script(path, OrbitWinConfig())


def test_target_maneuver_is_rejected(tmp_path: Path, workshop_script: Path) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    text = text.replace("Maneuver ImpulsiveBurn2(HumanSC);", "Maneuver ImpulsiveBurn2(AlienSC);")
    path = tmp_path / "target-burn.script"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ParseError):
        parse_gmat_script(path, OrbitWinConfig())


def test_duplicate_assignment_is_rejected(tmp_path: Path, workshop_script: Path) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    text = text.replace(
        "HumanSC.SMA = 6900;",
        "HumanSC.SMA = 6900;\nHumanSC.SMA = 7001;",
    )
    path = tmp_path / "duplicate.script"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ParseError, match="Duplicate assignment"):
        parse_gmat_script(path, OrbitWinConfig())


def test_unsynchronized_propagation_is_rejected(tmp_path: Path, workshop_script: Path) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    text = text.replace(
        "Propagate DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000, AlienSC.ElapsedSecs = 1000, OrbitColor = [0 128 255]};",
        "Propagate DefaultProp(HumanSC) {HumanSC.ElapsedSecs = 1000};",
        1,
    )
    path = tmp_path / "unsync.script"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ParseError, match="synchronize"):
        parse_gmat_script(path, OrbitWinConfig())


def test_reserved_resource_name_is_rejected(tmp_path: Path, different_plane_script: Path) -> None:
    # GMAT answers "Spacecraft object can not be named to a Command type" and
    # aborts before the mission sequence starts, so a script naming a resource
    # after a command is unrunnable no matter how sound its dynamics are.
    text = different_plane_script.read_text(encoding="utf-8").replace("AlienSC", "Target")
    script = tmp_path / "reserved.script"
    script.write_text(text, encoding="utf-8")
    with pytest.raises(ParseError, match="reserves that identifier"):
        parse_gmat_script(script, OrbitWinConfig())


def test_bundled_scripts_avoid_reserved_resource_names(project_root: Path) -> None:
    import re

    pattern = re.compile(r"^\s*(?:GMAT\s+)?Create\s+[A-Za-z_]\w*\s+([^;]+);", re.MULTILINE)
    scripts = sorted((project_root / "examples").rglob("*.script"))
    assert scripts, "no bundled scripts were found"
    for script in scripts:
        for names in pattern.findall(script.read_text(encoding="utf-8", errors="replace")):
            for name in names.split():
                assert name not in GMAT_RESERVED_RESOURCE_NAMES, f"{script.name}: {name}"


def test_reserved_name_matching_is_case_sensitive(
    tmp_path: Path, different_plane_script: Path
) -> None:
    # Measured against stock GmatConsole R2026a: "Target" is rejected while
    # "target", "TARGET" and "TaRgEt" are all accepted. Making this check
    # case-insensitive would reject scripts GMAT runs happily, so the exact-case
    # comparison is the behaviour and not an oversight.
    text = different_plane_script.read_text(encoding="utf-8")
    for spelling in ("target", "TARGET", "TaRgEt"):
        assert spelling not in GMAT_RESERVED_RESOURCE_NAMES
        script = tmp_path / f"{spelling}.script"
        script.write_text(text.replace("AlienSC", spelling), encoding="utf-8")
        scenario, _ = parse_gmat_script(script, OrbitWinConfig())
        assert scenario.target.name == spelling


def test_reserved_names_are_rejected_for_every_resource_type(
    tmp_path: Path, different_plane_script: Path
) -> None:
    # Also measured: GMAT answers "<type> object can not be named to a Command
    # type" for ImpulsiveBurn, ReportFile, Variable and CoordinateSystem alike,
    # so the guard is correctly independent of the resource type.
    text = different_plane_script.read_text(encoding="utf-8")
    script = tmp_path / "burn.script"
    script.write_text(text.replace("Burn1", "Vary"), encoding="utf-8")
    with pytest.raises(ParseError, match="reserves that identifier"):
        parse_gmat_script(script, OrbitWinConfig())


def test_reserved_name_is_caught_among_several_names_on_one_create_line(
    tmp_path: Path, different_plane_script: Path
) -> None:
    text = different_plane_script.read_text(encoding="utf-8")
    script = tmp_path / "multi.script"
    script.write_text(text.replace("Create Spacecraft AlienSC;", "Create Spacecraft AlienSC Report;"), encoding="utf-8")
    with pytest.raises(ParseError, match="reserves that identifier"):
        parse_gmat_script(script, OrbitWinConfig())


# GMAT R2026a accepts both `Create Type a, b;` and `Create Type a b;` -- each
# was run to check. Splitting the name list on whitespace alone left the comma
# attached to the first name, and everything downstream then described an
# object that does not exist.


def _rewritten(script: Path, tmp_path: Path, find: str, replace: str, drop: str = "") -> Path:
    text = script.read_text(encoding="utf-8-sig")
    assert find in text, find
    text = text.replace(find, replace, 1)
    if drop:
        assert drop in text, drop
        text = text.replace(drop, "", 1)
    out = tmp_path / "commas.script"
    out.write_text(text, encoding="ascii")
    return out


def test_a_comma_separated_create_declares_every_name(
    tmp_path: Path, different_plane_script: Path
) -> None:
    baseline, _ = parse_gmat_script(different_plane_script, OrbitWinConfig())
    rewritten = _rewritten(
        different_plane_script, tmp_path,
        "Create ImpulsiveBurn Burn1;\n", "Create ImpulsiveBurn Burn1, Burn2;\n",
        drop="Create ImpulsiveBurn Burn2;\n",
    )
    scenario, _ = parse_gmat_script(rewritten, OrbitWinConfig())
    assert [slot.name for slot in scenario.burn_slots] == [
        slot.name for slot in baseline.burn_slots
    ]


def test_the_spacecraft_may_be_declared_on_one_line(
    tmp_path: Path, different_plane_script: Path
) -> None:
    # This used to fail with "Missing required assignment Pursuer,.Epoch",
    # which is the trailing comma leaking into the error.
    rewritten = _rewritten(
        different_plane_script, tmp_path,
        "Create Spacecraft Pursuer;\n", "Create Spacecraft Pursuer, AlienSC;\n",
        drop="Create Spacecraft AlienSC;\n",
    )
    scenario, _ = parse_gmat_script(rewritten, OrbitWinConfig())
    assert scenario.pursuer.name == "Pursuer"
    assert scenario.target.name == "AlienSC"


def test_a_reserved_name_cannot_hide_in_a_comma_list(
    tmp_path: Path, different_plane_script: Path
) -> None:
    """The reserved-name guard was walked past by putting the name first.

    It only ever saw the comma-suffixed spelling, which matches nothing, so a
    script GMAT refuses to interpret parsed cleanly here.
    """
    rewritten = _rewritten(
        different_plane_script, tmp_path,
        "Create ImpulsiveBurn Burn1;\n",
        "Create Variable Target, helper;\nCreate ImpulsiveBurn Burn1;\n",
    )
    with pytest.raises(ParseError, match="reserves that identifier"):
        parse_gmat_script(rewritten, OrbitWinConfig())


def test_whitespace_separated_names_still_work(
    tmp_path: Path, different_plane_script: Path
) -> None:
    rewritten = _rewritten(
        different_plane_script, tmp_path,
        "Create ImpulsiveBurn Burn1;\n", "Create ImpulsiveBurn Burn1 Burn2;\n",
        drop="Create ImpulsiveBurn Burn2;\n",
    )
    scenario, _ = parse_gmat_script(rewritten, OrbitWinConfig())
    assert [slot.name for slot in scenario.burn_slots] == ["Burn1", "Burn2"]


_FIRST_PROPAGATE = (
    "Propagate DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000, "
    "AlienSC.ElapsedSecs = 1000, OrbitColor = [0 128 255]};"
)


def test_workshop_epoch_locks(workshop_script: Path) -> None:
    scenario, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    locks = [segment.epoch_lock for segment in scenario.propagate_segments]
    assert locks == ["propagator", "equal-goals", "propagator"]


# Every form below ran on GMAT R2026a with both craft ending on one epoch.
@pytest.mark.parametrize(
    "line, lock",
    [
        # A propagator of the target's own, as R3 in examples/ranking has.
        ("Propagate Synchronized DefaultProp(HumanSC) DefaultProp(AlienSC) "
         "{HumanSC.ElapsedSecs = 1000};", "synchronized"),
        ("Propagate DefaultProp(HumanSC, AlienSC) {AlienSC.ElapsedSecs = 1000};", "propagator"),
        ("Propagate DefaultProp(HumanSC, AlienSC, {HumanSC.ElapsedSecs = 1000});", "propagator"),
        ("Propagate 'coast (1) {x}' DefaultProp(HumanSC, AlienSC) "
         "{HumanSC.ElapsedSecs = 1000};", "propagator"),
        ("Propagate DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000}; "
         "% AlienSC.ElapsedSecs = 5", "propagator"),
        # The label is blanked before the comment is cut, or `%` would cut it.
        ("Propagate 'a % b' DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000};",
         "propagator"),
        ("Propagate DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000};;", "propagator"),
        # Options, not stop conditions: R2026a still ended at 1000 s with a 500 s tolerance.
        ("Propagate DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000, "
         "StopTolerance = 500};", "propagator"),
        ("Propagate DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000, "
         "OrbitColor = Yellow};", "propagator"),
    ],
)
def test_propagate_forms_that_hold_one_epoch(
    tmp_path: Path, workshop_script: Path, line: str, lock: str
) -> None:
    script = _rewritten(workshop_script, tmp_path, _FIRST_PROPAGATE, line)
    scenario, document = parse_gmat_script(script, OrbitWinConfig())
    segment = scenario.propagate_segments[0]
    assert set(segment.spacecraft_names) == {"HumanSC", "AlienSC"}
    assert segment.epoch_lock == lock
    assert segment.duration_s == 1000.0
    # The renderer rewrites these spans, so they must index the original line.
    text = document.lines[segment.line_index]
    assert [text[start:end] for start, end in segment.duration_spans] == ["1000"]


@pytest.mark.parametrize(
    "line, message",
    [
        # Whether AlienSC keeps pace depends on step control the line cannot show:
        # R2026a ended it 0 s off at this fixture's MaxStep 1 s, 189 s off at 60 s.
        ("Propagate DefaultProp(HumanSC) DefaultProp(AlienSC) {HumanSC.ElapsedSecs = 1000};",
         "cannot bound"),
        # GMAT R2026a refuses a second group outright.
        ("Propagate DefaultProp(HumanSC) DefaultProp(AlienSC) {HumanSC.ElapsedSecs = 1000} "
         "{AlienSC.ElapsedSecs = 1000};", "stop-condition groups"),
        ("Propagate BackProp DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000};",
         "BackProp"),
        ("Propagate synchronized DefaultProp(HumanSC) DefaultProp(AlienSC) "
         "{HumanSC.ElapsedSecs = 1000};", "unsupported mode"),
        ("Propagate DefaultProp(HumanSC) {HumanSC.ElapsedSecs = 1000, "
         "AlienSC.ElapsedSecs = 1000};", "does not propagate"),
        ("Propagate DefaultProp(HumanSC, HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000};",
         "more than once"),
        ("Propagate DefaultProp(HumanSC) and DefaultProp(AlienSC) {HumanSC.ElapsedSecs = 1000, "
         "AlienSC.ElapsedSecs = 1000};", "unsupported text"),
        ("Propagate DefaultProp(HumanSC, 3) {HumanSC.ElapsedSecs = 1000};",
         "not a spacecraft name"),
        ("Propagate {HumanSC.ElapsedSecs = 1000};", "names no propagator"),
        ("Propagate DefaultProp(HumanSC, AlienSC); % {HumanSC.ElapsedSecs = 1000}",
         "no explicit"),
        # GMAT stops at whichever condition comes first, which the parser cannot know.
        ("Propagate DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000, "
         "AlienSC.ElapsedSecs = 1000, HumanSC.Earth.TA = 330};", "not an elapsed-time goal"),
        ("Propagate DefaultProp(HumanSC, AlienSC, {HumanSC.ElapsedSecs = 1000, "
         "HumanSC.Earth.RMAG = 6500});", "not an elapsed-time goal"),
        ("Propagate DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000, "
         "HumanSC.Periapsis};", "not an elapsed-time goal"),
        # GMAT reads option names case-sensitively and refuses this one.
        ("Propagate DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000, "
         "orbitcolor = Yellow};", "not an elapsed-time goal"),
    ],
)
def test_propagate_forms_refused(
    tmp_path: Path, workshop_script: Path, line: str, message: str
) -> None:
    script = _rewritten(workshop_script, tmp_path, _FIRST_PROPAGATE, line)
    with pytest.raises(ParseError, match=message):
        parse_gmat_script(script, OrbitWinConfig())


def test_unsynchronized_propagation_can_be_allowed_explicitly(
    tmp_path: Path, workshop_script: Path
) -> None:
    script = _rewritten(
        workshop_script, tmp_path, _FIRST_PROPAGATE,
        "Propagate DefaultProp(HumanSC) DefaultProp(AlienSC) {HumanSC.ElapsedSecs = 1000};",
    )
    base = OrbitWinConfig()
    config = dataclasses.replace(
        base, policy=dataclasses.replace(base.policy, require_synchronized_propagation=False)
    )
    scenario, _ = parse_gmat_script(script, config)
    assert scenario.propagate_segments[0].epoch_lock is None


@pytest.mark.parametrize("scale", ["TAIGregorian", "TTGregorian", "A1Gregorian"])
def test_one_epoch_text_in_two_time_scales_is_refused(
    workshop_script: Path, tmp_path: Path, scale: str
) -> None:
    """The same text in UTC and TAI is 37 s apart, and the text check saw one epoch."""
    text = workshop_script.read_text(encoding="utf-8")
    edited = text.replace("AlienSC.DateFormat = UTCGregorian;", f"AlienSC.DateFormat = {scale};")
    assert edited != text
    path = tmp_path / "scales.script"
    path.write_text(edited, encoding="utf-8")
    with pytest.raises(ParseError, match=f"different time scales \\(UTCGregorian and {scale}\\)"):
        parse_gmat_script(path, OrbitWinConfig())


def test_an_omitted_date_format_is_gmats_default(workshop_script: Path, tmp_path: Path) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    edited = (
        text.replace("HumanSC.DateFormat = UTCGregorian;", "HumanSC.DateFormat = TAIModJulian;")
        .replace("AlienSC.DateFormat = UTCGregorian;", "")
        .replace("'15 Mar 2027 12:00:00.000'", "'30000'")
    )
    path = tmp_path / "default.script"
    path.write_text(edited, encoding="utf-8")
    scenario, _ = parse_gmat_script(path, OrbitWinConfig())
    assert scenario.pursuer.epoch_text == scenario.target.epoch_text == "30000"


def test_an_epoch_read_before_its_date_format_is_refused(
    workshop_script: Path, tmp_path: Path
) -> None:
    """R2026a read that Epoch as TAIModJulian, 37 s from the other craft's UTC."""
    text = workshop_script.read_text(encoding="utf-8")
    swapped = text.replace(
        "AlienSC.DateFormat = UTCGregorian;\nAlienSC.Epoch = '15 Mar 2027 12:00:00.000';",
        "AlienSC.Epoch = '15 Mar 2027 12:00:00.000';\nAlienSC.DateFormat = UTCGregorian;",
    )
    assert swapped != text
    path = tmp_path / "order.script"
    path.write_text(swapped, encoding="utf-8")
    with pytest.raises(ParseError, match=r"AlienSC\.Epoch on line \d+ is read before"):
        parse_gmat_script(path, OrbitWinConfig())


def test_an_epoch_in_a_scale_of_its_own_is_refused(workshop_script: Path, tmp_path: Path) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    marker = "AlienSC.Epoch = '15 Mar 2027 12:00:00.000';"
    path = tmp_path / "scaled.script"
    path.write_text(
        text.replace(marker, f"{marker}\nAlienSC.Epoch.TAIGregorian = '15 Mar 2027 12:00:00.000';"),
        encoding="utf-8",
    )
    with pytest.raises(ParseError, match=r"AlienSC\.Epoch\.TAIGregorian sets the epoch"):
        parse_gmat_script(path, OrbitWinConfig())


@pytest.mark.parametrize(
    ("command", "message"),
    [
        # A Toggle of plots is read (test_plot_toggle.py); of a ReportFile it is not.
        ("Toggle ReportFile1 Off;", "only a Toggle of plots is read"),
        ("Report ReportFile1 HumanSC.ElapsedSecs;", "neither a Propagate nor a Maneuver"),
        ("HumanSC.Epoch = '15 Mar 2027 13:00:00.000';", "neither a Propagate nor a Maneuver"),
        ("Stop;", "neither a Propagate nor a Maneuver"),
    ],
)
def test_an_unmodelled_mission_command_is_refused(
    workshop_script: Path, tmp_path: Path, command: str, message: str
) -> None:
    """Skipped silently before: Toggle Off left a certified upload ending at the last burn."""
    text = workshop_script.read_text(encoding="utf-8")
    marker = "Maneuver ImpulsiveBurn2(HumanSC);"
    assert marker in text
    path = tmp_path / "extra.script"
    path.write_text(text.replace(marker, f"{marker}\n{command}"), encoding="utf-8")
    with pytest.raises(ParseError, match=message):
        parse_gmat_script(path, OrbitWinConfig())


def test_comments_and_blank_lines_in_the_mission_sequence_are_fine(
    workshop_script: Path, tmp_path: Path
) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    marker = "Maneuver ImpulsiveBurn2(HumanSC);"
    path = tmp_path / "quiet.script"
    path.write_text(text.replace(marker, f"{marker}\n\n   % a note\n"), encoding="utf-8")
    scenario, _ = parse_gmat_script(path, OrbitWinConfig())
    assert len(scenario.burn_slots) == 2


_FM = "DefaultProp_ForceModel"
_PRIMARY = f"{_FM}.PrimaryBodies = {{Earth}};\n"
_DEGREE = f"{_FM}.GravityField.Earth.Degree = 0;\n"
_ORDER = f"{_FM}.GravityField.Earth.Order = 0;\n"
_TIDE = f"{_FM}.GravityField.Earth.TideModel = 'None';\n"


def _seed_warnings(
    workshop_script: Path, tmp_path: Path, edits: tuple[tuple[str, str], ...]
) -> list[str]:
    text = workshop_script.read_text(encoding="utf-8")
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path = tmp_path / "forces.script"
    path.write_text(text, encoding="utf-8")
    scenario, _ = parse_gmat_script(path, OrbitWinConfig())
    assert set(scenario.two_body_gaps) <= set(scenario.warnings)
    return list(scenario.two_body_gaps)


def test_the_point_mass_workshop_template_has_no_seed_warning(workshop_script: Path) -> None:
    scenario, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    assert scenario.two_body_gaps == ()
    assert not any("Analytical seed ignores" in w for w in scenario.warnings)


def test_an_additional_primary_body_is_a_gap(workshop_script: Path, tmp_path: Path) -> None:
    edits = ((_PRIMARY, f"{_FM}.PrimaryBodies = {{Earth, Luna}};\n"),)
    assert "additional primary bodies ['Luna']" in _seed_warnings(workshop_script, tmp_path, edits)[0]


_PROP_FM = "DefaultProp.FM = DefaultProp_ForceModel;\n"


def test_an_unused_force_model_is_not_a_gap(workshop_script: Path, tmp_path: Path) -> None:
    """R2026a: a bare ForceModel defaults to Earth 4x4, but nothing flies FM2."""
    unused = "Create ForceModel FM2;\nFM2.SRP = On;\nFM2.Drag = MSISE90;\n"
    assert _seed_warnings(workshop_script, tmp_path, ((_PROP_FM, f"{_PROP_FM}{unused}"),)) == []


def test_a_propagator_without_fm_flies_the_default_model(
    workshop_script: Path, tmp_path: Path
) -> None:
    """R2026a: with no Prop.FM, 86.291 km from point mass after 6000 s."""
    warnings = _seed_warnings(workshop_script, tmp_path, ((_PROP_FM, ""),))
    assert warnings == [
        "Analytical seed ignores non-spherical gravity: DefaultProp sets no FM, so GMAT "
        "flies its default force model, Earth Degree=4, Order=4"
    ]


def test_a_propagator_flown_only_on_a_later_line_is_read(
    workshop_script: Path, tmp_path: Path
) -> None:
    last = "Propagate DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000, AlienSC.ElapsedSecs = 1000, OrbitColor = [0 200 0]};"
    edits = (
        (_PROP_FM, f"{_PROP_FM}Create Propagator Late;\n"),
        (last, last.replace("DefaultProp(", "Late(")),
    )
    warnings = _seed_warnings(workshop_script, tmp_path, edits)
    assert len(warnings) == 1 and "Late sets no FM" in warnings[0]


def test_an_fm_the_script_does_not_create_is_refused(
    workshop_script: Path, tmp_path: Path
) -> None:
    with pytest.raises(ParseError, match="does not create as a ForceModel"):
        _seed_warnings(workshop_script, tmp_path, ((_PROP_FM, "DefaultProp.FM = Missing;\n"),))


def test_two_propagators_each_contribute_their_own_model(
    project_root: Path, tmp_path: Path
) -> None:
    script = project_root / "examples" / "ranking" / "R3_FastHighPerigee_TwoPropagators.script"
    config = tmp_path / "config.yaml"
    config.write_text(
        "pursuer_name: SC_Human\ntarget_name: SC_Alien\n"
        "policy:\n  rules: ranking\n  horizon_s: 5092.815\n"
        "  dv_limit_km_s: 0.25\n  dv_safe_limit_km_s: 0.249\n"
        "  total_dv_limit_km_s: 3.0\n  total_dv_safe_limit_km_s: 2.99\n",
        encoding="utf-8",
    )
    scenario, _ = parse_gmat_script(script, load_config(config))
    named = [gap for gap in scenario.two_body_gaps if "AlienProp_ForceModel" in gap]
    assert len(named) == 6 and len(scenario.two_body_gaps) == 12
    shared = tmp_path / "shared.script"
    text = script.read_text(encoding="utf-8")
    assert text.count("AlienProp.FM = AlienProp_ForceModel;") == 1
    shared.write_text(
        text.replace("AlienProp.FM = AlienProp_ForceModel;", "AlienProp.FM = DefaultProp_ForceModel;"),
        encoding="utf-8",
    )
    scenario, _ = parse_gmat_script(shared, load_config(config))
    assert len(scenario.two_body_gaps) == 6
    assert not [gap for gap in scenario.two_body_gaps if "AlienProp_ForceModel" in gap]


@pytest.mark.parametrize(
    ("edits", "expected"),
    [
        (((f"{_FM}.Drag = None;", f"{_FM}.Drag = MSISE90;"),), "drag"),
        # The form GMAT's GUI saves; the ranking rehearsal scripts use it.
        (((f"{_FM}.Drag = None;", f"{_FM}.Drag.AtmosphereModel = NRLMSISE00;"),), "drag"),
        (((f"{_FM}.Drag = None;", f"{_FM}.Drag.AtmosphereModel = 'JacchiaRoberts';"),), "drag"),
        (((f"{_FM}.SRP = Off;", f"{_FM}.SRP = On;"),), "solar radiation pressure"),
        (((f"{_FM}.RelativisticCorrection = Off;", f"{_FM}.RelativisticCorrection = On;"),), "relativity"),
        (((_PRIMARY, f"{_PRIMARY}{_FM}.PointMasses = {{Sun, Luna}};\n"),), "['Sun', 'Luna']"),
        (((_DEGREE, f"{_FM}.GravityField.Earth.Degree = 4;\n"),), "non-spherical gravity"),
        (((_ORDER, f"{_FM}.GravityField.Earth.Order = 4;\n"),), "non-spherical gravity"),
        (((_DEGREE, ""), (_ORDER, "")), "Degree=4 (GMAT default)"),
        (((_PRIMARY, ""), (_DEGREE, ""), (_ORDER, "")), "Degree=4 (GMAT default)"),
    ],
)
def test_each_force_the_seed_ignores_is_named_once(
    workshop_script: Path, tmp_path: Path, edits: tuple[tuple[str, str], ...], expected: str
) -> None:
    warnings = _seed_warnings(workshop_script, tmp_path, edits)
    assert len(warnings) == 1, warnings
    assert expected in warnings[0]


def test_tides_are_named_only_with_a_field_they_act_on(
    workshop_script: Path, tmp_path: Path
) -> None:
    """GMAT R2026a: TideModel leaves the state bit-identical below degree 2."""
    solid = (_TIDE, f"{_FM}.GravityField.Earth.TideModel = 'SolidAndPole';\n")
    assert _seed_warnings(workshop_script, tmp_path, (solid,)) == []
    degree_two = (_DEGREE, f"{_FM}.GravityField.Earth.Degree = 2;\n")
    warnings = _seed_warnings(workshop_script, tmp_path, (solid, degree_two))
    assert len(warnings) == 2
    assert "ignores tides" in warnings[1]


@pytest.mark.parametrize(
    "edits",
    [
        ((f"{_FM}.Drag = None;", f"{_FM}.Drag.AtmosphereModel = None;"),),
        ((f"{_FM}.Drag = None;", f"{_FM}.Drag.AtmosphereModel = 'None';"),),
        ((f"{_FM}.SRP = Off;", f"{_FM}.SRP = false;"),),
        ((_PRIMARY, f"{_PRIMARY}{_FM}.PointMasses = {{}};\n"),),
        ((_PRIMARY, f"{_PRIMARY}{_FM}.PointMasses = {{Earth}};\n"),),
        # Earth only as a point mass: GMAT gives it no harmonic field.
        ((_PRIMARY, f"{_FM}.PointMasses = {{Earth}};\n"), (_DEGREE, ""), (_ORDER, "")),
    ],
)
def test_a_force_that_is_off_is_not_named(
    workshop_script: Path, tmp_path: Path, edits: tuple[tuple[str, str], ...]
) -> None:
    assert _seed_warnings(workshop_script, tmp_path, edits) == []


def test_every_ranking_force_is_named_once(workshop_script: Path, tmp_path: Path) -> None:
    edits = (
        (_PRIMARY, f"{_PRIMARY}{_FM}.PointMasses = {{Sun, Luna}};\n"),
        (f"{_FM}.Drag = None;", f"{_FM}.Drag.AtmosphereModel = NRLMSISE00;"),
        (f"{_FM}.SRP = Off;", f"{_FM}.SRP = On;"),
        (f"{_FM}.RelativisticCorrection = Off;", f"{_FM}.RelativisticCorrection = On;"),
        (_DEGREE, f"{_FM}.GravityField.Earth.Degree = 20;\n"),
        (_ORDER, f"{_FM}.GravityField.Earth.Order = 20;\n"),
        (_TIDE, f"{_FM}.GravityField.Earth.TideModel = 'SolidAndPole';\n"),
    )
    warnings = _seed_warnings(workshop_script, tmp_path, edits)
    effects = ["drag", "solar radiation pressure", "relativity", "third-body", "non-spherical", "tides"]
    assert len(warnings) == len(effects)
    for effect, warning in zip(effects, warnings, strict=True):
        assert effect in warning


def test_a_resource_line_the_parser_cannot_read_is_refused(
    workshop_script: Path, tmp_path: Path
) -> None:
    """GMAT ran `AlienSC.DateFormat = UTCModJulian` without its semicolon; we skipped it."""
    text = workshop_script.read_text(encoding="utf-8")
    marker = "AlienSC.DateFormat = UTCGregorian;"
    path = tmp_path / "nosemicolon.script"
    path.write_text(text.replace(marker, marker.rstrip(";")), encoding="utf-8")
    with pytest.raises(ParseError, match="GMAT allows no semicolon, this parser does not"):
        parse_gmat_script(path, OrbitWinConfig())


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("Maneuver ImpulsiveBurn1 (HumanSC);", "a Maneuver the parser cannot read"),
        ("BeginMissionSequence;", "Multiple BeginMissionSequence"),
    ],
)
def test_a_mission_line_is_refused_with_its_own_reason(
    workshop_script: Path, tmp_path: Path, command: str, message: str
) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    marker = "Maneuver ImpulsiveBurn2(HumanSC);"
    path = tmp_path / "reason.script"
    path.write_text(text.replace(marker, f"{marker}\n{command}"), encoding="utf-8")
    with pytest.raises(ParseError, match=message):
        parse_gmat_script(path, OrbitWinConfig())


# GMAT SaveMission writes these initial values; none of them moves a craft.
@pytest.mark.parametrize(
    "lines",
    [
        "Create Variable lowerBound;\nlowerBound = -119;",
        "Create Variable loopCount tol;\nGMAT loopCount = 1; % start\ntol = 1.5e-3;",
        "Create Array Arr[2,2];\nArr(1, 1) = 3;\nGMAT Arr(2,2) = -0.5;",
        "Create String Str EmptyLine;\nStr = 'abc';\nEmptyLine     = ' ';",
    ],
)
def test_a_literal_initial_value_is_read_past(
    workshop_script: Path, tmp_path: Path, lines: str
) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    path = tmp_path / "literal.script"
    path.write_text(text.replace("BeginMissionSequence;", f"{lines}\nBeginMissionSequence;", 1),
                    encoding="utf-8")
    scenario, _ = parse_gmat_script(path, OrbitWinConfig())
    assert len(scenario.burn_slots) == 2


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        # R2026a ran this and copied AlienSC onto HumanSC: SMA 7000 -> 9000.
        ("HumanSC = AlienSC;", "neither a Create nor an assignment"),
        ("Create Variable x;\nx = HumanSC.SMA;", "neither a Create nor an assignment"),
        ("Create Variable x;\ny = 1;", "neither a Create nor an assignment"),
        ("x = 1;\nCreate Variable x;", "neither a Create nor an assignment"),
        ("Create Array A[1,1];\nA = 1;", "neither a Create nor an assignment"),
        ("Create Variable x;\nA(1, 1) = 1;", "neither a Create nor an assignment"),
        ("Create String s;\ns = 1;", "neither a Create nor an assignment"),
        ("Create Variable x;\nx = 1", "GMAT allows no semicolon"),
    ],
)
def test_any_other_bare_assignment_is_refused(
    workshop_script: Path, tmp_path: Path, lines: str, message: str
) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    path = tmp_path / "bare.script"
    path.write_text(text.replace("BeginMissionSequence;", f"{lines}\nBeginMissionSequence;", 1),
                    encoding="utf-8")
    with pytest.raises(ParseError, match=message):
        parse_gmat_script(path, OrbitWinConfig())
