"""The state probe: GMAT's own reading of each craft's initial epoch and state.

Fast tests, on the fake console; tests/integration/test_state_probe_gmat.py runs
the real one. See docs/ASYNC_START_AND_INSERTION.md §1.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from conftest import ORGANIZER_MATERIAL, organizer_scripts

from orbitwin.config import OrbitWinConfig
from orbitwin.errors import GmatRunError, ParseError
from orbitwin.gmat_parser import (
    OUTPUT_FILE_FIELDS,
    PROBE_POSITION_TOLERANCE_KM,
    PROBE_VELOCITY_TOLERANCE_KM_S,
    GmatScriptDocument,
    parse_gmat_script,
)
from orbitwin.models import (
    GmatRunResult,
    ProbedCraft,
    ProbedEpoch,
    StateProbe,
    StateVector,
    spacecraft_binding_sha256,
)
from orbitwin.state_probe import (
    PROBE_RESOURCE_NAME,
    build_probe_script,
    load_scenario,
    parse_probe_report,
)

ROOT = Path(__file__).resolve().parents[2]
ASYNC_FIXTURE = ROOT / "tests" / "fixtures" / "async" / "insertion_and_async_start.script"
#: One ModJulian double step between 16384 and 32768 days (2**-38).
A1_STEP_DAYS = 3.637978807091713e-12


def _fake_load(script: Path, fake_gmat: Path, fake_startup: Path, config=None):
    scenario, document, _ = load_scenario(
        script, config or OrbitWinConfig(), gmat_console=fake_gmat, startup_file=fake_startup
    )
    return scenario, document


def _run_record() -> GmatRunResult:
    return GmatRunResult(
        run_id="hand-made", success=True, return_code=0, started_at_utc="", duration_s=0.0,
        workdir="", script_path="", script_sha256="", executable_sha256="",
        startup_sha256=None, stdout_path="", stderr_path="", log_path=None,
        reports={}, report_sha256={},
    )


def _probe_for(
    script: Path, states: dict[str, StateVector], a1_mjd: dict[str, float]
) -> StateProbe:
    """A probe built by hand, as if GMAT had reported these states and epochs."""
    from orbitwin.state_probe import _resource_section

    _, craft, fields, created = _resource_section(GmatScriptDocument.read(script))
    return StateProbe(
        craft=tuple(
            ProbedCraft(
                name=name,
                epoch=ProbedEpoch(a1_mjd[name], 0.0, 0.0, 0.0, 0.0),
                state=states[name],
                binding_sha256=spacecraft_binding_sha256(name, fields, created),
            )
            for name in craft
        ),
        run=_run_record(),
    )


def _synchronous_insertion(tmp_path: Path) -> Path:
    """The fixture with SC_Human inserted at SC_Alien's epoch and one joint Propagate."""
    text = ASYNC_FIXTURE.read_text(encoding="utf-8")
    head = text.split("BeginMissionSequence;")[0].replace(
        "SC_Human.Epoch = '01 Jan 2000 15:00:00.000';",
        "SC_Human.Epoch = '01 Jan 2000 00:00:00.000';",
    )
    path = tmp_path / "synchronous_insertion.script"
    path.write_text(
        head + "BeginMissionSequence;\n"
        "Propagate DefaultProp(SC_Alien, SC_Human) {SC_Human.ElapsedSecs = 3000};\n",
        encoding="utf-8",
    )
    return path


def _insertion_config() -> OrbitWinConfig:
    return OrbitWinConfig(pursuer_name="SC_Human", target_name="SC_Alien")


INSERTED = StateVector(
    (484.95616288138166, 6282.3523391160525, 2739.983139963896),
    (-7.636759847503625, 0.5895872910739958, -0.0001829736440827734),
)
ALIEN = StateVector((7000.0, 0.0, 0.0), (0.0, 7.546053287267836, 0.0))


# --- the probe script -------------------------------------------------------


def test_the_probe_copy_reports_each_craft_before_anything_propagates(
    workshop_script: Path,
) -> None:
    document = GmatScriptDocument.read(workshop_script)
    probe_bytes, craft = build_probe_script(document)
    text = probe_bytes.decode("utf-8")
    assert craft == ("HumanSC", "AlienSC")
    mission = text.split("BeginMissionSequence;", 1)[1].strip().splitlines()
    assert len(mission) == 1 and mission[0].startswith(f"Report {PROBE_RESOURCE_NAME} ")
    for name in craft:
        for scale in ("A1", "UTC", "TAI", "TT", "TDB"):
            assert f"{name}.{scale}ModJulian" in mission[0]
        for axis in ("X", "Y", "Z", "VX", "VY", "VZ"):
            assert f"{name}.EarthMJ2000Eq.{axis}" in mission[0]
    assert f"{PROBE_RESOURCE_NAME}.Precision = 17;" in text
    # GmatConsole refuses OpenFrames outright; the resources are otherwise kept.
    assert "\nCreate OpenFramesInterface" not in text
    resources = text.split("BeginMissionSequence;", 1)[0]
    original = document.text.split("BeginMissionSequence;", 1)[0]
    kept = [line for line in original.splitlines() if "OpenFrames" not in line]
    assert all(line in resources for line in kept if line.strip())


def test_a_script_that_already_uses_the_probe_name_is_refused(
    workshop_script: Path, tmp_path: Path
) -> None:
    path = tmp_path / "taken.script"
    path.write_text(
        f"% {PROBE_RESOURCE_NAME}\n" + workshop_script.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ParseError, match="reserved name"):
        build_probe_script(GmatScriptDocument.read(path))


def _probe_report(path: Path, craft: tuple[str, ...], rows: int = 1, value: str = "1") -> Path:
    headers = [
        f"{name}.{quantity}"
        for name in craft
        for quantity in (
            *(f"{scale}ModJulian" for scale in ("A1", "UTC", "TAI", "TT", "TDB")),
            *(f"EarthMJ2000Eq.{axis}" for axis in ("X", "Y", "Z", "VX", "VY", "VZ")),
        )
    ]
    path.write_text(
        " ".join(headers) + "\n" + "\n".join(" ".join([value] * len(headers)) for _ in range(rows))
        + "\n",
        encoding="utf-8",
    )
    return path


def test_the_probe_report_is_read_bit_for_bit(tmp_path: Path) -> None:
    exact = "21545.125370768306"
    craft = parse_probe_report(_probe_report(tmp_path / "r.txt", ("A",), value=exact), ("A",), {}, {})
    assert craft[0].epoch.a1_mjd == float(exact) and craft[0].state.r_km[0] == float(exact)


@pytest.mark.parametrize(
    ("rows", "craft", "message"),
    [(2, ("A",), "2 rows, not one"), (1, ("A", "B"), r"lacks \['B\.A1ModJulian'")],
)
def test_a_malformed_probe_report_is_refused(
    tmp_path: Path, rows: int, craft: tuple[str, ...], message: str
) -> None:
    path = _probe_report(tmp_path / "r.txt", ("A",), rows=rows)
    with pytest.raises(GmatRunError, match=message):
        parse_probe_report(path, craft, {}, {})


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_a_non_finite_probe_value_is_refused(tmp_path: Path, value: str) -> None:
    """By the report reader, before the probe looks at a value."""
    with pytest.raises(GmatRunError, match="NaN or infinity"):
        parse_probe_report(_probe_report(tmp_path / "r.txt", ("A",), value=value), ("A",), {}, {})


# --- the fake console answers it ---------------------------------------------


@pytest.mark.parametrize(
    "fixture", ["workshop_script", "different_plane_script", "high_plane_script"]
)
def test_probed_states_match_the_in_process_reading(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    fake_gmat: Path,
    fake_startup: Path,
    fixture: str,
) -> None:
    script = request.getfixturevalue(fixture)
    own, _ = parse_gmat_script(script, OrbitWinConfig())
    probed, _ = _fake_load(script, fake_gmat, fake_startup)
    assert probed.state_probe is not None and own.state_probe is None
    assert probed.epoch_offset_s == 0.0 and own.epoch_offset_s is None
    for craft in (probed.pursuer, probed.target):
        assert craft.probed_epoch is not None
        assert craft.in_process_mismatch is not None
        position_km, velocity_km_s = craft.in_process_mismatch
        assert position_km <= PROBE_POSITION_TOLERANCE_KM
        assert velocity_km_s <= PROBE_VELOCITY_TOLERANCE_KM_S
    # The solver flies the probed state, not its own conversion.
    probe = probed.state_probe
    assert probed.pursuer.initial_state == probe.craft_named(probed.pursuer.name).state
    assert not any("in-process" in warning for warning in probed.warnings)


def test_one_instant_in_two_date_formats_needs_the_probe(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """12:00:00 UTC on 15 Mar 2027 is UTCModJulian 31480; the texts differ."""
    text = workshop_script.read_text(encoding="utf-8")
    edited = text.replace(
        "AlienSC.DateFormat = UTCGregorian;\nAlienSC.Epoch = '15 Mar 2027 12:00:00.000';",
        "AlienSC.DateFormat = UTCModJulian;\nAlienSC.Epoch = '31480.0';",
    )
    assert edited != text
    path = tmp_path / "formats.script"
    path.write_text(edited, encoding="utf-8")
    with pytest.raises(ParseError, match=r"epoch texts differ.*--gmat-console"):
        parse_gmat_script(path, OrbitWinConfig())
    scenario, _ = _fake_load(path, fake_gmat, fake_startup)
    assert scenario.epoch_offset_s is not None
    assert abs(scenario.epoch_offset_s) <= 1.0e-6


def test_the_fake_console_leaves_earth_fixed_to_real_gmat(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    with pytest.raises(GmatRunError, match="state probe failed"):
        _fake_load(_synchronous_insertion(tmp_path), fake_gmat, fake_startup)


_TA = "HumanSC.TA = 250;"
_ALIEN_UTC = "AlienSC.DateFormat = UTCGregorian;\nAlienSC.Epoch = '15 Mar 2027 12:00:00.000';"


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        # It used to report the default epoch, 21545 TAI, for this craft.
        (_ALIEN_UTC, "AlienSC.Epoch.UTCModJulian = 31480.0;", r"AlienSC\.Epoch\.<format>"),
        (_TA, f"{_TA}\nHumanSC.MA = 10;", r"HumanSC\.MA beside"),
        (_TA, f"{_TA}\nHumanSC.X = 7000;", r"HumanSC\.X beside"),
        (_TA, f"{_TA}\nHumanSC.UTCModJulian = 31480.2;", r"HumanSC\.UTCModJulian beside"),
        # It used to read the six as a true anomaly.
        (_TA, f"{_TA}\nHumanSC.AnomalyType = MA;", r"HumanSC\.AnomalyType = MA"),
        (_TA, f"{_TA}\nGMAT HumanSC.AnomalyType = MA;", r"HumanSC\.AnomalyType = MA"),
        (_ALIEN_UTC, "AlienSC.epoch.UTCModJulian = 31480.0;", r"AlienSC\.Epoch\.<format>"),
        (_TA, f"{_TA}\nHumanSC.ma = 10;", r"HumanSC\.ma beside"),
        (_TA, f"{_TA}\nHumanSC.Element1 = 9000;", r"HumanSC\.Element1 beside"),
        (_TA, f"{_TA}\nHumanSC.anomalytype = MA;", r"HumanSC\.anomalytype in the wrong case"),
        (_TA, f"{_TA}\nHumanSC.epoch = '15 Mar 2027 13:00:00.000';",
         r"HumanSC\.epoch in the wrong case"),
        (_TA, f"{_TA}\nHumanSC.coordinatesystem = EarthFixed;",
         r"HumanSC\.coordinatesystem in the wrong case"),
        (_TA, f"{_TA}\nHumanSC.displaystatetype = Keplerian;",
         r"HumanSC\.displaystatetype in the wrong case"),
        (_TA, f"{_TA}\nHumanSC.dateformat = UTCGregorian;",
         r"HumanSC\.dateformat in the wrong case"),
        (_ALIEN_UTC,
         "AlienSC.Epoch = '15 Mar 2027 12:00:00.000';\nAlienSC.DateFormat = UTCGregorian;",
         r"AlienSC\.Epoch before its DateFormat"),
    ],
    ids=["epoch-scale", "mean-anomaly", "second-type", "mission-epoch", "anomaly-type",
         "gmat-keyword", "epoch-scale-lower", "lower-case-field", "element1",
         "anomaly-type-lower", "epoch-lower", "coordinate-system-lower",
         "display-state-type-lower", "date-format-lower", "epoch-before-format"],
)
def test_the_fake_console_leaves_what_it_does_not_model_to_real_gmat(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path, old: str,
    new: str, reason: str,
) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "unmodelled.script"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(GmatRunError, match=f"fake GmatConsole cannot convert {reason}"):
        _fake_load(path, fake_gmat, fake_startup)


# --- the parser with a probe -------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ("", r"CoordinateSystem='EarthFixed'.*--gmat-console"),
        (
            "SC_Human.CoordinateSystem = EarthFixed;|SC_Human.CoordinateSystem = EarthMJ2000Eq;",
            r"DisplayStateType='SphericalAZFPA'.*--gmat-console",
        ),
    ],
)
def test_without_a_console_an_insertion_state_is_refused(
    tmp_path: Path, edit: str, message: str
) -> None:
    path = _synchronous_insertion(tmp_path)
    if edit:
        old, new = edit.split("|")
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    with pytest.raises(ParseError, match=message):
        parse_gmat_script(path, _insertion_config())


def test_with_a_probe_an_insertion_state_is_accepted(tmp_path: Path) -> None:
    path = _synchronous_insertion(tmp_path)
    a1 = 21544.500370768306
    probe = _probe_for(path, {"SC_Human": INSERTED, "SC_Alien": ALIEN}, {"SC_Human": a1, "SC_Alien": a1})
    scenario, _ = parse_gmat_script(path, _insertion_config(), probe=probe)
    assert scenario.pursuer.name == "SC_Human"
    assert scenario.pursuer.initial_state == INSERTED
    assert scenario.pursuer.coordinate_system == "EarthFixed"
    assert scenario.pursuer.in_process_mismatch is None  # nothing of ours to compare
    assert scenario.target.in_process_mismatch is not None
    assert scenario.epoch_offset_s == 0.0
    assert json.loads(json.dumps(scenario.to_dict()))["state_probe"]["craft"][0]["name"]


@pytest.mark.parametrize(
    ("steps", "accepted"), [(0, True), (1, True), (3, True), (-3, True), (4, False), (-4, False)]
)
def test_probed_epochs_within_a_microsecond_are_one_start(
    tmp_path: Path, steps: int, accepted: bool
) -> None:
    """Three A1 double steps (0.94 µs) are one start; four (1.26 µs) are not."""
    path = _synchronous_insertion(tmp_path)
    a1 = 21544.500370768306
    late = a1 + steps * A1_STEP_DAYS
    probe = _probe_for(
        path, {"SC_Human": INSERTED, "SC_Alien": ALIEN}, {"SC_Human": a1, "SC_Alien": late}
    )
    if accepted:
        scenario, _ = parse_gmat_script(path, _insertion_config(), probe=probe)
        assert scenario.epoch_offset_s is not None and abs(scenario.epoch_offset_s) <= 1.0e-6
    else:
        with pytest.raises(ParseError, match="has no epoch alignment as its first command"):
            parse_gmat_script(path, _insertion_config(), probe=probe)


def test_an_epoch_offset_needs_the_alignment_flown(tmp_path: Path) -> None:
    path = _synchronous_insertion(tmp_path)
    head, _, mission = path.read_text(encoding="utf-8").partition("BeginMissionSequence;\n")
    path.write_text(
        head + "BeginMissionSequence;\n"
        "Propagate DefaultProp(SC_Alien) {SC_Alien.UTCModJulian = SC_Human.UTCModJulian};\n"
        + mission,
        encoding="utf-8",
    )
    probe = _probe_for(
        path,
        {"SC_Human": INSERTED, "SC_Alien": ALIEN},
        {"SC_Human": 21545.125370768306, "SC_Alien": 21544.500370768306},
    )
    config = dataclasses.replace(
        _insertion_config(),
        policy=dataclasses.replace(
            _insertion_config().policy,
            mission_time_origin="target",
            closest_approach_window="after_alignment",
        ),
    )
    with pytest.raises(
        ParseError, match=r"54000 s apart; t = 0 is where GMAT's own run of the alignment"
    ):
        parse_gmat_script(path, config, probe=probe)


def test_a_probe_of_other_craft_assignments_is_refused(tmp_path: Path) -> None:
    path = _synchronous_insertion(tmp_path)
    a1 = 21544.500370768306
    probe = _probe_for(path, {"SC_Human": INSERTED, "SC_Alien": ALIEN}, {"SC_Human": a1, "SC_Alien": a1})
    path.write_text(
        path.read_text(encoding="utf-8").replace("SC_Human.VMAG = 7.2;", "SC_Human.VMAG = 7.3;"),
        encoding="utf-8",
    )
    with pytest.raises(ParseError, match="places SC_Human differently"):
        parse_gmat_script(path, _insertion_config(), probe=probe)


@pytest.mark.parametrize(
    ("anchor", "added"),
    [
        # A frame of its own, referenced by the craft, then changed.
        ("SC_Human.CoordinateSystem = EarthFixed;", "Topo.Origin = Luna;"),
        # A built-in body the conversion reads.
        ("Create ForceModel", "Earth.EopFileName = 'other_eop.txt';"),
    ],
)
def test_a_probe_of_a_differently_placed_craft_is_refused(
    tmp_path: Path, anchor: str, added: str
) -> None:
    path = _synchronous_insertion(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "SC_Human.CoordinateSystem = EarthFixed;", "SC_Human.CoordinateSystem = Topo;"
    ).replace(
        "Create Spacecraft SC_Alien;",
        "Create CoordinateSystem Topo;\nTopo.Origin = Earth;\nTopo.Axes = BodyFixed;\n\n"
        "Create Spacecraft SC_Alien;",
    )
    path.write_text(text, encoding="utf-8")
    a1 = 21544.500370768306
    probe = _probe_for(path, {"SC_Human": INSERTED, "SC_Alien": ALIEN}, {"SC_Human": a1, "SC_Alien": a1})
    parse_gmat_script(path, _insertion_config(), probe=probe)
    edited = text.replace("Topo.Origin = Earth;\n", "") if added.startswith("Topo") else text
    anchor = "SC_Human.CoordinateSystem = Topo;" if added.startswith("Topo") else anchor
    path.write_text(edited.replace(anchor, f"{added}\n{anchor}", 1), encoding="utf-8")
    with pytest.raises(ParseError, match=r"places SC_\w+ differently"):
        parse_gmat_script(path, _insertion_config(), probe=probe)


def test_a_probe_far_from_our_conversion_is_reported_and_used(workshop_script: Path) -> None:
    own, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    moved = StateVector(
        (own.pursuer.initial_state.r_km[0] + 1.0, *own.pursuer.initial_state.r_km[1:]),
        own.pursuer.initial_state.v_km_s,
    )
    a1 = 31480.000428638676
    probe = _probe_for(
        workshop_script,
        {"HumanSC": moved, "AlienSC": own.target.initial_state},
        {"HumanSC": a1, "AlienSC": a1},
    )
    scenario, _ = parse_gmat_script(workshop_script, OrbitWinConfig(), probe=probe)
    assert scenario.pursuer.initial_state == moved
    assert any("GMAT reads HumanSC's initial state 1 km" in w for w in scenario.warnings)


def test_a_probe_missing_a_craft_is_refused(workshop_script: Path) -> None:
    own, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    probe = _probe_for(
        workshop_script,
        {"HumanSC": own.pursuer.initial_state, "AlienSC": own.target.initial_state},
        {"HumanSC": 1.0, "AlienSC": 1.0},
    )
    probe = dataclasses.replace(probe, craft=probe.craft[:1])
    with pytest.raises(ParseError, match="did not report AlienSC"):
        parse_gmat_script(workshop_script, OrbitWinConfig(), probe=probe)


# --- state fields outside the DisplayStateType ------------------------------------

_KEPLERIAN = "HumanSC.DisplayStateType = Keplerian;"


_TWO_TYPES = "GMAT R2026a refuses fields of two state types"
_KEPLERIAN_SET = "AOP, ECC, INC, RAAN, SMA, TA beside"
_AS_CARTESIAN = "DisplayStateType='Cartesian'; they are Keplerian fields.*--gmat-console"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        # R2026a refused X after Keplerian and SMA after Cartesian ("not contained in
        # the same state type"); the rest are two types by GMAT's own tables.
        (_TA, f"{_TA}\nHumanSC.X = 7000;", f"X beside DisplayStateType='Keplerian'; {_TWO_TYPES}"),
        (_TA, f"{_TA}\nHumanSC.RadPer = 6900;", f"RadPer beside .*; {_TWO_TYPES}"),
        (_TA, f"{_TA}\nHumanSC.EquinoctialH = 0;", f"EquinoctialH beside .*; {_TWO_TYPES}"),
        (_TA, f"{_TA}\nHumanSC.DelaunayL = 1;", f"DelaunayL beside .*; {_TWO_TYPES}"),
        # Read by R2026a into the state beside the six: 1.5e4 and 2.6e4 km away.
        (_TA, f"{_TA}\nHumanSC.Element1 = 9000;",
         "Element1, which GMAT R2026a reads into the state beside the Keplerian.*--gmat-console"),
        (_TA, f"{_TA}\nHumanSC.CartesianX = 7000;",
         "CartesianX, which GMAT R2026a reads into the state.*--gmat-console"),
        (_TA, f"{_TA}\nHumanSC.CartesianState = 7000;",
         "CartesianState, which GMAT R2026a reads into the state.*--gmat-console"),
        (_TA, f"{_TA}\nHumanSC.KeplerianState = 7000;",
         "KeplerianState, which GMAT R2026a reads into the state.*--gmat-console"),
        # One other type's whole set, which GMAT read as that type.
        (_KEPLERIAN, "HumanSC.DisplayStateType = Cartesian;", f"{_KEPLERIAN_SET} {_AS_CARTESIAN}"),
        (_KEPLERIAN, "", f"{_KEPLERIAN_SET} {_AS_CARTESIAN}"),
    ],
    ids=["cartesian", "modified-keplerian", "equinoctial", "delaunay-capital", "element1",
         "cartesian-x", "cartesian-state", "keplerian-state",
         "keplerian-as-cartesian", "no-display-type"],
)
def test_without_a_console_another_types_fields_are_refused(
    workshop_script: Path, tmp_path: Path, old: str, new: str, message: str
) -> None:
    """The in-process reader read the six it knows and ignored the rest."""
    text = workshop_script.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "foreign.script"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ParseError, match=f"HumanSC sets {message}"):
        parse_gmat_script(path, OrbitWinConfig())


@pytest.mark.parametrize(
    ("added", "field", "canonical"),
    [("HumanSC.ta = 250;", "ta", "TA"), ("HumanSC.Sma = 6900;", "Sma", "SMA"),
     ("HumanSC.utcmodjulian = 31480.2;", "utcmodjulian", "UTCModJulian"),
     ("HumanSC.element1 = 9000;", "element1", "Element1"),
     ("HumanSC.DELAUNAYL = 1;", "DELAUNAYL", "DelaunayL or Delaunayl"),
     ("HumanSC.anomalytype = MA;", "anomalytype", "AnomalyType"),
     ("HumanSC.coordinatesystem = EarthFixed;", "coordinatesystem", "CoordinateSystem"),
     ("HumanSC.epoch = '15 Mar 2027 13:00:00.000';", "epoch", "Epoch"),
     ("HumanSC.displaystatetype = Cartesian;", "displaystatetype", "DisplayStateType"),
     ("HumanSC.dateformat = TAIModJulian;", "dateformat", "DateFormat")],
)
def test_without_a_console_a_wrongly_cased_field_is_refused(
    workshop_script: Path, tmp_path: Path, added: str, field: str, canonical: str
) -> None:
    """We ignored them; R2026a refused each it was given (DELAUNAYL was not tried)."""
    path = tmp_path / "cased.script"
    path.write_text(
        workshop_script.read_text(encoding="utf-8").replace(_TA, f"{_TA}\n{added}", 1),
        encoding="utf-8",
    )
    with pytest.raises(
        ParseError, match=rf"HumanSC\.{field} is GMAT's {canonical} in the wrong case"
    ):
        parse_gmat_script(path, OrbitWinConfig())


@pytest.mark.parametrize("kind", ["Foo", "keplerian"])
def test_without_a_console_an_unknown_state_type_is_refused_first(
    workshop_script: Path, tmp_path: Path, kind: str
) -> None:
    """GMAT R2026a refused 'keplerian' too; we read it as Keplerian."""
    path = tmp_path / "unknown.script"
    path.write_text(
        workshop_script.read_text(encoding="utf-8").replace(
            _KEPLERIAN, f"HumanSC.DisplayStateType = {kind};", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ParseError, match=rf"DisplayStateType='{kind}' is not one of GMAT's"):
        parse_gmat_script(path, OrbitWinConfig())


@pytest.mark.parametrize(("value", "accepted"), [("TA", True), ("MA", False), ("'EA'", False)])
def test_without_a_console_only_a_true_anomaly_is_read(
    workshop_script: Path, tmp_path: Path, value: str, accepted: bool
) -> None:
    """GMAT R2026a put the same six fields 5.6e5 km away with AnomalyType = MA."""
    own, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    path = tmp_path / "anomaly_type.script"
    path.write_text(
        workshop_script.read_text(encoding="utf-8").replace(
            _TA, f"{_TA}\nHumanSC.AnomalyType = {value};", 1
        ),
        encoding="utf-8",
    )
    if accepted:
        scenario, _ = parse_gmat_script(path, OrbitWinConfig())
        assert scenario.pursuer.initial_state == own.pursuer.initial_state
    else:
        with pytest.raises(ParseError, match=r"HumanSC\.AnomalyType='[ME]A'.*--gmat-console"):
            parse_gmat_script(path, OrbitWinConfig())


@pytest.mark.parametrize("anomaly", ["MA", "EA", "HA"])
@pytest.mark.parametrize("replace_ta", [False, True], ids=["beside-ta", "instead-of-ta"])
def test_without_a_console_an_anomaly_other_than_ta_is_refused(
    workshop_script: Path, tmp_path: Path, anomaly: str, replace_ta: bool
) -> None:
    """GMAT R2026a refuses MA ("other than True Anomaly"); we read TA and ignored it."""
    text = workshop_script.read_text(encoding="utf-8")
    added = f"HumanSC.{anomaly} = 10;"
    path = tmp_path / "anomaly.script"
    path.write_text(text.replace(_TA, added if replace_ta else f"{_TA}\n{added}", 1), "utf-8")
    with pytest.raises(
        ParseError, match=f"HumanSC sets {anomaly} beside .*refuses an anomaly other than TA"
    ):
        parse_gmat_script(path, OrbitWinConfig())


_ALIEN_EPOCH = "AlienSC.Epoch = '15 Mar 2027 12:00:00.000';"


@pytest.mark.parametrize(
    ("new", "message"),
    [
        # GMAT R2026a: 'The field name "UTCModJulian" on object ... is not permitted'.
        (f"{_ALIEN_EPOCH}\nAlienSC.UTCModJulian = 31480.2;",
         r"AlienSC\.UTCModJulian is set before"),
        (f"{_ALIEN_EPOCH}\nAlienSC.TAIGregorian = '15 Mar 2027 13:00:00.000';",
         r"AlienSC\.TAIGregorian is set before"),
        (f"{_ALIEN_EPOCH}\nAlienSC.CurrA1MJD = 31480.2;", r"AlienSC\.CurrA1MJD is set before"),
        # Read by GMAT; without Epoch it used to say only that Epoch was missing.
        ("AlienSC.Epoch.UTCModJulian = 31480.0;",
         r"AlienSC\.Epoch\.UTCModJulian sets the epoch in a scale of its own.*--gmat-console"),
        (f"{_ALIEN_EPOCH}\nAlienSC.Epoch.UTCModJulian = 31480.0;",
         r"AlienSC\.Epoch\.UTCModJulian sets the epoch in a scale of its own"),
    ],
    ids=["utc-modjulian", "tai-gregorian", "curr-a1-mjd", "epoch-scale-alone",
         "epoch-scale-beside"],
)
def test_without_a_console_an_epoch_field_gmat_reads_otherwise_is_refused(
    workshop_script: Path, tmp_path: Path, new: str, message: str
) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    assert _ALIEN_EPOCH in text
    path = tmp_path / "epoch.script"
    path.write_text(text.replace(_ALIEN_EPOCH, new, 1), encoding="utf-8")
    with pytest.raises(ParseError, match=message):
        parse_gmat_script(path, OrbitWinConfig())


@pytest.mark.parametrize(
    "added", ["HumanSC.AnomalyType = MA;", "HumanSC.Element1 = 9000;", "HumanSC.CartesianX = 7000;"]
)
def test_with_a_probe_what_gmat_reads_otherwise_is_compared_and_warned_of(
    workshop_script: Path, tmp_path: Path, added: str
) -> None:
    """GMAT's reading is used, and a probe that far from our six is warned of."""
    own, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    path = tmp_path / "read_otherwise.script"
    path.write_text(
        workshop_script.read_text(encoding="utf-8").replace(_TA, f"{_TA}\n{added}", 1),
        encoding="utf-8",
    )
    far = StateVector(
        (own.pursuer.initial_state.r_km[0] + 5.6e5, *own.pursuer.initial_state.r_km[1:]),
        own.pursuer.initial_state.v_km_s,
    )
    a1 = 31480.000428638676
    probe = _probe_for(
        path, {"HumanSC": far, "AlienSC": own.target.initial_state}, {"HumanSC": a1, "AlienSC": a1}
    )
    scenario, _ = parse_gmat_script(path, OrbitWinConfig(), probe=probe)
    assert scenario.pursuer.initial_state == far
    assert any("GMAT reads HumanSC's initial state 5.6e+05 km" in w for w in scenario.warnings)


def test_a_probe_reads_another_types_fields_and_skips_our_comparison(
    workshop_script: Path, tmp_path: Path
) -> None:
    own, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    path = tmp_path / "foreign.script"
    path.write_text(
        workshop_script.read_text(encoding="utf-8").replace(
            _KEPLERIAN, "HumanSC.DisplayStateType = Cartesian;", 1
        ),
        encoding="utf-8",
    )
    a1 = 31480.000428638676
    probe = _probe_for(
        path,
        {"HumanSC": own.pursuer.initial_state, "AlienSC": own.target.initial_state},
        {"HumanSC": a1, "AlienSC": a1},
    )
    scenario, _ = parse_gmat_script(path, OrbitWinConfig(), probe=probe)
    assert scenario.pursuer.initial_state == own.pursuer.initial_state
    assert scenario.pursuer.in_process_mismatch is None
    assert scenario.target.in_process_mismatch is not None


# --- files outside the run -------------------------------------------------------


def _absolute_report(workshop_script: Path, tmp_path: Path) -> tuple[Path, Path]:
    sentinel = tmp_path / "official_abs.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    text = workshop_script.read_text(encoding="utf-8")
    marker = "ReportFile1.Filename = 'ReportFile1.txt';"
    assert marker in text
    path = tmp_path / "absolute.script"
    path.write_text(text.replace(marker, f"ReportFile1.Filename = '{sentinel}';"), encoding="utf-8")
    return path, sentinel


def test_the_probe_copy_writes_none_of_the_templates_files(workshop_script: Path) -> None:
    text, _ = build_probe_script(GmatScriptDocument.read(workshop_script))
    live = [
        line for line in text.decode("utf-8").splitlines()
        if not line.startswith("%") and "ReportFile" in line
    ]
    assert live and all(PROBE_RESOURCE_NAME in line for line in live)


def test_a_report_outside_the_run_is_refused_before_launch(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Measured on R2026a: one probe with this ReportFile deleted the sentinel."""
    path, sentinel = _absolute_report(workshop_script, tmp_path)
    with pytest.raises(ParseError, match="Unsafe output filename"):
        _fake_load(path, fake_gmat, fake_startup)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_a_neutralised_report_is_not_cleared(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    import orbitwin.state_probe as state_probe

    path, sentinel = _absolute_report(workshop_script, tmp_path)
    # The probe's own gate off: it launches, and only the parse after it refuses.
    monkeypatch.setattr(state_probe, "assert_resources_supported", lambda *_args: None)
    with pytest.raises(ParseError, match="Unsafe output filename"):
        _fake_load(path, fake_gmat, fake_startup)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_each_scale_lands_in_its_own_field(
    workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """15 Mar 2027: TAI - UTC = 37 s, A1 - TAI = 0.0343817 s, TT - TAI = 32.184 s."""
    scenario, _ = _fake_load(workshop_script, fake_gmat, fake_startup)
    epoch = scenario.pursuer.probed_epoch
    assert epoch is not None
    assert (epoch.tai_mjd - epoch.utc_mjd) * 86400.0 == pytest.approx(37.0, abs=1e-5)
    assert (epoch.a1_mjd - epoch.tai_mjd) * 86400.0 == pytest.approx(0.0343817, abs=1e-5)
    assert (epoch.tt_mjd - epoch.tai_mjd) * 86400.0 == pytest.approx(32.184, abs=1e-5)
    assert abs(epoch.tdb_mjd - epoch.tt_mjd) * 86400.0 < 2.0e-3


# --- nothing reaches GMAT that could write outside the run ----------------------


def _with_resources(workshop_script: Path, tmp_path: Path, block: str) -> Path:
    path = tmp_path / "extra.script"
    path.write_text(
        workshop_script.read_text(encoding="utf-8").replace(
            "BeginMissionSequence;", block + "\nBeginMissionSequence;", 1
        ),
        encoding="utf-8",
    )
    return path


def test_an_included_report_file_never_reaches_gmat(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Measured on R2026a: the probe ran first and GMAT cleared the included file's path."""
    sentinel = tmp_path / "precious.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    included = tmp_path / "inc.script"
    included.write_text(
        f"Create ReportFile Evil;\nEvil.Filename = '{sentinel}';\n", encoding="utf-8"
    )
    path = _with_resources(workshop_script, tmp_path, f"#Include '{included}'")
    with pytest.raises(ParseError, match="Include"):
        _fake_load(path, fake_gmat, fake_startup)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


WRITERS = {
    "EphemerisFile": "Create EphemerisFile W;\nW.Spacecraft = AlienSC;\nW.Filename = '{0}';",
    "EclipseLocator": "Create EclipseLocator W;\nW.Spacecraft = AlienSC;\nW.Filename = '{0}';",
    "ContactLocator": "Create ContactLocator W;\nW.Target = AlienSC;\nW.Filename = '{0}';",
    "IntrusionLocator": "Create IntrusionLocator W;\nW.Filename = '{0}';",
    "DifferentialCorrector": "Create DifferentialCorrector W;\nW.ReportFile = '{0}';",
    "GmatFunction": "Create GmatFunction W;\nW.FunctionPath = '{0}';",
}


@pytest.mark.parametrize("kind", sorted(WRITERS))
def test_a_resource_that_writes_or_runs_files_is_refused_before_gmat(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path, kind: str
) -> None:
    sentinel = tmp_path / "precious.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    path = _with_resources(workshop_script, tmp_path, WRITERS[kind].format(sentinel))
    # The writers the rules allow are held to a relative path; the rest are never run.
    reason = (
        "Unsafe output filename"
        if kind in OUTPUT_FILE_FIELDS
        else f"creates {kind} W, which is not run: it writes files outside the run or runs code"
    )
    with pytest.raises(ParseError, match=reason):
        parse_gmat_script(path, OrbitWinConfig())
    with pytest.raises(ParseError, match=reason):
        _fake_load(path, fake_gmat, fake_startup)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_a_type_name_in_the_wrong_case_is_refused(workshop_script: Path, tmp_path: Path) -> None:
    """R2026a refused `Create reportfile` (rc 1); it is not ReportFile."""
    path = _with_resources(workshop_script, tmp_path, "Create reportfile R;")
    with pytest.raises(ParseError, match="creates reportfile R"):
        parse_gmat_script(path, OrbitWinConfig())


def test_every_shipped_template_creates_only_accepted_types(project_root: Path) -> None:
    from orbitwin.gmat_parser import assert_resources_supported

    ours = [*sorted((project_root / "examples").rglob("*.script")), ASYNC_FIXTURE]
    assert len(ours) >= 15
    theirs = organizer_scripts()
    if ORGANIZER_MATERIAL.is_file():
        # The top-level preliminary and the organizer's own two; none in the public export.
        assert len(theirs) >= 3, theirs
    scripts = [*ours, *theirs]
    for script in scripts:
        assert_resources_supported(GmatScriptDocument.read(script))


def test_the_probe_copy_keeps_only_what_places_a_craft(workshop_script: Path) -> None:
    text, _ = build_probe_script(GmatScriptDocument.read(workshop_script))
    live = {
        line.split()[1]
        for line in text.decode("utf-8").splitlines()
        if line.startswith("Create ")
    }
    assert live == {"Spacecraft", "ReportFile"}  # the ReportFile is the probe's own


def test_a_failed_probe_carries_gmats_own_error(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Its run directory is deleted, so the exception is all that is left."""
    path = tmp_path / "fails.script"
    path.write_text(
        "% FAKE_GMAT_PROBE_FAIL\n" + workshop_script.read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(GmatRunError, match=r"exit status 3\n  ERROR!!!!!! ---- forced fake probe"):
        _fake_load(path, fake_gmat, fake_startup)


def test_a_line_continuation_is_refused_by_name(workshop_script: Path, tmp_path: Path) -> None:
    """This line still reads as an assignment, so only the continuation check sees it."""
    from orbitwin.gmat_parser import _ASSIGN_RE

    line = "AlienSC.SMA = 9600 ...;"
    assert _ASSIGN_RE.match(line)
    path = tmp_path / "continued.script"
    path.write_text(
        workshop_script.read_text(encoding="utf-8").replace(
            "AlienSC.SMA = 9600;", line, 1
        ),
        encoding="utf-8",
    )
    number = path.read_text(encoding="utf-8").splitlines().index(line) + 1
    with pytest.raises(ParseError, match=f"^continued.script line {number} ends in `"):
        parse_gmat_script(path, OrbitWinConfig())


@pytest.mark.parametrize(
    ("value", "bare"),
    [
        ("OfficialReport.txt", True),
        ("ODC_2026_Team05.txt", True),
        ("Report File1.txt", True),  # R2026a wrote it into gmat-output/
        ("ODC_2026_Team(07).txt", True),  # likewise
        ("ODC_2026_Team{07}.txt", True),  # GMAT refuses it itself, writing nothing
        ("..txt", True),
        ("tab\there.txt", False),
        ("..\\..\\escape.txt", False),  # R2026a wrote it two levels up
        ("\\home\\someone\\sentinel.txt", False),  # R2026a overwrote that file
        ("C:\\x.txt", False),
        ("C:x.txt", False),
        ("sub\\..\\x.txt", False),
        ("a\\b/c.txt", False),
        ("sub/x.txt", False),
        ("../x.txt", False),
        ("/tmp/x.txt", False),
        ("..", False),
        (".", False),
        ("", False),
    ],
)
def test_only_a_bare_output_name_is_accepted(
    workshop_script: Path, tmp_path: Path, value: str, bare: bool
) -> None:
    from orbitwin.gmat_runner import assert_safe_output_filename

    text = workshop_script.read_text(encoding="utf-8")
    path = tmp_path / "named.script"
    path.write_text(
        text.replace("ReportFile1.Filename = 'ReportFile1.txt';", f"ReportFile1.Filename = '{value}';"),
        encoding="utf-8",
    )
    if bare:
        parse_gmat_script(path, OrbitWinConfig())
        assert_safe_output_filename(value)
    else:
        with pytest.raises(ParseError, match=r"ReportFile1\.Filename = .* is not a bare file name"):
            parse_gmat_script(path, OrbitWinConfig())
        # run_gmat's own check, for a script that never went through the parser.
        with pytest.raises(GmatRunError, match="is not a bare file name"):
            assert_safe_output_filename(value)


def test_the_probe_runs_outside_the_working_directory(
    workshop_script: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused probe leaves nothing behind, in the run directory or anywhere else."""
    import tempfile

    import orbitwin.state_probe as state_probe

    roots: list[Path] = []

    def refuse(script, console, startup, root, **kwargs):  # type: ignore[no-untyped-def]
        roots.append(Path(root))
        raise GmatRunError("test: the probe is refused")

    monkeypatch.setattr(state_probe, "probe_initial_states", refuse)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(GmatRunError, match="probe is refused"):
        state_probe.load_scenario(
            workshop_script, OrbitWinConfig(), gmat_console="console", startup_file="startup"
        )
    (root,) = roots
    assert root.is_relative_to(Path(tempfile.gettempdir()).resolve())
    assert not root.is_relative_to(tmp_path.resolve()) and not root.exists()


def test_the_probe_reads_a_field_before_its_comment(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """The probe binds a craft by the fields the parser reads, comment or not."""
    from orbitwin.state_probe import _resource_section

    script = tmp_path / "commented.script"
    script.write_text(workshop_script.read_text(encoding="ascii").replace(
        "AlienSC.SMA = 9600;", "AlienSC.SMA = 9600; % km", 1),
        encoding="ascii")
    _, _, fields, _ = _resource_section(GmatScriptDocument.read(script))
    assert fields["AlienSC"]["SMA"] == "9600"
    scenario, _ = _fake_load(script, fake_gmat, fake_startup)
    assert scenario.target.name == "AlienSC"


@pytest.mark.parametrize("probe", ["state", "alignment"])
def test_a_failed_probe_quotes_gmats_unmarked_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: str
) -> None:
    """SolarSystem prints `ERROR loading kernel ...` with no record marker; its run
    directory is deleted, so the exception has to carry it."""
    import orbitwin.state_probe as state_probe
    from orbitwin.models import EpochAlignment
    from orbitwin.state_probe import probe_alignment, probe_initial_states

    def failed(console, script, workdir, *args, **kwargs):  # type: ignore[no-untyped-def]
        out = Path(workdir)
        out.mkdir(parents=True)
        (out / "stdout.txt").write_text(
            "ERROR loading kernel /gmat/data/planetary_ephem/spk/de999.bsp\n", encoding="utf-8")
        return dataclasses.replace(
            _run_record(), success=False, return_code=1, workdir=str(out),
            stdout_path=str(out / "stdout.txt"), error="GmatConsole returned exit status 1",
        )

    monkeypatch.setattr(state_probe, "run_gmat", failed)
    with pytest.raises(GmatRunError, match=r"exit status 1\n  ERROR loading kernel "):
        if probe == "state":
            probe_initial_states(ASYNC_FIXTURE, "console", "startup", tmp_path / "p",
                                 timeout_s=1.0)
        else:
            line = "Propagate DefaultProp(SC_Alien) {SC_Alien.UTCModJulian = SC_Human.UTCModJulian};\n"
            alignment = EpochAlignment("SC_Alien", "SC_Human", "DefaultProp", "epoch-equality",
                                       1.0, line, 51)
            probe_alignment(ASYNC_FIXTURE, alignment, "binding", "console", "startup",
                            tmp_path / "p", timeout_s=1.0)


def test_a_name_created_twice_is_its_first_create(workshop_script: Path, tmp_path: Path) -> None:
    """GMAT ignores a repeated Create ("already exists, so ignored"), so the first decides."""
    from orbitwin.gmat_parser import assert_resources_supported
    from orbitwin.state_probe import _resource_section

    path = tmp_path / "twice.script"
    path.write_text(workshop_script.read_text(encoding="utf-8").replace(
        "BeginMissionSequence;",
        "Create Spacecraft HumanSC;\nCreate ReportFile Leak;\nCreate Variable Leak;\n"
        "Leak.Filename = '../leak.txt';\nBeginMissionSequence;", 1), encoding="utf-8")
    document = GmatScriptDocument.read(path)
    _, craft, _, created = _resource_section(document)
    assert sorted(craft) == ["AlienSC", "HumanSC"]
    assert created["Leak"] == "reportfile"
    # A ReportFile's Filename, however it is created again afterwards.
    with pytest.raises(ParseError, match="not a bare file name"):
        assert_resources_supported(document)
