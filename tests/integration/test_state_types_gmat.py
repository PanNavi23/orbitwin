"""Every state type and epoch format GMAT reads reaches the solver as GMAT reads it.

GMAT writes R6's two craft in each CoordinateSystem and DisplayStateType at
Precision 17; the rewritten script is probed and must give the baseline's
EarthMJ2000Eq state and A1 epoch. docs/ASYNC_START_AND_INSERTION.md, "Other
state types and epoch formats".
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from conftest import require_gmat

from orbitwin.config import load_config
from orbitwin.errors import GmatRunError, ParseError
from orbitwin.gmat_parser import (
    PROBE_POSITION_TOLERANCE_KM,
    PROBE_VELOCITY_TOLERANCE_KM_S,
    STATE_TYPE_FIELDS,
    parse_gmat_script,
)
from orbitwin.gmat_runner import run_gmat
from orbitwin.models import StateProbe
from orbitwin.report_parser import parse_numeric_report
from orbitwin.state_probe import probe_initial_states

ROOT = Path(__file__).resolve().parents[2]
R6 = ROOT / "examples" / "ranking" / "R6_InsertedLate_Aligned.script"
HORIZON_S = 6711.667

pytestmark = pytest.mark.gmat

#: Largest measured round trip of a closed-form type, per craft: SC_Human
#: 1.4e-11 km and 1.4e-14 km/s (EarthFixed Delaunay); SC_Alien, a 1e5 km
#: hyperbola, 5.5e-8 km and 4.8e-12 km/s (EarthFixed Keplerian, 16 digits).
ROUND_TRIP = {"SC_Human": (1.0e-10, 1.0e-13), "SC_Alien": (1.0e-7, 1.0e-11)}
#: GMAT's iterative conversions: Planetodetic 1.06e-4 km and 1.75e-8 km/s
#: (SC_Human), BrouwerMeanLong 2.9e-5 km and 3.2e-8 km/s.
ITERATIVE_KM, ITERATIVE_KM_S = 5.0e-4, 1.0e-7
_ITERATIVE = {"Planetodetic", "BrouwerMeanShort", "BrouwerMeanLong"}

_CLOSED_FORM = [
    "Cartesian", "Keplerian", "ModifiedKeplerian", "SphericalAZFPA", "SphericalRADEC",
    "Equinoctial", "AlternateEquinoctial", "ModifiedEquinoctial", "Delaunay",
]
#: Measured on R2026a. SC_Human is circular in EarthMJ2000Eq, so it has no
#: asymptote there; it has one in EarthFixed, where its speed is Earth-relative.
#: SC_Alien is hyperbolic, which only the Keplerian family, the spherical types,
#: ModifiedEquinoctial and the asymptotes describe. Planetodetic needs body-fixed axes.
READ = [
    *(("SC_Human", "EarthMJ2000Eq", kind) for kind in _CLOSED_FORM),
    ("SC_Human", "EarthMJ2000Eq", "BrouwerMeanShort"),
    ("SC_Human", "EarthMJ2000Eq", "BrouwerMeanLong"),
    *(("SC_Human", "EarthFixed", kind) for kind in STATE_TYPE_FIELDS),
    *(
        ("SC_Alien", system, kind)
        for system in ("EarthMJ2000Eq", "EarthFixed")
        for kind in (
            "Cartesian", "Keplerian", "ModifiedKeplerian", "SphericalAZFPA", "SphericalRADEC",
            "ModifiedEquinoctial", "OutgoingAsymptote", "IncomingAsymptote",
        )
    ),
    ("SC_Alien", "EarthFixed", "Planetodetic"),
]


def _resources() -> str:
    """R6's resources with both craft at 02:00:00 UTC and no ReportFile."""
    text = R6.read_text(encoding="utf-8").replace(
        "SC_Alien.Epoch = '09 Nov 2026 01:30:00.000';",
        "SC_Alien.Epoch = '09 Nov 2026 02:00:00.000';",
    )
    return text.split("Create ReportFile ReportFile1;")[0]


MISSION = (
    "BeginMissionSequence;\n"
    f"Propagate DefaultProp(SC_Alien, SC_Human) {{SC_Human.ElapsedSecs = {HORIZON_S}}};\n"
)


def _config():
    config = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    config = dataclasses.replace(config, policy=dataclasses.replace(config.policy, team_number=54))
    return dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, horizon_s=HORIZON_S)
    )


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, StateProbe]:
    # Not from the gmat fixture, which is per test.
    gmat = require_gmat()
    root = tmp_path_factory.mktemp("state_types")
    script = root / "base.script"
    script.write_text(_resources() + MISSION, encoding="utf-8")
    return script, probe_initial_states(script, *gmat, root / "probe", timeout_s=120)


def _state_lines(text: str, craft: str) -> tuple[int, int]:
    """R6's CoordinateSystem, DisplayStateType and six state lines of ``craft``."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{craft}.CoordinateSystem"))
    kind = lines[start + 1].removeprefix(f"{craft}.DisplayStateType = ").strip().rstrip(";")
    assert kind in STATE_TYPE_FIELDS, lines[start + 1]
    assert {
        line.split("=")[0].strip().removeprefix(f"{craft}.") for line in lines[start + 2:start + 8]
    } == set(STATE_TYPE_FIELDS[kind])
    return start, start + 8


def _with_state(
    text: str, craft: str, system: str, kind: str, fields: list[tuple[str, str]]
) -> str:
    lines = text.splitlines(keepends=True)
    start, end = _state_lines(text, craft)
    block = [f"{craft}.CoordinateSystem = {system};\n", f"{craft}.DisplayStateType = {kind};\n"]
    return "".join(
        lines[:start] + block + [f"{craft}.{name} = {value};\n" for name, value in fields]
        + lines[end:]
    )


def _gmat_fields(
    craft: str, system: str, kind: str, workdir: Path, gmat: tuple[str, str],
    resources: str | None = None,
) -> list[tuple[str, str]]:
    """The baseline craft's fields in (system, kind), as GMAT writes them at Precision 17."""
    names = STATE_TYPE_FIELDS[kind]
    script = workdir / "write.script"
    workdir.mkdir(parents=True)
    script.write_text(
        (_resources() if resources is None else resources)
        + "Create ReportFile W;\nW.Filename = 'w.txt';\nW.Precision = 17;\nBeginMissionSequence;\n"
        + f"{craft}.CoordinateSystem = {system};\n{craft}.DisplayStateType = {kind};\n"
        + f"Write {' '.join(f'{craft}.{name}' for name in names)} "
        + "{Style = Concise, ReportFile = W, MessageWindow = false};\n",
        encoding="utf-8",
    )
    result = run_gmat(gmat[0], script, workdir / "run", ["w.txt"], startup_file=gmat[1],
                      timeout_s=120)
    assert result.success, result.error
    values = Path(result.reports["w.txt"]).read_text(encoding="utf-8").splitlines()[-1].split()
    assert len(values) == len(names)
    return list(zip(names, values, strict=True))


@pytest.mark.parametrize(("craft", "system", "kind"), READ, ids=["-".join(case) for case in READ])
@pytest.mark.parametrize("digits", [17, 16], ids=["precision-17", "script-16"])
def test_each_state_type_probes_to_the_baseline_state(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe],
    craft: str, system: str, kind: str, digits: int,
) -> None:
    """At 17 digits and at the 16 GMAT writes in a saved script."""
    base, base_probe = baseline
    fields = [
        (name, value if digits == 17 else format(float(value), ".16g"))
        for name, value in _gmat_fields(craft, system, kind, tmp_path / "write", gmat)
    ]
    script = tmp_path / "variant.script"
    script.write_text(
        _with_state(base.read_text(encoding="utf-8"), craft, system, kind, fields), encoding="utf-8"
    )
    probe = probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)
    mine, theirs = probe.craft_named(craft), base_probe.craft_named(craft)
    assert mine is not None and theirs is not None
    assert mine.epoch == theirs.epoch
    km, km_s = (ITERATIVE_KM, ITERATIVE_KM_S) if kind in _ITERATIVE else ROUND_TRIP[craft]
    assert float(np.linalg.norm(mine.state.r - theirs.state.r)) <= km
    assert float(np.linalg.norm(mine.state.v - theirs.state.v)) <= km_s
    scenario, _ = parse_gmat_script(script, _config(), probe=probe)
    spec = scenario.pursuer if craft == "SC_Human" else scenario.target
    assert (spec.coordinate_system, spec.display_state_type) == (system, kind)
    assert spec.initial_state == mine.state
    assert scenario.epoch_offset_s == 0.0


def test_gmat_reads_alternate_equinoctial_by_line_order_and_the_probe_follows(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe]
) -> None:
    """The same six values in alphabetical order, SMA last, land 262 km away.

    Measured on R2026a with SC_Human; GMAT writes SMA first and reads that back.
    The probe is what GMAT flies either way: its first propagated row.
    """
    base, base_probe = baseline
    fields = _gmat_fields(
        "SC_Human", "EarthMJ2000Eq", "AlternateEquinoctial", tmp_path / "write", gmat
    )
    script = tmp_path / "reordered.script"
    script.write_text(
        _with_state(base.read_text(encoding="utf-8"), "SC_Human", "EarthMJ2000Eq",
                    "AlternateEquinoctial", sorted(fields)),
        encoding="utf-8",
    )
    probe = probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)
    mine, theirs = probe.craft_named("SC_Human"), base_probe.craft_named("SC_Human")
    assert mine is not None and theirs is not None
    # 261.625 km on R2026a. The values come from R6's EarthFixed SC_Human, which
    # another EOP file moves by metres.
    assert float(np.linalg.norm(mine.state.r - theirs.state.r)) == pytest.approx(261.625, abs=0.5)
    columns = ["SC_Human.A1ModJulian"] + [
        f"SC_Human.EarthMJ2000Eq.{axis}" for axis in ("X", "Y", "Z", "VX", "VY", "VZ")
    ]
    flown = tmp_path / "first_row.script"
    flown.write_text(
        script.read_text(encoding="utf-8").split("BeginMissionSequence;")[0]
        + "Create ReportFile First;\nFirst.Filename = 'first.txt';\nFirst.Precision = 17;\n"
        + "First.WriteHeaders = true;\nFirst.Add = {" + ", ".join(columns) + "};\n"
        + "BeginMissionSequence;\nPropagate DefaultProp(SC_Human) {SC_Human.ElapsedSecs = 10};\n",
        encoding="utf-8",
    )
    result = run_gmat(gmat[0], flown, tmp_path / "run", ["first.txt"], startup_file=gmat[1],
                      timeout_s=120)
    assert result.success, result.error
    first = parse_numeric_report(result.reports["first.txt"]).data[0]
    assert first[0] == mine.epoch.a1_mjd
    assert tuple(first[1:]) == (*mine.state.r_km, *mine.state.v_km_s)


def test_planetodetic_without_body_fixed_axes_fails_with_gmats_words(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe]
) -> None:
    base, _ = baseline
    fields = _gmat_fields("SC_Human", "EarthFixed", "Planetodetic", tmp_path / "write", gmat)
    script = tmp_path / "inertial_planetodetic.script"
    script.write_text(
        _with_state(base.read_text(encoding="utf-8"), "SC_Human", "EarthMJ2000Eq", "Planetodetic",
                    fields),
        encoding="utf-8",
    )
    with pytest.raises(
        GmatRunError, match="Planetodetic state type is only defined for coordinate systems with "
        "BodyFixed axes"
    ):
        probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)


def _epoch(craft: str, block: str) -> tuple[str, str]:
    return (
        f"{craft}.DateFormat = UTCGregorian;\n{craft}.Epoch = '09 Nov 2026 02:00:00.000';\n", block
    )


@pytest.mark.parametrize("craft", ["SC_Human", "SC_Alien"])
@pytest.mark.parametrize("scale", ["A1", "UTC", "TAI", "TT", "TDB"])
@pytest.mark.parametrize("form", ["DateFormat", "unquoted", "Epoch.scale"])
def test_each_modjulian_epoch_is_read_bit_for_bit(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe],
    craft: str, scale: str, form: str,
) -> None:
    """The shortest repr of the probed epoch gives back the same A1 day and state."""
    base, base_probe = baseline
    theirs = base_probe.craft_named(craft)
    assert theirs is not None
    mjd = repr(float(getattr(theirs.epoch, f"{scale.lower()}_mjd")))
    block = {
        "DateFormat": f"{craft}.DateFormat = {scale}ModJulian;\n{craft}.Epoch = '{mjd}';\n",
        "unquoted": f"{craft}.DateFormat = {scale}ModJulian;\n{craft}.Epoch = {mjd};\n",
        "Epoch.scale": f"{craft}.Epoch.{scale}ModJulian = {mjd};\n",
    }[form]
    old, new = _epoch(craft, block)
    text = base.read_text(encoding="utf-8")
    assert old in text
    script = tmp_path / "epoch.script"
    script.write_text(text.replace(old, new), encoding="utf-8")
    probe = probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)
    mine = probe.craft_named(craft)
    assert mine is not None
    assert (mine.epoch, mine.state) == (theirs.epoch, theirs.state)
    scenario, _ = parse_gmat_script(script, _config(), probe=probe)
    assert scenario.epoch_offset_s == 0.0


def test_a_16_digit_modjulian_epoch_is_one_start(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe]
) -> None:
    """Measured: one A1 double step (0.31 µs) off, which moves EarthFixed SC_Human 1.2e-7 km."""
    base, base_probe = baseline
    theirs = base_probe.craft_named("SC_Human")
    assert theirs is not None
    old, new = _epoch(
        "SC_Human",
        f"SC_Human.DateFormat = UTCModJulian;\nSC_Human.Epoch = '{theirs.epoch.utc_mjd:.16g}';\n",
    )
    text = base.read_text(encoding="utf-8")
    assert old in text
    script = tmp_path / "sixteen.script"
    script.write_text(text.replace(old, new), encoding="utf-8")
    probe = probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)
    mine = probe.craft_named("SC_Human")
    assert mine is not None
    assert abs(mine.epoch.a1_mjd - theirs.epoch.a1_mjd) * 86400.0 <= 1.0e-6
    assert float(np.linalg.norm(mine.state.r - theirs.state.r)) <= 1.0e-6
    scenario, _ = parse_gmat_script(script, _config(), probe=probe)
    assert scenario.epoch_offset_s is not None and abs(scenario.epoch_offset_s) <= 1.0e-6


#: GMAT's Gregorian text of 02:00:00 UTC, to the millisecond it prints.
GREGORIAN = {
    "A1": "09 Nov 2026 02:00:37.034", "TAI": "09 Nov 2026 02:00:37.000",
    "TT": "09 Nov 2026 02:01:09.184", "TDB": "09 Nov 2026 02:01:09.183",
}


@pytest.mark.parametrize("scale", sorted(GREGORIAN))
def test_a_gregorian_epoch_is_one_start_only_when_its_millisecond_is(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe], scale: str
) -> None:
    """TAI and TT are whole milliseconds from UTC; A1 (34.3817 ms) and TDB are not.

    Measured: TAI and TT land one A1 double step (0.31 µs) away and start with
    SC_Alien; A1 and TDB land 0.38 and 0.37 ms away and the craft start apart,
    which the parser takes up as any asynchronous start, not as one start.
    """
    base, base_probe = baseline
    old, new = _epoch(
        "SC_Human",
        f"SC_Human.DateFormat = {scale}Gregorian;\nSC_Human.Epoch = '{GREGORIAN[scale]}';\n",
    )
    text = base.read_text(encoding="utf-8")
    assert old in text
    script = tmp_path / "gregorian.script"
    script.write_text(text.replace(old, new), encoding="utf-8")
    probe = probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)
    mine, theirs = probe.craft_named("SC_Human"), base_probe.craft_named("SC_Human")
    assert mine is not None and theirs is not None
    offset_s = (mine.epoch.a1_mjd - theirs.epoch.a1_mjd) * 86400.0
    if scale in ("TAI", "TT"):
        assert abs(offset_s) <= 1.0e-6
        scenario, _ = parse_gmat_script(script, _config(), probe=probe)
        assert scenario.epoch_offset_s is not None and abs(scenario.epoch_offset_s) <= 1.0e-6
    else:
        assert 3.0e-4 < abs(offset_s) < 4.0e-4
        with pytest.raises(ParseError, match=r"start 0\.0003[78]\d* s apart"):
            parse_gmat_script(script, _config(), probe=probe)


@pytest.mark.parametrize(
    "field",
    [f"{scale}{form}" for scale in ("A1", "TAI", "UTC", "TT", "TDB")
     for form in ("ModJulian", "Gregorian")] + ["CurrA1MJD"],
)
def test_an_epoch_field_gmat_takes_only_in_the_mission_fails_closed(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe], field: str
) -> None:
    """GMAT refuses it before BeginMissionSequence; tests/unit covers the in-process reader."""
    base, _ = baseline
    value = "'09 Nov 2026 03:00:00.000'" if field.endswith("Gregorian") else "31353.6"
    old = "SC_Alien.Epoch = '09 Nov 2026 02:00:00.000';\n"
    script = tmp_path / "bare.script"
    script.write_text(
        base.read_text(encoding="utf-8").replace(old, f"{old}SC_Alien.{field} = {value};\n"),
        encoding="utf-8",
    )
    with pytest.raises(GmatRunError, match=f'The field name "{field}" on object "SC_Alien" is not '
                       "permitted"):
        probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)


@pytest.mark.parametrize("anomaly", ["MA", "EA", "HA"])
@pytest.mark.parametrize("keep_ta", [True, False], ids=["beside-ta", "instead-of-ta"])
def test_an_anomaly_other_than_ta_fails_with_gmats_words(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe], anomaly: str,
    keep_ta: bool,
) -> None:
    """tests/unit covers the in-process reader, which read TA and ignored it."""
    base, _ = baseline
    fields = _gmat_fields("SC_Alien", "EarthMJ2000Eq", "Keplerian", tmp_path / "write", gmat)
    kept = fields if keep_ta else [(name, value) for name, value in fields if name != "TA"]
    script = tmp_path / "anomaly.script"
    script.write_text(
        _with_state(base.read_text(encoding="utf-8"), "SC_Alien", "EarthMJ2000Eq", "Keplerian",
                    [*kept, (anomaly, "10")]),
        encoding="utf-8",
    )
    with pytest.raises(GmatRunError, match="anomaly of type other than True Anomaly not currently"):
        probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)


@pytest.mark.parametrize(("kind", "extra"), [("Keplerian", ("X", "-20000")),
                                             ("Cartesian", ("SMA", "-50000"))])
def test_fields_of_two_state_types_fail_with_gmats_words(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe], kind: str,
    extra: tuple[str, str],
) -> None:
    base, _ = baseline
    fields = _gmat_fields("SC_Alien", "EarthMJ2000Eq", kind, tmp_path / "write", gmat)
    script = tmp_path / "two_types.script"
    script.write_text(
        _with_state(base.read_text(encoding="utf-8"), "SC_Alien", "EarthMJ2000Eq", kind,
                    [*fields, extra]),
        encoding="utf-8",
    )
    with pytest.raises(GmatRunError, match="not contained in the same state type"):
        probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)


@pytest.mark.parametrize(("declared", "given"), [("Keplerian", "Cartesian"),
                                                 ("Cartesian", "Keplerian")])
def test_one_other_types_fields_are_read_as_that_type(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe], declared: str,
    given: str,
) -> None:
    """Whatever DisplayStateType says; tests/unit has the in-process reader refuse them."""
    base, base_probe = baseline
    fields = _gmat_fields("SC_Alien", "EarthMJ2000Eq", given, tmp_path / "write", gmat)
    script = tmp_path / "other_type.script"
    script.write_text(
        _with_state(base.read_text(encoding="utf-8"), "SC_Alien", "EarthMJ2000Eq", declared,
                    fields),
        encoding="utf-8",
    )
    probe = probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)
    mine, theirs = probe.craft_named("SC_Alien"), base_probe.craft_named("SC_Alien")
    assert mine is not None and theirs is not None
    km, km_s = ROUND_TRIP["SC_Alien"]
    assert float(np.linalg.norm(mine.state.r - theirs.state.r)) <= km
    assert float(np.linalg.norm(mine.state.v - theirs.state.v)) <= km_s
    scenario, _ = parse_gmat_script(script, _config(), probe=probe)
    assert scenario.target.initial_state == mine.state
    assert scenario.target.in_process_mismatch is None


@pytest.mark.parametrize("order", ["alphabetical", "reversed"])
def test_keplerian_is_read_in_any_line_order(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe], order: str
) -> None:
    """Unlike AlternateEquinoctial; the in-process reader ignores the order."""
    base, base_probe = baseline
    fields = _gmat_fields("SC_Alien", "EarthMJ2000Eq", "Keplerian", tmp_path / "write", gmat)
    lines = sorted(fields) if order == "alphabetical" else fields[::-1]
    script = tmp_path / "order.script"
    script.write_text(
        _with_state(base.read_text(encoding="utf-8"), "SC_Alien", "EarthMJ2000Eq", "Keplerian",
                    lines),
        encoding="utf-8",
    )
    probe = probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)
    mine, theirs = probe.craft_named("SC_Alien"), base_probe.craft_named("SC_Alien")
    assert mine is not None and theirs is not None
    km, km_s = ROUND_TRIP["SC_Alien"]
    assert float(np.linalg.norm(mine.state.r - theirs.state.r)) <= km
    assert float(np.linalg.norm(mine.state.v - theirs.state.v)) <= km_s


SINGULAR = {
    "circular": ("8000", "0", "30", "40", "50", "60"),
    "equatorial": ("8000", "0.1", "0", "40", "50", "60"),
    "circular-equatorial": ("8000", "0", "0", "40", "50", "60"),
}


@pytest.mark.parametrize("case", sorted(SINGULAR))
def test_the_in_process_reader_agrees_where_aop_or_raan_is_undefined(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe], case: str
) -> None:
    """Measured within 1.9e-12 km; the tolerance is the parser's own."""
    base, _ = baseline
    human = _gmat_fields("SC_Human", "EarthMJ2000Eq", "Cartesian", tmp_path / "write", gmat)
    text = _with_state(base.read_text(encoding="utf-8"), "SC_Human", "EarthMJ2000Eq",
                       "Cartesian", human)
    elements = list(zip(STATE_TYPE_FIELDS["Keplerian"], SINGULAR[case], strict=True))
    script = tmp_path / "singular.script"
    script.write_text(
        _with_state(text, "SC_Alien", "EarthMJ2000Eq", "Keplerian", elements), encoding="utf-8"
    )
    probe = probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)
    scenario, _ = parse_gmat_script(script, _config(), probe=probe)
    assert scenario.target.in_process_mismatch is not None
    position_km, velocity_km_s = scenario.target.in_process_mismatch
    assert position_km <= PROBE_POSITION_TOLERANCE_KM
    assert velocity_km_s <= PROBE_VELOCITY_TOLERANCE_KM_S


def test_anomaly_type_ma_moves_the_keplerian_state(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe]
) -> None:
    """Measured 5.6e5 km; tests/unit has the in-process reader refuse it."""
    base, base_probe = baseline
    fields = _gmat_fields("SC_Alien", "EarthMJ2000Eq", "Keplerian", tmp_path / "write", gmat)
    script = tmp_path / "anomaly_type.script"
    script.write_text(
        _with_state(base.read_text(encoding="utf-8"), "SC_Alien", "EarthMJ2000Eq", "Keplerian",
                    [*fields, ("AnomalyType", "MA")]),
        encoding="utf-8",
    )
    probe = probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)
    mine, theirs = probe.craft_named("SC_Alien"), base_probe.craft_named("SC_Alien")
    assert mine is not None and theirs is not None
    assert float(np.linalg.norm(mine.state.r - theirs.state.r)) > 1.0e3


@pytest.mark.parametrize(("field", "value"), [("Element1", "-60000"), ("CartesianX", "-30000")])
def test_gmat_reads_an_alias_into_the_state_beside_the_six(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe], field: str,
    value: str,
) -> None:
    """Measured 1.5e4 and 2.6e4 km; tests/unit has the in-process reader refuse them."""
    base, base_probe = baseline
    fields = _gmat_fields("SC_Alien", "EarthMJ2000Eq", "Keplerian", tmp_path / "write", gmat)
    script = tmp_path / "alias.script"
    script.write_text(
        _with_state(base.read_text(encoding="utf-8"), "SC_Alien", "EarthMJ2000Eq", "Keplerian",
                    [*fields, (field, value)]),
        encoding="utf-8",
    )
    probe = probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)
    mine, theirs = probe.craft_named("SC_Alien"), base_probe.craft_named("SC_Alien")
    assert mine is not None and theirs is not None
    assert float(np.linalg.norm(mine.state.r - theirs.state.r)) > 1.0e3


@pytest.mark.parametrize(
    ("field", "value"),
    [("ta", "10"), ("Sma", "-60000"), ("element1", "-60000"), ("anomalytype", "MA"),
     ("epoch", "'09 Nov 2026 03:00:00.000'"), ("utcmodjulian", "31353.6"),
     ("displaystatetype", "Cartesian"), ("dateformat", "TAIModJulian"),
     ("coordinatesystem", "EarthFixed")],
)
def test_a_field_in_the_wrong_case_fails_with_gmats_words(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe], field: str,
    value: str,
) -> None:
    """tests/unit has the in-process reader refuse them too."""
    base, _ = baseline
    fields = _gmat_fields("SC_Alien", "EarthMJ2000Eq", "Keplerian", tmp_path / "write", gmat)
    script = tmp_path / "cased.script"
    script.write_text(
        _with_state(base.read_text(encoding="utf-8"), "SC_Alien", "EarthMJ2000Eq", "Keplerian",
                    [*fields, (field, value)]),
        encoding="utf-8",
    )
    with pytest.raises(GmatRunError, match=f'The field name "{field}" on object "SC_Alien" is '
                       "not permitted"):
        probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)


def test_a_state_type_in_the_wrong_case_fails_with_gmats_words(
    tmp_path: Path, gmat: tuple[str, str], baseline: tuple[Path, StateProbe]
) -> None:
    base, _ = baseline
    fields = _gmat_fields("SC_Alien", "EarthMJ2000Eq", "Keplerian", tmp_path / "write", gmat)
    script = tmp_path / "lower.script"
    script.write_text(
        _with_state(base.read_text(encoding="utf-8"), "SC_Alien", "EarthMJ2000Eq", "keplerian",
                    fields),
        encoding="utf-8",
    )
    with pytest.raises(GmatRunError, match="Unknown state element representation: keplerian"):
        probe_initial_states(script, *gmat, tmp_path / "probe", timeout_s=120)


def test_solve_and_certify_a_planetodetic_insertion_end_to_end(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    """The renderer, the policy guard, verify's report sync and certify on a template
    whose craft are neither SphericalAZFPA nor in-process readable."""
    from conftest import quick_config

    from orbitwin.io import read_candidates
    from orbitwin.pipeline import solve_scenario
    from orbitwin.verify import certify_candidate

    fixture = ROOT / "tests" / "fixtures" / "async" / "insertion_and_async_start.script"
    resources = (
        fixture.read_text(encoding="utf-8").split("BeginMissionSequence;")[0]
        .replace("SC_Human.Epoch = '01 Jan 2000 15:00:00.000';",
                 "SC_Human.Epoch = '01 Jan 2000 00:00:00.000';")
        # 7.2 km/s puts perigee near 6031 km, inside the Earth.
        .replace("SC_Human.VMAG = 7.2;", "SC_Human.VMAG = 7.4;")
    )
    text = resources
    for craft, system, kind in (
        ("SC_Human", "EarthFixed", "Planetodetic"),
        ("SC_Alien", "EarthMJ2000Eq", "ModifiedEquinoctial"),
    ):
        fields = _gmat_fields(craft, system, kind, tmp_path / craft, gmat, resources)
        text = _with_state(text, craft, system, kind, fields)
    script = tmp_path / "planetodetic.script"
    script.write_text(
        text + "BeginMissionSequence;\n"
        "Propagate DefaultProp(SC_Alien, SC_Human) {SC_Human.ElapsedSecs = 3000};\n",
        encoding="utf-8",
    )
    base = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    base = dataclasses.replace(base, policy=dataclasses.replace(base.policy, team_number=54))
    config = dataclasses.replace(
        base,
        search=dataclasses.replace(quick_config(local_refine=True).search, workers=1),
        policy=dataclasses.replace(base.policy, horizon_s=3000.0),
        gmat=dataclasses.replace(base.gmat, verify_top=2, reruns=1, workers=1),
    )
    summary = solve_scenario(
        script, config, tmp_path / "solve", gmat_console=gmat[0], startup_file=gmat[1]
    )
    assert summary["gmat_verified_count"] >= 1
    best = next(
        candidate
        for candidate in read_candidates(tmp_path / "solve" / "candidates.json")
        if candidate.candidate_id == summary["best_candidate"]["candidate_id"]
    )
    certification = certify_candidate(
        script, best, config, gmat[0], gmat[1], tmp_path / "certify", runs=1
    )
    assert certification["conferred_stage"] == "public_gmat_verified"
    craft = {entry["name"] for entry in certification["state_probe"]["craft"]}
    assert craft == {"SC_Human", "SC_Alien"}
