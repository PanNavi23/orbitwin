from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from orbitwin.config import OrbitWinConfig, PolicyConfig, load_config
from orbitwin.errors import ParseError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.orbit import cartesian_to_keplerian

ROOT = Path(__file__).resolve().parents[2]
RANKING = ROOT / "examples" / "ranking"
CONFIG = ROOT / "config" / "ranking-rehearsal.yaml"


#: Ranking rules sec. 7, as GMAT R2026a accepts it; README.md says why each value.
_FULL_MODEL = {
    "CentralBody": "Earth",
    "PrimaryBodies": "{Earth}",
    "PointMasses": "{Sun, Luna}",
    "Drag.AtmosphereModel": "NRLMSISE00",
    "Drag.HistoricWeatherSource": "'ConstantFluxAndGeoMag'",
    "Drag.PredictedWeatherSource": "'ConstantFluxAndGeoMag'",
    "Drag.F107": "150",
    "Drag.F107A": "150",
    "Drag.MagneticIndex": "3",
    "Drag.DragModel": "'SPADFile'",
    "SRP": "On",
    "SRP.Flux": "1367",
    "SRP.SRPModel": "'SPADFile'",
    "SRP.Nominal_Sun": "149597870.691",
    "RelativisticCorrection": "On",
    "ErrorControl": "RSSStep",
    "GravityField.Earth.Degree": "20",
    "GravityField.Earth.Order": "20",
    "GravityField.Earth.StmLimit": "100",
    "GravityField.Earth.PotentialFile": "'EGM96.cof'",
    "GravityField.Earth.TideModel": "'SolidAndPole'",
}
_PROPAGATOR = {
    "Type": "RungeKutta89",
    "InitialStepSize": "1",
    "Accuracy": "9.999999999999999e-12",
    "MinStep": "0.001",
    "MaxStep": "1",
    "MaxStepAttempts": "50",
    "StopIfAccuracyIsViolated": "true",
}
#: R3: SC_Alien on a propagator of its own, on the same model (rule 2 as corrected).
_TWO_PROPAGATORS = "R3_FastHighPerigee_TwoPropagators.script"
#: Every spacecraft field except the six elements: mass and SPAD files feed drag
#: and SRP (DryMass 850 -> 1700 moves R1's closest approach only 10.5 m).
_SPAD_FILE = "'../data/vehicle/spad/SphericalModel.spo'"
_CRAFT = {
    "DateFormat": "UTCGregorian",
    "Epoch": "'09 Nov 2026 02:00:00.000'",
    "CoordinateSystem": "EarthMJ2000Eq",
    "DisplayStateType": "Keplerian",
    "DryMass": "850",
    "Cd": "2.2",
    "Cr": "1.8",
    "DragArea": "15",
    "SRPArea": "1",
    "SPADDragFile": _SPAD_FILE,
    "SPADDragScaleFactor": "1",
    "SPADSRPFile": _SPAD_FILE,
    "SPADSRPScaleFactor": "1",
    "AtmosDensityScaleFactor": "1",
    "ExtendedMassPropertiesModel": "'None'",
    "Id": "'SatId'",
    "Attitude": "CoordinateSystemFixed",
    "SPADSRPInterpolationMethod": "Bilinear",
    "SPADDragInterpolationMethod": "Bilinear",
    "AttitudeCoordinateSystem": "EarthMJ2000Eq",
    "EulerAngleSequence": "'321'",
}
_NAIF = {
    "SC_Alien": {"NAIFId": "-10000001", "NAIFIdReferenceFrame": "-9000001"},
    "SC_Human": {"NAIFId": "-10001001", "NAIFIdReferenceFrame": "-9001001"},
}
_ELEMENTS = ("SMA", "ECC", "INC", "RAAN", "AOP", "TA")


def _table() -> list[dict[str, Any]]:
    data = yaml.safe_load((RANKING / "scenarios.yaml").read_text(encoding="utf-8"))
    return list(data["scenarios"])


def _with_horizon(horizon_s: float) -> OrbitWinConfig:
    config = load_config(CONFIG)
    return dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, horizon_s=horizon_s)
    )


def test_the_rehearsal_config_carries_the_ranking_limits() -> None:
    policy = load_config(CONFIG).policy
    assert policy.rules == "ranking"
    assert (policy.dv_limit_km_s, policy.dv_safe_limit_km_s) == (0.25, 0.249)
    assert (policy.total_dv_limit_km_s, policy.total_dv_safe_limit_km_s) == (3.0, 2.99)
    # T_max differs per scenario, so the shared file must not carry one.
    assert policy.horizon_s is None


def _async_table() -> list[dict[str, Any]]:
    data = yaml.safe_load((RANKING / "async_scenarios.yaml").read_text(encoding="utf-8"))
    return list(data["scenarios"])


def test_every_script_has_a_row_and_every_row_a_script() -> None:
    rows = [row["script"] for row in (*_table(), *_async_table())]
    assert sorted(rows) == sorted(path.name for path in RANKING.glob("*.script"))


def test_each_asynchronous_row_carries_its_sources_boundary() -> None:
    sources = {row["script"]: row for row in _table()}
    for row in _async_table():
        source = sources[row["from"]]
        assert row["policy"]["mission_boundary_km"] == source["mission_boundary_km"], row["script"]
        # T_max from the earlier epoch (organizer, 2026-11-04): the source's plus how far
        # the earlier epoch moved back, the head start less what the later one moved on.
        assert row["policy"]["mission_time_origin"] == "earliest"
        moved_s = row.get("late_moved_s", 0)
        assert row["horizon_s"] == pytest.approx(
            source["horizon_s"] + row["lead_s"] - moved_s, abs=1e-9)


def test_without_the_announced_horizon_the_parse_stops() -> None:
    script = RANKING / _table()[0]["script"]
    with pytest.raises(ParseError, match=r"set policy\.horizon_s"):
        parse_gmat_script(script, load_config(CONFIG))


@pytest.mark.parametrize("row", _table(), ids=lambda row: row["script"])
def test_scenario_parses_under_the_ranking_config(row: dict[str, Any]) -> None:
    for key in ("mission_boundary_km", "horizon_s", "perigee_time_s", "perigee_altitude_km"):
        assert isinstance(row[key], int | float) and math.isfinite(row[key]), key
    horizon_s = float(row["horizon_s"])
    scenario, _ = parse_gmat_script(RANKING / row["script"], _with_horizon(horizon_s))

    assert (scenario.pursuer.name, scenario.target.name) == ("SC_Human", "SC_Alien")
    assert scenario.rules == "ranking"
    assert scenario.max_time_s == horizon_s
    assert scenario.target_period_s is None
    assert scenario.burn_slots == ()
    (segment,) = scenario.propagate_segments
    # The zero-burn run that measured T_max is the script as written.
    assert segment.duration_s == pytest.approx(horizon_s, abs=1e-9)

    def prefixed(name: str, fields: dict[str, str]) -> dict[str, str]:
        return {f"{name}.{key}": value for key, value in fields.items()}

    models = prefixed("DefaultProp_ForceModel", _FULL_MODEL)
    propagators = prefixed("DefaultProp", {"FM": "DefaultProp_ForceModel", **_PROPAGATOR})
    if row["script"] == _TWO_PROPAGATORS:
        assert segment.propagator_names == ("DefaultProp", "AlienProp")
        assert segment.spacecraft_names == ("SC_Human", "SC_Alien")
        assert segment.epoch_lock == "synchronized"
        models |= prefixed("AlienProp_ForceModel", _FULL_MODEL)
        propagators |= prefixed("AlienProp", {"FM": "AlienProp_ForceModel", **_PROPAGATOR})
    else:
        assert segment.propagator_names == ("DefaultProp",)
        assert segment.epoch_lock == "propagator"
    assert scenario.force_model_fields == models
    assert scenario.propagator_fields == propagators
    for craft in (scenario.pursuer, scenario.target):
        assert set(_ELEMENTS) <= set(craft.fields)
        rest = {key: value for key, value in craft.fields.items() if key not in _ELEMENTS}
        assert rest == {**_CRAFT, **_NAIF[craft.name]}, craft.name

    mu = scenario.mu_km3_s2
    target = cartesian_to_keplerian(scenario.target.initial_state, mu)
    assert target.ecc > 1.0 and target.sma_km < 0.0
    assert math.sqrt(-mu / target.sma_km) == pytest.approx(row["v_inf_km_s"], abs=1e-6)
    radius_km = load_config(CONFIG).policy.central_body_radius_km
    osculating_perigee_km = target.sma_km * (1.0 - target.ecc)
    # GMAT's perigee differs from the osculating one by what J2 does on the way in.
    assert row["perigee_altitude_km"] == pytest.approx(osculating_perigee_km - radius_km, abs=5.0)
    assert osculating_perigee_km < row["mission_boundary_km"]
    assert 0.0 < row["perigee_time_s"] < horizon_s

    normals = [
        np.cross(craft.initial_state.r, craft.initial_state.v)
        for craft in (scenario.pursuer, scenario.target)
    ]
    cosine = float(np.dot(*normals) / np.linalg.norm(normals[0]) / np.linalg.norm(normals[1]))
    assert math.degrees(math.acos(cosine)) == pytest.approx(row["plane_angle_deg"], abs=1e-3)


_STATES = {
    "Cartesian": ("X", "Y", "Z", "VX", "VY", "VZ"),
    "Keplerian": ("SMA", "ECC", "INC", "RAAN", "AOP", "TA"),
    "SphericalAZFPA": ("RMAG", "RA", "DEC", "VMAG", "AZI", "FPA"),
}
_PLACED = ("Epoch", "DisplayStateType", "CoordinateSystem", *sum(_STATES.values(), ()))


def _placement(text: str, craft: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        head, _, value = line.partition(" = ")
        name, _, field = head.partition(".")
        if name == craft and field in _PLACED:
            assert field not in fields, f"{craft}.{field} twice"
            fields[field] = value
    return fields


def _kept_lines(text: str, moved: set[str]) -> list[str]:
    kept = []
    for line in text.splitlines():
        head = line.split(" = ")[0]
        craft, _, field = head.partition(".")
        if line.startswith("%") or (craft in moved and field in _PLACED):
            continue
        kept.append(line)
    return kept


@pytest.mark.parametrize(
    "row",
    yaml.safe_load((RANKING / "async_scenarios.yaml").read_text(encoding="utf-8"))["scenarios"],
    ids=lambda row: row["script"].split("_")[0],
)
def test_an_asynchronous_rehearsal_changes_only_what_places_a_craft(row: dict[str, Any]) -> None:
    source = (RANKING / row["from"]).read_text(encoding="utf-8")
    built = (RANKING / row["script"]).read_text(encoding="utf-8")
    early, late = row["early"], ({"SC_Alien", "SC_Human"} - {row["early"]}).pop()
    moved_s = row.get("late_moved_s", 0)
    # R6 also inserts SC_Human from an EarthFixed state at the later epoch; R9 moves the late one on.
    moved = {row["early"]} | ({"SC_Human"} if row["script"].startswith("R6_") else set()) | (
        {late} if moved_s else set())
    alignment = f"Propagate DefaultProp({early}) {{{early}.UTCModJulian = {late}.UTCModJulian}};"
    expected = []
    for line in _kept_lines(source, moved):
        if line.startswith("ReportFile1.Add = {"):
            line = line.replace("{", "{SC_Human.Earth.Altitude, ", 1)
        if moved_s and line.startswith("Propagate DefaultProp(SC_Alien, SC_Human)"):
            goal = row["horizon_s"] - row["lead_s"]
            line = re.sub(r"ElapsedSecs = [0-9.]+", f"ElapsedSecs = {goal:.3f}", line)
        expected.append(line)
        if line == "BeginMissionSequence;" and row["policy"].get("add_epoch_alignment") is not True:
            expected.append(alignment)
    assert _kept_lines(built, moved) == expected
    # Each moved craft carries one state of one type, and only the early one moves in time.
    for craft in moved:
        placed = _placement(built, craft)
        state = placed["DisplayStateType"].rstrip(";")
        assert set(placed) - {"Epoch", "CoordinateSystem"} == {"DisplayStateType", *_STATES[state]}
        if craft == early:
            assert placed["Epoch"] == "'09 Nov 2026 01:30:00.000';"
        elif moved_s:
            assert placed["Epoch"] == "'09 Nov 2026 02:30:00.000';"
            assert (placed["CoordinateSystem"], state) == ("EarthMJ2000Eq;", "Cartesian")
        else:
            assert placed["Epoch"] == _placement(source, craft)["Epoch"]
            assert (placed["CoordinateSystem"], state) == ("EarthFixed;", "SphericalAZFPA")


def test_only_the_ranking_profiles_add_an_epoch_alignment() -> None:
    # Signed off 2026-11-03 in COMPETITION_POLICY.md for the ranking profiles only.
    signed = {"ranking.yaml", "ranking-onsite.yaml", "ranking-rehearsal.yaml"}
    assert PolicyConfig().add_epoch_alignment is False
    seen = set()
    for path in sorted((ROOT / "config").rglob("*.yaml")):
        policy = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("policy") or {}
        name = path.relative_to(ROOT / "config").as_posix()
        assert policy.get("add_epoch_alignment", False) is (name in signed), name
        seen.add(name)
    assert signed <= seen
