"""The corrector on real GMAT R2026a and the ranking force model.

SC_Human on EGM96 20x20, tides, drag, Sun, Moon, SRP and relativity; SC_Alien
on gravity only and a hyperbola, each with its own propagator under one
Synchronized Propagate, as in docs/GMAT_IN_THE_LOOP.md.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from orbitwin.config import OrbitWinConfig
from orbitwin.ephemeris import target_ephemeris_from_gmat
from orbitwin.frames import script_components_to_inertial
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.gmat_shoot import correct_with_gmat
from orbitwin.lambert import solve_lambert_all
from orbitwin.models import StateVector, as_vector3
from orbitwin.multi_burn import as_candidate
from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, propagate_universal

pytestmark = pytest.mark.gmat

MU = 398600.4418


def _craft(name: str, spo: Path, state: list[str]) -> list[str]:
    return [
        f"Create Spacecraft {name};",
        f"{name}.DateFormat = UTCGregorian;",
        f"{name}.Epoch = '05 Dec 2026 00:00:00.000';",
        f"{name}.CoordinateSystem = EarthMJ2000Eq;",
        *state,
        f"{name}.DryMass = 1000;",
        f"{name}.SPADDragFile = '{spo}';",
        f"{name}.SPADDragScaleFactor = 1;",
        f"{name}.SPADSRPFile = '{spo}';",
        f"{name}.SPADSRPScaleFactor = 1;",
        "",
    ]


def _keplerian(name: str, values: tuple[float, ...]) -> list[str]:
    fields = ("SMA", "ECC", "INC", "RAAN", "AOP", "TA")
    return [f"{name}.DisplayStateType = Keplerian;"] + [
        f"{name}.{field} = {value!r};" for field, value in zip(fields, values, strict=True)
    ]


def _force_model(name: str, *, full: bool) -> list[str]:
    lines = [
        f"Create ForceModel {name};",
        f"{name}.CentralBody = Earth;",
        f"{name}.PrimaryBodies = {{Earth}};",
        f"{name}.PointMasses = {{Sun, Luna}};",
        f"{name}.GravityField.Earth.Degree = 20;",
        f"{name}.GravityField.Earth.Order = 20;",
        f"{name}.GravityField.Earth.PotentialFile = 'EGM96.cof';",
        f"{name}.GravityField.Earth.TideModel = 'SolidAndPole';",
    ]
    if full:
        lines += [
            f"{name}.Drag.AtmosphereModel = NRLMSISE00;",
            f"{name}.Drag.DragModel = SPADFile;",
            f"{name}.SRP = On;",
            f"{name}.SRP.SRPModel = SPADFile;",
        ]
    else:
        lines += [f"{name}.Drag = None;", f"{name}.SRP = Off;"]
    return [*lines, f"{name}.RelativisticCorrection = On;", f"{name}.ErrorControl = RSSStep;", ""]


def _propagator(name: str, force_model: str) -> list[str]:
    return [
        f"Create Propagator {name};",
        f"{name}.FM = {force_model};",
        f"{name}.Type = RungeKutta89;",
        f"{name}.InitialStepSize = 60;",
        f"{name}.Accuracy = 1e-11;",
        f"{name}.MinStep = 0.001;",
        f"{name}.MaxStep = 60;",
        "",
    ]


def _official(
    path: Path, console: Path, human: list[str], alien: list[str], horizon_s: float
) -> Path:
    spo = console.resolve().parents[1] / "data" / "vehicle" / "spad" / "SphericalModel.spo"
    lines = [
        *_craft("SC_Human", spo, human),
        *_craft("SC_Alien", spo, alien),
        *_force_model("FullFM", full=True),
        *_force_model("GravFM", full=False),
        *_propagator("PropHuman", "FullFM"),
        *_propagator("PropAlien", "GravFM"),
        "BeginMissionSequence;",
        "Propagate Synchronized PropHuman(SC_Human) PropAlien(SC_Alien) "
        f"{{SC_Human.ElapsedSecs = {horizon_s!r}}};",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def _config(horizon_s: float, **policy: object) -> OrbitWinConfig:
    base = OrbitWinConfig()
    settings: dict[str, object] = {"rules": "ranking", "horizon_s": horizon_s}
    settings.update(policy)
    return dataclasses.replace(
        base, policy=dataclasses.replace(base.policy, **settings),
        pursuer_name="SC_Human", target_name="SC_Alien",
    )


def test_the_probe_closes_in_two_runs(tmp_path: Path, gmat: tuple[str, str]) -> None:
    """docs/GMAT_IN_THE_LOOP.md: one impulse at 300 s, Lambert to GMAT's target at 2700 s.

    The probe wrote the impulse in EarthMJ2000Eq and measured 18.565 km, 9.5 m,
    6 mm. Written in VNB, as this project writes it, GMAT turns the frame with
    its own state and the first miss is 21.470 km; one correction left 18 m
    (closest approach 10.8 m) on 2026-11-01.
    """
    console, startup = map(Path, gmat)
    script = _official(
        tmp_path / "probe.script", console,
        _keplerian("SC_Human", (6800.0, 0.0, 40.0, 60.0, 0.0, 0.0)),
        _keplerian("SC_Alien", (-14000.0, 1.5, 50.0, 75.0, 0.0, -30.0)),
        3600.0,
    )
    # The impulse is 4.67 km/s; only the corrector is under test, not the caps.
    config = _config(3600.0, dv_limit_km_s=5.0, dv_safe_limit_km_s=4.9)
    scenario, document = parse_gmat_script(script, config)
    target = target_ephemeris_from_gmat(
        scenario, document, config, console, startup, tmp_path / "coast"
    )
    at_burn = propagate_universal(scenario.pursuer.initial_state, 300.0, MU)
    solutions = solve_lambert_all(
        at_burn.r, np.asarray(target.state_at(2700.0).r), 2400.0, MU, max_revolutions=0
    )
    impulse = min((np.asarray(s.v1) - at_burn.v for s in solutions), key=np.linalg.norm)
    candidate = as_candidate([(300.0, impulse)], 2700.0, scenario, config)

    result = correct_with_gmat(
        document, scenario, candidate, config, console, startup, tmp_path / "shoot",
        max_rounds=3,
    )
    assert result.converged, result.reason
    assert result.gmat_runs <= 3
    assert result.log[0].miss_at_arrival_km > 5.0
    assert result.closest_approach_km is not None and result.closest_approach_km < 0.5
    assert result.candidate is not None and result.candidate.successful
    change = result.candidate.burns[0].magnitude_km_s - candidate.burns[0].magnitude_km_s
    assert abs(change) < 0.01


def test_a_last_impulse_at_the_cap_shares_its_correction(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    """Ranking caps, the last impulse at 0.249 km/s, the target placed two-body.

    Measured on 2026-11-01: 36.600 km, then 0.170 km after the two impulses
    shared the correction, the last held at the cap less 1e-9 km/s.
    """
    console, startup = map(Path, gmat)
    human = (7000.0, 0.0, 40.0, 60.0, 0.0, 0.0)
    config = _config(
        3600.0, dv_limit_km_s=0.25, dv_safe_limit_km_s=0.249, total_dv_limit_km_s=3.0,
        total_dv_safe_limit_km_s=2.95,
    )
    # Two-body schedule, VNB on the flown state.
    state = keplerian_to_cartesian(KeplerianElements(*human), MU)
    now_s, schedule = 0.0, []
    for time_s, components in ((300.0, (0.15, 0.08, 0.0)), (1300.0, (0.0, 0.0, 0.249))):
        state = propagate_universal(state, time_s - now_s, MU)
        now_s = time_s
        impulse = script_components_to_inertial(
            np.asarray(components), state, "Local", "Earth", "VNB"
        )
        schedule.append((time_s, impulse))
        state = StateVector(state.r_km, as_vector3(state.v + impulse))
    arrival = propagate_universal(state, 2700.0 - now_s, MU)
    # SC_Alien at perigee of an e = 1.5 hyperbola there, in a plane 50 degrees off.
    radial = arrival.r / np.linalg.norm(arrival.r)
    plane = np.cross(arrival.r, arrival.v)
    plane /= np.linalg.norm(plane)
    normal = np.cos(np.radians(50.0)) * plane + np.sin(np.radians(50.0)) * np.cross(radial, plane)
    normal -= radial * float(normal @ radial)
    normal /= np.linalg.norm(normal)
    speed = np.sqrt(MU * 2.5 / np.linalg.norm(arrival.r))
    alien = propagate_universal(
        StateVector(arrival.r_km, as_vector3(speed * np.cross(normal, radial))), -2700.0, MU
    )
    cartesian = ["SC_Alien.DisplayStateType = Cartesian;"] + [
        f"SC_Alien.{name} = {float(value)!r};"
        for name, value in zip(("X", "Y", "Z", "VX", "VY", "VZ"), (*alien.r, *alien.v),
                               strict=True)
    ]
    script = _official(
        tmp_path / "two.script", console, _keplerian("SC_Human", human), cartesian, 3600.0
    )
    scenario, document = parse_gmat_script(script, config)
    candidate = as_candidate(schedule, 2700.0, scenario, config)

    result = correct_with_gmat(
        document, scenario, candidate, config, console, startup, tmp_path / "shoot",
        max_rounds=3,
    )
    assert result.converged, result.reason
    assert result.log[0].mode == "last-two"
    assert result.log[0].miss_at_arrival_km > 5.0
    assert result.gmat_runs <= 3
    assert result.candidate is not None
    for burn in result.candidate.burns:
        assert burn.magnitude_km_s <= 0.249
    assert result.candidate.metadata["minimum_pursuer_radius_km"] > 6378.1363
