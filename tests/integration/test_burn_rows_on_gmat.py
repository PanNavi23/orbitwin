"""The report step after a burn on real GMAT R2026a, read with and without the rebuilt row (#106).

Ranking force model on SC_Human, two burns 300 and 1300 s, InitialStepSize and
MaxStep 60 s, so the first row after each burn is 60 s on. The
truth is a second flight of the same script with the Propagate after each
Maneuver split at 20 s, where the error peaks, so GMAT writes a row there.
Measured 2026-11-01: 1.511 and 2.214 km off as read before, 0.28 and 0.25 m
with the rebuilt row.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

from orbitwin.burn_rows import maneuvers_flown, with_post_burn_rows
from orbitwin.config import OrbitWinConfig
from orbitwin.diagnostic import add_diagnostic_report
from orbitwin.ephemeris import Ephemeris
from orbitwin.frames import script_components_to_inertial
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.gmat_runner import run_gmat
from orbitwin.models import StateVector, as_vector3
from orbitwin.multi_burn import as_candidate
from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, propagate_universal
from orbitwin.report_parser import common_epochs, evaluate_diagnostic_report, parse_numeric_report
from orbitwin.script_renderer import write_rendered_candidate

pytestmark = pytest.mark.gmat

MU = 398600.4418
HUMAN = (7000.0, 0.0, 40.0, 60.0, 0.0, 0.0)
COMPONENTS = ((0.15, 0.08, 0.0), (0.0, 0.0, 0.249))


def _script(path: Path, console: Path, alien: StateVector) -> Path:
    spo = console.resolve().parents[1] / "data" / "vehicle" / "spad" / "SphericalModel.spo"

    def craft(name: str, state: list[str]) -> list[str]:
        return [
            f"Create Spacecraft {name};", f"{name}.DateFormat = UTCGregorian;",
            f"{name}.Epoch = '05 Dec 2026 00:00:00.000';",
            f"{name}.CoordinateSystem = EarthMJ2000Eq;", *state, f"{name}.DryMass = 1000;",
            f"{name}.SPADDragFile = '{spo}';", f"{name}.SPADDragScaleFactor = 1;",
            f"{name}.SPADSRPFile = '{spo}';", f"{name}.SPADSRPScaleFactor = 1;", "",
        ]

    def force_model(name: str, full: bool) -> list[str]:
        lines = [
            f"Create ForceModel {name};", f"{name}.CentralBody = Earth;",
            f"{name}.PrimaryBodies = {{Earth}};", f"{name}.PointMasses = {{Sun, Luna}};",
            f"{name}.GravityField.Earth.Degree = 20;", f"{name}.GravityField.Earth.Order = 20;",
            f"{name}.GravityField.Earth.PotentialFile = 'EGM96.cof';",
            f"{name}.GravityField.Earth.TideModel = 'SolidAndPole';",
        ]
        lines += (
            [f"{name}.Drag.AtmosphereModel = NRLMSISE00;", f"{name}.Drag.DragModel = SPADFile;",
             f"{name}.SRP = On;", f"{name}.SRP.SRPModel = SPADFile;"]
            if full else [f"{name}.Drag = None;", f"{name}.SRP = Off;"]
        )
        return [*lines, f"{name}.RelativisticCorrection = On;", f"{name}.ErrorControl = RSSStep;", ""]

    def propagator(name: str, force: str) -> list[str]:
        return [
            f"Create Propagator {name};", f"{name}.FM = {force};", f"{name}.Type = RungeKutta89;",
            f"{name}.InitialStepSize = 60;", f"{name}.Accuracy = 1e-11;",
            f"{name}.MinStep = 0.001;", f"{name}.MaxStep = 60;", "",
        ]

    fields = ("SMA", "ECC", "INC", "RAAN", "AOP", "TA")
    lines = [
        *craft("SC_Human", ["SC_Human.DisplayStateType = Keplerian;"] + [
            f"SC_Human.{field} = {value!r};" for field, value in zip(fields, HUMAN, strict=True)
        ]),
        *craft("SC_Alien", ["SC_Alien.DisplayStateType = Cartesian;"] + [
            f"SC_Alien.{name} = {float(value)!r};"
            for name, value in zip(("X", "Y", "Z", "VX", "VY", "VZ"), (*alien.r, *alien.v),
                                   strict=True)
        ]),
        *force_model("FullFM", True), *force_model("GravFM", False),
        *propagator("PropHuman", "FullFM"), *propagator("PropAlien", "GravFM"),
        "BeginMissionSequence;",
        "Propagate Synchronized PropHuman(SC_Human) PropAlien(SC_Alien) "
        "{SC_Human.ElapsedSecs = 3600.0};", "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def _rows(path: str) -> tuple[np.ndarray, ...]:
    report = parse_numeric_report(path)
    times = common_epochs(report, "SC_Human", "SC_Alien")
    columns = []
    for craft in ("SC_Human", "SC_Alien"):
        for kind in ("", "V"):
            columns.append(np.column_stack(
                [report.column(f"{craft}.EarthMJ2000Eq.{kind}{axis}") for axis in "XYZ"]
            ))
    return (times, *columns)


def _pursuer_at(rows: tuple[np.ndarray, ...], time_s: float) -> np.ndarray:
    times, position, velocity = rows[0], rows[1], rows[2]
    keep = np.append(np.diff(times) > 1.0e-9, True)  # the later of two rows at one epoch
    return np.asarray(Ephemeris(times[keep], position[keep], velocity[keep]).state_at(time_s).r)


@pytest.mark.parametrize(
    ("epochs", "rows_at_burn"),
    [((300.0, 1300.0), 1), ((300.000004, 1320.000007), 2)],
    ids=["on-a-step", "just-past-a-step"],
)
def test_the_step_after_each_burn_is_read_with_the_post_burn_state(
    tmp_path: Path, epochs: tuple[float, float], rows_at_burn: int, gmat: tuple[str, str]
) -> None:
    """Just past a step, R2026a writes the step's row and the goal's, both pre-burn.

    Taking the second for a post-burn row left that case 1511.12 m off 20 s after
    the first burn (1282.96 m after the second); rebuilt, 0.28 and 0.07 m.
    """
    console, startup = gmat
    base = OrbitWinConfig()
    config = dataclasses.replace(
        base,
        policy=dataclasses.replace(
            base.policy, rules="ranking", horizon_s=3600.0, dv_limit_km_s=0.25,
            dv_safe_limit_km_s=0.249, total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.95,
        ),
        pursuer_name="SC_Human", target_name="SC_Alien",
    )
    state, now_s, schedule = keplerian_to_cartesian(KeplerianElements(*HUMAN), MU), 0.0, []
    for time_s, components in zip(epochs, COMPONENTS, strict=True):
        state = propagate_universal(state, time_s - now_s, MU)
        now_s = time_s
        impulse = script_components_to_inertial(np.asarray(components), state, "Local", "Earth", "VNB")
        schedule.append((time_s, impulse))
        state = StateVector(state.r_km, as_vector3(state.v + impulse))
    arrival = propagate_universal(state, 2700.0 - now_s, MU)
    radial = arrival.r / np.linalg.norm(arrival.r)
    normal = np.cross(arrival.r, arrival.v)
    normal /= np.linalg.norm(normal)
    normal = np.cos(np.radians(50.0)) * normal + np.sin(np.radians(50.0)) * np.cross(radial, normal)
    normal -= radial * float(normal @ radial)
    normal /= np.linalg.norm(normal)
    speed = np.sqrt(MU * 2.5 / np.linalg.norm(arrival.r))
    alien = propagate_universal(
        StateVector(arrival.r_km, as_vector3(speed * np.cross(normal, radial))), -2700.0, MU
    )
    official = _script(tmp_path / "official.script", Path(console), alien)
    scenario, document = parse_gmat_script(official, config)
    rendered_path = tmp_path / "rendered.script"
    write_rendered_candidate(
        rendered_path, tmp_path / "rendered.diff.txt", document, scenario,
        as_candidate(schedule, 2700.0, scenario, config), config,
    )
    flown_scenario, flown_document = parse_gmat_script(rendered_path, config)
    maneuvers = maneuvers_flown(flown_scenario)
    assert [maneuver.time_s for maneuver in maneuvers] == pytest.approx(list(epochs))
    sibling = add_diagnostic_report(flown_document, flown_scenario, "diagnostic.txt")
    text = sibling.script_bytes.decode("ascii")

    def split(match: re.Match[str]) -> str:
        line = match.group(2)
        goal = float(re.search(r"ElapsedSecs = ([0-9.eE+-]+)", line).group(1))
        first = re.sub(r"ElapsedSecs = [0-9.eE+-]+", "ElapsedSecs = 20.0", line)
        rest = re.sub(r"ElapsedSecs = [0-9.eE+-]+", f"ElapsedSecs = {goal - 20.0!r}", line)
        return f"{match.group(1)}{first}\n{rest}"

    truth_text = re.sub(r"(Maneuver \w+\(SC_Human\);\n)(Propagate [^\n]*)", split, text)
    assert truth_text.count("ElapsedSecs = 20.0") == 2
    runs = {}
    for name, body in (("plain", text), ("truth", truth_text)):
        path = tmp_path / f"{name}.script"
        path.write_text(body, encoding="ascii")
        run = run_gmat(console, path, tmp_path / f"run-{name}", ["diagnostic.txt"],
                       startup_file=startup, timeout_s=600.0)
        assert run.success, run.error
        runs[name] = run.reports["diagnostic.txt"]

    plain, truth = _rows(runs["plain"]), _rows(runs["truth"])
    for maneuver in maneuvers:
        at = np.abs(plain[0] - maneuver.time_s) <= 1.0e-5
        assert np.count_nonzero(at) == rows_at_burn
    rebuilt = with_post_burn_rows(*plain, maneuvers, runs["plain"])
    for maneuver in maneuvers:
        epoch = maneuver.time_s + 20.0
        row = int(np.argmin(np.abs(truth[0] - epoch)))
        assert abs(truth[0][row] - epoch) < 1.0e-5
        before = float(np.linalg.norm(_pursuer_at(plain, epoch) - truth[1][row]))
        after = float(np.linalg.norm(_pursuer_at(rebuilt, epoch) - truth[1][row]))
        assert before > 1.0, (maneuver.time_s, before)
        assert after < 0.01, (maneuver.time_s, after)
    event = evaluate_diagnostic_report(
        runs["plain"], "SC_Human", "SC_Alien", 5.0, maneuvers=maneuvers
    )
    assert event.first_entry_time_s is None and 30.0 < event.minimum_distance_km < 40.0
