from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from orbitwin.ephemeris import ephemeris_from_report, ephemeris_from_rows
from orbitwin.errors import DynamicsError, GmatRunError
from orbitwin.gmat_runner import EXIT_STATUS_ERROR
from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, propagate_universal

MU = 398600.4418
HEADER = "P.ElapsedSecs T.ElapsedSecs " + " ".join(
    f"{craft}.EarthMJ2000Eq.{axis}" for craft in "PT" for axis in ("X", "Y", "Z", "VX", "VY", "VZ")
)
HYPERBOLA = KeplerianElements(7000.0 / (1.0 - 1.5), 1.5, 50.0, 75.0, 0.0, -30.0)


def _two_body_report(path: Path, times: list[float], target_lag_s: float = 0.0) -> Path:
    start = keplerian_to_cartesian(HYPERBOLA, MU)
    rows = []
    for t in times:
        s = propagate_universal(start, t, MU)
        values = [t, t - target_lag_s, *s.r, *s.v, *s.r, *s.v]
        rows.append(" ".join(f"{float(v):.17g}" for v in values))
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "step_s, bound_km, bound_km_s", [(10.0, 1.0e-5, 1.0e-8), (60.0, 5.0e-3, 1.0e-5)]
)
def test_interpolation_tracks_a_hyperbola_through_perigee(
    tmp_path: Path, step_s: float, bound_km: float, bound_km_s: float
) -> None:
    # Measured: 1.4 mm and 1 um/s at 10 s rows, 1.75 m and 1.6 mm/s at 60 s.
    times = list(np.arange(0.0, 2700.0 + step_s, step_s))
    ephemeris = ephemeris_from_report(_two_body_report(tmp_path / "r.txt", times), "P", "T")
    start = keplerian_to_cartesian(HYPERBOLA, MU)
    for t in np.array(times[:-1]) + step_s / 2.0:
        want = propagate_universal(start, float(t), MU)
        got = ephemeris.state_at(float(t))
        assert np.linalg.norm(np.array(got.r) - want.r) < bound_km
        assert np.linalg.norm(np.array(got.v) - want.v) < bound_km_s


def test_an_epoch_written_twice_is_kept_once(tmp_path: Path) -> None:
    # GMAT writes the boundary between two Propagate segments on both sides.
    times = [0.0, 60.0, 120.0, 120.0, 180.0]
    ephemeris = ephemeris_from_report(_two_body_report(tmp_path / "r.txt", times), "P", "T")
    assert list(ephemeris.times_s) == [0.0, 60.0, 120.0, 180.0]


def test_rows_out_of_order_are_refused(tmp_path: Path) -> None:
    path = _two_body_report(tmp_path / "r.txt", [0.0, 120.0, 60.0])
    with pytest.raises(GmatRunError, match="increasing epochs"):
        ephemeris_from_report(path, "P", "T")


def test_rows_with_the_craft_at_different_epochs_are_refused(tmp_path: Path) -> None:
    path = _two_body_report(tmp_path / "r.txt", [0.0, 60.0], target_lag_s=2.0)
    with pytest.raises(GmatRunError, match="apart in epoch"):
        ephemeris_from_report(path, "P", "T")


@pytest.mark.parametrize("rows", [0, 1])
def test_too_few_rows_are_refused_as_a_run_error(rows: int) -> None:
    """Not an IndexError, which the corrector would not catch as a failed run."""
    with pytest.raises(GmatRunError, match="increasing epochs"):
        ephemeris_from_rows(np.zeros(rows), np.zeros((rows, 3)), np.zeros((rows, 3)), "r")


def test_a_time_outside_the_ephemeris_is_refused(tmp_path: Path) -> None:
    ephemeris = ephemeris_from_report(
        _two_body_report(tmp_path / "r.txt", [0.0, 60.0, 120.0]), "P", "T"
    )
    ephemeris.state_at(120.0)
    with pytest.raises(DynamicsError, match="outside the ephemeris"):
        ephemeris.state_at(121.0)


@pytest.mark.gmat
def test_a_real_coast_run_matches_two_body_on_point_mass_gravity(
    tmp_path: Path, workshop_script: Path, gmat: tuple[str, str]
) -> None:
    """The workshop template is point mass, so GMAT and two-body must agree."""
    import dataclasses

    from orbitwin.config import OrbitWinConfig
    from orbitwin.ephemeris import target_ephemeris_from_gmat
    from orbitwin.gmat_parser import parse_gmat_script

    console, startup = gmat
    base = OrbitWinConfig()
    config = dataclasses.replace(base, policy=dataclasses.replace(base.policy, horizon_s=3000.0))
    scenario, document = parse_gmat_script(workshop_script, config)
    ephemeris = target_ephemeris_from_gmat(
        scenario, document, config, console, startup, tmp_path / "coast"
    )
    assert ephemeris.times_s[-1] == pytest.approx(3000.0)
    for t in (0.0, 777.7, 1500.0, 2999.5):
        want = propagate_universal(scenario.target.initial_state, t, scenario.mu_km3_s2)
        got = ephemeris.state_at(t)
        assert np.linalg.norm(np.array(got.r) - want.r) < 1.0e-3


@pytest.mark.gmat
def test_the_coast_survives_a_pursuer_that_falls_and_matches_the_joint_one(
    tmp_path: Path, project_root: Path, gmat: tuple[str, str]
) -> None:
    """R1 with SC_Human from 6800 km apogee, 6300 km pericenter: unpowered it lands at
    1978.6 s, and a coast flying both craft failed (RK89 accuracy). The target alone
    flies; on R1 as written it equals the joint coast to the micrometre."""
    import dataclasses
    import re

    import orbitwin.ephemeris as ephemeris_module
    from orbitwin.config import load_config
    from orbitwin.ephemeris import target_ephemeris_from_gmat
    from orbitwin.gmat_parser import parse_gmat_script

    console, startup = gmat
    base = load_config(project_root / "config" / "ranking-rehearsal.yaml")
    config = dataclasses.replace(base, policy=dataclasses.replace(base.policy, horizon_s=6711.667))
    r1 = project_root / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
    text = r1.read_text(encoding="utf-8")
    for key, value in {"SMA": "6550", "ECC": repr(500 / 13100), "TA": "180"}.items():
        text = re.sub(rf"^SC_Human\.{key} = .*;$", f"SC_Human.{key} = {value};", text, flags=re.M)
    falling = tmp_path / "falling.script"
    falling.write_text(text, encoding="utf-8")
    scenario, document = parse_gmat_script(falling, config)
    ephemeris = target_ephemeris_from_gmat(
        scenario, document, config, console, startup, tmp_path / "falling"
    )
    assert ephemeris.times_s[-1] == pytest.approx(6711.667, abs=1.0e-6)

    scenario, document = parse_gmat_script(r1, config)
    alone = target_ephemeris_from_gmat(scenario, document, config, console, startup, tmp_path / "a")
    joint_mission = ephemeris_module._target_only_mission
    try:
        ephemeris_module._target_only_mission = lambda script, _: script  # type: ignore[assignment]
        reader = ephemeris_module.target_rows_from_report
        ephemeris_module.target_rows_from_report = (  # type: ignore[assignment]
            lambda path, target, alignment=None, **_: ephemeris_module.ephemeris_from_report(
                path, scenario.pursuer.name, target, alignment
            )
        )
        joint = target_ephemeris_from_gmat(
            scenario, document, config, console, startup, tmp_path / "j"
        )
    finally:
        ephemeris_module._target_only_mission = joint_mission  # type: ignore[assignment]
        ephemeris_module.target_rows_from_report = reader  # type: ignore[assignment]
    for t in (0.0, 1000.0, 4500.0, 6711.0):
        assert np.linalg.norm(alone.state_at(t).r - joint.state_at(t).r) < 1.0e-6


@pytest.mark.gmat
def test_measuring_a_start_that_falls_says_why_it_cannot(
    tmp_path: Path, project_root: Path, gmat: tuple[str, str]
) -> None:
    import re
    import subprocess
    import sys

    console, startup = gmat
    text = (project_root / "examples" / "ranking" / "R1_NearPlane_Reachable.script").read_text(
        encoding="utf-8"
    )
    for key, value in {"SMA": "6550", "ECC": repr(500 / 13100), "TA": "180"}.items():
        text = re.sub(rf"^SC_Human\.{key} = .*;$", f"SC_Human.{key} = {value};", text, flags=re.M)
    falling = tmp_path / "falling.script"
    falling.write_text(text, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "measure_ranking_scenario.py"),
         str(falling), "--boundary-km", "17000", "--gmat-console", console,
         "--startup-file", startup, "--output", str(tmp_path / "m"), "--provisional"],
        capture_output=True, text=True, timeout=600, check=False,
        env={**os.environ, "PYTHONPATH": str(project_root / "src")},
    )
    assert done.returncode == 2, done.stderr[-2000:]
    long = tmp_path / "m" / "long"
    assert done.stderr == (f"measure_ranking_scenario: {long / 'falling_long_diagnostic.script'}: "
                           f"{EXIT_STATUS_ERROR.format(1)}\n")
    # What R2026a printed for this start on 2026-11-23: the falling pursuer's step shrinks.
    assert ("RungeKutta89: Accuracy settings will be violated"
            in (long / "run" / "stdout.txt").read_text(encoding="utf-8"))
