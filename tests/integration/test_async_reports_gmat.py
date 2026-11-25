"""Verify and certify on stock GmatConsole, craft 5400 s apart with an alignment."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, PolicyConfig
from orbitwin.ephemeris import target_ephemeris_from_gmat
from orbitwin.report_parser import parse_numeric_report, report_clock
from orbitwin.search import zero_burn_candidate
from orbitwin.state_probe import load_scenario
from orbitwin.verify import certify_candidate, verify_candidate_with_gmat

ROOT = Path(__file__).resolve().parents[2]
CROSSING = ROOT / "tests" / "fixtures" / "async" / "aligned_crossing.script"

pytestmark = pytest.mark.gmat


def _config() -> OrbitWinConfig:
    return OrbitWinConfig(
        policy=dataclasses.replace(
            PolicyConfig(), mission_time_origin="pursuer", closest_approach_window="after_alignment"
        ),
        pursuer_name="SC_Human",
        target_name="SC_Alien",
    )


def test_an_aligned_crossing_verifies_and_certifies_on_the_solver_clock(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    """Measured on R2026a: entry at 1109.25 s after SC_Human's epoch, 540 rows before it."""
    console, startup = gmat
    config = _config()
    scenario, document, _ = load_scenario(CROSSING, config, gmat_console=console, startup_file=startup)
    assert scenario.alignment is not None
    assert scenario.alignment.offset_s == pytest.approx(5400.0, abs=1.0e-6)
    verified, _, evidence = verify_candidate_with_gmat(
        CROSSING, scenario, document, zero_burn_candidate(scenario, config), config, console,
        startup, tmp_path / "verify",
    )
    entry_s = verified.metadata["gmat_event"]["first_entry_time_s"]
    # The tracks cross 1500 s in at 13.2 m/s: inside 5 km from about 1120 s.
    assert verified.successful and 1050.0 < entry_s < 1150.0
    clock = report_clock(
        parse_numeric_report(evidence["diagnostic_report"]), "SC_Human", "SC_Alien",
        scenario.alignment,
    )
    # MaxStep 10 s over 5400 s, and the one row at t = 0 is not among them.
    assert clock.frozen_rows == 540 and clock.times_s[0] == pytest.approx(-5400.0, abs=1e-6)

    certificate = certify_candidate(
        CROSSING, zero_burn_candidate(scenario, config), config, console, startup,
        tmp_path / "certify",
    )
    assert certificate["frozen_rows"] == 540
    assert certificate["final_script_parity"]["first_entry_time_s"] == pytest.approx(entry_s)
    official = next((tmp_path / "certify" / "submission").glob("OfficialReport.txt"))
    check = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_report_rows.py"), str(official),
         "--frozen-rows", "540"],
        capture_output=True, text=True,
    )
    assert check.returncode == 0, check.stdout
    assert "LAST row is inside the sphere" in check.stdout


def test_the_coast_run_puts_an_early_target_on_the_solver_clock(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    console, startup = gmat
    config = _config()
    scenario, document, _ = load_scenario(CROSSING, config, gmat_console=console, startup_file=startup)
    ephemeris = target_ephemeris_from_gmat(
        scenario, document, config, console, startup, tmp_path / "coast"
    )
    assert ephemeris.times_s[0] == pytest.approx(0.0, abs=1.0e-6)
    assert ephemeris.times_s[-1] == pytest.approx(scenario.max_time_s, abs=1.0e-6)
    # SC_Alien 1500 s before the crossing: 12.2 km from SC_Human's start, two-body.
    import numpy as np

    gap = np.linalg.norm(ephemeris.positions_km[0] - scenario.pursuer.initial_state.r)
    assert gap == pytest.approx(12.2037, abs=1.0e-3)


def test_the_geodetic_altitude_is_recorded_beside_the_sphere(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    """At 51.6 deg the geodetic figure reads higher than r - R on every row."""
    console, startup = gmat
    config = _config()
    certificate = certify_candidate(
        CROSSING, zero_burn_candidate(load_scenario(
            CROSSING, config, gmat_console=console, startup_file=startup
        )[0], config), config, console, startup, tmp_path / "certify",
    )
    geodetic = certificate["flown_minimum_geodetic_altitude_km"]
    sphere = certificate["flown_minimum_altitude_km"]
    assert geodetic is not None and sphere is not None
    assert sphere < geodetic < sphere + 25.0
    gmat_event = certificate["candidate"]["metadata"]["gmat_event"]
    assert gmat_event["minimum_geodetic_altitude_km"] >= geodetic - 1.0e-9
