"""Each ranking rehearsal script runs in stock GmatConsole and ends at its T_max."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from orbitwin.config import load_config
from orbitwin.diagnostic import add_diagnostic_report
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.gmat_runner import run_gmat
from orbitwin.report_parser import evaluate_diagnostic_report, parse_numeric_report

ROOT = Path(__file__).resolve().parents[2]
RANKING = ROOT / "examples" / "ranking"
_ROWS = yaml.safe_load((RANKING / "scenarios.yaml").read_text(encoding="utf-8"))["scenarios"]

#: R2026a writes these reports byte for byte the same on every run; 10 m is slack
#: for a data-file update, under the 21 m that dropping Sun and Luna moves R1's
#: zero-burn closest approach. Smaller model changes are the unit test's to catch.
TOLERANCE_KM = 0.01


def _positions(report: Any, name: str) -> np.ndarray:
    return np.column_stack([report.column(f"{name}.EarthMJ2000Eq.{axis}") for axis in "XYZ"])


@pytest.mark.gmat
@pytest.mark.parametrize("row", _ROWS, ids=lambda row: row["script"])
def test_scenario_runs_to_its_horizon_in_gmatconsole(
    row: dict[str, Any], tmp_path: Path, gmat: tuple[str, str]
) -> None:
    console, startup = gmat
    config = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    config = dataclasses.replace(config, policy=dataclasses.replace(config.policy, team_number=54))
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, horizon_s=float(row["horizon_s"]))
    )
    scenario, document = parse_gmat_script(RANKING / row["script"], config)
    diagnostic = add_diagnostic_report(document, scenario, config.gmat.diagnostic_report_name)
    sibling = tmp_path / row["script"]
    sibling.write_bytes(diagnostic.script_bytes)
    official = [report.filename for report in scenario.report_files]
    result = run_gmat(
        console,
        sibling,
        tmp_path / "run",
        [*official, diagnostic.report_filename],
        startup_file=startup,
        timeout_s=config.gmat.timeout_s,
    )
    assert result.success, result.error

    # Also checks that both craft stayed on one epoch on every row.
    event = evaluate_diagnostic_report(
        result.reports[diagnostic.report_filename],
        scenario.pursuer.name,
        scenario.target.name,
        config.policy.intercept_radius_km,
        central_body_radius_km=config.policy.central_body_radius_km,
    )
    assert event.first_entry_time_s is None, "the zero-burn run should not intercept"
    assert event.minimum_distance_km == pytest.approx(
        row["zero_burn_closest_km"], abs=TOLERANCE_KM
    )
    assert event.minimum_pursuer_radius_km is not None
    assert event.minimum_pursuer_radius_km > config.policy.central_body_radius_km

    report = parse_numeric_report(result.reports[diagnostic.report_filename])
    times = report.column("SC_Alien.ElapsedSecs")
    alien = np.linalg.norm(_positions(report, "SC_Alien"), axis=1)
    assert times[-1] == pytest.approx(row["horizon_s"], abs=1e-6)
    np.testing.assert_allclose(
        _positions(report, "SC_Human")[-1], row["human_final_position_km"], rtol=0, atol=TOLERANCE_KM
    )
    lowest = int(np.argmin(alien))
    assert 0 < lowest < len(alien) - 1, "perigee should fall inside the run"
    radius_km = config.policy.central_body_radius_km
    # The lowest 1 s row is within 0.5 s of perigee, at most about 1 m above it.
    assert alien[lowest] - radius_km == pytest.approx(row["perigee_altitude_km"], abs=TOLERANCE_KM)
    assert times[lowest] == pytest.approx(row["perigee_time_s"], abs=0.5)
    # T_max: SC_Alien is back at the boundary on the last row, and not before it.
    assert alien[-1] == pytest.approx(row["mission_boundary_km"], abs=TOLERANCE_KM)
    assert np.all(alien[lowest:-1] < row["mission_boundary_km"])

    official_rows = parse_numeric_report(result.reports[official[0]])
    assert official_rows.column("SC_Human.ElapsedSecs")[-1] == pytest.approx(
        row["horizon_s"], abs=1e-6
    )
