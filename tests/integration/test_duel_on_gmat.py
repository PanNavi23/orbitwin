"""The duel rollout, mu and the propagator table against GmatConsole R2026a.

scripts/duel_gmat_check.py measures mu, scripts/duel_gmat_fit.py the table and
scripts/duel_gmat_boundary.py the rows at round ends.

Outputs go under ORBITWIN_DUEL_GMAT_OUT, or <repo>/output/duel/pytest, never
inside the GMAT install: every run goes through gmat_runner.run_gmat.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from orbitwin.duel.gmat_settings import SETTINGS
from orbitwin.duel.params import GMAT_MU_KM3_S2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import duel_gmat_boundary
import duel_gmat_check
import duel_gmat_fit

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.gmat


def _out() -> Path:
    base = Path(os.environ.get("ORBITWIN_DUEL_GMAT_OUT", ROOT / "output" / "duel" / "pytest"))
    return base / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def test_gmat_point_mass_mu_and_the_rollout_agree_with_the_coast(gmat: tuple[str, str]) -> None:
    console, startup = gmat
    out = _out()
    results = duel_gmat_check.check(console, startup, out)
    assert {result.form for result in results} == {"primary-degree0", "pointmass"}
    for result in results:
        assert result.fitted_mu == pytest.approx(GMAT_MU_KM3_S2, abs=1e-6)
        # A day of coasting agrees with GMAT to 1e-5 km at GMAT's mu, and the
        # repository's old analytical mu is visibly worse.
        assert result.max_error_km[str(GMAT_MU_KM3_S2)] < 1e-5
        assert result.max_error_km["398600.4418"] > 1e-4


def test_every_measured_setting_stays_inside_its_stored_row(gmat: tuple[str, str]) -> None:
    console, startup = gmat
    out = _out()
    table = duel_gmat_fit.measure(console, startup, out)
    assert set(table) == set(SETTINGS) - {"cautious"}
    for name, measured in table.items():
        row = SETTINGS[name]
        assert measured["c_measured"] <= row.row_gap_c
        assert measured["position_floor_km_measured"] <= row.position_floor_km
        assert measured["position_rate_km_s_measured"] <= row.position_rate_km_s
        assert measured["a_drift_km_measured"] + measured["a_method_km_measured"] <= row.a_km
        # GMAT's SMA is eq. (2) on its own state, to well under a metre.
        assert measured["a_method_km_measured"] < 1e-6


def test_every_round_boundary_has_a_report_row_in_both_propagate_forms(
    gmat: tuple[str, str]
) -> None:
    console, startup = gmat
    out = _out()
    results = duel_gmat_boundary.check(console, startup, out)
    assert {result.form for result in results} == {"joint", "synchronized"}
    for result in results:
        # The rounds end between integrator steps, and still each end has a row.
        assert result.interior_rows > 6 * duel_gmat_boundary.ROUNDS
        assert len(result.epoch_miss_s) == duel_gmat_boundary.ROUNDS
        assert max(result.epoch_miss_s) < 1e-5
        assert max(result.position_error_km) < 1e-4
        assert result.start_error_km < 1e-9
