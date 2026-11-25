from __future__ import annotations

from pathlib import Path

import pytest

from orbitwin.errors import GmatRunError
from orbitwin.report_parser import evaluate_diagnostic_report

HEADER = (
    "P.ElapsedSecs T.ElapsedSecs "
    "P.EarthMJ2000Eq.X P.EarthMJ2000Eq.Y P.EarthMJ2000Eq.Z "
    "P.EarthMJ2000Eq.VX P.EarthMJ2000Eq.VY P.EarthMJ2000Eq.VZ "
    "T.EarthMJ2000Eq.X T.EarthMJ2000Eq.Y T.EarthMJ2000Eq.Z "
    "T.EarthMJ2000Eq.VX T.EarthMJ2000Eq.VY T.EarthMJ2000Eq.VZ"
)


def _report(path: Path, target_lag_s: float, header: str = HEADER) -> Path:
    rows = [
        f"{t} {t - target_lag_s * k} {7010.0 - t} 0 0 -1 0 0 7000 50 0 0 0 0"
        for k, t in enumerate((0.0, 10.0, 20.0))
    ]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_rows_on_one_epoch_are_read(tmp_path: Path) -> None:
    event = evaluate_diagnostic_report(_report(tmp_path / "r.txt", 2.0e-7), "P", "T", 5.0)
    assert event.minimum_distance_km == pytest.approx(50.0)


def test_rows_two_seconds_apart_are_refused(tmp_path: Path) -> None:
    with pytest.raises(GmatRunError, match=r"4 s apart in epoch at report row 3"):
        evaluate_diagnostic_report(_report(tmp_path / "r.txt", 2.0), "P", "T", 5.0)


def test_a_report_without_the_targets_epoch_is_refused(tmp_path: Path) -> None:
    header = HEADER.replace("T.ElapsedSecs", "T.Unused")
    with pytest.raises(GmatRunError, match=r"T\.ElapsedSecs"):
        evaluate_diagnostic_report(_report(tmp_path / "r.txt", 0.0, header), "P", "T", 5.0)


def test_the_diagnostic_sibling_records_both_epochs(workshop_script: Path) -> None:
    from orbitwin.config import OrbitWinConfig
    from orbitwin.diagnostic import add_diagnostic_report
    from orbitwin.gmat_parser import parse_gmat_script

    config = OrbitWinConfig()
    scenario, document = parse_gmat_script(workshop_script, config)
    sibling = add_diagnostic_report(document, scenario, config.gmat.diagnostic_report_name)
    added = next(
        line for line in sibling.script_bytes.decode("utf-8").splitlines() if ".Add = {" in line
        and "OrbitWinDiagnostic" in line
    )
    assert "HumanSC.ElapsedSecs" in added and "AlienSC.ElapsedSecs" in added


def test_the_report_step_is_the_widest_gap_after_t_0(tmp_path: Path) -> None:
    path = tmp_path / "r.txt"
    rows = [f"{t} {t} {7010.0 - t} 0 0 -1 0 0 7000 50 0 0 0 0" for t in (0.0, 10.0, 35.0, 40.0)]
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    assert evaluate_diagnostic_report(path, "P", "T", 5.0).report_step_s == 25.0
