"""Reports on absolute epochs: the A1 check, the solver clock and the alignment rows.

See docs/ASYNC_START_AND_INSERTION.md §4 and §5.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import FAKE_STARTUP

from orbitwin.config import OrbitWinConfig, PolicyConfig
from orbitwin.diagnostic import add_diagnostic_report
from orbitwin.ephemeris import ephemeris_from_report, target_rows_from_report
from orbitwin.errors import GmatRunError
from orbitwin.models import EpochAlignment
from orbitwin.report_parser import (
    evaluate_diagnostic_report,
    parse_numeric_report,
    report_clock,
)
from orbitwin.search import zero_burn_candidate
from orbitwin.state_probe import load_scenario
from orbitwin.verify import verify_candidate_with_gmat

ROOT = Path(__file__).resolve().parents[2]
CROSSING = ROOT / "tests" / "fixtures" / "async" / "aligned_crossing.script"
FAKE = ROOT / "scripts" / "fake_gmat_console.py"
STARTUP = FAKE_STARTUP
SURFACE_KM = 6378.1363
A1_0 = 31042.0
DELTA_S = 5400.0
COLUMNS = (
    "P.ElapsedSecs T.ElapsedSecs P.A1ModJulian T.A1ModJulian "
    "P.EarthMJ2000Eq.X P.EarthMJ2000Eq.Y P.EarthMJ2000Eq.Z "
    "P.EarthMJ2000Eq.VX P.EarthMJ2000Eq.VY P.EarthMJ2000Eq.VZ "
    "T.EarthMJ2000Eq.X T.EarthMJ2000Eq.Y T.EarthMJ2000Eq.Z "
    "T.EarthMJ2000Eq.VX T.EarthMJ2000Eq.VY T.EarthMJ2000Eq.VZ"
)


def _alignment(early: str, late: str, offset_s: float = DELTA_S) -> EpochAlignment:
    return EpochAlignment(early, late, "Prop", "epoch-equality", offset_s, "", 0)


def _row(p_s, t_s, p_a1, t_a1, p_x, t_x) -> str:
    return " ".join(repr(float(v)) for v in (p_s, t_s, p_a1, t_a1, p_x, 0, 0, 0, 0, 0, t_x, 0, 0, 0, 0, 0))


def _target_early(path: Path, *, frozen_x: float = 9000.0, alien_x: float = 7000.0,
                  joint: tuple[tuple[float, float, float], ...] = ((0.0, 7000.0, 7100.0),
                                                                    (100.0, 7000.0, 7002.0),
                                                                    (200.0, 7000.0, 7100.0))) -> Path:
    """T flies alone for DELTA_S, P waits at its epoch; then both, on P's clock."""
    late_a1 = A1_0 + DELTA_S / 86400.0
    lines = [COLUMNS]
    for s in (0.0, 1800.0, 3600.0):
        # In the alignment rows T is 3 km from where P waits: not an interception.
        lines.append(_row(0.0, s, late_a1, A1_0 + s / 86400.0, frozen_x, frozen_x - 3.0))
    for t, p_x, t_x in joint:
        lines.append(_row(t, DELTA_S + t, late_a1 + t / 86400.0, late_a1 + t / 86400.0, p_x, t_x))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- the clock ---------------------------------------------------------------------


def test_after_an_alignment_rows_sit_on_the_late_crafts_clock(tmp_path: Path) -> None:
    report = parse_numeric_report(_target_early(tmp_path / "r.txt"))
    clock = report_clock(report, "P", "T", _alignment("T", "P"))
    assert clock.frozen_rows == 3
    assert list(clock.times_s) == [-5400.0, -3600.0, -1800.0, 0.0, 100.0, 200.0]


def test_without_a1_columns_an_offset_cannot_be_checked(tmp_path: Path) -> None:
    path = tmp_path / "r.txt"
    path.write_text("P.ElapsedSecs T.ElapsedSecs\n0 5400\n", encoding="utf-8")
    with pytest.raises(GmatRunError, match=r"no P\.A1ModJulian and T\.A1ModJulian"):
        report_clock(parse_numeric_report(path), "P", "T", _alignment("T", "P"))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        # Apart again after t = 0.
        (lambda rows: rows.__setitem__(5, rows[5].replace("31042.0636", "31042.0637", 1)), "row 5"),
        # The late craft moves before the early one reaches it.
        (lambda rows: rows.__setitem__(2, rows[2].replace("0.0 1800.0", "5.0 1800.0", 1)), "moves"),
    ],
)
def test_rows_apart_where_they_should_not_be_are_refused(tmp_path: Path, edit, message) -> None:
    path = _target_early(tmp_path / "r.txt")
    rows = path.read_text(encoding="utf-8").splitlines()
    before = list(rows)
    edit(rows)
    assert rows != before
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(GmatRunError, match=message):
        report_clock(parse_numeric_report(path), "P", "T", _alignment("T", "P"))


def test_leading_rows_apart_without_an_alignment_are_refused(tmp_path: Path) -> None:
    with pytest.raises(GmatRunError, match=r"apart in epoch \(A1\) at report row 1"):
        report_clock(parse_numeric_report(_target_early(tmp_path / "r.txt")), "P", "T")


def test_an_alignment_that_never_ends_is_refused(tmp_path: Path) -> None:
    path = _target_early(tmp_path / "r.txt", joint=())
    with pytest.raises(GmatRunError, match="never reach one A1 epoch"):
        report_clock(parse_numeric_report(path), "P", "T", _alignment("T", "P"))


def test_one_double_step_apart_is_one_epoch(tmp_path: Path) -> None:
    """A Precision 17 A1 day near MJD 31042 steps 0.31 µs; GMAT ends stops that far off."""
    a1 = A1_0 + DELTA_S / 86400.0
    path = tmp_path / "r.txt"
    path.write_text(
        COLUMNS + "\n" + _row(0.0, 0.0, a1, float(np.nextafter(a1, np.inf)), 7000.0, 7100.0)
        + "\n" + _row(1.0, 1.0, a1 + 1 / 86400, a1 + 1 / 86400, 7000.0, 7100.0) + "\n",
        encoding="utf-8",
    )
    assert report_clock(parse_numeric_report(path), "P", "T").frozen_rows == 0


# --- what is measured on which rows ---------------------------------------------------


def test_alignment_rows_are_not_an_interception(tmp_path: Path) -> None:
    """3 km apart while one waits is not two craft 3 km apart at one instant."""
    path = _target_early(tmp_path / "r.txt")
    event = evaluate_diagnostic_report(
        path, "P", "T", 5.0, central_body_radius_km=SURFACE_KM, alignment=_alignment("T", "P")
    )
    assert event.first_entry_time_s is not None and 0.0 < event.first_entry_time_s < 200.0


def test_an_early_pursuer_is_held_above_the_surface_through_the_alignment(tmp_path: Path) -> None:
    late_a1 = A1_0 + DELTA_S / 86400.0
    lines = [COLUMNS]
    for s, x in ((0.0, 7000.0), (1800.0, 6000.0), (3600.0, 7000.0)):  # P dips underground
        lines.append(_row(s, 0.0, A1_0 + s / 86400.0, late_a1, x, 8000.0))
    for t in (0.0, 100.0):
        lines.append(_row(DELTA_S + t, t, late_a1 + t / 86400, late_a1 + t / 86400, 7000.0, 8000.0))
    path = tmp_path / "r.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    early = evaluate_diagnostic_report(
        path, "P", "T", 5.0, central_body_radius_km=SURFACE_KM, alignment=_alignment("P", "T")
    )
    assert early.minimum_pursuer_radius_km is not None
    assert early.minimum_pursuer_radius_km <= 6000.0


def test_a_late_pursuer_waiting_is_not_measured_twice(tmp_path: Path) -> None:
    path = _target_early(tmp_path / "r.txt", frozen_x=7000.0)
    event = evaluate_diagnostic_report(
        path, "P", "T", 5.0, central_body_radius_km=SURFACE_KM, alignment=_alignment("T", "P")
    )
    assert event.minimum_pursuer_radius_km == pytest.approx(7000.0)


def test_the_ephemeris_starts_at_t_zero(tmp_path: Path) -> None:
    path = _target_early(tmp_path / "r.txt")
    ephemeris = ephemeris_from_report(path, "P", "T", _alignment("T", "P"))
    assert list(ephemeris.times_s) == [0.0, 100.0, 200.0]


def test_a_coast_after_the_target_aligns_is_read_from_where_it_meets_the_pursuer(
    tmp_path: Path,
) -> None:
    """The coast flies T alone after its alignment, so P waits at its epoch to the end."""
    late_a1 = A1_0 + DELTA_S / 86400.0
    lines = [COLUMNS]
    for s in (0.0, 2700.0, 5400.0, 5500.0, 5600.0):
        lines.append(_row(0.0, s, late_a1, A1_0 + s / 86400.0, 7000.0, 8000.0 + s))
    path = tmp_path / "coast.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ephemeris = target_rows_from_report(path, "T", _alignment("T", "P"))
    assert list(ephemeris.times_s) == [0.0, 100.0, 200.0]
    assert ephemeris.positions_km[0][0] == 13400.0


# --- the diagnostic columns and the fake console --------------------------------------


def _config() -> OrbitWinConfig:
    return OrbitWinConfig(
        policy=dataclasses.replace(
            PolicyConfig(), mission_time_origin="pursuer", closest_approach_window="after_alignment"
        ),
        pursuer_name="SC_Human",
        target_name="SC_Alien",
    )


def test_the_sibling_reports_both_a1_epochs_bit_exact() -> None:
    scenario, document, _ = load_scenario(CROSSING, _config(), gmat_console=FAKE, startup_file=STARTUP)
    text = add_diagnostic_report(document, scenario).script_bytes.decode("utf-8")
    assert "SC_Human.A1ModJulian, SC_Alien.A1ModJulian" in text
    assert "OrbitWinDiagnostic.Precision = 17;" in text


def test_verify_reads_an_aligned_flight_on_the_solver_clock(tmp_path: Path) -> None:
    """The fake writes the alignment's rows first, the late craft waiting."""
    config = _config()
    scenario, document, _ = load_scenario(CROSSING, config, gmat_console=FAKE, startup_file=STARTUP)
    candidate, _, evidence = verify_candidate_with_gmat(
        CROSSING, scenario, document, zero_burn_candidate(scenario, config), config, FAKE,
        STARTUP, tmp_path / "verify",
    )
    report = parse_numeric_report(evidence["diagnostic_report"])
    clock = report_clock(report, "SC_Human", "SC_Alien", scenario.alignment)
    assert clock.frozen_rows > 0 and clock.times_s[0] == pytest.approx(-DELTA_S)
    # The fake's default hit is at 10 s on the solver clock, not 10 s after SC_Alien's epoch.
    assert candidate.metadata["gmat_event"]["first_entry_time_s"] <= 10.0


def test_check_report_rows_skips_the_frozen_rows_and_names_its_time_column(tmp_path: Path) -> None:
    path = tmp_path / "official.txt"
    path.write_text(
        "SC_Human.UTCModJulian SC_Human.EarthMJ2000Eq.X SC_Human.EarthMJ2000Eq.Y "
        "SC_Human.EarthMJ2000Eq.Z SC_Alien.EarthMJ2000Eq.X SC_Alien.EarthMJ2000Eq.Y "
        "SC_Alien.EarthMJ2000Eq.Z\n"
        # Frozen: 1 km apart while SC_Human waits.
        "1 7000 0 0 7001 0 0\n2 7000 0 0 7100 0 0\n3 7000 0 0 7002 0 0\n",
        encoding="utf-8",
    )
    script = ROOT / "scripts" / "check_report_rows.py"
    run = [sys.executable, str(script), str(path)]
    skipped = subprocess.run([*run, "--frozen-rows", "1"], capture_output=True, text=True)
    assert skipped.returncode == 0, skipped.stdout
    assert "time column   : SC_Human.UTCModJulian" in skipped.stdout
    assert "rows within 5 km: 1" in skipped.stdout
    whole = subprocess.run(run, capture_output=True, text=True)
    assert "rows within 5 km: 2" in whole.stdout
    refused = subprocess.run([*run, "--frozen-rows", "3"], capture_output=True, text=True)
    assert refused.returncode != 0 and "leaves no row" in refused.stderr


def test_rows_are_counted_whatever_the_columns_hold(tmp_path: Path) -> None:
    """A UTCGregorian column is words; a Variable's header is a bare name (R2026a)."""
    from orbitwin.verify import _row_count

    path = tmp_path / "official.txt"
    path.write_text(
        "SC_Human.UTCGregorian dummy SC_Human.EarthMJ2000Eq.X\n"
        "01 Jan 2026 12:00:00.000 0 -6242.87\n\n01 Jan 2026 12:00:10.000 0 -6224.9\n",
        encoding="utf-8",
    )
    assert _row_count(path, headers=True) == 2
    path.write_text("01 Jan 2026 12:00:00.000 0 -6242.87\n", encoding="utf-8")
    assert _row_count(path, headers=False) == 1


def test_write_headers_is_read_off_the_report_file(tmp_path: Path) -> None:
    from orbitwin.gmat_parser import parse_gmat_script

    path = tmp_path / "t.script"
    path.write_text(
        CROSSING.read_text(encoding="utf-8").replace(
            "OfficialReport.WriteHeaders = true;", "OfficialReport.WriteHeaders = false;"
        ),
        encoding="utf-8",
    )
    assert [spec.write_headers for spec in load_scenario(
        CROSSING, _config(), gmat_console=FAKE, startup_file=STARTUP
    )[0].report_files] == [True]
    with pytest.raises(Exception, match="console"):
        parse_gmat_script(path, _config())
    assert [spec.write_headers for spec in load_scenario(
        path, _config(), gmat_console=FAKE, startup_file=STARTUP
    )[0].report_files] == [False]


def test_the_report_step_is_measured_on_the_joint_rows_only(tmp_path: Path) -> None:
    """The alignment's rows here are 1800 s apart, the joint ones 100 s."""
    path = _target_early(tmp_path / "r.txt")
    event = evaluate_diagnostic_report(path, "P", "T", 5.0, alignment=_alignment("T", "P"))
    assert event.report_step_s == 100.0


def test_the_closest_row_is_on_the_joint_rows_only(tmp_path: Path) -> None:
    """The alignment's rows, 3 km apart here, are scored by alignment_minimum_distance_km."""
    path = _target_early(tmp_path / "r.txt", joint=((0.0, 7000.0, 7100.0),
                                                    (100.0, 7000.0, 7050.0),
                                                    (200.0, 7000.0, 7100.0)))
    event = evaluate_diagnostic_report(path, "P", "T", 5.0, alignment=_alignment("T", "P"))
    assert (event.minimum_row_distance_km, event.minimum_row_distance_time_s) == (50.0, 100.0)
