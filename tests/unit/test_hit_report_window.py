from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orbitwin.report_parser import evaluate_diagnostic_report

DEPTH_KM = 1.5

HEADER = "P.ElapsedSecs T.ElapsedSecs " + " ".join(
    f"{craft}.EarthMJ2000Eq.{axis}" for craft in "PT" for axis in ("X", "Y", "Z", "VX", "VY", "VZ")
)


def _report(path: Path, rows: list[tuple[float, float, float]]) -> Path:
    """Each row is (t, x, vx) of the pursuer; the target sits still at the origin + 7000 km."""
    lines = [HEADER]
    for t, x, vx in rows:
        lines.append(f"{t} {t} {7000.0 + x} 0 0 {vx} 0 0 7000 0 0 0 0 0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_later_pass_does_not_hide_an_entry_with_no_row_inside(tmp_path: Path) -> None:
    """Fast through the sphere between two rows, then back inside much later.

    Counting the later rows said the entry rests on a row when it does not.
    """
    rows = [(0.0, -8.0, 10.0), (1.0, 2.0, 10.0), (2.0, 12.0, 10.0),
            (100.0, 1.0, 0.0), (101.0, 1.0, 0.0), (102.0, 1.0, 0.0)]
    event = evaluate_diagnostic_report(_report(tmp_path / "r.txt", rows), "P", "T", 5.0)
    assert event.first_entry_time_s == pytest.approx(0.3, abs=1e-6)
    assert event.samples_inside_radius == 1
    fast = [rows[0], (1.0, 6.0, 14.0), (2.0, 20.0, 14.0), *rows[3:]]
    event = evaluate_diagnostic_report(_report(tmp_path / "f.txt", fast), "P", "T", 5.0)
    assert event.first_entry_time_s is not None
    assert event.samples_inside_radius == 0
    # The passage lies between a row and the first row outside again.
    assert event.minimum_distance_km < 5.0


def test_a_hit_is_measured_on_its_first_passage(tmp_path: Path) -> None:
    """In to 3.5 km, out again, then a closer pass later: that one is not the hit."""
    rows = [(0.0, 9.0, -3.0), (1.0, 6.0, -2.0), (2.0, 4.0, -1.0), (3.0, 3.5, 0.0),
            (4.0, 4.0, 1.0), (5.0, 6.0, 2.0), (6.0, 9.0, 2.0),
            (50.0, 2.0, -1.0), (51.0, 0.5, 0.0), (52.0, 2.0, 1.0)]
    event = evaluate_diagnostic_report(_report(tmp_path / "r.txt", rows), "P", "T", 5.0)
    assert event.minimum_distance_km == pytest.approx(3.5, abs=1e-6)
    assert event.minimum_distance_time_s == pytest.approx(3.0, abs=1e-6)
    assert event.samples_inside_radius == 3


def _first_root(p0: float, v0: float, p1: float, v1: float, level: float) -> float:
    """First u in (0, 1) where the unit-interval cubic Hermite reaches ``level``."""
    a, b = 2 * p0 + v0 - 2 * p1 + v1, -3 * p0 - 2 * v0 + 3 * p1 - v1
    roots = np.roots([a, b, v0, p0 - level])
    return min(r.real for r in roots if abs(r.imag) < 1e-12 and 0.0 < r.real < 1.0)


def test_a_hit_stops_its_stop_depth_inside_or_at_its_deepest(tmp_path: Path) -> None:
    """Not on the first row inside: that one can be millimetres in, or a whole row late."""
    slow = [(t, 9.0 - 3.0 * t, -3.0) for t in (0.0, 1.0, 2.0, 3.0, 4.0)]
    event = evaluate_diagnostic_report(_report(tmp_path / "s.txt", slow), "P", "T", 5.0, stop_depth_km=DEPTH_KM)
    assert event.stop_time_s == pytest.approx((9.0 - (5.0 - DEPTH_KM)) / 3.0, abs=1e-9)
    assert event.stop_time_s < 2.0

    shallow_row = [(0.0, 9.0, -4.0), (1.0, 4.999, -3.0), (2.0, 1.0, -3.0), (3.0, 9.0, 4.0)]
    event = evaluate_diagnostic_report(_report(tmp_path / "r.txt", shallow_row), "P", "T", 5.0, stop_depth_km=DEPTH_KM)
    assert 1.0 < event.stop_time_s < 2.0

    grazing = [(0.0, 9.0, -2.0), (1.0, 4.98, 0.0), (2.0, 9.0, 2.0)]
    event = evaluate_diagnostic_report(_report(tmp_path / "g.txt", grazing), "P", "T", 5.0, stop_depth_km=DEPTH_KM)
    assert event.stop_time_s == pytest.approx(1.0, abs=1e-9)

    # Through the sphere between two rows.
    fast = [(0.0, -8.0, 10.0), (1.0, 6.0, 14.0), (2.0, 20.0, 14.0)]
    event = evaluate_diagnostic_report(_report(tmp_path / "f.txt", fast), "P", "T", 5.0, stop_depth_km=DEPTH_KM)
    assert event.samples_inside_radius == 0
    assert event.stop_time_s == pytest.approx(
        _first_root(-8.0, 10.0, 6.0, 14.0, -(5.0 - DEPTH_KM)), abs=1e-9
    )

    # Ends inside, short of the stop depth and still closing: where to stop is not measured.
    unfinished = [(0.0, 9.0, -4.0), (1.0, 4.99, -0.01)]
    event = evaluate_diagnostic_report(_report(tmp_path / "u.txt", unfinished), "P", "T", 5.0, stop_depth_km=DEPTH_KM)
    assert event.first_entry_time_s is not None
    assert event.stop_time_s is None

    miss = [(0.0, 9.0, 0.0), (1.0, 9.0, 0.0)]
    event = evaluate_diagnostic_report(_report(tmp_path / "m.txt", miss), "P", "T", 5.0, stop_depth_km=DEPTH_KM)
    assert event.stop_time_s is None


def test_the_stop_depth_is_the_one_passed(tmp_path: Path) -> None:
    """policy.stop_depth_km reaches the stop; without one, no stop is located."""
    slow = _report(tmp_path / "s.txt", [(t, 9.0 - 3.0 * t, -3.0) for t in (0.0, 1.0, 2.0, 3.0, 4.0)])
    for depth_km in (0.05, 0.6, 1.5, 2.5):
        event = evaluate_diagnostic_report(slow, "P", "T", 5.0, stop_depth_km=depth_km)
        assert event.stop_time_s == pytest.approx((4.0 + depth_km) / 3.0, abs=1e-9)
    # Capped at half the radius.
    event = evaluate_diagnostic_report(slow, "P", "T", 5.0, stop_depth_km=4.9)
    assert event.stop_time_s == pytest.approx(6.5 / 3.0, abs=1e-9)
    assert evaluate_diagnostic_report(slow, "P", "T", 5.0).stop_time_s is None
