from __future__ import annotations

from pathlib import Path

import pytest

from orbitwin.report_parser import evaluate_diagnostic_report

SURFACE_KM = 6378.1363
COLUMNS = (
    "P.ElapsedSecs T.ElapsedSecs "
    "P.EarthMJ2000Eq.X P.EarthMJ2000Eq.Y P.EarthMJ2000Eq.Z "
    "P.EarthMJ2000Eq.VX P.EarthMJ2000Eq.VY P.EarthMJ2000Eq.VZ "
    "T.EarthMJ2000Eq.X T.EarthMJ2000Eq.Y T.EarthMJ2000Eq.Z "
    "T.EarthMJ2000Eq.VX T.EarthMJ2000Eq.VY T.EarthMJ2000Eq.VZ"
)


def _report(path: Path, rows: list[tuple[float, float, float]]) -> Path:
    """A report where the pursuer sits on the x axis and the target trails it.

    Each row is (t, pursuer_x, separation). Only the radius and the relative
    distance matter here, so the velocities are whatever keeps the Hermite
    interpolant from inventing motion between samples.
    """
    lines = [COLUMNS]
    for time_s, radius_km, separation_km in rows:
        lines.append(
            f"{time_s} {time_s} {radius_km} 0 0 0 0 0 "
            f"{radius_km - separation_km} 0 0 0 0 0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_radius_comes_from_the_report(tmp_path: Path) -> None:
    # A pursuer that drops to 5000 km on its way in: below the surface, and a
    # number no analytical estimate contributed.
    path = _report(
        tmp_path / "diag.txt",
        [(0.0, 7000.0, 900.0), (100.0, 5000.0, 400.0), (200.0, 6800.0, 100.0)],
    )
    event = evaluate_diagnostic_report(path, "P", "T", 5.0, central_body_radius_km=SURFACE_KM)
    assert event.minimum_pursuer_radius_km == pytest.approx(5000.0)
    assert event.minimum_pursuer_radius_km < SURFACE_KM


def test_it_is_not_measured_unless_asked(tmp_path: Path) -> None:
    path = _report(tmp_path / "diag.txt", [(0.0, 7000.0, 900.0), (100.0, 5000.0, 400.0)])
    event = evaluate_diagnostic_report(path, "P", "T", 5.0)
    assert event.minimum_pursuer_radius_km is None


def test_it_measures_past_the_interception(tmp_path: Path) -> None:
    """Every flown row counts: nothing flown may go underground, before tint or after
    it (docs/ASYNC_START_AND_INSERTION.md §5). This read only to tint once."""
    path = _report(
        tmp_path / "diag.txt",
        [
            (0.0, 7000.0, 50.0),
            (100.0, 7000.0, 2.0),   # inside 5 km: this is tint
            (200.0, 3000.0, 400.0),  # the tail, far underground
            (300.0, 3000.0, 900.0),
        ],
    )
    event = evaluate_diagnostic_report(path, "P", "T", 5.0, central_body_radius_km=SURFACE_KM)
    assert event.first_entry_time_s is not None
    assert event.first_entry_time_s <= 100.0
    assert event.minimum_pursuer_radius_km is not None
    assert event.minimum_pursuer_radius_km <= 3000.0


def test_a_mission_that_never_closes_is_measured_throughout(tmp_path: Path) -> None:
    path = _report(
        tmp_path / "diag.txt",
        [(0.0, 7000.0, 900.0), (100.0, 6500.0, 800.0), (200.0, 6100.0, 700.0)],
    )
    event = evaluate_diagnostic_report(path, "P", "T", 5.0, central_body_radius_km=SURFACE_KM)
    assert event.first_entry_time_s is None
    assert event.minimum_pursuer_radius_km == pytest.approx(6100.0)


def _physical_rows(path: Path, depth_m: float, half_gap_s: float) -> tuple[Path, float]:
    """Two rows of a real two-body pursuer, half_gap_s either side of a pericenter
    depth_m under the surface; the target stays far off. Returns the lower row's radius."""
    import numpy as np

    from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, propagate_universal

    mu = 398600.4418
    rp, ra = SURFACE_KM - depth_m / 1000.0, SURFACE_KM + 600.0
    pericenter = keplerian_to_cartesian(
        KeplerianElements((rp + ra) / 2.0, (ra - rp) / (ra + rp), 30.0, 10.0, 0.0, 0.0), mu
    )
    lines = [COLUMNS]
    lowest = float("inf")
    for index, offset_s in enumerate((-half_gap_s, half_gap_s)):
        state = propagate_universal(pericenter, offset_s, mu)
        r, v = [float(x) for x in state.r], [float(x) for x in state.v]
        lowest = min(lowest, float(np.linalg.norm(r)))
        t = index * 2.0 * half_gap_s
        lines.append(
            f"{t!r} {t!r} {r[0]!r} {r[1]!r} {r[2]!r} {v[0]!r} {v[1]!r} {v[2]!r} "
            "100000 0 0 0 0 0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, lowest


# The cubic's error grows as the gap to the fourth power. Measured: a pericenter
# 1 m under is read 0.04 m low on rows 30 s apart, 0.58 m low at 60 s, 9 m low
# at 120 s; one 100 m under, 5.7 km low at 600 s. Low is the safe side.
@pytest.mark.parametrize(
    "depth_m, half_gap_s, tolerance_km",
    [(1.0, 15.0, 1.0e-3), (1.0, 30.0, 1.0e-2), (1.0, 60.0, 2.0e-2), (100.0, 300.0, 10.0)],
)
def test_a_crossing_between_two_rows_is_found(
    tmp_path: Path, depth_m: float, half_gap_s: float, tolerance_km: float
) -> None:
    """Both rows are above the surface; the arc between them is not."""
    path, lowest_row_km = _physical_rows(tmp_path / "diag.txt", depth_m, half_gap_s)
    assert lowest_row_km > SURFACE_KM
    event = evaluate_diagnostic_report(path, "P", "T", 5.0, central_body_radius_km=SURFACE_KM)
    assert event.minimum_pursuer_radius_km is not None
    assert event.minimum_pursuer_radius_km < SURFACE_KM
    true_km = SURFACE_KM - depth_m / 1000.0
    assert true_km - tolerance_km <= event.minimum_pursuer_radius_km <= true_km + 1.0e-6


def test_the_rows_after_the_entry_are_measured(tmp_path: Path) -> None:
    """The row past the entry and the ones after it are all flown."""
    path = _report(
        tmp_path / "diag.txt",
        [(0.0, 7000.0, 50.0), (100.0, 7000.0, 10.0), (200.0, 6000.0, 2.0), (300.0, 3000.0, 900.0)],
    )
    event = evaluate_diagnostic_report(path, "P", "T", 5.0, central_body_radius_km=SURFACE_KM)
    assert event.first_entry_time_s is not None and 100.0 < event.first_entry_time_s < 200.0
    assert event.minimum_pursuer_radius_km is not None
    assert event.minimum_pursuer_radius_km <= 3000.0
