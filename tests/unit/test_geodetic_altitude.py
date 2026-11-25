"""GMAT's geodetic Earth.Altitude, recorded beside the sphere check and never instead of it.

See docs/ASYNC_START_AND_INSERTION.md §5.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orbitwin.errors import GmatRunError
from orbitwin.report_parser import evaluate_diagnostic_report

SURFACE_KM = 6378.1363
HEAD = (
    "P.ElapsedSecs T.ElapsedSecs "
    "P.EarthMJ2000Eq.X P.EarthMJ2000Eq.Y P.EarthMJ2000Eq.Z "
    "P.EarthMJ2000Eq.VX P.EarthMJ2000Eq.VY P.EarthMJ2000Eq.VZ "
    "T.EarthMJ2000Eq.X T.EarthMJ2000Eq.Y T.EarthMJ2000Eq.Z "
    "T.EarthMJ2000Eq.VX T.EarthMJ2000Eq.VY T.EarthMJ2000Eq.VZ"
)


def _report(path: Path, rows: list[tuple[float, float, float | None]]) -> Path:
    """(t, pursuer radius on the x axis, reported altitude) per row."""
    altitude = any(row[2] is not None for row in rows)
    lines = [HEAD + (" P.Earth.Altitude" if altitude else "")]
    for time_s, radius_km, altitude_km in rows:
        lines.append(
            f"{time_s} {time_s} {radius_km} 0 0 0 0 0 {radius_km + 900.0} 0 0 0 0 0"
            + (f" {altitude_km}" if altitude else "")
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_lowest_reported_altitude_is_recorded(tmp_path: Path) -> None:
    # Off the equator geodetic altitude reads higher than r - R: 496.28 km at
    # RMAG 6871 and latitude 23.5 deg on R2026a, against 492.86 km.
    path = _report(tmp_path / "r.txt", [(0.0, 6871.0, 496.28), (100.0, 6900.0, 523.0)])
    event = evaluate_diagnostic_report(path, "P", "T", 5.0, central_body_radius_km=SURFACE_KM,
                                       geodetic_radius_km=SURFACE_KM)
    assert event.minimum_geodetic_altitude_km == 496.28
    assert event.minimum_pursuer_radius_km == pytest.approx(6871.0)


def test_an_altitude_below_the_sphere_is_a_report_that_lies(tmp_path: Path) -> None:
    path = _report(tmp_path / "r.txt", [(0.0, 6871.0, 496.28), (100.0, 6900.0, 400.0)])
    with pytest.raises(GmatRunError, match=r"row 2 reports P at geodetic altitude 400\.0 km"):
        evaluate_diagnostic_report(path, "P", "T", 5.0, central_body_radius_km=SURFACE_KM,
                                       geodetic_radius_km=SURFACE_KM)


def test_without_the_column_nothing_is_recorded(tmp_path: Path) -> None:
    path = _report(tmp_path / "r.txt", [(0.0, 6871.0, None), (100.0, 6900.0, None)])
    event = evaluate_diagnostic_report(path, "P", "T", 5.0, central_body_radius_km=SURFACE_KM,
                                       geodetic_radius_km=SURFACE_KM)
    assert event.minimum_geodetic_altitude_km is None


def test_a_positive_geodetic_altitude_does_not_lift_the_sphere_check(tmp_path: Path) -> None:
    """The sphere reads 6370 km, under the surface; geodetic says 3 km up. The sphere decides."""
    path = _report(tmp_path / "r.txt", [(0.0, 6871.0, 496.28), (100.0, 6370.0, 3.0)])
    event = evaluate_diagnostic_report(path, "P", "T", 5.0, central_body_radius_km=SURFACE_KM,
                                       geodetic_radius_km=SURFACE_KM)
    assert event.minimum_pursuer_radius_km is not None
    assert event.minimum_pursuer_radius_km < SURFACE_KM
    assert event.minimum_geodetic_altitude_km == 3.0


def test_the_cross_check_uses_gmats_radius_not_the_policys(tmp_path: Path) -> None:
    """policy.central_body_radius_km 6371 refused every sound row (R2026a, 51.6 deg)."""
    path = _report(tmp_path / "r.txt", [(0.0, 7000.0, 7000.0 - SURFACE_KM)])
    event = evaluate_diagnostic_report(
        path, "P", "T", 5.0, central_body_radius_km=6371.0, geodetic_radius_km=SURFACE_KM
    )
    assert event.minimum_geodetic_altitude_km == pytest.approx(7000.0 - SURFACE_KM)
    assert event.minimum_pursuer_radius_km == pytest.approx(7000.0)
    unchecked = evaluate_diagnostic_report(path, "P", "T", 5.0, central_body_radius_km=6371.0)
    assert unchecked.minimum_geodetic_altitude_km is None


def test_the_radius_follows_the_script(tmp_path: Path) -> None:
    from orbitwin.config import OrbitWinConfig
    from orbitwin.gmat_parser import parse_gmat_script

    root = Path(__file__).resolve().parents[2]
    source = (root / "examples" / "two_burn_rendezvous.script").read_text(encoding="utf-8")
    assert parse_gmat_script(root / "examples" / "two_burn_rendezvous.script",
                             OrbitWinConfig())[0].geodetic_radius_km == SURFACE_KM
    path = tmp_path / "radius.script"
    path.write_text(source.replace("Create Spacecraft HumanSC;", "Earth.EquatorialRadius = 6378.137;\nCreate Spacecraft HumanSC;"), encoding="utf-8")
    assert parse_gmat_script(path, OrbitWinConfig())[0].geodetic_radius_km == 6378.137
