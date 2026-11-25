"""The post-burn row R2026a does not write, and verification reading with it (#106)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orbitwin.burn_rows import FlownManeuver, maneuvers_flown, with_post_burn_rows
from orbitwin.config import OrbitWinConfig
from orbitwin.errors import GmatRunError
from orbitwin.frames import script_components_to_inertial
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import StateVector, as_vector3
from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, propagate_universal
from orbitwin.report_parser import evaluate_diagnostic_report

MU = 398600.4418
HEADER = "P.ElapsedSecs T.ElapsedSecs " + " ".join(
    f"{craft}.EarthMJ2000Eq.{axis}" for craft in "PT" for axis in ("X", "Y", "Z", "VX", "VY", "VZ")
)
VNB = ("Local", "Earth", "VNB")
LVLH = ("Local", "Earth", "LVLH")
INERTIAL = ("EarthMJ2000Eq", "Earth", "MJ2000Eq")
BURN = FlownManeuver(240.0, (0.25, 0.0, 0.0), VNB)


def _flight(
    *, post_burn_row: bool, burn: FlownManeuver = BURN, step_row_s: float | None = None
) -> tuple[list[tuple[float, StateVector, StateVector]], float]:
    """Rows every 240 s, a 250 m/s burn at 240 s, the target met at 360 s.

    Only the pre-burn row at 240 s, as R2026a writes it, unless asked. With
    `step_row_s`, an integrator step's row that far before the burn as well,
    as R2026a writes when a step lands within the tolerance of the goal.
    """
    pursuer = keplerian_to_cartesian(KeplerianElements(7000.0, 0.0, 28.0, 15.0, 0.0, 0.0), MU)
    pre = propagate_universal(pursuer, 240.0, MU)
    impulse = script_components_to_inertial(np.asarray(burn.components_km_s), pre, *burn.frame)
    post = StateVector(pre.r_km, as_vector3(pre.v + impulse))
    meet = propagate_universal(post, 120.0, MU)
    # The target crosses the pursuer's path at 360 s, 1 km/s across it.
    across = np.cross(meet.r, meet.v)
    target_at_meet = StateVector(
        meet.r_km, as_vector3(meet.v + across / np.linalg.norm(across))
    )
    rows = []
    if step_row_s is not None:
        rows.append((
            240.0 - step_row_s,
            propagate_universal(pursuer, 240.0 - step_row_s, MU),
            propagate_universal(target_at_meet, -120.0 - step_row_s, MU),
        ))
    for t in (0.0, 240.0, 480.0, 720.0):
        state = propagate_universal(pursuer, t, MU) if t <= 240.0 else propagate_universal(
            post, t - 240.0, MU
        )
        rows.append((t, state, propagate_universal(target_at_meet, t - 360.0, MU)))
        if t == 240.0 and post_burn_row:
            rows.append((t, post, propagate_universal(target_at_meet, -120.0, MU)))
    return sorted(rows, key=lambda row: row[0]), 360.0


def _report(path: Path, rows: list[tuple[float, StateVector, StateVector]]) -> Path:
    lines = [HEADER]
    for t, p, q in rows:
        values = [t, t, *p.r, *p.v, *q.r, *q.v]
        lines.append(" ".join(repr(float(value)) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_hit_in_the_step_after_a_burn_is_read_with_the_post_burn_state(
    tmp_path: Path,
) -> None:
    rows, meet_s = _flight(post_burn_row=False)
    path = _report(tmp_path / "r.txt", rows)
    fixed = evaluate_diagnostic_report(path, "P", "T", 5.0, maneuvers=[BURN])
    assert fixed.minimum_distance_km < 0.05
    assert fixed.minimum_distance_time_s == pytest.approx(meet_s, abs=1.0)
    assert fixed.first_entry_time_s is not None and meet_s - 10.0 < fixed.first_entry_time_s
    # Read as before, Hermite carries the pre-burn velocity into the step and
    # misses the hit: 7.48 km closest, no entry.
    old = evaluate_diagnostic_report(path, "P", "T", 5.0)
    assert old.first_entry_time_s is None and old.minimum_distance_km > 5.0


def test_a_post_burn_row_gmat_wrote_is_used_as_it_is(tmp_path: Path) -> None:
    written = _report(tmp_path / "w.txt", _flight(post_burn_row=True)[0])
    rebuilt = _report(tmp_path / "r.txt", _flight(post_burn_row=False)[0])
    with_row = evaluate_diagnostic_report(written, "P", "T", 5.0, maneuvers=[BURN])
    assert with_row == evaluate_diagnostic_report(written, "P", "T", 5.0)
    assert with_row.minimum_distance_km == pytest.approx(
        evaluate_diagnostic_report(rebuilt, "P", "T", 5.0, maneuvers=[BURN]).minimum_distance_km,
        abs=1.0e-9,
    )


def test_a_burn_after_the_last_row_is_skipped_and_one_without_a_row_refused() -> None:
    times = np.array([0.0, 100.0, 200.0])
    states = np.ones((3, 3))
    unchanged = with_post_burn_rows(
        times, states, states, states, states, [FlownManeuver(300.0, (0.1, 0, 0), VNB)], "r"
    )
    assert len(unchanged[0]) == 3
    with pytest.raises(GmatRunError, match="no row at the burn epoch 150"):
        with_post_burn_rows(
            times, states, states, states, states, [FlownManeuver(150.0, (0.1, 0, 0), VNB)], "r"
        )


def test_the_maneuvers_are_read_off_the_script(workshop_script: Path) -> None:
    scenario, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    flown = maneuvers_flown(scenario)
    assert len(flown) == len(scenario.burn_slots) == 2
    durations = [segment.duration_s for segment in scenario.propagate_segments]
    assert [maneuver.time_s for maneuver in flown] == pytest.approx(
        [durations[0], durations[0] + durations[1]]
    )
    for maneuver, slot in zip(flown, scenario.burn_slots, strict=True):
        assert maneuver.components_km_s == tuple(slot.vector)
        assert maneuver.frame == (slot.coordinate_system, slot.origin, slot.axes)


@pytest.mark.parametrize("frame", [VNB, LVLH, INERTIAL], ids=["VNB", "LVLH", "EarthMJ2000Eq"])
def test_the_post_burn_row_is_rebuilt_in_the_burn_s_own_frame(
    tmp_path: Path, frame: tuple[str, str, str]
) -> None:
    burn = FlownManeuver(240.0, (0.2, -0.1, 0.12), frame)
    rows, meet_s = _flight(post_burn_row=False, burn=burn)
    event = evaluate_diagnostic_report(
        _report(tmp_path / "r.txt", rows), "P", "T", 5.0, maneuvers=[burn]
    )
    assert event.minimum_distance_km < 0.05
    assert event.minimum_distance_time_s == pytest.approx(meet_s, abs=1.0)


def test_two_pre_burn_rows_at_the_epoch_still_get_the_post_burn_row(tmp_path: Path) -> None:
    """R2026a: a step 4e-6 s short of the goal, then the goal, both before the burn."""
    rows, meet_s = _flight(post_burn_row=False, step_row_s=4.0e-6)
    times = np.array([row[0] for row in rows])
    assert np.count_nonzero(np.abs(times - 240.0) <= 1.0e-4) == 2
    event = evaluate_diagnostic_report(
        _report(tmp_path / "r.txt", rows), "P", "T", 5.0, maneuvers=[BURN]
    )
    assert event.minimum_distance_km < 0.05
    assert event.minimum_distance_time_s == pytest.approx(meet_s, abs=1.0)


@pytest.mark.parametrize(
    ("offset_s", "accepted"),
    [
        (2.0e-6, True), (-2.0e-6, True), (5.0e-5, True), (-5.0e-5, True),
        (1.2e-4, False), (2.0e-4, False),
    ],
)
def test_the_burn_epoch_tolerance_is_a_tenth_of_a_millisecond(
    offset_s: float, accepted: bool
) -> None:
    """R2026a's rows sit up to 1.87 us off the summed goals; 5e-5 s is still the burn's,
    1.2e-4 s another step."""
    times = np.array([0.0, 240.0 + offset_s, 480.0])
    states = np.array([[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0], [-7000.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 7.5, 0.0], [-7.5, 0.0, 0.0], [0.0, -7.5, 0.0]])
    if not accepted:
        with pytest.raises(GmatRunError, match="no row at the burn epoch 240"):
            with_post_burn_rows(times, states, velocities, states, velocities, [BURN], "r")
        return
    rebuilt = with_post_burn_rows(times, states, velocities, states, velocities, [BURN], "r")
    assert len(rebuilt[0]) == 4 and rebuilt[0][2] == rebuilt[0][1]


def _two_rows_at_the_burn(jump_km_s: np.ndarray) -> tuple[np.ndarray, ...]:
    """A pre-burn row at 240 s and a second one there, the given jump above it."""
    rows, _ = _flight(post_burn_row=False)
    at = next(index for index, row in enumerate(rows) if row[0] == 240.0)
    times = np.array([row[0] for row in rows])
    pursuer_r = np.array([row[1].r for row in rows])
    pursuer_v = np.array([row[1].v for row in rows])
    target_r = np.array([row[2].r for row in rows])
    target_v = np.array([row[2].v for row in rows])
    return (
        np.insert(times, at + 1, 240.0),
        np.insert(pursuer_r, at + 1, pursuer_r[at], axis=0),
        np.insert(pursuer_v, at + 1, pursuer_v[at] + jump_km_s, axis=0),
        np.insert(target_r, at + 1, target_r[at], axis=0),
        np.insert(target_v, at + 1, target_v[at], axis=0),
    )


def _impulse() -> np.ndarray:
    rows, _ = _flight(post_burn_row=False)
    pre = next(row[1] for row in rows if row[0] == 240.0)
    return script_components_to_inertial(np.asarray(BURN.components_km_s), pre, *VNB)


@pytest.mark.parametrize("impulse_km_s", [0.25, 0.005, 0.0021])
def test_a_written_post_burn_row_with_gmat_s_noise_is_used_as_it_is(impulse_km_s: float) -> None:
    """9e-13 m/s off the rebuilt one, as measured on R2026a: still GMAT's own row.

    That is one ulp of a 7.5 km/s velocity, a fixed amount, so the match has to
    hold for small impulses as well as large.
    """
    burn = FlownManeuver(240.0, (impulse_km_s, 0.0, 0.0), VNB)
    pre = next(row[1] for row in _flight(post_burn_row=False)[0] if row[0] == 240.0)
    impulse = script_components_to_inertial(np.asarray(burn.components_km_s), pre, *VNB)
    rows = _two_rows_at_the_burn(impulse + np.array([9.0e-16, -9.0e-16, 9.0e-16]))
    rebuilt = with_post_burn_rows(*rows, [burn], "r")
    assert len(rebuilt[0]) == len(rows[0])


def test_a_second_row_half_a_percent_short_is_the_post_burn_row() -> None:
    rows = _two_rows_at_the_burn(0.995 * _impulse())
    assert len(with_post_burn_rows(*rows, [BURN], "r")[0]) == len(rows[0])


@pytest.mark.parametrize("fraction", [0.6, 0.985, 0.988])
def test_a_second_row_short_of_the_impulse_is_not_taken_for_post_burn(fraction: float) -> None:
    """1.2% short is outside the 1% match."""
    rows = _two_rows_at_the_burn(fraction * _impulse())
    rebuilt = with_post_burn_rows(*rows, [BURN], "r")
    assert len(rebuilt[0]) == len(rows[0]) + 1


def test_a_burn_on_the_last_row_gets_its_post_burn_row() -> None:
    """A run that stops at a burn epoch still reports the burn's own row."""
    rows, _ = _flight(post_burn_row=False)
    rows = [row for row in rows if row[0] <= 240.0]
    times = np.array([row[0] for row in rows])
    pursuer_r = np.array([row[1].r for row in rows])
    pursuer_v = np.array([row[1].v for row in rows])
    target_r = np.array([row[2].r for row in rows])
    target_v = np.array([row[2].v for row in rows])
    rebuilt = with_post_burn_rows(
        times, pursuer_r, pursuer_v, target_r, target_v, [BURN], "r"
    )
    assert len(rebuilt[0]) == len(times) + 1 and rebuilt[0][-1] == 240.0
    impulse = script_components_to_inertial(
        np.asarray(BURN.components_km_s), StateVector(rows[-1][1].r_km, rows[-1][1].v_km_s), *VNB
    )
    assert np.allclose(rebuilt[2][-1], pursuer_v[-1] + impulse, rtol=0.0, atol=1.0e-15)


def test_a_rebuilt_row_takes_the_impulse_on_the_later_pre_burn_state() -> None:
    """Of two pre-burn rows the later is nearer the burn; its state builds the frame."""
    burn = FlownManeuver(240.0, (0.2, -0.1, 0.12), VNB)
    pre = next(row[1] for row in _flight(post_burn_row=False)[0] if row[0] == 240.0)
    # Off the velocity, so the second row's VNB frame is not the first's.
    turned = script_components_to_inertial(np.asarray(burn.components_km_s), pre, *VNB)
    rows = list(_two_rows_at_the_burn(0.6 * turned))
    at = np.flatnonzero(rows[0] == 240.0)
    rebuilt = with_post_burn_rows(*rows, [burn], "r")
    later = StateVector(as_vector3(rows[1][at[-1]]), as_vector3(rows[2][at[-1]]))
    expected = rows[2][at[-1]] + script_components_to_inertial(
        np.asarray(burn.components_km_s), later, *VNB
    )
    assert np.array_equal(rebuilt[2][at[-1] + 1], expected)


def test_a_zero_maneuver_needs_no_row() -> None:
    """Nothing to rebuild, so no row at its epoch is required either."""
    times = np.array([0.0, 100.0, 200.0])
    states = np.ones((3, 3))
    zero = FlownManeuver(150.0, (0.0, 0.0, 0.0), VNB)
    rebuilt = with_post_burn_rows(times, states, states, states, states, [zero], "r")
    assert len(rebuilt[0]) == 3


def test_of_three_rows_at_the_epoch_the_last_is_judged_against_the_first() -> None:
    """Pre-burn, then 0.6 of the impulse, then the whole: the last is the post-burn row."""
    impulse = _impulse()
    times, pr, pv, qr, qv = _two_rows_at_the_burn(impulse)
    at = int(np.flatnonzero(times == 240.0)[0])
    times = np.insert(times, at + 1, 240.0)
    pr = np.insert(pr, at + 1, pr[at], axis=0)
    pv = np.insert(pv, at + 1, pv[at] + 0.6 * impulse, axis=0)
    qr = np.insert(qr, at + 1, qr[at], axis=0)
    qv = np.insert(qv, at + 1, qv[at], axis=0)
    rebuilt = with_post_burn_rows(times, pr, pv, qr, qv, [BURN], "r")
    assert len(rebuilt[0]) == len(times)


def test_the_lowest_radius_is_read_with_the_post_burn_state(tmp_path: Path) -> None:
    """A retrograde burn: Hermite across it with the pre-burn velocity read 6940.0 km."""
    burn = FlownManeuver(240.0, (-0.25, 0.0, 0.0), VNB)
    rows, _ = _flight(post_burn_row=False, burn=burn)
    event = evaluate_diagnostic_report(
        _report(tmp_path / "r.txt", rows), "P", "T", 5.0,
        central_body_radius_km=6378.1363, maneuvers=[burn],
    )
    post = StateVector(rows[1][1].r_km, as_vector3(rows[1][1].v + script_components_to_inertial(
        np.asarray(burn.components_km_s), rows[1][1], *burn.frame
    )))
    # To the last row: the radius is read on every flown row, past the interception too.
    lowest = min(
        float(np.linalg.norm(propagate_universal(post, float(s), MU).r))
        for s in np.linspace(0.0, 480.0, 4801)
    )
    assert event.minimum_pursuer_radius_km == pytest.approx(lowest, abs=1.0e-3)
