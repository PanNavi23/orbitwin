"""The capture detector and the radius crossings, against dense exact coasts."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import orbitwin.duel.events as events
from orbitwin.duel.events import (
    Arc,
    UnusableArc,
    below_radius,
    capture_facts,
    coast_rows,
    segment_bounds,
)
from orbitwin.errors import DynamicsError
from orbitwin.models import StateVector
from orbitwin.orbit import propagate_universal

from .conftest import MU, Orbit


def _crossing(circular: Orbit, duration: float = 1800.0) -> tuple[Arc, Arc]:
    """Two circles at 6878 km, planes 2 degrees apart, meeting near the node."""
    p = circular(6878.0, 51.6, 10.0, -20.0)
    e = circular(6878.0, 53.6, 10.0, -19.98)
    return Arc(p[0], p[1], duration, MU), Arc(e[0], e[1], duration, MU)


def _dense(pursuer: Arc, evader: Arc, step: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, pursuer.duration_s, step)
    rp, _ = pursuer.states(times)
    re, _ = evader.states(times)
    return times, np.linalg.norm(rp - re, axis=1)


def test_the_knots_alone_miss_a_capture_the_detector_finds(circular: Orbit) -> None:
    pursuer, evader = _crossing(circular)
    knots = np.linspace(0.0, 1800.0, 31)
    rp, _ = pursuer.states(knots)
    re, _ = evader.states(knots)
    assert np.linalg.norm(rp - re, axis=1).min() > 4.6  # samples say no capture at 3 km
    facts = capture_facts(pursuer, evader, 3.0, 60.0)
    times, distance = _dense(pursuer, evader)
    assert facts.first_entry_s == pytest.approx(times[np.argmax(distance <= 3.0)], abs=0.02)
    assert facts.minimum_distance_km == pytest.approx(distance.min(), abs=1e-4)
    (passage,) = facts.passages_favour
    assert passage.duration_s == pytest.approx((distance <= 3.0).sum() * 0.01, abs=0.03)


def test_a_radius_between_the_bound_and_the_truth_is_refined_and_cleared(
    circular: Orbit,
) -> None:
    pursuer, evader = _crossing(circular)
    times, distance = _dense(pursuer, evader)
    k = int(times[np.argmin(distance)] // 60.0)
    t0, t1 = 60.0 * k, 60.0 * (k + 1)
    p0, pv0 = pursuer.state(t0)
    e0, ev0 = evader.state(t0)
    p1, pv1 = pursuer.state(t1)
    e1, ev1 = evader.state(t1)
    rmin_p, rmin_e = pursuer.minimum_radius(), evader.minimum_radius()
    lower, _ = segment_bounds(p0 - e0, pv0 - ev0, p1 - e1, pv1 - ev1, 60.0,
                              MU / rmin_p**2 + MU / rmin_e**2, min(rmin_p, rmin_e), MU)
    truth = distance.min()
    assert lower < truth - 0.03
    radius = 0.5 * (lower + truth)
    facts = capture_facts(pursuer, evader, radius, 60.0)
    assert facts.first_entry_s is None
    assert len(facts.pieces) > 30  # the segment was split, not passed over


def test_the_bounds_never_cross_the_truth_on_random_segments() -> None:
    rng = np.random.default_rng(20261102)
    for _ in range(300):
        radius = rng.uniform(6600.0, 9000.0)
        r = rng.normal(size=3)
        r *= radius / np.linalg.norm(r)
        v = rng.normal(size=3)
        v -= v.dot(r) / r.dot(r) * r
        v *= math.sqrt(MU / radius) * rng.uniform(0.9, 1.1) / np.linalg.norm(v)
        offset = rng.normal(size=3) * rng.uniform(1.0, 40.0)
        dv = rng.normal(size=3) * rng.uniform(0.001, 0.3)
        h = rng.uniform(20.0, 120.0)
        pursuer = Arc(r, v, h, MU)
        evader = Arc(r + offset, v + dv, h, MU)
        p0, pv0 = pursuer.state(0.0)
        e0, ev0 = evader.state(0.0)
        p1, pv1 = pursuer.state(h)
        e1, ev1 = evader.state(h)
        rmin_p, rmin_e = pursuer.minimum_radius(), evader.minimum_radius()
        lower, upper = segment_bounds(p0 - e0, pv0 - ev0, p1 - e1, pv1 - ev1, h,
                                      MU / rmin_p**2 + MU / rmin_e**2, min(rmin_p, rmin_e), MU)
        _, distance = _dense(pursuer, evader, step=h / 2000.0)
        assert lower <= distance.min() + 1e-9
        assert upper >= distance.max() - 1e-9


def test_a_long_slow_capture_terminates_with_its_dwell(circular: Orbit) -> None:
    p = circular(6878.0, 51.6, 10.0, 0.0)
    e = circular(6878.0, 51.6, 10.0, 0.005)  # about 0.6 km ahead on the same circle
    pursuer = Arc(p[0], p[1] + np.array([0.0, 0.0, 0.0005]), 1800.0, MU)
    evader = Arc(e[0], e[1], 1800.0, MU)
    facts = capture_facts(pursuer, evader, 2.0, 60.0)
    _, distance = _dense(pursuer, evader, step=0.1)
    inside = (distance <= 2.0).sum() * 0.1
    assert inside > 300.0
    assert facts.passages_favour
    assert sum(p.duration_s for p in facts.passages_favour) <= inside + 0.2
    assert sum(p.duration_s for p in facts.passages_hurt) >= inside - 0.2


def test_enter_leave_and_enter_deeper_reports_the_first_entry(circular: Orbit) -> None:
    pursuer, evader = _crossing(circular, duration=7000.0)
    times, distance = _dense(pursuer, evader, step=0.05)
    inside = distance <= 3.0
    entries = np.flatnonzero(inside[1:] & ~inside[:-1]) + 1
    assert len(entries) >= 2  # the planes cross twice per orbit
    facts = capture_facts(pursuer, evader, 3.0, 60.0)
    assert facts.first_entry_s == pytest.approx(times[entries[0]], abs=0.06)
    assert len(facts.passages_favour) == len(entries)


def test_a_dip_between_knots_is_caught_and_each_passage_judged_alone() -> None:
    floor = 6378.1363 + 10.0
    periapsis = floor - 0.001
    e = 0.01
    a = periapsis / (1.0 - e)
    speed = math.sqrt(MU * (2.0 / periapsis - 1.0 / a))
    period = 2.0 * math.pi * math.sqrt(a**3 / MU)
    start = Arc(np.array([periapsis, 0.0, 0.0]), np.array([0.0, speed, 0.0]), period, MU)
    r_half, v_half = start.state(period / 2.0 + 17.0)
    arc = Arc(r_half, v_half, 3.0 * period, MU)
    knots = np.arange(0.0, 3.0 * period, 60.0)
    radii = np.linalg.norm(arc.states(knots)[0], axis=1)
    assert radii.min() > floor
    passages = below_radius(arc, floor)
    assert len(passages) == 3
    expected = 2.0 * math.sqrt(2.0 * 0.001 * periapsis**2 / (MU * e))
    for passage in passages:
        assert passage.duration_s == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize(
    ("e", "duration"),
    [(1.000001, 600.0), (0.9999, 3600.0), (0.99999, 600.0), (1.0001, 60.0),
     (0.999999, 3600.0), (0.999999, 86400.0)],
)
def test_high_eccentricity_arcs_coast(e: float, duration: float) -> None:
    speed = math.sqrt(MU * (1.0 + e) / 6878.0)
    arc = Arc(np.array([6878.0, 0.0, 0.0]), np.array([0.0, speed, 0.0]), duration, MU)
    r, _ = arc.state(duration)
    reference = propagate_universal(
        StateVector((6878.0, 0.0, 0.0), (0.0, speed, 0.0)), duration, MU
    )
    assert np.linalg.norm(r - np.array(reference.r_km)) < 1e-6 * max(1.0, np.linalg.norm(r))
    assert below_radius(arc, 6378.1363) == []


#: Found by the phase-1 code review: `conic.coast` does not converge at 180 s.
FAR_HYPERBOLA = (np.array([-562761.0455600958, 478353.4374472043, 0.0]),
                 np.array([-3.2554531725226434, 2.6262620276855513, 0.0]))


def test_a_far_hyperbola_coasts_through_the_fallback_solver() -> None:
    r0, v0 = FAR_HYPERBOLA
    arc = Arc(r0, v0, 1800.0, MU)
    reference = propagate_universal(StateVector(r0, v0), 180.0, MU)
    assert np.linalg.norm(arc.state(180.0)[0] - reference.r) < 1e-6
    knots = np.arange(0.0, 1801.0, 60.0)
    r, _ = arc.states(knots)
    assert np.linalg.norm(r[3] - reference.r) < 1e-6


@pytest.mark.parametrize("error", [DynamicsError, OverflowError, ValueError],
                         ids=lambda error: error.__name__)
def test_a_failing_batch_is_rerun_row_by_row_and_only_its_bad_row_lost(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    def batch(*_args: object) -> None:
        raise error("injected")

    real = events.coast_one

    def one(r: np.ndarray, v: np.ndarray, t: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
        if t == 120.0:
            raise error("injected row")
        return real(r, v, t, mu)

    monkeypatch.setattr(events, "coast_many", batch)
    monkeypatch.setattr(events, "coast_one", one)
    times = np.array([0.0, 60.0, 120.0, 180.0])
    r0 = np.tile([7000.0, 0.0, 0.0], (4, 1))
    v0 = np.tile([0.0, 7.5, 0.0], (4, 1))
    r, _, ok = coast_rows(r0, v0, times, MU)
    assert ok.tolist() == [True, True, False, True]
    assert np.isnan(r[2]).all() and np.isfinite(r[[0, 1, 3]]).all()
    expected = propagate_universal(StateVector((7000.0, 0.0, 0.0), (0.0, 7.5, 0.0)), 180.0, MU)
    assert np.linalg.norm(r[3] - expected.r) < 1e-8
    with pytest.raises(UnusableArc):
        Arc(r0[0], v0[0], 180.0, MU).states(times)
    with pytest.raises(UnusableArc):
        Arc(r0[0], v0[0], 180.0, MU).state(120.0)


_PARABOLA = (np.array([8000.0, 0.0, 0.0]), np.array([0.0, math.sqrt(2.0 * MU / 8000.0), 0.0]))
_ELLIPSE = (np.array([8000.0, 0.0, 0.0]), np.array([0.0, 0.9 * math.sqrt(2.0 * MU / 8000.0), 0.0]))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("state, t", [(_PARABOLA, 1e104), (_PARABOLA, -1e104),
                                      (_ELLIPSE, math.inf), (_ELLIPSE, -math.inf)],
                         ids=["parabola-1e104", "parabola-minus-1e104", "ellipse-inf",
                              "ellipse-minus-inf"])
def test_a_time_conic_coast_raises_on_makes_the_row_unusable(
    state: tuple[np.ndarray, np.ndarray], t: float
) -> None:
    """conic.coast raises OverflowError or ValueError here, which crashed the node."""
    r0, v0 = state
    with pytest.raises(DynamicsError):
        events.coast_one(r0, v0, t, MU)
    _, _, ok = coast_rows(np.tile(r0, (2, 1)), np.tile(v0, (2, 1)), np.array([60.0, t]), MU)
    assert ok.tolist() == [True, False]
    with pytest.raises(UnusableArc):
        Arc(r0, v0, 1800.0, MU).state(t)


def test_a_pair_holding_exactly_at_the_radius_stops_on_its_budget(circular: Orbit) -> None:
    p = circular(6878.0, 51.6, 10.0, 0.0)
    e = circular(6878.0, 51.6, 10.0, math.degrees(2.0 * math.asin(4.0 / (2.0 * 6878.0))))
    pursuer, evader = Arc(p[0], p[1], 1800.0, MU), Arc(e[0], e[1], 1800.0, MU)
    distance = float(np.linalg.norm(p[0] - e[0]))
    facts = capture_facts(pursuer, evader, distance, 60.0)
    assert facts.exhausted
    assert events.MAX_EVALUATIONS <= facts.evaluations <= events.MAX_EVALUATIONS + 1
    # Float noise proves a few slivers inside; the undecided rest hurts.
    assert sum(q.duration_s for q in facts.passages_hurt) > 1799.0


def test_the_secant_bound_is_used_where_the_tangents_are_weaker() -> None:
    # Out along w0 and back along w1: the tangent lines come within 5 km,
    # the chord stays 10 km away.
    rho0 = np.array([10.0, 0.0, 0.0])
    rho1 = np.array([10.0, 0.0, 0.0])
    lower, _ = segment_bounds(rho0, np.array([-1.0, 0.0, 0.0]), rho1, np.array([1.0, 0.0, 0.0]),
                              10.0, 1.0e-6, 1.0, MU)
    assert lower == pytest.approx(10.0 - 1.0e-6 * 100.0 / 8.0, abs=1e-12)


def test_the_gravity_gradient_bound_replaces_the_crude_one_when_tighter() -> None:
    rho = np.array([10.0, 0.0, 0.0])
    zero = np.zeros(3)
    a0 = 2.0 * MU / 7000.0**2
    lower, upper = segment_bounds(rho, zero, rho, zero, 60.0, a0, 7000.0, MU)
    rho_hi = 10.0 + a0 * 3600.0 / 2.0
    gradient = 2.0 * MU * rho_hi / (7000.0 - rho_hi / 2.0) ** 3
    assert gradient < a0 / 100.0
    assert lower == pytest.approx(10.0 - gradient * 3600.0 / 8.0)
    assert upper == pytest.approx(10.0 + gradient * 3600.0 / 8.0)


def test_a_radius_that_varies_in_time_is_met_inside_a_knot_segment(circular: Orbit) -> None:
    # 4 km apart all round; the radius crosses 4 km at 500 s, inside the
    # [480, 540] knot segment, so each test must take the piece's worse end.
    p = circular(6878.0, 51.6, 10.0, 0.0)
    e = circular(6878.0, 51.6, 10.0, math.degrees(2.0 * math.asin(4.0 / (2.0 * 6878.0))))
    pursuer, evader = Arc(p[0], p[1], 1800.0, MU), Arc(e[0], e[1], 1800.0, MU)

    def shrinking(t: float) -> float:
        return 5.0 - t / 500.0

    facts = capture_facts(pursuer, evader, shrinking, 60.0)
    (passage,) = facts.passages_favour
    assert passage.start_s == 0.0
    assert passage.end_s == pytest.approx(500.0, abs=0.5)
    (hurt,) = facts.passages_hurt
    assert hurt.end_s == pytest.approx(500.0, abs=0.5)
    assert facts.first_entry_s == 0.0 and not facts.inside_at_end

    def growing(t: float) -> float:
        return 3.5 + t / 1000.0

    facts = capture_facts(pursuer, evader, growing, 60.0)
    assert facts.first_entry_s == pytest.approx(500.0, abs=0.5)
    (hurt,) = facts.passages_hurt
    assert hurt.start_s == pytest.approx(500.0, abs=0.5)
    assert facts.inside_at_end and not facts.inside_at_start


def _dense_radius(arc: Arc, step: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, arc.duration_s + step / 2.0, step)
    return times, np.linalg.norm(arc.states(times)[0], axis=1)


@pytest.mark.parametrize("e", [0.0, 1e-9, 1e-6, 0.01, 0.3, 0.7, 0.95, 0.999, 1.0001, 1.5])
@pytest.mark.parametrize("nu_deg", [0.0, 100.0, 200.0, 300.0])
def test_radius_crossings_against_dense_truth_over_an_eccentricity_grid(
    e: float, nu_deg: float
) -> None:
    periapsis = 6500.0
    p = periapsis * (1.0 + e)
    nu = math.radians(nu_deg)
    if e >= 1.0:
        limit = 0.8 * math.acos(-1.0 / e)
        nu = math.atan2(math.sin(nu), math.cos(nu))
        nu = max(-limit, min(limit, nu))
    radius = p / (1.0 + e * math.cos(nu))
    r0 = np.array([radius * math.cos(nu), radius * math.sin(nu), 0.0])
    v0 = math.sqrt(MU / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
    arc = Arc(r0, v0, 20000.0, MU)
    _, radii = _dense_radius(arc, 0.5)
    threshold = periapsis + 0.25 * (min(radii.max(), 20000.0) - periapsis) + 1.0e-3
    passages = below_radius(arc, threshold)
    below = radii < threshold
    assert sum(q.duration_s for q in passages) == pytest.approx(below.sum() * 0.5, abs=1.5)
    entries = len(np.flatnonzero(below[1:] & ~below[:-1])) + int(below[0])
    assert len(passages) == entries
    for q in passages:
        assert abs(arc.radius(q.start_s) - threshold) < 1e-6 or q.start_s == 0.0
        assert abs(arc.radius(q.end_s) - threshold) < 1e-6 or q.end_s == arc.duration_s


def test_radius_crossings_match_an_independent_integrator() -> None:
    periapsis, e = 6500.0, 0.2
    a = periapsis / (1.0 - e)
    r0 = np.array([periapsis, 0.0, 0.0])
    v0 = np.array([0.0, math.sqrt(MU * (2.0 / periapsis - 1.0 / a)), 0.0])
    arc = Arc(r0, v0, 30000.0, MU)
    threshold = 7000.0

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return np.concatenate([y[3:], -MU * y[:3] / np.linalg.norm(y[:3]) ** 3])

    def crossing(_t: float, y: np.ndarray) -> float:
        return float(np.linalg.norm(y[:3])) - threshold

    solution = solve_ivp(rhs, (0.0, 30000.0), np.concatenate([r0, v0]), method="DOP853",
                         rtol=1e-12, atol=1e-9, events=crossing)
    truth = sorted(solution.t_events[0])
    crossings = [t for q in below_radius(arc, threshold) for t in (q.start_s, q.end_s)
                 if 0.0 < t < arc.duration_s]
    assert len(truth) == len(crossings) >= 8
    assert np.allclose(truth, crossings, atol=1e-3)


@pytest.mark.parametrize("one_minus_e", [1e-8, 1e-9, 3e-10, 1e-10])
@pytest.mark.parametrize("before_s", [903.0, 911.3, 1017.0])
def test_a_near_parabolic_dip_is_found_whatever_the_period(
    one_minus_e: float, before_s: float
) -> None:
    # The period is up to 1e18 s; times built as (first - P) + P lost the
    # periapsis to rounding and the dip below the floor with it.
    floor = 6388.1363
    periapsis = floor - 0.01
    speed = math.sqrt(MU * (2.0 - one_minus_e) / periapsis)
    r0, v0 = events.coast_one(np.array([periapsis, 0.0, 0.0]), np.array([0.0, 0.0, speed]),
                              -before_s, MU)
    arc = Arc(r0, v0, 1800.0, MU)
    (turn,) = arc.turning_times()
    assert turn == pytest.approx(before_s, abs=0.05)
    (passage,) = below_radius(arc, floor)
    assert passage.start_s < before_s < passage.end_s


def test_a_near_circle_treated_as_round_still_shows_its_dip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # e below the circular cut-off: no turning times, so only the closed-form
    # minimum shows the dip, and the sampling fallback must find it.
    radius, e = 7000.0, 5e-11
    a = radius / (1.0 - e)
    speed = math.sqrt(MU * (2.0 / radius - 1.0 / a))
    r0, v0 = events.coast_one(np.array([radius, 0.0, 0.0]), np.array([0.0, speed, 0.0]),
                              -900.0, MU)
    arc = Arc(r0, v0, 1800.0, MU)
    assert arc.turning_times() == []
    # Periapsis 7000 km at 900 s, ends about 1.5e-7 km higher.
    threshold = radius + 5e-8
    assert arc.minimum_radius() < threshold
    assert arc.radius(0.0) > threshold and arc.radius(1800.0) > threshold
    used: list[float] = []
    real = events._below_by_sampling

    def spy(arc_: Arc, radius_km: float, samples: int = 4001, *,
            refuse_none: bool = True) -> list[events.Passage]:
        used.append(radius_km)
        return real(arc_, radius_km, samples, refuse_none=refuse_none)

    monkeypatch.setattr(events, "_below_by_sampling", spy)
    (passage,) = below_radius(arc, threshold)
    assert used == [threshold]
    assert passage.start_s < 900.0 < passage.end_s


@pytest.mark.parametrize("e", np.linspace(0.999, 1.002, 13).tolist())
@pytest.mark.parametrize("duration", [60.0, 600.0, 3600.0, 86400.0])
def test_the_near_parabolic_sweep_coasts_and_agrees(e: float, duration: float) -> None:
    speed = math.sqrt(MU * (1.0 + e) / 6878.0)
    arc = Arc(np.array([6878.0, 0.0, 0.0]), np.array([0.0, speed, 0.0]), duration, MU)
    r, _ = arc.state(duration)
    reference = propagate_universal(StateVector((6878.0, 0.0, 0.0), (0.0, speed, 0.0)),
                                    duration, MU)
    assert np.linalg.norm(r - reference.r) < 1e-6 * max(1.0, float(np.linalg.norm(r)))
    assert below_radius(arc, 6378.1363) == []


def test_random_states_near_escape_coast_and_agree() -> None:
    rng = np.random.default_rng(20261103)
    for _ in range(200):
        radius = rng.uniform(6600.0, 9000.0)
        r = rng.normal(size=3)
        r *= radius / np.linalg.norm(r)
        v = rng.normal(size=3)
        v -= v.dot(r) / r.dot(r) * r
        v += r / radius * rng.normal() * 0.5
        v *= math.sqrt(MU * (1.0 + rng.uniform(0.9, 1.1)) / radius) / np.linalg.norm(v)
        t = float(rng.uniform(60.0, 86400.0))
        state, _ = Arc(r, v, t, MU).state(t)
        reference = propagate_universal(StateVector(tuple(r), tuple(v)), t, MU)
        assert np.linalg.norm(state - reference.r) < 1e-6 * max(1.0, float(np.linalg.norm(state)))


def _random_arc(rng: np.random.Generator, kind: str) -> Arc:
    periapsis = rng.uniform(6500.0, 8000.0)
    e = {"elliptic": rng.uniform(0.0, 0.8), "near-parabolic": rng.uniform(0.98, 1.02),
         "hyperbolic": rng.uniform(1.1, 3.0)}[kind]
    p = periapsis * (1.0 + e)
    nu = rng.uniform(-math.pi, math.pi)
    if e >= 1.0:
        nu = float(np.clip(nu, -0.7 * math.acos(-1.0 / e), 0.7 * math.acos(-1.0 / e)))
    radius = p / (1.0 + e * math.cos(nu))
    r = np.array([radius * math.cos(nu), radius * math.sin(nu), 0.0])
    v = math.sqrt(MU / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
    return Arc(r, v, float(rng.uniform(1800.0, 7200.0)), MU)


@pytest.mark.parametrize(
    "count", [24, pytest.param(256, marks=pytest.mark.slow)], ids=["fast", "all"]
)
def test_crossings_match_an_independent_integrator_on_random_arcs(count: int) -> None:
    rng = np.random.default_rng(20261104)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return np.concatenate([y[3:], -MU * y[:3] / np.linalg.norm(y[:3]) ** 3])

    checked = 0
    for k in range(count):
        arc = _random_arc(rng, ("elliptic", "near-parabolic", "hyperbolic")[k % 3])
        r_min = arc.minimum_radius()
        threshold = r_min + rng.uniform(0.001, 50.0)

        def crossing(_t: float, y: np.ndarray, level: float = threshold) -> float:
            return float(np.linalg.norm(y[:3])) - level

        # A short step, or the integrator's events step over a shallow dip.
        solution = solve_ivp(rhs, (0.0, arc.duration_s), np.concatenate([arc.r0, arc.v0]),
                             method="DOP853", rtol=1e-12, atol=1e-9, events=crossing,
                             max_step=2.0)
        truth = sorted(solution.t_events[0])
        found = [t for q in below_radius(arc, threshold) for t in (q.start_s, q.end_s)
                 if 0.0 < t < arc.duration_s]
        assert len(truth) == len(found), (k, truth, found)
        assert np.allclose(truth, found, atol=1e-3)
        checked += len(found)
    assert checked > count


@pytest.mark.parametrize("one_minus_e", [0.0, 1e-16, 1e-14, 1e-12, 1e-11])
@pytest.mark.parametrize("before_end_s", [20.0, 55.0, 80.0])
def test_a_periapsis_just_before_the_round_end_is_found_on_an_almost_parabola(
    one_minus_e: float, before_end_s: float
) -> None:
    # The core's closed-form periapsis time is off by up to 8e4 s here; a
    # periapsis it places after the round end hid a real Earth crossing.
    periapsis = 6378.1363 - 3.0
    speed = math.sqrt(MU * (2.0 - one_minus_e) / periapsis)
    t_peri = 1800.0 - before_end_s
    r0, v0 = events.coast_one(np.array([periapsis, 0.0, 0.0]), np.array([0.0, 0.0, speed]),
                              -t_peri, MU)
    arc = Arc(r0, v0, 1800.0, MU)
    (turn,) = arc.periapsis_times()
    assert turn == pytest.approx(t_peri, abs=1e-3)
    assert arc.minimum_radius() == pytest.approx(periapsis, abs=1e-6)
    (passage,) = below_radius(arc, 6378.1363)
    assert passage.start_s < t_peri < passage.end_s


def test_a_hyperbola_whose_periapsis_follows_the_round_has_no_turning_time() -> None:
    e, periapsis = 1.5, 7000.0
    speed = math.sqrt(MU * (1.0 + e) / periapsis)
    r0, v0 = events.coast_one(np.array([periapsis, 0.0, 0.0]), np.array([0.0, speed, 0.0]),
                              -1850.0, MU)
    arc = Arc(r0, v0, 1800.0, MU)
    assert arc.turning_times() == []
    assert arc.minimum_radius() == pytest.approx(arc.radius(1800.0))
    assert Arc(r0, v0, 1900.0, MU).turning_times() == pytest.approx([1850.0], abs=1e-3)


def test_a_near_circle_just_above_a_radius_is_not_refused() -> None:
    radius, e = 7000.0, 5e-11
    a = radius / (1.0 - e)
    speed = math.sqrt(MU * (2.0 / radius - 1.0 / a))
    # Starting at periapsis and staying near it: the closed form puts the
    # periapsis at t = 0, the arc never goes below its start.
    arc = Arc(np.array([radius, 0.0, 0.0]), np.array([0.0, speed, 0.0]), 60.0, MU)
    level = min(arc.radius(0.0), arc.radius(60.0)) - 1e-7
    assert arc.minimum_radius() < level
    assert below_radius(arc, level) == []


def _narrow_dip(width_s: float) -> tuple[Arc, float]:
    """An eccentric arc and a radius it stays under for about `width_s` mid-round."""
    periapsis, e = 7000.0, 0.1
    speed = math.sqrt(MU * (1.0 + e) / periapsis)
    r0, v0 = events.coast_one(np.array([periapsis, 0.0, 0.0]), np.array([0.0, speed, 0.0]),
                              -911.0, MU)
    curvature = speed**2 / periapsis - MU / periapsis**2
    return Arc(r0, v0, 1800.0, MU), periapsis + 0.5 * curvature * (width_s / 2.0) ** 2


@pytest.mark.parametrize("width_s", [5.0, 0.2])
def test_the_sampling_fallback_finds_a_narrow_dip(width_s: float) -> None:
    # 5 s falls between 41 samples but not 4001; 0.2 s only the minimum finds.
    arc, level = _narrow_dip(width_s)
    (passage,) = events._below_by_sampling(arc, level)
    assert passage.duration_s == pytest.approx(width_s, rel=0.05)
    assert passage.start_s < 911.0 < passage.end_s
    assert below_radius(arc, level) == [passage] or below_radius(arc, level)[0].duration_s \
        == pytest.approx(passage.duration_s, abs=1e-6)


def test_the_sampling_fallback_keeps_a_passage_open_at_the_round_end() -> None:
    periapsis, e = 7000.0, 0.1
    speed = math.sqrt(MU * (1.0 + e) / periapsis)
    r0, v0 = events.coast_one(np.array([periapsis, 0.0, 0.0]), np.array([0.0, speed, 0.0]),
                              -1790.0, MU)
    arc = Arc(r0, v0, 1800.0, MU)
    level = arc.radius(1790.0) + 1.0
    passages = events._below_by_sampling(arc, level)
    assert passages[-1].end_s == 1800.0


def test_the_sampling_fallback_finds_every_dip_not_only_the_deepest() -> None:
    # Two periapses in one long round, each under the radius for a few
    # seconds: the minimiser refines only the deepest sample, so the other
    # dip must come from the grid.
    periapsis, e = 7000.0, 0.01
    speed = math.sqrt(MU * (1.0 + e) / periapsis)
    a = periapsis / (1.0 - e)
    period = 2.0 * math.pi * math.sqrt(a**3 / MU)
    r0, v0 = events.coast_one(np.array([periapsis, 0.0, 0.0]), np.array([0.0, speed, 0.0]),
                              -600.0, MU)
    arc = Arc(r0, v0, 600.0 + 1.5 * period, MU)
    curvature = speed**2 / periapsis - MU / periapsis**2
    level = periapsis + 0.5 * curvature * 4.0**2  # about 8 s below each time
    passages = events._below_by_sampling(arc, level)
    assert len(passages) == 2
    assert passages[1].start_s - passages[0].start_s == pytest.approx(period, abs=1.0)


def test_a_near_parabolic_arc_still_inbound_at_the_round_end_has_no_periapsis() -> None:
    for one_minus_e in (1e-9, 0.0, -1e-9):
        periapsis = 7000.0
        speed = math.sqrt(MU * (2.0 - one_minus_e) / periapsis)
        r0, v0 = events.coast_one(np.array([periapsis, 0.0, 0.0]),
                                  np.array([0.0, speed, 0.0]), -2000.0, MU)
        arc = Arc(r0, v0, 1800.0, MU)
        assert arc.periapsis_times() == []
        assert arc.minimum_radius() == pytest.approx(arc.radius(1800.0))
        assert below_radius(arc, 7000.5) == [] or below_radius(arc, 7000.5)[0].end_s == 1800.0


def test_the_first_entry_carries_the_undecided_time_before_it(circular: Orbit) -> None:
    pursuer, evader = _crossing(circular)
    facts = capture_facts(pursuer, evader, 3.0, 60.0)
    assert 0.0 <= facts.first_entry_uncertainty_s <= 1e-5


def test_an_arc_whose_minimum_is_below_but_no_crossing_is_found_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A lost turning time: the exact minimum says the arc goes below and
    # neither the brackets nor the sampling find it. Fail closed.
    arc = Arc(np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.6, 0.0]), 600.0, MU)
    monkeypatch.setattr(Arc, "minimum_radius", lambda self: 6000.0)
    with pytest.raises(UnusableArc, match="no crossing was bracketed"):
        below_radius(arc, 6500.0)


def test_a_real_graze_resolves_well_inside_the_budget(circular: Orbit) -> None:
    pursuer, evader = _crossing(circular)
    facts = capture_facts(pursuer, evader, 3.0, 60.0)
    assert not facts.exhausted and facts.evaluations < 1000


def test_a_radial_arc_is_refused_rather_than_judged() -> None:
    # r x v = 0: there is no conic shape, and the arc falls through the centre.
    r0 = np.array([12345.0, 6789.0, -4321.0])
    for v0 in (np.zeros(3), -r0 / np.linalg.norm(r0)):
        with pytest.raises(UnusableArc, match="radial"):
            below_radius(Arc(r0, v0, 6000.0, MU), 6378.1363)


@pytest.mark.parametrize("transverse_km_s", [1e-12, 1e-9, 1e-7])
def test_a_near_radial_fall_crosses_the_earth(transverse_km_s: float) -> None:
    # Released almost at rest: a degenerate ellipse, 1 - e about 1e-24, whose
    # periapsis is near the centre half a period (about 3100 s) later. The
    # closed-form periapsis time is useless there.
    r0 = np.array([12345.0, 6789.0, -4321.0])
    radius = float(np.linalg.norm(r0))
    side = np.cross(r0, [0.0, 0.0, 1.0])
    v0 = side / np.linalg.norm(side) * transverse_km_s
    arc = Arc(r0, v0, 6000.0, MU)
    (periapsis,) = arc.periapsis_times()
    half_period = math.pi * math.sqrt((radius / 2.0) ** 3 / MU)
    assert periapsis == pytest.approx(half_period, abs=1.0)
    assert arc.minimum_radius() < 6378.1363
    (passage,) = below_radius(arc, 6378.1363)
    assert passage.start_s < periapsis < passage.end_s


def _fall(transverse_km_s: float = 1e-12) -> Arc:
    r0 = np.array([12345.0, 6789.0, -4321.0])
    side = np.cross(r0, [0.0, 0.0, 1.0])
    return Arc(r0, side / np.linalg.norm(side) * transverse_km_s, 6000.0, MU)


def _fall_half_period() -> float:
    radius = float(np.linalg.norm([12345.0, 6789.0, -4321.0]))
    return math.pi * math.sqrt((radius / 2.0) ** 3 / MU)


@pytest.mark.parametrize("scalar_at_start", [-1e-20, -1e-30, 0.0])
def test_a_start_the_batch_and_scalar_coasts_round_apart(
    scalar_at_start: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What CI hit on [0, 60]: released at rest, r0.v0 cancels to +-6e-9, the batch
    read exactly 0 and a fused multiply-add -4e-25, and brentq raised."""
    original = Arc._radial
    monkeypatch.setattr(Arc, "_radial",
                        lambda self, t: scalar_at_start if t == 0.0 else original(self, t))
    arc = _fall()
    (periapsis,) = arc.periapsis_times()
    assert periapsis == pytest.approx(_fall_half_period(), abs=1.0)
    assert arc.turning_times() == [periapsis]


def test_a_disagreement_beyond_rounding_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch row wrong by far more than rounding: a solver fault, not a turning point."""
    original = Arc.states

    def flipped(self: Arc, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r, v = original(self, times)
        v = v.copy()
        hit = np.flatnonzero(times == 600.0)
        v[hit] = -v[hit]
        return r, v

    monkeypatch.setattr(Arc, "states", flipped)
    with pytest.raises(UnusableArc, match="more than rounding"):
        _fall().periapsis_times()


def test_every_turning_point_of_a_multi_revolution_near_radial_ellipse() -> None:
    # 1 - e = 3e-7, flown for 3.3 periods: each periapsis dips under the
    # Earth, and each must be found from the r.v grid.
    one_minus_e, apoapsis = 3e-7, 20000.0
    a = apoapsis / (2.0 - one_minus_e)
    speed = math.sqrt(MU * (2.0 / apoapsis - 1.0 / a))
    period = 2.0 * math.pi * math.sqrt(a**3 / MU)
    arc = Arc(np.array([apoapsis, 0.0, 0.0]), np.array([0.0, speed, 0.0]), 3.3 * period, MU)
    periapses = arc.periapsis_times()
    assert periapses == pytest.approx([0.5 * period, 1.5 * period, 2.5 * period], abs=1e-3)
    turning = arc.turning_times()
    assert turning == pytest.approx([k * 0.5 * period for k in range(1, 7)], abs=1e-3)
    passages = below_radius(arc, 6378.1363)
    assert len(passages) == 3
    for passage, periapsis in zip(passages, periapses, strict=True):
        assert passage.start_s < periapsis < passage.end_s


def test_a_start_exactly_at_apoapsis_is_not_a_turning_point() -> None:
    one_minus_e, apoapsis = 3e-7, 20000.0
    a = apoapsis / (2.0 - one_minus_e)
    speed = math.sqrt(MU * (2.0 / apoapsis - 1.0 / a))
    arc = Arc(np.array([apoapsis, 0.0, 0.0]), np.array([0.0, speed, 0.0]), 600.0, MU)
    assert float(arc.r0 @ arc.v0) == 0.0
    assert arc.turning_times() == []


@pytest.mark.parametrize(("ratio", "refused"), [(5e-10, True), (2e-9, False)])
def test_the_radial_cut_is_at_1e9_of_r_v(ratio: float, refused: bool) -> None:
    r0 = np.array([12000.0, 0.0, 0.0])
    v0 = np.array([-2.0, 2.0 * ratio, 0.0])  # |r x v| / (|r||v|) = ratio
    arc = Arc(r0, v0, 600.0, MU)
    if refused:
        with pytest.raises(UnusableArc, match="radial"):
            arc.turning_times()
    else:
        assert arc.turning_times() == []


def test_a_pair_reaching_the_centre_is_refused_by_the_detector(
    monkeypatch: pytest.MonkeyPatch, circular: Orbit
) -> None:
    pursuer, evader = _crossing(circular)
    monkeypatch.setattr(Arc, "minimum_radius", lambda self: 0.0)
    with pytest.raises(UnusableArc, match="centre"):
        capture_facts(pursuer, evader, 3.0, 60.0)


def test_an_arc_replaced_or_with_its_inputs_changed_does_not_reuse_stale_turning_times(
) -> None:
    import dataclasses

    one_minus_e, apoapsis = 3e-7, 20000.0
    a = apoapsis / (2.0 - one_minus_e)
    speed = math.sqrt(MU * (2.0 / apoapsis - 1.0 / a))
    period = 2.0 * math.pi * math.sqrt(a**3 / MU)
    r0 = np.array([apoapsis, 0.0, 0.0])
    v0 = np.array([0.0, speed, 0.0])
    short = Arc(r0, v0, 0.4 * period, MU)
    assert short.turning_times() == []
    longer = dataclasses.replace(short, duration_s=2.2 * period)
    assert len(longer.turning_times()) == 4
    r0[0] = 30000.0  # the caller's array: the arc kept its own copy
    assert short.r0[0] == apoapsis
    with pytest.raises(ValueError):
        short.r0[0] = 1.0


def test_three_entries_inside_one_knot_segment_are_each_found(circular: Orbit) -> None:
    # A single 7000 s knot segment holds all three passes of the two planes
    # (half an orbit apart); none is skipped and the first entry is first.
    pursuer, evader = _crossing(circular, duration=7000.0)
    times, distance = _dense(pursuer, evader, step=0.05)
    inside = distance <= 3.0
    entries = np.flatnonzero(inside[1:] & ~inside[:-1]) + 1
    assert len(entries) == 3
    facts = capture_facts(pursuer, evader, 3.0, 7000.0)
    assert facts.first_entry_s == pytest.approx(times[entries[0]], abs=0.06)
    assert [q.start_s for q in facts.passages_favour] == pytest.approx(times[entries].tolist(),
                                                                         abs=0.06)


def test_a_periapsis_touching_the_radius_is_a_tangency_not_a_refusal() -> None:
    periapsis, e = 6378.1363, 0.01
    speed = math.sqrt(MU * (1.0 + e) / periapsis)
    r0, v0 = events.coast_one(np.array([periapsis, 0.0, 0.0]), np.array([0.0, 0.0, speed]),
                              -900.0, MU)
    arc = Arc(r0, v0, 1800.0, MU)
    passages = below_radius(arc, periapsis)
    assert all(q.duration_s < 1e-3 for q in passages)
