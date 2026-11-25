"""Two-body coasts and conic facts, against propagate_universal and DOP853."""

from __future__ import annotations

import math

import numpy as np
import pytest

from orbitwin import conic
from orbitwin.models import StateVector, as_vector3
from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, propagate_universal

MU = 398600.4418
ELEMENTS = [
    KeplerianElements(7000.0, 0.01, 28.0, 15.0, 0.0, 300.0),
    KeplerianElements(9000.0, 0.3, 43.0, 65.0, 10.0, 170.0),
    KeplerianElements(7000.0 / (1.0 - 1.5), 1.5, 50.0, 75.0, 0.0, -30.0),
]


@pytest.mark.parametrize("elements", ELEMENTS)
@pytest.mark.parametrize("dt_s", [0.5, 100.0, 1234.5, -800.0, 20000.0])
def test_coast_agrees_with_propagate_universal(elements: KeplerianElements, dt_s: float) -> None:
    state = keplerian_to_cartesian(elements, MU)
    want = propagate_universal(state, dt_s, MU)
    r, v = conic.coast(state.r, state.v, dt_s, MU)
    rows_r, rows_v = conic.coast_many(state.r[None], state.v[None], np.array([dt_s]), MU)
    assert np.linalg.norm(r - want.r) < 1.0e-8
    assert np.linalg.norm(v - want.v) < 1.0e-11
    assert np.linalg.norm(rows_r[0] - want.r) < 1.0e-8
    assert np.linalg.norm(rows_v[0] - want.v) < 1.0e-11


def test_coast_converges_near_a_parabola() -> None:
    """1 - e within 1e-3 of zero: 42% of these states did not converge, one row raised
    for the whole of coast_many, and propagate_universal was off by up to 15 m."""
    from scipy.integrate import solve_ivp

    generator = np.random.default_rng(20261101)
    n = 2000
    ecc = 1.0 + generator.choice([-1.0, 1.0], n) * 10.0 ** generator.uniform(-10.0, -3.0, n)
    p = generator.uniform(6600.0, 15000.0, n) * (1.0 + ecc)
    nu = generator.uniform(-0.9, 0.9, n)
    tilt = generator.uniform(-math.pi, math.pi, n)
    radius = p / (1.0 + ecc * np.cos(nu))
    speed = np.sqrt(MU / p)
    r = np.c_[radius * np.cos(nu), radius * np.sin(nu) * np.cos(tilt), radius * np.sin(nu) * np.sin(tilt)]
    v = np.c_[-speed * np.sin(nu), speed * (ecc + np.cos(nu)) * np.cos(tilt),
              speed * (ecc + np.cos(nu)) * np.sin(tilt)]
    dt = generator.uniform(-10000.0, 10000.0, n)

    rows_r, _ = conic.coast_many(r, v, dt, MU)
    for i in range(n):
        got_r, _ = conic.coast(r[i], v[i], float(dt[i]), MU)
        assert np.linalg.norm(got_r - rows_r[i]) < 1.0e-8
    for i in range(0, n, 100):
        flown = solve_ivp(
            lambda _, y: np.r_[y[3:], -MU * y[:3] / np.linalg.norm(y[:3]) ** 3],
            (0.0, float(dt[i])), np.r_[r[i], v[i]], method="DOP853", rtol=3e-14, atol=1e-13,
        )
        assert np.linalg.norm(rows_r[i] - flown.y[:3, -1]) < 1.0e-7
        universal = propagate_universal(StateVector(r[i], v[i]), float(dt[i]), MU)
        assert np.linalg.norm(universal.r - flown.y[:3, -1]) < 1.0e-7


@pytest.mark.parametrize("elements", ELEMENTS)
def test_the_closed_form_radius_minimum_is_the_sampled_one(elements: KeplerianElements) -> None:
    state = keplerian_to_cartesian(elements, MU)
    generator = np.random.default_rng(7)
    for _ in range(25):
        offset = float(generator.uniform(-3000.0, 3000.0))
        start = propagate_universal(state, offset, MU)
        duration = float(generator.uniform(10.0, 9000.0))
        end = propagate_universal(start, duration, MU)
        got, _ = conic.arc_minimum_radius(start.r, start.v, duration, end.r, MU)
        times = np.linspace(0.0, duration, 4001)
        rows, _ = conic.coast_many(
            np.tile(start.r, (len(times), 1)), np.tile(start.v, (len(times), 1)), times, MU
        )
        sampled = float(np.min(np.linalg.norm(rows, axis=1)))
        # The closed form is never above a sample, and sampling every
        # duration/4000 s gets within metres of it.
        assert got <= sampled + 1.0e-9
        assert sampled - got < 0.05


@pytest.mark.parametrize("eccentricity", [1.0 - 1.0e-9, 1.0, 1.0 + 1.0e-9, 0.999, 1.001])
def test_the_wait_for_periapsis_holds_near_a_parabola(eccentricity: float) -> None:
    periapsis = 7000.0
    p = periapsis * (1.0 + eccentricity)
    nu = math.radians(-60.0)
    radius = p / (1.0 + eccentricity * math.cos(nu))
    speed = math.sqrt(MU / p)
    state = StateVector(
        as_vector3([radius * math.cos(nu), radius * math.sin(nu), 0.0]),
        as_vector3([-speed * math.sin(nu), speed * (eccentricity + math.cos(nu)), 0.0]),
    )
    wait = conic.seconds_to_periapsis(state.r, state.v, MU)
    assert wait is not None and wait > 0.0
    there = propagate_universal(state, wait, MU)
    assert float(np.linalg.norm(there.r)) == pytest.approx(periapsis, abs=1.0e-3)
    assert conic.periapsis_radius_km(state.r, state.v, MU) == pytest.approx(periapsis, rel=1e-9)
