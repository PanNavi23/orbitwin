"""Review regressions for conic_batch.pairwise_screen: its remainder against the true
relative acceleration, and its bound against a DOP853 reference on wide segments."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from orbitwin import conic, conic_batch

MU = 398600.4418


def _floor(r: np.ndarray, v: np.ndarray, span: float) -> float:
    end = conic.coast(r, v, span, MU)[0]
    return conic.arc_minimum_radius(r, v, span, end, MU)[0]


def _segment(rp: np.ndarray, vp: np.ndarray, re: np.ndarray, ve: np.ndarray, h: float
             ) -> tuple[float, float]:
    """The screen's LB on one segment [0, h], and the A it subtracted (A h^2 / 8)."""
    knots = np.array([0.0, h])
    p = conic_batch.propagate_batch(rp[None], vp[None], knots, MU)
    e = conic_batch.propagate_batch(re[None], ve[None], knots, MU)
    lb = float(conic_batch.pairwise_screen(
        p.r_km, p.v_km_s, e.r_km, e.v_km_s, knots,
        [_floor(rp, vp, h)], [_floor(re, ve, h)], MU,
    ).lower_bound_km[0, 0])
    rho, w = p.r_km[0] - e.r_km[0], p.v_km_s[0] - e.v_km_s[0]

    def nearest(a: np.ndarray, d: np.ndarray) -> float:
        s = float(np.clip(-(a @ d) / (d @ d), 0.0, 1.0)) if d @ d > 0.0 else 0.0
        return float(np.linalg.norm(a + s * d))

    linear = max(min(nearest(rho[0], w[0] * h / 2), nearest(rho[1], -w[1] * h / 2)),
                 nearest(rho[0], rho[1] - rho[0]))
    return lb, 8.0 * (linear - lb) / (h * h)


def _same_velocity(radius: float, offset: np.ndarray) -> tuple[np.ndarray, ...]:
    """A circular pursuer and an evader `offset` km from it with the same velocity."""
    rp, vp = np.array([radius, 0.0, 0.0]), np.array([0.0, math.sqrt(MU / radius), 0.0])
    return rp, vp, rp + offset, vp.copy()


#: (pursuer radius, offset of the evader, segment length). The first twelve are a
#: radial offset that tidal separation grows faster than |rho0| + |w0| h; then an
#: evader 3000 km above, where only min(floors) bounds the gravity gradient; then
#: one antipodal, where |rho''| is both gravities added.
ACCELERATION_CASES = [
    *[(radius, (offset, 0.0, 0.0), h) for radius in (6400.0, 6700.0)
      for offset in (1.0, 10.0, 100.0) for h in (300.0, 1000.0)],
    (6400.0, (3000.0, 0.0, 0.0), 60.0),
    (6400.0, (-12800.0, 0.0, 0.0), 60.0),
]


@pytest.mark.parametrize("radius, offset, h", ACCELERATION_CASES)
def test_the_remainder_bounds_the_relative_acceleration(
    radius: float, offset: tuple[float, float, float], h: float
) -> None:
    """The proof needs A >= |rho''| over the whole segment, and A is observable as
    8 (max(Taylor, secant) - LB) / h^2. Fails with rho_hi missing A0 h^2/2 (A is
    0.53 to 0.93 of |rho''| on the radial cases), with r_lo the larger floor
    (0.95 on the 3000 km case) and with A0 missing one gravity (0.5, antipodal)."""
    rp, vp, re, ve = _same_velocity(radius, np.array(offset))
    _, accel = _segment(rp, vp, re, ve, h)
    times = np.linspace(0.0, h, 4001)
    p = conic.coast_many(np.tile(rp, (len(times), 1)), np.tile(vp, (len(times), 1)), times, MU)[0]
    e = conic.coast_many(np.tile(re, (len(times), 1)), np.tile(ve, (len(times), 1)), times, MU)[0]

    def g(r: np.ndarray) -> np.ndarray:
        return -MU * r / np.linalg.norm(r, axis=1, keepdims=True) ** 3

    largest = float(np.max(np.linalg.norm(g(p) - g(e), axis=1)))
    assert accel >= largest * (1.0 - 1.0e-9)  # A is recovered through a subtraction
    # And never more than the crude bound, which always holds.
    crude = MU / _floor(rp, vp, h) ** 2 + MU / _floor(re, ve, h) ** 2
    assert accel <= crude * (1.0 + 1.0e-12)


def _dop853_minimum(rp: np.ndarray, vp: np.ndarray, re: np.ndarray, ve: np.ndarray,
                    h: float) -> float:
    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        a, b = y[0:3], y[6:9]
        return np.concatenate([y[3:6], -MU * a / np.linalg.norm(a) ** 3,
                               y[9:12], -MU * b / np.linalg.norm(b) ** 3])

    done = solve_ivp(rhs, (0.0, h), np.concatenate([rp, vp, re, ve]), method="DOP853",
                     rtol=3.0e-14, atol=1.0e-12, dense_output=True)
    assert done.success
    times = np.linspace(0.0, h, 200001)
    y = done.sol(times)
    gap = np.linalg.norm(y[0:3] - y[6:9], axis=0)
    k = int(np.argmin(gap))

    def at(s: float) -> float:
        z = done.sol(s)
        return float(np.linalg.norm(z[0:3] - z[6:9]))

    lo, hi = times[max(k - 1, 0)], times[min(k + 1, len(times) - 1)]
    found = minimize_scalar(at, bounds=(lo, hi), method="bounded", options={"xatol": 1.0e-10})
    return min(float(found.fun), float(gap[k]))


def _pass(rperi: float, ecc: float, nu0: float, meet: float, miss: np.ndarray,
          kick: np.ndarray) -> tuple[np.ndarray, ...]:
    p = rperi * (1.0 + ecc)
    radius, speed = p / (1.0 + ecc * math.cos(nu0)), math.sqrt(MU / p)
    rp = np.array([radius * math.cos(nu0), radius * math.sin(nu0), 0.0])
    vp = np.array([-speed * math.sin(nu0), speed * (ecc + math.cos(nu0)), 0.0])
    r_meet, v_meet = conic.coast(rp, vp, meet, MU)
    re, ve = conic.coast(r_meet + miss, v_meet + kick, -meet, MU)
    return rp, vp, re, ve


#: Perigee at 6400 km inside the span where the arc has one; every pass meets mid-span.
CASES = {
    "circular, radial offset, slow": (6400.0, 0.0, 0.0, 600.0, [3.0, 0.0, 0.0], [0.0, 0.001, 0.0]),
    "near-parabolic through perigee, 11 km/s across": (
        6400.0, 1.0 - 1.0e-7, -0.9, 590.0, [0.0, 0.0, 2.0], [0.0, 0.0, 11.0]),
    "hyperbolic e = 1.5 against a crossing pass": (
        6400.0, 1.5, -0.9, 610.0, [0.5, -0.5, 1.0], [3.0, -2.0, 6.0]),
    "hyperbolic, the other near-parabolic and co-moving": (
        6400.0, 1.2, -0.9, 580.0, [2.0, 1.0, 0.0], [0.0, -0.4, 0.05]),
    "eccentric pursuer through perigee, slow evader": (
        6400.0, 0.7, -0.6, 620.0, [1.0, 1.0, 1.0], [0.05, -0.02, 0.01]),
}


@pytest.mark.parametrize("step", [30.0, 60.0, 120.0])
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_bound_holds_through_perigee_against_dop853(name: str, step: float) -> None:
    span = 1200.0
    rp, vp, re, ve = _pass(*CASES[name][:4], np.array(CASES[name][4]), np.array(CASES[name][5]))
    knots = np.arange(0.0, span + step / 2, step)
    p = conic_batch.propagate_batch(rp[None], vp[None], knots, MU)
    e = conic_batch.propagate_batch(re[None], ve[None], knots, MU)
    lb = float(conic_batch.pairwise_screen(
        p.r_km, p.v_km_s, e.r_km, e.v_km_s, knots,
        [_floor(rp, vp, span)], [_floor(re, ve, span)], MU,
    ).lower_bound_km[0, 0])
    assert lb <= _dop853_minimum(rp, vp, re, ve, span)


def _docstring_bound(rho0, rho1, w0, w1, h, floor_p, floor_e):  # type: ignore[no-untyped-def]
    """pairwise_screen's LB on one segment, from its docstring, in plain floats."""

    def nearest(a, d):  # type: ignore[no-untyped-def]
        dd = float(d @ d)
        s = min(1.0, max(0.0, -float(a @ d) / dd)) if dd > 0.0 else 0.0
        return float(np.linalg.norm(a + s * d))

    a0 = MU / floor_p**2 + MU / floor_e**2
    rho_hi = float(np.linalg.norm(rho0)) + float(np.linalg.norm(w0)) * h + a0 * h * h / 2.0
    low = min(floor_p, floor_e)
    accel = min(a0, 2.0 * MU * rho_hi / (low - rho_hi / 2.0) ** 3) if low > rho_hi / 2.0 else a0
    taylor = min(nearest(rho0, w0 * h / 2.0), nearest(rho1, -w1 * h / 2.0))
    secant = nearest(rho0, rho1 - rho0)
    return max(taylor, secant) - accel * h * h / 8.0


def test_the_screen_is_its_docstring() -> None:
    """Every term as written, on random pairs; catches a term that only loosens."""
    generator = np.random.default_rng(20261102)
    knots = np.linspace(0.0, 1800.0, 13)
    radius = generator.uniform(6600.0, 9000.0, 8)
    angle = generator.uniform(0.0, 2.0 * math.pi, 8)
    r0 = np.c_[radius * np.cos(angle), radius * np.sin(angle), generator.normal(0.0, 300.0, 8)]
    speed = np.sqrt(MU / radius) * generator.uniform(0.9, 1.1, 8)
    v0 = np.c_[-speed * np.sin(angle), speed * np.cos(angle), generator.normal(0.0, 0.3, 8)]
    flown = conic_batch.propagate_batch(r0, v0, knots, MU)
    floors = np.array([_floor(r0[i], v0[i], knots[-1]) for i in range(8)])
    p, e = slice(0, 4), slice(4, 8)
    got = conic_batch.pairwise_screen(
        flown.r_km[p], flown.v_km_s[p], flown.r_km[e], flown.v_km_s[e], knots,
        floors[p], floors[e], MU,
    ).lower_bound_km
    for i in range(4):
        for j in range(4):
            rho = flown.r_km[i] - flown.r_km[4 + j]
            w = flown.v_km_s[i] - flown.v_km_s[4 + j]
            want = min(
                _docstring_bound(rho[k], rho[k + 1], w[k], w[k + 1], knots[k + 1] - knots[k],
                                 floors[i], floors[4 + j])
                for k in range(len(knots) - 1)
            )
            assert got[i, j] == pytest.approx(want, rel=1e-12, abs=1e-9)


def test_every_segment_of_close_pairs_is_its_docstring() -> None:
    """Segment by segment on close pairs, so the secant and A1 branches win too."""
    from test_conic_batch import _conjunctions, _encounters

    def nearest(a: np.ndarray, d: np.ndarray) -> float:
        dd = float(d @ d)
        s = min(1.0, max(0.0, -float(a @ d) / dd)) if dd > 0.0 else 0.0
        return float(np.linalg.norm(a + s * d))

    generator = np.random.default_rng(7)
    pairs = [
        *zip(*_conjunctions(generator, 12, 120.0), strict=True),
        *zip(*_encounters(generator, 12)[:4], strict=True),
    ]
    knots = np.linspace(0.0, 120.0, 3)
    wins = {"secant": 0, "taylor": 0, "A1": 0, "A0": 0}
    for rp, vp, re, ve in pairs:
        p = conic_batch.propagate_batch(rp[None], vp[None], knots, MU)
        e = conic_batch.propagate_batch(re[None], ve[None], knots, MU)
        fp, fe = _floor(rp, vp, knots[-1]), _floor(re, ve, knots[-1])
        rho, w = p.r_km[0] - e.r_km[0], p.v_km_s[0] - e.v_km_s[0]
        for k in range(len(knots) - 1):
            h = knots[k + 1] - knots[k]
            got = conic_batch.pairwise_screen(
                p.r_km[:, k:k + 2], p.v_km_s[:, k:k + 2], e.r_km[:, k:k + 2],
                e.v_km_s[:, k:k + 2], knots[k:k + 2], [fp], [fe], MU,
            ).lower_bound_km[0, 0]
            want = _docstring_bound(rho[k], rho[k + 1], w[k], w[k + 1], h, fp, fe)
            assert got == pytest.approx(want, rel=1e-12, abs=1e-9)
            taylor = min(nearest(rho[k], w[k] * h / 2), nearest(rho[k + 1], -w[k + 1] * h / 2))
            wins["secant" if nearest(rho[k], rho[k + 1] - rho[k]) > taylor else "taylor"] += 1
            a0 = MU / fp**2 + MU / fe**2
            hi = np.linalg.norm(rho[k]) + np.linalg.norm(w[k]) * h + a0 * h * h / 2
            low = min(fp, fe)
            wins["A1" if low > hi / 2 and 2 * MU * hi / (low - hi / 2) ** 3 < a0 else "A0"] += 1
    assert min(wins.values()) > 0, wins
