"""Two-body coasts and the closed-form facts of a conic.

Nothing here knows a rulebook: it propagates, finds periapsis and bounds the
radius over an arc. Callers decide what those facts mean; the ranking search
is the only one so far, and the duel engine will be the second.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import DynamicsError
from .orbit import stumpff_c, stumpff_c_array, stumpff_s, stumpff_s_array

_LAGUERRE_ORDER = 5


def _initial_chi(dt: float, r0: float, vr0: float, alpha: float, mu: float) -> float:
    sqrt_mu = math.sqrt(mu)
    if alpha > 1.0e-12:
        return sqrt_mu * dt * alpha
    if alpha < -1.0e-12:
        sign = 1.0 if dt >= 0.0 else -1.0
        denominator = vr0 + sign * math.sqrt(-mu / alpha) * (1.0 - r0 * alpha)
        numerator = -2.0 * mu * alpha * dt
        if abs(denominator) > 1.0e-12 and numerator / denominator > 0.0:
            return sign * math.sqrt(-1.0 / alpha) * math.log(numerator / denominator)
        return sign * math.sqrt(-1.0 / alpha)
    return sqrt_mu * dt / r0


def coast(r: np.ndarray, v: np.ndarray, dt: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-body coast by universal variables, Laguerre-Conway iteration.

    It agrees with `orbit.propagate_universal` to 1e-8 km, and tests pin the
    two together. It takes 6.8 us a call against 32.4 us (2026-11-24), nearly
    all of the gap in propagate_universal's numpy 3-vector operations (25 us),
    which this does in Python floats. Negative `dt` is allowed, since SLSQP may
    step outside the linear constraints.
    """
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    if dt == 0.0:
        return np.array([x, y, z]), np.array([vx, vy, vz])
    r0 = math.sqrt(x * x + y * y + z * z)
    v2 = vx * vx + vy * vy + vz * vz
    rv = x * vx + y * vy + z * vz
    vr0 = rv / r0
    alpha = 2.0 / r0 - v2 / mu
    sqrt_mu = math.sqrt(mu)
    if alpha > 1.0e-12:
        # Whole revolutions change nothing and only slow the iteration down.
        period = 2.0 * math.pi / (sqrt_mu * alpha**1.5)
        if abs(dt) > period:
            dt = math.fmod(dt, period)
    chi = _initial_chi(dt, r0, vr0, alpha, mu)
    a1 = r0 * vr0 / sqrt_mu
    a2 = 1.0 - alpha * r0
    n = _LAGUERRE_ORDER
    for _ in range(80):
        zz = alpha * chi * chi
        c = stumpff_c(zz)
        s = stumpff_s(zz)
        f = a1 * chi * chi * c + a2 * chi**3 * s + r0 * chi - sqrt_mu * dt
        df = a1 * chi * (1.0 - zz * s) + a2 * chi * chi * c + r0
        d2f = a1 * (1.0 - zz * c) + a2 * chi * (1.0 - zz * s)
        root = math.sqrt(abs((n - 1) ** 2 * df * df - n * (n - 1) * f * d2f))
        step = n * f / (df + math.copysign(root, df))
        chi -= step
        if abs(step) <= 1.0e-13 * max(1.0, abs(chi)):
            break
    else:
        raise DynamicsError("universal-variable coast did not converge")
    zz = alpha * chi * chi
    c = stumpff_c(zz)
    s = stumpff_s(zz)
    f_lag = 1.0 - chi * chi / r0 * c
    g_lag = dt - chi**3 / sqrt_mu * s
    rx, ry, rz = f_lag * x + g_lag * vx, f_lag * y + g_lag * vy, f_lag * z + g_lag * vz
    rmag = math.sqrt(rx * rx + ry * ry + rz * rz)
    fdot = sqrt_mu / (rmag * r0) * (alpha * chi**3 * s - chi)
    gdot = 1.0 - chi * chi / rmag * c
    return (
        np.array([rx, ry, rz]),
        np.array([fdot * x + gdot * vx, fdot * y + gdot * vy, fdot * z + gdot * vz]),
    )


def coast_many(
    r0: np.ndarray, v0: np.ndarray, dt: np.ndarray, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """`coast` over rows: r0 and v0 are (M, 3), dt is (M,)."""
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    dt = np.array(dt, dtype=float)
    rmag0 = np.linalg.norm(r0, axis=1)
    vr0 = np.einsum("ij,ij->i", r0, v0) / rmag0
    alpha = 2.0 / rmag0 - np.einsum("ij,ij->i", v0, v0) / mu
    sqrt_mu = math.sqrt(mu)
    elliptic = alpha > 1.0e-12
    hyperbolic = alpha < -1.0e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        period = np.where(elliptic, 2.0 * math.pi / (sqrt_mu * np.abs(alpha) ** 1.5), np.inf)
    long_arc = elliptic & (np.abs(dt) > period)
    dt[long_arc] = np.fmod(dt[long_arc], period[long_arc])

    chi = sqrt_mu * dt / rmag0
    chi = np.where(elliptic, sqrt_mu * dt * alpha, chi)
    if np.any(hyperbolic):
        a = alpha[hyperbolic]
        sign = np.where(dt[hyperbolic] >= 0.0, 1.0, -1.0)
        denominator = vr0[hyperbolic] + sign * np.sqrt(-mu / a) * (1.0 - rmag0[hyperbolic] * a)
        numerator = -2.0 * mu * a * dt[hyperbolic]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator / denominator
            guess = np.where(
                (np.abs(denominator) > 1.0e-12) & (ratio > 0.0),
                sign * np.sqrt(-1.0 / a) * np.log(np.where(ratio > 0.0, ratio, 1.0)),
                sign * np.sqrt(-1.0 / a),
            )
        chi[hyperbolic] = guess

    a1 = rmag0 * vr0 / sqrt_mu
    a2 = 1.0 - alpha * rmag0
    n = _LAGUERRE_ORDER
    active = np.ones(len(dt), dtype=bool)
    for _ in range(80):
        if not np.any(active):
            break
        idx = np.flatnonzero(active)
        ch = chi[idx]
        zz = alpha[idx] * ch * ch
        c = stumpff_c_array(zz)
        s = stumpff_s_array(zz)
        f = a1[idx] * ch * ch * c + a2[idx] * ch**3 * s + rmag0[idx] * ch - sqrt_mu * dt[idx]
        df = a1[idx] * ch * (1.0 - zz * s) + a2[idx] * ch * ch * c + rmag0[idx]
        d2f = a1[idx] * (1.0 - zz * c) + a2[idx] * ch * (1.0 - zz * s)
        root = np.sqrt(np.abs((n - 1) ** 2 * df * df - n * (n - 1) * f * d2f))
        step = n * f / (df + np.copysign(root, df))
        chi[idx] = ch - step
        done = np.abs(step) <= 1.0e-13 * np.maximum(1.0, np.abs(chi[idx]))
        active[idx[done]] = False
    if np.any(active) or not np.all(np.isfinite(chi)):
        raise DynamicsError("universal-variable coast did not converge")
    zz = alpha * chi * chi
    c = stumpff_c_array(zz)
    s = stumpff_s_array(zz)
    f_lag = 1.0 - chi * chi / rmag0 * c
    g_lag = dt - chi**3 / sqrt_mu * s
    r = f_lag[:, None] * r0 + g_lag[:, None] * v0
    rmag = np.linalg.norm(r, axis=1)
    fdot = sqrt_mu / (rmag * rmag0) * (alpha * chi**3 * s - chi)
    gdot = 1.0 - chi * chi / rmag * c
    v = fdot[:, None] * r0 + gdot[:, None] * v0
    return r, v


def gravity(r: np.ndarray, mu: float) -> np.ndarray:
    radius = float(np.linalg.norm(r))
    gravity: np.ndarray = -mu * r / radius**3
    return gravity


def _conic_shape(
    r: np.ndarray, v: np.ndarray, mu: float
) -> tuple[float, float, float, float, float, float]:
    """radius, r.v, |v|^2, semi-latus rectum, eccentricity, e_vec . r -- in floats.

    numpy's cross on 3-vectors cost half of a refinement's run time.
    """
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    radius = math.sqrt(x * x + y * y + z * z)
    rv = x * vx + y * vy + z * vz
    v2 = vx * vx + vy * vy + vz * vz
    hx, hy, hz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    p = (hx * hx + hy * hy + hz * hz) / mu
    c1 = (v2 - mu / radius) / mu
    c2 = rv / mu
    ex, ey, ez = c1 * x - c2 * vx, c1 * y - c2 * vy, c1 * z - c2 * vz
    e = math.sqrt(ex * ex + ey * ey + ez * ez)
    return radius, rv, v2, p, e, ex * x + ey * y + ez * z


def periapsis_radius_km(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    """h^2 / (mu (1 + e)), for any conic."""
    _, _, _, p, e, _ = _conic_shape(r, v, mu)
    return p / (1.0 + e)


def periapsis_radius_gradient(r: np.ndarray, v: np.ndarray, mu: float) -> np.ndarray:
    """d r_p / d(r, v), from e^2 = 1 + 2 E p / mu."""
    radius, rv, v2, p, e, _ = _conic_shape(r, v, mu)
    energy = v2 / 2.0 - mu / radius
    dp = np.concatenate([v2 * r - rv * v, radius * radius * v - rv * r]) * (2.0 / mu)
    de_energy = np.concatenate([mu * r / radius**3, v])
    de = (p * de_energy + energy * dp) / (mu * e) if e > 1.0e-8 else np.zeros(6)
    gradient: np.ndarray = dp / (1.0 + e) - p * de / (1.0 + e) ** 2
    return gradient


def seconds_to_periapsis(r: np.ndarray, v: np.ndarray, mu: float) -> float | None:
    """Time until the next periapsis passage; None on an open arc moving away."""
    radius, rv, v2, p, e, e_dot_r = _conic_shape(r, v, mu)
    energy = v2 / 2.0 - mu / radius
    if e < 1.0e-10:
        # Circular: every point is a periapsis and the radius does not change.
        return 0.0
    cos_nu = min(1.0, max(-1.0, e_dot_r / (e * radius)))
    nu = math.acos(cos_nu)
    if rv < 0.0:
        nu = -nu
    if energy < 0.0 and e < 1.0:
        a = -mu / (2.0 * energy)
        big_e = 2.0 * math.atan(math.sqrt((1.0 - e) / (1.0 + e)) * math.tan(nu / 2.0))
        mean = big_e - e * math.sin(big_e)
        n = math.sqrt(mu / a**3)
        return ((-mean) % (2.0 * math.pi)) / n
    if rv >= 0.0:
        return None
    if e > 1.0 + 1.0e-9:
        a = -mu / (2.0 * energy)
        big_f = 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(nu / 2.0))
        mean = e * math.sinh(big_f) - big_f
        return -mean / math.sqrt(mu / (-a) ** 3)
    # Parabolic, by Barker's equation.
    d = math.tan(nu / 2.0)
    return -p * math.sqrt(p) / (2.0 * math.sqrt(mu)) * (d + d * d * d / 3.0)


def arc_minimum_radius(
    r: np.ndarray, v: np.ndarray, duration_s: float, r_end: np.ndarray, mu: float
) -> tuple[float, str]:
    """Smallest radius over a coast, in closed form, and where it occurs.

    Radius falls only towards periapsis, so the minimum is the periapsis when
    a passage lies inside the arc and an endpoint otherwise. Returns
    ("start" | "periapsis" | "end") with the value, which the gradient needs.
    """
    start = math.sqrt(float(r[0]) ** 2 + float(r[1]) ** 2 + float(r[2]) ** 2)
    end = math.sqrt(float(r_end[0]) ** 2 + float(r_end[1]) ** 2 + float(r_end[2]) ** 2)
    if duration_s <= 0.0:
        return start, "start"
    wait = seconds_to_periapsis(r, v, mu)
    if wait is not None and wait <= duration_s:
        periapsis = periapsis_radius_km(r, v, mu)
        if periapsis < min(start, end):
            return periapsis, "periapsis"
    if start <= end:
        return start, "start"
    return end, "end"
