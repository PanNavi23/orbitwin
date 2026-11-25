from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import root_scalar

from .errors import DynamicsError
from .models import StateVector, as_vector3

_EPS = 1.0e-12
#: GMAT R2026a refuses ECC with |1 - ECC| < 1e-7, both sides (measured at
#: 0.9999999, 1.00000001 refused; 1.0000001 accepted).
_PARABOLIC_ECC_BAND = 1.0e-7


@dataclass(frozen=True, slots=True)
class KeplerianElements:
    sma_km: float
    ecc: float
    inc_deg: float
    raan_deg: float
    aop_deg: float
    ta_deg: float


#: Taylor coefficients of S(z) = sum (-z)^k / (2k + 3)!. Eight terms reach
#: float64 roundoff for |z| < 1, where (root - sin root) / root^3 cancels.
_STUMPFF_S_SERIES = tuple(1.0 / math.factorial(2 * k + 3) for k in range(8))
_STUMPFF_S_SERIES_LIMIT = 1.0


def stumpff_c(z: float) -> float:
    """Stumpff C(z) in the half-angle form 2 sin^2(root/2) / z, which does not cancel."""
    if abs(z) < 1.0e-8:
        return 0.5 - z / 24.0 + z * z / 720.0 - z**3 / 40320.0 + z**4 / 3628800.0
    if z > 0.0:
        half = math.sin(0.5 * math.sqrt(z))
        return 2.0 * half * half / z
    root = math.sqrt(-z)
    if root > 350.0:
        raise DynamicsError("Stumpff C overflow for a very large negative argument")
    half = math.sinh(0.5 * root)
    return 2.0 * half * half / -z


def stumpff_s(z: float) -> float:
    """Stumpff S(z), by its series where the closed form cancels."""
    if abs(z) < _STUMPFF_S_SERIES_LIMIT:
        s0, s1, s2, s3, s4, s5, s6, s7 = _STUMPFF_S_SERIES
        return s0 - z * (s1 - z * (s2 - z * (s3 - z * (s4 - z * (s5 - z * (s6 - z * s7))))))
    if z > 0.0:
        root = math.sqrt(z)
        return (root - math.sin(root)) / (root**3)
    root = math.sqrt(-z)
    if root > 350.0:
        raise DynamicsError("Stumpff S overflow for a very large negative argument")
    return (math.sinh(root) - root) / (root**3)


#: Where the hyperbolic Stumpff branch overflows. The scalar forms raise there;
#: the array forms return NaN, because their only caller turns that raise into
#: a NaN anyway and an array has no way to raise for one element.
_STUMPFF_ARGUMENT_LIMIT = 350.0


def stumpff_c_array(z: np.ndarray) -> np.ndarray:
    """`stumpff_c` over an array, NaN where the scalar form would raise.

    Each branch is evaluated only on the elements that take it, with the
    operations the scalar form uses in the same order; the hyperbolic branch
    masks to NaN what the scalar form would have refused.

    That makes the two agree to within a unit in the last place, not to the
    bit. numpy selects its float64 transcendental kernels by CPU at run time,
    so `np.cos` and `math.cos` are identical on some machines and differ in
    their last place on others -- this repository's CI showed both, passing on
    one runner and failing a bit-identity assertion on another with numpy
    pinned to the same version. `tests/unit/test_lambert.py` bounds the
    difference and, separately, requires the roots it leads to be identical,
    which is the property that matters.
    """
    values = np.asarray(z, dtype=float)
    out = np.empty(values.shape, dtype=float)

    near_zero = np.abs(values) < 1.0e-8
    positive = ~near_zero & (values > 0.0)
    negative = ~near_zero & ~positive

    z_near = values[near_zero]
    out[near_zero] = (
        0.5 - z_near / 24.0 + z_near * z_near / 720.0
        - z_near**3 / 40320.0 + z_near**4 / 3628800.0
    )

    # errstate on both branches, not just the hyperbolic one. A caller that
    # hands this +inf gets NaN -- which is the contract, since the scalar form
    # raises there -- but without this it also got a RuntimeWarning on the way,
    # and a library that prints to stderr for an input it handles is a library
    # that gets `-W error` turned off rather than fixed.
    z_positive = values[positive]
    with np.errstate(invalid="ignore", divide="ignore"):
        half = np.sin(0.5 * np.sqrt(z_positive))
        out[positive] = 2.0 * half * half / z_positive

    z_negative = values[negative]
    with np.errstate(over="ignore", invalid="ignore"):
        root = np.sqrt(-z_negative)
        half = np.sinh(0.5 * root)
        hyperbolic = 2.0 * half * half / -z_negative
    hyperbolic[root > _STUMPFF_ARGUMENT_LIMIT] = np.nan
    out[negative] = hyperbolic
    return out


def stumpff_s_array(z: np.ndarray) -> np.ndarray:
    """`stumpff_s` over an array, NaN where the scalar form would raise."""
    values = np.asarray(z, dtype=float)
    out = np.empty(values.shape, dtype=float)

    near_zero = np.abs(values) < _STUMPFF_S_SERIES_LIMIT
    positive = ~near_zero & (values > 0.0)
    negative = ~near_zero & ~positive

    z_near = values[near_zero]
    total = np.zeros(z_near.shape, dtype=float)
    for coefficient in reversed(_STUMPFF_S_SERIES):
        total = total * -z_near + coefficient
    out[near_zero] = total

    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.sqrt(values[positive])
        out[positive] = (root - np.sin(root)) / (root**3)

    with np.errstate(over="ignore", invalid="ignore"):
        root = np.sqrt(-values[negative])
        hyperbolic = (np.sinh(root) - root) / (root**3)
    hyperbolic[root > _STUMPFF_ARGUMENT_LIMIT] = np.nan
    out[negative] = hyperbolic
    return out


def keplerian_to_cartesian(elements: KeplerianElements, mu_km3_s2: float) -> StateVector:
    """Convert classical Keplerian elements to an inertial Cartesian state.

    Takes ellipses (a > 0) and, in GMAT's convention, hyperbolas (a < 0, e > 1).
    p = a(1 - e^2) is positive for both; past a hyperbola's asymptotes, where
    1 + e cos(nu) <= 0, the formula returns a point on the unreachable branch.
    """
    if not (math.isfinite(mu_km3_s2) and mu_km3_s2 > 0.0):
        raise DynamicsError(f"mu must be finite and positive, got {mu_km3_s2!r}")
    a = float(elements.sma_km)
    e = float(elements.ecc)
    angles_deg = (elements.inc_deg, elements.raan_deg, elements.aop_deg, elements.ta_deg)
    if not all(math.isfinite(value) for value in (a, e, *angles_deg)) or e < 0.0:
        raise DynamicsError(f"Keplerian elements need finite values and ECC >= 0, got {elements}")
    if e < 1.0 and a <= 0.0:
        raise DynamicsError(f"An ellipse (ECC {e:.6g} < 1) needs a positive SMA, got {a:.6g} km")
    if e > 1.0 and a >= 0.0:
        raise DynamicsError(
            f"A hyperbola (ECC {e:.6g} > 1) needs a negative SMA, as GMAT writes it; "
            f"got {a:.6g} km"
        )
    if abs(1.0 - e) < _PARABOLIC_ECC_BAND:
        raise DynamicsError(f"ECC {e!r} is within 1e-7 of a parabola, which GMAT refuses")
    inc, raan, aop, ta = map(math.radians, angles_deg)
    p = a * (1.0 - e * e)
    denom = 1.0 + e * math.cos(ta)
    if denom <= _EPS:
        if e > 1.0:
            limit_deg = math.degrees(math.acos(-1.0 / e))
            raise DynamicsError(
                f"TA {elements.ta_deg:.6g} deg is at or past this hyperbola's asymptotes: "
                f"it cannot fall between {limit_deg:.6g} and {360.0 - limit_deg:.6g} deg"
            )
        raise DynamicsError("Keplerian state is singular at the supplied true anomaly")
    r_pf = np.array([p * math.cos(ta) / denom, p * math.sin(ta) / denom, 0.0])
    speed_scale = math.sqrt(mu_km3_s2 / p)
    v_pf = np.array([-speed_scale * math.sin(ta), speed_scale * (e + math.cos(ta)), 0.0])

    cr, sr = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(aop), math.sin(aop)
    rotation = np.array(
        [
            [cr * cw - sr * sw * ci, -cr * sw - sr * cw * ci, sr * si],
            [sr * cw + cr * sw * ci, -sr * sw + cr * cw * ci, -cr * si],
            [sw * si, cw * si, ci],
        ]
    )
    r = rotation @ r_pf
    v = rotation @ v_pf
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
        # e * e overflows past about 1e154.
        raise DynamicsError(f"Keplerian elements {elements} give a non-finite state")
    return StateVector(as_vector3(r), as_vector3(v))


def cartesian_to_keplerian(state: StateVector, mu_km3_s2: float) -> KeplerianElements:
    """Convert an inertial Cartesian state to classical elements."""
    r = state.r
    v = state.v
    rmag = float(np.linalg.norm(r))
    vmag = float(np.linalg.norm(v))
    if rmag <= 0.0:
        raise DynamicsError("Position magnitude must be positive")
    h = np.cross(r, v)
    hmag = float(np.linalg.norm(h))
    if hmag <= _EPS:
        raise DynamicsError("Angular momentum is zero")
    n = np.cross(np.array([0.0, 0.0, 1.0]), h)
    nmag = float(np.linalg.norm(n))
    evec = np.cross(v, h) / mu_km3_s2 - r / rmag
    e = float(np.linalg.norm(evec))
    energy = vmag * vmag / 2.0 - mu_km3_s2 / rmag
    if abs(energy) < _EPS:
        raise DynamicsError("Parabolic state has no finite semimajor axis")
    a = -mu_km3_s2 / (2.0 * energy)
    inc = math.acos(float(np.clip(h[2] / hmag, -1.0, 1.0)))

    # Kept as a block beside aop and ta, which cannot be ternaries (324d76d).
    if nmag > _EPS:  # noqa: SIM108
        raan = math.atan2(n[1], n[0]) % (2.0 * math.pi)
    else:
        raan = 0.0

    if e > 1.0e-10 and nmag > _EPS:
        aop = math.acos(float(np.clip(np.dot(n, evec) / (nmag * e), -1.0, 1.0)))
        if evec[2] < 0.0:
            aop = 2.0 * math.pi - aop
    else:
        aop = 0.0

    if e > 1.0e-10:
        ta = math.acos(float(np.clip(np.dot(evec, r) / (e * rmag), -1.0, 1.0)))
        if np.dot(r, v) < 0.0:
            ta = 2.0 * math.pi - ta
    elif nmag > _EPS:
        ta = math.acos(float(np.clip(np.dot(n, r) / (nmag * rmag), -1.0, 1.0)))
        if r[2] < 0.0:
            ta = 2.0 * math.pi - ta
    else:
        ta = math.atan2(r[1], r[0]) % (2.0 * math.pi)

    return KeplerianElements(
        sma_km=float(a),
        ecc=e,
        inc_deg=math.degrees(inc),
        raan_deg=math.degrees(raan),
        aop_deg=math.degrees(aop),
        ta_deg=math.degrees(ta),
    )


def first_radius_crossing_s(
    state: StateVector, radius_km: float, horizon_s: float, mu_km3_s2: float
) -> float | None:
    """When the two-body coast from `state` first comes within `radius_km`, or None
    if it does not by `horizon_s`. Closed form, so a brief graze is not stepped over."""
    if float(np.linalg.norm(state.r)) <= radius_km:
        return 0.0
    elements = cartesian_to_keplerian(state, mu_km3_s2)
    a, e = elements.sma_km, elements.ecc
    p = a * (1.0 - e * e)
    if e <= 1.0e-10 or p / (1.0 + e) > radius_km:
        return None
    # Inbound, the radius reaches radius_km at true anomaly -nu_x.
    nu_x = math.acos(min(1.0, max(-1.0, (p / radius_km - 1.0) / e)))
    nu0 = math.radians(elements.ta_deg)
    if e < 1.0:
        def mean(nu: float) -> float:
            big_e = 2.0 * math.atan2(
                math.sqrt(1.0 - e) * math.sin(nu / 2.0), math.sqrt(1.0 + e) * math.cos(nu / 2.0)
            )
            return big_e - e * math.sin(big_e)

        wait = ((mean(-nu_x) - mean(nu0)) % (2.0 * math.pi)) * math.sqrt(a**3 / mu_km3_s2)
    else:
        nu0 = nu0 - 2.0 * math.pi if nu0 > math.pi else nu0
        if nu0 > -nu_x:
            return None  # Already past periapsis, on the way out.

        def mean(nu: float) -> float:
            big_f = 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(nu / 2.0))
            return e * math.sinh(big_f) - big_f

        wait = (mean(-nu_x) - mean(nu0)) * math.sqrt((-a) ** 3 / mu_km3_s2)
    return wait if wait <= horizon_s else None


def orbital_period_s(sma_km: float, mu_km3_s2: float) -> float:
    if sma_km <= 0.0:
        raise DynamicsError("Orbital period requires positive semimajor axis")
    return 2.0 * math.pi * math.sqrt(sma_km**3 / mu_km3_s2)


def _universal_kepler_function(
    chi: float,
    dt_s: float,
    r0mag: float,
    vr0: float,
    alpha: float,
    sqrt_mu: float,
) -> tuple[float, float]:
    z = alpha * chi * chi
    c = stumpff_c(z)
    s = stumpff_s(z)
    f = (
        r0mag * vr0 / sqrt_mu * chi * chi * c
        + (1.0 - alpha * r0mag) * chi**3 * s
        + r0mag * chi
        - sqrt_mu * dt_s
    )
    derivative = (
        r0mag * vr0 / sqrt_mu * chi * (1.0 - z * s)
        + (1.0 - alpha * r0mag) * chi * chi * c
        + r0mag
    )
    return f, derivative


def _initial_universal_anomaly(
    dt_s: float,
    r0mag: float,
    vr0: float,
    alpha: float,
    mu_km3_s2: float,
) -> float:
    sqrt_mu = math.sqrt(mu_km3_s2)
    if abs(alpha) < 1.0e-10:
        return sqrt_mu * dt_s / r0mag
    if alpha > 0.0:
        return sqrt_mu * dt_s * alpha
    sign_dt = 1.0 if dt_s >= 0.0 else -1.0
    denominator = vr0 + sign_dt * math.sqrt(-mu_km3_s2 / alpha) * (1.0 - r0mag * alpha)
    numerator = -2.0 * mu_km3_s2 * alpha * dt_s
    if abs(denominator) > _EPS and numerator / denominator > 0.0:
        return sign_dt * math.sqrt(-1.0 / alpha) * math.log(numerator / denominator)
    return sign_dt * math.sqrt(-1.0 / alpha)


def _newton_universal_anomaly(
    guess: float,
    dt_s: float,
    r0mag: float,
    vr0: float,
    alpha: float,
    sqrt_mu: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    """scipy.optimize.newton(func, guess, fprime, tol, maxiter) on the Kepler function.

    The same root to the bit while no iterate overflows. Where a cube overflows,
    Python floats raise OverflowError and newton's np.float64 iterates raise
    another error; _universal_anomaly falls back to brentq on either. It
    evaluates the function once per step where newton calls it twice, and
    iterates in Python floats, which are faster than np.float64 scalars.
    """
    if tolerance <= 0:
        raise ValueError(f"tol too small ({tolerance:g} <= 0)")
    if max_iterations < 1:
        raise ValueError("maxiter must be greater than 0")
    p0 = guess
    for iteration in range(max_iterations):
        fval, fder = _universal_kepler_function(p0, dt_s, r0mag, vr0, alpha, sqrt_mu)
        if fval == 0:
            return p0
        if fder == 0:
            raise RuntimeError(
                f"Derivative was zero. Failed to converge after {iteration + 1} iterations,"
                f" value is {p0}."
            )
        p = p0 - fval / fder
        # newton's np.isclose(p, p0, rtol=0, atol=tolerance), which takes equal infinities as close.
        if p == p0 or abs(p - p0) <= tolerance:
            return p
        p0 = p
    raise RuntimeError(f"Failed to converge after {max_iterations} iterations, value is {p}.")


def _universal_anomaly(
    guess: float,
    dt_s: float,
    r0mag: float,
    vr0: float,
    alpha: float,
    sqrt_mu: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Newton's root of the Kepler function, or brentq's where Newton fails."""
    try:
        return _newton_universal_anomaly(
            guess, dt_s, r0mag, vr0, alpha, sqrt_mu, tolerance, max_iterations
        )
    except (RuntimeError, OverflowError, DynamicsError, ValueError) as newton_failure:
        return _bracketed_universal_anomaly(
            guess, dt_s, r0mag, vr0, alpha, sqrt_mu, tolerance, newton_failure
        )


def _bracketed_universal_anomaly(
    guess: float,
    dt_s: float,
    r0mag: float,
    vr0: float,
    alpha: float,
    sqrt_mu: float,
    tolerance: float,
    newton_failure: Exception,
) -> float:
    """brentq on a bracket doubled out from the guess; a failure here chains why Newton failed."""

    def func(chi: float) -> float:
        return _universal_kepler_function(chi, dt_s, r0mag, vr0, alpha, sqrt_mu)[0]

    scale = max(1.0, abs(guess), sqrt_mu * abs(dt_s) / r0mag)
    lo, hi = -scale, scale
    flo, fhi = func(lo), func(hi)
    for _ in range(80):
        if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi <= 0.0:
            break
        lo *= 2.0
        hi *= 2.0
        flo, fhi = func(lo), func(hi)
    else:
        raise DynamicsError("Could not bracket universal anomaly") from newton_failure
    result = root_scalar(func, bracket=(lo, hi), method="brentq", xtol=tolerance)
    if not result.converged:
        raise DynamicsError("Universal-variable propagation did not converge") from newton_failure
    return float(result.root)


def _type_name(value: object) -> str:
    kind = type(value)
    if kind.__module__ == "builtins":
        return kind.__qualname__
    return f"{kind.__module__}.{kind.__qualname__}"


def _real_number(name: str, value: Any) -> float:
    """value as a float; a 0-d array is its element. bool and timedelta64 are refused."""
    if type(value) is float:
        return value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    # numbers.Real takes numpy's integers and floats, and timedelta64, whose unit float() drops.
    if isinstance(value, (bool, np.timedelta64)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, not {_type_name(value)}")
    return float(value)


def propagate_universal(
    state: StateVector,
    dt_s: float,
    mu_km3_s2: float,
    *,
    tolerance: float = 1.0e-11,
    max_iterations: int = 80,
) -> StateVector:
    """Two-body propagation using universal variables.

    The routine supports elliptic and hyperbolic conics and negative propagation
    time. It validates the Lagrange identity before returning.
    """
    dt_s = _real_number("dt_s", dt_s)
    tolerance = _real_number("tolerance", tolerance)
    if not (math.isfinite(tolerance) and tolerance > 0.0):
        raise ValueError(f"tolerance must be finite and positive, not {tolerance!r}")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise TypeError(f"max_iterations must be an int, not {_type_name(max_iterations)}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1, not {max_iterations}")
    if abs(dt_s) < 1.0e-14:
        return state
    r0 = state.r
    v0 = state.v
    r0mag = float(np.linalg.norm(r0))
    if r0mag <= 0.0:
        raise DynamicsError("Initial position magnitude must be positive")
    v0sq = float(np.dot(v0, v0))
    vr0 = float(np.dot(r0, v0) / r0mag)
    alpha = 2.0 / r0mag - v0sq / mu_km3_s2
    sqrt_mu = math.sqrt(mu_km3_s2)
    guess = _initial_universal_anomaly(dt_s, r0mag, vr0, alpha, mu_km3_s2)
    try:
        chi = _universal_anomaly(guess, dt_s, r0mag, vr0, alpha, sqrt_mu, tolerance, max_iterations)
    except (OverflowError, ValueError) as error:
        raise DynamicsError(f"Universal anomaly not solved: {error}") from error

    z = alpha * chi * chi
    c = stumpff_c(z)
    s = stumpff_s(z)
    f = 1.0 - chi * chi / r0mag * c
    g = dt_s - chi**3 / sqrt_mu * s
    r = f * r0 + g * v0
    rmag = float(np.linalg.norm(r))
    if rmag <= 0.0 or not np.all(np.isfinite(r)):
        raise DynamicsError("Propagation produced an invalid position")
    fdot = sqrt_mu / (rmag * r0mag) * (alpha * chi**3 * s - chi)
    gdot = 1.0 - chi * chi / rmag * c
    v = fdot * r0 + gdot * v0
    identity_error = abs(f * gdot - fdot * g - 1.0)
    if identity_error > 2.0e-7:
        raise DynamicsError(f"Lagrange coefficient identity error is {identity_error:g}")
    if not np.all(np.isfinite(v)):
        raise DynamicsError("Propagation produced an invalid velocity")
    return StateVector(as_vector3(r), as_vector3(v))


def specific_energy(state: StateVector, mu_km3_s2: float) -> float:
    return float(np.dot(state.v, state.v) / 2.0 - mu_km3_s2 / np.linalg.norm(state.r))


def angular_momentum(state: StateVector) -> np.ndarray:
    return np.cross(state.r, state.v)
