from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar

from .errors import DynamicsError, LambertError
from .models import StateVector, as_vector3
from .orbit import (
    propagate_universal,
    stumpff_c,
    stumpff_c_array,
    stumpff_s,
    stumpff_s_array,
)


@dataclass(frozen=True, slots=True)
class LambertSolution:
    v1_km_s: tuple[float, float, float]
    v2_km_s: tuple[float, float, float]
    transfer_angle_rad: float
    revolution: int
    path_index: int
    direction: str
    z: float
    endpoint_error_km: float

    @property
    def v1(self) -> np.ndarray:
        return np.asarray(self.v1_km_s, dtype=float)

    @property
    def v2(self) -> np.ndarray:
        return np.asarray(self.v2_km_s, dtype=float)

    @property
    def branch(self) -> str:
        path = "short" if self.transfer_angle_rad <= math.pi else "long"
        return f"{self.direction}-{path}-M{self.revolution}-P{self.path_index}"


def _transfer_angles(r1: np.ndarray, r2: np.ndarray) -> list[tuple[float, str]]:
    r1mag = float(np.linalg.norm(r1))
    r2mag = float(np.linalg.norm(r2))
    cosine = float(np.clip(np.dot(r1, r2) / (r1mag * r2mag), -1.0, 1.0))
    base = math.acos(cosine)
    cross_z = float(np.cross(r1, r2)[2])
    if base < 1.0e-10 or abs(base - math.pi) < 1.0e-10:
        # The transfer plane is not unique. Nearby-time search/refinement is required.
        return [(base, "ambiguous"), (2.0 * math.pi - base, "ambiguous-reverse")]
    if cross_z >= 0.0:
        return [(base, "prograde"), (2.0 * math.pi - base, "retrograde")]
    return [(base, "retrograde"), (2.0 * math.pi - base, "prograde")]


def _lambert_function_factory(
    r1mag: float,
    r2mag: float,
    a_term: float,
    tof_s: float,
    mu_km3_s2: float,
) -> Callable[[float], float]:
    sqrt_mu_t = math.sqrt(mu_km3_s2) * tof_s

    def function(z: float) -> float:
        try:
            c = stumpff_c(z)
            s = stumpff_s(z)
        except DynamicsError:
            return math.nan
        if c <= 0.0 or not math.isfinite(c) or not math.isfinite(s):
            return math.nan
        y = r1mag + r2mag + a_term * (z * s - 1.0) / math.sqrt(c)
        if y < 0.0 or not math.isfinite(y):
            return math.nan
        x = math.sqrt(y / c)
        value = x**3 * s + a_term * math.sqrt(y) - sqrt_mu_t
        return value if math.isfinite(value) else math.nan

    return function


def _lambert_residual_array(
    z: np.ndarray,
    r1mag: float,
    r2mag: float,
    a_term: float,
    sqrt_mu_t: float,
) -> np.ndarray:
    """`_lambert_function_factory`'s residual, over an array.

    Same expressions in the same order as the scalar form, which makes the two
    agree to a rounding difference in their inputs rather than to the bit:
    numpy picks its float64 transcendental kernels by CPU at run time, so this
    is not something a build can promise.

    These values choose the brackets handed to brentq, so the difference has to
    be small enough not to choose a different one. It is, and that is asserted
    rather than argued -- `test_the_sampler_changes_no_root` compares the roots
    themselves, with and without this function, across every interval of five
    geometries. brentq refines each bracket with the scalar function, and the
    direct-root and tangent paths dedup against each other within 1e-7, so a
    last-place difference in a sample has nowhere to land.

    Every place the scalar form returns NaN -- a Stumpff overflow it raises on,
    a non-positive C, a negative y, a non-finite result -- this returns NaN for
    the same element.
    """
    values = np.asarray(z, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        c = stumpff_c_array(values)
        s = stumpff_s_array(values)
        unusable = ~np.isfinite(c) | ~np.isfinite(s) | (c <= 0.0)

        y = r1mag + r2mag + a_term * (values * s - 1.0) / np.sqrt(c)
        unusable |= ~np.isfinite(y) | (y < 0.0)

        x = np.sqrt(y / c)
        residual = x**3 * s + a_term * np.sqrt(y) - sqrt_mu_t

    return np.where(unusable | ~np.isfinite(residual), np.nan, residual)


def _root_intervals(max_revolutions: int) -> list[tuple[float, float, int]]:
    # Negative-z hyperbolic interval plus elliptic intervals separated by C(z)=0.
    intervals: list[tuple[float, float, int]] = [(-16.0 * math.pi**2, -1.0e-10, 0)]
    epsilon = 1.0e-7
    for revolution in range(max_revolutions + 1):
        left = (2.0 * math.pi * revolution) ** 2
        right = (2.0 * math.pi * (revolution + 1)) ** 2
        if revolution == 0:
            left = 0.0
        else:
            left += epsilon * max(1.0, left)
        right -= epsilon * max(1.0, right)
        intervals.append((left, right, revolution))
    return intervals


def _roots_in_interval(
    function: Callable[[float], float],
    left: float,
    right: float,
    *,
    samples: int,
    root_tolerance: float,
    sampler: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[float]:
    """Roots of `function` in [left, right].

    `sampler` is the same residual over an array. It is optional so a caller
    with only a scalar function -- a test, or a future branch shape -- still
    works; when it is supplied the sampling grid costs one numpy call instead
    of `samples` trips through the interpreter, which is 78% of the residual
    evaluations a one-burn sweep performs.

    brentq and the tangent-root guard keep the scalar function. They ask for
    one point at a time by nature, and routing them through numpy would make
    them slower, not faster.
    """
    # Chebyshev-like clustering protects roots near singular interval endpoints.
    theta = np.linspace(math.pi, 0.0, samples)
    points = (left + right) / 2.0 + (right - left) / 2.0 * np.cos(theta)
    points = np.unique(np.sort(points))
    if sampler is None:
        values = np.array([function(float(z)) for z in points], dtype=float)
    else:
        values = np.asarray(sampler(points), dtype=float)
    # Which adjacent pairs are usable, sit on a root, or straddle one. The
    # scalar loop this replaces asked those three questions per pair through
    # the interpreter: two float() casts, two isfinite calls, an abs and a
    # multiply, times `samples` pairs, times every interval of every branch of
    # every geometry a sweep visits. Measured on the 2026-09-05 scenario it was
    # 344,000 isfinite calls per round and the largest single cost in a Lambert
    # solve once the residual itself was vectorised.
    #
    # brentq is still called once per crossing, one at a time. Only the search
    # for crossings is vectorised, not the refinement of them.
    finite = np.isfinite(values)
    usable = finite[:-1] & finite[1:]
    on_root = usable & (np.abs(values[:-1]) <= root_tolerance)
    straddling = usable & (values[:-1] * values[1:] < 0.0)

    roots: list[float] = [float(z) for z in points[:-1][on_root]]
    for index in np.flatnonzero(straddling):
        try:
            roots.append(
                float(
                    brentq(
                        function,
                        float(points[index]),
                        float(points[index + 1]),
                        xtol=1.0e-12,
                        rtol=1.0e-12,
                    )
                )
            )
        except (ValueError, RuntimeError):
            continue
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size:
        best_index = int(finite_indices[np.argmin(np.abs(values[finite_indices]))])
        if abs(float(values[best_index])) < max(root_tolerance * 100.0, 1.0e-6):
            roots.append(float(points[best_index]))

        # Multi-revolution time-of-flight curves can have a minimum with both
        # endpoints of the interval on the same side. Minimize |F| as a tangent-root guard.
        def penalised_magnitude(z: float) -> float:
            """|F(z)|, with the non-finite regions pushed above any real minimum.

            One evaluation per point. The conditional expression this replaces
            called `function` twice -- once to test finiteness and once for the
            value -- and `function` is pure, so half of those were redundant.
            Measured on the 2026-09-05 scenario they were a tenth of every
            residual evaluation a sweep performed.
            """
            value = function(float(z))
            return abs(value) if math.isfinite(value) else 1.0e300

        try:
            result = minimize_scalar(
                penalised_magnitude,
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1.0e-10, "maxiter": 200},
            )
            if result.success and float(result.fun) < max(root_tolerance * 100.0, 1.0e-6):
                roots.append(float(result.x))
        except (ValueError, OverflowError, DynamicsError):
            pass

    roots.sort()
    deduplicated: list[float] = []
    for root in roots:
        if not deduplicated or abs(root - deduplicated[-1]) > 1.0e-7 * max(1.0, abs(root)):
            deduplicated.append(root)
    return deduplicated


def _velocities_from_root(
    r1: np.ndarray,
    r2: np.ndarray,
    a_term: float,
    z: float,
    mu_km3_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    r1mag = float(np.linalg.norm(r1))
    r2mag = float(np.linalg.norm(r2))
    c = stumpff_c(z)
    s = stumpff_s(z)
    if c <= 0.0:
        raise LambertError("Lambert root has non-positive Stumpff C")
    y = r1mag + r2mag + a_term * (z * s - 1.0) / math.sqrt(c)
    if y <= 0.0:
        raise LambertError("Lambert root has non-positive y")
    f = 1.0 - y / r1mag
    g = a_term * math.sqrt(y / mu_km3_s2)
    gdot = 1.0 - y / r2mag
    if abs(g) < 1.0e-13:
        raise LambertError("Lambert g coefficient is singular")
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        raise LambertError("Lambert reconstruction returned non-finite velocities")
    return v1, v2



def _shooting_fallback(
    r1: np.ndarray,
    r2: np.ndarray,
    tof_s: float,
    mu_km3_s2: float,
    endpoint_tolerance_km: float,
    reference_normal: np.ndarray | None,
) -> list[LambertSolution]:
    """Solve singular/near-collinear Lambert geometry by Cartesian shooting.

    This is deliberately a fallback, not the primary bulk solver. It is slower
    but avoids the undefined transfer plane at 0/180 degree endpoint geometry.
    """
    r1mag = float(np.linalg.norm(r1))
    r2mag = float(np.linalg.norm(r2))
    normal = None if reference_normal is None else np.asarray(reference_normal, dtype=float)
    if normal is None or np.linalg.norm(normal) < 1.0e-12:
        normal = np.cross(r1, r2)
    if np.linalg.norm(normal) < 1.0e-12:
        # Pick a deterministic axis least aligned with r1.
        axes = np.eye(3)
        axis = axes[int(np.argmin(np.abs(axes @ (r1 / r1mag))))]
        normal = np.cross(r1, axis)
    normal = normal / np.linalg.norm(normal)
    tangent = np.cross(normal, r1)
    tangent = tangent / np.linalg.norm(tangent)
    chord = (r2 - r1) / tof_s
    circular = math.sqrt(mu_km3_s2 / r1mag)
    transfer_a = max(1.0, 0.5 * (r1mag + r2mag))
    vis_viva = math.sqrt(max(1.0e-12, mu_km3_s2 * (2.0 / r1mag - 1.0 / transfer_a)))
    guesses: list[np.ndarray] = [chord]
    for sign in (1.0, -1.0):
        for scale in (0.5, 0.8, 1.0, 1.2, 1.6):
            guesses.append(sign * tangent * circular * scale + 0.15 * chord)
        guesses.append(sign * tangent * vis_viva)

    solutions: list[LambertSolution] = []
    for guess in guesses:
        def residual(velocity: np.ndarray) -> np.ndarray:
            try:
                propagated = propagate_universal(
                    StateVector(as_vector3(r1), as_vector3(velocity)),
                    tof_s,
                    mu_km3_s2,
                )
                miss_km: np.ndarray = propagated.r - r2
                return miss_km
            except (DynamicsError, ValueError, OverflowError):
                return np.full(3, 1.0e6)

        try:
            result = least_squares(
                residual,
                guess,
                method="trf",
                x_scale=max(1.0, circular),
                ftol=1.0e-12,
                xtol=1.0e-12,
                gtol=1.0e-12,
                max_nfev=250,
                bounds=(-25.0, 25.0),
            )
        except (ValueError, RuntimeError, DynamicsError):
            continue
        error = float(np.linalg.norm(residual(result.x)))
        if not result.success or error > endpoint_tolerance_km:
            continue
        try:
            arrival = propagate_universal(
                StateVector(as_vector3(r1), as_vector3(result.x)),
                tof_s,
                mu_km3_s2,
            )
        except DynamicsError:
            continue
        direction = "prograde" if np.cross(r1, result.x)[2] >= 0.0 else "retrograde"
        solution = LambertSolution(
            as_vector3(result.x),
            arrival.v_km_s,
            math.pi,
            0,
            0,
            direction + "-shooting",
            math.nan,
            error,
        )
        if not any(np.linalg.norm(solution.v1 - old.v1) < 1.0e-7 for old in solutions):
            solutions.append(solution)
    return solutions


def solve_lambert_all(
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    tof_s: float,
    mu_km3_s2: float,
    *,
    max_revolutions: int = 0,
    samples_per_interval: int = 120,
    endpoint_tolerance_km: float = 0.05,
    reference_normal: np.ndarray | None = None,
) -> list[LambertSolution]:
    """Enumerate validated 3-D universal-variable Lambert branches.

    Both angular-momentum directions are considered. Positive-z intervals are
    scanned between Stumpff singularities, which exposes zero- and multiple-
    revolution roots. Every returned velocity is independently propagated to
    the requested arrival epoch; roots exceeding ``endpoint_tolerance_km`` are
    rejected.
    """
    r1 = np.asarray(r1_km, dtype=float)
    r2 = np.asarray(r2_km, dtype=float)
    if r1.shape != (3,) or r2.shape != (3,):
        raise LambertError("Lambert endpoints must be three-dimensional vectors")
    if tof_s <= 0.0 or mu_km3_s2 <= 0.0:
        raise LambertError("Lambert time of flight and mu must be positive")
    r1mag = float(np.linalg.norm(r1))
    r2mag = float(np.linalg.norm(r2))
    if r1mag <= 0.0 or r2mag <= 0.0:
        raise LambertError("Lambert endpoint radius must be positive")

    solutions: list[LambertSolution] = []
    for transfer_angle, direction in _transfer_angles(r1, r2):
        cosine = math.cos(transfer_angle)
        denominator = 1.0 - cosine
        if denominator <= 1.0e-14:
            continue
        a_term = math.sin(transfer_angle) * math.sqrt(r1mag * r2mag / denominator)
        if abs(a_term) < 1.0e-12:
            continue
        function = _lambert_function_factory(r1mag, r2mag, a_term, tof_s, mu_km3_s2)
        sampler = partial(
            _lambert_residual_array,
            r1mag=r1mag,
            r2mag=r2mag,
            a_term=a_term,
            sqrt_mu_t=math.sqrt(mu_km3_s2) * tof_s,
        )
        for left, right, revolution in _root_intervals(max_revolutions):
            roots = _roots_in_interval(
                function,
                left,
                right,
                samples=samples_per_interval,
                root_tolerance=1.0e-7,
                sampler=sampler,
            )
            path_index = 0
            for root in roots:
                try:
                    v1, v2 = _velocities_from_root(r1, r2, a_term, root, mu_km3_s2)
                    propagated = propagate_universal(
                        StateVector(as_vector3(r1), as_vector3(v1)),
                        tof_s,
                        mu_km3_s2,
                    )
                except (DynamicsError, LambertError, OverflowError, ValueError):
                    continue
                endpoint_error = float(np.linalg.norm(propagated.r - r2))
                if endpoint_error > endpoint_tolerance_km:
                    continue
                solutions.append(
                    LambertSolution(
                        as_vector3(v1),
                        as_vector3(v2),
                        transfer_angle,
                        revolution,
                        path_index,
                        direction,
                        root,
                        endpoint_error,
                    )
                )
                path_index += 1

    # Universal-variable A becomes singular for exactly collinear endpoints.
    # Use a slower Cartesian shooting fallback when no validated branch exists.
    if not solutions:
        solutions.extend(
            _shooting_fallback(
                r1, r2, tof_s, mu_km3_s2, endpoint_tolerance_km, reference_normal
            )
        )

    # Deduplicate numerically equivalent roots while preserving deterministic order.
    solutions.sort(
        key=lambda item: (
            item.revolution,
            item.direction,
            item.transfer_angle_rad,
            item.z if math.isfinite(item.z) else math.inf,
        )
    )
    unique: list[LambertSolution] = []
    for solution in solutions:
        if any(
            np.linalg.norm(solution.v1 - previous.v1) < 1.0e-8
            and abs(solution.transfer_angle_rad - previous.transfer_angle_rad) < 1.0e-9
            for previous in unique
        ):
            continue
        unique.append(solution)
    return unique
