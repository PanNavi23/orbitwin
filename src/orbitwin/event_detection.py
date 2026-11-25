from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .models import Burn, StateVector
from .trajectory import sample_pair_trajectory_sequential


@dataclass(frozen=True, slots=True)
class EventResult:
    first_entry_time_s: float | None
    minimum_distance_km: float
    minimum_distance_time_s: float
    distance_at_end_km: float
    samples_inside_radius: int = 0
    first_entry_segment_dt_s: float | None = None
    #: Closest the pursuer comes to the centre of the central body, over the
    #: flown rows. None when the caller did not ask for it.
    minimum_pursuer_radius_km: float | None = None
    #: The lowest Earth.Altitude GMAT reports on those rows, geodetic: the
    #: organizer's criterion, a cross-check and not a gate. None without the column.
    minimum_geodetic_altitude_km: float | None = None
    #: Where a hit's submitted script stops: policy.stop_depth_km inside, or the deepest
    #: point of a shallower passage. Only a GMAT report that measured it sets it.
    stop_time_s: float | None = None
    #: Epoch of the last sample: where the validator reads the completion time.
    end_time_s: float | None = None
    #: Closest report-row distance and its epoch, as a validator reads rows; None off a report.
    minimum_row_distance_km: float | None = None
    minimum_row_distance_time_s: float | None = None
    #: Largest gap between consecutive report rows after t = 0; None off a report.
    report_step_s: float | None = None

    @property
    def entry_is_under_resolved(self) -> bool:
        """True when the crossing rests on too few independent samples.

        Fewer than two samples inside the sphere means the first-entry epoch is
        carried entirely by interpolation across one segment. That is not
        automatically wrong, but it must not be reported as if it were densely
        sampled evidence.
        """
        return self.first_entry_time_s is not None and self.samples_inside_radius < 2


def _hermite_coefficients(
    p0: np.ndarray,
    v0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Return coefficients [constant, u, u², u³] for each coordinate."""
    a0 = p0
    a1 = dt_s * v0
    a2 = -3.0 * p0 - 2.0 * dt_s * v0 + 3.0 * p1 - dt_s * v1
    a3 = 2.0 * p0 + dt_s * v0 - 2.0 * p1 + dt_s * v1
    return np.stack((a0, a1, a2, a3), axis=1)


def _distance_squared_polynomial(coefficients: np.ndarray) -> np.ndarray:
    polynomial = np.zeros(7, dtype=float)
    for axis in range(3):
        polynomial += np.convolve(coefficients[axis], coefficients[axis])
    return polynomial


def _real_unit_roots(polynomial_ascending: np.ndarray, tolerance: float = 1.0e-9) -> list[float]:
    # np.trim_zeros on a seven-element array cost 9.5 s of a 103 s profile; this
    # is the same operation without the dispatch.
    descending = polynomial_ascending[::-1]
    first = 0
    length = len(descending)
    while first < length and descending[first] == 0.0:
        first += 1
    if length - first <= 1:
        return []
    trimmed = descending[first:]
    roots = np.roots(trimmed)
    result: list[float] = []
    for root in roots:
        if abs(float(root.imag)) <= tolerance:
            value = float(root.real)
            if -tolerance <= value <= 1.0 + tolerance:
                result.append(float(np.clip(value, 0.0, 1.0)))
    result.sort()
    dedup: list[float] = []
    for value in result:
        if not dedup or abs(value - dedup[-1]) > 1.0e-8:
            dedup.append(value)
    return dedup


def _polyval_ascending(u: float, polynomial_ascending: np.ndarray) -> float:
    return float(np.polynomial.polynomial.polyval(u, polynomial_ascending))


def _first_nonpositive_u(
    polynomial_ascending: np.ndarray, upper: float, iterations: int = 60
) -> float:
    """Return the earliest u in [0, upper] with q(u) <= 0.

    The caller has already established q(0) > 0 and q(upper) <= 0. Bisection is
    used instead of a polynomial root so that localisation cannot fail when the
    root finder returns a double root as a complex conjugate pair.
    """
    low, high = 0.0, float(upper)
    if _polyval_ascending(high, polynomial_ascending) > 0.0:
        return high
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if _polyval_ascending(middle, polynomial_ascending) <= 0.0:
            high = middle
        else:
            low = middle
    return high


def _distinct_epochs_inside(
    times_s: np.ndarray, distances_km: np.ndarray, radius_km: float
) -> int:
    """Count epochs inside the sphere, not rows.

    The sampler deliberately emits a duplicate row at an impulsive burn: the
    position is identical and only the velocity jumps. Counting rows would
    report twice the independent evidence that exists, and this number is what
    decides whether a crossing is flagged as interpolated.
    """
    if len(times_s) == 0:
        return 0
    first_of_epoch = np.concatenate(([True], np.diff(times_s) > 1.0e-13))
    return int(np.count_nonzero((distances_km <= radius_km) & first_of_epoch))


def evaluate_hermite_samples(
    times_s: np.ndarray,
    relative_positions_km: np.ndarray,
    relative_velocities_km_s: np.ndarray,
    radius_km: float,
) -> EventResult:
    times = np.asarray(times_s, dtype=float)
    positions = np.asarray(relative_positions_km, dtype=float)
    velocities = np.asarray(relative_velocities_km_s, dtype=float)
    if times.ndim != 1 or len(times) < 1:
        raise ValueError("At least one trajectory sample is required")
    if positions.shape != (len(times), 3) or velocities.shape != (len(times), 3):
        raise ValueError("Position and velocity arrays must have shape (N,3)")
    if np.any(np.diff(times) < -1.0e-12):
        raise ValueError("Trajectory times must be non-decreasing")

    distances = np.linalg.norm(positions, axis=1)
    # Per-segment screening bound, computed once for every segment as a vector
    # and handed to the loop as plain Python floats. Doing it inside the loop
    # meant three norms and a subtract per segment through numpy's scalar path,
    # which costs more than the comparison it feeds; a degenerate nan input also
    # produced one RuntimeWarning per segment instead of none.
    #
    # Writing the cubic in Hermite basis,
    #     p'(u) = h00'(u)(p0 - p1) + h10'(u) dt v0 + h11'(u) dt v1
    # and on [0, 1] |h00'| <= 3/2 while |h10'|, |h11'| <= 1, so the arc length is
    # at most reach = 1.5|p1 - p0| + dt(|v0| + |v1|) and no point of a segment
    # can be nearer than max(d0, d1) - reach. Since |d0 - d1| is itself within
    # reach, that bound never exceeds min(d0, d1), so skipping a segment cannot
    # miss an improvement sitting at an endpoint either.
    if len(times) > 1:
        speeds = np.linalg.norm(velocities, axis=1)
        with np.errstate(invalid="ignore"):
            reach = 1.5 * np.linalg.norm(np.diff(positions, axis=0), axis=1) + np.diff(times) * (
                speeds[:-1] + speeds[1:]
            )
            segment_floor = (
                np.maximum(distances[:-1], distances[1:]) - reach
            ).tolist()
    else:
        segment_floor = []
    minimum_distance = float(distances[0])
    minimum_time = float(times[0])
    first_entry: float | None = float(times[0]) if distances[0] <= radius_km else None
    first_entry_segment_dt: float | None = 0.0 if first_entry is not None else None

    for index in range(len(times) - 1):
        t0, t1 = float(times[index]), float(times[index + 1])
        dt = t1 - t0
        if dt <= 1.0e-13:
            # Duplicate epochs may appear across an impulsive velocity discontinuity.
            distance = float(np.linalg.norm(positions[index + 1]))
            if distance < minimum_distance:
                minimum_distance, minimum_time = distance, t1
            if first_entry is None and distance <= radius_km:
                first_entry = t1
                first_entry_segment_dt = 0.0
            continue
        # Most segments of a long trajectory are nowhere near the sphere. Root
        # finding dominated the profile, so skip it wherever the bound proves it
        # cannot matter. nan floors compare false and fall through to the exact
        # path, which is the safe direction.
        if segment_floor[index] > (
            minimum_distance if first_entry is not None else max(minimum_distance, radius_km)
        ):
            continue

        coeff = _hermite_coefficients(
            positions[index], velocities[index], positions[index + 1], velocities[index + 1], dt
        )
        d2_poly = _distance_squared_polynomial(coeff)
        derivative = np.array([power * d2_poly[power] for power in range(1, len(d2_poly))])
        stationary = _real_unit_roots(derivative)
        for u in [0.0, *stationary, 1.0]:
            vector = sum(coeff[:, power] * (u**power) for power in range(4))
            distance = float(np.linalg.norm(vector))
            time = t0 + u * dt
            if distance < minimum_distance:
                minimum_distance, minimum_time = distance, time

        if first_entry is None:
            crossing_poly = d2_poly.copy()
            crossing_poly[0] -= radius_km * radius_km
            if float(crossing_poly[0]) <= 0.0:
                first_entry = t0
                first_entry_segment_dt = dt
            else:
                # Acceptance is decided by whether q reaches zero anywhere in the
                # segment, evaluated at the stationary points that already
                # produced the minimum distance. It must not be decided by the
                # sign of a root's slope, which vanishes exactly at a tangency.
                probes = sorted(
                    {
                        *_real_unit_roots(crossing_poly),
                        *(u for u in stationary if 0.0 < u < 1.0),
                        1.0,
                    }
                )
                inside = [u for u in probes if _polyval_ascending(u, crossing_poly) <= 0.0]
                if inside:
                    first_entry = t0 + _first_nonpositive_u(crossing_poly, min(inside)) * dt
                    first_entry_segment_dt = dt

    return EventResult(
        first_entry_time_s=first_entry,
        minimum_distance_km=minimum_distance,
        minimum_distance_time_s=minimum_time,
        distance_at_end_km=float(distances[-1]),
        samples_inside_radius=_distinct_epochs_inside(times, distances, radius_km),
        first_entry_segment_dt_s=first_entry_segment_dt,
        end_time_s=float(times[-1]),
    )


def evaluate_analytical_trajectory(
    pursuer_initial: StateVector,
    target_initial: StateVector,
    burns: tuple[Burn, ...],
    end_time_s: float,
    mu_km3_s2: float,
    radius_km: float,
    *,
    max_step_s: float = 2.0,
    central_body_radius_km: float | None = None,
) -> EventResult:
    """Evaluate the interception event, and optionally how low the pursuer flies.

    ``central_body_radius_km`` is answered from the pursuer's own inertial
    positions, which come back from the integrator that was going to run
    anyway. It is the minimum over the samples rather than over the Hermite
    interpolant: near pericenter the radius varies quadratically in time, so at
    the default 2 s step the two agree to about a metre -- measured at 1.1 m on
    a 1.5 km/s retrograde impulse and 1.3 m on a 0.8 km/s one -- while running
    the interpolant costs a second stationary-point solve on every segment,
    which no radius screen can skip. On one real scenario that was 31.3 s to
    59.6 s for eight candidates. A metre does not decide whether a trajectory
    is 570 km underground.

    It is measured only as far as the completion time. 初賽規則 §4 fixes Δrmin
    at the interception radius once tint has passed, so nothing after it can
    change the score; the submitted script runs on to 4TA only to carry the
    Δr(4TA) that §4 scores when there is no interception. A pass below the
    surface in that tail says nothing about the interception that already
    happened before it.
    """
    if end_time_s < 0.0:
        raise ValueError("end_time_s cannot be negative")
    # The pursuer velocities are what an exact Hermite minimum would need. This
    # takes the sampled one instead, for the reason in the docstring, and
    # test_the_sampled_minimum_tracks_the_exact_one pins the two together.
    times, positions, velocities, pursuer_positions, _pursuer_velocities = (
        sample_pair_trajectory_sequential(
            pursuer_initial,
            target_initial,
            burns,
            end_time_s,
            mu_km3_s2,
            max_step_s=max_step_s,
        )
    )
    event = evaluate_hermite_samples(times, positions, velocities, radius_km)
    if central_body_radius_km is None:
        return event
    cut = len(times)
    if event.first_entry_time_s is not None:
        # Keeping the first sample past tint keeps the segment containing tint
        # whole, which errs towards reporting a lower radius rather than
        # missing one.
        cut = min(
            len(times),
            int(np.searchsorted(times, event.first_entry_time_s, side="right")) + 1,
        )
    minimum_radius_km = float(np.min(np.linalg.norm(pursuer_positions[:cut], axis=1)))
    return replace(event, minimum_pursuer_radius_km=minimum_radius_km)
