from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.integrate import solve_ivp

from .conic import coast_many
from .errors import DynamicsError
from .models import (
    Burn,
    Scenario,
    StateVector,
    as_vector3,
    head_start_jump,
    pursuer_start,
    uses_head_start,
)
from .orbit import propagate_universal


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    time_s: float
    pursuer: StateVector
    target: StateVector


def apply_burn(state: StateVector, dv_inertial_km_s: tuple[float, float, float]) -> StateVector:
    velocity = state.v + np.asarray(dv_inertial_km_s, dtype=float)
    return StateVector(state.r_km, as_vector3(velocity))


def state_with_burns_at(
    initial_state: StateVector,
    burns: tuple[Burn, ...],
    time_s: float,
    mu_km3_s2: float,
    *,
    include_burn_at_time: bool = True,
    start_s: float = 0.0,
) -> StateVector:
    """Two-body, from ``initial_state`` at ``start_s`` (models.pursuer_start)."""
    if time_s < start_s - 1.0e-12:
        raise DynamicsError("Trajectory time cannot be before its start")
    state = initial_state
    cursor = start_s
    for burn in sorted(burns, key=lambda item: item.time_s):
        if burn.time_s > time_s + 1.0e-12:
            break
        state = propagate_universal(state, burn.time_s - cursor, mu_km3_s2)
        cursor = burn.time_s
        if include_burn_at_time or burn.time_s < time_s - 1.0e-12:
            state = apply_burn(state, burn.dv_inertial_km_s)
    return propagate_universal(state, time_s - cursor, mu_km3_s2)


def head_start_reference(scenario: Scenario) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GMAT's unburned head start, (t s on the solver clock, r km, v km/s): the
    alignment probe's rows of the pursuer (AlignedStates.early_track)."""
    probe = scenario.state_probe
    aligned = probe.aligned if probe is not None else None
    if aligned is None or not aligned.early_track:
        raise DynamicsError("a head-start plan needs the alignment probe's rows")
    rows = np.asarray(aligned.early_track, dtype=float)
    keep = np.concatenate([np.diff(rows[:, 0]) > 0.0, [True]])  # rows at one epoch
    rows = rows[keep]
    return rows[:, 0], rows[:, 1:4], rows[:, 4:7]


def reference_states(
    scenario: Scenario, times_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """GMAT's unburned pursuer at `times_s` (<= 0), Hermite between the probe's rows."""
    grid, r, v = head_start_reference(scenario)
    times = np.asarray(times_s, dtype=float)
    k = np.clip(np.searchsorted(grid, times, "right") - 1, 0, len(grid) - 2)
    h = (grid[k + 1] - grid[k])[:, None]
    u = ((times - grid[k]) / h[:, 0])[:, None]
    p0, p1, m0, m1 = r[k], r[k + 1], v[k] * h, v[k + 1] * h
    a2 = -3.0 * p0 - 2.0 * m0 + 3.0 * p1 - m1
    a3 = 2.0 * p0 + m0 - 2.0 * p1 + m1
    position = p0 + m0 * u + a2 * u * u + a3 * u**3
    velocity = (m0 + 2.0 * a2 * u + 3.0 * a3 * u * u) / h
    return position, velocity


def fly_pursuer(
    scenario: Scenario,
    schedule: list[tuple[float, np.ndarray]],
    until_s: float,
) -> tuple[list[StateVector], StateVector]:
    """Two-body under (time, inertial impulse) pairs: the pre-burn state at each and
    the state at `until_s`. With head-start burns it starts at the pursuer's epoch and
    takes on Scenario.head_start_defect at t = 0 (models.head_start_jump)."""
    times = [time_s for time_s, _ in schedule]
    now_s, state = pursuer_start(scenario, times)
    jump = head_start_jump(scenario, times)
    mu = scenario.mu_km3_s2

    def advance(state: StateVector, now_s: float, to_s: float) -> StateVector:
        nonlocal jump
        if jump is not None and now_s < 0.0 <= to_s:
            state = propagate_universal(state, -now_s, mu)
            state = StateVector(as_vector3(state.r + jump[0]), as_vector3(state.v + jump[1]))
            now_s, jump = 0.0, None
        return propagate_universal(state, to_s - now_s, mu)

    before: list[StateVector] = []
    for time_s, impulse in schedule:
        state = advance(state, now_s, time_s)
        now_s = time_s
        # A head-start burn's axes stay on this two-body arc, not GMAT's rows: on R9
        # GMAT's rows took the first flight's miss at arrival from 33.76 to 39.92 km.
        before.append(state)
        state = StateVector(as_vector3(state.r), as_vector3(state.v + np.asarray(impulse)))
    return before, advance(state, now_s, until_s)


def head_start_alignment_distance_km(scenario: Scenario, burns: tuple[Burn, ...]) -> float | None:
    """Closest the pursuer comes to the target waiting at its epoch while head-start
    burns fly it through the alignment; None without them. On the alignment probe's
    rows (head_start_reference), each moved by what the burns before it changed
    two-body, as GMAT's own rows would be read.

    What d_min counts of the alignment's rows under closest_approach_window
    "from_origin", for this candidate rather than the burn-free probe's.
    """
    if not uses_head_start(burns):
        return None
    grid, _, _ = head_start_reference(scenario)
    rows = grid[grid < 0.0]
    if not len(rows):
        return None
    start_s, start = pursuer_start(scenario, [burn.time_s for burn in burns])
    early = sorted((b for b in burns if b.time_s < 0.0), key=lambda b: b.time_s)
    mu = scenario.mu_km3_s2
    # Burned less unburned, two-body, at each row: zero before the first burn.
    delta_r = np.zeros((len(rows), 3))
    unburned_r, _ = coast_many(np.tile(start.r, (len(rows), 1)), np.tile(start.v, (len(rows), 1)),
                               rows - start_s, mu)
    state, now = start, start_s
    for index, burn in enumerate(early):
        state = propagate_universal(state, burn.time_s - now, mu)
        state = apply_burn(state, burn.dv_inertial_km_s)
        now = burn.time_s
        end = early[index + 1].time_s if index + 1 < len(early) else 0.0
        at = (rows >= burn.time_s) & (rows < end)
        if at.any():
            burned_r, _ = coast_many(np.tile(state.r, (int(at.sum()), 1)),
                                     np.tile(state.v, (int(at.sum()), 1)), rows[at] - now, mu)
            delta_r[at] = burned_r - unburned_r[at]
    reference_r, _ = reference_states(scenario, rows)
    target = np.asarray(scenario.target.initial_state.r, dtype=float)
    return float(np.linalg.norm(reference_r + delta_r - target, axis=1).min())

def sample_relative_trajectory(
    pursuer_initial: StateVector,
    target_initial: StateVector,
    burns: tuple[Burn, ...],
    times_s: np.ndarray,
    mu_km3_s2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference sampler that propagates each requested epoch independently.

    This function is retained for tests and spot checks. Bulk event detection
    uses :func:`sample_relative_trajectory_sequential`, which integrates each
    coast segment once instead of replaying from the initial epoch for every
    sample.
    """
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for time_s in times_s:
        pursuer = state_with_burns_at(pursuer_initial, burns, float(time_s), mu_km3_s2)
        target = propagate_universal(target_initial, float(time_s), mu_km3_s2)
        positions.append(pursuer.r - target.r)
        velocities.append(pursuer.v - target.v)
    return np.asarray(times_s, dtype=float), np.asarray(positions), np.asarray(velocities)


def _two_body_pair_derivative(_time_s: float, state: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    """Two-body acceleration for both spacecraft.

    Written in scalars on purpose. This is the innermost function of the
    integrator -- called 203,966 times for a single 12,000 s trajectory in the
    measured profile -- and at that call count numpy's dispatch overhead
    dominates the arithmetic. ``np.linalg.norm`` on a three-element vector alone
    accounted for 25.7 s of a 103 s profile.

    The expression order is kept identical to the array form, so results are
    bit-for-bit unchanged.
    """
    # tolist() converts once to Python floats. Indexing the array instead
    # allocates a numpy scalar per access, which at this call count costs more
    # than the arithmetic: 5.28 us/call against 2.71 us/call measured, for
    # bit-identical output.
    x0, y0, z0, vx0, vy0, vz0, x1, y1, z1, vx1, vy1, vz1 = state.tolist()
    radius0 = math.sqrt(x0 * x0 + y0 * y0 + z0 * z0)
    radius1 = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    if (
        not math.isfinite(radius0)
        or radius0 <= 1.0e-9
        or not math.isfinite(radius1)
        or radius1 <= 1.0e-9
    ):
        raise DynamicsError("Invalid radius during trajectory integration")
    cube0 = radius0**3
    cube1 = radius1**3
    return np.array(
        (
            vx0,
            vy0,
            vz0,
            -mu_km3_s2 * x0 / cube0,
            -mu_km3_s2 * y0 / cube0,
            -mu_km3_s2 * z0 / cube0,
            vx1,
            vy1,
            vz1,
            -mu_km3_s2 * x1 / cube1,
            -mu_km3_s2 * y1 / cube1,
            -mu_km3_s2 * z1 / cube1,
        )
    )


def sample_pair_trajectory_sequential(
    pursuer_initial: StateVector,
    target_initial: StateVector,
    burns: tuple[Burn, ...],
    end_time_s: float,
    mu_km3_s2: float,
    *,
    max_step_s: float,
    rtol: float = 2.0e-11,
    atol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate both spacecraft once per coast segment.

    Returns the sample epochs, the relative position and velocity the event
    detector works on, and the pursuer's own inertial position and velocity.
    The pursuer state is already in the integrator's 12-element vector, so
    returning it costs a slice rather than a second propagation, and it is what
    answers whether a candidate flies through the central body.

    Duplicate samples are deliberately emitted at an impulsive-burn epoch:
    the pre- and post-burn positions are identical while velocity jumps. This
    prevents the Hermite event detector from interpolating through the impulse.
    """
    if end_time_s < 0.0:
        raise DynamicsError("Trajectory end time cannot be negative")
    if max_step_s <= 0.0:
        raise DynamicsError("Trajectory max_step_s must be positive")

    ordered_burns = sorted(
        (burn for burn in burns if -1.0e-12 <= burn.time_s <= end_time_s + 1.0e-12),
        key=lambda item: item.time_s,
    )
    for previous, current in pairwise(ordered_burns):
        if current.time_s < previous.time_s - 1.0e-12:
            raise DynamicsError("Burn epochs must be non-decreasing")

    state = np.concatenate(
        (
            pursuer_initial.r,
            pursuer_initial.v,
            target_initial.r,
            target_initial.v,
        )
    ).astype(float)
    times: list[float] = [0.0]
    relative_positions: list[np.ndarray] = [state[0:3] - state[6:9]]
    relative_velocities: list[np.ndarray] = [state[3:6] - state[9:12]]
    pursuer_positions: list[np.ndarray] = [state[0:3].copy()]
    pursuer_velocities: list[np.ndarray] = [state[3:6].copy()]

    grouped: dict[float, list[Burn]] = {}
    for burn in ordered_burns:
        epoch = float(np.clip(burn.time_s, 0.0, end_time_s))
        grouped.setdefault(epoch, []).append(burn)

    def append_sample(time_s: float, pair_state: np.ndarray) -> None:
        times.append(float(time_s))
        relative_positions.append(pair_state[0:3] - pair_state[6:9])
        relative_velocities.append(pair_state[3:6] - pair_state[9:12])
        pursuer_positions.append(pair_state[0:3].copy())
        pursuer_velocities.append(pair_state[3:6].copy())

    cursor = 0.0
    boundaries = sorted({*grouped, float(end_time_s)})
    for boundary in boundaries:
        if boundary > cursor + 1.0e-13:
            try:
                solution = solve_ivp(
                    _two_body_pair_derivative,
                    (cursor, boundary),
                    state,
                    args=(mu_km3_s2,),
                    method="DOP853",
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step_s,
                )
            except (ValueError, FloatingPointError, OverflowError) as exc:
                raise DynamicsError(f"Trajectory integration failed: {exc}") from exc
            if not solution.success:
                raise DynamicsError(f"Trajectory integration failed: {solution.message}")
            # The first point duplicates the already-recorded segment start.
            for index in range(1, solution.t.size):
                append_sample(float(solution.t[index]), solution.y[:, index])
            state = solution.y[:, -1].copy()
            cursor = boundary

        burns_here = grouped.get(boundary, [])
        if burns_here:
            # Ensure a pre-burn sample exists even if multiple impulses share an epoch.
            if abs(times[-1] - boundary) > 1.0e-10:
                append_sample(boundary, state)
            for burn in burns_here:
                state[3:6] += np.asarray(burn.dv_inertial_km_s, dtype=float)
                append_sample(boundary, state)

    # end_time_s == 0 with no burn already has the initial sample.
    return (
        np.asarray(times, dtype=float),
        np.asarray(relative_positions, dtype=float),
        np.asarray(relative_velocities, dtype=float),
        np.asarray(pursuer_positions, dtype=float),
        np.asarray(pursuer_velocities, dtype=float),
    )


def sample_relative_trajectory_sequential(
    pursuer_initial: StateVector,
    target_initial: StateVector,
    burns: tuple[Burn, ...],
    end_time_s: float,
    mu_km3_s2: float,
    *,
    max_step_s: float,
    rtol: float = 2.0e-11,
    atol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The relative samples alone, for callers that do not need the pursuer."""
    times, positions, velocities, _, _ = sample_pair_trajectory_sequential(
        pursuer_initial,
        target_initial,
        burns,
        end_time_s,
        mu_km3_s2,
        max_step_s=max_step_s,
        rtol=rtol,
        atol=atol,
    )
    return times, positions, velocities
