"""Capped N-impulse search on the ranking score, for hits and near misses.

`multi_burn` closes its last leg with Lambert, so the hit is built in. Under
the ranking caps (2026/10/06: 0.25 km/s per impulse, 3 km/s in total) that
completion almost never fits, and a review found the one-, two- and
multi-burn searches returning only the zero-burn candidate on the synthetic
scripts. Here every impulse is free and capped, and a miss is scored on its
closest approach as eq. (2)-(3) score it.

Decision variables are N burn epochs, N impulses and an aim epoch t_f. The
pursuer flies two-body, or in a PursuerModel when the setup carries one
(`polish_in_model`); the target comes from a `TargetTrack`, either two-body
or the GMAT ephemeris. Each start runs SLSQP in phases:

1. approach: minimise log of the miss at t_f;
2. miss: maximise the distance and delta-v terms (score block only);
3. hit: maximise the time and delta-v terms with miss(t_f) <= the aim radius.

Without a score block, phase 3 minimises delta-v for the hit instead, and a
start that does not get inside minimises delta-v at the distance it reached:
the distance term first, then delta-v. The best few results then get one
longer phase.

Gradients come from state transition matrices chained across the burns:
two-body, or the model's own steps differenced. Caps, spacing, the burn
window and the radius floor are SLSQP constraints while searching and hard
walls afterwards: every result is repaired onto the caps and spacing, flown
again, measured over [0, T_max] and dropped if any wall fails. The radius
floor covers the whole arc to T_max.

Everything here is analytical; GMAT is the authority.
"""

from __future__ import annotations

import ctypes
import dataclasses
import math
import warnings
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import qmc

from .config import OrbitWinConfig, ScoreConfig, validate_score_for_rules
from .conic import arc_minimum_radius, coast, coast_many, gravity, periapsis_radius_gradient
from .constraints import apply_constraints
from .ephemeris import Ephemeris
from .errors import DynamicsError
from .event_detection import (
    _distance_squared_polynomial,
    _first_nonpositive_u,
    _hermite_coefficients,
    _real_unit_roots,
)
from .frames import inertial_to_script_components
from .lambert import solve_lambert_all
from .models import (
    Burn,
    Candidate,
    Scenario,
    StateVector,
    as_vector3,
    burn_frame,
    burn_may_follow_alignment,
    clock_stamps,
    head_start_lead_s,
    head_start_status,
    template_maneuvers,
)
from .pursuer_model import Arcs, PursuerModel
from .pursuer_model import arc_minimum_radius as model_arc_minimum_radius
from .pursuer_model import fly as pursuer_fly
from .rules import interception_rules
from .score import (
    HEAD_START_ALIGNMENT_KEY,
    _dv_for_score,
    _logistic_descending,
    apply_score,
    distance_term,
    distance_term_slope,
    ranking_key,
    score_candidate,
    score_stamp,
)
from .search import ParallelSearchError
from .trajectory import head_start_alignment_distance_km

#: Wall tolerances, as in multi_burn: a magnitude scaled onto the cap measures
#: a few ulp above it.
_CAP_TOLERANCE_KM_S = 1.0e-12
_TIME_TOLERANCE_S = 1.0e-9
#: Margins each start is refined under, added to the altitude floor in
#: SLSQP's constraint only; the hard wall stays the floor itself. Under the
#: section 7 model GMAT flew the R3 and R4 rehearsal winners 4.5 to 7.3 km
#: below the floor their two-body plans held, which 10 km covers. Margin 0 is
#: opt-in (--force-model-margins-km 0 10). With VNB burns, the default frame,
#: R4's margin-0 winners replay at 193.7 km / 23.11 and 95.4 km altitude
#: against margin 10's 46.5 km / 48.04, and R5's margin-0 best (a 13-burn
#: 88 km miss, 28.32 analytically) at 113.7 km / 23.46, under the zero
#: burn's 23.67; only with EarthMJ2000Eq burns does it beat it, at 108.2 km /
#: 24.40. Both margins cost 1.7 times the search time. Choosing between them
#: on GMAT's score is planned for the pipeline (GMAT_IN_THE_LOOP.md step 4),
#: not done yet: solve does not run this search.
FORCE_MODEL_MARGINS_KM = (10.0,)
#: Kept between the radius floor and what SLSQP is asked to hold, so a result
#: that converges onto the constraint still clears the wall when re-flown.
_FLOOR_MARGIN_KM = 0.05
#: Seconds per unit of a scaled epoch; impulses are scaled by the cap.
_TIME_SCALE_S = 1000.0
#: The aim epoch has to come at least this long after the last burn.
_AIM_AFTER_LAST_BURN_S = 1.0
#: Impulses at or under this are dropped from a result: 0.1 mm/s.
_NEGLIGIBLE_KM_S = 1.0e-7
#: Under this a polished copy is tried without the burn: 1 m/s.
_DUST_KM_S = 1.0e-3
#: What one more maneuver costs when results are ordered; see solution_key.
_POINTS_PER_BURN = 1.0e-3
_KM_S_PER_BURN = 1.0e-4
#: Without a score block a miss is ranked on distance first; 1 m per burn there.
_KM_PER_BURN = 1.0e-3
#: How far past its ends the target ephemeris may be read, as ephemeris.py allows.
_EPHEMERIS_SLACK_S = 1.0e-6
#: Central-difference steps for the arc STMs.
_FD_POSITION_KM = 1.0e-3
_FD_VELOCITY_KM_S = 1.0e-6


# --------------------------------------------------------------------------
# Target


class TargetTrack(Protocol):
    """Where the target is. It never maneuvers, so this is fixed per scenario."""

    def state(self, time_s: float) -> tuple[np.ndarray, np.ndarray]: ...

    def states(self, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class ConicTarget:
    """Two-body target, for scenarios without GMAT and for tests."""

    r0_km: tuple[float, float, float]
    v0_km_s: tuple[float, float, float]
    mu_km3_s2: float

    def state(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        return coast(np.array(self.r0_km), np.array(self.v0_km_s), time_s, self.mu_km3_s2)

    def states(self, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times_s, dtype=float)
        count = len(times)
        return coast_many(
            np.tile(self.r0_km, (count, 1)),
            np.tile(self.v0_km_s, (count, 1)),
            times,
            self.mu_km3_s2,
        )


@dataclass(frozen=True, slots=True)
class EphemerisTarget:
    """The GMAT target ephemeris, Hermite between its rows as `Ephemeris` does.

    The run may stop up to 1e-6 s short of T_max, which target_ephemeris_from_gmat
    accepts (measured on the R2 rehearsal script); the end rows' cubic covers it.
    """

    ephemeris: Ephemeris

    def state(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        r, v = self.states(np.array([float(time_s)]))
        return r[0], v[0]

    def states(self, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times_s, dtype=float)
        grid = self.ephemeris.times_s
        if np.any(times < grid[0] - _EPHEMERIS_SLACK_S) or np.any(
            times > grid[-1] + _EPHEMERIS_SLACK_S
        ):
            raise DynamicsError("a requested epoch is outside the target ephemeris")
        index = np.clip(np.searchsorted(grid, times, "right") - 1, 0, len(grid) - 2)
        dt = (grid[index + 1] - grid[index])[:, None]
        u = ((times - grid[index]) / dt[:, 0])[:, None]
        p0 = self.ephemeris.positions_km[index]
        v0 = self.ephemeris.velocities_km_s[index]
        p1 = self.ephemeris.positions_km[index + 1]
        v1 = self.ephemeris.velocities_km_s[index + 1]
        a1 = dt * v0
        a2 = -3.0 * p0 - 2.0 * dt * v0 + 3.0 * p1 - dt * v1
        a3 = 2.0 * p0 + dt * v0 - 2.0 * p1 + dt * v1
        position = p0 + a1 * u + a2 * u * u + a3 * u**3
        velocity = (a1 + 2.0 * a2 * u + 3.0 * a3 * u * u) / dt
        return position, velocity


@dataclass(frozen=True, slots=True)
class DelayedTarget:
    """A track on a clock that starts `delay_s` earlier: the head-start search's,
    from the pursuer's epoch. The target is not read before its own epoch."""

    track: Any
    delay_s: float

    def state(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        r, v = self.states(np.array([float(time_s)]))
        return r[0], v[0]

    def states(self, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times_s, dtype=float) - self.delay_s
        if np.any(times < -_EPHEMERIS_SLACK_S):
            raise DynamicsError("the target is read before its epoch")
        result: tuple[np.ndarray, np.ndarray] = self.track.states(np.maximum(times, 0.0))
        return result


def conic_target(scenario: Scenario) -> ConicTarget:
    state = scenario.target.initial_state
    return ConicTarget(state.r_km, state.v_km_s, scenario.mu_km3_s2)


# --------------------------------------------------------------------------
# Walls


@dataclass(frozen=True, slots=True)
class RankingLimits:
    """The walls, read off the policy once."""

    burn_cap_km_s: float
    total_cap_km_s: float | None
    separation_s: float
    first_burn_floor_s: float
    last_burn_ceiling_s: float
    radius_floor_km: float
    horizon_s: float
    intercept_radius_km: float
    #: Slots a template that already maneuvers exposes; None when it is open.
    slot_count: int | None
    #: The shortest first segment the renderer writes, plus 1 µs. With the
    #: first-burn floor at 0 after an alignment, a first burn is at 0 or past this.
    render_floor_s: float = 1.0e-3 + 1.0e-6
    #: A template that maneuvers right after its alignment: slot 0 fires at t = 0.
    first_slot_at_zero: bool = False
    #: policy.allow_zero_vector_unused_slots: whether a slot may carry an exact zero.
    zero_vector_slots: bool = True
    #: Where the target and the score start: the head start's length on its clock.
    window_start_s: float = 0.0
    #: The alignment's end inside the burn window, on the head start's clock: a
    #: burn is on it or render_floor_s clear of it.
    boundary_s: float | None = None
    #: This clock less the solver's: the head start's length, else 0.
    clock_offset_s: float = 0.0

    @classmethod
    def from_config(
        cls,
        scenario: Scenario,
        config: OrbitWinConfig,
        *,
        altitude_floor_km: float = 100.0,
        correction_margin_km_s: float = 0.0,
    ) -> RankingLimits:
        """`correction_margin_km_s` comes off each cap, for GMAT's correction."""
        policy = config.policy
        total = policy.total_dv_safe_limit_km_s
        # The renderer holds every Propagate segment, the last included, to
        # min_propagate_duration_s, with no tolerance; 1 us keeps a schedule
        # on that bound from rounding under it.
        segment_s = policy.min_propagate_duration_s + 1.0e-6
        return cls(
            burn_cap_km_s=policy.dv_safe_limit_km_s - correction_margin_km_s,
            total_cap_km_s=None if total is None else total - correction_margin_km_s,
            separation_s=max(policy.safe_burn_separation_s, segment_s),
            # After an alignment the first burn may be the alignment's own end.
            first_burn_floor_s=0.0 if burn_may_follow_alignment(scenario) else segment_s,
            last_burn_ceiling_s=scenario.max_time_s - max(policy.final_padding_s, segment_s),
            radius_floor_km=policy.central_body_radius_km + altitude_floor_km,
            horizon_s=scenario.max_time_s,
            intercept_radius_km=policy.intercept_radius_km,
            slot_count=len(scenario.burn_slots) if template_maneuvers(scenario) else None,
            render_floor_s=segment_s,
            first_slot_at_zero=(
                template_maneuvers(scenario)
                and bool(scenario.propagate_segments)
                and scenario.propagate_segments[0].implicit
            ),
            zero_vector_slots=policy.allow_zero_vector_unused_slots,
        )

    @classmethod
    def head_start(
        cls,
        scenario: Scenario,
        config: OrbitWinConfig,
        *,
        altitude_floor_km: float = 100.0,
        correction_margin_km_s: float = 0.0,
    ) -> RankingLimits:
        """The walls on the head start's clock, t = 0 at the pursuer's epoch.

        Solver t = 0 is lead_s on it. Every burn is a segment clear of the
        pursuer's epoch, and on or a segment clear of the alignment's end; the
        target and the score start there.
        """
        lead_s = head_start_lead_s(scenario)
        if lead_s <= 0.0:
            raise DynamicsError(f"no head start here: {head_start_status(scenario) or 'off'}")
        solver = cls.from_config(
            scenario, config, altitude_floor_km=altitude_floor_km,
            correction_margin_km_s=correction_margin_km_s,
        )
        return dataclasses.replace(
            solver,
            first_burn_floor_s=solver.render_floor_s,
            last_burn_ceiling_s=solver.last_burn_ceiling_s + lead_s,
            horizon_s=solver.horizon_s + lead_s,
            window_start_s=lead_s,
            boundary_s=lead_s,
            clock_offset_s=lead_s,
        )

    @property
    def max_burns(self) -> int:
        """As many as fit a separation apart in the burn window, or the slot count of
        a template that fixes it. The caps bound how much each burn and all of them
        spend, not how many there are."""
        window_s = self.last_burn_ceiling_s - self.first_burn_floor_s
        by_window = max(0, 1 + math.floor(window_s / self.separation_s + 1.0e-9))
        return by_window if self.slot_count is None else min(by_window, self.slot_count)

    def first_burn_floor_for(self, burn_count: int) -> float:
        """Unused leading slots of a slotted template come first, a separation apart."""
        unused = 0 if self.slot_count is None else self.slot_count - burn_count
        base = 0.0 if self.first_slot_at_zero else self.first_burn_floor_s
        return base + unused * self.separation_s

    def pins_first_burn(self, burn_count: int) -> bool:
        """Whether the first burn is slot 0 of a template that fires it at t = 0."""
        return self.first_slot_at_zero and burn_count == self.slot_count

    def slot_indices_for(self, burn_count: int) -> range:
        if self.slot_count is None:
            return range(burn_count)
        return range(self.slot_count - burn_count, self.slot_count)


def wall_violations(
    times_s: Sequence[float], impulses_km_s: np.ndarray, limits: RankingLimits
) -> list[str]:
    """Every wall except the radius floor, which needs the flight."""
    found: list[str] = []
    count = len(times_s)
    if count > limits.max_burns:
        found.append(f"{count} burns, more than {limits.max_burns}")
    magnitudes = [float(np.linalg.norm(impulse)) for impulse in impulses_km_s]
    for index, magnitude in enumerate(magnitudes):
        if not math.isfinite(magnitude) or magnitude > limits.burn_cap_km_s + _CAP_TOLERANCE_KM_S:
            found.append(f"burn {index}: |dv| {magnitude!r} km/s over {limits.burn_cap_km_s}")
    if limits.total_cap_km_s is not None and sum(magnitudes) > (
        limits.total_cap_km_s + _CAP_TOLERANCE_KM_S
    ):
        found.append(f"total {sum(magnitudes)!r} km/s over {limits.total_cap_km_s}")
    if count:
        if not all(math.isfinite(t) for t in times_s):
            found.append("a burn epoch is not finite")
            return found
        if limits.pins_first_burn(count) and times_s[0] != 0.0:
            found.append(f"first burn at {times_s[0]!r} s; the template fires slot 0 at t = 0")
        if times_s[0] < limits.first_burn_floor_for(count) - _TIME_TOLERANCE_S:
            found.append(f"first burn at {times_s[0]!r} s, before the floor")
        elif (
            limits.first_burn_floor_s == 0.0
            and _TIME_TOLERANCE_S < times_s[0] < limits.render_floor_s - _TIME_TOLERANCE_S
        ):
            found.append(f"first burn at {times_s[0]!r} s, between t = 0 and the render floor")
        if times_s[-1] > limits.last_burn_ceiling_s + _TIME_TOLERANCE_S:
            found.append(f"last burn at {times_s[-1]!r} s, after {limits.last_burn_ceiling_s}")
        if limits.boundary_s is not None:
            for index, time_s in enumerate(times_s):
                if _TIME_TOLERANCE_S < abs(time_s - limits.boundary_s) < (
                    limits.render_floor_s - _TIME_TOLERANCE_S
                ):
                    found.append(f"burn {index} at {time_s!r} s, inside the alignment's end")
        for index in range(1, count):
            gap = times_s[index] - times_s[index - 1]
            if gap < limits.separation_s - _TIME_TOLERANCE_S:
                found.append(f"burns {index - 1} and {index} are {gap!r} s apart")
    return found


def repair(
    times_s: Sequence[float], impulses_km_s: np.ndarray, limits: RankingLimits
) -> tuple[list[float], np.ndarray]:
    """Push a converged schedule onto the caps and the spacing it nearly holds.

    SLSQP meets its constraints to about 1e-10, so the moves are that small.
    What this cannot fix -- a schedule past the ceiling, a dive -- the walls
    then refuse.
    """
    count = len(times_s)
    times = [float(t) for t in times_s]
    if count:
        times[0] = max(times[0], limits.first_burn_floor_for(count))
        if limits.pins_first_burn(count):
            times[0] = 0.0
        if limits.first_burn_floor_s == 0.0 and times[0] < limits.render_floor_s:
            floor = limits.render_floor_s
            times[0] = 0.0 if times[0] < 0.5 * floor else floor
        for index in range(count):
            if index:
                times[index] = max(times[index], times[index - 1] + limits.separation_s)
            if limits.boundary_s is not None:
                # Forward, onto the alignment's end or a segment past it.
                lag = times[index] - limits.boundary_s
                if -limits.render_floor_s < lag < 0.0:
                    times[index] = limits.boundary_s
                elif 0.0 < lag < limits.render_floor_s:
                    times[index] = limits.boundary_s + limits.render_floor_s
    impulses = np.array(impulses_km_s, dtype=float).reshape(count, 3)
    for index in range(count):
        magnitude = float(np.linalg.norm(impulses[index]))
        if magnitude > limits.burn_cap_km_s:
            impulses[index] *= limits.burn_cap_km_s / magnitude
    total = float(sum(np.linalg.norm(impulse) for impulse in impulses))
    if limits.total_cap_km_s is not None and total > limits.total_cap_km_s:
        impulses *= limits.total_cap_km_s / total
    return times, impulses


# --------------------------------------------------------------------------
# Flight and exact measurement


@dataclass(frozen=True, slots=True)
class Flight:
    """The pursuer under a schedule: states either side of every burn."""

    times_s: np.ndarray
    impulses_km_s: np.ndarray
    initial_r: np.ndarray
    initial_v: np.ndarray
    #: (N, 3) each: before and after the impulse.
    r_at_burns: np.ndarray
    v_before: np.ndarray
    v_after: np.ndarray
    mu: float
    #: (epoch, Δr, Δv) the state takes on there: the head start's defect at the
    #: alignment's end (SearchSetup.jump); before a burn at that epoch.
    jump: tuple[float, np.ndarray, np.ndarray] | None = None

    def arc_start(self, index: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Arc `index` (0 before the first burn) from where its two-body coast begins:
        past the jump when it crosses it."""
        r, v, t = (
            (self.initial_r, self.initial_v, 0.0) if index == 0
            else (self.r_at_burns[index - 1], self.v_after[index - 1],
                  float(self.times_s[index - 1]))
        )
        end = float(self.times_s[index]) if index < len(self.times_s) else math.inf
        if self.jump is not None and t < self.jump[0] <= end:
            r, v = coast(r, v, self.jump[0] - t, self.mu)
            r, v, t = r + self.jump[1], v + self.jump[2], self.jump[0]
        return r, v, t

    def state(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        index = int(np.searchsorted(self.times_s, time_s, "right"))
        r, v, t = self.arc_start(index)
        if t > time_s:  # before the jump on an arc that crosses it
            r, v, t = (
                (self.initial_r, self.initial_v, 0.0) if index == 0
                else (self.r_at_burns[index - 1], self.v_after[index - 1],
                      float(self.times_s[index - 1]))
            )
        return coast(r, v, time_s - t, self.mu)


@dataclass(frozen=True, slots=True)
class ModelFlight:
    """The pursuer under a schedule in a PursuerModel, every arc flown to T_max's end.

    Arc j starts at 0 (j = 0) or at burn j - 1 and is kept node by node; a
    state between nodes is Hermite, as Flight's is exact.
    """

    times_s: np.ndarray
    impulses_km_s: np.ndarray
    initial_r: np.ndarray
    initial_v: np.ndarray
    r_at_burns: np.ndarray
    v_before: np.ndarray
    v_after: np.ndarray
    arcs: tuple[Arcs, ...]
    starts_s: np.ndarray

    def state(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        r, v = self.states(np.array([float(time_s)]))
        return r[0], v[0]

    def states(self, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times_s, dtype=float)
        arc = np.searchsorted(self.times_s, times, "right")
        r = np.empty((len(times), 3))
        v = np.empty((len(times), 3))
        for index in np.unique(arc):
            chosen = arc == index
            r[chosen], v[chosen] = self.arcs[index].sample(0, times[chosen] - self.starts_s[index])
        return r, v

    def minimum_radius(self) -> float:
        return min(model_arc_minimum_radius(arc)[0] for arc in self.arcs)


def fly_model(
    times_s: Sequence[float],
    impulses_km_s: np.ndarray,
    initial: StateVector,
    model: PursuerModel,
    horizon_s: float,
) -> ModelFlight:
    count = len(times_s)
    impulses = np.asarray(impulses_km_s, dtype=float).reshape(count, 3)
    flown = pursuer_fly(
        model, np.asarray(initial.r), np.asarray(initial.v), np.asarray(times_s, dtype=float),
        impulses, horizon_s,
    )
    return ModelFlight(
        np.asarray(times_s, dtype=float), impulses, np.asarray(initial.r, dtype=float),
        np.asarray(initial.v, dtype=float), flown.r_at_burns, flown.v_before, flown.v_after,
        flown.arcs, np.array([0.0, *(float(t) for t in times_s)]),
    )


def _fly(
    setup: SearchSetup, times_s: Sequence[float], impulses_km_s: np.ndarray
) -> Flight | ModelFlight:
    if setup.pursuer_model is None:
        return fly(times_s, impulses_km_s, setup.initial, setup.mu, setup.jump_at())
    return fly_model(
        times_s, impulses_km_s, setup.initial, setup.pursuer_model, setup.limits.horizon_s
    )


def fly(
    times_s: Sequence[float],
    impulses_km_s: np.ndarray,
    initial: StateVector,
    mu: float,
    jump: tuple[float, np.ndarray, np.ndarray] | None = None,
) -> Flight:
    r = np.asarray(initial.r, dtype=float)
    v = np.asarray(initial.v, dtype=float)
    count = len(times_s)
    impulses = np.asarray(impulses_km_s, dtype=float).reshape(count, 3)
    r_at = np.zeros((count, 3))
    before = np.zeros((count, 3))
    after = np.zeros((count, 3))
    now = 0.0
    position, velocity = r, v
    pending = jump
    for index, time_s in enumerate(times_s):
        if pending is not None and now < pending[0] <= float(time_s):
            position, velocity = coast(position, velocity, pending[0] - now, mu)
            position, velocity, now = position + pending[1], velocity + pending[2], pending[0]
            pending = None
        position, velocity = coast(position, velocity, float(time_s) - now, mu)
        r_at[index], before[index] = position, velocity
        velocity = velocity + impulses[index]
        after[index] = velocity
        now = float(time_s)
    return Flight(np.asarray(times_s, dtype=float), impulses, r, v, r_at, before, after, mu, jump)


@dataclass(frozen=True, slots=True)
class Measurement:
    """The ranking-score inputs of one flight over [0, T_max]."""

    minimum_distance_km: float
    minimum_distance_time_s: float
    #: First entry into the interception sphere; None for a miss.
    intercept_time_s: float | None
    distance_at_horizon_km: float
    #: Smallest pursuer radius over the whole arc to T_max.
    minimum_radius_km: float


def minimum_radius_over_horizon(flight: Flight | ModelFlight, horizon_s: float) -> float:
    if isinstance(flight, ModelFlight):
        return flight.minimum_radius()
    starts_r = [flight.initial_r, *flight.r_at_burns]
    starts_v = [flight.initial_v, *flight.v_after]
    epochs = [0.0, *(float(t) for t in flight.times_s)]
    ends = [*epochs[1:], horizon_s]
    pieces = []
    for index, (r, v, begin, end) in enumerate(zip(starts_r, starts_v, epochs, ends, strict=True)):
        if flight.jump is not None and begin < flight.jump[0] < end:
            # Up to the alignment's end two-body, past it from where the jump puts it.
            pieces.append((r, v, flight.jump[0] - begin))
            r, v, begin = flight.arc_start(index)
        pieces.append((r, v, end - begin))
    smallest = math.inf
    for r, v, duration in pieces:
        r_end, _ = coast(r, v, duration, flight.mu)
        smallest = min(smallest, arc_minimum_radius(r, v, duration, r_end, flight.mu)[0])
    return smallest


@dataclass(frozen=True, slots=True)
class MeasurementGrid:
    """Knots the relative motion is Hermite-interpolated between."""

    times_s: np.ndarray
    target_r: np.ndarray
    target_v: np.ndarray

    @classmethod
    def build(
        cls, target: TargetTrack, horizon_s: float, step_s: float, start_s: float = 0.0
    ) -> MeasurementGrid:
        count = max(2, math.ceil((horizon_s - start_s) / step_s) + 1)
        times = np.linspace(start_s, horizon_s, count)
        r, v = target.states(times)
        return cls(times, r, v)


def relative_knots(
    flight: Flight | ModelFlight, target: TargetTrack, grid: MeasurementGrid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative state at the grid and either side of every burn, in time order.

    At a burn epoch the pre-burn knot comes first, so the segment ending
    there is interpolated with the velocity it actually ends with.
    """
    burn_times = flight.times_s
    arc = np.searchsorted(burn_times, grid.times_s, "right") - 1
    start_r = np.vstack([flight.initial_r[None, :], flight.r_at_burns])
    start_v = np.vstack([flight.initial_v[None, :], flight.v_after])
    start_t = np.concatenate([[0.0], burn_times])
    if isinstance(flight, Flight) and flight.jump is not None:
        # The grid starts at the jump, so an arc that crosses it is read from past it.
        for index in np.unique(arc + 1):
            start_r[index], start_v[index], start_t[index] = flight.arc_start(int(index))
    if isinstance(flight, ModelFlight):
        p_r, p_v = flight.states(grid.times_s)
    else:
        p_r, p_v = coast_many(
            start_r[arc + 1], start_v[arc + 1], grid.times_s - start_t[arc + 1], flight.mu
        )
    times = grid.times_s
    rel_r = p_r - grid.target_r
    rel_v = p_v - grid.target_v
    # A head-start burn is before the target's epoch, outside the window.
    inside = burn_times >= grid.times_s[0]
    burn_times = burn_times[inside]
    if len(burn_times):
        t_r, t_v = target.states(burn_times)
        at = flight.r_at_burns[inside]
        times = np.concatenate([times, burn_times, burn_times])
        rel_r = np.vstack([rel_r, at - t_r, at - t_r])
        rel_v = np.vstack([rel_v, flight.v_before[inside] - t_v, flight.v_after[inside] - t_v])
        rank = np.concatenate(
            [np.ones(len(grid.times_s)), np.zeros(len(burn_times)), 2 * np.ones(len(burn_times))]
        )
        order = np.lexsort((rank, times))
        times, rel_r, rel_v = times[order], rel_r[order], rel_v[order]
    return times, rel_r, rel_v


def measure(
    flight: Flight | ModelFlight,
    target: TargetTrack,
    grid: MeasurementGrid,
    horizon_s: float,
    radius_km: float,
) -> Measurement:
    """Closest approach and first entry over [0, T_max], with no sampling gap.

    The relative state is known exactly at the grid and either side of every
    burn. Between knots it is a cubic Hermite, whose minimum and crossing are
    found as event_detection finds them; a segment that the arc-length bound
    proves cannot beat the best so far is skipped. The winning minimum and the
    first crossing are then polished on exact states.
    """
    times, rel_r, rel_v = relative_knots(flight, target, grid)
    distances = np.linalg.norm(rel_r, axis=1)
    dt = np.diff(times)
    speeds = np.linalg.norm(rel_v, axis=1)
    reach = 1.5 * np.linalg.norm(np.diff(rel_r, axis=0), axis=1) + dt * (speeds[:-1] + speeds[1:])
    floor = np.maximum(distances[:-1], distances[1:]) - reach
    live = dt > 1.0e-9

    def exact_distance(time_s: float) -> float:
        p, _ = flight.state(time_s)
        q, _ = target.state(time_s)
        return float(np.linalg.norm(p - q))

    def segment_coefficients(index: int) -> np.ndarray:
        return _hermite_coefficients(
            rel_r[index], rel_v[index], rel_r[index + 1], rel_v[index + 1], float(dt[index])
        )

    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    best_time = float(times[best_index])
    best_segment: int | None = None
    for index in np.argsort(np.where(live, floor, np.inf), kind="stable"):
        if not live[index] or floor[index] >= best_distance:
            break
        poly = _distance_squared_polynomial(segment_coefficients(int(index)))
        derivative = np.array([power * poly[power] for power in range(1, len(poly))])
        for u in _real_unit_roots(derivative):
            value = math.sqrt(max(0.0, float(np.polynomial.polynomial.polyval(u, poly))))
            if value < best_distance:
                best_distance = value
                best_time = float(times[index] + u * dt[index])
                best_segment = int(index)
    if best_segment is not None:
        low, high = float(times[best_segment]), float(times[best_segment + 1])
        polished = minimize_scalar(
            exact_distance, bounds=(low, high), method="bounded", options={"xatol": 1.0e-7}
        )
        at_guess = exact_distance(best_time)
        if float(polished.fun) <= at_guess:
            best_distance, best_time = float(polished.fun), float(polished.x)
        else:
            best_distance = at_guess

    intercept: float | None = None
    if best_distance <= radius_km:
        if distances[0] <= radius_km:
            intercept = float(times[0])
        else:
            for index in np.flatnonzero(live & (floor <= radius_km)):
                intercept = _first_entry_in_segment(
                    int(index), times, dt, distances, segment_coefficients, exact_distance,
                    radius_km,
                )
                if intercept is not None:
                    break
            if intercept is None:
                # The polished minimum is inside, so the crossing precedes it.
                intercept = _bisect_entry(exact_distance, float(times[0]), best_time, radius_km)
    radius = minimum_radius_over_horizon(flight, horizon_s)
    return Measurement(
        minimum_distance_km=best_distance,
        minimum_distance_time_s=best_time,
        intercept_time_s=intercept,
        distance_at_horizon_km=float(distances[-1]),
        minimum_radius_km=radius,
    )


def _bisect_entry(
    distance: Any, low: float, high: float, radius_km: float, iterations: int = 80
) -> float:
    """Earliest crossing in [low, high], given d(low) > r >= d(high), by bisection."""
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if distance(middle) <= radius_km:
            high = middle
        else:
            low = middle
        if high - low < 1.0e-9:
            break
    return high


def _first_entry_in_segment(
    index: int,
    times: np.ndarray,
    dt: np.ndarray,
    distances: np.ndarray,
    coefficients: Any,
    exact_distance: Any,
    radius_km: float,
) -> float | None:
    t0 = float(times[index])
    if distances[index] <= radius_km:
        return t0
    poly = _distance_squared_polynomial(coefficients(index))
    crossing = poly.copy()
    crossing[0] -= radius_km * radius_km
    derivative = np.array([power * poly[power] for power in range(1, len(poly))])
    probes = sorted(
        {*_real_unit_roots(crossing), *(u for u in _real_unit_roots(derivative) if 0 < u < 1), 1.0}
    )
    inside = [u for u in probes if np.polynomial.polynomial.polyval(u, crossing) <= 0.0]
    if not inside:
        return None
    u_cross = _first_nonpositive_u(crossing, min(inside))
    # Confirm on exact states: Hermite can graze where the true path does not.
    for u in (u_cross, min(inside), *inside):
        t_inside = t0 + u * float(dt[index])
        if exact_distance(t_inside) <= radius_km:
            return _bisect_entry(exact_distance, t0, t_inside, radius_km)
    return None


# --------------------------------------------------------------------------
# Local optimisation


@dataclass(frozen=True, slots=True)
class SearchSetup:
    """Everything a worker needs, picklable."""

    initial: StateVector
    mu: float
    target: TargetTrack
    limits: RankingLimits
    score: ScoreConfig | None
    #: Where a hit aims: the miss at t_f is held under this, inside the sphere.
    aim_radius_km: float
    grid: MeasurementGrid
    #: Added to the floor in SLSQP's constraint; see FORCE_MODEL_MARGINS_KM.
    force_model_margins_km: tuple[float, ...]
    #: SLSQP iterations per phase for every start, and for the polished few.
    screen_iterations: int
    polish_iterations: int
    #: Scenario.rules, whose distance term the miss phase maximises.
    rules: str
    #: Scenario.time_origin_offset_s: T_team is the solver time plus this.
    #: score stays on the origin's clock, which score_candidate moves itself.
    time_origin_offset_s: float = 0.0
    #: Flies the pursuer instead of two-body when set; the target is `target` either way.
    pursuer_model: PursuerModel | None = None
    #: On the head start's clock: the alignment's end and Scenario.head_start_defect,
    #: which the two-body flight takes on there (models.head_start_jump).
    jump_s: float | None = None
    jump_r_km: tuple[float, float, float] = (0.0, 0.0, 0.0)
    jump_v_km_s: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def jump_at(self) -> tuple[float, np.ndarray, np.ndarray] | None:
        if self.jump_s is None:
            return None
        return self.jump_s, np.asarray(self.jump_r_km), np.asarray(self.jump_v_km_s)


def _stm_batch(
    starts_r: np.ndarray, starts_v: np.ndarray, durations: np.ndarray, mu: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End states and 6x6 STMs of several coasts, by central differences."""
    legs = len(durations)
    steps = np.array([_FD_POSITION_KM] * 3 + [_FD_VELOCITY_KM_S] * 3)
    base = np.hstack([starts_r, starts_v])
    rows = np.repeat(base[:, None, :], 13, axis=1)
    for column in range(6):
        rows[:, 1 + 2 * column, column] += steps[column]
        rows[:, 2 + 2 * column, column] -= steps[column]
    flat = rows.reshape(legs * 13, 6)
    r, v = coast_many(flat[:, :3], flat[:, 3:], np.repeat(durations, 13), mu)
    ends = np.hstack([r, v]).reshape(legs, 13, 6)
    columns = (ends[:, 1::2, :] - ends[:, 2::2, :]) / (2.0 * steps)[None, :, None]
    return ends[:, 0, :3], ends[:, 0, 3:], np.transpose(columns, (0, 2, 1))


@dataclass(slots=True)
class _Evaluation:
    miss_km: float
    miss_gradient: np.ndarray
    floors_km: np.ndarray
    floor_gradients: np.ndarray
    total_dv_km_s: float
    total_dv_gradient: np.ndarray


def _central_derivative(function: Any, x: float, step: float) -> float:
    return float((function(x + step) - function(x - step)) / (2.0 * step))


class _Problem:
    """One burn count: the scaled decision vector, its functions and gradients.

    x = [t_1..t_N / s, u_1..u_N (impulse / cap, 3 each), t_f / s] with
    s = _TIME_SCALE_S. SLSQP asks for the objective and the constraints at the
    same x separately, so the last evaluation is cached.
    """

    def __init__(
        self, setup: SearchSetup, burn_count: int, floor_margin_km: float = 0.0
    ) -> None:
        self.setup = setup
        self.floor_margin_km = floor_margin_km
        self.n = burn_count
        self.cap = setup.limits.burn_cap_km_s
        self.size = 4 * burn_count + 1
        self.first_floor_s = setup.limits.first_burn_floor_for(burn_count)
        self._key: bytes | None = None
        self._value: _Evaluation | None = None

    # -- packing

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        n = self.n
        times = np.asarray(x[:n], dtype=float) * _TIME_SCALE_S
        impulses = np.asarray(x[n : 4 * n], dtype=float).reshape(n, 3) * self.cap
        return times, impulses, float(x[4 * n]) * _TIME_SCALE_S

    def pack(self, times: Sequence[float], impulses: np.ndarray, aim_s: float) -> np.ndarray:
        x = np.concatenate(
            [
                np.asarray(times, dtype=float) / _TIME_SCALE_S,
                np.asarray(impulses, dtype=float).reshape(-1) / self.cap,
                [aim_s / _TIME_SCALE_S],
            ]
        )
        low, high = zip(*self.bounds(), strict=True)
        clipped: np.ndarray = np.clip(x, low, high)
        return clipped

    def bounds(self) -> list[tuple[float, float]]:
        limits = self.setup.limits
        epochs = (self.first_floor_s / _TIME_SCALE_S, limits.last_burn_ceiling_s / _TIME_SCALE_S)
        aim = (
            max(self.first_floor_s + _AIM_AFTER_LAST_BURN_S, limits.window_start_s)
            / _TIME_SCALE_S,
            limits.horizon_s / _TIME_SCALE_S,
        )
        first = (0.0, 0.0) if limits.pins_first_burn(self.n) else epochs
        return [first] + [epochs] * (self.n - 1) + [(-1.0, 1.0)] * (3 * self.n) + [aim]

    # -- evaluation

    def evaluate(self, x: np.ndarray) -> _Evaluation:
        key = np.asarray(x, dtype=float).tobytes()
        if key == self._key and self._value is not None:
            return self._value
        value = self._evaluate(np.asarray(x, dtype=float))
        self._key, self._value = key, value
        return value

    def _evaluate(self, x: np.ndarray) -> _Evaluation:
        if self.setup.pursuer_model is not None:
            return self._evaluate_model(np.asarray(x, dtype=float))
        if self.setup.jump_s is not None:
            return self._evaluate_jump(np.asarray(x, dtype=float))
        setup, n, mu = self.setup, self.n, self.setup.mu
        horizon = setup.limits.horizon_s
        times, impulses, aim_s = self.unpack(x)
        size = self.size

        r_at = np.zeros((n, 3))
        v_before = np.zeros((n, 3))
        v_after = np.zeros((n, 3))
        position, velocity = setup.initial.r, setup.initial.v
        now = 0.0
        for k in range(n):
            position, velocity = coast(position, velocity, float(times[k]) - now, mu)
            r_at[k], v_before[k] = position, velocity
            velocity = velocity + impulses[k]
            v_after[k] = velocity
            now = float(times[k])

        # Legs after each burn, then the aim leg and the leg to the horizon.
        durations = np.concatenate([np.diff(times), [aim_s - times[-1], horizon - times[-1]]])
        leg_r = np.vstack([r_at, r_at[-1:]])
        leg_v = np.vstack([v_after, v_after[-1:]])
        end_r, end_v, stms = _stm_batch(leg_r, leg_v, durations, mu)

        # jacobians[a] = d(state just after burn a) / dx, built forward.
        jacobians = np.zeros((n, 6, size))
        for a in range(n):
            if a:
                jacobians[a] = stms[a - 1] @ jacobians[a - 1]
                jacobians[a, :3, a - 1] -= v_before[a] * _TIME_SCALE_S
                jacobians[a, 3:, a - 1] -= gravity(r_at[a], mu) * _TIME_SCALE_S
            jacobians[a, :3, a] += v_before[a] * _TIME_SCALE_S
            jacobians[a, 3:, a] += gravity(r_at[a], mu) * _TIME_SCALE_S
            jacobians[a, 3:, n + 3 * a : n + 3 * a + 3] += np.eye(3) * self.cap

        target_r, target_v = setup.target.state(aim_s)
        aim_r, aim_v = end_r[n - 1], end_v[n - 1]
        miss_vector = aim_r - target_r
        miss = float(np.linalg.norm(miss_vector))
        direction = miss_vector / max(miss, 1.0e-12)
        aim_jacobian = stms[n - 1][:3] @ jacobians[n - 1]
        aim_jacobian[:, n - 1] -= aim_v * _TIME_SCALE_S
        aim_jacobian[:, 4 * n] += (aim_v - target_v) * _TIME_SCALE_S
        miss_gradient = direction @ aim_jacobian

        # Radius over every arc to the horizon: arc 0 before the first burn,
        # arc j after burn j-1, the last one running to T_max.
        floors = np.zeros(n + 1)
        floor_gradients = np.zeros((n + 1, size))
        value, kind = arc_minimum_radius(
            setup.initial.r, setup.initial.v, float(times[0]), r_at[0], mu
        )
        floors[0] = value
        if kind == "end":
            radial_speed = float(np.dot(r_at[0], v_before[0])) / float(np.linalg.norm(r_at[0]))
            floor_gradients[0, 0] = radial_speed * _TIME_SCALE_S
        for j in range(1, n + 1):
            leg = j - 1 if j < n else n
            start_r, start_v = r_at[j - 1], v_after[j - 1]
            finish_r, finish_v = end_r[leg], end_v[leg]
            value, kind = arc_minimum_radius(start_r, start_v, float(durations[leg]), finish_r, mu)
            floors[j] = value
            if kind == "start":
                state_gradient = np.concatenate([start_r / np.linalg.norm(start_r), np.zeros(3)])
                duration_gradient = 0.0
            elif kind == "periapsis":
                state_gradient = periapsis_radius_gradient(start_r, start_v, mu)
                duration_gradient = 0.0
            else:
                unit = finish_r / np.linalg.norm(finish_r)
                state_gradient = unit @ stms[leg][:3]
                duration_gradient = float(np.dot(unit, finish_v))
            row = state_gradient @ jacobians[j - 1]
            row[j - 1] -= duration_gradient * _TIME_SCALE_S
            if j < n:
                row[j] += duration_gradient * _TIME_SCALE_S
            floor_gradients[j] = row

        magnitudes = np.sqrt(np.einsum("ij,ij->i", impulses, impulses) + 1.0e-18)
        total_gradient = np.zeros(size)
        total_gradient[n : 4 * n] = (impulses / magnitudes[:, None]).reshape(-1) * self.cap
        return _Evaluation(
            miss, miss_gradient, floors, floor_gradients, float(magnitudes.sum()), total_gradient
        )

    def _evaluate_jump(self, x: np.ndarray) -> _Evaluation:
        """_evaluate with the head start's jump as one more node: fixed in time, it
        adds the defect to the state and nothing to the derivatives."""
        setup, n, mu = self.setup, self.n, self.setup.mu
        jump = setup.jump_at()
        assert jump is not None
        horizon = setup.limits.horizon_s
        times, impulses, aim_s = self.unpack(x)
        size = self.size
        # Burn k's decision column, or -1 for the jump; the jump first at a tie.
        order = sorted([(float(t), 1, k) for k, t in enumerate(times)] + [(jump[0], 0, -1)])
        m = n + 1
        column = [k for _, _, k in order]
        r_at = np.zeros((m, 3))
        r_after = np.zeros((m, 3))
        v_before = np.zeros((m, 3))
        v_after = np.zeros((m, 3))
        position, velocity = setup.initial.r, setup.initial.v
        now = 0.0
        for i, (t, _, k) in enumerate(order):
            position, velocity = coast(position, velocity, t - now, mu)
            r_at[i], v_before[i] = position, velocity
            if k < 0:
                position, velocity = position + jump[1], velocity + jump[2]
            else:
                velocity = velocity + impulses[k]
            r_after[i], v_after[i] = position, velocity
            now = t
        node_t = np.array([t for t, _, _ in order])
        durations = np.concatenate([np.diff(node_t), [aim_s - node_t[-1], horizon - node_t[-1]]])
        leg_r = np.vstack([r_after, r_after[-1:]])
        leg_v = np.vstack([v_after, v_after[-1:]])
        end_r, end_v, stms = _stm_batch(leg_r, leg_v, durations, mu)

        jacobians = np.zeros((m, 6, size))
        for i in range(m):
            if i:
                jacobians[i] = stms[i - 1] @ jacobians[i - 1]
                if column[i - 1] >= 0:
                    jacobians[i, :3, column[i - 1]] -= v_before[i] * _TIME_SCALE_S
                    jacobians[i, 3:, column[i - 1]] -= gravity(r_at[i], mu) * _TIME_SCALE_S
            k = column[i]
            if k >= 0:
                jacobians[i, :3, k] += v_before[i] * _TIME_SCALE_S
                jacobians[i, 3:, k] += gravity(r_at[i], mu) * _TIME_SCALE_S
                jacobians[i, 3:, n + 3 * k : n + 3 * k + 3] += np.eye(3) * self.cap

        target_r, target_v = setup.target.state(aim_s)
        aim_r, aim_v = end_r[m - 1], end_v[m - 1]
        miss_vector = aim_r - target_r
        miss = float(np.linalg.norm(miss_vector))
        direction = miss_vector / max(miss, 1.0e-12)
        aim_jacobian = stms[m - 1][:3] @ jacobians[m - 1]
        if column[m - 1] >= 0:
            aim_jacobian[:, column[m - 1]] -= aim_v * _TIME_SCALE_S
        aim_jacobian[:, 4 * n] += (aim_v - target_v) * _TIME_SCALE_S
        miss_gradient = direction @ aim_jacobian

        floors = np.zeros(m + 1)
        floor_gradients = np.zeros((m + 1, size))
        value, kind = arc_minimum_radius(
            setup.initial.r, setup.initial.v, float(node_t[0]), r_at[0], mu
        )
        floors[0] = value
        if kind == "end" and column[0] >= 0:
            radial_speed = float(np.dot(r_at[0], v_before[0])) / float(np.linalg.norm(r_at[0]))
            floor_gradients[0, column[0]] = radial_speed * _TIME_SCALE_S
        for j in range(1, m + 1):
            leg = j - 1 if j < m else m
            start_r, start_v = r_after[j - 1], v_after[j - 1]
            finish_r, finish_v = end_r[leg], end_v[leg]
            value, kind = arc_minimum_radius(start_r, start_v, float(durations[leg]), finish_r, mu)
            floors[j] = value
            if kind == "start":
                state_gradient = np.concatenate([start_r / np.linalg.norm(start_r), np.zeros(3)])
                duration_gradient = 0.0
            elif kind == "periapsis":
                state_gradient = periapsis_radius_gradient(start_r, start_v, mu)
                duration_gradient = 0.0
            else:
                unit = finish_r / np.linalg.norm(finish_r)
                state_gradient = unit @ stms[leg][:3]
                duration_gradient = float(np.dot(unit, finish_v))
            row = state_gradient @ jacobians[j - 1]
            if column[j - 1] >= 0:
                row[column[j - 1]] -= duration_gradient * _TIME_SCALE_S
            if j < m and column[j] >= 0:
                row[column[j]] += duration_gradient * _TIME_SCALE_S
            floor_gradients[j] = row

        magnitudes = np.sqrt(np.einsum("ij,ij->i", impulses, impulses) + 1.0e-18)
        total_gradient = np.zeros(size)
        total_gradient[n : 4 * n] = (impulses / magnitudes[:, None]).reshape(-1) * self.cap
        return _Evaluation(
            miss, miss_gradient, floors, floor_gradients, float(magnitudes.sum()), total_gradient
        )

    def _evaluate_model(self, x: np.ndarray) -> _Evaluation:
        """_evaluate in the pursuer model: each leg flown with its 12 central-difference
        copies, so the STMs and the floors' gradients come from the same steps."""
        setup, n = self.setup, self.n
        model = setup.pursuer_model
        assert model is not None
        horizon = setup.limits.horizon_s
        times, impulses, aim_s = self.unpack(x)
        size = self.size
        offsets = np.array([_FD_POSITION_KM] * 3 + [_FD_VELOCITY_KM_S] * 3)

        def copies(r: np.ndarray, v: np.ndarray) -> np.ndarray:
            rows = np.repeat(np.concatenate([r, v])[None, :], 13, axis=0)
            for column in range(6):
                rows[1 + 2 * column, column] += offsets[column]
                rows[2 + 2 * column, column] -= offsets[column]
            return rows

        def stm(ends: np.ndarray) -> np.ndarray:
            return np.asarray(((ends[1::2] - ends[2::2]) / (2.0 * offsets)[:, None]).T)

        def accel(r: np.ndarray, v: np.ndarray) -> np.ndarray:
            return np.asarray(model.acceleration(r[None, :], v[None, :])[0])

        r_at = np.zeros((n, 3))
        v_before = np.zeros((n, 3))
        v_after = np.zeros((n, 3))
        first = model.propagate(
            np.asarray(setup.initial.r)[None, :], np.asarray(setup.initial.v)[None, :],
            float(times[0]),
        )
        r_at[0], v_before[0] = first.r[-1, 0], first.v[-1, 0]
        v_after[0] = v_before[0] + impulses[0]
        arcs: list[Arcs] = [first]
        stms: list[np.ndarray] = []
        for k in range(n - 1):
            rows = copies(r_at[k], v_after[k])
            arc = model.propagate(
                rows[:, :3], rows[:, 3:], float(times[k + 1] - times[k]), float(times[k])
            )
            ends = np.hstack([arc.r[-1], arc.v[-1]])
            r_at[k + 1], v_before[k + 1] = ends[0, :3], ends[0, 3:]
            v_after[k + 1] = v_before[k + 1] + impulses[k + 1]
            stms.append(stm(ends))
            arcs.append(arc)
        rows = copies(r_at[n - 1], v_after[n - 1])
        last = model.propagate(
            rows[:, :3], rows[:, 3:], float(horizon - times[n - 1]), float(times[n - 1])
        )
        arcs.append(last)
        stms.append(stm(np.hstack([last.r[-1], last.v[-1]])))

        jacobians = np.zeros((n, 6, size))
        for a in range(n):
            pushed = accel(r_at[a], v_before[a]) * _TIME_SCALE_S
            if a:
                jacobians[a] = stms[a - 1] @ jacobians[a - 1]
                jacobians[a, :3, a - 1] -= v_before[a] * _TIME_SCALE_S
                jacobians[a, 3:, a - 1] -= pushed
            jacobians[a, :3, a] += v_before[a] * _TIME_SCALE_S
            jacobians[a, 3:, a] += pushed
            jacobians[a, 3:, n + 3 * a : n + 3 * a + 3] += np.eye(3) * self.cap

        aim_rows_r, aim_rows_v = last.sample_all(aim_s - float(times[n - 1]))
        aim_r, aim_v = aim_rows_r[0], aim_rows_v[0]
        target_r, target_v = setup.target.state(aim_s)
        miss_vector = aim_r - target_r
        miss = float(np.linalg.norm(miss_vector))
        direction = miss_vector / max(miss, 1.0e-12)
        aim_stm = np.asarray(((aim_rows_r[1::2] - aim_rows_r[2::2]) / (2.0 * offsets)[:, None]).T)
        aim_jacobian = aim_stm @ jacobians[n - 1]
        aim_jacobian[:, n - 1] -= aim_v * _TIME_SCALE_S
        aim_jacobian[:, 4 * n] += (aim_v - target_v) * _TIME_SCALE_S
        miss_gradient = direction @ aim_jacobian

        floors = np.zeros(n + 1)
        floor_gradients = np.zeros((n + 1, size))
        value, at_s, kind = model_arc_minimum_radius(first)
        floors[0] = value
        if kind == "end":
            radial_speed = float(np.dot(r_at[0], v_before[0])) / float(np.linalg.norm(r_at[0]))
            floor_gradients[0, 0] = radial_speed * _TIME_SCALE_S
        for j in range(1, n + 1):
            arc = arcs[j]
            value, at_s, kind = model_arc_minimum_radius(arc)
            floors[j] = value
            duration_gradient = 0.0
            if kind == "start":
                start_r = arc.r[0, 0]
                state_gradient = np.concatenate([start_r / np.linalg.norm(start_r), np.zeros(3)])
            elif kind == "interior":
                here_r, _ = arc.sample_all(at_s)
                unit = here_r[0] / np.linalg.norm(here_r[0])
                here_stm = (here_r[1::2] - here_r[2::2]) / (2.0 * offsets)[:, None]
                state_gradient = here_stm @ unit
            else:
                finish_r, finish_v = arc.r[-1, 0], arc.v[-1, 0]
                unit = finish_r / np.linalg.norm(finish_r)
                state_gradient = unit @ stms[j - 1][:3]
                duration_gradient = float(np.dot(unit, finish_v))
            row = state_gradient @ jacobians[j - 1]
            row[j - 1] -= duration_gradient * _TIME_SCALE_S
            if j < n:
                row[j] += duration_gradient * _TIME_SCALE_S
            floor_gradients[j] = row

        magnitudes = np.sqrt(np.einsum("ij,ij->i", impulses, impulses) + 1.0e-18)
        total_gradient = np.zeros(size)
        total_gradient[n : 4 * n] = (impulses / magnitudes[:, None]).reshape(-1) * self.cap
        return _Evaluation(
            miss, miss_gradient, floors, floor_gradients, float(magnitudes.sum()), total_gradient
        )

    # -- score terms, from score.py's own functions

    def _time_term(self, time_s: float) -> float:
        score = self.setup.score
        assert score is not None
        return _logistic_descending(
            25.0, score.kt * (time_s + self.setup.time_origin_offset_s - score.ct)
        )

    def _dv_term(self, total_km_s: float) -> float:
        score = self.setup.score
        assert score is not None
        return _logistic_descending(
            25.0, score.kv * (_dv_for_score(total_km_s, score.delta_v_unit) - score.cv)
        )

    def _distance_term(self, miss_km: float) -> tuple[float, float]:
        """Value and slope, measured from the policy's sphere, which the search aims at."""
        score = self.setup.score
        assert score is not None
        rules = interception_rules(self.setup.rules)
        shifted = miss_km - self.setup.limits.intercept_radius_km + rules.intercept_radius_km
        return (distance_term(shifted, rules, score.cr_km),
                distance_term_slope(shifted, rules, score.cr_km))

    # -- phases

    def objective(self, x: np.ndarray, phase: str) -> tuple[float, np.ndarray]:
        e = self.evaluate(x)
        if phase == "approach":
            scale = e.miss_km * e.miss_km + 1.0
            return 0.5 * math.log(scale), e.miss_km * e.miss_gradient / scale
        if phase in {"hit-dv", "polish"}:
            return e.total_dv_km_s, e.total_dv_gradient
        dv_value = self._dv_term(e.total_dv_km_s)
        dv_slope = _central_derivative(self._dv_term, e.total_dv_km_s, 1.0e-6)
        if phase == "miss":
            distance, slope = self._distance_term(e.miss_km)
            gradient = slope * e.miss_gradient + dv_slope * e.total_dv_gradient
            return -(distance + dv_value) / 50.0, -gradient / 50.0
        if phase == "hit":
            aim_s = float(x[4 * self.n]) * _TIME_SCALE_S
            time_value = self._time_term(aim_s)
            time_slope = _central_derivative(self._time_term, aim_s, 1.0e-3)
            gradient = dv_slope * e.total_dv_gradient
            gradient[4 * self.n] += time_slope * _TIME_SCALE_S
            return -(time_value + dv_value) / 50.0, -gradient / 50.0
        raise ValueError(f"unknown phase {phase!r}")

    def constraints(self, x: np.ndarray, miss_bound_km: float | None) -> np.ndarray:
        e = self.evaluate(x)
        n, limits = self.n, self.setup.limits
        u = np.asarray(x[n : 4 * n]).reshape(n, 3)
        parts: list[Any] = [
            1.0 - np.einsum("ij,ij->i", u, u),
            np.diff(x[:n]) - limits.separation_s / _TIME_SCALE_S,
            [x[4 * n] - x[n - 1] - _AIM_AFTER_LAST_BURN_S / _TIME_SCALE_S],
            (e.floors_km - limits.radius_floor_km - self.floor_margin_km - _FLOOR_MARGIN_KM)
            / 1000.0,
        ]
        if self._total_binds():
            assert limits.total_cap_km_s is not None
            parts.append([(limits.total_cap_km_s - e.total_dv_km_s) / self.cap])
        if miss_bound_km is not None:
            parts.append([1.0 - (e.miss_km / miss_bound_km) ** 2])
        return np.concatenate([np.asarray(part, dtype=float) for part in parts])

    def constraint_jacobian(self, x: np.ndarray, miss_bound_km: float | None) -> np.ndarray:
        e = self.evaluate(x)
        n, size = self.n, self.size
        u = np.asarray(x[n : 4 * n]).reshape(n, 3)
        rows: list[np.ndarray] = []
        for k in range(n):
            row = np.zeros(size)
            row[n + 3 * k : n + 3 * k + 3] = -2.0 * u[k]
            rows.append(row)
        for k in range(n - 1):
            row = np.zeros(size)
            row[k + 1], row[k] = 1.0, -1.0
            rows.append(row)
        row = np.zeros(size)
        row[4 * n], row[n - 1] = 1.0, -1.0
        rows.append(row)
        rows.extend(e.floor_gradients / 1000.0)
        if self._total_binds():
            rows.append(-e.total_dv_gradient / self.cap)
        if miss_bound_km is not None:
            rows.append(-2.0 * e.miss_km * e.miss_gradient / miss_bound_km**2)
        return np.vstack(rows)

    def _total_binds(self) -> bool:
        total = self.setup.limits.total_cap_km_s
        return total is not None and self.n * self.cap > total

    def solve(
        self, x0: np.ndarray, phase: str, iterations: int, miss_bound_km: float | None = None
    ) -> np.ndarray:
        # ftol 1e-8 is 5e-7 points on the scaled score; 1e-10 cost 30 % more
        # time for the same scores to 1e-3 on DifferentPlane.
        result = minimize(
            self.objective,
            x0,
            args=(phase,),
            jac=True,
            method="SLSQP",
            bounds=self.bounds(),
            constraints=[
                {
                    "type": "ineq",
                    "fun": self.constraints,
                    "jac": self.constraint_jacobian,
                    "args": (miss_bound_km,),
                }
            ],
            options={"maxiter": iterations, "ftol": 1.0e-8},
        )
        x = np.asarray(result.x, dtype=float)
        if not np.all(np.isfinite(x)):
            return np.asarray(x0, dtype=float)
        return x


# --------------------------------------------------------------------------
# Starts


@dataclass(frozen=True, slots=True)
class RankingStart:
    label: str
    times_s: tuple[float, ...]
    impulses_km_s: tuple[tuple[float, float, float], ...]
    aim_s: float
    #: "search" runs every phase; "hit" and "miss" polish a solution in one.
    mode: str = "search"
    #: Added to the floor SLSQP holds; the wall is the floor either way.
    floor_margin_km: float = 0.0

    @property
    def burn_count(self) -> int:
        return len(self.times_s)


@dataclass(frozen=True, slots=True)
class Transfer:
    """The cheapest one-impulse Lambert transfer for a (departure, arrival) pair."""

    depart_s: float
    arrive_s: float
    impulse_km_s: tuple[float, float, float]

    @property
    def magnitude_km_s(self) -> float:
        return float(np.linalg.norm(self.impulse_km_s))


def lambert_transfers(
    setup: SearchSetup,
    config: OrbitWinConfig,
    *,
    departures: int = 16,
    arrivals: int = 32,
    samples: int = 96,
) -> list[Transfer]:
    """A grid of one-impulse transfers: where the target's geometry says to push.

    These impulses are mostly far over the cap. The starts split them into N
    capped impulses along the same direction, the shape the organizer's own
    preliminary answer had (1.5 + 0.741 km/s along one line).
    """
    limits = setup.limits
    depart = np.linspace(limits.first_burn_floor_s, limits.last_burn_ceiling_s, departures)
    arrive = np.linspace(
        max(limits.first_burn_floor_s + config.search.min_tof_s, limits.window_start_s),
        limits.horizon_s, arrivals,
    )
    count = len(depart)
    if setup.jump_s is None:
        pursuer_r, pursuer_v = coast_many(
            np.tile(setup.initial.r, (count, 1)), np.tile(setup.initial.v, (count, 1)), depart,
            setup.mu,
        )
    else:
        unburned = fly([], np.zeros((0, 3)), setup.initial, setup.mu, setup.jump_at())
        states = [unburned.state(float(t)) for t in depart]
        pursuer_r = np.array([r for r, _ in states])
        pursuer_v = np.array([v for _, v in states])
    target_r, _ = setup.target.states(arrive)
    found: list[Transfer] = []
    for i, depart_s in enumerate(depart):
        for j, arrive_s in enumerate(arrive):
            flight_s = float(arrive_s - depart_s)
            if flight_s < config.search.min_tof_s:
                continue
            try:
                solutions = solve_lambert_all(
                    pursuer_r[i],
                    target_r[j],
                    flight_s,
                    setup.mu,
                    max_revolutions=config.search.max_revolutions,
                    samples_per_interval=samples,
                    endpoint_tolerance_km=config.search.endpoint_tolerance_km,
                )
            except DynamicsError:
                continue
            impulses = [np.asarray(s.v1, dtype=float) - pursuer_v[i] for s in solutions]
            if not impulses:
                continue
            cheapest = min(impulses, key=lambda impulse: float(np.linalg.norm(impulse)))
            found.append(Transfer(float(depart_s), float(arrive_s), as_vector3(cheapest)))
    return found


def _transfer_value(transfer: Transfer, budget_km_s: float, setup: SearchSetup) -> float:
    """How promising a transfer is as a start: reachable first, then by score."""
    magnitude = transfer.magnitude_km_s
    if magnitude > budget_km_s:
        return -1000.0 - magnitude
    score = setup.score
    if score is None:
        return -magnitude
    time_term = _logistic_descending(
        25.0, score.kt * (transfer.arrive_s + setup.time_origin_offset_s - score.ct)
    )
    dv_term = _logistic_descending(
        25.0, score.kv * (_dv_for_score(magnitude, score.delta_v_unit) - score.cv)
    )
    return time_term + dv_term


def transfer_starts(
    transfers: Sequence[Transfer], burn_count: int, setup: SearchSetup, count: int
) -> list[RankingStart]:
    """Split the best transfers into `burn_count` capped impulses around their epoch."""
    limits = setup.limits
    cap = limits.burn_cap_km_s
    budget = burn_count * cap
    if limits.total_cap_km_s is not None:
        budget = min(budget, limits.total_cap_km_s)
    floor = limits.first_burn_floor_for(burn_count)
    span = (burn_count - 1) * limits.separation_s
    ranked = sorted(
        transfers,
        key=lambda t: (-_transfer_value(t, budget, setup), t.depart_s, t.arrive_s),
    )
    starts: list[RankingStart] = []
    taken: list[Transfer] = []
    for transfer in ranked:
        if len(starts) >= count:
            break
        first = max(floor, transfer.depart_s - span / 2.0)
        times = [first + k * limits.separation_s for k in range(burn_count)]
        if times[-1] > limits.last_burn_ceiling_s or transfer.arrive_s - times[-1] < 60.0:
            continue
        if any(
            abs(transfer.depart_s - other.depart_s) < 300.0
            and abs(transfer.arrive_s - other.arrive_s) < 300.0
            for other in taken
        ):
            continue
        impulse = np.asarray(transfer.impulse_km_s, dtype=float)
        magnitude = float(np.linalg.norm(impulse))
        if magnitude == 0.0:
            continue
        each = impulse / burn_count if magnitude / burn_count <= cap else impulse * cap / magnitude
        taken.append(transfer)
        starts.append(
            RankingStart(
                label=f"n{burn_count}_lam{len(starts)}_d{int(transfer.depart_s)}"
                f"_a{int(transfer.arrive_s)}",
                times_s=tuple(times),
                impulses_km_s=tuple(as_vector3(each) for _ in range(burn_count)),
                aim_s=transfer.arrive_s,
            )
        )
    return starts


def random_starts(
    burn_count: int, setup: SearchSetup, count: int, seed: int
) -> list[RankingStart]:
    """Scrambled Sobol points mapped onto schedules that already hold the walls."""
    if count <= 0:
        return []
    limits = setup.limits
    floor = limits.first_burn_floor_for(burn_count)
    slack = limits.last_burn_ceiling_s - floor - (burn_count - 1) * limits.separation_s
    if slack < 0.0:
        return []
    sampler = qmc.Sobol(4 * burn_count + 1, scramble=True, seed=seed)
    points = sampler.random_base2(max(0, math.ceil(math.log2(count))))[:count]
    starts: list[RankingStart] = []
    for index, point in enumerate(points):
        offsets = np.sort(point[:burn_count]) * slack
        times = [floor + k * limits.separation_s + float(offsets[k]) for k in range(burn_count)]
        impulses = []
        for k in range(burn_count):
            cos_polar = 2.0 * float(point[burn_count + 2 * k]) - 1.0
            azimuth = 2.0 * math.pi * float(point[burn_count + 2 * k + 1])
            sin_polar = math.sqrt(max(0.0, 1.0 - cos_polar * cos_polar))
            magnitude = limits.burn_cap_km_s * (0.25 + 0.75 * float(point[3 * burn_count + k]))
            impulses.append(
                as_vector3(
                    magnitude
                    * np.array(
                        [sin_polar * math.cos(azimuth), sin_polar * math.sin(azimuth), cos_polar]
                    )
                )
            )
        earliest = max(times[-1] + _AIM_AFTER_LAST_BURN_S, limits.window_start_s)
        aim = earliest + (limits.horizon_s - earliest) * float(point[4 * burn_count])
        starts.append(
            RankingStart(
                label=f"n{burn_count}_sob{index}",
                times_s=tuple(times),
                impulses_km_s=tuple(impulses),
                aim_s=aim,
            )
        )
    return starts


# --------------------------------------------------------------------------
# Refinement and results


@dataclass(frozen=True, slots=True)
class RankingSolution:
    """One schedule, flown, measured over [0, T_max] and inside every wall."""

    label: str
    phase: str
    times_s: tuple[float, ...]
    impulses_km_s: tuple[tuple[float, float, float], ...]
    measurement: Measurement
    total_dv_km_s: float
    #: The force-model margin the search held it to; GMAT decides between them.
    floor_margin_km: float = 0.0

    @property
    def burn_count(self) -> int:
        return len(self.times_s)

    @property
    def active_burn_count(self) -> int:
        """Burns that carry an impulse; a pinned slot may hold an exact zero."""
        return sum(1 for impulse in self.impulses_km_s if any(impulse))

    @property
    def successful(self) -> bool:
        return self.measurement.intercept_time_s is not None

    def to_dict(self) -> dict[str, Any]:
        m = self.measurement
        return {
            "label": self.label,
            "phase": self.phase,
            "floor_margin_km": self.floor_margin_km,
            "n_burns": self.burn_count,
            "total_dv_km_s": self.total_dv_km_s,
            "minimum_distance_km": m.minimum_distance_km,
            "minimum_distance_time_s": m.minimum_distance_time_s,
            "intercept_time_s": m.intercept_time_s,
            "distance_at_horizon_km": m.distance_at_horizon_km,
            "minimum_radius_km": m.minimum_radius_km,
            "burns": [
                {"time_s": t, "dv_inertial_km_s": list(dv)}
                for t, dv in zip(self.times_s, self.impulses_km_s, strict=True)
            ],
        }


def settle(
    label: str,
    phase: str,
    times_s: Sequence[float],
    impulses_km_s: np.ndarray,
    setup: SearchSetup,
    floor_margin_km: float = 0.0,
) -> RankingSolution | None:
    """Repair, fly and measure a schedule; None unless every wall holds.

    This is the only way a RankingSolution is made, so nothing reaches the
    caller without passing the walls on the schedule it will actually carry.
    """
    limits = setup.limits
    times, impulses = repair(times_s, impulses_km_s, limits)
    # A burn the optimizer drove to nothing is a Maneuver line for no reason.
    keep = [i for i, dv in enumerate(impulses) if np.linalg.norm(dv) > _NEGLIGIBLE_KM_S]
    if limits.pins_first_burn(len(times)) and 0 in keep and limits.zero_vector_slots:
        # Except where slot 0 fires at t = 0: dropping a later one moves that burn
        # to a later slot, and every slot's Maneuver is written anyway. An empty
        # one goes as early as the spacing allows, before any interception.
        impulses = impulses.copy()
        for index in range(1, len(times)):
            if index not in keep:
                impulses[index] = 0.0
                times[index] = times[index - 1] + limits.separation_s
        keep = list(range(len(times)))
    times, impulses = [times[i] for i in keep], impulses[keep]
    if wall_violations(times, impulses, limits):
        return None
    try:
        flight = _fly(setup, times, impulses)
        measured = measure(
            flight, setup.target, setup.grid, limits.horizon_s, limits.intercept_radius_km
        )
    except DynamicsError:
        return None
    if measured.intercept_time_s is not None and times and measured.intercept_time_s < times[-1]:
        # The mission ends at the first entry, so later burns cannot be flown.
        # Trimmed before the floor check, which then covers the arc that is.
        kept = [index for index, t in enumerate(times) if t <= measured.intercept_time_s]
        if limits.pins_first_burn(len(times)) and limits.zero_vector_slots and 0 in kept:
            # Fewer burns would unpin slot 0 off t = 0, so the slots stay and no
            # kept burn moves: each empty one takes the earliest place a
            # separation clear of its neighbours, before the entry.
            empty = len(times) - len(kept)
            sep = limits.separation_s
            kept_times = [times[i] for i in kept]
            places: list[float] = []
            # No tolerance: a place a hair past the entry would be trimmed again for
            # ever, and one a hair short of the next kept burn would move it.
            ends = [*kept_times[1:], measured.intercept_time_s + sep]
            for start, stop in zip(kept_times, ends, strict=True):
                place = start + sep
                while place + sep <= stop and len(places) < empty:
                    places.append(place)
                    place += sep
            if len(places) < empty:
                return None
            order = sorted(
                [(t, impulses[i]) for t, i in zip(kept_times, kept, strict=True)]
                + [(t, np.zeros(3)) for t in places],
                key=lambda item: item[0],
            )
            return settle(
                label, phase + "+trimmed", [t for t, _ in order],
                np.array([impulse for _, impulse in order]), setup, floor_margin_km,
            )
        return settle(
            label, phase + "+trimmed", [times[i] for i in kept], impulses[kept], setup,
            floor_margin_km,
        )
    if not measured.minimum_radius_km >= limits.radius_floor_km:
        return None
    return RankingSolution(
        label=label,
        phase=phase,
        times_s=tuple(times),
        impulses_km_s=tuple(as_vector3(impulse) for impulse in impulses),
        measurement=measured,
        total_dv_km_s=float(sum(np.linalg.norm(impulse) for impulse in impulses)),
        floor_margin_km=floor_margin_km,
    )


def _phases(
    problem: _Problem, x0: np.ndarray, mode: str, iterations: int
) -> list[tuple[str, np.ndarray]]:
    """Run the phases a start asks for and return each phase's end point.

    The hit phase runs whatever the approach reached: measured on the
    DifferentPlane geometry, it got from 2560 km and 1833 km approaches to
    1.05 km and 8.0 km, where the approach phase had stalled.
    """
    setup = problem.setup
    aim = setup.aim_radius_km
    scored = setup.score is not None
    hit_phase = "hit" if scored else "hit-dv"
    if mode == "hit":
        return [(hit_phase, problem.solve(x0, hit_phase, iterations, aim))]
    if mode == "miss":
        if scored:
            return [("miss", problem.solve(x0, "miss", iterations))]
        bound = problem.evaluate(x0).miss_km * (1.0 + 1.0e-6) + 1.0e-6
        return [("polish", problem.solve(x0, "polish", iterations, bound))]
    results = [("approach", problem.solve(x0, "approach", iterations))]
    if scored:
        results.append(("miss", problem.solve(results[0][1], "miss", iterations)))
    closest = min(results, key=lambda item: problem.evaluate(item[1]).miss_km)[1]
    results.append((hit_phase, problem.solve(closest, hit_phase, iterations, aim)))
    approach_miss = problem.evaluate(results[0][1]).miss_km
    if not scored and approach_miss > aim:
        # Distance first, then delta-v: keep what the approach reached, cheaper.
        bound = approach_miss * (1.0 + 1.0e-6) + 1.0e-6
        results.append(("polish", problem.solve(results[0][1], "polish", iterations, bound)))
    return results


def refine_start(
    start: RankingStart, setup: SearchSetup, iterations: int | None = None
) -> list[RankingSolution]:
    """Every phase the start asks for, each end point settled. Deterministic."""
    problem = _Problem(setup, start.burn_count, start.floor_margin_km)
    x0 = problem.pack(start.times_s, np.asarray(start.impulses_km_s), start.aim_s)
    budget = setup.screen_iterations if iterations is None else iterations
    try:
        results = _phases(problem, x0, start.mode, budget)
    except DynamicsError:
        return []
    solutions: list[RankingSolution] = []
    for phase, x in results:
        times, impulses, _ = problem.unpack(x)
        solution = settle(
            start.label, phase, list(times), impulses, setup, start.floor_margin_km
        )
        if solution is not None:
            solutions.append(solution)
    return solutions


def polish_starts(solution: RankingSolution) -> list[RankingStart]:
    """A solution as starts for one longer phase, aimed at its closest approach.

    One keeps every burn; one drops those under _DUST_KM_S. SLSQP leaves such
    dust where the delta-v norm has its kink: measured on the flyby rehearsal
    without a score block, eleven burns of 0.005 to 0.18 m/s beside one of
    125 m/s. Both are polished and ranked, so a burn that mattered survives.
    """
    if not solution.burn_count:
        # The miss phase can spend nothing; that is the zero-burn baseline.
        return []
    m = solution.measurement
    mode = "hit" if solution.successful else "miss"
    starts = [
        RankingStart(
            label=solution.label + "/polished",
            times_s=solution.times_s,
            impulses_km_s=solution.impulses_km_s,
            aim_s=m.minimum_distance_time_s,
            mode=mode,
            floor_margin_km=solution.floor_margin_km,
        )
    ]
    kept = [
        (time_s, impulse)
        for time_s, impulse in zip(solution.times_s, solution.impulses_km_s, strict=True)
        if math.sqrt(sum(c * c for c in impulse)) >= _DUST_KM_S
    ]
    # All dust prunes to nothing, the baseline search_ranking already ranks.
    if kept and len(kept) < solution.burn_count:
        starts.append(
            RankingStart(
                label=solution.label + "/pruned",
                times_s=tuple(time_s for time_s, _ in kept),
                impulses_km_s=tuple(impulse for _, impulse in kept),
                aim_s=m.minimum_distance_time_s,
                mode=mode,
                floor_margin_km=solution.floor_margin_km,
            )
        )
    return starts


def solution_key(
    solution: RankingSolution, setup: SearchSetup, scenario: Scenario, config: OrbitWinConfig
) -> tuple[Any, ...]:
    """Best first. The ranking score and its tie-break, or without coefficients
    the distance term first and then delta-v.

    Each maneuver also costs _POINTS_PER_BURN (without a score block,
    _KM_PER_BURN on a miss and _KM_S_PER_BURN) in the ordering only.
    Measured on the flyby rehearsal: eight burns, seven of them 1-4 mm/s,
    outscored the same transfer in one burn by 4e-5 points. The reported
    score is the official one.
    """
    m = solution.measurement
    if setup.score is not None:
        candidate = as_candidate(solution, scenario, config, setup.limits)
        breakdown = score_candidate(candidate, scenario, setup.score)
        candidate.score = breakdown.raw_score - _POINTS_PER_BURN * solution.active_burn_count
        candidate.metadata["event_refined"] = True
        return (*ranking_key(candidate, scenario=scenario)[:4], solution.label, solution.phase)
    intercept = m.intercept_time_s if m.intercept_time_s is not None else math.inf
    radius = setup.limits.intercept_radius_km
    return (
        max(m.minimum_distance_km, radius)
        + (_KM_PER_BURN * solution.active_burn_count if m.minimum_distance_km > radius else 0.0),
        solution.total_dv_km_s + _KM_S_PER_BURN * solution.active_burn_count,
        intercept,
        m.minimum_distance_km,
        solution.label,
        solution.phase,
    )


def as_candidate(
    solution: RankingSolution,
    scenario: Scenario,
    config: OrbitWinConfig,
    limits: RankingLimits,
    model: PursuerModel | None = None,
) -> Candidate:
    """The Candidate the pipeline understands, with script-frame components
    converted on the flown state, as multi_burn.as_candidate does: two-body,
    or `model`'s when the solution was flown in one."""
    mu = scenario.mu_km3_s2
    slots = limits.slot_indices_for(solution.burn_count)
    impulses = np.asarray(solution.impulses_km_s).reshape(-1, 3)
    # A head-start solution's clock starts lead_s before the solver's.
    offset_s = limits.clock_offset_s
    if offset_s and (model is not None or scenario.head_start_state is None):
        raise DynamicsError("a head-start solution is flown two-body from the probed state")
    initial = scenario.head_start_state if offset_s else scenario.pursuer.initial_state
    assert initial is not None
    flight: Flight | ModelFlight = (
        fly(solution.times_s, impulses, initial, mu, _jump(scenario, offset_s))
        if model is None
        else fly_model(solution.times_s, impulses, initial, model, limits.horizon_s)
    )
    burns: list[Burn] = []
    for index, (time_s, impulse) in enumerate(
        zip(np.asarray(solution.times_s) - offset_s, solution.impulses_km_s, strict=True)
    ):
        state = StateVector(as_vector3(flight.r_at_burns[index]), as_vector3(flight.v_before[index]))
        slot = slots[index]
        burns.append(
            Burn(
                time_s=float(time_s),
                dv_inertial_km_s=as_vector3(impulse),
                dv_script_km_s=as_vector3(
                    inertial_to_script_components(
                        np.asarray(impulse, dtype=float), state, *burn_frame(scenario, slot)
                    )
                ),
                slot_index=slot,
                frame=burn_frame(scenario, slot),
            )
        )
    m = solution.measurement
    hit = m.intercept_time_s is not None
    return Candidate(
        burns=tuple(burns),
        intercept_time_s=None if m.intercept_time_s is None else m.intercept_time_s - offset_s,
        minimum_distance_km=m.minimum_distance_km,
        distance_at_completion_km=(
            config.policy.intercept_radius_km if hit else m.distance_at_horizon_km
        ),
        successful=hit,
        analytical_endpoint_error_km=None,
        branch=f"ranking-search/{solution.burn_count}" + ("/head-start" if offset_s else ""),
        metadata={
            # A hit's is where the search aimed, for the GMAT correction; a
            # miss's is where its script ends, at T_max.
            "arrival_time_s": (m.minimum_distance_time_s if hit else limits.horizon_s) - offset_s,
            "event_refined": True,
            # Measured to T_max, not only to completion.
            "minimum_pursuer_radius_km": m.minimum_radius_km,
            "radius_measured_through_horizon": True,
            "search_label": solution.label,
            "search_phase": solution.phase,
            **({"pursuer_model": model.name} if model is not None else {}),
            **_head_start_window(scenario, burns),
            # What rank and certify check an archive against, as solve stamps it.
            "scenario_sha256": scenario.source_sha256,
            "horizon_s": scenario.max_time_s,
            **clock_stamps(scenario),
            "searched_score": score_stamp(config.score),
        },
    )


def _jump(scenario: Scenario, offset_s: float) -> tuple[float, np.ndarray, np.ndarray] | None:
    """The head start's defect at its clock's alignment end; None on the solver's."""
    if not offset_s:
        return None
    defect = scenario.head_start_defect
    if defect is None:
        raise DynamicsError("a head-start plan needs the alignment's two-body defect")
    return offset_s, defect.r, defect.v


def _head_start_window(scenario: Scenario, burns: list[Burn]) -> dict[str, float]:
    """Under from_origin, the alignment's closest approach with these head-start burns."""
    if scenario.alignment_minimum_distance_km is None:
        return {}
    window = head_start_alignment_distance_km(scenario, tuple(burns))
    return {} if window is None else {HEAD_START_ALIGNMENT_KEY: window}


def scored_candidate(
    solution: RankingSolution,
    scenario: Scenario,
    config: OrbitWinConfig,
    limits: RankingLimits,
    model: PursuerModel | None = None,
) -> Candidate:
    candidate = as_candidate(solution, scenario, config, limits, model)
    apply_constraints(candidate, scenario, config)
    apply_score(candidate, scenario, config.score)
    return candidate


_WORKER_SETUP: SearchSetup | None = None


def _init_worker(setup: SearchSetup) -> None:
    global _WORKER_SETUP
    _WORKER_SETUP = setup
    # This process only: workers are single-threaded and never return control.
    for setter, _ in _openblas_thread_controls():
        setter(1)


def _refine_in_worker(job: tuple[RankingStart, int]) -> list[RankingSolution]:
    if _WORKER_SETUP is None:
        raise DynamicsError("a ranking-search worker started without its initializer")
    start, iterations = job
    return refine_start(start, _WORKER_SETUP, iterations)


@dataclass(slots=True)
class RankingSearchResult:
    setup: SearchSetup
    starts: int
    #: Every surviving solution, best first, one per distinct schedule.
    solutions: list[RankingSolution] = field(default_factory=list)
    #: The zero-burn flight measured the same way, for comparison.
    baseline: RankingSolution | None = None


def build_setup(
    scenario: Scenario,
    config: OrbitWinConfig,
    *,
    target: TargetTrack | None = None,
    altitude_floor_km: float = 100.0,
    correction_margin_km_s: float = 0.0,
    force_model_margins_km: Sequence[float] = FORCE_MODEL_MARGINS_KM,
    aim_radius_km: float = 1.0,
    grid_step_s: float = 30.0,
    screen_iterations: int = 80,
    polish_iterations: int = 400,
    head_start: bool = False,
) -> SearchSetup:
    """With `head_start`, on the head start's clock (RankingLimits.head_start):
    the pursuer from its probed epoch state, the target from lead_s on."""
    make = RankingLimits.head_start if head_start else RankingLimits.from_config
    limits = make(
        scenario, config,
        altitude_floor_km=altitude_floor_km,
        correction_margin_km_s=correction_margin_km_s,
    )
    margins = tuple(sorted({float(margin) for margin in force_model_margins_km}))
    if not margins or not all(math.isfinite(m) and m >= 0.0 for m in margins):
        raise DynamicsError(
            f"force-model margins must be finite and >= 0, got {force_model_margins_km!r}"
        )
    if not 0.0 < aim_radius_km < limits.intercept_radius_km:
        raise DynamicsError(
            f"aim radius {aim_radius_km!r} km has to be inside the "
            f"{limits.intercept_radius_km} km sphere"
        )
    if config.score is not None:
        # Here, not first in a worker.
        validate_score_for_rules(config.score, interception_rules(scenario.rules))
    track: TargetTrack = target if target is not None else conic_target(scenario)
    initial = scenario.pursuer.initial_state
    jump: dict[str, Any] = {}
    if head_start:
        assert scenario.head_start_state is not None  # RankingLimits.head_start checked
        track = DelayedTarget(track, limits.clock_offset_s)
        initial = scenario.head_start_state
        at = _jump(scenario, limits.clock_offset_s)
        assert at is not None
        jump = {"jump_s": at[0], "jump_r_km": as_vector3(at[1]), "jump_v_km_s": as_vector3(at[2])}
    return SearchSetup(
        initial=initial,
        mu=scenario.mu_km3_s2,
        target=track,
        limits=limits,
        score=config.score,
        aim_radius_km=aim_radius_km,
        grid=MeasurementGrid.build(track, limits.horizon_s, grid_step_s, limits.window_start_s),
        force_model_margins_km=margins,
        screen_iterations=screen_iterations,
        polish_iterations=polish_iterations,
        rules=scenario.rules,
        time_origin_offset_s=scenario.time_origin_offset_s - limits.clock_offset_s,
        **jump,
    )


def build_starts(
    setup: SearchSetup,
    config: OrbitWinConfig,
    burn_counts: Sequence[int],
    *,
    transfer_count: int = 8,
    random_count: int = 4,
    seed: int = 20260818,
    transfer_grid: tuple[int, int] = (16, 32),
) -> list[RankingStart]:
    """Per margin and burn count: split Lambert transfers, then Sobol points. Seeded."""
    departures, arrivals = transfer_grid
    transfers = (
        lambert_transfers(setup, config, departures=departures, arrivals=arrivals)
        if transfer_count > 0
        else []
    )
    base: list[RankingStart] = []
    for burn_count in sorted(set(burn_counts)):
        base.extend(transfer_starts(transfers, burn_count, setup, transfer_count))
        base.extend(
            random_starts(
                burn_count, setup, random_count, zlib.crc32(f"{seed}:{burn_count}".encode())
            )
        )
    return [
        dataclasses.replace(start, label=f"{start.label}@m{margin:g}", floor_margin_km=margin)
        for margin in setup.force_model_margins_km
        for start in base
    ]


_OPENBLAS_SETTERS = (
    "scipy_openblas_set_num_threads64_",
    "scipy_openblas_set_num_threads",
    "openblas_set_num_threads64_",
    "openblas_set_num_threads",
)


def _openblas_thread_controls() -> list[tuple[Any, Any]]:
    """(set, get) thread-count functions of every OpenBLAS this process loaded."""
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    paths = {
        line.split()[-1]
        for line in maps.read_text(encoding="utf-8", errors="replace").splitlines()
        if "openblas" in line.lower() and line.split()[-1].startswith("/")
    }
    controls: list[tuple[Any, Any]] = []
    for path in sorted(paths):
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        for name in _OPENBLAS_SETTERS:
            setter = getattr(library, name, None)
            getter = getattr(library, name.replace("_set_", "_get_"), None)
            if setter is not None and getter is not None:
                controls.append((setter, getter))
                break
    return controls


#: The pool's start method; None is the platform default, on Linux fork
#: before Python 3.14 and forkserver from it.
_POOL_CONTEXT: Any = None


def _refine_all(
    jobs: Sequence[tuple[RankingStart, int]], setup: SearchSetup, workers: int
) -> list[list[RankingSolution]]:
    """Refine in worker processes, one even for workers=1, each on one BLAS thread.

    SLSQP's subproblem runs on OpenBLAS, which starts a thread per core. On
    R4 with 36 starts: 87.2 s serial and 103.8 s on six workers left alone,
    20.3 s and 4.2 s held to one thread. The thread count also changes the
    numerics (best 1.804975 against 1.804990 km/s), so serial runs go through
    a worker too and every worker count computes alike. The caller's process
    is never touched. A forked worker keeps the count its parent read at
    import, so the environment variable cannot do this; OpenBLAS's own setter
    can, found through /proc/self/maps.
    """
    if not jobs:
        return []
    if not _openblas_thread_controls():
        warnings.warn(
            "no OpenBLAS thread control found; ranking-search workers run with "
            "the default thread count, slower by about 4x when measured",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        with ProcessPoolExecutor(
            max_workers=max(1, min(workers, len(jobs))),
            mp_context=_POOL_CONTEXT,
            initializer=_init_worker,
            initargs=(setup,),
        ) as pool:
            return list(pool.map(_refine_in_worker, jobs))
    except BrokenProcessPool as exc:
        # As search.py does: say so rather than fall back in silence.
        raise ParallelSearchError(
            f"A ranking-search worker process died ({exc}). The result does not depend "
            "on the worker count, so fewer workers is the fallback; workers = 1 still "
            "runs one worker process."
        ) from exc


def search_ranking(
    scenario: Scenario,
    config: OrbitWinConfig,
    setup: SearchSetup,
    starts: Sequence[RankingStart],
    *,
    workers: int = 1,
    polish_top: int = 12,
) -> RankingSearchResult:
    """Screen every start, polish the best few, keep what survives the walls.

    Polishing takes the best solution of each of the `polish_top` best starts
    of every force-model margin through one more phase with
    `setup.polish_iterations`. Results are gathered
    in job order and sorted with the label as the last key, so the answer does
    not depend on the worker count.
    """
    limits = setup.limits
    for start in starts:
        if not 1 <= start.burn_count <= limits.max_burns:
            raise DynamicsError(
                f"start {start.label} has {start.burn_count} burns; "
                f"this template allows 1 to {limits.max_burns}"
            )

    def key(solution: RankingSolution) -> tuple[Any, ...]:
        return solution_key(solution, setup, scenario, config)

    screened = _refine_all([(start, setup.screen_iterations) for start in starts], setup, workers)
    best_per_start = polish_selection(screened, setup.force_model_margins_km, key, polish_top)
    polished = _refine_all(
        [
            (start, setup.polish_iterations)
            for solution in best_per_start
            for start in polish_starts(solution)
        ],
        setup,
        workers,
    )
    baseline = settle("zero-burn", "baseline", [], np.zeros((0, 3)), setup)
    pool = [s for batch in [*screened, *polished] for s in batch]
    solutions = rank_solutions(pool, key, baseline)
    return RankingSearchResult(setup, len(starts), solutions, baseline)


def polish_in_model(
    scenario: Scenario,
    config: OrbitWinConfig,
    model_setup: SearchSetup,
    searched: RankingSearchResult,
    *,
    top: int,
    workers: int = 1,
) -> RankingSearchResult:
    """Re-polish the best `top` two-body solutions of every margin in the pursuer model.

    Each keeps its burn count and starts from its own schedule, in its own
    phase: a hit stays a hit, aimed where it was. What comes back is settled
    and ranked in the model alone; the two-body ones are not kept.
    """
    if model_setup.pursuer_model is None:
        raise DynamicsError("polish_in_model needs a setup with a pursuer model")

    def key(solution: RankingSolution) -> tuple[Any, ...]:
        return solution_key(solution, model_setup, scenario, config)

    chosen = [
        solution
        for margin in model_setup.force_model_margins_km
        for solution in [
            s for s in searched.solutions if s.burn_count and s.floor_margin_km == margin
        ][:top]
    ]
    jobs = [
        (
            RankingStart(
                label=solution.label + "/model",
                times_s=solution.times_s,
                impulses_km_s=solution.impulses_km_s,
                # A miss's closest approach can precede its last burn, where the
                # model's last arc has no nodes to sample.
                aim_s=max(
                    solution.measurement.minimum_distance_time_s,
                    solution.times_s[-1] + _AIM_AFTER_LAST_BURN_S,
                ),
                mode="hit" if solution.successful else "miss",
                floor_margin_km=solution.floor_margin_km,
            ),
            model_setup.polish_iterations,
        )
        for solution in chosen
    ]
    polished = _refine_all(jobs, model_setup, workers)
    baseline = settle("zero-burn", "baseline", [], np.zeros((0, 3)), model_setup)
    pool = [s for batch in polished for s in batch]
    return RankingSearchResult(
        model_setup, len(jobs), rank_solutions(pool, key, baseline), baseline
    )


def polish_selection(
    screened: Sequence[Sequence[RankingSolution]],
    margins: Sequence[float],
    key: Callable[[RankingSolution], tuple[Any, ...]],
    polish_top: int,
) -> list[RankingSolution]:
    """Each start's best, the `polish_top` best of every margin in turn."""
    return [
        solution
        for margin in margins
        for solution in sorted(
            (
                min(batch, key=key)
                for batch in screened
                if batch and batch[0].floor_margin_km == margin
            ),
            key=key,
        )[:polish_top]
    ]


def _carries_a_burn(solution: RankingSolution) -> bool:
    """At least one impulse of _DUST_KM_S or more."""
    return any(math.sqrt(sum(c * c for c in dv)) >= _DUST_KM_S for dv in solution.impulses_km_s)


def best_per_margin(result: RankingSearchResult) -> dict[float, RankingSolution]:
    """The best solution searched under each force-model margin, for GMAT to judge.

    A schedule made only of dust prunes to the zero burn, which the baseline
    already is: on R5 the margin-10 "best" was a single 8.6 mm/s burn.
    """
    best: dict[float, RankingSolution] = {}
    for solution in result.solutions:
        if _carries_a_burn(solution):
            best.setdefault(solution.floor_margin_km, solution)
    return best


def schedule_signature(solution: RankingSolution) -> tuple[Any, ...]:
    """The schedule, rounded: starts that converge to one differ by about 1e-9 km/s."""
    return (
        tuple(round(t, 2) for t in solution.times_s),
        tuple(tuple(round(c, 6) for c in dv) for dv in solution.impulses_km_s),
    )


def rank_solutions(
    pool: Sequence[RankingSolution],
    key: Callable[[RankingSolution], tuple[Any, ...]],
    baseline: RankingSolution | None = None,
) -> list[RankingSolution]:
    """Best first, one per schedule.

    The zero-burn baseline belongs in the pool: measured on R3 with a 2500 s
    final padding, the best search result was one 1.4e-7 km/s burn scoring
    3e-7 below spending nothing. An all-dust schedule ranked behind it is the
    zero burn again and is dropped; on R5 such rows tied it at 0.0000 km/s.
    """
    seen: set[tuple[Any, ...]] = set()
    solutions: list[RankingSolution] = []
    zero_seen = False
    if baseline is not None:
        # Any zero-burn schedule is the baseline's; the baseline stands for it,
        # so one schedule never gets two candidate ids.
        pool = [*(s for s in pool if s.burn_count), baseline]
    # Best first, so a near-duplicate keeps its better copy.
    for solution in sorted(pool, key=key):
        if solution.burn_count == 0:
            zero_seen = True
        elif zero_seen and not _carries_a_burn(solution):
            continue
        signature = schedule_signature(solution)
        if signature not in seen:
            seen.add(signature)
            solutions.append(solution)
    return solutions
