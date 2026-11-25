"""N-impulse search where the last burn is a Lambert completion.

The one- and two-burn searches enumerate transfers. This one does not: it
schedules N-1 impulses freely and then *solves* for the last one, so the target
is met exactly by construction rather than by landing on a grid point. That
turned out to be what the 2026-09-05 scenario needed, where the announced
coefficients made impulses nearly free and time everything -- the enumerating
searches returned 74.997 and this returned 98.319 on the same scenario.

The free variables are the first burn epoch, the arrival epoch, the N-1 coast
gaps, and a direction and magnitude for each of the first N-1 impulses. Burn N
is whatever Lambert says closes the remaining leg.

Every constraint is a hard wall rather than a penalty term. That is not style:
a refinement carrying the burn cap as a soft penalty produced |dv| = 1495.098
m/s on 2026-09-05, over a limit that costs ten points per violating maneuver.
An optimizer rides a soft boundary; it cannot cross a hard one.

The objective is `score.score_candidate`, not a local copy of the formula. A
search optimising a second transcription of the scoring rules is a search that
will one day optimise the wrong thing.

See `docs/LESSONS_2026-09-05.md` for what this produced and what it cost.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .config import OrbitWinConfig
from .errors import DynamicsError
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
)
from .orbit import cartesian_to_keplerian, propagate_universal
from .score import score_candidate

#: Refinement restarts per job. The first is the start itself; the rest are
#: jittered. Five was enough for every start to reach the same family on
#: 2026-09-05, which is the evidence that the answer is geometric.
RESTARTS = 5

#: Simplex budget for one refinement. Nelder-Mead is used rather than a
#: gradient method because `fly` is discontinuous: a schedule that violates a
#: wall returns the barrier value, and Lambert can drop a branch between one
#: evaluation and the next.
#:
#: 12000 is a competition-day number -- generous, because the whole search
#: takes minutes and the answer is worth points. Callers that want a scan
#: rather than a refinement pass a smaller one.
MAX_ITERATIONS = 12000


def _simplex_options(max_iterations: int) -> dict[str, Any]:
    return {
        "xatol": 1.0e-9,
        "fatol": 1.0e-11,
        "maxiter": max_iterations,
        "maxfev": max_iterations,
        "adaptive": True,
    }


#: What `objective` returns for a schedule that breaks a wall. Far above any
#: reachable negative score, so the simplex walks away from it.
_BARRIER = 1.0e6

#: Lowest impulse of a random start, as a fraction of the cap. It was 0.6 km/s,
#: 40 % of the preliminary 1.495, and above the ranking round's 0.25 cap.
_RANDOM_START_MAGNITUDE_FRACTION = 0.4

#: Slack on the burn-cap wall, matching `search._candidates_for_pair`.
#:
#: `_unpack` clamps a magnitude to the cap and then multiplies it by a unit
#: vector built from two angles, whose norm is 1 only to rounding. An impulse
#: asked to sit exactly on the cap therefore measures a few times 1e-16 above
#: it, and a strict comparison rejects the best schedule found on 2026-09-05 --
#: whose first, third, fourth and fifth impulses are all exactly at the cap.
#: This is not margin being given away: the cap is the configured safe limit,
#: 5 m/s below the rule's 1500 m/s, and 1e-12 km/s is one nanometre per second.
_CAP_TOLERANCE_KM_S = 1.0e-12


@dataclass(frozen=True, slots=True)
class MultiBurnSolution:
    """One schedule, flown and scored.

    `arrival_s` is when the pursuer reaches the target's position, which is not
    the interception time: the 5 km sphere is entered slightly earlier, so the
    scored time is a little better than this one. Event detection owns that
    distinction; this search does not guess at it.
    """

    label: str
    burns: tuple[Burn, ...]
    arrival_s: float
    total_dv_km_s: float
    minimum_radius_km: float
    score: float

    @property
    def burn_count(self) -> int:
        return len(self.burns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_burns": self.burn_count,
            "score": self.score,
            "arrival_s": self.arrival_s,
            "total_dv_km_s": self.total_dv_km_s,
            "minimum_radius_km": self.minimum_radius_km,
            "burns": [
                {
                    "time_s": burn.time_s,
                    "dv_inertial_km_s": [float(x) for x in np.asarray(burn.dv_inertial_km_s)],
                }
                for burn in self.burns
            ],
        }


@dataclass(frozen=True, slots=True)
class MultiBurnLimits:
    """The walls, read off the policy once instead of at every evaluation."""

    burn_cap_km_s: float
    separation_s: float
    first_burn_floor_s: float
    radius_floor_km: float
    horizon_s: float
    #: The policy's total_dv_safe_limit_km_s; None when the rules set no total.
    total_cap_km_s: float | None = None
    #: The shortest first segment the renderer writes, plus 1 µs; a first burn
    #: allowed at 0 goes there or to 0, never between.
    render_floor_s: float = 1.0e-3 + 1.0e-6

    @classmethod
    def from_config(
        cls, scenario: Scenario, config: OrbitWinConfig, *, altitude_floor_km: float = 100.0
    ) -> MultiBurnLimits:
        policy = config.policy
        return cls(
            burn_cap_km_s=policy.dv_safe_limit_km_s,
            separation_s=policy.safe_burn_separation_s,
            first_burn_floor_s=(
                0.0 if burn_may_follow_alignment(scenario) else policy.min_propagate_duration_s
            ),
            radius_floor_km=policy.central_body_radius_km + altitude_floor_km,
            horizon_s=scenario.max_time_s,
            total_cap_km_s=policy.total_dv_safe_limit_km_s,
            render_floor_s=policy.min_propagate_duration_s + 1.0e-6,
        )


#: Impulse counts tried when the rules set no total: what won on 2026-09-05.
PRELIMINARY_BURN_COUNTS = (5, 6, 7)


def default_burn_counts(limits: MultiBurnLimits) -> tuple[tuple[int, ...], str]:
    """The impulse counts to try when none are given, and why those.

    Under a total cap the preliminary counts are kept, since they win when
    delta-v is dear, and the count that can spend the whole total is added:
    7 impulses at 0.249 km/s reach only 1.743 of 2.99 km/s.
    """
    if limits.total_cap_km_s is None:
        return PRELIMINARY_BURN_COUNTS, "no total delta-v cap, the preliminary default"
    # The epsilon keeps an exact multiple, like 2.988 / 0.249, from rounding up.
    most = max(1, math.ceil(limits.total_cap_km_s / limits.burn_cap_km_s - 1.0e-9))
    counts = tuple(sorted(set(PRELIMINARY_BURN_COUNTS) | {most}))
    return counts, (
        f"{most} impulses at {limits.burn_cap_km_s:g} km/s are needed to spend "
        f"the {limits.total_cap_km_s:g} km/s total"
    )


def minimum_radius_km(state: StateVector, duration_s: float, mu_km3_s2: float) -> float:
    """The smallest geocentric radius reached over `duration_s` of coasting.

    Closed form where it exists. On a closed orbit the radius has one minimum
    per revolution, at periapsis, so the segment's smallest radius is either
    periapsis -- when the segment contains a periapsis passage -- or one of the
    two endpoints. Sampling instead would step over a deep, brief periapsis,
    and 2026-09-05 measured every cheap Lambert arc on that scenario going
    underground, so a check that can miss one is not worth having.
    """
    radius_now = float(np.linalg.norm(np.asarray(state.r, dtype=float)))
    if duration_s <= 0.0:
        return radius_now

    elements = cartesian_to_keplerian(state, mu_km3_s2)
    semi_major_km, eccentricity = elements.sma_km, elements.ecc
    position = np.asarray(state.r, dtype=float)
    velocity = np.asarray(state.v, dtype=float)
    momentum = np.cross(position, velocity)
    periapsis_any_km = float(np.dot(momentum, momentum)) / mu_km3_s2 / (1.0 + eccentricity)
    energy = 0.5 * float(np.dot(velocity, velocity)) - mu_km3_s2 / radius_now
    if energy >= 0.0:
        # Unbound, d(r.v)/dt = v^2 - mu/r = 2E + mu/r > 0: r.v only rises, so the
        # radius has at most one minimum, periapsis. 40 samples missed it by km.
        if float(np.dot(position, velocity)) >= 0.0:
            return radius_now
        end = propagate_universal(state, duration_s, mu_km3_s2)
        if float(np.dot(np.asarray(end.r, dtype=float), np.asarray(end.v, dtype=float))) <= 0.0:
            return float(np.linalg.norm(np.asarray(end.r, dtype=float)))
        return periapsis_any_km
    if semi_major_km <= 0.0 or eccentricity >= 1.0:
        # Bound, but e rounded to 1: a near-radial ellipse, which the reasoning
        # above does not cover. No arc of it goes below its periapsis.
        return min(radius_now, periapsis_any_km)

    periapsis_km = semi_major_km * (1.0 - eccentricity)
    true_anomaly = math.radians(elements.ta_deg) % (2.0 * math.pi)
    eccentric_anomaly = (
        2.0
        * math.atan2(
            math.sqrt(1.0 - eccentricity) * math.sin(true_anomaly / 2.0),
            math.sqrt(1.0 + eccentricity) * math.cos(true_anomaly / 2.0),
        )
    ) % (2.0 * math.pi)
    mean_anomaly = eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly)
    mean_motion = math.sqrt(mu_km3_s2 / semi_major_km**3)
    seconds_to_periapsis = (2.0 * math.pi - mean_anomaly) / mean_motion
    if seconds_to_periapsis <= duration_s:
        return periapsis_km

    radius_end = float(
        np.linalg.norm(np.asarray(propagate_universal(state, duration_s, mu_km3_s2).r, dtype=float))
    )
    return min(radius_now, radius_end)


def _lambert_completion(
    state: StateVector,
    goal_km: np.ndarray,
    time_of_flight_s: float,
    config: OrbitWinConfig,
) -> np.ndarray | None:
    """The cheapest impulse that puts `state` on `goal_km` in the time given.

    None when no branch closes it. DynamicsError -- LambertError among them --
    is how a schedule the optimizer proposed turns out to be unflyable, and is
    the only thing caught. Anything else is a defect and is allowed to escape,
    never swallowed.
    """
    if time_of_flight_s <= config.search.min_tof_s:
        return None
    try:
        solutions = solve_lambert_all(
            np.asarray(state.r, dtype=float),
            goal_km,
            time_of_flight_s,
            config.policy.mu_km3_s2,
            max_revolutions=config.search.max_revolutions,
            samples_per_interval=config.search.one_burn_samples,
            endpoint_tolerance_km=config.search.endpoint_tolerance_km,
        )
    except DynamicsError:
        return None

    velocity = np.asarray(state.v, dtype=float)
    best: np.ndarray | None = None
    best_magnitude = math.inf
    for solution in solutions:
        impulse = np.asarray(solution.v1, dtype=float) - velocity
        magnitude = float(np.linalg.norm(impulse))
        if magnitude < best_magnitude:
            best, best_magnitude = impulse, magnitude
    return best


def _unpack(
    vector: np.ndarray, burn_count: int, limits: MultiBurnLimits
) -> tuple[float, float, list[float], list[np.ndarray]]:
    """Decision vector to a schedule, with the walls applied by clamping.

    Clamping rather than rejecting keeps the simplex's own moves legal, so it
    explores along a wall instead of falling off it. The rejections in
    `fly` cover what clamping cannot fix.
    """
    first_burn_s = max(limits.first_burn_floor_s, float(vector[0]))
    if limits.first_burn_floor_s == 0.0 and first_burn_s < limits.render_floor_s:
        # After an alignment: at t = 0 itself, or past the floor the renderer holds.
        floor = limits.render_floor_s
        first_burn_s = 0.0 if first_burn_s < 0.5 * floor else floor
    arrival_s = float(vector[1])
    gaps_s = [max(limits.separation_s, float(gap)) for gap in vector[2 : 2 + burn_count - 1]]

    base = 2 + (burn_count - 1)
    impulses: list[np.ndarray] = []
    for index in range(burn_count - 1):
        polar, azimuth, magnitude = vector[base + 3 * index : base + 3 * index + 3]
        clamped = float(np.clip(magnitude, 0.0, limits.burn_cap_km_s))
        impulses.append(
            clamped
            * np.array(
                [
                    math.sin(polar) * math.cos(azimuth),
                    math.sin(polar) * math.sin(azimuth),
                    math.cos(polar),
                ]
            )
        )
    return first_burn_s, arrival_s, gaps_s, impulses


def fly(
    vector: np.ndarray,
    burn_count: int,
    scenario: Scenario,
    config: OrbitWinConfig,
    limits: MultiBurnLimits,
) -> tuple[list[tuple[float, np.ndarray]], float, float, float] | None:
    """Fly one decision vector.

    Returns the burns, the total delta-v, the smallest radius reached, and the
    arrival epoch -- or None when the schedule cannot be flown at all.

    The radius is checked to arrival and no further. That is a deliberate and
    load-bearing limitation: `docs/LESSONS_2026-09-05.md` §4 records that
    requiring safety across the whole 4TA horizon costs 15 points on that
    scenario, and `scripts/audit_burn_table.py --require-safe-through-horizon`
    is the gate that says which of the two a candidate satisfies. Under a
    richer force model the audit holds the whole arc regardless.
    """
    mu = config.policy.mu_km3_s2
    first_burn_s, arrival_s, gaps_s, impulses = _unpack(vector, burn_count, limits)

    if arrival_s > limits.horizon_s:
        return None
    if first_burn_s + sum(gaps_s) + config.search.min_tof_s >= arrival_s:
        return None

    goal_km = np.asarray(propagate_universal(scenario.target.initial_state, arrival_s, mu).r,
                         dtype=float)
    state = propagate_universal(scenario.pursuer.initial_state, first_burn_s, mu)

    burns: list[tuple[float, np.ndarray]] = []
    # From t = 0: a start may be on an arc that dives before the first burn.
    smallest_radius_km = minimum_radius_km(scenario.pursuer.initial_state, first_burn_s, mu)
    now_s = first_burn_s
    for impulse, gap_s in zip(impulses, gaps_s, strict=True):
        state = StateVector(as_vector3(state.r), as_vector3(np.asarray(state.v) + impulse))
        burns.append((now_s, impulse))
        smallest_radius_km = min(smallest_radius_km, minimum_radius_km(state, gap_s, mu))
        state = propagate_universal(state, gap_s, mu)
        now_s += gap_s

    completion = _lambert_completion(state, goal_km, arrival_s - now_s, config)
    if completion is None:
        return None
    burns.append((now_s, completion))
    state = StateVector(as_vector3(state.r), as_vector3(np.asarray(state.v) + completion))
    smallest_radius_km = min(
        smallest_radius_km, minimum_radius_km(state, arrival_s - now_s, mu)
    )

    total_dv_km_s = sum(float(np.linalg.norm(impulse)) for _, impulse in burns)
    return burns, total_dv_km_s, smallest_radius_km, arrival_s


def as_candidate(
    burns: Sequence[tuple[float, np.ndarray]],
    arrival_s: float,
    scenario: Scenario,
    config: OrbitWinConfig,
) -> Candidate:
    """A schedule as the Candidate the rest of the project already understands.

    Distance is the sphere radius by construction: the last burn is the Lambert
    completion, so the miss at arrival is zero and the rules floor the scored
    distance at 5 km. Note that nothing here declares a total delta-v --
    `Candidate.total_dv_km_s` derives it from these burns, and its docstring
    records what happened the last time those two numbers were allowed to
    disagree.

    The script components are the inertial ones expressed in the frame the
    scenario's burn slots dictate, so a solution can be handed to the renderer
    without a second conversion step deciding it differently.

    That conversion has to be done on the *flown* state. The default frame is
    Local/Earth/VNB -- the organizer's own workshop convention -- whose basis is
    built from the spacecraft's velocity, and by the second impulse that
    velocity already carries the first. Converting against the unmaneuvered
    trajectory writes components that mean a different vector, and GMAT would
    fly the different one without complaint.
    """
    mu = config.policy.mu_km3_s2
    rendered: list[Burn] = []
    state = scenario.pursuer.initial_state
    now_s = 0.0
    for index, (time_s, impulse) in enumerate(burns):
        if time_s > now_s:
            state = propagate_universal(state, time_s - now_s, mu)
            now_s = time_s
        rendered.append(
            Burn(
                time_s=time_s,
                dv_inertial_km_s=as_vector3(impulse),
                dv_script_km_s=as_vector3(
                    inertial_to_script_components(
                        np.asarray(impulse, dtype=float), state, *burn_frame(scenario, index)
                    )
                ),
                slot_index=index,
                frame=burn_frame(scenario, index),
            )
        )
        state = StateVector(
            as_vector3(state.r), as_vector3(np.asarray(state.v, dtype=float) + impulse)
        )
    return Candidate(
        burns=tuple(rendered),
        intercept_time_s=arrival_s,
        minimum_distance_km=config.policy.intercept_radius_km,
        distance_at_completion_km=config.policy.intercept_radius_km,
        successful=True,
        analytical_endpoint_error_km=0.0,
        branch=f"multi-burn/{len(burns)}",
    )


def score_schedule(
    burns: Sequence[tuple[float, np.ndarray]],
    arrival_s: float,
    scenario: Scenario,
    config: OrbitWinConfig,
) -> float:
    """The official score for a schedule that reaches the target.

    From `score_candidate`, not from a second copy of the formula living here.
    """
    if config.score is None:
        raise DynamicsError(
            "multi-burn search needs the announced scoring coefficients; "
            "set `score` in the configuration"
        )
    candidate = as_candidate(burns, arrival_s, scenario, config)
    return float(score_candidate(candidate, scenario, config.score).raw_score)


def objective(
    vector: np.ndarray,
    burn_count: int,
    scenario: Scenario,
    config: OrbitWinConfig,
    limits: MultiBurnLimits,
) -> float:
    """Negative score for a legal schedule; above the barrier for anything else.

    The walls stay hard. Every infeasible schedule scores at least `_BARRIER`
    and every feasible one at most zero, so no amount of violation can ever be
    traded for score and the optimizer cannot ride a boundary -- which is the
    whole reason for walls instead of penalty terms, after a
    soft-penalty refinement handed back |dv| = 1495.098 m/s.

    What is not flat is the region *above* the barrier. A returned value of
    exactly `_BARRIER` for everything infeasible gives Nelder-Mead a plateau,
    and a simplex on a plateau cannot find its way off it: measured on the
    2026-09-05 scenario, every one of 21 structured starts needs a completion
    of 3 to 11 km/s against a 1.495 km/s cap, so all 21 begin infeasible and a
    flat barrier left them there. Adding the size of the violation makes the
    plateau a slope down to the feasible set, and the search can find an answer
    rather than only reproduce one handed to it.

    Feasible schedules never see this arithmetic, so it cannot bias them.
    """
    first_burn_s, arrival_s, gaps_s, _ = _unpack(vector, burn_count, limits)

    # Structural: the schedule does not describe a mission at all. Measured in
    # seconds, scaled so a violation of a whole horizon is worth less than the
    # step from "unflyable" to "flyable but illegal".
    overrun_s = max(0.0, arrival_s - limits.horizon_s)
    shortfall_s = max(
        0.0, first_burn_s + sum(gaps_s) + config.search.min_tof_s - arrival_s
    )
    if overrun_s > 0.0 or shortfall_s > 0.0:
        return 2.0 * _BARRIER + (overrun_s + shortfall_s) / limits.horizon_s

    flown = fly(vector, burn_count, scenario, config, limits)
    if flown is None:
        # Lambert closed no branch. There is no distance-to-feasible to report:
        # the leg either has a transfer or it does not.
        return 2.0 * _BARRIER

    burns, total_dv_km_s, smallest_radius_km, arrival_s = flown
    excess_km_s = sum(
        max(0.0, float(np.linalg.norm(impulse)) - limits.burn_cap_km_s - _CAP_TOLERANCE_KM_S)
        for _, impulse in burns
    )
    if limits.total_cap_km_s is not None:
        excess_km_s += max(0.0, total_dv_km_s - limits.total_cap_km_s - _CAP_TOLERANCE_KM_S)
    dive_km = max(0.0, limits.radius_floor_km - smallest_radius_km)
    if excess_km_s > 0.0 or dive_km > 0.0:
        return _BARRIER + excess_km_s + dive_km / limits.radius_floor_km

    return -score_schedule(burns, arrival_s, scenario, config)


def stable_seed(label: str) -> int:
    """A per-job RNG seed that does not move between processes.

    `hash()` on a str is salted per interpreter unless PYTHONHASHSEED is set,
    so seeding from it makes a worker's jitter depend on which process picked
    the job up. Seeds must be deterministic; CRC32 of the label is
    one.
    """
    return zlib.crc32(label.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class MultiBurnStart:
    """One labelled starting point for the refinement."""

    label: str
    burn_count: int
    vector: tuple[float, ...]


def start_from_solution(
    solution: MultiBurnSolution | dict[str, Any], limits: MultiBurnLimits, *, label: str
) -> MultiBurnStart:
    """Turn a known schedule back into a decision vector.

    Restarting from the best answer so far is how the search improves on it,
    and how a published solution can be checked rather than believed: on
    2026-09-05 an externally published six-burn table was reproduced to 1.5e-6
    points this way before anything else was concluded from it.
    """
    payload = solution.to_dict() if isinstance(solution, MultiBurnSolution) else solution
    burns = payload["burns"]
    burn_count = len(burns)
    vector: list[float] = [float(burns[0]["time_s"]), float(payload["arrival_s"])]
    vector.extend(
        float(burns[index + 1]["time_s"]) - float(burns[index]["time_s"])
        for index in range(burn_count - 1)
    )
    for burn in burns[: burn_count - 1]:
        impulse = np.asarray(burn["dv_inertial_km_s"], dtype=float)
        magnitude = float(np.linalg.norm(impulse))
        direction = impulse / magnitude if magnitude > 0.0 else np.array([0.0, 0.0, 1.0])
        vector.extend(
            (
                math.acos(float(np.clip(direction[2], -1.0, 1.0))),
                math.atan2(float(direction[1]), float(direction[0])),
                min(magnitude, limits.burn_cap_km_s),
            )
        )
    return MultiBurnStart(label=label, burn_count=burn_count, vector=tuple(vector))


def structured_starts(
    burn_counts: Iterable[int],
    limits: MultiBurnLimits,
    *,
    seed: int,
    first_burn_epochs: Sequence[float] = (1.0e-3, 200.0, 600.0),
    long_coast_s: float = 2200.0,
    random_starts: int = 6,
) -> list[MultiBurnStart]:
    """The start set, generated the same way every time.

    Two families. The structured ones put a single long coast in each slot in
    turn with every other gap at the minimum and every impulse at the cap --
    that shape is what won on 2026-09-05, where the best schedule was "burn,
    coast 2153 s, then fire three times back to back", not evenly spaced
    impulses. The random ones exist so the structured guess is not the only
    thing tried.

    Ordering is by (burn count, family, index) and the RNG is seeded, so the
    list is a function of its arguments alone.
    """
    generator = np.random.default_rng(seed)
    starts: list[MultiBurnStart] = []
    for burn_count in sorted(set(burn_counts)):
        for coast_slot in range(burn_count - 1):
            for first_burn_s in first_burn_epochs:
                gaps = [limits.separation_s] * (burn_count - 1)
                gaps[coast_slot] = long_coast_s
                vector = [first_burn_s, first_burn_s + sum(gaps) + 3000.0, *gaps]
                for _ in range(burn_count - 1):
                    vector.extend(
                        (
                            float(generator.uniform(0.0, math.pi)),
                            float(generator.uniform(-math.pi, math.pi)),
                            limits.burn_cap_km_s,
                        )
                    )
                starts.append(
                    MultiBurnStart(
                        label=f"n{burn_count}_coast{coast_slot}_t{int(first_burn_s)}",
                        burn_count=burn_count,
                        vector=tuple(vector),
                    )
                )
        for index in range(random_starts):
            gaps = [
                float(max(limits.separation_s, value))
                for value in generator.uniform(limits.separation_s, 1200.0, burn_count - 1)
            ]
            first_burn_s = float(generator.uniform(limits.first_burn_floor_s, 600.0))
            vector = [
                first_burn_s,
                first_burn_s + sum(gaps) + float(generator.uniform(1500.0, 4500.0)),
                *gaps,
            ]
            for _ in range(burn_count - 1):
                vector.extend(
                    (
                        float(generator.uniform(0.0, math.pi)),
                        float(generator.uniform(-math.pi, math.pi)),
                        float(
                            generator.uniform(
                                _RANDOM_START_MAGNITUDE_FRACTION * limits.burn_cap_km_s,
                                limits.burn_cap_km_s,
                            )
                        ),
                    )
                )
            starts.append(
                MultiBurnStart(
                    label=f"n{burn_count}_rand{index}",
                    burn_count=burn_count,
                    vector=tuple(vector),
                )
            )
    return starts


def _jitter_scale(burn_count: int) -> np.ndarray:
    """How far a restart is thrown from the best point found so far.

    Wide on the epochs and gaps, narrow on the directions and magnitudes: the
    schedule's shape is what has to be searched, and an impulse already sitting
    on the cap only needs nudging.
    """
    return np.array(
        [30.0, 30.0] + [25.0] * (burn_count - 1) + [0.04, 0.04, 0.02] * (burn_count - 1)
    )


def refine_start(
    start: MultiBurnStart,
    scenario: Scenario,
    config: OrbitWinConfig,
    limits: MultiBurnLimits,
    *,
    restarts: int = RESTARTS,
    max_iterations: int = MAX_ITERATIONS,
) -> MultiBurnSolution | None:
    """Nelder-Mead from one start, with jittered restarts from the best point.

    Deterministic: the jitter's generator is seeded from the label through
    `stable_seed`, so this returns the same answer whichever worker runs it and
    in whatever order.
    """
    best_vector = np.asarray(start.vector, dtype=float)
    best_value = objective(best_vector, start.burn_count, scenario, config, limits)
    generator = np.random.default_rng(stable_seed(start.label))
    scale = _jitter_scale(start.burn_count)

    for attempt in range(restarts):
        trial = (
            best_vector
            if attempt == 0
            else best_vector + generator.normal(0.0, 1.0, best_vector.size) * scale
        )
        result = minimize(
            objective,
            trial,
            args=(start.burn_count, scenario, config, limits),
            method="Nelder-Mead",
            options=_simplex_options(max_iterations),
        )
        if float(result.fun) < best_value:
            best_vector, best_value = np.asarray(result.x, dtype=float), float(result.fun)

    # A feasible schedule always scores at least 50 -- the distance term alone
    # is that, since the completion arrives exactly -- so its objective is at
    # most -50. Anything above zero never reached the feasible set.
    if best_value > 0.0:
        return None
    flown = fly(best_vector, start.burn_count, scenario, config, limits)
    if flown is None:
        return None
    burns, total_dv_km_s, smallest_radius_km, arrival_s = flown
    return MultiBurnSolution(
        label=start.label,
        burns=as_candidate(burns, arrival_s, scenario, config).burns,
        arrival_s=arrival_s,
        total_dv_km_s=total_dv_km_s,
        minimum_radius_km=smallest_radius_km,
        score=-best_value,
    )


_WORKER_STATE: tuple[Scenario, OrbitWinConfig, MultiBurnLimits, int, int] | None = None


def _init_worker(
    scenario: Scenario,
    config: OrbitWinConfig,
    limits: MultiBurnLimits,
    restarts: int,
    max_iterations: int,
) -> None:
    global _WORKER_STATE
    _WORKER_STATE = (scenario, config, limits, restarts, max_iterations)


def _refine_in_worker(start: MultiBurnStart) -> MultiBurnSolution | None:
    if _WORKER_STATE is None:
        raise DynamicsError(
            "a multi-burn worker started without its initializer, so it has no "
            "scenario to fly. The result is unchanged with workers = 1."
        )
    scenario, config, limits, restarts, max_iterations = _WORKER_STATE
    return refine_start(
        start, scenario, config, limits, restarts=restarts, max_iterations=max_iterations
    )


def search_multi_burn(
    scenario: Scenario,
    config: OrbitWinConfig,
    *,
    starts: Sequence[MultiBurnStart],
    workers: int = 1,
    altitude_floor_km: float = 100.0,
    restarts: int = RESTARTS,
    max_iterations: int = MAX_ITERATIONS,
) -> list[MultiBurnSolution]:
    """Refine every start and return the survivors, best first.

    Results are collected in the order the starts were given and only then
    sorted, so the returned list does not depend on which worker finished
    first. Ties break on the label for the same reason.
    """
    limits = MultiBurnLimits.from_config(scenario, config, altitude_floor_km=altitude_floor_km)
    solutions: list[MultiBurnSolution | None]
    if workers <= 1 or len(starts) < 2:
        solutions = [
            refine_start(
                start,
                scenario,
                config,
                limits,
                restarts=restarts,
                max_iterations=max_iterations,
            )
            for start in starts
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(starts)),
            initializer=_init_worker,
            initargs=(scenario, config, limits, restarts, max_iterations),
        ) as pool:
            solutions = list(pool.map(_refine_in_worker, starts))
    found = [solution for solution in solutions if solution is not None]
    found.sort(key=lambda solution: (-solution.score, solution.label))
    return found
