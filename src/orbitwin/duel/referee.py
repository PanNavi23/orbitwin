"""One round, refereed: both impulses at t = 0, two coasts, the events, a verdict.

`simulate_round` produces facts for a given capture radius. `judge` turns facts
into an outcome under one reading of the open rules (event order, row
sampling, Earth surface, early entry). `decide` is the decision convention of
DUEL_ENGINE.md: every open reading resolved against the side we play.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..errors import OrbitWinError
from ..rules import DUEL
from .events import Arc, CaptureFacts, Passage, Radius, below_radius, capture_facts
from .gmat_settings import PropagatorSetting, setting
from .params import R_EQ_KM, R_POLAR_KM, DuelParams
from .state import CraftState, MatchState, semi_major_axis, specific_energy

Role = Literal["pursuer", "evader"]
#: Whose capture radius a round was flown with: ours as one role, or the bare
#: R_cap for refereeing a round that has happened.
View = Literal["pursuer", "evader", "referee"]
Winner = Literal["pursuer", "evader", "unresolved"]


class RefereeError(OrbitWinError):
    """Facts that do not hold together; never guessed past."""

    code = "duel"


@dataclass(frozen=True, slots=True)
class RoundFacts:
    round_number: int
    duration_s: float
    view: View
    #: The capture radius used, at the round start and end (section 2.5).
    capture_radius_start_km: float
    capture_radius_end_km: float
    pursuer: Arc
    evader: Arc
    capture: CaptureFacts
    #: Stretches below each named radius, per craft.
    pursuer_below: dict[str, list[Passage]]
    evader_below: dict[str, list[Passage]]
    end_pursuer: CraftState
    end_evader: CraftState
    #: The evader before its impulse: the state early entry judges in round 1.
    start_evader: CraftState
    #: Eq. (2) on the evader's state right after its impulse; constant in two-body.
    evader_a_km: float
    #: Eq. (2) on the evader's state before its impulse.
    evader_a_initial_km: float
    #: Largest row gap the pair can have, by the bound of section 2.6.
    row_gap_bound_s: float


#: Section 2.4: the band g around our floor, the radius tolerance of the
#: parent's final check. A radius within g of the floor is a crossing.
OWN_FLOOR_BAND_KM = 1.0e-6


def own_floor_km(params: DuelParams) -> float:
    """The radius our own craft must stay above, g included.

    Every place that judges our floor (cells, leaf, the child's stored facts,
    the parent's re-flight) uses this one value, so a crossing is timed at
    the first moment the arc comes down to R_f + g.
    """
    return R_EQ_KM + params.choices.own_margin_km + OWN_FLOOR_BAND_KM


def propagator(params: DuelParams) -> PropagatorSetting:
    try:
        return setting(params.choices.row_gap_setting)
    except KeyError as error:
        raise RefereeError(str(error)) from error


def opponent_floor_km(params: DuelParams, duration_s: float) -> float:
    """R_polar less one craft's integration error over the round (section 2.4).

    The opponent's crossing counts for us only where GMAT surely has it below
    R_polar, so our radius must be under R_polar by its error e(t).
    """
    error = max(propagator(params).position_error_km(duration_s),
                params.choices.dr_int_floor_km)
    return R_POLAR_KM - error


def row_gap_bound(arc_p: Arc, arc_e: Arc, params: DuelParams) -> float:
    """Section 2.6: the larger of the two craft's bounds, or the stated gap."""
    choices = params.choices
    if choices.row_gap_s_override is not None:
        return choices.row_gap_s_override
    c = propagator(params).row_gap_c
    root_mu = math.sqrt(choices.mu_km3_s2)
    bound = max(c * math.pow(arc.maximum_radius(), 1.5) / root_mu for arc in (arc_p, arc_e))
    if choices.max_step_s is not None:
        bound = min(bound, choices.max_step_s)
    return bound


def integration_margin_km(params: DuelParams, t_s: float) -> float:
    """Section 2.5: dr_int(t) = e_P(t) + e_E(t), each floored live; non-decreasing."""
    each = max(propagator(params).position_error_km(t_s), params.choices.dr_int_floor_km)
    return 2.0 * each


def capture_radius(params: DuelParams, view: View) -> Radius:
    """R_cap less the margin for us as pursuer, plus it as evader, bare to referee."""
    r_cap = params.announced.r_cap_km
    if view == "referee":
        return lambda _t: r_cap
    if view == "pursuer":
        return lambda t: max(0.0, r_cap - integration_margin_km(params, t))
    if view == "evader":
        return lambda t: r_cap + integration_margin_km(params, t)
    raise RefereeError(f"unknown view {view!r}")


def simulate_round(
    state: MatchState,
    pursuer_impulse_km_s: np.ndarray,
    evader_impulse_km_s: np.ndarray,
    params: DuelParams,
    *,
    view: View,
    floors_km: dict[str, float] | None = None,
) -> RoundFacts:
    """Fly one round from the announced state with the executed impulses.

    `view` picks the capture radius (`capture_radius`); `decide` refuses facts
    flown for another role.
    """
    if not 1 <= state.round_number <= DUEL.rounds:
        raise RefereeError(f"round {state.round_number} is outside 1 to {DUEL.rounds}")
    mu = params.choices.mu_km3_s2
    duration = params.announced.t_round_s
    for name, impulse in (("pursuer", pursuer_impulse_km_s), ("evader", evader_impulse_km_s)):
        norm = float(np.linalg.norm(impulse))
        if norm > params.dv_lim_km_s * (1.0 + 1.0e-12):
            raise RefereeError(f"the {name} impulse {norm} km/s was not projected onto the cap")
    arc_p = Arc(state.pursuer.r, state.pursuer.v + pursuer_impulse_km_s, duration, mu)
    arc_e = Arc(state.evader.r, state.evader.v + evader_impulse_km_s, duration, mu)
    radius = capture_radius(params, view)
    capture = capture_facts(arc_p, arc_e, radius, params.choices.knot_spacing_s)
    floors = floors_km if floors_km is not None else {
        "equatorial": R_EQ_KM,
        "polar": R_POLAR_KM,
        "opponent-floor": opponent_floor_km(params, duration),
        "own-floor": own_floor_km(params),
    }
    end_p = CraftState.of(*arc_p.state(duration))
    end_e = CraftState.of(*arc_e.state(duration))
    a_post = semi_major_axis(arc_e.r0, arc_e.v0, mu)
    energy_post = specific_energy(arc_e.r0, arc_e.v0, mu)
    energy_end = specific_energy(end_e.r, end_e.v, mu)
    # Scaled by the potential, or by the energy itself on a far open arc.
    tolerance = 1.0e-9 * max(mu / float(np.linalg.norm(end_e.r)), abs(energy_post))
    if abs(energy_end - energy_post) > tolerance:
        raise RefereeError(f"evader energy drifted by {energy_end - energy_post} km^2/s^2")
    return RoundFacts(
        round_number=state.round_number,
        duration_s=duration,
        view=view,
        capture_radius_start_km=radius(0.0),
        capture_radius_end_km=radius(duration),
        pursuer=arc_p,
        evader=arc_e,
        capture=capture,
        pursuer_below={name: below_radius(arc_p, r) for name, r in floors.items()},
        evader_below={name: below_radius(arc_e, r) for name, r in floors.items()},
        end_pursuer=end_p,
        end_evader=end_e,
        start_evader=state.evader,
        evader_a_km=a_post,
        evader_a_initial_km=semi_major_axis(state.evader.r, state.evader.v, mu),
        row_gap_bound_s=row_gap_bound(arc_p, arc_e, params),
    )


def established(
    passages: list[Passage], dwell_s: float, round_end_s: float, boundary_rows_judged: bool
) -> float | None:
    """When the earliest counting passage would surely show on rows `dwell_s` apart.

    A passage counts if it lasts `dwell_s`, or, under `boundary_rows_judged`
    (section 2.6), if it holds at a round boundary. It counts at min(its own
    entry + dwell, round end), the entry being that of the passage that
    qualifies, not of a shorter one before it; a passage held at the round
    start under the reading counts at 0. With `dwell_s` = 0 an event counts
    at its first entry.
    """
    best: float | None = None
    for passage in passages:
        at_start = passage.start_s <= 0.0
        at_end = passage.end_s >= round_end_s
        by_dwell = passage.duration_s >= dwell_s
        by_boundary = boundary_rows_judged and (at_start or at_end)
        if not (by_dwell or by_boundary):
            continue
        if boundary_rows_judged and at_start:
            when = 0.0
        else:
            when = min(passage.start_s + dwell_s, round_end_s)
        if best is None or when < best:
            best = when
    return best


@dataclass(frozen=True, slots=True)
class Band:
    lo_km: float
    hi_km: float
    #: Whether an escaping evader counts as inside [lo, inf) (Q3).
    escape_inside: bool = False
    #: The margin left no band, so this is the core of the real one (section
    #: 2.5, the second exception); the tool flags it.
    core: bool = False
    #: A hyperbolic band widened past zero energy (us as pursuer) also takes in
    #: the parabola and every bound a at least this large.
    bound_from_km: float | None = None

    def contains(self, a_km: float) -> bool:
        if self.bound_from_km is not None:
            if math.isinf(a_km):
                return True
            if a_km > 0.0:
                return a_km >= self.bound_from_km
            return self.lo_km <= a_km <= self.hi_km
        if math.isinf(self.hi_km) and self.lo_km > 0.0:
            if a_km < 0.0 or math.isinf(a_km):
                return self.escape_inside
            return a_km >= self.lo_km
        if math.isinf(a_km):
            return False
        return self.lo_km <= a_km <= self.hi_km


@dataclass(frozen=True, slots=True)
class Reading:
    order: Literal["priority", "chronological"]
    #: 0 for continuous checking; else the largest row gap assumed.
    capture_dwell_s: float
    pursuer_crossing_dwell_s: float
    evader_crossing_dwell_s: float
    #: Which entry of `RoundFacts.*_below` is each craft's surface.
    pursuer_surface: str
    evader_surface: str
    #: Which capture passages count: proved inside, or not proved outside.
    capture_passages: Literal["favour", "hurt"]
    early_entry_wins: bool = False
    #: Under early entry, whether a later capture or crossing in the round beats it.
    entry_beats_later_events: bool = True
    #: Under early entry, whether an evader inside before its round-1 impulse has entered.
    count_initial_state: bool = False
    #: Section 2.6: whether an event that holds at a round boundary counts
    #: whatever its dwell.
    boundary_rows_judged: bool = True


@dataclass(frozen=True, slots=True)
class Outcome:
    winner: Winner | None
    reason: str
    time_s: float | None = None
    #: True when the deciding event counts only because it holds at a round
    #: boundary (`boundary_rows_judged`); section 2.6 flags such wins.
    via_boundary: bool = False
    #: Set by `decide`: the band was the core fallback.
    core_band: bool = False
    #: Set by `decide`: the capture detector ran out of evaluations and left
    #: the rest undecided, which counts against us.
    detector_exhausted: bool = False


def judge(facts: RoundFacts, reading: Reading, band: Band) -> Outcome:
    """The outcome of one round under one reading; `winner` None means play on."""
    outcome = _judge(facts, reading, band)
    return dataclasses.replace(outcome, detector_exhausted=facts.capture.exhausted)


def _judge(facts: RoundFacts, reading: Reading, band: Band) -> Outcome:
    end = facts.duration_s
    passages = (facts.capture.passages_favour if reading.capture_passages == "favour"
                else facts.capture.passages_hurt)
    boundary = reading.boundary_rows_judged

    def when(found: list[Passage], dwell: float) -> tuple[float | None, bool]:
        at = established(found, dwell, end, boundary)
        without = established(found, dwell, end, False) if boundary else at
        # Flagged when the event counts only because of the boundary reading.
        return at, at is not None and without is None

    capture_at, capture_boundary = when(passages, reading.capture_dwell_s)
    pursuer_cross, pursuer_boundary = when(facts.pursuer_below[reading.pursuer_surface],
                                           reading.pursuer_crossing_dwell_s)
    evader_cross, evader_boundary = when(facts.evader_below[reading.evader_surface],
                                         reading.evader_crossing_dwell_s)
    entered = reading.early_entry_wins and (
        band.contains(facts.evader_a_km)
        or (reading.count_initial_state and facts.round_number == 1
            and band.contains(facts.evader_a_initial_km))
    )

    if entered and reading.entry_beats_later_events and capture_at != 0.0:
        return Outcome("evader", "entered the safe interval at the round start", 0.0)

    if reading.order == "priority":
        if capture_at is not None:
            return Outcome("pursuer", "capture", capture_at, capture_boundary)
        if pursuer_cross is not None and evader_cross is not None:
            return Outcome("unresolved", "both craft crossed the Earth (Q4)")
        if pursuer_cross is not None:
            return Outcome("evader", "the pursuer crossed the Earth", pursuer_cross,
                           pursuer_boundary)
        if evader_cross is not None:
            return Outcome("pursuer", "the evader crossed the Earth", evader_cross,
                           evader_boundary)
    else:
        events: list[tuple[float, int, Winner, str, bool]] = []
        if capture_at is not None:
            events.append((capture_at, 0, "pursuer", "capture", capture_boundary))
        if pursuer_cross is not None:
            events.append((pursuer_cross, 1, "evader", "the pursuer crossed the Earth",
                           pursuer_boundary))
        if evader_cross is not None:
            events.append((evader_cross, 1, "pursuer", "the evader crossed the Earth",
                           evader_boundary))
        if events:
            events.sort(key=lambda event: (event[0], event[1]))
            first = events[0]
            tied = [e for e in events if e[0] == first[0] and e[1] == first[1] and e is not first]
            if tied:
                return Outcome("unresolved", "both craft crossed the Earth at once (Q4)", first[0])
            return Outcome(first[2], first[3], first[0], first[4])

    if entered:
        return Outcome("evader", "entered the safe interval and kept clear in the round", 0.0)
    if facts.round_number == DUEL.rounds:
        if band.contains(facts.evader_a_km):
            return Outcome("evader", f"a = {facts.evader_a_km:.6f} km is inside the interval")
        return Outcome("pursuer", f"a = {facts.evader_a_km:.6f} km is outside the interval")
    return Outcome(None, "no event; next round")


def margin_km(params: DuelParams, r: np.ndarray, v: np.ndarray, a_km: float) -> float:
    """Section 2.5: twice the error terms of a near `a_km`, for a state at (r, v), floored.

    The terms are GMAT's integration error in a over a round (measured per
    propagator setting, `a_integration_km` overriding) and a computed from the
    published, rounded state: a rounding of half a unit per component moves a
    vector by up to sqrt(3)/2 of a unit, the energy by |v| dv + mu/r^2 dr, and
    a by 2a^2/mu times that. The margin only matters for an evader whose a is
    near a band edge, so a is the edge's, not the state's: a state far from
    the band would otherwise get a margin as large as its own a squared.
    """
    choices = params.choices
    mu = choices.mu_km3_s2
    rounding = 2.0 * a_km * a_km / mu * _rounding_energy(params, r, v)
    integration = choices.a_integration_km
    if integration is None:
        integration = propagator(params).a_error_km(params.announced.t_round_s)
    terms = integration + rounding
    return max(2.0 * terms, choices.a_margin_min_km, choices.a_margin_floor_km)


def _rounding_energy(params: DuelParams, r: np.ndarray, v: np.ndarray) -> float:
    """How far rounding the published state can move its energy, km^2/s^2."""
    choices = params.choices
    radius = float(np.linalg.norm(r))
    speed = float(np.linalg.norm(v))
    half_vector = math.sqrt(3.0) / 2.0
    return half_vector * (speed * choices.file_velocity_km_s
                          + choices.mu_km3_s2 / (radius * radius) * choices.file_position_km)


def edge_energy_margin(params: DuelParams, states: list[tuple[np.ndarray, np.ndarray]],
                       edge_km: float) -> float:
    """The margin at one band edge in energy, km^2/s^2: margin_km times mu / (2a^2).

    An open edge (a = inf, energy 0) takes the km terms at a_lo, the smallest
    a in the band, where a km of a is the most energy.
    """
    a_ref = edge_km if math.isfinite(edge_km) else params.announced.a_lo_km
    worst = max(margin_km(params, r, v, a_ref) for r, v in states)
    return worst * params.choices.mu_km3_s2 / (2.0 * a_ref * a_ref)


def _a_of(energy: float, mu: float) -> float:
    return math.inf if energy == 0.0 else -mu / (2.0 * energy)


def band_for(params: DuelParams, role: Role, states: list[tuple[np.ndarray, np.ndarray]]) -> Band:
    """The interval, narrowed for us as evader and widened for us as pursuer.

    Each edge moves in energy, eps = -mu/(2a), by the largest margin over
    `states` evaluated at that edge (`edge_energy_margin`): a is judged on the
    published end state, and the rounding term depends on where on the orbit
    that is, so a caller passes every state the judgement could be made on.
    Moving in energy keeps a huge or open upper edge right: an evader barely
    bound can read as escaping once its end state is rounded (Q3).
    """
    mu = params.choices.mu_km3_s2
    lo, hi = params.announced.a_lo_km, params.announced.a_hi_km
    eps_lo = -mu / (2.0 * lo)
    eps_hi = 0.0 if math.isinf(hi) else -mu / (2.0 * hi)
    margin_lo = edge_energy_margin(params, states, lo)
    margin_hi = edge_energy_margin(params, states, hi)
    bound_side = lo > 0.0  # elliptic band (energy < 0), else hyperbolic (energy > 0)
    if role == "evader":
        inner_lo, inner_hi = eps_lo + margin_lo, eps_hi - margin_hi
        # Both moved edges stay on the band's side of zero energy, since each
        # moves inward from its own edge.
        if inner_lo < inner_hi:
            return Band(_a_of(inner_lo, mu), _a_of(inner_hi, mu), escape_inside=False)
        # The core of the real band, centre +- 10% of its half-width in
        # energy, so an open band [lo, inf) has one too (about 2 lo).
        centre, half = 0.5 * (eps_lo + eps_hi), 0.5 * (eps_hi - eps_lo)
        return Band(_a_of(centre - 0.1 * half, mu), _a_of(centre + 0.1 * half, mu),
                    escape_inside=False, core=True)
    outer_lo, outer_hi = eps_lo - margin_lo, eps_hi + margin_hi
    if bound_side:
        # Past zero energy the upper edge is open, and escapes count (Q3).
        wide_hi = _a_of(outer_hi, mu) if outer_hi < 0.0 else math.inf
        return Band(_a_of(outer_lo, mu), wide_hi, escape_inside=True)
    if outer_lo > 0.0:
        return Band(_a_of(outer_lo, mu), _a_of(outer_hi, mu), escape_inside=True)
    # A hyperbolic band widened past zero energy: every hyperbola up to the
    # upper edge, the parabola, and bound orbits down to energy outer_lo.
    bound_from = math.inf if outer_lo == 0.0 else _a_of(outer_lo, mu)
    return Band(-math.inf, _a_of(outer_hi, mu), escape_inside=True, bound_from_km=bound_from)


def decide(facts: RoundFacts, role: Role, params: DuelParams) -> Outcome:
    """The cell value for us: every open reading resolved against `role`.

    `facts` must have been simulated for this role (`view`), so the capture
    radius is R_cap shrunk by the margin for us as pursuer and grown for us as
    evader, with the floors "own-floor" (ours) and "opponent-floor" (theirs).

    `via_boundary` is set when the verdict differs from the one reached
    without `boundary_rows_judged`, under either event order (section 2.6).
    """
    if facts.view != role:
        raise RefereeError(f"facts flown for view {facts.view!r} cannot decide for the {role}")
    end, start = facts.end_evader, facts.start_evader
    states = [(facts.evader.r0, facts.evader.v0), (end.r, end.v)]
    if role == "pursuer" and facts.round_number == 1 and params.readings.early_entry_wins:
        # The initial state counts as an entry against us (section 3.6).
        states.append((start.r, start.v))
    band = band_for(params, role, states)
    boundary = params.readings.boundary_rows_judged
    worst = _worst(facts, role, params, band, boundary)
    via = boundary and _worst(facts, role, params, band, False).winner != worst.winner
    return dataclasses.replace(worst, via_boundary=via, core_band=band.core,
                               detector_exhausted=facts.capture.exhausted)


def _worst(facts: RoundFacts, role: Role, params: DuelParams, band: Band,
           boundary_rows_judged: bool) -> Outcome:
    gap = facts.row_gap_bound_s
    pursuer = role == "pursuer"
    outcomes = [
        judge(
            facts,
            Reading(
                order=order,
                capture_dwell_s=gap if pursuer else 0.0,
                pursuer_crossing_dwell_s=0.0 if pursuer else gap,
                evader_crossing_dwell_s=gap if pursuer else 0.0,
                pursuer_surface="own-floor" if pursuer else "opponent-floor",
                evader_surface="opponent-floor" if pursuer else "own-floor",
                capture_passages="favour" if pursuer else "hurt",
                early_entry_wins=params.readings.early_entry_wins,
                entry_beats_later_events=pursuer,
                count_initial_state=pursuer,
                boundary_rows_judged=boundary_rows_judged,
            ),
            band,
        )
        for order in ("priority", "chronological")
    ]
    ours = "pursuer" if role == "pursuer" else "evader"

    def badness(outcome: Outcome) -> int:
        if outcome.winner == ours:
            return 0
        if outcome.winner is None:
            return 1
        return 2  # a loss, or unresolved, which counts as one

    worst = max(outcomes, key=badness)
    if worst.winner == "unresolved":
        worst = Outcome("evader" if role == "pursuer" else "pursuer", worst.reason, worst.time_s)
    return worst
