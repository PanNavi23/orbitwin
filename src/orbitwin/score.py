from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeGuard, assert_never, overload

from .config import ScoreConfig
from .errors import ConfigurationError
from .models import Candidate, Scenario, uses_head_start
from .rules import DistanceTerm, InterceptionRules, interception_rules


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    distance_score: float
    time_score: float
    delta_v_score: float
    penalty_score: float
    raw_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "distance_score": self.distance_score,
            "time_score": self.time_score,
            "delta_v_score": self.delta_v_score,
            "penalty_score": self.penalty_score,
            "raw_score": self.raw_score,
        }


#: Past this exponent a logistic term is taken as 0, or its full weight; exp overflows at 709.
_EXP_GUARD = 700.0


def _logistic_descending(weight: float, exponent: float) -> float:
    if exponent >= 0.0:
        if exponent > _EXP_GUARD:
            return 0.0
        e_negative = math.exp(-exponent)
        return weight * e_negative / (1.0 + e_negative)
    if exponent < -_EXP_GUARD:
        return weight
    return weight / (1.0 + math.exp(exponent))


def _dv_for_score(total_dv_km_s: float, unit: str) -> float:
    return total_dv_km_s * 1000.0 if unit == "m/s" else total_dv_km_s


def _announced_cr(rules: InterceptionRules, cr_km: float | None) -> float:
    if cr_km is None or not (math.isfinite(cr_km) and cr_km > 0.0):
        raise ConfigurationError(
            f"score.cr_km is {cr_km!r}; the {rules.name} rules' distance term "
            f"({rules.distance_clause}) needs the announced C_r in km, which is never guessed"
        )
    return cr_km


def distance_term(distance_km: float, rules: InterceptionRules, cr_km: float | None) -> float:
    """Points for Δr_min = max(distance_km, sphere); none for an unknown distance."""
    if not math.isfinite(distance_km):
        return 0.0
    excess = max(distance_km, rules.intercept_radius_km) - rules.intercept_radius_km
    match rules.distance_term:
        case DistanceTerm.EXPONENTIAL:
            return 50.0 * math.exp(-excess / 100.0)
        case DistanceTerm.LOGISTIC:
            u = excess / _announced_cr(rules, cr_km)
            return _logistic_descending(100.0, u * u)
        case _:
            assert_never(rules.distance_term)


def distance_term_slope(distance_km: float, rules: InterceptionRules, cr_km: float | None) -> float:
    """dD/dd in points per km, from above at the sphere; 0 inside it."""
    sphere = rules.intercept_radius_km
    if not math.isfinite(distance_km) or distance_km < sphere:
        return 0.0
    value = distance_term(distance_km, rules, cr_km)
    match rules.distance_term:
        case DistanceTerm.EXPONENTIAL:
            return -value / 100.0
        case DistanceTerm.LOGISTIC:
            cr = _announced_cr(rules, cr_km)
            u = (distance_km - sphere) / cr
            return -value * (1.0 - value / 100.0) * 2.0 * u / cr
        case _:
            assert_never(rules.distance_term)


def distance_for_term_km(points: float, rules: InterceptionRules, cr_km: float | None) -> float:
    """The distance at which the term falls to ``points``, for 0 <= points < 50.

    Below the smallest term the formula yields, and at 0, it is the edge past
    which the term is exactly 0: a miss beyond it scores 0 there and only ties.
    """
    if not 0.0 <= points < 50.0:
        raise ValueError(f"a miss's distance term lies in [0, 50), not {points!r}")
    sphere = rules.intercept_radius_km
    match rules.distance_term:
        case DistanceTerm.EXPONENTIAL:
            # e^{-x/100} is above 0 down to the smallest subnormal.
            return sphere - 100.0 * math.log(max(points / 50.0, math.ulp(0.0)))
        case DistanceTerm.LOGISTIC:
            cr = _announced_cr(rules, cr_km)
            if points == 0.0:
                return sphere + cr * math.sqrt(_EXP_GUARD)
            # 100/(1 + e^{u^2}) = points at u^2 = ln(100/points - 1), never past the guard.
            u2 = min(math.log(100.0 - points) - math.log(points), _EXP_GUARD)
            return sphere + cr * math.sqrt(u2)
        case _:
            assert_never(rules.distance_term)


def on_solver_clock(score_config: ScoreConfig, scenario: Scenario) -> ScoreConfig:
    """The coefficients with Ct moved onto the solver clock.

    T_team runs from policy.mission_time_origin's epoch, Δ before the solver's
    t = 0 when that craft flies the alignment, so kt (t + Δ - Ct) is kt (t - Ct').
    """
    offset_s = scenario.time_origin_offset_s
    return dataclasses.replace(score_config, ct=score_config.ct - offset_s) if offset_s else score_config


@dataclass(frozen=True, slots=True)
class Ceilings:
    """What no answer, and no miss, can score above; a bound, not a reachable point."""

    any_answer: float
    any_miss: float
    #: The best of each term over its whole range, which a miss needs to reach a score.
    best_dv_term: float
    time_term_at_horizon: float
    best_time_term: float
    #: score.cr_km, which the ranking rules' distance term is inverted with.
    cr_km: float | None = None

    def miss_must_be_within_km(
        self, best: ScoreBreakdown, rules: InterceptionRules
    ) -> float | None:
        """How close a miss with the best delta-v term must come to outscore ``best``;
        None when none can, inf when any can.

        Taken from the terms, not the total: R2's zero burn passes 4711 km off, and its
        distance term, 1.8e-19 under the preliminary's term and 1.3e-105 at C_r 300 km,
        vanishes once added to 23.67. At C_r 100 km it is exactly 0, and a miss then
        beats it only inside the edge where its own term is still above 0.
        """
        if best.raw_score == 0.0:
            # Floored at zero (§4 note 1(d)): any positive score beats it.
            return math.inf
        rest = (best.distance_score + (best.time_score - self.time_term_at_horizon)
                + (best.delta_v_score - self.best_dv_term) - best.penalty_score)
        if rest >= 50.0:
            return None
        if rest < 0.0:
            return math.inf
        return distance_for_term_km(rest, rules, self.cr_km)


def ceilings(score_config: ScoreConfig, horizon_s: float, total_dv_cap_km_s: float) -> Ceilings:
    """Both terms are monotone in their argument, so each is best at an end of its range.
    The distance term is at most 50 for any miss under either rulebook, at d -> 5.

    ``horizon_s`` and ``score_config.ct`` must be on one clock: the announced
    one, or the solver's after `on_solver_clock`.
    """
    time_ends = [_logistic_descending(25.0, score_config.kt * (t - score_config.ct))
                 for t in (0.0, horizon_s)]
    dv_ends = [
        _logistic_descending(
            25.0, score_config.kv * (_dv_for_score(dv, score_config.delta_v_unit) - score_config.cv)
        )
        for dv in (0.0, total_dv_cap_km_s)
    ]
    best_dv = max(dv_ends)
    best_time = max(time_ends)
    return Ceilings(50.0 + best_time + best_dv, 50.0 + time_ends[1] + best_dv,
                    best_dv, time_ends[1], best_time, score_config.cr_km)


def scored_distance_km(candidate: Candidate, scenario: Scenario) -> float:
    """The distance the term is priced on, before the 5 km floor; inf when unknown."""
    # The preliminary scores Δr at completion (§4); the ranking rules the closest
    # approach over [0, T_max] (eq. 2, 3), which a failure carries measured to T_max.
    rules = interception_rules(scenario.rules)
    scored = (
        candidate.minimum_distance_km
        if rules.miss_distance == "closest-approach"
        else candidate.distance_at_completion_km
    )
    distance = float(scored) if scored is not None else float("inf")
    window = alignment_window_km(candidate, _alignment_minimum_km(scenario))
    return distance if window is None else min(distance, window)


#: Candidate.metadata key: the alignment's closest approach under its own head-start burns.
HEAD_START_ALIGNMENT_KEY = "head_start_alignment_distance_km"


def alignment_window_km(
    candidate: Candidate, alignment_minimum_distance_km: float | None
) -> float | None:
    """The alignment's closest approach d_min counts for `candidate`: the scenario's
    burn-free one, or with head-start burns its own (HEAD_START_ALIGNMENT_KEY),
    left out while unmeasured, which can only price the miss lower."""
    if alignment_minimum_distance_km is None or not uses_head_start(candidate.burns):
        return alignment_minimum_distance_km
    own = candidate.metadata.get(HEAD_START_ALIGNMENT_KEY)
    return None if own is None else float(own)


def _alignment_minimum_km(scenario: Scenario) -> float | None:
    """The alignment's closest approach where d_min counts it, under
    policy.closest_approach_window "from_origin"; None elsewhere."""
    if interception_rules(scenario.rules).miss_distance != "closest-approach":
        return None
    return scenario.alignment_minimum_distance_km


def scored_closest_approach_km(
    candidate: Candidate, alignment_minimum_distance_km: float | None
) -> float:
    """What a miss's distance term prices under the ranking rules, as
    scored_distance_km does; inf unless its closest approach was measured."""
    measured = measured_minimum_distance_km(candidate)
    window = alignment_window_km(candidate, alignment_minimum_distance_km)
    if window is None or not math.isfinite(measured):
        return measured
    return min(measured, window)


def _logistic_slope(weight: float, exponent: float) -> float:
    """-d/dz of weight / (1 + e^z), i.e. F (1 - F/weight) without its cancellation near
    weight; 0 where `_logistic_descending` saturates, so the two agree there."""
    if abs(exponent) > 700.0:
        return 0.0
    e_negative = math.exp(-abs(exponent))
    return weight * e_negative / (1.0 + e_negative) ** 2


@dataclass(frozen=True, slots=True)
class Levers:
    """Where one row's missing points are, and what each term pays for them.

    Headroom is each term's best over its range less the row's value; the
    deductions are headroom too, and ``total_headroom`` is ``any_answer`` less the
    unfloored score. The rates are marginal: exact derivatives at the row, which a
    250 m/s impulse already bends by up to 11 %. What whole impulses cost is in
    the ``*_points_lost`` fields, taken from the score itself.
    """

    hit: bool
    #: The unfloored score is below 0, so no small change moves the score: every rate is 0.
    floored: bool
    distance_headroom: float
    time_headroom: float
    delta_v_headroom: float
    penalty_headroom: float
    total_headroom: float
    #: -dS/dT_team. None for a miss: it completes at T_max whatever it flies, so its time
    #: term moves only by becoming a hit, worth up to ``time_headroom``.
    points_per_s_earlier: float | None
    #: -dS/dΔV per km/s, whatever unit Cv is announced in.
    points_per_km_s_saved: float
    #: -dS/dΔr_min for a miss, 0 at the sphere under the ranking rules; None for a hit.
    points_per_km_closer: float | None
    #: The rate at which delta-v must buy time to pay for itself, s per km/s, negative
    #: when more delta-v pays on its own; None for a miss, a floored score or kt <= 0,
    #: where an earlier arrival is not what pays.
    break_even_s_per_km_s: float | None
    #: One more impulse at the policy's safe per-impulse limit, and the score it costs.
    impulse_km_s: float
    impulse_points_lost: float
    #: The rest of the safe total budget, and the score spending all of it costs.
    remaining_budget_km_s: float
    remaining_budget_points_lost: float

    def to_dict(self) -> dict[str, object]:
        notes = []
        if not self.hit:
            notes.append("A miss completes at T_max, so its time term moves only by becoming "
                         "a hit, worth up to headroom.time.")
        if self.floored:
            notes.append("The score is floored at 0, so no small change moves it and every "
                         "rate is 0.")
        if self.hit and not self.floored and self.break_even_s_per_km_s is None:
            notes.append("No break-even: the time term does not reward arriving earlier here.")
        return {
            "hit": self.hit,
            "floored": self.floored,
            "notes": notes,
            "headroom": {
                "distance": self.distance_headroom,
                "time": self.time_headroom,
                "delta_v": self.delta_v_headroom,
                "penalties": self.penalty_headroom,
                "total": self.total_headroom,
            },
            "marginal": {
                "points_per_s_earlier": self.points_per_s_earlier,
                "points_per_km_s_saved": self.points_per_km_s_saved,
                "points_per_km_closer": self.points_per_km_closer,
                "break_even_s_per_km_s": self.break_even_s_per_km_s,
            },
            "spent": {
                "one_more_impulse": {"km_s": self.impulse_km_s,
                                     "points_lost": self.impulse_points_lost},
                "remaining_budget": {"km_s": self.remaining_budget_km_s,
                                     "points_lost": self.remaining_budget_points_lost},
            },
        }


def _plain(value: float) -> float:
    """``value`` with -0.0 made 0.0, which JSON would otherwise print as -0.0."""
    return value + 0.0


def levers(
    candidate: Candidate,
    breakdown: ScoreBreakdown,
    scenario: Scenario,
    score_config: ScoreConfig,
    total_dv_cap_km_s: float,
    *,
    impulse_km_s: float,
    budget_km_s: float,
) -> Levers:
    """∂S/∂T, ∂S/∂ΔV and ∂S/∂d at ``candidate``, each term's headroom, and what one more
    ``impulse_km_s``, and the rest of ``budget_km_s``, would cost.

    ``breakdown`` is its `score_candidate` result as ranked; ``score_config`` is the
    announced one, moved onto the solver clock here as `score_candidate` does.
    """
    clock = on_solver_clock(score_config, scenario)
    bound = ceilings(clock, scenario.max_time_s, total_dv_cap_km_s)
    distance, time, delta_v, penalty = (breakdown.distance_score, breakdown.time_score,
                                        breakdown.delta_v_score, breakdown.penalty_score)
    unfloored = distance + time + delta_v - penalty
    # At exactly 0 an improvement still raises the score.
    floored = unfloored < 0.0
    hit = candidate.intercept_time_s is not None
    per_s: float | None = None
    per_km: float | None = None
    if floored:
        per_km_s = 0.0
        per_s = 0.0 if hit else None
        per_km = None if hit else 0.0
    else:
        dv = _dv_for_score(candidate.total_dv_km_s, score_config.delta_v_unit)
        slope = _logistic_slope(25.0, score_config.kv * (dv - score_config.cv))
        # kv is per Cv's unit, scaled here to per km/s.
        per_km_s = score_config.kv * _dv_for_score(1.0, score_config.delta_v_unit) * slope
        if candidate.intercept_time_s is not None:
            completion_s = float(candidate.intercept_time_s)
            per_s = clock.kt * _logistic_slope(25.0, clock.kt * (completion_s - clock.ct))
        else:
            per_km = -distance_term_slope(scored_distance_km(candidate, scenario),
                                          interception_rules(scenario.rules), score_config.cr_km)
    break_even = per_km_s / per_s if per_s and clock.kt > 0.0 else None

    def lost(extra_km_s: float) -> float:
        # score_candidate's own delta-v term, sum and floor: exact, not linearised.
        spent = _dv_for_score(candidate.total_dv_km_s + extra_km_s, score_config.delta_v_unit)
        after = _logistic_descending(25.0, score_config.kv * (spent - score_config.cv))
        return max(0.0, unfloored) - max(0.0, distance + time + after - penalty)

    remaining_km_s = max(0.0, budget_km_s - candidate.total_dv_km_s)
    return Levers(
        hit=hit,
        floored=floored,
        distance_headroom=_plain(50.0 - distance),
        time_headroom=_plain(bound.best_time_term - time),
        delta_v_headroom=_plain(bound.best_dv_term - delta_v),
        penalty_headroom=_plain(penalty),
        total_headroom=_plain(bound.any_answer - unfloored),
        points_per_s_earlier=None if per_s is None else _plain(per_s),
        points_per_km_s_saved=_plain(per_km_s),
        points_per_km_closer=None if per_km is None else _plain(per_km),
        break_even_s_per_km_s=(
            _plain(break_even) if break_even is not None and math.isfinite(break_even) else None),
        impulse_km_s=_plain(impulse_km_s),
        impulse_points_lost=_plain(lost(impulse_km_s)),
        remaining_budget_km_s=_plain(remaining_km_s),
        remaining_budget_points_lost=_plain(lost(remaining_km_s)),
    )


def score_candidate(
    candidate: Candidate,
    scenario: Scenario,
    score_config: ScoreConfig,
    *,
    penalty_count: int | None = None,
) -> ScoreBreakdown:
    completion_time = (
        candidate.intercept_time_s
        if candidate.intercept_time_s is not None
        else scenario.max_time_s
    )
    rules = interception_rules(scenario.rules)
    distance_score = distance_term(
        scored_distance_km(candidate, scenario), rules, score_config.cr_km
    )
    clock = on_solver_clock(score_config, scenario)
    time_score = _logistic_descending(25.0, clock.kt * (float(completion_time) - clock.ct))
    scored_dv = _dv_for_score(candidate.total_dv_km_s, score_config.delta_v_unit)
    dv_score = _logistic_descending(25.0, score_config.kv * (scored_dv - score_config.cv))
    count = len(candidate.penalties) if penalty_count is None else penalty_count
    penalty = rules.penalty_points * count
    raw = max(0.0, distance_score + time_score + dv_score - penalty)
    return ScoreBreakdown(distance_score, time_score, dv_score, penalty, raw)


def apply_score(
    candidate: Candidate,
    scenario: Scenario,
    score_config: ScoreConfig | None,
    *,
    penalty_count: int | None = None,
) -> None:
    """Recompute and store the official score in place.

    Every path that mutates ``intercept_time_s``, ``distance_at_completion_km``,
    ``minimum_distance_km`` (the ranking rules score it) or ``total_dv_km_s``
    has to call this. Otherwise the candidate keeps a score
    that describes a state it no longer has, and ranking reads the stale value.
    """
    if score_config is None:
        return
    breakdown = score_candidate(candidate, scenario, score_config, penalty_count=penalty_count)
    candidate.score = breakdown.raw_score
    candidate.score_components = breakdown.to_dict()
    candidate.score_basis = candidate.current_score_basis


def measured_minimum_distance_km(candidate: Candidate) -> float:
    """Return the trajectory minimum distance, or infinity when it is unknown.

    Before event refinement, ``minimum_distance_km`` holds the Lambert endpoint
    residual, which is around 1e-10 km for any converged arc. Treating that as a
    measured closest approach would let an unrefined candidate win the official
    minimum-distance tie-break against a candidate that was actually evaluated.
    """
    if candidate.metadata.get("event_refined") is not True:
        return math.inf
    if candidate.minimum_distance_km is None:
        return math.inf
    return float(candidate.minimum_distance_km)


def tie_break_distance_km(candidate: Candidate, scenario: Scenario | None) -> float:
    """§6's d_min for a tie: the measured minimum over the same window as the scored one.

    Unmeasured stays infinite, so an unrefined candidate never wins a tie. None
    is a caller with no scenario, which has no window to apply.
    """
    measured = measured_minimum_distance_km(candidate)
    window = _alignment_minimum_km(scenario) if scenario is not None else None
    if window is None or math.isinf(measured):
        return measured
    return min(measured, window)


def ranking_key(
    candidate: Candidate, *, scenario: Scenario | None, score: float | None = None
) -> tuple[float, float, float, float, str]:
    """Rank by official score, then by the official tie-break order.

    COMPETITION_POLICY.md records the tie order as actual minimum distance,
    total delta-v, completion time, then the design explanation. Only the first
    three are machine-comparable, so the candidate id breaks the remainder
    deterministically instead of leaving the sort unstable.

    `score` overrides the stored value, so a caller can rank an archive under
    coefficients it is only trying out without writing those trial scores onto
    the candidates. `scenario` gives the distance its closest_approach_window.
    """
    supplied = candidate.score if score is None else score
    effective_score = supplied if supplied is not None else -math.inf
    intercept = candidate.intercept_time_s if candidate.intercept_time_s is not None else math.inf
    return (
        -effective_score,
        tie_break_distance_km(candidate, scenario),
        candidate.total_dv_km_s,
        intercept,
        candidate.candidate_id,
    )


#: The score block's fields, in the order messages name them.
SCORE_BLOCK_FIELDS = ("kt", "ct", "kv", "cv", "cr_km", "time_unit", "delta_v_unit")


@overload
def score_stamp(score_config: ScoreConfig) -> dict[str, float | str | None]: ...
@overload
def score_stamp(score_config: None) -> None: ...
@overload
def score_stamp(score_config: ScoreConfig | None) -> dict[str, float | str | None] | None: ...
def score_stamp(score_config: ScoreConfig | None) -> dict[str, float | str | None] | None:
    """The score block a search or a certification ran under, as files record it.

    None is a run without one: solve then orders by distance and delta-v, not by a score.
    """
    if score_config is None:
        return None
    return {name: getattr(score_config, name) for name in SCORE_BLOCK_FIELDS}


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _same_field(recorded: object, current: object) -> bool:
    """Equal, or numbers within 1e-12 of each other: 3000 from YAML is 3000.0."""
    if _is_number(recorded) and _is_number(current):
        return math.isclose(float(recorded), float(current), rel_tol=1.0e-12, abs_tol=0.0)
    return recorded == current


def _shown_value(value: object) -> str:
    return f"{float(value):.12g}" if _is_number(value) else repr(value)


def _shown_field(name: str, shown: str, block: Mapping[str, object]) -> str:
    match name:
        case "ct":
            return f"Ct {shown} {block.get('time_unit')}"
        case "cv":
            return f"Cv {shown} {block.get('delta_v_unit')}"
        case "cr_km":
            return f"C_r {shown} km"
        case "time_unit":
            return f"time unit {shown}"
        case "delta_v_unit":
            return f"delta-v unit {shown}"
        case _:
            return f"{name} {shown}"


def describe_score_stamp(stamp: object, names: Sequence[str] = SCORE_BLOCK_FIELDS) -> str:
    """`names` of a recorded score block, for a message."""
    if stamp is None:
        return "no score block (unscored)"
    if not isinstance(stamp, Mapping):
        return f"an unreadable score block {stamp!r}"
    return ", ".join(_shown_field(name, _shown_value(stamp.get(name)), stamp) for name in names)


def score_stamp_difference(
    recorded: object,
    current: ScoreConfig | Mapping[str, object] | None,
    names: Sequence[str] = SCORE_BLOCK_FIELDS,
) -> tuple[str, str] | None:
    """How a recorded score block differs from ``current`` on ``names``, as (recorded,
    current) text naming only the fields that differ; None when they agree."""
    block = score_stamp(current) if isinstance(current, ScoreConfig) else current
    if recorded is None and block is None:
        return None
    if recorded is None or block is None or not isinstance(recorded, Mapping):
        return describe_score_stamp(recorded), describe_score_stamp(block)
    differing = [name for name in names if not _same_field(recorded.get(name), block.get(name))]
    if not differing:
        return None
    then: list[str] = []
    now: list[str] = []
    for name in differing:
        old, new = _shown_value(recorded.get(name)), _shown_value(block.get(name))
        if old == new:
            # Apart by more than 1e-12 yet alike in 12 digits.
            old, new = repr(recorded.get(name)), repr(block.get(name))
        then.append(_shown_field(name, old, recorded))
        now.append(_shown_field(name, new, block))
    return ", ".join(then), ", ".join(now)


def normalised_stamp(stamp: object) -> object:
    """A recorded block with its numbers as floats, so 3000 and 3000.0 group as one."""
    if not isinstance(stamp, Mapping):
        return stamp
    return {key: float(value) if _is_number(value) else value for key, value in stamp.items()}
