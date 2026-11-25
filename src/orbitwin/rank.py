"""Re-rank a saved archive once the organizer announces (kt, Ct, kv, Cv), and C_r
under the ranking rules.

初賽規則 §5 scores a run as a distance term plus two logistic terms, one in
completion time and one in total delta-v. For any successful interception the
distance term is exactly 50 -- `tint` is by definition the first instant with
`Δr <= 5`, so `Δrmin` is 5 and `50*exp(0)` is 50 (the ranking rules' revised
term, 100/(1 + e^0), is 50 too). The competition among teams
that all succeed is therefore decided entirely by the two logistics, whose
centres and steepnesses are announced separately (briefing deck p.8) and are not
known while the search runs.

Measured on a real 120-candidate archive from `DifferentPlane_Circular`: the
eight Pareto-frontier points reduce to two that ever win, and which of the two
wins flips across a plausible sweep of the four coefficients -- 55.8% to 44.2%
over 625 combinations. Committing to one of them in advance costs up to 25
points, the whole weight of a logistic term; the best single advance bet still
gives up 10.5 points in its worst case. Keeping the frontier and choosing after
the announcement costs nothing.

That is what this module is for. It does not re-solve, and that is all it
can do for an archive searched under other coefficients: the ranking search
optimises the score it is given, so rank warns when the block changed, and the
answer is to search again.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable, Sequence
from typing import Any, TypedDict, assert_never

from .config import OrbitWinConfig, ScoreConfig
from .constraints import (
    FORMAL_FRONTIER,
    ROBUST_FRONTIER,
    apply_constraints,
    below_minimum_flown_altitude,
    flies_through_central_body,
    frontier_of,
    gmat_contradicts_hit,
    gmat_verified_on_record,
    interception_required,
    prefer_deep_hits,
    shallow_stop,
    stop_depth_mismatch,
    validate_candidate,
)
from .errors import OrbitWinError
from .gmat_failures import gmat_failures, pre_gmat_score, verification_error
from .mission_clock import clock_record
from .models import (
    Candidate,
    CandidateStage,
    Scenario,
    clock_stamp_mismatch,
    head_start_lead_s,
    head_start_status,
    uses_head_start,
)
from .rules import DistanceTerm, InterceptionRules, interception_rules
from .score import (
    ScoreBreakdown,
    alignment_window_km,
    apply_score,
    ceilings,
    distance_for_term_km,
    distance_term,
    levers,
    measured_minimum_distance_km,
    normalised_stamp,
    on_solver_clock,
    ranking_key,
    score_candidate,
    score_stamp,
    score_stamp_difference,
    scored_distance_km,
    tie_break_distance_km,
)
from .script_renderer import report_name_verdict, team_number_warnings

#: A logistic term carries 25 points. When the whole field sits inside this many
#: points of each other on one term, that term decides nothing, and the ranking
#: is really being made by the other one.
FLAT_TERM_POINTS = 0.5

#: Default relative nudge applied to each announced coefficient to see whether
#: the winner is sitting on a knife edge. This is not a forecast of what the
#: organizer will announce -- the announcement is exact. It is a check on
#: transcription: if a tenth of the value changes who we submit, the number is
#: worth reading twice.
DEFAULT_TOLERANCE = 0.10

_COEFFICIENTS = ("kt", "ct", "kv", "cv", "cr_km")

#: What a GMAT run in solve leaves on any candidate it flew, verified or not.
_GMAT_TRACES = (
    "gmat_event", "gmat_verification_error", "gmat_first_flight_closest_km", "gmat_correction",
)

NO_GMAT_ARCHIVE = (
    "No candidate in this archive carries a GMAT run: it comes from a solve run without "
    "--gmat-console and --startup-file. Under the ranking rules no row is submittable, since "
    "certify may fly any of them as a miss. Run solve again with GMAT (GMAT_CONSOLE and "
    "GMAT_STARTUP for scripts/competition_day.sh)."
)
#: Rows solve never marked renderable: scripts/hunt_ranking_score.py writes those.
NO_GMAT_HUNT = (
    "No candidate in this archive carries a GMAT run, and none carries solve's renderable "
    "mark: it comes from a tool that flies nothing in GMAT, such as "
    "scripts/hunt_ranking_score.py. Under the ranking rules no row is submittable as it stands. "
    "Certify the row you want, which refuses it if GMAT flies it worse than planned, or run "
    "solve with GMAT."
)


class WinnerStability(TypedDict):
    """What `winner_stability` answers, with each answer's type stated.

    A plain `dict[str, object]` says the sweep returns five things and nothing
    about what any of them is, so a reader -- and a checker -- cannot tell
    `not_tested`, a list of coefficient names, from `winner`, a single id. This
    is the same dict at run time; it is written down instead of inferred.
    """

    tolerance: float
    #: In rank_report the best submittable row, None when no row is.
    winner: str | None
    stable: bool
    changes: list[dict[str, object]]
    not_tested: list[str]


def _coefficients(score_config: ScoreConfig) -> tuple[str, ...]:
    """The announced ones: C_r only under the ranking rules, where it is required."""
    return tuple(name for name in _COEFFICIENTS if getattr(score_config, name) is not None)


def _with_coefficient(score_config: ScoreConfig, name: str, value: float) -> ScoreConfig:
    """`score_config` with one of the announced coefficients replaced.

    `dataclasses.replace(score_config, **{name: value})` is the same line, but a
    `str` key could equally name `time_unit` or `delta_v_unit` and set a unit to
    a float -- and a unit that is a number is precisely the mistake §5 scoring
    cannot survive, since `Cv` in km/s rather than m/s saturates the delta-v
    term. Naming the four makes it true, rather than intended, that only a
    coefficient moves.
    """
    match name:
        case "kt":
            return dataclasses.replace(score_config, kt=value)
        case "ct":
            return dataclasses.replace(score_config, ct=value)
        case "kv":
            return dataclasses.replace(score_config, kv=value)
        case "cv":
            return dataclasses.replace(score_config, cv=value)
        case "cr_km":
            return dataclasses.replace(score_config, cr_km=value)
    # Fail closed: _coefficients is the only caller and names only those five,
    # so another name means the two lists have drifted apart.
    raise OrbitWinError(f"{name} is not one of the announced scoring coefficients")


def _ordered(
    scenario: Scenario,
    candidates: Sequence[Candidate],
    score_config: ScoreConfig,
    config: OrbitWinConfig,
) -> list[tuple[Candidate, ScoreBreakdown]]:
    """Rank under these coefficients without writing anything to the candidates.

    The what-if sweep calls this many times with coefficients nobody announced.
    Storing those on the candidates would leave the archive scored by whichever
    trial happened to run last.
    """
    # Penalties are recomputed, not read. §5 deducts ten points per violating
    # maneuver and Candidate.penalties is the sole input to that term -- yet it
    # arrives from a file, written under whatever profile solve happened to use,
    # while rank runs under the announced one. Measured: an archive whose burn
    # is 2.0 km/s against a 1.5 km/s limit was crowned at 75.204 with
    # penalty_count 0; the correct 55.204 loses.
    scored = [
        (
            item,
            score_candidate(
                item,
                scenario,
                score_config,
                penalty_count=len(
                    validate_candidate(item, scenario, config).rule_violations
                ),
            ),
        )
        for item in candidates
    ]
    scored.sort(key=lambda pair: ranking_key(pair[0], scenario=scenario, score=pair[1].raw_score))
    return scored


def rank_under(
    scenario: Scenario,
    candidates: Sequence[Candidate],
    score_config: ScoreConfig,
    config: OrbitWinConfig | None = None,
) -> list[tuple[Candidate, ScoreBreakdown]]:
    """Rank under the announced coefficients and store the resulting scores."""
    config = config or OrbitWinConfig(score=score_config)
    scored = _ordered(scenario, candidates, score_config, config)
    for candidate, _ in scored:
        # The order used penalties recomputed under `config`; storing the score
        # from the file's penalties wrote 82.81 beside a rank made at 72.81.
        apply_constraints(candidate, scenario, config)
        # apply_score stays the only writer of score/score_components/basis, so
        # a candidate ranked here is indistinguishable from one the pipeline
        # scored, staleness detection included.
        apply_score(candidate, scenario, score_config)
    return scored


def _spread_scope(
    ranked: Iterable[tuple[Candidate, ScoreBreakdown]], misses_count: bool
) -> list[tuple[Candidate, ScoreBreakdown]]:
    pairs = list(ranked)
    return pairs if misses_count else [pair for pair in pairs if pair[0].successful] or pairs


def term_spread_points(
    ranked: Iterable[tuple[Candidate, ScoreBreakdown]],
    *,
    misses_count: bool = False,
) -> dict[str, float]:
    """How many points separate the best and worst candidate on each term.

    Measured over the successful candidates when there are any, because those
    are the ones the choice is really made between: a failure's 0 on distance
    puts it 50 points behind every success, and its completion time is the
    horizon, which would widen the time spread enough to hide a term that
    cannot separate two successes from each other.

    Under the ranking rules (`misses_count`) a miss can win, so all are measured.
    """
    breakdowns = [breakdown for _, breakdown in _spread_scope(ranked, misses_count)]
    if len(breakdowns) < 2:
        # max - min over one element is zero by arithmetic, and reporting that
        # as a flat term sends the operator to re-check coefficients they are
        # forbidden to guess. There is nothing to separate, so there is nothing
        # to say about separation.
        return {}
    return {
        "distance": max(b.distance_score for b in breakdowns)
        - min(b.distance_score for b in breakdowns),
        "time": max(b.time_score for b in breakdowns) - min(b.time_score for b in breakdowns),
        "delta_v": max(b.delta_v_score for b in breakdowns)
        - min(b.delta_v_score for b in breakdowns),
    }


#: Under this many points the distance term is about 0, whatever separates the candidates.
_DISTANCE_TAIL_POINTS = 0.01


def _distance_warning(
    spread: float, score_config: ScoreConfig, rules: InterceptionRules,
    distances_km: list[float], time_exempt: bool,
) -> str:
    """Why distance separates nobody: the plateau, the tail, or terms close together."""
    head = (f"The distance term spans {spread:.3f} points across the whole archive, so it "
            "cannot affect the ranking.")
    # Every miss completes at T_max, so then the time term is no help either.
    decide = "delta-v decides" if time_exempt else "time and delta-v decide"
    match rules.distance_term:
        case DistanceTerm.EXPONENTIAL:
            # The preliminary's term has no announced coefficient or unit.
            return (head + f" So {decide}. Distance has no announced coefficient or unit, so "
                    "this is about the archive and does not point at a unit mistake.")
        case DistanceTerm.LOGISTIC:
            cr = score_config.cr_km
            sphere = rules.intercept_radius_km
            plateau = distance_for_term_km(49.0, rules, cr)
            known = [max(d, sphere) for d in distances_km if math.isfinite(d)]
            if not known:
                return head + " No candidate has a measured closest approach."
            near, far = min(known), max(known)
            if far <= plateau:
                return (head + f" Every candidate passes within {plateau:.6g} km, where the "
                        f"term is at least 49 at C_r = {cr:g} km: it is flat there, so "
                        f"{decide}.")
            if all(distance_term(d, rules, cr) < _DISTANCE_TAIL_POINTS for d in known):
                return (head + f" Every candidate passes {near:.6g} km off or farther, where "
                        f"the term is under {_DISTANCE_TAIL_POINTS:g} point at C_r = {cr:g} km: "
                        f"it is about 0 for all of them and cannot separate them, so {decide}.")
            # Off the plateau and the tail the terms can still lie within half a point,
            # however far apart the candidates pass: R2's 3861 to 4718 km at C_r 1500.
            terms = [distance_term(d, rules, cr) for d in known]
            return (head + f" The candidates pass {near:.6g} to {far:.6g} km off, and at "
                    f"C_r = {cr:g} km their terms lie between {min(terms):.4g} and "
                    f"{max(terms):.4g} points, so the term cannot separate them, and "
                    f"{decide}.")
        case _:
            assert_never(rules.distance_term)


def _unit_warning(term: str, spread: float, score_config: ScoreConfig) -> str:
    """Explain a term that cannot separate anybody.

    A saturated logistic is what a unit mismatch looks like from the outside. A
    `Cv` announced in km/s and entered as m/s puts every candidate on the same
    tail of the curve, so the term returns the same constant for all of them and
    the ranking is silently made by the other term alone. That is worth up to 25
    points and produces no error anywhere.
    """
    if term == "delta_v":
        centre = (
            f"kv={score_config.kv}, Cv={score_config.cv} and "
            f"score.delta_v_unit={score_config.delta_v_unit!r}"
        )
    else:
        centre = (
            f"kt={score_config.kt}, Ct={score_config.ct} and "
            f"score.time_unit={score_config.time_unit!r}"
        )
    return (
        f"The {term} term spans {spread:.3f} points across the whole archive, so it cannot "
        f"affect the ranking. Either the announced centre lies far outside the range this "
        f"archive covers, or the unit is wrong: check {centre} against the announcement."
    )



def _ceilings(
    scenario: Scenario,
    config: OrbitWinConfig,
    best: tuple[Candidate, ScoreBreakdown] | None,
    note: str | None = None,
) -> dict[str, Any] | None:
    """The most any answer and any miss can score, and how close a miss with the best
    delta-v term must come to beat ``best``, the best submittable row; None without a
    total cap, since delta-v then has no end to its range."""
    cap = config.policy.total_dv_limit_km_s
    if config.score is None or cap is None:
        return None
    bound = ceilings(on_solver_clock(config.score, scenario), scenario.max_time_s, cap)
    shown: dict[str, Any] = {
        "any_answer": bound.any_answer,
        "any_miss": bound.any_miss,
        "best_submittable": None,
        "best_submittable_note": None,
        "best": None,
        "misses_that_beat_best": None,
        "miss_must_be_within_km": None,
    }
    if best is None:
        return shown
    within = bound.miss_must_be_within_km(best[1], interception_rules(scenario.rules))
    shown.update(
        best_submittable=best[0].candidate_id,
        best_submittable_note=note,
        best=best[1].raw_score,
        misses_that_beat_best=(
            "none" if within is None else "any" if math.isinf(within) else "closer"
        ),
        # JSON has no infinity: a number only when it is one.
        miss_must_be_within_km=within if within is not None and math.isfinite(within) else None,
    )
    return shown


def _levers(
    scenario: Scenario, config: OrbitWinConfig, best: tuple[Candidate, ScoreBreakdown] | None
) -> dict[str, object] | None:
    """`score.levers` at ``best``, the best submittable row, priced at the policy's safe
    limits; None without one, or without a total cap, as for the ceilings."""
    cap = config.policy.total_dv_limit_km_s
    if config.score is None or cap is None or best is None:
        return None
    policy = config.policy
    budget = policy.total_dv_safe_limit_km_s if policy.total_dv_safe_limit_km_s is not None else cap
    found = levers(best[0], best[1], scenario, config.score, cap,
                   impulse_km_s=policy.dv_safe_limit_km_s, budget_km_s=budget)
    return {"candidate_id": best[0].candidate_id, **found.to_dict()}


#: Where the operator searches again under the announced coefficients.
SEARCH_AGAIN = "search again under these coefficients (docs/RUNBOOK.md step 5)"


class SearchedScore(TypedDict):
    """The score blocks the archive was searched under, against the one it is ranked under."""

    ranked_under: dict[str, float | str | None]
    #: Each distinct block on record, None for an unscored search.
    searched_under: list[dict[str, float | str | None] | None]
    #: Candidates archived before solve recorded the block.
    unrecorded_count: int
    matches: bool
    warning: str | None
    #: For best_submittable, when the search did not aim at these coefficients.
    note: str | None


def _searched_score(candidates: Sequence[Candidate], score_config: ScoreConfig) -> SearchedScore:
    """Whether the search aimed at the coefficients rank now scores with.

    The search optimises the score it is given, so an archive searched under one
    block and ranked under another is a ranking of rows nobody looked for under
    the new one: R2 re-ranked from C_r 300 to 3000 chose the zero burn at 31.54,
    where a search under 3000 found 45.82.
    """
    blocks: dict[str, Any] = {}
    differences: dict[tuple[str, str], int] = {}
    unscored = unrecorded = 0
    for candidate in candidates:
        if "searched_score" not in candidate.metadata:
            unrecorded += 1
            continue
        recorded = candidate.metadata["searched_score"]
        key = normalised_stamp(recorded)
        blocks.setdefault(json.dumps(key, sort_keys=True, default=repr), key)
        difference = score_stamp_difference(recorded, score_config)
        if difference is None:
            continue
        if recorded is None:
            unscored += 1
        else:
            differences[difference] = differences.get(difference, 0) + 1
    current = score_stamp(score_config)
    parts: list[str] = []
    of = len(candidates)
    for (searched, ranked), count in sorted(differences.items()):
        parts.append(
            f"{count} of {of} candidates were searched under {searched}, ranked under {ranked}: "
            "the search optimised for the old coefficients"
        )
    if unscored:
        parts.append(
            f"{unscored} of {of} candidates were searched unscored, with no score block: the "
            "search ordered by distance and delta-v and never aimed at any score"
        )
    if unrecorded:
        parts.append(
            f"{unrecorded} of {of} candidates carry no record of the score block they were "
            "searched under, so nothing shows the search aimed at these coefficients"
        )
    warning = note = None
    if parts:
        warning = "; ".join(parts) + f". Re-ranking is not enough: {SEARCH_AGAIN}."
        note = (
            "this archive was not searched under the coefficients it is ranked under; "
            f"{SEARCH_AGAIN} before certifying its best row"
        )
    return {
        "ranked_under": current,
        "searched_under": list(blocks.values()),
        "unrecorded_count": unrecorded,
        "matches": not parts,
        "warning": warning,
        "note": note,
    }


def _cfg_for(score_config: ScoreConfig) -> OrbitWinConfig:
    """winner_stability takes a ScoreConfig, but scoring now needs the policy too."""
    return OrbitWinConfig(score=score_config)


def _assert_tolerance_moves_something(tolerance: float, score_config: ScoreConfig) -> None:
    """Refuse a tolerance that cannot change the answer, or changes it beyond recognition.

    Every one of these reports a stable winner for a comparison that never
    happened, or an alarm about a coefficient nobody could have announced:

    * 0 or negative: the nudged config is the announced one.
    * NaN: every comparison against it is false, so a bare `tolerance <= 0`
      check passes it. The sweep then flags all four coefficients at once and
      writes bare NaN into the report, which is not valid JSON.
    * Below about 2**-53: `value * (1.0 + tolerance) == value` bit for bit, so
      the sweep compares a config to itself and finds, correctly, that the
      winner did not change.
    * 1.0 or more: the downward nudge lands on exactly 0.0 and then goes
      negative, which inverts the logistic so a slower and more expensive
      mission scores higher. That is not a transcription error, it is a
      different scoring rule, and reporting the winner change as a knife edge
      is misleading.
    """
    if not math.isfinite(tolerance):
        raise OrbitWinError(f"Stability needs a finite tolerance, not {tolerance!r}")
    if tolerance <= 0.0:
        raise OrbitWinError("Stability needs a positive tolerance; 0 nudges nothing")
    if tolerance >= 1.0:
        raise OrbitWinError(
            f"A tolerance of {tolerance:g} drives a coefficient to zero and past it, "
            "inverting the logistic rather than nudging it; use less than 1.0"
        )
    for name in _coefficients(score_config):
        value = float(getattr(score_config, name))
        if value != 0.0 and value * (1.0 + tolerance) == value:
            raise OrbitWinError(
                f"A tolerance of {tolerance:g} does not move {name}={value:g} at double "
                "precision, so the sweep would compare the announced coefficients to "
                "themselves and report the winner stable without testing anything"
            )

def winner_stability(
    scenario: Scenario,
    candidates: Sequence[Candidate],
    score_config: ScoreConfig,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    config: OrbitWinConfig | None = None,
) -> WinnerStability:
    """Does nudging any one announced coefficient change who we submit?"""
    _assert_tolerance_moves_something(tolerance, score_config)
    # The policy decides the penalties, so it has to be the one rank_under used.
    config = config or _cfg_for(score_config)
    announced = _ordered(scenario, candidates, score_config, config)[0][0].candidate_id
    changes: list[dict[str, object]] = []
    untested: list[str] = []
    for name in _coefficients(score_config):
        value = float(getattr(score_config, name))
        if value == 0.0:
            # A relative nudge of zero is zero, so this coefficient would be
            # reported stable without ever having been moved.
            untested.append(name)
            continue
        for direction in (-1.0, 1.0):
            nudged = _with_coefficient(score_config, name, value * (1.0 + direction * tolerance))
            winner = _ordered(scenario, candidates, nudged, config)[0][0].candidate_id
            if winner != announced:
                changes.append(
                    {
                        "coefficient": name,
                        "from": value,
                        "to": value * (1.0 + direction * tolerance),
                        "winner": winner,
                    }
                )
    return {
        "tolerance": tolerance,
        "winner": announced,
        "stable": not changes,
        "changes": changes,
        "not_tested": untested,
    }



def submission_blockers(
    candidate: Candidate, scenario: Scenario, config: OrbitWinConfig
) -> tuple[str, ...]:
    """Why this candidate could not be submitted, in the words solve would use.

    `rank` names the candidate a human then hands to `certify`, which is the
    same decision `solve` gates at its emission step -- and this module applied
    none of those gates. Measured on a real 120-candidate archive: 80 were never
    event-refined, so their `successful` flag is the Lambert endpoint residual
    rather than a trajectory that was evaluated, and their `minimum_distance_km`
    sits around 1.8e-10 km; and one reached its interception by passing through
    the Earth. All of them were ranked, and a saturated pair of logistics scores
    such a candidate exactly 100.

    Reported per candidate rather than filtered out. The archive is what the
    operator is choosing from, and hiding a row is worse than labelling it.
    """
    blockers: list[str] = []
    refusal, _ = report_name_verdict(scenario, config)
    if refusal is not None:
        blockers.append(f"certify refuses the report names: {refusal}")
    failed = candidate.metadata.get("gmat_verification_error")
    if failed:
        # solve cleared `successful`, but under the ranking rules a miss still scores.
        blockers.append(f"GMAT verification failed: {failed}")
    elif not interception_required(scenario) and not gmat_verified_on_record(candidate):
        # certify scores whatever GMAT flies under these rules (R4: a hit flown 42 km off).
        blockers.append(
            "never verified by GMAT: its hit or closest approach is the analytical model's, "
            "and certify may fly it as a miss"
        )
    measured = candidate.metadata.get("event_refined") is True
    flown = candidate.metadata.get("gmat_event")
    if (
        candidate.metadata.get("scored_miss") is True
        and isinstance(flown, dict)
        and "interpolated_minimum_distance_km" not in flown
    ):
        # Archived before misses scored on their closest row: d_min may be nearer than the rows.
        blockers.append(
            "a miss verified on the Hermite minimum between report rows, not on its closest row; "
            "verify it again"
        )
    if not candidate.successful and interception_required(scenario):
        blockers.append("no interception")
    elif not candidate.successful and not measured:
        # The ranking rules score a miss on its closest approach, so it must be measured.
        blockers.append(
            "closest approach never measured: no event refinement, so the scored "
            "minimum distance is not from an evaluated trajectory"
        )
    elif not measured:
        # The one that matters most, because nothing else about the record says
        # so. score.measured_minimum_distance_km already refuses to read an
        # unrefined minimum distance for the §6 tie-break, for this reason.
        blockers.append(
            "interception never measured: no event refinement, so `successful` is a "
            "Lambert endpoint residual rather than an evaluated trajectory"
        )
    allowed = (
        {ROBUST_FRONTIER, FORMAL_FRONTIER}
        if config.policy.allow_formal_frontier_submission
        else {ROBUST_FRONTIER}
    )
    if frontier_of(candidate, config) not in allowed:
        blockers.append(
            "on the formal frontier, which policy.allow_formal_frontier_submission withholds"
        )
    if not config.policy.allow_body_intersecting_submission and flies_through_central_body(
        candidate, config
    ):
        blockers.append("reaches the interception through the central body")
    reason = gmat_contradicts_hit(candidate, config)
    if reason:
        blockers.append(reason)
    shallow = shallow_stop(candidate, config)
    if shallow and not config.policy.allow_shallow_stop_submission:
        blockers.append(
            f"{shallow}; certify refuses it without policy.allow_shallow_stop_submission"
        )
    if below_minimum_flown_altitude(candidate, config):
        flown_km = float(candidate.metadata["minimum_pursuer_radius_km"])
        blockers.append(
            f"flies down to {flown_km - config.policy.central_body_radius_km:.3f} km altitude, "
            f"under policy.minimum_flown_altitude_km = {config.policy.minimum_flown_altitude_km}"
        )
    verdict = validate_candidate(candidate, scenario, config)
    blockers.extend(verdict.violations)
    return tuple(blockers)

def _row(
    candidate: Candidate,
    breakdown: ScoreBreakdown,
    blockers: tuple[str, ...],
    claimed_penalties: int,
    scenario: Scenario,
) -> dict[str, object]:
    measured = measured_minimum_distance_km(candidate)
    tied_on = tie_break_distance_km(candidate, scenario)
    return {
        "candidate_id": candidate.candidate_id,
        "stage": candidate.stage.value,
        "stage_claimed_by_file": candidate.metadata.get("stage_claimed_by_file"),
        "successful": candidate.successful,
        "event_refined": candidate.metadata.get("event_refined") is True,
        "submittable": not blockers,
        "blockers": list(blockers),
        "intercept_time_s": candidate.intercept_time_s,
        "total_dv_km_s": candidate.total_dv_km_s,
        "distance_at_completion_km": candidate.distance_at_completion_km,
        # Both, deliberately. The raw field holds a Lambert endpoint residual
        # around 1e-10 km until event refinement replaces it, and the §6
        # tie-break reads the guarded one -- so a report showing only the raw
        # number shows the operator something the ranking did not use.
        "minimum_distance_km": candidate.minimum_distance_km,
        "measured_minimum_distance_km": None if measured == math.inf else measured,
        # What a tie reads: that, over the scored d_min's window.
        "tie_break_distance_km": None if tied_on == math.inf else tied_on,
        # Derived from the breakdown, not from candidate.penalties. The score
        # already recomputes this from validate_candidate; reading the file's
        # tuple here would put a different number in the column beside it.
        "penalty_count": round(breakdown.penalty_score / 10.0),
        "penalty_count_claimed_by_file": claimed_penalties,
        "score": breakdown.raw_score,
        "score_components": breakdown.to_dict(),
    }


def head_start_choice(
    scenario: Scenario,
    config: OrbitWinConfig,
    preferred: Sequence[Candidate],
    breakdowns: dict[int, ScoreBreakdown],
) -> dict[str, Any]:
    """Head-start burns or not for best_submittable: one wins only by
    policy.head_start_min_gain_points over the best submittable row without them,
    since how the validator reads a burn in the alignment's rows is not answered (Q5)."""
    gain = config.policy.head_start_min_gain_points
    with_hs = next((c for c in preferred if uses_head_start(c.burns)), None)
    without = next((c for c in preferred if not uses_head_start(c.burns)), None)

    def score(candidate: Candidate | None) -> float | None:
        return None if candidate is None else breakdowns[id(candidate)].raw_score

    record: dict[str, Any] = {
        "enabled": config.policy.head_start_burns,
        "lead_s": head_start_lead_s(scenario),
        "note": head_start_status(scenario),
        "min_gain_points": gain,
        "best_with": None if with_hs is None else with_hs.candidate_id,
        "best_with_score": score(with_hs),
        "best_without": None if without is None else without.candidate_id,
        "best_without_score": score(without),
        "gain_points": None,
        "chosen": "without",
    }
    if with_hs is not None and without is not None:
        record["gain_points"] = (breakdowns[id(with_hs)].raw_score
                                 - breakdowns[id(without)].raw_score)
    first = preferred[0] if preferred else None
    if first is None or not uses_head_start(first.burns):
        record["chosen"] = "without" if first is not None else None
    elif without is None:
        record["chosen"] = "head-start"
    else:
        record["chosen"] = "head-start" if record["gain_points"] >= gain else "without"
    return record


def head_start_window_warning(candidate: Candidate, scenario: Scenario) -> str | None:
    """Under from_origin, a head-start miss whose d_min is its own pass by SC_Alien
    waiting at its epoch, in the alignment's rows: a pass the burns can aim at, and
    whether the validator counts those rows is Q5. Said, so two people accept it."""
    if candidate.successful or not uses_head_start(candidate.burns):
        return None
    window = alignment_window_km(candidate, scenario.alignment_minimum_distance_km)
    flown = candidate.minimum_distance_km
    if window is None or flown is None or not window < flown:
        return None
    return (
        f"best_submittable {candidate.candidate_id} is a head-start miss priced on d_min "
        f"{window:.3f} km, its own pass by SC_Alien waiting at its epoch during the "
        f"alignment, not the {flown:.3f} km it comes after t = 0: closest_approach_window "
        "from_origin counts those rows, and how the validator reads them is not answered "
        "(Q5 of docs/ASYNC_START_AND_INSERTION.md)"
    )


def head_start_line(record: dict[str, Any]) -> str:
    """What rank prints of head_start_choice."""
    if not record.get("enabled"):
        return "head start: policy.head_start_burns is off"
    if not record.get("lead_s"):
        return f"head start: none here ({record.get('note')})"
    with_hs, without = record.get("best_with"), record.get("best_without")
    if with_hs is None:
        return (f"head start: {record['lead_s']:.3f} s open, no submittable row burns in it; "
                f"best_submittable {without} burns after the alignment")
    if record.get("gain_points") is None:
        return (f"head start: best_submittable {with_hs} burns in it; no submittable row "
                "without head-start burns")
    verb = "takes" if record["chosen"] == "head-start" else "does not take"
    return (f"head start: best_submittable {verb} it: {with_hs} {record['best_with_score']:.4f} "
            f"against {without} {record['best_without_score']:.4f} without, "
            f"{record['gain_points']:+.4f} points, needs {record['min_gain_points']:g} "
            "(policy.head_start_min_gain_points)")


def rank_report(
    scenario: Scenario,
    candidates: Sequence[Candidate],
    config: OrbitWinConfig,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    top: int = 10,
) -> dict[str, object]:
    """Rank a saved archive under announced coefficients and say what is fragile.

    Writes the announced score onto every candidate, the same way the pipeline
    does, so a caller holding them afterwards sees the official numbers rather
    than whatever the archive was written with. The what-if sweep does not.
    """
    score_config = config.score
    if score_config is None:
        raise OrbitWinError(
            "Ranking needs score.kt, score.ct, score.kv and score.cv with their units, and "
            "score.cr_km under the ranking rules. "
            "They are announced by the organizer and must not be guessed."
        )
    if not candidates:
        raise OrbitWinError("The candidate archive is empty, so there is nothing to rank")
    if top < 1:
        # 0 emits an empty ranking beside a populated winner, which reads as "it
        # ranked nothing"; negative slices from the end and silently drops the
        # worst row, which is how --top -1 looks like a request for everything.
        raise OrbitWinError(f"top must be at least 1; {top} would list nothing or drop rows")
    if not math.isfinite(scenario.max_time_s):
        # A failed run is scored on its separation at the horizon, so a horizon
        # that is not a number makes every failure's score meaningless rather
        # than merely unknown.
        raise OrbitWinError("The scenario horizon is not finite, so failures cannot be scored")

    stamped = {
        str(candidate.metadata.get("scenario_sha256"))
        for candidate in candidates
        if candidate.metadata.get("scenario_sha256")
    }
    unstamped = sum(1 for candidate in candidates if not candidate.metadata.get("scenario_sha256"))
    # One matching stamp used to admit every other one beside it.
    foreign = stamped - {scenario.source_sha256}
    if foreign:
        raise OrbitWinError(
            f"This archive holds candidates produced from a different script: they carry "
            f"{', '.join(sorted(foreign))} and {scenario.source_path} hashes to "
            f"{scenario.source_sha256}. The horizon comes from the script and is the "
            "completion time for every failed candidate, so ranking across the two is "
            "worth up to the whole 25-point time term."
        )
    # Before the horizon: a moved origin moves T_max on the solver clock too.
    for candidate in candidates:
        mismatch = clock_stamp_mismatch(candidate.metadata, scenario)
        if mismatch is not None:
            raise OrbitWinError(
                f"This archive was solved on another clock than it is being ranked on: "
                f"{mismatch} between the two configs"
            )

    horizons = {
        float(candidate.metadata["horizon_s"])
        for candidate in candidates
        if candidate.metadata.get("horizon_s") is not None
    }
    # Written as a match so that NaN, which compares false with everything, fails it.
    if not all(abs(horizon - scenario.max_time_s) <= 1.0e-6 for horizon in horizons):
        if config.policy.horizon_s is None:
            # 4 T_A follows the initial state: 45 µs apart through the probe and in-process.
            read = (
                "through the GMAT state probe"
                if scenario.state_probe is not None
                else "in-process, without a console"
            )
            reason = (
                f"Same script; the horizon is 4 T_A, which follows the initial state, read "
                f"here {read}. Solve and rank alike: both with --gmat-console or both without."
            )
        else:
            reason = (
                "Same script, so policy.horizon_s differs between the two configs; every "
                "failed candidate's time term moves with it."
            )
        raise OrbitWinError(
            f"This archive was solved under a horizon of "
            f"{', '.join(f'{h:.6f}' for h in sorted(horizons))} s and is being ranked under "
            f"{scenario.max_time_s:.6f} s. {reason}"
        )

    # Before rank_under writes to the candidates, not after.
    _assert_tolerance_moves_something(tolerance, score_config)
    searched = _searched_score(candidates, score_config)
    # Read before rank_under, which rewrites the penalties it ranked by.
    penalties_in_file = {id(candidate): len(candidate.penalties) for candidate in candidates}
    # And the scores solve chose by, which rank_under rewrites too.
    pre_gmat = {id(candidate): pre_gmat_score(candidate) for candidate in candidates}
    ranked = rank_under(scenario, candidates, score_config, config)
    misses_count = not interception_required(scenario)
    spreads = term_spread_points(ranked, misses_count=misses_count)
    blockers = {
        candidate.candidate_id: submission_blockers(candidate, scenario, config)
        for candidate, _ in ranked
    }
    # Over what could be submitted: a blocked row winning a nudge changes nothing.
    submittable = [c for c, _ in ranked if not blockers[c.candidate_id]]
    stability: WinnerStability = (
        winner_stability(scenario, submittable, score_config, tolerance=tolerance, config=config)
        if submittable
        else {"tolerance": tolerance, "winner": None, "stable": True, "changes": [],
              "not_tested": []}
    )
    # winner_stability leaves the announced scores in place because it ranks
    # through _ordered, but rank_under ran first and is what stored them.

    any_success = any(candidate.successful for candidate, _ in ranked)
    all_success = all(candidate.successful for candidate, _ in ranked)
    # A term is exempt only where 初賽規則 §4 makes it constant by construction,
    # and the two cases are mirror images rather than cumulative.
    #
    # When something succeeds, distance is 50 for every success, so a flat
    # distance term is the rules working. When nothing succeeds, Tteam is 4TA
    # for every candidate, so it is the *time* term that cannot discriminate --
    # and distance becomes the only thing that can, since §4 then scores
    # Δr(4TA). Exempting distance there was backwards: it suppressed the one
    # warning that mattered, about the term carrying 50 of the 100 points.
    # The ranking rules spread hits and misses together; a mix makes neither constant.
    if any_success and (all_success or not misses_count):
        exempt = {"distance"}
    elif not any_success:
        exempt = {"time"}
    else:
        exempt = set()
    rules = interception_rules(scenario.rules)
    in_scope = [scored_distance_km(c, scenario) for c, _ in _spread_scope(ranked, misses_count)]
    warnings: list[str] = [
        _distance_warning(spread, score_config, rules, in_scope, "time" in exempt)
        if term == "distance"
        else _unit_warning(term, spread, score_config)
        for term, spread in sorted(spreads.items())
        if term not in exempt and spread < FLAT_TERM_POINTS
    ]
    if searched["warning"] is not None:
        warnings.append(searched["warning"])
    warnings += team_number_warnings(scenario, config)
    if not stamped:
        warnings.append(
            "This archive carries no scenario fingerprint, so nothing checked that it "
            "was produced from this script. Re-run solve to stamp it, or confirm by hand."
        )
    elif unstamped:
        warnings.append(
            f"{unstamped} of the {len(candidates)} candidates carry no scenario fingerprint, "
            "so nothing checked that they came from this script; the others were checked."
        )
    if stability["not_tested"]:
        names = ", ".join(str(name) for name in stability["not_tested"])
        warnings.append(
            f"{names} is zero, so nudging it by a percentage does not move it and its "
            "effect on the winner was never tested."
        )
    if not stability["stable"]:
        moved = sorted({str(change["coefficient"]) for change in stability["changes"]})
        warnings.append(
            f"The winner changes when {', '.join(moved)} moves by "
            f"{tolerance:.0%}. Re-read the announced value before submitting."
        )
    deeper = [
        candidate.candidate_id for candidate, _ in ranked if stop_depth_mismatch(candidate, config)
    ]
    if deeper:
        shown = ", ".join(deeper[:5]) + (f" and {len(deeper) - 5} more" if len(deeper) > 5 else "")
        warnings.append(
            "GMAT-verified hits stopped at another depth than policy.stop_depth_km = "
            f"{config.policy.stop_depth_km:g} km: {shown}. Their completion times and scores "
            "are from that stop; certify measures them again."
        )
    winner, winner_breakdown = ranked[0]
    # As solve chooses: a hit that reached its stop depth before a shallower one.
    breakdowns = {id(c): b for c, b in ranked}
    preferred = prefer_deep_hits([c for c, _ in ranked if not blockers[c.candidate_id]], config)
    best_ok = (preferred[0], breakdowns[id(preferred[0])]) if preferred else None
    head_start = head_start_choice(scenario, config, preferred, breakdowns)
    if head_start["chosen"] == "without" and head_start["best_without"] is not None:
        chosen = next(c for c in preferred if c.candidate_id == head_start["best_without"])
        best_ok = (chosen, breakdowns[id(chosen)])
    if best_ok is not None and (passed := head_start_window_warning(best_ok[0], scenario)):
        warnings.append(passed)
    if blockers[winner.candidate_id]:
        reasons = "; ".join(blockers[winner.candidate_id])
        warnings.append(
            f"The top-scoring candidate {winner.candidate_id} could not be submitted as it "
            f"stands: {reasons}. Ranking does not filter, so read `submittable` per row."
        )
        if best_ok is None:
            warnings.append(
                "No candidate in this archive passes the submission gates, so nothing here "
                "is ready to hand to certify."
            )
        else:
            gap = winner_breakdown.raw_score - best_ok[1].raw_score
            warnings.append(
                f"The best submittable candidate is {best_ok[0].candidate_id} at "
                f"{best_ok[1].raw_score:.3f} points, {gap:.3f} below the top row."
            )
    elif best_ok is not None and best_ok[0] is not winner:
        warnings.append(
            f"The top-scoring candidate {winner.candidate_id} is a shallow hit: "
            f"{shallow_stop(winner, config)}. A verified hit in this archive reaches its stop "
            f"depth, so {best_ok[0].candidate_id} is the one to certify, "
            f"{winner_breakdown.raw_score - best_ok[1].raw_score:.3f} points below the top row."
        )
    # solve said so when it ran; the archive keeps only the verifications that raised.
    failures = gmat_failures(
        [("verification", c, error) for c, _ in ranked
         if (error := verification_error(c)) is not None],
        best_ok[0] if best_ok is not None else None,
        archived={c.candidate_id for c, _ in ranked}, who="the best submittable row",
        only_above=True, score_of=lambda candidate: pre_gmat[id(candidate)],
    )
    if failures["warning"] is not None:
        warnings.append(failures["warning"])
    if misses_count and not any(
        candidate.gmat_run_id is not None
        or any(key in candidate.metadata for key in _GMAT_TRACES)
        for candidate, _ in ranked
    ):
        solved = any("renderable" in candidate.metadata for candidate, _ in ranked)
        warnings.insert(0, NO_GMAT_ARCHIVE if solved else NO_GMAT_HUNT)
    # A file cannot confer a GMAT tier -- io.candidate_from_dict caps every
    # loaded stage at renderable -- so this fires for every archive read from
    # disk, which is every archive rank sees. Say what it means rather than
    # dressing a constant up as a finding.
    if winner.stage not in (CandidateStage.ORGANIZER_GMAT_VERIFIED, CandidateStage.CERTIFIED):
        claimed = winner.metadata.get("stage_claimed_by_file")
        warnings.append(
            f"The top-scoring candidate stands at stage {winner.stage.value}"
            + (f" (the file claimed {claimed})" if claimed else "")
            + ". A score is arithmetic; only certify establishes that GMAT reproduces it."
        )

    return {
        # What T_team and t = 0 mean here (docs/ASYNC_START_AND_INSERTION.md §3).
        "clock": {
            **clock_record(scenario, config.policy),
            "state_probe_report_sha256": (
                dict(scenario.state_probe.run.report_sha256)
                if scenario.state_probe is not None
                else None
            ),
            # Where t = 0's states came from, when the craft start apart.
            "alignment_probe_report_sha256": (
                dict(scenario.state_probe.aligned.run.report_sha256)
                if scenario.state_probe is not None and scenario.state_probe.aligned is not None
                else None
            ),
        },
        "coefficients": {
            **{name: getattr(score_config, name) for name in _coefficients(score_config)},
            "time_unit": score_config.time_unit,
            "delta_v_unit": score_config.delta_v_unit,
        },
        "ceilings": _ceilings(scenario, config, best_ok, searched["note"]),
        # Whether best_submittable burns in the head start, and the margin it took.
        "head_start": head_start,
        "searched_score": searched,
        "gmat_failures": failures,
        # Set whether or not a row is submittable: it is about the archive.
        "best_submittable_note": searched["note"],
        "levers": _levers(scenario, config, best_ok),
        "candidate_count": len(ranked),
        "successful_count": sum(1 for candidate, _ in ranked if candidate.successful),
        "winner": _row(
            winner, winner_breakdown, blockers[winner.candidate_id], penalties_in_file[id(winner)],
            scenario,
        ),
        "ranking": [
            _row(
                candidate,
                breakdown,
                blockers[candidate.candidate_id],
                penalties_in_file[id(candidate)],
                scenario,
            )
            for candidate, breakdown in ranked[:top]
        ],
        "listed": min(top, len(ranked)),
        "submittable_count": sum(1 for value in blockers.values() if not value),
        "gmat_verified_count": sum(1 for c, _ in ranked if gmat_verified_on_record(c)),
        "term_spread_points": spreads,
        "term_spread_scope": "successful" if any_success and not misses_count else "all",
        "winner_stability": stability,
        "warnings": warnings,
    }
