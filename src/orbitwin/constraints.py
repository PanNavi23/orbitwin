from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from .config import OrbitWinConfig
from .models import Candidate, Scenario, head_start_lead_s
from .report_parser import STOP_DEPTH_TOLERANCE, stop_depth_used_km
from .rules import interception_rules

#: A candidate every one of whose burns sits inside the production margins.
ROBUST_FRONTIER = "robust"
#: Legal under the competition rules, but outside at least one production
#: margin. COMPETITION_POLICY.md is explicit that 1.495 km/s and 101 s are ours
#: and not the organizer's, so a candidate between the margin and the rule is a
#: legal solution that this project would rather not submit -- not an invalid
#: one, and not one to discard unseen.
FORMAL_FRONTIER = "formal"

#: Slack on a delta-v comparison, for representation error only. The formal
#: total limit gets none, since the rule gives none (§4 note 1(c)).
DV_COMPARISON_TOLERANCE_KM_S = 1.0e-12


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    valid: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]
    #: One entry per *maneuver* that breaks a rule, each listing that
    #: maneuver's reasons. The official formula deducts ten points 每次違規機動
    #: -- per violating maneuver -- so counting reasons would charge a burn that
    #: breaks two constraints twice for one mistake.
    #: The subset of `violations` that breaks a competition rule. The official
    #: score deducts ten points per violation, so only rules belong here. The
    #: require_success condition is our own gate -- a candidate GMAT rejected
    #: has not broken a rule, and reporting a ten point deduction for it
    #: describes a penalty the organizer would never apply.
    rule_violations: tuple[str, ...] = ()


def _finite_number(value: object) -> bool:
    # numbers.Real so that numpy's float32 and int64 count; bool is excluded.
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:  # an int too large for a float
        return False


def interception_required(scenario: Scenario) -> bool:
    """True under the preliminary rules, where we submit only an interception.

    Both rulebooks score a miss. The preliminary scores it at 4·T_A and leaves
    open whether a failed mission is ranked on that score
    (docs/ASSUMPTIONS_AND_GAPS.md); the ranking rules score it on its closest
    approach (eq. 2, 3), where a near miss can outscore a late hit.
    """
    return not interception_rules(scenario.rules).miss_ranked


def validate_candidate(
    candidate: Candidate,
    scenario: Scenario,
    config: OrbitWinConfig,
    *,
    require_success: bool = False,
) -> ConstraintResult:
    violations: list[str] = []
    warnings: list[str] = []
    # Every comparison against NaN is False, so a NaN impulse or epoch passed
    # every check below; and NaN epochs cannot be put in order at all.
    finite_dv: set[int] = set()
    for index, burn in enumerate(candidate.burns):
        if all(math.isfinite(value) for value in (*burn.dv_inertial_km_s, *burn.dv_script_km_s)):
            finite_dv.add(index)
            # The limits and score read the inertial vector; GMAT flies the script one.
            inertial, script = burn.magnitude_km_s, math.hypot(*burn.dv_script_km_s)
            if abs(inertial - script) > 1.0e-9:
                violations.append(
                    f"burn[{index}] is {inertial!r} km/s in the inertial frame and {script!r} "
                    "km/s in the script frame; one impulse cannot be both"
                )
        if index not in finite_dv or not math.isfinite(burn.time_s):
            violations.append(f"burn[{index}] carries a non-finite number")
    measured = {
        "intercept_time_s": candidate.intercept_time_s,
        "minimum_distance_km": candidate.minimum_distance_km,
        "distance_at_completion_km": candidate.distance_at_completion_km,
        "minimum_pursuer_radius_km": candidate.metadata.get("minimum_pursuer_radius_km"),
    }
    for name, value in measured.items():
        if value is not None and not _finite_number(value):
            violations.append(f"{name} is not a finite number: {value!r}")
    # A broken number does not stop the rest being checked: returning early
    # charged no penalty for a 2 km/s burn beside a NaN one, and rank put it first.
    # Reasons grouped by the maneuver they belong to, keyed by its index in candidate.burns.
    by_burn: dict[int, list[str]] = {}
    # Every burn with a finite time takes part in the separation check.
    timed = sorted(
        (index for index, burn in enumerate(candidate.burns) if math.isfinite(burn.time_s)),
        key=lambda index: candidate.burns[index].time_s,
    )
    for position, index in enumerate(timed):
        burn = candidate.burns[index]
        if index in finite_dv:
            magnitude = burn.magnitude_km_s
            if magnitude > config.policy.dv_limit_km_s + DV_COMPARISON_TOLERANCE_KM_S:
                reason = (
                    f"magnitude {magnitude:.12g} km/s exceeds formal limit "
                    f"{config.policy.dv_limit_km_s:.12g} km/s"
                )
                violations.append(f"burn[{index}] {reason}")
                # The ranking rules disqualify it (§4 note 1(a)) rather than deduct.
                if interception_rules(scenario.rules).over_impulse_limit == "penalty":
                    by_burn.setdefault(index, []).append(reason)
            elif magnitude > config.policy.dv_safe_limit_km_s + DV_COMPARISON_TOLERANCE_KM_S:
                warnings.append(
                    f"burn[{index}] magnitude {magnitude:.12g} km/s exceeds production safety limit"
                )
        # Mission start is the earlier epoch: -lead when the pursuer may burn in its head start.
        if burn.time_s < -head_start_lead_s(scenario) - 1.0e-9:
            violations.append(f"burn[{index}] occurs before mission start")
            by_burn.setdefault(index, []).append("occurs before mission start")
        if burn.time_s > scenario.max_time_s + 1.0e-9:
            violations.append(f"burn[{index}] occurs after Tmax")
            by_burn.setdefault(index, []).append("occurs after Tmax")
        if position:
            previous = timed[position - 1]
            separation = burn.time_s - candidate.burns[previous].time_s
            # The rule charges Δt < 100 s. 1e-9 s is float error only: a gap of
            # 100 s between epochs near 1.6e4 s subtracts to 100 - 1.8e-12.
            if separation < config.policy.min_burn_separation_s - 1.0e-9:
                reason = (
                    f"separation {separation:.12g} s from burn[{previous}] is below "
                    f"{config.policy.min_burn_separation_s:.12g} s"
                )
                violations.append(f"burn[{previous}] to burn[{index}] {reason}")
                # Charged to the later maneuver: it is the one placed too close.
                by_burn.setdefault(index, []).append(reason)
            elif separation < config.policy.safe_burn_separation_s - 1.0e-9:
                warnings.append(
                    f"burn[{previous}] to burn[{index}] separation is below safety margin"
                )
    total_limit = config.policy.total_dv_limit_km_s
    total_safe = config.policy.total_dv_safe_limit_km_s
    if total_limit is not None and total_safe is not None:
        total = candidate.total_dv_km_s
        # Disqualifying, so not in rule_violations, which the score charges ten
        # points each; and compared with no slack, since the rule gives none.
        if total > total_limit:
            violations.append(
                f"total delta-v {total:.12g} km/s exceeds the {total_limit:.12g} km/s limit"
            )
        elif total > total_safe + DV_COMPARISON_TOLERANCE_KM_S:
            warnings.append(
                f"total delta-v {total:.12g} km/s exceeds production safety limit"
            )
    if candidate.intercept_time_s is not None and candidate.intercept_time_s > scenario.max_time_s + 1.0e-9:
        violations.append("intercept occurs after Tmax")
    rule_violations = tuple(
        f"burn[{index}]: " + "; ".join(reasons) for index, reasons in sorted(by_burn.items())
    )
    if require_success and not candidate.successful:
        violations.append("candidate is not a verified successful interception")
    return ConstraintResult(
        not violations, tuple(violations), tuple(warnings), rule_violations
    )


def frontier_of(candidate: Candidate, config: OrbitWinConfig) -> str:
    """Which frontier a candidate belongs to.

    Membership is decided by the production margins, not by the rules: a
    candidate is robust when every impulse is within dv_safe_limit_km_s and
    every gap is at least safe_burn_separation_s. Anything legal but outside
    those is on the formal frontier. So is a total over total_dv_safe_limit_km_s.
    """
    policy = config.policy
    if (
        policy.total_dv_safe_limit_km_s is not None
        and candidate.total_dv_km_s > policy.total_dv_safe_limit_km_s + DV_COMPARISON_TOLERANCE_KM_S
    ):
        return FORMAL_FRONTIER
    burns = sorted(candidate.burns, key=lambda burn: burn.time_s)
    for index, burn in enumerate(burns):
        if burn.magnitude_km_s > policy.dv_safe_limit_km_s + DV_COMPARISON_TOLERANCE_KM_S:
            return FORMAL_FRONTIER
        if index and burn.time_s - burns[index - 1].time_s < (
            policy.safe_burn_separation_s - 1.0e-9
        ):
            return FORMAL_FRONTIER
    return ROBUST_FRONTIER


def flies_through_central_body(candidate: Candidate, config: OrbitWinConfig) -> bool:
    """True when the pursuer passes inside the central body before completing.

    Point-mass gravity has no surface, so the search can buy an interception
    with a trajectory that goes underground: a large retrograde impulse from a
    7000 km circular orbit puts pericenter near 3300 km. 初賽規則 says nothing
    about it and GMAT will not object, which is why it is measured rather than
    assumed away.

    Unknown counts as clear. A candidate whose event was never refined carries
    no measurement, and inventing one would either invent a violation or hide a
    real one; the caller can see `event_refined` for itself.
    """
    radius_km = candidate.metadata.get("minimum_pursuer_radius_km")
    if radius_km is None:
        return False
    return float(radius_km) < config.policy.central_body_radius_km


def gmat_contradicts_hit(candidate: Candidate, config: OrbitWinConfig) -> str | None:
    """Why GMAT's correction flight says this analytical hit is not one, or None."""
    flown_km = candidate.metadata.get("gmat_first_flight_closest_km")
    if not candidate.successful or flown_km is None:
        return None
    if float(flown_km) <= config.policy.intercept_radius_km:
        return None
    # Verification flew it to T_max; an entry it found stands over the first flight.
    if (candidate.metadata.get("gmat_event") or {}).get("first_entry_time_s") is not None:
        return None
    reason = (
        f"GMAT flew it {float(flown_km):.3f} km from the target at closest, through its "
        "planned arrival, outside the sphere: not the hit it is scored as"
    )
    superseded_by = candidate.metadata.get("superseded_by")
    if superseded_by:
        return f"{reason}; certify its correction {superseded_by} instead"
    return f"{reason}; to score it as the miss it is, solve again with a larger --gmat-verify-top"


def gmat_verified_on_record(candidate: Candidate) -> bool:
    """Whether a finished GMAT verification of this candidate is on record.

    A file caps the stage at renderable, so this reads what verify writes last:
    the run and the script it flew, with no verification error after them.
    """
    event = candidate.metadata.get("gmat_event")
    return (
        bool(candidate.gmat_run_id)
        and isinstance(event, dict)
        and bool(event.get("flown_script_sha256"))
        and not candidate.metadata.get("gmat_verification_error")
    )


def stop_depth_mismatch(candidate: Candidate, config: OrbitWinConfig) -> str | None:
    """Why a GMAT-verified hit's archived completion is from another stop depth, or None.

    Its completion time and score were measured there; certify measures again.
    """
    event = candidate.metadata.get("gmat_event") or {}
    if not candidate.successful or event.get("stop_time_s") is None:
        return None
    setting = event.get("stop_depth_setting_km")
    configured = config.policy.stop_depth_km
    if setting is not None and abs(float(setting) - configured) <= 1.0e-12:
        return None
    measured = event.get("stop_depth_km")
    where = f", {float(measured) * 1000.0:.4g} m inside" if measured is not None else ""
    under = (
        f"under policy.stop_depth_km = {float(setting):g} km"
        if setting is not None
        else "before the stop depth was recorded"
    )
    return (
        f"Candidate {candidate.candidate_id} was verified {under}{where}, and "
        f"policy.stop_depth_km is {configured:g} km: its completion time and score are "
        "from that stop"
    )


def stop_depth_reached(candidate: Candidate, config: OrbitWinConfig) -> bool | None:
    """Whether a GMAT-measured hit's passage gets as deep as policy.stop_depth_km asks now.

    A stop short of the depth it was flown under is the passage's deepest point,
    so it answers for any setting. A stop at an older, shallower setting answers
    only when it is already deep enough. None when unknown: a miss, an unverified
    hit, one flown to T_max, or one stopped shallower than today asks; certify
    re-measures those.
    """
    event = candidate.metadata.get("gmat_event") or {}
    measured = event.get("stop_depth_km")
    if not candidate.successful or measured is None:
        return None
    tolerance = 1.0 - STOP_DEPTH_TOLERANCE
    if float(measured) >= _asked_depth_km(config.policy.stop_depth_km, config) * tolerance:
        return True
    setting = event.get("stop_depth_setting_km")
    # Unrecorded settings were 50 m stops, which say nothing about the passage.
    if setting is not None and float(measured) < _asked_depth_km(float(setting), config) * tolerance:
        return False
    return None


def _asked_depth_km(setting_km: float, config: OrbitWinConfig) -> float:
    return stop_depth_used_km(setting_km, config.policy.intercept_radius_km)


def shallow_stop(candidate: Candidate, config: OrbitWinConfig) -> str | None:
    """Why a verified hit may be a miss on the organizer's machine, or None.

    Data files alone moved R4's arrival 665 m (CSSI) and EarthFixed states 6-25 m (EOP).
    """
    if stop_depth_reached(candidate, config) is not False:
        return None
    event = candidate.metadata["gmat_event"]
    asked_km = _asked_depth_km(config.policy.stop_depth_km, config)
    return (
        f"Candidate {candidate.candidate_id}'s submitted script ends only "
        f"{float(event['stop_depth_km']) * 1000.0:.4g} m inside the "
        f"{config.policy.intercept_radius_km:g} km sphere, short of the "
        f"{asked_km * 1000.0:.4g} m policy.stop_depth_km asks for: a GMAT reading other data "
        "files (space weather, EOP) can fly it as a miss"
    )


def prefer_deep_hits(ordered: list[Candidate], config: OrbitWinConfig) -> list[Candidate]:
    """``ordered`` without its shallow hits when any hit in it reached its stop depth.

    A shallow hit is never chosen beside a deep one; between what remains, the
    order decides, so a miss ranked above the deep hit still comes first.
    """
    if not any(stop_depth_reached(candidate, config) for candidate in ordered):
        return ordered
    return [c for c in ordered if stop_depth_reached(c, config) is not False]


def below_minimum_flown_altitude(candidate: Candidate, config: OrbitWinConfig) -> bool:
    """True when policy.minimum_flown_altitude_km is set and the measured
    radius is under it. Off by default; unknown counts as clear, as above."""
    floor_km = config.policy.minimum_flown_altitude_km
    radius_km = candidate.metadata.get("minimum_pursuer_radius_km")
    if floor_km is None or radius_km is None:
        return False
    return float(radius_km) < config.policy.central_body_radius_km + floor_km


def apply_constraints(
    candidate: Candidate,
    scenario: Scenario,
    config: OrbitWinConfig,
    *,
    require_success: bool = False,
) -> ConstraintResult:
    """Validate and record the outcome on the candidate.

    ``Candidate.penalties`` is read by the official score as
    ``10.0 * len(candidate.penalties)`` but had no writer anywhere, so the
    penalty term was unreachable and always zero. Recording violations here
    gives it one.

    A formal candidate is still eliminated rather than penalised: our policy is
    stricter than the organizer's scoring, and a violating script must never be
    submitted at ten points off. The recorded penalties describe what the
    organizer's formula would deduct if it ever saw one, which is a diagnostic,
    not a route to submission.
    """
    result = validate_candidate(candidate, scenario, config, require_success=require_success)
    candidate.metadata["frontier"] = frontier_of(candidate, config)
    candidate.penalties = result.rule_violations
    candidate.warnings = tuple(sorted(set(candidate.warnings + result.warnings)))
    return result
