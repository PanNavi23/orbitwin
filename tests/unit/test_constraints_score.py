from __future__ import annotations

import math
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, ScoreConfig
from orbitwin.constraints import (
    FORMAL_FRONTIER,
    ROBUST_FRONTIER,
    apply_constraints,
    frontier_of,
    validate_candidate,
)
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Candidate
from orbitwin.score import measured_minimum_distance_km, ranking_key, score_candidate


def candidate_with_burns(*burns: Burn) -> Candidate:
    return Candidate(
        burns=tuple(burns),
        intercept_time_s=1000.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="test",
    )


def test_formal_dv_violation(workshop_script: Path) -> None:
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    burn = Burn(0.0, (1.500001, 0, 0), (1.500001, 0, 0))
    result = validate_candidate(candidate_with_burns(burn), scenario, config)
    assert not result.valid
    assert "formal limit" in result.violations[0]


def test_safe_margin_warning_without_formal_violation(workshop_script: Path) -> None:
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    burn = Burn(0.0, (1.498, 0, 0), (1.498, 0, 0))
    result = validate_candidate(candidate_with_burns(burn), scenario, config)
    assert result.valid
    assert result.warnings


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("where", ["time", "inertial", "script"])
def test_a_non_finite_burn_is_a_violation(workshop_script: Path, where: str, value: float) -> None:
    """Every comparison against NaN is False, so a NaN impulse passed the cap."""
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    vector = (value, 0.0, 0.0)
    burn = Burn(
        value if where == "time" else 100.0,
        vector if where == "inertial" else (0.1, 0.0, 0.0),
        vector if where == "script" else (0.1, 0.0, 0.0),
    )
    result = validate_candidate(candidate_with_burns(burn), scenario, config)
    assert not result.valid
    assert result.violations == ("burn[0] carries a non-finite number",)
    assert result.rule_violations == (), "our data, not a maneuver the organizer would charge"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (name, value)
        for name in ("intercept_time_s", "minimum_distance_km", "distance_at_completion_km")
        for value in (math.nan, math.inf)
    ]
    # Metadata arrives untyped from a file, so a string or a bool can sit there.
    + [("minimum_pursuer_radius_km", value) for value in (math.nan, math.inf, "7000", True)],
)
def test_a_non_finite_measurement_is_a_violation(
    workshop_script: Path, name: str, value: object
) -> None:
    """A NaN radius read as clear of the Earth, and a NaN distance scores NaN."""
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = candidate_with_burns(Burn(100.0, (0.1, 0, 0), (0.1, 0, 0)))
    if name == "minimum_pursuer_radius_km":
        candidate.metadata[name] = value
    else:
        setattr(candidate, name, value)
    result = validate_candidate(candidate, scenario, config)
    assert not result.valid
    assert f"{name} is not a finite number: {value!r}" in result.violations


def test_a_broken_measurement_does_not_hide_a_broken_rule(workshop_script: Path) -> None:
    """rank charged no penalty for a 99 s gap beside a NaN radius, and ranked it top."""
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = candidate_with_burns(
        Burn(0.0, (0.1, 0, 0), (0.1, 0, 0), 0), Burn(99.0, (0.1, 0, 0), (0.1, 0, 0), 1)
    )
    candidate.metadata["minimum_pursuer_radius_km"] = math.nan
    result = validate_candidate(candidate, scenario, config)
    assert not result.valid
    assert len(result.rule_violations) == 1
    assert any("minimum_pursuer_radius_km" in item for item in result.violations)


@pytest.mark.parametrize("where", ["time", "inertial", "script"])
def test_a_non_finite_burn_does_not_hide_the_others_rule_breaks(
    workshop_script: Path, where: str
) -> None:
    """rank charged a 2 km/s burn and a 10 s gap nothing beside a NaN burn: 98.529, not 88.529."""
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    broken = Burn(
        math.nan if where == "time" else 1000.0,
        (math.nan, 0, 0) if where == "inertial" else (0.1, 0, 0),
        (math.nan, 0, 0) if where == "script" else (0.1, 0, 0),
        0,
    )
    candidate = candidate_with_burns(
        broken,
        Burn(100.0, (2.0, 0, 0), (2.0, 0, 0), 1),
        Burn(500.0, (0.1, 0, 0), (0.1, 0, 0), 2),
        Burn(510.0, (0.1, 0, 0), (0.1, 0, 0), 3),
    )
    result = apply_constraints(candidate, scenario, config)
    assert not result.valid
    assert "burn[0] carries a non-finite number" in result.violations
    assert len(result.rule_violations) == 2, result.rule_violations
    # Labelled by index in candidate.burns, as the non-finite message is.
    assert result.rule_violations[0].startswith("burn[1]: magnitude")
    assert result.rule_violations[1].startswith("burn[3]: separation 10 s from burn[2]")
    assert len(candidate.penalties) == 2


@pytest.mark.parametrize("where", ["inertial", "script"])
def test_a_burn_with_a_non_finite_impulse_still_has_a_time(
    workshop_script: Path, where: str
) -> None:
    """origin/main charged the 50 s gap after a NaN impulse; dropping the burn charged nothing."""
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = candidate_with_burns(
        Burn(
            0.0,
            (math.nan, 0, 0) if where == "inertial" else (0.1, 0, 0),
            (math.nan, 0, 0) if where == "script" else (0.1, 0, 0),
            0,
        ),
        Burn(50.0, (0.1, 0, 0), (0.1, 0, 0), 1),
    )
    result = apply_constraints(candidate, scenario, config)
    assert not result.valid
    assert result.rule_violations == ("burn[1]: separation 50 s from burn[0] is below 100 s",)


def test_a_burn_whose_two_frames_differ_in_length_is_a_violation(workshop_script: Path) -> None:
    """GMAT flies 1.49 km/s while the cap, frontier and score read 0.3."""
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = candidate_with_burns(Burn(100.0, (0.3, 0, 0), (1.49, 0, 0)))
    result = apply_constraints(candidate, scenario, config)
    assert not result.valid
    assert any("inertial frame" in item and "script frame" in item for item in result.violations)
    assert result.rule_violations == (), "our data, not a maneuver the organizer would charge"


def test_a_burn_rotated_between_frames_is_consistent(workshop_script: Path) -> None:
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    component = 0.3 / math.sqrt(3.0)
    burn = Burn(100.0, (0.3, 0, 0), (component, -component, component))
    assert validate_candidate(candidate_with_burns(burn), scenario, config).valid


def test_an_integer_too_large_for_a_float_is_not_finite(workshop_script: Path) -> None:
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = candidate_with_burns(Burn(100.0, (0.1, 0, 0), (0.1, 0, 0)))
    candidate.metadata["minimum_pursuer_radius_km"] = 10**400
    assert not validate_candidate(candidate, scenario, config).valid


def test_a_numpy_measurement_is_a_number(workshop_script: Path) -> None:
    import numpy as np

    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = candidate_with_burns(Burn(100.0, (0.1, 0, 0), (0.1, 0, 0)))
    candidate.intercept_time_s = np.int64(1000)  # type: ignore[assignment]
    candidate.metadata["minimum_pursuer_radius_km"] = np.float32(7000.0)
    assert validate_candidate(candidate, scenario, config).valid


def test_burn_separation_violation(workshop_script: Path) -> None:
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    burns = (
        Burn(0.0, (0.1, 0, 0), (0.1, 0, 0), 0),
        Burn(99.999, (0.1, 0, 0), (0.1, 0, 0), 1),
    )
    result = validate_candidate(candidate_with_burns(*burns), scenario, config)
    assert not result.valid
    assert any("below 100" in item for item in result.violations)


def test_success_distance_component_is_fifty(workshop_script: Path) -> None:
    scenario, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    candidate = candidate_with_burns(Burn(0, (0.5, 0, 0), (0.5, 0, 0)))
    score = score_candidate(candidate, scenario, ScoreConfig(0.001, 1000, 0.004, 500))
    assert score.distance_score == pytest.approx(50.0)


def test_logistic_midpoints_are_12_point_5(workshop_script: Path) -> None:
    scenario, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    # total_dv_km_s is derived from the burns, so the delta-v midpoint has to
    # be expressed as an impulse a script could actually carry.
    candidate = Candidate(
        # cv is 800 m/s, so the delta-v midpoint is an 0.8 km/s impulse.
        burns=(Burn(0.0, (0.8, 0.0, 0.0), (0.8, 0.0, 0.0), 0),),
        intercept_time_s=2000.0,
        minimum_distance_km=5.0,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.0,
        branch="test",
    )
    score = score_candidate(
        candidate,
        scenario,
        ScoreConfig(kt=0.1, ct=2000.0, kv=0.1, cv=800.0, delta_v_unit="m/s"),
    )
    assert score.time_score == pytest.approx(12.5)
    assert score.delta_v_score == pytest.approx(12.5)


def _ranked(score: float, minimum_km: float, dv: float, intercept: float, tag: str) -> Candidate:
    # The delta-v has to come from a real impulse now, since total_dv_km_s is
    # derived. A single burn along x of the requested magnitude does that.
    candidate = Candidate(
        burns=(Burn(1000.0, (dv, 0.0, 0.0), (dv, 0.0, 0.0), 0),),
        intercept_time_s=intercept,
        minimum_distance_km=minimum_km,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=None,
        branch=tag,
        metadata={"arrival_time_s": intercept, "event_refined": True},
    )
    candidate.score = score
    return candidate


def _unwindowed(candidate: Candidate) -> tuple[float, float, float, float, str]:
    return ranking_key(candidate, scenario=None)


def test_equal_scores_break_on_minimum_distance_then_dv_then_time() -> None:
    # COMPETITION_POLICY.md: "Tie order is actual minimum distance, total
    # delta-v, completion time, then design explanation."
    far = _ranked(80.0, 4.0, 0.10, 1000.0, "far")
    near = _ranked(80.0, 0.5, 0.90, 9000.0, "near")
    assert sorted([far, near], key=_unwindowed)[0] is near

    cheap = _ranked(80.0, 1.0, 0.10, 9000.0, "cheap")
    costly = _ranked(80.0, 1.0, 0.90, 1000.0, "costly")
    assert sorted([costly, cheap], key=_unwindowed)[0] is cheap

    early = _ranked(80.0, 1.0, 0.10, 1000.0, "early")
    late = _ranked(80.0, 1.0, 0.10, 9000.0, "late")
    assert sorted([late, early], key=_unwindowed)[0] is early


def test_unrefined_lambert_residual_cannot_win_the_distance_tie_break() -> None:
    # An unrefined candidate carries the Lambert endpoint residual (~1e-10 km)
    # in minimum_distance_km. Taken at face value it beats every measured
    # closest approach.
    refined = _ranked(80.0, 0.5, 0.10, 1000.0, "refined")
    unrefined = _ranked(80.0, 3.2e-10, 0.10, 1000.0, "unrefined")
    unrefined.metadata["event_refined"] = False

    assert measured_minimum_distance_km(unrefined) == float("inf")
    assert sorted([unrefined, refined], key=_unwindowed)[0] is refined


def test_gmat_verification_outranks_a_higher_analytical_score() -> None:
    from functools import partial

    from orbitwin.pipeline import _sort_key

    unverified = _ranked(95.0, 0.1, 0.10, 1000.0, "unverified-high-score")
    verified = _ranked(60.0, 4.9, 0.90, 9000.0, "verified-low-score")
    from orbitwin.models import CandidateStage

    verified.promote(CandidateStage.PUBLIC_GMAT_VERIFIED, run_id="R")
    order = partial(_sort_key, scenario=None, scored=True)
    assert sorted([unverified, verified], key=order)[0] is verified


def test_apply_constraints_records_penalties_on_the_candidate(workshop_script: Path) -> None:
    # score_candidate reads 10.0 * len(candidate.penalties). Until something
    # writes that field the penalty term is unreachable and always zero.
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = candidate_with_burns(
        Burn(0.0, (9.0, 0.0, 0.0), (9.0, 0.0, 0.0), 0),
        Burn(10.0, (9.0, 0.0, 0.0), (9.0, 0.0, 0.0), 1),
    )
    assert candidate.penalties == ()

    result = apply_constraints(candidate, scenario, config)
    assert not result.valid
    # Three reasons across two maneuvers: both are over the magnitude limit and
    # the second is too close to the first.
    assert len(result.violations) == 3
    assert len(candidate.penalties) == 2

    scored = score_candidate(candidate, scenario, ScoreConfig(kt=0.001, ct=5000.0, kv=0.004, cv=800.0))
    assert scored.penalty_score == pytest.approx(20.0)


def test_one_maneuver_breaking_two_rules_is_penalised_once(workshop_script: Path) -> None:
    """The rules deduct per violating maneuver, not per reason.

    初賽規則 §5: "若設計過程中有不合規範者,例如某次機動速度增量 ΔVk > ΔVlim,
    每次違規機動扣 10 分". A single burn that is both over the limit and too
    close to its predecessor is one violating maneuver, and was being charged
    twice.
    """
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = candidate_with_burns(
        Burn(1000.0, (0.5, 0.0, 0.0), (0.5, 0.0, 0.0), 0),
        Burn(1050.0, (9.0, 0.0, 0.0), (9.0, 0.0, 0.0), 1),
    )
    result = apply_constraints(candidate, scenario, config)
    assert not result.valid
    assert len(result.violations) == 2, result.violations
    assert len(candidate.penalties) == 1, candidate.penalties
    # The one entry still carries both reasons.
    assert "magnitude" in candidate.penalties[0]
    assert "separation" in candidate.penalties[0]

    scored = score_candidate(candidate, scenario, ScoreConfig(kt=0.001, ct=5000.0, kv=0.004, cv=800.0))
    assert scored.penalty_score == pytest.approx(10.0)


def test_a_clean_candidate_records_no_penalties(workshop_script: Path) -> None:
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = candidate_with_burns(Burn(1000.0, (0.5, 0.0, 0.0), (0.5, 0.0, 0.0), 0))
    assert apply_constraints(candidate, scenario, config).valid
    assert candidate.penalties == ()


def _frontier_candidate(*burns: Burn) -> Candidate:
    candidate = candidate_with_burns(*burns)
    candidate.metadata["event_refined"] = True
    return candidate


def test_frontier_follows_the_production_margins_not_the_rules() -> None:
    config = OrbitWinConfig()
    safe = config.policy.dv_safe_limit_km_s
    gap = config.policy.safe_burn_separation_s

    inside = _frontier_candidate(Burn(1000.0, (safe - 0.01, 0.0, 0.0), (0.0, 0.0, 0.0), 0))
    assert frontier_of(inside, config) == ROBUST_FRONTIER

    # Legal under the 1.5 km/s rule, outside the 1.495 km/s margin.
    over_margin = _frontier_candidate(Burn(1000.0, (1.4990, 0.0, 0.0), (0.0, 0.0, 0.0), 0))
    assert over_margin.burns[0].magnitude_km_s <= config.policy.dv_limit_km_s
    assert frontier_of(over_margin, config) == FORMAL_FRONTIER

    # Legal under the 100 s rule, inside the 101 s margin.
    tight = _frontier_candidate(
        Burn(1000.0, (0.2, 0.0, 0.0), (0.0, 0.0, 0.0), 0),
        Burn(1000.0 + gap - 0.5, (0.2, 0.0, 0.0), (0.0, 0.0, 0.0), 1),
    )
    assert frontier_of(tight, config) == FORMAL_FRONTIER


def test_apply_constraints_records_the_frontier(workshop_script: Path) -> None:
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = _frontier_candidate(Burn(1000.0, (1.499, 0.0, 0.0), (1.499, 0.0, 0.0), 0))
    result = apply_constraints(candidate, scenario, config)
    assert result.valid, "1.499 km/s is legal under the 1.5 km/s rule"
    assert candidate.metadata["frontier"] == FORMAL_FRONTIER
    assert any("safety limit" in warning for warning in result.warnings)


def test_the_formal_frontier_cannot_crowd_out_the_robust_one() -> None:
    # A burn at 1.499 km/s buys time nothing at 1.495 can match, so on a shared
    # (time, delta-v) archive it displaces exactly the candidates this project
    # would rather submit. With keep=1 a shared archive keeps one candidate; two
    # frontiers keep one each.
    from orbitwin.search import _prune_per_frontier

    def point(dv: float, intercept: float, frontier: str, tag: str) -> Candidate:
        candidate = Candidate(
            burns=(Burn(1000.0, (dv, 0.0, 0.0), (dv, 0.0, 0.0), 0),),
            intercept_time_s=intercept,
            minimum_distance_km=0.1,
            distance_at_completion_km=5.0,
            successful=True,
            analytical_endpoint_error_km=0.1,
            branch=tag,
            metadata={"arrival_time_s": intercept, "frontier": frontier},
        )
        return candidate

    fast_formal = point(1.499, 3000.0, FORMAL_FRONTIER, "fast-formal")
    slow_robust = point(1.400, 9000.0, ROBUST_FRONTIER, "slow-robust")
    kept = _prune_per_frontier([fast_formal, slow_robust], keep=1)
    frontiers = {str(candidate.metadata["frontier"]) for candidate in kept}
    assert frontiers == {ROBUST_FRONTIER, FORMAL_FRONTIER}
    assert len(kept) == 2


def _plain_candidate() -> Candidate:
    return Candidate(
        burns=(),
        intercept_time_s=1000.0,
        minimum_distance_km=1.0,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=None,
        branch="t",
    )


def test_stage_and_verification_cannot_be_asserted_directly() -> None:
    # The emission gate hands the submission filename to whatever the stage
    # says. When both were plain fields, one assignment anywhere in the codebase
    # was enough to award an authority level nothing had earned.
    from orbitwin.models import CandidateStage

    candidate = _plain_candidate()
    with pytest.raises(AttributeError):
        candidate.stage = CandidateStage.CERTIFIED  # type: ignore[misc]
    with pytest.raises(AttributeError):
        candidate.verified_by_gmat = True  # type: ignore[misc]
    assert candidate.stage is CandidateStage.ANALYTICAL
    assert candidate.verified_by_gmat is False


def test_a_gmat_tier_requires_evidence() -> None:
    from orbitwin.models import CandidateStage

    candidate = _plain_candidate()
    with pytest.raises(ValueError, match="requires evidence"):
        candidate.promote(CandidateStage.CERTIFIED)
    with pytest.raises(ValueError, match="requires evidence"):
        candidate.promote(CandidateStage.PUBLIC_GMAT_VERIFIED)
    # Below the GMAT tiers no run has happened, so none is demanded.
    candidate.promote(CandidateStage.RENDERABLE)
    assert candidate.stage is CandidateStage.RENDERABLE

    candidate.promote(CandidateStage.PUBLIC_GMAT_VERIFIED, run_id="R1")
    assert candidate.stage is CandidateStage.PUBLIC_GMAT_VERIFIED
    assert candidate.stage_evidence["public_gmat_verified"] == {"run_id": "R1"}


def test_verification_follows_the_stage_and_the_verdict() -> None:
    from orbitwin.models import CandidateStage

    candidate = _plain_candidate()
    candidate.promote(CandidateStage.PUBLIC_GMAT_VERIFIED, run_id="R1")
    assert candidate.verified_by_gmat is True
    # A later rejection does not demote -- the evidence was still obtained --
    # but it does stop the candidate claiming verification.
    candidate.successful = False
    assert candidate.verified_by_gmat is False
    assert candidate.stage is CandidateStage.PUBLIC_GMAT_VERIFIED


def test_promotion_is_monotonic() -> None:
    from orbitwin.models import CandidateStage

    candidate = _plain_candidate()
    candidate.promote(CandidateStage.ORGANIZER_GMAT_VERIFIED, run_id="R1")
    candidate.promote(CandidateStage.ANALYTICAL)
    assert candidate.stage is CandidateStage.ORGANIZER_GMAT_VERIFIED


def test_a_file_claiming_verification_without_a_stage_restores_unverified() -> None:
    # An older candidates.json carries verified_by_gmat: true and no stage.
    # Restoring the boolean would resurrect a claim with nothing behind it.
    from orbitwin.io import candidate_from_dict

    restored = candidate_from_dict(
        {
            "candidate_id": "C-old",
            "burns": [],
            "intercept_time_s": 1000.0,
            "minimum_distance_km": 1.0,
            "distance_at_completion_km": 5.0,
            "total_dv_km_s": 0.0,
            "successful": True,
            "analytical_endpoint_error_km": None,
            "branch": "old",
            "verified_by_gmat": True,
            "metadata": {},
        }
    )
    assert restored.verified_by_gmat is False
    assert restored.stage.value == "analytical"


def test_stage_survives_pickling_for_the_worker_pool() -> None:
    import pickle

    from orbitwin.models import CandidateStage

    candidate = _plain_candidate()
    candidate.promote(CandidateStage.PUBLIC_GMAT_VERIFIED, run_id="R1")
    restored = pickle.loads(pickle.dumps(candidate))
    assert restored.stage is CandidateStage.PUBLIC_GMAT_VERIFIED
    assert restored.verified_by_gmat is True
    assert restored.stage_evidence == candidate.stage_evidence


def test_total_delta_v_is_derived_from_the_burns(workshop_script: Path) -> None:
    # It used to be a stored number nothing reconciled with the burns. A
    # candidate could declare 99 km/s while carrying a single 0.5 km/s impulse
    # and pass validate_candidate, the renderer and assert_final_script_policy
    # in silence, while the official score, the Pareto pruning and the tie-break
    # all read the declared figure.
    candidate = candidate_with_burns(
        Burn(1000.0, (0.3, 0.4, 0.0), (0.3, 0.4, 0.0), 0),
        Burn(2000.0, (0.6, 0.8, 0.0), (0.6, 0.8, 0.0), 1),
    )
    assert candidate.total_dv_km_s == pytest.approx(0.5 + 1.0)

    with pytest.raises(AttributeError):
        candidate.total_dv_km_s = 99.0  # type: ignore[misc]

    with pytest.raises(TypeError):
        Candidate(
            burns=(),
            intercept_time_s=1.0,
            minimum_distance_km=1.0,
            distance_at_completion_km=5.0,
            total_dv_km_s=99.0,  # type: ignore[call-arg]
            successful=True,
            analytical_endpoint_error_km=None,
            branch="declared",
        )


def test_the_internal_success_gate_is_not_a_competition_penalty(
    workshop_script: Path,
) -> None:
    # The official formula deducts ten points per violation, so only rule
    # breaches belong in penalties. "not a verified successful interception" is
    # our own gate: a candidate GMAT rejected has broken no rule, and reporting
    # a ten point deduction for it describes a penalty the organizer would
    # never apply.
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    rejected = candidate_with_burns(Burn(1000.0, (0.5, 0.0, 0.0), (0.5, 0.0, 0.0), 0))
    rejected.successful = False

    result = apply_constraints(rejected, scenario, config, require_success=True)
    assert result.valid is False
    assert any("verified successful interception" in v for v in result.violations)
    assert result.rule_violations == ()
    assert rejected.penalties == ()

    scored = score_candidate(rejected, scenario, ScoreConfig(kt=0.001, ct=5000.0, kv=0.004, cv=800.0))
    assert scored.penalty_score == 0.0

    # A real rule breach still lands there.
    illegal = candidate_with_burns(Burn(1000.0, (9.0, 0.0, 0.0), (9.0, 0.0, 0.0), 0))
    apply_constraints(illegal, scenario, config)
    assert illegal.penalties
    assert all("verified successful" not in p for p in illegal.penalties)


def test_the_sort_key_shape_does_not_depend_on_the_candidate() -> None:
    """Ordering mode belongs to the run, not to each candidate.

    Branching on `candidate.score is not None` produced keys of different
    lengths whose third element was `-score` for one candidate and
    `total_dv_km_s` for another. Both are floats, so a mixed list sorted
    without complaint while comparing quantities that have nothing to do with
    each other.
    """
    from functools import partial

    from orbitwin.pipeline import _sort_key

    scored = _ranked(80.0, 1.0, 0.5, 5000.0, "scored")
    unscored = _ranked(80.0, 1.0, 0.9, 5000.0, "unscored")
    unscored.score = None
    unscored.score_basis = None

    for mode in (True, False):
        key = partial(_sort_key, scenario=None, scored=mode)
        assert len(key(scored)) == len(key(unscored))
        # And the ordering never raises on a mixed list.
        sorted([scored, unscored], key=key)

    # In a scored run a candidate without a score ranks last, not first.
    order = partial(_sort_key, scenario=None, scored=True)
    assert sorted([unscored, scored], key=order)[0] is scored


def _ranking_rules(workshop_script: Path):
    import dataclasses

    base = OrbitWinConfig()
    config = dataclasses.replace(base, policy=dataclasses.replace(
        base.policy, rules="ranking", horizon_s=30000.0, dv_limit_km_s=0.25,
        dv_safe_limit_km_s=0.249, total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99,
    ))
    return parse_gmat_script(workshop_script, config)[0], config


#: 100 s after this epoch is 99.99999999999818 s after it in binary floating point.
_ROUNDED_EPOCH_S = 16326.762425465526


@pytest.mark.parametrize(
    ("first_s", "gap_s", "penalised"),
    [(1000.0, 100.0, False), (_ROUNDED_EPOCH_S, 100.0, False),
     (1000.0, 100.0 - 2.0e-9, True), (1000.0, 99.999, True)],
    ids=["exactly-100-s", "100-s-as-rounded", "2-ns-short", "1-ms-short"],
)
def test_the_ranking_rules_penalise_a_gap_under_100_s_and_not_one_of_100(
    workshop_script: Path, first_s: float, gap_s: float, penalised: bool
) -> None:
    """R2 C09: the rule charges Δt < 100 s. The 1e-9 s slack covers float error only."""
    scenario, config = _ranking_rules(workshop_script)
    second_s = first_s + gap_s
    candidate = candidate_with_burns(
        Burn(first_s, (0.1, 0.0, 0.0), (0.1, 0.0, 0.0), 0),
        Burn(second_s, (0.1, 0.0, 0.0), (0.1, 0.0, 0.0), 1),
    )
    if first_s == _ROUNDED_EPOCH_S:
        assert second_s - first_s < 100.0
    result = apply_constraints(candidate, scenario, config)
    assert result.valid is not penalised
    assert len(candidate.penalties) == int(penalised)
    scored = score_candidate(candidate, scenario, ScoreConfig(
        kt=0.001, ct=5000.0, kv=0.004, cv=800.0, delta_v_unit="m/s", cr_km=100.0))
    assert scored.penalty_score == (10.0 if penalised else 0.0)
    # Inside our 101 s margin either way.
    assert frontier_of(candidate, config) == FORMAL_FRONTIER


@pytest.mark.parametrize("exponent", [-700.5, -800.0, -1.0e6, -math.inf])
def test_a_logistic_far_below_its_centre_is_its_whole_weight(exponent: float) -> None:
    """R2 S03: past the exp guard the term saturates to its weight, not to 0."""
    from orbitwin.score import _logistic_descending

    assert _logistic_descending(25.0, exponent) == 25.0
    assert _logistic_descending(25.0, -exponent) == 0.0


def test_extreme_coefficients_score_the_saturated_terms_whole(workshop_script: Path) -> None:
    """kv 1 per m/s 1900 m/s under Cv, kt 1 per s 4000 s under Ct: exponents of -1900, -4000."""
    scenario, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    candidate = candidate_with_burns(Burn(0.0, (0.1, 0.0, 0.0), (0.1, 0.0, 0.0), 0))
    scored = score_candidate(candidate, scenario, ScoreConfig(
        kt=1.0, ct=5000.0, kv=1.0, cv=2000.0, delta_v_unit="m/s"))
    assert (scored.time_score, scored.delta_v_score) == (25.0, 25.0)


def test_a_tie_reads_the_closest_approach_over_the_scored_window(workshop_script: Path) -> None:
    """R2: under closest_approach_window "from_origin" d_min counts the alignment's
    rows, and the tie-break read the joint rows alone."""
    import dataclasses
    from functools import partial

    from orbitwin.pipeline import _sort_key
    from orbitwin.score import scored_distance_km, tie_break_distance_km

    scenario, _ = _ranking_rules(workshop_script)
    windowed = dataclasses.replace(scenario, alignment_minimum_distance_km=500.0)
    far_cheap = _ranked(70.0, 900.0, 0.1, 1000.0, "far-cheap")
    near_costly = _ranked(70.0, 600.0, 0.2, 1000.0, "near-costly")
    unrefined = _ranked(70.0, 3.2e-10, 0.1, 1000.0, "unrefined")
    unrefined.metadata["event_refined"] = False
    for miss in (far_cheap, near_costly, unrefined):
        miss.successful, miss.intercept_time_s = False, None
    archive = [near_costly, unrefined, far_cheap]
    assert sorted(archive, key=partial(ranking_key, scenario=scenario))[0] is near_costly
    # Both 500 km on the scored window, so delta-v decides; unmeasured stays last.
    assert sorted(archive, key=partial(ranking_key, scenario=windowed)) == [
        far_cheap, near_costly, unrefined]
    order = partial(_sort_key, scenario=windowed, scored=True, misses_count=True)
    assert sorted(archive, key=order)[0] is far_cheap
    assert tie_break_distance_km(far_cheap, windowed) == scored_distance_km(far_cheap, windowed)
    assert tie_break_distance_km(unrefined, windowed) == math.inf
    preliminary = dataclasses.replace(windowed, rules="preliminary")
    assert tie_break_distance_km(far_cheap, preliminary) == 900.0
