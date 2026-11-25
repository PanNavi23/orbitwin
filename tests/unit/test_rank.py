from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from conftest import TEAM_NUMBER, ranking_copy

from orbitwin.config import OrbitWinConfig, ScoreConfig
from orbitwin.errors import OrbitWinError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Candidate
from orbitwin.rank import (
    NO_GMAT_ARCHIVE,
    NO_GMAT_HUNT,
    rank_report,
    rank_under,
    winner_stability,
)
from orbitwin.score import score_candidate, score_stamp

# Choosing the submission after the coefficients are announced. 初賽規則 §5 fixes
# the distance term at 50 for every success, so the ranking among successes is
# made entirely by the two logistics, whose centres arrive on the day.

# Two real frontier points from a DifferentPlane_Circular archive: the fast
# expensive one and the slow cheap one. Across a 625-point coefficient sweep
# these were the only two that ever won, 44.2% and 55.8% of the time.
FAST = (13208.2, 0.7853)
CHEAP = (26030.7, 0.5439)


def _candidate(intercept_s: float, dv_km_s: float, branch: str) -> Candidate:
    burn = Burn(time_s=100.0, dv_inertial_km_s=(dv_km_s, 0.0, 0.0),
                dv_script_km_s=(dv_km_s, 0.0, 0.0))
    return Candidate(
        burns=(burn,),
        intercept_time_s=intercept_s,
        minimum_distance_km=1.0,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=None,
        branch=branch,
        metadata={"event_refined": True},
    )


@pytest.fixture
def frontier() -> list[Candidate]:
    return [_candidate(*FAST, "fast"), _candidate(*CHEAP, "cheap")]


@pytest.fixture()
def workshop_script(workshop_script: Path, tmp_path: Path) -> Path:
    """The workshop with the ranking report name; certify refuses it without one."""
    return ranking_copy(workshop_script, tmp_path)


@pytest.fixture
def scenario(workshop_script: Path):
    parsed, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    return parsed


def _score_config(**overrides) -> ScoreConfig:
    base = {"kt": 0.001, "ct": 20000.0, "kv": 0.004, "cv": 800.0}
    return ScoreConfig(**{**base, **overrides})


def _cfg(**overrides) -> OrbitWinConfig:
    """rank_report applies solve's submission gates, so it needs the whole config."""
    return OrbitWinConfig(score=_score_config(**overrides))


def test_the_announced_coefficients_decide_which_candidate_wins(scenario, frontier) -> None:
    """The whole reason the archive is kept instead of one candidate.

    A cheap Cv centre punishes the 785 m/s solution and the slow one wins; a
    tight Ct punishes the 26030 s solution and the fast one wins. Same archive,
    same trajectories, opposite submissions.
    """
    values_matter = rank_under(scenario, frontier, _score_config(cv=600.0, ct=30000.0))
    assert values_matter[0][0].branch == "cheap"

    time_matters = rank_under(scenario, frontier, _score_config(cv=2000.0, ct=15000.0))
    assert time_matters[0][0].branch == "fast"


def test_committing_early_costs_a_whole_logistic_term(scenario, frontier) -> None:
    """Measured: up to 25 points, not a rounding difference."""
    punishing = _score_config(kt=0.01, ct=15000.0, kv=0.0005, cv=2000.0)
    ranked = rank_under(scenario, frontier, punishing)
    best, worst = ranked[0][1], ranked[-1][1]
    assert best.time_score - worst.time_score > 24.0


def test_the_what_if_sweep_does_not_overwrite_the_announced_scores(
    scenario, frontier
) -> None:
    """The sweep ranks under coefficients nobody announced.

    Writing those onto the candidates would leave the archive scored by
    whichever trial ran last, and every downstream reader -- the CSV, certify,
    the operator -- would see it as the official score.
    """
    announced = _score_config()
    report = rank_report(scenario, frontier, OrbitWinConfig(score=announced), tolerance=0.5)
    # Snapshot before anything else runs. The first version of this test called
    # rank_under to compute the expected value -- and rank_under calls
    # apply_score, which repaired the very thing being checked, so the assertion
    # compared a value against itself. Measured under a mutation that made the
    # sweep write its trial scores: the archive ended up 8.13 points off the
    # announced score, score_is_stale was False, and all sixteen tests passed.
    stored = {candidate.candidate_id: candidate.score for candidate in frontier}
    for candidate in frontier:
        assert not candidate.score_is_stale
        # score_candidate is pure: it computes and writes nothing.
        expected = score_candidate(candidate, scenario, announced).raw_score
        assert stored[candidate.candidate_id] == pytest.approx(expected), (
            "the stored score is not the announced one, so a what-if nudge leaked"
        )
    assert report["winner"]["candidate_id"] == max(
        frontier, key=lambda c: stored[c.candidate_id]
    ).candidate_id


def test_a_delta_v_centre_in_the_wrong_unit_is_reported_not_silently_obeyed(
    scenario, frontier
) -> None:
    """Cv announced in km/s and entered against m/s saturates the term.

    Every candidate then scores the same 25 on delta-v, the ranking is made by
    time alone, and nothing raises. It is worth up to 25 points and the only
    symptom is that the term stopped separating anybody.
    """
    # Announced kv=4.0 per km/s and Cv=0.8 km/s. Fed 785.3 and 543.9 m/s, both
    # exponents land past 2000, both terms return exactly 0, spread 0.
    wrong_unit = _score_config(kv=4.0, cv=0.8, delta_v_unit="m/s")
    report = rank_report(scenario, frontier, OrbitWinConfig(score=wrong_unit))
    assert report["term_spread_points"]["delta_v"] < 0.5
    assert any("delta_v term spans" in warning for warning in report["warnings"])
    assert any("score.delta_v_unit" in warning for warning in report["warnings"])


def test_a_term_that_separates_candidates_is_not_reported(scenario, frontier) -> None:
    report = rank_report(scenario, frontier, _cfg())
    assert report["term_spread_points"]["delta_v"] > 0.5
    assert not any("delta_v term spans" in warning for warning in report["warnings"])


def test_an_all_success_archive_does_not_warn_about_its_flat_distance_term(
    scenario, frontier
) -> None:
    """Every success scores exactly 50 on distance. That is §4, not a fault."""
    report = rank_report(scenario, frontier, _cfg())
    assert report["term_spread_points"]["distance"] == pytest.approx(0.0)
    assert not any("distance term spans" in warning for warning in report["warnings"])


def test_a_winner_sitting_on_a_knife_edge_is_named(scenario, frontier) -> None:
    """Measured: these four put the two candidates 0.0003 points apart.

    At a tie that close every coefficient matters -- a 20% move in any of the
    four, in either direction, changes which script we submit. That is exactly
    the case where a mistyped digit costs the round and nothing complains.
    """
    edge = _score_config(kt=0.0005, ct=12750.0, kv=0.008, cv=630.0)
    stability = winner_stability(scenario, frontier, edge, tolerance=0.2)
    assert not stability["stable"]
    assert {change["coefficient"] for change in stability["changes"]} == {
        "kt", "ct", "kv", "cv",
    }
    report = rank_report(scenario, frontier, OrbitWinConfig(score=edge), tolerance=0.2)
    assert any("Re-read the announced value" in warning for warning in report["warnings"])


def test_a_stable_winner_reports_no_change(scenario, frontier) -> None:
    decisive = _score_config(kt=0.001, ct=30000.0, kv=0.02, cv=600.0)
    stability = winner_stability(scenario, frontier, decisive, tolerance=0.1)
    assert stability["stable"]
    assert stability["changes"] == []


def test_ranking_without_announced_coefficients_is_refused(scenario, frontier) -> None:
    with pytest.raises(OrbitWinError, match="must not be guessed"):
        rank_report(scenario, frontier, OrbitWinConfig())


def test_an_empty_archive_is_refused(scenario) -> None:
    with pytest.raises(OrbitWinError, match="nothing to rank"):
        rank_report(scenario, [], _cfg())


def test_an_unverified_winner_is_flagged(scenario, frontier) -> None:
    """A score is arithmetic. It is not evidence that GMAT flies the arc."""
    report = rank_report(scenario, frontier, _cfg())
    assert any("only certify establishes" in warning for warning in report["warnings"])


def test_a_horizon_that_is_not_a_number_is_refused(scenario, frontier) -> None:
    """A failure is scored on its separation at 4TA (§4).

    With a NaN horizon that completion time is NaN, and the logistic returns a
    number anyway. Refusing beats ranking on it.
    """
    broken = dataclasses.replace(scenario, max_time_s=float("nan"))
    with pytest.raises(OrbitWinError, match="horizon is not finite"):
        rank_report(broken, frontier, _cfg())


def test_a_coefficient_of_zero_is_reported_as_untested_not_as_stable(
    scenario, frontier
) -> None:
    """A percentage of zero is zero.

    The sweep would have moved Ct nowhere and then reported the winner stable,
    which reads as "checked and fine" rather than "never checked".
    """
    stability = winner_stability(scenario, frontier, _score_config(ct=0.0), tolerance=0.2)
    assert stability["not_tested"] == ["ct"]
    report = rank_report(scenario, frontier, _cfg(ct=0.0), tolerance=0.2)
    assert any("never tested" in warning for warning in report["warnings"])


def test_a_failure_cannot_hide_a_term_that_separates_no_success(
    scenario, frontier
) -> None:
    """A failed candidate sits at the horizon and scores 0 on distance.

    Measuring the time spread across it would show tens of points of range
    while every success sat on the same flat tail, so the flat-term check --
    the one that catches a wrong unit -- would stay silent.
    """
    failed = _candidate(FAST[0], FAST[1], "failed")
    failed.successful = False
    failed.intercept_time_s = None
    failed.distance_at_completion_km = 400.0

    # Ct sits between both intercepts and the horizon, so both successes score
    # 25.0000 on time and the failure scores 0.0000: 0 points of spread among
    # the candidates that can win, 25 across the archive.
    dead_time_term = _score_config(kt=0.01, ct=33000.0)
    report = rank_report(scenario, [*frontier, failed], OrbitWinConfig(score=dead_time_term))
    assert report["term_spread_scope"] == "successful"
    assert report["term_spread_points"]["time"] < 0.5
    assert any("time term spans" in warning for warning in report["warnings"])


def test_a_tolerance_of_zero_is_refused_rather_than_called_stable(scenario, frontier) -> None:
    """Nudging by 0% moves nothing, so every winner matches and nothing was tested.

    Same shape as the zero-valued coefficient above. That one was caught and
    this one was not, because the guard was written for the coefficient rather
    than for the multiplication both of them go through.
    """
    for tolerance in (0.0, -0.1):
        with pytest.raises(OrbitWinError, match="positive tolerance"):
            winner_stability(scenario, frontier, _score_config(), tolerance=tolerance)


def test_an_archive_where_nothing_succeeds_does_not_blame_the_time_unit(
    scenario, frontier
) -> None:
    """§4 makes Tteam the horizon for every failure, so time cannot separate them.

    Warning about it would send the operator to re-check kt, Ct and the time
    unit, all of which are fine. The mirror image of the distance exemption:
    distance is flat when everything succeeds, time is flat when nothing does.
    The two are mirror images, not cumulative -- exempting distance here too
    suppressed the only warning that mattered, since §4 scores a failed archive
    on Δr(4TA) and distance is then the sole discriminator.
    """
    failures = []
    for index, (intercept, dv) in enumerate((FAST, CHEAP)):
        candidate = _candidate(intercept, dv, f"failed{index}")
        candidate.successful = False
        candidate.intercept_time_s = None
        candidate.distance_at_completion_km = 120.0 + index * 300.0
        failures.append(candidate)

    report = rank_report(scenario, failures, _cfg())
    assert report["successful_count"] == 0
    assert report["term_spread_scope"] == "all"
    assert report["term_spread_points"]["time"] == pytest.approx(0.0)
    assert not any("time term spans" in warning for warning in report["warnings"])
    # The delta-v term still separates them, so it is not exempted.
    assert report["term_spread_points"]["delta_v"] > 0.5


def test_a_winner_whose_interception_was_never_measured_is_named_as_such(
    scenario, frontier
) -> None:
    """The finding that mattered most, reproduced.

    Measured on a real 120-candidate archive: 80 were never event-refined, so
    their `successful` flag is the Lambert endpoint residual and their
    `minimum_distance_km` sits around 1.8e-10 km. `solve` refuses to nominate
    such a candidate; ranking applied none of that, and a saturated pair of
    logistics scores one exactly 100.
    """
    unmeasured = _candidate(9000.0, 0.30, "unmeasured")
    unmeasured.metadata["event_refined"] = False
    unmeasured.minimum_distance_km = 1.8e-10

    report = rank_report(scenario, [*frontier, unmeasured], _cfg())
    assert report["winner"]["candidate_id"] == unmeasured.candidate_id, "it does win"
    assert report["winner"]["submittable"] is False
    assert report["winner"]["event_refined"] is False
    assert any("interception never measured" in b for b in report["winner"]["blockers"])
    assert any("could not be submitted" in w for w in report["warnings"])
    assert any("best submittable candidate" in w for w in report["warnings"])
    # And the ranking still lists it -- the archive is what the operator chooses
    # from, so a hidden row is worse than a labelled one.
    assert report["submittable_count"] == len(frontier)


def test_the_measured_minimum_distance_is_reported_beside_the_raw_one(
    scenario, frontier
) -> None:
    """§6's first tie-break reads the guarded value; the report showed the raw one.

    An unrefined candidate's raw field is a ~1e-10 km endpoint residual. Showing
    only that puts a number in front of the operator that the ranking did not use.
    """
    unmeasured = _candidate(9000.0, 0.30, "unmeasured")
    unmeasured.metadata["event_refined"] = False
    unmeasured.minimum_distance_km = 1.8e-10
    report = rank_report(scenario, [unmeasured], _cfg())
    row = report["winner"]
    assert row["minimum_distance_km"] == pytest.approx(1.8e-10)
    assert row["measured_minimum_distance_km"] is None, "unrefined means unmeasured"


def test_an_all_failure_archive_is_warned_about_its_flat_distance_term(scenario) -> None:
    """The exemption was backwards, and this is the case it hid.

    §4 scores a failed mission on Δr(4TA), so with nothing succeeding the
    distance term is the only discriminator and carries 50 of the 100 points.
    Exempting it there suppressed the one warning worth having.
    """
    failures = []
    for index, distance in enumerate((620.0, 640.0, 660.0)):
        candidate = _candidate(20000.0, 0.5, f"fail{index}")
        candidate.successful = False
        candidate.intercept_time_s = None
        candidate.distance_at_completion_km = distance
        failures.append(candidate)

    report = rank_report(scenario, failures, _cfg())
    assert report["successful_count"] == 0
    assert report["term_spread_points"]["distance"] < 0.5
    assert any("distance term spans" in w for w in report["warnings"])
    assert not any("kt=" in w for w in _distance_warnings(report)), "distance has no unit"
    # Time stays exempt: Tteam is the horizon for all of them by §4.
    assert not any("time term spans" in w for w in report["warnings"])


@pytest.mark.parametrize(
    "tolerance",
    [0.0, -0.1, float("nan"), float("inf"), 1e-17, 1.0, 1.5],
    ids=["zero", "negative", "nan", "inf", "below-eps", "exactly-one", "above-one"],
)
def test_a_tolerance_that_cannot_nudge_or_would_invert_is_refused(
    scenario, frontier, tolerance: float
) -> None:
    """All of one class, not one member of it.

    Measured: nan passes a bare `<= 0` check because every NaN comparison is
    false, and then flags all four coefficients at once and writes bare NaN into
    the report. 1e-17 leaves `kt * (1 + tol) == kt` bit for bit. 1.0 drives the
    downward nudge to exactly 0.0 and above that it goes negative, inverting the
    logistic so a slower, costlier mission scores higher.
    """
    with pytest.raises(OrbitWinError):
        winner_stability(scenario, frontier, _score_config(), tolerance=tolerance)


@pytest.mark.parametrize("top", [0, -1, -4])
def test_a_top_below_one_is_refused_rather_than_silently_dropping_rows(
    scenario, frontier, top: int
) -> None:
    """0 lists nothing beside a populated winner; negative slices from the end.

    `--top -1` reads like a request for everything and quietly omits the worst
    row, which is the shape of a truncation nobody can see.
    """
    with pytest.raises(OrbitWinError, match="at least 1"):
        rank_report(scenario, frontier, _cfg(), top=top)


def test_the_report_says_how_many_rows_it_listed(scenario, frontier) -> None:
    report = rank_report(scenario, frontier, _cfg(), top=1)
    assert report["candidate_count"] == 2
    assert report["listed"] == 1
    assert len(report["ranking"]) == 1


def test_an_archive_from_a_different_script_is_refused(
    workshop_script: Path, synthetic_script: Path
) -> None:
    """The horizon comes from the script, and it is every failure's completion time.

    Measured: the same failed candidates ranked against DifferentPlane_Circular
    and HighPlaneChange_Circular, whose horizons are 33988.714 s and 42830.675 s,
    scored 50.865 and 25.865 — exactly 25.000 apart, the whole time term, with no
    warning of any kind.
    """
    config = _cfg(kt=0.01, ct=38000.0)
    here, _ = parse_gmat_script(workshop_script, config)
    elsewhere, _ = parse_gmat_script(synthetic_script, config)
    assert here.source_sha256 != elsewhere.source_sha256

    candidate = _candidate(FAST[0], FAST[1], "stamped")
    candidate.metadata["scenario_sha256"] = elsewhere.source_sha256

    with pytest.raises(OrbitWinError, match="produced from a different script"):
        rank_report(here, [candidate], config)
    # And it ranks fine against the script it came from.
    assert rank_report(elsewhere, [candidate], config)["candidate_count"] == 1


def test_one_matching_stamp_does_not_admit_the_others(
    workshop_script: Path, synthetic_script: Path
) -> None:
    """A candidate from this script used to carry one from another script past the check."""
    config = _cfg(kt=0.01, ct=38000.0)
    here, _ = parse_gmat_script(workshop_script, config)
    elsewhere, _ = parse_gmat_script(synthetic_script, config)
    ours = _candidate(FAST[0], FAST[1], "ours")
    ours.metadata["scenario_sha256"] = here.source_sha256
    theirs = _candidate(CHEAP[0], CHEAP[1], "theirs")
    theirs.metadata["scenario_sha256"] = elsewhere.source_sha256

    with pytest.raises(OrbitWinError, match=elsewhere.source_sha256):
        rank_report(here, [ours, theirs], config)


def test_a_partly_stamped_archive_says_which_part_was_checked(workshop_script: Path) -> None:
    config = _cfg()
    here, _ = parse_gmat_script(workshop_script, config)
    ours = _candidate(FAST[0], FAST[1], "ours")
    ours.metadata["scenario_sha256"] = here.source_sha256
    report = rank_report(here, [ours, _candidate(CHEAP[0], CHEAP[1], "bare")], config)
    assert any("1 of the 2 candidates carry no scenario fingerprint" in w for w in report["warnings"])


def test_an_archive_from_another_horizon_is_refused(workshop_script: Path) -> None:
    """Same script, different policy.horizon_s: the hash cannot see that.

    Measured before this check: solved at 5000 s, ranked at 4*T_A = 39808 s,
    a failure's time term went from 12.5 to 2e-14 with no warning.
    """
    import dataclasses

    config = _cfg(kt=0.01, ct=5000.0)
    here, _ = parse_gmat_script(workshop_script, config)
    announced = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, horizon_s=5000.0)
    )
    solved, _ = parse_gmat_script(workshop_script, announced)
    candidate = _candidate(FAST[0], FAST[1], "stamped")
    candidate.metadata["scenario_sha256"] = here.source_sha256
    candidate.metadata["horizon_s"] = solved.max_time_s

    with pytest.raises(OrbitWinError, match="solved under a horizon of 5000"):
        rank_report(here, [candidate], config)
    assert rank_report(solved, [candidate], announced)["candidate_count"] == 1


def test_a_nan_horizon_in_the_archive_is_not_a_match(workshop_script: Path) -> None:
    """abs(nan - x) > 1e-6 is False, so the mismatch check let NaN through."""
    config = _cfg(kt=0.01, ct=5000.0)
    here, _ = parse_gmat_script(workshop_script, config)
    candidate = _candidate(FAST[0], FAST[1], "nan")
    candidate.metadata["horizon_s"] = float("nan")
    with pytest.raises(OrbitWinError, match="solved under a horizon of nan"):
        rank_report(here, [candidate], config)


def test_an_unstamped_archive_says_nothing_checked_it(scenario, frontier) -> None:
    """Older archives carry no fingerprint. Say so rather than implying a check."""
    report = rank_report(scenario, frontier, _cfg())
    assert any("no scenario fingerprint" in w for w in report["warnings"])


def test_penalties_are_recomputed_rather_than_read_from_the_file(scenario) -> None:
    """§5 deducts ten points per violating maneuver, and the file is not evidence.

    solve writes `penalties` under whatever profile it ran with; rank runs under
    the announced one. Measured: a 2.0 km/s burn against the 1.5 km/s limit,
    written with an empty penalties tuple, was crowned at 75.204 with
    penalty_count 0. The correct 55.204 loses to a legal candidate.
    """
    illegal = _candidate(13000.0, 2.0, "over-the-limit")
    illegal.penalties = ()
    legal = _candidate(26030.7, CHEAP[1], "legal")

    report = rank_report(scenario, [illegal, legal], _cfg())
    row = next(r for r in report["ranking"] if r["candidate_id"] == illegal.candidate_id)
    assert row["penalty_count"] == 1, "one violating maneuver, not zero and not two reasons"
    assert row["score_components"]["penalty_score"] == pytest.approx(10.0)
    assert row["submittable"] is False
    assert any("exceeds formal limit" in b for b in row["blockers"])


def test_a_single_candidate_archive_does_not_blame_the_units(scenario) -> None:
    """max - min over one element is zero by arithmetic, not by saturation.

    Reachable when keep_candidates prunes to one, or when only the zero-burn
    baseline survives. Reporting it as a flat term sends the operator to
    re-verify the one set of numbers they are forbidden to guess, at the moment
    they can least afford a false alarm.
    """
    report = rank_report(scenario, [_candidate(*FAST, "only")], _cfg())
    assert report["term_spread_points"] == {}
    assert not any("term spans" in warning for warning in report["warnings"])
    # The row itself is still ranked and reported.
    assert report["winner"]["candidate_id"]


def _two_burns(gap_s: float, intercept_s: float, branch: str) -> Candidate:
    burns = (
        Burn(time_s=10.0, dv_inertial_km_s=(0.1, 0.0, 0.0), dv_script_km_s=(0.1, 0.0, 0.0),
             slot_index=0),
        Burn(time_s=10.0 + gap_s, dv_inertial_km_s=(0.1, 0.0, 0.0),
             dv_script_km_s=(0.1, 0.0, 0.0), slot_index=1),
    )
    return Candidate(
        burns=burns, intercept_time_s=intercept_s, minimum_distance_km=1.0,
        distance_at_completion_km=5.0, successful=True, analytical_endpoint_error_km=None,
        branch=branch, metadata={"event_refined": True},
    )


def test_the_stored_score_is_the_one_the_candidate_was_ranked_by(scenario) -> None:
    """Ranked at 72.81 with the recomputed penalty, stored at 82.81 without it."""
    close = _two_burns(99.0, 13000.0, "close")
    assert close.penalties == ()
    ranked = rank_under(scenario, [close], _score_config(), _cfg())
    assert ranked[0][1].penalty_score == 10.0
    assert close.score == pytest.approx(ranked[0][1].raw_score)
    assert len(close.penalties) == 1
    assert not close.score_is_stale


def test_the_report_still_shows_the_penalties_the_file_claimed(scenario) -> None:
    report = rank_report(scenario, [_two_burns(99.0, 13000.0, "close")], _cfg())
    assert report["winner"]["penalty_count"] == 1
    assert report["winner"]["penalty_count_claimed_by_file"] == 0


def test_the_stability_sweep_ranks_under_the_policy_it_was_given(scenario) -> None:
    """Under a 50 s rule the 80 s gap is legal and wins; under the default it does not."""
    gapped = _two_burns(80.0, 13000.0, "gapped")
    spaced = _two_burns(150.0, 13500.0, "spaced")
    base = _cfg()
    # The sweep covers submittable rows; 80 s is inside the 101 s production margin.
    config = dataclasses.replace(base, policy=dataclasses.replace(
        base.policy, min_burn_separation_s=50.0, allow_formal_frontier_submission=True))
    report = rank_report(scenario, [gapped, spaced], config, tolerance=0.1)
    assert report["winner"]["candidate_id"] == gapped.candidate_id
    assert report["winner_stability"]["winner"] == gapped.candidate_id
    # The nudged rankings too: under the default 100 s rule every nudge flips it.
    assert report["winner_stability"]["stable"] is True


def test_a_refused_tolerance_leaves_the_archive_untouched(scenario) -> None:
    close = _two_burns(99.0, 13000.0, "close")
    with pytest.raises(OrbitWinError, match="positive tolerance"):
        rank_report(scenario, [close], _cfg(), tolerance=0.0)
    assert close.penalties == ()
    assert close.score is None


def _rules_cfg(rules: str, **overrides) -> OrbitWinConfig:
    if rules == "ranking":
        overrides.setdefault("cr_km", 100.0)  # §4 eq. (5) as revised 2026/11/04
    base = _cfg(**overrides)
    policy = dataclasses.replace(base.policy, rules=rules, horizon_s=30000.0,
                                 team_number=TEAM_NUMBER if rules == "ranking" else None)
    return dataclasses.replace(base, policy=policy)


def _miss(dv_km_s: float, closest_km: float, branch: str) -> Candidate:
    miss = _candidate(20000.0, dv_km_s, branch)
    miss.successful = False
    miss.intercept_time_s = None
    miss.minimum_distance_km = closest_km
    miss.distance_at_completion_km = 400.0
    return miss


def _flown(candidate: Candidate) -> Candidate:
    """As verify leaves a candidate GMAT flew: its run and the script it flew."""
    candidate.gmat_run_id = f"run-{candidate.branch}"
    event = candidate.metadata.setdefault("gmat_event", {})
    event["flown_script_sha256"] = "0" * 64
    event.setdefault("interpolated_minimum_distance_km", candidate.minimum_distance_km)
    return candidate


@pytest.mark.parametrize("rules", ["preliminary", "ranking"])
def test_a_measured_miss_is_submittable_only_under_the_ranking_rules(
    workshop_script: Path, rules: str
) -> None:
    """The ranking rules score a miss on its closest approach and certify accepts one.

    Measured before this: a 6 km near miss at 71.524 points was labelled "no
    interception" and the report steered the operator to a 52.082-point hit.
    """
    config = _rules_cfg(rules)
    scenario, _ = parse_gmat_script(workshop_script, config)
    hit = _candidate(29000.0, 1.4, "late-hit")
    miss = _flown(_miss(0.3, 6.0, "near-miss"))

    report = rank_report(scenario, [hit, miss], config)
    row = next(r for r in report["ranking"] if r["candidate_id"] == miss.candidate_id)
    if rules == "preliminary":
        assert row["blockers"] == ["no interception"]
        assert report["submittable_count"] == 1
        return
    assert report["winner"]["candidate_id"] == miss.candidate_id
    assert row["submittable"] is True and row["blockers"] == []
    assert not any("could not be submitted" in w for w in report["warnings"])
    # An archive of misses only is normal under the ranking force model.
    alone = rank_report(scenario, [miss], config)
    assert alone["submittable_count"] == 1
    assert not any("nothing here" in w for w in alone["warnings"])


def test_an_unmeasured_miss_is_blocked_for_what_it_lacks(workshop_script: Path) -> None:
    config = _rules_cfg("ranking")
    scenario, _ = parse_gmat_script(workshop_script, config)
    miss = _miss(0.3, 6.0, "unmeasured-miss")
    miss.metadata["event_refined"] = False
    blockers = rank_report(scenario, [miss], config)["winner"]["blockers"]
    assert any("closest approach never measured" in b for b in blockers)
    assert "no interception" not in blockers


def test_a_miss_that_can_win_is_inside_the_term_spread(workshop_script: Path) -> None:
    """With one hit the spread was taken over that hit alone and said nothing.

    Cv entered in m/s against a km/s announcement saturates delta-v for the hit
    and the misses alike; under the ranking rules those misses compete.
    """
    config = _rules_cfg("ranking", kv=4.0, cv=0.8, delta_v_unit="m/s")
    scenario, _ = parse_gmat_script(workshop_script, config)
    archive = [_candidate(*FAST, "hit"), _miss(0.3, 40.0, "miss0"), _miss(0.4, 90.0, "miss1")]
    report = rank_report(scenario, archive, config)
    assert report["term_spread_scope"] == "all"
    assert report["term_spread_points"]["delta_v"] < 0.5
    assert any("delta_v term spans" in w for w in report["warnings"])


def _distance_warnings(report: dict) -> list[str]:
    return [w for w in report["warnings"] if w.startswith("The distance term spans")]


def test_a_flat_distance_term_never_blames_a_unit(workshop_script: Path) -> None:
    """Hits plus near misses is the likely ranking-day archive.

    Measured: hits FAST and CHEAP with a 5.5 km miss put distance 0.249 points
    flat, and the report told the operator to re-check kt, Ct and the time unit.
    Since 2026/11/04 the distance term has C_r, so that is what it names.
    """
    config = _rules_cfg("ranking", kt=0.01, ct=33000.0)
    scenario, _ = parse_gmat_script(workshop_script, config)
    archive = [_candidate(*FAST, "hit0"), _candidate(*CHEAP, "hit1"), _miss(0.3, 5.5, "miss")]
    report = rank_report(scenario, archive, config)
    assert report["term_spread_points"]["distance"] < 0.5
    for warning in _distance_warnings(report):
        assert "kt=" not in warning and "unit is wrong" not in warning
    # Ct past T_max saturates time for hits and the miss alike; that still warns.
    assert report["term_spread_points"]["time"] < 0.5
    assert any("time term spans" in w and "time_unit" in w for w in report["warnings"])

    misses_only = [_miss(0.3, 5.5, "miss0"), _miss(0.5, 5.9, "miss1")]
    report = rank_report(scenario, misses_only, _rules_cfg("ranking"))
    assert report["term_spread_points"]["distance"] < 0.5
    (warning,) = _distance_warnings(report)
    assert "kt=" not in warning and "unit is wrong" not in warning
    # Inside the plateau that is the term itself, not C_r.
    assert "Every candidate passes within 25.0013 km" in warning and "check" not in warning
    assert warning.endswith("so delta-v decides.")


def test_hits_and_misses_on_the_plateau_do_not_blame_c_r(workshop_script: Path) -> None:
    """Hits with misses within 0.1 C_r: flat because the term is, at any C_r."""
    scenario, _ = parse_gmat_script(workshop_script, _rules_cfg("ranking"))
    archive = [_candidate(*FAST, "hit"), _miss(0.3, 12.0, "miss0"), _miss(0.3, 15.0, "miss1")]
    report = rank_report(scenario, archive, _rules_cfg("ranking", kt=0.01, ct=33000.0))
    (warning,) = _distance_warnings(report)
    assert "flat there" in warning and "check" not in warning
    # A hit completes before T_max, so time can still decide.
    assert warning.endswith("so time and delta-v decide.")


def test_a_c_r_far_above_the_misses_shows_its_plateau(workshop_script: Path) -> None:
    """Misses 40 and 900 km off are 49.9 and 0 points apart at 100 km, 0.04 at 1e5 km:
    the warning gives the plateau's edge, 20006 km, rather than guess at a unit."""
    scenario, _ = parse_gmat_script(workshop_script, _rules_cfg("ranking"))
    archive = [_miss(0.3, 40.0, "near"), _miss(0.3, 900.0, "far")]
    right = rank_report(scenario, archive, _rules_cfg("ranking"))
    assert right["term_spread_points"]["distance"] > 40.0
    assert not _distance_warnings(right)
    slipped = rank_report(scenario, archive, _rules_cfg("ranking", cr_km=1.0e5))
    assert slipped["term_spread_points"]["distance"] < 0.5
    (warning,) = _distance_warnings(slipped)
    assert "within 20006.3 km" in warning and "C_r = 100000 km" in warning
    assert slipped["coefficients"]["cr_km"] == 1.0e5


def test_misses_on_the_tail_do_not_blame_a_right_c_r(workshop_script: Path) -> None:
    """R2 at C_r 100 km: the zero burn 4711 km off and misses at 1778 to 2181 km all
    score about 0 on distance. That is the term at this C_r, not a sign C_r is wrong."""
    scenario, _ = parse_gmat_script(workshop_script, _rules_cfg("ranking"))
    archive = [_miss(0.0, 4711.155, "zero"), _miss(0.3, 1778.0, "miss0"),
               _miss(0.3, 2181.0, "miss1")]
    report = rank_report(scenario, archive, _rules_cfg("ranking"))
    (warning,) = _distance_warnings(report)
    assert "passes 1778 km off or farther" in warning
    assert "about 0 for all of them" in warning and "check" not in warning
    # Misses only: every one completes at T_max, so time cannot decide either.
    assert warning.endswith("so delta-v decides.")


@pytest.mark.parametrize(
    ("cr_km", "distances", "says"),
    [
        # 47.83 and 47.68 points.
        (1000.0, (300.0, 310.0),
         "pass 300 to 310 km off, and at C_r = 1000 km their terms lie between 47.68 and 47.83"),
        # R2-like, 3861 to 4718 km apart, all between 0.005 and 0.135 points: not one distance.
        (1500.0, (4718.45, 3861.0, 3861.0, 4401.0),
         "pass 3861 to 4718.45 km off, and at C_r = 1500 km their terms lie between 0.005149 "
         "and 0.1347"),
    ],
)
def test_misses_whose_terms_lie_close_cannot_be_separated(
    workshop_script: Path, cr_km: float, distances: tuple[float, ...], says: str
) -> None:
    scenario, _ = parse_gmat_script(workshop_script, _rules_cfg("ranking"))
    archive = [_miss(0.0 if i == 0 else 0.2, d, f"miss{i}") for i, d in enumerate(distances)]
    report = rank_report(scenario, archive, _rules_cfg("ranking", cr_km=cr_km))
    (warning,) = _distance_warnings(report)
    assert says in warning, warning
    assert "same distance" not in warning and "check" not in warning
    assert warning.endswith("and delta-v decides.")


def test_the_winner_is_tested_against_c_r_too(workshop_script: Path) -> None:
    """60 km off on 0.30 km/s against 6 km on 0.32 km/s: at C_r 700 km the cheap one
    wins by 0.06 points, at 350 km the near one by 0.40."""
    scenario, _ = parse_gmat_script(workshop_script, _rules_cfg("ranking"))
    # Flown: the sweep covers submittable rows.
    archive = [_flown(_miss(0.30, 60.0, "far-cheap")), _flown(_miss(0.32, 6.0, "near-dear"))]
    report = rank_report(scenario, archive, _rules_cfg("ranking", cr_km=700.0), tolerance=0.5)
    assert report["winner"]["candidate_id"] == archive[0].candidate_id
    changes = report["winner_stability"]["changes"]
    assert {"coefficient": "cr_km", "from": 700.0, "to": 350.0,
            "winner": archive[1].candidate_id} in changes


def _stopped(candidate: Candidate, depth_km: float) -> Candidate:
    candidate.metadata["gmat_event"] = {
        "stop_time_s": candidate.intercept_time_s, "stop_depth_km": depth_km,
        "stop_depth_setting_km": 1.5,
    }
    return candidate


def test_a_shallow_hit_is_not_submittable_without_the_opt_in(scenario, frontier) -> None:
    fast, cheap = frontier
    _stopped(fast, 0.01)
    _stopped(cheap, 1.5)
    time_matters = _cfg(cv=2000.0, ct=15000.0)
    report = rank_report(scenario, frontier, time_matters)
    assert report["winner"]["candidate_id"] == fast.candidate_id
    assert not report["winner"]["submittable"]
    assert any("only 10 m inside" in b and "allow_shallow_stop_submission" in b
               for b in report["winner"]["blockers"])
    assert any(f"best submittable candidate is {cheap.candidate_id}" in w
               for w in report["warnings"])


def test_with_the_opt_in_rank_still_names_the_deep_hit(scenario, frontier) -> None:
    fast, cheap = frontier
    _stopped(fast, 0.01)
    _stopped(cheap, 1.5)
    config = _cfg(cv=2000.0, ct=15000.0)
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, allow_shallow_stop_submission=True)
    )
    report = rank_report(scenario, frontier, config)
    assert report["winner"]["candidate_id"] == fast.candidate_id
    assert report["winner"]["submittable"]
    assert any(f"so {cheap.candidate_id} is the one to certify" in w
               for w in report["warnings"])


def test_a_shallow_hit_beside_no_deep_one_is_named_with_the_opt_in(scenario, frontier) -> None:
    fast, _cheap = frontier
    _stopped(fast, 0.01)  # cheap carries no GMAT stop: nothing to prefer it for.
    config = _cfg(cv=2000.0, ct=15000.0)
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, allow_shallow_stop_submission=True)
    )
    report = rank_report(scenario, frontier, config)
    assert report["winner"]["candidate_id"] == fast.candidate_id
    assert not any("is the one to certify" in w for w in report["warnings"])


def _ranking_scenario(workshop_script: Path):
    config = OrbitWinConfig(policy=dataclasses.replace(
        OrbitWinConfig().policy, rules="ranking", horizon_s=30000.0,
        dv_limit_km_s=0.25, dv_safe_limit_km_s=0.249,
        total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99, team_number=TEAM_NUMBER,
    ), score=_score_config(cr_km=100.0))
    return parse_gmat_script(workshop_script, config)[0], config


@pytest.mark.parametrize("opted_in", [False, True])
def test_rank_passes_a_shallow_hit_for_a_miss_ranked_above_the_deep_one(
    workshop_script: Path, opted_in: bool
) -> None:
    """Shallow ~97.9 > miss ~74.0 > deep ~72.7, as solve chooses (test_shallow_hits)."""
    scenario, config = _ranking_scenario(workshop_script)
    config = dataclasses.replace(config, policy=dataclasses.replace(
        config.policy, allow_shallow_stop_submission=opted_in))
    shallow = _flown(_stopped(_candidate(13208.2, 0.20, "shallow"), 0.01))
    deep = _flown(_stopped(_candidate(26030.7, 0.24, "deep"), 1.5))
    miss = _flown(Candidate(
        burns=(), intercept_time_s=None, minimum_distance_km=5.01,
        distance_at_completion_km=5.01, successful=False, analytical_endpoint_error_km=None,
        branch="zero-burn", metadata={"event_refined": True, "scored_miss": True},
    ))
    report = rank_report(scenario, [shallow, deep, miss], config)
    order = [row["candidate_id"] for row in report["ranking"]]
    assert order == [shallow.candidate_id, miss.candidate_id, deep.candidate_id]
    assert report["winner"]["submittable"] is opted_in
    assert report["ceilings"]["best_submittable"] == miss.candidate_id


def _through_the_file(candidates: list[Candidate], tmp_path: Path) -> list[Candidate]:
    """As rank reads them: a file caps every stage at renderable."""
    from orbitwin.io import read_candidates, write_candidates

    write_candidates(tmp_path / "candidates.json", candidates)
    return read_candidates(tmp_path / "candidates.json")


@pytest.mark.parametrize("through_file", [False, True])
def test_the_ranking_rules_block_a_row_gmat_never_flew(
    workshop_script: Path, tmp_path: Path, through_file: bool
) -> None:
    """R4, 2026-11-04: rank named a hit GMAT never flew the best submittable at 86.84,
    and certify flew it 42 km off with exit 0."""
    scenario, config = _ranking_scenario(workshop_script)
    archive = [_candidate(13208.2, 0.20, "unflown"), _flown(_candidate(26030.7, 0.24, "flown"))]
    if through_file:
        archive = _through_the_file(archive, tmp_path)
    unflown, flown = archive
    report = rank_report(scenario, archive, config)
    assert report["winner"]["candidate_id"] == unflown.candidate_id
    assert not report["winner"]["submittable"]
    assert any(b.startswith("never verified by GMAT") for b in report["winner"]["blockers"])
    assert report["ceilings"]["best_submittable"] == flown.candidate_id
    assert any(f"best submittable candidate is {flown.candidate_id}" in w
               for w in report["warnings"])
    assert report["submittable_count"] == report["gmat_verified_count"] == 1
    assert NO_GMAT_ARCHIVE not in report["warnings"]


def test_an_archive_solved_without_gmat_has_no_submittable_row(workshop_script: Path) -> None:
    """A1, 2026-11-04: a solve run with no GMAT pair, then rank recommended a 96.34 hit
    that certify flew 18.92 km off, for 66.26."""
    scenario, config = _ranking_scenario(workshop_script)
    miss = _miss(0.1, 6.0, "miss")
    archive = [_candidate(13208.2, 0.20, "a"), _candidate(26030.7, 0.24, "b"), miss]
    # As scripts/hunt_ranking_score.py writes them: never through solve's render probe.
    report = rank_report(scenario, archive, config)
    assert report["warnings"][0] == NO_GMAT_HUNT
    assert NO_GMAT_ARCHIVE not in report["warnings"]
    for candidate in archive:
        candidate.metadata["renderable"] = True
    report = rank_report(scenario, archive, config)
    assert report["submittable_count"] == report["gmat_verified_count"] == 0
    assert report["warnings"][0] == NO_GMAT_ARCHIVE
    assert report["winner_stability"]["winner"] is None
    assert report["ceilings"]["best_submittable"] is None
    assert not any("best submittable candidate is" in w for w in report["warnings"])
    assert any("nothing here is ready" in w for w in report["warnings"])

    # GMAT ran, and failed on the one it flew: not the archive of a solve without it.
    archive[0].metadata["gmat_verification_error"] = "GMAT flew it 19.859 km off"
    report = rank_report(scenario, archive, config)
    assert report["submittable_count"] == 0
    assert NO_GMAT_ARCHIVE not in report["warnings"] and NO_GMAT_HUNT not in report["warnings"]


def test_the_preliminary_rules_still_rank_an_unflown_hit_submittable(scenario, frontier) -> None:
    """certify refuses a GMAT miss there whatever rank says."""
    report = rank_report(scenario, frontier, _cfg())
    assert report["submittable_count"] == 2 and report["gmat_verified_count"] == 0
    assert NO_GMAT_ARCHIVE not in report["warnings"]


@pytest.mark.parametrize(
    "missing", ["nothing", "run", "event", "flown script", "error after the flight"]
)
def test_a_verification_is_on_record_only_when_verify_finished_it(missing: str) -> None:
    """What verify writes last, and the error pipeline records when it raised after
    writing it (an altitude floor, a constraint)."""
    from orbitwin.constraints import gmat_verified_on_record

    candidate = _flown(_candidate(*FAST, "flown"))
    if missing == "run":
        candidate.gmat_run_id = None
    elif missing == "event":
        del candidate.metadata["gmat_event"]
    elif missing == "flown script":
        candidate.metadata["gmat_event"] = {"stop_time_s": 13208.2}
    elif missing == "error after the flight":
        candidate.metadata["gmat_verification_error"] = "flies down to 95.4 km altitude"
    assert gmat_verified_on_record(candidate) is (missing == "nothing")


def test_the_winner_stability_sweep_is_over_submittable_rows(workshop_script: Path) -> None:
    """A blocked row that wins under a nudge is not a submission that changes.

    FAST and CHEAP sit 0.0003 points apart under these (see the knife-edge test).
    """
    config = _rules_cfg("ranking", kt=0.0005, ct=12750.0, kv=0.008, cv=630.0)
    scenario, _ = parse_gmat_script(workshop_script, config)
    archive = [_candidate(*FAST, "fast"), _candidate(*CHEAP, "cheap")]
    everything = winner_stability(scenario, archive, config.score, tolerance=0.2, config=config)
    assert not everything["stable"]
    blocked = next(c for c in archive if c.candidate_id == everything["winner"])
    flown = _flown(next(c for c in archive if c is not blocked))
    report = rank_report(scenario, archive, config, tolerance=0.2)
    stability = report["winner_stability"]
    assert stability["winner"] == flown.candidate_id
    assert stability["stable"] and stability["changes"] == []
    assert not any("The winner changes" in w for w in report["warnings"])


def _searched(candidates: list[Candidate], block: object) -> list[Candidate]:
    """As solve stamps the score block its search ran under."""
    for candidate in candidates:
        candidate.metadata["searched_score"] = block
    return candidates


def _capped(cr_km: float) -> OrbitWinConfig:
    """The ranking rules with their 3 km/s total cap, so rank reports ceilings."""
    config = _rules_cfg("ranking", cr_km=cr_km)
    policy = dataclasses.replace(config.policy, total_dv_limit_km_s=3.0,
                                 total_dv_safe_limit_km_s=2.99)
    return dataclasses.replace(config, policy=policy)


def _r2_archive(cr_km: float) -> list[Candidate]:
    """R2's shape: the zero burn and a closer miss, both flown, searched at `cr_km`."""
    archive = [_flown(_miss(0.0, 4711.157, "zero")), _flown(_miss(0.2, 900.0, "closer"))]
    return _searched(archive, score_stamp(_capped(cr_km).score))


def test_rank_warns_when_the_archive_was_searched_under_other_coefficients(
    workshop_script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R2: searched under C_r 300, ranked under 3000, rank chose the zero burn at
    31.54 and said nothing; a search under 3000 found 45.82."""
    config = _capped(3000.0)
    scenario, _ = parse_gmat_script(workshop_script, config)
    report = rank_report(scenario, _r2_archive(300.0), config)
    (warning,) = [w for w in report["warnings"] if "searched under" in w]
    assert "2 of 2 candidates were searched under C_r 300 km, ranked under C_r 3000 km" in warning
    assert "optimised for the old coefficients" in warning
    assert "search again" in warning and "RUNBOOK.md step 5" in warning
    searched = report["searched_score"]
    assert searched["matches"] is False and searched["warning"] == warning
    assert searched["searched_under"] == [score_stamp(_capped(300.0).score)]
    # Marked, not refused: the ranking is still informative.
    assert report["ceilings"]["best_submittable"] is not None
    assert report["best_submittable_note"] == searched["note"]
    assert report["ceilings"]["best_submittable_note"] == searched["note"]
    assert "search again" in searched["note"]


def test_rank_is_silent_when_the_archive_was_searched_under_these_coefficients(
    workshop_script: Path,
) -> None:
    config = _capped(3000.0)
    scenario, _ = parse_gmat_script(workshop_script, config)
    # Units and integers as YAML gives them compare by value.
    archive = _searched(_r2_archive(3000.0), {**score_stamp(config.score), "cr_km": 3000})
    report = rank_report(scenario, archive, config)
    assert not any("searched" in w for w in report["warnings"])
    assert report["searched_score"]["matches"] is True
    assert report["searched_score"]["warning"] is None
    assert report["best_submittable_note"] is None
    assert report["ceilings"]["best_submittable_note"] is None


@pytest.mark.parametrize("field", ["kt", "ct", "kv", "cv", "time_unit"])
def test_any_changed_field_of_the_block_is_named(workshop_script: Path, field: str) -> None:
    config = _capped(3000.0)
    scenario, _ = parse_gmat_script(workshop_script, config)
    block = score_stamp(config.score)
    block[field] = "min" if field == "time_unit" else float(block[field]) * 2.0
    report = rank_report(scenario, _searched(_r2_archive(3000.0), block), config)
    assert report["searched_score"]["matches"] is False
    assert "C_r" not in report["searched_score"]["warning"]


def test_rank_says_an_archive_was_searched_unscored(workshop_script: Path) -> None:
    config = _capped(3000.0)
    scenario, _ = parse_gmat_script(workshop_script, config)
    report = rank_report(scenario, _searched(_r2_archive(3000.0), None), config)
    (warning,) = [w for w in report["warnings"] if "unscored" in w]
    assert "2 of 2 candidates were searched unscored, with no score block" in warning
    assert "search again" in warning
    assert report["searched_score"]["searched_under"] == [None]
    assert report["best_submittable_note"] is not None


def test_rank_says_an_archive_does_not_record_its_score_block(workshop_script: Path) -> None:
    config = _capped(3000.0)
    scenario, _ = parse_gmat_script(workshop_script, config)
    archive = _r2_archive(3000.0)
    del archive[0].metadata["searched_score"]
    report = rank_report(scenario, archive, config)
    (warning,) = [w for w in report["warnings"] if "no record of the score block" in w]
    assert warning.startswith("1 of 2 candidates carry no record")
    assert report["searched_score"]["unrecorded_count"] == 1
    assert report["best_submittable_note"] is not None


def test_the_note_is_set_when_no_row_is_submittable(workshop_script: Path) -> None:
    """RUNBOOK: note None means searched under these; so it must not be None here."""
    config = _capped(3000.0)
    scenario, _ = parse_gmat_script(workshop_script, config)
    unflown = _searched([_miss(0.0, 4711.157, "zero")], score_stamp(_capped(300.0).score))
    report = rank_report(scenario, unflown, config)
    assert report["ceilings"]["best_submittable"] is None
    assert report["best_submittable_note"] is not None


def test_an_integer_and_a_float_block_are_one_block(workshop_script: Path) -> None:
    config = _capped(300.0)
    scenario, _ = parse_gmat_script(workshop_script, config)
    archive = _r2_archive(3000.0)
    archive[0].metadata["searched_score"] = {**score_stamp(_capped(3000.0).score), "cr_km": 3000}
    report = rank_report(scenario, archive, config)
    assert report["searched_score"]["searched_under"] == [score_stamp(_capped(3000.0).score)]
    (warning,) = [w for w in report["warnings"] if "searched under" in w]
    assert warning.startswith("2 of 2 candidates were searched under C_r 3000 km")


@pytest.mark.parametrize(("relative", "differs"), [(1.0e-13, False), (2.0e-12, True)])
def test_a_block_within_1e_12_is_the_same_block(
    workshop_script: Path, relative: float, differs: bool
) -> None:
    """12 digits printed and exact equality said "kv 0.0012 ... kv 0.0012"."""
    config = _capped(3000.0)
    config = dataclasses.replace(config, score=dataclasses.replace(config.score, kv=0.0012))
    scenario, _ = parse_gmat_script(workshop_script, config)
    kv = 0.0012 * (1.0 + relative)
    report = rank_report(
        scenario, _searched(_r2_archive(3000.0), {**score_stamp(config.score), "kv": kv}), config
    )
    assert report["searched_score"]["matches"] is not differs
    if differs:
        warning = report["searched_score"]["warning"]
        assert f"searched under kv {kv!r}, ranked under kv 0.0012:" in warning


# R2 mutants R02, R04, R05, R11, R12, R13 (2026-11-06): rank's gates and advice.


def _gated(*burns: tuple[float, float], branch: str = "gated") -> Candidate:
    """A verified hit flying ``burns``, (time s, km/s) along x."""
    return _flown(Candidate(
        burns=tuple(
            Burn(time_s=t, dv_inertial_km_s=(dv, 0.0, 0.0), dv_script_km_s=(dv, 0.0, 0.0),
                 slot_index=index)
            for index, (t, dv) in enumerate(burns)
        ),
        intercept_time_s=13208.2, minimum_distance_km=1.0, distance_at_completion_km=5.0,
        successful=True, analytical_endpoint_error_km=None, branch=branch,
        metadata={"event_refined": True},
    ))


@pytest.mark.parametrize(
    "burns",
    [((100.0, 0.2495),), ((100.0, 0.1), (200.5, 0.1)),
     tuple((100.0 + 200.0 * k, 0.2305) for k in range(13))],
    ids=["impulse-past-249-m-s", "gap-under-101-s", "total-past-2.99-km-s"],
)
def test_a_formal_frontier_row_is_not_submittable(workshop_script: Path, burns) -> None:
    scenario, config = _ranking_scenario(workshop_script)
    formal, robust = _gated(*burns), _gated((100.0, 0.1), branch="robust")
    report = rank_report(scenario, [formal, robust], config)
    row = next(r for r in report["ranking"] if r["candidate_id"] == formal.candidate_id)
    assert row["blockers"] == [
        "on the formal frontier, which policy.allow_formal_frontier_submission withholds"]
    assert not row["submittable"]
    assert report["ceilings"]["best_submittable"] == robust.candidate_id
    opted = dataclasses.replace(config, policy=dataclasses.replace(
        config.policy, allow_formal_frontier_submission=True))
    assert rank_report(scenario, [formal], opted)["winner"]["submittable"]


@pytest.mark.parametrize(
    ("burns", "reason"),
    [(((100.0, 0.2501),), "exceeds formal limit 0.25 km/s"),
     (tuple((100.0 + 200.0 * k, 0.2401) for k in range(13)), "exceeds the 3 km/s limit")],
    ids=["impulse", "total"],
)
def test_a_disqualifying_row_is_blocked_even_with_the_formal_frontier_allowed(
    workshop_script: Path, burns, reason: str
) -> None:
    """§4 notes 1(a) and 1(c) disqualify rather than deduct, so no penalty names it."""
    scenario, config = _ranking_scenario(workshop_script)
    config = dataclasses.replace(config, policy=dataclasses.replace(
        config.policy, allow_formal_frontier_submission=True))
    (blocker,) = rank_report(scenario, [_gated(*burns)], config)["winner"]["blockers"]
    assert reason in blocker


@pytest.mark.parametrize("allowed", [False, True])
def test_a_row_through_the_central_body_is_blocked_unless_allowed(
    workshop_script: Path, allowed: bool
) -> None:
    """With policy.minimum_flown_altitude_km unset, as outside the ranking profiles."""
    scenario, config = _ranking_scenario(workshop_script)
    assert config.policy.minimum_flown_altitude_km is None
    config = dataclasses.replace(config, policy=dataclasses.replace(
        config.policy, allow_body_intersecting_submission=allowed))
    crossing = _gated((100.0, 0.1))
    crossing.metadata["minimum_pursuer_radius_km"] = config.policy.central_body_radius_km - 1.0
    blockers = rank_report(scenario, [crossing], config)["winner"]["blockers"]
    assert blockers == ([] if allowed else ["reaches the interception through the central body"])


def test_the_ceilings_are_on_the_solver_clock_after_an_alignment(workshop_script: Path) -> None:
    """T_team starts 1800 s before solver t = 0, so Ct is 1800 s nearer on its clock."""
    from orbitwin.models import clock_stamps
    from orbitwin.score import ceilings, on_solver_clock

    scenario, config = _ranking_scenario(workshop_script)
    config = dataclasses.replace(config, score=_score_config(cr_km=100.0, ct=2000.0))
    aligned = dataclasses.replace(scenario, time_origin_offset_s=1800.0)
    row = _gated((100.0, 0.1))
    row.metadata.update(clock_stamps(aligned))
    shown = rank_report(aligned, [row], config)["ceilings"]
    assert config.score is not None
    cap = config.policy.total_dv_limit_km_s
    assert cap is not None
    expected = ceilings(on_solver_clock(config.score, aligned), aligned.max_time_s, cap)
    unshifted = ceilings(config.score, aligned.max_time_s, cap)
    assert shown["any_answer"] == pytest.approx(expected.any_answer, rel=1e-12)
    assert shown["any_miss"] == pytest.approx(expected.any_miss, rel=1e-12)
    assert abs(expected.any_answer - unshifted.any_answer) > 1.0


def test_the_levers_are_priced_at_the_safe_limits_not_the_rules(workshop_script: Path) -> None:
    scenario, config = _ranking_scenario(workshop_script)
    row = _gated((100.0, 0.2), (300.0, 0.2))
    spent = rank_report(scenario, [row], config)["levers"]["spent"]
    assert spent["one_more_impulse"]["km_s"] == 0.249
    assert spent["remaining_budget"]["km_s"] == pytest.approx(2.99 - 0.4, abs=1e-12)


def test_a_row_shows_the_distance_its_tie_break_reads(workshop_script: Path) -> None:
    from orbitwin.models import clock_stamps

    scenario, config = _ranking_scenario(workshop_script)
    windowed = dataclasses.replace(scenario, alignment_minimum_distance_km=500.0)
    miss = _flown(_miss(0.1, 900.0, "far"))
    miss.metadata.update(clock_stamps(windowed))
    row = rank_report(windowed, [miss], config)["winner"]
    assert (row["measured_minimum_distance_km"], row["tie_break_distance_km"]) == (900.0, 500.0)


def test_rank_breaks_a_tie_over_the_scored_window(workshop_script: Path) -> None:
    """Both misses score on the alignment's 500 km, so they tie and delta-v, then the id, decide."""
    from orbitwin.models import clock_stamps

    scenario, config = _ranking_scenario(workshop_script)
    windowed = dataclasses.replace(scenario, alignment_minimum_distance_km=500.0)
    first, last = sorted((_miss(0.1, 0.0, name) for name in ("a", "b")),
                         key=lambda c: c.candidate_id)
    # The one last by id passes nearer on the joint rows.
    first.minimum_distance_km, last.minimum_distance_km = 900.0, 600.0
    for miss in (first, last):
        _flown(miss).metadata.update(clock_stamps(windowed))
    ranking = rank_report(windowed, [last, first], config)["ranking"]
    assert [row["candidate_id"] for row in ranking] == [first.candidate_id, last.candidate_id]
    assert ranking[0]["score"] == ranking[1]["score"]


def test_rank_compares_a_failed_verification_on_the_score_solve_chose_it_by(
    workshop_script: Path,
) -> None:
    """Read before rank re-scores the archive: verify died before it recorded a score,
    so the file's own score is the one it was chosen by."""
    scenario, config = _ranking_scenario(workshop_script)
    best = _flown(_candidate(13208.2, 0.20, "best"))
    best.metadata["analytical_before_gmat"] = {"successful": True, "score": 60.0}
    lost = _miss(0.1, 5000.0, "lost")
    lost.score = 99.0
    lost.metadata.update(analytical_before_gmat={"successful": True},
                         gmat_verification_error="GmatConsole timed out after 300 seconds",
                         gmat_verification_error_type="GmatRunError")
    report = rank_report(scenario, [best, lost], config)
    assert report["ceilings"]["best_submittable"] == best.candidate_id
    row = next(r for r in report["ranking"] if r["candidate_id"] == lost.candidate_id)
    assert row["score"] < 60.0
    (run,) = report["gmat_failures"]["runs"]
    assert (run["pre_gmat_score"], run["above_best"]) == (99.0, True)
    warning = report["gmat_failures"]["warning"]
    assert warning in report["warnings"]
    assert f"ABOVE THE BEST: verification of {lost.candidate_id} (lost, pre-GMAT score 99)" in (
        warning)
