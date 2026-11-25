"""What solve and rank say about GMAT runs that raised (R2 H1, 2026-11-06)."""

from __future__ import annotations

import math

from orbitwin.gmat_failures import (
    SHOWN_ERROR_CHARS,
    gmat_failures,
    pre_gmat_score,
    verification_error,
)
from orbitwin.models import Burn, Candidate


def _candidate(branch: str, score: float | None, **metadata: object) -> Candidate:
    burn = Burn(time_s=100.0, dv_inertial_km_s=(0.1, 0.0, 0.0), dv_script_km_s=(0.1, 0.0, 0.0))
    candidate = Candidate(
        burns=(burn,) if branch != "zero-burn" else (), intercept_time_s=None,
        minimum_distance_km=10.0, distance_at_completion_km=10.0, successful=False,
        analytical_endpoint_error_km=None, branch=branch, metadata=dict(metadata),
    )
    candidate.score = score
    return candidate


def test_the_pre_gmat_score_is_the_one_recorded_before_the_flight() -> None:
    flown = _candidate("a", 97.9, analytical_before_gmat={"successful": True, "score": 25.0})
    assert pre_gmat_score(flown) == 25.0
    unscored = _candidate("b", 50.0, analytical_before_gmat={"score": None})
    assert pre_gmat_score(unscored) is None
    # Recorded by solve when verify died before it could: the score was never touched.
    early = _candidate("c", 61.5, analytical_before_gmat={"successful": True})
    assert pre_gmat_score(early) == 61.5


def test_a_verification_error_carries_its_type() -> None:
    failed = _candidate("a", 1.0, gmat_verification_error="GmatConsole timed out after 300 seconds",
                        gmat_verification_error_type="GmatRunError")
    assert verification_error(failed) == "GmatRunError: GmatConsole timed out after 300 seconds"
    assert verification_error(_candidate("b", 1.0)) is None


def test_failures_are_ordered_by_pre_gmat_score_and_marked_against_the_best() -> None:
    best = _candidate("best", 97.3, analytical_before_gmat={"score": 96.7})
    above = _candidate("above", None, analytical_before_gmat={"score": 97.0})
    below = _candidate("below", 25.0)
    unscored = _candidate("unscored", None)
    report = gmat_failures(
        [("verification", below, "GmatRunError: x"), ("correction", unscored, "OSError: y"),
         ("correction", above, "round 1 (flight) failed: GmatRunError: z")],
        best, archived={c.candidate_id for c in (best, above, below, unscored)}, who="the best",
    )
    assert [(r["candidate_id"], r["above_best"]) for r in report["runs"]] == [
        (above.candidate_id, True), (below.candidate_id, False), (unscored.candidate_id, False)]
    warning = report["warning"]
    assert warning.startswith("3 GMAT correction(s) or verification(s) raised")
    assert f"the best is {best.candidate_id} (pre-GMAT score 96.7, now 97.3), and 1 of them" in (
        warning)
    assert f": ABOVE THE BEST: correction of {above.candidate_id} (above, pre-GMAT score 97)" in (
        warning)
    assert f"; correction of {unscored.candidate_id} (unscored, pre-GMAT score unscored)" in (
        warning)


def test_without_a_best_every_failure_ranks_above_it() -> None:
    lost = _candidate("lost", 10.0)
    report = gmat_failures([("verification", lost, "GmatRunError: x")], None,
                           archived=set(), who="the chosen best")
    (run,) = report["runs"]
    assert run["above_best"] and not run["in_archive"]
    assert "the chosen best is none, so every one of them ranks above it" in report["warning"]
    assert "pre-GMAT score 10, not in candidates.json): GmatRunError: x" in report["warning"]


def test_only_above_says_nothing_of_the_rest_but_keeps_them() -> None:
    best = _candidate("best", 90.0)
    below = _candidate("below", 80.0)
    report = gmat_failures([("verification", below, "AltitudeFloorError: 95 km")], best,
                           archived={below.candidate_id}, who="the best", only_above=True)
    assert report["warning"] is None
    assert [r["candidate_id"] for r in report["runs"]] == [below.candidate_id]


def test_a_long_error_is_cut_in_the_warning_only() -> None:
    long = "GmatRunError: " + "x" * 1000
    lost = _candidate("lost", 1.0)
    report = gmat_failures([("verification", lost, long)], None, archived=set(), who="it")
    assert report["runs"][0]["error"] == long
    assert long[:SHOWN_ERROR_CHARS] in report["warning"]
    assert long[: SHOWN_ERROR_CHARS + 1] not in report["warning"]


def test_a_failure_level_with_the_best_is_not_above_it() -> None:
    best = _candidate("best", 90.0)
    level = _candidate("level", 90.0)
    report = gmat_failures([("verification", level, "GmatRunError: x")], best,
                           archived={level.candidate_id}, who="the best", only_above=True)
    assert report["runs"][0]["above_best"] is False
    assert report["warning"] is None


def test_a_verdict_of_gmat_is_no_failure() -> None:
    """R2 #187 review F1: pipeline tags what verify raised as GMAT's answer."""
    floor = _candidate("floor", 1.0, gmat_verification_error="down to 95.4 km",
                       gmat_verification_error_type="AltitudeFloorError",
                       gmat_verification_verdict=True)
    crash = _candidate("crash", 1.0, gmat_verification_error="exit status 3",
                       gmat_verification_error_type="GmatRunError",
                       gmat_verification_verdict=False)
    assert verification_error(floor) is None
    assert verification_error(crash) == "GmatRunError: exit status 3"


def test_a_report_must_reach_the_scripts_end() -> None:
    """R2 #187 review F2: a report cut short with exit 0 was taken as a verified miss.
    Its delta review: one row missing passed a tolerance of one 60 s step."""
    import pytest

    from orbitwin.diagnostic import DIAGNOSTIC_PRECISION
    from orbitwin.errors import GmatRunError
    from orbitwin.event_detection import EventResult
    from orbitwin.verify import report_reaches_mission_end

    def ending(end_s: float, step_s: float) -> EventResult:
        return EventResult(None, 10.0, 0.0, 10.0, end_time_s=end_s, report_step_s=step_s)

    from orbitwin.verify import REPORT_END_TOLERANCE_S

    end_s, step_s = 30000.0, 60.0
    slack_s = REPORT_END_TOLERANCE_S
    assert slack_s == 1.0e-3
    # The measured R2026a shortfalls reached 2.5e-6 s, and a row ending past the end passes.
    for short_s in (0.0, -1.2e-6, 2.5e-6, 0.999 * slack_s):
        report_reaches_mission_end(ending(end_s - short_s, step_s), end_s, "r.txt")
    # One 60 s row missing, the delta review's 5.36 s at-stop case, and just past the slack.
    for short_s in (step_s, 5.36, 1.001 * slack_s):
        with pytest.raises(GmatRunError, match="cut short"):
            report_reaches_mission_end(ending(end_s - short_s, step_s), end_s, "r.txt")
    # At 17 digits the time column rounds below a double's own step; at 6, by 0.3 s here.
    assert 10.0 ** (1 - DIAGNOSTIC_PRECISION) * end_s < 2.0 * math.ulp(end_s)
    rounding_s = 1.0e-5 * end_s
    short = ending(end_s - slack_s - 0.5 * rounding_s, step_s)
    report_reaches_mission_end(short, end_s, "r.txt", precision=6)
    with pytest.raises(GmatRunError, match="cut short"):
        report_reaches_mission_end(short, end_s, "r.txt")
    with pytest.raises(GmatRunError, match="cut short"):
        report_reaches_mission_end(ending(end_s - slack_s - 1.5 * rounding_s, step_s), end_s,
                                   "r.txt", precision=6)
