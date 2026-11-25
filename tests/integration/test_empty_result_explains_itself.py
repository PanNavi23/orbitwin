from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from orbitwin import pipeline as pipeline_module
from orbitwin.config import OrbitWinConfig, SearchConfig
from orbitwin.errors import PolicyError
from orbitwin.models import Burn, Candidate, CandidateStage
from orbitwin.pipeline import solve_scenario

# An empty result with an empty explanation is the one outcome an operator
# cannot act on, and the day is short.


def _config(**policy_overrides) -> OrbitWinConfig:
    base = OrbitWinConfig(
        search=SearchConfig(
            one_burn_samples=0, one_burn_grid_burns=1, one_burn_grid_arrivals=1,
            max_revolutions=0, two_burn_enabled=False, event_refine_top=2,
            keep_candidates=8, workers=1,
        )
    )
    if not policy_overrides:
        return base
    return dataclasses.replace(base, policy=dataclasses.replace(base.policy, **policy_overrides))


def _candidate(branch: str, **overrides) -> Candidate:
    fields: dict = {
        "burns": (Burn(1000.0, (0.2, 0.0, 0.0), (0.2, 0.0, 0.0), 0),),
        "intercept_time_s": 4000.0,
        "minimum_distance_km": 0.2,
        "distance_at_completion_km": 5.0,
        "successful": True,
        "analytical_endpoint_error_km": 0.1,
        "branch": branch,
        "metadata": {"arrival_time_s": 4000.0, "event_refined": True},
    }
    fields.update(overrides)
    return Candidate(**fields)


def _only(monkeypatch, *candidates: Candidate) -> None:
    monkeypatch.setattr(pipeline_module, "zero_burn_candidate", lambda *_: candidates[0])
    monkeypatch.setattr(pipeline_module, "search_one_burn", lambda *_: list(candidates[1:]))
    monkeypatch.setattr(pipeline_module, "search_two_burn_global", lambda *_: [])
    monkeypatch.setattr(pipeline_module, "slot_assignment_variants", lambda *_: [])


def test_a_candidate_held_back_for_the_earth_is_not_blamed_on_the_frontier(
    tmp_path: Path, workshop_script: Path, monkeypatch
) -> None:
    """The reason has to be the reason.

    A candidate inside the production margins but flying through the planet is
    refused by the same gate as one on the formal frontier. Reported as the
    latter, it sends an operator to set allow_formal_frontier_submission, which
    will not release it -- worse than saying nothing, because it looks like an
    answer.
    """
    underground = _candidate(
        "robust-but-underground",
        metadata={
            "arrival_time_s": 4000.0,
            "event_refined": True,
            "minimum_pursuer_radius_km": 5807.8,
        },
    )
    _only(monkeypatch, underground)
    summary = solve_scenario(workshop_script, _config(), tmp_path / "solve")

    assert summary["best_candidate"] is None
    (warning,) = summary["warnings"]
    assert "central body" in warning
    assert "5807.8" in warning
    assert "allow_body_intersecting_submission" in warning
    assert "formal frontier" not in warning
    assert "allow_formal_frontier_submission" not in warning


def test_a_candidate_on_the_formal_frontier_still_says_so(
    tmp_path: Path, workshop_script: Path, monkeypatch
) -> None:
    formal = _candidate(
        "formal-frontier",
        burns=(Burn(1000.0, (1.4999, 0.0, 0.0), (1.4999, 0.0, 0.0), 0),),
    )
    _only(monkeypatch, formal)
    summary = solve_scenario(workshop_script, _config(), tmp_path / "solve")

    assert summary["best_candidate"] is None
    (warning,) = summary["warnings"]
    assert "formal frontier" in warning
    assert "allow_formal_frontier_submission" in warning
    assert "central body" not in warning


def test_both_reasons_are_reported_when_both_apply(
    tmp_path: Path, workshop_script: Path, monkeypatch
) -> None:
    formal = _candidate(
        "formal-frontier", burns=(Burn(1000.0, (1.4999, 0.0, 0.0), (1.4999, 0.0, 0.0), 0),)
    )
    underground = _candidate(
        "robust-but-underground",
        metadata={
            "arrival_time_s": 4000.0, "event_refined": True,
            "minimum_pursuer_radius_km": 6000.0,
        },
    )
    _only(monkeypatch, formal, underground)
    summary = solve_scenario(workshop_script, _config(), tmp_path / "solve")

    assert summary["best_candidate"] is None
    joined = " ".join(summary["warnings"])
    assert "formal frontier" in joined
    assert "central body" in joined


def test_nothing_closing_is_explained_with_how_close_it_got(
    tmp_path: Path, workshop_script: Path, monkeypatch
) -> None:
    missed = _candidate(
        "missed", intercept_time_s=None, successful=False,
        minimum_distance_km=3663.2, distance_at_completion_km=4000.0,
    )
    _only(monkeypatch, missed)
    summary = solve_scenario(workshop_script, _config(), tmp_path / "solve")

    assert summary["best_candidate"] is None
    (warning,) = summary["warnings"]
    assert "No candidate reached" in warning
    assert "3663.2" in warning, "the closest approach is what says whether to widen or give up"


def test_an_unevaluated_interception_is_named_as_one(
    tmp_path: Path, workshop_script: Path, monkeypatch
) -> None:
    # successful is the Lambert endpoint residual until refinement replaces it,
    # around 1e-10 km for any converged arc. Reporting those as solutions would
    # advertise interceptions nothing has demonstrated.
    unrefined = _candidate("unrefined", metadata={"arrival_time_s": 4000.0})
    _only(monkeypatch, unrefined)
    summary = solve_scenario(workshop_script, _config(), tmp_path / "solve")

    assert summary["best_candidate"] is None
    (warning,) = summary["warnings"]
    assert "Lambert endpoint residual" in warning
    assert "event_refine_top" in warning


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: _candidate("missed", intercept_time_s=None, successful=False,
                                        minimum_distance_km=900.0), id="nothing-closed"),
        pytest.param(lambda: _candidate("unrefined", metadata={"arrival_time_s": 4000.0}),
                     id="unevaluated"),
        pytest.param(lambda: _candidate("formal", burns=(Burn(1000.0, (1.4999, 0.0, 0.0),
                                                             (1.4999, 0.0, 0.0), 0),)),
                     id="formal-frontier"),
        pytest.param(lambda: _candidate("underground", metadata={
            "arrival_time_s": 4000.0, "event_refined": True,
            "minimum_pursuer_radius_km": 5807.8}), id="central-body"),
    ],
)
def test_an_empty_result_is_never_silent(
    tmp_path: Path, workshop_script: Path, monkeypatch, make
) -> None:
    _only(monkeypatch, make())
    summary = solve_scenario(workshop_script, _config(), tmp_path / "solve")
    assert summary["best_candidate"] is None
    assert summary["warnings"], "an empty result that explains nothing cannot be acted on"


def test_solve_refuses_to_emit_the_submission_name(
    tmp_path: Path, workshop_script: Path, monkeypatch
) -> None:
    # The comment beside best_script_name states this as a property. Nothing in
    # solve promotes that far today, so reaching it means a candidate arrived
    # carrying the stage, and the name would assert replays this run never made.
    arrived_certified = _candidate("already-certified")
    arrived_certified.promote(CandidateStage.PUBLIC_GMAT_VERIFIED, run_id="elsewhere")
    arrived_certified.promote(CandidateStage.ORGANIZER_GMAT_VERIFIED, run_id="elsewhere")
    arrived_certified.promote(CandidateStage.CERTIFIED, run_id="elsewhere")
    assert arrived_certified.stage is CandidateStage.CERTIFIED
    _only(monkeypatch, arrived_certified)
    with pytest.raises(PolicyError, match="only certify may emit"):
        solve_scenario(workshop_script, _config(), tmp_path / "solve")


@pytest.mark.parametrize(
    ("raised", "expected", "not_expected"),
    [
        pytest.param(
            "GmatRunError", "GMAT rejected this candidate", "could not be completed",
            id="gmat-said-no",
        ),
        pytest.param(
            "PolicyError", "verification could not be completed (PolicyError)",
            "GMAT rejected", id="our-side-said-no",
        ),
        pytest.param(
            "TypeError", "verification could not be completed (TypeError)",
            "GMAT rejected", id="a-refactor-said-no",
        ),
        pytest.param(
            "AltitudeFloorError", "GMAT flew it under the altitude floor",
            "could not be completed", id="the-altitude-floor-said-no",
        ),
        pytest.param(
            "ConstraintVerdictError", "GMAT flew it, and its flight breaks a constraint",
            "could not be completed", id="a-constraint-said-no",
        ),
        pytest.param(
            "StopVerdictError", "the script stopped at its hit does not end inside",
            "GMAT rejected", id="the-stop-said-no",
        ),
    ],
)
def test_a_failed_verification_says_whose_failure_it_was(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
    monkeypatch,
    raised,
    expected,
    not_expected,
) -> None:
    """Catching everything is right; blaming everything on GMAT is not.

    Nothing may be promoted on a run that did not finish, so the handler takes
    any exception. It then said "GMAT verification failed" whatever failed -- a
    refused output directory, a constraint this candidate breaks, a TypeError
    from a refactor -- and sent the reader to the wrong logs.
    """
    from orbitwin.errors import (
        AltitudeFloorError,
        ConstraintVerdictError,
        GmatRunError,
        StopVerdictError,
    )

    exceptions = {
        "GmatRunError": GmatRunError, "PolicyError": PolicyError, "TypeError": TypeError,
        "AltitudeFloorError": AltitudeFloorError, "ConstraintVerdictError": ConstraintVerdictError,
        "StopVerdictError": StopVerdictError,
    }

    def explode(*_args, **_kwargs):
        raise exceptions[raised]("something went wrong")

    _only(monkeypatch, _candidate("verifiable"))
    monkeypatch.setattr(pipeline_module, "verify_candidate_with_gmat", explode)
    summary = solve_scenario(
        workshop_script, _config(), tmp_path / "solve",
        gmat_console=fake_gmat, startup_file=fake_startup, gmat_verify_top=1,
    )

    assert summary["gmat_verified_count"] == 0
    assert summary["best_candidate"] is None
    records = json.loads((tmp_path / "solve" / "candidates.json").read_text(encoding="utf-8"))
    attempted = [c for c in records if "gmat_verification_error" in c["metadata"]]
    assert attempted, "the attempt should be recorded"
    joined = " ".join(warning for c in attempted for warning in c["warnings"])
    assert expected in joined, joined
    assert not_expected not in joined, joined
    assert attempted[0]["metadata"]["gmat_verification_error_type"] == raised
