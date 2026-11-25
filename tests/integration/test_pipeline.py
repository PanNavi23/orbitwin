from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

# quick_config now lives in tests/conftest.py; see the note there.
from conftest import quick_config, stamped, startup_for_console_elsewhere

from orbitwin.config import OrbitWinConfig, ScoreConfig
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import file_sha256
from orbitwin.pipeline import solve_scenario
from orbitwin.rules import rule_set_record
from orbitwin.score import score_stamp
from orbitwin.search import search_one_burn
from orbitwin.verify import certify_candidate, verify_candidate_with_gmat


def test_workshop_search_recovers_hohmann_intercept(workshop_script: Path) -> None:
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidates = search_one_burn(scenario, config)
    assert candidates
    best = min(candidates, key=lambda candidate: candidate.total_dv_km_s)
    assert best.total_dv_km_s < 0.61
    assert best.burns[0].time_s == 1000.0
    assert best.intercept_time_s is not None and best.intercept_time_s < 4725.0  # entry 4719.4 s, arrival 4728.5 s


def test_solve_writes_reproducible_artifacts(tmp_path: Path, workshop_script: Path) -> None:
    summary = solve_scenario(workshop_script, quick_config(), tmp_path / "solve", render_top=2)
    assert summary["successful_count"] >= 1
    assert summary["rule_set"] == rule_set_record("preliminary")
    assert (tmp_path / "solve" / "candidates.json").is_file()
    # `rank` refuses an archive solved under another horizon only if it is stamped.
    scenario = parse_gmat_script(workshop_script, quick_config())[0]
    archived = json.loads((tmp_path / "solve" / "candidates.json").read_text(encoding="utf-8"))
    rows = archived["candidates"] if isinstance(archived, dict) else archived
    assert rows and all(row["metadata"]["horizon_s"] == scenario.max_time_s for row in rows)
    assert all(row["metadata"]["scenario_sha256"] == scenario.source_sha256 for row in rows)
    # No GMAT ran, so the artifact must not carry the submission name.
    best_dir = tmp_path / "solve" / "best_candidate"
    assert (best_dir / "analytical_candidate.script").is_file()
    assert not (best_dir / "team_solution.script").exists()
    assert summary["gmat_verification_attempted"] is False
    assert summary["best_candidate_is_gmat_verified"] is False


@pytest.mark.parametrize("scored", [False, True])
def test_solve_records_the_score_block_it_searched_under(
    tmp_path: Path, workshop_script: Path, scored: bool
) -> None:
    """rank compares it with the block it ranks under (R2: C_r 300 ranked at 3000)."""
    config = quick_config()
    if scored:
        config = dataclasses.replace(
            config, score=ScoreConfig(kt=0.001, ct=20000.0, kv=4.0, cv=0.8, delta_v_unit="km/s")
        )
    summary = solve_scenario(workshop_script, config, tmp_path / "solve", render_top=1)
    expected = score_stamp(config.score)
    assert (expected is None) is not scored
    assert summary["searched_score"] == expected
    rows = json.loads((tmp_path / "solve" / "candidates.json").read_text(encoding="utf-8"))
    assert rows and all(row["metadata"]["searched_score"] == expected for row in rows)


def test_fake_gmat_diagnostic_promotes_candidate(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
) -> None:
    config = quick_config()
    scenario, document = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    verified, result, evidence = verify_candidate_with_gmat(
        workshop_script,
        scenario,
        document,
        candidate,
        config,
        fake_gmat,
        fake_startup,
        tmp_path / "verify",
    )
    assert result.success
    assert verified.verified_by_gmat
    assert evidence["event"]["first_entry_time_s"] is not None


def test_fake_certification_repeats_exact_script(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
) -> None:
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    stamped(candidate, workshop_script, config)
    evidence = certify_candidate(
        workshop_script,
        candidate,
        config,
        fake_gmat,
        fake_startup,
        tmp_path / "certify",
        runs=3,
    )
    assert len(evidence["exact_runs"]) == 3
    submission = tmp_path / "certify" / "submission"
    # The test double is not an organizer-compatible build, so the artifact must
    # not claim the submission name.
    assert evidence["stage"] == "public_gmat_verified"
    assert (submission / "public_gmat_candidate.script").is_file()
    assert not (submission / "team_solution.script").exists()
    # What certify names, the script to carry and the reference report, is what it wrote.
    from orbitwin.cli import upload_summary

    summary = upload_summary(evidence, tmp_path / "certify")
    listed = [line.strip() for line in summary if line.startswith("  ")]
    assert listed == [
        str(submission / "public_gmat_candidate.script"), str(submission / "ReportFile1.txt")
    ]
    assert (submission / "ReportFile1.txt").is_file()
    certification = json.loads((submission / "certification.json").read_text(encoding="utf-8"))
    assert certification["final_script_sha256"]
    assert certification["rule_set"] == rule_set_record("preliminary")
    # Interception must be demonstrated against the submitted bytes, not only
    # against the earlier sibling render.
    parity = certification["final_script_parity"]
    assert parity["first_entry_time_s"] is not None


def test_pipeline_skips_unrenderable_candidate_without_losing_legal_fallback(
    tmp_path: Path,
    workshop_script: Path,
    monkeypatch,
) -> None:
    import orbitwin.pipeline as pipeline_module
    from orbitwin.models import Burn, Candidate

    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    late = scenario.max_time_s - 10.0
    unrenderable = Candidate(
        burns=(Burn(late, (0.01, 0.0, 0.0), (0.01, 0.0, 0.0), 0),),
        intercept_time_s=scenario.max_time_s - 5.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="unrenderable-late",
        metadata={"event_refined": True},
    )
    legal = Candidate(
        burns=(
            Burn(100.0, (0.2, 0.0, 0.0), (0.2, 0.0, 0.0), 0),
            Burn(250.0, (0.2, 0.0, 0.0), (0.2, 0.0, 0.0), 1),
        ),
        intercept_time_s=500.0,
        minimum_distance_km=0.2,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.2,
        branch="legal-two-burn",
        metadata={"event_refined": True},
    )
    monkeypatch.setattr(pipeline_module, "zero_burn_candidate", lambda *_: unrenderable)
    monkeypatch.setattr(pipeline_module, "search_one_burn", lambda *_: [legal])
    monkeypatch.setattr(pipeline_module, "search_two_burn_global", lambda *_: [])

    summary = solve_scenario(workshop_script, config, tmp_path / "solve", render_top=2)
    assert summary["renderable_count"] == 1
    assert summary["best_candidate"]["candidate_id"] == legal.candidate_id
    assert (tmp_path / "solve" / "best_candidate" / "analytical_candidate.script").is_file()


def scoring_config() -> OrbitWinConfig:
    base = quick_config()
    return OrbitWinConfig(
        search=base.search,
        policy=base.policy,
        gmat=base.gmat,
        score=ScoreConfig(kt=0.001, ct=5000.0, kv=0.004, cv=800.0, delta_v_unit="m/s"),
    )


def test_gmat_verification_rescores_the_candidate(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
) -> None:
    # GMAT owns intercept time and completion distance, and the official score
    # reads both. A candidate that keeps its analytical score after GMAT moved
    # the intercept is ranked on a state it no longer has.
    config = scoring_config()
    scenario, document = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    before_time = candidate.intercept_time_s
    before_score = candidate.score
    assert before_score is not None

    verified, _, _ = verify_candidate_with_gmat(
        workshop_script,
        scenario,
        document,
        candidate,
        config,
        fake_gmat,
        fake_startup,
        tmp_path / "verify",
    )

    assert verified.intercept_time_s != before_time
    assert verified.score != before_score
    assert verified.score_components["time_score"] > 0.0


def test_gmat_process_failure_never_promotes_the_candidate(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
    monkeypatch,
) -> None:
    # A GMAT run that dies before it can update the candidate leaves the
    # analytical success flag untouched. Selection reads that flag, so without
    # an explicit reset the top analytical candidate is emitted under the
    # submission name having never been replayed at all.
    import orbitwin.pipeline as pipeline_module
    from orbitwin.errors import GmatRunError

    def explode(*_args, **_kwargs):
        raise GmatRunError("GmatConsole returned exit status 1")

    monkeypatch.setattr(pipeline_module, "verify_candidate_with_gmat", explode)
    summary = solve_scenario(
        workshop_script,
        quick_config(),
        tmp_path / "solve",
        render_top=2,
        # Real enough to answer the state probe; verification is monkeypatched.
        gmat_console=fake_gmat,
        startup_file=fake_startup,
        gmat_verify_top=3,
    )

    assert summary["gmat_verification_attempted"] is True
    assert summary["gmat_verified_count"] == 0
    assert summary["best_candidate"] is None
    assert not (tmp_path / "solve" / "best_candidate" / "team_solution.script").exists()

    candidates = json.loads((tmp_path / "solve" / "candidates.json").read_text(encoding="utf-8"))
    attempted = [c for c in candidates if "gmat_verification_error" in c["metadata"]]
    assert attempted, "the verification attempt should be recorded"
    for candidate in attempted:
        assert candidate["successful"] is False
        assert candidate["verified_by_gmat"] is False
        # The analytical verdict is preserved rather than destroyed.
        assert candidate["metadata"]["analytical_successful"] is True


def test_selection_follows_gmat_not_the_analytical_order(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
    monkeypatch,
) -> None:
    # The first-ranked candidate is rejected by GMAT and the second is accepted.
    # Selection must follow GMAT, not the analytical order.
    import orbitwin.pipeline as pipeline_module
    from orbitwin.errors import PolicyError
    from orbitwin.models import CandidateStage

    seen: list[str] = []

    def verify(_script, _scenario, _document, candidate, *_args, **_kwargs):
        seen.append(candidate.candidate_id)
        if len(seen) == 1:
            candidate.intercept_time_s = None
            candidate.successful = False
            raise PolicyError("GMAT candidate failed constraints: no entry")
        candidate.successful = True
        candidate.promote(CandidateStage.PUBLIC_GMAT_VERIFIED, run_id="stub")
        return candidate, None, {}

    monkeypatch.setattr(pipeline_module, "verify_candidate_with_gmat", verify)
    summary = solve_scenario(
        workshop_script,
        quick_config(),
        tmp_path / "solve",
        render_top=2,
        # Real enough to answer the state probe; verification is monkeypatched.
        gmat_console=fake_gmat,
        startup_file=fake_startup,
        gmat_verify_top=2,
    )

    assert len(seen) >= 2
    best = summary["best_candidate"]
    assert best is not None
    assert best["candidate_id"] != seen[0]
    assert best["verified_by_gmat"] is True
    # solve() never certifies, so it can never produce the submission name.
    assert summary["best_candidate_script"] == "public_gmat_candidate.script"
    assert summary["best_candidate_stage"] == "public_gmat_verified"
    best_dir = tmp_path / "solve" / "best_candidate"
    assert (best_dir / "public_gmat_candidate.script").is_file()
    assert not (best_dir / "team_solution.script").exists()


def test_certification_rejects_a_perturbing_diagnostic_sibling(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
    monkeypatch,
) -> None:
    # If attaching the diagnostic ReportFile changed the official report, then
    # first entry was proven on a script that does not propagate like the one
    # being submitted, and the certification is worthless.
    import orbitwin.verify as verify_module
    from orbitwin.errors import GmatRunError

    real_replay = verify_module._replay_with_diagnostic

    def perturbed(*args, **kwargs):
        event, result = real_replay(*args, **kwargs)
        tampered = dict(result.report_sha256)
        for name in tampered:
            tampered[name] = "0" * 64
        return event, dataclasses.replace(result, report_sha256=tampered)

    monkeypatch.setattr(verify_module, "_replay_with_diagnostic", perturbed)
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    stamped(candidate, workshop_script, config)
    with pytest.raises(GmatRunError, match="changed official report"):
        certify_candidate(
            workshop_script,
            candidate,
            config,
            fake_gmat,
            fake_startup,
            tmp_path / "certify",
            runs=1,
        )


def test_only_an_organizer_build_produces_team_solution_script(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
) -> None:
    # Release status meaning (ASSUMPTIONS_AND_GAPS.md) ranks the organizer's GMAT above any build we installed
    # ourselves. The submission name follows the declaration, not the run.
    import dataclasses as dc

    from orbitwin.models import CandidateStage

    base = quick_config()
    config = dc.replace(
        base,
        gmat=dc.replace(
            base.gmat,
            organizer_compatible=True,
            # The declaration alone is not accepted -- see
            # test_declaring_the_organizer_build_requires_naming_the_binary.
            organizer_executable_sha256=file_sha256(fake_gmat),
        ),
    )
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    stamped(candidate, workshop_script, config)
    evidence = certify_candidate(
        workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "certify", runs=1
    )
    assert evidence["stage"] == CandidateStage.CERTIFIED.value
    assert (tmp_path / "certify" / "submission" / "team_solution.script").is_file()


def test_declared_organizer_build_still_needs_a_matching_fingerprint(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
) -> None:
    # Declaring the build organizer-compatible is not enough once a fingerprint
    # is on record: the executable that actually ran has to match it.
    import dataclasses as dc

    base = quick_config()
    config = dc.replace(
        base,
        gmat=dc.replace(
            base.gmat, organizer_compatible=True, organizer_executable_sha256="ab" * 32
        ),
    )
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    stamped(candidate, workshop_script, config)
    evidence = certify_candidate(
        workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "certify", runs=1
    )
    assert evidence["stage"] == "public_gmat_verified"
    assert not (tmp_path / "certify" / "submission" / "team_solution.script").exists()


def test_failed_certification_leaves_no_submission_directory(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
    monkeypatch,
) -> None:
    import orbitwin.verify as verify_module
    from orbitwin.errors import GmatRunError

    def refuse(*_args, **_kwargs):
        raise GmatRunError("simulated parity failure")

    monkeypatch.setattr(verify_module, "_replay_with_diagnostic", refuse)
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    stamped(candidate, workshop_script, config)
    with pytest.raises(GmatRunError):
        certify_candidate(
            workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1
        )
    assert not (tmp_path / "c" / "submission").exists()


def test_solve_refuses_to_emit_a_candidate_with_a_stale_score(
    tmp_path: Path,
    workshop_script: Path,
    monkeypatch,
) -> None:
    # Mutating a score input without recomputing leaves a number ranking still
    # reads. The emission gate must refuse rather than write the artifact.
    import orbitwin.pipeline as pipeline_module
    from orbitwin.errors import PolicyError

    config = scoring_config()
    original = pipeline_module.search_one_burn

    def stale(scenario, cfg, **kwargs):
        candidates = original(scenario, cfg, **kwargs)
        for candidate in candidates:
            candidate.intercept_time_s = (candidate.intercept_time_s or 0.0) + 1.0
        return candidates

    monkeypatch.setattr(pipeline_module, "search_one_burn", stale)
    with pytest.raises(PolicyError, match="no longer has"):
        solve_scenario(workshop_script, config, tmp_path / "solve", render_top=1)


def test_the_emission_gate_backstops_the_renderability_check(
    tmp_path: Path,
    workshop_script: Path,
    monkeypatch,
) -> None:
    # Today an intercept past Tmax is caught earlier: _schedule_slots refuses to
    # schedule it and the candidate is classified unrenderable. That is one
    # layer. This test removes it -- standing in for a future renderer that
    # clamps instead of refusing -- and asserts the gate still refuses to write.
    import orbitwin.pipeline as pipeline_module
    from orbitwin.errors import PolicyError
    from orbitwin.models import Burn, Candidate

    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    late = Candidate(
        burns=(Burn(1000.0, (0.5, 0.0, 0.0), (0.5, 0.0, 0.0), 0),),
        intercept_time_s=scenario.max_time_s + 500.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="intercept-after-tmax",
        metadata={"event_refined": True},
    )
    monkeypatch.setattr(pipeline_module, "zero_burn_candidate", lambda *_: late)
    monkeypatch.setattr(pipeline_module, "search_one_burn", lambda *_: [])
    monkeypatch.setattr(pipeline_module, "search_two_burn_global", lambda *_: [])
    monkeypatch.setattr(pipeline_module, "render_candidate", lambda *_a, **_k: None)

    with pytest.raises(PolicyError, match="violates the competition rules"):
        solve_scenario(workshop_script, config, tmp_path / "solve", render_top=0)
    assert not (tmp_path / "solve" / "best_candidate").exists()


def test_an_over_limit_burn_is_stopped_before_the_gate(
    tmp_path: Path,
    workshop_script: Path,
    monkeypatch,
) -> None:
    # Defence in depth: a 9 km/s burn never reaches the emission gate because
    # the script-level guard rejects the rendered bytes first.
    import orbitwin.pipeline as pipeline_module
    from orbitwin.errors import PolicyError
    from orbitwin.models import Burn, Candidate

    illegal = Candidate(
        burns=(Burn(1000.0, (9.0, 0.0, 0.0), (9.0, 0.0, 0.0), 0),),
        intercept_time_s=4000.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="over-limit",
        metadata={"event_refined": True},
    )
    monkeypatch.setattr(pipeline_module, "zero_burn_candidate", lambda *_: illegal)
    monkeypatch.setattr(pipeline_module, "search_one_burn", lambda *_: [])
    monkeypatch.setattr(pipeline_module, "search_two_burn_global", lambda *_: [])

    with pytest.raises(PolicyError, match="exceeds the formal limit"):
        solve_scenario(workshop_script, quick_config(), tmp_path / "solve", render_top=1)


def test_worker_count_does_not_change_the_result(tmp_path: Path, workshop_script: Path) -> None:
    # Parallelism is only acceptable here if it is invisible in the output.
    # Work is split into contiguous chunks and collected through map(), which
    # yields in input order, so every worker count must produce the identical
    # candidate file.
    import dataclasses as dc

    base = quick_config(local_refine=True)
    digests = []
    for workers in (1, 3):
        config = dc.replace(base, search=dc.replace(base.search, workers=workers))
        destination = tmp_path / f"w{workers}"
        solve_scenario(workshop_script, config, destination, render_top=1)
        digests.append((destination / "candidates.json").read_text(encoding="utf-8"))
    assert digests[0] == digests[1]


def test_gmat_verification_runs_concurrently_without_crossing_candidates(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
) -> None:
    # Each run gets its own working and output directory, and each task mutates
    # only its own candidate, so several GmatConsole runs at once must give the
    # same answer as one at a time.
    import dataclasses as dc

    base = quick_config()
    results = []
    for gmat_workers in (1, 3):
        config = dc.replace(base, gmat=dc.replace(base.gmat, workers=gmat_workers))
        destination = tmp_path / f"g{gmat_workers}"
        summary = solve_scenario(
            workshop_script,
            config,
            destination,
            render_top=1,
            gmat_console=fake_gmat,
            startup_file=fake_startup,
            gmat_verify_top=3,
        )
        results.append(
            (
                summary["gmat_verified_count"],
                summary["best_candidate"] and summary["best_candidate"]["candidate_id"],
                summary["best_candidate_script"],
            )
        )
    assert results[0] == results[1]
    assert results[0][0] > 0, "the test double should verify at least one candidate"


def _narrow_margin_config(safe_dv: float, allow_formal: bool) -> OrbitWinConfig:
    import dataclasses as dc

    base = quick_config()
    return dc.replace(
        base,
        policy=dc.replace(
            base.policy,
            dv_safe_limit_km_s=safe_dv,
            allow_formal_frontier_submission=allow_formal,
        ),
    )


def test_a_formal_frontier_candidate_is_withheld_and_the_reason_is_reported(
    tmp_path: Path, workshop_script: Path
) -> None:
    # Official burns keep the production margin, so a
    # candidate outside it must not be emitted. "No solution" and "solutions
    # exist but none is submittable" are different answers and the summary has
    # to say which one happened.
    # 0.30 km/s puts the Hohmann-scale burns of this scenario outside the
    # margin while leaving them well inside the 1.5 km/s rule.
    summary = solve_scenario(
        workshop_script,
        _narrow_margin_config(0.30, allow_formal=False),
        tmp_path / "withheld",
        render_top=1,
    )
    assert summary["frontiers"]["formal"]["count"] > 0
    assert summary["frontiers"]["formal"]["submittable"] is False
    assert summary["best_candidate"] is None
    assert not (tmp_path / "withheld" / "best_candidate").exists()
    assert any("formal frontier" in warning for warning in summary["warnings"])


def test_a_formal_frontier_candidate_is_usable_when_explicitly_allowed(
    tmp_path: Path, workshop_script: Path
) -> None:
    summary = solve_scenario(
        workshop_script,
        _narrow_margin_config(0.30, allow_formal=True),
        tmp_path / "allowed",
        render_top=1,
    )
    assert summary["frontiers"]["formal"]["submittable"] is True
    best = summary["best_candidate"]
    assert best is not None
    assert best["metadata"]["frontier"] == "formal"


def test_widening_the_search_to_the_rule_boundary_leaves_the_robust_set_alone(
    tmp_path: Path, workshop_script: Path
) -> None:
    # The sweep now runs at 1.5 km/s rather than 1.495. Every extra candidate it
    # finds must land on the formal frontier, never change the robust one.
    # The local refinement searches at the same limit, so it runs here too.
    config = quick_config(local_refine=True)
    summary = solve_scenario(workshop_script, config, tmp_path / "wide")
    robust = summary["frontiers"]["robust"]
    assert robust["count"] > 0
    for candidate in json.loads(
        (tmp_path / "wide" / "candidates.json").read_text(encoding="utf-8")
    ):
        magnitude = max((burn["magnitude_km_s"] for burn in candidate["burns"]), default=0.0)
        if candidate["metadata"]["frontier"] == "robust":
            assert magnitude <= config.policy.dv_safe_limit_km_s + 1e-12
        else:
            assert magnitude <= config.policy.dv_limit_km_s + 1e-12


def test_the_gate_computes_the_frontier_rather_than_trusting_metadata(
    tmp_path: Path, workshop_script: Path, monkeypatch
) -> None:
    # A candidate deserialised from an older candidates.json carries no frontier
    # tag. Reading metadata with a "robust" default would let a 1.499 km/s burn
    # straight through the gate that exists to stop it.
    import orbitwin.pipeline as pipeline_module
    from orbitwin.models import Burn, Candidate

    untagged = Candidate(
        burns=(Burn(1000.0, (1.499, 0.0, 0.0), (1.499, 0.0, 0.0), 0),),
        intercept_time_s=5000.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="untagged-formal",
        metadata={"arrival_time_s": 5000.0, "event_refined": True},
    )
    assert "frontier" not in untagged.metadata

    monkeypatch.setattr(pipeline_module, "zero_burn_candidate", lambda *_: untagged)
    monkeypatch.setattr(pipeline_module, "search_one_burn", lambda *_: [])
    monkeypatch.setattr(pipeline_module, "search_two_burn_global", lambda *_: [])
    monkeypatch.setattr(pipeline_module, "slot_assignment_variants", lambda *_a, **_k: [])

    summary = solve_scenario(workshop_script, quick_config(), tmp_path / "untagged", render_top=1)
    assert summary["best_candidate"] is None
    assert any("formal frontier" in warning for warning in summary["warnings"])


def test_certification_refuses_a_formal_frontier_candidate(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
) -> None:
    # certify is handed a candidate id from candidates.json, so without its own
    # gate it becomes the way around solve()'s.
    import dataclasses as dc

    from orbitwin.errors import PolicyError
    from orbitwin.models import Burn, Candidate

    formal = Candidate(
        burns=(Burn(1000.0, (1.499, 0.0, 0.0), (1.499, 0.0, 0.0), 0),),
        intercept_time_s=4000.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="formal",
        metadata={"arrival_time_s": 4000.0, "event_refined": True},
    )
    config = quick_config()
    stamped(formal, workshop_script, config)
    with pytest.raises(PolicyError, match="formal"):
        certify_candidate(
            workshop_script, formal, config, fake_gmat, fake_startup, tmp_path / "c1", runs=1
        )

    allowed = dc.replace(
        config, policy=dc.replace(config.policy, allow_formal_frontier_submission=True)
    )
    evidence = certify_candidate(
        workshop_script, formal, allowed, fake_gmat, fake_startup, tmp_path / "c2", runs=1
    )
    assert evidence["frontier"] == "formal"


def test_select_candidates_keeps_one_of_each_frontier(workshop_script: Path) -> None:
    # The archive-wide prune in the pipeline would otherwise undo the separation
    # search_one_burn just made.
    from orbitwin.models import Burn, Candidate
    from orbitwin.search import select_candidates

    def point(dv: float, intercept: float, tag: str) -> Candidate:
        return Candidate(
            burns=(Burn(1000.0, (dv, 0.0, 0.0), (dv, 0.0, 0.0), 0),),
            intercept_time_s=intercept,
            minimum_distance_km=0.1,
            distance_at_completion_km=5.0,
            successful=True,
            analytical_endpoint_error_km=0.1,
            branch=tag,
            metadata={"arrival_time_s": intercept},
        )

    config = quick_config()
    fast_formal = point(1.499, 3000.0, "fast-formal")
    slow_robust = point(1.400, 9000.0, "slow-robust")
    kept = select_candidates([fast_formal, slow_robust], 1, config)
    assert len(kept) == 2
    assert {candidate.branch for candidate in kept} == {"fast-formal", "slow-robust"}


def test_certification_refuses_a_score_that_does_not_match_the_replay(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
    monkeypatch,
) -> None:
    # PR #5 proved the exact final script still intercepts. It said nothing
    # about the score, which was computed from the earlier diagnostic run, so
    # the certificate could carry a number describing a different run.
    import dataclasses as dc

    import orbitwin.verify as verify_module
    from orbitwin.errors import GmatRunError

    config = scoring_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]

    real_replay = verify_module._replay_with_diagnostic

    def shifted(*args, **kwargs):
        event, result = real_replay(*args, **kwargs)
        return dc.replace(event, first_entry_time_s=(event.first_entry_time_s or 0.0) + 900.0), result

    monkeypatch.setattr(verify_module, "_replay_with_diagnostic", shifted)
    stamped(candidate, workshop_script, config)
    with pytest.raises(GmatRunError, match="other than the submitted bytes"):
        certify_candidate(
            workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "mismatch", runs=1
        )


def test_certification_records_the_replayed_score(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
) -> None:
    config = scoring_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    stamped(candidate, workshop_script, config)
    evidence = certify_candidate(
        workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "ok", runs=1
    )
    parity = evidence["final_script_parity"]
    assert parity["replayed_score"] == pytest.approx(evidence["candidate"]["score"])
    assert evidence["candidate"]["score_is_stale"] is False


def _console_that_never_intercepts(tmp_path: Path, fake_gmat: Path) -> Path:
    """A GmatConsole double whose report shows the pair never closing.

    The bundled double always writes a report with an interception in it, so
    "GMAT ran and said no" is unreachable through it -- and that is the branch
    where the analytical verdict has to survive.
    """
    source = fake_gmat.read_text(encoding="utf-8")
    replaced = source.replace(
        "pursuer_x = 7000.0 + closest_s",
        "pursuer_x = 9000.0 + closest_s",
    )
    assert replaced != source, "the test double's report rows moved"
    (tmp_path / "scripts").mkdir()
    console = tmp_path / "scripts" / "never_intercepts.py"
    console.write_text(replaced, encoding="utf-8")
    console.chmod(0o755)
    return console


def test_a_console_with_no_verification_budget_is_reported(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # Three of the four bundled profiles ship gmat.verify_top = 0, so passing a
    # console to them runs nothing at all. The filename stays honest, but an
    # operator who supplied a console asked for verification and got silence.
    summary = solve_scenario(
        workshop_script,
        quick_config(),
        tmp_path / "nobudget",
        render_top=1,
        gmat_console=fake_gmat,
        startup_file=fake_startup,
    )
    assert summary["gmat_verification_attempted"] is False
    assert summary["best_candidate_script"] == "analytical_candidate.script"
    assert any("verify_top is 0" in warning for warning in summary["warnings"])


def test_the_analytical_verdict_survives_a_gmat_rejection(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path
) -> None:
    # verify_candidate_with_gmat overwrites `successful` from the report and
    # only then raises, so reading it in the handler recorded GMAT's verdict
    # twice and lost the analytical one -- which is the only thing this field
    # exists to preserve.
    console = _console_that_never_intercepts(tmp_path, fake_gmat)
    startup = startup_for_console_elsewhere(tmp_path)
    solve_scenario(
        workshop_script,
        quick_config(),
        tmp_path / "rejected",
        render_top=1,
        gmat_console=console,
        startup_file=startup,
        gmat_verify_top=2,
    )
    candidates = json.loads(
        (tmp_path / "rejected" / "candidates.json").read_text(encoding="utf-8")
    )
    attempted = [c for c in candidates if "gmat_verification_error" in c["metadata"]]
    assert attempted, "at least one candidate should have been attempted"
    for candidate in attempted:
        assert candidate["successful"] is False
        assert candidate["metadata"]["analytical_successful"] is True
        before = candidate["metadata"]["analytical_before_gmat"]
        assert before["successful"] is True
        assert before["intercept_time_s"] is not None


def test_an_empty_result_says_whether_gmat_was_even_asked(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path
) -> None:
    # "GMAT rejected everything" and "GMAT was given one of twenty" are
    # different answers and an empty best_candidate looks identical.
    console = _console_that_never_intercepts(tmp_path, fake_gmat)
    startup = startup_for_console_elsewhere(tmp_path)
    summary = solve_scenario(
        workshop_script,
        quick_config(),
        tmp_path / "onlyone",
        render_top=1,
        gmat_console=console,
        startup_file=startup,
        gmat_verify_top=1,
    )
    assert summary["best_candidate"] is None
    message = next(w for w in summary["warnings"] if "verified none" in w)
    assert "none of the 1 candidate(s) it was given" in message
    assert "never attempted" in message
    assert "gmat.verify_top is 1" in message


def test_a_withheld_verified_candidate_is_not_reported_as_unverified(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # A candidate can verify and still be withheld by the frontier gate. Saying
    # "GMAT verified none" there would be false.
    summary = solve_scenario(
        workshop_script,
        _narrow_margin_config(0.30, allow_formal=False),
        tmp_path / "withheld_verified",
        render_top=1,
        gmat_console=fake_gmat,
        startup_file=fake_startup,
        gmat_verify_top=2,
    )
    assert summary["gmat_verified_count"] > 0
    assert summary["best_candidate"] is None
    assert not any("verified none" in warning for warning in summary["warnings"])
    assert any("formal frontier" in warning for warning in summary["warnings"])


def test_solve_never_writes_the_submission_name_anywhere(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # The gate that decides best_candidate/ is not the only place that names a
    # file. verify_candidate_with_gmat used to render its working copy as
    # team_solution.script, so every gmat_verification/<id>/ directory held one
    # for a candidate that had reached public_gmat_verified at best. An operator
    # searching the output tree for the submission would have found four.
    destination = tmp_path / "solve"
    summary = solve_scenario(
        workshop_script,
        quick_config(),
        destination,
        render_top=1,
        gmat_console=fake_gmat,
        startup_file=fake_startup,
        gmat_verify_top=2,
    )
    assert summary["gmat_verified_count"] > 0
    assert summary["best_candidate_stage"] != "certified"
    assert list(destination.rglob("team_solution*")) == []


def test_only_the_certified_submission_carries_the_submission_name(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    import dataclasses as dc

    base = quick_config()
    config = dc.replace(
        base,
        gmat=dc.replace(
            base.gmat,
            organizer_compatible=True,
            organizer_executable_sha256=file_sha256(fake_gmat),
        ),
    )
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    root = tmp_path / "certify"
    stamped(candidate, workshop_script, config)
    certify_candidate(workshop_script, candidate, config, fake_gmat, fake_startup, root, runs=1)
    named = sorted(path.relative_to(root) for path in root.rglob("team_solution*"))
    assert named, "a certified run must produce the submission"
    assert all(path.parts[0] == "submission" for path in named), named


def test_the_untried_count_only_includes_demonstrated_interceptions(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path
) -> None:
    # An unrefined candidate's `successful` is its Lambert endpoint residual,
    # around 1e-10 km for any converged arc. Counting those advertises solutions
    # that were never demonstrated.
    import dataclasses as dc

    # A wider grid than the refinement budget, so candidates exist whose only
    # claim to success is the Lambert residual.
    base = quick_config()
    config = dc.replace(
        base,
        search=dc.replace(
            base.search, one_burn_grid_burns=4, one_burn_grid_arrivals=4, event_refine_top=1
        ),
    )
    console = _console_that_never_intercepts(tmp_path, fake_gmat)
    startup = startup_for_console_elsewhere(tmp_path)
    destination = tmp_path / "untried"
    summary = solve_scenario(
        workshop_script,
        config,
        destination,
        render_top=1,
        gmat_console=console,
        startup_file=startup,
        gmat_verify_top=1,
    )
    candidates = json.loads((destination / "candidates.json").read_text(encoding="utf-8"))
    attempted = {c["candidate_id"] for c in candidates if "gmat_verification_error" in c["metadata"]}
    demonstrated = [
        c
        for c in candidates
        if c["candidate_id"] not in attempted
        and c["successful"]
        and c["metadata"].get("renderable")
        and c["metadata"].get("event_refined") is True
    ]
    residual_only = [
        c
        for c in candidates
        if c["candidate_id"] not in attempted
        and c["successful"]
        and c["metadata"].get("renderable")
        and c["metadata"].get("event_refined") is not True
    ]
    assert residual_only, "the scenario should contain unrefined candidates to exclude"
    assert len(residual_only) > len(demonstrated), (
        "the exclusion has to change the number, or the test proves nothing"
    )

    message = next(w for w in summary["warnings"] if "never attempted" in w)
    assert f"A further {len(demonstrated)} candidate(s)" in message
    assert "demonstrated interception" in message


def test_a_successful_verification_keeps_the_analytical_baseline(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # GMAT replaces the intercept epoch and the closest approach. The distance
    # between the two answers is the measurement that says whether the seed
    # layer can be trusted, and it was being discarded on every run that worked.
    destination = tmp_path / "baseline"
    solve_scenario(
        workshop_script,
        quick_config(),
        destination,
        render_top=1,
        gmat_console=fake_gmat,
        startup_file=fake_startup,
        gmat_verify_top=2,
    )
    candidates = json.loads((destination / "candidates.json").read_text(encoding="utf-8"))
    verified = [c for c in candidates if c["verified_by_gmat"]]
    assert verified
    for candidate in verified:
        before = candidate["metadata"]["analytical_before_gmat"]
        assert before["successful"] is True
        assert before["intercept_time_s"] is not None
        assert before["intercept_time_s"] != candidate["intercept_time_s"]


def test_certification_refuses_a_script_with_no_report(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
) -> None:
    # The competition asks for the modified script *and* its report. With no
    # ReportFile there is nothing to upload and nothing for the byte-parity
    # comparison to compare, so certification passed on no evidence at all and
    # produced a submission directory containing no report.
    from orbitwin.errors import PolicyError

    text = workshop_script.read_text(encoding="utf-8")
    stripped: list[str] = []
    dropping = False
    for line in text.splitlines(keepends=True):
        head = line.strip()
        if head.startswith("Create ReportFile"):
            dropping = True
        elif head.startswith("Create ") or head.startswith("BeginMissionSequence"):
            dropping = False
        stripped.append(("% " + line) if (dropping or head.startswith("ReportFile1")) else line)
    script = tmp_path / "noreport.script"
    script.write_text("".join(stripped), encoding="utf-8")

    config = quick_config()
    scenario, _ = parse_gmat_script(script, config)
    assert scenario.report_files == ()
    candidate = search_one_burn(scenario, config)[0]
    stamped(candidate, script, config)
    with pytest.raises(PolicyError, match="no ReportFile"):
        certify_candidate(
            script, candidate, config, fake_gmat, fake_startup, tmp_path / "cert", runs=1
        )


def test_certification_checks_the_runs_executed_the_submitted_bytes(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
    monkeypatch,
) -> None:
    # The run results carry the hash of the script each run actually executed,
    # and nothing compared it. Certification could say "these bytes intercept"
    # and "these bytes are the submission" without linking the two.
    import dataclasses as dc

    import orbitwin.verify as verify_module
    from orbitwin.errors import GmatRunError

    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]

    real_run = verify_module.run_gmat
    seen: list[str] = []

    def tampered(*args, **kwargs):
        result = real_run(*args, **kwargs)
        seen.append(result.script_sha256)
        # Only the exact runs are re-hashed against the submission; leave the
        # diagnostic and parity runs alone so the failure is unambiguous.
        if "exact-run" in str(args[2]):
            return dc.replace(result, script_sha256="f" * 64)
        return result

    monkeypatch.setattr(verify_module, "run_gmat", tampered)
    stamped(candidate, workshop_script, config)
    with pytest.raises(GmatRunError, match="executed"):
        certify_candidate(
            workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "chain", runs=1
        )
    assert seen, "the runs should have happened before the check"


@pytest.mark.parametrize(
    "edit",
    [
        ("ReportFile1.WriteHeaders = true;", "ReportFile1.WriteHeaders = false;"),
        # A whole Array is written on as many lines as it has rows.
        (
            "ReportFile1.Add = {HumanSC.A1ModJulian, HumanSC.EarthMJ2000Eq.X};",
            "ReportFile1.Add = {HumanSC.A1ModJulian, HumanSC.EarthMJ2000Eq.X, M};",
        ),
    ],
    ids=["no-headers", "array-column"],
)
def test_certification_counts_official_rows_as_gmat_writes_them(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, edit
) -> None:
    text = workshop_script.read_text(encoding="utf-8").replace(*edit)
    if "M}" in edit[1]:
        text = text.replace("Create ReportFile ReportFile1;", "Create Array M[2,2];\nCreate ReportFile ReportFile1;")
    assert text != workshop_script.read_text(encoding="utf-8")
    script = tmp_path / "official.script"
    script.write_text(text, encoding="utf-8")
    config = quick_config()
    scenario, _ = parse_gmat_script(script, config)
    candidate = search_one_burn(scenario, config)[0]
    stamped(candidate, script, config)
    evidence = certify_candidate(
        script, candidate, config, fake_gmat, fake_startup, tmp_path / "certify", runs=1
    )
    assert evidence["frozen_rows"] == 0
