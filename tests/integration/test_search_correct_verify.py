"""solve under the ranking rules: ranking search, GMAT correction, verification."""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from conftest import TEAM_NUMBER, quick_config, with_ranking_report

import orbitwin.gmat_loop as gmat_loop
from orbitwin.config import GmatConfig, OrbitWinConfig, ScoreConfig, validate_config
from orbitwin.errors import ConfigurationError, GmatRunError
from orbitwin.gmat_shoot import CorrectionResult, CorrectionStep
from orbitwin.io import read_candidates
from orbitwin.models import Burn, Candidate, CandidateStage
from orbitwin.pipeline import solve_scenario

MU = 398600.4418


def _script(tmp_path: Path, no_maneuver_script: Path) -> Path:
    """The pursuer on 7000 km, the target 300 km above, phased for a transfer."""
    target_km = 7300.0
    transfer_s = math.pi * math.sqrt(((7000.0 + target_km) / 2.0) ** 3 / MU)
    target_ta = (180.0 - math.degrees(math.sqrt(MU / target_km**3) * transfer_s)) % 360.0
    text = no_maneuver_script.read_text(encoding="utf-8")
    for old, new in {
        "Pursuer.TA = 300;": "Pursuer.TA = 0;",
        "AlienSC.SMA = 9000;": f"AlienSC.SMA = {target_km!r};",
        "AlienSC.INC = 43;": "AlienSC.INC = 28;",
        "AlienSC.RAAN = 65;": "AlienSC.RAAN = 15;",
        "AlienSC.TA = 20;": f"AlienSC.TA = {target_ta!r};",
    }.items():
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "coplanar.script"
    path.write_text(with_ranking_report(text), encoding="utf-8")
    return path


def _config(**gmat: Any) -> OrbitWinConfig:
    base = quick_config(local_refine=True)
    return dataclasses.replace(
        base,
        pursuer_name="Pursuer",
        target_name="AlienSC",
        search=dataclasses.replace(
            base.search,
            one_burn_samples=0,
            two_burn_enabled=False,
            ranking_search=True,
            ranking_transfer_starts=1,
            ranking_random_starts=0,
            ranking_polish_top=1,
            ranking_screen_iterations=20,
            ranking_polish_iterations=20,
        ),
        policy=dataclasses.replace(
            base.policy,
            rules="ranking",
            horizon_s=4000.0,
            team_number=TEAM_NUMBER,
            dv_limit_km_s=0.25,
            dv_safe_limit_km_s=0.249,
            total_dv_limit_km_s=3.0,
            total_dv_safe_limit_km_s=2.99,
        ),
        gmat=GmatConfig(**gmat),
    )


def _solve(
    script: Path, config: OrbitWinConfig, out: Path, fake_gmat: Path | None, fake_startup: Path
) -> tuple[dict[str, Any], list[Candidate]]:
    summary = solve_scenario(
        script, config, out, gmat_console=fake_gmat,
        startup_file=fake_startup if fake_gmat is not None else None,
    )
    return summary, read_candidates(out / "candidates.json")


def _stages(out: Path) -> dict[str, str]:
    """Each candidate's stage as solve wrote it; read_candidates does not restore one."""
    written = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    rows = written["candidates"] if isinstance(written, dict) else written
    return {row["candidate_id"]: row["stage"] for row in rows}


def test_solve_searches_corrects_and_verifies(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _script(tmp_path, no_maneuver_script)
    summary, candidates = _solve(
        script, _config(correct_top=2, verify_top=0), tmp_path / "solve", fake_gmat, fake_startup
    )
    assert summary["ranking_search"]["target"] == "gmat"
    searched = [c for c in candidates if c.branch.startswith("ranking-search/")]
    assert searched and summary["ranking_search"]["kept"] >= 1
    assert all(c.metadata["horizon_s"] == 4000.0 for c in candidates)
    corrections = summary["gmat_corrections"]
    assert 1 <= len(corrections) <= 2
    assert len({c.candidate_id for c in candidates}) == len(candidates)
    stages = _stages(tmp_path / "solve")
    for record in corrections:
        if record["converged"]:
            # Verified though verify_top is 0: every correction is.
            assert stages[record["corrected_id"]] == "public_gmat_verified"
    assert summary["best_candidate_is_gmat_verified"] is any(r["converged"] for r in corrections)


def test_without_a_console_the_search_aims_at_the_conic(
    tmp_path: Path, no_maneuver_script: Path, fake_startup: Path
) -> None:
    script = _script(tmp_path, no_maneuver_script)
    summary, candidates = _solve(
        script, _config(correct_top=2), tmp_path / "solve", None, fake_startup
    )
    assert summary["ranking_search"]["target"] == "two-body"
    assert summary["gmat_corrections"] == []
    assert any(c.branch.startswith("ranking-search/") for c in candidates)


def _fake_corrections(
    monkeypatch: pytest.MonkeyPatch, outcome: str, details: list[dict[str, Any]] | None = None,
    first_km: float = 46.5,
) -> list[str]:
    """Replace the corrector; returns the ids it was handed, and fills `details`.

    Its first round flies the original `first_km` from the target at closest.
    """
    seen: list[str] = []

    def correct(document, scenario, candidate, config, *args, **kwargs) -> CorrectionResult:
        seen.append(candidate.candidate_id)
        if details is not None:
            details.append({"workers": config.gmat.workers, **kwargs})
        fixed = None
        if outcome == "unchanged":
            fixed = candidate
        elif outcome in ("moved", "unrenderable", "closest"):
            burns = list(candidate.burns)
            last = burns[-1]
            burns[-1] = dataclasses.replace(
                last,
                time_s=4000.0 if outcome == "unrenderable" else last.time_s,
                dv_inertial_km_s=tuple(0.999 * c for c in last.dv_inertial_km_s),
                dv_script_km_s=tuple(0.999 * c for c in last.dv_script_km_s),
            )
            # A new schedule is a new candidate, unstamped, as gmat_shoot builds one.
            unstamped = {
                key: value for key, value in candidate.metadata.items()
                if key not in ("scenario_sha256", "horizon_s", "searched_score")
            }
            fixed = dataclasses.replace(
                candidate, burns=tuple(burns), metadata=unstamped, candidate_id="",
                _stage=CandidateStage.ANALYTICAL,
            )
        log = (
            CorrectionStep(
                round=1, kind="flight", run_ids=(), wall_s=0.0, gmat_s=0.0,
                closest_approach_km=first_km,
            ),
        )
        if outcome == "closest":
            # Unconverged: the moved schedule is only a closest flight.
            return CorrectionResult(
                converged=False, candidate=None, closest_flight=fixed,
                closest_approach_km=6.0, closest_approach_time_s=None, first_entry_time_s=None,
                reason="test: closest flight only", rounds=1, gmat_runs=1, wall_s=0.0, log=log,
            )
        return CorrectionResult(
            converged=fixed is not None, candidate=fixed, closest_flight=None,
            closest_approach_km=None, closest_approach_time_s=None, first_entry_time_s=None,
            reason=None if fixed is not None else "test: did not converge",
            rounds=1, gmat_runs=1, wall_s=0.0, log=log,
        )

    monkeypatch.setattr(gmat_loop, "correct_with_gmat", correct)
    return seen


@pytest.mark.parametrize(
    "outcome", ["unconverged", "closest", "unchanged", "moved", "unrenderable"]
)
def test_what_a_correction_adds(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    script = _script(tmp_path, no_maneuver_script)
    # What solve keeps with no correction at all.
    _, plain = _solve(script, _config(verify_top=0), tmp_path / "plain", fake_gmat, fake_startup)
    _baseline_ids = [c.candidate_id for c in plain]
    seen = _fake_corrections(monkeypatch, outcome)
    summary, candidates = _solve(
        script, _config(correct_top=1, verify_top=0), tmp_path / "solve", fake_gmat, fake_startup
    )
    assert len(seen) == 1
    ids = [c.candidate_id for c in candidates]
    assert len(set(ids)) == len(ids)
    (record,) = summary["gmat_corrections"]
    stages = _stages(tmp_path / "solve")
    (zero,) = [c.candidate_id for c in candidates if c.branch == "zero-burn"]
    # The zero burn is always verified under the ranking rules; the rest is the correction.
    verified = [
        key for key, stage in stages.items() if stage == "public_gmat_verified" and key != zero
    ]
    # A console with a correction or a scoring miss to replay did verify something.
    assert not any("nothing was verified" in w for w in summary["warnings"])
    assert summary["gmat_verification_attempted"] is True
    assert stages[zero] == "public_gmat_verified"
    if outcome in ("unconverged", "closest", "unrenderable"):
        # Only the record and the zero burn: verify_top is 0, so nothing else is flown.
        assert verified == []
        assert record["corrected_id"] not in ids
        assert summary["renderable_count"] == len(_baseline_ids)
        assert set(ids) == set(_baseline_ids)
        if outcome == "unrenderable":
            assert record["reason"].startswith("corrected, but not renderable: ")
        else:
            assert not record["converged"]
        return
    if outcome == "unchanged":
        assert record["corrected_id"] == seen[0] and verified == [seen[0]]
    else:
        assert record["corrected_id"] not in (seen[0], None)
        assert verified == [record["corrected_id"]]
        added = next(c for c in candidates if c.candidate_id == record["corrected_id"])
        assert added.metadata["renderable"] is True and added.metadata["horizon_s"] == 4000.0
        # Searched unscored, as the candidate it corrects.
        assert "searched_score" in added.metadata and added.metadata["searched_score"] is None
        # In the pool best is chosen from, beside every original.
        assert summary["renderable_count"] == len(_baseline_ids) + 1
        assert set(ids) == {*_baseline_ids, record["corrected_id"]}


@pytest.mark.parametrize(
    "change, match",
    [
        ({"search": {"ranking_search": True}}, "needs a rulebook with an announced T_max"),
        ({"gmat": {"correct_top": -1}}, "correct_top"),
        ({"gmat": {"verify_misses": -1}}, "verify_misses"),
        ({"gmat": {"correct_rounds": 0}}, "correct_rounds"),
        ({"gmat": {"correct_aim_km": 5.5}}, "correct_aim_km"),
        ({"gmat": {"correct_aim_km": 0.0}}, "correct_aim_km"),
        ({"search": {"ranking_force_model_margin_km": -1.0}}, "margin"),
        ({"search": {"ranking_transfer_starts": -1}}, "ranking_transfer_starts"),
        ({"search": {"ranking_random_starts": -1}}, "ranking_random_starts"),
        ({"search": {"ranking_polish_top": -1}}, "ranking_polish_top"),
        ({"search": {"ranking_screen_iterations": 0}}, "iterations"),
        ({"search": {"ranking_polish_iterations": 0}}, "iterations"),
        ({"search": {"ranking_correction_margin_km_s": -0.01}}, "correction_margin"),
        ({"search": {"ranking_correction_margin_km_s": 1.495}}, "correction_margin"),
        (
            {
                "search": {
                    "ranking_search": True, "ranking_transfer_starts": 0,
                    "ranking_random_starts": 0,
                },
                "policy": {
                    "rules": "ranking", "horizon_s": 4000.0, "dv_limit_km_s": 0.25,
                    "dv_safe_limit_km_s": 0.249, "total_dv_limit_km_s": 3.0,
                    "total_dv_safe_limit_km_s": 2.99,
                },
            },
            "search nothing",
        ),
    ],
)
def test_the_settings_are_checked(change: dict[str, dict[str, Any]], match: str) -> None:
    base = OrbitWinConfig()
    config = dataclasses.replace(
        base,
        **{
            section: dataclasses.replace(getattr(base, section), **fields)
            for section, fields in change.items()
        },
    )
    with pytest.raises(ConfigurationError, match=match):
        validate_config(config)


@pytest.mark.parametrize("without", [False, True])
def test_the_margins_searched_are_recorded(
    tmp_path: Path, no_maneuver_script: Path, fake_startup: Path, without: bool
) -> None:
    script = _script(tmp_path, no_maneuver_script)
    config = _config()
    config = dataclasses.replace(
        config,
        search=dataclasses.replace(config.search, ranking_search_without_margin=without),
    )
    _, candidates = _solve(script, config, tmp_path / "solve", None, fake_startup)
    written = json.loads((tmp_path / "solve" / "summary.json").read_text(encoding="utf-8"))
    margins = [0.0, 10.0] if without else [10.0]
    assert written["ranking_search"]["force_model_margins_km"] == margins
    searched = {
        c.metadata["force_model_margin_km"]
        for c in candidates
        if c.branch.startswith("ranking-search/")
    }
    assert searched <= set(margins)


def test_only_hits_with_burns_are_corrected(
    tmp_path: Path, no_maneuver_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.search import zero_burn_candidate

    config = _config(correct_top=2, workers=2)
    scenario, document = parse_gmat_script(_script(tmp_path, no_maneuver_script), config)
    hit = Candidate(
        burns=(), intercept_time_s=100.0, minimum_distance_km=1.0,
        distance_at_completion_km=5.0, successful=True, analytical_endpoint_error_km=None,
        branch="test",
    )
    searched, _ = gmat_loop.ranking_search_candidates(
        scenario, document, config, tmp_path, None, None
    )
    hits = [c for c in searched if c.successful]
    misses = [dataclasses.replace(c, successful=False, candidate_id="x" + c.candidate_id)
              for c in hits]
    pool = [zero_burn_candidate(scenario, config), hit, *misses, *hits, *hits]
    chosen = gmat_loop.correction_choices(pool, config)
    assert [c.candidate_id for c in chosen] == [c.candidate_id for c in [*hits, *hits][:2]]


@pytest.mark.parametrize("without", [False, True])
def test_the_searchs_zero_burn_is_left_to_solve(
    tmp_path: Path, no_maneuver_script: Path, monkeypatch: pytest.MonkeyPatch, without: bool
) -> None:
    from orbitwin.gmat_parser import parse_gmat_script

    config = _config()
    config = dataclasses.replace(
        config,
        search=dataclasses.replace(config.search, ranking_search_without_margin=without),
    )
    # Two per margin: the zero burn, unmargined, must not take one of them.
    monkeypatch.setattr(gmat_loop, "KEEP_TOP", 2)
    scenario, document = parse_gmat_script(_script(tmp_path, no_maneuver_script), config)
    real = gmat_loop.search_ranking

    def baseline_first(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        assert result.baseline is not None
        result.solutions.insert(0, result.baseline)
        return result

    monkeypatch.setattr(gmat_loop, "search_ranking", baseline_first)
    searched, _ = gmat_loop.ranking_search_candidates(
        scenario, document, config, tmp_path, None, None
    )
    assert searched and all(candidate.burns for candidate in searched)
    if without:
        unmargined = [c for c in searched if c.metadata["force_model_margin_km"] == 0.0]
        assert len(unmargined) == 2


def test_the_zero_burn_is_verified_when_a_miss_scores(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _script(tmp_path, no_maneuver_script)
    _solve(script, _config(verify_top=1), tmp_path / "solve", fake_gmat, fake_startup)
    stages = _stages(tmp_path / "solve")
    (zero,) = [c for c in read_candidates(tmp_path / "solve" / "candidates.json")
               if c.branch == "zero-burn"]
    verified = [key for key, stage in stages.items() if stage == "public_gmat_verified"]
    assert zero.candidate_id in verified and len(verified) == 2


def test_the_zero_burn_takes_no_slot_when_only_a_hit_scores(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    solve_scenario(
        workshop_script, quick_config(), tmp_path / "solve",
        gmat_console=fake_gmat, startup_file=fake_startup, gmat_verify_top=1,
    )
    stages = _stages(tmp_path / "solve")
    (zero,) = [c for c in read_candidates(tmp_path / "solve" / "candidates.json")
               if c.branch == "zero-burn"]
    verified = [key for key, stage in stages.items() if stage == "public_gmat_verified"]
    assert len(verified) == 1 and zero.candidate_id not in verified


def test_each_margin_keeps_its_own_top(
    tmp_path: Path, no_maneuver_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.ranking_search import schedule_signature

    config = _config()
    config = dataclasses.replace(
        config,
        search=dataclasses.replace(
            config.search, ranking_search_without_margin=True, ranking_random_starts=2
        ),
    )
    scenario, document = parse_gmat_script(_script(tmp_path, no_maneuver_script), config)
    real = gmat_loop.search_ranking
    captured: list[Any] = []

    def capture(*args: Any, **kwargs: Any) -> Any:
        captured.append(real(*args, **kwargs))
        return captured[-1]

    monkeypatch.setattr(gmat_loop, "search_ranking", capture)
    monkeypatch.setattr(gmat_loop, "KEEP_TOP", 2)
    searched, _ = gmat_loop.ranking_search_candidates(
        scenario, document, config, tmp_path, None, None
    )
    (result,) = captured
    # Labels repeat across solutions; the schedule does not.
    kept = {
        (
            tuple(round(b.time_s, 2) for b in c.burns),
            tuple(tuple(round(x, 6) for x in b.dv_inertial_km_s) for b in c.burns),
        )
        for c in searched
    }
    for margin in (0.0, 10.0):
        own = [s for s in result.solutions
               if s.floor_margin_km == margin and s is not result.baseline][:2]
        assert len(own) == 2, "the scenario must give each margin two solutions"
        for solution in own:
            assert schedule_signature(solution) in kept, solution.label


def test_one_failing_correction_loses_nothing_else(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = gmat_loop.correct_with_gmat
    calls: list[str] = []

    def first_raises(document, scenario, candidate, *args: Any, **kwargs: Any) -> Any:
        calls.append(candidate.candidate_id)
        if len(calls) == 1:
            raise ValueError("a bug in one correction")
        return real(document, scenario, candidate, *args, **kwargs)

    monkeypatch.setattr(gmat_loop, "correct_with_gmat", first_raises)
    summary, _ = _solve(
        _script(tmp_path, no_maneuver_script), _config(correct_top=2, verify_top=0),
        tmp_path / "solve", fake_gmat, fake_startup,
    )
    first, second = summary["gmat_corrections"]
    assert first["reason"] == "the correction raised ValueError: a bug in one correction"
    assert not first["converged"] and first["gmat_runs"] is None
    assert "first_raises" in first["traceback"] and "ValueError" in first["traceback"]
    assert first["error"] == "ValueError: a bug in one correction"
    assert [(r["kind"], r["candidate_id"], r["error"]) for r in summary["gmat_failures"]["runs"]] == [
        ("correction", first["candidate_id"], first["error"])]
    assert second["converged"] and second["error"] is None
    assert (tmp_path / "solve" / "candidates.json").is_file()


def test_a_failed_target_run_falls_back_to_the_conic(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orbitwin.errors import GmatRunError

    def fails(*args: Any, **kwargs: Any) -> Any:
        raise GmatRunError("coast run failed")

    monkeypatch.setattr(gmat_loop, "target_ephemeris_from_gmat", fails)
    summary, candidates = _solve(
        _script(tmp_path, no_maneuver_script), _config(), tmp_path / "solve", fake_gmat,
        fake_startup,
    )
    assert summary["ranking_search"]["target"] == "two-body"
    assert summary["ranking_search"]["target_error"] == "GmatRunError: coast run failed"
    assert any("GMAT's target run failed with GmatRunError" in w for w in summary["warnings"])
    assert any(c.branch.startswith("ranking-search/") for c in candidates)


def test_a_full_archive_keeps_the_zero_burn_and_the_searchs_results(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    config = _config(verify_top=1)
    config = dataclasses.replace(
        config, search=dataclasses.replace(config.search, keep_candidates=1)
    )
    summary, candidates = _solve(
        _script(tmp_path, no_maneuver_script), config, tmp_path / "solve", fake_gmat, fake_startup
    )
    (zero,) = [c for c in candidates if c.branch == "zero-burn"]
    assert _stages(tmp_path / "solve")[zero.candidate_id] == "public_gmat_verified"
    searched = [c for c in candidates if c.branch.startswith("ranking-search/")]
    assert len(searched) == summary["ranking_search"]["kept"]


@pytest.mark.parametrize(
    "outcome, first_km, superseded, verify_top",
    [
        ("moved", 46.5, True, 1),
        ("moved", 3.0, False, 1),
        ("unrenderable", 46.5, False, 1),
        ("unconverged", 45.4, False, 1),
        ("unconverged", 45.4, False, 0),
        ("unconverged", 3.0, False, 1),
    ],
)
def test_an_original_gmat_flew_as_a_miss_is_blocked(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, outcome: str, first_km: float, superseded: bool,
    verify_top: int,
) -> None:
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.rank import submission_blockers

    script = _script(tmp_path, no_maneuver_script)
    seen = _fake_corrections(monkeypatch, outcome, first_km=first_km)
    config = _config(correct_top=1, verify_top=verify_top)
    summary, candidates = _solve(script, config, tmp_path / "solve", fake_gmat, fake_startup)
    stages = _stages(tmp_path / "solve")
    (record,) = summary["gmat_corrections"]
    assert record["first_flight_closest_km"] == first_km
    original = next(c for c in candidates if c.candidate_id == seen[0])
    assert original.metadata["gmat_first_flight_closest_km"] == first_km
    scenario, _ = parse_gmat_script(script, config)
    blockers = submission_blockers(original, scenario, config)
    outside = [r for r in blockers if "outside the sphere" in r]
    # Whatever its correction did, GMAT flew it outside: rank names why, unless
    # verification has since flown it to T_max and found an entry (the fake does).
    entered = (original.metadata.get("gmat_event") or {}).get("first_entry_time_s") is not None
    assert bool(outside) is (first_km > 5.0 and not entered)
    assert (outside and f"{first_km:.3f} km" in outside[0]) or not outside
    assert ("superseded_by" in original.metadata) is superseded
    assert any("certify its correction" in r for r in outside) is superseded
    verified = [key for key, stage in stages.items() if stage == "public_gmat_verified"]
    if superseded:
        # Never flown again: the correction, the zero burn and one other.
        assert stages[seen[0]] != "public_gmat_verified"
        assert len(verified) == 3
    elif verify_top:
        assert stages[seen[0]] == "public_gmat_verified" and entered
    else:
        # Unverified, so the first flight is all GMAT has said: blocked, in solve too.
        assert stages[seen[0]] != "public_gmat_verified" and outside
        assert summary["best_candidate"]["candidate_id"] != seen[0]


def test_certify_refuses_a_hit_gmat_flew_as_a_miss(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    import orbitwin.verify as verify
    from orbitwin.errors import PolicyError
    from orbitwin.gmat_parser import parse_gmat_script

    script = _script(tmp_path, no_maneuver_script)
    config = _config()
    scenario, _ = parse_gmat_script(script, config)
    searched, _ = gmat_loop.ranking_search_candidates(scenario, None, config, tmp_path, None, None)  # type: ignore[arg-type]
    hit = next(c for c in searched if c.successful)

    # Evidence on file: refused before anything is created.
    flagged = dataclasses.replace(hit, metadata={**hit.metadata, "gmat_first_flight_closest_km": 45.4})
    with pytest.raises(PolicyError, match=r"45\.400 km from the target at closest.*larger --gmat-verify-top"):
        verify.certify_candidate(script, flagged, config, fake_gmat, fake_startup, tmp_path / "c1", runs=1)
    assert not (tmp_path / "c1").exists()

    # Verified, it is scored as the miss GMAT flies, and certify takes it as one.
    verified = dataclasses.replace(flagged, successful=False)
    from orbitwin.constraints import gmat_contradicts_hit

    assert gmat_contradicts_hit(verified, config) is None
    # Verified as a hit after all, found later than the first flight reached.
    entered = dataclasses.replace(
        flagged, metadata={**flagged.metadata, "gmat_event": {"first_entry_time_s": 3000.0}}
    )
    assert gmat_contradicts_hit(entered, config) is None


def test_a_hit_outside_the_margins_is_not_corrected(
    tmp_path: Path, no_maneuver_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orbitwin.gmat_parser import parse_gmat_script

    config = _config(correct_top=5)
    scenario, document = parse_gmat_script(_script(tmp_path, no_maneuver_script), config)
    searched, _ = gmat_loop.ranking_search_candidates(
        scenario, document, config, tmp_path, None, None
    )
    hit = next(c for c in searched if c.successful)
    first = hit.burns[0]
    scale = 0.2495 / first.magnitude_km_s
    formal = dataclasses.replace(
        hit,
        burns=(
            dataclasses.replace(
                first,
                dv_inertial_km_s=tuple(scale * x for x in first.dv_inertial_km_s),
                dv_script_km_s=tuple(scale * x for x in first.dv_script_km_s),
            ),
            *hit.burns[1:],
        ),
        candidate_id="",
    )
    assert gmat_loop.correction_choices([formal, hit], config) == [hit]


def test_the_settings_reach_the_search_and_the_corrector(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    for name in ("build_setup", "build_starts", "search_ranking"):
        real = getattr(gmat_loop, name)

        def spy(*args: Any, _real: Any = real, _name: str = name, **kwargs: Any) -> Any:
            captured[_name] = kwargs
            return _real(*args, **kwargs)

        monkeypatch.setattr(gmat_loop, name, spy)
    details: list[dict[str, Any]] = []
    _fake_corrections(monkeypatch, "unconverged", details)
    config = _config(correct_top=1, verify_top=0, correct_aim_km=0.25, correct_rounds=3, workers=3)
    config = dataclasses.replace(
        config,
        search=dataclasses.replace(
            config.search, ranking_screen_iterations=21, ranking_polish_iterations=22,
            ranking_polish_top=2, ranking_correction_margin_km_s=0.01,
            ranking_transfer_starts=2, ranking_random_starts=1,
        ),
        policy=dataclasses.replace(config.policy, minimum_flown_altitude_km=150.0),
    )
    _solve(_script(tmp_path, no_maneuver_script), config, tmp_path / "solve", fake_gmat,
           fake_startup)
    setup = captured["build_setup"]
    assert (setup["screen_iterations"], setup["polish_iterations"]) == (21, 22)
    assert setup["correction_margin_km_s"] == 0.01 and setup["altitude_floor_km"] == 150.0
    assert (captured["build_starts"]["transfer_count"], captured["build_starts"]["random_count"]) \
        == (2, 1)
    assert captured["search_ranking"]["polish_top"] == 2
    # Each correction measures J alone; the corrections themselves run three at a time.
    assert details == [{"workers": 1, "aim_km": 0.25, "max_rounds": 3}]


def test_the_floor_defaults_to_100_km(
    tmp_path: Path, no_maneuver_script: Path, fake_startup: Path
) -> None:
    config = _config()
    assert config.policy.minimum_flown_altitude_km is None
    summary, _ = _solve(_script(tmp_path, no_maneuver_script), config, tmp_path / "s", None,
                        fake_startup)
    assert summary["ranking_search"]["altitude_floor_km"] == 100.0


def test_corrections_run_gmat_workers_at_a_time(
    tmp_path: Path, no_maneuver_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from orbitwin.gmat_parser import parse_gmat_script

    config = _config(correct_top=2, workers=2)
    scenario, document = parse_gmat_script(_script(tmp_path, no_maneuver_script), config)
    searched, _ = gmat_loop.ranking_search_candidates(
        scenario, document, config, tmp_path, None, None
    )
    hits = [c for c in searched if c.successful][:2]
    assert len(hits) == 2
    together = threading.Barrier(2, timeout=10)

    def correct(document, scenario, candidate, *args: Any, **kwargs: Any) -> CorrectionResult:
        together.wait()  # Breaks unless both corrections are running at once.
        return CorrectionResult(
            converged=False, candidate=None, closest_flight=None, closest_approach_km=None,
            closest_approach_time_s=None, first_entry_time_s=None, reason="test",
            rounds=1, gmat_runs=1, wall_s=0.0,
        )

    monkeypatch.setattr(gmat_loop, "correct_with_gmat", correct)
    queue = gmat_loop.GmatQueue(config.gmat.workers)
    results: list[Any] = []
    for hit in hits:
        queue.add(
            1,
            partial(gmat_loop.correct_one, hit, document, scenario, config, "c", None, tmp_path),
            results.append,
        )
    queue.run()
    assert [r.reason for r in results] == ["test", "test"]


class _Peak:
    """The most calls in flight at once; each call holds 0.2 s so free threads overlap.

    The calls numbered in `meet` also wait for each other, so a pool that
    should run them together reaches that peak however the threads start.
    """

    def __init__(self, meet: tuple[int, ...] = ()) -> None:
        import threading

        self._lock = threading.Lock()
        self._running = 0
        self._meet = meet
        self._barrier = threading.Barrier(len(meet), timeout=10) if meet else None
        self.peak = 0
        self.calls = 0

    def hold(self) -> None:
        import contextlib
        import threading
        import time

        with self._lock:
            self._running += 1
            self.calls += 1
            call = self.calls
            self.peak = max(self.peak, self._running)
        if self._barrier is not None and call in self._meet:
            # Broken when they run one at a time; the peak then says so.
            with contextlib.suppress(threading.BrokenBarrierError):
                self._barrier.wait()
        time.sleep(0.2)
        with self._lock:
            self._running -= 1


def test_corrections_never_run_more_than_gmat_workers_at_once(
    tmp_path: Path, no_maneuver_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orbitwin.gmat_parser import parse_gmat_script

    config = _config(correct_top=4, workers=2)
    scenario, document = parse_gmat_script(_script(tmp_path, no_maneuver_script), config)
    searched, _ = gmat_loop.ranking_search_candidates(
        scenario, document, config, tmp_path, None, None
    )
    pair = [c for c in searched if c.successful][:2]
    assert len(pair) == 2
    # Four distinct hits, so each would get a correction directory of its own.
    hits = [
        dataclasses.replace(c, candidate_id=f"{c.candidate_id}-{k}") for k in (0, 1) for c in pair
    ]
    flights = _Peak()

    def correct(document, scenario, candidate, *args: Any, **kwargs: Any) -> CorrectionResult:
        flights.hold()
        return CorrectionResult(
            converged=False, candidate=None, closest_flight=None, closest_approach_km=None,
            closest_approach_time_s=None, first_entry_time_s=None, reason="test",
            rounds=1, gmat_runs=1, wall_s=0.0,
        )

    monkeypatch.setattr(gmat_loop, "correct_with_gmat", correct)
    queue = gmat_loop.GmatQueue(config.gmat.workers)
    results: list[Any] = []
    for hit in hits:
        queue.add(
            1,
            partial(gmat_loop.correct_one, hit, document, scenario, config, "c", None, tmp_path),
            results.append,
        )
    queue.run()
    assert [r.reason for r in results] == ["test"] * 4
    assert flights.calls == 4
    assert flights.peak <= 2


def test_verification_never_runs_more_than_gmat_workers_at_once(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import orbitwin.pipeline as pipeline

    flights = _Peak(meet=(2, 3))  # the zero burn alone, then the first two others

    def verify(*args: Any, **kwargs: Any) -> None:
        flights.hold()

    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    _solve(
        _script(tmp_path, no_maneuver_script), _config(correct_top=0, verify_top=4, workers=2),
        tmp_path / "solve", fake_gmat, fake_startup,
    )
    # The zero burn alone, then four others: two at once, never three.
    assert flights.calls == 5
    assert flights.peak == 2


def test_a_correction_whose_verification_fails_stays_renderable(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import orbitwin.pipeline as pipeline

    _fake_corrections(monkeypatch, "moved")

    def refuses(*args: Any, **kwargs: Any) -> None:
        raise ValueError("verification refused")

    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", refuses)
    summary, _ = _solve(
        _script(tmp_path, no_maneuver_script), _config(correct_top=1, verify_top=0),
        tmp_path / "solve", fake_gmat, fake_startup,
    )
    (record,) = summary["gmat_corrections"]
    assert _stages(tmp_path / "solve")[record["corrected_id"]] == "renderable"


def test_the_defaults() -> None:
    gmat = GmatConfig()
    assert (gmat.correct_top, gmat.correct_rounds, gmat.correct_aim_km) == (0, 4, 0.5)
    assert gmat.verify_misses == 0
    search = OrbitWinConfig().search
    assert search.ranking_search is False
    assert (search.ranking_force_model_margin_km, search.ranking_correction_margin_km_s) == (
        10.0, 0.0
    )


def test_the_search_seeds_up_to_the_burn_budget(
    tmp_path: Path, no_maneuver_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window fits many more burns 100 s apart than the budget is meant to search."""
    from orbitwin.gmat_parser import parse_gmat_script

    base = _config()
    config = dataclasses.replace(base, search=dataclasses.replace(base.search, ranking_max_burns=2))
    scenario, document = parse_gmat_script(_script(tmp_path, no_maneuver_script), config)
    counts: list[int] = []
    real = gmat_loop.build_starts

    def capture(setup, config, burn_counts, **kwargs):  # type: ignore[no-untyped-def]
        counts.extend(burn_counts)
        assert setup.limits.max_burns > 2
        return real(setup, config, burn_counts, **kwargs)

    monkeypatch.setattr(gmat_loop, "build_starts", capture)
    gmat_loop.ranking_search_candidates(scenario, document, config, tmp_path, None, None)
    assert counts == [1, 2]


def test_an_original_stands_down_only_for_a_verified_correction(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A correction whose verification fails leaves its original in the running."""
    import orbitwin.pipeline as pipeline
    from orbitwin.errors import GmatRunError
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.rank import submission_blockers

    script = _script(tmp_path, no_maneuver_script)
    seen = _fake_corrections(monkeypatch, "moved", first_km=46.5)
    corrections: list[Candidate] = []
    real_record, real_verify = pipeline.correction_record, pipeline.verify_candidate_with_gmat

    def record(*args: Any, **kwargs: Any) -> Any:
        fixed, entry = real_record(*args, **kwargs)
        if fixed is not None:
            corrections.append(fixed)
        return fixed, entry

    def verify(script_path: Path, scenario: Any, document: Any, candidate: Candidate,
               *args: Any, **kwargs: Any) -> Any:
        if any(candidate is fixed for fixed in corrections):
            raise GmatRunError("test: the correction's replay failed")
        return real_verify(script_path, scenario, document, candidate, *args, **kwargs)

    monkeypatch.setattr(pipeline, "correction_record", record)
    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    config = _config(correct_top=1, verify_top=1)
    summary, candidates = _solve(script, config, tmp_path / "solve", fake_gmat, fake_startup)
    (record,) = summary["gmat_corrections"]
    original = next(c for c in candidates if c.candidate_id == seen[0])
    correction = next(c for c in candidates if c.candidate_id == record["corrected_id"])
    assert "gmat_verification_error" in correction.metadata
    assert "superseded_by" not in original.metadata
    scenario, _ = parse_gmat_script(script, config)
    assert any(
        "GMAT verification failed" in r for r in submission_blockers(correction, scenario, config)
    )


def test_the_search_seeds_no_more_burns_than_the_template_has_slots(
    tmp_path: Path, workshop_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two slots and a budget of 13: three burns would not render."""
    from orbitwin.gmat_parser import parse_gmat_script

    base = quick_config()
    config = dataclasses.replace(
        base,
        search=dataclasses.replace(
            base.search, ranking_search=True, ranking_max_burns=13, ranking_transfer_starts=1,
            ranking_random_starts=0, ranking_polish_top=1, ranking_screen_iterations=5,
            ranking_polish_iterations=5,
        ),
        policy=dataclasses.replace(
            base.policy, rules="ranking", horizon_s=30000.0, dv_limit_km_s=0.25,
            dv_safe_limit_km_s=0.249, total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99,
        ),
    )
    scenario, document = parse_gmat_script(workshop_script, config)
    counts: list[int] = []
    real = gmat_loop.build_starts

    def capture(setup, config, burn_counts, **kwargs):  # type: ignore[no-untyped-def]
        counts.extend(burn_counts)
        return real(setup, config, burn_counts, **kwargs)

    monkeypatch.setattr(gmat_loop, "build_starts", capture)
    gmat_loop.ranking_search_candidates(scenario, document, config, tmp_path, None, None)
    assert counts == [1, 2]


def _spy_choices(monkeypatch: pytest.MonkeyPatch) -> list[Candidate]:
    """The hits solve hands the corrector, in its order."""
    import orbitwin.pipeline as pipeline

    chosen: list[Candidate] = []
    real = pipeline.correction_choices

    def spy(*args: Any, **kwargs: Any) -> list[Candidate]:
        out = real(*args, **kwargs)
        chosen.extend(out)
        return out

    monkeypatch.setattr(pipeline, "correction_choices", spy)
    return chosen


def test_a_correction_is_verified_while_the_next_is_still_correcting(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import orbitwin.pipeline as pipeline

    chosen = _spy_choices(monkeypatch)
    _fake_corrections(monkeypatch, "moved")
    fake = gmat_loop.correct_with_gmat
    verifying = threading.Event()
    overlapped: list[bool] = []

    def correct(document: Any, scenario: Any, candidate: Candidate, *args: Any,
                **kwargs: Any) -> CorrectionResult:
        if candidate is not chosen[0]:
            # Before, nothing was verified until this returned.
            overlapped.append(verifying.wait(30))
        return fake(document, scenario, candidate, *args, **kwargs)

    real_verify = pipeline.verify_candidate_with_gmat

    def verify(script_path: Path, scenario: Any, document: Any, candidate: Candidate,
               *args: Any, **kwargs: Any) -> Any:
        if candidate.branch != "zero-burn":
            verifying.set()
        return real_verify(script_path, scenario, document, candidate, *args, **kwargs)

    monkeypatch.setattr(gmat_loop, "correct_with_gmat", correct)
    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    summary, _ = _solve(
        _script(tmp_path, no_maneuver_script), _config(correct_top=2, verify_top=0, workers=2),
        tmp_path / "solve", fake_gmat, fake_startup,
    )
    assert len(chosen) == 2 and overlapped == [True]
    assert all(record["converged"] for record in summary["gmat_corrections"])
    stages = _stages(tmp_path / "solve")
    assert all(
        stages[record["corrected_id"]] == "public_gmat_verified"
        for record in summary["gmat_corrections"]
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_corrections_and_verifications_share_gmat_workers(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, workers: int,
) -> None:
    import threading
    import time

    import orbitwin.pipeline as pipeline

    _fake_corrections(monkeypatch, "moved")
    fake = gmat_loop.correct_with_gmat
    real_verify = pipeline.verify_candidate_with_gmat
    lock = threading.Lock()
    live = {"now": 0, "max": 0, "calls": 0}

    def counted(call: Any) -> Any:
        def run(*args: Any, **kwargs: Any) -> Any:
            with lock:
                live["now"] += 1
                live["calls"] += 1
                live["max"] = max(live["max"], live["now"])
            try:
                time.sleep(0.05)
                return call(*args, **kwargs)
            finally:
                with lock:
                    live["now"] -= 1
        return run

    monkeypatch.setattr(gmat_loop, "correct_with_gmat", counted(fake))
    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", counted(real_verify))
    _solve(
        _script(tmp_path, no_maneuver_script),
        _config(correct_top=2, verify_top=2, workers=workers),
        tmp_path / "solve", fake_gmat, fake_startup,
    )
    assert live["calls"] >= 4
    assert live["max"] <= workers


def _outcome(summary: dict[str, Any], out: Path) -> Any:
    """What solve decided, without wall times or paths."""
    written = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    rows = written["candidates"] if isinstance(written, dict) else written
    return (
        [{k: v for k, v in r.items() if k != "wall_s"} for r in summary["gmat_corrections"]],
        (summary["best_candidate"] or {}).get("candidate_id"),
        [
            (
                r["candidate_id"], r["stage"], r["successful"], r["score"],
                r["metadata"].get("superseded_by"),
                r["metadata"].get("gmat_first_flight_closest_km"),
            )
            for r in rows
        ],
    )


def test_the_outcome_does_not_depend_on_which_correction_ends_first(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    chosen = _spy_choices(monkeypatch)
    _fake_corrections(monkeypatch, "moved", first_km=46.5)
    fake = gmat_loop.correct_with_gmat
    script = _script(tmp_path, no_maneuver_script)
    outcomes = []
    for slow in (0, 1):
        chosen.clear()

        def correct(document: Any, scenario: Any, candidate: Candidate, *args: Any,
                    slow: int = slow, **kwargs: Any) -> CorrectionResult:
            if candidate is chosen[slow]:
                time.sleep(0.5)
            return fake(document, scenario, candidate, *args, **kwargs)

        monkeypatch.setattr(gmat_loop, "correct_with_gmat", correct)
        out = tmp_path / f"solve-{slow}"
        summary, _ = _solve(
            script, _config(correct_top=2, verify_top=1, workers=2), out, fake_gmat, fake_startup
        )
        assert len(chosen) == 2
        outcomes.append(_outcome(summary, out))
    assert outcomes[0] == outcomes[1]
    assert any(row[4] for row in outcomes[0][2])  # an original stood down


def test_an_original_gmat_hit_as_planned_is_verified_with_its_first_flight(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its verification.json records the first flight, as it did before the overlap."""
    import orbitwin.pipeline as pipeline

    seen = _fake_corrections(monkeypatch, "unchanged", first_km=0.3)
    real_verify = pipeline.verify_candidate_with_gmat
    stamped: dict[str, Any] = {}

    def verify(script_path: Path, scenario: Any, document: Any, candidate: Candidate,
               *args: Any, **kwargs: Any) -> Any:
        stamped[candidate.candidate_id] = candidate.metadata.get("gmat_first_flight_closest_km")
        return real_verify(script_path, scenario, document, candidate, *args, **kwargs)

    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    _solve(
        _script(tmp_path, no_maneuver_script), _config(correct_top=2, verify_top=0, workers=2),
        tmp_path / "solve", fake_gmat, fake_startup,
    )
    assert seen and all(stamped[key] == 0.3 for key in seen)


def test_replays_go_first_until_the_corrections_left_fit_in_one_round(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One worker: a replay overtakes the corrections waiting while more than one
    waits, and then falls behind them; the zero burn is replayed last."""
    import orbitwin.pipeline as pipeline

    chosen = _spy_choices(monkeypatch)
    _fake_corrections(monkeypatch, "moved")
    fake = gmat_loop.correct_with_gmat
    real_verify = pipeline.verify_candidate_with_gmat
    order: list[str] = []

    def correct(document: Any, scenario: Any, candidate: Candidate, *args: Any,
                **kwargs: Any) -> CorrectionResult:
        order.append(f"c{chosen.index(candidate)}")
        return fake(document, scenario, candidate, *args, **kwargs)

    def verify(script_path: Path, scenario: Any, document: Any, candidate: Candidate,
               *args: Any, **kwargs: Any) -> Any:
        order.append("zero" if candidate.branch == "zero-burn" else "v")
        return real_verify(script_path, scenario, document, candidate, *args, **kwargs)

    monkeypatch.setattr(gmat_loop, "correct_with_gmat", correct)
    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    base = _config(correct_top=6, verify_top=0, workers=1)
    config = dataclasses.replace(
        base, search=dataclasses.replace(base.search, ranking_random_starts=8)
    )
    _solve(_script(tmp_path, no_maneuver_script), config, tmp_path / "solve", fake_gmat,
           fake_startup)
    count = len(chosen)
    assert count >= 3
    # Simulated: after correction k, count - k - 1 wait; more than one, its replay goes next.
    expected: list[str] = []
    behind = 0
    for k in range(count):
        expected.append(f"c{k}")
        if count - k - 1 > 1:
            expected.append("v")
        else:
            behind += 1
    expected += ["v"] * behind + ["zero"]
    assert order == expected


def test_a_slow_first_correction_does_not_hold_back_the_next_replay(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import orbitwin.pipeline as pipeline

    chosen = _spy_choices(monkeypatch)
    _fake_corrections(monkeypatch, "moved")
    fake = gmat_loop.correct_with_gmat
    verifying = threading.Event()
    overlapped: list[bool] = []

    def correct(document: Any, scenario: Any, candidate: Candidate, *args: Any,
                **kwargs: Any) -> CorrectionResult:
        if candidate is chosen[0]:
            overlapped.append(verifying.wait(30))
        return fake(document, scenario, candidate, *args, **kwargs)

    real_verify = pipeline.verify_candidate_with_gmat

    def verify(script_path: Path, scenario: Any, document: Any, candidate: Candidate,
               *args: Any, **kwargs: Any) -> Any:
        if candidate.branch != "zero-burn":
            verifying.set()
        return real_verify(script_path, scenario, document, candidate, *args, **kwargs)

    monkeypatch.setattr(gmat_loop, "correct_with_gmat", correct)
    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    summary, _ = _solve(
        _script(tmp_path, no_maneuver_script), _config(correct_top=2, verify_top=0, workers=2),
        tmp_path / "solve", fake_gmat, fake_startup,
    )
    assert len(chosen) == 2 and overlapped == [True]
    # Recorded in the order chosen all the same.
    assert [r["candidate_id"] for r in summary["gmat_corrections"]] == [
        c.candidate_id for c in chosen
    ]


def test_a_twin_of_a_chosen_original_is_replayed_after_its_first_flight_is_stamped(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A correction resolving to an original whose own correction is still running
    waits for that one's stamp, found by id: the archive may hold two objects of one id."""
    import time

    import orbitwin.pipeline as pipeline
    from orbitwin.gmat_shoot import CorrectionStep

    real_prune = pipeline.prune_beside_model_plans
    state: dict[str, Any] = {}

    def prune(renderable: Any, config: Any) -> list[Candidate]:
        kept = real_prune(renderable, config)
        hits = [c for c in kept if c.successful and c.burns]
        twin = dataclasses.replace(hits[1], metadata=dict(hits[1].metadata))
        assert twin.candidate_id == hits[1].candidate_id and twin is not hits[1]
        state.update(a=hits[0], b=hits[1], twin=twin)
        kept.insert(kept.index(hits[1]) + 1, twin)  # unchosen with correct_top=2
        return kept

    log = (CorrectionStep(round=1, kind="flight", run_ids=(), wall_s=0.0, gmat_s=0.0,
                          closest_approach_km=46.5),)

    def correct(document: Any, scenario: Any, candidate: Candidate, *args: Any,
                **kwargs: Any) -> CorrectionResult:
        if candidate is state["a"]:
            # a's correction lands on b's schedule, which resolves to the twin.
            fixed = dataclasses.replace(
                state["b"], metadata=dict(state["b"].metadata), candidate_id="",
                _stage=CandidateStage.ANALYTICAL,
            )
            return CorrectionResult(
                converged=True, candidate=fixed, closest_flight=None, closest_approach_km=None,
                closest_approach_time_s=None, first_entry_time_s=None, reason=None,
                rounds=1, gmat_runs=1, wall_s=0.0, log=log,
            )
        time.sleep(1.0)  # b's own correction ends last and stamps the twin
        return CorrectionResult(
            converged=False, candidate=None, closest_flight=None, closest_approach_km=None,
            closest_approach_time_s=None, first_entry_time_s=None, reason="test",
            rounds=1, gmat_runs=1, wall_s=0.0, log=log,
        )

    stamped: dict[int, Any] = {}
    real_verify = pipeline.verify_candidate_with_gmat

    def verify(script_path: Path, scenario: Any, document: Any, candidate: Candidate,
               *args: Any, **kwargs: Any) -> Any:
        stamped[id(candidate)] = candidate.metadata.get("gmat_first_flight_closest_km")
        return real_verify(script_path, scenario, document, candidate, *args, **kwargs)

    monkeypatch.setattr(pipeline, "prune_beside_model_plans", prune)
    monkeypatch.setattr(gmat_loop, "correct_with_gmat", correct)
    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    base = _config(correct_top=2, verify_top=0, workers=2)
    config = dataclasses.replace(
        base, search=dataclasses.replace(base.search, ranking_random_starts=8)
    )
    _solve(_script(tmp_path, no_maneuver_script), config, tmp_path / "solve", fake_gmat,
           fake_startup)
    assert stamped[id(state["twin"])] == 46.5


def test_two_corrections_on_one_schedule_keep_the_first_chosen(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """However they finish: before the overlap the first chosen was kept."""
    import time

    import orbitwin.pipeline as pipeline
    from orbitwin.gmat_shoot import CorrectionStep

    real_prune = pipeline.prune_beside_model_plans
    state: dict[str, Any] = {}

    def prune(renderable: Any, config: Any) -> list[Candidate]:
        kept = real_prune(renderable, config)
        a = next(c for c in kept if c.successful and c.burns)
        last = a.burns[-1]
        # The same epochs, slots, branch and arrival; another impulse, so another id.
        sibling = dataclasses.replace(
            a,
            burns=(*a.burns[:-1], dataclasses.replace(
                last, dv_inertial_km_s=tuple(0.998 * c for c in last.dv_inertial_km_s),
                dv_script_km_s=tuple(0.998 * c for c in last.dv_script_km_s),
            )),
            metadata=dict(a.metadata), candidate_id="",
        )
        assert sibling.candidate_id != a.candidate_id
        kept.append(sibling)
        state.update(a=a, sibling=sibling)
        return kept

    log = (CorrectionStep(round=1, kind="flight", run_ids=(), wall_s=0.0, gmat_s=0.0,
                          closest_approach_km=46.5),)

    def correct(document: Any, scenario: Any, candidate: Candidate, *args: Any,
                **kwargs: Any) -> CorrectionResult:
        if candidate is state["a"]:
            time.sleep(0.5)  # the first chosen converges last
        a = state["a"]
        last = a.burns[-1]
        fixed = dataclasses.replace(
            a,
            burns=(*a.burns[:-1], dataclasses.replace(
                last, dv_inertial_km_s=tuple(0.999 * c for c in last.dv_inertial_km_s),
                dv_script_km_s=tuple(0.999 * c for c in last.dv_script_km_s),
            )),
            metadata={**a.metadata, "test_from": candidate.candidate_id},
            candidate_id="", _stage=CandidateStage.ANALYTICAL,
        )
        return CorrectionResult(
            converged=True, candidate=fixed, closest_flight=None, closest_approach_km=None,
            closest_approach_time_s=None, first_entry_time_s=None, reason=None,
            rounds=1, gmat_runs=1, wall_s=0.0, log=log,
        )

    monkeypatch.setattr(pipeline, "prune_beside_model_plans", prune)
    monkeypatch.setattr(
        pipeline, "correction_choices", lambda *args: [state["a"], state["sibling"]]
    )
    monkeypatch.setattr(gmat_loop, "correct_with_gmat", correct)
    summary, candidates = _solve(
        _script(tmp_path, no_maneuver_script), _config(correct_top=2, verify_top=0, workers=2),
        tmp_path / "solve", fake_gmat, fake_startup,
    )
    first, second = summary["gmat_corrections"]
    assert first["corrected_id"] == second["corrected_id"]
    (kept,) = [c for c in candidates if c.candidate_id == first["corrected_id"]]
    assert kept.metadata["test_from"] == state["a"].candidate_id
    assert _stages(tmp_path / "solve")[kept.candidate_id] == "public_gmat_verified"


def test_an_original_and_its_correction_share_a_schedule_key_in_any_burn_order(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    """The corrector writes burns in time order; an original need not hold them so."""
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.pipeline import _schedule_key

    config = _config()
    scenario, document = parse_gmat_script(_script(tmp_path, no_maneuver_script), config)
    searched, _ = gmat_loop.ranking_search_candidates(
        scenario, document, config, tmp_path, None, None
    )
    hit = next(c for c in searched if len(c.burns) > 1)
    reversed_ = dataclasses.replace(hit, burns=tuple(reversed(hit.burns)), candidate_id="")
    assert _schedule_key(reversed_) == _schedule_key(hit)


def test_an_original_two_corrections_hit_as_planned_is_replayed_once(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the overlap it was replayed twice into one directory, and the refused
    second run un-verified it."""
    import orbitwin.pipeline as pipeline

    chosen = _spy_choices(monkeypatch)
    _fake_corrections(monkeypatch, "unchanged")
    fake = gmat_loop.correct_with_gmat

    def correct(document: Any, scenario: Any, candidate: Candidate, *args: Any,
                **kwargs: Any) -> CorrectionResult:
        # Both corrections come back as the first chosen, unchanged.
        return fake(document, scenario, chosen[0], *args, **kwargs)

    replays: list[str] = []
    real_verify = pipeline.verify_candidate_with_gmat

    def verify(script_path: Path, scenario: Any, document: Any, candidate: Candidate,
               *args: Any, **kwargs: Any) -> Any:
        replays.append(candidate.candidate_id)
        return real_verify(script_path, scenario, document, candidate, *args, **kwargs)

    monkeypatch.setattr(gmat_loop, "correct_with_gmat", correct)
    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    summary, _ = _solve(
        _script(tmp_path, no_maneuver_script), _config(correct_top=2, verify_top=0, workers=1),
        tmp_path / "solve", fake_gmat, fake_startup,
    )
    assert len(chosen) == 2
    assert [r["corrected_id"] for r in summary["gmat_corrections"]] == [chosen[0].candidate_id] * 2
    assert replays.count(chosen[0].candidate_id) == 1
    assert _stages(tmp_path / "solve")[chosen[0].candidate_id] == "public_gmat_verified"


SCORE = ScoreConfig(kt=0.003, ct=3200.0, kv=0.0012, cv=2000.0, delta_v_unit="m/s", cr_km=100.0)


def _scored(**gmat: Any) -> OrbitWinConfig:
    return dataclasses.replace(_config(**gmat), score=SCORE)


def _missing(tmp_path: Path, no_maneuver_script: Path, miss_km: float = 6.0) -> Path:
    """The coplanar script, with the fake console told to miss everything by `miss_km`."""
    script = _script(tmp_path, no_maneuver_script)
    script.write_text(
        f"% FAKE_GMAT_MISS_KM {miss_km}\n" + script.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return script


def _offer_misses(
    monkeypatch: pytest.MonkeyPatch, distances_km: list[float]
) -> list[Candidate]:
    """The ranking search's first plan also offered as analytic misses, one per
    distance, each its own schedule; filled in, in that order, once solve searches."""
    import orbitwin.pipeline as pipeline
    from orbitwin.constraints import apply_constraints
    from orbitwin.score import apply_score

    real = pipeline.ranking_search_candidates
    offered: list[Candidate] = []

    def search(scenario: Any, document: Any, config: OrbitWinConfig, *args: Any,
               **kwargs: Any) -> Any:
        found, record = real(scenario, document, config, *args, **kwargs)
        plan = next(c for c in found if c.burns)
        largest = max(range(len(plan.burns)), key=lambda k: plan.burns[k].magnitude_km_s)
        for index, distance_km in enumerate(distances_km):
            burns = list(plan.burns)
            # 1 m/s more each, far past _same_plan's 5e-5 km/s: a plan of its own.
            factor = 1.0 + 1.0e-3 * (index + 1) / burns[largest].magnitude_km_s
            burns[largest] = dataclasses.replace(
                burns[largest],
                dv_inertial_km_s=tuple(factor * c for c in burns[largest].dv_inertial_km_s),
                dv_script_km_s=tuple(factor * c for c in burns[largest].dv_script_km_s),
            )
            miss = dataclasses.replace(
                plan, burns=tuple(burns), candidate_id="", successful=False,
                intercept_time_s=None, minimum_distance_km=distance_km,
                distance_at_completion_km=distance_km, metadata=dict(plan.metadata),
                score=None, score_components={},
            )
            apply_constraints(miss, scenario, config)
            apply_score(miss, scenario, config.score)
            offered.append(miss)
        return [*found, *offered], record

    monkeypatch.setattr(pipeline, "ranking_search_candidates", search)
    return offered


def _zero_score(out: Path) -> float:
    (zero,) = [c for c in read_candidates(out / "candidates.json") if c.branch == "zero-burn"]
    analytical = zero.metadata.get("analytical_before_gmat") or {}
    return float(analytical.get("score", zero.score))


def test_without_verify_misses_a_scored_miss_is_never_flown(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 2 P1-1: with verify_top 0, as ranking.yaml has it, a miss the analytic
    score puts above the zero burn is neither corrected nor replayed, so never chosen."""
    offered = _offer_misses(monkeypatch, [6.0])
    out = tmp_path / "solve"
    _solve(_missing(tmp_path, no_maneuver_script), _scored(correct_top=2, verify_top=0),
           out, fake_gmat, fake_startup)
    (miss,) = offered
    assert miss.score is not None and miss.score > _zero_score(out)
    assert _stages(out)[miss.candidate_id] == "renderable"


def test_verify_misses_flies_the_best_scored_misses(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offered = _offer_misses(monkeypatch, [6.0, 20.0])
    out = tmp_path / "solve"
    summary, _ = _solve(
        _missing(tmp_path, no_maneuver_script),
        _scored(correct_top=2, verify_top=0, verify_misses=1), out, fake_gmat, fake_startup,
    )
    near, far = offered
    assert near.score is not None and far.score is not None and near.score > far.score
    stages = _stages(out)
    assert stages[near.candidate_id] == "public_gmat_verified"
    assert stages[far.candidate_id] == "renderable"
    (record,) = summary["gmat_miss_verifications"]
    assert record["candidate_id"] == near.candidate_id and record["verified_by_gmat"]
    assert record["analytical"]["score"] == pytest.approx(near.score)
    # GMAT's distance replaces the analytic one: the fake misses everything by 6 km.
    assert record["gmat_minimum_distance_km"] == pytest.approx(6.0, abs=0.5)
    # So the zero burn, which pays no delta-v, outscores it, as on R5, and is chosen.
    assert summary["best_candidate"]["branch"] == "zero-burn"
    assert summary["best_candidate"]["score"] > record["gmat_score"]


def test_solve_records_the_zero_burns_closest_approach_as_gmat_flew_it(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
) -> None:
    """What RUNBOOK step 5 passes as --zero-burn-dmin-km; the fake misses by 6 km."""
    out = tmp_path / "solve"
    summary, candidates = _solve(
        _missing(tmp_path, no_maneuver_script),
        _scored(correct_top=0, verify_top=0, verify_misses=1), out, fake_gmat, fake_startup,
    )
    zero = next(c for c in candidates if c.branch == "zero-burn")
    # A file confers no GMAT stage on what it reads back; the stage solve wrote does.
    assert _stages(out)[zero.candidate_id] == "public_gmat_verified"
    assert zero.minimum_distance_km is not None
    assert summary["zero_burn_closest_approach"] == {
        "km": zero.minimum_distance_km, "measured_by": "gmat", "gmat_verification_error": None,
    }
    assert zero.minimum_distance_km == pytest.approx(6.0, abs=0.5)


@pytest.mark.parametrize("rules", ["ranking", "preliminary"])
def test_solve_prints_the_zero_burn_line_runbook_step_4_reads(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], rules: str,
) -> None:
    """Through the CLI on the fake console; only a term with C_r prices the zero burn."""
    import orbitwin.cli as cli

    config = _scored(correct_top=0, verify_top=0, verify_misses=1)
    if rules == "preliminary":
        config = dataclasses.replace(
            config,
            policy=dataclasses.replace(config.policy, rules="preliminary", horizon_s=None,
                                       total_dv_limit_km_s=None, total_dv_safe_limit_km_s=None),
            search=dataclasses.replace(config.search, ranking_search=False),
            score=dataclasses.replace(config.score, cr_km=None),
        )
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    out = tmp_path / "solve"
    code = cli.main(["solve", str(_missing(tmp_path, no_maneuver_script)), "--config", "given",
                     "--output", str(out), "--gmat-console", str(fake_gmat),
                     "--startup-file", str(fake_startup),
                     *(["--rules", rules] if rules != "ranking" else [])])
    assert code == 0
    printed = [line for line in capsys.readouterr().err.splitlines()
               if line.startswith("orbitwin solve: the zero burn passes")]
    if rules == "preliminary":
        assert printed == []
        return
    closest = json.loads((out / "summary.json").read_text("utf-8"))["zero_burn_closest_approach"]
    assert printed == [
        f"orbitwin solve: the zero burn passes {closest['km']:.3f} km off (GMAT); "
        f"scripts/ranking_day_config.py --zero-burn-dmin-km {closest['km']:.3f}"
    ]


def test_misses_take_no_correction_or_replay_from_the_hits(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fake hits everything here, so the hits' corrections converge and replay.
    script = _script(tmp_path, no_maneuver_script)
    runs: dict[int, tuple[list[Any], set[str]]] = {}
    for budget in (0, 2):
        with monkeypatch.context() as patch:
            offered = _offer_misses(patch, [6.0, 7.0])
            chosen = _spy_choices(patch)
            out = tmp_path / f"solve-{budget}"
            summary, _ = _solve(
                script, _scored(correct_top=2, verify_top=1, verify_misses=budget), out,
                fake_gmat, fake_startup,
            )
            verified = {k for k, v in _stages(out).items() if v == "public_gmat_verified"}
            flown = [r["candidate_id"] for r in summary["gmat_miss_verifications"]]
            assert flown == [c.candidate_id for c in offered][:budget]
            runs[budget] = (
                [c.candidate_id for c in chosen] + [
                    r["corrected_id"] for r in summary["gmat_corrections"]
                ],
                verified - {c.candidate_id for c in offered},
            )
    assert runs[0] == runs[2]
    assert runs[0][1]


def test_a_miss_scored_under_the_zero_burn_is_not_flown(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offered = _offer_misses(monkeypatch, [1.0e5])
    out = tmp_path / "solve"
    summary, _ = _solve(
        _missing(tmp_path, no_maneuver_script),
        _scored(verify_top=0, verify_misses=3), out, fake_gmat, fake_startup,
    )
    (miss,) = offered
    assert miss.score is not None and miss.score <= _zero_score(out)
    assert _stages(out)[miss.candidate_id] == "renderable"
    assert summary["gmat_miss_verifications"] == []


def test_a_verified_miss_can_be_chosen(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chosen on GMAT's score: here the zero burn's replay fails, so the miss is best."""
    import orbitwin.pipeline as pipeline
    from orbitwin.errors import GmatRunError

    offered = _offer_misses(monkeypatch, [6.0])
    real_verify = pipeline.verify_candidate_with_gmat

    def verify(script_path: Path, scenario: Any, document: Any, candidate: Candidate,
               *args: Any, **kwargs: Any) -> Any:
        if candidate.branch == "zero-burn":
            raise GmatRunError("test: the zero burn's replay failed")
        return real_verify(script_path, scenario, document, candidate, *args, **kwargs)

    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    summary, _ = _solve(
        _missing(tmp_path, no_maneuver_script),
        _scored(correct_top=0, verify_top=0, verify_misses=1), tmp_path / "solve",
        fake_gmat, fake_startup,
    )
    (miss,) = offered
    assert summary["best_candidate"]["candidate_id"] == miss.candidate_id
    assert summary["best_candidate_is_gmat_verified"] is True


@pytest.mark.parametrize("rules", ["ranking", "preliminary"])
def test_only_rules_that_score_a_miss_look_for_misses_to_fly(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, rules: str,
) -> None:
    import orbitwin.pipeline as pipeline

    asked: list[int] = []

    def every_miss(candidates: Any, config: Any, **kwargs: Any) -> list[Candidate]:
        asked.append(len(candidates))
        return []

    monkeypatch.setattr(pipeline, "miss_choices", every_miss)
    base = quick_config()
    policy = base.policy
    if rules == "ranking":
        policy = dataclasses.replace(
            policy, rules="ranking", horizon_s=30000.0, dv_limit_km_s=0.25,
            dv_safe_limit_km_s=0.249, total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99,
        )
    # The preliminary's distance term has no C_r, and its config refuses one.
    score = SCORE if rules == "ranking" else dataclasses.replace(SCORE, cr_km=None)
    config = dataclasses.replace(base, policy=policy, score=score,
                                 gmat=GmatConfig(verify_top=1, verify_misses=5))
    solve_scenario(
        workshop_script, config, tmp_path / "solve",
        gmat_console=fake_gmat, startup_file=fake_startup,
    )
    assert bool(asked) is (rules == "ranking")


def _choices(
    pool: list[Candidate], config: OrbitWinConfig, alignment_km: float | None = None
) -> list[Candidate]:
    return gmat_loop.miss_choices(pool, config, alignment_minimum_distance_km=alignment_km)


def _miss(burn_dv: float, score: float | None, **changes: Any) -> Candidate:
    burn = Burn(time_s=100.0, dv_inertial_km_s=(burn_dv, 0.0, 0.0),
                dv_script_km_s=(burn_dv, 0.0, 0.0), slot_index=0)
    fields: dict[str, Any] = dict(
        burns=(burn,), intercept_time_s=None, minimum_distance_km=50.0,
        distance_at_completion_km=50.0, successful=False, analytical_endpoint_error_km=None,
        branch="ranking-search/1", score=score,
    )
    fields.update(changes)
    return Candidate(**fields)


def test_miss_choices_order_budget_and_floor() -> None:
    config = _scored(verify_misses=2)
    zero = _miss(0.0, 30.0, burns=(), branch="zero-burn")
    best, second, third = _miss(0.1, 40.0), _miss(0.11, 35.0), _miss(0.12, 31.0)
    under, hit = _miss(0.13, 29.0), _miss(0.14, 45.0, successful=True)
    variant = dataclasses.replace(
        best, candidate_id="",
        burns=(dataclasses.replace(best.burns[0], slot_index=1),),
    )
    outside = _miss(0.2499, 39.0)  # over the 0.249 km/s production margin
    modelled = _miss(0.15, 33.0, metadata={"pursuer_model": "j2"})
    pool = [hit, best, variant, outside, second, modelled, third, zero, under]
    chosen = _choices(pool, config)
    # One per trajectory, robust only, above the zero burn; model plans budgeted apart.
    assert chosen == [best, second, modelled]
    assert _choices(pool, _scored(verify_misses=0)) == []
    # No zero burn to submit: a miss only has to score above nothing.
    assert _choices([best, second, third], _scored(verify_misses=3)) == [
        best, second, third
    ]
    # Through the Earth takes no run; under the altitude floor is GMAT's flight to judge.
    low = _miss(0.16, 41.0, metadata={"minimum_pursuer_radius_km": 6378.1363 + 99.0})
    underground = _miss(0.17, 42.0, metadata={"minimum_pursuer_radius_km": 6000.0})
    floored = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, minimum_flown_altitude_km=100.0)
    )
    assert _choices([underground, low, best, zero], floored) == [low, best]
    through = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, allow_body_intersecting_submission=True)
    )
    assert _choices([underground, low, best, zero], through) == [underground, low]


def test_miss_choices_fly_one_of_a_plan_found_from_several_starts() -> None:
    """R5 kept 13 copies of one 13-burn miss, up to 6.4e-6 km/s and 3.1 ms apart;
    GMAT flew all eight it was given to 113.65 km."""
    config = _scored(verify_misses=2)
    zero = _miss(0.0, 20.0, burns=(), branch="zero-burn")
    plan = _miss(0.1, 30.0)
    copy = _miss(0.1 + 6.4e-6, 29.9999, candidate_id="")
    copy = dataclasses.replace(
        copy, candidate_id="",
        burns=(dataclasses.replace(copy.burns[0], time_s=100.0031),),
    )
    apart = _miss(0.1 + 1.0e-4, 29.99)
    assert _choices([plan, copy, apart, zero], config) == [plan, apart]
    # Five times the spread measured is still one plan.
    wider = _miss(0.1 + 3.0e-5, 29.998)
    assert _choices([plan, wider, apart, zero], config) == [plan, apart]
    # Burns compared in time order, whatever order they are listed in; a burn
    # more is another plan.
    second = Burn(time_s=300.0, dv_inertial_km_s=(0.0, 0.1, 0.0),
                  dv_script_km_s=(0.0, 0.1, 0.0), slot_index=1)
    two = dataclasses.replace(plan, candidate_id="", burns=(*plan.burns, second))
    shuffled = dataclasses.replace(plan, candidate_id="", burns=(second, *plan.burns),
                                   score=29.0)
    assert _choices([two, shuffled, plan, zero], config) == [two, plan]


def test_the_ranking_profiles_replay_misses() -> None:
    import yaml

    root = Path(__file__).resolve().parents[2]
    budgets = {
        name: yaml.safe_load((root / "config" / name).read_text(encoding="utf-8"))["gmat"]
        for name in ("ranking.yaml", "ranking-rehearsal.yaml")
    }
    assert {gmat["verify_misses"] for gmat in budgets.values()} == {4}
    # With the zero burn, one round of the workers; a pursuer model would double the 4.
    assert all(gmat["verify_misses"] + 1 <= gmat["workers"] for gmat in budgets.values())
    searches = [
        yaml.safe_load((root / "config" / name).read_text(encoding="utf-8"))["search"]
        for name in budgets
    ]
    assert all(search.get("ranking_pursuer_model", "two-body") == "two-body" for search in searches)


def test_misses_are_replayed_after_every_correction_and_hit_replay(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One worker, so the queue's order is the order run."""
    import orbitwin.pipeline as pipeline

    offered = _offer_misses(monkeypatch, [6.0, 7.0])
    _fake_corrections(monkeypatch, "moved")
    order: list[str] = []
    real_verify = pipeline.verify_candidate_with_gmat
    real_correct = gmat_loop.correct_with_gmat

    def verify(script_path: Path, scenario: Any, document: Any, candidate: Candidate,
               *args: Any, **kwargs: Any) -> Any:
        order.append(f"verify {candidate.candidate_id}")
        return real_verify(script_path, scenario, document, candidate, *args, **kwargs)

    def correct(document: Any, scenario: Any, candidate: Candidate, *args: Any,
                **kwargs: Any) -> Any:
        order.append(f"correct {candidate.candidate_id}")
        return real_correct(document, scenario, candidate, *args, **kwargs)

    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    monkeypatch.setattr(gmat_loop, "correct_with_gmat", correct)
    summary, _ = _solve(_script(tmp_path, no_maneuver_script),
                        _scored(correct_top=2, verify_top=0, verify_misses=2, workers=1),
                        tmp_path / "solve", fake_gmat, fake_startup)
    misses = [f"verify {c.candidate_id}" for c in offered]
    (zero,) = [c for c in read_candidates(tmp_path / "solve" / "candidates.json")
               if c.branch == "zero-burn"]
    corrected = [r["corrected_id"] for r in summary["gmat_corrections"]]
    assert sum(entry.startswith("correct ") for entry in order) == 2 and all(corrected)
    # Both corrections replayed, then the zero burn, which shares the last tier.
    assert {f"verify {i}" for i in corrected} <= set(order[:-3])
    assert order[-3:] == [f"verify {zero.candidate_id}", *misses]


def _flown(km: float, burn_dv: float, **changes: Any) -> Candidate:
    """A refined miss `km` off, at `burn_dv` km/s, unscored."""
    metadata = {"event_refined": True, **changes.pop("metadata", {})}
    return _miss(burn_dv, None, minimum_distance_km=km, metadata=metadata, **changes)


def test_unscored_miss_choices_fly_the_closest_misses_nearer_than_the_zero_burn() -> None:
    """Step 4a's search has no coefficients: R5 verified only its zero burn (3049 km)
    and left a 117 km miss, 53.11 points at C_r 300 against 23.67, unflown."""
    config = _config(verify_misses=2)
    assert config.score is None
    zero = _flown(3049.0, 0.0, burns=(), branch="zero-burn")
    near, middle, far = _flown(117.0, 0.1), _flown(500.0, 0.11), _flown(1000.0, 0.12)
    beyond = _flown(4000.0, 0.13)
    unrefined = _miss(0.14, None, minimum_distance_km=1.0)
    outside = _flown(50.0, 0.2499)  # over the 0.249 km/s production margin
    hit = _flown(1.0, 0.15, successful=True)
    copy = dataclasses.replace(
        near, candidate_id="", burns=(dataclasses.replace(near.burns[0], slot_index=1),),
    )
    # Zero burn first, as the unscored order puts it; then by delta-v.
    pool = [zero, hit, far, middle, beyond, unrefined, outside, copy, near]
    assert _choices(pool, config) == [copy, middle]
    assert _choices(pool, _config(verify_misses=5)) == [copy, middle, far]
    # No zero burn: every refined miss is nearer than nothing.
    assert _choices([beyond, far], config) == [far, beyond]
    assert _choices(pool, _config(verify_misses=0)) == []


def _verified_miss(km: float, burn_dv: float, **changes: Any) -> Candidate:
    candidate = _flown(km, burn_dv, **changes)
    candidate.metadata["scored_miss"] = True
    candidate.promote(CandidateStage.PUBLIC_GMAT_VERIFIED, run_id="test")
    return candidate


def test_closest_certifiable_miss_is_the_nearest_verified_miss() -> None:
    from orbitwin.pipeline import closest_certifiable_miss, closest_miss_line

    def _closest(eligible: list[Candidate], alignment_km: float | None = None) -> Any:
        return closest_certifiable_miss(eligible, alignment_minimum_distance_km=alignment_km)

    zero = _verified_miss(3049.0, 0.0, burns=(), branch="zero-burn")
    near, far = _verified_miss(117.0, 0.1), _verified_miss(500.0, 0.05)
    unverified = _flown(10.0, 0.1)
    beyond = _verified_miss(4000.0, 0.01)
    hit = _verified_miss(1.0, 0.2, successful=True)
    assert not unverified.verified_by_gmat and near.verified_by_gmat and hit.verified_by_gmat
    assert _closest([zero, hit, far, near, unverified, beyond]) is near
    # A tie on distance goes to the one spending less, whatever the ids say.
    cheap = next(c for c in (_verified_miss(117.0, dv) for dv in (0.09, 0.08, 0.07, 0.06))
                 if c.candidate_id > near.candidate_id)
    assert _closest([zero, near, cheap]) is cheap
    assert _closest([zero, beyond]) is None
    assert _closest([zero]) is None
    # Step 4a's net does not lean on a head start: final-02 is never a head-start miss.
    early = _verified_miss(50.0, 0.1)
    early = dataclasses.replace(
        early, candidate_id="", burns=(dataclasses.replace(early.burns[0], time_s=-100.0),))
    early.promote(CandidateStage.PUBLIC_GMAT_VERIFIED, run_id="test")
    assert early.verified_by_gmat
    assert _closest([zero, near, early]) is near
    assert _closest([beyond]) is beyond
    line = closest_miss_line({"closest_certifiable_miss": near.candidate_id,
                              "best_certifiable_without_shallow_opt_in": zero.candidate_id})
    assert line == (f"the closest verified miss is {near.candidate_id}; RUNBOOK step 4a "
                    f"certifies it beside {zero.candidate_id}")
    assert closest_miss_line({"closest_certifiable_miss": None}) is None


def test_an_unscored_solve_verifies_and_names_the_closest_miss(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Through the CLI, as step 4 runs it: the miss is flown and named for step 4a."""
    import orbitwin.cli as cli

    offered = _offer_misses(monkeypatch, [40.0, 20.0, 60.0])
    config = _config(correct_top=0, verify_top=0, verify_misses=2)
    assert config.score is None
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    out = tmp_path / "solve"
    code = cli.main(["solve", str(_missing(tmp_path, no_maneuver_script, 6.0)), "--config",
                     "given", "--output", str(out), "--gmat-console", str(fake_gmat),
                     "--startup-file", str(fake_startup)])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text("utf-8"))
    stages = _stages(out)
    first, second, third = offered
    assert stages[second.candidate_id] == stages[first.candidate_id] == "public_gmat_verified"
    assert stages[third.candidate_id] == "renderable"
    assert [r["candidate_id"] for r in summary["gmat_miss_verifications"]] == [
        second.candidate_id, first.candidate_id]
    # The fake flies every plan 6 km off, the zero burn too: none passes nearer.
    assert summary["best_candidate"]["branch"] == "zero-burn"
    assert summary["closest_certifiable_miss"] is None
    assert "closest verified miss" not in capsys.readouterr().err


def test_an_unscored_solve_names_a_verified_miss_when_the_zero_burn_is_farther(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import orbitwin.cli as cli
    import orbitwin.pipeline as pipeline

    offered = _offer_misses(monkeypatch, [20.0])
    real_verify = pipeline.verify_candidate_with_gmat

    def verify(script_path: Path, scenario: Any, document: Any, candidate: Candidate,
               *args: Any, **kwargs: Any) -> Any:
        result = real_verify(script_path, scenario, document, candidate, *args, **kwargs)
        if candidate.branch == "zero-burn":
            # As on R5: GMAT's zero burn passes 3049 km off, the miss 117 km.
            candidate.minimum_distance_km = 3049.0
        return result

    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    config = _config(correct_top=0, verify_top=0, verify_misses=1)
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    out = tmp_path / "solve"
    assert cli.main(["solve", str(_missing(tmp_path, no_maneuver_script, 6.0)), "--config",
                     "given", "--output", str(out), "--gmat-console", str(fake_gmat),
                     "--startup-file", str(fake_startup)]) == 0
    summary = json.loads((out / "summary.json").read_text("utf-8"))
    (miss,) = offered
    best = summary["best_certifiable_without_shallow_opt_in"]
    assert summary["best_candidate"]["branch"] == "zero-burn"
    assert summary["closest_certifiable_miss"] == miss.candidate_id != best
    printed = [line for line in capsys.readouterr().err.splitlines()
               if "closest verified miss" in line]
    assert printed == [f"orbitwin solve: the closest verified miss is {miss.candidate_id}; "
                       f"RUNBOOK step 4a certifies it beside {best}"]


def test_from_origin_misses_are_compared_on_the_distance_the_term_prices() -> None:
    """Under from_origin every miss is scored on min(d, the alignment's minimum)
    (score.scored_distance_km); one that only ties the zero burn there is not nearer."""
    from orbitwin.pipeline import closest_certifiable_miss

    config = _config(verify_misses=2)
    zero = _flown(3049.0, 0.0, burns=(), branch="zero-burn")
    near, middle = _flown(117.0, 0.1), _flown(500.0, 0.11)
    assert _choices([zero, middle, near], config, 50.0) == []
    assert _choices([zero, middle, near], config, 300.0) == [near]
    assert _choices([zero, middle, near], config, None) == [near, middle]
    # An unmeasured closest approach is no nearer under a window either.
    unrefined = _miss(0.12, None, minimum_distance_km=1.0)
    assert _choices([unrefined], config, 300.0) == []
    flown = [_verified_miss(3049.0, 0.0, burns=(), branch="zero-burn"),
             _verified_miss(117.0, 0.1), _verified_miss(500.0, 0.11)]
    assert closest_certifiable_miss(flown, alignment_minimum_distance_km=50.0) is None
    assert closest_certifiable_miss(flown, alignment_minimum_distance_km=300.0) is flown[1]


def _solve_with_a_farther_zero_burn(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, *, alignment_km: float | None = None,
    flown_miss_km: float | None = None,
) -> tuple[dict[str, Any], Candidate]:
    """One offered miss, 20 km off in two-body; the fake flies it 6 km off, or
    `flown_miss_km`, and the zero burn 3049 km off."""
    import orbitwin.pipeline as pipeline

    offered = _offer_misses(monkeypatch, [20.0])
    real_verify = pipeline.verify_candidate_with_gmat

    def verify(script_path: Path, scenario: Any, document: Any, candidate: Candidate,
               *args: Any, **kwargs: Any) -> Any:
        result = real_verify(script_path, scenario, document, candidate, *args, **kwargs)
        if candidate.branch == "zero-burn":
            candidate.minimum_distance_km = 3049.0
        elif flown_miss_km is not None and candidate in offered:
            candidate.minimum_distance_km = flown_miss_km
        return result

    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    if alignment_km is not None:
        real_load = pipeline.load_scenario

        def load(*args: Any, **kwargs: Any) -> Any:
            scenario, document, probe = real_load(*args, **kwargs)
            # As policy.closest_approach_window from_origin records it.
            return (dataclasses.replace(scenario, alignment_minimum_distance_km=alignment_km),
                    document, probe)

        monkeypatch.setattr(pipeline, "load_scenario", load)
    out = tmp_path / "solve"
    summary, _ = _solve(_missing(tmp_path, no_maneuver_script, 6.0),
                        _config(correct_top=0, verify_top=0, verify_misses=1), out,
                        fake_gmat, fake_startup)
    (miss,) = offered
    return summary, miss


def test_solve_names_no_miss_that_only_ties_the_zero_burn_under_from_origin(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, miss = _solve_with_a_farther_zero_burn(
        tmp_path, no_maneuver_script, fake_gmat, fake_startup, monkeypatch, alignment_km=1.0)
    assert _stages(tmp_path / "solve")[miss.candidate_id] == "renderable"
    assert summary["gmat_miss_verifications"] == []
    assert summary["closest_certifiable_miss"] is None


def test_solve_names_no_flown_miss_that_only_ties_the_zero_burn_under_from_origin(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nearer in two-body (20 km against the 50 km window), so flown; GMAT's
    100 km then scores 50 km, as the zero burn does."""
    summary, miss = _solve_with_a_farther_zero_burn(
        tmp_path, no_maneuver_script, fake_gmat, fake_startup, monkeypatch,
        alignment_km=50.0, flown_miss_km=100.0)
    assert _stages(tmp_path / "solve")[miss.candidate_id] == "public_gmat_verified"
    assert summary["closest_certifiable_miss"] is None


def test_solve_names_no_verified_miss_the_gate_withholds(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flown, but through the central body: certify would refuse it."""
    import orbitwin.pipeline as pipeline

    offered: list[str] = []
    real = pipeline.flies_through_central_body

    def through(candidate: Candidate, config: OrbitWinConfig) -> bool:
        return candidate.candidate_id in offered or real(candidate, config)

    monkeypatch.setattr(pipeline, "flies_through_central_body", through)
    real_choices = pipeline.miss_choices

    def choices(*args: Any, **kwargs: Any) -> list[Candidate]:
        chosen = real_choices(*args, **kwargs)
        offered.extend(c.candidate_id for c in chosen)
        return chosen

    monkeypatch.setattr(pipeline, "miss_choices", choices)
    summary, miss = _solve_with_a_farther_zero_burn(
        tmp_path, no_maneuver_script, fake_gmat, fake_startup, monkeypatch)
    assert offered == [miss.candidate_id]
    assert _stages(tmp_path / "solve")[miss.candidate_id] == "public_gmat_verified"
    assert summary["closest_certifiable_miss"] is None


def _flaky(tmp_path: Path, fake_gmat: Path, *runs: str) -> Path:
    """The fake console, exiting 3 as a crash or a timeout would, for a run under any of `runs`."""
    console = tmp_path / "flaky_console.py"
    console.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        f"if any(run in place for run in {list(runs)!r} for place in [os.getcwd(), *sys.argv]):\n"
        "    sys.exit(3)\n"
        f"sys.exit(subprocess.call([sys.executable, {str(fake_gmat)!r}, *sys.argv[1:]]))\n",
        encoding="utf-8",
    )
    console.chmod(0o755)
    return console


def _clean_run(tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path,
               fake_startup: Path) -> tuple[Path, OrbitWinConfig, Candidate, list[str]]:
    """The script, its config, the zero burn the fake hits and wins with, and the
    plans corrected, best first: the R1 shape."""
    script = _script(tmp_path, no_maneuver_script)
    config = _scored(correct_top=2, verify_top=0)
    summary, candidates = _solve(script, config, tmp_path / "clean", fake_gmat, fake_startup)
    assert summary["gmat_failures"] == {"warning": None, "runs": []}
    (zero,) = [c for c in candidates if c.branch == "zero-burn"]
    assert summary["best_candidate"]["candidate_id"] == zero.candidate_id
    return script, config, zero, [r["candidate_id"] for r in summary["gmat_corrections"]]


def test_solve_names_the_gmat_runs_that_raised_and_which_beat_its_choice(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """R2 H1: the best plans' GmatConsole runs exited 3, and solve chose the next one
    with exit 0, no warning and nothing on stderr."""
    import orbitwin.cli as cli

    script, config, zero, (lost, kept) = _clean_run(
        tmp_path, no_maneuver_script, fake_gmat, fake_startup
    )
    console = _flaky(tmp_path, fake_gmat, f"gmat_verification/{zero.candidate_id}",
                     f"gmat_correction/{lost}")
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    out = tmp_path / "flaky"
    assert cli.main(["solve", str(script), "--config", "given", "--output", str(out),
                     "--gmat-console", str(console), "--startup-file", str(fake_startup)]) == 0
    summary = json.loads((out / "summary.json").read_text("utf-8"))
    assert summary["best_candidate"]["candidate_id"] == kept
    failures = summary["gmat_failures"]
    # By the score each was chosen by: the zero burn's two-body one is a far miss.
    assert [(r["kind"], r["candidate_id"], r["above_best"], r["in_archive"])
            for r in failures["runs"]] == [
        ("correction", lost, True, True), ("verification", zero.candidate_id, False, True)]
    correction, verification = failures["runs"]
    assert correction["error"] == (
        "round 1 (flight) failed: GmatRunError: GmatConsole returned exit status 3")
    assert verification["error"] == "GmatRunError: GmatConsole returned exit status 3"
    (record,) = [r for r in summary["gmat_corrections"] if r["candidate_id"] == lost]
    assert record["error"] == correction["error"]
    warning = failures["warning"]
    assert warning in summary["warnings"]
    assert f"orbitwin solve: WARNING: {warning}" in capsys.readouterr().err.splitlines()
    assert f"the chosen best is {kept} (pre-GMAT score" in warning
    assert "1 of them scored above it before GMAT" in warning
    assert f": ABOVE THE BEST: correction of {lost} (" in warning
    assert f"; verification of {zero.candidate_id} (zero-burn, " in warning
    assert "run solve again, or `orbitwin certify --candidate <id>`" in warning


def test_rank_names_a_verification_that_raised_above_its_best_row(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The hit whose verification crashed ranks below the zero burn GMAT flew, so no
    other warning names it; before GMAT it scored above the zero burn's two-body miss."""
    import orbitwin.cli as cli

    script, config, zero, (lost, _) = _clean_run(
        tmp_path, no_maneuver_script, fake_gmat, fake_startup
    )
    monkeypatch.setattr(cli, "load_config", lambda _path: config)

    def rank(archive: Path) -> tuple[dict[str, Any], str]:
        capsys.readouterr()
        assert cli.main(["rank", str(script), "--config", "given", "--candidates",
                         str(archive / "candidates.json"), "--output",
                         str(archive / "rank.json")]) == 0
        return json.loads((archive / "rank.json").read_text("utf-8")), capsys.readouterr().err

    clean, err = rank(tmp_path / "clean")
    assert clean["gmat_failures"]["warning"] is None and "raised" not in err
    console = _flaky(tmp_path, fake_gmat, f"gmat_verification/{lost}")
    out = tmp_path / "flaky"
    summary, _ = _solve(script, config, out, console, fake_startup)
    assert summary["best_candidate"]["candidate_id"] == zero.candidate_id
    report, err = rank(out)
    assert report["ceilings"]["best_submittable"] == zero.candidate_id
    row = next(r for r in report["ranking"] if r["candidate_id"] == lost)
    assert not row["submittable"] and row["score"] < report["winner"]["score"]
    assert report["winner"]["candidate_id"] == zero.candidate_id
    (run,) = [r for r in report["gmat_failures"]["runs"] if r["candidate_id"] == lost]
    assert run["above_best"] and run["kind"] == "verification"
    warning = report["gmat_failures"]["warning"]
    assert warning in report["warnings"]
    assert f"orbitwin rank: WARNING: {warning}" in err.splitlines()
    assert f"the best submittable row is {zero.candidate_id}" in warning
    assert f"ABOVE THE BEST: verification of {lost}" in warning


#: The fake console, staged per run: a run under a `crash` path exits 3; the submitted
#: script's stopped flight misses by 20 km under a `stop_outside` path, and under a
#: `stop_leaves` path passes the target at 0.5 s, leaving the sphere before its stop; the
#: reports of a run under a `truncate` path keep their first third, and those of the stopped
#: flight under a `drop_last_row` path lose their last row, with exit 0.
_STAGED = r'''#!{python}
import os, subprocess, sys
from pathlib import Path
CRASH, TRUNCATE, DROP_LAST_ROW = {crash!r}, {truncate!r}, {drop_last_row!r}
STOPPED = [({stop_outside!r}, "% FAKE_GMAT_MISS_KM 20"), ({stop_leaves!r}, "% FAKE_GMAT_FLYBY 0.5 1")]
places = [os.getcwd(), *sys.argv]
def under(paths):
    return any(path in place for path in paths for place in places)
if under(CRASH):
    sys.exit(3)
argv = sys.argv[1:]
run = argv.index("--run") + 1
for paths, directive in STOPPED:
    if under(paths) and "_at_stop" in argv[run]:
        script = Path(argv[run])
        moved = script.with_name("staged_" + script.name)
        moved.write_text(directive + "\n" + script.read_text(encoding="utf-8"), encoding="utf-8")
        argv[run] = str(moved)
code = subprocess.call([sys.executable, {fake!r}, *argv])
cut = 3 if under(TRUNCATE) else 0
if under(DROP_LAST_ROW) and "_at_stop" in argv[run]:
    cut = -1
if cut:
    startup = Path(argv[argv.index("--startup_file") + 1])
    output = next(line.split("=", 1)[1].strip() for line in startup.read_text().splitlines()
                  if line.strip().upper().startswith("OUTPUT_PATH"))
    for report in Path(output).glob("*.txt"):
        if not report.name.lower().startswith("gmatlog"):
            rows = report.read_text().splitlines(True)
            report.write_text("".join(rows[:-1] if cut < 0 else rows[: max(2, len(rows) // 3)]))
sys.exit(code)
'''


def _staged(tmp_path: Path, fake_gmat: Path, *, crash: tuple[str, ...] = (),
            stop_outside: tuple[str, ...] = (), stop_leaves: tuple[str, ...] = (),
            truncate: tuple[str, ...] = (), drop_last_row: tuple[str, ...] = ()) -> Path:
    console = tmp_path / "staged_console.py"
    console.write_text(_STAGED.format(python=sys.executable, fake=str(fake_gmat), crash=crash,
                                      stop_outside=stop_outside, stop_leaves=stop_leaves,
                                      truncate=truncate, drop_last_row=drop_last_row),
                       encoding="utf-8")
    console.chmod(0o755)
    return console


def _verify_zero(tmp_path: Path, script: Path, config: OrbitWinConfig, console: Path,
                 fake_startup: Path) -> Any:
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.search import zero_burn_candidate
    from orbitwin.verify import verify_candidate_with_gmat

    scenario, document = parse_gmat_script(script, config)
    zero = zero_burn_candidate(scenario, config)
    return verify_candidate_with_gmat(script, scenario, document, zero, config, console,
                                      fake_startup, tmp_path / "verify")


@pytest.mark.parametrize("verdict", ["floor", "constraint", "stop-no-entry", "stop-outside"])
def test_verify_tags_gmats_verdicts_apart_from_failed_runs(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path, verdict: str,
) -> None:
    """R2 #187 review F1: each refusal of a flight raises a GmatVerdictError."""
    from orbitwin.errors import (
        AltitudeFloorError,
        ConstraintVerdictError,
        GmatVerdictError,
        StopVerdictError,
    )

    script = _script(tmp_path, no_maneuver_script)
    config = _scored()
    console = fake_gmat
    expected: type[Exception] = StopVerdictError
    if verdict == "floor":
        config = dataclasses.replace(config, policy=dataclasses.replace(
            config.policy, minimum_flown_altitude_km=5000.0))
        expected = AltitudeFloorError
    elif verdict == "constraint":
        # The preliminary rules require a hit, and GMAT flies it 6 km off.
        script = _missing(tmp_path, no_maneuver_script)
        config = dataclasses.replace(config, policy=dataclasses.replace(
            config.policy, rules="preliminary"))
        expected = ConstraintVerdictError
    elif verdict == "stop-no-entry":
        console = _staged(tmp_path, fake_gmat, stop_outside=("verify",))
    else:
        console = _staged(tmp_path, fake_gmat, stop_leaves=("verify",))
    with pytest.raises(expected, match="no longer enters" if verdict == "stop-no-entry"
                       else "outside the 5 km sphere" if verdict == "stop-outside" else "") as raised:
        _verify_zero(tmp_path, script, config, console, fake_startup)
    assert isinstance(raised.value, GmatVerdictError)
    # A crash in the same flight is no verdict.
    with pytest.raises(GmatRunError) as crashed:
        _verify_zero(tmp_path / "crash", script, _scored(),
                     _staged(tmp_path, fake_gmat, crash=("verify",)), fake_startup)
    assert not isinstance(crashed.value, GmatVerdictError)


def test_a_stopped_flight_missing_its_last_row_is_a_cut_report_not_a_verdict(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
) -> None:
    """Delta review of #187: only that flight is cut, and it then never enters, which
    raised as GMAT's "no longer enters" verdict and was hidden from the warning."""
    from orbitwin.errors import GmatVerdictError

    script = _script(tmp_path, no_maneuver_script)
    console = _staged(tmp_path, fake_gmat, drop_last_row=("verify",))
    with pytest.raises(GmatRunError, match="the report was cut short") as raised:
        _verify_zero(tmp_path, script, _scored(), console, fake_startup)
    assert not isinstance(raised.value, GmatVerdictError)
    assert "diagnostic-run-at-stop" in str(raised.value)


def test_solve_lists_a_crash_and_a_cut_report_but_no_verdict(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2 #187 review F1, F2: on R3 and R4 eight altitude floors were listed as crashes
    ABOVE THE BEST; and a report cut short with exit 0 was taken as a verified miss."""
    import orbitwin.pipeline as pipeline
    from orbitwin.errors import AltitudeFloorError, ConstraintVerdictError

    script = _script(tmp_path, no_maneuver_script)
    config = _scored(correct_top=4, verify_top=0)
    summary, candidates = _solve(script, config, tmp_path / "clean", fake_gmat, fake_startup)
    assert summary["gmat_failures"]["runs"] == []
    (zero,) = [c for c in candidates if c.branch == "zero-burn"]
    corrected = [r["candidate_id"] for r in summary["gmat_corrections"]]
    assert len(corrected) == 4
    stop, floor, constraint, crash = corrected
    real = pipeline.verify_candidate_with_gmat

    def verify(script_path: Path, scenario: Any, document: Any, candidate: Candidate,
               *args: Any, **kwargs: Any) -> Any:
        if candidate.candidate_id == floor:
            raise AltitudeFloorError("GMAT flew it down to 95.4 km altitude")
        if candidate.candidate_id == constraint:
            raise ConstraintVerdictError("GMAT candidate failed constraints: intercept after Tmax")
        return real(script_path, scenario, document, candidate, *args, **kwargs)

    monkeypatch.setattr(pipeline, "verify_candidate_with_gmat", verify)
    console = _staged(tmp_path, fake_gmat, crash=(f"gmat_verification/{crash}",),
                      stop_leaves=(f"gmat_verification/{stop}",),
                      truncate=(f"gmat_verification/{zero.candidate_id}",))
    out = tmp_path / "staged"
    summary, candidates = _solve(script, config, out, console, fake_startup)
    by_id = {c.candidate_id: c for c in candidates}
    for verdict in (stop, floor, constraint):
        assert by_id[verdict].metadata["gmat_verification_verdict"] is True
        assert by_id[verdict].metadata["gmat_verification_error"]
    for failed in (crash, zero.candidate_id):
        assert by_id[failed].metadata["gmat_verification_verdict"] is False
    runs = summary["gmat_failures"]["runs"]
    assert sorted(r["candidate_id"] for r in runs) == sorted([crash, zero.candidate_id])
    (cut,) = [r for r in runs if r["candidate_id"] == zero.candidate_id]
    assert "the report was cut short" in cut["error"]
    warning = summary["gmat_failures"]["warning"]
    assert warning.startswith("2 GMAT correction(s) or verification(s) raised")
    assert all(verdict not in warning for verdict in (stop, floor, constraint))
    # Nor does rank, which reads the same tag from the archive.
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.rank import rank_report

    scenario, _ = parse_gmat_script(script, config)
    report = rank_report(scenario, read_candidates(out / "candidates.json"), config)
    assert all(r["candidate_id"] in (crash, zero.candidate_id)
               for r in report["gmat_failures"]["runs"])
