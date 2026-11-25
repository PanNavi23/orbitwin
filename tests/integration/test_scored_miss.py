from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
from conftest import TEAM_NUMBER, quick_config, ranking_copy, stamped

from orbitwin.config import OrbitWinConfig, ScoreConfig
from orbitwin.errors import OrbitWinError, PolicyError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.pipeline import _distinct_trajectories_first, _sort_key, solve_scenario
from orbitwin.rules import rule_set_record
from orbitwin.search import search_one_burn
from orbitwin.verify import verify_candidate_with_gmat

SCORE = ScoreConfig(kt=0.0001, ct=20000.0, kv=1.0, cv=1.5, delta_v_unit="km/s")
#: The same terms as the ranking rules revised 2026/11/04 take them: m/s, and a C_r.
RANKING_SCORE = ScoreConfig(kt=0.0001, ct=20000.0, kv=0.001, cv=1500.0, delta_v_unit="m/s",
                            cr_km=100.0)


@pytest.fixture()
def workshop_script(workshop_script: Path, tmp_path: Path) -> Path:
    """The workshop with the ranking report name; certify refuses it without one."""
    return ranking_copy(workshop_script, tmp_path)


def _config(rules: str) -> OrbitWinConfig:
    base = quick_config()
    policy = dataclasses.replace(base.policy, rules=rules, horizon_s=30000.0)
    if rules == "ranking":
        policy = dataclasses.replace(policy, team_number=TEAM_NUMBER)
    if rules == "ranking":
        # solve and certify refuse the preliminary 1.5 km/s under these rules.
        policy = dataclasses.replace(
            policy, dv_limit_km_s=0.25, dv_safe_limit_km_s=0.249,
            total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99,
        )
    return dataclasses.replace(base, policy=policy,
                               score=RANKING_SCORE if rules == "ranking" else SCORE)


@pytest.fixture()
def near_script(workshop_script: Path, tmp_path: Path) -> Path:
    """The workshop template with AlienSC just ahead of HumanSC.

    The template's own one-burn transfers all exceed the ranking round's
    0.25 km/s; here the cheapest hit costs about 0.02 km/s.
    """
    text = workshop_script.read_text(encoding="utf-8")
    for old, new in (
        ("AlienSC.SMA = 9600;", "AlienSC.SMA = 6950;"),
        ("AlienSC.TA = 311.265;", "AlienSC.TA = 250.766;"),
    ):
        assert old in text
        text = text.replace(old, new)
    script = tmp_path / "near.script"
    script.write_text(text, encoding="utf-8")
    return script


def _missing_by(base: Path, tmp_path: Path, miss_km: float) -> Path:
    """`base`, with the fake console told to miss by `miss_km`."""
    script = tmp_path / "miss.script"
    script.write_text(
        f"% FAKE_GMAT_MISS_KM {miss_km}\n" + base.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return script


def _verify(script: Path, config: OrbitWinConfig, fake_gmat: Path, fake_startup: Path,
            out: Path):
    scenario, document = parse_gmat_script(script, config)
    candidate = search_one_burn(scenario, config)[0]
    return verify_candidate_with_gmat(
        script, scenario, document, candidate, config, fake_gmat, fake_startup, out
    )


def test_the_ranking_rules_verify_a_miss_and_score_its_closest_approach(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _missing_by(near_script, tmp_path, 6.0)
    verified, _, evidence = _verify(script, _config("ranking"), fake_gmat, fake_startup,
                                    tmp_path / "verify")
    assert evidence["event"]["first_entry_time_s"] is None
    assert verified.successful is False
    assert verified.verified_by_gmat is True
    assert verified.score_components["distance_score"] == pytest.approx(
        100.0 / (1.0 + math.exp(((6.0 - 5.0) / 100.0) ** 2))
    )


def test_the_preliminary_rules_still_refuse_a_miss(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _missing_by(near_script, tmp_path, 6.0)
    with pytest.raises(PolicyError, match="not a verified successful interception"):
        _verify(script, _config("preliminary"), fake_gmat, fake_startup, tmp_path / "verify")


def test_a_hit_is_unaffected_by_the_rulebook(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    for rules in ("preliminary", "ranking"):
        verified, _, _ = _verify(near_script, _config(rules), fake_gmat, fake_startup,
                                 tmp_path / rules)
        assert verified.successful and verified.verified_by_gmat
        assert verified.metadata["scored_miss"] is False


def test_a_better_scored_miss_is_ordered_and_chosen_ahead_of_a_hit(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    config = _config("ranking")
    scenario, _ = parse_gmat_script(near_script, config)
    hit = search_one_burn(scenario, config)[0]
    miss = dataclasses.replace(hit, candidate_id="miss", successful=False)
    hit.score, miss.score = 60.0, 70.0
    ranked = sorted(
        [hit, miss],
        key=lambda c: _sort_key(c, scenario=scenario, scored=True, misses_count=True),
    )
    assert ranked[0] is miss
    preliminary = sorted([hit, miss], key=lambda c: _sort_key(c, scenario=scenario, scored=True))
    assert preliminary[0] is hit
    assert _distinct_trajectories_first([miss, hit], 1, misses_count=True) == [miss]
    assert _distinct_trajectories_first([miss, hit], 1) == [hit]


@pytest.mark.parametrize("rules, chosen", [("ranking", True), ("preliminary", False)])
def test_solve_names_a_verified_miss_only_under_the_ranking_rules(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path,
    rules: str, chosen: bool,
) -> None:
    script = _missing_by(near_script, tmp_path, 6.0)
    summary = solve_scenario(
        script, _config(rules), tmp_path / "solve", render_top=2,
        gmat_console=fake_gmat, startup_file=fake_startup, gmat_verify_top=2,
    )
    assert (summary["best_candidate"] is not None) is chosen
    if chosen:
        archived = json.loads((tmp_path / "solve" / "candidates.json").read_text("utf-8"))
        best = next(c for c in archived if c["candidate_id"] == summary["best_candidate"]["candidate_id"])
        assert best["successful"] is False and best["verified_by_gmat"] is True


def _planned_miss(candidate):
    """`candidate` as the archive holds a miss: no entry, scored on its closest approach."""
    candidate.successful = False
    candidate.intercept_time_s = None
    candidate.minimum_distance_km = 6.0
    candidate.distance_at_completion_km = 6.0
    return candidate


@pytest.mark.parametrize("rules", ["ranking", "preliminary"])
def test_a_miss_is_certified_only_under_the_ranking_rules(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path, rules: str
) -> None:
    from orbitwin.verify import certify_candidate

    script = _missing_by(near_script, tmp_path, 6.0)
    config = _config(rules)
    scenario, _ = parse_gmat_script(script, config)
    candidate = _planned_miss(search_one_burn(scenario, config)[0])
    stamped(candidate, script, config)
    if rules == "preliminary":
        with pytest.raises(PolicyError):
            certify_candidate(script, candidate, config, fake_gmat, fake_startup,
                              tmp_path / "certify", runs=2)
        return
    evidence = certify_candidate(script, candidate, config, fake_gmat, fake_startup,
                                 tmp_path / "certify", runs=2)
    certification = json.loads(
        (tmp_path / "certify" / "submission" / "certification.json").read_text("utf-8")
    )
    assert certification["final_script_parity"]["first_entry_time_s"] is None
    assert certification["rule_set"] == rule_set_record("ranking")
    assert certification["planned_hit_certified_as_miss"] is False
    assert certification["gmat_comparison"]["message"] is None
    assert certification["lifted_first_flight_contradiction"] is None
    assert evidence["stage"] == "public_gmat_verified"


#: _config("ranking") as the CLI reads it: RANKING_SCORE.
RANKING_PROFILE = (
    "policy:\n  rules: ranking\n  horizon_s: 30000.0\n  dv_limit_km_s: 0.25\n"
    "  dv_safe_limit_km_s: 0.249\n  total_dv_limit_km_s: 3.0\n"
    f"  total_dv_safe_limit_km_s: 2.99\n  team_number: {TEAM_NUMBER}\n"
    "  mission_time_origin: earliest\n"
    "score:\n  kt: 0.0001\n  ct: 20000.0\n  kv: 0.001\n  cv: 1500.0\n  cr_km: 100.0\n"
    "  time_unit: s\n  delta_v_unit: m/s\n"
)
QUICK_SEARCH = (
    "search:\n  one_burn_samples: 0\n  one_burn_grid_burns: 2\n  one_burn_grid_arrivals: 2\n"
    "  max_revolutions: 0\n  keep_candidates: 20\n  render_top: 3\n  event_step_s: 20.0\n"
    "  event_refine_top: 3\n  two_burn_enabled: false\n  two_burn_samples: 0\n"
)

#: R4's C-284837d0b0cc6be5 (2026-11-04): a hit in the archive that rank called the best
#: submittable, 42.166 km off on GMAT's rows, certified as a miss with exit 0.
R4_GMAT_MISS_KM = 42.165818


def _planned_hit(script: Path, config: OrbitWinConfig):
    scenario, _ = parse_gmat_script(script, config)
    candidate = stamped(search_one_burn(scenario, config)[0], script, config)
    assert candidate.successful and candidate.intercept_time_s is not None
    return candidate


def _certify(script: Path, candidate, config: OrbitWinConfig, fake_gmat: Path,
             fake_startup: Path, output: Path, **options):
    from orbitwin.verify import certify_candidate

    return certify_candidate(script, candidate, config, fake_gmat, fake_startup, output,
                             runs=2, **options)


def test_certify_refuses_a_hit_that_gmat_flies_as_a_miss(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _missing_by(near_script, tmp_path, R4_GMAT_MISS_KM)
    config = _config("ranking")
    candidate = _planned_hit(script, config)
    planned_s = candidate.intercept_time_s
    output = tmp_path / "certify"
    with pytest.raises(PolicyError) as refused:
        _certify(script, candidate, config, fake_gmat, fake_startup, output)
    message = str(refused.value)
    assert f"the archive holds a hit, first inside the 5 km sphere at {planned_s:.3f} s" in message
    assert "GMAT now flies a miss, closest approach 42.166 km" in message
    assert "the archive's claim is" in message and "the script certify would submit" in message
    assert "Solve never flew it in GMAT" in message
    assert "--accept-gmat-loss" in message
    # Refused on the verification flight, kept aside, before any exact run or submission.
    kept = tmp_path / "certify.refused"
    assert str(kept / "diagnostic" / "verification.json") in message
    assert not output.exists() and (kept / "diagnostic" / "verification.json").is_file()
    assert not (kept / "exact-run-01").exists() and not (kept / "submission").exists()
    # The same command with the opt-in, into the same --output.
    with pytest.warns(RuntimeWarning, match="will be certified as GMAT flies it if the "
                                            "remaining checks pass"):
        evidence = _certify(script, _planned_hit(script, config), config, fake_gmat,
                            fake_startup, output, accept_gmat_loss=True)
    assert evidence["planned_hit_certified_as_miss"] is True
    assert "42.166 km" in evidence["accepted_gmat_loss"]
    assert evidence["candidate"]["successful"] is False
    assert evidence["final_script_parity"]["first_entry_time_s"] is None
    assert evidence["final_script_parity"]["minimum_distance_km"] == pytest.approx(
        R4_GMAT_MISS_KM
    )
    assert evidence["candidate"]["metadata"]["analytical_before_gmat"]["successful"] is True


def test_certify_refuses_a_hit_whose_gmat_verification_failed_in_solve(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """solve clears `successful` when verification raises; the plan was still a hit."""
    script = _missing_by(near_script, tmp_path, R4_GMAT_MISS_KM)
    config = _config("ranking")
    candidate = _planned_hit(script, config)
    candidate.successful = False
    candidate.metadata["analytical_successful"] = True
    candidate.metadata["gmat_verification_error"] = "GmatConsole timed out after 300.0 s"
    with pytest.raises(PolicyError) as refused:
        _certify(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c")
    assert "the archive holds a hit" in str(refused.value)
    assert "verification of it failed (GmatConsole timed out after 300.0 s)" in str(refused.value)
    candidate = _planned_hit(script, config)
    candidate.successful = False
    candidate.metadata["analytical_successful"] = True
    candidate.metadata["gmat_verification_error"] = "GmatConsole timed out after 300.0 s"
    evidence = _certify(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c",
                        accept_gmat_loss=True)
    assert evidence["planned_hit_certified_as_miss"] is True
    assert evidence["candidate"]["metadata"]["analytical_before_gmat"]["successful"] is True


def test_certify_says_when_a_verified_hit_now_flies_as_a_miss(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _missing_by(near_script, tmp_path, R4_GMAT_MISS_KM)
    config = _config("ranking")
    entry_s = _planned_hit(script, config).intercept_time_s - 3.0
    plan = {"successful": True, "intercept_time_s": entry_s - 40.0,
            "minimum_distance_km": 0.0, "score": 99.0}

    def verified_in_solve():
        candidate = _planned_hit(script, config)
        candidate.gmat_run_id = "solve-run"
        candidate.metadata["gmat_event"] = {"flown_script_sha256": "0" * 64,
                                            "first_entry_time_s": entry_s}
        candidate.metadata["analytical_before_gmat"] = dict(plan)
        return candidate

    with pytest.raises(PolicyError) as refused:
        _certify(script, verified_in_solve(), config, fake_gmat, fake_startup, tmp_path / "c")
    message = str(refused.value)
    assert f"first inside the 5 km sphere at {entry_s:.3f} s and completing at" in message
    assert "GMAT's data files or build changed between the two flights" in message
    assert "certify a row rank lists" not in message
    # The plan solve recorded stands; this flight is GMAT's, not the analytical one.
    evidence = _certify(script, verified_in_solve(), config, fake_gmat, fake_startup,
                        tmp_path / "c", accept_gmat_loss=True)
    assert evidence["candidate"]["metadata"]["analytical_before_gmat"] == plan


def test_certify_refuses_a_miss_that_gmat_flies_wider(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _missing_by(near_script, tmp_path, R4_GMAT_MISS_KM)
    config = _config("ranking")
    scenario, _ = parse_gmat_script(script, config)
    candidate = stamped(_planned_miss(search_one_burn(scenario, config)[0]), script, config)
    with pytest.raises(PolicyError) as refused:
        _certify(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c")
    message = str(refused.value)
    assert "the archive holds a miss, 6.000 km from the target at closest" in message
    assert "GMAT now flies a miss, closest approach 42.166 km" in message


@pytest.mark.parametrize("rules", ["ranking", "preliminary"])
def test_certify_refuses_a_hit_that_gmat_enters_much_later(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path, rules: str
) -> None:
    script = tmp_path / "late.script"
    script.write_text("% FAKE_GMAT_FLYBY 20000 7\n" + near_script.read_text("utf-8"), "utf-8")
    config = _config(rules)
    with pytest.raises(PolicyError) as refused:
        _certify(script, _planned_hit(script, config), config, fake_gmat, fake_startup,
                 tmp_path / "c")
    assert "GMAT now flies a hit first inside at 1999" in str(refused.value)
    # Without coefficients to price it, the late entry alone is refused.
    unpriced = dataclasses.replace(config, score=None)
    with pytest.raises(PolicyError, match=r"No coefficients are set in this config.*an entry "
                                          r"over 60 s late.*Solve never flew it in GMAT"):
        _certify(script, _planned_hit(script, unpriced), unpriced, fake_gmat, fake_startup,
                 tmp_path / "unpriced")
    planned_s = _planned_hit(script, unpriced).intercept_time_s
    evidence = _certify(script, _planned_hit(script, unpriced), unpriced, fake_gmat,
                        fake_startup, tmp_path / "unpriced", accept_gmat_loss=True)
    compared = evidence["gmat_comparison"]
    assert evidence["candidate"]["successful"] is True
    assert compared["refused_without_opt_in"] is True and evidence["accepted_gmat_loss"]
    assert compared["planned_entry_s"] == pytest.approx(planned_s)
    assert compared["flown_entry_s"] == pytest.approx(19995.0, abs=1.0)
    assert compared["late_entry_s"] == pytest.approx(compared["flown_entry_s"] - planned_s)
    assert (compared["planned_basis"], compared["flown_basis"]) == ("first entry", "completion")
    assert compared["planned_score"] is None and compared["loss_points"] is None


def test_without_coefficients_the_refusal_prices_nothing(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _missing_by(near_script, tmp_path, R4_GMAT_MISS_KM)
    config = dataclasses.replace(_config("ranking"), score=None)
    candidate = _planned_hit(script, config)
    candidate.score = 92.0
    with pytest.raises(PolicyError) as refused:
        _certify(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c")
    message = str(refused.value)
    assert "No coefficients are set in this config, so neither is priced" in message
    assert "92.0" not in message and "points" not in message


def test_without_coefficients_a_wider_miss_is_refused(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """run/day.yaml has no score block; a 6 km plan flown 42 km off was certified, exit 0."""
    script = _missing_by(near_script, tmp_path, R4_GMAT_MISS_KM)
    config = dataclasses.replace(_config("ranking"), score=None)
    scenario, _ = parse_gmat_script(script, config)
    candidate = stamped(_planned_miss(search_one_burn(scenario, config)[0]), script, config)
    with pytest.raises(PolicyError, match=r"No coefficients are set.*a miss flown more than "
                                          r"0\.1 km wider"):
        _certify(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c")
    # Inside the tolerance it goes on.
    (tmp_path / "near").mkdir()
    near = _missing_by(near_script, tmp_path / "near", 6.05)
    candidate = stamped(_planned_miss(search_one_burn(scenario, config)[0]), near, config)
    evidence = _certify(near, candidate, config, fake_gmat, fake_startup, tmp_path / "near-c")
    assert evidence["gmat_comparison"]["flown_closest_km"] == pytest.approx(6.05)


def test_a_small_loss_is_reported_and_recorded(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """0.05 to 0.5 points: certified, said aloud, and recorded; a re-flight is identical."""
    script = _missing_by(near_script, tmp_path, 6.1)
    config = _config("ranking")
    assert config.score is not None
    config = dataclasses.replace(config, score=dataclasses.replace(config.score, cr_km=5.0))
    scenario, _ = parse_gmat_script(script, config)
    candidate = stamped(_planned_miss(search_one_burn(scenario, config)[0]), script, config)
    with pytest.warns(RuntimeWarning, match="Certify goes on"):
        evidence = _certify(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c")
    compared = evidence["gmat_comparison"]
    assert 0.05 < compared["loss_points"] < 0.5
    assert compared["refused_without_opt_in"] is False and evidence["accepted_gmat_loss"] is None
    assert compared["planned_score"] - compared["flown_score"] == compared["loss_points"]


def test_a_plan_is_compared_with_the_script_certify_submits(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """GMAT reproduces the plan's entry exactly, then completes 1.5 km inside: the
    submitted script scores that completion, so the stop depth is a real loss."""
    config = _config("ranking")
    entry_s = _planned_hit(near_script, config).intercept_time_s
    script = tmp_path / "exact.script"
    script.write_text(f"% FAKE_GMAT_FLYBY {entry_s + 5.0!r} 7\n"
                      + near_script.read_text("utf-8"), "utf-8")
    assert config.score is not None
    # Steep enough that the stop's seconds alone would cost over half a point.
    steep = dataclasses.replace(config, score=dataclasses.replace(config.score, kt=0.1,
                                                                    ct=entry_s))
    with pytest.raises(PolicyError, match=r"the script certify would submit .* \(at GMAT's "
                                          r"first entry, before the stop depth"):
        _certify(script, _planned_hit(script, steep), steep, fake_gmat, fake_startup,
                 tmp_path / "c")
    evidence = _certify(script, _planned_hit(script, steep), steep, fake_gmat, fake_startup,
                        tmp_path / "c", accept_gmat_loss=True)
    compared = evidence["gmat_comparison"]
    assert (compared["planned_basis"], compared["flown_basis"]) == ("first entry", "completion")
    assert compared["loss_points"] > 0.5 and abs(compared["entry_loss_points"]) < 0.05
    # The loss is against what is submitted: the replay of the final script.
    assert compared["flown_score"] == pytest.approx(
        evidence["final_script_parity"]["replayed_score"], abs=1e-9)
    assert evidence["candidate"]["intercept_time_s"] > entry_s + 1.0


def test_every_certify_refusal_frees_the_output(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = tmp_path / "fail.script"
    script.write_text("% FAKE_GMAT_FAIL\n" + near_script.read_text("utf-8"), "utf-8")
    config = _config("ranking")
    output = tmp_path / "final"
    # A dangling link where the first refusal would go.
    (tmp_path / "final.refused").symlink_to(tmp_path / "nowhere")
    with pytest.raises(OrbitWinError, match="is free again"):
        _certify(script, _planned_hit(script, config), config, fake_gmat, fake_startup, output)
    assert not output.exists()
    assert (tmp_path / "final.refused-2" / "diagnostic").is_dir()
    assert (tmp_path / "final.refused").is_symlink()


def test_a_refusal_repeats_the_archive_warnings(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Warned at the start, they scroll away before the plots stop on the day."""
    script = tmp_path / "fail.script"
    script.write_text("% FAKE_GMAT_FAIL\n" + near_script.read_text("utf-8"), "utf-8")
    config = _config("ranking")
    candidate = _planned_hit(script, config)
    del candidate.metadata["searched_score"]
    with pytest.warns(RuntimeWarning, match="carries no searched_score"), \
            pytest.raises(OrbitWinError, match="is free again") as refused:
        _certify(script, candidate, config, fake_gmat, fake_startup, tmp_path / "final")
    assert "Archive warnings, again: " in str(refused.value)
    assert f"Candidate {candidate.candidate_id} carries no searched_score" in str(refused.value)
    stamped_again = _planned_hit(script, config)
    with pytest.raises(OrbitWinError, match="is free again") as clean:
        _certify(script, stamped_again, config, fake_gmat, fake_startup, tmp_path / "again")
    assert "Archive warnings" not in str(clean.value)


def test_the_flip_is_recorded_only_when_it_happened(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    config = _config("ranking")
    hit = _certify(near_script, _planned_hit(near_script, config), config, fake_gmat,
                   fake_startup, tmp_path / "hit", accept_gmat_loss=True)
    assert hit["candidate"]["successful"] is True
    assert hit["planned_hit_certified_as_miss"] is False and hit["accepted_gmat_loss"] is None
    script = _missing_by(near_script, tmp_path, 6.0)
    scenario, _ = parse_gmat_script(script, config)
    miss = _certify(script, stamped(_planned_miss(search_one_burn(scenario, config)[0]), script,
                                    config),
                    config, fake_gmat, fake_startup, tmp_path / "miss", accept_gmat_loss=True)
    assert miss["planned_hit_certified_as_miss"] is False and miss["accepted_gmat_loss"] is None


def test_the_opt_in_lifts_the_first_flight_refusal(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _missing_by(near_script, tmp_path, 19.859)
    config = _config("ranking")
    candidate = _planned_hit(script, config)
    candidate.metadata["gmat_first_flight_closest_km"] = 19.859
    with pytest.raises(PolicyError, match="or certify it as GMAT flies it with --accept-gmat-loss"):
        _certify(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c")
    assert not (tmp_path / "c").exists()
    with pytest.warns(RuntimeWarning, match="GMAT flew it 19.859 km"):
        evidence = _certify(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c",
                            accept_gmat_loss=True)
    assert evidence["planned_hit_certified_as_miss"] is True
    assert "GMAT flew it 19.859 km" in evidence["lifted_first_flight_contradiction"]


def test_the_preliminary_rules_refuse_that_hit_as_before(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _missing_by(near_script, tmp_path, R4_GMAT_MISS_KM)
    config = _config("preliminary")
    for opted_in in (False, True):
        with pytest.raises(PolicyError, match="not a verified successful interception"):
            _certify(script, _planned_hit(script, config), config, fake_gmat, fake_startup,
                     tmp_path / f"certify-{opted_in}", accept_gmat_loss=opted_in)


def test_the_cli_refuses_that_hit_with_exit_2(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from orbitwin.cli import main
    from orbitwin.config import load_config
    from orbitwin.io import write_candidates

    profile = tmp_path / "ranking.yaml"
    profile.write_text(RANKING_PROFILE, encoding="utf-8")
    script = _missing_by(near_script, tmp_path, R4_GMAT_MISS_KM)
    candidate = _planned_hit(script, load_config(profile))
    archive = tmp_path / "candidates.json"
    write_candidates(archive, [candidate])
    argv = [
        "certify", str(script), "--config", str(profile), "--candidates", str(archive),
        "--candidate", candidate.candidate_id, "--gmat-console", str(fake_gmat),
        "--startup-file", str(fake_startup), "--runs", "2", "--output", str(tmp_path / "final"),
    ]
    assert main(argv) == 2
    assert "GMAT now flies a miss" in capsys.readouterr().err
    assert main([*argv, "--accept-gmat-loss"]) == 0
    certification = json.loads(
        (tmp_path / "final" / "submission" / "certification.json").read_text("utf-8")
    )
    assert certification["planned_hit_certified_as_miss"] is True


def test_a_miss_that_fails_after_promotion_is_not_left_verified(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    """verify promotes before it scores; a failure there must undo the claim."""
    import orbitwin.verify as verify_module

    def broken(*_args, **_kwargs):
        raise RuntimeError("scoring failed after promotion")

    monkeypatch.setattr(verify_module, "apply_score", broken)
    script = _missing_by(near_script, tmp_path, 6.0)
    summary = solve_scenario(
        script, _config("ranking"), tmp_path / "solve", render_top=2,
        gmat_console=fake_gmat, startup_file=fake_startup, gmat_verify_top=2,
    )
    assert summary["best_candidate"] is None
    archived = json.loads((tmp_path / "solve" / "candidates.json").read_text("utf-8"))
    attempted = [c for c in archived if "gmat_verification_error" in c["metadata"]]
    assert attempted and all(c["verified_by_gmat"] is False for c in attempted)


def test_rank_calls_a_verified_miss_submittable_as_certify_does(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.rank import rank_report

    script = _missing_by(near_script, tmp_path, 6.0)
    config = _config("ranking")
    verified, _, _ = _verify(script, config, fake_gmat, fake_startup, tmp_path / "verify")
    scenario, _ = parse_gmat_script(script, config)
    report = rank_report(scenario, [verified], config)
    assert report["winner"]["successful"] is False
    assert report["winner"]["blockers"] == []
    assert report["submittable_count"] == 1


def test_rank_states_how_close_a_miss_must_come_to_beat_the_winner(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    import math

    from orbitwin.rank import rank_report
    from orbitwin.score import ceilings, on_solver_clock

    script = _missing_by(near_script, tmp_path, 6.0)
    config = _config("ranking")
    verified, _, _ = _verify(script, config, fake_gmat, fake_startup, tmp_path / "verify")
    scenario, _ = parse_gmat_script(script, config)
    report = rank_report(scenario, [verified], config)
    shown = report["ceilings"]
    assert config.score is not None and config.policy.total_dv_limit_km_s is not None
    bound = ceilings(on_solver_clock(config.score, scenario), scenario.max_time_s,
                     config.policy.total_dv_limit_km_s)
    assert (shown["any_answer"], shown["any_miss"]) == (bound.any_answer, bound.any_miss)
    assert shown["best_submittable"] == verified.candidate_id
    assert shown["best"] == report["winner"]["score"]
    assert shown["misses_that_beat_best"] == "closer"
    # A miss with the best delta-v term may stand a little further off than the
    # winner, which paid for its approach.
    terms = report["winner"]["score_components"]
    rest = (terms["distance_score"] + terms["time_score"] - bound.time_term_at_horizon
            + terms["delta_v_score"] - bound.best_dv_term)
    assert shown["miss_must_be_within_km"] == pytest.approx(
        5.0 + 100.0 * math.sqrt(math.log(100.0 / rest - 1.0))
    )
    # The term revised 2026/11/04 is flat at 5 km: at C_r 100 km the winner's 6 km and
    # its delta-v buy 4.5 km of room, where the old term's 0.5 pt/km slope left 5 m.
    closest = verified.minimum_distance_km
    assert closest is not None and closest + 4.0 < shown["miss_must_be_within_km"] < closest + 5.0


def test_rank_compares_with_the_best_submittable_row_not_the_top_one(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.rank import rank_report

    script = _missing_by(near_script, tmp_path, 6.0)
    config = _config("ranking")
    verified, _, _ = _verify(script, config, fake_gmat, fake_startup, tmp_path / "verify")
    scenario, _ = parse_gmat_script(near_script, config)
    failed_hit = search_one_burn(scenario, config)[0]
    # The search's first candidate is the one verified: give the stand-in its own id.
    failed_hit.candidate_id = "C-failed-hit"
    failed_hit.metadata["gmat_verification_error"] = "GMAT flew it 19.859 km off"
    report = rank_report(scenario, [verified, failed_hit], config)
    top = report["winner"]
    shown = report["ceilings"]
    assert top["candidate_id"] == failed_hit.candidate_id and not top["submittable"]
    assert shown["best_submittable"] == verified.candidate_id
    assert shown["best"] < top["score"]
    assert report["levers"]["candidate_id"] == verified.candidate_id


def test_rank_ceilings_are_on_the_solver_clock(near_script: Path) -> None:
    """After an alignment T_team's origin is Δ before solver t = 0, so the earliest
    completion scores kt (Δ - Ct), not kt (0 - Ct)."""
    import dataclasses

    from orbitwin.rank import _ceilings

    config = _config("ranking")
    assert config.score is not None
    config = dataclasses.replace(config, score=dataclasses.replace(config.score, ct=1000.0,
                                                                     kt=0.01))
    scenario, _ = parse_gmat_script(near_script, config)
    plain = _ceilings(scenario, config, None)
    shifted = _ceilings(dataclasses.replace(scenario, time_origin_offset_s=1800.0), config, None)
    assert plain is not None and shifted is not None
    assert plain["any_answer"] - shifted["any_answer"] > 20.0
    # T_max moves with the clock too: with Ct at T_max the miss's time term is
    # 12.5 unshifted and 25/(1+e^{kt 1800}) once T_max is 1800 s later on the
    # scored clock; a horizon shifted twice would read 25/(1+e^{kt 3600}).
    later = dataclasses.replace(config, score=dataclasses.replace(
        config.score, ct=scenario.max_time_s, kt=0.0005))
    plain = _ceilings(scenario, later, None)
    shifted = _ceilings(dataclasses.replace(scenario, time_origin_offset_s=1800.0), later, None)
    assert plain is not None and shifted is not None
    import math

    once = 25.0 / (1.0 + math.exp(0.0005 * 1800.0))
    assert shifted["any_miss"] - plain["any_miss"] == pytest.approx(once - 12.5, abs=1e-9)


def test_rank_writes_json_whatever_misses_can_do(near_script: Path) -> None:
    import json

    from orbitwin.rank import _ceilings
    from orbitwin.score import ScoreBreakdown

    config = _config("ranking")
    scenario, _ = parse_gmat_script(near_script, config)
    stand_in = search_one_burn(scenario, config)[0]
    for terms, which in (((0.0, 0.0, 0.0, 0.0, 0.0), "any"),
                         ((50.0, 25.0, 25.0, 0.0, 100.0), "none")):
        shown = _ceilings(scenario, config, (stand_in, ScoreBreakdown(*terms)))
        assert shown is not None and shown["misses_that_beat_best"] == which
        assert shown["miss_must_be_within_km"] is None
        json.dumps(shown, allow_nan=False)


def test_rank_holds_back_a_miss_verified_on_the_interpolated_distance(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.rank import rank_report

    script = _missing_by(near_script, tmp_path, 6.0)
    config = _config("ranking")
    verified, _, _ = _verify(script, config, fake_gmat, fake_startup, tmp_path / "verify")
    scenario, _ = parse_gmat_script(script, config)
    # As an archive written before misses scored on their closest row records it.
    del verified.metadata["gmat_event"]["interpolated_minimum_distance_km"]
    (blocker,) = rank_report(scenario, [verified], config)["winner"]["blockers"]
    assert "verify it again" in blocker


@pytest.mark.parametrize("rules", ["ranking", "preliminary"])
def test_solve_says_when_it_flew_nothing_in_gmat(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path, rules: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from orbitwin.pipeline import SOLVED_WITHOUT_GMAT

    without = solve_scenario(near_script, _config(rules), tmp_path / "without", render_top=2)
    assert (SOLVED_WITHOUT_GMAT in without["warnings"]) is (rules == "ranking")
    if rules == "ranking":
        from orbitwin.cli import main

        profile = tmp_path / "ranking.yaml"
        profile.write_text(QUICK_SEARCH + RANKING_PROFILE, encoding="utf-8")
        assert main(["solve", str(near_script), "--config", str(profile), "--render-top", "2",
                     "--output", str(tmp_path / "cli")]) == 0
        assert f"orbitwin solve: {SOLVED_WITHOUT_GMAT}" in capsys.readouterr().err
    with_gmat = solve_scenario(near_script, _config(rules), tmp_path / "with", render_top=2,
                               gmat_console=fake_gmat, startup_file=fake_startup)
    assert SOLVED_WITHOUT_GMAT not in with_gmat["warnings"]


def _expected_levers(report: dict, candidate, scenario, config) -> dict:
    from orbitwin.score import ScoreBreakdown, levers

    breakdown = ScoreBreakdown(**report["winner"]["score_components"])
    policy = config.policy
    found = levers(candidate, breakdown, scenario, config.score, policy.total_dv_limit_km_s,
                   impulse_km_s=policy.dv_safe_limit_km_s,
                   budget_km_s=policy.total_dv_safe_limit_km_s)
    return {"candidate_id": candidate.candidate_id, **found.to_dict()}


@pytest.mark.parametrize("miss_km", [6.0, None])
def test_rank_states_the_levers_of_the_best_submittable_row(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path,
    miss_km: float | None,
) -> None:
    from orbitwin.rank import rank_report

    script = near_script if miss_km is None else _missing_by(near_script, tmp_path, miss_km)
    config = _config("ranking")
    verified, _, _ = _verify(script, config, fake_gmat, fake_startup, tmp_path / "verify")
    scenario, _ = parse_gmat_script(script, config)
    report = rank_report(scenario, [verified], config)
    shown = report["levers"]
    assert shown == _expected_levers(report, verified, scenario, config)
    assert shown["candidate_id"] == report["ceilings"]["best_submittable"]
    assert shown["hit"] is (miss_km is None)
    headroom = shown["headroom"]
    assert headroom["total"] == pytest.approx(
        report["ceilings"]["any_answer"] - report["winner"]["score"], abs=1e-9)
    # The profile's safe limits, not the rules': 249 m/s and 2.99 km/s.
    spent = shown["spent"]
    assert spent["one_more_impulse"]["km_s"] == config.policy.dv_safe_limit_km_s == 0.249
    assert spent["remaining_budget"]["km_s"] == pytest.approx(2.99 - verified.total_dv_km_s)
    rates = shown["marginal"]
    if miss_km is None:
        assert rates["points_per_s_earlier"] > 0.0 and rates["points_per_km_closer"] is None
    else:
        assert rates["points_per_s_earlier"] is None and rates["break_even_s_per_km_s"] is None
        # 100/(1+e^{u^2}) at u = (6 - 5)/C_r falls at D (1 - D/100) 2u/C_r per km.
        d = 100.0 / (1.0 + math.exp((1.0 / 100.0) ** 2))
        assert rates["points_per_km_closer"] == pytest.approx(d * (1.0 - d / 100.0) * 2.0e-4,
                                                              rel=1e-3)
    json.dumps(shown, allow_nan=False)


def test_rank_has_no_levers_without_a_submittable_row(near_script: Path) -> None:
    from orbitwin.rank import rank_report

    config = _config("ranking")
    scenario, _ = parse_gmat_script(near_script, config)
    failed_hit = search_one_burn(scenario, config)[0]
    failed_hit.metadata["gmat_verification_error"] = "GMAT flew it 19.859 km off"
    report = rank_report(scenario, [failed_hit], config)
    assert report["ceilings"]["best_submittable"] is None and report["levers"] is None


def test_rank_has_no_levers_without_a_total_cap(near_script: Path) -> None:
    """Delta-v then has no end to its range, as for the ceilings."""
    from orbitwin.rank import _levers
    from orbitwin.score import score_candidate

    config = _config("ranking")
    scenario, _ = parse_gmat_script(near_script, config)
    row = search_one_burn(scenario, config)[0]
    assert config.score is not None
    best = (row, score_candidate(row, scenario, config.score))
    assert _levers(scenario, config, best) is not None
    uncapped = dataclasses.replace(config, policy=dataclasses.replace(
        config.policy, total_dv_limit_km_s=None, total_dv_safe_limit_km_s=None))
    assert _levers(scenario, uncapped, best) is None


def test_without_coefficients_a_verified_miss_that_widens_at_all_is_reported(
    tmp_path: Path, near_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """A verified row re-flies bit-identically, so 50 m is drift, not the plan's model."""
    script = _missing_by(near_script, tmp_path, 6.05)
    config = dataclasses.replace(_config("ranking"), score=None)
    scenario, _ = parse_gmat_script(script, config)
    candidate = stamped(_planned_miss(search_one_burn(scenario, config)[0]), script, config)
    candidate.gmat_run_id = "solve-run"
    candidate.metadata["gmat_event"] = {"flown_script_sha256": "0" * 64,
                                        "interpolated_minimum_distance_km": 6.0}
    with pytest.warns(RuntimeWarning, match="GMAT's data files or build changed.*Certify goes on"):
        evidence = _certify(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c")
    compared = evidence["gmat_comparison"]
    assert compared["planned_basis"] == compared["flown_basis"] == "closest approach"
    assert compared["flown_closest_km"] - compared["planned_closest_km"] == pytest.approx(0.05)
    assert compared["refused_without_opt_in"] is False
    # Past 0.1 km it is refused, verified or not.
    (tmp_path / "wide").mkdir()
    wide = _missing_by(near_script, tmp_path / "wide", 6.2)
    candidate = stamped(_planned_miss(search_one_burn(scenario, config)[0]), wide, config)
    candidate.gmat_run_id = "solve-run"
    candidate.metadata["gmat_event"] = {"flown_script_sha256": "0" * 64,
                                        "interpolated_minimum_distance_km": 6.0}
    with pytest.raises(PolicyError, match="GMAT's data files or build changed"):
        _certify(wide, candidate, config, fake_gmat, fake_startup, tmp_path / "wide-c")


def test_the_cli_prints_where_a_refused_certification_went(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An error other than ours carries the note on its own."""
    import orbitwin.cli as cli

    def refused(*_args, **_kwargs):
        error = OSError(28, "No space left on device")
        error.add_note("This certification is kept in /run/final.refused")
        raise error

    monkeypatch.setattr(cli, "certify_candidate", refused)
    from orbitwin.models import Candidate

    row = Candidate(burns=(), intercept_time_s=None, minimum_distance_km=6.0,
                    distance_at_completion_km=6.0, successful=False,
                    analytical_endpoint_error_km=None, branch="zero-burn", candidate_id="C-x")
    monkeypatch.setattr(cli, "read_candidates", lambda _path: [row])
    assert cli.main(["certify", "s.script", "--candidates", "c.json", "--candidate", "C-x",
                     "--gmat-console", "g", "--startup-file", "s", "--output", "o",
                     "--rules", "preliminary"]) == 75
    err = capsys.readouterr().err
    # A full disk is this machine's failure, not a verdict: certify's no-answer exit.
    assert "orbitwin: ERROR [io]: [Errno 28] No space left on device" in err
    assert "orbitwin: This certification is kept in /run/final.refused" in err
