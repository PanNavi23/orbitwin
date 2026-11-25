from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from conftest import capped_one_burn, quick_config, ranking_copy, ranking_quick_config, stamped

from orbitwin.config import OrbitWinConfig, ScoreConfig
from orbitwin.errors import GmatRunError, PolicyError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.verify import certify_candidate, measuring_candidate, verify_candidate_with_gmat

#: policy.stop_depth_km's default; test_a_hit_is_submitted_... pins it.
STOP_DEPTH_KM = quick_config().policy.stop_depth_km


@pytest.fixture()
def workshop_script(workshop_script: Path, tmp_path: Path) -> Path:
    """The workshop with the ranking report name; certify refuses it without one."""
    return ranking_copy(workshop_script, tmp_path)


def _missing(workshop_script: Path, tmp_path: Path) -> Path:
    script = tmp_path / "miss.script"
    script.write_text(
        "% FAKE_GMAT_MISS_KM 6\n" + workshop_script.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return script


def test_verification_flies_a_gmat_miss_on_to_the_horizon(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Ended at the predicted interception, a GMAT miss was measured only that far."""
    config = ranking_quick_config()
    script = _missing(workshop_script, tmp_path)
    scenario, document = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    assert candidate.successful and candidate.intercept_time_s < scenario.max_time_s
    verify_candidate_with_gmat(
        script, scenario, document, candidate, config, fake_gmat, fake_startup, tmp_path / "v"
    )
    second = tmp_path / "v" / "verification_candidate_to_horizon.script"
    flown = parse_gmat_script(second, config)[0]
    total = sum(segment.duration_s for segment in flown.propagate_segments)
    assert total == pytest.approx(scenario.max_time_s)


def test_a_gmat_hit_is_not_flown_on_past_its_interception(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Under a drag model the arc after a hit re-entered and GMAT failed the run."""
    config = ranking_quick_config()
    scenario, document = parse_gmat_script(workshop_script, config)
    candidate = capped_one_burn(scenario, config)
    verified, _, _ = verify_candidate_with_gmat(
        workshop_script, scenario, document, candidate, config, fake_gmat, fake_startup,
        tmp_path / "v",
    )
    assert verified.successful
    flown = parse_gmat_script(tmp_path / "v" / "verification_candidate.script", config)[0]
    assert sum(segment.duration_s for segment in flown.propagate_segments) < scenario.max_time_s
    assert not (tmp_path / "v" / "verification_candidate_to_horizon.script").exists()


def test_verification_first_flies_without_an_archived_stop(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """A stop measured by an earlier run is replaced, not flown again."""
    config = ranking_quick_config()
    scenario, document = parse_gmat_script(workshop_script, config)
    flown: list[bytes] = []
    for name, stale in (("clean", None), ("stale", {"stop_time_s": scenario.max_time_s - 10.0})):
        candidate = capped_one_burn(scenario, config)
        if stale is not None:
            candidate.metadata["gmat_event"] = stale
        verify_candidate_with_gmat(
            workshop_script, scenario, document, candidate, config, fake_gmat, fake_startup,
            tmp_path / name,
        )
        flown.append((tmp_path / name / "verification_candidate.script").read_bytes())
    assert flown[0] == flown[1]


def test_certify_refuses_a_final_script_that_dives_on_replay(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    """The replay of the submitted bytes is what runs to T_max for a miss."""
    import orbitwin.verify as verify_module

    real = verify_module._replay_with_diagnostic

    def diving(*args, **kwargs):
        event, run = real(*args, **kwargs)
        return dataclasses.replace(event, minimum_pursuer_radius_km=2248.4), run

    monkeypatch.setattr(verify_module, "_replay_with_diagnostic", diving)
    config = ranking_quick_config()
    script = _missing(workshop_script, tmp_path)
    scenario, _ = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    stamped(candidate, script, config)
    with pytest.raises(
        PolicyError,
        match=r"inside the .* central body before completing.* "
        r"The ranking rules disqualify it \(2026/11/04 §4 note 1\(b\)\)\.",
    ):
        certify_candidate(
            script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1,
            accept_gmat_loss=True,
        )
    assert not (tmp_path / "c" / "submission").exists()


def test_certify_refuses_a_hit_whose_last_report_row_is_outside(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    """The validator reads the completion time off the last row (2026-09-05)."""
    import orbitwin.verify as verify_module

    real = verify_module._replay_with_diagnostic

    def ends_outside(*args, **kwargs):
        event, run = real(*args, **kwargs)
        return dataclasses.replace(event, distance_at_end_km=5.88), run

    monkeypatch.setattr(verify_module, "_replay_with_diagnostic", ends_outside)
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = capped_one_burn(scenario, config)
    stamped(candidate, workshop_script, config)
    with pytest.raises(GmatRunError, match=r"last report row is 5\.88 km"):
        certify_candidate(
            workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1
        )
    assert not (tmp_path / "c" / "submission").exists()


def test_certify_names_the_horizon_policy_when_it_is_the_reason(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    import orbitwin.verify as verify_module

    real = verify_module._replay_with_diagnostic

    def ends_outside(*args, **kwargs):
        event, run = real(*args, **kwargs)
        return dataclasses.replace(event, distance_at_end_km=900.0), run

    monkeypatch.setattr(verify_module, "_replay_with_diagnostic", ends_outside)
    base = quick_config()
    config = dataclasses.replace(
        base, policy=dataclasses.replace(base.policy, mission_end_policy="horizon")
    )
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = capped_one_burn(scenario, config)
    stamped(candidate, workshop_script, config)
    with pytest.raises(GmatRunError, match="mission_end_policy is horizon"):
        certify_candidate(
            workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1
        )


def test_verification_flies_to_the_horizon_under_the_preliminary_rules_too(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """GMAT entering later than predicted is not a miss under either rulebook."""
    config = quick_config()
    script = _missing(workshop_script, tmp_path)
    scenario, document = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    with pytest.raises(PolicyError):  # a miss fails the preliminary rules
        verify_candidate_with_gmat(
            script, scenario, document, candidate, config, fake_gmat, fake_startup,
            tmp_path / "v",
        )
    second = tmp_path / "v" / "verification_candidate_to_horizon.script"
    flown = parse_gmat_script(second, config)[0]
    total = sum(segment.duration_s for segment in flown.propagate_segments)
    assert total == pytest.approx(scenario.max_time_s)


def test_the_submitted_script_of_a_hit_still_ends_at_the_interception(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Only the replay runs to T_max; a submitted hit that did lost 22 points on 2026-09-05."""
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = capped_one_burn(scenario, config)
    stamped(candidate, workshop_script, config)
    certify_candidate(
        workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1
    )
    # Not official_original.script, the organizer's own copy beside it.
    submitted = tmp_path / "c" / "submission" / "public_gmat_candidate.script"
    flown = parse_gmat_script(submitted, config)[0]
    total = sum(segment.duration_s for segment in flown.propagate_segments)
    assert total < scenario.max_time_s - 1.0


def _flyby(workshop_script: Path, tmp_path: Path, closest_s: float, *extra: str) -> Path:
    script = tmp_path / f"flyby-{closest_s:g}.script"
    lines = [f"% FAKE_GMAT_FLYBY {closest_s} 7", *extra]
    script.write_text(
        "\n".join(lines) + "\n" + workshop_script.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return script


def _total_s(script: Path, config: OrbitWinConfig) -> float:
    return sum(segment.duration_s for segment in parse_gmat_script(script, config)[0].propagate_segments)


def test_a_hit_is_submitted_ending_its_stop_depth_inside_and_scored_there(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """The double passes through the target at 3000 s at 1 km/s, a row every 7 s.

    The literals pin policy.stop_depth_km's default of 1.5 km.
    """
    config = dataclasses.replace(
        quick_config(), score=ScoreConfig(kt=0.001, ct=5000.0, kv=1.0, cv=1.0)
    )
    script = _flyby(workshop_script, tmp_path, 3000.0)
    scenario, _ = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    stamped(candidate, script, config)
    evidence = certify_candidate(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1)
    certified = evidence["candidate"]
    assert certified["metadata"]["gmat_event"]["first_entry_time_s"] == pytest.approx(2995.0)
    # Completion is the last row the validator reads, not the interpolated entry.
    assert certified["intercept_time_s"] == pytest.approx(2996.5, abs=1e-9)
    assert certified["minimum_distance_km"] == pytest.approx(3.5, abs=1e-9)
    assert certified["metadata"]["gmat_event"]["stop_depth_setting_km"] == 1.5
    submitted = tmp_path / "c" / "submission" / "public_gmat_candidate.script"
    assert _total_s(submitted, config) == pytest.approx(2996.5, abs=1e-9)
    assert evidence["final_script_parity"]["end_time_s"] == pytest.approx(2996.5, abs=1e-9)


def test_a_late_hit_is_found_on_a_flight_that_aborts_after_it(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Past the predicted 4891 s; the flight on to T_max dies at 6500 s, as drag does below 100 km."""
    config = ranking_quick_config()
    script = _flyby(workshop_script, tmp_path, 6000.0, "% FAKE_GMAT_ABORT_AT_S 6500")
    scenario, document = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    verified, _, _ = verify_candidate_with_gmat(
        script, scenario, document, candidate, config, fake_gmat, fake_startup, tmp_path / "v"
    )
    assert verified.successful
    assert verified.metadata["gmat_event"]["stop_time_s"] == pytest.approx(5995.0 + STOP_DEPTH_KM)
    assert _total_s(tmp_path / "v" / "verification_candidate_at_stop.script", config) == (
        pytest.approx(5995.0 + STOP_DEPTH_KM)
    )


def test_an_abort_before_the_interception_is_still_a_failure(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    config = ranking_quick_config()
    script = _flyby(workshop_script, tmp_path, 6000.0, "% FAKE_GMAT_ABORT_AT_S 5900")
    scenario, document = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    with pytest.raises(GmatRunError, match="exit status 1"):
        verify_candidate_with_gmat(
            script, scenario, document, candidate, config, fake_gmat, fake_startup, tmp_path / "v"
        )


def test_a_hit_verified_with_a_score_certifies_without_one(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Measured to wherever verification stopped, the closest approach left such hits stale."""
    scored = dataclasses.replace(
        quick_config(), score=ScoreConfig(kt=0.001, ct=5000.0, kv=1.0, cv=1.0)
    )
    script = _flyby(workshop_script, tmp_path, 3000.0)
    scenario, document = parse_gmat_script(script, scored)
    candidate = capped_one_burn(scenario, scored)
    candidate, _, _ = verify_candidate_with_gmat(
        script, scenario, document, candidate, scored, fake_gmat, fake_startup, tmp_path / "solve"
    )
    solved = (candidate.intercept_time_s, candidate.minimum_distance_km)
    stamped(candidate, script, dataclasses.replace(scored, score=None))
    evidence = certify_candidate(
        script, candidate, dataclasses.replace(scored, score=None), fake_gmat, fake_startup,
        tmp_path / "c", runs=1,
    )
    certified = evidence["candidate"]
    assert (certified["intercept_time_s"], certified["minimum_distance_km"]) == solved
    assert certified["score_is_stale"] is False


@pytest.mark.parametrize(
    ("closest_s", "field"),
    [
        (3000.0, "first_entry_time_s"),
        (3000.0, "end_time_s"),
        (3000.0, "minimum_distance_km"),
        (None, "end_time_s"),
        # Scored on its closest row; the Hermite minimum still guards the burns' rows.
        (None, "minimum_row_distance_km"),
        (None, "minimum_distance_km"),
    ],
)
def test_certify_refuses_a_replay_that_moves_a_score_input(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
    monkeypatch,
    field: str,
    closest_s: float | None,
) -> None:
    """The replay flies the bytes verification flew; a hit or a miss must come back unchanged."""
    import orbitwin.verify as verify_module

    real = verify_module._replay_with_diagnostic

    def moved(*args, **kwargs):
        event, run = real(*args, **kwargs)
        value = getattr(event, field)
        return dataclasses.replace(event, **{field: None if value is None else value + 1.0e-3}), run

    monkeypatch.setattr(verify_module, "_replay_with_diagnostic", moved)
    config = ranking_quick_config()
    if closest_s is None:
        script = _missing(workshop_script, tmp_path)
    else:
        script = _flyby(workshop_script, tmp_path, closest_s)
    scenario, _ = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    expected = (
        "other than the submitted bytes"
        if closest_s is not None or field in ("minimum_distance_km", "minimum_row_distance_km")
        else None
    )
    stamped(candidate, script, config)
    # Without closest_s the planned hit is flown as a miss, certified as one on purpose.
    if expected is None:  # A miss completes at T_max whatever its last row says.
        certify_candidate(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c",
                          runs=1, accept_gmat_loss=True)
        return
    with pytest.raises(GmatRunError, match=expected):
        certify_candidate(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c",
                          runs=1, accept_gmat_loss=True)


def test_certify_says_when_a_ranking_hit_loses_its_entry_on_replay(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    import orbitwin.verify as verify_module

    real = verify_module._replay_with_diagnostic

    def lost(*args, **kwargs):
        event, run = real(*args, **kwargs)
        return dataclasses.replace(event, first_entry_time_s=None), run

    monkeypatch.setattr(verify_module, "_replay_with_diagnostic", lost)
    config = ranking_quick_config()
    script = _flyby(workshop_script, tmp_path, 3000.0)
    scenario, _ = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    stamped(candidate, script, config)
    with pytest.raises(GmatRunError, match="does not enter the intercept sphere on replay"):
        certify_candidate(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1)


def test_verification_ignores_a_stop_measured_on_another_run(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    config = quick_config()
    script = _flyby(workshop_script, tmp_path, 3000.0)
    scenario, document = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    candidate.metadata["gmat_event"] = {"stop_time_s": 1.0e9}
    verified, _, _ = verify_candidate_with_gmat(
        script, scenario, document, candidate, config, fake_gmat, fake_startup, tmp_path / "hit"
    )
    assert verified.metadata["gmat_event"]["stop_time_s"] == pytest.approx(2995.0 + STOP_DEPTH_KM)

    config = ranking_quick_config()
    script = _missing(workshop_script, tmp_path)
    scenario, document = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    candidate.metadata["gmat_event"] = {"stop_time_s": 5.05}
    missed, _, _ = verify_candidate_with_gmat(
        script, scenario, document, candidate, config, fake_gmat, fake_startup, tmp_path / "miss"
    )
    assert not missed.successful
    assert missed.metadata["gmat_event"]["stop_time_s"] is None


def _verify(script: Path, config: OrbitWinConfig, out: Path, fake_gmat: Path, fake_startup: Path):
    scenario, document = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    return verify_candidate_with_gmat(
        script, scenario, document, candidate, config, fake_gmat, fake_startup, out
    )[0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"distance_at_end_km": 5.2}, "last report row is 5.2 km"),
        ({"first_entry_time_s": None}, "no longer enters"),
    ],
)
def test_verification_refuses_a_stop_flight_that_ends_outside(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch,
    change: dict, message: str,
) -> None:
    """solve promotes on verification alone, so the check cannot be left to certify."""
    import orbitwin.verify as verify_module

    real = verify_module.evaluate_diagnostic_report

    def at_stop(path, *args, **kwargs):
        event = real(path, *args, **kwargs)
        return dataclasses.replace(event, **change) if "at-stop" in str(path) else event

    monkeypatch.setattr(verify_module, "evaluate_diagnostic_report", at_stop)
    with pytest.raises(GmatRunError, match=message):
        _verify(_flyby(workshop_script, tmp_path, 3000.0), quick_config(), tmp_path / "v", fake_gmat, fake_startup)


def test_the_stop_flight_may_not_rest_on_a_partial_report(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    """Only a measuring flight may abort; the submitted script has to run to its end."""
    import orbitwin.verify as verify_module

    real = verify_module.run_gmat

    def aborting(console, script, workdir, *args, **kwargs):
        run = real(console, script, workdir, *args, **kwargs)
        if "at-stop" not in str(workdir):
            return run
        return dataclasses.replace(
            run, success=False, error="forced abort", reports={}, partial_reports=run.reports
        )

    monkeypatch.setattr(verify_module, "run_gmat", aborting)
    with pytest.raises(GmatRunError, match="forced abort"):
        _verify(_flyby(workshop_script, tmp_path, 3000.0), quick_config(), tmp_path / "v", fake_gmat, fake_startup)


def test_an_entry_short_of_the_stop_depth_is_flown_on(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """The first flight ends 20 m inside: an entry, but no measured place to stop."""
    from orbitwin.script_renderer import render_candidate

    config = quick_config()
    scenario, document = parse_gmat_script(workshop_script, config)
    candidate = capped_one_burn(scenario, config)
    first_end = render_candidate(
        document, scenario, measuring_candidate(candidate, scenario, config), config
    ).mission_end_s
    script = _flyby(workshop_script, tmp_path, first_end + 4.98)
    verified = _verify(script, config, tmp_path / "v", fake_gmat, fake_startup)
    assert (tmp_path / "v" / "verification_candidate_to_horizon.script").exists()
    assert verified.metadata["gmat_event"]["stop_time_s"] == pytest.approx(
        first_end - 0.02 + STOP_DEPTH_KM
    )


def test_a_hit_still_inside_at_t_max_stops_at_its_deepest_row(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    config = ranking_quick_config()
    script = _flyby(workshop_script, tmp_path, config.policy.horizon_s + 4.97)
    verified = _verify(script, config, tmp_path / "v", fake_gmat, fake_startup)
    # Entered 0.03 s before T_max, never STOP_DEPTH_KM deep: stopped on the last, deepest row.
    assert verified.metadata["gmat_event"]["stop_time_s"] == pytest.approx(config.policy.horizon_s, abs=1e-6)
    assert verified.intercept_time_s == pytest.approx(config.policy.horizon_s, abs=1e-6)


def test_a_shallow_hit_says_how_little_it_is_inside(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _flyby(workshop_script, tmp_path, 3000.0, "% FAKE_GMAT_MISS_KM 4.99")
    verified = _verify(script, quick_config(), tmp_path / "v", fake_gmat, fake_startup)
    assert verified.metadata["gmat_event"]["stop_depth_km"] == pytest.approx(0.01, abs=1e-6)
    assert any("only 10 m inside" in warning for warning in verified.warnings)


def test_a_hit_under_the_horizon_policy_is_scored_at_t_max(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Its upload runs to T_max, and the validator reads completion off the last row."""
    config = ranking_quick_config()
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, mission_end_policy="horizon")
    )
    verified = _verify(workshop_script, config, tmp_path / "v", fake_gmat, fake_startup)
    assert verified.successful
    assert verified.intercept_time_s == pytest.approx(config.policy.horizon_s)
    assert not (tmp_path / "v" / "verification_candidate_at_stop.script").exists()


def test_certify_refuses_a_final_script_other_than_the_one_flown(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    import orbitwin.verify as verify_module

    real = verify_module.verify_candidate_with_gmat

    def other_bytes(*args, **kwargs):
        verified, run, evidence = real(*args, **kwargs)
        verified.metadata["gmat_event"]["flown_script_sha256"] = "0" * 64
        return verified, run, evidence

    monkeypatch.setattr(verify_module, "verify_candidate_with_gmat", other_bytes)
    script = _flyby(workshop_script, tmp_path, 3000.0)
    config = quick_config()
    scenario, _ = parse_gmat_script(script, config)
    candidate = stamped(capped_one_burn(scenario, config), script, config)
    with pytest.raises(GmatRunError, match="not the script GMAT verified"):
        certify_candidate(
            script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1
        )
    assert not (tmp_path / "c" / "submission").exists()


def test_certify_refuses_a_miss_that_enters_on_replay(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    import orbitwin.verify as verify_module

    real = verify_module._replay_with_diagnostic

    def entered(*args, **kwargs):
        event, run = real(*args, **kwargs)
        return dataclasses.replace(event, first_entry_time_s=100.0), run

    monkeypatch.setattr(verify_module, "_replay_with_diagnostic", entered)
    config = ranking_quick_config()
    script = _missing(workshop_script, tmp_path)
    scenario, _ = parse_gmat_script(script, config)
    candidate = stamped(capped_one_burn(scenario, config), script, config)
    with pytest.raises(GmatRunError, match="verification measured a miss"):
        certify_candidate(
            script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1,
            accept_gmat_loss=True,
        )


def _deeper(config: OrbitWinConfig, depth_km: float) -> OrbitWinConfig:
    return dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, stop_depth_km=depth_km)
    )


def test_verification_stops_at_the_configured_depth(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    config = _deeper(quick_config(), 0.6)
    script = _flyby(workshop_script, tmp_path, 3000.0)
    verified = _verify(script, config, tmp_path / "v", fake_gmat, fake_startup)
    event = verified.metadata["gmat_event"]
    assert event["stop_time_s"] == pytest.approx(2995.6, abs=1e-9)
    assert event["stop_depth_km"] == pytest.approx(0.6, abs=1e-9)
    assert event["stop_depth_setting_km"] == 0.6
    assert verified.intercept_time_s == pytest.approx(2995.6, abs=1e-9)


@pytest.mark.parametrize(("miss_km", "warned"), [(4.2, True), (3.6, True), (3.4, False)])
def test_a_passage_short_of_the_stop_depth_says_so(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    miss_km: float, warned: bool,
) -> None:
    """0.8 and 1.4 km deep fall short of 1.5 km; the old half-depth threshold kept quiet."""
    script = _flyby(workshop_script, tmp_path, 3000.0, f"% FAKE_GMAT_MISS_KM {miss_km}")
    verified = _verify(script, quick_config(), tmp_path / "v", fake_gmat, fake_startup)
    shallow = [w for w in verified.warnings if "short of policy.stop_depth_km = 1500 m" in w]
    assert bool(shallow) is warned
    if warned:
        assert f"only {(5.0 - miss_km) * 1000.0:.4g} m inside" in shallow[0]


def test_reverification_replaces_the_resolution_warnings_it_carried(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _flyby(workshop_script, tmp_path, 3000.0, "% FAKE_GMAT_MISS_KM 4.0")
    scenario, document = parse_gmat_script(script, quick_config())
    candidate = capped_one_burn(scenario, quick_config())
    candidate.warnings = ("kept",)
    first, _, _ = verify_candidate_with_gmat(
        script, scenario, document, candidate, quick_config(), fake_gmat, fake_startup,
        tmp_path / "a",
    )
    assert any("only 1000 m inside" in w for w in first.warnings)
    again, _, _ = verify_candidate_with_gmat(
        script, scenario, document, first, _deeper(quick_config(), 0.6), fake_gmat, fake_startup,
        tmp_path / "b",
    )
    assert "kept" in again.warnings
    assert not any("m inside the" in w for w in again.warnings)


def test_certify_says_when_the_archive_was_verified_at_another_depth(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    config = quick_config()
    script = _flyby(workshop_script, tmp_path, 3000.0)
    scenario, _ = parse_gmat_script(script, config)
    candidate = stamped(capped_one_burn(scenario, config), script, config)
    candidate.metadata["gmat_event"] = {"stop_time_s": 2995.05, "stop_depth_km": 0.05}
    with pytest.warns(RuntimeWarning, match="before the stop depth was recorded, 50 m inside"):
        evidence = certify_candidate(
            script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1
        )
    assert any("measures it again" in w for w in evidence["archive_warnings"])
    assert evidence["candidate"]["intercept_time_s"] == pytest.approx(2996.5, abs=1e-9)


def test_the_first_flight_goes_deep_enough_not_to_fly_on(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Entered 1 s before the predicted end at 1 km/s: 1 km in, short of 1.5 km, until extended."""
    from orbitwin.script_renderer import render_candidate

    config = ranking_quick_config()
    scenario, document = parse_gmat_script(workshop_script, config)
    candidate = capped_one_burn(scenario, config)
    predicted_end = render_candidate(document, scenario, candidate, config).mission_end_s
    extended_end = render_candidate(
        document, scenario, measuring_candidate(candidate, scenario, config), config
    ).mission_end_s
    assert extended_end >= predicted_end + 1.0
    script = _flyby(workshop_script, tmp_path, predicted_end + 4.0)
    verified = _verify(script, config, tmp_path / "v", fake_gmat, fake_startup)
    assert not (tmp_path / "v" / "verification_candidate_to_horizon.script").exists()
    assert verified.metadata["gmat_event"]["stop_time_s"] == pytest.approx(predicted_end + 0.5)


def test_a_shallow_first_flight_reaches_the_row_outside(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """1 km deep at most, out again 3 s after closest: radius / speed ended still inside."""
    import numpy as np

    from orbitwin.script_renderer import render_candidate
    from orbitwin.trajectory import state_with_burns_at

    config = ranking_quick_config()
    scenario, document = parse_gmat_script(workshop_script, config)
    candidate = capped_one_burn(scenario, config)
    entry_s = candidate.intercept_time_s
    pursuer = state_with_burns_at(
        scenario.pursuer.initial_state, candidate.burns, entry_s, scenario.mu_km3_s2
    )
    target = state_with_burns_at(scenario.target.initial_state, (), entry_s, scenario.mu_km3_s2)
    speed = float(np.linalg.norm(np.asarray(pursuer.v) - np.asarray(target.v)))
    half = dataclasses.replace(candidate, intercept_time_s=entry_s + 5.0 / speed)
    half_end = render_candidate(document, scenario, half, config).mission_end_s
    extended = measuring_candidate(candidate, scenario, config)
    end = render_candidate(document, scenario, extended, config).mission_end_s
    assert end >= half_end + 1.0
    closest_s = end - 3.5  # exit at end - 0.5, after half_end
    script = _flyby(workshop_script, tmp_path, closest_s, "% FAKE_GMAT_MISS_KM 4.0")
    verified = _verify(script, config, tmp_path / "v", fake_gmat, fake_startup)
    assert not (tmp_path / "v" / "verification_candidate_to_horizon.script").exists()
    assert verified.metadata["gmat_event"]["stop_time_s"] == pytest.approx(closest_s, abs=1e-6)
