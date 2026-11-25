"""A hit whose submitted script never got policy.stop_depth_km deep is not a deep hit.

The stop depth is margin against GMAT setups reading other data files: the
CSSI space-weather file moved R4's arrival 665 m, EOP moved insertion 6-25 m.
A hit a few metres inside can be a miss on the organizer's machine.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from conftest import quick_config, stamped

from orbitwin.config import OrbitWinConfig
from orbitwin.constraints import prefer_deep_hits, shallow_stop, stop_depth_reached
from orbitwin.errors import PolicyError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Candidate, CandidateStage
from orbitwin.pipeline import solve_scenario
from orbitwin.search import search_one_burn
from orbitwin.verify import certify_candidate


def _flyby(workshop_script: Path, tmp_path: Path, miss_km: float) -> Path:
    """The double passes the target at 3000 s at 1 km/s, ``miss_km`` off, a row every 7 s."""
    script = tmp_path / f"flyby-{miss_km:g}.script"
    script.write_text(
        f"% FAKE_GMAT_FLYBY 3000.0 7\n% FAKE_GMAT_MISS_KM {miss_km}\n"
        + workshop_script.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return script


def _allowing(config: OrbitWinConfig) -> OrbitWinConfig:
    return dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, allow_shallow_stop_submission=True)
    )


def _hit(script: Path, config: OrbitWinConfig) -> Candidate:
    scenario, _ = parse_gmat_script(script, config)
    return stamped(search_one_burn(scenario, config)[0], script, config)


def test_certify_refuses_a_hit_that_stops_10_m_inside(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    config = quick_config()
    script = _flyby(workshop_script, tmp_path, 4.99)
    with pytest.raises(PolicyError, match=r"only 10 m inside.*allow_shallow_stop_submission"):
        certify_candidate(
            script, _hit(script, config), config, fake_gmat, fake_startup, tmp_path / "c",
            runs=1,
        )
    assert not (tmp_path / "c" / "submission").exists()


def test_an_archived_deep_stop_does_not_get_a_shallow_hit_certified(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """certify measures the depth again; the file's claim is not evidence."""
    config = quick_config()
    script = _flyby(workshop_script, tmp_path, 4.99)
    candidate = _hit(script, config)
    candidate.metadata["gmat_event"] = {
        "stop_time_s": 2996.5, "stop_depth_km": 1.5, "stop_depth_setting_km": 1.5,
    }
    with pytest.raises(PolicyError, match="only 10 m inside"):
        certify_candidate(
            script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1
        )


def test_the_opt_in_certifies_a_shallow_hit_and_says_so(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    config = _allowing(quick_config())
    script = _flyby(workshop_script, tmp_path, 4.99)
    with pytest.warns(RuntimeWarning, match="only 10 m inside.*certified under"):
        evidence = certify_candidate(
            script, _hit(script, config), config, fake_gmat, fake_startup, tmp_path / "c",
            runs=1,
        )
    assert "only 10 m inside" in evidence["shallow_stop"]
    written = json.loads(
        (tmp_path / "c" / "submission" / "certification.json").read_text(encoding="utf-8")
    )
    assert written["shallow_stop"] == evidence["shallow_stop"]


@pytest.mark.parametrize("miss_km", [0.0, 3.5])
def test_a_hit_that_reaches_the_stop_depth_needs_no_opt_in(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, miss_km: float
) -> None:
    """3.5 km off reaches exactly 1.5 km deep, at its closest."""
    config = quick_config()
    script = _flyby(workshop_script, tmp_path, miss_km)
    evidence = certify_candidate(
        script, _hit(script, config), config, fake_gmat, fake_startup, tmp_path / "c", runs=1
    )
    assert evidence["shallow_stop"] is None


def _event(depth_km: float | None, setting_km: float | None = 1.5) -> dict:
    return {"stop_time_s": 1000.0, "stop_depth_km": depth_km, "stop_depth_setting_km": setting_km}


def _candidate(dv_km_s: float, *, successful: bool = True, event: dict | None = None) -> Candidate:
    burn = Burn(time_s=100.0, dv_inertial_km_s=(dv_km_s, 0.0, 0.0),
                dv_script_km_s=(dv_km_s, 0.0, 0.0))
    return Candidate(
        burns=(burn,), intercept_time_s=1000.0, minimum_distance_km=1.0,
        distance_at_completion_km=5.0, successful=successful,
        analytical_endpoint_error_km=None, branch="t",
        metadata={"gmat_event": event} if event is not None else {},
    )


@pytest.mark.parametrize(
    ("event", "successful", "today_km", "reached"),
    [
        (_event(1.5), True, 1.5, True),
        (_event(1.485), True, 1.5, True),  # 1 %: Hermite error on coarse rows
        (_event(1.4849), True, 1.5, False),
        # Stopped at an older 0.6 km setting: the passage may go deeper, certify measures it.
        (_event(0.6, 0.6), True, 1.5, None),
        (_event(0.6, 0.6), True, 0.6, True),
        # Short of its own 1.5 km, so 1.2 km is the passage's deepest: enough for 0.5 km today.
        (_event(1.2, 1.5), True, 0.5, True),
        (_event(1.2, 1.5), True, 1.3, False),
        (_event(0.05, None), True, 1.5, None),  # stopped before the setting was recorded
        (_event(1.5, None), True, 1.5, True),
        (_event(2.5, 3.0), True, 1.5, True),  # asked past half the radius: stopped at 2.5
        (_event(2.4, 3.0), True, 2.5, False),
        (_event(None), True, 1.5, None),  # flown to T_max under the horizon policy
        (_event(0.01), False, 1.5, None),  # a miss has no stop
        (None, True, 1.5, None),  # never verified
    ],
)
def test_stop_depth_reached(
    event: dict | None, successful: bool, today_km: float, reached: bool | None
) -> None:
    config = quick_config()
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, stop_depth_km=today_km)
    )
    candidate = _candidate(0.1, successful=successful, event=event)
    assert stop_depth_reached(candidate, config) is reached
    assert (shallow_stop(candidate, config) is not None) is (reached is False)


def test_prefer_deep_hits_drops_shallow_hits_only_beside_a_deep_one() -> None:
    config = quick_config()
    shallow = _candidate(0.1, event=_event(0.01))
    miss = _candidate(0.2, successful=False)
    deep = _candidate(0.3, event=_event(1.5))
    unmeasured = _candidate(0.4)
    assert prefer_deep_hits([shallow, miss, deep, unmeasured], config) == [miss, deep, unmeasured]
    # Without a deep hit nothing is dropped: the order still decides.
    assert prefer_deep_hits([shallow, miss, unmeasured], config) == [shallow, miss, unmeasured]
    assert prefer_deep_hits([], config) == []


def _solve(
    tmp_path: Path, name: str, script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, depth_km: dict[str, float], default_km: float,
) -> dict:
    """solve with GMAT stubbed: every candidate is a verified hit at its listed depth."""
    import orbitwin.pipeline as pipeline_module

    def verify(_script, _scenario, _document, candidate, *_args, **_kwargs):
        candidate.successful = True
        candidate.metadata["event_refined"] = True
        candidate.metadata["gmat_event"] = _event(
            depth_km.get(candidate.candidate_id, default_km)
        )
        candidate.promote(CandidateStage.PUBLIC_GMAT_VERIFIED, run_id="stub")
        return candidate, None, {}

    monkeypatch.setattr(pipeline_module, "verify_candidate_with_gmat", verify)
    return solve_scenario(
        script, quick_config(), tmp_path / name, render_top=3, gmat_console=fake_gmat,
        startup_file=fake_startup, gmat_verify_top=3,
    )


def test_solve_prefers_a_deep_hit_over_a_better_ranked_shallow_one(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deep = _solve(tmp_path, "deep", workshop_script, fake_gmat, fake_startup, monkeypatch, {}, 1.5)
    first = deep["best_candidate"]["candidate_id"]
    assert deep["best_candidate_shallow_stop"] is None
    assert not any("stop_depth" in w for w in deep["warnings"])

    mixed = _solve(tmp_path, "mixed", workshop_script, fake_gmat, fake_startup, monkeypatch,
                   {first: 0.01}, 1.5)
    assert mixed["best_candidate"]["candidate_id"] != first
    assert mixed["best_candidate_shallow_stop"] is None
    assert any("passed over because their submitted script stops short" in w and first in w
               for w in mixed["warnings"])


def test_solve_names_a_shallow_hit_only_when_no_hit_is_deep_and_says_so(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deep = _solve(tmp_path, "deep", workshop_script, fake_gmat, fake_startup, monkeypatch, {}, 1.5)
    shallow = _solve(tmp_path, "shallow", workshop_script, fake_gmat, fake_startup, monkeypatch,
                     {}, 0.01)
    assert shallow["best_candidate"]["candidate_id"] == deep["best_candidate"]["candidate_id"]
    assert "only 10 m inside" in shallow["best_candidate_shallow_stop"]
    assert any(
        w.startswith("No verified hit reached policy.stop_depth_km")
        and "certify refuses it unless policy.allow_shallow_stop_submission" in w
        for w in shallow["warnings"]
    )


def _ranking_rules() -> OrbitWinConfig:
    """Time and delta-v weighted so an early hit, the zero burn's miss and a late hit
    score about 75, 73 and 50."""
    from orbitwin.config import ScoreConfig

    base = quick_config()
    policy = dataclasses.replace(
        base.policy, rules="ranking", horizon_s=30000.0,
        dv_limit_km_s=0.25, dv_safe_limit_km_s=0.249,
        total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99,
    )
    score = ScoreConfig(kt=0.01, ct=10000.0, kv=0.05, cv=50.0, delta_v_unit="m/s", cr_km=100.0)
    return dataclasses.replace(base, policy=policy, score=score)


def _near(workshop_script: Path, tmp_path: Path) -> Path:
    """The workshop with SC_Alien just ahead, so hits fit 250 m/s (as test_scored_miss)."""
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


def _solve_ranked(
    tmp_path: Path, name: str, script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, early: str | None, late_depth_km: float,
) -> tuple[dict, list[str]]:
    """GMAT stubbed under the ranking rules. The zero burn is a 5.01 km miss; ``early``
    a hit at 1000 s 10 m deep; every other hit at 29000 s, ``late_depth_km`` deep.

    Returns the summary and the ids of the hits it verified.
    """
    import orbitwin.pipeline as pipeline_module
    from orbitwin.score import apply_score

    hits: list[str] = []

    def verify(_script, scenario, _document, candidate, config, *_args, **_kwargs):
        candidate.metadata["event_refined"] = True
        if candidate.branch == "zero-burn":
            candidate.successful = False
            candidate.intercept_time_s = None
            candidate.minimum_distance_km = candidate.distance_at_completion_km = 5.01
            candidate.metadata["scored_miss"] = True
        else:
            hits.append(candidate.candidate_id)
            first = candidate.candidate_id == early
            candidate.successful = True
            candidate.intercept_time_s = 1000.0 if first else 29000.0
            candidate.distance_at_completion_km = 5.0
            candidate.minimum_distance_km = 3.0
            candidate.metadata["scored_miss"] = False
            candidate.metadata["gmat_event"] = _event(0.01 if first else late_depth_km)
        candidate.promote(CandidateStage.PUBLIC_GMAT_VERIFIED, run_id="stub")
        apply_score(candidate, scenario, config.score)
        return candidate, None, {}

    monkeypatch.setattr(pipeline_module, "verify_candidate_with_gmat", verify)
    summary = solve_scenario(
        script, _ranking_rules(), tmp_path / name, render_top=3,
        gmat_console=fake_gmat, startup_file=fake_startup, gmat_verify_top=3,
    )
    return summary, sorted(hits)


def _scores(tmp_path: Path, name: str) -> dict[str, float]:
    archived = json.loads((tmp_path / name / "candidates.json").read_text(encoding="utf-8"))
    rows = archived if isinstance(archived, list) else archived["candidates"]
    return {row["candidate_id"]: row["score"] for row in rows if row.get("score") is not None}


def test_a_shallow_hit_loses_to_a_deep_one_and_a_miss_between_them_wins(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shallow ~75 > miss ~73 > deep ~50: the shallow hit goes, then the order decides."""
    script = _near(workshop_script, tmp_path)
    _, hits = _solve_ranked(tmp_path, "probe", script, fake_gmat, fake_startup,
                            monkeypatch, None, 1.5)
    assert len(hits) >= 2
    summary, _ = _solve_ranked(tmp_path, "mixed", script, fake_gmat, fake_startup,
                               monkeypatch, hits[0], 1.5)
    scores = _scores(tmp_path, "mixed")
    best = summary["best_candidate"]
    assert best["branch"] == "zero-burn" and best["successful"] is False
    assert scores[hits[0]] > best["score"] > max(scores[h] for h in hits[1:])
    assert summary["best_candidate_shallow_stop"] is None
    assert any(hits[0] in w and "passed over" in w for w in summary["warnings"])


def test_with_no_deep_hit_solve_names_what_certify_takes_without_the_opt_in(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _near(workshop_script, tmp_path)
    _, hits = _solve_ranked(tmp_path, "probe", script, fake_gmat, fake_startup,
                            monkeypatch, None, 1.5)
    summary, _ = _solve_ranked(tmp_path, "shallow", script, fake_gmat, fake_startup,
                               monkeypatch, hits[0], 0.01)
    assert summary["best_candidate"]["candidate_id"] == hits[0]
    assert "only 10 m inside" in summary["best_candidate_shallow_stop"]
    miss = summary["best_certifiable_without_shallow_opt_in"]
    assert miss is not None and miss not in hits
    assert any(f"the best it accepts without that setting is {miss}" in w
               for w in summary["warnings"])
