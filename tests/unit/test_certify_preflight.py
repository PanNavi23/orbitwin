"""What certify refuses before it creates anything.

A candidate handed straight to certify did not pass through rank, which
checks the archive's script and horizon stamps; and a refusal that left a
diagnostic/ directory behind meant a retry needed a new output directory.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import pytest
from conftest import FAKE_STARTUP

from orbitwin.config import OrbitWinConfig
from orbitwin.errors import PolicyError
from orbitwin.models import Candidate, Scenario


def _setup(workshop_script: Path) -> tuple[OrbitWinConfig, Candidate, Scenario]:
    from conftest import quick_config

    from orbitwin.search import search_one_burn
    from orbitwin.state_probe import load_scenario

    config = quick_config()
    # Probed, as certify reads it: its 4 T_A is about 42 µs off the in-process one.
    scenario, _, _ = load_scenario(
        workshop_script, config,
        gmat_console=Path(__file__).resolve().parents[2] / "scripts" / "fake_gmat_console.py",
        startup_file=FAKE_STARTUP,
    )
    return config, search_one_burn(scenario, config)[0], scenario


def _stamped(candidate: Candidate, **stamps: object) -> Candidate:
    return dataclasses.replace(candidate, metadata={**candidate.metadata, **stamps})


@pytest.mark.parametrize(
    "stamps, horizon_offset_s, match",
    [
        ({"scenario_sha256": "0" * 64}, 0.0, "hashes to"),
        ({}, -1000.0, "horizon"),
        # A millisecond apart is another policy.horizon_s, not rounding.
        ({}, 1.0e-3, "horizon"),
    ],
)
def test_contradicting_stamps_are_refused_before_anything_is_made(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    stamps: dict[str, object], horizon_offset_s: float, match: str,
) -> None:
    from orbitwin.verify import certify_candidate

    config, candidate, scenario = _setup(workshop_script)
    stamped = _stamped(
        candidate,
        scenario_sha256=scenario.source_sha256,
        horizon_s=scenario.max_time_s + horizon_offset_s,
    )
    with pytest.raises(PolicyError, match=match):
        certify_candidate(
            workshop_script, _stamped(stamped, **stamps), config, fake_gmat, fake_startup,
            tmp_path / "out", runs=1,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("announced", [False, True], ids=["4TA", "announced"])
def test_a_horizon_from_the_unprobed_state_says_to_probe(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, announced: bool
) -> None:
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.verify import certify_candidate

    config, candidate, _ = _setup(workshop_script)
    unprobed, _ = parse_gmat_script(workshop_script, config)
    horizon_s = unprobed.max_time_s
    if announced:
        config = dataclasses.replace(
            config, policy=dataclasses.replace(config.policy, horizon_s=horizon_s - 1.0)
        )
    stale = _stamped(candidate, scenario_sha256=unprobed.source_sha256, horizon_s=horizon_s)
    with pytest.raises(PolicyError, match="being certified under") as refused:
        certify_candidate(
            workshop_script, stale, config, fake_gmat, fake_startup, tmp_path / "out", runs=1
        )
    assert ("solve with --gmat-console" in str(refused.value)) is not announced
    assert not (tmp_path / "out").exists()


def test_the_pre_render_ignores_an_archived_stop(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verification flies first without the stop a previous run measured, so
    # the render certify checks up front must be that one.
    import orbitwin.verify as verify
    from orbitwin.verify import certify_candidate

    config, candidate, scenario = _setup(workshop_script)
    stale = _stamped(
        candidate,
        scenario_sha256=scenario.source_sha256,
        horizon_s=scenario.max_time_s,
        gmat_event={"stop_time_s": 1.0},
    )
    seen: list[dict[str, object]] = []

    def refuse(document: object, scenario: object, candidate: Candidate, config: object) -> None:
        seen.append(dict(candidate.metadata))
        raise PolicyError("refused by the test")

    monkeypatch.setattr(verify, "render_candidate", refuse)
    with pytest.raises(PolicyError, match="refused by the test"):
        certify_candidate(
            workshop_script, stale, config, fake_gmat, fake_startup, tmp_path / "out", runs=1
        )
    assert len(seen) == 1 and "gmat_event" not in seen[0]
    assert not (tmp_path / "out").exists()


def test_missing_stamps_are_warned_and_recorded(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.verify import certify_candidate

    config, candidate, scenario = _setup(workshop_script)
    with pytest.warns(RuntimeWarning, match="scenario_sha256 or horizon_s"):
        evidence = certify_candidate(
            workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "bare",
            runs=1,
        )
    assert evidence["archive_warnings"]
    stamped = _stamped(
        candidate, scenario_sha256=scenario.source_sha256, horizon_s=scenario.max_time_s,
        searched_score=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        evidence = certify_candidate(
            workshop_script, stamped, config, fake_gmat, fake_startup, tmp_path / "stamped",
            runs=1,
        )
    assert evidence["archive_warnings"] == []


def test_a_wrong_frame_leaves_the_output_free_for_a_retry(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.verify import certify_candidate

    config, candidate, scenario = _setup(workshop_script)
    candidate = _stamped(
        candidate, scenario_sha256=scenario.source_sha256, horizon_s=scenario.max_time_s
    )
    wrong = dataclasses.replace(
        candidate,
        burns=tuple(
            dataclasses.replace(burn, frame=("EarthMJ2000Eq", "Earth", "MJ2000Eq"))
            for burn in candidate.burns
        ),
    )
    output = tmp_path / "out"
    with pytest.raises(PolicyError, match="written in EarthMJ2000Eq"):
        certify_candidate(workshop_script, wrong, config, fake_gmat, fake_startup, output, runs=1)
    assert not output.exists()
    evidence = certify_candidate(
        workshop_script, candidate, config, fake_gmat, fake_startup, output, runs=1
    )
    assert evidence["stage"]


def test_certify_records_its_score_block_and_warns_on_another_searched_one(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """check_report_rows holds the day's config to the block; a search under another
    block optimised for other coefficients (R2: C_r 300 certified at 3000)."""
    from orbitwin.config import ScoreConfig
    from orbitwin.score import score_stamp
    from orbitwin.verify import certify_candidate

    config, candidate, scenario = _setup(workshop_script)
    config = dataclasses.replace(
        config, score=ScoreConfig(kt=0.001, ct=20000.0, kv=4.0, cv=0.8, delta_v_unit="km/s")
    )
    stamps = {"scenario_sha256": scenario.source_sha256, "horizon_s": scenario.max_time_s}
    same = _stamped(candidate, **stamps, searched_score=score_stamp(config.score))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        evidence = certify_candidate(
            workshop_script, same, config, fake_gmat, fake_startup, tmp_path / "same", runs=1,
        )
    assert evidence["archive_warnings"] == []
    assert evidence["score_block"] == score_stamp(config.score)
    # The hand check of the GUI's report reads it there (check_report_rows --verification).
    verification = json.loads(
        (tmp_path / "same" / "diagnostic" / "verification.json").read_text(encoding="utf-8")
    )
    assert verification["score_block"] == score_stamp(config.score)
    # And it verified the script certify submits.
    assert verification["candidate_script_sha256"] == evidence["final_script_sha256"]
    # And the rows before t = 0 as certify counts them, which the GUI path has no other record of.
    assert verification["frozen_rows"] == evidence["frozen_rows"]
    other = _stamped(candidate, **stamps, searched_score={**score_stamp(config.score), "cv": 0.5})
    with pytest.warns(RuntimeWarning, match="searched under Cv 0.5 km/s and is certified under "
                                            "Cv 0.8 km/s"):
        evidence = certify_candidate(
            workshop_script, other, config, fake_gmat, fake_startup, tmp_path / "other", runs=1,
        )
    (warning,) = evidence["archive_warnings"]
    assert "search again" in warning


def test_a_candidate_without_its_searched_block_is_warned(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """rank says it too; certify is handed candidates rank never saw."""
    from orbitwin.verify import certify_candidate

    config, candidate, scenario = _setup(workshop_script)
    stamped = _stamped(
        candidate, scenario_sha256=scenario.source_sha256, horizon_s=scenario.max_time_s
    )
    with pytest.warns(RuntimeWarning, match="carries no searched_score"):
        evidence = certify_candidate(
            workshop_script, stamped, config, fake_gmat, fake_startup, tmp_path / "out", runs=1,
        )
    (warning,) = evidence["archive_warnings"]
    assert "carries no searched_score" in warning
