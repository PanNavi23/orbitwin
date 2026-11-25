"""On stock GmatConsole, R1 solved and certified uploads ODC_2026_Team54.txt."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orbitwin.config import load_config
from orbitwin.io import read_candidates
from orbitwin.models import file_sha256
from orbitwin.pipeline import solve_scenario
from orbitwin.verify import certify_candidate

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
REPORT = "ODC_2026_Team54.txt"

pytestmark = pytest.mark.gmat


def test_r1_certifies_under_the_team_number(tmp_path: Path, gmat: tuple[str, str]) -> None:
    console, startup = gmat
    config = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    config = dataclasses.replace(config, policy=dataclasses.replace(config.policy, team_number=54))
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, horizon_s=6711.667)
    )
    assert config.policy.team_number == 54
    original = R1.read_bytes()

    summary = solve_scenario(
        R1, config, tmp_path / "solve", gmat_console=console, startup_file=startup
    )
    best = summary["best_candidate"]
    assert best is not None and best["successful"], summary["warnings"]
    candidate = next(
        item
        for item in read_candidates(tmp_path / "solve" / "candidates.json")
        if item.candidate_id == best["candidate_id"]
    )

    evidence = certify_candidate(
        R1, candidate, config, console, startup, tmp_path / "certify", runs=3
    )
    submission = tmp_path / "certify" / "submission"
    assert (submission / REPORT).is_file()
    assert not list(submission.glob("*TeamXX*"))
    assert evidence["official_report_files"] == [REPORT]
    hashes = {run["report_sha256"][REPORT] for run in evidence["exact_runs"]}
    assert len(evidence["exact_runs"]) == 3 and len(hashes) == 1
    assert file_sha256(submission / REPORT) in hashes
    assert evidence["submission_files"][REPORT] in hashes
    assert {name for run in evidence["exact_runs"] for name in run["report_sha256"]} == {REPORT}

    script = (submission / evidence["submission_script_name"]).read_text(encoding="utf-8")
    assert f"ReportFile1.Filename = '{REPORT}';" in script
    assert "TeamXX" not in script
    assert (submission / "official_original.script").read_bytes() == original
    assert R1.read_bytes() == original
