"""On stock GmatConsole: certify judges the official report on its printed rows.

R3 (2026-11-06) T16: a ReportFile1 in the organizer's workshop style, the
pursuer's time and position only, had no upload: every fallback ended on
"cannot locate X/Y/Z". T14: at Precision 4 certify's own step 7 refused its
report, 1258 m inside against the 1500 m stop depth, and the certificate
claimed 95.7635 where the printed rows gave 95.7623.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from orbitwin.io import candidate_from_dict, write_candidates
from orbitwin.models import file_sha256

ROOT = Path(__file__).resolve().parents[2]
BIND = ROOT / "tests" / "fixtures" / "report_bind"
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
REPORT = "ODC_2026_Team54.txt"

pytestmark = pytest.mark.gmat


def _certify(tmp_path: Path, capsys: pytest.CaptureFixture[str],
             edit: tuple[str, str], gmat: tuple[str, str]) -> tuple[int, str, Path, Path]:
    """R1's certified hit on R1 with ``edit`` made, through the CLI."""
    from orbitwin import cli

    console, startup = gmat
    text = R1.read_text(encoding="utf-8")
    assert edit[0] in text
    template = tmp_path / R1.name
    template.write_text(text.replace(edit[0], edit[1]), encoding="utf-8")
    data = json.loads((BIND / "R1_hit.candidate.json").read_text(encoding="utf-8"))
    data["metadata"]["scenario_sha256"] = file_sha256(template)
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["team_number"] = 54
    raw["policy"].update(horizon_s=6711.667, mission_time_origin="earliest")
    raw["score"] = {"kt": 0.003, "ct": 5369.3336, "kv": 0.0012, "cv": 2000.0,
                    "time_unit": "s", "delta_v_unit": "m/s", "cr_km": 300.0}
    day = tmp_path / "day.yaml"
    day.write_text(yaml.safe_dump(raw), encoding="utf-8")
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate_from_dict(data)])
    code = cli.main(["certify", str(template), "--config", str(day), "--candidates",
                     str(candidates), "--candidate", data["candidate_id"], "--gmat-console",
                     console, "--startup-file", startup, "--runs", "1", "--accept-gmat-loss",
                     "--output", str(tmp_path / "final")])
    return code, capsys.readouterr().err, day, tmp_path / "final" / "submission"


def _step7(day: Path, submission: Path) -> subprocess.CompletedProcess[str]:
    import sys

    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_report_rows.py"), str(submission / REPORT),
         "--config", str(day), "--certificate", str(submission / "certification.json")],
        capture_output=True, text=True, check=False)


def test_a_report_without_the_target_is_certified_on_the_diagnostic_flight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], gmat: tuple[str, str]
) -> None:
    from orbitwin import cli

    workshop = ("ReportFile1.Add = {SC_Human.ElapsedSecs, SC_Human.EarthMJ2000Eq.X, "
                "SC_Human.EarthMJ2000Eq.Y, SC_Human.EarthMJ2000Eq.Z};")
    text = R1.read_text(encoding="utf-8")
    add = next(line for line in text.splitlines() if line.startswith("ReportFile1.Add"))
    code, err, day, submission = _certify(tmp_path, capsys, (add, workshop), gmat)
    assert code == 0, err
    # No velocity either: the hand binding held its end alone.
    assert err.splitlines()[-1] == cli.PRECHECK_PASS_NO_TARGET_END_TIME, err
    record = json.loads((submission / "certification.json").read_text(encoding="utf-8"))
    assert record["binding_pre_check"]["result"] == "pass-target-not-in-report-end-time-only"
    hand, step7 = record["binding_pre_check"]["runs"]
    assert hand["exit_code"] == 7 and step7["exit_code"] == 0
    rows = record["official_report_rows"]
    assert rows["carries_target_position"] is False and rows["last_row_distance_km"] is None
    assert rows["proof"].startswith("the official report carries no SC_Alien position")
    assert rows["score"] == record["final_script_parity"]["replayed_score"]
    done = _step7(day, submission)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines()[-1] == (
        "WARNING: the official report carries no SC_Alien position; d_min and the interception "
        "were proven on certify's diagnostic flight of the same script, not on this report")
    assert f"score {rows['score']:.4f}" in done.stdout


@pytest.mark.parametrize("precision", [4, 6, 8])
def test_certify_passes_its_own_step_7_or_refuses_at_a_coarse_precision(
    precision: int, tmp_path: Path, capsys: pytest.CaptureFixture[str], gmat: tuple[str, str]
) -> None:
    """At Precision 4 coordinates near 18000 km print to 10 km: the stop is deepened to the
    2.5 km cap, and certify passes only where step 7 does. 6 and 8 must pass."""
    from orbitwin import cli

    code, err, day, submission = _certify(
        tmp_path, capsys, ("ReportFile1.Precision = 16;", f"ReportFile1.Precision = {precision};"), gmat)
    if code != 0:
        assert precision == 4 and not submission.exists(), err
        assert "step 7 would refuse it" in err, err
        return
    assert err.splitlines()[-1] == cli.PRECHECK_PASS, err
    record = json.loads((submission / "certification.json").read_text(encoding="utf-8"))
    rows = record["official_report_rows"]
    margin = record["candidate"]["metadata"]["gmat_event"]["printed_margin_km"]
    assert margin > 0.0 and rows["stop_depth_km"] >= 1.5 * 0.99, (margin, rows)
    # The certificate's score is the printed rows' score, which step 7 prints.
    assert rows["score"] == record["final_script_parity"]["replayed_score"]
    done = _step7(day, submission)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"The score matches the certificate's {rows['score']:.4f}" in done.stdout
