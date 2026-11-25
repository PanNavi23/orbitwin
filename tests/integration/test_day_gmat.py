"""orbitwin day on stock GmatConsole R2026a, end to end, with a search cut to seconds.

R1 as written, and R1 with an OpenFrames view (report_bind/R1_openframes.script),
whose report comes from certify's console copy (RUNBOOK step 6c). Each ends with the
two files in <final root>/<UTC>/, `latest` pointing there and step 7 passing on the
copies.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import quick_config

from orbitwin.cli import main
from orbitwin.day import NOTE_NAME

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
PLOTS = ROOT / "tests" / "fixtures" / "report_bind" / "R1_openframes.script"
REPORT = "ODC_2026_Team54.txt"

pytestmark = pytest.mark.gmat


#: The search values quick_config sets, laid over config/ranking.yaml's.
QUICK_SEARCH_KEYS = ("one_burn_samples", "one_burn_grid_burns", "one_burn_grid_arrivals",
                     "max_revolutions", "keep_candidates", "render_top", "event_step_s",
                     "event_refine_top", "two_burn_enabled", "two_burn_samples")


def _quick_base(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["team_number"] = 54
    quick = dataclasses.asdict(quick_config().search)
    raw["search"].update({key: quick[key] for key in QUICK_SEARCH_KEYS}, ranking_search=False)
    raw["gmat"].update(correct_top=2, verify_misses=1)
    path = tmp_path / "quick-ranking.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _through(folder: Path, code: int, *also: str) -> int:
    """As two people would: each readback confirmed once shown, rank's warnings accepted."""
    for _ in range(6):
        if code == 0:
            break
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        confirmed = state.get("confirmed_readbacks") or {}
        pending = [step for step, digest in (state.get("shown_readbacks") or {}).items()
                   if (confirmed.get(step) or {}).get("sha256") != digest]
        decision = (["--confirm-readback", pending[0], "--accept-config-warnings"] if pending
                    else ["--accept-rank-warnings"] if state.get("shown_rank_warnings")
                    and state.get("accepted_rank_warnings") != state["shown_rank_warnings"]
                    else None)
        if decision is None:
            break
        code = main(["day", "--resume", str(folder), "--allow-non-main", *decision, *also])
    return code


def _run(
    tmp_path: Path, gmat: tuple[str, str], script: Path, by_eye: tuple[str, ...] = ()
) -> Path:
    console, startup = gmat
    final = tmp_path / "final"
    code = main(["day", str(script), "--horizon-s", "6711.667", "--mission-boundary-km", "17000",
                 "--kt-per-s", "0.003", "--ct-s", "5369.3336", "--kv-per-m-s", "0.0012",
                 "--cv-m-s", "2000", "--cr-km", "300", "--base", str(_quick_base(tmp_path)),
                 "--gmat-console", console, "--startup-file", startup,
                 "--work-root", str(tmp_path / "days"), "--final-root", str(final),
                 # A branch under review is not origin/main yet.
                 "--allow-non-main"])
    (folder,) = (tmp_path / "days").iterdir()
    code = _through(folder, code)
    if by_eye:
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        shown = {name for name, item in state["checked"].items() if item.get("end_time_only")}
        assert shown == set(by_eye), state["checked"]
        given = [word for name in by_eye for word in ("--by-eye-passed", name)]
        code = main(["day", "--resume", str(folder), "--allow-non-main", *given])
    assert code == 0
    target = final / os.readlink(final / "latest")
    script_name = "public_gmat_candidate.script"
    assert sorted(path.name for path in target.iterdir()) == sorted(
        [REPORT, script_name, "SHA256SUMS", NOTE_NAME])
    summary = (target / NOTE_NAME).read_text(encoding="utf-8")
    assert "PRE-CHECK    : PASS" in summary and "audit        : audit_burn_table.py exit 0" in summary
    assert ("PRE-CHECK    : PASS (END TIME ONLY, BY EYE)" in summary) == bool(by_eye)
    chosen = summary.split("chosen       : ", 1)[1].split(" ", 1)[0]
    certificate = folder / "run" / chosen / "submission" / "certification.json"
    # Step 7 again, by hand, on the copies.
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_report_rows.py"), str(target / REPORT),
         "--config", str(folder / "run" / "day-scored.yaml"), "--certificate", str(certificate),
         "--script", str(target / script_name)],
        capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert done.returncode == 0, done.stdout + done.stderr
    score = summary.split("score        : ", 1)[1].split(" ", 1)[0]
    assert f"-> score {score}" in done.stdout or f"= {score}" in done.stdout
    return certificate


def test_the_day_on_r1(tmp_path: Path, gmat: tuple[str, str]) -> None:
    certificate = _run(tmp_path, gmat, R1)
    assert "console_copy" not in json.loads(certificate.read_text(encoding="utf-8"))


def test_the_day_on_r1_with_plots_runs_the_console_copy(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    certificate = _run(tmp_path, gmat, PLOTS)
    body = json.loads(certificate.read_text(encoding="utf-8"))
    assert body["console_copy"]["removed"]
    uploaded = (certificate.parent / body["submission_script_name"]).read_text(encoding="ascii")
    assert "Create OpenFramesInterface" in uploaded


def test_the_day_on_r1_without_the_pursuers_velocities_takes_a_by_eye_pass(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    """The report cannot show a burn, so certify ends on PRE-CHECK PASS (end time only) and
    the day waits for 6b.7's by-eye comparison (#193 review, item 3)."""
    text = R1.read_text(encoding="ascii")
    dropped = "".join(f", SC_Human.EarthMJ2000Eq.{axis}" for axis in ("VX", "VY", "VZ"))
    assert text.count(dropped) == 1
    script = tmp_path / "R1_no_velocity.script"
    script.write_text(text.replace(dropped, ""), encoding="ascii")
    certificate = _run(tmp_path, gmat, script, by_eye=("final", "final-01"))
    body = json.loads(certificate.read_text(encoding="utf-8"))
    assert body["binding_pre_check"]["result"] == "pass-end-time-only"
