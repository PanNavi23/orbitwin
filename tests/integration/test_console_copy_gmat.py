"""gmat.console_copy on stock GmatConsole R2026a.

GmatConsole refuses R1 with an OpenFrames view (report_bind/R1_openframes.script,
exit 1), and with a GroundTrack created before its ReportFile it writes no report.
Certify then runs the proven copy less those plots, packages its report beside the
script with the plots, and ends on PRE-CHECK PASS; step 7 passes on that report.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from orbitwin import cli
from orbitwin.io import candidate_from_dict, write_candidates
from orbitwin.models import file_sha256

ROOT = Path(__file__).resolve().parents[2]
BIND = ROOT / "tests" / "fixtures" / "report_bind"
PLOTS = BIND / "R1_openframes.script"
REPORT = "ODC_2026_Team54.txt"

pytestmark = pytest.mark.gmat


def _day(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["team_number"] = 54
    raw["policy"].update(horizon_s=6711.667, mission_time_origin="earliest")
    raw["score"] = {"kt": 0.003, "ct": 5369.3336, "kv": 0.0012, "cv": 2000.0,
                    "time_unit": "s", "delta_v_unit": "m/s", "cr_km": 300.0}
    raw["gmat"]["console_copy"] = True
    day = tmp_path / "day.yaml"
    day.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return day


def _ground_track_r1(tmp_path: Path) -> Path:
    """R1 with the two-burn template's GroundTrack created before its ReportFile, and no
    OpenFrames."""
    text = (ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script").read_text(
        encoding="ascii")
    block = "".join(
        line.replace("HumanSC", "SC_Human").replace("AlienSC", "SC_Alien")
        for line in (ROOT / "examples" / "two_burn_rendezvous.script").read_text(
            encoding="ascii").splitlines(keepends=True)
        if line.startswith(("Create GroundTrack ", "DefaultGroundTrackPlot."))
    ) + "\n"
    assert block.startswith("Create GroundTrack ")
    assert text.count("Create ReportFile ReportFile1;\n") == 1
    path = tmp_path / "R1_ground_track.script"
    path.write_text(text.replace("Create ReportFile ReportFile1;\n",
                                 block + "Create ReportFile ReportFile1;\n"), encoding="ascii")
    return path


def _toggled_r1(tmp_path: Path) -> Path:
    """R1's OpenFrames template turning its view off before the flight and on after."""
    text = PLOTS.read_text(encoding="ascii")
    begin = "BeginMissionSequence;\n"
    assert text.count(begin) == 1
    text = text.replace(begin, begin + "Toggle DefaultOrbitView Off;\n")
    path = tmp_path / "R1_openframes_toggle.script"
    path.write_text(text.rstrip("\n") + "\nToggle DefaultOrbitView On;\n", encoding="ascii")
    return path


def _pen_r1(tmp_path: Path) -> Path:
    """R1's OpenFrames template lifting the view's pen and putting it down, with an XYPlot
    it marks a point on: the copy leaves out the first two and keeps the XYPlot's."""
    text = PLOTS.read_text(encoding="ascii")
    begin = "BeginMissionSequence;\n"
    xy = "Create XYPlot XY;\nXY.XVariable = SC_Human.ElapsedSecs;\nXY.YVariables = {SC_Human.SMA};\n"
    path = tmp_path / "R1_openframes_pen.script"
    path.write_text(text.replace(begin, xy + begin + "PenUp DefaultOrbitView;\nMarkPoint XY;\n"
                                 "PenDown DefaultOrbitView;\n"), encoding="ascii")
    return path


@pytest.mark.parametrize("template", ["openframes", "openframes_toggle", "openframes_pen",
                                      "ground_track"])
def test_certify_uploads_the_copys_report_beside_the_plots(
    template: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], gmat: tuple[str, str]
) -> None:
    console, startup = gmat
    script = {"openframes": lambda: PLOTS, "openframes_toggle": lambda: _toggled_r1(tmp_path),
              "openframes_pen": lambda: _pen_r1(tmp_path),
              "ground_track": lambda: _ground_track_r1(tmp_path)}[template]()
    day = _day(tmp_path)
    data = json.loads((BIND / "R1_hit.candidate.json").read_text(encoding="utf-8"))
    data["metadata"]["scenario_sha256"] = file_sha256(script)
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate_from_dict(data)])
    final = tmp_path / "final"
    code = cli.main(["certify", str(script), "--config", str(day), "--candidates",
                     str(candidates), "--candidate", data["candidate_id"], "--gmat-console",
                     console, "--startup-file", startup, "--runs", "2", "--output", str(final)])
    err = capsys.readouterr().err
    assert code == 0, err
    assert err.splitlines()[-1] == cli.PRECHECK_PASS, err
    submission = final / "submission"
    record = json.loads((submission / "certification.json").read_text(encoding="utf-8"))
    copy = record["console_copy"]
    # GmatConsole refused the submitted bytes as R2026a does: exit 1, or no report.
    assert copy["submitted_run"]["return_code"] == (0 if template == "ground_track" else 1)
    assert {run["script_sha256"] for run in record["exact_runs"]} == {copy["copy_sha256"]}
    uploaded = submission / record["submission_script_name"]
    kept = "Create GroundTrack" if template == "ground_track" else "Create OpenFramesInterface"
    assert kept in uploaded.read_text(encoding="ascii")
    if template == "openframes_pen":
        assert "MarkPoint XY;" in Path(copy["script"]).read_text(encoding="ascii")
    assert not re.search(r"^(?:Create (?:OpenFrames|GroundTrack)|Toggle|PenUp|PenDown)",
                         Path(copy["script"]).read_text(encoding="ascii"), re.M)
    if template == "openframes_toggle":
        assert uploaded.read_text(encoding="ascii").count("Toggle DefaultOrbitView") == 2
    report = submission / REPORT
    sibling = record["final_script_parity"]["sibling_run"]["report_sha256"][REPORT]
    assert file_sha256(report) == sibling == record["submission_files"][REPORT]
    assert record["binding_pre_check"]["result"] == "pass"
    step7 = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_report_rows.py"), str(report),
         "--config", str(day), "--certificate", str(submission / "certification.json")],
        capture_output=True, text=True, check=False,
    )
    assert step7.returncode == 0, step7.stdout + step7.stderr
    assert "rebuilt and proven again here" in step7.stderr


def test_baseline_runs_the_two_burn_templates_copy(tmp_path: Path, gmat: tuple[str, str]) -> None:
    """Our two-burn template: OpenFrames, a GroundTrack before the ReportFile."""
    from orbitwin.config import load_config
    from orbitwin.verify import baseline_original_script

    console, startup = gmat
    raw = yaml.safe_load((ROOT / "config" / "standard.yaml").read_text(encoding="utf-8"))
    raw.setdefault("gmat", {})["console_copy"] = True
    day = tmp_path / "day.yaml"
    day.write_text(yaml.safe_dump(raw), encoding="utf-8")
    evidence = baseline_original_script(ROOT / "examples" / "two_burn_rendezvous.script",
                                        load_config(day), console, startup,
                                        tmp_path / "baseline")
    copy = evidence["console_copy"]
    assert copy["removed"] == ["OpenFramesInterface DefaultOrbitView",
                               "GroundTrack DefaultGroundTrackPlot",
                               "OpenFramesView DefaultOrbitView_View"]
    assert evidence["run"]["success"] and evidence["run"]["script_sha256"] == copy["copy_sha256"]
    assert Path(evidence["run"]["reports"]["ReportFile1.txt"]).stat().st_size > 0
