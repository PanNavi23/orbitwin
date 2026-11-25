"""The GUI path on stock GmatConsole: step 7 holds the report to the plot-bearing script.

Certify stops at exact-run-01 on a template with plots and leaves the
staging/final.script the organizer's GUI runs. Its rows are the plot-free
copy's: the plots write none. So the plot-free certified report stands in
for the GUI's, and must pass against the staging script; a report of the same
candidate flown past the stop must not.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from orbitwin.cli import gui_precheck
from orbitwin.config import load_config
from orbitwin.io import candidate_from_dict, write_candidates
from orbitwin.models import file_sha256
from orbitwin.verify import certify_candidate

ROOT = Path(__file__).resolve().parents[2]
BIND = ROOT / "tests" / "fixtures" / "report_bind"
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
#: R1 with an OpenFrames view, its window and view settings our two-burn template's.
PLOTS = BIND / "R1_openframes.script"
REPORT = "ODC_2026_Team54.txt"

pytestmark = pytest.mark.gmat


def test_the_gui_report_is_held_to_the_staging_script(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    console, startup = gmat
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["team_number"] = 54
    raw["policy"].update(horizon_s=6711.667, mission_time_origin="earliest")
    raw["score"] = {"kt": 0.003, "ct": 5369.3336, "kv": 0.0012, "cv": 2000.0,
                    "time_unit": "s", "delta_v_unit": "m/s", "cr_km": 300.0}
    # The GUI path: with the console copy off, certify stops on the plots.
    raw["gmat"]["console_copy"] = False
    day = tmp_path / "day.yaml"
    day.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(day)
    # R1's certified hit (phase review R1); solved on the plot template it is stamped with that.
    data = json.loads((BIND / "R1_hit.candidate.json").read_text(encoding="utf-8"))
    certify_candidate(R1, candidate_from_dict(data), config, console, startup,
                      tmp_path / "plain", runs=1)
    data["metadata"]["scenario_sha256"] = file_sha256(PLOTS)
    with pytest.raises(Exception, match="OpenFrames"):
        certify_candidate(PLOTS, candidate_from_dict(data), config, console, startup,
                          tmp_path / "plots", runs=1)
    staging = tmp_path / "plots.refused" / "staging" / "final.script"
    assert "Create OpenFramesInterface" in staging.read_text(encoding="utf-8")

    def check(report: Path) -> subprocess.CompletedProcess[str]:
        usb = tmp_path / f"usb-{report.parent.parent.name}"
        usb.mkdir()
        shutil.copyfile(report, usb / REPORT)
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_report_rows.py"), str(usb / REPORT),
             "--config", str(day), "--script", str(staging), "--verification",
             str(tmp_path / "plots.refused" / "diagnostic" / "verification.json")],
            capture_output=True, text=True, check=False,
        )

    right = check(tmp_path / "plain" / "submission" / REPORT)
    assert right.returncode == 0, right.stdout + right.stderr
    assert "hold its 1 burn and no other" in right.stdout
    assert "score 96.3908" in right.stdout
    # The pre-check the refusal prints, with this run's own paths, passes on our report.
    command = gui_precheck(tmp_path / "plots.refused", str(day))
    assert command is not None
    pre = subprocess.run(command, capture_output=True, text=True, check=False)
    assert pre.returncode == 0, pre.stdout + pre.stderr
    assert "hold its 1 burn and no other" in pre.stdout
    # The same candidate's measuring flight, which runs past the stop.
    past = check(tmp_path / "plain" / "diagnostic" / "diagnostic-run" / "gmat-output" / REPORT)
    assert past.returncode == 2 and "not this script's report" in past.stderr, past.stderr


#: (template, T_max, the d_min window) of each rehearsal: examples/ranking/README.md.
REHEARSALS = {
    "R1": ("R1_NearPlane_Reachable.script", 6711.667, None),
    "R6": ("R6_InsertedLate_Aligned.script", 8511.667, "after_alignment"),
    "R8": ("R8_PursuerFirst_Aligned.script", 8511.667, "after_alignment"),
}


@pytest.mark.parametrize(("case", "max_step"), [("R1", None), ("R6", None), ("R8", 2700)])
def test_certify_ends_on_pre_check_pass(
    case: str, max_step: int | None, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    gmat: tuple[str, str],
) -> None:
    """Certify runs step 6's pre-check on its own report and ends on its verdict. R8 at
    MaxStep 2700 ends its alignment on two rows and needs its frozen rows passed."""
    from orbitwin import cli

    console, startup = gmat
    name, horizon_s, window = REHEARSALS[case]
    template = ROOT / "examples" / "ranking" / name
    data = json.loads((BIND / f"{case}_hit.candidate.json").read_text(encoding="utf-8"))
    if max_step is not None:
        text = template.read_text(encoding="utf-8")
        assert "DefaultProp.MaxStep = 1;" in text and "DefaultProp.InitialStepSize = 1;" in text
        template = tmp_path / name
        template.write_text(text.replace("DefaultProp.MaxStep = 1;", f"DefaultProp.MaxStep = "
                                         f"{max_step};").replace(
            "DefaultProp.InitialStepSize = 1;", "DefaultProp.InitialStepSize = 60;"),
            encoding="utf-8")
        data["metadata"]["scenario_sha256"] = file_sha256(template)
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["team_number"] = 54
    raw["policy"].update(horizon_s=horizon_s, mission_time_origin="earliest")
    if window is not None:
        raw["policy"]["closest_approach_window"] = window
    raw["score"] = {"kt": 0.003, "ct": horizon_s * 0.8, "kv": 0.0012, "cv": 2000.0,
                    "time_unit": "s", "delta_v_unit": "m/s", "cr_km": 300.0}
    day = tmp_path / "day.yaml"
    day.write_text(yaml.safe_dump(raw), encoding="utf-8")
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate_from_dict(data)])
    code = cli.main(["certify", str(template), "--config", str(day), "--candidates",
                     str(candidates), "--candidate", data["candidate_id"], "--gmat-console",
                     console, "--startup-file", startup, "--runs", "1", "--accept-gmat-loss",
                     "--output", str(tmp_path / "final")])
    err = capsys.readouterr().err
    assert code == 0, err
    assert err.splitlines()[-1] == cli.PRECHECK_PASS, err
    record = json.loads((tmp_path / "final" / "submission" / "certification.json").read_text())
    assert record["binding_pre_check"]["result"] == "pass"


@pytest.mark.parametrize("max_step", [None, 60, 2700])
@pytest.mark.parametrize("case", ["R6", "R8"])
def test_a_miss_burning_right_after_the_alignment_passes_its_pre_check(
    case: str, max_step: int | None, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    gmat: tuple[str, str],
) -> None:
    """The hit's first burn alone, at t = 0: a scored miss. At MaxStep 60 and 2700 R8's
    alignment ends on two rows; certify and step 7 must count t = 0 alike, or the
    pre-check FAILS a valid candidate."""
    from orbitwin import cli

    console, startup = gmat
    name, horizon_s, window = REHEARSALS[case]
    template = ROOT / "examples" / "ranking" / name
    data = json.loads((BIND / f"{case}_hit.candidate.json").read_text(encoding="utf-8"))
    data["burns"] = data["burns"][:1]
    assert data["burns"][0]["time_s"] == 0.0
    data["total_dv_km_s"] = data["vector_sum_dv_km_s"] = data["burns"][0]["magnitude_km_s"]
    data["candidate_id"] = f"C-miss-{case}-{max_step}"
    data["successful"] = False
    for key in ("intercept_time_s", "distance_at_completion_km", "score", "score_components"):
        data.pop(key, None)
    data["metadata"].pop("gmat_event", None)
    if max_step is not None:
        text = template.read_text(encoding="utf-8")
        template = tmp_path / name
        template.write_text(text.replace("DefaultProp.MaxStep = 1;", f"DefaultProp.MaxStep = "
                                         f"{max_step};").replace(
            "DefaultProp.InitialStepSize = 1;", "DefaultProp.InitialStepSize = 60;"),
            encoding="utf-8")
        data["metadata"]["scenario_sha256"] = file_sha256(template)
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["team_number"] = 54
    raw["policy"].update(horizon_s=horizon_s, mission_time_origin="earliest",
                         closest_approach_window=window)
    raw["score"] = {"kt": 0.003, "ct": horizon_s * 0.8, "kv": 0.0012, "cv": 2000.0,
                    "time_unit": "s", "delta_v_unit": "m/s", "cr_km": 300.0}
    day = tmp_path / "day.yaml"
    day.write_text(yaml.safe_dump(raw), encoding="utf-8")
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate_from_dict(data)])
    final = tmp_path / "final"
    code = cli.main(["certify", str(template), "--config", str(day), "--candidates",
                     str(candidates), "--candidate", data["candidate_id"], "--gmat-console",
                     console, "--startup-file", startup, "--runs", "1", "--accept-gmat-loss",
                     "--output", str(final)])
    err = capsys.readouterr().err
    assert code == 0, err
    assert err.splitlines()[-1] == cli.PRECHECK_PASS, err
    certificate = final / "submission" / "certification.json"
    record = json.loads(certificate.read_text(encoding="utf-8"))
    assert not record["candidate"]["successful"]
    verification = final / "diagnostic" / "verification.json"
    frozen = json.loads(verification.read_text(encoding="utf-8"))["frozen_rows"]
    assert frozen == record["frozen_rows"]
    report = final / "submission" / "ODC_2026_Team54.txt"
    checker = [sys.executable, str(ROOT / "scripts" / "check_report_rows.py"), str(report),
               "--config", str(day)]
    # Step 7 by the certificate, and by hand with the verification's frozen rows.
    for extra in (["--certificate", str(certificate)],
                  ["--script", str(final / "submission" / record["submission_script_name"]),
                   "--verification", str(verification), "--frozen-rows", str(frozen)]):
        done = subprocess.run([*checker, *extra], capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "a scored miss" in done.stdout


def _r1_day(tmp_path: Path, *, console_copy: bool = True) -> Path:
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["team_number"] = 54
    raw["policy"].update(horizon_s=6711.667, mission_time_origin="earliest")
    raw["gmat"]["console_copy"] = console_copy
    raw["score"] = {"kt": 0.003, "ct": 5369.3336, "kv": 0.0012, "cv": 2000.0,
                    "time_unit": "s", "delta_v_unit": "m/s", "cr_km": 300.0}
    day = tmp_path / "day.yaml"
    day.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return day


@pytest.mark.parametrize("console_copy", [False, True])
def test_a_planned_hit_flown_as_a_miss_on_plots_passes_its_pre_check(
    console_copy: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str], gmat: tuple[str, str]
) -> None:
    """R1's hit with its burn at 0.95 misses. Verify scores the flight it flew on to T_max,
    and the plots stop's pre-check must check that report, not the measuring flight's.
    Under the console copy the miss is certified, and its uploaded report pre-checked."""
    from orbitwin import cli

    console, startup = gmat
    data = json.loads((BIND / "R1_hit.candidate.json").read_text(encoding="utf-8"))
    for burn in data["burns"]:
        for key in ("dv_inertial_km_s", "dv_script_km_s"):
            burn[key] = [0.95 * value for value in burn[key]]
        burn["magnitude_km_s"] *= 0.95
    data["total_dv_km_s"] = data["vector_sum_dv_km_s"] = sum(
        burn["magnitude_km_s"] for burn in data["burns"])
    data["candidate_id"] = "C-loss-095"
    data["metadata"]["scenario_sha256"] = file_sha256(PLOTS)
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate_from_dict(data)])
    code = cli.main(["certify", str(PLOTS), "--config",
                     str(_r1_day(tmp_path, console_copy=console_copy)), "--candidates",
                     str(candidates), "--candidate", data["candidate_id"], "--gmat-console",
                     console, "--startup-file", startup, "--runs", "1", "--accept-gmat-loss",
                     "--output", str(tmp_path / "final")])
    err = capsys.readouterr().err
    if console_copy:
        assert code == 0, err
        assert err.splitlines()[-1] == cli.PRECHECK_PASS, err
        record = json.loads((tmp_path / "final" / "submission" / "certification.json")
                            .read_text(encoding="utf-8"))
        assert "console_copy" in record and not record["candidate"]["successful"]
        assert record["binding_pre_check"]["result"] == "pass"
        return
    assert code == 2 and cli.GUI_STOP in err
    assert err.splitlines()[-1] == cli.PRECHECK_PASS, err
    record = json.loads((tmp_path / "final.refused" / "diagnostic" / "verification.json")
                        .read_text(encoding="utf-8"))
    assert not record["candidate"]["successful"]
    assert Path(record["diagnostic_report"]).parent.parent.name == "diagnostic-run-to-horizon"
    assert record["binding_pre_check"]["result"] == "pass"


def test_a_second_report_file_is_not_pre_checked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], gmat: tuple[str, str]
) -> None:
    """R1 with a ReportFile2 of two columns: checked too, it exited 1 and FAILED every
    candidate. Only the official report is."""
    from orbitwin import cli

    console, startup = gmat
    text = R1.read_text(encoding="utf-8")
    anchor = "ReportFile1.AppendToExistingFile = false;\n"
    assert text.count(anchor) == 1
    template = tmp_path / "R1_two_reports.script"
    template.write_text(text.replace(anchor, anchor + (
        "Create ReportFile ReportFile2;\nReportFile2.Filename = 'extra_log.txt';\n"
        "ReportFile2.Add = {SC_Human.ElapsedSecs, SC_Human.EarthMJ2000Eq.X};\n"
        "ReportFile2.WriteReport = true;\n")), encoding="utf-8")
    data = json.loads((BIND / "R1_hit.candidate.json").read_text(encoding="utf-8"))
    data["metadata"]["scenario_sha256"] = file_sha256(template)
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate_from_dict(data)])
    code = cli.main(["certify", str(template), "--config", str(_r1_day(tmp_path)),
                     "--candidates", str(candidates), "--candidate", data["candidate_id"],
                     "--gmat-console", console, "--startup-file", startup, "--runs", "1",
                     "--output", str(tmp_path / "final")])
    err = capsys.readouterr().err
    assert code == 0, err
    assert err.splitlines()[-1] == cli.PRECHECK_PASS, err
    record = json.loads((tmp_path / "final" / "submission" / "certification.json")
                        .read_text(encoding="utf-8"))
    runs = record["binding_pre_check"]["runs"]
    # By hand, then step 7 itself, both on the official report alone.
    assert [Path(run["command"][2]).name for run in runs] == ["ODC_2026_Team54.txt"] * 2
    assert "extra_log.txt" in record["official_report_files"]


def test_a_report_without_the_pursuers_velocity_passes_on_its_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], gmat: tuple[str, str]
) -> None:
    """Review of the docs, P1: an official report without SC_Human's EarthMJ2000Eq velocity
    cannot show a burn, and every candidate's pre-check FAILED. It now passes end-time only."""
    from orbitwin import cli

    console, startup = gmat
    name, horizon_s, _ = REHEARSALS["R1"]
    text = (ROOT / "examples" / "ranking" / name).read_text(encoding="utf-8")
    velocity = ", SC_Human.EarthMJ2000Eq.VX, SC_Human.EarthMJ2000Eq.VY, SC_Human.EarthMJ2000Eq.VZ"
    assert velocity in text
    template = tmp_path / name
    template.write_text(text.replace(velocity, ""), encoding="utf-8")
    data = json.loads((BIND / "R1_hit.candidate.json").read_text(encoding="utf-8"))
    data["metadata"]["scenario_sha256"] = file_sha256(template)
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["team_number"] = 54
    raw["policy"].update(horizon_s=horizon_s, mission_time_origin="earliest")
    raw["score"] = {"kt": 0.003, "ct": horizon_s * 0.8, "kv": 0.0012, "cv": 2000.0,
                    "time_unit": "s", "delta_v_unit": "m/s", "cr_km": 300.0}
    day = tmp_path / "day.yaml"
    day.write_text(yaml.safe_dump(raw), encoding="utf-8")
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate_from_dict(data)])
    code = cli.main(["certify", str(template), "--config", str(day), "--candidates",
                     str(candidates), "--candidate", data["candidate_id"], "--gmat-console",
                     console, "--startup-file", startup, "--runs", "1", "--accept-gmat-loss",
                     "--output", str(tmp_path / "final")])
    err = capsys.readouterr().err
    assert code == 0, err
    assert err.splitlines()[-1] == cli.PRECHECK_PASS_END_TIME, err
    submission = tmp_path / "final" / "submission"
    record = json.loads((submission / "certification.json").read_text())
    assert record["binding_pre_check"]["result"] == "pass-end-time-only"
    header = (submission / REPORT).read_text(encoding="utf-8").splitlines()[0].split()
    assert "SC_Human.EarthMJ2000Eq.VX" not in header and "SC_Human.EarthMJ2000Eq.X" in header
