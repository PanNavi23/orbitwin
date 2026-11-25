"""certify says how to check the report and which files to upload."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from orbitwin import cli

CHECKER = str(cli._CHECK_REPORT_ROWS)


def _result(script: str, conferred: str, reports: list[str] | None = None) -> dict[str, object]:
    return {
        "submission_script_name": script,
        "official_report_files": reports or ["ODC_2026_Team54.txt"],
        "stage": conferred,
        "conferred_stage": conferred,
    }


#: R1's certified hit, whose official report is ODC_2026_Team54.txt.
SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "report_bind" / "R1_hit_step60.script"


def _submission(tmp_path: Path, *names: str, official: str = "ODC_2026_Team54.txt") -> Path:
    """The submission folder, our script's official report named `official`."""
    submission = tmp_path / "final" / "submission"
    submission.mkdir(parents=True)
    for name in names:
        if name.endswith(".script"):
            text = SCRIPT.read_text(encoding="utf-8")
            assert "ReportFile1.Filename = 'ODC_2026_Team54.txt';" in text
            (submission / name).write_text(text.replace(
                "ReportFile1.Filename = 'ODC_2026_Team54.txt';",
                f"ReportFile1.Filename = '{official}';"), encoding="utf-8")
        else:
            (submission / name).write_text("x\n", encoding="utf-8")
    return submission


def test_a_public_run_checks_first_then_lists_the_upload(tmp_path: Path) -> None:
    submission = _submission(tmp_path, "ODC_2026_Team54.txt", "public_gmat_candidate.script")

    lines = cli.upload_summary(
        _result("public_gmat_candidate.script", "public_gmat_verified"),
        tmp_path / "final", "run/day-scored.yaml",
    )

    assert lines == [
        "orbitwin certify: step 7, must exit 0: " + shlex.join([
            sys.executable, CHECKER, str((submission / "ODC_2026_Team54.txt").resolve()),
            "--config", str(Path("run/day-scored.yaml").resolve()),
            "--certificate", str((submission / "certification.json").resolve()),
        ]),
        "orbitwin certify: before the GUI, the binding step 7 holds their report to, on ours, "
        "must exit 0: " + shlex.join([
            sys.executable, CHECKER, str((submission / "ODC_2026_Team54.txt").resolve()),
            "--config", str(Path("run/day-scored.yaml").resolve()),
            "--script", str(submission / "public_gmat_candidate.script"),
            "--verification", str(tmp_path / "final" / "diagnostic" / "verification.json"),
        ]),
        # The user's decision of 2026-11-06: the upload is the submission, on every path.
        "orbitwin certify: once step 7 exits 0 on the report, the upload is these files from "
        "this machine, as they are (docs/RUNBOOK.md step 6):",
        f"  {submission / 'public_gmat_candidate.script'}",
        f"  {submission / 'ODC_2026_Team54.txt'}",
        "orbitwin certify: never upload official_original.script, certification.json, the "
        ".diff.txt or anything else from run/; the first is the organizer's own input.",
        "orbitwin certify: public_gmat_candidate.script is named for what this run "
        "established, public_gmat_verified; do not rename it. team_solution.script is "
        "written only for a GMAT the organizer has confirmed as theirs "
        "(docs/RUNBOOK.md step 6).",
    ]


def test_the_printed_check_runs_from_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every path in it is absolute, given relative ones from the directory certify ran in."""
    monkeypatch.chdir(tmp_path)
    _submission(tmp_path, "ODC_2026_Team54.txt", "public_gmat_candidate.script")

    lines = cli.upload_summary(
        _result("public_gmat_candidate.script", "public_gmat_verified"), Path("final"), "day.yaml"
    )

    command = shlex.split(lines[0].split(": ", 2)[2])
    assert Path(CHECKER).is_file()
    listed = [line.strip() for line in lines if line.startswith("  ")]
    assert listed == [str(tmp_path / "final" / "submission" / name)
                      for name in ("public_gmat_candidate.script", "ODC_2026_Team54.txt")]
    assert "the upload is these files from this machine" in lines[2]
    binding = shlex.split(lines[1].split("must exit 0: ", 1)[1])
    assert all(Path(path).is_absolute() for path in binding[:3] + binding[4::2]), binding
    assert binding[-2:] == ["--verification",
                            str(tmp_path / "final" / "diagnostic" / "verification.json")]
    paths = [command[0], command[1], command[2], command[4], command[6]]
    assert all(Path(path).is_absolute() for path in paths), command
    assert command[4] == str(tmp_path / "day.yaml")


def test_without_a_config_it_says_to_certify_again(tmp_path: Path) -> None:
    """check_report_rows refuses --certificate without --config."""
    _submission(tmp_path, "ODC_2026_Team54.txt", "public_gmat_candidate.script")

    lines = cli.upload_summary(
        _result("public_gmat_candidate.script", "public_gmat_verified"), tmp_path / "final"
    )

    assert "certify again with the day's config" in lines[0]
    assert not [line for line in lines if "check_report_rows" in line]


def test_a_certified_run_adds_nothing_about_the_name(tmp_path: Path) -> None:
    _submission(tmp_path, "ODC_2026_Team54.txt", "team_solution.script")

    lines = cli.upload_summary(
        _result("team_solution.script", "certified"), tmp_path / "final", "day.yaml"
    )

    assert lines[3].endswith("team_solution.script")
    assert not [line for line in lines if "do not rename" in line]


def test_the_name_follows_what_this_run_conferred(tmp_path: Path) -> None:
    """`stage` can stay certified from elsewhere; the file is named for this run."""
    result = _result("public_gmat_candidate.script", "public_gmat_verified")
    result["stage"] = "certified"

    lines = cli.upload_summary(result, tmp_path / "final", "day.yaml")

    assert "established, public_gmat_verified;" in lines[-1]


def test_only_the_official_report_is_checked(tmp_path: Path) -> None:
    """Review of #185: a second ReportFile's report has no craft positions and exits 1, so
    checking every written report FAILED every candidate. Its final delta review: the upload
    named both, B.txt first when declared first. Only A.txt is checked and uploaded."""
    _submission(tmp_path, "A.txt", "B.txt", "public_gmat_candidate.script", official="A.txt")
    result = _result("public_gmat_candidate.script", "public_gmat_verified", ["B.txt", "A.txt"])

    lines = cli.upload_summary(result, tmp_path / "final", "day.yaml")

    listed = [line.strip() for line in lines if line.startswith("  ")]
    assert [Path(line).name for line in listed] == ["public_gmat_candidate.script", "A.txt"]
    # The upload is our script and the official report, from this machine (#189).
    upload = next(line for line in lines if "the upload is" in line)
    assert upload == ("orbitwin certify: once step 7 exits 0 on the report, the upload is "
                      "these files from this machine, as they are (docs/RUNBOOK.md step 6):")
    assert [line for line in lines if "B.txt" in line] == [
        "orbitwin certify: not uploaded or checked, another ReportFile's output: B.txt"]
    for kind in ("step 7, must exit 0", "before the GUI"):
        checks = [line for line in lines if kind in line]
        assert [Path(shlex.split(line.split("must exit 0: ", 1)[1])[2]).name
                for line in checks] == ["A.txt"], kind
    commands = cli.precheck_commands(result, tmp_path / "final", "day.yaml")
    # By hand, then step 7 itself, which certify's own report must pass (R3 T14).
    assert [Path(command[2]).name for command in commands] == ["A.txt", "A.txt"]
    assert "--certificate" not in commands[0] and "--certificate" in commands[1]


def test_a_script_that_names_no_official_report_says_why(tmp_path: Path) -> None:
    """Final delta review of #185: step 7's commands fell back to every report."""
    submission = _submission(tmp_path, "A.txt", "B.txt")
    (submission / "public_gmat_candidate.script").write_text("x\n", encoding="utf-8")
    result = _result("public_gmat_candidate.script", "public_gmat_verified", ["A.txt", "B.txt"])

    lines = cli.upload_summary(result, tmp_path / "final", "day.yaml")

    why = "public_gmat_candidate.script has no ReportFile GMAT writes"
    assert lines[0] == f"orbitwin certify: no step 7 command: {submission}/{why}"
    named = [line for line in lines if "must exit 0" in line or "A.txt" in line or "B.txt" in line]
    assert not named, lines
    # Nothing is named for upload when the official report is not known.
    assert f"orbitwin certify: no upload can be named: {submission}/{why}. Find the official " \
        "report in the script before step 7 (docs/RUNBOOK.md step 6)." in lines
    assert not [line for line in lines if line.startswith("  ")]
    assert not [line for line in lines if "GUI" in line and "upload" in line]
    with pytest.raises(cli.PrecheckUnavailable, match="has no ReportFile"):
        cli.precheck_commands(result, tmp_path / "final", "day.yaml")


def test_a_report_without_the_pursuers_velocity_is_pre_checked_on_its_end(
    tmp_path: Path,
) -> None:
    """Review of the docs, P1: it cannot show a burn, so check_report_rows needs the flag."""
    submission = _submission(tmp_path, "ODC_2026_Team54.txt", "public_gmat_candidate.script")
    script = submission / "public_gmat_candidate.script"
    text = script.read_text(encoding="utf-8")
    for axis in ("VX", "VY", "VZ"):
        assert f", SC_Human.EarthMJ2000Eq.{axis}" in text
        text = text.replace(f", SC_Human.EarthMJ2000Eq.{axis}", "")
    result = _result("public_gmat_candidate.script", "public_gmat_verified")
    config = str(Path(__file__).resolve().parents[2] / "config" / "ranking.yaml")
    with_velocity = cli.precheck_commands(result, tmp_path / "final", config)
    assert "--accept-end-time-binding-only" not in with_velocity[0]
    script.write_text(text, encoding="utf-8")
    command, step7 = cli.precheck_commands(result, tmp_path / "final", config)
    assert command[-1] == "--accept-end-time-binding-only"
    assert "--accept-end-time-binding-only" not in step7
    # Review of #191, P2-1: printed as "must exit 0", which it never does by design.
    [line] = [line for line in cli.upload_summary(result, tmp_path / "final", config)
              if "before the GUI" in line]
    assert ("on ours, must exit 3 (end time only; the 6b.7 by-eye check is mandatory): "
            + shlex.join(command)) in line


def test_a_file_that_is_not_there_says_so(tmp_path: Path) -> None:
    _submission(tmp_path, "public_gmat_candidate.script")

    lines = cli.upload_summary(
        _result("public_gmat_candidate.script", "public_gmat_verified"), tmp_path / "final",
        "day.yaml",
    )

    assert f"  {tmp_path / 'final' / 'submission' / 'ODC_2026_Team54.txt'}   (MISSING)" in lines


def test_a_path_with_a_space_survives_the_shell(tmp_path: Path) -> None:
    submission = _submission(tmp_path, "Report File1.txt", "public_gmat_candidate.script",
                             official="Report File1.txt")
    result = _result("public_gmat_candidate.script", "public_gmat_verified", ["Report File1.txt"])

    lines = cli.upload_summary(result, tmp_path / "final", "day.yaml")

    command = next(line for line in lines if "must exit 0" in line).split(": ", 2)[2]
    assert shlex.split(command)[2] == str((submission / "Report File1.txt").resolve())


def test_certify_prints_the_summary_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    project_root: Path, workshop_script: Path,
) -> None:
    monkeypatch.setattr(cli, "read_candidates", lambda path: [_Named("C-0")])
    monkeypatch.setattr(
        cli, "certify_candidate",
        lambda *args, **kwargs: _result("public_gmat_candidate.script", "public_gmat_verified"),
    )
    candidates = tmp_path / "candidates.json"
    candidates.write_text("[]", encoding="utf-8")
    config = str(project_root / "config" / "quick.yaml")
    _submission(tmp_path, "public_gmat_candidate.script")

    code = cli.main([
        "certify", str(workshop_script), "--config", config, "--rules", "preliminary",
        "--candidates", str(candidates), "--candidate", "C-0",
        "--gmat-console", "unused", "--startup-file", "unused",
        "--output", str(tmp_path / "final"),
    ])

    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["submission_script_name"] == "public_gmat_candidate.script"
    assert f"  {tmp_path / 'final' / 'submission' / 'ODC_2026_Team54.txt'}" in captured.err
    assert f"--config {shlex.quote(str(Path(config).resolve()))}" in captured.err


class _Named:
    def __init__(self, candidate_id: str) -> None:
        self.candidate_id = candidate_id
