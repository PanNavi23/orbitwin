"""certify and baseline under gmat.console_copy, on the test double.

GmatConsole is played as R2026a answers a script with plots: exit 1 and an
error for every OpenFrames line, or, for a GroundTrack created before its
ReportFile, a clean exit and no report. The copy less those plots then runs,
its report is packaged beside the submitted script, plots and all, and
certification.json binds both.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
from conftest import openframes_errors, quick_config, refuse_runs_of, stamped

import orbitwin.verify as verify_module
from orbitwin import cli
from orbitwin.config import OrbitWinConfig
from orbitwin.errors import GmatRunError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.gmat_runner import EXIT_STATUS_ERROR, NO_FRESH_REPORT_ERROR
from orbitwin.io import write_candidates
from orbitwin.models import file_sha256
from orbitwin.search import zero_burn_candidate
from orbitwin.verify import baseline_original_script, certify_candidate

COPY_ON = "No console copy was run (gmat.console_copy)"


def _config(on: bool = True) -> OrbitWinConfig:
    config = quick_config()
    return dataclasses.replace(config, gmat=dataclasses.replace(config.gmat, console_copy=on))


def _console_refuses_plots(
    monkeypatch: pytest.MonkeyPatch, *, also: str | None = None,
    raise_in: str | None = None, fail_in: str | None = None,
    error: OSError | None = None,
) -> list[str]:
    """Play GmatConsole as R2026a on plots; ``also``: one more error beside them.

    ``raise_in``, ``fail_in``: a workdir whose run raises inside run_gmat, or fails.
    Returns the workdirs run, in order.
    """
    real = verify_module.run_gmat
    ran: list[str] = []

    def run(console, script, workdir, reports, *args, **kwargs):  # type: ignore[no-untyped-def]
        ran.append(Path(workdir).name)
        if Path(workdir).name == raise_in:
            raise error or OSError("the console vanished")
        result = real(console, script, workdir, reports, *args, **kwargs)
        if Path(workdir).name == fail_in:
            return dataclasses.replace(result, success=False, reports={}, report_sha256={},
                                       error=EXIT_STATUS_ERROR.format(1), return_code=1)
        if re.search(r"^Create OpenFrames", Path(script).read_text(encoding="ascii"), re.M):
            with open(result.stdout_path, "a", encoding="utf-8") as stdout:
                said = openframes_errors(Path(result.script_path))
                if also:
                    said.insert(0, also)
                stdout.write("".join(f"{line}\n" for line in said))
            return dataclasses.replace(result, success=False, reports={}, report_sha256={},
                                       error=EXIT_STATUS_ERROR.format(1), return_code=1)
        return result

    monkeypatch.setattr(verify_module, "run_gmat", run)
    return ran


def _certify(tmp_path: Path, script: Path, fake_gmat: Path, fake_startup: Path,
             config: OrbitWinConfig, runs: int = 2) -> dict[str, object]:
    scenario, _ = parse_gmat_script(script, config)
    candidate = stamped(zero_burn_candidate(scenario, config), script, config)
    return certify_candidate(script, candidate, config, fake_gmat, fake_startup,
                             tmp_path / "final", runs=runs)


def test_certify_packages_the_copys_report_beside_the_script_with_its_plots(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = _console_refuses_plots(monkeypatch)
    result = _certify(tmp_path, workshop_script, fake_gmat, fake_startup, _config())
    final = tmp_path / "final"
    submission = final / "submission"
    record = result["console_copy"]
    assert isinstance(record, dict)
    copy = Path(record["script"])
    assert copy.parent == final / "console_copy"
    # The submitted bytes ran first and were refused; every exact run then ran the copy.
    assert ran.count("exact-run-01") == 1 and "exact-run-02" not in ran
    assert record["submitted_run"]["return_code"] == 1
    runs = result["exact_runs"]
    assert isinstance(runs, list)
    assert [Path(run["workdir"]).name for run in runs] == ["console-exact-run-01",
                                                           "console-exact-run-02"]
    assert {run["script_sha256"] for run in runs} == {record["copy_sha256"], file_sha256(copy)}
    # The upload: our script, plots and all, and the copy's report, both bound.
    script = submission / str(result["submission_script_name"])
    assert "Create OpenFramesInterface" in script.read_text(encoding="ascii")
    assert file_sha256(script) == result["final_script_sha256"] == record["submitted_sha256"]
    report = submission / "ReportFile1.txt"
    assert report.read_bytes() == Path(runs[-1]["reports"]["ReportFile1.txt"]).read_bytes()
    listed = result["submission_files"]
    assert isinstance(listed, dict)
    assert listed["ReportFile1.txt"] == file_sha256(report)
    assert listed[script.name] == file_sha256(script)
    # The sibling, run off the submitted bytes, wrote the same report.
    parity = result["final_script_parity"]
    assert isinstance(parity, dict)
    assert parity["sibling_run"]["report_sha256"]["ReportFile1.txt"] == file_sha256(report)
    # Never certified: GmatConsole did not run the submitted bytes themselves.
    assert script.name != "team_solution.script"
    stage = result["candidate"]["stage_evidence"]  # type: ignore[index]
    assert "certification_of_console_copy" in stage
    # Item 10: the certificate holds no NaN or infinity a strict reader refuses.
    json.loads((submission / "certification.json").read_text(encoding="utf-8"),
               parse_constant=lambda name: pytest.fail(f"{name} in the certificate"))


def test_with_the_switch_off_the_refusal_stands_as_main_words_it(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _console_refuses_plots(monkeypatch)
    with pytest.raises(GmatRunError) as raised:
        _certify(tmp_path, workshop_script, fake_gmat, fake_startup, _config(on=False))
    assert cli.GUI_STOP in str(raised.value)
    assert "console copy" not in str(raised.value)
    assert not (tmp_path / "final.refused" / "console_copy").exists()


def test_an_error_besides_the_plots_gets_no_copy_and_says_so(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 9: said only when GMAT did report something else."""
    other = ('1: x.script: **** ERROR **** Interpreter Exception: The value of "maybe" for '
             'field "ShowPlot" on object "DefaultGroundTrackPlot" is not an allowed value.')
    _console_refuses_plots(monkeypatch, also=other)
    with pytest.raises(GmatRunError) as raised:
        _certify(tmp_path, workshop_script, fake_gmat, fake_startup, _config())
    message = str(raised.value)
    assert f"{COPY_ON}: GMAT reported errors besides the ones the plots explain" in message
    assert '"maybe"' in message and cli.GUI_STOP not in message


def test_errors_past_the_cap_get_no_copy_and_say_so(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _console_refuses_plots(monkeypatch)
    monkeypatch.setattr(verify_module, "_ERROR_LINES", 3)
    with pytest.raises(GmatRunError) as raised:
        _certify(tmp_path, workshop_script, fake_gmat, fake_startup, _config())
    assert f"{COPY_ON}: GMAT wrote more than 3 errors, so the plots may not be all" in str(
        raised.value)


def test_a_failure_not_about_plots_mentions_no_copy(
    tmp_path: Path, different_plane_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = verify_module.run_gmat

    def run(console, script, workdir, reports, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = real(console, script, workdir, reports, *args, **kwargs)
        if Path(workdir).name != "exact-run-01":
            return result
        return dataclasses.replace(result, success=False, reports={}, report_sha256={},
                                   error=EXIT_STATUS_ERROR.format(1), return_code=1)

    monkeypatch.setattr(verify_module, "run_gmat", run)
    with pytest.raises(GmatRunError) as raised:
        _certify(tmp_path, different_plane_script, fake_gmat, fake_startup, _config())
    assert "console copy" not in str(raised.value)


def test_a_copy_it_cannot_prove_gets_no_run_and_keeps_the_gui_advice(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Toggle of an XYPlot, which the copy keeps for GmatConsole, and of an OpenFrames
    view, which it leaves out: one line cannot be both, so no copy runs."""
    text = workshop_script.read_text(encoding="ascii")
    begin = re.search(r"^BeginMissionSequence;\r?\n", text, re.M)
    assert begin is not None
    xy = "Create XYPlot XY;\nXY.XVariable = HumanSC.ElapsedSecs;\n"
    bad = tmp_path / "workshop.script"
    bad.write_text(text[:begin.start()] + xy + text[begin.start():begin.end()]
                   + "Toggle XY DefaultOrbitView Off;\n" + text[begin.end():], encoding="ascii")
    ran = _console_refuses_plots(monkeypatch)
    with pytest.raises(GmatRunError) as raised:
        _certify(tmp_path, bad, fake_gmat, fake_startup, _config())
    message = str(raised.value)
    assert f"{COPY_ON}: line " in message and "not a plot command naming only" in message
    # The GUI path stays open, so the CLI still prints its pre-check.
    assert cli.GUI_STOP in message
    assert not any(name.startswith("console-") for name in ran)
    assert not list((tmp_path / "final.refused" / "console_copy").glob("*"))


@pytest.mark.parametrize("how", ["raise", "fail"])
def test_a_failed_run_of_the_copy_is_reported_as_the_copys(
    how: str, tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 8: a run of the copy that raises inside run_gmat, or fails, is named as the copy's."""
    where = "console-exact-run-01"
    _console_refuses_plots(monkeypatch, **{f"{how}_in": where})  # type: ignore[arg-type]
    with pytest.raises(GmatRunError) as raised:
        _certify(tmp_path, workshop_script, fake_gmat, fake_startup, _config())
    message = str(raised.value)
    assert "That was a run of final_console_copy.script, the console copy of final.script" in (
        message)
    assert "with the plots left out it still fails" in message
    said = ("Exact final run 1 failed: running it raised OSError: the console vanished"
            if how == "raise" else EXIT_STATUS_ERROR.format(1))
    assert message.startswith(said), message


def test_a_later_failed_run_of_the_copy_says_the_first_ran(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _console_refuses_plots(monkeypatch, fail_in="console-exact-run-02")
    with pytest.raises(GmatRunError) as raised:
        _certify(tmp_path, workshop_script, fake_gmat, fake_startup, _config())
    assert "its first run succeeded, so this run, not the copy, failed" in str(raised.value)


def test_a_ground_track_that_stops_the_report_gets_a_copy(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2026a: OpenFrames commented out, the GroundTrack before the ReportFile, a clean
    exit and no report."""
    lines = workshop_script.read_text(encoding="ascii").splitlines(keepends=True)
    kept = [line for line in lines if not re.match(
        r"(?:Create OpenFrames\w+ |DefaultOrbitView(?:_View)?\.)", line)]
    script = tmp_path / "ground.script"
    script.write_text("".join(kept), encoding="ascii")
    real = verify_module.run_gmat

    def run(console, path, workdir, reports, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = real(console, path, workdir, reports, *args, **kwargs)
        if re.search(r"^Create GroundTrack", Path(path).read_text(encoding="ascii"), re.M):
            return dataclasses.replace(result, success=False, reports={}, report_sha256={},
                                       error=NO_FRESH_REPORT_ERROR.format(reports[0]),
                                       return_code=0)
        return result

    monkeypatch.setattr(verify_module, "run_gmat", run)
    result = _certify(tmp_path, script, fake_gmat, fake_startup, _config(), runs=1)
    assert result["console_copy"]["removed"] == ["GroundTrack DefaultGroundTrackPlot"]  # type: ignore[index]


def test_baseline_runs_the_copy_and_says_so(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = _console_refuses_plots(monkeypatch)
    evidence = baseline_original_script(workshop_script, _config(), fake_gmat, fake_startup,
                                        tmp_path / "baseline")
    record = evidence["console_copy"]
    assert ran == ["run", "console-run"]
    assert evidence["ran_script"] == record["script"]
    assert evidence["ran_sha256"] == record["copy_sha256"] == evidence["run"]["script_sha256"]
    assert record["submitted_sha256"] == evidence["original_sha256_before"]
    assert file_sha256(workshop_script) == evidence["original_sha256_after"]
    notice = cli.baseline_notice(evidence, workshop_script)
    assert notice and "its copy proven to differ only by leaving out" in notice[0]


def test_baseline_off_refuses_as_before(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _console_refuses_plots(monkeypatch)
    with pytest.raises(GmatRunError, match="Solve still runs on its private copy"):
        baseline_original_script(workshop_script, _config(on=False), fake_gmat, fake_startup,
                                 tmp_path / "baseline")


def test_the_cli_names_the_upload_from_here_and_pre_checks_it(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import yaml

    _console_refuses_plots(monkeypatch)
    config = _config()
    day = tmp_path / "day.yaml"
    day.write_text(yaml.safe_dump({"gmat": {"console_copy": True}, "search": dataclasses.asdict(
        config.search)}), encoding="utf-8")
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = stamped(zero_burn_candidate(scenario, config), workshop_script, config)
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate])
    final = tmp_path / "final"
    # quick_config is under the preliminary rules, which #190 makes certify name.
    code = cli.main(["certify", str(workshop_script), "--config", str(day),
                     "--rules", "preliminary", "--candidates",
                     str(candidates), "--candidate", candidate.candidate_id, "--gmat-console",
                     str(fake_gmat), "--startup-file", str(fake_startup), "--runs", "1",
                     "--output", str(final)])
    err = capsys.readouterr().err
    assert code == 0, err
    submission = final / "submission"
    record = json.loads((submission / "certification.json").read_text(encoding="utf-8"))
    script = submission / record["submission_script_name"]
    assert "(docs/RUNBOOK.md step 6c)" in err and "GUI" not in err.split("step 6c")[1]
    # Step 7 and its pre-check carry --rules, as certify needed it (#190).
    printed = [line for line in err.splitlines()
               if "step 7, must exit 0: " in line or "on ours, must exit" in line]
    assert len(printed) == 2 and all(line.endswith("--rules preliminary") for line in printed)
    assert f"\n  {submission / 'ReportFile1.txt'}\n  {script}\n" in err
    # The pre-check and step 7 both name the report that is uploaded.
    commands = cli.precheck_commands(record, final, str(day))
    assert commands[0][2] == str((submission / "ReportFile1.txt").resolve())
    assert "step 7, must exit 0: " in err and str(submission / "ReportFile1.txt") in err
    assert record["binding_pre_check"]["result"] == "skipped"


def test_a_refusal_of_other_bytes_gets_no_copy(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The copy is of the bytes parsed; a refusal of any others proves nothing about them."""
    _console_refuses_plots(monkeypatch)
    refused = verify_module.run_gmat

    def run(console, script, workdir, reports, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = refused(console, script, workdir, reports, *args, **kwargs)
        if Path(workdir).name == "exact-run-01":
            return dataclasses.replace(result, script_sha256="0" * 64)
        return result

    monkeypatch.setattr(verify_module, "run_gmat", run)
    with pytest.raises(GmatRunError, match=f"{re.escape(COPY_ON)}: final.script changed"):
        _certify(tmp_path, workshop_script, fake_gmat, fake_startup, _config())
    assert not list((tmp_path / "final.refused" / "console_copy").glob("*"))


def test_a_toggle_of_the_plots_is_certified_through_the_copy(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GmatConsole adds a record for the Toggle ("Nonexistent Subscriber ... referenced in
    the Toggle"); the plots explain it, and the copy leaves the Toggle out with them."""
    text = workshop_script.read_text(encoding="ascii")
    begin = re.search(r"^BeginMissionSequence;\r?\n", text, re.M)
    assert begin is not None
    script = tmp_path / "toggled.script"
    script.write_text(text[:begin.end()] + "Toggle DefaultOrbitView Off;\n" + text[begin.end():],
                      encoding="ascii")
    _console_refuses_plots(monkeypatch)
    result = _certify(tmp_path, script, fake_gmat, fake_startup, _config(), runs=1)
    record = result["console_copy"]
    assert isinstance(record, dict)
    assert "Toggle" not in Path(record["script"]).read_text(encoding="ascii")
    uploaded = tmp_path / "final" / "submission" / str(result["submission_script_name"])
    assert "Toggle DefaultOrbitView Off;" in uploaded.read_text(encoding="ascii")


def test_a_copy_proven_against_other_bytes_is_refused(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M10 of the review of #189: custody holds the copy to the script being certified."""
    _console_refuses_plots(monkeypatch)
    real = verify_module.console_copy_for

    def other(*args, **kwargs):  # type: ignore[no-untyped-def]
        path, copy = real(*args, **kwargs)
        return path, dataclasses.replace(copy, submitted_sha256="0" * 64)

    monkeypatch.setattr(verify_module, "console_copy_for", other)
    with pytest.raises(GmatRunError, match=r"The console copy was proven against 0{64}"):
        _certify(tmp_path, workshop_script, fake_gmat, fake_startup, _config())


def test_a_later_refusal_of_the_submitted_bytes_gets_no_copy(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M14 of the review of #189: once GmatConsole has run the submitted bytes, a later
    run refusing them is that run's failure, never a reason to switch to the copy."""
    refuse_runs_of(monkeypatch, "exact-run-02")
    ran: list[str] = []
    counted = verify_module.run_gmat

    def run(console, script, workdir, reports, *args, **kwargs):  # type: ignore[no-untyped-def]
        ran.append(Path(workdir).name)
        return counted(console, script, workdir, reports, *args, **kwargs)

    monkeypatch.setattr(verify_module, "run_gmat", run)
    with pytest.raises(GmatRunError) as raised:
        _certify(tmp_path, workshop_script, fake_gmat, fake_startup, _config(), runs=3)
    assert "Exact final run 2 failed" in str(raised.value) or EXIT_STATUS_ERROR.format(1) in str(
        raised.value)
    assert "console copy" not in str(raised.value)
    assert not any(name.startswith("console-") for name in ran)
    assert "exact-run-03" not in ran


def test_a_full_disk_under_the_copys_run_is_no_answer_not_a_refusal(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#199 review P2a: verify wrapped the OSError run_gmat raised as GmatRunError, and
    certify called this machine's failure a verdict (exit 2)."""
    import errno

    from orbitwin.gmat_runner import no_answer

    _console_refuses_plots(monkeypatch, raise_in="console-exact-run-02",
                           error=OSError(errno.ENOSPC, "No space left on device"))
    with pytest.raises(GmatRunError) as raised:
        _certify(tmp_path, workshop_script, fake_gmat, fake_startup, _config())
    assert no_answer(raised.value)
    # Not a refusal, so not kept as one (#199 review, Low 3).
    assert (tmp_path / "final.no-answer").is_dir() and not (tmp_path / "final.refused").exists()


@pytest.mark.parametrize(("code", "answered"), [("ENOSPC", False), ("ENOENT", True)])
def test_a_copy_this_machine_cannot_write_is_no_answer_not_the_gui_path(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, code: str, answered: bool,
) -> None:
    """#199 review P2a: write_console_copy's OSError carried "Run it in the GMAT GUI" and
    went down the GUI path. A missing file is no machine failure: the GUI advice stands."""
    import errno

    from orbitwin.gmat_runner import no_answer

    def unwritable(*_: object) -> None:
        raise OSError(getattr(errno, code), "cannot write the copy")

    _console_refuses_plots(monkeypatch)
    monkeypatch.setattr(verify_module, "write_console_copy", unwritable)
    with pytest.raises(GmatRunError) as raised:
        _certify(tmp_path, workshop_script, fake_gmat, fake_startup, _config())
    assert no_answer(raised.value) is not answered
    assert cli.GUI_STOP in str(raised.value)
    assert (tmp_path / ("final.refused" if answered else "final.no-answer")).is_dir()
