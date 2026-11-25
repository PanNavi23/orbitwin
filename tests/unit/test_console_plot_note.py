"""certify and baseline say why GmatConsole could not run a script that draws.

R2026a's GmatConsole refuses any script creating an OpenFrames object (the
workshop template: 58 errors, rc 1), and a GroundTrack created before a
ReportFile left it writing no report. The organizer's GUI-saved templates
carry both, and certify stops at exact-run-01. When the plots are all GMAT
objects to, the error names them and sends the report to the GMAT GUI. When
GMAT's own error records (`**** ERROR ****`) say more, those are quoted
first, each once with the script line GMAT printed, and the GUI is not
offered: it shares that interpreter and would refuse the script too.
"""

from __future__ import annotations

import dataclasses
import signal
from pathlib import Path

import pytest
from conftest import kept_note, openframes_errors, quick_config, refuse_runs_of, stamped

import orbitwin.verify as verify_module
from orbitwin.errors import GmatRunError
from orbitwin.gmat_parser import GmatScriptDocument, parse_gmat_script, report_file_specs
from orbitwin.gmat_runner import (
    EXIT_STATUS_ERROR,
    FILE_SIZE_ERROR,
    NO_FRESH_REPORT_ERROR,
    TIMEOUT_ERROR,
    TIMEOUT_RETURN_CODE,
    gmat_errors,
    run_gmat,
)
from orbitwin.models import GmatRunResult
from orbitwin.search import zero_burn_candidate
from orbitwin.verify import (
    GROUND_TRACK_TYPES,
    OPENFRAMES_TYPES,
    PlotRefusal,
    baseline_original_script,
    certify_candidate,
    plot_refusal,
    refusal_message,
)

EXIT_1 = EXIT_STATUS_ERROR.format(1)
NO_REPORT = NO_FRESH_REPORT_ERROR.format("ReportFile1.txt")
TIMEOUT = TIMEOUT_ERROR.format(300)
SIGXFSZ = FILE_SIZE_ERROR.format(2147483648)
MAYBE = (
    '214: x.script: **** ERROR **** Interpreter Exception: The value of "maybe" for field '
    '"ShowPlot" on object "DefaultGroundTrackPlot" is not an allowed value. The allowed '
    "values are: [true false] in line:"
)
MAYBE_AT = '   " 220: DefaultGroundTrackPlot.ShowPlot = maybe;"'
WARNING = (
    '1: x.script: *** WARNING *** Interpreter Exception: GroundTrack object named '
    '"DefaultGroundTrackPlot" already exists, so ignored in line:'
)
OPENFRAMES = (
    "GmatConsole cannot create OpenFramesInterface DefaultOrbitView, OpenFramesView "
    "DefaultOrbitView_View (OpenFrames)"
)
GROUND = (
    "it creates GroundTrack DefaultGroundTrackPlot before that ReportFile, and on R2026a a "
    "GroundTrack created before its ReportFile left GmatConsole writing no report"
)
GUI = "Run it in the GMAT GUI and put the report it writes through check_report_rows " \
    "(docs/RUNBOOK.md step 6)"


def _doc(script: Path) -> GmatScriptDocument:
    return GmatScriptDocument.read(script)


def _result(
    tmp_path: Path, error: str | None, return_code: int = 0, said: list[str] | None = None,
    files: int = 1,
) -> GmatRunResult:
    """A run whose log and stdout (``files`` of them) hold ``said``, as GMAT writes both."""
    paths = [tmp_path / "stdout.txt", tmp_path / "GmatLog.txt"][:files]
    for path in paths:
        path.write_text("".join(f"{line}\n" for line in said or []), encoding="utf-8")
    return GmatRunResult(
        run_id="hand-made", success=error is None, return_code=return_code, started_at_utc="",
        duration_s=0.0, workdir=str(tmp_path), script_path="", script_sha256="",
        executable_sha256="", startup_sha256=None, stdout_path=str(paths[0]),
        stderr_path=str(tmp_path / "absent"), log_path=str(paths[-1]) if files > 1 else None,
        reports={}, report_sha256={}, error=error,
    )


def test_run_gmat_words_a_refusal_as_verify_reads_it(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """The two strings plot_refusal reads, as run_gmat writes them: a rewording breaks here."""
    failing = tmp_path / "fail.script"
    failing.write_text("% FAKE_GMAT_FAIL\n" + workshop_script.read_text(encoding="ascii"),
                       encoding="ascii")
    failed = run_gmat(fake_gmat, failing, tmp_path / "a", ["ReportFile1.txt"],
                      startup_file=fake_startup)
    assert (failed.return_code, failed.error) == (3, EXIT_STATUS_ERROR.format(3))
    quiet = run_gmat(fake_gmat, workshop_script, tmp_path / "b", ["absent.txt"],
                     startup_file=fake_startup)
    assert (quiet.return_code, quiet.error) == (0, NO_FRESH_REPORT_ERROR.format("absent.txt"))


def test_an_exit_status_names_the_openframes_objects(
    tmp_path: Path, workshop_script: Path
) -> None:
    result = _result(tmp_path, EXIT_1, 1, openframes_errors(workshop_script), files=2)
    assert plot_refusal(_doc(workshop_script), result) == PlotRefusal(OPENFRAMES, ())
    assert refusal_message(_doc(workshop_script), result, "failed").endswith(GUI)


def test_the_gui_note_restates_a_name_kept_as_written(
    tmp_path: Path, workshop_script: Path
) -> None:
    # The GUI path has no certification.json to say it.
    result = _result(tmp_path, EXIT_1, 1, openframes_errors(workshop_script), files=2)
    kept = refusal_message(_doc(workshop_script), result, "failed",
                           kept_as_written="ReportFile1.txt")
    assert kept.endswith(GUI + ". Its official report keeps the organizer's name "
                         "'ReportFile1.txt' (policy.report_filename_as_written): upload it "
                         "under that exact name")
    baseline = refusal_message(_doc(workshop_script), result, "failed", baseline=True,
                               kept_as_written="ReportFile1.txt")
    assert "organizer's name" not in baseline


def test_a_missing_report_names_the_ground_track_created_before_it(
    tmp_path: Path, workshop_script: Path
) -> None:
    assert plot_refusal(_doc(workshop_script), _result(tmp_path, NO_REPORT)) == PlotRefusal(GROUND, ())


@pytest.mark.parametrize(
    "edit, missing",
    [
        # The measured order: after its ReportFile, a GroundTrack left the report written.
        (("Create GroundTrack DefaultGroundTrackPlot;\n", ""), "ReportFile1.txt"),
        # A report the script does not write is not the one GMAT should have written.
        (None, "elsewhere.txt"),
    ],
    ids=["ground-track-after", "undeclared-report"],
)
def test_a_missing_report_is_otherwise_reported_plainly(
    tmp_path: Path, workshop_script: Path, edit: tuple[str, str] | None, missing: str
) -> None:
    script = workshop_script
    if edit is not None:
        text = workshop_script.read_text(encoding="ascii").replace(*edit)
        text = text.replace("BeginMissionSequence;",
                            "Create GroundTrack DefaultGroundTrackPlot;\nBeginMissionSequence;", 1)
        script = tmp_path / "after.script"
        script.write_text(text, encoding="ascii")
    error = NO_FRESH_REPORT_ERROR.format(missing)
    result = _result(tmp_path, error)
    assert plot_refusal(_doc(script), result) == PlotRefusal("", ())
    assert refusal_message(_doc(script), result, "failed") == error


#: Each input: (error, exit status, GMAT's lines, the reason); "openframes" stands
#: for R2026a's lines.
INPUTS = {
    "openframes": (EXIT_1, 1, ["openframes"], OPENFRAMES),
    "no-report": (NO_REPORT, 0, [], GROUND),
    "success": (None, 0, [], ""),
    "timeout": (TIMEOUT, TIMEOUT_RETURN_CODE, ["openframes"], ""),
    "sigxfsz": (SIGXFSZ, -signal.SIGXFSZ, ["openframes"], ""),
    "stale": ("GMAT report appears stale: x.txt", 0, [], ""),
    "bare-exit": (EXIT_1, 1, [], OPENFRAMES),
    "maybe": (EXIT_1, 1, ["openframes", MAYBE, MAYBE_AT], OPENFRAMES),
    "warning": (EXIT_1, 1, ["openframes", WARNING], OPENFRAMES),
    # Not GMAT's records: an echoed path, and a comment GMAT printed back.
    "path-noise": (EXIT_1, 1, ["openframes", "***** file: /runs/ERRORS/final.script",
                               "Interpreting % ERROR bars off"], OPENFRAMES),
    # Another build's words for the same refusal.
    "other-wording": (EXIT_1, 1, [
        '1: x.script: **** ERROR **** Interpreter Exception: Unknown type "OpenFramesInterface" '
        'for "DefaultOrbitView"'], OPENFRAMES),
    "report-and-error": (NO_REPORT, 0, ["**** ERROR **** Interpreter Exception: else in line:"],
                         GROUND),
}


def _input(tmp_path: Path, script: Path, name: str) -> GmatRunResult:
    error, code, said, _ = INPUTS[name]
    lines = [line for each in said
             for line in (openframes_errors(script) if each == "openframes" else [each])]
    return _result(tmp_path, error, code, lines, files=2)


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_the_reason_is_mains_and_the_gui_only_for_the_plots_alone(
    tmp_path: Path, workshop_script: Path, name: str
) -> None:
    """main named the plots on each of these; the GUI is offered only where nothing else
    GMAT said would stop the GUI too."""
    result = _input(tmp_path, workshop_script, name)
    reason, other, _ = plot_refusal(_doc(workshop_script), result)
    assert reason == INPUTS[name][3]
    message = refusal_message(_doc(workshop_script), result, "failed")
    assert (GUI in message) is bool(reason and not other)
    if other:
        assert "neither runs it as written" in message and "GMAT GUI, which shares" in message


@pytest.mark.parametrize(
    "name, quoted",
    [("openframes", 0), ("warning", 0), ("path-noise", 0), ("timeout", 0), ("sigxfsz", 0),
     ("maybe", 1), ("other-wording", 1), ("report-and-error", 1), ("bare-exit", 0)],
)
def test_only_gmats_own_other_records_are_quoted_once(
    tmp_path: Path, workshop_script: Path, name: str, quoted: int
) -> None:
    """GMAT writes each record to its log and its stdout; each is quoted once."""
    other = plot_refusal(_doc(workshop_script), _input(tmp_path, workshop_script, name)).other
    assert len(other) == quoted, other
    assert all(line.startswith("**** ERROR ****") for line in other)


def test_another_error_beside_the_plots_is_quoted_with_its_line_and_no_gui(
    tmp_path: Path, workshop_script: Path
) -> None:
    """R2026a: the workshop with ShowPlot = maybe still fails with OpenFrames gone."""
    result = _input(tmp_path, workshop_script, "maybe")
    for baseline, then in (
        (False, "stop and take this to the organizer (docs/RUNBOOK.md, Stop rules); if not, "
         "the rendered edits are at fault"),
        (True, "The official script may not be edited: stop and take this to the organizer "
         "(docs/RUNBOOK.md, Stop rules)"),
    ):
        message = refusal_message(_doc(workshop_script), result, "failed", baseline=baseline)
        assert message.startswith(
            f"{EXIT_1}. GMAT reported: **** ERROR **** Interpreter Exception: The value of "
            '"maybe" for field "ShowPlot" on object "DefaultGroundTrackPlot" is not an allowed '
            "value. The allowed values are: [true false] in line: (line 220: "
            "DefaultGroundTrackPlot.ShowPlot = maybe;). GMAT rejects"
        )
        assert f"; besides, {OPENFRAMES}." in message and message.endswith(then)
        assert "Run it in the GMAT GUI" not in message


def test_records_that_differ_only_by_their_prefix_are_quoted_once(
    tmp_path: Path, workshop_script: Path
) -> None:
    """Thirty copies of one error must not fill the three quoted places."""
    same = [f"{n}: /run/x.script: **** ERROR **** Interpreter Exception: Same thing in line:"
            for n in range(30)]
    lines = [*same, "**** ERROR **** Interpreter Exception: The real one in line:"]
    other = plot_refusal(_doc(workshop_script), _result(tmp_path, EXIT_1, 1, lines)).other
    assert other == ("**** ERROR **** Interpreter Exception: Same thing in line:",
                     "**** ERROR **** Interpreter Exception: The real one in line:")


@pytest.mark.parametrize("change", ["type", "name"])
def test_an_openframes_record_must_name_what_the_script_creates(
    tmp_path: Path, workshop_script: Path, change: str
) -> None:
    # Without the script line GMAT prints after each, only R2026a's words can say.
    lines = [
        line.replace('The "OpenFramesView"', 'The "OpenFramesVector"') if change == "type"
        else line.replace('object "DefaultOrbitView"', 'object "Elsewhere"')
        for line in openframes_errors(workshop_script) if not line.lstrip().startswith('"')
    ]
    reason, other, _ = plot_refusal(_doc(workshop_script), _result(tmp_path, EXIT_1, 1, lines))
    assert reason == OPENFRAMES and len(other) == 1


def test_a_missing_lhs_that_is_not_an_openframes_object_is_quoted(
    tmp_path: Path, workshop_script: Path
) -> None:
    """GMAT's LHS record for a craft is the script's own fault, not the plugin's."""
    craft = ('9: x.script: **** ERROR **** Interpreter Exception: Cannot find LHS object named '
             '"HumanSCC" in line:')
    lines = [*openframes_errors(workshop_script), craft]
    reason, other, _ = plot_refusal(_doc(workshop_script), _result(tmp_path, EXIT_1, 1, lines))
    assert reason == OPENFRAMES and other == (craft[craft.index("****"):],)


def test_reading_stops_at_the_cap_and_offers_no_gui(
    tmp_path: Path, workshop_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the cap the rest is unread, and might hold an error the GUI refuses too."""
    lines = [*openframes_errors(workshop_script), MAYBE]
    monkeypatch.setattr(verify_module, "_ERROR_LINES", 3)
    opened: list[str] = []
    real_open = open

    def counting(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    result = _result(tmp_path, EXIT_1, 1, lines, files=2)
    with monkeypatch.context() as patched:
        patched.setattr("builtins.open", counting)
        reason, other, complete = plot_refusal(_doc(workshop_script), result)
    # Stopped in the first file: the second was never opened.
    assert len(opened) == 1 and not complete and reason == OPENFRAMES and other == ()
    message = refusal_message(_doc(workshop_script), result, "failed")
    assert message.startswith(f"{EXIT_1}. GMAT wrote more than 3 errors and not all were "
                              "read. What GMAT said is incomplete")
    assert "the first follow" not in message and "Run it in the GMAT GUI" not in message
    assert message.endswith("(docs/RUNBOOK.md, Stop rules); if not, the rendered edits are "
                            "at fault")


def test_the_cap_counts_distinct_errors_not_lines(
    tmp_path: Path, workshop_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GMAT writes each record to its log and its stdout: that is one error, not two."""
    monkeypatch.setattr(verify_module, "_ERROR_LINES", 4)
    records = [f"{n}: x.script: **** ERROR **** Interpreter Exception: Error {n} in line:"
               for n in range(4)]
    result = _result(tmp_path, EXIT_1, 1, records, files=2)
    _, other, complete = plot_refusal(_doc(workshop_script), result)
    assert complete and len(other) == 4
    message = refusal_message(_doc(workshop_script), result, "failed")
    assert "not all were read" not in message
    records.append("9: x.script: **** ERROR **** Interpreter Exception: Error 9 in line:")
    result = _result(tmp_path, EXIT_1, 1, records, files=2)
    message = refusal_message(_doc(workshop_script), result, "failed")
    assert "not all were read; the first follow. GMAT reported: " in message


@pytest.mark.parametrize(
    "said, attributed",
    [
        # Another build's words, on an OpenFrames line: the plots'.
        (['1: x.script: **** ERROR **** Interpreter Exception: Unknown type in line:',
          '   " 166: Create OpenFramesInterface DefaultOrbitView;"'], True),
        (['2: x.script: **** ERROR **** Interpreter Exception: Bad thing in line:',
          '   " 175: DefaultOrbitView.DrawObject = [ true true true ];"'], True),
        # R2026a's words, but GMAT names a line that is not an OpenFrames object's.
        (['3: x.script: **** ERROR **** Interpreter Exception: Cannot find LHS object named '
          '"DefaultOrbitView" in line:', '   " 231: ReportFile1.Filename = \'x.txt\';"'], False),
        # A Create, but of a craft; and of that name, but not as the script creates it.
        (['4: x.script: **** ERROR **** Interpreter Exception: Unknown type in line:',
          '   " 30: Create Spacecraft HumanSC;"'], False),
        (['5: x.script: **** ERROR **** Interpreter Exception: Unknown type in line:',
          '   " 166: Create OrbitView DefaultOrbitView;"'], False),
    ],
    ids=["create-line", "field-line", "other-line", "craft-create", "other-type-create"],
)
def test_a_record_is_the_plots_by_the_line_gmat_names(
    tmp_path: Path, workshop_script: Path, said: list[str], attributed: bool
) -> None:
    reason, other, _ = plot_refusal(_doc(workshop_script), _result(tmp_path, EXIT_1, 1, said))
    assert reason == OPENFRAMES and (other == ()) is attributed


def test_the_double_gives_run_gmats_exit_status(tmp_path: Path, fake_startup: Path) -> None:
    """A real run_gmat timeout, and the statuses its other failures carry."""
    from conftest import run_gmat_return_code

    console = tmp_path / "sleepy"
    console.write_text("#!/bin/sh\nsleep 30\n", encoding="ascii")
    console.chmod(0o755)
    script = tmp_path / "idle.script"
    script.write_text("BeginMissionSequence;\n", encoding="ascii")
    ran = run_gmat(console, script, tmp_path / "run", ["r.txt"], startup_file=fake_startup,
                   timeout_s=0.2)
    assert ran.error == TIMEOUT_ERROR.format(0.2)
    assert run_gmat_return_code(ran.error) == ran.return_code == TIMEOUT_RETURN_CODE
    assert run_gmat_return_code(SIGXFSZ) == -signal.SIGXFSZ
    assert run_gmat_return_code(EXIT_STATUS_ERROR.format(3)) == 3


def test_an_error_without_the_marker_is_quoted(tmp_path: Path, workshop_script: Path) -> None:
    """R2026a prints some fatal errors without `**** ERROR ****`; they are GMAT's too."""
    bare = "Command Exception: Spacecraft epochs do not match in a Propagator"
    said = [*openframes_errors(workshop_script), bare]
    reason, other, _ = plot_refusal(_doc(workshop_script), _result(tmp_path, EXIT_1, 1, said))
    assert reason == OPENFRAMES and other == (bare,)


@pytest.mark.parametrize(("writes", "reason"), [("true", GROUND), ("false", "")])
def test_a_report_the_script_does_not_write_blames_no_ground_track(
    tmp_path: Path, workshop_script: Path, writes: str, reason: str
) -> None:
    """WriteReport = false: no GMAT writes that report, so its absence is no GroundTrack's."""
    script = tmp_path / "unwritten.script"
    script.write_text(workshop_script.read_text(encoding="ascii").replace(
        "ReportFile1.WriteReport = true;", f"ReportFile1.WriteReport = {writes};", 1),
        encoding="ascii")
    [spec] = report_file_specs(script)
    assert spec.write_report is (writes == "true")
    assert plot_refusal(_doc(script), _result(tmp_path, NO_REPORT)) == PlotRefusal(reason, ())


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        # GMAT keeps the first Create: an XYPlot before the report, no GroundTrack.
        (("Create GroundTrack DefaultGroundTrackPlot;",
          "Create XYPlot DefaultGroundTrackPlot;"), ""),
        (("Create ReportFile ReportFile1;",
          "Create ReportFile ReportFile1;\nCreate XYPlot DefaultGroundTrackPlot;"), GROUND),
    ],
    ids=["xyplot-first", "ground-track-first"],
)
def test_a_name_created_twice_is_what_its_first_create_made(
    tmp_path: Path, workshop_script: Path, edit: tuple[str, str], reason: str
) -> None:
    text = workshop_script.read_text(encoding="ascii")
    assert text.count(edit[0]) == 1
    text = text.replace(*edit)
    if not reason:
        text = text.replace("Create ReportFile ReportFile1;",
                            "Create ReportFile ReportFile1;\nCreate GroundTrack "
                            "DefaultGroundTrackPlot;")
    script = tmp_path / "twice.script"
    script.write_text(text, encoding="ascii")
    assert plot_refusal(_doc(script), _result(tmp_path, NO_REPORT)) == PlotRefusal(reason, ())


def test_no_note_for_a_script_without_them(
    tmp_path: Path, plain_workshop_script: Path, commented_workshop_script: Path
) -> None:
    for script in (plain_workshop_script, commented_workshop_script):
        assert plot_refusal(_doc(script), _result(tmp_path, EXIT_1, 1)).reason == ""
        assert plot_refusal(_doc(script), _result(tmp_path, NO_REPORT)).reason == ""


def test_the_private_copies_leave_out_every_type_named() -> None:
    from orbitwin.diagnostic import GUI_SUBSCRIBER_TYPES

    assert OPENFRAMES_TYPES | GROUND_TRACK_TYPES <= GUI_SUBSCRIBER_TYPES


def test_certify_says_why_the_exact_run_failed(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = stamped(zero_burn_candidate(scenario, config), workshop_script, config)
    refuse_runs_of(monkeypatch, "final.script")
    with pytest.raises(GmatRunError) as raised:
        certify_candidate(
            workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "certify",
            runs=1,
        )
    # The script to take to the GUI is still there, where the refusal says.
    assert str(raised.value) == (
        f"{EXIT_1}. final.script: {OPENFRAMES}. {GUI}. {kept_note(tmp_path / 'certify')}"
    )
    kept = tmp_path / "certify.refused"
    assert f"({kept / 'staging' / 'final.script'}, " in str(raised.value)
    assert (kept / "staging" / "final.script").is_file()


def test_the_plots_stop_repeats_the_archive_warnings_after_the_gui_advice(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RUNBOOK's GUI phrase is in the whole message certify prints, warnings and all."""
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = stamped(zero_burn_candidate(scenario, config), workshop_script, config)
    del candidate.metadata["searched_score"]
    refuse_runs_of(monkeypatch, "final.script")
    with pytest.warns(RuntimeWarning), pytest.raises(GmatRunError) as raised:
        certify_candidate(
            workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "certify",
            runs=1,
        )
    assert str(raised.value) == (
        f"{EXIT_1}. final.script: {OPENFRAMES}. {GUI}. {kept_note(tmp_path / 'certify')} "
        f"Archive warnings, again: Candidate {candidate.candidate_id} carries no "
        "searched_score, so nothing shows its search aimed at these coefficients; rank says "
        "the same."
    )
    assert "Run it in the GMAT GUI" in str(raised.value)


def test_a_later_exact_run_refused_says_why_too(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = stamped(zero_burn_candidate(scenario, config), workshop_script, config)
    refuse_runs_of(monkeypatch, "exact-run-02")
    with pytest.raises(GmatRunError, match=rf"^{EXIT_1}\. final\.script: GmatConsole cannot"):
        certify_candidate(workshop_script, candidate, config, fake_gmat, fake_startup,
                          tmp_path / "certify", runs=3)


def test_baseline_says_why_it_failed(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refuse_runs_of(monkeypatch, workshop_script.name)
    with pytest.raises(GmatRunError) as raised:
        baseline_original_script(
            workshop_script, quick_config(), fake_gmat, fake_startup, tmp_path / "baseline"
        )
    assert "OpenFramesInterface DefaultOrbitView" in str(raised.value)
    assert "Solve still runs on its private copy" in str(raised.value)
    assert "step" not in str(raised.value)


@pytest.mark.parametrize("template", ["plain_workshop_script", "commented_workshop_script"])
def test_certify_adds_nothing_to_a_failure_without_plots(
    tmp_path: Path, template: str, request: pytest.FixtureRequest, fake_gmat: Path,
    fake_startup: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = request.getfixturevalue(template)
    config = quick_config()
    scenario, _ = parse_gmat_script(script, config)
    candidate = stamped(zero_burn_candidate(scenario, config), script, config)
    refuse_runs_of(monkeypatch, "final.script", error=EXIT_1)
    with pytest.raises(GmatRunError) as raised:
        certify_candidate(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c",
                          runs=1)
    assert str(raised.value) == f"{EXIT_1}. {kept_note(tmp_path / 'c')}"


def test_a_bare_exit_status_still_names_the_openframes_objects(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exit status with no records: main's advice, as run_gmat would have said it."""
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = stamped(zero_burn_candidate(scenario, config), workshop_script, config)
    refuse_runs_of(monkeypatch, "final.script", error=EXIT_1)
    with pytest.raises(GmatRunError) as raised:
        certify_candidate(workshop_script, candidate, config, fake_gmat, fake_startup,
                          tmp_path / "c", runs=1)
    assert str(raised.value) == (
        f"{EXIT_1}. final.script: {OPENFRAMES}. {GUI}. {kept_note(tmp_path / 'c')}"
    )


def test_a_relative_output_is_named_by_its_full_paths(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator takes staging/final.script from this message to the GUI (RUNBOOK step 6)."""
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = stamped(zero_burn_candidate(scenario, config), workshop_script, config)
    refuse_runs_of(monkeypatch, "final.script", error=EXIT_1)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    with pytest.raises(GmatRunError) as raised:
        certify_candidate(workshop_script, candidate, config, fake_gmat, fake_startup,
                          Path("run/final"), runs=1)
    assert (tmp_path / "run" / "final.refused" / "staging" / "final.script").is_file()
    # kept_note names them from the absolute tmp_path.
    assert str(raised.value).endswith(f". {kept_note(tmp_path / 'run' / 'final')}")


# R2026a's own logs (output/reportformat/review3-runs/cw0, output/stopdepth/runs/R4_exact/
# cssi_missing): a message that runs on, and one message on six lines.
COLUMN_WIDTH = [
    '1: /runs/x/final.script: **** ERROR **** Subscriber Exception: The value of "0" for '
    'field "ColumnWidth" on object "ReportFile1" is not an allowed value.',
    "The allowed values are: [Integer Number > 0 ] in line:",
    '   " 127: ReportFile1.ColumnWidth = 0;"',
    "",
]
DRAG = [
    line
    for number in range(82, 88)
    for line in (
        f"{number - 81}: /runs/x/R4.script: **** ERROR **** Interpreter Exception: "
        '/runs/x/R4.script: The scripted force type "Drag" is not a known force or force model '
        "setting",
        " in line:",
        f'   "  {number}: DefaultProp_ForceModel.Drag.F{number} = 150;"',
        "",
    )
]


def test_a_message_keeps_its_lines_and_its_location(tmp_path: Path) -> None:
    """GMAT puts ` in line:` and the location after the whole message."""
    result = _result(tmp_path, EXIT_1, 1, COLUMN_WIDTH, files=2)
    result = dataclasses.replace(result, workdir="/runs/x")
    [error, *_] = gmat_errors(result).errors
    assert str(error) == (
        '**** ERROR **** Subscriber Exception: The value of "0" for field "ColumnWidth" on '
        'object "ReportFile1" is not an allowed value. The allowed values are: [Integer Number '
        "> 0 ] in line: (line 127: ReportFile1.ColumnWidth = 0;)"
    )


def test_one_message_on_six_lines_is_six_errors(tmp_path: Path) -> None:
    result = dataclasses.replace(_result(tmp_path, EXIT_1, 1, DRAG, files=2), workdir="/runs/x")
    errors = gmat_errors(result).errors
    assert [error.line for error in errors] == list(range(82, 88))
    # The run's directory is ours, not GMAT's: it is not quoted.
    assert all("/runs/x" not in error.text and "R4.script: The" in error.text for error in errors)


def test_an_unmarked_gmat_error_is_kept_and_an_echoed_word_is_not(tmp_path: Path) -> None:
    said = ["ERROR loading kernel /gmat/data/planetary_ephem/spk/de999.bsp",
            "Interpreting % ERROR bars off", "***** file: /runs/ERRORS/final.script"]
    errors = gmat_errors(_result(tmp_path, EXIT_1, 1, said)).errors
    assert [(error.kind, error.text) for error in errors] == [
        ("loose", "ERROR loading kernel /gmat/data/planetary_ephem/spk/de999.bsp")]


@pytest.mark.parametrize(
    ("said", "ours"),
    [
        (['1: x.script: **** ERROR **** ODEModel Exception Thrown: The file name '
          '"../tides/Earth.tide" does not exist in line:',
          '   " 80: DefaultProp_ForceModel.GravityField.Earth.TideFile = \'../tides/Earth.tide\';"'],
         True),
        (['1: x.script: **** ERROR **** SolarSystem exception: Cannot open the observed space '
          "weather file does-not-exist.txt in line:",
          '   " 81: DefaultProp_ForceModel.Drag.AtmosphereModel = NRLMSISE00;"'], True),
        (["ERROR loading the main Solar System ephemeris (SPK) or leap second (LSK) kernel."],
         True),
        ([MAYBE, MAYBE_AT], False),
        # A script error that merely says "not found".
        (['1: x.script: **** ERROR **** Interpreter Exception: Object type not found in the '
          "definition block in line:", '   " 30: Create HumanSC;"'], False),
    ],
    ids=["tide-file", "space-weather-file", "kernel", "script-value", "not-found"],
)
def test_a_file_gmat_could_not_open_may_be_ours(
    tmp_path: Path, plain_workshop_script: Path, said: list[str], ours: bool
) -> None:
    """A data file our install or run folder lacks is our fault, not the official script's."""
    result = _result(tmp_path, EXIT_1, 1, said)
    message = refusal_message(_doc(plain_workshop_script), result, "failed", baseline=True)
    assert ("the fault is ours" in message) is ours
    assert "neither runs it as written" in message and "Stop rules" in message


def test_more_than_three_errors_are_counted_not_quoted(tmp_path: Path) -> None:
    result = dataclasses.replace(_result(tmp_path, EXIT_1, 1, DRAG, files=2), workdir="/runs/x")
    message = refusal_message(_doc(Path(__file__).parents[2] / "examples" / "synthetic"
                                   / "NoManeuver_Circular.script"), result, "failed")
    assert message.count("(line 8") == 3 and "(and 3 more)" in message


def test_a_warning_ends_the_message_before_it(tmp_path: Path) -> None:
    """The warning's location is the warning's, not the error's before it."""
    said = ["2: /x/s.script: **** ERROR **** Initialization Exception: something failed at init",
            '1: /x/s.script: *** WARNING *** Interpreter Exception: ImpulsiveBurn object named '
            '"B" already exists, so ignored in line:',
            '   " 151: Create ImpulsiveBurn B;"']
    [error] = gmat_errors(_result(tmp_path, EXIT_1, 1, said)).errors
    assert (error.text, error.line) == (
        "**** ERROR **** Initialization Exception: something failed at init", 0)


def test_the_runbook_tells_the_gui_case_by_the_phrases_certify_prints(
    tmp_path: Path, workshop_script: Path
) -> None:
    """RUNBOOK steps 4a, 6 and 6c quote these; each must still be in the message it names."""
    runbook = " ".join((Path(__file__).resolve().parents[2] / "docs" / "RUNBOOK.md")
                       .read_text(encoding="utf-8").split())
    gui = "Run it in the GMAT GUI"
    besides = {OPENFRAMES: "; besides, GmatConsole cannot create",
               GROUND: "; besides, it creates GroundTrack"}
    assert runbook.count(f'"{gui}"') == 3
    assert all(f'"...{phrase} ..."' in runbook for phrase in besides.values())
    seen = set()
    for name in sorted(INPUTS):
        (tmp_path / name).mkdir()
        result = _input(tmp_path / name, workshop_script, name)
        reason, other, complete = plot_refusal(_doc(workshop_script), result)
        for kept in (None, "ODC_2026_XX.txt"):
            message = refusal_message(_doc(workshop_script), result, "failed",
                                      kept_as_written=kept)
            assert (gui in message) is bool(reason and not other and complete), name
            if reason and other:
                assert besides[reason] in message, (name, message)
                seen.add(reason)
    assert seen == set(besides)
