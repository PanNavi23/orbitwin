"""certify tells a GmatConsole run that gave no answer from GMAT's verdict.

R4 (2026-11-07): a SIGSEGV and a hang in certify's state probe both exited 2, the
refusal's status, and `orbitwin day` read them as verdicts on the candidate.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import FAKE_STARTUP

from orbitwin import cli
from orbitwin.day import EXIT_CERTIFY_NO_ANSWER
from orbitwin.errors import (
    AltitudeFloorError,
    GmatRunError,
    OrbitWinError,
    PolicyError,
    StopVerdictError,
)
from orbitwin.gmat_runner import (
    EXIT_STATUS_ERROR,
    FILE_SIZE_ERROR,
    NO_FRESH_REPORT_ERROR,
    STALE_REPORT_ERROR,
    STOPPING_ERROR,
    TIMEOUT_ERROR,
    gave_no_answer,
    run_gmat,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("message", [
    "The state probe failed: " + EXIT_STATUS_ERROR.format(-11),
    EXIT_STATUS_ERROR.format(-9),
    # A wrapping shell reports the signal as 128 + N.
    EXIT_STATUS_ERROR.format(139),
    "The state probe failed: " + TIMEOUT_ERROR.format(300.0),
    NO_FRESH_REPORT_ERROR.format("ODC_2026_Team54.txt"),
    STALE_REPORT_ERROR.format("ODC_2026_Team54.txt"),
    STOPPING_ERROR,
])
def test_a_crash_a_timeout_and_no_report_are_no_answer(message: str) -> None:
    assert gave_no_answer(message)


@pytest.mark.parametrize("said", [
    # said_by_gmat's lines, verify's _reported and refusal_message (#199 review P2b).
    "\n  **** ERROR **** Interpreter Exception: Undefined object",
    ". GMAT reported: Command Exception: the propagator diverged",
    ". GMAT rejects final.script for these, and the GMAT GUI would too",
])
@pytest.mark.parametrize("failure", [
    EXIT_STATUS_ERROR.format(-6),
    TIMEOUT_ERROR.format(300.0),
    NO_FRESH_REPORT_ERROR.format("ODC_2026_Team54.txt"),
])
def test_a_failure_gmat_gave_reasons_for_is_an_answer(failure: str, said: str) -> None:
    assert gave_no_answer(failure) and not gave_no_answer(failure + said)


@pytest.mark.parametrize("message", [
    # GMAT's own refusal: the plots, a script error.
    EXIT_STATUS_ERROR.format(1),
    EXIT_STATUS_ERROR.format(2),
    # The file-size cap stops a Propagate goal never reached, every time.
    FILE_SIZE_ERROR.format(1 << 30),
    EXIT_STATUS_ERROR.format(-25),
    EXIT_STATUS_ERROR.format(153),
    "GMAT output contains fatal marker: Exception",
    "Candidate did not enter the 5 km sphere in GMAT",
    "The exact final script does not enter the intercept sphere on replay",
])
def test_gmats_answers_are_answers(message: str) -> None:
    assert not gave_no_answer(message)


def test_only_a_run_error_that_is_no_verdict_is_no_answer() -> None:
    crash = "The state probe failed: " + EXIT_STATUS_ERROR.format(-11)
    assert cli.no_answer(GmatRunError(crash))
    # This machine, not GMAT: a full disk under certify (R4 H, on a real one).
    assert cli.no_answer(OSError(28, "No space left on device"))
    assert cli.no_answer(PermissionError(13, "Permission denied"))
    # A file named wrong is the operator's, and fails the same way again.
    assert not cli.no_answer(FileNotFoundError(2, "No such file or directory"))
    assert not cli.no_answer(OSError("no errno"))
    # Behind what certify raised, as verify wraps a run that raised (#199 review P2a).
    try:
        try:
            raise OSError(28, "No space left on device")
        except OSError as exc:
            raise GmatRunError(f"running it raised OSError: {exc}. {cli.GUI_STOP}") from exc
    except GmatRunError as wrapped:
        assert cli.no_answer(wrapped)
        verdict = StopVerdictError("a verdict")
        verdict.__cause__ = wrapped
        assert not cli.no_answer(verdict)
    for exc in (StopVerdictError(crash), AltitudeFloorError(crash), PolicyError(crash),
                OrbitWinError(crash), ValueError(crash), GmatRunError("exit status 1"),
                GmatRunError(f"{crash}. {cli.GUI_STOP}")):
        assert not cli.no_answer(exc), exc


def _certify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raised: Exception) -> int:
    def certify_candidate(*_: object, **__: object) -> dict[str, object]:
        raise raised

    monkeypatch.setattr(cli, "read_candidates",
                        lambda _: [SimpleNamespace(candidate_id="C-best")])
    monkeypatch.setattr(cli, "certify_candidate", certify_candidate)
    return cli.main(["certify", str(ROOT / "examples" / "ranking" /
                                     "R1_NearPlane_Reachable.script"),
                     "--config", str(ROOT / "config" / "ranking.yaml"),
                     "--candidates", str(tmp_path / "candidates.json"),
                     "--candidate", "C-best", "--gmat-console", "GmatConsole",
                     "--startup-file", "startup.txt", "--output", str(tmp_path / "final")])


def test_certify_exits_75_on_no_answer_and_2_on_a_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    crash = GmatRunError("The state probe failed: " + EXIT_STATUS_ERROR.format(-11))
    assert _certify(monkeypatch, tmp_path, crash) == EXIT_CERTIFY_NO_ANSWER == 75
    err = capsys.readouterr().err
    assert "orbitwin: ERROR [gmat-run]: The state probe failed: GmatConsole returned exit status -11" in err
    assert err.rstrip().endswith(
        "orbitwin certify: NO ANSWER: GmatConsole crashed, timed out or wrote no report, which "
        "is no verdict on the candidate; run certify again (exit 75)")
    hang = GmatRunError("The state probe failed: " + TIMEOUT_ERROR.format(300.0))
    assert _certify(monkeypatch, tmp_path, hang) == 75
    capsys.readouterr()
    full = OSError(28, "No space left on device")
    assert _certify(monkeypatch, tmp_path, full) == 75
    assert ("NO ANSWER: this machine failed (OSError: No space left on device), which is no "
            "verdict on the candidate" in capsys.readouterr().err)
    # An operator's typo will fail the same way again: not "run certify again".
    typo = FileNotFoundError(2, "No such file or directory", "/nonexistent/candidates.json")
    assert _certify(monkeypatch, tmp_path, typo) == 2
    assert "NO ANSWER" not in capsys.readouterr().err
    verdict = GmatRunError("Candidate did not enter the 5 km sphere in GMAT")
    assert _certify(monkeypatch, tmp_path, verdict) == 2
    assert "NO ANSWER" not in capsys.readouterr().err


@pytest.mark.parametrize(("body", "timeout_s"), [
    # R4's fakegmat/segv: dies of SIGSEGV before GMAT starts.
    ("import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n", 30.0),
    # R4's fakegmat/hang: sleeps past any timeout.
    ("import time\ntime.sleep(600)\n", 1.0),
])
def test_what_run_gmat_says_of_a_crash_and_a_hang_is_no_answer(
    tmp_path: Path, body: str, timeout_s: float,
) -> None:
    import sys

    console = tmp_path / "bin" / "GmatConsole"
    console.parent.mkdir()
    console.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    console.chmod(0o755)
    script = tmp_path / "s.script"
    script.write_text("% nothing\n", encoding="ascii")
    ran = run_gmat(console, script, tmp_path / "run", ["r.txt"], timeout_s=timeout_s,
                   startup_file=FAKE_STARTUP)
    assert ran.error is not None and gave_no_answer(ran.error), ran.error


def test_no_answer_behind_the_plots_stop_skips_the_gui_pre_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """#199 review P2a: a console copy this machine could not write carries the plots stop's
    words; certify must not send its report to the GUI."""
    import errno

    def gui_precheck(*_: object) -> list[str]:
        raise AssertionError("the GUI path's pre-check ran")

    monkeypatch.setattr(cli, "gui_precheck", gui_precheck)
    try:
        raise OSError(errno.ENOSPC, "No space left on device")
    except OSError as exc:
        stop = GmatRunError(f"GmatConsole cannot create OpenFrames. {cli.GUI_STOP}")
        stop.__cause__ = exc
    stop.__dict__["kept_dir"] = tmp_path / "final.no-answer"
    assert _certify(monkeypatch, tmp_path, stop) == 75
    assert "NO ANSWER: this machine failed (OSError: No space left on device)" in (
        capsys.readouterr().err)


@pytest.mark.parametrize(("said", "answered"), [("", False), ("**** ERROR ****", True)])
def test_a_diagnostic_run_carries_gmats_words(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, said: str, answered: bool,
) -> None:
    """A SIGABRT GMAT logged an error before is GMAT's answer; one it did not is none."""
    import dataclasses

    from conftest import quick_config, stamped

    import orbitwin.verify as verify_module
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.gmat_runner import no_answer
    from orbitwin.search import zero_burn_candidate

    real = verify_module.run_gmat

    def run(console, script, workdir, reports, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = real(console, script, workdir, reports, *args, **kwargs)
        if not Path(workdir).name.startswith("diagnostic-run"):
            return result
        if said:
            with open(result.stdout_path, "a", encoding="utf-8") as stdout:
                stdout.write(f"1: {script}: {said} Interpreter Exception: Undefined object\n")
        return dataclasses.replace(result, success=False, reports={}, report_sha256={},
                                   partial_reports={}, error=EXIT_STATUS_ERROR.format(-6),
                                   return_code=-6)

    monkeypatch.setattr(verify_module, "run_gmat", run)
    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = stamped(zero_burn_candidate(scenario, config), workshop_script, config)
    with pytest.raises(GmatRunError) as raised:
        verify_module.certify_candidate(workshop_script, candidate, config, fake_gmat,
                                        fake_startup, tmp_path / "final", runs=1)
    assert no_answer(raised.value) is not answered, str(raised.value)
