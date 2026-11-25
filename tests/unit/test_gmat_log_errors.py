"""An error GMAT logs in a run that exits 0 is recorded, warned and, in certify, refused.

R3 (2026-11-06): a GmatConsole that appended an Interpreter Exception to its log
after a run exiting 0 was certified, and certify ended on PRE-CHECK PASS.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from conftest import FAKE_STARTUP

from orbitwin import gmat_runner
from orbitwin.config import OrbitWinConfig
from orbitwin.errors import GmatRunError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.gmat_runner import log_error_records, run_gmat, unexplained_log_errors

#: Two records as R2026a writes them, the second ending the file without a blank line.
LOG = (
    "Kernel ../data/time/SPICELeapSecondKernel.tls has been loaded.\n"
    '1: /x/run/final.script: **** ERROR **** Interpreter Exception: Cannot find LHS object '
    'named "Plot1" in line:\n'
    '   " 119: Plot1.SolverIterations = Current;"\n'
    "\n"
    "Mission run completed.\n"
    "2: /x/run/final.script: **** ERROR **** Interpreter Exception: injected\n"
)


def test_each_record_is_read_whole_without_the_run_path() -> None:
    assert log_error_records(LOG) == [
        '**** ERROR **** Interpreter Exception: Cannot find LHS object named "Plot1" in line: '
        '" 119: Plot1.SolverIterations = Current;"',
        "**** ERROR **** Interpreter Exception: injected",
    ]
    assert log_error_records("Mission run completed.\n*** Error loading \"libMatlab\"\n") == []


def test_no_record_is_benign_until_a_run_shows_it_so(monkeypatch: pytest.MonkeyPatch) -> None:
    records = log_error_records(LOG)
    assert gmat_runner.BENIGN_LOG_ERRORS == ()
    assert unexplained_log_errors(records) == records
    monkeypatch.setattr(gmat_runner, "BENIGN_LOG_ERRORS", (re.compile("Plot1"),))
    assert unexplained_log_errors(records) == records[1:]


def _logging(workshop_script: Path, tmp_path: Path) -> Path:
    script = tmp_path / "logs_an_error.script"
    script.write_text(
        "% FAKE_GMAT_LOG_ERROR Interpreter Exception: injected after the run\n"
        + workshop_script.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return script


def test_a_run_that_logs_an_error_records_and_warns_it(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _logging(workshop_script, tmp_path)
    scenario, _ = parse_gmat_script(script, OrbitWinConfig())
    with pytest.warns(RuntimeWarning, match=r"WARNING: GmatConsole exited 0 and .* records 1 "):
        result = run_gmat(fake_gmat, script, tmp_path / "run",
                          [report.filename for report in scenario.report_files],
                          startup_file=fake_startup, timeout_s=10.0)
    assert result.success
    assert result.log_errors == [
        "**** ERROR **** Interpreter Exception: injected after the run \" 1: the line\""]
    assert result.to_dict()["log_errors"] == result.log_errors
    clean = run_gmat(fake_gmat, workshop_script, tmp_path / "clean",
                     [report.filename for report in scenario.report_files],
                     startup_file=fake_startup, timeout_s=10.0)
    assert clean.success and clean.log_errors == []


def _candidate(script: Path):  # type: ignore[no-untyped-def]
    from conftest import quick_config

    from orbitwin.search import search_one_burn
    from orbitwin.state_probe import load_scenario

    config = quick_config()
    root = Path(__file__).resolve().parents[2]
    scenario, _, _ = load_scenario(
        script, config, gmat_console=root / "scripts" / "fake_gmat_console.py",
        startup_file=FAKE_STARTUP,
    )
    return config, search_one_burn(scenario, config)[0]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_certify_refuses_an_error_logged_in_a_run_that_exited_0(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orbitwin.verify import certify_candidate

    script = _logging(workshop_script, tmp_path)
    config, candidate = _candidate(script)
    with pytest.raises(GmatRunError, match=r"logged 1 error.* none of which is known harmless"):
        certify_candidate(script, candidate, config, fake_gmat, fake_startup,
                          tmp_path / "refused", runs=1)
    # verify kept its flights' records.
    (kept,) = tmp_path.glob("refused.refused*")
    recorded = json.loads((kept / "diagnostic" / "verification.json").read_text("utf-8"))
    assert recorded["gmat_log_errors"] and all(
        records == ["**** ERROR **** Interpreter Exception: injected after the run "
                    "\" 1: the line\""]
        for records in recorded["gmat_log_errors"].values())
    assert recorded["gmat_run"]["log_errors"]
    # Named benign, it is certified and recorded.
    monkeypatch.setattr(gmat_runner, "BENIGN_LOG_ERRORS", (re.compile("injected after the run"),))
    _, candidate = _candidate(script)
    certificate = certify_candidate(script, candidate, config, fake_gmat, fake_startup,
                                    tmp_path / "benign", runs=1)
    assert certificate["gmat_log_errors"]
    assert all(run["log_errors"] for run in certificate["exact_runs"])
