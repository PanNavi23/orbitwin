from __future__ import annotations

import argparse
import atexit
import contextlib
import dataclasses
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    OrbitWinConfig,
    add_rules_argument,
    load_config,
    rulebook_refusal,
    validate_config,
)
from .day import EXIT_CERTIFY_NO_ANSWER, add_day_parser, run_day
from .design import seconds_after_target, set_pursuer_epoch, verify_design
from .errors import ConfigurationError, OrbitWinError, error_code
from .gmat_parser import GmatScriptDocument, report_file_specs, stop_depth_advice
from .gmat_runner import GUI_STOP, is_test_double, machine_error, no_answer
from .installed import FRESH_PYCACHE, compiled_afresh, installed
from .installed import describe as describe_installed
from .io import read_candidates
from .layout import CHECKOUT
from .mission_clock import clock_lines, horizon_override_record
from .models import STAGE_SCRIPT_NAME, CandidateStage, ReportFileSpec, Scenario, StateProbe
from .official_report import official_report_warnings
from .pinned import assert_pinned_environment
from .pipeline import SOLVED_WITHOUT_GMAT, closest_miss_line, solve_scenario, zero_burn_line
from .rank import DEFAULT_TOLERANCE, NO_GMAT_ARCHIVE, NO_GMAT_HUNT, head_start_line, rank_report
from .report_format import position_frames
from .rules import (
    DuelRules,
    RulePackError,
    interception_rules,
    rule_set,
    rule_set_document,
    rule_sets,
)
from .script_renderer import (
    assert_final_script_policy,
    assert_team_number_for_submission,
    official_report_files,
    official_report_of,
    report_name_verdict,
    team_number_warnings,
)
from .spad import spad_check
from .state_probe import load_scenario
from .time_origin import time_origin_evidence
from .verify import baseline_original_script, certify_candidate
from .workspace import (
    WORKSPACE_COMMANDS,
    add_workspace_parsers,
    init_command,
    run_command,
    status_command,
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is not at least 1")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbitwin",
        description="Deterministic 3-D GMAT orbital-interception toolkit",
    )
    parser.add_argument("--version", action=_Version)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Parse and inspect an organizer GMAT script")
    inspect.add_argument("script")
    inspect.add_argument("--config")
    add_rules_argument(inspect)
    inspect.add_argument("--output")
    _add_optional_gmat_pair(inspect)

    solve = sub.add_parser("solve", help="Run analytical search and optionally GMAT verification")
    solve.add_argument("script")
    solve.add_argument("--config")
    add_rules_argument(solve)
    solve.add_argument("--output", required=True)
    solve.add_argument("--render-top", type=int)
    solve.add_argument("--gmat-console")
    solve.add_argument("--startup-file")
    solve.add_argument("--gmat-verify-top", type=int)
    solve.add_argument(
        "--workers",
        type=int,
        help="Worker processes for the analytical sweep; 0 or less means CPU count - 1",
    )
    solve.add_argument(
        "--gmat-workers", type=int,
        help="Concurrent GmatConsole runs while correcting and verifying",
    )
    solve.add_argument(
        "--seed-count",
        type=int,
        help="Independent Sobol draws to repeat the sweep with and union; 1 is one sweep",
    )

    baseline = sub.add_parser("baseline", help="Execute the exact original script once")
    baseline.add_argument("script")
    baseline.add_argument("--config")
    add_rules_argument(baseline)
    baseline.add_argument("--gmat-console", required=True)
    baseline.add_argument("--startup-file", required=True)
    baseline.add_argument("--output", required=True)

    policy = sub.add_parser("verify-policy", help="Assert only approved final-script fields changed")
    policy.add_argument("original")
    policy.add_argument("generated")
    policy.add_argument("--config")
    add_rules_argument(policy)
    _add_optional_gmat_pair(policy)

    rank = sub.add_parser(
        "rank",
        help="Re-rank a saved archive under the announced (kt, Ct, kv, Cv[, C_r]); it warns "
        "when the archive was searched under others",
    )
    rank.add_argument("script")
    rank.add_argument("--config")
    add_rules_argument(rank)
    rank.add_argument("--candidates", required=True)
    rank.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    rank.add_argument("--top", type=int, default=10)
    rank.add_argument("--output")
    _add_optional_gmat_pair(rank)

    certify = sub.add_parser("certify", help="GMAT-verify and reproduce the exact final submission")
    certify.add_argument("script")
    certify.add_argument("--config")
    add_rules_argument(certify)
    certify.add_argument("--candidates", required=True)
    certify.add_argument("--candidate", required=True)
    certify.add_argument("--gmat-console", required=True)
    certify.add_argument("--startup-file", required=True)
    certify.add_argument("--runs", type=_positive_int, help="exact runs, at least 1")
    certify.add_argument("--output", required=True)
    certify.add_argument(
        "--accept-gmat-loss",
        action="store_true",
        help="certify a candidate as GMAT flies it when that scores below the archive's "
        "claim: a planned hit flown as a miss, a later entry or a wider miss",
    )
    epoch = sub.add_parser(
        "set-epoch", help="Write the official script with the pursuer's epoch set, where the "
        "rules let a team set it (docs/DESIGN_VARIABLES.md)")
    epoch.add_argument("script", help="the official script, as the organizer gave it")
    epoch.add_argument("--config")
    add_rules_argument(epoch)
    when = epoch.add_mutually_exclusive_group(required=True)
    when.add_argument("--after-target-s", type=float,
                      help="the pursuer's epoch, in seconds after the target's")
    when.add_argument("--epoch", help="the pursuer's epoch, UTCGregorian "
                      "('DD Mon YYYY HH:MM:SS.mmm')")
    epoch.add_argument("--json", action="store_true", help="print JSON")
    rules = sub.add_parser("rules", help="List the rule sets, or show one")
    rules.add_argument("name", nargs="?", help="a rule set's id or name; all of them when left out")
    rules.add_argument("--json", action="store_true", help="print JSON")
    add_day_parser(sub)
    add_workspace_parsers(sub)
    return parser


def _rules_command(name: str | None, as_json: bool) -> int:
    """Print every rule set, or one in full."""
    try:
        chosen = rule_sets() if name is None else [rule_set(name)]
    except (KeyError, RulePackError) as exc:
        print(f"orbitwin: ERROR [rules]: {exc.args[0]}", file=sys.stderr)
        return 2
    if name is not None:
        print(json.dumps(rule_set_document(chosen[0]), ensure_ascii=False, indent=2))
        return 0
    if as_json:
        print(json.dumps([rule_set_document(rules) for rules in chosen], ensure_ascii=False,
                         indent=2))
        return 0
    for rules in chosen:
        kind = "duel" if isinstance(rules, DuelRules) else "interception"
        history = ", historical" if getattr(rules, "historical", False) else ""
        print(f"{rules.id:32} {kind}{history}: {rules.source.document} ({rules.source.header_date})")
    return 0


def _add_optional_gmat_pair(command: argparse.ArgumentParser) -> None:
    help_text = "with --startup-file: read the initial states through the GMAT state probe"
    command.add_argument("--gmat-console", help=help_text)
    command.add_argument("--startup-file")


def _parse_with_optional_probe(
    script: str, config: OrbitWinConfig, gmat_console: str | None, startup_file: str | None
) -> tuple[Scenario, StateProbe | None]:
    """For commands with no output directory: the probe's files are not kept."""
    _require_gmat_pair(gmat_console, startup_file)
    scenario, _, _ = load_scenario(
        script, config, gmat_console=gmat_console, startup_file=startup_file
    )
    return scenario, scenario.state_probe


def _require_gmat_pair(gmat_console: str | None, startup_file: str | None) -> None:
    """A GmatConsole without its startup file is not a usable pair.

    The startup file is what gives each run its own OUTPUT_PATH and what the
    organizer tier is fingerprinted against, so half the pair is never what the
    caller meant. Refusing here says so at the command line rather than three
    steps into a run.
    """
    if (gmat_console is None) != (startup_file is None):
        raise OrbitWinError("--gmat-console and --startup-file must be supplied together")


#: scripts/check_report_rows.py beside the package, as a checkout has it.
_CHECK_REPORT_ROWS = (Path("/nonexistent") if CHECKOUT is None
                      else CHECKOUT / "scripts" / "check_report_rows.py")
#: The same check as a module, which an installed package carries.
_CHECK_MODULE = "orbitwin.tools.check_report_rows"


_PRECHECK = "before the GUI, the binding step 7 holds their report to, on ours, must exit "


def _checker() -> list[str]:
    """Step 7's command: the checkout's script, as the RUNBOOK names it, else the module."""
    if _CHECK_REPORT_ROWS.is_file():
        return [sys.executable, str(_CHECK_REPORT_ROWS)]
    return [sys.executable, "-m", _CHECK_MODULE]


def _frozen(record: dict[str, object]) -> list[str]:
    """--frozen-rows from a certificate or verification, when the run had an alignment."""
    rows = record.get("frozen_rows")
    return ["--frozen-rows", str(rows)] if isinstance(rows, int) and rows > 0 else []


class PrecheckUnavailable(Exception):
    """Why step 6's pre-check cannot be built for a run."""


def _file_name(filename: str) -> str:
    return filename.strip().strip("'\"").replace("\\", "/").rsplit("/", 1)[-1]


def _official_spec(script: Path) -> ReportFileSpec:
    """The script's official ReportFile: the report uploaded and checked."""
    try:
        official = official_report_of(report_file_specs(script))
    except (OrbitWinError, OSError) as exc:
        raise PrecheckUnavailable(f"{script} does not say its official report: {exc}") from exc
    if official is None:
        raise PrecheckUnavailable(f"{script} has no ReportFile GMAT writes")
    return official


def _official_name(script: Path) -> str:
    """The file the script's official ReportFile writes."""
    return _file_name(_official_spec(script).filename)


#: check_report_rows's flag for a report that cannot show a burn, and certify's last line
#: when the pre-check passed so: exit 3, the end and a row at each burn epoch only.
_END_ONLY = "--accept-end-time-binding-only"
PRECHECK_PASS_END_TIME = ("PRE-CHECK PASS (end time only): the 6b.7 by-eye comparison is "
                          "mandatory")


def _end_time_only(script: Path, config: str) -> list[str]:
    """[_END_ONLY] when the official report has no EarthMJ2000Eq position and velocity of
    the pursuer, which check_report_rows needs to see a burn (its _pursuer_state_columns)."""
    try:
        pursuer = load_config(config).pursuer_name or "SC_Human"
    except (OrbitWinError, OSError):
        # check_report_rows says what is wrong with the config.
        return []
    added = set(_official_spec(script).parameters)
    shown = all({f"{pursuer}.EarthMJ2000Eq.{axis}", f"{pursuer}.{axis}"} & added
                for axis in ("X", "Y", "Z", "VX", "VY", "VZ"))
    return [] if shown else [_END_ONLY]


#: check_report_rows's exit for a report with no target position, judged by hand: the
#: pre-check takes it only beside a --certificate run that passed.
_TARGET_NOT_IN_REPORT = 7
PRECHECK_PASS_NO_TARGET = (
    "PRE-CHECK PASS (target not in report): the official report carries no target position; "
    "d_min and the interception were proven on certify's diagnostic flight of the same script")
PRECHECK_PASS_NO_TARGET_END_TIME = (
    f"{PRECHECK_PASS_NO_TARGET}; its burns were bound by the end time only: the 6b.7 by-eye "
    "comparison is mandatory")


def _carries_no_target(script: Path, config: str) -> bool:
    """Whether the script's official report has no X, Y and Z of the target."""
    try:
        target = load_config(config).target_name or "SC_Alien"
        return not position_frames(_official_spec(script).parameters, target)
    except (OrbitWinError, OSError, PrecheckUnavailable):
        return False


def gui_precheck(kept: Path, config: str) -> list[str]:
    """Step 6's pre-check for a certify stopped on plots, on the run kept in ``kept``.

    Our own report of the flight verification.json names, the one it scored
    (stopped at the hit, the whole flight for a miss, or flown on to T_max when
    a planned hit missed), against the staging script the GUI runs.
    """
    staging = kept / "staging" / "final.script"
    verification = kept / "diagnostic" / "verification.json"
    for path in (staging, verification):
        if not path.is_file():
            raise PrecheckUnavailable(f"the run kept in {kept} has no {path.name}")
    try:
        record = json.loads(verification.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PrecheckUnavailable(f"{verification} is unreadable: {exc}") from exc
    flown = record.get("diagnostic_report") if isinstance(record, dict) else None
    if not isinstance(flown, str):
        raise PrecheckUnavailable(f"{verification} names no diagnostic_report")
    # Written before certify set the run aside (verify._set_aside), resolved:
    # <run>/diagnostic/<flight>/gmat-output/<file>, <run> beside kept under its old name.
    recorded = Path(flown)
    run = kept.parent.resolve() / re.sub(r"\.refused(-[0-9]+)?$", "", kept.name)
    if (not recorded.is_absolute() or len(recorded.parents) < 4
            or Path(os.path.realpath(recorded.parents[3])) != run
            or recorded.parents[2].name != "diagnostic"
            or recorded.parents[0].name != "gmat-output"
            or not recorded.parents[1].name.startswith("diagnostic-run")):
        raise PrecheckUnavailable(
            f"{verification} names {flown!r}, not a flight of {run}, the run kept in {kept}")
    report = (kept / "diagnostic" / recorded.parents[1].name / "gmat-output"
              / _official_name(staging))
    if not report.is_file():
        raise PrecheckUnavailable(f"the flight verification.json names wrote no {report}")
    return [*_checker(), str(report),
            "--config", str(Path(config).resolve()), "--script", str(staging),
            "--verification", str(verification), *_frozen(record),
            *_end_time_only(staging, config)]


def precheck_commands(
    result: dict[str, object], output: Path, config: str
) -> list[list[str]]:
    """Step 6's pre-check for a certified run: our official report bound to our script by
    hand, as a GUI's would be, and then step 7 itself, which certify must pass."""
    submission = Path(os.path.abspath(output)) / "submission"
    script = submission / str(result["submission_script_name"])
    verification = str(submission.parent / "diagnostic" / "verification.json")
    report = str((submission / _official_name(script)).resolve())
    return [[*_checker(), report,
             "--config", str(Path(config).resolve()), "--script", str(script),
             "--verification", verification, *_frozen(result),
             *_end_time_only(script, config)],
            step7_command(submission, report, config)]


def step7_command(submission: Path, report: str, config: str) -> list[str]:
    """docs/RUNBOOK.md step 7, on the report certify packaged."""
    return [*_checker(), report, "--config", str(Path(config).resolve()),
            "--certificate", str((submission / "certification.json").resolve())]


#: check_report_rows's exit status for §4 note 1(b) undecided: (record, line).
_SURFACE_RESULTS = {4: ("surface-not-judged", "NOT JUDGED"),
                    5: ("surface-template-below", "TEMPLATE BELOW"),
                    6: ("surface-undecided", "UNDECIDED")}


def _precheck_line(command: list[str], no_target: bool = False) -> str:
    """The printed pre-check: exit 3 is its pass for a report that cannot show a burn, and 7
    for one with no target position, beside step 7's pass."""
    expected = ("7 (target not in report; step 7 below must exit 0)" if no_target
                else "3 (end time only; the 6b.7 by-eye check is mandatory)" if _END_ONLY in command
                else "0")
    return f"{_PRECHECK}{expected}: {shlex.join(command)}"


#: The last line of certify's output when step 6's pre-check passed.
PRECHECK_PASS = "PRE-CHECK PASS"


def _first_refusal(done: subprocess.CompletedProcess[str]) -> str:
    """The line that says why check_report_rows did not pass."""
    errors = [line for line in done.stderr.splitlines() if "error:" in line]
    verdicts = [line for line in done.stdout.splitlines() if line.startswith(("VERDICT", "WARNING"))]
    rest = [line for line in done.stderr.splitlines() if line.strip()]
    return (errors or verdicts or rest or [f"exit status {done.returncode}"])[0]


def binding_precheck(
    commands: list[list[str]], console: str | None, config: str | None,
    unavailable: str | None = None,
) -> tuple[str, dict[str, object]]:
    """Run step 6's pre-check: the line certify ends on, and the record it keeps.

    It never changes certify's exit status (docs/RUNBOOK.md step 6).
    """
    if console is not None and is_test_double(console):
        return ("orbitwin certify: PRE-CHECK SKIPPED: the console is orbitwin's test double, "
                "whose reports are not GMAT's", {"result": "skipped", "why": "test double"})
    why = ("no --config was given" if config is None
           else unavailable or "no report to check" if not commands
           else None)
    if why is not None:
        return (f"WARNING: PRE-CHECK NOT RUN: {why}; run step 6's pre-check by hand "
                "(docs/RUNBOOK.md step 6)", {"result": "not run", "why": why})
    runs: list[dict[str, object]] = []
    for command in commands:
        try:
            done = subprocess.run(command, capture_output=True, text=True, errors="replace",
                                  check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            why = f"{command[2] if command[1] == '-m' else command[1]} did not run: {exc}"
            return (f"WARNING: PRE-CHECK NOT RUN: {why}; run step 6's pre-check by hand "
                    "(docs/RUNBOOK.md step 6)", {"result": "not run", "why": why})
        lines = [line for line in done.stdout.splitlines() if line.strip()]
        runs.append({"command": command, "exit_code": done.returncode,
                     "first_refusal": None if done.returncode == 0 else _first_refusal(done),
                     "last_line": lines[-1] if lines else ""})
    # Exit 3 is check_report_rows's pass that bound the end alone; only asked for so.
    end_only = [run for run, command in zip(runs, commands, strict=True)
                if _END_ONLY in command and run["exit_code"] == 3]
    # check_report_rows's §4 note 1(b) outcomes that are neither a pass nor a refusal.
    surface = [run for run in runs if run["exit_code"] in _SURFACE_RESULTS]
    # Exit 7 by hand: no target position. Only step 7's pass on the same report makes it one.
    certified = [run for run, command in zip(runs, commands, strict=True)
                 if "--certificate" in command]
    step7 = bool(certified) and all(run["exit_code"] == 0 or run in surface for run in certified)
    no_target = [(run, command) for run, command in zip(runs, commands, strict=True)
                 if step7 and "--certificate" not in command
                 and run["exit_code"] == _TARGET_NOT_IN_REPORT]
    failed = next((run for run in runs if run["exit_code"] != 0 and run not in end_only
                   and run not in surface and run not in [hand for hand, _ in no_target]), None)
    if failed is None and surface:
        result, what = _SURFACE_RESULTS[int(str(surface[0]["exit_code"]))]
        return (f"WARNING: PRE-CHECK SURFACE {what}: {surface[0]['last_line']}",
                {"result": result, "runs": runs})
    if failed is None and no_target:
        if any(_END_ONLY in command for _, command in no_target):
            return (PRECHECK_PASS_NO_TARGET_END_TIME,
                    {"result": "pass-target-not-in-report-end-time-only", "runs": runs})
        return PRECHECK_PASS_NO_TARGET, {"result": "pass-target-not-in-report", "runs": runs}
    if failed is None and end_only:
        return PRECHECK_PASS_END_TIME, {"result": "pass-end-time-only", "runs": runs}
    if failed is None:
        return PRECHECK_PASS, {"result": "pass", "runs": runs}
    return (f"WARNING: PRE-CHECK FAILED: {failed['first_refusal']}",
            {"result": "failed", "runs": runs})


def _record(path: Path, precheck: dict[str, object], key: str = "binding_pre_check") -> None:
    """Add the pre-check (or ``key``) to certification.json or verification.json, as they
    are written."""
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(body, dict):
        body[key] = precheck
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8")


def _listed(path: Path) -> str:
    return f"  {path}" + ("" if path.is_file() else "   (MISSING)")


def upload_summary(
    result: dict[str, object], output: Path, config: str | None = None,
    rules: str | None = None,
) -> list[str]:
    """The step 7 check, then what is uploaded.

    The upload is the submission's script and its official report, from this
    machine, on every path (thc1006, 2026-11-06; docs/RUNBOOK.md step 6); another
    ReportFile's output is neither uploaded nor checked. The submission folder
    also holds official_original.script, the organizer's own input kept for the
    record. Ours is named for the evidence tier this run conferred. ``rules`` is
    the --rules certify ran under, which the printed commands need too (#190).
    """
    # Absolute, so each path works from whatever directory it is read in.
    submission = Path(os.path.abspath(output)) / "submission"
    reports = result["official_report_files"]
    assert isinstance(reports, list)
    script = submission / str(result["submission_script_name"])
    files = [submission / str(name) for name in reports] + [script]
    # Only the official report is uploaded and checked; the others are GMAT's output.
    official: Path | None = None
    unknown = ""
    try:
        official = submission / _official_name(script)
    except PrecheckUnavailable as why:
        unknown = str(why)
    lines: list[str] = []
    # Repeated here, beside the upload, so a non-stock SPAD file is not read past.
    notes = result.get("spad_warnings")
    if isinstance(notes, list):
        lines += [f"orbitwin certify: {note}" for note in notes]
    if config is None:
        lines.append(
            "orbitwin certify: no --config was given, so this ran the default rules; certify "
            "again with the day's config before step 7 (docs/RUNBOOK.md)."
        )
    else:
        # Absolute, so the command runs from whatever directory it is pasted into.
        certificate = str((submission / "certification.json").resolve())
        if official is None:
            lines.append(f"orbitwin certify: no step 7 command: {unknown}")
        else:
            command = [*_checker(), str(official.resolve()),
                       "--config", str(Path(config).resolve()), "--certificate", certificate,
                       *_rules_flag(rules)]
            lines.append(f"orbitwin certify: step 7, must exit 0: {shlex.join(command)}")
            # The binding step 7 also holds a GUI's report to; a false refusal shows here first.
            hand = precheck_commands(result, output, config)[0]
            no_target = _carries_no_target(script, config)
            lines.append(f"orbitwin certify: "
                         f"{_precheck_line([*hand, *_rules_flag(rules)], no_target)}")
            if no_target:
                lines.append(
                    f"orbitwin certify: {official.name} carries no target position: by hand, or "
                    "from a GMAT GUI, it cannot be judged (exit 7); step 7 above passes it on "
                    "certify's diagnostic flight of the same script, with a WARNING saying so. "
                    "Ask the organizer how their validator computes d_min (docs/RUNBOOK.md step 7)")
    kept = result.get("report_filename_as_written")
    if kept is not None:
        lines.append(
            f"orbitwin certify: the official report keeps the organizer's name {kept!r} "
            "(policy.report_filename_as_written, their written answer); upload it under "
            "that exact name."
        )
    copy = result.get("console_copy")
    if official is None:
        lines.append(
            f"orbitwin certify: no upload can be named: {unknown}. Find the official report "
            "in the script before step 7 (docs/RUNBOOK.md step 6)."
        )
    elif isinstance(copy, dict):
        removed = ", ".join(map(str, copy.get("removed") or []))
        lines.append(
            f"orbitwin certify: GmatConsole refused {script.name} for its plots alone, so every "
            f"exact run ran {Path(str(copy.get('script'))).name}, its copy proven to differ only "
            f"by leaving out {removed} (gmat.console_copy). The report below is that copy's; the "
            f"diagnostic sibling, run off {script.name}, wrote it byte for byte. Once step 7 "
            "exits 0 on it, the upload is these two files from this machine, as they are "
            "(docs/RUNBOOK.md step 6c):"
        )
        lines += [_listed(official), _listed(script)]
    else:
        lines.append(
            "orbitwin certify: once step 7 exits 0 on the report, the upload is these files from "
            "this machine, as they are (docs/RUNBOOK.md step 6):"
        )
        lines += [_listed(script), _listed(official)]
    others = [path.name for path in files[:-1] if official and path.name != official.name]
    if others:
        lines.append(
            "orbitwin certify: not uploaded or checked, another ReportFile's output: "
            + ", ".join(others)
        )
    if isinstance(copy, dict):
        lines.append(
            f"orbitwin certify: {script.name} keeps every plot line of the official script; "
            "never upload the console copy, official_original.script, certification.json, the "
            ".diff.txt or anything else from run/."
        )
    else:
        lines.append(
            "orbitwin certify: never upload official_original.script, certification.json, the "
            ".diff.txt or anything else from run/; the first is the organizer's own input."
        )
    return lines + _stage_note(script, result)


def _stage_note(script: Path, result: dict[str, object]) -> list[str]:
    """Why the submission script is not team_solution.script, when it is not."""
    certified = STAGE_SCRIPT_NAME[CandidateStage.CERTIFIED]
    if script.name == certified:
        return []
    why = []
    if result["conferred_stage"] != CandidateStage.ORGANIZER_GMAT_VERIFIED.value:
        why.append(f"{certified} is written only for a GMAT the organizer has confirmed as "
                   "theirs (docs/RUNBOOK.md step 6)")
    if isinstance(result.get("console_copy"), dict):
        why.append(f"{certified} is written only when GmatConsole ran the submitted bytes "
                   "themselves, and here it ran their console copy")
    return [
        f"orbitwin certify: {script.name} is named for what this run established, "
        f"{result['conferred_stage']}; do not rename it. " + "; ".join(why) + "."
    ]


def baseline_notice(result: dict[str, object], script: Path) -> list[str]:
    """What a baseline that ran a console copy ran, and what its report is."""
    copy = result.get("console_copy")
    if not isinstance(copy, dict):
        return []
    return [
        f"orbitwin baseline: GmatConsole refused {script.name} for its plots alone, so this "
        f"ran {Path(str(copy.get('script'))).name}, its copy proven to differ only by leaving "
        f"out {', '.join(map(str, copy.get('removed') or []))} (gmat.console_copy); "
        "baseline.json records it as console_copy and the run as that copy's."
    ]


#: Written into the archive by scripts/build_release.py, beside src/ as the
#: editable install has it. An unpacked tree has no other record of its commit.
BUILD_INFO = Path(__file__).resolve().parents[2] / "BUILD_INFO.json"


def describe_build(path: Path | None = None) -> str:
    """The version, and the commit the archive was built from."""
    path = BUILD_INFO if path is None else path
    if not path.is_file():
        package = installed() if CHECKOUT is None else None
        if package is not None:
            return describe_installed(package)
        return f"orbitwin {__version__}, no {path.name}: not an unpacked release archive"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        commit, committed = str(info["commit"]), str(info["committed_utc"])
        version, dirty = str(info["version"]), info["dirty"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return f"orbitwin {__version__}, {path} is unreadable ({type(error).__name__}: {error})"
    line = f"orbitwin {__version__}, commit {commit}, committed {committed}"
    if version != __version__:
        line += f", but {path.name} names version {version}"
    if dirty is not False:
        line += ", built with uncommitted changes"
    provenance = info.get("provenance")
    if isinstance(provenance, dict) and provenance.get("allow_non_main") is True and (
        provenance.get("matches") is None
    ):
        line += (f", NOT origin/main: built with --allow-non-main on "
                 f"{provenance.get('branch')}")
    return line


class _Version(argparse.Action):
    """--version, with the commit from BUILD_INFO.json, read only when asked."""

    def __init__(self, option_strings: list[str], dest: str = argparse.SUPPRESS) -> None:
        super().__init__(option_strings, dest, default=argparse.SUPPRESS, nargs=0,
                         help="show the version and the commit this archive was built from")

    def __call__(
        self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
        values: object, option_string: str | None = None,
    ) -> None:
        print(describe_build())
        parser.exit()


def _rules_flag(rules: str | None) -> list[str]:
    """--rules for check_report_rows, which needs it as certify did."""
    return [] if rules is None else ["--rules", rules]


def compile_afresh(given: list[str]) -> None:
    """An installed package's day reads no bytecode it did not compile itself: run it again
    with an empty bytecode cache. The code that does this is the one part no check holds."""
    if CHECKOUT is not None or not given or given[0] not in ("day", "run"):
        return
    if os.environ.get(FRESH_PYCACHE):
        # The run itself: it holds its cache while it lives, and removes it when it ends,
        # if it is the one made for it.
        if compiled_afresh() and Path(sys.pycache_prefix or "").name.startswith(_FRESH_NAME):
            _HELD_CACHE.append(_hold(Path(str(sys.pycache_prefix))))
            atexit.register(shutil.rmtree, str(sys.pycache_prefix), True)
        return
    _sweep_stale_caches()
    try:
        fresh = tempfile.mkdtemp(prefix=_FRESH_NAME)
    except OSError as exc:
        message = f"cannot make an empty bytecode cache for this run: {exc}"
        if given[0] == "run" and "--json" in given:
            print(json.dumps({"error": {"code": "io", "message": message}}, indent=2))
        else:
            print(f"orbitwin: ERROR [io]: {message}", file=sys.stderr)
        raise SystemExit(2) from None
    env = {**os.environ, "PYTHONPYCACHEPREFIX": fresh, FRESH_PYCACHE: fresh}
    sys.stdout.flush()
    sys.stderr.flush()
    # -P: the directory it is started in is not put first on sys.path.
    os.execve(sys.executable, [sys.executable, "-P", "-m", "orbitwin", *given], env)


#: The name the empty bytecode cache of a re-run starts with.
_FRESH_NAME = "orbitwin-pyc-"
#: A cache older than this, that no run holds, was left by a run that was killed.
_STALE_S = 86400.0
#: The lock this run holds on its cache while it lives; the sweep passes a held one by.
_CACHE_LOCK = ".held"
_HELD_CACHE: list[Any] = []


def _hold(cache: Path) -> Any:
    import fcntl

    handle = (cache / _CACHE_LOCK).open("a")
    fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
    return handle


def _held(cache: Path) -> bool:
    """Whether a live run holds ``cache``; creates nothing."""
    import fcntl

    try:
        handle = (cache / _CACHE_LOCK).open("r")
    except OSError:
        return False
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
    return False


def _sweep_stale_caches() -> None:
    """Remove this user's caches left by runs that were killed; a running day loses only
    bytecode it compiles again."""
    import time

    base = Path(tempfile.gettempdir())
    now = time.time()
    for path in base.glob(f"{_FRESH_NAME}*"):
        try:
            info = path.lstat()
        except OSError:
            continue
        if (not path.is_symlink() and path.is_dir() and info.st_uid == os.getuid()
                and now - info.st_mtime > _STALE_S and not _held(path)):
            shutil.rmtree(path, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    given = sys.argv[1:] if argv is None else argv
    compile_afresh(given)
    if "--json" in given and given and given[0] in WORKSPACE_COMMANDS:
        # A usage error is a refusal too: JSON on stdout, as the command's own.
        said = io.StringIO()
        try:
            with contextlib.redirect_stderr(said):
                args = _parser().parse_args(given)
        except SystemExit as exc:
            if exc.code in (0, None):
                raise
            lines = said.getvalue().strip().splitlines()
            print(json.dumps({"error": {"code": "usage",
                                        "message": lines[-1] if lines else "usage error"}},
                             ensure_ascii=False, indent=2))
            return 2
    else:
        args = _parser().parse_args(argv)
    if args.command == "rules":
        return _rules_command(args.name, args.json)
    if args.command == "init":
        return init_command(args)
    if args.command == "status":
        return status_command(args)
    if args.command in ("day", "run"):
        # Its steps are the other commands, each with its own config.
        try:
            assert_pinned_environment()
        except OrbitWinError as exc:
            if args.command == "run" and args.json:
                print(json.dumps({"error": {"code": "environment", "message": str(exc)}},
                                 ensure_ascii=False, indent=2))
            else:
                print(f"orbitwin: ERROR [environment]: {exc}", file=sys.stderr)
            return 2
        return run_day(args) if args.command == "day" else run_command(args, run_day)
    try:
        assert_pinned_environment()
        config = load_config(args.config)
        refusal = rulebook_refusal(config, args.config, args.rules)
        if refusal is not None:
            raise ConfigurationError(refusal)
        rules = interception_rules(config.policy.rules)
        if rules.historical:
            print(
                f"orbitwin: policy.rules is {rules.name!r}, the historical rulebook of "
                f"{rules.source.header_date}. The ranking round is config/ranking.yaml; "
                "the duel has no mission config.",
                file=sys.stderr,
            )
        # A copy set-epoch wrote is held to its record before anything reads it.
        template = getattr(args, "script", None) or getattr(args, "original", None)
        design = (verify_design(template, rules, config.pursuer_name, config.target_name)
                  if template is not None and args.command != "set-epoch" else None)
        if args.command == "set-epoch":
            if config.pursuer_name is None or config.target_name is None:
                raise ConfigurationError("set-epoch needs pursuer_name and target_name in the "
                                         "config")
            after = (args.after_target_s if args.epoch is None
                     else seconds_after_target(args.script, config.target_name, args.epoch))
            copy, record = set_pursuer_epoch(
                args.script, rules, config.pursuer_name, config.target_name,
                after_target_s=after, horizon_s=config.policy.horizon_s)
            if args.json:
                print(json.dumps({"script": str(copy), **record}, ensure_ascii=False, indent=2))
            else:
                print(f"orbitwin set-epoch: {copy}: {record['pursuer']}.Epoch "
                      f"'{record['pursuer_epoch']}', {after:g} s after {record['target']}'s "
                      f"({record['clause']}); run every later step on it", file=sys.stderr)
                print(copy)
            return 0
        if args.command == "inspect":
            _require_gmat_pair(args.gmat_console, args.startup_file)
            scenario, document, _ = load_scenario(
                args.script, config, gmat_console=args.gmat_console,
                startup_file=args.startup_file,
            )
            # Advice only: the team sets policy.stop_depth_km.
            advice = stop_depth_advice(scenario, config.policy.stop_depth_km)
            print(f"orbitwin: {advice['advice']}", file=sys.stderr)
            spad, spad_notes = spad_check(
                scenario, GmatScriptDocument.read(args.script), args.gmat_console,
                args.startup_file,
            )
            for note in spad_notes:
                print(f"orbitwin inspect: {note}", file=sys.stderr)
            # Only when the craft start apart; it flies SC_Alien once more in GMAT.
            origin = time_origin_evidence(
                scenario, document, config, args.gmat_console, args.startup_file
            )
            # Before any search: certify would refuse these names at the end.
            refusal, name_notes = report_name_verdict(scenario, config)
            text = json.dumps(
                {
                    **scenario.to_dict(), "stop_depth_advice": advice, "spad_files": spad,
                    "time_origin_evidence": origin.to_dict() if origin is not None else None,
                    "announced_horizon_before_perigee": horizon_override_record(scenario),
                    "report_names": {
                        "submitted": [
                            spec.filename for spec in official_report_files(scenario, config)
                        ] if refusal is None else None,
                        "refusal": refusal,
                        "notes": list(name_notes),
                    },
                },
                ensure_ascii=False, indent=2, sort_keys=True,
            )
            # stderr, so `> scenario.json` keeps working.
            for warning in official_report_warnings(scenario, rules):
                print(f"orbitwin inspect: {warning}", file=sys.stderr)
            for line in clock_lines(scenario, config.policy):
                print(f"orbitwin inspect: {line}", file=sys.stderr)
            for line in origin.lines if origin is not None else ():
                print(f"orbitwin inspect: {line}", file=sys.stderr)
            for warning in team_number_warnings(scenario, config):
                print(f"orbitwin inspect: WARNING: {warning}", file=sys.stderr)
            if args.output:
                # A fresh output directory is the normal competition-day case,
                # and inspect is the first command run into it.
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text, encoding="utf-8")
            print(text)
        elif args.command == "solve":
            _require_gmat_pair(args.gmat_console, args.startup_file)
            # Worker counts are a runtime resource decision, so the flags
            # override the profile rather than the other way round.
            if args.workers is not None:
                config = dataclasses.replace(
                    config, search=dataclasses.replace(config.search, workers=args.workers)
                )
            if args.seed_count is not None:
                config = dataclasses.replace(
                    config, search=dataclasses.replace(config.search, seed_count=args.seed_count)
                )
            if args.gmat_workers is not None:
                config = dataclasses.replace(
                    config, gmat=dataclasses.replace(config.gmat, workers=args.gmat_workers)
                )
            # Every override above lands after load_config already validated, so
            # without this a flag can put the config into a state the profile
            # could never have expressed. Measured: --seed-count -5 exited 0 and
            # ran a single sweep, so an operator who asked for repeated draws
            # got one and was told it worked.
            validate_config(config)
            result = solve_scenario(
                args.script,
                config,
                args.output,
                render_top=args.render_top,
                gmat_console=args.gmat_console,
                startup_file=args.startup_file,
                gmat_verify_top=args.gmat_verify_top,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            listed = result.get("warnings")
            if isinstance(listed, list) and SOLVED_WITHOUT_GMAT in listed:
                print(f"orbitwin solve: {SOLVED_WITHOUT_GMAT}", file=sys.stderr)
            failures = result.get("gmat_failures")
            if isinstance(failures, dict) and failures["warning"] is not None:
                print(f"orbitwin solve: WARNING: {failures['warning']}", file=sys.stderr)
            evidence = result.get("time_origin_evidence")
            for line in evidence["lines"] if isinstance(evidence, dict) else ():
                print(f"orbitwin solve: {line}", file=sys.stderr)
            # Only a term with C_r prices the zero burn's distance worth reading back.
            takes_cr = interception_rules(config.policy.rules).distance_term.takes_cr
            zero_burn = zero_burn_line(result) if takes_cr else None
            if zero_burn is not None:
                print(f"orbitwin solve: {zero_burn}", file=sys.stderr)
            closest = closest_miss_line(result)
            if closest is not None:
                print(f"orbitwin solve: {closest}", file=sys.stderr)
        elif args.command == "baseline":
            result = baseline_original_script(
                args.script,
                config,
                args.gmat_console,
                args.startup_file,
                args.output,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            if result["epoch_alignment_note"] is not None:
                print(f"orbitwin baseline: {result['epoch_alignment_note']}", file=sys.stderr)
            for line in baseline_notice(result, Path(args.script)):
                print(line, file=sys.stderr)
        elif args.command == "verify-policy":
            original, probe = _parse_with_optional_probe(
                args.original, config, args.gmat_console, args.startup_file
            )
            assert_team_number_for_submission(original, config)
            assert_final_script_policy(args.original, args.generated, config, probe=probe)
            print("policy: PASS")
        elif args.command == "rank":
            scenario, _ = _parse_with_optional_probe(
                args.script, config, args.gmat_console, args.startup_file
            )
            report = rank_report(
                scenario,
                read_candidates(args.candidates),
                config,
                tolerance=args.tolerance,
                top=args.top,
            )
            # allow_nan=False: json.dumps happily writes bare NaN and Infinity,
            # which no strict parser accepts, so a report carrying one is a file
            # that says JSON and is not.
            text = json.dumps(
                report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            if args.output:
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text, encoding="utf-8")
            print(text)
            listed = report["warnings"]
            for message in (NO_GMAT_ARCHIVE, NO_GMAT_HUNT):
                if isinstance(listed, list) and message in listed:
                    print(f"orbitwin rank: {message}", file=sys.stderr)
            searched = report["searched_score"]
            if isinstance(searched, dict) and searched["warning"] is not None:
                print(f"orbitwin rank: WARNING: {searched['warning']}", file=sys.stderr)
            failures = report["gmat_failures"]
            if isinstance(failures, dict) and failures["warning"] is not None:
                print(f"orbitwin rank: WARNING: {failures['warning']}", file=sys.stderr)
            if isinstance(report["head_start"], dict) and report["head_start"]["enabled"]:
                print(f"orbitwin rank: {head_start_line(report['head_start'])}", file=sys.stderr)
        elif args.command == "certify":
            candidates = read_candidates(args.candidates)
            candidate = next(
                (item for item in candidates if item.candidate_id == args.candidate), None
            )
            if candidate is None:
                raise OrbitWinError(f"Candidate {args.candidate} was not found")
            result = certify_candidate(
                args.script,
                candidate,
                config,
                args.gmat_console,
                args.startup_file,
                args.output,
                runs=args.runs,
                accept_gmat_loss=args.accept_gmat_loss,
            )
            if design is not None:
                _record(Path(os.path.abspath(args.output)) / "submission" / "certification.json",
                        design, key="design")
            # Flushed now: piped into one stream, the pre-check line must still come last.
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
            for line in upload_summary(result, Path(args.output), args.config, args.rules):
                print(line, file=sys.stderr)
            commands: list[list[str]] = []
            unavailable = None
            try:
                if args.config is not None:
                    commands = [[*command, *_rules_flag(args.rules)] for command in
                                precheck_commands(result, Path(args.output), args.config)]
            except PrecheckUnavailable as why:
                unavailable = str(why)
            last, precheck = binding_precheck(commands, args.gmat_console, args.config,
                                              unavailable)
            _record(Path(os.path.abspath(args.output)) / "submission" / "certification.json",
                    precheck)
            print(last, file=sys.stderr)
        else:
            raise AssertionError(args.command)
    except (OrbitWinError, ValueError, OSError) as exc:
        print(f"orbitwin: ERROR [{error_code(exc)}]: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"orbitwin: {note}", file=sys.stderr)
        kept = getattr(exc, "kept_dir", None)
        unanswered = args.command == "certify" and no_answer(exc)
        if args.command == "certify" and kept is not None and GUI_STOP in str(exc) and (
                not unanswered):
            command, unavailable = None, None
            try:
                if args.config is not None:
                    command = [*gui_precheck(Path(kept), args.config), *_rules_flag(args.rules)]
            except PrecheckUnavailable as why:
                unavailable = str(why)
            if command is not None:
                print(f"orbitwin certify: {_precheck_line(command)}", file=sys.stderr)
                staging = Path(kept) / "staging" / "final.script"
                if args.config is not None and _carries_no_target(staging, args.config):
                    print("orbitwin certify: WARNING: the official report carries no target "
                          "position and no certificate was written, so nothing can pass a GMAT "
                          "GUI's report of it (docs/RUNBOOK.md step 7)", file=sys.stderr)
            last, precheck = binding_precheck([command] if command else [], args.gmat_console,
                                              args.config, unavailable)
            _record(Path(kept) / "diagnostic" / "verification.json", precheck)
            if design is not None:
                _record(Path(kept) / "diagnostic" / "verification.json", design, key="design")
            print(last, file=sys.stderr)
        if unanswered:
            machine = machine_error(exc)
            cause = ("GmatConsole crashed, timed out or wrote no report" if machine is None
                     else f"this machine failed ({type(machine).__name__}: "
                     f"{machine.strerror or machine})")
            print(f"orbitwin certify: NO ANSWER: {cause}, which is no verdict on the "
                  f"candidate; run certify again (exit {EXIT_CERTIFY_NO_ANSWER})",
                  file=sys.stderr)
            return EXIT_CERTIFY_NO_ANSWER
        return 2
    return 0


# `python -m orbitwin.cli` is the module path the traceback and the import line
# both name, so it is the one a reader reaches for. Without this it imports the
# module, runs nothing and exits 0 -- a silent success, which is the one outcome
# this project treats as worse than a crash.
if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
