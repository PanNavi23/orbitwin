"""orbitwin day: RUNBOOK steps 1 to 7 in one command, failing closed at every gate.

The steps run through a scripted executor that writes what each RUNBOOK command
writes, so every branch is reached in milliseconds; tests/integration/
test_day_fake_console.py runs the real commands on the fake console.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from conftest import scratch_git_environment

from orbitwin import day as day_module
from orbitwin.cli import (
    PRECHECK_PASS,
    PRECHECK_PASS_END_TIME,
    PRECHECK_PASS_NO_TARGET,
    PRECHECK_PASS_NO_TARGET_END_TIME,
    _parser,
    main,
)
from orbitwin.day import EXIT_PAUSED, EXIT_STOP, NOTE_NAME, Completed, run_day, step7_score

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
REPORT = "ODC_2026_Team54.txt"
NOTE = NOTE_NAME
CANDIDATES_ALL = ("final", "final-01", "final-02")
SCRIPT = "public_gmat_candidate.script"
#: Before the fixture below replaces it.
REAL_PROVENANCE = day_module.provenance
#: What ranking_day_config.py prints for R1 with --mission-boundary-km 16900
#: --tmax-as-announced.
PUTS = ("WARNING: T_max 6711.667 s puts SC_Alien 17001.813 km from the centre, +101.813 km from "
        "the announced boundary 16900.0 km (tolerance 10.5 km, two-body), climbing +5.6047 "
        "km/s: the rules put it at the boundary, climbing. Check T_max, the boundary and the "
        "time origin against the announcement")
TAKEN = ("WARNING: T_max 6711.667 s taken as announced (--tmax-as-announced): SC_Alien is "
         "+101.813 km off the 16900.0 km boundary there (+18.2 s at +5.6047 km/s, two-body)")
SCORE = ["--kt-per-s", "0.003", "--ct-s", "5369.3336", "--kv-per-m-s", "0.0012",
         "--cv-m-s", "2000", "--cr-km", "300"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_script(argv: list[str]) -> list[str]:
    """`python -m orbitwin.tools.<tool> ...` as `python <tool>.py ...`, read by position here."""
    if argv[1:2] == ["-m"] and len(argv) > 2 and argv[2].startswith("orbitwin.tools."):
        return [argv[0], argv[2].removeprefix("orbitwin.tools.") + ".py", *argv[3:]]
    return argv


class Steps:
    """Each RUNBOOK command as the executor sees it, writing what it writes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], dict[str, str]]] = []
        #: Each argv as the day gave it, before `_as_script`.
        self.raw: list[list[str]] = []
        self.exit: dict[str, int] = {}
        self.readback: dict[str, list[str]] = {}
        self.stderr: dict[str, str] = {}
        self.summary: dict[str, Any] = {
            "best_candidate_is_gmat_verified": True,
            "best_certifiable_without_shallow_opt_in": "C-first",
            "closest_certifiable_miss": "C-closest",
            "zero_burn_closest_approach": {"km": 604.311, "measured_by": "gmat"},
        }
        self.rank: dict[str, Any] = {"best_submittable_note": None,
                                     "ceilings": {"best_submittable": "C-best"},
                                     "winner": {"candidate_id": "C-best"}}
        self.precheck = {"final": "pass", "final-01": "pass", "final-02": "pass"}
        self.last_line: dict[str, str] = {}
        self.scores = {"final": 96.3908, "final-01": 25.1735, "final-02": 53.11}
        self.step7_exit: dict[str, int] = {}
        self.step7_text: dict[str, str] = {}
        self.copy_score: float | None = None
        self.extra_bind_run = False
        self.origin_craft = "SC_Human"
        self.head_start_s = 0.0
        self.stdout_audit = "AUDIT: PASS\n"
        self.searches = 0

    def label(self, argv: list[str]) -> str:
        argv = _as_script(argv)
        tool = Path(argv[1]).name if len(argv) > 1 else ""
        if argv[0] == "bash":
            return "05-search" if argv[-1].endswith("day-scored.yaml") else "04-search"
        if tool == "ranking_day_config.py":
            return "05-scored-config" if "--day-config" in argv else "02-day-config"
        if tool == "audit_burn_table.py":
            return "07-audit-" + Path(argv[3]).parent.name
        if tool == "check_report_rows.py" and "--verification" in argv:
            return "07-bind-" + Path(argv[2]).parents[1].name
        if tool == "check_report_rows.py":
            report = Path(argv[2])
            return ("07-check-" + report.parents[1].name if report.parent.name == "submission"
                    else "07-check-copy")
        if argv[1:3] == ["-m", "orbitwin"]:
            if argv[3] == "certify":
                return f"certify-{Path(argv[argv.index('--output') + 1]).name}"
            return {"inspect": "03-inspect", "baseline": "03-baseline", "rank": "05-rank"}[argv[3]]
        raise AssertionError(argv)

    def __call__(self, argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        self.raw.append(list(argv))
        argv = _as_script(argv)
        label = self.label(argv)
        self.calls.append((label, argv, env))
        code = self.exit.get(label, 0)
        out, err = "", self.stderr.get(label, "")
        if label.startswith("07-audit-"):
            out = self.stdout_audit
        if code == 0:
            out, err = getattr(self, "_" + label.split("-", 1)[1].replace("-", "_").split("_")[0],
                               self._nothing)(label, argv, err)
        if label.startswith("07-check-"):
            name = label.removeprefix("07-check-")
            code = self.step7_exit.get(name, 0)
        return Completed(code, out, err)

    def _nothing(self, label: str, argv: list[str], err: str) -> tuple[str, str]:
        return "", err

    def _day(self, label: str, argv: list[str], err: str) -> tuple[str, str]:
        output = Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"# {label}\n", encoding="utf-8")
        return "\n".join(["config      : " + str(output), *self.readback.get(label, [])]), err

    _scored = _day

    def _baseline(self, label: str, argv: list[str], err: str) -> tuple[str, str]:
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "baseline.json").write_text("{}", encoding="utf-8")
        return "{}", err

    def _search(self, label: str, argv: list[str], err: str) -> tuple[str, str]:
        run = Path(argv[-2])
        self.searches += 1
        number = 1
        while (run / f"search-{number:02d}").exists():
            number += 1
        folder = run / f"search-{number:02d}"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        (folder / "candidates.json").write_text("[]", encoding="utf-8")
        latest = run / "latest"
        if latest.is_symlink():
            latest.unlink()
        latest.symlink_to(folder.name)
        return "", err

    def _rank(self, label: str, argv: list[str], err: str) -> tuple[str, str]:
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps(self.rank),
                                                           encoding="utf-8")
        return "{}", err

    def _final(self, label: str, argv: list[str], err: str) -> tuple[str, str]:
        output = Path(argv[argv.index("--output") + 1])
        name = output.name
        submission = output / "submission"
        submission.mkdir(parents=True)
        (output / "diagnostic").mkdir()
        (output / "diagnostic" / "verification.json").write_text("{}", encoding="utf-8")
        (submission / SCRIPT).write_text(
            R1.read_text(encoding="utf-8").replace("TeamXX", "Team54"), encoding="utf-8")
        (submission / REPORT).write_text(f"rows of {name}\n", encoding="utf-8")
        body = {
            "submission_script_name": SCRIPT,
            "official_report_files": [REPORT],
            "submission_files": {SCRIPT: _sha(submission / SCRIPT),
                                 REPORT: _sha(submission / REPORT)},
            "binding_pre_check": {"result": self.precheck[name], "runs": [{
                "exit_code": 7 if "target" in self.precheck[name] else 3, "command": [
                    "python", "check_report_rows.py", str(submission / REPORT), "--config",
                    "day.yaml", "--script", str(submission / SCRIPT), "--verification",
                    str(output / "diagnostic" / "verification.json"),
                    "--accept-end-time-binding-only"]}]
                + ([{"exit_code": 0, "command": ["python", "check_report_rows.py"]}]
                   if self.extra_bind_run else [])
                # Step 7 itself, which certify's pre-check runs last (R3, T14).
                + [{"exit_code": 0, "command": [
                    "python", "check_report_rows.py", str(submission / REPORT), "--config",
                    "day.yaml", "--certificate", str(submission / "certification.json")]}]
                if self.precheck[name].endswith("end-time-only") else []},
            "candidate": {"candidate_id": argv[argv.index("--candidate") + 1],
                          "successful": name == "final", "total_dv_km_s": 1.5,
                          "burns": [{"time_s": 101.0, "dv_inertial_km_s": [0.1, 0.0, 0.0]}]},
            "clock": {"t_team_s": 4301.4377, "mission_time_origin": "earliest",
                      "time_origin_craft": self.origin_craft,
                      "alignment_offset_s": self.head_start_s},
            "final_script_parity": {"minimum_distance_km": 3.5,
                                    "replayed_score": 96.3908 if name == "final" else None},
        }
        (submission / "certification.json").write_text(json.dumps(body), encoding="utf-8")
        default = PRECHECK_PASS if self.precheck[name] == "pass" else "WARNING: PRE-CHECK ..."
        return "{}", err + self.last_line.get(name, default) + "\n"


    def _audit(self, label: str, argv: list[str], err: str) -> tuple[str, str]:
        return self.stdout_audit, err

    def _check(self, label: str, argv: list[str], err: str) -> tuple[str, str]:
        name = label.removeprefix("07-check-")
        if name == "copy":
            source = Path(argv[argv.index("--certificate") + 1]).parents[1].name
            score = self.scores[source] if self.copy_score is None else self.copy_score
        else:
            score = self.scores[name]
        text = self.step7_text.get(name, "LAST row      : t = 4301.437731 s   dr = 3.500001 km\n"
                                   f"  validator reads last row   Tteam =  4301.4377 s "
                                   f"-> score {score:.4f}\n")
        return text, err

    def ran(self, label: str) -> int:
        return sum(1 for done, _, _ in self.calls if done == label)

    def argv(self, label: str) -> list[str]:
        return next(argv for done, argv, _ in self.calls if done == label)


@pytest.fixture
def steps() -> Steps:
    return Steps()


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
          fake_gmat: Path, fake_startup: Path) -> dict[str, Path]:
    monkeypatch.setattr(day_module, "DISK_LOW", tmp_path / "no-DISK_LOW")
    monkeypatch.setattr(day_module, "provenance", lambda root, expect, allow: {
        "source": "git", "commit": "c0ffee", "expected": "c0ffee", "on_main": True,
        "notes": []})
    stamps = iter(f"20261109T{n:06d}Z" for n in range(1000))
    monkeypatch.setattr(day_module, "_utc_stamp", lambda: next(stamps))
    return {"work": tmp_path / "days", "final": tmp_path / "final", "console": fake_gmat,
            "startup": fake_startup, "tmp": tmp_path}


def _args(roots: dict[str, Path], *extra: str, script: Path = R1,
          score: bool = True, team: str | None = "54") -> argparse.Namespace:
    """A new day's arguments; --team-number ``team`` unless ``extra`` gives one or it is None."""
    teamed = [] if team is None or "--team-number" in extra else ["--team-number", team]
    argv = ["day", str(script), "--horizon-s", "6711.667", *(SCORE if score else []),
            "--gmat-console", str(roots["console"]), "--startup-file", str(roots["startup"]),
            "--work-root", str(roots["work"]), "--final-root", str(roots["final"]),
            "--min-free-gb", "0.001", *teamed, *extra]
    return _parser().parse_args(argv)


def _resume(roots: dict[str, Path], folder: Path, *extra: str) -> argparse.Namespace:
    return _parser().parse_args(["day", "--resume", str(folder), "--min-free-gb", "0.001",
                                 *extra])


def _folder(roots: dict[str, Path]) -> Path:
    """The newest day folder."""
    return sorted(path for path in roots["work"].iterdir() if path.is_dir())[-1]


def _pending(folder: Path) -> list[str]:
    """Readbacks printed at a pause and not yet confirmed."""
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    confirmed = state.get("confirmed_readbacks") or {}
    return [step for step, digest in (state.get("shown_readbacks") or {}).items()
            if (confirmed.get(step) or {}).get("sha256") != digest]


def _confirmed(roots: dict[str, Path], code: int, executor: Any, *again: str) -> int:
    """Go on past each readback pause as two people would, with --confirm-readback."""
    for _ in range(4):
        if code != EXIT_PAUSED or not roots["work"].exists():
            return code
        pending = _pending(_folder(roots))
        if not pending:
            return code
        code = run_day(_resume(roots, _folder(roots), "--confirm-readback", pending[0],
                               *again), executor)
    return code


def _day(roots: dict[str, Path], executor: Any, *extra: str, script: Path = R1,
         score: bool = True) -> int:
    """A new day, through its readback pauses."""
    again = [flag for flag in extra if flag == "--allow-non-main"]
    return _confirmed(roots, run_day(_args(roots, *extra, script=script, score=score),
                                     executor), executor, *again)


def _nothing_copied(roots: dict[str, Path]) -> None:
    final = roots["final"]
    assert not (final / "latest").is_symlink()
    assert not final.exists() or not [p for p in final.iterdir() if not p.name.endswith(
        ".not-proven")]


# -- the success path --------------------------------------------------------------------


def test_the_day_copies_the_best_two_files_proves_them_and_moves_latest(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    assert _day(roots, steps) == 0
    latest = roots["final"] / "latest"
    target = roots["final"] / os.readlink(latest)
    assert latest.is_symlink() and not os.path.isabs(os.readlink(latest))
    assert sorted(path.name for path in target.iterdir()) == sorted(
        [REPORT, SCRIPT, "SHA256SUMS", NOTE])
    folder = _folder(roots)
    submission = folder / "run" / "final" / "submission"
    for name in (REPORT, SCRIPT):
        assert (target / name).read_bytes() == (submission / name).read_bytes()
        assert not os.access(target / name, os.W_OK)
    sums = (target / "SHA256SUMS").read_text(encoding="utf-8")
    assert sums == f"{_sha(target / SCRIPT)}  {SCRIPT}\n{_sha(target / REPORT)}  {REPORT}\n"
    done = subprocess.run(["sha256sum", "--check", "--quiet", "SHA256SUMS"], cwd=target,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    summary = (target / NOTE).read_text(encoding="utf-8")
    for line in ("chosen       : final (C-best)", "score        : 96.3908", "T_team       : "
                 "4301.4377 s", "delta-v      : 1.5 km/s = 1500.000 m/s",
                 "d_min        : 3.500000 km", "PRE-CHECK    : PASS",
                 "step 7       : exit 0 in the day folder, exit 0 on these copies",
                 "audit        : audit_burn_table.py exit 0",
                 f"startup file : {roots['startup']} (SHA-256 {_sha(roots['startup'])})",
                 "others       : final-01 25.1735, final-02 53.11",
                 "Upload exactly these two files, as they are:\nscript       : "
                 f"{SCRIPT}\nreport       : {REPORT}\n"):
        assert line in summary
    # Four runs: the first three stopped at the readbacks of steps 2, 3 and 5.
    expected = "c0ffee (the expected commit c0ffee)"
    for line in (f"run 0 (20261109T000001Z): {expected}; 02-day-config\n",
                 f"run 1 (20261109T000002Z): {expected}; 03-inspect\n",
                 f"run 2 (20261109T000004Z): {expected}; 03-baseline, 04-search, "
                 "05-scored-config\n",
                 f"run 3 (20261109T000006Z): {expected}; 05-search, 05-rank, certify-final, "
                 "07-audit-final, the copy\n"):
        assert f"commit       : {line}" in summary
    for step, run in (("2", 1), ("3", 2), ("5", 3)):
        assert (f"readback     : step {step} confirmed by run {run} (--confirm-readback "
                f"{step})") in summary
    # Step 7 ran on the copies themselves, the script included, before they took the name.
    partial = target.with_name(target.name + ".partial")
    copy = steps.argv("07-check-copy")
    assert copy[2] == str(partial / REPORT)
    assert copy[copy.index("--script") + 1] == str(partial / SCRIPT)
    assert copy[copy.index("--certificate") + 1] == str(submission / "certification.json")
    assert copy[copy.index("--config") + 1] == str(folder / "run" / "day-scored.yaml")
    shown = capsys.readouterr()
    assert summary in shown.out
    assert (f"upload exactly these two files: {target / SCRIPT} {target / REPORT}"
            in shown.err)
    # The RUNBOOK's steps, in its order, each once.
    order = [label for label, _, _ in steps.calls]
    assert order == ["02-day-config", "03-inspect", "03-baseline", "04-search",
                     "certify-final-01", "certify-final-02", "05-scored-config", "05-search",
                     "05-rank", "certify-final", "07-check-final", "07-check-final-01",
                     "07-check-final-02", "07-audit-final", "07-check-copy"]
    audit = steps.argv("07-audit-final")
    burns = folder / "run" / "final" / "burns.json"
    assert audit[2:] == [str(folder / "input" / "official.script"), str(burns), "--config",
                         str(folder / "run" / "day-scored.yaml"), "--gmat-console",
                         str(roots["console"]), "--startup-file", str(roots["startup"])]
    assert json.loads(burns.read_text(encoding="utf-8")) == {"burns": [
        {"time_s": 101.0, "dv_inertial_km_s": [0.1, 0.0, 0.0]}]}


def test_the_commands_are_the_runbooks(roots: dict[str, Path], steps: Steps) -> None:
    assert _day(roots, steps, "--tmax-as-announced", "--mission-boundary-km", "17000") == 0
    folder = _folder(roots)
    official = str(folder / "input" / "official.script")
    first = steps.argv("02-day-config")
    assert first[first.index("--script") + 1] == official
    assert "--tmax-as-announced" in first and "--mission-boundary-km" in first
    scored = steps.argv("05-scored-config")
    # Step 5 takes the boundary and the as-announced T_max from step 2's config.
    assert "--tmax-as-announced" not in scored and "--mission-boundary-km" not in scored
    assert scored[scored.index("--day-config") + 1] == str(folder / "run" / "day.yaml")
    assert scored[scored.index("--delta-v-unit") + 1] == "m/s"
    assert scored[scored.index("--zero-burn-dmin-km") + 1] == "604.311"
    for name, config, search in (("final-01", "day.yaml", "search-01"),
                                 ("final-02", "day.yaml", "search-01"),
                                 ("final", "day-scored.yaml", "search-02")):
        argv = steps.argv(f"certify-{name}")
        assert argv[argv.index("--runs") + 1] == "3"
        assert argv[argv.index("--config") + 1] == str(folder / "run" / config)
        assert argv[argv.index("--candidates") + 1] == str(
            folder / "run" / search / "candidates.json")
    second = steps.argv("certify-final-02")
    assert second[second.index("--candidate") + 1] == "C-closest"
    ranked = steps.argv("05-rank")
    assert ranked[ranked.index("--candidates") + 1] == str(
        folder / "run" / "search-02" / "candidates.json")


def test_a_zero_burn_that_touches_gives_no_readback_flag(
    roots: dict[str, Path], steps: Steps,
) -> None:
    """ranking_day_config.py refuses --zero-burn-dmin-km 0; the flag is optional."""
    steps.summary["zero_burn_closest_approach"] = {"km": 0.0, "measured_by": "gmat"}
    assert _day(roots, steps) == 0
    assert "--zero-burn-dmin-km" not in steps.argv("05-scored-config")


def test_a_tie_goes_to_the_runbooks_order(roots: dict[str, Path], steps: Steps) -> None:
    steps.scores = {"final": 25.1735, "final-01": 25.1735, "final-02": 20.0}
    assert _day(roots, steps) == 0
    summary = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "chosen       : final (" in summary


def test_a_safety_net_that_scores_higher_is_the_upload(
    roots: dict[str, Path], steps: Steps,
) -> None:
    steps.scores = {"final": 40.0, "final-01": 25.0, "final-02": 53.11}
    assert _day(roots, steps) == 0
    summary = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "chosen       : final-02 (C-closest)" in summary


def test_a_rerun_goes_to_a_new_timestamp(roots: dict[str, Path]) -> None:
    assert _day(roots, Steps()) == 0
    first = os.readlink(roots["final"] / "latest")
    before = {path.name: path.read_bytes() for path in (roots["final"] / first).iterdir()}
    assert _day(roots, Steps()) == 0
    second = os.readlink(roots["final"] / "latest")
    assert second != first
    assert {path.name: path.read_bytes() for path in (roots["final"] / first).iterdir()} == before
    assert len([path for path in roots["work"].iterdir() if path.is_dir()]) == 2


def test_no_closest_miss_and_none_submittable(roots: dict[str, Path], steps: Steps) -> None:
    steps.summary["closest_certifiable_miss"] = "C-first"
    steps.rank["ceilings"] = {"best_submittable": None}
    assert _day(roots, steps) == 0
    assert steps.ran("certify-final-02") == 0 and steps.ran("certify-final") == 0
    summary = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "chosen       : final-01 (C-first)" in summary


# -- refusals before anything ----------------------------------------------------------


def test_a_latest_that_is_not_a_symlink_is_refused_first(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    (roots["final"] / "latest").mkdir(parents=True)
    assert _day(roots, steps) == EXIT_STOP
    assert "is not a symlink" in capsys.readouterr().err
    assert not steps.calls and not roots["work"].exists()


def test_disk_low_and_free_space_are_checked_first(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    flag = roots["tmp"] / "DISK_LOW"
    flag.touch()
    monkeypatch.setattr(day_module, "DISK_LOW", flag)
    assert _day(roots, steps) == EXIT_STOP
    assert "DISK_LOW exists" in capsys.readouterr().err
    flag.unlink()
    monkeypatch.setattr(day_module, "_free_bytes", lambda path: 5 * 2**30)
    assert _day(roots, steps, "--min-free-gb", "20") == EXIT_STOP
    assert "5.0 GiB free" in capsys.readouterr().err
    assert not steps.calls and not roots["work"].exists()


def test_disk_is_checked_again_before_each_step(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = steps.__call__
    flag = roots["tmp"] / "DISK_LOW"
    monkeypatch.setattr(day_module, "DISK_LOW", flag)

    def filling(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        done = real(argv, env, cwd, log)
        if steps.label(argv) == "04-search":
            flag.touch()
        return done

    assert _day(roots, filling) == EXIT_STOP
    assert [label for label, _, _ in steps.calls][-1] == "04-search"
    _nothing_copied(roots)


@pytest.mark.parametrize("half", ["--gmat-console", "--startup-file"])
def test_the_gmat_pair_is_required(
    roots: dict[str, Path], steps: Steps, half: str, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Half the pair by flag is refused, not filled from the environment (BASH_ENV exports
    one on this workstation): orbitwin, competition_day.sh and ranking_day_config.py refuse
    it too."""
    other = roots["tmp"] / "other-gmat"
    other.mkdir()
    for name in ("GmatConsole", "gmat_startup_file.txt"):
        (other / name).write_text("", encoding="utf-8")
    monkeypatch.setenv("GMAT_CONSOLE", str(other / "GmatConsole"))
    monkeypatch.setenv("GMAT_STARTUP", str(other / "gmat_startup_file.txt"))
    args = _args(roots)
    setattr(args, half.removeprefix("--").replace("-", "_"), None)
    assert run_day(args, steps) == EXIT_STOP
    assert "must be given together" in capsys.readouterr().err
    # Neither flag: both from the environment, and half of it is refused too.
    args.gmat_console = args.startup_file = None
    monkeypatch.delenv("GMAT_STARTUP")
    assert run_day(args, steps) == EXIT_STOP
    assert "GMAT_CONSOLE and GMAT_STARTUP must be set together" in capsys.readouterr().err
    monkeypatch.delenv("GMAT_CONSOLE")
    assert run_day(args, steps) == EXIT_STOP
    assert "every step of the day flies GMAT" in capsys.readouterr().err
    assert not steps.calls and not roots["work"].exists()


@pytest.mark.parametrize("name", ["../tide.txt", "/abs/tide.txt"])
def test_a_companion_must_lie_under_the_scripts_folder(
    roots: dict[str, Path], steps: Steps, name: str, capsys: pytest.CaptureFixture[str],
) -> None:
    assert _day(roots, steps, "--companion", name) == EXIT_STOP
    assert "relative to the official script's folder" in capsys.readouterr().err
    assert not steps.calls


def test_the_official_script_and_its_files_are_copied_read_only(
    roots: dict[str, Path], steps: Steps,
) -> None:
    given = roots["tmp"] / "given"
    (given / "data").mkdir(parents=True)
    script = given / "official.script"
    script.write_bytes(R1.read_bytes())
    (given / "data" / "Earth.tide").write_text("tide\n", encoding="utf-8")
    assert _day(roots, steps, "--companion", "data/Earth.tide", script=script) == 0
    folder = _folder(roots)
    copy = folder / "input" / "official.script"
    assert copy.read_bytes() == R1.read_bytes() and not os.access(copy, os.W_OK)
    assert (folder / "input" / "data" / "Earth.tide").read_text(encoding="utf-8") == "tide\n"
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["official"]["sha256"] == _sha(R1)
    assert state["official"]["companions"] == {"data/Earth.tide": _sha(
        given / "data" / "Earth.tide")}


def test_a_checkout_other_than_the_expected_commit_is_refused(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(day_module, "provenance", REAL_PROVENANCE)
    repo = _repository(roots["tmp"], on_main=False)
    monkeypatch.setattr(day_module, "ROOT", repo)
    first = _git(repo, "rev-parse", "HEAD~1")
    assert _day(roots, steps, "--expect-commit", first) == EXIT_STOP
    assert f"not the expected {first[:12]}" in capsys.readouterr().err
    assert not steps.calls
    assert _day(roots, steps, "--expect-commit", first, "--allow-non-main") == 0
    summary = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "(NOT as expected: --allow-non-main: this is commit " in summary


# -- provenance ---------------------------------------------------------------------------


def _git(root: Path, *argv: str) -> str:
    return subprocess.run(["git", "-C", str(root), *argv], capture_output=True, text=True,
                          check=True, env=scratch_git_environment(root)).stdout.strip()


def _repository(tmp_path: Path, on_main: bool = True) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    _git(tmp_path, "init", "-q", str(root))
    (root / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "-m", "one")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    if not on_main:
        (root / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
        _git(root, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "-am", "two")
    return root


def test_provenance_takes_the_expected_commit_and_fetches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetch moved origin/main under a day in progress (#193 review, item 5)."""
    provenance = REAL_PROVENANCE
    asked: list[tuple[str, ...]] = []
    real = day_module._git

    def git(root: Path, *argv: str, timeout_s: float = 30.0) -> str | None:
        asked.append(argv)
        return real(root, *argv, timeout_s=timeout_s)

    monkeypatch.setattr(day_module, "_git", git)
    root = _repository(tmp_path)
    head = _git(root, "rev-parse", "HEAD")
    record = provenance(root, head, False)
    assert record["on_main"] is True and record["commit"] == head
    assert provenance(root, head[:12].upper(), False)["on_main"] is True
    # origin/main moving on is nothing to the day.
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD~0")
    assert provenance(root, head, False)["on_main"] is True
    with pytest.raises(day_module.DayStop, match="no --expect-commit names the commit"):
        provenance(root, None, False)
    with pytest.raises(day_module.DayStop, match="is not a commit"):
        provenance(root, "main", False)
    (root / "src" / "a.py").write_text("x = 3\n", encoding="utf-8")
    with pytest.raises(day_module.DayStop, match=r"local changes: M src/a\.py"):
        provenance(root, head, False)
    _git(root, "checkout", "-q", "src/a.py")
    (root / "src" / "stray.py").write_text("", encoding="utf-8")
    with pytest.raises(day_module.DayStop, match=r"local changes: \?\? src/stray.py"):
        provenance(root, head, False)
    # Untracked outside the code is the day's own output.
    (root / "src" / "stray.py").unlink()
    (root / "run").mkdir()
    (root / "run" / "x").write_text("", encoding="utf-8")
    assert provenance(root, head, False)["on_main"] is True
    off = _repository(tmp_path / "off", on_main=False)
    with pytest.raises(day_module.DayStop, match="not the expected"):
        provenance(off, head, False)
    allowed = provenance(off, head, True)
    assert allowed["on_main"] is False and "--allow-non-main" in allowed["notes"][-1]
    assert asked and not any("fetch" in argv for argv in asked)


def test_provenance_of_an_archive_reads_build_info(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    import orbitwin.cli as cli

    provenance = REAL_PROVENANCE
    info = tmp_path / "BUILD_INFO.json"
    commit = "abc1234def"
    info.write_text(json.dumps({"commit": commit, "committed_utc": "x", "version": "v",
                                "dirty": False}), encoding="utf-8")
    monkeypatch.setattr(cli, "BUILD_INFO", info)
    monkeypatch.setattr(cli, "__version__", "v")
    files = {"src/orbitwin/day.py": "x = 1\n", "scripts/s.py": "", "config/ranking.yaml": "a: 1\n",
             "docs/RUNBOOK.md": "# r\n"}
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    (tmp_path / "SHA256SUMS.txt").write_text("".join(
        f"{_sha(tmp_path / name)}  {name}\n" for name in files), encoding="utf-8")
    # Caches and bytecode are the release's to make, as check_release_tree.py says.
    (tmp_path / "src" / "orbitwin" / "__pycache__").mkdir()
    (tmp_path / "src" / "orbitwin" / "__pycache__" / "day.cpython-313.pyc").write_bytes(b"")
    assert provenance(tmp_path, commit[:7], False)["commit"] == commit
    # A hand edit, or a file the release does not have, under src, scripts or config.
    (tmp_path / "config" / "ranking.yaml").write_text("a: 1\n# hand edit\n", encoding="utf-8")
    (tmp_path / "scripts" / "extra.py").write_text("", encoding="utf-8")
    with pytest.raises(day_module.DayStop, match=r"differs from its SHA256SUMS.txt: "
                       r"config/ranking.yaml \(changed\), scripts/extra.py \(not in the "
                       r"release\)"):
        provenance(tmp_path, commit, False)
    allowed = provenance(tmp_path, commit, True)
    assert allowed["on_main"] is False and allowed["local_changes"] == [
        "config/ranking.yaml (changed)", "scripts/extra.py (not in the release)"]
    (tmp_path / "config" / "ranking.yaml").write_text("a: 1\n", encoding="utf-8")
    (tmp_path / "scripts" / "extra.py").unlink()
    # Outside the code it is not the day's business.
    (tmp_path / "docs" / "RUNBOOK.md").write_text("# edited\n", encoding="utf-8")
    (tmp_path / "src" / "orbitwin" / "day.py").unlink()
    with pytest.raises(day_module.DayStop, match=r"src/orbitwin/day.py \(missing\)"):
        provenance(tmp_path, commit, False)
    (tmp_path / "src" / "orbitwin" / "day.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(day_module.DayStop, match="not the expected"):
        provenance(tmp_path, "abc1235", False)
    info.write_text(json.dumps({"commit": commit, "committed_utc": "x", "version": "v",
                                "dirty": False, "provenance": {"allow_non_main": True,
                                                               "matches": None}}),
                    encoding="utf-8")
    with pytest.raises(day_module.DayStop, match="not built from origin/main"):
        provenance(tmp_path, commit, False)
    info.unlink()
    with pytest.raises(day_module.DayStop, match="neither a git checkout"):
        provenance(tmp_path, commit, False)


# -- infrastructure steps stop the day --------------------------------------------------


@pytest.mark.parametrize("label", ["02-day-config", "03-inspect", "03-baseline", "04-search",
                                   "05-scored-config", "05-search", "05-rank"])
def test_a_step_that_fails_stops_the_day_there(
    roots: dict[str, Path], steps: Steps, label: str, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.exit[label] = 2
    steps.stderr[label] = "orbitwin: ERROR: something\n"
    assert _day(roots, steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert f"step {label} exited 2: orbitwin: ERROR: something" in err
    assert "Nothing was copied for upload" in err
    assert [done for done, _, _ in steps.calls][-1] == label
    _nothing_copied(roots)


def test_a_step_that_exits_0_without_its_file_stops(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Steps, "_baseline", lambda self, label, argv, err: ("", err))
    assert _day(roots, steps) == EXIT_STOP
    assert "step 03-baseline exited 0 but wrote no" in capsys.readouterr().err
    _nothing_copied(roots)


def test_the_window_is_never_guessed(roots: dict[str, Path], steps: Steps,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    steps.exit["02-day-config"] = 2
    steps.stderr["02-day-config"] = (
        "refused, nothing written: SC_Alien and SC_Human start 1800 s apart; set "
        "policy.closest_approach_window. ...\n")
    assert _day(roots, steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert "orbitwin day does not guess it" in err
    assert "--closest-approach-window after_alignment" in err
    assert "--closest-approach-window from_origin" in err


def test_the_window_given_is_passed_on(roots: dict[str, Path], steps: Steps) -> None:
    assert _day(roots, steps, "--closest-approach-window", "from_origin") == 0
    argv = steps.argv("02-day-config")
    assert argv[argv.index("--closest-approach-window") + 1] == "from_origin"


@pytest.mark.parametrize("label", ["02-day-config", "05-scored-config"])
def test_a_readback_warning_stops_until_two_people_read_it(
    roots: dict[str, Path], steps: Steps, label: str, capsys: pytest.CaptureFixture[str],
) -> None:
    warning = "WARNING: the time term moves under 0.5 points over its whole range"
    steps.readback[label] = [warning]
    assert _day(roots, steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert warning in err and "--accept-config-warnings" in err
    folder = _folder(roots)
    _nothing_copied(roots)
    calls = len(steps.calls)
    step = label[1]
    code = run_day(_resume(roots, folder, "--accept-config-warnings", "--confirm-readback",
                           step), steps)
    assert _confirmed(roots, code, steps) == 0
    # The config was not written again: it is the file the people read.
    assert steps.ran(label) == 1 and len(steps.calls) > calls
    summary = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert f"accepted     : {label}: {warning}" in summary


def test_the_as_announced_warning_asked_for_does_not_stop(
    roots: dict[str, Path], steps: Steps,
) -> None:
    """Both lines ranking_day_config.py prints for it, as it prints them (PUTS, TAKEN)."""
    steps.readback["02-day-config"] = steps.readback["05-scored-config"] = [PUTS, TAKEN]
    boundary = ("--mission-boundary-km", "16900")
    assert _day(roots, steps, "--tmax-as-announced", *boundary) == 0
    # Without the flag both are lines to read.
    assert _day(roots, steps, *boundary) == EXIT_STOP
    # Another T_max's or boundary's lines are not the ones the flag asked for.
    steps.readback["02-day-config"] = [PUTS.replace("6711.667", "6711.6"), TAKEN]
    assert _day(roots, steps, "--tmax-as-announced", *boundary) == EXIT_STOP
    steps.readback["02-day-config"] = [PUTS, TAKEN.replace("16900.0", "17000.0")]
    assert _day(roots, steps, "--tmax-as-announced", *boundary) == EXIT_STOP
    # The taking alone is not the pair.
    steps.readback["02-day-config"] = [TAKEN]
    assert _day(roots, steps, "--tmax-as-announced", *boundary) == EXIT_STOP


def test_a_search_that_verified_nothing_stops_4a(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.summary["best_candidate_is_gmat_verified"] = False
    assert _day(roots, steps) == EXIT_STOP
    assert "verified nothing in GMAT" in capsys.readouterr().err
    assert steps.ran("certify-final-01") == 0


def test_a_rank_searched_under_other_coefficients_stops(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.rank["best_submittable_note"] = "searched under another score block"
    assert _day(roots, steps) == EXIT_STOP
    assert "rank: searched under another score block" in capsys.readouterr().err
    assert steps.ran("certify-final") == 0


# -- candidate gates: a person decides --------------------------------------------------


@pytest.mark.parametrize("refusal", ["orbitwin: ERROR [gmat-verdict]: GMAT flies it worse",
                                     "orbitwin: ERROR: GMAT flies it worse"])
def test_a_refused_certify_stops_before_the_copy_and_exclude_goes_on(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str], refusal: str,
) -> None:
    steps.exit["certify-final-02"] = 2
    # The refusal is quoted, not the line printed before it: the coded form and the older one.
    steps.stderr["certify-final-02"] = f"orbitwin certify: a note first\n{refusal}\n"
    assert _day(roots, steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert f"final-02: certify refused it (exit 2): {refusal}" in err
    folder = _folder(roots)
    # A verdict: both forms, and which one applies.
    assert ("final-02: certify's refusal is its verdict on the candidate, so --exclude applies"
            f". To leave it out and choose among the rest, run:\n\norbitwin day --resume "
            f"{folder} --exclude final-02\n\n--include final-02 runs its certify again "
            "instead. That applies only when its log shows GMAT did not finish a run") in err
    assert "That is no verdict" not in err
    # Every other step still ran, so a decision costs seconds; the refusal ran once, not
    # again at the resume that confirmed step 5's readback.
    assert steps.ran("certify-final") == 1 and steps.ran("07-check-final") == 1
    assert steps.ran("certify-final-02") == 1
    _nothing_copied(roots)
    folder = _folder(roots)
    before = len(steps.calls)
    assert run_day(_resume(roots, folder, "--exclude", "final-02"), steps) == 0
    rerun = [label for label, _, _ in steps.calls[before:]]
    assert rerun == ["07-check-final", "07-check-final-01", "07-audit-final", "07-check-copy"]
    summary = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "excluded     : final-02" in summary and "chosen       : final (" in summary


@pytest.mark.parametrize(("result", "said"), [
    ("failed", "PRE-CHECK FAILED"),
    ("pass-end-time-only", "PRE-CHECK PASS (end time only)"),
    ("pass-target-not-in-report-end-time-only",
     "PRE-CHECK PASS (target not in report; end time only)"),
    ("surface-not-judged", "PRE-CHECK SURFACE NOT JUDGED"),
    ("surface-template-below", "PRE-CHECK SURFACE TEMPLATE BELOW"),
    ("surface-undecided", "PRE-CHECK SURFACE UNDECIDED"),
    ("not run", "PRE-CHECK NOT RUN"),
    ("skipped", "PRE-CHECK SKIPPED"),
    ("absent", "PRE-CHECK absent"),
])
def test_every_precheck_but_pass_stops_before_the_copy(
    roots: dict[str, Path], steps: Steps, result: str, said: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    steps.precheck["final-01"] = result
    assert _day(roots, steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert f"final-01: {said}" in err
    # Its report is not even checked; the others are.
    assert steps.ran("07-check-final-01") == 0 and steps.ran("07-check-final") == 1
    _nothing_copied(roots)


def test_a_recorded_pass_needs_certifys_own_last_line(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.last_line["final"] = PRECHECK_PASS_END_TIME
    assert _day(roots, steps) == EXIT_STOP
    assert "final: PRE-CHECK pass: certify's last line was" in capsys.readouterr().err
    _nothing_copied(roots)


@pytest.mark.parametrize("code", [1, 2, 3, 4, 5, 6])
def test_every_step7_exit_but_0_stops_before_the_copy(
    roots: dict[str, Path], steps: Steps, code: int, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.step7_exit["final-02"] = code
    assert _day(roots, steps) == EXIT_STOP
    assert f"final-02: step 7 exit {code}" in capsys.readouterr().err
    _nothing_copied(roots)


def test_a_step7_without_a_score_stops(roots: dict[str, Path], steps: Steps,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    steps.step7_text["final"] = "VERDICT: the LAST row is inside the sphere.\n"
    assert _day(roots, steps) == EXIT_STOP
    assert "final: step 7 exited 0 but printed no score" in capsys.readouterr().err


def test_nothing_certified_stops(roots: dict[str, Path], steps: Steps,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    steps.summary.update(best_certifiable_without_shallow_opt_in=None,
                         closest_certifiable_miss=None)
    steps.rank["ceilings"] = {"best_submittable": None}
    assert _day(roots, steps) == EXIT_STOP
    assert "no candidate was certified" in capsys.readouterr().err
    _nothing_copied(roots)


def test_a_copy_that_step7_does_not_pass_never_becomes_latest(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    assert _day(roots, steps) == 0
    first = os.readlink(roots["final"] / "latest")
    steps_again = Steps()
    steps_again.step7_exit["copy"] = 2
    assert _day(roots, steps_again) == EXIT_STOP
    assert os.readlink(roots["final"] / "latest") == first
    kept = [path.name for path in roots["final"].iterdir() if path.name.endswith(".not-proven")]
    assert len(kept) == 1 and "never as latest" in capsys.readouterr().err
    # A copy scored otherwise than in the day folder is refused too.
    third = Steps()
    third.copy_score = 96.0
    assert _day(roots, third) == EXIT_STOP
    assert os.readlink(roots["final"] / "latest") == first


def test_latest_is_swapped_by_rename(roots: dict[str, Path], steps: Steps,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    replaced: list[tuple[str, str]] = []
    real = os.replace

    def replace(source: str | Path, target: str | Path) -> None:
        replaced.append((Path(source).name, Path(target).name))
        real(source, target)

    monkeypatch.setattr(day_module.os, "replace", replace)
    assert _day(roots, steps) == 0
    assert (f".latest.{os.getpid()}", "latest") in replaced


# -- pause and resume --------------------------------------------------------------------


def test_without_coefficients_it_pauses_after_4a_and_resumes_with_them(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    assert _day(roots, steps, score=False) == EXIT_PAUSED
    err = capsys.readouterr().err
    assert "PAUSED: steps 1 to 4a are done" in err and "--kt-per-s <kt>" in err
    assert [label for label, _, _ in steps.calls][-1] == "certify-final-02"
    folder = _folder(roots)
    _nothing_copied(roots)
    assert run_day(_resume(roots, folder), steps) == EXIT_PAUSED
    assert run_day(_resume(roots, folder, *SCORE), steps) == EXIT_PAUSED
    # The recorded ones bind from now on.
    other = [*SCORE[:-1], "3000"]
    assert run_day(_resume(roots, folder, *other), steps) == EXIT_STOP
    assert "not the ones recorded" in capsys.readouterr().err
    assert _confirmed(roots, run_day(_resume(roots, folder), steps), steps) == 0
    assert steps.ran("04-search") == 1 and steps.ran("certify-final-01") == 1


def test_half_the_coefficients_is_refused(roots: dict[str, Path], steps: Steps,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    args = _args(roots, score=False)
    args.kt_per_s = 0.003
    assert run_day(args, steps) == EXIT_STOP
    assert "missing --ct-s --kv-per-m-s --cv-m-s --cr-km" in capsys.readouterr().err
    assert not steps.calls


def test_resume_refuses_new_inputs(roots: dict[str, Path], steps: Steps,
                                   capsys: pytest.CaptureFixture[str]) -> None:
    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    for extra in (["--horizon-s", "6000"], ["--base", "x.yaml"], ["--gmat-workers", "0"],
                  ["--final-root", "/elsewhere"], ["--team-number", "54"]):
        assert run_day(_resume(roots, folder, *extra), steps) == EXIT_STOP
        assert "given again" in capsys.readouterr().err
    args = _resume(roots, folder)
    args.script = str(R1)
    assert run_day(args, steps) == EXIT_STOP


def test_resume_reruns_only_what_did_not_end_0(roots: dict[str, Path], steps: Steps) -> None:
    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    del steps.exit["05-rank"]
    before = len(steps.calls)
    assert run_day(_resume(roots, folder), steps) == 0
    assert [label for label, _, _ in steps.calls[before:]] == [
        "05-rank", "certify-final", "07-check-final", "07-check-final-01",
        "07-check-final-02", "07-audit-final", "07-check-copy"]


def test_resume_refuses_a_changed_file(roots: dict[str, Path], steps: Steps,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    del steps.exit["05-rank"]
    candidates = folder / "run" / "search-02" / "candidates.json"
    candidates.write_text("[1]", encoding="utf-8")
    assert run_day(_resume(roots, folder), steps) == EXIT_STOP
    assert f"{candidates} is not what step 05-search wrote" in capsys.readouterr().err
    candidates.write_text("[]", encoding="utf-8")
    official = folder / "input" / "official.script"
    official.chmod(0o644)
    official.write_text("changed", encoding="utf-8")
    assert run_day(_resume(roots, folder), steps) == EXIT_STOP
    assert "is not the copy step 1 made" in capsys.readouterr().err
    _nothing_copied(roots)


def test_resume_refuses_another_commit_or_gmat(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(day_module, "provenance", lambda root, expect, allow: {
        "commit": "beef", "on_main": True, "notes": []})
    assert run_day(_resume(roots, folder), steps) == EXIT_STOP
    assert "one day folder, one commit" in capsys.readouterr().err
    monkeypatch.setattr(day_module, "provenance", lambda root, expect, allow: state["commit"])
    state["gmat"]["console_sha256"] = "0" * 64
    (folder / "state.json").write_text(json.dumps(state), encoding="utf-8")
    assert run_day(_resume(roots, folder), steps) == EXIT_STOP
    assert "is not the file this day folder ran" in capsys.readouterr().err


def test_an_interrupted_step_is_set_aside(roots: dict[str, Path], steps: Steps) -> None:
    steps.exit["certify-final"] = 130
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    left = folder / "run" / "final"
    (left / "staging").mkdir(parents=True)
    del steps.exit["certify-final"]
    assert run_day(_resume(roots, folder), steps) == 0
    assert (folder / "run" / "final.interrupted" / "staging").is_dir()
    assert (folder / "run" / "final" / "submission" / "certification.json").is_file()


def test_one_day_folder_runs_once_at_a_time(roots: dict[str, Path], steps: Steps,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    with (folder / ".day.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        assert run_day(_resume(roots, folder), steps) == EXIT_STOP
    assert "another orbitwin day is running" in capsys.readouterr().err


def test_resume_of_no_day_folder(roots: dict[str, Path], steps: Steps,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    assert run_day(_resume(roots, roots["tmp"]), steps) == EXIT_STOP
    assert "is not a day folder" in capsys.readouterr().err


# -- the environment of each step ------------------------------------------------------


def test_each_step_gets_the_days_gmat_pair_and_no_bash_env(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BASH_ENV re-exports a GMAT pair into competition_day.sh's bash otherwise."""
    monkeypatch.setenv("BASH_ENV", str(roots["tmp"] / "tokens.env"))
    monkeypatch.setenv("GMAT_CONSOLE", "/somewhere/else/GmatConsole")
    monkeypatch.setenv("GMAT_STARTUP", "/somewhere/else/startup.txt")
    assert _day(roots, steps) == 0
    folder = _folder(roots)
    for _, _, env in steps.calls:
        assert "BASH_ENV" not in env
        assert env["GMAT_CONSOLE"] == str(roots["console"])
        assert env["GMAT_STARTUP"] == str(roots["startup"])
        assert env["PATH"].split(os.pathsep)[0] == str(folder / ".bin")
        assert env["PYTHONPATH"] == str(ROOT / "src")


def test_the_environments_pair_is_taken_when_no_flag_names_one(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GMAT_CONSOLE", str(roots["console"]))
    monkeypatch.setenv("GMAT_STARTUP", str(roots["startup"]))
    args = _args(roots)
    args.gmat_console = args.startup_file = None
    assert _confirmed(roots, run_day(args, steps), steps) == 0
    state = json.loads((_folder(roots) / "state.json").read_text(encoding="utf-8"))
    assert state["gmat"]["console"] == str(roots["console"])


def test_the_python_shim_runs_this_interpreter(roots: dict[str, Path], steps: Steps) -> None:
    import sys

    assert _day(roots, steps) == 0
    shim = _folder(roots) / ".bin" / "python"
    done = subprocess.run([str(shim), "-c", "import sys; print(sys.executable)"],
                          capture_output=True, text=True, check=True)
    assert done.stdout.strip() == sys.executable


def test_run_logged_keeps_both_streams(tmp_path: Path) -> None:
    import sys

    done = day_module.run_logged(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); "
         "sys.exit(3)"], dict(os.environ), tmp_path, tmp_path / "logs" / "step")
    assert (done.returncode, done.stdout, done.stderr) == (3, "out\n", "err\n")
    assert (tmp_path / "logs" / "step.out").read_text(encoding="utf-8") == "out\n"
    assert (tmp_path / "logs" / "step.err").read_text(encoding="utf-8") == "err\n"


# -- pieces ------------------------------------------------------------------------------


def test_step7_score_reads_a_hit_and_a_miss_and_nothing_else() -> None:
    hit = ("  validator reads last row   Tteam =   4301.4377 s -> score 96.3908\n"
           "  rule 4 first entry         Tteam =   4300.0000 s -> score 96.4000\n")
    assert step7_score(hit) == 96.3908
    miss = "  scored miss: distance 0.0123 + time 1.2 + delta-v 22.0 = 23.2123\n"
    assert step7_score(miss) == 23.2123
    assert step7_score("VERDICT: the LAST row is inside the sphere.\n") is None
    assert step7_score(hit + hit) is None
    assert step7_score("  scored miss: distance nan + time 0 + delta-v 0 = nan\n") is None


def test_the_cli_runs_day(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["day", "--resume", str(tmp_path), "--allow-non-main"]) == EXIT_STOP
    assert "is not a day folder" in capsys.readouterr().err


def test_the_day_parser_takes_no_abbreviation(capsys: pytest.CaptureFixture[str]) -> None:
    """--kt would stand for --kt-per-s and drop the unit."""
    with pytest.raises(SystemExit):
        _parser().parse_args(["day", "x.script", "--kt", "0.003"])
    assert "unrecognized arguments: --kt" in capsys.readouterr().err


def test_a_recorded_result_other_than_pass_stops_even_after_a_pass_line(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.precheck["final"] = "failed"
    steps.last_line["final"] = PRECHECK_PASS
    assert _day(roots, steps) == EXIT_STOP
    assert "final: PRE-CHECK FAILED" in capsys.readouterr().err
    _nothing_copied(roots)


def test_a_submission_changed_after_its_check_is_not_copied(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    real = steps.__call__

    def tampering(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        if steps.label(argv) == "07-check-final-02":
            report = _folder(roots) / "run" / "final" / "submission" / REPORT
            report.write_text("edited\n", encoding="utf-8")
        return real(argv, env, cwd, log)

    assert _day(roots, tampering) == EXIT_STOP
    assert f"{REPORT} is not the file certify listed" in capsys.readouterr().err
    _nothing_copied(roots)


def test_only_warnings_already_shown_can_be_accepted(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    """--accept-config-warnings given for step 2's lines does not wave step 5's through."""
    steps.readback["02-day-config"] = ["WARNING: one"]
    steps.readback["05-scored-config"] = ["WARNING: two"]
    # Given before any line was printed, it accepts nothing.
    assert _day(roots, steps, "--accept-config-warnings") == EXIT_STOP
    assert "WARNING: one" in capsys.readouterr().err
    folder = _folder(roots)
    accept = ("--accept-config-warnings", "--confirm-readback")
    assert run_day(_resume(roots, folder, *accept, "2"), steps) == EXIT_PAUSED
    assert run_day(_resume(roots, folder, *accept, "3"), steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert "WARNING: two" in err and steps.ran("05-search") == 0
    assert run_day(_resume(roots, folder, *accept, "5"), steps) == 0
    summary = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "02-day-config: WARNING: one" in summary and "05-scored-config: WARNING: two" in summary


def test_an_interrupt_says_how_to_go_on(roots: dict[str, Path], steps: Steps,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    real = steps.__call__

    def interrupted(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        if steps.label(argv) == "05-search":
            raise KeyboardInterrupt
        return real(argv, env, cwd, log)

    assert _day(roots, interrupted) == EXIT_STOP
    assert (f"interrupted. Finished steps are kept. Go on with:\n\n"
            f"orbitwin day --resume {_folder(roots)}\n\n") in capsys.readouterr().err
    _nothing_copied(roots)


# -- review of #193: readbacks, provenance, latest, the copy ------------------------------


def test_the_as_announced_pair_is_exempt_on_a_real_readback(tmp_path: Path) -> None:
    """ranking_day_config.py itself, on R1, reading the script in-process."""
    import sys

    from orbitwin.day import _as_announced, _readback_warnings

    env = {key: value for key, value in os.environ.items()
           if key not in {"BASH_ENV", "GMAT_CONSOLE", "GMAT_STARTUP"}}
    env["PYTHONPATH"] = str(ROOT / "src")
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ranking_day_config.py"), "--script", str(R1),
         "--output", str(tmp_path / "day.yaml"), "--horizon-s", "6711.667",
         "--team-number", "54", "--mission-boundary-km", "16900", "--tmax-as-announced",
         # One GmatConsole run at once: a machine of fewer cores than the profile's 8
         # would add a WARNING about them.
         "--gmat-workers", "1"],
        env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    lines = [line for line in done.stdout.splitlines() if line.startswith("WARNING:")]
    assert lines == [PUTS, TAKEN]
    assert _readback_warnings(done.stdout, _as_announced(6711.667, 16900.0)) == []
    assert _readback_warnings(done.stdout, []) == lines
    assert _readback_warnings(done.stdout, _as_announced(6711.667, 17000.0)) == lines
    assert _readback_warnings(done.stdout, _as_announced(6711.6, 16900.0)) == lines


def test_each_readback_is_shown_and_waits_for_two_people(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    """Cv typed in km/s raises no WARNING; only a person reading the readback sees it."""
    steps.readback["02-day-config"] = ["T_max       : 6711.667 s = 1 h 51 min 51.667 s"]
    steps.readback["05-scored-config"] = ["Cv          : 2.0 m/s", "NOTE: a note"]
    steps.stderr["03-inspect"] = ("orbitwin inspect: mission clock: origin earliest; the craft "
                                  "start together\n")
    assert run_day(_args(roots), steps) == EXIT_PAUSED
    err = capsys.readouterr().err
    assert "T_max       : 6711.667 s = 1 h 51 min 51.667 s" in err
    assert "--confirm-readback 2" in err and steps.ran("03-inspect") == 0
    folder = _folder(roots)
    # A confirmation of what has not been shown confirms nothing.
    assert run_day(_resume(roots, folder, "--confirm-readback", "5"), steps) == EXIT_PAUSED
    assert steps.ran("03-inspect") == 0
    assert run_day(_resume(roots, folder, "--confirm-readback", "2"), steps) == EXIT_PAUSED
    err = capsys.readouterr().err
    # Step 3: inspect's clock, before anything searches.
    assert "orbitwin inspect: mission clock: origin earliest" in err
    assert "--confirm-readback 3" in err and steps.ran("03-baseline") == 0
    assert run_day(_resume(roots, folder, "--confirm-readback", "3"), steps) == EXIT_STOP
    err = capsys.readouterr().err
    # A NOTE stops as a WARNING does.
    assert "Cv          : 2.0 m/s" in err and "NOTE: a note" in err
    assert "printed 1 WARNING or NOTE line(s)" in err and steps.ran("05-search") == 0
    accept = ("--accept-config-warnings", "--confirm-readback", "5")
    assert run_day(_resume(roots, folder, *accept), steps) == 0
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert set(state["confirmed_readbacks"]) == {"2", "3", "5"}
    assert state["runs"][1]["decisions"] == {"confirm_readback": ["5"]}


def test_a_confirmation_on_a_new_day_is_refused(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_day(_args(roots, "--confirm-readback", "2"), steps) == EXIT_STOP
    assert "a new day folder has none yet" in capsys.readouterr().err
    assert not steps.calls


def test_a_set_epoch_copy_is_refused(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    from orbitwin.design import set_pursuer_epoch
    from orbitwin.rules import interception_rules

    official = roots["tmp"] / "official.script"
    shutil.copyfile(R1, official)
    copy, _ = set_pursuer_epoch(official, interception_rules("ranking-onsite"), "SC_Human",
                                "SC_Alien", after_target_s=60.0)
    link = roots["tmp"] / "linked.script"
    link.symlink_to(copy)
    for given in (copy, link):
        assert run_day(_args(roots, script=given), steps) == EXIT_STOP
        assert "is a set-epoch copy" in capsys.readouterr().err
    assert not steps.calls


def test_rank_warnings_are_shown_and_accepted_only_once_shown(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    spread = "the delta-v term moves under 0.5 points across the archive: check the unit"
    steps.rank["warnings"] = [spread]
    steps.rank["levers"] = {"marginal": {"points_per_s_earlier": 0.0028}}
    steps.stderr["05-rank"] = "orbitwin rank: WARNING: searched under another score block\n"
    assert _day(roots, steps, "--accept-rank-warnings") == EXIT_STOP
    err = capsys.readouterr().err
    assert f"WARNING: {spread}" in err and "WARNING: searched under another score block" in err
    assert '"points_per_s_earlier": 0.0028' in err and "--accept-rank-warnings" in err
    assert steps.ran("certify-final") == 0
    folder = _folder(roots)
    assert run_day(_resume(roots, folder, "--accept-rank-warnings"), steps) == 0
    note = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert f"accepted     : 05-rank: {spread}" in note


def test_each_run_is_recorded_with_its_provenance(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A resume over a local change said "(origin/main)" (#193 review, F3)."""
    repo = _repository(roots["tmp"])
    monkeypatch.setattr(day_module, "provenance", REAL_PROVENANCE)
    monkeypatch.setattr(day_module, "ROOT", repo)
    head = _git(repo, "rev-parse", "HEAD")
    steps.exit["05-rank"] = 1
    assert _day(roots, steps, "--expect-commit", head) == EXIT_STOP
    folder = _folder(roots)
    (repo / "src" / "a.py").write_text("x = 99\n", encoding="utf-8")
    del steps.exit["05-rank"]
    assert run_day(_resume(roots, folder), steps) == EXIT_STOP
    assert "local changes: M src/a.py" in capsys.readouterr().err
    assert run_day(_resume(roots, folder, "--allow-non-main"), steps) == 0
    note = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert f"(the expected commit {head}); 02-day-config" in note
    assert ("(NOT as expected: --allow-non-main: the checkout has local changes: M src/a.py); "
            "05-rank, certify-final, 07-audit-final, the copy") in note
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert [run["provenance"]["on_main"] for run in state["runs"]] == [True, True, True,
                                                                        True, False]
    assert state["inputs"]["expect_commit"] == head


def test_a_final_root_that_is_no_folder_is_refused_first(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    roots["final"].write_text("not a folder\n", encoding="utf-8")
    assert _day(roots, steps) == EXIT_STOP
    assert "is not a folder" in capsys.readouterr().err
    assert not steps.calls


def test_a_copy_cut_short_is_left_as_partial_and_named(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import shutil

    real = shutil.copyfile

    def full(source: Path, target: Path) -> object:
        if Path(target).name == REPORT and Path(target).parent.parent == roots["final"]:
            Path(target).write_bytes(b"trunc")
            raise OSError(28, "No space left on device")
        return real(source, target)

    monkeypatch.setattr(day_module.shutil, "copyfile", full)
    assert _day(roots, steps) == EXIT_STOP
    err = capsys.readouterr().err
    (left,) = list(roots["final"].iterdir())
    assert left.name.endswith(".partial")
    assert "No space left on device" in err and f"{left} was left and is not an upload" in err
    assert not (roots["final"] / "latest").is_symlink()


def test_an_interrupt_during_the_copy_names_what_it_left(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    real = steps.__call__

    def interrupted(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        if steps.label(argv) == "07-check-copy":
            raise KeyboardInterrupt
        return real(argv, env, cwd, log)

    assert _day(roots, interrupted) == EXIT_STOP
    err = capsys.readouterr().err
    (left,) = list(roots["final"].iterdir())
    assert left.name.endswith(".partial") and f"{left} was left and is not an upload" in err


def test_a_stop_says_where_latest_points(roots: dict[str, Path],
                                         capsys: pytest.CaptureFixture[str]) -> None:
    assert _day(roots, Steps()) == 0
    earlier = roots["final"] / os.readlink(roots["final"] / "latest")
    capsys.readouterr()
    today = Steps()
    today.exit["05-search"] = 1
    assert _day(roots, today) == EXIT_STOP
    err = capsys.readouterr().err
    assert (f"{roots['final'] / 'latest'} still points at {earlier} (T_max 6711.667 s; "
            f"official {R1} (SHA-256 {_sha(R1)}); written {earlier.name})") in err


def test_a_newer_runs_latest_is_not_replaced_unless_forced(
    roots: dict[str, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """Run A (T_max typed wrong) still running; run B started later with it right."""
    first = Steps()
    real = first.__call__
    done: dict[str, str] = {}

    def a(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        if first.label(argv) == "05-rank" and "b" not in done:
            assert _day(roots, Steps()) == 0
            done["b"] = os.readlink(roots["final"] / "latest")
        return real(argv, env, cwd, log)

    args_a = _args(roots)
    args_a.horizon_s = 6000.0
    code = _confirmed(roots, run_day(args_a, a), a)
    a_folder = sorted(path for path in roots["work"].iterdir())[0]
    assert code == EXIT_STOP and os.readlink(roots["final"] / "latest") == done["b"]
    err = capsys.readouterr().err
    assert f"points at {done['b']}, delivered by another day run" in err
    assert len([path for path in roots["final"].iterdir() if not path.is_symlink()]) == 1
    assert run_day(_resume(roots, a_folder, "--force-latest"), a) == 0
    final = roots["final"] / os.readlink(roots["final"] / "latest")
    assert "T_max        : 6000.0 s" in (final / NOTE).read_text(encoding="utf-8")


def test_the_audit_runs_on_the_choice_and_stops_on_a_failure(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.exit["07-audit-final"] = 1
    steps.stdout_audit = "AUDIT: FAIL (impulse 2 over 250 m/s)\n"
    assert _day(roots, steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert "audit_burn_table.py exited 1 on final" in err and "--exclude final" in err
    _nothing_copied(roots)
    folder = _folder(roots)
    assert run_day(_resume(roots, folder, "--exclude", "final"), steps) == 0
    assert steps.ran("07-audit-final-02") == 1
    note = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "chosen       : final-02 (C-closest)" in note


def test_include_takes_back_an_exclude(roots: dict[str, Path], steps: Steps) -> None:
    steps.exit["certify-final-02"] = 2
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    # Left out; the copy then fails its check, so this folder delivers nothing yet.
    steps.step7_exit["copy"] = 2
    assert run_day(_resume(roots, folder, "--exclude", "final-02"), steps) == EXIT_STOP
    assert json.loads((folder / "state.json").read_text())["excluded"] == ["final-02"]
    del steps.exit["certify-final-02"], steps.step7_exit["copy"]
    assert run_day(_resume(roots, folder, "--include", "final-02"), steps) == 0
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    # Refused once, not again until --include asked for it.
    assert state["excluded"] == [] and steps.ran("certify-final-02") == 2
    assert "final-02" in state["checked"]


def test_a_locked_day_folder_is_not_written(roots: dict[str, Path], steps: Steps) -> None:
    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    before = (folder / "state.json").read_bytes()
    with (folder / ".day.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        assert run_day(_resume(roots, folder, "--exclude", "final"), steps) == EXIT_STOP
    assert (folder / "state.json").read_bytes() == before


def test_the_coefficient_pause_names_the_fallback(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    assert _day(roots, steps, score=False) == EXIT_PAUSED
    folder = _folder(roots)
    assert (f"if no coefficients come, the upload is final-01 "
            f"({folder / 'run' / 'final-01' / 'submission'})") in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--work-root", "--final-root"])
def test_a_root_under_the_disk_guard_is_refused(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch, flag: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    guarded = roots["tmp"] / "output"
    monkeypatch.setattr(day_module, "GUARDED", guarded)
    assert run_day(_args(roots, flag, str(guarded / "x")), steps) == EXIT_STOP
    assert "the disk guard compresses" in capsys.readouterr().err
    assert not steps.calls


@pytest.mark.parametrize(("flag", "value", "said"), [
    ("kt_per_s", -0.003, "is negative"), ("kv_per_m_s", -0.0012, "is negative"),
    ("kt_per_s", float("nan"), "is not finite"), ("cv_m_s", float("inf"), "is not finite"),
])
def test_a_coefficient_no_announcement_has_is_refused(
    roots: dict[str, Path], steps: Steps, flag: str, value: float, said: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _args(roots)
    setattr(args, flag, value)
    assert run_day(args, steps) == EXIT_STOP
    assert said in capsys.readouterr().err and not steps.calls


# -- the mutants the review found surviving ------------------------------------------------


def test_a_copy_unlike_its_source_is_refused(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import shutil

    real = shutil.copyfile

    def flipping(source: Path, target: Path) -> object:
        real(source, target)
        if Path(target).parent.parent == roots["final"] and Path(target).name == REPORT:
            Path(target).write_bytes(Path(target).read_bytes() + b"x")
        return target

    monkeypatch.setattr(day_module.shutil, "copyfile", flipping)
    assert _day(roots, steps) == EXIT_STOP
    assert "does not match what certify listed" in capsys.readouterr().err
    assert steps.ran("07-check-copy") == 0 and not (roots["final"] / "latest").is_symlink()


def test_disk_is_checked_again_before_the_copy(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real = steps.__call__
    flag = roots["tmp"] / "DISK_LOW"
    monkeypatch.setattr(day_module, "DISK_LOW", flag)

    def filling(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        done = real(argv, env, cwd, log)
        if steps.label(argv) == "07-audit-final":
            flag.touch()
        return done

    assert _day(roots, filling) == EXIT_STOP
    assert "DISK_LOW exists" in capsys.readouterr().err
    assert steps.ran("07-check-copy") == 0 and not roots["final"].exists()


def test_step_5s_search_must_be_a_new_one(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real = Steps._search

    def again(self: Steps, label: str, argv: list[str], err: str) -> tuple[str, str]:
        if label == "05-search":
            return "", err
        return real(self, label, argv, err)

    monkeypatch.setattr(Steps, "_search", again)
    assert _day(roots, steps) == EXIT_STOP
    assert "step 5's search is step 4's" in capsys.readouterr().err
    assert steps.ran("05-rank") == 0


def test_a_search_summary_without_the_verified_flag_stops_4a(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    del steps.summary["best_candidate_is_gmat_verified"]
    assert _day(roots, steps) == EXIT_STOP
    assert "verified nothing in GMAT" in capsys.readouterr().err
    assert steps.ran("certify-final-01") == 0


def test_an_excluded_final_is_never_certified(roots: dict[str, Path], steps: Steps) -> None:
    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    del steps.exit["05-rank"]
    assert run_day(_resume(roots, _folder(roots), "--exclude", "final"), steps) == 0
    assert steps.ran("certify-final") == 0
    note = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "chosen       : final-02 (C-closest)" in note


def test_an_official_copy_unlike_the_given_script_is_refused(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import shutil

    real = shutil.copyfile

    def flipping(source: Path, target: Path) -> object:
        real(source, target)
        if Path(target).name == "official.script":
            Path(target).write_bytes(b"% not the organizer's\n")
        return target

    monkeypatch.setattr(day_module.shutil, "copyfile", flipping)
    assert _day(roots, steps) == EXIT_STOP
    assert "official.script does not match" in capsys.readouterr().err
    assert not steps.calls


def test_a_confirmation_given_before_its_readback_is_shown_confirms_nothing(
    roots: dict[str, Path], steps: Steps,
) -> None:
    assert run_day(_args(roots), steps) == EXIT_PAUSED
    folder = _folder(roots)
    every = ("--confirm-readback", "2", "--confirm-readback", "3", "--confirm-readback", "5")
    assert run_day(_resume(roots, folder, *every), steps) == EXIT_PAUSED
    assert steps.ran("03-baseline") == 0 and _pending(folder) == ["3"]
    assert run_day(_resume(roots, folder, *every), steps) == EXIT_PAUSED
    assert steps.ran("05-scored-config") == 1 and steps.ran("05-search") == 0
    assert _pending(folder) == ["5"]


def test_rank_warnings_accepted_before_they_are_shown_stop_all_the_same(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.rank["warnings"] = ["The time term spans 0.010 points across the whole archive"]
    assert run_day(_args(roots), steps) == EXIT_PAUSED
    folder = _folder(roots)
    for step in ("2", "3"):
        assert run_day(_resume(roots, folder, "--confirm-readback", step), steps) == EXIT_PAUSED
    assert run_day(_resume(roots, folder, "--confirm-readback", "5",
                           "--accept-rank-warnings"), steps) == EXIT_STOP
    assert "rank printed 1 warning(s)" in capsys.readouterr().err
    assert steps.ran("certify-final") == 0


def test_the_note_beside_the_upload_cannot_be_taken_for_one() -> None:
    assert NOTE == "README_NOT_FOR_UPLOAD.md"


# -- the GitHub review of #193 -------------------------------------------------------------


def test_inspects_warnings_stop_until_read(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    warning = ("orbitwin inspect: WARNING: T_max 6711.667 s is unchecked: SC_Alien is near "
               "circular")
    steps.stderr["03-inspect"] = f"orbitwin inspect: mission clock: origin earliest\n{warning}\n"
    assert _day(roots, steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert warning in err and "--accept-config-warnings --confirm-readback 3" in err
    assert steps.ran("03-baseline") == 0
    folder = _folder(roots)
    code = run_day(_resume(roots, folder, "--accept-config-warnings", "--confirm-readback",
                           "3"), steps)
    assert _confirmed(roots, code, steps) == 0


def test_an_end_time_only_pass_is_delivered_after_the_by_eye_comparison(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    """A report with no SC_Human velocities ends certify on PRE-CHECK PASS (end time only);
    --exclude only walked down to the zero burn (#193 review, item 3)."""
    for name in CANDIDATES_ALL:
        steps.precheck[name] = "pass-end-time-only"
        steps.last_line[name] = PRECHECK_PASS_END_TIME
    steps.exit["07-bind-final"] = 3
    assert _day(roots, steps) == EXIT_STOP
    err = capsys.readouterr().err
    folder = _folder(roots)
    report = folder / "run" / "final" / "submission" / REPORT
    # Side by side: the report, step 7's reading of its last row, the certificate.
    assert ("final: PRE-CHECK PASS (end time only): the 6b.7 by-eye comparison is mandatory. "
            f"In {report}, the last row's SC_Human.ElapsedSecs must read the certificate's "
            "T_team 4301.4377 s; step 7 read it as 4301.437731 s and scored it 96.3908 "
            "(certified 96.3908). Once the report and these agree, orbitwin day --resume "
            f"{folder} --by-eye-passed final (docs/RUNBOOK.md step 6b.7)") in err
    # final-01 was certified at step 4a, before the coefficients: step 7 alone prices it.
    assert ("scored it 25.1735 (certified without coefficients: step 7's is the only price)"
            in err)
    assert steps.ran("07-check-final") == 1 and steps.ran("07-check-final-01") == 1
    by_eye = ("--by-eye-passed", "final", "--exclude", "final-01", "--exclude", "final-02")
    assert run_day(_resume(roots, folder, *by_eye), steps) == 0
    # certify's binding, run again: still exit 3, end time only.
    bind = steps.argv("07-bind-final")
    assert bind[-1] == "--accept-end-time-binding-only"
    note = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "PRE-CHECK    : PASS (END TIME ONLY, BY EYE)" in note
    assert ("by eye       : final matched RUNBOOK 6b.7, said by run 4 (--by-eye-passed final)"
            in note) and "end-time-only binding exited [3]" in note
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["by_eye_passed"]["final"]["binding_exits"] == [3]


def test_a_by_eye_pass_needs_the_binding_to_exit_3_still(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.precheck["final"] = "pass-end-time-only"
    steps.last_line["final"] = PRECHECK_PASS_END_TIME
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    steps.exit["07-bind-final"] = 2
    assert run_day(_resume(roots, folder, "--by-eye-passed", "final"), steps) == EXIT_STOP
    assert "end-time-only binding exited [2], not 3" in capsys.readouterr().err
    _nothing_copied(roots)


def test_a_by_eye_pass_is_taken_only_for_an_end_time_only_candidate(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    assert run_day(_resume(roots, _folder(roots), "--by-eye-passed", "final"),
                   steps) == EXIT_STOP
    assert "no stop of this day folder showed final ending on PRE-CHECK PASS (end time only)" \
        in capsys.readouterr().err
    assert steps.ran("certify-final") == 0


def test_a_by_eye_pass_goes_with_a_certify_run_again(roots: dict[str, Path],
                                                     steps: Steps) -> None:
    steps.precheck["final-02"] = "pass-end-time-only"
    steps.last_line["final-02"] = PRECHECK_PASS_END_TIME
    steps.exit["07-bind-final-02"] = 3
    steps.scores["final-02"] = 99.0
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    # Its copy fails, so nothing is delivered from this folder yet.
    steps.step7_exit["copy"] = 2
    assert run_day(_resume(roots, folder, "--by-eye-passed", "final-02"), steps) == EXIT_STOP
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert "final-02" in state["by_eye_passed"]
    # Certified again (here: as an interrupted certify is), its report is another, and the
    # by-eye pass goes with the old one.
    state["steps"]["certify-final-02"]["exit"] = 130
    (folder / "state.json").write_text(json.dumps(state), encoding="utf-8")
    assert run_day(_resume(roots, folder), steps) == EXIT_STOP
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert "final-02" not in state["by_eye_passed"]


def test_certifys_gui_stop_points_at_6b_and_the_staging_script(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.exit["certify-final-02"] = 2
    kept = roots["tmp"] / "elsewhere" / "final-02.refused-2"
    steps.stderr["certify-final-02"] = (
        "orbitwin: ERROR: GmatConsole returned exit status 1. final.script: GmatConsole cannot "
        "create OpenFramesInterface DefaultOrbitView. Run it in the GMAT GUI and put the report "
        "it writes through check_report_rows (docs/RUNBOOK.md step 6)\n"
        f"orbitwin: This certification is kept in {kept} ({kept}/staging/final.script), and "
        "/x is free again; a certify stopped with Ctrl-C is not moved\n"
        "orbitwin certify: PRE-CHECK PASS\n")
    assert _day(roots, steps) == EXIT_STOP
    err = capsys.readouterr().err
    # The folder certify named, not one guessed by its time.
    assert ("final-02: certify stopped on the plots and no console copy ran: the report comes "
            f"from the GMAT GUI, running {kept}/staging/final.script "
            "(docs/RUNBOOK.md step 6b)") in err
    assert "next fallback, 4a" not in err.split("final-02:", 1)[1].split("\n", 1)[0]


def test_a_search_that_verified_nothing_is_searched_again_on_resume(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.summary["best_candidate_is_gmat_verified"] = False
    assert _day(roots, steps) == EXIT_STOP
    assert "A resume searches again" in capsys.readouterr().err
    steps.summary["best_candidate_is_gmat_verified"] = True
    assert run_day(_resume(roots, _folder(roots)), steps) == EXIT_PAUSED
    assert steps.ran("04-search") == 2 and steps.ran("certify-final-01") == 1


def test_a_rank_of_another_score_block_is_searched_again_on_resume(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.rank["best_submittable_note"] = "searched under another score block"
    assert _day(roots, steps) == EXIT_STOP
    assert "A resume searches again under day-scored.yaml" in capsys.readouterr().err
    steps.rank["best_submittable_note"] = None
    assert run_day(_resume(roots, _folder(roots)), steps) == 0
    assert steps.ran("05-search") == 2 and steps.ran("05-rank") == 2


def test_an_unreadable_certification_is_a_stop_not_a_traceback(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real = Steps._final

    def broken(self: Steps, label: str, argv: list[str], err: str) -> tuple[str, str]:
        out, err = real(self, label, argv, err)
        output = Path(argv[argv.index("--output") + 1])
        if output.name == "final-01":
            body = json.loads((output / "submission" / "certification.json").read_text())
            del body["submission_script_name"]
            (output / "submission" / "certification.json").write_text(json.dumps(body))
        if output.name == "final-02":
            (output / "submission" / SCRIPT).write_text("% no ReportFile\n")
        return out, err

    monkeypatch.setattr(Steps, "_final", broken)
    assert _day(roots, steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert ("final-01: its certification could not be read: KeyError: "
            "'submission_script_name'") in err
    assert "final-02: its certification could not be read: PrecheckUnavailable:" in err
    assert "Traceback" not in err


# -- the E2E rehearsal of #193 ---------------------------------------------------------------


# A step that ignores the Ctrl-C the day passes on, so it always writes all of its lines.
_STUBBORN = ("import pathlib, signal, sys, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n"
             "pathlib.Path(sys.argv[1]).touch()\n"
             "for n in range(200):\n    print(n)\n    print(n, file=sys.stderr)\n"
             "time.sleep(0.2)\nprint('last', flush=True)\n"
             "pathlib.Path(sys.argv[1] + '.end').touch()\n")


def _ctrl_c() -> None:
    os.kill(os.getpid(), signal.SIGINT)


def _interrupted_run(tmp_path: Path) -> None:
    try:
        day_module.run_logged([sys.executable, "-c", _STUBBORN, str(tmp_path / "ignoring")],
                              dict(os.environ), tmp_path, tmp_path / "logs" / "05-search")
    except KeyboardInterrupt:
        return
    pytest.fail("the Ctrl-C was lost")


def _whole(tmp_path: Path) -> None:
    out = (tmp_path / "logs" / "05-search.out").read_text(encoding="utf-8").split()
    assert out[-1] == "last" and len(out) == 201
    assert len((tmp_path / "logs" / "05-search.err").read_text().split()) == 200
    assert not (tmp_path / day_module.RUNNING).exists() or not any(
        (tmp_path / day_module.RUNNING).iterdir())


def test_a_ctrl_c_while_the_pumps_finish_keeps_the_log_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#203: a Ctrl-C in the pump join closed the logs under the pumps; on 3.11 and 3.12
    is_alive() said False at once, so joining again did not help."""
    real_pump = day_module._pump

    def pump(*args: Any) -> None:
        # The step has written everything and the day waits on the pumps: then the Ctrl-C.
        if args[4] == "":
            while not (tmp_path / "ignoring.end").exists():
                time.sleep(0.01)
            time.sleep(0.1)
            _ctrl_c()
            time.sleep(0.2)
        real_pump(*args)

    monkeypatch.setattr(day_module, "_pump", pump)
    _interrupted_run(tmp_path)
    _whole(tmp_path)


def test_a_ctrl_c_before_the_wait_is_not_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#203: a Ctrl-C between the step's start and the wait on it escaped with the step
    unkilled, its note left and its log empty; held instead, the step would run to its end."""
    real_note = day_module._note_child
    ready = tmp_path / "ready"

    def note(*args: Any) -> Path:
        while not ready.exists():
            time.sleep(0.01)
        _ctrl_c()
        return real_note(*args)

    monkeypatch.setattr(day_module, "_note_child", note)
    program = ("import pathlib, signal, sys, time\n"
               "signal.signal(signal.SIGINT, lambda *_: (print('got SIGINT', flush=True), "
               "sys.exit(0)))\npathlib.Path(sys.argv[1]).touch()\ntime.sleep(30)\n")
    started = time.monotonic()
    try:
        day_module.run_logged([sys.executable, "-c", program, str(ready)], dict(os.environ),
                              tmp_path, tmp_path / "logs" / "05-search")
    except KeyboardInterrupt:
        pass
    else:
        pytest.fail("the Ctrl-C was lost")
    assert time.monotonic() - started < 15
    assert (tmp_path / "logs" / "05-search.out").read_text(encoding="utf-8") == "got SIGINT\n"
    assert not any((tmp_path / day_module.RUNNING).iterdir())


def test_a_ctrl_c_a_finalizer_swallowed_still_stops_the_day(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    """#203: Popen.__del__ or a WeakSet callback ate the KeyboardInterrupt, and the day ran
    on to deliver and move latest."""
    calls = []

    def swallowing(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        calls.append(log.name)
        done = steps(argv, env, cwd, log)
        if len(calls) == 1:
            try:
                _ctrl_c()
                time.sleep(1)
            except KeyboardInterrupt:
                pass
        return done

    assert run_day(_args(roots), swallowing) == EXIT_STOP
    assert len(calls) == 1
    assert "interrupted." in capsys.readouterr().err
    assert not (roots["final"] / "latest").exists()


_HELD_DOWN = ("import os, signal, sys\nfrom orbitwin import day\nday._stopping = True\n"
              "signal.signal(signal.SIGINT, day._day_ctrl_c)\nleft = [200]\n"
              "def tracer(frame, event, arg):\n"
              "    if event == 'line' and left[0] > 0:\n"
              "        left[0] -= 1\n        os.kill(os.getpid(), signal.SIGINT)\n"
              "    return tracer\n"
              "sys.settrace(tracer)\nos.kill(os.getpid(), signal.SIGINT)\n"
              "for _ in range(50):\n    day._stop_if_ctrl_c.__name__\n"
              "sys.settrace(None)\nprint('noted', day._ctrl_c_seen)\n")


def test_a_held_down_ctrl_c_does_not_hang_the_handler() -> None:
    """A Ctrl-C re-entering the day's handler while it ran deadlocked on an Event's lock."""
    done = subprocess.run([sys.executable, "-c", _HELD_DOWN], capture_output=True, text=True,
                          timeout=60, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert done.returncode == 0, done.stderr
    assert done.stdout == "noted True\n"


def test_a_day_started_with_sigint_ignored_keeps_ignoring_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A day started in the background, SIGINT ignored, is not made interruptible."""
    real_note = day_module._note_child

    def note(*args: Any) -> Path:
        _ctrl_c()
        return real_note(*args)

    monkeypatch.setattr(day_module, "_note_child", note)
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        done = day_module.run_logged([sys.executable, "-c", "print('done')"], dict(os.environ),
                                     tmp_path, tmp_path / "logs" / "05-search")
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    except KeyboardInterrupt:
        pytest.fail("a day that ignored SIGINT was interrupted")
    finally:
        signal.signal(signal.SIGINT, previous)
    assert done.returncode == 0 and done.stdout == "done\n"


def test_a_ctrl_c_once_latest_moved_says_it_moved(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#203 round 2: a Ctrl-C just after the latest swap said latest was not moved."""
    real = os.replace

    def replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        real(src, dst, *args, **kwargs)
        if Path(dst).name == "latest":
            _ctrl_c()

    monkeypatch.setattr(os, "replace", replace)
    code = _day(roots, steps)
    err = capsys.readouterr().err
    assert code == EXIT_STOP and (roots["final"] / "latest").is_symlink()
    assert "latest was not moved" not in err and "and latest points at it." in err


@pytest.mark.parametrize("where", ["paused", "stop"])
def test_a_ctrl_c_while_a_pause_or_stop_prints_does_not_escape(
    where: str, roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#203 round 2: a Ctrl-C while the PAUSED or STOP message was written lost it."""
    real = day_module._where

    def told(*args: Any) -> str:
        _ctrl_c()
        return real(*args)

    if where == "stop":
        steps.exit["02-day-config"] = 1
    monkeypatch.setattr(day_module, "_where", told)
    try:
        code = run_day(_args(roots), steps)
    except KeyboardInterrupt:
        pytest.fail(f"the Ctrl-C escaped while the {where} message was written")
    assert code == (EXIT_PAUSED if where == "paused" else EXIT_STOP)


def test_a_ctrl_c_while_the_upload_is_named_does_not_escape(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#203 round 2: a Ctrl-C after delivery ended in a traceback, the upload unnamed."""
    fired: list[int] = []

    def said(*args: Any, **kwargs: Any) -> None:
        if kwargs.get("end") == "" and not fired:
            fired.append(1)
            _ctrl_c()
        print(*args, **kwargs)

    monkeypatch.setattr(day_module, "print", said, raising=False)
    try:
        code = _day(roots, steps)
    except KeyboardInterrupt:
        pytest.fail("the Ctrl-C escaped after delivery")
    assert code == 0 and fired
    assert "upload exactly these two files" in capsys.readouterr().err


def test_a_resume_interrupted_as_it_starts_names_the_resume(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#203 round 2: a resume cut short before it took up its folder said to start over."""
    assert run_day(_args(roots), steps) == EXIT_PAUSED
    folder = _folder(roots)
    capsys.readouterr()

    def cut_short() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(day_module, "ExitStack", cut_short)
    assert run_day(_resume(roots, folder), steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert "run the day again from the start" not in err
    assert f"orbitwin day --resume {folder}" in err


def test_a_held_down_ctrl_c_still_names_the_resume(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#203: a second Ctrl-C while the stop was written ended the day with a traceback."""
    def go_on(*_args: Any) -> str:
        _ctrl_c()
        return " Go on with: the resume."

    def interrupted(*_args: Any) -> Completed:
        raise KeyboardInterrupt

    monkeypatch.setattr(day_module, "_go_on", go_on)
    try:
        code = run_day(_args(roots), interrupted)
    except KeyboardInterrupt:
        pytest.fail("the second Ctrl-C escaped run_day")
    assert code == EXIT_STOP
    assert "interrupted. Go on with: the resume." in capsys.readouterr().err


def test_a_step_echo_is_marked_and_step_7_kept_to_its_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """certify's own "the upload is these two files" is its submission, not the day's."""
    import sys

    said = "orbitwin certify: once step 7 exits 0 on the report, the upload is these files"
    program = f"import sys; print({said!r}, file=sys.stderr)"
    day_module.run_logged([sys.executable, "-c", program], dict(os.environ), tmp_path,
                          tmp_path / "logs" / "certify-final")
    assert f"  [certify-final] {said}\n" in capsys.readouterr().err
    day_module.run_logged([sys.executable, "-c", "import sys; print('usage: x', "
                           "file=sys.stderr); sys.exit(2)"], dict(os.environ), tmp_path,
                          tmp_path / "logs" / "07-check-final")
    assert capsys.readouterr().err == ""
    assert (tmp_path / "logs" / "07-check-final.err").read_text() == "usage: x\n"


def test_a_delivered_day_folder_is_not_delivered_again(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    assert _day(roots, steps) == 0
    folder = _folder(roots)
    target = roots["final"] / os.readlink(roots["final"] / "latest")
    before = (folder / "state.json").read_bytes()
    finals = sorted(roots["final"].iterdir())
    calls = len(steps.calls)
    capsys.readouterr()
    assert run_day(_resume(roots, folder, "--exclude", "final"), steps) == 0
    assert f"{folder} already delivered to {target}; nothing was changed" in \
        capsys.readouterr().err
    assert (folder / "state.json").read_bytes() == before
    assert sorted(roots["final"].iterdir()) == finals and len(steps.calls) == calls
    # Gone from under it: a stop, not a second upload.
    for path in target.iterdir():
        path.chmod(0o644)
    (target / NOTE).unlink()
    assert run_day(_resume(roots, folder), steps) == EXIT_STOP
    assert "which is gone or incomplete" in capsys.readouterr().err


@pytest.mark.parametrize(("craft", "head_start_s", "window", "said"), [
    ("SC_Human", 0.0, None, "start        : the craft start together"),
    ("SC_Alien", 1800.0, "after_alignment",
     "start        : SC_Alien 1800.0 s first; d_min window after_alignment"),
])
def test_the_note_names_the_time_origin_and_the_window(
    roots: dict[str, Path], steps: Steps, craft: str, head_start_s: float,
    window: str | None, said: str,
) -> None:
    steps.origin_craft, steps.head_start_s = craft, head_start_s
    extra = () if window is None else ("--closest-approach-window", window)
    assert _day(roots, steps, *extra) == 0
    note = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert (f"time origin  : {craft}'s epoch (mission_time_origin earliest): T_max, T_team "
            "and Ct count from it") in note
    assert said in note


def test_the_disk_guards_file_can_be_named(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A release tree's own output/DISK_LOW is not the one the workstation guard writes."""
    flag = roots["tmp"] / "guard" / "DISK_LOW"
    flag.parent.mkdir()
    flag.touch()
    monkeypatch.setenv("ORBITWIN_DISK_LOW_FILE", str(flag))
    assert run_day(_args(roots), steps) == EXIT_STOP
    assert f"{flag} exists" in capsys.readouterr().err and not steps.calls


# -- the delta review of bddea10 -------------------------------------------------------------


def test_inspects_as_announced_line_is_exempt_on_real_output(tmp_path: Path) -> None:
    """inspect prints the boundary offset alone, without the taking (#193 delta, P2-2)."""
    import sys

    from orbitwin.day import _as_announced, _inspect_readback

    env = {key: value for key, value in os.environ.items()
           if key not in {"BASH_ENV", "GMAT_CONSOLE", "GMAT_STARTUP"}}
    env["PYTHONPATH"] = str(ROOT / "src")
    config = tmp_path / "day.yaml"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "ranking_day_config.py"),
                    "--script", str(R1), "--output", str(config), "--horizon-s", "6711.667",
                    "--team-number", "54", "--mission-boundary-km", "16900",
                    "--tmax-as-announced"],
                   env=env, capture_output=True, text=True, check=True)
    done = subprocess.run([sys.executable, "-m", "orbitwin", "inspect", str(R1), "--config",
                           str(config)], env=env, capture_output=True, text=True, check=True)
    shown = _inspect_readback(done.stderr, [])["warnings"]
    assert shown == [f"orbitwin inspect: {PUTS}"]
    assert _inspect_readback(done.stderr, _as_announced(6711.667, 16900.0)[:1])["warnings"] == []
    assert _inspect_readback(done.stderr, _as_announced(6711.667, 17000.0)[:1])["warnings"] \
        == shown


def test_as_announced_goes_through_step_3_too(roots: dict[str, Path], steps: Steps) -> None:
    steps.readback["02-day-config"] = steps.readback["05-scored-config"] = [PUTS, TAKEN]
    steps.stderr["03-inspect"] = f"orbitwin inspect: {PUTS}\n"
    assert _day(roots, steps, "--tmax-as-announced", "--mission-boundary-km", "16900") == 0
    assert _day(roots, steps, "--tmax-as-announced", "--mission-boundary-km", "16901") \
        == EXIT_STOP


def test_every_printed_resume_carries_allow_non_main(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_day(_args(roots, "--allow-non-main"), steps) == EXIT_PAUSED
    folder = _folder(roots)
    err = capsys.readouterr().err
    assert f"\n\norbitwin day --resume {folder} --allow-non-main --confirm-readback 2\n\n" \
        in err
    # Taken at a resume, it is carried from then on too.
    steps.exit["certify-final-02"] = 2
    assert _day(roots, steps) == EXIT_STOP
    second = _folder(roots)
    assert "--allow-non-main" not in capsys.readouterr().err.split("STOP")[-1]
    assert run_day(_resume(roots, second, "--allow-non-main"), steps) == EXIT_STOP
    assert f"\n\norbitwin day --resume {second} --allow-non-main --exclude final-02\n\n" in \
        capsys.readouterr().err


def test_each_readback_pause_says_what_a_wrong_value_costs(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_day(_args(roots), steps) == EXIT_PAUSED
    err = capsys.readouterr().err
    assert ("If a value is wrong, start a new day folder with the corrected input;\n"
            "the steps done so far run again (about 1 min here).\norbitwin day: Day folder: "
            f"{_folder(roots)}.\n") in err
    # The command a person types next, alone on its line.
    assert f"then run:\n\norbitwin day --resume {_folder(roots)} --confirm-readback 2\n\n" in err


def test_only_rank_warnings_that_can_mean_a_wrong_input_stop(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    """Two of rank's warnings come with every archive (#193 delta, P2-4)."""
    every = ["The winner changes when ct, cv, kt, kv moves by 10%. Re-read the announced value "
             "before submitting.",
             "The top-scoring candidate stands at stage renderable (the file claimed "
             "public_gmat_verified). A score is arithmetic; only certify establishes that GMAT "
             "reproduces it."]
    steps.rank["warnings"] = every
    assert _day(roots, steps) == 0
    err = capsys.readouterr().err
    assert all(f"shown: {line}" in err for line in every)
    spread = ("The delta-v term spans 0.010 points across the whole archive: check "
              "score.delta_v_unit")
    zero = "kt is zero, so nudging it by a percentage does not move it"
    for line in (spread, zero):
        steps.rank["warnings"] = [*every, line]
        assert _day(roots, steps) == EXIT_STOP
        err = capsys.readouterr().err
        assert f"WARNING: {line}" in err and "rank printed 1 warning(s) that can mean" in err


def test_a_by_eye_binding_must_exit_3_on_every_run(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.precheck["final"] = "pass-end-time-only"
    steps.last_line["final"] = PRECHECK_PASS_END_TIME
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    steps.exit["07-bind-final"] = 0
    assert run_day(_resume(roots, folder, "--by-eye-passed", "final"), steps) == EXIT_STOP
    assert "end-time-only binding exited [0], not 3" in capsys.readouterr().err


def test_a_by_eye_pass_needs_every_binding_run_end_time_only(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    """A pre-check with one run that bound more than the end time is no by-eye case."""
    steps.precheck["final"] = "pass-end-time-only"
    steps.last_line["final"] = PRECHECK_PASS_END_TIME
    steps.extra_bind_run = True
    steps.exit["07-bind-final"] = 3
    assert _day(roots, steps) == EXIT_STOP
    assert run_day(_resume(roots, _folder(roots), "--by-eye-passed", "final"),
                   steps) == EXIT_STOP
    assert "final: its pre-check records no end-time-only binding to run again" in \
        capsys.readouterr().err
    _nothing_copied(roots)


def test_a_full_pass_wins_a_tie_with_a_by_eye_one(roots: dict[str, Path], steps: Steps) -> None:
    steps.precheck["final"] = "pass-end-time-only"
    steps.last_line["final"] = PRECHECK_PASS_END_TIME
    steps.exit["07-bind-final"] = 3
    steps.scores = {"final": 50.0, "final-01": 50.0, "final-02": 10.0}
    assert _day(roots, steps) == EXIT_STOP
    assert run_day(_resume(roots, _folder(roots), "--by-eye-passed", "final"), steps) == 0
    note = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "chosen       : final-01 (C-first)" in note


def test_a_proven_upload_a_stop_left_behind_latest_is_taken_up(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stopped between the rename and latest, it said "Nothing was copied" (#193 delta, L2)."""
    real = day_module.Day._newer_latest
    calls = []

    def interrupted(self: day_module.Day, final_root: Path) -> str | None:
        calls.append(final_root)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(self, final_root)

    monkeypatch.setattr(day_module.Day, "_newer_latest", interrupted)
    assert _day(roots, steps) == EXIT_STOP
    (target,) = [path for path in roots["final"].iterdir() if not path.is_symlink()]
    err = capsys.readouterr().err
    assert f"{target} holds this day folder's proven upload; latest was not moved" in err
    assert not (roots["final"] / "latest").exists()
    monkeypatch.setattr(day_module.Day, "_newer_latest", real)
    calls_before = len(steps.calls)
    assert run_day(_resume(roots, _folder(roots)), steps) == 0
    assert os.readlink(roots["final"] / "latest") == target.name
    assert len(steps.calls) == calls_before
    # A crash before state.json recorded it: found by its note.
    folder = _folder(roots)
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    state.pop("uploads")
    (folder / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (roots["final"] / "latest").unlink()
    assert run_day(_resume(roots, folder), steps) == 0
    assert os.readlink(roots["final"] / "latest") == target.name
    assert json.loads((folder / "state.json").read_text())["uploads"][0]["recovered"] is True


def test_the_note_says_what_each_confirmation_hashed(roots: dict[str, Path],
                                                     steps: Steps) -> None:
    assert _day(roots, steps) == 0
    note = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    for step, of in (("2", "config"), ("3", "the lines shown"), ("5", "config")):
        assert f"(--confirm-readback {step}) at " in note
        line = next(line for line in note.splitlines()
                    if line.startswith(f"readback     : step {step} "))
        assert f", SHA-256 of {of} " in line


def test_a_pass_without_the_target_in_the_report_is_delivered_and_said(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    """R3 T16: the workshop report carries no SC_Alien position; step 7 judges it."""
    for name in CANDIDATES_ALL:
        steps.precheck[name] = "pass-target-not-in-report"
        steps.last_line[name] = PRECHECK_PASS_NO_TARGET
    assert _day(roots, steps) == 0
    assert ("step 7 (final): PRE-CHECK PASS (target not in report): d_min and the interception "
            "were proven on certify's diagnostic flight") in capsys.readouterr().err
    note = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "PRE-CHECK    : PASS-TARGET-NOT-IN-REPORT\n" in note
    assert ("d_min        : 3.500000 km, and the interception, from certify's diagnostic "
            "flight of the same script: the official report carries no target position") in note
    assert "(closest report row)" not in note


def test_a_pass_without_the_target_needs_certifys_own_last_line(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.precheck["final"] = "pass-target-not-in-report"
    steps.last_line["final"] = PRECHECK_PASS
    assert _day(roots, steps) == EXIT_STOP
    assert "final: PRE-CHECK PASS (target not in report)" in capsys.readouterr().err
    _nothing_copied(roots)


def test_a_by_eye_pass_without_the_target_needs_its_binding_to_exit_7(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    """Without the target or the pursuer's velocity: the binding exits 7, never 3."""
    steps.precheck["final"] = "pass-target-not-in-report-end-time-only"
    steps.last_line["final"] = PRECHECK_PASS_NO_TARGET_END_TIME
    assert _day(roots, steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert ("final: PRE-CHECK PASS (target not in report; end time only): d_min and the "
            "interception were proven on certify's diagnostic flight, and the 6b.7 by-eye "
            "comparison is mandatory. In ") in err
    steps.exit["07-bind-final"] = 7
    assert run_day(_resume(roots, _folder(roots), "--by-eye-passed", "final"), steps) == 0
    # The step 7 run certify's pre-check recorded is not run again as a binding.
    assert steps.ran("07-bind-final") == 1
    # Review of #195, P2-2: it is not the other end-time-only pass, nor d_min off our rows.
    note = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "PRE-CHECK    : PASS-TARGET-NOT-IN-REPORT-END-TIME-ONLY (BY EYE)\n" in note
    assert "from certify's diagnostic flight of the same script" in note


def test_a_by_eye_pass_without_the_target_does_not_stand_on_a_binding_exit_3(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.precheck["final"] = "pass-target-not-in-report-end-time-only"
    steps.last_line["final"] = PRECHECK_PASS_NO_TARGET_END_TIME
    assert _day(roots, steps) == EXIT_STOP
    steps.exit["07-bind-final"] = 3
    assert run_day(_resume(roots, _folder(roots), "--by-eye-passed", "final"),
                   steps) == EXIT_STOP
    assert "end-time-only binding exited [3], not 7" in capsys.readouterr().err
    _nothing_copied(roots)


# -- R4: no answer from GMAT, a failure no gate names, a step a killed day left ------------


NO_ANSWER = ("orbitwin: ERROR: The state probe failed: GmatConsole returned exit status -11\n"
             "orbitwin certify: NO ANSWER: GmatConsole crashed, timed out or wrote no report, "
             "which is no verdict on the candidate; run certify again (exit 75)\n")


def _failing(steps: Steps, label: str, codes: list[int]) -> Any:
    """steps, except that `label` exits each of codes in turn before it passes."""
    left = list(codes)

    def run(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        if steps.label(argv) == label and left:
            steps.calls.append((label, argv, env))
            log.parent.mkdir(parents=True, exist_ok=True)
            log.with_suffix(".err").write_text(NO_ANSWER, encoding="utf-8")
            return Completed(left.pop(0), "", NO_ANSWER)
        return steps(argv, env, cwd, log)

    return run


def test_certify_with_no_answer_once_is_run_again_at_once(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    """R4 E: GmatConsole died of SIGSEGV in certify-final, which read as a refusal; the
    --exclude the stop printed delivered final-01 at 0.513 points less."""
    assert _day(roots, _failing(steps, "certify-final", [75])) == 0
    assert steps.ran("certify-final") == 2
    summary = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "chosen       : final (C-best)" in summary
    err = capsys.readouterr().err
    assert ("certify-final got no answer from GmatConsole (a crash, a timeout or no report), "
            "which is no verdict on the candidate; running it again, once") in err
    logs = _folder(roots) / "logs"
    assert "exit status -11" in (logs / "certify-final.err.interrupted").read_text()


def test_certify_with_no_answer_twice_stops_as_no_verdict_and_a_resume_runs_it(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    executor = _failing(steps, "certify-final", [75, 75])
    assert _day(roots, executor) == EXIT_STOP
    folder = _folder(roots)
    shown = capsys.readouterr().err
    # Once at once, not again.
    assert shown.count("running it again, once") == 1
    err = shown.split("STOP: ", 1)[1]
    assert steps.ran("certify-final") == 2
    assert ("final: certify gave no verdict (exit 75: no answer: GmatConsole crashed, timed "
            "out or wrote no report, or this machine failed): orbitwin: ERROR: The state probe "
            "failed: GmatConsole returned exit status -11 (docs/RUNBOOK.md step 6 (certify it "
            "again))") in err
    assert (f"That is no verdict on final. To certify it again, run:\n\n"
            f"orbitwin day --resume {folder}\n\n") in err
    # Not the refusal's way out, which lost points on R4.
    assert "--exclude applies" not in err and "--exclude <name>" not in err
    _nothing_copied(roots)
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["steps"]["certify-final"]["exit"] == 75
    # The resume it names certifies final again, and final is the upload.
    assert run_day(_resume(roots, folder), executor) == 0
    assert steps.ran("certify-final") == 3
    summary = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "chosen       : final (C-best)" in summary and "excluded     : none" in summary


@pytest.mark.parametrize("code", [1, -9, 130, None])
def test_a_certify_cut_short_is_no_verdict_and_is_not_run_again_at_once(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
    code: int | None,
) -> None:
    """Killed (the OOM killer's -9), a traceback's 1: only no answer from GMAT, 75, is run
    again at once; each is still no verdict, and a resume runs it again."""
    executor = _failing(steps, "certify-final", [code])  # type: ignore[list-item]
    assert _day(roots, executor) == EXIT_STOP
    assert steps.ran("certify-final") == 1
    err = capsys.readouterr().err
    assert f"final: certify gave no verdict (exit {code}: it did not finish)" in err
    assert "That is no verdict on final." in err
    assert run_day(_resume(roots, _folder(roots)), executor) == 0
    assert steps.ran("certify-final") == 2


def test_a_refusal_and_no_answer_each_get_their_own_command(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    steps.exit["certify-final-02"] = 2
    assert _day(roots, _failing(steps, "certify-final", [75, 75])) == EXIT_STOP
    folder = _folder(roots)
    err = capsys.readouterr().err
    assert f"run:\n\norbitwin day --resume {folder}\n\n" in err
    assert f"run:\n\norbitwin day --resume {folder} --exclude final-02\n\n" in err
    assert "--include final-02 runs its certify again instead" in err
    assert "A person decides each other one" not in err


def _full_after(steps: Steps, label: str, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """steps, and a disk that fills as `label` ends (R4 H filled it 60 s into step 4): no
    write gets a byte until the day's reserve is given up, and no free space shows."""
    full = {"on": False}
    real_write = day_module._write_new

    def write(path: Path, text: str) -> None:
        if full["on"] and (path.parent / day_module.RESERVE_NAME).exists():
            raise OSError(errno.ENOSPC, "No space left on device", str(path))
        real_write(path, text)

    real_free = day_module._free_bytes
    monkeypatch.setattr(day_module, "_write_new", write)
    monkeypatch.setattr(day_module, "_free_bytes",
                        lambda path: 0 if full["on"] else real_free(path))

    def run(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        done = steps(argv, env, cwd, log)
        if steps.label(argv) == label:
            full["on"] = True
        return done

    return run, full


def test_a_disk_that_fills_keeps_the_steps_record_and_names_the_resume(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    executor, full = _full_after(steps, "04-search", monkeypatch)
    assert _day(roots, executor) == EXIT_STOP
    folder = _folder(roots)
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert f"the disk is full; {folder / day_module.RESERVE_NAME} was given up" in err
    assert (f"0.0 GiB free under {folder}, under --min-free-gb 0.001; free space first, then "
            f"run:\n\norbitwin day --resume {folder}\n\n") in err
    # Step 4's record was written all the same, so a resume does not search again.
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["steps"]["04-search"]["exit"] == 0
    assert not [path for path in folder.iterdir() if path.name.startswith(".state.json.")]
    full["on"] = False
    assert _confirmed(roots, run_day(_resume(roots, folder), executor), executor) == 0
    assert steps.ran("04-search") == 1
    # Taken again for the next time.
    assert (folder / day_module.RESERVE_NAME).stat().st_size == day_module.RESERVE_BYTES


def test_a_failure_no_gate_names_says_what_failed_where_and_how_to_go_on(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    """R4 H: ENOSPC in _write_json ended the day in a traceback with no resume line."""
    def failing(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        if steps.label(argv) == "05-search":
            raise OSError(errno.ENOSPC, "No space left on device", str(log))
        return steps(argv, env, cwd, log)

    assert _day(roots, failing) == EXIT_STOP
    folder = _folder(roots)
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert (f"orbitwin day: FAILED while step 5 (05-search): OSError: [Errno 28] No space left "
            f"on device: '{folder / 'logs' / '05-search'}'. The disk is full: free space there "
            f"first. Finished steps are kept. Go on with:\n\norbitwin day --resume {folder}\n\n"
            f"orbitwin day: Day folder: {folder}.\norbitwin day: Nothing was copied for upload."
            ) in err

    def broken(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        if steps.label(argv) == "05-rank":
            raise RuntimeError("a bug")
        return steps(argv, env, cwd, log)

    assert run_day(_resume(roots, folder), broken) == EXIT_STOP
    err = capsys.readouterr().err
    # A bug keeps its traceback, and the same way on.
    assert "Traceback" in err and "RuntimeError: a bug" in err
    assert ("FAILED while step 5 (05-rank): RuntimeError: a bug. Finished steps are kept. Go "
            f"on with:\n\norbitwin day --resume {folder}\n\n") in err
    assert run_day(_resume(roots, folder), steps) == 0
    assert steps.ran("05-search") == 1


def test_a_failure_before_any_state_says_to_start_again(
    roots: dict[str, Path], steps: Steps, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def full(path: Path, text: str) -> None:
        raise OSError(errno.ENOSPC, "No space left on device", str(path))

    monkeypatch.setattr(day_module, "_write_new", full)
    monkeypatch.setattr(day_module, "_reserve", lambda folder: None)
    assert run_day(_args(roots), steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert "FAILED while starting: OSError" in err
    assert "No day folder was recorded: run the day again from the start." in err


def test_a_write_cut_short_leaves_no_staged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = Path.write_text

    def short(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        real(self, data[:10], *args, **kwargs)
        raise OSError(errno.ENOSPC, "No space left on device", str(self))

    monkeypatch.setattr(Path, "write_text", short)
    with pytest.raises(OSError):
        day_module._write_json(tmp_path / "state.json", {"a": 1})
    assert list(tmp_path.iterdir()) == []


def test_a_write_that_fails_otherwise_leaves_no_staged_file(tmp_path: Path) -> None:
    (tmp_path / "state.json").mkdir()
    with pytest.raises(OSError):
        day_module._write_json(tmp_path / "state.json", {"a": 1})
    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.json"]


# -- a step a killed day left running ---------------------------------------------------


def test_each_step_is_noted_while_it_runs_and_holds_the_lock(tmp_path: Path) -> None:
    import sys

    # The day writes the note once Popen returns, which the child may run ahead of.
    program = ("import os, json, time\n"
               "path = os.path.join('.running', str(os.getpid()))\n"
               "deadline = time.monotonic() + 10\n"
               "while not os.path.exists(path) and time.monotonic() < deadline:\n"
               "    time.sleep(0.01)\n"
               "note = json.load(open(path))\n"
               "print(note['step'], note['start'] is not None)\n"
               "for fd in os.listdir('/proc/self/fd'):\n"
               "    try:\n"
               "        print(os.readlink(f'/proc/self/fd/{fd}'))\n"
               "    except OSError:\n"
               "        pass\n")
    with day_module._locked(tmp_path):
        done = day_module.run_logged([sys.executable, "-c", program], dict(os.environ),
                                     tmp_path, tmp_path / "logs" / "certify-final")
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0] == "certify-final True"
    assert str(tmp_path / ".day.lock") in lines[1:]
    assert not list((tmp_path / ".running").iterdir())


def _orphan(folder: Path, hold_lock: bool) -> subprocess.Popen[bytes]:
    """What a killed day leaves: its step, still running, noted in .running."""
    import sys

    with (folder / ".day.lock").open("a") as handle:
        if hold_lock:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 pass_fds=(handle.fileno(),) if hold_lock else ())
    day_module._note_child(folder / day_module.RUNNING, child.pid, "certify-final",
                           [sys.executable, "-m", "orbitwin", "certify"])
    return child


@pytest.mark.parametrize("hold_lock", [True, False])
def test_a_resume_refuses_while_a_step_a_killed_day_left_runs(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str], hold_lock: bool,
) -> None:
    """R4 C, Dp: SIGKILL of the day left its search or certify running, and an immediate
    --resume ran a second one beside it in the same run/."""
    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    before = (folder / "state.json").read_bytes()
    child = _orphan(folder, hold_lock)
    try:
        capsys.readouterr()
        assert run_day(_resume(roots, folder), steps) == EXIT_STOP
        err = capsys.readouterr().err
        assert ("a step an earlier run of this day folder started is still running, and would "
                "run beside the resume in the same run/:\n  certify-final (started ") in err
        assert (f"its processes: {child.pid}; stop them all with: kill -TERM -- -{child.pid}"
                "\nWait for it to end") in err
        # The resume to type afterwards, alone on its line (#199 review, Low 2).
        assert f"Then run:\n\norbitwin day --resume {folder}\n" in err
        assert (folder / "state.json").read_bytes() == before
    finally:
        child.kill()
        child.wait()
    del steps.exit["05-rank"]
    assert run_day(_resume(roots, folder), steps) == 0
    # Ended, its note is gone.
    assert not list((folder / day_module.RUNNING).iterdir())


def test_a_note_of_a_pid_now_another_process_is_not_a_step(
    roots: dict[str, Path], steps: Steps,
) -> None:
    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    running = folder / day_module.RUNNING
    note = day_module._note_child(running, os.getpid(), "certify-final", ["certify"])
    body = json.loads(note.read_text(encoding="utf-8"))
    # This process, but started at another time: the pid was reused.
    body["start"] = body["start"].rsplit("/", 1)[0] + "/1"
    note.write_text(json.dumps(body), encoding="utf-8")
    del steps.exit["05-rank"]
    assert run_day(_resume(roots, folder), steps) == 0
    assert not note.exists()


# -- #199 review -----------------------------------------------------------------------


def test_a_zombie_is_not_a_step(roots: dict[str, Path], steps: Steps) -> None:
    """A step that exited and was not reaped yet, its parent the killed day's."""
    import sys
    import time

    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    # A session of its own, as a step's: the group lookup must skip the zombie too.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"],
                             start_new_session=True)
    note = day_module._note_child(folder / day_module.RUNNING, child.pid, "05-rank", ["rank"])
    deadline = time.monotonic() + 10
    while day_module._proc_start(child.pid) is not None:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    try:
        # Exited, unreaped: /proc still lists it, as a zombie.
        assert Path(f"/proc/{child.pid}").exists()
        del steps.exit["05-rank"]
        assert run_day(_resume(roots, folder), steps) == 0
        assert not note.exists()
    finally:
        child.wait()


def test_a_step_left_in_its_group_by_a_killed_leader_still_counts(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    """#199 review P1: SIGKILL of solve left its workers running; the noted step was gone."""
    import signal
    import sys
    import time

    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    worker = folder / "worker.pid"
    leader = subprocess.Popen(
        [sys.executable, "-c", "import os, sys, time\nif os.fork() == 0:\n"
         f"    open({str(worker)!r}, 'w').write(str(os.getpid()))\n    time.sleep(60)\n"
         "    sys.exit(0)\ntime.sleep(60)\n"], start_new_session=True)
    day_module._note_child(folder / day_module.RUNNING, leader.pid, "05-search",
                           ["bash", "competition_day.sh"])
    deadline = time.monotonic() + 10
    while not worker.exists() or not worker.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    orphan = int(worker.read_text())
    leader.kill()
    leader.wait()
    try:
        capsys.readouterr()
        assert run_day(_resume(roots, folder), steps) == EXIT_STOP
        err = capsys.readouterr().err
        assert "05-search (started " in err
        assert (f"its processes: {orphan}; stop them all with: kill -TERM -- -{leader.pid}"
                ) in err
    finally:
        os.killpg(leader.pid, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while day_module._proc_start(orphan) is not None:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    del steps.exit["05-rank"]
    assert run_day(_resume(roots, folder), steps) == 0


def test_a_step_ending_kills_what_it_left_in_its_group(tmp_path: Path) -> None:
    """#199 review P1: a killed solve's workers held the step's pipes and the lock, so the
    pumps never ended and the day hung."""
    import sys
    import time

    worker = tmp_path / "worker.pid"
    program = ("import os, sys, time\nif os.fork() == 0:\n"
               f"    open({str(worker)!r} + '.tmp', 'w').write(str(os.getpid()))\n"
               f"    os.replace({str(worker)!r} + '.tmp', {str(worker)!r})\n"
               "    time.sleep(600)\nwhile not os.path.exists(" + repr(str(worker)) + "):\n"
               "    time.sleep(0.01)\nprint('step done', flush=True)\nos._exit(137)\n")
    began = time.monotonic()
    with day_module._locked(tmp_path):
        done = day_module.run_logged([sys.executable, "-c", program], dict(os.environ),
                                     tmp_path, tmp_path / "logs" / "05-search")
    assert done.returncode == 137 and done.stdout == "step done\n"
    assert time.monotonic() - began < day_module.PUMP_WAIT_S
    orphan = int(worker.read_text())
    deadline = time.monotonic() + 10
    while day_module._proc_start(orphan) is not None:
        assert time.monotonic() < deadline, "the worker the step left is still running"
        time.sleep(0.05)
    # Nothing holds the lock now.
    with (tmp_path / ".day.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_a_step_that_escapes_its_group_does_not_keep_the_day_running(tmp_path: Path) -> None:
    """The pumps are daemons, and stop waiting: a process in a session of its own keeps
    the pipes, and the day still returns and exits."""
    import sys

    sleeper = tmp_path / "sleeper.pid"
    child = ("import os, subprocess, sys\n"
             "p = subprocess.Popen(['sleep', '60'], start_new_session=True)\n"
             f"open({str(sleeper)!r} + '.tmp', 'w').write(str(p.pid))\n"
             f"os.replace({str(sleeper)!r} + '.tmp', {str(sleeper)!r})\n")
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import os, sys\nfrom pathlib import Path\nfrom orbitwin import day\n"
        "day.PUMP_WAIT_S = 1.0\n"
        f"done = day.run_logged([sys.executable, '-c', {child!r}], dict(os.environ), "
        f"Path({str(tmp_path)!r}), Path({str(tmp_path / 'logs' / 'x')!r}))\n"
        "print('returned', done.returncode)\n", encoding="utf-8")
    try:
        ran = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True,
                             timeout=30, check=False,
                             env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        assert ran.returncode == 0 and ran.stdout == "returned 0\n", ran.stderr
    finally:
        if sleeper.exists():
            os.kill(int(sleeper.read_text()), 9)


def test_ctrl_c_is_passed_on_to_the_step_in_its_own_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import time

    real = os.waitid
    calls = []

    log = tmp_path / "logs" / "05-search.out"

    def waitid(*args: Any) -> Any:
        calls.append(args)
        if len(calls) == 1:
            # Ctrl-C once the step is up and listening.
            while "ready" not in (log.read_text(encoding="utf-8") if log.exists() else ""):
                time.sleep(0.02)
            raise KeyboardInterrupt
        return real(*args)

    monkeypatch.setattr(day_module.os, "waitid", waitid)
    monkeypatch.setattr(day_module, "INTERRUPT_WAIT_S", 5.0)
    # os.write, not print: SIGINT can land while print('ready') is still flushing, and a
    # print in the handler then raised "reentrant call" and wrote nothing (py3.11 CI).
    program = ("import os, signal, sys, time\n"
               "signal.signal(signal.SIGINT, lambda *_: (os.write(1, b'got SIGINT\\n'), "
               "sys.exit(0)))\nos.write(1, b'ready\\n')\ntime.sleep(30)\n")
    with pytest.raises(KeyboardInterrupt):
        day_module.run_logged([sys.executable, "-c", program], dict(os.environ), tmp_path,
                              tmp_path / "logs" / "05-search")
    out = log.read_text(encoding="utf-8")
    assert out == "ready\ngot SIGINT\n", out


def test_a_printed_resume_is_quoted_to_paste(roots: dict[str, Path], steps: Steps,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    """#199 review Low 1: a work root with a space printed a resume argparse refused."""
    import shlex

    roots["work"] = roots["tmp"] / "work root"
    assert run_day(_args(roots), steps) == EXIT_PAUSED
    folder = _folder(roots)
    err = capsys.readouterr().err
    line = next(line for line in err.splitlines() if line.startswith("orbitwin day --resume"))
    assert line == f"orbitwin day --resume {shlex.quote(str(folder))} --confirm-readback 2"
    argv = shlex.split(line)[1:]
    assert _parser().parse_args([*argv, "--min-free-gb", "0.001"]).resume == str(folder)


def test_a_step_that_fails_names_the_resume(roots: dict[str, Path], steps: Steps,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    steps.exit["05-rank"] = -9
    assert _day(roots, steps) == EXIT_STOP
    assert "step 05-rank exited -9: " in capsys.readouterr().err
    folder = _folder(roots)
    assert run_day(_resume(roots, folder), steps) == EXIT_STOP
    assert (f"a resume runs it again:\n\norbitwin day --resume {folder}\n"
            in capsys.readouterr().err)


def test_every_candidate_without_an_answer_is_named_to_exclude(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    executor = _failing(steps, "certify-final", [75] * 2)
    both = _failing(steps, "certify-final-02", [75] * 8)

    def run(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        return (executor if steps.label(argv) == "certify-final" else both)(argv, env, cwd, log)

    assert _day(roots, run) == EXIT_STOP
    err = capsys.readouterr().err
    assert "That is no verdict on final, final-02." in err
    assert ("a person decides; --exclude final --exclude final-02 then leaves them out, or "
            "name only some.") in err


def test_no_answer_that_quotes_the_plots_stop_is_not_the_gui_path(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    """#199 review P2a: a console copy this machine could not write carries the plots stop's
    words; exit 75 says it is no answer."""
    from orbitwin.cli import GUI_STOP

    executor = _failing(steps, "certify-final", [75, 75])
    steps.stderr["certify-final"] = f"orbitwin: ERROR: ... {GUI_STOP} ...\n"
    real = executor

    def said(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        done = real(argv, env, cwd, log)
        if steps.label(argv) == "certify-final" and done.returncode == 75:
            return Completed(75, "", f"orbitwin: ERROR: no space. {GUI_STOP}\n" + NO_ANSWER)
        return done

    assert _day(roots, said) == EXIT_STOP
    err = capsys.readouterr().err
    assert "That is no verdict on final." in err and "(docs/RUNBOOK.md step 6b)" not in err


def test_a_group_noted_under_another_boot_is_not_a_step(
    roots: dict[str, Path], steps: Steps,
) -> None:
    """After a reboot the pids, and so the groups, are other processes'."""
    import sys

    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             start_new_session=True)
    try:
        note = day_module._note_child(folder / day_module.RUNNING, other.pid, "05-search",
                                      ["search"])
        body = json.loads(note.read_text(encoding="utf-8"))
        body["start"] = "another-boot/" + body["start"].rsplit("/", 1)[1]
        note.write_text(json.dumps(body), encoding="utf-8")
        del steps.exit["05-rank"]
        assert run_day(_resume(roots, folder), steps) == 0
        assert not note.exists()
    finally:
        other.kill()
        other.wait()


def test_a_log_that_cannot_be_written_does_not_stop_the_reading() -> None:
    """A full disk under the logs: the step's output is still read, so it never blocks."""
    import io

    class Full(io.StringIO):
        def write(self, text: str) -> int:
            raise OSError(errno.ENOSPC, "No space left on device")

    kept: list[str] = []
    shown: list[str] = []
    done = threading.Event()
    day_module._pump(io.StringIO("a\nb\nc\n"), Full(), kept, shown.append, "> ", done)
    assert kept == ["a\n", "b\n", "c\n"] and shown == ["> a\n", "> b\n", "> c\n"]
    assert done.is_set()


def test_a_terminal_that_stops_reading_costs_nothing_of_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#199 review round 2: Ctrl-S or a stalled ssh held the stderr pump on the terminal;
    the log and the text the day reads certify's verdict from kept 1 line of 52."""
    import sys
    import time

    read, write = os.pipe()
    os.set_blocking(write, False)
    with suppress_full():
        while True:
            os.write(write, b"." * 4096)
    os.set_blocking(write, True)
    terminal = os.fdopen(write, "w", buffering=1)
    monkeypatch.setattr(sys, "stderr", terminal)
    monkeypatch.setattr(day_module, "ECHO_WAIT_S", 0.5)
    step = ("import sys\n"
            "for i in range(50): print(f'progress {i:05d} ' + 'x' * 60, file=sys.stderr)\n"
            "print('orbitwin: ERROR: the verdict line', file=sys.stderr)\n"
            "print('orbitwin certify: NO ANSWER: last line (exit 75)', file=sys.stderr)\n"
            "sys.exit(75)\n")
    began = time.monotonic()
    try:
        done = day_module.run_logged([sys.executable, "-c", step], dict(os.environ), tmp_path,
                                     tmp_path / "logs" / "certify-final")
        took = time.monotonic() - began
    finally:
        # Let the echo's thread go: the terminal reads again.
        os.set_blocking(read, False)
        with suppress_full():
            while os.read(read, 1 << 16):
                pass
    lines = done.stderr.splitlines()
    assert done.returncode == 75 and len(lines) == 52
    assert lines[-2:] == ["orbitwin: ERROR: the verdict line",
                          "orbitwin certify: NO ANSWER: last line (exit 75)"]
    assert (tmp_path / "logs" / "certify-final.err").read_text(encoding="utf-8") == done.stderr
    # Neither the pumps' 10 s nor the terminal held it.
    assert took < day_module.PUMP_WAIT_S


def suppress_full() -> Any:
    import contextlib

    return contextlib.suppress(BlockingIOError)


def test_a_note_whose_pid_is_another_session_leader_now_is_stale(
    roots: dict[str, Path], steps: Steps,
) -> None:
    """#199 review round 2: a reused pid that leads a session of its own was counted as the
    step's group, and the kill printed would have stopped it."""
    import sys

    steps.exit["05-rank"] = 1
    assert _day(roots, steps) == EXIT_STOP
    folder = _folder(roots)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             start_new_session=True)
    try:
        note = day_module._note_child(folder / day_module.RUNNING, other.pid, "05-search",
                                      ["search"])
        body = json.loads(note.read_text(encoding="utf-8"))
        # This boot, but another start: the step that had the pid is gone.
        body["start"] = body["start"].rsplit("/", 1)[0] + "/1"
        note.write_text(json.dumps(body), encoding="utf-8")
        del steps.exit["05-rank"]
        assert run_day(_resume(roots, folder), steps) == 0
        assert not note.exists()
        assert other.poll() is None
    finally:
        other.kill()
        other.wait()


def test_a_steps_lines_reach_the_terminal_before_the_day_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The echo has its own thread; a terminal that reads, slowly, still gets every line
    before run_logged returns, so the day's next lines come after them."""
    import io
    import sys
    import time

    class Slow(io.StringIO):
        def write(self, text: str) -> int:
            time.sleep(0.01)
            return super().write(text)

    terminal = Slow()
    monkeypatch.setattr(sys, "stderr", terminal)
    step = "import sys\nfor i in range(40): print(i, file=sys.stderr)\n"
    day_module.run_logged([sys.executable, "-c", step], dict(os.environ), tmp_path,
                          tmp_path / "logs" / "05-search")
    shown = terminal.getvalue().splitlines()
    assert shown == [f"  [05-search] {i}" for i in range(40)]


def test_a_day_that_ends_on_a_terminal_still_stalled_exits_cleanly(tmp_path: Path) -> None:
    """The echo's thread, held in a write to the terminal at exit, must hold no lock the
    interpreter needs: written through sys.stderr it aborted with a fatal error, 134."""
    import sys

    driver = tmp_path / "driver.py"
    driver.write_text(
        "import os, sys\nfrom pathlib import Path\nfrom orbitwin import day\n"
        "read, write = os.pipe()\nos.set_blocking(write, False)\n"
        "try:\n    while True:\n        os.write(write, b'.' * 4096)\n"
        "except BlockingIOError:\n    pass\nos.set_blocking(write, True)\n"
        "sys.stderr = os.fdopen(write, 'w', buffering=1)\nday.ECHO_WAIT_S = 0.2\n"
        "step = \"import sys\\nfor i in range(52): print(i, file=sys.stderr)\\n\"\n"
        f"done = day.run_logged([sys.executable, '-c', step], dict(os.environ), "
        f"Path({str(tmp_path)!r}), Path({str(tmp_path / 'logs' / 'x')!r}))\n"
        "print(len(done.stderr.splitlines()), file=sys.__stdout__, flush=True)\n",
        encoding="utf-8")
    ran = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True,
                         timeout=60, check=False,
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert ran.returncode == 0 and ran.stdout == "52\n", (ran.returncode, ran.stderr[-2000:])


def test_head_start_burns_are_turned_off_on_the_day_and_recorded(
    roots: dict[str, Path], steps: Steps,
) -> None:
    assert _day(roots, steps, "--no-head-start-burns") == 0
    first = steps.argv("02-day-config")
    assert "--no-head-start-burns" in first
    # Step 5 takes it from step 2's config.
    assert "--no-head-start-burns" not in steps.argv("05-scored-config")
    summary = (roots["final"] / "latest" / NOTE).read_text(encoding="utf-8")
    assert "head start   : closed: turned off on the day (--no-head-start-burns)" in summary
    folder = _folder(roots)
    assert run_day(_resume(roots, folder, "--no-head-start-burns"), steps) == EXIT_STOP


def test_the_team_number_goes_to_step_2_only(roots: dict[str, Path], steps: Steps) -> None:
    assert _day(roots, steps, "--team-number", "54") == 0
    first = steps.argv("02-day-config")
    assert first[first.index("--team-number") + 1] == "54"
    # Step 5 takes it from step 2's config.
    assert "--team-number" not in steps.argv("05-scored-config")


def test_a_day_without_a_team_number_stops_before_its_folder(
    roots: dict[str, Path], steps: Steps, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A resume cannot add --team-number, so a day folder made without one would be lost."""
    import yaml

    assert run_day(_args(roots, team=None), steps) == EXIT_STOP
    assert "sets no policy.team_number: pass --team-number" in capsys.readouterr().err
    assert not steps.calls and not roots["work"].exists()
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["team_number"] = 54
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert _confirmed(roots, run_day(_args(roots, "--base", str(base), team=None), steps),
                      steps) == 0
    state = json.loads((_folder(roots) / "state.json").read_text(encoding="utf-8"))
    assert state["inputs"]["team_number"] is None
    assert "--team-number" not in steps.argv("02-day-config")


def test_a_day_whose_base_disagrees_or_will_not_load_stops_before_its_folder(
    roots: dict[str, Path], steps: Steps, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    import yaml

    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["team_number"] = 7
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert run_day(_args(roots, "--base", str(base)), steps) == EXIT_STOP
    assert "--team-number 54, but" in capsys.readouterr().err
    broken = tmp_path / "broken.yaml"
    broken.write_text("policy: [\n", encoding="utf-8")
    assert run_day(_args(roots, "--base", str(broken)), steps) == EXIT_STOP
    assert f"--base {broken}" in capsys.readouterr().err
    assert not steps.calls and not roots["work"].exists()


def test_the_team_number_is_recorded_with_the_day(roots: dict[str, Path], steps: Steps) -> None:
    assert _day(roots, steps) == 0
    state = json.loads((_folder(roots) / "state.json").read_text(encoding="utf-8"))
    assert state["inputs"]["team_number"] == 54


def test_a_team_number_that_is_not_one_is_a_usage_error(roots: dict[str, Path],
                                                         steps: Steps) -> None:
    for bad in ("0", "x", "5.5", "5_4", " 54", "\uff15\uff14"):
        with pytest.raises(SystemExit):
            _day(roots, steps, "--team-number", bad)
    assert not steps.calls


def test_the_note_says_how_long_before_the_later_epoch_a_head_start_burn_fires() -> None:
    from types import SimpleNamespace

    from orbitwin.day import Checked, Day

    day = SimpleNamespace(inputs={}, state={"final_head_start": {"gain_points": 1.0833}})
    chosen = Checked("final", Path("c.json"), 0, 95.0491, "pass", True)
    line = Day.head_start_note(
        day, chosen,  # type: ignore[arg-type]
        {"burns": [{"time_s": -1896.0529827}, {"time_s": 507.68}]},
        {"head_start_s": 3600.000000104774, "time_origin_craft": "SC_Human"},
    )
    assert line == ("open, 3600.000 s (policy.head_start_burns); the upload burns in it: "
                    "1896.05 s before SC_Alien's epoch (mission clock 1703.95 s); rank's "
                    "head-start margin +1.0833 points")


def test_the_day_runs_the_commands_the_package_carries(roots: dict[str, Path],
                                                       steps: Steps) -> None:
    """Not scripts/ beside a checkout: an installed package has none."""
    assert _day(roots, steps) == 0
    tools = [argv for argv in steps.raw if argv[1:2] != ["-m"] or argv[2] != "orbitwin"]
    assert tools
    for argv in tools:
        if argv[0] == "bash":
            assert argv[1] == str(day_module.TOOLS / "competition_day.sh")
        else:
            assert argv[:2] == [sys.executable, "-m"]
            assert argv[2] in {"orbitwin.tools.ranking_day_config",
                               "orbitwin.tools.check_report_rows",
                               "orbitwin.tools.audit_burn_table"}, argv
    assert {argv[2] for argv in tools if argv[0] != "bash"} == {
        "orbitwin.tools.ranking_day_config", "orbitwin.tools.check_report_rows",
        "orbitwin.tools.audit_burn_table"}


def test_no_step_puts_the_day_folder_first_on_its_path(roots: dict[str, Path],
                                                       steps: Steps) -> None:
    """Each step runs in the day folder; a module planted there must not be imported."""
    assert _day(roots, steps) == 0
    assert steps.calls
    assert all(env.get("PYTHONSAFEPATH") == "1" for _, _, env in steps.calls)


def test_a_bytecode_folder_that_is_a_link_stops_the_day(roots: dict[str, Path], steps: Steps,
                                                        capsys: pytest.CaptureFixture[str]
                                                        ) -> None:
    assert run_day(_args(roots), steps) == EXIT_PAUSED
    folder = _folder(roots)
    victim = roots["tmp"] / "victim"
    victim.mkdir()
    (victim / "keep").write_text("x", encoding="utf-8")
    shutil.rmtree(folder / ".pycache")
    (folder / ".pycache").symlink_to(victim)
    capsys.readouterr()
    assert run_day(_resume(roots, folder, "--confirm-readback", "2"), steps) == EXIT_STOP
    assert "not the folder the day keeps its bytecode in" in capsys.readouterr().err
    assert (victim / "keep").is_file()


# -- the announcement entered twice -------------------------------------------------------


def _entries(roots: dict[str, Path], score: bool = True, **mission: str) -> list[str]:
    from test_announcement import SCORE as ENTRY_SCORE
    from test_announcement import _entry

    folder = roots["tmp"] / "entries"
    folder.mkdir(exist_ok=True)
    stamp = len(list(folder.iterdir()))
    timing = {"T_max": "09 Nov 2026 03:51:51.667 UTC", **mission}
    return [str(_entry(folder / f"{who}-{stamp}.toml", who, mission=timing,
                       score=ENTRY_SCORE if score else None)) for who in ("Alice", "Bob")]


def _announced(roots: dict[str, Path], entries: list[str], *extra: str) -> argparse.Namespace:
    return _parser().parse_args([
        "day", str(R1), "--announcement", entries[0], "--announcement", entries[1],
        "--gmat-console", str(roots["console"]), "--startup-file", str(roots["startup"]),
        "--work-root", str(roots["work"]), "--final-root", str(roots["final"]),
        "--min-free-gb", "0.001", "--team-number", "54", *extra])


def test_the_day_takes_t_max_and_the_score_from_two_entries(
    roots: dict[str, Path], steps: Steps
) -> None:
    code = _confirmed(roots, run_day(_announced(roots, _entries(roots)), steps), steps)
    assert code == 0
    folder = _folder(roots)
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    inputs = state["inputs"]
    assert inputs["horizon_s"] == 6711.667 and inputs["mission_boundary_km"] is None
    assert inputs["score"] == {"kt_per_s": 0.003, "ct_s": 5369.334, "kv_per_m_s": 0.0012,
                               "cv_m_s": 2000.0, "cr_km": 300.0}
    copies = ["input/announcement-1.toml", "input/announcement-2.toml"]
    assert inputs["announcement"] == inputs["score_announcement"] == copies
    for rel in copies:
        assert not os.access(folder / rel, os.W_OK)
        assert state["announcement_files"][rel] == _sha(folder / rel)
    step2 = steps.argv("02-day-config")
    assert "--horizon-s" not in step2 and "--mission-boundary-km" not in step2
    assert [step2[i + 1] for i, arg in enumerate(step2) if arg == "--announcement"] == [
        str(folder / rel) for rel in copies]
    step5 = steps.argv("05-scored-config")
    assert "--kt-per-s" not in step5 and "--horizon-s" not in step5
    assert step5.count("--announcement") == 2 and "--day-config" in step5


@pytest.mark.parametrize(("extra", "says"), [
    (["--horizon-s", "6711.667"], "drop --horizon-s"),
    (["--mission-boundary-km", "17000"], "drop --horizon-s, --mission-boundary-km"),
    (SCORE, "and the score flags"),
])
def test_the_entries_take_the_place_of_the_typed_values(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str],
    extra: list[str], says: str
) -> None:
    assert run_day(_announced(roots, _entries(roots), *extra), steps) == EXIT_STOP
    assert says in capsys.readouterr().err and not steps.calls


def test_entries_that_differ_stop_before_a_day_folder_is_made(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    from test_announcement import SCORE as ENTRY_SCORE
    from test_announcement import _entry

    alice = _entry(roots["tmp"] / "a.toml", "Alice")
    bob = _entry(roots["tmp"] / "b.toml", "Bob", score={**ENTRY_SCORE, "kt": "0.0095 1/s"})
    assert run_day(_announced(roots, [str(alice), str(bob)]), steps) == EXIT_STOP
    assert "score.kt: '3e-3 1/s' by Alice, '0.0095 1/s' by Bob" in capsys.readouterr().err
    assert not roots["work"].exists() or not list(roots["work"].iterdir())


def test_one_entry_is_not_enough(roots: dict[str, Path], steps: Steps,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    entries = _entries(roots)
    args = _announced(roots, entries)
    args.announcement = entries[:1]
    assert run_day(args, steps) == EXIT_STOP
    assert "given twice" in capsys.readouterr().err


def test_a_score_announced_later_comes_in_two_entries_on_the_resume(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    _confirmed(roots, run_day(_announced(roots, _entries(roots, score=False)), steps), steps)
    folder = _folder(roots)
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["inputs"]["score"] is None and state["inputs"]["score_announcement"] is None
    err = capsys.readouterr().err
    assert "--announcement <entry> --announcement <entry>" in err and "--kt-per-s" not in err
    # Typed flags on a day that took T_max from entries.
    assert run_day(_resume(roots, folder, *SCORE), steps) == EXIT_STOP
    assert "give the score the same way" in capsys.readouterr().err
    # Entries that put T_max elsewhere.
    later = _entries(roots, T_max="09 Nov 2026 03:51:52.667 UTC")
    assert run_day(_resume(roots, folder, "--announcement", later[0], "--announcement",
                           later[1]), steps) == EXIT_STOP
    assert "put T_max at 6712.667 and" in capsys.readouterr().err
    bounded = _entries(roots, boundary="17000 km")
    assert run_day(_resume(roots, folder, "--announcement", bounded[0], "--announcement",
                           bounded[1]), steps) == EXIT_STOP
    assert "put the boundary at 17000.0 and" in capsys.readouterr().err
    unscored = _entries(roots, score=False)
    assert run_day(_resume(roots, folder, "--announcement", unscored[0], "--announcement",
                           unscored[1]), steps) == EXIT_STOP
    assert "give no [score]" in capsys.readouterr().err
    both = _entries(roots)
    assert run_day(_resume(roots, folder, "--announcement", both[0], "--announcement",
                           both[1], *SCORE), steps) == EXIT_STOP
    assert "not both" in capsys.readouterr().err
    one = _resume(roots, folder, "--announcement", both[0])
    assert run_day(one, steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert "given twice" in err and "RUNBOOK.md step 5" in err
    # A copy an interrupted resume left behind, read-only and unrecorded.
    leftover = folder / "input" / "score-announcement-1.toml"
    leftover.write_text("half", encoding="utf-8")
    leftover.chmod(0o444)
    scored = _entries(roots)
    code = run_day(_resume(roots, folder, "--announcement", scored[0], "--announcement",
                           scored[1]), steps)
    assert _confirmed(roots, code, steps) == 0
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    rels = ["input/score-announcement-1.toml", "input/score-announcement-2.toml"]
    assert state["inputs"]["score_announcement"] == rels
    assert state["inputs"]["score"]["kt_per_s"] == 0.003
    assert set(rels) <= set(state["announcement_files"])
    assert state["announcement_files"][rels[0]] == _sha(folder / rels[0])
    step5 = steps.argv("05-scored-config")
    assert [step5[i + 1] for i, arg in enumerate(step5) if arg == "--announcement"] == [
        str(folder / rel) for rel in rels]


def test_an_entry_changed_in_the_day_folder_stops_the_resume(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    _confirmed(roots, run_day(_announced(roots, _entries(roots, score=False)), steps), steps)
    folder = _folder(roots)
    copy = folder / "input" / "announcement-1.toml"
    copy.chmod(0o644)
    text = copy.read_text(encoding="utf-8")
    assert "03:51:51.667" in text
    copy.write_text(text.replace("03:51:51.667", "03:51:52.667"), encoding="utf-8")
    capsys.readouterr()
    assert run_day(_resume(roots, folder), steps) == EXIT_STOP
    assert "is not the copy step 1 made" in capsys.readouterr().err


def test_an_announced_boundary_reaches_step_2_through_the_entries(
    roots: dict[str, Path], steps: Steps
) -> None:
    _confirmed(roots, run_day(_announced(roots, _entries(roots, boundary="17000 km")), steps),
               steps)
    state = json.loads((_folder(roots) / "state.json").read_text(encoding="utf-8"))
    assert state["inputs"]["mission_boundary_km"] == 17000.0
    step2 = steps.argv("02-day-config")
    assert "--mission-boundary-km" not in step2 and step2.count("--announcement") == 2


def test_a_day_begun_with_numbers_takes_no_entries_on_the_resume(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    _day(roots, steps, score=False)
    folder = _folder(roots)
    entries = _entries(roots)
    capsys.readouterr()
    assert run_day(_resume(roots, folder, "--announcement", entries[0], "--announcement",
                           entries[1]), steps) == EXIT_STOP
    assert "took T_max as a number" in capsys.readouterr().err


def test_entries_given_again_once_the_score_is_recorded_copy_nothing(
    roots: dict[str, Path], steps: Steps
) -> None:
    entries = _entries(roots)
    assert run_day(_announced(roots, entries), steps) == EXIT_PAUSED
    folder = _folder(roots)
    copied = sorted(p.name for p in (folder / "input").iterdir())
    again = _resume(roots, folder, "--announcement", entries[0], "--announcement", entries[1],
                    "--confirm-readback", "2")
    assert run_day(again, steps) == EXIT_PAUSED
    assert sorted(p.name for p in (folder / "input").iterdir()) == copied
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["inputs"]["score_announcement"] == ["input/announcement-1.toml",
                                                     "input/announcement-2.toml"]


# -- the max review of 2026-11-13 -------------------------------------------------------


def _alien_entries(roots: dict[str, Path], name: str, score: bool = True,
                   **alien: str) -> list[str]:
    from test_announcement import ALIEN, _entry
    from test_announcement import SCORE as ENTRY_SCORE

    return [str(_entry(roots["tmp"] / f"{name}-{who}.toml", who, alien={**ALIEN, **alien},
                       score=ENTRY_SCORE if score else None)) for who in ("Alice", "Bob")]


def test_entries_the_script_contradicts_stop_before_a_day_folder(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    wrong = _alien_entries(roots, "wrong", TA="241.312 deg")
    assert run_day(_announced(roots, wrong), steps) == EXIT_STOP
    assert "the script is not what was announced: SC_Alien.TA" in capsys.readouterr().err
    assert not roots["work"].exists() or not list(roots["work"].iterdir())


def test_score_entries_the_script_contradicts_are_not_recorded(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    _confirmed(roots, run_day(_announced(roots, _entries(roots, score=False)), steps), steps)
    folder = _folder(roots)
    wrong = _alien_entries(roots, "wrong", TA="241.312 deg")
    capsys.readouterr()
    assert run_day(_resume(roots, folder, "--announcement", wrong[0], "--announcement",
                           wrong[1]), steps) == EXIT_STOP
    err = capsys.readouterr().err
    assert "the script is not what was announced" in err and "RUNBOOK.md step 5" in err
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["inputs"]["score"] is None and not list((folder / "input").glob("score-*"))
    # The organizer answers the script is right: entries of the score alone go on.
    alone = []
    for who in ("Alice", "Bob"):
        path = roots["tmp"] / f"alone-{who}.toml"
        path.write_text('format = "orbitwin-announcement/1"\n'
                        f'entered_by = "{who}"\n[score]\nC_r = "300 km"\nkt = "3e-3 1/s"\n'
                        'Ct = "5369.334 s"\nkv = "0.0012 1/(m/s)"\nCv = "2000 m/s"\n',
                        encoding="utf-8")
        alone.append(str(path))
    code = run_day(_resume(roots, folder, "--announcement", alone[0], "--announcement",
                           alone[1]), steps)
    assert _confirmed(roots, code, steps) == 0
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["inputs"]["score"]["cr_km"] == 300.0


def test_some_score_flags_with_entries_say_drop_them(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_day(_announced(roots, _entries(roots), "--kt-per-s", "0.003"), steps) == EXIT_STOP
    assert "drop --horizon-s, --mission-boundary-km and the score flags" in (
        capsys.readouterr().err)


def test_the_day_folders_own_copy_is_not_an_entry(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    _confirmed(roots, run_day(_announced(roots, _entries(roots, score=False)), steps), steps)
    folder = _folder(roots)
    inside = folder / "input" / "score-announcement-1.toml"
    shutil.copyfile(_entries(roots)[0], inside)
    capsys.readouterr()
    assert run_day(_resume(roots, folder, "--announcement", str(inside), "--announcement",
                           _entries(roots)[1]), steps) == EXIT_STOP
    assert "name the entry where it was typed" in capsys.readouterr().err


def test_a_companion_named_like_an_entry_copy_stops_the_day(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    script = roots["tmp"] / "official" / "official.script"
    script.parent.mkdir()
    shutil.copyfile(R1, script)
    (script.parent / "announcement-1.toml").write_text("a tide file", encoding="utf-8")
    args = _announced(roots, _entries(roots), "--companion", "announcement-1.toml")
    args.script = str(script)
    assert run_day(args, steps) == EXIT_STOP
    assert "a file the day folder records already" in capsys.readouterr().err


def test_a_new_day_needs_t_max_from_its_entries(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    alone = []
    for who in ("Alice", "Bob"):
        path = roots["tmp"] / f"alone-{who}.toml"
        path.write_text('format = "orbitwin-announcement/1"\n'
                        f'entered_by = "{who}"\n[score]\nC_r = "300 km"\nkt = "3e-3 1/s"\n'
                        'Ct = "5369.334 s"\nkv = "0.0012 1/(m/s)"\nCv = "2000 m/s"\n',
                        encoding="utf-8")
        alone.append(str(path))
    assert run_day(_announced(roots, alone), steps) == EXIT_STOP
    assert "the entries give no T_max" in capsys.readouterr().err
    assert not roots["work"].exists() or not list(roots["work"].iterdir())


def test_a_companion_named_like_a_score_copy_stops_the_resume(
    roots: dict[str, Path], steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    script = roots["tmp"] / "official" / "official.script"
    script.parent.mkdir()
    shutil.copyfile(R1, script)
    # The second copy's name is taken: the first must not be copied either.
    (script.parent / "score-announcement-2.toml").write_text("a tide file", encoding="utf-8")
    args = _announced(roots, _entries(roots, score=False), "--companion",
                      "score-announcement-2.toml")
    args.script = str(script)
    _confirmed(roots, run_day(args, steps), steps)
    folder = _folder(roots)
    entries = _entries(roots)
    capsys.readouterr()
    assert run_day(_resume(roots, folder, "--announcement", entries[0], "--announcement",
                           entries[1]), steps) == EXIT_STOP
    assert "a file the day folder records already" in capsys.readouterr().err
    assert (folder / "input" / "score-announcement-2.toml").read_text(encoding="utf-8") == (
        "a tide file")
    assert not (folder / "input" / "score-announcement-1.toml").exists()
