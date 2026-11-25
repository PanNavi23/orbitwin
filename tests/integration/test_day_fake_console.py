"""orbitwin day running the RUNBOOK's real commands on the fake console.

Every step runs and ends 0, and the day still stops before the copy: certify's
pre-check is SKIPPED on the test double, which is never the day's console.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest
import yaml
from conftest import quick_config, startup_for_console_elsewhere

from orbitwin import day as day_module
from orbitwin.cli import main
from orbitwin.day import EXIT_PAUSED, EXIT_STOP

ROOT = Path(__file__).resolve().parents[2]
RANKING = ROOT / "examples" / "ranking"


@pytest.fixture(autouse=True)
def _no_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """provenance fetches origin; test_day_command covers it."""
    monkeypatch.setattr(day_module, "provenance", lambda root, expect, allow: {
        "source": "git", "commit": "c0ffee", "on_main": False, "notes": ["test"]})


#: The search values quick_config sets, laid over config/ranking.yaml's.
QUICK_SEARCH_KEYS = ("one_burn_samples", "one_burn_grid_burns", "one_burn_grid_arrivals",
                     "max_revolutions", "keep_candidates", "render_top", "event_step_s",
                     "event_refine_top", "two_burn_enabled", "two_burn_samples")


def _quick_base(tmp_path: Path) -> Path:
    """config/ranking.yaml with a search that takes seconds."""
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    quick = dataclasses.asdict(quick_config().search)
    raw["search"].update({key: quick[key] for key in QUICK_SEARCH_KEYS}, ranking_search=False)
    raw["gmat"].update(correct_top=2, verify_misses=1, workers=2)
    raw["policy"]["team_number"] = 54
    path = tmp_path / "quick-ranking.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _with_local_spad(script: str, tmp_path: Path) -> Path:
    """The template reading its SPAD file from its own folder, which the fake console finds."""
    folder = tmp_path / "given"
    folder.mkdir()
    text = (RANKING / script).read_text(encoding="utf-8")
    assert "'../data/vehicle/spad/SphericalModel.spo'" in text
    (folder / script).write_text(text.replace("'../data/vehicle/spad/SphericalModel.spo'",
                                              "'SphericalModel.spo'"), encoding="utf-8")
    (folder / "SphericalModel.spo").write_text("stand-in\n", encoding="utf-8")
    return folder / script


def _day(tmp_path: Path, script: Path, horizon_s: str, fake_gmat: Path, fake_startup: Path,
         *extra: str) -> int:
    return main(["day", str(script), "--companion", "SphericalModel.spo",
                 "--horizon-s", horizon_s, "--kt-per-s", "0.003", "--ct-s", "5369.3336",
                 "--kv-per-m-s", "0.0012", "--cv-m-s", "2000", "--cr-km", "300",
                 "--base", str(_quick_base(tmp_path)), "--gmat-console", str(fake_gmat),
                 "--startup-file", str(fake_startup), "--work-root", str(tmp_path / "days"),
                 "--final-root", str(tmp_path / "final"), "--min-free-gb", "0.001",
                 *extra])


def test_every_runbook_command_runs_and_the_test_double_never_uploads(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    script = _with_local_spad("R1_NearPlane_Reachable.script", tmp_path)
    assert _day(tmp_path, script, "6711.667", fake_gmat, fake_startup) == EXIT_PAUSED
    err = capsys.readouterr().err
    # The real readback, on the terminal before anything uses it.
    assert "T_max       : 6711.667 s = 1 h 51 min 51.667 s" in err
    (folder,) = (tmp_path / "days").iterdir()
    resume = ["day", "--resume", str(folder), "--min-free-gb", "0.001"]
    assert main([*resume, "--confirm-readback", "2"]) == EXIT_PAUSED
    err = capsys.readouterr().err
    # Step 3: the craft inspect names as the time origin, and the T_max check.
    assert "orbitwin inspect: mission clock: origin earliest" in err
    assert "orbitwin inspect: T_max check" in err and "--confirm-readback 3" in err
    assert main([*resume, "--confirm-readback", "3"]) == EXIT_PAUSED
    err = capsys.readouterr().err
    assert "C_r         : 300.0 km" in err and "--confirm-readback 5" in err
    assert main([*resume, "--confirm-readback", "5"]) == EXIT_STOP
    err = capsys.readouterr().err
    # rank's warnings every archive gets are shown, and stop nothing.
    assert "---- rank (step 5) ----" in err and "shown: The top-scoring candidate" in err
    assert "final: PRE-CHECK SKIPPED" in err and "final-01: PRE-CHECK SKIPPED" in err
    assert "Nothing was copied for upload" in err
    assert not (tmp_path / "final").exists()
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert {name: record["exit"] for name, record in state["steps"].items()} == {
        name: 0 for name in ("02-day-config", "03-inspect", "03-baseline", "04-search",
                             "certify-final-01", "05-scored-config", "05-search", "05-rank",
                             "certify-final")}
    assert (folder / "input" / "SphericalModel.spo").is_file()
    for name in ("day.yaml", "day-scored.yaml", "rank-scored.json", "search-01", "search-02",
                 "final-01", "final"):
        assert (folder / "run" / name).exists(), name
    # A resume finds every step done and stops at the same gate, running nothing again.
    assert main(resume) == EXIT_STOP
    err = capsys.readouterr().err
    assert err.count("done earlier with exit 0, its files unchanged; skipped") == 9
    assert "final: PRE-CHECK SKIPPED" in err


def test_craft_that_start_apart_need_the_window_given(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    script = _with_local_spad("R7_NoAlignment_NearMiss.script", tmp_path)
    assert _day(tmp_path, script, "7591.381", fake_gmat, fake_startup) == EXIT_STOP
    err = capsys.readouterr().err
    assert "start 1800 s apart" in err and "orbitwin day does not guess it" in err
    (folder,) = (tmp_path / "days").iterdir()
    assert not (folder / "run" / "day.yaml").exists()


def _once(tmp_path: Path, fake_gmat: Path, body: str) -> Path:
    """R4's fakegmat wrappers on the test double: the first GmatConsole run of certify-final
    (its folder run/final) does `body` instead of running."""
    import sys

    console = tmp_path / "once" / "GmatConsole"
    console.parent.mkdir()
    fired = tmp_path / "once" / "fired"
    console.write_text(
        f"#!{sys.executable}\nimport os, signal, sys, time\n"
        # Still the test double, as is_test_double reads it.
        "ORBITWIN_GMAT_TEST_DOUBLE = True\n"
        "if any('/run/final/' in arg for arg in sys.argv[1:]):\n"
        "    try:\n"
        f"        os.close(os.open({str(fired)!r}, os.O_CREAT | os.O_EXCL | os.O_WRONLY))\n"
        "    except FileExistsError:\n"
        "        pass\n"
        "    else:\n"
        f"        {body}\n"
        f"os.execv(sys.executable, [sys.executable, {str(fake_gmat)!r}, *sys.argv[1:]])\n",
        encoding="utf-8")
    console.chmod(0o755)
    return console


@pytest.mark.parametrize(("body", "timeout_s"), [
    ("os.kill(os.getpid(), signal.SIGSEGV)", 300.0),
    ("time.sleep(600)", 3.0),
])
def test_certify_that_gmat_failed_once_is_run_again_not_refused(
    tmp_path: Path, fake_gmat: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch, body: str, timeout_s: float,
) -> None:
    """R4 E and F: a SIGSEGV or a hang in certify-final exited 2 and stopped the day as a
    refusal; the --exclude it printed delivered final-01, 0.513 points down."""
    real = _quick_base

    def quick(path: Path) -> Path:
        base = real(path)
        raw = yaml.safe_load(base.read_text(encoding="utf-8"))
        raw["gmat"]["timeout_s"] = timeout_s
        base.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return base

    monkeypatch.setitem(globals(), "_quick_base", quick)
    script = _with_local_spad("R1_NearPlane_Reachable.script", tmp_path)
    console = _once(tmp_path, fake_gmat, body)
    startup = startup_for_console_elsewhere(tmp_path)
    assert _day(tmp_path, script, "6711.667", console, startup) == EXIT_PAUSED
    (folder,) = (tmp_path / "days").iterdir()
    resume = ["day", "--resume", str(folder), "--min-free-gb", "0.001"]
    for step in ("2", "3"):
        assert main([*resume, "--confirm-readback", step]) == EXIT_PAUSED
    capsys.readouterr()
    # The test double's own stop, PRE-CHECK SKIPPED, as without the crash.
    assert main([*resume, "--confirm-readback", "5"]) == EXIT_STOP
    err = capsys.readouterr().err
    assert (tmp_path / "once" / "fired").is_file()
    assert "certify-final got no answer from GmatConsole" in err
    assert "final: PRE-CHECK SKIPPED" in err and "no verdict" not in err.split("STOP")[-1]
    first = (folder / "logs" / "certify-final.err.interrupted").read_text(encoding="utf-8")
    assert ("exit status -11" if "SIGSEGV" in body else "timed out after 3 seconds") in first
    assert first.rstrip().endswith("which is no verdict on the candidate; run certify again "
                                   "(exit 75)")
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["steps"]["certify-final"]["exit"] == 0
    # Nothing of either run is left noted as running.
    assert not list((folder / ".running").iterdir())


def _bwrap_tmpfs(where: Path) -> list[str] | None:
    """bwrap's argv for a 64 MiB disk of its own at `where`, or None where it cannot."""
    import shutil
    import subprocess

    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return None
    argv = [bwrap, "--dev-bind", "/", "/", "--size", str(64 << 20), "--tmpfs", str(where)]
    where.mkdir(exist_ok=True)
    probe = subprocess.run([*argv, "true"], capture_output=True, check=False)
    return argv if probe.returncode == 0 else None


def test_a_disk_that_fills_mid_day_stops_with_the_resume_and_its_records(
    tmp_path: Path, fake_gmat: Path,
) -> None:
    """R4 H on a real full disk: the work and final roots on a 64 MiB tmpfs, filled to the
    last byte when certify-final starts; then emptied, and the resume printed taken."""
    import subprocess
    import sys

    disk = tmp_path / "disk"
    bwrap = _bwrap_tmpfs(disk)
    if bwrap is None:
        pytest.skip("bwrap cannot give this test a disk of its own")
    script = _with_local_spad("R1_NearPlane_Reachable.script", tmp_path)
    fill = (f"open({str(disk / 'filler')!r}, 'ab').close()\n"
            "        with open(" + repr(str(disk / "filler")) + ", 'ab') as f:\n"
            "            try:\n"
            "                while True:\n"
            "                    f.write(bytes(1 << 16)); f.flush()\n"
            "            except OSError:\n"
            "                pass")
    console = _once(tmp_path, fake_gmat, fill)
    startup = startup_for_console_elsewhere(tmp_path)
    driver = tmp_path / "driver.py"
    driver.write_text(f"""
import json, os, re, sys
from pathlib import Path
from orbitwin import day
from orbitwin.cli import main
day.provenance = lambda root, expect, allow: {{"source": "git", "commit": "c0ffee",
                                             "on_main": False, "notes": ["test"]}}
disk = Path({str(disk)!r})
codes = [main(["day", {str(script)!r}, "--companion", "SphericalModel.spo",
               "--horizon-s", "6711.667", "--kt-per-s", "0.003", "--ct-s", "5369.3336",
               "--kv-per-m-s", "0.0012", "--cv-m-s", "2000", "--cr-km", "300",
               "--base", {str(_quick_base(tmp_path))!r}, "--gmat-console", {str(console)!r},
               "--startup-file", {str(startup)!r}, "--work-root", str(disk / "days"),
               "--final-root", str(disk / "final"), "--min-free-gb", "0.001"])]
(folder,) = (disk / "days").iterdir()
resume = ["day", "--resume", str(folder), "--min-free-gb", "0.001"]
for step in ("2", "3", "5"):
    codes.append(main([*resume, "--confirm-readback", step]))
state = json.loads((folder / "state.json").read_text())
full = {{name: record["exit"] for name, record in state["steps"].items()}}
os.unlink(disk / "filler")
codes.append(main(resume))
again = {{name: record["exit"] for name, record in
         json.loads((folder / "state.json").read_text())["steps"].items()}}
print(json.dumps({{"codes": codes, "full": full, "again": again}}))
""", encoding="utf-8")
    done = subprocess.run([*bwrap, sys.executable, str(driver)], capture_output=True,
                          text=True, check=False, timeout=600,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    stops = done.stderr.split("orbitwin day: resuming")
    full_day = stops[3]
    assert result["codes"] == [EXIT_PAUSED, EXIT_PAUSED, EXIT_PAUSED, EXIT_STOP, EXIT_STOP], (
        full_day[-3000:])
    assert "Traceback" not in full_day, full_day[-3000:]
    # What was done before the disk filled stays done; certify on a full disk is no verdict.
    assert all(result["full"][name] == 0 for name in (
        "04-search", "certify-final-01", "05-search", "05-rank")), result
    assert result["full"]["certify-final"] == 75
    told = full_day.rsplit("orbitwin day: STOP", 1)[-1]
    # The retry stops on the free space, with the resume to take once it is freed.
    assert "certify-final got no answer" in full_day
    assert f"free space first, then run:\n\norbitwin day --resume {disk / 'days'}" in told
    assert "--exclude" not in told
    # Emptied, the resume runs only what did not end 0 and reaches the double's own stop.
    last = stops[4]
    assert "final: PRE-CHECK SKIPPED" in last and "Traceback" not in last
    assert result["again"]["certify-final"] == 0
    assert "step 5 (05-search): done earlier with exit 0" in last


@pytest.mark.slow
def test_a_resume_after_a_killed_day_waits_for_the_step_it_left(
    tmp_path: Path, fake_gmat: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R4 Dp: SIGKILL of the day 3 s into certify-final left certify running, and the
    resume typed at once ran a second certify beside it in the same run/."""
    import signal
    import subprocess
    import sys
    import time

    real = _quick_base

    def quick(path: Path) -> Path:
        base = real(path)
        raw = yaml.safe_load(base.read_text(encoding="utf-8"))
        raw["gmat"]["timeout_s"] = 8.0
        base.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return base

    monkeypatch.setitem(globals(), "_quick_base", quick)
    script = _with_local_spad("R1_NearPlane_Reachable.script", tmp_path)
    console = _once(tmp_path, fake_gmat, "time.sleep(600)")
    startup = startup_for_console_elsewhere(tmp_path)
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import sys\nfrom orbitwin import day\n"
        "day.provenance = lambda root, expect, allow: {'source': 'git', 'commit': 'c0ffee', "
        "'on_main': False, 'notes': ['test']}\n"
        "from orbitwin.cli import main\nsys.exit(main(sys.argv[1:]))\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    def day(*argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(driver), "day", *argv], env=env,
                              capture_output=True, text=True, check=False, timeout=300)

    first = day(str(script), "--companion", "SphericalModel.spo", "--horizon-s", "6711.667",
                "--kt-per-s", "0.003", "--ct-s", "5369.3336", "--kv-per-m-s", "0.0012",
                "--cv-m-s", "2000", "--cr-km", "300", "--base", str(_quick_base(tmp_path)),
                "--gmat-console", str(console), "--startup-file", str(startup),
                "--work-root", str(tmp_path / "days"), "--final-root", str(tmp_path / "final"),
                "--min-free-gb", "0.001")
    assert first.returncode == EXIT_PAUSED
    (folder,) = (tmp_path / "days").iterdir()
    resume = ["--resume", str(folder), "--min-free-gb", "0.001"]
    for step in ("2", "3"):
        assert day(*resume, "--confirm-readback", step).returncode == EXIT_PAUSED
    killed = subprocess.Popen([sys.executable, str(driver), "day", *resume,
                               "--confirm-readback", "5"], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fired = tmp_path / "once" / "fired"
    deadline = time.monotonic() + 240
    while not fired.exists():
        assert time.monotonic() < deadline and killed.poll() is None
        time.sleep(0.1)
    killed.send_signal(signal.SIGKILL)
    killed.wait()
    (note,) = (folder / ".running").iterdir()
    orphan = int(note.name)
    os.kill(orphan, 0)
    # At once, as the RUNBOOK's resume would be typed.
    refused = day(*resume)
    err = refused.stderr
    assert refused.returncode == EXIT_STOP
    assert "certify-final (started " in err and f"its processes: {orphan}" in err, err
    assert f"kill -TERM -- -{orphan}" in err
    assert "Wait for it to end" in err
    # Once it ended (its GMAT timed out), the resume runs certify again.
    while note.exists() and Path(f"/proc/{orphan}").exists():
        assert time.monotonic() < deadline
        time.sleep(0.2)
    done = day(*resume)
    err = done.stderr
    assert done.returncode == EXIT_STOP and "final: PRE-CHECK SKIPPED" in err, err
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    assert state["steps"]["certify-final"]["exit"] == 0
    assert not list((folder / ".running").iterdir())


def _descendants(pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="ascii", errors="replace")
        except OSError:
            continue
        children.setdefault(int(stat.rsplit(")", 1)[-1].split()[1]), []).append(
            int(entry.name))
    found, stack = [], [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return ""


@pytest.mark.slow
def test_a_solve_killed_under_the_day_ends_its_step_and_the_resume_works(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path,
) -> None:
    """#199 review P1 (e3c): SIGKILL of `orbitwin solve` under 05-search left its workers
    holding the step's pipes and the day's lock; the day hung and Ctrl-C did not end it."""
    import signal
    import subprocess
    import sys
    import threading
    import time

    script = _with_local_spad("R1_NearPlane_Reachable.script", tmp_path)
    # The ranking search, whose solve runs a pool of workers, as the review's quick2.yaml.
    base = _quick_base(tmp_path)
    raw = yaml.safe_load(base.read_text(encoding="utf-8"))
    raw["search"].update(ranking_search=True, workers=2)
    base.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import sys\nfrom orbitwin import day\n"
        "day.provenance = lambda root, expect, allow: {'source': 'git', 'commit': 'c0ffee', "
        "'on_main': False, 'notes': ['test']}\n"
        "from orbitwin.cli import main\nsys.exit(main(sys.argv[1:]))\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    def day(*argv: str, timeout: float = 300) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(driver), "day", *argv], env=env,
                              capture_output=True, text=True, check=False, timeout=timeout)

    first = day(str(script), "--companion", "SphericalModel.spo", "--horizon-s", "6711.667",
                "--kt-per-s", "0.003", "--ct-s", "5369.3336", "--kv-per-m-s", "0.0012",
                "--cv-m-s", "2000", "--cr-km", "300", "--base", str(base),
                "--gmat-console", str(fake_gmat), "--startup-file", str(fake_startup),
                "--work-root", str(tmp_path / "days"), "--final-root", str(tmp_path / "final"),
                "--min-free-gb", "0.001")
    assert first.returncode == EXIT_PAUSED
    (folder,) = (tmp_path / "days").iterdir()
    resume = ["--resume", str(folder), "--min-free-gb", "0.001"]
    for step in ("2", "3"):
        assert day(*resume, "--confirm-readback", step).returncode == EXIT_PAUSED
    killed: list[int] = []
    done_flag: list[bool] = []

    def watch() -> None:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline and not done_flag:
            for note in (folder / ".running").glob("*"):
                if '"step": "05-search"' not in note.read_text(encoding="utf-8"):
                    continue
                for pid in _descendants(int(note.name)):
                    if "orbitwin solve" in _cmdline(pid) and _descendants(pid):
                        os.kill(pid, signal.SIGKILL)
                        killed.append(pid)
                        return
            time.sleep(0.05)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    stopped = day(*resume, "--confirm-readback", "5", timeout=240)
    done_flag.append(True)
    watcher.join(timeout=5)
    assert killed, "the search ended before its solve had workers"
    # The day ended on its own, with the step's failure and the resume to type.
    assert stopped.returncode == EXIT_STOP, stopped.stderr[-3000:]
    assert "STOP: step 05-search exited " in stopped.stderr
    assert f"a resume runs it again:\n\norbitwin day --resume {folder}\n" in stopped.stderr
    # Nothing of the step is left, and the resume runs at once.
    assert not list((folder / ".running").iterdir())
    done = day(*resume)
    assert done.returncode == EXIT_STOP and "final: PRE-CHECK SKIPPED" in done.stderr, (
        done.stderr[-3000:])
