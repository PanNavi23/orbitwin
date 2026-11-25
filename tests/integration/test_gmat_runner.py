from __future__ import annotations

import os
import shutil
import signal
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.gmat_runner import derive_startup_file, run_gmat
from orbitwin.verify import baseline_original_script


def test_derived_startup_isolates_output(tmp_path: Path, fake_startup: Path) -> None:
    target = derive_startup_file(
        fake_startup,
        tmp_path / "startup.txt",
        tmp_path / "out",
        tmp_path / "log.txt",
    )
    text = target.read_text(encoding="utf-8")
    assert str((tmp_path / "out").resolve()) in text
    assert str((tmp_path / "log.txt").resolve()) in text
    assert fake_startup.read_text(encoding="utf-8").startswith("ROOT_PATH")


def test_fake_gmat_runner_produces_fresh_report(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    scenario, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    result = run_gmat(
        fake_gmat,
        workshop_script,
        tmp_path / "run",
        [report.filename for report in scenario.report_files],
        startup_file=fake_startup,
        timeout_s=10.0,
    )
    assert result.success
    assert result.report_sha256["ReportFile1.txt"]



def test_a_failed_run_keeps_what_it_wrote_apart_from_its_reports(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """A drag model below 100 km aborts the mission; the rows before it are still written."""
    script = tmp_path / "aborts.script"
    script.write_text(
        "% FAKE_GMAT_ABORT_AT_S 5\n" + workshop_script.read_text(encoding="utf-8"), encoding="utf-8"
    )
    scenario, _ = parse_gmat_script(script, OrbitWinConfig())
    names = [report.filename for report in scenario.report_files]
    result = run_gmat(
        fake_gmat, script, tmp_path / "run", names, startup_file=fake_startup, timeout_s=10.0
    )
    assert not result.success
    assert result.reports == {} and result.report_sha256 == {}
    assert set(result.partial_reports) == set(names)

def test_baseline_preserves_original_bytes(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    before = workshop_script.read_bytes()
    evidence = baseline_original_script(
        workshop_script,
        OrbitWinConfig(),
        fake_gmat,
        fake_startup,
        tmp_path / "baseline",
    )
    assert evidence["original_sha256_before"] == evidence["original_sha256_after"]
    assert workshop_script.read_bytes() == before


def test_fake_gmat_console_keeps_its_executable_bit(fake_gmat: Path) -> None:
    # The orchestration tests spawn this file as a process. Losing the mode bit
    # during packaging silently disables every GMAT integration test.
    assert os.access(fake_gmat, os.X_OK), f"{fake_gmat} is not executable"


def test_a_timed_out_run_keeps_no_partial_report(tmp_path: Path) -> None:
    """A killed process can leave a torn last row."""
    console, startup, script = _shell_console(
        tmp_path,
        'out=$(sed -n "s/^OUTPUT_PATH *= *//p" "$2")\n'
        'printf "a b\\n1 2\\n" > "${out}R.txt"\nexec sleep 30\n',
    )
    result = run_gmat(console, script, tmp_path / "run", ["R.txt"], startup_file=startup,
                      timeout_s=1.0)
    assert not result.success and "timed out" in (result.error or "")
    assert (tmp_path / "run" / "gmat-output" / "R.txt").is_file()
    assert result.partial_reports == {}


def test_a_run_that_writes_past_the_size_limit_is_stopped_and_says_so(tmp_path: Path) -> None:
    """The kernel ends it with SIGXFSZ; a goal GMAT never reaches wrote 457 MB in 40 s."""
    console, startup, script = _shell_console(
        tmp_path,
        'out=$(sed -n "s/^OUTPUT_PATH *= *//p" "$2")\n'
        'exec head -c 100000 /dev/zero > "${out}R.txt"\n',
    )
    result = run_gmat(console, script, tmp_path / "run", ["R.txt"], startup_file=startup,
                      timeout_s=10.0, file_size_limit_bytes=4096)
    assert result.return_code == -signal.SIGXFSZ
    assert "4096 byte limit" in (result.error or "") and "SIGXFSZ" in (result.error or "")
    assert (tmp_path / "run" / "gmat-output" / "R.txt").stat().st_size == 4096


def test_a_run_has_no_core_dumps(tmp_path: Path) -> None:
    console, startup, script = _shell_console(
        tmp_path, 'ulimit -c > "$(dirname "$4")/core.txt"\nexit 0\n'
    )
    run_gmat(console, script, tmp_path / "run", [], startup_file=startup, timeout_s=10.0)
    assert (tmp_path / "run" / "core.txt").read_text().strip() == "0"


def test_epochs_do_not_match_is_not_fatal(tmp_path: Path) -> None:
    """R2026a prints it on successful runs whose alignment ended a double step off."""
    console, startup, script = _shell_console(tmp_path, "echo 'Epochs do not match'\nexit 0\n")
    result = run_gmat(console, script, tmp_path / "run", [], startup_file=startup, timeout_s=10.0)
    assert result.success, result.error


def test_on_started_is_called_once_gmat_has_run_and_not_when_refused(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.errors import GmatRunError

    started: list[None] = []
    run_gmat(
        fake_gmat, workshop_script, tmp_path / "ran", ["ReportFile1.txt"],
        startup_file=fake_startup, timeout_s=10.0, on_started=lambda: started.append(None),
    )
    assert len(started) == 1
    with pytest.raises(GmatRunError, match="startup file"):
        run_gmat(
            fake_gmat, workshop_script, tmp_path / "refused", ["ReportFile1.txt"],
            startup_file=None, on_started=lambda: started.append(None),
        )
    assert len(started) == 1


def test_console_output_that_is_not_utf8_is_kept_not_raised(tmp_path: Path) -> None:
    """A byte like 0xff on stdout raised UnicodeDecodeError after GMAT had run."""
    console = tmp_path / "console.sh"
    console.write_text("#!/bin/sh\nprintf 'GMAT \\377\\376 output\\n'\nexit 0\n")
    console.chmod(0o755)
    startup = tmp_path / "startup.txt"
    startup.write_text("OUTPUT_PATH = ./\n")
    script = tmp_path / "s.script"
    script.write_text("% nothing\n")
    result = run_gmat(console, script, tmp_path / "run", [], startup_file=startup, timeout_s=10.0)
    assert result.success, result.error
    assert "GMAT \ufffd\ufffd output" in Path(result.stdout_path).read_text(encoding="utf-8")


def _shell_console(
    tmp_path: Path, body: str, *, executable: bool = True
) -> tuple[Path, Path, Path]:
    console = tmp_path / "console.sh"
    console.write_text("#!/bin/sh\n" + body)
    console.chmod(0o755 if executable else 0o644)
    startup = tmp_path / "startup.txt"
    startup.write_text("OUTPUT_PATH = ./\n")
    script = tmp_path / "s.script"
    script.write_text("% nothing\n")
    return console, startup, script


@pytest.mark.parametrize(
    "body",
    [
        # Exits 0, then stdout.txt is a directory and cannot be written.
        'log="$4"; mkdir -p "$(dirname "$log")/stdout.txt"\nexit 0\n',
        # Times out with stdout.txt a directory: the timeout path's write fails.
        'log="$4"; mkdir -p "$(dirname "$log")/stdout.txt"\nexec sleep 30\n',
    ],
    ids=["exited", "timed-out"],
)
def test_on_started_counts_a_run_whose_output_cannot_then_be_written(
    tmp_path: Path, body: str
) -> None:
    console, startup, script = _shell_console(tmp_path, body)
    started: list[None] = []
    with pytest.raises(IsADirectoryError):
        run_gmat(console, script, tmp_path / "run", [], startup_file=startup,
                 timeout_s=1.0, on_started=lambda: started.append(None))
    assert len(started) == 1


def test_on_started_counts_a_timed_out_run_once(tmp_path: Path) -> None:
    console, startup, script = _shell_console(tmp_path, "exec sleep 30\n")
    started: list[None] = []
    result = run_gmat(console, script, tmp_path / "run", [], startup_file=startup,
                      timeout_s=0.5, on_started=lambda: started.append(None))
    assert not result.success and "timed out" in (result.error or "")
    assert len(started) == 1


def test_on_started_is_not_called_for_a_console_that_cannot_start(tmp_path: Path) -> None:
    console, startup, script = _shell_console(tmp_path, "exit 0\n", executable=False)
    started: list[None] = []
    with pytest.raises(PermissionError):
        run_gmat(console, script, tmp_path / "run", [], startup_file=startup,
                 timeout_s=5.0, on_started=lambda: started.append(None))
    assert started == []


def test_a_timeout_ends_a_console_that_does_not_exec(tmp_path: Path) -> None:
    """A wrapper's child held the pipes, and communicate() waited out the child."""
    import time

    console, startup, script = _shell_console(tmp_path, "sleep 30\nexit 0\n")
    started = time.monotonic()
    result = run_gmat(console, script, tmp_path / "run", [], startup_file=startup, timeout_s=1.0)
    assert time.monotonic() - started < 15.0
    assert "timed out" in (result.error or "")


def test_a_report_cut_by_the_size_cap_is_not_kept_as_partial(tmp_path: Path) -> None:
    """A torn last row, '6999.' of 6999.123, parses as a number."""
    console, startup, script = _shell_console(
        tmp_path,
        'out=$(sed -n "s/^OUTPUT_PATH *= *//p" "$2")\n'
        'exec head -c 100000 /dev/zero > "${out}R.txt"\n',
    )
    result = run_gmat(console, script, tmp_path / "run", ["R.txt"], startup_file=startup,
                      timeout_s=10.0, file_size_limit_bytes=4096)
    assert result.return_code == -signal.SIGXFSZ
    assert result.partial_reports == {}


def test_the_cap_stays_under_a_hard_limit_already_in_force(tmp_path: Path) -> None:
    """An unprivileged process cannot raise a hard limit; asking to refused every run."""
    import subprocess
    import sys

    console, startup, script = _shell_console(tmp_path, 'ulimit -f > "$(dirname "$4")/f.txt"\nexit 0\n')
    # In a child: a hard limit lowered here could not be raised again.
    code = (
        "import resource, sys\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))\n"
        "from orbitwin.gmat_runner import run_gmat\n"
        f"r = run_gmat({str(console)!r}, {str(script)!r}, {str(tmp_path / 'run')!r}, [],"
        f" startup_file={str(startup)!r}, timeout_s=10.0, file_size_limit_bytes=1 << 30)\n"
        "sys.exit(0 if r.success else 1)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    # ulimit -f counts 512-byte blocks: 1 MiB.
    assert (tmp_path / "run" / "f.txt").read_text().strip() == "2048"


def test_a_shells_153_is_the_size_cap_too(tmp_path: Path) -> None:
    """A wrapper that does not exec reports its child's SIGXFSZ as 128 + 25."""
    console, startup, script = _shell_console(
        tmp_path,
        'out=$(sed -n "s/^OUTPUT_PATH *= *//p" "$2")\n'
        'head -c 100000 /dev/zero > "${out}R.txt"\nexit $?\n',
    )
    result = run_gmat(console, script, tmp_path / "run", ["R.txt"], startup_file=startup,
                      timeout_s=10.0, file_size_limit_bytes=4096)
    assert result.return_code == 153
    assert "SIGXFSZ" in (result.error or "") and result.partial_reports == {}


@pytest.mark.skipif(shutil.which("prlimit") is None, reason="needs util-linux prlimit")
def test_the_cap_holds_before_popen_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Set on the pid after Popen returned, the cap missed whatever the console ran first."""
    import subprocess
    import time

    console, startup, script = _shell_console(
        tmp_path, "ulimit -f > f.txt\n(ulimit -f; ulimit -c) >> f.txt\nexec sleep 1\n"
    )
    report = tmp_path / "f.txt"

    class SlowPopen(subprocess.Popen):  # type: ignore[type-arg]
        # A loaded host: the console runs before Popen's caller does.
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[call-overload]
            if str(console.resolve()) not in args[0]:  # type: ignore[operator]
                return
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and (
                not report.is_file() or len(report.read_text().split()) < 3
            ):
                time.sleep(0.01)

    monkeypatch.setattr(subprocess, "Popen", SlowPopen)
    run_gmat(console, script, tmp_path / "run", [], startup_file=startup,
             timeout_s=10.0, file_size_limit_bytes=4096)
    # ulimit -f counts 512-byte blocks.
    assert report.read_text().split() == ["8", "8", "0"]


def test_without_prlimit_the_cap_is_set_on_the_pid_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orbitwin import gmat_runner

    which = shutil.which
    monkeypatch.setattr(gmat_runner, "_PRLIMIT", None)
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "prlimit" else which(name))
    # Late enough for the cap set after Popen.
    console, startup, script = _shell_console(
        tmp_path, 'sleep 1\nulimit -f > "$(dirname "$4")/f.txt"\nulimit -c >> "$(dirname "$4")/f.txt"\n'
    )
    with pytest.warns(RuntimeWarning, match="prlimit is not available"):
        result = run_gmat(console, script, tmp_path / "run", [], startup_file=startup,
                          timeout_s=10.0, file_size_limit_bytes=4096)
    assert result.success, result.error
    assert (tmp_path / "run" / "f.txt").read_text().split() == ["8", "0"]


@pytest.mark.parametrize(
    ("limit", "error"),
    [(-1, ValueError), (1 << 63, ValueError), (4096.0, TypeError), (True, TypeError)],
)
def test_a_size_limit_that_caps_nothing_or_is_not_an_int_is_refused(
    tmp_path: Path, limit: object, error: type[Exception]
) -> None:
    console, startup, script = _shell_console(tmp_path, "exit 0\n")
    with pytest.raises(error, match="file_size_limit_bytes"):
        run_gmat(console, script, tmp_path / "run", [], startup_file=startup,
                 file_size_limit_bytes=limit)  # type: ignore[arg-type]


def test_gmat_dies_with_the_process_that_ran_it(tmp_path: Path) -> None:
    """Its own session hides it from a group SIGTERM; it is told to die with us."""
    import subprocess
    import sys
    import time

    console, startup, script = _shell_console(
        tmp_path, 'echo $$ > "$(dirname "$4")/pid.txt"\nexec sleep 300\n'
    )
    code = (
        "from orbitwin.gmat_runner import run_gmat\n"
        f"run_gmat({str(console)!r}, {str(script)!r}, {str(tmp_path / 'run')!r}, [],"
        f" startup_file={str(startup)!r}, timeout_s=300.0)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", code])
    pid_file = tmp_path / "run" / "pid.txt"
    deadline = time.monotonic() + 30.0
    while not pid_file.is_file() or not pid_file.read_text().strip():
        assert time.monotonic() < deadline
        time.sleep(0.1)
    child = int(pid_file.read_text())
    parent.terminate()
    parent.wait(timeout=30.0)
    deadline = time.monotonic() + 10.0
    while Path(f"/proc/{child}").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    try:
        state = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):  # reaped, before or during the read
        state = "gone"
    assert state in {"gone", "Z", "X"}
