"""The suite make package runs in the extracted archive: slow is not hung.

Most stall limits here are 5 s; a few tests use 1 or 2 s where the child
only has to exit or never prints at all. A bare `python -c` child started in 0.15 to
0.39 s at load 30 on 20 cores, and a 1 s limit failed 4 runs in 12 under
`nice -n 19` at that load. A whole pytest is slower to start: 0.10 s idle,
but on 2026-11-03 one was still opening its first temporary file 5 s after
it was started, on a CI host running two suites at once (each 571 s against
160 s). So silence before the first byte has its own, longer limit.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import scratch_git_environment

STALL_S = 5.0


def _load_build_release(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "orbitwin_build_release_smoke", project_root / "scripts" / "build_release.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_extracted_suite_is_allowed_longer_than_it_takes(project_root: Path) -> None:
    """The guard was 180 s, then 1800 s, and the suite outgrew both.

    On 2026-11-02 one `make package` passed in 1798 s and the next was stopped
    at 1800 s; the same suite from the same archive passed in 3032 s beside
    it. Silence is what stops a run now; the backstop sits far above that.
    """
    build_release = _load_build_release(project_root)
    measured_extracted_suite_s = 3032

    assert 4 * measured_extracted_suite_s <= build_release.EXTRACTED_TEST_TIMEOUT_S
    startup_s = build_release.EXTRACTED_TEST_STARTUP_S
    assert 300 <= startup_s < build_release.EXTRACTED_TEST_TIMEOUT_S, "start-up has room, bounded"
    assert build_release.EXTRACTED_TEST_STALL_S >= 300, "one slow test is not a hang"


def _extracted(tmp_path: Path) -> Path:
    """An unpacked archive: no output/, which is not shipped."""
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True)
    return extracted


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _log(extracted: Path) -> str:
    return (extracted / "output" / "extracted_fast_test.log").read_text(encoding="utf-8")


def _gone(pid: int, wait_s: float = 10.0) -> bool:
    """Not running: no such process, or a zombie its new parent has yet to reap.

    An orphan is reparented to init or a subreaper, which reaps it when it
    gets to it; until then kill(pid, 0) still succeeds on the dead process.
    """
    deadline = time.monotonic() + wait_s
    while True:
        try:
            state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):  # ESRCH: reaped between open and read
            return True
        if state in {"Z", "X"}:
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)


def test_gone_counts_a_process_reaped_between_open_and_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading /proc/<pid>/stat of a process reaped after open() fails with ESRCH."""

    def reaped(self: Path, *args: object, **kwargs: object) -> str:
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(Path, "read_text", reaped)
    assert _gone(1, wait_s=0.0)


def test_gone_does_not_swallow_a_refused_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """EPERM (hidepid=noaccess) says nothing about whether the process runs."""

    def refused(self: Path, *args: object, **kwargs: object) -> str:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(Path, "read_text", refused)
    with pytest.raises(PermissionError):
        _gone(1, wait_s=0.0)


def test_a_silent_suite_is_stopped_with_everything_it_started(
    project_root: Path, tmp_path: Path
) -> None:
    """A hang is silence. The log keeps what came before it, and no worker survives.

    Killing pytest alone left its pool workers running with parent 1.
    """
    build_release = _load_build_release(project_root)
    extracted = _extracted(tmp_path)
    pids = tmp_path / "pids"
    child = (
        "import os, subprocess, sys, time\n"
        "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{worker.pid}}')\n"
        "print('[ 41%]', flush=True)\n"
        "time.sleep(600)\n"
    )
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="printed nothing for") as raised:
        build_release.extracted_smoke_test(
            extracted, _python(child), stall_s=STALL_S, poll_s=0.1, grace_s=1.0
        )

    assert time.monotonic() - started < 60, "stopped, not waited out"
    assert "extracted-smoke-failure.log" in str(raised.value)
    assert "[ 41%]" in _log(extracted)
    suite, worker = (int(pid) for pid in pids.read_text(encoding="utf-8").split())
    assert _gone(suite)
    assert _gone(worker), "the suite's own child outlived it"


def test_a_member_that_ignores_sigint_is_killed_too(project_root: Path, tmp_path: Path) -> None:
    """multiprocessing's resource tracker and forkserver ignore SIGINT; SIGKILL has to follow.

    The suite leaves at the SIGINT, before the grace period runs out, so
    stopping once the leader has gone would leave the member behind.
    """
    build_release = _load_build_release(project_root)
    extracted = _extracted(tmp_path)
    pids = tmp_path / "pids"
    child = (
        "import os, signal, subprocess, sys, time\n"
        "member = subprocess.Popen([sys.executable, '-c', "
        "'import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(600)'])\n"
        f"open({str(pids)!r}, 'w').write(str(member.pid))\n"
        "print('started', flush=True)\n"
        "time.sleep(600)\n"
    )

    with pytest.raises(RuntimeError, match="printed nothing for"):
        build_release.extracted_smoke_test(
            extracted, _python(child), stall_s=STALL_S, poll_s=0.1, grace_s=5.0
        )

    assert _gone(int(pids.read_text(encoding="utf-8")))


def test_a_suite_that_ends_leaves_nothing_running(project_root: Path, tmp_path: Path) -> None:
    """A worker the suite left behind when it exited normally is stopped as well."""
    build_release = _load_build_release(project_root)
    extracted = _extracted(tmp_path)
    pids = tmp_path / "pids"
    child = (
        "import subprocess, sys\n"
        "left = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"open({str(pids)!r}, 'w').write(str(left.pid))\n"
        "print('done', flush=True)\n"
    )

    build_release.extracted_smoke_test(extracted, _python(child), poll_s=0.1)

    assert _gone(int(pids.read_text(encoding="utf-8")))


def test_a_stopped_suite_gets_sigint_first_and_says_where_it_was(
    project_root: Path, tmp_path: Path
) -> None:
    """pytest prints the test it was in on SIGINT; that is the end of the log."""
    build_release = _load_build_release(project_root)
    extracted = _extracted(tmp_path)
    child = (
        "import sys, time\n"
        "print('started', flush=True)\n"
        "try:\n"
        "    time.sleep(600)\n"
        "except KeyboardInterrupt:\n"
        "    print('interrupted in test_this_one_hangs', file=sys.stderr, flush=True)\n"
    )

    with pytest.raises(RuntimeError, match="printed nothing for"):
        build_release.extracted_smoke_test(
            extracted, _python(child), stall_s=STALL_S, poll_s=0.1, grace_s=5.0
        )

    assert _log(extracted).endswith("interrupted in test_this_one_hangs\n")


def test_a_slow_suite_that_keeps_printing_is_not_called_hung(
    project_root: Path, tmp_path: Path
) -> None:
    """It runs past the stall limit in total, never silent for that long.

    It writes without flushing; PYTHONUNBUFFERED is what makes that reach the
    log as it is written. pytest flushes its own progress, but not everything
    the suite's processes print does.
    """
    build_release = _load_build_release(project_root)
    extracted = _extracted(tmp_path)
    child = "import sys, time\nfor _ in range(16):\n    sys.stdout.write('.'); time.sleep(0.5)"

    build_release.extracted_smoke_test(extracted, _python(child), stall_s=STALL_S, poll_s=0.1)

    assert _log(extracted) == "." * 16


def test_a_slow_start_is_not_a_stall(project_root: Path, tmp_path: Path) -> None:
    """Silent for three stall limits before its first byte, then done."""
    build_release = _load_build_release(project_root)
    extracted = _extracted(tmp_path)
    child = "import time\ntime.sleep(3)\nprint('collected', flush=True)\n"

    build_release.extracted_smoke_test(extracted, _python(child), stall_s=1.0, poll_s=0.1)

    assert _log(extracted) == "collected\n"


def test_a_start_that_never_prints_is_stopped(project_root: Path, tmp_path: Path) -> None:
    build_release = _load_build_release(project_root)
    extracted = _extracted(tmp_path)
    started = time.monotonic()

    with pytest.raises(RuntimeError, match=r"printed nothing in their first \d+ s.*still starting"):
        build_release.extracted_smoke_test(
            extracted, _python("import time; time.sleep(600)"), stall_s=60.0, poll_s=0.1,
            grace_s=1.0, startup_s=2.0,
        )

    assert time.monotonic() - started < 30


def test_stderr_reaches_the_log(project_root: Path, tmp_path: Path) -> None:
    """faulthandler's dump and pytest's INTERNALERROR go to stderr."""
    build_release = _load_build_release(project_root)
    extracted = _extracted(tmp_path)

    build_release.extracted_smoke_test(
        extracted, _python("import sys; sys.stderr.write('on stderr\\n')"), poll_s=0.1
    )

    assert _log(extracted) == "on stderr\n"


def test_the_default_suite_dumps_a_long_test_into_the_log(project_root: Path) -> None:
    import inspect

    build_release = _load_build_release(project_root)
    source = inspect.getsource(build_release.extracted_smoke_test)

    assert '"-o", "faulthandler_timeout=300"' in source
    assert build_release.EXTRACTED_TEST_STALL_S > 300, "the dump lands before the stop"


def test_the_backstop_stops_a_suite_that_never_ends(project_root: Path, tmp_path: Path) -> None:
    build_release = _load_build_release(project_root)
    extracted = _extracted(tmp_path)
    child = "import sys, time\nwhile True:\n    sys.stdout.write('.'); time.sleep(0.1)"

    with pytest.raises(RuntimeError, match="were still running at 2 s"):
        build_release.extracted_smoke_test(
            extracted, _python(child), stall_s=60.0, timeout_s=2.0, poll_s=0.1, grace_s=1.0
        )


def test_a_failing_extracted_suite_names_the_log(project_root: Path, tmp_path: Path) -> None:
    """"Fast tests failed" alone does not say where the output went."""
    build_release = _load_build_release(project_root)
    extracted = _extracted(tmp_path)

    with pytest.raises(RuntimeError, match=r"exited 1\).*extracted-smoke-failure\.log"):
        build_release.extracted_smoke_test(
            extracted, _python("import sys; print('1 failed'); sys.exit(1)"), poll_s=0.1
        )

    assert "1 failed" in _log(extracted)


def _stalling_checkout(
    project_root: Path, tmp_path: Path, slow_start_s: float = 0.0
) -> tuple[Path, dict[str, str]]:
    """A checkout whose archive carries a suite that prints a sentinel, then hangs.

    With slow_start_s the suite first sits that long in its conftest, before
    pytest has printed anything.
    """
    import hashlib
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git is not installed; packaging needs a git checkout")
    environment = scratch_git_environment(tmp_path)
    # The build extracts its archive here, where the Ctrl-C test can find the log.
    (tmp_path / "tmp").mkdir()
    environment["TMPDIR"] = str(tmp_path / "tmp")
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    source = (project_root / "scripts" / "build_release.py").read_text(encoding="utf-8")
    for needle, replacement in (
        ("EXTRACTED_TEST_STALL_S = 600.0", f"EXTRACTED_TEST_STALL_S = {STALL_S}"),
        ("_EXTRACTED_TEST_POLL_S = 5.0", "_EXTRACTED_TEST_POLL_S = 0.1"),
        ("_EXTRACTED_TEST_GRACE_S = 10.0", "_EXTRACTED_TEST_GRACE_S = 5.0"),
        ("EXTRACTED_TEST_STARTUP_S = 600.0", "EXTRACTED_TEST_STARTUP_S = 60.0"),
    ):
        assert needle in source, f"the constant this test shortens moved: {needle}"
        source = source.replace(needle, replacement)
    (repo / "scripts" / "build_release.py").write_text(source, encoding="utf-8")
    (repo / "scripts" / "make_manifest.py").write_bytes(
        (project_root / "scripts" / "make_manifest.py").read_bytes()
    )
    (repo / "VERSION").write_text("0.0.0\n", encoding="utf-8")
    # A one-wheel offline bundle, so this builds whether or not main() ships one.
    (repo / ".gitignore").write_text("vendor/wheels/*.whl\n", encoding="utf-8")
    wheels = repo / "vendor" / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "demo-1.0-py3-none-any.whl").write_bytes(b"wheel")
    (wheels / "SHA256SUMS.txt").write_text(
        f"{hashlib.sha256(b'wheel').hexdigest()}  ./demo-1.0-py3-none-any.whl\n",
        encoding="utf-8",
    )
    # As pyproject.toml has it: -q prints no header, so collection is start-up.
    (repo / "pytest.ini").write_text("[pytest]\naddopts = -q\n", encoding="utf-8")
    # A suite that sits out the first SIGINT, as one stuck in a C call does, so
    # what follows it (a second Ctrl-C, or the SIGKILL) is what stops it.
    (repo / "conftest.py").write_text(
        f"import os, time\n\ntime.sleep({slow_start_s!r})\n\ndef pytest_collection_finish(session):\n"
        f"    open({str(tmp_path / 'suite.pid')!r}, 'w').write(str(os.getpid()))\n"
        "    print('SENTINEL', flush=True)\n"
        "    try:\n        time.sleep(600)\n"
        "    except KeyboardInterrupt:\n        time.sleep(600)\n",
        encoding="utf-8",
    )
    origin = repo.parent / "origin.git"
    for args in (("init", "-q"), ("config", "user.email", "t@example.com"),
                 ("config", "user.name", "t"), ("add", "."), ("commit", "-qm", "init"),
                 # A release is built from origin/main.
                 ("init", "-q", "--bare", str(origin)), ("remote", "add", "origin", str(origin)),
                 ("push", "-q", "origin", "HEAD:refs/heads/main")):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                       env=environment)
    return repo, environment


@pytest.mark.parametrize("slow_start_s", [0.0, 2 * STALL_S], ids=["prompt", "slow-start"])
def test_a_stopped_build_copies_what_the_suite_printed(
    project_root: Path, tmp_path: Path, slow_start_s: float
) -> None:
    """main() copies the log next to the archive on a stop as on a failure.

    On a loaded CI host the suite was stopped before its first byte, and the
    log held a KeyboardInterrupt in tempfile, not the sentinel; slow-start
    makes that start-up take twice the stall limit.
    """
    repo, environment = _stalling_checkout(project_root, tmp_path, slow_start_s)

    completed = subprocess.run(
        [sys.executable, "scripts/build_release.py", "--project-root", ".", "--output-dir", "dist"],
        cwd=repo, capture_output=True, text=True, check=False, env=environment, timeout=180,
    )

    assert completed.returncode != 0
    assert "printed nothing for" in completed.stderr
    [copied] = (repo / "dist").glob("tasa_orbitwin-0.0.0-*.extracted-smoke-failure.log")
    assert "SENTINEL" in copied.read_text(encoding="utf-8")


def _restore_signals() -> None:
    """The build starts with default signals, whatever this test runner ignores."""
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(number, signal.SIG_DFL)


def _build_until_the_suite_runs(repo: Path, tmp_path: Path, environment: dict[str, str]):
    build = subprocess.Popen(
        [sys.executable, "scripts/build_release.py", "--project-root", ".", "--output-dir", "dist"],
        cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment,
        preexec_fn=_restore_signals,
    )
    deadline = time.monotonic() + 180
    while not (tmp_path / "suite.pid").is_file():
        if build.poll() is not None or time.monotonic() > deadline:
            _, stderr = build.communicate(timeout=60) if build.poll() is not None else ("", "")
            build.kill()
            pytest.fail(f"the suite never started: {stderr[-2000:]}")
        time.sleep(0.2)
    time.sleep(0.5)  # past the pid write, to the sentinel
    return build, int((tmp_path / "suite.pid").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "signals",
    [(signal.SIGINT,), (signal.SIGINT, signal.SIGINT), (signal.SIGTERM,), (signal.SIGHUP,),
     (signal.SIGQUIT,)],
    ids=["ctrl-c", "ctrl-c-twice", "sigterm", "sighup", "ctrl-backslash"],
)
def test_an_interrupted_build_stops_the_suite_and_keeps_its_log(
    project_root: Path, tmp_path: Path, signals: tuple[int, ...]
) -> None:
    """With the stall limit at minutes the operator's likeliest move is Ctrl-C.

    main() caught Exception only, so a KeyboardInterrupt left no log. In its
    own session the suite no longer got the terminal's signals: a second
    Ctrl-C during the grace period, SIGTERM, a closed terminal (SIGHUP) or
    Ctrl-\\ left it running with parent 1. The suite here sits out the first
    SIGINT, so one signal waits the grace period (5 s) and two do not.
    """
    repo, environment = _stalling_checkout(project_root, tmp_path)
    build, suite = _build_until_the_suite_runs(repo, tmp_path, environment)
    try:
        started = time.monotonic()
        for index, number in enumerate(signals):
            if index:
                time.sleep(0.5)
            build.send_signal(number)
        _, stderr = build.communicate(timeout=60)
        elapsed = time.monotonic() - started
        survived = not _gone(suite)
    finally:
        # Only so a failure here leaves nothing running; judged above, before this.
        if build.poll() is None:
            build.kill()
        if not _gone(suite, wait_s=0):
            with contextlib.suppress(ProcessLookupError):  # reaped since the check
                os.kill(suite, signal.SIGKILL)

    assert build.returncode != 0
    assert "KeyboardInterrupt" in stderr
    assert "stopping the extracted suite" in stderr
    [copied] = (repo / "dist").glob("tasa_orbitwin-0.0.0-*.extracted-smoke-failure.log")
    assert "SENTINEL" in copied.read_text(encoding="utf-8")
    assert not survived, "the suite outlived the build"
    if len(signals) == 2:
        assert elapsed < 4.0, f"the second Ctrl-C did not cut the grace period short: {elapsed:.1f} s"
    else:
        assert elapsed >= 4.5, f"stopped before the grace period ran out: {elapsed:.1f} s"


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGHUP], ids=["ctrl-c", "hangup"])
def test_a_build_whose_stderr_is_gone_still_stops_the_suite(
    project_root: Path, tmp_path: Path, number: int
) -> None:
    """A closed terminal, or a `| tee` the same Ctrl-C killed, makes writing to stderr fail.

    The stop message was printed before the stop, and its OSError skipped it.
    """
    repo, environment = _stalling_checkout(project_root, tmp_path)
    build, suite = _build_until_the_suite_runs(repo, tmp_path, environment)
    try:
        assert build.stderr is not None
        build.stderr.close()
        build.send_signal(number)
        build.wait(timeout=60)
        survived = not _gone(suite)
    finally:
        if build.poll() is None:
            build.kill()
        if not _gone(suite, wait_s=0):
            with contextlib.suppress(ProcessLookupError):  # reaped since the check
                os.kill(suite, signal.SIGKILL)

    assert build.returncode != 0
    assert not survived, "the suite outlived the build"


def test_nohup_keeps_the_build_running_through_sighup(
    project_root: Path, tmp_path: Path
) -> None:
    """A SIGHUP this process was told to ignore (nohup) stays ignored while the suite runs."""
    build_release = _load_build_release(project_root)
    extracted = _extracted(tmp_path)
    child = (
        "import os, signal, time\n"
        "os.kill(os.getppid(), signal.SIGHUP)\n"
        "time.sleep(0.5)\n"
        "print('done', flush=True)\n"
    )
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        build_release.extracted_smoke_test(extracted, _python(child), poll_s=0.1)
    except KeyboardInterrupt:
        pytest.fail("the ignored SIGHUP stopped the build")
    finally:
        signal.signal(signal.SIGHUP, previous)

    assert _log(extracted) == "done\n"


def test_a_suite_started_with_sigint_ignored_still_gets_it(
    project_root: Path, tmp_path: Path
) -> None:
    """Started from a script with `&`, the build inherits SIGINT ignored, and the suite with it."""
    build_release = _load_build_release(project_root)
    extracted = _extracted(tmp_path)
    child = (
        "import sys, time\n"
        "print('started', flush=True)\n"
        "try:\n"
        "    time.sleep(600)\n"
        "except KeyboardInterrupt:\n"
        "    print('interrupted', file=sys.stderr, flush=True)\n"
    )
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with pytest.raises(RuntimeError, match="printed nothing for"):
            build_release.extracted_smoke_test(
                extracted, _python(child), stall_s=STALL_S, poll_s=0.1, grace_s=5.0
            )
    finally:
        signal.signal(signal.SIGINT, previous)

    assert _log(extracted).endswith("interrupted\n")


