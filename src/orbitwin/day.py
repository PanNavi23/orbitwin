"""orbitwin day: docs/RUNBOOK.md steps 1 to 7 as one command that fails closed.

Every step is the RUNBOOK's own command, run as a child process with its output
under <day>/logs/. <day>/state.json records each step's exit status and the
SHA-256 of what it wrote; --resume skips a step only while both still hold.
Two people confirm each readback (steps 2 and 5) before the day goes on.
Nothing reaches the final folder unless every candidate left in the running
certified with PRE-CHECK PASS and passed step 7 with exit 0, and the chosen one
passed audit_burn_table.py; `latest` moves only after step 7 passed again on
the copies.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import errno
import fcntl
import hashlib
import json
import math
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import IO, Any

from .announcement import (
    Entry,
    against_script,
    horizon_s,
    read_pair,
    read_script,
    written_epochs,
)
from .argtypes import positive_finite, positive_int
from .config import CLOSEST_APPROACH_WINDOWS, load_config, ranking_profile
from .design import record_path
from .errors import OrbitWinError
from .installed import FRESH_PYCACHE, installed
from .layout import CHECKOUT
from .models import file_sha256
from .rules import interception_rules

#: The checkout this module runs from; for an installed package, the directory above it,
#: which holds no .git, BUILD_INFO.json or disk guard (provenance then needs
#: --allow-non-main).
ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1]
#: The RUNBOOK commands the day runs, carried by the package.
TOOLS = Path(__file__).resolve().parent / "tools"
DISK_LOW = ROOT / "output" / "DISK_LOW"
#: Under the checkout; for an installed package, relative: under the directory the day
#: is started in, resolved when it starts.
_HOME = CHECKOUT if CHECKOUT is not None else Path()
DEFAULT_FINAL_ROOT = _HOME / "run" / "upload"
DEFAULT_WORK_ROOT = _HOME / "run" / "day"
#: How the help names that place.
_UNDER = "" if CHECKOUT is not None else ", under the directory it is started in"
#: The disk guard's tree: it compresses idle files there (FIX_PREAMBLE).
GUARDED = ROOT / "output"
STATE_FORMAT = "orbitwin-day/1"
#: Beside the two upload files, named so it cannot be taken for one.
NOTE_NAME = "README_NOT_FOR_UPLOAD.md"
#: RUNBOOK step 4a, 6: certify's reruns.
CERTIFY_RUNS = 3
#: The candidates, in the order RUNBOOK step 4a lists them; a tie goes to the first.
#: final-03: rank's best without head-start burns, certified beside a final that burns in one.
CANDIDATES = ("final", "final-01", "final-02", "final-03")
#: Exits of `orbitwin day`: 0 the upload is in place, 2 a stop, 3 paused for a person.
EXIT_STOP = 2
EXIT_PAUSED = 3
#: certify's exit when GmatConsole gave no answer (cli.no_answer): EX_TEMPFAIL, not a verdict.
EXIT_CERTIFY_NO_ANSWER = 75
#: Kept beside state.json and given up for it when the disk fills, so the record is written.
RESERVE_NAME = ".state.reserve"
RESERVE_BYTES = 1 << 20
#: One note per step process while it runs, so a resume can see one a killed day left.
RUNNING = ".running"

_SCORE_FLAGS = ("kt_per_s", "ct_s", "kv_per_m_s", "cv_m_s", "cr_km")
#: Recorded on the first run; a resume takes them from state.json.
_FIXED_FLAGS = ("script", "companion", "horizon_s", "tmax_as_announced", "mission_boundary_km",
                "team_number",
                "closest_approach_window", "pursuer_name", "target_name", "base",
                "gmat_workers", "report_filename_as_written",
                "accept_announced_horizon_before_perigee_s", "work_root", "final_root",
                "gmat_console", "startup_file", "expect_commit", "no_head_start_burns")
#: The readbacks two people confirm, by RUNBOOK step: the configs, and inspect's clock.
READBACKS = ("2", "3", "5")
#: A command's refusal: `orbitwin: ERROR [<code>]: ...`, or the older `orbitwin: ERROR: ...`.
_ERROR_LINE = re.compile(r"orbitwin: ERROR( \[[a-z-]+\])?:")
#: check_report_rows's score lines, a hit's and a scored miss's.
_HIT_SCORE = re.compile(r"^\s*validator reads last row\s+Tteam =\s*\S+ s -> score (\S+)$", re.M)
_MISS_SCORE = re.compile(r"^\s*scored miss: .* = (\S+)$", re.M)


class DayStop(OrbitWinError):
    """A gate that did not pass: nothing is copied and `latest` stays."""

    def __init__(self, message: str, runbook: str) -> None:
        super().__init__(message)
        self.runbook = runbook


class DayPaused(OrbitWinError):
    """A person has to read something first: a readback, or the coefficients to come."""


@dataclasses.dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str
    stderr: str


#: Runs one step: (argv, env, cwd, log stem) -> its outcome. Tests pass their own.
Executor = Callable[[list[str], dict[str, str], Path, Path], Completed]


class _Echo:
    """Lines for the terminal, written by a thread of their own: a terminal that stops
    reading (Ctrl-S, a stalled ssh) holds that thread alone, never the log or the text a
    step's verdict is read from. A real terminal is written by its file descriptor, so a
    write it holds keeps no lock of sys.stderr's that the interpreter needs on exit."""

    def __init__(self, terminal: IO[str]) -> None:
        self.terminal = terminal
        try:
            self.fd: int | None = terminal.fileno()
            terminal.flush()
        except (OSError, ValueError, AttributeError):
            self.fd = None
        self.encoding = getattr(terminal, "encoding", None) or "utf-8"
        self.lines: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.pending = 0
        self.lock = threading.Lock()
        threading.Thread(target=self._write, daemon=True).start()

    def __call__(self, text: str) -> None:
        with self.lock:
            self.pending += 1
        self.lines.put(text)

    def _write(self) -> None:
        while True:
            text = self.lines.get()
            # A lost terminal: the line is dropped, and the log keeps it.
            with suppress(OSError, ValueError):
                if self.fd is None:
                    self.terminal.write(text)
                    self.terminal.flush()
                else:
                    data = text.encode(self.encoding, errors="replace")
                    while data:
                        data = data[os.write(self.fd, data):]
            with self.lock:
                self.pending -= 1

    def drain(self, timeout_s: float) -> None:
        """Wait for the lines so far to reach the terminal, so the day's next lines come
        after them; give up after timeout_s, the terminal not reading."""
        deadline = time.monotonic() + timeout_s
        while self.pending and time.monotonic() < deadline:
            time.sleep(0.01)


def _pump(stream: IO[str], log: IO[str], kept: list[str],
          echo: Callable[[str], None] | None, mark: str, done: threading.Event) -> None:
    try:
        for line in stream:
            kept.append(line)
            # A full disk: the step's output is still read and kept, so it never blocks.
            with suppress(OSError, ValueError):
                log.write(line)
                log.flush()
            if echo is not None:
                echo(mark + line)
    finally:
        done.set()


#: Steps whose output only the day's verdict on it reaches the screen with: step 7's
#: checks, whose refusals the stop names.
_QUIET = ("07-check-", "07-bind-")


#: The day folder's lock, open while the day runs; each step inherits it (run_logged).
_HELD_LOCKS: list[int] = []


#: How long the pumps may read on once a step's process group is gone.
PUMP_WAIT_S = 10.0
#: How long an interrupted step has to end on its own Ctrl-C.
INTERRUPT_WAIT_S = 30.0
#: How long the day waits for a step's lines to reach a terminal that is not reading.
ECHO_WAIT_S = 2.0


def run_logged(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
    """argv, its stdout into <log>.out and its stderr into <log>.err and onto ours, each
    line marked with the step: certify names an upload of its own, which is not the day's.

    cwd is the day folder. The step runs in a session of its own, holds the folder's lock
    and is noted in <cwd>/.running while it runs, so a resume after the day was killed
    finds it (_still_running). Once it exits, whatever it left in its process group (a
    killed solve's workers, holding its pipes and the lock) is killed.

    A Ctrl-C is noted, raised only while the day waits on the step, and raised again once
    the logs are whole: anywhere else it would close them under the pumps or be lost.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    err: list[str] = []
    echo = None if log.name.startswith(_QUIET) else _Echo(sys.stderr)
    ctrl_c = _CtrlC()
    with (ctrl_c.held(),
          log.with_suffix(".out").open("w", encoding="utf-8") as out_log,
          log.with_suffix(".err").open("w", encoding="utf-8") as err_log):
        child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, errors="replace",
                                 pass_fds=tuple(_HELD_LOCKS), start_new_session=True)
        assert child.stdout is not None and child.stderr is not None
        try:
            note = _note_child(cwd / RUNNING, child.pid, log.name, argv)
        except OSError:
            # Unnoted, it could outlive a kill unseen: it does not run.
            _signal_group(child.pid, signal.SIGKILL)
            child.communicate()
            raise
        done = [threading.Event(), threading.Event()]
        pumps = [threading.Thread(target=_pump, args=(child.stdout, out_log, out, None, "",
                                                      done[0]), daemon=True),
                 threading.Thread(target=_pump, args=(child.stderr, err_log, err, echo,
                                                      f"  [{log.name}] ", done[1]),
                                  daemon=True)]
        for pump in pumps:
            pump.start()
        try:
            with ctrl_c.raising():
                # Exited but not reaped: its group id cannot be reused while it is killed.
                os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        except KeyboardInterrupt:
            ctrl_c.came = True
            _interrupt(child, ctrl_c)
        finally:
            _signal_group(child.pid, signal.SIGKILL)
            code = child.wait()
            note.unlink(missing_ok=True)
            for finished in done:
                finished.wait(PUMP_WAIT_S)
            if echo is not None:
                echo.drain(ECHO_WAIT_S)
        # Popen.__del__ swallows a KeyboardInterrupt: let it run while ours is held.
        del child
    if ctrl_c.came:
        raise KeyboardInterrupt
    return Completed(code, "".join(out), "".join(err))


class _CtrlC:
    """SIGINT while a step runs: noted always, raised only inside raising()."""

    def __init__(self) -> None:
        self.came = False
        self._raise = False
        self._unraised = False

    def _handler(self, _signum: int, _frame: Any) -> None:
        self.came = True
        if self._raise:
            self._raise = False
            raise KeyboardInterrupt
        self._unraised = True

    @contextmanager
    def held(self) -> Iterator[None]:
        with _sigint_handler(self._handler):
            yield

    @contextmanager
    def raising(self) -> Iterator[None]:
        self._raise = True
        try:
            if self._unraised:
                self._unraised = False
                raise KeyboardInterrupt
            yield
        finally:
            self._raise = False


@contextmanager
def _sigint_handler(handler: Any) -> Iterator[None]:
    """SIGINT to handler for the block. Only the main thread receives signals, a day started
    with SIGINT ignored keeps it so, and a handler installed outside Python stays."""
    if (threading.current_thread() is not threading.main_thread()
            or signal.getsignal(signal.SIGINT) in (signal.SIG_IGN, None)):
        yield
        return
    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


#: Set by the day's SIGINT handler: a finalizer that swallowed the KeyboardInterrupt
#: (Popen.__del__, a WeakSet callback) would otherwise let the day run on. A plain flag,
#: not an Event: a Ctrl-C held down re-enters the handler, which must take no lock.
_ctrl_c_seen = False


#: Set once the day has begun to stop or finish: from then on a Ctrl-C is noted, not
#: raised, so it cannot cut short what the day says. A plain flag: one store, no lock.
_stopping = False


def _day_ctrl_c(_signum: int, _frame: Any) -> None:
    global _ctrl_c_seen
    _ctrl_c_seen = True
    if not _stopping:
        raise KeyboardInterrupt


def _stop_if_ctrl_c() -> None:
    if _ctrl_c_seen:
        raise KeyboardInterrupt


def _between_ctrl_c_checks(executor: Executor) -> Executor:
    def run(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> Completed:
        _stop_if_ctrl_c()
        done = executor(argv, env, cwd, log)
        _stop_if_ctrl_c()
        return done
    return run


def _signal_group(pgid: int, number: int) -> None:
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, number)


def _interrupt(child: subprocess.Popen[str], ctrl_c: _CtrlC) -> None:
    """Ctrl-C reached the day alone, the step having a session of its own: pass it on, and
    wait; a second Ctrl-C, or INTERRUPT_WAIT_S, and the caller kills the step's group."""
    _signal_group(child.pid, signal.SIGINT)
    deadline = time.monotonic() + INTERRUPT_WAIT_S
    with suppress(KeyboardInterrupt), ctrl_c.raising():
        while time.monotonic() < deadline:
            if os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT):
                return
            time.sleep(0.1)


def _proc_start(pid: int) -> str | None:
    """The boot and the start time of process pid, None if it is gone or a zombie: a pid
    names the same process only while both match."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return None
    # After the command name, which may hold anything: field 3 (state), field 22 (starttime).
    fields = stat.rsplit(")", 1)[-1].split()
    if len(fields) < 20 or fields[0] in ("Z", "X"):
        return None
    return f"{boot}/{fields[19]}"


def _note_child(folder: Path, pid: int, step: str, argv: list[str]) -> Path:
    """pid, a session leader, its start and its group (its pid) while the step runs."""
    folder.mkdir(exist_ok=True)
    note = folder / str(pid)
    note.write_text(json.dumps({
        "pid": pid, "pgid": pid, "start": _proc_start(pid), "step": step, "argv": argv,
        "utc": datetime.datetime.now(datetime.UTC).strftime("%H:%M:%SZ")}) + "\n",
        encoding="utf-8")
    return note


def _processes() -> dict[int, tuple[int, int, int]]:
    """Each live process: (parent, process group, session)."""
    found: dict[int, tuple[int, int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="ascii", errors="replace")
            fields = stat.rsplit(")", 1)[-1].split()
            if fields[0] in ("Z", "X"):
                continue
            found[int(entry.name)] = (int(fields[1]), int(fields[2]), int(fields[3]))
        except (OSError, IndexError, ValueError):
            continue
    return found


def _boot() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return None


def _still_running(folder: Path, tidy: bool) -> list[str]:
    """What the steps of an earlier run of this day folder left running: the step itself
    (its pid and start time), or any process still in its group and session, a killed
    solve's workers among them. With tidy (the lock held), the notes of those that ended
    are removed."""
    running = folder / RUNNING
    if not running.is_dir():
        return []
    table = _processes()
    boot = _boot()
    lines = []
    for path in sorted(running.iterdir()):
        try:
            note = json.loads(path.read_text(encoding="utf-8"))
            pid, start = int(note["pid"]), note["start"]
            pgid = int(note.get("pgid", pid))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        now = _proc_start(pid)
        alive = start is not None and now == start
        if now is not None and not alive:
            # The pid is another process's now: neither it nor its group is the step's.
            if tidy:
                path.unlink(missing_ok=True)
            continue
        group = sorted(other for other, (_, group, session) in table.items()
                       if group == pgid and session == pgid)
        if not alive and not (group and start is not None
                              and str(start).split("/", 1)[0] == boot):
            if tidy:
                path.unlink(missing_ok=True)
            continue
        argv = " ".join(str(word) for word in note.get("argv") or [])
        lines.append(f"{note.get('step')} (started {note.get('utc')}): {argv[:160]}; its "
                     f"processes: {' '.join(map(str, group or [pid]))}; stop them all with: "
                     f"kill -TERM -- -{pgid}")
    return lines


def _utc_stamp() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


def _fresh_folder(root: Path, suffix: str = "") -> Path:
    """root/<UTC><suffix>, never one that exists, nor one whose <UTC> does: a second run in
    the same second waits for the next."""
    for _ in range(3):
        stamp = _utc_stamp()
        folder = root / f"{stamp}{suffix}"
        if suffix and (root / stamp).exists():
            time.sleep(1.0)
            continue
        try:
            folder.mkdir()
        except FileExistsError:
            time.sleep(1.0)
            continue
        return folder
    raise DayStop(f"{root} has a folder for every second just now; try again", "-")


def _alone(command: str) -> str:
    """A command on a line of its own, blank lines around it, to copy whole."""
    return f"\n{command}\n\n"


def _say(message: str) -> None:
    print(f"orbitwin day [{datetime.datetime.now(datetime.UTC):%H:%M:%S}Z]: {message}",
          file=sys.stderr, flush=True)


def _write_json(path: Path, body: object) -> None:
    """Whole or not at all: a crash leaves the previous state.json. A full disk takes the
    reserve beside it (_reserve), so the record of what filled it is still written."""
    text = json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    staged = path.with_name(f".{path.name}.{os.getpid()}")
    try:
        _write_new(staged, text)
    except OSError as exc:
        reserve = path.with_name(RESERVE_NAME)
        if exc.errno != errno.ENOSPC or not reserve.is_file():
            raise
        reserve.unlink()
        _say(f"the disk is full; {reserve} was given up to write {path}")
        _write_new(staged, text)
    try:
        os.replace(staged, path)
    except OSError:
        staged.unlink(missing_ok=True)
        raise


def _write_new(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except OSError:
        path.unlink(missing_ok=True)
        raise


def _reserve(folder: Path) -> None:
    """RESERVE_BYTES of the disk, held beside state.json for _write_json; none if the disk
    has not got them, as check_disk then stops the day anyway."""
    reserve = folder / RESERVE_NAME
    if reserve.is_file() and reserve.stat().st_size >= RESERVE_BYTES:
        return
    try:
        with reserve.open("wb") as handle:
            os.posix_fallocate(handle.fileno(), 0, RESERVE_BYTES)
    except OSError:
        reserve.unlink(missing_ok=True)


def _git(root: Path, *argv: str, timeout_s: float = 30.0) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(root), *argv], capture_output=True, text=True,
                              timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _is_commit(text: str) -> bool:
    return 7 <= len(text) <= 40 and all(char in "0123456789abcdef" for char in text)


def provenance(root: Path, expect: str | None, allow_non_main: bool) -> dict[str, Any]:
    """The commit this runs: the one chosen the day before (RUNBOOK step 0), unchanged.

    Nothing is fetched: on the day origin/main may move, and the day must not.
    A git checkout is judged by HEAD and its local changes; an unpacked archive
    by its BUILD_INFO.json, as `orbitwin --version`.
    """
    notes: list[str] = []
    expect = None if expect is None else expect.strip().lower()
    if expect is not None and not _is_commit(expect):
        raise DayStop(f"--expect-commit {expect!r} is not a commit (7 to 40 hex digits)", "0")
    if CHECKOUT is None:
        # An installed package: the commit direct_url.json names, held to RECORD. A git
        # repository or a BUILD_INFO.json around the install says nothing of what runs.
        package = installed()
        head = None if package is None else package.commit
        record: dict[str, Any] = {
            "source": "installed", "commit": head, "expected": expect,
            "version": None if package is None else package.version,
            "url": None if package is None else package.url,
            "local_changes": [] if package is None else list(package.changed),
            "startup_files": [] if package is None else list(package.startup),
            "compiled_afresh": package is not None and package.fresh,
            "notes": notes}
        why = ("this orbitwin is neither a git checkout nor an unpacked release archive, and "
               "carries no installer's record of itself" if package is None
               else "ORBITWIN_FRESH_PYCACHE is set in the environment, which kept this run "
               "from starting itself with an empty bytecode cache; unset it"
               if not package.fresh and os.environ.get(FRESH_PYCACHE)
               else "this process may have run .pyc files found beside the sources, which no "
               "hash holds; `orbitwin day` and `orbitwin run` start themselves with an empty "
               "bytecode cache to compile afresh" if not package.fresh
               else f"this installed orbitwin {package.version} records no commit (installed "
               f"from {package.url or 'an index'}); install it from git at the chosen commit"
               if head is None
               else f"the installed files differ from what was installed: "
               f"{', '.join(package.changed)}" if package.changed else None)
    elif (root / ".git").exists():
        head = _git(root, "rev-parse", "HEAD")
        changed = _git(root, "status", "--porcelain", "--untracked-files=no")
        # A stray module is a change too, where the commands are read from.
        stray = _git(root, "status", "--porcelain", "--untracked-files=all", "--",
                     "src", "scripts", "config")
        dirty = sorted({*(changed or "").splitlines(), *(stray or "").splitlines()})
        record = {"source": "git", "commit": head, "expected": expect,
                                  "local_changes": dirty, "notes": notes}
        local = None if head is None or changed is None else ""
        why = ("git cannot read this checkout" if local is None
               else f"the checkout has local changes: {', '.join(dirty)}" if dirty else None)
    else:
        from .cli import describe_build

        line = describe_build()
        info = root / "BUILD_INFO.json"
        head = None
        if info.is_file():
            try:
                head = str(json.loads(info.read_text(encoding="utf-8"))["commit"]).lower()
            except (OSError, ValueError, KeyError, TypeError):
                head = None
        altered = None if head is None else _release_changes(root)
        record = {"source": "BUILD_INFO.json", "commit": head, "expected": expect,
                  "describe": line, "local_changes": altered or [], "notes": notes}
        why = ("this is neither a git checkout nor an unpacked release archive" if head is None
               else "the archive was not built from origin/main" if "NOT origin/main" in line
               else "the archive was built with uncommitted changes" if "uncommitted" in line
               else f"the release tree differs from its {RELEASE_MANIFEST}: "
               f"{', '.join(altered)}" if altered else None)
    if why is None and head is not None:
        why = ("no --expect-commit names the commit chosen the day before (RUNBOOK step 0)"
               if expect is None
               else f"this is commit {head[:12]}, not the expected {expect[:12]}"
               if not head.startswith(expect) else None)
    record["on_main"] = why is None
    if why is not None:
        if not allow_non_main:
            raise DayStop(f"{why}. The day runs the commit written down the day before; check "
                          "it out, or pass --allow-non-main to run this one anyway, recorded "
                          "beside the upload", "0")
        notes.append(f"--allow-non-main: {why}")
    return record


#: An unpacked release's manifest, and the parts of it the day's commands come from.
RELEASE_MANIFEST = "SHA256SUMS.txt"
_RELEASE_CODE = ("src", "scripts", "config")
#: What a release tree may hold there besides its files (check_release_tree.py's).
_RELEASE_CACHES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})


def _release_changes(root: Path) -> list[str]:
    """Files under src, scripts and config that are not the release's: changed, missing,
    or not in its SHA256SUMS.txt (a hand edit on the day, an older tree unzipped under)."""
    manifest = root / RELEASE_MANIFEST
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [f"{RELEASE_MANIFEST} (missing)"]
    listed = {}
    for line in lines:
        digest, _, name = line.partition("  ")
        if name.split("/", 1)[0] in _RELEASE_CODE:
            listed[name] = digest
    changes = [f"{name} ({'missing' if not (root / name).is_file() else 'changed'})"
               for name, digest in sorted(listed.items())
               if not (root / name).is_file() or file_sha256(root / name) != digest]
    for top in _RELEASE_CODE:
        for directory, subdirectories, files in os.walk(root / top):
            subdirectories[:] = sorted(name for name in subdirectories
                                       if name not in _RELEASE_CACHES
                                       and not name.endswith(".egg-info"))
            for name in files:
                relative = (Path(directory) / name).relative_to(root).as_posix()
                if relative not in listed and not name.endswith((".pyc", ".pyo")):
                    changes.append(f"{relative} (not in the release)")
    return sorted(changes)


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _disk_low_files() -> list[Path]:
    """The disk guard's flag: output/DISK_LOW beside this checkout, and the one
    $ORBITWIN_DISK_LOW_FILE names (a release tree's own output/ is not the guard's)."""
    named = os.environ.get("ORBITWIN_DISK_LOW_FILE")
    return [DISK_LOW, *([Path(named)] if named else [])]


def check_disk(paths: Sequence[Path], min_free_gb: float, resume: str | None = None) -> None:
    """output/DISK_LOW and the free space where the day writes (FIX_PREAMBLE)."""
    then = "" if resume is None else f", then run:\n\n{resume}\n"
    for flag in _disk_low_files():
        if flag.exists():
            raise DayStop(f"{flag} exists: the disk guard found the disk short; free space "
                          f"first{then}", "0")
    for path in paths:
        free_gb = _free_bytes(path) / 2**30
        if free_gb < min_free_gb:
            raise DayStop(f"{free_gb:.1f} GiB free under {path}, under --min-free-gb "
                          f"{min_free_gb:g}; free space first{then}", "0")


def _gmat_pair(console: str | None, startup: str | None) -> tuple[Path, Path]:
    """Both from the flags or both from the environment, as the other tools take them."""
    if (console is None) != (startup is None):
        raise DayStop("--gmat-console and --startup-file must be given together; half from "
                      "the environment could be another GMAT's", "0")
    if console is None:
        console = os.environ.get("GMAT_CONSOLE") or None
        startup = os.environ.get("GMAT_STARTUP") or None
        if (console is None) != (startup is None):
            raise DayStop("GMAT_CONSOLE and GMAT_STARTUP must be set together", "0")
    if console is None or startup is None:
        raise DayStop("give --gmat-console and --startup-file, or export GMAT_CONSOLE and "
                      "GMAT_STARTUP: every step of the day flies GMAT", "0")
    # Absolute, not resolved: the RUNBOOK's commands name GmatConsole as given.
    pair = Path(console).expanduser().absolute(), Path(startup).expanduser().absolute()
    for path in pair:
        if not path.is_file():
            raise DayStop(f"{path} is not a file", "0")
    return pair


def _check_final_root(final_root: Path) -> None:
    """A folder, or nothing yet, with latest a symlink if anything."""
    if final_root.exists() and not final_root.is_dir():
        raise DayStop(f"{final_root} is not a folder; move it away or name another "
                      "--final-root", "-")
    latest = final_root / "latest"
    if latest.exists() and not latest.is_symlink():
        raise DayStop(f"{latest} is not a symlink; move it away first", "-")


def _outside_the_guard(path: Path, flag: str) -> None:
    """The disk guard compresses idle files under output/, a submission among them."""
    if path == GUARDED or GUARDED in path.parents:
        raise DayStop(f"{flag} {path} is under {GUARDED}, whose idle files the disk guard "
                      "compresses; name a folder outside it", "-")


def _team_given(team_number: int | None, base: Path, *, named: bool) -> None:
    """What step 2 would refuse about the team number, said before a day folder is made,
    since a resume cannot change --team-number or --base. A --base that will not load
    stops here too; the default one is left to step 2."""
    try:
        written = load_config(base).policy.team_number
    except (OSError, OrbitWinError) as exc:
        if named:
            raise DayStop(f"--base {base}: {exc}", "2") from None
        return
    if team_number is None and written is None:
        raise DayStop(f"{base} sets no policy.team_number: pass --team-number, which the "
                      "official report's file name needs", "2")
    if team_number is not None and written is not None and written != team_number:
        raise DayStop(f"--team-number {team_number}, but {base} sets policy.team_number "
                      f"{written}; give one of them", "2")


def _latest_note(final_root: Path) -> str:
    """Where latest points now, from the record beside its two files."""
    latest = final_root / "latest"
    if not latest.is_symlink():
        return f"{latest} does not exist"
    target = final_root / os.readlink(latest)
    note = next((path for path in (target / NOTE_NAME, target / "SUMMARY.txt")
                 if path.is_file()), None)
    said: dict[str, str] = {}
    if note is not None:
        for line in note.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(" : ")
            said.setdefault(key.strip(), value.strip())
    return (f"{latest} still points at {target} (T_max {said.get('T_max', 'unknown')}; "
            f"official {said.get('official', 'unknown')}; written "
            f"{said.get('written', os.readlink(latest))})")


def _relative_companion(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise DayStop(f"--companion {name!r} must be a path relative to the official "
                      "script's folder, without '..' (RUNBOOK step 1)", "1")
    return path


def _as_announced(horizon_s: float, boundary_km: float | None) -> list[re.Pattern[str]]:
    """The two lines ranking_day_config.py prints for a T_max taken as announced: the
    parser's offset from the boundary (mission_clock.horizon_warning) and the taking."""
    if boundary_km is None:
        return []
    boundary = re.escape(f"{boundary_km!r} km")
    return [
        re.compile(re.escape(f"WARNING: T_max {horizon_s!r} s puts SC_Alien ")
                   + r"\S+ km from the centre, [+-]\S+ km from the announced boundary "
                   + boundary + r" \(tolerance \S+ km, two-body\), (?:not )?climbing "
                   r"[+-]\S+ km/s: the rules put it at the boundary, climbing\. Check T_max, "
                   r"the boundary and the time origin against the announcement$"),
        re.compile(re.escape(f"WARNING: T_max {horizon_s!r} s taken as announced "
                             "(--tmax-as-announced): SC_Alien is ")
                   + r"[+-]\S+ km off the " + boundary + r" boundary there \([^()]*\)$"),
    ]


def _readback_warnings(stdout: str, as_announced: list[re.Pattern[str]]) -> list[str]:
    """The readback's WARNING and NOTE lines a person has to read (RUNBOOK step 5), less
    the pair the flag asked for."""
    warnings = [line for line in stdout.splitlines() if line.startswith(("WARNING:", "NOTE:"))]
    asked = [line for line in warnings for pattern in as_announced if pattern.match(line)]
    # Both or neither: the offset line alone is a T_max the flag did not take.
    if len(as_announced) == 2 and len(asked) == 2 and len(set(asked)) == 2:
        return [line for line in warnings if line not in asked]
    return warnings


def _json(path: Path, runbook: str) -> dict[str, Any]:
    """A JSON object a step wrote, or a stop naming the step."""
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DayStop(f"{path} is unreadable: {exc}", runbook) from None
    if not isinstance(body, dict):
        raise DayStop(f"{path} is not a JSON object", runbook)
    return body


def _inspect_readback(stderr: str, asked: list[re.Pattern[str]]) -> dict[str, Any]:
    """What inspect says a person checks at step 3, and the WARNING lines among it, less
    the boundary offset a T_max taken as announced makes (the same line step 2 printed)."""
    lines = [line for line in stderr.splitlines() if line.startswith("orbitwin")]
    warnings = [line for line in lines if "WARNING:" in line and not any(
        pattern.match(line.removeprefix("orbitwin inspect: ")) for pattern in asked)]
    return {"readback": "\n".join(lines) + "\n", "warnings": warnings}


def _km(value: object) -> str:
    return f"{value:.6f} km" if isinstance(value, int | float) else f"{value} km"


def step7_score(stdout: str) -> float | None:
    """The score step 7 printed: a hit's on its last row, a miss's on its closest."""
    found = _HIT_SCORE.findall(stdout) or _MISS_SCORE.findall(stdout)
    if len(found) != 1:
        return None
    try:
        value = float(found[0])
    except ValueError:
        return None
    return value if math.isfinite(value) else None


#: rank's warnings that every archive can get, shown without stopping the day; any other
#: (rank._unit_warning, _distance_warning, a coefficient of zero never tested, another
#: score block, team number, scenario fingerprint, no GMAT run) stops it.
_RANK_SHOWN = tuple(re.compile(pattern) for pattern in (
    # rank's sensitivity of the winner to a 10% change of a coefficient.
    r"The winner changes when .* moves by \d+%\. Re-read the announced value",
    # A file caps every loaded stage at renderable: rank says so of every archive.
    r"The top-scoring candidate stands at stage \w+",
    r"GMAT-verified hits stopped at another depth than policy\.stop_depth_km",
    r"The top-scoring candidate \S+ could not be submitted as it stands",
    r"No candidate in this archive passes the submission gates",
    r"The best submittable candidate is \S+ at ",
    r"The top-scoring candidate \S+ is a shallow hit",
    # solve's own GMAT failures, as solve printed them.
    r"\d+ GMAT correction\(s\) or verification\(s\) raised",
    # rank.head_start_line: whether best_submittable burns in the head start.
    r"head start: ",
))
#: certify's refusal, naming where it set the run aside (verify._set_aside).
_KEPT = re.compile(r"This certification is kept in (/\S+?)(?: \(|,| and )")
#: check_report_rows's last row, as step 7 prints it.
_LAST_ROW = re.compile(r"^LAST row\s*: t = (\S+?)\s*s?\s{2,}", re.M)
#: check_report_rows's flag for a report that cannot show a burn (cli._END_ONLY).
_END_ONLY = "--accept-end-time-binding-only"


@dataclasses.dataclass
class Problem:
    """Why a candidate is not uploaded as it stands, and where the RUNBOOK takes it."""

    candidate: str
    what: str
    runbook: str
    end_time_only: bool = False
    #: certify gave no verdict on it (no answer, or cut short): a resume runs it again.
    unfinished: bool = False
    #: certify refused it: its verdict, which --include alone runs again.
    refused: bool = False


#: certification.json's binding_pre_check result: (what it means, the RUNBOOK step).
_PRECHECK_OUTCOMES = {
    "failed": ("PRE-CHECK FAILED: do not upload it; take the next fallback", "6, then 4a"),
    "pass-end-time-only": ("PRE-CHECK PASS (end time only): the 6b.7 by-eye comparison is "
                           "mandatory", "6b.7"),
    # The official report carries no target position (RUNBOOK step 3): step 7 judges it.
    "pass-target-not-in-report": ("PRE-CHECK PASS (target not in report): d_min and the "
                                  "interception were proven on certify's diagnostic flight; "
                                  "ask the organizer how its validator computes d_min",
                                  "3, 6 and 7"),
    "pass-target-not-in-report-end-time-only": (
        "PRE-CHECK PASS (target not in report; end time only): d_min and the interception "
        "were proven on certify's diagnostic flight, and the 6b.7 by-eye comparison is "
        "mandatory", "6b.7"),
    "surface-not-judged": ("PRE-CHECK SURFACE NOT JUDGED: read the lowest altitude off GMAT "
                           "by eye, or ask the organizer", "6 and 7"),
    "surface-template-below": ("PRE-CHECK SURFACE TEMPLATE BELOW: ask the organizer",
                               "6 and 7"),
    "surface-undecided": ("PRE-CHECK SURFACE UNDECIDED: read GMAT's Earth.Altitude by eye, "
                          "or ask the organizer", "6 and 7"),
    "not run": ("PRE-CHECK NOT RUN: run step 6's pre-check by hand", "6"),
    "skipped": ("PRE-CHECK SKIPPED: the console is orbitwin's test double, never the day's",
                "6"),
}
#: check_report_rows exits other than 0: (what it means, the RUNBOOK step).
_STEP7_OUTCOMES = {
    1: ("step 7 exit 1: a VERDICT failed or the report could not be read; not uploaded, "
        "take the next fallback", "7, then 4a"),
    2: ("step 7 exit 2: refused or disqualified; not uploaded, take the next fallback",
        "7, then 4a"),
    3: ("step 7 exit 3: burns not bound; upload only after the 6b.7 by-eye comparison",
        "6b.7"),
    4: ("step 7 exit 4: SURFACE NOT JUDGED; a person reads the lowest altitude off GMAT",
        "7"),
    5: ("step 7 exit 5: the template's initial state is below the surface; ask the "
        "organizer", "7"),
    6: ("step 7 exit 6: SURFACE UNDECIDED; read GMAT's Earth.Altitude by eye", "7"),
    7: ("step 7 exit 7: NOT JUDGED BY HAND, a report without the target's position; only the "
        "--certificate form on certify's report judges it", "7"),
}


@dataclasses.dataclass
class Checked:
    """One certified candidate after step 7."""

    candidate: str
    certificate: Path
    exit_code: int
    score: float | None
    precheck: str
    #: Its certified plan burns before the later epoch.
    head_start: bool = False


class Day:
    """One day folder: its state, its steps and their gates."""

    def __init__(self, folder: Path, state: dict[str, Any], executor: Executor,
                 args: argparse.Namespace) -> None:
        self.folder = folder
        self.state = state
        self.executor = _between_ctrl_c_checks(executor)
        self.accept_config_warnings = bool(args.accept_config_warnings)
        self.accept_rank_warnings = bool(args.accept_rank_warnings)
        self.confirm_readback = set(args.confirm_readback or ())
        self.force_latest = bool(args.force_latest)
        self.retry = set(args.include or ())
        self.by_eye = list(args.by_eye_passed or ())
        #: What a delivery cut short left in the final root, named on the way out.
        self.left: Path | None = None
        #: A proven upload made by this run, whether or not latest moved to it.
        self.made: Path | None = None
        #: What the day was doing, for a failure nothing else names.
        self.doing = "starting"
        self.run_dir = folder / "run"
        self.logs = folder / "logs"
        self.official = folder / "input" / "official.script"
        self.bin = folder / ".bin"
        #: This run's bytecode, emptied when it starts.
        self.pycache = folder / ".pycache"

    # -- bookkeeping ----------------------------------------------------------------

    def save(self) -> None:
        _write_json(self.folder / "state.json", self.state)

    @property
    def resume(self) -> str:
        """The command a stop prints, as this day folder must be resumed."""
        return _resume_command(self.folder, bool(self.state.get("allow_non_main")))

    def minutes_so_far(self) -> str:
        seconds = sum(float(record.get("seconds") or 0.0) for record in self.steps.values())
        return f"about {max(1, math.ceil(seconds / 60.0))} min"

    @property
    def inputs(self) -> dict[str, Any]:
        return self.state["inputs"]  # type: ignore[no-any-return]

    @property
    def steps(self) -> dict[str, dict[str, Any]]:
        return self.state["steps"]  # type: ignore[no-any-return]

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.folder).as_posix()

    def _hashes(self, paths: Sequence[Path]) -> dict[str, str]:
        return {self._rel(path): file_sha256(path) for path in paths}

    def verified(self, name: str) -> dict[str, Any] | None:
        """The step's record if it ended 0 and what it wrote is unchanged; raises on a change."""
        record = self.steps.get(name)
        if record is None or record.get("exit") != 0:
            return None
        for rel, digest in record["outputs"].items():
            path = self.folder / rel
            if not path.is_file() or file_sha256(path) != digest:
                raise DayStop(f"{path} is not what step {name} wrote (state.json records "
                              f"SHA-256 {digest}); this day folder cannot be resumed: start a "
                              "new one", "-")
        return record

    def environment(self) -> dict[str, str]:
        """Our own environment, with the day's GMAT pair and none from BASH_ENV.

        BASH_ENV can export another GMAT pair into competition_day.sh's bash.
        """
        gmat = self.state["gmat"]
        env = {key: value for key, value in os.environ.items()
               if key not in {"BASH_ENV", "GMAT_CONSOLE", "GMAT_STARTUP", "PYTHONPATH",
                              "ORBITWIN_FRESH_PYCACHE"}}
        # PYTHONSAFEPATH: the day folder, each step's cwd, is not put first on sys.path.
        # PYTHONPYCACHEPREFIX: bytecode compiled afresh from the sources provenance held,
        # never a .pyc found beside them.
        env.update(GMAT_CONSOLE=gmat["console"], GMAT_STARTUP=gmat["startup"],
                   PYTHONPATH=str(SRC), PYTHONSAFEPATH="1",
                   PYTHONPYCACHEPREFIX=str(self.pycache),
                   PATH=f"{self.bin}{os.pathsep}{env.get('PATH', '')}")
        return env

    def _python_shim(self) -> None:
        """`python` for competition_day.sh: this interpreter, its venv kept (not a symlink)."""
        self.bin.mkdir(exist_ok=True)
        shim = self.bin / "python"
        shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
        shim.chmod(0o755)

    def _set_aside(self, path: Path) -> None:
        """What an interrupted run of a step left, kept beside under a new name."""
        if not (path.exists() or path.is_symlink()):
            return
        number = 1
        while True:
            aside = path.with_name(f"{path.name}.interrupted" + ("" if number == 1
                                                                 else f"-{number}"))
            if not (aside.exists() or aside.is_symlink()):
                break
            number += 1
        path.rename(aside)
        _say(f"{path} was left by an earlier run of this step; kept as {aside}")

    def step(self, name: str, runbook: str, argv: list[str], *, leftovers: Sequence[Path] = (),
             outputs: Callable[[Completed], Sequence[Path]] = lambda _: (),
             data: Callable[[Completed], dict[str, Any]] = lambda _: {}
             ) -> tuple[dict[str, Any], bool]:
        """Run `argv` unless state.json holds it done; (its record, whether it ran now)."""
        done = self.verified(name)
        if done is not None:
            _say(f"step {runbook} ({name}): done earlier with exit 0, its files unchanged; "
                 "skipped")
            return done, False
        self.doing = f"step {runbook} ({name})"
        check_disk([self.folder], self.state["min_free_gb"], self.resume)
        for path in leftovers:
            self._set_aside(path)
        _say(f"step {runbook} ({name}): {' '.join(argv)}")
        started = time.monotonic()
        completed = self.executor(argv, self.environment(), self.folder, self.logs / name)
        record: dict[str, Any] = {"argv": argv, "exit": completed.returncode,
                                  "seconds": round(time.monotonic() - started, 3),
                                  "outputs": {}, "data": {},
                                  "run": len(self.state["runs"]) - 1}
        if completed.returncode == 0:
            written = list(outputs(completed))
            missing = [path for path in written if not path.is_file()]
            if missing:
                record["exit"] = None
                self.steps[name] = record
                self.save()
                raise DayStop(f"step {name} exited 0 but wrote no {missing[0]}", runbook)
            record["outputs"] = self._hashes(written)
            record["data"] = data(completed)
        record["last_line"] = (completed.stderr.strip().splitlines() or [""])[-1]
        if completed.returncode != 0:
            # What orbitwin said on the way out, for the stop to name (certify's GUI stop).
            record["said"] = [line for line in completed.stderr.splitlines()
                              if line.startswith("orbitwin")][-20:]
        self.steps[name] = record
        self.save()
        _say(f"step {runbook} ({name}): exit {completed.returncode} in {record['seconds']:g} s "
             f"(log {self.logs / name}.err)")
        return record, True

    def unpass(self, names: Sequence[str], why: str) -> None:
        """A gate judged after the step ended 0: the step did not pass, and a resume runs
        it again rather than stopping on the same line."""
        for name in names:
            record = self.steps.get(name)
            if record is not None:
                record["exit"] = None
                record["gate"] = why
        self.save()

    def must(self, record: dict[str, Any], name: str, runbook: str, why: str = "") -> None:
        if record["exit"] != 0:
            raise DayStop(f"step {name} exited {record['exit']}: {record['last_line'] or why}; "
                          f"its log is {self.logs / name}.err. Once its cause is dealt with, "
                          f"a resume runs it again:\n\n{self.resume}\n", runbook)

    # -- commands -------------------------------------------------------------------

    def _gmat_flags(self) -> list[str]:
        gmat = self.state["gmat"]
        return ["--gmat-console", gmat["console"], "--startup-file", gmat["startup"]]

    def _orbitwin(self, *argv: str) -> list[str]:
        return [sys.executable, "-m", "orbitwin", *argv]

    def _script(self, name: str, *argv: str) -> list[str]:
        return [sys.executable, "-m", f"orbitwin.tools.{Path(name).stem}", *argv]

    # -- the steps ------------------------------------------------------------------

    def preserve(self) -> None:
        """Step 1: the official script and its files, hashed and made read-only."""
        record = self.state["official"]
        for rel, digest in {"input/official.script": record["sha256"],
                            **{f"input/{name}": sha for name, sha
                               in record["companions"].items()},
                            **self.state.get("announcement_files", {})}.items():
            path = self.folder / rel
            if not path.is_file() or file_sha256(path) != digest:
                raise DayStop(f"{path} is not the copy step 1 made (SHA-256 {digest}); start a "
                              "new day folder", "1")

    def config(self, name: str, runbook: str, output: Path, extra: list[str],
               zero_burn_km: float | None = None) -> Path:
        inputs = self.inputs
        entries = inputs.get("announcement")
        given = (["--horizon-s", repr(inputs["horizon_s"])] if not entries else
                 [] if "--announcement" in extra else
                 [arg for rel in entries for arg in ("--announcement", str(self.folder / rel))])
        argv = self._script("ranking_day_config.py", "--script", str(self.official),
                            "--output", str(output), *given, *self._gmat_flags(), *extra)
        if zero_burn_km is not None:
            argv += ["--zero-burn-dmin-km", repr(zero_burn_km)]
        asked = (_as_announced(inputs["horizon_s"], inputs["mission_boundary_km"])
                 if inputs["tmax_as_announced"] else [])
        record, ran = self.step(
            name, runbook, argv, leftovers=[output], outputs=lambda _: [output],
            data=lambda done: {"warnings": _readback_warnings(done.stdout, asked),
                               "readback": done.stdout})
        if record["exit"] != 0:
            window = "policy.closest_approach_window" in record["last_line"] or (
                "closest_approach_window" in self._log_text(name))
            if window and inputs["closest_approach_window"] is None:
                raise DayStop(
                    "the craft start on different epochs, and which rows d_min counts is still "
                    "unanswered: orbitwin day does not guess it. Start again with "
                    "--closest-approach-window after_alignment or --closest-approach-window "
                    "from_origin, as a person decides and writes down "
                    "(docs/ASYNC_START_AND_INSERTION.md, \"The window\")", "2")
        self.must(record, name, runbook)
        self.readback_gate(name, runbook, str(output), file_sha256(output), record, ran,
                           "config")
        return output

    def readback_gate(self, name: str, runbook: str, what: str, digest: str,
                      record: dict[str, Any], ran: bool, hashed: str) -> None:
        """RUNBOOK steps 2, 3 and 5: two people read the readback before anything uses it.

        Only what this day folder already printed at a stop can be accepted: a
        WARNING or NOTE line, and the readback by its SHA-256.
        """
        confirmed = self.state.setdefault("confirmed_readbacks", {})
        warnings = record["data"].get("warnings") or []
        shown_warnings = self.state.setdefault("shown_config_warnings", {})
        shown = self.state.setdefault("shown_readbacks", {})
        accepted = self.state.setdefault("accepted_config_warnings", {})
        warned = bool(warnings) and accepted.get(name) != warnings and not (
            self.accept_config_warnings and shown_warnings.get(name) == warnings)
        done = (confirmed.get(runbook) or {}).get("sha256") == digest
        agreed = runbook in self.confirm_readback and shown.get(runbook) == digest
        if ran or warned or not (done or agreed):
            self.echo_readback(name, what, record)
        if warned or not (done or agreed):
            shown_warnings[name] = warnings
            shown[runbook] = digest
            self.save()
        if warned:
            raise DayStop(
                f"the readback of {what} printed {len(warnings)} WARNING or NOTE line(s):\n  "
                + "\n  ".join(warnings)
                + f"\nTwo people read them, above and in {self.logs / name}, against the "
                "announcement; the tool exits 0 even so. If they are as announced, run "
                f"{self.resume} --accept-config-warnings "
                f"--confirm-readback {runbook}", runbook)
        if warnings:
            accepted[name] = warnings
        if not done:
            if not agreed:
                raise DayPaused(
                    f"step {runbook}'s readback of {what} is above,\n"
                    f"and in {self.logs / name}.out and .err.\n"
                    "Two people check every value against the announcement "
                    f"(RUNBOOK step {runbook}), then run:\n"
                    + _alone(f"{self.resume} --confirm-readback {runbook}")
                    + "If a value is wrong, start a new day folder with the corrected input;\n"
                    f"the steps done so far run again ({self.minutes_so_far()} here).")
            confirmed[runbook] = {"sha256": digest, "run": len(self.state["runs"]) - 1,
                                  "utc": _utc_stamp(), "of": hashed}
            _say(f"step {runbook}'s readback confirmed (--confirm-readback {runbook})")
        self.save()

    def echo_readback(self, name: str, what: str, record: dict[str, Any]) -> None:
        text = str(record["data"].get("readback", ""))
        print(f"---- readback of {what} ({name}) ----\n{text.rstrip()}\n---- end of "
              "readback ----", file=sys.stderr, flush=True)

    def _log_text(self, name: str, suffix: str = ".err") -> str:
        try:
            return (self.logs / f"{name}{suffix}").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def day_config(self) -> Path:
        """Step 2."""
        inputs = self.inputs
        extra: list[str] = []
        if inputs["base"] is not None:
            extra += ["--base", inputs["base"]]
        typed = () if inputs.get("announcement") else (
            ("--mission-boundary-km", "mission_boundary_km"),)
        for flag, key in (*typed,
                          ("--closest-approach-window", "closest_approach_window"),
                          ("--team-number", "team_number"),
                          ("--pursuer-name", "pursuer_name"), ("--target-name", "target_name"),
                          ("--gmat-workers", "gmat_workers"),
                          ("--report-filename-as-written", "report_filename_as_written"),
                          ("--accept-announced-horizon-before-perigee-s",
                           "accept_announced_horizon_before_perigee_s")):
            value = inputs[key]
            if value is not None:
                extra += [flag, value if isinstance(value, str) else repr(value)]
        if inputs["tmax_as_announced"]:
            extra.append("--tmax-as-announced")
        if inputs.get("no_head_start_burns"):
            extra.append("--no-head-start-burns")
        return self.config("02-day-config", "2", self.run_dir / "day.yaml", extra)

    def inspect_and_baseline(self, config: Path) -> None:
        """Step 3."""
        inputs = self.inputs
        asked = (_as_announced(inputs["horizon_s"], inputs["mission_boundary_km"])[:1]
                 if inputs["tmax_as_announced"] else [])
        record, ran = self.step(
            "03-inspect", "3", self._orbitwin(
                "inspect", str(self.official), "--config", str(config), *self._gmat_flags(),
                "--output", str(self.run_dir / "scenario.json")),
            data=lambda done: _inspect_readback(done.stderr, asked))
        self.must(record, "03-inspect", "3")
        # The craft `mission clock:` names, the T_max check: against the announcement.
        text = str(record["data"].get("readback", ""))
        self.readback_gate("03-inspect", "3", "inspect's mission clock and T_max check",
                           hashlib.sha256(text.encode("utf-8")).hexdigest(), record, ran,
                           "the lines shown")
        baseline = self.run_dir / "baseline"
        record, _ = self.step(
            "03-baseline", "3", self._orbitwin(
                "baseline", str(self.official), "--config", str(config), *self._gmat_flags(),
                "--output", str(baseline)),
            leftovers=[baseline], outputs=lambda _: [baseline / "baseline.json"])
        self.must(record, "03-baseline", "3 (Stop rules)")

    def search(self, name: str, runbook: str, config: Path) -> Path:
        """Step 4 or 5: competition_day.sh into a new search-NN; returns it."""
        latest = self.run_dir / "latest"

        def searched(_: Completed) -> dict[str, Any]:
            return {"search": os.readlink(latest)}

        def written(_: Completed) -> list[Path]:
            if not latest.is_symlink():
                return [latest / "summary.json"]
            folder = self.run_dir / os.readlink(latest)
            return [folder / "summary.json", folder / "candidates.json"]

        self._python_shim()
        record, _ = self.step(name, runbook, [
            "bash", str(TOOLS / "competition_day.sh"), str(self.official),
            str(self.run_dir), str(config)], outputs=written, data=searched)
        self.must(record, name, runbook)
        return self.run_dir / str(record["data"]["search"])

    def certify(self, name: str, runbook: str, candidate: str, config: Path,
                search: Path) -> None:
        output = self.run_dir / name
        earlier = self.steps.get(f"certify-{name}") or {}
        # A refusal is certify's verdict, the same each time; no answer from GMAT, or an
        # interrupt, is not one.
        if earlier.get("exit") == 2 and name not in self.retry:
            _say(f"certify-{name} refused earlier (exit 2); not run again. --include {name} "
                 "runs it again")
            return
        ran = False
        for attempt in (1, 2):
            record, now = self.step(
                f"certify-{name}", runbook, self._orbitwin(
                    "certify", str(self.official), "--config", str(config), "--candidates",
                    str(search / "candidates.json"), "--candidate", candidate,
                    *self._gmat_flags(), "--runs", str(CERTIFY_RUNS), "--output", str(output)),
                leftovers=[output],
                outputs=lambda _: [*sorted(path for path in (output / "submission").rglob("*")
                                           if path.is_file()),
                                   output / "diagnostic" / "verification.json"],
                data=lambda _: {"candidate": candidate, "config": self._rel(config),
                                "search": self._rel(search)})
            ran = ran or now
            if not now or record["exit"] != EXIT_CERTIFY_NO_ANSWER or attempt == 2:
                break
            _say(f"certify-{name} got no answer from GmatConsole (a crash, a timeout or no "
                 "report), which is no verdict on the candidate; running it again, once")
            for suffix in (".out", ".err"):
                self._set_aside(self.logs / f"certify-{name}{suffix}")
        if ran and (self.state.get("by_eye_passed") or {}).pop(name, None) is not None:
            # A by-eye pass was of the report that certify run wrote, not this one.
            self.save()

    def safety_nets(self, search: Path, config: Path) -> list[str]:
        """Step 4a: certify search-01's best, and its closest verified miss when named."""
        summary = _json(search / "summary.json", "4a")
        if summary.get("best_candidate_is_gmat_verified") is not True:
            self.unpass(["04-search"], "verified nothing in GMAT")
            raise DayStop(f"{search.name} verified nothing in GMAT; check GMAT_CONSOLE and "
                          "GMAT_STARTUP. A resume searches again with this day folder's GMAT "
                          "pair; for another pair, start a new day folder", "4a")
        first = summary.get("best_certifiable_without_shallow_opt_in")
        if summary.get("best_certifiable_uses_head_start") and summary.get(
                "best_certifiable_without_head_start") is not None:
            # Before the coefficients nothing prices the head start's gain: the net burns after it.
            _say(f"{search.name}'s best {first} burns in the head start; final-01 is the best "
                 "without head-start burns (policy.head_start_min_gain_points needs a score)")
            first = summary["best_certifiable_without_head_start"]
        third = summary.get("closest_certifiable_miss")
        nets = []
        if first is None:
            _say(f"{search.name}'s best is a shallow hit and no other verified row is "
                 "certifiable without the opt-in: no final-01 (RUNBOOK step 4a, fallback 4)")
        else:
            nets.append(("final-01", str(first)))
        if third is not None and third != first:
            nets.append(("final-02", str(third)))
        names = []
        for name, candidate in nets:
            if name in self.state["excluded"]:
                _say(f"{name} is excluded (--exclude); not certified")
                continue
            self.certify(name, "4a", candidate, config, search)
            names.append(name)
        return names

    def scored_config(self, search: Path, day: Path) -> Path:
        """Step 5's config: step 2's plus the coefficients."""
        score = self.inputs["score"]
        summary = _json(search / "summary.json", "5")
        zero = summary.get("zero_burn_closest_approach") or {}
        km = zero.get("km")
        # Optional for the readback; a zero burn that touches gives no line.
        zero_km = float(km) if isinstance(km, int | float) and km > 0 else None
        entries = self.inputs.get("score_announcement")
        extra = ([arg for rel in entries for arg in ("--announcement", str(self.folder / rel))]
                 if entries else
                 ["--kt-per-s", repr(score["kt_per_s"]), "--ct-s", repr(score["ct_s"]),
                  "--kv", repr(score["kv_per_m_s"]), "--cv", repr(score["cv_m_s"]),
                  "--delta-v-unit", "m/s", "--cr-km", repr(score["cr_km"])])
        extra += ["--day-config", str(day)]
        if self.inputs["gmat_workers"] is not None:
            extra += ["--gmat-workers", repr(self.inputs["gmat_workers"])]
        return self.config("05-scored-config", "5", self.run_dir / "day-scored.yaml", extra,
                           zero_km)

    def rank(self, config: Path, search: Path) -> str | None:
        """Step 5's rank: best_submittable, or None when nothing is."""
        output = self.run_dir / "rank-scored.json"
        record, _ = self.step(
            "05-rank", "5", self._orbitwin(
                "rank", str(self.official), "--config", str(config), "--candidates",
                str(search / "candidates.json"), "--top", "10", "--output", str(output),
                *self._gmat_flags()),
            leftovers=[output], outputs=lambda _: [output],
            data=lambda done: {"stderr_warnings": [
                line.removeprefix("orbitwin rank: ") for line in done.stderr.splitlines()
                if line.startswith("orbitwin rank: ")]})
        self.must(record, "05-rank", "5")
        report = _json(output, "5")
        note = report.get("best_submittable_note")
        if note is not None:
            self.unpass(["05-search", "05-rank"], str(note))
            raise DayStop(f"rank: {note}. A resume searches again under day-scored.yaml and "
                          "ranks that archive", "5")
        self.rank_gate(report, record)
        best = (report.get("ceilings") or {}).get("best_submittable")
        head_start = report.get("head_start") or {}
        self.state["final_head_start"] = {
            "chosen": head_start.get("chosen"), "gain_points": head_start.get("gain_points"),
            "without": head_start.get("best_without")
            if head_start.get("chosen") == "head-start" else None}
        self.save()
        if best is None:
            winner = report.get("winner") or {}
            _say("rank names no submittable row, so there is no run/final; the winner's "
                 f"blockers: {winner.get('blockers')} (RUNBOOK step 5)")
        return None if best is None else str(best)

    def rank_gate(self, report: dict[str, Any], record: dict[str, Any]) -> None:
        """rank's warnings that point at a wrong input (a term flat over the archive, a
        coefficient of zero, another score block, another script, no GMAT) stop the day
        and are accepted only once shown; the ones every archive gets are shown only."""
        listed = [str(item) for item in report.get("warnings") or []]
        every = list(dict.fromkeys(line.removeprefix("WARNING: ") for line in
                                   [*listed, *record["data"].get("stderr_warnings", [])]))
        warnings = [line for line in every if not any(p.match(line) for p in _RANK_SHOWN)]
        print("---- rank (step 5) ----", file=sys.stderr)
        for line in every:
            kind = "WARNING" if line in warnings else "shown"
            print(f"{kind}: {line}", file=sys.stderr)
        for key in ("ceilings", "levers", "term_spread_points"):
            print(f"{key}: {json.dumps(report.get(key), ensure_ascii=False, sort_keys=True)}",
                  file=sys.stderr)
        print("---- end of rank ----", file=sys.stderr, flush=True)
        if not warnings:
            return
        shown = self.state.setdefault("shown_rank_warnings", [])
        accepted = self.state.get("accepted_rank_warnings")
        if accepted != warnings and not (self.accept_rank_warnings and shown == warnings):
            self.state["shown_rank_warnings"] = warnings
            self.save()
            raise DayStop(
                f"rank printed {len(warnings)} warning(s) that can mean a wrong input, above "
                "and in "
                f"{self.run_dir / 'rank-scored.json'}. Two people read them (RUNBOOK step 5, "
                f"appendix B); if they change nothing, run {self.resume} "
                "--accept-rank-warnings", "5")
        self.state["accepted_rank_warnings"] = warnings
        self.save()

    def check(self, name: str, config: Path) -> Checked | Problem:
        """Step 7 on one candidate, after its certify; always run again (1 s)."""
        from .cli import PrecheckUnavailable

        try:
            return self._check(name, config)
        except (PrecheckUnavailable, KeyError, TypeError, ValueError, OSError) as exc:
            return Problem(name, f"its certification could not be read: "
                           f"{type(exc).__name__}: {exc}", "6 and 7, by hand")

    def _check(self, name: str, config: Path) -> Checked | Problem:
        from .cli import (
            GUI_STOP,
            PRECHECK_PASS,
            PRECHECK_PASS_END_TIME,
            PRECHECK_PASS_NO_TARGET,
            PRECHECK_PASS_NO_TARGET_END_TIME,
            _official_name,
        )

        record = self.verified(f"certify-{name}")
        if record is None:
            earlier = self.steps.get(f"certify-{name}") or {}
            said = earlier.get("said") or []
            # A plots stop is a refusal; no answer that quotes one is not.
            if earlier.get("exit") == 2 and any(GUI_STOP in line for line in said):
                # Where certify itself said it set the run aside (verify._set_aside).
                kept = next((found[1] for line in said
                             if (found := _KEPT.search(line)) is not None), None)
                staging = (f"{kept}/staging/final.script" if kept is not None
                           else "the staging/final.script certify names in its log "
                           f"({self.logs / ('certify-' + name)}.err)")
                return Problem(name, "certify stopped on the plots and no console copy ran: "
                               f"the report comes from the GMAT GUI, running {staging}", "6b")
            code = earlier.get("exit")
            error = next((line for line in said if _ERROR_LINE.match(line)),
                         earlier.get("last_line", ""))
            if code == 2:
                return Problem(name, f"certify refused it (exit 2): {error}", "6 (a refusal "
                               "is a failed certification: take the next fallback, 4a)",
                               refused=True)
            why = ("no answer: GmatConsole crashed, timed out or wrote no report, or this "
                   "machine failed" if code == EXIT_CERTIFY_NO_ANSWER else "it did not finish")
            return Problem(name, f"certify gave no verdict (exit {code}: {why}): {error}",
                           "6 (certify it again)", unfinished=True)
        submission = self.run_dir / name / "submission"
        certificate = submission / "certification.json"
        body = _json(certificate, "6")
        precheck = str((body.get("binding_pre_check") or {}).get("result", "absent"))
        # Each pass with certify's last line for it; the second kind needs 6b.7 by eye.
        passes = {"pass": PRECHECK_PASS, "pass-target-not-in-report": PRECHECK_PASS_NO_TARGET}
        by_eye = {"pass-end-time-only": PRECHECK_PASS_END_TIME,
                  "pass-target-not-in-report-end-time-only": PRECHECK_PASS_NO_TARGET_END_TIME}
        end_time_only = precheck in by_eye and record["last_line"] == by_eye[precheck]
        if not end_time_only and (precheck not in passes
                                  or record["last_line"] != passes[precheck]):
            what, runbook = _PRECHECK_OUTCOMES.get(
                precheck, (f"PRE-CHECK {precheck}: certify's last line was "
                           f"{record['last_line']!r}", "6"))
            return Problem(name, what, runbook)
        report = submission / _official_name(submission / str(body["submission_script_name"]))
        argv = self._script("check_report_rows.py", str(report), "--config", str(config),
                            "--certificate", str(certificate))
        _say(f"step 7 ({name}): {' '.join(argv)}")
        done = self.executor(argv, self.environment(), self.folder,
                             self.logs / f"07-check-{name}")
        if done.returncode != 0:
            what, runbook = _STEP7_OUTCOMES.get(
                done.returncode, (f"step 7 exit {done.returncode}", "7"))
            return Problem(name, what, runbook)
        score = step7_score(done.stdout)
        if score is None:
            return Problem(name, "step 7 exited 0 but printed no score: check that --config "
                           "is the scored config", "7")
        if precheck.startswith("pass-target-not-in-report"):
            _say(f"step 7 ({name}): {_PRECHECK_OUTCOMES[precheck][0]}")
        if end_time_only:
            if name not in (self.state.get("by_eye_passed") or {}):
                return Problem(name, self.by_eye_note(name, report, body, done.stdout, score),
                               "6b.7", end_time_only=True)
            bound = self.rebind(name, body)
            if bound is not None:
                return bound
        label = precheck
        if end_time_only:
            label = ("pass (end time only, by eye)" if precheck == "pass-end-time-only"
                     else f"{precheck} (by eye)")
        burns = (body.get("candidate") or {}).get("burns") or []
        early = any(isinstance(burn, dict) and isinstance(burn.get("time_s"), int | float)
                    and burn["time_s"] < 0.0 for burn in burns)
        return Checked(name, certificate, 0, score, label, early)

    def by_eye_note(self, name: str, report: Path, body: dict[str, Any], printed: str,
                    score: float) -> str:
        """What RUNBOOK 6b.7 compares, side by side: the report's last row, step 7's reading
        of it, and the certificate."""
        clock = body.get("clock") or {}
        column = f"{clock.get('time_origin_craft')}.ElapsedSecs"
        last = _LAST_ROW.search(printed)
        replayed = (body.get("final_script_parity") or {}).get("replayed_score")
        certified = ("certified without coefficients: step 7's is the only price"
                     if replayed is None else f"certified {replayed}")
        precheck = str((body.get("binding_pre_check") or {}).get("result"))
        outcome = _PRECHECK_OUTCOMES.get(precheck, _PRECHECK_OUTCOMES["pass-end-time-only"])
        return (f"{outcome[0]}. In {report}, the last row's "
                f"{column} must read the certificate's T_team {clock.get('t_team_s')} s; step 7 "
                f"read it as {last[1] if last else 'unknown'} s and scored it {score:.4f} "
                f"({certified}). Once the report and these agree, {self.resume} "
                f"--by-eye-passed {name}")

    def rebind(self, name: str, body: dict[str, Any]) -> Problem | None:
        """certify's binding pre-check for a by-eye pass, again: it must still exit 3, the
        end and a row at each burn epoch bound, under --accept-end-time-binding-only; 7 for a
        report without the target, which binds the same and then says it cannot judge.
        The pre-check's step 7 run (--certificate) is step 7's, run on its own."""
        record = body.get("binding_pre_check") or {}
        expected = 7 if str(record.get("result")).startswith("pass-target-not-in-report") else 3
        runs = [run for run in record.get("runs") or []
                if not (isinstance(run, dict) and isinstance(run.get("command"), list)
                        and "--certificate" in run["command"])]
        commands: list[list[str]] = []
        for run in runs:
            command = run.get("command") if isinstance(run, dict) else None
            if isinstance(command, list) and _END_ONLY in command:
                commands.append([str(word) for word in command])
        if not commands or len(commands) != len(runs):
            return Problem(name, "its pre-check records no end-time-only binding to run "
                           "again", "6")
        exits = []
        for index, command in enumerate(commands):
            _say(f"step 6's binding again ({name}, by eye): {' '.join(command)}")
            done = self.executor(command, self.environment(), self.folder,
                                 self.logs / f"07-bind-{name}-{index}")
            exits.append(done.returncode)
        self.state.setdefault("by_eye_passed", {})[name]["binding_exits"] = exits
        self.save()
        if any(code != expected for code in exits):
            return Problem(name, f"step 6's end-time-only binding exited {exits}, not "
                           f"{expected}: the by-eye pass does not stand", "6 and 6b.7")
        return None

    def pick(self, config: Path) -> Checked:
        """The highest-scoring candidate; a stop if any left in the running did not pass."""
        excluded = self.state["excluded"]
        checked: list[Checked] = []
        problems: list[Problem] = []
        for name in CANDIDATES:
            if name in excluded or f"certify-{name}" not in self.steps:
                continue
            outcome = self.check(name, config)
            if isinstance(outcome, Checked):
                checked.append(outcome)
            else:
                problems.append(outcome)
        self.state["checked"] = {
            **{item.candidate: {"score": item.score, "step7_exit": item.exit_code,
                                "precheck": item.precheck} for item in checked},
            **{item.candidate: {"problem": item.what, "runbook": item.runbook,
                                "end_time_only": item.end_time_only} for item in problems}}
        self.save()
        if problems:
            raise DayStop(self.problems_note(problems),
                          ", ".join(sorted({item.runbook for item in problems})))
        if not checked:
            raise DayStop("no candidate was certified" + (
                f" (excluded: {', '.join(excluded)})" if excluded else ""),
                "4a, fallback 4 (the zero burn, by hand)")
        def order(item: Checked) -> tuple[Any, ...]:
            return (item.score, "by eye" not in item.precheck, -CANDIDATES.index(item.candidate))

        best = max(checked, key=order)
        without = [item for item in checked if not item.head_start]
        gain = load_config(config).policy.head_start_min_gain_points
        if best.head_start and without:
            # As rank: a head start only by policy.head_start_min_gain_points.
            other = max(without, key=order)
            margin = float(best.score or 0.0) - float(other.score or 0.0)
            _say(f"head start: {best.candidate} burns in it, {margin:+.4f} points over "
                 f"{other.candidate}, needs {gain:g} (policy.head_start_min_gain_points)")
            if margin < gain:
                best = other
        _say("step 7 scores: " + ", ".join(
            f"{item.candidate} {item.score:.4f}" + (" (head start)" if item.head_start else "")
            for item in checked) + f"; the upload is {best.candidate}")
        return best

    def problems_note(self, problems: Sequence[Problem]) -> str:
        """Each candidate that did not pass, and the command that applies to it."""
        lines = ["not every candidate passed:"]
        lines += [f"  {item.candidate}: {item.what} (docs/RUNBOOK.md step {item.runbook})"
                  for item in problems]
        unfinished = [item.candidate for item in problems if item.unfinished]
        refused = [item.candidate for item in problems if item.refused]
        if unfinished:
            names = ", ".join(unfinished)
            lines += [f"That is no verdict on {names}. To certify it again, run:",
                      f"\n{self.resume}\n",
                      "If it ends the same way again, a person decides; "
                      + " ".join(f"--exclude {name}" for name in unfinished)
                      + (" then leaves it out." if len(unfinished) == 1
                         else " then leaves them out, or name only some.")]
        for name in refused:
            lines += [f"{name}: certify's refusal is its verdict on the candidate, so "
                      "--exclude applies. To leave it out and choose among the rest, run:",
                      f"\n{self.resume} --exclude {name}\n",
                      f"--include {name} runs its certify again instead. That applies only "
                      "when its log shows GMAT did not finish a run (a crash, a timeout, no "
                      f"report): {self.logs / ('certify-' + name)}.err"]
        if len(unfinished) + len(refused) < len(problems):
            lines.append("A person decides each other one: --exclude <name> leaves one out "
                         f"({self.resume} --exclude <name>); or follow the RUNBOOK by hand "
                         "from the step named.")
        return "\n".join(lines)

    def audit(self, chosen: Checked, config: Path) -> None:
        """RUNBOOK step 7's audit_burn_table.py on the burns certify certified."""
        body = _json(chosen.certificate, "7")
        burns = self.run_dir / chosen.candidate / "burns.json"
        name = f"07-audit-{chosen.candidate}"
        listed = (body.get("candidate") or {}).get("burns")
        if not isinstance(listed, list):
            raise DayStop(f"{chosen.certificate} lists no candidate burns to audit", "7")
        if self.verified(name) is None:
            burns.write_text(json.dumps({"burns": listed}), encoding="utf-8")
        record, _ = self.step(
            name, "7", self._script("audit_burn_table.py", str(self.official), str(burns),
                                    "--config", str(config), *self._gmat_flags()),
            outputs=lambda _: [burns])
        if record["exit"] != 0:
            audited = next((line for line in self._log_text(name, ".out").splitlines()
                            if line.startswith("AUDIT:")), record["last_line"])
            raise DayStop(f"audit_burn_table.py exited {record['exit']} on {chosen.candidate}: "
                          f"{audited}; its log is {self.logs / name}.out. A person decides; "
                          f"{self.resume} --exclude {chosen.candidate} "
                          "leaves it out", "7")

    def _newer_latest(self, final_root: Path) -> str | None:
        """latest's folder when another day run delivered it after this one began."""
        latest = final_root / "latest"
        if not latest.is_symlink():
            return None
        current = os.readlink(latest)
        ours = {Path(item["folder"]).name for item in self.state.get("uploads", [])}
        stamp = current.split(".", 1)[0]
        return current if current not in ours and stamp > self.state["created_utc"] else None

    def deliver(self, chosen: Checked, config: Path, final_root: Path) -> Path:
        """The two files into <final_root>/<UTC>.partial/, proven there, renamed, then latest."""
        from .cli import PrecheckUnavailable

        try:
            return self._deliver(chosen, config, final_root)
        except (PrecheckUnavailable, KeyError, TypeError, ValueError, OSError) as exc:
            left = f"; it left {self.left}, which is not an upload" if self.left else ""
            raise DayStop(f"the copy into {final_root} failed: {type(exc).__name__}: "
                          f"{exc}{left}", "6 and 7 (copy by hand)") from None

    def _deliver(self, chosen: Checked, config: Path, final_root: Path) -> Path:
        from .cli import _official_name

        body = _json(chosen.certificate, "7")
        submission = chosen.certificate.parent
        script_name = str(body["submission_script_name"])
        report_name = _official_name(submission / script_name)
        listed = body["submission_files"]
        for name in (script_name, report_name):
            if listed.get(name) != file_sha256(submission / name):
                raise DayStop(f"{submission / name} is not the file certify listed", "6")
        check_disk([final_root], self.state["min_free_gb"], self.resume)
        _check_final_root(final_root)
        newer = self._newer_latest(final_root)
        if newer is not None and not self.force_latest:
            raise DayStop(f"{final_root / 'latest'} points at {newer}, delivered by another day "
                          f"run after {self.folder} began; this one would replace it. Run "
                          f"{self.resume} --force-latest only if this "
                          "day folder's is the upload", "-")
        final_root.mkdir(parents=True, exist_ok=True)
        partial = _fresh_folder(final_root, ".partial")
        self.left = partial
        stamp = partial.name.removesuffix(".partial")
        target = final_root / stamp
        for name in (script_name, report_name):
            shutil.copyfile(submission / name, partial / name)
            if file_sha256(partial / name) != listed[name]:
                raise DayStop(f"{partial / name} does not match what certify listed; "
                              f"{partial} is not an upload", "6")
        (partial / "SHA256SUMS").write_text(
            "".join(f"{listed[name]}  {name}\n" for name in (script_name, report_name)),
            encoding="utf-8")
        argv = self._script("check_report_rows.py", str(partial / report_name), "--config",
                            str(config), "--certificate", str(chosen.certificate),
                            "--script", str(partial / script_name))
        _say(f"step 7 on the copy: {' '.join(argv)}")
        done = self.executor(argv, self.environment(), self.folder, self.logs / "07-check-copy")
        again = step7_score(done.stdout)
        if done.returncode != 0 or again != chosen.score:
            refused = final_root / f"{stamp}.not-proven"
            partial.rename(refused)
            self.left = refused
            raise DayStop(f"step 7 on the copied files exited {done.returncode} with score "
                          f"{again} (the day folder's: exit 0, {chosen.score}); the copy is "
                          f"kept as {refused}, never as latest", "7")
        self.state.setdefault("uploads", []).append(
            {"folder": str(target), "candidate": chosen.candidate, "score": chosen.score,
             "run": len(self.state["runs"]) - 1})
        (partial / NOTE_NAME).write_text(
            self.summary(chosen, body, script_name, report_name, target), encoding="utf-8")
        for path in partial.iterdir():
            path.chmod(0o444)
        # Proven: only now does it take the name an upload has.
        _stop_if_ctrl_c()
        partial.rename(target)
        self.left = None
        self.made = target
        self.save()
        newer = self._newer_latest(final_root)
        if newer is not None and not self.force_latest:
            raise DayStop(f"{target} holds this day folder's proven pair, but "
                          f"{final_root / 'latest'} now points at {newer}, delivered by another "
                          "day run meanwhile; latest was left there. --force-latest moves it", "-")
        staged = final_root / f".latest.{os.getpid()}"
        if staged.is_symlink() or staged.exists():
            staged.unlink()
        staged.symlink_to(stamp)
        _stop_if_ctrl_c()
        os.replace(staged, final_root / "latest")
        return target

    def _chain(self, chosen: Checked) -> list[str]:
        """The steps the delivered pair came from, in order."""
        names = ["02-day-config", "03-inspect", "03-baseline", "04-search"]
        late = chosen.candidate in ("final", "final-03")
        if not late:
            names.append(f"certify-{chosen.candidate}")
        names += ["05-scored-config", "05-search", "05-rank"]
        if late:
            names.append(f"certify-{chosen.candidate}")
        return [*names, f"07-audit-{chosen.candidate}"]

    def summary(self, chosen: Checked, body: dict[str, Any], script_name: str,
                report_name: str, target: Path) -> str:
        candidate = body.get("candidate") or {}
        parity = body.get("final_script_parity") or {}
        clock = body.get("clock") or {}
        total = candidate.get("total_dv_km_s")
        runs = self.state["runs"]
        score = self.inputs["score"]
        gmat = self.state["gmat"]
        others = [f"{name} {item.get('score', item.get('problem'))}"
                  for name, item in self.state.get("checked", {}).items()
                  if name != chosen.candidate]
        produced = {name: self.steps[name].get("run", 0) for name in self._chain(chosen)
                    if name in self.steps}
        used = sorted({*produced.values(), len(runs) - 1})
        lines = [
            "# Not for upload. orbitwin day's record of the two files beside it",
            "",
            "Upload exactly these two files, as they are:",
            f"script       : {script_name}",
            f"report       : {report_name}",
            "",
            f"chosen       : {chosen.candidate} ({candidate.get('candidate_id')}) from "
            f"{chosen.certificate.parent.parent}",
            f"score        : {chosen.score:.4f} (step 7, under {self.run_dir / 'day-scored.yaml'})",
            f"outcome      : {'hit' if candidate.get('successful') else 'scored miss'}",
            f"T_team       : {clock.get('t_team_s')} s",
            f"delta-v      : {total} km/s"
            + (f" = {total * 1000.0:.3f} m/s" if isinstance(total, int | float) else ""),
            (f"d_min        : {_km(parity.get('minimum_distance_km'))} (closest report row)"
             if not chosen.precheck.startswith("pass-target-not-in-report") else
             f"d_min        : {_km(parity.get('minimum_distance_km'))}, and the interception, "
             "from certify's diagnostic flight of the same script: the official report "
             "carries no target position"),
            f"PRE-CHECK    : {chosen.precheck.upper()}",
            "step 7       : exit 0 in the day folder, exit 0 on these copies",
            f"audit        : audit_burn_table.py exit 0 ({self.logs / ('07-audit-' + chosen.candidate)}.out)",
            f"GMAT         : {gmat['console']} (SHA-256 {gmat['console_sha256']})",
            f"startup file : {gmat['startup']} (SHA-256 {gmat['startup_sha256']})",
            f"official     : {self.state['official']['source']} (SHA-256 "
            f"{self.state['official']['sha256']})",
            f"T_max        : {self.inputs['horizon_s']!r} s"
            + (" (as announced)" if self.inputs["tmax_as_announced"] else ""),
            f"time origin  : {clock.get('time_origin_craft')}'s epoch (mission_time_origin "
            f"{clock.get('mission_time_origin')}): T_max, T_team and Ct count from it",
            "start        : " + (
                f"{clock.get('time_origin_craft')} {clock.get('alignment_offset_s')} s first; "
                f"d_min window {self.inputs['closest_approach_window']}"
                if clock.get("alignment_offset_s") else "the craft start together"),
            "head start   : " + self.head_start_note(chosen, candidate, clock),
            f"coefficients : kt {score['kt_per_s']!r} /s, Ct {score['ct_s']!r} s, kv "
            f"{score['kv_per_m_s']!r} per m/s, Cv {score['cv_m_s']!r} m/s, C_r "
            f"{score['cr_km']!r} km",
            f"others       : {', '.join(others) or 'none'}",
            f"excluded     : {', '.join(self.state['excluded']) or 'none'}",
            f"day folder   : {self.folder}",
            f"written      : {target.name}",
        ]
        for step, item in sorted((self.state.get("confirmed_readbacks") or {}).items()):
            lines.append(f"readback     : step {step} confirmed by run {item['run']} "
                         f"(--confirm-readback {step}) at {item['utc']}, SHA-256 of "
                         f"{item.get('of', 'the config')} {item['sha256']}")
        for index in used:
            record = runs[index]["provenance"]
            how = (f"the expected commit {record.get('expected')}" if record.get("on_main")
                   else "NOT as expected: " + "; ".join(record.get("notes") or []))
            steps = [name for name, run in produced.items() if run == index]
            if index == len(runs) - 1:
                steps.append("the copy")
            lines.append(f"commit       : run {index} ({runs[index]['started_utc']}): "
                         f"{record.get('commit')} ({how}); {', '.join(steps)}")
        for name, item in sorted((self.state.get("by_eye_passed") or {}).items()):
            lines.append(f"by eye       : {name} matched RUNBOOK 6b.7, said by run {item['run']} "
                         f"(--by-eye-passed {name}) at {item['utc']}; step 6's end-time-only "
                         f"binding exited {item.get('binding_exits')}")
        accepted = self.state.get("accepted_config_warnings") or {}
        lines += [f"accepted     : {name}: {line}" for name, items in accepted.items()
                  for line in items]
        lines += [f"accepted     : 05-rank: {line}"
                  for line in self.state.get("accepted_rank_warnings") or []]
        return "\n".join(lines) + "\n"

    def head_start_note(self, chosen: Checked, candidate: dict[str, Any],
                        clock: dict[str, Any]) -> str:
        """README's line: whether head-start burns were open and whether the upload uses one."""
        lead = clock.get("head_start_s")
        if self.inputs.get("no_head_start_burns"):
            return "closed: turned off on the day (--no-head-start-burns)"
        if not lead:
            return f"closed ({clock.get('head_start_note') or 'policy.head_start_burns is off'})"
        early = [float(burn["time_s"]) for burn in candidate.get("burns") or []
                 if isinstance(burn, dict) and isinstance(burn.get("time_s"), int | float)
                 and burn["time_s"] < 0.0]
        origin = clock.get("time_origin_craft")
        late = "SC_Alien" if origin == "SC_Human" else "the later craft"
        burned = ", ".join(f"{-t:.2f} s before {late}'s epoch (mission clock "
                           f"{float(lead) + t:.2f} s)" for t in early)
        rank = self.state.get("final_head_start") or {}
        gain = rank.get("gain_points")
        return (f"open, {float(lead):.3f} s (policy.head_start_burns); the upload "
                + (f"burns in it: {burned}" if chosen.head_start
                   else "burns only after the alignment")
                + (f"; rank's head-start margin {gain:+.4f} points" if gain is not None else ""))

    def fallback_note(self) -> str:
        """What 4a left certified, for a day whose coefficients never come."""
        nets = [name for name in ("final-01", "final-02")
                if (self.steps.get(f"certify-{name}") or {}).get("exit") == 0]
        if not nets:
            return ("4a certified nothing; without coefficients there is no upload (RUNBOOK "
                    "step 4a, fallback 4)")
        return (f"if no coefficients come, the upload is {nets[0]} "
                f"({self.run_dir / nets[0] / 'submission'}), once step 7 passes on it by hand "
                "(RUNBOOK step 4a: before the coefficients, the first that passed)")

    # -- the flow -------------------------------------------------------------------

    def record_by_eye(self) -> None:
        """--by-eye-passed: only for a candidate a stop showed ending on the end-time pass."""
        for name in self.by_eye:
            shown = (self.state.get("checked") or {}).get(name) or {}
            if not shown.get("end_time_only"):
                raise DayStop(f"--by-eye-passed {name}: no stop of this day folder showed "
                              f"{name} ending on PRE-CHECK PASS (end time only)", "6b.7")
            self.state.setdefault("by_eye_passed", {})[name] = {
                "run": len(self.state["runs"]) - 1, "utc": _utc_stamp()}
        self.save()

    def recover(self) -> Path | None:
        """A proven upload of this day folder (renamed to <UTC>/, so step 7 passed on it)
        that latest does not point at: a stop or a crash came between the rename and the
        move of latest, before or after state.json recorded it."""
        final_root = Path(self.state["final_root"])
        if not final_root.is_dir():
            return None
        uploads = self.state.get("uploads") or []
        found = [Path(uploads[-1]["folder"])] if uploads else [
            target for target in sorted(final_root.iterdir(), reverse=True)
            if not target.is_symlink() and (target / NOTE_NAME).is_file()
            and f"day folder   : {self.folder}\n" in (target / NOTE_NAME).read_text(
                encoding="utf-8", errors="replace")]
        for target in found[:1]:
            if not uploads:
                self.state["uploads"] = [{"folder": str(target), "recovered": True,
                                          "run": len(self.state["runs"]) - 1}]
                self.save()
            self.made = target
            _say(f"{target} is this day folder's proven upload, which latest did not point "
                 "at: a stop or crash came after it was proven; latest moves to it now")
            newer = self._newer_latest(final_root)
            if newer is not None and not self.force_latest:
                raise DayStop(f"{final_root / 'latest'} points at {newer}, delivered by "
                              "another day run; --force-latest moves it to " + str(target), "-")
            staged = final_root / f".latest.{os.getpid()}"
            if staged.is_symlink() or staged.exists():
                staged.unlink()
            staged.symlink_to(target.name)
            _stop_if_ctrl_c()
            os.replace(staged, final_root / "latest")
            return target
        return None

    def run(self) -> Path:
        recovered = self.recover()
        if recovered is not None:
            return recovered
        # A run's steps compile from source into a cache no earlier run left anything in.
        if self.pycache.is_symlink() or (self.pycache.exists() and not self.pycache.is_dir()):
            raise DayStop(f"{self.pycache} is not the folder the day keeps its bytecode in "
                          "(a link or a file); move it away, then resume", "-")
        shutil.rmtree(self.pycache, ignore_errors=True)
        self.pycache.mkdir()
        self.record_by_eye()
        self.preserve()
        day = self.day_config()
        self.inspect_and_baseline(day)
        first = self.search("04-search", "4", day)
        self.safety_nets(first, day)
        if self.inputs["score"] is None:
            raise DayPaused(
                "steps 1 to 4a are done; the coefficients are not given yet.\n"
                "Once announced, run:\n"
                + _alone(f"{self.resume} --announcement <entry> --announcement <entry>"
                         if self.inputs.get("announcement") else
                         f"{self.resume} --kt-per-s <kt> --ct-s <Ct> "
                         "--kv-per-m-s <kv> --cv-m-s <Cv> --cr-km <C_r>")
                + "Meanwhile, " + self.fallback_note() + ".")
        scored = self.scored_config(first, day)
        second = self.search("05-search", "5", scored)
        if second == first:
            raise DayStop(f"step 5's search is step 4's, {first}", "5")
        best = self.rank(scored, second)
        if best is not None and "final" not in self.state["excluded"]:
            self.certify("final", "6", best, scored, second)
        without = (self.state.get("final_head_start") or {}).get("without")
        if best is not None and without is not None and "final-03" not in self.state["excluded"]:
            # final burns in the head start: the best without it goes beside it (step 4a's rule).
            self.certify("final-03", "6", str(without), scored, second)
        chosen = self.pick(scored)
        self.audit(chosen, scored)
        return self.deliver(chosen, scored, Path(self.state["final_root"]))


def _parser_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("script", nargs="?", help="the official script, as the organizer gave it")
    parser.add_argument("--companion", action="append",
                        help="a file that came with the script, relative to its folder "
                        "(a tide, gravity or SPAD file; RUNBOOK step 1); repeat for each")
    parser.add_argument("--horizon-s", type=positive_finite, help="T_max, as announced")
    parser.add_argument("--announcement", action="append",
                        help="an entry of the announcement typed by one person, given twice "
                        "(docs/ANNOUNCEMENT.md): T_max, the boundary and, once announced, the "
                        "score, in place of their flags; on a resume, the score's entries")
    parser.add_argument("--tmax-as-announced", action="store_true",
                        help="T_max is the announced value although SC_Alien is off the "
                        "boundary there (RUNBOOK step 2)")
    parser.add_argument("--mission-boundary-km", type=positive_finite,
                        help="the mission boundary radius, if announced")
    parser.add_argument("--closest-approach-window", choices=CLOSEST_APPROACH_WINDOWS,
                        help="which rows d_min counts when the craft start apart; a person "
                        "decides (docs/ASYNC_START_AND_INSERTION.md)")
    parser.add_argument("--kt-per-s", type=float, help="kt, per s")
    parser.add_argument("--ct-s", type=float, help="Ct, in s")
    parser.add_argument("--kv-per-m-s", type=float, help="kv, per m/s")
    parser.add_argument("--cv-m-s", type=float, help="Cv, in m/s")
    parser.add_argument("--cr-km", type=positive_finite, help="C_r, in km")
    parser.add_argument("--pursuer-name", help="if the template does not use SC_Human")
    parser.add_argument("--target-name", help="if the template does not use SC_Alien")
    parser.add_argument("--base", help="the profile step 2 copies (RUNBOOK appendix E); "
                        "config/ranking.yaml by default (the packaged copy when installed)")
    parser.add_argument("--gmat-workers", type=int, help="GmatConsole runs at once")
    parser.add_argument("--team-number", type=positive_int,
                        help="the team's number for the official report's file name (step 2); "
                        "needed unless --base sets policy.team_number")
    parser.add_argument("--report-filename-as-written",
                        help="only after the organizer's written answer (RUNBOOK step 6)")
    parser.add_argument("--accept-announced-horizon-before-perigee-s", type=positive_finite,
                        help="only after the organizer's written answer (RUNBOOK step 3)")
    parser.add_argument("--no-head-start-burns", action="store_true",
                        help="no burn before the later craft's epoch, if the organizer says so "
                        "on the day (policy.head_start_burns off; RUNBOOK step 2)")
    parser.add_argument("--gmat-console", help="GmatConsole, with --startup-file; else "
                        "GMAT_CONSOLE and GMAT_STARTUP")
    parser.add_argument("--startup-file", help="its stock startup file, with --gmat-console")
    parser.add_argument("--work-root",
                        help=f"where day folders go (default {DEFAULT_WORK_ROOT}{_UNDER})")
    parser.add_argument("--final-root",
                        help=f"where the upload goes (default {DEFAULT_FINAL_ROOT}{_UNDER}); tests "
                        "name another")
    parser.add_argument("--resume", help="a day folder to go on with; its inputs are recorded")
    parser.add_argument("--confirm-readback", action="append", choices=READBACKS,
                        help="two people read this step's readback, printed at the pause "
                        "before; recorded")
    parser.add_argument("--exclude", action="append", choices=CANDIDATES,
                        help="leave this candidate out, after a person decided so")
    parser.add_argument("--include", action="append", choices=CANDIDATES,
                        help="take back an --exclude")
    parser.add_argument("--accept-config-warnings", action="store_true",
                        help="after two people read the readback's WARNING lines")
    parser.add_argument("--accept-rank-warnings", action="store_true",
                        help="after two people read rank's warnings")
    parser.add_argument("--force-latest", action="store_true",
                        help="move latest although another day run delivered after this one "
                        "began")
    parser.add_argument("--expect-commit",
                        help="the commit chosen the day before (RUNBOOK step 0); HEAD must be "
                        "it, nothing is fetched")
    parser.add_argument("--by-eye-passed", action="append", choices=CANDIDATES,
                        help="RUNBOOK 6b.7's by-eye comparison matched for this candidate, "
                        "which ended on PRE-CHECK PASS (end time only); recorded")
    parser.add_argument("--allow-non-main", action="store_true",
                        help="run another commit than --expect-commit, or local changes, "
                        "recorded beside the upload")
    parser.add_argument("--min-free-gb", type=positive_finite, default=20.0,
                        help="refuse with less free space than this (default 20)")


def add_day_parser(sub: Any) -> None:
    day = sub.add_parser(
        "day", allow_abbrev=False,
        help=f"RUNBOOK steps 1 to 7 in one command: the upload into {DEFAULT_FINAL_ROOT}/<UTC>/",
        description=(__doc__ or "").splitlines()[0])
    _parser_arguments(day)


def _score(args: argparse.Namespace) -> dict[str, float] | None:
    given = {name: getattr(args, name) for name in _SCORE_FLAGS}
    if all(value is None for value in given.values()):
        return None
    missing = [f"--{name.replace('_', '-')}" for name, value in given.items() if value is None]
    if missing:
        raise DayStop(f"the score needs all five coefficients; missing {' '.join(missing)}",
                      "5")
    return _checked_score(given)


#: Each score key as an entry names it.
_ENTRY_SCORE = {"kt_per_s": "kt", "ct_s": "Ct", "kv_per_m_s": "kv", "cv_m_s": "Cv",
                "cr_km": "C_r"}


def _checked_score(given: dict[str, float], entered: bool = False,
                   runbook: str = "5") -> dict[str, float]:
    def named(key: str) -> str:
        return f"score.{_ENTRY_SCORE[key]}" if entered else f"--{key.replace('_', '-')}"

    for name, value in given.items():
        if not math.isfinite(value):
            raise DayStop(f"{named(name)} {value!r} is not finite", runbook)
    for name in ("kt_per_s", "kv_per_m_s"):
        if given[name] < 0.0:
            raise DayStop(f"{named(name)} {given[name]!r} is negative: a score that rises "
                          "with time or delta-v; read the announcement again", runbook)
    return given


def _score_flags(args: argparse.Namespace) -> bool:
    return any(getattr(args, name) is not None for name in _SCORE_FLAGS)


def _entries(args: argparse.Namespace, base: str | None, pursuer: str | None,
             target: str | None, runbook: str, script: Path
             ) -> tuple[tuple[Entry, Entry, Entry], float | None]:
    """The two entries --announcement names, agreed and held to ``script``, and T_max in
    seconds from them (None when they give the score alone). Checked before anything is
    recorded, so a day folder never keeps entries its own steps will refuse."""
    if len(args.announcement) != 2:
        raise DayStop("--announcement is given twice, once for each person's entry", runbook)
    try:
        config = load_config(Path(base) if base else ranking_profile())
        pursuer, target = pursuer or config.pursuer_name, target or config.target_name
        if not pursuer or not target:
            raise DayStop("the base config does not name both craft; pass --pursuer-name and "
                          "--target-name", runbook)
        entries = read_pair(*(Path(path).expanduser().resolve() for path in args.announcement))
        rules = interception_rules(config.policy.rules)
        view = read_script(script)
        wrong, _, _ = against_script(entries[2], view, pursuer, rules)
        if wrong:
            raise DayStop("the script is not what was announced: " + "; ".join(wrong)
                          + ". Stop and ask the organizer which is right", runbook)
        if entries[2].t_max is None and entries[2].t_max_s is None:
            return entries, None
        return entries, horizon_s(entries[2], rules, pursuer, target, written_epochs(view))
    except DayStop:
        raise
    except (OSError, OrbitWinError) as exc:
        raise DayStop(f"--announcement: {exc}", runbook) from None


def _entry_score(entry: Entry, runbook: str) -> dict[str, float] | None:
    if entry.score is None:
        return None
    score = entry.score
    return _checked_score(entered=True, runbook=runbook, given={
        "kt_per_s": float(score["kt"].value), "ct_s": float(score["Ct"].value),
        "kv_per_m_s": float(score["kv"].value), "cv_m_s": float(score["Cv"].value),
        "cr_km": float(score["C_r"].value)})


def _copy_entries(folder: Path, entries: tuple[Entry, Entry, Entry], name: str,
                  recorded: dict[str, str], runbook: str) -> dict[str, str]:
    """Both entries into the day folder's input/, read-only, by SHA-256. A file there that
    state.json does not record is a copy an earlier run did not finish, and goes."""
    files = {}
    for index, entry in enumerate(entries[:2], 1):
        rel = f"input/{name}-{index}.toml"
        target = folder / rel
        if rel in recorded:
            raise DayStop(f"{target} is a file the day folder records already (a companion, or "
                          "entries taken before); start a new day folder", runbook)
        if entry.path.resolve().is_relative_to((folder / "input").resolve()):
            raise DayStop(f"{entry.path} is in the day folder; name the entry where it was "
                          "typed", runbook)
    # Every check above passes before anything is copied.
    for index, entry in enumerate(entries[:2], 1):
        rel = f"input/{name}-{index}.toml"
        target = folder / rel
        target.unlink(missing_ok=True)
        shutil.copyfile(entry.path, target)
        target.chmod(0o444)
        if file_sha256(target) != entry.sha256:
            raise DayStop(f"{entry.path} changed while it was copied; copy it again", runbook)
        files[rel] = entry.sha256
    return files


@contextmanager
def _locked(folder: Path) -> Iterator[None]:
    """The day folder to this run alone, and to the steps it starts, which inherit the lock:
    a step a killed day left running still holds it."""
    with (folder / ".day.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise DayStop(_running_note(folder, _still_running(folder, tidy=False)),
                          "-") from None
        left = _still_running(folder, tidy=True)
        if left:
            raise DayStop(_running_note(folder, left), "-")
        _HELD_LOCKS.append(handle.fileno())
        try:
            yield
        finally:
            _HELD_LOCKS.remove(handle.fileno())


def _running_note(folder: Path, left: list[str]) -> str:
    resume = _resume_command(folder, _recorded_non_main(folder))
    if not left:
        return (f"another orbitwin day is running on {folder}, or something an earlier run "
                f"started still holds its lock. `ps -eo pid,args | rg {folder.name}` lists "
                "them; if no orbitwin day is among them, stop every one it lists (kill <their "
                f"pids>). Then run:\n\n{resume}\n")
    return ("a step an earlier run of this day folder started is still running, and would "
            "run beside the resume in the same run/:\n  " + "\n  ".join(left)
            + "\nWait for it to end (it ends as its step does), or stop it with the kill "
            f"named. Then run:\n\n{resume}\n")


def _resume_command(folder: Path, allow_non_main: bool) -> str:
    """The resume a stop prints, quoted to be pasted as it is."""
    return f"orbitwin day --resume {shlex.quote(str(folder))}" + (
        " --allow-non-main" if allow_non_main else "")


def _recorded_non_main(folder: Path) -> bool:
    """Whether state.json, read as text, records --allow-non-main; False if unreadable."""
    try:
        text = (folder / "state.json").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return '"allow_non_main": true' in text


def _run_record(args: argparse.Namespace, commit: dict[str, Any]) -> dict[str, Any]:
    """One invocation: when, which commit, and the decisions it carried."""
    return {"started_utc": _utc_stamp(), "provenance": commit,
            "decisions": {name: getattr(args, name) for name in (
                "confirm_readback", "by_eye_passed", "exclude", "include",
                "accept_config_warnings",
                "accept_rank_warnings", "force_latest", "allow_non_main")
                if getattr(args, name) not in (None, False)}}


def _new_day(args: argparse.Namespace, commit: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if args.script is None or (args.horizon_s is None and not args.announcement):
        raise DayStop("give the official script and --horizon-s T_max, or two --announcement "
                      "entries (or --resume a day folder)", "1, 2")
    if args.announcement and (args.horizon_s is not None or args.mission_boundary_km
                              is not None or _score_flags(args)):
        raise DayStop("--announcement gives T_max, the boundary and the score; drop "
                      "--horizon-s, --mission-boundary-km and the score flags", "2")
    if args.confirm_readback:
        raise DayStop("--confirm-readback confirms a readback already printed; a new day "
                      "folder has none yet", "2")
    official = Path(args.script).expanduser().resolve()
    if not official.is_file():
        raise DayStop(f"{official} is not a file", "1")
    if record_path(official).is_file():
        raise DayStop(f"{official} is a set-epoch copy ({record_path(official).name}); the day "
                      "runs under rules that fix the epoch and cannot check its record", "1")
    companions = [_relative_companion(name) for name in args.companion or ()]
    for rel in companions:
        if not (official.parent / rel).is_file():
            raise DayStop(f"--companion {rel}: {official.parent / rel} is not a file", "1")
    console, startup = _gmat_pair(args.gmat_console, args.startup_file)
    final_root = Path(args.final_root or DEFAULT_FINAL_ROOT).expanduser().resolve()
    work_root = Path(args.work_root or DEFAULT_WORK_ROOT).expanduser().resolve()
    _outside_the_guard(work_root, "--work-root")
    _outside_the_guard(final_root, "--final-root")
    _check_final_root(final_root)
    base = None if args.base is None else str(Path(args.base).expanduser().resolve())
    _team_given(args.team_number, Path(base) if base else ranking_profile(),
                named=base is not None)
    entries, horizon, boundary = None, args.horizon_s, args.mission_boundary_km
    if args.announcement:
        entries, horizon = _entries(args, base, args.pursuer_name, args.target_name, "2",
                                    official)
        if horizon is None:
            raise DayStop("the entries give no T_max, which a new day needs", "2")
        boundary = (None if entries[2].boundary is None
                    else float(entries[2].boundary.value))
        score = _entry_score(entries[2], "2")
    else:
        score = _score(args)
    check_disk([work_root, final_root], args.min_free_gb)
    work_root.mkdir(parents=True, exist_ok=True)
    folder = _fresh_folder(work_root)
    (folder / "input").mkdir()
    # RUNBOOK step 1: copied, made read-only, hashed.
    copy = folder / "input" / "official.script"
    shutil.copyfile(official, copy)
    copy.chmod(0o444)
    hashed = {}
    for rel in companions:
        target = folder / "input" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(official.parent / rel, target)
        target.chmod(0o444)
        hashed[rel.as_posix()] = file_sha256(target)
    if file_sha256(copy) != file_sha256(official):
        raise DayStop(f"{copy} does not match {official}", "1")
    entered = {} if entries is None else _copy_entries(
        folder, entries, "announcement", {f"input/{rel}": sha for rel, sha in hashed.items()},
        "2")
    state = {
        "format": STATE_FORMAT,
        "created_utc": folder.name,
        "commit": commit,
        "runs": [],
        "inputs": {
            "horizon_s": horizon, "tmax_as_announced": args.tmax_as_announced,
            "mission_boundary_km": boundary,
            # Step 2 hands ranking_day_config these in place of the numbers.
            "announcement": sorted(entered) or None,
            "score_announcement": sorted(entered) if entered and score else None,
            "closest_approach_window": args.closest_approach_window,
            "team_number": args.team_number,
            "pursuer_name": args.pursuer_name, "target_name": args.target_name, "base": base,
            "gmat_workers": args.gmat_workers,
            "report_filename_as_written": args.report_filename_as_written,
            "accept_announced_horizon_before_perigee_s":
                args.accept_announced_horizon_before_perigee_s,
            "no_head_start_burns": args.no_head_start_burns,
            "score": score,
            "expect_commit": None if args.expect_commit is None
            else args.expect_commit.strip().lower(),
        },
        "gmat": {"console": str(console), "console_sha256": file_sha256(console),
                 "startup": str(startup), "startup_sha256": file_sha256(startup)},
        "official": {"source": str(official), "sha256": file_sha256(copy),
                     "companions": hashed},
        "announcement_files": entered,
        "final_root": str(final_root),
        "min_free_gb": args.min_free_gb,
        # Every resume printed then carries it (--allow-non-main).
        "allow_non_main": bool(args.allow_non_main),
        "excluded": sorted(set(args.exclude or ()) - set(args.include or ())),
        "steps": {},
    }
    print(f"orbitwin day: day folder {folder}; official script SHA-256 "
          f"{state['official']['sha256']}", file=sys.stderr)
    return folder, state


def _recorded(args: argparse.Namespace, folder: Path) -> dict[str, Any]:
    """state.json as the day folder holds it, before anything is judged against it."""
    given = [f"--{name.replace('_', '-')}" if name != "script" else "the script"
             for name in _FIXED_FLAGS
             if getattr(args, name) is not None and getattr(args, name) is not False]
    if given:
        raise DayStop(f"--resume takes the inputs {folder}/state.json recorded; "
                      f"{', '.join(given)} given again. Start a new day folder to change them",
                      "-")
    try:
        state: dict[str, Any] = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DayStop(f"{folder} is not a day folder: {exc}", "-") from None
    if not isinstance(state, dict) or state.get("format") != STATE_FORMAT:
        raise DayStop(f"{folder}/state.json is not {STATE_FORMAT}", "-")
    return state


def _delivered(state: dict[str, Any]) -> Path | None:
    """The upload this day folder made, when latest points at it; a stop if it is gone.
    One latest does not point at is taken up by Day.recover."""
    uploads = state.get("uploads") or []
    if not uploads:
        return None
    target = Path(uploads[-1]["folder"])
    if not (target / NOTE_NAME).is_file():
        raise DayStop(f"this day folder delivered to {target}, which is gone or incomplete; "
                      "start a new day folder", "-")
    latest = target.parent / "latest"
    return target if latest.is_symlink() and os.readlink(latest) == target.name else None


def _resumed(args: argparse.Namespace, commit: dict[str, Any], folder: Path,
             state: dict[str, Any]) -> dict[str, Any]:
    """state.json, checked against this checkout and GMAT; written by the caller."""
    if state["commit"].get("commit") != commit.get("commit"):
        raise DayStop(f"{folder} was run by commit {state['commit'].get('commit')}, and this "
                      f"is {commit.get('commit')}: one day folder, one commit", "-")
    gmat = state["gmat"]
    for key in ("console", "startup"):
        path = Path(gmat[key])
        if not path.is_file() or file_sha256(path) != gmat[f"{key}_sha256"]:
            raise DayStop(f"{path} is not the file this day folder ran (SHA-256 "
                          f"{gmat[f'{key}_sha256']})", "-")
    _check_final_root(Path(state["final_root"]))
    inputs = state["inputs"]
    entered: dict[str, str] = {}
    if args.announcement:
        if not inputs.get("announcement"):
            raise DayStop(f"{folder} took T_max as a number; give the score with its flags", "5")
        if _score_flags(args):
            raise DayStop("give the score as two entries or as flags, not both", "5")
        entries, horizon = _entries(args, inputs["base"], inputs["pursuer_name"],
                                    inputs["target_name"], "5",
                                    folder / "input" / "official.script")
        boundary = (None if entries[2].boundary is None
                    else float(entries[2].boundary.value))
        # Entries that give the score alone leave T_max and the boundary as recorded.
        for what, theirs, ours in (("T_max", horizon, inputs["horizon_s"]),
                                   ("the boundary", boundary, inputs["mission_boundary_km"])):
            if theirs is not None and theirs != ours:
                raise DayStop(f"the entries put {what} at {theirs!r} and {folder} at "
                              f"{ours!r}; read the announcement again", "5")
        score = _entry_score(entries[2], "5")
        if score is None:
            raise DayStop("the entries give no [score]; enter it once it is announced", "5")
        if inputs["score"] is None:
            companions = {f"input/{rel}": sha
                          for rel, sha in state["official"]["companions"].items()}
            entered = _copy_entries(folder, entries, "score-announcement",
                                    {**companions, **state.get("announcement_files", {})}, "5")
    elif inputs.get("announcement") and _score_flags(args):
        raise DayStop(f"{folder} took T_max from two entries; give the score the same way, "
                      "--announcement twice", "5")
    else:
        score = _score(args)
    recorded = state["inputs"]["score"]
    if score is not None and recorded is not None and score != recorded:
        raise DayStop(f"the coefficients given, {score}, are not the ones recorded, "
                      f"{recorded}; a value read wrong needs a new day folder", "5")
    if recorded is None:
        state["inputs"]["score"] = score
        if entered:
            inputs["score_announcement"] = sorted(entered)
            state["announcement_files"] = {**state.get("announcement_files", {}), **entered}
    if args.allow_non_main:
        state["allow_non_main"] = True
    state["excluded"] = sorted({*state["excluded"], *(args.exclude or ())}
                               - set(args.include or ()))
    state["min_free_gb"] = args.min_free_gb
    print(f"orbitwin day: resuming {folder}", file=sys.stderr)
    return state


def run_day(args: argparse.Namespace, executor: Executor = run_logged) -> int:
    """`orbitwin day`: 0 with the upload in place, 2 at a stop, 3 paused for a person."""
    global _stopping, _ctrl_c_seen
    _stopping = _ctrl_c_seen = False
    try:
        with _sigint_handler(_day_ctrl_c):
            return _run_day(args, executor)
    finally:
        _stopping = _ctrl_c_seen = False


def _run_day(args: argparse.Namespace, executor: Executor) -> int:
    global _stopping
    folder: Path | None = None
    day: Day | None = None
    final_root: Path | None = None
    try:
        with ExitStack() as stack:
            if args.resume:
                folder = Path(args.resume).expanduser().resolve()
                if not folder.is_dir():
                    raise DayStop(f"{folder} is not a day folder", "-")
                # Before state.json is read or written.
                stack.enter_context(_locked(folder))
                state = _recorded(args, folder)
                delivered = _delivered(state)
                if delivered is not None:
                    print(f"orbitwin day: {folder} already delivered to {delivered}; nothing "
                          "was changed. A new day needs a new day folder.", file=sys.stderr)
                    return 0
                commit = provenance(ROOT, state["inputs"].get("expect_commit"),
                                    args.allow_non_main)
                state = _resumed(args, commit, folder, state)
            else:
                commit = provenance(ROOT, args.expect_commit, args.allow_non_main)
                folder, state = _new_day(args, commit)
                stack.enter_context(_locked(folder))
            final_root = Path(state["final_root"])
            _reserve(folder)
            state.setdefault("runs", []).append(_run_record(args, commit))
            _write_json(folder / "state.json", state)
            day = Day(folder, state, executor, args)
            target = day.run()
            # Each ending sets _stopping first: a Ctrl-C held down would otherwise cut short
            # what it says.
            _stopping = True
    except KeyboardInterrupt:
        _stopping = True
        folder = folder or _resumed_folder(args)
        print("orbitwin day: interrupted." + _go_on(folder, day)
              + _where(folder, day, final_root), file=sys.stderr)
        return EXIT_STOP
    except DayPaused as paused:
        _stopping = True
        print(f"orbitwin day: PAUSED: {paused}{_where(folder, day, final_root)}",
              file=sys.stderr)
        return EXIT_PAUSED
    except DayStop as stop:
        _stopping = True
        follow = ("" if stop.runbook == "-"
                  else f"\norbitwin day: Follow docs/RUNBOOK.md step {stop.runbook}.")
        print(f"orbitwin day: STOP: {stop}{follow}{_where(folder, day, final_root)}",
              file=sys.stderr)
        return EXIT_STOP
    except Exception as exc:
        _stopping = True
        # A full disk, a bug: no gate said it, so say what failed, where, and how to go on.
        if not isinstance(exc, OSError):
            traceback.print_exc()
        folder = folder or _resumed_folder(args)
        doing = "starting" if day is None else day.doing
        full = (" The disk is full: free space there first."
                if isinstance(exc, OSError) and exc.errno == errno.ENOSPC else "")
        print(f"orbitwin day: FAILED while {doing}: {type(exc).__name__}: {exc}.{full}"
              + _go_on(folder, day) + _where(folder, day, final_root), file=sys.stderr)
        return EXIT_STOP
    note = (target / NOTE_NAME).read_text(encoding="utf-8")
    print(note, end="")
    script = next(line.split(" : ", 1)[1] for line in note.splitlines()
                  if line.startswith("script "))
    report = next(line.split(" : ", 1)[1] for line in note.splitlines()
                  if line.startswith("report "))
    print(f"orbitwin day: {target.parent / 'latest'} points at {target}.\n"
          f"orbitwin day: upload exactly these two files: {target / script} {target / report}",
          file=sys.stderr)
    return 0


def _resumed_folder(args: argparse.Namespace) -> Path | None:
    """The folder a resume names, for a stop that came before it was taken up."""
    return Path(args.resume).expanduser().resolve() if args.resume else None


def _go_on(folder: Path | None, day: Day | None) -> str:
    """How to go on after an interrupt or a failure: the resume, or a new day folder."""
    if folder is None or not (folder / "state.json").is_file():
        return " No day folder was recorded: run the day again from the start."
    resume = day.resume if day is not None else _resume_command(
        folder, _recorded_non_main(folder))
    return f" Finished steps are kept. Go on with:\n\n{resume}\n"


def _where(folder: Path | None, day: Day | None, final_root: Path | None) -> str:
    """Where a stop leaves things, a line each: the day folder, what a cut-short copy left,
    latest."""
    lines = [] if folder is None else [f"Day folder: {folder}."]
    left = None if day is None else day.left
    made = None if day is None else day.made
    moved = (made is not None and final_root is not None
             and (final_root / "latest").is_symlink()
             and (final_root / "latest").resolve() == made.resolve())
    lines.append(f"{made} holds this day folder's proven upload, and latest points at it."
                 if moved
                 else f"{made} holds this day folder's proven upload; latest was not moved to it."
                 if made is not None
                 else f"Nothing was copied for upload; {left} was left and is not an upload."
                 if left is not None else "Nothing was copied for upload.")
    if final_root is not None:
        lines.append(f"{_latest_note(final_root)}.")
    return "".join(f"\norbitwin day: {line}" for line in lines)
