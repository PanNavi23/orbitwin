#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_manifest import (
    assert_tree_matches_git,
    build_manifest,
    executable_files,
    tracked_files,
    tree_differences,
)

PROJECT_DIRNAME = "tasa_orbitwin"
FIXED_TIMESTAMP = (2026, 8, 18, 0, 0, 0)
#: The offline bundle's manifest. It is tracked; the wheels it lists are not.
WHEEL_MANIFEST = Path("vendor") / "wheels" / "SHA256SUMS.txt"
#: The commit the archive holds, for `orbitwin --version`. Written for the
#: archive only: one left in the checkout would name a commit it has moved past.
BUILD_INFO = Path("BUILD_INFO.json")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


#: What clean() removes: caches, which never reach the archive anyway. It
#: also removes src/*.egg-info, coverage data and bytecode. scripts/clean.sh,
#: which `make clean` runs, is this function in shell.
CLEANED = (".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov")
#: Not searched for bytecode, by name at any depth: the repository, runs and environments.
_NOT_WALKED = frozenset({".git", ".venv", "output", "dist", "build"})


def _remove(path: Path) -> None:
    """Remove a file, a tree, or a symbolic link itself, never what it points at."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _sweep_bytecode(directory: Path) -> None:
    """Every __pycache__ and *.pyc/*.pyo below, of any type; no link is followed.

    Another checkout below, a directory holding .git (a worktree has a .git file), is not
    searched: its bytecode is its own.
    """
    try:
        entries = list(os.scandir(directory))
    except OSError as error:
        raise RuntimeError(f"clean: cannot list {directory}: {error}") from error
    for entry in entries:
        if entry.name in _NOT_WALKED:
            continue
        path = Path(entry.path)
        if entry.name == "__pycache__":
            _remove(path)
        elif entry.is_dir(follow_symlinks=False):
            if not os.path.lexists(os.path.join(entry.path, ".git")):
                _sweep_bytecode(path)
        elif entry.name.endswith((".pyc", ".pyo")):
            _remove(path)


def clean(root: Path, output_dir: Path | None = None) -> None:
    """Remove caches and bytecode, sparing the directory being written to.

    The archive is enumerated from git, and make_manifest.included() drops
    output/, build/ and dist/ even where git tracks a file in them, so nothing
    this could remove reaches the release. It used to remove output/, build/
    and dist/ as well: `make package` deleted every earlier run and every
    earlier release, and with `PACKAGE_DIR=output/offline/dist` or
    `dist/offline` the destination too, which failed with FileNotFoundError.
    A destination inside a cache directory still spares that directory.

    A symbolic link with a matching name is removed, not followed, and a
    matching name is removed whatever it is: scripts/clean.sh does the same.
    """
    spared = output_dir.resolve() if output_dir is not None else None
    for name in CLEANED:
        target = root / name
        if spared is not None and spared.is_relative_to(target.resolve()):
            continue
        _remove(target)
    _sweep_bytecode(root)
    # Metadata setuptools leaves on a build; installs read their own copy.
    if (root / "src").is_dir():
        for entry in os.scandir(root / "src"):
            if entry.name.endswith(".egg-info") and not entry.name.startswith("."):
                _remove(Path(entry.path))
    # .coverage.* are parallel-mode data an interrupted run did not combine.
    for entry in os.scandir(root):
        if entry.name == ".coverage" or entry.name.startswith(".coverage."):
            _remove(Path(entry.path))


#: A line as fetch_offline_wheels.sh writes it (`sha256sum ./*.whl`), or in
#: sha256sum's binary form: a lowercase digest, one space, ' ' or '*', then a
#: .whl in vendor/wheels itself. That is a strict subset of what `sha256sum -c`
#: accepts (it also takes upper case, CRLF and comments): anything else did
#: not come from the fetch, and the manifest is the record of what was tested.
_WHEEL_LINE = re.compile(r"([0-9a-f]{64}) [ *](?:\./)?([^/\s*]+\.whl)")


def wheel_bundle(root: Path) -> list[Path]:
    """The wheels vendor/wheels/SHA256SUMS.txt lists, each checked against it.

    The wheels are gitignored, so tracked_files() leaves them out, and an
    archive built from it alone failed RUNBOOK step 0 on the first line:
    `sha256sum -c` found none of the 38 files, and bootstrap_offline.sh then
    stops with "no wheels". Unpacked on a machine with no network, that
    archive cannot be installed. So the listed wheels go in, and only those.
    A manifest that is missing, empty or malformed, or a listed wheel that is
    missing, a link or changed, is refused here, where it can still be fixed.
    """
    manifest = root / WHEEL_MANIFEST
    problems: list[str] = []
    wheels: list[Path] = []
    if not manifest.is_file():
        problems.append(f"{WHEEL_MANIFEST} is missing")
    else:
        # Bytes, not read_text(): universal newlines would hide a CRLF line.
        try:
            lines = manifest.read_bytes().decode("utf-8").split("\n")
        except UnicodeDecodeError as error:
            problems.append(f"{WHEEL_MANIFEST} is not UTF-8 text ({error.reason})")
            lines = []
        if lines and lines[-1] == "":
            lines.pop()
        seen: set[str] = set()
        for number, line in enumerate(lines, start=1):
            match = _WHEEL_LINE.fullmatch(line)
            if match is None:
                problems.append(f"line {number} is not `<sha256>  ./<name>.whl`: {line!r}")
                continue
            expected, name = match.groups()
            path = manifest.parent / name
            if name in seen:
                problems.append(f"{name}: listed twice")
            elif path.is_symlink():
                problems.append(f"{name}: a symbolic link")
            elif not path.is_file():
                problems.append(f"{name}: missing")
            elif sha256(path) != expected:
                problems.append(f"{name}: differs from the manifest")
            else:
                wheels.append(path)
            seen.add(name)
        if not lines and not problems:
            problems.append(f"{WHEEL_MANIFEST} lists no wheels")
    if problems:
        raise RuntimeError(
            "The offline wheel bundle is not the one its manifest describes, so the archive "
            "could not be installed with no network (docs/RUNBOOK.md step 0):\n  "
            + "\n  ".join(problems)
            + f"\nPut back the wheels {WHEEL_MANIFEST} lists, or refetch them with "
            "scripts/fetch_offline_wheels.sh (it needs the network). If the manifest it "
            "writes differs from the committed one, the bundle is a different set from the "
            "one tested: commit it and test again. --without-wheels builds an archive that "
            "needs the network to install."
        )
    return sorted(wheels)


def verify_wheel_bundle(extracted_root: Path, expected: int) -> None:
    """Check the extracted bundle with RUNBOOK step 0's own command, then count it."""
    directory = extracted_root / WHEEL_MANIFEST.parent
    command = ["sha256sum", "--check", "--strict", "--quiet", WHEEL_MANIFEST.name]
    try:
        completed = subprocess.run(
            command, cwd=directory, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as missing:
        raise RuntimeError(
            "sha256sum is not installed here; RUNBOOK step 0 checks the bundle with it"
        ) from missing
    if completed.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(command)}` fails in the extracted {WHEEL_MANIFEST.parent}:\n"
            + completed.stdout + completed.stderr
        )
    found = len(wheel_bundle(extracted_root))
    if found != expected:
        raise RuntimeError(f"the extracted archive holds {found} wheels, not {expected}")


def build_identity(root: Path, version: str) -> dict[str, object]:
    """HEAD, its commit time in UTC, and whether a tracked file differs from it.

    The commit time, not the build's: two builds of one commit stay the same bytes.
    """

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"`git {' '.join(args)}` failed in {root}; a release must be built from a git "
                f"checkout: {completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    commit = git("rev-parse", "--verify", "HEAD^{commit}")
    seconds = int(git("show", "-s", "--format=%ct", commit))
    return {
        "artifact": PROJECT_DIRNAME,
        "version": version,
        "commit": commit,
        "committed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds)),
        "dirty": bool(tree_differences(root)),
        "manifest": "SHA256SUMS.txt",
    }


#: Where a release comes from; any other HEAD needs --allow-non-main.
RELEASE_REF = "refs/remotes/origin/main"
#: How long the fetch before a build may take; past it the local ref is used.
FETCH_TIMEOUT_S = 60.0


def _git_quiet(root: Path, *args: str, timeout_s: float | None = None) -> str | None:
    """git's output, or None when it fails or times out."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False,
            timeout=timeout_s, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _age(seconds: float) -> str:
    hours = seconds / 3600.0
    return f"{hours:.1f} h" if hours < 48.0 else f"{hours / 24.0:.1f} days"


def release_provenance(
    root: Path, commit: str, version: str, allow_non_main: bool,
    now_s: float | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Where HEAD stands against origin/main, fetched first when the network allows.

    The main checkout sat for days on a stale branch, and `make package`
    archived whatever HEAD it had. HEAD must be origin/main, or the commit of
    the release tag v<VERSION>; anything else only with --allow-non-main, which
    the archive's BUILD_INFO.json records.
    """
    notes: list[str] = []
    fetched = _git_quiet(root, "fetch", "--quiet", "origin", "main",
                         timeout_s=FETCH_TIMEOUT_S) is not None
    main = _git_quiet(root, "rev-parse", "--verify", "-q", f"{RELEASE_REF}^{{commit}}")
    tag = f"v{version}"
    tagged = _git_quiet(root, "rev-parse", "--verify", "-q", f"refs/tags/{tag}^{{commit}}")
    branch = _git_quiet(root, "symbolic-ref", "-q", "--short", "HEAD") or "(detached HEAD)"
    age_s = None
    if not fetched and main is not None:
        # The reflog's newest entry is when the ref last moved here: the last fetch that did.
        entry = _git_quiet(root, "log", "-g", "-n1", "--format=%gd", "--date=unix", RELEASE_REF)
        stamp = entry.rsplit("@{", 1)[-1].rstrip("}") if entry else ""
        if stamp.isdigit():
            age_s = (time.time() if now_s is None else now_s) - int(stamp)
        notes.append(
            "could not fetch origin main; compared with the local origin/main, "
            + ("last updated " + _age(age_s) + " ago" if age_s is not None
               else "of unknown age (no reflog)")
        )
    matches = "origin/main" if main == commit else tag if tagged == commit else None
    record: dict[str, object] = {
        "branch": branch,
        "origin_main": main,
        "origin_main_fetched": fetched,
        "origin_main_age_s": age_s,
        "matches": matches,
        "allow_non_main": allow_non_main,
    }
    if matches is not None:
        return record, notes
    against = "no origin/main" if main is None else f"origin/main is {main[:12]}"
    where = f"HEAD is {commit[:12]} on {branch}; {against}" + (
        f", and {tag} is {tagged[:12]}" if tagged else f", and there is no tag {tag}"
    )
    if not allow_non_main:
        raise RuntimeError(
            f"{where}. A release is built from origin/main or its release tag; check out "
            "origin/main and build again, or pass --allow-non-main to archive this commit "
            "anyway, recorded in BUILD_INFO.json" + (f" ({notes[0]})" if notes else "")
        )
    notes.append(f"NOT ORIGIN/MAIN (--allow-non-main): {where}")
    return record, notes


def create_zip(
    root: Path, archive: Path, wheels: list[Path] | None = None, comment: str = ""
) -> None:
    archive.unlink(missing_ok=True)
    # The manifests are generated immediately before packaging, so they are not
    # tracked files and have to be added explicitly.
    payload = sorted(
        {*tracked_files(root), root / "SHA256SUMS.txt", root / BUILD_INFO, *(wheels or [])}
    )
    executables = executable_files(root)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # `unzip -z` shows it: which commit an archive under a reused name came from.
        zf.comment = comment.encode("utf-8")
        for path in payload:
            if not path.is_file():
                continue
            relative = Path(PROJECT_DIRNAME) / path.relative_to(root)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=FIXED_TIMESTAMP)
            # A wheel is a zip already; deflating it again saves nothing.
            info.compress_type = (
                zipfile.ZIP_STORED if path.suffix == ".whl" else zipfile.ZIP_DEFLATED
            )
            # Git records only 0644 and 0755; the bits on disk add the umask of
            # whoever checked out or fetched, or the drive's own. Storing git's
            # keeps the archive's bytes a function of the commit and the bundle.
            info.external_attr = (0o755 if path in executables else 0o644) << 16
            zf.writestr(info, path.read_bytes())


def extract_all_with_modes(zf: zipfile.ZipFile, destination: Path) -> None:
    """Extract an archive, restoring the Unix permission bits it stores.

    ``ZipFile.extractall`` deliberately ignores ``external_attr``, so a file
    archived as 0755 comes back as 0644. ``create_zip`` records the mode
    correctly; without this the executable bit is lost at extraction, which
    silently disables every test that spawns scripts/fake_gmat_console.py as a
    process -- that is, the whole GMAT orchestration path.
    """
    for info in zf.infolist():
        target = zf.extract(info, destination)
        mode = (info.external_attr >> 16) & 0o777
        if mode:
            os.chmod(target, mode)


def verify_manifest(extracted_root: Path) -> None:
    manifest = extracted_root / "SHA256SUMS.txt"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = extracted_root / relative
        if not path.is_file():
            raise RuntimeError(f"Manifest path missing after extraction: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Manifest mismatch for {relative}: {actual} != {expected}")


#: The extracted suite is called hung when its output stops growing this long.
#:
#: It is the `make test` selection in one process. That took about four
#: minutes when this guard was a 1800 s wall clock. On 2026-11-02 (2270 tests,
#: a 20-core machine shared with other GMAT runs) one `make package` passed in
#: 1798 s and the next was stopped at 1800 s and called a hang; the same suite
#: from the same archive, run beside it, passed in 3032 s. A wall clock cannot
#: tell a slow machine from a stuck suite. pytest prints a character per test
#: and flushes it, so a long silence can.
EXTRACTED_TEST_STALL_S = 600.0
#: Before its first byte the suite is starting: Python, pytest, its plugins,
#: collection. That silence has a limit of its own, so a silence limit
#: shortened for a test does not also time the start-up.
EXTRACTED_TEST_STARTUP_S = 600.0
#: A backstop for a suite that keeps printing and never ends.
EXTRACTED_TEST_TIMEOUT_S = 4 * 3600.0
_EXTRACTED_TEST_POLL_S = 5.0
#: How long a stopped suite gets, after SIGINT, to print where it was.
_EXTRACTED_TEST_GRACE_S = 10.0


def _stop_group(process: subprocess.Popen[bytes], grace_s: float = 0.0) -> None:
    """Stop the suite and every process in its group, such as pool workers.

    Killing pytest alone left its ProcessPoolExecutor workers running. The
    suite has a session of its own, so its process group is all of them
    except what starts a session of its own (gmat_runner's consoles), which
    ends when its own timeout does. With a grace period the group gets SIGINT
    first. The SIGKILL comes whatever happens during the grace period: a
    second Ctrl-C there used to skip it and leave the suite running.
    """
    try:
        if grace_s > 0:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGINT)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=grace_s)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def _default_signals() -> None:
    """In the suite, before exec: SIGINT and the rest back to their defaults.

    Started from a script with `&`, the build inherits SIGINT ignored, and
    the suite with it; the stop's SIGINT then did nothing and its log said
    nothing about where it was.
    """
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(number, signal.SIG_DFL)


def _watch(
    process: subprocess.Popen[bytes], log_fd: int, stall_s: float, timeout_s: float,
    poll_s: float, startup_s: float,
) -> int:
    """Wait for the suite while its log grows; raise when it stops or runs too long.

    Silence counts against stall_s from the first byte on, and against
    startup_s before it.
    """
    started = time.monotonic()
    size, changed = 0, started
    while True:
        try:
            return process.wait(timeout=poll_s)
        except subprocess.TimeoutExpired:
            pass
        now, current = time.monotonic(), os.fstat(log_fd).st_size
        if current != size:
            size, changed = current, now
        silent, elapsed = now - changed, now - started
        limit = stall_s if size else startup_s
        if silent < limit and elapsed < timeout_s:
            continue
        if elapsed >= timeout_s:
            reason = f"were still running at {timeout_s:.0f} s"
        elif size:
            reason = f"printed nothing for {silent:.0f} s"
        else:
            reason = f"printed nothing in their first {silent:.0f} s"
        # faulthandler dumps a test that runs 300 s, not a start-up that hangs.
        where = ("a test that ran 300 s has its stack there (faulthandler), and the end shows "
                 "where SIGINT found them" if size else "the end shows where SIGINT found "
                 "them, still starting")
        raise RuntimeError(
            f"Fast tests after extracting the release archive {reason}, and were stopped "
            f"after {elapsed:.0f} s. What they printed is copied next to the archive as "
            f"tasa_orbitwin-<version>-<commit>.extracted-smoke-failure.log; {where}."
        )


def extracted_smoke_test(
    extracted_root: Path,
    command: list[str] | None = None,
    stall_s: float = EXTRACTED_TEST_STALL_S,
    timeout_s: float = EXTRACTED_TEST_TIMEOUT_S,
    poll_s: float = _EXTRACTED_TEST_POLL_S,
    grace_s: float = _EXTRACTED_TEST_GRACE_S,
    startup_s: float = EXTRACTED_TEST_STARTUP_S,
) -> None:
    """Run the fast suite in the extracted tree, its output going to the log as it comes.

    The log is written while the suite runs, so a stopped run leaves one that
    shows how far it got. pytest flushes each result itself; PYTHONUNBUFFERED
    does the same for anything else the suite's processes print.
    faulthandler_timeout makes pytest write the stack of a test that has run
    for 300 s into the log. That output restarts the stall clock, so one
    stuck test is stopped after 300 + 600 s.

    In its own session the suite no longer gets the terminal's SIGINT or
    SIGHUP, so this process turns SIGTERM and SIGHUP into the same stop as
    Ctrl-C while it runs.
    """
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(extracted_root / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    if command is None:
        command = [
            sys.executable, "-m", "pytest", "-m", "not slow and not gmat",
            "-o", "faulthandler_timeout=300",
        ]
    log = extracted_root / "output" / "extracted_fast_test.log"
    log.parent.mkdir(exist_ok=True)
    # SIGTERM, SIGHUP and Ctrl-\ (SIGQUIT) stop the suite as Ctrl-C does. One
    # this process was told to ignore (nohup, `&` in a script) stays ignored.
    handled = (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
    previous: dict[signal.Signals, object] = {}
    if threading.current_thread() is threading.main_thread():
        for handled_signal in handled:
            if signal.getsignal(handled_signal) is not signal.SIG_IGN:
                previous[handled_signal] = signal.signal(handled_signal, _interrupt)
    try:
        with log.open("wb") as handle:
            process = subprocess.Popen(
                command, cwd=extracted_root, env=environment, stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                preexec_fn=_default_signals,
            )
            try:
                returncode = _watch(
                    process, handle.fileno(), stall_s, timeout_s, poll_s, startup_s
                )
            except BaseException:
                # A stall or a signal: pass SIGINT on, then stop the rest. The
                # message must not stand in the way: with the terminal closed or
                # a `| tee` killed by the same Ctrl-C, writing it raises OSError.
                try:
                    with contextlib.suppress(OSError):
                        print(
                            f"stopping the extracted suite: SIGINT, then SIGKILL within "
                            f"{grace_s:.0f} s (Ctrl-C again stops it at once)",
                            file=sys.stderr, flush=True,
                        )
                finally:
                    _stop_group(process, grace_s)
                raise
            _stop_group(process)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)  # type: ignore[arg-type]
    if returncode != 0:
        raise RuntimeError(
            f"Fast tests failed after extracting the release archive (pytest exited "
            f"{returncode}). The full output is copied next to the archive as "
            "tasa_orbitwin-<version>-<commit>.extracted-smoke-failure.log"
        )


# Relative to the project root, and gitignored. The previous default was an
# absolute path that exists inside one container image and nowhere else, so
# `make package` failed with a bare ENOENT on every other machine -- including
# the competition laptop, which is the one run that has to work.
DEFAULT_OUTPUT_DIR = "dist"


def resolve_output_dir(project_root: Path, output_dir: str) -> Path:
    """The directory the archive is written to, created if it is absent.

    A relative path is resolved against the project root rather than the
    caller's working directory, so `make package` from a subdirectory lands
    where `make package` from the root does. An absolute path is taken as given.
    """
    resolved = Path(output_dir)
    if not resolved.is_absolute():
        resolved = project_root / resolved
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-extracted-test", action="store_true")
    parser.add_argument(
        "--without-wheels", action="store_true",
        help="leave the vendor/wheels bundle out; the archive then needs the network to install",
    )
    parser.add_argument(
        "--allow-non-main", action="store_true",
        help="archive a HEAD that is neither origin/main nor the release tag; BUILD_INFO.json "
        "records its branch and commit",
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    identity = build_identity(root, version)
    commit = str(identity["commit"])
    provenance, provenance_notes = release_provenance(
        root, commit, version, args.allow_non_main
    )
    for note in provenance_notes:
        print(f"build_release: {note}", file=sys.stderr)
    # A different name, so an archive that needs the network is never taken for one that does not.
    suffix = "-without-wheels" if args.without_wheels else ""
    # The commit in the name: every archive was tasa_orbitwin-0.3.0.zip, and
    # what said which commit it held, the zip comment, is lost on extraction.
    archive = (
        resolve_output_dir(root, args.output_dir)
        / f"tasa_orbitwin-{version}-{commit[:12]}{suffix}.zip"
    )
    # Built and checked under this name, renamed only once every check passed:
    # an archive left by a failed build would look like a release.
    partial = archive.with_name(archive.name + ".partial")
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    smoke_log = archive.with_name(archive.name.removesuffix(".zip") + ".extracted-smoke-failure.log")
    # An earlier archive and its sidecar stay until this build replaces them;
    # a run that fails says so (see the bottom of this file). What a failed run
    # left is removed.
    _building["archive"] = archive
    sidecar_partial = sidecar.with_name(sidecar.name + ".partial")
    for stale in (partial, sidecar_partial, smoke_log):
        stale.unlink(missing_ok=True)
    # Before anything is cleaned or generated: the archive has to describe a
    # commit, so refuse a tree that does not match one.
    assert_tree_matches_git(root)
    wheels = [] if args.without_wheels else wheel_bundle(root)
    clean(root, archive.parent)
    if build_identity(root, version) != identity:
        raise RuntimeError("HEAD or a tracked file changed during the build; build again")
    info = root / BUILD_INFO
    try:
        info.write_text(json.dumps({**identity, "provenance": provenance}, indent=2) + "\n",
                        encoding="utf-8")
        build_manifest(root, root / "SHA256SUMS.txt", [info, *wheels])
        create_zip(root, partial, wheels, f"{archive.name} from commit {commit}")
    finally:
        info.unlink(missing_ok=True)
    with zipfile.ZipFile(partial) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
    with tempfile.TemporaryDirectory(prefix="orbitwin-release-check-") as temp:
        with zipfile.ZipFile(partial) as zf:
            extract_all_with_modes(zf, Path(temp))
        extracted = Path(temp) / PROJECT_DIRNAME
        verify_manifest(extracted)
        if not args.without_wheels:
            verify_wheel_bundle(extracted, len(wheels))
        if not args.skip_extracted_test:
            try:
                extracted_smoke_test(extracted)
            except BaseException:  # Ctrl-C too: the log says where the suite was
                failure_log = extracted / "output" / "extracted_fast_test.log"
                if failure_log.is_file():
                    shutil.copy2(failure_log, smoke_log)
                raise
    # Every check passed. The new sidecar is written aside first, since that
    # write is the one step that can fail for lack of space; only then does the
    # earlier archive go, and after it only renames remain. Interrupted
    # anywhere, there is no archive beside a sidecar that does not match it.
    zip_hash = sha256(partial)
    sidecar_partial.write_text(f"{zip_hash}  {archive.name}\n", encoding="utf-8")
    archive.unlink(missing_ok=True)
    # From here an archive under this name can only be this run's.
    _building["replaced"] = True
    os.replace(sidecar_partial, sidecar)
    os.replace(partial, archive)
    print(f"archive={archive}")
    # Written down before the day, it is what RUNBOOK step 0 checks the laptop against.
    print(f"commit={commit}")
    if provenance["matches"] is None:
        print(f"provenance=NOT origin/main: branch {provenance['branch']}, commit {commit} "
              "(--allow-non-main)")
    else:
        print(f"provenance={provenance['matches']}")
    print(f"sha256={zip_hash}")
    print(f"sidecar={sidecar}")
    print(f"wheels={len(wheels)}" + (" (--without-wheels: needs the network to install)"
                                     if args.without_wheels else ""))
    return 0


#: The archive main() is building, and whether it got as far as replacing it.
_building: dict[str, object] = {}


def _note_earlier_archive() -> None:
    """A failed run leaves an earlier archive in place; say it is not this run's."""
    archive = _building.get("archive")
    if not isinstance(archive, Path) or not archive.is_file() or _building.get("replaced"):
        return
    try:
        with zipfile.ZipFile(archive) as zf:
            origin = zf.comment.decode("utf-8", errors="replace") or "no commit recorded"
    except Exception:
        # A note must not hide the error it follows: zipfile also raises
        # NotImplementedError and UnicodeDecodeError on a damaged archive.
        origin = "not a readable zip archive"
    print(
        f"note: {archive} and its sidecar are from an earlier build ({origin}); "
        "this run did not replace them.",
        file=sys.stderr,
    )


def _run() -> int:
    """main(), with a refusal printed as its message: the reason, not a traceback."""
    try:
        return main()
    except RuntimeError as refused:
        print(f"build_release: refused: {refused}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(_run())
    except BaseException as stopped:
        if not (isinstance(stopped, SystemExit) and stopped.code in (0, None)):
            _note_earlier_archive()
        raise
