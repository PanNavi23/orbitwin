"""The offline wheel bundle in the release archive (RUNBOOK step 0)."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
from conftest import scratch_git_environment


def _load_build_release(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "orbitwin_build_release_wheels", project_root / "scripts" / "build_release.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WHEEL = "demo-1.0-py3-none-any.whl"
FETCHED = b"the wheel that was fetched"


def _digest(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> None:
    import subprocess

    subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True,
        env=scratch_git_environment(repo.parent),
    )


def _checkout_with_a_bundle(
    project_root: Path, tmp_path: Path, wheel_bytes: bytes = FETCHED, manifest: str | None = None
) -> Path:
    """A git checkout holding build_release.py and a one-wheel bundle, the wheel untracked."""
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git is not installed; packaging needs a git checkout")
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("build_release.py", "make_manifest.py"):
        (repo / "scripts" / name).write_bytes((project_root / "scripts" / name).read_bytes())
    (repo / "VERSION").write_text("0.0.0\n", encoding="utf-8")
    (repo / ".gitignore").write_text("vendor/wheels/*.whl\n", encoding="utf-8")
    wheels = repo / "vendor" / "wheels"
    wheels.mkdir(parents=True)
    (wheels / WHEEL).write_bytes(wheel_bytes)
    if manifest is None:
        manifest = f"{_digest(FETCHED)}  ./{WHEEL}\n"
    (wheels / "SHA256SUMS.txt").write_text(manifest, encoding="utf-8", newline="")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    _git(repo, "init", "-q", "--bare", str(tmp_path / "origin.git"))
    _git(repo, "remote", "add", "origin", str(tmp_path / "origin.git"))
    _git(repo, "push", "-q", "origin", "HEAD:refs/heads/main")
    return repo


def _package(repo: Path, *flags: str, publish: bool = True, **environment: str):
    """build_release.py in ``repo``; with ``publish``, HEAD is first pushed as origin/main."""
    import subprocess
    import sys

    if publish:
        _git(repo, "push", "-q", "-f", "origin", "HEAD:refs/heads/main")
    return subprocess.run(
        [sys.executable, "scripts/build_release.py", "--project-root", ".",
         "--output-dir", "dist", "--skip-extracted-test", *flags],
        cwd=repo, capture_output=True, text=True, check=False,
        env={**scratch_git_environment(repo.parent), **environment},
    )


def _head(repo: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
        check=True, env=scratch_git_environment(repo.parent),
    ).stdout.strip()


def _archive(repo: Path, suffix: str = "") -> Path:
    """The archive of the commit checked out: its name carries the commit."""
    return repo / "dist" / f"tasa_orbitwin-0.0.0-{_head(repo)[:12]}{suffix}.zip"


def test_the_archive_carries_the_wheel_bundle(project_root: Path, tmp_path: Path) -> None:
    """RUNBOOK step 0 installs from vendor/wheels with no network.

    The wheels are gitignored and the archive was built from git alone, so the
    unpacked release had the manifest and none of the 38 wheels it lists.
    """
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    (repo / "vendor" / "wheels" / "unlisted-1.0-py3-none-any.whl").write_bytes(b"stray")

    completed = _package(repo)

    assert completed.returncode == 0, completed.stderr
    assert "wheels=1" in completed.stdout
    with zipfile.ZipFile(_archive(repo)) as zf:
        info = zf.getinfo(f"tasa_orbitwin/vendor/wheels/{WHEEL}")
        assert zf.read(info) == FETCHED
        assert info.compress_type == zipfile.ZIP_STORED
        names = zf.namelist()
        root_manifest = zf.read("tasa_orbitwin/SHA256SUMS.txt").decode("utf-8")
    assert "tasa_orbitwin/vendor/wheels/SHA256SUMS.txt" in names
    assert not [name for name in names if "unlisted" in name], "only listed wheels ship"
    assert f"{_digest(FETCHED)}  vendor/wheels/{WHEEL}" in root_manifest.splitlines()


def test_two_builds_of_one_commit_are_the_same_bytes(project_root: Path, tmp_path: Path) -> None:
    """Whatever mode the fetch or a copy left on the wheels, executable included."""
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    archive = _archive(repo)
    wheel = repo / "vendor" / "wheels" / WHEEL

    wheel.chmod(0o600)
    assert _package(repo).returncode == 0
    first = archive.read_bytes()
    wheel.chmod(0o755)
    assert _package(repo).returncode == 0

    assert archive.read_bytes() == first
    assert not list((repo / "dist").glob("*.partial")), "renamed, not copied"
    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo(f"tasa_orbitwin/vendor/wheels/{WHEEL}")
    assert (info.external_attr >> 16) & 0o777 == 0o644


def test_every_entry_has_the_mode_git_records(project_root: Path, tmp_path: Path) -> None:
    """Executables stay executable after extraction; the fake console depends on it."""
    import subprocess

    repo = _checkout_with_a_bundle(project_root, tmp_path)
    tool = repo / "scripts" / "tool.sh"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    tool.chmod(0o755)
    _git(repo, "add", "scripts/tool.sh")
    _git(repo, "commit", "-qm", "an executable")
    assert _package(repo).returncode == 0

    staged = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-s"], capture_output=True, text=True, check=True,
        env=scratch_git_environment(tmp_path),
    ).stdout.splitlines()
    recorded = {line.split("\t", 1)[1]: int(line.split()[0][-3:], 8) for line in staged}
    with zipfile.ZipFile(_archive(repo)) as zf:
        stored = {
            info.filename.removeprefix("tasa_orbitwin/"): (info.external_attr >> 16) & 0o777
            for info in zf.infolist()
        }
    assert recorded["scripts/tool.sh"] == 0o755
    for name, mode in recorded.items():
        assert stored[name] == mode, name
    assert stored[f"vendor/wheels/{WHEEL}"] == 0o644
    assert stored["SHA256SUMS.txt"] == 0o644


def test_the_modes_come_from_git_where_the_disk_cannot_hold_them(
    project_root: Path, tmp_path: Path
) -> None:
    """With core.fileMode false, as git sets it on FAT, exFAT or NTFS, git status shows no mode.

    A checkout on such a drive reads every file as 0755, or every one as 0644
    under noexec or fmask, and git calls the tree clean either way. The
    archive took its modes from the disk, so it differed by drive, and the
    fake console's lost executable bit failed the GMAT orchestration tests.
    """
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    # git quotes a name like the second one unless asked for -z.
    tools = [repo / "scripts" / "tool.sh", repo / "scripts" / "\u5de5\u5177 two.sh"]
    for tool in tools:
        tool.write_text("#!/bin/sh\n", encoding="utf-8")
        tool.chmod(0o755)
    _git(repo, "add", "scripts")
    _git(repo, "commit", "-qm", "executables")
    assert _package(repo).returncode == 0
    archive = _archive(repo)
    first = archive.read_bytes()
    _git(repo, "config", "core.fileMode", "false")
    for tool in tools:
        tool.chmod(0o644)
    (repo / "VERSION").chmod(0o755)

    completed = _package(repo)

    assert completed.returncode == 0, completed.stderr
    assert archive.read_bytes() == first
    with zipfile.ZipFile(archive) as zf:
        for tool in tools:
            info = zf.getinfo(f"tasa_orbitwin/scripts/{tool.name}")
            assert (info.external_attr >> 16) & 0o777 == 0o755, tool.name
        assert (zf.getinfo("tasa_orbitwin/VERSION").external_attr >> 16) & 0o777 == 0o644


def test_the_sidecar_checks_the_archive_and_the_archive_names_its_commit(
    project_root: Path, tmp_path: Path
) -> None:
    """Every archive was tasa_orbitwin-0.3.0.zip; only its comment named the commit."""
    import shutil
    import subprocess

    if shutil.which("sha256sum") is None:
        pytest.skip("sha256sum is not installed")
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    completed = _package(repo)
    assert completed.returncode == 0, completed.stderr
    head = _head(repo)
    name = f"tasa_orbitwin-0.0.0-{head[:12]}.zip"

    checked = subprocess.run(
        ["sha256sum", "--check", "--strict", f"{name}.sha256"],
        cwd=repo / "dist", capture_output=True, text=True, check=False,
    )

    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert f"commit={head}\n" in completed.stdout
    with zipfile.ZipFile(repo / "dist" / name) as zf:
        assert zf.comment.decode() == f"{name} from commit {head}"


def test_the_archive_says_which_commit_it_holds_once_extracted(
    project_root: Path, tmp_path: Path
) -> None:
    """The zip comment was the only record, and extraction drops it."""
    import json
    import subprocess

    repo = _checkout_with_a_bundle(project_root, tmp_path)
    assert _package(repo).returncode == 0
    committed = subprocess.run(
        ["git", "-C", str(repo), "show", "-s", "--format=%cd",
         "--date=format-local:%Y-%m-%dT%H:%M:%SZ", "HEAD"], capture_output=True,
        text=True, check=True, env={**scratch_git_environment(tmp_path), "TZ": "UTC"},
    ).stdout.strip()

    with zipfile.ZipFile(_archive(repo)) as zf:
        info = json.loads(zf.read("tasa_orbitwin/BUILD_INFO.json"))
        root_manifest = zf.read("tasa_orbitwin/SHA256SUMS.txt").decode("utf-8")
        digest = _digest(zf.read("tasa_orbitwin/BUILD_INFO.json"))

    assert info == {
        "artifact": "tasa_orbitwin", "version": "0.0.0", "commit": _head(repo),
        "committed_utc": committed, "dirty": False,
        "manifest": "SHA256SUMS.txt",
        "provenance": {
            "branch": info["provenance"]["branch"], "origin_main": _head(repo),
            "origin_main_fetched": True, "origin_main_age_s": None, "matches": "origin/main",
            "allow_non_main": False,
        },
    }
    assert f"{digest}  BUILD_INFO.json" in root_manifest.splitlines()
    assert not (repo / "BUILD_INFO.json").exists(), "a checkout moves past the commit it names"


def test_a_tracked_file_changed_during_the_build_stops_it(
    project_root: Path, tmp_path: Path
) -> None:
    """BUILD_INFO.json would name a commit the archive does not hold."""
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    source = repo / "scripts" / "build_release.py"
    text = source.read_text(encoding="utf-8")
    target = "    clean(root, archive.parent)\n"
    assert text.count(target) == 1
    change = "    (root / 'VERSION').write_text('0.0.0\\n\\n', encoding='utf-8')\n"
    source.write_text(text.replace(target, change + target, 1), encoding="utf-8")
    _git(repo, "commit", "-qam", "change VERSION after the check")

    completed = _package(repo)

    assert completed.returncode != 0
    assert "changed during the build" in completed.stderr
    assert not (repo / "dist").exists() or not list((repo / "dist").glob("*.zip"))
    assert not (repo / "BUILD_INFO.json").exists()


@pytest.mark.parametrize(
    ("flag", "change"),
    [("--assume-unchanged", "edit"), ("--skip-worktree", "edit"), ("--skip-worktree", "delete")],
)
def test_a_file_git_is_told_not_to_look_at_still_counts(
    project_root: Path, tmp_path: Path, flag: str, change: str
) -> None:
    """git status passes an edit under these flags: the archive named the commit, "dirty": false,
    and held other bytes."""
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    notes = repo / "NOTES.txt"
    notes.write_text("as committed\n", encoding="utf-8")
    _git(repo, "add", "NOTES.txt")
    _git(repo, "commit", "-qm", "notes")
    _git(repo, "update-index", flag, "NOTES.txt")
    if change == "edit":
        notes.write_text("not what the commit holds\n", encoding="utf-8")
    else:
        notes.unlink()

    completed = _package(repo)

    assert completed.returncode != 0
    assert "uncommitted changes" in completed.stderr
    assert f"{'M' if change == 'edit' else 'D'} NOTES.txt (marked --assume-unchanged" in (
        completed.stderr
    )
    assert not (repo / "dist").exists() or not list((repo / "dist").glob("*.zip"))


def test_an_archive_of_another_commit_is_left_alone(project_root: Path, tmp_path: Path) -> None:
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    assert _package(repo).returncode == 0
    earlier = _archive(repo)
    data = earlier.read_bytes()
    (repo / "NOTES.txt").write_text("a later commit\n", encoding="utf-8")
    _git(repo, "add", "NOTES.txt")
    _git(repo, "commit", "-qm", "later")

    assert _package(repo).returncode == 0

    assert _archive(repo) != earlier
    assert earlier.read_bytes() == data
    assert len(list((repo / "dist").glob("*.zip"))) == 2


@pytest.mark.parametrize(
    ("refusal", "flags"),
    [("bundle", ()), ("dirty tree", ()), ("dirty tree", ("--without-wheels",))],
)
def test_a_refused_build_keeps_the_earlier_archive_and_says_whose_it_is(
    project_root: Path, tmp_path: Path, refusal: str, flags: tuple[str, ...]
) -> None:
    """A refusal deleted a validated archive; keeping it silently let it pass as this run's.

    The earlier archive and its sidecar stay, the run says they are not its
    own and from which commit they came, and what a failed run left goes.
    """
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    assert _package(repo).returncode == 0
    assert _package(repo, "--without-wheels").returncode == 0
    earlier_commit = _head(repo)
    dist = repo / "dist"
    kept = {path.name: path.read_bytes() for path in dist.iterdir()}
    name = _archive(repo, "-without-wheels" if flags else "").name.removesuffix(".zip")
    (dist / f"{name}.zip.partial").write_bytes(b"a failed run's")
    (dist / f"{name}.extracted-smoke-failure.log").write_text("a failed run's\n", encoding="utf-8")
    if refusal == "bundle":
        (repo / "vendor" / "wheels" / WHEEL).unlink()
    else:
        (repo / "VERSION").write_text("0.0.0\n\n", encoding="utf-8")

    completed = _package(repo, *flags)

    assert completed.returncode != 0
    assert ("missing" if refusal == "bundle" else "uncommitted changes") in completed.stderr
    assert {path.name: path.read_bytes() for path in dist.iterdir()} == kept
    assert f"note: {dist / name}.zip and its sidecar are from an earlier build" in completed.stderr
    assert f"from commit {earlier_commit}" in completed.stderr


@pytest.mark.parametrize(
    "step",
    ["sidecar.write", "disk full", "archive.unlink", "sidecar.replace", "archive.replace",
     "printed"],
)
def test_an_interrupted_rename_never_leaves_a_mismatched_pair(
    project_root: Path, tmp_path: Path, step: str
) -> None:
    """Interrupted, or out of space, at each step of putting the new archive in place.

    Whatever is left, an archive is never beside a sidecar that does not
    check it. Up to and including the sidecar write the earlier pair is kept
    whole, and the run says it is the earlier one; the next build clears what
    the failed one left. The earlier pair is under this commit's name:
    another commit's archive has another.
    """
    import shutil
    import subprocess

    if shutil.which("sha256sum") is None:
        pytest.skip("sha256sum is not installed")
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    source = repo / "scripts" / "build_release.py"
    text = source.read_text(encoding="utf-8")
    write = '    sidecar_partial.write_text(f"{zip_hash}  {archive.name}\\n", encoding="utf-8")\n'
    # Only when asked: the last build here has to finish.
    guard = "    if os.environ.get('ORBITWIN_TEST_INTERRUPT'):\n"
    stop = guard + "        raise KeyboardInterrupt\n"
    # A real ENOSPC leaves the partial sidecar behind, empty.
    full = guard + (
        "        sidecar_partial.write_bytes(b'')\n        raise OSError(28, 'No space left on device')\n"
    )
    target, injected = {
        "sidecar.write": (write, stop),
        "disk full": (write, full),
        "archive.unlink": ("    archive.unlink(missing_ok=True)\n    # From here", stop),
        "sidecar.replace": ("    os.replace(sidecar_partial, sidecar)\n", stop),
        "archive.replace": ("    os.replace(partial, archive)\n", stop),
        # After the rename: the archive is this run's, and the note must not call it earlier.
        "printed": ('    print(f"archive={archive}")\n', stop),
    }[step]
    assert text.count(target) == 1, f"the step this interrupts is not one line any more: {step}"
    source.write_text(text.replace(target, injected + target, 1), encoding="utf-8")
    _git(repo, "commit", "-qam", "interrupt")
    # An earlier pair under the name this build writes, unlike what it writes.
    dist = repo / "dist"
    dist.mkdir()
    name = _archive(repo).name
    (dist / name).write_bytes(b"an earlier build")
    (dist / f"{name}.sha256").write_text(f"{_digest(b'an earlier build')}  {name}\n",
                                         encoding="utf-8")
    earlier = {path.name: path.read_bytes() for path in dist.iterdir()}

    completed = _package(repo, ORBITWIN_TEST_INTERRUPT="1")

    assert completed.returncode != 0
    if (dist / name).exists():
        checked = subprocess.run(
            ["sha256sum", "--check", "--strict", f"{name}.sha256"],
            cwd=dist, capture_output=True, text=True, check=False,
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    if step in {"sidecar.write", "disk full", "archive.unlink"}:
        for kept, data in earlier.items():
            assert (dist / kept).read_bytes() == data, f"the earlier {kept} was not kept"
        assert "are from an earlier build" in completed.stderr
    else:
        assert "are from an earlier build" not in completed.stderr

    # A refused build clears what the failed one left, the partial sidecar included.
    (repo / "VERSION").write_text("0.0.0\n\n", encoding="utf-8")
    assert _package(repo).returncode != 0
    assert not list(dist.glob("*.partial")), sorted(path.name for path in dist.iterdir())
    _git(repo, "checkout", "--", "VERSION")
    assert _package(repo).returncode == 0
    assert sorted(path.name for path in dist.iterdir()) == [name, f"{name}.sha256"]


def test_the_archive_does_not_depend_on_the_umask(project_root: Path, tmp_path: Path) -> None:
    """Git records 0644 and 0755; a checkout under umask 002 gave 0664 and other bytes."""
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    archive = _archive(repo)
    (repo / "scripts" / "build_release.py").chmod(0o644)
    assert _package(repo).returncode == 0
    first = archive.read_bytes()
    # Group write, which git does not record and a umask of 002 leaves.
    for path in (repo / "VERSION", repo / "scripts" / "build_release.py", repo / "SHA256SUMS.txt"):
        path.chmod(path.stat().st_mode | 0o020)

    assert _package(repo).returncode == 0

    assert archive.read_bytes() == first
    with zipfile.ZipFile(archive) as zf:
        modes = {(info.external_attr >> 16) & 0o777 for info in zf.infolist()}
    assert modes <= {0o644, 0o755}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("differs", f"{WHEEL}: differs from the manifest"),
        ("missing", f"{WHEEL}: missing"),
        ("link", f"{WHEEL}: a symbolic link"),
    ],
)
def test_a_listed_wheel_that_is_not_the_fetched_one_is_refused(
    project_root: Path, tmp_path: Path, change: str, message: str
) -> None:
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    wheel = repo / "vendor" / "wheels" / WHEEL
    if change == "differs":
        wheel.write_bytes(b"something else")
    elif change == "missing":
        wheel.unlink()
    else:
        target = repo / "vendor" / "wheels" / "real.bin"
        target.write_bytes(FETCHED)
        wheel.unlink()
        wheel.symlink_to(target.name)

    completed = _package(repo)

    assert completed.returncode != 0
    assert message in completed.stderr
    assert "Put back the wheels" in completed.stderr
    assert not list((repo / "dist").glob("*.zip")), "no archive"


FORMAT = "is not `<sha256>  ./<name>.whl`"


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("", "lists no wheels"),
        (None, "is missing"),
        (f"{_digest(FETCHED)}  ./{WHEEL} \n", FORMAT),
        (f"{_digest(FETCHED)}   ./{WHEEL}\n", FORMAT),
        (f"{_digest(FETCHED)}  *./{WHEEL}\n", FORMAT),
        (f"{_digest(FETCHED)}  ./{WHEEL}\r\n", FORMAT),
        (f"{_digest(FETCHED).upper()}  ./{WHEEL}\n", FORMAT),
        (f"\n{_digest(FETCHED)}  ./{WHEEL}\n", FORMAT),
        (f"{_digest(FETCHED)}  ./{WHEEL}\n{_digest(FETCHED)}  ./{WHEEL}\n", "listed twice"),
        (f"{_digest(FETCHED)}  ../{WHEEL}\n", FORMAT),
        (f"{_digest(FETCHED)}  ./sub/{WHEEL}\n", FORMAT),
        (f"{_digest(FETCHED)}  ./notes.txt\n", FORMAT),
        (b"\xff\xfe not text", "is not UTF-8 text"),
    ],
)
def test_a_manifest_the_fetch_did_not_write_is_refused(
    project_root: Path, tmp_path: Path, manifest: str | bytes | None, message: str
) -> None:
    """Only the fetch script's form, a strict subset of what `sha256sum -c` takes."""
    repo = _checkout_with_a_bundle(
        project_root, tmp_path, manifest=manifest if isinstance(manifest, str) else ""
    )
    if isinstance(manifest, bytes):
        (repo / "vendor" / "wheels" / "SHA256SUMS.txt").write_bytes(manifest)
        _git(repo, "commit", "-qam", "binary manifest")
    if manifest is None:
        _git(repo, "rm", "-q", "vendor/wheels/SHA256SUMS.txt")
        _git(repo, "commit", "-qm", "no manifest")

    completed = _package(repo)

    assert completed.returncode != 0
    assert message in completed.stderr


def test_without_wheels_builds_an_archive_named_for_what_it_lacks(
    project_root: Path, tmp_path: Path
) -> None:
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    (repo / "vendor" / "wheels" / WHEEL).unlink()

    completed = _package(repo, "--without-wheels")

    assert completed.returncode == 0, completed.stderr
    assert "needs the network" in completed.stdout
    assert not _archive(repo).exists()
    with zipfile.ZipFile(_archive(repo, "-without-wheels")) as zf:
        assert not [name for name in zf.namelist() if name.endswith(".whl")]


def test_a_build_that_fails_after_zipping_leaves_no_archive(
    project_root: Path, tmp_path: Path
) -> None:
    """It is checked as a .partial, so nothing it made lands under the release's name."""
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    source = repo / "scripts" / "build_release.py"
    # Fail the first check after create_zip.
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "def verify_manifest(extracted_root: Path) -> None:\n",
            "def verify_manifest(extracted_root: Path) -> None:\n"
            "    raise RuntimeError('injected')\n",
        ),
        encoding="utf-8",
    )
    _git(repo, "commit", "-qam", "inject")
    archive = _archive(repo)
    archive.parent.mkdir()
    sidecar = archive.with_name(archive.name + ".sha256")
    sidecar.write_text("stale\n", encoding="utf-8")

    completed = _package(repo)

    assert completed.returncode != 0
    assert "injected" in completed.stderr
    assert not list((repo / "dist").glob("*.zip")), "no archive"
    # Not written by this run; an earlier one's is left alone.
    assert sidecar.read_text(encoding="utf-8") == "stale\n"
    # What was built is kept under a name nobody would upload, until the next build.
    assert archive.with_name(archive.name + ".partial").is_file()
    assert _package(repo).returncode != 0
    assert len(list((repo / "dist").glob("*.partial"))) == 1


def test_a_wheel_swapped_after_hashing_stops_the_build(project_root: Path, tmp_path: Path) -> None:
    """main() checks the extracted wheels, not only the ones it hashed.

    Swapped while zipping, after the root manifest was written, the root
    manifest's check after extraction is the one that sees it.
    """
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    source = repo / "scripts" / "build_release.py"
    # The wheel's bytes change between wheel_bundle() and the archive.
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "            zf.writestr(info, path.read_bytes())\n",
            "            zf.writestr(info, b'swapped' if path.suffix == '.whl' else path.read_bytes())\n",
        ),
        encoding="utf-8",
    )
    _git(repo, "commit", "-qam", "swap")

    completed = _package(repo)

    assert completed.returncode != 0
    assert f"Manifest mismatch for vendor/wheels/{WHEEL}" in completed.stderr
    assert not list((repo / "dist").glob("*.zip")), "no archive"


def test_a_wheel_swapped_before_the_root_manifest_is_caught_by_step_zeros_check(
    project_root: Path, tmp_path: Path
) -> None:
    """Swapped between wheel_bundle() and build_manifest(), as a refetch in another terminal could.

    The root manifest then records the new bytes and its own check passes;
    only verify_wheel_bundle(), step 0's `sha256sum --check --strict`, sees
    that the wheel is not the one vendor/wheels/SHA256SUMS.txt lists.
    """
    import shutil

    if shutil.which("sha256sum") is None:
        pytest.skip("sha256sum is not installed")
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    source = repo / "scripts" / "build_release.py"
    text = source.read_text(encoding="utf-8")
    target = "    clean(root, archive.parent)\n"
    assert text.count(target) == 1
    swap = f"    (root / 'vendor' / 'wheels' / {WHEEL!r}).write_bytes(b'swapped')\n"
    source.write_text(text.replace(target, swap + target, 1), encoding="utf-8")
    _git(repo, "commit", "-qam", "swap before the manifest")

    completed = _package(repo)

    assert completed.returncode != 0
    assert "sha256sum --check --strict" in completed.stderr
    assert not list((repo / "dist").glob("*.zip")), "no archive"


def _extracted_bundle(tmp_path: Path) -> Path:
    extracted = tmp_path / "extracted"
    wheels = extracted / "vendor" / "wheels"
    wheels.mkdir(parents=True)
    (wheels / WHEEL).write_bytes(FETCHED)
    (wheels / "SHA256SUMS.txt").write_text(f"{_digest(FETCHED)}  ./{WHEEL}\n", encoding="utf-8")
    return extracted


@pytest.mark.parametrize("damage", ["changed", "removed"])
def test_the_extracted_bundle_is_checked_with_sha256sum(
    project_root: Path, tmp_path: Path, damage: str
) -> None:
    """A wheel changed or lost between the hash and the archive is caught after extraction."""
    import shutil

    if shutil.which("sha256sum") is None:
        pytest.skip("sha256sum is not installed")
    build_release = _load_build_release(project_root)
    extracted = _extracted_bundle(tmp_path)
    build_release.verify_wheel_bundle(extracted, 1)

    wheel = extracted / "vendor" / "wheels" / WHEEL
    if damage == "changed":
        wheel.write_bytes(b"swapped after hashing")
    else:
        wheel.unlink()
    with pytest.raises(RuntimeError, match="sha256sum --check --strict"):
        build_release.verify_wheel_bundle(extracted, 1)


def test_the_extracted_bundle_must_hold_every_wheel_shipped(
    project_root: Path, tmp_path: Path
) -> None:
    import shutil

    if shutil.which("sha256sum") is None:
        pytest.skip("sha256sum is not installed")
    build_release = _load_build_release(project_root)

    with pytest.raises(RuntimeError, match="holds 1 wheels, not 2"):
        build_release.verify_wheel_bundle(_extracted_bundle(tmp_path), 2)


def test_the_bundle_reads_binary_mode_manifest_lines(project_root: Path, tmp_path: Path) -> None:
    """`sha256sum -b` writes `HASH *NAME`; the fetch script writes `HASH  ./NAME`."""
    build_release = _load_build_release(project_root)
    wheels = tmp_path / "vendor" / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "a-1-py3-none-any.whl").write_bytes(b"a")
    (wheels / "b-1-py3-none-any.whl").write_bytes(b"b")
    (wheels / "SHA256SUMS.txt").write_text(
        f"{_digest(b'a')}  ./a-1-py3-none-any.whl\n{_digest(b'b')} *b-1-py3-none-any.whl\n",
        encoding="utf-8",
    )

    assert [path.name for path in build_release.wheel_bundle(tmp_path)] == [
        "a-1-py3-none-any.whl", "b-1-py3-none-any.whl",
    ]


def test_the_bundle_resolves_offline_on_every_python_it_covers(
    project_root: Path, tmp_path: Path
) -> None:
    """The pins alone do not show the transitive wheels are there; a resolver does.

    It sees only the wheels the manifest lists, the ones the archive ships, and
    the build backend the bootstrap installs next.
    """
    import re
    import shutil
    import subprocess

    wheels = project_root / "vendor" / "wheels"
    if shutil.which("uv") is None or not list(wheels.glob("*.whl")):
        pytest.skip("needs uv and the fetched wheel bundle")
    names = [
        line.split()[-1].lstrip("*").removeprefix("./")
        for line in (wheels / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    ]
    missing = [name for name in names if not (wheels / name).is_file()]
    if missing:
        pytest.skip(
            f"vendor/wheels does not hold the set SHA256SUMS.txt lists ({len(missing)} missing, "
            f"e.g. {missing[0]}); refetch with scripts/fetch_offline_wheels.sh. make package "
            "refuses this bundle too."
        )
    listed = tmp_path / "listed"
    listed.mkdir()
    for name in names:
        (listed / name).symlink_to(wheels / name)
    backend = tmp_path / "backend.in"
    backend.write_text("setuptools\nwheel\n", encoding="utf-8")
    fetch = (project_root / "scripts" / "fetch_offline_wheels.sh").read_text(encoding="utf-8")
    versions = re.search(r'VERSIONS="\$\{WHEEL_PYTHON_VERSIONS:-([0-9. ]+)\}"', fetch)
    assert versions
    for version in versions.group(1).split():
        completed = subprocess.run(
            ["uv", "pip", "compile", "--offline", "--no-index", "--find-links", str(listed),
             "--only-binary", ":all:", "--python-version", version,
             "--python-platform", "x86_64-manylinux_2_28", "--no-header", "--quiet",
             "--cache-dir", str(tmp_path / "cache"),
             str(project_root / "requirements-dev.txt"), str(backend)],
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, f"Python {version}: {completed.stderr}"


def test_the_shipped_manifest_lists_a_wheel_for_every_pin(project_root: Path) -> None:
    """Every pin in requirements-dev.txt, for each interpreter the bundle covers.

    `sha256sum -c` passes on a bundle missing a file it does not list, and the
    committed manifest once lacked types-PyYAML for exactly that reason.
    """
    import re

    manifest = (project_root / "vendor" / "wheels" / "SHA256SUMS.txt").read_text(encoding="utf-8")
    names = [line.split()[-1].removeprefix("*").removeprefix("./") for line in manifest.splitlines()]
    pins: dict[str, str] = {}
    for source in ("requirements.txt", "requirements-dev.txt"):
        for line in (project_root / source).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "==" in line:
                name, _, version = line.partition("==")
                pins[re.sub(r"[-_.]+", "_", name.strip()).lower()] = version.strip()
    assert {"numpy", "scipy", "pyyaml", "pytest", "types_pyyaml"} <= set(pins)

    for interpreter in ("311", "312", "313", "314"):
        for name, version in pins.items():
            prefix = f"{name}-{version}-"
            matching = [
                wheel for wheel in names
                if wheel.lower().startswith(prefix)
                and (f"-cp{interpreter}-" in wheel or "-py3-none-" in wheel or "-abi3-" in wheel)
            ]
            assert matching, f"no {name}=={version} wheel for Python {interpreter[0]}.{interpreter[1:]}"


def _side_commit(repo: Path) -> str:
    """A commit on a branch of its own, ahead of origin/main."""
    _git(repo, "checkout", "-q", "-b", "stale-branch")
    (repo / "notes.txt").write_text("not on main\n", encoding="utf-8")
    _git(repo, "add", "notes.txt")
    _git(repo, "commit", "-qm", "not on main")
    return _head(repo)


def test_a_head_that_is_not_origin_main_is_refused(project_root: Path, tmp_path: Path) -> None:
    """The main checkout sat on a stale branch, and make package archived its HEAD."""
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    head = _side_commit(repo)

    refused = _package(repo, publish=False)
    assert refused.returncode == 1
    assert refused.stderr.startswith("build_release: refused: HEAD is ")
    assert "Traceback" not in refused.stderr
    assert f"HEAD is {head[:12]} on stale-branch; origin/main is " in refused.stderr
    assert "--allow-non-main" in refused.stderr
    assert not list((repo / "dist").glob("*.zip"))

    allowed = _package(repo, "--allow-non-main", publish=False)
    assert allowed.returncode == 0, allowed.stderr
    assert "NOT ORIGIN/MAIN (--allow-non-main): HEAD is" in allowed.stderr
    assert f"provenance=NOT origin/main: branch stale-branch, commit {head}" in allowed.stdout
    with zipfile.ZipFile(_archive(repo)) as zf:
        info = json.loads(zf.read("tasa_orbitwin/BUILD_INFO.json"))
    assert info["commit"] == head
    assert info["provenance"]["branch"] == "stale-branch"
    assert info["provenance"]["matches"] is None
    assert info["provenance"]["allow_non_main"] is True


def test_the_release_tag_is_a_release_too(project_root: Path, tmp_path: Path) -> None:
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    _side_commit(repo)
    _git(repo, "tag", "-a", "v0.0.0", "-m", "release")
    completed = _package(repo, publish=False)
    assert completed.returncode == 0, completed.stderr
    assert "provenance=v0.0.0" in completed.stdout


def test_the_fetch_finds_a_main_that_moved_on(project_root: Path, tmp_path: Path) -> None:
    """The local origin/main can be as stale as the checkout; it is fetched first."""
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", "-b", "main", str(tmp_path / "origin.git"), str(other))
    (other / "later.txt").write_text("merged since\n", encoding="utf-8")
    _git(other, "add", "later.txt")
    _git(other, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "later")
    _git(other, "push", "-q", "origin", "HEAD:refs/heads/main")

    refused = _package(repo, publish=False)
    assert refused.returncode != 0
    assert f"HEAD is {_head(repo)[:12]}" in refused.stderr
    assert f"origin/main is {_head(other)[:12]}" in refused.stderr


def test_without_a_fetch_the_local_ref_is_used_and_its_age_given(
    project_root: Path, tmp_path: Path
) -> None:
    repo = _checkout_with_a_bundle(project_root, tmp_path)
    _git(repo, "fetch", "-q", "origin")
    _git(repo, "remote", "set-url", "origin", str(tmp_path / "unreachable.git"))

    completed = _package(repo, publish=False)
    assert completed.returncode == 0, completed.stderr
    assert ("could not fetch origin main; compared with the local origin/main, last updated "
            "0.0 h ago") in completed.stderr

    _side_commit(repo)
    refused = _package(repo, publish=False)
    assert refused.returncode != 0
    assert "(could not fetch origin main; compared with the local origin/main" in refused.stderr


def test_the_version_line_says_a_build_was_not_main(tmp_path: Path) -> None:
    from orbitwin.cli import describe_build

    info = {"commit": "c" * 40, "committed_utc": "2026-11-06T00:00:00Z", "version": "0.3.0",
            "dirty": False, "provenance": {"branch": "stale-branch", "matches": None,
                                           "allow_non_main": True}}
    path = tmp_path / "BUILD_INFO.json"
    path.write_text(json.dumps(info), encoding="utf-8")
    assert describe_build(path).endswith(
        ", NOT origin/main: built with --allow-non-main on stale-branch")
    info["provenance"] = {"branch": "main", "matches": "origin/main", "allow_non_main": False}
    path.write_text(json.dumps(info), encoding="utf-8")
    assert "NOT origin/main" not in describe_build(path)
