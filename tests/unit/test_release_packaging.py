from __future__ import annotations

import importlib.util
import os
import zipfile
from pathlib import Path

import pytest
from conftest import own_git, scratch_git_environment


def _load_script(project_root: Path, name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, project_root / "scripts" / name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_build_release(project_root: Path):
    return _load_script(project_root, "build_release.py", "orbitwin_build_release")


@pytest.fixture
def make_manifest(project_root: Path):
    return _load_script(project_root, "make_manifest.py", "orbitwin_make_manifest")


@pytest.fixture
def git_checkout(project_root: Path, make_manifest) -> Path:
    """The project root, or skip when it is not a git checkout.

    These assertions describe packaging behaviour, which only has meaning where
    git can answer. The release archive itself is not a checkout, and
    build_release.py runs this suite against the extracted tree, so without the
    skip they fail there by construction rather than by defect.
    """
    import shutil

    if not (project_root / ".git").exists():
        pytest.skip("not a git checkout; packaging provenance cannot be checked here")
    if shutil.which("git") is None:
        pytest.skip("git is not installed; packaging provenance cannot be checked here")
    return project_root


@pytest.fixture
def archive_with_executable(tmp_path: Path, project_root: Path):
    build_release = _load_build_release(project_root)
    source = tmp_path / "src"
    source.mkdir()
    script = source / "console.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    script.chmod(0o755)
    (source / "plain.txt").write_text("data\n", encoding="utf-8")

    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(source.rglob("*")):
            info = zipfile.ZipInfo(path.name, date_time=build_release.FIXED_TIMESTAMP)
            info.external_attr = (path.stat().st_mode & 0o777) << 16
            zf.writestr(info, path.read_bytes())
    return build_release, archive


def test_plain_extractall_loses_the_executable_bit(
    archive_with_executable, tmp_path: Path
) -> None:
    # Documents why extract_all_with_modes exists. zipfile.extractall ignores
    # external_attr by design, so the release archive stores 0755 and hands back
    # 0644 -- which is how the GMAT orchestration tests came to be disabled.
    _, archive = archive_with_executable
    destination = tmp_path / "naive"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)
    assert not os.access(destination / "console.py", os.X_OK)


def test_extract_all_with_modes_preserves_the_executable_bit(
    archive_with_executable, tmp_path: Path
) -> None:
    build_release, archive = archive_with_executable
    destination = tmp_path / "restored"
    with zipfile.ZipFile(archive) as zf:
        build_release.extract_all_with_modes(zf, destination)
    assert os.access(destination / "console.py", os.X_OK)
    assert (destination / "console.py").stat().st_mode & 0o777 == 0o755
    assert not os.access(destination / "plain.txt", os.X_OK)


def test_manifest_enumerates_git_not_the_working_tree(make_manifest, git_checkout: Path) -> None:
    # Walking the filesystem packages whatever happens to be lying in the
    # working directory. A scratch output a fuzz run left in the checkout was
    # picked up exactly this way and would have shipped in the archive and
    # been recorded in SHA256SUMS.txt as part of the project.
    stray = git_checkout / "docs" / "test_stray_artifact.json"
    stray.write_text("{}\n", encoding="utf-8")
    try:
        assert make_manifest.included(stray, git_checkout) is True
        assert stray not in make_manifest.tracked_files(git_checkout)
    finally:
        stray.unlink()


def test_manifest_handles_non_ascii_tracked_paths(make_manifest, tmp_path: Path) -> None:
    """Plain `git ls-files` renders such a name as a C-style quoted escape, which would
    resolve to a path that does not exist. -z is load-bearing."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is not installed; packaging needs a git checkout")
    repo = tmp_path / "repo"
    names = ("docs/規則 2026.pdf", "naïve/notes.txt", "README.md")
    for name in names:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(b"x\n")
    for args in (("init", "-q"), ("add", "-A"),
                 ("-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "x")):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                       env=scratch_git_environment(tmp_path))
    # What the -z path is for: without it git quotes both non-ASCII names.
    quoted = subprocess.run(["git", "-C", str(repo), "ls-files"], check=True,
                            capture_output=True, env=scratch_git_environment(tmp_path)).stdout
    assert quoted.count(b'"') == 4

    tracked = make_manifest.tracked_files(repo)
    assert tracked == sorted(repo / name for name in names)
    assert all(path.is_file() for path in tracked)


def test_manifest_never_lists_itself(make_manifest, tmp_path: Path) -> None:
    """Only the root SHA256SUMS.txt is the manifest being written."""
    root = tmp_path / "checkout"
    for relative in ("SHA256SUMS.txt", "vendor/wheels/SHA256SUMS.txt", "docs/SHA256SUMS.txt"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("x\n", encoding="utf-8")

    assert make_manifest.included(root / "SHA256SUMS.txt", root) is False
    assert make_manifest.included(root / "vendor/wheels/SHA256SUMS.txt", root) is True
    assert make_manifest.included(root / "docs/SHA256SUMS.txt", root) is True


def test_manifest_lists_the_wheel_bundles_own_manifest(make_manifest, git_checkout: Path) -> None:
    """Excluded by name, it never reached the archive, and step 0 checks against it."""
    relative = {path.relative_to(git_checkout) for path in make_manifest.tracked_files(git_checkout)}
    assert Path("vendor/wheels/SHA256SUMS.txt") in relative


def test_release_refuses_an_uncommitted_tree(
    make_manifest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil
    import subprocess

    # RUNBOOK step 0 runs this suite on the organizer's machine, which need not have git.
    if shutil.which("git") is None:
        pytest.skip("git is not installed; packaging needs a git checkout")
    # assert_tree_matches_git runs git in this process's environment.
    own_git(monkeypatch, tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    def run(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    run("add", "a.txt")
    run("commit", "-qm", "init")

    make_manifest.assert_tree_matches_git(repo)  # clean tree is accepted

    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        make_manifest.assert_tree_matches_git(repo)

    # An untracked file is not a reason to refuse: it is no longer packaged.
    run("checkout", "--", "a.txt")
    (repo / "scratch.log").write_text("noise\n", encoding="utf-8")
    make_manifest.assert_tree_matches_git(repo)


def test_both_module_entry_points_refuse_an_empty_command_line(project_root: Path) -> None:
    """`python -m orbitwin.cli` used to import, run nothing and exit 0.

    A silent success is the one outcome ruled out (a failure is never reported
    as success), and it is worse here than elsewhere: the operator sees rc=0, no
    output and no artifacts, on a machine they get one afternoon with.
    """
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    for module in ("orbitwin", "orbitwin.cli"):
        completed = subprocess.run(
            [sys.executable, "-m", module],
            cwd=project_root,
            env=env,
            capture_output=True,
        )
        assert completed.returncode != 0, f"{module} exited 0 with no command"


def test_a_relative_output_dir_is_relative_to_the_project_root(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`make package` has to land in the same place from any directory.

    The default was an absolute path that exists inside one container image
    and nowhere else, so the target failed with a bare ENOENT everywhere else --
    including the competition laptop. Resolving against the caller's working
    directory instead would trade that for a quieter version of the same
    problem: the archive would exist, somewhere nobody looked.
    """
    build_release = _load_build_release(project_root)
    root = tmp_path / "checkout"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolved = build_release.resolve_output_dir(root, build_release.DEFAULT_OUTPUT_DIR)

    assert resolved == (root / "dist").resolve()
    assert resolved.is_dir(), "the directory has to be created, not merely named"


def test_an_absolute_output_dir_is_taken_as_given(project_root: Path, tmp_path: Path) -> None:
    build_release = _load_build_release(project_root)
    destination = tmp_path / "somewhere" / "deep"

    resolved = build_release.resolve_output_dir(tmp_path / "checkout", str(destination))

    assert resolved == destination.resolve()
    assert resolved.is_dir()


def test_resolving_an_output_dir_twice_is_not_an_error(project_root: Path, tmp_path: Path) -> None:
    """Two `make package` runs in a row, which is the normal case."""
    build_release = _load_build_release(project_root)
    root = tmp_path / "checkout"
    root.mkdir()

    first = build_release.resolve_output_dir(root, "dist")
    (first / "tasa_orbitwin-0.0.0.zip").write_bytes(b"stale")
    second = build_release.resolve_output_dir(root, "dist")

    assert first == second


def test_the_default_output_dir_is_gitignored(project_root: Path) -> None:
    """An archive written into a tracked directory shows up as release noise."""
    build_release = _load_build_release(project_root)
    ignored = (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert f"/{build_release.DEFAULT_OUTPUT_DIR}/" in ignored


def test_clean_leaves_the_runs_and_the_releases_alone(project_root: Path, tmp_path: Path) -> None:
    """`make package` deleted output/, build/ and dist/: every run, every release."""
    build_release = _load_build_release(project_root)
    root = tmp_path / "checkout"
    kept = [root / "output" / "R1" / "summary.json", root / "dist" / "tasa_orbitwin-0.2.0.zip",
            root / "build" / "lib" / "x.txt"]
    for path in kept:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    build_release.clean(root, build_release.resolve_output_dir(root, "output/offline/dist"))

    assert all(path.read_text(encoding="utf-8") == "{}\n" for path in kept)


@pytest.mark.parametrize(
    "relative",
    ["output/offline/dist", "dist/offline", "build/release", ".ruff_cache/x", ".ruff_cache/a/b"],
)
def test_clean_spares_a_destination_and_removes_every_cache(
    project_root: Path, tmp_path: Path, relative: str
) -> None:
    """`PACKAGE_DIR=output/offline/dist` and `dist/offline` failed with FileNotFoundError.

    Spared means left in place, not removed and made again: an earlier
    archive in it and a sibling both stay. Every cache that does not hold
    the destination goes.
    """
    build_release = _load_build_release(project_root)
    root = tmp_path / "checkout"
    for name in (*build_release.CLEANED, "build", "dist"):
        (root / name / "inner").mkdir(parents=True)
    output_dir = build_release.resolve_output_dir(root, relative)
    earlier = output_dir / "earlier.zip"
    earlier.write_bytes(b"old")
    sibling = output_dir.parent / "sibling.txt"
    sibling.write_text("kept\n", encoding="utf-8")

    build_release.clean(root, output_dir)

    assert earlier.read_bytes() == b"old"
    assert sibling.read_text(encoding="utf-8") == "kept\n"
    assert set(build_release.CLEANED) == {".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov"}
    holder = relative.split("/")[0]
    for name in build_release.CLEANED:
        if name != holder:
            assert not (root / name).exists(), f"{name}/ does not hold the archive and still goes"
    assert (root / "build" / "inner").is_dir() and (root / "dist" / "inner").is_dir()


@pytest.mark.parametrize("relative", [".ruff_cache2", ".ruff_cachex/offline", "."])
def test_a_name_that_only_starts_like_a_cleaned_directory_spares_nothing(
    project_root: Path, tmp_path: Path, relative: str
) -> None:
    """Path containment, not a string prefix: .ruff_cache2 does not hold .ruff_cache."""
    build_release = _load_build_release(project_root)
    root = tmp_path / "checkout"
    (root / ".ruff_cache").mkdir(parents=True)
    (root / ".mypy_cache").mkdir()

    build_release.clean(root, build_release.resolve_output_dir(root, relative))

    assert not (root / ".ruff_cache").exists()
    assert not (root / ".mypy_cache").exists()


def test_clean_removes_bytecode_but_not_from_runs_or_environments(
    project_root: Path, tmp_path: Path
) -> None:
    build_release = _load_build_release(project_root)
    root = tmp_path / "checkout"
    swept = [root / "src" / "pkg" / "__pycache__" / "m.cpython-313.pyc", root / "scripts" / "x.pyc"]
    kept = [root / "output" / "run" / "__pycache__" / "m.pyc", root / ".venv" / "lib" / "y.pyc"]
    for path in (*swept, *kept):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    build_release.clean(root, build_release.resolve_output_dir(root, "dist"))

    assert not any(path.exists() for path in swept)
    assert not (root / "src" / "pkg" / "__pycache__").exists()
    assert all(path.exists() for path in kept)


def test_a_tracked_symbolic_link_is_refused(make_manifest, tmp_path: Path) -> None:
    """The archive would hold what it points at: a run, or a file outside the checkout."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is not installed; packaging needs a git checkout")
    repo = tmp_path / "repo"
    (repo / "output" / "R1").mkdir(parents=True)
    (repo / "output" / "R1" / "summary.json").write_text("{}\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "latest.json").symlink_to("../output/R1/summary.json")
    for args in (("init", "-q"), ("add", "docs/latest.json"),
                 ("-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "x")):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                       env=scratch_git_environment(tmp_path))

    with pytest.raises(RuntimeError, match=r"docs/latest\.json"):
        make_manifest.tracked_files(repo)


@pytest.mark.parametrize("elsewhere", [False, True], ids=["from-root", "from-elsewhere"])
# .pytest_cache/release: the one kind of destination clean() could still remove,
# so main() has to hand it over resolved against the project root.
@pytest.mark.parametrize(
    "destination", ["dist", "dist/offline", "output/offline/dist", ".pytest_cache/release"]
)
def test_package_writes_into_output(
    project_root: Path, tmp_path: Path, destination: str, elsewhere: bool
) -> None:
    """main() end to end, with runs under output/ beside the destination.

    Run from the root and from another directory with --project-root, so a
    destination resolved against the working directory would land elsewhere.

    One run file is tracked with `git add -f`, so the archive check exercises
    make_manifest.included(), the only thing keeping output/ out now.
    """
    import hashlib
    import shutil
    import subprocess
    import sys

    if shutil.which("git") is None:
        pytest.skip("git is not installed; packaging needs a git checkout")
    environment = scratch_git_environment(tmp_path)
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("build_release.py", "make_manifest.py"):
        (repo / "scripts" / name).write_bytes((project_root / "scripts" / name).read_bytes())
    (repo / "VERSION").write_text("0.0.0\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        "/output/\n/dist/\n/.pytest_cache/\nvendor/wheels/*.whl\n", encoding="utf-8"
    )
    # A one-wheel offline bundle, so this builds whether or not main() ships one.
    wheels = repo / "vendor" / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "demo-1.0-py3-none-any.whl").write_bytes(b"wheel")
    (wheels / "SHA256SUMS.txt").write_text(
        f"{hashlib.sha256(b'wheel').hexdigest()}  ./demo-1.0-py3-none-any.whl\n",
        encoding="utf-8",
    )
    tracked = repo / "output" / "R0" / "forced.json"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("{}\n", encoding="utf-8")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, env=environment
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("add", ".")
    git("add", "-f", "output/R0/forced.json")
    git("commit", "-qm", "init")
    # A release is built from origin/main.
    git("init", "-q", "--bare", str(tmp_path / "origin.git"))
    git("remote", "add", "origin", str(tmp_path / "origin.git"))
    git("push", "-q", "origin", "HEAD:refs/heads/main")
    run = repo / "output" / "R1" / "summary.json"
    run.parent.mkdir(parents=True)
    run.write_text("{}\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(repo / "scripts" / "build_release.py"),
         "--project-root", str(repo) if elsewhere else ".",
         "--output-dir", destination, "--skip-extracted-test"],
        cwd=tmp_path if elsewhere else repo, capture_output=True, text=True, check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    [archive] = (repo / destination).glob("tasa_orbitwin-0.0.0-*.zip")
    assert not (tmp_path / destination).exists(), "resolved against the project root"
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert "tasa_orbitwin/VERSION" in names
    assert not [name for name in names if name.startswith("tasa_orbitwin/output/")]
    assert run.read_text(encoding="utf-8") == "{}\n"
    assert tracked.read_text(encoding="utf-8") == "{}\n"


def test_clean_spares_the_directory_the_archive_goes_into(
    project_root: Path, tmp_path: Path
) -> None:
    """main() creates the destination, then cleans, then opens the archive in it.

    `dist` was on clean's list and is the default destination, so the clean
    deleted what the first step made and `create_zip` failed with
    FileNotFoundError, the moment the default moved to dist/.
    """
    build_release = _load_build_release(project_root)
    root = tmp_path / "checkout"
    (root / "build").mkdir(parents=True)
    (root / ".ruff_cache").mkdir()

    output_dir = build_release.resolve_output_dir(root, "dist")
    build_release.clean(root, output_dir)

    assert output_dir.is_dir(), "the archive's directory must survive the clean"
    assert (root / "build").is_dir(), "build/ is not a cache"
    assert not (root / ".ruff_cache").exists()


def test_clean_keeps_earlier_releases_when_the_archive_goes_elsewhere(
    project_root: Path, tmp_path: Path
) -> None:
    """dist/ holds earlier releases, and building into output/ deleted them."""
    build_release = _load_build_release(project_root)
    root = tmp_path / "checkout"
    (root / "dist").mkdir(parents=True)
    (root / "dist" / "tasa_orbitwin-0.2.0.zip").write_bytes(b"earlier")

    output_dir = build_release.resolve_output_dir(root, str(tmp_path / "elsewhere"))
    build_release.clean(root, output_dir)

    assert (root / "dist" / "tasa_orbitwin-0.2.0.zip").read_bytes() == b"earlier"
    assert output_dir.is_dir()
