"""What the release archive carries: what the competition machine uses, not the dev tooling."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
from conftest import repo_files, scratch_git_environment

#: Tracked, and of no use on the day: notebooks/, and a path scripts/private_paths.txt names.
NOT_SHIPPED = ("devtools/settings.json", "notebooks/dashboard.ipynb")
#: Under the same names one level down, which are not the dev tooling.
SHIPPED = ("docs/devtools/notes.md", "tests/notebooks/case.txt", "VERSION")


def test_the_archive_leaves_out_the_dev_tooling(project_root: Path, tmp_path: Path) -> None:
    """The archive shipped the dev tooling and notebooks/ to the competition machine."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed; packaging needs a git checkout")
    environment = scratch_git_environment(tmp_path)
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("build_release.py", "make_manifest.py"):
        shutil.copy2(project_root / "scripts" / name, repo / "scripts" / name)
    (repo / "scripts" / "private_paths.txt").write_text("# dev tooling\ndevtools\n",
                                                        encoding="utf-8")
    (repo / ".gitignore").write_text("/dist/\nvendor/wheels/*.whl\n", encoding="utf-8")
    for name in (*NOT_SHIPPED, *SHIPPED):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("0.0.0\n" if name == "VERSION" else name + "\n", encoding="utf-8")
    wheels = repo / "vendor" / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "demo-1.0-py3-none-any.whl").write_bytes(b"wheel")
    (wheels / "SHA256SUMS.txt").write_text(
        f"{hashlib.sha256(b'wheel').hexdigest()}  ./demo-1.0-py3-none-any.whl\n", encoding="utf-8"
    )
    for args in (("init", "-q"), ("config", "user.email", "t@example.com"),
                 ("config", "user.name", "t"), ("add", "."), ("commit", "-qm", "init"),
                 # A release is built from origin/main.
                 ("init", "-q", "--bare", str(tmp_path / "origin.git")),
                 ("remote", "add", "origin", str(tmp_path / "origin.git")),
                 ("push", "-q", "origin", "HEAD:refs/heads/main")):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                       env=environment)

    completed = subprocess.run(
        [sys.executable, "scripts/build_release.py", "--project-root", ".", "--output-dir", "dist",
         "--skip-extracted-test"],
        cwd=repo, capture_output=True, text=True, check=False, env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    [archive] = (repo / "dist").glob("*.zip")
    with zipfile.ZipFile(archive) as zf:
        names = {name.removeprefix("tasa_orbitwin/") for name in zf.namelist()}
        listed = {line.split("  ", 1)[1]
                  for line in zf.read("tasa_orbitwin/SHA256SUMS.txt").decode().splitlines()}
    for name in NOT_SHIPPED:
        assert name not in names
        assert name not in listed
    for name in SHIPPED:
        assert name in names
        assert name in listed


def test_this_checkout_ships_none_of_the_dev_tooling(project_root: Path) -> None:
    if not (project_root / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout")
    spec = importlib.util.spec_from_file_location(
        "orbitwin_make_manifest_contents", project_root / "scripts" / "make_manifest.py"
    )
    assert spec and spec.loader
    make_manifest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_manifest)
    tracked = [path.relative_to(project_root).as_posix()
               for path in repo_files(project_root, "*", tracked_only=True)]

    shipped = {path.relative_to(project_root).parts[0]
               for path in make_manifest.tracked_files(project_root)}

    assert {name.split("/")[0] for name in tracked} >= make_manifest.not_shipped(project_root), (
        "each excluded directory is tracked, or the exclusion is dead"
    )
    assert not shipped & make_manifest.not_shipped(project_root)
    # What step 0 runs and the suite reads stays.
    assert {"src", "scripts", "tests", "docs", "vendor", ".github"} <= shipped
    # make validate's evidence stays on the machine that ran it.
    assert "validation" not in shipped


def _make_manifest(project_root: Path) -> object:
    spec = importlib.util.spec_from_file_location(
        "orbitwin_make_manifest_private", project_root / "scripts" / "make_manifest.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_private_paths_are_the_commits_not_the_working_copys(
    project_root: Path, tmp_path: Path
) -> None:
    """Read beside the script, an untracked list changed what a clean commit packaged."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    make_manifest: Any = _make_manifest(project_root)
    environment = scratch_git_environment(tmp_path)
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    for args in (["init", "-q"], ["add", "README.md"], ["commit", "-qm", "c"]):
        subprocess.run([*git, *args], check=True, capture_output=True, env=environment)
    (repo / "scripts" / "private_paths.txt").write_text("devtools\n", encoding="utf-8")

    assert make_manifest.not_shipped(repo) == make_manifest.NOT_SHIPPED  # untracked

    for args in (["add", "scripts/private_paths.txt"], ["commit", "-qm", "c"]):
        subprocess.run([*git, *args], check=True, capture_output=True, env=environment)
    make_manifest.not_shipped.cache_clear()
    assert make_manifest.not_shipped(repo) == make_manifest.NOT_SHIPPED | {"devtools"}

    # An unpacked archive has no git: its own file is the list, also inside another
    # checkout, whose HEAD is not the archive's.
    unpacked = tmp_path / "unpacked"
    shutil.copytree(repo, unpacked, ignore=shutil.ignore_patterns(".git"))
    assert make_manifest.not_shipped(unpacked) == make_manifest.NOT_SHIPPED | {"devtools"}
    inside = repo / "output" / "unpacked"
    shutil.copytree(unpacked, inside)
    (inside / "scripts" / "private_paths.txt").write_text("other\n", encoding="utf-8")
    assert make_manifest.not_shipped(inside) == make_manifest.NOT_SHIPPED | {"other"}
