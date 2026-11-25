"""scripts/check_release_tree.py: an unpacked release holds what its manifest lists, and no more."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SHIPPED = ("VERSION", "src/orbitwin/cli.py", "tests/unit/test_x.py", "input/original/README.md",
           "docs/規則.pdf")
#: What bootstrap_offline.sh, the suite and steps 1 to 7 leave in the tree.
MADE_HERE = (".venv/bin/python", ".venv/bundle-requirements.txt", "run/day.yaml",
             "output/x.json", "input/original/official.script",
             "src/orbitwin/__pycache__/cli.cpython-311.pyc", "src/orbitwin.egg-info/PKG-INFO",
             ".pytest_cache/v/cache/lastfailed", "tests/unit/__pycache__/x.pyc", "GmatLog.txt",
             ".coverage")


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "tasa_orbitwin"
    for name in SHIPPED:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(name, encoding="utf-8")
    (root / "SHA256SUMS.txt").write_text(
        "".join(f"{'0' * 64}  {name}\n" for name in SHIPPED), encoding="utf-8"
    )
    return root


def _check(project_root: Path, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(project_root / "scripts" / "check_release_tree.py"), str(root)],
        capture_output=True, text=True, check=False,
    )


def test_what_step_zero_and_the_day_make_is_not_refused(
    project_root: Path, tmp_path: Path
) -> None:
    root = _tree(tmp_path)
    for name in MADE_HERE:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x", encoding="utf-8")

    completed = _check(project_root, root)

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "extra",
    ["src/orbitwin/stale.py", "tests/unit/conftest.py", "src/orbitwin/x.pyc.orig", "input/notes.txt",
     "scripts/__pycache__.py", "link"],
)
def test_a_file_the_archive_did_not_bring_is_refused(
    project_root: Path, tmp_path: Path, extra: str
) -> None:
    """Unzipped over an older tree, step 0 passed with the older tree's conftest.py in place."""
    root = _tree(tmp_path)
    if extra == "link":
        (tmp_path / "elsewhere").mkdir()
        (root / "link").symlink_to(tmp_path / "elsewhere")
    else:
        (root / extra).parent.mkdir(parents=True, exist_ok=True)
        (root / extra).write_text("from an older tree\n", encoding="utf-8")

    completed = _check(project_root, root)

    assert completed.returncode == 1
    assert f"\n  {extra}\n" in completed.stderr
    assert "new, empty directory" in completed.stderr


def test_a_tree_with_no_manifest_is_refused(project_root: Path, tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "SHA256SUMS.txt").unlink()

    completed = _check(project_root, root)

    assert completed.returncode == 1
    assert "not an unpacked release archive" in completed.stderr
