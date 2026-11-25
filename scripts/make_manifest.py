#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import hashlib
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "htmlcov",
    "build",
    "dist",
}
EXCLUDED_NAMES = {".coverage"}
#: Tracked, but of no use on the competition machine, so left out of the
#: archive: notebooks/ needs Jupyter, which the offline bundle does not carry.
#: No test reads it. not_shipped() adds the private paths.
NOT_SHIPPED = frozenset({"notebooks"})
#: Development tooling only some repositories track, one top-level path a line.
PRIVATE_PATHS = Path("scripts/private_paths.txt")


@functools.cache
def not_shipped(root: Path) -> frozenset[str]:
    """NOT_SHIPPED and the private paths of the tree at root.

    In a checkout they come from its HEAD, as the archive does, so an untracked
    list changes nothing; a tree without git (an unpacked archive) has its own file.
    """
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=False)
    except FileNotFoundError:
        top = None
    if top is not None and top.returncode == 0 and (
            os.path.realpath(top.stdout.strip()) == os.path.realpath(root)):
        shown = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{PRIVATE_PATHS.as_posix()}"],
                               capture_output=True, check=False)
        text = shown.stdout.decode("utf-8") if shown.returncode == 0 else ""
    else:
        try:
            text = (root / PRIVATE_PATHS).read_text(encoding="utf-8")
        except FileNotFoundError:
            text = ""
    lines = (line.split("#", 1)[0].strip() for line in text.splitlines())
    return NOT_SHIPPED | {line for line in lines if line}
#: The manifest being written. Only the root one: vendor/wheels/SHA256SUMS.txt
#: is the offline bundle's, and excluding it by name kept it out of the release,
#: so RUNBOOK step 0's `sha256sum -c` had nothing to check against.
MANIFEST = Path("SHA256SUMS.txt")


def included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
        return False
    if relative == MANIFEST or path.name in EXCLUDED_NAMES or path.suffix in {".pyc", ".pyo"}:
        return False
    if relative.parts and (relative.parts[0] == "output"
                           or relative.parts[0] in not_shipped(root)):
        return False
    return path.is_file()


def tracked_files(root: Path) -> list[Path]:
    """Return the files git tracks, as absolute paths, sorted.

    The archive and the manifest must describe a commit. Walking the filesystem
    describes whatever happens to be lying in the working directory instead, so
    an editor backup, a scratch script or a stray tool output ships in the
    release and is recorded in SHA256SUMS.txt as though it were part of the
    project. Enumerating from git removes that whole class of accident.

    -z is required: plain `git ls-files` renders a tracked file's non-ASCII name
    as a C-style quoted escape.
    """
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cannot enumerate tracked files under {root}: git exited "
            f"{completed.returncode}. A release must be built from a git checkout."
        )
    paths = [
        root / name.decode("utf-8")
        for name in completed.stdout.split(b"\0")
        if name
    ]
    # The archive stores what a link points at, which may be a run under
    # output/ or a file outside the checkout. None is tracked today.
    links = [path.relative_to(root).as_posix() for path in paths if path.is_symlink()]
    if links:
        raise RuntimeError(
            "Refusing to package tracked symbolic links; the archive would hold their "
            "targets: " + ", ".join(links)
        )
    return sorted(path for path in paths if included(path, root) and path.is_file())


def executable_files(root: Path) -> frozenset[Path]:
    """The tracked files git records as 100755, as absolute paths.

    The archive takes its modes from here, not the disk: with core.fileMode
    false, as git sets it on a drive without Unix modes, git status passes a
    checkout whose bits are all set or all clear.
    """
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-s", "-z"], capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cannot read the modes git records under {root}: git exited {completed.returncode}"
        )
    executables: set[Path] = set()
    for entry in completed.stdout.split(b"\0"):
        if entry:
            fields, _, name = entry.partition(b"\t")
            if fields.split(b" ", 1)[0] == b"100755":
                executables.add(root / name.decode("utf-8"))
    return frozenset(executables)


def tree_differences(root: Path) -> list[str]:
    """How the tracked files on disk differ from HEAD; empty when they do not.

    `git status` alone trusted a file marked --assume-unchanged or
    --skip-worktree, so an edit to one shipped under the commit's name. Every
    tracked file is therefore also hashed and compared with HEAD's blob.
    """

    def git(*args: str, data: bytes | None = None) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(root), *args], input=data, capture_output=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Cannot inspect the git status of {root}: git {args[0]} failed")
        return completed.stdout

    # -z: unquoted names, as ls-tree -z gives them.
    differences = [entry for entry in git("status", "--porcelain", "-z", "--untracked-files=no")
                   .decode("utf-8").split("\0") if entry.strip()]
    seen = {entry[3:] for entry in differences}
    blobs: dict[str, str] = {}
    for entry in git("ls-tree", "-r", "-z", "HEAD").split(b"\0"):
        if entry:
            fields, _, name = entry.partition(b"\t")
            mode, kind, blob = fields.decode().split()
            if kind == "blob" and mode != "120000" and name.decode("utf-8") not in seen:
                blobs[name.decode("utf-8")] = blob
    present = sorted(name for name in blobs if (root / name).is_file())
    hidden = "marked --assume-unchanged or --skip-worktree"
    differences += [f" D {name} ({hidden})" for name in sorted(set(blobs) - set(present))]
    hashed = git("hash-object", "--stdin-paths",
                 data="".join(f"{name}\n" for name in present).encode("utf-8"))
    for name, blob in zip(present, hashed.decode().split(), strict=True):
        if blob != blobs[name]:
            differences.append(f" M {name} ({hidden})")
    return differences


def assert_tree_matches_git(root: Path) -> None:
    """Refuse to package a tree whose tracked files differ from the commit.

    Enumerating from git fixes which files ship. It does not fix their contents:
    an uncommitted edit to a tracked file would still be packaged. Untracked
    files are ignored here because they are no longer packaged either way.
    """
    dirty = tree_differences(root)
    if dirty:
        raise RuntimeError(
            "Refusing to build a release from uncommitted changes; the archive "
            "would not describe any commit:\n  " + "\n  ".join(dirty)
        )


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build_manifest(
    root: Path, destination: Path, extra: Sequence[Path] = ()
) -> list[tuple[str, str]]:
    """Hash what git tracks, and `extra`: files that ship untracked (the wheels)."""
    rows = [
        (digest(path), path.relative_to(root).as_posix())
        for path in sorted({*tracked_files(root), *extra})
    ]
    destination.write_text(
        "".join(f"{hash_value}  {relative}\n" for hash_value, relative in rows),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="SHA256SUMS.txt")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = build_manifest(root, root / args.output)
    print(f"manifest_entries={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
