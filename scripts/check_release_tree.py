#!/usr/bin/env python3
"""Refuse an unpacked release that holds files its SHA256SUMS.txt does not list.

RUNBOOK step 0 runs this from the unpacked archive, after `sha256sum --check`,
which checks every listed file but says nothing about others: an archive
unzipped over an older tree kept the older tree's files, a stale module or
conftest.py among them, and step 0 passed.

What step 0 and the later steps make are not refused: .venv/, run/, output/,
dist/, build/, what the operator copies into input/original/, GMAT's
GmatLog.txt, caches and bytecode, and setuptools' src/*.egg-info.

    python3 -I scripts/check_release_tree.py [ROOT]

Standard library only, and no syntax past 3.8: it runs before the bootstrap,
on whatever python3 the machine has.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MANIFEST = "SHA256SUMS.txt"
#: Top-level entries step 0 and the later steps make.
MADE_HERE = frozenset({".venv", "run", "output", "dist", "build", "GmatLog.txt"})
#: Directory names that are caches wherever they are.
CACHES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov"})


def made_here(relative: str) -> bool:
    parts = relative.split("/")
    if parts[0] in MADE_HERE or relative.startswith("input/original/"):
        return True
    if any(part in CACHES for part in parts[:-1]) or parts[0] == ".coverage":
        return True
    if parts[0] == "src" and len(parts) > 1 and parts[1].endswith(".egg-info"):
        return True
    return parts[-1].endswith((".pyc", ".pyo")) or parts[-1].startswith(".coverage.")


def unlisted(root: Path) -> list[str]:
    """Files under root the manifest does not list, beside the manifest itself."""
    listed = {MANIFEST}
    for line in (root / MANIFEST).read_text(encoding="utf-8").splitlines():
        listed.add(line.split("  ", 1)[1])
    found = []
    for directory, subdirectories, files in os.walk(root):
        base = Path(directory).relative_to(root).as_posix()
        prefix = "" if base == "." else base + "/"
        # A link to a directory is reported, not walked.
        names = files + [name for name in subdirectories if os.path.islink(os.path.join(directory, name))]
        subdirectories[:] = sorted(
            name for name in subdirectories
            if not os.path.islink(os.path.join(directory, name))
            and not made_here(prefix + name + "/x") and name not in CACHES
        )
        found += [prefix + name for name in names
                  if prefix + name not in listed and not made_here(prefix + name)]
    return sorted(found)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not (root / MANIFEST).is_file():
        print(f"error: {root / MANIFEST} is missing: this is not an unpacked release archive.",
              file=sys.stderr)
        return 1
    extra = unlisted(root)
    if extra:
        print(f"error: {root} holds {len(extra)} file(s) {MANIFEST} does not list, such as an "
              "older tree the archive was unzipped over:", file=sys.stderr)
        for name in extra[:20]:
            print(f"  {name}", file=sys.stderr)
        if len(extra) > 20:
            print(f"  ... and {len(extra) - 20} more", file=sys.stderr)
        print("Unzip the archive again into a new, empty directory (docs/RUNBOOK.md step 0).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
