"""What an installed orbitwin records about itself (not a checkout, not a release archive).

The installer writes it beside the package: the version, the source in direct_url.json
(PEP 610; a commit when installed from git), and RECORD, the hash of every file it put
there. The files under orbitwin/, direct_url.json and the `orbitwin` command are held to
those hashes. Bytecode cannot be held to its source (code objects compare equal across
specialized instructions that run differently), so a day runs only in a process that
compiled every module afresh (`compiled_afresh`). What else the environment loads (other packages, .pth files, sitecustomize) is
the environment's: it is listed, not judged.
"""

from __future__ import annotations

import base64
import csv
import dataclasses
import hashlib
import json
import os
import re
import sys
from importlib import metadata
from pathlib import Path

from .layout import PACKAGE

DISTRIBUTION = "orbitwin"
#: Hashes RECORD may use for a file to count as checked; a weaker one is a change.
_STRONG = frozenset({"sha256", "sha384", "sha512"})
#: A git commit: SHA-1 or SHA-256, in full.
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
#: Set by `orbitwin day` and `run` to the empty bytecode cache they re-run themselves with.
FRESH_PYCACHE = "ORBITWIN_FRESH_PYCACHE"


@dataclasses.dataclass(frozen=True)
class Installed:
    version: str
    #: The commit direct_url.json names (installed from git), or None.
    commit: str | None
    #: Where it was installed from, as direct_url.json says, or None (an index).
    url: str | None
    #: What is not as RECORD says: changed, missing, added, unreadable, or bytecode that
    #: is not its source's.
    changed: tuple[str, ...]
    #: What the environment loads at startup besides: .pth files and sitecustomize.
    startup: tuple[str, ...] = ()
    #: Whether this process compiled every module afresh (`compiled_afresh`).
    fresh: bool = False


def installed() -> Installed | None:
    """The record of the orbitwin this module belongs to, or None when it has none."""
    package = (PACKAGE / "__init__.py").resolve()
    # Every orbitwin record on the path: a stale one elsewhere must not hide this one's.
    ours = []
    for dist in metadata.distributions(name=DISTRIBUTION):
        try:
            located = Path(str(dist.locate_file("orbitwin/__init__.py"))).resolve()
        except (OSError, ValueError):
            continue
        if located == package:
            ours.append(dist)
    if not ours:
        return None
    record = _record(ours[0])
    if len(ours) > 1:
        # Which one Python reads first is not a record of what was installed.
        record = dataclasses.replace(record, changed=tuple(sorted(
            [*record.changed, f"{len(ours)} installer records for this one package"])))
    return record


def _record(dist: metadata.Distribution) -> Installed:
    commit = url = None
    notes: list[str] = []
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "null")
    except ValueError:
        direct = None
        notes.append("direct_url.json (unreadable)")
    if isinstance(direct, dict):
        url = direct.get("url") if isinstance(direct.get("url"), str) else None
        vcs = direct.get("vcs_info")
        if isinstance(vcs, dict) and isinstance(vcs.get("commit_id"), str):
            named = vcs["commit_id"].strip().lower()
            if _COMMIT.fullmatch(named):
                commit = named
            else:
                notes.append(f"direct_url.json (commit_id {named!r} is not a commit)")
    rows = _rows(dist)
    if direct is not None:
        # The commit is read from it: it must be the file the installer wrote.
        own = next((name for name in rows if name.endswith("/direct_url.json")
                    and name.split("/", 1)[0].endswith(".dist-info")), None)
        if own is None or not rows[own]:
            notes.append("direct_url.json (no hash in RECORD)")
    return Installed(version=dist.version, commit=commit, url=url,
                     changed=tuple(sorted([*notes, *_changes(dist, rows)])),
                     startup=tuple(_startup(dist)), fresh=compiled_afresh())


def compiled_afresh() -> bool:
    """Whether this process reads bytecode only from the empty cache it was started with,
    so every module it imports was compiled from the hash-held sources."""
    fresh = os.environ.get(FRESH_PYCACHE)
    if fresh is None or sys.pycache_prefix is None:
        return False
    cache = Path(sys.pycache_prefix)
    try:
        info = cache.lstat()
    except OSError:
        return False
    # Ours alone (as mkdtemp makes it): no one else can have put bytecode in it.
    return (cache.resolve() == Path(fresh).resolve() and not cache.is_symlink()
            and info.st_uid == os.getuid() and info.st_mode & 0o077 == 0)


def _rows(dist: metadata.Distribution) -> dict[str, str]:
    """RECORD as written, path to hash ("" when none). `dist.files` leaves out rows whose
    file is gone (Python 3.12 on), which would hide a missing file."""
    rows: dict[str, str] = {}
    for row in csv.reader((dist.read_text("RECORD") or "").splitlines()):
        if row:
            rows[row[0]] = row[1] if len(row) > 1 else ""
    return rows


def _digest(path: Path, mode: str) -> str:
    raw = hashlib.new(mode, path.read_bytes()).digest()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _held(dist: metadata.Distribution, name: str, hashed: str, *,
          need_hash: bool) -> str | None:
    """Why ``name`` is not the file RECORD lists, or None."""
    path = Path(str(dist.locate_file(name)))
    mode, _, value = hashed.partition("=")
    if not path.is_file():
        return f"{name} (missing)"
    if not value:
        return f"{name} (no hash in RECORD)" if need_hash else None
    if mode not in _STRONG:
        return f"{name} ({mode} in RECORD, not sha256 or stronger)"
    try:
        same = _digest(path, mode) == value
    except (OSError, ValueError) as exc:
        return f"{name} (unreadable: {exc})"
    return None if same else f"{name} (changed)"


def _changes(dist: metadata.Distribution, rows: dict[str, str]) -> list[str]:
    ours = {name: hashed for name, hashed in rows.items()
            if name.split("/", 1)[0] == "orbitwin" and "__pycache__" not in name.split("/")}
    changes = [] if ours else ["RECORD (missing or empty)"]
    for name, hashed in sorted(rows.items()):
        if "__pycache__" in name.split("/"):
            continue
        # Everything under orbitwin/ needs a hash; elsewhere (the `orbitwin` command,
        # direct_url.json, METADATA) a hash RECORD gives is held, RECORD itself has none.
        why = _held(dist, name, hashed, need_hash=name in ours)
        if why is not None:
            changes.append(why)
    for directory, subdirectories, files in os.walk(PACKAGE):
        here = Path(directory)
        for sub in list(subdirectories):
            # A linked folder is code the walk would not see.
            if (here / sub).is_symlink():
                changes.append(f"{(here / sub).relative_to(PACKAGE.parent).as_posix()} "
                               "(a link, not installed)")
        subdirectories[:] = sorted(name for name in subdirectories
                                   if name != "__pycache__" and not (here / name).is_symlink())
        for file in files:
            name = (here / file).relative_to(PACKAGE.parent).as_posix()
            # Bytecode outside __pycache__ is imported in place of the source: not cache.
            if name not in ours:
                changes.append(f"{name} (not installed)")
    return changes


def _startup(dist: metadata.Distribution) -> list[str]:
    """The .pth files and sitecustomize beside the package, which Python runs at startup."""
    site = Path(str(dist.locate_file(""))).resolve()
    found = sorted(path.name for path in site.glob("*.pth"))
    found += [name for name in ("sitecustomize.py", "usercustomize.py")
              if (site / name).is_file()]
    return found


def describe(record: Installed) -> str:
    """One line for `orbitwin --version`."""
    source = (f"commit {record.commit}" if record.commit is not None
              else "no commit recorded")
    line = f"orbitwin {record.version}, installed, {source}"
    if record.url is not None:
        line += f", from {record.url}"
    if record.changed:
        line += f", {len(record.changed)} file(s) differ from what was installed"
    return line
