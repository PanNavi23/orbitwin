#!/usr/bin/env python3
"""Which pull requests actually change the answer?

The stack is a tree, not a list: #33, #37 and #41 each fork. A per-tip walk
recomputes the shared 33-layer spine once per leaf and cannot express a fork.
This solves every distinct branch head exactly once and compares each pull
request against *its own base branch*, which is the only comparison a stacked
PR is actually claiming anything about.

Volatile evidence fields are stripped before comparison: gmat_run_id carries a
timestamp and a nonce, and startup_sha256 hashes a derived file embedding a
per-run output path, so two identical runs disagree on them by design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

#: orbitwin.pinned.OVERRIDE: the solve below refuses versions off the pins without it.
PIN_OVERRIDE = "ORBITWIN_ALLOW_UNPINNED"
VOLATILE_KEY = re.compile(r"run_id|sha256|_path|directory|report|timestamp|log", re.IGNORECASE)
VOLATILE_VALUE = re.compile(r"\d{8}T\d{6}Z|/tmp/|/var/folders/|scratchpad")


def _strip(node: object) -> object:
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in sorted(node.items()) if not VOLATILE_KEY.search(k)}
    if isinstance(node, list):
        return [_strip(item) for item in node]
    if isinstance(node, str) and VOLATILE_VALUE.search(node):
        return "<volatile>"
    return node


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            stripped = _strip(json.load(handle))
    except (json.JSONDecodeError, OSError):
        return None
    return hashlib.sha256(json.dumps(stripped, sort_keys=True).encode()).hexdigest()[:16]


def _prs() -> list[dict]:
    out = subprocess.run(
        ["gh", "pr", "list", "--state", "open", "--limit", "200",
         "--json", "number,headRefName,baseRefName,title"],
        capture_output=True, text=True, check=True,
    ).stdout
    return sorted(json.loads(out), key=lambda p: p["number"])


def _solve(worktree: Path, branch: str, out_root: Path, python: str) -> tuple[str | None, str | None, str]:
    """Check the worktree out at a branch and run one search."""
    checkout = subprocess.run(
        ["git", "-C", str(worktree), "checkout", "-q", "--detach", f"origin/{branch}"],
        capture_output=True, text=True,
    )
    if checkout.returncode != 0:
        return None, None, f"checkout failed: {checkout.stderr.strip()[:60]}"

    config = worktree / "config" / "quick.yaml"
    # The caller's copy, so every branch solves the same bytes, older ones included.
    script = Path(__file__).resolve().parents[1] / "examples" / "two_burn_rendezvous.script"
    if not config.is_file():
        return None, None, "no config/quick.yaml"
    if not script.is_file():
        return None, None, "no example script"

    out = out_root / re.sub(r"[^A-Za-z0-9]+", "_", branch)
    environment = {"PYTHONPATH": str(worktree / "src"), "PATH": "/usr/bin:/bin",
                   "HOME": str(Path.home())}
    if PIN_OVERRIDE in os.environ:
        environment[PIN_OVERRIDE] = os.environ[PIN_OVERRIDE]
    # quick.yaml is the preliminary's; a CLI that has --rules needs it named.
    cli = worktree / "src" / "orbitwin" / "cli.py"
    named = cli.is_file() and "add_rules_argument" in cli.read_text(encoding="utf-8")
    rules = ["--rules", "preliminary"] if named else []
    result = subprocess.run(
        [python, "-m", "orbitwin", "solve", str(script),
         "--config", str(config), *rules, "--output", str(out)],
        capture_output=True, text=True, cwd=worktree, env=environment, timeout=900,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()
        return None, None, f"solve failed: {tail[-1][:70] if tail else '?'}"
    return _digest(out / "candidates.json"), _digest(out / "summary.json"), ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    worktree = Path(args.worktree)
    # solve runs in the worktree, and the digests are read from here.
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    prs = _prs()
    branches = {"main"}
    for pr in prs:
        branches.add(pr["headRefName"])
        branches.add(pr["baseRefName"])

    results: dict[str, tuple[str | None, str | None, str]] = {}
    for index, branch in enumerate(sorted(branches), start=1):
        results[branch] = _solve(worktree, branch, out_root, args.python)
        note = results[branch][2] or f"cand={results[branch][0]} summ={results[branch][1]}"
        print(f"[{index:>2}/{len(branches)}] {branch:<55} {note}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print(f"{'PR':>4}  {'verdict':<16} {'base':<48} title", flush=True)
    print("=" * 100, flush=True)

    rows = []
    for pr in prs:
        head = results[pr["headRefName"]]
        base = results[pr["baseRefName"]]
        if head[2]:
            verdict = "BROKEN"
        elif base[2]:
            verdict = "NO-BASELINE"
        elif head[0] != base[0]:
            verdict = "ANSWER-CHANGED"
        elif head[1] != base[1]:
            verdict = "SUMMARY-ONLY"
        else:
            verdict = "no-op"
        rows.append((pr["number"], verdict, pr["baseRefName"], pr["title"], head[2] or base[2]))
        print(f"{pr['number']:>4}  {verdict:<16} {pr['baseRefName'][:47]:<48} {pr['title'][:60]}", flush=True)

    print("\n--- tally ---", flush=True)
    for kind in ("ANSWER-CHANGED", "SUMMARY-ONLY", "no-op", "BROKEN", "NO-BASELINE"):
        hits = [r[0] for r in rows if r[1] == kind]
        print(f"{kind:<16} {len(hits):>3}  {', '.join('#' + str(h) for h in hits)}", flush=True)

    (out_root / "verdicts.json").write_text(json.dumps(
        [{"pr": r[0], "verdict": r[1], "base": r[2], "title": r[3], "note": r[4]} for r in rows],
        indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
