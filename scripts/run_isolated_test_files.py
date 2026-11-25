#!/usr/bin/env python3
"""Run every unit and integration test file in its own Python process.

Two things this buys that a single pytest run does not. State that outlives a
test file cannot reach the next one -- module-level caches, a worker pool left
behind, an environment variable a test set. And a file that hangs is bounded by
its own timeout rather than the whole suite's.

It costs little. Measured on the same selection `make test` uses -- every
pytest.mark.slow test lives under tests/slow, so tests/unit plus
tests/integration is exactly that set -- one process took 234 s and eighteen
took 250 s on 2026-08-21, about 7%. It is not the default because `make test` is the target
people add -k, -x and --lf to, not because of the clock.

(An earlier version of this said isolation cost minutes, and then that it was
faster. Both were wrong: the first was a guess, and the second came of timing
`pytest -q`, which includes tests/slow, against a runner that does not.)

Output is written straight to the log file handle rather than through a pipe:
pytest can leave a child holding the write end, and reading a pipe that never
closes is a hang with no message.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


#: What `make test` selects. Passing it per file means the runner covers the
#: same tests, so a file of slow-only tests contributes nothing and costs a
#: second rather than being excluded by a directory name nobody updates.
SELECTION = "not slow and not gmat"

#: A file still running after this long is killed; pyproject.toml's
#: faulthandler_timeout is kept below it (tests/unit/test_make_targets.py).
TIMEOUT_PER_FILE_S = 900.0


def test_files(root: Path) -> list[Path]:
    """Every test file under tests/, discovered rather than listed.

    This used to name tests/unit and tests/integration. Adding a third
    directory would then have dropped out of the release gate silently, which
    uses this runner in place of the single-process suite -- it would have
    reported PASS having never opened the files.
    """
    return sorted((root / "tests").rglob("test_*.py"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--log", type=Path, default=Path("output/make/pytest_isolated.log"))
    parser.add_argument("--timeout-per-file", type=float, default=TIMEOUT_PER_FILE_S)
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    log = arguments.log if arguments.log.is_absolute() else root / arguments.log
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    files = test_files(root)
    if not files:
        raise SystemExit("No test files found under tests/")

    failures: list[str] = []
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        for path in files:
            relative = path.relative_to(root).as_posix()
            handle.write(f"=== {relative} ===\n")
            handle.flush()
            file_started = time.monotonic()
            try:
                completed = subprocess.run(
                    # No cache: each file runs once, and it would write into the checkout.
                    [sys.executable, "-m", "pytest", relative, "-q", "-m", SELECTION,
                     "-p", "no:cacheprovider"],
                    cwd=root,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=arguments.timeout_per_file,
                    check=False,
                )
                code = completed.returncode
            except subprocess.TimeoutExpired:
                handle.write(f"\nTIMEOUT after {arguments.timeout_per_file:g} s\n")
                failures.append(f"{relative}: timed out")
                code = None
            elapsed = time.monotonic() - file_started
            if code is not None:
                handle.write(f"\nreturn_code={code}\n")
                # 5 is pytest's "nothing was collected", which is the right
                # answer for a file whose tests the selection excludes.
                if code not in (0, 5):
                    failures.append(f"{relative}: exit {code}")
            handle.write(f"duration_s={elapsed:.3f}\n\n")
            handle.flush()
            status = "ok" if code in (0, 5) else "FAIL"
            print(f"{relative:<52} {elapsed:7.2f}s  {status}", flush=True)

        duration = time.monotonic() - started
        handle.write(f"files={len(files)}\nduration_s={duration:.3f}\n")
        handle.write(f"status={'FAIL' if failures else 'PASS'}\n")
        for failure in failures:
            handle.write(f"- {failure}\n")

    print(f"\nfiles={len(files)} duration_s={duration:.1f} status={'FAIL' if failures else 'PASS'}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
