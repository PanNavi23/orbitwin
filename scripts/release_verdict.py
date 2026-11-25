#!/usr/bin/env python3
"""The last step of release_validate.sh: PASS only if every recorded result says so.

No `assert`: PYTHONOPTIMIZE strips them, and this would then print PASS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


class VerdictError(SystemExit):
    """A result that is missing or says the release is not ready."""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise VerdictError(f"no result from this run: {path} ({error.strerror})") from None


def verdict(root: Path) -> list[str]:
    """Notes for results that were skipped; raises VerdictError on any other."""
    failures = []
    fuzz = json.loads(_read(root / "ood_fuzz_5000.json"))
    if fuzz.get("failure_count") != 0:
        failures.append(f"OOD fuzz: {fuzz.get('failure_count')} failures, "
                        f"{fuzz.get('failures', [])[:3]}")
    matrix = json.loads(_read(root / "scenario_ood_matrix_20.json"))
    if matrix.get("pipeline_exception_count") != 0:
        failures.append(f"scenario matrix: {matrix.get('pipeline_exception_count')} exceptions, "
                        f"{matrix.get('exceptions', [])[:3]}")
    cert = json.loads(_read(root / "fake_certification" / "submission" / "certification.json"))
    runs = cert.get("exact_runs", [])
    if len(runs) != 3 or not all(run.get("success") is True for run in runs):
        failures.append(f"fake certification: exact runs {[run.get('success') for run in runs]}, "
                        "want three successes")
    # It ran under `set +e`, so nothing else reads it.
    dependency = _read(root / "pip_check.exit_code").strip()
    log = _read(root / "pip_check.log")
    if dependency not in ("0", "SKIPPED"):
        failures.append(f"dependency check exited {dependency}:\n{log}")
    if failures:
        raise VerdictError("release validation: FAIL\n" + "\n".join(failures))
    return [f"dependency check: {log.strip()}"] if dependency == "SKIPPED" else []


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "output/validation")
    for note in verdict(root):
        print(note)
    print("release validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
