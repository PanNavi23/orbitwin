"""ranking_day_config.py's T_max check on the templates only GMAT reads (R6, R7, R8)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from conftest import tmax_as_announced_cases

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.gmat


@pytest.mark.parametrize("name", ["R6", "R7", "R8"])
def test_a_t_max_announced_rounded_is_taken_with_the_flag(
    tmp_path: Path, name: str, gmat: tuple[str, str]
) -> None:
    console, startup = gmat

    def run(script: str, horizon: str, boundary: int, out: Path, *flags: str):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "ranking_day_config.py"),
             "--script", str(ROOT / "examples" / "ranking" / script), "--output", str(out),
             "--horizon-s", horizon, "--mission-boundary-km", str(boundary),
             "--closest-approach-window", "after_alignment", "--team-number", "54",
             "--gmat-console", console, "--startup-file", startup, *flags],
            capture_output=True, text=True, check=False, timeout=600,
        )

    tmax_as_announced_cases(run, name, tmp_path)
