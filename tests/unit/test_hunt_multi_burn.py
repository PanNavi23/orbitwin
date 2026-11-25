"""scripts/hunt_multi_burn.py picks its impulse counts from the total delta-v cap."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

RANKING_YAML = """\
policy:
  rules: ranking
  horizon_s: 20000.0
  dv_limit_km_s: 0.25
  dv_safe_limit_km_s: 0.249
  total_dv_limit_km_s: 3.0
  total_dv_safe_limit_km_s: 2.99
score: {kt: 0.001, ct: 5000.0, kv: 0.001, cv: 1500.0, time_unit: s, delta_v_unit: m/s,
        cr_km: 100.0}
"""


@pytest.fixture()
def hunt(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[set[int]]]:
    spec = importlib.util.spec_from_file_location(
        "orbitwin_hunt_multi_burn", project_root / "scripts" / "hunt_multi_burn.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    searched: list[set[int]] = []

    def no_search(scenario: object, config: object, *, starts: list[Any], **_: object) -> list:
        # The wiring is what is under test, not the minutes-long search.
        searched.append({start.burn_count for start in starts})
        return []

    monkeypatch.setattr(module, "search_multi_burn", no_search)
    return module, searched


def _run(module: Any, monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["hunt_multi_burn.py", *argv])
    return int(module.main())


def test_without_burns_the_counts_come_from_the_total_cap(
    hunt: tuple[Any, list[set[int]]],
    monkeypatch: pytest.MonkeyPatch,
    high_plane_script: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, searched = hunt
    config = tmp_path / "ranking.yaml"
    config.write_text(RANKING_YAML, encoding="utf-8")
    assert _run(module, monkeypatch, str(high_plane_script), "--config", str(config),
                "--workers", "1") == 1
    assert searched == [{5, 6, 7, 13}]
    assert "burns [5, 6, 7, 13]: 13 impulses at 0.249 km/s" in capsys.readouterr().out


def test_explicit_burns_still_win(
    hunt: tuple[Any, list[set[int]]],
    monkeypatch: pytest.MonkeyPatch,
    high_plane_script: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, searched = hunt
    config = tmp_path / "ranking.yaml"
    config.write_text(RANKING_YAML, encoding="utf-8")
    _run(module, monkeypatch, str(high_plane_script), "--config", str(config),
         "--workers", "1", "--burns", "4")
    assert searched == [{4}]
    assert "burns [4]: given with --burns" in capsys.readouterr().out
