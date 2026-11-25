"""scripts/rehearse_ranking.py writes a loadable config per scenario and placeholder C_r."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from orbitwin.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "rehearse_ranking", ROOT / "scripts" / "rehearse_ranking.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rehearse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *args: str) -> list[Any]:
    module = _module()
    solved: list[Any] = []

    def solve(script: Path, config: Any, output: Path, **_: Any) -> dict[str, Any]:
        solved.append((Path(script).name, config, output.name))
        return {"best_candidate": None, "best_candidate_stage": None, "gmat_corrections": [],
                "gmat_verified_count": 0}

    monkeypatch.setattr(module, "solve_scenario", solve)
    monkeypatch.setenv("GMAT_CONSOLE", "unused")
    monkeypatch.setenv("GMAT_STARTUP", "unused")
    monkeypatch.setattr(sys, "argv", ["rehearse_ranking.py", "--output",
                                      str(tmp_path / "out"), *args])
    assert module.main() == 0
    return solved


def test_the_placeholder_c_r_is_written_in_the_revised_units(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (solved,) = _rehearse(monkeypatch, tmp_path, "R5")
    script, config, output = solved
    assert (script, output) == ("R5_TiltedPlane_CloseMiss.script", "R5")
    assert config.score.cr_km == 300.0 and config.score.delta_v_unit == "m/s"
    assert (config.score.kv, config.score.cv) == (0.0012, 2000.0)
    assert load_config(tmp_path / "out" / "R5.yaml").score == config.score


def test_several_c_r_run_each_scenario_once_per_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    solved = _rehearse(monkeypatch, tmp_path, "R2", "R5", "--cr-km", "100", "300", "1000", "3000")
    assert [(output, config.score.cr_km) for _, config, output in solved] == [
        ("R2-cr100", 100.0), ("R2-cr300", 300.0), ("R2-cr1000", 1000.0), ("R2-cr3000", 3000.0),
        ("R5-cr100", 100.0), ("R5-cr300", 300.0), ("R5-cr1000", 1000.0), ("R5-cr3000", 3000.0),
    ]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (["0"], "argument --cr-km: not a positive finite number: '0'"),
        (["-300"], "argument --cr-km: not a positive finite number: '-300'"),
        (["nan"], "argument --cr-km: not a positive finite number: 'nan'"),
        (["inf"], "argument --cr-km: not a positive finite number: 'inf'"),
        # Equal as run names though not as floats: one would overwrite the other.
        (["300", "300.0000001"], "must differ in their run names: ['cr300', 'cr300']"),
    ],
)
def test_c_r_values_are_refused_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    values: list[str], message: str,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "solve_scenario", lambda *a, **k: pytest.fail("ran"))
    monkeypatch.setattr(sys, "argv", ["rehearse_ranking.py", "--output", str(tmp_path / "out"),
                                      "R5", "--cr-km", *values])
    with pytest.raises(SystemExit) as refused:
        module.main()
    assert refused.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_c_r_is_only_taken_with_its_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    monkeypatch.setattr(sys, "argv", ["rehearse_ranking.py", "--output", str(tmp_path / "out"),
                                      "R5", "--cr", "300"])
    with pytest.raises(SystemExit) as refused:
        module.main()
    assert refused.value.code == 2 and "unrecognized arguments: --cr" in capsys.readouterr().err
