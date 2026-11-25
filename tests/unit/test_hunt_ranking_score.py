"""scripts/hunt_ranking_score.py, run as an operator would run it."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from conftest import bare_environment

from orbitwin.io import read_candidates
from orbitwin.models import Candidate
from orbitwin.ranking_search import RankingSearchResult

MU = 398600.4418


def _script(tmp_path: Path, no_maneuver_script: Path) -> Path:
    """The pursuer on 7000 km, the target 300 km above, phased for a transfer."""
    target_km = 7300.0
    transfer_s = math.pi * math.sqrt(((7000.0 + target_km) / 2.0) ** 3 / MU)
    target_ta = (180.0 - math.degrees(math.sqrt(MU / target_km**3) * transfer_s)) % 360.0
    text = no_maneuver_script.read_text(encoding="utf-8")
    for old, new in {
        "Pursuer.TA = 300;": "Pursuer.TA = 0;",
        "AlienSC.SMA = 9000;": f"AlienSC.SMA = {target_km!r};",
        "AlienSC.INC = 43;": "AlienSC.INC = 28;",
        "AlienSC.RAAN = 65;": "AlienSC.RAAN = 15;",
        "AlienSC.TA = 20;": f"AlienSC.TA = {target_ta!r};",
    }.items():
        text = text.replace(old, new)
    path = tmp_path / "coplanar.script"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("cli_floor, code", [(None, 0), ("100", 2), ("200", 0)])
def test_the_hunt_takes_its_floor_from_the_policy(
    tmp_path: Path, no_maneuver_script: Path, project_root: Path,
    cli_floor: str | None, code: int,
) -> None:
    config = tmp_path / "ranking.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "pursuer_name": "Pursuer",
                "target_name": "AlienSC",
                "policy": {
                    "rules": "ranking", "horizon_s": 4000.0,
                    "dv_limit_km_s": 0.25, "dv_safe_limit_km_s": 0.249,
                    "total_dv_limit_km_s": 3.0, "total_dv_safe_limit_km_s": 2.99,
                    "minimum_flown_altitude_km": 150.0,
                },
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable, str(project_root / "scripts" / "hunt_ranking_score.py"),
        str(_script(tmp_path, no_maneuver_script)), "--config", str(config),
        "--burns", "1", "--transfer-starts", "1", "--random-starts", "0",
        "--polish-top", "1", "--screen-iterations", "20", "--polish-iterations", "20",
        "--workers", "1",
    ]
    if cli_floor is not None:
        command += ["--altitude-floor-km", cli_floor]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == code, done.stdout + done.stderr
    if code:
        assert "minimum_flown_altitude_km = 150" in done.stderr
        return
    floor = 150.0 if cli_floor is None else float(cli_floor)
    assert f"radius >= {6378.1363 + floor:.1f} km" in done.stdout


def test_the_hunt_writes_a_hit_inside_the_walls(
    tmp_path: Path, no_maneuver_script: Path, project_root: Path
) -> None:
    config = tmp_path / "ranking.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "pursuer_name": "Pursuer",
                "target_name": "AlienSC",
                "policy": {
                    "rules": "ranking",
                    "horizon_s": 4000.0,
                    "dv_limit_km_s": 0.25,
                    "dv_safe_limit_km_s": 0.249,
                    "total_dv_limit_km_s": 3.0,
                    "total_dv_safe_limit_km_s": 2.99,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "hunt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "hunt_ranking_score.py"),
            str(_script(tmp_path, no_maneuver_script)),
            "--config", str(config),
            "--burns", "1", "2",
            "--transfer-starts", "2",
            "--random-starts", "1",
            "--polish-top", "1",
            "--screen-iterations", "60",
            "--polish-iterations", "100",
            "--workers", "1",
            "--output", str(output),
            "--candidates-output", str(tmp_path / "candidates.json"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "distance, then dv" in completed.stdout
    # Ranked with the rest now, so printed once.
    assert completed.stdout.count("zero-burn") == 1
    best = json.loads(output.read_text(encoding="utf-8"))[0]
    assert best["intercept_time_s"] is not None
    assert best["minimum_radius_km"] >= 6378.1363 + 100.0
    for burn in best["candidate"]["burns"]:
        assert burn["magnitude_km_s"] <= 0.249 + 1.0e-12

    # The candidates file is what rank and certify read.
    from orbitwin.config import load_config
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.io import read_candidates
    from orbitwin.script_renderer import render_candidate

    archive = tmp_path / "candidates.json"
    candidates = read_candidates(archive)
    assert candidates and all(burn.frame for c in candidates for burn in c.burns)
    script = tmp_path / "coplanar.script"
    scenario, document = parse_gmat_script(script, load_config(config))
    # Every row the JSON shows, once each, stamped for this script and horizon.
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert len(candidates) == len(rows) >= 2
    ids = [c.candidate_id for c in candidates]
    assert len(ids) == len(set(ids))
    assert sum(1 for c in candidates if not c.burns) == 1
    for candidate in candidates:
        assert candidate.metadata["scenario_sha256"] == scenario.source_sha256
        assert candidate.metadata["horizon_s"] == scenario.max_time_s
        # Hunted with no score block, which rank names below.
        assert "searched_score" in candidate.metadata
        assert candidate.metadata["searched_score"] is None
    for candidate in candidates:
        render_candidate(document, scenario, candidate, load_config(config))
    scored = tmp_path / "scored.yaml"
    body = yaml.safe_load(config.read_text(encoding="utf-8"))
    body["score"] = {"kt": 0.002, "ct": 2000.0, "kv": 0.002, "cv": 1000.0, "time_unit": "s",
                     "delta_v_unit": "m/s", "cr_km": 100.0}
    scored.write_text(yaml.safe_dump(body), encoding="utf-8")
    ranked = subprocess.run(
        [sys.executable, "-m", "orbitwin", "rank", str(script), "--config", str(scored),
         "--candidates", str(archive), "--output", str(tmp_path / "rank")],
        capture_output=True, text=True, timeout=300, check=False,
        env=bare_environment(PYTHONPATH=str(project_root / "src"), PATH="/usr/bin:/bin"),
    )
    assert ranked.returncode == 0, ranked.stdout + ranked.stderr
    assert "orbitwin rank: WARNING: " in ranked.stderr and "searched unscored" in ranked.stderr


def _load_hunt(project_root: Path) -> ModuleType:
    """In process, so a test can reach the search it runs."""
    spec = importlib.util.spec_from_file_location(
        "hunt_ranking_score", project_root / "scripts" / "hunt_ranking_score.py"
    )
    assert spec is not None and spec.loader is not None
    hunt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hunt)
    return hunt


def _hunt_zero_burns(
    hunt: ModuleType, tmp_path: Path, no_maneuver_script: Path,
    monkeypatch: pytest.MonkeyPatch, *extra: str,
) -> list[Candidate]:
    """Run the hunt, check no candidate id is written twice, return the zero burns."""
    config = tmp_path / "ranking.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "pursuer_name": "Pursuer",
                "target_name": "AlienSC",
                "policy": {
                    "rules": "ranking", "horizon_s": 4000.0,
                    "dv_limit_km_s": 0.25, "dv_safe_limit_km_s": 0.249,
                    "total_dv_limit_km_s": 3.0, "total_dv_safe_limit_km_s": 2.99,
                },
            }
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "candidates.json"
    monkeypatch.setattr(sys, "argv", [
        "hunt_ranking_score.py", str(_script(tmp_path, no_maneuver_script)),
        "--config", str(config), "--burns", "1", "--transfer-starts", "2",
        "--random-starts", "0", "--polish-top", "2", "--screen-iterations", "20",
        "--polish-iterations", "20", "--workers", "1", "--candidates-output", str(archive),
        *extra,
    ])
    assert hunt.main() == 0
    candidates = read_candidates(archive)
    ids = [c.candidate_id for c in candidates]
    assert len(ids) == len(set(ids))
    return [c for c in candidates if not c.burns]


@pytest.mark.usefixtures("pruning_polish")
def test_the_hunt_writes_the_zero_burn_once(
    tmp_path: Path, no_maneuver_script: Path, project_root: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    hunt = _load_hunt(project_root)
    zero = _hunt_zero_burns(hunt, tmp_path, no_maneuver_script, monkeypatch)
    assert len(zero) == 1 and zero[0].metadata["search_label"] == "zero-burn"
    assert capsys.readouterr().out.count("zero-burn") == 1


def test_the_hunt_writes_a_schedule_the_search_returns_twice_once(
    tmp_path: Path, no_maneuver_script: Path, project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hunt's own guard, under a search that returns the zero burn under
    another label beside the baseline, as search_ranking once did."""
    hunt = _load_hunt(project_root)
    real = hunt.search_ranking

    def doubled(*args: Any, **kwargs: Any) -> RankingSearchResult:
        result = real(*args, **kwargs)
        assert result.baseline is not None
        result.solutions.insert(0, dataclasses.replace(result.baseline, label="a/polished"))
        return result

    monkeypatch.setattr(hunt, "search_ranking", doubled)
    zero = _hunt_zero_burns(hunt, tmp_path, no_maneuver_script, monkeypatch)
    assert len(zero) == 1


def test_the_hunt_shows_the_zero_burn_outside_the_top_rows(
    tmp_path: Path, no_maneuver_script: Path, project_root: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """best_per_margin never holds the zero burn, so only its own row brings it in."""
    hunt = _load_hunt(project_root)
    real = hunt.search_ranking
    seen: list[RankingSearchResult] = []

    def recorded(*args: Any, **kwargs: Any) -> RankingSearchResult:
        seen.append(real(*args, **kwargs))
        return seen[-1]

    monkeypatch.setattr(hunt, "search_ranking", recorded)
    zero = _hunt_zero_burns(hunt, tmp_path, no_maneuver_script, monkeypatch, "--top", "1")
    result = seen[0]
    assert result.solutions[0] is not result.baseline
    assert result.baseline not in hunt.best_per_margin(result).values()
    assert len(zero) == 1 and zero[0].metadata["search_label"] == "zero-burn"
    assert capsys.readouterr().out.count("zero-burn") == 1


def test_the_burn_budget_is_checked() -> None:
    from orbitwin.config import OrbitWinConfig, validate_config
    from orbitwin.errors import ConfigurationError

    base = OrbitWinConfig()
    assert base.search.ranking_max_burns == 13
    bad = dataclasses.replace(base, search=dataclasses.replace(base.search, ranking_max_burns=0))
    with pytest.raises(ConfigurationError, match="ranking_max_burns"):
        validate_config(bad)


def test_without_burns_the_hunt_seeds_up_to_the_budget(
    tmp_path: Path, no_maneuver_script: Path, project_root: Path
) -> None:
    config = tmp_path / "ranking.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "pursuer_name": "Pursuer",
                "target_name": "AlienSC",
                "search": {"ranking_max_burns": 2},
                "policy": {
                    "rules": "ranking", "horizon_s": 4000.0,
                    "dv_limit_km_s": 0.25, "dv_safe_limit_km_s": 0.249,
                    "total_dv_limit_km_s": 3.0, "total_dv_safe_limit_km_s": 2.99,
                },
            }
        ),
        encoding="utf-8",
    )
    done = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "hunt_ranking_score.py"),
         str(_script(tmp_path, no_maneuver_script)), "--config", str(config),
         "--transfer-starts", "1", "--random-starts", "0", "--polish-top", "1",
         "--screen-iterations", "10", "--polish-iterations", "10", "--workers", "1"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "burns [1, 2]" in done.stdout
