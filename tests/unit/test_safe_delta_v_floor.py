"""scripts/safe_delta_v_floor.py against the ranking round's total delta-v cap.

The rows below are two real rows from HighPlaneChange_Circular. Under a 2.99 km/s
total the 14.68 km/s arrival cannot be one impulse, and before the cap was read
the table offered it as a seed with "<= 59 burns".
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from orbitwin.config import OrbitWinConfig

EARLY = {
    "arrival_s": 1450.0, "dv_km_s": 14.680904155183768, "burn_s": 0.0,
    "arc_floor_km": 6571.808373420417, "burns_needed": 10,
    "time_points": 24.99516230882247, "dv_points": 4.715690007238165e-05,
    "score": 74.99520946572254,
}
CHEAP = {
    "arrival_s": 10050.0, "dv_km_s": 0.7572458042386396, "burn_s": 6375.0,
    "arc_floor_km": 7202.552038430963, "burns_needed": 1,
    "time_points": 12.18756508789474, "dv_points": 16.939945729721735,
    "score": 79.12751081761647,
}

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


@pytest.fixture(scope="module")
def floor(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "orbitwin_safe_delta_v_floor", project_root / "scripts" / "safe_delta_v_floor.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preliminary_lines_are_what_they_were(floor) -> None:
    """Copied from the origin/main output on HighPlaneChange_Circular, preliminary policy."""
    assert floor.total_cap(OrbitWinConfig().policy) is None
    assert floor.table_line(EARLY, None) == (
        "      1450   14.6809       0.0      6571.8     10  24.995   0.000   74.9952"
    )
    assert floor.table_line(CHEAP, None) == (
        "     10050    0.7572    6375.0      7202.6      1  12.188  16.940   79.1275"
    )
    assert floor.best_lines([CHEAP, EARLY], None) == [
        "best arrivals by score -- SEED A MULTI-IMPULSE SEARCH HERE, do not stop at these numbers.",
        "The delta-v is a single impulse's, which is an upper bound: on the 2026-09-05",
        "scenario the flown five-impulse solution beat this column by 2.15 km/s, so the",
        "scores below understate what these arrivals are worth.",
        "  arrive    10050 s   dV <=  0.7572 km/s   <= 1 burn(s)   burn@  6375.0 s   "
        "score >= 79.1275",
        "  arrive     1450 s   dV <= 14.6809 km/s   <= 10 burn(s)   burn@     0.0 s   "
        "score >= 74.9952",
    ]


def test_the_safe_total_is_the_cap_and_the_formal_one_the_fallback(floor) -> None:
    policy = dataclasses.replace(
        OrbitWinConfig().policy, total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99
    )
    assert floor.total_cap(policy) == (2.99, "policy.total_dv_safe_limit_km_s")
    formal_only = dataclasses.replace(policy, total_dv_safe_limit_km_s=None)
    assert floor.total_cap(formal_only) == (3.0, "policy.total_dv_limit_km_s")


def test_rows_over_the_total_are_marked_and_not_offered_as_seeds(floor) -> None:
    assert floor.table_line(EARLY, 2.99).endswith("  over total")
    assert not floor.table_line(CHEAP, 2.99).endswith("over total")

    lines = floor.best_lines([CHEAP, EARLY], 2.99)
    assert "Left out 1 arrival(s) whose single impulse exceeds the 2.99 km/s total." in lines
    assert [line for line in lines if line.startswith("  arrive")] == [
        floor.best_lines([CHEAP], None)[-1]
    ]


def test_when_nothing_fits_the_total_the_summary_says_so(floor) -> None:
    lines = floor.best_lines([EARLY], 2.99)
    assert not any(line.startswith("  arrive") for line in lines)
    assert lines[-1].startswith("  none fits the total")


def test_a_ranking_run_names_the_cap_and_marks_the_rows(
    floor, project_root: Path, high_plane_script: Path, tmp_path: Path
) -> None:
    config = tmp_path / "ranking.yaml"
    config.write_text(RANKING_YAML, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "safe_delta_v_floor.py"),
         str(high_plane_script), "--config", str(config), "--workers", "1",
         "--arrivals", "1450:10100:8600", "--burns", "0:9000:75",
         "--output", str(tmp_path / "floor.json")],
        capture_output=True, text=True, check=True, timeout=300,
    )
    out = completed.stdout.splitlines()
    assert any(line.startswith("total cap = 2.99 km/s from policy.total_dv_safe_limit_km_s")
               for line in out)
    rows = {line.split()[0]: line for line in out if line[:10].strip() in {"1450", "10050"}}
    assert rows["1450"].endswith("over total")
    assert not rows["10050"].endswith("over total")
    seeds = [line for line in out if line.startswith("  arrive")]
    assert len(seeds) == 1 and "10050 s" in seeds[0]
    marked = {row["arrival_s"]: row["over_total_cap"]
              for row in json.loads((tmp_path / "floor.json").read_text(encoding="utf-8"))}
    assert marked == {1450.0: True, 10050.0: False}


def _falling(different_plane_script: Path, tmp_path: Path) -> Path:
    """SC_Human from 6864 km apogee, 6336 km pericenter: unpowered it lands near 2200 s."""
    text = different_plane_script.read_text(encoding="utf-8")
    for old, new in {"Pursuer.SMA = 7000;": "Pursuer.SMA = 6600;",
                     "Pursuer.ECC = 0;": "Pursuer.ECC = 0.04;",
                     "Pursuer.TA = 300;": "Pursuer.TA = 180;"}.items():
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "falling.script"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_burn_after_the_start_has_landed_is_not_safe(
    floor, different_plane_script: Path, tmp_path: Path
) -> None:
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.orbit import first_radius_crossing_s

    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(_falling(different_plane_script, tmp_path), config)
    landed_s = first_radius_crossing_s(
        scenario.pursuer.initial_state, 6378.1363, scenario.max_time_s, scenario.mu_km3_s2
    )
    assert landed_s is not None and 2100.0 < landed_s < 2300.0
    floor._STATE.update(scenario=scenario, config=config, floor_km=6378.1363, arc_end=None)
    late = floor._row(9000.0, np.arange(3200.0, 6001.0, 200.0))
    assert late is None  # every burn in the grid comes after the landing
    early = floor._row(9000.0, np.arange(200.0, 2001.0, 200.0))
    assert early is None or early["burn_s"] < landed_s


#: R1's two-body rows at 4200 and 4400 s: 360 m/s breaks the 250 m/s cap as one impulse.
OVER_CAP = {
    "arrival_s": 4200.0, "dv_km_s": 0.3601, "burn_s": 0.0, "arc_floor_km": 6783.5,
    "burns_needed": 2, "time_points": 24.273, "dv_points": 21.935, "score": 96.2075,
}
UNDER_CAP = {
    "arrival_s": 4400.0, "dv_km_s": 0.1550, "burn_s": 0.0, "arc_floor_km": 6800.0,
    "burns_needed": 1, "time_points": 23.706, "dv_points": 22.538, "score": 96.2436,
}
MARGIN = {**UNDER_CAP, "arrival_s": 4500.0, "dv_km_s": 0.2495}
CAPS = (0.25, 0.249)


def test_a_single_impulse_over_the_ranking_cap_is_marked(floor) -> None:
    from orbitwin.config import PolicyConfig

    ranking = OrbitWinConfig(policy=dataclasses.replace(
        PolicyConfig(), rules="ranking", dv_limit_km_s=0.25, dv_safe_limit_km_s=0.249))
    assert floor.burn_caps(ranking) == CAPS
    assert floor.burn_caps(OrbitWinConfig()) is None
    assert floor.table_line(OVER_CAP, 2.99, CAPS).endswith("  over burn cap")
    assert floor.table_line(MARGIN, 2.99, CAPS).endswith("  over burn margin")
    assert floor.table_line(UNDER_CAP, 2.99, CAPS) == floor.table_line(UNDER_CAP, 2.99)


def test_no_score_is_claimed_as_a_bound_under_the_ranking_cap(floor) -> None:
    lines = floor.best_lines([UNDER_CAP, OVER_CAP, MARGIN], 2.99, CAPS)
    assert not any("score >=" in line or "<= " in line for line in lines)
    seeds = [line for line in lines if line.startswith("  arrive")]
    assert seeds[0].endswith("two-body score 96.2436, within the cap as one burn")
    assert seeds[1].endswith("OVER BURN CAP: 2 burn(s) at 249 m/s")
    assert seeds[2].endswith("OVER BURN MARGIN: 1 burn(s) at 249 m/s")
    assert "No score here is a bound: each is two-body's, for one impulse." in lines


def test_a_ranking_run_names_the_forces_and_the_clock(
    project_root: Path, tmp_path: Path
) -> None:
    import yaml

    raw = yaml.safe_load((project_root / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"].update(horizon_s=6711.667, mission_time_origin="earliest")
    raw["score"] = {"kt": 0.003, "ct": 5369.3336, "kv": 0.0012, "cv": 2000.0, "time_unit": "s",
                    "delta_v_unit": "m/s", "cr_km": 300.0}
    config = tmp_path / "day.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "safe_delta_v_floor.py"),
         str(project_root / "examples" / "ranking" / "R1_NearPlane_Reachable.script"),
         "--config", str(config), "--workers", "1", "--arrivals", "4200:4500:200",
         "--burns", "0:100:100", "--output", str(tmp_path / "floor.json")],
        capture_output=True, text=True, check=True, timeout=300,
    )
    out = completed.stdout
    assert "T_max = 6711.6670 s on the solver clock, as every time below" in out
    assert "mission clock: origin earliest;" in out
    assert "burn cap = 250 m/s (§4 note 1(a), disqualifies), ours 249 m/s" in out
    assert "Two-body leaves out these forces the script flies (§7)" in out
    assert "  Analytical seed ignores drag:" in out
    rows = {line.split()[0]: line for line in out.splitlines() if line[:10].strip() in
            {"4200", "4400"}}
    assert rows["4200"].endswith("over burn cap") and "over burn" not in rows["4400"]
    assert "score >=" not in out
    marked = {row["arrival_s"]: row["over_burn_cap"]
              for row in json.loads((tmp_path / "floor.json").read_text(encoding="utf-8"))}
    assert marked == {4200.0: True, 4400.0: False}
