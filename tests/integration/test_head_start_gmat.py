"""Head-start burns on real GMAT R2026a: R9, rendered, flown and read back.

One verification flies a burn 1000 s into R9's 3600 s head start, one at
policy.min_propagate_duration_s before SC_Alien's epoch, and one after the alignment.
The floor burn leaves the alignment about 1 ms to fly, and GMAT then writes two
rows at t = 0 (output/review/pr202-physics/logs/gmat-floor-edge.txt): harmless,
pinned here.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from orbitwin.burn_rows import BURN_EPOCH_TOLERANCE_S
from orbitwin.config import ScoreConfig, load_config
from orbitwin.frames import script_components_to_inertial
from orbitwin.models import Burn, Candidate
from orbitwin.report_parser import parse_numeric_report, report_clock
from orbitwin.state_probe import load_scenario
from orbitwin.trajectory import fly_pursuer
from orbitwin.verify import verify_candidate_with_gmat

pytestmark = pytest.mark.gmat

ROOT = Path(__file__).resolve().parents[2]
R9 = ROOT / "examples" / "ranking" / "R9_PursuerFirst_LongHeadStart.script"
#: policy.min_propagate_duration_s: the last head-start burn this far before SC_Alien's epoch.
FLOOR_S = 1.0e-3


def _config():
    config = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    config = dataclasses.replace(config, policy=dataclasses.replace(config.policy, team_number=54))
    return dataclasses.replace(
        config,
        policy=dataclasses.replace(
            config.policy, horizon_s=8511.667, mission_time_origin="earliest",
            closest_approach_window="after_alignment", mission_boundary_km=17000.0,
        ),
        score=ScoreConfig(kt=0.003, ct=6809.3336, kv=0.0012, cv=2000.0, delta_v_unit="m/s",
                          cr_km=300.0),
    )


def test_head_start_burns_fly_in_gmat_as_rendered(tmp_path: Path, gmat: tuple[str, str]) -> None:
    console, startup = gmat
    config = _config()
    scenario, document, _ = load_scenario(R9, config, gmat_console=console, startup_file=startup)
    assert scenario.head_start_s == pytest.approx(3600.0, abs=1e-6)
    frame = scenario.synthesized_burn_frame
    plan = [(-1000.0, (0.0, 0.01, 0.0)), (-FLOOR_S, (0.005, 0.0, 0.0)), (200.0, (0.0, 0.0, 0.01))]
    burns: list[Burn] = []
    schedule: list[tuple[float, np.ndarray]] = []
    for slot, (time_s, components) in enumerate(plan):
        pre = fly_pursuer(scenario, [*schedule, (time_s, np.zeros(3))], time_s)[0][-1]
        inertial = script_components_to_inertial(np.array(components), pre, *frame)
        schedule.append((time_s, inertial))
        burns.append(Burn(time_s, tuple(inertial), components, slot, frame))  # type: ignore[arg-type]
    candidate = Candidate(
        burns=tuple(burns), intercept_time_s=None, minimum_distance_km=None,
        distance_at_completion_km=None, successful=False, analytical_endpoint_error_km=None,
        branch="ranking-search/3/head-start",
    )
    candidate.metadata.update(arrival_time_s=scenario.max_time_s)
    verified, run, evidence = verify_candidate_with_gmat(
        R9, scenario, document, candidate, config, console, startup, tmp_path / "verify"
    )
    assert run.success and not evidence["gmat_log_errors"]
    # verify checked each impulse GMAT flew against the script's Element1-3, the
    # head-start two in the alignment's rows included, and the floor on every row.
    assert verified.metadata["minimum_pursuer_radius_km"] > 6378.1363 + 100.0
    report = parse_numeric_report(evidence["diagnostic_report"])
    clock = report_clock(report, "SC_Human", "SC_Alien", scenario.alignment)
    times = clock.times_s
    assert clock.frozen_rows == evidence["frozen_rows"] > 0
    # Both head-start burns have a row at their epoch, before t = 0.
    for time_s in (-1000.0, -FLOOR_S):
        assert np.any(np.abs(times[: clock.frozen_rows] - time_s) <= BURN_EPOCH_TOLERANCE_S)
    # The floor burn leaves the alignment 1 ms, less the lead's A1 quantisation
    # (3e-7 s): GMAT writes a row there that is already one epoch, then t = 0 again.
    at_zero = np.flatnonzero(np.abs(times) <= 1.0e-6)
    assert len(at_zero) == 2 and np.all(np.diff(times) >= -1.0e-9)
    assert times[-1] == pytest.approx(scenario.max_time_s, abs=1e-3)

    # Step 7's hand form binds the official report GMAT wrote to that script: the
    # head-start jumps in the alignment rows, the duplicate t = 0 row, the end.
    import yaml

    raw = yaml.safe_load((ROOT / "config" / "ranking-rehearsal.yaml").read_text("utf-8"))
    raw["policy"].update(horizon_s=8511.667, mission_time_origin="earliest",
                         closest_approach_window="after_alignment", mission_boundary_km=17000.0)
    raw["score"] = {"kt": 0.003, "ct": 6809.3336, "kv": 0.0012, "cv": 2000.0,
                    "time_unit": "s", "delta_v_unit": "m/s", "cr_km": 300.0}
    official = run.reports["ODC_2026_Team54.txt"]
    script = tmp_path / "verify" / "verification_candidate.script"
    for setting, code in ((True, 0), (False, 2)):
        raw["policy"]["head_start_burns"] = setting
        day = tmp_path / f"day-{setting}.yaml"
        day.write_text(yaml.safe_dump(raw), encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_report_rows.py"), official,
             "--config", str(day), "--script", str(script), "--verification",
             str(tmp_path / "verify" / "verification.json")],
            capture_output=True, text=True, check=False,
        )
        assert done.returncode == code, done.stdout[-2000:] + done.stderr[-2000:]
        if code == 2:
            assert "turns policy.head_start_burns off" in done.stderr
