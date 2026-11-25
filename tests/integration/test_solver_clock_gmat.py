"""Solve and certify on stock GmatConsole with the craft 5400 s apart, either first."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from conftest import quick_config

from orbitwin.config import ScoreConfig
from orbitwin.io import read_candidates
from orbitwin.orbit import propagate_universal
from orbitwin.pipeline import solve_scenario
from orbitwin.report_parser import parse_numeric_report
from orbitwin.state_probe import load_scenario
from orbitwin.verify import certify_candidate

ROOT = Path(__file__).resolve().parents[2]
CROSSING = ROOT / "tests" / "fixtures" / "async" / "aligned_crossing.script"
GMAT_MU = 398600.4415

pytestmark = pytest.mark.gmat


def _config(origin: str):
    base = quick_config(local_refine=True)
    return dataclasses.replace(
        base,
        policy=dataclasses.replace(
            base.policy, mission_time_origin=origin, closest_approach_window="after_alignment"
        ),
        pursuer_name="SC_Human",
        target_name="SC_Alien",
        score=ScoreConfig(kt=0.003, ct=6000.0, kv=1.2, cv=2.0, delta_v_unit="km/s"),
        gmat=dataclasses.replace(base.gmat, verify_top=4, workers=4),
    )


def _pursuer_early(tmp_path: Path) -> Path:
    """The fixture's geometry with SC_Human first: SC_Human back 5400 s, SC_Alien on time."""
    text = CROSSING.read_text(encoding="utf-8")
    craft = {}
    for name in ("SC_Human", "SC_Alien"):
        values = [
            float(line.split("=")[1].strip(" ;"))
            for line in text.splitlines()
            if line.startswith(f"{name}.") and line.split(".")[1].split(" ")[0]
            in ("X", "Y", "Z", "VX", "VY", "VZ")
        ]
        craft[name] = values
    from orbitwin.models import StateVector

    human = StateVector(tuple(craft["SC_Human"][:3]), tuple(craft["SC_Human"][3:]))  # type: ignore[arg-type]
    alien = StateVector(tuple(craft["SC_Alien"][:3]), tuple(craft["SC_Alien"][3:]))  # type: ignore[arg-type]
    moved = {
        "SC_Human": propagate_universal(human, -5400.0, GMAT_MU),
        "SC_Alien": propagate_universal(alien, 5400.0, GMAT_MU),
    }
    lines = []
    for line in text.splitlines():
        head = line.split(" = ")[0]
        name, _, field = head.partition(".")
        if name in moved and field in ("X", "Y", "Z", "VX", "VY", "VZ"):
            state = moved[name]
            value = [*state.r_km, *state.v_km_s][("X", "Y", "Z", "VX", "VY", "VZ").index(field)]
            line = f"{head} = {float(value)!r};"
        lines.append(line)
    text = "\n".join(lines) + "\n"
    text = text.replace("SC_Human.Epoch = '01 Jan 2026 12:00:00.000';", "SC_Human.Epoch = 'X';")
    text = text.replace(
        "SC_Alien.Epoch = '01 Jan 2026 10:30:00.000';", "SC_Alien.Epoch = '01 Jan 2026 12:00:00.000';"
    )
    text = text.replace("SC_Human.Epoch = 'X';", "SC_Human.Epoch = '01 Jan 2026 10:30:00.000';")
    text = text.replace(
        "Propagate Prop(SC_Alien) {SC_Alien.UTCModJulian = SC_Human.UTCModJulian};",
        "Propagate Prop(SC_Human) {SC_Human.UTCModJulian = SC_Alien.UTCModJulian};",
    )
    path = tmp_path / "pursuer_early.script"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("origin", ["target", "pursuer"])
@pytest.mark.parametrize("early", ["target", "pursuer"])
def test_solve_and_certify_on_the_solver_clock(
    tmp_path: Path, gmat: tuple[str, str], early: str, origin: str
) -> None:
    console, startup = gmat
    script = CROSSING if early == "target" else _pursuer_early(tmp_path)
    config = _config(origin=origin)
    scenario, _, _ = load_scenario(script, config, gmat_console=console, startup_file=startup)
    alignment = scenario.alignment
    assert alignment is not None and alignment.offset_s == pytest.approx(5400.0, abs=1e-6)
    early_name = "SC_Alien" if early == "target" else "SC_Human"
    assert alignment.early == early_name
    # The early craft at t = 0 is GMAT's flight of the alignment; point mass agrees with two-body.
    probed = scenario.state_probe.craft_named(early_name).state  # type: ignore[union-attr]
    at_zero = scenario.target if early == "target" else scenario.pursuer
    two_body = propagate_universal(probed, 5400.0, GMAT_MU)
    assert float(np.linalg.norm(at_zero.initial_state.r - two_body.r)) < 1.0e-6
    # T_team from the origin's epoch: 5400 s before t = 0 only when it is the early one.
    expected = 5400.0 if early == origin else 0.0
    assert scenario.time_origin_offset_s == pytest.approx(expected, abs=1e-6)

    summary = solve_scenario(
        script, config, tmp_path / "solve", gmat_console=console, startup_file=startup
    )
    best = summary["best_candidate"]
    assert best is not None and best["successful"] and best["verified_by_gmat"]
    # The tracks cross 1500 s after t = 0: whatever the burns, inside it well before that.
    assert 0.0 < best["intercept_time_s"] < 1500.0
    candidate = next(
        item for item in read_candidates(tmp_path / "solve" / "candidates.json")
        if item.candidate_id == best["candidate_id"]
    )
    certificate = certify_candidate(
        script, candidate, config, console, startup, tmp_path / "certify"
    )
    assert certificate["frozen_rows"] == 540
    assert certificate["final_script_parity"]["replayed_score"] == pytest.approx(best["score"])
    assert certificate["candidate"]["metadata"]["time_origin_offset_s"] == pytest.approx(
        scenario.time_origin_offset_s
    )
    # The uploaded report's last epoch, less the origin's own, is T_team as certified.
    official = parse_numeric_report(tmp_path / "certify" / "submission" / "OfficialReport.txt")
    origin_name = "SC_Alien" if origin == "target" else "SC_Human"
    epoch = scenario.state_probe.craft_named(origin_name).epoch  # type: ignore[union-attr]
    last_s = (float(official.column("SC_Human.UTCModJulian")[-1]) - epoch.utc_mjd) * 86400.0
    assert last_s == pytest.approx(certificate["clock"]["t_team_s"], abs=1.0e-3)
