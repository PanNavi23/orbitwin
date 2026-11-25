"""Verification reads the step after a burn with the post-burn state (#106).

The fake console writes one row at each burn, as R2026a does, with the
pre-burn velocity; the track after it is the post-burn one. Read without
the rebuilt row, Hermite bends the step after the burn.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from conftest import capped_one_burn, ranking_copy, ranking_quick_config

from orbitwin import verify as verify_module
from orbitwin.config import OrbitWinConfig
from orbitwin.errors import GmatRunError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Candidate
from orbitwin.verify import certify_candidate

MISS_KM = 6.0
ROW_STEP_S = 7.0


@pytest.fixture()
def workshop_script(workshop_script: Path, tmp_path: Path) -> Path:
    """The workshop with the ranking report name; certify refuses it without one."""
    return ranking_copy(workshop_script, tmp_path)


def _miss_after_the_burn(workshop: Path, tmp_path: Path) -> tuple[Path, OrbitWinConfig, Candidate]:
    """A 6 km miss a third of the way into the report step after the last burn."""
    config = ranking_quick_config()
    scenario, _ = parse_gmat_script(workshop, config)
    candidate = capped_one_burn(scenario, config, over_km_s=0.8 * config.policy.dv_safe_limit_km_s)
    last_s = max(burn.time_s for burn in candidate.burns)
    next_row_s = ROW_STEP_S * math.ceil(last_s / ROW_STEP_S + 1.0e-9)
    closest_s = last_s + (next_row_s - last_s) / 3.0
    script = tmp_path / "miss.script"
    script.write_text(
        f"% FAKE_GMAT_FLYBY {closest_s!r} {ROW_STEP_S}\n% FAKE_GMAT_MISS_KM {MISS_KM}\n"
        + workshop.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return script, config, candidate


def test_a_miss_in_the_step_after_a_burn_is_certified_at_its_true_distance(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script, config, candidate = _miss_after_the_burn(workshop_script, tmp_path)
    # The workshop's hit, flown as a miss: certified as one only on purpose.
    evidence = certify_candidate(
        script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1,
        accept_gmat_loss=True,
    )
    # Interpolated across the step after the burn, on the post-burn row.
    flown = evidence["candidate"]["metadata"]["gmat_event"]
    parity = evidence["final_script_parity"]
    assert flown["interpolated_minimum_distance_km"] == pytest.approx(MISS_KM, abs=1.0e-6)
    assert parity["interpolated_minimum_distance_km"] == pytest.approx(MISS_KM, abs=1.0e-6)
    # Scored on the closest row, the burn's: a third of the 7 s step at 1 km/s off.
    scored = evidence["candidate"]["minimum_distance_km"]
    assert parity["minimum_distance_km"] == scored
    assert MISS_KM < scored <= math.hypot(MISS_KM, ROW_STEP_S / 3.0) + 1.0e-6


@pytest.mark.parametrize("dropped_at", ["flight", "replay"])
def test_reading_either_run_without_the_maneuvers_is_caught(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch, dropped_at: str,
) -> None:
    """Dropping `maneuvers=` at one call site: 6.0092 km against 6.0, and refused."""
    script, config, candidate = _miss_after_the_burn(workshop_script, tmp_path)
    real = verify_module.evaluate_diagnostic_report

    def reading(path, *args, maneuvers=(), **kwargs):
        replay = "final-parity" in str(path)
        if (dropped_at == "replay") == replay:
            maneuvers = ()
        return real(path, *args, maneuvers=maneuvers, **kwargs)

    monkeypatch.setattr(verify_module, "evaluate_diagnostic_report", reading)
    with pytest.raises(GmatRunError, match="but replaying the exact final script gives"):
        certify_candidate(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c",
                          runs=1, accept_gmat_loss=True)
