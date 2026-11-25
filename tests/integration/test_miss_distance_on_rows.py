"""A scored miss's distance is its closest report row, the one a validator reading rows sees."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from conftest import FAKE_STARTUP, capped_one_burn, ranking_copy, ranking_quick_config, stamped

from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Candidate, Scenario
from orbitwin.report_parser import parse_numeric_report
from orbitwin.verify import certify_candidate, verify_candidate_with_gmat

MISS_KM = 10.0


@pytest.fixture()
def workshop_script(workshop_script: Path, tmp_path: Path) -> Path:
    """The workshop with the ranking report name; certify refuses it without one."""
    return ranking_copy(workshop_script, tmp_path)


def _passing(workshop_script: Path, tmp_path: Path, step_s: float,
             closest_s: float = 3030.0) -> Path:
    """The double passes MISS_KM off the target at 1 km/s at `closest_s`, a row every `step_s`."""
    script = tmp_path / f"pass-{step_s:g}.script"
    script.write_text(
        f"% FAKE_GMAT_FLYBY {closest_s} {step_s}\n% FAKE_GMAT_MISS_KM {MISS_KM}\n"
        + workshop_script.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return script


def _row_minimum_km(report: Path, scenario: Scenario) -> float:
    rows = parse_numeric_report(report)
    p, t = scenario.pursuer.name, scenario.target.name
    relative = np.column_stack([
        rows.column(f"{p}.EarthMJ2000Eq.{axis}") - rows.column(f"{t}.EarthMJ2000Eq.{axis}")
        for axis in "XYZ"
    ])
    return float(np.linalg.norm(relative, axis=1).min())


def _verify(script: Path, out: Path) -> tuple[Candidate, Scenario, dict]:
    config = ranking_quick_config()
    scenario, document = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    verified, _, evidence = verify_candidate_with_gmat(
        script, scenario, document, candidate, config,
        Path(__file__).resolve().parents[2] / "scripts" / "fake_gmat_console.py",
        FAKE_STARTUP, out,
    )
    return verified, scenario, evidence


def test_a_miss_on_sparse_rows_scores_its_closest_row(
    tmp_path: Path, workshop_script: Path
) -> None:
    """Rows 60 s apart straddle the pass 30 s off: the closest row is 31.6 km, the
    Hermite minimum between them 10 km, worth 1.7 points more than the rows show at
    C_r 100 km (9.2 under the term before 2026/11/04)."""
    verified, scenario, evidence = _verify(_passing(workshop_script, tmp_path, 60.0),
                                           tmp_path / "v")
    assert not verified.successful and verified.metadata["scored_miss"] is True
    rows_km = _row_minimum_km(Path(evidence["diagnostic_report"]), scenario)
    assert rows_km == pytest.approx(math.hypot(MISS_KM, 30.0), abs=1e-6)
    event = verified.metadata["gmat_event"]
    assert event["interpolated_minimum_distance_km"] == pytest.approx(MISS_KM, abs=1e-6)
    assert verified.minimum_distance_km == pytest.approx(rows_km, abs=1e-9)
    assert event["minimum_distance_km"] == pytest.approx(rows_km, abs=1e-9)
    # The epoch of that row, 30 s from the pass at 3030 s.
    assert event["minimum_distance_time_s"] in (pytest.approx(3000.0), pytest.approx(3060.0))
    assert event["interpolated_minimum_distance_time_s"] == pytest.approx(3030.0)
    assert verified.score_components["distance_score"] == pytest.approx(
        100.0 / (1.0 + math.exp(((rows_km - 5.0) / 100.0) ** 2))
    )


def test_a_miss_on_dense_rows_scores_nearly_as_before(
    tmp_path: Path, workshop_script: Path
) -> None:
    """Rows 1 s apart, the pass half a second from either: 12.5 m, not 30 km."""
    verified, _, _ = _verify(_passing(workshop_script, tmp_path, 1.0, 3030.5), tmp_path / "v")
    assert verified.minimum_distance_km == pytest.approx(math.hypot(MISS_KM, 0.5), abs=1e-6)
    assert verified.metadata["gmat_event"]["interpolated_minimum_distance_km"] == (
        pytest.approx(MISS_KM, abs=1e-6)
    )


def test_a_hit_keeps_its_interpolated_closest_approach(
    tmp_path: Path, workshop_script: Path
) -> None:
    """Flown on to T_max, a pass through the target falls between rows 30 s from it.

    A hit scores 50 whatever; its distance is the tie-break, measured as before.
    """
    config = ranking_quick_config()
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, mission_end_policy="horizon")
    )
    script = tmp_path / "hit.script"
    script.write_text(
        "% FAKE_GMAT_FLYBY 3030 60\n" + workshop_script.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    scenario, document = parse_gmat_script(script, config)
    verified, _, evidence = verify_candidate_with_gmat(
        script, scenario, document, capped_one_burn(scenario, config), config,
        Path(__file__).resolve().parents[2] / "scripts" / "fake_gmat_console.py",
        FAKE_STARTUP, tmp_path / "v",
    )
    assert verified.successful
    rows_km = _row_minimum_km(Path(evidence["diagnostic_report"]), scenario)
    assert rows_km == pytest.approx(30.0, abs=1e-6)
    assert verified.minimum_distance_km == pytest.approx(0.0, abs=1e-6)
    assert verified.metadata["gmat_event"]["interpolated_minimum_distance_km"] == (
        verified.minimum_distance_km
    )


def test_certify_replays_the_row_distance_it_was_verified_on(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    config = ranking_quick_config()
    script = _passing(workshop_script, tmp_path, 60.0)
    scenario, _ = parse_gmat_script(script, config)
    candidate = stamped(capped_one_burn(scenario, config), script, config)
    # A planned hit GMAT flies as a miss, certified as one on purpose.
    evidence = certify_candidate(script, candidate, config, fake_gmat, fake_startup,
                                 tmp_path / "c", runs=1, accept_gmat_loss=True)
    parity = evidence["final_script_parity"]
    rows_km = math.hypot(MISS_KM, 30.0)
    assert parity["minimum_distance_km"] == pytest.approx(rows_km, abs=1e-6)
    assert parity["interpolated_minimum_distance_km"] == pytest.approx(MISS_KM, abs=1e-6)
    certified = json.loads(
        (tmp_path / "c" / "submission" / "certification.json").read_text(encoding="utf-8")
    )
    expected = evidence["candidate"]["score"]
    assert certified["final_script_parity"]["replayed_score"] == pytest.approx(expected)
    assert evidence["candidate"]["score_components"]["distance_score"] == pytest.approx(
        100.0 / (1.0 + math.exp(((rows_km - 5.0) / 100.0) ** 2))
    )


def test_a_miss_without_its_row_distance_is_refused() -> None:
    from orbitwin.event_detection import EventResult
    from orbitwin.verify import _closest_approach

    event = EventResult(None, 10.0, 3030.0, 400.0)
    with pytest.raises(RuntimeError, match="no report rows"):
        _closest_approach(event, False)
    assert _closest_approach(event, True) == (10.0, 3030.0)
