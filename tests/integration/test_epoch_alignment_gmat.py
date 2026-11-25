"""The epoch alignment on stock GmatConsole, on the tutorial's own shape."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from orbitwin.config import OrbitWinConfig, PolicyConfig
from orbitwin.ephemeris import target_ephemeris_from_gmat
from orbitwin.errors import PolicyError
from orbitwin.frames import script_components_to_inertial
from orbitwin.models import Burn, Candidate, as_vector3
from orbitwin.report_parser import parse_numeric_report
from orbitwin.state_probe import load_scenario
from orbitwin.verify import verify_candidate_with_gmat

ROOT = Path(__file__).resolve().parents[2]
ASYNC_FIXTURE = ROOT / "tests" / "fixtures" / "async" / "insertion_and_async_start.script"

pytestmark = pytest.mark.gmat


def _config(**policy: object) -> OrbitWinConfig:
    return OrbitWinConfig(
        policy=dataclasses.replace(PolicyConfig(), **policy),  # type: ignore[arg-type]
        pursuer_name="SC_Human",
        target_name="SC_Alien",
    )


def test_the_tutorials_alignment_is_read_at_the_probed_offset(gmat: tuple[str, str]) -> None:
    console, startup = gmat
    config = _config(mission_time_origin="pursuer", closest_approach_window="after_alignment")
    scenario, _, _ = load_scenario(ASYNC_FIXTURE, config, gmat_console=console, startup_file=startup)
    probe = scenario.state_probe
    assert probe is not None and probe.aligned is not None
    # SC_Human waits at its insertion; SC_Alien is where GMAT flew it in 15 h.
    assert scenario.pursuer.initial_state == probe.craft_named("SC_Human").state  # type: ignore[union-attr]
    assert scenario.target.initial_state != probe.craft_named("SC_Alien").state  # type: ignore[union-attr]
    assert abs(float(np.linalg.norm(scenario.target.initial_state.r)) - 7000.0) < 1.0e-3
    alignment = scenario.alignment
    assert alignment is not None and alignment.form == "epoch-equality"
    assert (alignment.early, alignment.late, alignment.propagator) == (
        "SC_Alien", "SC_Human", "DefaultProp"
    )
    assert alignment.offset_s == pytest.approx(54000.0, abs=1.0e-6)
    assert [segment.duration_s for segment in scenario.propagate_segments] == [3000.0]


def test_a_burn_at_t_zero_after_a_zero_length_alignment_flies(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    """One start, the tutorial's alignment line, and our burn right after it."""
    console, startup = gmat
    script = tmp_path / "aligned.script"
    script.write_text(
        ASYNC_FIXTURE.read_text(encoding="utf-8").replace(
            "SC_Alien.Epoch = '01 Jan 2000 00:00:00.000';",
            "SC_Alien.Epoch = '01 Jan 2000 15:00:00.000';",
        ),
        encoding="utf-8",
    )
    config = _config()
    scenario, document, _ = load_scenario(script, config, gmat_console=console, startup_file=startup)
    assert scenario.alignment is not None and scenario.alignment.offset_s == 0.0
    impulse = script_components_to_inertial(
        np.array([0.001, 0.0, 0.0]), scenario.pursuer.initial_state, "Local", "Earth", "VNB"
    )
    prograde = Burn(0.0, as_vector3(impulse), (0.001, 0.0, 0.0), 0, ("Local", "Earth", "VNB"))
    candidate = Candidate(
        burns=(prograde,), intercept_time_s=None, minimum_distance_km=None,
        distance_at_completion_km=None, successful=False, analytical_endpoint_error_km=None,
        branch="test",
    )
    # A miss, which the preliminary rules do not verify; GMAT flew it to T_max.
    with pytest.raises(PolicyError, match="not a verified successful interception"):
        verify_candidate_with_gmat(
            script, scenario, document, candidate, config, console, startup,
            tmp_path / "verify", timeout_s=300,
        )
    sibling = (tmp_path / "verify" / "diagnostic_sibling.script").read_text()
    mission = sibling.split("BeginMissionSequence;\n")[1].splitlines()
    assert mission[0] == scenario.alignment.text.strip()
    # The burn report brackets the Maneuver; verify checked the t = 0 impulse from it.
    report = mission[1]
    assert report.startswith("Report OrbitWinBurnCheck SC_Human.A1ModJulian ")
    assert mission[1:4] == [report, "Maneuver ImpulsiveBurn1(SC_Human);", report]
    [path] = (tmp_path / "verify" / "diagnostic-run").rglob(
        config.gmat.diagnostic_report_name
    )
    report = parse_numeric_report(path)
    times = report.column("SC_Human.ElapsedSecs")
    speed = np.linalg.norm(
        [report.column(f"SC_Human.EarthMJ2000Eq.V{axis}")[:2] for axis in "XYZ"], axis=0
    )
    # Measured on R2026a: the one row at t = 0 is the pre-burn state, written by the alignment.
    assert times[0] == 0.0 and times[1] > 0.0
    assert speed[0] == pytest.approx(float(np.linalg.norm(scenario.pursuer.initial_state.v)))
    assert speed[1] - speed[0] == pytest.approx(0.001, abs=2.0e-5)
    assert times[-1] == pytest.approx(scenario.max_time_s, abs=1.0e-6)


def test_the_coast_run_replays_the_alignment(tmp_path: Path, gmat: tuple[str, str]) -> None:
    console, startup = gmat
    script = tmp_path / "aligned.script"
    script.write_text(
        ASYNC_FIXTURE.read_text(encoding="utf-8").replace(
            "SC_Alien.Epoch = '01 Jan 2000 00:00:00.000';",
            "SC_Alien.Epoch = '01 Jan 2000 15:00:00.000';",
        ),
        encoding="utf-8",
    )
    config = _config()
    scenario, document, _ = load_scenario(script, config, gmat_console=console, startup_file=startup)
    ephemeris = target_ephemeris_from_gmat(
        scenario, document, config, console, startup, tmp_path / "coast"
    )
    assert ephemeris.times_s[0] == 0.0
    assert ephemeris.times_s[-1] == pytest.approx(scenario.max_time_s, abs=1.0e-6)
    assert tuple(ephemeris.positions_km[0]) == scenario.target.initial_state.r_km
    coast = (tmp_path / "coast" / "coast_diagnostic.script").read_text()
    assert coast.split("BeginMissionSequence;\n")[1].startswith(scenario.alignment.text)  # type: ignore[union-attr]
