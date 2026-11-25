from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from conftest import quick_config

from orbitwin.config import GmatConfig, OrbitWinConfig, PolicyConfig, ScoreConfig, validate_config
from orbitwin.constraints import stop_depth_mismatch
from orbitwin.errors import ConfigurationError
from orbitwin.gmat_parser import drag_weather_sources, parse_gmat_script, stop_depth_advice
from orbitwin.models import Burn, Candidate
from orbitwin.rank import rank_report
from orbitwin.trajectory import state_with_burns_at
from orbitwin.verify import measuring_candidate

#: R4 moved 664.7 m on the CSSI file at +30 % F10.7/Ap (R2026a, 2026-11-02).
R4_CSSI_SHIFT_KM = 0.665


def test_the_default_stop_depth_covers_the_measured_shift_and_fits_the_sphere() -> None:
    policy = PolicyConfig()
    assert policy.stop_depth_km >= 2.0 * R4_CSSI_SHIFT_KM
    assert policy.stop_depth_km <= 0.5 * policy.intercept_radius_km
    assert GmatConfig().correct_aim_km <= policy.intercept_radius_km - policy.stop_depth_km


def _with(policy: dict | None = None, gmat: dict | None = None) -> OrbitWinConfig:
    base = OrbitWinConfig()
    return dataclasses.replace(
        base,
        policy=dataclasses.replace(base.policy, **(policy or {})),
        gmat=dataclasses.replace(base.gmat, **(gmat or {})),
    )


@pytest.mark.parametrize("depth_km", [0.0, -0.1, 2.6, math.nan, math.inf])
def test_a_stop_depth_outside_half_the_radius_is_refused(depth_km: float) -> None:
    with pytest.raises(ConfigurationError, match="stop_depth_km"):
        validate_config(_with({"stop_depth_km": depth_km}))


@pytest.mark.parametrize("depth_km", [0.05, 1.5, 2.5])
def test_a_stop_depth_inside_half_the_radius_is_accepted(depth_km: float) -> None:
    validate_config(_with({"stop_depth_km": depth_km}))


def test_the_stop_depth_loads_from_yaml(tmp_path: Path) -> None:
    from orbitwin.config import load_config

    path = tmp_path / "c.yaml"
    path.write_text("policy:\n  stop_depth_km: 0.05\n", encoding="utf-8")
    assert load_config(path).policy.stop_depth_km == 0.05
    path.write_text("policy:\n  stop_depth_km: '1.5'\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="stop_depth_km"):
        load_config(path)


@pytest.mark.parametrize(
    ("aim_km", "depth_km", "ok"),
    [(3.5, 1.5, True), (3.6, 1.5, False), (4.0, 1.5, False), (4.9, 0.05, True)],
)
def test_the_correction_aim_leaves_room_for_the_stop_depth(
    aim_km: float, depth_km: float, ok: bool
) -> None:
    config = _with({"stop_depth_km": depth_km}, {"correct_aim_km": aim_km})
    if ok:
        validate_config(config)
    else:
        with pytest.raises(ConfigurationError, match="correct_aim_km"):
            validate_config(config)


def test_drag_weather_sources_names_each_drag_model_and_gmat_defaults() -> None:
    fields = {
        "FileFM.Drag.AtmosphereModel": "NRLMSISE00",
        "FileFM.Drag.HistoricWeatherSource": "'CSSISpaceWeatherFile'",
        "FileFM.Drag.PredictedWeatherSource": "'CSSISpaceWeatherFile'",
        "SchattenFM.Drag.AtmosphereModel": "MSISE90",
        "SchattenFM.Drag.PredictedWeatherSource": "'SchattenFile'",
        "FluxFM.Drag.AtmosphereModel": "JacchiaRoberts",
        "OldFM.Drag": "JacchiaRoberts",
        "OffFM.Drag.AtmosphereModel": "None",
        "OffFM.Drag.HistoricWeatherSource": "'CSSISpaceWeatherFile'",
        "PlainFM.CentralBody": "Earth",
    }
    sources = {source["force_model"]: source for source in drag_weather_sources(fields)}
    assert set(sources) == {"FileFM", "SchattenFM", "FluxFM", "OldFM"}
    assert sources["FileFM"]["reads_weather_file"]
    assert sources["SchattenFM"]["historic_weather_source"] == "ConstantFluxAndGeoMag"
    assert sources["SchattenFM"]["reads_weather_file"]
    assert sources["FluxFM"]["predicted_weather_source"] == "ConstantFluxAndGeoMag"
    assert not sources["FluxFM"]["reads_weather_file"]
    assert sources["OldFM"]["atmosphere_model"] == "JacchiaRoberts"


def _r4(project_root: Path, tmp_path: Path, weather: str | None) -> tuple[Path, Path]:
    """R4 with its drag on another weather source, and the rehearsal profile with a T_max."""
    text = (project_root / "examples" / "ranking" / "R4_LowPerigee_Drag.script").read_text(
        encoding="utf-8"
    )
    if weather is not None:
        text = text.replace("'ConstantFluxAndGeoMag'", f"'{weather}'")
    script = tmp_path / "r4.script"
    script.write_text(text, encoding="utf-8")
    profile = tmp_path / "p.yaml"
    rehearsal = (project_root / "config" / "ranking-rehearsal.yaml").read_text(encoding="utf-8")
    profile.write_text(rehearsal.replace("  horizon_s: null", "  horizon_s: 6000.0"), encoding="utf-8")
    return script, profile


@pytest.mark.parametrize(
    ("weather", "recommended", "said"),
    [
        (None, None, "historic ConstantFluxAndGeoMag, predicted ConstantFluxAndGeoMag"),
        ("CSSISpaceWeatherFile", 1.5, "reads a space-weather file"),
    ],
)
def test_inspect_reports_the_weather_source_and_never_sets_the_depth(
    project_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    weather: str | None, recommended: float | None, said: str,
) -> None:
    from orbitwin.cli import main
    from orbitwin.config import load_config

    script, profile = _r4(project_root, tmp_path, weather)
    scenario, _ = parse_gmat_script(script, load_config(profile))
    advice = stop_depth_advice(scenario, 0.05)
    assert advice["recommended_stop_depth_km"] == recommended
    assert advice["configured_stop_depth_km"] == 0.05
    assert said in advice["advice"]
    assert ("50 m covered the EOP/UT1" in advice["advice"]) is (recommended is None)
    assert main(["inspect", str(script), "--config", str(profile)]) == 0
    captured = capsys.readouterr()
    shown = json.loads(captured.out)["stop_depth_advice"]
    assert shown["drag_weather_sources"][0]["reads_weather_file"] is (recommended is not None)
    assert said in captured.err
    assert shown["configured_stop_depth_km"] == load_config(profile).policy.stop_depth_km


def test_inspect_says_when_nothing_carries_drag(
    workshop_script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from orbitwin.cli import main

    assert main(["inspect", str(workshop_script), "--rules", "preliminary"]) == 0
    captured = capsys.readouterr()
    assert "No ForceModel carries drag. 50 m covered the EOP/UT1" in captured.err
    assert json.loads(captured.out)["stop_depth_advice"]["drag_weather_sources"] == []


def _hit(event: dict | None, successful: bool = True, branch: str = "b") -> Candidate:
    burn = Burn(time_s=100.0, dv_inertial_km_s=(0.1, 0.0, 0.0), dv_script_km_s=(0.1, 0.0, 0.0))
    metadata: dict = {"event_refined": True}
    if event is not None:
        metadata["gmat_event"] = event
    return Candidate(
        burns=(burn,), intercept_time_s=3000.0, minimum_distance_km=1.0,
        distance_at_completion_km=5.0, successful=successful,
        analytical_endpoint_error_km=None, branch=branch, metadata=metadata,
    )


def test_a_hit_verified_at_another_stop_depth_is_named() -> None:
    config = OrbitWinConfig()
    assert stop_depth_mismatch(_hit({"stop_time_s": 1.0, "stop_depth_setting_km": 1.5}), config) is None
    assert stop_depth_mismatch(_hit(None), config) is None
    assert stop_depth_mismatch(_hit({"stop_time_s": None}), config) is None
    assert stop_depth_mismatch(_hit({"stop_time_s": 1.0}, successful=False), config) is None
    old = stop_depth_mismatch(_hit({"stop_time_s": 1.0, "stop_depth_km": 0.0500005}), config)
    assert old is not None and "before the stop depth was recorded, 50 m inside" in old
    other = stop_depth_mismatch(
        _hit({"stop_time_s": 1.0, "stop_depth_km": 0.6, "stop_depth_setting_km": 0.6}), config
    )
    assert other is not None and "policy.stop_depth_km = 0.6 km" in other


def test_rank_warns_about_hits_verified_at_another_stop_depth(workshop_script: Path) -> None:
    config = OrbitWinConfig(score=ScoreConfig(kt=0.001, ct=20000.0, kv=0.004, cv=800.0))
    scenario, _ = parse_gmat_script(workshop_script, config)
    deep = _hit({"stop_time_s": 1.0, "stop_depth_km": 1.5, "stop_depth_setting_km": 1.5})
    shallow = _hit({"stop_time_s": 1.0, "stop_depth_km": 0.05}, branch="shallow")
    shallow.intercept_time_s = 2000.0
    report = rank_report(scenario, [deep], config)
    assert not any("another depth" in warning for warning in report["warnings"])
    report = rank_report(scenario, [deep, shallow], config)
    named = [warning for warning in report["warnings"] if "another depth" in warning]
    assert len(named) == 1 and shallow.candidate_id in named[0]
    assert deep.candidate_id not in named[0]


def test_the_first_flight_ends_twice_radius_over_speed_past_the_predicted_entry(
    workshop_script: Path,
) -> None:
    from orbitwin.search import search_one_burn

    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    assert candidate.successful and candidate.intercept_time_s is not None
    entry_s = candidate.intercept_time_s
    pursuer = state_with_burns_at(
        scenario.pursuer.initial_state, candidate.burns, entry_s, scenario.mu_km3_s2
    )
    target = state_with_burns_at(scenario.target.initial_state, (), entry_s, scenario.mu_km3_s2)
    speed = float(np.linalg.norm(np.asarray(pursuer.v) - np.asarray(target.v)))
    flown = measuring_candidate(candidate, scenario, config)
    assert flown.intercept_time_s == pytest.approx(entry_s + 10.0 / speed, rel=1e-12)
    assert flown.intercept_time_s - entry_s >= config.policy.stop_depth_km / speed
    # The candidate itself is untouched.
    assert candidate.intercept_time_s == entry_s
    missed = dataclasses.replace(candidate, successful=False)
    assert measuring_candidate(missed, scenario, config) is missed


def test_the_first_flight_is_not_extended_after_an_epoch_alignment(workshop_script: Path) -> None:
    """Two-body states from each craft's own epoch are not on the solver clock."""
    from orbitwin.models import EpochAlignment
    from orbitwin.search import search_one_burn

    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    aligned = dataclasses.replace(
        scenario,
        alignment=EpochAlignment(
            early=scenario.target.name, late=scenario.pursuer.name, propagator="P",
            form="elapsed", offset_s=5400.0, text="", line_index=None,
        ),
    )
    assert measuring_candidate(candidate, aligned, config) is candidate
    assert measuring_candidate(candidate, scenario, config) is not candidate
