"""policy.minimum_flown_altitude_km: off by default, then a refusal in verify and certify.

The ranking search holds a 100 km floor in two-body. Under the section 7
drag model GMAT flew R4's margin-0 winners to 95.4 km, which the central-body
line alone would have accepted.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from conftest import stamped

from orbitwin.config import OrbitWinConfig, validate_config
from orbitwin.constraints import below_minimum_flown_altitude
from orbitwin.errors import ConfigurationError
from orbitwin.models import Candidate


def _flown(radius_km: float | None) -> Candidate:
    metadata = {} if radius_km is None else {"minimum_pursuer_radius_km": radius_km}
    return Candidate(
        burns=(), intercept_time_s=None, minimum_distance_km=20.0,
        distance_at_completion_km=900.0, successful=False,
        analytical_endpoint_error_km=None, branch="test", metadata=metadata,
    )


def _with(altitude_km: object) -> OrbitWinConfig:
    base = OrbitWinConfig()
    return dataclasses.replace(
        base, policy=dataclasses.replace(base.policy, minimum_flown_altitude_km=altitude_km)
    )


def test_it_is_off_by_default_and_reads_the_measured_radius() -> None:
    surface = OrbitWinConfig().policy.central_body_radius_km
    low = _flown(surface + 95.4)
    assert OrbitWinConfig().policy.minimum_flown_altitude_km is None
    assert below_minimum_flown_altitude(low, OrbitWinConfig()) is False
    assert below_minimum_flown_altitude(low, _with(100.0)) is True
    assert below_minimum_flown_altitude(_flown(surface + 104.7), _with(100.0)) is False
    assert below_minimum_flown_altitude(_flown(None), _with(100.0)) is False


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), "100", True])
def test_a_bad_value_is_refused(value: object) -> None:
    with pytest.raises(ConfigurationError, match="minimum_flown_altitude_km"):
        validate_config(_with(value))


@pytest.mark.parametrize("floor_km", [None, 100.0])
def test_solve_and_rank_agree_with_certify(
    tmp_path: Path, workshop_script: Path, monkeypatch, floor_km: float | None
) -> None:
    """R4's margin-0 best flew to 95.37 km: withheld by solve, blocked by rank,
    refused by certify (below) when the floor is 100, and accepted by all three
    when it is unset."""
    from orbitwin import pipeline as pipeline_module
    from orbitwin.config import SearchConfig
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.models import Burn
    from orbitwin.pipeline import solve_scenario
    from orbitwin.rank import submission_blockers

    base = OrbitWinConfig(
        search=SearchConfig(
            one_burn_samples=0, one_burn_grid_burns=1, one_burn_grid_arrivals=1,
            max_revolutions=0, two_burn_enabled=False, event_refine_top=2,
            keep_candidates=8, workers=1,
        )
    )
    config = dataclasses.replace(
        base, policy=dataclasses.replace(base.policy, minimum_flown_altitude_km=floor_km)
    )
    surface = config.policy.central_body_radius_km
    low = Candidate(
        burns=(Burn(1000.0, (0.2, 0.0, 0.0), (0.2, 0.0, 0.0), 0),),
        intercept_time_s=4000.0, minimum_distance_km=0.2, distance_at_completion_km=5.0,
        successful=True, analytical_endpoint_error_km=0.1, branch="low",
        metadata={
            "arrival_time_s": 4000.0, "event_refined": True,
            "minimum_pursuer_radius_km": surface + 95.37,
        },
    )
    monkeypatch.setattr(pipeline_module, "zero_burn_candidate", lambda *_: low)
    monkeypatch.setattr(pipeline_module, "search_one_burn", lambda *_: [])
    monkeypatch.setattr(pipeline_module, "search_two_burn_global", lambda *_: [])
    monkeypatch.setattr(pipeline_module, "slot_assignment_variants", lambda *_: [])
    summary = solve_scenario(workshop_script, config, tmp_path / "solve")
    scenario, _ = parse_gmat_script(workshop_script, config)
    blockers = submission_blockers(low, scenario, config)
    if floor_km is None:
        assert summary["best_candidate"] is not None
        assert not blockers
    else:
        assert summary["best_candidate"] is None
        (warning,) = summary["warnings"]
        assert "95.37 km altitude" in warning and "minimum_flown_altitude_km" in warning
        assert any("95.370 km altitude" in blocker for blocker in blockers)


def test_the_ranking_profiles_set_it_and_the_rest_do_not(project_root: Path) -> None:
    from orbitwin.config import load_config

    for name in ("ranking.yaml", "ranking-rehearsal.yaml"):
        assert load_config(project_root / "config" / name).policy.minimum_flown_altitude_km == 100
    for path in ("history/preliminary/odc2026.yaml", "standard.yaml"):
        assert load_config(project_root / "config" / path).policy.minimum_flown_altitude_km is None


def test_verify_refuses_with_its_own_error(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """The double flies a few hundred km up; a 5000 km floor stands in for a low flight."""
    from conftest import quick_config

    from orbitwin.errors import AltitudeFloorError
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.search import search_one_burn
    from orbitwin.verify import verify_candidate_with_gmat

    config = quick_config()
    scenario, document = parse_gmat_script(workshop_script, config)
    high = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, minimum_flown_altitude_km=5000.0)
    )
    with pytest.raises(AltitudeFloorError, match="minimum_flown_altitude_km"):
        verify_candidate_with_gmat(
            workshop_script, scenario, document, search_one_burn(scenario, config)[0], high,
            fake_gmat, fake_startup, tmp_path / "verify",
        )


@pytest.mark.parametrize(
    "floor_km, flown_km, refused",
    [(100.0, 95.4, True), (100.0, 99.95, True), (100.0, 100.0, False),
     (100.0, 100.5, False), (None, 95.4, False)],
)
def test_certify_holds_the_floor_to_the_metre(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch,
    floor_km: float | None, flown_km: float, refused: bool,
) -> None:
    from conftest import quick_config

    import orbitwin.verify as verify_module
    from orbitwin.errors import AltitudeFloorError
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.search import search_one_burn
    from orbitwin.verify import certify_candidate

    config = quick_config()
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, minimum_flown_altitude_km=floor_km)
    )
    scenario, _ = parse_gmat_script(workshop_script, config)
    surface = config.policy.central_body_radius_km
    real_verify = verify_module.verify_candidate_with_gmat

    def flown(*args, **kwargs):
        verified, run, evidence = real_verify(*args, **kwargs)
        verified.metadata["minimum_pursuer_radius_km"] = surface + flown_km
        return verified, run, evidence

    monkeypatch.setattr(verify_module, "verify_candidate_with_gmat", flown)
    candidate = search_one_burn(scenario, config)[0]
    stamped(candidate, workshop_script, config)
    if refused:
        with pytest.raises(AltitudeFloorError, match=rf"{flown_km:.3f} km altitude"):
            certify_candidate(
                workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "no",
                runs=1,
            )
        assert not (tmp_path / "no" / "submission").exists()
        return
    evidence = certify_candidate(
        workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "yes", runs=1
    )
    assert evidence["flown_minimum_radius_km"] == pytest.approx(surface + flown_km)
    assert evidence["flown_minimum_altitude_km"] == pytest.approx(flown_km)
    assert evidence["minimum_flown_altitude_km"] == floor_km


def test_the_boundary_is_on_the_floor() -> None:
    surface = OrbitWinConfig().policy.central_body_radius_km
    assert below_minimum_flown_altitude(_flown(surface + 100.0), _with(100.0)) is False
    assert below_minimum_flown_altitude(_flown(surface + 99.95), _with(100.0)) is True
