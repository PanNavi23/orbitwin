from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from conftest import stamped

from orbitwin.config import OrbitWinConfig
from orbitwin.constraints import flies_through_central_body
from orbitwin.event_detection import evaluate_analytical_trajectory
from orbitwin.models import Burn, Candidate
from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian

MU_KM3_S2 = 398600.4418
SURFACE_KM = 6378.1363


def _circular(sma_km: float, true_anomaly_deg: float):
    return keplerian_to_cartesian(
        KeplerianElements(sma_km, 0.0, 30.0, 0.0, 0.0, true_anomaly_deg), MU_KM3_S2
    )


def _retrograde(state, magnitude_km_s: float, at_s: float) -> Burn:
    """The impulse that lowers pericenter fastest: straight against velocity."""
    velocity = np.asarray(state.v_km_s, dtype=float)
    dv = tuple(-magnitude_km_s * velocity / np.linalg.norm(velocity))
    return Burn(time_s=at_s, dv_inertial_km_s=dv, dv_script_km_s=dv, slot_index=0)


def test_a_retrograde_impulse_is_measured_below_the_surface() -> None:
    # 1.5 km/s against velocity from a 7000 km circular orbit. The rules cap
    # the impulse at exactly this, so the search can and does reach it.
    pursuer = _circular(7000.0, 0.0)
    target = _circular(7000.0, 40.0)
    burn = _retrograde(pursuer, 1.5, 10.0)
    event = evaluate_analytical_trajectory(
        pursuer, target, (burn,), 5000.0, MU_KM3_S2, 5.0,
        max_step_s=5.0, central_body_radius_km=SURFACE_KM,
    )
    assert event.minimum_pursuer_radius_km is not None
    assert event.minimum_pursuer_radius_km < SURFACE_KM
    assert event.minimum_pursuer_radius_km == pytest.approx(3309.1, abs=5.0)


def test_a_coasting_circular_orbit_stays_at_its_radius() -> None:
    pursuer = _circular(7000.0, 0.0)
    target = _circular(9000.0, 40.0)
    event = evaluate_analytical_trajectory(
        pursuer, target, (), 5000.0, MU_KM3_S2, 5.0,
        max_step_s=5.0, central_body_radius_km=SURFACE_KM,
    )
    assert event.minimum_pursuer_radius_km == pytest.approx(7000.0, abs=1e-3)


def test_the_measurement_stops_at_the_completion_time() -> None:
    # 初賽規則 §4 fixes Δrmin once tint has passed, and the submitted script
    # runs on to 4TA only to carry Δr(4TA) for the failure case. An impulse
    # that drives the pursuer underground *after* the interception belongs to
    # that tail and says nothing about the interception before it.
    pursuer = _circular(7000.0, 0.0)
    # 4 km of chord at 7000 km: inside the 5 km sphere from the first instant.
    target = _circular(7000.0, np.degrees(2.0 * np.arcsin(4.0 / (2.0 * 7000.0))))
    late = _retrograde(pursuer, 1.5, 100.0)

    event = evaluate_analytical_trajectory(
        pursuer, target, (late,), 5000.0, MU_KM3_S2, 5.0,
        max_step_s=5.0, central_body_radius_km=SURFACE_KM,
    )
    assert event.first_entry_time_s == pytest.approx(0.0, abs=1e-9)
    assert event.minimum_pursuer_radius_km == pytest.approx(7000.0, abs=1.0), (
        "the pass below the surface happens after the interception"
    )

    # Without the burn there is nothing underground at all, which pins the
    # test on the cut rather than on the burn being ignored.
    plain = evaluate_analytical_trajectory(
        pursuer, target, (), 5000.0, MU_KM3_S2, 5.0,
        max_step_s=5.0, central_body_radius_km=SURFACE_KM,
    )
    assert plain.minimum_pursuer_radius_km == pytest.approx(7000.0, abs=1e-3)


def test_the_predicate_reads_the_measurement_and_the_configured_radius() -> None:
    config = OrbitWinConfig()
    underground = Candidate(
        burns=(), intercept_time_s=1.0, minimum_distance_km=1.0,
        distance_at_completion_km=5.0, successful=True,
        analytical_endpoint_error_km=None, branch="test",
        metadata={"minimum_pursuer_radius_km": 5807.8},
    )
    clear = Candidate(
        burns=(), intercept_time_s=1.0, minimum_distance_km=1.0,
        distance_at_completion_km=5.0, successful=True,
        analytical_endpoint_error_km=None, branch="test",
        metadata={"minimum_pursuer_radius_km": 6995.5},
    )
    unmeasured = Candidate(
        burns=(), intercept_time_s=1.0, minimum_distance_km=1.0,
        distance_at_completion_km=5.0, successful=True,
        analytical_endpoint_error_km=None, branch="test", metadata={},
    )
    assert flies_through_central_body(underground, config) is True
    assert flies_through_central_body(clear, config) is False
    # Unknown is not a violation: inventing one would be as wrong as hiding one.
    assert flies_through_central_body(unmeasured, config) is False

    # The threshold is the configured radius, not a constant.
    lowered = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, central_body_radius_km=3000.0)
    )
    assert flies_through_central_body(underground, lowered) is False


def test_the_sampled_minimum_tracks_the_exact_one() -> None:
    """Why the radius is read off the samples instead of the interpolant.

    Running the Hermite stationary-point solve a second time, on the pursuer's
    own position, costs a pass over every segment that no radius screen can
    skip: 31.3 s to 59.6 s for eight candidates on the different-plane
    scenario. Near pericenter the radius varies quadratically in time, so at
    the default step the sampled minimum is the same number to about a metre.
    """
    from orbitwin.event_detection import evaluate_hermite_samples
    from orbitwin.trajectory import sample_pair_trajectory_sequential

    pursuer = _circular(7000.0, 0.0)
    target = _circular(7000.0, 10.0)
    for magnitude_km_s in (1.5, 0.8):
        burn = _retrograde(pursuer, magnitude_km_s, 10.0)
        times, _rel_p, _rel_v, positions, velocities = sample_pair_trajectory_sequential(
            pursuer, target, (burn,), 6000.0, MU_KM3_S2, max_step_s=2.0
        )
        exact = evaluate_hermite_samples(
            times, positions, velocities, SURFACE_KM
        ).minimum_distance_km
        sampled = float(np.min(np.linalg.norm(positions, axis=1)))
        assert sampled >= exact, "the sampled minimum cannot dip below the interpolated one"
        assert sampled - exact < 0.01, f"{magnitude_km_s} km/s: {(sampled - exact) * 1000:.1f} m"


def test_certification_asks_whether_the_trajectory_goes_through_the_body(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    """solve refuses to nominate one; certify is what writes the submission.

    The gate lived only in the pipeline, so naming the candidate directly still
    certified it -- a gate on the recommendation is not a gate on the
    submission, the same shape as an authority level that arrived in a JSON
    file. The regression is that certify never asked, so that is what this
    drives: the answer is forced and the two outcomes are checked.

    Forced on the verification flight, since the double's report never dips.
    """
    from conftest import quick_config

    import orbitwin.verify as verify_module
    from orbitwin.errors import PolicyError
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.search import search_one_burn
    from orbitwin.verify import certify_candidate

    config = quick_config()
    assert config.policy.allow_body_intersecting_submission is False
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]

    real_verify = verify_module.verify_candidate_with_gmat

    def diving(*args, **kwargs):
        verified, run, evidence = real_verify(*args, **kwargs)
        verified.metadata["minimum_pursuer_radius_km"] = 2248.4
        return verified, run, evidence

    monkeypatch.setattr(verify_module, "verify_candidate_with_gmat", diving)
    stamped(candidate, workshop_script, config)
    with pytest.raises(PolicyError, match=r"inside the .* central body"):
        certify_candidate(
            workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "no", runs=1
        )
    assert not (tmp_path / "no" / "submission").exists(), "no submission may survive the refusal"

    # It is a policy, not a law: the rules say nothing about it, so the same
    # candidate certifies when the case is explicitly accepted.
    permitted = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, allow_body_intersecting_submission=True)
    )
    evidence = certify_candidate(
        workshop_script, candidate, permitted, fake_gmat, fake_startup, tmp_path / "yes", runs=1
    )
    assert evidence["stage"]
