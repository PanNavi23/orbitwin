from __future__ import annotations

import numpy as np
import pytest

from orbitwin.event_detection import evaluate_hermite_samples


def test_first_entry_linear_crossing() -> None:
    times = np.array([0.0, 10.0])
    positions = np.array([[10.0, 0, 0], [0.0, 0, 0]])
    velocities = np.array([[-1.0, 0, 0], [-1.0, 0, 0]])
    result = evaluate_hermite_samples(times, positions, velocities, 5.0)
    assert result.first_entry_time_s == pytest.approx(5.0, abs=1e-8)
    assert result.minimum_distance_km == pytest.approx(0.0, abs=1e-9)


def test_hidden_between_sample_crossing_detected() -> None:
    # Both endpoint distances are 10 km. Cubic Hermite velocity pulls the path
    # through the origin between them.
    times = np.array([0.0, 2.0])
    positions = np.array([[10.0, 0, 0], [10.0, 0, 0]])
    velocities = np.array([[-20.0, 0, 0], [20.0, 0, 0]])
    result = evaluate_hermite_samples(times, positions, velocities, 5.0)
    assert result.first_entry_time_s is not None
    assert result.minimum_distance_km < 1e-7


def test_duplicate_time_handles_impulsive_velocity_jump() -> None:
    times = np.array([0.0, 1.0, 1.0, 2.0])
    positions = np.array([[10.0, 0, 0], [9.0, 0, 0], [9.0, 0, 0], [4.0, 0, 0]])
    velocities = np.array([[-1.0, 0, 0], [-1.0, 0, 0], [-5.0, 0, 0], [-5.0, 0, 0]])
    result = evaluate_hermite_samples(times, positions, velocities, 5.0)
    assert result.first_entry_time_s == pytest.approx(1.8, abs=1e-7)


def test_initially_inside_sphere() -> None:
    result = evaluate_hermite_samples(
        np.array([0.0]),
        np.array([[4.0, 0, 0]]),
        np.array([[0.0, 0, 0]]),
        5.0,
    )
    assert result.first_entry_time_s == 0.0


def test_sequential_trajectory_matches_independent_universal_propagation() -> None:
    from orbitwin.models import Burn, StateVector
    from orbitwin.trajectory import (
        sample_relative_trajectory,
        sample_relative_trajectory_sequential,
    )

    mu = 398600.4418
    pursuer = StateVector((7000.0, 0.0, 0.0), (0.0, 7.45, 0.7))
    target = StateVector((8100.0, 200.0, -150.0), (-0.2, 6.9, 1.0))
    burn = Burn(300.0, (0.02, -0.01, 0.03), (0.02, -0.01, 0.03), 0)
    times, positions, velocities = sample_relative_trajectory_sequential(
        pursuer,
        target,
        (burn,),
        900.0,
        mu,
        max_step_s=30.0,
    )
    unique_times = np.unique(times)
    _, expected_positions, expected_velocities = sample_relative_trajectory(
        pursuer,
        target,
        (burn,),
        unique_times,
        mu,
    )
    # Select the final duplicate at a burn epoch, which is the post-burn state.
    actual_positions = np.array([positions[np.flatnonzero(times == time)[-1]] for time in unique_times])
    actual_velocities = np.array([velocities[np.flatnonzero(times == time)[-1]] for time in unique_times])
    assert np.max(np.linalg.norm(actual_positions - expected_positions, axis=1)) < 2.0e-6
    assert np.max(np.linalg.norm(actual_velocities - expected_velocities, axis=1)) < 2.0e-9


def test_exact_tangential_contact_is_an_interception() -> None:
    # The rule is |dr(t)| <= 5 km, so grazing the sphere succeeds. The distance
    # polynomial has a double root here: q(u*) == 0 with q'(u*) == 0, which a
    # slope-sign test cannot accept.
    speed = 10.0
    times = np.array([0.0, 1.0])
    positions = np.array([[5.0, -speed * 0.5, 0.0], [5.0, speed * 0.5, 0.0]])
    velocities = np.array([[0.0, speed, 0.0], [0.0, speed, 0.0]])
    result = evaluate_hermite_samples(times, positions, velocities, 5.0)
    assert result.minimum_distance_km == pytest.approx(5.0, abs=1e-12)
    assert result.first_entry_time_s is not None
    assert result.first_entry_time_s == pytest.approx(0.5, abs=1e-6)


def test_minimum_below_radius_always_yields_a_first_entry() -> None:
    # Sweep the boundary: whenever the reported minimum is inside the sphere a
    # first-entry epoch must exist, otherwise the two answers contradict.
    rng = np.random.default_rng(20260820)
    for exponent in range(0, 13):
        d_min = 5.0 - 10.0 ** (-exponent)
        for _ in range(25):
            speed = float(rng.uniform(0.3, 4.0))
            dt = float(rng.uniform(0.5, 4.0))
            u_star = float(rng.uniform(0.25, 0.75))
            normal = rng.normal(size=3)
            normal /= np.linalg.norm(normal)
            tangent = rng.normal(size=3)
            tangent -= tangent.dot(normal) * normal
            tangent /= np.linalg.norm(tangent)
            p0 = d_min * normal + speed * (-u_star * dt) * tangent
            p1 = d_min * normal + speed * ((1.0 - u_star) * dt) * tangent
            velocity = speed * tangent
            result = evaluate_hermite_samples(
                np.array([0.0, dt]), np.array([p0, p1]), np.array([velocity, velocity]), 5.0
            )
            if result.minimum_distance_km <= 5.0:
                assert result.first_entry_time_s is not None, (d_min, dt, speed)


def test_entry_resolution_is_reported() -> None:
    # A crossing that no raw sample witnesses is legitimate but interpolated.
    times = np.array([0.0, 2.0])
    positions = np.array([[10.0, 0, 0], [10.0, 0, 0]])
    velocities = np.array([[-20.0, 0, 0], [20.0, 0, 0]])
    result = evaluate_hermite_samples(times, positions, velocities, 5.0)
    assert result.samples_inside_radius == 0
    assert result.first_entry_segment_dt_s == pytest.approx(2.0)
    assert result.entry_is_under_resolved is True

    dense = np.linspace(0.0, 10.0, 101)
    positions = np.column_stack([10.0 - dense, np.zeros_like(dense), np.zeros_like(dense)])
    velocities = np.column_stack(
        [-np.ones_like(dense), np.zeros_like(dense), np.zeros_like(dense)]
    )
    result = evaluate_hermite_samples(dense, positions, velocities, 5.0)
    assert result.samples_inside_radius > 2
    assert result.entry_is_under_resolved is False


def test_entry_resolution_does_not_double_count_a_burn_epoch() -> None:
    # The sampler emits a duplicate row at an impulsive burn -- same position,
    # jumped velocity. Counting rows would report two independent samples where
    # there is one position, and that number decides whether the crossing is
    # reported as interpolated.
    times = np.array([0.0, 1.0, 1.0, 2.0])
    positions = np.array([[9.0, 0, 0], [4.0, 0, 0], [4.0, 0, 0], [3.0, 0, 0]])
    velocities = np.array([[-5.0, 0, 0], [-5.0, 0, 0], [-1.0, 0, 0], [-1.0, 0, 0]])
    result = evaluate_hermite_samples(times, positions, velocities, 5.0)
    # Three rows are inside the sphere but only two distinct epochs are.
    assert result.samples_inside_radius == 2


def test_segment_screening_bound_is_rigorous() -> None:
    # evaluate_hermite_samples skips a segment when
    #     max(d0, d1) - (1.5|p1 - p0| + dt(|v0| + |v1|))
    # already exceeds both the running minimum and the radius. The bound comes
    # from writing the cubic in Hermite basis, where |h00'| <= 3/2 and
    # |h10'|, |h11'| <= 1 on [0, 1], so the arc length cannot exceed it. If the
    # bound is ever weakened the skip starts discarding real crossings, and the
    # failure is silent, so it is pinned here rather than only in a comment.
    import math

    rng = np.random.default_rng(11)
    samples = np.linspace(0.0, 1.0, 2001)
    for _ in range(4000):
        scale = float(rng.choice([1.0, 10.0, 1000.0]))
        p0 = rng.normal(scale=scale, size=3)
        p1 = p0 + rng.normal(scale=float(rng.choice([0.1, 1.0, 50.0])), size=3)
        v0 = rng.normal(scale=5.0, size=3)
        v1 = rng.normal(scale=5.0, size=3)
        dt = float(rng.uniform(0.05, 5.0))

        a0, a1 = p0, dt * v0
        a2 = -3.0 * p0 - 2.0 * dt * v0 + 3.0 * p1 - dt * v1
        a3 = 2.0 * p0 + dt * v0 - 2.0 * p1 + dt * v1
        curve = (
            a0[None, :]
            + np.outer(samples, a1)
            + np.outer(samples**2, a2)
            + np.outer(samples**3, a3)
        )
        true_minimum = float(np.linalg.norm(curve, axis=1).min())

        reach = 1.5 * float(np.linalg.norm(p1 - p0)) + dt * (
            float(np.linalg.norm(v0)) + float(np.linalg.norm(v1))
        )
        bound = max(float(np.linalg.norm(p0)), float(np.linalg.norm(p1))) - reach
        assert true_minimum >= bound - 1e-9, (true_minimum, bound)
        # The bound must also never exceed min(d0, d1), otherwise skipping a
        # segment could discard an improvement sitting at an endpoint.
        assert bound <= min(float(np.linalg.norm(p0)), float(np.linalg.norm(p1))) + 1e-9
        assert math.isfinite(reach)


def test_screening_still_finds_a_crossing_the_endpoints_hide() -> None:
    # Both endpoints sit far outside the sphere and only the interior of the
    # segment enters it. The screening bound has to keep this segment.
    times = np.array([0.0, 4.0])
    positions = np.array([[400.0, 0, 0], [400.0, 0, 0]])
    velocities = np.array([[-400.0, 0, 0], [400.0, 0, 0]])
    result = evaluate_hermite_samples(times, positions, velocities, 5.0)
    assert result.first_entry_time_s is not None
    assert result.minimum_distance_km < 5.0
    assert result.samples_inside_radius == 0


def _slow_closing_pair():
    """Two near-identical orbits that drift into the sphere late in the horizon."""
    import math

    from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian

    mu = 398600.4418
    sma = 7000.0
    pursuer = keplerian_to_cartesian(KeplerianElements(sma, 0.0, 30.0, 10.0, 0.0, 0.0), mu)
    target = keplerian_to_cartesian(
        KeplerianElements(sma * 1.0002, 0.0, 30.0, 10.0, 0.0, 0.2), mu
    )
    horizon = 4.0 * 2.0 * math.pi * math.sqrt(sma**3 / mu)
    return pursuer, target, mu, horizon


def test_an_interception_after_the_arrival_epoch_is_still_legal() -> None:
    # 初賽規則 §4 accepts an interception at any t <= Tmax. Evaluating a
    # candidate only as far as its Lambert arrival cannot see one that closes
    # later, and reports the candidate as a failure.
    from orbitwin.event_detection import evaluate_analytical_trajectory

    pursuer, target, mu, horizon = _slow_closing_pair()
    full = evaluate_analytical_trajectory(pursuer, target, (), horizon, mu, 5.0, max_step_s=10.0)
    assert full.first_entry_time_s is not None
    assert full.first_entry_time_s > 0.25 * horizon, "the closure should be well after an early arrival"

    early = evaluate_analytical_trajectory(
        pursuer, target, (), full.first_entry_time_s * 0.5, mu, 5.0, max_step_s=10.0
    )
    assert early.first_entry_time_s is None


def test_the_failure_distance_depends_on_where_it_is_measured() -> None:
    # §4 sets Tteam = 4TA when there is no interception, so the score's distance
    # term is Δr(4TA). Measuring at an earlier arrival measures a different
    # instant, and on a slowly closing pair the two differ by kilometres.
    import math

    from orbitwin.event_detection import evaluate_analytical_trajectory
    from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian

    mu = 398600.4418
    sma = 7000.0
    pursuer = keplerian_to_cartesian(KeplerianElements(sma, 0.0, 30.0, 10.0, 0.0, 0.0), mu)
    target = keplerian_to_cartesian(
        KeplerianElements(sma * 1.00002, 0.0, 30.0, 10.0, 0.0, 0.30), mu
    )
    horizon = 4.0 * 2.0 * math.pi * math.sqrt(sma**3 / mu)

    early = evaluate_analytical_trajectory(pursuer, target, (), 500.0, mu, 5.0, max_step_s=10.0)
    full = evaluate_analytical_trajectory(pursuer, target, (), horizon, mu, 5.0, max_step_s=10.0)
    assert early.first_entry_time_s is None and full.first_entry_time_s is None
    assert abs(early.distance_at_end_km - full.distance_at_end_km) > 1.0


def test_extending_the_window_does_not_coarsen_the_grid() -> None:
    """What the second refinement pass costs, and what it does not.

    refine_events evaluates a candidate to its Lambert arrival and re-evaluates
    only the failures out to Tmax, because §4 scores Δr(4TA) on a mission that
    never closes and accepts an interception at any t <= Tmax. I wrote in that
    change that only failures pay for the second pass, and worried in the same
    breath that a longer window would thin the samples. It does not: the
    sampler is given a maximum step, not a sample count, so the second pass is
    strictly more samples at the same resolution.
    """
    import numpy as np

    from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian
    from orbitwin.trajectory import sample_pair_trajectory_sequential

    mu = 398600.4418
    pursuer = keplerian_to_cartesian(KeplerianElements(7000.0, 0.0, 30.0, 0.0, 0.0, 0.0), mu)
    target = keplerian_to_cartesian(KeplerianElements(9000.0, 0.0, 60.0, 20.0, 0.0, 10.0), mu)

    short_times, *_ = sample_pair_trajectory_sequential(
        pursuer, target, (), 4000.0, mu, max_step_s=2.0
    )
    long_times, *_ = sample_pair_trajectory_sequential(
        pursuer, target, (), 23314.0, mu, max_step_s=2.0
    )
    assert len(long_times) > len(short_times)
    assert float(np.max(np.diff(short_times))) <= 2.0 + 1e-9
    assert float(np.max(np.diff(long_times))) <= 2.0 + 1e-9
    # Roughly proportional to the window, which is what a fixed step gives.
    assert len(long_times) / len(short_times) == pytest.approx(23314.0 / 4000.0, rel=0.05)
