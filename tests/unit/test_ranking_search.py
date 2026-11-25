"""The capped search on the ranking score: kernels, closest approach, walls.

Two constructed coplanar scenarios carry the end-to-end tests. The pursuer
sits on a 7000 km circle; one tangential impulse at the 0.249 km/s cap raises
its apoapsis to R_MAX, and the target circles at R_MAX - 30 km (a hit is in
reach) or R_MAX + 10 km (it is not: the pursuer can never get within 10 km of
a radius it cannot reach with one impulse). The target is phased to be at the
apoapsis when the pursuer arrives.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import multiprocessing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from orbitwin import ranking_search as rs
from orbitwin.config import OrbitWinConfig, ScoreConfig
from orbitwin.constraints import apply_constraints
from orbitwin.errors import DynamicsError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Scenario, StateVector, as_vector3
from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, propagate_universal
from orbitwin.script_renderer import render_candidate
from orbitwin.trajectory import state_with_burns_at

MU = 398600.4418
CAP = 0.249
R_PURSUER = 7000.0
HORIZON_S = 4000.0
SCORE = ScoreConfig(kt=0.002, ct=2000.0, kv=0.002, cv=1000.0, delta_v_unit="m/s", cr_km=100.0)


def _apoapsis_after_cap() -> float:
    speed = math.sqrt(MU / R_PURSUER) + CAP
    a = 1.0 / (2.0 / R_PURSUER - speed * speed / MU)
    return 2.0 * a - R_PURSUER


R_MAX = _apoapsis_after_cap()


def _ranking_config(score: ScoreConfig | None) -> OrbitWinConfig:
    base = OrbitWinConfig()
    policy = dataclasses.replace(
        base.policy,
        rules="ranking",
        horizon_s=HORIZON_S,
        dv_limit_km_s=0.25,
        dv_safe_limit_km_s=CAP,
        total_dv_limit_km_s=3.0,
        total_dv_safe_limit_km_s=2.99,
    )
    return dataclasses.replace(
        base, policy=policy, score=score, pursuer_name="Pursuer", target_name="AlienSC"
    )


def _coplanar_script(tmp_path: Path, no_maneuver_script: Path, target_radius_km: float) -> Path:
    """Pursuer at TA 0; the target where it will be when a Hohmann-like arc arrives."""
    transfer_a = (R_PURSUER + target_radius_km) / 2.0
    transfer_s = math.pi * math.sqrt(transfer_a**3 / MU)
    target_rate = math.sqrt(MU / target_radius_km**3)
    target_ta = (180.0 - math.degrees(target_rate * transfer_s)) % 360.0
    text = no_maneuver_script.read_text(encoding="utf-8")
    replacements = {
        "Pursuer.SMA = 7000;": f"Pursuer.SMA = {R_PURSUER!r};",
        "Pursuer.INC = 28;": "Pursuer.INC = 30;",
        "Pursuer.RAAN = 15;": "Pursuer.RAAN = 40;",
        "Pursuer.TA = 300;": "Pursuer.TA = 0;",
        "AlienSC.SMA = 9000;": f"AlienSC.SMA = {target_radius_km!r};",
        "AlienSC.INC = 43;": "AlienSC.INC = 30;",
        "AlienSC.RAAN = 65;": "AlienSC.RAAN = 40;",
        "AlienSC.TA = 20;": f"AlienSC.TA = {target_ta!r};",
    }
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / f"coplanar_{int(target_radius_km)}.script"
    path.write_text(text, encoding="utf-8")
    return path


def _scenario(
    tmp_path: Path, no_maneuver_script: Path, target_radius_km: float, score: ScoreConfig | None
) -> tuple[Scenario, OrbitWinConfig]:
    config = _ranking_config(score)
    script = _coplanar_script(tmp_path, no_maneuver_script, target_radius_km)
    return parse_gmat_script(script, config)[0], config


def _small_search(
    scenario: Scenario, config: OrbitWinConfig, burn_counts: list[int], workers: int = 1
) -> rs.RankingSearchResult:
    setup = rs.build_setup(
        scenario, config, screen_iterations=60, polish_iterations=120, grid_step_s=30.0
    )
    starts = rs.build_starts(
        setup, config, burn_counts, transfer_count=3, random_count=2, transfer_grid=(8, 12)
    )
    return rs.search_ranking(scenario, config, setup, starts, workers=workers, polish_top=2)


# -- target tracks


def test_the_ephemeris_track_is_the_ephemeris_and_reaches_a_short_end() -> None:
    """GMAT stopped 5e-7 s short of T_max on R2; target_ephemeris_from_gmat takes it."""
    from orbitwin.ephemeris import Ephemeris

    hyperbola = KeplerianElements(7000.0 / (1.0 - 1.5), 1.5, 50.0, 75.0, 0.0, -30.0)
    start = keplerian_to_cartesian(hyperbola, MU)
    horizon = 600.0
    times = np.append(np.arange(0.0, horizon, 60.0), horizon - 5.0e-7)
    states = [propagate_universal(start, float(t), MU) for t in times]
    ephemeris = Ephemeris(
        times, np.array([s.r for s in states]), np.array([s.v for s in states])
    )
    track = rs.EphemerisTarget(ephemeris)
    for t in (12.5, 299.0, 590.0):
        got_r, got_v = track.state(t)
        want = ephemeris.state_at(t)
        assert np.linalg.norm(got_r - want.r) < 1.0e-9
        assert np.linalg.norm(got_v - want.v) < 1.0e-12
    grid = rs.MeasurementGrid.build(track, horizon, 30.0)
    exact = propagate_universal(start, horizon, MU)
    assert np.linalg.norm(grid.target_r[-1] - exact.r) < 1.0e-2
    with pytest.raises(DynamicsError):
        track.state(horizon + 1.0e-3)


def test_a_burn_inside_a_grid_segment_ends_it_with_the_pre_burn_velocity() -> None:
    flight0, target, _ = _crossing(3.0)
    impulse = np.array([[0.0, 0.2, -0.1]])
    start = StateVector(as_vector3(flight0.initial_r), as_vector3(flight0.initial_v))
    flight = rs.fly([1360.0], impulse, start, MU)
    grid = rs.MeasurementGrid.build(target, 3000.0, 300.0)
    times, _, rel_v = rs.relative_knots(flight, target, grid)
    at_burn = np.flatnonzero(np.abs(times - 1360.0) < 1.0e-9)
    assert len(at_burn) == 2
    _, target_v = target.state(1360.0)
    assert rel_v[at_burn[0]] == pytest.approx(flight.v_before[0] - target_v, abs=1e-12)
    assert rel_v[at_burn[1]] == pytest.approx(flight.v_after[0] - target_v, abs=1e-12)
    assert np.all(np.diff(times) >= 0.0)


@pytest.mark.parametrize("phase", ["approach", "miss", "hit", "hit-dv"])
def test_every_phase_objective_has_its_own_gradient(
    tmp_path: Path, no_maneuver_script: Path, phase: str
) -> None:
    # The hit-time term needs a time where it still has a slope.
    score = dataclasses.replace(SCORE, kt=0.002, ct=2500.0)
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, score)
    setup = rs.build_setup(scenario, config)
    problem = rs._Problem(setup, 2)
    start = rs.random_starts(2, setup, 1, 5)[0]
    x = problem.pack(start.times_s, np.asarray(start.impulses_km_s), start.aim_s)
    _, gradient = problem.objective(x, phase)
    for index in range(x.size):
        up, down = x.copy(), x.copy()
        up[index] += 1.0e-6
        down[index] -= 1.0e-6
        slope = (problem.objective(up, phase)[0] - problem.objective(down, phase)[0]) / 2.0e-6
        assert slope == pytest.approx(gradient[index], rel=1e-4, abs=1e-6), index


def test_the_miss_phase_prices_the_sphere_as_the_score_does(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    """Flat at 50 inside and, since the 2026/11/04 term, flat at its edge too.

    The 0.5 pt/km pull inside was the old term's edge slope. Without it the
    two-body bests of R1, R2, R4 and R5 at C_r 300 km came out the same.
    """
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    problem = rs._Problem(rs.build_setup(scenario, config), 1)
    assert problem._distance_term(5.0) == (50.0, 0.0)
    assert problem._distance_term(2.0) == (50.0, 0.0)
    assert problem._distance_term(105.0)[1] < 0.0


def test_no_burn_survives_past_the_first_entry(tmp_path: Path, no_maneuver_script: Path) -> None:
    """The Hohmann hit, plus a burn after it: the mission ends at the entry."""
    target_km = R_MAX - 30.0
    scenario, config = _scenario(tmp_path, no_maneuver_script, target_km, SCORE)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    velocity = scenario.pursuer.initial_state.v
    speed = float(np.linalg.norm(velocity))
    needed = math.sqrt(2.0 * MU * target_km / (R_PURSUER * (R_PURSUER + target_km))) - speed
    impulses = np.array([velocity / speed * needed, [0.0, 0.1, 0.0]])
    solution = rs.settle("late", "hit", [1.0e-3, 3900.0], impulses, setup)
    assert solution is not None and solution.successful
    assert solution.burn_count == 1 and solution.phase.endswith("+trimmed")
    assert solution.measurement.intercept_time_s is not None
    assert max(solution.times_s) <= solution.measurement.intercept_time_s


def test_a_dive_after_the_hit_is_trimmed_not_refused(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    """The Hohmann hit, then a retrograde 0.249 km/s burn that would bring
    periapsis under the floor before an 8000 s horizon. The mission ends at
    the entry, so the burn is dropped and the arc that is flown is checked."""
    target_km = R_MAX - 30.0
    scenario, config = _scenario(tmp_path, no_maneuver_script, target_km, SCORE)
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, horizon_s=8000.0)
    )
    scenario, _ = parse_gmat_script(Path(scenario.source_path), config)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    velocity = scenario.pursuer.initial_state.v
    speed = float(np.linalg.norm(velocity))
    needed = math.sqrt(2.0 * MU * target_km / (R_PURSUER * (R_PURSUER + target_km))) - speed
    hohmann = velocity / speed * needed
    flight = rs.fly([1.0e-3, 3300.0], np.array([hohmann, [0.0, 0.0, 0.0]]),
                    scenario.pursuer.initial_state, MU)
    retro = -flight.v_before[1] / np.linalg.norm(flight.v_before[1]) * CAP
    whole = rs.fly([1.0e-3, 3300.0], np.array([hohmann, retro]), scenario.pursuer.initial_state, MU)
    assert rs.minimum_radius_over_horizon(whole, 8000.0) < setup.limits.radius_floor_km
    solution = rs.settle(
        "dive", "hit", [1.0e-3, 3300.0], np.array([hohmann, retro]), setup, 10.0
    )
    assert solution is not None and solution.phase.endswith("+trimmed")
    assert solution.burn_count == 1 and solution.successful
    # The trimmed schedule keeps the margin it was searched under.
    assert solution.floor_margin_km == 10.0
    assert solution.measurement.minimum_radius_km >= setup.limits.radius_floor_km


def test_the_gradients_are_the_finite_differences(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    setup = rs.build_setup(scenario, config)
    problem = rs._Problem(setup, 2)
    start = rs.random_starts(2, setup, 1, 5)[0]
    x = problem.pack(start.times_s, np.asarray(start.impulses_km_s), start.aim_s)
    exact = problem.evaluate(x)
    for index in range(x.size):
        up, down = x.copy(), x.copy()
        up[index] += 1.0e-6
        down[index] -= 1.0e-6
        high, low = problem._evaluate(up), problem._evaluate(down)
        miss = (high.miss_km - low.miss_km) / 2.0e-6
        floors = (high.floors_km - low.floors_km) / 2.0e-6
        assert miss == pytest.approx(exact.miss_gradient[index], rel=1e-5, abs=1e-4)
        assert floors == pytest.approx(exact.floor_gradients[:, index], rel=1e-5, abs=1e-4)


# -- closest approach and first entry


def _crossing(offset_km: float) -> tuple[rs.Flight, rs.ConicTarget, float]:
    """Two craft that pass `offset_km` apart at t* = 1350 s, about 1 km/s."""
    at_time = 1350.0
    target_now = keplerian_to_cartesian(KeplerianElements(7200.0, 0.0, 40.0, 10.0, 0.0, 0.0), MU)
    relative_v = np.array([0.3, -0.5, 0.8])
    side = np.cross(relative_v, target_now.v)
    side = side / np.linalg.norm(side)
    pursuer_now = StateVector(
        as_vector3(target_now.r + offset_km * side), as_vector3(target_now.v + relative_v)
    )
    pursuer0 = propagate_universal(pursuer_now, -at_time, MU)
    target0 = propagate_universal(target_now, -at_time, MU)
    flight = rs.fly([], np.zeros((0, 3)), pursuer0, MU)
    return flight, rs.ConicTarget(target0.r_km, target0.v_km_s, MU), at_time


def _independent_distance(flight: rs.Flight, target: rs.ConicTarget, time_s: float) -> float:
    pursuer = propagate_universal(
        StateVector(as_vector3(flight.initial_r), as_vector3(flight.initial_v)), time_s, MU
    )
    other = propagate_universal(StateVector(target.r0_km, target.v0_km_s), time_s, MU)
    return float(np.linalg.norm(pursuer.r - other.r))


def test_a_crossing_between_knots_is_found_exactly() -> None:
    """Knots 300 s apart, the craft 300 km apart at the nearest knot."""
    flight, target, at_time = _crossing(0.0)
    grid = rs.MeasurementGrid.build(target, 3000.0, 300.0)
    measured = rs.measure(flight, target, grid, 3000.0, 5.0)
    knots = [_independent_distance(flight, target, float(t)) for t in grid.times_s]
    assert min(knots) > 100.0
    assert measured.minimum_distance_km < 1.0e-5
    assert measured.minimum_distance_time_s == pytest.approx(at_time, abs=1.0e-3)
    # About 5 km / 1 km/s before the crossing.
    expected_entry = _bisect(lambda t: _independent_distance(flight, target, t) - 5.0, 0.0, at_time)
    assert measured.intercept_time_s == pytest.approx(expected_entry, abs=1.0e-6)


def _bisect(function: object, low: float, high: float) -> float:
    for _ in range(100):
        middle = 0.5 * (low + high)
        if function(middle) <= 0.0:  # type: ignore[operator]
            high = middle
        else:
            low = middle
    return high


@pytest.mark.parametrize("offset_km", [3.0, 12.0])
def test_a_near_pass_matches_an_independent_minimiser(offset_km: float) -> None:
    flight, target, at_time = _crossing(offset_km)
    grid = rs.MeasurementGrid.build(target, 3000.0, 30.0)
    measured = rs.measure(flight, target, grid, 3000.0, 5.0)
    from scipy.optimize import minimize_scalar

    reference = minimize_scalar(
        lambda t: _independent_distance(flight, target, t),
        bounds=(at_time - 60.0, at_time + 60.0),
        method="bounded",
        options={"xatol": 1.0e-9},
    )
    assert measured.minimum_distance_km == pytest.approx(float(reference.fun), abs=1.0e-6)
    assert (measured.intercept_time_s is not None) == (offset_km < 5.0)


# -- walls


def _independent_wall_check(
    times: list[float], impulses: list[np.ndarray], scenario: Scenario, config: OrbitWinConfig
) -> None:
    policy = config.policy
    window_s = scenario.max_time_s - policy.final_padding_s - policy.min_propagate_duration_s
    assert len(times) <= 1 + math.floor(window_s / policy.safe_burn_separation_s + 1.0e-9)
    for impulse in impulses:
        assert float(np.linalg.norm(impulse)) <= policy.dv_safe_limit_km_s + 1.0e-12
    assert policy.total_dv_safe_limit_km_s is not None
    assert sum(float(np.linalg.norm(i)) for i in impulses) <= (
        policy.total_dv_safe_limit_km_s + 1.0e-12
    )
    if times:
        assert times[0] >= policy.min_propagate_duration_s - 1.0e-9
        assert times[-1] <= scenario.max_time_s - policy.final_padding_s + 1.0e-9
    for before, after in itertools.pairwise(times):
        assert after - before >= policy.safe_burn_separation_s - 1.0e-9
    burns = tuple(
        Burn(t, as_vector3(dv), as_vector3(dv)) for t, dv in zip(times, impulses, strict=True)
    )
    radii = [
        float(
            np.linalg.norm(
                state_with_burns_at(scenario.pursuer.initial_state, burns, float(t), MU).r
            )
        )
        for t in np.linspace(0.0, scenario.max_time_s, 201)
    ]
    assert min(radii) >= policy.central_body_radius_km + 100.0 - 1.0e-6


def test_no_settled_schedule_crosses_a_wall(tmp_path: Path, no_maneuver_script: Path) -> None:
    """Adversarial schedules: over the cap, too close, out of the window, diving."""
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    generator = np.random.default_rng(20261101)
    kept = 0
    for _ in range(100):
        count = int(generator.integers(0, 16))
        times = sorted(generator.uniform(-50.0, HORIZON_S + 50.0, count).tolist())
        if count and generator.uniform() < 0.3:
            times = [times[0] + 40.0 * k for k in range(count)]
        impulses = generator.normal(0.0, 1.0, (count, 3))
        impulses *= (generator.uniform(0.0, 3.0 * CAP, count) / np.linalg.norm(impulses, axis=1))[
            :, None
        ]
        solution = rs.settle("fuzz", "fuzz", times, impulses, setup)
        if solution is None:
            continue
        kept += 1
        _independent_wall_check(
            list(solution.times_s), [np.array(i) for i in solution.impulses_km_s], scenario, config
        )
    assert kept > 10


def test_a_dive_below_the_floor_is_refused(tmp_path: Path, no_maneuver_script: Path) -> None:
    """Twelve retrograde impulses at the cap put periapsis 1300 km underground."""
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    times = [1.0 + 101.0 * k for k in range(12)]
    flight = rs.fly(times[:1], np.zeros((1, 3)), scenario.pursuer.initial_state, MU)
    retro = -flight.v_before[0] / np.linalg.norm(flight.v_before[0]) * CAP
    assert rs.settle("dive", "dive", times, np.tile(retro, (12, 1)), setup) is None


def test_the_wall_is_the_floor_and_the_margins_are_searched(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    """Periapsis at 105 km altitude is legal: the 100 km floor is the wall,
    and the 10 km force-model margin binds only what SLSQP is asked to hold."""
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    periapsis = config.policy.central_body_radius_km + 105.0
    velocity = scenario.pursuer.initial_state.v
    speed = float(np.linalg.norm(velocity))
    wanted = math.sqrt(2.0 * MU * periapsis / (R_PURSUER * (R_PURSUER + periapsis)))
    retro = (velocity / speed * (wanted - speed))[None, :]
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    assert setup.limits.radius_floor_km == pytest.approx(
        config.policy.central_body_radius_km + 100.0
    )
    # 10 km by default; 0 only when asked for.
    assert setup.force_model_margins_km == rs.FORCE_MODEL_MARGINS_KM == (10.0,)
    kept = rs.settle("low", "miss", [1.0e-3], retro, setup, 10.0)
    assert kept is not None and kept.floor_margin_km == 10.0
    assert kept.measurement.minimum_radius_km == pytest.approx(periapsis, abs=1.0e-6)
    starts = rs.build_starts(setup, config, [1], transfer_count=1, random_count=1,
                             transfer_grid=(4, 6))
    assert {start.floor_margin_km for start in starts} == {10.0}
    both = rs.build_setup(scenario, config, grid_step_s=60.0, force_model_margins_km=(10.0, 0.0))
    assert both.force_model_margins_km == (0.0, 10.0)
    starts = rs.build_starts(both, config, [1], transfer_count=1, random_count=1,
                             transfer_grid=(4, 6))
    assert sorted({start.floor_margin_km for start in starts}) == [0.0, 10.0]
    assert len({start.label for start in starts}) == len(starts)


@pytest.mark.parametrize("margin", [0.0, 10.0])
def test_slsqp_is_held_above_the_wall(
    tmp_path: Path, no_maneuver_script: Path, margin: float
) -> None:
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    setup = rs.build_setup(scenario, config)
    problem = rs._Problem(setup, 2, margin)
    start = rs.random_starts(2, setup, 1, 5)[0]
    x = problem.pack(start.times_s, np.asarray(start.impulses_km_s), start.aim_s)
    floors = problem.evaluate(x).floors_km
    held = problem.constraints(x, None)[2 + 1 + 1 : 2 + 1 + 1 + 3]
    wall = (floors - setup.limits.radius_floor_km - margin) / 1000.0
    assert np.all(held < wall)


def test_both_margins_reach_the_results(tmp_path: Path, no_maneuver_script: Path) -> None:
    """Each margin's starts come back tagged with it, and each margin's best is
    kept for GMAT even when the other margin's ranks higher."""
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    setup = rs.build_setup(
        scenario, config, screen_iterations=40, grid_step_s=60.0, force_model_margins_km=(0, 10)
    )
    starts = rs.build_starts(setup, config, [1], transfer_count=1, random_count=0,
                             transfer_grid=(6, 8))
    batches = rs._refine_all([(start, 40) for start in starts], setup, 1)
    for start, batch in zip(starts, batches, strict=True):
        assert batch and {s.floor_margin_km for s in batch} == {start.floor_margin_km}
    low, high = batches[0][0], dataclasses.replace(batches[1][0], label="other")
    result = rs.RankingSearchResult(setup, 2, [low, high], None)
    assert rs.best_per_margin(result) == {low.floor_margin_km: low, high.floor_margin_km: high}
    # The margin with the worse results still gets its own polish.
    better, worse = (low, high) if low.floor_margin_km == 0.0 else (high, low)

    def key(solution: rs.RankingSolution) -> tuple[object, ...]:
        return (0 if solution is better else 1, solution.label)

    chosen = rs.polish_selection([[better], [worse]], setup.force_model_margins_km, key, 1)
    assert chosen == [better, worse]


def test_the_baseline_stands_for_every_zero_burn_schedule(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    """A polished schedule pruned to nothing sorted ahead of the baseline and
    took its place, under another label and candidate id."""
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX + 10.0, SCORE)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    baseline = rs.settle("zero-burn", "baseline", [], np.zeros((0, 3)), setup)
    pruned = rs.settle("a/pruned", "miss", [], np.zeros((0, 3)), setup)
    real = rs.settle("real", "miss", [500.0], np.array([[0.0, 0.1, 0.0]]), setup)
    assert baseline is not None and pruned is not None and real is not None

    def key(solution: rs.RankingSolution) -> tuple[object, ...]:
        return (solution.burn_count, solution.label)

    ranked = rs.rank_solutions([pruned, real], key, baseline)
    assert ranked[0] is baseline and len(ranked) == 2


def test_dust_behind_the_zero_burn_is_dropped(tmp_path: Path, no_maneuver_script: Path) -> None:
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX + 10.0, SCORE)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    baseline = rs.settle("zero-burn", "baseline", [], np.zeros((0, 3)), setup)
    dust = rs.settle("dust", "miss", [500.0], np.array([[0.0, 8.6e-6, 0.0]]), setup)
    assert baseline is not None and dust is not None

    def behind(solution: rs.RankingSolution) -> tuple[object, ...]:
        return (solution.burn_count, solution.label)

    def ahead(solution: rs.RankingSolution) -> tuple[object, ...]:
        return (-solution.burn_count, solution.label)

    assert rs.rank_solutions([dust, baseline], behind) == [baseline]
    # Dust that does better than spending nothing is kept.
    assert rs.rank_solutions([dust, baseline], ahead) == [dust, baseline]
    # The same with the baseline passed as search_ranking passes it.
    assert rs.rank_solutions([dust], behind, baseline) == [baseline]
    assert rs.rank_solutions([dust], ahead, baseline) == [dust, baseline]


@pytest.mark.usefixtures("pruning_polish")
def test_a_polish_pruned_to_nothing_leaves_one_zero_burn(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX + 10.0, SCORE)
    result = _small_search(scenario, config, [1])
    zero = [s for s in result.solutions if not s.burn_count]
    assert len(zero) == 1 and zero[0] is result.baseline
    assert zero[0].label == "zero-burn"


def test_dust_and_the_zero_burn_are_no_margin_s_best(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    """An all-dust schedule prunes to the zero burn, which the baseline is."""
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX + 10.0, SCORE)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    baseline = rs.settle("zero-burn", "baseline", [], np.zeros((0, 3)), setup)
    dust = rs.settle("dust", "miss", [500.0], np.array([[0.0, 8.6e-6, 0.0]]), setup, 10.0)
    real = rs.settle("real", "miss", [500.0], np.array([[0.0, 0.1, 0.0]]), setup, 10.0)
    assert baseline is not None and dust is not None and real is not None
    result = rs.RankingSearchResult(setup, 3, [baseline, dust, real], baseline)
    assert rs.best_per_margin(result) == {10.0: real}


@pytest.mark.parametrize("margins", [(), (-1.0,), (float("nan"),)])
def test_a_bad_margin_is_refused(
    tmp_path: Path, no_maneuver_script: Path, margins: tuple[float, ...]
) -> None:
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    with pytest.raises(DynamicsError, match="margins"):
        rs.build_setup(scenario, config, force_model_margins_km=margins)


def test_repair_pulls_a_converged_schedule_onto_the_walls(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    limits = rs.RankingLimits.from_config(scenario, config)
    impulses = np.array([[CAP * (1.0 + 1.0e-9), 0.0, 0.0]] * 13)
    times, fixed = rs.repair([0.0, *(101.0 * k - 1.0e-7 for k in range(1, 13))], impulses, limits)
    assert not rs.wall_violations(times, fixed, limits)
    assert times[0] == limits.first_burn_floor_s
    assert float(np.sum(np.linalg.norm(fixed, axis=1))) <= 2.99 + 1.0e-12


def test_a_slotted_template_caps_the_burn_count(different_plane_script: Path) -> None:
    config = dataclasses.replace(
        _ranking_config(SCORE), pursuer_name=None, target_name=None
    )
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, horizon_s=33988.71424199411)
    )
    scenario, _ = parse_gmat_script(different_plane_script, config)
    limits = rs.RankingLimits.from_config(scenario, config)
    assert limits.max_burns == 2
    # One burn uses the last slot, after the first slot's zero vector.
    assert list(limits.slot_indices_for(1)) == [1]
    assert limits.first_burn_floor_for(1) == pytest.approx(1.0e-3 + 101.0)


# -- end to end


def test_finds_the_hit_that_is_in_reach(tmp_path: Path, no_maneuver_script: Path) -> None:
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    result = _small_search(scenario, config, [1, 2])
    assert result.solutions
    best = result.solutions[0]
    assert best.measurement.intercept_time_s is not None
    for solution in result.solutions:
        _independent_wall_check(
            list(solution.times_s), [np.array(i) for i in solution.impulses_km_s], scenario, config
        )
    candidate = rs.scored_candidate(best, scenario, config, result.setup.limits)
    assert apply_constraints(candidate, scenario, config).valid
    assert not candidate.warnings
    assert candidate.score is not None and candidate.score > 50.0
    # The epoch the search aimed at, which the GMAT correction aims at too.
    assert candidate.metadata["arrival_time_s"] == best.measurement.minimum_distance_time_s
    assert candidate.metadata["arrival_time_s"] >= candidate.intercept_time_s
    baseline = rs.scored_candidate(result.baseline, scenario, config, result.setup.limits)  # type: ignore[arg-type]
    assert baseline.score is not None and candidate.score > baseline.score


def test_returns_a_near_miss_when_no_hit_is_in_reach(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    """One impulse cannot lift the pursuer within 10 km of the target's radius."""
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX + 10.0, None)
    result = _small_search(scenario, config, [1])
    assert result.solutions and result.baseline is not None
    best = result.solutions[0]
    assert best.measurement.intercept_time_s is None
    assert best.measurement.minimum_distance_km >= 10.0 - 1.0e-6
    # Close enough to be worth most of the distance term: 100/(1 + e^{((d-5)/100)^2}).
    assert best.measurement.minimum_distance_km < 40.0
    assert best.measurement.minimum_distance_km < result.baseline.measurement.minimum_distance_km
    # A miss's script runs to T_max: its completion time is the horizon.
    candidate = rs.scored_candidate(best, scenario, config, result.setup.limits)
    assert candidate.metadata["arrival_time_s"] == result.setup.limits.horizon_s


#: Any count but one; 2, 3, 4 and 8 all change SLSQP's numerics here (measured).
CALLER_BLAS_THREADS = 4


@pytest.fixture
def busy_blas(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[Any, Any]]]:
    """The caller on CALLER_BLAS_THREADS OpenBLAS threads, restored afterwards.

    CI exports OPENBLAS_NUM_THREADS=1, so the caller already ran on one and the
    tests that use this could not fail. Forked, the workers start from it too.
    """
    controls = rs._openblas_thread_controls()
    before = [int(getter()) for _, getter in controls]
    for setter, _ in controls:
        setter(CALLER_BLAS_THREADS)
    try:
        assert [int(getter()) for _, getter in controls] == [CALLER_BLAS_THREADS] * len(controls)
        monkeypatch.setattr(rs, "_POOL_CONTEXT", multiprocessing.get_context("fork"))
        yield controls
    finally:
        for (setter, _), count in zip(controls, before, strict=True):
            setter(count)


def test_the_search_does_not_depend_on_the_worker_count(
    tmp_path: Path, no_maneuver_script: Path, busy_blas: list[tuple[Any, Any]]
) -> None:
    # Without OpenBLAS the caller's count cannot be raised, and a pass would say nothing.
    if not busy_blas:
        pytest.skip("no OpenBLAS here (Accelerate, MKL, or no /proc), no thread count to vary")
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX + 10.0, SCORE)
    serial = _small_search(scenario, config, [1, 2], workers=1)
    parallel = _small_search(scenario, config, [1, 2], workers=2)
    assert [s.to_dict() for s in serial.solutions] == [s.to_dict() for s in parallel.solutions]


def test_polishing_skips_an_empty_schedule_and_prunes_dust(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    empty = rs.settle("empty", "miss", [], np.zeros((0, 3)), setup)
    assert empty is not None and rs.polish_starts(empty) == []
    dusty = rs.settle(
        "dusty", "hit", [10.0, 200.0], np.array([[0.2, 0.0, 0.0], [0.0, 5.0e-4, 0.0]]), setup
    )
    assert dusty is not None
    polished, pruned = rs.polish_starts(dusty)
    assert polished.burn_count == 2 and pruned.burn_count == 1


@pytest.mark.parametrize("score", [SCORE, None])
def test_a_dust_burn_does_not_win_the_ordering(
    tmp_path: Path, no_maneuver_script: Path, score: ScoreConfig | None
) -> None:
    """Both hit, so the distance term ties and only the dust tells them apart."""
    target_km = R_MAX - 30.0
    scenario, config = _scenario(tmp_path, no_maneuver_script, target_km, score)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    # The Hohmann impulse the target was phased for.
    velocity = scenario.pursuer.initial_state.v
    speed = float(np.linalg.norm(velocity))
    needed = math.sqrt(2.0 * MU * target_km / (R_PURSUER * (R_PURSUER + target_km))) - speed
    hohmann = velocity / speed * needed
    one = rs.settle("one", "hit", [1.0e-3], hohmann[None, :], setup)
    dust = rs.settle(
        "dust", "hit", [1.0e-3, 400.0], np.array([hohmann, [0.0, 1.0e-6, 0.0]]), setup
    )
    assert one is not None and dust is not None and dust.burn_count == 2
    assert one.successful and dust.successful

    def key(solution: rs.RankingSolution) -> tuple[object, ...]:
        return rs.solution_key(solution, setup, scenario, config)

    assert sorted([dust, one], key=key)[0] is one


@pytest.mark.parametrize("segment_s", [1.0e-3, 300.0])
def test_every_returned_solution_renders(
    tmp_path: Path, no_maneuver_script: Path, segment_s: float
) -> None:
    """300 s is over the 101 s separation, which validate_config refuses; a
    review fed it in anyway and got schedules the renderer refused."""
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    config = dataclasses.replace(
        config,
        policy=dataclasses.replace(config.policy, min_propagate_duration_s=segment_s),
    )
    result = _small_search(scenario, config, [1, 2])
    assert len(result.solutions) > 3
    _, document = parse_gmat_script(Path(scenario.source_path), config)
    for solution in result.solutions:
        render_candidate(
            document, scenario, rs.scored_candidate(solution, scenario, config, result.setup.limits),
            config,
        )
        for before, after in itertools.pairwise(solution.times_s):
            assert after - before >= segment_s - 1.0e-9


def test_the_burn_window_keeps_every_segment_over_the_minimum(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    """A 300 s segment minimum over a 20 s final padding: the window has to
    end 300 s before T_max, and both ends keep the 1 us pad."""
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX + 10.0, SCORE)
    config = dataclasses.replace(
        config,
        policy=dataclasses.replace(
            config.policy, min_propagate_duration_s=300.0, final_padding_s=20.0
        ),
    )
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    limits = setup.limits
    assert limits.last_burn_ceiling_s < HORIZON_S - 300.0
    assert limits.first_burn_floor_for(1) > 300.0
    times = [limits.first_burn_floor_for(2), limits.last_burn_ceiling_s]
    solution = rs.settle("ends", "miss", times, np.array([[0.01, 0.0, 0.0]] * 2), setup)
    assert solution is not None and not solution.successful
    _, document = parse_gmat_script(Path(scenario.source_path), config)
    render_candidate(
        document, scenario, rs.scored_candidate(solution, scenario, config, limits), config
    )


def test_the_search_ranks_the_baseline_itself(tmp_path: Path, no_maneuver_script: Path) -> None:
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX + 10.0, SCORE)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    result = rs.search_ranking(scenario, config, setup, [], workers=1)
    assert result.baseline is not None and result.solutions == [result.baseline]


@pytest.mark.parametrize("score", [SCORE, None])
def test_spending_nothing_outranks_dust(
    tmp_path: Path, no_maneuver_script: Path, score: ScoreConfig | None
) -> None:
    """The R3 shape: a lone 1.4e-7 km/s burn must not be named best."""
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX + 10.0, score)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    baseline = rs.settle("zero-burn", "baseline", [], np.zeros((0, 3)), setup)
    dust = rs.settle("dust", "miss", [500.0], np.array([[0.0, 1.4e-7, 0.0]]), setup)
    assert baseline is not None and dust is not None and dust.burn_count == 1
    ranked = rs.rank_solutions(
        [dust, baseline], lambda s: rs.solution_key(s, setup, scenario, config)
    )
    assert ranked[0] is baseline
    assert rs.polish_starts(dust) == [rs.polish_starts(dust)[0]]


@pytest.mark.parametrize(
    "rules, setting",
    [("ranking", None), ("preliminary", None), ("ranking", "EarthMJ2000Eq"), ("ranking", "VNB")],
)
def test_added_burns_use_the_configured_frame(
    tmp_path: Path, no_maneuver_script: Path, rules: str, setting: str | None
) -> None:
    """VNB unless policy.synthesized_burn_frame opts in; the guard follows it."""
    from orbitwin.config import SYNTHESIZED_BURN_FRAMES, validate_config
    from orbitwin.errors import PolicyError
    from orbitwin.script_renderer import assert_final_script_policy, write_rendered_candidate

    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    policy, score = config.policy, config.score
    if rules == "preliminary":
        # The preliminary's distance term has no C_r, and its config refuses one.
        policy = dataclasses.replace(policy, rules=rules, horizon_s=None)
        score = dataclasses.replace(SCORE, cr_km=None)
    if setting is not None:
        policy = dataclasses.replace(policy, synthesized_burn_frame=setting)
    config = dataclasses.replace(config, policy=policy, score=score)
    validate_config(config)
    scenario, document = parse_gmat_script(Path(scenario.source_path), config)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    impulses = np.array([[0.1, 0.05, 0.0], [0.0, 0.2, 0.1]])
    solution = rs.settle("frame", "frame", [500.0, 601.0], impulses, setup)
    assert solution is not None
    candidate = rs.scored_candidate(solution, scenario, config, setup.limits)
    written = tmp_path / "candidate.script"
    write_rendered_candidate(
        written, tmp_path / "candidate.diff.txt", document, scenario, candidate, config
    )
    assert_final_script_policy(Path(scenario.source_path), written, config)
    frame = SYNTHESIZED_BURN_FRAMES[setting or "VNB"]
    assert f".CoordinateSystem = {frame[0]};" in written.read_text(encoding="utf-8")
    for burn in candidate.burns:
        assert burn.frame == frame
        if frame[0] == "EarthMJ2000Eq":
            assert burn.dv_script_km_s == pytest.approx(burn.dv_inertial_km_s, abs=1.0e-15)
    # The guard holds a script to the frame configured now.
    other = "VNB" if frame[2] != "VNB" else "EarthMJ2000Eq"
    flipped = dataclasses.replace(
        config, policy=dataclasses.replace(policy, synthesized_burn_frame=other)
    )
    with pytest.raises(PolicyError, match=r"CoordinateSystem|Axes"):
        assert_final_script_policy(Path(scenario.source_path), written, flipped)


def test_components_from_another_frame_setting_are_refused(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    """Right length, wrong vector: a record written inertially, rendered as VNB."""
    from orbitwin.errors import PolicyError
    from orbitwin.io import candidate_from_dict

    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    inertial = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, synthesized_burn_frame="EarthMJ2000Eq")
    )
    there, _ = parse_gmat_script(Path(scenario.source_path), inertial)
    setup = rs.build_setup(there, inertial, grid_step_s=60.0)
    solution = rs.settle("frame", "frame", [500.0], np.array([[0.1, 0.05, 0.0]]), setup)
    assert solution is not None
    record = rs.scored_candidate(solution, there, inertial, setup.limits).to_dict()
    loaded = candidate_from_dict(record)
    assert loaded.burns[0].frame == ("EarthMJ2000Eq", "Earth", "MJ2000Eq")
    _, document = parse_gmat_script(Path(scenario.source_path), config)
    with pytest.raises(PolicyError, match="written in EarthMJ2000Eq"):
        render_candidate(document, scenario, loaded, config)
    render_candidate(document, there, loaded, inertial)


@pytest.mark.parametrize(
    "written, read", [("EarthMJ2000Eq", "VNB"), ("VNB", "EarthMJ2000Eq"), ("VNB", "VNB")]
)
def test_a_record_without_a_frame_is_checked_against_its_impulse(
    tmp_path: Path, no_maneuver_script: Path, written: str, read: str
) -> None:
    """Records from before burns carried a frame: recomputed, not trusted."""
    from orbitwin.errors import PolicyError
    from orbitwin.io import candidate_from_dict

    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)

    def setting(frame: str) -> OrbitWinConfig:
        return dataclasses.replace(
            config, policy=dataclasses.replace(config.policy, synthesized_burn_frame=frame)
        )

    there, _ = parse_gmat_script(Path(scenario.source_path), setting(written))
    setup = rs.build_setup(there, setting(written), grid_step_s=60.0)
    impulses = np.array([[0.1, 0.05, 0.0], [0.0, -0.08, 0.12]])
    solution = rs.settle("old", "miss", [500.0, 1500.0], impulses, setup)
    assert solution is not None
    record = rs.scored_candidate(solution, there, setting(written), setup.limits).to_dict()
    for burn in record["burns"]:
        del burn["frame"]
    legacy = candidate_from_dict(record)
    here, document = parse_gmat_script(Path(scenario.source_path), setting(read))
    if written == read:
        render_candidate(document, here, legacy, setting(read))
    else:
        with pytest.raises(PolicyError, match="records no frame"):
            render_candidate(document, here, legacy, setting(read))


@pytest.mark.parametrize("value", ["vnb", "Inertial", 1, None])
def test_an_unknown_burn_frame_setting_is_refused(value: object) -> None:
    from orbitwin.config import validate_config
    from orbitwin.errors import ConfigurationError

    config = _ranking_config(SCORE)
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, synthesized_burn_frame=value)
    )
    with pytest.raises(ConfigurationError, match="synthesized_burn_frame"):
        validate_config(config)


def test_slot_components_mean_the_impulse_on_the_pre_burn_state(
    different_plane_script: Path,
) -> None:
    """DifferentPlane's slots are Local/VNB, which turn with the velocity."""
    from orbitwin.frames import script_components_to_inertial
    from orbitwin.models import burn_frame

    config = dataclasses.replace(_ranking_config(SCORE), pursuer_name=None, target_name=None)
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, horizon_s=33988.71424199411)
    )
    scenario, _ = parse_gmat_script(different_plane_script, config)
    setup = rs.build_setup(scenario, config, grid_step_s=600.0)
    impulses = np.array([[0.2, 0.1, 0.0], [0.0, 0.0, 0.2]])
    solution = rs.settle("slots", "slots", [1000.0, 1500.0], impulses, setup)
    assert solution is not None
    candidate = rs.scored_candidate(solution, scenario, config, setup.limits)
    flight = rs.fly(solution.times_s, impulses, scenario.pursuer.initial_state, MU)
    for index, burn in enumerate(candidate.burns):
        before = StateVector(
            as_vector3(flight.r_at_burns[index]), as_vector3(flight.v_before[index])
        )
        recovered = script_components_to_inertial(
            np.asarray(burn.dv_script_km_s), before, *burn_frame(scenario, burn.slot_index)
        )
        assert recovered == pytest.approx(np.asarray(burn.dv_inertial_km_s), abs=1.0e-12)


def test_the_callers_blas_threads_are_left_alone(
    tmp_path: Path, no_maneuver_script: Path, busy_blas: list[tuple[Any, Any]]
) -> None:
    if not busy_blas:
        pytest.skip("no OpenBLAS here (Accelerate, MKL, or no /proc), nothing to leave alone")
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX + 10.0, SCORE)
    _small_search(scenario, config, [1], workers=1)
    assert [int(getter()) for _, getter in busy_blas] == [CALLER_BLAS_THREADS] * len(busy_blas)


def _die_in_the_worker(job: object) -> object:
    import os

    os._exit(3)


def test_a_dead_worker_is_reported_as_a_parallel_search_error(
    tmp_path: Path, no_maneuver_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orbitwin.search import ParallelSearchError

    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX + 10.0, SCORE)
    setup = rs.build_setup(scenario, config, screen_iterations=5, grid_step_s=60.0)
    start = rs.random_starts(1, setup, 1, 3)[0]
    monkeypatch.setattr(rs, "_refine_in_worker", _die_in_the_worker)
    with pytest.raises(ParallelSearchError, match="worker process died"):
        rs._refine_all([(start, 5)], setup, 1)


def test_a_missing_blas_control_is_said(
    tmp_path: Path, no_maneuver_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX + 10.0, SCORE)
    setup = rs.build_setup(scenario, config, screen_iterations=5, grid_step_s=60.0)
    start = rs.random_starts(1, setup, 1, 3)[0]
    monkeypatch.setattr(rs, "_openblas_thread_controls", list)
    with pytest.warns(RuntimeWarning, match="OpenBLAS"):
        rs._refine_all([(start, 5)], setup, 1)


def test_a_candidate_renders_with_both_frames_equal(
    tmp_path: Path, no_maneuver_script: Path
) -> None:
    scenario, config = _scenario(tmp_path, no_maneuver_script, R_MAX - 30.0, SCORE)
    setup = rs.build_setup(scenario, config, grid_step_s=60.0)
    times = [500.0, 601.0, 900.0]
    impulses = np.array([[0.1, 0.05, 0.0], [0.0, 0.2, 0.1], [-0.1, 0.0, 0.2]])
    solution = rs.settle("frames", "frames", times, impulses, setup)
    assert solution is not None
    candidate = rs.scored_candidate(solution, scenario, config, setup.limits)
    for burn in candidate.burns:
        script = float(np.linalg.norm(burn.dv_script_km_s))
        assert script == pytest.approx(burn.magnitude_km_s, abs=1.0e-9)
    _, document = parse_gmat_script(Path(scenario.source_path), config)
    render_candidate(document, scenario, candidate, config)


def test_the_burn_count_is_what_the_window_holds_not_total_over_cap() -> None:
    """14 burns of 0.1 km/s 101 s apart spend 1.4 km/s, inside every rule; ceil(2.99 /
    0.249) = 13 once refused them. What bounds the count is the window."""
    limits = rs.RankingLimits(
        burn_cap_km_s=0.249, total_cap_km_s=2.99, separation_s=101.0,
        first_burn_floor_s=0.001001, last_burn_ceiling_s=3980.0,
        radius_floor_km=6478.1363, horizon_s=4000.0, intercept_radius_km=5.0, slot_count=None,
    )
    assert limits.max_burns == 1 + int((3980.0 - 0.001001) // 101.0) == 40
    times = [1.0 + 101.0 * k for k in range(14)]
    impulses = np.array([[0.1, 0.0, 0.0]] * 14)
    assert rs.wall_violations(times, impulses, limits) == []
    crowded = [0.001001 + 101.0 * k for k in range(41)]
    found = rs.wall_violations(crowded, np.zeros((41, 3)), limits)
    assert "41 burns, more than 40" in found


def test_a_preliminary_total_does_not_cap_the_count_at_a_constant() -> None:
    limits = rs.RankingLimits(
        burn_cap_km_s=1.495, total_cap_km_s=None, separation_s=101.0,
        first_burn_floor_s=0.001001, last_burn_ceiling_s=20000.0,
        radius_floor_km=6478.1363, horizon_s=20020.0, intercept_radius_km=5.0, slot_count=None,
    )
    assert limits.max_burns == 1 + int((20000.0 - 0.001001) // 101.0)
