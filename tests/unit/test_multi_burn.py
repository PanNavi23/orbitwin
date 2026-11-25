"""The N-impulse search, its walls, and the schedule it found on 2026-09-05.

`docs/LESSONS_2026-09-05.md` records a result -- 98.32, six impulses, replayed
under real GMAT -- that nothing in the repository could reproduce, because the
search that produced it lived in a scratch directory. The first two tests make
that number a fact about this code rather than a claim in a document; they need
the organizer's problem and skip in a tree without it.

Every other test holds the search's walls on a scenario of our own: the
preliminary's rules and coefficients on examples/synthetic/NoManeuver_Circular.script,
an open template as the preliminary's was, and a schedule this search found there
(tests/fixtures/multi_burn_schedule.json, which says how).
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from conftest import organizer_file

from orbitwin.config import OrbitWinConfig, load_config
from orbitwin.errors import DynamicsError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Scenario, StateVector, as_vector3
from orbitwin.multi_burn import (
    MultiBurnLimits,
    MultiBurnStart,
    default_burn_counts,
    fly,
    minimum_radius_km,
    objective,
    refine_start,
    score_schedule,
    search_multi_burn,
    stable_seed,
    start_from_solution,
    structured_starts,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "odc2026_best_multi_burn.json"
OWN_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "multi_burn_schedule.json"


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schedule() -> dict:
    return json.loads(OWN_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def own_config(odc_config: OrbitWinConfig) -> OrbitWinConfig:
    """The preliminary's walls and coefficients; the open template names no pursuer."""
    return dataclasses.replace(odc_config, pursuer_name="Pursuer", target_name="AlienSC")


@pytest.fixture(scope="module")
def own_scenario(no_maneuver_script: Path, own_config: OrbitWinConfig,
                 schedule: dict) -> Scenario:
    import hashlib

    assert hashlib.sha256(no_maneuver_script.read_bytes()).hexdigest() == (
        schedule["_scenario_sha256"]), "the schedule was found on another script"
    scenario, _ = parse_gmat_script(str(no_maneuver_script), own_config)
    return scenario


@pytest.fixture(scope="module")
def own_limits(own_scenario: Scenario, own_config: OrbitWinConfig) -> MultiBurnLimits:
    return MultiBurnLimits.from_config(own_scenario, own_config)


@pytest.fixture(scope="module")
def odc_config(project_root: Path) -> OrbitWinConfig:
    return load_config(str(project_root / "config" / "history" / "preliminary" / "odc2026.yaml"))


@pytest.fixture(scope="module")
def odc_scenario(project_root: Path, odc_config: OrbitWinConfig) -> Scenario:
    script = organizer_file("preliminary-problem")
    scenario, _ = parse_gmat_script(str(script), odc_config)
    return scenario


@pytest.fixture(scope="module")
def odc_limits(odc_scenario: Scenario, odc_config: OrbitWinConfig) -> MultiBurnLimits:
    return MultiBurnLimits.from_config(odc_scenario, odc_config)


def test_the_recorded_schedule_still_scores_what_it_scored(
    recorded: dict,
    odc_scenario: Scenario,
    odc_config: OrbitWinConfig,
    odc_limits: MultiBurnLimits,
) -> None:
    """98.3171845866, six impulses, 6.1839784861 km/s, arriving at 5676.6416 s.

    LESSONS reports 98.3185410 for the same schedule. The difference is not a
    disagreement: that figure is scored at the interception time, which is when
    the trajectory first enters the 5 km sphere, and this one is scored at the
    arrival epoch, when it reaches the target exactly. Entering the sphere
    happens 0.4 s earlier, and 0.4 s is worth 0.0014 points here.
    """
    start = start_from_solution(recorded, odc_limits, label="recorded")
    flown = fly(
        np.asarray(start.vector, dtype=float),
        start.burn_count,
        odc_scenario,
        odc_config,
        odc_limits,
    )
    assert flown is not None, "the recorded schedule no longer flies"
    burns, total_dv_km_s, smallest_radius_km, arrival_s = flown

    assert len(burns) == recorded["n_burns"] == 6
    assert total_dv_km_s == pytest.approx(recorded["total_dv_km_s"], abs=1.0e-9)
    assert arrival_s == pytest.approx(recorded["arrival_s"], abs=1.0e-6)
    assert smallest_radius_km == pytest.approx(recorded["minimum_radius_km"], abs=1.0e-6)

    score = score_schedule(burns, arrival_s, odc_scenario, odc_config)
    assert score == pytest.approx(recorded["score"], abs=1.0e-9)


def _inside_every_wall(found: dict, config: OrbitWinConfig, limits: MultiBurnLimits) -> None:
    burns = found["burns"]
    magnitudes = [
        float(np.linalg.norm(np.asarray(burn["dv_inertial_km_s"], dtype=float)))
        for burn in burns
    ]
    gaps = [burns[i + 1]["time_s"] - burns[i]["time_s"] for i in range(len(burns) - 1)]

    assert max(magnitudes) <= limits.burn_cap_km_s + 1.0e-12
    assert max(magnitudes) <= config.policy.dv_limit_km_s
    assert min(gaps) >= limits.separation_s - 1.0e-9
    assert min(gaps) >= config.policy.min_burn_separation_s
    assert burns[0]["time_s"] >= config.policy.min_propagate_duration_s
    assert found["minimum_radius_km"] > config.policy.central_body_radius_km


def test_the_recorded_schedule_is_inside_every_wall(
    recorded: dict, odc_config: OrbitWinConfig, odc_limits: MultiBurnLimits
) -> None:
    """Not just legal -- inside the configured margins, which are tighter.

    LESSONS §5 measured what shaving those margins buys: 0.0016 points, for a
    fivefold cut in headroom against a ten-point-per-maneuver deduction.
    """
    _inside_every_wall(recorded, odc_config, odc_limits)


def test_our_schedule_is_inside_every_wall(
    schedule: dict, own_config: OrbitWinConfig, own_limits: MultiBurnLimits
) -> None:
    """The tests below take it as a legal schedule to push against each wall."""
    _inside_every_wall(schedule, own_config, own_limits)
    assert schedule["minimum_radius_km"] >= own_limits.radius_floor_km


def test_our_schedule_still_flies_what_it_was_found_with(
    schedule: dict,
    own_scenario: Scenario,
    own_config: OrbitWinConfig,
    own_limits: MultiBurnLimits,
) -> None:
    """The fixture's numbers are what search_multi_burn returned for it (its _provenance)."""
    start = start_from_solution(schedule, own_limits, label="schedule")
    flown = fly(np.asarray(start.vector, dtype=float), start.burn_count, own_scenario,
                own_config, own_limits)
    assert flown is not None, "our schedule no longer flies"
    burns, total_dv_km_s, smallest_radius_km, arrival_s = flown

    assert len(burns) == schedule["n_burns"] == 5
    assert total_dv_km_s == pytest.approx(schedule["total_dv_km_s"], abs=1.0e-9)
    assert arrival_s == pytest.approx(schedule["arrival_s"], abs=1.0e-6)
    assert smallest_radius_km == pytest.approx(schedule["minimum_radius_km"], abs=1.0e-6)
    score = score_schedule(burns, arrival_s, own_scenario, own_config)
    assert score == pytest.approx(schedule["score"], abs=1.0e-9)


def test_an_impulse_exactly_on_the_cap_is_not_rejected(
    schedule: dict,
    own_scenario: Scenario,
    own_config: OrbitWinConfig,
    own_limits: MultiBurnLimits,
) -> None:
    """Two of the five impulses sit exactly on the cap, and must be allowed.

    `_unpack` clamps a magnitude to the cap and multiplies it by a unit vector
    built from two angles; that vector's norm is 1 only to rounding, so the
    result measures a few times 1e-16 above the cap. A strict `>` comparison
    rejected the whole schedule and `objective` returned the barrier for the
    best answer this project had (the 2026-09-05 one, four of six on the cap).
    """
    at_cap = [
        magnitude
        for magnitude in (
            float(np.linalg.norm(np.asarray(burn["dv_inertial_km_s"], dtype=float)))
            for burn in schedule["burns"]
        )
        if math.isclose(magnitude, own_limits.burn_cap_km_s, rel_tol=0.0, abs_tol=1.0e-9)
    ]
    assert len(at_cap) == 2, "the fixture no longer exercises the boundary"

    start = start_from_solution(schedule, own_limits, label="schedule")
    flown = fly(np.asarray(start.vector, dtype=float), start.burn_count, own_scenario,
                own_config, own_limits)
    assert flown is not None
    assert max(float(np.linalg.norm(impulse)) for _, impulse in flown[0]) > (
        own_limits.burn_cap_km_s), "no impulse flies over the cap by rounding: nothing is tested"
    value = objective(
        np.asarray(start.vector, dtype=float),
        start.burn_count,
        own_scenario,
        own_config,
        own_limits,
    )
    assert value == pytest.approx(-schedule["score"], abs=1.0e-9)


def test_an_impulse_over_the_cap_is_rejected(
    schedule: dict,
    own_scenario: Scenario,
    own_config: OrbitWinConfig,
    own_limits: MultiBurnLimits,
) -> None:
    """The tolerance is for rounding, not for margin."""
    start = start_from_solution(schedule, own_limits, label="schedule")
    vector = np.asarray(start.vector, dtype=float)
    # The magnitude of the second impulse, the first on the cap: two epochs, then
    # n-1 gaps, then (polar, azimuth, magnitude) per impulse.
    magnitude_index = 2 + (start.burn_count - 1) + 3 + 2
    assert vector[magnitude_index] == pytest.approx(own_limits.burn_cap_km_s)

    # _unpack clamps, so pushing the decision variable over the cap cannot make
    # the flown impulse illegal. Raising the wall's own value is what shows the
    # check is live: with a cap below what the schedule uses, it must refuse.
    lowered = MultiBurnLimits(
        burn_cap_km_s=1.0,
        separation_s=own_limits.separation_s,
        first_burn_floor_s=own_limits.first_burn_floor_s,
        radius_floor_km=own_limits.radius_floor_km,
        horizon_s=own_limits.horizon_s,
    )
    value = objective(vector, start.burn_count, own_scenario, own_config, lowered)
    assert value > 0.0, "a schedule needing more than the cap must not score"


@pytest.mark.parametrize("total_cap_km_s, scores", [(5.6, False), (5.75, True), (None, True)])
def test_the_total_delta_v_is_a_wall_too(
    schedule: dict,
    own_scenario: Scenario,
    own_config: OrbitWinConfig,
    own_limits: MultiBurnLimits,
    total_cap_km_s: float | None,
    scores: bool,
) -> None:
    """Ranking rules §4 note 1(c). Our schedule spends 5.720 km/s."""
    start = start_from_solution(schedule, own_limits, label="schedule")
    walls = dataclasses.replace(own_limits, total_cap_km_s=total_cap_km_s)
    value = objective(
        np.asarray(start.vector, dtype=float), start.burn_count, own_scenario, own_config, walls
    )
    assert (value < 0.0) is scores


def test_the_total_wall_comes_from_the_policy(
    schedule: dict, own_scenario: Scenario, own_config: OrbitWinConfig
) -> None:
    policy = dataclasses.replace(
        own_config.policy, total_dv_limit_km_s=5.7, total_dv_safe_limit_km_s=5.6
    )
    config = dataclasses.replace(own_config, policy=policy)
    walls = MultiBurnLimits.from_config(own_scenario, config)
    start = start_from_solution(schedule, walls, label="schedule")
    value = objective(
        np.asarray(start.vector, dtype=float), start.burn_count, own_scenario, config, walls
    )
    assert walls.total_cap_km_s == 5.6 and value > 0.0


def test_a_schedule_that_dives_below_the_floor_is_rejected(
    schedule: dict,
    own_scenario: Scenario,
    own_config: OrbitWinConfig,
    own_limits: MultiBurnLimits,
) -> None:
    """LESSONS §4: point-mass gravity has no ground, and the optimizer knows.

    Every cheap Lambert arc on the 2026-09-05 scenario passed through the Earth.
    Raising the floor above the trajectory must refuse it, or the wall is
    decorative.
    """
    start = start_from_solution(schedule, own_limits, label="schedule")
    impossible = MultiBurnLimits(
        burn_cap_km_s=own_limits.burn_cap_km_s,
        separation_s=own_limits.separation_s,
        first_burn_floor_s=own_limits.first_burn_floor_s,
        radius_floor_km=schedule["minimum_radius_km"] + 1.0,
        horizon_s=own_limits.horizon_s,
    )
    value = objective(
        np.asarray(start.vector, dtype=float),
        start.burn_count,
        own_scenario,
        own_config,
        impossible,
    )
    assert value > 0.0


def test_a_schedule_past_the_horizon_does_not_fly(
    schedule: dict,
    own_scenario: Scenario,
    own_config: OrbitWinConfig,
    own_limits: MultiBurnLimits,
) -> None:
    start = start_from_solution(schedule, own_limits, label="schedule")
    vector = np.asarray(start.vector, dtype=float)
    vector[1] = own_limits.horizon_s + 1.0
    assert fly(vector, start.burn_count, own_scenario, own_config, own_limits) is None


def test_gaps_are_clamped_up_to_the_separation_rule(
    schedule: dict,
    own_scenario: Scenario,
    own_config: OrbitWinConfig,
    own_limits: MultiBurnLimits,
) -> None:
    """Asking for an illegal gap gives a legal one, not an illegal schedule."""
    start = start_from_solution(schedule, own_limits, label="schedule")
    vector = np.asarray(start.vector, dtype=float)
    vector[2] = 1.0  # first gap, far below the rule
    flown = fly(vector, start.burn_count, own_scenario, own_config, own_limits)
    assert flown is not None
    burns = flown[0]
    gaps = [burns[i + 1][0] - burns[i][0] for i in range(len(burns) - 1)]
    assert min(gaps) >= own_limits.separation_s - 1.0e-9


def test_scoring_refuses_without_the_announced_coefficients(
    schedule: dict, own_scenario: Scenario, own_config: OrbitWinConfig, own_limits: MultiBurnLimits
) -> None:
    """The coefficients are entered, never guessed."""
    start = start_from_solution(schedule, own_limits, label="schedule")
    flown = fly(
        np.asarray(start.vector, dtype=float),
        start.burn_count,
        own_scenario,
        own_config,
        own_limits,
    )
    assert flown is not None
    burns, _, _, arrival_s = flown

    import dataclasses

    without = dataclasses.replace(own_config, score=None)
    with pytest.raises(DynamicsError, match="coefficients"):
        score_schedule(burns, arrival_s, own_scenario, without)


def test_stable_seed_does_not_move_between_processes() -> None:
    """`hash()` on a str is salted per interpreter; this must not be.

    The scratch search seeded each job's jitter with `abs(hash(label))`, so the
    answer depended on which process picked the job up. Seeds must be
    deterministic.
    """
    import subprocess
    import sys

    expected = stable_seed("n6_coast2_t0")
    program = (
        "from orbitwin.multi_burn import stable_seed; print(stable_seed('n6_coast2_t0'))"
    )
    for salt in ("0", "1", "random"):
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": salt, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        )
        assert int(completed.stdout.strip()) == expected


def test_the_start_set_is_a_function_of_its_arguments(own_limits: MultiBurnLimits) -> None:
    first = structured_starts((5, 6), own_limits, seed=2026)
    second = structured_starts((6, 5), own_limits, seed=2026)
    assert [start.label for start in first] == [start.label for start in second]
    assert [start.vector for start in first] == [start.vector for start in second]

    different = structured_starts((5, 6), own_limits, seed=2027)
    assert [start.vector for start in different] != [start.vector for start in first]

    counts = {start.burn_count for start in first}
    assert counts == {5, 6}
    assert len({start.label for start in first}) == len(first), "labels must be unique"


@pytest.mark.parametrize("cap_km_s", [0.249, 1.495])
def test_random_starts_draw_their_impulses_under_any_cap(cap_km_s: float) -> None:
    """A 0.6 km/s floor made the ranking round's 0.249 cap raise ValueError."""
    limits = MultiBurnLimits(
        burn_cap_km_s=cap_km_s,
        separation_s=101.0,
        first_burn_floor_s=1.0e-3,
        radius_floor_km=6478.136,
        horizon_s=30000.0,
    )
    starts = structured_starts((3,), limits, seed=2026)
    # Three burns: two epochs, two gaps, then (polar, azimuth, magnitude) twice.
    magnitudes = [m for start in starts if "rand" in start.label for m in start.vector[6::3]]
    assert len(magnitudes) == 6 * 2
    assert min(magnitudes) >= 0.4 * cap_km_s
    assert max(magnitudes) <= cap_km_s


def test_the_start_set_puts_one_long_coast_in_every_slot(
    own_limits: MultiBurnLimits,
) -> None:
    """The winning shape was one long coast, not evenly spaced impulses."""
    starts = structured_starts((6,), own_limits, seed=2026, random_starts=0)
    coasts = set()
    for start in starts:
        gaps = list(start.vector[2 : 2 + 5])
        coasts.add(gaps.index(max(gaps)))
    assert coasts == {0, 1, 2, 3, 4}


def _arriving_late(schedule: dict, limits: MultiBurnLimits) -> MultiBurnStart:
    """Our schedule, 20 s later at the target: feasible, and not where the simplex stops."""
    seeded = start_from_solution(schedule, limits, label="schedule")
    late = [*seeded.vector]
    late[1] += 20.0
    return MultiBurnStart(label="schedule-late", burn_count=seeded.burn_count, vector=tuple(late))


def test_refinement_is_reproducible(
    schedule: dict, own_scenario: Scenario, own_config: OrbitWinConfig, own_limits: MultiBurnLimits
) -> None:
    """Same start, same answer, however many times it is run.

    A small iteration budget: this asserts that two runs agree, which a
    half-converged simplex shows just as well as a converged one and in a
    fraction of the time. The start is our schedule arriving 20 s late, so the
    simplex has somewhere to go, and its restarts, thrown by the jitter's
    generator, improve on the first. A structured start, which this used, never
    reached the feasible set in this budget, and two Nones agree whatever the seed.
    """
    start = _arriving_late(schedule, own_limits)
    budget = {"restarts": 4, "max_iterations": 60}
    first = refine_start(start, own_scenario, own_config, own_limits, **budget)
    second = refine_start(start, own_scenario, own_config, own_limits, **budget)
    assert first is not None and second is not None
    assert first.score == pytest.approx(second.score, abs=0.0)
    assert first.arrival_s == pytest.approx(second.arrival_s, abs=0.0)
    once = refine_start(start, own_scenario, own_config, own_limits, restarts=1,
                        max_iterations=60)
    assert once is not None and first.score > once.score, (
        "the jittered restarts found nothing better, so their seed is not tested")


def test_worker_count_does_not_change_the_answer(
    schedule: dict, own_scenario: Scenario, own_config: OrbitWinConfig, own_limits: MultiBurnLimits
) -> None:
    """Parallelism is an implementation detail, not an input.

    The late start's restarts, thrown by the generator a worker seeds from the
    start's label, improve on its first (the test above), so a seed that moved
    with the process would change the answer here.
    """
    seeded = start_from_solution(schedule, own_limits, label="schedule")
    starts = [
        seeded,
        MultiBurnStart(label="schedule-twin", burn_count=seeded.burn_count, vector=seeded.vector),
        _arriving_late(schedule, own_limits),
    ]
    budget = {"restarts": 4, "max_iterations": 40}
    sequential = search_multi_burn(
        own_scenario, own_config, starts=starts, workers=1, **budget
    )
    parallel = search_multi_burn(own_scenario, own_config, starts=starts, workers=2, **budget)

    assert [s.label for s in sequential] == [s.label for s in parallel]
    assert [s.score for s in sequential] == [s.score for s in parallel]


def test_results_come_back_best_first_with_a_stable_tie_break(
    schedule: dict, own_scenario: Scenario, own_config: OrbitWinConfig, own_limits: MultiBurnLimits
) -> None:
    """The late start, given first, refines to a lower score than the twins, which tie."""
    seeded = start_from_solution(schedule, own_limits, label="b-schedule")
    twin = MultiBurnStart(label="a-schedule", burn_count=seeded.burn_count, vector=seeded.vector)
    found = search_multi_burn(
        own_scenario,
        own_config,
        starts=[_arriving_late(schedule, own_limits), seeded, twin],
        workers=1,
        restarts=1,
        max_iterations=40,
    )
    assert [s.score for s in found] == sorted((s.score for s in found), reverse=True)
    assert len({s.score for s in found}) == 2, "nothing for the order to put first"
    if len(found) == 3 and found[0].score == found[1].score:
        assert [s.label for s in found] == ["a-schedule", "b-schedule", "schedule-late"]


def test_minimum_radius_finds_a_periapsis_inside_the_segment(
    odc_config: OrbitWinConfig,
) -> None:
    """A brief deep periapsis is exactly what sampling would step over."""
    mu = odc_config.policy.mu_km3_s2
    apoapsis_km = 20_000.0
    periapsis_km = 7_000.0
    semi_major_km = 0.5 * (apoapsis_km + periapsis_km)
    speed_at_apoapsis = math.sqrt(mu * (2.0 / apoapsis_km - 1.0 / semi_major_km))
    at_apoapsis = StateVector(
        as_vector3([apoapsis_km, 0.0, 0.0]), as_vector3([0.0, speed_at_apoapsis, 0.0])
    )
    period_s = 2.0 * math.pi * math.sqrt(semi_major_km**3 / mu)

    # Half a period from apoapsis is periapsis exactly.
    assert minimum_radius_km(at_apoapsis, period_s * 0.55, mu) == pytest.approx(
        periapsis_km, rel=1.0e-9
    )
    # A short arc away from periapsis must not report it.
    assert minimum_radius_km(at_apoapsis, period_s * 0.05, mu) > periapsis_km


@pytest.mark.parametrize("eccentricity", [1.01, 1.6, 3.0])
def test_minimum_radius_finds_the_periapsis_of_an_open_arc(eccentricity: float) -> None:
    """Forty samples stepped over it: 731 s before periapsis, the grid lands at 700 and 750."""
    from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, propagate_universal

    mu = 398600.4418
    periapsis_km = 6578.0
    elements = KeplerianElements(periapsis_km / (1.0 - eccentricity), eccentricity, 30, 20, 10, 0)
    at_periapsis = keplerian_to_cartesian(elements, mu)
    inbound = propagate_universal(at_periapsis, -731.234, mu)
    assert minimum_radius_km(inbound, 2000.0, mu) == pytest.approx(periapsis_km, abs=1.0e-6)

    # Wholly inbound: the end is lowest. Wholly outbound: the start is.
    before = propagate_universal(at_periapsis, -2000.0, mu)
    end = propagate_universal(before, 1000.0, mu)
    assert minimum_radius_km(before, 1000.0, mu) == pytest.approx(
        float(np.linalg.norm(end.r)), abs=1.0e-6
    )
    after = propagate_universal(at_periapsis, 100.0, mu)
    assert minimum_radius_km(after, 1000.0, mu) == pytest.approx(
        float(np.linalg.norm(after.r)), abs=1.0e-9
    )


def test_a_near_radial_ellipse_is_not_taken_for_an_open_arc() -> None:
    """e rounds to 1.0 here, yet the orbit is bound and falls through the Earth."""
    mu = 398600.4418
    state = StateVector(as_vector3([10000.0, 0.0, 0.0]), as_vector3([3.0, 1.0e-8, 0.0]))
    assert 0.5 * 9.0 - mu / 10000.0 < 0.0
    assert minimum_radius_km(state, 4000.0, mu) < 6378.0


def test_minimum_radius_of_a_standing_still_segment_is_where_it_stands(
    own_config: OrbitWinConfig, own_scenario: Scenario
) -> None:
    state = own_scenario.pursuer.initial_state
    here = float(np.linalg.norm(np.asarray(state.r, dtype=float)))
    assert minimum_radius_km(state, 0.0, own_config.policy.mu_km3_s2) == pytest.approx(here)


def test_the_last_burn_actually_closes_the_leg(
    schedule: dict, own_scenario: Scenario, own_config: OrbitWinConfig, own_limits: MultiBurnLimits
) -> None:
    """The whole structure rests on burn N being a Lambert completion.

    If it is not, the search is optimising a score for an interception that
    does not happen -- the failure mode `score_schedule`'s fixed 5 km distance
    would hide.
    """
    from orbitwin.orbit import propagate_universal

    start = start_from_solution(schedule, own_limits, label="schedule")
    flown = fly(
        np.asarray(start.vector, dtype=float),
        start.burn_count,
        own_scenario,
        own_config,
        own_limits,
    )
    assert flown is not None
    burns, _, _, arrival_s = flown

    mu = own_config.policy.mu_km3_s2
    state = own_scenario.pursuer.initial_state
    now = 0.0
    for time_s, impulse in burns:
        if time_s > now:
            state = propagate_universal(state, time_s - now, mu)
            now = time_s
        state = StateVector(
            as_vector3(state.r), as_vector3(np.asarray(state.v, dtype=float) + impulse)
        )
    final = propagate_universal(state, arrival_s - now, mu)
    target = propagate_universal(own_scenario.target.initial_state, arrival_s, mu)
    miss_km = float(
        np.linalg.norm(np.asarray(final.r, dtype=float) - np.asarray(target.r, dtype=float))
    )
    assert miss_km < own_config.search.endpoint_tolerance_km


def test_script_components_are_converted_on_the_flown_state(
    schedule: dict, own_scenario: Scenario, own_config: OrbitWinConfig, own_limits: MultiBurnLimits
) -> None:
    """The default frame is Local/Earth/VNB, whose basis follows the velocity.

    By the second impulse that velocity already carries the first, so
    converting against the unmaneuvered trajectory writes components that mean
    a different vector -- and GMAT would fly the different one in silence. The
    round trip through the flown state is what says the components mean what
    the search decided.
    """
    from orbitwin.frames import script_components_to_inertial
    from orbitwin.models import burn_frame
    from orbitwin.multi_burn import as_candidate
    from orbitwin.orbit import propagate_universal

    assert burn_frame(own_scenario, 0) == ("Local", "Earth", "VNB"), (
        "this assertion only means something in a velocity-dependent frame"
    )

    start = start_from_solution(schedule, own_limits, label="schedule")
    flown = fly(
        np.asarray(start.vector, dtype=float),
        start.burn_count,
        own_scenario,
        own_config,
        own_limits,
    )
    assert flown is not None
    burns = flown[0]
    candidate = as_candidate(burns, flown[3], own_scenario, own_config)

    mu = own_config.policy.mu_km3_s2
    state = own_scenario.pursuer.initial_state
    now = 0.0
    unmaneuvered_differs = False
    for index, burn in enumerate(candidate.burns):
        if burn.time_s > now:
            state = propagate_universal(state, burn.time_s - now, mu)
            now = burn.time_s

        recovered = script_components_to_inertial(
            np.asarray(burn.dv_script_km_s, dtype=float), state, *burn_frame(own_scenario, index)
        )
        expected = np.asarray(burn.dv_inertial_km_s, dtype=float)
        assert np.allclose(recovered, expected, atol=1.0e-12), (
            f"burn {index + 1} does not round trip through its own frame"
        )

        if index > 0:
            drifting = propagate_universal(
                own_scenario.pursuer.initial_state, burn.time_s, mu
            )
            wrong = script_components_to_inertial(
                np.asarray(burn.dv_script_km_s, dtype=float),
                drifting,
                *burn_frame(own_scenario, index),
            )
            if not np.allclose(wrong, expected, atol=1.0e-6):
                unmaneuvered_differs = True

        state = StateVector(
            as_vector3(state.r),
            as_vector3(np.asarray(state.v, dtype=float) + np.asarray(burn.dv_inertial_km_s)),
        )

    assert unmaneuvered_differs, (
        "the unmaneuvered state gives the same answer here, so this test would "
        "not have caught the bug it exists for"
    )


def test_a_solution_reports_the_delta_v_its_burns_carry(
    schedule: dict, own_scenario: Scenario, own_config: OrbitWinConfig, own_limits: MultiBurnLimits
) -> None:
    """Two sources for one number, so they are checked against each other.

    `fly` sums the impulse magnitudes and `MultiBurnSolution` stores that;
    the burns it carries derive their own. `Candidate.total_dv_km_s` exists as
    a property because those two were once allowed to disagree -- a candidate
    could declare 99 km/s while carrying 0.5 -- and the same shape is here.
    """
    seeded = start_from_solution(schedule, own_limits, label="schedule")
    found = search_multi_burn(
        own_scenario,
        own_config,
        starts=[seeded],
        workers=1,
        restarts=1,
        max_iterations=40,
    )
    assert found, "the schedule must survive its own refinement"
    for solution in found:
        from_burns = sum(burn.magnitude_km_s for burn in solution.burns)
        assert solution.total_dv_km_s == pytest.approx(from_burns, abs=1.0e-12)
        assert solution.burn_count == len(solution.burns) == schedule["n_burns"]
        assert solution.to_dict()["total_dv_km_s"] == solution.total_dv_km_s


def test_the_barrier_slopes_toward_the_feasible_set(
    schedule: dict, own_scenario: Scenario, own_config: OrbitWinConfig, own_limits: MultiBurnLimits
) -> None:
    """A hard wall, but not a plateau.

    Every infeasible schedule must score above every feasible one -- that is
    the wall, and it is what stops an optimizer trading a violation for score.
    But a barrier returning one constant gives Nelder-Mead nothing to walk
    down, and on the 2026-09-05 scenario all 21 structured starts began
    infeasible: the Lambert completion they needed was 3 to 11 km/s against a
    1.495 km/s cap. With a flat barrier the search could only reproduce a seeded
    answer, never find one.

    The violation is measured on the radius floor rather than the burn cap,
    because `_unpack` clamps magnitudes: lowering the cap does not make this
    schedule illegal, it makes a different legal one.
    """
    start = start_from_solution(schedule, own_limits, label="schedule")
    vector = np.asarray(start.vector, dtype=float)

    feasible = objective(vector, start.burn_count, own_scenario, own_config, own_limits)
    assert feasible < 0.0

    def with_floor(radius_floor_km: float) -> float:
        walls = MultiBurnLimits(
            burn_cap_km_s=own_limits.burn_cap_km_s,
            separation_s=own_limits.separation_s,
            first_burn_floor_s=own_limits.first_burn_floor_s,
            radius_floor_km=radius_floor_km,
            horizon_s=own_limits.horizon_s,
        )
        return objective(vector, start.burn_count, own_scenario, own_config, walls)

    flown_floor_km = schedule["minimum_radius_km"]
    mild = with_floor(flown_floor_km + 1.0)
    severe = with_floor(flown_floor_km + 500.0)

    assert feasible < mild < severe, "the barrier must rank violations, not flatten them"
    assert mild > 0.0, "every infeasible schedule stays above every feasible one"

    # Structurally impossible is worse still, and ranked among itself.
    beyond = vector.copy()
    beyond[1] = own_limits.horizon_s + 100.0
    far_beyond = vector.copy()
    far_beyond[1] = own_limits.horizon_s + 10_000.0
    near_value = objective(beyond, start.burn_count, own_scenario, own_config, own_limits)
    far_value = objective(far_beyond, start.burn_count, own_scenario, own_config, own_limits)
    assert severe < near_value < far_value


def test_the_safe_floor_script_uses_the_same_closed_form() -> None:
    """It kept its own sampled copy and called arcs 1100-1900 km underground safe."""
    import importlib.util

    from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, propagate_universal

    path = Path(__file__).resolve().parents[2] / "scripts" / "safe_delta_v_floor.py"
    spec = importlib.util.spec_from_file_location("safe_delta_v_floor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mu = 398600.4418
    for eccentricity in (1.01, 1.6, 3.0):
        elements = KeplerianElements(6578.0 / (1.0 - eccentricity), eccentricity, 30, 20, 10, 0)
        inbound = propagate_universal(keplerian_to_cartesian(elements, mu), -731.234, mu)
        assert module._arc_floor(inbound, 2000.0, mu) == pytest.approx(6578.0, abs=1.0e-6)


def _walls(burn_cap_km_s: float, total_cap_km_s: float | None) -> MultiBurnLimits:
    return MultiBurnLimits(
        burn_cap_km_s=burn_cap_km_s,
        separation_s=101.0,
        first_burn_floor_s=1.0e-3,
        radius_floor_km=6478.1363,
        horizon_s=20000.0,
        total_cap_km_s=total_cap_km_s,
    )


def test_without_a_total_cap_the_default_counts_are_the_preliminary_ones() -> None:
    counts, why = default_burn_counts(_walls(1.495, None))
    assert counts == (5, 6, 7)
    assert "preliminary" in why


def test_under_the_ranking_caps_the_default_counts_can_spend_the_total() -> None:
    """7 impulses at 0.249 km/s spend 1.743 of 2.99 km/s; 13 are needed for all of it."""
    walls = _walls(0.249, 2.99)
    counts, why = default_burn_counts(walls)
    assert counts == (5, 6, 7, 13)
    assert max(counts) * walls.burn_cap_km_s >= 2.99 > (max(counts) - 1) * walls.burn_cap_km_s
    assert "13" in why and "2.99" in why


def test_the_default_counts_come_from_a_ranking_policy(high_plane_script: Path) -> None:
    policy = dataclasses.replace(
        OrbitWinConfig().policy,
        rules="ranking",
        horizon_s=20000.0,
        dv_limit_km_s=0.25,
        dv_safe_limit_km_s=0.249,
        total_dv_limit_km_s=3.0,
        total_dv_safe_limit_km_s=2.99,
    )
    config = dataclasses.replace(OrbitWinConfig(), policy=policy)
    scenario, _ = parse_gmat_script(str(high_plane_script), config)
    walls = MultiBurnLimits.from_config(scenario, config)
    assert default_burn_counts(walls)[0] == (5, 6, 7, 13)

    preliminary, _ = parse_gmat_script(str(high_plane_script), OrbitWinConfig())
    walls = MultiBurnLimits.from_config(preliminary, OrbitWinConfig())
    assert default_burn_counts(walls)[0] == (5, 6, 7)


@pytest.mark.parametrize(
    ("burn_cap_km_s", "total_cap_km_s", "expected"),
    [
        (0.249, 2.988, (5, 6, 7, 12)),  # an exact multiple must not round up to 13
        (0.249, 1.2, (5, 6, 7)),
        (0.249, 0.2, (1, 5, 6, 7)),
        (1.495, 2.99, (2, 5, 6, 7)),
    ],
)
def test_the_default_counts_keep_the_preliminary_ones_and_add_the_total(
    burn_cap_km_s: float, total_cap_km_s: float, expected: tuple[int, ...]
) -> None:
    counts, _ = default_burn_counts(_walls(burn_cap_km_s, total_cap_km_s))
    assert counts == expected


def test_the_coast_before_the_first_burn_is_measured(
    different_plane_script: Path, tmp_path: Path
) -> None:
    """A start that lands near 2200 s, with the first burn at 3000 s, went below the
    surface before it; fly() read the radius from the first burn on."""
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.multi_burn import fly

    text = different_plane_script.read_text(encoding="utf-8")
    for old, new in {"Pursuer.SMA = 7000;": "Pursuer.SMA = 6600;",
                     "Pursuer.ECC = 0;": "Pursuer.ECC = 0.04;",
                     "Pursuer.TA = 300;": "Pursuer.TA = 180;"}.items():
        text = text.replace(old, new)
    path = tmp_path / "falling.script"
    path.write_text(text, encoding="utf-8")
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(path, config)
    limits = MultiBurnLimits.from_config(scenario, config)
    flown = fly(np.array([3000.0, 6000.0]), 1, scenario, config, limits)
    assert flown is not None
    assert flown[2] < 6378.1363 - 30.0  # pericenter 6336 km, passed at about 2400 s
