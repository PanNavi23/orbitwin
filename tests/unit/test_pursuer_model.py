"""The pursuer model: J2 about the pole of date and GMAT's SPAD drag term.

The numbers it carries were measured on GMAT R2026a (pursuer_model.py says
how); these tests pin the mechanics: it reduces to the conic, its gradients
are the flight's, a plan it makes keeps its frames through the corrector and
the slot variants, and a script it was not measured on is refused.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from orbitwin import gmat_shoot
from orbitwin import ranking_search as rs
from orbitwin.config import OrbitWinConfig, ScoreConfig, load_config
from orbitwin.conic import coast, periapsis_radius_km, seconds_to_periapsis
from orbitwin.errors import ConfigurationError, PolicyError
from orbitwin.frames import inertial_to_script_components
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Candidate, as_vector3, burn_frame
from orbitwin.pursuer_model import (
    R2026A_NRLMSISE00_F150_SPAD_SPHERE_850KG,
    DragTable,
    arc_minimum_radius,
    calibration_mismatches,
    checked_pursuer_model,
    fly,
    model_for,
    precessed_pole,
)
from orbitwin.search import slot_assignment_variants

MU = 398600.4418
R4 = Path(__file__).resolve().parents[2] / "examples" / "ranking" / "R4_LowPerigee_Drag.script"


def _conic_model():
    return dataclasses.replace(model_for("j2", None, MU), j2=0.0)


def _ranking_config() -> OrbitWinConfig:
    base = OrbitWinConfig()
    policy = dataclasses.replace(
        base.policy, rules="ranking", horizon_s=5219.966, dv_limit_km_s=0.25,
        dv_safe_limit_km_s=0.249, total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99,
    )
    return dataclasses.replace(
        base, policy=policy, pursuer_name="SC_Human", target_name="SC_Alien",
        score=ScoreConfig(kt=0.003, ct=4175.9728, kv=0.0012, cv=2000.0, delta_v_unit="m/s",
                          cr_km=100.0),
    )


# -- the model


def test_without_j2_or_drag_it_flies_the_conic() -> None:
    model = _conic_model()
    r0, v0 = np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.2, 1.0])
    arc = model.propagate(r0, v0, 3000.0)
    r, v = coast(r0, v0, 3000.0, MU)
    assert np.linalg.norm(arc.r[-1, 0] - r) < 1.0e-4
    assert np.linalg.norm(arc.v[-1, 0] - v) < 1.0e-7
    between, _ = arc.sample(0, np.array([1234.5]))
    assert np.linalg.norm(between[0] - coast(r0, v0, 1234.5, MU)[0]) < 1.0e-5


def test_samples_at_nodes_are_the_nodes() -> None:
    model = model_for("j2-spad-drag", 0.27, MU)
    arc = model.propagate(np.array([6600.0, 0.0, 0.0]), np.array([0.0, 7.7, 0.5]), 95.0, 7.0)
    np.testing.assert_allclose(arc.times_s, [0.0, 8.0, 23.0, 38.0, 53.0, 68.0, 83.0, 95.0])
    r, v = arc.sample(0, arc.times_s)
    np.testing.assert_allclose(r, arc.r[:, 0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(v, arc.v[:, 0], rtol=0, atol=1e-12)


def test_j2_about_z_is_the_textbook_term() -> None:
    model = dataclasses.replace(model_for("j2", None, MU), mu_km3_s2=MU)
    r = np.array([[6000.0, 2500.0, 3100.0]])
    x, y, z = r[0]
    rn = float(np.linalg.norm(r))
    factor = -1.5 * model.j2 * MU * model.radius_km**2 / rn**5
    expected = -MU * r[0] / rn**3 + factor * np.array(
        [x * (1 - 5 * z * z / rn**2), y * (1 - 5 * z * z / rn**2), z * (3 - 5 * z * z / rn**2)]
    )
    np.testing.assert_allclose(model.acceleration(r, np.zeros((1, 3)))[0], expected, rtol=1e-12)


def test_the_pole_of_date() -> None:
    assert precessed_pole(0.0) == pytest.approx((0.0, 0.0, 1.0), abs=1e-15)
    # 2026-11-09 02:00 UTC; 0.1495 deg from J2000's z.
    pole = precessed_pole(0.26854439792127016)
    assert pole == pytest.approx((2.60932467e-03, -7.83495086e-06, 0.99999659568), abs=1e-10)
    assert math.fsum(c * c for c in pole) == pytest.approx(1.0, abs=1e-15)


def _energy(r: np.ndarray, v: np.ndarray) -> float:
    return float(0.5 * v @ v - MU / np.linalg.norm(r))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_drag_pushes_the_way_its_sign_says(sign: float) -> None:
    """GMAT's SPAD sphere adds energy through perigee; textbook drag takes it."""
    knots, logs = R2026A_NRLMSISE00_F150_SPAD_SPHERE_850KG
    model = dataclasses.replace(
        model_for("j2-spad-drag", None, MU), j2=0.0, drag=DragTable(knots, logs, sign)
    )
    # Perigee at 110 km, 1500 s ahead.
    rp, ra = 6378.1363 + 110.0, 6378.1363 + 420.0
    a = 0.5 * (rp + ra)
    r0 = np.array([0.0, ra, 0.0])
    v0 = np.array([-math.sqrt(MU * (2 / ra - 1 / a)), 0.0, 0.0])
    arc = model.propagate(r0, v0, 2.0 * math.pi * math.sqrt(a**3 / MU))
    change = _energy(arc.r[-1, 0], arc.v[-1, 0]) - _energy(r0, v0)
    assert sign * change > 1.0e-3


def test_the_drag_table_holds_its_knots_and_stays_finite() -> None:
    knots, logs = R2026A_NRLMSISE00_F150_SPAD_SPHERE_850KG
    table = DragTable(knots, logs, 1.0)
    np.testing.assert_allclose(np.log(table.rho_b(np.array(knots))), logs, atol=1e-12)
    values = table.rho_b(np.array([-500.0, 0.0, 50.0, 99.0, 421.0, 1500.0, 1.0e6]))
    assert np.all(np.isfinite(values)) and np.all(values > 0.0)
    assert np.all(np.diff(values) <= 0.0)


def test_the_lowest_radius_of_an_arc() -> None:
    model = _conic_model()
    r0, v0 = np.array([6800.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.5])
    to_periapsis = seconds_to_periapsis(r0, v0, MU)
    assert to_periapsis is not None
    value, at_s, kind = arc_minimum_radius(model.propagate(r0, v0, to_periapsis + 500.0))
    assert kind == "interior"
    assert value == pytest.approx(periapsis_radius_km(r0, v0, MU), abs=1e-4)
    assert at_s == pytest.approx(to_periapsis, abs=1e-2)
    value, at_s, kind = arc_minimum_radius(model.propagate(r0, v0, to_periapsis - 400.0))
    assert kind == "end" and at_s == pytest.approx(to_periapsis - 400.0)
    late_r, late_v = coast(r0, v0, to_periapsis + 300.0, MU)
    value, at_s, kind = arc_minimum_radius(model.propagate(late_r, late_v, 900.0))
    assert kind == "start" and value == pytest.approx(np.linalg.norm(late_r))


def test_spans_run_backwards_and_stand_still() -> None:
    model = _conic_model()
    r0, v0 = np.array([6800.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.5])
    back = model.propagate(r0, v0, -500.0, 1000.0)
    # Nodes on the grid from t = 0: 990, 975, ... 510, then 500.
    np.testing.assert_allclose(back.times_s[:3], [0.0, -10.0, -25.0])
    assert back.times_s[-1] == -500.0 and np.all(np.diff(back.times_s) < 0.0)
    np.testing.assert_allclose(back.r[-1, 0], coast(r0, v0, -500.0, MU)[0], atol=1e-5)
    there, _ = back.sample(0, np.array([-123.4]))
    np.testing.assert_allclose(there[0], coast(r0, v0, -123.4, MU)[0], atol=1e-5)
    to_periapsis = seconds_to_periapsis(r0, v0, MU)
    assert to_periapsis is not None
    after_r, after_v = coast(r0, v0, to_periapsis + 300.0, MU)
    # Off the grid, so perigee falls between nodes and the root is solved.
    value, _, kind = arc_minimum_radius(model.propagate(after_r, after_v, -600.0, 3007.0))
    assert kind == "interior"
    assert value == pytest.approx(periapsis_radius_km(r0, v0, MU), abs=1e-4)
    still = model.propagate(r0, v0, 0.0, 42.0)
    assert arc_minimum_radius(still) == (pytest.approx(np.linalg.norm(r0)), 0.0, "start")
    np.testing.assert_array_equal(still.sample(0, np.array([0.0]))[0][0], r0)


def test_outside_a_span_is_a_taylor_step_from_its_end() -> None:
    """A burn a microsecond before a grid node leaves a partial step that long;
    Hermite extrapolated off it blew the miss up to 2e5 km."""
    model = _conic_model()
    r0, v0 = np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0])
    arc = model.propagate(r0, v0, 600.0, 15.0 - 1.0e-6)
    assert arc.times_s[1] == pytest.approx(1.0e-6)
    backward = model.propagate(r0, v0, -600.0, 15.0 + 1.0e-6)
    for span, elapsed in ((arc, -0.5), (arc, 600.5), (backward, 0.5), (backward, -600.5)):
        r, v = span.sample(0, np.array([elapsed]))
        expected_r, expected_v = coast(r0, v0, elapsed, MU)
        # RK4 drifts 2e-6 km in 600 s; jerk * dt^2 / 2 is 1e-6 km/s over 0.5 s.
        np.testing.assert_allclose(r[0], expected_r, rtol=0, atol=1e-5)
        np.testing.assert_allclose(v[0], expected_v, rtol=0, atol=5e-6)
        every_r, every_v = span.sample_all(elapsed)
        np.testing.assert_array_equal(every_r[0], r[0])
        np.testing.assert_array_equal(every_v[0], v[0])


# -- the search


@pytest.fixture
def coplanar(tmp_path: Path, no_maneuver_script: Path):
    """A 7000 km pursuer and a coplanar target a capped impulse can reach, as in
    test_ranking_search."""
    speed = math.sqrt(MU / 7000.0) + 0.249
    apoapsis = 2.0 / (2.0 / 7000.0 - speed * speed / MU) - 7000.0
    target_radius = apoapsis - 30.0
    transfer_s = math.pi * math.sqrt(((7000.0 + target_radius) / 2.0) ** 3 / MU)
    target_ta = (180.0 - math.degrees(math.sqrt(MU / target_radius**3) * transfer_s)) % 360.0
    text = no_maneuver_script.read_text(encoding="utf-8")
    for old, new in {
        "Pursuer.SMA = 7000;": "Pursuer.SMA = 7000.0;",
        "Pursuer.INC = 28;": "Pursuer.INC = 30;",
        "Pursuer.RAAN = 15;": "Pursuer.RAAN = 40;",
        "Pursuer.TA = 300;": "Pursuer.TA = 0;",
        "AlienSC.SMA = 9000;": f"AlienSC.SMA = {target_radius!r};",
        "AlienSC.INC = 43;": "AlienSC.INC = 30;",
        "AlienSC.RAAN = 65;": "AlienSC.RAAN = 40;",
        "AlienSC.TA = 20;": f"AlienSC.TA = {target_ta!r};",
    }.items():
        assert old in text
        text = text.replace(old, new)
    script = tmp_path / "coplanar.script"
    script.write_text(text, encoding="utf-8")
    base = OrbitWinConfig()
    config = dataclasses.replace(
        base,
        policy=dataclasses.replace(
            base.policy, rules="ranking", horizon_s=4000.0, dv_limit_km_s=0.25,
            dv_safe_limit_km_s=0.249, total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99,
        ),
        score=ScoreConfig(kt=0.002, ct=2000.0, kv=0.002, cv=1000.0, delta_v_unit="m/s",
                          cr_km=100.0),
        pursuer_name="Pursuer", target_name="AlienSC",
    )
    return parse_gmat_script(script, config)[0], config


def _problem(scenario, config, model, times=(300.0, 1200.0), aim_s: float = 2600.0):
    setup = dataclasses.replace(rs.build_setup(scenario, config), pursuer_model=model)
    problem = rs._Problem(setup, len(times), 10.0)
    impulses = np.array([[0.05, 0.12, -0.01], [-0.02, 0.06, 0.03]][: len(times)])
    return problem, problem.pack(list(times), impulses, aim_s)


def _dense_air(model):
    """Air at the coplanar scenario's 620-1000 km, enough to exercise the drag terms."""
    return dataclasses.replace(model, drag=DragTable((500.0, 700.0, 900.0), (-21.0, -23.0, -25.0), 1.0))


def _assert_gradients_are_the_flights(problem, x: np.ndarray) -> None:
    value = problem._evaluate(x)
    analytic = np.vstack([value.miss_gradient, value.floor_gradients])

    def measured(at: np.ndarray) -> np.ndarray:
        e = problem._evaluate(at)
        return np.concatenate([[e.miss_km], e.floors_km])

    for column in range(len(x)):
        step = 1.0e-6 if problem.n <= column < 4 * problem.n else 1.0e-5
        up, down = x.copy(), x.copy()
        up[column] += step
        down[column] -= step
        numeric = (measured(up) - measured(down)) / (2.0 * step)
        np.testing.assert_allclose(analytic[:, column], numeric, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("with_drag", [False, True])
def test_model_gradients_are_the_flights(coplanar, with_drag: bool) -> None:
    scenario, config = coplanar
    model = model_for("j2", 0.27, scenario.mu_km3_s2)
    problem, x = _problem(scenario, config, _dense_air(model) if with_drag else model)
    _assert_gradients_are_the_flights(problem, x)


@pytest.mark.parametrize("with_drag", [False, True])
def test_interior_floor_gradients_are_the_flights(coplanar, with_drag: bool) -> None:
    """A retrograde first burn puts perigee inside the last arc, the floor kind
    that carries R4's plans."""
    scenario, config = coplanar
    model = model_for("j2", 0.27, scenario.mu_km3_s2)
    model = _dense_air(model) if with_drag else model
    setup = dataclasses.replace(rs.build_setup(scenario, config), pursuer_model=model)
    problem = rs._Problem(setup, 2, 10.0)
    initial = scenario.pursuer.initial_state
    _, v300 = coast(np.asarray(initial.r), np.asarray(initial.v), 300.0, scenario.mu_km3_s2)
    impulses = np.array([-0.12 * v300 / np.linalg.norm(v300), [0.0, 0.01, 0.0]])
    flight = rs.fly_model([300.0, 1200.0], impulses, initial, model, setup.limits.horizon_s)
    assert arc_minimum_radius(flight.arcs[-1])[2] == "interior"
    _assert_gradients_are_the_flights(problem, problem.pack([300.0, 1200.0], impulses, 2600.0))


def test_the_problem_flies_what_fly_model_flies(coplanar) -> None:
    """The same grid from t = 0 in both, so SLSQP optimises the flight that is settled.
    Burns off the 15 s grid: on it, a grid from each leg's start is the same grid."""
    scenario, config = coplanar
    model = _dense_air(model_for("j2", 0.27, scenario.mu_km3_s2))
    problem, x = _problem(scenario, config, model, (307.3, 1213.7), 2603.9)
    times, impulses, aim_s = problem.unpack(x)
    value = problem._evaluate(x)
    flight = rs.fly_model(list(times), impulses, scenario.pursuer.initial_state, model,
                          problem.setup.limits.horizon_s)
    target_r, _ = problem.setup.target.state(aim_s)
    assert value.miss_km == pytest.approx(
        float(np.linalg.norm(flight.state(aim_s)[0] - target_r)), abs=1e-9
    )
    np.testing.assert_allclose(
        value.floors_km, [arc_minimum_radius(arc)[0] for arc in flight.arcs], atol=1e-9
    )


def test_a_miss_is_polished_from_an_aim_after_its_last_burn(
    coplanar, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario, config = coplanar
    setup = rs.build_setup(scenario, config)
    early = rs.Measurement(50.0, 120.0, None, 50.0, 7000.0)
    miss = rs.RankingSolution(
        "m", "miss", (100.0, 300.0), ((0.0, 0.01, 0.0),) * 2, early, 0.02, floor_margin_km=10.0
    )
    seen: list = []
    monkeypatch.setattr(rs, "_refine_all", lambda jobs, *_: seen.extend(jobs) or [])
    model = model_for("j2", 0.27, scenario.mu_km3_s2)
    rs.polish_in_model(
        scenario, config, dataclasses.replace(setup, pursuer_model=model),
        rs.RankingSearchResult(setup, 1, [miss]), top=1,
    )
    (start, _), = seen
    assert start.mode == "miss" and start.aim_s == pytest.approx(301.0)


def test_without_perturbations_the_model_problem_is_the_two_body_one(coplanar) -> None:
    scenario, config = coplanar
    problem, x = _problem(scenario, config, _conic_model())
    two_body = rs._Problem(dataclasses.replace(problem.setup, pursuer_model=None), 2, 10.0)
    model, conic = problem._evaluate(x), two_body._evaluate(x)
    assert model.miss_km == pytest.approx(conic.miss_km, abs=1e-4)
    np.testing.assert_allclose(model.floors_km, conic.floors_km, atol=1e-4)
    np.testing.assert_allclose(model.miss_gradient, conic.miss_gradient, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(model.floor_gradients, conic.floor_gradients, rtol=1e-3, atol=1e-3)


def test_a_model_flight_is_post_burn_at_a_burn_epoch(coplanar) -> None:
    scenario, _ = coplanar
    model = model_for("j2", 0.27, scenario.mu_km3_s2)
    flight = rs.fly_model(
        [500.0], np.array([[0.0, 0.1, 0.0]]), scenario.pursuer.initial_state, model, 4000.0
    )
    r, v = flight.state(500.0)
    np.testing.assert_allclose(r, flight.r_at_burns[0], atol=1e-9)
    np.testing.assert_allclose(v, flight.v_after[0], atol=1e-12)


def test_polishing_in_the_model_hits_in_it_and_keeps_its_frames(coplanar) -> None:
    scenario, config = coplanar
    setup = rs.build_setup(scenario, config, screen_iterations=40, polish_iterations=60)
    starts = rs.build_starts(setup, config, [1, 2], transfer_count=3, random_count=2,
                             transfer_grid=(8, 12))
    searched = rs.search_ranking(scenario, config, setup, starts, polish_top=2)
    model = model_for("j2", 0.27, scenario.mu_km3_s2)
    polished = rs.polish_in_model(
        scenario, config, dataclasses.replace(setup, pursuer_model=model), searched, top=2
    )
    best = polished.solutions[0]
    assert best.successful and best.measurement.minimum_distance_km <= 5.0
    flight = rs.fly_model(best.times_s, np.array(best.impulses_km_s),
                          scenario.pursuer.initial_state, model, setup.limits.horizon_s)
    assert rs.measure(flight, setup.target, setup.grid, setup.limits.horizon_s, 5.0) == (
        best.measurement
    )
    candidate = rs.scored_candidate(best, scenario, config, setup.limits, model)
    assert candidate.metadata["pursuer_model"] == "j2"
    burns = tuple(sorted(candidate.burns, key=lambda burn: burn.time_s))
    gmat_shoot._check_frames(burns, scenario, model)
    with pytest.raises(PolicyError):
        gmat_shoot._check_frames(burns, scenario, None)


# -- the corrector and the slot variants


def test_the_corrector_takes_frames_on_the_plans_model(workshop_script: Path) -> None:
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    model = model_for("j2", None, scenario.mu_km3_s2)
    schedule = [(400.0, np.array([0.05, 0.02, -0.01])), (1500.0, np.array([0.0, 0.04, 0.01]))]
    states = gmat_shoot._flown_states(scenario, schedule, model)
    initial = scenario.pursuer.initial_state
    flown = fly(model, np.asarray(initial.r), np.asarray(initial.v), np.array([400.0, 1500.0]),
                np.array([dv for _, dv in schedule]), 3000.0)
    np.testing.assert_array_equal(np.asarray(states[1].r), flown.r_at_burns[1])
    two_body = gmat_shoot._flown_states(scenario, schedule)
    assert np.linalg.norm(np.asarray(states[1].r) - np.asarray(two_body[1].r)) > 1.0
    original = tuple(
        Burn(t, as_vector3(dv), as_vector3(inertial_to_script_components(
            dv, state, *burn_frame(scenario, slot))), slot, burn_frame(scenario, slot))
        for slot, ((t, dv), state) in enumerate(zip(schedule, states, strict=True))
    )
    changed = [schedule[0], (1500.0, np.array([0.0, 0.05, 0.01]))]
    expressed = gmat_shoot._express(scenario, changed, [0, 1], original, model)
    assert expressed[0] is original[0]
    np.testing.assert_allclose(
        expressed[1].dv_script_km_s,
        inertial_to_script_components(changed[1][1], states[1], *burn_frame(scenario, 1)),
        atol=1e-15,
    )


def test_gmat_sensitivity_probes_take_frames_on_the_plans_model(
    tmp_path: Path, workshop_script: Path
) -> None:
    from types import SimpleNamespace

    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    model = model_for("j2", None, scenario.mu_km3_s2)
    schedule = [(400.0, np.array([0.05, 0.02, -0.01])), (1500.0, np.array([0.0, 0.04, 0.01]))]
    states = gmat_shoot._flown_states(scenario, schedule, model)
    original = tuple(
        Burn(t, as_vector3(dv), as_vector3(inertial_to_script_components(
            dv, state, *burn_frame(scenario, slot))), slot, burn_frame(scenario, slot))
        for slot, ((t, dv), state) in enumerate(zip(schedule, states, strict=True))
    )
    flown: list[tuple[Burn, ...]] = []

    def fly_probe(burns, _directory):
        flown.append(burns)
        return SimpleNamespace(miss_km=np.asarray(burns[-1].dv_inertial_km_s) * 1000.0)

    base = SimpleNamespace(miss_km=np.asarray(original[-1].dv_inertial_km_s) * 1000.0)
    gmat_shoot._measure_sensitivity(
        fly_probe, scenario, schedule, [0, 1], original, base, tmp_path / "s", 1, model
    )
    assert len(flown) == 6
    for burns in flown:
        gmat_shoot._check_frames(burns, scenario, model)
    with pytest.raises(PolicyError):
        gmat_shoot._check_frames(flown[0], scenario, None)


def test_slot_variants_of_a_model_plan_take_its_states(workshop_script: Path) -> None:
    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    model = model_for("j2", None, scenario.mu_km3_s2)
    dv = np.array([0.05, 0.02, -0.01])
    state = gmat_shoot._flown_states(scenario, [(900.0, dv)], model)[0]
    candidate = Candidate(
        burns=(Burn(900.0, as_vector3(dv), as_vector3(inertial_to_script_components(
            dv, state, *burn_frame(scenario, 0))), 0, burn_frame(scenario, 0)),),
        intercept_time_s=2000.0, minimum_distance_km=1.0, distance_at_completion_km=5.0,
        successful=True, analytical_endpoint_error_km=None, branch="ranking-search/1",
        metadata={"arrival_time_s": 2000.0, "pursuer_model": "j2"},
    )
    variants = slot_assignment_variants(scenario, config, [candidate])
    assert variants
    burn = variants[0].burns[0]
    np.testing.assert_allclose(
        burn.dv_script_km_s,
        inertial_to_script_components(dv, state, *burn_frame(scenario, burn.slot_index)),
        atol=1e-15,
    )
    assert variants[0].metadata["pursuer_model"] == "j2"


# -- what the model was measured on


def _r4(tmp_path: Path, **swaps: str):
    text = R4.read_text(encoding="utf-8")
    for old, new in swaps.items():
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "r4.script"
    path.write_text(text, encoding="utf-8")
    return parse_gmat_script(path, _ranking_config())


def test_the_rehearsal_force_model_is_what_the_table_was_measured_on(tmp_path: Path) -> None:
    scenario, document = _r4(tmp_path)
    assert calibration_mismatches(scenario, document, "j2-spad-drag") == []
    model = checked_pursuer_model(scenario, document, "j2-spad-drag")
    assert model is not None and model.drag is not None and model.drag.sign == 1.0
    assert checked_pursuer_model(scenario, document, "two-body") is None
    # J2 alone flew R4 65 km from GMAT, worse than two-body.
    assert any("j2 leaves drag out" in m for m in calibration_mismatches(scenario, document, "j2"))


def test_j2_is_refused_under_the_deprecated_drag_form_too(tmp_path: Path) -> None:
    text = R4.read_text(encoding="utf-8")
    kept = [line for line in text.splitlines() if ".Drag." not in line]
    kept.insert(kept.index("DefaultProp_ForceModel.SRP = On;"),
                "DefaultProp_ForceModel.Drag = NRLMSISE00;")
    path = tmp_path / "old_drag.script"
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    scenario, document = parse_gmat_script(path, _ranking_config())
    found = calibration_mismatches(scenario, document, "j2")
    assert any("DefaultProp_ForceModel.Drag is NRLMSISE00" in m for m in found), found


@pytest.mark.parametrize(
    ("old", "new", "named"),
    [
        ("Drag.F107 = 150;", "Drag.F107 = 140;", "Drag.F107"),
        ("'ConstantFluxAndGeoMag';\nDefaultProp_ForceModel.Drag.F107 ",
         "'CSSISpaceWeatherFile';\nDefaultProp_ForceModel.Drag.F107 ", "PredictedWeatherSource"),
        ("AtmosphereModel = NRLMSISE00;", "AtmosphereModel = JacchiaRoberts;", "AtmosphereModel"),
        ("Drag.DragModel = 'SPADFile';", "Drag.DragModel = 'Spherical';", "DragModel"),
        ("SC_Human.DryMass = 850;", "SC_Human.DryMass = 900;", "DryMass"),
        ("Degree = 20;", "Degree = 0;", "Degree"),
        ("'EGM96.cof'", "'JGM3.cof'", "PotentialFile"),
        ("DefaultProp.FM = DefaultProp_ForceModel;", "", "no FM"),
        # Which of the two drag fields GMAT obeys when both are set is not measured.
        ("DefaultProp_ForceModel.SRP = On;",
         "DefaultProp_ForceModel.Drag = JacchiaRoberts;\nDefaultProp_ForceModel.SRP = On;",
         "beside Drag.AtmosphereModel"),
        # GMAT divides drag by the total mass; a tank adds fuel to the 850 kg.
        ("SC_Human.EulerAngleSequence = '321';",
         "SC_Human.EulerAngleSequence = '321';\nCreate ChemicalTank HumanTank;\n"
         "HumanTank.FuelMass = 850;\nSC_Human.Tanks = {HumanTank};", "Tanks"),
    ],
)
def test_a_script_the_table_was_not_measured_on_is_refused(
    tmp_path: Path, old: str, new: str, named: str
) -> None:
    scenario, document = _r4(tmp_path, **{old: new})
    found = calibration_mismatches(scenario, document, "j2-spad-drag")
    assert any(named in reason for reason in found), found
    with pytest.raises(ConfigurationError, match=named):
        checked_pursuer_model(scenario, document, "j2-spad-drag")


def test_fields_left_at_gmats_defaults_are_its_defaults(tmp_path: Path) -> None:
    scenario, document = _r4(tmp_path, **{
        "DefaultProp_ForceModel.Drag.F107 = 150;\n": "",
        "DefaultProp_ForceModel.Drag.MagneticIndex = 3;\n": "",
        "SC_Human.DryMass = 850;\n": "",
        "SC_Human.SPADDragScaleFactor = 1;\n": "",
    })
    assert calibration_mismatches(scenario, document, "j2-spad-drag") == []


def test_an_unknown_model_is_refused_by_the_config(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("search:\n  ranking_pursuer_model: j2-drag\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="ranking_pursuer_model"):
        load_config(path)
    path.write_text("search:\n  ranking_pursuer_model_top: 0\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="ranking_pursuer_model_top"):
        load_config(path)


def test_a_misfit_model_falls_back_to_two_body_and_says_so(tmp_path: Path) -> None:
    from orbitwin.gmat_loop import ranking_search_candidates

    scenario, document = _r4(tmp_path, **{"Drag.F107 = 150;": "Drag.F107 = 140;"})
    config = _ranking_config()
    config = dataclasses.replace(config, search=dataclasses.replace(
        config.search, ranking_search=True, ranking_pursuer_model="j2-spad-drag",
        ranking_transfer_starts=1, ranking_random_starts=0, ranking_max_burns=1,
        ranking_screen_iterations=5, ranking_polish_iterations=5, ranking_polish_top=1,
        workers=1,
    ))
    candidates, record = ranking_search_candidates(
        scenario, document, config, tmp_path / "out", None, None
    )
    assert record["pursuer_model"] == "two-body"
    assert "Drag.F107" in record["pursuer_model_error"]
    assert all("pursuer_model" not in c.metadata for c in candidates)


# -- beside main's plans, not instead of them


def _plan(time_s: float, dv: float, arrival_s: float, model: str | None) -> Candidate:
    metadata: dict = {"arrival_time_s": arrival_s, "event_refined": True}
    if model is not None:
        metadata["pursuer_model"] = model
    return Candidate(
        burns=(Burn(time_s, (dv, 0.0, 0.0), (dv, 0.0, 0.0), 0),),
        intercept_time_s=arrival_s, minimum_distance_km=1.0, distance_at_completion_km=5.0,
        successful=True, analytical_endpoint_error_km=None, branch="ranking-search/1",
        metadata=metadata,
    )


def test_model_plans_join_after_the_prune() -> None:
    """Pruned with them, a model plan dropped the two-body plan GMAT scored best on R1."""
    from orbitwin.pipeline import prune_beside_model_plans
    from orbitwin.search import select_candidates

    base = OrbitWinConfig()
    # The archive is full: on R1 the frontier's model plans pushed it past keep.
    config = dataclasses.replace(base, search=dataclasses.replace(base.search, keep_candidates=1))
    two_body = _plan(10.0, 0.30, 4000.0, None)
    in_model = _plan(10.0, 0.20, 3000.0, "j2-spad-drag")
    pruned = select_candidates([two_body, in_model], 1, config)
    assert [c.total_dv_km_s for c in pruned] == [pytest.approx(0.2)]
    kept = prune_beside_model_plans([two_body, in_model], config)
    assert sorted(c.total_dv_km_s for c in kept) == [pytest.approx(0.2), pytest.approx(0.3)]


def test_model_plans_have_a_correction_budget_of_their_own() -> None:
    from orbitwin import gmat_loop

    config = OrbitWinConfig()
    config = dataclasses.replace(config, gmat=dataclasses.replace(config.gmat, correct_top=2))
    models = [_plan(10.0, 0.1 + 0.01 * k, 3000.0, "j2") for k in range(3)]
    two_body = [_plan(10.0, 0.2 + 0.01 * k, 3500.0, None) for k in range(3)]
    chosen = gmat_loop.correction_choices([*models, *two_body], config)
    assert chosen == [*models[:2], *two_body[:2]]


def test_model_plans_have_a_verification_budget_of_their_own() -> None:
    from orbitwin.pipeline import verification_beside_model_plans

    two_body = [_plan(10.0, 0.2 + 0.01 * k, 3500.0 + 400.0 * k, None) for k in range(2)]
    in_model = [_plan(10.0, 0.1 + 0.01 * k, 3000.0 + 400.0 * k, "j2") for k in range(2)]
    chosen = verification_beside_model_plans([*in_model, *two_body], 1, misses_count=True)
    assert len(chosen) == 2
    assert sum("pursuer_model" in c.metadata for c in chosen) == 1
    without = verification_beside_model_plans(two_body, 1, misses_count=True)
    assert [c.candidate_id for c in without] == [chosen[0].candidate_id]


def test_the_two_body_plans_are_offered_beside_the_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On R3 the model's plans alone scored 81.3081 in GMAT, main's 81.3918."""
    from orbitwin import gmat_loop

    scenario, document = _r4(tmp_path)
    config = _ranking_config()
    config = dataclasses.replace(config, search=dataclasses.replace(
        config.search, ranking_search=True, ranking_pursuer_model="j2-spad-drag",
        ranking_transfer_starts=1, ranking_random_starts=0, ranking_max_burns=1,
        ranking_screen_iterations=5, ranking_polish_iterations=5, ranking_polish_top=1,
        workers=1,
    ))
    polished: list = []
    real = gmat_loop.polish_in_model

    def recording(*args, **kwargs):
        polished.append(real(*args, **kwargs))
        return polished[-1]

    monkeypatch.setattr(gmat_loop, "polish_in_model", recording)
    two_body_only = dataclasses.replace(
        config, search=dataclasses.replace(config.search, ranking_pursuer_model="two-body")
    )
    main, _ = gmat_loop.ranking_search_candidates(
        scenario, document, two_body_only, tmp_path / "main", None, None
    )
    both, record = gmat_loop.ranking_search_candidates(
        scenario, document, config, tmp_path / "model", None, None
    )
    assert record["pursuer_model"] == "j2-spad-drag" and len(polished) == 1
    assert main
    assert [c.candidate_id for c in both if "pursuer_model" not in c.metadata] == [
        c.candidate_id for c in main
    ]
