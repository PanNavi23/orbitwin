from __future__ import annotations

import math

import numpy as np
import pytest

from orbitwin.lambert import LambertSolution, _shooting_fallback, solve_lambert_all
from orbitwin.models import StateVector
from orbitwin.orbit import (
    KeplerianElements,
    cartesian_to_keplerian,
    keplerian_to_cartesian,
    orbital_period_s,
    propagate_universal,
)

MU_KM3_S2 = 398600.4418

# One tolerance for every velocity comparison here, chosen from measurement and
# not from taste. Over thirty circular-to-circular radius pairs from 6800 km to
# 300000 km the worst gap between the solver's two-impulse total and the
# closed-form Hohmann total was 9.6e-13 km/s, and the typical gap was 1e-15
# km/s -- double-precision noise. On the universal-variable side the seven
# known arcs below reproduce a known velocity to 3.5e-15 km/s. 1e-9 km/s is
# three decades above the worst thing measured, which is the room a different
# SciPy or BLAS build needs, and six decades below a millimetre per second,
# which is finer than any number that could reach a GMAT ImpulsiveBurn field.
# The rule caps one impulse at 1.5 km/s, so this guards the quantity that
# matters nine orders of magnitude below it.
VELOCITY_TOL_KM_S = 1.0e-9

# Same reasoning for positions. The worst endpoint miss measured over the same
# thirty radius pairs was 2.7e-9 km; the intercept sphere is 5 km.
POSITION_TOL_KM = 1.0e-6

# And for the transfer conic's shape and period.
RELATIVE_TOL = 1.0e-9

# Angles are compared in degrees, the script boundary's unit. Every
# conic reconstructed from a returned velocity came back with an inclination
# error of exactly 0 in the cases here, so this is headroom rather than a
# measured bound. The near-antipodal tests deliberately use looser angle
# tolerances and say why at the point of use.
ANGLE_TOL_DEG = 1.0e-9


def _circular_speed_km_s(radius_km: float) -> float:
    return math.sqrt(MU_KM3_S2 / radius_km)


def _transfer_inclination_deg(r1_km: np.ndarray, v1_km_s: np.ndarray) -> float:
    """Inclination of the conic the returned departure velocity puts you on."""
    momentum = np.cross(r1_km, v1_km_s)
    momentum = momentum / np.linalg.norm(momentum)
    return math.degrees(math.acos(float(np.clip(momentum[2], -1.0, 1.0))))


def _true_endpoint_miss_km(
    r1_km: np.ndarray, v1_km_s: np.ndarray, tof_s: float, r2_km: np.ndarray
) -> float:
    """Re-propagate a returned departure velocity and measure where it lands.

    Independent of whatever the solver recorded in ``endpoint_error_km``, which
    is the point: this is what checks that the recorded number is honest.
    """
    arrival = propagate_universal(
        StateVector(
            tuple(float(x) for x in r1_km),
            tuple(float(x) for x in v1_km_s),
        ),
        tof_s,
        MU_KM3_S2,
    )
    return float(np.linalg.norm(arrival.r - r2_km))


# ---------------------------------------------------------------------------
# 1. Hohmann golden case
# ---------------------------------------------------------------------------


def test_hohmann_transfer_reproduces_the_closed_form_two_impulse_cost() -> None:
    """The one case where the right answer is known on paper, so check it.

    A Hohmann transfer between coplanar circular orbits is the half-ellipse
    that touches both, so its endpoints are exactly 180 degrees apart and its
    flight time is exactly half the transfer ellipse's period. Every quantity
    below is derived here from mu, r1 and r2 alone -- no constant is pasted in
    -- so the algebra can be checked by reading it.

    Note which code path this exercises, because it is not the obvious one.
    ``_transfer_angles`` tags 180 degrees ``ambiguous``, and the A term collapses
    to 1.5e-12 -- just above the 1e-12 guard, so the root scan does run and finds
    the right root at z = pi^2. The reconstruction is what fails: ``r2 - f*r1``
    cancels exactly, ``v1`` comes back as the zero vector, and propagating that
    misses by 47386 km, so the endpoint check throws it away. Both ambiguous
    entries do the same, ``solutions`` stays empty, and ``solve_lambert_all``
    falls through to the Cartesian shooting fallback, which is what answers here.
    Fail-closed validation is doing real work in this case rather than passing a
    degenerate velocity along.

    So the golden case for the textbook answer covers the fallback, and the known
    arcs in the next test cover the universal-variable path. Between them both
    solvers are pinned to something other than their own output.
    """
    r1_km = 7000.0
    r2_km = 42164.0

    # Closed form. Transfer ellipse: perigee at r1, apogee at r2.
    transfer_sma_km = 0.5 * (r1_km + r2_km)
    transfer_ecc = (r2_km - r1_km) / (r2_km + r1_km)
    # Half the transfer ellipse's period, which is also the Lambert time input.
    tof_s = math.pi * math.sqrt(transfer_sma_km**3 / MU_KM3_S2)
    # Vis-viva at each end of the transfer ellipse.
    speed_at_perigee = math.sqrt(MU_KM3_S2 * (2.0 / r1_km - 1.0 / transfer_sma_km))
    speed_at_apogee = math.sqrt(MU_KM3_S2 * (2.0 / r2_km - 1.0 / transfer_sma_km))
    # The textbook form, written independently of the vis-viva speeds above so
    # that the two ways of saying it can be checked against each other.
    departure_dv = _circular_speed_km_s(r1_km) * (
        math.sqrt(2.0 * r2_km / (r1_km + r2_km)) - 1.0
    )
    arrival_dv = _circular_speed_km_s(r2_km) * (
        1.0 - math.sqrt(2.0 * r1_km / (r1_km + r2_km))
    )
    assert departure_dv == pytest.approx(
        speed_at_perigee - _circular_speed_km_s(r1_km), abs=VELOCITY_TOL_KM_S
    )
    assert arrival_dv == pytest.approx(
        _circular_speed_km_s(r2_km) - speed_at_apogee, abs=VELOCITY_TOL_KM_S
    )

    # Geometry: departure and arrival in the equatorial plane, 180 degrees
    # apart, on prograde circular orbits.
    r1 = np.array([r1_km, 0.0, 0.0])
    r2 = np.array([-r2_km, 0.0, 0.0])
    departure_circular = np.array([0.0, _circular_speed_km_s(r1_km), 0.0])
    arrival_circular = np.array([0.0, -_circular_speed_km_s(r2_km), 0.0])

    solutions = solve_lambert_all(
        r1,
        r2,
        tof_s,
        MU_KM3_S2,
        endpoint_tolerance_km=1.0e-3,
        reference_normal=np.array([0.0, 0.0, 1.0]),
    )

    # Both angular-momentum directions and nothing spurious. The retrograde
    # half-ellipse is the same conic flown the other way round, so it reaches
    # the same point at the same time for a much larger impulse.
    assert len(solutions) == 2, [s.branch for s in solutions]

    def total_dv(solution: LambertSolution) -> float:
        return float(np.linalg.norm(solution.v1 - departure_circular)) + float(
            np.linalg.norm(arrival_circular - solution.v2)
        )

    cheapest = min(solutions, key=total_dv)
    dearest = max(solutions, key=total_dv)

    assert float(np.linalg.norm(cheapest.v1 - departure_circular)) == pytest.approx(
        departure_dv, abs=VELOCITY_TOL_KM_S
    )
    assert float(np.linalg.norm(arrival_circular - cheapest.v2)) == pytest.approx(
        arrival_dv, abs=VELOCITY_TOL_KM_S
    )
    assert total_dv(cheapest) == pytest.approx(
        departure_dv + arrival_dv, abs=VELOCITY_TOL_KM_S
    )
    # The wrong branch is not marginally worse, it is 18 km/s worse. A solver
    # that lost track of which direction it was flying could not hide here.
    assert total_dv(dearest) > total_dv(cheapest) + 10.0

    # The transfer conic itself, not just its cost. This is what would catch a
    # solver that happened to produce the right impulse magnitudes off a wrong
    # orbit, and -- through the period identity -- a solver that stopped
    # reading tof_s as seconds.
    transfer = cartesian_to_keplerian(
        StateVector(tuple(float(x) for x in r1), cheapest.v1_km_s), MU_KM3_S2
    )
    assert transfer.sma_km == pytest.approx(transfer_sma_km, rel=RELATIVE_TOL)
    assert transfer.ecc == pytest.approx(transfer_ecc, abs=RELATIVE_TOL)
    assert orbital_period_s(transfer.sma_km, MU_KM3_S2) == pytest.approx(
        2.0 * tof_s, rel=RELATIVE_TOL
    )

    # Coplanar in means coplanar out. A stray out-of-plane component would be
    # free delta-v spent on nothing.
    assert abs(float(cheapest.v1[2])) < VELOCITY_TOL_KM_S
    assert abs(float(cheapest.v2[2])) < VELOCITY_TOL_KM_S
    assert transfer.inc_deg == pytest.approx(0.0, abs=ANGLE_TOL_DEG)

    # And the answer is an answer: it actually arrives where it was asked to.
    assert cheapest.endpoint_error_km < POSITION_TOL_KM
    assert _true_endpoint_miss_km(r1, cheapest.v1, tof_s, r2) < POSITION_TOL_KM


@pytest.mark.parametrize(
    ("sma_km", "ecc", "inc_deg", "raan_deg", "aop_deg", "ta_deg", "period_fraction"),
    [
        # Circular: departure and arrival radii are equal.
        (8000.0, 0.0, 63.4, 40.0, 0.0, 0.0, 0.125),
        (8000.0, 0.0, 63.4, 40.0, 0.0, 0.0, 0.25),
        (7200.0, 0.0, 28.5, 0.0, 0.0, 0.0, 0.25),
        (7000.0, 0.0, 98.0, 200.0, 0.0, 0.0, 0.4),
        # Eccentric: departure and arrival radii differ, which is what makes an
        # r1/r2 transposition inside the Lagrange coefficients visible at all.
        # On a circular arc such a swap is arithmetically invisible.
        (9000.0, 0.25, 47.0, 120.0, 33.0, 10.0, 0.18),
        (9000.0, 0.25, 47.0, 120.0, 33.0, 200.0, 0.3),
        (11000.0, 0.42, 75.0, 310.0, 270.0, 95.0, 0.22),
    ],
)
def test_known_arc_costs_nothing_on_the_universal_variable_path(
    sma_km: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float,
    aop_deg: float,
    ta_deg: float,
    period_fraction: float,
) -> None:
    """The other closed form: an arc of the orbit you are already on is free.

    Two points on one orbit, separated by the time it actually takes to fly
    between them, have exactly one sensible Lambert answer -- the velocity you
    already had. So the required impulse is exactly zero and every component of
    the answer is known in advance, including the out-of-plane one. The truth
    here comes from ``keplerian_to_cartesian``, which shares no code with
    ``lambert.py``.

    The out-of-plane component is the point. The solver may not collapse
    different orbital planes onto a coplanar Hohmann assumption, and a solver
    that had quietly done so would return a velocity lying in the equatorial
    plane. It would still be a plausible-looking vector of about the right
    magnitude. It would miss the true velocity by kilometres per second, and
    the zero here would become 6.3.

    Both circular and eccentric arcs are covered because they fail differently.
    A circular arc pins the plane but cannot see a departure/arrival radius
    transposition, since the two radii are the same number.
    """
    initial = keplerian_to_cartesian(
        KeplerianElements(sma_km, ecc, inc_deg, raan_deg, aop_deg, ta_deg), MU_KM3_S2
    )
    tof_s = orbital_period_s(sma_km, MU_KM3_S2) * period_fraction
    arrival = propagate_universal(initial, tof_s, MU_KM3_S2)
    departure_radius_km = float(np.linalg.norm(initial.r))
    arrival_radius_km = float(np.linalg.norm(arrival.r))
    if ecc == 0.0:
        assert departure_radius_km == pytest.approx(arrival_radius_km, rel=1.0e-12)
    else:
        # Otherwise the eccentric cases would not be testing what they claim.
        assert abs(departure_radius_km - arrival_radius_km) > 100.0

    solutions = solve_lambert_all(
        initial.r,
        arrival.r,
        tof_s,
        MU_KM3_S2,
        endpoint_tolerance_km=1.0e-3,
        reference_normal=np.cross(initial.r, initial.v),
    )
    assert solutions

    # Every one of these came off the universal-variable root scan, not the
    # shooting fallback, which records z as NaN.
    assert all(math.isfinite(solution.z) for solution in solutions)

    free = min(solutions, key=lambda s: float(np.linalg.norm(s.v1 - initial.v)))
    assert float(np.linalg.norm(free.v1 - initial.v)) < VELOCITY_TOL_KM_S
    assert float(np.linalg.norm(free.v2 - arrival.v)) < VELOCITY_TOL_KM_S

    # Stated again as the thing that would break: the out-of-plane component is
    # reproduced, not dropped.
    assert float(free.v1[2]) == pytest.approx(
        float(initial.v[2]), abs=VELOCITY_TOL_KM_S
    )
    assert abs(float(initial.v[2])) > 1.0

    # A circular arc has a closed-form speed as well as a known vector.
    if ecc == 0.0:
        assert float(np.linalg.norm(free.v1)) == pytest.approx(
            _circular_speed_km_s(sma_km), abs=VELOCITY_TOL_KM_S
        )

    # The conic is the orbit it started on.
    transfer = cartesian_to_keplerian(
        StateVector(initial.r_km, free.v1_km_s), MU_KM3_S2
    )
    assert transfer.sma_km == pytest.approx(sma_km, rel=RELATIVE_TOL)
    assert transfer.ecc == pytest.approx(ecc, abs=1.0e-12)
    assert transfer.inc_deg == pytest.approx(inc_deg, abs=ANGLE_TOL_DEG)

    assert free.endpoint_error_km < POSITION_TOL_KM


# ---------------------------------------------------------------------------
# 2. Near-antipodal geometry
# ---------------------------------------------------------------------------

# A 7000 km circular pursuer, a 7100 km target, and exactly the flight time of
# the half-ellipse between them. Everything in this section sits at or just off
# the 180 degree mark and watches what happens.
_NEAR_R1_KM = 7000.0
_NEAR_R2_KM = 7100.0
_NEAR_SMA_KM = 0.5 * (_NEAR_R1_KM + _NEAR_R2_KM)
_NEAR_TOF_S = math.pi * math.sqrt(_NEAR_SMA_KM**3 / MU_KM3_S2)
_NEAR_R1 = np.array([_NEAR_R1_KM, 0.0, 0.0])
_NEAR_DEPARTURE_V = np.array([0.0, _circular_speed_km_s(_NEAR_R1_KM), 0.0])
_PURSUER_PLANE_NORMAL = np.array([0.0, 0.0, 1.0])
# Speed at the near end of that half-ellipse.
_TRANSFER_SPEED_KM_S = math.sqrt(
    MU_KM3_S2 * (2.0 / _NEAR_R1_KM - 1.0 / _NEAR_SMA_KM)
)


def _near_antipodal_arrival(offset_deg: float, tilt_deg: float) -> np.ndarray:
    """Arrival point ``offset_deg`` short of antipodal, tilted out of plane.

    ``tilt_deg`` rotates the small transverse part of the arrival vector out of
    the pursuer's plane. At ``offset_deg = 0`` the two endpoints are collinear
    and ``tilt_deg`` has no effect at all -- which is exactly the degeneracy:
    the plane of the transfer stops being a function of the endpoints.
    """
    offset = math.radians(offset_deg)
    tilt = math.radians(tilt_deg)
    transverse = np.array([0.0, math.cos(tilt), math.sin(tilt)])
    radial = np.array([-math.cos(offset), 0.0, 0.0])
    return _NEAR_R2_KM * (radial + math.sin(offset) * transverse)


def test_antipodal_transfer_plane_is_chosen_by_the_caller_not_by_the_geometry() -> None:
    """Measured behaviour at exactly 180 degrees, pinned so it cannot drift.

    Two collinear position vectors and a flight time do not determine an orbit
    plane: every plane through the line joining them contains a valid transfer.
    ``solve_lambert_all`` neither refuses nor guesses badly. It hands the
    decision to ``reference_normal`` and returns a transfer that genuinely
    arrives, to about 1e-11 km, in whichever plane the caller named.

    The consequence is worth stating plainly, because it is the hazard rather
    than the bug: for one unchanged boundary-value problem the answer's cost
    ranges from 0.027 km/s to 10.69 km/s depending only on the hint. The hint
    is not a preference here, it is the missing half of the question.
    ``search.py`` supplies the pursuer's own orbit normal, which is the right
    choice; a caller that supplied nothing would be relying on
    ``_shooting_fallback``'s deterministic axis pick, which is arbitrary.
    """
    r2 = np.array([-_NEAR_R2_KM, 0.0, 0.0])

    # Closed forms for the two extremes. The coplanar transfer only has to
    # change speed; the polar one leaves the departure velocity untouched and
    # adds an orthogonal one, so Pythagoras gives its cost.
    coplanar_dv = _TRANSFER_SPEED_KM_S - _circular_speed_km_s(_NEAR_R1_KM)
    polar_dv = math.hypot(_TRANSFER_SPEED_KM_S, _circular_speed_km_s(_NEAR_R1_KM))

    def cheapest_in_plane(normal: np.ndarray) -> LambertSolution:
        solutions = solve_lambert_all(
            _NEAR_R1,
            r2,
            _NEAR_TOF_S,
            MU_KM3_S2,
            endpoint_tolerance_km=1.0e-3,
            reference_normal=normal,
        )
        assert solutions, f"no solution for reference_normal={normal}"
        for solution in solutions:
            # The only machine-readable sign that the plane was not derived
            # from the geometry: the shooting fallback records no z root.
            assert math.isnan(solution.z)
            assert solution.direction.endswith("-shooting")
            assert solution.endpoint_error_km < POSITION_TOL_KM
            assert (
                _true_endpoint_miss_km(_NEAR_R1, solution.v1, _NEAR_TOF_S, r2)
                < POSITION_TOL_KM
            )
        return min(
            solutions, key=lambda s: float(np.linalg.norm(s.v1 - _NEAR_DEPARTURE_V))
        )

    in_pursuer_plane = cheapest_in_plane(_PURSUER_PLANE_NORMAL)
    polar = cheapest_in_plane(np.array([0.0, 1.0, 0.0]))
    tilted = cheapest_in_plane(np.array([0.0, 1.0, 1.0]) / math.sqrt(2.0))

    # 1e-6 deg rather than ANGLE_TOL_DEG: these planes are reconstructed by a
    # least-squares shooter rather than in closed form, and the whole point of
    # the geometry is that the plane is barely determined. The claim is which
    # plane came back, not that it came back to machine precision.
    assert _transfer_inclination_deg(_NEAR_R1, in_pursuer_plane.v1) == pytest.approx(
        0.0, abs=1.0e-6
    )
    assert _transfer_inclination_deg(_NEAR_R1, polar.v1) == pytest.approx(90.0, abs=1.0e-6)
    assert _transfer_inclination_deg(_NEAR_R1, tilted.v1) == pytest.approx(45.0, abs=1.0e-6)

    measured_coplanar = float(np.linalg.norm(in_pursuer_plane.v1 - _NEAR_DEPARTURE_V))
    measured_polar = float(np.linalg.norm(polar.v1 - _NEAR_DEPARTURE_V))
    assert measured_coplanar == pytest.approx(coplanar_dv, abs=VELOCITY_TOL_KM_S)
    assert measured_polar == pytest.approx(polar_dv, abs=VELOCITY_TOL_KM_S)
    # Same r1, same r2, same tof, same mu; four hundred times the fuel. Stated
    # on the solver's own two answers rather than on the closed forms, because
    # the spread is the finding and it should not be provable from algebra
    # alone that this file wrote.
    assert measured_polar > measured_coplanar + 10.0
    assert measured_polar / measured_coplanar > 100.0


def test_near_antipodal_out_of_plane_target_yields_nothing_rather_than_a_wrong_burn() -> None:
    """Characterisation of the narrow band where the solver returns an empty list.

    Just off 180 degrees the universal-variable reconstruction loses endpoint
    accuracy fast: measured on this geometry the miss grows from 1.5e-7 km at
    0.1 degrees short of antipodal to 1.4e-4 km at 0.01 degrees and 2.7e-3 km
    at 0.001 degrees. Where that loss exceeds the caller's
    ``endpoint_tolerance_km`` the branch is thrown away, and the shooting
    fallback is then the only route to an answer -- but its trial velocities
    are aimed by ``reference_normal``, and at exactly half a revolution an
    in-plane trial velocity has no out-of-plane response to follow. Pointed at
    the pursuer's plane it cannot find a target sitting outside it.

    So nothing comes back. That is the fail-closed outcome asked for
    and is much better than a confident wrong impulse. It is still a branch
    that exists and is not found: drop the hint and the same call returns the
    correct polar transfer to 5e-13 km/s. The band is reachable at shipped
    tolerances -- 1e-4 degrees at ``deep``'s 0.01 km, 3e-4 degrees at
    ``standard``'s 0.02 km -- and is left as measured rather than papered over.

    If this assertion ever fails because solutions appeared, that is probably
    an improvement. Check them against the no-hint answer below before
    deleting anything.
    """
    r2 = _near_antipodal_arrival(1.0e-4, 90.0)
    assert float(r2[2]) == pytest.approx(0.0123918, abs=1.0e-6)

    deep_profile_tolerance_km = 0.01
    with_pursuer_plane_hint = solve_lambert_all(
        _NEAR_R1,
        r2,
        _NEAR_TOF_S,
        MU_KM3_S2,
        endpoint_tolerance_km=deep_profile_tolerance_km,
        reference_normal=_PURSUER_PLANE_NORMAL,
    )
    assert with_pursuer_plane_hint == [], (
        "Characterisation, not a requirement. Solutions appearing here is "
        "probably an improvement -- check them against the no-hint answer "
        "below, which is the branch this geometry is currently losing."
    )

    without_hint = solve_lambert_all(
        _NEAR_R1,
        r2,
        _NEAR_TOF_S,
        MU_KM3_S2,
        endpoint_tolerance_km=deep_profile_tolerance_km,
    )
    assert without_hint
    polar_dv = math.hypot(_TRANSFER_SPEED_KM_S, _circular_speed_km_s(_NEAR_R1_KM))
    for solution in without_hint:
        assert float(np.linalg.norm(solution.v1 - _NEAR_DEPARTURE_V)) == pytest.approx(
            polar_dv, abs=VELOCITY_TOL_KM_S
        )
        assert solution.endpoint_error_km < POSITION_TOL_KM
        # 1e-3 deg: the target is 1e-4 deg short of antipodal, so the true
        # transfer is very nearly but not exactly polar.
        assert _transfer_inclination_deg(_NEAR_R1, solution.v1) == pytest.approx(
            90.0, abs=1.0e-3
        )


def test_no_near_antipodal_solution_understates_its_own_endpoint_miss() -> None:
    """The invariant the two tests above lean on, checked over the whole band.

    Ill-conditioning is survivable as long as the solver is honest about it.
    Whatever plane it lands in, every returned solution must arrive where
    ``endpoint_error_km`` says it does, and within the tolerance it was given.
    The measurement behind it was wider than the test. Sweeping the offset over
    nine decades at two mantissas, three out-of-plane tilts, three tolerances,
    with and without a plane hint, the reported miss matched an independent
    re-propagation exactly: 839 solutions, worst discrepancy 0.0 km. That is
    why this is an equality and not a bound.

    What runs here is a subset of the same grid, because the full one takes
    five minutes -- most of these calls fall through to the shooting fallback,
    which is twelve least-squares solves each, and this belongs in the fast
    suite rather than behind the slow marker. Nothing is relaxed, only fewer
    points are visited: every offset decade is still covered, in the plane and
    perpendicular to it, with and without a plane hint, at ``deep``'s 0.01 km,
    the tightest tolerance any shipped profile uses.
    """
    tolerance_km = 0.01
    checked = 0
    for exponent in range(-9, 1):
        offset_deg = 10.0**exponent
        for tilt_deg in (0.0, 90.0):
            r2 = _near_antipodal_arrival(offset_deg, tilt_deg)
            for normal in (_PURSUER_PLANE_NORMAL, None):
                for solution in solve_lambert_all(
                    _NEAR_R1,
                    r2,
                    _NEAR_TOF_S,
                    MU_KM3_S2,
                    endpoint_tolerance_km=tolerance_km,
                    reference_normal=normal,
                ):
                    miss = _true_endpoint_miss_km(_NEAR_R1, solution.v1, _NEAR_TOF_S, r2)
                    assert solution.endpoint_error_km == pytest.approx(
                        miss, abs=1.0e-12
                    ), (offset_deg, tilt_deg, solution.branch)
                    assert solution.endpoint_error_km <= tolerance_km
                    checked += 1
    assert checked > 50, checked


# ---------------------------------------------------------------------------
# 3. Fallback-path canary
# ---------------------------------------------------------------------------
#
# There is no `simple_lambert` in this tree, and no simplified Lambert
# formulation anywhere: `solve_lambert_all` is the only public entry point and
# `_lambert_function_factory` the only universal-variable implementation. What
# does exist is a second, independent solver -- `_shooting_fallback`, a
# Cartesian least-squares shooter that `solve_lambert_all` reaches for only
# when the universal-variable scan returns nothing at all. On well-conditioned
# geometry it is therefore never executed by the public API and nothing else in
# the suite calls it, so it can drift out of agreement with the primary solver
# without a single test going red. These are the canaries for that.


def _known_arc(
    elements: KeplerianElements, tof_s: float
) -> tuple[StateVector, StateVector]:
    initial = keplerian_to_cartesian(elements, MU_KM3_S2)
    return initial, propagate_universal(initial, tof_s, MU_KM3_S2)


def test_shooting_fallback_agrees_with_the_universal_variable_solver() -> None:
    """Same question, two independent solvers, one answer.

    ``_shooting_fallback`` shoots in Cartesian velocity space and never touches
    the Stumpff functions; the primary path solves for the universal variable z
    and reconstructs the velocity from Lagrange coefficients. They share only
    ``propagate_universal``. Where both find a branch they agreed to within
    7.7e-15 km/s on every case measured, so 1e-9 km/s here is slack, not a
    concession.

    Containment is asserted in one direction only, and deliberately. The
    fallback is a local method seeded from a handful of in-plane guesses: on
    the eighth-period arc it does not find the 315-degree retrograde branch at
    all, which sits 16.1 km/s away from anything it tries. That is the
    documented reason it is a fallback. What it must never do is invent a
    branch the primary solver does not have, or return a different velocity for
    one it does.
    """
    cases: list[tuple[str, np.ndarray, np.ndarray, float]] = [
        # The existing Vallado reference geometry, from test_lambert.py.
        (
            "vallado",
            np.array([5000.0, 10000.0, 2100.0]),
            np.array([-14600.0, 2500.0, 7000.0]),
            3600.0,
        ),
    ]
    for fraction in (0.125, 0.25, 0.4):
        elements = KeplerianElements(8000.0, 0.0, 63.4, 40.0, 0.0, 0.0)
        tof_s = orbital_period_s(8000.0, MU_KM3_S2) * fraction
        initial, arrival = _known_arc(elements, tof_s)
        cases.append((f"inclined-circular-{fraction}", initial.r, arrival.r, tof_s))
    eccentric = KeplerianElements(9000.0, 0.25, 47.0, 120.0, 33.0, 10.0)
    initial, arrival = _known_arc(eccentric, 3300.0)
    cases.append(("eccentric-inclined", initial.r, arrival.r, 3300.0))

    for name, r1, r2, tof_s in cases:
        reference_normal = np.cross(r1, r2)
        primary = solve_lambert_all(
            r1,
            r2,
            tof_s,
            MU_KM3_S2,
            endpoint_tolerance_km=1.0e-3,
            reference_normal=reference_normal,
        )
        fallback = _shooting_fallback(
            r1, r2, tof_s, MU_KM3_S2, 1.0e-3, reference_normal
        )
        assert primary, name
        assert fallback, name
        # Nothing here should have needed the fallback: if the primary path
        # started producing NaN z values on clean geometry the comparison below
        # would be the fallback against itself.
        assert all(math.isfinite(solution.z) for solution in primary), name

        for shot in fallback:
            gap = min(
                float(np.linalg.norm(shot.v1 - solution.v1)) for solution in primary
            )
            assert gap < VELOCITY_TOL_KM_S, (name, shot.branch, gap)
            assert shot.endpoint_error_km < POSITION_TOL_KM, (name, shot.branch)
            assert (
                _true_endpoint_miss_km(r1, shot.v1, tof_s, r2) < POSITION_TOL_KM
            ), (name, shot.branch)


def test_shooting_fallback_reproduces_a_velocity_known_without_either_solver() -> None:
    """Agreement is not enough on its own: both could drift the same way.

    Two points on one circular orbit and the time between them have a known
    departure velocity -- the circular velocity. Ask the fallback directly and
    it must return that, including the out-of-plane component, without the
    primary solver being involved at all.
    """
    for inc_deg, raan_deg, sma_km, fraction in (
        (63.4, 40.0, 8000.0, 0.25),
        (28.5, 0.0, 7200.0, 0.125),
        (98.0, 200.0, 7000.0, 0.4),
    ):
        elements = KeplerianElements(sma_km, 0.0, inc_deg, raan_deg, 0.0, 0.0)
        tof_s = orbital_period_s(sma_km, MU_KM3_S2) * fraction
        initial, arrival = _known_arc(elements, tof_s)
        shots = _shooting_fallback(
            initial.r,
            arrival.r,
            tof_s,
            MU_KM3_S2,
            1.0e-3,
            np.cross(initial.r, initial.v),
        )
        assert shots, (inc_deg, fraction)
        best = min(shots, key=lambda s: float(np.linalg.norm(s.v1 - initial.v)))
        assert float(np.linalg.norm(best.v1 - initial.v)) < VELOCITY_TOL_KM_S
        assert float(np.linalg.norm(best.v2 - arrival.v)) < VELOCITY_TOL_KM_S
        assert abs(float(initial.v[2])) > 1.0
        assert float(best.v1[2]) == pytest.approx(
            float(initial.v[2]), abs=VELOCITY_TOL_KM_S
        )
