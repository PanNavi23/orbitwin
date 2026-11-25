from __future__ import annotations

import math
from functools import partial

import numpy as np

from orbitwin.errors import DynamicsError
from orbitwin.lambert import solve_lambert_all
from orbitwin.orbit import (
    KeplerianElements,
    keplerian_to_cartesian,
    orbital_period_s,
    propagate_universal,
)

MU = 398600.4418


def test_vallado_reference_case() -> None:
    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    solutions = solve_lambert_all(r1, r2, 3600.0, MU, endpoint_tolerance_km=1e-3)
    reference = np.array([-5.99249502, 1.92536671, 3.24563805])
    assert any(np.linalg.norm(solution.v1 - reference) < 1e-6 for solution in solutions)


def test_different_plane_endpoint_reconstruction() -> None:
    initial = keplerian_to_cartesian(KeplerianElements(8000, 0.1, 52, 35, 15, 70), MU)
    arrival_state = propagate_universal(initial, 4100.0, MU)
    r1 = np.array([7000.0, 0.0, 0.0])
    solutions = solve_lambert_all(
        r1,
        arrival_state.r,
        4100.0,
        MU,
        endpoint_tolerance_km=1e-3,
        reference_normal=np.array([0.0, 0.0, 1.0]),
    )
    assert solutions
    assert min(solution.endpoint_error_km for solution in solutions) < 1e-5
    assert any(abs(solution.v1[2]) > 1e-3 for solution in solutions)


def test_multiple_revolution_roots_are_enumerated() -> None:
    initial = keplerian_to_cartesian(KeplerianElements(8000, 0.05, 35, 20, 10, 30), MU)
    period = orbital_period_s(8000, MU)
    tof = 1.8 * period
    arrival = propagate_universal(initial, tof, MU)
    solutions = solve_lambert_all(
        initial.r,
        arrival.r,
        tof,
        MU,
        max_revolutions=2,
        samples_per_interval=180,
        endpoint_tolerance_km=1e-3,
        reference_normal=np.cross(initial.r, initial.v),
    )
    assert any(solution.revolution == 1 for solution in solutions)
    assert any(solution.revolution == 2 for solution in solutions)


def test_antipodal_shooting_fallback() -> None:
    r1 = np.array([6900.0, 0.0, 0.0])
    r2 = np.array([-9600.0, 0.0, 0.0])
    solutions = solve_lambert_all(
        r1,
        r2,
        3728.738,
        MU,
        endpoint_tolerance_km=1e-3,
        reference_normal=np.array([0.0, 0.0, 1.0]),
    )
    assert solutions
    assert min(solution.endpoint_error_km for solution in solutions) < 1e-6


def test_the_tangent_root_guard_evaluates_each_point_once() -> None:
    """`function` is pure, so calling it twice for one point is pure waste.

    The tangent-root guard used a conditional expression that called it once to
    test finiteness and once for the value. Measured on the 2026-09-05 scenario
    that was a tenth of every residual evaluation a one-burn sweep performed,
    all of it redundant.

    The signature of that bug is exact and does not depend on iteration counts:
    the same z arrives twice in a row. Nothing else in `_roots_in_interval`
    evaluates a point consecutively -- the sampling grid is strictly increasing
    and brentq never repeats its last bracket point immediately.
    """
    from orbitwin.lambert import _roots_in_interval

    seen: list[float] = []

    def recorded(z: float) -> float:
        seen.append(z)
        # A parabola with two real roots inside the interval, so the bracketing
        # path, the brentq path and the tangent guard all run.
        return (z - 3.0) ** 2 - 1.0

    _roots_in_interval(recorded, 0.0, 8.0, samples=32, root_tolerance=1.0e-7)

    assert len(seen) > 32, "the guard did not run at all"
    repeats = [
        (index, value)
        for index, value in enumerate(seen[1:], start=1)
        if value == seen[index - 1]
    ]
    assert not repeats, (
        f"{len(repeats)} point(s) evaluated twice in a row, first at index "
        f"{repeats[0][0]} with z={repeats[0][1]!r}"
    )


def test_the_vectorised_residual_agrees_with_the_scalar_one() -> None:
    """Close enough that it cannot move a bracket. Not closer, and not asserted so.

    numpy picks its float64 transcendental kernels by CPU at run time, so
    `np.cos` and `math.cos` agree to the last bit on some machines and differ in
    their last place on others -- this repository's CI showed both, passing on
    the 3.11 runner and failing on 3.13 and 3.14 with numpy pinned to the same
    2.4.6 on all three.

    The bound is measured, not chosen. Patching numpy's cos, sin, cosh and sinh
    to move every result one ulp -- harsher than a real kernel disagreement,
    which is systematic rather than independent per call -- gives:

        stumpff_c  7.1e-16    stumpff_s  4.3e-16    residual  2.2e-14

    scaled by max(|value|, 1). Neither ulp counts nor relative error work here:
    both functions have zeros. Before the half-angle C and the series S the
    same experiment gave 2.2e-8, 1.4e-8 and 6.7e-9: `1 - cos` and
    `root - sin(root)` cancelled, 2,400 to one at root = 0.05.

    1e-12 is asserted: twenty times the worst measured with the shift either
    way (4.6e-14 at minus one ulp), tight enough to catch a wrong branch or a
    mistyped constant. The cancellation itself is guarded by
    test_orbit.test_stumpff_is_accurate_where_its_closed_form_cancels.

    The NaN mask is not asserted equal. Under the same one-ulp experiment it is
    not, because a sample sitting on `C <= 0` or `y < 0` can fall either side.
    `test_the_sampler_changes_no_root` is what absorbs that, and it is the
    contract.
    """
    import math as stdlib_math

    from orbitwin.lambert import _lambert_function_factory, _lambert_residual_array
    from orbitwin.orbit import stumpff_c, stumpff_c_array, stumpff_s, stumpff_s_array

    grid = np.concatenate(
        [
            np.linspace(-16.0 * stdlib_math.pi**2, -1.0e-10, 800),
            np.linspace(1.0e-10, (4.0 * stdlib_math.pi) ** 2, 800),
            np.linspace(-1.0e-8, 1.0e-8, 200),
            np.array([-160000.0, -122500.0, 0.0]),
        ]
    )

    def scalar_or_nan(scalar_fn, values: np.ndarray) -> np.ndarray:
        out = np.empty(values.size, dtype=float)
        for index, value in enumerate(values):
            try:
                out[index] = scalar_fn(float(value))
            except DynamicsError:
                out[index] = stdlib_math.nan
        return out

    def worst_scaled(expected: np.ndarray, actual: np.ndarray) -> float:
        usable = ~np.isnan(expected) & ~np.isnan(actual)
        assert usable.sum() > expected.size // 2, "most of the grid should be usable"
        return float(
            np.max(
                np.abs(expected[usable] - actual[usable])
                / np.maximum(np.abs(expected[usable]), 1.0)
            )
        )

    for scalar_fn, array_fn, name in (
        (stumpff_c, stumpff_c_array, "stumpff_c"),
        (stumpff_s, stumpff_s_array, "stumpff_s"),
    ):
        apart = worst_scaled(scalar_or_nan(scalar_fn, grid), array_fn(grid))
        assert apart <= 1.0e-12, f"{name}: {apart:.3g} apart, scaled"

    r1mag, r2mag, a_term = 6800.0, 7100.0, 5200.0
    mu, tof = 398600.4418, 5600.0
    function = _lambert_function_factory(r1mag, r2mag, a_term, tof, mu)
    apart = worst_scaled(
        np.array([function(float(z)) for z in grid]),
        _lambert_residual_array(grid, r1mag, r2mag, a_term, stdlib_math.sqrt(mu) * tof),
    )
    assert apart <= 1.0e-12, f"residual: {apart:.3g} apart, scaled"


def test_the_sampler_changes_no_root() -> None:
    """The contract: the vectorised path is an accelerator, not a second answer.

    "The same roots" cannot mean bit-equal, and that is measured rather than
    conceded. Under a one-ulp kernel disagreement the sampler can hand brentq a
    bracket one grid cell over, and brentq -- which converges to xtol = rtol =
    1e-12 -- then lands on a neighbouring representable number: 54.56630188816812
    against ...13, observed.

    What must hold, and does, is that the same number of roots come back and
    that each agrees to well inside brentq's own tolerance. Measured across
    every interval `_root_intervals` produces for five geometries, under that
    same one-ulp experiment: 25 of 25 intervals matched in count, worst relative
    difference 2.3e-15.

    1e-12 is asserted, which is brentq's tolerance and 400 times the worst
    measured. A root that moved more than that would mean a different bracket,
    not a different last digit.
    """
    from orbitwin.lambert import (
        _lambert_function_factory,
        _lambert_residual_array,
        _roots_in_interval,
    )

    mu = 398600.4418

    from orbitwin.lambert import _root_intervals

    checked = 0
    for r1mag, r2mag, a_term, tof in (
        (6800.0, 7100.0, 5200.0, 5600.0),
        (6800.0, 7100.0, -5200.0, 5600.0),
        (7000.0, 7000.0, 6800.0, 12000.0),
        (6600.0, 9000.0, 3100.0, 2500.0),
        (6800.0, 7100.0, 5200.0, 19000.0),
    ):
        function = _lambert_function_factory(r1mag, r2mag, a_term, tof, mu)
        sampler = partial(
            _lambert_residual_array,
            r1mag=r1mag,
            r2mag=r2mag,
            a_term=a_term,
            sqrt_mu_t=math.sqrt(mu) * tof,
        )
        for left, right, _ in _root_intervals(3):
            without = _roots_in_interval(
                function, left, right, samples=128, root_tolerance=1.0e-7
            )
            with_sampler = _roots_in_interval(
                function, left, right, samples=128, root_tolerance=1.0e-7, sampler=sampler
            )
            assert len(without) == len(with_sampler), (
                f"a_term={a_term} tof={tof} interval=({left:.6g}, {right:.6g}) "
                f"found {len(without)} roots without the sampler and "
                f"{len(with_sampler)} with it:\n"
                f"  scalar   {without}\n  sampled  {with_sampler}"
            )
            for expected_root, sampled_root in zip(without, with_sampler, strict=True):
                assert abs(expected_root - sampled_root) <= 1.0e-12 * max(
                    abs(expected_root), 1.0
                ), (
                    f"a_term={a_term} tof={tof}: root {expected_root!r} became "
                    f"{sampled_root!r}, further than brentq's own tolerance"
                )
            checked += 1
    assert checked >= 25, f"only {checked} intervals compared"


def test_the_array_stumpff_forms_are_quiet_on_input_the_scalar_ones_refuse() -> None:
    """NaN is the contract. A RuntimeWarning on the way to it is not.

    The scalar forms raise on a non-finite argument -- ValueError for +inf,
    DynamicsError past the hyperbolic overflow limit -- and the array forms
    answer NaN for the same elements, because an array has no way to raise for
    one of them. That much is deliberate.

    What was not deliberate: the positive branch computed `1 - cos(sqrt(z))`
    outside any errstate while the hyperbolic branch had one, so +inf produced
    the right answer and printed `invalid value encountered in cos` on the way.
    A library that writes to stderr for an input it handles is a library whose
    warnings get switched off rather than read.
    """
    import warnings

    from orbitwin.orbit import stumpff_c_array, stumpff_s_array

    hostile = np.array([np.inf, -np.inf, np.nan, -200000.0, 0.0, 1.0, -1.0])

    for name, array_fn in (("stumpff_c_array", stumpff_c_array), ("stumpff_s_array", stumpff_s_array)):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(all="warn"):
                out = array_fn(hostile)

        assert np.isnan(out[:4]).all(), f"{name}: refused input must come back NaN"
        assert np.isfinite(out[4:]).all(), f"{name}: ordinary input must survive"
