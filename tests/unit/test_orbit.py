from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from orbitwin.errors import DynamicsError
from orbitwin.models import StateVector
from orbitwin.orbit import (
    KeplerianElements,
    angular_momentum,
    cartesian_to_keplerian,
    keplerian_to_cartesian,
    orbital_period_s,
    propagate_universal,
    specific_energy,
    stumpff_c,
    stumpff_s,
)

MU = 398600.4418


@pytest.mark.parametrize("z", [-4.0, -1e-10, 0.0, 1e-10, 4.0])
def test_stumpff_identity_and_finiteness(z: float) -> None:
    c = stumpff_c(z)
    s = stumpff_s(z)
    assert math.isfinite(c) and math.isfinite(s)
    assert c > 0.0 and s > 0.0


def _stumpff_by_decimal_series(z: float, first: int) -> float:
    """sum (-z)^k / (2k + first)!, in 50 digits; first is 2 for C and 3 for S."""
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 50
        x = Decimal(z)
        term = Decimal(1) / math.factorial(first)
        total, k = term, 0
        while abs(term) > Decimal(10) ** -45:
            k += 1
            term *= -x / ((2 * k + first - 1) * (2 * k + first))
            total += term
        return float(total)


def test_stumpff_is_accurate_where_its_closed_form_cancels() -> None:
    """1 - cos and root - sin cancel for small z: 1.9e-8 and 4.5e-8 relative before."""
    from orbitwin.orbit import stumpff_c_array, stumpff_s_array

    grid = np.concatenate([-np.logspace(-9.0, 1.5, 211), np.logspace(-9.0, 1.5, 211)])
    for scalar, array, first in ((stumpff_c, stumpff_c_array, 2), (stumpff_s, stumpff_s_array, 3)):
        want = np.array([_stumpff_by_decimal_series(float(z), first) for z in grid])
        for got in (np.array([scalar(float(z)) for z in grid]), array(grid)):
            assert np.max(np.abs(got - want) / want) < 4.0e-15


def test_keplerian_cartesian_roundtrip() -> None:
    elements = KeplerianElements(9200.0, 0.12, 63.4, 240.0, 15.0, 222.0)
    state = keplerian_to_cartesian(elements, MU)
    recovered = cartesian_to_keplerian(state, MU)
    assert recovered.sma_km == pytest.approx(elements.sma_km, rel=1e-11)
    assert recovered.ecc == pytest.approx(elements.ecc, abs=1e-12)
    assert recovered.inc_deg == pytest.approx(elements.inc_deg, abs=1e-10)
    assert recovered.raan_deg == pytest.approx(elements.raan_deg, abs=1e-10)
    assert recovered.aop_deg == pytest.approx(elements.aop_deg, abs=1e-9)
    assert recovered.ta_deg == pytest.approx(elements.ta_deg, abs=1e-9)


def test_one_period_returns_to_initial_state() -> None:
    state = keplerian_to_cartesian(KeplerianElements(6900, 0, 28.5, 45, 0, 250), MU)
    period = orbital_period_s(6900, MU)
    propagated = propagate_universal(state, period, MU)
    assert np.linalg.norm(propagated.r - state.r) < 1e-7
    assert np.linalg.norm(propagated.v - state.v) < 1e-10


def test_forward_backward_roundtrip() -> None:
    state = keplerian_to_cartesian(KeplerianElements(12000, 0.4, 78, 150, 22, 35), MU)
    forward = propagate_universal(state, 4567.89, MU)
    backward = propagate_universal(forward, -4567.89, MU)
    assert np.linalg.norm(backward.r - state.r) < 1e-7
    assert np.linalg.norm(backward.v - state.v) < 1e-10


def test_energy_and_momentum_conserved() -> None:
    state = keplerian_to_cartesian(KeplerianElements(10000, 0.3, 20, 40, 60, 80), MU)
    initial_energy = specific_energy(state, MU)
    initial_h = angular_momentum(state)
    propagated = propagate_universal(state, 12345.0, MU)
    assert specific_energy(propagated, MU) == pytest.approx(initial_energy, rel=1e-11)
    assert np.linalg.norm(angular_momentum(propagated) - initial_h) < 1e-7


def test_zero_time_is_exact_identity() -> None:
    state = StateVector((7000.0, 0.0, 0.0), (0.0, 7.5, 1.0))
    assert propagate_universal(state, 0.0, MU) == state


# The ranking rules (2026/10/06 §3) allow an escaping SC_Alien; GMAT writes its SMA negative.
_HYPERBOLA = KeplerianElements(-14000.0, 1.5, 120.0, 30.0, 25.0, -40.0)


def test_hyperbolic_roundtrip_in_gmat_convention() -> None:
    state = keplerian_to_cartesian(_HYPERBOLA, MU)
    recovered = cartesian_to_keplerian(state, MU)
    assert recovered.sma_km == pytest.approx(_HYPERBOLA.sma_km, rel=1e-11)
    assert recovered.ecc == pytest.approx(_HYPERBOLA.ecc, abs=1e-12)
    assert recovered.inc_deg == pytest.approx(_HYPERBOLA.inc_deg, abs=1e-10)
    assert recovered.raan_deg == pytest.approx(_HYPERBOLA.raan_deg, abs=1e-10)
    assert recovered.aop_deg == pytest.approx(_HYPERBOLA.aop_deg, abs=1e-9)
    assert recovered.ta_deg == pytest.approx(_HYPERBOLA.ta_deg % 360.0, abs=1e-9)
    r = float(np.linalg.norm(state.r))
    v = float(np.linalg.norm(state.v))
    assert v * v == pytest.approx(MU * (2.0 / r - 1.0 / _HYPERBOLA.sma_km), rel=1e-12)


def test_hyperbolic_propagation_matches_keplers_equation() -> None:
    a, e = _HYPERBOLA.sma_km, _HYPERBOLA.ecc

    def mean_anomaly(ta_deg: float) -> float:
        half = math.radians(ta_deg) / 2.0
        anomaly = 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(half))
        return e * math.sinh(anomaly) - anomaly

    # Through perigee, from -40 to +100 deg.
    dt_s = (mean_anomaly(100.0) - mean_anomaly(-40.0)) / math.sqrt(MU / (-a) ** 3)
    expected = keplerian_to_cartesian(dataclasses.replace(_HYPERBOLA, ta_deg=100.0), MU)
    propagated = propagate_universal(keplerian_to_cartesian(_HYPERBOLA, MU), dt_s, MU)
    assert np.linalg.norm(propagated.r - expected.r) < 1e-6
    assert np.linalg.norm(propagated.v - expected.v) < 1e-9


def test_hyperbola_is_bounded_by_its_asymptotes() -> None:
    limit_deg = math.degrees(math.acos(-1.0 / _HYPERBOLA.ecc))
    for inside_deg in (limit_deg - 0.5, 360.0 - limit_deg + 0.5):
        state = keplerian_to_cartesian(dataclasses.replace(_HYPERBOLA, ta_deg=inside_deg), MU)
        assert np.all(np.isfinite(state.r))
    # GMAT's own wording: TA cannot fall between 131.81 and 228.19 deg.
    with pytest.raises(DynamicsError, match=r"between 131\.81 and 228\.19 deg"):
        keplerian_to_cartesian(dataclasses.replace(_HYPERBOLA, ta_deg=limit_deg + 0.5), MU)
    # Within 1e-12 of the asymptote the radius is ~1e17 km: refused, not flown.
    grazing_deg = math.degrees(math.acos((-1.0 + 5.0e-13) / _HYPERBOLA.ecc))
    with pytest.raises(DynamicsError, match="asymptotes"):
        keplerian_to_cartesian(dataclasses.replace(_HYPERBOLA, ta_deg=grazing_deg), MU)


@pytest.mark.parametrize(
    "sma_km, ecc, message",
    [
        (14000.0, 1.5, "negative SMA"),
        (-7000.0, 0.1, "positive SMA"),
        (0.0, 0.5, "positive SMA"),
        (0.0, 1.5, "negative SMA"),
        (7000.0, 1.0, "parabola"),
        # GMAT R2026a refuses |1 - ECC| < 1e-7 on both sides.
        (7000.0, 0.9999999, "parabola"),
        (-7.0e14, 1.00000001, "parabola"),
        (math.nan, 0.1, "finite"),
        (-14000.0, math.inf, "finite"),
        (7000.0, -0.1, "ECC >= 0"),
        # e * e overflows to inf from finite inputs.
        (-1.0, 1.0e300, "non-finite state"),
    ],
)
def test_elements_naming_no_conic_are_refused(sma_km: float, ecc: float, message: str) -> None:
    with pytest.raises(DynamicsError, match=message):
        keplerian_to_cartesian(KeplerianElements(sma_km, ecc, 30.0, 0.0, 0.0, 0.0), MU)


@pytest.mark.parametrize("value", [math.nan, math.inf])
@pytest.mark.parametrize("field", ["inc_deg", "raan_deg", "aop_deg", "ta_deg"])
def test_a_non_finite_angle_is_refused(field: str, value: float) -> None:
    # inf reaches math.cos, which raises ValueError rather than DynamicsError.
    elements = dataclasses.replace(_HYPERBOLA, **{field: value})
    with pytest.raises(DynamicsError, match="finite"):
        keplerian_to_cartesian(elements, MU)


def test_just_outside_the_parabolic_band_is_accepted() -> None:
    # GMAT R2026a accepts 1.0000001: |1 - e| is 1.0000000006e-7, not below 1e-7.
    state = keplerian_to_cartesian(KeplerianElements(-7.0e10, 1.0000001, 30.0, 0.0, 0.0, 0.0), MU)
    assert np.all(np.isfinite(state.r))


@pytest.mark.parametrize("mu", [math.nan, math.inf, 0.0, -MU])
def test_an_unusable_mu_is_refused(mu: float) -> None:
    with pytest.raises(DynamicsError, match="mu must be finite"):
        keplerian_to_cartesian(_HYPERBOLA, mu)


_R_EARTH_KM = 6378.1363


@pytest.mark.parametrize(
    "elements, horizon_s, expected_s",
    [
        # Apogee 6800 km, pericenter 6300 km: from apogee, descending, ascending.
        (KeplerianElements(6550.0, 500 / 13100, 30.0, 10.0, 0.0, 180.0), 6000.0, 1978.6058927),
        (KeplerianElements(6550.0, 500 / 13100, 30.0, 10.0, 0.0, 250.0), 6000.0, 891.9812178),
        (KeplerianElements(6550.0, 500 / 13100, 30.0, 10.0, 0.0, 100.0), 6000.0, 3214.3854255),
        # The same ascending start with the crossing after the horizon.
        (KeplerianElements(6550.0, 500 / 13100, 30.0, 10.0, 0.0, 100.0), 500.0, None),
        (KeplerianElements(7000.0, 0.05, 30.0, 10.0, 0.0, 180.0), 6000.0, None),
        (KeplerianElements(-8000.0, 1.7, 40.0, 10.0, 0.0, 260.0), 20000.0, 1826.2378983),
        (KeplerianElements(-8000.0, 1.7, 40.0, 10.0, 0.0, 100.0), 20000.0, None),
    ],
    ids=["apogee", "descending", "ascending", "after-horizon", "above", "inbound", "outbound"],
)
def test_the_first_surface_crossing_matches_a_dense_bisection(
    elements: KeplerianElements, horizon_s: float, expected_s: float | None
) -> None:
    """Expected values: a 0.5 s scan of propagate_universal, bisected to 1e-12 s."""
    from orbitwin.orbit import first_radius_crossing_s

    state = keplerian_to_cartesian(elements, MU)
    got = first_radius_crossing_s(state, _R_EARTH_KM, horizon_s, MU)
    if expected_s is None:
        assert got is None
    else:
        assert got == pytest.approx(expected_s, abs=1.0e-5)
        assert float(np.linalg.norm(propagate_universal(state, got, MU).r)) == pytest.approx(
            _R_EARTH_KM, abs=1.0e-6
        )


@pytest.mark.parametrize("depth_m", [1.0, 0.01])
def test_a_graze_millimetres_deep_is_found(depth_m: float) -> None:
    """Rows 30 s either side of such a pericenter both sit hundreds of metres up."""
    from orbitwin.orbit import first_radius_crossing_s

    rp, ra = _R_EARTH_KM - depth_m / 1000.0, _R_EARTH_KM + 600.0
    state = keplerian_to_cartesian(
        KeplerianElements((rp + ra) / 2, (ra - rp) / (ra + rp), 30.0, 10.0, 0.0, 180.0), MU
    )
    got = first_radius_crossing_s(state, _R_EARTH_KM, 6000.0, MU)
    assert got is not None
    assert float(np.linalg.norm(propagate_universal(state, got, MU).r)) <= _R_EARTH_KM + 1.0e-8
    before = min(
        float(np.linalg.norm(propagate_universal(state, t, MU).r))
        for t in np.linspace(0.0, got - 1.0e-3, 4001)
    )
    assert before > _R_EARTH_KM


def test_a_start_inside_the_radius_is_at_zero() -> None:
    from orbitwin.orbit import first_radius_crossing_s

    state = keplerian_to_cartesian(KeplerianElements(6300.0, 0.0, 0.0, 0.0, 0.0, 0.0), MU)
    assert first_radius_crossing_s(state, _R_EARTH_KM, 100.0, MU) == 0.0
