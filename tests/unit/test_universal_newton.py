"""propagate_universal's Newton loop is scipy.optimize.newton's, to the bit.

orbit._newton_universal_anomaly replaced a call to newton(func, guess,
fprime=fprime, ...), which ran np.isclose and a second evaluation of the
Kepler function on every step, on np.float64 iterates. With it and
as_vector3's ndarray path, the tests' quick_config(local_refine=True) solve of
examples/two_burn_rendezvous.script took 6.59 s against 13.53 s (median of 5,
i5-14500, CPUs 12-19, one BLAS thread, 2026-11-24). These tests hold it to
newton itself where no iterate overflows: the same root to the bit, or an
exception of the same class. Where a cube overflows, Python floats raise
OverflowError and newton's np.float64 iterates raise another error; both fall
back to brentq (orbit._universal_anomaly).
"""

from __future__ import annotations

import math
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import newton

from orbitwin import orbit
from orbitwin.errors import DynamicsError
from orbitwin.models import StateVector

MU_KM3_S2 = 398600.4418


def _by_newton(guess, dt_s, r0mag, vr0, alpha, sqrt_mu, tolerance, max_iterations, **options):
    def func(chi):
        return orbit._universal_kepler_function(chi, dt_s, r0mag, vr0, alpha, sqrt_mu)[0]

    def fprime(chi):
        return orbit._universal_kepler_function(chi, dt_s, r0mag, vr0, alpha, sqrt_mu)[1]

    return newton(func, guess, fprime=fprime, tol=tolerance, maxiter=max_iterations, **options)


def _outcome(solver, *arguments):
    """The root's bits, or the class of the exception the solver raised; inf - inf warns."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = solver(*arguments)
    except Exception as error:  # the class is what is compared
        return "raised", type(error)
    return "returned", float(value).hex()


def _states(seed: int, count: int):
    """Elliptic, hyperbolic and near-parabolic states, each with forward and backward steps."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        r0mag = float(rng.uniform(6500.0, 60000.0))
        speed_ratio = float(rng.choice([rng.uniform(0.3, 1.38), rng.uniform(1.42, 3.0),
                                        math.sqrt(2.0) * (1.0 + float(rng.uniform(-1e-9, 1e-9)))]))
        speed = speed_ratio * math.sqrt(MU_KM3_S2 / r0mag)
        flight_path = float(rng.uniform(-1.4, 1.4))
        vr0 = speed * math.sin(flight_path)
        alpha = 2.0 / r0mag - speed * speed / MU_KM3_S2
        dt_s = float(rng.choice([-1.0, 1.0])) * float(10.0 ** rng.uniform(-3.0, 5.5))
        yield dt_s, r0mag, vr0, alpha


@pytest.mark.parametrize("seed", range(4))
def test_the_root_is_newtons_to_the_bit(seed: int) -> None:
    sqrt_mu = math.sqrt(MU_KM3_S2)
    compared = 0
    for dt_s, r0mag, vr0, alpha in _states(seed, 500):
        guess = orbit._initial_universal_anomaly(dt_s, r0mag, vr0, alpha, MU_KM3_S2)
        for tolerance, max_iterations in ((1.0e-11, 80), (1.0e-11, 2), (1.0e-14, 80), (1e-6, 1)):
            arguments = (guess, dt_s, r0mag, vr0, alpha, sqrt_mu, tolerance, max_iterations)
            assert (_outcome(orbit._newton_universal_anomaly, *arguments)
                    == _outcome(_by_newton, *arguments)), arguments
            compared += 1
    assert compared == 2000


F64 = np.float64

#: Kepler values and derivatives planted in the loop. The real function returns
#: floats; two rows plant np.float64, which newton itself iterates on.
PLANTED = {
    "root-at-guess": (0.0, 1.0), "zero-derivative": (1.0, 0.0), "both-zero": (0.0, 0.0),
    "nan-value": (math.nan, 1.0), "nan-derivative": (1.0, math.nan), "inf-value": (math.inf, 1.0),
    "inf-derivative": (1.0, math.inf), "minus-inf": (-math.inf, 2.0),
    "tiny-step": (1e-300, 1e300), "huge-step": (1e300, 1e-300),
    "f64-minus-zero-derivative": (F64(2.5), F64(-0.0)), "f64-small-step": (F64(3e-12), F64(1.0)),
}


@pytest.mark.parametrize("values", list(PLANTED.values()), ids=list(PLANTED))
@pytest.mark.parametrize("guess", [0.0, 3.5, -2.0, math.inf, math.nan])
def test_every_exit_is_newtons(monkeypatch: pytest.MonkeyPatch, values, guess: float) -> None:
    """The roots, the zero derivative, NaN and inf: each way out of the loop."""
    monkeypatch.setattr(orbit, "_universal_kepler_function", lambda *_: values)
    arguments = (guess, 100.0, 7000.0, 0.0, 1.0 / 7000.0, 1.0, 1.0e-11, 5)
    assert (_outcome(orbit._newton_universal_anomaly, *arguments)
            == _outcome(_by_newton, *arguments))


@pytest.mark.parametrize("error", [DynamicsError, ValueError, ZeroDivisionError, OverflowError],
                         ids=lambda error: error.__name__)
@pytest.mark.parametrize("on_call", [1, 2, 3])
def test_an_error_in_the_kepler_function_is_newtons(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception], on_call: int
) -> None:
    """None may become a root.

    DynamicsError, ValueError and OverflowError make it fall back to brentq;
    ZeroDivisionError escapes propagate_universal.

    newton evaluates the function and its derivative separately, so on_call
    counts this loop's evaluations and the stub for newton raises on the same step.
    """
    def raising_after(calls_per_step: int):
        calls = [0]

        def kepler(*_):
            calls[0] += 1
            if calls[0] > (on_call - 1) * calls_per_step:
                raise error("planted")
            return F64(1.0), F64(4.0)
        return kepler

    arguments = (3.5, 100.0, 7000.0, 0.0, 1.0 / 7000.0, 1.0, 1.0e-11, 5)
    monkeypatch.setattr(orbit, "_universal_kepler_function", raising_after(1))
    ours = _outcome(orbit._newton_universal_anomaly, *arguments)
    monkeypatch.setattr(orbit, "_universal_kepler_function", raising_after(2))
    assert ours == _outcome(_by_newton, *arguments) == ("raised", error)


@pytest.mark.parametrize("tolerance, max_iterations", [(0.0, 5), (-1.0, 5), (1e-11, 0), (1e-11, -3)])
def test_bad_arguments_are_refused_as_newton_refuses_them(
    tolerance: float, max_iterations: int
) -> None:
    arguments = (1.0, 100.0, 7000.0, 0.0, 1.0 / 7000.0, 1.0, tolerance, max_iterations)
    assert _outcome(orbit._newton_universal_anomaly, *arguments) == ("raised", ValueError)
    assert _outcome(_by_newton, *arguments) == ("raised", ValueError)


def test_each_step_evaluates_the_kepler_function_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two evaluations a step, as newton makes, passed every other test and cost a fifth of the gain.

    newton calls the function and its derivative once a step, and once more when the
    value is zero, so its steps are its calls less its iterations.
    """
    kepler = orbit._universal_kepler_function
    calls = [0]

    def counted(*arguments):
        calls[0] += 1
        return kepler(*arguments)

    monkeypatch.setattr(orbit, "_universal_kepler_function", counted)
    sqrt_mu = math.sqrt(MU_KM3_S2)
    steps = 0
    for dt_s, r0mag, vr0, alpha in _states(0, 50):
        guess = orbit._initial_universal_anomaly(dt_s, r0mag, vr0, alpha, MU_KM3_S2)
        arguments = (guess, dt_s, r0mag, vr0, alpha, sqrt_mu, 1.0e-11, 80)
        _, result = _by_newton(*arguments, full_output=True)
        calls[0] = 0
        orbit._newton_universal_anomaly(*arguments)
        assert calls[0] == result.function_calls - result.iterations, arguments
        steps += calls[0]
    assert steps > 100


R_KM = 8000.0
PARABOLIC = StateVector((R_KM, 0.0, 0.0), (0.0, math.sqrt(2.0 * MU_KM3_S2 / R_KM), 0.0))
ELLIPTIC = StateVector((R_KM, 0.0, 0.0), (0.0, 0.9 * math.sqrt(2.0 * MU_KM3_S2 / R_KM), 0.0))


@pytest.mark.parametrize("dt_s", [1e104, 1e200, -1e300])
def test_an_overflow_while_solving_is_a_dynamics_error(dt_s: float) -> None:
    """Callers catch DynamicsError; an OverflowError from the bracket's cube got past them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(DynamicsError):
            orbit.propagate_universal(PARABOLIC, dt_s, MU_KM3_S2)


@pytest.mark.parametrize("error", [OverflowError, ValueError], ids=lambda error: error.__name__)
def test_an_overflow_or_domain_error_in_every_solver_is_a_dynamics_error(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    def kepler(*_):
        raise error("planted")

    monkeypatch.setattr(orbit, "_universal_kepler_function", kepler)
    with pytest.raises(DynamicsError) as raised:
        orbit.propagate_universal(ELLIPTIC, 100.0, MU_KM3_S2)
    assert isinstance(raised.value.__cause__, error)


@pytest.mark.parametrize("error", [ZeroDivisionError, FloatingPointError, RuntimeError],
                         ids=lambda error: error.__name__)
def test_any_other_error_while_solving_escapes(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    """A defect, not an arc that cannot be flown: callers drop a DynamicsError's arc."""
    def kepler(*_):
        raise error("planted")

    monkeypatch.setattr(orbit, "_universal_kepler_function", kepler)
    with pytest.raises(error, match="planted"):
        orbit.propagate_universal(ELLIPTIC, 100.0, MU_KM3_S2)


@pytest.mark.parametrize("dt_s", [100.0, 0.0])
@pytest.mark.parametrize("options, error", [
    ({"tolerance": 0.0}, ValueError), ({"tolerance": -1e-11}, ValueError),
    ({"tolerance": math.nan}, ValueError), ({"tolerance": math.inf}, ValueError),
    ({"tolerance": "1e-11"}, TypeError), ({"tolerance": True}, TypeError),
    ({"max_iterations": 0}, ValueError), ({"max_iterations": -5}, ValueError),
    ({"max_iterations": 0.5}, TypeError), ({"max_iterations": 80.0}, TypeError),
    ({"max_iterations": True}, TypeError), ({"max_iterations": np.int64(80)}, TypeError),
], ids=["tol-0", "tol-negative", "tol-nan", "tol-inf", "tol-str", "tol-bool", "iter-0",
        "iter-negative", "iter-half", "iter-float", "iter-bool", "iter-int64"])
def test_bad_solver_arguments_are_refused_on_entry(
    options: dict, error: type[Exception], dt_s: float
) -> None:
    """Newton's own refusal sent them to brentq, which ignored max_iterations."""
    with pytest.raises(error, match=next(iter(options))):
        orbit.propagate_universal(ELLIPTIC, dt_s, MU_KM3_S2, **options)


def _zero_d(value, dtype=None) -> np.ndarray:
    array = np.array(value, dtype=dtype)
    assert array.ndim == 0
    return array


@pytest.mark.parametrize("dt_s", [
    "100", b"100", Decimal(100), None, [100.0], np.array([100.0]), 100.0 + 0j,
    True, np.bool_(True), _zero_d(True), np.timedelta64(100, "s"), np.timedelta64(100, "Y"),
    np.timedelta64(100, "ns"), np.timedelta64(100), _zero_d(np.timedelta64(100, "s")),
    np.datetime64("2026-11-24"), _zero_d(100.0 + 0j), _zero_d("100"),
], ids=["str", "bytes", "Decimal", "None", "list", "1-element", "complex", "bool", "bool_",
        "0-d-bool", "timedelta64-s", "timedelta64-Y", "timedelta64-ns", "timedelta64",
        "0-d-timedelta64", "datetime64", "0-d-complex", "0-d-str"])
def test_a_time_that_is_not_a_real_number_is_refused(dt_s) -> None:
    """float() took "100", b"100" and Decimal(100), which the scipy version refused, and
    took a timedelta64's count without its unit."""
    with pytest.raises(TypeError, match="dt_s must be a real number"):
        orbit.propagate_universal(ELLIPTIC, dt_s, MU_KM3_S2)


@pytest.mark.parametrize("dt_s", [
    100, np.int64(100), np.uint8(100), np.float32(100.0), np.float64(100.0), Fraction(201, 2),
    _zero_d(100.0), _zero_d(100), _zero_d(100.0, np.float32), _zero_d(Fraction(201, 2), object),
], ids=["int", "int64", "uint8", "float32", "float64", "Fraction", "0-d-float64", "0-d-int64",
        "0-d-float32", "0-d-object"])
def test_a_real_number_gives_what_its_float_gives(dt_s) -> None:
    ours = orbit.propagate_universal(ELLIPTIC, dt_s, MU_KM3_S2)
    expected = orbit.propagate_universal(ELLIPTIC, float(dt_s), MU_KM3_S2)
    assert [float(x).hex() for x in (*ours.r, *ours.v)] == [
        float(x).hex() for x in (*expected.r, *expected.v)
    ]
