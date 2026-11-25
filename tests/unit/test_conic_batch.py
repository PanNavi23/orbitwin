"""conic_batch against conic and propagate_universal, the screen's bound, and the JAX backend."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from orbitwin import conic, conic_batch
from orbitwin.models import StateVector
from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, propagate_universal

MU = 398600.4418
ELEMENTS = [
    KeplerianElements(7000.0, 0.01, 28.0, 15.0, 0.0, 300.0),
    KeplerianElements(9000.0, 0.3, 43.0, 65.0, 10.0, 170.0),
    KeplerianElements(7000.0 / (1.0 - 1.5), 1.5, 50.0, 75.0, 0.0, -30.0),
]
#: Backwards, under a second, and several revolutions of the ellipses either way.
TIMES = np.array([0.0, 0.5, 100.0, 1234.5, -800.0, 20000.0, -50000.0])
SPAN_S = 1800.0
KNOTS = np.arange(0.0, SPAN_S + 1.0, 60.0)
JAX_REASON = "jax is optional; install jax[cuda12]==0.6.2 to test the JAX backend"


def _states() -> tuple[np.ndarray, np.ndarray]:
    states = [keplerian_to_cartesian(elements, MU) for elements in ELEMENTS]
    return np.array([s.r for s in states]), np.array([s.v for s in states])


def _near_parabolic(generator: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """1 - e within 1e-6 of zero, from both sides."""
    ecc = 1.0 + generator.choice([-1.0, 1.0], n) * 10.0 ** generator.uniform(-10.0, -6.0, n)
    p = generator.uniform(6600.0, 15000.0, n) * (1.0 + ecc)
    nu = generator.uniform(-0.9, 0.9, n)
    tilt = generator.uniform(-math.pi, math.pi, n)
    radius = p / (1.0 + ecc * np.cos(nu))
    speed = np.sqrt(MU / p)
    r = np.c_[radius * np.cos(nu), radius * np.sin(nu) * np.cos(tilt),
              radius * np.sin(nu) * np.sin(tilt)]
    v = np.c_[-speed * np.sin(nu), speed * (ecc + np.cos(nu)) * np.cos(tilt),
              speed * (ecc + np.cos(nu)) * np.sin(tilt)]
    return r, v


def _encounters(
    generator: np.random.Generator, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pursuers on random LEO orbits, each passed by an evader at a random time,
    0.3 to 8 km away and 0.01 to 12 km/s faster; also those relative speeds."""
    rp, vp, re, ve = (np.empty((count, 3)) for _ in range(4))
    speeds = 10.0 ** generator.uniform(-2.0, math.log10(12.0), count)
    for i in range(count):
        radius = generator.uniform(6700.0, 8000.0)
        th, inc = generator.uniform(0.0, 2.0 * math.pi), generator.uniform(0.0, math.pi)
        rp[i] = radius * np.array([math.cos(th), math.sin(th) * math.cos(inc),
                                   math.sin(th) * math.sin(inc)])
        vp[i] = math.sqrt(MU / radius) * generator.uniform(0.97, 1.03) * np.array(
            [-math.sin(th), math.cos(th) * math.cos(inc), math.cos(th) * math.sin(inc)]
        )
        meet = generator.uniform(0.0, SPAN_S)
        r_meet, v_meet = conic.coast(rp[i], vp[i], meet, MU)
        offset, kick = generator.normal(size=3), generator.normal(size=3)
        offset *= generator.uniform(0.3, 8.0) / np.linalg.norm(offset)
        kick *= speeds[i] / np.linalg.norm(kick)
        re[i], ve[i] = conic.coast(r_meet + offset, v_meet + kick, -meet, MU)
    return rp, vp, re, ve, speeds


def _conjunctions(
    generator: np.random.Generator, count: int, span: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pursuers on random circular LEO orbits, each with an evader circular in the same
    plane, 30 to 600 km off radially, along track or across at a random time, kicked by
    up to 0.05 km/s: slow and wide, where the gravity-gradient bound is tight."""
    rp, vp, re, ve = (np.empty((count, 3)) for _ in range(4))
    for i in range(count):
        radius = generator.uniform(6700.0, 7500.0)
        th, inc = generator.uniform(0.0, 2.0 * math.pi), generator.uniform(0.0, math.pi)
        rp[i] = radius * np.array([math.cos(th), math.sin(th) * math.cos(inc),
                                   math.sin(th) * math.sin(inc)])
        vp[i] = math.sqrt(MU / radius) * np.array(
            [-math.sin(th), math.cos(th) * math.cos(inc), math.cos(th) * math.sin(inc)]
        )
        meet = generator.uniform(0.0, span)
        r_meet, v_meet = conic.coast(rp[i], vp[i], meet, MU)
        up = r_meet / np.linalg.norm(r_meet)
        normal = np.cross(r_meet, v_meet) / np.linalg.norm(np.cross(r_meet, v_meet))
        side = (up, np.cross(normal, up), normal)[int(generator.integers(0, 3))]
        r_e = r_meet + generator.choice([-1.0, 1.0]) * generator.uniform(30.0, 600.0) * side
        v_e = math.sqrt(MU / np.linalg.norm(r_e)) * np.cross(normal, r_e / np.linalg.norm(r_e))
        v_e += generator.normal(size=3) * generator.uniform(0.0, 0.05)
        re[i], ve[i] = conic.coast(r_e, v_e, -meet, MU)
    return rp, vp, re, ve


def _floors(r: np.ndarray, v: np.ndarray, span: float) -> np.ndarray:
    ends = [conic.coast(r[i], v[i], span, MU)[0] for i in range(len(r))]
    return np.array([conic.arc_minimum_radius(r[i], v[i], span, ends[i], MU)[0]
                     for i in range(len(r))])


def _true_minimum(
    rp: np.ndarray, vp: np.ndarray, re: np.ndarray, ve: np.ndarray,
    span: float = SPAN_S, *, paired: bool = False,
) -> np.ndarray:
    """min |r_p - r_e| over [0, span], every 0.05 s and then Brent around the best: for
    every pursuer against every evader, or with `paired` row i against row i only."""
    times = np.arange(0.0, span + 0.025, 0.05)

    def dense(r: np.ndarray, v: np.ndarray) -> np.ndarray:
        rows, _ = conic.coast_many(np.repeat(r, len(times), axis=0),
                                   np.repeat(v, len(times), axis=0), np.tile(times, len(r)), MU)
        return rows.reshape(len(r), len(times), 3)

    p, e = dense(rp, vp), dense(re, ve)
    pairs = [(i, i) for i in range(len(rp))] if paired else [
        (i, j) for i in range(len(rp)) for j in range(len(re))
    ]
    out = []
    for i, j in pairs:
        sampled = np.linalg.norm(p[i] - e[j], axis=1)
        k = int(np.argmin(sampled))

        def gap(s: float, i: int = i, j: int = j) -> float:
            return float(np.linalg.norm(conic.coast(rp[i], vp[i], s, MU)[0]
                                        - conic.coast(re[j], ve[j], s, MU)[0]))

        lo, hi = times[max(k - 1, 0)], times[min(k + 1, len(times) - 1)]
        found = minimize_scalar(gap, bounds=(lo, hi), method="bounded", options={"xatol": 1.0e-9})
        out.append(min(float(found.fun), float(sampled[k])))
    return np.array(out) if paired else np.array(out).reshape(len(rp), len(re))


def _screen(backend: Any, rp: np.ndarray, vp: np.ndarray, re: np.ndarray,
            ve: np.ndarray, knots: np.ndarray = KNOTS) -> conic_batch.Screen:
    p = conic_batch.propagate_batch(rp, vp, knots, MU)
    e = conic_batch.propagate_batch(re, ve, knots, MU)
    span = float(knots[-1])
    return backend.pairwise_screen(p.r_km, p.v_km_s, e.r_km, e.v_km_s, knots,
                                   _floors(rp, vp, span), _floors(re, ve, span), MU)


def test_propagate_batch_agrees_with_coast_many_and_propagate_universal() -> None:
    r0, v0 = _states()
    got = conic_batch.propagate_batch(r0, v0, TIMES, MU)
    assert got.converged.all()
    for i in range(len(r0)):
        rows_r, rows_v = conic.coast_many(
            np.tile(r0[i], (len(TIMES), 1)), np.tile(v0[i], (len(TIMES), 1)), TIMES, MU
        )
        assert np.max(np.linalg.norm(got.r_km[i] - rows_r, axis=1)) < 1.0e-9
        assert np.max(np.linalg.norm(got.v_km_s[i] - rows_v, axis=1)) < 1.0e-12
        for k, dt in enumerate(TIMES):
            want = propagate_universal(StateVector(r0[i], v0[i]), float(dt), MU)
            assert np.linalg.norm(got.r_km[i, k] - want.r) < 1.0e-8
            assert np.linalg.norm(got.v_km_s[i, k] - want.v) < 1.0e-11
    # Zero time is the start state exactly, as in conic.coast.
    assert np.array_equal(got.r_km[:, 0], r0) and np.array_equal(got.v_km_s[:, 0], v0)


def test_propagate_batch_converges_near_a_parabola_with_times_per_row() -> None:
    generator = np.random.default_rng(20261102)
    r, v = _near_parabolic(generator, 2000)
    dt = generator.uniform(-10000.0, 10000.0, (2000, 4))
    got = conic_batch.propagate_batch(r, v, dt, MU)
    assert got.converged.all()
    rows_r, rows_v = conic.coast_many(np.repeat(r, 4, axis=0), np.repeat(v, 4, axis=0),
                                      dt.ravel(), MU)
    assert np.max(np.linalg.norm(got.r_km.reshape(-1, 3) - rows_r, axis=1)) < 1.0e-8
    assert np.max(np.linalg.norm(got.v_km_s.reshape(-1, 3) - rows_v, axis=1)) < 1.0e-11
    for i in range(0, 2000, 100):
        want = propagate_universal(StateVector(r[i], v[i]), float(dt[i, 0]), MU)
        assert np.linalg.norm(got.r_km[i, 0] - want.r) < 1.0e-8


def test_a_row_that_fails_leaves_the_others_alone() -> None:
    r0, v0 = _states()
    r0 = np.vstack([r0[0], [np.nan, 0.0, 0.0], [0.0, 0.0, 0.0], r0[2]])
    v0 = np.vstack([v0[0], v0[1], v0[1], v0[2]])
    got = conic_batch.propagate_batch(r0, v0, TIMES, MU)
    assert got.converged.tolist() == [True, False, False, True]
    assert np.isnan(got.r_km[1:3]).all() and np.isnan(got.v_km_s[1:3]).all()
    alone = conic_batch.propagate_batch(r0[[0, 3]], v0[[0, 3]], TIMES, MU)
    assert np.array_equal(got.r_km[[0, 3]], alone.r_km)
    assert np.array_equal(got.v_km_s[[0, 3]], alone.v_km_s)


def test_a_row_out_of_iterations_is_marked_not_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """With one iteration only the zero-time row converges; the other is for conic.coast."""
    monkeypatch.setattr(conic_batch, "_MAX_ITERATIONS", 1)
    r0, v0 = _states()
    got = conic_batch.propagate_batch(r0[[1, 1]], v0[[1, 1]], [[0.0, 0.0], [600.0, 4000.0]], MU)
    assert got.converged.tolist() == [True, False]
    assert np.array_equal(got.r_km[0], [r0[1], r0[1]]) and np.isnan(got.r_km[1]).all()


def test_chunks_and_tiles_change_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    rp, vp, re, ve, _ = _encounters(np.random.default_rng(11), 5)
    whole_p = conic_batch.propagate_batch(rp, vp, KNOTS, MU)
    whole = _screen(conic_batch, rp, vp, re, ve)
    # One row per coast chunk; 2 x 2 pairs per screen tile, ragged at the edge.
    monkeypatch.setattr(conic_batch, "_NUMPY_ELEMENTS", 4 * len(KNOTS))
    assert np.array_equal(conic_batch.propagate_batch(rp, vp, KNOTS, MU).r_km, whole_p.r_km)
    tiled = _screen(conic_batch, rp, vp, re, ve)
    assert np.array_equal(tiled.lower_bound_km, whole.lower_bound_km)
    assert np.array_equal(tiled.knot_distance_km, whole.knot_distance_km)


def test_the_screen_never_bounds_above_the_true_minimum() -> None:
    rp, vp, re, ve, speeds = _encounters(np.random.default_rng(5), 6)
    screen = _screen(conic_batch, rp, vp, re, ve)
    truth = _true_minimum(rp, vp, re, ve)
    assert np.all(screen.lower_bound_km <= truth)
    assert np.all(screen.lower_bound_km <= screen.knot_distance_km)
    # And tight where it is used. A0 alone would leave 7.6 km at 60 s knots in LEO.
    slack = np.diag(truth - screen.lower_bound_km)
    assert np.all(slack[speeds < 0.5] < 0.1)
    assert np.all(slack < 2.5)


def test_the_screen_keeps_a_fast_pass_between_knots() -> None:
    """Planes 90 degrees apart, 10.8 km/s relative: 2 km apart at 90 s, 323 km at the knots."""
    meet = 90.0
    rp, vp = conic.coast(np.array([6878.0, 0.0, 0.0]),
                         np.array([0.0, math.sqrt(MU / 6878.0), 0.0]), -meet, MU)
    re, ve = conic.coast(np.array([6880.0, 0.0, 0.0]),
                         np.array([0.0, 0.0, math.sqrt(MU / 6880.0)]), -meet, MU)
    screen = _screen(conic_batch, rp[None], vp[None], re[None], ve[None])
    truth = _true_minimum(rp[None], vp[None], re[None], ve[None])
    assert truth[0, 0] == pytest.approx(2.0, abs=1.0e-6)
    assert screen.knot_distance_km[0, 0] > 300.0
    assert screen.lower_bound_km[0, 0] <= truth[0, 0]


def test_the_bound_holds_where_its_remainder_is_tight() -> None:
    """Wide, slow conjunctions at 60 s knots: LB comes within 0.1 km of the minimum
    there, and half the remainder would put it above on some."""
    knots = np.array([0.0, 60.0, 120.0])
    rp, vp, re, ve = _conjunctions(np.random.default_rng(0), 120, float(knots[-1]))
    screen = _screen(conic_batch, rp, vp, re, ve, knots)
    margin = _true_minimum(rp, vp, re, ve, float(knots[-1]), paired=True) - np.diag(
        screen.lower_bound_km
    )
    assert np.all(margin >= 0.0)
    assert np.min(margin) < 0.1


def test_a_pair_with_a_failed_row_is_never_cleared() -> None:
    rp, vp, re, ve, _ = _encounters(np.random.default_rng(3), 3)
    p = conic_batch.propagate_batch(rp, vp, KNOTS, MU)
    e = conic_batch.propagate_batch(re, ve, KNOTS, MU)
    p.r_km[1] = np.nan
    floors_p, floors_e = _floors(rp, vp, SPAN_S), _floors(re, ve, SPAN_S)
    floors_p[1] = np.nan
    screen = conic_batch.pairwise_screen(p.r_km, p.v_km_s, e.r_km, e.v_km_s, KNOTS,
                                         floors_p, floors_e, MU)
    assert np.all(screen.lower_bound_km[1] == -np.inf)
    assert np.all(np.isfinite(screen.lower_bound_km[[0, 2]]))


def test_the_screen_refuses_a_floor_above_a_knot() -> None:
    rp, vp, re, ve, _ = _encounters(np.random.default_rng(3), 2)
    p = conic_batch.propagate_batch(rp, vp, KNOTS, MU)
    e = conic_batch.propagate_batch(re, ve, KNOTS, MU)
    too_high = np.min(np.linalg.norm(p.r_km, axis=2), axis=1) + 1.0
    with pytest.raises(ValueError, match="above a row's own radius"):
        conic_batch.pairwise_screen(p.r_km, p.v_km_s, e.r_km, e.v_km_s, KNOTS,
                                    too_high, _floors(re, ve, SPAN_S), MU)


def test_select_refuses_an_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown backend 'cupy'"):
        conic_batch.select("cupy")


def test_importing_conic_batch_does_not_import_jax() -> None:
    code = "import sys, orbitwin.conic_batch; print('jax' in sys.modules)"
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert done.stdout.strip() == "False"


class _NumpyAsJax:
    """Stands in for _JaxKernels, so the start-up check and the fallback run without JAX."""

    name, note = "jax", "fake jax"

    def coast(self, *args: Any) -> tuple[np.ndarray, ...]:
        return conic_batch._NUMPY.coast(*args)

    def screen(self, *args: Any) -> tuple[np.ndarray, ...]:
        return conic_batch._NUMPY.screen(*args)


class _Float32(_NumpyAsJax):
    def coast(self, *args: Any) -> tuple[np.ndarray, ...]:
        r, v, ok = super().coast(*args)
        return r.astype(np.float32), v.astype(np.float32), ok

    def screen(self, *args: Any) -> tuple[np.ndarray, ...]:
        return tuple(a.astype(np.float32) for a in super().screen(*args))


class _Drifting(_NumpyAsJax):
    def coast(self, *args: Any) -> tuple[np.ndarray, ...]:
        r, v, ok = super().coast(*args)
        return r * (1.0 + 1.0e-6), v, ok


@pytest.mark.parametrize("kernels, reason", [(_Float32, "float32, not float64"),
                                             (_Drifting, "r differs from numpy")])
def test_jax_that_fails_the_start_up_check_is_refused(
    monkeypatch: pytest.MonkeyPatch, kernels: type, reason: str
) -> None:
    monkeypatch.setattr(conic_batch, "_load_jax", kernels)
    with pytest.warns(RuntimeWarning, match=reason):
        backend = conic_batch.select("jax")
    assert backend.name == "numpy" and "refused at start-up" in backend.note


def test_a_missing_jax_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "jax", None)
    with pytest.warns(RuntimeWarning, match=r"refused at start-up \(ModuleNotFoundError"):
        backend = conic_batch.select("jax")
    assert backend.name == "numpy"
    r0, v0 = _states()
    got = backend.propagate_batch(r0, v0, TIMES, MU)
    assert np.array_equal(got.r_km, conic_batch.propagate_batch(r0, v0, TIMES, MU).r_km)


class _LostAfterStartUp(_NumpyAsJax):
    calls = 0

    def coast(self, *args: Any) -> tuple[np.ndarray, ...]:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("device lost")
        return super().coast(*args)


def test_a_jax_failure_after_start_up_is_redone_on_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conic_batch, "_load_jax", _LostAfterStartUp)
    backend = conic_batch.select("jax")
    assert backend.name == "jax"
    r0, v0 = _states()
    with pytest.warns(RuntimeWarning, match="failed in propagate_batch.*device lost"):
        got = backend.propagate_batch(r0, v0, TIMES, MU)
    assert backend.name == "numpy" and "device lost" in backend.note
    assert np.array_equal(got.r_km, conic_batch.propagate_batch(r0, v0, TIMES, MU).r_km)


@pytest.fixture(scope="module")
def jax_backend() -> conic_batch.Backend:
    pytest.importorskip("jax", reason=JAX_REASON)
    backend = conic_batch.select("jax")
    assert backend.name == "jax", backend.note
    return backend


def test_jax_matches_numpy(
    jax_backend: conic_batch.Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Several coast blocks and 8 x 8 screen tiles, ragged at the edge.
    monkeypatch.setattr(conic_batch, "_JAX_ROWS", 128)
    monkeypatch.setattr(conic_batch, "_JAX_TILE_ELEMENTS", 64 * 64)
    generator = np.random.default_rng(20261103)
    r0, v0 = _states()
    near_r, near_v = _near_parabolic(generator, 300)
    r0 = np.vstack([r0, near_r, [[np.nan, 0.0, 0.0]]])
    v0 = np.vstack([v0, near_v, [[0.0, 7.5, 0.0]]])
    for times in (TIMES, generator.uniform(-10000.0, 10000.0, (len(r0), 5))):
        want = conic_batch.propagate_batch(r0, v0, times, MU)
        got = jax_backend.propagate_batch(r0, v0, times, MU)
        assert np.array_equal(got.converged, want.converged) and want.converged.sum() == len(r0) - 1
        ok = want.converged
        for a, b in ((got.r_km, want.r_km), (got.v_km_s, want.v_km_s)):
            relative = np.linalg.norm(a[ok] - b[ok], axis=-1) / np.linalg.norm(b[ok], axis=-1)
            assert np.max(relative) <= 1.0e-12
        assert np.isnan(got.r_km[~ok]).all()

    rp, vp, re, ve, _ = _encounters(generator, 20)
    want_s, got_s = _screen(conic_batch, rp, vp, re, ve), _screen(jax_backend, rp, vp, re, ve)
    for a, b in ((got_s.lower_bound_km, want_s.lower_bound_km),
                 (got_s.knot_distance_km, want_s.knot_distance_km)):
        assert np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0)) <= 1.0e-12

    # NaN survives the GPU's min and max: a failed row, one bad knot, an unknown floor.
    p = conic_batch.propagate_batch(rp[:3], vp[:3], KNOTS, MU)
    p.r_km[0], p.r_km[1, 5] = np.nan, np.nan
    floors = np.array([6000.0, 6000.0, np.nan])
    e = conic_batch.propagate_batch(re, ve, KNOTS, MU)
    got_nan = jax_backend.pairwise_screen(p.r_km, p.v_km_s, e.r_km, e.v_km_s, KNOTS, floors,
                                          _floors(re, ve, SPAN_S), MU)
    assert np.all(got_nan.lower_bound_km == -np.inf)
    assert jax_backend.name == "jax", jax_backend.note


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
def test_jax_without_float64_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    jax = pytest.importorskip("jax", reason=JAX_REASON)
    was = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    monkeypatch.setattr(jax.config, "update", lambda name, value: None)
    try:
        with pytest.warns(RuntimeWarning, match="float32, not float64"):
            backend = conic_batch.select("jax")
    finally:
        monkeypatch.undo()
        jax.config.update("jax_enable_x64", was)
    assert backend.name == "numpy"


def test_jax_compiles_once_per_knot_count_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracing is compiling; counted where the kernels are traced with jax.numpy."""
    pytest.importorskip("jax", reason=JAX_REASON)
    traced: list[str] = []

    def counted(name: str, kernel: Any) -> Any:
        def wrapper(xp: Any, *args: Any) -> Any:
            if xp is not np:
                traced.append(name)
            return kernel(xp, *args)
        return wrapper

    monkeypatch.setattr(conic_batch, "_coast_start", counted("coast", conic_batch._coast_start))
    monkeypatch.setattr(conic_batch, "_screen_tile", counted("screen", conic_batch._screen_tile))
    backend = conic_batch.select("jax")
    assert backend.name == "jax" and traced == ["coast", "screen"]

    rp, vp, re, ve, _ = _encounters(np.random.default_rng(2), 3)
    for rows, knots in ((3, 31), (1500, 61), (200, 64)):
        times = np.linspace(0.0, 1800.0, knots)
        p = backend.propagate_batch(np.resize(rp, (rows, 3)), np.resize(vp, (rows, 3)), times, MU)
        e = backend.propagate_batch(re, ve, times, MU)
        backend.propagate_batch(re, ve, np.tile(times, (3, 1)), MU)
        backend.pairwise_screen(p.r_km, p.v_km_s, e.r_km, e.v_km_s, times,
                                np.full(rows, 6000.0), np.full(3, 6000.0), MU)
    assert traced == ["coast", "screen"]
    times = np.linspace(0.0, 1800.0, 65)
    backend.propagate_batch(rp, vp, times, MU)
    assert traced == ["coast", "screen", "coast"]
