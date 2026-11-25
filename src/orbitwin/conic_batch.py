"""Two-body coasts over a grid of times, and a pairwise close-approach screen.

NumPy is the default and the only dependency. `select("jax")` adds JAX,
imported only then, with float64 enabled and checked against NumPy on a fixed
input. If JAX is missing, refuses or fails in any later call, the backend
warns, says why in `note`, and continues on NumPy. Both backends run the
same formulas in the same order.
"""

from __future__ import annotations

import importlib
import math
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, TypeVar

import numpy as np

from .conic import _LAGUERRE_ORDER
from .orbit import _STUMPFF_ARGUMENT_LIMIT, _STUMPFF_S_SERIES, _STUMPFF_S_SERIES_LIMIT

_MAX_ITERATIONS = 80
_STEP_TOLERANCE = 1.0e-13
#: Elements per temporary in a NumPy chunk, about 2 MB.
_NUMPY_ELEMENTS = 1 << 18
#: JAX compiles once per shape: rows per coast call, the multiple the knot
#: count is padded to, and pair-knots per screen tile (67 MB a temporary).
_JAX_ROWS = 1024
_JAX_KNOTS = 64
_JAX_TILE_ELEMENTS = 1 << 23
#: Start-up check tolerance, relative. Parity is ~1e-13; float32 is ~1e-7.
_CHECK_TOLERANCE = 1.0e-9

_T = TypeVar("_T")


class Coasts(NamedTuple):
    """States at every (row, time), and per row whether it converged.

    A row that did not converge is NaN throughout; redo it with `conic.coast`.
    """

    r_km: np.ndarray
    v_km_s: np.ndarray
    converged: np.ndarray


class Screen(NamedTuple):
    """Per (pursuer, evader) pair: a lower bound on the distance over the
    knots' span, -inf where it could not be computed, and the knot minimum."""

    lower_bound_km: np.ndarray
    knot_distance_km: np.ndarray


class _Arc(NamedTuple):
    x: Any
    y: Any
    z: Any
    vx: Any
    vy: Any
    vz: Any
    rmag0: Any
    alpha: Any
    a1: Any
    a2: Any
    sqrt_mu: Any
    dt: Any


def _stumpff(xp: Any, z: Any) -> tuple[Any, Any]:
    """`orbit.stumpff_c_array` and `stumpff_s_array` as selects, which JAX can trace."""
    positive = z > 0.0
    zp = xp.where(positive, z, 1.0)
    zn = xp.where(positive, 1.0, -z)
    overflow = xp.sqrt(zn) > _STUMPFF_ARGUMENT_LIMIT

    near_c = xp.abs(z) < 1.0e-8
    zc = xp.where(near_c, z, 0.0)
    series_c = 0.5 - zc / 24.0 + zc * zc / 720.0 - zc * zc * zc / 40320.0 + (
        zc * zc * zc * zc / 3628800.0
    )
    half = xp.sin(0.5 * xp.sqrt(zp))
    half_h = xp.sinh(0.5 * xp.sqrt(zn))
    c_neg = xp.where(overflow, xp.nan, 2.0 * half_h * half_h / zn)
    c = xp.where(near_c, series_c, xp.where(positive, 2.0 * half * half / zp, c_neg))

    near_s = xp.abs(z) < _STUMPFF_S_SERIES_LIMIT
    zs = xp.where(near_s, z, 0.0)
    series_s = xp.zeros_like(zs)
    for coefficient in reversed(_STUMPFF_S_SERIES):
        series_s = series_s * -zs + coefficient
    root = xp.sqrt(zp)
    root_h = xp.sqrt(zn)
    s_pos = (root - xp.sin(root)) / (root * root * root)
    s_neg = xp.where(overflow, xp.nan, (xp.sinh(root_h) - root_h) / (root_h * root_h * root_h))
    s = xp.where(near_s, series_s, xp.where(positive, s_pos, s_neg))
    return c, s


def _coast_start(xp: Any, r0: Any, v0: Any, t: Any, mu: Any) -> tuple[_Arc, Any, Any]:
    """`conic.coast_many`'s set-up and first guess; also which elements can iterate."""
    x, y, z = r0[:, 0:1], r0[:, 1:2], r0[:, 2:3]
    vx, vy, vz = v0[:, 0:1], v0[:, 1:2], v0[:, 2:3]
    rmag0 = xp.sqrt(x * x + y * y + z * z)
    vr0 = (x * vx + y * vy + z * vz) / rmag0
    alpha = 2.0 / rmag0 - (vx * vx + vy * vy + vz * vz) / mu
    sqrt_mu = xp.sqrt(mu)
    dt = t + xp.zeros_like(alpha)
    elliptic = alpha > 1.0e-12
    hyperbolic = alpha < -1.0e-12

    # Whole revolutions change nothing and only slow the iteration down.
    ae = xp.where(elliptic, alpha, 1.0)
    period = 2.0 * math.pi / (sqrt_mu * (ae * xp.sqrt(ae)))
    dt = xp.where(elliptic & (xp.abs(dt) > period), xp.fmod(dt, period), dt)

    ah = xp.where(hyperbolic, alpha, -1.0)
    sign = xp.where(dt >= 0.0, 1.0, -1.0)
    denominator = vr0 + sign * xp.sqrt(-mu / ah) * (1.0 - rmag0 * ah)
    usable = xp.abs(denominator) > 1.0e-12
    ratio = -2.0 * mu * ah * dt / xp.where(usable, denominator, 1.0)
    logarithm = xp.where(usable & (ratio > 0.0), xp.log(xp.where(ratio > 0.0, ratio, 1.0)), 1.0)
    chi = xp.where(
        elliptic,
        sqrt_mu * dt * alpha,
        xp.where(hyperbolic, sign * xp.sqrt(-1.0 / ah) * logarithm, sqrt_mu * dt / rmag0),
    )
    chi = xp.where(dt == 0.0, 0.0, chi)
    pending = (
        (rmag0 > 0.0) & xp.isfinite(rmag0) & xp.isfinite(alpha) & xp.isfinite(vr0)
        & xp.isfinite(chi)
    )
    arc = _Arc(
        x, y, z, vx, vy, vz, rmag0, alpha, rmag0 * vr0 / sqrt_mu, 1.0 - alpha * rmag0, sqrt_mu, dt
    )
    return arc, chi, pending


def _laguerre(xp: Any, arc: _Arc, chi: Any, pending: Any, converged: Any) -> tuple[Any, Any, Any]:
    """One Laguerre-Conway step; an element stops moving once its step is small."""
    zz = arc.alpha * chi * chi
    c, s = _stumpff(xp, zz)
    chi3 = chi * chi * chi
    f = arc.a1 * chi * chi * c + arc.a2 * chi3 * s + arc.rmag0 * chi - arc.sqrt_mu * arc.dt
    df = arc.a1 * chi * (1.0 - zz * s) + arc.a2 * chi * chi * c + arc.rmag0
    d2f = arc.a1 * (1.0 - zz * c) + arc.a2 * chi * (1.0 - zz * s)
    n = _LAGUERRE_ORDER
    root = xp.sqrt(xp.abs((n - 1) ** 2 * df * df - n * (n - 1) * f * d2f))
    step = n * f / (df + xp.copysign(root, df))
    moved = chi - step
    done = xp.abs(step) <= _STEP_TOLERANCE * xp.maximum(1.0, xp.abs(moved))
    return (
        xp.where(pending, moved, chi),
        pending & ~done & xp.isfinite(moved),
        converged | (pending & done),
    )


def _coast_end(xp: Any, arc: _Arc, chi: Any, converged: Any) -> tuple[Any, Any, Any]:
    zz = arc.alpha * chi * chi
    c, s = _stumpff(xp, zz)
    chi3 = chi * chi * chi
    f = 1.0 - chi * chi / arc.rmag0 * c
    g = arc.dt - chi3 / arc.sqrt_mu * s
    rx, ry, rz = f * arc.x + g * arc.vx, f * arc.y + g * arc.vy, f * arc.z + g * arc.vz
    rmag = xp.sqrt(rx * rx + ry * ry + rz * rz)
    fdot = arc.sqrt_mu / (rmag * arc.rmag0) * (arc.alpha * chi3 * s - chi)
    gdot = 1.0 - chi * chi / rmag * c
    r = xp.stack([rx, ry, rz], axis=-1)
    v = xp.stack([fdot * arc.x + gdot * arc.vx, fdot * arc.y + gdot * arc.vy,
                  fdot * arc.z + gdot * arc.vz], axis=-1)
    ok = converged & xp.all(xp.isfinite(r), axis=-1) & xp.all(xp.isfinite(v), axis=-1)
    return r, v, ok


def _nearest(xp: Any, p: tuple[Any, Any, Any], d: tuple[Any, Any, Any]) -> Any:
    """Distance from the origin to the segment p + s d, s in [0, 1]."""
    dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    moving = dd > 0.0
    toward = -(p[0] * d[0] + p[1] * d[1] + p[2] * d[2]) / xp.where(moving, dd, 1.0)
    s = xp.where(moving, xp.clip(toward, 0.0, 1.0), 0.0)
    qx, qy, qz = p[0] + s * d[0], p[1] + s * d[1], p[2] + s * d[2]
    return xp.sqrt(qx * qx + qy * qy + qz * qz)


def _screen_tile(
    xp: Any, rp: Any, vp: Any, re: Any, ve: Any, h: Any, floor_p: Any, floor_e: Any, mu: Any
) -> tuple[Any, Any]:
    """`pairwise_screen` on one tile: rp, vp are (a, K, 3), re, ve (b, K, 3), h (K-1,)."""
    rho = [rp[:, None, :, i] - re[None, :, :, i] for i in range(3)]
    w = [vp[:, None, :, i] - ve[None, :, :, i] for i in range(3)]
    dist = xp.sqrt(rho[0] * rho[0] + rho[1] * rho[1] + rho[2] * rho[2])
    rho0 = (rho[0][..., :-1], rho[1][..., :-1], rho[2][..., :-1])
    rho1 = (rho[0][..., 1:], rho[1][..., 1:], rho[2][..., 1:])
    w0 = (w[0][..., :-1], w[1][..., :-1], w[2][..., :-1])
    w1 = (w[0][..., 1:], w[1][..., 1:], w[2][..., 1:])

    crude = (mu / (floor_p * floor_p))[:, None, None] + (mu / (floor_e * floor_e))[None, :, None]
    low = xp.minimum(floor_p[:, None, None], floor_e[None, :, None])
    speed = xp.sqrt(w0[0] * w0[0] + w0[1] * w0[1] + w0[2] * w0[2])
    reach = dist[..., :-1] + speed * h + 0.5 * crude * h * h
    room = low - 0.5 * reach
    clear = xp.where(room > 0.0, room, 1.0)
    tidal = 2.0 * mu * reach / (clear * clear * clear)
    accel = xp.where(room > 0.0, xp.minimum(crude, tidal), crude)

    half = 0.5 * h
    taylor = xp.minimum(
        _nearest(xp, rho0, (w0[0] * half, w0[1] * half, w0[2] * half)),
        _nearest(xp, rho1, (-w1[0] * half, -w1[1] * half, -w1[2] * half)),
    )
    secant = _nearest(xp, rho0, (rho1[0] - rho0[0], rho1[1] - rho0[1], rho1[2] - rho0[2]))
    bound = xp.maximum(taylor, secant) - 0.5 * accel * half * half
    return xp.min(bound, axis=2), xp.min(dist, axis=2)


class _Kernels(Protocol):
    name: str
    note: str

    def coast(
        self, r0: np.ndarray, v0: np.ndarray, t: np.ndarray, mu: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def screen(
        self, rp: np.ndarray, vp: np.ndarray, re: np.ndarray, ve: np.ndarray, h: np.ndarray,
        floor_p: np.ndarray, floor_e: np.ndarray, mu: float,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class _NumpyKernels:
    name = note = "numpy"

    def coast(
        self, r0: np.ndarray, v0: np.ndarray, t: np.ndarray, mu: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, k = len(r0), t.shape[1]
        r, v = np.empty((n, k, 3)), np.empty((n, k, 3))
        ok = np.empty((n, k), dtype=bool)
        rows = max(1, _NUMPY_ELEMENTS // k)
        with np.errstate(all="ignore"):
            for i in range(0, n, rows):
                part = slice(i, i + rows)
                times = t[part] if len(t) > 1 else t
                arc, chi, pending = _coast_start(np, r0[part], v0[part], times, mu)
                converged = np.zeros_like(pending)
                for _ in range(_MAX_ITERATIONS):
                    if not pending.any():
                        break
                    chi, pending, converged = _laguerre(np, arc, chi, pending, converged)
                r[part], v[part], ok[part] = _coast_end(np, arc, chi, converged)
        return r, v, ok

    def screen(
        self, rp: np.ndarray, vp: np.ndarray, re: np.ndarray, ve: np.ndarray, h: np.ndarray,
        floor_p: np.ndarray, floor_e: np.ndarray, mu: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        m, n = len(rp), len(re)
        lb, d = np.empty((m, n)), np.empty((m, n))
        side = max(1, math.isqrt(_NUMPY_ELEMENTS // rp.shape[1]))
        with np.errstate(all="ignore"):
            for i in range(0, m, side):
                p = slice(i, i + side)
                for j in range(0, n, side):
                    e = slice(j, j + side)
                    lb[p, e], d[p, e] = _screen_tile(
                        np, rp[p], vp[p], re[e], ve[e], h, floor_p[p], floor_e[e], mu
                    )
        return lb, d


def _round_up(count: int, multiple: int) -> int:
    return -(-count // multiple) * multiple


def _pad(array: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Repeat the last entry along `axis` up to `size`."""
    widths = [(0, 0)] * array.ndim
    widths[axis] = (0, size - array.shape[axis])
    return np.pad(array, widths, mode="edge")


class _JaxKernels:
    """The NumPy kernels jitted on fixed, padded shapes; host arrays in and out."""

    name = "jax"

    def __init__(self) -> None:
        # Imported here so that installs without JAX never need it; via importlib
        # so that mypy types it as Any whether or not it is installed.
        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", True)
        self._jnp = jnp = importlib.import_module("jax.numpy")
        lax = importlib.import_module("jax.lax")
        self.note = f"jax {jax.__version__} on {jax.devices()[0]}"

        def coast(r0: Any, v0: Any, t: Any, mu: Any) -> tuple[Any, Any, Any]:
            arc, chi, pending = _coast_start(jnp, r0, v0, t, mu)

            def more(state: tuple[Any, ...]) -> Any:
                return (state[0] < _MAX_ITERATIONS) & jnp.any(state[2])

            def step(state: tuple[Any, ...]) -> tuple[Any, ...]:
                return (state[0] + 1, *_laguerre(jnp, arc, *state[1:]))

            start = (0, chi, pending, jnp.zeros_like(pending))
            _, chi, _, converged = lax.while_loop(more, step, start)
            return _coast_end(jnp, arc, chi, converged)

        def screen(*args: Any) -> tuple[Any, Any]:
            return _screen_tile(jnp, *args)

        self._coast = jax.jit(coast)
        self._screen = jax.jit(screen)

    def coast(
        self, r0: np.ndarray, v0: np.ndarray, t: np.ndarray, mu: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, k = len(r0), t.shape[1]
        rows, knots = _round_up(n, _JAX_ROWS), _round_up(k, _JAX_KNOTS)
        r0, v0 = _pad(r0, rows, 0), _pad(v0, rows, 0)
        t = _pad(_pad(np.broadcast_to(t, (n, k)), knots, 1), rows, 0)
        mu_d = self._jnp.asarray(mu, dtype=self._jnp.float64)
        parts = [
            self._coast(r0[i : i + _JAX_ROWS], v0[i : i + _JAX_ROWS], t[i : i + _JAX_ROWS], mu_d)
            for i in range(0, rows, _JAX_ROWS)
        ]
        r, v, ok = (np.concatenate([np.asarray(part[q]) for part in parts]) for q in range(3))
        return r[:n, :k], v[:n, :k], ok[:n, :k]

    def screen(
        self, rp: np.ndarray, vp: np.ndarray, re: np.ndarray, ve: np.ndarray, h: np.ndarray,
        floor_p: np.ndarray, floor_e: np.ndarray, mu: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        jnp = self._jnp
        m, n, k = len(rp), len(re), rp.shape[1]
        knots = _round_up(k, _JAX_KNOTS)
        tile = 1 << max(3, int(math.log2(_JAX_TILE_ELEMENTS // knots)) // 2)
        # A repeated last knot adds zero-length segments, bounded by that knot's distance.
        hd = jnp.asarray(np.concatenate([h, np.zeros(knots - k)]))
        mu_d = jnp.asarray(mu, dtype=jnp.float64)

        def blocks(*arrays: np.ndarray) -> list[list[Any]]:
            rows = _round_up(len(arrays[0]), tile)
            padded = [_pad(a if a.ndim == 1 else _pad(a, knots, 1), rows, 0) for a in arrays]
            return [[jnp.asarray(a[i : i + tile]) for a in padded] for i in range(0, rows, tile)]

        pursuers, evaders = blocks(rp, vp, floor_p), blocks(re, ve, floor_e)
        tiles = [
            [self._screen(p[0], p[1], e[0], e[1], hd, p[2], e[2], mu_d) for e in evaders]
            for p in pursuers
        ]
        lb, d = (
            np.asarray(jnp.concatenate([jnp.concatenate([t[q] for t in row], axis=1)
                                        for row in tiles]))
            for q in range(2)
        )
        return lb[:m, :n], d[:m, :n]


_NUMPY = _NumpyKernels()


def _float64(*arrays: np.ndarray) -> None:
    for array in arrays:
        if array.dtype != np.float64:
            raise TypeError(f"an output came back as {array.dtype}, not float64")


def _coast_inputs(
    r0_km: Any, v0_km_s: Any, t_s: Any, mu: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    r0 = np.asarray(r0_km, dtype=float)
    v0 = np.asarray(v0_km_s, dtype=float)
    t = np.asarray(t_s, dtype=float)
    if r0.ndim != 2 or r0.shape[1] != 3 or v0.shape != r0.shape:
        raise ValueError(f"r0 and v0 must both be (N, 3), got {r0.shape} and {v0.shape}")
    if t.ndim == 1:
        t = t[None, :]
    if t.ndim != 2 or t.shape[1] == 0 or t.shape[0] not in {1, len(r0)}:
        raise ValueError(f"times must be (K,) or (N, K) with N = {len(r0)}, got {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("times must be finite")
    if not (math.isfinite(mu) and mu > 0.0):
        raise ValueError(f"mu must be finite and positive, got {mu!r}")
    return r0, v0, t, float(mu)


def _screen_inputs(
    rp_km: Any, vp_km_s: Any, re_km: Any, ve_km_s: Any, t_s: Any,
    rp_min_km: Any, re_min_km: Any, mu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """rp, vp, re, ve, the segment lengths and the two floors, checked."""
    t = np.asarray(t_s, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        raise ValueError("knot times must be finite, strictly increasing, and at least two")
    if not (math.isfinite(mu) and mu > 0.0):
        raise ValueError(f"mu must be finite and positive, got {mu!r}")
    out: list[np.ndarray] = []
    for side, r_in, v_in, floor_in in (("p", rp_km, vp_km_s, rp_min_km),
                                       ("e", re_km, ve_km_s, re_min_km)):
        r, v = np.asarray(r_in, dtype=float), np.asarray(v_in, dtype=float)
        floor = np.asarray(floor_in, dtype=float)
        if r.ndim != 3 or r.shape[1:] != (len(t), 3) or v.shape != r.shape:
            raise ValueError(f"r{side} and v{side} must be (rows, {len(t)}, 3), "
                             f"got {r.shape} and {v.shape}")
        if floor.shape != (len(r),):
            raise ValueError(f"r{side}_min_km must be ({len(r)},), got {floor.shape}")
        if np.any(floor <= 0.0):
            raise ValueError(f"r{side}_min_km must be positive")
        # A floor above a knot cannot bound the radius between knots. NaN rows pass, as -inf.
        if np.any(floor > np.min(np.sqrt(np.sum(r * r, axis=2)), axis=1) * (1.0 + 1.0e-12)):
            raise ValueError(f"r{side}_min_km is above a row's own radius at a knot")
        out += [r, v, floor]
    rp, vp, floor_p, re, ve, floor_e = out
    return rp, vp, re, ve, np.diff(t), floor_p, floor_e


def _fell_back(stage: str, error: Exception) -> str:
    note = f"jax {stage} ({type(error).__name__}: {error}); using numpy"
    warnings.warn(note, RuntimeWarning, stacklevel=3)
    return note


class Backend:
    """`propagate_batch` and `pairwise_screen` on NumPy or JAX; `select` makes one.

    A JAX failure in any call is redone on NumPy, and NumPy is used from then on.
    """

    def __init__(self, kernels: _Kernels, note: str) -> None:
        self._kernels = kernels
        self.note = note

    @property
    def name(self) -> str:
        return self._kernels.name

    def propagate_batch(self, r0_km: Any, v0_km_s: Any, t_s: Any, mu: float) -> Coasts:
        """See the module-level `propagate_batch`."""
        r0, v0, t, mu = _coast_inputs(r0_km, v0_km_s, t_s, mu)

        def run(kernels: _Kernels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            r, v, ok = kernels.coast(r0, v0, t, mu)
            _float64(r, v)
            return r, v, ok

        r, v, ok = self._run("propagate_batch", run, empty=not len(r0))
        converged = np.all(ok, axis=1)
        good = converged[:, None, None]
        return Coasts(np.where(good, r, np.nan), np.where(good, v, np.nan), converged)

    def pairwise_screen(
        self, rp_km: Any, vp_km_s: Any, re_km: Any, ve_km_s: Any, t_s: Any,
        rp_min_km: Any, re_min_km: Any, mu: float,
    ) -> Screen:
        """See the module-level `pairwise_screen`."""
        args = _screen_inputs(rp_km, vp_km_s, re_km, ve_km_s, t_s, rp_min_km, re_min_km, mu)

        def run(kernels: _Kernels) -> tuple[np.ndarray, np.ndarray]:
            lb, d = kernels.screen(*args, float(mu))
            _float64(lb, d)
            return lb, d

        lb, d = self._run("pairwise_screen", run, empty=not (len(args[0]) and len(args[2])))
        return Screen(np.where(np.isfinite(lb), lb, -np.inf), d)

    def _run(self, call: str, work: Callable[[_Kernels], _T], *, empty: bool) -> _T:
        if self._kernels is not _NUMPY and not empty:
            try:
                return work(self._kernels)
            except Exception as error:
                self._kernels = _NUMPY
                self.note = _fell_back(f"failed in {call}", error)
        return work(_NUMPY)


def _load_jax() -> _Kernels:
    return _JaxKernels()


def _check_input() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Four states from one periapsis, 7000 km: circular, the same 2 degrees off
    and 1 km out, hyperbolic, and within 1e-7 of a parabola."""
    mu = 398600.4418
    r0 = np.array([[7000.0, 0.0, 0.0], [7001.0, 0.0, 0.0], [7000.0, 0.0, 0.0], [7000.0, 0.0, 0.0]])
    v0 = np.empty((4, 3))
    shapes = ((0.0, 0.0), (0.0, 2.0), (1.5, 30.0), (1.0 - 1e-7, 60.0))
    for row, (ecc, tilt_deg) in enumerate(shapes):
        tilt = math.radians(tilt_deg)
        speed = math.sqrt(mu * (1.0 + ecc) / r0[row, 0])
        v0[row] = [0.0, speed * math.cos(tilt), speed * math.sin(tilt)]
    return r0, v0, np.array([0.0, 60.0, 120.0, 900.0, 7200.0]), mu


def _check(kernels: _Kernels) -> None:
    """Refuse kernels that, on the fixed input, are not float64 or not NumPy's."""
    r0, v0, t, mu = _check_input()
    r, v, ok = _NUMPY.coast(r0, v0, t[None, :], mu)
    floors = 0.9 * np.min(np.linalg.norm(r, axis=2), axis=1)
    screen = (r[::2], v[::2], r[1::2], v[1::2], np.diff(t), floors[::2], floors[1::2], mu)
    want = (r, v, *_NUMPY.screen(*screen))
    got_r, got_v, got_ok = kernels.coast(r0, v0, t[None, :], mu)
    got = (got_r, got_v, *kernels.screen(*screen))
    _float64(*got)
    if not np.array_equal(got_ok, ok):
        raise ValueError("convergence differs from numpy on the start-up input")
    for name, a, b in zip(("r", "v", "lb", "d_knot"), got, want, strict=True):
        error = float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1.0))
        if not error <= _CHECK_TOLERANCE:
            raise ValueError(f"{name} differs from numpy by {error:.3g} on the start-up input")


def select(requested: str = "numpy") -> Backend:
    """The backend named, "numpy" or "jax"; JAX falls back to NumPy, with a warning.

    "jax" turns on jax_enable_x64 for the whole process and compiles its kernels.
    """
    if requested == "numpy":
        return Backend(_NUMPY, _NUMPY.note)
    if requested != "jax":
        raise ValueError(f"unknown backend {requested!r}; expected 'numpy' or 'jax'")
    try:
        kernels = _load_jax()
        _check(kernels)
    except Exception as error:
        return Backend(_NUMPY, _fell_back("refused at start-up", error))
    return Backend(kernels, kernels.note)


_DEFAULT = Backend(_NUMPY, _NUMPY.note)


def propagate_batch(r0_km: Any, v0_km_s: Any, t_s: Any, mu: float) -> Coasts:
    """Coast every row of r0, v0 (N, 3) to each time in t (K,) or (N, K); NumPy.

    The same universal-variable Laguerre-Conway solve as `conic.coast_many`,
    but a row that does not converge is marked and NaN instead of raising for
    the whole batch.
    """
    return _DEFAULT.propagate_batch(r0_km, v0_km_s, t_s, mu)


def pairwise_screen(
    rp_km: Any, vp_km_s: Any, re_km: Any, ve_km_s: Any, t_s: Any,
    rp_min_km: Any, re_min_km: Any, mu: float,
) -> Screen:
    """A lower bound on min |r_p - r_e| over the knots t (K,) for every pair; NumPy.

    rp, vp are (m, K, 3) and re, ve (n, K, 3), states at the knots;
    rp_min_km (m,) and re_min_km (n,) must bound each row's radius from below
    over the whole span, as `conic.arc_minimum_radius` does. The lowest knot is
    not such a bound: a periapsis between knots is lower, and only a floor above
    a knot is refused. Per segment of
    length h, with rho = r_p - r_e and w = rho':

        A0 = mu / rp_min^2 + mu / re_min^2, both gravities at their largest
        A1 = 2 mu rho_hi / (r_lo - rho_hi / 2)^3, the gravity gradient across
             the chord, if r_lo = min(rp_min, re_min) > rho_hi / 2, where
             rho_hi = |rho0| + |w0| h + A0 h^2 / 2 bounds |rho| on the segment
        LB = max(Taylor, secant) - min(A0, A1) h^2 / 8, where
             Taylor = min(min |rho0 + w0 tau|, min |rho1 - w1 tau|), tau in [0, h/2]
             secant = min |rho| on the chord rho0 -> rho1

    LB above a radius proves the pair outside it over the span; nothing else
    follows from it, and it never shows a capture. A pair that touches a
    non-converged row is -inf.
    """
    return _DEFAULT.pairwise_screen(rp_km, vp_km_s, re_km, ve_km_s, t_s, rp_min_km, re_min_km, mu)
