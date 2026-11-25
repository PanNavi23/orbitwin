"""Facts about one round's coasts: captures between knots, radius crossings.

Nothing here decides who wins; `referee.py` does. Every radius is passed in.

The capture detector bounds the relative distance on each knot segment from
both sides (DUEL_ENGINE.md section 3.2). With rho = r_P - r_E, rho'' =
g(r_P) - g(r_E) is bounded by A = min(A0, A1): A0 = mu/rP_min^2 + mu/rE_min^2,
and the gravity-gradient bound A1 = 2 mu rho_hi / (r_lo - rho_hi/2)^3. On a
segment of length h

    LB = max(Taylor, secant) - A h^2/8,  UB = max over half-segment ends + A h^2/8

where Taylor uses the tangent lines from both ends over half the segment and
the secant the chord; both have the same remainder A h^2/8. A piece is proved outside
when LB exceeds the radius, proved inside when UB is within it, and is
otherwise halved, on exact coasts, down to 1e-6 s.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ..conic import coast, coast_many, periapsis_radius_km, seconds_to_periapsis
from ..errors import DynamicsError
from ..models import StateVector
from ..orbit import propagate_universal

_LOG = logging.getLogger(__name__)

#: Pieces narrower than this are left undecided.
MIN_PIECE_S = 1.0e-6
#: Exact coasts allowed per detector call before the rest is left undecided.
#: A grazing pass resolves in a few hundred; a pair that holds exactly at the
#: radius for a whole round would otherwise cost about 45 us per evaluation.
MAX_EVALUATIONS = 20_000

Radius = Callable[[float], float]

#: Below this e an arc is treated as a circle (the core's own cut).
CIRCULAR_E = 1.0e-10
#: Within this of e = 1 the periapsis is found from r.v, not the closed form.
NEAR_PARABOLIC = 1.0e-6


#: What a coast that cannot be flown raises: conic.coast overflows past |t| ~ 6e103 s
#: and takes no infinite t on an ellipse.
_COAST_FAILURES = (DynamicsError, OverflowError, ValueError)


def coast_one(r: np.ndarray, v: np.ndarray, t: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """`conic.coast`, falling back to `orbit.propagate_universal` where it fails.

    `coast` can miss on a far outbound hyperbola at short times (its starting
    guess, measured at r = 738595 km, e = 1.28, t = 180 s); the slower
    Newton-Raphson and brentq solver of the same equations converges there.
    """
    try:
        return coast(r, v, t, mu)
    except _COAST_FAILURES:
        start = StateVector((float(r[0]), float(r[1]), float(r[2])),
                            (float(v[0]), float(v[1]), float(v[2])))
        state = propagate_universal(start, t, mu)
        return state.r, state.v


def coast_rows(
    r0: np.ndarray, v0: np.ndarray, times: np.ndarray, mu: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`coast_many`, re-run row by row when the batch fails; returns (r, v, ok).

    `coast_many` raises for the whole batch when one row fails (section 2.3).
    Rows that fail on both solvers are NaN with `ok` false, and logged.
    """
    count = len(times)
    try:
        r, v = coast_many(r0, v0, times, mu)
        return r, v, np.ones(count, dtype=bool)
    except _COAST_FAILURES:
        pass
    r = np.full((count, 3), np.nan)
    v = np.full((count, 3), np.nan)
    ok = np.zeros(count, dtype=bool)
    for k in range(count):
        try:
            r[k], v[k] = coast_one(r0[k], v0[k], float(times[k]), mu)
            ok[k] = True
        except _COAST_FAILURES as error:
            _LOG.warning("coast row %d failed on both solvers: %s", k, error)
    return r, v, ok


#: |r.v| / (|r| |v|) the batch and scalar coasts may round apart; measured at most
#: 3.5e-16 relative on 400 near-parabolic arcs.
ROUNDING_RADIAL = 1.0e-12


class UnusableArc(DynamicsError):
    """An arc that neither coast solver can fly; the action it came from is unusable."""


@dataclass(frozen=True, slots=True)
class Arc:
    """One craft's coast after its impulse, from t = 0 to `duration_s`."""

    r0: np.ndarray
    v0: np.ndarray
    duration_s: float
    mu: float
    #: Results that depend only on the arc, filled on first use. Not an init
    #: field, so dataclasses.replace starts a new arc with an empty one.
    _cache: dict[str, tuple[list[float], list[float]]] = field(
        default_factory=dict, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        # Own read-only copies, so a caller changing its arrays cannot make
        # the cached results stale.
        for name in ("r0", "v0"):
            array = np.array(getattr(self, name), dtype=float)
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        try:
            return coast_one(self.r0, self.v0, t, self.mu)
        except _COAST_FAILURES as error:
            raise UnusableArc(f"no solver flies this arc to t = {t} s: {error}") from error

    def states(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = len(times)
        r, v, ok = coast_rows(
            np.tile(self.r0, (count, 1)), np.tile(self.v0, (count, 1)), times, self.mu
        )
        if not ok.all():
            raise UnusableArc(f"no solver flies this arc to t = {times[~ok].tolist()} s")
        return r, v

    def radius(self, t: float) -> float:
        return float(np.linalg.norm(self.state(t)[0]))

    def minimum_radius(self) -> float:
        """Smallest radius over the arc, or a lower bound on it for a near-circle."""
        ends = min(self.radius(0.0), self.radius(self.duration_s))
        e = self.eccentricity()
        if e < CIRCULAR_E:
            # No turning times are found; the radius stays within 2ae of an end.
            alpha = 2.0 / float(np.linalg.norm(self.r0)) - float(self.v0 @ self.v0) / self.mu
            return ends - 2.0 * e / alpha
        if self.periapsis_times():
            return min(ends, periapsis_radius_km(self.r0, self.v0, self.mu))
        return ends

    def period(self) -> float | None:
        alpha = 2.0 / float(np.linalg.norm(self.r0)) - float(np.dot(self.v0, self.v0)) / self.mu
        if alpha <= 0.0:
            return None
        return 2.0 * math.pi / math.sqrt(self.mu * alpha**3)

    def eccentricity(self) -> float:
        r, v, mu = self.r0, self.v0, self.mu
        radius = float(np.linalg.norm(r))
        e_vec = ((float(np.dot(v, v)) - mu / radius) * r - float(np.dot(r, v)) * v) / mu
        return float(np.linalg.norm(e_vec))

    def _radial(self, t: float) -> float:
        r, v = self.state(t)
        return float(r @ v)

    def _turning(self) -> tuple[list[float], list[float]]:
        """(periapsis times, apoapsis times) inside (0, duration), computed once."""
        cached = self._cache.get("turning")
        if cached is None:
            cached = self._cache["turning"] = self._find_turning()
        return list(cached[0]), list(cached[1])

    def _find_turning(self) -> tuple[list[float], list[float]]:
        periapses: list[float] = []
        apoapses: list[float] = []
        h = float(np.linalg.norm(np.cross(self.r0, self.v0)))
        if h <= 1.0e-9 * float(np.linalg.norm(self.r0)) * float(np.linalg.norm(self.v0)):
            # A radial arc has no conic shape the closed forms can use, and
            # it may pass through the centre: refused, not guessed.
            raise UnusableArc("a radial arc (zero angular momentum) cannot be judged")
        e = self.eccentricity()
        if e < CIRCULAR_E:
            # Circular: the radius does not change, so nothing turns.
            return periapses, apoapses
        period = self.period()
        duration = self.duration_s
        if abs(1.0 - e) < NEAR_PARABOLIC:
            # The core's closed-form periapsis time is off by 0.13 s at
            # 1 - e = 1e-10 and by hours nearer 1. Here the turning points
            # are where r.v changes sign: - to + at a periapsis, + to - at an
            # apoapsis. They are half a period apart, so a grid of at most
            # P/8 holds at most one per interval.
            step = 60.0 if period is None else min(60.0, period / 8.0)
            grid = np.linspace(0.0, duration, max(2, math.ceil(duration / step) + 1))
            r, v = self.states(grid)
            values = np.einsum("ij,ij->i", r, v)
            # A zero counts as positive, so an exact zero on the grid is found
            # once, by the interval that ends (- to 0) or starts (0 to -) there.
            for k in range(len(grid) - 1):
                f0, f1 = float(values[k]), float(values[k + 1])
                if (f0 < 0.0) == (f1 < 0.0):
                    continue
                # The batch and the scalar coast round apart near a zero: on CI a
                # 1e-12 km/s fall from rest read r.v(0) as 0 on one and -4e-25 on
                # the other. Then the turning point is the end within rounding of
                # zero; anything larger is a solver fault, refused.
                a, b = float(grid[k]), float(grid[k + 1])
                g0, g1 = self._radial(a), self._radial(b)
                if (g0 < 0.0) == (g1 < 0.0):
                    root = a if abs(g0) <= abs(g1) else b
                    r_end, v_end = self.state(root)
                    scale = float(np.linalg.norm(r_end)) * float(np.linalg.norm(v_end))
                    if abs(self._radial(root)) > ROUNDING_RADIAL * scale:
                        raise UnusableArc(
                            f"the batch and scalar coasts disagree on r.v over "
                            f"[{a:g}, {b:g}] s by more than rounding"
                        )
                else:
                    root = float(brentq(self._radial, a, b, xtol=1.0e-9, rtol=1.0e-15))
                if 0.0 < root < duration:
                    (periapses if f0 < 0.0 else apoapses).append(root)
            return periapses, apoapses
        first = seconds_to_periapsis(self.r0, self.v0, self.mu)
        if first is None:
            return periapses, apoapses
        if period is None:
            if 0.0 < first < duration:
                periapses.append(first)
            return periapses, apoapses
        # first is in [0, period). Every time is built from `first` by adding
        # multiples of the period, never by subtracting one first: with a
        # period of 1e17 s, (first - P) + P loses first to rounding.
        half = period / 2.0
        if 0.0 < first - half < duration:
            apoapses.append(first - half)  # the apoapsis before the next periapsis
        k = 0
        while first + k * period < duration:
            periapsis = first + k * period
            if periapsis > 0.0:
                periapses.append(periapsis)
            if periapsis + half < duration:
                apoapses.append(periapsis + half)
            k += 1
        return periapses, apoapses

    def periapsis_times(self) -> list[float]:
        return self._turning()[0]

    def turning_times(self) -> list[float]:
        """Periapsis and apoapsis passages inside (0, duration)."""
        periapses, apoapses = self._turning()
        return sorted(periapses + apoapses)

    def maximum_radius(self) -> float:
        """Largest radius over the arc: an apoapsis inside it, else an end."""
        points = [0.0, self.duration_s, *self.turning_times()]
        return max(self.radius(t) for t in points)


@dataclass(frozen=True, slots=True)
class Passage:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def below_radius(arc: Arc, radius_km: float) -> list[Passage]:
    """Every stretch of the arc spent below `radius_km`, in time order.

    Between the ends and the turning points the radius is monotone, so each
    stretch is bracketed and its ends found by root search on the exact coast.
    """
    points = [0.0, *arc.turning_times(), arc.duration_s]
    passages: list[Passage] = []
    inside_since: float | None = None

    def f(t: float) -> float:
        return arc.radius(t) - radius_km

    previous = points[0]
    value_previous = f(previous)
    if value_previous < 0.0:
        inside_since = 0.0
    for point in points[1:]:
        value = f(point)
        if (value_previous < 0.0) != (value < 0.0):
            crossing = float(brentq(f, previous, point, xtol=1.0e-9, rtol=1.0e-12))
            if value < 0.0:
                inside_since = crossing
            elif inside_since is not None:
                passages.append(Passage(inside_since, crossing))
                inside_since = None
        previous, value_previous = point, value
    if inside_since is not None:
        passages.append(Passage(inside_since, arc.duration_s))
    # A minimum within 1e-12 of the radius is a tangency, not a crossing: the
    # coast never goes below it, and refusing it would discard the round.
    if not passages and arc.minimum_radius() < radius_km * (1.0 - 1.0e-12):
        # The minimum says the arc may go below and the brackets saw nothing:
        # a near-circle, whose minimum is only a lower bound, or a lost
        # turning time.
        near_circle = arc.eccentricity() < CIRCULAR_E
        passages = _below_by_sampling(arc, radius_km, refuse_none=not near_circle)
    return passages


def _below_by_sampling(
    arc: Arc, radius_km: float, samples: int = 4001, *, refuse_none: bool = True
) -> list[Passage]:
    """Fallback: bracket on a dense grid plus the arc's minimum, then root-find.

    With `refuse_none`, finding nothing is refused: the caller's minimum was
    exact and below the radius, so the sampling missed a dip.
    """
    times = np.linspace(0.0, arc.duration_s, samples)
    radii = np.linalg.norm(arc.states(times)[0], axis=1)
    k = int(np.argmin(radii))
    lo, hi = float(times[max(k - 1, 0)]), float(times[min(k + 1, samples - 1)])
    best = minimize_scalar(arc.radius, bounds=(lo, hi), method="bounded",
                           options={"xatol": 1.0e-9})
    points = np.sort(np.append(times, float(best.x)))
    values = [arc.radius(float(t)) - radius_km for t in points]
    passages: list[Passage] = []
    inside_since = 0.0 if values[0] < 0.0 else None
    for (t0, f0), (t1, f1) in zip(zip(points, values, strict=True),
                                  zip(points[1:], values[1:], strict=True), strict=False):
        if (f0 < 0.0) == (f1 < 0.0):
            continue
        crossing = float(brentq(lambda t: arc.radius(t) - radius_km, float(t0), float(t1),
                                xtol=1.0e-9, rtol=1.0e-12))
        if f1 < 0.0:
            inside_since = crossing
        elif inside_since is not None:
            passages.append(Passage(inside_since, crossing))
            inside_since = None
    if inside_since is not None:
        passages.append(Passage(inside_since, arc.duration_s))
    if not passages and refuse_none:
        raise UnusableArc(f"the arc dips below {radius_km} km but no crossing was bracketed")
    return passages


@dataclass(slots=True)
class Piece:
    start_s: float
    end_s: float
    #: "outside", "inside" or "undecided".
    label: str
    #: Exact distances at the ends, km.
    d_start: float
    d_end: float


@dataclass(slots=True)
class CaptureFacts:
    pieces: list[Piece]
    #: First time the distance is within the radius; None for no capture.
    first_entry_s: float | None
    #: Width of undecided time just before the first entry.
    first_entry_uncertainty_s: float
    #: Undecided slivers with no exact state inside: possible grazes.
    grazes_s: list[float] = field(default_factory=list)
    #: Runs of pieces not proved outside: what hurts us counts over these.
    passages_hurt: list[Passage] = field(default_factory=list)
    #: Runs of pieces proved inside: what favours us counts over these.
    passages_favour: list[Passage] = field(default_factory=list)
    minimum_distance_km: float = math.inf
    minimum_time_s: float = 0.0
    inside_at_start: bool = False
    inside_at_end: bool = False
    #: True when the evaluation budget ran out; the rest is undecided.
    exhausted: bool = False
    #: Exact coasts the detector made, at most MAX_EVALUATIONS.
    evaluations: int = 0


def _point_segment(p: np.ndarray, w: np.ndarray, span: float) -> float:
    """min over tau in [0, span] of |p + w tau|."""
    ww = float(np.dot(w, w))
    tau = 0.0 if ww == 0.0 else min(max(-float(np.dot(p, w)) / ww, 0.0), span)
    return float(np.linalg.norm(p + w * tau))


def _chord(a: np.ndarray, b: np.ndarray) -> float:
    """min over the chord from a to b of the norm."""
    return _point_segment(a, b - a, 1.0)


def segment_bounds(
    rho0: np.ndarray,
    w0: np.ndarray,
    rho1: np.ndarray,
    w1: np.ndarray,
    h: float,
    a0: float,
    r_lo: float,
    mu: float,
) -> tuple[float, float]:
    """(LB, UB) on |rho| over a segment of length h; see the module docstring."""
    rho_hi = min(
        float(np.linalg.norm(rho0)) + float(np.linalg.norm(w0)) * h,
        float(np.linalg.norm(rho1)) + float(np.linalg.norm(w1)) * h,
    ) + a0 * h * h / 2.0
    accel = a0
    if r_lo > rho_hi / 2.0:
        accel = min(a0, 2.0 * mu * rho_hi / (r_lo - rho_hi / 2.0) ** 3)
    remainder = accel * h * h / 8.0
    half = h / 2.0
    taylor = min(_point_segment(rho0, w0, half), _point_segment(rho1, -w1, half))
    secant = _chord(rho0, rho1)
    lower = max(taylor, secant) - remainder
    upper = max(
        float(np.linalg.norm(rho0)),
        float(np.linalg.norm(rho0 + w0 * half)),
        float(np.linalg.norm(rho1 - w1 * half)),
        float(np.linalg.norm(rho1)),
    ) + remainder
    return lower, upper


def as_radius(radius: Radius | float) -> Radius:
    if callable(radius):
        return radius
    constant = float(radius)

    def fixed(_t: float) -> float:
        return constant

    return fixed


def capture_facts(
    pursuer: Arc, evader: Arc, radius: Radius | float, knot_spacing_s: float
) -> CaptureFacts:
    """Classify the round into pieces proved inside, proved outside, undecided.

    `radius` may vary with time (margins that grow, section 2.5); on each piece
    the detector uses its least favourable end value for each test: a piece is
    proved outside only beyond the larger end value and inside only within the
    smaller.
    """
    if pursuer.duration_s != evader.duration_s:
        raise ValueError("both arcs must cover the same round")
    radius_at = as_radius(radius)
    duration = pursuer.duration_s
    mu = pursuer.mu
    count = max(1, math.ceil(duration / knot_spacing_s))
    times = np.linspace(0.0, duration, count + 1)
    rp, vp = pursuer.states(times)
    re, ve = evader.states(times)
    rmin_p, rmin_e = pursuer.minimum_radius(), evader.minimum_radius()
    if not (rmin_p > 0.0 and rmin_e > 0.0):
        raise UnusableArc(f"an arc reaches the centre (minimum radii {rmin_p}, {rmin_e} km)")
    a0 = mu / rmin_p**2 + mu / rmin_e**2
    r_lo = min(rmin_p, rmin_e)
    evaluations = 0
    exhausted = False

    def relative(t: float) -> tuple[np.ndarray, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        p_r, p_v = pursuer.state(t)
        e_r, e_v = evader.state(t)
        return p_r - e_r, p_v - e_v

    pieces: list[Piece] = []
    for k in range(count):
        stack = [(float(times[k]), float(times[k + 1]),
                  rp[k] - re[k], vp[k] - ve[k], rp[k + 1] - re[k + 1], vp[k + 1] - ve[k + 1])]
        while stack:
            t0, t1, rho0, w0, rho1, w1 = stack.pop()
            h = t1 - t0
            lb, ub = segment_bounds(rho0, w0, rho1, w1, h, a0, r_lo, mu)
            r_start, r_end = radius_at(t0), radius_at(t1)
            d0, d1 = float(np.linalg.norm(rho0)), float(np.linalg.norm(rho1))
            if lb > max(r_start, r_end):
                pieces.append(Piece(t0, t1, "outside", d0, d1))
                continue
            if ub <= min(r_start, r_end):
                pieces.append(Piece(t0, t1, "inside", d0, d1))
                continue
            if h < MIN_PIECE_S or evaluations >= MAX_EVALUATIONS:
                exhausted = exhausted or h >= MIN_PIECE_S
                pieces.append(Piece(t0, t1, "undecided", d0, d1))
                continue
            tm = 0.5 * (t0 + t1)
            rho_m, w_m = relative(tm)
            # Right half first on the stack, so the left half is resolved first.
            stack.append((tm, t1, rho_m, w_m, rho1, w1))
            stack.append((t0, tm, rho0, w0, rho_m, w_m))

    facts = CaptureFacts(pieces=pieces, first_entry_s=None, first_entry_uncertainty_s=0.0,
                         exhausted=exhausted, evaluations=evaluations)
    _first_entry(facts, radius_at)
    facts.passages_hurt = _runs(pieces, {"inside", "undecided"})
    facts.passages_favour = _runs(pieces, {"inside"})
    facts.inside_at_start = pieces[0].d_start <= radius_at(0.0)
    facts.inside_at_end = pieces[-1].d_end <= radius_at(duration)
    _closest_approach(facts, pursuer, evader, times, rp - re, a0, r_lo)
    return facts


def _runs(pieces: list[Piece], labels: set[str]) -> list[Passage]:
    runs: list[Passage] = []
    start: float | None = None
    end = 0.0
    for piece in pieces:
        if piece.label in labels:
            if start is None:
                start = piece.start_s
            end = piece.end_s
        elif start is not None:
            runs.append(Passage(start, end))
            start = None
    if start is not None:
        runs.append(Passage(start, end))
    return runs


def _first_entry(facts: CaptureFacts, radius_at: Radius) -> None:
    undecided_width = 0.0
    for piece in facts.pieces:
        if piece.label == "outside":
            undecided_width = 0.0
            continue
        if piece.label == "inside":
            facts.first_entry_s = piece.start_s
            facts.first_entry_uncertainty_s = undecided_width
            return
        if piece.d_start <= radius_at(piece.start_s):
            facts.first_entry_s = piece.start_s
            facts.first_entry_uncertainty_s = undecided_width
            return
        if piece.d_end <= radius_at(piece.end_s):
            facts.first_entry_s = piece.end_s
            facts.first_entry_uncertainty_s = undecided_width + (piece.end_s - piece.start_s)
            return
        facts.grazes_s.append(0.5 * (piece.start_s + piece.end_s))
        undecided_width += piece.end_s - piece.start_s


def _closest_approach(
    facts: CaptureFacts,
    pursuer: Arc,
    evader: Arc,
    times: np.ndarray,
    rho: np.ndarray,
    a0: float,
    r_lo: float,
) -> None:
    distances = np.linalg.norm(rho, axis=1)
    best = int(np.argmin(distances))
    facts.minimum_distance_km = float(distances[best])
    facts.minimum_time_s = float(times[best])

    def distance(t: float) -> float:
        return float(np.linalg.norm(pursuer.state(t)[0] - evader.state(t)[0]))

    _, vp = pursuer.states(times)
    _, ve = evader.states(times)
    w = vp - ve
    order = np.argsort(np.minimum(distances[:-1], distances[1:]))
    for k in order:
        h = float(times[k + 1] - times[k])
        lb, _ = segment_bounds(rho[k], w[k], rho[k + 1], w[k + 1], h, a0, r_lo, pursuer.mu)
        if lb >= facts.minimum_distance_km:
            continue
        result = minimize_scalar(distance, bounds=(float(times[k]), float(times[k + 1])),
                                 method="bounded", options={"xatol": 1.0e-6})
        if float(result.fun) < facts.minimum_distance_km:
            facts.minimum_distance_km = float(result.fun)
            facts.minimum_time_s = float(result.x)
    for piece in facts.pieces:
        if piece.label == "inside":
            for t, d in ((piece.start_s, piece.d_start), (piece.end_s, piece.d_end)):
                if d < facts.minimum_distance_km:
                    facts.minimum_distance_km, facts.minimum_time_s = d, t
