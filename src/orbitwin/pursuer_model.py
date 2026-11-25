"""The pursuer under J2 and GMAT's SPAD drag term, for the ranking search.

Under the section 7 force model GMAT flies two-body ranking plans 27 and 47 km
wide on R3 and R4. Flown in GMAT R2026a with the forces switched on one at a
time (2026-11-02), J2 moved the pursuer 30.2 and 41.8 km, drag 10.9 and 65.5 km
on top of it, the rest of EGM96 20x20 0.6 km, and tides, Sun/Luna, SRP and
relativity under 10 m together. J2 plus the drag table below lands 0.5 and
1.1 km from GMAT's full model.

GMAT's SPADFile drag with SphericalModel.spo pushes along the relative wind:
the pursuer gains orbital energy through perigee. The table carries the sign
GMAT flies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .errors import ConfigurationError, DynamicsError
from .gmat_parser import GmatScriptDocument, propagator_calls
from .models import Scenario

#: GMAT's ModJulian is the Julian date less this.
_GMAT_MJD_OFFSET = 2430000.0
_J2000_JD = 2451545.0

#: EGM96 C20, unnormalised, and the radius it is given with.
EGM96_J2 = 1.0826266835531513e-3
EGM96_RADIUS_KM = 6378.1363
WGS84_FLATTENING = 1.0 / 298.257223563
EARTH_ROTATION_RAD_S = 7.292115146706979e-5
#: The drag table is extrapolated no further than these: below, it only keeps
#: SLSQP's trial points finite; above, drag is nothing at these speeds.
_DENSITY_FLOOR_KM = 50.0
_DENSITY_CEILING_KM = 2000.0

#: ln(rho * B / m^-1) against altitude, measured on GMAT R2026a's residual
#: accelerations along R3 and R4 flights: NRLMSISE00, constant F10.7 = F10.7A =
#: 150, MagneticIndex 3, SPAD SphericalModel.spo, 850 kg. 8602 rows from 109 to
#: 420 km; rms 7 % about the table. All at 2026-11-09 02:00 UTC, below 160 km
#: all south of the equator: the same R4 plan flown 6 to 18 h later moved
#: 0.7 to 0.9 km at arrival, 30 days later 2.5 km, 182 days 5.5 km.
R2026A_NRLMSISE00_F150_SPAD_SPHERE_850KG = (
    (100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 140.0, 150.0, 160.0, 180.0, 200.0,
     250.0, 300.0, 350.0, 420.0),
    (-18.90821, -19.87722, -20.84623, -21.64273, -22.18621, -22.61811, -23.19713,
     -23.90978, -24.46061, -24.9674, -25.66532, -26.38784, -27.60491, -28.66636,
     -29.61584, -30.75443),
)


@dataclass(frozen=True, slots=True)
class DragTable:
    """rho * B, piecewise exponential in altitude, and which way it pushes."""

    knots_km: tuple[float, ...]
    log_rho_b: tuple[float, ...]
    #: +1 along the relative wind, as GMAT's SPAD sphere; -1 against it.
    sign: float

    def extended(self) -> tuple[np.ndarray, np.ndarray]:
        """Knots carried on with the end slopes to _DENSITY_FLOOR_KM and 2000 km,
        so one np.interp covers every altitude."""
        knots = list(self.knots_km)
        logs = list(self.log_rho_b)
        low_slope = (logs[1] - logs[0]) / (knots[1] - knots[0])
        high_slope = (logs[-1] - logs[-2]) / (knots[-1] - knots[-2])
        if knots[0] > _DENSITY_FLOOR_KM:
            logs.insert(0, logs[0] + low_slope * (_DENSITY_FLOOR_KM - knots[0]))
            knots.insert(0, _DENSITY_FLOOR_KM)
        if knots[-1] < _DENSITY_CEILING_KM:
            logs.append(logs[-1] + high_slope * (_DENSITY_CEILING_KM - knots[-1]))
            knots.append(_DENSITY_CEILING_KM)
        return np.asarray(knots, dtype=float), np.asarray(logs, dtype=float)

    def rho_b(self, altitude_km: np.ndarray) -> np.ndarray:
        knots, logs = self.extended()
        value: np.ndarray = np.exp(np.interp(altitude_km, knots, logs))
        return value


def precessed_pole(julian_centuries_tt: float) -> tuple[float, float, float]:
    """Earth's mean pole of date in J2000 axes, IAU 1976 precession.

    GMAT's J2 acts about the pole of date; on R4 the J2000 z-axis put the
    pursuer 0.41 km from GMAT's J2-only flight, the pole of date 3 m.
    """
    t = julian_centuries_tt
    arcsec = math.pi / 180.0 / 3600.0
    zeta = (2306.2181 * t + 0.30188 * t * t + 0.017998 * t**3) * arcsec
    theta = (2004.3109 * t - 0.42665 * t * t - 0.041833 * t**3) * arcsec
    # Third row of P = R3(-z) R2(theta) R3(-zeta); the angle z drops out of it.
    return (
        math.cos(zeta) * math.sin(theta),
        -math.sin(zeta) * math.sin(theta),
        math.cos(theta),
    )


@dataclass(frozen=True, slots=True)
class Arcs:
    """Fixed-step flights of several start states over one span, every node kept.

    `times_s` are the nodes' times from the start, shared by every row;
    r, v and a are (nodes, rows, 3).
    """

    times_s: np.ndarray
    r: np.ndarray
    v: np.ndarray
    a: np.ndarray

    def _locate(self, elapsed_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment index, fraction through it and its signed length, per time."""
        nodes = self.times_s
        elapsed = np.asarray(elapsed_s, dtype=float)
        forward = nodes[-1] >= nodes[0]
        keys = nodes if forward else -nodes
        probe = elapsed if forward else -elapsed
        index = np.clip(np.searchsorted(keys, probe, "right") - 1, 0, len(nodes) - 2)
        h = nodes[index + 1] - nodes[index]
        safe = np.where(h == 0.0, 1.0, h)
        u = np.where(h == 0.0, 0.0, (elapsed - nodes[index]) / safe)
        return index, u, h

    def _outside(self, elapsed: np.ndarray) -> np.ndarray:
        """-1 before the span's first node, +1 past its last, 0 within; in elapsed time."""
        low, high = sorted((float(self.times_s[0]), float(self.times_s[-1])))
        side: np.ndarray = np.where(elapsed < low, -1, np.where(elapsed > high, 1, 0))
        return side

    def sample(self, row: int, elapsed_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row `row` at each elapsed time: cubic Hermite on position and on velocity,
        and a second-order Taylor step from the nearer end outside the span."""
        elapsed = np.asarray(elapsed_s, dtype=float)
        index, u, h = self._locate(elapsed)
        h00, h10, h01, h11 = (weight[:, None] for weight in _basis(u))
        hc = h[:, None]
        r0, r1 = self.r[index, row], self.r[index + 1, row]
        v0, v1 = self.v[index, row], self.v[index + 1, row]
        a0, a1 = self.a[index, row], self.a[index + 1, row]
        position: np.ndarray = h00 * r0 + h10 * hc * v0 + h01 * r1 + h11 * hc * v1
        velocity: np.ndarray = h00 * v0 + h10 * hc * a0 + h01 * v1 + h11 * hc * a1
        side = self._outside(elapsed)
        if np.any(side):
            # Hermite on a partial step a microsecond long would explode here.
            forward = self.times_s[-1] >= self.times_s[0]
            end = np.where((side > 0) == forward, len(self.times_s) - 1, 0)[side != 0]
            dt = (elapsed[side != 0] - self.times_s[end])[:, None]
            r, v, a = self.r[end, row], self.v[end, row], self.a[end, row]
            position[side != 0] = r + v * dt + 0.5 * a * dt * dt
            velocity[side != 0] = v + a * dt
        return position, velocity

    def sample_all(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Every row at one elapsed time, as sample() takes it."""
        side = int(self._outside(np.array([elapsed_s]))[0])
        if side:
            forward = self.times_s[-1] >= self.times_s[0]
            end = len(self.times_s) - 1 if (side > 0) == forward else 0
            dt = float(elapsed_s - self.times_s[end])
            r_out: np.ndarray = self.r[end] + self.v[end] * dt + 0.5 * self.a[end] * dt * dt
            v_out: np.ndarray = self.v[end] + self.a[end] * dt
            return r_out, v_out
        index, u, h = self._locate(np.array([elapsed_s]))
        i, step = int(index[0]), float(h[0])
        h00, h10, h01, h11 = (float(weight[0]) for weight in _basis(u))
        r: np.ndarray = (h00 * self.r[i] + h10 * step * self.v[i] + h01 * self.r[i + 1]
                         + h11 * step * self.v[i + 1])
        v: np.ndarray = (h00 * self.v[i] + h10 * step * self.a[i] + h01 * self.v[i + 1]
                         + h11 * step * self.a[i + 1])
        return r, v


def _basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u2 = u * u
    u3 = u2 * u
    return 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u, -2 * u3 + 3 * u2, u3 - u2


@dataclass(frozen=True, slots=True)
class PursuerModel:
    """Point mass, J2 about the pole of date, and a tabulated drag term."""

    name: str
    mu_km3_s2: float
    j2: float
    radius_km: float
    flattening: float
    pole: tuple[float, float, float]
    rotation_rad_s: float
    drag: DragTable | None
    #: RK4 step. On R4's 12-burn plan, against 2 s steps at t_f: 8 m at 10 s,
    #: 41 m at 15 s, 127 m at 20 s; the model is 0.5 to 2.4 km from GMAT.
    step_s: float = 15.0
    _drag_knots: np.ndarray | None = field(init=False, repr=False, compare=False, default=None)
    _drag_logs: np.ndarray | None = field(init=False, repr=False, compare=False, default=None)
    _pole: np.ndarray | None = field(init=False, repr=False, compare=False, default=None)
    #: omega [pole]x, transposed: r @ _spin is the co-rotating air's velocity.
    _spin: np.ndarray | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        knots, logs = (None, None) if self.drag is None else self.drag.extended()
        px, py, pz = self.pole
        cross = np.array([[0.0, -pz, py], [pz, 0.0, -px], [-py, px, 0.0]])
        object.__setattr__(self, "_drag_knots", knots)
        object.__setattr__(self, "_drag_logs", logs)
        object.__setattr__(self, "_pole", np.array(self.pole, dtype=float))
        object.__setattr__(self, "_spin", self.rotation_rad_s * cross.T)

    def acceleration(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        pole = self._pole
        assert pole is not None and self._spin is not None
        r2 = np.einsum("ij,ij->i", r, r)
        rn = np.sqrt(r2)
        along_pole = r @ pole
        sin2 = along_pole * along_pole / r2
        g = -self.mu_km3_s2 / (r2 * rn)
        k = 1.5 * self.j2 * self.radius_km**2 / r2
        a: np.ndarray = r * (g * (1.0 + k * (1.0 - 5.0 * sin2)))[:, None] + np.outer(
            2.0 * g * k * along_pole, pole
        )
        if self.drag is None or self._drag_knots is None or self._drag_logs is None:
            return a
        altitude = rn - self.radius_km * (1.0 - self.flattening * sin2)
        wind = v - r @ self._spin
        speed = np.sqrt(np.einsum("ij,ij->i", wind, wind))
        # rho*B in 1/m with km/s speeds: 0.5 * rhoB * 1e3 * |w| * w in km/s^2.
        push = (self.drag.sign * 0.5e3) * np.exp(
            np.interp(altitude, self._drag_knots, self._drag_logs)
        ) * speed
        a += push[:, None] * wind
        return a

    def nodes(self, start_s: float, duration_s: float) -> np.ndarray:
        """Node times from the start: every multiple of step_s since t = 0 strictly
        inside the span, and its ends. A moved burn moves only the partial steps
        at its ends, so the flight stays smooth in the burn epochs."""
        end_s = start_s + duration_s
        low, high = min(start_s, end_s), max(start_s, end_s)
        grid = np.arange(math.floor(low / self.step_s) + 1, math.ceil(high / self.step_s))
        inside = grid * self.step_s
        inside = inside[(inside > low) & (inside < high)]
        if duration_s < 0.0:
            inside = inside[::-1]
        return np.concatenate([[start_s], inside, [end_s]]) - start_s

    def propagate(
        self, r: np.ndarray, v: np.ndarray, duration_s: float, start_s: float = 0.0
    ) -> Arcs:
        """RK4 from each (r, v) over one span starting at start_s, on nodes()."""
        r = np.array(r, dtype=float, ndmin=2)
        v = np.array(v, dtype=float, ndmin=2)
        duration = float(duration_s)
        if not (math.isfinite(duration) and math.isfinite(start_s) and np.all(np.isfinite(r))
                and np.all(np.isfinite(v))):
            raise DynamicsError("a pursuer-model flight has a non-finite start or duration")
        times = self.nodes(float(start_s), duration)
        rs = np.empty((len(times), *r.shape))
        vs = np.empty_like(rs)
        accs = np.empty_like(rs)
        a = self.acceleration(r, v)
        rs[0], vs[0], accs[0] = r, v, a
        for k, h in enumerate(np.diff(times)):
            r2 = r + 0.5 * h * v
            v2 = v + 0.5 * h * a
            a2 = self.acceleration(r2, v2)
            r3 = r + 0.5 * h * v2
            v3 = v + 0.5 * h * a2
            a3 = self.acceleration(r3, v3)
            r4 = r + h * v3
            v4 = v + h * a3
            a4 = self.acceleration(r4, v4)
            r = r + h / 6.0 * (v + 2.0 * v2 + 2.0 * v3 + v4)
            v = v + h / 6.0 * (a + 2.0 * a2 + 2.0 * a3 + a4)
            a = self.acceleration(r, v)
            rs[k + 1], vs[k + 1], accs[k + 1] = r, v, a
        if not np.all(np.isfinite(rs)):
            raise DynamicsError("a pursuer-model flight left the finite numbers")
        return Arcs(times, rs, vs, accs)


@dataclass(frozen=True, slots=True)
class Flown:
    """One schedule flown arc by arc: arc j starts at 0 or at burn j - 1, the last
    runs to `horizon_s`."""

    arcs: tuple[Arcs, ...]
    r_at_burns: np.ndarray
    v_before: np.ndarray
    v_after: np.ndarray


def fly(
    model: PursuerModel,
    initial_r: np.ndarray,
    initial_v: np.ndarray,
    times_s: np.ndarray,
    impulses_km_s: np.ndarray,
    horizon_s: float,
) -> Flown:
    """The schedule in `model`. Every caller flies it here, so the pre-burn
    states a plan's frames were taken on come back bit for bit."""
    times = np.asarray(times_s, dtype=float).reshape(-1)
    count = len(times)
    impulses = np.asarray(impulses_km_s, dtype=float).reshape(count, 3)
    r = np.asarray(initial_r, dtype=float)
    v = np.asarray(initial_v, dtype=float)
    starts = [0.0, *times]
    ends = [*times, horizon_s]
    r_at = np.zeros((count, 3))
    before = np.zeros((count, 3))
    after = np.zeros((count, 3))
    arcs: list[Arcs] = []
    for index, (begin, end) in enumerate(zip(starts, ends, strict=True)):
        arc = model.propagate(r[None, :], v[None, :], end - begin, begin)
        arcs.append(arc)
        if index < count:
            r, v = arc.r[-1, 0], arc.v[-1, 0]
            r_at[index], before[index] = r, v
            v = v + impulses[index]
            after[index] = v
    return Flown(tuple(arcs), r_at, before, after)


def arc_minimum_radius(arcs: Arcs, row: int = 0) -> tuple[float, float, str]:
    """Lowest radius of one row, its elapsed time, and "start", "end" or "interior".

    An interior minimum is where r . v changes sign beside the lowest node,
    solved on the Hermite states, so its value moves smoothly with the flight
    and the gradient SLSQP gets from the STM there is the value's.
    """
    times = arcs.times_s
    radii = np.linalg.norm(arcs.r[:, row], axis=1)
    duration = float(times[-1])
    if duration == 0.0:
        return float(radii[0]), 0.0, "start"
    direction = 1.0 if duration > 0.0 else -1.0
    radial = np.einsum("ij,ij->i", arcs.r[:, row], arcs.v[:, row]) * direction
    last = len(times) - 1
    lowest = int(np.argmin(radii))
    if lowest == 0 and radial[0] >= 0.0:
        return float(radii[0]), 0.0, "start"
    if lowest == last and radial[-1] <= 0.0:
        return float(radii[-1]), duration, "end"
    segment = min(max(lowest - 1 if radial[lowest] >= 0.0 else lowest, 0), last - 1)

    def rate(elapsed_s: float) -> float:
        r, v = arcs.sample(row, np.array([elapsed_s]))
        return float(r[0] @ v[0]) * direction

    low, high = float(times[segment]), float(times[segment + 1])
    if not rate(low) < 0.0 < rate(high):
        # The lowest node is the minimum to within its own sampling.
        return float(radii[lowest]), float(times[lowest]), "interior"
    at_s = float(brentq(rate, min(low, high), max(low, high), xtol=1.0e-7, rtol=1.0e-12))
    position, _ = arcs.sample(row, np.array([at_s]))
    return float(np.linalg.norm(position[0])), at_s, "interior"


def model_for(name: str, julian_centuries_tt: float | None, mu_km3_s2: float) -> PursuerModel:
    """The named pursuer model; the pole is J2000's z without an epoch."""
    pole = (0.0, 0.0, 1.0) if julian_centuries_tt is None else precessed_pole(julian_centuries_tt)
    if name == "j2":
        drag = None
    elif name == "j2-spad-drag":
        knots, logs = R2026A_NRLMSISE00_F150_SPAD_SPHERE_850KG
        drag = DragTable(knots, logs, sign=1.0)
    else:
        raise DynamicsError(f"unknown pursuer model {name!r}")
    return PursuerModel(
        name=name,
        mu_km3_s2=mu_km3_s2,
        j2=EGM96_J2,
        radius_km=EGM96_RADIUS_KM,
        flattening=WGS84_FLATTENING,
        pole=pole,
        rotation_rad_s=EARTH_ROTATION_RAD_S,
        drag=drag,
    )


def pursuer_model_for(scenario: Scenario, name: str) -> PursuerModel | None:
    """The model `name` names for this scenario, None for two-body.

    The pole is precessed to the pursuer's probed TT epoch; without a probe it
    stays J2000's z.
    """
    if name == "two-body":
        return None
    epoch = scenario.pursuer.probed_epoch
    centuries = (
        None if epoch is None
        else (epoch.tt_mjd + _GMAT_MJD_OFFSET - _J2000_JD) / 36525.0
    )
    return model_for(name, centuries, scenario.mu_km3_s2)


#: What the drag table was measured under, on the pursuer's force model and on
#: the pursuer. Compared as numbers where they are numbers.
_SPAD_DRAG_FORCES = {
    "Drag.AtmosphereModel": "NRLMSISE00",
    "Drag.DragModel": "SPADFile",
    "Drag.F107": "150",
    "Drag.F107A": "150",
    "Drag.MagneticIndex": "3",
    "Drag.HistoricWeatherSource": "ConstantFluxAndGeoMag",
    "Drag.PredictedWeatherSource": "ConstantFluxAndGeoMag",
}
_SPAD_DRAG_CRAFT = {"DryMass": "850", "SPADDragScaleFactor": "1", "AtmosDensityScaleFactor": "1"}
#: GMAT's defaults for the fields above a script may leave out.
_FORCE_DEFAULTS = {
    "CentralBody": "Earth",
    "Drag.F107": "150",
    "Drag.F107A": "150",
    "Drag.MagneticIndex": "3",
    "Drag.HistoricWeatherSource": "ConstantFluxAndGeoMag",
    "Drag.PredictedWeatherSource": "ConstantFluxAndGeoMag",
}
_CRAFT_DEFAULTS = {"DryMass": "850", "SPADDragScaleFactor": "1", "AtmosDensityScaleFactor": "1"}


def _plain(value: str | None) -> str:
    return "" if value is None else value.strip().strip("'\"").strip()


def _same(value: str | None, wanted: str) -> bool:
    text = _plain(value)
    try:
        return float(text) == float(wanted)
    except ValueError:
        return text.lower() == wanted.lower()


def pursuer_force_models(
    scenario: Scenario, document: GmatScriptDocument
) -> tuple[list[str], list[str]]:
    """The ForceModels of every propagator that flies the pursuer, in mission order,
    and the propagators among them that name none."""
    names: list[str] = []
    unnamed: list[str] = []
    for segment in scenario.propagate_segments:
        if segment.implicit:
            continue
        for propagator, craft in propagator_calls(document.lines[segment.line_index]):
            if scenario.pursuer.name not in craft:
                continue
            force_model = _plain(scenario.propagator_fields.get(f"{propagator}.FM"))
            if not force_model:
                if propagator not in unnamed:
                    unnamed.append(propagator)
            elif force_model not in names:
                names.append(force_model)
    return names, unnamed


def calibration_mismatches(
    scenario: Scenario, document: GmatScriptDocument, name: str
) -> list[str]:
    """Why `name` does not describe this script's pursuer; empty when it does.

    Both models need EGM96 of degree 2 or more. j2-spad-drag also needs the
    drag settings and craft the table was measured on; j2 needs no drag at all,
    since J2 without the drag that follows it flew R4 65 km from GMAT, worse
    than two-body.
    """
    if name == "two-body":
        return []
    models, unnamed = pursuer_force_models(scenario, document)
    # GMAT gives a propagator without FM a force model of its own, not this one.
    found = [f"{propagator} flies {scenario.pursuer.name} with no FM" for propagator in unnamed]
    if not models:
        return [*found, f"no ForceModel is found for {scenario.pursuer.name}'s propagators"]
    fields = scenario.force_model_fields
    for model in models:
        def field_of(key: str, model: str = model) -> str | None:
            return fields.get(f"{model}.{key}", _FORCE_DEFAULTS.get(key))

        if not _same(field_of("CentralBody"), "Earth"):
            found.append(f"{model}.CentralBody is {field_of('CentralBody')!r}, not Earth")
        potential = _plain(field_of("GravityField.Earth.PotentialFile")).lower()
        if not potential.endswith("egm96.cof"):
            found.append(f"{model}.GravityField.Earth.PotentialFile is {potential!r}, not EGM96")
        try:
            degree = float(_plain(field_of("GravityField.Earth.Degree")))
        except ValueError:
            degree = math.nan
        if not degree >= 2:
            found.append(f"{model}.GravityField.Earth.Degree is under 2 or not stated")
        # GMAT still reads the deprecated `Drag = <model>` form as well.
        for key in ("Drag", "Drag.AtmosphereModel"):
            drag = _plain(fields.get(f"{model}.{key}"))
            if name == "j2" and drag and drag.lower() != "none":
                found.append(f"{model}.{key} is {drag}; j2 leaves drag out")
        if name == "j2-spad-drag":
            for key, wanted in _SPAD_DRAG_FORCES.items():
                if not _same(field_of(key), wanted):
                    found.append(f"{model}.{key} is {field_of(key)!r}, not {wanted}")
            old_form = fields.get(f"{model}.Drag")
            if old_form is not None and not _same(old_form, "NRLMSISE00"):
                found.append(f"{model}.Drag is {old_form!r} beside Drag.AtmosphereModel")
    if name == "j2-spad-drag":
        craft = scenario.pursuer.fields
        for key, wanted in _SPAD_DRAG_CRAFT.items():
            value = craft.get(key, _CRAFT_DEFAULTS.get(key))
            if not _same(value, wanted):
                found.append(f"{scenario.pursuer.name}.{key} is {value!r}, not {wanted}")
        # GMAT divides drag by the total mass; the table holds 850 kg and no fuel.
        tanks = _plain(craft.get("Tanks")).strip("{} ")
        if tanks:
            found.append(
                f"{scenario.pursuer.name}.Tanks is {craft.get('Tanks')!r}; the table is for "
                "850 kg with no tank"
            )
        spad = _plain(craft.get("SPADDragFile")).replace("\\", "/").rsplit("/", 1)[-1]
        if spad.lower() != "sphericalmodel.spo":
            found.append(
                f"{scenario.pursuer.name}.SPADDragFile is {craft.get('SPADDragFile')!r}, "
                "not SphericalModel.spo"
            )
    return found


def checked_pursuer_model(
    scenario: Scenario, document: GmatScriptDocument, name: str
) -> PursuerModel | None:
    """pursuer_model_for, refused with the reasons when the script is not what
    the model was measured on."""
    mismatches = calibration_mismatches(scenario, document, name)
    if mismatches:
        raise ConfigurationError(
            f"search.ranking_pursuer_model {name} does not fit this script: "
            + "; ".join(mismatches)
        )
    return pursuer_model_for(scenario, name)
