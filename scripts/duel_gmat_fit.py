"""Fit the duel's per-setting GMAT constants: row-gap c, position drift, a drift.

For each propagator setting the organizer might use, one craft is flown in
GmatConsole R2026a (point-mass Earth) on a grid of eccentricities and
starting phases, one report row per integrator step. From the rows:

- c: the largest gap * sqrt(mu) / r_hi^1.5 over all row gaps, r_hi the larger
  radius of the gap's two rows (a gap that straddles an apoapsis makes r_hi
  too small, which only makes c larger), times 1.5 (DUEL_ENGINE.md 2.6);
- position error against our coast from the same state, as floor + rate * t
  (see `summarise`), per craft;
- a: the largest |SMA_GMAT - a_0|, km, plus the largest |SMA_GMAT - eq. (2)
  on GMAT's own state|, which measures how GMAT computes a.

The stored values are 1.5 times the measured ones; the table in
`orbitwin.duel.gmat_settings` is copied from this script's table.json.

GmatConsole runs only through gmat_runner.run_gmat.

    python scripts/duel_gmat_fit.py --out ../output/duel/gmat-fit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from orbitwin.conic import coast_many
from orbitwin.duel.params import GMAT_MU_KM3_S2
from orbitwin.gmat_runner import run_gmat

MU = GMAT_MU_KM3_S2
SETTINGS = {
    "rk89-default": ("RungeKutta89", 1e-11),
    "rk89-1e-10": ("RungeKutta89", 1e-10),
    "rk89-1e-9": ("RungeKutta89", 1e-9),
    "pd78": ("PrinceDormand78", 1e-11),
    "rk56": ("RungeKutta56", 1e-11),
    "pd45": ("PrinceDormand45", 1e-11),
}
#: Periapsis 7000 km; e from circular to hyperbolic.
ECCENTRICITIES = (0.0, 0.3, 0.5, 0.7, 0.78, 0.9, 0.95, 0.99, 1.2, 2.0)
PHASES_DEG = (0.0, 120.0, 240.0)
PERIAPSIS_KM = 7000.0
#: Every orbit is flown this long, so the stored rows hold over it for all of
#: them (gmat_settings.MEASURED_SPAN_S).
DURATION_S = 2.0 * 86400.0
REPORT = "fit.txt"
#: No cap on the step: GMAT's default MaxStep of 2700 s cut the row gaps of
#: the eccentric orbits exactly where c is largest, so a fit under it holds
#: only for MaxStep <= 2700. With the cap lifted c is the integrator's own.
MAX_STEP_S = 1.0e6
#: The position error's floor is the largest error this early.
FLOOR_WINDOW_S = 600.0


def initial_state(e: float, phase_deg: float) -> tuple[np.ndarray, np.ndarray, float]:
    """State at true anomaly `phase` on the conic with periapsis 7000 km."""
    p = PERIAPSIS_KM * (1.0 + e)
    nu = math.radians(phase_deg)
    if e >= 1.0:
        limit = math.acos(-1.0 / e) * 0.9
        nu = max(-limit, min(limit, nu if nu <= math.pi else nu - 2.0 * math.pi))
    r = p / (1.0 + e * math.cos(nu))
    position = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    velocity = math.sqrt(MU / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
    return position, velocity, DURATION_S


def script(kind: str, accuracy: float, r: np.ndarray, v: np.ndarray, duration: float) -> str:
    axes = ("X", "Y", "Z", "VX", "VY", "VZ")
    lines = [
        "Create Spacecraft S;", "S.DateFormat = UTCGregorian;",
        "S.Epoch = '09 Nov 2026 00:00:00.000';", "S.CoordinateSystem = EarthMJ2000Eq;",
        "S.DisplayStateType = Cartesian;",
        *(f"S.{axis} = {float(value)!r};" for axis, value in zip(axes, (*r, *v), strict=True)),
        "Create ForceModel FM;", "FM.CentralBody = Earth;", "FM.PointMasses = {Earth};",
        "FM.Drag = None;", "FM.SRP = Off;", "FM.RelativisticCorrection = Off;",
        "FM.ErrorControl = RSSStep;",
        "Create Propagator Prop;", "Prop.FM = FM;", f"Prop.Type = {kind};",
        "Prop.InitialStepSize = 60;", f"Prop.Accuracy = {accuracy};",
        "Prop.MinStep = 0.001;", f"Prop.MaxStep = {MAX_STEP_S};",
        "Create ReportFile Out;", f"Out.Filename = '{REPORT}';", "Out.Precision = 16;",
        "Out.WriteHeaders = false;",
        "Out.Add = {S.ElapsedSecs, " + ", ".join(f"S.EarthMJ2000Eq.{a}" for a in axes)
        + ", S.Earth.SMA};",
        "BeginMissionSequence;", f"Propagate Prop(S) {{S.ElapsedSecs = {duration}}};",
    ]
    return "\n".join(lines) + "\n"


def fly(console: str, startup: str, run_dir: Path, kind: str, accuracy: float,
        e: float, phase: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One craft in GmatConsole: elapsed seconds, error against our coast, rows, SMA."""
    r0, v0, duration = initial_state(e, phase)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "fit.script"
    path.write_text(script(kind, accuracy, r0, v0, duration), encoding="ascii")
    run = run_gmat(console, path, run_dir / "run", [REPORT], startup_file=startup,
                   timeout_s=600.0)
    if not run.success:
        raise RuntimeError(f"{run_dir.name}: {run.error}")
    rows = np.loadtxt(run.reports[REPORT], ndmin=2)
    t, state, sma = rows[:, 0], rows[:, 1:7], rows[:, 7]
    count = len(t)
    coast_r, _ = coast_many(np.tile(r0, (count, 1)), np.tile(v0, (count, 1)), t, MU)
    error = np.linalg.norm(coast_r - state[:, :3], axis=1)
    return t, error, state, sma


def summarise(runs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
              a0s: list[float]) -> dict[str, float]:
    """The setting's row: measured maxima and the stored values (times 1.5).

    The position error is bounded by floor + rate * t: the floor is the
    largest error in the first 600 s, the rate the largest (error - floor)/t
    after that, so every row measured is under the line.
    """
    root_mu = math.sqrt(MU)
    c = 0.0
    floor = 0.0
    a_drift = 0.0
    a_method = 0.0
    for (t, error, state, sma), a0 in zip(runs, a0s, strict=True):
        radius = np.linalg.norm(state[:, :3], axis=1)
        gaps = np.diff(t)
        r_hi = np.maximum(radius[:-1], radius[1:])
        live = gaps > 1e-6
        c = max(c, float(np.max(gaps[live] * root_mu / r_hi[live] ** 1.5)))
        floor = max(floor, float(np.max(error[t <= FLOOR_WINDOW_S])))
        a_drift = max(a_drift, float(np.max(np.abs(sma - a0))))
        speed2 = np.einsum("ij,ij->i", state[:, 3:], state[:, 3:])
        a_method = max(a_method, float(np.max(np.abs(sma - 1.0 / (2.0 / radius - speed2 / MU)))))
    rate = 0.0
    for t, error, _, _ in runs:
        late = t > FLOOR_WINDOW_S
        if late.any():
            rate = max(rate, float(np.max((error[late] - floor) / t[late])))
    return {
        "c_measured": c,
        "position_floor_km_measured": floor,
        "position_rate_km_s_measured": rate,
        "a_drift_km_measured": a_drift,
        "a_method_km_measured": a_method,
        "row_gap_c": 1.5 * c,
        "position_floor_km": 1.5 * floor,
        "position_rate_km_s": 1.5 * rate,
        "a_km": 1.5 * (a_drift + a_method),
    }


def measure(console: str, startup: str, out: Path,
            settings: dict[str, tuple[str, float]] = SETTINGS,
            eccentricities: tuple[float, ...] = ECCENTRICITIES,
            phases: tuple[float, ...] = PHASES_DEG) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for name, (kind, accuracy) in settings.items():
        runs = []
        a0s = []
        for e in eccentricities:
            for phase in phases:
                runs.append(fly(console, startup, out / name / f"e{e}-p{int(phase)}",
                                kind, accuracy, e, phase))
                r0, v0, _ = initial_state(e, phase)
                a0s.append(1.0 / (2.0 / float(np.linalg.norm(r0)) - float(np.dot(v0, v0)) / MU))
        table[name] = summarise(runs, a0s)
        print(name, table[name], flush=True)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    console, startup = os.environ.get("GMAT_CONSOLE"), os.environ.get("GMAT_STARTUP")
    if not (console and startup):
        print("GMAT_CONSOLE and GMAT_STARTUP must be set", file=sys.stderr)
        return 2
    out = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    table = measure(console, startup, out)
    (out / "table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
