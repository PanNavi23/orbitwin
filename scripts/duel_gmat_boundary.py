"""Check that GMAT writes a report row at every round boundary (DUEL_ENGINE.md 2.6).

`boundary_rows_judged` assumes the organizer's report has a row at each round
end, wherever the integrator's steps fall. Two craft (a 7000 km circle and an
eccentric orbit to 13000 km) are flown for six rounds of an off-grid length
under RK89 at its default accuracy, in both `Propagate P(A, B)` and
`Propagate Synchronized P(A) P(B)` forms, with A burning 10 m/s between
rounds as a match does. For each form the check finds, for every round end
k * T, a row whose epoch is within 1e-5 s of it for both craft, compares
that row's positions with our coast, and checks the first row is the
initial state.

    python scripts/duel_gmat_boundary.py --out ../output/duel/gmat-boundary
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from orbitwin.conic import coast
from orbitwin.duel.params import GMAT_MU_KM3_S2
from orbitwin.gmat_runner import run_gmat

MU = GMAT_MU_KM3_S2
ROUND_S = 1234.567
ROUNDS = 6
REPORT = "boundary.txt"
CRAFT = {
    "A": (np.array([7000.0, 0.0, 0.0]), np.array([0.0, math.sqrt(MU / 7000.0), 0.0])),
    "B": (np.array([0.0, 7000.0 * math.cos(0.9), 7000.0 * math.sin(0.9)]),
          np.array([-math.sqrt(MU * (2 / 7000.0 - 1 / 10000.0)), 0.0, 0.0])),
}
FORMS = {
    "joint": "Propagate Prop(A, B) {A.ElapsedSecs = %s};",
    "synchronized": "Propagate Synchronized Prop(A) Prop(B) {A.ElapsedSecs = %s};",
}
AXES = ("X", "Y", "Z", "VX", "VY", "VZ")
#: A's burn after each round but the last, km/s, EarthMJ2000Eq.
BURN_KM_S = np.array([0.0, 0.01, 0.0])


def script(form: str) -> str:
    lines: list[str] = []
    for name, (r, v) in CRAFT.items():
        lines += [
            f"Create Spacecraft {name};", f"{name}.DateFormat = UTCGregorian;",
            f"{name}.Epoch = '09 Nov 2026 00:00:00.000';",
            f"{name}.CoordinateSystem = EarthMJ2000Eq;", f"{name}.DisplayStateType = Cartesian;",
            *(f"{name}.{axis} = {float(value)!r};"
              for axis, value in zip(AXES, (*r, *v), strict=True)),
        ]
    lines += [
        "Create ForceModel FM;", "FM.CentralBody = Earth;", "FM.PointMasses = {Earth};",
        "FM.Drag = None;", "FM.SRP = Off;", "FM.RelativisticCorrection = Off;",
        "FM.ErrorControl = RSSStep;",
        "Create Propagator Prop;", "Prop.FM = FM;", "Prop.Type = RungeKutta89;",
        "Prop.InitialStepSize = 60;", "Prop.Accuracy = 1e-11;", "Prop.MinStep = 0.001;",
        "Prop.MaxStep = 2700;",
        "Create ReportFile Out;", f"Out.Filename = '{REPORT}';", "Out.Precision = 16;",
        "Out.WriteHeaders = false;",
        "Out.Add = {A.A1ModJulian, B.A1ModJulian, "
        + ", ".join(f"{c}.EarthMJ2000Eq.{axis}" for c in CRAFT for axis in AXES) + "};",
        "Create ImpulsiveBurn DV;", "DV.CoordinateSystem = EarthMJ2000Eq;",
        *(f"DV.Element{k + 1} = {float(x)!r};" for k, x in enumerate(BURN_KM_S)),
        "DV.DecrementMass = false;",
        "BeginMissionSequence;",
    ]
    for k in range(ROUNDS):
        lines.append(FORMS[form] % ROUND_S)
        if k < ROUNDS - 1:
            lines.append("Maneuver DV(A);")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class BoundaryResult:
    form: str
    rows: int
    #: Per round end: the smallest |row epoch - k T| over rows, s, for each craft.
    epoch_miss_s: list[float]
    #: Per round end: the larger craft's position error against our coast, km.
    position_error_km: list[float]
    #: Rows strictly between two round ends: whether any exist (steps are rows).
    interior_rows: int
    #: The first row against the initial states, km.
    start_error_km: float


def check(console: str, startup: str, out: Path) -> list[BoundaryResult]:
    results = []
    for form in FORMS:
        run_dir = out / form
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "boundary.script"
        path.write_text(script(form), encoding="ascii")
        run = run_gmat(console, path, run_dir / "run", [REPORT], startup_file=startup,
                       timeout_s=600.0)
        if not run.success:
            raise RuntimeError(f"{form}: {run.error}")
        table = np.loadtxt(run.reports[REPORT], ndmin=2)
        t_a = (table[:, 0] - table[0, 0]) * 86400.0
        t_b = (table[:, 1] - table[0, 1]) * 86400.0
        misses: list[float] = []
        errors: list[float] = []
        states = {name: (r.copy(), v.copy()) for name, (r, v) in CRAFT.items()}
        for k in range(1, ROUNDS + 1):
            goal = k * ROUND_S
            row = int(np.argmin(np.abs(t_a - goal)))
            misses.append(max(abs(float(t_a[row]) - goal), abs(float(t_b[row]) - goal)))
            worst = 0.0
            for index, name in enumerate(CRAFT):
                states[name] = coast(*states[name], ROUND_S, MU)
                got = table[row, 2 + 6 * index:5 + 6 * index]
                worst = max(worst, float(np.linalg.norm(states[name][0] - got)))
            errors.append(worst)
            r_a, v_a = states["A"]
            states["A"] = (r_a, v_a + BURN_KM_S)
        start = max(float(np.linalg.norm(table[0, 2 + 6 * i:8 + 6 * i] - np.concatenate(rv)))
                    for i, rv in enumerate(CRAFT.values()))
        ends = np.array([k * ROUND_S for k in range(ROUNDS + 1)])
        interior = int(np.sum(np.min(np.abs(t_a[:, None] - ends[None, :]), axis=1) > 1e-3))
        results.append(BoundaryResult(form, len(t_a), misses, errors, interior, start))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    console, startup = os.environ.get("GMAT_CONSOLE"), os.environ.get("GMAT_STARTUP")
    if not (console and startup):
        print("GMAT_CONSOLE and GMAT_STARTUP must be set", file=sys.stderr)
        return 2
    out = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results = check(console, startup, out)
    text = json.dumps([asdict(r) for r in results], indent=2)
    (out / "boundary.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
