"""Measure GMAT's point-mass mu and check the duel rollout against GmatConsole.

Two craft (an eccentric orbit and an inclined circle) are flown for a day in
GMAT R2026a under each point-mass ForceModel form the organizer might use,
RK89 at 1e-12, one row every step. Each row is compared with the duel's
two-body coast; mu is fitted by least squares on position. GmatConsole runs
only through gmat_runner.run_gmat with a derived startup file, so nothing is
written inside the GMAT install.

    python scripts/duel_gmat_check.py --out ../output/duel/gmat-check
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from orbitwin.conic import coast_many
from orbitwin.gmat_runner import run_gmat

DURATION_S = 86400.0
CANDIDATES = (398600.4415, 398600.4418)
#: Eccentric (rp 7000 km, ra 13000 km) and an inclined 7000 km circle.
CRAFT = {
    "A": (np.array([7000.0, 0.0, 0.0]),
          np.array([0.0, math.sqrt(398600.4415 * (2 / 7000.0 - 1 / 10000.0)), 0.0])),
    "B": (np.array([0.0, 7000.0 * math.cos(0.9), 7000.0 * math.sin(0.9)]),
          np.array([-math.sqrt(398600.4415 / 7000.0), 0.0, 0.0])),
}
FORMS = {
    "primary-degree0": [
        "FM.CentralBody = Earth;", "FM.PrimaryBodies = {Earth};",
        "FM.GravityField.Earth.Degree = 0;", "FM.GravityField.Earth.Order = 0;",
        "FM.GravityField.Earth.PotentialFile = 'JGM2.cof';",
    ],
    "pointmass": ["FM.CentralBody = Earth;", "FM.PointMasses = {Earth};"],
}
REPORT = "duel_rollout.txt"


def script(form: str) -> str:
    lines: list[str] = []
    for name, (r, v) in CRAFT.items():
        lines += [
            f"Create Spacecraft {name};", f"{name}.DateFormat = UTCGregorian;",
            f"{name}.Epoch = '09 Nov 2026 00:00:00.000';",
            f"{name}.CoordinateSystem = EarthMJ2000Eq;", f"{name}.DisplayStateType = Cartesian;",
            *(f"{name}.{axis} = {float(value)!r};"
              for axis, value in zip(("X", "Y", "Z", "VX", "VY", "VZ"), (*r, *v), strict=True)),
        ]
    lines += [
        "Create ForceModel FM;", *FORMS[form], "FM.Drag = None;", "FM.SRP = Off;",
        "FM.RelativisticCorrection = Off;", "FM.ErrorControl = RSSStep;",
        "Create Propagator Prop;", "Prop.FM = FM;", "Prop.Type = RungeKutta89;",
        "Prop.InitialStepSize = 60;", "Prop.Accuracy = 1e-12;", "Prop.MinStep = 0.001;",
        "Prop.MaxStep = 300;",
        "Create ReportFile Out;", f"Out.Filename = '{REPORT}';", "Out.Precision = 16;",
        "Out.WriteHeaders = false;",
        "Out.Add = {A.ElapsedSecs, "
        + ", ".join(f"{c}.EarthMJ2000Eq.{axis}" for c in CRAFT
                    for axis in ("X", "Y", "Z", "VX", "VY", "VZ")) + "};",
        "BeginMissionSequence;",
        f"Propagate Prop(A, B) {{A.ElapsedSecs = {DURATION_S}}};",
    ]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class FormResult:
    form: str
    rows: int
    fitted_mu: float
    #: Largest position error against the coast, per candidate mu, km.
    max_error_km: dict[str, float]


def _errors(times: np.ndarray, gmat: np.ndarray, mu: float) -> np.ndarray:
    errors = []
    for index, (r, v) in enumerate(CRAFT.values()):
        count = len(times)
        position, _ = coast_many(np.tile(r, (count, 1)), np.tile(v, (count, 1)), times, mu)
        errors.append(np.linalg.norm(position - gmat[:, 6 * index:6 * index + 3], axis=1))
    return np.concatenate(errors)


def check(console: str, startup: str, out: Path) -> list[FormResult]:
    results = []
    for form in FORMS:
        run_dir = out / form
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "duel_rollout.script"
        path.write_text(script(form), encoding="ascii")
        run = run_gmat(console, path, run_dir / "run", [REPORT], startup_file=startup,
                       timeout_s=600.0)
        if not run.success:
            raise RuntimeError(f"{form}: {run.error}")
        table = np.loadtxt(run.reports[REPORT], ndmin=2)
        times, gmat = table[:, 0], table[:, 1:]
        # Fit the offset, not mu itself: the bounded method adds sqrt(eps)*|x| to
        # its tolerance, about 6e-3 at x = 4e5, wider than the whole bracket.
        base = CANDIDATES[0]

        def misfit(
            d: float, times: np.ndarray = times, gmat: np.ndarray = gmat, base: float = base
        ) -> float:
            return float(np.sum(_errors(times, gmat, base + d) ** 2))

        fit = minimize_scalar(misfit, bounds=(-0.002, 0.002), method="bounded",
                              options={"xatol": 1e-8})
        results.append(FormResult(
            form, len(times), base + float(fit.x),
            {f"{mu}": float(_errors(times, gmat, mu).max()) for mu in CANDIDATES},
        ))
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
    summary = [result.__dict__ for result in results]
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
