#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitwin.config import OrbitWinConfig, SearchConfig
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.search import search_one_burn

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sample_case(rng: np.random.Generator, minimum_pericenter_km: float) -> dict[str, float]:
    """Draw one scenario whose two spacecraft are actually in orbit.

    The ranges alone do not guarantee that. `a1` reaches down to 6800 km with
    `e1` up to 0.3, so an unfiltered draw can put a pericenter at 4760 km --
    inside the 6378.14 km Earth -- and `gmat_parser` refuses exactly that state.
    That refusal is correct, but this generator exists to measure the pipeline
    on varied *valid* different-plane geometry, so a scenario it will not accept
    is a wasted case rather than a finding. On seed 20260818 four of twenty
    draws were underground; redrawing them costs six extra draws.

    The premise check itself is covered by tests/unit/test_scenario_premises.py.
    """
    while True:
        values = {
            "a1": float(rng.uniform(6800, 14000)),
            "e1": float(rng.uniform(0, 0.3)),
            "i1": float(rng.uniform(0, 120)),
            "o1": float(rng.uniform(0, 360)),
            "w1": float(rng.uniform(0, 360)),
            "t1": float(rng.uniform(0, 360)),
            "a2": float(rng.uniform(7000, 18000)),
            "e2": float(rng.uniform(0, 0.25)),
            "i2": float(rng.uniform(0, 120)),
            "o2": float(rng.uniform(0, 360)),
            "w2": float(rng.uniform(0, 360)),
            "t2": float(rng.uniform(0, 360)),
        }
        pursuer_pericenter_km = values["a1"] * (1.0 - values["e1"])
        target_pericenter_km = values["a2"] * (1.0 - values["e2"])
        if (
            pursuer_pericenter_km > minimum_pericenter_km
            and target_pericenter_km > minimum_pericenter_km
        ):
            return values


def script_text(values: dict[str, float]) -> str:
    return f"""% Generated OOD matrix scenario.
Create Spacecraft Pursuer;
Pursuer.DateFormat = UTCGregorian;
Pursuer.Epoch = '26 Jun 2026 05:00:00.000';
Pursuer.CoordinateSystem = EarthMJ2000Eq;
Pursuer.DisplayStateType = Keplerian;
Pursuer.SMA = {values['a1']:.12g};
Pursuer.ECC = {values['e1']:.12g};
Pursuer.INC = {values['i1']:.12g};
Pursuer.RAAN = {values['o1']:.12g};
Pursuer.AOP = {values['w1']:.12g};
Pursuer.TA = {values['t1']:.12g};
Create Spacecraft AlienSC;
AlienSC.DateFormat = UTCGregorian;
AlienSC.Epoch = '26 Jun 2026 05:00:00.000';
AlienSC.CoordinateSystem = EarthMJ2000Eq;
AlienSC.DisplayStateType = Keplerian;
AlienSC.SMA = {values['a2']:.12g};
AlienSC.ECC = {values['e2']:.12g};
AlienSC.INC = {values['i2']:.12g};
AlienSC.RAAN = {values['o2']:.12g};
AlienSC.AOP = {values['w2']:.12g};
AlienSC.TA = {values['t2']:.12g};
Create ForceModel FM;
FM.CentralBody = Earth;
FM.PrimaryBodies = {{Earth}};
FM.Drag = None;
FM.SRP = Off;
FM.GravityField.Earth.Degree = 0;
FM.GravityField.Earth.Order = 0;
Create Propagator Prop;
Prop.FM = FM;
Prop.Type = RungeKutta89;
Prop.InitialStepSize = 1;
Prop.Accuracy = 1e-12;
Prop.MinStep = 0.001;
Prop.MaxStep = 2;
Prop.MaxStepAttempts = 50;
Prop.StopIfAccuracyIsViolated = true;
Create ImpulsiveBurn Burn1;
Burn1.CoordinateSystem = Local;
Burn1.Origin = Earth;
Burn1.Axes = VNB;
Burn1.Element1 = 0;
Burn1.Element2 = 0;
Burn1.Element3 = 0;
Burn1.DecrementMass = false;
Create ImpulsiveBurn Burn2;
Burn2.CoordinateSystem = Local;
Burn2.Origin = Earth;
Burn2.Axes = VNB;
Burn2.Element1 = 0;
Burn2.Element2 = 0;
Burn2.Element3 = 0;
Burn2.DecrementMass = false;
Create ReportFile OfficialReport;
OfficialReport.Filename = 'OfficialReport.txt';
OfficialReport.Precision = 16;
OfficialReport.Add = {{Pursuer.A1ModJulian, Pursuer.EarthMJ2000Eq.X}};
OfficialReport.WriteHeaders = true;
OfficialReport.AppendToExistingFile = false;
BeginMissionSequence;
Propagate Prop(Pursuer, AlienSC) {{Pursuer.ElapsedSecs = 500, AlienSC.ElapsedSecs = 500}};
Maneuver Burn1(Pursuer);
Propagate Prop(Pursuer, AlienSC) {{Pursuer.ElapsedSecs = 2500, AlienSC.ElapsedSecs = 2500}};
Maneuver Burn2(Pursuer);
Propagate Prop(Pursuer, AlienSC) {{Pursuer.ElapsedSecs = 1000, AlienSC.ElapsedSecs = 1000}};
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", default="output/scenario_ood_matrix.json")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    config = OrbitWinConfig(
        search=SearchConfig(
            seed=args.seed,
            one_burn_samples=24,
            one_burn_grid_burns=3,
            one_burn_grid_arrivals=6,
            max_revolutions=1,
            keep_candidates=30,
            event_step_s=8.0,
            two_burn_enabled=False,
        )
    )
    results: list[dict[str, object]] = []
    exceptions: list[dict[str, object]] = []
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="orbitwin-ood-") as temp:
        temp_path = Path(temp)
        for index in range(args.cases):
            values = sample_case(rng, config.policy.central_body_radius_km)
            path = temp_path / f"scene-{index:03d}.script"
            path.write_text(script_text(values), encoding="utf-8")
            try:
                scenario, _ = parse_gmat_script(path, config)
                candidates = search_one_burn(scenario, config)
                results.append(
                    {
                        "case": index,
                        "candidate_count": len(candidates),
                        "best_dv_km_s": candidates[0].total_dv_km_s if candidates else None,
                        "plane_delta_deg": abs(values["i1"] - values["i2"]),
                    }
                )
            except Exception as exc:
                exceptions.append({"case": index, "exception": repr(exc), "values": values})
    evidence = {
        "cases": args.cases,
        "seed": args.seed,
        "pipeline_exception_count": len(exceptions),
        "cases_with_legal_one_burn_candidate": sum(item["candidate_count"] > 0 for item in results),
        "results": results,
        "exceptions": exceptions,
        "elapsed_s": time.perf_counter() - start,
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 1 if exceptions else 0


if __name__ == "__main__":
    raise SystemExit(main())
