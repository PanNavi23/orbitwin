#!/usr/bin/env python3
"""Run solve on the R1-R9 rehearsal scenarios with real GMAT; the step-4 table.

    python scripts/rehearse_ranking.py --output run/rehearsal [--both-margins] [R3 R6] \
        [--cr-km 100 300 1000 3000]

R6-R9 start the craft on different epochs (examples/ranking/async_scenarios.yaml);
their rows also carry the policy the asynchronous start needs.

Each scenario gets config/ranking-rehearsal.yaml with its own T_max, the
ranking search and correction on, verify_top 8, the profile's gmat.workers
and placeholder coefficients (kt 0.003 /s, Ct 0.8 T_max, kv 0.0012 per m/s, Cv
2000 m/s, C_r 300 km). The placeholders are not the day's: the table shows the
pipeline, not a score. --cr-km runs each scenario once per C_r given.
Needs $GMAT_CONSOLE and $GMAT_STARTUP.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitwin.argtypes import positive_finite, positive_int
from orbitwin.config import load_config
from orbitwin.io import prepare_empty_output, read_candidates
from orbitwin.pipeline import solve_scenario
from orbitwin.verify import certify_candidate

ROOT = Path(__file__).resolve().parents[1]
#: PLACEHOLDER, not announced: C_r comes on the day (rules 2026/11/04, §4 eq. (5)).
PLACEHOLDER_CR_KM = 300.0


def _cr_label(cr_km: float) -> str:
    return f"cr{cr_km:g}"


def main() -> int:
    # No abbreviations: --cr would otherwise stand for --cr-km and drop the unit.
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0],
                                     allow_abbrev=False)
    parser.add_argument("names", nargs="*", default=[f"R{index}" for index in range(1, 10)])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--both-margins", action="store_true", help="also search the 0 km force-model margin"
    )
    parser.add_argument(
        "--certify", action="store_true", help="certify each scenario's best candidate too"
    )
    parser.add_argument(
        "--team-number", type=positive_int,
        help="the team number for the official report's file name; --certify needs it",
    )
    parser.add_argument(
        "--cr-km", type=positive_finite, nargs="+", default=[PLACEHOLDER_CR_KM], metavar="KM",
        help=f"placeholder C_r values in km to run each scenario under (default "
        f"{PLACEHOLDER_CR_KM:g})",
    )
    args = parser.parse_args()
    # The run's directory and config are named by this, so two values must not share it.
    labels = [_cr_label(value) for value in args.cr_km]
    if len(set(labels)) != len(labels):
        parser.error(f"--cr-km values must differ in their run names: {labels}")
    if args.certify and args.team_number is None:
        parser.error("--certify needs --team-number: certify refuses an official report "
                     "without the team's number under the ranking rules")
    console, startup = os.environ.get("GMAT_CONSOLE"), os.environ.get("GMAT_STARTUP")
    if not console or not startup:
        parser.error("set GMAT_CONSOLE and GMAT_STARTUP")
    prepare_empty_output(args.output, "rehearsal")
    base = yaml.safe_load((ROOT / "config" / "ranking-rehearsal.yaml").read_text(encoding="utf-8"))
    if args.team_number is not None:
        base["policy"]["team_number"] = args.team_number
    rows = [
        row
        for table in ("scenarios.yaml", "async_scenarios.yaml")
        for row in yaml.safe_load(
            (ROOT / "examples" / "ranking" / table).read_text(encoding="utf-8")
        )["scenarios"]
    ]
    unknown = sorted(set(args.names) - {row["script"].split("_")[0] for row in rows})
    if unknown:
        parser.error(f"no rehearsal named {', '.join(unknown)}")
    results = []
    refused = False
    jobs = [(row, cr_km) for row in rows if row["script"].split("_")[0] in args.names
            for cr_km in args.cr_km]
    for row, cr_km in jobs:
        scenario = row["script"].split("_")[0]
        name = scenario if len(args.cr_km) == 1 else f"{scenario}-{_cr_label(cr_km)}"
        horizon = float(row["horizon_s"])
        config = json.loads(json.dumps(base))
        config["policy"]["horizon_s"] = horizon
        config["policy"].update(row.get("policy", {}))
        config["search"].update(
            ranking_search=True, ranking_search_without_margin=args.both_margins
        )
        config["gmat"].update(verify_top=8, correct_top=40)
        config["score"] = {
            "kt": 0.003, "ct": 0.8 * horizon, "kv": 0.0012, "cv": 2000.0,
            "time_unit": "s", "delta_v_unit": "m/s", "cr_km": cr_km,
        }
        path = args.output / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        began = time.perf_counter()
        summary = solve_scenario(
            ROOT / "examples" / "ranking" / row["script"], load_config(path),
            args.output / name, gmat_console=console, startup_file=startup,
        )
        best = summary["best_candidate"] or {}
        corrections = summary["gmat_corrections"]
        result = {
            "scenario": scenario,
            "cr_km": cr_km,
            "wall_s": round(time.perf_counter() - began, 1),
            "best_id": best.get("candidate_id"),
            "best_branch": best.get("branch"),
            "best_stage": summary["best_candidate_stage"],
            "best_score": best.get("score"),
            "corrections": len(corrections),
            "converged": sum(record["converged"] for record in corrections),
            "correction_runs": sum(record["gmat_runs"] or 0 for record in corrections),
            "verified": summary["gmat_verified_count"],
        }
        if args.certify and not best.get("candidate_id"):
            result["certified"] = "nothing to certify"
        elif args.certify:
            try:
                candidate = next(
                    item for item in read_candidates(args.output / name / "candidates.json")
                    if item.candidate_id == best["candidate_id"]
                )
                certificate = certify_candidate(
                    ROOT / "examples" / "ranking" / row["script"], candidate, load_config(path),
                    console, startup, args.output / f"{name}-certify",
                )
            except Exception as exc:  # recorded in results.json; the others still run
                result["certified"] = f"refused: {type(exc).__name__}: {exc}"[:300]
                refused = True
                results.append(result)
                print(json.dumps(result), flush=True)
                (args.output / "results.json").write_text(
                    json.dumps(results, indent=1), encoding="utf-8"
                )
                continue
            result.update(
                certified=certificate["conferred_stage"],
                frozen_rows=certificate["frozen_rows"],
                replayed_score=certificate["final_script_parity"]["replayed_score"],
                t_team_s=certificate["clock"]["t_team_s"],
            )
        results.append(result)
        print(json.dumps(result), flush=True)
        # After each scenario, so a later failure keeps what ran.
        (args.output / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
