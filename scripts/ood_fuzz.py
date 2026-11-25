#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitwin.lambert import solve_lambert_all
from orbitwin.orbit import (
    KeplerianElements,
    keplerian_to_cartesian,
    orbital_period_s,
    propagate_universal,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MU = 398600.4418
#: Each scale's offset from A1, days. TDB is moved off TT so a mix-up shows.
_SCALES = ("A1", "TAI", "UTC", "TT", "TDB")
_SCALE_DAYS = {"A1": 0.0, "TAI": -0.0343817 / 86400, "UTC": -37.0343817 / 86400,
               "TT": 32.1496183 / 86400, "TDB": 32.1512183 / 86400}


def alignment_cases(
    rng: np.random.Generator, count: int
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Random alignment lines against random probed epochs, judged by the rules of
    docs/ASYNC_START_AND_INSERTION.md §2 as they read, and by the reason refused.

    The template maneuvers, so a single-craft segment that is not an alignment is
    refused for not holding the craft together, as a template's would be.
    """
    import dataclasses
    import tempfile

    from orbitwin.config import OrbitWinConfig, PolicyConfig
    from orbitwin.errors import ParseError
    from orbitwin.gmat_parser import GmatScriptDocument, parse_gmat_script
    from orbitwin.models import (
        AlignedStates,
        GmatRunResult,
        ProbedCraft,
        ProbedEpoch,
        StateProbe,
        StateVector,
        alignment_binding_sha256,
        spacecraft_binding_sha256,
    )
    from orbitwin.state_probe import _resource_section

    template = (PROJECT_ROOT / "examples" / "two_burn_rendezvous.script").read_text(
        encoding="utf-8"
    )
    head, _, mission = template.partition("BeginMissionSequence;\n")
    config = OrbitWinConfig(
        policy=dataclasses.replace(
            PolicyConfig(), horizon_s=200000.0, mission_time_origin="pursuer",
            closest_approach_window="after_alignment",
        ),
    )
    earliest = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, mission_time_origin="earliest")
    )
    run = GmatRunResult("fuzz", True, 0, "", 0.0, "", "", "", "", None, "", "", None, {}, {})
    states = {
        "HumanSC": StateVector((7000.0, 0.0, 0.0), (0.0, 7.546, 0.0)),
        "AlienSC": StateVector((0.0, 9000.0, 0.0), (-6.655, 0.0, 0.0)),
    }
    failures: list[dict[str, object]] = []
    outcomes = {"accepted": 0, "refused": 0}
    with tempfile.TemporaryDirectory(prefix="orbitwin-fuzz-") as scratch:
        path = Path(scratch) / "case.script"
        for case in range(count):
            band = int(rng.integers(6))
            early_a1 = 31479.5 + float(rng.uniform(0.0, 1.0))
            if band == 0:
                late_a1 = early_a1
            elif band == 1:  # 1 to 4 double steps: three are one start, four are apart
                late_a1 = early_a1
                for _ in range(int(rng.integers(1, 5))):
                    late_a1 = float(np.nextafter(late_a1, np.inf))
            else:
                delta = (float(rng.uniform(0.4e-6, 0.9e-6)), float(rng.uniform(1.1e-6, 10.0)),
                         float(rng.uniform(10.0, 1e5)), float(rng.uniform(0.9e-6, 2e-6)))[band - 2]
                late_a1 = early_a1 + delta / 86400.0
            actual_early, actual_late = (("AlienSC", "HumanSC") if rng.random() < 0.5
                                         else ("HumanSC", "AlienSC"))
            a1 = {actual_early: early_a1, actual_late: late_a1}
            flies = actual_early if rng.random() < 0.8 else actual_late
            other = "HumanSC" if flies == "AlienSC" else "AlienSC"
            offset_s = (a1[other] - a1[flies]) * 86400.0
            form = ("equality", "value", "elapsed")[int(rng.integers(3))]
            scale = _SCALES[int(rng.integers(5))]
            error_s = (0.0, 0.4e-6, -0.4e-6, 0.9e-6, -0.9e-6, 1.1e-6, -1.1e-6, 3e-6, -3e-6)[
                int(rng.integers(9))
            ]
            faults: list[str] = []
            if form == "equality":
                other_scale = scale if rng.random() < 0.9 else _SCALES[(_SCALES.index(scale) + 1) % 5]
                other_name = other if rng.random() < 0.9 else flies
                goal = f"{flies}.{scale}ModJulian = {other_name}.{other_scale}ModJulian"
                if other_name != other:
                    faults.append("not the other spacecraft")
                if other_scale != scale:
                    faults.append("one scale")
            elif form == "value":
                wanted = a1[other] + _SCALE_DAYS[scale]
                # In double steps: three (0.94 µs) are within 1e-6 s, four (1.26 µs) are not.
                steps = int(rng.integers(-5, 6))
                value = wanted
                for _ in range(abs(steps)):
                    value = float(np.nextafter(value, np.inf if steps > 0 else -np.inf))
                goal = f"{flies}.{scale}ModJulian = {value!r}"
                # Within 1e-6 s of the later epoch, and the epoch itself at one start.
                at_start = offset_s == 0.0
                if abs(value - wanted) * 86400.0 > 1e-6 or (at_start and value != wanted):
                    faults.append("probed epoch")
            else:
                days = rng.random() < 0.3
                amount = (offset_s + error_s) / (86400.0 if days else 1.0)
                goal = f"{flies}.{'ElapsedDays' if days else 'ElapsedSecs'} = {amount!r}"
            if form == "elapsed" and abs(offset_s) <= 1e-6:
                # A segment, not an alignment: one too many, flying one craft.
                faults += ["cannot keep them synchronized", "must alternate"]
            elif offset_s < 0.0:
                faults.append("never reaches" if form != "elapsed" else "further apart")
            elif 0.0 < offset_s <= 1e-6:
                faults.append("ambiguous")
            elif form == "elapsed" and abs(error_s) > 1e-6:
                faults.append("but it starts")
            line = f"Propagate DefaultProp({flies}) {{{goal}}};\n"
            path.write_text(head + "BeginMissionSequence;\n" + line + mission, encoding="utf-8")
            _, _, fields, created = _resource_section(GmatScriptDocument.read(path))
            bindings = {name: spacecraft_binding_sha256(name, fields, created) for name in states}
            force = {
                f"{name}.{key}": value
                for name, kind in created.items() if kind == "forcemodel"
                for key, value in fields.get(name, {}).items()
            }
            props = {
                f"{name}.{key}": value
                for name, kind in created.items() if kind == "propagator"
                for key, value in fields.get(name, {}).items()
            }
            probe = StateProbe(
                craft=tuple(
                    ProbedCraft(
                        name,
                        ProbedEpoch(**{
                            f"{item.lower()}_mjd": a1[name] + _SCALE_DAYS[item] for item in _SCALES
                        }),
                        states[name], bindings[name],
                    )
                    for name in states
                ),
                run=run,
                aligned=AlignedStates(
                    states=dict(states), a1_mjd=late_a1, early_minimum_radius_km=7000.0,
                    minimum_distance_km=1000.0,
                    binding_sha256=alignment_binding_sha256(
                        line, [bindings["HumanSC"], bindings["AlienSC"]], force, props
                    ),
                    run=run,
                ),
            )
            reason = ""
            try:
                scenario, _ = parse_gmat_script(path, config, probe=probe)
            except ParseError as exc:
                reason = str(exc)
                accepted = False
            else:
                accepted = scenario.alignment is not None
                if not accepted:
                    reason = "parsed with no alignment"
                elif (scenario.alignment.early != flies  # type: ignore[union-attr]
                      or abs(scenario.alignment.offset_s - offset_s) > 1e-9):  # type: ignore[union-attr]
                    failures.append({"alignment_case": case, "line": line.strip(),
                                     "reason": "accepted with the wrong early craft or offset"})
                else:
                    # The organizer's origin: the craft with the earlier probed epoch.
                    resolved, _ = parse_gmat_script(path, earliest, probe=probe)
                    started = flies if offset_s > 0.0 else "HumanSC"
                    if (resolved.time_origin_craft != started
                            or abs(resolved.time_origin_offset_s - offset_s) > 1e-9):
                        failures.append({"alignment_case": case, "line": line.strip(),
                                         "reason": "earliest resolved to "
                                         f"{resolved.time_origin_craft}, not {started}"})
            outcomes["accepted" if accepted else "refused"] += 1
            expected = not faults
            explained = accepted or any(fault in reason for fault in faults)
            if accepted != expected or not explained:
                failures.append({"alignment_case": case, "line": line.strip(),
                                 "offset_s": offset_s, "expected": faults or "accepted",
                                 "parser": reason[:200] or "accepted"})
    return failures, outcomes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", default="output/ood_fuzz.json")
    parser.add_argument(
        "--alignment-cases", type=int, default=300,
        help="random epoch alignments against random probed epochs (0 skips)",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    failures: list[dict[str, object]] = []
    branch_counts: dict[str, int] = {}
    start = time.perf_counter()
    worst_error = 0.0
    for case_index in range(args.cases):
        sma = float(rng.uniform(6700.0, 30000.0))
        ecc = float(rng.uniform(0.0, 0.65))
        inc = float(rng.uniform(0.0, 179.0))
        raan = float(rng.uniform(0.0, 360.0))
        aop = float(rng.uniform(0.0, 360.0))
        ta = float(rng.uniform(0.0, 360.0))
        initial = keplerian_to_cartesian(KeplerianElements(sma, ecc, inc, raan, aop, ta), MU)
        period = orbital_period_s(sma, MU)
        period_fraction = float(rng.uniform(0.08, 2.6))
        tof = max(30.0, period_fraction * period)
        arrival = propagate_universal(initial, tof, MU)
        max_revolutions = min(2, max(0, math.floor(period_fraction)))
        try:
            solutions = solve_lambert_all(
                initial.r,
                arrival.r,
                tof,
                MU,
                max_revolutions=max_revolutions,
                samples_per_interval=72,
                endpoint_tolerance_km=0.02,
                reference_normal=np.cross(initial.r, initial.v),
            )
        except Exception as exc:  # Captured as reproducible validation evidence.
            failures.append({"case": case_index, "exception": repr(exc)})
            continue
        if not solutions:
            failures.append(
                {
                    "case": case_index,
                    "reason": "no validated Lambert solution",
                    "elements": [sma, ecc, inc, raan, aop, ta],
                    "tof_s": tof,
                    "max_revolutions": max_revolutions,
                }
            )
            continue
        error = min(solution.endpoint_error_km for solution in solutions)
        worst_error = max(worst_error, error)
        for solution in solutions:
            branch_counts[solution.branch] = branch_counts.get(solution.branch, 0) + 1
    alignment_failures, alignment_outcomes = alignment_cases(rng, args.alignment_cases)
    failures.extend(alignment_failures)
    elapsed = time.perf_counter() - start
    evidence = {
        "alignment_cases": args.alignment_cases,
        "alignment_outcomes": alignment_outcomes,
        "cases": args.cases,
        "seed": args.seed,
        "failures": failures,
        "failure_count": len(failures),
        "worst_best_endpoint_error_km": worst_error,
        "branch_counts": branch_counts,
        "elapsed_s": elapsed,
        "python": sys.version,
        "numpy": np.__version__,
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
