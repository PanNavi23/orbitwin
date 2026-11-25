#!/usr/bin/env python3
"""Map the cheapest Earth-safe single impulse against arrival time.

This is the first thing to run once the scoring coefficients are announced, and
it answers the question the Pareto frontier cannot: *which arrival times can be
bought at all, and for how many impulses.*

For every arrival time it sweeps the burn epoch, solves the 3-D Lambert transfer,
takes the cheapest impulse whose coasting arc stays above the central body, and
prints the delta-v that arrival costs, the burns needed to pay for it under the
per-impulse cap, and the score the announced coefficients give it.

**Read the delta-v column as an upper bound, not a floor.** One impulse is the
cheapest way to reach an arrival only when nothing else is free to move; several
impulses spread over the transfer routinely beat it, and here that gap is large.
Measured on the same scenario: this table wants 8.34 km/s for an arrival near
5700 s, and the five-impulse solution that was actually flown reached 5675 s on
6.19 km/s -- 2.15 km/s cheaper than the "floor" the single-impulse sweep reports.
So a column reading 8 km/s does not mean that arrival costs 8 km/s; it means it
is reachable, six impulses would pay for it outright, and a multi-impulse search
seeded there will probably do better. The score column inherits the same bias
and understates what the arrival is worth.

What the table is for is the shape: where the reachable arrivals are, which ones
the score curve rewards, and roughly how many impulses to budget. Take the best
few rows and hand them to a multi-impulse refinement; do not take the numbers as
the answer.

Measured 2026-09-05, where this mattered: the announced kt/Ct made the time term
a cliff at 6505 s while the delta-v term still paid 24 points at 6 km/s. Impulses
were nearly free and time was everything -- the opposite of the "cheapest
delta-v, one or two burns" default. The built-in sweep's best was 21558 s for
74.997 points; this table pointed at 5675 s, which is 98.32 with five burns.

The `--through-horizon` flag decides how much of the trajectory has to clear the
surface: by default only the arc up to the arrival, which is what 初賽規則 §4
scores; with the flag, the whole run to T_max, because point-mass gravity has no
surface and an optimiser will happily buy speed with a dive that comes back down
after the interception.

Under the ranking rules one impulse over 250 m/s disqualifies (§4 note 1(a)), so
almost every row is a single impulse no submission can fly. Those rows are
marked 'over burn cap', no score is claimed as a bound, the forces of §7 that
two-body leaves out are listed, and the times are on the solver clock, whose
t = 0 is the later craft's epoch when the two start apart.

    python scripts/safe_delta_v_floor.py official.script --config config/x.yaml \\
        --arrivals 1200:9000:50 --burns 0:8000:15
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitwin.config import add_rules_argument, load_config, rulebook_refusal
from orbitwin.constraints import DV_COMPARISON_TOLERANCE_KM_S
from orbitwin.errors import DynamicsError
from orbitwin.lambert import solve_lambert_all
from orbitwin.mission_clock import clock_lines
from orbitwin.models import StateVector
from orbitwin.multi_burn import minimum_radius_km
from orbitwin.orbit import propagate_universal
from orbitwin.pinned import refuse_unpinned
from orbitwin.rules import interception_rules
from orbitwin.score import on_solver_clock
from orbitwin.state_probe import add_probe_arguments, load_scenario_for_tool

_STATE: dict[str, object] = {}


def _grid(spec: str) -> np.ndarray:
    start, stop, step = (float(part) for part in spec.split(":"))
    return np.arange(start, stop, step)


def _logistic(points: float, exponent: float) -> float:
    if exponent > 700.0:
        return 0.0
    if exponent < -700.0:
        return points
    return points / (1.0 + math.exp(exponent))


def _arc_floor(state: StateVector, duration_s: float, mu: float) -> float:
    """Lowest geocentric radius this coasting arc actually reaches.

    The search's own closed form. This file kept a copy whose open arcs were
    still sampled at 40 points, and listed arcs 1100 to 1900 km underground as
    safe once --through-horizon stretched the samples to 580 s apart.
    """
    try:
        return minimum_radius_km(state, duration_s, mu)
    except DynamicsError:
        # Radial or parabolic to rounding: no conic to reason with. Refuse the
        # arc rather than call it safe.
        return 0.0


def _row(arrival_s: float, burn_grid: np.ndarray) -> dict[str, object] | None:
    scenario = _STATE["scenario"]
    config = _STATE["config"]
    mu = config.policy.mu_km3_s2
    floor_km = float(_STATE["floor_km"])
    horizon = float(_STATE["arc_end"]) if _STATE["arc_end"] else arrival_s
    goal = propagate_universal(scenario.target.initial_state, arrival_s, mu).r
    best: tuple[float, float, float, list[float]] | None = None
    for burn_s in burn_grid:
        tof = arrival_s - float(burn_s)
        if tof <= 90.0:
            continue
        # The coast to the burn, too: a start may dive before it.
        before = _arc_floor(scenario.pursuer.initial_state, float(burn_s), mu)
        if before < floor_km:
            continue
        start = propagate_universal(scenario.pursuer.initial_state, float(burn_s), mu)
        try:
            solutions = solve_lambert_all(
                start.r, goal, tof, mu,
                max_revolutions=config.search.max_revolutions,
                samples_per_interval=64,
            )
        except Exception:
            continue
        for solution in solutions:
            if solution.endpoint_error_km > 0.3:
                continue
            dv = np.asarray(solution.v1_km_s) - start.v
            cost = float(np.linalg.norm(dv))
            if best is not None and cost >= best[0]:
                continue
            after = StateVector(tuple(start.r), tuple(start.v + dv))
            span = (horizon if horizon > arrival_s else arrival_s) - float(burn_s)
            if _arc_floor(after, span, mu) < floor_km:
                continue
            low = min(before, _arc_floor(after, span, mu))
            best = (cost, float(burn_s), low, [float(v) for v in dv])
    if best is None:
        return None
    cost, burn_s, arc_low, dv = best
    score_config = config.score
    row: dict[str, object] = {
        "arrival_s": arrival_s, "dv_km_s": cost, "burn_s": burn_s,
        "arc_floor_km": arc_low,
        "burns_needed": max(1, math.ceil(cost / config.policy.dv_safe_limit_km_s)),
        "dv_inertial_km_s": dv,
    }
    if score_config is not None:
        clock = on_solver_clock(score_config, scenario)
        time_term = _logistic(25.0, clock.kt * (arrival_s - clock.ct))
        dv_value = cost if score_config.delta_v_unit == "km/s" else cost * 1000.0
        dv_term = _logistic(25.0, score_config.kv * (dv_value - score_config.cv))
        row |= {"time_points": time_term, "dv_points": dv_term,
                "score": 50.0 + time_term + dv_term}
    return row


def total_cap(policy) -> tuple[float, str] | None:
    """The total delta-v the table is held to, and the policy field it came from."""
    if policy.total_dv_safe_limit_km_s is not None:
        return policy.total_dv_safe_limit_km_s, "policy.total_dv_safe_limit_km_s"
    if policy.total_dv_limit_km_s is not None:
        return policy.total_dv_limit_km_s, "policy.total_dv_limit_km_s"
    return None


def over_total(row: dict[str, object], cap_km_s: float | None) -> bool:
    return cap_km_s is not None and float(row["dv_km_s"]) > cap_km_s


def burn_caps(config) -> tuple[float, float] | None:
    """(formal, ours) per-impulse caps in km/s when going over the formal one disqualifies."""
    if interception_rules(config.policy.rules).over_impulse_limit != "disqualify":
        return None
    return config.policy.dv_limit_km_s, config.policy.dv_safe_limit_km_s


def burn_flag(row: dict[str, object], caps: tuple[float, float] | None) -> str | None:
    """'over burn cap' for a single impulse the rules disqualify, 'over burn margin' for ours."""
    if caps is None:
        return None
    cost = float(row["dv_km_s"])
    if cost > caps[0] + DV_COMPARISON_TOLERANCE_KM_S:
        return "over burn cap"
    if cost > caps[1] + DV_COMPARISON_TOLERANCE_KM_S:
        return "over burn margin"
    return None


def table_line(
    row: dict[str, object], cap_km_s: float | None, caps: tuple[float, float] | None = None
) -> str:
    score = row.get("score")
    line = (f"{row['arrival_s']:>10.0f} {row['dv_km_s']:>9.4f} {row['burn_s']:>9.1f} "
            f"{row['arc_floor_km']:>11.1f} {row['burns_needed']:>6d} "
            f"{row.get('time_points', float('nan')):>7.3f} "
            f"{row.get('dv_points', float('nan')):>7.3f} "
            f"{score if score is not None else float('nan'):>9.4f}")
    flag = burn_flag(row, caps)
    if flag is not None:
        line += f"  {flag}"
    return line + "  over total" if over_total(row, cap_km_s) else line


def best_lines(
    ranked: list[dict[str, object]], cap_km_s: float | None,
    caps: tuple[float, float] | None = None,
) -> list[str]:
    """The seeding summary for rows already sorted best first."""
    if caps is not None:
        return _capped_best_lines(ranked, cap_km_s, caps)
    lines = [
        "best arrivals by score -- SEED A MULTI-IMPULSE SEARCH HERE, do not stop at these numbers.",
        "The delta-v is a single impulse's, which is an upper bound: on the 2026-09-05",
        "scenario the flown five-impulse solution beat this column by 2.15 km/s, so the",
        "scores below understate what these arrivals are worth.",
    ]
    within = [row for row in ranked if not over_total(row, cap_km_s)]
    if len(within) < len(ranked):
        lines.append(f"Left out {len(ranked) - len(within)} arrival(s) whose single impulse "
                     f"exceeds the {cap_km_s:g} km/s total.")
    lines += [f"  arrive {row['arrival_s']:8.0f} s   dV <= {row['dv_km_s']:7.4f} km/s   "
              f"<= {row['burns_needed']} burn(s)   burn@ {row['burn_s']:7.1f} s   "
              f"score >= {row['score']:.4f}" for row in within[:8]]
    if not within:
        lines.append("  none fits the total as a single impulse; a multi-impulse search may still.")
    return lines


def _capped_best_lines(
    ranked: list[dict[str, object]], cap_km_s: float | None, caps: tuple[float, float]
) -> list[str]:
    """best_lines when one impulse over the cap disqualifies: no score is a bound."""
    lines = [
        "best arrivals by two-body score -- SEED A MULTI-IMPULSE SEARCH HERE, do not stop at "
        "these numbers.",
        f"One impulse over {caps[0] * 1000:g} m/s disqualifies: a row 'over burn cap' is no "
        "plan anyone can fly.",
        f"Its burn count is how many {caps[1] * 1000:g} m/s impulses add up to the same "
        "delta-v; a split need not cost that.",
        "No score here is a bound: each is two-body's, for one impulse.",
    ]
    within = [row for row in ranked if not over_total(row, cap_km_s)]
    if len(within) < len(ranked):
        lines.append(f"Left out {len(ranked) - len(within)} arrival(s) whose single impulse "
                     f"exceeds the {cap_km_s:g} km/s total.")
    for row in within[:8]:
        flag = burn_flag(row, caps)
        head = (f"  arrive {row['arrival_s']:8.0f} s   one impulse {row['dv_km_s']:7.4f} km/s   "
                f"burn@ {row['burn_s']:7.1f} s   ")
        lines.append(head + (
            f"{flag.upper()}: {row['burns_needed']} burn(s) at {caps[1] * 1000:g} m/s"
            if flag is not None
            else f"two-body score {row['score']:.4f}, within the cap as one burn"
        ))
    if not within:
        lines.append("  none fits the total as a single impulse; a multi-impulse search may still.")
    return lines


def _init(scenario, config, floor_km, arc_end) -> None:
    _STATE.update(scenario=scenario, config=config, floor_km=floor_km, arc_end=arc_end)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("script")
    parser.add_argument("--config", required=True)
    add_rules_argument(parser)
    parser.add_argument("--arrivals", default="1200:12000:50", help="start:stop:step in seconds")
    parser.add_argument("--burns", default="0:9000:15", help="start:stop:step in seconds")
    parser.add_argument("--altitude-margin-km", type=float, default=100.0,
                        help="how far above the central body every arc must stay")
    parser.add_argument("--through-horizon", action="store_true",
                        help="require the arc to clear the surface for the whole run to T_max, "
                             "not just up to the arrival")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output")
    add_probe_arguments(parser)
    args = parser.parse_args()
    refuse_unpinned()

    config = load_config(args.config)
    refusal = rulebook_refusal(config, args.config, args.rules)
    if refusal is not None:
        parser.error(refusal)
    scenario, _ = load_scenario_for_tool(
        parser, args.script, config, args.gmat_console, args.startup_file
    )
    floor_km = config.policy.central_body_radius_km + args.altitude_margin_km
    arrivals = _grid(args.arrivals)
    arrivals = arrivals[arrivals <= scenario.max_time_s]
    burn_grid = _grid(args.burns)
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    arc_end = scenario.max_time_s if args.through_horizon else 0.0

    period = (
        f"T_A = {scenario.target_period_s:.4f} s"
        if scenario.target_period_s is not None
        else "target on an open orbit, no T_A"
    )
    caps = burn_caps(config)
    clock = " on the solver clock, as every time below" if caps is not None else ""
    print(f"{period}   T_max = {scenario.max_time_s:.4f} s{clock}")
    if caps is not None:
        for line in clock_lines(scenario, config.policy):
            print(line)
        clause = interception_rules(config.policy.rules).impulse_clause
        print(f"burn cap = {caps[0] * 1000:g} m/s ({clause}, disqualifies), ours "
              f"{caps[1] * 1000:g} m/s: a single impulse "
              "over either is marked 'over burn cap' or 'over burn margin'")
        if scenario.two_body_gaps:
            print("Two-body leaves out these forces the script flies (§7), so every delta-v, "
                  "arrival and score below is two-body's, not GMAT's:")
            for gap in scenario.two_body_gaps:
                print(f"  {gap}")
    print(f"floor = {floor_km:.1f} km   arcs checked to "
          f"{'T_max' if args.through_horizon else 'the arrival'}   workers = {workers}")
    cap = total_cap(config.policy)
    cap_km_s = cap[0] if cap is not None else None
    if cap is not None:
        print(f"total cap = {cap[0]:g} km/s from {cap[1]}: rows needing more are marked "
              "'over total' and left out of the seeds")
    if config.score is None:
        print("no score block in the config: delta-v and burn counts only, no ranking")
    print()
    print(f"{'arrive(s)':>10} {'dV km/s':>9} {'burn@(s)':>9} {'arc low km':>11} "
          f"{'burns':>6} {'t pts':>7} {'dV pts':>7} {'SCORE':>9}")

    with Pool(workers, initializer=_init,
              initargs=(scenario, config, floor_km, arc_end)) as pool:
        rows = pool.map(partial(_row, burn_grid=burn_grid), arrivals, chunksize=2)
    rows = [row for row in rows if row is not None]
    for row in rows:
        print(table_line(row, cap_km_s, caps))
        if cap_km_s is not None:
            row["over_total_cap"] = over_total(row, cap_km_s)
        if caps is not None:
            row["over_burn_cap"] = burn_flag(row, caps) == "over burn cap"
            row["over_burn_margin"] = burn_flag(row, caps) is not None

    if not rows:
        print("\nNo arrival in this window has an Earth-safe single impulse. Widen --arrivals, "
              "lower --altitude-margin-km, or the geometry needs more than one burn everywhere.")
        return 1

    if config.score is not None:
        rows.sort(key=lambda row: -float(row["score"]))
        print()
        print("\n".join(best_lines(rows, cap_km_s, caps)))
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\nwritten {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
