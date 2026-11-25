#!/usr/bin/env python3
"""Re-derive every claim about a burn table from the organizer's initial states.

Nothing here trusts the solver that produced the numbers. The trajectory is
replayed by independent propagation on a dense grid, and each rule in the
printed §2-§6 is a separate assertion against that replay, except where the
replay cannot answer it (below).

What this caught on 2026-09-05, none of which the solver reported:

  * A 99.996-point candidate that flew 2444 km *under the Atlantic*. Point-mass
    gravity has no surface, so an integrator will happily fly through the planet
    and report an interception on the far side. That is not a trajectory.
  * A refinement whose soft penalty was ridden to `|dv| = 1495.098 m/s`, over
    the cap it was supposed to respect.
  * Trajectories that clear the surface up to the interception and come back
    down afterwards -- legal under the printed rules, and worth knowing before
    submitting rather than after.

Usage:

    python scripts/audit_burn_table.py official.script burns.json --config config/x.yaml

`burns.json` is `{"burns": [{"time_s": ..., "dv_inertial_km_s": [x, y, z]}, ...]}`.
Exit status is 0 only if every judged check passes.

The replay is two-body. When the forces the script flies are richer, as under
the ranking rules' §7, two-body's interception time and score are not GMAT's,
so the interception and the score are printed but not judged: GMAT's report is
the evidence for those. Maneuver legality is judged as always, and the surface
and altitude walls are judged on the replay through T_max. The verdict line
names what was left unjudged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import assert_never

import numpy as np

from orbitwin.argtypes import positive_finite
from orbitwin.config import add_rules_argument, load_config, rulebook_refusal
from orbitwin.constraints import DV_COMPARISON_TOLERANCE_KM_S, interception_required
from orbitwin.errors import DynamicsError
from orbitwin.models import (
    Scenario,
    StateVector,
    head_start_jump,
    head_start_lead_s,
    pursuer_start,
)
from orbitwin.orbit import cartesian_to_keplerian, propagate_universal, specific_energy
from orbitwin.pinned import refuse_unpinned
from orbitwin.rules import DistanceTerm, InterceptionRules, interception_rules
from orbitwin.score import on_solver_clock
from orbitwin.state_probe import add_probe_arguments, load_scenario_for_tool


class Audit:
    def __init__(self) -> None:
        self.ok = True

    def check(self, label: str, passed: bool, detail: str) -> None:
        self.ok = self.ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {detail}")

    def unjudged(self, label: str, passed: bool, detail: str) -> None:
        self.note(label, f"{detail} (two-body {'pass' if passed else 'fail'}, not judged)")

    def note(self, label: str, detail: str) -> None:
        print(f"  [ -- ] {label}: {detail}")


def _logistic(points: float, exponent: float) -> float:
    if exponent > 700.0:
        return 0.0
    if exponent < -700.0:
        return points
    return points / (1.0 + math.exp(exponent))


def distance_term(distance_km: float, rules: InterceptionRules, cr_km: float | None) -> float:
    """Δr_min = max(d, 5): 50 e^{-(Δr_min-5)/100}, or 100/(1+e^{((Δr_min-5)/C_r)^2}) since
    the ranking rules' 2026/11/04 revision. Written here, not imported from score.py."""
    excess = max(distance_km, rules.intercept_radius_km) - rules.intercept_radius_km
    match rules.distance_term:
        case DistanceTerm.EXPONENTIAL:
            return 50.0 * math.exp(-excess / 100.0)
        case DistanceTerm.LOGISTIC:
            if cr_km is None:
                raise ValueError("the ranking rules' distance term needs C_r")
            u = excess / cr_km
            return _logistic(100.0, u * u)
        case _:
            assert_never(rules.distance_term)


def _replay(
    scenario: Scenario, burns: list[tuple[float, np.ndarray]], mu: float, end_s: float,
    step_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Dense independent replay: (times, pursuer radii, relative distances, the closest
    approach in a head start). Head-start burns (t < 0) fly from the probed state at the
    pursuer's epoch while the target waits at its own; their distances are inf here."""
    times: list[float] = []
    radii: list[float] = []
    distances: list[float] = []
    cursor, state = pursuer_start(scenario, [epoch for epoch, _ in burns])
    jump = head_start_jump(scenario, [epoch for epoch, _ in burns])
    waiting = scenario.target.initial_state.r
    alignment_km = math.inf
    boundaries: list[tuple[float, np.ndarray | None]] = [
        (epoch, dv) for epoch, dv in burns if epoch < 0.0]
    if boundaries:
        boundaries.append((0.0, None))
    boundaries += [(epoch, dv) for epoch, dv in burns if epoch >= 0.0] + [(end_s, None)]
    for boundary, dv in boundaries:
        span = boundary - cursor
        if span > 0.0:
            steps = max(2, int(span / step_s))
            # The alignment's end is sampled once, past the jump.
            ends_at_jump = jump is not None and boundary == 0.0 and dv is None
            for k in range(steps + (0 if ends_at_jump else 1)):
                offset = span * k / steps
                here = propagate_universal(state, offset, mu)
                times.append(cursor + offset)
                radii.append(float(np.linalg.norm(here.r)))
                if cursor + offset < 0.0:
                    alignment_km = min(alignment_km, float(np.linalg.norm(here.r - waiting)))
                    distances.append(math.inf)
                    continue
                there = propagate_universal(
                    scenario.target.initial_state, cursor + offset, mu
                )
                distances.append(float(np.linalg.norm(here.r - there.r)))
            state = propagate_universal(state, span, mu)
        if jump is not None and boundary == 0.0 and dv is None:
            # The alignment's end: GMAT's aligned state plus what the head start changed.
            state = StateVector(tuple(state.r + jump[0]), tuple(state.v + jump[1]))
            jump = None
        cursor = boundary
        if dv is not None:
            state = StateVector(tuple(state.r), tuple(state.v + dv))
    return np.asarray(times), np.asarray(radii), np.asarray(distances), alignment_km


def main() -> int:
    # No abbreviations: --cr would otherwise stand for --cr-km and drop the unit.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("script")
    parser.add_argument("burns")
    parser.add_argument("--config", required=True)
    add_rules_argument(parser)
    parser.add_argument("--step-s", type=float, default=0.25)
    parser.add_argument("--altitude-margin-km", type=float, default=100.0)
    parser.add_argument("--cr-km", type=positive_finite,
                        help="C_r in km for the ranking rules' distance term, if not the config's")
    parser.add_argument("--require-safe-through-horizon", action="store_true",
                        help="fail if the pursuer goes below the surface at any point up to T_max, "
                             "not just up to the interception; always so for a ranking miss or "
                             "a force model two-body leaves forces out of")
    add_probe_arguments(parser)
    args = parser.parse_args()
    refuse_unpinned()

    config = load_config(args.config)
    refusal = rulebook_refusal(config, args.config, args.rules)
    if refusal is not None:
        parser.error(refusal)
    rules = interception_rules(config.policy.rules)
    if args.cr_km is not None and not rules.distance_term.takes_cr:
        parser.error(f"--cr-km is the ranking rules' C_r; the {rules.name} rules' distance term "
                     "has none")
    cr_km = args.cr_km if args.cr_km is not None else (
        config.score.cr_km if config.score else None)
    scenario, _ = load_scenario_for_tool(
        parser, args.script, config, args.gmat_console, args.startup_file
    )
    mu = config.policy.mu_km3_s2
    body_km = config.policy.central_body_radius_km
    radius_km = config.policy.intercept_radius_km
    horizon_s = scenario.max_time_s

    data = json.loads(Path(args.burns).read_text(encoding="utf-8"))
    burns = sorted(
        ((float(entry["time_s"]), np.asarray(entry["dv_inertial_km_s"], dtype=float))
         for entry in data["burns"]),
        key=lambda item: item[0],
    )
    audit = Audit()

    print("=== 1. maneuver legality ===")
    for index, (_, dv) in enumerate(burns, 1):
        magnitude_km_s = float(np.linalg.norm(dv))
        limit_km_s = config.policy.dv_limit_km_s
        magnitude, limit = magnitude_km_s * 1000.0, limit_km_s * 1000.0
        audit.check(f"burn {index} magnitude",
                    magnitude_km_s <= limit_km_s + DV_COMPARISON_TOLERANCE_KM_S,
                    f"{magnitude:.4f} m/s <= {limit:.0f} (margin {limit - magnitude:.4f})")
    total_limit_km_s = config.policy.total_dv_limit_km_s
    if total_limit_km_s is not None:
        burned_km_s = sum(float(np.linalg.norm(dv)) for _, dv in burns)
        # No slack: over the total disqualifies (ranking §4 note 1(c)), as in constraints.py.
        audit.check("total delta-v", burned_km_s <= total_limit_km_s,
                    f"{burned_km_s:.12g} km/s <= {total_limit_km_s:.12g} km/s "
                    f"(margin {total_limit_km_s - burned_km_s:.3e})")
    for index in range(1, len(burns)):
        gap = burns[index][0] - burns[index - 1][0]
        audit.check(f"burns {index}-{index + 1} separation",
                    gap >= config.policy.min_burn_separation_s - 1e-9,
                    f"{gap:.6f} s >= {config.policy.min_burn_separation_s} s")
    if burns:
        # The earlier epoch, -lead, when policy.head_start_burns opens the head start.
        start_s = -head_start_lead_s(scenario)
        audit.check("first burn not before mission start", burns[0][0] >= start_s,
                    f"t1 = {burns[0][0]:.6f} s >= {start_s:.6f} s")
        if burns[0][0] < start_s:
            print("AUDIT: FAIL (a burn before the mission start cannot be replayed)")
            return 1
        audit.check("all burns inside the window", burns[-1][0] <= horizon_s,
                    f"last burn {burns[-1][0]:.6f} s <= {horizon_s:.6f} s"
                    + (" (T_max on the solver clock)" if scenario.time_origin_offset_s else ""))

    gaps = scenario.two_body_gaps
    if gaps:
        print()
        print("Two-body leaves out these forces the script flies:")
        for force in gaps:
            print(f"  {force}")
        # Measured on R1: a hit GMAT certified replays here as a 19.97 km miss.
        print("so the interception and the score below are not judged; judge them on GMAT's")
        print("report (scripts/check_report_rows.py for a hit, docs/RUNBOOK.md step 7 for a")
        print("scored miss). The walls are judged through T_max.")

    print()
    print("=== 2. interception ===" + (" (two-body, not judged)" if gaps else ""))
    times, radii, distances, alignment_km = _replay(scenario, burns, mu, horizon_s, args.step_s)
    inside = np.flatnonzero(distances <= radius_km)
    intercept_s = float(times[inside[0]]) if inside.size else None
    closest = int(distances.argmin())
    if gaps:
        audit.unjudged("enters the sphere", inside.size > 0,
                       f"first inside at t = {intercept_s:.4f} s" if inside.size
                       else f"never within {radius_km:g} km on a {args.step_s:g} s grid")
    elif not interception_required(scenario) and not inside.size:
        # The ranking rules score a miss on its closest approach (eq. 2, 3), so
        # a miss is an answer there, not a failure; only the walls are judged.
        audit.note("enters the sphere",
                   f"no: a miss, scored on its closest approach of "
                   f"{distances[closest]:.6g} km (ranking rules eq. 2, 3)")
    else:
        audit.check("enters the sphere", inside.size > 0,
                    f"first inside at t = {intercept_s:.4f} s" if inside.size
                    else f"never within {radius_km:g} km on a {args.step_s:g} s grid")
    audit.note("closest approach",
               f"{distances[closest]:.9e} km at t = {times[closest]:.4f} s")
    audit.note("separation at T_max", f"{distances[-1]:.4f} km")

    print()
    print("=== 3. geometry ===" + (" (two-body, judged through T_max)" if gaps else ""))
    floor_all = float(radii.min())
    audit.note("lowest radius over the whole run to T_max",
               f"{floor_all:.4f} km at t = {times[int(radii.argmin())]:.2f} s "
               f"(altitude {floor_all - body_km:.4f} km)")
    floor_before = float("inf")
    # A ranking miss flies to T_max. Under richer forces two-body's interception
    # time is not GMAT's, so the walls cover the whole run there too. Measured on
    # R1: two-body radius is within 0.28 km of GMAT's over the first 533 s, and
    # a dive two-body takes to 4199 km GMAT flew to 73.6 km altitude and aborted.
    whole_arc = bool(gaps) or (intercept_s is None and not interception_required(scenario))
    if whole_arc:
        audit.check("outside the central body through T_max", floor_all > body_km,
                    f"{floor_all:.4f} km vs surface {body_km:.4f} km")
        audit.check(f"{args.altitude_margin_km:g} km altitude margin through T_max",
                    floor_all > body_km + args.altitude_margin_km,
                    f"altitude {floor_all - body_km:.4f} km")
    elif intercept_s is not None:
        before = times <= intercept_s
        floor_before = float(radii[before].min())
        audit.check("outside the central body up to the interception", floor_before > body_km,
                    f"{floor_before:.4f} km vs surface {body_km:.4f} km")
        audit.check(f"{args.altitude_margin_km:g} km altitude margin up to the interception",
                    floor_before > body_km + args.altitude_margin_km,
                    f"altitude {floor_before - body_km:.4f} km")
    if not whole_arc:
        if args.require_safe_through_horizon:
            audit.check("never inside the central body, through T_max", floor_all > body_km,
                        f"{floor_all:.4f} km vs surface {body_km:.4f} km")
        elif floor_all <= body_km and intercept_s is not None and floor_before > body_km:
            audit.note("post-interception",
                       f"the pursuer reaches {floor_all:.1f} km after the interception. The "
                       "printed rules end the mission at Tteam and say nothing about this; "
                       "pass --require-safe-through-horizon to refuse it.")

    print()
    print("=== 4. dynamics ===" + (" (the two-body replay's own consistency)" if gaps else ""))
    cursor, state = pursuer_start(scenario, [epoch for epoch, _ in burns])
    jump = head_start_jump(scenario, [epoch for epoch, _ in burns])
    for index, (epoch, dv) in enumerate(burns, 1):
        if jump is not None and cursor < 0.0 <= epoch:
            state = propagate_universal(state, -cursor, mu)
            state, cursor = StateVector(tuple(state.r + jump[0]), tuple(state.v + jump[1])), 0.0
            jump = None
        pre = propagate_universal(state, epoch - cursor, mu)
        post = StateVector(tuple(pre.r), tuple(pre.v + dv))
        try:
            elements = cartesian_to_keplerian(post, mu)
            audit.note(f"after burn {index}",
                       f"a = {elements.sma_km:.3f} km, e = {elements.ecc:.9f}, "
                       f"i = {elements.inc_deg:.4f} deg, "
                       f"pericentre = {elements.sma_km * (1 - elements.ecc):.3f} km")
        except DynamicsError as exc:
            audit.note(f"after burn {index}", f"not an ellipse: {exc}")
        audit.check(f"burn {index} moves velocity only",
                    float(np.linalg.norm(post.r - pre.r)) == 0.0, "position unchanged")
        state, cursor = post, epoch
    if jump is not None:
        state = propagate_universal(state, -cursor, mu)
        state, cursor = StateVector(tuple(state.r + jump[0]), tuple(state.v + jump[1])), 0.0
    if cursor < horizon_s:
        end = propagate_universal(state, horizon_s - cursor, mu)
        drift = abs(specific_energy(end, mu) - specific_energy(state, mu))
        audit.check("energy conserved on the final coast", drift < 1e-9,
                    f"|dE| = {drift:.3e} km^2/s^2")

    total_dv = sum(float(np.linalg.norm(dv)) for _, dv in burns)
    print()
    print("=== 5. score ===")
    completion = intercept_s if intercept_s is not None else horizon_s
    # A miss scores Δr at T_max under the preliminary, the closest approach under
    # the ranking rules (eq. 2, 3).
    closest_scored = interception_rules(scenario.rules).miss_distance == "closest-approach"
    missed_by = float(distances.min() if closest_scored else distances[-1])
    if closest_scored and scenario.alignment_minimum_distance_km is not None:
        # closest_approach_window from_origin: the alignment's rows count too, this
        # replay's own where head-start burns move the pursuer through them.
        missed_by = min(missed_by, alignment_km if math.isfinite(alignment_km)
                        else scenario.alignment_minimum_distance_km)
    dr_min = max(radius_km if intercept_s is not None else missed_by, radius_km)
    if closest_scored and intercept_s is None and not gaps:
        # verify, certify and check_report_rows.py price it on GMAT's closest report row.
        print(f"  closest approach {missed_by:.6f} km from this replay's ~{args.step_s:g} s "
              "grid; the submission is scored on GMAT's closest report row")
    print(f"  total delta-V = {total_dv:.9f} km/s = {total_dv * 1000:.4f} m/s")
    t_team = (f"  T_team        = {completion + scenario.time_origin_offset_s:.6f} s"
              + (f" ({completion:.6f} s on the solver clock)"
                 if scenario.time_origin_offset_s else ""))
    if gaps:
        print("  T_team and the score: not judged, take them from GMAT's report")
        if args.cr_km is not None:
            print("  --cr-km not used: nothing is scored here")
    elif config.score is None:
        print(t_team)
        print("  no score block in the config; coefficients are not guessed")
        if args.cr_km is not None:
            print("  --cr-km not used: without the other coefficients nothing is scored")
    else:
        print(t_team)
        # T_team from policy.mission_time_origin, Δ before t = 0 when that craft aligns.
        clock = on_solver_clock(config.score, scenario)
        time_term = _logistic(25.0, clock.kt * (completion - clock.ct))
        distance_points = distance_term(dr_min, rules, cr_km)
        if cr_km is not None:
            print(f"  C_r           = {cr_km:g} km")
        for unit, value in (("km/s", total_dv), ("m/s", total_dv * 1000.0)):
            dv_term = _logistic(25.0, config.score.kv * (value - config.score.cv))
            marker = "  <- config" if unit == config.score.delta_v_unit else ""
            print(f"  Cv read as {unit:>4}: distance {distance_points:.4f} + time "
                  f"{time_term:.4f} + dV {dv_term:.4f} = "
                  f"{distance_points + time_term + dv_term:.6f}{marker}")

    print()
    scope = " (interception and score not judged: two-body leaves out forces)" if gaps else ""
    print("AUDIT:", ("PASS" if audit.ok else "FAIL") + scope)
    return 0 if audit.ok else 1


if __name__ == "__main__":
    sys.exit(main())
