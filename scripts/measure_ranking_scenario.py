#!/usr/bin/env python3
"""Measure a ranking rehearsal scenario on stock GMAT.

    PYTHONPATH=src python scripts/measure_ranking_scenario.py \
        examples/ranking/R1_NearPlane_Reachable.script --boundary-km 17000 \
        --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" \
        --output output/measure-R1

Prints the entry examples/ranking/scenarios.yaml keeps for the script, plus
drift and wall time as comments.

T_max is when SC_Alien, past perigee, first climbs back to the mission boundary
(ranking rules 2026/10/06 sec. 3). It is read off a zero-burn GMAT run that goes
`--beyond-s` past the script's own Propagate goal, with the target's radius
interpolated between report rows by cubic Hermite on r and dr/dt. The script is
then run once as written, timed, and once more with the diagnostic report, which
gives the zero-burn closest approach over [0, goal] and SC_Human's position on
the last row. The script's goal must equal the measured T_max to within
GOAL_TOLERANCE_S unless --provisional is given.

Drift is the two-body propagation of each craft's initial state, the model the
search plans in, against GMAT's rows at 30 and 60 minutes.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitwin.config import OrbitWinConfig, load_config
from orbitwin.diagnostic import add_diagnostic_report
from orbitwin.errors import GmatRunError, PolicyError
from orbitwin.gmat_parser import (
    AlignedStatesNeeded,
    GmatScriptDocument,
    parse_gmat_script,
)
from orbitwin.gmat_runner import run_gmat
from orbitwin.io import prepare_empty_output
from orbitwin.models import Scenario, StateProbe
from orbitwin.orbit import cartesian_to_keplerian, propagate_universal
from orbitwin.report_parser import evaluate_diagnostic_report, parse_numeric_report
from orbitwin.state_probe import probe_initial_states
from orbitwin.verify import written_reports

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "ranking-rehearsal.yaml"
#: The goal is written to the millisecond; 5 ms is at most 40 m of radius at 8 km/s.
GOAL_TOLERANCE_S = 5.0e-3


def _hermite(t0: float, t1: float, y0: float, y1: float, d0: float, d1: float, t: float) -> float:
    h = t1 - t0
    s = (t - t0) / h
    return (
        (2 * s**3 - 3 * s**2 + 1) * y0
        + (s**3 - 2 * s**2 + s) * h * d0
        + (-2 * s**3 + 3 * s**2) * y1
        + (s**3 - s**2) * h * d1
    )


def _crossing(t: np.ndarray, r: np.ndarray, rdot: np.ndarray, k: int, level: float) -> float:
    """Time in [t[k-1], t[k]] where the Hermite radius reaches `level`, by bisection."""
    lo, hi = float(t[k - 1]), float(t[k])
    args = (float(t[k - 1]), float(t[k]), float(r[k - 1]), float(r[k]))
    slopes = (float(rdot[k - 1]), float(rdot[k]))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _hermite(*args, *slopes, mid) < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _states(path: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    report = parse_numeric_report(path)
    times = report.column(f"{name}.ElapsedSecs")
    position = np.column_stack([report.column(f"{name}.EarthMJ2000Eq.{a}") for a in "XYZ"])
    velocity = np.column_stack(
        [report.column(f"{name}.EarthMJ2000Eq.V{a}") for a in "XYZ"]
    )
    return times, position, velocity


def _with_goal(scenario: Scenario, document: GmatScriptDocument, goal_s: float, dest: Path) -> Path:
    segment = scenario.propagate_segments[0]
    value = goal_s / 86400.0 if segment.unit == "ElapsedDays" else goal_s
    line = document.lines[segment.line_index]
    for start, end in sorted(segment.duration_spans, reverse=True):
        line = line[:start] + repr(value) + line[end:]
    lines = list(document.lines)
    lines[segment.line_index] = line
    encoded = "".join(lines).encode("utf-8")
    if document.encoding.has_utf8_bom:
        encoded = b"\xef\xbb\xbf" + encoded
    dest.write_bytes(encoded)
    return dest


def _parse(
    script: Path, config: OrbitWinConfig, probe: StateProbe
) -> tuple[Scenario, GmatScriptDocument]:
    # The parse needs some horizon under the ranking rules; nothing here reads it.
    placeholder = dataclasses.replace(config.policy, horizon_s=1.0e9)
    try:
        return parse_gmat_script(
            script, dataclasses.replace(config, policy=placeholder), probe=probe
        )
    except AlignedStatesNeeded as exc:
        raise SystemExit(
            f"{script}: the craft start apart; this tool measures synchronous templates"
        ) from exc


def _diagnostic_run(
    script: Path,
    config: OrbitWinConfig,
    args: argparse.Namespace,
    workdir: Path,
    probe: StateProbe,
) -> tuple[Scenario, str, float]:
    scenario, document = _parse(script, config, probe)
    diagnostic = add_diagnostic_report(document, scenario, config.gmat.diagnostic_report_name)
    sibling = workdir / f"{script.stem}_diagnostic.script"
    workdir.mkdir(parents=True, exist_ok=True)
    sibling.write_bytes(diagnostic.script_bytes)
    names = [*written_reports(scenario), diagnostic.report_filename]
    result = run_gmat(
        args.gmat_console, sibling, workdir / "run", names,
        startup_file=args.startup_file, timeout_s=args.timeout_s,
    )
    if not result.success:
        raise GmatRunError(f"{sibling}: {result.error}")
    return scenario, result.reports[diagnostic.report_filename], result.duration_s


def _drift_km(scenario: Scenario, name: str, path: str, mark_s: float) -> tuple[float, float]:
    times, position, _ = _states(path, name)
    row = int(np.argmin(np.abs(times - mark_s)))
    craft = scenario.pursuer if scenario.pursuer.name == name else scenario.target
    two_body = propagate_universal(craft.initial_state, float(times[row]), scenario.mu_km3_s2)
    return float(np.linalg.norm(two_body.r - position[row])), float(times[row])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("script", type=Path)
    parser.add_argument("--boundary-km", type=float, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gmat-console", required=True)
    parser.add_argument("--startup-file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beyond-s", type=float, default=900.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--provisional", action="store_true",
        help="the script's goal is a guess; skip the goal == T_max check",
    )
    args = parser.parse_args()
    try:
        prepare_empty_output(args.output, "measure")
    except PolicyError as exc:
        parser.error(str(exc))
    config = load_config(args.config)

    probe = probe_initial_states(
        args.script, args.gmat_console, args.startup_file, args.output / "state_probe",
        timeout_s=args.timeout_s, extra_resource_types=config.policy.extra_resource_types,
    )
    scenario, document = _parse(args.script, config, probe)
    goal_s = scenario.propagate_segments[0].duration_s
    extended = _with_goal(
        scenario, document, goal_s + args.beyond_s, args.output / f"{args.script.stem}_long.script"
    )
    _, long_report, _ = _diagnostic_run(extended, config, args, args.output / "long", probe)
    t, position, velocity = _states(long_report, scenario.target.name)
    r = np.linalg.norm(position, axis=1)
    rdot = np.einsum("ij,ij->i", position, velocity) / r
    low = int(np.argmin(r))
    if not 0 < low < len(r) - 1:
        raise SystemExit(f"no perigee inside the run: radius is least at row {low} of {len(r)}")
    k = low if rdot[low] >= 0.0 else low + 1
    perigee_s = float(t[k - 1] - rdot[k - 1] * (t[k] - t[k - 1]) / (rdot[k] - rdot[k - 1]))
    perigee_km = _hermite(t[k - 1], t[k], r[k - 1], r[k], rdot[k - 1], rdot[k], perigee_s)
    out = np.nonzero((r >= args.boundary_km) & (np.arange(len(r)) > low))[0]
    if out.size == 0:
        raise SystemExit(f"SC_Alien does not reach {args.boundary_km} km by the end; raise --beyond-s")
    t_max = _crossing(t, r, rdot, int(out[0]), args.boundary_km)
    if not args.provisional and abs(t_max - goal_s) > GOAL_TOLERANCE_S:
        raise SystemExit(
            f"measured T_max {t_max:.4f} s differs from the script's goal {goal_s!r} s by more "
            f"than {GOAL_TOLERANCE_S} s; write the new goal, or pass --provisional"
        )

    target = cartesian_to_keplerian(scenario.target.initial_state, scenario.mu_km3_s2)
    pursuer_h = np.cross(scenario.pursuer.initial_state.r, scenario.pursuer.initial_state.v)
    target_h = np.cross(scenario.target.initial_state.r, scenario.target.initial_state.v)
    cosine = np.dot(pursuer_h, target_h) / np.linalg.norm(pursuer_h) / np.linalg.norm(target_h)
    radius_km = config.policy.central_body_radius_km

    official = written_reports(scenario)
    exact = run_gmat(
        args.gmat_console, args.script, args.output / "exact", official,
        startup_file=args.startup_file, timeout_s=args.timeout_s,
    )
    if not exact.success:
        raise GmatRunError(f"{args.script}: {exact.error}")
    _, diagnostic_report, diagnostic_s = _diagnostic_run(
        args.script, config, args, args.output / "diagnostic", probe
    )
    event = evaluate_diagnostic_report(
        diagnostic_report, scenario.pursuer.name, scenario.target.name,
        config.policy.intercept_radius_km, alignment=scenario.alignment,
    )
    end_t, end_position, _ = _states(diagnostic_report, scenario.target.name)
    _, human_position, _ = _states(diagnostic_report, scenario.pursuer.name)

    print(f"- script: {args.script.name}")
    print(f"  mission_boundary_km: {args.boundary_km:g}")
    print(f"  horizon_s: {t_max:.3f}")
    print(f"  # script goal {goal_s!r} s; SC_Alien at "
          f"{float(np.linalg.norm(end_position[-1])):.4f} km on its last row ({end_t[-1]:.3f} s)")
    print(f"  # SC_Alien at t = 0: {r[0]:.1f} km, "
          f"{'inside' if r[0] < args.boundary_km else 'outside'} the boundary")
    print(f"  perigee_time_s: {perigee_s:.3f}")
    print(f"  perigee_altitude_km: {perigee_km - radius_km:.3f}")
    print(f"  # osculating perigee altitude {target.sma_km * (1 - target.ecc) - radius_km:.3f} km")
    v_inf = math.sqrt(-scenario.mu_km3_s2 / target.sma_km) if target.sma_km < 0 else math.nan
    print(f"  v_inf_km_s: {v_inf:.4f}")
    print(f"  plane_angle_deg: {math.degrees(math.acos(float(np.clip(cosine, -1, 1)))):.3f}")
    print(f"  zero_burn_closest_km: {event.minimum_distance_km:.3f}"
          f"  # at {event.minimum_distance_time_s:.1f} s")
    print("  human_final_position_km: [" + ", ".join(f"{v:.4f}" for v in human_position[-1]) + "]")
    for name in (scenario.pursuer.name, scenario.target.name):
        for mark in (1800.0, 3600.0):
            drift, at = _drift_km(scenario, name, long_report, mark)
            print(f"  # two-body drift {name} at {at:.0f} s: {drift:.3f} km")
    print(f"  # GmatConsole wall, exact script to goal: {exact.duration_s:.2f} s")
    print(f"  # GmatConsole wall, diagnostic sibling to goal: {diagnostic_s:.2f} s")
    return 0


def _main() -> int:
    try:
        return main()
    except (GmatRunError, PolicyError) as exc:
        # A start that lands unpowered cannot be flown with no burns, which is
        # what this measures; say so instead of a traceback.
        print(f"measure_ranking_scenario: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(_main())
