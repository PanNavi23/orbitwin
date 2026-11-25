#!/usr/bin/env python3
"""Search for the best N-impulse schedule the geometry allows.

The one- and two-burn searches in `orbitwin solve` enumerate transfers and
prefer cheap ones. When the announced coefficients make impulses nearly free
and time expensive -- as they did on 2026-09-05 -- that preference is exactly
backwards: the built-in search returned 74.997 and this returned 98.319 on the
same scenario. Run `scripts/safe_delta_v_floor.py` first to see which of the
two situation you are in, then run this.

    python scripts/hunt_multi_burn.py OFFICIAL.script --config run/score.yaml \
        --burns 5 6 7 --output run/hunt.json

The configuration must carry the announced `score` coefficients with their
units; the search optimises the official formula and cannot start without them.

Seeding from a known schedule is how a published or previous answer gets
checked rather than believed:

    python scripts/hunt_multi_burn.py OFFICIAL.script --config run/score.yaml \
        --seed-from previous_best.json --burns 6

What comes out is analytical. It is a candidate for GMAT, not a result --
`scripts/audit_burn_table.py` replays it independently and
`orbitwin certify` is what turns one into a submission.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitwin.config import add_rules_argument, load_config, rulebook_refusal
from orbitwin.multi_burn import (
    MultiBurnLimits,
    MultiBurnSolution,
    default_burn_counts,
    search_multi_burn,
    start_from_solution,
    structured_starts,
)
from orbitwin.pinned import refuse_unpinned
from orbitwin.state_probe import add_probe_arguments, load_scenario_for_tool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("script", help="the organizer's .script")
    parser.add_argument("--config", required=True, help="config carrying the score coefficients")
    add_rules_argument(parser)
    parser.add_argument(
        "--burns",
        type=int,
        nargs="+",
        default=None,
        help="impulse counts to try; default 5 6 7, or up to what spends the total cap",
    )
    parser.add_argument(
        "--seed-from",
        action="append",
        default=[],
        metavar="JSON",
        help="a previous solution to restart from; repeatable",
    )
    parser.add_argument("--output", help="where to write the best schedule as JSON")
    parser.add_argument("--workers", type=int, default=0, help="0 means os.cpu_count() - 1")
    parser.add_argument("--start-seed", type=int, default=2026, help="RNG seed for the start set")
    parser.add_argument(
        "--altitude-floor-km",
        type=float,
        default=100.0,
        help="how far above the central body the arc must stay, up to arrival",
    )
    parser.add_argument("--top", type=int, default=15, help="rows to print")
    add_probe_arguments(parser)
    arguments = parser.parse_args()
    refuse_unpinned()

    config = load_config(arguments.config)
    refusal = rulebook_refusal(config, arguments.config, arguments.rules)
    if refusal is not None:
        parser.error(refusal)
    if config.score is None:
        parser.error(
            "the configuration has no `score` block. This search optimises the official "
            "formula, so the announced kt, Ct, kv, Cv (and C_r under the ranking rules) "
            "must be entered with their units."
        )
    scenario, _ = load_scenario_for_tool(
        parser, arguments.script, config, arguments.gmat_console, arguments.startup_file
    )
    limits = MultiBurnLimits.from_config(
        scenario, config, altitude_floor_km=arguments.altitude_floor_km
    )

    if arguments.burns is None:
        counts, why = default_burn_counts(limits)
        arguments.burns = list(counts)
    else:
        why = "given with --burns"

    starts = list(structured_starts(arguments.burns, limits, seed=arguments.start_seed))
    for path in arguments.seed_from:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        starts.append(start_from_solution(payload, limits, label=f"seed:{Path(path).name}"))

    workers = arguments.workers if arguments.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    print(
        f"scenario {Path(arguments.script).name}   burns {arguments.burns}   "
        f"starts {len(starts)}   workers {workers}",
        flush=True,
    )
    print(f"burns {arguments.burns}: {why}", flush=True)
    print(
        f"walls: |dv| <= {limits.burn_cap_km_s} km/s, gap >= {limits.separation_s} s, "
        f"radius >= {limits.radius_floor_km:.1f} km to arrival, arrival <= {limits.horizon_s:.1f} s",
        flush=True,
    )

    found: list[MultiBurnSolution] = search_multi_burn(
        scenario,
        config,
        starts=starts,
        workers=workers,
        altitude_floor_km=arguments.altitude_floor_km,
    )
    if not found:
        print("\nNo start produced a schedule inside every wall.")
        print("Widen --burns, raise --altitude-floor-km's tolerance, or check the coefficients.")
        return 1

    print(f"\n{'label':>24} {'N':>2} {'score':>11} {'arrive s':>10} {'dV km/s':>9} {'r_min km':>10}")
    for solution in found[: arguments.top]:
        print(
            f"{solution.label:>24} {solution.burn_count:>2} {solution.score:>11.5f} "
            f"{solution.arrival_s:>10.2f} {solution.total_dv_km_s:>9.4f} "
            f"{solution.minimum_radius_km:>10.1f}"
        )

    best = found[0]
    print(
        f"\nBEST {best.score:.7f}   N={best.burn_count}   arrive {best.arrival_s:.4f} s   "
        f"dV {best.total_dv_km_s:.6f} km/s"
    )
    for index, burn in enumerate(best.burns, 1):
        print(f"  burn {index}: t={burn.time_s:11.4f} s   |dv|={burn.magnitude_km_s * 1000:9.3f} m/s")

    print(
        "\nAnalytical only. Replay it with scripts/audit_burn_table.py before believing it, "
        "and remember the radius wall stops at arrival -- pass "
        "--require-safe-through-horizon there to check the rest of the arc. "
        "Under a richer force model it checks the whole arc anyway, and leaves "
        "the interception to GMAT's report."
    )

    if arguments.output:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(best.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
