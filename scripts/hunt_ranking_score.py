#!/usr/bin/env python3
"""Search for the schedule that scores best under the ranking rules' caps.

`scripts/hunt_multi_burn.py` closes the last leg with Lambert; under the
ranking caps (0.25 km/s per impulse, 3 km/s in total) that last impulse almost
never fits. This one keeps every impulse free and capped, and scores near
misses as eq. (2), (3) and (5) do, so it returns a near miss when no hit is in reach.

    python scripts/hunt_ranking_score.py OFFICIAL.script --config run/ranking.yaml \
        --output run/ranking_hunt.json

Without a `score` block it maximises the distance term first and then
minimises delta-v. `--target-from-gmat` aims at the target as GMAT flies it
(one run of $GMAT_CONSOLE with $GMAT_STARTUP) instead of a two-body estimate.

What comes out is analytical: a candidate for GMAT, not a result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitwin.config import add_rules_argument, load_config, rulebook_refusal
from orbitwin.ephemeris import target_ephemeris_from_gmat
from orbitwin.io import write_candidates
from orbitwin.pinned import refuse_unpinned
from orbitwin.ranking_search import (
    FORCE_MODEL_MARGINS_KM,
    EphemerisTarget,
    RankingSolution,
    TargetTrack,
    best_per_margin,
    build_setup,
    build_starts,
    schedule_signature,
    scored_candidate,
    search_ranking,
)
from orbitwin.state_probe import add_probe_arguments, load_scenario_for_tool


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("script", help="the organizer's .script")
    parser.add_argument("--config", required=True, help="ranking config; score block optional")
    add_rules_argument(parser)
    parser.add_argument(
        "--burns", type=int, nargs="+",
        help="impulse counts; default 1 to search.ranking_max_burns",
    )
    parser.add_argument("--transfer-starts", type=int, default=8, help="Lambert-split starts per N")
    parser.add_argument("--random-starts", type=int, default=4, help="Sobol starts per N")
    parser.add_argument("--polish-top", type=int, default=12, help="solutions polished longer")
    parser.add_argument("--screen-iterations", type=int, default=80)
    parser.add_argument("--polish-iterations", type=int, default=400)
    parser.add_argument("--aim-radius-km", type=float, default=1.0, help="where a hit aims")
    parser.add_argument(
        "--altitude-floor-km",
        type=float,
        help="how far above the central body the pursuer stays, over the whole arc to T_max; "
        "default policy.minimum_flown_altitude_km, else 100",
    )
    parser.add_argument(
        "--force-model-margins-km",
        type=float,
        nargs="+",
        default=list(FORCE_MODEL_MARGINS_KM),
        help="each start is searched with each margin added to the floor; replay each "
        "margin's best in GMAT to choose (not automatic yet)",
    )
    parser.add_argument(
        "--correction-margin-km-s",
        type=float,
        default=0.0,
        help="taken off each cap and the total, as headroom for a GMAT correction",
    )
    parser.add_argument("--grid-step-s", type=float, default=30.0)
    parser.add_argument("--target-from-gmat", action="store_true")
    parser.add_argument("--gmat-workdir", default="run/ranking_hunt_target")
    parser.add_argument("--workers", type=int, default=0, help="0 means os.cpu_count() - 1")
    parser.add_argument("--seed", type=int, help="start-set seed; default search.seed")
    parser.add_argument("--output", help="where to write the best solutions as JSON")
    parser.add_argument(
        "--candidates-output",
        help="the same solutions as a candidates.json that orbitwin rank and certify read",
    )
    parser.add_argument("--top", type=int, default=15, help="rows to print and write")
    add_probe_arguments(parser)
    arguments = parser.parse_args()
    refuse_unpinned()

    began = time.perf_counter()
    config = load_config(arguments.config)
    refusal = rulebook_refusal(config, arguments.config, arguments.rules)
    if refusal is not None:
        parser.error(refusal)
    policy_floor_km = config.policy.minimum_flown_altitude_km
    if arguments.altitude_floor_km is None:
        arguments.altitude_floor_km = 100.0 if policy_floor_km is None else policy_floor_km
    elif policy_floor_km is not None and arguments.altitude_floor_km < policy_floor_km:
        parser.error(
            f"--altitude-floor-km {arguments.altitude_floor_km:g} is under "
            f"policy.minimum_flown_altitude_km = {policy_floor_km:g}, which verify and certify "
            "would refuse"
        )
    console, startup = arguments.gmat_console, arguments.startup_file
    if arguments.target_from_gmat and console is None:
        console, startup = os.environ.get("GMAT_CONSOLE"), os.environ.get("GMAT_STARTUP")
        if not console:
            parser.error("--target-from-gmat needs --gmat-console or GMAT_CONSOLE and its startup")
    # The console that flies the target also reads the initial states.
    scenario, document = load_scenario_for_tool(parser, arguments.script, config, console, startup)
    target: TargetTrack | None = None
    if arguments.target_from_gmat:
        assert console is not None
        target = EphemerisTarget(
            target_ephemeris_from_gmat(
                scenario, document, config, console, startup, arguments.gmat_workdir
            )
        )
    setup = build_setup(
        scenario,
        config,
        target=target,
        altitude_floor_km=arguments.altitude_floor_km,
        force_model_margins_km=arguments.force_model_margins_km,
        correction_margin_km_s=arguments.correction_margin_km_s,
        aim_radius_km=arguments.aim_radius_km,
        grid_step_s=arguments.grid_step_s,
        screen_iterations=arguments.screen_iterations,
        polish_iterations=arguments.polish_iterations,
    )
    limits = setup.limits
    budget = min(config.search.ranking_max_burns, limits.max_burns)
    counts = arguments.burns or list(range(1, budget + 1))
    if not all(1 <= count <= limits.max_burns for count in counts):
        parser.error(f"--burns must lie in 1..{limits.max_burns}, as many as fit the burn window")
    workers = arguments.workers if arguments.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    seed = arguments.seed if arguments.seed is not None else config.search.seed
    starts = build_starts(
        setup, config, counts,
        transfer_count=arguments.transfer_starts, random_count=arguments.random_starts, seed=seed,
    )
    print(
        f"scenario {Path(arguments.script).name}   burns {counts}   starts {len(starts)}   "
        f"workers {workers}   score {'given' if config.score else 'none: distance, then dv'}",
        flush=True,
    )
    total = "none" if limits.total_cap_km_s is None else f"{limits.total_cap_km_s} km/s"
    print(
        f"walls: |dv| <= {limits.burn_cap_km_s} km/s, total <= {total}, gap >= "
        f"{limits.separation_s} s, radius >= {limits.radius_floor_km:.1f} km to "
        f"T_max = {limits.horizon_s:.1f} s, target {'GMAT' if target else 'two-body'}",
        flush=True,
    )
    result = search_ranking(
        scenario, config, setup, starts, workers=workers, polish_top=arguments.polish_top
    )
    elapsed = time.perf_counter() - began

    # The top rows, the best of each margin if not among them, and the zero
    # burn once, so GMAT verification has every kind to choose from.
    shown: list[RankingSolution] = []
    for solution in [
        *result.solutions[: arguments.top],
        *best_per_margin(result).values(),
        *([result.baseline] if result.baseline is not None else []),
    ]:
        if all(schedule_signature(solution) != schedule_signature(s) for s in shown):
            shown.append(solution)

    def row(name: str, solution: RankingSolution) -> str:
        candidate = scored_candidate(solution, scenario, config, limits)
        m = solution.measurement
        score = f"{candidate.score:9.4f}" if candidate.score is not None else "        -"
        hit = f"{m.intercept_time_s:9.1f}" if m.intercept_time_s is not None else "     miss"
        return (
            f"{name:>30} {len(candidate.burns):>2} {score} {m.minimum_distance_km:10.3f} "
            f"{hit} {candidate.total_dv_km_s:7.4f} {m.minimum_radius_km:8.1f} "
            f"{solution.floor_margin_km:6g}"
        )

    print(f"\n{'label':>30} {'N':>2} {'score':>9} {'d_min km':>10} {'t_int s':>9} "
          f"{'dV km/s':>7} {'r_min km':>8} {'margin':>6}")
    for solution in shown:
        name = "zero-burn" if solution is result.baseline else f"{solution.label}:{solution.phase}"
        print(row(name[-30:], solution))
    print(f"\n{len(result.solutions)} solutions inside every wall, {elapsed:.1f} s")
    best = result.solutions[0]
    for index, (time_s, impulse) in enumerate(
        zip(best.times_s, best.impulses_km_s, strict=True), 1
    ):
        magnitude = sum(c * c for c in impulse) ** 0.5
        print(f"  burn {index}: t={time_s:11.4f} s   |dv|={magnitude * 1000:8.3f} m/s")
    print("\nAnalytical only. Replay it with GMAT before believing it.")
    if arguments.output:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                **solution.to_dict(),
                "candidate": scored_candidate(solution, scenario, config, limits).to_dict(),
            }
            for solution in shown
        ]
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {destination}")
    if arguments.candidates_output:
        write_candidates(
            arguments.candidates_output,
            [scored_candidate(solution, scenario, config, limits) for solution in shown],
        )
        print(f"wrote {arguments.candidates_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
