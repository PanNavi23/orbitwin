from __future__ import annotations

import dataclasses
import json
import math
import warnings
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

from .config import OrbitWinConfig, validate_config
from .constraints import (
    FORMAL_FRONTIER,
    ROBUST_FRONTIER,
    below_minimum_flown_altitude,
    flies_through_central_body,
    frontier_of,
    gmat_contradicts_hit,
    interception_required,
    prefer_deep_hits,
    shallow_stop,
    validate_candidate,
)
from .errors import (
    AltitudeFloorError,
    ConstraintVerdictError,
    GmatRunError,
    GmatVerdictError,
    PolicyError,
    StopVerdictError,
)
from .gmat_failures import gmat_failures, verification_error
from .gmat_loop import (
    GmatQueue,
    correct_one,
    correction_choices,
    correction_record,
    miss_choices,
    ranking_search_candidates,
)
from .gmat_parser import scenario_json
from .io import prepare_empty_output, write_candidates, write_candidates_csv
from .mission_clock import horizon_override_record
from .models import (
    STAGE_SCRIPT_NAME,
    Candidate,
    CandidateStage,
    Scenario,
    clock_stamps,
    uses_head_start,
)
from .rules import interception_rules, rule_set_record
from .score import (
    measured_minimum_distance_km,
    ranking_key,
    score_stamp,
    scored_closest_approach_km,
)
from .script_renderer import render_candidate, team_number_warnings, write_rendered_candidate
from .search import (
    search_one_burn,
    select_candidates,
    slot_assignment_variants,
    union_seeds,
    zero_burn_candidate,
)
from .spad import spad_check
from .state_probe import load_scenario
from .time_origin import template_goal, time_origin_evidence
from .two_burn import search_two_burn_global
from .verify import verify_candidate_with_gmat

#: rank says the same of the archive this writes.
SOLVED_WITHOUT_GMAT = (
    "Solve ran without --gmat-console and --startup-file, so it flew nothing in GMAT. Under "
    "the ranking rules nothing in this archive is submittable, since certify may fly any row "
    "as a miss: run solve again with GMAT (GMAT_CONSOLE and GMAT_STARTUP for "
    "scripts/competition_day.sh)."
)


def _sort_key(
    candidate: Candidate, *, scenario: Scenario | None, scored: bool, misses_count: bool = False
) -> tuple[float | str, ...]:
    """Order by evidence level first, then by the official ranking.

    GMAT-verified candidates precede unverified ones and successes precede
    failures regardless of score, because a higher analytical score on an
    unverified candidate is not a stronger claim. Within a tier the official
    order applies: score, then minimum distance, total delta-v, completion time.

    ``scored`` is a property of the run, not of the candidate. Branching per
    candidate produced keys of different lengths whose third element was
    ``-score`` for one and ``total_dv_km_s`` for the other -- both floats, so a
    mixed list sorted without complaint and compared quantities that have
    nothing to do with each other.

    The last element is the candidate id, which is what makes the order total
    rather than merely deterministic-looking, so the key is floats and one
    string -- not the ``tuple[float, ...]`` this used to claim. Both branches
    end that way, which is the only reason mixing them is safe at all: tuple
    comparison reaches the string only after every float ahead of it tied, and
    then compares a string to a string.
    """
    verified_key = 0.0 if candidate.verified_by_gmat else 1.0
    # Under the ranking rules a scored miss competes with a hit on score alone.
    success_key = 0.0 if candidate.successful or (scored and misses_count) else 1.0
    if scored:
        return (verified_key, success_key, *ranking_key(candidate, scenario=scenario))
    return (
        verified_key,
        success_key,
        candidate.total_dv_km_s,
        measured_minimum_distance_km(candidate),
        candidate.intercept_time_s if candidate.intercept_time_s is not None else math.inf,
        candidate.candidate_id,
    )


def _distinct_trajectories_first(
    candidates: list[Candidate], limit: int, *, misses_count: bool = False
) -> list[Candidate]:
    """Choose which candidates GMAT replays, distinct trajectories first.

    Slot variants carry the same impulses at the same epochs. Only which
    ImpulsiveBurn resource holds them differs, and with it how the unused slots
    are scheduled -- so they render to different scripts and GMAT integrates
    them in different chunks. Measured on the different-plane scenario, two
    variants of one impulse replayed 1.8e-9 km apart in closest approach.

    That is 1.8 micrometres. It cannot change the interception, which is
    decided at 5 km; it cannot change the score, whose distance term is exactly
    50 on any success; and it cannot change a tie-break at any resolution the
    organizer's report carries. Meanwhile 37% of the archive on that scenario
    is slot variants -- 29 candidates over 18 distinct trajectories -- and the
    top two by rank were the same impulse, so three GMAT runs covered two
    trajectories.

    On the day the GMAT budget is the scarce thing: every run is time on a
    machine that is not ours. So one candidate per trajectory is taken first,
    and the variants it skipped fill whatever budget is left. Nothing is
    excluded, only reordered, and every pass preserves rank order so the choice
    stays deterministic.

    The reordering stays inside the success tier, and that limit is the whole
    difference between this being an improvement and a regression. Reaching
    across it, a first version of this spent a run on the zero-burn baseline --
    a candidate the analytical layer had already measured as missing -- in
    place of a second script for a trajectory that hits. A replay cannot turn
    the miss into a submission, and the variant it displaced is a different
    script that could be one.
    """

    def by_trajectory(group: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
        seen: set[tuple[Any, ...]] = set()
        distinct: list[Candidate] = []
        variants: list[Candidate] = []
        for candidate in group:
            # The impulses themselves, not the slots that carry them. A variant
            # is built with dataclasses.replace and keeps dv_inertial_km_s
            # untouched, so exact equality groups them.
            key = tuple(
                sorted((burn.time_s, tuple(burn.dv_inertial_km_s)) for burn in candidate.burns)
            )
            if key in seen:
                variants.append(candidate)
            else:
                seen.add(key)
                distinct.append(candidate)
        return distinct, variants

    hits = [candidate for candidate in candidates if candidate.successful or misses_count]
    misses = [candidate for candidate in candidates if not (candidate.successful or misses_count)]
    distinct, variants = by_trajectory(hits)
    return (distinct + variants + misses)[:limit]


def prune_beside_model_plans(
    renderable: Sequence[Candidate], config: OrbitWinConfig
) -> list[Candidate]:
    """select_candidates on the two-body plans, then every plan flown in a pursuer model.

    A model plan's numbers are GMAT's to within a few km, a two-body plan's are
    not; pruned together, they dropped the two-body plan GMAT scored best on R1.
    """
    kept = select_candidates(
        [c for c in renderable if "pursuer_model" not in c.metadata and not uses_head_start(c.burns)],
        config.search.keep_candidates,
        config,
    )
    # Head-start plans apart too, so none prunes a plan without them (rank's fallback).
    early = select_candidates(
        [c for c in renderable if "pursuer_model" not in c.metadata and uses_head_start(c.burns)],
        config.search.keep_candidates,
        config,
    )
    return [*kept, *early, *(c for c in renderable if "pursuer_model" in c.metadata)]


def verification_beside_model_plans(
    candidates: Sequence[Candidate], count: int, *, misses_count: bool
) -> list[Candidate]:
    """_distinct_trajectories_first on the two-body plans, then again on those flown
    in a pursuer model, so a model plan never takes a two-body plan's replay."""
    return [
        candidate
        for in_model in (False, True)
        for candidate in _distinct_trajectories_first(
            [c for c in candidates if ("pursuer_model" in c.metadata) == in_model],
            count,
            misses_count=misses_count,
        )
    ]


def _schedule_key(candidate: Candidate) -> tuple[Any, ...]:
    """What a candidate id hashes besides the impulse vectors, in the time order
    a correction writes its burns in."""
    return (
        tuple(
            (burn.time_s, burn.slot_index)
            for burn in sorted(candidate.burns, key=lambda burn: burn.time_s)
        ),
        candidate.branch,
        candidate.metadata.get("arrival_time_s"),
    )


def _zero_burn_closest_approach(candidates: Sequence[Candidate]) -> dict[str, Any] | None:
    """What ranking_day_config.py --zero-burn-dmin-km prices: GMAT's when it flew it."""
    zero = next((c for c in candidates if c.branch == "zero-burn"), None)
    if zero is None or zero.minimum_distance_km is None:
        return None
    # verify writes GMAT's distance before checks that can still fail it, so the
    # distance's source is its own record, not the stage.
    flown = (zero.metadata.get("gmat_event") or {}).get("minimum_distance_km")
    return {
        "km": float(zero.minimum_distance_km),
        "measured_by": "gmat" if flown == zero.minimum_distance_km else "two-body",
        "gmat_verification_error": zero.metadata.get("gmat_verification_error"),
    }


def zero_burn_line(summary: dict[str, Any]) -> str | None:
    """The line solve prints for step 5 of docs/RUNBOOK.md, or None without a zero burn."""
    closest = summary.get("zero_burn_closest_approach")
    if not closest:
        return None
    source = "GMAT" if closest["measured_by"] == "gmat" else "two-body, not GMAT"
    if closest.get("gmat_verification_error"):
        source += ", though its verification failed"
    return (f"the zero burn passes {closest['km']:.3f} km off ({source}); "
            f"scripts/ranking_day_config.py --zero-burn-dmin-km {closest['km']:.3f}")


def closest_certifiable_miss(
    eligible: Sequence[Candidate], *, alignment_minimum_distance_km: float | None
) -> Candidate | None:
    """The verified miss in `eligible` that passes closest, nearer than the zero
    burn on the distance the term prices; RUNBOOK step 4a certifies it beside the best.

    Without coefficients the best miss is the zero burn, which spends least; this
    is the other end of the trade, and only the announced score can pick between them.
    """
    def scored_km(candidate: Candidate) -> float:
        return scored_closest_approach_km(candidate, alignment_minimum_distance_km)

    zero_km = min(
        (scored_km(c) for c in eligible if c.branch == "zero-burn"), default=math.inf
    )
    # Step 4a's net does not lean on a head start, as for final-01.
    closer = [
        c for c in eligible
        if c.verified_by_gmat and not c.successful and scored_km(c) < zero_km
        and not uses_head_start(c.burns)
    ]
    return min(
        closer, key=lambda c: (scored_km(c), c.total_dv_km_s, c.candidate_id), default=None
    )


def closest_miss_line(summary: dict[str, Any]) -> str | None:
    """The line solve prints for step 4a of docs/RUNBOOK.md, or None without one."""
    wanted = summary.get("closest_certifiable_miss")
    if wanted is None:
        return None
    best = summary.get("best_certifiable_without_shallow_opt_in")
    if wanted == best:
        return f"the closest verified miss is {wanted}, the best as well"
    beside = "" if best is None else f" beside {best}"
    return f"the closest verified miss is {wanted}; RUNBOOK step 4a certifies it{beside}"


def solve_scenario(
    script_path: str | Path,
    config: OrbitWinConfig,
    output_dir: str | Path,
    *,
    render_top: int | None = None,
    gmat_console: str | Path | None = None,
    startup_file: str | Path | None = None,
    gmat_verify_top: int | None = None,
) -> dict[str, Any]:
    validate_config(config)  # a config built in code never went through load_config
    output = Path(output_dir)
    # Read first, so a refused script or probe leaves no directory to retry around.
    scenario, document, probe_evidence = load_scenario(
        script_path, config, gmat_console=gmat_console, startup_file=startup_file
    )
    # Before the output exists: a SPAD file GMAT cannot find refuses the run.
    spad_record, spad_notes = spad_check(scenario, document, gmat_console, startup_file)
    if scenario.alignment is not None and scenario.alignment.offset_s > 0.0:
        template_goal(scenario, document)  # the evidence below refuses a goal it cannot read
    for note in spad_notes:
        warnings.warn(note, RuntimeWarning, stacklevel=2)
    prepare_empty_output(output, "solve")
    if probe_evidence is not None:
        scenario = probe_evidence.place(scenario, output / "state_probe")
    (output / "scenario.json").write_text(scenario_json(scenario), encoding="utf-8")
    # Report only: it checks T_max against each epoch and switches nothing.
    origin_evidence = time_origin_evidence(
        scenario, document, config, gmat_console, startup_file, output / "time_origin_evidence"
    )

    raw_candidates: list[Candidate] = [zero_burn_candidate(scenario, config)]
    # Stamped so `rank` can tell whether an archive belongs to the script it was
    # handed. Measured: ranking the DifferentPlane archive against the
    # HighPlaneChange script produced a confident report 25.000 points out --
    # the entire time term -- because a failure is scored at 4TA and the two
    # horizons are 33988.714 s and 42830.675 s.
    scenario_fingerprint = scenario.source_sha256
    # Which solutions the sweep finds depends on where its samples land, so
    # repeating it under independent draws and keeping the union is the one
    # widening measured that cannot lose a candidate. seed_count = 1 runs the
    # loop once with the configured seed and is the single-seed search exactly.
    seeds = union_seeds(config.search)
    duplicates = 0
    for seed in seeds:
        drawn = dataclasses.replace(
            config, search=dataclasses.replace(config.search, seed=seed)
        )
        raw_candidates.extend(search_one_burn(scenario, drawn))
        if config.search.two_burn_enabled:
            raw_candidates.extend(search_two_burn_global(scenario, drawn))
    if len(seeds) > 1:
        # The grid and the organizer's timeline landmarks do not depend on the
        # Sobol seed, so every extra sweep re-finds them. Dropping the repeats
        # here keeps the renderability pass from paying for each one again.
        # Order is preserved, so the first seed's candidates stay first.
        # Keeping the first copy is wrong, and the reason is subtle enough that
        # the first version of this shipped it. candidate_id hashes the burns,
        # the branch and the arrival epoch -- not intercept_time_s, not
        # minimum_distance_km, not metadata["event_refined"]. Each sweep refines
        # only its own top event_refine_top, so the same id can come out of one
        # sweep unrefined and out of another refined, and "first" means
        # "least-measured". Measured on DifferentPlane_Circular with three
        # seeds: 184 of 369 ids appeared more than once and 2 of those kept an
        # unrefined copy over a refined one, discarding a first-entry time this
        # run had already computed and leaving intercept_time_s 1.35 s wrong.
        # Named for what they hold rather than `best` and `order`, both of which
        # this function already uses -- for the single candidate it nominates and
        # for a sort key. Neither pair overlaps at run time, but mypy reads the
        # second assignment against the first and reported eighteen errors about
        # a Candidate that is a dict and a partial that is a list of str.
        by_id: dict[str, Candidate] = {}
        first_seen: list[str] = []
        duplicates = 0
        for candidate in raw_candidates:
            kept = by_id.get(candidate.candidate_id)
            if kept is None:
                by_id[candidate.candidate_id] = candidate
                first_seen.append(candidate.candidate_id)
                continue
            duplicates += 1
            if kept.metadata.get("event_refined") is not True and (
                candidate.metadata.get("event_refined") is True
            ):
                by_id[candidate.candidate_id] = candidate
        raw_candidates = [by_id[key] for key in first_seen]
    ranking_search: dict[str, Any] | None = None
    if config.search.ranking_search:
        found, ranking_search = ranking_search_candidates(
            scenario, document, config, output, gmat_console, startup_file
        )
        raw_candidates.extend(found)
    # The same impulses carried by different official slots render differently
    # and are not interchangeable for renderability, so both are offered.
    raw_candidates.extend(slot_assignment_variants(scenario, config, tuple(raw_candidates)))
    for candidate in raw_candidates:
        candidate.metadata["scenario_sha256"] = scenario_fingerprint
        # The horizon is not in the script's hash once policy.horizon_s sets it.
        candidate.metadata["horizon_s"] = scenario.max_time_s
        # Nor is the clock T_team runs on (policy.mission_time_origin).
        candidate.metadata.update(clock_stamps(scenario))
        # Nor the coefficients the search aimed at: rank warns when they change.
        candidate.metadata["searched_score"] = score_stamp(config.score)

    # A mathematically valid candidate may not fit the organizer's fixed number
    # and ordering of burn slots (for example, a late one-burn intercept in a
    # two-slot template). Determine renderability *before* Pareto pruning so an
    # unrenderable low-delta-v point cannot dominate and discard the best legal
    # fallback.
    renderable_raw: list[Candidate] = []
    unrenderable_raw: list[Candidate] = []
    for candidate in raw_candidates:
        try:
            render_candidate(document, scenario, candidate, config)
        except PolicyError as exc:
            candidate.metadata["renderable"] = False
            candidate.warnings = tuple(
                sorted({*candidate.warnings, f"Not renderable in official template: {exc}"})
            )
            unrenderable_raw.append(candidate)
        else:
            candidate.metadata["renderable"] = True
            candidate.promote(CandidateStage.RENDERABLE)
            renderable_raw.append(candidate)

    renderable_candidates = prune_beside_model_plans(renderable_raw, config)
    misses_count = not interception_required(scenario)
    if misses_count:
        # The prune ranks every miss after every hit, but a miss scores here:
        # the zero burn and the ranking search's own results stay in the pool.
        kept_ids = {candidate.candidate_id for candidate in renderable_candidates}
        for candidate in renderable_raw:
            protected = candidate.branch == "zero-burn" or candidate.branch.startswith(
                "ranking-search/"
            )
            if protected and candidate.candidate_id not in kept_ids:
                kept_ids.add(candidate.candidate_id)
                renderable_candidates.append(candidate)
    # Retain a bounded analytical record of unrenderable points for diagnosis,
    # but never mix them into the submission Pareto archive.
    unrenderable_candidates = select_candidates(
        unrenderable_raw, min(50, config.search.keep_candidates), config
    )
    candidates = [*renderable_candidates, *unrenderable_candidates]
    order = partial(
        _sort_key, scenario=scenario, scored=config.score is not None, misses_count=misses_count
    )
    candidates.sort(key=order)
    renderable_candidates.sort(key=order)

    count_to_render = render_top if render_top is not None else config.search.render_top
    development_dir = output / "development_candidates"
    to_render = [
        candidate
        for in_model in (False, True)
        for candidate in [
            c for c in renderable_candidates if ("pursuer_model" in c.metadata) == in_model
        ][:count_to_render]
    ]
    for candidate in to_render:
        candidate_dir = development_dir / candidate.candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        write_rendered_candidate(
            candidate_dir / "candidate.script",
            candidate_dir / "candidate.script.diff.txt",
            document,
            scenario,
            candidate,
            config,
        )
        (candidate_dir / "candidate.json").write_text(
            json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    verify_count = gmat_verify_top if gmat_verify_top is not None else config.gmat.verify_top
    zero_burns = (
        [c for c in renderable_candidates if c.branch == "zero-burn"] if misses_count else []
    )

    def _verify_one(candidate: Candidate, console: str | Path) -> None:
        # Captured before the call. verify_candidate_with_gmat overwrites
        # `successful` from the diagnostic report and only then raises, so
        # reading it in the handler records GMAT's verdict twice and loses
        # the analytical one -- which is the only thing this field exists
        # to preserve. verify records the same thing on entry, but a run
        # that dies before it gets there leaves nothing behind.
        analytical = {
            "successful": candidate.successful,
            "intercept_time_s": candidate.intercept_time_s,
            "minimum_distance_km": candidate.minimum_distance_km,
        }
        try:
            verify_candidate_with_gmat(
                script_path,
                scenario,
                document,
                candidate,
                config,
                console,
                startup_file,
                output / "gmat_verification" / candidate.candidate_id,
            )
        except Exception as exc:  # Evidence is retained; failed candidates are not promoted.
            # Fail closed. A run that raised before it could update the
            # candidate -- process crash, interpreter error, timeout -- leaves
            # the analytical success flag in place, and that flag is what
            # selection reads. Clearing it is the difference between "GMAT
            # never ran this" and "GMAT accepted this".
            candidate.metadata["analytical_successful"] = analytical["successful"]
            candidate.metadata.setdefault("analytical_before_gmat", analytical)
            candidate.metadata["gmat_verification_error"] = str(exc)
            candidate.metadata["gmat_verification_error_type"] = type(exc).__name__
            # GMAT's answer about this plan, not a run that failed to give one.
            candidate.metadata["gmat_verification_verdict"] = isinstance(exc, GmatVerdictError)
            # verified_by_gmat is derived from stage and successful, so
            # clearing the success is enough to un-verify it.
            candidate.successful = False
            candidate.metadata["scored_miss"] = False
            # Catching everything is right -- nothing may be promoted on a
            # run that did not finish -- but "GMAT verification failed" was
            # said whatever failed. A refused output directory, a constraint
            # this candidate breaks, a TypeError from a refactor: all of them
            # read as GMAT's verdict and send the reader to the wrong logs.
            # GmatRunError is GMAT reporting; anything else is this side.
            if isinstance(exc, AltitudeFloorError):
                blame = "GMAT flew it under the altitude floor"
            elif isinstance(exc, ConstraintVerdictError):
                blame = "GMAT flew it, and its flight breaks a constraint"
            elif isinstance(exc, StopVerdictError):
                blame = "GMAT flew it, and the script stopped at its hit does not end inside"
            elif isinstance(exc, GmatRunError):
                blame = "GMAT rejected this candidate"
            else:
                blame = f"verification could not be completed ({type(exc).__name__})"
            candidate.warnings = tuple(
                sorted({*candidate.warnings, f"{blame}: {exc}"})
            )

    corrected: list[Candidate] = []
    corrections: list[dict[str, Any]] = []
    superseded: set[str] = set()
    #: Originals flown outside the sphere, with the correction that may replace them.
    replaceable: list[tuple[Candidate, Candidate]] = []
    chosen = (
        correction_choices(renderable_candidates, config)
        if gmat_console is not None and config.gmat.correct_top > 0
        else []
    )
    # Never chosen for correction, and verify_top 0 in the ranking profile.
    missed = (
        miss_choices(
            renderable_candidates, config,
            alignment_minimum_distance_km=scenario.alignment_minimum_distance_km,
        )
        if gmat_console is not None and misses_count
        else []
    )
    if gmat_console is not None and (chosen or zero_burns or missed):
        console = gmat_console
        # Each correction is verified as soon as it converges, in the
        # corrections' own gmat.workers. The lists below are built in the order
        # chosen once all have finished.
        known = {candidate.candidate_id: candidate for candidate in renderable_candidates}
        new_ids: set[str] = set()
        records: dict[int, dict[str, Any]] = {}
        targets: dict[int, Candidate | None] = {}
        queue = GmatQueue(config.gmat.workers)
        queued: set[int] = set()
        # Every original's first flight is stamped before it is replayed, as before.
        unstamped: dict[str, int] = {}
        for original in chosen:
            unstamped[original.candidate_id] = unstamped.get(original.candidate_id, 0) + 1
        held: dict[str, list[Candidate]] = {}
        # Two corrections share an id only if their originals share burn epochs,
        # slots, branch and arrival, which a correction keeps. One waits for
        # every earlier-chosen sibling, so the first chosen wins as before.
        unarrived: dict[Any, set[int]] = {}
        for index, original in enumerate(chosen):
            unarrived.setdefault(_schedule_key(original), set()).add(index)
        deferred: dict[int, Candidate] = {}

        def _verify_tier() -> int:
            # Ahead of the corrections, for an early hit, until they fit in one
            # round: then behind, so the last ones are not left to run alone.
            return 0 if queue.waiting(1) > queue.workers else 2

        def _verify(candidate: Candidate, tier: int | None = None) -> None:
            # Once each: two runs on one candidate would share its directory.
            if id(candidate) in queued:
                return
            queued.add(id(candidate))
            if unstamped.get(candidate.candidate_id):
                held.setdefault(candidate.candidate_id, []).append(candidate)
            else:
                queue.add(
                    _verify_tier() if tier is None else tier,
                    partial(_verify_one, candidate, console),
                )

        def _corrected(index: int, result: Any) -> None:
            fixed, record = correction_record(chosen[index], result)
            records[index] = record
            targets[index] = None
            # What GMAT made of the original, whatever its correction did.
            original_id = record["candidate_id"]
            first_km = record["first_flight_closest_km"]
            if first_km is not None:
                known[original_id].metadata["gmat_first_flight_closest_km"] = first_km
            unstamped[original_id] -= 1
            if not unstamped[original_id]:
                for candidate in held.pop(original_id, []):
                    queue.add(_verify_tier(), partial(_verify_one, candidate, console))
            unarrived[_schedule_key(chosen[index])].discard(index)
            if fixed is not None:
                deferred[index] = fixed
            for waiting in sorted(deferred):
                siblings = unarrived.get(_schedule_key(deferred[waiting]), set())
                if not any(sibling < waiting for sibling in siblings):
                    _resolve(waiting, deferred.pop(waiting))

        def _resolve(index: int, fixed: Candidate) -> None:
            if fixed.candidate_id in known:
                # GMAT hit it as planned: the same burns, so the same candidate.
                targets[index] = known[fixed.candidate_id]
                _verify(known[fixed.candidate_id])
                return
            fixed.metadata["scenario_sha256"] = scenario_fingerprint
            fixed.metadata["horizon_s"] = scenario.max_time_s
            fixed.metadata.update(clock_stamps(scenario))
            fixed.metadata["searched_score"] = score_stamp(config.score)
            try:
                render_candidate(document, scenario, fixed, config)
            except PolicyError as exc:
                records[index]["reason"] = f"corrected, but not renderable: {exc}"
                return
            fixed.metadata["renderable"] = True
            fixed.promote(CandidateStage.RENDERABLE)
            known[fixed.candidate_id] = fixed
            new_ids.add(fixed.candidate_id)
            targets[index] = fixed
            _verify(fixed)

        for index, original in enumerate(chosen):
            queue.add(
                1,
                partial(
                    correct_one, original, document, scenario, config, console,
                    startup_file, output,
                ),
                partial(_corrected, index),
            )
        # Every correction is verified, besides verify_count others. So is the
        # zero burn when a miss scores: spending nothing is often the answer.
        for candidate in zero_burns:
            _verify(candidate, 3)
        # Last tier: a waiting correction or hit replay always starts first.
        for candidate in missed:
            _verify(candidate, 3)
        queue.run()
        if deferred or held:
            raise RuntimeError("a correction or its replay was never released")
        added: list[Candidate] = []
        for index in range(len(chosen)):
            corrections.append(records[index])
            target = targets[index]
            if target is None:
                continue
            corrected.append(target)
            if target.candidate_id in new_ids and all(target is not kept for kept in added):
                added.append(target)
        candidates.extend(added)
        renderable_candidates.extend(added)
        # Flown outside the sphere an original is no hit, and rank and certify
        # refuse it. One GMAT flew inside may still be a hit, and keeps its
        # chance of a slot.
        added_by_id = {candidate.candidate_id: candidate for candidate in added}
        for record in corrections:
            first_km = record["first_flight_closest_km"]
            replacement = added_by_id.get(record["corrected_id"])
            if (
                first_km is not None
                and replacement is not None
                and first_km > config.policy.intercept_radius_km
            ):
                replaceable.append((known[record["candidate_id"]], replacement))
        # Only the unverified are chosen from below, and verification has not
        # touched their keys, so their order is the one before it.
        candidates.sort(key=order)
        renderable_candidates.sort(key=order)

    # The console this run will actually replay through, or None if it will not
    # replay at all -- a console with verify_top at zero is the same "no" as no
    # console. `gmat_attempted` is that same answer as a bool, which several
    # later branches want, but a bool carries no proof about the console it came
    # from: the block below hands the console to verify_candidate_with_gmat,
    # which requires one. Deriving both from a single narrowed name is what
    # keeps the two from drifting apart.
    verify_console = (
        gmat_console if verify_count > 0 or corrected or zero_burns or missed else None
    )
    gmat_attempted = verify_console is not None
    attempted_ids: set[str] = set()
    attempted_count = 0
    if verify_console is not None:
        # Verified above, alongside the corrections.
        always = [*corrected, *zero_burns, *missed]

        # An original stands down only for a correction GMAT has verified; one
        # whose correction failed keeps its chance of a slot below.
        for original, replacement in replaceable:
            if replacement.verified_by_gmat:
                original.metadata["superseded_by"] = replacement.candidate_id
                superseded.add(original.candidate_id)
        skipped = {candidate.candidate_id for candidate in always} | superseded
        others = verification_beside_model_plans(
            [c for c in renderable_candidates if c.candidate_id not in skipped],
            verify_count,
            misses_count=misses_count,
        )
        # Each run has its own directory and mutates only its own candidate;
        # the sort below waits for all of them.
        rest = GmatQueue(config.gmat.workers)
        for candidate in others:
            rest.add(0, partial(_verify_one, candidate, verify_console))
        rest.run()
        attempted_ids = {candidate.candidate_id for candidate in [*always, *others]}
        # Runs, not entries: a candidate two corrections resolve to is flown once.
        attempted_count = len({id(candidate) for candidate in always}) + len(others)
        # The two lists share Candidate objects but are distinct lists, so
        # sorting one does not reorder the other. Selection reads
        # renderable_candidates, so that is the list which has to be re-sorted.
        candidates.sort(key=order)
        renderable_candidates.sort(key=order)

    write_candidates(output / "candidates.json", candidates)
    write_candidates_csv(output / "candidates.csv", candidates)

    # An official burn keeps the production margin rather
    # than sit on the rule boundary, so the formal frontier is archived and
    # reported but cannot leave the archive unless that is turned on
    # deliberately. Certify and rank hold the same line.
    submittable_frontiers = (
        {ROBUST_FRONTIER, FORMAL_FRONTIER}
        if config.policy.allow_formal_frontier_submission
        else {ROBUST_FRONTIER}
    )

    def _withheld_reason(candidate: Candidate) -> str | None:
        """Why the gate refuses this candidate, or None if it does not.

        The reason is returned rather than a bare no because the summary has to
        report it, and there is now more than one. Reporting the wrong one is
        worse than reporting none: a candidate held back for flying through the
        Earth, described as sitting on the formal frontier, sends an operator to
        set allow_formal_frontier_submission, which will not release it.
        """
        # Computed, not read. A candidate deserialised from an older
        # candidates.json carries no frontier tag, and defaulting a missing tag
        # to "robust" would let a 1.499 km/s burn through the gate that exists
        # to stop it. The tag in metadata is for reporting; the gate decides.
        if frontier_of(candidate, config) not in submittable_frontiers:
            return "formal-frontier"
        # A candidate that reaches the interception by passing through the
        # central body is legal on the page -- the preliminary rules say
        # nothing about it and point-mass gravity integrates straight through
        # -- and impossible in fact. §8 leaves the final reading to the
        # organizer and §6 ends in a design presentation, so it is not the
        # thing to submit. It stays in the archive; only the gate refuses it.
        if not config.policy.allow_body_intersecting_submission and flies_through_central_body(
            candidate, config
        ):
            return "central-body"
        if below_minimum_flown_altitude(candidate, config):
            return "minimum-altitude"
        if gmat_contradicts_hit(candidate, config):
            return "gmat-contradicted"
        return None

    def _submittable(candidate: Candidate) -> bool:
        return _withheld_reason(candidate) is None

    passed_over: list[Candidate] = []
    eligible: list[Candidate] = []
    if gmat_attempted:
        # Only a candidate GMAT actually replayed may be promoted here.
        eligible = [
            candidate
            for candidate in renderable_candidates
            if candidate.verified_by_gmat and _submittable(candidate)
        ]
        # A deep hit beats a shallow one whatever the scores: data-file drift
        # can turn the shallow one into a miss on the organizer's machine.
        preferred = prefer_deep_hits(eligible, config)
        best = preferred[0] if preferred else None
        passed_over = eligible[: eligible.index(best)] if best is not None else []
    else:
        # Without GMAT the strongest available claim is analytical, and it only
        # means anything for candidates whose trajectory event was evaluated
        # rather than inherited from a Lambert endpoint residual.
        best = next(
            (
                candidate
                for candidate in renderable_candidates
                if (candidate.successful or misses_count)
                and candidate.metadata.get("event_refined") is True
                and _submittable(candidate)
            ),
            None,
        )

    # The filename states the authority behind the artifact. solve() never
    # certifies, so it can never produce team_solution.script -- that name
    # belongs to certify_candidate, after the exact-script replay and the
    # official report parity check have both passed on an organizer build.
    #
    # Derived inside the branch, not beside it. A filename exists exactly when a
    # candidate to write under it does, and stating that as one `if` rather than
    # as a ternary followed by three more tests is what lets everything below
    # read the name as a string: it is None only in the case this block does not
    # run.
    best_script_name: str | None = None
    if best is not None:
        best_script_name = STAGE_SCRIPT_NAME[best.stage]
        if best_script_name == STAGE_SCRIPT_NAME[CandidateStage.CERTIFIED]:
            # The paragraph above is a property, and this is the property.
            # Nothing in solve promotes that far today -- verification stops at
            # the organizer tier -- so reaching here means a candidate arrived
            # already carrying the stage, and the name would assert replays this
            # run never made. Compared as a name rather than as a stage on
            # purpose: the guarantee is about what gets written, and a future
            # stage mapped onto team_solution.script has to trip this too.
            raise PolicyError(
                f"Candidate {best.candidate_id} reached solve already at "
                f"{CandidateStage.CERTIFIED.value}; only certify may emit "
                f"{STAGE_SCRIPT_NAME[CandidateStage.CERTIFIED]}"
            )
        if best.score_is_stale:
            raise PolicyError(
                f"Candidate {best.candidate_id} carries a score computed from inputs it no longer has"
            )
        # The search already drops constraint violations, but the gate is the
        # last place anything is written and it should not depend on an earlier
        # filter having run. A candidate can reach here from candidates.json, a
        # patched search, or a future code path.
        gate = validate_candidate(best, scenario, config, require_success=not misses_count)
        if not gate.valid:
            raise PolicyError(
                f"Candidate {best.candidate_id} violates the competition rules and will not be "
                "written: " + "; ".join(gate.violations)
            )

    # "No solution" and "solutions exist but none is submittable under the
    # configured margins" are different answers, and the operator has to be able
    # to tell them apart without reading candidates.json.
    pipeline_warnings: list[str] = [*spad_notes, *team_number_warnings(scenario, config)]
    if misses_count and gmat_console is None:
        pipeline_warnings.append(SOLVED_WITHOUT_GMAT)
    if origin_evidence is not None:
        if origin_evidence.track_error is not None and config.policy.horizon_s is not None:
            pipeline_warnings.append(origin_evidence.lines[2])
        if origin_evidence.verdict != origin_evidence.configured_reading:
            # Undetermined too: nothing here backs the configured origin's T_max.
            pipeline_warnings.append(origin_evidence.lines[-1])
    if ranking_search is not None and ranking_search["target_error"]:
        pipeline_warnings.append(
            "The ranking search aimed at the two-body target: GMAT's target run failed with "
            f"{ranking_search['target_error']}"
        )
    if ranking_search is not None and ranking_search.get("pursuer_model_error"):
        pipeline_warnings.append(
            f"The ranking search flew the pursuer two-body: {ranking_search['pursuer_model_error']}"
        )
    if raw_candidates and not renderable_raw:
        # Measured: with policy.allow_zero_vector_unused_slots off, every
        # candidate raises PolicyError in the render probe, each is filed as
        # "not renderable in official template", and solve exits 0 with an empty
        # warnings list. A global policy decision had been rendered as a
        # per-candidate verdict: a swallowed failure reported as success.
        reasons = sorted(
            {
                warning
                for candidate in unrenderable_raw
                for warning in candidate.warnings
                if warning.startswith("Not renderable")
            }
        )[:3]
        pipeline_warnings.append(
            f"None of the {len(raw_candidates)} candidates can be rendered into the official "
            "template, so this run produced no submittable script. "
            + (f"First reasons: {'; '.join(reasons)}" if reasons else "")
        )
    if passed_over and best is not None:
        pipeline_warnings.append(
            f"{len(passed_over)} verified hit(s) ranked above {best.candidate_id} were passed "
            "over because their submitted script stops short of policy.stop_depth_km while "
            "another verified hit reaches it: "
            + ", ".join(c.candidate_id for c in passed_over[:5])
        )
    # A crash or a timeout loses that plan, and the choice above was made without it.
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    failures = gmat_failures(
        [
            *(("correction", by_id[record["candidate_id"]], str(record["error"]))
              for record in corrections if record.get("error")),
            *(("verification", candidate, error) for candidate in candidates
              if (error := verification_error(candidate)) is not None),
        ],
        best, archived=set(by_id), who="the chosen best",
    )
    if failures["warning"] is not None:
        pipeline_warnings.append(failures["warning"])
    best_shallow = shallow_stop(best, config) if best is not None else None
    # What certify takes without the opt-in, when the named best needs it.
    certifiable = (
        next((c for c in eligible if shallow_stop(c, config) is None), None)
        if best_shallow
        else best
    )
    # Step 4a's net when the best burns in the head start (COMPETITION_POLICY.md).
    without_head_start = (
        next((c for c in eligible
              if shallow_stop(c, config) is None and not uses_head_start(c.burns)), None)
        if certifiable is not None and uses_head_start(certifiable.burns)
        else certifiable
    )
    closest_miss = (
        closest_certifiable_miss(
            eligible, alignment_minimum_distance_km=scenario.alignment_minimum_distance_km
        )
        if misses_count else None
    )
    if best_shallow:
        opted_in = config.policy.allow_shallow_stop_submission
        pipeline_warnings.append(
            f"No verified hit reached policy.stop_depth_km, so the best candidate is a shallow "
            f"one. {best_shallow}. "
            + (
                "policy.allow_shallow_stop_submission is set, so certify will take it; "
                if opted_in
                else "certify refuses it unless policy.allow_shallow_stop_submission is set; "
            )
            + (
                f"the best it accepts without that setting is {certifiable.candidate_id}."
                if certifiable is not None
                else "no other verified candidate is submittable."
            )
        )
    if gmat_console is not None and not gmat_attempted:
        # The filename stays honest either way, but an operator who passed a
        # console asked for verification and got silence.
        pipeline_warnings.append(
            "A GMAT console was supplied but gmat.verify_top is 0, so nothing was verified. "
            "Set gmat.verify_top in the profile or pass --gmat-verify-top."
        )
    if gmat_attempted and best is None:
        # "GMAT rejected everything we gave it" and "GMAT was only given one of
        # twenty" are different answers, and an empty result looks the same.
        # Only claim the first when it is true: a candidate can verify and still
        # be withheld by the frontier gate, which the warning below covers.
        verified_attempted = sum(
            1
            for candidate in renderable_candidates
            if candidate.candidate_id in attempted_ids
            and candidate.verified_by_gmat
        )
        if verified_attempted == 0:
            # Only count candidates whose interception was actually evaluated.
            # An unrefined candidate's `successful` is its Lambert endpoint
            # residual, around 1e-10 km for any converged arc, so counting those
            # would advertise solutions that have not been demonstrated -- the
            # same residual this project already refuses to rank on.
            untried = [
                candidate
                for candidate in renderable_candidates
                if candidate.candidate_id not in attempted_ids
                and candidate.successful
                and candidate.metadata.get("event_refined") is True
            ]
            message = (
                f"GMAT verified none of the {attempted_count} candidate(s) it was given."
            )
            if untried:
                message += (
                    f" A further {len(untried)} candidate(s) with a demonstrated interception "
                    f"were never attempted because gmat.verify_top is {verify_count}; raise it "
                    "to reach them."
                )
            pipeline_warnings.append(message)
    if best is None:
        explained = len(pipeline_warnings)
        withheld: dict[str, list[Candidate]] = {}
        for candidate in renderable_candidates:
            if not (candidate.successful or misses_count):
                continue
            reason = _withheld_reason(candidate)
            if reason is not None:
                withheld.setdefault(reason, []).append(candidate)
        for reason, group in sorted(withheld.items()):
            if reason == "formal-frontier":
                pipeline_warnings.append(
                    f"{len(group)} candidate(s) were withheld because they sit on the "
                    f"formal frontier: legal under the rules but outside the production margins of "
                    f"{config.policy.dv_safe_limit_km_s:.6g} km/s and "
                    f"{config.policy.safe_burn_separation_s:.6g} s. Set "
                    "policy.allow_formal_frontier_submission to consider them."
                )
            elif reason == "gmat-contradicted":
                pipeline_warnings.append(
                    f"{len(group)} candidate(s) were withheld because GMAT flew them outside "
                    "the sphere in their correction's first round: "
                    + "; ".join(str(gmat_contradicts_hit(c, config)) for c in group[:3])
                )
            elif reason == "minimum-altitude":
                lowest = min(
                    float(candidate.metadata["minimum_pursuer_radius_km"]) for candidate in group
                )
                pipeline_warnings.append(
                    f"{len(group)} candidate(s) were withheld because they fly down "
                    f"to {lowest - config.policy.central_body_radius_km:.6g} km altitude, under "
                    f"policy.minimum_flown_altitude_km = "
                    f"{config.policy.minimum_flown_altitude_km}."
                )
            else:
                deepest = min(
                    float(candidate.metadata["minimum_pursuer_radius_km"]) for candidate in group
                )
                rulebook = interception_rules(scenario.rules)
                pipeline_warnings.append(
                    f"{len(group)} candidate(s) were withheld because their trajectory "
                    f"passes inside the {config.policy.central_body_radius_km:.6g} km central body "
                    f"before completing -- the deepest reaches {deepest:.6g} km. "
                    + (
                        f"The {rulebook.name} rules disqualify it ({rulebook.earth_crossing_clause})."
                        if rulebook.earth_crossing_clause
                        else "The preliminary rules do not forbid it and GMAT integrates "
                        "straight through; policy.allow_body_intersecting_submission decides "
                        "whether one may be submitted."
                    )
                )
        if len(pipeline_warnings) == explained:
            # Nothing above accounts for the empty result, and an empty result
            # with an empty explanation is the one outcome an operator cannot
            # act on. The two commonest causes reach here: no candidate closed,
            # and candidates closed but only on a Lambert endpoint residual.
            unrefined = [
                candidate
                for candidate in renderable_candidates
                if candidate.successful and candidate.metadata.get("event_refined") is not True
            ]
            if unrefined:
                pipeline_warnings.append(
                    f"{len(unrefined)} candidate(s) close on their Lambert endpoint residual but "
                    "none had its trajectory evaluated, and an unevaluated interception is not "
                    f"evidence of one. Raise search.event_refine_top above "
                    f"{config.search.event_refine_top}."
                )
            else:
                measured = [
                    float(candidate.minimum_distance_km)
                    for candidate in candidates
                    if candidate.metadata.get("event_refined") is True
                    and candidate.minimum_distance_km is not None
                ]
                closest = f"{min(measured):.6g} km" if measured else "not measured"
                pipeline_warnings.append(
                    f"No candidate reached the {config.policy.intercept_radius_km:.6g} km sphere "
                    f"within the {scenario.max_time_s:.6g} s horizon. {len(candidates)} candidate(s) "
                    f"are archived and the closest approach among the evaluated ones is {closest}. "
                    "Widen the search budget, or the scenario may not be reachable inside the "
                    "impulse limit."
                )

    if best is not None:
        best_dir = output / "best_candidate"
        best_dir.mkdir(parents=True, exist_ok=True)
        # Named from the candidate being written, which is where the name came
        # from a hundred lines up and where it still comes from. Reading
        # `best_script_name` here instead would read a `str | None` that is a
        # string only because `best` is not None -- true, and stated nowhere
        # that this expression can see.
        script_name = STAGE_SCRIPT_NAME[best.stage]
        write_rendered_candidate(
            best_dir / script_name,
            best_dir / f"{script_name}.diff.txt",
            document,
            scenario,
            best,
            config,
        )
        (best_dir / "candidate.json").write_text(
            json.dumps(best.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    summary = {
        "scenario": scenario.to_dict(),
        "candidate_count": len(candidates),
        "search_seeds": list(seeds),
        # What the search optimised for; None when it ran unscored.
        "searched_score": score_stamp(config.score),
        "rule_set": rule_set_record(config.policy.rules),
        "spad_files": spad_record,
        "duplicate_candidates_dropped": duplicates,
        "successful_count": sum(candidate.successful for candidate in candidates),
        "renderable_count": len(renderable_candidates),
        "gmat_verified_count": sum(candidate.verified_by_gmat for candidate in candidates),
        "gmat_verification_attempted": gmat_attempted,
        "ranking_search": ranking_search,
        "time_origin_evidence": (
            origin_evidence.to_dict() if origin_evidence is not None else None
        ),
        "announced_horizon_before_perigee": horizon_override_record(scenario),
        "gmat_corrections": corrections,
        "gmat_failures": failures,
        "gmat_miss_verifications": [
            {
                "candidate_id": candidate.candidate_id,
                "branch": candidate.branch,
                "analytical": candidate.metadata.get("analytical_before_gmat"),
                "verified_by_gmat": candidate.verified_by_gmat,
                # Unverified, score and distance may be GMAT's or still the analytic ones.
                "gmat_score": candidate.score if candidate.verified_by_gmat else None,
                "gmat_minimum_distance_km": (
                    candidate.minimum_distance_km if candidate.verified_by_gmat else None
                ),
                "gmat_event": candidate.metadata.get("gmat_event"),
                "error": candidate.metadata.get("gmat_verification_error"),
            }
            for candidate in missed
        ],
        "frontiers": {
            name: {
                "count": sum(
                    1
                    for candidate in candidates
                    if str(candidate.metadata.get("frontier", ROBUST_FRONTIER)) == name
                ),
                "best_candidate_id": next(
                    (
                        candidate.candidate_id
                        for candidate in renderable_candidates
                        if str(candidate.metadata.get("frontier", ROBUST_FRONTIER)) == name
                        and (candidate.successful or misses_count)
                    ),
                    None,
                ),
                "submittable": name in submittable_frontiers,
            }
            for name in (ROBUST_FRONTIER, FORMAL_FRONTIER)
        },
        "zero_burn_closest_approach": _zero_burn_closest_approach(candidates),
        "best_candidate": best.to_dict() if best else None,
        "best_candidate_script": best_script_name,
        "best_candidate_stage": best.stage.value if best is not None else None,
        "best_candidate_is_gmat_verified": bool(best and best.verified_by_gmat),
        "best_candidate_shallow_stop": best_shallow,
        "best_certifiable_without_shallow_opt_in": (
            certifiable.candidate_id if certifiable is not None else None
        ),
        "closest_certifiable_miss": (
            closest_miss.candidate_id if closest_miss is not None else None
        ),
        "best_certifiable_uses_head_start": (
            certifiable is not None and uses_head_start(certifiable.burns)
        ),
        "best_certifiable_without_head_start": (
            without_head_start.candidate_id if without_head_start is not None else None
        ),
        "warnings": [*scenario.warnings, *pipeline_warnings],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
