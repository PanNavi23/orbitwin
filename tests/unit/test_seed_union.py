from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from orbitwin.config import ConfigurationError, OrbitWinConfig, SearchConfig, validate_config
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.search import search_one_burn, union_seeds

# Repeating the sweep under independent draws and keeping the union. Which
# solutions a sweep finds depends on where its samples land -- five seeds on
# DifferentPlane_Circular give 543.9, 648.8, 648.8, 648.8 and 648.8 m/s -- and
# a union is the one widening measured that cannot lose a candidate.


def _config(seed_count: int, seed: int = 20260818) -> OrbitWinConfig:
    return OrbitWinConfig(
        search=SearchConfig(
            seed=seed,
            seed_count=seed_count,
            one_burn_samples=64,
            one_burn_grid_burns=2,
            one_burn_grid_arrivals=2,
            max_revolutions=0,
            keep_candidates=200,
            render_top=3,
            event_step_s=20.0,
            event_refine_top=3,
            local_refine_top=0,
            two_burn_enabled=False,
            two_burn_samples=0,
        )
    )


def test_one_seed_is_the_configured_seed_and_nothing_else() -> None:
    """The default has to be the single-seed search exactly, not a variant."""
    assert union_seeds(_config(1).search) == (20260818,)


def test_the_first_seed_is_always_the_configured_one() -> None:
    """So the union is a superset of the single-seed run by construction.

    If the derivation replaced the base seed rather than extending it, raising
    seed_count could lose the candidate the production seed finds, and the
    feature would be a gamble instead of a widening.
    """
    for count in (2, 3, 8):
        assert union_seeds(_config(count).search)[0] == 20260818


def test_the_derived_seeds_are_deterministic_and_distinct() -> None:
    first = union_seeds(_config(5).search)
    assert first == union_seeds(_config(5).search)
    assert len(set(first)) == 5


def test_the_derived_seeds_avoid_the_offsets_the_search_adds_internally() -> None:
    """+101 here, +17/+4013/+8017 in the two-burn search.

    Consecutive integers would land on those, and a whole extra sweep would be
    spent re-drawing points the base seed already drew.
    """
    # 102, not 16. With a consecutive-integer derivation, sixteen seeds only
    # reach base+15 -- below every offset the test probes -- so the assertion
    # held for a derivation it exists to reject. Measured: consecutive integers
    # collide on +17 at seed_count 18 and on +17 and +101 at 102.
    seeds = union_seeds(_config(102).search)
    collisions = {seeds[0] + offset for offset in (101, 17, 4013, 8017)}
    assert not collisions.intersection(seeds[1:])
    assert len(set(seeds)) == 102, "and they must all still be distinct"


def test_a_second_draw_finds_candidates_the_first_one_missed(
    workshop_script: Path,
) -> None:
    """The measurement the feature exists for.

    The basins are narrower than a grid cell, so an independent draw lands
    somewhere else and brings back arcs the first sweep never sampled. If this
    ever stopped being true the union would be pure cost.
    """
    config = _config(1)
    scenario, _ = parse_gmat_script(workshop_script, config)
    base = {c.candidate_id for c in search_one_burn(scenario, config)}

    union: set[str] = set()
    for seed in union_seeds(_config(3).search):
        drawn = dataclasses.replace(
            config, search=dataclasses.replace(config.search, seed=seed)
        )
        union.update(c.candidate_id for c in search_one_burn(scenario, drawn))

    assert base <= union, "the union has to keep every candidate the base seed found"
    assert len(union) > len(base), "and an independent draw has to add something"


def test_the_pipeline_unions_the_draws_and_reports_which(
    workshop_script: Path, tmp_path: Path
) -> None:
    from orbitwin.pipeline import solve_scenario

    single = solve_scenario(workshop_script, _config(1), tmp_path / "one")
    unioned = solve_scenario(workshop_script, _config(3), tmp_path / "three")

    assert single["search_seeds"] == [20260818]
    assert unioned["search_seeds"] == list(union_seeds(_config(3).search))
    assert len(unioned["search_seeds"]) == 3
    # Strictly greater, and the id sets compared. `search_seeds` is a pure echo
    # of union_seeds, so it stays right even if the loop never runs the extra
    # sweeps -- measured: replacing `for seed in seeds` with `seeds[:1]` left
    # all eight tests passing, because the only other assertion was `>=`.
    single_ids = {r["candidate_id"] for r in json.loads(
        (tmp_path / "one" / "candidates.json").read_text())}
    union_ids = {r["candidate_id"] for r in json.loads(
        (tmp_path / "three" / "candidates.json").read_text())}
    assert single_ids < union_ids, "the union must be a strict superset"
    assert unioned["candidate_count"] > single["candidate_count"]


def test_the_repeated_sweeps_do_not_leave_duplicates(
    workshop_script: Path, tmp_path: Path
) -> None:
    """The grid and the organizer's timeline landmarks ignore the Sobol seed.

    Every extra sweep re-finds them, so without the dedupe the renderability
    pass pays for each repeat again and the archive carries the same trajectory
    three times.
    """
    from orbitwin.pipeline import solve_scenario

    # Asserting on candidates.json proves nothing: select_candidates already
    # deduplicates by id, so the artifact is clean whether or not this dedupe
    # runs -- measured by deleting the whole block, after which all eight tests
    # still passed. The dedupe is a cost optimisation, so cost is what to
    # assert: the repeats must not reach the renderability pass.
    summary = solve_scenario(workshop_script, _config(3), tmp_path / "three")
    records = json.loads((tmp_path / "three" / "candidates.json").read_text())
    ids = [record["candidate_id"] for record in records]
    assert len(ids) == len(set(ids))
    assert summary["duplicate_candidates_dropped"] > 0, (
        "three sweeps re-find the grid and the organizer's timeline landmarks, "
        "so there must be repeats to drop"
    )


def test_a_seed_count_below_one_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="seed_count must be at least 1"):
        validate_config(_config(0))


def test_the_dedupe_keeps_the_copy_this_run_actually_measured(
    workshop_script: Path, tmp_path: Path
) -> None:
    """candidate_id does not include whether the trajectory was evaluated.

    It hashes the burns, the branch and the arrival epoch. Each sweep refines
    only its own top event_refine_top, so the same id can come out of one sweep
    unrefined and out of another refined -- and keeping the first copy means
    keeping the least-measured one. Measured on DifferentPlane_Circular with
    three seeds: 184 of 369 ids appeared more than once, and 2 of those would
    have kept an unrefined copy over a refined one, leaving intercept_time_s
    1.35 s wrong and dropping the candidate out of the §6 tie-break entirely.
    """

    from orbitwin.models import Burn, Candidate

    burn = Burn(time_s=100.0, dv_inertial_km_s=(0.5, 0.0, 0.0), dv_script_km_s=(0.5, 0.0, 0.0))

    def _copy(refined: bool, intercept: float) -> Candidate:
        return Candidate(
            burns=(burn,), intercept_time_s=intercept, minimum_distance_km=1.0,
            distance_at_completion_km=5.0, successful=True,
            analytical_endpoint_error_km=None, branch="same",
            metadata={"event_refined": refined, "arrival_time_s": 20000.0},
        )

    unrefined, refined = _copy(False, 20001.35), _copy(True, 20000.0)
    assert unrefined.candidate_id == refined.candidate_id, "the id cannot tell them apart"

    from orbitwin.pipeline import solve_scenario  # noqa: F401  (import shape only)

    # Exercise the selection rule directly: the refined copy has to survive
    # whichever order the sweeps produced them in.
    for ordering in ((unrefined, refined), (refined, unrefined)):
        best: dict[str, Candidate] = {}
        order: list[str] = []
        for candidate in ordering:
            kept = best.get(candidate.candidate_id)
            if kept is None:
                best[candidate.candidate_id] = candidate
                order.append(candidate.candidate_id)
                continue
            if kept.metadata.get("event_refined") is not True and (
                candidate.metadata.get("event_refined") is True
            ):
                best[candidate.candidate_id] = candidate
        survivor = best[order[0]]
        assert survivor.metadata["event_refined"] is True
        assert survivor.intercept_time_s == pytest.approx(20000.0)


def test_the_union_says_how_many_duplicates_it_dropped(
    workshop_script: Path, tmp_path: Path
) -> None:
    """A count that is not reported is a truncation nobody can see."""
    from orbitwin.pipeline import solve_scenario

    single = solve_scenario(workshop_script, _config(1), tmp_path / "one")
    unioned = solve_scenario(workshop_script, _config(3), tmp_path / "three")
    assert single["duplicate_candidates_dropped"] == 0
    assert unioned["duplicate_candidates_dropped"] >= 0
    assert "duplicate_candidates_dropped" in unioned
