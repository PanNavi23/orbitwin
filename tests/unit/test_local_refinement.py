from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orbitwin.config import ConfigurationError, OrbitWinConfig, SearchConfig, validate_config
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.search import search_one_burn

# The sweep's grid is coarse against this landscape and the basins are narrower
# than a cell, so which solutions it finds depends on where it lands. Measured
# across five Sobol seeds on the different-plane scenario, the cheapest
# candidate the production profile finds ranges from 543.9 to 648.8 m/s.
#
# Nothing measured fixes that. A denser linspace grid is a different sample set
# rather than a superset -- 14x28 lost three of eight successes on the
# high-plane scenario -- and four times the Sobol samples moved one seed by
# 30 m/s and left the spread unchanged. Looking around what was already found
# is the only option that cannot lose, which is why it is the one that shipped.


def _config(**search: object) -> OrbitWinConfig:
    base = OrbitWinConfig(
        search=SearchConfig(
            one_burn_samples=24, one_burn_grid_burns=5, one_burn_grid_arrivals=8,
            two_burn_enabled=False, event_refine_top=8, keep_candidates=40, workers=1,
        )
    )
    return dataclasses.replace(base, search=dataclasses.replace(base.search, **search))


def _frontier(candidates) -> set[tuple[float, float]]:
    return {
        (float(c.intercept_time_s), float(c.total_dv_km_s))
        for c in candidates
        if c.successful and c.intercept_time_s is not None
    }


def test_looking_around_cannot_lose_what_the_sweep_found(
    different_plane_script: Path,
) -> None:
    """The property none of the other knobs had.

    Widening the sweep resamples, and resampling drops solutions. This only
    adds candidates, around pairs already in hand, so every point the sweep
    reached is still reachable.
    """
    without = _config(local_refine_top=0)
    scenario, _ = parse_gmat_script(different_plane_script, without)
    swept = _frontier(search_one_burn(scenario, without))
    assert swept, "the fixture needs at least one success"

    with_local = _config(local_refine_top=6, local_refine_steps=5)
    refined = _frontier(search_one_burn(scenario, with_local))

    for point in swept:
        assert any(
            other[0] <= point[0] and other[1] <= point[1] for other in refined
        ), f"looking around lost {point}"


def test_it_adds_candidates_rather_than_replacing_them(
    different_plane_script: Path,
) -> None:
    without = _config(local_refine_top=0)
    scenario, _ = parse_gmat_script(different_plane_script, without)
    swept = len(_frontier(search_one_burn(scenario, without)))
    with_local = _config(local_refine_top=6, local_refine_steps=5)
    refined = len(_frontier(search_one_burn(scenario, with_local)))
    assert refined >= swept


def test_it_can_be_turned_off_and_costs_nothing_then(
    different_plane_script: Path,
) -> None:
    off = _config(local_refine_top=0)
    scenario, _ = parse_gmat_script(different_plane_script, off)
    a = _frontier(search_one_burn(scenario, off))
    b = _frontier(search_one_burn(scenario, _config(local_refine_top=0, local_refine_steps=9)))
    assert a == b, "with the top at zero the step count must not matter"


def test_the_settings_are_validated() -> None:
    validate_config(_config(local_refine_top=0))
    validate_config(_config(local_refine_top=1, local_refine_steps=1))
    with pytest.raises(ConfigurationError, match="local_refine_top"):
        validate_config(_config(local_refine_top=-1))
    with pytest.raises(ConfigurationError, match="local_refine_steps"):
        validate_config(_config(local_refine_steps=0))


def test_the_local_refinement_gives_the_same_answer_on_any_worker_count(
    different_plane_script: Path,
) -> None:
    """It used to run its pairs on one core while the sweep used the machine.

    963 of the 2,044 pairs a solve evaluates came from here, all serial, which
    is most of what the worker curve was flattening against: 3.12x on 19
    workers before, 4.07x after, and 14 physical cores now reach the same
    number as 19 logical ones.

    Joining the pool is only safe if it cannot change the answer. The chunks
    are contiguous and collected through map(), which yields in input order, so
    the concatenation is the serial list element for element -- and this is the
    assertion that says so rather than the comment.
    """
    scenario, _ = parse_gmat_script(str(different_plane_script), _config())

    def frontier_for(workers: int) -> set[tuple[float, float]]:
        config = _config(local_refine_top=3, local_refine_steps=3, workers=workers)
        return _frontier(search_one_burn(scenario, config))

    sequential = frontier_for(1)
    assert sequential, "the scenario produced nothing to compare"
    assert frontier_for(2) == sequential
    assert frontier_for(4) == sequential


def test_the_local_refinement_and_the_sweep_share_one_evaluator(
    different_plane_script: Path,
) -> None:
    """Two copies of the chunking would be two places to get it wrong.

    The sweep's parallel path and the refinement's were separate, and only one
    of them was parallel. They are the same function now, so a change to how
    pairs are distributed cannot apply to one and miss the other.
    """
    import inspect

    from orbitwin import search

    source = inspect.getsource(search)
    assert source.count("ProcessPoolExecutor(") == 2, (
        "a third pool site means the pair evaluation has been copied again"
    )
    assert "_evaluate_pairs(" in inspect.getsource(search._look_around)
    assert "_evaluate_pairs(" in inspect.getsource(search.search_one_burn)
