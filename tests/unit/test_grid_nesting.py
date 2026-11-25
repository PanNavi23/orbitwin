from __future__ import annotations

import pathlib

import numpy as np
import pytest

from orbitwin.config import load_config
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.search import _MIN_TAIL_INTERVALS, _tail_intervals, _time_samples

CONFIG = pathlib.Path(__file__).resolve().parents[2] / "config"

# A wider profile has to be a superset of a narrower one, or "search harder" is
# a gamble instead of a widening.
#
# It was a gamble. deep.yaml ran a 24x48 grid against standard's 12x24, and
# linspace(a, b, 12) and linspace(a, b, 24) share only their two endpoints --
# ten of standard's twelve burn epochs and twenty-two of its twenty-four
# arrivals simply vanished. Measured on HighPlaneChange_Circular, densifying
# the grid alone made the cheapest solution *worse*, 812.3 -> 922.4 m/s, while
# on DifferentPlane_Circular it made it better, 543.9 -> 489.9. Same change,
# +10% on one scenario and -14% on the other, decided by where the solutions
# happened to sit.
#
# n -> 2n-1 keeps every point, because halving each interval leaves the old
# ones in place. That is the whole fix, and it also has to survive
# dense_window_fraction, which sets the grid's upper bound: change it and every
# point moves except the first.
#
# The grid the profiles name is only the dense part of the window. Beyond
# dense_max _time_samples appends a coarser tail, and that tail was left off the
# ladder: its interval count was one_burn_grid_burns // 3, 7 for standard and 15
# for deep, and 7 does not divide 15. Measured on the workshop scenario, 282 of
# standard's 1687 pairs were missing from deep, all of them in the tail. Nothing
# measured had depended on it -- on both bundled scenarios every Pareto frontier
# point sits in the first fifth of the dense region, thousands of seconds short
# of dense_max -- so closing it changed no frontier. It closes a hole in the
# structure rather than an observed loss.


def _grid(count: int, low: float = 1.0e-3, high: float = 33830.0) -> set[float]:
    # _time_samples rounds to 9 decimals before storing, so compare what it stores.
    return {round(float(value), 9) for value in np.linspace(low, high, count)}


@pytest.mark.parametrize("count", [6, 8, 12, 23, 24, 45, 47])
def test_doubling_the_intervals_keeps_every_point(count: int) -> None:
    assert _grid(count) <= _grid(2 * count - 1)


@pytest.mark.parametrize("count", [12, 23, 24, 47])
def test_doubling_the_points_keeps_only_the_endpoints(count: int) -> None:
    """The trap. 24 looks like "twice 12" and is a different sample set."""
    assert len(_grid(count) & _grid(2 * count)) == 2


def test_deep_is_a_superset_of_standard() -> None:
    """The property the profile names promise.

    Checked on the configured numbers rather than on a search, so it fails at
    the line someone edits rather than three scenarios later.
    """
    standard = load_config(CONFIG / "standard.yaml").search
    deep = load_config(CONFIG / "deep.yaml").search

    assert deep.one_burn_grid_burns == 2 * standard.one_burn_grid_burns - 1
    assert deep.one_burn_grid_arrivals == 2 * standard.one_burn_grid_arrivals - 1
    # dense_window_fraction sets dense_max = tmax * fraction, the grid's upper
    # bound. A different bound moves every interior point, so nesting the counts
    # is not enough on its own.
    assert deep.dense_window_fraction == standard.dense_window_fraction
    # Sobol is nested by construction, so more samples can only add points.
    assert deep.one_burn_samples >= standard.one_burn_samples


def test_the_deeper_profile_really_samples_a_superset(workshop_script) -> None:
    """The end-to-end version: same scenario, both profiles, compare the pairs.

    Every pair, tail included. The tail used to be excluded here because its
    interval count was one_burn_grid_burns // 3, which does not double: 23//3 =
    7 against 45//3 = 15, and 7 does not divide 15. Measured on this scenario,
    282 of standard's 1687 pairs -- six of its seven tail burn epochs times all
    47 arrivals -- were absent from deep. _tail_intervals doubles instead, and
    the count is now zero.
    """
    standard = load_config(CONFIG / "standard.yaml")
    deep = load_config(CONFIG / "deep.yaml")
    scenario, _ = parse_gmat_script(workshop_script, standard)

    narrow = set(_time_samples(scenario, standard))
    wide = set(_time_samples(scenario, deep))
    dense_max = scenario.max_time_s * standard.search.dense_window_fraction

    # Named separately so a regression says which half of the window broke.
    missing = narrow - wide
    missing_tail = {pair for pair in missing if pair[0] > dense_max}
    assert not missing, (
        f"{len(missing)} of {len(narrow)} pairs are not in deep, "
        f"{len(missing_tail)} of them beyond dense_max"
    )
    # Guard the guard: an empty tail would make the assertion above vacuous for
    # the region it was added to cover.
    assert {pair for pair in narrow if pair[0] > dense_max}


@pytest.mark.parametrize("script", ["high_plane_script", "different_plane_script"])
def test_both_bundled_scenarios_nest_too(script: str, request) -> None:
    """The workshop template is not the only shape the tail has to nest in.

    dense_max and burn_max come from the scenario horizon, so a scenario with a
    different period puts the tail somewhere else entirely.
    """
    standard = load_config(CONFIG / "standard.yaml")
    deep = load_config(CONFIG / "deep.yaml")
    path = request.getfixturevalue(script)
    scenario, _ = parse_gmat_script(path, standard)

    narrow = set(_time_samples(scenario, standard))
    assert not narrow - set(_time_samples(scenario, deep))


@pytest.mark.parametrize("count", [5, 6, 7, 8, 12, 16, 23, 24, 45, 47, 93, 100, 511])
def test_the_tail_interval_count_doubles_with_the_dense_one(count: int) -> None:
    """The tail rides the same ladder, without knowing which rung it is on.

    linspace(a, b, n) sits inside linspace(a, b, m) exactly when n-1 divides
    m-1, so the tail nests only if its interval count doubles when the dense
    one does. one_burn_grid_burns // 3 did not, and a fix anchored on 23 and 45
    would break one rung later: (2d) // 3 == 2 * (d // 3) fails whenever
    d % 3 == 2, and deep's 44 dense intervals are exactly that case.
    """
    assert _tail_intervals(2 * count - 1) == 2 * _tail_intervals(count)


@pytest.mark.parametrize("count", [5, 6, 8, 12, 23, 45])
def test_the_tail_points_themselves_nest(count: int) -> None:
    """The property the interval count exists to buy, on real tail bounds.

    dense_max and burn_max for standard and deep on DifferentPlane_Circular.
    Compared after round(x, 9), which is what _time_samples stores.
    """
    low, high = 28890.406938, 33888.714487
    narrow = _grid(_tail_intervals(count) + 1, low, high)
    wide = _grid(_tail_intervals(2 * count - 1) + 1, low, high)
    assert narrow <= wide


def test_the_tail_floor_is_not_lowered() -> None:
    """Two intervals minimum, so the tail always has an interior point.

    The doubling identity does not hold below five burns, where a dense grid of
    at most three intervals asks for fewer intervals than the floor allows.
    That is the floor binding, not the ladder failing. It is the floor the
    previous expression had, and lowering it to make the identity hold
    everywhere would thin the tail of exactly the smallest profiles to tidy a
    statement no shipped profile reaches.
    """
    for count in (2, 3, 4):
        assert _tail_intervals(count) == _MIN_TAIL_INTERVALS


def test_only_the_two_ladder_profiles_and_ood_test_change() -> None:
    """Every other burn count keeps the answer one_burn_grid_burns // 3 gave.

    That is the point rather than a coincidence. quick's 6 and the 2, 3, 4 and
    5 that tests build configs with sit on the floor either way, and the 12
    SearchConfig defaults to gets 4 either way, so no test that constructs an
    OrbitWinConfig() sees a different sample set. Only ood-test's 8, standard's
    23 and deep's 45 move, one tail interval upward each.
    """
    for count in (2, 3, 4, 5, 6, 7, 12, 13, 24):
        assert _tail_intervals(count) == max(2, count // 3)
    for count, expected in ((8, 4), (23, 8), (45, 16)):
        assert _tail_intervals(count) == expected


def test_quick_and_ood_test_are_deliberately_off_the_ladder() -> None:
    """They are not narrower standards, and pretending otherwise buys nothing.

    Nesting needs the sample *window* to agree, not just the counts.
    dense_max = tmax * dense_window_fraction and burn_max = tmax - min_tof_s
    are both endpoints of the grids, and quick and ood-test disagree with
    standard on both: 0.80 and 0.90 against 0.85, 120 s and 60 s against 100 s.
    Measured on both bundled scenarios, 4 of quick's 168 pairs are in
    standard's set today, and putting only the counts on the ladder would leave
    that number exactly where it is -- every interior point would still land
    somewhere else. Aligning the window as well would work and would take quick
    from 168 pairs to 444, its dense grid from 72 to 288. quick exists so that
    preflight.sh and release_validate.sh finish, so it keeps its own window and
    stays off the ladder on purpose.
    """
    standard = load_config(CONFIG / "standard.yaml").search
    for name in ("quick.yaml", "ood-test.yaml"):
        search = load_config(CONFIG / name).search
        assert (
            search.dense_window_fraction != standard.dense_window_fraction
            or search.min_tof_s != standard.min_tof_s
        ), f"{name} now shares standard's window; its counts should join the ladder too"
