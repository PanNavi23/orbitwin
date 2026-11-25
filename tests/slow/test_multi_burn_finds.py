"""Does the N-impulse search find an answer, or only reproduce a seeded one?

Separate from `tests/unit/test_multi_burn.py` because answering it means
running a real refinement, and the fast suite is not the place for seventeen
seconds of Nelder-Mead. But it is the property the search exists for: the unit
tests all start from the recorded schedule, so every one of them would pass on
a search that could do nothing but hand that schedule back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import organizer_file

from orbitwin.config import OrbitWinConfig, load_config
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Scenario
from orbitwin.multi_burn import (
    MultiBurnLimits,
    objective,
    refine_start,
    structured_starts,
)

#: What the recorded schedule scores. A start that is not that schedule is not
#: expected to match it, only to get somewhere sensible.
RECORDED_SCORE = 98.3171845866


@pytest.fixture(scope="module")
def odc(project_root: Path) -> tuple[Scenario, OrbitWinConfig, MultiBurnLimits]:
    script = organizer_file("preliminary-problem")
    config = load_config(str(project_root / "config" / "history" / "preliminary" / "odc2026.yaml"))
    scenario, _ = parse_gmat_script(str(script), config)
    return scenario, config, MultiBurnLimits.from_config(scenario, config)


@pytest.mark.slow
def test_a_start_that_is_not_the_answer_reaches_the_feasible_set(
    odc: tuple[Scenario, OrbitWinConfig, MultiBurnLimits],
) -> None:
    """No seed, no knowledge of the schedule -- just a structured guess.

    `n6_coast0_t200` begins needing a 5.28 km/s Lambert completion against a
    1.495 km/s cap, which is to say it begins infeasible like every structured
    start does. Measured on 2026-09-05's scenario it crosses into the feasible
    set within about a thousand evaluations and settles near 98.28, against the
    98.3172 the recorded schedule scores.

    With the flat barrier this was impossible by construction: every infeasible
    schedule returned the same number, so there was nothing to descend.
    """
    scenario, config, limits = odc
    start = next(
        candidate
        for candidate in structured_starts((6,), limits, seed=2026, random_starts=0)
        if candidate.label == "n6_coast0_t200"
    )

    at_start = objective(
        np.asarray(start.vector, dtype=float), start.burn_count, scenario, config, limits
    )
    assert at_start > 0.0, "this start is supposed to begin outside the feasible set"

    solution = refine_start(start, scenario, config, limits, restarts=1, max_iterations=4000)

    assert solution is not None, "the search reached nothing from an unseeded start"
    assert solution.score > 90.0, f"reached the feasible set but only scored {solution.score}"
    assert solution.score <= RECORDED_SCORE + 1.0e-6, (
        "a start beating the recorded schedule is good news, but the fixture and "
        "docs/LESSONS_2026-09-05.md both need updating before it is believed"
    )
    assert solution.minimum_radius_km >= limits.radius_floor_km
    assert solution.arrival_s <= limits.horizon_s
    for burn in solution.burns:
        assert burn.magnitude_km_s <= limits.burn_cap_km_s + 1.0e-12
