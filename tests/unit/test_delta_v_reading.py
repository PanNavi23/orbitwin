from __future__ import annotations

import math

import pytest

from orbitwin.models import Burn, Candidate

# 初賽規則 §5 eq. 1 writes ΔVteam = Σk ΔVk and never defines ΔVk. Read as the
# sum of impulse magnitudes it is fuel; read as the magnitude of the vector sum,
# two opposing impulses would nearly cancel and cost almost nothing. Docs Q14
# asks the organizer. We score the first and report the second, so the answer
# costs a re-rank rather than a re-search.


def _candidate(*vectors: tuple[float, float, float]) -> Candidate:
    burns = tuple(
        Burn(time_s=100.0 * (index + 1), dv_inertial_km_s=vector, dv_script_km_s=vector)
        for index, vector in enumerate(vectors)
    )
    return Candidate(
        burns=burns,
        intercept_time_s=20000.0,
        minimum_distance_km=1.0,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=None,
        branch="test",
    )


def test_one_impulse_makes_the_two_readings_the_same_number() -> None:
    """Which is why the ambiguity cannot touch a single-burn candidate.

    Measured on a real archive: all eight Pareto frontier points were
    single-burn, so the reading changed nothing about what we would submit.
    """
    candidate = _candidate((0.3, -0.4, 0.0))
    assert candidate.total_dv_km_s == pytest.approx(0.5)
    assert candidate.vector_sum_dv_km_s == pytest.approx(candidate.total_dv_km_s)


def test_opposing_impulses_are_charged_in_full_by_the_reading_we_score() -> None:
    """The whole point. Fuel does not come back when you burn retrograde."""
    candidate = _candidate((1.0, 0.0, 0.0), (-0.9, 0.0, 0.0))
    assert candidate.total_dv_km_s == pytest.approx(1.9)
    assert candidate.vector_sum_dv_km_s == pytest.approx(0.1)


def test_the_scored_reading_is_never_below_the_other_one() -> None:
    """The triangle inequality, which is what makes our number safe.

    We can under-rate our own submission if the organizer means the vector sum.
    We can never over-rate it, so no candidate is submitted claiming a total the
    organizer would compute as larger.
    """
    for vectors in (
        ((1.0, 0.0, 0.0), (-0.9, 0.0, 0.0)),
        ((0.4, 0.5, -0.2), (0.1, -0.7, 0.3), (-0.2, 0.1, 0.6)),
        ((0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
        ((1.2, -0.3, 0.4),),
    ):
        candidate = _candidate(*vectors)
        assert candidate.vector_sum_dv_km_s <= candidate.total_dv_km_s + 1e-12


def test_same_direction_impulses_agree_because_nothing_cancels() -> None:
    candidate = _candidate((0.5, 0.0, 0.0), (0.5, 0.0, 0.0))
    assert candidate.total_dv_km_s == pytest.approx(1.0)
    assert candidate.vector_sum_dv_km_s == pytest.approx(1.0)


def test_both_readings_reach_the_archive() -> None:
    """A number nobody can see answers no question on the day."""
    payload = _candidate((1.0, 0.0, 0.0), (-0.9, 0.0, 0.0)).to_dict()
    assert payload["total_dv_km_s"] == pytest.approx(1.9)
    assert payload["vector_sum_dv_km_s"] == pytest.approx(0.1)


def test_the_per_impulse_limit_is_unaffected_by_the_reading() -> None:
    """§2 bounds each ‖Δvk‖, so a submission legal under one is legal under both.

    Measured: the six multi-burn candidates in a real archive peaked at 1462.3
    m/s on a single impulse, inside the 1500 m/s limit, while their two readings
    differed by up to 4.7x.
    """
    candidate = _candidate((1.4001, 0.0, 0.0), (-1.3611, 0.0, 0.0))
    assert max(burn.magnitude_km_s for burn in candidate.burns) <= 1.5
    assert candidate.total_dv_km_s == pytest.approx(2.7612)
    assert candidate.vector_sum_dv_km_s == pytest.approx(0.039, abs=1e-3)
    assert not math.isclose(candidate.total_dv_km_s, candidate.vector_sum_dv_km_s)


@pytest.mark.parametrize("components", [2, 4], ids=["too-few", "too-many"])
def test_a_burn_vector_that_is_not_three_components_is_refused(components: int) -> None:
    """Two numbers disagreeing about the same delta-v is worse than a crash.

    Measured on the unvalidated reader: two components produced a plausible
    total_dv_km_s and then raised a bare IndexError from inside to_dict() while
    *writing* the archive. Four were worse and silent -- total_dv_km_s 9.0139
    km/s against vector_sum_dv_km_s 0.5000, computed over different data, in one
    record. The docstring claims the triangle inequality holds between them; it
    only holds if both read the same three numbers.
    """
    from orbitwin.io import candidate_from_dict

    vector = [0.3, 0.4, 0.0, 9.0][:components]
    record = {
        "burns": [
            {"time_s": 1.0, "dv_inertial_km_s": vector, "dv_script_km_s": vector}
        ],
        "successful": True,
        "branch": "x",
    }
    with pytest.raises(ValueError, match="exactly 3 components"):
        candidate_from_dict(record)
