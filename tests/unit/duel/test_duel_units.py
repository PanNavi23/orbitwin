"""The conversion layer, the projection, the grid and the submission line."""

from __future__ import annotations

import numpy as np
import pytest

from orbitwin.duel.units import (
    Impulse,
    SubmissionError,
    collect_round,
    executed_for,
    executed_km_s,
    format_line,
    parse_line,
    project_m_s,
    snap,
)


def test_an_impulse_over_the_cap_is_scaled_onto_it() -> None:
    assert project_m_s(np.array([300.0, 400.0, 0.0]), 250.0) == pytest.approx([150.0, 200.0, 0.0])


def test_zero_and_an_impulse_under_the_cap_are_unchanged() -> None:
    assert project_m_s(np.zeros(3), 250.0).tolist() == [0.0, 0.0, 0.0]
    assert project_m_s(np.array([10.0, -20.0, 5.0]), 250.0).tolist() == [10.0, -20.0, 5.0]


def test_projection_is_radial_not_per_component() -> None:
    # Per-component clipping at 250 would leave [250, 250, 0], norm 353.6.
    projected = project_m_s(np.array([400.0, 400.0, 0.0]), 250.0)
    assert np.linalg.norm(projected) == pytest.approx(250.0)
    assert projected[0] == pytest.approx(projected[1])


def test_metres_and_kilometres_round_trip() -> None:
    vector = np.array([123.45, -6.78, 0.01])
    assert executed_km_s(vector, 250.0) * 1000.0 == pytest.approx(vector, abs=1e-12)
    assert Impulse((12345, -678, 1)).km_s * 1000.0 == pytest.approx(vector, abs=1e-12)


def test_a_grid_vector_snaps_to_itself() -> None:
    # A float truncation (int(x * 100)) changes several per cent of these.
    for whole in range(0, 100_001, 7):
        value = whole / 100.0
        assert snap(np.array([value, 0.0, 0.0]), 2000.0).centi == (whole, 0, 0)


def test_snap_keeps_the_norm_within_the_cap_and_moves_little() -> None:
    rng = np.random.default_rng(20261102)
    for cap in (250.0, 250.005, 99.997):
        for _ in range(2000):
            vector = rng.normal(size=3) * cap
            impulse = snap(vector, cap)
            assert impulse.norm_centi_squared() <= (cap * 100.0) ** 2
            moved = np.linalg.norm(impulse.m_s - project_m_s(vector, cap))
            assert moved <= 0.0174


def test_the_line_has_exactly_two_decimals_and_reads_back() -> None:
    impulse = Impulse((15000, -20000, 0))
    line = format_line("pursuer", 3, impulse)
    assert line == "追捕方, 3, 150.00, -200.00, 0.00"
    parsed = parse_line(line)
    assert parsed.role == "pursuer" and parsed.round_number == 3
    assert parsed.impulse == impulse


def test_negative_zero_is_never_written() -> None:
    assert format_line("evader", 1, Impulse((0, 0, 0))) == "逃脫方, 1, 0.00, 0.00, 0.00"
    assert snap(np.array([-0.001, 0.0, 0.0]), 250.0).centi == (0, 0, 0)
    assert format_line("evader", 1, Impulse((-1, 0, 0))) == "逃脫方, 1, -0.01, 0.00, 0.00"


def test_full_width_commas_and_more_decimals_are_read() -> None:
    parsed = parse_line("逃脫方\uff0c2\uff0c1.234\uff0c0.50\uff0c-3.00")
    assert parsed.vector_m_s == (1.234, 0.5, -3.0)
    assert parsed.impulse is None


def test_a_line_with_fewer_than_two_decimals_is_refused() -> None:
    with pytest.raises(SubmissionError, match="at least 2"):
        parse_line("追捕方, 1, 1.5, 0.00, 0.00")


def test_a_second_line_for_the_same_role_is_refused() -> None:
    with pytest.raises(SubmissionError, match="second pursuer"):
        collect_round(["追捕方, 2, 1.00, 0.00, 0.00", "追捕方, 2, 2.00, 0.00, 0.00"], 2)
    with pytest.raises(SubmissionError, match="round 3 in round 2"):
        collect_round(["逃脫方, 3, 1.00, 0.00, 0.00"], 2)


def test_a_missing_line_runs_as_zero() -> None:
    assert executed_for(None, 250.0).tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "line",
    [
        "追捕方, ², 1.00, 0.00, 0.00",
        f"追捕方, {chr(0xFF13)}, 1.00, 0.00, 0.00",  # full-width 3
        f"追捕方, 1, {chr(0xFF11)}.{chr(0xFF10) * 2}, 0.00, 0.00",  # full-width 1.00
        f"追捕方, 1, {chr(0x0661)}.{chr(0x0660) * 2}, 0.00, 0.00",  # Arabic-Indic 1.00
        "追捕方, 1, 1.00, 0.00, 0.00,",
        "追捕方, 1, 1.00, 0.00, 0.0\n0",
    ],
)
def test_lines_with_other_digits_or_fields_are_refused(line: str) -> None:
    with pytest.raises(SubmissionError):
        parse_line(line)


def test_halfway_components_snap_to_the_smaller_norm() -> None:
    assert snap(np.array([0.125, -0.125, 0.375]), 250.0).centi == (12, -12, 37)
    assert snap(np.array([-0.005, 0.005, 0.0]), 250.0).centi == (0, 0, 0)


def test_every_grid_vector_within_the_cap_snaps_to_itself() -> None:
    rng = np.random.default_rng(20261102)
    cap = 250.0
    limit = int(cap * 100)
    vectors = [(15000, 20000, 0), (0, 0, limit), (-limit, 0, 0), (1, -1, 0)]
    while len(vectors) < 3000:
        centi = tuple(int(x) for x in rng.integers(-limit, limit + 1, size=3))
        if sum(c * c for c in centi) <= limit * limit:
            vectors.append(centi)
    for centi in vectors:
        assert snap(np.array(centi) / 100.0, cap).centi == centi
        assert parse_line(format_line("evader", 2, Impulse(centi))).impulse == Impulse(centi)


def test_every_component_from_0_to_1000_m_s_snaps_and_reads_back_as_itself() -> None:
    for centi in range(0, 100_001):
        for vector in ((centi, 0, 0), (0, -centi, 0)):
            impulse = snap(np.array(vector) / 100.0, 1000.0)
            assert impulse.centi == vector
    for centi in range(0, 100_001, 7):
        line = format_line("pursuer", 1, Impulse((centi, -centi, 0)))
        assert parse_line(line).impulse == Impulse((centi, -centi, 0))


def test_a_hair_over_the_cap_is_projected() -> None:
    over = np.array([250.0 * 1.00005, 0.0, 0.0])
    assert np.linalg.norm(project_m_s(over, 250.0)) == pytest.approx(250.0, rel=1e-15)


def test_trailing_zeros_past_two_decimals_stay_on_the_grid() -> None:
    parsed = parse_line("逃脫方, 4, 1.2300, -0.0500, 12.000000")
    assert parsed.impulse == Impulse((123, -5, 1200))
    assert parse_line("逃脫方, 4, 1.2301, 0.00, 0.00").impulse is None


def test_a_late_line_runs_as_the_rule_default_and_an_over_cap_one_is_scaled() -> None:
    late = executed_for(None, 250.0)
    assert late.tolist() == [0.0, 0.0, 0.0]
    over = executed_for(parse_line("逃脫方, 2, 300.00, 400.00, 0.00"), 250.0)
    assert over == pytest.approx([0.150, 0.200, 0.0])
    under = executed_for(parse_line("追捕方, 2, 3.00, -4.00, 0.00"), 250.0)
    assert under == pytest.approx([0.003, -0.004, 0.0])


@pytest.mark.parametrize("cap", [4.35, 0.29, 1.15, 2.01, 250.0, 1000.0])
def test_a_grid_vector_at_a_grid_cap_snaps_to_itself(cap: float) -> None:
    centi = round(cap * 100)
    assert snap(np.array([cap, 0.0, 0.0]), cap).centi == (centi, 0, 0)
    assert snap(np.array([0.0, -cap, 0.0]), cap).centi == (0, -centi, 0)


def test_every_grid_cap_keeps_its_own_grid_point() -> None:
    for centi in range(1, 100_001):
        cap = centi / 100.0
        assert snap(np.array([cap, 0.0, 0.0]), cap).centi == (centi, 0, 0)


def test_a_vector_a_hair_over_the_cap_is_scaled_onto_it() -> None:
    over = np.array([0.0, 250.0 * (1.0 + 1e-10), 0.0])
    assert float(np.linalg.norm(project_m_s(over, 250.0))) <= 250.0


@pytest.mark.parametrize("round_field", ["1.00", "0", "7", "-1", "+1"])
def test_a_round_that_is_not_1_to_6_written_plainly_is_refused(round_field: str) -> None:
    with pytest.raises(SubmissionError):
        parse_line(f"追捕方, {round_field}, 1.00, 0.00, 0.00")


@pytest.mark.parametrize("round_number", [0, 7])
def test_format_line_refuses_a_round_outside_the_match(round_number: int) -> None:
    with pytest.raises(SubmissionError):
        format_line("pursuer", round_number, Impulse((1, 0, 0)))


def test_a_corner_one_centi_squared_over_the_cap_is_not_taken() -> None:
    # Cap 5 centi: the nearest corner (5, 1) has 26 > 25 centi^2.
    impulse = snap(np.array([0.04903, 0.00981, 0.0]), 0.05)
    assert impulse.norm_centi_squared() <= 25
