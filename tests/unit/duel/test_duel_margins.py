"""Section 2.5 margins: the measured GMAT table, dr_int on R_cap, the a margin."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from orbitwin.duel.gmat_settings import MEASURED_SPAN_S, SETTINGS, PropagatorSetting
from orbitwin.duel.params import DuelParams, from_dict
from orbitwin.duel.referee import (
    capture_radius,
    decide,
    integration_margin_km,
    margin_km,
    row_gap_bound,
    simulate_round,
)
from orbitwin.duel.state import CraftState, MatchState

from .conftest import MU, Orbit


def _params(choices: dict[str, Any] | None = None, **announced: Any) -> DuelParams:
    base = {
        "dv_lim_m_s": 250.0, "r_cap_km": 3.0, "t_round_s": 600.0,
        "a_lo_km": 6800.0, "a_hi_km": 7000.0,
        "pursuer_r0_km": [6878.0, 0.0, 0.0], "pursuer_v0_km_s": [0.0, 7.6, 0.0],
        "pursuer_epoch": "e",
        "evader_r0_km": [6878.0, 0.0, 0.0], "evader_v0_km_s": [0.0, 7.6, 0.0],
        "evader_epoch": "e",
    }
    return from_dict({"announced": {**base, **announced}, "choices": choices or {}})


def test_the_cautious_row_is_the_worst_of_every_column() -> None:
    measured = {name: row for name, row in SETTINGS.items() if name != "cautious"}
    assert len(measured) == 6
    for f in dataclasses.fields(PropagatorSetting):
        values = [getattr(row, f.name) for row in measured.values()]
        assert all(math.isfinite(v) and v > 0.0 for v in values)
        assert getattr(SETTINGS["cautious"], f.name) == max(values)


def test_the_integration_margin_is_both_craft_on_the_measured_row_with_its_floor() -> None:
    cautious = SETTINGS["cautious"]
    params = _params()
    assert integration_margin_km(params, 0.0) == pytest.approx(2.0 * cautious.position_floor_km)
    expected = 2.0 * (cautious.position_floor_km + cautious.position_rate_km_s * 600.0)
    assert integration_margin_km(params, 600.0) == pytest.approx(expected)
    pd78 = _params({"row_gap_setting": "pd78"})
    assert integration_margin_km(pd78, 600.0) == pytest.approx(
        2.0 * SETTINGS["pd78"].position_error_km(600.0))
    floored = _params({"dr_int_floor_km": 0.01})
    assert integration_margin_km(floored, 600.0) == pytest.approx(0.02)


def test_the_capture_radius_shrinks_for_us_as_pursuer_and_grows_as_evader() -> None:
    params = _params()
    for t in (0.0, 300.0, 600.0):
        margin = integration_margin_km(params, t)
        assert capture_radius(params, "referee")(t) == 3.0
        assert capture_radius(params, "pursuer")(t) == pytest.approx(3.0 - margin)
        assert capture_radius(params, "evader")(t) == pytest.approx(3.0 + margin)
    huge = _params({"dr_int_floor_km": 5.0})
    assert capture_radius(huge, "pursuer")(0.0) == 0.0


def _pair_at(circular: Orbit, separation_km: float) -> MatchState:
    """Two craft on one circle, `separation_km` apart for the whole round."""
    radius = 6878.0
    du = math.degrees(2.0 * math.asin(separation_km / (2.0 * radius)))
    p = circular(radius, 51.6, 10.0, 0.0)
    e = circular(radius, 51.6, 10.0, du)
    return MatchState(1, "e", CraftState.of(*p), CraftState.of(*e))


def test_a_pass_within_the_integration_margin_of_r_cap_is_judged_against_us(
    circular: Orbit,
) -> None:
    params = _params()
    margin = integration_margin_km(params, 0.0)
    for offset, referee, pursuer, evader in (
        (-0.5 * margin, "pursuer", None, "pursuer"),
        (+0.5 * margin, None, None, "pursuer"),
    ):
        state = _pair_at(circular, 3.0 + offset)
        flown = {view: simulate_round(state, np.zeros(3), np.zeros(3), params, view=view)
                 for view in ("referee", "pursuer", "evader")}
        assert (flown["referee"].capture.first_entry_s is not None) == (referee is not None)
        assert decide(flown["pursuer"], "pursuer", params).winner == pursuer
        assert decide(flown["evader"], "evader", params).winner == evader
        assert flown["pursuer"].capture_radius_start_km == pytest.approx(3.0 - margin)
        assert flown["evader"].capture_radius_end_km == pytest.approx(
            3.0 + integration_margin_km(params, 600.0))


def test_the_a_margin_is_twice_the_integration_and_rounding_terms() -> None:
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.8, 0.1])
    params = _params({"a_integration_km": 0.002, "a_margin_min_km": 1e-12,
                      "file_position_km": 1e-3, "file_velocity_km_s": 1e-6})
    half = math.sqrt(3.0) / 2.0
    for a in (7500.0, -50000.0):  # a band edge, either sign
        rounding = (2.0 * a * a / 7000.0**2) * half * 1e-3 \
            + (2.0 * a * a * float(np.linalg.norm(v)) / MU) * half * 1e-6
        assert margin_km(params, r, v, a) == pytest.approx(2.0 * (0.002 + rounding), rel=1e-12)


def test_the_a_integration_term_comes_from_the_measured_row_unless_set() -> None:
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])
    exact = {"file_position_km": 1e-15, "file_velocity_km_s": 1e-18, "a_margin_min_km": 1e-12}
    for name in ("cautious", "pd78", "rk89-default"):
        params = _params({**exact, "row_gap_setting": name})
        assert margin_km(params, r, v, 7000.0) == pytest.approx(2.0 * SETTINGS[name].a_km,
                                                                rel=1e-6)
    longer = _params(exact, t_round_s=2.0 * MEASURED_SPAN_S)
    assert margin_km(longer, r, v, 7000.0) == pytest.approx(4.0 * SETTINGS["cautious"].a_km, rel=1e-6)
    zero = _params({**exact, "a_integration_km": 0.0})
    assert margin_km(zero, r, v, 7000.0) == pytest.approx(1e-12, abs=1e-12)


def test_the_row_gap_bound_is_the_larger_of_the_two_craft(circular: Orbit) -> None:
    params = _params()
    leo = circular(6878.0, 51.6, 10.0, 0.0)
    geo = circular(42164.0, 0.0, 0.0, 0.0)
    c = SETTINGS["cautious"].row_gap_c
    expected = c * 42164.0**1.5 / math.sqrt(MU)
    for p, e in ((leo, geo), (geo, leo)):
        state = MatchState(1, "e", CraftState.of(*p), CraftState.of(*e))
        facts = simulate_round(state, np.zeros(3), np.zeros(3), params, view="referee")
        assert facts.row_gap_bound_s == pytest.approx(expected, rel=1e-9)
        assert row_gap_bound(facts.pursuer, facts.evader, params) == facts.row_gap_bound_s
    stated = _params({"row_gap_s_override": 5.0})
    facts = simulate_round(MatchState(1, "e", CraftState.of(*leo), CraftState.of(*geo)),
                           np.zeros(3), np.zeros(3), stated, view="referee")
    assert facts.row_gap_bound_s == 5.0


def test_the_position_line_grows_as_t_squared_past_the_measured_span() -> None:
    row = SETTINGS["cautious"]
    span = MEASURED_SPAN_S
    assert row.position_error_km(span) == pytest.approx(
        row.position_floor_km + row.position_rate_km_s * span)
    assert row.position_error_km(2.0 * span) == pytest.approx(
        row.position_floor_km + row.position_rate_km_s * 4.0 * span)


def test_the_stored_table_is_one_and_a_half_times_the_recorded_fit() -> None:
    import json
    from pathlib import Path

    recorded = json.loads((Path(__file__).parent / "data" / "gmat_fit_20261102T020420Z.json")
                          .read_text(encoding="utf-8"))
    assert set(recorded) == set(SETTINGS) - {"cautious"}
    for name, fit in recorded.items():
        row = SETTINGS[name]
        for stored, measured in (
            (row.row_gap_c, fit["c_measured"]),
            (row.position_floor_km, fit["position_floor_km_measured"]),
            (row.position_rate_km_s, fit["position_rate_km_s_measured"]),
            (row.a_km, fit["a_drift_km_measured"] + fit["a_method_km_measured"]),
        ):
            assert 1.5 * measured <= stored <= 1.5 * measured * 1.001


def test_the_measured_span_is_the_two_days_the_fit_flies() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import duel_gmat_fit

    assert MEASURED_SPAN_S == duel_gmat_fit.DURATION_S == 2.0 * 86400.0


def test_the_fit_flies_with_the_step_cap_lifted() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import duel_gmat_fit

    # A cap under the measured span would set the largest gaps itself.
    assert duel_gmat_fit.MAX_STEP_S >= MEASURED_SPAN_S
