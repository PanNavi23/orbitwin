"""Floors, terminal bands and row gaps flown through simulate_round, not edited facts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from orbitwin.duel.events import coast_one
from orbitwin.duel.gmat_settings import SETTINGS
from orbitwin.duel.params import FLATTENING, R_EQ_KM, R_POLAR_KM, DuelParams, from_dict
from orbitwin.duel.referee import (
    OWN_FLOOR_BAND_KM,
    decide,
    margin_km,
    own_floor_km,
    simulate_round,
)
from orbitwin.duel.state import CraftState, MatchState

from .conftest import MU

FAR = CraftState.of(np.array([0.0, 42164.0, 0.0]), np.array([-math.sqrt(MU / 42164.0), 0.0, 0.0]))


def _params(readings: dict[str, Any] | None = None, choices: dict[str, Any] | None = None,
            **announced: Any) -> DuelParams:
    base = {
        "dv_lim_m_s": 250.0, "r_cap_km": 3.0, "t_round_s": 1800.0,
        "a_lo_km": 6800.0, "a_hi_km": 7000.0,
        "pursuer_r0_km": [6878.0, 0.0, 0.0], "pursuer_v0_km_s": [0.0, 7.6, 0.0],
        "pursuer_epoch": "e",
        "evader_r0_km": [6878.0, 0.0, 0.0], "evader_v0_km_s": [0.0, 7.6, 0.0],
        "evader_epoch": "e",
    }
    return from_dict({"announced": {**base, **announced}, "readings": readings or {},
                      "choices": choices or {}})


def _dipping(periapsis_km: float, e: float = 0.01, before_s: float = 900.0) -> CraftState:
    """A craft that reaches `periapsis_km` `before_s` seconds into the round."""
    speed = math.sqrt(MU * (1.0 + e) / periapsis_km)
    r, v = coast_one(np.array([periapsis_km, 0.0, 0.0]), np.array([0.0, 0.0, speed]),
                     -before_s, MU)
    return CraftState.of(r, v)


def test_the_polar_radius_is_gmat_earth() -> None:
    assert R_EQ_KM == 6378.1363
    assert FLATTENING == 0.0033527
    assert pytest.approx(6356.7523, abs=1e-4) == R_POLAR_KM


@pytest.mark.parametrize(("above_floor_km", "crossed"), [(5e-7, True), (2e-6, False)])
def test_our_floor_includes_the_band_g(above_floor_km: float, crossed: bool) -> None:
    params = _params()
    floor = R_EQ_KM + params.choices.own_margin_km
    assert own_floor_km(params) == floor + OWN_FLOOR_BAND_KM
    state = MatchState(2, "e", FAR, _dipping(floor + above_floor_km))
    facts = simulate_round(state, np.zeros(3), np.zeros(3), params, view="evader")
    assert bool(facts.evader_below["own-floor"]) == crossed
    assert decide(facts, "evader", params).winner == ("pursuer" if crossed else None)


def test_an_opponent_between_the_polar_and_equatorial_radii_has_not_crossed() -> None:
    params = _params()
    between = MatchState(2, "e", FAR, _dipping(R_EQ_KM - 5.0))
    facts = simulate_round(between, np.zeros(3), np.zeros(3), params, view="pursuer")
    assert facts.evader_below["equatorial"] and not facts.evader_below["polar"]
    assert decide(facts, "pursuer", params).winner is None
    # 20 km below R_polar mid-round, for longer than the pair's row gap.
    leo = CraftState.of(np.array([0.0, 7000.0, 0.0]), np.array([-math.sqrt(MU / 7000.0), 0.0, 0.0]))
    under = MatchState(2, "e", leo, _dipping(R_POLAR_KM - 20.0))
    facts = simulate_round(under, np.zeros(3), np.zeros(3), params, view="pursuer")
    (passage,) = facts.evader_below["polar"]
    assert 0.0 < passage.start_s < passage.end_s < 1800.0
    assert passage.duration_s > facts.row_gap_bound_s
    assert decide(facts, "pursuer", params).winner == "pursuer"


def test_an_evader_starting_inside_and_burning_out_has_entered_for_us_as_pursuer() -> None:
    params = _params(readings={"early_entry_wins": True})
    r = np.array([6900.0, 0.0, 0.0])
    v = np.array([0.0, math.sqrt(MU / 6900.0), 0.0])
    state = MatchState(1, "e", FAR, CraftState.of(r, v))
    out = np.array([0.0, 0.25, 0.0])  # raises a well above 7000 km
    facts = simulate_round(state, np.zeros(3), out, params, view="pursuer")
    assert facts.evader_a_initial_km == pytest.approx(6900.0)
    assert facts.evader_a_km > 7100.0
    assert decide(facts, "pursuer", params).winner == "evader"
    facts = simulate_round(state, np.zeros(3), out, params, view="evader")
    assert decide(facts, "evader", params).winner is None


def test_the_live_a_floor_bounds_the_margin_from_below() -> None:
    r, v = np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0])
    assert margin_km(_params(choices={"a_margin_floor_km": 3.0}), r, v, 7000.0) == 3.0
    assert margin_km(_params(), r, v, 7000.0) < 1.0


def test_the_band_covers_the_pre_impulse_state_judged_by_early_entry() -> None:
    # Round 1, as pursuer under early entry: a is judged on the state before
    # the impulse, so its margin must be in the band.
    params = _params(readings={"early_entry_wins": True})
    r = np.array([42164.0, 0.0, 0.0])
    v = np.array([0.0, math.sqrt(MU / 42164.0), 0.0])
    before = margin_km(params, r, v, 42164.0)
    state = MatchState(1, "e", FAR, CraftState.of(r, v))
    dv = np.array([0.0, -0.25, 0.0])  # retrograde: slower, so a smaller rounding term
    facts = simulate_round(state, np.zeros(3), dv, params, view="pursuer")
    after = max(margin_km(params, facts.evader.r0, facts.evader.v0, 42164.0),
                margin_km(params, facts.end_evader.r, facts.end_evader.v, 42164.0))
    assert before > after
    lo = 42164.0 + 0.5 * (before + after)  # inside the band only with the larger margin
    probe = _params(readings={"early_entry_wins": True}, a_lo_km=lo, a_hi_km=lo + 1000.0)
    facts = simulate_round(state, np.zeros(3), dv, probe, view="pursuer")
    assert decide(facts, "pursuer", probe).winner == "evader"


def test_the_row_gap_bound_follows_an_apoapsis_inside_the_round() -> None:
    # a = 10000 km, e = 0.3, starting a little after periapsis: the apoapsis
    # (13000 km) is inside the round and sets the bound.
    params = _params(t_round_s=6000.0)
    c = SETTINGS["cautious"].row_gap_c
    a, rp = 10000.0, 7000.0
    v_p = math.sqrt(MU * (2.0 / rp - 1.0 / a))
    r0, v0 = coast_one(np.array([rp, 0.0, 0.0]), np.array([0.0, v_p, 0.0]), 600.0, MU)
    state = MatchState(1, "e", CraftState.of(np.array([6878.0, 0.0, 0.0]),
                                             np.array([0.0, 7.6, 0.0])), CraftState.of(r0, v0))
    facts = simulate_round(state, np.zeros(3), np.zeros(3), params, view="referee")
    assert facts.row_gap_bound_s == pytest.approx(c * 13000.0**1.5 / math.sqrt(MU), rel=1e-6)
    # Starting just before apoapsis, the apoapsis comes before the next periapsis.
    half = math.pi * math.sqrt(a**3 / MU)
    r1, v1 = coast_one(np.array([rp, 0.0, 0.0]), np.array([0.0, v_p, 0.0]), half - 300.0, MU)
    state = MatchState(1, "e", state.pursuer, CraftState.of(r1, v1))
    facts = simulate_round(state, np.zeros(3), np.zeros(3), params, view="referee")
    assert facts.row_gap_bound_s == pytest.approx(c * 13000.0**1.5 / math.sqrt(MU), rel=1e-6)


def test_a_stated_gap_and_a_known_max_step() -> None:
    state = MatchState(1, "e", CraftState.of(np.array([6878.0, 0.0, 0.0]),
                                             np.array([0.0, 7.6, 0.0])), FAR)
    continuous = _params(choices={"row_gap_s_override": 0.0})
    facts = simulate_round(state, np.zeros(3), np.zeros(3), continuous, view="referee")
    assert facts.row_gap_bound_s == 0.0
    capped = _params(choices={"max_step_s": 60.0})
    facts = simulate_round(state, np.zeros(3), np.zeros(3), capped, view="referee")
    assert facts.row_gap_bound_s == 60.0


def test_their_crossing_counts_for_us_only_below_r_polar_by_the_integration_error() -> None:
    # The evader reaches its periapsis at the round end, held there, so the
    # boundary reading counts it whatever its dwell. GMAT could have it up to
    # e(T) higher, so a dip shallower than that is no crossing for us.
    from orbitwin.duel.referee import opponent_floor_km

    params = _params()
    error = R_POLAR_KM - opponent_floor_km(params, 1800.0)
    assert error == pytest.approx(SETTINGS["cautious"].position_error_km(1800.0))
    leo = CraftState.of(np.array([0.0, 7000.0, 0.0]), np.array([-math.sqrt(MU / 7000.0), 0.0, 0.0]))
    for depth, counted in ((0.5 * error, False), (2.0 * error, True)):
        state = MatchState(2, "e", leo, _dipping(R_POLAR_KM - depth, e=0.001, before_s=1800.0))
        facts = simulate_round(state, np.zeros(3), np.zeros(3), params, view="pursuer")
        assert facts.evader_below["polar"]
        assert decide(facts, "pursuer", params).winner == ("pursuer" if counted else None)
