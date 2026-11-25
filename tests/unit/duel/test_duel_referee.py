"""Verdicts: event orders, dwell rules by role, floors, the terminal judgement."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from orbitwin.duel.events import Passage
from orbitwin.duel.params import R_EQ_KM, DuelParams, from_dict
from orbitwin.duel.referee import (
    Band,
    Reading,
    RefereeError,
    RoundFacts,
    View,
    band_for,
    decide,
    established,
    judge,
    simulate_round,
)
from orbitwin.duel.state import CraftState, MatchState
from orbitwin.duel.units import executed_for

from .conftest import MU, Orbit


def _params(**announced: Any) -> DuelParams:
    raw = {
        "announced": {
            "dv_lim_m_s": 250.0,
            "r_cap_km": 3.0,
            "t_round_s": 1800.0,
            "a_lo_km": 6800.0,
            "a_hi_km": 7000.0,
            "pursuer_r0_km": [6878.0, 0.0, 0.0],
            "pursuer_v0_km_s": [0.0, 7.6, 0.0],
            "pursuer_epoch": "e",
            "evader_r0_km": [6878.0, 0.0, 0.0],
            "evader_v0_km_s": [0.0, 7.6, 0.0],
            "evader_epoch": "e",
            **announced,
        }
    }
    return from_dict(raw)


def _crossing_state(circular: Orbit, round_number: int = 1) -> MatchState:
    p = circular(6878.0, 51.6, 10.0, -20.0)
    e = circular(6878.0, 53.6, 10.0, -19.98)
    return MatchState(round_number, "e", CraftState.of(*p), CraftState.of(*e))


def _neutral(**overrides: Any) -> Reading:
    values: dict[str, Any] = {
        "order": "priority",
        "capture_dwell_s": 0.0,
        "pursuer_crossing_dwell_s": 0.0,
        "evader_crossing_dwell_s": 0.0,
        "pursuer_surface": "equatorial",
        "evader_surface": "equatorial",
        "capture_passages": "favour",
    }
    values.update(overrides)
    return Reading(**values)


def _facts(capture: list[Passage], pursuer_below: list[Passage], evader_below: list[Passage],
           circular: Orbit, round_number: int = 1, evader_a: float = 6878.0,
           view: View = "referee") -> RoundFacts:
    facts = simulate_round(_crossing_state(circular, round_number), np.zeros(3), np.zeros(3),
                           _params(), view=view)
    capture_facts = dataclasses.replace(facts.capture, passages_favour=capture,
                                        passages_hurt=capture)
    names = ("equatorial", "polar", "own-floor", "opponent-floor")
    below = dict.fromkeys(names, pursuer_below)
    below_e = dict.fromkeys(names, evader_below)
    return dataclasses.replace(facts, capture=capture_facts, pursuer_below=below,
                               evader_below=below_e, evader_a_km=evader_a)


def test_the_crossing_capture_is_refereed(circular: Orbit) -> None:
    facts = simulate_round(_crossing_state(circular), np.zeros(3), np.zeros(3), _params(),
                           view="referee")
    outcome = judge(facts, _neutral(), Band(6800.0, 7000.0))
    assert outcome.winner == "pursuer"
    assert outcome.time_s == pytest.approx(308.45, abs=0.02)


def test_a_short_capture_counts_against_us_as_evader_not_for_us_as_pursuer(
    circular: Orbit,
) -> None:
    # The 2-degree pass spends 13.6 s inside 3 km: under any row gap above that.
    params = _params()
    state = _crossing_state(circular)
    for role in ("pursuer", "evader"):
        facts = simulate_round(state, np.zeros(3), np.zeros(3), params, view=role)
        assert facts.row_gap_bound_s > 14.0
        expected = "pursuer" if role == "evader" else None
        assert decide(facts, role, params).winner == expected


def test_dwell_is_per_passage_and_a_round_end_counts_whatever_its_dwell() -> None:
    two_short = [Passage(100.0, 160.0), Passage(400.0, 460.0)]
    assert established(two_short, 100.0, 1800.0, True) is None
    assert established([Passage(100.0, 220.0)], 100.0, 1800.0, True) == pytest.approx(200.0)
    assert established([Passage(1790.0, 1800.0)], 100.0, 1800.0, True) == pytest.approx(1800.0)
    assert established([Passage(0.0, 5.0)], 100.0, 1800.0, True) == 0.0


def test_boundary_events_need_the_dwell_when_boundary_rows_are_not_judged() -> None:
    assert established([Passage(1790.0, 1800.0)], 100.0, 1800.0, False) is None
    assert established([Passage(0.0, 5.0)], 100.0, 1800.0, False) is None
    assert established([Passage(0.0, 150.0)], 100.0, 1800.0, False) == pytest.approx(100.0)


def test_a_counted_event_is_timed_by_the_passage_that_qualifies() -> None:
    gap = 100.0
    # A 0.5 gap dip, then a 1.2 gap one: counts at the second entry plus the gap.
    passages = [Passage(300.0, 350.0), Passage(600.0, 720.0)]
    assert established(passages, gap, 1800.0, True) == pytest.approx(700.0)
    # A dip that begins 0.3 gap before the round end and holds there.
    assert established([Passage(1770.0, 1800.0)], gap, 1800.0, True) == pytest.approx(1800.0)
    # Hurting events count at their first entry.
    assert established(passages, 0.0, 1800.0, True) == pytest.approx(300.0)


def test_a_boundary_only_win_is_marked(circular: Orbit) -> None:
    facts = _facts([Passage(1795.0, 1800.0)], [], [], circular)
    gap = 100.0
    reading = _neutral(capture_dwell_s=gap)
    outcome = judge(facts, reading, Band(1.0, 2.0))
    assert outcome.winner == "pursuer" and outcome.via_boundary
    strict = _neutral(capture_dwell_s=gap, boundary_rows_judged=False)
    assert judge(facts, strict, Band(1.0, 2.0)).winner is None
    long_pass = _facts([Passage(1000.0, 1800.0)], [], [], circular)
    assert not judge(long_pass, reading, Band(1.0, 2.0)).via_boundary


def test_our_floor_carries_the_band_g() -> None:
    from orbitwin.duel.referee import OWN_FLOOR_BAND_KM, own_floor_km

    params = _params()
    assert own_floor_km(params) == pytest.approx(R_EQ_KM + 10.0 + OWN_FLOOR_BAND_KM)


def test_event_orders_on_a_pursuer_that_crosses_then_captures(circular: Orbit) -> None:
    facts = _facts([Passage(500.0, 600.0)], [Passage(200.0, 260.0)], [], circular)
    assert judge(facts, _neutral(order="priority"), Band(1.0, 2.0)).winner == "pursuer"
    assert judge(facts, _neutral(order="chronological"), Band(1.0, 2.0)).winner == "evader"


def test_a_double_crossing(circular: Orbit) -> None:
    with_capture = _facts([Passage(500.0, 600.0)], [Passage(300.0, 360.0)],
                          [Passage(200.0, 260.0)], circular)
    assert judge(with_capture, _neutral(order="priority"), Band(1, 2)).winner == "pursuer"
    assert judge(with_capture, _neutral(order="chronological"), Band(1, 2)).winner == "pursuer"
    without = _facts([], [Passage(300.0, 360.0)], [Passage(200.0, 260.0)], circular)
    assert judge(without, _neutral(order="priority"), Band(1, 2)).winner == "unresolved"
    assert judge(without, _neutral(order="chronological"), Band(1, 2)).winner == "pursuer"


def test_our_floor_beats_a_capture_not_yet_established(circular: Orbit) -> None:
    params = _params()
    gap = 120.0
    entry_before_dip = _facts([Passage(500.0, 900.0)], [Passage(550.0, 600.0)], [], circular,
                              view="pursuer")
    entry_before_dip = dataclasses.replace(entry_before_dip, row_gap_bound_s=gap)
    assert decide(entry_before_dip, "pursuer", params).winner == "evader"
    established_first = _facts([Passage(300.0, 900.0)], [Passage(550.0, 600.0)], [], circular,
                               view="pursuer")
    established_first = dataclasses.replace(established_first, row_gap_bound_s=gap)
    assert decide(established_first, "pursuer", params).winner == "pursuer"


def test_terminal_a_is_judged_on_a_not_on_the_radius(circular: Orbit) -> None:
    round_six = _facts([], [], [], circular, round_number=6, evader_a=6900.0)
    assert judge(round_six, _neutral(), Band(6800.0, 7000.0)).winner == "evader"
    outside = dataclasses.replace(round_six, evader_a_km=7100.0)
    assert judge(outside, _neutral(), Band(6800.0, 7000.0)).winner == "pursuer"


def test_eccentric_evader_whose_radius_is_outside_but_a_inside_wins() -> None:
    params = _params(a_lo_km=7400.0, a_hi_km=7600.0, t_round_s=300.0)
    r = np.array([6878.0, 0.0, 0.0])
    a = 7500.0
    v = np.array([0.0, math.sqrt(MU * (2.0 / 6878.0 - 1.0 / a)), 0.0])
    state = MatchState(6, "e", CraftState.of(r + np.array([0.0, 0.0, 3000.0]), v), CraftState.of(r, v))
    facts = simulate_round(state, np.zeros(3), np.zeros(3), params, view="referee")
    # Radius stays below the interval all round; a sits in its middle.
    assert facts.evader.maximum_radius() < 7400.0
    assert facts.evader_a_km == pytest.approx(7500.0, rel=1e-12)
    assert judge(facts, _neutral(), Band(7400.0, 7600.0)).winner == "evader"
    # The reverse: radius inside the interval, a outside it.
    high = MatchState(6, "e", state.pursuer, CraftState.of(np.array([7500.0, 0.0, 0.0]),
                                                          np.array([0.0, 7.0, 0.0])))
    reverse = simulate_round(high, np.zeros(3), np.zeros(3), params, view="referee")
    assert not 7400.0 <= reverse.evader_a_km <= 7600.0
    assert judge(reverse, _neutral(), Band(7400.0, 7600.0)).winner == "pursuer"


def test_reaching_the_interval_before_round_six_decides_nothing(circular: Orbit) -> None:
    early = _facts([], [], [], circular, round_number=3, evader_a=6900.0)
    assert judge(early, _neutral(), Band(6800.0, 7000.0)).winner is None


def test_the_early_entry_switch_and_its_ordering_by_role(circular: Orbit) -> None:
    facts = _facts([Passage(500.0, 900.0)], [], [], circular, round_number=3, evader_a=6900.0)
    band = Band(6800.0, 7000.0)
    pursuer_view = _neutral(early_entry_wins=True, entry_beats_later_events=True)
    evader_view = _neutral(early_entry_wins=True, entry_beats_later_events=False)
    assert judge(facts, pursuer_view, band).winner == "evader"
    assert judge(facts, evader_view, band).winner == "pursuer"


def test_hyperbolic_and_one_sided_intervals() -> None:
    assert not Band(6800.0, 7000.0).contains(-50000.0)
    assert Band(-60000.0, -40000.0).contains(-50000.0)
    assert Band(8000.0, math.inf, escape_inside=True).contains(-50000.0)
    assert not Band(8000.0, math.inf, escape_inside=False).contains(math.inf)


def test_the_margin_narrows_for_the_evader_and_widens_for_the_pursuer() -> None:
    params = _params()
    r = np.array([6878.0, 0.0, 0.0])
    v = np.array([0.0, 7.6, 0.0])
    ours = band_for(params, "evader", [(r, v)])
    theirs = band_for(params, "pursuer", [(r, v)])
    assert ours.lo_km > 6800.0 and ours.hi_km < 7000.0
    assert theirs.lo_km < 6800.0 and theirs.hi_km > 7000.0
    narrow = _params(a_lo_km=6900.0, a_hi_km=6900.0005)
    core = band_for(narrow, "evader", [(r, v)])
    assert 6900.0 < core.lo_km < core.hi_km < 6900.0005
    assert core.core and not ours.core and not theirs.core


def test_an_initial_separation_inside_the_radius_is_a_capture_at_zero(circular: Orbit) -> None:
    p = circular(6878.0, 51.6, 10.0, 0.0)
    state = MatchState(1, "e", CraftState.of(*p), CraftState.of(p[0] + 1.0, p[1]))
    facts = simulate_round(state, np.zeros(3), np.zeros(3), _params(), view="referee")
    outcome = judge(facts, _neutral(capture_dwell_s=500.0), Band(6800.0, 7000.0))
    assert outcome.winner == "pursuer" and outcome.time_s == 0.0


def test_a_missing_submission_flies_as_zero(circular: Orbit) -> None:
    state = _crossing_state(circular)
    late = simulate_round(state, executed_for(None, 250.0), executed_for(None, 250.0), _params(),
                          view="referee")
    assert late.capture.first_entry_s == pytest.approx(308.45, abs=0.02)


def test_our_own_dip_counts_however_short_and_theirs_only_with_the_gap(
    circular: Orbit,
) -> None:
    params = _params()
    assert R_EQ_KM + params.choices.own_margin_km == pytest.approx(6388.1363)
    short_dip = [Passage(700.0, 705.0)]
    ours_as_evader = dataclasses.replace(_facts([], [], short_dip, circular, view="evader"),
                                         row_gap_bound_s=120.0)
    assert decide(ours_as_evader, "evader", params).winner == "pursuer"
    theirs_as_pursuer = dataclasses.replace(_facts([], [], short_dip, circular, view="pursuer"),
                                            row_gap_bound_s=120.0)
    assert decide(theirs_as_pursuer, "pursuer", params).winner is None


def test_the_energy_check_trips_on_a_corrupted_end_state(
    circular: Orbit, monkeypatch: pytest.MonkeyPatch
) -> None:
    import orbitwin.duel.events as events

    real = events.coast_one

    def corrupt(r: np.ndarray, v: np.ndarray, t: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
        end_r, end_v = real(r, v, t, mu)
        return (end_r, end_v * (1.0 + 1e-6)) if t == 1800.0 else (end_r, end_v)

    monkeypatch.setattr(events, "coast_one", corrupt)
    with pytest.raises(RefereeError, match="energy drifted"):
        simulate_round(_crossing_state(circular), np.zeros(3), np.zeros(3), _params(),
                       view="referee")


@pytest.mark.parametrize("e", [0.9999, 0.99999, 1.000001, 1.0001])
def test_near_parabolic_rounds_pass_the_energy_check(e: float) -> None:
    speed = math.sqrt(MU * (1.0 + e) / 6878.0)
    evader = CraftState.of(np.array([6878.0, 0.0, 0.0]), np.array([0.0, speed, 0.0]))
    pursuer = CraftState.of(np.array([0.0, 42164.0, 0.0]),
                            np.array([-math.sqrt(MU / 42164.0), 0.0, 0.0]))
    params = _params(t_round_s=3600.0, a_lo_km=-1.0e9, a_hi_km=-1.0e3)
    facts = simulate_round(MatchState(2, "e", pursuer, evader), np.zeros(3), np.zeros(3), params,
                           view="referee")
    assert facts.capture.first_entry_s is None


def test_a_far_outbound_hyperbola_is_refereed() -> None:
    r = np.array([-562761.0455600958, 478353.4374472043, 0.0])
    v = np.array([-3.2554531725226434, 2.6262620276855513, 0.0])
    pursuer = CraftState.of(np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0]))
    params = _params(a_lo_km=-1.0e6, a_hi_km=-1.0e3)
    facts = simulate_round(MatchState(6, "e", pursuer, CraftState.of(r, v)), np.zeros(3),
                           np.zeros(3), params, view="evader")
    assert decide(facts, "evader", params).winner == "evader"


def test_an_impulse_over_the_cap_is_refused_by_the_referee(circular: Orbit) -> None:
    over = np.array([0.25 * 1.005, 0.0, 0.0])
    with pytest.raises(RefereeError, match="not projected"):
        simulate_round(_crossing_state(circular), over, np.zeros(3), _params(), view="referee")


def test_the_energy_check_holds_to_1e9_of_the_potential(
    circular: Orbit, monkeypatch: pytest.MonkeyPatch
) -> None:
    import orbitwin.duel.events as events

    real = events.coast_one

    def drift(r: np.ndarray, v: np.ndarray, t: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
        end_r, end_v = real(r, v, t, mu)
        # dE = v^2 x: 3e-9 of v trips 1e-9 mu/r and not 1e-8 mu/r.
        return (end_r, end_v * (1.0 + 3e-9)) if t == 1800.0 else (end_r, end_v)

    monkeypatch.setattr(events, "coast_one", drift)
    with pytest.raises(RefereeError, match="energy drifted"):
        simulate_round(_crossing_state(circular), np.zeros(3), np.zeros(3), _params(),
                       view="referee")


def test_band_ends_are_inside() -> None:
    assert Band(6800.0, 7000.0).contains(6800.0) and Band(6800.0, 7000.0).contains(7000.0)
    assert Band(8000.0, math.inf).contains(8000.0)
    assert not Band(6800.0, 7000.0).contains(7000.000001)


@pytest.mark.parametrize("round_number", [0, 7])
def test_a_round_outside_the_match_is_refused(circular: Orbit, round_number: int) -> None:
    with pytest.raises(RefereeError, match="outside 1 to 6"):
        simulate_round(_crossing_state(circular, round_number), np.zeros(3), np.zeros(3),
                       _params(), view="referee")


def test_the_energy_check_uses_the_energy_itself_on_a_far_open_arc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import orbitwin.duel.events as events

    # 1e7 km out at 3 km/s: |E| = 4.5 km^2/s^2 against mu/r = 0.04.
    r = np.array([1.0e7, 0.0, 0.0])
    v = np.array([0.0, 3.0, 0.0])
    real = events.coast_one

    def noisy(r_: np.ndarray, v_: np.ndarray, t: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
        end_r, end_v = real(r_, v_, t, mu)
        # dE = v^2 x = 9 x: 1e-10 trips mu/r scaling (4e-11) but not |E| (4.5e-9).
        return (end_r, end_v * (1.0 + 1e-10)) if t == 1800.0 else (end_r, end_v)

    monkeypatch.setattr(events, "coast_one", noisy)
    far = CraftState.of(r, v)
    pursuer = CraftState.of(np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0]))
    params = _params(a_lo_km=-1e9, a_hi_km=-1e3)
    simulate_round(MatchState(2, "e", pursuer, far), np.zeros(3), np.zeros(3), params,
                   view="referee")


def test_an_impulse_a_hair_over_the_cap_is_refused_by_the_referee(circular: Orbit) -> None:
    over = np.array([0.0, 0.25 * (1.0 + 1e-9), 0.0])
    with pytest.raises(RefereeError, match="not projected"):
        simulate_round(_crossing_state(circular), over, np.zeros(3), _params(), view="referee")
