"""`decide`: each open reading wired against the role we play, one test per wire."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from orbitwin.duel.events import Passage
from orbitwin.duel.params import DuelParams, from_dict
from orbitwin.duel.referee import (
    RefereeError,
    Role,
    RoundFacts,
    View,
    band_for,
    decide,
    judge,
    margin_km,
    simulate_round,
)
from orbitwin.duel.state import CraftState, MatchState

from .conftest import MU, Orbit

GAP = 120.0
FLOORS = ("equatorial", "polar", "own-floor", "opponent-floor")


def _params(readings: dict[str, Any] | None = None, choices: dict[str, Any] | None = None,
            **announced: Any) -> DuelParams:
    raw: dict[str, Any] = {
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
        },
        "readings": readings or {},
        "choices": choices or {},
    }
    return from_dict(raw)


def _apart(circular: Orbit, params: DuelParams, view: View, round_number: int = 3) -> RoundFacts:
    """Two craft a quarter of an orbit apart: no event of their own."""
    p = circular(6878.0, 51.6, 10.0, 0.0)
    e = circular(6878.0, 51.6, 10.0, 90.0)
    state = MatchState(round_number, "e", CraftState.of(*p), CraftState.of(*e))
    return simulate_round(state, np.zeros(3), np.zeros(3), params, view=view)


def _edit(facts: RoundFacts, *, favour: list[Passage] | None = None,
          hurt: list[Passage] | None = None, pursuer: dict[str, list[Passage]] | None = None,
          evader: dict[str, list[Passage]] | None = None, **fields: Any) -> RoundFacts:
    capture = dataclasses.replace(
        facts.capture,
        passages_favour=favour if favour is not None else [],
        passages_hurt=hurt if hurt is not None else (favour or []),
    )
    below_p = {name: (pursuer or {}).get(name, []) for name in FLOORS}
    below_e = {name: (evader or {}).get(name, []) for name in FLOORS}
    fields.setdefault("row_gap_bound_s", GAP)
    return dataclasses.replace(facts, capture=capture, pursuer_below=below_p,
                               evader_below=below_e, **fields)


def _decide(circular: Orbit, role: Role, params: DuelParams | None = None,
            round_number: int = 3, **edits: Any) -> str | None:
    params = params or _params()
    facts = _edit(_apart(circular, params, role, round_number), **edits)
    return decide(facts, role, params).winner


def test_our_crossing_is_judged_on_our_floor_and_theirs_under_the_polar_radius(
    circular: Orbit,
) -> None:
    short = [Passage(700.0, 705.0)]
    long = [Passage(700.0, 700.0 + 2 * GAP)]
    # As pursuer: our dip below our own floor counts however short.
    assert _decide(circular, "pursuer", pursuer={"own-floor": short}) == "evader"
    assert _decide(circular, "pursuer", pursuer={"polar": short, "equatorial": short}) is None
    # Their dip counts only below R_polar, not below our floor or R_eq.
    assert _decide(circular, "pursuer", evader={"own-floor": long, "equatorial": long,
                                                 "polar": long}) is None
    assert _decide(circular, "pursuer", evader={"opponent-floor": long}) == "pursuer"
    # The same, as evader.
    assert _decide(circular, "evader", evader={"own-floor": short}) == "pursuer"
    assert _decide(circular, "evader", evader={"polar": short, "equatorial": short}) is None
    assert _decide(circular, "evader", pursuer={"own-floor": long, "equatorial": long,
                                                 "polar": long}) is None
    assert _decide(circular, "evader", pursuer={"opponent-floor": long}) == "evader"


def test_their_crossing_needs_the_row_gap_and_ours_does_not(circular: Orbit) -> None:
    half = [Passage(700.0, 700.0 + 0.5 * GAP)]
    over = [Passage(700.0, 700.0 + 1.2 * GAP)]
    assert _decide(circular, "pursuer", evader={"opponent-floor": half}) is None
    assert _decide(circular, "pursuer", evader={"opponent-floor": over}) == "pursuer"
    assert _decide(circular, "evader", pursuer={"opponent-floor": half}) is None
    assert _decide(circular, "evader", pursuer={"opponent-floor": over}) == "evader"
    assert _decide(circular, "evader", evader={"own-floor": half}) == "pursuer"


def test_a_capture_counts_for_us_only_when_proved_and_against_us_when_possible(
    circular: Orbit,
) -> None:
    possible = [Passage(500.0, 900.0)]
    assert _decide(circular, "pursuer", favour=[], hurt=possible) is None
    assert _decide(circular, "evader", favour=[], hurt=possible) == "pursuer"
    assert _decide(circular, "pursuer", favour=possible, hurt=possible) == "pursuer"


def test_a_short_capture_needs_the_gap_for_us_but_not_against_us(circular: Orbit) -> None:
    short = [Passage(500.0, 500.0 + 0.5 * GAP)]
    assert _decide(circular, "pursuer", favour=short) is None
    assert _decide(circular, "evader", favour=short) == "pursuer"


def test_a_double_crossing_without_a_capture_is_our_loss_in_either_role(circular: Orbit) -> None:
    ours_first = [Passage(200.0, 200.0 + 2 * GAP)]
    theirs_later = [Passage(600.0, 600.0 + 2 * GAP)]
    params = _params()
    as_pursuer = _edit(_apart(circular, params, "pursuer"),
                       pursuer={"own-floor": theirs_later}, evader={"opponent-floor": ours_first})
    # Priority leaves it unresolved, chronological gives it to the pursuer;
    # we take the worse, and unresolved is a loss.
    outcome = decide(as_pursuer, "pursuer", params)
    assert outcome.winner == "evader" and "Q4" in outcome.reason
    as_evader = _edit(_apart(circular, params, "evader"),
                      pursuer={"opponent-floor": theirs_later}, evader={"own-floor": ours_first})
    assert decide(as_evader, "evader", params).winner == "pursuer"


def test_early_entry_is_resolved_by_role(circular: Orbit) -> None:
    early = _params(readings={"early_entry_wins": True})
    capture = [Passage(500.0, 900.0)]
    # As pursuer, the entry at the round start beats a later capture.
    assert _decide(circular, "pursuer", early, favour=capture,
                   evader_a_km=6900.0) == "evader"
    # As evader, a later capture beats our entry.
    assert _decide(circular, "evader", early, favour=capture, evader_a_km=6900.0) == "pursuer"
    # A capture already holding at t = 0 is not later than the entry.
    at_start = [Passage(0.0, 900.0)]
    assert _decide(circular, "pursuer", early, favour=at_start,
                   evader_a_km=6900.0) == "pursuer"
    # Without the reading, entering early decides nothing.
    assert _decide(circular, "pursuer", favour=[], evader_a_km=6900.0) is None


def test_the_initial_state_counts_as_an_entry_only_against_us_as_pursuer(
    circular: Orbit,
) -> None:
    early = _params(readings={"early_entry_wins": True})
    fields = {"evader_a_km": 7500.0, "evader_a_initial_km": 6900.0}
    assert _decide(circular, "pursuer", early, round_number=1, **fields) == "evader"
    assert _decide(circular, "evader", early, round_number=1, **fields) is None
    assert _decide(circular, "pursuer", early, round_number=2, **fields) is None


def test_an_escaping_evader_is_inside_a_one_sided_band_only_against_us(circular: Orbit) -> None:
    params = _params(a_lo_km=8000.0, a_hi_km=math.inf)
    for role, expected in (("pursuer", "evader"), ("evader", "pursuer")):
        facts = _edit(_apart(circular, params, role, round_number=6), evader_a_km=-50000.0)
        assert decide(facts, role, params).winner == expected


def test_the_terminal_band_is_narrowed_for_us_as_evader_and_widened_as_pursuer(
    circular: Orbit,
) -> None:
    params = _params()
    facts = _apart(circular, params, "evader", round_number=6)
    end = facts.end_evader
    margin = max(margin_km(params, facts.evader.r0, facts.evader.v0, 6800.0),
                 margin_km(params, end.r, end.v, 6800.0))
    just_inside = 6800.0 + 0.5 * margin
    just_outside = 6800.0 - 0.5 * margin
    assert _decide(circular, "evader", params, round_number=6,
                   evader_a_km=just_inside) == "pursuer"
    assert _decide(circular, "pursuer", params, round_number=6,
                   evader_a_km=just_outside) == "evader"
    assert _decide(circular, "evader", params, round_number=6, evader_a_km=6900.0) == "evader"
    assert _decide(circular, "pursuer", params, round_number=6, evader_a_km=7100.0) == "pursuer"


def test_the_band_takes_the_margin_of_the_round_end_state() -> None:
    # Apoapsis to periapsis on a = 10000 km, e = 0.3: the rounding term of a is
    # larger at periapsis, where the published end state is (faster, lower).
    a, rp, ra = 10000.0, 7000.0, 13000.0
    half_period = math.pi * math.sqrt(a**3 / MU)
    r0 = np.array([-ra, 0.0, 0.0])
    v0 = np.array([0.0, -math.sqrt(MU * (2.0 / ra - 1.0 / a)), 0.0])
    far = (np.array([42164.0, 0.0, 0.0]), np.array([0.0, math.sqrt(MU / 42164.0), 0.0]))
    state = MatchState(6, "e", CraftState.of(*far), CraftState.of(r0, v0))
    probe = _params(t_round_s=half_period)
    facts = simulate_round(state, np.zeros(3), np.zeros(3), probe, view="evader")
    end = facts.end_evader
    assert float(np.linalg.norm(end.r)) == pytest.approx(rp, rel=1e-9)
    at_start = margin_km(probe, r0, v0, a)
    at_end = margin_km(probe, end.r, end.v, a)
    assert at_end > 1.3 * at_start
    lo = a - 0.5 * (at_start + at_end)
    params = _params(t_round_s=half_period, a_lo_km=lo, a_hi_km=lo + 1000.0)
    facts = simulate_round(state, np.zeros(3), np.zeros(3), params, view="evader")
    assert band_for(params, "evader", [(r0, v0)]).contains(facts.evader_a_km)
    assert decide(facts, "evader", params).winner == "pursuer"


def test_decide_refuses_facts_flown_for_another_view(circular: Orbit) -> None:
    params = _params()
    for view, role in (("referee", "pursuer"), ("pursuer", "evader"), ("evader", "pursuer")):
        facts = _apart(circular, params, view)
        with pytest.raises(RefereeError, match="view"):
            decide(facts, role, params)


def test_decide_flags_the_core_band_and_an_exhausted_detector(circular: Orbit) -> None:
    narrow = _params(a_lo_km=6900.0, a_hi_km=6900.0005)
    facts = _apart(circular, narrow, "evader", round_number=6)
    centred = _edit(facts, evader_a_km=6900.00025)
    outcome = decide(centred, "evader", narrow)
    assert outcome.core_band and outcome.winner == "evader"
    assert not outcome.detector_exhausted
    tired = dataclasses.replace(centred,
                                capture=dataclasses.replace(centred.capture, exhausted=True))
    assert decide(tired, "evader", narrow).detector_exhausted
    assert not decide(_apart(circular, _params(), "evader", 6), "evader", _params()).core_band


def test_chronological_ties_go_to_the_capture_then_are_unresolved_between_crossings(
    circular: Orbit,
) -> None:
    from orbitwin.duel.referee import Band, Reading

    params = _params()
    facts = _edit(_apart(circular, params, "referee"), favour=[Passage(500.0, 900.0)],
                  pursuer={"equatorial": [Passage(500.0, 600.0)]})
    reading = Reading("chronological", 0.0, 0.0, 0.0, "equatorial", "equatorial", "favour")
    assert judge(facts, reading, Band(1.0, 2.0)).winner == "pursuer"
    both = _edit(_apart(circular, params, "referee"),
                 pursuer={"equatorial": [Passage(500.0, 600.0)]},
                 evader={"equatorial": [Passage(500.0, 600.0)]})
    assert judge(both, reading, Band(1.0, 2.0)).winner == "unresolved"
    first = _edit(_apart(circular, params, "referee"),
                  pursuer={"equatorial": [Passage(500.0, 600.0)]},
                  evader={"equatorial": [Passage(499.0, 600.0)]})
    assert judge(first, reading, Band(1.0, 2.0)).winner == "pursuer"


def test_a_verdict_that_rests_on_a_boundary_row_in_either_order_is_flagged(
    circular: Orbit,
) -> None:
    # A capture at 400 s wins under priority whatever the rows. Under the
    # chronological order the evader's 5 s dip at the round start beats our
    # own dip at 200 s only if the start row is judged.
    params = _params()
    facts = _edit(_apart(circular, params, "pursuer"),
                  favour=[Passage(400.0, 400.0 + 2 * GAP)],
                  evader={"opponent-floor": [Passage(0.0, 5.0)]},
                  pursuer={"own-floor": [Passage(200.0, 201.0)]})
    outcome = decide(facts, "pursuer", params)
    assert outcome.winner == "pursuer"
    strict = _params(readings={"boundary_rows_judged": False})
    assert decide(facts, "pursuer", strict).winner == "evader"
    assert outcome.via_boundary
    plain = _edit(_apart(circular, params, "pursuer"), evader={"opponent-floor": [Passage(300.0, 600.0)]})
    assert decide(plain, "pursuer", params).winner == "pursuer"
    assert not decide(plain, "pursuer", params).via_boundary


def test_a_state_far_from_the_band_gets_the_margin_of_the_band_edge() -> None:
    # Found by the fourth code review: with the margin taken at the state's
    # own a, a barely hyperbolic evader (a = -1e10 km) got a margin of 1.6e10
    # km and counted as inside [9000, 9500].
    params = _params(a_lo_km=9000.0, a_hi_km=9500.0, t_round_s=3600.0, dv_lim_m_s=5000.0)
    pursuer = CraftState((0.0, -9000.0, 1000.0), (math.sqrt(MU / 9000.0), 0.0, 0.0))
    r = (7000.0, 0.0, 0.0)
    escaping = CraftState(r, (0.0, math.sqrt(MU * (2.0 / 7000.0 + 1.0 / 1e10)), 0.0))
    state = MatchState(6, "e", pursuer, escaping)
    facts = simulate_round(state, np.zeros(3), np.zeros(3), params, view="pursuer")
    assert decide(facts, "pursuer", params).winner == "pursuer"
    # An evader starting on a = 1e6 km and braking to a = 9100 km is inside.
    wide = CraftState(r, (0.0, math.sqrt(MU * (2.0 / 7000.0 - 1.0 / 1e6)), 0.0))
    brake = np.array([0.0, math.sqrt(MU * (2.0 / 7000.0 - 1.0 / 9100.0)) - wide.v[1], 0.0])
    facts = simulate_round(MatchState(6, "e", pursuer, wide), np.zeros(3), brake, params,
                           view="evader")
    assert facts.evader_a_km == pytest.approx(9100.0)
    outcome = decide(facts, "evader", params)
    assert outcome.winner == "evader" and not outcome.core_band


def test_the_pre_impulse_state_widens_the_band_only_for_round_one_early_entry(
    circular: Orbit,
) -> None:
    # A fast pre-impulse state would widen the pursuer's band; it is judged
    # only by early entry in round 1, so elsewhere it must not.
    params = _params(a_lo_km=9000.0, a_hi_km=9500.0)
    facts = _apart(circular, params, "pursuer", round_number=6)
    fast = CraftState((6500.0, 0.0, 0.0), (0.0, 10.5, 0.0))
    end = facts.end_evader
    without = band_for(params, "pursuer", [(facts.evader.r0, facts.evader.v0), (end.r, end.v)])
    with_start = band_for(params, "pursuer", [(facts.evader.r0, facts.evader.v0),
                                              (end.r, end.v), (fast.r, fast.v)])
    assert with_start.lo_km < without.lo_km
    a = 0.5 * (with_start.lo_km + without.lo_km)
    edited = _edit(facts, start_evader=fast, evader_a_km=a)
    assert decide(edited, "pursuer", params).winner == "pursuer"
    early = _params(a_lo_km=9000.0, a_hi_km=9500.0, readings={"early_entry_wins": True})
    first = _edit(_apart(circular, early, "pursuer", round_number=1), start_evader=fast,
                  evader_a_km=12000.0, evader_a_initial_km=a)
    assert decide(first, "pursuer", early).winner == "evader"


@pytest.mark.parametrize("a_km", [1e11, 1e12])
def test_a_barely_bound_evader_is_outside_a_one_sided_band_for_us(a_km: float) -> None:
    # Rounded to 1e-6 km/s, its published end state can read as escaping,
    # which Q3 resolves against us as evader; as pursuer it counts inside.
    params = _params(a_lo_km=20000.0, a_hi_km=math.inf)
    pursuer = CraftState((0.0, -9000.0, 1000.0), (math.sqrt(MU / 9000.0), 0.0, 0.0))
    evader = CraftState((7000.0, 0.0, 0.0), (0.0, math.sqrt(MU * (2.0 / 7000.0 - 1.0 / a_km)), 0.0))
    state = MatchState(6, "e", pursuer, evader)
    facts = simulate_round(state, np.zeros(3), np.zeros(3), params, view="evader")
    assert decide(facts, "evader", params).winner == "pursuer"
    facts = simulate_round(state, np.zeros(3), np.zeros(3), params, view="pursuer")
    assert decide(facts, "pursuer", params).winner == "evader"
    # Comfortably bound and above a_lo: inside for us.
    bound = CraftState((7000.0, 0.0, 0.0), (0.0, math.sqrt(MU * (2.0 / 7000.0 - 1.0 / 1e6)), 0.0))
    facts = simulate_round(MatchState(6, "e", pursuer, bound), np.zeros(3), np.zeros(3), params,
                           view="evader")
    assert decide(facts, "evader", params).winner == "evader"


def test_the_upper_edge_takes_its_own_margin_on_a_wide_band() -> None:
    # [7000, 42164]: the rounding term at the GEO edge is 36 times that at
    # 7000 km, so an a half the GEO margin under 42164 is outside for us.
    params = _params(a_lo_km=7000.0, a_hi_km=42164.0, t_round_s=600.0)
    pursuer = CraftState((0.0, -9000.0, 1000.0), (math.sqrt(MU / 9000.0), 0.0, 0.0))
    r = np.array([42164.0, 0.0, 0.0])
    v = np.array([0.0, math.sqrt(MU / 42164.0), 0.0])
    probe = simulate_round(MatchState(6, "e", pursuer, CraftState.of(r, v)), np.zeros(3),
                           np.zeros(3), params, view="evader")
    end = probe.end_evader
    hi_margin = max(margin_km(params, r, v, 42164.0), margin_km(params, end.r, end.v, 42164.0))
    lo_margin = max(margin_km(params, r, v, 7000.0), margin_km(params, end.r, end.v, 7000.0))
    assert hi_margin > 3.0 * lo_margin
    facts = _edit(probe, evader_a_km=42164.0 - 0.5 * hi_margin)
    assert decide(facts, "evader", params).winner == "pursuer"
    facts = _edit(probe, evader_a_km=42164.0 - 2.0 * hi_margin)
    assert decide(facts, "evader", params).winner == "evader"


def test_undecided_time_is_never_a_proved_capture_for_us(circular: Orbit) -> None:
    # A pair holding exactly at the radius leaves the detector undecided
    # for most of the round; as pursuer that is no capture.
    from orbitwin.duel.events import capture_facts

    params = _params(r_cap_km=4.0)
    p = circular(6878.0, 51.6, 10.0, 0.0)
    e = circular(6878.0, 51.6, 10.0, math.degrees(2.0 * math.asin(4.0 / (2.0 * 6878.0))))
    facts = simulate_round(MatchState(3, "e", CraftState.of(*p), CraftState.of(*e)), np.zeros(3),
                           np.zeros(3), params, view="pursuer")
    # 1e-9 km under the separation, so float noise proves nothing inside.
    distance = float(np.linalg.norm(p[0] - e[0])) - 1e-9
    exhausted = capture_facts(facts.pursuer, facts.evader, distance, 60.0)
    assert exhausted.exhausted
    assert sum(q.duration_s for q in exhausted.passages_hurt) > 1000.0
    assert not [q for q in exhausted.pieces if q.label == "inside"]
    assert exhausted.passages_favour == []
    tired = dataclasses.replace(facts, capture=exhausted, row_gap_bound_s=GAP)
    assert decide(tired, "pursuer", params).winner is None
    assert decide(dataclasses.replace(tired, view="evader"), "evader", params).winner == "pursuer"


def _escape_edge(params: DuelParams, states: list[tuple[np.ndarray, np.ndarray]]) -> float:
    from orbitwin.duel.referee import edge_energy_margin

    return MU / (2.0 * edge_energy_margin(params, states, math.inf))


def _round_six_at(a_km: float, params: DuelParams) -> RoundFacts:
    pursuer = CraftState((0.0, -9000.0, 1000.0), (math.sqrt(MU / 9000.0), 0.0, 0.0))
    evader = CraftState((7000.0, 0.0, 0.0),
                        (0.0, math.sqrt(MU * (2.0 / 7000.0 - 1.0 / a_km)), 0.0))
    return simulate_round(MatchState(6, "e", pursuer, evader), np.zeros(3), np.zeros(3), params,
                          view="evader")


def test_the_open_edge_sits_where_rounding_could_read_the_evader_as_escaping() -> None:
    params = _params(a_lo_km=20000.0, a_hi_km=math.inf)
    probe = _round_six_at(1e9, params)
    states = [(probe.evader.r0, probe.evader.v0), (probe.end_evader.r, probe.end_evader.v)]
    edge = _escape_edge(params, states)
    from orbitwin.duel.gmat_settings import SETTINGS
    from orbitwin.duel.referee import _rounding_energy

    rounding = max(_rounding_energy(params, r, v) for r, v in states)
    # Twice the rounding share, plus GMAT's a error taken in energy at a_lo.
    integration = SETTINGS["cautious"].a_error_km(1800.0)
    expected = 2.0 * rounding + MU * integration / 20000.0**2
    assert edge == pytest.approx(MU / (2.0 * expected), rel=1e-9)
    assert edge < MU / (4.0 * rounding)
    assert decide(_edit(probe, evader_a_km=0.98 * edge), "evader", params).winner == "evader"
    assert decide(_edit(probe, evader_a_km=1.02 * edge), "evader", params).winner == "pursuer"


def test_the_open_edge_carries_the_live_floor_in_energy() -> None:
    # A 12 km floor on a at a_lo = 20000 km is 0.006 km^2/s^2 of energy, which
    # puts the escape edge at a_lo^2 / 12 = 3.3e7 km.
    params = _params(a_lo_km=20000.0, a_hi_km=math.inf, choices={"a_margin_floor_km": 12.0})
    probe = _round_six_at(1e7, params)
    assert decide(probe, "evader", params).winner == "evader"
    for a_km in (1e8, 1e9):
        assert decide(_round_six_at(a_km, params), "evader", params).winner == "pursuer"
    assert decide(_edit(probe, evader_a_km=20000.0**2 / 12.0 * 0.99), "evader",
                  params).winner == "evader"
    assert decide(_edit(probe, evader_a_km=20000.0**2 / 12.0 * 1.01), "evader",
                  params).winner == "pursuer"


def test_a_huge_finite_upper_edge_is_narrowed_like_an_open_one() -> None:
    params = _params(a_lo_km=20000.0, a_hi_km=1e12)
    probe = _round_six_at(1e9, params)
    band = band_for(params, "evader", [(probe.evader.r0, probe.evader.v0)])
    assert not band.core and 2e4 < band.lo_km < band.hi_km < 1e10
    assert decide(_edit(probe, evader_a_km=1e11), "evader", params).winner == "pursuer"


def test_the_core_fallback_is_centred_in_energy_even_for_a_one_sided_band() -> None:
    # A floor so large that nothing in [a_lo, inf) survives the narrowing:
    # the core is the energy centre +- 10% of the half-width, about 2 a_lo.
    params = _params(a_lo_km=20000.0, a_hi_km=math.inf, choices={"a_margin_floor_km": 1e6})
    probe = _round_six_at(30000.0, params)
    band = band_for(params, "evader", [(probe.evader.r0, probe.evader.v0)])
    assert band.core
    assert band.lo_km == pytest.approx(20000.0 / 0.55) and band.hi_km == pytest.approx(20000.0 / 0.45)
    outcome = decide(_edit(probe, evader_a_km=40000.0), "evader", params)
    assert outcome.core_band and outcome.winner == "evader"
    assert decide(_edit(probe, evader_a_km=30000.0), "evader", params).winner == "pursuer"
    # A finite band's core is centred in energy too.
    finite = _params(a_lo_km=6900.0, a_hi_km=6900.0005)
    core = band_for(finite, "evader", [(probe.evader.r0, probe.evader.v0)])
    eps = [-MU / (2.0 * a) for a in (6900.0, 6900.0005)]
    centre, half = sum(eps) / 2.0, (eps[1] - eps[0]) / 2.0
    assert core.lo_km == pytest.approx(-MU / (2.0 * (centre - 0.1 * half)), rel=1e-12)


def test_a_hyperbolic_band_widened_past_zero_energy_takes_in_the_parabola_and_bound_orbits(
) -> None:
    # As pursuer, energy edges [outer_lo < 0, outer_hi]: an evader at a = +1e12
    # km, on a parabola, or on any hyperbola below the upper edge is inside.
    params = _params(a_lo_km=-1e12, a_hi_km=-1.0)
    r, v = np.array([7000.0, 0.0, 0.0]), np.array([0.0, 11.5, 0.0])
    band = band_for(params, "pursuer", [(r, v)])
    assert band.bound_from_km is not None and 0.0 < band.bound_from_km < 1e12
    for a_km in (1e12, math.inf, -1e15, -1e6, -1.0):
        assert band.contains(a_km), a_km
    assert not band.contains(0.5 * band.bound_from_km)
    assert not band.contains(-0.5)
    narrow = band_for(_params(a_lo_km=-60000.0, a_hi_km=-40000.0), "evader", [(r, v)])
    assert -60000.0 < narrow.lo_km < narrow.hi_km < -40000.0
    wide = band_for(_params(a_lo_km=-60000.0, a_hi_km=-40000.0), "pursuer", [(r, v)])
    assert wide.bound_from_km is None and wide.lo_km < -60000.0 and -40000.0 < wide.hi_km < 0.0


def test_the_pursuer_band_moves_each_finite_edge_by_its_energy_margin() -> None:
    from orbitwin.duel.referee import edge_energy_margin

    params = _params(a_lo_km=7000.0, a_hi_km=42164.0)
    states = [(np.array([42164.0, 0.0, 0.0]), np.array([0.0, 3.07, 0.0])),
              (np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.6, 0.0]))]
    for role, sign_lo in (("pursuer", -1.0), ("evader", +1.0)):
        band = band_for(params, role, states)
        for edge, got, sign in ((7000.0, band.lo_km, sign_lo), (42164.0, band.hi_km, -sign_lo)):
            energy = -MU / (2.0 * edge) + sign * edge_energy_margin(params, states, edge)
            assert got == pytest.approx(-MU / (2.0 * energy), rel=1e-12)


def test_hyperbolic_band_edges_move_by_their_own_energy_margin_in_both_roles() -> None:
    from orbitwin.duel.referee import edge_energy_margin

    states = [(np.array([7000.0, 0.0, 0.0]), np.array([0.0, 11.5, 0.0])),
              (np.array([20000.0, 0.0, 0.0]), np.array([0.0, 7.2, 0.0]))]

    def moved(params: DuelParams, edge: float, sign: float) -> float:
        energy = -MU / (2.0 * edge) + sign * edge_energy_margin(params, states, edge)
        return -MU / (2.0 * energy)

    finite = _params(a_lo_km=-60000.0, a_hi_km=-40000.0)
    for role, sign_lo in (("pursuer", -1.0), ("evader", +1.0)):
        band = band_for(finite, role, states)
        assert band.lo_km == pytest.approx(moved(finite, -60000.0, sign_lo), rel=1e-12)
        assert band.hi_km == pytest.approx(moved(finite, -40000.0, -sign_lo), rel=1e-12)
    # Widened past zero energy: the bound part starts where the lower edge's
    # energy went, and the upper edge moves by its own margin.
    wide = _params(a_lo_km=-1e12, a_hi_km=-1.0)
    band = band_for(wide, "pursuer", states)
    assert band.bound_from_km == pytest.approx(moved(wide, -1e12, -1.0), rel=1e-12)
    assert band.hi_km == pytest.approx(moved(wide, -1.0, +1.0), rel=1e-12)


def test_the_pre_impulse_state_is_left_out_after_round_one_even_under_early_entry(
    circular: Orbit,
) -> None:
    early = _params(a_lo_km=9000.0, a_hi_km=9500.0, readings={"early_entry_wins": True})
    facts = _apart(circular, early, "pursuer", round_number=3)
    fast = CraftState((6500.0, 0.0, 0.0), (0.0, 10.5, 0.0))
    end = facts.end_evader
    without = band_for(early, "pursuer", [(facts.evader.r0, facts.evader.v0), (end.r, end.v)])
    with_start = band_for(early, "pursuer", [(facts.evader.r0, facts.evader.v0),
                                             (end.r, end.v), (fast.r, fast.v)])
    a = 0.5 * (with_start.lo_km + without.lo_km)
    assert decide(_edit(facts, start_evader=fast, evader_a_km=a), "pursuer", early).winner is None


def test_via_boundary_is_not_set_when_the_verdict_stands_without_the_reading(
    circular: Orbit,
) -> None:
    # A capture held from the round start for far longer than the row gap:
    # it counts with or without the boundary reading.
    params = _params()
    facts = _edit(_apart(circular, params, "pursuer"), favour=[Passage(0.0, 900.0)])
    outcome = decide(facts, "pursuer", params)
    assert outcome.winner == "pursuer" and not outcome.via_boundary
    from orbitwin.duel.referee import Band, Reading

    reading = Reading("priority", GAP, 0.0, 0.0, "own-floor", "opponent-floor", "favour")
    assert not judge(facts, reading, Band(1.0, 2.0)).via_boundary


def test_judge_reports_an_exhausted_detector_too(circular: Orbit) -> None:
    from orbitwin.duel.referee import Band, Reading

    facts = _apart(circular, _params(), "referee")
    tired = dataclasses.replace(facts, capture=dataclasses.replace(facts.capture, exhausted=True))
    reading = Reading("priority", 0.0, 0.0, 0.0, "equatorial", "equatorial", "favour")
    assert judge(tired, reading, Band(6800.0, 7000.0)).detector_exhausted
    assert not judge(facts, reading, Band(6800.0, 7000.0)).detector_exhausted
