"""Whose epoch the mission clock starts on, and whether T_max fits SC_Alien's coast.

T_team, T_max and Ct count from the earlier of the two craft's epochs
(config.ORGANIZER_TIME_ORIGIN_ANSWER). The ranking rules make T_max the time SC_Alien,
past perigee, first climbs out to the mission boundary: a T_max before that
perigee is not one, and one where SC_Alien is not at the announced boundary was
typed wrong or belongs to another scenario or clock.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import numpy as np

from .config import PolicyConfig
from .conic import seconds_to_periapsis
from .errors import ParseError
from .models import (
    CLEARLY_ECCENTRIC,
    EpochAlignment,
    HorizonCheck,
    Scenario,
    StateVector,
    head_start_lead_s,
    head_start_status,
)
from .orbit import cartesian_to_keplerian, propagate_universal

#: ranking_day_config.py's flag for policy.accept_announced_horizon_before_perigee_s.
OVERRIDE_FLAG = "--accept-announced-horizon-before-perigee-s"
#: How far the two-body coast may put SC_Alien from GMAT's at T_max. Measured on
#: R1-R5 under the §7 model, 2026-11-04: at most 1.81 km at T_max, 0.24 s at perigee.
TWO_BODY_RADIUS_SLACK_KM = 10.0


def written_half_step(text: str, unit: float, coarsest: float) -> float:
    """Half the last written digit of ``text``, times ``unit``, at most half ``coarsest``."""
    try:
        exponent = Decimal(text.strip()).as_tuple().exponent
    except InvalidOperation as exc:
        raise ParseError(f"{text!r} is not a decimal number") from exc
    if not isinstance(exponent, int):
        raise ParseError(f"{text!r} is not a finite number")
    # Clamped first: 10.0**400 overflows, and anything past a second is held to it.
    return 0.5 * min(10.0 ** min(exponent, 6) * unit, coarsest)


def number_half_step(value: float, unit: float, coarsest: float) -> float:
    """written_half_step of a number read from YAML; a whole one is whole units."""
    if float(value).is_integer():
        return 0.5 * min(unit, coarsest)
    return written_half_step(repr(float(value)), unit, coarsest)


def time_origin_craft(
    origin: str | None, pursuer: str, target: str, alignment: EpochAlignment | None
) -> str:
    """The craft whose epoch is t = 0 for T_team, T_max and Ct, and whose
    ElapsedSecs is T_team. With one start either reads the same; the pursuer's."""
    if origin == "target":
        return target
    if origin == "pursuer":
        return pursuer
    if alignment is not None and alignment.offset_s > 0.0:
        return alignment.early
    return pursuer


def check_horizon(
    alien_at_t0: StateVector,
    max_time_s: float,
    horizon_s: float,
    boundary_km: float | None,
    mu_km3_s2: float,
) -> HorizonCheck:
    """Where the two-body coast puts SC_Alien at T_max.

    One climbing at t = 0 passed its perigee before, in the alignment or before
    its epoch. A near-circular one's perigee is kept but places nothing
    (CLEARLY_ECCENTRIC).
    """
    ecc = cartesian_to_keplerian(alien_at_t0, mu_km3_s2).ecc
    perigee_s = None
    if float(np.dot(alien_at_t0.r, alien_at_t0.v)) < 0.0:
        perigee_s = seconds_to_periapsis(
            np.asarray(alien_at_t0.r), np.asarray(alien_at_t0.v), mu_km3_s2
        )
    at = propagate_universal(alien_at_t0, max_time_s, mu_km3_s2)
    radius = float(np.linalg.norm(at.r))
    rate = float(np.dot(at.r, at.v)) / radius
    tolerance = None
    if boundary_km is not None:
        tolerance = (
            number_half_step(boundary_km, 1.0, 1.0)
            + abs(rate) * number_half_step(horizon_s, 1.0, 1.0)
            + TWO_BODY_RADIUS_SLACK_KM
        )
    return HorizonCheck(max_time_s, ecc, perigee_s, radius, rate, boundary_km, tolerance)


def horizon_refusal(check: HorizonCheck, horizon_s: float, origin: str, offset_s: float) -> str:
    perigee = check.perigee_s
    assert perigee is not None
    return (
        f"T_max {horizon_s!r} s from {origin}'s epoch is t = {check.max_time_s:.3f} s on the "
        f"solver clock, before SC_Alien's perigee at t = {perigee:.3f} s "
        f"({perigee + offset_s:.3f} s from {origin}'s epoch, two-body). The rules put T_max "
        "after that perigee, so this is not the announced T_max: check that it was entered "
        "in seconds, for this scenario, and counted from the earlier craft's epoch. If it was "
        "read correctly, the announcement and the rules conflict: STOP and ask the organizer whether "
        "this T_max is right (docs/COMPETITION_POLICY.md, Order of authority). Only after they confirm it in writing, write "
        f"the config again with scripts/ranking_day_config.py {OVERRIDE_FLAG} <that T_max>"
    )


def horizon_override_warning(
    check: HorizonCheck, horizon_s: float, origin: str, offset_s: float
) -> str:
    perigee = check.perigee_s
    assert perigee is not None
    return (
        f"OVERRIDE policy.accept_announced_horizon_before_perigee_s: T_max {horizon_s!r} s from "
        f"{origin}'s epoch, t = {check.max_time_s:.3f} s on the solver clock, is before "
        f"SC_Alien's perigee at t = {perigee:.3f} s ({perigee + offset_s:.3f} s from "
        f"{origin}'s epoch, two-body), which the rules rule out; taken as the organizer "
        "confirmed it"
    )


def near_circular_warning(check: HorizonCheck, horizon_s: float) -> str:
    return (
        f"WARNING: T_max {horizon_s!r} s is unchecked: SC_Alien is near circular (osculating "
        f"e {check.ecc:.3g}, at most {CLEARLY_ECCENTRIC:g}), so the rules' premise, that SC_Alien "
        "passes perigee and climbs out to the boundary, does not hold, and J2 alone puts its "
        "osculating perigee anywhere in one period. Confirm T_max with the organizer"
    )


def horizon_override_record(scenario: Scenario) -> dict[str, float] | None:
    """What inspect, solve, rank and certify record when the override was used."""
    check = scenario.horizon_check
    if not scenario.horizon_before_perigee_accepted or check is None:
        return None
    perigee = check.perigee_s
    assert perigee is not None
    return {
        "max_time_s": scenario.max_time_s,
        "alien_perigee_s": perigee,
        "alien_perigee_from_origin_s": perigee + scenario.time_origin_offset_s,
    }


def horizon_warning(check: HorizonCheck, horizon_s: float) -> str:
    assert check.boundary_km is not None and check.tolerance_km is not None
    return (
        f"WARNING: T_max {horizon_s!r} s puts SC_Alien {check.radius_km:.3f} km from the centre, "
        f"{check.radius_km - check.boundary_km:+.3f} km from the announced boundary "
        f"{check.boundary_km!r} km (tolerance {check.tolerance_km:.3g} km, two-body), "
        f"{'climbing' if check.radial_rate_km_s > 0.0 else 'not climbing'} "
        f"{check.radial_rate_km_s:+.4f} km/s: the rules put it at the boundary, climbing. "
        "Check T_max, the boundary and the time origin against the announcement"
    )


def horizon_line(check: HorizonCheck) -> str:
    """What the check found, for inspect."""
    perigee = (
        f"passes perigee at t = {check.perigee_s:.3f} s"
        if check.perigee_s is not None
        else "has no perigee ahead of t = 0"
    ) + f" (osculating e {check.ecc:.3g}" + (
        ", near circular: its perigee places nothing)" if check.near_circular else ")"
    )
    if check.boundary_km is None:
        boundary = "no boundary announced (policy.mission_boundary_km)"
    else:
        assert check.tolerance_km is not None
        verdict = "OFF" if check.off_boundary else "at"
        boundary = (
            f"{verdict} the {check.boundary_km!r} km boundary: "
            f"{check.radius_km - check.boundary_km:+.3f} km, tolerance "
            f"{check.tolerance_km:.3g} km"
        )
    return (
        f"T_max check (two-body, solver clock): SC_Alien {perigee}; at T_max, "
        f"t = {check.max_time_s:.3f} s, it is {check.radius_km:.3f} km from the centre, "
        f"{'climbing' if check.radial_rate_km_s > 0.0 else 'not climbing'} "
        f"{check.radial_rate_km_s:+.4f} km/s; {boundary}"
    )


def clock_line(
    origin_setting: str | None,
    origin: str,
    alignment: EpochAlignment | None,
    offset_s: float,
    horizon_s: float | None,
    max_time_s: float,
) -> str:
    """The resolved origin, Δ and T_max on the solver clock, for inspect."""
    announced = "not announced" if horizon_s is None else f"{horizon_s!r} s"
    if alignment is None or alignment.offset_s <= 0.0:
        return (
            f"mission clock: origin {origin_setting or 'unset'}; the craft start together, so "
            f"T_team is the solver clock; T_max {announced} is {max_time_s:.6f} s on it"
        )
    late = alignment.late
    return (
        f"mission clock: origin {origin_setting} is {origin}'s epoch; {alignment.early} starts "
        f"Δ = {alignment.offset_s:.6f} s before {late}. T_team is solver t + "
        f"{offset_s:.6f} s; T_max {announced} from {origin}'s epoch is {max_time_s:.6f} s "
        f"on the solver clock, which starts at {late}'s epoch"
    )


def clock_lines(scenario: Scenario, policy: PolicyConfig) -> list[str]:
    """The resolved origin and, under an announced T_max, the check above."""
    lines = [
        clock_line(
            policy.mission_time_origin, scenario.time_origin_craft or scenario.pursuer.name,
            scenario.alignment, scenario.time_origin_offset_s, policy.horizon_s,
            scenario.max_time_s,
        )
    ]
    if scenario.horizon_check is not None:
        lines.append(horizon_line(scenario.horizon_check))
    if policy.head_start_burns:
        lead = head_start_lead_s(scenario)
        lines.append(
            f"head start: open, {scenario.pursuer.name} may burn in its {lead:.6f} s before "
            f"{scenario.target.name}'s epoch (policy.head_start_burns)" if lead > 0.0
            else f"head start: closed, {head_start_status(scenario)}"
        )
    lines += [w for w in scenario.warnings if w.startswith(("OVERRIDE ", "WARNING: T_max "))]
    return lines


def clock_record(scenario: Scenario, policy: PolicyConfig) -> dict[str, object]:
    """The clock rank and the certificate record: Δ, the origin and what it resolved to."""
    return {
        "alignment_offset_s": (
            scenario.alignment.offset_s if scenario.alignment is not None else 0.0
        ),
        "mission_time_origin": policy.mission_time_origin,
        # Whose ElapsedSecs is T_team.
        "time_origin_craft": scenario.time_origin_craft,
        "time_origin_offset_s": scenario.time_origin_offset_s,
        # Set when policy.accept_announced_horizon_before_perigee_s took T_max.
        "announced_horizon_before_perigee": horizon_override_record(scenario),
        # How long before t = 0 the pursuer may burn (policy.head_start_burns), and why not.
        "head_start_burns": policy.head_start_burns,
        "head_start_s": head_start_lead_s(scenario),
        "head_start_note": head_start_status(scenario),
    }
