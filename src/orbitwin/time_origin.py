"""What the template and SC_Alien's GMAT track say about whose epoch is t = 0.

With the craft on different epochs, T_max, C_t and T_team count from SC_Alien's
epoch or SC_Human's; the two readings differ by Δ, the head start. The organizer
answered that it is the earlier one (config.ORGANIZER_TIME_ORIGIN_ANSWER), which
the ranking rules' policy.mission_time_origin "earliest" encodes. This reads two
kinds of evidence for each reading and reports them, so a T_max entered from the
other epoch is seen; nothing is switched.

- The template: a joint Propagate's ElapsedSecs goal counts from that Propagate
  (measured on R2026a), so its end lies Δ later from the early craft's epoch
  than from the late one's. Does either equal the announced T_max?
- The boundary: rules §3 define T_max as when SC_Alien, past perigee, first
  climbs out to the mission boundary. Where is GMAT's SC_Alien at each
  reading's T_max?
"""

from __future__ import annotations

import dataclasses
import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import OrbitWinConfig, PolicyConfig
from .ephemeris import Ephemeris, target_ephemeris_from_gmat
from .errors import GmatRunError, OrbitWinError, ParseError
from .gmat_parser import GmatScriptDocument, parse_gmat_script
from .mission_clock import number_half_step, time_origin_craft, written_half_step
from .models import Scenario
from .report_parser import EPOCH_TOLERANCE_S

READINGS = ("target", "pursuer")
#: The coarsest step a written goal is taken to be rounded to when matching;
#: whether the template can tell the readings apart uses the step as written.
#: T_max and the radius come from YAML as floats: exact to the last digit, or
#: to the unit when whole.
_COARSEST_STEP_S = 1.0
_COARSEST_STEP_KM = 1.0
#: Where another GMAT build or data set may put SC_Alien. On R4, the lowest
#: perigee, drag moves it 3.6 m by T_max and SRP 1.2 m more (examples/ranking/README.md).
MODEL_SLACK_KM = 1.0
#: How far the coast's first row may sit from SC_Alien's epoch: about 30 A1 steps.
_TRACK_START_TOLERANCE_S = 1.0e-5
_BISECTIONS = 80


@dataclass(frozen=True, slots=True)
class ReadingEvidence:
    """One reading of the origin: T_max counts from ``craft``'s epoch."""

    reading: str
    craft: str
    #: Solver t = 0 less this craft's epoch: Δ when it is the early one, else 0.
    offset_s: float
    #: Where the template's Propagates end, from this craft's epoch.
    template_end_s: float
    #: None without an announced T_max.
    template_matches: bool | None
    #: This reading's T_max on the solver clock; None without one announced.
    instant_s: float | None
    alien_radius_km: float | None = None
    alien_radial_rate_km_s: float | None = None
    #: Past SC_Alien's first perigee from this craft's epoch on; climbing there
    #: already counts as past one.
    after_perigee: bool | None = None
    #: When SC_Alien, past that perigee, first reaches policy.mission_boundary_km,
    #: from this craft's epoch.
    boundary_crossing_s: float | None = None
    boundary_tolerance_s: float | None = None
    boundary_matches: bool | None = None
    #: Why SC_Alien's track rules this reading out, if it does.
    refuted_because: str | None = None

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class TimeOriginEvidence:
    """Both readings, the figures behind them, and the lines inspect prints."""

    early: str
    late: str
    #: Δ, late minus early probed A1 epoch.
    offset_s: float
    horizon_s: float | None
    boundary_km: float | None
    configured: str | None
    #: The reading policy.mission_time_origin resolves to; None when unset.
    configured_reading: str | None
    #: The template's Propagates after the alignment, summed: from the late epoch.
    template_goal_s: float
    #: False when policy.add_epoch_alignment wrote the alignment: the goal was
    #: written for a script with none, so its end against ours is our arithmetic.
    template_counted: bool
    template_tolerance_s: float | None
    #: The reading the template's goal picks, if it counts as evidence.
    template_supports: str | None
    readings: tuple[ReadingEvidence, ...]
    #: SC_Alien's first perigee from its epoch, on the solver clock; None when
    #: it is already climbing there or never turns.
    perigee_s: float | None
    perigee_radius_km: float | None
    #: Climbing on its first row; None without a track.
    outbound_from_start: bool | None
    #: The widest of the readings' boundary tolerances.
    boundary_tolerance_s: float | None
    #: Why SC_Alien's track was not read; None when it was.
    track_error: str | None
    verdict: str | None
    reason: str
    lines: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def template_goal(
    scenario: Scenario, document: GmatScriptDocument
) -> tuple[float, float, float]:
    """The template's Propagates after the alignment, summed, and half their written
    step, held to a second and as written."""
    total_s = 0.0
    rounding_s = 0.0
    written_s = 0.0
    for segment in scenario.propagate_segments:
        total_s += segment.duration_s
        if segment.implicit:
            continue
        if not segment.duration_spans:
            raise ParseError(f"Propagate line {segment.line_index + 1} has no goal to read")
        unit = 86400.0 if segment.unit == "ElapsedDays" else 1.0
        line = document.lines[segment.line_index]
        goals = [line[start:end] for start, end in segment.duration_spans]
        rounding_s += max(written_half_step(goal, unit, _COARSEST_STEP_S) for goal in goals)
        written_s += max(written_half_step(goal, unit, math.inf) for goal in goals)
    if not math.isfinite(total_s):
        raise ParseError(f"The template's Propagates end at {total_s!r} s")
    return total_s, rounding_s, written_s


def _radius(track: Ephemeris, time_s: float) -> tuple[float, float]:
    """Geocentric distance and its rate, Hermite between the track's rows."""
    state = track.state_at(time_s)
    radius = float(np.linalg.norm(state.r))
    return radius, float(np.dot(state.r, state.v)) / radius


def _bisect(
    track: Ephemeris, lo: float, hi: float, above: Callable[[float, float], bool]
) -> float:
    """The time in [lo, hi] where ``above(radius, rate)`` turns true."""
    for _ in range(_BISECTIONS):
        mid = 0.5 * (lo + hi)
        if above(*_radius(track, mid)):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _perigee(track: Ephemeris, from_s: float) -> tuple[float | None, bool]:
    """SC_Alien's first perigee from ``from_s`` on, and whether it already climbs there."""
    if _radius(track, from_s)[1] > 0.0:
        return None, True
    r = track.positions_km
    rates = np.einsum("ij,ij->i", r, track.velocities_km_s) / np.linalg.norm(r, axis=1)
    turned = np.flatnonzero((track.times_s > from_s) & (rates >= 0.0))
    if not turned.size:
        return None, False
    k = int(turned[0])
    lo = max(float(track.times_s[k - 1]), from_s) if k else from_s
    time_s = _bisect(track, lo, float(track.times_s[k]), lambda _radius_km, rate: rate >= 0.0)
    return time_s, False


def _crossing(track: Ephemeris, from_s: float, boundary_km: float) -> float | None:
    """The first time after ``from_s`` that SC_Alien climbs through ``boundary_km``."""
    if _radius(track, from_s)[0] >= boundary_km:
        return None
    radii = np.linalg.norm(track.positions_km, axis=1)
    out = np.flatnonzero((track.times_s > from_s) & (radii >= boundary_km))
    if not out.size:
        return None
    k = int(out[0])
    lo = max(float(track.times_s[k - 1]), from_s)
    time_s = _bisect(
        track, lo, float(track.times_s[k]), lambda radius, _rate: radius >= boundary_km
    )
    # No climb rate there: a touch, and nothing to divide the tolerance by.
    return time_s if _radius(track, time_s)[1] > 0.0 else None


def _climb_out(track: Ephemeris, passed_s: float, boundary_km: float) -> float | None:
    """The first climb through ``boundary_km`` after a perigee, from ``passed_s`` on.

    Already beyond it while climbing, a closed SC_Alien climbs out again only
    after its next perigee.
    """
    r = track.positions_km
    rates = np.einsum("ij,ij->i", r, track.velocities_km_s) / np.linalg.norm(r, axis=1)
    while _radius(track, passed_s)[0] >= boundary_km:
        falling = np.flatnonzero((track.times_s > passed_s) & (rates < 0.0))
        if not falling.size:
            return None
        perigee_s, _ = _perigee(track, float(track.times_s[falling[0]]))
        if perigee_s is None:
            return None
        passed_s = perigee_s
    return _crossing(track, passed_s, boundary_km)


def _phrase(matches: list[str], decisive: bool) -> str:
    said = f"the {matches[0]} reading" if len(matches) == 1 else ("both" if matches else "neither")
    if decisive or not matches:
        return said
    return f"{said} (not counted: the numbers as written cannot tell readings Δ apart)"


def assess_time_origin(
    scenario: Scenario,
    document: GmatScriptDocument,
    policy: PolicyConfig,
    track: Ephemeris | None,
    track_error: str | None = None,
) -> TimeOriginEvidence:
    """The evidence for each reading from the template and SC_Alien's GMAT track.

    ``track`` is SC_Alien on the solver clock from its own epoch through at least
    policy.horizon_s; _alien_track flies Δ/2 past it. One that stops at T_max can
    miss a crossing just after it, which reads as a boundary matching neither
    reading: undetermined, never a wrong reading.
    """
    alignment = scenario.alignment
    if alignment is None or alignment.offset_s <= 0.0:
        raise ParseError("The craft start together; there is no time origin to choose")
    delta_s = alignment.offset_s
    horizon = policy.horizon_s
    boundary = policy.mission_boundary_km
    target, pursuer = scenario.target.name, scenario.pursuer.name
    craft = {"target": target, "pursuer": pursuer}
    offsets = {name: delta_s if craft[name] == alignment.early else 0.0 for name in READINGS}
    configured_reading = None
    if policy.mission_time_origin is not None:
        origin = scenario.time_origin_craft or time_origin_craft(
            policy.mission_time_origin, pursuer, target, alignment
        )
        configured_reading = next(name for name in READINGS if craft[name] == origin)
    goal_s, goal_rounding_s, goal_written_s = template_goal(scenario, document)

    template_tolerance = None
    # Whether the numbers as written can tell readings Δ apart at all.
    template_spread = math.inf
    template_match: dict[str, bool | None] = dict.fromkeys(READINGS)
    if horizon is not None:
        horizon_half = number_half_step(horizon, 1.0, _COARSEST_STEP_S)
        template_tolerance = goal_rounding_s + horizon_half + EPOCH_TOLERANCE_S
        template_spread = goal_written_s + horizon_half + EPOCH_TOLERANCE_S
        template_match = {
            name: abs(goal_s + offsets[name] - horizon) <= template_tolerance for name in READINGS
        }
    if horizon is None and track_error is None:
        track_error = "policy.horizon_s is not set, so there is no announced T_max to place"
        track = None
    if track_error is not None:
        track = None
    elif track is None:
        track_error = "no track was given"

    found: dict[str, dict[str, Any]] = {name: {} for name in READINGS}
    for name in READINGS:
        # The parser refuses this origin too: nothing would be left to fly.
        if horizon is not None and horizon - offsets[name] < EPOCH_TOLERANCE_S:
            found[name]["refuted_because"] = (
                f"T_max from {craft[name]}'s epoch ends before {alignment.late}'s epoch, "
                "before the two fly together"
            )
    perigee_s: float | None = None
    perigee_km: float | None = None
    outbound: bool | None = None
    if track is not None and horizon is not None:
        first_s = float(track.times_s[0])
        start = -delta_s if alignment.early == target else 0.0
        if abs(first_s - start) > _TRACK_START_TOLERANCE_S:
            raise GmatRunError(
                f"{target}'s GMAT track starts at t = {first_s!r} s, not at its epoch, "
                f"t = {start!r} s"
            )
        end_s = float(track.times_s[-1])
        if end_s < horizon - EPOCH_TOLERANCE_S:
            raise GmatRunError(
                f"{target}'s GMAT track ends at t = {end_s!r} s, short of T_max {horizon!r} s"
            )
        perigee_s, outbound = _perigee(track, first_s)
        if perigee_s is not None:
            perigee_km = _radius(track, perigee_s)[0]
        horizon_rounding = number_half_step(horizon, 1.0, _COARSEST_STEP_S)
        for name in READINGS:
            entry = found[name]
            # Each reading's mission starts at its craft's epoch, and so does the
            # search for the perigee and the climb out that define its T_max.
            window_s = max(first_s, -offsets[name])
            own_perigee, climbing = _perigee(track, window_s)
            passed_s = window_s if climbing else own_perigee if own_perigee is not None else math.inf
            crossing = (
                _climb_out(track, passed_s, boundary)
                if boundary is not None and math.isfinite(passed_s)
                else None
            )
            if crossing is not None:
                assert boundary is not None
                tolerance = (
                    horizon_rounding
                    + (number_half_step(boundary, 1.0, _COARSEST_STEP_KM) + MODEL_SLACK_KM)
                    / _radius(track, crossing)[1]
                    + EPOCH_TOLERANCE_S
                )
                entry.update(
                    boundary_crossing_s=crossing + offsets[name],
                    boundary_tolerance_s=tolerance,
                    boundary_matches=abs(crossing + offsets[name] - horizon) <= tolerance,
                )
            elif boundary is not None:
                entry["boundary_matches"] = False
            if "refuted_because" in entry:
                continue
            at = min(max(horizon - offsets[name], first_s), end_s)
            radius, rate = _radius(track, at)
            entry.update(
                alien_radius_km=radius, alien_radial_rate_km_s=rate, after_perigee=at > passed_s
            )
            if at < passed_s:
                entry["refuted_because"] = f"{target} has not yet passed perigee there"
            elif rate <= 0.0:
                entry["refuted_because"] = f"{target} is not climbing there"

    readings = tuple(
        ReadingEvidence(
            reading=name,
            craft=craft[name],
            offset_s=offsets[name],
            template_end_s=goal_s + offsets[name],
            template_matches=template_match[name],
            instant_s=(horizon - offsets[name]) if horizon is not None else None,
            **found[name],
        )
        for name in READINGS
    )
    # The organizer wrote no alignment, so where their goal ends is ours to say.
    template_counted = alignment.line_index is not None
    template_decisive = 2.0 * template_spread < delta_s
    template_hits = [item.reading for item in readings if item.template_matches]
    # A goal equal to T_max itself is what an author writes who counts from the
    # late epoch, and also one who did not take the head start off: only a goal
    # short of T_max by Δ says which epoch was meant.
    template_supports = (
        template_hits[0]
        if template_counted and template_decisive and len(template_hits) == 1
        and craft[template_hits[0]] == alignment.early
        else None
    )
    tolerances = [
        item.boundary_tolerance_s for item in readings if item.boundary_tolerance_s is not None
    ]
    boundary_tolerance = max(tolerances) if tolerances else None
    boundary_decisive = boundary_tolerance is not None and 2.0 * boundary_tolerance < delta_s
    support: dict[str, list[str]] = {}
    if template_supports is not None:
        support.setdefault(template_supports, []).append("the template")
    boundary_hits = [item.reading for item in readings if item.boundary_matches]
    if boundary_decisive and len(boundary_hits) == 1:
        support.setdefault(boundary_hits[0], []).append("the boundary")
    refuted = [item for item in readings if item.refuted_because is not None]
    verdict: str | None = None
    if len(support) > 1:
        reason = "the template and the boundary support different readings"
    elif support:
        (verdict, sources), = support.items()
        said = f"{' and '.join(sources)} {'supports' if len(sources) == 1 else 'support'} it"
        against = [item for item in refuted if item.reading == verdict]
        if against:
            reason = f"{said}, but {against[0].refuted_because}"
            verdict = None
        else:
            reason = said
    elif len(refuted) == 1:
        verdict = next(name for name in READINGS if name != refuted[0].reading)
        reason = f"by elimination: for the {refuted[0].reading} reading {refuted[0].refuted_because}"
    elif refuted:
        reason = f"neither reading fits {target}'s track"
    elif boundary is not None and track_error is None and not boundary_hits:
        reason = "the boundary matches neither reading, and nothing else picks one"
    else:
        reason = f"neither the template nor {target}'s track picks one"
    # A radius that fits neither reading, or only the other, says something is off.
    if (
        verdict is not None and boundary is not None and track_error is None
        and verdict not in boundary_hits
    ):
        reason += (
            ", but the boundary matches the other reading" if boundary_hits
            else ", but the boundary matches neither reading"
        )
        verdict = None
    if verdict is not None and track_error is not None:
        reason += f"; {target}'s track was not read"

    evidence = TimeOriginEvidence(
        early=alignment.early,
        late=alignment.late,
        offset_s=delta_s,
        horizon_s=horizon,
        boundary_km=boundary,
        configured=policy.mission_time_origin,
        configured_reading=configured_reading,
        template_goal_s=goal_s,
        template_counted=template_counted,
        template_tolerance_s=template_tolerance,
        template_supports=template_supports,
        readings=readings,
        perigee_s=perigee_s,
        perigee_radius_km=perigee_km,
        outbound_from_start=outbound,
        boundary_tolerance_s=boundary_tolerance,
        track_error=track_error,
        verdict=verdict,
        reason=reason,
        lines=(),
    )
    return dataclasses.replace(
        evidence, lines=_lines(evidence, template_decisive, boundary_decisive)
    )


def _lines(
    evidence: TimeOriginEvidence, template_decisive: bool, boundary_decisive: bool
) -> tuple[str, ...]:
    by = {item.reading: item for item in evidence.readings}
    target, pursuer = by["target"], by["pursuer"]
    head = f"time origin: {evidence.early} starts {evidence.offset_s:.6f} s before {evidence.late}; "
    if evidence.configured == "earliest":
        head += (
            "T_max, C_t and T_team count from the earlier epoch, the organizer answered "
            "on 2026-11-04. This checks T_max against that reading and the other."
        )
    else:
        head += (
            "the preliminary rules do not say whose epoch T_max, C_t and T_team count from. "
            "This only reports; policy.mission_time_origin is what a person set."
        )
    lines = [head]
    goal = (
        f"time origin, template: its Propagates end {evidence.template_goal_s:.4f} s after the "
        f"two fly together, {target.template_end_s:.4f} s from {target.craft}'s epoch and "
        f"{pursuer.template_end_s:.4f} s from {pursuer.craft}'s"
    )
    if evidence.horizon_s is None or evidence.template_tolerance_s is None:
        lines.append(goal + "; policy.horizon_s is not set")
    else:
        hits = [item.reading for item in evidence.readings if item.template_matches]
        said = (
            f"; T_max {evidence.horizon_s!r} s matches "
            f"{_phrase(hits, template_decisive or not evidence.template_counted)} "
            f"(tolerance {evidence.template_tolerance_s:.3g} s)"
        )
        if not evidence.template_counted:
            said += (
                ". Not counted: the template has no alignment and policy.add_epoch_alignment "
                "wrote ours, so its goal was not written against one"
            )
        elif template_decisive and len(hits) == 1 and evidence.template_supports is None:
            said += (
                f". Not counted: a goal of T_max itself also comes from a template that "
                f"counts from {evidence.early}'s epoch and did not take the head start off"
            )
        lines.append(goal + said)
    if evidence.track_error is not None:
        lines.append(f"time origin, track: not read: {evidence.track_error}")
    else:
        if evidence.perigee_s is not None:
            perigee = (
                f"passes perigee at t = {evidence.perigee_s:.4f} s, "
                f"{evidence.perigee_radius_km:.3f} km from the centre"
            )
        elif evidence.outbound_from_start:
            perigee = "is already climbing at its epoch, so its perigee is before it"
        else:
            perigee = "does not pass perigee within the track"
        parts = []
        for item in evidence.readings:
            if item.alien_radius_km is None or item.alien_radial_rate_km_s is None:
                parts.append(f"at {item.craft}'s epoch + T_max: {item.refuted_because}")
                continue
            parts.append(
                f"at {item.craft}'s epoch + T_max (t = {item.instant_s or 0.0:.4f} s) "
                f"{item.alien_radius_km:.3f} km, "
                f"{'past' if item.after_perigee else 'before'} perigee, "
                f"{'climbing' if item.alien_radial_rate_km_s > 0.0 else 'not climbing'} "
                f"{item.alien_radial_rate_km_s:+.4f} km/s"
            )
        lines.append(
            f"time origin, track (t from {evidence.late}'s epoch): GMAT's {target.craft} "
            f"{perigee}; " + "; ".join(parts)
        )
        if evidence.boundary_km is None:
            lines.append(
                "time origin, boundary: policy.mission_boundary_km is not set; set it if the "
                "organizer announces the radius"
            )
        elif evidence.boundary_tolerance_s is None:
            lines.append(
                f"time origin, boundary: {target.craft} does not climb through "
                f"{evidence.boundary_km!r} km past perigee within the track, so neither "
                "reading matches"
            )
        else:
            hits = [item.reading for item in evidence.readings if item.boundary_matches]
            reached = " and ".join(
                f"{item.boundary_crossing_s:.4f} s from {item.craft}'s epoch"
                if item.boundary_crossing_s is not None
                else f"no climb through it within the track from {item.craft}'s epoch"
                for item in evidence.readings
            )
            lines.append(
                f"time origin, boundary: past perigee {target.craft} first reaches "
                f"{evidence.boundary_km!r} km {reached}; T_max {evidence.horizon_s!r} s matches "
                f"{_phrase(hits, boundary_decisive)} "
                f"(tolerance {evidence.boundary_tolerance_s:.3g} s)"
            )
    chosen = (
        f"{evidence.verdict} ({by[evidence.verdict].craft}'s epoch)"
        if evidence.verdict is not None
        else "undetermined"
    )
    configured = evidence.configured
    said = f"{configured}" if configured == evidence.configured_reading else (
        f"{configured}, the {evidence.configured_reading} reading"
    )
    if configured is None:
        stance = "policy.mission_time_origin is not set"
    elif evidence.verdict is None or evidence.configured_reading == evidence.verdict:
        stance = f"policy.mission_time_origin is {said}"
    elif configured == "earliest":
        stance = (
            f"policy.mission_time_origin is {said}, which this evidence does not support: "
            "T_max may have been entered from the other craft's epoch. Check it against the "
            "announcement; nothing was switched"
        )
    else:
        stance = (
            f"policy.mission_time_origin is {said}, which this evidence does not "
            "support; a person decides, nothing was switched"
        )
    lines.append(f"time origin verdict: {chosen}, {evidence.reason}; {stance}. {_numbers(evidence)}")
    return tuple(lines)


def _numbers(evidence: TimeOriginEvidence) -> str:
    """The figures behind the verdict, which end its line."""
    announced = "not set" if evidence.horizon_s is None else f"{evidence.horizon_s!r} s"
    parts = [f"Δ {evidence.offset_s:.6f} s", f"T_max {announced}"]
    for item in evidence.readings:
        figures = [f"template end {item.template_end_s:.4f} s"]
        if item.boundary_crossing_s is not None:
            figures.append(f"boundary at {item.boundary_crossing_s:.4f} s")
        if item.alien_radius_km is not None:
            figures.append(f"r(T_max) {item.alien_radius_km:.3f} km")
        parts.append(f"{item.reading} ({item.craft}): " + ", ".join(figures))
    return "; ".join(parts) + "."


def time_origin_evidence(
    scenario: Scenario,
    document: GmatScriptDocument,
    config: OrbitWinConfig,
    gmat_console: str | Path | None,
    startup_file: str | Path | None,
    workdir: str | Path | None = None,
) -> TimeOriginEvidence | None:
    """assess_time_origin on a GMAT coast of SC_Alien; None when the craft start together.

    The coast flies past T_max from the later craft's epoch, the later of the
    two readings' ends. Without ``workdir`` it runs in a temporary directory.
    """
    alignment = scenario.alignment
    if alignment is None or alignment.offset_s <= 0.0:
        return None
    track: Ephemeris | None = None
    error: str | None = None
    if config.policy.horizon_s is None:
        error = "policy.horizon_s is not set, so there is no announced T_max to place"
    elif gmat_console is None:
        error = "no GMAT console; only GMAT's track of SC_Alien is used here"
    else:
        try:
            if workdir is None:
                with tempfile.TemporaryDirectory(prefix="orbitwin-origin-") as scratch:
                    track = _alien_track(
                        scenario, document, config, gmat_console, startup_file,
                        Path(scratch) / "coast",
                    )
            else:
                track = _alien_track(
                    scenario, document, config, gmat_console, startup_file, Path(workdir)
                )
        except (OrbitWinError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    try:
        return assess_time_origin(scenario, document, config.policy, track, error)
    except GmatRunError as exc:
        return assess_time_origin(
            scenario, document, config.policy, None, f"{type(exc).__name__}: {exc}"
        )


def _alien_track(
    scenario: Scenario,
    document: GmatScriptDocument,
    config: OrbitWinConfig,
    gmat_console: str | Path,
    startup_file: str | Path | None,
    workdir: Path,
) -> Ephemeris:
    """SC_Alien from its own epoch to Δ/2 past T_max after the later craft's, as GMAT
    flies it.

    A match that picks a reading needs the crossing within its tolerance, under
    Δ/2, of its T_max, so Δ/2 more holds every one: T_max is announced rounded,
    and on R8 GMAT's SC_Alien reached the boundary 0.2 ms after it. A crossing
    cut off reads as matching neither reading, which only makes the verdict
    undetermined.
    """
    alignment = scenario.alignment
    assert alignment is not None
    horizon = config.policy.horizon_s
    assert horizon is not None
    late = "target" if alignment.late == scenario.target.name else "pursuer"
    # Counted from the later epoch the solver flies all of it.
    later = dataclasses.replace(
        config,
        policy=dataclasses.replace(
            config.policy, mission_time_origin=late, horizon_s=horizon + 0.5 * alignment.offset_s
        ),
    )
    coast_scenario, _ = parse_gmat_script(document.path, later, probe=scenario.state_probe)
    return target_ephemeris_from_gmat(
        coast_scenario, document, later, gmat_console, startup_file, workdir,
        alignment_rows=True,
    )
