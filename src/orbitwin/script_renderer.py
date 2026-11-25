from __future__ import annotations

import difflib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .burn_rows import maneuvers_flown
from .config import SCHEDULE_TOLERANCE_S, OrbitWinConfig
from .console_copy import assert_plot_lines_unchanged
from .errors import PolicyError
from .frames import inertial_to_script_components
from .gmat_parser import (
    _ASSIGN_RE,
    _BEGIN_MISSION_RE,
    GMAT_RESERVED_RESOURCE_NAMES,
    GmatScriptDocument,
    _code_part,
    _strip_quotes,
    parse_gmat_script,
    split_create_names,
)
from .models import (
    Burn,
    Candidate,
    ReportFileSpec,
    Scenario,
    StateProbe,
    as_vector3,
    burn_frame,
    burn_may_follow_alignment,
    head_start_lead_s,
    head_start_status,
    synthesized_burn_frame,
    template_maneuvers,
    uses_head_start,
)
from .rules import interception_rules
from .trajectory import state_with_burns_at

#: Matched against a line's code part; the comment and line ending are kept after it.
_ASSIGN_VALUE_RE = re.compile(
    r"^(?P<prefix>\s*(?:GMAT\s+)?[A-Za-z_]\w*\.[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*=\s*)"
    r"(?P<value>.*?)(?P<suffix>\s*;\s*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RenderResult:
    script_bytes: bytes
    burn_times_s: tuple[float, ...]
    mission_end_s: float
    segment_durations_s: tuple[float, ...]
    changed_lines: tuple[int, ...]
    semantic_diff: str


#: One grammar for the placeholder and for our number: "Team" in any case, not
#: right after a letter, at most one " ", "_" or "-", then a run of letters and
#: digits. A run of only X's or only N's is the placeholder; ours is the number.
_TEAM_TOKEN_RE = re.compile(r"(?<![a-z])(?P<team>team[ _-]?)(?P<tail>[a-z0-9]+)", re.IGNORECASE)
_PLACEHOLDER_TAIL_RE = re.compile(r"x+|n+", re.IGNORECASE)
_TEAM_GRAMMAR = (
    "A placeholder is 'Team' in any case, not right after a letter, then at most one "
    "space, '_' or '-', then a run of X's or a run of N's, not mixed, with no letter or "
    "digit after it: TeamXX, Team_xx, Team-NN, 'Team XX'"
)


def _is_placeholder(tail: str) -> bool:
    return _PLACEHOLDER_TAIL_RE.fullmatch(tail) is not None


def official_report_filename(filename: str, team_number: int | None) -> str:
    """``filename`` with each TeamXX placeholder numbered; unchanged without a number."""
    if team_number is None:
        return filename
    return _TEAM_TOKEN_RE.sub(
        lambda m: f"{m['team']}{team_number}" if _is_placeholder(m["tail"]) else m[0], filename
    )


def _team_number_problem(filename: str, team_number: int) -> str | None:
    """Why ``filename`` does not carry ``team_number``, or None when it does."""
    tails = [m["tail"] for m in _TEAM_TOKEN_RE.finditer(filename)]
    ours = str(team_number)
    others = sorted({tail for tail in tails if tail.isdigit() and tail != ours})
    if others:
        return f"carries the wrong team number ({', '.join(others)}, not {ours})"
    unclear = [
        tail for tail in tails
        if tail != ours and (tail[0].isdigit() or tail[0].lower() in "xn")
    ]
    if unclear:
        return f"has Team{unclear[0]}, which is neither a placeholder nor Team{ours}"
    if ours not in tails:
        return "has no team-number placeholder"
    return None


def official_report_files(
    scenario: Scenario, config: OrbitWinConfig
) -> tuple[ReportFileSpec, ...]:
    """The official ReportFiles as a final script must declare them."""
    numbered = tuple(
        replace(
            spec, filename=official_report_filename(spec.filename, config.policy.team_number)
        )
        for spec in scenario.report_files
    )
    # Two reports on one path: GMAT keeps one, and the parity checks compare it
    # with itself. Windows paths ignore case, so a template's own TeamXX and
    # TEAMXX are one file there too.
    if len({spec.filename.casefold() for spec in numbered}) < len(numbered):
        names = ", ".join(f"{spec.name} {spec.filename!r}" for spec in numbered)
        raise PolicyError(
            f"The report names, numbered, make two of them one file (Windows ignores case): "
            f"{names}"
        )
    return numbered


def _placeholder_reports(scenario: Scenario) -> list[ReportFileSpec]:
    return [
        spec for spec in scenario.report_files
        if any(_is_placeholder(m["tail"]) for m in _TEAM_TOKEN_RE.finditer(spec.filename))
    ]


def official_report(scenario: Scenario) -> ReportFileSpec | None:
    """The report the organizer scores: of the reports GMAT writes, ReportFile1, else the first.

    One with WriteReport = false writes no file, so it is never uploaded.
    """
    return official_report_of(scenario.report_files)


def official_report_of(specs: Iterable[ReportFileSpec]) -> ReportFileSpec | None:
    """official_report on a script's ReportFiles alone."""
    written = [spec for spec in specs if spec.write_report]
    named = [spec for spec in written if spec.name == "ReportFile1"]
    return named[0] if named else next(iter(written), None)


def report_name_verdict(
    scenario: Scenario, config: OrbitWinConfig
) -> tuple[str | None, tuple[str, ...]]:
    """(why certify refuses the report names or None, notes on reports left as written).

    Under the ranking rules every placeholder takes our number, the official
    report (official_report) must carry it unless the organizer has said in
    writing to keep its name (policy.report_filename_as_written), and no report
    may carry another team's. Only the reports GMAT writes are judged.
    """
    team = config.policy.team_number
    placeholders = _placeholder_reports(scenario)
    if not interception_rules(config.policy.rules).report_must_carry_team:
        if team is not None and scenario.report_files and not placeholders:
            names = ", ".join(repr(spec.filename) for spec in scenario.report_files)
            return None, (
                f"policy.team_number is {team}, but the official report names ({names}) "
                "carry no Team placeholder; they are left as the template has them",
            )
        return None, ()
    if team is None:
        return (
            "policy.team_number is unset; the ranking rules need it to name the report",
            (),
        )
    official = official_report(scenario)
    if official is None:
        return None, ()
    try:
        numbered = official_report_files(scenario, config)
    except PolicyError as exc:
        return str(exc), ()
    written = [spec for spec in scenario.report_files if spec.write_report]
    placeholders = [spec for spec in placeholders if spec.write_report]
    problems: dict[str, str | None] = {}
    shown: dict[str, str] = {}
    for spec, after in zip(scenario.report_files, numbered, strict=True):
        problems[spec.name] = _team_number_problem(after.filename, team)
        shown[spec.name] = f"{spec.name}.Filename {spec.filename!r}" + (
            f" (numbered {after.filename!r})" if after.filename != spec.filename else ""
        )
    numbered_names = {spec.name for spec in placeholders}
    faults: dict[str, str] = {}
    for spec in written:
        problem = problems[spec.name]
        if problem is None:
            continue
        if problem.startswith("carries the wrong") or spec.name in numbered_names:
            faults[spec.name] = f"{shown[spec.name]} {problem}"
    if placeholders and official.name not in numbered_names:
        for spec in placeholders:
            faults.setdefault(
                spec.name,
                f"{shown[spec.name]} has a placeholder and the official report "
                f"{official.name} has none: ask the organizer which report takes the number",
            )
    as_written = config.policy.report_filename_as_written
    kept = False
    if as_written is not None:
        if as_written != official.filename:
            faults.setdefault(
                official.name,
                f"policy.report_filename_as_written is {as_written!r}, not the official "
                f"report's {official.name}.Filename {official.filename!r}",
            )
        elif official.name in numbered_names:
            faults.setdefault(
                official.name,
                f"{shown[official.name]} has a placeholder, which takes the number; "
                "policy.report_filename_as_written is only for a name with none",
            )
        else:
            kept = True
    elif problems[official.name] is not None:
        faults.setdefault(
            official.name,
            f"the official report {shown[official.name]} {problems[official.name]}: the "
            "official script names no team placeholder: stop and ask the organizer how to "
            "name the report. Only if they answer in writing to keep it, write the day "
            f"config again with --report-filename-as-written '{official.filename}'",
        )
    if faults:
        return (
            f"Under the ranking rules the official report must carry Team{team}, every "
            f"placeholder must take it and no report may carry another team's number: "
            f"{'; '.join(faults.values())}. {_TEAM_GRAMMAR}. The official script may not "
            "be edited: stop and ask the organizer how the report must be named."
        ), ()
    notes = [
        f"{shown[spec.name]} {problems[spec.name]}; it is left as the organizer wrote it"
        for spec in written
        if spec.name != official.name and problems[spec.name] is not None
    ]
    if kept:
        notes.insert(0, (
            f"the official report {official.name}.Filename {official.filename!r} is kept as "
            "written, by policy.report_filename_as_written (the organizer's written answer)"
        ))
    return None, tuple(notes)


def team_number_warnings(scenario: Scenario, config: OrbitWinConfig) -> tuple[str, ...]:
    """What solve, inspect, rank and certify say about the official report names."""
    refusal, notes = report_name_verdict(scenario, config)
    if refusal is not None:
        return (f"certify will refuse to submit this: {refusal}", *notes)
    return notes


def assert_team_number_for_submission(scenario: Scenario, config: OrbitWinConfig) -> None:
    """Under the ranking rules the submission carries our team number."""
    refusal, _ = report_name_verdict(scenario, config)
    if refusal is not None:
        raise PolicyError(refusal)


def _number_report_filenames(
    lines: list[str], scenario: Scenario, config: OrbitWinConfig
) -> list[str]:
    """Number each official Filename's placeholder in place; returns the diff lines.

    Only the value between ``=`` and ``;`` is touched, so spacing, a trailing
    comment and the line ending stay.
    """
    end = next(
        (index for index, line in enumerate(lines) if _BEGIN_MISSION_RE.match(_code_part(line))),
        len(lines),
    )
    diff: list[str] = []
    for spec, wanted in zip(
        scenario.report_files, official_report_files(scenario, config), strict=True
    ):
        if wanted.filename == spec.filename:
            continue
        found = None
        for index in range(end):
            match = _ASSIGN_RE.match(_code_part(lines[index].rstrip("\r\n")))
            if match and match.group("object") == spec.name and match.group("field") == "Filename":
                found = index, match
                break
        if found is None:
            raise PolicyError(
                f"{spec.name} has no Filename line to number; GMAT names its report "
                f"{spec.filename!r}"
            )
        index, match = found
        start, stop = match.span("value")
        line = lines[index]
        value = official_report_filename(line[start:stop], config.policy.team_number)
        if _strip_quotes(value) != wanted.filename:  # pragma: no cover - defensive
            raise PolicyError(f"Could not safely number {spec.name}.Filename: {line!r}")
        lines[index] = line[:start] + value + line[stop:]
        diff.append(f"{spec.name}.Filename: {spec.filename!r} -> {wanted.filename!r}")
    return diff


def _format_float(value: float) -> str:
    """Render a double so that reading it back gives the same double.

    ``.16g`` does not. Sixteen significant digits are not enough to round-trip
    an IEEE-754 double in general: 44.2% of random doubles come back different,
    and 12 of the 33 burn components produced by one real search did. The
    script would then carry an impulse the candidate does not hold, and every
    number recorded about that candidate -- its total delta-v, its score, its
    place in the archive -- would describe the other one.

    repr() gives the shortest string that does round-trip, so the script stays
    readable as well as exact: 0.6388 stays "0.6388" where ".17g" would write
    "0.63880000000000003" for the same double.

    One value does not come back: anything under 5e-16 is written as 0, so a
    component of 1e-16 km/s reads back as 0.0 and the script does not hold what
    the candidate holds. That is deliberate and it is the only exception. A
    Lambert solve leaves a component of that size wherever the transfer is very
    nearly planar, and 1e-16 km/s is 1e-13 m/s -- below anything §5 sums or any
    report resolves, but printed in the script it looks like a number someone
    chose. The candidate keeps the residue and the submission says zero.
    """
    if not math.isfinite(value):
        raise PolicyError("Cannot render a non-finite number")
    if abs(value) < 5.0e-16:
        value = 0.0
    rendered = repr(value)
    if float(rendered) != value:  # pragma: no cover - defensive
        raise PolicyError(f"Cannot render {value!r} without losing precision")
    return rendered


def _report_step_s(scenario: Scenario) -> float | None:
    """The interval GMAT writes report rows at, or None if it cannot be read.

    On a two-body arc the integrator runs at its ceiling, so the row spacing is
    ``Propagator.MaxStep``. Verified against a real R2026a run: rows exactly
    1.000000 s apart under ``MaxStep = 1``, phased from the last Maneuver.
    It is the MaxStep of the final segment's first propagator, which drives a
    Synchronized line; R2026a wrote 1 s rows under `Synchronized PH(..) PA(..)`
    with PH at 1 s and PA at 60 s.
    """
    if not scenario.propagate_segments:
        return None
    lead = scenario.propagate_segments[-1].propagator_names[0].lower()
    for key, value in scenario.propagator_fields.items():
        name, _, field = key.rpartition(".")
        if name.lower() != lead or field.lower() != "maxstep":
            continue
        try:
            step = float(value.strip().strip("'").strip('"'))
        except ValueError:
            return None
        return step if step > 0.0 else None
    return None


def _schedule_slots(
    scenario: Scenario,
    candidate: Candidate,
    config: OrbitWinConfig,
) -> tuple[tuple[float, ...], float, tuple[float, ...]]:
    # A template that declares slots fixes how many maneuvers can be expressed.
    # One that declares none leaves it to us, and we add exactly the maneuvers
    # the candidate holds -- so there are no unused slots to fill with an exact
    # zero vector, and no late-burn that cannot be scheduled.
    # A template that already maneuvers fixes how many impulses can be
    # expressed. One that does not leaves it to us, and we write exactly the
    # maneuvers the candidate holds -- so there are no unused slots to fill
    # with an exact zero vector, however many resources are declared.
    slot_count = len(scenario.burn_slots) if template_maneuvers(scenario) else len(candidate.burns)
    if len(candidate.burns) > slot_count:
        raise PolicyError(
            f"Candidate needs {len(candidate.burns)} burns but the official template exposes {slot_count} slots"
        )
    # `Burn`, not `object`. Every value put in below is one, and the loop that
    # reads them back needs burn.time_s and burn.dv_script_km_s -- so widening
    # the value type only moved the moment the mistake would show up from the
    # dict to the attribute access.
    used: dict[int, Burn] = {}
    for burn in candidate.burns:
        if not 0 <= burn.slot_index < slot_count:
            raise PolicyError(f"Burn refers to slot {burn.slot_index}, outside the official template")
        if burn.slot_index in used:
            raise PolicyError(f"Two burns are assigned to official slot {burn.slot_index}")
        used[burn.slot_index] = burn

    # Slots are walked in template order. An unused slot takes the earliest
    # legal epoch, which lets a candidate occupy a later slot while the earlier
    # ones carry an exact zero vector -- the only way to express a late single
    # burn in a template whose remaining slots would otherwise be scheduled
    # after it and run past Tmax.
    floor_s = config.policy.min_propagate_duration_s
    # The production margin unless the formal frontier is explicitly allowed to
    # be submitted, in which case the rule boundary applies.
    separation_s = (
        config.policy.min_burn_separation_s
        if config.policy.allow_formal_frontier_submission
        else config.policy.safe_burn_separation_s
    )
    # After an epoch alignment a burn may fire at t = 0 itself, a Maneuver with
    # no Propagate before it (docs/ASYNC_START_AND_INSERTION.md §2). A template
    # written that way fixes slot 0 there; an open one may choose it.
    fixed_at_zero = bool(scenario.propagate_segments) and scenario.propagate_segments[0].implicit
    may_fire_at_zero = burn_may_follow_alignment(scenario)
    # Head-start burns take the first slots, in the alignment: the pursuer's
    # epoch is -lead, each segment and what is left of the alignment at least floor_s.
    lead_s = head_start_lead_s(scenario)
    early = sum(1 for burn in candidate.burns if burn.time_s < 0.0)
    times: list[float] = []
    previous = -lead_s
    for index in range(early):
        head = used.get(index)
        if lead_s <= 0.0 or head is None or head.time_s >= 0.0:
            raise PolicyError(
                f"Slot {index} burn before t = 0: only an open template's first slots may fire "
                f"in a head start, under policy.head_start_burns "
                f"({head_start_status(scenario) or 'off'})"
            )
        earliest = previous + (floor_s if index == 0 else separation_s)
        # No slack at the two segment floors: the parser holds them exactly.
        short = head.time_s - previous < floor_s if index == 0 else head.time_s < earliest - 1.0e-9
        if short or -head.time_s < floor_s:
            raise PolicyError(
                f"Slot {index} burn at {head.time_s:.12g} s is outside the head start "
                f"[{earliest:.12g}, {-floor_s:.12g}] s: the segment before it, or the "
                "alignment after it, would be under the floor or the burn separation"
            )
        times.append(float(head.time_s))
        previous = float(head.time_s)
    for index in range(early, slot_count):
        if index == early and early:
            earliest = max(floor_s, previous + separation_s)
        else:
            earliest = previous + (floor_s if index == 0 else separation_s)
        if index == 0 and fixed_at_zero:
            earliest = 0.0
        # Not `burn`: that name already holds each burn the candidate carries,
        # in the loop above that filled `used`. This one is the burn assigned to
        # *this* slot, and there may not be one.
        slot_burn = used.get(index)
        if slot_burn is None:
            if not config.policy.allow_zero_vector_unused_slots:
                # ASSUMPTIONS_AND_GAPS.md records filling an unused official
                # slot with an exact zero vector as pending organizer
                # confirmation. This is the switch for the answer being no, so
                # it has to refuse rather than render the thing it forbids.
                raise PolicyError(
                    f"Official slot {index} carries no maneuver, so rendering it means "
                    "writing an exact zero vector, and "
                    "policy.allow_zero_vector_unused_slots is off. Obtain the organizer's "
                    "written permission, or supply a candidate that fills every slot."
                )
            times.append(earliest)
            previous = earliest
            continue
        if slot_burn.time_s < -1.0e-9:
            raise PolicyError("Burn time cannot be negative")
        if not config.policy.allow_zero_vector_unused_slots and not any(
            slot_burn.dv_script_km_s
        ):
            # As the policy guard reads it: an exact zero is an exact zero, used or not.
            raise PolicyError(
                f"Official slot {index} carries an exact zero vector, and "
                "policy.allow_zero_vector_unused_slots is off"
            )
        if index == 0 and fixed_at_zero and abs(slot_burn.time_s) > 1.0e-9:
            raise PolicyError(
                f"Slot 0 burn at {slot_burn.time_s:.12g} s: the template maneuvers right "
                "after its epoch alignment, at t = 0, with no Propagate between them to move it"
            )
        if (
            index == early
            and (fixed_at_zero or may_fire_at_zero)
            and abs(slot_burn.time_s) <= 1.0e-9
            and (not early or previous + separation_s <= 1.0e-9)
        ):
            times.append(0.0)
            previous = 0.0
            continue
        if slot_burn.time_s < earliest - 1.0e-9:
            reason = (
                "GMAT hangs on a zero-length ElapsedSecs goal, so the template cannot "
                "express a maneuver at exactly mission start"
                if index == 0
                else "the preceding slot sets this floor via the burn separation"
            )
            raise PolicyError(
                f"Slot {index} burn at {slot_burn.time_s:.12g} s is earlier than the "
                f"{earliest:.12g} s floor: {reason}"
            )
        times.append(float(slot_burn.time_s))
        previous = float(slot_burn.time_s)

    completion = candidate.intercept_time_s
    if completion is None:
        completion = float(candidate.metadata.get("arrival_time_s", scenario.max_time_s))
    completion = min(max(float(completion), 0.0), scenario.max_time_s)

    # Rows after the alignment phase from its end when every burn is in the head start.
    last_burn = times[-1] if len(times) > early else 0.0
    if last_burn + config.policy.final_padding_s > scenario.max_time_s + SCHEDULE_TOLERANCE_S:
        raise PolicyError(
            "Official template has more burn slots than can be safely scheduled before Tmax; "
            "obtain permission to remove unused Maneuver commands"
        )
    # Where the sequence stops is a scoring decision, not a formatting one.
    #
    # 初賽規則 §4 makes Tteam = min(tint, 4TA), which reads the same whether the
    # report stops at the interception or runs on -- the interception is in it
    # either way. That is why this used to be the horizon unconditionally, and
    # on 2026-09-05 it cost 22.2266 points: the organizer's validator reported
    # C = 23815.43 s (= 4TA) as the completion time for a run whose own D =
    # 2.394 km confirmed the 5 km entry at 5982.99 s. It reads the completion
    # time off the last row, so a report that runs to the horizon is scored as
    # a mission that never intercepted.
    #
    # A hit now ends inside the sphere, so its last row is the interception; a
    # miss still runs to max_time_s, where its distance is scored.
    if (
        config.policy.mission_end_policy == "interception"
        and candidate.successful
        and candidate.intercept_time_s is not None
    ):
        # Stop where GMAT measured the hit. Before a measurement, stop on the
        # predicted first row inside: rows are MaxStep apart from the last
        # Maneuver, not from tint. min_propagate_duration_s is the floor (a zero
        # goal hangs GMAT); interception_tail_s is the fallback.
        completion_s = max(float(candidate.intercept_time_s), last_burn)
        step_s = _report_step_s(scenario)
        measured_stop = candidate.metadata.get("gmat_event", {}).get("stop_time_s")
        if measured_stop is not None:
            # GMAT flew these burns and found where the hit is inside. The step
            # prediction below is wrong whenever InitialStepSize != MaxStep.
            completion_s = last_burn
            tail = float(measured_stop) - last_burn
            if tail < config.policy.min_propagate_duration_s + SCHEDULE_TOLERANCE_S:
                # Inside before the last burn, which still has to fly.
                tail = config.policy.interception_tail_s
        elif step_s is not None and completion_s > last_burn:
            steps = math.ceil((completion_s - last_burn) / step_s - SCHEDULE_TOLERANCE_S)
            aligned = last_burn + steps * step_s
            tail = max(aligned - completion_s, config.policy.min_propagate_duration_s)
        else:
            tail = config.policy.interception_tail_s
        mission_end = min(completion_s + tail, scenario.max_time_s)
    else:
        mission_end = scenario.max_time_s
    durations: list[float] = []
    post = times[early:]
    cursor = 0.0
    for time_s in post:
        duration = time_s - cursor
        if duration < -1.0e-9:
            raise PolicyError("Burn schedule is not monotonic")
        durations.append(max(0.0, duration))
        cursor = time_s
    durations.append(mission_end - cursor)
    minimum = config.policy.min_propagate_duration_s
    for index, duration in enumerate(durations):
        if index == 0 and duration == 0.0 and post and post[0] == 0.0:
            continue  # the implicit segment after an alignment; nothing propagates
        if duration < minimum:
            raise PolicyError(
                f"Propagate segment {index} would last {duration:.12g} s, below the "
                f"{minimum:.12g} s floor; GMAT hangs on a zero-length ElapsedSecs goal, so "
                "the official template cannot express a maneuver at exactly mission start"
            )
    # A slotted template fixes the number of segments; an open one is split
    # into one more segment than there are maneuvers, and the check is that the
    # schedule and the split agree rather than that either matches the file.
    expected_segments = (
        len(scenario.propagate_segments)
        if template_maneuvers(scenario)
        else slot_count - early + 1
    )
    if len(durations) != expected_segments:
        raise PolicyError("Internal schedule does not match the official mission sequence")
    return tuple(times), float(mission_end), tuple(durations)


def _patch_assignment(line: str, new_value: float) -> str:
    code = _code_part(line.rstrip("\r\n"))
    match = _ASSIGN_VALUE_RE.match(code)
    if not match:
        raise PolicyError(f"Could not safely patch assignment line: {line!r}")
    return (
        match.group("prefix") + _format_float(new_value) + match.group("suffix")
        + line[len(code):]
    )


def _patch_spans(line: str, spans: Iterable[tuple[int, int]], values: Iterable[float]) -> str:
    patched = line
    operations = sorted(zip(spans, values, strict=True), key=lambda item: item[0][0], reverse=True)
    for (start, end), value in operations:
        patched = patched[:start] + _format_float(value) + patched[end:]
    return patched


def _synthesized_burn_names(document: GmatScriptDocument, count: int) -> list[str]:
    """Names for maneuvers we add, that GMAT will accept and a reader expects.

    ``ImpulsiveBurn1``, ``ImpulsiveBurn2`` and so on is what GMAT's own GUI
    generates and what the organizer's workshop example uses, so a submitted
    script reads as if it had been built there. A name already in the file, or
    one GMAT reserves for a mission-sequence command, is skipped rather than
    collided with.
    """
    names: list[str] = []
    index = 1
    while len(names) < count:
        name = f"ImpulsiveBurn{index}"
        index += 1
        if name in GMAT_RESERVED_RESOURCE_NAMES:
            continue
        if re.search(rf"\b{re.escape(name)}\b", document.text):
            continue
        names.append(name)
    return names


def _burn_resource_block(
    name: str,
    values: tuple[float, float, float],
    newline: str,
    frame: tuple[str, str, str],
) -> list[str]:
    coordinate_system, origin, axes = frame
    fields = [
        f"{name}.CoordinateSystem = {coordinate_system};",
        f"{name}.Origin = {origin};",
        f"{name}.Axes = {axes};",
        f"{name}.Element1 = {_format_float(values[0])};",
        f"{name}.Element2 = {_format_float(values[1])};",
        f"{name}.Element3 = {_format_float(values[2])};",
        # False on purpose: §5 sums the impulse magnitudes and says nothing
        # about mass, and a decrementing mass would make the delta-v a candidate
        # spends depend on the order it spends it in.
        f"{name}.DecrementMass = false;",
        f"{name}.Isp = 300;",
        f"{name}.GravitationalAccel = 9.81;",
    ]
    return [f"Create ImpulsiveBurn {name};{newline}"] + [f"{field}{newline}" for field in fields]


def _render_open_template(
    document: GmatScriptDocument,
    scenario: Scenario,
    candidate: Candidate,
    segment_durations: tuple[float, ...],
) -> tuple[list[str], list[str], list[str]]:
    """Write the maneuvers into a template that declares none.

    初賽規則 §2 says B 可進行多次瞬時脈衝機動 and 說明會簡報 p.8 says 由參賽者對
    Spacecraft B 進行機動. A template whose mission sequence is a single
    Propagate offers no way to do either, so the maneuvers are ours to add --
    not a liberty taken with the organizer's file, but the only reading under
    which the task they set can be performed at all.

    What is added is bounded and stated: one Create ImpulsiveBurn block per
    impulse, one Maneuver command per impulse, and the single Propagate split
    into one more segment than there are impulses. Every other line is the
    organizer's, untouched, and assert_final_script_policy checks exactly that.
    Head-start burns get their own pursuer-alone pairs, returned third, for
    render_candidate to put before the alignment.
    """
    lines = list(document.lines)
    newline = document.encoding.newline
    # The parser found this line, so the same pattern finds it here.
    begin_index = next(
        index for index, line in enumerate(lines) if _BEGIN_MISSION_RE.match(_code_part(line))
    )
    segment = scenario.propagate_segments[0]
    ordered = sorted(candidate.burns, key=lambda burn: burn.time_s)
    # _schedule_slots refuses a schedule whose slot order is not its time order,
    # so these two agree and the impulse written into position k is the one
    # whose epoch ends segment k.
    declared = scenario.burn_slots
    synthesized_names = _synthesized_burn_names(document, max(0, len(ordered) - len(declared)))

    names: list[str] = []
    resources: list[str] = []
    diff_lines: list[str] = []
    for position, burn in enumerate(ordered):
        # Coerced through the one constructor rather than a bare comprehension:
        # a burn carrying two components would otherwise reach _burn_resource_block
        # and raise IndexError from inside the string that names Element3, and a
        # burn carrying four would render the first three and silently drop the
        # rest into a delta-v total that counted it.
        values = as_vector3(burn.dv_script_km_s)
        if position < len(declared):
            # The organizer declared this resource and left it unmaneuvered.
            # Using it is a smaller edit than adding one beside it, and its
            # components are already lines in their file.
            slot = declared[position]
            names.append(slot.name)
            old_values = (slot.element1_km_s, slot.element2_km_s, slot.element3_km_s)
            for component, line_index in enumerate(slot.assignment_lines):
                lines[line_index] = _patch_assignment(lines[line_index], values[component])
                diff_lines.append(
                    f"{slot.name}.Element{component + 1}: "
                    f"{_format_float(old_values[component])} -> "
                    f"{_format_float(values[component])}"
                )
        else:
            name = synthesized_names[position - len(declared)]
            names.append(name)
            resources.extend(
                _burn_resource_block(name, values, newline, synthesized_burn_frame(scenario))
            )
            resources.append(newline)
            diff_lines.append(
                f"+ Create ImpulsiveBurn {name}: "
                + ", ".join(_format_float(value) for value in values)
                + " km/s"
            )

    # The organizer's own Propagate line is the template for every segment, so
    # the propagator, the spacecraft list and the syntax are theirs; only the
    # duration differs between copies. A NUL or Ctrl-Z after its code, where GMAT
    # stops reading, stays on the last copy only: on any other, the Maneuvers
    # and segments after it would not run. The earlier copies drop the comment too.
    early = sum(1 for burn in ordered if burn.time_s < 0.0)
    head_start: list[str] = []
    cursor = -head_start_lead_s(scenario)
    alignment = scenario.alignment
    for name, burn in zip(names[:early], ordered[:early], strict=True):
        assert alignment is not None  # _schedule_slots opens no head start without one
        craft = scenario.pursuer.name
        duration_s = burn.time_s - cursor
        cursor = burn.time_s
        head_start.append(
            f"Propagate {alignment.propagator}({craft}) "
            f"{{{craft}.ElapsedSecs = {_format_float(duration_s)}}};{newline}"
        )
        head_start.append(f"Maneuver {name}({craft});{newline}")
        diff_lines.append(f"+ head start: {_format_float(duration_s)} s, Maneuver {name}({craft})")
    names = names[early:]
    template = lines[segment.line_index]
    bare = _without_stop(template, newline)
    mission: list[str] = []
    for position, duration_s in enumerate(segment_durations):
        value = duration_s / 86400.0 if segment.unit == "ElapsedDays" else duration_s
        if position == 0 and duration_s == 0.0:
            # A burn at t = 0 directly after the alignment; _schedule_slots allows no other.
            diff_lines.append(f"  segment[{position}]: none, the burn follows the alignment")
        else:
            last = position == len(segment_durations) - 1
            copy = _patch_spans(
                template if last else bare,
                segment.duration_spans,
                [value] * len(segment.duration_spans),
            )
            # A file's last line may have no newline; a Maneuver must not join it.
            if position < len(names) and not copy.endswith(("\n", "\r")):
                copy += newline
            mission.append(copy)
            diff_lines.append(
                f"  segment[{position}].{segment.unit}: {_format_float(duration_s)} s"
            )
        if position < len(names):
            mission.append(f"Maneuver {names[position]}({scenario.pursuer.name});{newline}")
            diff_lines.append(f"+ Maneuver {names[position]}({scenario.pursuer.name})")

    rendered = (
        lines[: begin_index + 1]
        + lines[begin_index + 1 : segment.line_index]
        + mission
        + lines[segment.line_index + 1 :]
    )
    if resources:
        rendered[begin_index:begin_index] = [f"{SYNTHESIS_MARKER}{newline}", *resources]
    return rendered, diff_lines, head_start


#: How far a frameless record's components may sit from the ones recomputed
#: here, 1 mm/s. Two-body states from different propagators put them within
#: about 1e-13 km/s; the wrong frame puts them a sizeable part of |dv| away.
_LEGACY_FRAME_TOLERANCE_KM_S = 1.0e-6


def _assert_burn_frames(scenario: Scenario, candidate: Candidate) -> None:
    """Refuse components written in a frame other than the one they would land in.

    A candidates.json from a run with another policy.synthesized_burn_frame, or
    another template, carries components of the right length that mean a
    different vector here. A record from before burns carried their frame is
    checked by recomputing a synthesized burn's components on the pre-burn
    two-body state; a declared slot's frame never depended on a setting.
    """
    for burn in candidate.burns:
        wanted = burn_frame(scenario, burn.slot_index)
        if burn.frame is None:
            if uses_head_start(candidate.burns):
                raise PolicyError(f"Burn at {burn.time_s:.12g} s of a head-start plan records "
                                  "no frame")
            if template_maneuvers(scenario) or burn.slot_index < len(scenario.burn_slots):
                continue
            state = state_with_burns_at(
                scenario.pursuer.initial_state,
                candidate.burns,
                burn.time_s,
                scenario.mu_km3_s2,
                include_burn_at_time=False,
            )
            expected = inertial_to_script_components(
                np.asarray(burn.dv_inertial_km_s, dtype=float), state, *wanted
            )
            gap = float(np.linalg.norm(expected - np.asarray(burn.dv_script_km_s, dtype=float)))
            if gap > _LEGACY_FRAME_TOLERANCE_KM_S:
                raise PolicyError(
                    f"Burn at {burn.time_s:.12g} s records no frame, and its components are "
                    f"{gap * 1000.0:.6g} m/s from its impulse written in {'/'.join(wanted)}; "
                    "it was written in another frame"
                )
            continue
        if tuple(part.strip().lower() for part in burn.frame) != tuple(
            part.strip().lower() for part in wanted
        ):
            raise PolicyError(
                f"Burn at {burn.time_s:.12g} s was written in {'/'.join(burn.frame)}, "
                f"but slot {burn.slot_index} here is {'/'.join(wanted)}"
            )


def _first_command(lines: list[str]) -> int:
    """The first Propagate or Maneuver line after BeginMissionSequence."""
    begin = next(
        index for index, line in enumerate(lines) if _BEGIN_MISSION_RE.match(_code_part(line))
    )
    return next(
        index
        for index in range(begin + 1, len(lines))
        if _PROPAGATE_LINE_RE.match(lines[index]) or _MANEUVER_LINE_RE.match(lines[index])
    )


def _add_alignment(lines: list[str], scenario: Scenario) -> int | None:
    """Insert the alignment policy.add_epoch_alignment asked for as the first mission
    command, and return its index."""
    alignment = scenario.alignment
    if alignment is None or alignment.line_index is not None:
        return None
    first = _first_command(lines)
    lines.insert(first, alignment.text)
    return first


def with_added_alignment(document: GmatScriptDocument, scenario: Scenario) -> bytes:
    """``document`` with nothing changed but the alignment a render adds
    (policy.add_epoch_alignment)."""
    lines = list(document.lines)
    if "".join(lines) != document.text:  # pragma: no cover - GmatScriptDocument.read holds it
        raise PolicyError(f"{document.path.name}: its lines do not rejoin to its text")
    if _add_alignment(lines, scenario) is None:
        raise PolicyError(f"{document.path.name} needs no epoch alignment added")
    encoded = "".join(lines).encode("utf-8")
    return b"\xef\xbb\xbf" + encoded if document.encoding.has_utf8_bom else encoded


def render_candidate(
    document: GmatScriptDocument,
    scenario: Scenario,
    candidate: Candidate,
    config: OrbitWinConfig,
) -> RenderResult:
    _assert_burn_frames(scenario, candidate)
    burn_times, mission_end, segment_durations = _schedule_slots(scenario, candidate, config)
    changed: set[int] = set()
    diff_lines: list[str] = [f"candidate: {candidate.candidate_id}"]

    if not template_maneuvers(scenario):
        lines, synthesized, head_start = _render_open_template(
            document, scenario, candidate, segment_durations
        )
        diff_lines.extend(synthesized)
        diff_lines.extend(_number_report_filenames(lines, scenario, config))
        if _add_alignment(lines, scenario) is not None:
            diff_lines.append(f"+ alignment: {scenario.alignment.text.strip()}")  # type: ignore[union-attr]
        if head_start:
            # Before the alignment, which is still the first command.
            lines[_first_command(lines):_first_command(lines)] = head_start
        text = "".join(lines)
        encoded = text.encode("utf-8")
        if document.encoding.has_utf8_bom:
            encoded = b"\xef\xbb\xbf" + encoded
        return RenderResult(
            script_bytes=encoded,
            burn_times_s=burn_times,
            mission_end_s=mission_end,
            segment_durations_s=segment_durations,
            changed_lines=(),
            semantic_diff="\n".join(diff_lines) + "\n",
        )

    lines = list(document.lines)
    candidate_burns = {burn.slot_index: burn for burn in candidate.burns}

    for slot_index, slot in enumerate(scenario.burn_slots):
        values = (
            candidate_burns[slot_index].dv_script_km_s
            if slot_index in candidate_burns
            else (0.0, 0.0, 0.0)
        )
        old_values = (slot.element1_km_s, slot.element2_km_s, slot.element3_km_s)
        for component_index, line_index in enumerate(slot.assignment_lines):
            lines[line_index] = _patch_assignment(lines[line_index], float(values[component_index]))
            changed.add(line_index)
            diff_lines.append(
                f"{slot.name}.Element{component_index + 1}: "
                f"{_format_float(old_values[component_index])} -> {_format_float(float(values[component_index]))}"
            )

    for segment_index, (segment, duration_s) in enumerate(
        zip(scenario.propagate_segments, segment_durations, strict=True)
    ):
        if segment.implicit:
            continue
        value = duration_s / 86400.0 if segment.unit == "ElapsedDays" else duration_s
        lines[segment.line_index] = _patch_spans(
            lines[segment.line_index],
            segment.duration_spans,
            [value] * len(segment.duration_spans),
        )
        changed.add(segment.line_index)
        diff_lines.append(
            f"segment[{segment_index}].{segment.unit}: "
            f"{_format_float(segment.duration_s)} s -> {_format_float(duration_s)} s"
        )

    before = list(lines)
    diff_lines.extend(_number_report_filenames(lines, scenario, config))
    changed.update(index for index, line in enumerate(lines) if line != before[index])
    # After the in-place edits, so the line indices above still hold; those
    # past the added line then move down one.
    inserted = _add_alignment(lines, scenario)
    if inserted is not None:
        changed = {index + 1 if index >= inserted else index for index in changed}
        diff_lines.append(f"+ alignment: {scenario.alignment.text.strip()}")  # type: ignore[union-attr]
    text = "".join(lines)
    encoded = text.encode("utf-8")
    if document.encoding.has_utf8_bom:
        encoded = b"\xef\xbb\xbf" + encoded
    return RenderResult(
        script_bytes=encoded,
        burn_times_s=burn_times,
        mission_end_s=mission_end,
        segment_durations_s=segment_durations,
        changed_lines=tuple(sorted(changed)),
        semantic_diff="\n".join(diff_lines) + "\n",
    )


def write_rendered_candidate(
    output_script: str | Path,
    output_diff: str | Path,
    document: GmatScriptDocument,
    scenario: Scenario,
    candidate: Candidate,
    config: OrbitWinConfig,
) -> RenderResult:
    result = render_candidate(document, scenario, candidate, config)
    script_path = Path(output_script)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_bytes(result.script_bytes)
    Path(output_diff).write_text(result.semantic_diff, encoding="utf-8")
    assert_final_script_policy(document.path, script_path, config, probe=scenario.state_probe)
    return result


_PROPAGATE_LINE_RE = re.compile(r"^\s*Propagate\b", re.IGNORECASE)
_MANEUVER_LINE_RE = re.compile(r"^\s*Maneuver(?:\s+'[^']*')?\s+(?P<burn>[A-Za-z_]\w*)\(", re.IGNORECASE)
_ELEMENT_LINE_RE = re.compile(
    r"^\s*(?:GMAT\s+)?(?P<object>[A-Za-z_]\w*)\.Element[123]\s*=", re.IGNORECASE
)
_CREATE_BURN_RE = re.compile(
    r"^\s*(?:GMAT\s+)?Create\s+ImpulsiveBurn\s+(?P<names>[^;]+);", re.IGNORECASE
)
_FIELD_LINE_RE = re.compile(
    r"^\s*(?:GMAT\s+)?(?P<object>[A-Za-z_]\w*)\.[A-Za-z_]\w*\s*=", re.IGNORECASE
)
SYNTHESIS_MARKER = "% Maneuvers added by OrbitWin; every other line is the organizer's."


def _declared_burn_names(document: GmatScriptDocument) -> set[str]:
    names: set[str] = set()
    for line in document.lines:
        match = _CREATE_BURN_RE.match(line)
        if match:
            names.update(split_create_names(match.group("names")))
    return names


def _assert_ascii(document: GmatScriptDocument) -> None:
    """A submitted script has to be loadable, and GMAT will not load this one.

    Measured on GMAT R2026a: a script containing any byte above 127 is refused
    outright, before anything is interpreted --

        Interpreter Exception: The script "..." contains characters outside of
        the ASCII character set; please fix the file before proceeding.

    A submission that cannot be opened scores nothing, and this is the sort of
    thing that arrives by accident: a pasted comment, a name with an accent, a
    non-breaking space from a document. Cheap to check, and the alternative is
    finding out on the organizer's machine.
    """
    for index, line in enumerate(document.lines):
        try:
            line.encode("ascii")
        except UnicodeEncodeError as exc:
            offending = line[exc.start : exc.end]
            raise PolicyError(
                f"Final script line {index + 1} contains the non-ASCII character "
                f"{offending!r}; GMAT R2026a refuses to interpret such a script at all"
            ) from exc


def _burn_field_values(document: GmatScriptDocument, burn_name: str) -> dict[str, str]:
    """Every ``<burn_name>.<field> = <value>`` the script writes, as text."""
    pattern = re.compile(
        rf"^\s*(?:GMAT\s+)?{re.escape(burn_name)}\.(?P<field>[A-Za-z_]\w*)\s*=\s*"
        rf"(?P<value>.*?)\s*;",
        re.IGNORECASE,
    )
    values: dict[str, str] = {}
    for line in document.lines:
        match = pattern.match(line)
        if match:
            values[match.group("field")] = match.group("value")
    return values


def _maneuvered_burn_names(document: GmatScriptDocument) -> set[str]:
    names: set[str] = set()
    for line in document.lines:
        match = _MANEUVER_LINE_RE.match(line)
        if match:
            names.add(match.group("burn"))
    return names


def _assert_only_maneuvers_were_written(
    original_document: GmatScriptDocument, generated_document: GmatScriptDocument
) -> None:
    """Everything in the organizer's file is still there, and what is new is ours.

    This used to be a line-count equality plus a whitelist of line indices,
    which is the right check for a template that already declares the
    maneuvers: nothing is added, and only the numbers move. A template whose
    mission sequence is a single Propagate declares none, and the maneuvers
    §2 requires have to be written into it, so the final script is necessarily
    longer than the original.

    So the check is now a diff rather than an index set, and it is not weaker.
    Every line the generated script does not share with the original has to be
    a maneuver -- a Create ImpulsiveBurn block for a resource the original did
    not declare, a Maneuver command naming one of those, a Propagate command,
    a burn component assignment, or blank. Anything else, anywhere in the file,
    is refused by name and line number.
    """
    added_burns = _declared_burn_names(generated_document) - _declared_burn_names(original_document)
    # A resource the original declares but never maneuvers is one the organizer
    # provided for the contestant to use. Wiring it into the mission is the
    # smallest edit that uses it, and is no more than adding our own would be.
    maneuverable = added_burns | (
        _declared_burn_names(original_document) - _maneuvered_burn_names(original_document)
    )

    def removable(line: str) -> bool:
        # A Propagate the mission split, or a component the candidate rewrote.
        return bool(_PROPAGATE_LINE_RE.match(line) or _ELEMENT_LINE_RE.match(line))

    def addable(line: str) -> str | None:
        if not line.strip():
            return None
        if line.strip() == SYNTHESIS_MARKER:
            return None
        if _PROPAGATE_LINE_RE.match(line) or _ELEMENT_LINE_RE.match(line):
            return None
        maneuver = _MANEUVER_LINE_RE.match(line)
        if maneuver:
            if maneuver.group("burn") in maneuverable:
                return None
            return (
                f"Maneuver of {maneuver.group('burn')}, which the original mission already "
                "maneuvers"
            )
        create = _CREATE_BURN_RE.match(line)
        if create:
            names = set(split_create_names(create.group("names")))
            if names <= added_burns:
                return None
            return f"Create of {sorted(names - added_burns)}"
        field = _FIELD_LINE_RE.match(line)
        if field:
            if field.group("object") in added_burns:
                return None
            return f"assignment to {field.group('object')}, which is not a maneuver we added"
        return "line that is not part of a maneuver"

    matcher = difflib.SequenceMatcher(
        None, list(original_document.lines), list(generated_document.lines), autojunk=False
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for offset, line in enumerate(original_document.lines[i1:i2]):
            if not removable(line):
                raise PolicyError(
                    f"Final script changed or dropped line {i1 + offset + 1} of the official "
                    f"script: {line.strip()!r}"
                )
        for offset, line in enumerate(generated_document.lines[j1:j2]):
            reason = addable(line)
            if reason is not None:
                raise PolicyError(
                    f"Final script adds a {reason} at line {j1 + offset + 1}: {line.strip()!r}"
                )


def assert_final_script_policy(
    original_path: str | Path,
    generated_path: str | Path,
    config: OrbitWinConfig,
    *,
    probe: StateProbe | None = None,
) -> None:
    """``probe`` is the original's state probe; the generated script must keep its craft."""
    original_scenario, original_document = parse_gmat_script(original_path, config, probe=probe)
    generated_scenario, generated_document = parse_gmat_script(
        generated_path, config, probe=probe
    )
    _assert_ascii(generated_document)
    # The GUI reads the plot lines; GmatConsole and the console copy never do.
    assert_plot_lines_unchanged(original_document, generated_document)
    # The one edit outside the maneuvers: the team number in the report name.
    official_reports = official_report_files(original_scenario, config)
    _assert_report_names(official_reports, generated_scenario.report_files, config)
    numbered = list(original_document.lines)
    _number_report_filenames(numbered, original_scenario, config)
    _assert_only_maneuvers_were_written(
        replace(original_document, lines=tuple(numbered), text="".join(numbered)),
        generated_document,
    )
    _assert_alignment_kept(original_scenario, generated_scenario, config)
    _assert_head_start_allowed(original_scenario, generated_scenario)

    def spacecraft_signature(scenario: Scenario) -> tuple[object, ...]:
        return (
            scenario.pursuer.name,
            scenario.pursuer.epoch_text,
            scenario.pursuer.coordinate_system,
            scenario.pursuer.display_state_type,
            scenario.pursuer.fields,
            scenario.target.name,
            scenario.target.epoch_text,
            scenario.target.coordinate_system,
            scenario.target.display_state_type,
            scenario.target.fields,
        )

    if spacecraft_signature(original_scenario) != spacecraft_signature(generated_scenario):
        raise PolicyError("Spacecraft initial configuration changed")
    if original_scenario.force_model_fields != generated_scenario.force_model_fields:
        raise PolicyError("ForceModel changed")
    if original_scenario.propagator_fields != generated_scenario.propagator_fields:
        raise PolicyError("Propagator changed")
    if official_reports != generated_scenario.report_files:
        raise PolicyError("Official ReportFile contract changed")
    original_burn_contract = [
        (slot.name, slot.spacecraft, slot.coordinate_system, slot.origin, slot.axes)
        for slot in original_scenario.burn_slots
    ]
    generated_burn_contract = [
        (slot.name, slot.spacecraft, slot.coordinate_system, slot.origin, slot.axes)
        for slot in generated_scenario.burn_slots
    ]
    if template_maneuvers(original_scenario):
        # A template that already maneuvers keeps its resources exactly: the
        # frames are the organizer's and only the components move.
        if original_burn_contract != generated_burn_contract:
            raise PolicyError("Burn resource contract changed")
        # Nor may a Propagate go: after an alignment its absence reads as the
        # implicit zero-length segment, and the count would still agree.
        implicit = [segment.implicit for segment in original_scenario.propagate_segments]
        if implicit != [segment.implicit for segment in generated_scenario.propagate_segments]:
            raise PolicyError(
                "Final script changes which Propagate segments the template has; only "
                "their durations may move"
            )
        if _segment_shapes(original_scenario, original_document) != _segment_shapes(
            generated_scenario, generated_document
        ):
            raise PolicyError(
                "Final script changes a Propagate command other than its durations: its "
                "propagators, craft, goals or options"
            )
    else:
        # Every segment is a copy of the template's one Propagate, durations aside.
        # A copy of a last line without a line end may carry the file's newline,
        # and one before the last drops a stop character the line ends in.
        newline = original_document.encoding.newline
        template_shape = _segment_shapes(original_scenario, original_document)
        template_shape += [
            shape + newline for shape in template_shape if not shape.endswith(("\n", "\r"))
        ]
        template_shape += [_without_stop(shape, newline) for shape in template_shape]
        for shape in _segment_shapes(generated_scenario, generated_document):
            if shape not in template_shape:
                raise PolicyError(
                    f"Final script carries a Propagate that is not the template's own with "
                    f"other durations: {shape.strip()!r}"
                )
        declared_frames = {
            slot.name: (slot.coordinate_system, slot.origin, slot.axes)
            for slot in original_scenario.burn_slots
        }
        # A template that declares none has its maneuvers written by us, so
        # there is no contract to preserve -- but there is one to meet. Every
        # maneuver must be the pursuer's and must use the frame this project
        # emits, so a synthesized script cannot quietly acquire a burn on the
        # target or in a frame nothing converted the impulse into.
        coordinate_system, origin, axes = synthesized_burn_frame(original_scenario)
        expected_fields = {
            "CoordinateSystem": coordinate_system,
            "Origin": origin,
            "Axes": axes,
            # false is not a default worth inheriting silently: §5 sums the
            # impulse magnitudes and says nothing about mass, and under a force
            # model with drag or SRP a decrementing mass makes the trajectory
            # depend on the order the delta-v was spent in.
            "DecrementMass": "false",
        }
        for slot in generated_scenario.burn_slots:
            if slot.spacecraft != generated_scenario.pursuer.name:
                raise PolicyError(f"Synthesized burn {slot.name} maneuvers {slot.spacecraft}")
            if slot.name in declared_frames:
                # Reused, so the organizer's declaration is the contract.
                if (slot.coordinate_system, slot.origin, slot.axes) != declared_frames[slot.name]:
                    raise PolicyError(
                        f"{slot.name} was redeclared in "
                        f"{slot.coordinate_system}/{slot.origin}/{slot.axes}, not the "
                        f"{'/'.join(declared_frames[slot.name])} the original declares"
                    )
                continue
            written = _burn_field_values(generated_document, slot.name)
            for field, expected in expected_fields.items():
                actual = written.get(field)
                if actual is None or actual.strip().lower() != expected.lower():
                    raise PolicyError(
                        f"Synthesized burn {slot.name}.{field} is {actual!r}, not the "
                        f"{expected!r} this project emits"
                    )
    _assert_generated_values_are_legal(original_scenario, generated_scenario, config)


def _assert_report_names(
    official: tuple[ReportFileSpec, ...],
    generated: tuple[ReportFileSpec, ...],
    config: OrbitWinConfig,
) -> None:
    """Name the report whose Filename moved; the full contract is compared later."""
    written = {spec.name: spec.filename for spec in generated}
    for spec in official:
        actual = written.get(spec.name)
        if actual is not None and actual != spec.filename:
            raise PolicyError(
                f"Final script sets {spec.name}.Filename to {actual!r}, not {spec.filename!r}; "
                "only its Team placeholder may change, to policy.team_number "
                f"({config.policy.team_number})"
            )


def _without_stop(line: str, newline: str) -> str:
    """``line``'s command alone, with a line end, if it holds a NUL or Ctrl-Z, where
    GMAT stops reading. Its comment goes too: one ending in `...` before the stop
    would continue the copy into the Maneuver after it."""
    stops = [index for index in (line.find("\x00"), line.find("\x1a")) if index >= 0]
    if not stops:
        return line
    command = _code_part(line[: min(stops)]).rstrip(" \t")
    return command + (line[len(line.rstrip("\r\n")):] or newline)


def _segment_shapes(scenario: Scenario, document: GmatScriptDocument) -> list[str]:
    """Each explicit segment's line with its durations blanked: all a render may change."""
    shapes: list[str] = []
    for segment in scenario.propagate_segments:
        if segment.implicit:
            continue
        line = document.lines[segment.line_index]
        for start, end in sorted(segment.duration_spans, reverse=True):
            line = line[:start] + "<duration>" + line[end:]
        shapes.append(line)
    return shapes


def _assert_alignment_kept(
    original: Scenario, generated: Scenario, config: OrbitWinConfig
) -> None:
    """The alignment is the organizer's line byte for byte, or exactly the one we add."""
    want, got = original.alignment, generated.alignment
    if want is None:
        if got is not None:
            raise PolicyError(
                f"Final script adds an epoch alignment, {got.text.strip()!r}, that the "
                "original does not need"
            )
        return
    if want.line_index is None and not config.policy.add_epoch_alignment:  # pragma: no cover
        # The parser refuses the original first; this holds if that ever moves.
        raise PolicyError("An epoch alignment may be added only under policy.add_epoch_alignment")
    if got is None or got.line_index is None:
        raise PolicyError(
            f"Final script drops the epoch alignment {want.text.strip()!r}"
        )
    if got.text != want.text:
        raise PolicyError(
            f"Final script changes the epoch alignment {want.text.strip()!r} to "
            f"{got.text.strip()!r}"
        )
    if (got.early, got.late, got.propagator, got.form) != (
        want.early, want.late, want.propagator, want.form
    ):  # pragma: no cover - same text, same reading
        raise PolicyError("Final script reads its epoch alignment differently")


def _assert_head_start_allowed(original: Scenario, generated: Scenario) -> None:
    """Burns before the alignment only where policy.head_start_burns opens the
    original's head start; the parser holds their shape (gmat_parser._check_head_start)."""
    if not generated.head_start_segments:
        return
    lead_s = head_start_lead_s(original)
    if lead_s <= 0.0:
        raise PolicyError(
            "Final script burns before the epoch alignment, and the original has no head "
            f"start to burn in: {head_start_status(original) or 'policy.head_start_burns is off'}"
        )
    if generated.head_start_s != lead_s:  # pragma: no cover - same alignment, same probe
        raise PolicyError("Final script reads its head start differently from the original")


def _assert_generated_values_are_legal(
    original_scenario: Scenario,
    generated_scenario: Scenario,
    config: OrbitWinConfig,
) -> None:
    """Check that the values which *were* allowed to change are still legal.

    The comparisons above establish that nothing forbidden moved. They say
    nothing about whether what did move is within the rules, so on their own
    they would accept a final script carrying a 99 km/s burn.

    The formal limits are used here, not the production safety margins. This is
    the rule boundary, and a candidate deliberately parked between the safe
    limit and the formal one must not be rejected by the guard that exists to
    catch rule violations.
    """
    policy = config.policy
    # Only the burns the mission performs. A resource the script declares but
    # never maneuvers has no epoch and no effect -- 初賽規則 §5 counts
    # maneuvers, and there is none -- so it has nothing to be checked against.
    # Checking it anyway indexed an epoch that does not exist.
    maneuvered = [
        slot for slot in generated_scenario.burn_slots if slot.maneuver_line is not None
    ]
    # The parser guarantees a strictly alternating Propagate/Maneuver sequence
    # with one more segment than burns, so burn k fires after segments 0..k;
    # head-start burns first, before t = 0.
    burn_times_s = [maneuver.time_s for maneuver in maneuvers_flown(generated_scenario)]
    elapsed = 0.0
    for index, segment in enumerate(generated_scenario.propagate_segments):
        if not segment.implicit and segment.duration_s < policy.min_propagate_duration_s:
            raise PolicyError(
                f"Propagate segment {index} lasts {segment.duration_s:.12g} s, below the "
                f"{policy.min_propagate_duration_s:.12g} s floor; a zero-length goal hangs GMAT"
            )
        elapsed += segment.duration_s
    if len(burn_times_s) != len(maneuvered):  # pragma: no cover - the parser pairs them
        raise PolicyError("Final script's Maneuvers do not pair with its Propagate segments")
    start_s = -head_start_lead_s(original_scenario)

    horizon_s = original_scenario.max_time_s
    if elapsed > horizon_s + 1.0e-8:
        raise PolicyError(
            f"Final script propagates {elapsed:.12g} s, beyond the {horizon_s:.12g} s mission horizon"
        )

    for index, slot in enumerate(maneuvered):
        # Every supported burn frame is orthonormal, so the component norm is
        # the impulse magnitude regardless of Local/VNB, Local/LVLH or MJ2000Eq.
        magnitude = float(math.sqrt(sum(value * value for value in slot.vector)))
        if not any(slot.vector) and not policy.allow_zero_vector_unused_slots:
            # The renderer refuses to *create* one of these; this refuses to
            # *submit* one, and the two were not symmetric. Measured: with the
            # flag off, render_candidate raised PolicyError while
            # assert_final_script_policy passed the very script an earlier run
            # had produced with the flag on -- which is the artifact the
            # operator uploads, and every final output must pass this gate. A
            # team told "zero-vector maneuvers are not allowed" would have set
            # the flag, seen the policy check go green,
            # and submitted the thing it forbids.
            raise PolicyError(
                f"{slot.name} carries an exact zero vector and "
                "policy.allow_zero_vector_unused_slots is off. Obtain the organizer's "
                "written permission, or re-render from a candidate that fills every slot."
            )
        if magnitude > policy.dv_limit_km_s + 1.0e-12:
            raise PolicyError(
                f"{slot.name} magnitude {magnitude:.12g} km/s exceeds the formal limit "
                f"{policy.dv_limit_km_s:.12g} km/s"
            )
        if burn_times_s[index] < start_s - 1.0e-9:
            raise PolicyError(f"{slot.name} fires before mission start")
        if burn_times_s[index] > horizon_s + 1.0e-9:
            raise PolicyError(f"{slot.name} fires after the mission horizon")
        if index:
            separation = burn_times_s[index] - burn_times_s[index - 1]
            if separation < policy.min_burn_separation_s - 1.0e-9:
                raise PolicyError(
                    f"{maneuvered[index - 1].name} to {slot.name} separation "
                    f"{separation:.12g} s is below the formal {policy.min_burn_separation_s:.12g} s minimum"
                )

    if policy.total_dv_limit_km_s is not None:
        total = sum(
            float(math.sqrt(sum(value * value for value in slot.vector))) for slot in maneuvered
        )
        if total > policy.total_dv_limit_km_s:
            raise PolicyError(
                f"Final script spends {total:.12g} km/s in total, over the "
                f"{policy.total_dv_limit_km_s:.12g} km/s limit"
            )
