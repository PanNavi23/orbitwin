"""The report row GMAT leaves out after a Maneuver.

R2026a writes one row where a Propagate ends, before the Maneuver, and none
after it: the next row is one step later, and GMAT starts each Propagate at
the propagator's InitialStepSize. Hermite between those two rows uses the
pre-burn velocity at the burn. With InitialStepSize 60 s that put the
pursuer 1.51 and 2.21 km off 20 s after each of two burns (issue #106); the
review measured 1.5-12.9 km at 60 s and 0.2-8 mm at InitialStepSize 1 s, with
MaxStep 60 s either way. The missing row is the pre-burn one with the script
components applied on GMAT's own state, which the review measured against
GMAT's own post-burn state to 9e-13 m/s; read with it, the same two points
were 0.28 and 0.25 m off.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import GmatRunError
from .frames import inertial_to_script_components, script_components_to_inertial
from .models import ArrayLike3, Scenario, StateVector, as_vector3

#: How far a row may sit from the summed Propagate goals and still be the burn
#: epoch's. Measured on R2026a: 0.35 us at 300 s, 0.9 us over 20 burns to
#: 9.4 ks, 1.87 us over 6 burns to 40 ks, growing per segment; GMAT's own help
#: says it propagates "to a time quantization of a few microseconds". Segments
#: are at least 1e-3 s, so this cannot reach another segment's end.
BURN_EPOCH_TOLERANCE_S = 1.0e-4

#: A second row at a burn epoch is the post-burn one when its velocity jump is
#: the impulse to this fraction of the impulse's length. A written post-burn
#: row matches the rebuilt one to 9e-13 m/s. Two pre-burn rows within
#: BURN_EPOCH_TOLERANCE_S of the epoch are up to 2e-4 s apart and differ by
#: gravity over that: 1.7e-3 m/s at 500 km altitude, at most 2.0e-3 m/s at
#: the surface. Their jump falls within 1% of an impulse only if the impulse is
#: within 1% of that length, so any impulse over 2.1 mm/s is told apart.
POST_BURN_MATCH = 1.0e-2

#: How far the impulse GMAT flew may be from the script's components, in the
#: burn's frame on GMAT's pre-burn state. Precision 17 rows are bit-exact, and
#: R2026a matched to 2.7e-16 km/s in Local/VNB, Local/LVLH and MJ2000Eq (E2E
#: audit A4); the ranking rules' per-burn tolerance ε_v is 1e-8 km/s.
FLOWN_IMPULSE_TOLERANCE_KM_S = 1.0e-10
#: A Maneuver does not move the craft, and both rows print the same doubles.
_SAME_EPOCH_S = 1.0e-6
_SAME_POSITION_KM = 1.0e-9
#: The pursuer's quantities in each burn report row, in the order the sibling asks for them.
BURN_REPORT_QUANTITIES = (
    "A1ModJulian",
    *(f"EarthMJ2000Eq.{axis}" for axis in ("X", "Y", "Z", "VX", "VY", "VZ")),
)
_EPOCH = BURN_REPORT_QUANTITIES.index("A1ModJulian")
_R = [BURN_REPORT_QUANTITIES.index(f"EarthMJ2000Eq.{axis}") for axis in ("X", "Y", "Z")]
_V = [BURN_REPORT_QUANTITIES.index(f"EarthMJ2000Eq.{axis}") for axis in ("VX", "VY", "VZ")]


@dataclass(frozen=True, slots=True)
class FlownManeuver:
    """One Maneuver as the script performs it."""

    time_s: float
    components_km_s: ArrayLike3
    #: (CoordinateSystem, Origin, Axes) of its ImpulsiveBurn.
    frame: tuple[str, str, str]
    #: Its ImpulsiveBurn.
    name: str = ""
    #: Its Maneuver command's line in the parsed script.
    line_index: int | None = None


def maneuvers_flown(scenario: Scenario) -> tuple[FlownManeuver, ...]:
    """The script's Maneuvers in order, each at the end of the segments before it.

    Read from a script parsed after rendering, so the epochs are the ones GMAT
    is given. A template that maneuvers nothing gives none. Head-start burns
    come first, before t = 0: the alignment ends Scenario.head_start_s after
    the pursuer's epoch.
    """
    maneuvered = sorted(
        (slot for slot in scenario.burn_slots if slot.maneuver_line is not None),
        key=lambda slot: slot.maneuver_line or 0,
    )
    flown: list[FlownManeuver] = []
    early = len(scenario.head_start_segments)
    elapsed_s = -scenario.head_start_s
    for segment, slot in zip(scenario.head_start_segments, maneuvered[:early], strict=True):
        elapsed_s += segment.duration_s
        flown.append(FlownManeuver(
            elapsed_s, as_vector3(slot.vector),
            (slot.coordinate_system, slot.origin, slot.axes), slot.name, slot.maneuver_line,
        ))
    maneuvered = maneuvered[early:]
    elapsed_s = 0.0
    for index, segment in enumerate(scenario.propagate_segments):
        elapsed_s += segment.duration_s
        if index < len(maneuvered):
            slot = maneuvered[index]
            flown.append(
                FlownManeuver(
                    elapsed_s,
                    as_vector3(slot.vector),
                    (slot.coordinate_system, slot.origin, slot.axes),
                    slot.name,
                    slot.maneuver_line,
                )
            )
    return tuple(flown)


def check_flown_impulses(
    path: str | Path | None,
    maneuvers: Sequence[FlownManeuver],
    *,
    complete: bool,
) -> None:
    """Refuse a run in which GMAT flew another impulse than the script's Element1-3.

    ``path`` is the sibling's burn report (diagnostic.add_diagnostic_report with
    check_burns): the pursuer's A1 epoch, position and velocity before and after
    each Maneuver, in order, as BURN_REPORT_QUANTITIES. A run that stopped
    early (``complete`` False) is checked on the pairs it wrote.

    An exact-zero claim is checked on the jump's length, which no frame
    changes, so a slot left unused in a frame frames.py cannot build passes.

    E2E audit A4: an ImpulsiveBurn alias assigned after the Element lines made
    GMAT fly 70.69 m/s where every gate claimed 91.15.
    """
    if path is None:
        if complete:
            raise GmatRunError("The run wrote no burn report, so the impulses flown are unknown")
        return
    rows = [
        line.split() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    expected = 2 * len(maneuvers)
    if len(rows) > expected or (complete and len(rows) != expected):
        raise GmatRunError(
            f"{path} has {len(rows)} rows for {len(maneuvers)} Maneuvers; it needs two each"
        )
    for index, maneuver in enumerate(maneuvers[: len(rows) // 2]):
        pair = rows[2 * index : 2 * index + 2]
        try:
            pre, post = (np.array([float(value) for value in row]) for row in pair)
        except ValueError as exc:
            raise GmatRunError(f"{path} row {2 * index + 1} or {2 * index + 2}: {exc}") from exc
        width = len(BURN_REPORT_QUANTITIES)
        if pre.shape != (width,) or post.shape != (width,) or not np.all(np.isfinite([pre, post])):
            raise GmatRunError(
                f"{path} rows {2 * index + 1}-{2 * index + 2} are not {width} finite values"
            )
        label = f"Maneuver {index + 1} ({maneuver.name or 'unnamed'}, t = {maneuver.time_s!r} s)"
        if abs(post[_EPOCH] - pre[_EPOCH]) * 86400.0 > _SAME_EPOCH_S or (
            np.linalg.norm(post[_R] - pre[_R]) > _SAME_POSITION_KM
        ):
            raise GmatRunError(f"{path}: the rows around {label} are not one instant and place")
        claimed = np.asarray(maneuver.components_km_s, dtype=float)
        jump = post[_V] - pre[_V]
        frame = "/".join(maneuver.frame)
        if np.any(claimed):
            flown = inertial_to_script_components(
                jump, StateVector(as_vector3(pre[_R]), as_vector3(pre[_V])), *maneuver.frame
            )
        else:
            flown, frame = jump, "EarthMJ2000Eq"
        gap_km_s = float(np.linalg.norm(flown - claimed))
        if not gap_km_s <= FLOWN_IMPULSE_TOLERANCE_KM_S:
            raise GmatRunError(
                f"GMAT flew {label} as {flown.tolist()} km/s in {frame}, "
                f"|dv| {float(np.linalg.norm(flown)):.9g} km/s; the script's Element1-3 say "
                f"{claimed.tolist()}, |dv| {float(np.linalg.norm(claimed)):.9g} km/s "
                f"({gap_km_s:.3g} km/s apart, tolerance {FLOWN_IMPULSE_TOLERANCE_KM_S:g}). "
                "Something in the script changes the impulse GMAT flies"
            )


def with_post_burn_rows(
    times_s: np.ndarray,
    pursuer_r: np.ndarray,
    pursuer_v: np.ndarray,
    target_r: np.ndarray,
    target_v: np.ndarray,
    maneuvers: Sequence[FlownManeuver],
    source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows with a post-burn row after each Maneuver's pre-burn one.

    A Maneuver after the last row (a run that stopped early) is skipped; one
    on the last row gets its post-burn row appended. One inside the report
    without a row at its epoch is refused: Hermite across it would be wrong and
    nothing could say by how much.

    Two rows at an epoch are not evidence of a post-burn row. R2026a writes
    two pre-burn rows when an integrator step lands within the tolerance of
    the Propagate goal (300.000004 s against 300 s, 3.3e-5 m/s apart), so the
    later row is taken as post-burn only when the jump between them is the
    impulse, to POST_BURN_MATCH of its length.
    """
    for maneuver in sorted(maneuvers, key=lambda item: item.time_s, reverse=True):
        if not np.any(np.asarray(maneuver.components_km_s, dtype=float)):
            continue
        if len(times_s) == 0 or maneuver.time_s > times_s[-1] + BURN_EPOCH_TOLERANCE_S:
            continue
        at = np.flatnonzero(np.abs(times_s - maneuver.time_s) <= BURN_EPOCH_TOLERANCE_S)
        if len(at) == 0:
            raise GmatRunError(f"{source} has no row at the burn epoch {maneuver.time_s!r} s")
        row = int(at[-1])
        impulse = script_components_to_inertial(
            np.asarray(maneuver.components_km_s, dtype=float),
            StateVector(as_vector3(pursuer_r[at[0]]), as_vector3(pursuer_v[at[0]])),
            *maneuver.frame,
        )
        if len(at) > 1:
            jump = pursuer_v[row] - pursuer_v[at[0]]
            if np.linalg.norm(jump - impulse) <= POST_BURN_MATCH * np.linalg.norm(impulse):
                continue
            impulse = script_components_to_inertial(
                np.asarray(maneuver.components_km_s, dtype=float),
                StateVector(as_vector3(pursuer_r[row]), as_vector3(pursuer_v[row])),
                *maneuver.frame,
            )
        times_s = np.insert(times_s, row + 1, times_s[row])
        pursuer_r = np.insert(pursuer_r, row + 1, pursuer_r[row], axis=0)
        pursuer_v = np.insert(pursuer_v, row + 1, pursuer_v[row] + impulse, axis=0)
        target_r = np.insert(target_r, row + 1, target_r[row], axis=0)
        target_v = np.insert(target_v, row + 1, target_v[row], axis=0)
    return times_s, pursuer_r, pursuer_v, target_r, target_v


def split_at_alignment(
    maneuvers: Sequence[FlownManeuver],
) -> tuple[tuple[FlownManeuver, ...], tuple[FlownManeuver, ...]]:
    """Head-start maneuvers, in the alignment's rows, and the rest, from t = 0 on."""
    before = tuple(m for m in maneuvers if m.time_s < -BURN_EPOCH_TOLERANCE_S)
    return before, tuple(m for m in maneuvers if m.time_s >= -BURN_EPOCH_TOLERANCE_S)
