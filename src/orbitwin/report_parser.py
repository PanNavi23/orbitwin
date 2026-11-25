from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .burn_rows import FlownManeuver, split_at_alignment, with_post_burn_rows
from .errors import GmatRunError
from .event_detection import EventResult, evaluate_hermite_samples
from .models import EpochAlignment
from .report_format import (
    DEFAULT_REPORT_FORMAT,
    ReportFormat,
    read_rows,
)

#: Report rows written with Precision = 16 resolve ElapsedSecs to about 1e-11 s
#: over a day; a synchronized pair measured 0 to the nanosecond on R2026a.
EPOCH_TOLERANCE_S = 1.0e-6

#: Geodetic altitude is never below r minus the equatorial radius; this is
#: rounding, and anything further means the columns are not what they say.
GEODETIC_SLACK_KM = 1.0e-6

#: Share of policy.stop_depth_km a stop may fall short by before verify says
#: so: room for Hermite error on coarse rows, far under any real shortfall.
STOP_DEPTH_TOLERANCE = 0.01


def stop_depth_used_km(stop_depth_km: float, intercept_radius_km: float) -> float:
    """The depth a hit is stopped at: policy.stop_depth_km, at most half the radius."""
    return min(stop_depth_km, 0.5 * intercept_radius_km)


@dataclass(frozen=True, slots=True)
class NumericReport:
    headers: tuple[str, ...]
    data: np.ndarray

    def column(self, *candidates: str) -> np.ndarray:
        normalized = {header.lower(): index for index, header in enumerate(self.headers)}
        for candidate in candidates:
            lowered = candidate.lower()
            if lowered in normalized:
                return self.data[:, normalized[lowered]]
            matching = [
                index
                for index, header in enumerate(self.headers)
                if header.lower().endswith(lowered)
            ]
            if len(matching) == 1:
                return self.data[:, matching[0]]
        raise GmatRunError(f"Report lacks required column; tried {candidates}")


def parse_numeric_report(
    path: str | Path,
    report_format: ReportFormat = DEFAULT_REPORT_FORMAT,
    column_names: Sequence[str] | None = None,
) -> NumericReport:
    """Read a report written in `report_format`, every value finite.

    `column_names` (the ReportFile's Add list) names the columns of a report
    without headers, and places a text column in one with headers.
    """
    headers, data = read_rows(path, report_format, column_names)
    if not headers:
        raise GmatRunError(
            f"Report {path} has no header row; its column names must come from the "
            "ReportFile's Add list"
        )
    if len(headers) != data.shape[1]:
        raise GmatRunError(
            f"Report {path} has {data.shape[1]} values a row for {len(headers)} headers"
        )
    bad = np.flatnonzero(~np.all(np.isfinite(data), axis=1))
    if bad.size:
        raise GmatRunError(f"Report {path} data row {bad[0] + 1} contains NaN or infinity")
    return NumericReport(headers, data)


@dataclass(frozen=True, slots=True)
class ReportClock:
    """Each row on the solver clock, and how many leading rows the alignment wrote."""

    #: The late craft's ElapsedSecs from t = 0; before it, the early craft's
    #: less its ElapsedSecs at t = 0, so negative.
    times_s: np.ndarray
    #: Rows written while the late craft waited at its epoch, before t = 0.
    #: Exactly one row sits at t = 0 and is not among them.
    frozen_rows: int


def _a1_columns(report: NumericReport, *names: str) -> list[np.ndarray] | None:
    headers = {header.lower() for header in report.headers}
    if not all(f"{name}.A1ModJulian".lower() in headers for name in names):
        return None
    return [report.column(f"{name}.A1ModJulian") for name in names]


def report_clock(
    report: NumericReport,
    pursuer_name: str,
    target_name: str,
    alignment: EpochAlignment | None = None,
) -> ReportClock:
    """The rows on the solver clock, after checking both craft are on one epoch.

    Each row holds both craft at their own epochs. Without Synchronized, GMAT
    stops them at different ones (R2026a: 167 s with 60 s adaptive steps), and
    a distance between rows taken at two instants is not a distance. With both
    A1ModJulian columns the check is on absolute epochs, since ElapsedSecs
    counts from each craft's own epoch; only the alignment's leading rows may
    differ, with the late craft frozen there (docs/ASYNC_START_AND_INSERTION.md §4).
    """
    apart = alignment is not None and alignment.offset_s > 0.0
    a1 = _a1_columns(report, pursuer_name, target_name)
    if a1 is None:
        if apart:
            raise GmatRunError(
                f"The report has no {pursuer_name}.A1ModJulian and {target_name}.A1ModJulian "
                "columns, so it cannot show the craft on one epoch after the alignment"
            )
        times = report.column(f"{pursuer_name}.ElapsedSecs", "ElapsedSecs")
        lag = np.abs(times - report.column(f"{target_name}.ElapsedSecs"))
        if lag.size and float(lag.max()) > EPOCH_TOLERANCE_S:
            row = int(lag.argmax())
            raise GmatRunError(
                f"{pursuer_name} and {target_name} are {lag[row]:.6g} s apart in epoch at report "
                f"row {row + 1}; the Propagate commands do not hold them together"
            )
        return ReportClock(times, 0)
    lag = np.abs(a1[0] - a1[1]) * 86400.0
    off = lag > EPOCH_TOLERANCE_S
    frozen = int(np.argmin(off)) if not off.all() else len(off)
    later = np.flatnonzero(off[frozen:])
    if later.size or (frozen and not apart):
        row = frozen + int(later[0]) if later.size else int(np.argmax(off))
        raise GmatRunError(
            f"{pursuer_name} and {target_name} are {lag[row]:.6g} s apart in epoch (A1) at "
            f"report row {row + 1}; the Propagate commands do not hold them together"
        )
    if frozen == len(off):
        raise GmatRunError(
            f"{pursuer_name} and {target_name} never reach one A1 epoch in the report; the "
            "alignment did not end at the later craft's epoch"
        )
    if alignment is None or not apart:
        return ReportClock(report.column(f"{pursuer_name}.ElapsedSecs"), 0)
    early = report.column(f"{alignment.early}.ElapsedSecs")
    late = report.column(f"{alignment.late}.ElapsedSecs")
    late_a1 = a1[0] if alignment.late == pursuer_name else a1[1]
    waiting = late[: frozen + 1]
    if np.any(np.abs(waiting) > EPOCH_TOLERANCE_S) or np.any(
        np.abs(late_a1[: frozen + 1] - late_a1[frozen]) * 86400.0 > EPOCH_TOLERANCE_S
    ):
        raise GmatRunError(
            f"{alignment.late} moves in the report before {alignment.early} reaches its epoch; "
            "the alignment is not the only command before t = 0"
        )
    times = np.concatenate((early[:frozen] - early[frozen], late[frozen:]))
    return ReportClock(times, frozen)


def alignment_row_distance_km(
    path: str | Path, pursuer_name: str, target_name: str, alignment: EpochAlignment | None
) -> float | None:
    """The closest the craft come on the alignment's rows, as the probe measures
    them; None without such rows."""
    report = parse_numeric_report(path)
    frozen = report_clock(report, pursuer_name, target_name, alignment).frozen_rows
    if not frozen:
        return None
    gap = np.column_stack([
        report.column(f"{pursuer_name}.EarthMJ2000Eq.{axis}")[:frozen]
        - report.column(f"{target_name}.EarthMJ2000Eq.{axis}")[:frozen]
        for axis in "XYZ"
    ])
    return float(np.linalg.norm(gap, axis=1).min())


def common_epochs(report: NumericReport, pursuer_name: str, target_name: str) -> np.ndarray:
    """report_clock's times for a report with no alignment rows."""
    return report_clock(report, pursuer_name, target_name).times_s


def relative_event(
    times: np.ndarray,
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    intercept_radius_km: float,
    *,
    stop_depth_km: float | None,
) -> EventResult:
    """The interception event from relative rows, a hit measured on its first passage.

    With ``stop_depth_km`` (policy.stop_depth_km) it also locates where a hit's
    submitted script stops; None leaves stop_time_s unset.
    """
    event = evaluate_hermite_samples(
        times, relative_position, relative_velocity, intercept_radius_km
    )
    if event.first_entry_time_s is not None:
        # A hit is measured on its first passage through the sphere, up to the
        # first row outside it again; a later pass is not the interception.
        distances = np.linalg.norm(relative_position, axis=1)
        first_row = int(np.searchsorted(times, event.first_entry_time_s))
        outside = np.flatnonzero(distances[first_row:] > intercept_radius_km)
        closed = bool(outside.size)
        window = slice(max(first_row - 1, 0), first_row + int(outside[0]) + 1 if closed else None)
        passage = evaluate_hermite_samples(
            times[window], relative_position[window], relative_velocity[window], intercept_radius_km
        )
        # The earliest point stop_depth_km inside, or the deepest point of a
        # passage that never gets that deep. A report that ends inside the
        # passage short of either has not measured where to stop.
        stop_s: float | None = None
        if stop_depth_km is not None:
            deep = evaluate_hermite_samples(
                times[window],
                relative_position[window],
                relative_velocity[window],
                intercept_radius_km - stop_depth_used_km(stop_depth_km, intercept_radius_km),
            )
            stop_s = deep.first_entry_time_s
            if stop_s is None and closed:
                stop_s = passage.minimum_distance_time_s
        event = replace(
            event,
            minimum_distance_km=passage.minimum_distance_km,
            minimum_distance_time_s=passage.minimum_distance_time_s,
            samples_inside_radius=passage.samples_inside_radius,
            stop_time_s=stop_s,
        )
    return event


def evaluate_diagnostic_report(
    path: str | Path,
    pursuer_name: str,
    target_name: str,
    intercept_radius_km: float,
    *,
    central_body_radius_km: float | None = None,
    maneuvers: Sequence[FlownManeuver] = (),
    alignment: EpochAlignment | None = None,
    stop_depth_km: float | None = None,
    geodetic_radius_km: float | None = None,
) -> EventResult:
    """Read the interception event, and how low the pursuer flew, from a run.

    ``stop_depth_km`` is policy.stop_depth_km, for relative_event to locate a
    hit's stop; without it stop_time_s stays None.

    ``maneuvers`` are the script's, from burn_rows.maneuvers_flown; each gets
    the post-burn row GMAT does not write, so no Hermite segment spans a burn.

    ``central_body_radius_km`` answers the same question
    evaluate_analytical_trajectory answers, from the same columns the relative
    distance already comes out of, and it is the better answer for a candidate
    GMAT has replayed: the analytical estimate is two-body, and this is what
    the configured force model actually flew. It is measured over every row
    the pursuer is propagated in, to the end of the report: the alignment when
    the pursuer is the early craft, and past an interception too, since
    nothing flown may go underground (docs/ASYNC_START_AND_INSERTION.md §5).
    Relative quantities ignore the alignment's rows.
    """
    report = parse_numeric_report(path)
    clock = report_clock(report, pursuer_name, target_name, alignment)
    times = clock.times_s

    def state_columns(spacecraft: str, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        position = np.column_stack(
            [
                report.column(f"{spacecraft}.EarthMJ2000Eq.{axis}", f"{spacecraft}.{axis}")
                for axis in ("X", "Y", "Z")
            ]
        )
        velocity = np.column_stack(
            [
                report.column(f"{spacecraft}.EarthMJ2000Eq.{axis}", f"{spacecraft}.{axis}")
                for axis in ("VX", "VY", "VZ")
            ]
        )
        return position, velocity

    pursuer_position, pursuer_velocity = state_columns(pursuer_name, "pursuer")
    target_position, target_velocity = state_columns(target_name, "target")
    # The joint rows GMAT wrote; the alignment's are scored apart (alignment_minimum_distance_km).
    rows_km = np.linalg.norm(
        pursuer_position[clock.frozen_rows:] - target_position[clock.frozen_rows:], axis=1
    )
    closest_row = int(rows_km.argmin())
    closest_row_s = float(times[clock.frozen_rows + closest_row])
    # Head-start burns in the alignment's rows, the rest after; each side gets its
    # post-burn rows, so the joint rows still start at the alignment's end.
    early, later = split_at_alignment(maneuvers)
    parts = [
        with_post_burn_rows(
            times[rows], pursuer_position[rows], pursuer_velocity[rows],
            target_position[rows], target_velocity[rows], burns, str(path),
        )
        for rows, burns in ((slice(0, clock.frozen_rows), early),
                            (slice(clock.frozen_rows, None), later))
    ]
    if early and not len(parts[0][0]):
        raise GmatRunError(f"{path} has no rows of {pursuer_name}'s head start")
    times, pursuer_position, pursuer_velocity, target_position, target_velocity = (
        np.concatenate([before, after]) for before, after in zip(*parts, strict=True)
    )
    joint = slice(len(parts[0][0]), None)
    relative_position = pursuer_position[joint] - target_position[joint]
    relative_velocity = pursuer_velocity[joint] - target_velocity[joint]
    event = relative_event(
        times[joint], relative_position, relative_velocity, intercept_radius_km,
        stop_depth_km=stop_depth_km,
    )
    joint_s = clock.times_s[clock.frozen_rows:]
    event = replace(
        event,
        minimum_row_distance_km=float(rows_km[closest_row]),
        minimum_row_distance_time_s=closest_row_s,
        report_step_s=float(np.diff(joint_s).max()) if joint_s.size > 1 else 0.0,
    )
    if central_body_radius_km is None:
        return event
    # The pursuer waits frozen through an alignment that flies the target.
    pursuer_first = alignment is not None and alignment.early == pursuer_name
    flown = slice(0 if pursuer_first else joint.start, None)
    # Between rows too: on rows 60 s apart a pericenter 1 m underground reads 197 m up.
    ground = evaluate_hermite_samples(
        times[flown], pursuer_position[flown], pursuer_velocity[flown], central_body_radius_km
    )
    # Checked against GMAT's own equatorial radius, not the policy's sphere.
    geodetic = (
        None
        if geodetic_radius_km is None
        else _geodetic_altitude(
            report, pursuer_name, slice(0 if pursuer_first else clock.frozen_rows,
                                        len(report.data)), geodetic_radius_km, path
        )
    )
    return replace(
        event,
        minimum_pursuer_radius_km=ground.minimum_distance_km,
        minimum_geodetic_altitude_km=geodetic,
    )


def _geodetic_altitude(
    report: NumericReport,
    pursuer_name: str,
    rows: slice,
    central_body_radius_km: float,
    path: str | Path,
) -> float | None:
    """The lowest reported Earth.Altitude on the flown rows, checked against the sphere.

    Read on rows only, as the organizer reads them. It is recorded, never a
    gate: the sphere check above is never looser (docs/ASYNC_START_AND_INSERTION.md §5).
    """
    header = f"{pursuer_name}.Earth.Altitude".lower()
    if header not in {name.lower() for name in report.headers}:
        return None
    altitude = report.column(f"{pursuer_name}.Earth.Altitude")[rows]
    position = np.column_stack(
        [report.column(f"{pursuer_name}.EarthMJ2000Eq.{axis}") for axis in "XYZ"]
    )[rows]
    sphere = np.linalg.norm(position, axis=1) - central_body_radius_km
    below = np.flatnonzero(altitude < sphere - GEODETIC_SLACK_KM)
    if below.size:
        row = int(below[0]) + (rows.start or 0)
        raise GmatRunError(
            f"{path} row {row + 1} reports {pursuer_name} at geodetic altitude "
            f"{float(altitude[below[0]])!r} km, below the {float(sphere[below[0]])!r} km it is above a "
            f"{central_body_radius_km:.6f} km sphere; the columns are not what they say"
        )
    return float(altitude.min()) if altitude.size else None
