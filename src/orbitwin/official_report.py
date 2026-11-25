"""What each official ReportFile can show, checked against the rulebook.

The warnings are for `orbitwin inspect`. verify and certify read the official
report's printed rows here (read_official_rows), as check_report_rows reads them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import GmatRunError
from .models import ReportFileSpec, Scenario, StateVector
from .report_format import (
    GREGORIAN_WIDTH,
    is_epoch_parameter,
    is_text_parameter,
    position_frames,
    read_rows,
    report_format,
)
from .rules import InterceptionRules

#: The position resolution must be this fraction of the sphere radius or finer.
RESOLUTION_FRACTION = 0.01

#: Finest the epoch column must resolve: a hundredth of the templates' 1 s
#: report step, since the validator reads the completion time off a row.
TIME_RESOLUTION_S = 0.01

#: What R2026a writes for a Gregorian epoch.
_GREGORIAN_SAMPLE = "09 Nov 2026 02:00:00.000"

#: A GMAT ModJulian has five integer digits from 1968 to 2214.
_MODJULIAN_DAYS = 1.0e4


def _largest_radius_km(state: StateVector, mu_km3_s2: float) -> float:
    """The starting radius, or the apoapsis when the orbit is bound."""
    r, v = state.r, state.v
    radius = float(np.linalg.norm(r))
    energy = 0.5 * float(v @ v) - mu_km3_s2 / radius
    if energy >= 0.0:
        return radius
    sma = -mu_km3_s2 / (2.0 * energy)
    h = float(np.linalg.norm(np.cross(r, v)))
    ecc = math.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / mu_km3_s2**2))
    return max(radius, sma * (1.0 + ecc))


def position_resolution_km(precision: int, magnitude_km: float) -> float:
    """One unit in the last digit of a coordinate of `magnitude_km`.

    GMAT writes Precision significant digits (R2026a: Precision = 3 writes
    4766.09 km as 4.77e+03).
    """
    return last_digit(precision, magnitude_km)


def last_digit(precision: int, magnitude: float) -> float:
    """One unit in the last of `precision` significant digits of `magnitude`."""
    return float(10.0 ** (math.floor(math.log10(magnitude)) - precision + 1))


def epoch_resolution_s(parameter: str, precision: int, horizon_s: float) -> float:
    """How finely a numeric epoch column resolves time over the mission.

    R2026a at Precision = 7 wrote A1ModJulian 31217.70876 as 31217.71.
    """
    kind = parameter.rsplit(".", 1)[-1].lower()
    if kind == "elapsedsecs":
        return last_digit(precision, max(horizon_s, 1.0))
    if kind == "elapseddays":
        return 86400.0 * last_digit(precision, max(horizon_s / 86400.0, 1.0e-6))
    return 86400.0 * last_digit(precision, _MODJULIAN_DAYS)


def _positions(spec: ReportFileSpec, craft: str) -> set[str]:
    """Frames in which `spec` carries all of X, Y and Z of `craft`; as the check reads them."""
    return set(position_frames(spec.parameters, craft))


def official_report_warnings(scenario: Scenario, rules: InterceptionRules) -> list[str]:
    """Why an official ReportFile may not show what `rules` score."""
    warnings: list[str] = []
    radius_km = rules.intercept_radius_km
    pursuer, target = scenario.pursuer.name, scenario.target.name
    magnitude_km = max(
        _largest_radius_km(craft.initial_state, scenario.mu_km3_s2)
        for craft in (scenario.pursuer, scenario.target)
    )
    # script_renderer.official_report_of: ReportFile1 among those written, else the first.
    written = [spec for spec in scenario.report_files if spec.write_report]
    official = next((spec for spec in written if spec.name == "ReportFile1"),
                    written[0] if written else None)
    for spec in scenario.report_files:
        name = spec.name
        if not spec.write_report:
            warnings.append(f"{name}.WriteReport is false: GMAT writes no file for it")
        fmt = report_format(spec)
        reason = fmt.unreadable()
        if reason is not None:
            warnings.append(f"{name}: its {reason}")
        texts = [p for p in spec.parameters if is_text_parameter(p)]
        # Sign, point and e-NN around the digits; a Gregorian epoch is 24.
        widest = max(spec.precision + 6, GREGORIAN_WIDTH if texts else 0)
        if spec.append:
            warnings.append(
                f"{name}.AppendToExistingFile is true: each run adds its rows to the file, and "
                "a report holding two runs is judged on the second"
            )
        if spec.solver_iterations.lower() == "all":
            warnings.append(
                f"{name}.SolverIterations is {spec.solver_iterations}: a solver's passes are "
                "written too, and the report holds more than the flown run"
            )
        if texts:
            warnings.append(
                f"{name} has text columns {texts}; check_report_rows reads them only by their "
                "place in the Add list, with --script or --columns"
            )
            if fmt.separator is not None and fmt.separator in _GREGORIAN_SAMPLE:
                warnings.append(
                    f"{name}: its delimiter {fmt.separator!r} occurs inside a Gregorian epoch "
                    f"({_GREGORIAN_SAMPLE}), so those rows do not split"
                )
        if fmt.fixed_width and not fmt.write_headers and spec.column_width <= widest:
            warnings.append(
                f"{name}: FixedWidth without headers gives each value exactly ColumnWidth = "
                f"{spec.column_width} characters, and at Precision = {spec.precision} a value "
                f"can take {widest}; neighbouring values would run together"
            )
        elif fmt.fixed_width:
            # Measured: with headers a column takes max(ColumnWidth, name) + 3, and
            # GMAT widens a Gregorian one to fit (27 at ColumnWidth 20).
            numeric = spec.precision + 6
            for parameter in spec.parameters:
                stride = max(spec.column_width, len(parameter)) + 3
                if not is_text_parameter(parameter) and stride <= numeric:
                    warnings.append(
                        f"{name}: FixedWidth gives column {parameter} {stride} characters, and "
                        f"at Precision = {spec.precision} a value can take {numeric}; "
                        "neighbouring values would run together"
                    )
                    break
        shared = _positions(spec, pursuer) & _positions(spec, target)
        if not _positions(spec, target) and spec is official:
            warnings.append(
                f"{name}, the official report, carries no {target} position (the organizer's "
                f"workshop style): no distance can be read from it, by us or by hand. certify "
                f"proves d_min and the interception on its diagnostic flight of the same "
                f"script, step 7's --certificate check passes the report with a WARNING saying "
                f"so, and certify ends on PRE-CHECK PASS (target not in report). A report from "
                f"a GMAT GUI cannot be judged (check_report_rows exits 7). How the organizer's "
                f"validator computes d_min from its own data is an open question: ask it "
                f"(docs/RUNBOOK.md step 3)"
            )
        elif not shared:
            warnings.append(
                f"{name} does not carry X, Y and Z of both {pursuer} and {target} in one "
                "frame, so the relative distance cannot be read from it"
            )
        epochs = [p for p in spec.parameters if is_epoch_parameter(p)]
        if not epochs:
            warnings.append(
                f"{name} has no numeric epoch column (ElapsedSecs, ElapsedDays or a ModJulian), "
                "so the completion time cannot be read from it"
            )
        # Every epoch column: which one the validator reads is not known.
        # The earlier craft's ElapsedSecs runs on past T_max by the alignment.
        offset_s = scenario.alignment.offset_s if scenario.alignment is not None else 0.0
        horizon_s = scenario.max_time_s + abs(offset_s)
        for epoch in epochs:
            resolution_s = epoch_resolution_s(epoch, spec.precision, horizon_s)
            if resolution_s > TIME_RESOLUTION_S:
                warnings.append(
                    f"{name}.Precision = {spec.precision} resolves {epoch} to "
                    f"{resolution_s:g} s, coarser than {TIME_RESOLUTION_S:g} s; the "
                    "completion time is read off a row"
                )
        alignment = scenario.alignment
        if alignment is not None and alignment.offset_s > 0.0:
            origin = scenario.time_origin_craft or (
                alignment.early if scenario.time_origin_offset_s else alignment.late
            )
            column = f"{origin}.ElapsedSecs"
            warnings.append(
                f"note: the epoch alignment writes rows before t = 0; check_report_rows needs "
                f"--frozen-rows from certification.json and --time-column {column}"
                + ("" if column in spec.parameters else f", which {name} does not carry")
            )
        if epochs and not any(p.endswith(".ElapsedSecs") for p in spec.parameters):
            warnings.append(
                f"note: {name} has no ElapsedSecs, so check_report_rows prints no score lines "
                "for it; T_team is clock.t_team_s in certification.json"
            )
        resolution_km = position_resolution_km(spec.precision, magnitude_km)
        if resolution_km > RESOLUTION_FRACTION * radius_km:
            warnings.append(
                f"{name}.Precision = {spec.precision} resolves a coordinate near "
                f"{magnitude_km:.0f} km to {resolution_km:g} km, coarser than "
                f"{RESOLUTION_FRACTION:g} of the {radius_km:g} km sphere ({rules.name} rules)"
            )
        if rules.earth_crossing != "not-ruled" and not any(
            p.startswith(f"{pursuer}.") and p.endswith(".Altitude") for p in spec.parameters
        ):
            warnings.append(
                f"note: {name} has no {pursuer} Altitude column. {rules.earth_crossing_clause} "
                f"of the {rules.name} rules disqualifies an Earth crossing, and the organizer "
                "said on 2026-11-01 it judges that on Altitude"
            )
    return warnings


def _shared_frame(names: Sequence[str], pursuer: str, target: str) -> str | None:
    """The frame both craft's X, Y and Z are in, EarthMJ2000Eq first, as the check reads it."""
    shared = set(position_frames(names, pursuer)) & set(position_frames(names, target))
    return min(shared, key=lambda frame: (frame != "EarthMJ2000Eq", frame)) if shared else None


def printed_distance_margin_km(
    spec: ReportFileSpec | None, pursuer: str, target: str, magnitude_km: float
) -> float:
    """How far the distance read off `spec`'s printed rows may sit from GMAT's, 0 when it
    carries no distance: each coordinate is within half a last digit, so each axis of the
    difference within one digit."""
    if spec is None or magnitude_km <= 0.0 or _shared_frame(spec.parameters, pursuer, target) is None:
        return 0.0
    return math.sqrt(3.0) * last_digit(spec.precision, magnitude_km)


@dataclass(frozen=True, slots=True)
class OfficialRows:
    """What the official report's printed rows show, as check_report_rows reads them."""

    precision: int
    #: The day clock's column, the time-origin craft's ElapsedSecs; None without one.
    time_column: str | None
    carries_target_position: bool
    carries_pursuer_position: bool
    #: The last row's time on time_column, as printed.
    last_row_time_s: float | None
    #: On the printed digits, after the frozen rows; None without both craft's positions.
    last_row_distance_km: float | None
    minimum_row_distance_km: float | None
    #: The frozen rows' closest approach, in EarthMJ2000Eq only.
    alignment_minimum_row_distance_km: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_official_rows(
    path: str | Path, spec: ReportFileSpec, pursuer: str, target: str, *,
    frozen_rows: int, time_columns: Sequence[str],
) -> OfficialRows:
    """The official report's rows as printed; `time_columns`, in order, may be the day clock."""
    header, data = read_rows(path, report_format(spec), spec.parameters)
    names = list(header)
    if len(names) != data.shape[1]:
        raise GmatRunError(f"{path}: {len(names)} column names for {data.shape[1]} values a row")
    time_column = next((name for name in time_columns if name in names), None)
    last_s = float(data[-1, names.index(time_column)]) if time_column is not None else None
    frame = _shared_frame(names, pursuer, target)
    distances = None
    alignment_km = None
    if frame is not None:
        own = position_frames(names, pursuer)[frame]
        theirs = position_frames(names, target)[frame]
        distances = np.linalg.norm(data[:, list(own)] - data[:, list(theirs)], axis=1)
        if frozen_rows > 0 and frame == "EarthMJ2000Eq":
            alignment_km = float(distances[:frozen_rows].min())
        distances = distances[frozen_rows:]
    return OfficialRows(
        precision=spec.precision,
        time_column=time_column,
        carries_target_position=bool(position_frames(names, target)),
        carries_pursuer_position=bool(position_frames(names, pursuer)),
        last_row_time_s=last_s,
        last_row_distance_km=None if distances is None else float(distances[-1]),
        minimum_row_distance_km=None if distances is None else float(distances.min()),
        alignment_minimum_row_distance_km=alignment_km,
    )


def same_printed_value(printed: float, value: float, precision: int) -> bool:
    """Whether `printed` is `value` written to `precision` significant digits, as GMAT
    writes it, read back.

    The text read back is not `value` itself: R8's T_team 5911.5072750952095 s prints
    at 16 digits as 5911.507275095209, the next double down, 9.1e-13 s away where half
    the 16th digit is 5e-13 s. Within half a printed digit or 2 ulp of `value` passes.
    """
    if float(f"{value:.{precision}g}") == printed:
        return True
    if value == 0.0:
        return printed == 0.0
    slack = max(0.5 * last_digit(precision, abs(value)), 2.0 * math.ulp(value))
    return abs(printed - value) <= slack * (1.0 + 1.0e-9)
