"""Split a GMAT ReportFile into rows the way its format fields wrote them.

The one reader of report text: parse_numeric_report and
scripts/check_report_rows.py both use read_rows. Imports only numpy, so the
pre-upload check reads a report without the script parser or the config loader.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import GmatRunError
from .models import ReportFileSpec

#: Characters a number or a GMAT parameter name can hold, and '%', which
#: read_rows skips at the start of a line. Split on one of these, a row does
#: not come apart into its values.
_UNSPLITTABLE = frozenset("0123456789.+-_%")

#: Two numbers written end to end: `...000-18013.4`, `4766.09-4264.3`.
_FUSED_RE = re.compile(r"\d[+-]\d|\.\d*\.")


@dataclass(frozen=True, slots=True)
class ReportFormat:
    """The ReportFile fields that decide how a row splits into values.

    Measured on R2026a (tests/fixtures/gmat_report_formats):
    - FixedWidth pads each value with the delimiter repeated: `1.5,,,,2.5`.
    - FixedWidth without headers pads with spaces whatever the delimiter is,
      and the stride is ColumnWidth exactly, so a narrow one fuses values.
    - Without FixedWidth one delimiter separates values, headers or not.
    - LeftJustify and ZeroFill move padding and add zeros; they split the same.
    """

    write_headers: bool = True
    #: The character GMAT writes; see gmat_parser._report_delimiter.
    delimiter: str = " "
    fixed_width: bool = True
    column_width: int = 23

    @property
    def separator(self) -> str | None:
        """What separates values in the file; None is a run of whitespace."""
        if (self.fixed_width and not self.write_headers) or self.delimiter.isspace():
            return None
        return self.delimiter

    def unreadable(self) -> str | None:
        """Why rows in this format cannot be split, or None."""
        if len(self.delimiter) != 1:
            return f"delimiter {self.delimiter!r} is not one character"
        separator = self.separator
        if separator is not None and (separator.isalpha() or separator in _UNSPLITTABLE):
            return (
                f"delimiter {separator!r} can occur inside a number or a column name, "
                "so a row does not split into its values"
            )
        return None

    def split(self, line: str) -> list[str]:
        """The values in one line of the report."""
        reason = self.unreadable()
        if reason is not None:
            raise GmatRunError(f"Cannot read a ReportFile whose {reason}")
        separator = self.separator
        if separator is None:
            return line.split()
        parts = [part.strip() for part in line.strip().split(separator)]
        if self.fixed_width:
            # The empty parts are padding.
            return [part for part in parts if part]
        return parts

    def describe(self) -> str:
        return (
            f"delimiter {self.delimiter!r}, WriteHeaders {self.write_headers}, "
            f"FixedWidth {self.fixed_width}"
        )


DEFAULT_REPORT_FORMAT = ReportFormat()


def report_format(spec: ReportFileSpec) -> ReportFormat:
    return ReportFormat(
        write_headers=spec.write_headers,
        delimiter=spec.delimiter,
        fixed_width=spec.fixed_width,
        column_width=spec.column_width,
    )


def fused_columns_hint(fmt: ReportFormat, parts: Sequence[str]) -> str:
    """Why a line does not read, when its values ran together under FixedWidth."""
    if not fmt.fixed_width or not any(_FUSED_RE.search(part) for part in parts):
        return ""
    width = "max(ColumnWidth, its name) + 3" if fmt.write_headers else "ColumnWidth"
    return (
        f"; values ran together: under FixedWidth each column is {width} characters "
        f"(ColumnWidth = {fmt.column_width}), and a longer value runs into the next"
    )


#: GMAT writes these as text: `09 Nov 2026 02:00:00.000` (R2026a).
_GREGORIAN_RE = re.compile(r"^\d{2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}:\d{2}\.\d{3}$")
#: Characters in one Gregorian value.
GREGORIAN_WIDTH = 24


#: Numeric epochs GMAT reports for a spacecraft.
_EPOCH_RE = re.compile(r"^(?:ElapsedSecs|ElapsedDays|\w+ModJulian)$", re.IGNORECASE)


def is_text_parameter(name: str) -> bool:
    return name.rsplit(".", 1)[-1].endswith("Gregorian")


def is_epoch_parameter(name: str) -> bool:
    return bool(_EPOCH_RE.match(name.rsplit(".", 1)[-1]))


def position_frames(names: Sequence[str], craft: str) -> dict[str, tuple[int, int, int]]:
    """Each frame in which `names` holds X, Y and Z of `craft`, with their indices.

    `Craft.X` is EarthMJ2000Eq: R2026a writes it identically.
    """
    found: dict[str, dict[str, int]] = {}
    for index, name in enumerate(names):
        parts = name.split(".")
        if parts[0] != craft or parts[-1] not in {"X", "Y", "Z"} or len(parts) not in (2, 3):
            continue
        frame = parts[1] if len(parts) == 3 else "EarthMJ2000Eq"
        found.setdefault(frame, {}).setdefault(parts[-1], index)
    return {
        frame: (axes["X"], axes["Y"], axes["Z"])
        for frame, axes in found.items()
        if len(axes) == 3
    }


def _row(
    tokens: list[str], text: Sequence[bool], separator: str | None
) -> list[float] | None:
    """Values of a row whose columns may be text; None if it does not fit them.

    A text column is NaN. With whitespace as separator a Gregorian value is
    four tokens.
    """
    per_column = [4 if is_text and separator is None else 1 for is_text in text]
    if len(tokens) != sum(per_column):
        return None
    values: list[float] = []
    start = 0
    for is_text, count in zip(text, per_column, strict=True):
        value = " ".join(tokens[start:start + count])
        start += count
        if is_text:
            if not _GREGORIAN_RE.match(value):
                return None
            values.append(float("nan"))
        else:
            try:
                values.append(float(value))
            except ValueError:
                return None
    return values


def read_rows(
    path: str | Path,
    fmt: ReportFormat = DEFAULT_REPORT_FORMAT,
    column_names: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    """The column names and numeric rows of a report written in `fmt`.

    With headers, the first line that is not all numbers names the columns;
    without, `column_names` (the ReportFile's Add list) does, or nothing. A
    line that does not read as numbers after that is refused: skipping it
    would move the last row. Names are not checked against the row width.

    Text columns (Gregorian epochs) are read as NaN, and only when
    `column_names` says where they are.
    """
    names = tuple(column_names or ())
    text = [is_text_parameter(name) for name in names]
    header: tuple[str, ...] | None = None if fmt.write_headers else names
    rows: list[list[float]] = []
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    for line_number, line in enumerate(content.splitlines(), start=1):
        if line.strip().startswith("%"):
            continue
        parts = fmt.split(line)
        if not any(parts):
            continue
        if any(text):
            # A header line never fits the Add list's text columns.
            row = _row(parts, text, fmt.separator)
        else:
            try:
                row = [float(value) for value in parts]
            except ValueError:
                row = None
        if row is not None:
            rows.append(row)
            continue
        if rows or header is not None:
            raise GmatRunError(
                f"{path}: line {line_number} is not numeric in this ReportFile format "
                f"({fmt.describe()}){fused_columns_hint(fmt, parts)}"
            )
        header = tuple(parts)
    if not rows:
        raise GmatRunError(f"{path}: no numeric rows found")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise GmatRunError(f"{path}: rows have differing column counts {sorted(widths)}")
    # A file with no header row still has the names it was given.
    return header or names, np.asarray(rows, dtype=float)
