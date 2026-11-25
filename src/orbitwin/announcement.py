"""What the organizer announces, entered as printed, and held to the day's script.

Two people each enter the announcement into a TOML file of their own: every
value in quotes as it is printed, with its unit, and times in UTC as printed.
``read_pair`` refuses two entries that differ in any value; ``against_script``
holds the agreed one to the fields the script assigns, each to the digits the
announcement printed.
"""

from __future__ import annotations

import datetime
import hashlib
import math
import re
import tomllib
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .gmat_parser import GmatScriptDocument, spacecraft_field_lines, spacecraft_fields
from .rules import InterceptionRules

FORMAT = "orbitwin-announcement/1"
#: The state fields an entry may give, by GMAT's names, and the unit each is printed in.
STATE_UNITS = {
    "SMA": "km", "ECC": "", "INC": "deg", "RAAN": "deg", "AOP": "deg", "TA": "deg",
    "RMAG": "km", "RA": "deg", "DEC": "deg", "VMAG": "km/s", "AZI": "deg", "FPA": "deg",
    "X": "km", "Y": "km", "Z": "km", "VX": "km/s", "VY": "km/s", "VZ": "km/s",
}
#: Fields given as a name, compared as written.
LABELS = ("CoordinateSystem", "DisplayStateType")
#: What GMAT assumes of a Spacecraft field the script does not set.
GMAT_DEFAULTS = {"CoordinateSystem": "EarthMJ2000Eq", "DisplayStateType": "Cartesian",
                 "DateFormat": "TAIModJulian", "AnomalyType": "TA"}
#: Fields GMAT also reads into a craft's state; a script that sets one is not compared.
STATE_ALIASES = frozenset({
    *(f"Element{n}" for n in range(1, 7)),
    *(f"Cartesian{axis}" for axis in ("X", "Y", "Z", "VX", "VY", "VZ")),
    "CartesianState", "KeplerianState", "ModifiedKeplerianState", "SphericalAZFPAState",
    "SphericalRADECState", "EquinoctialState",
})
#: The score's coefficients in the units the 2026/11/04 revision fixes.
SCORE_UNITS = {"C_r": "km", "kt": "1/s", "Ct": "s", "kv": "1/(m/s)", "Cv": "m/s"}
#: A script's value is a double: printed to 16 or 17 digits it can differ from the
#: announcement's in the last one, so no value is held closer than this share of itself.
_RELATIVE_FLOOR = Decimal("1e-12")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
# Thousands may be grouped with commas, as a slide prints them.
_NUMBER = re.compile(r"[+-]?(?:[1-9]\d{0,2}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d{1,3})?")
_INSTANT = re.compile(
    r"(\d{2}) ([A-Z][a-z]{2}) (\d{4}) (\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))? UTC")


class AnnouncementError(ConfigurationError):
    """An entry that cannot be read, two that differ, or a script that contradicts them."""


@dataclass(frozen=True)
class Quantity:
    text: str
    value: Decimal
    unit: str
    #: Half a unit in the last printed digit: any value within it prints as ``text``.
    half_step: Decimal

    def holds(self, other: Decimal) -> bool:
        apart = abs(other - self.value)
        # Past 1e15 a double no longer tells one turn from the next.
        if self.unit == "deg" and apart.adjusted() < 15:
            # One direction printed in another range of 360 degrees.
            apart %= 360
            apart = min(apart, 360 - apart)
        return apart <= max(self.half_step, abs(self.value) * _RELATIVE_FLOOR)


@dataclass(frozen=True)
class Instant:
    text: str
    #: UTC, naive. Leap seconds are not counted.
    moment: datetime.datetime
    half_step_s: Decimal

    def seconds_after(self, other: Instant) -> Decimal:
        delta = self.moment - other.moment
        return (Decimal(delta.days) * 86400 + delta.seconds
                + Decimal(delta.microseconds) / 1_000_000)


@dataclass(frozen=True)
class Label:
    text: str


@dataclass(frozen=True)
class Entry:
    path: Path
    sha256: str
    entered_by: str
    epochs: dict[str, Instant]
    states: dict[str, dict[str, Quantity]]
    labels: dict[str, dict[str, Label]]
    #: None in an entry that gives only the score, or T_max only in seconds.
    t_max: Instant | None
    t_max_s: Quantity | None
    #: The mission boundary, a geocentric radius, when one is announced.
    boundary: Quantity | None
    score: dict[str, Quantity] | None


def _quantity(text: object, unit: str, where: str, positive: bool = False) -> Quantity:
    if not isinstance(text, str):
        raise AnnouncementError(f"{where} is {text!r}; write it in quotes, as printed")
    # A slide's minus sign is often U+2212, and its degrees a sign.
    plain = text.strip().replace("\N{MINUS SIGN}", "-")
    if plain.endswith("\N{DEGREE SIGN}"):
        plain = plain[:-1].rstrip() + " deg"
    parts = plain.split(None, 1)
    number = parts[0] if parts else ""
    given = parts[1].strip() if len(parts) == 2 else ""
    if not _NUMBER.fullmatch(number):
        raise AnnouncementError(f"{where} = {text!r} does not start with a number")
    if given != unit:
        raise AnnouncementError(f"{where} = {text!r} is in {given or 'no unit'}; it is "
                                f"announced in {unit or 'no unit'}")
    value = Decimal(number.replace(",", ""))
    if not math.isfinite(float(value)):
        raise AnnouncementError(f"{where} = {text!r} is out of range")
    if positive and value <= 0:
        raise AnnouncementError(f"{where} = {text!r} is not positive")
    exponent = value.as_tuple().exponent
    assert isinstance(exponent, int)
    return Quantity(text, value, unit, Decimal(5).scaleb(exponent - 1))


def _instant(text: object, where: str) -> Instant:
    match = _INSTANT.fullmatch(text.strip()) if isinstance(text, str) else None
    if match is None or match.group(2) not in _MONTHS or not isinstance(text, str):
        raise AnnouncementError(f"{where} = {text!r} is not 'DD Mon YYYY HH:MM:SS[.mmm] UTC'")
    day, month, year, hour, minute, second, millis = match.groups()
    try:
        moment = datetime.datetime(int(year), _MONTHS.index(month) + 1, int(day), int(hour),
                                   int(minute), int(second),
                                   int((millis or "0").ljust(3, "0")) * 1000)
    except ValueError:
        raise AnnouncementError(f"{where} = {text!r} is no such date and time") from None
    return Instant(text, moment, Decimal(5).scaleb(-len(millis or "") - 1))


def _label(text: object, where: str) -> Label:
    if not isinstance(text, str) or not re.fullmatch(r"[A-Za-z_]\w*", text.strip()):
        raise AnnouncementError(f"{where} = {text!r} is not a GMAT name")
    return Label(text.strip())


def _table(raw: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise AnnouncementError(f"{where}: [{key}] is not a table")
    return value


def _only(table: dict[str, Any], allowed: set[str] | dict[str, str], where: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise AnnouncementError(f"{where}: {', '.join(unknown)} is not a field an announcement "
                                f"gives here; known: {', '.join(sorted(allowed))}")


def read_entry(path: Path) -> Entry:
    """One person's entry, every value checked for its form and unit."""
    where = str(path)
    if not path.name.isprintable():
        raise AnnouncementError(f"{where!r}: name the entry with printable characters")
    try:
        if not path.is_file():
            raise AnnouncementError(f"{where} is not a file")
        data = path.read_bytes()
        raw = tomllib.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise AnnouncementError(f"{where}: {type(exc).__name__}: {exc}") from None
    _only(raw, {"format", "entered_by", "craft", "mission", "score"}, where)
    if raw.get("format") != FORMAT:
        raise AnnouncementError(f"{where}: format is {raw.get('format')!r}, not {FORMAT!r}")
    entered_by = raw.get("entered_by")
    if not isinstance(entered_by, str) or not entered_by.strip():
        raise AnnouncementError(f"{where}: entered_by names no one")
    if not entered_by.isprintable():
        raise AnnouncementError(f"{where}: entered_by {entered_by!r} is not one line of "
                                "printable text")
    epochs: dict[str, Instant] = {}
    states: dict[str, dict[str, Quantity]] = {}
    labels: dict[str, dict[str, Label]] = {}
    for craft, fields in _table(raw, "craft", where).items():
        if not isinstance(fields, dict):
            raise AnnouncementError(f"{where}: [craft.{craft}] is not a table")
        _only(fields, {"Epoch", *LABELS, *STATE_UNITS}, f"{where} [craft.{craft}]")
        if "Epoch" in fields:
            epochs[craft] = _instant(fields["Epoch"], f"{craft}.Epoch")
        labels[craft] = {name: _label(fields[name], f"{craft}.{name}")
                         for name in LABELS if name in fields}
        states[craft] = {name: _quantity(text, STATE_UNITS[name], f"{craft}.{name}")
                         for name, text in fields.items() if name in STATE_UNITS}
    mission = _table(raw, "mission", where)
    _only(mission, {"T_max", "T_max_s", "boundary"}, f"{where} [mission]")
    t_max, t_max_s, boundary = (mission.get(k) for k in ("T_max", "T_max_s", "boundary"))
    if mission and t_max is None and t_max_s is None:
        raise AnnouncementError(f"{where}: [mission] gives no T_max")
    score = None
    if "score" in raw:
        table = _table(raw, "score", where)
        _only(table, SCORE_UNITS, f"{where} [score]")
        missing = sorted(set(SCORE_UNITS) - set(table))
        if missing:
            raise AnnouncementError(f"{where}: [score] lacks {', '.join(missing)}")
        score = {name: _quantity(table[name], unit, f"score.{name}", positive=name == "C_r")
                 for name, unit in SCORE_UNITS.items()}
    if not mission and score is None:
        raise AnnouncementError(f"{where} gives neither [mission] nor [score]")
    return Entry(
        path=path, sha256=hashlib.sha256(data).hexdigest(), entered_by=entered_by.strip(),
        epochs=epochs, states=states, labels=labels,
        t_max=None if t_max is None else _instant(t_max, "T_max"),
        t_max_s=None if t_max_s is None else _quantity(t_max_s, "s", "T_max_s", positive=True),
        boundary=None if boundary is None else _quantity(boundary, "km", "boundary",
                                                          positive=True),
        score=score)


def _flat(entry: Entry) -> dict[str, Quantity | Instant | Label]:
    flat: dict[str, Quantity | Instant | Label] = {
        f"{c}.Epoch": i for c, i in entry.epochs.items()}
    flat |= {f"{c}.{n}": q for c, fields in entry.states.items() for n, q in fields.items()}
    flat |= {f"{c}.{n}": label for c, fields in entry.labels.items()
             for n, label in fields.items()}
    for key, value in (("T_max", entry.t_max), ("T_max_s", entry.t_max_s),
                       ("boundary", entry.boundary)):
        if value is not None:
            flat[key] = value
    flat |= {f"score.{n}": q for n, q in (entry.score or {}).items()}
    return flat


def _same(a: Quantity | Instant | Label | None, b: Quantity | Instant | Label | None) -> bool:
    if isinstance(a, Instant) and isinstance(b, Instant):
        return a.moment == b.moment
    if isinstance(a, Quantity) and isinstance(b, Quantity):
        return (a.value, a.unit) == (b.value, b.unit)
    return isinstance(a, Label) and isinstance(b, Label) and a.text == b.text


def differences(first: Entry, second: Entry) -> list[str]:
    """Each value the two entries do not agree on, by value and unit, not by spelling."""
    one, two = _flat(first), _flat(second)
    lines = []
    for key in sorted(set(one) | set(two)):
        a, b = one.get(key), two.get(key)
        if not _same(a, b):
            lines.append(f"{key}: {a.text if a else 'not given'!r} by {first.entered_by}, "
                         f"{b.text if b else 'not given'!r} by {second.entered_by}")
    return lines


def _coarser(a: Any, b: Any) -> Any:
    if a is None:
        return None
    if isinstance(a, Quantity):
        return replace(a, half_step=max(a.half_step, b.half_step))
    return replace(a, half_step_s=max(a.half_step_s, b.half_step_s))


def read_pair(first_path: Path, second_path: Path) -> tuple[Entry, Entry, Entry]:
    """Both entries and the one they agree on, each value to the coarser of the two
    printings; refuses entries by one person or that differ in any value."""
    first, second = read_entry(first_path), read_entry(second_path)
    if first.entered_by.casefold() == second.entered_by.casefold():
        raise AnnouncementError(f"{first_path} and {second_path} are both entered by "
                                f"{first.entered_by!r}; two people enter one each")
    differ = differences(first, second)
    if differ:
        raise AnnouncementError("the two entries differ, so at least one was typed wrong; "
                                "read the announcement again: " + "; ".join(differ))
    agreed = replace(
        first,
        epochs={c: _coarser(i, second.epochs[c]) for c, i in first.epochs.items()},
        states={c: {n: _coarser(q, second.states[c][n]) for n, q in fields.items()}
                for c, fields in first.states.items()},
        t_max=_coarser(first.t_max, second.t_max),
        t_max_s=_coarser(first.t_max_s, second.t_max_s),
        boundary=_coarser(first.boundary, second.boundary),
        score=None if first.score is None else {
            n: _coarser(q, second.score[n]) for n, q in first.score.items()}  # type: ignore[index]
    )
    return first, second, agreed


@dataclass(frozen=True)
class ScriptView:
    """Each Spacecraft's fields as the script assigns them, and the line of each."""

    fields: dict[str, dict[str, str]]
    lines: dict[str, dict[str, int]]


def read_script(script: Path) -> ScriptView:
    document = GmatScriptDocument.read(script)
    return ScriptView(spacecraft_fields(document), spacecraft_field_lines(document))


def _unquoted(text: str) -> str:
    text = text.strip()
    return text[1:-1] if len(text) >= 2 and text[0] == text[-1] == "'" else text


def _written(view: ScriptView, name: str, field: str) -> str:
    """A field's value as GMAT reads it: the script's, or GMAT's default."""
    fields = view.fields.get(name, {})
    return _unquoted(fields[field]) if field in fields else GMAT_DEFAULTS[field]


def _written_epoch(view: ScriptView, name: str) -> tuple[datetime.datetime | None, str]:
    """The craft's epoch as the script writes it in UTCGregorian, or why it is not read."""
    fields, lines = view.fields.get(name, {}), view.lines.get(name, {})
    overrides = sorted(field for field in fields if field.startswith("Epoch."))
    if overrides:
        return None, f"the script also sets {', '.join(overrides)}"
    if "Epoch" not in fields:
        return None, "the script does not set it"
    written = _unquoted(fields["Epoch"])
    if "DateFormat" in lines and lines["Epoch"] < lines["DateFormat"]:
        # GMAT reads an Epoch in the format set when its line runs.
        return None, f"the script sets {written!r} before its DateFormat"
    if _written(view, name, "DateFormat") != "UTCGregorian":
        return None, f"the script writes {written!r}, not in UTCGregorian"
    try:
        return _instant(written + " UTC", name).moment, ""
    except AnnouncementError:
        return None, f"the script writes {written!r}"


def written_epochs(view: ScriptView) -> dict[str, datetime.datetime]:
    return {name: moment for name in view.fields
            if (moment := _written_epoch(view, name)[0]) is not None}


def horizon_s(entry: Entry, rules: InterceptionRules, pursuer: str, target: str,
              written: dict[str, datetime.datetime] | None = None) -> float:
    """T_max in seconds from the time origin: the earlier epoch, or the target's when the
    rules let a team set the pursuer's. The origin is the script's own epoch when
    ``written`` has it (against_script has held it to the announced one), and the seconds
    are T_max_s when it is printed finer than T_max."""
    if entry.t_max is None:
        if entry.t_max_s is None:
            raise AnnouncementError(f"{entry.path}: the entry gives no T_max")
        return float(entry.t_max_s.value)
    needed = [target] if rules.pursuer_epoch != "fixed" else [pursuer, target]
    missing = [name for name in needed if name not in entry.epochs]
    if missing:
        raise AnnouncementError(
            f"{entry.path}: T_max counts from the earlier epoch, and the entry gives none for "
            f"{', '.join(missing)}")
    # The script's own epoch where it writes one, else the announced; of two equal, the
    # finer printing.
    def start(name: str) -> tuple[datetime.datetime, Decimal]:
        if written and name in written:
            return written[name], Decimal(0)
        return entry.epochs[name].moment, entry.epochs[name].half_step_s

    first = min(needed, key=start)
    origin = entry.epochs[first]
    moment, origin_slack = start(first)
    origin_text = (origin.text if not written or first not in written
                   else f"{first}'s epoch in the script, {moment:%d %b %Y %H:%M:%S.%f} UTC")
    delta = entry.t_max.moment - moment
    seconds = (Decimal(delta.days) * 86400 + delta.seconds
               + Decimal(delta.microseconds) / 1_000_000)
    if seconds <= 0:
        raise AnnouncementError(f"T_max {entry.t_max.text} is not after {origin_text}")
    if entry.t_max_s is None:
        return float(seconds)
    # Each of the three is rounded where it was printed. The organizer may have counted
    # from the announced epoch, which is the script's to its printed digits.
    slack = entry.t_max_s.half_step + entry.t_max.half_step_s + origin.half_step_s
    if abs(entry.t_max_s.value - seconds) > slack:
        raise AnnouncementError(
            f"T_max_s {entry.t_max_s.text} is not T_max {entry.t_max.text} less "
            f"{origin_text}, which is {seconds} s; read both again")
    # The seconds counted here carry T_max's rounding and, from an announced epoch, its own.
    finer = entry.t_max_s.half_step < entry.t_max.half_step_s + origin_slack
    return float(entry.t_max_s.value if finer else seconds)


def _number(text: str) -> Decimal | None:
    try:
        value = Decimal(text.strip())
    except InvalidOperation:
        return None
    # A double holds no more; past it the comparison itself overflows.
    return value if value.is_finite() and abs(value.adjusted()) <= 308 else None


def against_script(entry: Entry, view: ScriptView, pursuer: str,
                   rules: InterceptionRules) -> tuple[list[str], list[str], list[str]]:
    """What the script contradicts, what it does not write the way it was announced, and
    what agrees, each named ``craft.field``."""
    wrong: list[str] = []
    unchecked: list[str] = []
    agree: list[str] = []
    named = set(entry.epochs) | {c for c, f in entry.states.items() if f} | {
        c for c, f in entry.labels.items() if f}
    for name in sorted(named):
        fields, lines = view.fields.get(name), view.lines.get(name, {})
        if fields is None:
            wrong.append(f"{name}: the script has no such craft (it flies "
                         f"{' and '.join(sorted(view.fields))})")
            continue
        if name in entry.epochs:
            announced = entry.epochs[name]
            moment, why = _written_epoch(view, name)
            slack = datetime.timedelta(seconds=float(announced.half_step_s))
            free = name == pursuer and rules.pursuer_epoch != "fixed"
            if moment is None:
                unchecked.append(f"{name}.Epoch ({why})")
            elif (moment < announced.moment - slack) if free else (
                    abs(moment - announced.moment) > slack):
                wrong.append(f"{name}.Epoch: announced {announced.text}"
                             f"{' or later' if free else ''}, the script "
                             f"{_unquoted(fields['Epoch'])}")
            else:
                agree.append(f"{name}.Epoch")
        given = entry.labels.get(name, {})
        state = entry.states.get(name, {})
        # Why the numbers cannot be read as the announcement's, if they cannot.
        blocked = sorted(STATE_ALIASES & set(fields))
        reason = f"the script also sets {', '.join(blocked)}" if blocked else ""
        if not reason and _written(view, name, "AnomalyType") != "TA":
            reason = f"the script's AnomalyType is {_written(view, name, 'AnomalyType')}"
        for field in LABELS:
            reads = _written(view, name, field)
            if field in given and given[field].text != reads:
                unchecked.append(f"{name}.{field} (announced {given[field].text}, the script "
                                 f"reads {reads})")
                if field == "CoordinateSystem" and not reason:
                    reason = f"the script's state is in {reads}"
            elif field in given:
                agree.append(f"{name}.{field}")
            elif state:
                unchecked.append(f"{name}.{field} (not announced; the script reads {reads})")
        if state and not reason and "CoordinateSystem" in lines and any(
                lines.get(field, -1) < lines["CoordinateSystem"] for field in state
                if field in fields):
            reason = "the script sets its state before its CoordinateSystem"
        for field, quantity in state.items():
            text = fields.get(field)
            value = None if text is None else _number(text)
            if reason:
                unchecked.append(f"{name}.{field} ({reason})")
            elif value is None:
                how = "does not set it" if text is None else f"writes {text!r}"
                unchecked.append(f"{name}.{field} (the script {how})")
            elif not quantity.holds(value):
                wrong.append(f"{name}.{field}: announced {quantity.text}, the script {text}")
            else:
                agree.append(f"{name}.{field}")
    return wrong, unchecked, agree
