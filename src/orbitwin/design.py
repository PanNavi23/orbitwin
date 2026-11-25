"""Design variables a rulebook lets a team set in the template: the pursuer's epoch.

Under a rule set whose `pursuer_epoch` is not "fixed" (the onsite ranking rules, §4
入軌時刻) a team chooses when SC_Human starts, not before SC_Alien. `set_pursuer_epoch`
writes a copy of the official script beside it with that one line changed, and a record
of the change beside the copy. Every later command takes the copy as the official
script; `verify_design` holds it to its record: the original unchanged, the copy
differing from it in that one line, the epoch where the rules allow it.
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import PolicyError
from .models import file_sha256
from .rules import InterceptionRules

RECORD_FORMAT = "orbitwin-design/1"
#: GMAT's UTCGregorian text, to the millisecond it prints.
_GREGORIAN = "%d %b %Y %H:%M:%S.%f"
_MS = datetime.timedelta(milliseconds=1)


class DesignError(PolicyError):
    """The template's design variables cannot be set, or a copy is not its record's."""


@dataclass(frozen=True, slots=True)
class _Epoch:
    index: int
    line: str
    text: str
    moment: datetime.datetime


def _lines(data: bytes) -> list[str]:
    """Lines split on "\\n" only, ends kept; any byte that is not UTF-8 survives the trip."""
    parts = data.decode("utf-8", "surrogateescape").split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _bytes(lines: list[str]) -> bytes:
    return "".join(lines).encode("utf-8", "surrogateescape")


def _assigned(lines: list[str], craft: str, field: str) -> list[tuple[int, str, str]]:
    """Every `<craft>.<field> = <value>` outside comments: (line index, line, value)."""
    # `GMAT <craft>.<field> = ...` assigns it too, and GMAT takes a line without `;`.
    pattern = re.compile(
        rf"^\s*(?:GMAT\s+)?{re.escape(craft)}\.{field}\s*=\s*(.*?)\s*;?\s*(%.*)?$")
    found = []
    for index, line in enumerate(lines):
        match = pattern.match(line.rstrip("\r\n"))
        if match:
            found.append((index, line, match.group(1)))
    return found


def _epoch(lines: list[str], craft: str, where: str) -> _Epoch:
    formats = _assigned(lines, craft, "DateFormat")
    epochs = _assigned(lines, craft, "Epoch")
    if len(epochs) != 1 or len(formats) > 1:
        raise DesignError(f"{where}: {craft} must set Epoch once (and DateFormat at most "
                          f"once); it sets them {len(epochs)} and {len(formats)} times")
    if formats and epochs[0][0] < formats[0][0]:
        # GMAT reads the Epoch in the format set when its line runs (gmat_parser).
        raise DesignError(f"{where}: {craft}.Epoch on line {epochs[0][0] + 1} comes before "
                          f"its DateFormat on line {formats[0][0] + 1}")
    form = formats[0][2] if formats else "TAIModJulian"
    if form != "UTCGregorian":
        raise DesignError(f"{where}: {craft}.DateFormat is {form}; only UTCGregorian epochs "
                          "can be set so far")
    index, line, value = epochs[0]
    text = value.strip()
    if len(text) < 2 or text[0] != "'" or text[-1] != "'":
        raise DesignError(f"{where}: {craft}.Epoch is not a quoted UTCGregorian text: {value}")
    try:
        moment = datetime.datetime.strptime(text[1:-1], _GREGORIAN)
    except ValueError:
        raise DesignError(f"{where}: {craft}.Epoch {text} is not UTCGregorian "
                          "('DD Mon YYYY HH:MM:SS.mmm')") from None
    return _Epoch(index, line, text[1:-1], moment)


def _gregorian(moment: datetime.datetime) -> str:
    return moment.strftime("%d %b %Y %H:%M:%S.") + f"{moment.microsecond // 1000:03d}"


def _allowed(rules: InterceptionRules) -> str:
    if rules.pursuer_epoch == "fixed" or rules.pursuer_epoch_clause is None:
        raise DesignError(f"the {rules.id} rules fix the pursuer's epoch; only a rule set "
                          "that lets a team set it (orbitwin rules) can")
    return rules.pursuer_epoch_clause


def _edited_line(epoch: _Epoch, text: str) -> str:
    """``epoch``'s line with only the text between the quotes replaced."""
    quoted = f"'{epoch.text}'"
    head, found, tail = epoch.line.partition(quoted)
    assert found, epoch.line
    return f"{head}'{text}'{tail}"


def set_pursuer_epoch(script: str | Path, rules: InterceptionRules, pursuer: str,
                      target: str, *, after_target_s: float,
                      horizon_s: float | None = None) -> tuple[Path, dict[str, Any]]:
    """Write ``script`` with the pursuer's epoch ``after_target_s`` after the target's,
    beside it, and its record; (the copy, the record)."""
    clause = _allowed(rules)
    original = Path(script).resolve()
    if record_path(original).is_file():
        raise DesignError(f"{original.name} is itself a copy set-epoch wrote; set the epoch "
                          "on the official script it was made from")
    data = original.read_bytes()
    lines = _lines(data)
    target_epoch = _epoch(lines, target, original.name)
    pursuer_epoch = _epoch(lines, pursuer, original.name)
    if not after_target_s >= 0.0:
        raise DesignError(f"the pursuer's epoch may not precede the target's ({clause}); "
                          f"{after_target_s} s is before it")
    if horizon_s is not None and after_target_s >= horizon_s:
        raise DesignError(f"{after_target_s} s after the target's epoch is at or past T_max, "
                          f"{horizon_s} s: the pursuer would have no time")
    try:
        offset = datetime.timedelta(seconds=after_target_s)
        moment = target_epoch.moment + offset
    except OverflowError:
        raise DesignError(f"{after_target_s} s after the target's epoch is no date") from None
    if offset % _MS:
        raise DesignError(f"{after_target_s} s is not a whole number of milliseconds, which "
                          "is what a UTCGregorian epoch prints")
    text = _gregorian(moment)
    lines[pursuer_epoch.index] = _edited_line(pursuer_epoch, text)
    copy = original.with_name(f"{original.stem}.epoch-{moment:%Y%m%dT%H%M%S}"
                              f"{moment.microsecond // 1000:03d}Z{original.suffix}")
    copy.write_bytes(_bytes(lines))
    record: dict[str, Any] = {
        "format": RECORD_FORMAT,
        "original": original.name,
        "original_sha256": file_sha256(original),
        "script_sha256": file_sha256(copy),
        "rules": rules.id,
        "clause": clause,
        "pursuer": pursuer,
        "target": target,
        "line": pursuer_epoch.index + 1,
        "target_epoch": target_epoch.text,
        "pursuer_epoch_was": pursuer_epoch.text,
        "pursuer_epoch": text,
        "after_target_s": after_target_s,
    }
    record_path(copy).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    return copy, record


def seconds_after_target(script: str | Path, target: str, epoch: str) -> float:
    """How long after ``target``'s epoch in ``script`` the UTCGregorian ``epoch`` is."""
    path = Path(script)
    target_epoch = _epoch(_lines(path.read_bytes()), target, path.name)
    try:
        moment = datetime.datetime.strptime(epoch.strip(), _GREGORIAN)
    except ValueError:
        raise DesignError(f"{epoch!r} is not UTCGregorian ('DD Mon YYYY HH:MM:SS.mmm')") from None
    return (moment - target_epoch.moment).total_seconds()


def record_path(script: Path) -> Path:
    return script.with_name(script.name + ".design.json")


def verify_design(script: str | Path, rules: InterceptionRules, pursuer_name: str | None,
                  target_name: str | None) -> dict[str, Any] | None:
    """The record of ``script``'s design edit, checked against the config's craft; None
    when it has none."""
    copy = Path(script).resolve()
    beside = record_path(copy)
    if not beside.is_file():
        return None
    try:
        record = json.loads(beside.read_text(encoding="utf-8"))
        original = copy.with_name(str(record["original"]))
        line = int(record["line"]) - 1
        pursuer, target = str(record["pursuer"]), str(record["target"])
        original_sha256, script_sha256 = (str(record["original_sha256"]),
                                          str(record["script_sha256"]))
        if record.get("format") != RECORD_FORMAT or Path(str(record["original"])).name != (
                record["original"]):
            raise ValueError("not an orbitwin design record")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DesignError(f"{beside} cannot be read: {exc}") from None
    clause = _allowed(rules)
    if (pursuer, target) != (pursuer_name, target_name):
        raise DesignError(f"{copy.name} sets {pursuer}'s epoch against {target}'s; the config's "
                          f"pursuer is {pursuer_name} and its target {target_name}")
    if not original.is_file() or file_sha256(original) != original_sha256:
        raise DesignError(f"{copy.name} was made from {original.name} with SHA-256 "
                          f"{original_sha256}, which is not beside it as it was")
    if record_path(original).is_file():
        raise DesignError(f"{copy.name} was made from {original.name}, itself a copy; set "
                          "the epoch on the official script")
    if file_sha256(copy) != script_sha256:
        raise DesignError(f"{copy.name} is not the copy its record lists (SHA-256 differs)")
    was, now = _lines(original.read_bytes()), _lines(copy.read_bytes())
    changed = [index for index in range(max(len(was), len(now)))
               if index >= len(was) or index >= len(now) or was[index] != now[index]]
    if len(was) != len(now) or changed != [line]:
        raise DesignError(f"{copy.name} differs from {original.name} in lines "
                          f"{[index + 1 for index in changed]}, not only {pursuer}'s epoch")
    target_epoch = _epoch(now, target, copy.name)
    pursuer_epoch = _epoch(now, pursuer, copy.name)
    if pursuer_epoch.index != line or _epoch(was, pursuer, original.name).index != line:
        raise DesignError(f"{copy.name}: line {line + 1} is not {pursuer}'s epoch")
    if _edited_line(_epoch(was, pursuer, original.name), pursuer_epoch.text) != now[line]:
        raise DesignError(f"{copy.name}: line {line + 1} changes more than {pursuer}'s epoch")
    if pursuer_epoch.moment < target_epoch.moment:
        raise DesignError(f"{copy.name}: {pursuer}'s epoch {pursuer_epoch.text} precedes "
                          f"{target}'s {target_epoch.text} ({clause})")
    return {**record, "verified_under": rules.id}
