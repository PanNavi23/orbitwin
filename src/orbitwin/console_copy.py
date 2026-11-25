"""The console copy: a script less the plots GmatConsole cannot run, proven line by line.

GmatConsole refuses a script creating an OpenFrames object (R2026a: the
workshop template, 58 errors, exit 1), and a GroundTrack created before a
ReportFile left it writing no report. Under gmat.console_copy certify and
baseline then run this copy instead, and the submission keeps every plot line.

The copy is the submitted bytes with whole lines deleted: each Create of an
OpenFrames or GroundTrack object, each assignment to one, and each mission
command (Toggle, PenUp, PenDown, ClearPlot, MarkPoint) naming only such
objects. prove_console_copy reads both files' bytes and refuses anything else:
a kept line that is not byte-identical, a deleted line it cannot read as one of
those, or a kept line naming a deleted object.

Whether the GMAT GUI takes the deleted lines is not judged here: only the GUI
reads them, and they are the official template's own, byte for byte, in order
(assert_plot_lines_unchanged), so the GUI runs the submitted script exactly
when it runs the organizer's.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import PolicyError
from .gmat_parser import (
    _BEGIN_MISSION_RE,
    _CREATE_RE,
    GmatScriptDocument,
    _code_part,
    _resource_assignments,
    created_types,
    report_file_specs,
    split_create_names,
)
from .gmat_runner import PLOT_TYPES
from .openframes import OPENFRAMES_TYPES

#: On the workshop template a GroundTrack left GmatConsole writing no report.
GROUND_TRACK_TYPES = frozenset({"GroundTrack", "GroundTrackPlot"})
#: What the copy leaves out. Other plots (OrbitView, XYPlot) stay: GmatConsole runs them.
COPIED_TYPES = OPENFRAMES_TYPES | GROUND_TRACK_TYPES
#: Mission commands that only drive subscribers.
PLOT_COMMANDS = frozenset({"Toggle", "PenUp", "PenDown", "ClearPlot", "MarkPoint"})

_NAME = r"[A-Za-z_]\w*"
#: R2026a's grammar: `Create` and the type are case-sensitive, `GMAT Create`
#: runs, names are separated by a comma or blanks, and `;;` runs (the GUI too).
_STRICT_CREATE_RE = re.compile(
    rf"[ \t]*(?:GMAT[ \t]+)?Create[ \t]+(?P<type>{_NAME})[ \t]+"
    rf"(?P<names>{_NAME}(?:(?:[ \t]*,[ \t]*|[ \t]+){_NAME})*)[ \t]*(?:;[ \t]*)+"
)
#: One assignment of one field. A single-quoted value may hold ; or %.
_STRICT_ASSIGN_RE = re.compile(
    rf"[ \t]*(?:GMAT[ \t]+)?(?P<object>{_NAME})[ \t]*\.(?P<field>{_NAME})"
    r"[ \t]*=(?P<value>(?:[^;'\n]|'[^'\n]*')*);(?:[ \t]*;)*[ \t]*"
)
#: Names apart by blanks, as the GUI takes a Toggle (it refuses a comma).
_STRICT_COMMAND_RE = re.compile(
    rf"[ \t]*(?P<command>{'|'.join(sorted(PLOT_COMMANDS))})[ \t]+"
    rf"(?P<names>{_NAME}(?:[ \t]+{_NAME})*)[ \t]*(?:;[ \t]*)+"
)
_LINE_RE = re.compile(rb"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")


@dataclass(frozen=True, slots=True)
class ConsoleCopy:
    """A copy prove_console_copy proved, from the bytes it read."""

    script_bytes: bytes
    submitted_sha256: str
    copy_sha256: str
    #: "Type Name" of each object left out, in the order the script creates them.
    removed: tuple[str, ...]
    #: 1-based lines of the submitted script left out.
    deleted_lines: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "submitted_sha256": self.submitted_sha256,
            "copy_sha256": self.copy_sha256,
            "removed": list(self.removed),
            "deleted_lines": list(self.deleted_lines),
        }


def _physical_lines(raw: bytes) -> list[bytes]:
    """Lines as GmatConsole reads them, endings kept: split at CR, LF and CRLF only."""
    return _LINE_RE.findall(raw)


def build_console_copy(document: GmatScriptDocument) -> bytes | None:
    """The script less its OpenFrames and GroundTrack lines, or None if it creates none.

    Not checked here: prove_console_copy refuses anything else in it.
    """
    types = created_types(document)
    plots = {name for name, kind in types.items() if kind in COPIED_TYPES}
    if not plots:
        return None
    subject = re.compile(r"\s*(?:GMAT\s+)?(?:Create\s+(?P<type>\w+)\s|(?P<object>\w+)\s*\.)")
    kept: list[str] = []
    mission = False
    for line in document.lines:
        code = _code_part(line.rstrip("\r\n"))
        if not mission and _BEGIN_MISSION_RE.match(code):
            mission = True
        elif not mission:
            match = subject.match(code)
            if match and (match["type"] in COPIED_TYPES or match["object"] in plots):
                continue
        else:
            words = re.findall(r"\w+", code)
            if words and words[0] in PLOT_COMMANDS and set(words[1:]) & plots:
                continue
        kept.append(line)
    return "".join(kept).encode("ascii")


def _naming(names: set[str]) -> re.Pattern[str]:
    """A standalone use of one of ``names``: not a field such as ``Craft.SMA``."""
    return re.compile(r"(?<![\w.])(?:" + "|".join(map(re.escape, sorted(names))) + r")(?!\w)")


def _all_creates(document: GmatScriptDocument) -> list[tuple[int, str, list[str]]]:
    """Every live Create before BeginMissionSequence: (index, type, names)."""
    found = []
    for index, line in enumerate(document.lines):
        code = _code_part(line.rstrip("\r\n"))
        if _BEGIN_MISSION_RE.match(code):
            break
        match = _CREATE_RE.match(code)
        if match:
            found.append((index, match["type"], split_create_names(match["names"])))
    return found


def prove_console_copy(submitted: Path, copy: Path) -> ConsoleCopy:
    """Prove ``copy`` is ``submitted`` less only its plot lines, both read from disk."""
    document, copied = GmatScriptDocument.read(submitted), GmatScriptDocument.read(copy)
    proven = proven_copy(document, copied.raw_bytes, copy.name)
    _assert_nothing_else_differs(document, copied, proven.removed)
    return proven


def proven_copy(document: GmatScriptDocument, copied: bytes, copy_name: str) -> ConsoleCopy:
    """Prove ``copied`` is ``document`` less only its OpenFrames and GroundTrack lines.

    Walking the submitted lines in order, each is
    either the next line of the copy, byte for byte with its line ending, or
    left out; a line is left out exactly when it is a Create of those types, an
    assignment to one of their objects, or a plot command naming only them,
    each read strictly. Every such object is created once, and no line the
    copy keeps names one. Anything else is refused.
    """
    submitted = document.path
    before, after = _physical_lines(document.raw_bytes), _physical_lines(copied)
    if len(before) != len(document.lines):  # pragma: no cover - one line model
        raise PolicyError(f"{submitted.name} does not split into lines one way")
    creates = _all_creates(document)
    created: dict[str, str] = {}
    for index, kind, names in creates:
        for name in names:
            if name in created and (kind in COPIED_TYPES or created[name] in COPIED_TYPES):
                raise PolicyError(
                    f"line {index + 1} of {submitted.name} creates {name} again, which the "
                    "console copy would leave out; it refuses a script that does"
                )
            created.setdefault(name, kind)
    plots = {name: kind for name, kind in created.items() if kind in COPIED_TYPES}
    if not plots:
        raise PolicyError(f"{submitted.name} creates no OpenFrames or GroundTrack object")
    named = _naming(set(plots))
    begin = next(
        (i for i, line in enumerate(document.lines)
         if _BEGIN_MISSION_RE.match(_code_part(line.rstrip("\r\n")))),
        None,
    )
    if begin is None:
        raise PolicyError(f"{submitted.name} has no BeginMissionSequence")
    deleted: list[int] = []
    removed: list[str] = []
    position = 0
    for index, raw in enumerate(before):
        text = document.lines[index].rstrip("\r\n")
        code = _code_part(text)
        where = f"line {index + 1} of {submitted.name}"
        shown = repr(text.strip()[:80])
        leave = False
        if index < begin:
            create = _STRICT_CREATE_RE.fullmatch(code)
            assign = _STRICT_ASSIGN_RE.fullmatch(code)
            if create and create["type"] in COPIED_TYPES:
                leave = True
                removed += [f"{create['type']} {name}" for name in
                            split_create_names(create["names"])]
            elif assign and assign["object"] in plots:
                leave = True
            elif re.search(rf"\b(?:{'|'.join(sorted(COPIED_TYPES))})\b", code) or (
                named.search(code)
            ):
                raise PolicyError(
                    f"{where} names a plot the console copy leaves out, and is not a Create "
                    f"of one or one assignment to one that it can read: {shown}"
                )
        else:
            command = _STRICT_COMMAND_RE.fullmatch(code) if index > begin else None
            words = split_create_names(command["names"]) if command else []
            if command and words[-1] in ("On", "Off") and command["command"] == "Toggle":
                words = words[:-1]
            if command and words and all(word in plots for word in words):
                leave = True
            elif named.search(code):
                raise PolicyError(
                    f"{where} names a plot the console copy leaves out, and is not a plot "
                    f"command naming only such plots: {shown}"
                )
        if leave:
            deleted.append(index + 1)
            continue
        if position >= len(after) or after[position] != raw:
            raise PolicyError(
                f"{copy_name} differs from {submitted.name} other than by leaving out plot "
                f"lines: {where} is not its line {position + 1}"
            )
        position += 1
    if position != len(after):
        raise PolicyError(f"{copy_name} has lines after the last of {submitted.name}")
    if sorted(name.split(" ", 1)[1] for name in removed) != sorted(plots):
        raise PolicyError(  # pragma: no cover - every Create of them is left out above
            f"The console copy {copy_name} leaves out {removed}; {submitted.name} creates {plots}"
        )
    return ConsoleCopy(
        copied,
        hashlib.sha256(document.raw_bytes).hexdigest(),
        hashlib.sha256(copied).hexdigest(),
        tuple(removed),
        tuple(deleted),
    )


def _assert_nothing_else_differs(
    submitted: GmatScriptDocument, copy: GmatScriptDocument, removed: tuple[str, ...]
) -> None:
    """The resources, their fields and the ReportFiles, read back from both, agree."""
    plots = {name.split(" ", 1)[1] for name in removed}
    kept_types = {k: v for k, v in created_types(submitted).items() if k not in plots}
    kept_fields = {k: v for k, v in _resource_assignments(submitted).items() if k not in plots}
    if (
        created_types(copy) != kept_types
        or _resource_assignments(copy) != kept_fields
        or report_file_specs(copy) != report_file_specs(submitted)
    ):  # pragma: no cover - the line walk above leaves nothing else to differ
        raise PolicyError(f"The console copy {copy.path.name} reads back other resources")


def write_console_copy(submitted: Path, workdir: Path) -> tuple[Path, ConsoleCopy]:
    """Write ``submitted``'s console copy into ``workdir`` and prove it from disk.

    A copy that fails its proof is removed, so what stays on disk was proven.
    """
    copy = build_console_copy(GmatScriptDocument.read(submitted))
    if copy is None:
        raise PolicyError(f"{submitted.name} creates no OpenFrames or GroundTrack object")
    workdir.mkdir(parents=True, exist_ok=True)
    path = (workdir / f"{submitted.stem}_console_copy{submitted.suffix}").resolve()
    path.write_bytes(copy)
    try:
        return path, prove_console_copy(submitted, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise


_SUBJECT_RE = re.compile(r"\s*(?:GMAT\s+)?(?:Create\s+(?P<type>\w+)\s|(?P<object>\w+)\s*\.)")


def plot_lines(document: GmatScriptDocument) -> tuple[str, ...]:
    """Every line of a plot object, as written, in order: a Create of any plot
    type, an assignment to a plot, or a mission command (Toggle, PenUp, ...)
    naming one."""
    plots = {name for name, kind in created_types(document).items() if kind in PLOT_TYPES}
    found: list[str] = []
    mission = False
    for line in document.lines:
        code = _code_part(line.rstrip("\r\n"))
        if not mission and _BEGIN_MISSION_RE.match(code):
            mission = True
        elif not mission:
            match = _SUBJECT_RE.match(code)
            if match and (match["type"] in PLOT_TYPES or match["object"] in plots):
                found.append(line)
        else:
            words = re.findall(r"\w+", code)
            if words and words[0] in PLOT_COMMANDS and set(words[1:]) & plots:
                found.append(line)
    return tuple(found)


def assert_plot_lines_unchanged(
    official: GmatScriptDocument, submitted: GmatScriptDocument
) -> None:
    """Refuse a submitted script whose plot lines are not the official template's.

    Nothing here edits a plot line, so they must come through byte for byte,
    line endings included, the same lines in the same order. Then the GMAT GUI
    takes the submitted script's plots exactly when it takes the organizer's,
    and the console copy, which leaves out only such lines, is proven of them.
    """
    before, after = plot_lines(official), plot_lines(submitted)
    if before == after:
        return
    changed = next(
        (index for index, pair in enumerate(zip(before, after, strict=False))
         if pair[0] != pair[1]),
        min(len(before), len(after)),
    )
    def shown(lines: tuple[str, ...]) -> str:
        # Unstripped: a trailing blank or a changed line ending is the difference.
        return repr(lines[changed][:120]) if changed < len(lines) else "nothing"

    raise PolicyError(
        f"{submitted.path.name} does not carry {official.path.name}'s plot lines as they are: "
        f"plot line {changed + 1} is {shown(after)} where the official script has "
        f"{shown(before)} ({len(after)} plot lines against {len(before)})"
    )
