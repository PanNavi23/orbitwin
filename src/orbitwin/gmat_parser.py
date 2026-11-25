from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .config import (
    ORGANIZER_TIME_ORIGIN_ANSWER,
    REFUSED_RESOURCE_TYPES,
    SYNTHESIZED_BURN_FRAMES,
    OrbitWinConfig,
)
from .errors import DynamicsError, ParseError, PolicyError
from .gmat_runner import OUTPUT_FILE_FIELDS, PLOT_TYPES, is_bare_file_name
from .mission_clock import (
    check_horizon,
    horizon_override_warning,
    horizon_refusal,
    horizon_warning,
    near_circular_warning,
    time_origin_craft,
)
from .models import (
    BurnSlot,
    EpochAlignment,
    EpochLock,
    HorizonCheck,
    ProbedEpoch,
    PropagateSegment,
    ReportFileSpec,
    Scenario,
    ScriptEncoding,
    SpacecraftSpec,
    StateProbe,
    StateVector,
    alignment_binding_sha256,
    as_vector3,
    file_sha256,
    spacecraft_binding_sha256,
)
from .orbit import (
    KeplerianElements,
    cartesian_to_keplerian,
    first_radius_crossing_s,
    keplerian_to_cartesian,
    orbital_period_s,
    propagate_universal,
)
from .rules import interception_rules

# GMAT refuses to create a resource whose name collides with a mission-sequence
# keyword: "Spacecraft object can not be named to a Command type". The list below
# was produced by feeding each candidate name to a stock GmatConsole R2026a and
# recording which ones the interpreter rejected, so it reflects the binary rather
# than the reference manual. Names such as Save, ElseIf, Collocation and Command
# are deliberately absent because that build accepts them.
GMAT_RESERVED_RESOURCE_NAMES = frozenset(
    {
        "Achieve",
        "Assignment",
        "BeginFiniteBurn",
        "BeginMissionSequence",
        "BeginScript",
        "CallFunction",
        "ClearPlot",
        "Create",
        "Else",
        "EndFiniteBurn",
        "EndFor",
        "EndIf",
        "EndOptimize",
        "EndScript",
        "EndTarget",
        "EndWhile",
        "Equation",
        "FindEvents",
        "For",
        "GMAT",
        "Global",
        "If",
        "Maneuver",
        "MarkPoint",
        "Minimize",
        "NonlinearConstraint",
        "Optimize",
        "PenDown",
        "PenUp",
        "Propagate",
        "Report",
        "RunEstimator",
        "RunSimulator",
        "RunSmoother",
        "ScriptEvent",
        "Set",
        "Stop",
        "Target",
        "Toggle",
        "Vary",
        "While",
        "Write",
    }
)

_CREATE_RE = re.compile(
    r"^\s*(?:GMAT\s+)?Create\s+(?P<type>[A-Za-z_]\w*)\s+(?P<names>[^;]+);",
    re.IGNORECASE,
)
_ASSIGN_RE = re.compile(
    r"^\s*(?:GMAT\s+)?(?P<object>[A-Za-z_]\w*)\.(?P<field>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*=\s*(?P<value>.*?)\s*;\s*$",
    re.IGNORECASE,
)
# How GMAT SaveMission writes a Variable, Array element or String initial value.
_LITERAL_INIT_RE = re.compile(
    r"^\s*(?:GMAT\s+)?(?P<name>[A-Za-z_]\w*)\s*(?P<index>\(\s*\d+\s*(?:,\s*\d+\s*)?\))?"
    r"\s*=\s*(?:(?P<number>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)|'[^']*')"
    r"\s*;\s*$"
)
_LITERAL_TYPES = ("variable", "array", "string")
_MANEUVER_RE = re.compile(
    r"^\s*Maneuver(?:\s+'[^']*')?\s+(?P<burn>[A-Za-z_]\w*)\((?P<spacecraft>[A-Za-z_]\w*)\)\s*;"
    # Nothing after it: R2026a refuses a second command on the line.
    r"\s*$",
    re.IGNORECASE,
)
_ELAPSED_RE = re.compile(
    r"(?P<spacecraft>[A-Za-z_]\w*)\.(?P<unit>ElapsedSecs|ElapsedDays)\s*=\s*(?P<value>[-+0-9.eE]+)",
    re.IGNORECASE,
)
# The options Propagate.html lists inside braces; GMAT spells them case-sensitively.
_STOP_OPTION_RE = re.compile(
    r"OrbitColor\s*=\s*(?:\[[0-9\s]*\]|[A-Za-z]\w*)|StopTolerance\s*=\s*[-+0-9.eE]+"
)
_STOP_GROUP_BODY_RE = re.compile(r"\{(?P<body>[^{}]*)\}")
# An alignment's goal: `A.<E> = B.<E>` or `A.<E> = <number>`. Case-sensitive,
# as GMAT is; anything else is not read as one.
_EPOCH_GOAL_RE = re.compile(
    r"(?P<craft>[A-Za-z_]\w*)\.(?P<scale>A1|TAI|UTC|TT|TDB)ModJulian\s*=\s*"
    r"(?:(?P<other>[A-Za-z_]\w*)\.(?P<other_scale>A1|TAI|UTC|TT|TDB)ModJulian"
    r"|(?P<number>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))"
)
_PROPAGATE_RE = re.compile(r"^\s*Propagate\b", re.IGNORECASE)
_BEGIN_MISSION_RE = re.compile(r"^\s*BeginMissionSequence\s*;", re.IGNORECASE)
_PROPAGATE_LABEL_RE = re.compile(r"^\s*Propagate\s+(?P<label>'[^']*')", re.IGNORECASE)
_STOP_GROUP_RE = re.compile(r"(?:,\s*)?\{[^{}]*\}")
_PROPAGATOR_CALL_RE = re.compile(r"(?P<propagator>[A-Za-z_]\w*)\s*\((?P<spacecraft>[^()]*)\)")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")


def _propagate_scan_text(line: str) -> str:
    """Blank a Propagate line's trailing comment, then its label, which may hold
    text that reads as syntax. Columns are kept so goal spans index the original.
    The cut comes first: a label must not hide the NUL GMAT stops at."""
    cut = len(_code_part(line))
    line = line[:cut] + " " * (len(line) - cut)
    match = _PROPAGATE_LABEL_RE.match(line)
    if match is not None:
        start, end = match.span("label")
        line = line[:start] + " " * (end - start) + line[end:]
    return line


def propagator_calls(line: str) -> list[tuple[str, tuple[str, ...]]]:
    """(propagator, craft) for each call on a Propagate line, read as the parser reads it."""
    head = re.sub(r"^\s*Propagate\b", "", _propagate_scan_text(line), count=1, flags=re.IGNORECASE)
    head = _STOP_GROUP_RE.sub(" ", head)
    return [
        (call.group("propagator"),
         tuple(name.strip() for name in call.group("spacecraft").split(",") if name.strip()))
        for call in _PROPAGATOR_CALL_RE.finditer(head)
    ]


def _propagated_spacecraft(
    scan: str, line_index: int
) -> tuple[tuple[str, ...], tuple[str, ...], EpochLock | None]:
    """Return the craft a Propagate line steps, its propagators in call order,
    and what holds the craft to one epoch.

    Craft come from the propagator calls, not the goals: under `Synchronized`
    one craft's goal stops them all. Separate calls without it return None,
    since GMAT halts every craft at the first goal reached. `scan` is the
    output of `_propagate_scan_text`.
    """
    head = re.sub(r"^\s*Propagate\b", "", scan, count=1, flags=re.IGNORECASE)
    depth = outside_groups = 0
    for char in head:
        depth += {"(": 1, ")": -1}.get(char, 0)
        outside_groups += char == "{" and depth == 0
    if outside_groups > 1:
        # GMAT R2026a: "The propsetup string '' does not identify any spacecraft".
        raise ParseError(
            f"Propagate line {line_index + 1} has {outside_groups} stop-condition groups "
            "after its calls; GMAT reads one"
        )
    head = _STOP_GROUP_RE.sub(" ", head).strip().rstrip(";").rstrip()
    calls = list(_PROPAGATOR_CALL_RE.finditer(head))
    if not calls:
        raise ParseError(f"Propagate line {line_index + 1} names no propagator and spacecraft")
    synchronized = False
    for mode in head[: calls[0].start()].split():
        if mode == "Synchronized":
            synchronized = True
        elif mode == "BackProp":
            raise ParseError(
                f"Propagate line {line_index + 1} propagates backwards (BackProp); every "
                "duration and epoch in the segment would run the other way"
            )
        else:
            raise ParseError(f"Propagate line {line_index + 1} has an unsupported mode {mode!r}")
    gaps = [head[calls[index].end() : calls[index + 1].start()] for index in range(len(calls) - 1)]
    stray = "".join([*gaps, head[calls[-1].end() :]]).strip()
    if stray:
        raise ParseError(
            f"Propagate line {line_index + 1} has unsupported text around its propagator "
            f"calls: {stray!r}"
        )
    names: list[str] = []
    for call in calls:
        for raw in call.group("spacecraft").split(","):
            name = raw.strip()
            if not _IDENTIFIER_RE.match(name):
                raise ParseError(
                    f"Propagate line {line_index + 1} lists {name!r} in "
                    f"{call.group('propagator')}(...), which is not a spacecraft name"
                )
            if name in names:
                raise ParseError(
                    f"Propagate line {line_index + 1} propagates {name} more than once"
                )
            names.append(name)
    propagators = tuple(call.group("propagator") for call in calls)
    if len(calls) == 1:
        return tuple(names), propagators, "propagator"
    return tuple(names), propagators, "synchronized" if synchronized else None


_UTF8_BOM = b"\xef\xbb\xbf"


def _name_the_byte(raw: bytes, offset: int) -> str:
    """What the byte at ``offset`` is, said the way a person can act on."""
    for width in (1, 2, 3, 4):
        chunk = raw[offset : offset + width]
        if len(chunk) < width:
            break
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(text) == 1:
            return f"the start of {text!r} (U+{ord(text):04X}, {width} UTF-8 bytes)"
    return "not the start of a valid UTF-8 character either, so the file is not even UTF-8"


#: GmatConsole ends a line at CR, LF or CRLF. str.splitlines also ends one at VT,
#: FF and the file, group and record separators, so no reader uses it on a script.
_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")
_BREAK_RE = re.compile(rb"\r\n|\r|\n")
#: Where GMAT stops reading: NUL on every build, and Ctrl-Z (0x1A) on Windows,
#: whose text-mode stream takes it for the end of the file. R2026a on Linux reads
#: Ctrl-Z as text and refuses it as a command, so in code it is refused too.
_STOP_CHARACTERS = "\x00\x1a"
_CTRL_Z = "\x1a"
#: Text to GmatConsole, but whitespace to Python's \s or a line break to
#: str.splitlines: in code, our regexes would read past what GMAT reads. The
#: script is ASCII, so these are all of them (tests/unit/test_script_line_model.py).
CODE_CONTROL_CHARACTERS = frozenset(
    char
    for char in map(chr, range(128))
    if char.isspace() or len(f"a{char}b".splitlines()) > 1
) - frozenset(" \t\r\n")
_CODE_CONTROL_RE = re.compile("[" + re.escape("".join(sorted(CODE_CONTROL_CHARACTERS))) + "]")
_UNUSUAL_RE = re.compile("[" + re.escape("".join(sorted({*CODE_CONTROL_CHARACTERS, *_STOP_CHARACTERS}))) + "]")
#: What a refusal of the input itself tells the operator to do.
STOP_ADVICE = (
    "; if this is the official script, it may not be edited: stop and take it to the "
    "organizer (docs/RUNBOOK.md, Stop rules)"
)


def script_lines(text: str) -> tuple[str, ...]:
    """A script's lines as GmatConsole reads them, endings kept."""
    return tuple(_LINE_RE.findall(text))


def _code_part(line: str) -> str:
    """What GMAT reads of a line as its command (R2026a TextParser::EvaluateBlock).

    GMAT stops reading at NUL. It looks only at the line's first %: that % is
    text if it lies between the line's first two single quotes, and then the
    whole line is the command, comment and all; otherwise the comment starts
    there. Double quotes are not quotes. A prefix, so spans hold.
    """
    nul = line.find("\x00")
    if nul >= 0:
        line = line[:nul]
    percent = line.find("%")
    if percent < 0:
        return line
    quote = line.find("'")
    if 0 <= quote < percent < line.find("'", quote + 1):
        return line
    return line[:percent]


def _unquoted(code: str) -> str:
    """``code`` with each single-quoted span emptied: GMAT's only quote is '."""
    return re.sub(r"'[^']*'", "''", code) if "'" in code else code


def _continues(view: str) -> bool:
    """ScriptReadWriter::HasEllipse on a line GMAT reads as a command: its text,
    trailing blanks and then semicolons trimmed, ends in `...`, comment and all."""
    body = view.strip(" \t")
    return bool(body) and not body.startswith("%") and view.rstrip(" \t").rstrip(";").endswith(
        "..."
    )


def _assert_one_line_model(source: Path, text: str, lines: tuple[str, ...]) -> None:
    """Refuse what would make a reader here see other commands than GMAT runs.

    GMAT stops reading a script at NUL (R2026a, exit status 0 and a short
    report), and a Windows build at Ctrl-Z too. Padding or a comment after one
    changes nothing; a command after one would not run, so that is refused.
    CODE_CONTROL_CHARACTERS are text to GMAT and whitespace or a break to our
    regexes, so in code they are refused; in a comment neither side reads them.
    A line whose first % is quoted is the command whole, comment and all; one
    with a % outside its quotes is refused, as no reader here models it. So is
    a double quote outside single quotes, and a line GMAT continues onto the next.
    """
    unusual = _UNUSUAL_RE.search(text) is not None
    double = '"' in text
    dots = "..." in text
    stopped: tuple[int, str] | None = None
    for number, line in enumerate(lines, 1):
        code = _code_part(line)
        if stopped is not None:
            at, char = stopped
            if code.strip():
                where = ("a NUL byte, where GMAT stops reading the script" if char == "\x00"
                         else "Ctrl-Z (0x1A), where a Windows GMAT stops reading the script")
                raise ParseError(
                    f"{source.name} line {at} holds {where}, and line {number} has a command "
                    f"after it that would not run{STOP_ADVICE}"
                )
            if char == "\x00":
                continue  # GMAT never reads a line past a NUL
        if dots:
            # As GMAT reads the line: up to a NUL, and on Windows up to a Ctrl-Z.
            read = line.rstrip("\r\n").split("\x00", 1)[0]
            if _continues(read) or _continues(read.split(_CTRL_Z, 1)[0]):
                raise ParseError(
                    f"{source.name} line {number} ends in `...`, comment included, which GMAT "
                    "reads as a line continuation: it joins the next line to this one, or "
                    f"refuses the script, and this parser does not model that: "
                    f"{line.strip()[:80]!r}{STOP_ADVICE}"
                )
        if unusual:
            if _CTRL_Z in code:
                raise ParseError(
                    f"{source.name} line {number} holds Ctrl-Z (0x1A) outside a comment: R2026a "
                    f"refuses it as a command, and a Windows GMAT stops reading there{STOP_ADVICE}"
                )
            if "\x00" in line:
                # Every build stops here, whatever stopped a Windows one before.
                stopped = (number if stopped is None else stopped[0], "\x00")
            elif stopped is None and _CTRL_Z in line:
                stopped = (number, _CTRL_Z)
            control = _CODE_CONTROL_RE.search(code)
            if control:
                quoted = control.start() > code.find("'") >= 0 and "'" in code[control.start():]
                how = ("in a quoted value, which GMAT keeps as text and this parser does not "
                       "model" if quoted else
                       "outside a comment, which GmatConsole reads as text and this parser as "
                       "whitespace or a line break")
                raise ParseError(
                    f"{source.name} line {number} holds the control character "
                    f"{control.group()!r} {how}{STOP_ADVICE}"
                )
        if double and '"' in _unquoted(code):
            raise ParseError(
                f"{source.name} line {number} holds a double quote outside single quotes and "
                "outside a comment, which GMAT does not read as a quote (R2026a refused "
                "\"Report%1.txt\" as a value) and this parser does not model: "
                f"{line.strip()[:80]!r}{STOP_ADVICE}"
            )
        # Only a quoted first % leaves one in the code part; a % in a later quoted
        # value is text to GMAT too.
        if "%" in code and "%" in _unquoted(code):
            raise ParseError(
                f"{source.name} line {number}: its first % is inside quotes, so GMAT reads "
                "the whole line, comment and all, as the command, and this parser does "
                f"not model that: {line.strip()[:80]!r}{STOP_ADVICE}"
            )


def _assert_source_is_ascii(source: Path, raw: bytes) -> None:
    """Refuse an input script GMAT will not open, at the first point it is visible.

    Measured on GMAT R2026a: a script containing any byte above 127 is refused
    outright, before anything is interpreted --

        Interpreter Exception: The script "..." contains characters outside of
        the ASCII character set; please fix the file before proceeding.

    ``script_renderer._assert_ascii`` already says that about the script we
    generate. Nothing said it about the script we are handed. On competition day
    the organizer's file is one nobody has seen, and every byte of it that is not
    a burn value or an elapsed time is copied into the submission verbatim -- so a
    curly quote in a comment, an accent in a resource name or a mark left by a
    text editor is carried through into a submission that scores nothing because
    it cannot be opened.

    All four ways it can arrive were reproduced against this parser before this
    check existed. A CJK comment and an accented resource name both parsed
    without a word -- the accented one became ``scenario.pursuer.name``, because
    ``_ASSIGN_RE`` spells an object as ``[A-Za-z_]\\w*`` and Python's ``\\w`` is
    Unicode-aware -- and were caught only by the renderer's guard at the end of
    the run, which reported them against the *final* script: true, but it sends
    the operator looking for our bug in a defect that was in the input all along.
    A lone 0xFF raised UnicodeDecodeError out of ``read``, with a byte position
    into the decode stream and no line number in it. The fourth, a UTF-8
    byte-order mark, was caught by nothing at all: see the branch below.

    The message names the byte, its offset and its line because the person
    reading it is under a clock and needs to reach the character, not to start an
    investigation.
    """
    if raw.isascii():
        # The whole point is that this runs on every script every time, so the
        # answer for a clean file is one C-level scan and nothing else.
        return
    offsets = [index for index, value in enumerate(raw) if value > 0x7F]
    offset = offsets[0]
    breaks = list(_BREAK_RE.finditer(raw, 0, offset))
    line_number = len(breaks) + 1
    line_start = breaks[-1].end() if breaks else 0
    following = _BREAK_RE.search(raw, offset)
    line_text = raw[line_start : following.start() if following else len(raw)]
    excerpt = line_text.rstrip(b"\r").decode("utf-8", errors="replace")
    if len(excerpt) > 200:
        excerpt = excerpt[:200] + " [truncated]"
    remainder = (
        ""
        if len(offsets) == 1
        else f" It is the first of {len(offsets)} such bytes in the file."
    )

    if raw.startswith(_UTF8_BOM):
        # Deliberate, and that is exactly why it needs saying. `ScriptEncoding`
        # records the mark and the renderer puts it back byte-for-byte, so the
        # original's bytes survive a round trip -- which is the right instinct for
        # a file we must not mutate, and here it preserves three bytes above 127
        # into the artifact we submit. Worse, it is the one non-ASCII case no
        # later gate can catch: `assert_final_script_policy` was measured to pass
        # a BOM'd script, because `_assert_ascii` reads the decoded lines and
        # `utf-8-sig` has already eaten the mark. Refusing here is not a reversal
        # of that design; it is the only place left that can see the problem.
        raise ParseError(
            f"{source} begins with a UTF-8 byte-order mark: the non-ASCII bytes "
            f"EF BB BF at byte offset 0, line 1. GMAT R2026a refuses to interpret a "
            "script containing any byte above 127, and the mark is preserved verbatim "
            "into anything rendered from this file, where no later check can see it -- "
            "the final-script policy guard reads decoded lines, and decoding has already "
            "removed it. Re-save the script as UTF-8 without a byte-order mark, or strip "
            "its first three bytes, and run again."
        )

    raise ParseError(
        f"{source} line {line_number}, column {offset - line_start + 1} (byte offset "
        f"{offset}) is the non-ASCII byte 0x{raw[offset]:02X}, which is "
        f"{_name_the_byte(raw, offset)}.{remainder} GMAT R2026a refuses to interpret a "
        'script containing any byte above 127 -- \'The script "..." contains characters '
        "outside of the ASCII character set; please fix the file before proceeding' -- so "
        "neither this script nor anything rendered from it can be submitted. The line "
        f"reads: {excerpt!r}"
    )


@dataclass(frozen=True, slots=True)
class GmatScriptDocument:
    path: Path
    raw_bytes: bytes
    text: str
    lines: tuple[str, ...]
    encoding: ScriptEncoding

    @classmethod
    def read(cls, path: str | Path) -> GmatScriptDocument:
        source = Path(path)
        raw = source.read_bytes()
        # Before the decode, so a byte that is not UTF-8 at all is reported as a
        # line and column of this script rather than as a codec's byte position.
        _assert_source_is_ascii(source, raw)
        has_bom = raw.startswith(_UTF8_BOM)
        decoded = raw.decode("utf-8-sig")
        lines = script_lines(decoded)
        _assert_one_line_model(source, decoded, lines)
        newline = "\r\n" if b"\r\n" in raw else "\n"
        return cls(
            source,
            raw,
            decoded,
            lines,
            ScriptEncoding(newline=newline, has_utf8_bom=has_bom),
        )


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _as_float(value: str, label: str) -> float:
    try:
        result = float(_strip_quotes(value))
    except ValueError as exc:
        raise ParseError(f"{label} is not a real number: {value!r}") from exc
    if not math.isfinite(result):
        raise ParseError(f"{label} is not finite")
    return result


def _as_bool(value: str, label: str) -> bool:
    lowered = _strip_quotes(value).strip().lower()
    if lowered in {"true", "on"}:
        return True
    if lowered in {"false", "off"}:
        return False
    raise ParseError(f"{label} is not a GMAT boolean: {value!r}")


def _object_fields(
    assignments: dict[str, dict[str, tuple[str, int]]], object_name: str
) -> dict[str, str]:
    return {field: value_line[0] for field, value_line in assignments.get(object_name, {}).items()}


def _report_delimiter(value: str, label: str) -> str:
    """The character GMAT writes for a Delimiter assignment.

    Measured on R2026a: it writes the first character of the text, so '\\t'
    writes a backslash and ';,' a semicolon; '' writes a space; an unquoted
    comma works; double quotes are an interpreter error.
    """
    text = value.strip()
    if text.startswith('"'):
        raise ParseError(f"{label} is double-quoted, which GMAT R2026a refuses: {value!r}")
    if len(text) >= 2 and text[0] == text[-1] == "'":
        text = text[1:-1]
    return text[0] if text else " "


def _report_column_width(value: str, label: str) -> int:
    width = _as_float(value, label)
    if width < 1 or width != int(width):
        raise ParseError(f"{label} is not a positive integer: {value!r}")
    return int(width)


def _report_file_spec(name: str, fields: dict[str, str]) -> ReportFileSpec:
    params_text = fields.get("Add", "{}").strip()
    if params_text.startswith("{") and params_text.endswith("}"):
        parameters = tuple(part.strip() for part in params_text[1:-1].split(",") if part.strip())
    else:
        parameters = (params_text,) if params_text else ()
    return ReportFileSpec(
        name=name,
        filename=_strip_quotes(fields.get("Filename", f"{name}.txt")),
        parameters=parameters,
        precision=int(_as_float(fields.get("Precision", "16"), f"{name}.Precision")),
        append=_as_bool(fields.get("AppendToExistingFile", "false"), f"{name}.Append"),
        write_report=_as_bool(fields.get("WriteReport", "true"), f"{name}.WriteReport"),
        write_headers=_as_bool(fields.get("WriteHeaders", "true"), f"{name}.WriteHeaders"),
        delimiter=_report_delimiter(fields.get("Delimiter", "' '"), f"{name}.Delimiter"),
        fixed_width=_as_bool(fields.get("FixedWidth", "true"), f"{name}.FixedWidth"),
        column_width=_report_column_width(fields.get("ColumnWidth", "23"), f"{name}.ColumnWidth"),
        left_justify=_as_bool(fields.get("LeftJustify", "On"), f"{name}.LeftJustify"),
        zero_fill=_as_bool(fields.get("ZeroFill", "Off"), f"{name}.ZeroFill"),
        solver_iterations=_strip_quotes(fields.get("SolverIterations", "Current")),
    )


def report_file_specs(path: str | Path | GmatScriptDocument) -> tuple[ReportFileSpec, ...]:
    """The script's ReportFiles, without parsing the rest of it.

    For tools that only read a report. A duplicate assignment to a ReportFile
    field is refused, as parse_gmat_script refuses it.
    """
    document = path if isinstance(path, GmatScriptDocument) else GmatScriptDocument.read(path)
    names = [name for name, kind in created_types(document).items() if kind.lower() == "reportfile"]
    fields: dict[str, dict[str, str]] = {name: {} for name in names}
    in_mission = False
    for line_index, line in enumerate(document.lines):
        code = _code_part(line)
        if _BEGIN_MISSION_RE.match(code):
            in_mission = True
            continue
        match = _ASSIGN_RE.match(code)
        named = re.match(r"^\s*(?:GMAT\s+)?([A-Za-z_]\w*)\s*\.", code)
        if in_mission:
            # A field set while the mission runs changes what is written after it.
            if named and named.group(1) in fields:
                raise ParseError(
                    f"Line {line_index + 1} sets ReportFile {named.group(1)} in the mission "
                    f"sequence: {line.strip()[:80]!r}"
                )
            continue
        if match is None and named and named.group(1) in fields:
            raise ParseError(
                f"Line {line_index + 1} sets a field of ReportFile {named.group(1)} in a form "
                f"this parser cannot read: {line.strip()[:80]!r}"
            )
        if match and match.group("object") in fields:
            found = fields[match.group("object")]
            if match.group("field") in found:
                raise ParseError(
                    f"Duplicate assignment for {match.group('object')}.{match.group('field')} "
                    f"on line {line_index + 1}"
                )
            found[match.group("field")] = match.group("value")
    return tuple(_report_file_spec(name, fields[name]) for name in names)


@dataclass(frozen=True, slots=True)
class ScheduledManeuver:
    """One Maneuver command, at the sum of the Propagate goals before it."""

    burn: str
    spacecraft: str
    #: From the end of the epoch alignment, or from the start without one; a
    #: head-start burn's from the alignment craft's own epoch.
    time_s: float
    components_km_s: tuple[float, float, float]
    #: (CoordinateSystem, Origin, Axes) of its ImpulsiveBurn.
    frame: tuple[str, str, str]
    line_index: int
    #: Before the epoch alignment (COMPETITION_POLICY.md, head-start burns).
    head_start: bool = False


@dataclass(frozen=True, slots=True)
class MissionSchedule:
    """What a mission sequence flies, read without the initial states."""

    #: The craft the first command flies alone, the epoch alignment's; else None.
    alignment_craft: str | None
    maneuvers: tuple[ScheduledManeuver, ...]
    #: The sum of the Propagate goals after the alignment.
    end_s: float
    #: The longest step the flown propagators take; None if a step size is not a number.
    max_step_s: float | None = None
    #: The head-start goals, inside the alignment's length; 0 without head-start burns.
    head_start_s: float = 0.0

    @property
    def total_dv_km_s(self) -> float:
        return float(sum(math.hypot(*maneuver.components_km_s) for maneuver in self.maneuvers))


def mission_schedule(path: str | Path | GmatScriptDocument) -> MissionSchedule:
    """The burns and Propagate goals of a script, for tools that check a report of it.

    parse_gmat_script needs the initial states, which only GmatConsole reads
    for an EarthFixed insertion or two epochs. This reads the same mission
    lines the same way and fails closed on the same commands. The alignment's
    length needs the epochs, so it is left to the report.
    """
    document = path if isinstance(path, GmatScriptDocument) else GmatScriptDocument.read(path)
    types = {name: kind.lower() for name, kind in created_types(document).items()}
    assignments: dict[str, dict[str, tuple[str, int]]] = {}
    alignment_craft: str | None = None
    maneuvers: list[ScheduledManeuver] = []
    elapsed_s = 0.0
    flown: set[str] = set()
    in_mission = first = False
    aligning_line = _head_start_alignment_line(document)
    head_start_s = 0.0
    for line_index, line in enumerate(document.lines):
        code = _code_part(line)
        command = code.strip()
        if _BEGIN_MISSION_RE.match(code):
            if in_mission:
                raise ParseError("Multiple BeginMissionSequence commands are not supported")
            in_mission = first = True
            continue
        if not in_mission:
            match = _ASSIGN_RE.match(code)
            if match:
                fields = assignments.setdefault(match.group("object"), {})
                if match.group("field") in fields:
                    raise ParseError(
                        f"Duplicate assignment for {match.group('object')}.{match.group('field')} "
                        f"on lines {fields[match.group('field')][1] + 1} and {line_index + 1}"
                    )
                fields[match.group("field")] = (match.group("value"), line_index)
            continue
        if not command or plot_command(code, document) is not None:
            continue
        is_first, first = first, False
        maneuver = _MANEUVER_RE.match(code)
        if maneuver:
            burn, craft = maneuver.group("burn"), maneuver.group("spacecraft")
            if types.get(burn) != "impulsiveburn":
                raise ParseError(f"Maneuver references undeclared ImpulsiveBurn {burn}")
            if any(earlier.burn == burn for earlier in maneuvers):
                raise ParseError(
                    f"Burn {burn} is maneuvered more than once; repeated burn objects are "
                    "unsupported"
                )
            slot = _slot_from_resource(assignments, burn, craft, line_index)
            before = aligning_line is not None and line_index < aligning_line
            maneuvers.append(ScheduledManeuver(
                burn, craft, head_start_s if before else elapsed_s,
                (slot.element1_km_s, slot.element2_km_s, slot.element3_km_s),
                (slot.coordinate_system, slot.origin, slot.axes), line_index, before,
            ))
            continue
        if not _PROPAGATE_RE.match(code):
            raise ParseError(
                f"Line {line_index + 1} of the mission sequence is neither a Propagate nor a "
                f"Maneuver: {line.strip()[:80]!r}. Nothing else there is modelled"
            )
        scan = _propagate_scan_text(line)
        stepped, propagators, _ = _propagated_spacecraft(scan, line_index)
        flown.update(propagators)
        items = [item.strip() for group in _STOP_GROUP_BODY_RE.finditer(scan)
                 for item in group.group("body").split(",")]
        goals = [item for item in items if not _STOP_OPTION_RE.fullmatch(item)]
        if line_index == aligning_line or (aligning_line is None and is_first
                                           and len(stepped) == 1):
            # Its length is the head start, which only the epochs give.
            alignment_craft = stepped[0]
            continue
        if aligning_line is not None and line_index < aligning_line:
            found = _HEAD_START_RE.match(line.rstrip("\r\n"))
            aligning = _propagated_spacecraft(
                _propagate_scan_text(document.lines[aligning_line]), aligning_line)[0]
            if found is None or stepped != aligning:
                raise ParseError(
                    f"Line {line_index + 1}, before the epoch alignment, is not a head-start "
                    f"`Propagate <P>({'/'.join(aligning)}) {{<craft>.ElapsedSecs = <s>}};`: "
                    f"{line.strip()[:80]!r}"
                )
            head_start_s += _as_float(found.group("value"), f"line {line_index + 1}'s goal")
            continue
        if len(stepped) < 2:
            raise ParseError(
                f"Propagate line {line_index + 1} flies {list(stepped)} alone after the first "
                "command, so the craft are no longer on one clock"
            )
        if not goals or not all(_ELAPSED_RE.fullmatch(item) for item in goals):
            raise ParseError(
                f"Propagate line {line_index + 1} stops on {goals}, not on elapsed-time goals "
                "alone, so its length cannot be read"
            )
        units = {found.group("unit").lower() for found in map(_ELAPSED_RE.fullmatch, goals)
                 if found}
        if len(units) != 1:
            raise ParseError(f"Mixed elapsed-time units on Propagate line {line_index + 1}")
        values = [
            float(found.group("value"))
            * (86400.0 if found.group("unit").lower() == "elapseddays" else 1.0)
            for found in (_ELAPSED_RE.fullmatch(item) for item in goals) if found
        ]
        if max(values) - min(values) > 1.0e-8:
            raise ParseError(
                f"Spacecraft do not share the same elapsed-time goal on line {line_index + 1}"
            )
        elapsed_s += values[0]
    if not in_mission:
        raise ParseError("BeginMissionSequence was not found")
    early = [m for m in maneuvers if m.head_start and m.spacecraft != alignment_craft]
    if early:
        raise ParseError(
            f"Line {early[0].line_index + 1} maneuvers {early[0].spacecraft} before the epoch "
            f"alignment, which flies {alignment_craft} alone"
        )
    return MissionSchedule(alignment_craft, tuple(maneuvers), elapsed_s,
                           _longest_step_s(assignments, flown), head_start_s)


def _head_start_alignment_line(document: GmatScriptDocument) -> int | None:
    """The epoch alignment's line when head-start pairs come before it, else None."""
    in_mission = False
    commands: list[tuple[int, str]] = []
    for line_index, line in enumerate(document.lines):
        code = _code_part(line)
        if _BEGIN_MISSION_RE.match(code):
            in_mission = True
            continue
        if not in_mission or not code.strip() or plot_command(code, document) is not None:
            continue
        if _MANEUVER_RE.match(code):
            commands.append((line_index, "M"))
            continue
        if not _PROPAGATE_RE.match(code):
            return None
        if _EPOCH_GOAL_RE.search(_propagate_scan_text(line)):
            kinds = "".join(kind for _, kind in commands)
            return line_index if kinds and kinds == "PM" * (len(kinds) // 2) else None
        commands.append((line_index, "P"))
    return None


#: GMAT's Propagator defaults (Propagator.html), s.
_GMAT_MAX_STEP_S = 2700.0
_GMAT_INITIAL_STEP_S = 60.0


def _longest_step_s(
    assignments: dict[str, dict[str, tuple[str, int]]], propagators: set[str]
) -> float | None:
    """The longest step these propagators take: MaxStep, or InitialStepSize for a fixed step."""
    longest = 0.0
    for name in propagators:
        fields = assignments.get(name, {})
        for field, default in (("MaxStep", _GMAT_MAX_STEP_S),
                               ("InitialStepSize", _GMAT_INITIAL_STEP_S)):
            text = _strip_quotes(fields[field][0]) if field in fields else None
            try:
                step = abs(float(text)) if text is not None else default
            except ValueError:
                return None
            if not math.isfinite(step):
                return None
            longest = max(longest, step)
    return longest or None


def _required_field(fields: dict[str, str], object_name: str, field_name: str) -> str:
    try:
        return fields[field_name]
    except KeyError as exc:
        raise ParseError(f"Missing required assignment {object_name}.{field_name}") from exc


#: GMAT's Degree/Order when unset (ForceModel.html; GmatConsole R2026a gives
#: the same state as an explicit 4x4).
_GMAT_DEFAULT_GRAVITY_DEGREE = 4
_FORCE_OFF = frozenset({"off", "false", "none"})


def _ignored_force_warnings(name: str, fields: dict[str, str], central_body: str) -> list[str]:
    """One warning per effect in ForceModel ``name`` that point-mass two-body leaves out."""
    values = {key.lower(): (key, _strip_quotes(value).strip()) for key, value in fields.items()}

    def turned_on(field: str) -> str | None:
        key, value = values.get(field.lower(), (field, "None"))
        return None if value.lower() in _FORCE_OFF else f"{name}.{key}={value}"

    def bodies(field: str) -> list[str] | None:
        if field.lower() not in values:
            return None
        listed = values[field.lower()][1].strip("{}")
        return [body.strip() for body in listed.split(",") if body.strip()]

    warnings: list[str] = []
    # Drag is the deprecated spelling; GUI-saved scripts set Drag.AtmosphereModel.
    drag = [setting for setting in (turned_on("Drag"), turned_on("Drag.AtmosphereModel")) if setting]
    if drag:
        warnings.append(f"Analytical seed ignores drag: {', '.join(drag)}")
    for field, effect in (("SRP", "solar radiation pressure"), ("RelativisticCorrection", "relativity")):
        setting = turned_on(field)
        if setting:
            warnings.append(f"Analytical seed ignores {effect}: {setting}")
    point_masses = bodies("PointMasses") or []
    third_bodies = [body for body in point_masses if body.lower() != central_body.lower()]
    if third_bodies:
        warnings.append(f"Analytical seed ignores third-body point masses {third_bodies}: {name}.PointMasses")
    primaries = bodies("PrimaryBodies")
    if primaries is None:
        # Measured on R2026a: with no PrimaryBodies the central body still gets
        # the default 4x4 field, unless PointMasses names it.
        named = any(body.lower() == central_body.lower() for body in point_masses)
        primaries = [] if named else [central_body]
    others = [body for body in primaries if body != central_body]
    if others:
        warnings.append(f"Analytical seed ignores additional primary bodies {others}")
    for body in primaries:
        size: dict[str, tuple[float, str]] = {}
        for part in ("Degree", "Order"):
            entry = values.get(f"gravityfield.{body}.{part}".lower())
            size[part] = (
                (_as_float(entry[1], f"{name}.{entry[0]}"), entry[1])
                if entry
                else (float(_GMAT_DEFAULT_GRAVITY_DEGREE), f"{_GMAT_DEFAULT_GRAVITY_DEGREE} (GMAT default)")
            )
        if size["Degree"][0] > 0 or size["Order"][0] > 0:
            warnings.append(
                f"Analytical seed ignores non-spherical gravity: {name}.GravityField.{body} "
                f"Degree={size['Degree'][1]}, Order={size['Order'][1]}"
            )
        # Measured on R2026a: TideModel changes nothing below degree 2.
        tide = turned_on(f"GravityField.{body}.TideModel")
        if tide and size["Degree"][0] >= 2:
            warnings.append(f"Analytical seed ignores tides: {tide}")
    return warnings


#: GMAT's weather sources when unset (ForceModel.html, R2026a).
_GMAT_DEFAULT_WEATHER_SOURCE = "ConstantFluxAndGeoMag"
_WEATHER_FILES = frozenset({"cssispaceweatherfile", "schattenfile"})
#: Recommended policy.stop_depth_km for drag that reads a weather file: twice
#: the 665 m R4 moved on the CSSI file at +30 % F10.7/Ap (R2026a, 2026-11-02).
WEATHER_FILE_STOP_DEPTH_KM = 1.5


def drag_weather_sources(force_model_fields: dict[str, str]) -> list[dict[str, Any]]:
    """Each drag-carrying ForceModel's space-weather sources, GMAT's default when unset."""
    models: dict[str, dict[str, tuple[str, str]]] = {}
    for key, value in force_model_fields.items():
        name, _, field = key.partition(".")
        models.setdefault(name, {})[field.lower()] = (field, _strip_quotes(value).strip())
    sources: list[dict[str, Any]] = []
    for name, fields in sorted(models.items()):
        atmosphere = next(
            (fields[f][1] for f in ("drag.atmospheremodel", "drag")
             if f in fields and fields[f][1].lower() not in _FORCE_OFF),
            None,
        )
        if atmosphere is None:
            continue
        historic = fields.get("drag.historicweathersource", ("", _GMAT_DEFAULT_WEATHER_SOURCE))[1]
        predicted = fields.get("drag.predictedweathersource", ("", _GMAT_DEFAULT_WEATHER_SOURCE))[1]
        sources.append({
            "force_model": name,
            "atmosphere_model": atmosphere,
            "historic_weather_source": historic,
            "predicted_weather_source": predicted,
            "reads_weather_file": bool({historic.lower(), predicted.lower()} & _WEATHER_FILES),
        })
    return sources


def stop_depth_advice(scenario: Scenario, stop_depth_km: float) -> dict[str, Any]:
    """What the template's drag says about policy.stop_depth_km. Advice only."""
    sources = drag_weather_sources(scenario.force_model_fields)
    reading = [source["force_model"] for source in sources if source["reads_weather_file"]]
    if reading:
        text = (
            f"Drag in {', '.join(reading)} reads a space-weather file, which the organizer's "
            "copy may hold other values in: R4 on the CSSI file at +30 % F10.7/Ap moved "
            f"665 m. Recommended policy.stop_depth_km: {WEATHER_FILE_STOP_DEPTH_KM:g} km."
        )
    else:
        carrying = "; ".join(
            f"{source['force_model']} historic {source['historic_weather_source']}, "
            f"predicted {source['predicted_weather_source']}"
            for source in sources
        )
        text = (
            f"Drag reads no space-weather file ({carrying})."
            if sources
            else "No ForceModel carries drag."
        ) + (
            " 50 m covered the EOP/UT1 (<= 0.11 m) and EarthFixed insertion (<= 25.5 m) "
            "changes measured on constant flux; other data files were not measured."
        )
    return {
        "drag_weather_sources": sources,
        "recommended_stop_depth_km": WEATHER_FILE_STOP_DEPTH_KM if reading else None,
        "configured_stop_depth_km": stop_depth_km,
        "advice": f"{text} policy.stop_depth_km is {stop_depth_km:g} km.",
    }


def _flown_force_models(
    created: dict[str, list[str]],
    assignments: dict[str, dict[str, tuple[str, int]]],
    propagators: list[str],
) -> list[tuple[str, str | None]]:
    """(propagator, its FM) for each propagator flown, each FM once; None when unset."""
    declared = set(created.get("forcemodel", []))
    flown: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for prop_name in propagators:
        fields = {key.lower(): value for key, value in _object_fields(assignments, prop_name).items()}
        force_name = _strip_quotes(fields["fm"]).strip() if "fm" in fields else None
        if force_name is not None and force_name not in declared:
            raise ParseError(
                f"{prop_name}.FM is {force_name!r}, which the script does not create as a ForceModel"
            )
        if force_name is None or force_name not in seen:
            flown.append((prop_name, force_name))
        if force_name is not None:
            seen.add(force_name)
    return flown


#: Probed against in-process states. Measured on every example template:
#: ≤ 1.1e-11 km and ≤ 3.5e-9 km/s, the latter from our mu 398600.4418 against
#: GMAT's 398600.4415 for Earth.
PROBE_POSITION_TOLERANCE_KM = 1.0e-6
PROBE_VELOCITY_TOLERANCE_KM_S = 1.0e-8
#: Probed A1 epochs this close are one start. Precision 17 is bit-exact; one A1
#: day step near MJD 21545 is 0.31 µs and GMAT jointly propagated 1.57 µs apart.
SYNCHRONOUS_EPOCH_TOLERANCE_S = 1.0e-6

_NEEDS_CONSOLE = (
    "; only GmatConsole can read that, so run with --gmat-console and --startup-file "
    "(the state probe, docs/ASYNC_START_AND_INSERTION.md §1)"
)


#: Each state type's fields (SpacecraftOrbitState.html), in the order R2026a's
#: Write {Style = Script} prints them. GMAT refuses a craft that sets fields of
#: two types before BeginMissionSequence.
STATE_TYPE_FIELDS: dict[str, tuple[str, ...]] = {
    "Cartesian": ("X", "Y", "Z", "VX", "VY", "VZ"),
    "Keplerian": ("SMA", "ECC", "INC", "RAAN", "AOP", "TA"),
    "ModifiedKeplerian": ("RadPer", "RadApo", "INC", "RAAN", "AOP", "TA"),
    "SphericalAZFPA": ("RMAG", "RA", "DEC", "VMAG", "AZI", "FPA"),
    "SphericalRADEC": ("RMAG", "RA", "DEC", "VMAG", "RAV", "DECV"),
    "Planetodetic": (
        "PlanetodeticRMAG", "PlanetodeticLON", "PlanetodeticLAT", "PlanetodeticVMAG",
        "PlanetodeticAZI", "PlanetodeticHFPA",
    ),
    "Equinoctial": ("SMA", "EquinoctialH", "EquinoctialK", "EquinoctialP", "EquinoctialQ", "MLONG"),
    "AlternateEquinoctial": (
        "SMA", "EquinoctialH", "EquinoctialK", "AltEquinoctialP", "AltEquinoctialQ", "MLONG",
    ),
    "ModifiedEquinoctial": (
        "SemilatusRectum", "ModEquinoctialF", "ModEquinoctialG", "ModEquinoctialH",
        "ModEquinoctialK", "TLONG",
    ),
    "Delaunay": ("Delaunayl", "Delaunayg", "Delaunayh", "DelaunayL", "DelaunayG", "DelaunayH"),
    "OutgoingAsymptote": (
        "OutgoingRadPer", "OutgoingC3Energy", "OutgoingRHA", "OutgoingDHA", "OutgoingBVAZI", "TA",
    ),
    "IncomingAsymptote": (
        "IncomingRadPer", "IncomingC3Energy", "IncomingRHA", "IncomingDHA", "IncomingBVAZI", "TA",
    ),
    "BrouwerMeanShort": (
        "BrouwerShortSMA", "BrouwerShortECC", "BrouwerShortINC", "BrouwerShortRAAN",
        "BrouwerShortAOP", "BrouwerShortMA",
    ),
    "BrouwerMeanLong": (
        "BrouwerLongSMA", "BrouwerLongECC", "BrouwerLongINC", "BrouwerLongRAAN",
        "BrouwerLongAOP", "BrouwerLongMA",
    ),
}
#: GMAT R2026a refuses each beside or instead of TA ("anomaly of type other than
#: True Anomaly not currently allowed").
_OTHER_ANOMALIES = frozenset({"MA", "EA", "HA"})
_ORBIT_STATE_FIELDS: frozenset[str] = _OTHER_ANOMALIES.union(*STATE_TYPE_FIELDS.values())
#: R2026a read Element1 and CartesianX into the state beside SC_Alien's Keplerian
#: six, 1.5e4 and 2.6e4 km away; their siblings are refused alike, unmeasured.
_STATE_ALIASES = frozenset(
    {f"Element{index}" for index in range(1, 7)}
    | {f"Cartesian{axis}" for axis in ("X", "Y", "Z", "VX", "VY", "VZ")}
    # CartesianX too (Spacecraft.cpp:4353, 4360); R2026a took `= 7000` for each.
    | {"CartesianState", "KeplerianState"}
)


def _in_process_state(
    name: str, fields: dict[str, str], coordinate_system: str, state_type: str, mu_km3_s2: float
) -> StateVector:
    if coordinate_system.lower() != "earthmj2000eq":
        raise ParseError(
            f"{name}.CoordinateSystem={coordinate_system!r}; the in-process reader only "
            f"accepts EarthMJ2000Eq{_NEEDS_CONSOLE}"
        )
    if state_type not in STATE_TYPE_FIELDS:
        # Case-sensitive: R2026a refused 'keplerian' ("Unknown state element representation").
        raise ParseError(
            f"{name}.DisplayStateType={state_type!r} is not one of GMAT's state types, "
            f"which are case-sensitive: {', '.join(STATE_TYPE_FIELDS)}"
        )
    if state_type not in ("Keplerian", "Cartesian"):
        raise ParseError(
            f"Unsupported {name}.DisplayStateType={state_type!r} for the in-process reader"
            f"{_NEEDS_CONSOLE}"
        )
    # The reader takes only its type's six fields; anything else it would ignore.
    given = _ORBIT_STATE_FIELDS & fields.keys()
    foreign = sorted(given - set(STATE_TYPE_FIELDS[state_type]))
    if foreign:
        owner = next(
            (other for other, names in STATE_TYPE_FIELDS.items() if given <= set(names)), None
        )
        if given & _OTHER_ANOMALIES:
            why = "GMAT R2026a refuses an anomaly other than TA"
        elif owner is None:
            why = "GMAT R2026a refuses fields of two state types before BeginMissionSequence"
        else:
            why = f"they are {owner} fields{_NEEDS_CONSOLE}"
        raise ParseError(
            f"{name} sets {', '.join(foreign)} beside DisplayStateType={state_type!r}; {why}"
        )
    if state_type == "Keplerian":
        elements = KeplerianElements(
            sma_km=_as_float(_required_field(fields, name, "SMA"), f"{name}.SMA"),
            ecc=_as_float(_required_field(fields, name, "ECC"), f"{name}.ECC"),
            inc_deg=_as_float(_required_field(fields, name, "INC"), f"{name}.INC"),
            raan_deg=_as_float(_required_field(fields, name, "RAAN"), f"{name}.RAAN"),
            aop_deg=_as_float(_required_field(fields, name, "AOP"), f"{name}.AOP"),
            ta_deg=_as_float(_required_field(fields, name, "TA"), f"{name}.TA"),
        )
        return keplerian_to_cartesian(elements, mu_km3_s2)
    return StateVector(
        (
            _as_float(_required_field(fields, name, "X"), f"{name}.X"),
            _as_float(_required_field(fields, name, "Y"), f"{name}.Y"),
            _as_float(_required_field(fields, name, "Z"), f"{name}.Z"),
        ),
        (
            _as_float(_required_field(fields, name, "VX"), f"{name}.VX"),
            _as_float(_required_field(fields, name, "VY"), f"{name}.VY"),
            _as_float(_required_field(fields, name, "VZ"), f"{name}.VZ"),
        ),
    )


def _binding(
    name: str,
    assignments: dict[str, dict[str, tuple[str, int]]],
    created: dict[str, list[str]],
) -> str:
    return spacecraft_binding_sha256(
        name,
        {item: _object_fields(assignments, item) for item in assignments},
        {item: kind for kind, items in created.items() for item in items},
    )


#: Epoch fields GMAT takes only after BeginMissionSequence; R2026a refuses each
#: before it ("The field name ... is not permitted").
_MISSION_EPOCH_FIELDS = frozenset(
    {"CurrA1MJD"}
    | {f"{scale}{form}" for scale in ("A1", "TAI", "UTC", "TT", "TDB")
       for form in ("ModJulian", "Gregorian")}
)


#: Each field the in-process reader knows, by its lower case. Delaunayl and
#: DelaunayL are both GMAT's, so a name may have two spellings.
_KNOWN_SPELLINGS: dict[str, frozenset[str]] = {}
for _spelled in sorted(
    _ORBIT_STATE_FIELDS | _STATE_ALIASES | _MISSION_EPOCH_FIELDS
    | {"AnomalyType", "DisplayStateType", "CoordinateSystem", "DateFormat", "Epoch"}
):
    _KNOWN_SPELLINGS[_spelled.lower()] = _KNOWN_SPELLINGS.get(
        _spelled.lower(), frozenset()
    ) | {_spelled}
del _spelled


def _refuse_what_only_gmat_reads(name: str, fields: dict[str, str], state_type: str) -> None:
    """Without a probe: what GMAT refuses, or reads otherwise than we would."""
    # R2026a refused 'ta', 'Sma', 'utcmodjulian', 'element1' and each governing
    # field in lower case ("The field name ... is not permitted"); we ignored them.
    cased = sorted(
        field for field in fields if field not in _KNOWN_SPELLINGS.get(field.lower(), {field})
    )
    if cased:
        raise ParseError(
            f"{name}.{cased[0]} is GMAT's "
            f"{' or '.join(sorted(_KNOWN_SPELLINGS[cased[0].lower()]))} in the wrong case; "
            "GmatConsole R2026a refused each such spelling tried"
        )
    # Measured on R2026a: AnomalyType = MA put the same six fields 5.6e5 km away.
    # With a probe the in-process comparison of the six still runs and warns of
    # it, as it does for the aliases.
    aliases = sorted(_STATE_ALIASES & fields.keys())
    if aliases:
        raise ParseError(
            f"{name} sets {aliases[0]}, which GMAT R2026a reads into the state beside "
            f"the {state_type} fields{_NEEDS_CONSOLE}"
        )
    anomaly_type = _strip_quotes(fields.get("AnomalyType", "TA"))
    if state_type == "Keplerian" and anomaly_type != "TA":
        raise ParseError(
            f"{name}.AnomalyType={anomaly_type!r}; GMAT reads the elements by it, the "
            f"in-process reader only with TA{_NEEDS_CONSOLE}"
        )
    mission_only = sorted(_MISSION_EPOCH_FIELDS & fields.keys())
    if mission_only:
        raise ParseError(
            f"{name}.{mission_only[0]} is set before BeginMissionSequence, which GmatConsole "
            "R2026a refuses; the epoch is DateFormat and Epoch there"
        )
    # GMAT also takes Epoch.<format> = ..., in that format whatever DateFormat says.
    scaled = sorted(field for field in fields if field.lower().startswith("epoch."))
    if scaled:
        raise ParseError(f"{name}.{scaled[0]} sets the epoch in a scale of its own{_NEEDS_CONSOLE}")


def _build_spacecraft(
    name: str,
    fields: dict[str, str],
    mu_km3_s2: float,
    probe: StateProbe | None = None,
    binding_sha256: str = "",
) -> SpacecraftSpec:
    coordinate_system = _strip_quotes(fields.get("CoordinateSystem", "EarthMJ2000Eq"))
    state_type = _strip_quotes(fields.get("DisplayStateType", "Cartesian"))
    probed = probe.craft_named(name) if probe is not None else None
    if probe is not None and probed is None:
        raise ParseError(f"The state probe did not report {name}")
    if probed is None:
        _refuse_what_only_gmat_reads(name, fields, state_type)
        epoch = _strip_quotes(_required_field(fields, name, "Epoch"))
        state = _in_process_state(name, fields, coordinate_system, state_type, mu_km3_s2)
        return SpacecraftSpec(name, epoch, coordinate_system, state_type, fields, state)
    if probed.binding_sha256 != binding_sha256:
        raise ParseError(
            f"The state probe was taken on a script that places {name} differently: its "
            "assignments, its CoordinateSystem or a built-in body's differ from this one's"
        )
    mismatch: tuple[float, float] | None = None
    try:
        own = _in_process_state(name, fields, coordinate_system, state_type, mu_km3_s2)
    except (ParseError, DynamicsError, ValueError):
        own = None  # GMAT read it; our reader cannot, so there is nothing to compare.
    if own is not None:
        mismatch = (
            float(np.linalg.norm(own.r - probed.state.r)),
            float(np.linalg.norm(own.v - probed.state.v)),
        )
    return SpacecraftSpec(
        name,
        _strip_quotes(fields.get("Epoch", "")),
        coordinate_system,
        state_type,
        fields,
        probed.state,
        probed_epoch=probed.epoch,
        in_process_mismatch=mismatch,
    )


def _assert_one_epoch_text(
    pursuer: SpacecraftSpec,
    target: SpacecraftSpec,
    assignments: dict[str, dict[str, tuple[str, int]]],
) -> None:
    """Without a probe, both craft must start on one epoch text in one scale."""
    if pursuer.epoch_text != target.epoch_text:
        raise ParseError(
            f"Pursuer and target epoch texts differ ({pursuer.epoch_text!r} and "
            f"{target.epoch_text!r}); whether they are one instant{_NEEDS_CONSOLE}"
        )
    # Each craft's Epoch is read in its own DateFormat, so one text in UTC and in
    # TAI is 37 s apart. TAIModJulian is GMAT's default when none is assigned.
    # GMAT reads Epoch in the format set when that line runs; R2026a put it 37 s
    # from what the text says. Epoch.<format> is refused in _build_spacecraft.
    for spec in (pursuer, target):
        lines = {field: line for field, (_value, line) in assignments[spec.name].items()}
        if "DateFormat" in lines and lines["Epoch"] < lines["DateFormat"]:
            raise ParseError(
                f"{spec.name}.Epoch on line {lines['Epoch'] + 1} is read before "
                f"{spec.name}.DateFormat is set on line {lines['DateFormat'] + 1}"
                f"{_NEEDS_CONSOLE}"
            )
    formats = [
        _strip_quotes(spec.fields.get("DateFormat", "TAIModJulian")) for spec in (pursuer, target)
    ]
    if formats[0] != formats[1]:
        raise ParseError(
            f"Pursuer and target epochs read the same text in different time scales "
            f"({formats[0]} and {formats[1]}){_NEEDS_CONSOLE}"
        )


_ALIGNMENT_DOC = "docs/ASYNC_START_AND_INSERTION.md §2"


class AlignedStatesNeeded(ParseError):
    """Δ > 0, and the probe does not yet say where the alignment leaves the craft.

    state_probe.load_scenario catches it and flies the alignment in GMAT.
    """

    def __init__(self, alignment: EpochAlignment, binding_sha256: str) -> None:
        super().__init__(
            f"{alignment.early} and {alignment.late} start {alignment.offset_s:.9g} s apart; "
            "t = 0 is where GMAT's own run of the alignment leaves them, which needs "
            "--gmat-console and --startup-file (docs/ASYNC_START_AND_INSERTION.md §3)"
        )
        self.alignment = alignment
        self.binding_sha256 = binding_sha256


def _probed_mjd(epoch: ProbedEpoch, scale: str) -> float:
    return float(getattr(epoch, f"{scale.lower()}_mjd"))


@dataclass(frozen=True, slots=True)
class _AligningLine:
    """A first mission command that flies one craft to an epoch goal."""

    line_index: int
    propagator: str
    craft: str
    goal: re.Match[str]


def _alignment_from_goal(
    aligning: _AligningLine,
    specs: dict[str, SpacecraftSpec],
    text: str,
) -> EpochAlignment:
    """Check an epoch-goal alignment against the probed epochs."""
    line = aligning.line_index + 1
    if aligning.craft not in specs:
        raise ParseError(
            f"Propagate line {line} aligns {aligning.craft!r}, which is not one of the "
            f"spacecraft {sorted(specs)}"
        )
    early = specs[aligning.craft]
    late_name = next(name for name in specs if name != aligning.craft)
    late = specs[late_name]
    goal = aligning.goal
    if goal.group("craft") != early.name:
        raise ParseError(
            f"Propagate line {line} stops on {goal.group('craft')}'s epoch while it "
            f"propagates {early.name}; an alignment's goal is on the craft it flies"
        )
    assert early.probed_epoch is not None and late.probed_epoch is not None
    offset_s = (late.probed_epoch.a1_mjd - early.probed_epoch.a1_mjd) * 86400.0
    scale = goal.group("scale")
    if goal.group("other") is not None:
        if goal.group("other") != late.name:
            raise ParseError(
                f"Propagate line {line} aligns {early.name} to {goal.group('other')!r}, "
                f"which is not the other spacecraft {late.name}"
            )
        if goal.group("other_scale") != scale:
            raise ParseError(
                f"Propagate line {line} compares {scale}ModJulian with "
                f"{goal.group('other_scale')}ModJulian; both sides must be one scale"
            )
        form: Literal["epoch-equality", "epoch-value"] = "epoch-equality"
    else:
        form = "epoch-value"
        value = float(goal.group("number"))
        wanted = _probed_mjd(late.probed_epoch, scale)
        gap_s = abs(value - wanted) * 86400.0
        # At one start the goal is the craft's own epoch; anything else is behind it.
        exact = offset_s != 0.0 or (
            value == wanted == _probed_mjd(early.probed_epoch, scale)
        )
        if gap_s > SYNCHRONOUS_EPOCH_TOLERANCE_S or not exact:
            raise ParseError(
                f"Propagate line {line} stops {early.name} at {scale}ModJulian "
                f"{goal.group('number')}, {gap_s:.9g} s from {late.name}'s probed epoch "
                f"{wanted!r}; an alignment must end there within "
                f"{SYNCHRONOUS_EPOCH_TOLERANCE_S:g} s, and exactly when the two start together"
            )
    return EpochAlignment(
        early=early.name,
        late=late.name,
        propagator=aligning.propagator,
        form=form,
        offset_s=offset_s,
        text=text,
        line_index=aligning.line_index,
    )


def _check_alignment_offset(alignment: EpochAlignment, specs: dict[str, SpacecraftSpec]) -> None:
    early, late = specs[alignment.early], specs[alignment.late]
    assert early.probed_epoch is not None and late.probed_epoch is not None
    where = (
        f"line {alignment.line_index + 1}" if alignment.line_index is not None else "the alignment"
    )
    apart = (
        f"(probed A1 epochs {early.probed_epoch.a1_mjd!r} and {late.probed_epoch.a1_mjd!r})"
    )
    if alignment.offset_s < 0.0:
        consequence = (
            "flying it on only takes the two further apart"
            if alignment.form == "elapsed"
            else "GMAT never reaches that goal and propagates until it is stopped"
        )
        raise ParseError(
            f"Mission sequence {where} flies {early.name} alone to {late.name}'s epoch, but "
            f"{early.name} starts {-alignment.offset_s:.9g} s later {apart}: {consequence}"
        )
    if 0.0 < alignment.offset_s <= SYNCHRONOUS_EPOCH_TOLERANCE_S:
        raise ParseError(
            f"Mission sequence {where} aligns {early.name} to {late.name}, which start "
            f"{alignment.offset_s:.3g} s apart {apart}: within the "
            f"{SYNCHRONOUS_EPOCH_TOLERANCE_S:g} s that is one start, yet not the same "
            "instant, so where the alignment ends is ambiguous"
        )


def _resolve_alignment(
    aligning: _AligningLine | None,
    segments: list[PropagateSegment],
    first_command_line: int | None,
    maneuver_lines: list[int],
    specs: dict[str, SpacecraftSpec],
    pursuer: SpacecraftSpec,
    target: SpacecraftSpec,
    document: GmatScriptDocument,
    config: OrbitWinConfig,
) -> tuple[EpochAlignment | None, list[PropagateSegment]]:
    """The template's epoch alignment, or the one to add, and the segments after it.

    Three forms are read as the first mission command: `{A.<E> = B.<E>}`,
    `{A.<E> = <number>}` and `{A.ElapsedSecs = <Δ>}`. The last is an ordinary
    single-craft segment until the probed epochs say the craft start apart.
    """
    probed = pursuer.probed_epoch is not None and target.probed_epoch is not None
    if aligning is not None and not probed:
        raise ParseError(
            f"Propagate line {aligning.line_index + 1} is an epoch alignment; whether it ends "
            f"where the other craft starts{_NEEDS_CONSOLE}"
        )
    if not probed:
        return None, segments
    assert pursuer.probed_epoch is not None and target.probed_epoch is not None
    apart_s = (target.probed_epoch.a1_mjd - pursuer.probed_epoch.a1_mjd) * 86400.0
    alignment: EpochAlignment | None = None
    if aligning is not None:
        alignment = _alignment_from_goal(aligning, specs, document.lines[aligning.line_index])
    elif (
        abs(apart_s) > SYNCHRONOUS_EPOCH_TOLERANCE_S
        and segments
        and segments[0].line_index == first_command_line
        and len(segments[0].spacecraft_names) == 1
    ):
        first = segments[0]
        if first.spacecraft_names[0] not in specs:
            raise ParseError(
                f"Propagate line {first.line_index + 1} flies {first.spacecraft_names[0]!r}, "
                f"which is not one of the spacecraft {sorted(specs)}"
            )
        early = specs[first.spacecraft_names[0]]
        late = target if early is pursuer else pursuer
        assert early.probed_epoch is not None and late.probed_epoch is not None
        offset_s = (late.probed_epoch.a1_mjd - early.probed_epoch.a1_mjd) * 86400.0
        if offset_s > 0.0 and abs(first.duration_s - offset_s) > SYNCHRONOUS_EPOCH_TOLERANCE_S:
            raise ParseError(
                f"Propagate line {first.line_index + 1} flies {early.name} alone for "
                f"{first.duration_s!r} s, but it starts {offset_s:.9g} s before {late.name}; "
                f"an alignment ends at the later epoch within {SYNCHRONOUS_EPOCH_TOLERANCE_S:g} s "
                f"({_ALIGNMENT_DOC})"
            )
        alignment = EpochAlignment(
            early=early.name,
            late=late.name,
            propagator=first.propagator_names[0],
            form="elapsed",
            offset_s=offset_s,
            text=document.lines[first.line_index],
            line_index=first.line_index,
        )
        segments = segments[1:]
    elif abs(apart_s) > SYNCHRONOUS_EPOCH_TOLERANCE_S:
        early, late = (pursuer, target) if apart_s > 0.0 else (target, pursuer)
        apart = (
            f"{pursuer.name} and {target.name} start {abs(apart_s):.9g} s apart (probed A1 "
            f"epochs {pursuer.probed_epoch.a1_mjd!r} and {target.probed_epoch.a1_mjd!r})"
        )
        if not config.policy.add_epoch_alignment:
            split = bool(segments) and len(segments[0].propagator_names) > 1
            consequence = (
                "GMAT runs a split Propagate with both ElapsedSecs equal while the epochs stay "
                "apart, which is not one instant"
                if split
                else "GMAT aborts a joint Propagate of craft on different epochs "
                "(\"Spacecraft epochs do not match\")"
            )
            raise ParseError(
                f"{apart}, and the mission sequence has no epoch alignment as its first "
                f"command: {consequence}. policy.add_epoch_alignment adds one; the ranking "
                "profiles set it, signed off 2026-11-03 in COMPETITION_POLICY.md pending the "
                f"organizer's answer to Q4 ({_ALIGNMENT_DOC})"
            )
        calls = (
            propagator_calls(document.lines[segments[0].line_index])
            if segments and segments[0].line_index == first_command_line
            else []
        )
        flying = [prop for prop, craft in calls if early.name in craft]
        if len(flying) != 1:
            raise ParseError(
                f"{apart}; policy.add_epoch_alignment adds one on the propagator the first "
                f"Propagate flies {early.name} with, and the first mission command has none"
            )
        newline = document.encoding.newline
        assert early.probed_epoch is not None and late.probed_epoch is not None
        alignment = EpochAlignment(
            early=early.name,
            late=late.name,
            propagator=flying[0],
            # A1, not UTC: the UTC equality form stops 1 s short of a late epoch
            # written inside a leap second.
            form="epoch-equality",
            offset_s=(late.probed_epoch.a1_mjd - early.probed_epoch.a1_mjd) * 86400.0,
            text=(
                f"Propagate {flying[0]}({early.name}) "
                f"{{{early.name}.A1ModJulian = {late.name}.A1ModJulian}};{newline}"
            ),
            line_index=None,
        )
    if alignment is None:
        return None, segments
    _check_alignment_offset(alignment, specs)
    after = sorted(
        [(segment.line_index, "propagate") for segment in segments]
        + [(line, "maneuver") for line in maneuver_lines]
    )
    if after and after[0][1] == "maneuver":
        # Measured on R2026a: rc 0 at Δ = 54000 s and at Δ = 0, and the one row
        # at t = 0 is the pre-burn state.
        segments = [
            PropagateSegment(
                line_index=after[0][0],
                duration_s=0.0,
                unit="ElapsedSecs",
                spacecraft_names=(pursuer.name, target.name),
                propagator_names=(),
                duration_spans=(),
                epoch_lock="instant",
                implicit=True,
            ),
            *segments,
        ]
    if alignment.offset_s > 0.0:
        pair = f"{alignment.early} and {alignment.late}"
        missing = [
            name
            for name, value in (
                ("policy.mission_time_origin", config.policy.mission_time_origin),
                ("policy.closest_approach_window", config.policy.closest_approach_window),
            )
            if value is None
        ]
        if missing:
            raise ParseError(
                f"{pair} start {alignment.offset_s:.9g} s apart; set "
                f"{' and '.join(missing)}. The ranking rules' origin is 'earliest', by "
                f"{ORGANIZER_TIME_ORIGIN_ANSWER}; whether the alignment's rows count toward "
                "d_min is still asked of the organizer (docs/ASYNC_START_AND_INSERTION.md §3)"
            )
    return alignment, segments


def _is_literal_initial_value(line: str, literal_names: dict[str, str]) -> bool:
    """Whether a resource line only gives a declared Variable, Array element or
    String a literal value. `Pursuer = AlienSC;` copies one craft onto the other."""
    match = _LITERAL_INIT_RE.match(_code_part(line))
    if match is None:
        return False
    kind = literal_names.get(match.group("name"))
    numeric = match.group("number") is not None
    if match.group("index") is not None:
        return kind == "array" and numeric
    return (kind == "variable" and numeric) or (kind == "string" and not numeric)


#: Mission commands that drive subscribers only: they change no flight and no report.
PLOT_COMMANDS = ("Toggle", "PenUp", "PenDown", "MarkPoint", "ClearPlot")
#: Each as R2026a's GUI builds it (tests/fixtures/console_copy): names apart by
#: blanks, and for a Toggle then On or Off as written. A comma, a lower-case
#: command, `off`, no name or no state it refuses; a missing `;` it takes, and so
#: is refused here only.
_PLOT_COMMAND_RE = re.compile(
    r"[ \t]*(?P<command>Toggle|PenUp|PenDown|MarkPoint|ClearPlot)[ \t]+"
    r"(?P<names>[A-Za-z_]\w*(?:[ \t]+[A-Za-z_]\w*)*)"
    r"[ \t]*(?:;[ \t]*)+"
)
_PLOT_COMMAND_WORD_RE = re.compile(
    r"\s*(?:" + "|".join(PLOT_COMMANDS) + r")\b", re.IGNORECASE
)


#: The plots that are subscribers. An OpenFramesView, Vector or SensorMask is
#: not one, and the GUI refuses a Toggle or PenUp of it.
TOGGLED_PLOT_TYPES = PLOT_TYPES - {"OpenFramesView", "OpenFramesVector", "OpenFramesSensorMask"}
#: What each command may name. R2026a's Sandbox refuses a MarkPoint or ClearPlot of
#: anything but an XYPlot ("should be an XYPlot to use the MarkPoint command"), on a
#: run to the end; PenUp, PenDown and Toggle of a GroundTrack or OpenFrames view ran.
COMMAND_PLOT_TYPES = {
    "Toggle": TOGGLED_PLOT_TYPES, "PenUp": TOGGLED_PLOT_TYPES, "PenDown": TOGGLED_PLOT_TYPES,
    "MarkPoint": frozenset({"XYPlot"}), "ClearPlot": frozenset({"XYPlot"}),
}


def plot_command(code: str, document: GmatScriptDocument) -> tuple[str, ...] | None:
    """The plots a mission line's Toggle, PenUp, PenDown, MarkPoint or ClearPlot
    names, or None when it is none of them.

    Of a plot each changes no report and no flight, so it is read; of anything
    else, a ReportFile above all, it may change what the uploaded report holds,
    and it is refused, as is any other form.
    """
    word = _PLOT_COMMAND_WORD_RE.match(code)
    if word is None:
        return None
    command = word.group().strip()
    match = _PLOT_COMMAND_RE.fullmatch(code.rstrip("\r\n"))
    names = tuple(split_create_names(match["names"])) if match else ()
    if match and match["command"] == "Toggle":
        names = names[:-1] if names[-1:] in (("On",), ("Off",)) else ()
    shown = repr(code.strip()[:80])
    if not names:
        form = (" On;` or `Off;`" if command.lower() == "toggle" else ";`")
        raise ParseError(
            f"{command} {shown} is not one the parser reads: `{command} <plot> [<plot> ...]"
            f"{form}, the names apart by blanks{STOP_ADVICE}"
        )
    types = created_types(document)
    assert match is not None
    allowed = COMMAND_PLOT_TYPES[match["command"]]
    other = [name for name in names if types.get(name) not in allowed]
    if other:
        what = ", ".join(f"{name} ({types.get(name, 'nothing the script creates')})"
                         for name in other)
        plots = "XYPlots" if allowed == {"XYPlot"} else "plots"
        raise ParseError(
            f"{command} {shown} names {what}: only a {command} of {plots} is read; one of "
            "anything else can change what the uploaded report holds or stop the run"
        )
    return names


def split_create_names(names: str) -> list[str]:
    """The object names in one ``Create`` command.

    GMAT R2026a accepts both `Create Type a, b;` and `Create Type a b;` --
    verified by running each. Splitting on whitespace alone left the comma
    attached, so `Create ImpulsiveBurn Burn1, Burn2;` declared an object called
    "Burn1," and the script was then refused for maneuvering an undeclared
    Burn1; it also walked past the reserved-name check, which had only ever
    seen the comma-suffixed spelling.

    Splitting on commas alone leaves the *space* attached, which is what the
    renderer's policy guard did. The two halves of the same fix disagreed, so
    a space-separated declaration parsed here and was refused there -- and told
    the reader the original "already maneuvers" a burn it never maneuvers.
    One function now, so they cannot drift apart again.
    """
    return [part for part in re.split(r"[,\s]+", names.strip()) if part]


def _describe_the_choice(
    spacecraft_names: list[str],
    created: dict[str, list[str]],
    assignments: dict[str, dict[str, tuple[str, int]]],
    config: OrbitWinConfig,
    probe: StateProbe | None,
) -> str:
    """What the operator needs to name the pursuer, at the moment they need it.

    This is the first command run on competition day, against a file nobody has
    seen, and it stops with a question. Answering it wrongly is not a small
    error -- solve for the wrong spacecraft and every number afterwards is
    confidently about the other one -- so the question comes with the evidence.
    """
    tracked: set[str] = set()
    for report_name in created.get("reportfile", []):
        added = assignments.get(report_name, {}).get("Add")
        if not added:
            continue
        for name in spacecraft_names:
            if re.search(rf"\b{re.escape(name)}\b", added[0]):
                tracked.add(name)

    lines: list[str] = []
    for name in sorted(spacecraft_names):
        try:
            spacecraft = _build_spacecraft(
                name,
                _object_fields(assignments, name),
                config.policy.mu_km3_s2,
                probe,
                _binding(name, assignments, created),
            )
            elements = cartesian_to_keplerian(spacecraft.initial_state, config.policy.mu_km3_s2)
            shape = (
                f"SMA {elements.sma_km:.3f} km, ECC {elements.ecc:.6g}, "
                f"INC {elements.inc_deg:.6g} deg"
            )
            if elements.ecc <= config.policy.target_circular_tolerance:
                shape += " -- circular"
        except (ParseError, DynamicsError, ValueError):
            shape = "its orbit could not be read"
        if name in tracked:
            shape += "; the official ReportFile tracks it"
        lines.append(f"  {name}: {shape}")
    return "\n".join(lines)


#: ImpulsiveBurn fields that leave the flown impulse as Element1-3 in the frame
#: read below (R2026a Burn.cpp:63-73, ImpulsiveBurn.cpp:55-61). The mass fields
#: only feed DecrementMass(), which Fire() runs after adding the impulse
#: (ImpulsiveBurn.cpp:251-371); VectorFormat is stored and never read
#: (Burn.cpp:671, 768). Measured on R2026a: Isp, GravitationalAccel,
#: DeltaTankMass and VectorFormat left the flown impulse as written to 2e-16 km/s.
BURN_FIELDS = frozenset(
    {
        "CoordinateSystem",
        "Origin",
        "Axes",
        "Element1",
        "Element2",
        "Element3",
        "DecrementMass",
        "Tank",
        "Isp",
        "GravitationalAccel",
        "DeltaTankMass",
        "VectorFormat",
    }
)
#: R2026a's deprecated aliases for Element1-3 (Burn.cpp:433-467).
BURN_ALIASES = {"V": "Element1", "N": "Element2", "B": "Element3"}
#: The rest of R2026a's ImpulsiveBurn fields: it refuses SpacecraftName when set.
_BURN_FIELDS_UNMODELLED = frozenset({"SpacecraftName", "Covariance"})


def _refuse_unmodelled_burn_fields(
    assignments: dict[str, dict[str, tuple[str, int]]], burn_name: str, *, aliases: bool
) -> None:
    """Refuse a field that could make GMAT fly another impulse than Element1-3 say.

    E2E audit A4: `ImpulsiveBurn1.V = 0;` after the Element lines. GMAT flew
    70.69 m/s; the render, verify-policy and the certificate all claimed 91.15.
    ``aliases`` only for a burn that becomes a slot: GMAT runs a spare one's V/N/B.
    """
    for field, (_value, line_index) in sorted(
        assignments.get(burn_name, {}).items(), key=lambda item: item[1][1]
    ):
        if field in BURN_FIELDS or (field in BURN_ALIASES and not aliases):
            continue
        if field in BURN_ALIASES:
            raise ParseError(
                f"Line {line_index + 1} sets {burn_name}.{field}. GMAT reads {field} as "
                f"{BURN_ALIASES[field]} and flies whichever of the two is assigned last, "
                f"but only {BURN_ALIASES[field]} is read and rewritten here, so the impulse "
                "flown could differ from the one claimed"
            )
        if field == "SpacecraftName":
            gmat = "GmatConsole R2026a refuses the script"
        elif field in _BURN_FIELDS_UNMODELLED:
            gmat = "R2026a defines it and nothing here models it"
        else:
            gmat = "GmatConsole R2026a refuses the script (the field name is not permitted)"
        raise ParseError(
            f"Line {line_index + 1} sets {burn_name}.{field}, which is not an ImpulsiveBurn "
            f"field modelled here; {gmat}. Modelled: {', '.join(sorted(BURN_FIELDS))}"
        )


def _slot_from_resource(
    assignments: dict[str, dict[str, tuple[str, int]]],
    burn_name: str,
    pursuer_name: str,
    maneuver_line: int | None,
) -> BurnSlot:
    """Read one declared ImpulsiveBurn, whether or not the mission uses it."""
    _refuse_unmodelled_burn_fields(assignments, burn_name, aliases=True)
    fields = {field: value for field, (value, _line) in assignments.get(burn_name, {}).items()}
    line_indices: list[int] = []
    values: list[float] = []
    for element in ("Element1", "Element2", "Element3"):
        if element not in assignments.get(burn_name, {}):
            raise ParseError(f"{burn_name}.{element} must be explicitly assigned")
        raw_value, line_index = assignments[burn_name][element]
        values.append(_as_float(raw_value, f"{burn_name}.{element}"))
        line_indices.append(line_index)
    return BurnSlot(
        name=burn_name,
        spacecraft=pursuer_name,
        coordinate_system=_strip_quotes(fields.get("CoordinateSystem", "Local")),
        origin=_strip_quotes(fields.get("Origin", "Earth")),
        axes=_strip_quotes(fields.get("Axes", "VNB")),
        element1_km_s=values[0],
        element2_km_s=values[1],
        element3_km_s=values[2],
        assignment_lines=(line_indices[0], line_indices[1], line_indices[2]),
        maneuver_line=maneuver_line,
    )


#: Allowed, and kept in the probe copy: they place a craft or a craft names them.
PROBE_KEPT_TYPES = frozenset(
    {
        "Spacecraft",
        "CoordinateSystem",
        "Variable",
        "Array",
        "String",
        # Legacy names R2026a still runs, with a deprecation warning.
        "FuelTank",
        "Thruster",
        "ChemicalTank",
        "ElectricTank",
        "ChemicalThruster",
        "ElectricThruster",
        "SolarPowerSystem",
        "NuclearPowerSystem",
        "Plate",
        "GroundStation",
        "Barycenter",
        "LibrationPoint",
    }
)
#: Allowed, and commented out of the probe copy.
PROBE_DROPPED_TYPES = PLOT_TYPES | frozenset(
    {
        "ForceModel",
        "Propagator",
        "ImpulsiveBurn",
        "FiniteBurn",
        "Formation",
        "ReportFile",
        "EphemerisFile",
        "DifferentialCorrector",
    }
)
ALLOWED_RESOURCE_TYPES = PROBE_KEPT_TYPES | PROBE_DROPPED_TYPES


#: What a craft carries; kept in the probe copy, and kept wherever a craft names it.
HARDWARE_TYPES = frozenset(
    {
        "FuelTank",
        "Thruster",
        "ChemicalTank",
        "ElectricTank",
        "ChemicalThruster",
        "ElectricThruster",
        "SolarPowerSystem",
        "NuclearPowerSystem",
    }
)
#: Fields by which a craft or its hardware names other resources.
REFERENCE_FIELDS = frozenset({"Tanks", "Thrusters", "PowerSystem", "AddHardware", "Tank"})
#: An input-file field, as gmat_runner hashes them.
_FILE_FIELD_NAME_RE = re.compile(r"[A-Za-z_]\w*(?:File|FileName|Filename|Kernel|Kernels|KernelName)")


def _resource_assignments(document: GmatScriptDocument) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for line in document.lines:
        code = _code_part(line)
        if _BEGIN_MISSION_RE.match(code):
            break
        match = _ASSIGN_RE.match(code)
        if match:
            found.setdefault(match.group("object"), {})[match.group("field")] = match.group("value")
    return found


def spacecraft_fields(document: GmatScriptDocument) -> dict[str, dict[str, str]]:
    """Each Spacecraft's fields as the script assigns them, values as written."""
    assignments = _resource_assignments(document)
    return {name: dict(assignments.get(name, {})) for name, kind in created_types(document).items()
            if kind.casefold() == "spacecraft"}


def spacecraft_field_lines(document: GmatScriptDocument) -> dict[str, dict[str, int]]:
    """The line index of each Spacecraft field's last assignment, as spacecraft_fields
    reads it."""
    craft = {name for name, kind in created_types(document).items()
             if kind.casefold() == "spacecraft"}
    found: dict[str, dict[str, int]] = {}
    for index, line in enumerate(document.lines):
        code = _code_part(line)
        if _BEGIN_MISSION_RE.match(code):
            break
        match = _ASSIGN_RE.match(code)
        if match and match.group("object") in craft:
            found.setdefault(match.group("object"), {})[match.group("field")] = index
    return found


def referenced_resources(document: GmatScriptDocument) -> set[str]:
    """The hardware a Spacecraft or kept hardware names, followed through (thruster to tank).

    GMAT refuses a craft whose hardware is missing: R2026a, "Nonexistent
    Hardware" for a tank commented out of a private copy. Hardware is a kept
    hardware type or a type admitted by policy.extra_resource_types; an allowed
    type that is not hardware (an EphemerisFile in AddHardware) stays dropped.
    """
    types = created_types(document)
    assignments = _resource_assignments(document)
    frontier = [name for name, kind in types.items() if kind in PROBE_KEPT_TYPES]
    found: set[str] = set()
    while frontier:
        for field, value in assignments.get(frontier.pop(), {}).items():
            if field.rsplit(".", 1)[-1] not in REFERENCE_FIELDS:
                continue
            for name in re.findall(r"[A-Za-z_]\w*", value):
                kind = types.get(name)
                hardware = kind in HARDWARE_TYPES or (
                    kind is not None and kind not in ALLOWED_RESOURCE_TYPES
                )
                if hardware and name not in found:
                    found.add(name)
                    frontier.append(name)
    return found


def names_with_file_fields(document: GmatScriptDocument) -> set[str]:
    """Resources assigned a field that names an input file."""
    return {
        name
        for name, fields in _resource_assignments(document).items()
        if any(_FILE_FIELD_NAME_RE.fullmatch(field.rsplit(".", 1)[-1]) for field in fields)
    }


def created_at(document: GmatScriptDocument) -> dict[str, int]:
    """The line index of each resource's first Create, before BeginMissionSequence."""
    found: dict[str, int] = {}
    for index, line in enumerate(document.lines):
        code = _code_part(line)
        if _BEGIN_MISSION_RE.match(code):
            break
        match = _CREATE_RE.match(code)
        if match:
            for name in split_create_names(match.group("names")):
                found.setdefault(name, index)
    return found


def created_types(document: GmatScriptDocument) -> dict[str, str]:
    """Each resource name the script creates, with its type as written."""
    found: dict[str, str] = {}
    for line in document.lines:
        code = _code_part(line)
        if _BEGIN_MISSION_RE.match(code):
            break
        match = _CREATE_RE.match(code)
        if match:
            # GMAT keeps the first Create of a name and ignores a repeat.
            for name in split_create_names(match.group("names")):
                found.setdefault(name, match.group("type"))
    return found


def assert_resources_supported(
    document: GmatScriptDocument, extra_types: Iterable[str] = ()
) -> None:
    """Refuse what must never reach GMAT, before anything runs it.

    The state probe launches GmatConsole before the full parse, so everything
    that could make GMAT touch a file the parser cannot see is checked here:
    Include, a resource line the parser cannot read, a
    resource type outside ALLOWED_RESOURCE_TYPES and ``extra_types``, and an
    output file outside the run directory.
    """
    if re.search(r"^\s*#?Include\b", document.text, re.IGNORECASE | re.MULTILINE):
        raise ParseError("Include macros are rejected because immutable content could be hidden")
    allowed = ALLOWED_RESOURCE_TYPES | (frozenset(extra_types) - REFUSED_RESOURCE_TYPES)
    types: dict[str, str] = {}
    literal_names: dict[str, str] = {}
    for line_index, line_with_newline in enumerate(document.lines):
        line = line_with_newline.rstrip("\r\n")
        code = _code_part(line)
        if _BEGIN_MISSION_RE.match(code):
            return
        command = code.strip()
        if not command:
            continue
        create_match = _CREATE_RE.match(code)
        if create_match:
            object_type = create_match.group("type")
            names = split_create_names(create_match.group("names"))
            if not names:
                raise ParseError(f"Create command on line {line_index + 1} has no object name")
            if object_type not in allowed:
                refused = object_type in REFUSED_RESOURCE_TYPES
                raise ParseError(
                    f"Line {line_index + 1} creates {object_type} {names[0]}, which is not run: "
                    + (
                        "it writes files outside the run or runs code"
                        if refused
                        else "only these types are accepted, "
                        + ", ".join(sorted(allowed))
                        + "; add it to policy.extra_resource_types once a person has checked "
                        "that it neither writes nor runs anything"
                    )
                )
            for name in names:
                if name in GMAT_RESERVED_RESOURCE_NAMES:
                    raise ParseError(
                        f"Line {line_index + 1} creates {object_type} \"{name}\", but GMAT "
                        "reserves that identifier for a mission-sequence command and will "
                        "refuse to interpret the script"
                    )
            for name in names:
                types.setdefault(name, object_type)
            if object_type.lower() in _LITERAL_TYPES:
                # Array dimensions are dropped: `Create Array A[3,1] B[2,2]`.
                bare = re.sub(r"\[[^\]]*\]", " ", create_match.group("names"))
                for name in re.findall(r"[A-Za-z_]\w*", bare):
                    literal_names[name] = object_type.lower()
            continue
        assign_match = _ASSIGN_RE.match(code)
        if assign_match:
            owner, field = assign_match.group("object"), assign_match.group("field")
            if field in OUTPUT_FILE_FIELDS.get(types.get(owner, ""), ()):
                value = _strip_quotes(assign_match.group("value"))
                if not is_bare_file_name(value):
                    raise ParseError(
                        f"Line {line_index + 1}: {owner}.{field} = {value!r} is not a bare file "
                        "name, so it could be written outside the run directory (Unsafe output "
                        "filename)"
                    )
            continue
        # A resource line nothing below reads was skipped, while GMAT may run
        # it: `AlienSC.DateFormat = UTCModJulian` without its semicolon put
        # that craft 37 s from the other and the scale check never saw it.
        if not _is_literal_initial_value(line, literal_names):
            shown = repr(line.strip()[:80])
            hint = "" if command.endswith(";") else "; GMAT allows no semicolon, this parser does not"
            raise ParseError(
                f"Line {line_index + 1} is neither a Create nor an assignment the parser "
                f"can read: {shown}{hint}"
            )


#: The only command a render writes before an alignment, then a Maneuver.
_HEAD_START_RE = re.compile(
    r"^Propagate (?P<propagator>[A-Za-z_]\w*)\((?P<craft>[A-Za-z_]\w*)\) "
    r"\{(?P=craft)\.ElapsedSecs = (?P<value>[-+0-9.eE]+)\};$"
)


def _head_start_shape(segments: list[PropagateSegment], maneuver_lines: list[int]) -> bool:
    """Propagate, Maneuver, ... Maneuver, each Propagate flying one craft."""
    kinds = sorted([(segment.line_index, "P") for segment in segments]
                   + [(line, "M") for line in maneuver_lines])
    return (
        bool(segments)
        and "".join(kind for _, kind in kinds) == "PM" * len(segments)
        and all(len(segment.spacecraft_names) == 1 for segment in segments)
    )


def _head_start_lead(
    alignment: EpochAlignment | None,
    pursuer: str,
    target: str,
    origin: str | None,
    config: OrbitWinConfig,
) -> tuple[float, str | None]:
    """The alignment's length when policy.head_start_burns opens it, and why not."""
    if not config.policy.head_start_burns:
        return 0.0, None
    if alignment is None or alignment.offset_s <= 0.0:
        return 0.0, "the craft start together, so there is no head start"
    if alignment.early != pursuer:
        return 0.0, f"{target} starts first, so the head start is not {pursuer}'s"
    if alignment.form == "elapsed":
        return 0.0, (
            "the alignment stops on an ElapsedSecs goal, which a burn before it would carry "
            "past the later epoch"
        )
    if origin != pursuer:
        return 0.0, f"T_team does not count from {pursuer}'s epoch"
    return alignment.offset_s, None


def _check_head_start(
    segments: list[PropagateSegment],
    document: GmatScriptDocument,
    alignment: EpochAlignment | None,
    pursuer: str,
    lead_s: float,
    note: str | None,
    config: OrbitWinConfig,
) -> None:
    """Head-start pairs: the pursuer alone, on the alignment's propagator, each
    segment at least min_propagate_duration_s, the last burn that much before
    the later epoch (COMPETITION_POLICY.md, head-start burns)."""
    first = segments[0].line_index + 1
    if not config.policy.head_start_burns:
        raise ParseError(
            f"Line {first} starts burns before the epoch alignment, and "
            "policy.head_start_burns is off"
        )
    if lead_s <= 0.0 or alignment is None:
        raise ParseError(f"Line {first} starts burns before the epoch alignment, but {note}")
    floor_s = config.policy.min_propagate_duration_s
    flown_s = 0.0
    for segment in segments:
        line = document.lines[segment.line_index].rstrip("\r\n")
        match = _HEAD_START_RE.match(line)
        if (
            match is None
            or match.group("craft") != pursuer
            or match.group("propagator") != alignment.propagator
            or segment.spacecraft_names != (pursuer,)
        ):
            raise ParseError(
                f"Line {segment.line_index + 1}, before the epoch alignment, is not "
                f"`Propagate {alignment.propagator}({pursuer}) {{{pursuer}.ElapsedSecs = <s>}};`: "
                f"{line.strip()[:80]!r}"
            )
        if not segment.duration_s >= floor_s:
            raise ParseError(
                f"Line {segment.line_index + 1} lasts {segment.duration_s!r} s, under the "
                f"{floor_s!r} s floor; a zero-length goal hangs GMAT"
            )
        flown_s += segment.duration_s
    if not lead_s - flown_s >= floor_s:
        raise ParseError(
            f"The burns before the epoch alignment end {flown_s!r} s after {pursuer}'s epoch, "
            f"not {floor_s!r} s before the later epoch {lead_s!r} s on: each must come "
            "strictly before it"
        )


def parse_gmat_script(
    path: str | Path, config: OrbitWinConfig, *, probe: StateProbe | None = None
) -> tuple[Scenario, GmatScriptDocument]:
    """Parse an organizer script, fail closed.

    With ``probe`` (state_probe.probe_initial_states on this script or one that
    keeps its craft assignments) the initial states and epochs are GMAT's;
    without one they are our own reading of EarthMJ2000Eq Keplerian/Cartesian.
    """
    document = GmatScriptDocument.read(path)
    assert_resources_supported(document, config.policy.extra_resource_types)

    created: dict[str, list[str]] = {}
    assignments: dict[str, dict[str, tuple[str, int]]] = {}
    begin_mission_index: int | None = None
    maneuvers: list[tuple[int, str, str]] = []
    propagate_segments: list[PropagateSegment] = []
    aligning: _AligningLine | None = None
    first_command_line: int | None = None
    #: Before an alignment that is not the first command: pursuer-alone pairs.
    head_start: list[PropagateSegment] = []
    head_start_lines: set[int] = set()

    for line_index, line_with_newline in enumerate(document.lines):
        line = line_with_newline.rstrip("\r\n")
        code = _code_part(line)
        command = code.strip()
        shown = repr(line.strip()[:80])
        if not command or _BEGIN_MISSION_RE.match(code) or begin_mission_index is None:
            pass  # resource lines: assert_resources_supported
        elif plot_command(code, document) is not None:
            continue  # a plot's, so no flight or report changes; it is no first command
        elif not (_MANEUVER_RE.match(code) or _PROPAGATE_RE.match(code)):
            if re.match(r"Maneuver\b", command, re.IGNORECASE):
                raise ParseError(
                    f"Line {line_index + 1} is a Maneuver the parser cannot read: {shown}; "
                    "it reads `Maneuver Burn(Spacecraft);`"
                )
            # Skipping these used to be silent. Report, or a Toggle of a
            # ReportFile (plot_command), changes what the uploaded report holds, an assignment was read as an initial
            # state, and a loop or If changes the flight; none is modelled.
            raise ParseError(
                f"Line {line_index + 1} of the mission sequence is neither a Propagate nor a "
                f"Maneuver: {shown}. Nothing else there is modelled, so the script cannot be "
                "solved or rewritten safely"
            )
        create_match = _CREATE_RE.match(code)
        if create_match:
            object_type = create_match.group("type")
            names = split_create_names(create_match.group("names"))
            if not names:
                raise ParseError(f"Create command on line {line_index + 1} has no object name")
            created.setdefault(object_type.lower(), []).extend(names)
            continue
        assignment_match = _ASSIGN_RE.match(code)
        if assignment_match:
            object_name = assignment_match.group("object")
            field_name = assignment_match.group("field")
            object_map = assignments.setdefault(object_name, {})
            if field_name in object_map:
                raise ParseError(
                    f"Duplicate assignment for {object_name}.{field_name} on lines "
                    f"{object_map[field_name][1] + 1} and {line_index + 1}"
                )
            object_map[field_name] = (assignment_match.group("value"), line_index)
        if _BEGIN_MISSION_RE.match(code):
            if begin_mission_index is not None:
                raise ParseError("Multiple BeginMissionSequence commands are not supported")
            begin_mission_index = line_index
            continue
        if begin_mission_index is not None and line_index > begin_mission_index:
            if command and first_command_line is None:
                first_command_line = line_index
            maneuver_match = _MANEUVER_RE.match(code)
            if maneuver_match:
                maneuvers.append(
                    (line_index, maneuver_match.group("burn"), maneuver_match.group("spacecraft"))
                )
                continue
            if _PROPAGATE_RE.match(code):
                scan = _propagate_scan_text(line)
                items = [
                    item.strip()
                    for group in _STOP_GROUP_BODY_RE.finditer(scan)
                    for item in group.group("body").split(",")
                ]
                epoch_goals = [item for item in items if _EPOCH_GOAL_RE.fullmatch(item)]
                if epoch_goals:
                    if first_command_line != line_index:
                        if aligning is not None or not _head_start_shape(
                            propagate_segments, [line for line, _, _ in maneuvers]
                        ):
                            raise ParseError(
                                f"Propagate line {line_index + 1} stops on {epoch_goals[0]!r}; "
                                "an epoch alignment is read only as the first mission command, "
                                "or after head-start burns: one-craft Propagate and Maneuver "
                                f"pairs ({_ALIGNMENT_DOC}; COMPETITION_POLICY.md)"
                            )
                        head_start, propagate_segments = propagate_segments, []
                        head_start_lines = {line for line, _, _ in maneuvers}
                    stepped, propagators, _ = _propagated_spacecraft(scan, line_index)
                    others = [
                        item
                        for item in items
                        if item not in epoch_goals and not _STOP_OPTION_RE.fullmatch(item)
                    ]
                    if len(epoch_goals) != 1 or others or len(stepped) != 1:
                        raise ParseError(
                            f"Propagate line {line_index + 1} is read as an epoch alignment "
                            "only when it flies one craft to one epoch goal and nothing else; "
                            f"it flies {list(stepped)} on {items}"
                        )
                    goal = _EPOCH_GOAL_RE.fullmatch(epoch_goals[0])
                    assert goal is not None
                    aligning = _AligningLine(line_index, propagators[0], stepped[0], goal)
                    continue
                for group in _STOP_GROUP_BODY_RE.finditer(scan):
                    for item in group.group("body").split(","):
                        if not (
                            _ELAPSED_RE.fullmatch(item.strip())
                            or _STOP_OPTION_RE.fullmatch(item.strip())
                        ):
                            # GMAT stops at the first condition met, so the goal is not the length.
                            raise ParseError(
                                f"Propagate line {line_index + 1} stops on {item.strip()!r}, "
                                "which is not an elapsed-time goal; GMAT ends the segment at "
                                "whichever condition comes first, so its length cannot be read"
                            )
                matches = list(_ELAPSED_RE.finditer(scan))
                if not matches:
                    raise ParseError(
                        f"Propagate line {line_index + 1} has no explicit ElapsedSecs/ElapsedDays goal"
                    )
                units = {match.group("unit") for match in matches}
                if len(units) != 1:
                    raise ParseError(f"Mixed elapsed-time units on Propagate line {line_index + 1}")
                unit = next(iter(units))
                raw_values = [float(match.group("value")) for match in matches]
                duration_s_values = [value * (86400.0 if unit.lower() == "elapseddays" else 1.0) for value in raw_values]
                if max(duration_s_values) - min(duration_s_values) > 1.0e-8:
                    raise ParseError(
                        f"Spacecraft do not share the same elapsed-time goal on line {line_index + 1}"
                    )
                spans = tuple(match.span("value") for match in matches)
                stepped, propagators, epoch_lock = _propagated_spacecraft(scan, line_index)
                goal_owners = {match.group("spacecraft") for match in matches}
                strangers = goal_owners - set(stepped)
                if strangers:
                    raise ParseError(
                        f"Propagate line {line_index + 1} stops on {sorted(strangers)}, which "
                        f"it does not propagate ({list(stepped)})"
                    )
                if epoch_lock is None and goal_owners == set(stepped):
                    # Holds only while the steppers stay in lockstep. R2026a: the organizer's
                    # workshop template's split Propagate ended 0 s apart (MaxStep 1 s), 167 s
                    # apart at 60 s steps.
                    epoch_lock = "equal-goals"
                propagate_segments.append(
                    PropagateSegment(
                        line_index=line_index,
                        duration_s=duration_s_values[0],
                        unit="ElapsedDays" if unit.lower() == "elapseddays" else "ElapsedSecs",
                        spacecraft_names=stepped,
                        propagator_names=propagators,
                        duration_spans=spans,
                        epoch_lock=epoch_lock,
                    )
                )

    if begin_mission_index is None:
        raise ParseError("BeginMissionSequence was not found")
    declared_propagators = set(created.get("propagator", []))
    for line_index, used in [
        *((segment.line_index, segment.propagator_names) for segment in propagate_segments),
        *([(aligning.line_index, (aligning.propagator,))] if aligning is not None else []),
    ]:
        unknown = sorted(set(used) - declared_propagators)
        if unknown:
            raise ParseError(
                f"Propagate line {line_index + 1} uses {unknown}, which the script does not "
                "create as a Propagator"
            )
    spacecraft_names = created.get("spacecraft", [])
    if len(spacecraft_names) != 2:
        raise ParseError(f"Expected exactly two Spacecraft resources, found {spacecraft_names}")
    # A mission sequence with no Maneuver command is not a broken script. The
    # rules put the maneuvers there: 初賽規則 §2 says B 可進行多次瞬時脈衝機動 and
    # 說明會簡報 p.8 says 由參賽者對 Spacecraft B 進行機動. A template that
    # declares none is one where every maneuver is still to be written, which
    # is a different job from filling in slots but the same competition.
    burn_names = created.get("impulsiveburn", [])
    for burn_name in burn_names:
        _refuse_unmodelled_burn_fields(assignments, burn_name, aliases=False)

    maneuver_spacecraft = {spacecraft for _, _, spacecraft in maneuvers}
    if config.pursuer_name:
        pursuer_name = config.pursuer_name
        if pursuer_name not in spacecraft_names:
            raise ParseError(f"Configured pursuer {pursuer_name!r} is not a Spacecraft")
        if any(spacecraft != pursuer_name for _, _, spacecraft in maneuvers):
            raise ParseError("A non-pursuer spacecraft has a Maneuver command")
    elif maneuvers:
        if len(maneuver_spacecraft) != 1:
            raise ParseError(f"Cannot infer a unique pursuer from Maneuver commands: {maneuver_spacecraft}")
        pursuer_name = next(iter(maneuver_spacecraft))
    else:
        # Which spacecraft is which is read off the Maneuver commands, and a
        # template without them says nothing about it. Guessing would be
        # guessing at the whole problem -- solve for the wrong one and every
        # answer is confidently wrong -- so it is configured or it stops here.
        raise ParseError(
            "The mission sequence contains no Maneuver command, so nothing in the script says "
            "which spacecraft the contestant maneuvers. Set pursuer_name (and optionally "
            "target_name) in the config. What the script does say:\n"
            + _describe_the_choice(spacecraft_names, created, assignments, config, probe)
            + "\nThe briefing has Spacecraft A circular in the preliminary, and in the "
            "organizer's own example the ReportFile tracks the spacecraft that maneuvers. "
            "Choosing backwards tends to show up on the next run as the target-not-circular "
            "warning."
        )
    target_name = config.target_name or next(name for name in spacecraft_names if name != pursuer_name)
    if target_name == pursuer_name or target_name not in spacecraft_names:
        raise ParseError("Configured target is invalid")
    if any(spacecraft == target_name for _, _, spacecraft in maneuvers):
        raise ParseError("Target spacecraft has a Maneuver command, which violates the preliminary policy")

    spacecraft_specs = {
        name: _build_spacecraft(
            name,
            _object_fields(assignments, name),
            config.policy.mu_km3_s2,
            probe,
            _binding(name, assignments, created),
        )
        for name in spacecraft_names
    }
    pursuer = spacecraft_specs[pursuer_name]
    target = spacecraft_specs[target_name]
    warnings: list[str] = []
    epoch_offset_s: float | None = None
    if pursuer.probed_epoch is not None and target.probed_epoch is not None:
        # Absolute instants, so a leap second or two DateFormats cannot fool it.
        epoch_offset_s = (target.probed_epoch.a1_mjd - pursuer.probed_epoch.a1_mjd) * 86400.0
        for spec in (pursuer, target):
            if spec.in_process_mismatch is None:
                continue
            position_km, velocity_km_s = spec.in_process_mismatch
            if (
                position_km > PROBE_POSITION_TOLERANCE_KM
                or velocity_km_s > PROBE_VELOCITY_TOLERANCE_KM_S
            ):
                warnings.append(
                    f"GMAT reads {spec.name}'s initial state {position_km:.3g} km and "
                    f"{velocity_km_s:.3g} km/s from our in-process conversion of the same "
                    f"fields (tolerance {PROBE_POSITION_TOLERANCE_KM:g} km, "
                    f"{PROBE_VELOCITY_TOLERANCE_KM_S:g} km/s); the probed state is used"
                )
    else:
        _assert_one_epoch_text(pursuer, target, assignments)
    alignment, propagate_segments = _resolve_alignment(
        aligning,
        propagate_segments,
        first_command_line,
        [line for line, _, _ in maneuvers if line not in head_start_lines],
        spacecraft_specs,
        pursuer,
        target,
        document,
        config,
    )
    ordered = interception_rules(config.policy.rules)
    if (ordered.pursuer_epoch == "at-or-after-target" and alignment is not None
            and alignment.offset_s > 0.0 and alignment.early == pursuer.name):
        # With or without a set-epoch record: the rules say who may start first.
        raise PolicyError(
            f"{pursuer.name} starts {alignment.offset_s:.3f} s before {target.name}, which "
            f"the {ordered.id} rules forbid ({ordered.pursuer_epoch_clause})")
    burn_slots: list[BurnSlot] = []
    seen_burns: set[str] = set()
    for maneuver_line, burn_name, spacecraft_name in maneuvers:
        if spacecraft_name != pursuer_name:
            continue
        if burn_name in seen_burns:
            raise ParseError(f"Burn {burn_name} is maneuvered more than once; repeated burn objects are unsupported")
        seen_burns.add(burn_name)
        if burn_name not in burn_names:
            raise ParseError(f"Maneuver references undeclared ImpulsiveBurn {burn_name}")
        burn_slots.append(
            _slot_from_resource(assignments, burn_name, pursuer_name, maneuver_line)
        )

    if not maneuvers:
        # Nothing is maneuvered, so any declared ImpulsiveBurn is a resource the
        # organizer provided and left for the contestant to use. Using theirs is
        # a smaller edit than adding our own beside it, and their frame is what
        # they expect the impulse written in.
        for burn_name in burn_names:
            if burn_name in assignments and all(
                element in assignments[burn_name]
                for element in ("Element1", "Element2", "Element3")
            ):
                burn_slots.append(_slot_from_resource(assignments, burn_name, pursuer_name, None))
            else:
                # Its components are GMAT defaults rather than lines, so there
                # is nothing to rewrite in place and a maneuver of our own is
                # used instead. Harmless, and not something to discover from a
                # diff: part of the organizer's template went unused.
                warnings.append(
                    f"ImpulsiveBurn {burn_name} is declared without explicit "
                    "Element1/2/3 assignments, so it cannot carry an impulse and is left alone"
                )
        # One Propagate and no Maneuver, however many resources are declared.
        if len(propagate_segments) != 1:
            raise ParseError(
                "A mission sequence with no Maneuver command must be a single Propagate; "
                f"found {len(propagate_segments)}"
            )
    elif len(propagate_segments) != len(burn_slots) - len(head_start) + 1:
        raise ParseError(
            "Supported competition template must alternate Propagate/Maneuver and contain exactly one more "
            f"Propagate segment than burn slots; found {len(propagate_segments)} segments and "
            f"{len(burn_slots) - len(head_start)} burns"
        )
    if maneuvers:
        command_lines = sorted(
            [
                (segment.line_index, "propagate")
                for segment in propagate_segments
                if not segment.implicit
            ]
            + [
                (slot.maneuver_line, "maneuver")
                for slot in burn_slots
                if slot.maneuver_line is not None and slot.maneuver_line not in head_start_lines
            ]
        )
        expected = ["propagate"]
        for _ in burn_slots[len(head_start):]:
            expected.extend(["maneuver", "propagate"])
        kinds = [kind for _, kind in command_lines]
        if propagate_segments and propagate_segments[0].implicit:
            kinds.insert(0, "propagate")
        if kinds != expected:
            raise ParseError(
                "Mission sequence is not the supported Propagate/Maneuver alternating pattern"
            )

    if config.policy.require_synchronized_propagation:
        expected_names = {pursuer_name, target_name}
        for segment in propagate_segments:
            # Not `names`. That name is already the list of objects one `Create`
            # declares, two hundred lines up in this same function, and reusing
            # it here for a set of spacecraft on a Propagate line made two
            # unrelated things look like one.
            propagated = set(segment.spacecraft_names)
            if propagated != expected_names:
                raise ParseError(
                    f"Propagate line {segment.line_index + 1} propagates {sorted(propagated)}, "
                    f"not {sorted(expected_names)}, so it cannot keep them synchronized"
                )
            if segment.epoch_lock is None:
                raise ParseError(
                    f"Propagate line {segment.line_index + 1} steps the craft in separate "
                    "calls without `Synchronized` and stops on one craft's goal: GMAT halts "
                    "the rest wherever their own steps reached, which the script alone "
                    "cannot bound. Set policy.require_synchronized_propagation to false only "
                    "after a GMAT replay shows both craft on one epoch"
                )

    reports = [
        _report_file_spec(report_name, _object_fields(assignments, report_name))
        for report_name in created.get("reportfile", [])
    ]

    force_fields: dict[str, str] = {}
    for force_name in created.get("forcemodel", []):
        force_fields.update({f"{force_name}.{key}": value for key, value in _object_fields(assignments, force_name).items()})
    propagator_fields: dict[str, str] = {}
    for prop_name in created.get("propagator", []):
        propagator_fields.update({f"{prop_name}.{key}": value for key, value in _object_fields(assignments, prop_name).items()})

    aligned = None
    if alignment is not None and alignment.offset_s > 0.0:
        binding = alignment_binding_sha256(
            alignment.text,
            [_binding(name, assignments, created) for name in (pursuer.name, target.name)],
            force_fields,
            propagator_fields,
        )
        aligned = probe.aligned if probe is not None else None
        if aligned is None:
            raise AlignedStatesNeeded(alignment, binding)
        if aligned.binding_sha256 != binding:
            raise ParseError(
                "The alignment probe was flown on a script with another alignment, force "
                "model, propagator or initial state than this one"
            )
        # The solver's t = 0 is the alignment's end; both craft start there.
        pursuer = dataclasses.replace(pursuer, initial_state=aligned.states[pursuer.name])
        target = dataclasses.replace(target, initial_state=aligned.states[target.name])

    # The central body decides mu, and mu decides the period, the horizon and
    # every Lambert arc. Our two-body layer carries one mu, from the config, so
    # a script about another body is not a harder problem -- it is a different
    # one, answered with the wrong constant and no sign that anything is off.
    central_body = config.policy.central_body_name
    for key, value in force_fields.items():
        lowered = key.lower()
        named = _strip_quotes(value).strip()
        if lowered.endswith(".centralbody"):
            if named.lower() != central_body.lower():
                raise ParseError(
                    f"{key} is {named!r}, but the analytical layer models {central_body!r} "
                    f"with mu={config.policy.mu_km3_s2:.6g} km^3/s^2. Set policy.mu_km3_s2, "
                    "policy.central_body_radius_km and policy.central_body_name together, or "
                    "the period, the horizon and every transfer are computed about the wrong body"
                )
        elif lowered.endswith(".primarybodies"):
            bodies = [item.strip() for item in named.strip("{}").split(",") if item.strip()]
            if central_body not in bodies:
                raise ParseError(
                    f"{key} is {named!r}, which does not include the modelled central body "
                    f"{central_body!r}"
                )
        else:
            gravity = re.search(r"\.gravityfield\.([a-z_]\w*)\.", lowered)
            if gravity and gravity.group(1) != central_body.lower():
                raise ParseError(
                    f"{key} configures a gravity field about {gravity.group(1)!r}, not the "
                    f"modelled central body {central_body!r}"
                )
    # Only what the Propagate commands fly: an unused ForceModel is not a gap,
    # and a propagator without FM flies GMAT's default, measured on R2026a
    # 86.291 km from point mass after 6000 s.
    flown_propagators = list(dict.fromkeys([
        *(name for segment in propagate_segments for name in segment.propagator_names),
        *([alignment.propagator] if alignment is not None else []),
    ]))
    gaps: list[str] = []
    for prop_name, model in _flown_force_models(created, assignments, flown_propagators):
        if model is None:
            gaps.append(
                f"Analytical seed ignores non-spherical gravity: {prop_name} sets no FM, so GMAT "
                f"flies its default force model, {central_body} Degree={_GMAT_DEFAULT_GRAVITY_DEGREE}, "
                f"Order={_GMAT_DEFAULT_GRAVITY_DEGREE}"
            )
        else:
            gaps.extend(
                _ignored_force_warnings(model, _object_fields(assignments, model), central_body)
            )
    warnings.extend(gaps)
    target_elements = cartesian_to_keplerian(target.initial_state, config.policy.mu_km3_s2)
    if not (math.isfinite(target_elements.sma_km) and math.isfinite(target_elements.ecc)):
        # nan fails both tests below and would read as an open orbit.
        raise ParseError(
            f"{target.name}'s orbital elements are not finite (SMA {target_elements.sma_km}, "
            f"ECC {target_elements.ecc}) under mu = {config.policy.mu_km3_s2}"
        )
    target_is_closed = target_elements.sma_km > 0.0 and target_elements.ecc < 1.0
    if (
        config.policy.horizon_s is None
        and target_is_closed
        and target_elements.ecc > config.policy.target_circular_tolerance
    ):
        # 說明會簡報 p.8: 在初賽中,Spacecraft A 將以一圓軌道繞行地球. Said, not
        # refused. The period is just as well defined for an ellipse, so
        # nothing here is ambiguous -- but the scenario is then not the one the
        # briefing described, and that is worth hearing before trusting an
        # answer built on top of it. An announced horizon means the ranking rules.
        warnings.append(
            f"Preliminary briefing states the target orbit is circular; "
            f"{target.name} has eccentricity {target_elements.ecc:.6g}"
        )
    body_km = config.policy.central_body_radius_km
    for spacecraft in (pursuer, target):
        start_km = math.hypot(*spacecraft.initial_state.r_km)
        if start_km <= body_km:
            raise ParseError(
                f"{spacecraft.name} starts {start_km:.6g} km from the centre, inside the "
                f"{body_km:.6g} km central body: the initial state is not a flight"
            )
    target_period = (
        orbital_period_s(target_elements.sma_km, config.policy.mu_km3_s2)
        if target_is_closed
        else None
    )
    if config.policy.horizon_s is not None:
        # Even for a closed target: 4*T_A is the preliminary's rule.
        max_time = float(config.policy.horizon_s)
        if not math.isfinite(max_time):
            raise ParseError(f"policy.horizon_s is not finite: {max_time!r}")
    elif interception_rules(config.policy.rules).horizon == "announced":
        raise ParseError(
            f"policy.rules is {config.policy.rules!r}, whose T_max is announced on the day; "
            "set policy.horizon_s to it rather than fall back to 4*T_A"
        )
    elif target_period is not None:
        max_time = 4.0 * target_period
    else:
        raise ParseError(
            f"{target.name} is on an open orbit (ECC {target_elements.ecc:.6g}, SMA "
            f"{target_elements.sma_km:.6g} km) and has no 4*T_A; set policy.horizon_s to "
            "the T_max announced on the day (ranking rules 2026/10/06 §3), not a guess"
        )
    time_origin_offset_s = 0.0
    alignment_minimum_distance_km: float | None = None
    origin = time_origin_craft(
        config.policy.mission_time_origin, pursuer.name, target.name, alignment
    )
    if alignment is not None and aligned is not None:
        # T_max and T_team run from the origin craft's epoch; the solver's from
        # the alignment's end, Δ later when that craft is the early one.
        if origin == alignment.early and alignment.offset_s >= max_time:
            raise ParseError(
                f"{alignment.early} starts {alignment.offset_s:.9g} s before {alignment.late}, "
                f"and T_max = {max_time:.9g} s runs from {origin}'s epoch, so it is reached "
                "before the two fly together; nothing is left to fly"
            )
        if origin == alignment.early:
            time_origin_offset_s = alignment.offset_s
            max_time -= time_origin_offset_s
        # From the later craft's epoch the alignment's rows come before the origin.
        if config.policy.closest_approach_window == "from_origin" and origin == alignment.early:
            alignment_minimum_distance_km = aligned.minimum_distance_km
        if alignment.early == pursuer.name:
            # No burn may come before t = 0 (Q2), so nothing can save this arc.
            if aligned.early_minimum_radius_km < body_km:
                raise ParseError(
                    f"{pursuer.name} flies alone through the alignment down to "
                    f"{aligned.early_minimum_radius_km:.6g} km from the centre, inside the "
                    f"{body_km:.6g} km central body, before any burn may be placed: the "
                    "template is infeasible"
                )
            floor_km = config.policy.minimum_flown_altitude_km
            if floor_km is not None and aligned.early_minimum_radius_km < body_km + floor_km:
                # verify measures every row from the first, so it would refuse each candidate.
                raise ParseError(
                    f"{pursuer.name} flies alone through the alignment down to "
                    f"{aligned.early_minimum_radius_km - body_km:.3f} km altitude, under "
                    f"policy.minimum_flown_altitude_km = {floor_km}, before any burn may be "
                    "placed: every candidate would be refused"
                )
    head_start_s, head_start_note = _head_start_lead(
        alignment, pursuer.name, target.name, origin, config
    )
    head_start_state = head_start_defect = None
    if head_start_s > 0.0 and probe is not None:
        probed = probe.craft_named(pursuer.name)
        head_start_state = probed.state if probed is not None else None
    if head_start_state is not None:
        # The unburned two-body flight against GMAT's own, at t = 0: 66 km and 69 m/s
        # on R9's 3600 s head start, 20 km on R8's 1800 s.
        coasted = propagate_universal(head_start_state, head_start_s, config.policy.mu_km3_s2)
        head_start_defect = StateVector(
            as_vector3(pursuer.initial_state.r - coasted.r),
            as_vector3(pursuer.initial_state.v - coasted.v),
        )
    if head_start:
        _check_head_start(head_start, document, alignment, pursuer.name, head_start_s,
                          head_start_note, config)
    # A pericenter inside the body is not refused: an inserted craft may start on
    # an arc it must burn off (the tutorial's insertion has its pericenter near
    # 6031 km). The surface checks on every flown arc hold the line.
    for spacecraft in (pursuer, target):
        crossing_s = first_radius_crossing_s(
            spacecraft.initial_state, body_km, max_time, config.policy.mu_km3_s2
        )
        if crossing_s is not None:
            consequence = (
                "a submission must burn before then, and the zero burn is not one"
                if spacecraft is pursuer
                else "before T_max"
            )
            warnings.append(
                f"{spacecraft.name}, unpowered, reaches the {body_km:.6g} km central body at "
                f"t = {crossing_s:.3f} s (two-body): {consequence}"
            )
    horizon: HorizonCheck | None = None
    horizon_before_perigee_accepted = False
    if config.policy.horizon_s is not None and (
        interception_rules(config.policy.rules).horizon == "announced"
    ):
        check = horizon = check_horizon(
            target.initial_state, max_time, config.policy.horizon_s,
            config.policy.mission_boundary_km, config.policy.mu_km3_s2,
        )
        accepted = config.policy.accept_announced_horizon_before_perigee_s
        if check.near_circular:
            warnings.append(near_circular_warning(check, config.policy.horizon_s))
        if check.before_perigee and accepted == config.policy.horizon_s:
            horizon_before_perigee_accepted = True
            warnings.append(horizon_override_warning(
                check, config.policy.horizon_s, origin, time_origin_offset_s
            ))
        elif check.before_perigee:
            raise ParseError(
                horizon_refusal(check, config.policy.horizon_s, origin, time_origin_offset_s)
            )
        if check.off_boundary:
            warnings.append(horizon_warning(check, config.policy.horizon_s))

    scenario = Scenario(
        source_path=str(Path(path).resolve()),
        source_sha256=file_sha256(path),
        pursuer=pursuer,
        target=target,
        burn_slots=tuple(burn_slots),
        propagate_segments=tuple(propagate_segments),
        report_files=tuple(reports),
        force_model_fields=force_fields,
        propagator_fields=propagator_fields,
        mu_km3_s2=config.policy.mu_km3_s2,
        target_period_s=target_period,
        max_time_s=max_time,
        encoding=document.encoding,
        warnings=tuple(warnings),
        two_body_gaps=tuple(gaps),
        rules=config.policy.rules,
        synthesized_burn_frame=SYNTHESIZED_BURN_FRAMES[config.policy.synthesized_burn_frame],
        state_probe=probe,
        epoch_offset_s=epoch_offset_s,
        alignment=alignment,
        geodetic_radius_km=(
            _as_float(
                _object_fields(assignments, "Earth").get("EquatorialRadius", "6378.1363"),
                "Earth.EquatorialRadius",
            )
            if central_body.lower() == "earth"
            else None
        ),
        time_origin_offset_s=time_origin_offset_s,
        time_origin_craft=origin,
        horizon_check=horizon,
        horizon_before_perigee_accepted=horizon_before_perigee_accepted,
        alignment_minimum_distance_km=alignment_minimum_distance_km,
        head_start_s=head_start_s,
        head_start_state=head_start_state,
        head_start_defect=head_start_defect,
        head_start_note=head_start_note,
        head_start_segments=tuple(head_start),
    )
    return scenario, document


def scenario_json(scenario: Scenario) -> str:
    return json.dumps(scenario.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
