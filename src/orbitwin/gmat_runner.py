from __future__ import annotations

import errno
import os
import re
import resource
import shutil
import signal
import subprocess
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

from .errors import GmatRunError, GmatVerdictError
from .models import GmatRunResult, Scenario, file_sha256

#: The line scripts/fake_gmat_console.py declares itself with.
TEST_DOUBLE_MARKER = b"\nORBITWIN_GMAT_TEST_DOUBLE = True\n"


def is_test_double(console: str | Path) -> bool:
    """Whether ``console`` is orbitwin's GmatConsole test double, by the line it carries."""
    try:
        with open(console, "rb") as handle:
            return TEST_DOUBLE_MARKER in handle.read(4096)
    except OSError:
        return False


#: Draw and write nothing; console runs of private copies comment them out.
PLOT_TYPES = frozenset(
    {
        "OrbitView",
        "GroundTrack",
        "GroundTrackPlot",
        "XYPlot",
        "Enhanced3DView",
        "DynamicDataDisplay",
        "OpenFramesInterface",
        "OpenFramesView",
        "OpenFramesVector",
        "OpenFramesSensorMask",
    }
)
#: Output-file fields: relative and without ".." so they land in the run's
#: OUTPUT_PATH. R2026a wrote a relative EphemerisFile Filename there even with
#: VEHICLE_EPHEM_PATH moved elsewhere, and refused one in a subdirectory.
OUTPUT_FILE_FIELDS: dict[str, tuple[str, ...]] = {
    "ReportFile": ("Filename",),
    "EphemerisFile": ("Filename",),
    "DifferentialCorrector": ("ReportFile",),
}


#: GMAT on Linux reads a backslash as "/", so a POSIX path check let '..\..\x.txt'
#: climb out of the run; ":" is a drive. R2026a wrote 'Report File1.txt' and
#: 'Team(07).txt' into gmat-output/, so nothing else is refused.
_PATH_CHARACTERS = frozenset("/\\:")


def is_bare_file_name(value: str) -> bool:
    """Whether an output file name can only land in the run's OUTPUT_PATH."""
    return (
        value.strip(".") != ""
        and not any(char in _PATH_CHARACTERS for char in value)
        and value.isprintable()
    )


_OUTPUT_PATH_RE = re.compile(r"^\s*OUTPUT_PATH\s*=.*$", re.IGNORECASE)
_LOG_FILE_RE = re.compile(r"^\s*LOG_FILE\s*=.*$", re.IGNORECASE)
_FATAL_MARKERS = (
    "ERROR!!!!!!",
    "could not read script",
    "script file did not parse",
    "Moderator::RunMission failed",
    "Unhandled Exception",
)


#: How GMAT opens an error record in its log; the record runs on to the next blank line.
LOG_ERROR_MARKER = "**** ERROR ****"
#: Errors a run may log and still be certified. Empty: of 19,945 GmatLog.txt files of
#: completed R2026a runs under output/ on 2026-11-07, the R1-R8 rehearsals among them, none
#: holds one. Add a record only with the run that showed it harmless.
BENIGN_LOG_ERRORS: tuple[re.Pattern[str], ...] = ()


def log_error_records(text: str) -> list[str]:
    """Each error record in a GmatLog, from its marker to the blank line ending it, on one line.

    The script path GMAT prefixes is left out: it names the run directory.
    """
    records: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if LOG_ERROR_MARKER in line:
            if current is not None:
                records.append(" ".join(current))
            current = [line[line.index(LOG_ERROR_MARKER):].strip()]
        elif current is not None and line.strip():
            current.append(line.strip())
        elif current is not None:
            records.append(" ".join(current))
            current = None
    if current is not None:
        records.append(" ".join(current))
    return records


def unexplained_log_errors(records: Iterable[str]) -> list[str]:
    """The records BENIGN_LOG_ERRORS does not name."""
    return [record for record in records
            if not any(pattern.search(record) for pattern in BENIGN_LOG_ERRORS)]


def _timeout_output_text(value: str | bytes | None) -> str:
    """Whatever a TimeoutExpired managed to capture, as text.

    `subprocess.run` is called below in text mode, but on POSIX what a
    TimeoutExpired carries is still bytes (Python 3.13 returned b'abc' with
    encoding= set), and elsewhere it can be str; this takes either, since
    `write_text` takes only text.
    Decoding with `errors="replace"` matches how the same bytes are written on
    the path that did not time out -- a half-written line from a killed
    GmatConsole is evidence, and it should not become an exception of its own
    while being filed.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


_STARTUP_ENTRY_RE = re.compile(r"^\s*(?P<key>[A-Za-z][A-Za-z0-9_]*)\s*=\s*(?P<value>.*?)\s*$")
#: Startup-file entries GMAT writes rather than reads.
_STARTUP_OUTPUTS = frozenset({"LOG_FILE", "SCREENSHOT_FILE"})
#: Entries only the GUI reads. MyGmat.ini changes whenever the GUI is opened.
_GUI_ONLY_KEY_RE = re.compile(
    r"(?:^|_)(?:PERSONALIZATION|TEXTURE|ICON|SPLASH|STAR|CONSTELLATION|BORDER|MODEL|HELP)(?:_|$)"
)
_CREATE_RE = re.compile(
    r"^[ \t]*(?:GMAT\s+)?Create\s+(?P<type>[A-Za-z_]\w*)\s+(?P<names>[^;\n]+);", re.MULTILINE
)
#: Resource fields that name an input file: EopFileName, DEFilename, PotentialFile, ...
#: A bare FileName/Filename is not one: a body's texture, or a ReportFile's output.
_FILE_FIELD_RE = re.compile(
    r"^[ \t]*(?:GMAT\s+)?(?P<object>[A-Za-z_]\w*)\.(?P<field>(?:[A-Za-z_]\w*\.)*"
    r"[A-Za-z_]\w*(?:File|FileName|Filename|Kernel|Kernels|KernelName))\s*=\s*(?P<value>[^;\n]*);",
    re.MULTILINE,
)
#: Their file fields are outputs or drawings, not data.
_WRITERS_AND_PLOTS = PLOT_TYPES | frozenset(OUTPUT_FILE_FIELDS)
_GUI_ONLY_FIELDS = frozenset({"ModelFile", "TextureMapFileName"})
_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_HASH_LOCK = threading.Lock()


def _startup_entries(
    startup_file: str | Path, gmat_console: str | Path
) -> tuple[list[Path], dict[str, Path]]:
    """The startup file's existing ``*_PATH`` directories and ``*_FILE`` files.

    Relative paths resolve against the executable's directory, which is where
    run_gmat starts GmatConsole; ``<KEY>_PATH/rest`` expands an earlier entry.
    """
    directories, files, _ = _startup_tables(startup_file, gmat_console)
    unique: list[Path] = []
    for directory in directories.values():
        if directory not in unique:
            unique.append(directory)
    return unique, files


def startup_data_paths(startup_file: str | Path, gmat_console: str | Path) -> dict[str, Path]:
    """Every existing directory a startup file's ``*_PATH`` entries name, by key."""
    return _startup_tables(startup_file, gmat_console)[0]


def _startup_tables(
    startup_file: str | Path, gmat_console: str | Path
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    """Existing ``*_PATH`` directories, existing ``*_FILE`` files, and every ``*_FILE`` as named."""
    base = Path(gmat_console).resolve().parent
    paths: dict[str, str] = {}
    files: dict[str, Path] = {}
    declared: dict[str, Path] = {}
    for line in Path(startup_file).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = _STARTUP_ENTRY_RE.match(line)
        if match is None:
            continue
        key, value = match.group("key").upper(), match.group("value")
        head, slash, rest = value.partition("/")
        if slash and head in paths:
            value = paths[head].rstrip("/") + "/" + rest
        candidate = Path(value) if Path(value).is_absolute() else base / value
        if key.endswith("_PATH"):
            paths[key] = value
            continue
        if key.endswith("_FILE"):
            declared[key] = candidate
        if (
            key.endswith("_FILE")
            and key not in _STARTUP_OUTPUTS
            and not _GUI_ONLY_KEY_RE.search(key)
            and candidate.is_file()
        ):
            files[key] = candidate.resolve()
    directories: dict[str, Path] = {}
    for key, value in paths.items():
        directory = Path(value) if Path(value).is_absolute() else base / value
        if directory.is_dir():
            directories[key] = directory.resolve()
    return directories, files, declared


#: Spacecraft fields naming the SPAD file a SPADFile drag or SRP force reads.
SPAD_FILE_FIELDS = ("SPADDragFile", "SPADSRPFile")


def resolve_spad_file(
    named: str | None,
    field: str,
    script_dir: str | Path | None,
    startup_file: str | Path,
    gmat_console: str | Path,
) -> Path | None:
    """The file GMAT opens for a Spacecraft's SPADDragFile/SPADSRPFile, or None.

    FileManager::FindPath for input (nasa/GMAT): a blank name is SPAD_SRP_FILE
    for SRP and nothing for drag; an absolute name as written; else the
    script's folder, GMAT's working directory (None: one holding only the
    script, as run_gmat's are); then a ./ or ../ name against GmatConsole's
    folder, any other against SPAD_SRP_FILE's folder for SRP, or the process
    directory, which run_gmat makes GmatConsole's, for drag, which has no type.
    None for a field never set: GMAT then has no file at all.
    """
    if named is None:
        return None
    declared = (
        _startup_tables(startup_file, gmat_console)[2].get("SPAD_SRP_FILE")
        if field == "SPADSRPFile"
        else None
    )
    name = named.replace("\\", "/")
    if not name:
        return declared.resolve() if declared is not None and declared.is_file() else None
    given = Path(name)
    if given.is_absolute():
        return given.resolve() if given.is_file() else None
    bin_dir = Path(gmat_console).resolve().parent
    places = [] if script_dir is None else [Path(script_dir) / given]
    if name.startswith(("./", "../")) or declared is None:
        places.append(bin_dir / given)
    else:
        places.append(declared.parent / given)
    return next((place.resolve() for place in places if place.is_file()), None)


def startup_data_files(startup_file: str | Path, gmat_console: str | Path) -> dict[str, Path]:
    """Every existing file a startup file's ``*_FILE`` entries name, by key."""
    return _startup_entries(startup_file, gmat_console)[1]


def script_data_files(
    script: str | Path, startup_file: str | Path, gmat_console: str | Path
) -> dict[str, Path | str]:
    """Data files the script's resources name, such as Earth.EopFileName or a PotentialFile.

    They override or add to the startup file's. R2026a opened Earth.EopFileName
    and a CSSI file only from its working directory, the executable's, or the
    file's startup path; a gravity or tide file beside the script first
    (FileManager::FindPath). A name found nowhere is returned as written, so
    the manifest still shows it.
    """
    script_dir = Path(script).resolve().parent
    fallbacks = _fallback_dirs(startup_file, gmat_console)
    found: dict[str, Path | str] = {}
    for key, leaf, names in _data_file_fields(script):
        if leaf in SPAD_FILE_FIELDS and len(names) == 1:
            spad = resolve_spad_file(names[0], leaf, script_dir, startup_file, gmat_console)
            found[key] = spad if spad is not None else names[0]
            continue
        beside = script_dir if leaf in BESIDE_SCRIPT_FIELDS else None
        for index, name in enumerate(names):
            indexed = key + (f"[{index}]" if len(names) > 1 else "")
            place = _resolve_data_file(gmat_file_name(leaf, name), beside, fallbacks)
            found[indexed] = place if place is not None else name
    return found


#: Fields GMAT looks for beside the script first. Measured on R2026a: a
#: PotentialFile or TideFile beside the template, rc 0 in place and "does not
#: exist" from a run folder; Earth.EopFileName and Drag.CSSISpaceWeatherFile
#: beside it, never found ("Error opening EopFile", "Cannot open the CSSI
#: space weather file ... nor ... ../data/atmosphere/earth/").
BESIDE_SCRIPT_FIELDS = frozenset({"PotentialFile", "TideFile", *SPAD_FILE_FIELDS})
#: GMAT adds these to a name with no "." in it (HarmonicField, GravityField).
_DEFAULT_EXTENSIONS = {"PotentialFile": ".cof", "TideFile": ".tide"}


def gmat_file_name(leaf: str, name: str) -> str:
    """``name`` as GMAT completes it for the field ``leaf``."""
    extension = _DEFAULT_EXTENSIONS.get(leaf)
    return name + extension if extension and name and "." not in name else name


def _fallback_dirs(startup_file: str | Path, gmat_console: str | Path) -> list[Path]:
    """GmatConsole's folder, then the startup file's paths."""
    return [Path(gmat_console).resolve().parent, *_startup_entries(startup_file, gmat_console)[0]]


def _resolve_data_file(name: str, script_dir: Path | None, fallbacks: list[Path]) -> Path | None:
    """Where GMAT would open ``name``: as written if absolute, else in ``script_dir``
    when given, then the fallbacks."""
    given = Path(name.replace("\\", "/"))
    if given.is_absolute():
        return given.resolve() if given.is_file() else None
    places = ([] if script_dir is None else [script_dir]) + fallbacks
    return next((place.resolve() for place in (d / given for d in places) if place.is_file()), None)


def force_model_file_problems(
    script: str | Path,
    startup_file: str | Path,
    gmat_console: str | Path,
    flown: frozenset[str] | None = None,
) -> list[str]:
    """Each file a ForceModel of ``script`` names that GMAT would miss, run in place or by us.

    GMAT refuses the whole script for a missing one (R2026a, TideFile); the
    state probe drops the force models, so without this the first sign was a
    bare exit status deep in solve.
    """
    created = _created_types(script)
    # A drag file is opened only when a force model ``flown`` (default: any a
    # Propagator names) starts; a gravity or tide file as the line is read
    # (R2026a: "... does not exist in line").
    if flown is None:
        flown = frozenset(_PROPAGATOR_FM_RE.findall(_resources_text(script)))
    template_dir = Path(script).resolve().parent
    fallbacks = _fallback_dirs(startup_file, gmat_console)
    problems: list[str] = []
    for key, leaf, names in _data_file_fields(script):
        owner = key.removeprefix("script:").split(".", 1)[0]
        if created.get(owner) != "ForceModel":
            continue
        beside = leaf in BESIDE_SCRIPT_FIELDS
        if not beside and owner not in flown:
            continue
        for name in (gmat_file_name(leaf, name) for name in names):
            in_place = _resolve_data_file(name, template_dir if beside else None, fallbacks)
            ours = (
                in_place if beside and stageable_name(name)
                else _resolve_data_file(name, None, fallbacks)
            )
            if in_place is None:
                where = (
                    "beside the template, then beside GmatConsole and in the startup paths"
                    if beside else "beside GmatConsole and in the startup paths, not beside "
                    "the template"
                )
                problems.append(f"{key.removeprefix('script:')} = {name!r}, found nowhere ({where})")
            elif ours is None:
                problems.append(
                    f"{key.removeprefix('script:')} = {name!r}, found beside the template only, "
                    "by a name with '..' that our run folders cannot hold"
                )
    return problems


_PROPAGATOR_FM_RE = re.compile(
    r"^[ \t]*(?:GMAT\s+)?[A-Za-z_]\w*\.FM\s*=\s*([A-Za-z_]\w*)\s*;", re.MULTILINE
)


def _resources_text(script: str | Path) -> str:
    text = Path(script).read_text(encoding="utf-8", errors="replace")
    return re.split(r"^\s*BeginMissionSequence\b", text, maxsplit=1, flags=re.MULTILINE)[0]


def _created_types(script: str | Path) -> dict[str, str]:
    created: dict[str, str] = {}
    for match in _CREATE_RE.finditer(_resources_text(script)):
        for name in re.split(r"[,\s]+", match.group("names").strip()):
            if name:
                created[name] = match.group("type")
    return created


def _data_file_fields(script: str | Path) -> list[tuple[str, str, list[str]]]:
    """(manifest key, field leaf, names) for each input-file field of the script's resources."""
    resources = _resources_text(script)
    created = _created_types(script)
    fields: list[tuple[str, str, list[str]]] = []
    for match in _FILE_FIELD_RE.finditer(resources):
        owner, field = match.group("object"), match.group("field")
        leaf = field.rsplit(".", 1)[-1]
        if created.get(owner) in _WRITERS_AND_PLOTS or leaf in _GUI_ONLY_FIELDS:
            continue
        names = [item.strip().strip("'\"") for item in match.group("value").strip("{}").split(",")]
        fields.append((f"script:{owner}.{field}", leaf, [name for name in names if name]))
    return fields


def stageable_name(name: str) -> bool:
    """Whether a run's folder can hold ``name`` where GMAT looks first, beside the script."""
    relative = PurePosixPath(name.replace("\\", "/"))
    return bool(name) and not relative.is_absolute() and ".." not in relative.parts


def stage_template_files(
    script: str | Path,
    template_dir: str | Path,
    run_dir: str | Path,
    reserved: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Copy the files ``script`` names beside the template into ``run_dir``, as named.

    GMAT reads a relative data-file name from the script's folder first, so a
    template shipped with a .tide or .spo beside it flies with that file; a copy
    run alone found none (R2026a: 'The file name "X.tide" does not exist').
    A name whose first part is in ``reserved``, a file the run writes itself, is refused.
    """
    source_dir, target_dir = Path(template_dir).resolve(), Path(run_dir).resolve()
    #: copy -> the template's file, both resolved.
    staged: dict[str, str] = {}
    if source_dir == target_dir:
        return staged
    for key, leaf, names in _data_file_fields(script):
        if leaf not in BESIDE_SCRIPT_FIELDS:
            continue  # GMAT would not look for it there
        for name in (gmat_file_name(leaf, name) for name in names):
            if not stageable_name(name):
                continue
            relative = name.replace("\\", "/")
            source = source_dir / relative
            if not source.is_file():
                continue
            if PurePosixPath(relative).parts[0] in reserved:
                raise GmatRunError(
                    f"{key.removeprefix('script:')} names {name!r}, found beside the template, "
                    "which is a name this run writes itself"
                )
            target = target_dir / relative
            if target.exists() and not (target.is_file() and file_sha256(target) == file_sha256(source)):
                raise GmatRunError(
                    f"{key.removeprefix('script:')} names {name!r}, found beside the template, but "
                    f"{target} is already there in the run folder"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            staged[str(target.resolve())] = str(source.resolve())
    return staged


def _cached_sha256(path: Path) -> str:
    # Keyed on size and mtime: the data files are ~150 MB and every run lists them.
    status = path.stat()
    key = (str(path), status.st_size, status.st_mtime_ns)
    with _HASH_LOCK:
        known = _HASH_CACHE.get(key)
    if known is None:
        known = file_sha256(path)
        with _HASH_LOCK:
            _HASH_CACHE[key] = known
    return known


def data_file_hashes(
    startup_file: str | Path, gmat_console: str | Path
) -> dict[str, dict[str, str]]:
    """The SHA-256 of each data file the startup file names (EOP, leap seconds, ...)."""
    return {
        key: {"path": str(path), "sha256": _cached_sha256(path)}
        for key, path in sorted(startup_data_files(startup_file, gmat_console).items())
    }


def script_data_file_hashes(
    script: str | Path, startup_file: str | Path, gmat_console: str | Path
) -> dict[str, dict[str, str]]:
    """As data_file_hashes, for the files the script itself names; "not found" if unresolved."""
    hashes: dict[str, dict[str, str]] = {}
    for key, path in sorted(script_data_files(script, startup_file, gmat_console).items()):
        if isinstance(path, Path):
            hashes[key] = {"path": str(path), "sha256": _cached_sha256(path)}
            continue
        hashes[key] = {"path": path, "sha256": "not found"}
        # GMAT may still find it where we did not look; then it reads data nothing hashed.
        warnings.warn(
            f"{key.removeprefix('script:')} names {path!r}, which was not found where GMAT "
            "looks for it from this run; the manifest records it unhashed",
            RuntimeWarning,
            stacklevel=2,
        )
    return hashes


#: run_gmat's words for its failures. verify and the tests read them to tell
#: a refusal for plots from any other failure; keep them shared.
EXIT_STATUS_ERROR = "GmatConsole returned exit status {}"
NO_FRESH_REPORT_ERROR = "Expected fresh GMAT report was not created: {}"
TIMEOUT_ERROR = "GmatConsole timed out after {:g} seconds"
FILE_SIZE_ERROR = (
    "GmatConsole wrote a file past the {} byte limit and was stopped (SIGXFSZ); a Propagate "
    "goal it never reaches does that"
)
#: The exit status run_gmat records for a timeout, which kills the group.
TIMEOUT_RETURN_CODE = -1
STALE_REPORT_ERROR = "GMAT report appears stale: {}"
STOPPING_ERROR = "GmatConsole runs are being stopped"
#: GMAT's error record; any other line with ERROR in it (a path, a comment) is not one.
ERROR_RECORD = "**** ERROR ****"
#: A fatal error GMAT prints without the marker: `Command Exception: ...`.
_BARE_EXCEPTION_RE = re.compile(r"\b[A-Z][A-Za-z]* Exception(?: Thrown)?:")
#: The script line GMAT prints after a record: `   " 167: DefaultOrbitView.Size = ...;"`.
_RECORD_LOCATION_RE = re.compile(r'^\s*"\s*(?P<line>\d+):\s?(?P<text>.*?)"\s*$')
#: GMAT's run summary after any failure, and the test double's own.
_SUMMARY = "ERROR!!!!!!"
#: GMAT's unmarked error lines, such as SolarSystem's `ERROR loading kernel ...`, which
#: start the line: an echoed comment or path holding ERROR does not.
_LOOSE_ERROR_RE = re.compile(r"^\s*ERROR [a-z]")
#: How many lines a record's message may run to before its ` in line:` location.
_MESSAGE_LINES = 8

#: verify's words for a refusal for the plots alone, whose report comes from the GUI.
GUI_STOP = "Run it in the GMAT GUI"
#: OSErrors of this machine, not of what certify was asked: no answer (no_answer).
MACHINE_ERRNOS = frozenset({
    errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.ENOMEM, errno.ENOBUFS, errno.EMFILE,
    errno.ENFILE, errno.EAGAIN, errno.EBUSY, errno.EROFS, errno.EACCES, errno.EPERM,
    errno.ETXTBSY,
})
_EXIT_STATUS_RE = re.compile(re.escape(EXIT_STATUS_ERROR.format("")) + r"(-?\d+)")
_NO_ANSWER_PREFIXES = tuple(
    text.split("{", 1)[0]
    for text in (TIMEOUT_ERROR, NO_FRESH_REPORT_ERROR, STALE_REPORT_ERROR, STOPPING_ERROR)
)
#: GMAT said why a run failed: said_by_gmat's lines, verify's _reported and refusal_message.
_GMAT_SAID = ("\n  ", "GMAT reported: ", "GMAT rejects ", ERROR_RECORD)


def gave_no_answer(message: str) -> bool:
    """A run_gmat failure that says nothing about the script: GmatConsole killed by a signal,
    timed out or stopped, or no fresh report, with no error GMAT wrote. A run cut at the
    file-size cap, an exit status GMAT chose and a fatal marker it printed are answers."""
    if any(said in message for said in _GMAT_SAID):
        return False
    if any(prefix in message for prefix in _NO_ANSWER_PREFIXES):
        return True
    for found in _EXIT_STATUS_RE.finditer(message):
        code = int(found[1])
        # subprocess gives a signal as -N, a wrapping shell as 128 + N.
        signalled = -code if code < 0 else code - 128 if 128 < code <= 128 + 64 else 0
        if signalled and signalled != signal.SIGXFSZ:
            return True
    return False


def machine_error(exc: BaseException) -> OSError | None:
    """The OSError of MACHINE_ERRNOS that ``exc`` is, or was raised from or during."""
    link: BaseException | None = exc
    seen: set[int] = set()
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        if isinstance(link, OSError) and link.errno in MACHINE_ERRNOS:
            return link
        link = link.__cause__ or link.__context__
    return None


def no_answer(exc: BaseException) -> bool:
    """certify ended on a GmatConsole run that gave no answer, or on this machine
    (machine_error), not on a verdict: running it again may well certify the candidate."""
    if isinstance(exc, GmatVerdictError):
        return False
    if machine_error(exc) is not None:
        return True
    text = str(exc)
    return (isinstance(exc, GmatRunError) and GUI_STOP not in text
            and gave_no_answer(text))


@dataclass(frozen=True, slots=True)
class GmatError:
    """One error GMAT reported, without the run's path."""

    text: str
    #: "record" (**** ERROR ****), "exception" (a bare `<Kind> Exception:`), "loose"
    #: (another `ERROR ...` line) or "summary".
    kind: str
    #: The script line GMAT printed after it, 1-based, and its text; 0 and "" if none.
    line: int = 0
    source: str = ""

    def __str__(self) -> str:
        return f"{self.text} (line {self.line}: {self.source})" if self.line else self.text


@dataclass(frozen=True, slots=True)
class GmatErrors:
    errors: tuple[GmatError, ...]
    #: False when reading stopped at the limit.
    complete: bool


def _error_start(line: str) -> tuple[str, str] | None:
    if ERROR_RECORD in line:
        return line[line.index(ERROR_RECORD):].strip(), "record"
    if "WARNING" in line:
        return None
    if line.lstrip().startswith(_SUMMARY):
        return line.strip(), "summary"
    bare = _BARE_EXCEPTION_RE.search(line)
    if bare:
        return line[bare.start():].strip(), "exception"
    if _LOOSE_ERROR_RE.match(line):
        return line.strip(), "loose"
    return None


def _errors_in(path: str) -> Iterator[tuple[str, str, int, str]]:
    """(text, kind, line, source) for each error in one of GMAT's outputs.

    GMAT ends a message that names a script line with ` in line:` and then the
    line, `   " 127: ..."`, and a message may span lines before that: those join
    its text. Lines that lead to no location are not part of it.
    """
    pending: tuple[str, str, list[str]] | None = None
    with open(path, encoding="utf-8", errors="replace") as lines:
        for line in lines:
            if pending is not None:
                text, kind, more = pending
                where = _RECORD_LOCATION_RE.match(line)
                if where:
                    yield " ".join([text, *more]), kind, int(where["line"]), where["text"].strip()
                    pending = None
                    continue
                start = _error_start(line)
                # A warning, a start that _error_start declines, ends the message too.
                if (start is None and line.strip() and kind != "summary"
                        and "WARNING" not in line and len(more) < _MESSAGE_LINES):
                    more.append(line.strip())
                    continue
                yield text, kind, 0, ""
                pending = None
            start = _error_start(line)
            if start is not None:
                pending = (*start, [])
    if pending is not None:
        yield pending[0], pending[1], 0, ""


def gmat_errors(result: GmatRunResult, limit: int = 10_000) -> GmatErrors:
    """GMAT's errors from a run's log, stdout and stderr, each once.

    GMAT writes every record to both its log and stdout, so each is kept once,
    keyed by its text, with the `N: <path>:` prefix and the run's directory
    stripped, and by the script line GMAT printed after it. Reading stops past
    ``limit`` distinct errors.
    """
    run_dirs = [f"{Path(result.workdir)}/", f"{Path(result.workdir).resolve()}/"] if result.workdir else []
    found: dict[tuple[str, int], GmatError] = {}
    for path in (result.log_path, result.stdout_path, result.stderr_path):
        if not path or not Path(path).is_file():
            continue
        for text, kind, line, source in _errors_in(path):
            for run_dir in run_dirs:
                text = text.replace(run_dir, "")
            if (text, line) in found:
                continue
            if len(found) >= limit:
                return GmatErrors(tuple(found.values()), False)
            found[text, line] = GmatError(text, kind, line, source)
    return GmatErrors(tuple(found.values()), True)


def said_by_gmat(result: GmatRunResult, limit: int = 5) -> str:
    """The first of GMAT's errors, one per indented line, for an exception message."""
    return "".join(f"\n  {error}" for error in gmat_errors(result, limit).errors)


def assert_safe_output_filename(filename: str) -> None:
    """GMAT clears a ReportFile at start; one outside the run directory is someone's file."""
    if not is_bare_file_name(filename):
        raise GmatRunError(
            f"Unsafe report filename in script: {filename!r} is not a bare file name"
        )


def derive_startup_file(
    original_startup: str | Path,
    destination: str | Path,
    output_directory: str | Path,
    log_file: str | Path,
) -> Path:
    source = Path(original_startup)
    text = source.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    output_value = str(Path(output_directory).resolve()) + os.sep
    log_value = str(Path(log_file).resolve())
    lines = text.splitlines()
    found_output = False
    found_log = False
    rendered: list[str] = []
    for line in lines:
        if _OUTPUT_PATH_RE.match(line):
            rendered.append(f"OUTPUT_PATH            = {output_value}")
            found_output = True
        elif _LOG_FILE_RE.match(line):
            rendered.append(f"LOG_FILE               = {log_value}")
            found_log = True
        else:
            rendered.append(line)
    if not found_output:
        rendered.append(f"OUTPUT_PATH            = {output_value}")
    if not found_log:
        rendered.append(f"LOG_FILE               = {log_value}")
    target = Path(destination)
    target.write_text(newline.join(rendered) + newline, encoding="utf-8")
    return target


#: The largest file a run may write unless the caller sizes it. A goal GMAT
#: never reaches wrote 457 MB in 40 s on R2026a; the timeout alone let it run on.
DEFAULT_FILE_SIZE_LIMIT_BYTES = 2 * 1024**3


_FLOOR_FILE_SIZE_LIMIT_BYTES = 256 * 1024**2
#: Per row of the widest report: 40 columns of 32 characters.
_ROW_BYTES = 40 * 32


def file_size_limit_for(scenario: Scenario) -> int:
    """A file cap for runs of ``scenario``: twenty times the rows its span takes
    at the smallest MaxStep, never under 256 MiB nor over the default."""
    # A fixed-step integrator steps at InitialStepSize whatever MaxStep says.
    steps = []
    for key, value in scenario.propagator_fields.items():
        if key.rpartition(".")[2].lower() not in ("maxstep", "initialstepsize"):
            continue
        try:
            step = float(value.strip().strip("'\""))
        except ValueError:
            continue
        if step > 0.0:
            steps.append(step)
    if not steps:
        return DEFAULT_FILE_SIZE_LIMIT_BYTES
    offset_s = abs(scenario.alignment.offset_s) if scenario.alignment is not None else 0.0
    rows = (offset_s + scenario.max_time_s) / min(steps) + 100.0
    wanted = int(20.0 * rows * _ROW_BYTES)
    return max(_FLOOR_FILE_SIZE_LIMIT_BYTES, min(DEFAULT_FILE_SIZE_LIMIT_BYTES, wanted))


_PDEATHSIG: list[str] | None = None
_PDEATHSIG_LOCK = threading.Lock()


def _pdeathsig_prefix() -> list[str]:
    """``setpriv --pdeathsig KILL --`` where it works, else nothing and a warning.

    util-linux before 2.33 has setpriv without --pdeathsig; every run would fail.
    """
    global _PDEATHSIG
    with _PDEATHSIG_LOCK:
        if _PDEATHSIG is None:
            setpriv = shutil.which("setpriv")
            prefix = [setpriv, "--pdeathsig", "KILL", "--"] if setpriv else []
            if prefix and subprocess.run(
                [*prefix, "true"], capture_output=True, check=False
            ).returncode != 0:
                prefix = []
            if not prefix:
                warnings.warn(
                    "setpriv --pdeathsig is not available, so a GmatConsole run outlives this "
                    "process if it is killed",
                    RuntimeWarning,
                    stacklevel=3,
                )
            _PDEATHSIG = prefix
        return list(_PDEATHSIG)


_PRLIMIT: list[str] | None = None
_PRLIMIT_LOCK = threading.Lock()


def _prlimit_binary() -> list[str]:
    """``prlimit`` where it runs, else nothing and a warning."""
    global _PRLIMIT
    with _PRLIMIT_LOCK:
        if _PRLIMIT is None:
            prlimit = shutil.which("prlimit")
            # Absolute: Popen runs it from GmatConsole's directory.
            found = [os.path.abspath(prlimit)] if prlimit else []
            if found and subprocess.run(
                [*found, "--core=0:0", "--", "true"], capture_output=True, check=False
            ).returncode != 0:
                found = []
            if not found:
                warnings.warn(
                    "prlimit is not available, so the file size cap is set after GmatConsole "
                    "starts and a child it forks first runs without one",
                    RuntimeWarning,
                    stacklevel=4,
                )
            _PRLIMIT = found
        return list(_PRLIMIT)


def _file_size_cap(file_size_limit_bytes: int, hard: int) -> int:
    """Never above a hard limit already in force, which an unprivileged process cannot raise."""
    return file_size_limit_bytes if hard == resource.RLIM_INFINITY else min(hard, file_size_limit_bytes)


def _limit_prefix(file_size_limit_bytes: int) -> list[str]:
    """``prlimit`` with the caps, so they hold from exec on for every child.

    Not preexec_fn, which is unsafe with threads; pipeline.py verifies in a
    ThreadPoolExecutor. The child inherits our hard limit.
    """
    prlimit = _prlimit_binary()
    if not prlimit:
        return []
    cap = _file_size_cap(file_size_limit_bytes, resource.getrlimit(resource.RLIMIT_FSIZE)[1])
    return [*prlimit, f"--fsize={cap}:{cap}", "--core=0:0", "--"]


def _limit_process(pid: int, file_size_limit_bytes: int) -> None:
    """The caps on a running pid, without prlimit: a child forked before this has none."""
    _, hard = resource.prlimit(pid, resource.RLIMIT_FSIZE)
    cap = _file_size_cap(file_size_limit_bytes, hard)
    resource.prlimit(pid, resource.RLIMIT_FSIZE, (cap, cap))
    resource.prlimit(pid, resource.RLIMIT_CORE, (0, 0))


def _kill_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def _drain(process: subprocess.Popen[str]) -> tuple[str | None, str | None]:
    """What the killed process wrote, waiting no longer than it takes to die."""
    try:
        return process.communicate(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return None, None


#: Runs in flight, so an interrupted pool can end them: GmatConsole has its
#: own session, and no Ctrl-C reaches it or the threads that wait on it.
_LIVE: set[subprocess.Popen[str]] = set()
_LIVE_LOCK = threading.Lock()
#: Threads between stop_runs and resume_runs; new runs are refused while any are.
_STOPPERS: set[int] = set()


def stop_runs() -> None:
    """Kill every GmatConsole run in flight and refuse new ones until this
    thread calls resume_runs. Calling it again only kills again. It acts on the
    whole process: one solve at a time."""
    with _LIVE_LOCK:
        _STOPPERS.add(threading.get_ident())
        for process in _LIVE:
            if process.returncode is None:  # not yet reaped, so its pid is still its own
                _kill_group(process)


def resume_runs() -> None:
    with _LIVE_LOCK:
        _STOPPERS.discard(threading.get_ident())


def run_gmat(
    gmat_console: str | Path,
    script: str | Path,
    workdir: str | Path,
    report_filenames: list[str],
    *,
    startup_file: str | Path | None = None,
    timeout_s: float = 300.0,
    on_started: Callable[[], None] | None = None,
    file_size_limit_bytes: int = DEFAULT_FILE_SIZE_LIMIT_BYTES,
    template_dir: str | Path | None = None,
    expected_sha256: str | None = None,
) -> GmatRunResult:
    """Run one script in GmatConsole; `on_started` is called once the process has run.

    No file it writes may grow past ``file_size_limit_bytes``; the kernel ends
    a run that tries with SIGXFSZ. ``template_dir`` is the organizer's folder:
    the data files the script names beside it are copied beside the run's copy.
    ``expected_sha256`` is the hash of the bytes the caller checked; the run's copy
    must be those bytes, or nothing runs.
    """
    if isinstance(file_size_limit_bytes, bool) or not isinstance(file_size_limit_bytes, int):
        raise TypeError(f"file_size_limit_bytes must be an int, not {file_size_limit_bytes!r}")
    # No file grows past 2**63 - 1 bytes, so a larger cap caps nothing.
    if not 0 <= file_size_limit_bytes < 1 << 63:
        raise ValueError(f"file_size_limit_bytes must be in [0, 2**63), not {file_size_limit_bytes}")
    if startup_file is None:
        # Without one there is no derived startup file, so no per-run
        # OUTPUT_PATH, so GMAT writes its reports wherever its own installation
        # points -- not into the run directory this function then searches. The
        # run fails with "expected fresh report was not created", which says
        # nothing about the cause. Every isolation guarantee here rests on the
        # derived startup file, so its absence is a refusal rather than a
        # confusing failure three steps later.
        raise GmatRunError(
            "run_gmat requires the startup file that matches the GmatConsole build: "
            "per-run output isolation and the environment fingerprint both come from it"
        )
    executable = Path(gmat_console).resolve()
    source_script = Path(script).resolve()
    if not executable.is_file():
        raise GmatRunError(f"GmatConsole does not exist: {executable}")
    if not source_script.is_file():
        raise GmatRunError(f"GMAT script does not exist: {source_script}")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:10]
    run_dir = Path(workdir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir / "gmat-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    local_script = run_dir / source_script.name
    if local_script.resolve() != source_script:
        shutil.copy2(source_script, local_script)
    if expected_sha256 is not None and file_sha256(local_script) != expected_sha256:
        # Between the caller's check and this copy: GMAT would run bytes nothing checked.
        raise GmatRunError(
            f"{source_script.name} is {file_sha256(local_script)} now, not the "
            f"{expected_sha256} that was checked, so GmatConsole was not started"
        )
    staged = (
        {} if template_dir is None else stage_template_files(
            local_script, template_dir, run_dir, frozenset({
                local_script.name, "gmat-output", "stdout.txt", "stderr.txt", "GmatLog.txt",
                "orbitwin_startup.txt",
            }),
        )
    )
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    log_path = run_dir / "GmatLog.txt"
    derived_startup: Path | None = None
    if startup_file is not None:
        derived_startup = derive_startup_file(
            startup_file,
            run_dir / "orbitwin_startup.txt",
            output_dir,
            log_path,
        )
    for filename in report_filenames:
        assert_safe_output_filename(filename)
        candidate = output_dir / Path(filename)
        if candidate.exists():
            candidate.unlink()

    command = [str(executable)]
    # Its own session (below) keeps a group signal from reaching it, so it is
    # told to die with us instead: SIGTERM ends Python with no exception to clean up on.
    if not os.access(executable, os.X_OK):
        # As Popen would say it, before setpriv turns it into an exit status.
        raise PermissionError(13, "Permission denied", str(executable))
    limit_prefix = _limit_prefix(file_size_limit_bytes)
    command = [*_pdeathsig_prefix(), *limit_prefix, *command]
    if derived_startup is not None:
        command.extend(["--startup_file", str(derived_startup)])
    command.extend(["--logfile", str(log_path), "--verbose", "off", "--run", str(local_script)])
    environment = os.environ.copy()
    library_dir = executable.parent.parent / "lib"
    if library_dir.is_dir():
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = str(library_dir) + (":" + existing if existing else "")
    start_wall = time.time()
    started_at = datetime.now(UTC).isoformat()
    error: str | None = None
    return_code = -1
    # Its own process group, so a timeout ends whatever it started too: a
    # child left holding the pipes kept communicate() waiting for ever.
    with _LIVE_LOCK:
        if _STOPPERS:
            raise GmatRunError(STOPPING_ERROR)
        process = subprocess.Popen(
            command,
            cwd=executable.parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Logged output only: a byte that is not UTF-8 must not end the run.
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        _LIVE.add(process)
    stdout: str | None
    stderr: str | None
    try:
        if not limit_prefix:
            _limit_process(process.pid, file_size_limit_bytes)
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            stdout, stderr = _drain(process)
            error = TIMEOUT_ERROR.format(timeout_s)
        else:
            return_code = int(process.returncode)
            # subprocess reports the signal itself, -25; a wrapping shell, 153.
            if return_code in (-signal.SIGXFSZ, 128 + signal.SIGXFSZ):
                error = FILE_SIZE_ERROR.format(file_size_limit_bytes)
    except BaseException:
        # An interrupt or a refused limit (without prlimit) must not leave GMAT running.
        _kill_group(process)
        _drain(process)
        raise
    finally:
        with _LIVE_LOCK:
            _LIVE.discard(process)
    if on_started is not None:
        on_started()
    stdout_path.write_text(_timeout_output_text(stdout), encoding="utf-8", errors="replace")
    stderr_path.write_text(_timeout_output_text(stderr), encoding="utf-8", errors="replace")
    duration = time.time() - start_wall

    combined = ""
    for path in (stdout_path, stderr_path, log_path):
        if path.exists():
            combined += "\n" + path.read_text(encoding="utf-8", errors="replace")
    if error is None and return_code != 0:
        error = EXIT_STATUS_ERROR.format(return_code)
    if error is None:
        lowered = combined.lower()
        marker = next((item for item in _FATAL_MARKERS if item.lower() in lowered), None)
        if marker:
            error = f"GMAT output contains fatal marker: {marker}"

    reports: dict[str, str] = {}
    report_hashes: dict[str, str] = {}
    partial: dict[str, str] = {}
    # A run killed mid-write, by the timeout or the size cap, may end on a torn row.
    if error is not None and not (
        error.startswith("GmatConsole timed out")
        or return_code in (-signal.SIGXFSZ, 128 + signal.SIGXFSZ)
    ):
        for filename in report_filenames:
            report_path = output_dir / filename
            if report_path.is_file() and report_path.stat().st_mtime >= start_wall - 1.0:
                partial[filename] = str(report_path)
    if error is None:
        for filename in report_filenames:
            report_path = output_dir / filename
            if not report_path.is_file() or report_path.stat().st_size == 0:
                error = NO_FRESH_REPORT_ERROR.format(filename)
                break
            if report_path.stat().st_mtime < start_wall - 1.0:
                error = STALE_REPORT_ERROR.format(filename)
                break
            reports[filename] = str(report_path)
            report_hashes[filename] = file_sha256(report_path)

    log_errors: list[str] = []
    if error is None:
        # GmatConsole exits 0 past an error it logged: the run is evidence, the record too.
        source = log_path if log_path.exists() else stdout_path
        log_errors = log_error_records(source.read_text(encoding="utf-8", errors="replace"))
        if log_errors:
            warnings.warn(
                f"WARNING: GmatConsole exited 0 and {source} records {len(log_errors)} error(s); "
                f"the first: {log_errors[0]}", RuntimeWarning, stacklevel=2,
            )
    return GmatRunResult(
        run_id=run_id,
        success=error is None,
        return_code=return_code,
        started_at_utc=started_at,
        duration_s=duration,
        workdir=str(run_dir),
        script_path=str(local_script),
        script_sha256=file_sha256(local_script),
        executable_sha256=file_sha256(executable),
        # The supplied file, not the derived copy. Hashing the copy recorded a
        # different value on every run -- measured 69469a0b and 6399743e for
        # two runs of the same script against the same startup file -- so the
        # field looked like an environment fingerprint and could never be
        # compared to one.
        startup_sha256=file_sha256(startup_file) if startup_file is not None else None,
        derived_startup_sha256=file_sha256(derived_startup) if derived_startup else None,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        log_path=str(log_path) if log_path.exists() else None,
        reports=reports,
        report_sha256=report_hashes,
        error=error,
        partial_reports=partial,
        data_file_sha256={
            **data_file_hashes(startup_file, executable),
            # A staged copy is recorded as the template's file, the same in every run.
            **{
                key: {**entry, "path": staged.get(entry["path"], entry["path"])}
                for key, entry in script_data_file_hashes(
                    local_script, startup_file, executable
                ).items()
            },
        },
        log_errors=log_errors,
    )
