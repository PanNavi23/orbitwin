"""Workspaces: one folder per contest day, driven by `orbitwin init`, `run` and `status`.

A workspace holds what `orbitwin day` otherwise takes as flags: orbitwin.toml (the team,
the rule set, GMAT), announce.toml (the values announced on the day) and input/ (the
organizer's script and the files that came with it). `orbitwin run` turns them into an
`orbitwin day` run with its day folders under days/ and the upload under upload/.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import math
import os
import re
import shlex
import sys
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .argtypes import positive_finite, positive_int
from .config import CLOSEST_APPROACH_WINDOWS
from .errors import OrbitWinError
from .models import file_sha256

FORMAT = 1
WORKSPACE_FILE = "orbitwin.toml"
ANNOUNCE_FILE = "announce.toml"
INPUT_DIR = "input"
DAYS_DIR = "days"
UPLOAD_DIR = "upload"
#: The rule set `orbitwin day` runs; init takes no other until day does.
DAY_RULES = "tasa-odc-2026/ranking"
SCORE_KEYS = ("kt_per_s", "ct_s", "kv_per_m_s", "cv_m_s", "cr_km")
#: announce.toml's [decisions]: day flags a person settles on the day.
DECISION_KEYS = ("closest_approach_window", "no_head_start_burns", "tmax_as_announced",
                 "report_filename_as_written", "accept_announced_horizon_before_perigee_s")
WORKSPACE_COMMANDS = ("init", "run", "status")
#: Exits: 0 done, 2 refused or stopped, 3 paused for a person (as `orbitwin day`).
EXIT_ERROR = 2


class WorkspaceError(OrbitWinError):
    """A workspace that cannot be used as it is; ``code`` names the kind for --json."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclasses.dataclass(frozen=True)
class Workspace:
    root: Path
    rules: str
    team_number: int
    pursuer_name: str | None = None
    target_name: str | None = None
    base: Path | None = None
    expect_commit: str | None = None
    gmat_console: Path | None = None
    startup_file: Path | None = None
    gmat_workers: int | None = None
    #: `orbitwin day --min-free-gb` for every run, unless the run gives its own.
    min_free_gb: float | None = None

    @property
    def days(self) -> Path:
        return self.root / DAYS_DIR

    @property
    def upload(self) -> Path:
        return self.root / UPLOAD_DIR


@dataclasses.dataclass(frozen=True)
class Announce:
    horizon_s: float | None = None
    mission_boundary_km: float | None = None
    score: dict[str, float] | None = None
    #: The score keys given, when only some are: run refuses, status names the rest.
    score_given: tuple[str, ...] = ()
    decisions: dict[str, Any] = dataclasses.field(default_factory=dict)

    def missing(self) -> list[str]:
        """What announce.toml still lacks for a full run, by key."""
        missing = [] if self.horizon_s is not None else ["horizon_s"]
        if self.score is None:
            missing += [f"score.{key}" for key in SCORE_KEYS if key not in self.score_given]
        return missing


# -- reading --------------------------------------------------------------------------


def _toml(path: Path, kind: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise WorkspaceError(f"{kind}-missing", f"{path} does not exist") from None
    except (OSError, ValueError) as exc:
        # ValueError: TOMLDecodeError, and also bytes that are not UTF-8 or an integer
        # too long to convert.
        raise WorkspaceError(f"{kind}-invalid", f"{path}: {exc}") from None


def _unknown(table: Mapping[str, Any], known: Sequence[str], where: str, code: str) -> None:
    extra = sorted(set(table) - set(known))
    if extra:
        raise WorkspaceError(code, f"{where}: unknown key {', '.join(extra)}; known: "
                             f"{', '.join(known)}")


def _number(value: Any, key: str, where: str, code: str, *, positive: bool) -> float:
    # bool is an int to Python, and never a number here.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WorkspaceError(code, f"{where}: {key} must be a number, not {value!r}")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise WorkspaceError(code, f"{where}: {key} must be a "
                             f"{'positive ' if positive else ''}finite number, not {value!r}")
    return number


def _text(value: Any, key: str, where: str, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkspaceError(code, f"{where}: {key} must be a non-empty string, not {value!r}")
    return value


def _path(value: Any, key: str, where: str, root: Path, *, resolve: bool = True) -> Path:
    text = _text(value, key, where, "workspace-invalid")
    path = Path(text).expanduser()
    path = path if path.is_absolute() else root / path
    # GMAT paths stay as written, symlinks and ".." and all, as `orbitwin day` records them:
    # ".." after a link is the kernel's to resolve, not ours.
    return path.resolve() if resolve else path


def read_workspace(folder: str | os.PathLike[str]) -> Workspace:
    """orbitwin.toml of ``folder``, checked; WorkspaceError on anything off."""
    root = Path(folder).expanduser().resolve()
    file = root / WORKSPACE_FILE
    if not root.is_dir():
        raise WorkspaceError("workspace-missing", f"{root} is not a folder; create a "
                             "workspace with `orbitwin init`")
    if not file.is_file():
        raise WorkspaceError("workspace-missing", f"{root} has no {WORKSPACE_FILE}; create "
                             f"one with `orbitwin init {shlex.quote(str(root))} "
                             "--team-number <n>`")
    data = _toml(file, "workspace")
    where = str(file)
    _unknown(data, ("format", "rules", "team_number", "pursuer_name", "target_name", "base",
                    "expect_commit", "min_free_gb", "gmat"), where, "workspace-invalid")
    if data.get("format") != FORMAT:
        raise WorkspaceError("workspace-invalid",
                             f"{where}: format is {data.get('format')!r}, not {FORMAT}")
    rules = _text(data.get("rules"), "rules", where, "workspace-invalid")
    team = data.get("team_number")
    if isinstance(team, bool) or not isinstance(team, int) or team < 1:
        raise WorkspaceError("workspace-invalid",
                             f"{where}: team_number must be a whole number of 1 or more, "
                             f"not {team!r}")
    gmat = data.get("gmat", {})
    if not isinstance(gmat, dict):
        raise WorkspaceError("workspace-invalid", f"{where}: [gmat] must be a table")
    _unknown(gmat, ("console", "startup_file", "workers"), f"{where} [gmat]",
             "workspace-invalid")
    console = None if "console" not in gmat else _path(gmat["console"], "gmat.console",
                                                       where, root, resolve=False)
    startup = None if "startup_file" not in gmat else _path(
        gmat["startup_file"], "gmat.startup_file", where, root, resolve=False)
    if (console is None) != (startup is None):
        raise WorkspaceError("workspace-invalid", f"{where}: give gmat.console and "
                             "gmat.startup_file together, or neither (GMAT_CONSOLE and "
                             "GMAT_STARTUP then)")
    workers = gmat.get("workers")
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int)
                                or workers < 1):
        raise WorkspaceError("workspace-invalid", f"{where}: gmat.workers must be a whole "
                             f"number of 1 or more, not {workers!r}")
    optional = {key: _text(data[key], key, where, "workspace-invalid")
                for key in ("pursuer_name", "target_name", "expect_commit") if key in data}
    if "expect_commit" in optional and not COMMIT.fullmatch(optional["expect_commit"]):
        raise WorkspaceError("workspace-invalid", f"{where}: expect_commit "
                             f"{optional['expect_commit']!r} is not a commit (7 to 40 hex "
                             "digits)")
    min_free_gb = None if "min_free_gb" not in data else _number(
        data["min_free_gb"], "min_free_gb", where, "workspace-invalid", positive=True)
    return Workspace(
        root=root, rules=rules, team_number=team, min_free_gb=min_free_gb,
        base=None if "base" not in data else _path(data["base"], "base", where, root),
        gmat_console=console, startup_file=startup, gmat_workers=workers, **optional)


def read_announce(workspace: Workspace) -> Announce:
    """announce.toml, checked; a value left commented out is not announced yet."""
    file = workspace.root / ANNOUNCE_FILE
    code = "announce-invalid"
    data = _toml(file, "announce")
    where = str(file)
    _unknown(data, ("horizon_s", "mission_boundary_km", "score", "decisions"), where, code)
    numbers = {key: _number(data[key], key, where, code, positive=True)
               for key in ("horizon_s", "mission_boundary_km") if key in data}
    score_table = data.get("score", {})
    decisions = data.get("decisions", {})
    for name, table in (("score", score_table), ("decisions", decisions)):
        if not isinstance(table, dict):
            raise WorkspaceError(code, f"{where}: [{name}] must be a table")
    _unknown(score_table, SCORE_KEYS, f"{where} [score]", code)
    _unknown(decisions, DECISION_KEYS, f"{where} [decisions]", code)
    score = {key: _number(score_table[key], f"score.{key}", where, code,
                          positive=key == "cr_km")
             for key in SCORE_KEYS if key in score_table}
    checked: dict[str, Any] = {}
    for key, value in decisions.items():
        if key in ("no_head_start_burns", "tmax_as_announced"):
            if not isinstance(value, bool):
                raise WorkspaceError(code, f"{where}: decisions.{key} must be true or false, "
                                     f"not {value!r}")
        elif key == "closest_approach_window":
            if value not in CLOSEST_APPROACH_WINDOWS:
                raise WorkspaceError(code, f"{where}: decisions.{key} must be one of "
                                     f"{', '.join(CLOSEST_APPROACH_WINDOWS)}, not {value!r}")
        elif key == "report_filename_as_written":
            value = _text(value, f"decisions.{key}", where, code)
        else:
            value = _number(value, f"decisions.{key}", where, code, positive=True)
        checked[key] = value
    return Announce(
        horizon_s=numbers.get("horizon_s"),
        mission_boundary_km=numbers.get("mission_boundary_km"),
        score=score if len(score) == len(SCORE_KEYS) else None,
        score_given=tuple(key for key in SCORE_KEYS if key in score),
        decisions=checked)


@dataclasses.dataclass(frozen=True)
class Inputs:
    script: Path
    companions: tuple[str, ...]


def find_inputs(workspace: Workspace) -> Inputs:
    """The one .script at the top of input/; every other file under it goes with it."""
    folder = workspace.root / INPUT_DIR
    if not folder.is_dir():
        raise WorkspaceError("template-missing", f"{folder} does not exist")
    if not os.access(folder, os.R_OK | os.X_OK):
        raise WorkspaceError("io", f"{folder} cannot be read")
    visible = [path for path in folder.rglob("*")
               if not any(part.startswith(".") for part in path.relative_to(folder).parts)]
    for path in visible:
        if path.is_symlink() and path.is_dir():
            raise WorkspaceError("input-name", f"{path} is a link to a folder, whose files "
                                 "would not go with the script; copy the folder in instead")
        # rglob passes over a folder it cannot read, and its files with it.
        if path.is_dir() and not os.access(path, os.R_OK | os.X_OK):
            raise WorkspaceError("io", f"{path} cannot be read")
    files = sorted(path for path in visible if path.is_file())
    for path in files:
        if not os.access(path, os.R_OK):
            raise WorkspaceError("io", f"{path} cannot be read")
        try:
            path.relative_to(folder).as_posix().encode("utf-8")
        except UnicodeEncodeError:
            raise WorkspaceError("input-name", f"{path!s} has a file name that is not UTF-8 "
                                 "(a Windows archive's, often); rename it") from None
    scripts = [path for path in files
               if path.parent == folder and path.suffix.lower() == ".script"]
    if len(scripts) != 1:
        raise WorkspaceError(
            "template-missing" if not scripts else "template-ambiguous",
            f"{folder} must hold exactly one .script, the organizer's; it holds "
            f"{len(scripts)}" + (f": {', '.join(path.name for path in scripts)}"
                                 if scripts else ""))
    if scripts[0].is_symlink():
        # The day reads the files beside the script's target, not these.
        raise WorkspaceError("input-name", f"{scripts[0]} is a link; copy the script itself "
                             "into input/, as the files beside it go with it")
    return Inputs(script=scripts[0], companions=tuple(
        path.relative_to(folder).as_posix() for path in files if path != scripts[0]))


# -- day folders ------------------------------------------------------------------------


def _state(folder: Path) -> dict[str, Any] | None:
    """state.json, or None when it is not one `orbitwin day` wrote."""
    from .day import STATE_FORMAT

    try:
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or state.get("format") != STATE_FORMAT or not all(
            isinstance(state.get(key), dict) for key in ("inputs", "official", "gmat")):
        return None
    return state


#: A commit as `orbitwin day --expect-commit` takes it.
COMMIT = re.compile(r"[0-9a-fA-F]{7,40}")
#: A day folder's name, the UTC second `orbitwin day` made it.
DAY_NAME = re.compile(r"\d{8}T\d{6}Z")


def day_folders(workspace: Workspace) -> list[Path]:
    """The day folders under days/, oldest first; anything else there is not one."""
    if not workspace.days.is_dir():
        return []
    # One without state.json is kept: a day stopped before writing it, and it is unreadable.
    return sorted(path for path in workspace.days.iterdir()
                  if DAY_NAME.fullmatch(path.name) and path.is_dir())


def delivered(state: Mapping[str, Any]) -> bool:
    """Whether this day folder's last upload is where latest points."""
    uploads = state.get("uploads")
    last = uploads[-1] if isinstance(uploads, list) and uploads else None
    folder = last.get("folder") if isinstance(last, dict) else None
    if not isinstance(folder, str) or not os.path.isabs(folder):
        return False
    target = Path(folder)
    latest = target.parent / "latest"
    return latest.is_symlink() and os.readlink(latest) == target.name


# -- the day's argv -------------------------------------------------------------------


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _number_text(value: float) -> str:
    # repr is exact: the day records the float the flag parses back to.
    return repr(float(value))


def fixed_argv(workspace: Workspace, announce: Announce, inputs: Inputs) -> list[str]:
    """`orbitwin day` arguments a day folder records on its first run."""
    if announce.horizon_s is None:
        raise WorkspaceError("announce-incomplete", f"{workspace.root / ANNOUNCE_FILE} gives "
                             "no horizon_s (T_max, in s); a day cannot start without it")
    # --flag=value throughout: a value may start with "-".
    argv = [str(inputs.script)]
    for companion in inputs.companions:
        argv.append(f"--companion={companion}")
    argv += [f"--horizon-s={_number_text(announce.horizon_s)}",
             f"--team-number={workspace.team_number}",
             f"--work-root={workspace.days}", f"--final-root={workspace.upload}"]
    if announce.mission_boundary_km is not None:
        argv.append(f"--mission-boundary-km={_number_text(announce.mission_boundary_km)}")
    for key in ("pursuer_name", "target_name", "expect_commit"):
        value = getattr(workspace, key)
        if value is not None:
            argv.append(f"{_flag(key)}={value}")
    if workspace.base is not None:
        argv.append(f"--base={workspace.base}")
    if workspace.gmat_console is not None and workspace.startup_file is not None:
        argv += [f"--gmat-console={workspace.gmat_console}",
                 f"--startup-file={workspace.startup_file}"]
    if workspace.gmat_workers is not None:
        argv.append(f"--gmat-workers={workspace.gmat_workers}")
    for key, value in announce.decisions.items():
        if value is True:
            argv.append(_flag(key))
        elif value is not False:
            argv.append(f"{_flag(key)}="
                        f"{value if isinstance(value, str) else _number_text(value)}")
    return argv


def score_argv(announce: Announce) -> list[str]:
    if announce.score_given and announce.score is None:
        given = set(announce.score_given)
        raise WorkspaceError("announce-incomplete", "announce.toml [score] gives "
                             f"{', '.join(announce.score_given)} but not "
                             f"{', '.join(key for key in SCORE_KEYS if key not in given)}: "
                             "all five, or none until they are announced")
    if announce.score is None:
        return []
    return [f"{_flag(key)}={_number_text(announce.score[key])}" for key in SCORE_KEYS]


def _day_parser() -> argparse.ArgumentParser:
    from .day import _parser_arguments

    parser = argparse.ArgumentParser(prog="orbitwin day", allow_abbrev=False,
                                     exit_on_error=False)
    _parser_arguments(parser)
    return parser


def day_namespace(argv: list[str]) -> argparse.Namespace:
    """``argv`` parsed as `orbitwin day` parses it, its usage errors a WorkspaceError."""
    parser = _day_parser()
    said = io.StringIO()
    try:
        with contextlib.redirect_stderr(said):
            return parser.parse_args(argv)
    except argparse.ArgumentError as exc:
        reason = str(exc)
    except SystemExit:
        # Python 3.11 still exits on some errors; the reason is the last line it printed.
        lines = said.getvalue().strip().splitlines()
        reason = lines[-1] if lines else "refused"
    raise WorkspaceError("announce-invalid", f"orbitwin day refuses the workspace's "
                         f"values: {reason}")


def _recorded_differences(state: Mapping[str, Any], fixed: argparse.Namespace,
                          inputs: Inputs) -> list[str]:
    """What the workspace now says that the day folder recorded otherwise."""
    recorded = state.get("inputs") or {}
    differences = []
    given = {key: getattr(fixed, key) for key in SCORE_KEYS}
    if recorded.get("score") is not None and None not in given.values() and (
            recorded["score"] != given):
        # The day refuses this too; said here, status says it before a run.
        differences.append(f"score: recorded {recorded['score']}, now {given}")
    for key in ("horizon_s", "tmax_as_announced", "mission_boundary_km",
                "closest_approach_window", "team_number", "pursuer_name", "target_name",
                "gmat_workers", "report_filename_as_written",
                "accept_announced_horizon_before_perigee_s", "no_head_start_burns"):
        if recorded.get(key) != getattr(fixed, key):
            differences.append(f"{key}: recorded {recorded.get(key)!r}, now "
                               f"{getattr(fixed, key)!r}")
    base = None if fixed.base is None else str(Path(fixed.base).resolve())
    if recorded.get("base") != base:
        differences.append(f"base: recorded {recorded.get('base')!r}, now {base!r}")
    expect = None if fixed.expect_commit is None else fixed.expect_commit.strip().lower()
    if recorded.get("expect_commit") != expect:
        differences.append(f"expect_commit: recorded {recorded.get('expect_commit')!r}, "
                           f"now {expect!r}")
    official = state.get("official") or {}
    if official.get("sha256") != file_sha256(inputs.script):
        differences.append(f"the script: recorded SHA-256 {official.get('sha256')}, "
                           f"{inputs.script.name} now {file_sha256(inputs.script)}")
    companions = {rel: file_sha256(inputs.script.parent / rel) for rel in inputs.companions}
    if (official.get("companions") or {}) != companions:
        differences.append("the files beside the script in input/ are not the ones recorded")
    final_root = str(Path(fixed.final_root).expanduser().resolve())
    if state.get("final_root") != final_root:
        differences.append(f"the upload folder: recorded {state.get('final_root')}, the "
                           f"workspace's is {final_root} (a copied or moved workspace)")
    gmat = state.get("gmat") or {}
    if fixed.gmat_console is not None and (
            gmat.get("console") != str(Path(fixed.gmat_console).expanduser().absolute())
            or gmat.get("startup") != str(Path(fixed.startup_file).expanduser().absolute())):
        differences.append(f"gmat: recorded {gmat.get('console')} with {gmat.get('startup')}")
    return differences


@dataclasses.dataclass(frozen=True)
class Plan:
    """What `orbitwin run` does: start a day folder, or go on with one."""

    argv: list[str]
    resume: Path | None


RUN_FLAGS = ("confirm_readback", "by_eye_passed", "exclude", "include",
             "accept_config_warnings", "accept_rank_warnings", "force_latest",
             "allow_non_main", "min_free_gb")


def _run_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    for key in RUN_FLAGS:
        value = getattr(args, key, None)
        if value is None or value is False:
            continue
        if value is True:
            argv.append(_flag(key))
        elif isinstance(value, list):
            for item in value:
                argv += [_flag(key), str(item)]
        else:
            argv += [_flag(key), _number_text(value) if isinstance(value, float) else value]
    return argv


def _unreadable(workspace: Workspace, newest: Path, folders: Sequence[Path]) -> str:
    """Why the newest day folder cannot go on, and the ways forward."""
    new = f"`{_new_command(workspace)}` starts a new day folder"
    if (newest / "state.json").exists():
        return f"{newest}/state.json cannot be read; {new}"
    if _day_lock_held(newest):
        return (f"{newest} holds no state.json yet and an `orbitwin day` holds its lock: "
                "a day is starting in it; wait for it")
    here = shlex.quote(str(workspace.root))
    back = ("" if len(folders) < 2 else
            f"; once it is moved out of {workspace.days}, `orbitwin run {here}` takes up "
            f"{folders[-2].name} again, if that one can go on (`orbitwin status {here}` "
            "says)")
    return (f"{newest} holds no state.json: the day stopped before it recorded anything, "
            f"so nothing in it is finished{back}. Or {new}")


def _day_lock_held(folder: Path) -> bool:
    """Whether a process holds ``folder``'s .day.lock now; creates nothing."""
    import fcntl

    try:
        handle = (folder / ".day.lock").open("r")
    except OSError:
        return False
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return True
    return False


def _new_command(workspace: Workspace) -> str:
    """The command that starts a new day folder in ``workspace``, as it can be pasted."""
    return (f"orbitwin run --new {shlex.quote(str(workspace.root))}"
            + (" --allow-non-main" if workspace.expect_commit is None else ""))


def plan_run(workspace: Workspace, args: argparse.Namespace) -> Plan:
    """The `orbitwin day` argv for this run: a resume of the newest unfinished day folder,
    or a new one when there is none or --new is given."""
    if workspace.rules != DAY_RULES:
        raise WorkspaceError("rules-unsupported", f"{workspace.root / WORKSPACE_FILE} names "
                             f"{workspace.rules}; `orbitwin day` runs {DAY_RULES} only so far")
    announce = read_announce(workspace)
    inputs = find_inputs(workspace)
    scores = score_argv(announce)
    fixed = fixed_argv(workspace, announce, inputs)
    run = _run_argv(args)
    if getattr(args, "min_free_gb", None) is None and workspace.min_free_gb is not None:
        run += ["--min-free-gb", _number_text(workspace.min_free_gb)]
    folders = day_folders(workspace)
    newest = folders[-1] if folders else None
    state = None if newest is None else _state(newest)
    if newest is not None and not args.new:
        if state is None:
            raise WorkspaceError("day-unreadable", _unreadable(workspace, newest, folders))
        if delivered(state):
            raise WorkspaceError("day-delivered", f"{newest} already delivered "
                                 f"{state['uploads'][-1]['folder']}, and latest points at "
                                 f"it; `{_new_command(workspace)}` starts another day folder")
        differences = _recorded_differences(state, day_namespace([*fixed, *scores]), inputs)
        if differences:
            raise WorkspaceError(
                "day-inputs-changed", f"{newest} was started with other values than the "
                "workspace holds now:\n  " + "\n  ".join(differences)
                + "\nA day folder keeps what it started with. Put the values back, or "
                f"start a new day folder with `{_new_command(workspace)}` (finished steps "
                "are not carried over).")
        argv = ["--resume", str(newest), *scores, *run]
        day_namespace(argv)
        return Plan(argv=argv, resume=newest)
    if workspace.expect_commit is None and not getattr(args, "allow_non_main", False):
        raise WorkspaceError(
            "commit-unset", f"{workspace.root / WORKSPACE_FILE} sets no expect_commit: a day "
            "runs only the commit chosen the day before (docs/RUNBOOK.md step 0). Set it to "
            "the commit `orbitwin --version` or `git rev-parse HEAD` names, or pass "
            "--allow-non-main, which the upload's note records")
    argv = [*fixed, *scores, *run]
    day_namespace(argv)
    return Plan(argv=argv, resume=None)


# -- status -----------------------------------------------------------------------------


def _table(state: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = state.get(key)
    return value if isinstance(value, dict) else {}


def _pending_readbacks(state: Mapping[str, Any]) -> list[str]:
    """Readbacks the day printed at a pause and nobody has confirmed since."""
    confirmed = _table(state, "confirmed_readbacks")
    return sorted(str(step) for step, digest in _table(state, "shown_readbacks").items()
                  if not isinstance(confirmed.get(step), dict)
                  or confirmed[step].get("sha256") != digest)


def _warnings_to_accept(state: Mapping[str, Any]) -> list[str]:
    """--accept-* flags a stop of this day folder asked for and no run has given since."""
    flags = []
    shown, accepted = _table(state, "shown_config_warnings"), _table(state,
                                                                      "accepted_config_warnings")
    if any(warnings and accepted.get(name) != warnings for name, warnings in shown.items()):
        flags.append("--accept-config-warnings")
    rank = state.get("shown_rank_warnings")
    if rank and state.get("accepted_rank_warnings") != rank:
        flags.append("--accept-rank-warnings")
    return flags


def _day_summary(folder: Path) -> dict[str, Any]:
    state = _state(folder)
    if state is None:
        return {"folder": str(folder), "readable": False, "created_utc": None, "steps": {},
                "readbacks_confirmed": [], "readbacks_pending": [],
                "warnings_to_accept": [], "uploads": [],
                "delivered": False, "score_given": False, "allow_non_main": False}
    uploads = state.get("uploads")
    return {
        "folder": str(folder),
        "readable": True,
        "created_utc": state.get("created_utc"),
        "steps": {str(name): record.get("exit")
                  for name, record in _table(state, "steps").items()
                  if isinstance(record, dict)},
        "readbacks_confirmed": sorted(_table(state, "confirmed_readbacks")),
        "readbacks_pending": _pending_readbacks(state),
        "warnings_to_accept": _warnings_to_accept(state),
        "uploads": [{key: upload.get(key) for key in ("folder", "candidate", "score")}
                    for upload in (uploads if isinstance(uploads, list) else [])
                    if isinstance(upload, dict)],
        "delivered": delivered(state),
        "score_given": state["inputs"].get("score") is not None,
        "allow_non_main": state.get("allow_non_main") is True,
    }


def _latest(workspace: Workspace) -> dict[str, Any] | None:
    latest = workspace.upload / "latest"
    if not latest.is_symlink():
        return None
    target = workspace.upload / os.readlink(latest)
    from .day import NOTE_NAME

    files = {}
    note = target / NOTE_NAME
    try:
        text = note.read_text(encoding="utf-8", errors="replace") if note.is_file() else ""
    except OSError as exc:
        return {"folder": str(target), "error": f"{type(exc).__name__}: {exc}"}
    if text:
        for line in text.splitlines():
            key, _, value = line.partition(" : ")
            if key.strip() in ("script", "report"):
                files[key.strip()] = str(target / value.strip())
    return {"folder": str(target), **files}


def status(workspace: Workspace) -> dict[str, Any]:
    """Everything `orbitwin status` reports, and the next thing to do."""
    problems: list[dict[str, str]] = []
    announce: Announce | None = None
    inputs: Inputs | None = None
    try:
        announce = read_announce(workspace)
    except WorkspaceError as exc:
        problems.append({"code": exc.code, "message": str(exc)})
    try:
        inputs = find_inputs(workspace)
    except WorkspaceError as exc:
        problems.append({"code": exc.code, "message": str(exc)})
    except OSError as exc:
        problems.append({"code": "io", "message": f"{type(exc).__name__}: {exc}"})
    days = [_day_summary(folder) for folder in day_folders(workspace)]
    missing = [] if announce is None else announce.missing()
    if not problems and "horizon_s" not in missing and not (days and days[-1]["delivered"]):
        # What `orbitwin run` would refuse, said here rather than at the run.
        try:
            plan_run(workspace, argparse.Namespace(new=False))
        except WorkspaceError as exc:
            if exc.code != "day-unreadable":
                problems.append({"code": exc.code, "message": str(exc)})
        except OSError as exc:
            problems.append({"code": "io", "message": f"{type(exc).__name__}: {exc}"})
    running = _running(workspace)
    here = shlex.quote(str(workspace.root))
    if running:
        next_step = "an `orbitwin run` is running on this workspace; wait for it"
    elif problems:
        next_step = "fix the problems listed"
    elif days and not days[-1]["readable"]:
        next_step = _unreadable(workspace, Path(days[-1]["folder"]),
                                [Path(day["folder"]) for day in days])
    elif days and days[-1]["delivered"]:
        next_step = (f"upload the two files latest points at; `{_new_command(workspace)}` "
                     "starts another day folder")
    elif "horizon_s" in missing:
        next_step = f"give horizon_s (T_max) in {ANNOUNCE_FILE}, then `orbitwin run {here}`"
    elif days:
        day = days[-1]
        non_main = " --allow-non-main" if day["allow_non_main"] else ""
        name = Path(day["folder"]).name
        accept = "".join(f" {flag}" for flag in day["warnings_to_accept"])
        read = " and the WARNING lines its last stop printed" if accept else ""
        if day["readbacks_pending"]:
            step = day["readbacks_pending"][0]
            next_step = (f"two people read step {step}'s readback{read} ({name}/logs), then "
                         f"`orbitwin run {here}{non_main}{accept} --confirm-readback {step}`")
        elif accept:
            next_step = (f"two people read the warnings the last stop printed (the "
                         f"files it named, and {name}/logs), then "
                         f"`orbitwin run {here}{non_main}{accept}`")
        else:
            next_step = f"`orbitwin run {here}{non_main}` goes on with {name}"
            if not day["score_given"] and "score.kt_per_s" in missing:
                next_step += (f"; it pauses for the coefficients until [score] in "
                              f"{ANNOUNCE_FILE} has them")
        next_step += ("; a stop it printed may name flags to add, as the last lines of "
                      "its run say")
    else:
        next_step = f"`orbitwin run {here}` starts the day"
    return {
        "workspace": str(workspace.root),
        "rules": workspace.rules,
        "team_number": workspace.team_number,
        "script": None if inputs is None else str(inputs.script),
        "companions": [] if inputs is None else list(inputs.companions),
        "announce_missing": missing,
        "problems": problems,
        "days": days,
        "latest": _latest(workspace),
        "running": running,
        "next": next_step,
    }


def status_lines(report: Mapping[str, Any]) -> Iterator[str]:
    yield f"workspace : {report['workspace']}"
    yield f"rules     : {report['rules']}, team {report['team_number']}"
    yield f"script    : {report['script'] or 'none'}"
    if report["companions"]:
        yield f"with it   : {', '.join(report['companions'])}"
    yield ("announced : everything" if not report["announce_missing"]
           else f"not yet   : {', '.join(report['announce_missing'])}")
    for problem in report["problems"]:
        yield f"PROBLEM [{problem['code']}]: {problem['message']}"
    for day in report["days"]:
        steps = " ".join(f"{name}={'ok' if code == 0 else code}"
                         for name, code in day["steps"].items()) or "no step yet"
        if not day["readable"]:
            steps = "state.json cannot be read"
        yield f"day {Path(day['folder']).name}: {steps}"
        if day["readbacks_confirmed"]:
            yield f"  readbacks confirmed: {', '.join(day['readbacks_confirmed'])}"
        for upload in day["uploads"]:
            what = ("recovered" if upload["candidate"] is None
                    else f"{upload['candidate']}, score {upload['score']}")
            yield (f"  upload {upload['folder']} ({what})"
                   + (", latest" if day["delivered"] else ""))
    latest = report["latest"]
    if latest is not None:
        yield f"latest    : {latest['folder']}"
        if "error" in latest:
            yield f"  its note cannot be read: {latest['error']}"
        for key in ("script", "report"):
            if key in latest:
                yield f"  {key:7} {latest[key]}"
    yield f"next      : {report['next']}"


# -- init -------------------------------------------------------------------------------


def _toml_string(value: object) -> str:
    """A TOML basic string: JSON's escapes, and DEL, which TOML refuses bare too."""
    return json.dumps(str(value), ensure_ascii=False).replace("\x7f", "\\u007f")


def _workspace_text(rules: str, team_number: int, console: Path | None,
                    startup: Path | None, expect_commit: str | None = None) -> str:
    gmat = (f"console = {_toml_string(console)}\nstartup_file = {_toml_string(startup)}\n"
            if console is not None
            else '# Both, or neither: then the GMAT_CONSOLE and GMAT_STARTUP variables.\n'
                 '# console = "/path/to/GmatConsole"\n'
                 '# startup_file = "/path/to/gmat_startup_file.txt"\n')
    commit = (f"expect_commit = {json.dumps(expect_commit)}  # chosen the day before "
              "(docs/RUNBOOK.md step 0)" if expect_commit is not None
              else '# expect_commit = "<commit>"  # needed to start a day: the commit chosen '
              "the day before (docs/RUNBOOK.md step 0)")
    return f"""\
# OrbitWin workspace, written by `orbitwin init`. `orbitwin run <this folder>` reads it,
# announce.toml and the organizer's script in {INPUT_DIR}/.
format = {FORMAT}
rules = {json.dumps(rules)}
team_number = {team_number}
# pursuer_name = "SC_Human"     # only when the template names its craft otherwise
# target_name = "SC_Alien"
# base = "profile.yaml"         # the profile step 2 copies; the rule set's own by default
{commit}
# min_free_gb = 20.0            # free disk a run needs, in GB; orbitwin day's default is 20

[gmat]
{gmat}# workers = 4    # GmatConsole runs at once
"""


ANNOUNCE_TEXT = """\
# The values announced on the day, in the units named. Delete the leading "# " of a
# line to give its value; leave it as it is while the value is not announced.
# `orbitwin status <workspace>` says what is still missing.

# T_max in s, as announced; needed before `orbitwin run`.
# horizon_s = 86400.0

# The mission boundary radius in km from the Earth's centre, if announced.
# mission_boundary_km = 50000.0

# The score coefficients: all five, or none while they are not announced. With none
# the run stops after the safety nets (docs/RUNBOOK.md step 4a); fill them in and
# `orbitwin run` again.
[score]
# kt_per_s = 0.001
# ct_s = 80000.0
# kv_per_m_s = 0.01
# cv_m_s = 2000.0
# cr_km = 1000.0

# Decisions a person makes on the day (docs/RUNBOOK.md).
[decisions]
# closest_approach_window = "after_alignment"  # or "from_origin"; needed when the craft start apart
# no_head_start_burns = true                   # no burn before the later craft's epoch
# tmax_as_announced = true                     # step 2: T_max kept although off the boundary
# report_filename_as_written = "Name.txt"      # only after the organizer's written answer
# accept_announced_horizon_before_perigee_s = 86400.0  # only after a written answer
"""


def init_workspace(folder: str | os.PathLike[str], rules: str, team_number: int,
                   console: str | None, startup: str | None,
                   expect_commit: str | None = None) -> Workspace:
    """Write a new workspace into ``folder``; one already there is left alone."""
    from .rules import DuelRules, rule_set

    try:
        chosen = rule_set(rules)
    except KeyError as exc:
        raise WorkspaceError("rules-unknown", str(exc.args[0])) from None
    if isinstance(chosen, DuelRules) or chosen.id != DAY_RULES:
        raise WorkspaceError("rules-unsupported", f"{chosen.id}: `orbitwin run` drives "
                             f"`orbitwin day`, which runs {DAY_RULES} only so far; the other "
                             "commands take any rule set (`orbitwin rules`)")
    if console == "" or startup == "":
        raise WorkspaceError("usage", "--gmat-console or --startup-file is empty (an "
                             "unset variable?); give both paths, or neither")
    if (console is None) != (startup is None):
        raise WorkspaceError("usage", "give --gmat-console and --startup-file together, "
                             "or neither")
    # Absolute, not resolved: `orbitwin day` records GmatConsole as given.
    paths = [None if text is None else Path(text).expanduser().absolute()
             for text in (console, startup)]
    for path in paths:
        if path is None:
            continue
        if not path.is_file():
            raise WorkspaceError("gmat-missing", f"{path} is not a file")
        try:
            str(path).encode("utf-8")
        except UnicodeEncodeError:
            raise WorkspaceError("usage", f"{path!s} is not a UTF-8 path, which "
                                 f"{WORKSPACE_FILE} cannot hold") from None
    if expect_commit is not None and not COMMIT.fullmatch(expect_commit):
        raise WorkspaceError("usage", f"--expect-commit {expect_commit!r} is not a commit "
                             "(7 to 40 hex digits)")
    root = Path(folder).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise WorkspaceError("workspace-exists", f"{root} exists and is not a folder")
    from .day import GUARDED

    if root == GUARDED or GUARDED in root.parents:
        raise WorkspaceError("usage", f"{root} is under {GUARDED}, whose idle files the disk "
                             "guard compresses and `orbitwin day` refuses; name a folder "
                             "outside it")
    taken = [name for name in (WORKSPACE_FILE, ANNOUNCE_FILE) if (root / name).exists()]
    if taken:
        raise WorkspaceError("workspace-exists", f"{root} already holds {', '.join(taken)}; "
                             "nothing was changed")
    texts = {ANNOUNCE_FILE: ANNOUNCE_TEXT,
             WORKSPACE_FILE: _workspace_text(chosen.id, team_number, paths[0], paths[1],
                                             expect_commit)}
    for text in texts.values():
        try:
            tomllib.loads(text)
        except ValueError as exc:
            raise WorkspaceError("usage", f"a value given cannot be written to "
                                 f"{WORKSPACE_FILE}: {exc}") from None
    for name in (INPUT_DIR, DAYS_DIR, UPLOAD_DIR):
        (root / name).mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        # orbitwin.toml last: a workspace with it is whole. "x" never overwrites.
        for name, text in texts.items():
            with (root / name).open("x", encoding="utf-8") as handle:
                written.append(root / name)
                handle.write(text)
    except FileExistsError:
        for path in written:
            path.unlink(missing_ok=True)
        raise WorkspaceError("workspace-exists", f"{root} gained a workspace file while "
                             "init ran; nothing of it was kept") from None
    except OSError:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return read_workspace(root)


# -- the commands ------------------------------------------------------------------------


def add_workspace_parsers(sub: Any) -> None:
    from .day import CANDIDATES, READBACKS

    init = sub.add_parser("init", allow_abbrev=False,
                          help="Create a workspace: a folder for one contest day")
    init.add_argument("folder")
    init.add_argument("--team-number", type=positive_int, required=True,
                      help="the team's number, for the official report's file name")
    init.add_argument("--rules", default=DAY_RULES, help=f"a rule set's id (default {DAY_RULES})")
    init.add_argument("--gmat-console", help="GmatConsole, with --startup-file")
    init.add_argument("--startup-file", help="its stock startup file, with --gmat-console")
    init.add_argument("--expect-commit",
                      help="the commit chosen the day before (docs/RUNBOOK.md step 0); "
                      "needed to start a day, settable later in orbitwin.toml")
    init.add_argument("--json", action="store_true", help="print JSON")

    run = sub.add_parser("run", allow_abbrev=False,
                         help="Run the day (orbitwin day) on a workspace, or go on with it")
    run.add_argument("folder")
    run.add_argument("--new", action="store_true",
                     help="start a new day folder although the newest is unfinished")
    run.add_argument("--json", action="store_true",
                     help="the day's own output to stderr, a JSON summary on stdout")
    run.add_argument("--confirm-readback", action="append", choices=READBACKS,
                     help="two people read this step's readback; as orbitwin day")
    run.add_argument("--by-eye-passed", action="append", choices=CANDIDATES)
    run.add_argument("--exclude", action="append", choices=CANDIDATES)
    run.add_argument("--include", action="append", choices=CANDIDATES)
    run.add_argument("--accept-config-warnings", action="store_true")
    run.add_argument("--accept-rank-warnings", action="store_true")
    run.add_argument("--force-latest", action="store_true")
    run.add_argument("--allow-non-main", action="store_true")
    run.add_argument("--min-free-gb", type=positive_finite)

    show = sub.add_parser("status", allow_abbrev=False,
                          help="Show a workspace: inputs, day folders, the upload")
    show.add_argument("folder")
    show.add_argument("--json", action="store_true", help="print JSON")


def _error(exc: WorkspaceError, command: str, as_json: bool) -> int:
    if as_json:
        print(json.dumps({"error": {"code": exc.code, "message": str(exc)}},
                         ensure_ascii=False, indent=2))
    else:
        print(f"orbitwin {command}: ERROR [{exc.code}]: {exc}", file=sys.stderr)
    return EXIT_ERROR


def init_command(args: argparse.Namespace) -> int:
    try:
        workspace = init_workspace(args.folder, args.rules, args.team_number,
                                   args.gmat_console, args.startup_file, args.expect_commit)
    except WorkspaceError as exc:
        return _error(exc, "init", args.json)
    except OSError as exc:
        return _error(WorkspaceError("io", f"{type(exc).__name__}: {exc}"), "init", args.json)
    here = shlex.quote(str(workspace.root))
    steps = [f"put the organizer's .script, and any file it reads, into "
             f"{workspace.root / INPUT_DIR}",
             f"fill in {workspace.root / ANNOUNCE_FILE} as the values are announced",
             f"orbitwin run {here}"]
    if args.json:
        print(json.dumps({"workspace": str(workspace.root), "rules": workspace.rules,
                          "team_number": workspace.team_number, "next": steps},
                         ensure_ascii=False, indent=2))
    else:
        print(f"orbitwin init: wrote {workspace.root / WORKSPACE_FILE} and "
              f"{workspace.root / ANNOUNCE_FILE}", file=sys.stderr)
        for number, step in enumerate(steps, 1):
            print(f"  {number}. {step}", file=sys.stderr)
    return 0


def status_command(args: argparse.Namespace) -> int:
    try:
        report = status(read_workspace(args.folder))
    except WorkspaceError as exc:
        return _error(exc, "status", args.json)
    except OSError as exc:
        return _error(WorkspaceError("io", f"{type(exc).__name__}: {exc}"), "status",
                      args.json)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for line in status_lines(report):
            print(line)
    return 0


def _running(workspace: Workspace) -> bool:
    """Whether an `orbitwin run` holds the workspace now; creates nothing."""
    import fcntl

    try:
        handle = (workspace.root / ".run.lock").open("r")
    except OSError:
        return False
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return True
    return False


@contextlib.contextmanager
def _workspace_lock(workspace: Workspace) -> Iterator[None]:
    """One `orbitwin run` per workspace: a second would start a day folder of its own."""
    import fcntl
    import time

    with (workspace.root / ".run.lock").open("a") as handle:
        # A few tries: `orbitwin status` holds it shared for an instant to look.
        for attempt in range(20):
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if attempt == 19:
                    raise WorkspaceError("workspace-busy", f"another `orbitwin run` is "
                                         f"running on {workspace.root}") from None
                time.sleep(0.05)
        yield


def run_command(args: argparse.Namespace, run_day: Any) -> int:
    """Plan the day's argv from the workspace and hand it to ``run_day``."""
    with contextlib.ExitStack() as stack:
        try:
            workspace = read_workspace(args.folder)
            stack.enter_context(_workspace_lock(workspace))
            plan = plan_run(workspace, args)
        except WorkspaceError as exc:
            return _error(exc, "run", args.json)
        except OSError as exc:
            return _error(WorkspaceError("io", f"{type(exc).__name__}: {exc}"), "run",
                          args.json)
        print(f"orbitwin run: orbitwin day {shlex.join(plan.argv)}", file=sys.stderr)
        namespace = day_namespace(plan.argv)
        with contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext():
            code = int(run_day(namespace))
    if code != 0 and day_folders(workspace):
        print(f"orbitwin run: where the day printed `orbitwin day --resume <day folder> "
              f"<flags>`, `orbitwin run {shlex.quote(str(workspace.root))} <flags>` does the "
              f"same; the coefficients go into {ANNOUNCE_FILE}", file=sys.stderr)
    if args.json:
        try:
            summary = status(workspace)
        except OSError as exc:
            summary = {"error": {"code": "io", "message": f"{type(exc).__name__}: {exc}"}}
        print(json.dumps({"exit": code, **summary}, ensure_ascii=False, indent=2))
    return code

