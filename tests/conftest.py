from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, ScoreConfig, SearchConfig
from orbitwin.gmat_runner import is_test_double
from orbitwin.models import Candidate, Scenario, clock_stamps
from orbitwin.pinned import OVERRIDE

# For test_gmat_fixture.py and test_ruff_gate.py, which run pytest inside a test.
pytest_plugins = ["pytester"]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def workshop_script(project_root: Path) -> Path:
    """Our two-burn template, in the shape of the 2026 workshop template (not that file)."""
    return project_root / "examples" / "two_burn_rendezvous.script"


@pytest.fixture(scope="session")
def different_plane_script(project_root: Path) -> Path:
    return project_root / "examples" / "synthetic" / "DifferentPlane_Circular.script"


@pytest.fixture(scope="session")
def high_plane_script(project_root: Path) -> Path:
    return project_root / "examples" / "synthetic" / "HighPlaneChange_Circular.script"


@pytest.fixture(scope="session")
def no_maneuver_script(project_root: Path) -> Path:
    """The shape the competition template is expected to take.

    One long Propagate, no ImpulsiveBurn declared, so the maneuvers 初賽規則 §2
    allows have to be written into it rather than filled in.
    """
    return project_root / "examples" / "synthetic" / "NoManeuver_Circular.script"


@pytest.fixture(scope="session")
def unmaneuvered_burns_script(project_root: Path) -> Path:
    """The competition template, still declaring the burns it no longer uses."""
    return project_root / "examples" / "synthetic" / "UnmaneuveredBurns_Circular.script"


#: The competition organizer's material a copy may carry (NOTICE); the public export has none.
ORGANIZER_MATERIAL = Path(__file__).resolve().parents[1] / "ORGANIZER_MATERIAL.txt"


@dataclasses.dataclass(frozen=True)
class OrganizerMaterial:
    """ORGANIZER_MATERIAL.txt read: its patterns, header lines and named files (key: path)."""

    patterns: tuple[str, ...]
    headers: tuple[str, ...]
    files: dict[str, str]


def read_organizer_material(path: Path | None = None) -> OrganizerMaterial | None:
    """The list at path (ORGANIZER_MATERIAL by default), or None when there is none, as in
    the public export.

    Each line is a kind and its value apart by spaces or tabs; `file` takes a key and a
    path. Blank lines and text from a `#` at the start or after a blank are ignored.
    A line of another kind, or one without its value, fails the test reading it.
    """
    path = ORGANIZER_MATERIAL if path is None else path
    if not path.is_file():
        return None
    patterns: list[str] = []
    headers: list[str] = []
    files: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = re.sub(r"(?:^|\s)#.*", "", raw).strip()
        if not line:
            continue
        kind, value = [*line.split(None, 1), ""][:2]
        where = f"{path.name}:{number}"
        if kind == "pattern" and value:
            patterns.append(value)
        elif kind == "header" and value:
            headers.append(value)
        elif kind == "file" and len(value.split(None, 1)) == 2:
            key, relative = value.split(None, 1)
            if key in files:
                pytest.fail(f"{where} names the file {key} a second time")
            files[key] = relative
        else:
            pytest.fail(f"{where} is not a pattern, header or `file <key> <path>` line: {raw!r}")
    return OrganizerMaterial(tuple(patterns), tuple(headers), files)


def organizer_file(key: str) -> Path:
    """A file of the organizer's material by its key in ORGANIZER_MATERIAL.txt.

    Skips in a tree without the list, the public export; with the list, a key it does
    not name, or a path not in this tree, fails.
    """
    listed = ORGANIZER_MATERIAL
    material = read_organizer_material(listed)
    if material is None:
        pytest.skip(f"no {listed.name}: the organizer's {key} is not in this tree")
    if key not in material.files:
        pytest.fail(f"{listed.name} has no `file {key} <path>` line")
    path = listed.parent / material.files[key]
    if not path.exists():
        pytest.fail(f"{listed.name} names {material.files[key]} for {key}, "
                    "which is not in this tree")
    return path


def _fixed_prefix(pattern: str) -> str:
    """The folder a pattern's matches are all under ("" for the top), or the pattern itself
    when it has no wildcard."""
    wild = min((i for i in (pattern.find(c) for c in "*?[") if i >= 0), default=-1)
    return pattern if wild < 0 else pattern[:wild].rpartition("/")[0]


def organizer_matches(pattern: str, glob: str = "*") -> list[Path]:
    """The files under the list's folder that a pattern matches (fnmatch, * crossing /),
    walking only the folder its matches can be in, not the whole checkout."""
    import fnmatch

    root = ORGANIZER_MATERIAL.parent
    prefix = root / _fixed_prefix(pattern)
    candidates = [prefix] if prefix.is_file() else (
        prefix.rglob(glob) if prefix.is_dir() else [])
    return sorted(path for path in candidates
                  if path.is_file() and fnmatch.fnmatch(path.name, glob)
                  and fnmatch.fnmatchcase(path.relative_to(root).as_posix(), pattern))


def organizer_scripts() -> list[Path]:
    """Every GMAT script of the organizer's material in this tree; none in the public export."""
    material = read_organizer_material()
    if material is None:
        return []
    return sorted({path for pattern in material.patterns
                   for path in organizer_matches(pattern, "*.script")})


#: What R2026a's GUI said of edits to plot lines and plot commands (README.md beside it).
GUI_ORACLE = Path(__file__).resolve().parent / "fixtures" / "console_copy" / "gui_oracle_cases.json"


def gui_oracle_script(data: dict[str, object], case: dict[str, object]) -> str:
    """A GUI oracle case's script: its base with base_edits, then its edits, each replacing
    the first occurrence of its old text.

    A case without its own base, or base_edits, takes the top-level one. A base whose
    bytes are not the ones the GUI was run on (base_sha256, beside it) fails: the oracle
    would no longer say what the GUI does with this text, and has to be measured again.
    """
    import hashlib

    owner = case if "base" in case else data
    base = Path(__file__).resolve().parents[1] / str(owner["base"])
    measured = hashlib.sha256(base.read_bytes()).hexdigest()
    assert measured == owner["base_sha256"], (
        f"{owner['base']} is not the base {case['name']} was measured on; measure the GUI "
        "oracle again (tests/fixtures/console_copy/README.md) and record its sha256")
    text = base.read_bytes().decode("ascii")
    edits = case.get("base_edits", data["base_edits"])
    for old, new in [*edits, *case["edits"]]:  # type: ignore[misc]
        assert old in text, (case["name"], old)
        text = text.replace(old, new, 1)
    return text


#: Command-scope configuration (`git -c` and its children): listed by --local-env-vars, kept.
COMMAND_CONFIG = ("GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT")


def drop_repository_variables() -> None:
    """Unset what `git rev-parse --local-env-vars` lists but COMMAND_CONFIG, so a test's git
    cannot reach the caller's repository: git exports GIT_INDEX_FILE to a commit hook, and
    in a linked worktree GIT_DIR too, to commit hooks and to `git rebase -x`.

    The configuration git and ruff read stays: the user's files, GIT_CEILING_DIRECTORIES,
    and COMMAND_CONFIG with the GIT_CONFIG_KEY_* and GIT_CONFIG_VALUE_* it counts.
    """
    if shutil.which("git") is None:
        return  # no git for them to misdirect
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        names = subprocess.run(
            ["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True, check=True,
            env={**environment, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else exc
        pytest.exit(f"`git rev-parse --local-env-vars` failed, so the caller's GIT_DIR and "
                    f"GIT_INDEX_FILE stay set: {detail}")
    for name in names:
        if name not in COMMAND_CONFIG:
            os.environ.pop(name, None)


def pytest_configure(config: pytest.Config) -> None:
    drop_repository_variables()


def _scratch_git(home: Path) -> dict[str, str]:
    # GIT_CONFIG_GLOBAL needs git 2.32; HOME and XDG_CONFIG_HOME cover older ones.
    return {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(home), "XDG_CONFIG_HOME": str(home)}


def scratch_git_environment(home: Path) -> dict[str, str]:
    """The environment for git in a scratch repository: no GIT_* of the caller's, no global
    or system configuration (a global commit.gpgsign failed every scratch commit), home as
    HOME."""
    kept = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    return {**kept, **_scratch_git(home)}


def own_git(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """scratch_git_environment for the rest of a test."""
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    for key, value in _scratch_git(home).items():
        monkeypatch.setenv(key, value)


def inline_run(pytester: pytest.Pytester, *args: str | os.PathLike[str],
               **kwargs: object) -> pytest.HookRecorder:
    """pytester.inline_run with no cache and no faulthandler: in-process, the inner
    faulthandler re-arms on this test's captured stderr, closed after it."""
    return pytester.inline_run("-p", "no:cacheprovider", "-p", "no:faulthandler", *args,
                               **kwargs)


def _made_here() -> Callable[[str], bool]:
    """scripts/check_release_tree.py's made_here: what a release makes after unpacking."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "check_release_tree.py"
    spec = importlib.util.spec_from_file_location("orbitwin_check_release_tree", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.made_here


def _walked_names(root: Path, hidden: tuple[str, ...] = ()) -> list[str]:
    """Every file under root but hidden ones not named in hidden, those in hidden folders
    and what made_here names."""
    made_here = _made_here()
    found: list[str] = []
    for directory, folders, files in os.walk(root):
        base = Path(directory).relative_to(root).as_posix()
        prefix = "" if base == "." else f"{base}/"
        folders[:] = [name for name in folders
                      if not name.startswith(".") and not made_here(f"{prefix}{name}/x")]
        found += [prefix + name for name in files
                  if (not name.startswith(".") or name in hidden) and not made_here(prefix + name)]
    return found


def repo_files(root: Path, pattern: str, *, tracked_only: bool = False) -> list[Path]:
    """The files under root that git tracks, or would (untracked, not ignored) unless
    tracked_only, whose path from root matches pattern (fnmatch: * crosses /).

    Without git, as in an unpacked release, the files _walked_names lists.
    """
    import fnmatch

    untracked = [] if tracked_only else ["--others", "--exclude-standard"]
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", *untracked],
            capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        listed = b""
    names = [os.fsdecode(name) for name in listed.split(b"\0") if name] or _walked_names(root)
    return [root / name for name in sorted(names)
            if fnmatch.fnmatchcase(name, pattern) and (root / name).is_file()]


def ignored_files(root: Path, names: tuple[str, ...]) -> list[Path]:
    """The files under root named one of names that git ignores, outside ignored folders.

    Without git, the hidden ones of them that repo_files' walk leaves out.
    """
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--others", "--ignored",
             "--exclude-standard", "--directory"], capture_output=True, check=True,
        ).stdout
        found = [os.fsdecode(name) for name in listed.split(b"\0") if name]
    except (OSError, subprocess.CalledProcessError):
        found = [name for name in _walked_names(root, names) if "/." in f"/{name}"]
    return [root / name for name in sorted(found)
            if name.rsplit("/", 1)[-1] in names and (root / name).is_file()]


#: The test double and its startup file, whose relative paths GMAT resolves from the
#: console's folder: from scripts/, ROOT_PATH is the checkout.
FAKE_GMAT = Path(__file__).resolve().parents[1] / "scripts" / "fake_gmat_console.py"
FAKE_STARTUP = Path(__file__).resolve().parent / "fixtures" / "gmat" / "fake_startup.txt"


def startup_for_console_elsewhere(folder: Path) -> Path:
    """FAKE_STARTUP for a console copied out of scripts/: the same file, its gravity data
    named absolutely (a later entry overrides an earlier one)."""
    earth = FAKE_STARTUP.parent / "fake_gmat_data" / "gravity" / "earth"
    path = folder / "fake_startup.txt"
    path.write_text(FAKE_STARTUP.read_text(encoding="utf-8") + f"EARTH_POT_PATH = {earth}/\n",
                    encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def fake_gmat() -> Path:
    return FAKE_GMAT


@pytest.fixture(scope="session")
def fake_startup() -> Path:
    return FAKE_STARTUP


#: What a gmat-marked test fails with when it cannot run on GMAT.
GMAT_MISSING = "GMAT_CONSOLE and GMAT_STARTUP must name a real GMAT install"


def gmat_install() -> tuple[str, str] | str:
    """(GMAT_CONSOLE, GMAT_STARTUP), or why they do not name a GMAT install.

    The console is resolved and checked as run_gmat does, a path, never looked up on PATH.
    """
    console, startup = os.environ.get("GMAT_CONSOLE"), os.environ.get("GMAT_STARTUP")
    if not (console and startup):
        return "one or both are not set"
    try:
        executable = Path(console).resolve()
        if not executable.is_file():
            return f"GMAT_CONSOLE is not a file: {executable}"
        if not os.access(executable, os.X_OK):
            return f"GMAT_CONSOLE is not executable: {executable}"
        if is_test_double(executable):
            return f"GMAT_CONSOLE is orbitwin's test double: {executable}"
        if not Path(startup).is_file():
            return f"GMAT_STARTUP is not a file: {startup}"
    except (OSError, RuntimeError) as exc:
        return f"{console} or {startup} could not be checked: {type(exc).__name__}: {exc}"
    return console, startup


def require_gmat() -> tuple[str, str]:
    """gmat_install's pair, or the test fails saying why there is none."""
    found = gmat_install()
    if isinstance(found, str):
        pytest.fail(f"{GMAT_MISSING}: {found}", pytrace=False)
    return found


#: What the gmat fixture fails an unmarked test with: `-m "not gmat"` would select it.
GMAT_UNMARKED = "a test taking gmat must be marked gmat"


#: The pair pytest_runtest_setup checked, for the gmat fixture.
GMAT_PAIR = pytest.StashKey[tuple[str, str]]()


@pytest.fixture
def gmat(request: pytest.FixtureRequest) -> tuple[str, str]:
    """GMAT_CONSOLE and GMAT_STARTUP, checked by gmat_install, for a gmat-marked test."""
    if request.node.get_closest_marker("gmat") is None:
        pytest.fail(GMAT_UNMARKED, pytrace=False)
    return request.node.stash[GMAT_PAIR]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """A gmat-marked test fails without GMAT, whether or not it asks for the fixture."""
    if item.get_closest_marker("gmat") is not None:
        item.stash[GMAT_PAIR] = require_gmat()


# Shared because three unit test files were importing it from
# tests.integration.test_pipeline, which resolves only when the working
# directory is on sys.path -- true under `python -m pytest` from the repository
# root and false under the `pytest` console script or from any other directory.
# There is no __init__.py anywhere under tests/, so `tests` is a namespace
# package that exists by accident of invocation.
def quick_config(*, local_refine: bool = False) -> OrbitWinConfig:
    """A search of seconds. local_refine=True adds search_one_burn's local refinement, most of
    a solve's time, for a test that needs what it finds."""
    search = SearchConfig(
        one_burn_samples=0,
        one_burn_grid_burns=2,
        one_burn_grid_arrivals=2,
        max_revolutions=0,
        keep_candidates=20,
        render_top=3,
        event_step_s=20.0,
        event_refine_top=3,
        two_burn_enabled=False,
        two_burn_samples=0,
    )
    if not local_refine:
        search = dataclasses.replace(search, local_refine_top=0)
    return OrbitWinConfig(search=search)


def ranking_quick_config() -> OrbitWinConfig:
    """quick_config under the ranking rules, with a horizon and a score fake runs can use."""
    base = quick_config()
    policy = dataclasses.replace(
        base.policy, rules="ranking", horizon_s=30000.0, team_number=TEAM_NUMBER,
        dv_limit_km_s=0.25, dv_safe_limit_km_s=0.249,
        total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99,
    )
    # #175: ranking rules need the distance term's C_r and the delta-v in m/s.
    score = ScoreConfig(kt=0.0001, ct=20000.0, kv=0.001, cv=1500.0, delta_v_unit="m/s",
                        cr_km=100.0)
    return dataclasses.replace(base, policy=policy, score=score)


def capped_one_burn(
    scenario: Scenario, config: OrbitWinConfig, *, over_km_s: float | None = None
) -> Candidate:
    """search_one_burn's first plan, each burn over `over_km_s` (config's safe
    limit by default) cut to 0.8 of that limit; the fake flies the same rows whatever the burns."""
    from orbitwin.search import search_one_burn

    candidate = search_one_burn(scenario, quick_config())[0]
    cap = 0.8 * config.policy.dv_safe_limit_km_s
    over = config.policy.dv_safe_limit_km_s if over_km_s is None else over_km_s
    burns = tuple(
        dataclasses.replace(
            burn,
            dv_inertial_km_s=tuple(v * cap / burn.magnitude_km_s for v in burn.dv_inertial_km_s),
            dv_script_km_s=tuple(v * cap / burn.magnitude_km_s for v in burn.dv_script_km_s),
        )
        if burn.magnitude_km_s > over
        else burn
        for burn in candidate.burns
    )
    return dataclasses.replace(candidate, burns=burns)


def stamped(candidate: Candidate, script: str | Path, config: OrbitWinConfig) -> Candidate:
    """Stamp the script hash, horizon, clock and score block solve writes, as an archive has.

    certify warns on a candidate without them; test_certify_preflight tests that.
    Read through the fake console's probe, as certify reads it.
    """
    from orbitwin.score import score_stamp
    from orbitwin.state_probe import load_scenario

    scenario, _, _ = load_scenario(script, config, gmat_console=FAKE_GMAT,
                                   startup_file=FAKE_STARTUP)
    candidate.metadata["scenario_sha256"] = scenario.source_sha256
    candidate.metadata["horizon_s"] = scenario.max_time_s
    candidate.metadata.update(clock_stamps(scenario))
    candidate.metadata["searched_score"] = score_stamp(config.score)
    return candidate


@pytest.fixture
def synthetic_script() -> Path:
    """A second scenario, addressed from this file rather than from the cwd.

    Written as a fixture because a plain `Path("examples/...")` in a test is
    the third time in this change set that a relative path made a test pass
    only from the repository root -- and it fails silently in the direction
    that matters, by not running at all.
    """
    return (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "synthetic"
        / "DifferentPlane_Circular.script"
    )


@pytest.fixture
def pruning_polish(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    """Every ranking-search polish also returns its schedule pruned to nothing.

    A real hunt did this and wrote the zero burn twice. The workers are forked,
    so they inherit the patch. A forkserver or spawned worker would not, and
    the test would pass without it, so the workers log each call to a file and
    teardown requires one.
    """
    import multiprocessing

    import numpy as np

    from orbitwin import ranking_search as rs

    real = rs.refine_start
    calls = tmp_path_factory.mktemp("pruning_polish") / "calls"

    def refine(
        start: rs.RankingStart, setup: rs.SearchSetup, iterations: int | None = None
    ) -> list[rs.RankingSolution]:
        solutions = real(start, setup, iterations)
        if start.mode != "search":
            nothing = rs.settle(
                start.label, start.mode, [], np.zeros((0, 3)), setup, start.floor_margin_km
            )
            assert nothing is not None
            solutions.append(nothing)
            with calls.open("a", encoding="utf-8") as log:
                log.write(f"{os.getpid()}\n")
        return solutions

    monkeypatch.setattr(rs, "refine_start", refine)
    monkeypatch.setattr(rs, "_POOL_CONTEXT", multiprocessing.get_context("fork"))
    yield
    logged = calls.read_text(encoding="utf-8").split() if calls.is_file() else []
    assert logged, "no worker ran the patched refine_start"


_ALL_TYPES_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "examples" / "synthetic" / "DifferentPlane_Circular.script"
)

#: Every newly allowed type, as the R2026a console accepts it (measured, rc 0).
KEPT_BLOCK = """\
Create ChemicalTank T1;
Create ElectricTank ET1;
Create ChemicalThruster TH1;
TH1.Tank = {T1};
Create ElectricThruster ETH1;
ETH1.Tank = {ET1};
Create SolarPowerSystem SPS;
Create NuclearPowerSystem NPS;
Create Plate PL;
Create GroundStation GS;
GS.Location1 = 25;
Create Barycenter BC;
Create LibrationPoint LP;
Pursuer.Tanks = {T1, ET1};
Pursuer.Thrusters = {TH1, ETH1};
Pursuer.PowerSystem = SPS;
"""
DROPPED_BLOCK = """\
Create FiniteBurn FB;
FB.Thrusters = {TH1};
Create Formation F;
F.Add = {AlienSC};
Create GroundTrackPlot GT;
GT.Add = {Pursuer, Earth};
Create DynamicDataDisplay DDD;
DDD.AddParameters = {1, Pursuer.X};
Create DifferentialCorrector DC;
DC.ReportFile = 'dc.txt';
Create EphemerisFile EF;
EF.Spacecraft = AlienSC;
EF.Filename = 'alien.oem';
Create OpenFramesVector OFV;
Create OpenFramesSensorMask OFS;
"""
KEPT_NAMES = ("T1", "ET1", "TH1", "ETH1", "SPS", "NPS", "PL", "GS", "BC", "LP")
DROPPED_NAMES = ("FB", "F", "GT", "DDD", "DC", "EF", "OFV", "OFS")


def all_types_script(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "all_types.script"
    path.write_text(
        _ALL_TYPES_TEMPLATE.read_text(encoding="utf-8").replace(
            "BeginMissionSequence;", KEPT_BLOCK + DROPPED_BLOCK + extra + "BeginMissionSequence;", 1
        ),
        encoding="utf-8",
    )
    return path


LEGACY_BLOCK = """\
Create FuelTank FT;
Create Thruster TH;
TH.Tank = {FT};
Pursuer.Tanks = {FT};
Pursuer.Thrusters = {TH};
"""


def legacy_script(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "legacy.script"
    path.write_text(
        _ALL_TYPES_TEMPLATE.read_text(encoding="utf-8").replace(
            "BeginMissionSequence;", LEGACY_BLOCK + extra + "BeginMissionSequence;", 1
        ),
        encoding="utf-8",
    )
    return path


def without_rulebook_note(stderr: str) -> str:
    """stderr less the line the CLI prints when it runs under the historical rules."""
    return "".join(
        line for line in stderr.splitlines(keepends=True)
        if not line.startswith("orbitwin: policy.rules is 'preliminary', the historical")
    )


def bare_environment(**variables: str) -> dict[str, str]:
    """`variables` for a subprocess, and ORBITWIN_ALLOW_UNPINNED if this run has it.

    A suite run with the override on versions off the pins failed wherever a
    test ran orbitwin with only PYTHONPATH and PATH: that dropped it.
    """
    kept = {OVERRIDE: os.environ[OVERRIDE]} if OVERRIDE in os.environ else {}
    return {**kept, **variables}


#: The workshop template's plots, in the order it creates them.
WORKSHOP_PLOTS = ("DefaultOrbitView", "DefaultGroundTrackPlot", "DefaultOrbitView_View")
#: A line of each plot, Create or field, as the template writes them.
WORKSHOP_PLOT_LINE = re.compile(
    rf"^(?:Create (?:OpenFrames\w+|GroundTrack) |(?:{'|'.join(WORKSHOP_PLOTS)})\.)"
)


@pytest.fixture
def commented_workshop_script(workshop_script: Path, tmp_path: Path) -> Path:
    """The workshop template with its plot lines commented out, as a person would."""
    lines = workshop_script.read_text(encoding="ascii").splitlines(keepends=True)
    script = tmp_path / "commented_workshop.script"
    script.write_text(
        "".join(f"% {line}" if WORKSHOP_PLOT_LINE.match(line) else line for line in lines),
        encoding="ascii",
    )
    return script


@pytest.fixture
def plain_workshop_script(workshop_script: Path, tmp_path: Path) -> Path:
    """The workshop template with its plots deleted, every other line as it was."""
    lines = workshop_script.read_text(encoding="ascii").splitlines(keepends=True)
    script = tmp_path / "plain_workshop.script"
    script.write_text(
        "".join(line for line in lines if not WORKSHOP_PLOT_LINE.match(line)), encoding="ascii"
    )
    return script


def openframes_errors(script: Path) -> list[str]:
    """GmatConsole R2026a's lines for a script creating OpenFrames objects, as it words them,
    on GMAT's lines and with its comments skipped."""
    from orbitwin.gmat_parser import _code_part, script_lines

    lines = [_code_part(line.rstrip("\r\n")) for line in
             script_lines(script.read_text(encoding="ascii"))]
    pattern = re.compile(r"\s*Create (OpenFrames\w+) (\w+);")
    created = {match[2]: match[1] for match in map(pattern.match, lines) if match}
    said = []
    for number, line in enumerate(lines, 1):
        create = pattern.match(line)
        field = re.match(r"\s*(\w+)\.", line)
        toggle = re.match(r"\s*(Toggle|PenUp|PenDown|MarkPoint|ClearPlot)((?: \w+)+);", line)
        names = toggle[2].split() if toggle else []
        toggled = [name for name in names if name in created]
        if create:
            what = (f'Cannot create an object "{create[2]}". The "{create[1]}" is an unknown '
                    "object type or invalid object name or dimension in line:")
        elif field and field[1] in created:
            what = f'Cannot find LHS object named "{field[1]}" in line:'
        elif toggled:
            # R2026a names the first one it cannot find.
            assert toggle is not None
            what = (f'Nonexistent Subscriber "{toggled[0]}" referenced in the {toggle[1]} '
                    "in line:")
        else:
            continue
        said += [f"{len(said) // 2 + 1}: {script}: **** ERROR **** Interpreter Exception: {what}",
                 f'   " {number}: {line.strip()}"']
    return [*said, "ERROR!!!!!! ---- Console Application Execution Failed: Errors were found in "
            f'the script named "{script}"']


def run_gmat_return_code(error: str) -> int:
    """The exit status run_gmat records beside one of its own failure messages."""
    import signal

    from orbitwin.gmat_runner import (
        EXIT_STATUS_ERROR,
        FILE_SIZE_ERROR,
        TIMEOUT_ERROR,
        TIMEOUT_RETURN_CODE,
    )

    prefix = EXIT_STATUS_ERROR.format("")
    if error.startswith(prefix):
        return int(error[len(prefix):])
    if error.startswith(TIMEOUT_ERROR.split("{")[0]):
        return TIMEOUT_RETURN_CODE
    if error.startswith(FILE_SIZE_ERROR.split("{")[0]):
        return -signal.SIGXFSZ
    return 0



def refuse_runs_of(monkeypatch: pytest.MonkeyPatch, name: str, *, error: str | None = None) -> None:
    """GmatConsole as it answers a script with plots: the fake console runs them.

    A run is refused when its script or its run directory is ``name``: with
    R2026a's exit 1 and errors for each OpenFrames object, or with ``error``
    and the exit status run_gmat would have given it.
    """
    import orbitwin.verify as verify_module
    from orbitwin.gmat_runner import EXIT_STATUS_ERROR

    real = verify_module.run_gmat

    def run(console, script, workdir, reports, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = real(console, script, workdir, reports, *args, **kwargs)
        if name not in (Path(script).name, Path(workdir).name):
            return result
        if error is None:
            with open(result.stdout_path, "a", encoding="utf-8") as stdout:
                stdout.write("".join(f"{line}\n" for line in
                                     openframes_errors(Path(result.script_path))))
            failure = EXIT_STATUS_ERROR.format(1)
        else:
            failure = error
        return dataclasses.replace(
            result, success=False, reports={}, report_sha256={}, error=failure,
            return_code=run_gmat_return_code(failure),
        )

    monkeypatch.setattr(verify_module, "run_gmat", run)


#: The organizer's report name in the ranking templates. Under the ranking rules
#: certify refuses unless a report carries our number, so a ranking test needs one.
RANKING_REPORT = "ODC_2026_TeamXX.txt"
TEAM_NUMBER = 54


def with_ranking_report(text: str) -> str:
    """``text`` with its first ReportFile named as the ranking templates name it."""
    import re

    found = re.search(r"^Create ReportFile (\w+)", text, re.MULTILINE)
    assert found, "no ReportFile"
    renamed, count = re.subn(
        rf"^({found[1]}\.Filename\s*=\s*)'[^']*'", rf"\g<1>'{RANKING_REPORT}'", text,
        count=1, flags=re.MULTILINE,
    )
    assert count == 1, "no Filename line"
    return renamed


def ranking_copy(script: Path, directory: Path) -> Path:
    """A copy of ``script`` in ``directory`` with the ranking report name."""
    copy = directory / f"ranking_{script.name}"
    copy.write_text(with_ranking_report(script.read_text(encoding="utf-8")), encoding="utf-8")
    return copy


def kept_note(output: Path) -> str:
    """What certify adds to a refusal once it has moved `output` aside, naming the
    script and verification it kept."""
    kept = output.with_name(output.name + ".refused")
    named = ", ".join(str(kept / name) for name in (
        "staging/final.script", "diagnostic/verification.json") if (kept / name).is_file())
    return (
        f"This certification is kept in {kept}" + (f" ({named})" if named else "")
        + f", and {output} is free again; a certify stopped with Ctrl-C is not moved, so "
        "give the next one a fresh --output."
    )


#: (script, exact T_max s, boundary km) as examples/ranking/README.md announces them.
_ANNOUNCED = {
    "R1": ("R1_NearPlane_Reachable.script", 6711.667, 17000),
    "R2": ("R2_SteepPlane_NearMiss.script", 5791.381, 21000),
    "R3": ("R3_FastHighPerigee_TwoPropagators.script", 5092.815, 26000),
    "R4": ("R4_LowPerigee_Drag.script", 5219.966, 13500),
    "R5": ("R5_TiltedPlane_CloseMiss.script", 5791.747, 21000),
    "R6": ("R6_InsertedLate_Aligned.script", 8511.667, 17000),
    "R7": ("R7_NoAlignment_NearMiss.script", 7591.381, 21000),
    "R8": ("R8_PursuerFirst_Aligned.script", 8511.667, 17000),
}
#: Rounded to (s, T_max): the ones the boundary check refuses without the flag, as the
#: review of #190 measured them (output/review/pr190/logs/tmax_margins.tsv).
_REFUSED_ROUNDED = {("R1", 60), ("R2", 60), ("R3", 10), ("R3", 60), ("R5", 60), ("R6", 60),
                    ("R7", 60), ("R8", 60)}


def tmax_as_announced_cases(
    run: Callable[..., subprocess.CompletedProcess[str]], name: str, tmp_path: Path,
) -> None:
    """Exact passes without the flag; rounded to 10 s or the minute passes with it, and
    without it is refused exactly where SC_Alien is off the boundary."""
    script, exact, boundary = _ANNOUNCED[name]
    for step in (0, 10, 60):
        horizon = f"{exact:.3f}" if step == 0 else f"{round(exact / step) * step:.0f}"
        plain = tmp_path / f"{name}-{step}.yaml"
        done = run(script, horizon, boundary, plain)
        refused = (name, step) in _REFUSED_ROUNDED
        assert done.returncode == (2 if refused else 0), (step, done.stdout + done.stderr)
        assert plain.exists() is not refused
        if refused:
            assert "Read the announcement again; if this is the announced value, rerun with " \
                "--tmax-as-announced" in done.stderr
        if step == 0:
            continue
        flagged = tmp_path / f"{name}-{step}-flag.yaml"
        done = run(script, horizon, boundary, flagged, "--tmax-as-announced")
        assert done.returncode == 0, (step, done.stderr)
        from orbitwin.config import load_config

        assert load_config(flagged).policy.tmax_as_announced_s == float(horizon)
        assert f"as announced: T_max {float(horizon)!r} s is taken" in done.stdout
        assert (f"WARNING: T_max {float(horizon)!r} s taken as announced" in done.stdout) \
            is refused
