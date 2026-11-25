"""ranking and duel share the core and nothing else, in either direction."""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

#: Shared by both games. Imports only itself.
CORE = {
    "__init__", "burn_rows", "conic", "conic_batch", "errors", "event_detection", "frames",
    "gmat_runner", "installed", "lambert", "layout", "models", "orbit", "report_format",
    "report_parser", "rules", "rulepacks", "trajectory",
}
#: The ranking round, and the preliminary it grew from.
RANKING = {
    "announcement", "argtypes", "config", "console_copy", "constraints", "day", "design", "diagnostic",
    "ephemeris",
    "gmat_failures", "gmat_loop", "gmat_parser", "gmat_shoot", "io", "mission_clock",
    "multi_burn", "official_report", "openframes", "pareto", "pinned",
    "pipeline", "pursuer_model", "rank", "ranking_search", "score", "script_renderer", "search",
    "spad", "state_probe", "time_origin", "tools", "two_burn", "verify", "workspace",
}
#: The command line, which may dispatch to either game.
ENTRY = {"__main__", "cli"}
SOURCE = Path(__file__).resolve().parents[2] / "src" / "orbitwin"
DUEL = SOURCE / "duel"


def _module(path: Path) -> str:
    """`orbitwin.x` for orbitwin/x.py, `orbitwin.duel.x` for orbitwin/duel/x.py."""
    parts = list(path.relative_to(SOURCE.parent).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path: Path) -> set[str]:
    """The orbitwin modules `path` names, absolute, including `from . import x` and literals
    handed to importlib.import_module or __import__."""
    module = _module(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                assert node.level <= len(parts), f"{path.name}: relative import above orbitwin"
                base = ".".join(parts[: len(parts) - node.level + 1])
            else:
                base = ""
            stem = ".".join(filter(None, (base, node.module)))
            # `from orbitwin import x` and `from . import x` name modules, not attributes.
            found |= {f"{stem}.{alias.name}" for alias in node.names} | {stem}
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and (
                (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                or (isinstance(node.func, ast.Name) and node.func.id in {"import_module", "__import__"})
            )
        ):
            name = node.args[0].value
            if name.startswith("."):
                anchor = node.args[1] if len(node.args) > 1 else None
                assert isinstance(anchor, ast.Constant) and isinstance(anchor.value, str), (
                    f"{path.name}: relative import_module without a literal package"
                )
                name = importlib.util.resolve_name(name, anchor.value)
            found.add(name)
    return {name for name in found if name == "orbitwin" or name.startswith("orbitwin.")}


def _side(name: str) -> str:
    """Which side the orbitwin module `name` is on."""
    top = name.split(".")[1] if "." in name else "__init__"
    if top == "duel":
        return "duel"
    for side, members in (("core", CORE), ("ranking", RANKING), ("entry", ENTRY)):
        if top in members:
            return side
    return "attribute"  # `from orbitwin import Burn`: a name in orbitwin/__init__.py


def _files() -> list[Path]:
    return sorted(SOURCE.rglob("*.py"))


def test_every_module_is_on_one_side() -> None:
    top = {path.stem for path in SOURCE.glob("*.py")} | {
        path.parent.name for path in SOURCE.glob("*/__init__.py") if path.parent != DUEL}
    assert not CORE & RANKING and not (CORE | RANKING) & ENTRY
    assert top == CORE | RANKING | ENTRY, (
        f"unclassified: {sorted(top - CORE - RANKING - ENTRY)}, "
        f"gone: {sorted((CORE | RANKING | ENTRY) - top)}"
    )


@pytest.mark.parametrize(
    "source, imported",
    [
        ("from .. import pipeline", "ranking"),
        ("from ..ranking_search import fly", "ranking"),
        ("from orbitwin import ranking_search", "ranking"),
        ("import orbitwin.verify", "ranking"),
        ("importlib.import_module('orbitwin.verify')", "ranking"),
        ("importlib.import_module('.verify', 'orbitwin')", "ranking"),
        ("from .. import orbit", "core"),
        ("from .engine import play", "duel"),
    ],
)
def test_the_reader_sees_each_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, imported: str
) -> None:
    root = tmp_path / "orbitwin"
    (root / "duel").mkdir(parents=True)
    path = root / "duel" / "probe.py"
    path.write_text(source + "\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "SOURCE", root)
    assert imported in {_side(name) for name in _imports(path)}


def test_core_imports_only_core() -> None:
    for path in _files():
        if path.parent == SOURCE and path.stem in CORE:
            sides = {_side(name) for name in _imports(path)} - {"core", "attribute"}
            assert not sides, f"{path.name} imports {sides}"


def test_no_ranking_module_imports_the_duel() -> None:
    for path in _files():
        if (path.parent == SOURCE and path.stem in RANKING) or path.parent == SOURCE / "tools":
            assert "duel" not in {_side(name) for name in _imports(path)}, f"{path.name}"


def test_the_duel_imports_no_ranking_module() -> None:
    if not DUEL.is_dir():
        pytest.skip("orbitwin.duel does not exist yet")
    for path in sorted(DUEL.rglob("*.py")):
        sides = {_side(name) for name in _imports(path)} & {"ranking", "entry"}
        assert not sides, f"{path.relative_to(SOURCE)} imports {sides}"


def _loaded_after_importing(modules: list[str], forbidden: set[str]) -> str:
    code = (
        "import sys\n"
        f"for name in {modules!r}:\n"
        "    __import__(name)\n"
        "loaded = {m.split('.')[1] for m in sys.modules if m.startswith('orbitwin.')}\n"
        f"print(sorted(loaded.intersection({sorted(forbidden)!r})))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(SOURCE.parent)},
    )
    return done.stdout.strip()


def test_importing_the_core_loads_no_ranking_module() -> None:
    """Catches what the reader cannot: an import inside a function body run at import time,
    or a chain through a module the lists above misfile."""
    core = [f"orbitwin.{name}" for name in sorted(CORE - {"__init__"})]
    assert _loaded_after_importing(core, RANKING | ENTRY | {"duel"}) == "[]"


def test_importing_the_duel_loads_no_ranking_module() -> None:
    if not DUEL.is_dir():
        pytest.skip("orbitwin.duel does not exist yet")
    duel = [_module(path) for path in sorted(DUEL.rglob("*.py"))]
    assert _loaded_after_importing(duel, RANKING | ENTRY) == "[]"
