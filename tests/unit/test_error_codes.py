"""Every refusal names a code docs/ERROR_CODES.md lists."""

from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import orbitwin
from orbitwin.cli import main
from orbitwin.errors import OrbitWinError, error_code

ROOT = Path(__file__).resolve().parents[2]
DOC = (ROOT / "docs" / "ERROR_CODES.md").read_text(encoding="utf-8")
LISTED = set(re.findall(r"^\| `([a-z-]+)` \|", DOC, re.M))


def _subclasses(cls: type[OrbitWinError]) -> set[type[OrbitWinError]]:
    found = set()
    for sub in cls.__subclasses__():
        found |= {sub, *_subclasses(sub)}
    return found


def test_every_error_class_has_a_listed_code() -> None:
    for module in pkgutil.walk_packages(orbitwin.__path__, "orbitwin."):
        importlib.import_module(module.name)
    from orbitwin.day import DayPaused, DayStop
    from orbitwin.workspace import WorkspaceError

    # The day's own stops and the workspace's errors are worded and coded elsewhere.
    elsewhere = {DayStop, DayPaused, WorkspaceError}
    unlisted = {f"{cls.__qualname__}: {cls.code}" for cls in _subclasses(OrbitWinError)
                if cls not in elsewhere and cls.code not in LISTED}
    assert unlisted == set()
    assert {"environment", "io", "value", "error"} <= LISTED


@pytest.mark.parametrize(("exc", "code"), [
    (OSError(28, "No space left on device"), "io"),
    (ValueError("x"), "value"),
    (KeyError("x"), "error"),
])
def test_the_code_of_what_is_not_ours(exc: BaseException, code: str) -> None:
    assert error_code(exc) == code


def test_a_refusal_names_its_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("policy: [\n", encoding="utf-8")
    assert main(["inspect", str(ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"),
                 "--config", str(config)]) == 2
    line = next(line for line in capsys.readouterr().err.splitlines()
                if line.startswith("orbitwin: ERROR"))
    code = re.match(r"orbitwin: ERROR \[([a-z-]+)\]: ", line)
    assert code is not None and code.group(1) in LISTED, line


def test_rules_names_its_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["rules", "no-such/rules"]) == 2
    assert capsys.readouterr().err.startswith("orbitwin: ERROR [rules]: ")
