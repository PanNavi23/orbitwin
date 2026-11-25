"""The `gmat` fixture's check (tests/conftest.py): what each wrong setting fails with."""

from __future__ import annotations

import os
from pathlib import Path

import conftest
import pytest
from conftest import GMAT_MISSING, GMAT_UNMARKED, gmat_install, inline_run

from orbitwin.gmat_runner import TEST_DOUBLE_MARKER


@pytest.fixture
def unset(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv("GMAT_CONSOLE", raising=False)
    monkeypatch.delenv("GMAT_STARTUP", raising=False)
    return monkeypatch


def _check(monkeypatch: pytest.MonkeyPatch, console: Path | str, startup: Path) -> object:
    monkeypatch.setenv("GMAT_CONSOLE", str(console))
    monkeypatch.setenv("GMAT_STARTUP", str(startup))
    return gmat_install()


def test_the_check_says_what_is_wrong(tmp_path: Path, unset: pytest.MonkeyPatch) -> None:
    assert gmat_install() == "one or both are not set"
    console, startup = tmp_path / "bin" / "GmatConsole", tmp_path / "startup.txt"
    console.parent.mkdir()
    console.write_text("", encoding="ascii")
    # Executable, so a lookup on PATH would find it.
    console.chmod(0o755)
    startup.write_text("", encoding="ascii")

    assert _check(unset, console, startup) == (str(console), str(startup))
    assert _check(unset, console.parent, startup) == f"GMAT_CONSOLE is not a file: {console.parent}"
    assert _check(unset, console, tmp_path / "none.txt") == \
        f"GMAT_STARTUP is not a file: {tmp_path / 'none.txt'}"
    double = tmp_path / "fake_gmat_console.py"
    double.write_bytes(b"#!/bin/sh" + TEST_DOUBLE_MARKER)
    double.chmod(0o755)
    assert _check(unset, double, startup) == f"GMAT_CONSOLE is orbitwin's test double: {double}"
    console.chmod(0o644)
    assert _check(unset, console, startup) == f"GMAT_CONSOLE is not executable: {console}"
    console.chmod(0o755)
    # A bare name is a path from the working directory, as run_gmat takes it.
    unset.setenv("PATH", f"{console.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    unset.chdir(tmp_path)
    assert _check(unset, "GmatConsole", startup) == \
        f"GMAT_CONSOLE is not a file: {tmp_path / 'GmatConsole'}"


def test_a_symlink_loop_is_a_message_not_a_traceback(
    tmp_path: Path, unset: pytest.MonkeyPatch
) -> None:
    # resolve() raises RuntimeError on a loop before Python 3.13 and returns after it.
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    startup = tmp_path / "startup.txt"
    startup.write_text("", encoding="ascii")
    said = _check(unset, loop, startup)
    assert isinstance(said, str) and (
        said.startswith("GMAT_CONSOLE is not a file: ")
        or said.startswith(f"{loop} or {startup} could not be checked: RuntimeError")), said


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches any folder")
def test_a_console_that_cannot_be_checked_is_a_message_not_a_traceback(
    tmp_path: Path, unset: pytest.MonkeyPatch
) -> None:
    # is_file() raises PermissionError here before Python 3.14 and returns False after it.
    locked = tmp_path / "locked"
    locked.mkdir()
    console = locked / "GmatConsole"
    console.write_text("#!/bin/sh\n", encoding="ascii")
    console.chmod(0o755)
    startup = tmp_path / "startup.txt"
    startup.write_text("", encoding="ascii")
    locked.chmod(0)
    try:
        said = _check(unset, console, startup)
    finally:
        locked.chmod(0o755)
    assert said in (
        f"{console} or {startup} could not be checked: PermissionError: "
        f"[Errno 13] Permission denied: '{console}'",
        f"GMAT_CONSOLE is not a file: {console}"), said


GATED = """\
import pytest

@pytest.mark.gmat
def test_plain(): pass

@pytest.mark.gmat
@pytest.mark.skip(reason="skipped")
def test_skipped(): pass

@pytest.mark.gmat
@pytest.mark.xfail(reason="xfailed")
def test_xfailed(): assert False

@pytest.mark.gmat
class TestMarked:
    def test_in_class(self): pass

def test_unmarked(): pass
"""
MODULE_MARKED = """\
import pytest

pytestmark = pytest.mark.gmat

def test_in_marked_module(): pass
"""


def _run(pytester: pytest.Pytester, *files: str, args: tuple[str, ...] = ()) -> pytest.HookRecorder:
    pytester.makeini("[pytest]\nmarkers =\n    gmat: GMAT\n")
    for number, source in enumerate(files):
        pytester.makepyfile(**{f"test_{number}": source})
    return inline_run(pytester, *args, plugins=[conftest])


def _setup(run: pytest.HookRecorder) -> dict[str, pytest.TestReport]:
    """Each test's setup report, by test name."""
    return {report.nodeid.rsplit("::", 1)[1]: report
            for report in run.getreports("pytest_runtest_logreport") if report.when == "setup"}


def test_without_gmat_every_gmat_test_errors_whatever_its_marks(
    pytester: pytest.Pytester, unset: pytest.MonkeyPatch
) -> None:
    """pytest_runtest_setup fails a gmat test that takes no fixture, and runs before the
    skip and xfail marks are read."""
    setup = _setup(_run(pytester, GATED, MODULE_MARKED))

    for name in ("test_plain", "test_skipped", "test_xfailed", "test_in_class",
                 "test_in_marked_module"):
        assert setup[name].failed, (name, setup[name].outcome)
        assert f"{GMAT_MISSING}: one or both are not set" in str(setup[name].longrepr), name
    assert setup["test_unmarked"].passed


def test_with_gmat_a_gmat_test_runs_and_gets_console_then_startup(
    pytester: pytest.Pytester, tmp_path: Path, unset: pytest.MonkeyPatch
) -> None:
    console, startup = tmp_path / "GmatConsole", tmp_path / "startup.txt"
    console.write_text("#!/bin/sh\n", encoding="ascii")
    console.chmod(0o755)
    startup.write_text("", encoding="ascii")
    uses = f"""\
import pytest

@pytest.mark.gmat
def test_uses(gmat): assert gmat == ({str(console)!r}, {str(startup)!r})
"""
    unset.setenv("GMAT_CONSOLE", str(console))
    unset.setenv("GMAT_STARTUP", str(startup))

    _run(pytester, uses, MODULE_MARKED).assertoutcome(passed=2)


def test_the_fixture_takes_the_pair_the_setup_checked(
    pytester: pytest.Pytester, tmp_path: Path, unset: pytest.MonkeyPatch
) -> None:
    """One check per test, so the fixture's pair is the one the setup passed."""
    console, startup = tmp_path / "GmatConsole", tmp_path / "startup.txt"
    console.write_text("#!/bin/sh\n", encoding="ascii")
    console.chmod(0o755)
    startup.write_text("", encoding="ascii")
    unset.setenv("GMAT_CONSOLE", str(console))
    unset.setenv("GMAT_STARTUP", str(startup))
    checks: list[object] = []

    def counted() -> object:
        checks.append(gmat_install())
        return checks[-1]

    unset.setattr(conftest, "gmat_install", counted)
    uses = "import pytest\n\n@pytest.mark.gmat\ndef test_uses(gmat): pass\n"

    _run(pytester, uses).assertoutcome(passed=1)
    assert checks == [(str(console), str(startup))]


@pytest.mark.parametrize("only", ["GMAT_CONSOLE", "GMAT_STARTUP"])
def test_one_variable_alone_fails_a_gmat_test(
    pytester: pytest.Pytester, tmp_path: Path, unset: pytest.MonkeyPatch, only: str
) -> None:
    startup = tmp_path / "startup.txt"
    startup.write_text("", encoding="ascii")
    unset.setenv(only, str(startup))

    setup = _setup(_run(pytester, MODULE_MARKED))

    assert f"{GMAT_MISSING}: one or both are not set" in str(
        setup["test_in_marked_module"].longrepr)


TAKERS = """\
import pytest

def test_takes_the_fixture(gmat): pass

def test_asks_for_the_fixture(request): request.getfixturevalue("gmat")

@pytest.mark.parametrize("gmat", ["x"], indirect=True)
def test_given_the_fixture_indirectly(gmat): pass

@pytest.mark.parametrize("gmat", ["a message"])
def test_given_gmat_as_a_parameter(gmat): pass

def test_takes_nothing(): pass
"""
OWN_FIXTURE = """\
import pytest

@pytest.fixture
def gmat(): return {"section": 1}

def test_takes_its_own(gmat): assert gmat == {"section": 1}
"""


def test_an_unmarked_test_taking_the_fixture_fails(
    pytester: pytest.Pytester, unset: pytest.MonkeyPatch
) -> None:
    """Unmarked, `make test` ran it on this workstation's GMAT and `make test-gmat` never did.
    A module's own fixture named gmat is not this one."""
    passed, _, failed = _run(pytester, TAKERS, OWN_FIXTURE).listoutcomes()

    assert sorted(report.nodeid.rsplit("::", 1)[1] for report in failed) == [
        "test_asks_for_the_fixture", "test_given_the_fixture_indirectly[x]",
        "test_takes_the_fixture"]
    assert all(GMAT_UNMARKED in str(report.longrepr) for report in failed)
    assert sorted(report.nodeid.rsplit("::", 1)[1] for report in passed) == [
        "test_given_gmat_as_a_parameter[a message]", "test_takes_its_own", "test_takes_nothing"]
