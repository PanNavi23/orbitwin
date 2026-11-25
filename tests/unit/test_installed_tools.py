"""An installed orbitwin carries the commands `orbitwin day` and certify run, and the profile."""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "orbitwin"
TOOLS = ("check_report_rows", "ranking_day_config", "audit_burn_table")


def _package_data() -> list[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return list(data["tool"]["setuptools"]["package-data"]["orbitwin"])


def _shipped(path: Path) -> bool:
    """Whether setuptools puts ``path`` in the wheel: a module, or package data."""
    relative = path.relative_to(PACKAGE).as_posix()
    return path.suffix == ".py" or relative == "py.typed" or any(
        fnmatch.fnmatch(relative, pattern) for pattern in _package_data())


def test_every_file_the_package_reads_is_shipped() -> None:
    # Notes for people (Markdown) are not read by the package and stay out.
    files = [path for path in PACKAGE.rglob("*") if path.is_file()
             and "__pycache__" not in path.parts and path.suffix != ".md"]
    left_out = [path.relative_to(PACKAGE).as_posix() for path in files if not _shipped(path)]
    assert left_out == []


def test_the_packaged_profile_is_the_checkouts() -> None:
    assert (PACKAGE / "profiles" / "ranking.yaml").read_bytes() == (
        ROOT / "config" / "ranking.yaml").read_bytes()


@pytest.fixture
def installed(tmp_path: Path) -> Path:
    """What a wheel installs, alone in a site directory outside the checkout."""
    site = tmp_path / "site"
    for path in PACKAGE.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and _shipped(path):
            target = site / "orbitwin" / path.relative_to(PACKAGE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return site


def _run(site: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "BASH_ENV", "GMAT_CONSOLE", "GMAT_STARTUP")}
    env["PYTHONPATH"] = str(site)
    return subprocess.run([sys.executable, "-s", *argv], cwd=site.parent, env=env,
                          capture_output=True, text=True, check=False, timeout=300)


@pytest.mark.parametrize("tool", TOOLS)
def test_each_runbook_tool_runs_from_an_installed_package(installed: Path, tool: str) -> None:
    done = _run(installed, "-m", f"orbitwin.tools.{tool}", "--help")
    assert done.returncode == 0, done.stderr
    assert "usage:" in done.stdout


def test_an_installed_package_finds_its_profile_and_search_script(installed: Path) -> None:
    done = _run(installed, "-c", "from orbitwin.config import ranking_profile, load_config\n"
                "from orbitwin.day import TOOLS\n"
                "print(ranking_profile())\n"
                "print((TOOLS / 'competition_day.sh').is_file())\n"
                "print(load_config(ranking_profile()).policy.rules)")
    assert done.returncode == 0, done.stderr
    profile, script, rules = done.stdout.splitlines()
    assert Path(profile) == installed / "orbitwin" / "profiles" / "ranking.yaml"
    assert script == "True"
    assert rules == "ranking"


def test_the_checkout_scripts_run_the_packaged_code() -> None:
    """scripts/<tool>.py is the package's file, executed: a test that patches it, patches
    what runs."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("crr_probe", ROOT / "scripts" /
                                                  "check_report_rows.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._penalized.__code__.co_filename == str(
        PACKAGE / "tools" / "check_report_rows.py")
    assert module._penalized.__globals__ is module.__dict__


def test_files_beside_an_installed_package_are_not_a_checkout(installed: Path) -> None:
    """A stray scripts/ or config/ two levels up is not taken for the repository's."""
    (installed.parent / "scripts").mkdir()
    (installed.parent / "scripts" / "check_report_rows.py").write_text("raise SystemExit(9)\n",
                                                                      encoding="utf-8")
    (installed.parent / "config").mkdir()
    (installed.parent / "config" / "ranking.yaml").write_text("policy:\n  rules: preliminary\n",
                                                             encoding="utf-8")
    done = _run(installed, "-c", "from orbitwin.layout import CHECKOUT\n"
                "from orbitwin.config import ranking_profile\n"
                "from orbitwin.cli import _checker\n"
                "from orbitwin.day import DEFAULT_WORK_ROOT\n"
                "print(CHECKOUT); print(ranking_profile()); print(_checker()[1:]);"
                " print(DEFAULT_WORK_ROOT)")
    assert done.returncode == 0, done.stderr
    checkout, profile, checker, work = done.stdout.splitlines()
    assert checkout == "None"
    assert Path(profile) == installed / "orbitwin" / "profiles" / "ranking.yaml"
    assert checker == "['-m', 'orbitwin.tools.check_report_rows']"
    # Relative: resolved under the directory the day is started in, when it starts.
    assert work == "run/day"


def test_the_checkout_is_found_from_its_own_package() -> None:
    from orbitwin.layout import CHECKOUT

    assert CHECKOUT == ROOT


def test_the_search_script_refuses_a_python_that_imports_another_orbitwin(
    installed: Path,
) -> None:
    script = installed / "orbitwin" / "tools" / "competition_day.sh"
    env = {key: value for key, value in os.environ.items()
           if key not in ("BASH_ENV", "GMAT_CONSOLE", "GMAT_STARTUP")}
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    # The checkout's package, not the installed one beside the script.
    env["PYTHONPATH"] = str(ROOT / "src")
    done = subprocess.run(["bash", str(script), "a.script", "out", "c.yaml"], env=env,
                          cwd=installed.parent, capture_output=True, text=True, check=False,
                          timeout=120)
    assert done.returncode == 2
    assert "not " + str(installed / "orbitwin") in done.stderr
