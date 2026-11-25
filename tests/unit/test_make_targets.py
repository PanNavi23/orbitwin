"""The Makefile, scripts/release_validate.sh and pyproject.toml agree with each other."""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest


def _recipe(project_root: Path, target: str) -> str:
    """The recipe lines of one Makefile target, continuation lines joined."""
    text = (project_root / "Makefile").read_text(encoding="utf-8").replace("\\\n", " ")
    found = re.search(rf"^{re.escape(target)}:[^\n]*\n((?:\t[^\n]*\n)+)", text, re.M)
    assert found, target
    return found.group(1)


def _selection(command: str) -> str:
    found = re.search(r"""-m (["'])([^"']+)\1""", command)
    assert found, command
    return found.group(2)


def _pyproject(project_root: Path) -> dict:
    return tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))


def _clean_environment() -> dict[str, str]:
    """os.environ without what an outer make, pytest or coverage run would hand down.

    `make test PYTEST_ARGS=...` passes its variables to every nested make in MAKEFLAGS.
    BASH_ENV is a file bash sources first, which may set PATH again.
    """
    return {
        key: value for key, value in os.environ.items()
        if not key.startswith(("COV_", "COVERAGE_", "PYTEST_", "MAKE", "MFLAGS"))
        and key not in ("BASH_ENV", "ENV")
    }


def _fake_python(tmp_path: Path, marker: Path) -> Path:
    """An interpreter without site-packages, which leaves a mark when it runs."""
    fake = tmp_path / "bin" / "fakepy"
    fake.parent.mkdir(parents=True)
    fake.write_text(f'#!/bin/sh\n: > "{marker}"\nexec "{sys.executable}" -I -S "$@"\n',
                    encoding="utf-8")
    fake.chmod(0o755)
    return fake


def test_coverage_measures_the_selection_the_release_gate_does(project_root: Path) -> None:
    """`make coverage` took `not gmat`, slow tests included; release_validate.sh did not."""
    validate = (project_root / "scripts" / "release_validate.sh").read_text(encoding="utf-8")
    gate = next(line for line in validate.splitlines() if "--cov" in line)

    assert _selection(_recipe(project_root, "coverage")) == _selection(gate)
    assert _selection(gate) == _selection(_recipe(project_root, "test"))


@pytest.mark.parametrize("target", ["test-isolated", "coverage"])
def test_diagnostic_targets_leave_the_evidence_alone(project_root: Path, target: str) -> None:
    """They wrote into make validate's evidence."""
    recipe = _recipe(project_root, target)
    makefile = (project_root / "Makefile").read_text(encoding="utf-8")
    report_dir = re.search(r"^REPORT_DIR \?= (\S+)$", makefile, re.M)

    assert "validation/" not in recipe
    assert "$(REPORT_DIR)/" in recipe
    assert report_dir and report_dir.group(1).split("/")[0] == "output"
    assert "/output/" in (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_coverage_follows_worker_processes_and_subprocesses(project_root: Path) -> None:
    """Without them ProcessPoolExecutor workers and CLI subprocesses count as missed."""
    run = _pyproject(project_root)["tool"]["coverage"]["run"]

    assert run["source"] == ["orbitwin"]
    assert "multiprocessing" in run["concurrency"] and "thread" in run["concurrency"]
    assert run["parallel"] is True
    assert "subprocess" in run["patch"]
    for command in (_recipe(project_root, "coverage"),
                    (project_root / "scripts" / "release_validate.sh").read_text("utf-8")):
        assert "--cov=" not in command, "a --cov=PATH overrides the configured source"


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_validate_passes_its_interpreter_on(project_root: Path) -> None:
    printed = subprocess.run(
        ["make", "-n", "-C", str(project_root), "validate", "PYTHON=/opt/py/bin/python"],
        capture_output=True, text=True, check=True, env=_clean_environment(),
    ).stdout

    assert 'PYTHON="/opt/py/bin/python" PYTEST_ARGS="" bash scripts/release_validate.sh' in printed


def test_validate_refuses_an_interpreter_before_touching_the_evidence(
    project_root: Path, tmp_path: Path
) -> None:
    """It deleted 67 tracked files and emptied environment.json, then died on `import scipy`."""
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(project_root / "scripts" / "release_validate.sh", root / "scripts")
    for name in ("requirements.txt", "requirements-dev.txt"):
        shutil.copy2(project_root / name, root / name)
    evidence = root / "output" / "validation" / "fake_baseline" / "baseline.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}\n", encoding="utf-8")
    environment = root / "output" / "validation" / "environment.json"
    environment.write_text('{"python": "3.13"}\n', encoding="utf-8")
    marker = tmp_path / "ran"
    _fake_python(tmp_path, marker)

    completed = subprocess.run(
        ["bash", str(root / "scripts" / "release_validate.sh")],
        cwd=tmp_path, capture_output=True, text=True, check=False,
        env={**_clean_environment(), "PYTHON": "bin/fakepy"},
    )

    assert completed.returncode != 0
    assert marker.exists(), "the script ran some other interpreter"
    assert "cannot import numpy" in completed.stderr
    assert "Nothing was changed" in completed.stderr
    assert evidence.read_text(encoding="utf-8") == "{}\n"
    assert environment.read_text(encoding="utf-8") == '{"python": "3.13"}\n'
    assert not (root / "output" / "validation" / "project_dependency_check.json").exists()


#: A file no run writes (a note, make fuzz's output): a restore or a cleanup must keep it.
KEPT_BY_HAND = "README.md"


def _validate_checkout(project_root: Path, root: Path, *, copy_src: bool = False) -> Path:
    """release_validate.sh with what its checks read, and one evidence file. Returns it."""
    (root / "scripts").mkdir(parents=True)
    (root / "tests" / "unit").mkdir(parents=True)
    shutil.copy2(project_root / "scripts" / "release_validate.sh", root / "scripts")
    shutil.copy2(project_root / "tests" / "unit" / "test_make_targets.py", root / "tests" / "unit")
    for name in ("requirements.txt", "requirements-dev.txt", "pyproject.toml"):
        shutil.copy2(project_root / name, root / name)
    if copy_src:
        shutil.copytree(project_root / "src" / "orbitwin", root / "src" / "orbitwin",
                        ignore=shutil.ignore_patterns("__pycache__"))
    else:
        (root / "src").symlink_to(project_root / "src")
    evidence = root / "output" / "validation" / "fake_baseline" / "baseline.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}\n", encoding="utf-8")
    (root / "output" / "validation" / "environment.json").write_text("{}\n", encoding="utf-8")
    (root / "output" / "validation" / KEPT_BY_HAND).write_text("written by hand\n", encoding="utf-8")
    return evidence


def _validate(root: Path, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(root / "scripts" / "release_validate.sh")],
        capture_output=True, text=True, check=False,
        env={**_clean_environment(), "PYTHON": sys.executable, **environment},
    )


def _tree_bytes(root: Path) -> dict[str, tuple[bytes, int, int]]:
    """Contents, mode and mtime: a restore must put back the tree, not just the bytes."""
    return {
        path.relative_to(root).as_posix(): (
            path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns
        )
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _age(root: Path) -> None:
    """A fixed past mtime on everything, so a copy that does not keep it shows."""
    for path in root.rglob("*"):
        os.utime(path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))


@pytest.mark.parametrize(
    "arguments",
    ["-k make", '-k "a or b"', "-m slow", "tests/unit", "-x", "--deselect tests/unit",
     "-p no:cov", "-n", "-n -k", "-n 2 --lf"],
)
def test_validate_takes_only_xdist_options(
    project_root: Path, tmp_path: Path, arguments: str
) -> None:
    """`-k make` shrank the coverage evidence to a sliver and still printed PASS."""
    evidence = _validate_checkout(project_root, tmp_path / "checkout")

    completed = _validate(tmp_path / "checkout", PYTEST_ARGS=arguments)

    assert completed.returncode == 2, completed.stderr
    assert ("only -n, --numprocesses, --maxprocesses, --dist, --maxschedchunk and --durations"
            in completed.stderr)
    assert evidence.read_text(encoding="utf-8") == "{}\n"


def _ci_fast_suite_arguments(project_root: Path) -> str:
    """PYTEST_ARGS of ci.yml's Fast suite step."""
    import yaml

    workflow = yaml.safe_load((project_root / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
    steps = [step for job in workflow["jobs"].values() for step in job.get("steps", [])
             if step.get("name") == "Fast suite"]
    assert len(steps) == 1
    found = re.search(r'PYTEST_ARGS="([^"]*)"', steps[0]["run"])
    assert found, steps[0]["run"]
    return found.group(1)


@pytest.mark.parametrize(
    "arguments",
    ["-n 2 --dist worksteal", "-n16", "--numprocesses=auto", "--durations=25", "--durations 0",
     "-n auto --dist worksteal --durations=25", "--maxschedchunk=1", "--dist load --maxschedchunk 1",
     pytest.param(None, id="ci.yml")],
)
def test_validate_accepts_xdist_options(
    project_root: Path, tmp_path: Path, arguments: str | None
) -> None:
    """They pass the screen and reach the interpreter check, which this one fails."""
    if arguments is None:
        arguments = _ci_fast_suite_arguments(project_root)
    _validate_checkout(project_root, tmp_path / "checkout")
    marker = tmp_path / "ran"

    completed = _validate(tmp_path / "checkout", PYTEST_ARGS=arguments,
                          PYTHON=str(_fake_python(tmp_path, marker)))

    assert marker.exists()
    assert "cannot import numpy" in completed.stderr


def test_validate_refuses_options_pytest_refuses_before_touching_the_evidence(
    project_root: Path, tmp_path: Path
) -> None:
    """`-n` without xdist would only fail at the coverage step, 15 steps in."""
    evidence = _validate_checkout(project_root, tmp_path / "checkout")

    completed = _validate(tmp_path / "checkout", PYTEST_ARGS="--dist no-such-mode")

    assert completed.returncode != 0
    # With xdist an invalid choice, without it an unrecognized argument.
    assert "--dist" in completed.stderr
    assert evidence.read_text(encoding="utf-8") == "{}\n"
    assert (tmp_path / "checkout" / "output" / "validation" / "environment.json").read_text() == "{}\n"


def test_validate_names_a_python_directory_that_is_not_there(
    project_root: Path, tmp_path: Path
) -> None:
    """The failed cd was masked, and PYTHON became /python."""
    evidence = _validate_checkout(project_root, tmp_path / "checkout")

    completed = _validate(tmp_path / "checkout", PYTHON=str(tmp_path / "gone" / "bin" / "python"))

    assert completed.returncode == 2
    assert f"no directory {tmp_path / 'gone' / 'bin'}" in completed.stderr
    assert evidence.read_text(encoding="utf-8") == "{}\n"


def test_a_failed_validate_puts_the_evidence_back(project_root: Path, tmp_path: Path) -> None:
    """It stopped part-way with old and new evidence mixed and nothing to say so.

    Here the slow step finds no slow tests, exit 5, after the fake_* evidence
    is deleted and environment.json rewritten.
    """
    root = tmp_path / "checkout"
    _validate_checkout(project_root, root, copy_src=True)
    _age(root / "output" / "validation")
    before = _tree_bytes(root / "output" / "validation")
    # A copy cut short by a kill, made before anything changed.
    (root / "output" / "validate" / "previous.partial" / "half").mkdir(parents=True)

    completed = _validate(root)

    assert completed.returncode == 5, completed.stderr[-2000:]
    assert "release validation: FAILED (exit 5). output/validation/ is restored" in completed.stderr
    assert _tree_bytes(root / "output" / "validation") == before
    failed = list((root / "output" / "validate").glob("failed-*"))
    assert len(failed) == 1
    assert not (failed[0] / "fake_baseline").exists()
    assert (failed[0] / "environment.json").read_text(encoding="utf-8") != "{}\n"
    assert not (root / "output" / "validate" / "previous").exists()
    assert not (root / "output" / "validate" / "previous.partial").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through read-only modes")
def test_validate_refuses_read_only_evidence(project_root: Path, tmp_path: Path) -> None:
    """rm -rf deleted 95 files under a read-only validation/ before it failed."""
    root = tmp_path / "checkout"
    evidence = _validate_checkout(project_root, root)
    before = _tree_bytes(root / "output" / "validation")
    (root / "output" / "validation" / "fake_baseline").chmod(0o555)
    try:
        completed = _validate(root)
    finally:
        (root / "output" / "validation" / "fake_baseline").chmod(0o755)

    assert completed.returncode == 2
    assert "output/validation/fake_baseline is not writable; nothing was changed" in completed.stderr
    assert _tree_bytes(root / "output" / "validation") == before
    assert evidence.exists()
    assert not (root / "output" / "validate" / "previous").exists()


def test_validate_stops_on_a_copy_left_by_a_killed_run(project_root: Path, tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    evidence = _validate_checkout(project_root, root)
    (root / "output" / "validate" / "previous").mkdir(parents=True)

    completed = _validate(root)

    assert completed.returncode == 2
    assert "left from an interrupted run" in completed.stderr
    assert evidence.read_text(encoding="utf-8") == "{}\n"


def test_a_refused_run_makes_no_evidence_folder(project_root: Path, tmp_path: Path) -> None:
    """Made before the refusal, an empty one turned `mv previous output/validation` into a nest."""
    root = tmp_path / "checkout"
    _validate_checkout(project_root, root)
    shutil.rmtree(root / "output" / "validation")
    (root / "output" / "validate" / "previous").mkdir(parents=True)

    completed = _validate(root)

    assert completed.returncode == 2
    assert "left from an interrupted run" in completed.stderr
    assert not (root / "output" / "validation").exists()


def test_validate_refuses_linked_evidence(project_root: Path, tmp_path: Path) -> None:
    """cp -a copied the link, and the cleanup deleted through it."""
    root = tmp_path / "checkout"
    _validate_checkout(project_root, root)
    elsewhere = tmp_path / "elsewhere"
    (root / "output" / "validation").rename(elsewhere)
    (root / "output" / "validation").symlink_to(elsewhere)
    before = _tree_bytes(elsewhere)

    completed = _validate(root)

    assert completed.returncode == 2
    assert "output/validation is a symbolic link" in completed.stderr
    assert _tree_bytes(elsewhere) == before


def test_the_fake_campaign_certifies_what_it_solved(project_root: Path, tmp_path: Path) -> None:
    """solve ran without the console, so certify saw a 4 T_A about 42 us apart and refused it."""
    output = tmp_path / "evidence"

    completed = subprocess.run(
        ["bash", str(project_root / "scripts" / "fake_cli_evidence.sh"), str(output)],
        capture_output=True, text=True, check=False,
        env={**_clean_environment(), "PYTHON": sys.executable},
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    certification = json.loads(
        (output / "fake_certification" / "submission" / "certification.json").read_text("utf-8")
    )
    assert [run["success"] for run in certification["exact_runs"]] == [True, True, True]


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_make_clean_is_the_package_clean(project_root: Path, tmp_path: Path) -> None:
    """It deleted dist/ with the release, walked .venv, worktrees and runs, and left egg-info.

    Run with an interpreter that does not exist: before, clean needed only rm and find.
    """
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(project_root / "Makefile", root)
    shutil.copy2(project_root / "scripts" / "clean.sh", root / "scripts")
    kept = [
        "pyproject.toml",
        "dist/tasa_orbitwin-0.3.0.zip", "dist/tasa_orbitwin-0.3.0.zip.sha256", "build/lib/x",
        "output/R1/summary.json", "output/R1/__pycache__/m.pyc",
        ".venv/lib/pkg/__pycache__/m.pyc", ".worktrees/w/.git",
        ".worktrees/w/src/__pycache__/m.pyc",
        ".git/objects/q.pyc",
    ]
    removed = [
        "src/tasa_orbitwin.egg-info/PKG-INFO", "src/orbitwin/__pycache__/m.pyc",
        ".pytest_cache/v", ".mypy_cache/v", ".ruff_cache/v", "htmlcov/index.html",
        ".coverage", ".coverage.host.1.x",
    ]
    for relative in (*kept, *removed):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("x", encoding="utf-8")

    subprocess.run(["make", "-C", str(root), "clean", "PYTHON=/nonexistent/python"],
                   capture_output=True, text=True, check=True, env=_clean_environment())

    assert [path for path in kept if not (root / path).exists()] == []
    assert [path for path in removed if (root / path).exists()] == []
    assert not (root / "src" / "tasa_orbitwin.egg-info").exists()


def test_the_dev_extra_is_gone(project_root: Path) -> None:
    """`.[dev]` lacked types-PyYAML and pinned nothing; requirements-dev.txt is the install."""
    assert "optional-dependencies" not in _pyproject(project_root)["project"]


def test_mypy_checks_against_the_oldest_supported_python(project_root: Path) -> None:
    """At 3.12 a call 3.11 lacks, itertools.batched, passed the type check."""
    pyproject = _pyproject(project_root)
    floor = re.fullmatch(r">=(\d+\.\d+)", pyproject["project"]["requires-python"])

    assert floor
    assert pyproject["tool"]["mypy"]["python_version"] == floor.group(1)
    assert pyproject["tool"]["ruff"]["target-version"] == "py" + floor.group(1).replace(".", "")


def _limits_not_above(timeout_s: float, limits_s: dict[str, float]) -> dict[str, float]:
    return {where: limit for where, limit in limits_s.items() if not 0 < timeout_s < limit}


def test_a_hung_test_dumps_before_ci_cancels_the_job(project_root: Path) -> None:
    """faulthandler_timeout is below ci.yml's reviewed timeouts, which test_ci_runner
    pins with the file, and below run_isolated_test_files' per-file kill."""
    from test_ci_runner import CHECKS_MIN, GITHUB_RUNNERS_MIN, OWN_RUNNERS_MIN

    timeout_s = _pyproject(project_root)["tool"]["pytest"]["ini_options"]["faulthandler_timeout"]
    spec = importlib.util.spec_from_file_location(
        "isolated_runner", project_root / "scripts" / "run_isolated_test_files.py")
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    assert _limits_not_above(timeout_s, {
        "test job, own runners": 60 * OWN_RUNNERS_MIN,
        "test job, GitHub's runners": 60 * GITHUB_RUNNERS_MIN,
        "checks job": 60 * CHECKS_MIN,
        "run_isolated_test_files.py": runner.TIMEOUT_PER_FILE_S}) == {}


@pytest.mark.parametrize(("timeout_s", "late"), [
    pytest.param(600, {}, id="below-all"),
    pytest.param(900, {"b": 900}, id="equal"),
    pytest.param(1000, {"b": 900}, id="above-one"),
    pytest.param(0, {"a": 1800, "b": 900}, id="zero"),
])
def test_a_limit_at_or_below_the_dump_is_reported(timeout_s: float, late: dict) -> None:
    assert _limits_not_above(timeout_s, {"a": 1800, "b": 900}) == late


def test_the_bundled_backend_reads_the_license_expression(project_root: Path) -> None:
    """An SPDX string needs setuptools 77; the offline bootstrap builds with the bundled one."""
    pyproject = _pyproject(project_root)
    floor = next(
        requirement for requirement in pyproject["build-system"]["requires"]
        if requirement.startswith("setuptools")
    )
    constraints = (project_root / "vendor" / "wheels" / "constraints.txt").read_text("utf-8")
    bundled = re.search(r"^setuptools==(\d+)\.", constraints, re.M)

    assert pyproject["project"]["license"] == "Apache-2.0"
    assert floor == "setuptools>=77"
    assert bundled and int(bundled.group(1)) >= 77


def test_the_metadata_claims_hold(project_root: Path) -> None:
    """Every license file named ships, and Typing :: Typed has its PEP 561 marker."""
    project = _pyproject(project_root)["project"]

    assert project["name"] == "orbitwin"
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    assert all((project_root / name).is_file() for name in project["license-files"])
    license_sha256 = hashlib.sha256((project_root / "LICENSE").read_bytes()).hexdigest()
    assert license_sha256 == "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    assert "Typing :: Typed" in project["classifiers"]
    assert (project_root / "src" / "orbitwin" / "py.typed").is_file()
    assert not [c for c in project["classifiers"] if c.startswith("License ::")]


def _organizer_material(project_root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """ORGANIZER_MATERIAL.txt's patterns and header lines; none in a copy without the file.

    A list with no pattern or no header would make the test below pass on nothing.
    Imported here: release_validate.sh's tests copy this file without tests/conftest.py."""
    from conftest import read_organizer_material

    material = read_organizer_material(project_root / "ORGANIZER_MATERIAL.txt")
    if material is None:
        return (), ()
    if not material.patterns or not material.headers:
        pytest.fail("ORGANIZER_MATERIAL.txt lists no pattern or no header line")
    return material.patterns, material.headers


def _shipped_files(project_root: Path) -> list[Path]:
    """git's tracked files in a checkout with commits; otherwise the manifest's rule, less the
    run/ and input/ trees a used release grows."""
    spec = importlib.util.spec_from_file_location(
        "orbitwin_make_manifest_notice", project_root / "scripts" / "make_manifest.py"
    )
    assert spec and spec.loader
    make_manifest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_manifest)
    if (project_root / ".git").exists() and shutil.which("git"):
        tracked = list(make_manifest.tracked_files(project_root))
        if tracked:
            return tracked
    return sorted(path for path in project_root.rglob("*")
                  if path.relative_to(project_root).parts[0] not in {"run", "input"}
                  and make_manifest.included(path, project_root))


def test_notice_names_exactly_the_organizer_scripts(project_root: Path) -> None:
    """Every shipped script with an organizer template's header matches a listed pattern.

    The list (ORGANIZER_MATERIAL.txt, which NOTICE names) claims no script of ours: a
    pattern matching a script without that header fails. A copy without the list, as
    the public export is, ships no organizer script at all.
    """
    patterns, headers = _organizer_material(project_root)
    assert "ORGANIZER_MATERIAL.txt" in (project_root / "NOTICE").read_text(encoding="utf-8")
    organizer, ours = [], []
    for path in _shipped_files(project_root):
        if path.suffix != ".script":
            continue
        head = path.read_text(encoding="utf-8", errors="replace")[:200]
        relative = path.relative_to(project_root).as_posix()
        (organizer if any(h in head for h in headers) else ours).append(relative)

    def covered(name: str) -> bool:
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)

    if headers:
        # The top-level preliminary and the organizer's own two.
        assert len(organizer) >= 3
    assert [name for name in organizer if not covered(name)] == []
    assert [name for name in ours if covered(name)] == []
    assert all(any(fnmatch.fnmatchcase(name, p) for name in organizer)
               or ("*" not in p and (project_root / p).is_file()) for p in patterns)


def test_every_python_the_classifiers_name_is_tested(project_root: Path) -> None:
    classifiers = _pyproject(project_root)["project"]["classifiers"]
    named = sorted(c.rsplit(" ", 1)[1] for c in classifiers
                   if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c))
    workflow = (project_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    tested = re.search(r"python-version: \[([^]]*)\]", workflow)

    assert tested
    assert named == sorted(v.strip().strip('"') for v in tested.group(1).split(","))


#: Everything clean() decides about: removed, spared by name or as another checkout, or not matched.
_CLEAN_TREE = (
    "pyproject.toml", ".pytest_cache/v", ".ruff_cache/v", "htmlcov/index.html",
    ".coverage", ".coverage.host.1.x", ".coverage.dir/x", ".coveragerc",
    "src/tasa_orbitwin.egg-info/PKG-INFO", "src/notes.egg-info.txt", "src/.hidden.egg-info/x",
    "src/orbitwin/__pycache__/m.cpython-313.pyc", "src/orbitwin/old.pyc", "scripts/x.pyo",
    "scripts/keep.py", "tests/a/b/__pycache__/c.pyc", "src/orbitwin/build/__pycache__/m.pyc",
    "tests/output/x.pyc", "docs/dist/y.pyc", ".venv/lib/__pycache__/m.pyc",
    ".worktrees/w/.git", ".worktrees/w/src/__pycache__/m.pyc", "other/m.pyc",
    "output/R1/__pycache__/m.pyc",
    "dist/tasa_orbitwin-0.3.0.zip", "build/lib/z.pyc", ".git/objects/q.pyc",
    "deep/.git.d/q.pyc", "deep/.pytest_cache/v", "deep/.coverage", "clone/.git/HEAD", "clone/m.pyc", "weird.pyc/inner.txt", "-rf/__pycache__/m.pyc",
    "--x.pyc", "sp ace/__pycache__/m.pyc", "new\nline/__pycache__/m.pyc",
)
#: Symbolic links in the tree, to targets outside it. Each is removed or kept, never followed.
_CLEAN_LINKS = {
    ".mypy_cache": "cache", "src/linked.egg-info": "egg", "tests/c/__pycache__": "pycache",
    "scripts/link.pyc": "file.pyc", "scripts/dirlink.pyc": "dir", ".coverage.link": "data",
    ".coverage.dangling": "missing", "tests/linked": "tree", "docs/dangling.pyc": "missing",
    "other/.git": "missing",
}
_OUTSIDE = ("cache/v", "egg/PKG-INFO", "pycache/m.pyc", "file.pyc", "dir/x", "data",
            "tree/__pycache__/m.pyc", "tree/n.pyc")


def _remaining(root: Path) -> set[str]:
    """Every entry under root; a link is listed, not followed."""
    found = set()
    for directory, subdirectories, files in os.walk(root):
        for name in (*subdirectories, *files):
            found.add((Path(directory) / name).relative_to(root).as_posix())
    return found


def _clean_tree(root: Path, outside: Path) -> None:
    for relative in (*_CLEAN_TREE, *(f"../{outside.name}/{name}" for name in _OUTSIDE)):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("x", encoding="utf-8")
    for link, target in _CLEAN_LINKS.items():
        (root / link).parent.mkdir(parents=True, exist_ok=True)
        (root / link).symlink_to(outside / target)


def test_the_shell_clean_is_build_release_clean(project_root: Path, tmp_path: Path) -> None:
    """Two implementations of one contract: make clean's and make package's.

    They differed on links, and on a directory named .coverage.<x>: the shell
    stopped there and Python raised IsADirectoryError, which broke make package.
    """
    spec = importlib.util.spec_from_file_location(
        "orbitwin_build_release_clean", project_root / "scripts" / "build_release.py"
    )
    assert spec and spec.loader
    build_release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_release)
    trees = []
    for name in ("python", "shell"):
        root = tmp_path / name
        _clean_tree(root, tmp_path / f"{name}-outside")
        trees.append(root)
    (trees[1] / "scripts" / "clean.sh").write_bytes(
        (project_root / "scripts" / "clean.sh").read_bytes()
    )

    build_release.clean(trees[0])
    subprocess.run(["sh", str(trees[1] / "scripts" / "clean.sh")], check=True, cwd="/",
                   capture_output=True, env={"PATH": "/usr/bin:/bin"})

    left = _remaining(trees[0])
    assert _remaining(trees[1]) - {"scripts/clean.sh"} == left
    for name in ("python", "shell"):
        assert _remaining(tmp_path / f"{name}-outside") == _remaining(
            tmp_path / "python-outside"
        ) == {"cache", "egg", "pycache", "dir", "tree", "tree/__pycache__", *_OUTSIDE}
    assert {"dist/tasa_orbitwin-0.3.0.zip", "tests/linked", "weird.pyc", "src/.hidden.egg-info",
            ".coveragerc", "--x.pyc"} - left == {"--x.pyc"}
    # Another checkout, a .git file, folder or dangling link in it, keeps its bytecode;
    # caches below the root are not caches clean removes.
    assert {".worktrees/w/src/__pycache__/m.pyc", "other/m.pyc", "clone/m.pyc",
            ".git/objects/q.pyc", "deep/.pytest_cache/v", "deep/.coverage"} <= left
    assert "deep/.git.d/q.pyc" not in left
    assert not left & {"src/orbitwin/old.pyc", ".coverage.dir", ".mypy_cache",
                       "src/linked.egg-info", "tests/c/__pycache__", "scripts/link.pyc",
                       "scripts/dirlink.pyc", ".coverage.link", ".coverage.dangling",
                       "docs/dangling.pyc", "new\nline/__pycache__"}


def _checkout(root: Path, project_root: Path) -> None:
    """Enough of a checkout for clean.sh to accept it, with bytecode to remove."""
    for relative in ("pyproject.toml", "src/orbitwin/__pycache__/m.pyc"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("x", encoding="utf-8")
    (root / "scripts").mkdir()
    shutil.copy2(project_root / "scripts" / "clean.sh", root / "scripts")


def test_clean_through_a_symlink_cleans_the_checkout_it_belongs_to(
    project_root: Path, tmp_path: Path
) -> None:
    """Run as ~/bin/x -> scripts/clean.sh, it cleaned ~."""
    root = tmp_path / "repo"
    _checkout(root, project_root)
    home = tmp_path / "home"
    for relative in ("proj/pkg/__pycache__/m.pyc", "proj/lib.pyc", ".pytest_cache/v"):
        (home / relative).parent.mkdir(parents=True, exist_ok=True)
        (home / relative).write_text("x", encoding="utf-8")
    (home / "bin").mkdir()
    (home / "bin" / "orbitwin-clean").symlink_to(root / "scripts" / "clean.sh")
    before = _remaining(home)

    subprocess.run(["sh", str(home / "bin" / "orbitwin-clean")], cwd=home, check=True,
                   capture_output=True, env={"PATH": "/usr/bin:/bin"})

    assert _remaining(home) == before
    assert not (root / "src" / "orbitwin" / "__pycache__").exists()


@pytest.mark.parametrize("how", ["stdin", "sourced"])
def test_clean_without_its_own_path_removes_nothing(
    project_root: Path, tmp_path: Path, how: str
) -> None:
    """`sh < scripts/clean.sh` cleaned the parent directory, sibling projects included."""
    parent = tmp_path / "parent"
    root = parent / "repo"
    _checkout(root, project_root)
    for relative in ("sibling/pkg/__pycache__/s.pyc", ".mypy_cache/v", ".coverage"):
        (parent / relative).parent.mkdir(parents=True, exist_ok=True)
        (parent / relative).write_text("x", encoding="utf-8")
    before = _remaining(parent)
    script = root / "scripts" / "clean.sh"

    with script.open("rb") as stdin:
        completed = subprocess.run(
            ["sh"] if how == "stdin" else ["sh", "-c", f". '{script}'"],
            cwd=root, stdin=stdin if how == "stdin" else subprocess.DEVNULL,
            capture_output=True, text=True, check=False, env={"PATH": "/usr/bin:/bin"},
        )

    assert completed.returncode == 2
    assert "is not an OrbitWin checkout; nothing removed" in completed.stderr
    assert _remaining(parent) == before


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_make_coverage_writes_only_under_the_report_dir(
    project_root: Path, tmp_path: Path
) -> None:
    """It left .coverage and .pytest_cache/ in the checkout beside output/make/."""
    pytest.importorskip("pytest_cov")
    root = tmp_path / "checkout"
    (root / "src" / "orbitwin").mkdir(parents=True)
    (root / "tests").mkdir()
    for name in ("Makefile", "pyproject.toml"):
        shutil.copy2(project_root / name, root / name)
    (root / "src" / "orbitwin" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "tests" / "test_one.py").write_text(
        "import orbitwin\n\n\ndef test_one() -> None:\n    assert orbitwin.VALUE == 1\n",
        encoding="utf-8",
    )
    before = _remaining(root)

    completed = subprocess.run(
        ["make", "-C", str(root), "coverage", f"PYTHON={sys.executable}",
         "PYTEST_ARGS=tests/test_one.py"],
        capture_output=True, text=True, check=False, env=_clean_environment(),
    )

    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    added = {path.split("/")[0] for path in _remaining(root) - before}
    assert added - {"src", "tests"} == {"output"}, added
    assert (root / "output" / "make" / "coverage.xml").is_file()
    assert not [path for path in _remaining(root) - before
                if path.startswith(("src/", "tests/")) and "__pycache__" not in path]


def _commands(script: str) -> list[str]:
    """Shell lines with their continuations joined."""
    return script.replace("\\\n", " ").splitlines()


def test_pytest_args_reach_the_coverage_run_and_nothing_else(project_root: Path) -> None:
    """`-n 16` on the slow or isolated steps would change evidence the release relies on."""
    validate = (project_root / "scripts" / "release_validate.sh").read_text(encoding="utf-8")
    pytests = [line for line in _commands(validate) if "-m pytest" in line]

    assert len(pytests) >= 3
    for line in pytests:
        expected = "--cov" in line or "--collect-only" in line
        assert ("pytest_args" in line) == expected, line
    # The whole list, expanded into any command, not only `python -m pytest`.
    expanded = [line for line in _commands(validate)
                if re.search(r'"\$\{pytest_args\[@\]\}"', line)]
    assert len(expanded) == 2
    assert all("--cov" in line or "--collect-only" in line for line in expanded), expanded


@pytest.mark.parametrize(
    "path", [".coverage", ".coverage.thc1006.12345.XabcdeYx", "output/make/coverage.xml"]
)
def test_coverage_data_is_ignored(project_root: Path, path: str) -> None:
    """Parallel mode leaves .coverage.<host>.<pid>.<random> when a run is cut short."""
    if shutil.which("git") is None or not (project_root / ".git").exists():
        pytest.skip("not a git checkout")
    ignored = subprocess.run(
        ["git", "-C", str(project_root), "check-ignore", "-q", "--no-index", path],
        check=False, env=_clean_environment(),
    )

    assert ignored.returncode == 0, f"{path} is not gitignored"


def _python_without_pip(tmp_path: Path) -> Path:
    """An interpreter that cannot import pip; anything else goes to this one."""
    fake = tmp_path / "bin" / "python"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(
        '#!/bin/sh\n[ "$1" = "-c" ] && [ "$2" = "import pip" ] && exit 1\n'
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake


def _dependency_check(
    project_root: Path, tmp_path: Path, python: Path, path: str
) -> tuple[str, str]:
    out = tmp_path / "evidence"
    bash = shutil.which("bash")
    assert bash
    subprocess.run(
        [bash, str(project_root / "scripts" / "dependency_check.sh"), str(out)],
        check=True, capture_output=True,
        env={**_clean_environment(), "PYTHON": str(python), "PATH": path},
    )
    return ((out / "pip_check.exit_code").read_text(encoding="utf-8").strip(),
            (out / "pip_check.log").read_text(encoding="utf-8"))


def _tools(tmp_path: Path, *names: str) -> str:
    """A PATH holding only these programs."""
    directory = tmp_path / "tools"
    directory.mkdir(exist_ok=True)
    for name in names:
        found = shutil.which(name)
        assert found, name
        (directory / name).symlink_to(found)
    return str(directory)


def test_a_venv_without_pip_is_checked_with_uv(project_root: Path, tmp_path: Path) -> None:
    """`python -m pip check` in a uv venv recorded exit 1, 'No module named pip', and PASS."""
    if shutil.which("uv") is None:
        pytest.skip("needs uv")
    python = _python_without_pip(tmp_path)

    code, log = _dependency_check(
        project_root, tmp_path, python, _tools(tmp_path, "uv", "mkdir")
    )

    assert log.startswith(f"$ uv pip check --python {python}\n"), log
    assert "No module named pip" not in log
    assert code == "0", log


def test_neither_pip_nor_uv_is_recorded_as_skipped(project_root: Path, tmp_path: Path) -> None:
    python = _python_without_pip(tmp_path)

    code, log = _dependency_check(project_root, tmp_path, python, _tools(tmp_path, "mkdir"))

    assert code == "SKIPPED"
    assert log.startswith("SKIPPED: ")


def test_pip_is_used_where_the_interpreter_has_it(project_root: Path, tmp_path: Path) -> None:
    fake = tmp_path / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.write_text(
        '#!/bin/sh\n[ "$1" = "-c" ] && exit 0\necho "demo 1.0 requires missing"\nexit 1\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)

    code, log = _dependency_check(project_root, tmp_path, fake, _tools(tmp_path, "mkdir"))

    assert log == f"$ {fake} -m pip check\ndemo 1.0 requires missing\n"
    assert code == "1"


def _verdict_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "orbitwin_release_verdict", project_root / "scripts" / "release_verdict.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _certification(runs: tuple[bool, ...] = (True,) * 3) -> str:
    return json.dumps({"exact_runs": [{"success": run} for run in runs]})


def _results(root: Path, *, fuzz: int = 0, matrix: int = 0, runs: tuple[bool, ...] = (True,) * 3,
             code: str = "0", log: str = "$ uv pip check\nAll installed packages are compatible\n",
             omit: str = "") -> Path:
    """What release_verdict.py reads, as a run that passed would leave it."""
    (root / "fake_certification" / "submission").mkdir(parents=True)
    files = {
        "ood_fuzz_5000.json": json.dumps({"failure_count": fuzz, "failures": ["x"] * fuzz}),
        "scenario_ood_matrix_20.json": json.dumps(
            {"pipeline_exception_count": matrix, "exceptions": ["x"] * matrix}
        ),
        "fake_certification/submission/certification.json": _certification(runs),
        "pip_check.exit_code": code + "\n",
        "pip_check.log": log,
    }
    for name, text in files.items():
        if name != omit:
            (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("results", "message"),
    [({"code": "1", "log": "$ python -m pip check\nNo module named pip\n"},
      "dependency check exited 1"),
     ({"fuzz": 2}, "OOD fuzz: 2 failures"),
     ({"matrix": 1}, "scenario matrix: 1 exceptions"),
     ({"runs": (True, False, True)}, "exact runs [True, False, True]"),
     ({"runs": (True, True)}, "exact runs [True, True]"),
     ({"omit": "pip_check.exit_code"}, "no result from this run"),
     ({"omit": "ood_fuzz_5000.json"}, "no result from this run")],
)
def test_the_verdict_fails_on_any_failed_result(
    project_root: Path, tmp_path: Path, results: dict[str, Any], message: str
) -> None:
    """The dependency check ran under `set +e` and nothing read it; PASS was printed anyway."""
    verdict = _verdict_module(project_root)
    root = _results(tmp_path / "validation", **results)

    with pytest.raises(SystemExit, match=re.escape(message)):
        verdict.verdict(root)


@pytest.mark.parametrize(("code", "noted"), [("0", False), ("SKIPPED", True)])
def test_the_verdict_passes_a_clean_run(
    project_root: Path, tmp_path: Path, code: str, noted: bool
) -> None:
    root = _results(tmp_path / "validation", code=code,
                    log="SKIPPED: python has no pip and uv is not on PATH\n")

    assert bool(_verdict_module(project_root).verdict(root)) == noted


def test_the_verdict_holds_under_python_optimize(project_root: Path, tmp_path: Path) -> None:
    """With `assert` as the gate, PYTHONOPTIMIZE=1 stripped it and PASS was printed."""
    root = _results(tmp_path / "validation", code="1", fuzz=3)

    completed = subprocess.run(
        [sys.executable, "-O", str(project_root / "scripts" / "release_verdict.py"), str(root)],
        capture_output=True, text=True, check=False,
        env={**_clean_environment(), "PYTHONOPTIMIZE": "1"},
    )

    assert completed.returncode != 0
    assert "PASS" not in completed.stdout
    assert "OOD fuzz: 3 failures" in completed.stderr
    assert "dependency check exited 1" in completed.stderr
    source = (project_root / "scripts" / "release_verdict.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*assert\b", source, re.M)


@pytest.mark.parametrize("variable", ["PYTEST_ADDOPTS", "PYTEST_PLUGINS"])
def test_validate_refuses_pytest_settings_from_the_environment(
    project_root: Path, tmp_path: Path, variable: str
) -> None:
    """`PYTEST_ADDOPTS=-k make` shrank the evidence past the PYTEST_ARGS screen."""
    evidence = _validate_checkout(project_root, tmp_path / "checkout")

    completed = _validate(tmp_path / "checkout", **{variable: "-k make"})

    assert completed.returncode == 2
    assert f"{variable} is set" in completed.stderr
    assert evidence.read_text(encoding="utf-8") == "{}\n"


@pytest.mark.skipif(shutil.which("flock") is None, reason="needs flock")
def test_a_second_validate_is_refused(project_root: Path, tmp_path: Path) -> None:
    """Two at once: the second took the first's copy, and the evidence was lost."""
    import fcntl

    root = tmp_path / "checkout"
    evidence = _validate_checkout(project_root, root)
    (root / "output" / "validate").mkdir(parents=True)
    with (root / "output" / "validate" / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        completed = _validate(root)

    assert completed.returncode == 2
    assert "another make validate is running in this checkout" in completed.stderr
    assert evidence.read_text(encoding="utf-8") == "{}\n"
    assert not (root / "output" / "validate" / "previous").exists()


#: Stands in for the interpreter: the heavy steps are stubbed, the rest runs.
_STUB_PYTHON = """#!/bin/sh
export PYTHONPATH="{stubs}:${{PYTHONPATH:-}}"
output=""; previous=""
for argument in "$@"; do
  [ "$previous" = "--output" ] && output=$argument
  previous=$argument
done
case "$*" in
  "-m pytest -p no:cacheprovider --collect-only"*) ;;
  "-m pytest -m slow"*)
    if [ -n "${{SIGNAL_AT_SLOW:-}}" ]; then
      # Not before bash waits on the pipeline: a signal while it still forks
      # tee loses to this step's exit 1, and the EXIT trap runs instead.
      tries=0
      until [ "$(cat "/proc/$PPID/wchan" 2>/dev/null)" = do_wait ]; do
        tries=$((tries + 1))
        if [ "$tries" -gt 2000 ]; then echo "stub: bash never waited on the step" >&2; exit 97; fi
        sleep 0.01
      done
      kill -"$SIGNAL_AT_SLOW" "$PPID"; sleep 3; exit 1
    fi
    if [ -n "${{SIGNAL_AND_FAIL:-}}" ]; then kill -TERM "$PPID"; exit 1; fi
    echo "stub: $*"; exit 0 ;;
  "-m pytest"*|*run_isolated_test_files.py*) echo "stub: $*"; exit 0 ;;
  *ood_fuzz.py*)
    [ -n "${{FUZZ_SILENT:-}}" ] && exit 0
    printf '{{"failure_count": %s, "failures": []}}' "${{FUZZ_FAILURES:-0}}" > "$output"; exit 0 ;;
  *scenario_ood_matrix.py*)
    printf '{{"pipeline_exception_count": 0, "exceptions": []}}' > "$output"; exit 0 ;;
esac
exec "{python}" "$@"
"""


#: fake_cli_evidence.sh in a moment, not 17 to 24 s: what the verdict reads of a run that
#: passed, and, as the real one does, files the evidence held before, which a restore must undo.
_PASSING_CERTIFICATION = f"""#!/bin/sh
mkdir -p "$1/fake_certification/submission" "$1/fake_baseline/run"
echo '{_certification()}' > "$1/fake_certification/submission/certification.json"
echo '{{}}' > "$1/fake_baseline/baseline.json"
echo stub > "$1/fake_baseline/run/stdout.txt"
echo stub > "$1/fake_certification_cli.log"
"""


def _stubbed_checkout(
    project_root: Path, tmp_path: Path, *, certification: bool = False
) -> tuple[Path, Path]:
    """A checkout the whole campaign runs in, in seconds. Returns it and its interpreter.

    certification runs the real fake_cli_evidence.sh in place of _PASSING_CERTIFICATION.
    """
    root = tmp_path / "checkout"
    _validate_checkout(project_root, root, copy_src=True)
    for name in ("dependency_check.sh", "release_verdict.py"):
        shutil.copy2(project_root / "scripts" / name, root / "scripts")
    if certification:
        for name in ("fake_cli_evidence.sh", "fake_gmat_console.py"):
            shutil.copy2(project_root / "scripts" / name, root / "scripts")
        for name in ("config", "examples"):
            shutil.copytree(project_root / name, root / name)
        shutil.copytree(project_root / "tests" / "fixtures" / "gmat",
                        root / "tests" / "fixtures" / "gmat")
    else:
        (root / "scripts" / "fake_cli_evidence.sh").write_text(_PASSING_CERTIFICATION,
                                                               encoding="utf-8")
    # Results an earlier run left, which this run must not take for its own.
    (root / "output" / "validation" / "ood_fuzz_5000.json").write_text(
        '{"failure_count": 0, "failures": []}', encoding="utf-8"
    )
    (root / "output" / "validation" / "pip_check.exit_code").write_text("0\n", encoding="utf-8")
    (root / "output" / "validation" / "pip_check.log").write_text("from an earlier run\n", encoding="utf-8")
    stubs = tmp_path / "stubs"
    for tool in ("ruff", "mypy"):
        (stubs / tool).mkdir(parents=True)
        (stubs / tool / "__init__.py").write_text("", encoding="utf-8")
        (stubs / tool / "__main__.py").write_text("print('stub')\n", encoding="utf-8")
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text(_STUB_PYTHON.format(stubs=stubs, python=sys.executable), encoding="utf-8")
    python.chmod(0o755)
    return root, python


def test_the_stub_certification_passes_the_verdict(project_root: Path, tmp_path: Path) -> None:
    """Where it stands in, a run that fails must fail on its own cause alone."""
    root = _results(tmp_path / "validation", omit="fake_certification/submission/certification.json")
    stub = tmp_path / "fake_cli_evidence.sh"
    stub.write_text(_PASSING_CERTIFICATION, encoding="utf-8")
    subprocess.run(["sh", str(stub), str(root)], check=True)
    assert _verdict_module(project_root).verdict(root) == []


def test_a_validate_that_passes_writes_its_own_results(
    project_root: Path, tmp_path: Path
) -> None:
    """The verdict must read this run's results, and the copy must go when it passes.

    The dependency check's call could be dropped and an earlier run's exit code 0
    read in its place; a run that passed kept output/validate/previous.
    """
    root, python = _stubbed_checkout(project_root, tmp_path, certification=True)

    completed = _validate(root, PYTHON=str(python))

    assert completed.returncode == 0, completed.stdout[-1500:] + completed.stderr[-1500:]
    assert completed.stdout.rstrip().endswith("release validation: PASS")
    assert (root / "output" / "validation" / "pip_check.log").read_text() != "from an earlier run\n"
    assert (root / "output" / "validation" / "fake_certification" / "submission").is_dir()
    assert (root / "output" / "validation" / KEPT_BY_HAND).read_text() == "written by hand\n"
    assert not (root / "output" / "validate" / "previous").exists()
    assert not list((root / "output" / "validate").glob("failed-*"))


def test_a_validate_in_a_checkout_without_evidence_starts_it(
    project_root: Path, tmp_path: Path
) -> None:
    """A fresh checkout has no output/validation/; the run stopped at its snapshot."""
    root = tmp_path / "checkout"
    _validate_checkout(project_root, root, copy_src=True)
    shutil.rmtree(root / "output" / "validation")

    # Its slow step finds no slow test here (exit 5), so it fails past the snapshot.
    completed = _validate(root)

    assert completed.returncode == 5, completed.stderr[-1500:]
    assert "output/validation/ is restored" in completed.stderr
    assert list((root / "output" / "validation").iterdir()) == []
    failed = list((root / "output" / "validate").glob("failed-*"))
    assert len(failed) == 1 and (failed[0] / "environment.json").is_file()


def _commit_all(directory: Path) -> str:
    """A repository at directory holding everything in it, committed. Returns HEAD."""
    identity = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    from conftest import scratch_git_environment  # here: _validate_checkout copies no conftest

    environment = scratch_git_environment(directory)
    for command in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "c"]):
        subprocess.run(["git", "-C", str(directory), *command], check=True,
                       env={**environment, **identity})
    return subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True, env=environment).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
@pytest.mark.parametrize("repository", ["own", "outer"])
def test_the_evidence_names_its_own_commit(
    project_root: Path, tmp_path: Path, repository: str
) -> None:
    """make package builds HEAD; a tree without .git took the commit of a repository above it."""
    root = tmp_path / "checkout"
    _validate_checkout(project_root, root, copy_src=True)
    shutil.rmtree(root / "output" / "validation")
    head = _commit_all(root if repository == "own" else tmp_path)

    completed = _validate(root)  # fails at the slow step, past environment.json

    assert completed.returncode == 5, completed.stderr[-1500:]
    (failed,) = (root / "output" / "validate").glob("failed-*")
    environment = json.loads((failed / "environment.json").read_text(encoding="utf-8"))
    if repository == "own":
        assert (environment["commit"], environment["tracked_changes"]) == (head, False)
    else:
        assert (environment["commit"], environment["tracked_changes"]) == (None, None)


def _ignored_at_start(name: str) -> bool:
    """Whether SIG<name> is ignored in this process, and so in the bash it starts."""
    return signal.getsignal(getattr(signal, f"SIG{name}")) == signal.SIG_IGN


def test_a_stop_bash_cannot_trap_is_skipped_not_failed(
    project_root: Path, tmp_path: Path
) -> None:
    """Under nohup the HUP case failed 100 runs in 100."""
    case = ("tests/unit/test_make_targets.py::"
            "test_a_validate_that_fails_or_is_stopped_restores_the_evidence[hup]")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-rs",
         "--basetemp", str(tmp_path / "inner"), case],
        cwd=project_root, capture_output=True, text=True, check=False,
        env={**_clean_environment(), "PYTHONPATH": str(project_root / "src"),
             "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN),
    )
    assert completed.returncode == 0, completed.stdout[-1500:]
    assert "1 skipped" in completed.stdout
    assert "SIGHUP is ignored here (nohup?)" in completed.stdout


@pytest.mark.parametrize(
    ("environment", "status", "cause"),
    [({"FUZZ_FAILURES": "4"}, 1, "OOD fuzz: 4 failures"),
     ({"FUZZ_SILENT": "1"}, 1, "no result from this run: output/validation/ood_fuzz_5000.json"),
     ({"SIGNAL_AT_SLOW": "TERM"}, 143, None), ({"SIGNAL_AT_SLOW": "HUP"}, 129, None)],
    ids=["fuzz-failures", "fuzz-silent", "term", "hup"],
)
def test_a_validate_that_fails_or_is_stopped_restores_the_evidence(
    project_root: Path, tmp_path: Path, environment: dict[str, str], status: int,
    cause: str | None,
) -> None:
    """A failed result must fail the run; TERM or HUP left 97 files changed."""
    stopped = environment.get("SIGNAL_AT_SLOW")
    if stopped is not None and _ignored_at_start(stopped):
        pytest.skip(f"SIG{stopped} is ignored here (nohup?): bash cannot trap a signal "
                    "ignored when it starts")
    root, python = _stubbed_checkout(project_root, tmp_path)
    _age(root / "output" / "validation")
    before = _tree_bytes(root / "output" / "validation")

    completed = _validate(root, PYTHON=str(python), **environment)

    assert completed.returncode == status, completed.stderr[-1500:]
    assert "release validation: PASS" not in completed.stdout
    assert f"release validation: FAILED (exit {status}). output/validation/ is restored" in (
        completed.stderr
    )
    if cause is not None:
        assert cause in completed.stderr, completed.stderr[-1500:]
    assert _tree_bytes(root / "output" / "validation") == before


def test_a_snapshot_cut_short_is_never_taken_for_the_evidence(
    project_root: Path, tmp_path: Path
) -> None:
    """A kill or a full disk during the copy left a partial `previous` the next run trusted."""
    root = tmp_path / "checkout"
    evidence = _validate_checkout(project_root, root)
    tools = tmp_path / "failing-cp"
    tools.mkdir()
    (tools / "cp").write_text('#!/bin/sh\nmkdir -p "$3/half"\nexit 1\n', encoding="utf-8")
    (tools / "cp").chmod(0o755)

    completed = _validate(root, PATH=f"{tools}:{os.environ['PATH']}")

    assert completed.returncode == 1
    assert evidence.read_text(encoding="utf-8") == "{}\n"
    assert not (root / "output" / "validate" / "previous").exists()


def test_a_second_signal_does_not_stop_the_restore(project_root: Path, tmp_path: Path) -> None:
    """A signal between the restore's two moves left no validation/ and no FAILED line.

    Here mv itself sends TERM to the script as the restore starts.
    """
    root, python = _stubbed_checkout(project_root, tmp_path)
    before = _tree_bytes(root / "output" / "validation")
    tools = tmp_path / "signalling-mv"
    tools.mkdir()
    real_mv = shutil.which("mv")
    (tools / "mv").write_text(
        f'#!/bin/sh\n[ "$1 $2" = "-T output/validation" ] && kill -TERM "$PPID"\n'
        f'exec "{real_mv}" "$@"\n',
        encoding="utf-8",
    )
    (tools / "mv").chmod(0o755)

    completed = _validate(root, PYTHON=str(python), SIGNAL_AT_SLOW="TERM",
                          PATH=f"{tools}:{os.environ['PATH']}")

    assert completed.returncode == 143, completed.stderr[-1500:]
    assert "output/validation/ is restored" in completed.stderr
    assert _tree_bytes(root / "output" / "validation") == before


def test_a_signal_and_a_failure_together_still_restore(project_root: Path, tmp_path: Path) -> None:
    """The step fails as TERM arrives, so the EXIT trap and TERM's trap are both pending.

    Run that way, TERM's `exit` fired inside the EXIT trap and cut the restore
    short, with exit 1 and no message, in about half the runs here (seen once
    under load in the suite). Four runs, each of which must restore.
    """
    root, python = _stubbed_checkout(project_root, tmp_path)
    before = _tree_bytes(root / "output" / "validation")

    for _ in range(4):
        completed = _validate(root, PYTHON=str(python), SIGNAL_AND_FAIL="1")

        assert completed.returncode in (1, 143)
        assert "output/validation/ is restored" in completed.stderr, completed.returncode
        assert _tree_bytes(root / "output" / "validation") == before


def test_two_failures_in_one_second_keep_apart(project_root: Path, tmp_path: Path) -> None:
    """failed-<seconds> collided: the second nested into the first, the third was not restored.

    date here prints one instant for every run, as two runs in the same
    second (or nanosecond) would see.
    """
    root, python = _stubbed_checkout(project_root, tmp_path)
    before = _tree_bytes(root / "output" / "validation")
    tools = tmp_path / "frozen-date"
    tools.mkdir()
    (tools / "date").write_text("#!/bin/sh\necho 20261104T120000.000000000\n", encoding="utf-8")
    (tools / "date").chmod(0o755)

    for _ in range(3):
        completed = _validate(root, PYTHON=str(python), FUZZ_FAILURES="1",
                              PATH=f"{tools}:{os.environ['PATH']}")

        assert completed.returncode == 1
        assert "OOD fuzz: 1 failures" in completed.stderr, completed.stderr[-800:]
        assert "output/validation/ is restored" in completed.stderr, completed.stderr[-800:]
        assert _tree_bytes(root / "output" / "validation") == before
    failed = sorted((root / "output" / "validate").glob("failed-*"))
    assert len(failed) == 3
    for directory in failed:
        assert not (directory / "validation").exists(), f"{directory.name} nests a run"
        assert (directory / "environment.json").is_file()


@pytest.mark.skipif(os.geteuid() == 0, reason="root lists any directory")
def test_package_clean_names_a_directory_it_cannot_list(project_root: Path, tmp_path: Path) -> None:
    """An unlistable directory raised a bare PermissionError past _run, which prints no cause."""
    spec = importlib.util.spec_from_file_location(
        "orbitwin_build_release_unlistable", project_root / "scripts" / "build_release.py"
    )
    assert spec and spec.loader
    build_release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_release)
    locked = tmp_path / "checkout" / "locked"
    locked.mkdir(parents=True)
    locked.chmod(0o000)
    try:
        with pytest.raises(RuntimeError, match=r"clean: cannot list .*locked"):
            build_release.clean(tmp_path / "checkout")
    finally:
        locked.chmod(0o755)
