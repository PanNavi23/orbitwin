"""orbitwin and the day's scripts refuse numpy, SciPy or PyYAML off the pins."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType

import pytest
from conftest import bare_environment

from orbitwin import cli, pinned
from orbitwin.errors import OrbitWinError


def _requirements(tmp_path: Path) -> Path:
    path = tmp_path / "requirements.txt"
    path.write_text(
        "# pins\nnumpy==2.4.6\nscipy==1.17.1  # the solver\nPyYAML==6.0.3\n", encoding="utf-8"
    )
    return path


def _installed(monkeypatch: pytest.MonkeyPatch, **versions: str) -> None:
    real = metadata.version

    def version(name: str) -> str:
        if name in versions:
            if versions[name] == "missing":
                raise metadata.PackageNotFoundError(name)
            return versions[name]
        return real(name)

    monkeypatch.setattr(pinned.metadata, "version", version)


def test_the_checkout_runs_on_its_own_pins() -> None:
    assert pinned.REQUIREMENTS.is_file()
    assert pinned.unpinned() == []


def test_another_version_or_a_missing_package_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(pinned.OVERRIDE, raising=False)
    _installed(monkeypatch, numpy="2.3.5", PyYAML="missing", scipy="1.17.1")

    with pytest.raises(OrbitWinError) as raised:
        pinned.assert_pinned_environment(_requirements(tmp_path))

    message = str(raised.value)
    assert "numpy 2.3.5 (pinned 2.4.6)" in message
    assert "PyYAML missing (pinned 6.0.3)" in message
    assert "scipy" not in message
    assert ". .venv/bin/activate" in message


def test_a_newer_version_is_refused_as_well(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.[dev]` alone resolved newer releases, not older ones."""
    monkeypatch.delenv(pinned.OVERRIDE, raising=False)
    _installed(monkeypatch, numpy="2.5.4", scipy="1.18.1")

    with pytest.raises(OrbitWinError, match=r"numpy 2\.5\.4 \(pinned 2\.4\.6\)"):
        pinned.assert_pinned_environment(_requirements(tmp_path))


def test_only_the_value_1_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _installed(monkeypatch, numpy="2.5.4")
    monkeypatch.setenv(pinned.OVERRIDE, "0")

    with pytest.raises(OrbitWinError):
        pinned.assert_pinned_environment(_requirements(tmp_path))


def test_the_override_and_a_tree_without_requirements_are_not_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed(monkeypatch, numpy="2.3.5")
    monkeypatch.setenv(pinned.OVERRIDE, "1")
    pinned.assert_pinned_environment(_requirements(tmp_path))
    monkeypatch.delenv(pinned.OVERRIDE)
    pinned.assert_pinned_environment(tmp_path / "absent.txt")


def test_orbitwin_refuses_before_doing_anything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workshop_script: Path
) -> None:
    monkeypatch.delenv(pinned.OVERRIDE, raising=False)
    _installed(monkeypatch, scipy="1.12.0")

    code = cli.main(["inspect", str(workshop_script)])

    captured = capsys.readouterr()
    assert code == 2
    assert "orbitwin: ERROR [environment]:" in captured.err
    assert "scipy 1.12.0 (pinned 1.17.1)" in captured.err
    assert captured.out == ""


def test_the_day_config_refuses_and_writes_nothing(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = importlib.util.spec_from_file_location(
        "orbitwin_ranking_day_config_pinned", project_root / "scripts" / "ranking_day_config.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv(pinned.OVERRIDE, raising=False)
    _installed(monkeypatch, numpy="2.3.5")
    output = tmp_path / "day.yaml"
    monkeypatch.setattr(
        "sys.argv", ["ranking_day_config.py", "--output", str(output), "--horizon-s", "6711.667"]
    )

    assert module.main() == 2
    assert "numpy 2.3.5 (pinned 2.4.6)" in capsys.readouterr().err
    assert not output.exists()


def _script_module(project_root: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(
        f"orbitwin_pinned_{name}", project_root / "scripts" / f"{name}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _main(module: ModuleType, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> int | None:
    monkeypatch.setattr(sys, "argv", argv)
    try:
        return int(module.main())
    except SystemExit as stopped:
        assert stopped.code is None or isinstance(stopped.code, int), stopped.code
        return stopped.code


#: The scripts a competition day may run besides orbitwin (the runbook's, and the
#: preliminary's hunt and floor tools), with
#: what they read first: none of it exists, so a refusal is the only way out.
_DAY_SCRIPTS = {
    "audit_burn_table": ["s.script", "burns.json", "--config", "day.yaml"],
    # As step 7 runs it: the refusal comes before the day's files are read.
    "check_report_rows": ["report.txt", "--config", "day.yaml", "--certificate", "c.json"],
    # By hand, with flags only.
    "check_report_rows by hand": ["report.txt", "--kt", "0.003", "--horizon-s", "6711.667"],
    "hunt_multi_burn": ["s.script", "--config", "day.yaml", "--output", "out.json"],
    "hunt_ranking_score": ["s.script", "--config", "day.yaml", "--output", "out.json",
                           "--candidates-output", "candidates.json"],
    "safe_delta_v_floor": ["s.script", "--config", "day.yaml", "--output", "out.json"],
}


@pytest.mark.parametrize("package", ["numpy", "scipy", "PyYAML"])
@pytest.mark.parametrize("name", sorted(_DAY_SCRIPTS))
def test_the_day_scripts_refuse_before_reading_anything(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], name: str, package: str,
) -> None:
    script = name.split()[0]
    module = _script_module(project_root, script, monkeypatch)
    monkeypatch.delenv(pinned.OVERRIDE, raising=False)
    _installed(monkeypatch, **{package: "0.1"})
    monkeypatch.chdir(tmp_path)

    code = _main(module, [f"{script}.py", *_DAY_SCRIPTS[name]], monkeypatch)

    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.startswith("refused: ")
    assert f"{package} 0.1 (pinned " in captured.err
    assert captured.out == ""
    assert list(tmp_path.iterdir()) == []


def test_the_override_lets_a_day_script_run(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _script_module(project_root, "audit_burn_table", monkeypatch)
    monkeypatch.setenv(pinned.OVERRIDE, "1")
    _installed(monkeypatch, numpy="0.1")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(OSError):
        _main(module, ["audit_burn_table.py", *_DAY_SCRIPTS["audit_burn_table"]], monkeypatch)
    assert "refused" not in capsys.readouterr().err


def _off_pin(tmp_path: Path) -> Path:
    """A directory whose numpy metadata says 0.1; numpy itself is still imported."""
    info = tmp_path / "off-pin" / "numpy-0.1.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: numpy\nVersion: 0.1\n",
                                   encoding="utf-8")
    return info.parent


def test_a_bare_environment_keeps_the_override(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workshop_script: Path
) -> None:
    """A suite run with the override failed where a test ran orbitwin with PYTHONPATH and PATH."""
    path = os.pathsep.join([str(_off_pin(tmp_path)), str(project_root / "src")])
    command = [sys.executable, "-m", "orbitwin", "inspect", str(workshop_script), "--rules",
               "preliminary"]

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False,
            env=bare_environment(PYTHONPATH=path, PATH="/usr/bin:/bin"),
        )

    monkeypatch.delenv(pinned.OVERRIDE, raising=False)
    refused = run()
    assert refused.returncode == 2 and "numpy 0.1 (pinned " in refused.stderr
    # The value as set, so 0 still refuses.
    monkeypatch.setenv(pinned.OVERRIDE, "0")
    assert run().returncode == 2
    monkeypatch.setenv(pinned.OVERRIDE, "1")
    ran = run()
    assert ran.returncode == 0, ran.stderr


def test_pr_answer_delta_passes_the_override_to_its_solves(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module(project_root, "pr_answer_delta", monkeypatch)
    assert module.PIN_OVERRIDE == pinned.OVERRIDE
    worktree = tmp_path / "worktree"
    for name in ("config/quick.yaml", "examples/two_burn_rendezvous.script"):
        (worktree / name).parent.mkdir(parents=True, exist_ok=True)
        (worktree / name).write_text("", encoding="utf-8")
    environments: list[dict[str, str]] = []

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        if "env" in options:
            environments.append(dict(options["env"]))  # type: ignore[call-overload]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setenv("ORBITWIN_TEST_STRAY", "x")
    for value in ("1", "0", None):
        if value is None:
            monkeypatch.delenv(pinned.OVERRIDE)
        else:
            monkeypatch.setenv(pinned.OVERRIDE, value)
        module._solve(worktree, "main", tmp_path / "out", sys.executable)

    assert [env.get(pinned.OVERRIDE) for env in environments] == ["1", "0", None]
    for env in environments:
        assert env["PYTHONPATH"] == str(worktree / "src")
        assert set(env) <= {"PYTHONPATH", "PATH", "HOME", pinned.OVERRIDE}, "still stripped"


#: Subprocesses that set PYTHONPATH on a bare environment and run nothing
#: that checks the pins.
_BARE_BY_DESIGN = {
    "tests/cross_profile/test_duel_side.py": "imports the duel profile only",
    "tests/unit/test_multi_burn.py": "runs stable_seed under a fixed PYTHONHASHSEED",
}


def test_a_test_that_strips_the_environment_keeps_the_override(project_root: Path) -> None:
    """A literal {"PYTHONPATH": ...} for env= dropped the override; bare_environment() keeps it."""
    import ast

    found: list[str] = []
    for path in sorted((project_root / "tests").rglob("*.py")):
        relative = path.relative_to(project_root).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.keyword) or node.arg != "env":
                continue
            value = node.value
            if not isinstance(value, ast.Dict) or None in value.keys:  # **os.environ keeps it
                continue
            names = {key.value for key in value.keys if isinstance(key, ast.Constant)}
            if "PYTHONPATH" in names and relative not in _BARE_BY_DESIGN:
                found.append(f"{relative}:{node.lineno}")

    assert found == [], "use bare_environment() from conftest"


def test_pr_answer_delta_reads_its_digests_where_solve_wrote_them(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative --out: solve wrote under the worktree, the digests were read here, every PR no-op."""
    module = _script_module(project_root, "pr_answer_delta", monkeypatch)
    seen: list[Path] = []
    monkeypatch.setattr(module, "_prs", lambda: [])
    monkeypatch.setattr(module, "_solve", lambda worktree, branch, out_root, python: (
        seen.append(out_root) or (None, None, "")))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["pr_answer_delta.py", "--worktree", "w", "--out", "rel/out"])

    assert module.main() == 0
    assert seen == [tmp_path / "rel" / "out"]
