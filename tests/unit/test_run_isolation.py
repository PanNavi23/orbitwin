from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from conftest import quick_config

from orbitwin.errors import GmatRunError, OrbitWinError, PolicyError
from orbitwin.gmat_runner import run_gmat
from orbitwin.io import prepare_empty_output

# Two ways a run can quietly borrow something it did not produce: a directory
# that already holds an older run, and a GmatConsole with no startup file to
# give it its own OUTPUT_PATH.


def _quick(project_root: Path, tmp_path: Path) -> Path:
    """config/quick.yaml with conftest.quick_config's search.

    These tests check the day flow, not what the search finds; a day run takes 3 s, not 21 s.
    """
    import yaml

    raw = yaml.safe_load((project_root / "config" / "quick.yaml").read_text(encoding="utf-8"))
    raw["search"] = dataclasses.asdict(quick_config().search)
    path = tmp_path / "quick.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_a_new_or_empty_directory_is_accepted(tmp_path: Path) -> None:
    fresh = tmp_path / "new" / "nested"
    prepare_empty_output(fresh, "solve")
    assert fresh.is_dir()

    existing = tmp_path / "existing"
    existing.mkdir()
    prepare_empty_output(existing, "solve")
    assert existing.is_dir()


def test_a_directory_holding_an_earlier_run_is_refused(tmp_path: Path) -> None:
    # The shape that matters: an older run reached CERTIFIED and left the
    # submission name behind. A newer run whose best candidate is only
    # analytical would leave it sitting there, current-looking and unearned.
    used = tmp_path / "solve"
    used.mkdir()
    (used / "team_solution.script").write_text("stale", encoding="utf-8")
    with pytest.raises(PolicyError, match="already holds a run"):
        prepare_empty_output(used, "solve")
    assert (used / "team_solution.script").read_text(encoding="utf-8") == "stale", (
        "the old evidence is refused, not deleted"
    )


def test_a_hidden_file_still_counts_as_a_run(tmp_path: Path) -> None:
    used = tmp_path / "solve"
    used.mkdir()
    (used / ".keep").write_text("", encoding="utf-8")
    with pytest.raises(PolicyError):
        prepare_empty_output(used, "solve")


def test_solve_refuses_to_write_over_an_earlier_run(
    tmp_path: Path, workshop_script: Path
) -> None:
    from orbitwin.config import OrbitWinConfig
    from orbitwin.pipeline import solve_scenario

    output = tmp_path / "solve"
    output.mkdir()
    (output / "team_solution.script").write_text("from an older release", encoding="utf-8")
    with pytest.raises(PolicyError, match="solve output directory"):
        solve_scenario(workshop_script, OrbitWinConfig(), output)


def test_baseline_refuses_before_it_runs_gmat(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # baseline runs GMAT into output_dir/run, so the check has to come first.
    # Checking afterwards would be checking a directory this call had filled.
    from orbitwin.config import OrbitWinConfig
    from orbitwin.verify import baseline_original_script

    output = tmp_path / "baseline"
    output.mkdir()
    (output / "baseline.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PolicyError, match="baseline output directory"):
        baseline_original_script(
            workshop_script, OrbitWinConfig(), fake_gmat, fake_startup, output
        )
    assert not (output / "run").exists(), "it must refuse before GmatConsole is invoked"


def test_gmat_will_not_run_without_the_startup_file(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path
) -> None:
    # Without it there is no derived startup file, so no per-run OUTPUT_PATH,
    # so GMAT writes wherever its own installation points and the report this
    # function then looks for is simply absent -- a failure that says nothing
    # about the cause.
    with pytest.raises(GmatRunError, match="startup file"):
        run_gmat(fake_gmat, workshop_script, tmp_path / "run", ["ReportFile1.txt"])


def test_the_cli_refuses_half_a_gmat_pair() -> None:
    from orbitwin.cli import _require_gmat_pair

    _require_gmat_pair(None, None)
    _require_gmat_pair("/path/to/GmatConsole", "/path/to/gmat_startup_file.txt")
    with pytest.raises(OrbitWinError, match="must be supplied together"):
        _require_gmat_pair("/path/to/GmatConsole", None)
    with pytest.raises(OrbitWinError, match="must be supplied together"):
        _require_gmat_pair(None, "/path/to/gmat_startup_file.txt")


def test_inspect_creates_the_directory_it_writes_into(
    tmp_path: Path, workshop_script: Path
) -> None:
    # A fresh output directory is the normal competition-day case and inspect
    # is the first command run into it, so it cannot require one to exist.
    from orbitwin.cli import main

    destination = tmp_path / "brand" / "new" / "scenario.json"
    assert main(["inspect", str(workshop_script), "--output", str(destination),
                 "--rules", "preliminary"]) == 0
    assert destination.is_file()


def test_competition_day_keeps_every_attempt(
    tmp_path: Path, project_root: Path, workshop_script: Path, fake_gmat: Path,
    fake_startup: Path,
) -> None:
    """The documented flow is to look at the table and run again.

    solve refuses a directory that already holds a run, which is right and
    which must not stand in the way here: the rerun happens under time
    pressure on someone else's machine, after the score coefficients are
    announced. Each attempt gets its own numbered directory instead.
    """
    import os
    import subprocess
    import sys

    # A wrapper rather than a symlink: Python resolves a symlinked interpreter
    # back to its base and loses the virtualenv, and then the script cannot
    # import numpy.
    shim = tmp_path / "bin"
    shim.mkdir()
    launcher = shim / "python"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    environment = {
        **{key: value for key, value in os.environ.items()
           if not key.startswith("GMAT_") and key != "BASH_ENV"},
        "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(project_root / "src"),
        "GMAT_CONSOLE": str(fake_gmat),
        "GMAT_STARTUP": str(fake_startup),
    }
    output = tmp_path / "day"
    quick = _quick(project_root, tmp_path)
    for attempt in (1, 2, 3):
        completed = subprocess.run(
            [
                "bash", str(project_root / "scripts" / "competition_day.sh"), "--preliminary",
                str(workshop_script), str(output), str(quick),
            ],
            cwd=project_root, env=environment, capture_output=True, text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert f"search-{attempt:02d}" in completed.stdout

    kept = sorted(path.name for path in output.iterdir() if path.is_dir() and not path.is_symlink())
    assert kept == ["search-01", "search-02", "search-03"]
    assert (output / "latest").readlink() == Path("search-03")
    for name in kept:
        assert (output / name / "candidates.csv").is_file(), f"{name} lost its table"


def _day(project_root: Path, tmp_path: Path, script: Path, output: Path, config: Path,
         *flags: str, gmat: dict[str, Path] | None = None, preliminary: bool = True):
    """competition_day.sh with only the GMAT variables in `gmat` set; ``preliminary`` names
    the rules quick.yaml is under."""
    import subprocess

    named = ["--preliminary"] if preliminary else []
    return subprocess.run(
        ["bash", str(project_root / "scripts" / "competition_day.sh"), *named, *flags, str(script),
         str(output), str(config)],
        cwd=project_root, env=_day_environment(project_root, tmp_path, gmat),
        capture_output=True, text=True,
    )


def _day_environment(project_root: Path, tmp_path: Path, gmat: dict[str, Path] | None):
    import os
    import sys

    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    launcher = shim / "python"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    # BASH_ENV, sourced by every bash, can export the pair again.
    return {
        **{key: value for key, value in os.environ.items()
           if not key.startswith("GMAT_") and key != "BASH_ENV"},
        "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(project_root / "src"),
        **{key: str(value) for key, value in (gmat or {}).items()},
    }


@pytest.fixture
def gmat_pair(fake_gmat: Path, fake_startup: Path) -> dict[str, Path]:
    return {"GMAT_CONSOLE": fake_gmat, "GMAT_STARTUP": fake_startup}


def test_competition_day_leaves_no_latest_when_a_run_fails(
    tmp_path: Path, project_root: Path, workshop_script: Path, gmat_pair: dict[str, Path]
) -> None:
    """Else rank would read the older, unscored search through run/latest."""
    output = tmp_path / "day"
    quick = _quick(project_root, tmp_path)
    scored = tmp_path / "scored.yaml"
    scored.write_text(
        quick.read_text(encoding="utf-8")
        + "score:\n  kt: 0.001\n  ct: 5000.0\n  kv: 0.004\n  cv: 800.0\n"
        "  time_unit: s\n  delta_v_unit: m/s\n",
        encoding="utf-8",
    )
    first = _day(project_root, tmp_path, workshop_script, output, quick, gmat=gmat_pair)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "No coefficients in" in first.stdout
    assert (output / "latest").readlink() == Path("search-01")
    second = _day(project_root, tmp_path, workshop_script, output, scored, gmat=gmat_pair)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "Rank and certify" in second.stdout
    assert (output / "latest").readlink() == Path("search-02")
    failed = _day(project_root, tmp_path, tmp_path / "missing.script", output, scored,
                  gmat=gmat_pair)
    assert failed.returncode != 0
    assert not (output / "latest").exists() and not (output / "latest").is_symlink()


def test_competition_day_refuses_a_latest_that_is_not_a_symlink(
    tmp_path: Path, project_root: Path, workshop_script: Path, gmat_pair: dict[str, Path]
) -> None:
    output = tmp_path / "day"
    (output / "latest").mkdir(parents=True)
    done = _day(project_root, tmp_path, workshop_script, output,
                project_root / "config" / "quick.yaml", gmat=gmat_pair)
    assert done.returncode == 1 and "not a symlink" in done.stderr
    assert not (output / "search-01").exists()


@pytest.mark.parametrize("half", ["GMAT_CONSOLE", "GMAT_STARTUP"])
def test_competition_day_refuses_half_the_gmat_pair(
    tmp_path: Path, project_root: Path, workshop_script: Path, gmat_pair: dict[str, Path],
    half: str,
) -> None:
    """It used to say the pair was not set and solve without GMAT, exit 0 (A5-7)."""
    output = tmp_path / "day"
    for flags in ((), ("--analysis-only",)):
        done = _day(project_root, tmp_path, workshop_script, output,
                    project_root / "config" / "quick.yaml", *flags,
                    gmat={half: gmat_pair[half]})
        assert done.returncode == 2, done.stdout + done.stderr
        assert "GMAT_CONSOLE and GMAT_STARTUP must be set together" in done.stderr
        assert not output.exists()


def test_competition_day_refuses_to_run_without_gmat(
    tmp_path: Path, project_root: Path, workshop_script: Path
) -> None:
    """A1, 2026-11-04: without the pair, solve flew nothing in GMAT and exited 0, rank
    named a 96.34 hit and certify certified it as an 18.92 km miss."""
    output = tmp_path / "day"
    done = _day(project_root, tmp_path, workshop_script, output,
                project_root / "config" / "quick.yaml")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "GMAT_CONSOLE and GMAT_STARTUP are not set" in done.stderr
    assert "--analysis-only" in done.stderr
    assert not output.exists()


def test_an_analysis_only_run_stays_out_of_the_day_flow(
    tmp_path: Path, project_root: Path, workshop_script: Path, gmat_pair: dict[str, Path]
) -> None:
    output = tmp_path / "day"
    quick = _quick(project_root, tmp_path)
    day = _day(project_root, tmp_path, workshop_script, output, quick, gmat=gmat_pair)
    assert day.returncode == 0, day.stdout + day.stderr
    scenario = (output / "scenario.json").read_bytes()
    for attempt in (1, 2):
        done = _day(project_root, tmp_path, workshop_script, output, quick, "--analysis-only")
        assert done.returncode == 0, done.stdout + done.stderr
        assert "not the day flow" in done.stdout and "Rank and certify" not in done.stdout
        assert (output / f"analysis-{attempt:02d}" / "candidates.json").is_file()
    assert (output / "latest").readlink() == Path("search-01")
    assert not (output / "search-02").exists()
    assert (output / "scenario.json").read_bytes() == scenario
    refused = _day(project_root, tmp_path, workshop_script, output, quick, "--analysis-only",
                   gmat=gmat_pair)
    assert refused.returncode == 2 and "--analysis-only runs no GMAT" in refused.stderr
    assert not (output / "analysis-03").exists()


def test_analysis_only_is_read_wherever_it_stands(
    tmp_path: Path, project_root: Path, workshop_script: Path, gmat_pair: dict[str, Path]
) -> None:
    """Last on the line with the pair exported, it used to run the day flow."""
    import subprocess

    output = tmp_path / "day"
    quick = project_root / "config" / "quick.yaml"
    script = project_root / "scripts" / "competition_day.sh"
    for argv in ([str(workshop_script), str(output), str(quick), "--analysis-only"],
                 [str(workshop_script), "--analysis-only", str(output), str(quick)]):
        done = subprocess.run(
            ["bash", str(script), "--preliminary", *argv], cwd=project_root,
            capture_output=True, text=True,
            env=_day_environment(project_root, tmp_path, gmat_pair),
        )
        assert done.returncode == 2 and "--analysis-only runs no GMAT" in done.stderr
    assert not output.exists()
    for extra in (["--analysis"], [str(quick)]):
        done = _day(project_root, tmp_path, workshop_script, output, quick, *extra)
        assert done.returncode == 2 and "Usage: competition_day.sh" in done.stderr
    # A misspelt flag is not taken for a path, even where the count would pass.
    done = subprocess.run(
        ["bash", str(script), str(workshop_script), str(output), "--analysis-onyl", str(quick)],
        cwd=project_root, capture_output=True, text=True,
        env=_day_environment(project_root, tmp_path, None),
    )
    assert done.returncode == 2 and "Unknown option --analysis-onyl" in done.stderr
    # An unset variable expands to nothing; it is not a path.
    done = subprocess.run(
        ["bash", str(script), str(workshop_script), "", str(quick)], cwd=project_root,
        capture_output=True, text=True, env=_day_environment(project_root, tmp_path, gmat_pair),
    )
    assert done.returncode == 2 and "An empty argument. Usage" in done.stderr


def test_competition_day_takes_no_default_config(tmp_path: Path, project_root: Path) -> None:
    """Its old default, config/standard.yaml, solved under the preliminary rules."""
    import subprocess

    output = tmp_path / "day"
    completed = subprocess.run(
        ["bash", str(project_root / "scripts" / "competition_day.sh"),
         str(project_root / "examples" / "ranking" / "R1_NearPlane_Reachable.script"),
         str(output)],
        cwd=project_root, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "ranking_day_config.py" in completed.stderr
    assert not output.exists()


def test_competition_day_probes_when_a_console_is_set(
    tmp_path: Path, project_root: Path, workshop_script: Path, fake_gmat: Path,
    fake_startup: Path,
) -> None:
    import json
    import os
    import subprocess
    import sys

    shim = tmp_path / "bin"
    shim.mkdir()
    launcher = shim / "python"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    environment = {
        **{key: value for key, value in os.environ.items() if key != "BASH_ENV"},
        "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(project_root / "src"),
        "GMAT_CONSOLE": str(fake_gmat),
        "GMAT_STARTUP": str(fake_startup),
    }
    output = tmp_path / "day"
    completed = subprocess.run(
        [
            "bash", str(project_root / "scripts" / "competition_day.sh"), "--preliminary",
            str(workshop_script), str(output), str(_quick(project_root, tmp_path)),
        ],
        cwd=project_root, env=environment, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for written in (output / "scenario.json", output / "search-01" / "scenario.json"):
        assert json.loads(written.read_text(encoding="utf-8"))["state_probe"] is not None


def test_the_isolated_runner_covers_every_test_file(project_root: Path) -> None:
    """It used to name tests/unit and tests/integration.

    The release gate runs it in place of the single-process suite, so a third
    directory would have dropped out of the gate silently -- reported PASS
    having never opened the files. It discovers them now, and the marker
    expression does the selecting, exactly as `make test` does.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "runner", project_root / "scripts" / "run_isolated_test_files.py"
    )
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    discovered = {path.resolve() for path in runner.test_files(project_root)}
    on_disk = {path.resolve() for path in (project_root / "tests").rglob("test_*.py")}
    assert discovered == on_disk
    assert any(path.parent.name == "slow" for path in discovered), (
        "tests/slow is covered too; the marker expression is what deselects it"
    )
    assert runner.SELECTION == "not slow and not gmat", (
        "the runner must select what `make test` selects, or it covers something else"
    )


def test_competition_day_runs_only_the_ranking_rules_unless_told(
    tmp_path: Path, project_root: Path, workshop_script: Path, gmat_pair: dict[str, Path]
) -> None:
    """It took a config of any rules; a preliminary one ran with only orbitwin's note."""
    import yaml

    output = tmp_path / "day"
    (output / "search-01").mkdir(parents=True)
    (output / "latest").symlink_to("search-01")
    quick = project_root / "config" / "quick.yaml"
    for config in (quick, project_root / "config" / "standard.yaml"):
        done = _day(project_root, tmp_path, workshop_script, output, config, gmat=gmat_pair,
                    preliminary=False)
        assert done.returncode == 2, done.stdout + done.stderr
        assert "is under the preliminary rules (policy.rules defaults to preliminary), not " \
            "ranking." in done.stderr
        assert "--preliminary runs the preliminary rules on purpose" in done.stderr
        # Refused before anything ran or moved.
        assert (output / "latest").readlink() == Path("search-01")
        assert not (output / "search-02").exists()
    raw = yaml.safe_load((project_root / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["horizon_s"] = 30000.0
    ranking = tmp_path / "ranking.yaml"
    ranking.write_text(yaml.safe_dump(raw), encoding="utf-8")
    done = _day(project_root, tmp_path, workshop_script, output, ranking, gmat=gmat_pair)
    assert done.returncode == 2 and "--preliminary, but" in done.stderr
    assert (output / "latest").readlink() == Path("search-01")
    # A rule set's id names it as well as its name does.
    raw["policy"]["rules"] = "tasa-odc-2026/ranking"
    ranking.write_text(yaml.safe_dump(raw), encoding="utf-8")
    done = _day(project_root, tmp_path, workshop_script, output, ranking, gmat=gmat_pair)
    assert done.returncode == 2
    assert f"--preliminary, but {ranking} is under the ranking rules" in done.stderr
    preliminary = tmp_path / "preliminary.yaml"
    preliminary.write_text(yaml.safe_dump({"policy": {"rules": "tasa-odc-2026/preliminary"}}),
                           encoding="utf-8")
    done = _day(project_root, tmp_path, workshop_script, output, preliminary, gmat=gmat_pair,
                preliminary=False)
    assert done.returncode == 2 and "is under the preliminary rules" in done.stderr
    unreadable = tmp_path / "broken.yaml"
    unreadable.write_text("policy: [\n", encoding="utf-8")
    done = _day(project_root, tmp_path, workshop_script, output, unreadable, gmat=gmat_pair,
                preliminary=False)
    assert done.returncode == 2 and str(unreadable) in done.stderr
    assert (output / "latest").readlink() == Path("search-01")
