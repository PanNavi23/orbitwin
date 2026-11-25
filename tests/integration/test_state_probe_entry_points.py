"""Every entry point that holds a console reads the initial states through the probe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import quick_config

from orbitwin.cli import main
from orbitwin.config import OrbitWinConfig
from orbitwin.ephemeris import target_ephemeris_from_gmat
from orbitwin.errors import GmatRunError, ParseError, PolicyError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.gmat_shoot import correct_with_gmat
from orbitwin.io import read_candidates
from orbitwin.pipeline import solve_scenario
from orbitwin.search import search_one_burn
from orbitwin.state_probe import PROBE_REPORT_FILENAME, load_scenario
from orbitwin.verify import baseline_original_script, certify_candidate


def test_solve_with_a_console_probes_first(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    solve_scenario(
        workshop_script, quick_config(), tmp_path / "solve",
        gmat_console=fake_gmat, startup_file=fake_startup, gmat_verify_top=0,
    )
    probed = tmp_path / "solve" / "state_probe" / "run" / "gmat-output" / PROBE_REPORT_FILENAME
    assert probed.is_file()
    written = json.loads((tmp_path / "solve" / "scenario.json").read_text(encoding="utf-8"))
    assert written["state_probe"]["craft"][0]["name"] == "HumanSC"
    assert written["epoch_offset_s"] == 0.0
    # The record points at the copy that was kept, not at the deleted scratch.
    assert written["state_probe"]["run"]["reports"][PROBE_REPORT_FILENAME] == str(probed.resolve())


def test_a_failed_probe_stops_solve(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    path = tmp_path / "probe_fails.script"
    path.write_text(
        "% FAKE_GMAT_PROBE_FAIL\n" + workshop_script.read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(GmatRunError, match="state probe failed"):
        solve_scenario(
            path, quick_config(), tmp_path / "solve",
            gmat_console=fake_gmat, startup_file=fake_startup, gmat_verify_top=0,
        )
    assert not (tmp_path / "solve").exists()
    solve_scenario(
        workshop_script, quick_config(), tmp_path / "solve",
        gmat_console=fake_gmat, startup_file=fake_startup, gmat_verify_top=0,
    )


def test_a_refused_baseline_leaves_nothing_and_can_be_retried(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    path = tmp_path / "probe_fails.script"
    path.write_text(
        "% FAKE_GMAT_PROBE_FAIL\n" + workshop_script.read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(GmatRunError, match="state probe failed"):
        baseline_original_script(path, OrbitWinConfig(), fake_gmat, fake_startup, tmp_path / "b")
    assert not (tmp_path / "b").exists()
    evidence = baseline_original_script(
        workshop_script, OrbitWinConfig(), fake_gmat, fake_startup, tmp_path / "b"
    )
    assert evidence["scenario"]["state_probe"]["craft"][0]["name"] == "HumanSC"
    assert (tmp_path / "b" / "state_probe" / "state_probe.script").is_file()


def test_a_refused_certification_leaves_nothing_and_can_be_retried(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    config = quick_config()
    # Probed, as certify reads it: 4 T_A from GMAT's state is about 42 µs off ours.
    scenario, _, _ = load_scenario(
        workshop_script, config, gmat_console=fake_gmat, startup_file=fake_startup
    )
    candidate = search_one_burn(scenario, config)[0]
    candidate.metadata["scenario_sha256"] = "not-this-script"
    with pytest.raises(PolicyError, match="was produced from a script hashing to"):
        certify_candidate(
            workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1
        )
    assert not (tmp_path / "c").exists()
    candidate.metadata.update(scenario_sha256=scenario.source_sha256, horizon_s=scenario.max_time_s)
    certify_candidate(
        workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1
    )


def test_inspect_takes_an_optional_console(
    workshop_script: Path, fake_gmat: Path, fake_startup: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect", str(workshop_script), "--rules", "preliminary"]) == 0
    assert json.loads(capsys.readouterr().out)["state_probe"] is None
    assert main([
        "inspect", str(workshop_script), "--rules", "preliminary",
        "--gmat-console", str(fake_gmat), "--startup-file", str(fake_startup),
    ]) == 0
    run = json.loads(capsys.readouterr().out)["state_probe"]["run"]
    assert run["success"] is True
    # Its directory is gone, so no path to it is recorded.
    assert run["workdir"] == run["script_path"] == run["stdout_path"] == ""
    assert set(run["reports"].values()) == {""}
    assert main(["inspect", str(workshop_script), "--rules", "preliminary",
                 "--gmat-console", str(fake_gmat)]) == 2
    assert "--startup-file must be supplied together" in capsys.readouterr().err


def _certify(workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path):
    config = quick_config()
    scenario, _, _ = load_scenario(
        workshop_script, config, gmat_console=fake_gmat, startup_file=fake_startup
    )
    candidate = search_one_burn(scenario, config)[0]
    candidate.metadata.update(scenario_sha256=scenario.source_sha256, horizon_s=scenario.max_time_s)
    return certify_candidate(
        workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "certify", runs=2
    )


def test_the_certificate_carries_the_probe_and_the_data_files(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    certification = _certify(workshop_script, tmp_path, fake_gmat, fake_startup)
    # The fake startup file names no data file; the workshop names a gravity file.
    assert set(certification["gmat_data_files"]) == {
        "script:DefaultProp_ForceModel.GravityField.Earth.PotentialFile"
    }
    assert "not found" not in str(certification["gmat_data_files"])
    assert certification["state_probe"]["craft"][1]["name"] == "AlienSC"
    assert (tmp_path / "certify" / "state_probe" / "state_probe.script").is_file()


def test_a_certificate_over_changing_data_files_is_refused(
    workshop_script: Path,
    tmp_path: Path,
    fake_gmat: Path,
    fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import orbitwin.gmat_runner as runner

    calls = iter(range(1000))
    monkeypatch.setattr(
        runner,
        "data_file_hashes",
        lambda *_: {"EOP_FILE": {"path": "eop", "sha256": "v1" if next(calls) < 3 else "v2"}},
    )
    with pytest.raises(GmatRunError, match="data files changed between the runs"):
        _certify(workshop_script, tmp_path, fake_gmat, fake_startup)
    assert not (tmp_path / "certify" / "submission").exists()


# --- a script only the probe can read, through every entry point ------------------
#
# One start instant in two DateFormats: refused in-process, read by the fake
# console's probe. Every place that re-parses a derived script must pass the
# probe on, or it refuses the script the run started from.

SCORE_YAML = "score: {kt: 0.001, ct: 5000.0, kv: 1.0, cv: 1.5, time_unit: s, delta_v_unit: km/s}\n"


def _two_formats(workshop_script: Path, tmp_path: Path) -> Path:
    text = workshop_script.read_text(encoding="utf-8")
    edited = text.replace(
        "AlienSC.DateFormat = UTCGregorian;\nAlienSC.Epoch = '15 Mar 2027 12:00:00.000';",
        "AlienSC.DateFormat = UTCModJulian;\nAlienSC.Epoch = '31480.0';",
    )
    assert edited != text
    path = tmp_path / "formats.script"
    path.write_text(edited, encoding="utf-8")
    return path


def test_solve_verifies_and_certify_certifies_a_script_only_the_probe_reads(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _two_formats(workshop_script, tmp_path)
    config = quick_config()
    with pytest.raises(ParseError, match="--gmat-console"):
        parse_gmat_script(script, config)
    # Verification re-parses each flown script and the renderer's guard the original.
    summary = solve_scenario(
        script, config, tmp_path / "solve",
        gmat_console=fake_gmat, startup_file=fake_startup, gmat_verify_top=1,
    )
    assert summary["gmat_verified_count"] == 1
    best = next(
        candidate
        for candidate in read_candidates(tmp_path / "solve" / "candidates.json")
        if candidate.candidate_id == summary["best_candidate"]["candidate_id"]
    )
    # certify's final replay re-parses the final script.
    certification = certify_candidate(
        script, best, config, fake_gmat, fake_startup, tmp_path / "certify", runs=1
    )
    assert certification["state_probe"] is not None


def test_the_coast_run_reads_a_script_only_the_probe_reads(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _two_formats(workshop_script, tmp_path)
    scenario, document, _ = load_scenario(
        script, OrbitWinConfig(), gmat_console=fake_gmat, startup_file=fake_startup
    )
    ephemeris = target_ephemeris_from_gmat(
        scenario, document, OrbitWinConfig(), fake_gmat, fake_startup, tmp_path / "coast"
    )
    assert ephemeris.times_s[-1] == pytest.approx(scenario.max_time_s)


def test_the_gmat_correction_flies_a_script_only_the_probe_reads(
    workshop_script: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _two_formats(workshop_script, tmp_path)
    config = quick_config()
    scenario, document, _ = load_scenario(
        script, config, gmat_console=fake_gmat, startup_file=fake_startup
    )
    candidate = search_one_burn(scenario, config)[0]
    # Each flight re-parses its rendered script; a refusal there would end round 1 unflown.
    result = correct_with_gmat(
        document, scenario, candidate, config, fake_gmat, fake_startup, tmp_path / "shoot"
    )
    assert result.gmat_runs >= 1, result.reason


def test_verify_policy_and_rank_read_it_through_the_probe(
    workshop_script: Path,
    tmp_path: Path,
    fake_gmat: Path,
    fake_startup: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _two_formats(workshop_script, tmp_path)
    console = ["--gmat-console", str(fake_gmat), "--startup-file", str(fake_startup)]
    preliminary = ["--rules", "preliminary"]
    assert main(["verify-policy", str(script), str(script), *preliminary]) == 2
    assert "--gmat-console" in capsys.readouterr().err
    assert main(["verify-policy", str(script), str(script), *preliminary, *console]) == 0
    capsys.readouterr()

    solve_scenario(
        script, quick_config(), tmp_path / "solve",
        gmat_console=fake_gmat, startup_file=fake_startup, gmat_verify_top=0,
    )
    config = tmp_path / "score.yaml"
    config.write_text(SCORE_YAML, encoding="utf-8")
    rank = [
        "rank", str(script), "--candidates", str(tmp_path / "solve" / "candidates.json"),
        "--config", str(config), *preliminary,
    ]
    assert main(rank) == 2
    assert "--gmat-console" in capsys.readouterr().err
    assert main([*rank, *console]) == 0


def test_certify_compares_the_probes_data_files_too(
    workshop_script: Path,
    tmp_path: Path,
    fake_gmat: Path,
    fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import orbitwin.gmat_runner as runner

    config = quick_config()
    scenario, _, _ = load_scenario(
        workshop_script, config, gmat_console=fake_gmat, startup_file=fake_startup
    )
    candidate = search_one_burn(scenario, config)[0]
    candidate.metadata.update(scenario_sha256=scenario.source_sha256, horizon_s=scenario.max_time_s)
    calls = iter(range(1000))
    # Only certify's first run, the probe, read another EOP.
    monkeypatch.setattr(
        runner,
        "data_file_hashes",
        lambda *_: {"EOP_FILE": {"path": "eop", "sha256": "v0" if next(calls) == 0 else "v1"}},
    )
    with pytest.raises(GmatRunError, match="data files changed between the runs"):
        certify_candidate(
            workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1
        )


@pytest.mark.parametrize("solved_with_console", [True, False])
def test_rank_says_why_a_4_ta_horizon_moved(
    workshop_script: Path,
    tmp_path: Path,
    fake_gmat: Path,
    fake_startup: Path,
    capsys: pytest.CaptureFixture[str],
    solved_with_console: bool,
) -> None:
    """Probed and in-process 4 T_A differ by about 42 µs; the refusal names the cause."""
    console = ["--gmat-console", str(fake_gmat), "--startup-file", str(fake_startup)]
    solve_scenario(
        workshop_script, quick_config(), tmp_path / "solve",
        gmat_console=fake_gmat if solved_with_console else None,
        startup_file=fake_startup if solved_with_console else None,
        gmat_verify_top=0,
    )
    config = tmp_path / "score.yaml"
    config.write_text(SCORE_YAML, encoding="utf-8")
    rank = [
        "rank", str(workshop_script), "--candidates", str(tmp_path / "solve" / "candidates.json"),
        "--config", str(config), "--rules", "preliminary",
        *([] if solved_with_console else console),
    ]
    assert main(rank) == 2
    error = capsys.readouterr().err
    assert "is being ranked under" in error
    assert "4 T_A" in error and "--gmat-console" in error


def test_rank_gives_an_announced_horizon_no_console_hint(
    workshop_script: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import dataclasses

    base = quick_config()
    solved = dataclasses.replace(base, policy=dataclasses.replace(base.policy, horizon_s=20000.0))
    solve_scenario(workshop_script, solved, tmp_path / "solve")
    config = tmp_path / "score.yaml"
    config.write_text(SCORE_YAML + "policy: {horizon_s: 21000.0}\n", encoding="utf-8")
    rank = [
        "rank", str(workshop_script), "--candidates", str(tmp_path / "solve" / "candidates.json"),
        "--config", str(config), "--rules", "preliminary",
    ]
    assert main(rank) == 2
    error = capsys.readouterr().err
    assert "policy.horizon_s differs" in error and "--gmat-console" not in error
