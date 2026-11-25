"""Every day-path tool runs the ranking rules unless told otherwise.

policy.rules defaults to the preliminary, so `orbitwin inspect` with no
--config, or with one that sets no rules, ran the preliminary rules with a note
on stderr and exit 0; so did audit_burn_table.py, check_report_rows.py, the hunt
scripts, safe_delta_v_floor.py and competition_day.sh on such a config.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from orbitwin.cli import main
from orbitwin.config import OrbitWinConfig, PolicyConfig, load_config, rulebook_refusal

DAY = "The ranking day needs the day's config"
R1 = Path("examples") / "ranking" / "R1_NearPlane_Reachable.script"


def _preliminary(tmp_path: Path, score: bool = False) -> Path:
    """R1 under a config that sets no rules: the preliminary, by default."""
    raw: dict[str, object] = {"pursuer_name": "SC_Human", "target_name": "SC_Alien",
                              "policy": {"horizon_s": 6711.667}}
    if score:
        raw["score"] = {"kt": 0.003, "ct": 5369.3336, "kv": 0.0012, "cv": 2000.0,
                        "time_unit": "s", "delta_v_unit": "m/s"}
    path = tmp_path / ("prelim-scored.yaml" if score else "prelim.yaml")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _ranking(project_root: Path, tmp_path: Path) -> Path:
    raw = yaml.safe_load((project_root / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["horizon_s"] = 6711.667
    path = tmp_path / "ranking.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_only_the_ranking_rules_need_no_flag() -> None:
    ranking = OrbitWinConfig(policy=PolicyConfig(rules="ranking"))
    preliminary = OrbitWinConfig()
    assert rulebook_refusal(ranking, "day.yaml", None) is None
    assert rulebook_refusal(ranking, "day.yaml", "ranking") is None
    assert rulebook_refusal(preliminary, "quick.yaml", "preliminary") is None
    assert rulebook_refusal(preliminary, None, "preliminary") is None
    missing = rulebook_refusal(preliminary, None, None)
    assert missing is not None and missing.startswith("no --config was given")
    assert DAY in missing and "pass --rules preliminary" in missing
    unset = rulebook_refusal(preliminary, "quick.yaml", None)
    assert unset is not None and "quick.yaml is under the preliminary rules" in unset
    assert DAY in unset
    # A flag that names the other rulebook is not an override.
    other = rulebook_refusal(ranking, "day.yaml", "preliminary")
    assert other is not None and other.startswith("--rules preliminary, but day.yaml")
    other = rulebook_refusal(preliminary, None, "ranking")
    assert other is not None and "no --config was given, so this would run the preliminary" \
        in other


@pytest.mark.parametrize("command", ["inspect", "solve", "rank", "certify", "baseline",
                                     "verify-policy"])
def test_every_orbitwin_command_refuses_the_implicit_preliminary(
    command: str, workshop_script: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import orbitwin.cli as cli

    # Refused before any of them reads its script.
    monkeypatch.setattr(cli, "load_scenario", None)
    rest = {
        "inspect": [],
        "solve": ["--output", str(tmp_path / "out")],
        "rank": ["--candidates", str(tmp_path / "c.json")],
        "certify": ["--candidates", str(tmp_path / "c.json"), "--candidate", "x",
                    "--gmat-console", "g", "--startup-file", "s", "--output", str(tmp_path / "o")],
        "baseline": ["--gmat-console", "g", "--startup-file", "s", "--output",
                     str(tmp_path / "o")],
        "verify-policy": [str(workshop_script)],
    }[command]
    for config in ([], ["--config", str(_preliminary(tmp_path))]):
        assert main([command, str(workshop_script), *config, *rest]) == 2
        err = capsys.readouterr().err
        assert DAY in err and "--rules preliminary" in err, err
    assert not (tmp_path / "out").exists() and not (tmp_path / "o").exists()


def test_inspect_runs_the_preliminary_when_it_is_named(
    workshop_script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect", str(workshop_script), "--rules", "preliminary"]) == 0
    assert DAY not in capsys.readouterr().err
    # And a flag naming the rulebook the config is not under is refused.
    assert main(["inspect", str(workshop_script), "--rules", "ranking"]) == 2
    assert "--rules ranking, but no --config was given" in capsys.readouterr().err


def test_inspect_takes_the_ranking_config_without_a_flag(
    project_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    day = _ranking(project_root, tmp_path)
    assert main(["inspect", str(project_root / R1), "--config", str(day)]) == 0
    assert main(["inspect", str(project_root / R1), "--config", str(day),
                 "--rules", "preliminary"]) == 2
    assert f"--rules preliminary, but {day} is under the ranking rules" in (
        capsys.readouterr().err
    )


def _script(project_root: Path, name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(project_root / "scripts" / name), *args],
                          cwd=project_root, capture_output=True, text=True, check=False,
                          timeout=300)


#: Each tool's arguments after the config; the missing script fails after the rulebook.
_TOOLS = {
    "audit_burn_table.py": ["missing.script", "missing.json"],
    "hunt_multi_burn.py": ["missing.script"],
    "hunt_ranking_score.py": ["missing.script"],
    "safe_delta_v_floor.py": ["missing.script"],
}


@pytest.mark.parametrize("tool", sorted(_TOOLS))
def test_every_day_script_refuses_the_implicit_preliminary(
    tool: str, project_root: Path, tmp_path: Path
) -> None:
    config = str(_preliminary(tmp_path, score=True))
    refused = _script(project_root, tool, *_TOOLS[tool], "--config", config)
    assert refused.returncode == 2 and DAY in refused.stderr, refused.stderr
    named = _script(project_root, tool, *_TOOLS[tool], "--config", config,
                    "--rules", "preliminary")
    assert DAY not in named.stderr
    assert "missing.script" in named.stderr, named.stderr


def test_audit_runs_the_preliminary_when_it_is_named(project_root: Path, tmp_path: Path) -> None:
    burns = tmp_path / "burns.json"
    burns.write_text('{"burns": []}', encoding="utf-8")
    config = str(_preliminary(tmp_path))
    refused = _script(project_root, "audit_burn_table.py", str(R1), str(burns), "--config", config)
    assert refused.returncode == 2 and DAY in refused.stderr
    named = _script(project_root, "audit_burn_table.py", str(R1), str(burns), "--config", config,
                    "--rules", "preliminary")
    assert "AUDIT:" in named.stdout, named.stderr


def test_check_report_rows_refuses_a_preliminary_config(project_root: Path, tmp_path: Path) -> None:
    report = tmp_path / "report.txt"
    report.write_text("SC_Human.ElapsedSecs\n0\n", encoding="utf-8")
    config = str(_preliminary(tmp_path))
    refused = _script(project_root, "check_report_rows.py", str(report), "--config", config)
    assert refused.returncode == 2 and DAY in refused.stderr, refused.stderr
    named = _script(project_root, "check_report_rows.py", str(report), "--config", config,
                    "--rules", "preliminary")
    assert DAY not in named.stderr
    # --rules was an override of the config's rulebook; now it has to agree with it.
    day = str(_ranking(project_root, tmp_path))
    other = _script(project_root, "check_report_rows.py", str(report), "--config", day,
                    "--rules", "preliminary")
    assert other.returncode == 2 and "--rules preliminary, but" in other.stderr
    assert load_config(day).policy.rules == "ranking"


def test_certify_hands_its_rulebook_to_the_pre_check(
    workshop_script: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """check_report_rows refuses a preliminary config without --rules, as certify does."""
    import orbitwin.cli as cli
    from orbitwin.errors import GmatRunError

    class Named:
        candidate_id = "C-0"

    seen: list[list[list[str]]] = []
    monkeypatch.setattr(cli, "read_candidates", lambda _path: [Named()])
    monkeypatch.setattr(cli, "upload_summary", lambda *_a, **_k: [])
    monkeypatch.setattr(cli, "precheck_commands", lambda *_a: [["check", "report"]])
    monkeypatch.setattr(cli, "gui_precheck", lambda *_a: ["check", "gui-report"])
    monkeypatch.setattr(cli, "binding_precheck",
                        lambda commands, *_a: (seen.append(commands), ("done", {}))[1])
    monkeypatch.setattr(cli, "certify_candidate", lambda *_a, **_k: {})
    argv = ["certify", str(workshop_script), "--config", str(_preliminary(tmp_path)),
            "--rules", "preliminary", "--candidates", "c.json", "--candidate", "C-0",
            "--gmat-console", "g", "--startup-file", "s", "--output", str(tmp_path / "o")]
    assert main(argv) == 0

    def stopped(*_a: object, **_k: object) -> None:
        error = GmatRunError(cli.GUI_STOP)
        error.__dict__["kept_dir"] = str(tmp_path / "kept")
        raise error

    monkeypatch.setattr(cli, "certify_candidate", stopped)
    assert main(argv) == 2
    capsys.readouterr()
    assert seen == [[["check", "report", "--rules", "preliminary"]],
                    [["check", "gui-report", "--rules", "preliminary"]]]
