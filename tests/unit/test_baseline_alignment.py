"""Baseline on a template whose craft start apart with no alignment of its own.

GMAT aborts such a script as written ("Spacecraft epochs do not match"), or,
with a split Propagate, runs it with the craft their head start apart, and
policy.add_epoch_alignment makes solve and certify add one line. Baseline adds
the same line, and nothing else, to a copy and runs that, so the zero-burn
baseline exists for it too (R7, 2026-11-05: baseline exit 2, certify 23.1809).
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
from pathlib import Path

import pytest
from conftest import FAKE_STARTUP

from orbitwin import cli
from orbitwin.config import OrbitWinConfig, PolicyConfig
from orbitwin.errors import GmatRunError
from orbitwin.gmat_runner import gmat_errors, run_gmat
from orbitwin.script_renderer import with_added_alignment
from orbitwin.state_probe import load_scenario
from orbitwin.verify import ALIGNED_BASELINE_NAME, baseline_original_script

ROOT = Path(__file__).resolve().parents[2]
OPEN = ROOT / "examples" / "synthetic" / "NoManeuver_Circular.script"
FAKE = ROOT / "scripts" / "fake_gmat_console.py"
STARTUP = FAKE_STARTUP
ALIGN = "Propagate Prop(AlienSC) {AlienSC.A1ModJulian = Pursuer.A1ModJulian};\n"
JOINT = (
    "Propagate Prop(Pursuer, AlienSC) {Pursuer.ElapsedSecs = 30000, "
    "AlienSC.ElapsedSecs = 30000};\n"
)


def _config(**policy: object) -> OrbitWinConfig:
    return OrbitWinConfig(
        policy=dataclasses.replace(PolicyConfig(), **policy),  # type: ignore[arg-type]
        pursuer_name="Pursuer", target_name="AlienSC",
    )


def _asynchronous() -> OrbitWinConfig:
    return _config(mission_time_origin="target", closest_approach_window="after_alignment",
                   horizon_s=86400.0, add_epoch_alignment=True)


def _script(tmp_path: Path, mission: str, *, early: bool = True) -> Path:
    """test_epoch_alignment's template: AlienSC 15 h before Pursuer when ``early``."""
    resources = OPEN.read_text(encoding="utf-8").split("BeginMissionSequence;")[0]
    if early:
        resources = resources.replace(
            "AlienSC.Epoch = '26 Jun 2026 05:00:00.000';",
            "AlienSC.Epoch = '25 Jun 2026 14:00:00.000';",
        )
    path = tmp_path / "t.script"
    path.write_text(resources + "BeginMissionSequence;\n" + mission, encoding="utf-8")
    return path


OWN = "Propagate Prop(AlienSC) {AlienSC.UTCModJulian = Pursuer.UTCModJulian};\n"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_fake_aborts_a_joint_propagate_of_craft_apart(tmp_path: Path) -> None:
    """As R2026a did on R7; without this the double ran what GMAT refuses."""
    path = _script(tmp_path, JOINT)
    result = run_gmat(FAKE, path, tmp_path / "run", [], startup_file=STARTUP, timeout_s=60)
    assert not result.success
    assert [str(error) for error in gmat_errors(result).errors] == [
        "Command Exception: Spacecraft epochs do not match in a Propagator used in the "
        "Propagate command"
    ]


def test_baseline_runs_the_official_script_with_the_added_alignment(tmp_path: Path) -> None:
    path = _script(tmp_path, JOINT)
    before = _sha(path)

    evidence = baseline_original_script(
        path, _asynchronous(), FAKE, STARTUP, tmp_path / "b"
    )

    ran = Path(evidence["ran_script"])
    assert ran == (tmp_path / "b" / ALIGNED_BASELINE_NAME).resolve()
    assert evidence["epoch_alignment_added"] == ALIGN
    assert evidence["run"]["success"] is True
    assert evidence["ran_sha256"] == _sha(ran) == evidence["run"]["script_sha256"]
    assert evidence["original_sha256_before"] == evidence["original_sha256_after"] == before
    # One line added, as the first mission command, and nothing else changed.
    diff = [
        line for line in difflib.ndiff(
            path.read_text(encoding="utf-8").splitlines(True),
            ran.read_text(encoding="utf-8").splitlines(True),
        ) if line[:2] in ("+ ", "- ")
    ]
    assert diff == ["+ " + ALIGN]
    assert ran.read_text(encoding="utf-8").split("BeginMissionSequence;\n")[1].startswith(ALIGN)
    recorded = json.loads((tmp_path / "b" / "baseline.json").read_text(encoding="utf-8"))
    assert recorded["epoch_alignment_added"] == ALIGN


@pytest.mark.parametrize("mission", ["synchronous", "own alignment"])
def test_baseline_runs_the_official_script_itself_when_nothing_is_added(
    tmp_path: Path, mission: str
) -> None:
    if mission == "synchronous":
        path, config = _script(tmp_path, JOINT, early=False), _config(add_epoch_alignment=True)
    else:
        path, config = _script(tmp_path, OWN + JOINT), _asynchronous()

    evidence = baseline_original_script(path, config, FAKE, STARTUP, tmp_path / "b")

    assert evidence["epoch_alignment_added"] is None
    assert evidence["epoch_alignment_note"] is None
    assert evidence["ran_script"] == str(path.resolve())
    assert evidence["ran_sha256"] == evidence["original_sha256_before"]
    assert not (tmp_path / "b" / ALIGNED_BASELINE_NAME).exists()


def test_a_failed_aligned_baseline_names_the_copy_it_ran(tmp_path: Path) -> None:
    path = _script(tmp_path, "% FAKE_GMAT_FAIL\n" + JOINT)

    with pytest.raises(GmatRunError) as caught:
        baseline_original_script(
            path, _asynchronous(), FAKE, STARTUP, tmp_path / "b"
        )

    message = str(caught.value)
    assert f"baseline ran {(tmp_path / 'b' / ALIGNED_BASELINE_NAME).resolve()}" in message
    assert repr(ALIGN.strip()) in message
    assert "Spacecraft epochs do not match" in message


def test_the_cli_says_on_stderr_which_script_baseline_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _script(tmp_path, JOINT)
    config = _asynchronous()
    monkeypatch.setattr(cli, "load_config", lambda *args, **kwargs: config)

    code = cli.main([
        "baseline", str(path), "--gmat-console", str(FAKE), "--startup-file", str(STARTUP),
        "--output", str(tmp_path / "b"), "--rules", config.policy.rules,
    ])

    err = capsys.readouterr().err
    assert code == 0, err
    ran = (tmp_path / "b" / ALIGNED_BASELINE_NAME).resolve()
    assert f"orbitwin baseline: baseline ran {ran}, the official script with the epoch " in err
    assert "GMAT aborts the official script as written" in err
    assert "A GMAT GUI run of the zero burn runs this copy too, from beside" in err


SPLIT = (
    "Propagate Synchronized Prop(Pursuer) Prop(AlienSC) {Pursuer.ElapsedSecs = 30000, "
    "AlienSC.ElapsedSecs = 30000};\n"
)


def test_a_split_propagate_is_not_said_to_abort(tmp_path: Path) -> None:
    """R2026a runs the split form apart (rc 0, output/review/pr183/logs/real-R7-splitsync.json)."""
    path = _script(tmp_path, SPLIT)

    evidence = baseline_original_script(path, _asynchronous(), FAKE, STARTUP, tmp_path / "b")

    note = evidence["epoch_alignment_note"]
    assert evidence["epoch_alignment_added"] == ALIGN
    assert "split Propagate with both ElapsedSecs equal while the epochs stay apart" in note
    assert "aborts" not in note


def test_a_joint_propagate_is_said_to_abort(tmp_path: Path) -> None:
    evidence = baseline_original_script(
        _script(tmp_path, JOINT), _asynchronous(), FAKE, STARTUP, tmp_path / "b"
    )

    note = evidence["epoch_alignment_note"]
    assert 'GMAT aborts the official script as written ("Spacecraft epochs do not match")' in note
    assert "split" not in note


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["LF", "CRLF"])
def test_the_added_line_keeps_the_scripts_line_ending(tmp_path: Path, newline: str) -> None:
    path = _script(tmp_path, JOINT)
    path.write_bytes(path.read_text(encoding="utf-8").replace("\n", newline).encode("utf-8"))
    scenario, document, _ = load_scenario(
        path, _asynchronous(), gmat_console=FAKE, startup_file=STARTUP
    )

    added = with_added_alignment(document, scenario)

    line = ALIGN.replace("\n", newline).encode("utf-8")
    assert added.count(line) == 1
    assert added.replace(line, b"", 1) == path.read_bytes() == document.raw_bytes
