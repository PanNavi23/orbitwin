from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import quick_config

from orbitwin.errors import GmatRunError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.pipeline import solve_scenario
from orbitwin.search import search_one_burn
from orbitwin.verify import verify_candidate_with_gmat


def _drifting(workshop_script: Path, tmp_path: Path) -> Path:
    script = tmp_path / "drift.script"
    script.write_text(
        "% FAKE_GMAT_EPOCH_DRIFT_S 3\n" + workshop_script.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return script


def test_a_replay_whose_epochs_drift_is_refused(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _drifting(workshop_script, tmp_path)
    config = quick_config()
    scenario, document = parse_gmat_script(script, config)
    candidate = search_one_burn(scenario, config)[0]
    with pytest.raises(GmatRunError, match="apart in epoch"):
        verify_candidate_with_gmat(
            script, scenario, document, candidate, config, fake_gmat, fake_startup,
            tmp_path / "verify",
        )


def test_solve_never_promotes_a_drifting_replay(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    summary = solve_scenario(
        _drifting(workshop_script, tmp_path), quick_config(), tmp_path / "solve",
        render_top=2, gmat_console=fake_gmat, startup_file=fake_startup, gmat_verify_top=2,
    )
    assert summary["best_candidate"] is None
    archived = json.loads((tmp_path / "solve" / "candidates.json").read_text("utf-8"))
    rejected = [c for c in archived if "gmat_verification_error" in c["metadata"]]
    assert rejected
    assert all("apart in epoch" in c["metadata"]["gmat_verification_error"] for c in rejected)
    assert all(c["verified_by_gmat"] is False for c in rejected)
