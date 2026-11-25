from __future__ import annotations

import dataclasses as dc
from pathlib import Path

import pytest
from conftest import quick_config, stamped

from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import STAGE_SCRIPT_NAME, CandidateStage, file_sha256
from orbitwin.search import search_one_burn
from orbitwin.verify import certify_candidate

# What a certification artifact is called. The stage a candidate carries and
# the stage a run establishes are different things, and only the second may
# name the file.

SUBMISSION_NAME = STAGE_SCRIPT_NAME[CandidateStage.CERTIFIED]


def _arriving_at(scenario, config, stage: CandidateStage):
    candidate = search_one_burn(scenario, config)[0]
    for step in (
        CandidateStage.PUBLIC_GMAT_VERIFIED,
        CandidateStage.ORGANIZER_GMAT_VERIFIED,
        CandidateStage.CERTIFIED,
    ):
        candidate.promote(step, run_id="elsewhere")
        if step is stage:
            break
    assert candidate.stage is stage
    return candidate


@pytest.mark.parametrize(
    "arriving",
    [CandidateStage.ORGANIZER_GMAT_VERIFIED, CandidateStage.CERTIFIED],
    ids=["organizer", "certified"],
)
def test_the_artifact_name_follows_this_run_not_the_candidate(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, arriving
) -> None:
    """Gating the promotion was not enough.

    A candidate already at the top never needed promoting. promote() is
    monotonic on purpose -- evidence obtained elsewhere was still obtained --
    so the stage survives, and the name was read off it. A candidate arriving
    CERTIFIED produced the submission filename from a public R2026a run with
    gmat.organizer_compatible unset. The `certified` case is that hole;
    `organizer` is the one the first fix covered, kept so both stay closed.
    """
    config = quick_config()
    assert config.gmat.organizer_compatible is False
    scenario, _ = parse_gmat_script(workshop_script, config)

    evidence = certify_candidate(
        workshop_script,
        stamped(_arriving_at(scenario, config, arriving), workshop_script, config),
        config,
        fake_gmat,
        fake_startup,
        tmp_path / "certify",
        runs=1,
    )

    expected = STAGE_SCRIPT_NAME[CandidateStage.PUBLIC_GMAT_VERIFIED]
    assert evidence["conferred_stage"] == CandidateStage.PUBLIC_GMAT_VERIFIED.value
    assert evidence["submission_script_name"] == expected
    assert not (tmp_path / "certify" / "submission" / SUBMISSION_NAME).exists()
    # The stage itself is not walked back: the evidence was obtained, somewhere.
    assert evidence["stage"] == arriving.value


def test_an_organizer_build_still_names_the_submission(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # The other half. A fix that costs the legitimate path its name is not a fix.
    base = quick_config()
    config = dc.replace(
        base,
        gmat=dc.replace(
            base.gmat,
            organizer_compatible=True,
            organizer_executable_sha256=file_sha256(fake_gmat),
        ),
    )
    scenario, _ = parse_gmat_script(workshop_script, config)

    evidence = certify_candidate(
        workshop_script,
        stamped(search_one_burn(scenario, config)[0], workshop_script, config),
        config,
        fake_gmat,
        fake_startup,
        tmp_path / "certify",
        runs=1,
    )

    assert evidence["conferred_stage"] == CandidateStage.CERTIFIED.value
    assert evidence["submission_script_name"] == SUBMISSION_NAME
    assert (tmp_path / "certify" / "submission" / SUBMISSION_NAME).is_file()
