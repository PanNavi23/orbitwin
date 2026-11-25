from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from conftest import stamped

from orbitwin.config import OrbitWinConfig, validate_config
from orbitwin.errors import ConfigurationError
from orbitwin.models import CandidateStage, GmatRunResult, file_sha256
from orbitwin.verify import gmat_authority_stage

# CERTIFIED, and with it the name team_solution.script, is reachable only from
# ORGANIZER_GMAT_VERIFIED, and that tier is reachable only through
# gmat.organizer_compatible. So that flag is the whole distance between "we ran
# public R2026a this morning" and a filename asserting the organizer's build.


def _run(executable_sha256: str, startup_sha256: str | None = None) -> GmatRunResult:
    return GmatRunResult(
        run_id="r", success=True, return_code=0, started_at_utc="2026-08-20T00:00:00Z",
        duration_s=1.0, workdir="/tmp", script_path="/tmp/s.script",
        script_sha256="00" * 32, executable_sha256=executable_sha256,
        startup_sha256=startup_sha256, stdout_path="/tmp/o", stderr_path="/tmp/e",
        log_path=None, reports={}, report_sha256={},
    )


def _gmat(**overrides) -> OrbitWinConfig:
    base = OrbitWinConfig()
    return dataclasses.replace(base, gmat=dataclasses.replace(base.gmat, **overrides))


def test_declaring_the_organizer_build_requires_naming_the_binary() -> None:
    with pytest.raises(ConfigurationError, match="organizer_executable_sha256"):
        validate_config(_gmat(organizer_compatible=True))
    # With the fingerprint it is accepted.
    validate_config(_gmat(organizer_compatible=True, organizer_executable_sha256="ab" * 32))
    # And the flag off needs nothing.
    validate_config(_gmat(organizer_compatible=False))


def test_a_mismatched_binary_stays_at_the_public_tier() -> None:
    config = _gmat(organizer_compatible=True, organizer_executable_sha256="ab" * 32)
    assert gmat_authority_stage(config, _run("ab" * 32)) is CandidateStage.ORGANIZER_GMAT_VERIFIED
    assert gmat_authority_stage(config, _run("cd" * 32)) is CandidateStage.PUBLIC_GMAT_VERIFIED


def test_the_startup_file_can_be_pinned_too() -> None:
    # It selects the gravity, leap-second and EOP data, so it changes the
    # physics as surely as the binary does.
    config = _gmat(
        organizer_compatible=True,
        organizer_executable_sha256="ab" * 32,
        organizer_startup_sha256="ef" * 32,
    )
    assert gmat_authority_stage(
        config, _run("ab" * 32, "ef" * 32)
    ) is CandidateStage.ORGANIZER_GMAT_VERIFIED
    assert gmat_authority_stage(
        config, _run("ab" * 32, "99" * 32)
    ) is CandidateStage.PUBLIC_GMAT_VERIFIED
    assert gmat_authority_stage(config, _run("ab" * 32, None)) is CandidateStage.PUBLIC_GMAT_VERIFIED


def test_both_fingerprints_must_be_hex_digests() -> None:
    with pytest.raises(ConfigurationError, match="organizer_executable_sha256"):
        validate_config(_gmat(organizer_executable_sha256="not-a-digest"))
    with pytest.raises(ConfigurationError, match="organizer_startup_sha256"):
        validate_config(
            _gmat(organizer_executable_sha256="ab" * 32, organizer_startup_sha256="ab" * 31)
        )


def test_the_recorded_startup_hash_is_the_supplied_file(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path, workshop_script: Path
) -> None:
    """The fingerprint has to be of a file the organizer also has.

    run_gmat rewrites the startup file per run so runs cannot read each other's
    reports, and the recorded hash used to be of that rewritten copy: two runs
    of the same script against the same startup file reported 69469a0b and
    6399743e. It looked like an environment fingerprint and could never be
    compared with one.
    """
    from orbitwin.gmat_runner import run_gmat

    results = [
        run_gmat(
            fake_gmat, workshop_script, tmp_path / f"run{index}",
            ["ReportFile1.txt"], startup_file=fake_startup, timeout_s=120.0,
        )
        for index in (1, 2)
    ]
    supplied = file_sha256(fake_startup)
    assert [result.startup_sha256 for result in results] == [supplied, supplied]
    assert results[0].derived_startup_sha256 != results[1].derived_startup_sha256, (
        "the derived copies differ by construction, which is why they are not the fingerprint"
    )


def test_a_file_cannot_confer_a_gmat_tier(tmp_path: Path) -> None:
    """Evidence does not travel in JSON.

    Demonstrated before this was closed: a candidates.json claiming
    organizer_gmat_verified made `certify` produce team_solution.script from a
    public R2026a run with gmat.organizer_compatible unset. The realistic route
    is not a hand edit -- the briefing has private preliminary computation on
    our machines and the real runs on theirs, so a file written by a solve over
    there and certified over here is the ordinary way to do it.
    """
    import json

    from orbitwin.io import candidate_from_dict, read_candidates, write_candidates
    from orbitwin.models import Burn, Candidate

    burn = Burn(
        time_s=100.0, dv_inertial_km_s=(0.5, 0.0, 0.0),
        dv_script_km_s=(0.5, 0.0, 0.0), slot_index=0,
    )
    honest = Candidate(
        burns=(burn,), intercept_time_s=1000.0, minimum_distance_km=0.5,
        distance_at_completion_km=5.0, successful=True,
        analytical_endpoint_error_km=None, branch="test",
        metadata={"event_refined": True},
    )
    honest.promote(CandidateStage.RENDERABLE)
    path = tmp_path / "candidates.json"
    write_candidates(path, [honest])

    for claim in ("public_gmat_verified", "organizer_gmat_verified", "certified"):
        records = json.loads(path.read_text(encoding="utf-8"))
        records[0]["stage"] = claim
        path.write_text(json.dumps(records), encoding="utf-8")
        (restored,) = read_candidates(path)
        assert restored.stage is CandidateStage.RENDERABLE, claim
        assert restored.verified_by_gmat is False
        # Nothing is lost; it is recorded as a claim, which is all it is.
        assert restored.metadata["stage_claimed_by_file"] == claim

    # A record that claims nothing above RENDERABLE round-trips untouched.
    write_candidates(path, [honest])
    (round_tripped,) = read_candidates(path)
    assert round_tripped.stage is CandidateStage.RENDERABLE
    assert "stage_claimed_by_file" not in round_tripped.metadata
    assert candidate_from_dict(honest.to_dict()).stage is CandidateStage.RENDERABLE


def test_certification_gates_on_this_run_not_on_what_the_candidate_carries(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # The same refusal for a candidate that never came from a file. promote()
    # is monotonic, so one that reached the organizer tier somewhere else keeps
    # it, and would otherwise hand that authority to a run on another build.
    from conftest import quick_config

    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.search import search_one_burn
    from orbitwin.verify import certify_candidate

    config = quick_config()
    assert config.gmat.organizer_compatible is False
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    candidate.promote(
        CandidateStage.ORGANIZER_GMAT_VERIFIED, run_id="from-somewhere-else"
    )

    stamped(candidate, workshop_script, config)
    evidence = certify_candidate(
        workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "certify", runs=1
    )
    assert evidence["stage"] == CandidateStage.ORGANIZER_GMAT_VERIFIED.value
    assert evidence["submission_script_name"] != "team_solution.script"
    assert not (tmp_path / "certify" / "submission" / "team_solution.script").exists()
