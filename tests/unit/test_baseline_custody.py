"""Baseline holds its run to the official bytes, and an interrupt stays an interrupt.

A changed or moved official script outranks the run's own failure, so it is
reported first. Ctrl-C or SystemExit while the script is mid-edit or gone
must still exit as an interrupt, not as a PolicyError or FileNotFoundError.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
from conftest import quick_config

import orbitwin.verify as verify_module
from orbitwin.errors import GmatRunError, PolicyError
from orbitwin.verify import baseline_original_script

FAKE = "f" * 64


def _wrap(monkeypatch: pytest.MonkeyPatch, hook):  # type: ignore[no-untyped-def]
    inner = verify_module.run_gmat

    def run(console, script, workdir, *args, **kwargs):  # type: ignore[no-untyped-def]
        return hook(inner, console, script, workdir, *args, **kwargs)

    monkeypatch.setattr(verify_module, "run_gmat", run)


def _copy_of(tmp_path: Path, workshop_script: Path) -> Path:
    script = tmp_path / "orig.script"
    script.write_bytes(workshop_script.read_bytes())
    return script


def test_baseline_holds_its_run_to_the_original_bytes(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hook(inner, console, script, workdir, *args, **kwargs):  # type: ignore[no-untyped-def]
        return dataclasses.replace(inner(console, script, workdir, *args, **kwargs),
                                   script_sha256=FAKE)

    _wrap(monkeypatch, hook)
    with pytest.raises(GmatRunError, match=rf"Baseline ran {FAKE}, not the official script"):
        baseline_original_script(workshop_script, quick_config(), fake_gmat, fake_startup,
                                 tmp_path / "b")


@pytest.mark.parametrize("raises", [False, True], ids=["fails", "raises"])
def test_a_changed_original_is_reported_before_the_failure(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, raises: bool,
) -> None:
    script = _copy_of(tmp_path, workshop_script)

    def hook(inner, console, run_script, workdir, *args, **kwargs):  # type: ignore[no-untyped-def]
        script.write_bytes(script.read_bytes() + b"% changed by the run\n")
        if raises:
            raise GmatRunError("GmatConsole could not be started")
        result = inner(console, run_script, workdir, *args, **kwargs)
        return dataclasses.replace(result, success=False, error="GMAT broke")

    _wrap(monkeypatch, hook)
    with pytest.raises(PolicyError, match="official script changed during baseline"):
        baseline_original_script(script, quick_config(), fake_gmat, fake_startup, tmp_path / "b")


def test_a_moved_original_is_a_policy_error(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _copy_of(tmp_path, workshop_script)

    def hook(inner, console, run_script, workdir, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = inner(console, run_script, workdir, *args, **kwargs)
        script.unlink()
        return result

    _wrap(monkeypatch, hook)
    with pytest.raises(PolicyError, match="official script was moved during baseline"):
        baseline_original_script(script, quick_config(), fake_gmat, fake_startup, tmp_path / "b")


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("moved", [False, True])
def test_an_interrupted_baseline_stays_interrupted(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, interrupt: type[BaseException], moved: bool,
) -> None:
    """Ctrl-C while the script is changed or gone exits 130, not as a PolicyError."""
    script = _copy_of(tmp_path, workshop_script)

    def hook(inner, console, run_script, workdir, *args, **kwargs):  # type: ignore[no-untyped-def]
        if moved:
            script.unlink()
        else:
            script.write_bytes(script.read_bytes() + b"% mid-edit\n")
        raise interrupt()

    _wrap(monkeypatch, hook)
    with pytest.raises(interrupt):
        baseline_original_script(script, quick_config(), fake_gmat, fake_startup, tmp_path / "b")


def test_a_changed_original_still_shows_the_runs_own_failure(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PolicyError leads, and the run's failure is not lost behind it."""
    script = _copy_of(tmp_path, workshop_script)

    def hook(inner, console, run_script, workdir, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = inner(console, run_script, workdir, *args, **kwargs)
        script.write_bytes(script.read_bytes() + b"% changed by the run\n")
        return dataclasses.replace(result, success=False, error="GMAT broke")

    _wrap(monkeypatch, hook)
    with pytest.raises(PolicyError, match="official script changed during baseline; the run "
                       "also failed: GMAT broke"):
        baseline_original_script(script, quick_config(), fake_gmat, fake_startup, tmp_path / "b")


def _append(extra: bytes):  # type: ignore[no-untyped-def]
    return lambda path: path.write_bytes(path.read_bytes() + extra)


#: (how the run leaves the official script, whether the run then fails, what is said)
CUSTODY = {
    "unchanged": (lambda path: None, True, GmatRunError, "exit status 1"),
    "moved-run-succeeded": (lambda path: path.unlink(), False, PolicyError, "moved"),
    "comment-added": (_append(b"% changed by the run\n"), True, PolicyError, "changed"),
    "unlinked": (lambda path: path.unlink(), True, PolicyError, "moved"),
    "renamed": (lambda path: path.rename(path.with_suffix(".moved")), True, PolicyError,
                "moved"),
    "nul-then-command": (_append(b"\x00\nCreate Spacecraft Extra;\n"), True, PolicyError,
                         "changed"),
    "non-ascii": (_append("% café\n".encode()), True, PolicyError, "changed"),
    "bom": (lambda path: path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes()), True,
            PolicyError, "changed"),
    "ff-in-code": (_append(b"Create\x0cSpacecraft Extra;\n"), True, PolicyError, "changed"),
    "vt-in-code": (_append(b"x\x0b= 1;\n"), True, PolicyError, "changed"),
    "ctrl-z-in-code": (_append(b"x = 1;\x1a\n"), True, PolicyError, "changed"),
    "emptied": (lambda path: path.write_bytes(b""), True, PolicyError, "changed"),
    "unreadable": (lambda path: path.chmod(0), True, PolicyError, "could not re-read"),
    "nul-then-command-run-succeeded": (_append(b"\x00\nCreate Spacecraft Extra;\n"), False,
                                       PolicyError, "changed"),
}


@pytest.mark.parametrize("case", [
    pytest.param(case, marks=pytest.mark.skipif(
        case == "unreadable" and os.geteuid() == 0, reason="root reads a mode-0 file"))
    for case in sorted(CUSTODY)
])
def test_custody_is_checked_before_the_failure_is_explained(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    """Explaining a failure read the official script again, so a moved or unreadable one
    raised FileNotFoundError or ParseError instead of the custody refusal."""
    mutate, fails, raised, said = CUSTODY[case]
    script = _copy_of(tmp_path, workshop_script)

    def hook(inner, console, run_script, workdir, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = inner(console, run_script, workdir, *args, **kwargs)
        mutate(script)
        if fails:
            result = dataclasses.replace(result, success=False, return_code=1,
                                         error="GmatConsole returned exit status 1")
        return result

    _wrap(monkeypatch, hook)
    try:
        with pytest.raises(raised, match=said) as caught:
            baseline_original_script(script, quick_config(), fake_gmat, fake_startup,
                                     tmp_path / "b")
    finally:
        if script.exists():
            script.chmod(0o644)
    if raised is PolicyError and fails:
        assert "the run also failed: GmatConsole returned exit status 1" in str(caught.value)


def test_a_script_changed_between_parse_and_run_is_not_run(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_gmat checks its copy against the bytes parsed, before GmatConsole starts."""
    import orbitwin.gmat_runner as gmat_runner

    script = _copy_of(tmp_path, workshop_script)
    started: list[object] = []
    real_popen = gmat_runner.subprocess.Popen

    def popen(*args, **kwargs):  # type: ignore[no-untyped-def]
        started.append(args)
        return real_popen(*args, **kwargs)

    def hook(inner, console, run_script, workdir, *args, **kwargs):  # type: ignore[no-untyped-def]
        script.write_bytes(script.read_bytes() + b"% after the parse\n")
        return inner(console, run_script, workdir, *args, **kwargs)

    _wrap(monkeypatch, hook)
    monkeypatch.setattr(gmat_runner.subprocess, "Popen", popen)
    with pytest.raises(PolicyError, match=r"official script changed during baseline; baseline also "
                       r"stopped: orig\.script is [0-9a-f]{64} now, not the"):
        baseline_original_script(script, quick_config(), fake_gmat, fake_startup, tmp_path / "b")
    # The state probe ran before; the baseline run's GmatConsole never started.
    run_dir = str((tmp_path / "b" / "run").resolve())
    assert started and not any(run_dir in " ".join(map(str, args[0])) for args in started)


def test_the_hash_held_is_of_the_bytes_parsed(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changed after the parse and before baseline hashed it: still not run."""
    import orbitwin.gmat_runner as gmat_runner

    script = _copy_of(tmp_path, workshop_script)
    real_load = verify_module.load_scenario
    started: list[object] = []
    real_popen = gmat_runner.subprocess.Popen

    def load(*args, **kwargs):  # type: ignore[no-untyped-def]
        loaded = real_load(*args, **kwargs)
        script.write_bytes(script.read_bytes() + b"% after the parse\n")
        return loaded

    def popen(*args, **kwargs):  # type: ignore[no-untyped-def]
        started.append(args)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(verify_module, "load_scenario", load)
    monkeypatch.setattr(gmat_runner.subprocess, "Popen", popen)
    with pytest.raises(PolicyError, match=r"official script changed during baseline; baseline also "
                       r"stopped: orig\.script is [0-9a-f]{64} now, not the"):
        baseline_original_script(script, quick_config(), fake_gmat, fake_startup, tmp_path / "b")
    run_dir = str((tmp_path / "b" / "run").resolve())
    assert not any(run_dir in " ".join(map(str, args[0])) for args in started)


@pytest.mark.parametrize("before", [True, False], ids=["before-the-parse", "after-the-parse"])
def test_certify_runs_only_the_final_bytes_it_approved(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch: pytest.MonkeyPatch, before: bool,
) -> None:
    """The exact runs are held to the bytes the policy guard and the flown-hash check
    approved, changed before or after the parse that names the reports."""
    import sys

    from conftest import stamped

    import orbitwin.gmat_runner as gmat_runner
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.search import zero_burn_candidate
    from orbitwin.verify import certify_candidate

    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = stamped(zero_burn_candidate(scenario, config), workshop_script, config)
    real_parse = verify_module.parse_gmat_script

    def parse(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        # certify_candidate runs its body in _certify.
        mine = sys._getframe(1).f_code.co_name == "_certify"
        if mine and before:
            Path(path).write_bytes(Path(path).read_bytes() + b"% before the parse\n")
        parsed = real_parse(path, *args, **kwargs)
        if mine and not before:
            Path(path).write_bytes(Path(path).read_bytes() + b"% after the parse\n")
        return parsed

    started: list[object] = []
    real_popen = gmat_runner.subprocess.Popen

    def popen(*args, **kwargs):  # type: ignore[no-untyped-def]
        started.append(args)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(verify_module, "parse_gmat_script", parse)
    monkeypatch.setattr(gmat_runner.subprocess, "Popen", popen)
    with pytest.raises(GmatRunError, match=r"^final\.script is [0-9a-f]{64} now, not the "
                       r"[0-9a-f]{64} that was checked"):
        certify_candidate(workshop_script, candidate, config, fake_gmat, fake_startup,
                          tmp_path / "certify", runs=1)
    run_dir = str((tmp_path / "certify" / "exact-run-01").resolve())
    assert started and not any(run_dir in " ".join(map(str, args[0])) for args in started)
