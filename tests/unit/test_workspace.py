"""Workspaces: `orbitwin init`, `run` and `status` over `orbitwin day`."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from test_day_command import R1, SCRIPT, Steps

from orbitwin import day as day_module
from orbitwin import workspace as ws
from orbitwin.cli import _parser, main
from orbitwin.day import EXIT_PAUSED, run_day

SCORE = {"kt_per_s": 0.003, "ct_s": 5369.3336, "kv_per_m_s": 0.0012, "cv_m_s": 2000,
         "cr_km": 300}


@pytest.fixture
def steps() -> Steps:
    return Steps()


@pytest.fixture
def made(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gmat: Path,
         fake_startup: Path) -> Path:
    """A workspace on the fake console, with R1 in input/; nothing announced."""
    monkeypatch.setattr(day_module, "DISK_LOW", tmp_path / "no-DISK_LOW")
    monkeypatch.setattr(day_module, "provenance", lambda root, expect, allow: {
        "source": "git", "commit": "c0ffee", "expected": "c0ffee", "on_main": True,
        "notes": []})
    stamps = iter(f"20261109T{n:06d}Z" for n in range(1000))
    monkeypatch.setattr(day_module, "_utc_stamp", lambda: next(stamps))
    root = tmp_path / "contest"
    assert main(["init", str(root), "--team-number", "54", "--gmat-console", str(fake_gmat),
                 "--startup-file", str(fake_startup), "--expect-commit", "c0ffee0"]) == 0
    shutil.copyfile(R1, root / "input" / R1.name)
    return root


def _announce(root: Path, text: str) -> None:
    (root / "announce.toml").write_text(text, encoding="utf-8")


def _full(root: Path, *, horizon: float = 6711.667, score: bool = True,
          extra: str = "") -> None:
    lines = [f"horizon_s = {horizon}", extra]
    if score:
        lines += ["[score]", *(f"{key} = {value}" for key, value in SCORE.items())]
    _announce(root, "\n".join(lines) + "\n")


def _run(root: Path, steps: Steps, *flags: str) -> int:
    args = _parser().parse_args(["run", str(root), "--min-free-gb", "0.001", *flags])
    return ws.run_command(args, lambda namespace: run_day(namespace, steps))


def _through(root: Path, steps: Steps, code: int) -> int:
    """Past each readback pause, as two people confirming it would."""
    for _ in range(4):
        if code != EXIT_PAUSED:
            return code
        folder = ws.day_folders(ws.read_workspace(root))[-1]
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        confirmed = state.get("confirmed_readbacks") or {}
        pending = [step for step, digest in (state.get("shown_readbacks") or {}).items()
                   if (confirmed.get(step) or {}).get("sha256") != digest]
        if not pending:
            return code
        code = _run(root, steps, "--confirm-readback", pending[0])
    return code


# -- init -------------------------------------------------------------------------------


def test_init_writes_a_workspace_that_reads_back(made: Path, fake_gmat: Path) -> None:
    workspace = ws.read_workspace(made)
    assert (workspace.rules, workspace.team_number) == ("tasa-odc-2026/ranking", 54)
    assert workspace.gmat_console == fake_gmat.absolute()
    assert sorted(path.name for path in made.iterdir()) == [
        "announce.toml", "days", "input", "orbitwin.toml", "upload"]
    # The template announces nothing: every value is commented out.
    announce = ws.read_announce(workspace)
    assert announce.missing() == ["horizon_s", *(f"score.{key}" for key in ws.SCORE_KEYS)]
    assert announce.decisions == {}


def test_init_leaves_an_existing_workspace_alone(made: Path,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    before = (made / "orbitwin.toml").read_bytes()
    assert main(["init", str(made), "--team-number", "7", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "workspace-exists"
    assert (made / "orbitwin.toml").read_bytes() == before


@pytest.mark.parametrize(("argv", "code"), [
    (["--rules", "tasa-odc-2026/preliminary"], "rules-unsupported"),
    (["--rules", "tasa-odc-2026/duel"], "rules-unsupported"),
    (["--rules", "no-such/rules"], "rules-unknown"),
    (["--gmat-console", "/bin/true"], "usage"),
    (["--gmat-console", "/no/such", "--startup-file", "/no/such/either"], "gmat-missing"),
])
def test_init_refuses_what_run_could_not_use(tmp_path: Path, argv: list[str], code: str,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["init", str(tmp_path / "w"), "--team-number", "54", "--json", *argv]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == code
    assert not (tmp_path / "w").exists()


def test_init_needs_the_team_number(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["init", str(tmp_path / "w")])


# -- reading --------------------------------------------------------------------------


@pytest.mark.parametrize(("change", "says"), [
    (("team_number = 54", "team_number = true"), "team_number"),
    (("team_number = 54", "team_number = 0"), "team_number"),
    (("format = 1", "format = 2"), "format"),
    (("format = 1", "format = 1\nteam = 54"), "unknown key team"),
    (("[gmat]", "[gmat]\nworker = 4"), "unknown key worker"),
])
def test_a_workspace_file_off_in_any_way_is_refused(made: Path, change: tuple[str, str],
                                                    says: str) -> None:
    file = made / "orbitwin.toml"
    text = file.read_text(encoding="utf-8")
    assert change[0] in text
    file.write_text(text.replace(change[0], change[1], 1), encoding="utf-8")
    with pytest.raises(ws.WorkspaceError, match=says) as raised:
        ws.read_workspace(made)
    assert raised.value.code == "workspace-invalid"


@pytest.mark.parametrize(("text", "says"), [
    ("horizon_s = true\n", "horizon_s must be a number"),
    ("horizon_s = -5\n", "positive finite"),
    ("horizon_s = nan\n", "positive finite"),
    ("horizon_s = \"6711\"\n", "must be a number"),
    ("tmax_s = 6711\n", "unknown key tmax_s"),
    ("[score]\ncr_km = 0\n", "positive finite"),
    ("[score]\nkt = 1\n", "unknown key kt"),
    ("[decisions]\nclosest_approach_window = \"all\"\n", "after_alignment, from_origin"),
    ("[decisions]\nno_head_start_burns = \"yes\"\n", "true or false"),
    ("horizon_s = 6711\nhorizon_s = 6712\n", "announce.toml"),
])
def test_an_announcement_off_in_any_way_is_refused(made: Path, text: str, says: str) -> None:
    _announce(made, text)
    with pytest.raises(ws.WorkspaceError, match=says) as raised:
        ws.read_announce(ws.read_workspace(made))
    assert raised.value.code == "announce-invalid"


def test_some_coefficients_but_not_all_are_named_and_refused(made: Path, steps: Steps,
                                                             capsys: Any) -> None:
    _announce(made, "horizon_s = 6711.667\n[score]\nkt_per_s = 0.003\ncr_km = 300\n")
    announce = ws.read_announce(ws.read_workspace(made))
    assert announce.score is None
    assert announce.missing() == ["score.ct_s", "score.kv_per_m_s", "score.cv_m_s"]
    assert _run(made, steps, "--json") == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "announce-incomplete" and "all five" in error["message"]
    assert not steps.calls


def test_the_input_folder_takes_one_script_and_carries_the_rest(made: Path) -> None:
    workspace = ws.read_workspace(made)
    (made / "input" / "data").mkdir()
    (made / "input" / "data" / "tide.dat").write_text("1\n", encoding="utf-8")
    (made / "input" / ".DS_Store").write_text("", encoding="utf-8")
    inputs = ws.find_inputs(workspace)
    assert inputs.script.name == R1.name and inputs.companions == ("data/tide.dat",)
    shutil.copyfile(R1, made / "input" / "other.SCRIPT")
    with pytest.raises(ws.WorkspaceError, match="exactly one") as raised:
        ws.find_inputs(workspace)
    assert raised.value.code == "template-ambiguous"
    for path in (made / "input").glob("*.*"):
        path.unlink()
    with pytest.raises(ws.WorkspaceError) as raised:
        ws.find_inputs(workspace)
    assert raised.value.code == "template-missing"


def test_every_announced_value_reaches_the_day_flag_of_its_name(made: Path) -> None:
    _full(made, extra="mission_boundary_km = 16900\n[decisions]\n"
          "closest_approach_window = \"from_origin\"\nno_head_start_burns = true\n"
          "tmax_as_announced = true\nreport_filename_as_written = \"R.txt\"\n"
          "accept_announced_horizon_before_perigee_s = 6711.667\n")
    workspace = ws.read_workspace(made)
    announce = ws.read_announce(workspace)
    argv = ws.fixed_argv(workspace, announce, ws.find_inputs(workspace))
    parsed = ws.day_namespace([*argv, *ws.score_argv(announce)])
    assert set(announce.decisions) == set(ws.DECISION_KEYS)
    for key, value in announce.decisions.items():
        assert getattr(parsed, key) == value
    assert (parsed.horizon_s, parsed.mission_boundary_km, parsed.team_number) == (
        6711.667, 16900.0, 54)
    assert {key: getattr(parsed, key) for key in ws.SCORE_KEYS} == {
        key: float(value) for key, value in SCORE.items()}
    assert Path(parsed.work_root) == made / "days" and Path(parsed.final_root) == made / "upload"


# -- run ---------------------------------------------------------------------------------


def test_run_without_t_max_starts_nothing(made: Path, steps: Steps,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(made, steps) == 2
    assert "ERROR [announce-incomplete]" in capsys.readouterr().err
    assert not steps.calls and ws.day_folders(ws.read_workspace(made)) == []


def test_run_carries_a_day_from_the_workspace_to_the_upload(
    made: Path, steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    _full(made, score=False)
    # Steps 1 to 4a, then the coefficients are not there yet.
    assert _through(made, steps, _run(made, steps)) == EXIT_PAUSED
    assert "coefficients are not given yet" in capsys.readouterr().err
    _full(made)
    assert _through(made, steps, _run(made, steps)) == 0
    workspace = ws.read_workspace(made)
    assert len(ws.day_folders(workspace)) == 1
    report = ws.status(workspace)
    assert report["days"][0]["delivered"] and report["problems"] == []
    assert Path(report["latest"]["script"]).name == SCRIPT
    assert Path(report["latest"]["script"]).parent.parent == made / "upload"
    assert report["next"].startswith("upload the two files")
    # Delivered: a further run says so and starts nothing.
    calls = len(steps.calls)
    assert _run(made, steps) == 2
    assert "ERROR [day-delivered]" in capsys.readouterr().err
    assert len(steps.calls) == calls


def test_a_changed_value_does_not_resume_the_day_folder_with_another(
    made: Path, steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    _full(made)
    assert _run(made, steps) == EXIT_PAUSED
    _full(made, horizon=6800.0)
    calls = len(steps.calls)
    assert _run(made, steps) == 2
    err = capsys.readouterr().err
    assert "ERROR [day-inputs-changed]" in err and "horizon_s: recorded 6711.667" in err
    assert len(steps.calls) == calls
    (made / "input" / R1.name).write_text("changed\n", encoding="utf-8")
    _full(made)
    assert _run(made, steps) == 2
    assert "the script: recorded SHA-256" in capsys.readouterr().err
    # --new takes the new values into a day folder of their own.
    shutil.copyfile(R1, made / "input" / R1.name)
    _full(made, horizon=6800.0)
    assert _run(made, steps, "--new") == EXIT_PAUSED
    folders = ws.day_folders(ws.read_workspace(made))
    assert len(folders) == 2
    recorded = json.loads((folders[-1] / "state.json").read_text(encoding="utf-8"))
    assert recorded["inputs"]["horizon_s"] == 6800.0


def test_run_json_keeps_stdout_for_its_summary(made: Path, steps: Steps,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    _full(made)
    assert _run(made, steps, "--json") == EXIT_PAUSED
    out = capsys.readouterr().out
    summary = json.loads(out)
    assert summary["exit"] == EXIT_PAUSED and len(summary["days"]) == 1


def test_run_passes_the_per_run_flags_and_nothing_fixed_on_a_resume(made: Path,
                                                                    steps: Steps) -> None:
    _full(made)
    assert _run(made, steps) == EXIT_PAUSED
    workspace = ws.read_workspace(made)
    args = _parser().parse_args(["run", str(made), "--confirm-readback", "2",
                                 "--exclude", "final-02", "--accept-rank-warnings"])
    plan = ws.plan_run(workspace, args)
    assert plan.resume == ws.day_folders(workspace)[-1]
    assert plan.argv[:2] == ["--resume", str(plan.resume)]
    parsed: argparse.Namespace = ws.day_namespace(plan.argv)
    assert parsed.confirm_readback == ["2"] and parsed.exclude == ["final-02"]
    assert parsed.accept_rank_warnings and parsed.script is None and parsed.horizon_s is None


# -- status -----------------------------------------------------------------------------


def test_status_names_what_is_missing_and_what_to_do(made: Path,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", str(made), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["announce_missing"][0] == "horizon_s" and report["days"] == []
    assert "horizon_s" in report["next"]
    assert main(["status", str(made)]) == 0
    out = capsys.readouterr().out
    assert "not yet   : horizon_s" in out and "next      :" in out


def test_status_of_a_folder_that_is_not_a_workspace(tmp_path: Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", str(tmp_path), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "workspace-missing"


# -- review findings ----------------------------------------------------------------------


def test_a_copied_workspace_does_not_resume_into_the_original_upload(
    made: Path, steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    _full(made)
    assert _run(made, steps) == EXIT_PAUSED
    copy = made.parent / "copy"
    shutil.copytree(made, copy, symlinks=True)
    calls = len(steps.calls)
    assert _run(copy, steps) == 2
    err = capsys.readouterr().err
    assert "ERROR [day-inputs-changed]" in err and "the upload folder: recorded" in err
    assert len(steps.calls) == calls


def test_status_reports_the_readbacks_confirmed(made: Path, steps: Steps) -> None:
    _full(made)
    assert _through(made, steps, _run(made, steps)) == 0
    report = ws.status(ws.read_workspace(made))
    assert report["days"][0]["readbacks_confirmed"] == ["2", "3", "5"]


@pytest.mark.parametrize("change", [
    {"uploads": ["x"]}, {"uploads": [{"folder": None}]}, {"steps": ["x"]},
    {"confirmed_readbacks": ["2"]}, {"inputs": ["x"]}, {"format": "other"},
])
def test_status_lists_a_mangled_state_file_rather_than_crashing(
    made: Path, steps: Steps, change: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _full(made)
    assert _run(made, steps) == EXIT_PAUSED
    folder = ws.day_folders(ws.read_workspace(made))[-1]
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    (folder / "state.json").write_text(json.dumps({**state, **change}), encoding="utf-8")
    # A ./latest where a folder of None would look for one must not count.
    monkeypatch.chdir(made.parent)
    (made.parent / "latest").symlink_to("None")
    assert main(["status", str(made), "--json"]) == 0
    day = json.loads(capsys.readouterr().out)["days"][-1]
    assert not day["delivered"]


def test_a_number_too_large_for_a_float_is_refused_as_such(made: Path) -> None:
    _announce(made, "horizon_s = " + "9" * 400 + "\n")
    with pytest.raises(ws.WorkspaceError, match="positive finite") as raised:
        ws.read_announce(ws.read_workspace(made))
    assert raised.value.code == "announce-invalid"


@pytest.mark.parametrize("argv", [["--confirm-readback", "9"], ["--no-such-flag"]])
def test_run_json_answers_a_usage_error_in_json(made: Path, argv: list[str],
                                                capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(made), "--json", *argv]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "usage" and error["message"]


def test_run_json_answers_an_unpinned_environment_in_json(
    made: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from orbitwin import cli
    from orbitwin.errors import OrbitWinError

    def refuse() -> None:
        raise OrbitWinError("numpy is not the pinned version")

    monkeypatch.setattr(cli, "assert_pinned_environment", refuse)
    assert main(["run", str(made), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "environment"


def test_a_file_named_like_a_flag_goes_with_the_script(made: Path) -> None:
    (made / "input" / "-notes.txt").write_text("x\n", encoding="utf-8")
    _full(made, extra="[decisions]\nreport_filename_as_written = \"-R.txt\"\n")
    workspace = ws.read_workspace(made)
    announce = ws.read_announce(workspace)
    parsed = ws.day_namespace(ws.fixed_argv(workspace, announce, ws.find_inputs(workspace)))
    assert parsed.companion == ["-notes.txt"]
    assert parsed.report_filename_as_written == "-R.txt"


def test_a_linked_folder_in_input_is_refused(made: Path, tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tide.dat").write_text("1\n", encoding="utf-8")
    (made / "input" / "data").symlink_to(tmp_path / "data")
    with pytest.raises(ws.WorkspaceError, match="link to a folder") as raised:
        ws.find_inputs(ws.read_workspace(made))
    assert raised.value.code == "input-name"


def test_only_day_named_folders_are_day_folders(made: Path, steps: Steps) -> None:
    _full(made)
    assert _run(made, steps) == EXIT_PAUSED
    workspace = ws.read_workspace(made)
    real = ws.day_folders(workspace)[-1]
    for name in ("backup-old", f"{real.name}.bak"):
        shutil.copytree(real, made / "days" / name)
    assert ws.day_folders(workspace) == [real]


def test_day_namespace_keeps_the_reason_of_a_refusal() -> None:
    with pytest.raises(ws.WorkspaceError) as raised:
        ws.day_namespace(["--no-such-flag"])
    assert "no-such-flag" in str(raised.value)


def test_init_leaves_nothing_behind_when_a_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    real = Path.open

    def failing(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if self.name == ws.WORKSPACE_FILE and "x" in mode:
            raise OSError(28, "No space left on device")
        return real(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing)
    root = tmp_path / "w"
    assert main(["init", str(root), "--team-number", "54", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "io"
    assert not (root / ws.ANNOUNCE_FILE).exists() and not (root / ws.WORKSPACE_FILE).exists()
    monkeypatch.setattr(Path, "open", real)
    assert main(["init", str(root), "--team-number", "54"]) == 0


def test_a_linked_gmat_pair_is_recorded_as_named(made: Path, tmp_path: Path,
                                                fake_gmat: Path, fake_startup: Path,
                                                steps: Steps) -> None:
    (tmp_path / "bin").mkdir()
    console, startup = tmp_path / "bin" / "GmatConsole", tmp_path / "bin" / "startup.txt"
    console.symlink_to(fake_gmat)
    startup.symlink_to(fake_startup)
    other = tmp_path / "linked"
    assert main(["init", str(other), "--team-number", "54", "--gmat-console", str(console),
                 "--startup-file", str(startup), "--expect-commit", "c0ffee0"]) == 0
    assert ws.read_workspace(other).gmat_console == console
    shutil.copyfile(R1, other / "input" / R1.name)
    _full(other)
    assert _run(other, steps) == EXIT_PAUSED
    # The resume finds nothing changed: the pair is the one recorded, as named.
    assert ws.plan_run(ws.read_workspace(other),
                       _parser().parse_args(["run", str(other)])).resume is not None


# -- second review round ----------------------------------------------------------------


def test_a_gmat_path_through_a_link_and_dotdot_is_kept_as_written(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path,
) -> None:
    (tmp_path / "real" / "deep").mkdir(parents=True)
    (tmp_path / "real" / "bin").mkdir()
    shutil.copyfile(fake_gmat, tmp_path / "real" / "bin" / "GmatConsole")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "GmatConsole").write_text("IMPOSTOR\n", encoding="utf-8")
    (tmp_path / "link").symlink_to(tmp_path / "real" / "deep")
    named = tmp_path / "link" / ".." / "bin" / "GmatConsole"
    root = tmp_path / "w"
    assert main(["init", str(root), "--team-number", "54", "--gmat-console", str(named),
                 "--startup-file", str(fake_startup)]) == 0
    console = ws.read_workspace(root).gmat_console
    assert console == named
    assert console is not None and "IMPOSTOR" not in console.read_text(encoding="utf-8")


@pytest.mark.parametrize("body", [b"# \xa0\xff Big5\nhorizon_s = 1\n",
                                  b"horizon_s = " + b"9" * 5000 + b"\n"])
def test_an_announcement_python_cannot_decode_is_a_refusal(
    made: Path, body: bytes, capsys: pytest.CaptureFixture[str],
) -> None:
    (made / "announce.toml").write_bytes(body)
    args = _parser().parse_args(["run", str(made), "--json"])
    assert ws.run_command(args, lambda namespace: pytest.fail("ran")) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "announce-invalid"
    assert main(["status", str(made), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["problems"][0]["code"] == "announce-invalid"


def test_status_lists_what_run_would_refuse(made: Path, steps: Steps,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    _full(made)
    assert _run(made, steps) == EXIT_PAUSED
    _full(made, horizon=6800.0)
    report = ws.status(ws.read_workspace(made))
    assert [problem["code"] for problem in report["problems"]] == ["day-inputs-changed"]
    assert report["next"] == "fix the problems listed"
    _announce(made, "horizon_s = 6711.667\n[score]\nkt_per_s = 0.003\n")
    report = ws.status(ws.read_workspace(made))
    assert [problem["code"] for problem in report["problems"]] == ["announce-incomplete"]


def test_a_second_run_on_the_same_workspace_is_refused(
    made: Path, steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    import fcntl

    _full(made)
    with (made / ".run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        assert _run(made, steps, "--json") == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "workspace-busy"
    assert not steps.calls


def test_init_refuses_a_folder_the_day_would_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(day_module, "GUARDED", tmp_path)
    assert main(["init", str(tmp_path / "w"), "--team-number", "54", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "usage"
    assert not (tmp_path / "w").exists()


def test_an_abbreviated_flag_is_not_taken(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["status", str(tmp_path), "--js"])


def test_an_unreadable_input_folder_is_said_to_be_unreadable(made: Path) -> None:
    folder = made / "input"
    folder.chmod(0)
    try:
        with pytest.raises(ws.WorkspaceError) as raised:
            ws.find_inputs(ws.read_workspace(made))
    finally:
        folder.chmod(0o755)
    assert raised.value.code == "io"


# -- third review round -----------------------------------------------------------------


def test_a_linked_script_is_refused(made: Path, tmp_path: Path) -> None:
    (tmp_path / "odc").mkdir()
    (made / "input" / R1.name).rename(tmp_path / "odc" / R1.name)
    (made / "input" / R1.name).symlink_to(tmp_path / "odc" / R1.name)
    with pytest.raises(ws.WorkspaceError, match="is a link") as raised:
        ws.find_inputs(ws.read_workspace(made))
    assert raised.value.code == "input-name"


def test_an_unreadable_subfolder_of_input_is_not_passed_over(made: Path) -> None:
    folder = made / "input" / "data"
    folder.mkdir()
    (folder / "grav.cof").write_text("1\n", encoding="utf-8")
    folder.chmod(0)
    try:
        with pytest.raises(ws.WorkspaceError, match="cannot be read") as raised:
            ws.find_inputs(ws.read_workspace(made))
    finally:
        folder.chmod(0o755)
    assert raised.value.code == "io"


def test_an_unreadable_script_is_a_problem_status_lists(made: Path, steps: Steps) -> None:
    _full(made)
    assert _run(made, steps) == EXIT_PAUSED
    script = made / "input" / R1.name
    script.chmod(0)
    try:
        report = ws.status(ws.read_workspace(made))
    finally:
        script.chmod(0o644)
    assert [problem["code"] for problem in report["problems"]] == ["io"]


def test_a_day_needs_the_commit_chosen_the_day_before(
    made: Path, steps: Steps, capsys: pytest.CaptureFixture[str],
) -> None:
    file = made / "orbitwin.toml"
    text = file.read_text(encoding="utf-8")
    assert 'expect_commit = "c0ffee0"' in text
    file.write_text(text.replace('expect_commit = "c0ffee0"', "# no commit"), encoding="utf-8")
    _full(made)
    assert [problem["code"] for problem in ws.status(ws.read_workspace(made))["problems"]] == [
        "commit-unset"]
    assert _run(made, steps, "--json") == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "commit-unset"
    assert not steps.calls
    assert _run(made, steps, "--allow-non-main") == EXIT_PAUSED


def test_changed_coefficients_are_listed_before_a_run(made: Path, steps: Steps) -> None:
    _full(made)
    assert _run(made, steps) == EXIT_PAUSED
    (made / "announce.toml").write_text(
        (made / "announce.toml").read_text(encoding="utf-8").replace("cv_m_s = 2000",
                                                                    "cv_m_s = 2100"),
        encoding="utf-8")
    problems = ws.status(ws.read_workspace(made))["problems"]
    assert [problem["code"] for problem in problems] == ["day-inputs-changed"]
    assert "score: recorded" in problems[0]["message"]


def test_status_says_when_a_run_holds_the_workspace(made: Path) -> None:
    import fcntl

    _full(made)
    workspace = ws.read_workspace(made)
    assert not ws.status(workspace)["running"]
    assert not (made / ".run.lock").exists()
    with (made / ".run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        report = ws.status(workspace)
    assert report["running"] and "running" in report["next"]


def test_init_writes_a_gmat_path_outside_the_basic_plane(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path,
) -> None:
    console = tmp_path / "GMAT \U0001f680" / "GmatConsole"
    console.parent.mkdir()
    shutil.copyfile(fake_gmat, console)
    root = tmp_path / "w"
    assert main(["init", str(root), "--team-number", "54", "--gmat-console", str(console),
                 "--startup-file", str(fake_startup)]) == 0
    assert ws.read_workspace(root).gmat_console == console


def test_no_resume_hint_when_no_day_folder_was_made(made: Path,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    _full(made)
    args = _parser().parse_args(["run", str(made)])
    assert ws.run_command(args, lambda namespace: 2) == 2
    assert "orbitwin day --resume" not in capsys.readouterr().err


# -- fourth review round ----------------------------------------------------------------


def test_status_names_allow_non_main_for_a_folder_started_with_it(made: Path,
                                                                 steps: Steps) -> None:
    file = made / "orbitwin.toml"
    file.write_text(file.read_text(encoding="utf-8").replace('expect_commit = "c0ffee0"',
                                                             "# none"), encoding="utf-8")
    _full(made)
    assert _run(made, steps, "--allow-non-main") == EXIT_PAUSED
    report = ws.status(ws.read_workspace(made))
    assert report["problems"] == []
    assert "--allow-non-main --confirm-readback 2`" in report["next"]


def test_a_commit_in_orbitwin_toml_that_is_not_one_is_refused(made: Path) -> None:
    file = made / "orbitwin.toml"
    file.write_text(file.read_text(encoding="utf-8").replace('"c0ffee0"', '"main"'),
                    encoding="utf-8")
    with pytest.raises(ws.WorkspaceError, match="not a commit") as raised:
        ws.read_workspace(made)
    assert raised.value.code == "workspace-invalid"


def test_run_waits_out_status_looking_at_the_lock(made: Path, steps: Steps) -> None:
    import fcntl
    import threading

    _full(made)
    handle = (made / ".run.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_SH)
    threading.Timer(0.2, handle.close).start()
    assert _run(made, steps) == EXIT_PAUSED


def test_status_lists_an_unreadable_note_beside_latest(made: Path, steps: Steps,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    _full(made)
    assert _through(made, steps, _run(made, steps)) == 0
    latest = made / "upload" / "latest"
    note = latest.resolve() / day_module.NOTE_NAME
    capsys.readouterr()
    note.chmod(0)
    try:
        assert main(["status", str(made), "--json"]) == 0
    finally:
        note.chmod(0o444)
    assert "error" in json.loads(capsys.readouterr().out)["latest"]


def test_init_writes_a_gmat_path_with_a_delete_character(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path,
) -> None:
    console = tmp_path / "GMAT\x7fbin" / "GmatConsole"
    console.parent.mkdir()
    shutil.copyfile(fake_gmat, console)
    root = tmp_path / "w"
    assert main(["init", str(root), "--team-number", "54", "--gmat-console", str(console),
                 "--startup-file", str(fake_startup)]) == 0
    assert ws.read_workspace(root).gmat_console == console


# -- fifth review round -----------------------------------------------------------------


def test_every_suggested_new_day_command_can_be_pasted(made: Path, steps: Steps) -> None:
    _full(made)
    assert _run(made, steps) == EXIT_PAUSED
    _full(made, horizon=6800.0)
    with pytest.raises(ws.WorkspaceError) as raised:
        ws.plan_run(ws.read_workspace(made), _parser().parse_args(["run", str(made)]))
    command = f"orbitwin run --new {made}`"
    assert command in str(raised.value)
    file = made / "orbitwin.toml"
    file.write_text(file.read_text(encoding="utf-8").replace('expect_commit = "c0ffee0"',
                                                             "# none"), encoding="utf-8")
    folder = ws.day_folders(ws.read_workspace(made))[-1]
    (folder / "state.json").write_text("{", encoding="utf-8")
    report = ws.status(ws.read_workspace(made))
    assert f"orbitwin run --new {made} --allow-non-main`" in report["next"]


def test_status_names_the_readback_to_confirm_and_the_coefficients(made: Path,
                                                                  steps: Steps) -> None:
    _full(made, score=False)
    assert _run(made, steps) == EXIT_PAUSED
    report = ws.status(ws.read_workspace(made))
    assert report["days"][0]["readbacks_pending"] == ["2"]
    assert f"`orbitwin run {made} --confirm-readback 2`" in report["next"]
    assert _run(made, steps, "--confirm-readback", "2") == EXIT_PAUSED
    assert _run(made, steps, "--confirm-readback", "3") == EXIT_PAUSED
    report = ws.status(ws.read_workspace(made))
    assert report["days"][0]["readbacks_pending"] == []
    assert "pauses for the coefficients until [score]" in report["next"]


def test_a_day_folder_without_state_is_the_newest_and_unreadable(made: Path, steps: Steps,
                                                                capsys: Any) -> None:
    _full(made)
    assert _run(made, steps) == EXIT_PAUSED
    (made / "days" / "20991231T235959Z").mkdir()
    assert _run(made, steps, "--json") == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "day-unreadable"
    report = ws.status(ws.read_workspace(made))
    assert not report["days"][-1]["readable"] and "holds no state.json" in report["next"]


def test_init_names_an_empty_gmat_flag(tmp_path: Path, fake_startup: Path,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["init", str(tmp_path / "w"), "--team-number", "54", "--json",
                 "--gmat-console", "", "--startup-file", str(fake_startup)]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "usage" and "empty" in error["message"]


def test_min_free_gb_in_the_workspace_reaches_every_run(made: Path, steps: Steps) -> None:
    file = made / "orbitwin.toml"
    file.write_text(file.read_text(encoding="utf-8").replace(
        "# min_free_gb = 20.0", "min_free_gb = 0.001"), encoding="utf-8")
    _full(made)
    workspace = ws.read_workspace(made)
    plan = ws.plan_run(workspace, _parser().parse_args(["run", str(made)]))
    assert ws.day_namespace(plan.argv).min_free_gb == 0.001
    plan = ws.plan_run(workspace, _parser().parse_args(["run", str(made),
                                                        "--min-free-gb", "3"]))
    assert ws.day_namespace(plan.argv).min_free_gb == 3.0


# -- sixth review round -----------------------------------------------------------------


def test_status_names_the_warning_flag_a_readback_stop_asked_for(made: Path,
                                                                 steps: Steps) -> None:
    steps.readback["02-day-config"] = ["WARNING: T_max is off the boundary"]
    _full(made)
    assert _run(made, steps) == 2
    report = ws.status(ws.read_workspace(made))
    assert report["days"][0]["warnings_to_accept"] == ["--accept-config-warnings"]
    assert "--accept-config-warnings --confirm-readback 2`" in report["next"]
    # The command it names goes on.
    assert _run(made, steps, "--accept-config-warnings", "--confirm-readback", "2") == (
        EXIT_PAUSED)
    report = ws.status(ws.read_workspace(made))
    assert report["days"][0]["warnings_to_accept"] == []


def test_a_left_over_day_folder_names_the_one_before_it(made: Path, steps: Steps) -> None:
    _full(made)
    assert _run(made, steps) == EXIT_PAUSED
    earlier = ws.day_folders(ws.read_workspace(made))[-1]
    left = made / "days" / "20991231T235959Z"
    left.mkdir()
    with pytest.raises(ws.WorkspaceError) as raised:
        ws.plan_run(ws.read_workspace(made), _parser().parse_args(["run", str(made)]))
    assert "holds no state.json" in str(raised.value)
    assert f"`orbitwin run {made}` takes up {earlier.name} again" in str(raised.value)
    assert f"takes up {earlier.name} again" in ws.status(ws.read_workspace(made))["next"]
    # A day starting in it holds its lock: wait, never move it.
    import fcntl

    with (left / ".day.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        assert "a day is starting in it" in ws.status(ws.read_workspace(made))["next"]
    (left / ".day.lock").unlink()
    left.rmdir()
    assert _run(made, steps, "--confirm-readback", "2") == EXIT_PAUSED
