"""The pursuer's epoch as a design variable: the onsite ranking rules let a team set it."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import bare_environment

from orbitwin.cli import main
from orbitwin.design import DesignError, record_path, set_pursuer_epoch, verify_design
from orbitwin.rules import RANKING, RulePackError, interception_rules, read_rule_set

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
ONSITE = interception_rules("ranking-onsite")
ONSITE_CONFIG = ROOT / "config" / "ranking-onsite.yaml"
NAMES = ("SC_Human", "SC_Alien")


@pytest.fixture
def template(tmp_path: Path) -> Path:
    path = tmp_path / "official.script"
    shutil.copyfile(R1, path)
    return path


def _set(path: Path, after: float = 3600.0, **kwargs: object) -> tuple[Path, dict]:
    return set_pursuer_epoch(path, ONSITE, "SC_Human", "SC_Alien", after_target_s=after,
                             **kwargs)  # type: ignore[arg-type]


# -- the rules -----------------------------------------------------------------------


def test_only_the_onsite_rules_let_a_team_set_the_pursuers_epoch() -> None:
    assert (ONSITE.pursuer_epoch, ONSITE.source.header_date) == ("at-or-after-target",
                                                                 "2026/11/08")
    assert "入軌時刻" in str(ONSITE.pursuer_epoch_clause)
    for name in ("ranking", "preliminary"):
        rules = interception_rules(name)
        assert (rules.pursuer_epoch, rules.pursuer_epoch_clause) == ("fixed", None)


def test_a_pack_must_name_the_clause_exactly_when_the_epoch_is_free() -> None:
    text = (ROOT / "src/orbitwin/rulepacks/tasa_odc_2026/ranking-onsite.toml").read_text(
        encoding="utf-8")
    without = "\n".join(line for line in text.splitlines()
                        if not line.startswith("pursuer_epoch_clause"))
    with pytest.raises(RulePackError, match="pursuer_epoch_clause"):
        read_rule_set(without, "test")


def test_the_onsite_profile_is_rankings_under_the_onsite_rules() -> None:
    ranking = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    onsite = yaml.safe_load(ONSITE_CONFIG.read_text(encoding="utf-8"))
    del ranking["policy"]["head_start_min_gain_points"]
    ranking["policy"].update(rules="ranking-onsite", head_start_burns=False)
    assert onsite == ranking


# -- setting it ------------------------------------------------------------------------


def test_set_epoch_changes_that_one_line_and_records_it(template: Path) -> None:
    copy, record = _set(template)
    was = template.read_text(encoding="utf-8").splitlines(keepends=True)
    now = copy.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = [index for index, (a, b) in enumerate(zip(was, now, strict=True)) if a != b]
    assert changed == [record["line"] - 1]
    assert now[changed[0]] == "SC_Human.Epoch = '09 Nov 2026 03:00:00.000';\n"
    assert copy.parent == template.parent and copy.name.endswith("20261109T030000000Z.script")
    assert record["after_target_s"] == 3600.0 and record["rules"] == ONSITE.id
    assert json.loads(record_path(copy).read_text(encoding="utf-8")) == record
    assert verify_design(copy, ONSITE, *NAMES) == {**record, "verified_under": ONSITE.id}


def test_crlf_and_the_rest_of_the_line_are_kept(tmp_path: Path) -> None:
    text = R1.read_text(encoding="utf-8").replace(
        "SC_Human.Epoch = '09 Nov 2026 02:00:00.000';",
        "  SC_Human.Epoch = '09 Nov 2026 02:00:00.000';   % starts here")
    path = tmp_path / "crlf.script"
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    copy, _ = _set(path, 0.5)
    assert b"  SC_Human.Epoch = '09 Nov 2026 02:00:00.500';   % starts here\r\n" in (
        copy.read_bytes())
    assert copy.read_bytes().count(b"\r\n") == path.read_bytes().count(b"\r\n")


@pytest.mark.parametrize(("after", "kwargs", "says"), [
    (-1.0, {}, "may not precede"),
    (float("nan"), {}, "may not precede"),
    (7000.0, {"horizon_s": 7000.0}, "at or past T_max"),
    (0.0004, {}, "whole number of milliseconds"),
])
def test_an_epoch_the_rules_or_the_text_cannot_take_is_refused(
    template: Path, after: float, kwargs: dict, says: str,
) -> None:
    with pytest.raises(DesignError, match=says):
        _set(template, after, **kwargs)
    assert list(template.parent.iterdir()) == [template]


def test_rules_that_fix_the_epoch_refuse_to_set_it(template: Path) -> None:
    with pytest.raises(DesignError, match="fix the pursuer's epoch"):
        set_pursuer_epoch(template, RANKING, "SC_Human", "SC_Alien", after_target_s=10.0)


@pytest.mark.parametrize(("old", "new", "says"), [
    ("SC_Human.DateFormat = UTCGregorian;", "SC_Human.DateFormat = UTCModJulian;",
     "only UTCGregorian"),
    ("SC_Human.Epoch = '09 Nov 2026 02:00:00.000';",
     "SC_Human.Epoch = '09 Nov 2026 02:00:00.000';\nSC_Human.Epoch = '09 Nov 2026 02:00:00.000';",
     "must set Epoch once"),
    ("SC_Human.DateFormat = UTCGregorian;",
     "SC_Human.DateFormat = UTCGregorian;\nSC_Human.DateFormat = UTCGregorian;",
     "DateFormat at most once"),
    # GMAT would read the text in TAIModJulian's default, 37 s off (gmat_parser).
    ("SC_Human.DateFormat = UTCGregorian;\nSC_Human.Epoch = '09 Nov 2026 02:00:00.000';",
     "SC_Human.Epoch = '09 Nov 2026 02:00:00.000';\nSC_Human.DateFormat = UTCGregorian;",
     "comes before its DateFormat"),
    # Read without its `;`, it was taken for no DateFormat at all: TAIModJulian.
    ("SC_Human.DateFormat = UTCGregorian;", "SC_Human.DateFormat = UTCModJulian",
     "DateFormat is UTCModJulian"),
])
def test_a_template_it_cannot_read_one_way_is_refused(template: Path, old: str, new: str,
                                                      says: str) -> None:
    text = template.read_text(encoding="utf-8")
    assert old in text
    template.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(DesignError, match=says):
        _set(template)


# -- holding a copy to its record ---------------------------------------------------------


def test_a_copy_edited_elsewhere_is_refused(template: Path) -> None:
    copy, _ = _set(template)
    copy.write_text(copy.read_text(encoding="utf-8").replace("SC_Alien.SMA", "SC_Alien.SMA "),
                    encoding="utf-8")
    with pytest.raises(DesignError, match="not the copy its record lists"):
        verify_design(copy, ONSITE, *NAMES)


def test_a_changed_original_is_refused(template: Path) -> None:
    copy, _ = _set(template)
    template.write_text(template.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DesignError, match="not beside it as it was"):
        verify_design(copy, ONSITE, *NAMES)


def test_a_record_that_hides_a_second_change_is_refused(template: Path) -> None:
    """A copy and record made to match each other still differ from the original twice."""
    copy, record = _set(template)
    text = copy.read_text(encoding="utf-8").replace("SC_Alien.ECC", "SC_Alien.ECC ")
    copy.write_text(text, encoding="utf-8")
    from orbitwin.models import file_sha256

    record_path(copy).write_text(json.dumps({**record, "script_sha256": file_sha256(copy)}),
                                 encoding="utf-8")
    with pytest.raises(DesignError, match="not only SC_Human's epoch"):
        verify_design(copy, ONSITE, *NAMES)


def test_a_record_with_the_epoch_before_the_target_is_refused(template: Path) -> None:
    copy, record = _set(template)
    text = copy.read_text(encoding="utf-8").replace("'09 Nov 2026 03:00:00.000'",
                                                    "'09 Nov 2026 01:00:00.000'")
    copy.write_text(text, encoding="utf-8")
    from orbitwin.models import file_sha256

    record_path(copy).write_text(json.dumps({**record, "script_sha256": file_sha256(copy)}),
                                 encoding="utf-8")
    with pytest.raises(DesignError, match="precedes"):
        verify_design(copy, ONSITE, *NAMES)


def test_a_copy_read_under_rules_that_fix_the_epoch_is_refused(template: Path) -> None:
    copy, _ = _set(template)
    with pytest.raises(DesignError, match="fix the pursuer's epoch"):
        verify_design(copy, RANKING, *NAMES)


def test_a_script_without_a_record_is_left_alone(template: Path) -> None:
    assert verify_design(template, RANKING, *NAMES) is None


# -- the commands -----------------------------------------------------------------------


def test_set_epoch_from_the_command_line(template: Path, capsys: pytest.CaptureFixture[str]
                                         ) -> None:
    assert main(["set-epoch", str(template), "--config", str(ONSITE_CONFIG), "--rules",
                 "ranking-onsite", "--epoch", "09 Nov 2026 02:30:00.250", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["pursuer_epoch"] == "09 Nov 2026 02:30:00.250"
    assert out["after_target_s"] == 1800.25 and Path(out["script"]).is_file()


def test_set_epoch_under_the_ranking_rules_is_a_policy_refusal(
    template: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["set-epoch", str(template), "--config", str(ROOT / "config" / "ranking.yaml"),
                 "--after-target-s", "60"]) == 2
    assert capsys.readouterr().err.startswith("orbitwin: ERROR [policy]: ")


def test_every_command_holds_a_copy_to_its_record_first(
    template: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    copy, _ = _set(template)
    copy.write_text(copy.read_text(encoding="utf-8") + "% edited\n", encoding="utf-8")
    assert main(["inspect", str(copy), "--config", str(ONSITE_CONFIG), "--rules",
                 "ranking-onsite"]) == 2
    assert "not the copy its record lists" in capsys.readouterr().err


def test_the_day_config_tool_takes_the_onsite_rules(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ranking_day_config.py"), "--output",
         str(tmp_path / "day.yaml"), "--base", str(ONSITE_CONFIG), "--script", str(R1),
         "--horizon-s", "6711.667", "--tmax-as-announced", "--team-number", "54"],
        capture_output=True, text=True, check=False, timeout=300,
        env=bare_environment(PYTHONPATH=str(ROOT / "src"), PATH="/usr/bin:/bin"))
    assert done.returncode == 0, done.stderr
    assert "rules       : ranking-onsite" in done.stdout
    assert "(§6 note (a))" in done.stdout and "(§6 note (d))" in done.stdout


# -- second look -----------------------------------------------------------------------


def test_bytes_that_are_not_utf8_come_through_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "big5.script"
    path.write_bytes(R1.read_bytes() + "% 註解\n".encode("cp950"))
    copy, _ = _set(path)
    assert copy.read_bytes().endswith("% 註解\n".encode("cp950"))
    assert verify_design(copy, ONSITE, *NAMES) is not None


@pytest.mark.parametrize("after", [float("inf"), 1e15])
def test_an_offset_past_any_date_is_refused(template: Path, after: float) -> None:
    with pytest.raises(DesignError):
        _set(template, after)


def test_a_copy_is_not_set_again(template: Path) -> None:
    copy, _ = _set(template)
    with pytest.raises(DesignError, match="itself a copy"):
        _set(copy, 60.0)


def test_a_record_that_lacks_a_field_is_refused(template: Path) -> None:
    copy, record = _set(template)
    del record["script_sha256"]
    record_path(copy).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(DesignError, match="cannot be read"):
        verify_design(copy, ONSITE, *NAMES)


def test_a_copy_made_from_a_copy_is_refused(template: Path) -> None:
    copy, record = _set(template)
    record_path(template).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(DesignError, match="itself a copy"):
        verify_design(copy, ONSITE, *NAMES)


# -- review: roles, order, spellings ------------------------------------------------------


def test_a_record_that_swaps_the_craft_is_refused(template: Path) -> None:
    copy, _ = set_pursuer_epoch(template, ONSITE, "SC_Alien", "SC_Human",
                                after_target_s=3600.0)
    with pytest.raises(DesignError, match="the config's pursuer is SC_Human"):
        verify_design(copy, ONSITE, *NAMES)


def test_an_epoch_assigned_again_in_the_gmat_form_is_seen(template: Path) -> None:
    text = template.read_text(encoding="utf-8")
    template.write_text(text + "GMAT SC_Human.Epoch = '09 Nov 2026 02:00:00.000';\n",
                        encoding="utf-8")
    with pytest.raises(DesignError, match="must set Epoch once"):
        _set(template)


def _forged(copy: Path, record: dict, old: str, new: str, **changes: object) -> None:
    from orbitwin.models import file_sha256

    text = copy.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    copy.write_text(text.replace(old, new), encoding="utf-8")
    record_path(copy).write_text(json.dumps({**record, "script_sha256": file_sha256(copy),
                                             **changes}), encoding="utf-8")


def test_a_record_pointing_at_another_line_is_refused(template: Path) -> None:
    """The one changed line is the target's epoch, and the record says so."""
    copy, record = _set(template)
    original = template.read_text(encoding="utf-8").splitlines()
    alien = next(i for i, line in enumerate(original) if line.startswith("SC_Alien.Epoch"))
    copy.write_bytes(template.read_bytes())
    _forged(copy, record, "SC_Alien.Epoch = '09 Nov 2026 02:00:00.000';",
            "SC_Alien.Epoch = '09 Nov 2026 01:00:00.000';", line=alien + 1)
    with pytest.raises(DesignError, match="is not SC_Human's epoch"):
        verify_design(copy, ONSITE, *NAMES)


def test_a_change_beside_the_epoch_on_its_line_is_refused(template: Path) -> None:
    copy, record = _set(template)
    _forged(copy, record, "SC_Human.Epoch = '09 Nov 2026 03:00:00.000';",
            "SC_Human.Epoch = '09 Nov 2026 03:00:00.000'; % edited")
    with pytest.raises(DesignError, match="changes more than SC_Human's epoch"):
        verify_design(copy, ONSITE, *NAMES)


def test_a_record_of_another_format_is_refused(template: Path) -> None:
    copy, record = _set(template)
    record_path(copy).write_text(json.dumps({**record, "format": "other/1"}),
                                 encoding="utf-8")
    with pytest.raises(DesignError, match="cannot be read"):
        verify_design(copy, ONSITE, *NAMES)


def test_verify_policy_holds_its_original_to_the_record(
    template: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    copy, _ = _set(template)
    copy.write_text(copy.read_text(encoding="utf-8") + "% edited\n", encoding="utf-8")
    assert main(["verify-policy", str(copy), str(copy), "--config", str(ONSITE_CONFIG),
                 "--rules", "ranking-onsite"]) == 2
    assert "not the copy its record lists" in capsys.readouterr().err


def test_the_onsite_rules_refuse_a_pursuer_that_starts_first(tmp_path: Path) -> None:
    """Record or not: §4 入軌時刻 puts SC_Human at or after SC_Alien."""
    from test_head_start_burns import _config, _load, _script

    from orbitwin.errors import PolicyError

    with pytest.raises(PolicyError, match=r"starts 5400\.000 s before SC_Alien"):
        _load(_script(tmp_path), _config(rules="ranking-onsite"))


def test_the_onsite_rules_fly_a_copy_whose_pursuer_starts_after(tmp_path: Path) -> None:
    from test_head_start_burns import _config, _load, _script

    config = _config(rules="ranking-onsite", head_start_burns=False)
    copy, _ = set_pursuer_epoch(_script(tmp_path, pursuer_first=False), ONSITE,
                                "SC_Human", "SC_Alien", after_target_s=7200.0)
    assert verify_design(copy, ONSITE, "SC_Human", "SC_Alien") is not None
    scenario, _, _ = _load(copy, config)
    assert scenario.alignment is not None
    assert scenario.alignment.early == "SC_Alien"
    assert scenario.alignment.offset_s == pytest.approx(7200.0, abs=1e-3)


def _certify(monkeypatch: pytest.MonkeyPatch, template: Path, output: Path,
             certify_candidate: object) -> int:
    from types import SimpleNamespace

    from orbitwin import cli

    monkeypatch.setattr(cli, "read_candidates", lambda _: [SimpleNamespace(candidate_id="C")])
    monkeypatch.setattr(cli, "certify_candidate", certify_candidate)
    monkeypatch.setattr(cli, "upload_summary", lambda *_: [])
    monkeypatch.setattr(cli, "precheck_commands", lambda *_: [])
    monkeypatch.setattr(cli, "binding_precheck", lambda *_: ("PRE-CHECK SKIPPED", {}))
    copy, _ = _set(template)
    return main(["certify", str(copy), "--config", str(ONSITE_CONFIG), "--rules",
                 "ranking-onsite", "--candidates", str(output / "candidates.json"),
                 "--candidate", "C", "--gmat-console", "GmatConsole", "--startup-file",
                 "startup.txt", "--output", str(output)])


def test_certify_records_the_design_beside_its_certificate(
    monkeypatch: pytest.MonkeyPatch, template: Path, tmp_path: Path,
) -> None:
    certificate = tmp_path / "final" / "submission" / "certification.json"

    def certify_candidate(*_: object, **__: object) -> dict[str, object]:
        certificate.parent.mkdir(parents=True)
        certificate.write_text("{}", encoding="utf-8")
        return {}

    assert _certify(monkeypatch, template, tmp_path / "final", certify_candidate) == 0
    design = json.loads(certificate.read_text(encoding="utf-8"))["design"]
    assert (design["after_target_s"], design["verified_under"]) == (3600.0, ONSITE.id)


def test_certify_stopped_for_the_gui_records_the_design_in_verification(
    monkeypatch: pytest.MonkeyPatch, template: Path, tmp_path: Path,
) -> None:
    from orbitwin.errors import GmatRunError
    from orbitwin.gmat_runner import GUI_STOP

    kept = tmp_path / "kept"
    verification = kept / "diagnostic" / "verification.json"
    verification.parent.mkdir(parents=True)
    verification.write_text("{}", encoding="utf-8")

    def certify_candidate(*_: object, **__: object) -> dict[str, object]:
        stop = GmatRunError(f"exact-run-01 refused for its plots. {GUI_STOP}")
        stop.kept_dir = str(kept)  # type: ignore[attr-defined]
        raise stop

    monkeypatch.setattr("orbitwin.cli.gui_precheck", lambda *_: [])
    assert _certify(monkeypatch, template, tmp_path / "final", certify_candidate) == 2
    design = json.loads(verification.read_text(encoding="utf-8"))["design"]
    assert design["after_target_s"] == 3600.0
