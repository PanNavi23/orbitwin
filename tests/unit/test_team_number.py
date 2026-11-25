"""policy.team_number replaces the TeamXX placeholder in the official report name.

The organizer requires it (docs/LESSONS_2026-09-05.md §7); nothing else in the
name, or anywhere else, may move.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import warnings
from pathlib import Path

import pytest
from conftest import quick_config, stamped

from orbitwin.config import OrbitWinConfig, load_config, validate_config
from orbitwin.errors import ConfigurationError, PolicyError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.script_renderer import (
    assert_final_script_policy,
    assert_team_number_for_submission,
    official_report_filename,
    official_report_files,
    render_candidate,
    team_number_warnings,
    write_rendered_candidate,
)
from orbitwin.search import search_one_burn, zero_burn_candidate

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
R1_HORIZON_S = 6711.667
PLACEHOLDER_LINE = "ReportFile1.Filename = 'ODC_2026_TeamXX.txt';"
NUMBERED_LINE = "ReportFile1.Filename = 'ODC_2026_Team54.txt';"
_NOT_PROFILES = {"editable-fields.yaml", "tolerances.yaml"}


def _ranking(team_number: int | None = 54) -> OrbitWinConfig:
    config = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    return dataclasses.replace(
        config,
        policy=dataclasses.replace(
            config.policy, horizon_s=R1_HORIZON_S, team_number=team_number
        ),
    )


def _r1(tmp_path: Path, old: str = "", new: str = "", name: str = "R1.script") -> Path:
    text = R1.read_text(encoding="utf-8")
    if old:
        assert old in text
        text = text.replace(old, new, 1)
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _render(original: Path, config: OrbitWinConfig, tmp_path: Path) -> Path:
    scenario, document = parse_gmat_script(original, config)
    out = tmp_path / "rendered"
    out.mkdir(exist_ok=True)
    script = out / "final.script"
    write_rendered_candidate(
        script, out / "final.diff", document, scenario, zero_burn_candidate(scenario, config),
        config,
    )
    return script


def _mutate(script: Path, old: str, new: str) -> Path:
    text = script.read_text(encoding="utf-8")
    assert old in text, old
    mutated = script.with_name("mutated.script")
    mutated.write_text(text.replace(old, new, 1), encoding="utf-8")
    return mutated


# --- config ---------------------------------------------------------------


def test_the_field_defaults_to_none() -> None:
    assert OrbitWinConfig().policy.team_number is None


@pytest.mark.parametrize("value", [0, -54, True, "54", 54.0])
def test_only_a_positive_integer_is_a_team_number(value: object) -> None:
    config = OrbitWinConfig()
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, team_number=value)  # type: ignore[arg-type]
    )
    with pytest.raises(ConfigurationError, match=r"policy\.team_number"):
        validate_config(config)


def test_a_leading_zero_in_yaml_is_refused(tmp_path: Path) -> None:
    # YAML 1.1 reads 054 as octal 44.
    path = tmp_path / "c.yaml"
    path.write_text("policy:\n  team_number: 054\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="decimal integer"):
        load_config(path)


@pytest.mark.parametrize("profile", ["ranking.yaml", "ranking-rehearsal.yaml"])
def test_the_ranking_profiles_carry_no_team(profile: str) -> None:
    """The day path takes it from --team-number; no profile names our team."""
    assert load_config(ROOT / "config" / profile).policy.team_number is None


def test_the_other_profiles_carry_none() -> None:
    profiles = [
        *(ROOT / "config").glob("*.yaml"),
        *(ROOT / "config" / "history" / "preliminary").glob("*.yaml"),
    ]
    for path in sorted(profiles):
        if path.name in {"ranking.yaml", "ranking-rehearsal.yaml"} | _NOT_PROFILES:
            continue
        assert load_config(path).policy.team_number is None, path.name


# --- the name -------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ODC_2026_TeamXX.txt", "ODC_2026_Team54.txt"),
        ("TeamX.txt", "Team54.txt"),
        ("TeamXXXX_report.txt", "Team54_report.txt"),
        ("ODC_2026_Teamxx.txt", "ODC_2026_Team54.txt"),
        ("ODC_2026_teamXX.txt", "ODC_2026_team54.txt"),
        ("ODC_2026_TEAMXX.txt", "ODC_2026_TEAM54.txt"),
        ("ODC_2026_Team_XX.txt", "ODC_2026_Team_54.txt"),
        ("ODC_2026_Team XX.txt", "ODC_2026_Team 54.txt"),
        ("ODC_2026_Team-XX.txt", "ODC_2026_Team-54.txt"),
        ("ODC_2026_TeamNN.txt", "ODC_2026_Team54.txt"),
        ("ODC_2026_Team_nn.txt", "ODC_2026_Team_54.txt"),
        ("ODC2026TeamXX.txt", "ODC2026Team54.txt"),
        ("TeamXX_TeamXX.txt", "Team54_Team54.txt"),
        ("ODC_2026_TeamXX_100%.txt", "ODC_2026_Team54_100%.txt"),
        ("ODC_2026_TeamXX_SteamXX.txt", "ODC_2026_Team54_SteamXX.txt"),
        # Right after a letter, "team" is part of a word: never numbered.
        ("MyTeamXX.txt", "MyTeamXX.txt"),
        ("SteamX.txt", "SteamX.txt"),
        ("SteamXX.txt", "SteamXX.txt"),
        ("Steam_NN.txt", "Steam_NN.txt"),
        ("UpstreamTeamxx.txt", "UpstreamTeamxx.txt"),
        # Not a placeholder: left, and certify refuses it under the ranking rules.
        ("ODC_2026_Team.txt", "ODC_2026_Team.txt"),
        ("ODC_2026_TeamXXL.txt", "ODC_2026_TeamXXL.txt"),
        ("ODC_2026_TeamXN.txt", "ODC_2026_TeamXN.txt"),
        ("ODC_2026_Team__XX.txt", "ODC_2026_Team__XX.txt"),
        ("ODC_2026_TeamName.txt", "ODC_2026_TeamName.txt"),
        ("ODC_2026_Team00.txt", "ODC_2026_Team00.txt"),
        ("ODC_2026_XX.txt", "ODC_2026_XX.txt"),
        ("ReportFile1.txt", "ReportFile1.txt"),
    ],
)
def test_only_the_placeholder_token_is_replaced(name: str, expected: str) -> None:
    assert official_report_filename(name, 54) == expected
    assert official_report_filename(name, None) == name


# --- the renderer ---------------------------------------------------------


def test_r1_renders_the_team_number_and_nothing_else(tmp_path: Path) -> None:
    original = _r1(tmp_path)
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    rendered = _render(original, _ranking(), tmp_path)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    old = original.read_text(encoding="utf-8").splitlines(keepends=True)
    new = rendered.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(old) == len(new)
    changed = [(a, b) for a, b in zip(old, new, strict=True) if a != b]
    assert (PLACEHOLDER_LINE + "\n", NUMBERED_LINE + "\n") in changed
    assert all(a.startswith("Propagate") for a, _ in changed if a != PLACEHOLDER_LINE + "\n")
    scenario, _ = parse_gmat_script(rendered, _ranking())
    assert [spec.filename for spec in scenario.report_files] == ["ODC_2026_Team54.txt"]
    assert "ReportFile1.Filename: 'ODC_2026_TeamXX.txt' -> 'ODC_2026_Team54.txt'" in (
        rendered.with_name("final.diff").read_text(encoding="utf-8")
    )


def test_without_a_number_the_name_is_left(tmp_path: Path) -> None:
    rendered = _render(_r1(tmp_path), _ranking(None), tmp_path)
    assert PLACEHOLDER_LINE in rendered.read_text(encoding="utf-8")


def test_the_edit_is_line_preserving(tmp_path: Path) -> None:
    line = "GMAT ReportFile1.Filename  =  'ODC_2026_TeamXX.txt' ; % TeamXX stays here"
    original = _r1(tmp_path, PLACEHOLDER_LINE, line)
    original.write_bytes(original.read_bytes().replace(b"\n", b"\r\n"))
    rendered = _render(original, _ranking(), tmp_path)
    raw = rendered.read_bytes()
    assert (
        b"GMAT ReportFile1.Filename  =  'ODC_2026_Team54.txt' ; % TeamXX stays here\r\n" in raw
    )
    assert raw.count(b"\r\n") == raw.count(b"\n")


def test_a_slotted_template_renders_the_number(tmp_path: Path, workshop_script: Path) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    old = "ReportFile1.Filename = 'ReportFile1.txt';"
    assert old in text
    original = tmp_path / "workshop.script"
    original.write_text(text.replace(old, PLACEHOLDER_LINE), encoding="utf-8")
    config = quick_config()
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, team_number=54)
    )
    scenario, document = parse_gmat_script(original, config)
    candidate = search_one_burn(scenario, config)[0]
    result = render_candidate(document, scenario, candidate, config)
    line_index = document.lines.index(PLACEHOLDER_LINE + "\n")
    assert line_index in result.changed_lines
    assert result.script_bytes.decode("utf-8").splitlines()[line_index] == NUMBERED_LINE


def test_both_official_reports_are_numbered(tmp_path: Path) -> None:
    second = (
        "Create ReportFile ReportFile2;\n"
        "ReportFile2.Filename = 'second_TeamXXX.txt';\n"
        "ReportFile2.Add = {SC_Human.ElapsedSecs};\n"
    )
    original = _r1(tmp_path, "BeginMissionSequence;", second + "BeginMissionSequence;")
    rendered = _render(original, _ranking(), tmp_path)
    scenario, _ = parse_gmat_script(rendered, _ranking())
    assert [spec.filename for spec in scenario.report_files] == [
        "ODC_2026_Team54.txt",
        "second_Team54.txt",
    ]
    # Only one of the two renamed is not the contract.
    half = _mutate(rendered, "'second_Team54.txt'", "'second_TeamXXX.txt'")
    with pytest.raises(PolicyError, match="ReportFile2"):
        assert_final_script_policy(original, half, _ranking())


def test_a_placeholder_outside_a_report_filename_is_left(tmp_path: Path) -> None:
    extra = (
        "% ODC_2026_TeamXX.txt is the upload\n"
        "Create EphemerisFile EF;\n"
        "EF.Spacecraft = SC_Alien;\n"
        "EF.Filename = 'alien_TeamXX.oem';\n"
    )
    original = _r1(tmp_path, "BeginMissionSequence;", extra + "BeginMissionSequence;")
    config = _ranking()
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, extra_resource_types=())
    )
    rendered = _render(original, config, tmp_path)
    text = rendered.read_text(encoding="utf-8")
    assert "% ODC_2026_TeamXX.txt is the upload\n" in text
    assert "EF.Filename = 'alien_TeamXX.oem';" in text
    assert NUMBERED_LINE in text
    for old, new in [
        ("% ODC_2026_TeamXX.txt is the upload", "% ODC_2026_Team54.txt is the upload"),
        ("EF.Filename = 'alien_TeamXX.oem';", "EF.Filename = 'alien_Team54.oem';"),
    ]:
        with pytest.raises(PolicyError):
            assert_final_script_policy(original, _mutate(rendered, old, new), config)


def test_a_placeholder_in_the_default_name_with_no_line_is_refused(tmp_path: Path) -> None:
    # No Filename line: GMAT names it TeamXX.txt, and there is no line to patch.
    original = _r1(
        tmp_path,
        "Create ReportFile ReportFile1;\n" + PLACEHOLDER_LINE + "\n",
        "Create ReportFile ReportFile1;\n",
    )
    text = original.read_text(encoding="utf-8").replace("ReportFile1", "TeamXX")
    original.write_text(text, encoding="utf-8")
    scenario, document = parse_gmat_script(original, _ranking())
    assert scenario.report_files[0].filename == "TeamXX.txt"
    with pytest.raises(PolicyError, match="no Filename line"):
        render_candidate(document, scenario, zero_burn_candidate(scenario, _ranking()), _ranking())


# --- the guard ------------------------------------------------------------


def test_the_guard_accepts_exactly_the_numbered_name(tmp_path: Path) -> None:
    original = _r1(tmp_path)
    rendered = _render(original, _ranking(), tmp_path)
    assert_final_script_policy(original, rendered, _ranking())


@pytest.mark.parametrize(
    "new",
    [
        "'ODC_2027_Team54.txt'",  # prefix
        "'odc_2026_Team54.txt'",
        "'ODC_2026_Team54.csv'",  # extension
        "'ODC_2026_Team54.txt.txt'",
        "'ODC_2026_Team054.txt'",  # leading zero
        "'ODC_2026_Team-54.txt'",  # negative
        "'ODC_2026_Team55.txt'",  # another team
        "'ODC_2026_Team54 .txt'",
        "'ODC_2026_TeamXX.txt'",  # left unnumbered although configured
    ],
)
def test_the_guard_refuses_any_other_name(tmp_path: Path, new: str) -> None:
    original = _r1(tmp_path)
    rendered = _render(original, _ranking(), tmp_path)
    mutated = _mutate(rendered, "'ODC_2026_Team54.txt'", new)
    with pytest.raises(PolicyError, match=r"ReportFile1\.Filename"):
        assert_final_script_policy(original, mutated, _ranking())


def test_the_guard_refuses_a_number_nobody_configured(tmp_path: Path) -> None:
    original = _r1(tmp_path)
    rendered = _render(original, _ranking(), tmp_path)
    with pytest.raises(PolicyError, match=r"ReportFile1\.Filename"):
        assert_final_script_policy(original, rendered, _ranking(None))


@pytest.mark.parametrize(
    "new",
    [
        "'ODC_2026_Teamxx_SteamXX.txt'",  # left unnumbered
        "'ODC_2026_TEAM54_SteamXX.txt'",  # numbered, but not as the template wrote it
        "'ODC_2026_team54_SteamXX.txt'",
        "'ODC_2026_Team54_Steam54.txt'",  # a word that only looks like a placeholder
    ],
)
def test_the_guard_refuses_numbering_a_mixed_case_name(tmp_path: Path, new: str) -> None:
    original = _r1(tmp_path, "ODC_2026_TeamXX", "ODC_2026_Teamxx_SteamXX")
    rendered = _render(original, _ranking(), tmp_path)
    assert "'ODC_2026_Team54_SteamXX.txt'" in rendered.read_text(encoding="utf-8")
    assert_final_script_policy(original, rendered, _ranking())
    mutated = _mutate(rendered, "'ODC_2026_Team54_SteamXX.txt'", new)
    with pytest.raises(PolicyError, match=r"ReportFile1\.Filename"):
        assert_final_script_policy(original, mutated, _ranking())


@pytest.mark.parametrize(
    "placeholder, numbered",
    [
        ("Teamxx", "Team54"),
        ("TEAMXX", "TEAM54"),
        ("Team_XX", "Team_54"),
        ("Team XX", "Team 54"),
        ("TeamNN", "Team54"),
    ],
)
def test_a_placeholder_variant_is_numbered_in_the_final_bytes(
    tmp_path: Path, placeholder: str, numbered: str
) -> None:
    original = _r1(tmp_path, "TeamXX", placeholder)
    rendered = _render(original, _ranking(), tmp_path)
    old = original.read_bytes().splitlines(keepends=True)
    new = rendered.read_bytes().splitlines(keepends=True)
    changed = [(a, b) for a, b in zip(old, new, strict=True) if a != b]
    line = "ReportFile1.Filename = 'ODC_2026_{}.txt';\n"
    assert (line.format(placeholder).encode(), line.format(numbered).encode()) in changed
    assert all(a.startswith(b"Propagate") for a, _ in changed if b"Filename" not in a)
    assert_final_script_policy(original, rendered, _ranking())
    assert_team_number_for_submission(parse_gmat_script(original, _ranking())[0], _ranking())
    # Left as the template has it is refused.
    unnumbered = _mutate(rendered, f"'ODC_2026_{numbered}.txt'", f"'ODC_2026_{placeholder}.txt'")
    with pytest.raises(PolicyError, match=r"ReportFile1\.Filename"):
        assert_final_script_policy(original, unnumbered, _ranking())


@pytest.mark.parametrize(
    "old, new",
    [
        ("ReportFile1.Precision = 16;", "ReportFile1.Precision = 17;"),
        ("ReportFile1.WriteHeaders = true;", "ReportFile1.WriteHeaders = false;"),
        ("{SC_Alien.ElapsedSecs, ", "{"),
    ],
)
def test_the_rest_of_the_contract_still_holds(tmp_path: Path, old: str, new: str) -> None:
    original = _r1(tmp_path)
    rendered = _render(original, _ranking(), tmp_path)
    with pytest.raises(PolicyError):
        assert_final_script_policy(original, _mutate(rendered, old, new), _ranking())


def test_the_guard_refuses_an_added_report_carrying_the_name(tmp_path: Path) -> None:
    original = _r1(tmp_path)
    rendered = _render(original, _ranking(), tmp_path)
    added = (
        "Create ReportFile Extra;\nExtra.Filename = 'ODC_2026_Team54.txt';\n"
        "BeginMissionSequence;"
    )
    with pytest.raises(PolicyError):
        assert_final_script_policy(
            original, _mutate(rendered, "BeginMissionSequence;", added), _ranking()
        )


# --- certify --------------------------------------------------------------


@pytest.mark.parametrize("filename", ["ODC_2026_TeamXX.txt", "ReportFile1.txt"])
def test_ranking_certify_refuses_without_a_number(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path, filename: str
) -> None:
    # Unsetting the key was a one-line way past the name check.
    from orbitwin.verify import certify_candidate

    config = _ranking(None)
    original = _r1(tmp_path, "ODC_2026_TeamXX.txt", filename)
    scenario, _ = parse_gmat_script(original, config)
    candidate = zero_burn_candidate(scenario, config)
    with pytest.raises(PolicyError, match=r"policy\.team_number is unset"):
        certify_candidate(
            original, candidate, config, fake_gmat, fake_startup, tmp_path / "out", runs=1
        )
    assert not (tmp_path / "out").exists()


_WRONG_NUMBER = {"ODC_2026_Team55.txt", "ODC_2026_Team540.txt", "ODC_2026_Team054.txt",
                 "ODC_2026_Team55_Team54.txt"}


@pytest.mark.parametrize(
    "filename",
    [
        "ReportFile1.txt",
        "ODC_2026_Team55.txt",  # another team's
        "ODC_2026_Team540.txt",
        "ODC_2026_Team054.txt",
        "ODC_2026_TeamXXL.txt",
        "ODC_2026_TeamXN.txt",
        "Run54.txt",
        "ODC_2026_Team54X.txt",  # half filled in
        "ODC_2026_Team54XX.txt",
        "ODC_2026_Team54_Team54X.txt",
        "ODC_2026_Steam54.txt",
        "ODC_2026_MyTeam54.txt",
        "ODC_2026_Team55_Team54.txt",
        "ODC_2026_TeamXX_TeamXXL.txt",  # one placeholder taken, one left
    ],
)
def test_ranking_refuses_a_name_that_cannot_carry_the_number(
    tmp_path: Path, filename: str
) -> None:
    original = _r1(tmp_path, "ODC_2026_TeamXX.txt", filename)
    scenario, _ = parse_gmat_script(original, _ranking())
    message = (
        rf"must carry Team54.*ReportFile1\.Filename '{re.escape(filename)}'.*"
        r"may not be edited.*ask the organizer"
    )
    with pytest.raises(PolicyError, match=message) as refused:
        assert_team_number_for_submission(scenario, _ranking())
    text = str(refused.value)
    assert ("wrong team number" in text) == (filename in _WRONG_NUMBER)
    if filename.endswith("Team54X.txt"):
        assert "has Team54X, which is neither a placeholder nor Team54" in text
    # The grammar as the code reads it: separators, and X's or N's but not mixed.
    assert "space, '_' or '-'" in text and "not mixed" in text
    (warning,) = team_number_warnings(scenario, _ranking())
    assert warning.startswith("certify will refuse") and filename in warning


def test_ranking_certify_refuses_before_writing_anything(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.verify import certify_candidate

    original = _r1(tmp_path, "ODC_2026_TeamXX.txt", "ReportFile1.txt")
    scenario, _ = parse_gmat_script(original, _ranking())
    with pytest.raises(PolicyError, match=r"ReportFile1\.Filename 'ReportFile1\.txt'"):
        certify_candidate(
            original, zero_burn_candidate(scenario, _ranking()), _ranking(), fake_gmat,
            fake_startup, tmp_path / "out", runs=1,
        )
    assert not (tmp_path / "out").exists()


def test_a_name_already_carrying_the_number_is_accepted(tmp_path: Path) -> None:
    original = _r1(tmp_path, "ODC_2026_TeamXX.txt", "ODC_2026_Team_54.txt")
    scenario, _ = parse_gmat_script(original, _ranking())
    assert_team_number_for_submission(scenario, _ranking())
    assert team_number_warnings(scenario, _ranking()) == ()


def _second_report(tmp_path: Path, first: str, second: str) -> Path:
    report = (
        "Create ReportFile ReportFile2;\n"
        f"ReportFile2.Filename = '{second}';\n"
        "ReportFile2.Add = {SC_Human.ElapsedSecs};\n"
    )
    original = _r1(tmp_path, "ODC_2026_TeamXX.txt", first)
    text = original.read_text(encoding="utf-8").replace(
        "BeginMissionSequence;", report + "BeginMissionSequence;", 1
    )
    original.write_text(text, encoding="utf-8")
    return original


def test_a_report_without_a_placeholder_is_left_and_named(tmp_path: Path) -> None:
    original = _second_report(tmp_path, "ODC_2026_TeamXX.txt", "aux_dv.txt")
    scenario, _ = parse_gmat_script(original, _ranking())
    assert_team_number_for_submission(scenario, _ranking())
    assert [spec.filename for spec in official_report_files(scenario, _ranking())] == [
        "ODC_2026_Team54.txt", "aux_dv.txt",
    ]
    (note,) = team_number_warnings(scenario, _ranking())
    assert "ReportFile2.Filename 'aux_dv.txt'" in note and "left as the organizer" in note
    rendered = _render(original, _ranking(), tmp_path)
    assert "ReportFile2.Filename = 'aux_dv.txt';" in rendered.read_text(encoding="utf-8")
    assert_final_script_policy(original, rendered, _ranking())


def test_every_report_that_cannot_carry_the_number_is_named(tmp_path: Path) -> None:
    original = _second_report(tmp_path, "first.txt", "ODC_2026_Team55.txt")
    scenario, _ = parse_gmat_script(original, _ranking())
    with pytest.raises(PolicyError) as refused:
        assert_team_number_for_submission(scenario, _ranking())
    text = str(refused.value)
    assert "ReportFile1.Filename 'first.txt' has no team-number placeholder" in text
    assert "ReportFile2.Filename 'ODC_2026_Team55.txt' carries the wrong team number" in text


def test_a_placeholder_that_cannot_take_the_number_refuses_even_beside_one_that_does(
    tmp_path: Path,
) -> None:
    original = _second_report(tmp_path, "ODC_2026_TeamXX.txt", "aux_TeamXX_TeamXXL.txt")
    scenario, _ = parse_gmat_script(original, _ranking())
    with pytest.raises(PolicyError, match=r"ReportFile2\.Filename 'aux_TeamXX_TeamXXL\.txt' "
                       r"\(numbered 'aux_Team54_TeamXXL\.txt'\) has TeamXXL"):
        assert_team_number_for_submission(scenario, _ranking())


def test_numbering_may_not_merge_two_reports_that_differ_only_in_case(tmp_path: Path) -> None:
    # Windows paths ignore case: Team54 and TEAM54 are one file there.
    original = _second_report(tmp_path, "ODC_2026_TeamXX.txt", "ODC_2026_TEAMNN.txt")
    scenario, document = parse_gmat_script(original, _ranking())
    with pytest.raises(PolicyError, match="two of them one file"):
        render_candidate(document, scenario, zero_burn_candidate(scenario, _ranking()), _ranking())


def test_the_preliminary_rules_only_warn_about_a_name_with_no_number(
    tmp_path: Path, workshop_script: Path
) -> None:
    config = _numbered(quick_config())
    validate_config(config)
    assert config.policy.rules == "preliminary"
    scenario, _ = parse_gmat_script(workshop_script, config)
    assert [spec.filename for spec in scenario.report_files] == ["ReportFile1.txt"]
    assert_team_number_for_submission(scenario, config)
    (warning,) = team_number_warnings(scenario, config)
    assert "no Team placeholder" in warning
    unnumbered = dataclasses.replace(config, policy=dataclasses.replace(config.policy,
                                                                        team_number=None))
    assert team_number_warnings(scenario, unnumbered) == ()


def test_preliminary_certify_numbers_a_variant_too(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # One grammar under both rulebooks, once a number is set; none is under the
    # preliminary profiles, which leave every name alone.
    from orbitwin.verify import certify_candidate

    config = _numbered(quick_config())
    original = _workshop(tmp_path, workshop_script, "ODC2026Team_xx.txt")
    scenario, _ = parse_gmat_script(original, config)
    candidate = stamped(search_one_burn(scenario, config)[0], original, config)
    evidence = certify_candidate(
        original, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1
    )
    submission = tmp_path / "c" / "submission"
    assert evidence["official_report_files"] == ["ODC2026Team_54.txt"]
    assert (submission / "ODC2026Team_54.txt").is_file()
    assert not (submission / "ODC2026Team_xx.txt").exists()
    script = (submission / evidence["submission_script_name"]).read_bytes()
    assert b"ReportFile1.Filename = 'ODC2026Team_54.txt';" in script
    assert evidence["report_name_warnings"] == []


CROSSING = ROOT / "tests" / "fixtures" / "async" / "aligned_crossing.script"


def _crossing_ranking() -> OrbitWinConfig:
    from orbitwin.config import PolicyConfig, ScoreConfig

    return OrbitWinConfig(
        policy=dataclasses.replace(
            PolicyConfig(), rules="ranking", horizon_s=20000.0, total_dv_limit_km_s=3.0,
            total_dv_safe_limit_km_s=2.9, dv_limit_km_s=0.25, dv_safe_limit_km_s=0.249,
            mission_time_origin="earliest", closest_approach_window="after_alignment",
            team_number=54,
        ),
        pursuer_name="SC_Human",
        target_name="SC_Alien",
        score=ScoreConfig(kt=0.003, ct=6000.0, kv=0.0012, cv=2000.0, delta_v_unit="m/s",
                          cr_km=100.0),
    )


@pytest.mark.parametrize(
    "template, submitted",
    [("ODC_2026_Team_xx.txt", "ODC_2026_Team_54.txt"), ("ODC_2026_TEAMNN.txt", "ODC_2026_TEAM54.txt")],
)
def test_ranking_certify_submits_a_variant_under_the_number(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path, template: str, submitted: str
) -> None:
    from orbitwin.state_probe import load_scenario
    from orbitwin.verify import certify_candidate

    config = _crossing_ranking()
    validate_config(config)
    text = CROSSING.read_text(encoding="utf-8")
    assert "OfficialReport.Filename = 'OfficialReport.txt';" in text
    script = tmp_path / "crossing.script"
    script.write_text(
        text.replace("'OfficialReport.txt'", f"'{template}'") + "% FAKE_GMAT_MISS_KM 900\n",
        encoding="utf-8",
    )
    scenario, _, _ = load_scenario(script, config, gmat_console=fake_gmat,
                                   startup_file=fake_startup)
    miss = stamped(zero_burn_candidate(scenario, config), script, config)
    evidence = certify_candidate(script, miss, config, fake_gmat, fake_startup, tmp_path / "c")
    submission = tmp_path / "c" / "submission"
    assert evidence["official_report_files"] == [submitted]
    assert (submission / submitted).is_file()
    assert not (submission / template).exists()
    final = (submission / evidence["submission_script_name"]).read_text(encoding="utf-8")
    assert f"OfficialReport.Filename = '{submitted}';" in final
    assert evidence["report_name_warnings"] == []


def test_rank_blocks_every_candidate_when_certify_would_refuse_the_names(tmp_path: Path) -> None:
    from orbitwin.rank import submission_blockers

    original = _r1(tmp_path, "ODC_2026_TeamXX.txt", "ReportFile1.txt")
    scenario, _ = parse_gmat_script(original, _ranking())
    candidate = zero_burn_candidate(scenario, _ranking())
    blockers = submission_blockers(candidate, scenario, _ranking())
    assert any(b.startswith("certify refuses the report names") for b in blockers)
    numbered, _ = parse_gmat_script(_r1(tmp_path, name="R1-ok.script"), _ranking())
    assert not any("report names" in b for b in submission_blockers(candidate, numbered, _ranking()))


def test_inspect_says_before_the_search_that_certify_would_refuse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from orbitwin.cli import main

    raw = (ROOT / "config" / "ranking-rehearsal.yaml").read_text(encoding="utf-8")
    config = tmp_path / "c.yaml"
    config.write_text(raw.replace("horizon_s: null", f"horizon_s: {R1_HORIZON_S}")
                      .replace("  rules: ranking\n", "  rules: ranking\n  team_number: 54\n"),
                      encoding="utf-8")
    original = _r1(tmp_path, "ODC_2026_TeamXX.txt", "ReportFile1.txt")
    assert main(["inspect", str(original), "--config", str(config)]) == 0
    out, err = capsys.readouterr()
    assert "WARNING: certify will refuse" in err
    names = json.loads(out)["report_names"]
    assert names["submitted"] is None and "ReportFile1.txt" in names["refusal"]
    assert main(["inspect", str(_r1(tmp_path, name="ok.script")), "--config", str(config)]) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["report_names"] == {
        "submitted": ["ODC_2026_Team54.txt"], "refusal": None, "notes": [],
    }
    assert "certify will refuse" not in err


def _workshop(tmp_path: Path, workshop_script: Path, filename: str) -> Path:
    text = workshop_script.read_text(encoding="utf-8")
    old = "ReportFile1.Filename = 'ReportFile1.txt';"
    assert old in text
    path = tmp_path / "workshop.script"
    path.write_text(text.replace(old, f"ReportFile1.Filename = '{filename}';"), encoding="utf-8")
    return path


def _numbered(config: OrbitWinConfig) -> OrbitWinConfig:
    return dataclasses.replace(config, policy=dataclasses.replace(config.policy, team_number=54))


def test_certify_submits_the_numbered_report(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.verify import certify_candidate

    config = _numbered(quick_config())
    original = _workshop(tmp_path, workshop_script, "ODC_2026_TeamXX.txt")
    original_bytes = original.read_bytes()
    scenario, _ = parse_gmat_script(original, config)
    candidate = stamped(search_one_burn(scenario, config)[0], original, config)
    evidence = certify_candidate(
        original, candidate, config, fake_gmat, fake_startup, tmp_path / "certify", runs=2
    )
    submission = tmp_path / "certify" / "submission"
    assert (submission / "ODC_2026_Team54.txt").is_file()
    assert not (submission / "ODC_2026_TeamXX.txt").exists()
    assert (submission / "official_original.script").read_bytes() == original_bytes
    assert original.read_bytes() == original_bytes
    assert "ODC_2026_Team54.txt" in evidence["submission_files"]
    assert evidence["official_report_files"] == ["ODC_2026_Team54.txt"]
    for run in evidence["exact_runs"]:
        assert set(run["report_sha256"]) == {"ODC_2026_Team54.txt"}
    certification = json.loads((submission / "certification.json").read_text(encoding="utf-8"))
    assert certification["official_report_files"] == ["ODC_2026_Team54.txt"]
    assert "'ODC_2026_Team54.txt'" in (submission / evidence["submission_script_name"]).read_text(
        encoding="utf-8"
    )


def test_certify_warns_when_there_is_no_placeholder(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.verify import certify_candidate

    config = _numbered(quick_config())
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = stamped(search_one_burn(scenario, config)[0], workshop_script, config)
    with pytest.warns(RuntimeWarning, match="no Team placeholder"):
        evidence = certify_candidate(
            workshop_script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1
        )
    assert evidence["official_report_files"] == ["ReportFile1.txt"]
    assert any("no Team placeholder" in item for item in evidence["report_name_warnings"])


def test_solve_keeps_working_and_says_why_certify_will_refuse(
    tmp_path: Path, workshop_script: Path
) -> None:
    from orbitwin.pipeline import solve_scenario

    config = quick_config()
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, team_number=None)
    )
    original = _workshop(tmp_path, workshop_script, "ODC_2026_TeamXX.txt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = solve_scenario(original, config, tmp_path / "solve")
    # Preliminary rules: nothing to say, the name is the template's.
    assert not any("team_number" in item for item in summary["warnings"])
    numbered = solve_scenario(
        _workshop(tmp_path, workshop_script, "ReportFile1.txt"),
        _numbered(config),
        tmp_path / "solve-2",
    )
    assert any("no Team placeholder" in item for item in numbered["warnings"])


def test_ranking_solve_is_told_certify_will_refuse(tmp_path: Path) -> None:
    scenario, _ = parse_gmat_script(_r1(tmp_path), _ranking(None))
    assert any("policy.team_number" in item for item in team_number_warnings(scenario, _ranking(None)))
    assert team_number_warnings(scenario, _ranking()) == ()


@pytest.mark.parametrize("other", ["ODC_2026_TeamXXX.txt", "ODC_2026_Team54.txt"])
def test_numbering_may_not_merge_two_reports(tmp_path: Path, other: str) -> None:
    second = (
        "Create ReportFile ReportFile2;\n"
        f"ReportFile2.Filename = '{other}';\n"
        "ReportFile2.Add = {SC_Human.ElapsedSecs};\n"
    )
    original = _r1(tmp_path, "BeginMissionSequence;", second + "BeginMissionSequence;")
    scenario, document = parse_gmat_script(original, _ranking())
    with pytest.raises(PolicyError, match="two of them one file"):
        render_candidate(document, scenario, zero_burn_candidate(scenario, _ranking()), _ranking())
    # The template itself is fine, and so is it unnumbered.
    assert_final_script_policy(original, original, _ranking(None))
    with pytest.raises(PolicyError, match="two of them one file"):
        assert_final_script_policy(original, original, _ranking())


def test_verify_policy_refuses_an_unnumbered_ranking_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from orbitwin.cli import main

    raw = (ROOT / "config" / "ranking-rehearsal.yaml").read_text(encoding="utf-8")
    raw = raw.replace("  team_number: 54\n", "").replace("horizon_s: null", "horizon_s: 6711.667")
    config = tmp_path / "c.yaml"
    config.write_text(raw, encoding="utf-8")
    original = _r1(tmp_path)
    assert main(["verify-policy", str(original), str(original), "--config", str(config)]) == 2
    assert "policy.team_number" in capsys.readouterr().err
    numbered = _render(original, _ranking(), tmp_path)
    assert main(["verify-policy", str(original), str(numbered), "--config", str(config)]) == 2
    config.write_text(raw.replace("  rules: ranking\n", "  rules: ranking\n  team_number: 54\n"))
    assert main(["verify-policy", str(original), str(numbered), "--config", str(config)]) == 0


def test_verify_policy_refuses_a_name_without_a_placeholder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from orbitwin.cli import main

    raw = (ROOT / "config" / "ranking-rehearsal.yaml").read_text(encoding="utf-8")
    config = tmp_path / "c.yaml"
    config.write_text(raw.replace("horizon_s: null", "horizon_s: 6711.667")
                      .replace("  rules: ranking\n", "  rules: ranking\n  team_number: 54\n"),
                      encoding="utf-8")
    original = _r1(tmp_path, "ODC_2026_TeamXX.txt", "ODC_2026_Team_00.txt")
    assert main(["verify-policy", str(original), str(original), "--config", str(config)]) == 2
    assert "ask the organizer" in capsys.readouterr().err


# --- the official report (review of 2026-11-05) ---------------------------


@pytest.mark.parametrize("filename", ["ReportFile1.txt", "ODC_2026_XX.txt"])
def test_an_official_name_with_no_placeholder_is_refused_with_what_to_do(
    tmp_path: Path, filename: str
) -> None:
    # GMAT's default name, as in the organizer's 2026-11-01 video.
    original = _r1(tmp_path, "ODC_2026_TeamXX.txt", filename)
    scenario, _ = parse_gmat_script(original, _ranking())
    with pytest.raises(PolicyError) as refused:
        assert_team_number_for_submission(scenario, _ranking())
    text = str(refused.value)
    assert "the official script names no team placeholder: stop and ask the organizer how " \
        "to name the report" in text
    assert f"--report-filename-as-written '{filename}'" in text


def _as_written(name: str | None) -> OrbitWinConfig:
    config = _ranking()
    return dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, report_filename_as_written=name)
    )


def test_the_organizer_s_answer_keeps_the_exact_official_name(tmp_path: Path) -> None:
    original = _second_report(tmp_path, "ReportFile1.txt", "aux_dv.txt")
    config = _as_written("ReportFile1.txt")
    validate_config(config)
    scenario, _ = parse_gmat_script(original, config)
    assert_team_number_for_submission(scenario, config)
    kept, aux = team_number_warnings(scenario, config)
    assert "kept as written, by policy.report_filename_as_written" in kept
    assert "aux_dv.txt" in aux
    rendered = _render(original, config, tmp_path)
    assert "ReportFile1.Filename = 'ReportFile1.txt';" in rendered.read_text(encoding="utf-8")
    assert_final_script_policy(original, rendered, config)


@pytest.mark.parametrize(
    "template, as_written, reason",
    [
        ("ReportFile1.txt", "reportfile1.txt", "not the official report's"),  # not exact
        ("ReportFile1.txt", "ODC_2026_Team54.txt", "not the official report's"),
        ("ODC_2026_TeamXX.txt", "ODC_2026_TeamXX.txt", "has a placeholder, which takes"),
        ("ODC_2026_Team55.txt", "ODC_2026_Team55.txt", "wrong team number"),
    ],
)
def test_the_override_is_refused_unless_it_is_the_exact_unnumbered_name(
    tmp_path: Path, template: str, as_written: str, reason: str
) -> None:
    original = _r1(tmp_path, "ODC_2026_TeamXX.txt", template)
    config = _as_written(as_written)
    scenario, _ = parse_gmat_script(original, config)
    with pytest.raises(PolicyError, match=reason):
        assert_team_number_for_submission(scenario, config)


@pytest.mark.parametrize("value", ["", 54, True])
def test_the_override_is_a_name(value: object) -> None:
    config = _as_written(value)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="report_filename_as_written"):
        validate_config(config)


def test_reportfile1_is_the_official_report_wherever_it_is_declared(tmp_path: Path) -> None:
    original = _r1(tmp_path)
    text = original.read_text(encoding="utf-8").replace(
        "Create ReportFile ReportFile1;",
        "Create ReportFile AuxFirst;\nAuxFirst.Filename = 'aux_first.txt';\n"
        "AuxFirst.Add = {SC_Human.ElapsedSecs};\nCreate ReportFile ReportFile1;", 1,
    )
    original.write_text(text, encoding="utf-8")
    scenario, _ = parse_gmat_script(original, _ranking())
    assert [spec.name for spec in scenario.report_files] == ["AuxFirst", "ReportFile1"]
    assert_team_number_for_submission(scenario, _ranking())
    (note,) = team_number_warnings(scenario, _ranking())
    assert "AuxFirst.Filename 'aux_first.txt'" in note
    renamed = _r1(tmp_path, "ODC_2026_TeamXX.txt", "aux_TeamXX.txt", name="aux.script")
    text = renamed.read_text(encoding="utf-8").replace(
        "Create ReportFile ReportFile1;",
        "Create ReportFile AuxFirst;\nAuxFirst.Filename = 'ODC_2026_Report.txt';\n"
        "AuxFirst.Add = {SC_Human.ElapsedSecs};\nCreate ReportFile ReportFile1;", 1,
    )
    renamed.write_text(text, encoding="utf-8")
    scenario, _ = parse_gmat_script(renamed, _ranking())
    assert_team_number_for_submission(scenario, _ranking())  # ReportFile1 takes the number


def test_a_wrong_number_in_any_report_refuses(tmp_path: Path) -> None:
    original = _second_report(tmp_path, "ODC_2026_TeamXX.txt", "debug_Team55.txt")
    scenario, _ = parse_gmat_script(original, _ranking())
    with pytest.raises(PolicyError, match=r"ReportFile2\.Filename 'debug_Team55\.txt' carries "
                       r"the wrong team number"):
        assert_team_number_for_submission(scenario, _ranking())


def test_a_placeholder_only_in_another_report_refuses(tmp_path: Path) -> None:
    original = _second_report(tmp_path, "ODC_2026_Report.txt", "aux_TeamXX.txt")
    scenario, _ = parse_gmat_script(original, _ranking())
    with pytest.raises(PolicyError, match=r"ReportFile2\.Filename 'aux_TeamXX\.txt' \(numbered "
                       r"'aux_Team54\.txt'\) has a placeholder and the official report "
                       r"ReportFile1 has none: ask the organizer"):
        assert_team_number_for_submission(scenario, _ranking())


@pytest.mark.parametrize("second", ["ODC_2026_TEAMXX.txt", "ODC_2026_TeamXX.txt"])
def test_two_names_one_file_on_windows_are_refused_numbered_or_not(
    tmp_path: Path, second: str
) -> None:
    original = _second_report(tmp_path, "ODC_2026_TeamXX.txt", second)
    scenario, _ = parse_gmat_script(original, _ranking())
    with pytest.raises(PolicyError, match="two of them one file"):
        assert_team_number_for_submission(scenario, _ranking())
    with pytest.raises(PolicyError, match="two of them one file"):
        official_report_files(scenario, _ranking(None))


_AUX = (
    "Create ReportFile AuxReport;\nAuxReport.Filename = '{aux}';\nAuxReport.Precision = 16;\n"
    "AuxReport.Add = {{SC_Human.UTCModJulian, SC_Human.EarthMJ2000Eq.X}};\n"
    "AuxReport.WriteHeaders = true;\n"
)


def _crossing_with(tmp_path: Path, official: str, aux: str | None) -> Path:
    text = CROSSING.read_text(encoding="utf-8").replace("'OfficialReport.txt'", f"'{official}'")
    marker = "OfficialReport.AppendToExistingFile = false;\n"
    assert marker in text
    if aux is not None:
        text = text.replace(marker, marker + _AUX.format(aux=aux), 1)
    script = tmp_path / "crossing.script"
    script.write_text(text + "% FAKE_GMAT_MISS_KM 900\n", encoding="utf-8")
    return script


@pytest.mark.parametrize(
    "official, aux, reason",
    [
        ("ODC_2026_Report.txt", "aux_TeamXX.txt", "has a placeholder and the official report"),
        ("ODC_2026_Team55.txt", "ODC_2026_TeamXX_debug.txt", "wrong team number"),
    ],
)
def test_ranking_certify_refuses_the_reviewer_s_two_probes(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path, official: str, aux: str, reason: str
) -> None:
    # Both certified on cbbb704 with the official report unnumbered.
    from orbitwin.state_probe import load_scenario
    from orbitwin.verify import certify_candidate

    config = _crossing_ranking()
    script = _crossing_with(tmp_path, official, aux)
    scenario, _, _ = load_scenario(script, config, gmat_console=fake_gmat,
                                   startup_file=fake_startup)
    assert scenario.report_files[0].name == "OfficialReport"
    miss = stamped(zero_burn_candidate(scenario, config), script, config)
    with pytest.raises(PolicyError, match=reason):
        certify_candidate(script, miss, config, fake_gmat, fake_startup, tmp_path / "c")
    assert not (tmp_path / "c").exists()


def test_ranking_certify_records_the_organizer_s_answer(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.state_probe import load_scenario
    from orbitwin.verify import certify_candidate

    config = _crossing_ranking()
    config = dataclasses.replace(config, policy=dataclasses.replace(
        config.policy, report_filename_as_written="ReportFile1.txt"))
    validate_config(config)
    script = _crossing_with(tmp_path, "ReportFile1.txt", None)
    scenario, _, _ = load_scenario(script, config, gmat_console=fake_gmat,
                                   startup_file=fake_startup)
    miss = stamped(zero_burn_candidate(scenario, config), script, config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evidence = certify_candidate(script, miss, config, fake_gmat, fake_startup,
                                     tmp_path / "c")
    assert evidence["official_report_files"] == ["ReportFile1.txt"]
    assert evidence["report_filename_as_written"] == "ReportFile1.txt"
    certificate = json.loads(
        (tmp_path / "c" / "submission" / "certification.json").read_text(encoding="utf-8")
    )
    assert certificate["report_filename_as_written"] == "ReportFile1.txt"
    assert any("kept as written" in item for item in evidence["report_name_warnings"])


# --- only the reports GMAT writes are uploaded (#173) ---------------------

_WRITTEN_ADD = (
    "{SC_Human.UTCModJulian, SC_Human.EarthMJ2000Eq.X, SC_Human.EarthMJ2000Eq.Y, "
    "SC_Human.EarthMJ2000Eq.Z, SC_Alien.EarthMJ2000Eq.X, SC_Alien.EarthMJ2000Eq.Y, "
    "SC_Alien.EarthMJ2000Eq.Z}"
)


def _unwritten_reportfile1(tmp_path: Path, first: str, second: str) -> Path:
    """ReportFile1 with WriteReport = false, then a written Report2."""
    text = CROSSING.read_text(encoding="utf-8").replace("OfficialReport", "ReportFile1")
    text = text.replace("'ReportFile1.txt'", f"'{first}'")
    assert "ReportFile1.WriteReport = true;" in text
    text = text.replace("ReportFile1.WriteReport = true;", "ReportFile1.WriteReport = false;")
    marker = "ReportFile1.AppendToExistingFile = false;\n"
    assert marker in text
    second_report = (
        f"Create ReportFile Report2;\nReport2.Filename = '{second}';\n"
        f"Report2.Precision = 16;\nReport2.Add = {_WRITTEN_ADD};\nReport2.WriteHeaders = true;\n"
    )
    script = tmp_path / "unwritten.script"
    script.write_text(
        text.replace(marker, marker + second_report, 1) + "% FAKE_GMAT_MISS_KM 900\n",
        encoding="utf-8",
    )
    return script


def _certify_crossing(script: Path, config: OrbitWinConfig, fake_gmat: Path,
                      fake_startup: Path, out: Path):
    from orbitwin.state_probe import load_scenario
    from orbitwin.verify import certify_candidate

    scenario, _, _ = load_scenario(script, config, gmat_console=fake_gmat,
                                   startup_file=fake_startup)
    miss = stamped(zero_burn_candidate(scenario, config), script, config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return scenario, certify_candidate(script, miss, config, fake_gmat, fake_startup, out)


def test_an_unwritten_reportfile1_is_not_the_official_report(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # Its placeholder took 54, and the upload, Report2, carried none.
    from orbitwin.script_renderer import official_report
    from orbitwin.state_probe import load_scenario

    config = _crossing_ranking()
    script = _unwritten_reportfile1(tmp_path, "ODC_2026_TeamXX.txt", "ODC_2026_Report.txt")
    scenario, _, _ = load_scenario(script, config, gmat_console=fake_gmat,
                                   startup_file=fake_startup)
    assert official_report(scenario).name == "Report2"  # type: ignore[union-attr]
    with pytest.raises(PolicyError, match=r"the official report Report2\.Filename "
                       r"'ODC_2026_Report\.txt' has no team-number placeholder: the official "
                       r"script names no team placeholder"):
        _certify_crossing(script, config, fake_gmat, fake_startup, tmp_path / "c")
    assert not (tmp_path / "c").exists()


def test_a_written_report_takes_the_number_beside_an_unwritten_reportfile1(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # Refused before, and the override could not help.
    config = _crossing_ranking()
    script = _unwritten_reportfile1(tmp_path, "ODC_2026_Report.txt", "ODC_2026_TeamXX.txt")
    _, evidence = _certify_crossing(script, config, fake_gmat, fake_startup, tmp_path / "c")
    submission = tmp_path / "c" / "submission"
    assert evidence["official_report_files"] == ["ODC_2026_Team54.txt"]
    assert (submission / "ODC_2026_Team54.txt").is_file()
    assert not (submission / "ODC_2026_Report.txt").exists()
    assert evidence["report_name_warnings"] == []


def test_an_unwritten_report_is_not_judged(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.state_probe import load_scenario

    config = _crossing_ranking()
    script = _unwritten_reportfile1(tmp_path, "ODC_2026_Team55.txt", "ODC_2026_TeamXX.txt")
    scenario, _, _ = load_scenario(script, config, gmat_console=fake_gmat,
                                   startup_file=fake_startup)
    assert_team_number_for_submission(scenario, config)
    assert team_number_warnings(scenario, config) == ()


def test_the_override_is_restated_where_there_is_no_certificate_to_read(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.cli import upload_summary

    config = dataclasses.replace(_crossing_ranking(), policy=dataclasses.replace(
        _crossing_ranking().policy, report_filename_as_written="ReportFile1.txt"))
    script = _crossing_with(tmp_path, "ReportFile1.txt", None)
    _, evidence = _certify_crossing(script, config, fake_gmat, fake_startup, tmp_path / "c")
    lines = upload_summary(evidence, tmp_path / "c", str(ROOT / "config" / "ranking.yaml"))
    assert any("keeps the organizer's name 'ReportFile1.txt'" in line for line in lines)
    plain = dict(evidence, report_filename_as_written=None)
    assert not any("organizer's name" in line
                   for line in upload_summary(plain, tmp_path / "c", None))
