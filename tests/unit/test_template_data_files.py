"""Data files the template names beside itself are what every GMAT run reads.

GMAT looks for a relative data-file name in the script's folder first. Our runs
copy the script into a folder of their own, so a tide file shipped beside the
template was missing there: R2026a stopped with 'The file name "X.tide" does
not exist', where the organizer's GMAT, running the template in place, flew it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from conftest import quick_config, stamped

from orbitwin.errors import ConfigurationError, GmatRunError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.gmat_runner import (
    force_model_file_problems,
    run_gmat,
    script_data_files,
    stage_template_files,
    stageable_name,
)
from orbitwin.search import search_one_burn
from orbitwin.verify import certify_candidate

TIDE = "DefaultProp_ForceModel.GravityField.Earth.TideFile"


def _script(folder: Path, *fields: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "official.script"
    script.write_text(
        "Create ForceModel FM;\n"
        + "".join(f"{field}\n" for field in fields)
        + "Create ReportFile ReportFile1;\nReportFile1.Filename = 'beside.txt';\n"
        "BeginMissionSequence;\n",
        encoding="ascii",
    )
    return script


def _file(path: Path, text: str = "k 2 0 0.30190\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")
    return path


@pytest.mark.parametrize(
    ("named", "where"),
    [
        ("Earth.tide", "Earth.tide"),
        ("./Earth.tide", "Earth.tide"),
        ("tides/Earth.tide", "tides/Earth.tide"),
        ("tides\\Earth.tide", "tides/Earth.tide"),
        # A Windows path is relative to GMAT on Linux (R2026a found it beside the script).
        ("C:\\data\\Earth.tide", "C:/data/Earth.tide"),
    ],
)
def test_a_file_beside_the_template_is_copied_beside_the_run(
    tmp_path: Path, named: str, where: str
) -> None:
    template = _script(tmp_path / "template", f"FM.GravityField.Earth.TideFile = '{named}';")
    source = _file(tmp_path / "template" / where)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staged = stage_template_files(template, template.parent, run_dir)
    copy = run_dir / where
    assert copy.read_bytes() == source.read_bytes()
    assert staged == {str(copy.resolve()): str(source.resolve())}


def test_what_cannot_or_need_not_be_copied_is_left(tmp_path: Path) -> None:
    outside = _file(tmp_path / "Up.tide")
    absolute = _file(tmp_path / "elsewhere" / "Abs.tide")
    template = _script(
        tmp_path / "template",
        "FM.GravityField.Earth.TideFile = '../Up.tide';",
        f"FM.GravityField.Luna.TideFile = '{absolute}';",
        "FM.GravityField.Mars.TideFile = 'NotThere.tide';",
        "FM.Drag.CSSISpaceWeatherFile = 'SW.txt';",
        "Earth.EopFileName = 'eop.txt';",
        "Create Spacecraft SC;",
        "SC.ModelFile = 'aura.3ds';",
    )
    # An output and a drawing are not data files GMAT reads, and R2026a never
    # looked beside the script for a CSSI or EOP file.
    _file(tmp_path / "template" / "beside.txt")
    _file(tmp_path / "template" / "aura.3ds")
    _file(tmp_path / "template" / "SW.txt")
    _file(tmp_path / "template" / "eop.txt")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert stage_template_files(template, template.parent, run_dir) == {}
    assert sorted(item.name for item in run_dir.iterdir()) == []
    assert outside.is_file()


@pytest.mark.parametrize(
    ("name", "stageable"),
    [
        ("Earth.tide", True), ("./a/b.tide", True), ("a\\b.tide", True),
        ("../x.tide", False), ("a/../../x.tide", False), ("..\\x.tide", False),
        ("/abs/x.tide", False), ("", False),
    ],
)
def test_which_names_a_run_folder_can_hold(name: str, stageable: bool) -> None:
    assert stageable_name(name) is stageable


def test_a_different_file_already_in_the_run_folder_is_refused(tmp_path: Path) -> None:
    template = _script(tmp_path / "template", "FM.GravityField.Earth.TideFile = 'Earth.tide';")
    _file(tmp_path / "template" / "Earth.tide")
    run_dir = tmp_path / "run"
    _file(run_dir / "Earth.tide", "k 2 0 9.9\n")
    with pytest.raises(GmatRunError, match="already there in the run folder"):
        stage_template_files(template, template.parent, run_dir)
    # The same bytes, as a rerun into one folder leaves them, are fine.
    _file(run_dir / "Earth.tide")
    assert stage_template_files(template, template.parent, run_dir)


def test_nothing_is_copied_when_the_run_is_the_templates_own_folder(tmp_path: Path) -> None:
    template = _script(tmp_path / "template", "FM.GravityField.Earth.TideFile = 'Earth.tide';")
    _file(tmp_path / "template" / "Earth.tide")
    assert stage_template_files(template, template.parent, template.parent) == {}


def test_the_manifest_looks_beside_the_script_first(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _script(tmp_path / "run", "FM.GravityField.Earth.TideFile = 'tides\\Earth.tide';")
    beside = _file(tmp_path / "run" / "tides" / "Earth.tide")
    found = script_data_files(script, fake_startup, fake_gmat)
    assert found["script:FM.GravityField.Earth.TideFile"] == beside.resolve()


def test_the_manifest_never_takes_an_eop_or_cssi_file_from_beside_the_script(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _script(
        tmp_path / "run", "Earth.EopFileName = 'eop.txt';", "FM.Drag.CSSISpaceWeatherFile = 'SW.txt';"
    )
    _file(tmp_path / "run" / "eop.txt")
    _file(tmp_path / "run" / "SW.txt")
    found = script_data_files(script, fake_startup, fake_gmat)
    assert found["script:Earth.EopFileName"] == "eop.txt"
    assert found["script:FM.Drag.CSSISpaceWeatherFile"] == "SW.txt"


def test_run_gmat_reads_the_templates_tide_file_and_records_it_as_the_templates(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    template = _script(tmp_path / "template", "FM.GravityField.Earth.TideFile = 'Earth.tide';")
    tide = _file(tmp_path / "template" / "Earth.tide")
    alone = run_gmat(fake_gmat, template, tmp_path / "alone", [], startup_file=fake_startup)
    assert not alone.success
    assert 'The file name "Earth.tide" does not exist' in Path(alone.stderr_path).read_text()
    entries = []
    for name in ("one", "two"):
        run = run_gmat(
            fake_gmat, template, tmp_path / name, [], startup_file=fake_startup,
            template_dir=template.parent,
        )
        assert run.success, run.error
        entries.append(run.data_file_sha256["script:FM.GravityField.Earth.TideFile"])
    # One path in every run, so certify's one-set-of-files check holds.
    assert entries[0] == entries[1]
    assert entries[0]["path"] == str(tide.resolve())


def test_certify_flies_a_template_with_a_tide_file_beside_it(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    text = workshop_script.read_text(encoding="utf-8-sig")
    text, count = re.subn(
        r"^(DefaultProp_ForceModel\.GravityField\.Earth\.TideModel = .*)$",
        rf"\1\n{TIDE} = 'Earth.tide';", text, flags=re.MULTILINE,
    )
    assert count == 1
    folder = tmp_path / "original"
    folder.mkdir()
    template = folder / "official.script"
    template.write_text(text, encoding="ascii")
    tide = _file(folder / "Earth.tide")
    config = quick_config()
    scenario, _ = parse_gmat_script(template, config)
    candidate = search_one_burn(scenario, config)[0]
    stamped(candidate, template, config)
    certify_candidate(
        template, candidate, config, fake_gmat, fake_startup, tmp_path / "certify", runs=2
    )
    certification = json.loads(
        (tmp_path / "certify" / "submission" / "certification.json").read_text(encoding="utf-8")
    )
    assert certification["gmat_data_files"][f"script:{TIDE}"]["path"] == str(tide.resolve())


@pytest.mark.parametrize(
    ("named", "said"),
    [
        ("Earth.tide", "'Earth.tide', found nowhere (beside the template, then"),
        ("../up/Earth.tide", "found beside the template only, by a name with '..'"),
    ],
)
def test_a_force_model_file_gmat_would_miss_is_refused_before_any_run(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    named: str, said: str,
) -> None:
    # The state probe drops the force models, so it never tripped on these.
    from orbitwin.pipeline import solve_scenario

    _file(tmp_path / "up" / "Earth.tide")
    text = workshop_script.read_text(encoding="utf-8-sig")
    folder = tmp_path / "original"
    folder.mkdir()
    template = folder / "official.script"
    template.write_text(
        text.replace("BeginMissionSequence;", f"{TIDE} = '{named}';\nBeginMissionSequence;"),
        encoding="ascii",
    )
    with pytest.raises(ConfigurationError, match=re.escape(said)) as raised:
        solve_scenario(template, quick_config(), tmp_path / "solve",
                       gmat_console=fake_gmat, startup_file=fake_startup)
    assert "RUNBOOK step 1" in str(raised.value)
    assert not (tmp_path / "solve").exists()


def test_a_cssi_file_beside_the_template_is_refused_where_gmat_would_not_look(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.pipeline import solve_scenario

    text = workshop_script.read_text(encoding="utf-8-sig")
    folder = tmp_path / "original"
    folder.mkdir()
    template = folder / "official.script"
    template.write_text(
        text.replace(
            "BeginMissionSequence;",
            "DefaultProp_ForceModel.Drag.CSSISpaceWeatherFile = 'SW.txt';\nBeginMissionSequence;",
        ),
        encoding="ascii",
    )
    _file(folder / "SW.txt")
    with pytest.raises(ConfigurationError, match=re.escape("not beside the template")):
        solve_scenario(template, quick_config(), tmp_path / "solve",
                       gmat_console=fake_gmat, startup_file=fake_startup)


def test_a_name_without_an_extension_is_completed_as_gmat_does(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # HarmonicField and GravityField add .cof or .tide to a name with no "." in it.
    template = _script(
        tmp_path / "template",
        "FM.GravityField.Earth.PotentialFile = 'EGM96';",
        "FM.GravityField.Earth.TideFile = 'EarthTest';",
    )
    source = _file(tmp_path / "template" / "EarthTest.tide")
    assert force_model_file_problems(template, fake_startup, fake_gmat) == []
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staged = stage_template_files(template, template.parent, run_dir)
    assert staged == {str((run_dir / "EarthTest.tide").resolve()): str(source.resolve())}


@pytest.mark.parametrize("named", ["orbitwin_startup.txt", "stdout.txt", "gmat-output/x.tide"])
def test_a_file_named_like_one_the_run_writes_is_refused(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path, named: str
) -> None:
    template = _script(tmp_path / "template", f"FM.GravityField.Earth.TideFile = '{named}';")
    _file(tmp_path / "template" / named)
    with pytest.raises(GmatRunError, match="a name this run writes itself"):
        run_gmat(fake_gmat, template, tmp_path / "run", [], startup_file=fake_startup,
                 template_dir=template.parent)


def test_a_drag_file_of_a_force_model_no_propagator_flies_is_not_checked(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # GMAT opens a CSSI file only when the force model starts.
    template = _script(
        tmp_path / "template",
        "FM.Drag.CSSISpaceWeatherFile = 'Missing.txt';",
        "Create ForceModel Flown;",
        "Flown.Drag.CSSISpaceWeatherFile = 'AlsoMissing.txt';",
        "Create Propagator Prop;",
        "Prop.FM = Flown;",
    )
    problems = force_model_file_problems(template, fake_startup, fake_gmat)
    assert len(problems) == 1 and "Flown.Drag.CSSISpaceWeatherFile" in problems[0]


def test_a_drag_file_of_a_propagator_no_propagate_uses_is_not_checked(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.spad import spad_check

    text = workshop_script.read_text(encoding="utf-8-sig")
    folder = tmp_path / "original"
    folder.mkdir()
    template = folder / "official.script"
    template.write_text(
        text.replace(
            "BeginMissionSequence;",
            "Create ForceModel Spare_ForceModel;\n"
            "Spare_ForceModel.Drag.CSSISpaceWeatherFile = 'Missing.txt';\n"
            "Create Propagator Spare;\nSpare.FM = Spare_ForceModel;\nBeginMissionSequence;",
        ),
        encoding="ascii",
    )
    scenario, document = parse_gmat_script(template, quick_config())
    assert spad_check(scenario, document, fake_gmat, fake_startup) == ([], [])


def test_the_fake_console_completes_a_tide_name_as_gmat_does(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    template = _script(tmp_path / "template", "FM.GravityField.Earth.TideFile = 'EarthTest';")
    _file(tmp_path / "template" / "EarthTest.tide")
    run = run_gmat(fake_gmat, template, tmp_path / "run", [], startup_file=fake_startup,
                   template_dir=template.parent)
    assert run.success, run.error
