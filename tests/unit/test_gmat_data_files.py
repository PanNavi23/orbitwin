"""Every GMAT run records the data files its startup file points at."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from orbitwin.gmat_runner import (
    data_file_hashes,
    run_gmat,
    script_data_files,
    startup_data_files,
)
from orbitwin.models import GmatRunResult, file_sha256


def test_the_startup_files_data_files_are_found_and_hashed(tmp_path: Path) -> None:
    bin_dir = tmp_path / "gmat" / "bin"
    data = tmp_path / "gmat" / "data"
    (data / "time").mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    (data / "time" / "tai-utc.dat").write_text("leap\n", encoding="utf-8")
    absolute = tmp_path / "eop.txt"
    absolute.write_text("eop\n", encoding="utf-8")
    startup = tmp_path / "startup.txt"
    startup.write_text(
        "# LEAP_SECS_FILE = nowhere\n"
        "ROOT_PATH = ../\n"
        "DATA_PATH = ROOT_PATH/data/\n"
        "TIME_PATH = DATA_PATH/time/\n"
        "LEAP_SECS_FILE = TIME_PATH/tai-utc.dat\n"
        f"EOP_FILE = {absolute}\n"
        "EGM96_FILE = DATA_PATH/gravity/earth/EGM96.cof\n"
        "LOG_FILE = TIME_PATH/tai-utc.dat\n",
        encoding="utf-8",
    )
    console = bin_dir / "GmatConsole"
    found = startup_data_files(startup, console)
    assert found == {
        "LEAP_SECS_FILE": (data / "time" / "tai-utc.dat").resolve(),
        "EOP_FILE": absolute.resolve(),
    }
    hashes = data_file_hashes(startup, console)
    assert hashes["EOP_FILE"] == {
        "path": str(absolute.resolve()),
        "sha256": hashlib.sha256(b"eop\n").hexdigest(),
    }
    # The cache must not hide a refreshed file.
    absolute.write_text("refreshed eop\n", encoding="utf-8")
    assert data_file_hashes(startup, console)["EOP_FILE"]["sha256"] == (
        hashlib.sha256(b"refreshed eop\n").hexdigest()
    )


def test_a_run_records_them(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    eop = tmp_path / "eop.txt"
    eop.write_text("eop\n", encoding="utf-8")
    startup = tmp_path / "startup.txt"
    startup.write_text(
        fake_startup.read_text(encoding="utf-8") + f"EOP_FILE = {eop}\n", encoding="utf-8"
    )
    result = run_gmat(
        fake_gmat, workshop_script, tmp_path / "run", [], startup_file=startup, timeout_s=10.0
    )
    assert result.success
    assert result.data_file_sha256 == {
        "EOP_FILE": {"path": str(eop.resolve()), "sha256": hashlib.sha256(b"eop\n").hexdigest()},
        # Found through the fake startup file's EARTH_POT_PATH.
        "script:DefaultProp_ForceModel.GravityField.Earth.PotentialFile": {
            "path": str(
                (fake_startup.parent / "fake_gmat_data" / "gravity" / "earth" / "EGM96.cof").resolve()
            ),
            "sha256": file_sha256(fake_startup.parent / "fake_gmat_data/gravity/earth/EGM96.cof"),
        },
    }


def test_files_only_the_gui_reads_are_left_out(tmp_path: Path) -> None:
    gui = {
        "PERSONALIZATION_FILE": "MyGmat.ini",
        "EARTH_TEXTURE_FILE": "Earth.jpg",
        "MAIN_ICON_FILE": "GMAT.ico",
        "SPLASH_FILE": "splash.png",
        "STAR_FILE": "stars.txt",
        "CONSTELLATION_FILE": "constellations.txt",
        "BORDER_FILE": "border.txt",
        "SPACECRAFT_MODEL_FILE": "aura.3ds",
        "HELP_HTML_FILE": "help.html",
    }
    lines = []
    for key, name in [*gui.items(), ("EOP_FILE", "eop.txt")]:
        (tmp_path / name).write_text(key, encoding="utf-8")
        lines.append(f"{key} = {tmp_path / name}")
    startup = tmp_path / "startup.txt"
    startup.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert set(startup_data_files(startup, tmp_path / "GmatConsole")) == {"EOP_FILE"}


def test_opening_the_gui_between_runs_does_not_stop_certify(
    tmp_path: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
    monkeypatch,
) -> None:
    """MyGmat.ini changes whenever the GUI opens; certify compares the data files."""
    from conftest import quick_config

    import orbitwin.gmat_runner as runner
    from orbitwin.search import search_one_burn
    from orbitwin.state_probe import load_scenario
    from orbitwin.verify import certify_candidate

    ini = tmp_path / "MyGmat.ini"
    eop = tmp_path / "eop.txt"
    eop.write_text("eop\n", encoding="utf-8")
    startup = tmp_path / "startup.txt"
    startup.write_text(
        fake_startup.read_text(encoding="utf-8")
        + f"PERSONALIZATION_FILE = {ini}\nEOP_FILE = {eop}\n",
        encoding="utf-8",
    )
    real = runner.data_file_hashes
    opened = iter(range(100))

    def gui_opened_meanwhile(*args):
        ini.write_text(f"[View] opened {next(opened)}\n", encoding="utf-8")
        return real(*args)

    monkeypatch.setattr(runner, "data_file_hashes", gui_opened_meanwhile)
    config = quick_config()
    scenario, _, _ = load_scenario(
        workshop_script, config, gmat_console=fake_gmat, startup_file=startup
    )
    candidate = search_one_burn(scenario, config)[0]
    candidate.metadata.update(scenario_sha256=scenario.source_sha256, horizon_s=scenario.max_time_s)
    certification = certify_candidate(
        workshop_script, candidate, config, fake_gmat, startup, tmp_path / "c", runs=2
    )
    assert "EOP_FILE" in certification["gmat_data_files"]
    assert not any("PERSONALIZATION" in key for key in certification["gmat_data_files"])


def _named_by_the_script(workshop_script: Path, tmp_path: Path, lines: str) -> Path:
    text = workshop_script.read_text(encoding="utf-8")
    path = tmp_path / "named.script"
    path.write_text(text.replace("Create ForceModel", lines + "\nCreate ForceModel", 1), encoding="utf-8")
    return path


def test_files_the_script_names_are_hashed_too(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Earth.EopFileName replaces the startup EOP: 23 m on SC_Human's insertion state."""
    eop = tmp_path / "eop_copy.txt"
    eop.write_text("ut1-utc 0.10\n", encoding="utf-8")
    de = tmp_path / "de.bsp"
    de.write_text("de\n", encoding="utf-8")
    script = _named_by_the_script(
        workshop_script,
        tmp_path,
        f"Earth.EopFileName = '{eop}';\nSolarSystem.SPKFilename = '{de}';\n"
        "SolarSystem.LSKFilename = 'nowhere.tls';\nLuna.EopFileName = '';",
    )

    def recorded() -> dict:
        result = run_gmat(
            fake_gmat, script, tmp_path / f"run{len(list(tmp_path.glob('run*')))}", [],
            startup_file=fake_startup, timeout_s=10.0,
        )
        assert result.success
        return result.data_file_sha256

    first = recorded()
    assert first["script:Earth.EopFileName"] == {
        "path": str(eop.resolve()), "sha256": hashlib.sha256(b"ut1-utc 0.10\n").hexdigest()
    }
    assert first["script:SolarSystem.SPKFilename"]["path"] == str(de.resolve())
    assert first["script:SolarSystem.LSKFilename"] == {"path": "nowhere.tls", "sha256": "not found"}
    assert not any(key.startswith("script:Luna") for key in first)  # '' is the startup file's
    # The template's own ReportFile Filename is an output, not data.
    assert not any("ReportFile1" in key for key in first)
    eop.write_text("ut1-utc 0.15\n", encoding="utf-8")
    assert recorded()["script:Earth.EopFileName"] != first["script:Earth.EopFileName"]


def test_a_named_file_not_found_is_recorded_and_warned(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = _named_by_the_script(workshop_script, tmp_path, "SolarSystem.LSKFilename = 'nowhere.tls';")
    with pytest.warns(RuntimeWarning, match=r"SolarSystem\.LSKFilename names 'nowhere\.tls'"):
        result = run_gmat(
            fake_gmat, script, tmp_path / "run", [], startup_file=fake_startup, timeout_s=10.0
        )
    assert result.data_file_sha256["script:SolarSystem.LSKFilename"]["sha256"] == "not found"


def test_a_built_in_bodys_file_name_is_its_texture(tmp_path: Path, workshop_script: Path) -> None:
    """CelestialBody.FileName is the OrbitView texture map, not data."""
    script = _named_by_the_script(
        workshop_script, tmp_path, "Earth.FileName = '../data/graphics/texture/Earth.jpg';"
    )
    startup = tmp_path / "startup.txt"
    startup.write_text("ROOT_PATH = ../\n", encoding="utf-8")
    named = script_data_files(script, startup, tmp_path / "bin" / "GmatConsole")
    assert not any(key.startswith("script:Earth") for key in named)


def _run_with(files: dict[str, str]) -> GmatRunResult:
    return GmatRunResult(
        run_id="r", success=True, return_code=0, started_at_utc="", duration_s=0.0,
        workdir="", script_path="", script_sha256="", executable_sha256="",
        startup_sha256=None, stdout_path="", stderr_path="", log_path=None,
        reports={}, report_sha256={},
        data_file_sha256={key: {"path": key, "sha256": value} for key, value in files.items()},
    )


BOTH = {"EOP_FILE": "e1", "script:FM.PotentialFile": "g1"}


@pytest.mark.parametrize(
    ("runs", "probe", "refused"),
    [
        ([BOTH, BOTH], {"EOP_FILE": "e1"}, None),  # the probe copy names fewer
        ([BOTH, {"script:FM.PotentialFile": "g1"}], None, "EOP_FILE"),  # deleted between runs
        ([BOTH, BOTH], {"EOP_FILE": "e1", "LSK_FILE": "l1"}, "LSK_FILE"),  # only the probe read it
        ([BOTH, {**BOTH, "EOP_FILE": "e2"}], None, "EOP_FILE"),  # changed
        ([BOTH, BOTH], {"EOP_FILE": "e2"}, "EOP_FILE"),  # the probe read another
    ],
)
def test_every_run_behind_a_certificate_read_the_same_files(
    runs: list[dict[str, str]], probe: dict[str, str] | None, refused: str | None
) -> None:
    from orbitwin.errors import GmatRunError
    from orbitwin.verify import _assert_one_set_of_data_files

    def check() -> None:
        _assert_one_set_of_data_files(
            [_run_with(files) for files in runs], _run_with(probe) if probe is not None else None
        )

    if refused is None:
        check()
    else:
        with pytest.raises(GmatRunError, match=f"\\({refused}\\)"):
            check()
