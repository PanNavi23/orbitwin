"""The SPAD file the template's drag and SRP read, resolved as GMAT's FileManager::FindPath does.

GMAT's stock SphericalModel.spo has az/el in the reverse of GMAT's convention
(GMT-6588), so its drag pushes along the relative wind. Our verification flies
it; another file would fly differently on the organizer's machine.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import warnings
from pathlib import Path

import pytest
from conftest import quick_config

from orbitwin.config import load_config
from orbitwin.errors import ConfigurationError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.gmat_runner import resolve_spad_file, script_data_files
from orbitwin.spad import (
    GMT_6588,
    STOCK_SPHERICAL_MODEL_SHA256,
    spad_check,
    spad_files,
    spad_uses,
)

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
#: Stands in for SphericalModel.spo: the tests hash it as the stock file.
STOCK_BYTES = b"stock SphericalModel.spo stand-in\n"


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A GMAT layout: bin/GmatConsole, bin/gmat_startup_file.txt, data/vehicle/spad."""
    import orbitwin.spad as spad_module

    monkeypatch.setattr(
        spad_module, "STOCK_SPHERICAL_MODEL_SHA256", hashlib.sha256(STOCK_BYTES).hexdigest()
    )
    bin_dir = tmp_path / "gmat" / "bin"
    spad_dir = tmp_path / "gmat" / "data" / "vehicle" / "spad"
    bin_dir.mkdir(parents=True)
    spad_dir.mkdir(parents=True)
    console = bin_dir / "GmatConsole"
    console.write_text("", encoding="utf-8")
    startup = bin_dir / "gmat_startup_file.txt"
    startup.write_text(
        "ROOT_PATH = ../\nDATA_PATH = ROOT_PATH/data/\nSPAD_PATH = DATA_PATH/vehicle/spad/\n"
        "SPAD_SRP_FILE = SPAD_PATH/SphericalModel.spo\nEARTH_POT_PATH = DATA_PATH/gravity/earth/\n",
        encoding="utf-8",
    )
    # R1's force model names it; spad_check refuses a force-model file GMAT would miss.
    gravity = tmp_path / "gmat" / "data" / "gravity" / "earth"
    gravity.mkdir(parents=True)
    (gravity / "EGM96.cof").write_bytes(b"EGM96 stand-in\n")
    (spad_dir / "SphericalModel.spo").write_bytes(STOCK_BYTES)
    (spad_dir / "Other.spo").write_bytes(b"another vehicle\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    return {"bin": bin_dir, "spad": spad_dir, "console": console, "startup": startup,
            "scripts": scripts}


def _resolve(
    install: dict[str, Path], named: str | None, field: str = "SPADDragFile"
) -> Path | None:
    return resolve_spad_file(named, field, install["scripts"], install["startup"],
                             install["console"])


def test_a_dot_dot_name_is_found_beside_gmat_console(install: dict[str, Path]) -> None:
    """What the ranking templates write."""
    found = _resolve(install, "../data/vehicle/spad/SphericalModel.spo")
    assert found == (install["spad"] / "SphericalModel.spo").resolve()


def test_the_scripts_folder_is_searched_first(install: dict[str, Path]) -> None:
    beside = install["scripts"] / "../data/vehicle/spad"
    beside.mkdir(parents=True)
    (beside / "SphericalModel.spo").write_bytes(b"the script folder's copy\n")
    found = _resolve(install, "../data/vehicle/spad/SphericalModel.spo")
    assert found == (beside / "SphericalModel.spo").resolve()


def test_a_bare_name_is_found_in_spad_path_for_srp_only(install: dict[str, Path]) -> None:
    """SPAD_SRP_FILE is a FileManager type; SPAD drag has none, so GMAT tries its cwd."""
    assert _resolve(install, "Other.spo", "SPADSRPFile") == (install["spad"] / "Other.spo").resolve()
    assert _resolve(install, "Other.spo", "SPADDragFile") is None
    (install["bin"] / "Other.spo").write_bytes(b"in bin\n")
    assert _resolve(install, "Other.spo", "SPADDragFile") == (install["bin"] / "Other.spo").resolve()


def test_an_absolute_name_is_taken_as_written(install: dict[str, Path]) -> None:
    other = install["spad"] / "Other.spo"
    assert _resolve(install, str(other)) == other.resolve()
    assert _resolve(install, str(install["spad"] / "Missing.spo")) is None


def test_windows_separators(install: dict[str, Path]) -> None:
    found = _resolve(install, "..\\data\\vehicle\\spad\\SphericalModel.spo")
    assert found == (install["spad"] / "SphericalModel.spo").resolve()


def test_a_blank_name_is_the_startup_default_for_srp_only(install: dict[str, Path]) -> None:
    """FindPath swaps a blank name for the type's default file; drag has no type."""
    stock = (install["spad"] / "SphericalModel.spo").resolve()
    assert _resolve(install, "", "SPADSRPFile") == stock
    assert _resolve(install, "", "SPADDragFile") is None
    # Never set: Spacecraft keeps an empty full path and throws when the force asks.
    assert _resolve(install, None, "SPADSRPFile") is None


def test_a_bare_srp_name_without_a_startup_default_is_looked_for_beside_gmat_console(
    install: dict[str, Path],
) -> None:
    install["startup"].write_text("ROOT_PATH = ../\n", encoding="utf-8")
    assert _resolve(install, "Other.spo", "SPADSRPFile") is None
    (install["bin"] / "Other.spo").write_bytes(b"in bin\n")
    assert _resolve(install, "Other.spo", "SPADSRPFile") == (install["bin"] / "Other.spo").resolve()


def test_the_data_manifest_resolves_spad_fields_the_same_way(
    tmp_path: Path, install: dict[str, Path]
) -> None:
    """Its generic lookup found a bare drag name in SPAD_PATH, where GMAT does not look."""
    script = _template(tmp_path, "bare.script", "Create Spacecraft SC;\nSC.SPADDragFile = 'Other.spo';\n"
                       "SC.SPADSRPFile = 'Other.spo';\nBeginMissionSequence;\n")
    found = script_data_files(script, install["startup"], install["console"])
    assert found["script:SC.SPADDragFile"] == "Other.spo"
    assert found["script:SC.SPADSRPFile"] == (install["spad"] / "Other.spo").resolve()


def _template(tmp_path: Path, name: str, text: str) -> Path:
    script = tmp_path / "scripts" / name
    script.write_text(text, encoding="utf-8")
    return script


def _r1(tmp_path: Path, *edits: tuple[str, str]) -> Path:
    text = R1.read_text(encoding="utf-8")
    for pattern, replacement in edits:
        text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
        assert count, pattern
    return _template(tmp_path, "R1.script", text)


def _files(script: Path, install: dict[str, Path]):
    config = load_config(ROOT / "config" / "ranking.yaml")
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, horizon_s=6711.667)
    )
    scenario, document = parse_gmat_script(script, config)
    return scenario, document, spad_files(scenario, document, install["startup"], install["console"])


def test_the_ranking_template_reads_the_stock_file_for_both_forces_and_craft(
    tmp_path: Path, install: dict[str, Path]
) -> None:
    _, _, files = _files(_r1(tmp_path), install)
    assert [(f.craft, f.field) for f in files] == [
        ("SC_Alien", "SPADDragFile"), ("SC_Alien", "SPADSRPFile"),
        ("SC_Human", "SPADDragFile"), ("SC_Human", "SPADSRPFile"),
    ]
    assert all(f.stock and f.force_models == ("DefaultProp_ForceModel",) for f in files)


def test_another_spo_is_recorded_and_warned_about_loudly(
    tmp_path: Path, install: dict[str, Path]
) -> None:
    script = _r1(tmp_path, (r"^SC_Human\.SPADDragFile = .*$",
                            "SC_Human.SPADDragFile = '../data/vehicle/spad/Other.spo';"))
    scenario, document, _ = _files(script, install)
    # Returned for the caller to show once; inspect printed it twice when this warned too.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        record, notes = spad_check(scenario, document, install["console"], install["startup"])
    assert record is not None
    odd = [r for r in record if not r["stock"]]
    assert [(r["craft"], r["field"]) for r in odd] == [("SC_Human", "SPADDragFile")]
    assert odd[0]["sha256"] == hashlib.sha256(b"another vehicle\n").hexdigest()
    assert len(notes) == 1 and GMT_6588 in notes[0] and "SC_Human.SPADDragFile" in notes[0]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ((r"^SC_Human\.SPADDragFile = .*$", "SC_Human.SPADDragFile = 'Missing.spo';"),
         "SC_Human.SPADDragFile = 'Missing.spo'"),
        ((r"^SC_Alien\.SPADSRPFile = .*\n", ""), "SC_Alien sets no SPADSRPFile"),
        # Beside the template by a '..' name: a run folder cannot hold it there.
        ((r"^SC_Human\.SPADDragFile = .*$", "SC_Human.SPADDragFile = '../beside/Beside.spo';"),
         "SC_Human.SPADDragFile = '../beside/Beside.spo', read by DefaultProp_ForceModel "
         "(found beside the template only, by a name with '..'"),
    ],
)
def test_a_file_gmat_would_not_find_is_refused(
    tmp_path: Path, install: dict[str, Path], edit: tuple[str, str], message: str
) -> None:
    (tmp_path / "beside").mkdir()
    (tmp_path / "beside" / "Beside.spo").write_bytes(STOCK_BYTES)
    with pytest.raises(ConfigurationError, match=re.escape(message)):
        _files(_r1(tmp_path, edit), install)


@pytest.mark.parametrize("named", ["Beside.spo", "./Beside.spo", "spad\\Beside.spo"])
def test_a_file_beside_the_template_is_what_every_run_reads(
    tmp_path: Path, install: dict[str, Path], named: str
) -> None:
    """run_gmat copies it beside each run's script, where GMAT looks first."""
    beside = install["scripts"] / named.replace("\\", "/")
    beside.parent.mkdir(parents=True, exist_ok=True)
    beside.write_bytes(b"the organizer's own vehicle\n")
    escaped = named.replace("\\", "\\\\")  # a re.sub replacement
    script = _r1(tmp_path, (r"^SC_Human\.SPADDragFile = .*$", f"SC_Human.SPADDragFile = '{escaped}';"))
    _, _, files = _files(script, install)
    human = next(f for f in files if (f.craft, f.field) == ("SC_Human", "SPADDragFile"))
    assert human.path == human.template_path == str(beside.resolve())
    assert not human.stock


def test_a_file_that_depends_on_the_scripts_folder_is_warned_about(
    tmp_path: Path, install: dict[str, Path]
) -> None:
    """GMAT's working directory is the script's folder; our runs copy it into an empty one."""
    beside = tmp_path / "data" / "vehicle" / "spad"
    beside.mkdir(parents=True)
    (beside / "SphericalModel.spo").write_bytes(b"a copy beside the template\n")
    scenario, document, files = _files(_r1(tmp_path), install)
    assert all(f.stock and f.path == str((install["spad"] / "SphericalModel.spo").resolve())
               for f in files)
    assert all(f.template_path == str((beside / "SphericalModel.spo").resolve()) for f in files)
    _, notes = spad_check(scenario, document, install["console"], install["startup"])
    assert len(notes) == 4 and all("DEPENDS ON THE SCRIPT'S FOLDER" in n for n in notes)


@pytest.mark.parametrize(
    "edits",
    [
        # Drag switched off: SPADFile as DragModel is never read.
        ((r"^DefaultProp_ForceModel\.Drag\.AtmosphereModel = .*$",
          "DefaultProp_ForceModel.Drag.AtmosphereModel = None;"),),
        ((r"^DefaultProp_ForceModel\.Drag\.DragModel = .*$",
          "DefaultProp_ForceModel.Drag.DragModel = 'Spherical';"),),
    ],
)
def test_only_forces_that_read_a_spad_file_are_checked(
    tmp_path: Path, install: dict[str, Path], edits: tuple[tuple[str, str], ...]
) -> None:
    script = _r1(tmp_path, *edits,
                 (r"^SC_Human\.SPADDragFile = .*$", "SC_Human.SPADDragFile = 'Missing.spo';"))
    _, _, files = _files(script, install)
    assert {f.field for f in files} == {"SPADSRPFile"}


def test_srp_off_reads_no_spad_file(tmp_path: Path, install: dict[str, Path]) -> None:
    script = _r1(tmp_path, (r"^DefaultProp_ForceModel\.SRP = On;$",
                            "DefaultProp_ForceModel.SRP = Off;"),
                 (r"^SC_Alien\.SPADSRPFile = .*\n", ""))
    _, _, files = _files(script, install)
    assert {f.field for f in files} == {"SPADDragFile"}


def test_a_template_without_spad_forces_records_an_empty_list(workshop_script: Path) -> None:
    scenario, document = parse_gmat_script(workshop_script, quick_config())
    assert spad_uses(scenario, document) == {}
    assert spad_check(scenario, document, None, None) == ([], [])


def test_without_a_console_the_file_is_not_resolved_and_the_summary_says_so(
    tmp_path: Path, install: dict[str, Path]
) -> None:
    scenario, document, _ = _files(_r1(tmp_path), install)
    record, notes = spad_check(scenario, document, None, None)
    assert record is None and "not resolved or hashed" in notes[0]


@pytest.mark.gmat
def test_the_stock_hash_is_r2026as_file(gmat: tuple[str, str]) -> None:
    console, _ = gmat
    stock = Path(console).resolve().parents[1] / "data" / "vehicle" / "spad" / "SphericalModel.spo"
    if not stock.is_file():
        pytest.fail(f"{stock} is not there; the install is not stock R2026a")
    assert hashlib.sha256(stock.read_bytes()).hexdigest() == STOCK_SPHERICAL_MODEL_SHA256


def _ranking_quick():
    base = quick_config()
    return dataclasses.replace(
        base,
        policy=dataclasses.replace(base.policy, rules="ranking", horizon_s=6711.667,
                                   dv_limit_km_s=0.25, dv_safe_limit_km_s=0.249,
                                   total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99),
        pursuer_name="SC_Human", target_name="SC_Alien",
    )


def _absolute(tmp_path: Path, install: dict[str, Path], human_drag: str) -> Path:
    """R1 with every SPAD name absolute, so the fake console's folder does not matter."""
    stock = str(install["spad"] / "SphericalModel.spo")
    return _r1(
        tmp_path,
        (r"^(SC_\w+\.SPAD(?:Drag|SRP)File) = .*$", rf"\1 = '{stock}';"),
        (r"^SC_Human\.SPADDragFile = .*$", f"SC_Human.SPADDragFile = '{human_drag}';"),
    )


def test_solve_records_the_spad_files_and_warns_in_its_summary(
    tmp_path: Path, install: dict[str, Path], fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.pipeline import solve_scenario

    script = _absolute(tmp_path, install, str(install["spad"] / "Other.spo"))
    with pytest.warns(RuntimeWarning, match="NOT GMAT'S STOCK"):
        summary = solve_scenario(script, _ranking_quick(), tmp_path / "solve",
                                 gmat_console=fake_gmat, startup_file=fake_startup,
                                 gmat_verify_top=0)
    record = summary["spad_files"]
    assert len(record) == 4 and sum(not r["stock"] for r in record) == 1
    assert any("NOT GMAT'S STOCK" in w and GMT_6588 in w for w in summary["warnings"])
    written = json.loads((tmp_path / "solve" / "summary.json").read_text(encoding="utf-8"))
    assert written["spad_files"] == record


def test_solve_refuses_a_missing_spad_file_before_writing_anything(
    tmp_path: Path, install: dict[str, Path], fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin.pipeline import solve_scenario

    script = _absolute(tmp_path, install, str(install["spad"] / "Missing.spo"))
    with pytest.raises(ConfigurationError, match=r"Missing\.spo"):
        solve_scenario(script, _ranking_quick(), tmp_path / "solve",
                       gmat_console=fake_gmat, startup_file=fake_startup, gmat_verify_top=0)
    assert not (tmp_path / "solve").exists()
