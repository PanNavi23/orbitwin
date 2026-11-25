"""On stock GmatConsole, a tide file beside the template flies in our run folders."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from orbitwin.gmat_runner import gmat_errors, run_gmat, said_by_gmat

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
REPORT = "ODC_2026_TeamXX.txt"

pytestmark = pytest.mark.gmat


def test_a_tide_file_beside_the_template_is_found_from_the_run_folder(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    console, startup = gmat
    text = R1.read_text(encoding="utf-8")
    text = re.sub(r"ElapsedSecs = 6711\.667", "ElapsedSecs = 60", text)
    text = text.replace(
        "GravityField.Earth.TideModel = 'SolidAndPole';\n",
        "GravityField.Earth.TideModel = 'SolidAndPole';\n"
        "DefaultProp_ForceModel.GravityField.Earth.TideFile = 'Earth.tide';\n",
    )
    folder = tmp_path / "original"
    folder.mkdir()
    template = folder / "official.script"
    template.write_text(text, encoding="ascii")
    tide = folder / "Earth.tide"
    tide.write_text("k 2 0 0.29525\nk 2 1 0.29470\nk 2 2 0.29801\n", encoding="ascii")

    alone = run_gmat(console, template, tmp_path / "alone", [REPORT], startup_file=startup)
    assert not alone.success
    assert any('"Earth.tide" does not exist' in str(error)
               for error in gmat_errors(alone, 10).errors)

    staged = run_gmat(
        console, template, tmp_path / "staged", [REPORT], startup_file=startup,
        template_dir=folder,
    )
    assert staged.success, said_by_gmat(staged)
    key = "script:DefaultProp_ForceModel.GravityField.Earth.TideFile"
    assert staged.data_file_sha256[key]["path"] == str(tide.resolve())
