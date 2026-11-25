"""scripts/check_report_rows.py reads the report in its ReportFile's format.

tests/fixtures/report_rows holds GMAT R2026a reports of one verified R1 hit
(R1_hit.script), of the same hit run 30 s on (over), and of the zero-burn R1
template (miss), each in the formats of MANIFEST.json. Each file is the header,
the first two rows and the last rows, byte for byte; MANIFEST.json has the
SHA-256 of the whole report.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "tests" / "fixtures" / "report_rows"
MANIFEST = json.loads((ROWS / "MANIFEST.json").read_text(encoding="utf-8"))
FORMATS: dict[str, dict[str, str]] = MANIFEST["formats"]
ADD_LIST = (
    "SC_Alien.ElapsedSecs,SC_Alien.EarthMJ2000Eq.X,SC_Alien.EarthMJ2000Eq.Y,"
    "SC_Alien.EarthMJ2000Eq.Z,SC_Alien.EarthMJ2000Eq.VX,SC_Alien.EarthMJ2000Eq.VY,"
    "SC_Alien.EarthMJ2000Eq.VZ,SC_Human.ElapsedSecs,SC_Human.EarthMJ2000Eq.X,"
    "SC_Human.EarthMJ2000Eq.Y,SC_Human.EarthMJ2000Eq.Z,SC_Human.EarthMJ2000Eq.VX,"
    "SC_Human.EarthMJ2000Eq.VY,SC_Human.EarthMJ2000Eq.VZ"
)
#: The same format given as flags instead of a script.
FLAGS = {
    "default": [],
    "comma": ["--delimiter", ","],
    "semicolon": ["--delimiter", ";"],
    "noheader": ["--write-headers", "false", "--columns", ADD_LIST],
    "nofixed": ["--fixed-width", "false"],
    "noheader_comma_nofixed": [
        "--write-headers", "false", "--delimiter", ",", "--fixed-width", "false",
        "--columns", ADD_LIST,
    ],
    "backslash_t": ["--delimiter", "\\"],
}
EXIT = {"hit": 0, "over": 1, "miss": 1}


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_report_rows", ROOT / "scripts" / "check_report_rows.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = _load()


def run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str | Path
) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["check_report_rows.py", *map(str, args)])
    try:
        code = CHECK.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
    out, err = capsys.readouterr()
    # The file and script names are the lines that differ between formats.
    kept = (line for line in out.splitlines() if not line.startswith(("file ", "script ")))
    return code, "\n".join(kept), err


#: R1_hit.script's last goal, which the over run flew 30 s further.
HIT_GOAL = "ElapsedSecs = 4517.611829571932"
#: The zero-burn R1 template, which flew the miss (tests/integration/test_report_formats_gmat.py).
MISS = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"


def kind_of(report: str) -> str:
    """Which flight of R1 a fixture is: the hit, the hit flown 30 s on, or the zero burn."""
    return "over" if "_over" in report else "miss" if report.startswith("R1_miss") else "hit"


def script_in(tmp_path: Path, fields: dict[str, str], kind: str = "hit") -> Path:
    """The script of an R1 flight, ReportFile1's fields set as in its GMAT run."""
    text = (MISS if kind == "miss" else ROWS / "R1_hit.script").read_text(encoding="utf-8")
    if kind == "over":
        text = text.replace(HIT_GOAL, "ElapsedSecs = 4547.611829571932")
    lines = []
    for line in text.splitlines(keepends=True):
        key = line.split("=", 1)[0].strip().removeprefix("ReportFile1.")
        lines.append(f"ReportFile1.{key} = {fields[key]};\n" if key in fields else line)
    path = tmp_path / "final.script"
    path.write_text("".join(lines), encoding="utf-8")
    return path


#: R1_hit.script's ReportFile1.Filename.
UPLOADED = "ODC_2026_TeamXX.txt"
#: A probe without the pursuer's EarthMJ2000Eq velocity binds on its end only.
END_ONLY = "--accept-end-time-binding-only"
#: These fixtures keep their first and last rows only, so held to their script by hand the
#: binding warns of the rows missing between: exit 3 for a pass, and WARNING lines.
CUT = ["--binding", "warn"]


def unwarned(result: tuple[int, str, str]) -> tuple[int, str, str]:
    """A pass that warned past a cut fixture's missing rows, read as the pass it is."""
    code, out, err = result
    kept = "\n".join(line for line in out.splitlines() if not line.startswith("WARNING:"))
    return (0 if code == 3 else code), kept, err


def uploaded(tmp_path: Path, report: Path, name: str = UPLOADED) -> Path:
    """`report` under the name its script's ReportFile writes, as GMAT leaves it."""
    folder = tmp_path / "upload"
    folder.mkdir(exist_ok=True)
    (folder / name).write_bytes(report.read_bytes())
    return folder / name


def bound(tmp_path: Path, report: str, fields: dict[str, str]) -> tuple[Path, Path]:
    """A fixture under its uploaded name, and the script of its flight in its format."""
    return uploaded(tmp_path, ROWS / report), script_in(tmp_path, fields, kind_of(report))


@pytest.mark.parametrize("kind", sorted(EXIT))
@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_every_format_gets_the_default_verdict(
    kind: str, fmt: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    default = run(monkeypatch, capsys, ROWS / f"R1_{kind}_default.txt")
    assert default[0] == EXIT[kind], default
    report, script = bound(tmp_path, f"R1_{kind}_{fmt}.txt", FORMATS[fmt])
    assert unwarned(run(monkeypatch, capsys, report, "--script", script, *CUT)) == default
    assert run(monkeypatch, capsys, report, *FLAGS[fmt]) == default


@pytest.mark.parametrize("fmt", ["comma", "semicolon", "noheader", "backslash_t"])
def test_without_the_format_the_report_is_refused(
    fmt: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Main refused these too; only the wording changed.
    code, out, err = run(monkeypatch, capsys, ROWS / f"R1_hit_{fmt}.txt")
    assert code == 1 and "VERDICT" not in out
    assert any(why in err for why in ("is not numeric", "no numeric rows", "cannot locate"))


def test_a_non_numeric_row_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Main skipped it and judged the rows around it; skipping can move the last row.
    lines = (ROWS / "R1_hit_default.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    report = tmp_path / "report.txt"
    report.write_text("".join([*lines[:4], "garbage here\n", *lines[4:]]), encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, report)
    assert code == 1 and "VERDICT" not in out and "line 5 is not numeric" in err


def test_write_report_false_writes_nothing_to_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = script_in(tmp_path, {"WriteReport": "false"})
    code, _, err = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--script", script)
    assert code == 1 and "WriteReport is false" in err


def test_two_report_files_need_a_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = script_in(tmp_path, {})
    text = script.read_text(encoding="utf-8")
    script.write_text(
        text.replace(
            "BeginMissionSequence;",
            "Create ReportFile Extra;\nExtra.Filename = 'extra.txt';\nExtra.Delimiter = ',';\n"
            "BeginMissionSequence;",
        ),
        encoding="utf-8",
    )
    report = ROWS / "R1_hit_comma.txt"
    code, _, err = run(monkeypatch, capsys, report, "--script", script)
    assert code == 1 and "--report-file" in err
    # Extra reads it, but the report uploaded is ReportFile1's.
    code, _, err = run(monkeypatch, capsys, report, "--script", script, "--report-file", "Extra")
    assert code == 2 and "Extra is not" in err and "official report, ReportFile1" in err


def test_script_and_fields_are_not_both_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = script_in(tmp_path, {})
    code, _, err = run(
        monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--script", script, "--delimiter", ","
    )
    assert code == 2 and "--script gives the ReportFile format" in err


def test_the_sphere_is_the_rulebooks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = ROWS / "R1_hit_default.txt"
    for rules in ("ranking", "preliminary"):
        code, out, _ = run(monkeypatch, capsys, report, "--rules", rules)
        assert code == 0 and "rows within 5 km: 1" in out
    # Last row is 4.950002 km out.
    code, out, _ = run(monkeypatch, capsys, report, "--radius-km", "4.9")
    assert code == 1 and "rows within 4.9 km: 0" in out


def test_a_repeated_header_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Two runs in one file (AppendToExistingFile): main read on into the
    # second run and judged its last row. Now refused.
    text = (ROWS / "R1_hit_default.txt").read_text(encoding="utf-8")
    report = tmp_path / "twice.txt"
    report.write_text(text + text, encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, report)
    assert code == 1 and "VERDICT" not in out and "is not numeric" in err


def test_without_script_it_imports_only_numpy() -> None:
    code = (
        "import runpy, sys\n"
        f"sys.argv = ['check', {str(ROWS / 'R1_hit_default.txt')!r}]\n"
        "try:\n"
        f"    runpy.run_path({str(ROOT / 'scripts' / 'check_report_rows.py')!r}, "
        "run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0, exc.code\n"
        "print(sorted(m for m in ('scipy', 'yaml') if m in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip().splitlines()[-1] == "[]"


PROBES: dict[str, dict[str, str]] = {
    name: probe.get("fields", {}) for name, probe in MANIFEST["probes"].items()
}


def test_a_header_less_report_holding_two_runs_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT: AppendToExistingFile = true, run twice. Main and the first cut of
    # this read on into the second run and passed it.
    script = script_in(tmp_path, PROBES["append_noheader"])
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_append_noheader.txt", "--script", script)
    assert code == 1 and "VERDICT" not in out
    assert "SC_Alien.ElapsedSecs goes back from 4517.612829 to 0.000000" in err
    assert "AppendToExistingFile is true" in err


def test_a_miss_and_a_hit_in_one_file_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "two.txt"
    report.write_text(
        (ROWS / "R1_miss_noheader.txt").read_text(encoding="utf-8")
        + (ROWS / "R1_hit_noheader.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    code, out, err = run(monkeypatch, capsys, report, *FLAGS["noheader"])
    assert code == 1 and "VERDICT" not in out and "goes back from" in err


def _verdict(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(("LAST row", "VERDICT"))]


@pytest.mark.parametrize(
    "probe",
    ["greg_default", "greg_comma", "greg_noheader_nofixed", "greg_noheader_comma_nofixed"],
)
def test_a_gregorian_column_is_read_by_its_place(
    probe: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # GMAT writes `09 Nov 2026 02:00:00.000`: four tokens on whitespace, one on a comma.
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    report, script = bound(tmp_path, f"R1_{probe}.txt", PROBES[probe])
    code, out, err = run(monkeypatch, capsys, report, "--script", script, END_ONLY, *CUT)
    assert (code, _verdict(out)) == (3, _verdict(default[1])), err
    code, out, _ = run(
        monkeypatch, capsys, report, "--script", script, END_ONLY, *CUT,
        "--time-column", "SC_Human.UTCGregorian",
    )
    assert code == 1 and "VERDICT" not in out


def test_a_gregorian_column_needs_the_add_list(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_greg_default.txt")
    assert code == 1 and "VERDICT" not in out and "is not numeric" in err


def test_a_fused_gregorian_column_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT, no headers, ColumnWidth 23: `02:00:00.000-18013.43061577812`.
    script = script_in(tmp_path, PROBES["greg_noheader"])
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_greg_noheader.txt", "--script", script)
    assert code == 1 and "VERDICT" not in out and "is not numeric" in err


@pytest.mark.parametrize("probe", ["short_names", "earthfixed"])
def test_positions_in_any_shared_frame_are_found(
    probe: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # SC.X is SC.EarthMJ2000Eq.X; EarthFixed gives the same distance (R2026a).
    # The EarthMJ2000Eq columns GMAT also wrote are renamed out of reach.
    lines = (ROWS / f"R1_{probe}.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    report = tmp_path / "report.txt"
    report.write_text(
        "".join([lines[0].replace(".EarthMJ2000Eq.", ".Unused."), *lines[1:]]), encoding="utf-8"
    )
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    code, out, err = run(monkeypatch, capsys, report)
    assert (code, _verdict(out)) == (0, _verdict(default[1])), err


def test_a_shared_frame_other_than_earthmj2000eq_is_used() -> None:
    header, data = CHECK.read_report(str(ROWS / "R1_earthfixed.txt"))
    fixed_only = [name.replace("EarthMJ2000Eq", "Unused") for name in header]
    columns = data.shape[1]
    pursuer = CHECK.position_columns(fixed_only, columns, "SC_Human", "SC_Alien")
    target = CHECK.position_columns(fixed_only, columns, "SC_Alien", "SC_Human")
    assert [header[i] for i in pursuer] == [f"SC_Human.EarthFixed.{a}" for a in "XYZ"]
    inertial = CHECK.position_columns(header, columns, "SC_Human", "SC_Alien")
    assert [header[i] for i in inertial] == [f"SC_Human.EarthMJ2000Eq.{a}" for a in "XYZ"]
    fixed_km = np.linalg.norm(data[:, pursuer] - data[:, target], axis=1)
    target_inertial = CHECK.position_columns(header, columns, "SC_Alien", "SC_Human")
    inertial_km = np.linalg.norm(data[:, inertial] - data[:, target_inertial], axis=1)
    np.testing.assert_allclose(fixed_km, inertial_km, atol=1e-9)



def test_two_runs_behind_a_first_column_that_is_not_time_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = script_in(tmp_path, PROBES["altitude_first_append_noheader"])
    code, out, err = run(
        monkeypatch, capsys, ROWS / "R1_altitude_first_append_noheader.txt", "--script", script
    )
    assert code == 1 and "VERDICT" not in out
    assert "SC_Alien.ElapsedSecs goes back from 4517.612829 to 0.000000" in err


def test_column_width_must_be_positive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--column-width", "0")
    assert code == 2 and "not a positive integer" in err


def test_a_text_time_column_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = script_in(tmp_path, PROBES["greg_default"])
    code, _, err = run(
        monkeypatch, capsys, ROWS / "R1_greg_default.txt", "--script", script,
        "--time-column", "SC_Human.UTCGregorian",
    )
    assert code == 1 and "time column SC_Human.UTCGregorian is text" in err


def test_without_column_names_column_0_is_not_judged_as_time(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT, no headers, Altitude first. Main passed it; so does --script.
    code, out, err = run(
        monkeypatch, capsys, ROWS / "R1_altitude_first_noheader.txt",
        "--write-headers", "false", "--pursuer-cols", "6,7,8", "--target-cols", "2,3,4",
    )
    # Without names nothing says which column is the pursuer's altitude (§4 note 1(b)).
    assert code == CHECK.SURFACE_UNJUDGED_EXIT and "VERDICT: the LAST row is inside the sphere." in out, err
    assert "0 column names for 9 values a row" in err


def test_values_run_together_under_headers_are_explained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT, ColumnWidth 1 with short names: SC_Alien.X got 13 characters.
    script = script_in(tmp_path, PROBES["short_names_cw1"])
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_short_names_cw1.txt", "--script", script)
    assert code == 1 and "VERDICT" not in out
    assert "values ran together" in err and "max(ColumnWidth, its name) + 3" in err


def test_script_without_the_venv_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "orbitwin.gmat_parser", None)
    script = script_in(tmp_path, {})
    code, _, err = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--script", script)
    assert code == 1 and "--script needs the project's venv" in err


def test_craft_in_no_shared_frame_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Taking each craft's own frame read 9919 km where the distance is 4.95 km.
    lines = (ROWS / "R1_earthfixed.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    header = (
        lines[0]
        .replace("SC_Human.EarthMJ2000Eq.", "SC_Human.Unused.")
        .replace("SC_Alien.EarthFixed.", "SC_Alien.Unused2.")
    )
    report = tmp_path / "report.txt"
    report.write_text("".join([header, *lines[1:]]), encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, report)
    assert code == 1 and "VERDICT" not in out and "cannot locate SC_Human" in err


def test_a_short_name_is_earthmj2000eq_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = (ROWS / "R1_earthfixed.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    header = (
        lines[0]
        .replace("SC_Human.EarthMJ2000Eq.", "SC_Human.")
        .replace("SC_Alien.EarthMJ2000Eq.", "SC_Alien.Unused1.")
        .replace("SC_Human.EarthFixed.", "SC_Human.Unused2.")
    )
    report = tmp_path / "report.txt"
    report.write_text("".join([header, *lines[1:]]), encoding="utf-8")
    # SC_Human.X is inertial and SC_Alien is only in EarthFixed: no shared frame.
    code, _, err = run(monkeypatch, capsys, report)
    assert code == 1 and "cannot locate" in err


def test_the_row_number_counts_frozen_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = script_in(tmp_path, PROBES["append_noheader"])
    report = ROWS / "R1_append_noheader.txt"
    for frozen in ("0", "1"):
        code, _, err = run(
            monkeypatch, capsys, report, "--script", script, "--frozen-rows", frozen
        )
        assert code == 1 and "at data row 8," in err


def test_solver_passes_are_refused_as_more_than_one_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT, SolverIterations = All around a Target loop: each pass restarts at 0.
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_solver_iterations_all.txt")
    assert code == 1 and "VERDICT" not in out and "SolverIterations = All" in err



def test_columns_may_keep_their_braces(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    code, out, err = run(
        monkeypatch, capsys, ROWS / "R1_hit_noheader.txt",
        "--write-headers", "false", "--columns", "{" + ADD_LIST.replace(",", ", ") + "}",
    )
    assert (code, _verdict(out)) == (0, _verdict(default[1]))
    assert "time column   : SC_Alien.ElapsedSecs" in out and "not in seconds" not in err


R6 = ROWS / "R6_team54.txt"
#: Σ|Δv| of R1_hit.script's one burn.
HIT_TOTAL_M_S = 1000.0 * float(np.linalg.norm(
    [0.0607629102585799, -0.04239320559177472, 0.06628738223431527]))
SCORE = ["--kt", "0.003982", "--ct", "6505.65", "--kv", "1.1862", "--cv", "9.0643",
         "--total-dv", "6.996"]



def test_equal_epochs_in_an_alignment_are_one_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # SC_Human waits at 0 through the alignment: equal, never going back. The
    # preliminary rules, where the later craft's clock is not refused.
    code, out, err = run(monkeypatch, capsys, R6, "--time-column", "SC_Human.ElapsedSecs",
                         "--rules", "preliminary")
    assert code == 0 and "VERDICT: the LAST row is inside" in out, err





def test_the_named_time_column_is_the_one_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = script_in(tmp_path, PROBES["append_noheader"])
    code, _, err = run(
        monkeypatch, capsys, ROWS / "R1_append_noheader.txt", "--script", script,
        "--time-column", "SC_Human.ElapsedSecs",
    )
    assert code == 1 and "SC_Human.ElapsedSecs goes back" in err


def test_a_stray_line_after_header_less_rows_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Read with the default format: no header was found, and none may appear later.
    lines = (ROWS / "R1_hit_noheader.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    report = tmp_path / "report.txt"
    report.write_text("".join([*lines[:3], "garbage here\n", *lines[3:]]), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--pursuer-cols", "8,9,10",
                       "--target-cols", "1,2,3")
    assert code == 1 and "line 4 is not numeric" in err


def test_a_missing_report_is_a_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(monkeypatch, capsys, tmp_path / "nothing.txt")
    assert code == 1 and "nothing.txt" in err and "Traceback" not in err


def test_the_shared_frame_wins_over_a_craft_s_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # SC_Human in EarthMJ2000Eq and EarthFixed, SC_Alien only in EarthFixed.
    lines = (ROWS / "R1_earthfixed.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    header = lines[0].replace("SC_Alien.EarthMJ2000Eq.", "SC_Alien.Unused1.")
    report = tmp_path / "report.txt"
    report.write_text("".join([header, *lines[1:]]), encoding="utf-8")
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    code, out, err = run(monkeypatch, capsys, report)
    assert (code, _verdict(out)) == (0, _verdict(default[1])), err




def test_a_report_without_names_is_not_scored(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(
        monkeypatch, capsys, ROWS / "R1_hit_noheader.txt", "--pursuer-cols", "8,9,10",
        "--target-cols", "1,2,3", *SCORE,
    )
    assert code == CHECK.SURFACE_UNJUDGED_EXIT and "VERDICT: the LAST row is inside" in out
    assert "score not computed: no column names" in out and "-> score" not in out





def test_wrong_frozen_rows_are_named_not_the_rejected_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No alignment, two rows skipped: no ElapsedSecs reads 0 on the first row kept.
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--frozen-rows", "2",
                         *SCORE)
    assert code == 0 and "-> score" not in out
    assert "check --frozen-rows" in err and "pass --time-column SC_" not in err



def _write(path: Path, names: list[str], data: np.ndarray) -> Path:
    path.write_text(
        " ".join(names) + "\n"
        + "\n".join(" ".join(repr(float(v)) for v in row) for row in data) + "\n",
        encoding="utf-8",
    )
    return path





def test_either_of_two_equal_elapsed_secs_is_scored(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No alignment: both craft share t = 0, and either column is seconds since it.
    for column in ("SC_Alien.ElapsedSecs", "SC_Human.ElapsedSecs"):
        code, out, _ = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt",
                           "--time-column", column, *SCORE)
        assert code == 0 and "-> score" in out, column


R6_T_TEAM_S = MANIFEST["probes"]["R6_team54"]["t_team_s"]


def test_the_time_column_is_found_by_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT, Altitude first; main read 430.42 km of altitude as t.
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_altitude_first.txt", *SCORE)
    assert (code, _verdict(out)) == (0, _verdict(default[1])), err
    assert "time column   : SC_Alien.ElapsedSecs" in out
    # Found by name it is not scored: only --time-column says it is T_team.
    assert "-> score" not in out and "SC_Alien.ElapsedSecs was found by name" in out


def test_an_aligned_report_needs_the_time_column_named(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Our certified R6: main read Altitude as t and printed Tteam 1095.69 s.
    code, out, err = run(monkeypatch, capsys, R6, *SCORE)
    assert code == 1 and "VERDICT" not in out
    assert "SC_Alien.ElapsedSecs and SC_Human.ElapsedSecs differ" in err
    assert "the craft with the earlier epoch" in err and "clock.time_origin_craft" in err


def test_the_named_origin_gives_the_certified_t_team(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # certification.json: mission_time_origin target, clock.t_team_s 5911.532175.
    code, out, _ = run(monkeypatch, capsys, R6, "--frozen-rows", "6",
                       "--time-column", "SC_Alien.ElapsedSecs", *SCORE)
    assert code == 0 and f"LAST row      : t = {R6_T_TEAM_S:.6f} s" in out
    assert f"Tteam = {R6_T_TEAM_S:11.4f} s -> score" in out
    # SC_Human waits through the alignment: its clock is 1800 s short, and refused.
    code, out, err = run(monkeypatch, capsys, R6, "--frozen-rows", "6",
                         "--time-column", "SC_Human.ElapsedSecs", *SCORE)
    assert code == 1 and "VERDICT" not in out
    assert "SC_Alien.ElapsedSecs runs ahead" in err and "--time-column is SC_Human" in err


def test_one_elapsed_secs_in_an_aligned_report_is_not_scored_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # R6 without SC_Human.ElapsedSecs: nothing in the report says SC_Alien is the origin.
    header, data = CHECK.read_report(str(R6))
    keep = [i for i, name in enumerate(header) if name != "SC_Human.ElapsedSecs"]
    report = _write(tmp_path / "r6.txt", [header[i] for i in keep], data[:, keep])
    code, out, _ = run(monkeypatch, capsys, report, "--frozen-rows", "6", *SCORE)
    assert code == 0 and "VERDICT: the LAST row is inside" in out
    assert "-> score" not in out and "holds SC_Human still" in out
    code, out, _ = run(monkeypatch, capsys, report, "--frozen-rows", "6",
                       "--time-column", "SC_Alien.ElapsedSecs", *SCORE)
    assert f"Tteam = {R6_T_TEAM_S:11.4f} s -> score" in out


def test_a_waiting_craft_reports_its_frozen_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, err = run(monkeypatch, capsys, R6, "--time-column", "SC_Alien.ElapsedSecs")
    assert "SC_Human holds still for 6 rows before it moves" in err and "--frozen-rows is 0" in err
    _, _, err = run(monkeypatch, capsys, R6, "--frozen-rows", "6",
                    "--time-column", "SC_Alien.ElapsedSecs")
    assert "holds still" not in err


def test_an_elapsed_secs_is_preferred_to_a_modjulian(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT, A1ModJulian first; main scored days as seconds.
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_mjd_first.txt", *SCORE)
    assert code == 0 and "time column   : SC_Alien.ElapsedSecs" in out


def test_a_time_column_in_days_is_not_scored(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_mjd_first.txt",
                       "--time-column", "SC_Human.A1ModJulian", *SCORE)
    assert code == 0 and "-> score" not in out
    assert "SC_Human.A1ModJulian is not an ElapsedSecs" in out
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_mjd_first_over.txt",
                       "--time-column", "SC_Human.A1ModJulian")
    assert code == 1 and "(in SC_Human.A1ModJulian) after the entry" in out
    assert "first row     : t = 31353.583762\n" in out


def test_only_a_modjulian_is_the_time_column_but_not_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = (ROWS / "R1_mjd_first.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    report = tmp_path / "report.txt"
    report.write_text("".join([lines[0].replace("ElapsedSecs", "Count"), *lines[1:]]),
                      encoding="utf-8")
    code, out, _ = run(monkeypatch, capsys, report, *SCORE)
    assert code == 0 and "time column   : SC_Human.A1ModJulian" in out and "-> score" not in out


def test_no_epoch_at_all_needs_a_time_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = (ROWS / "R1_altitude_first.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    report = tmp_path / "report.txt"
    report.write_text("".join([lines[0].replace("ElapsedSecs", "Count"), *lines[1:]]),
                      encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, report)
    assert code == 1 and "VERDICT" not in out and "no ElapsedSecs or numeric epoch" in err
    code, out, err = run(monkeypatch, capsys, report, "--time-column", "SC_Human.Earth.Altitude",
                         *SCORE)
    assert code == 0 and "no column is a numeric epoch" in err and "-> score" not in out
    assert "first row     : t = 430.419392\n" in out


def test_frozen_rows_one_too_many_is_not_scored(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No alignment: the default ElapsedSecs must read 0 on the first row kept.
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--frozen-rows", "1",
                       *SCORE)
    assert code == 0 and "-> score" not in out and "check --frozen-rows" in out


def test_a_gregorian_first_column_is_passed_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT, UTCGregorian first in the Add list.
    report, script = bound(tmp_path, "R1_greg_first.txt", PROBES["greg_first"])
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    code, out, err = run(monkeypatch, capsys, report, "--script", script, END_ONLY, *CUT)
    assert (code, _verdict(out)) == (3, _verdict(default[1])), err
    assert "time column   : SC_Alien.ElapsedSecs" in out


@pytest.mark.parametrize("probe", ["a1_utc", "days_a1"])
def test_epochs_of_two_kinds_are_not_an_alignment(
    probe: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT, no ElapsedSecs: A1 and UTC ModJulian, or ElapsedDays and A1, always differ.
    code, out, err = run(monkeypatch, capsys, ROWS / f"R1_{probe}.txt", *SCORE)
    assert code == 0 and "VERDICT: the LAST row is inside" in out, err
    assert "-> score" not in out


def test_a_second_run_in_a_modjulian_only_report_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A second run goes back by about 0.05 days, not seconds.
    lines = (ROWS / "R1_mjd_first.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    report = tmp_path / "twice.txt"
    report.write_text(
        "".join([lines[0].replace("ElapsedSecs", "Count"), *lines[1:], *lines[1:]]),
        encoding="utf-8",
    )
    code, out, err = run(monkeypatch, capsys, report)
    assert code == 1 and "VERDICT" not in out and "SC_Human.A1ModJulian goes back" in err


def test_both_craft_still_on_the_first_rows_is_no_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = (ROWS / "R1_hit_default.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    report = tmp_path / "doubled.txt"
    report.write_text("".join([lines[0], lines[1], lines[1], *lines[2:]]), encoding="utf-8")
    _, _, err = run(monkeypatch, capsys, report, "--time-column", "SC_Alien.ElapsedSecs")
    assert "holds still" not in err


def test_an_unnamed_time_column_has_no_unit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out, _ = run(monkeypatch, capsys, ROWS / "R1_hit_noheader.txt", "--pursuer-cols",
                    "8,9,10", "--target-cols", "1,2,3")
    assert "first row     : t = 0.000000\n" in out


def test_alignment_rows_in_a_rotating_frame_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT, R6 with EarthFixed positions only: on the alignment rows the craft are
    # at two epochs, and the EarthFixed distance there is up to 231 km off.
    import lzma

    script = r6_script_in(tmp_path, PROBES["R6_earthfixed"])
    whole = tmp_path / "R6_earthfixed.txt"
    # The whole flight: a cut alignment is refused before the frame is looked at.
    whole.write_bytes(lzma.decompress((ROWS / "R6_earthfixed.txt.xz").read_bytes()))
    report = uploaded(tmp_path, whole, "ODC_2026_Team54.txt")
    code, out, err = run(monkeypatch, capsys, report, "--script", script, END_ONLY,
                         "--time-column", "SC_Alien.ElapsedSecs")
    assert code == 1 and "VERDICT" not in out and "positions in EarthFixed" in err
    # Skipped, the alignment's rows are not judged; without velocity, the end alone binds.
    code, out, _ = run(monkeypatch, capsys, report, "--script", script, "--frozen-rows", "1800",
                       END_ONLY, "--time-column", "SC_Alien.ElapsedSecs")
    assert code == 3 and f"LAST row      : t = {R6_T_TEAM_S:.6f} s" in out
    assert "no EarthMJ2000Eq position and velocity of SC_Human" in out
    # An EarthFixed-only report: (b) from the true pole, within 0.3 m of GMAT's.
    line = next(line for line in out.splitlines() if line.startswith("surface"))
    assert "SC_Human's EarthFixed position, WGS-84" in line


def r6_script_in(tmp_path: Path, fields: dict[str, str]) -> Path:
    """R6_team54.script, the R6 rehearsal certify submitted, with ReportFile1's fields."""
    lines = []
    for line in (ROWS / "R6_team54.script").read_text(encoding="utf-8").splitlines(keepends=True):
        key = line.split("=", 1)[0].strip().removeprefix("ReportFile1.")
        lines.append(f"ReportFile1.{key} = {fields[key]};\n" if key in fields else line)
    path = tmp_path / "r6.script"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_a_named_column_under_an_alignment_is_noted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out, err = run(monkeypatch, capsys, R6, "--frozen-rows", "6", "--rules", "preliminary",
                      "--time-column", "SC_Human.ElapsedSecs", *SCORE)
    assert "-> score" in out
    assert "only if its craft is the time origin" in err


def test_no_elapsed_secs_points_at_the_certificate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out, _ = run(monkeypatch, capsys, ROWS / "R1_a1_utc.txt", *SCORE)
    assert "T_team is clock.t_team_s in certification.json" in out


def test_a_coarse_precision_is_not_an_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT, Precision 6: the 0.001 s first segment repeats SC_Alien's position, not its epoch.
    report, script = bound(tmp_path, "R1_precision6.txt", PROBES["R1_precision6"])
    code, out, err = run(monkeypatch, capsys, report, "--script", script, END_ONLY, *CUT,
                         "--time-column", "SC_Alien.ElapsedSecs", *SCORE[:-2])
    assert code == 3 and "holds still" not in err and "epoch alignment" not in err
    assert "-> score" in out


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_score_lines_do_not_depend_on_the_format(
    fmt: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # The script gives the total, in m/s under the ranking rules.
    named = ["--time-column", "SC_Alien.ElapsedSecs", *SCORE[:-2]]
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", *named,
                  "--total-dv", repr(HIT_TOTAL_M_S))
    assert "-> score" in default[1]
    report, script = bound(tmp_path, f"R1_hit_{fmt}.txt", FORMATS[fmt])
    bound_run = unwarned(run(monkeypatch, capsys, report, "--script", script, *named, *CUT))
    # What §4 note 1 was judged on names the script, which only one of the two has.
    judged = "  (§4 note 1(a), (c), (d) judged on "
    assert f"{judged}--total-dv;" in default[1] and f"{judged}{script}, " in bound_run[1]
    assert [_unjudged(x) for x in bound_run] == [_unjudged(x) for x in default]


def _unjudged(value: object) -> object:
    if not isinstance(value, str):
        return value
    return "\n".join(line for line in value.splitlines()
                     if not line.startswith("  (§4 note 1(a), (c), (d) judged on "))


def test_two_elapsed_secs_half_a_second_apart_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    header, data = CHECK.read_report(str(ROWS / "R1_hit_default.txt"))
    data[:, header.index("SC_Human.ElapsedSecs")] += 0.5
    code, _, err = run(monkeypatch, capsys, _write(tmp_path / "r.txt", header, data))
    assert code == 1 and "differ, as after an epoch alignment" in err


def test_two_modjulians_that_differ_ask_for_a_time_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    header, data = CHECK.read_report(str(ROWS / "R1_mjd_first.txt"))
    names = [name.replace("ElapsedSecs", "Count") for name in header]
    extra = np.column_stack([data, data[:, 0] + 0.01])
    code, _, err = run(monkeypatch, capsys, _write(tmp_path / "r.txt",
                                                   [*names, "SC_Alien.A1ModJulian"], extra))
    assert code == 1 and "--time-column" in err and "mission_time_origin" not in err


def test_more_names_than_values_are_not_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = (ROWS / "R1_hit_default.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    report = tmp_path / "r.txt"
    report.write_text("".join([lines[0].rstrip() + " Extra\n", *lines[1:]]), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report)
    assert code == 1 and "cannot locate" in err


def test_report_lines_that_differ_in_width_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = (ROWS / "R1_hit_default.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    report = tmp_path / "r.txt"
    report.write_text("".join([*lines[:3], lines[3].rstrip() + " 1.0\n", *lines[4:]]),
                      encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report)
    assert code == 1 and "rows have differing column counts [14, 15]" in err


def test_report_file_needs_a_script(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--report-file", "R")
    assert code == 2 and "--report-file names a ReportFile in --script" in err


def test_column_width_from_the_flag_explains_a_fused_row(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = ROOT / "tests" / "fixtures" / "gmat_report_formats" / "O_noheader_narrow.txt"
    names = ("HumanSC.A1ModJulian,HumanSC.EarthMJ2000Eq.X,HumanSC.EarthMJ2000Eq.Y,"
             "HumanSC.EarthMJ2000Eq.Z,AlienSC.EarthMJ2000Eq.X,AlienSC.EarthMJ2000Eq.Y,"
             "AlienSC.EarthMJ2000Eq.Z")
    code, _, err = run(monkeypatch, capsys, report, "--write-headers", "false",
                       "--column-width", "10", "--columns", names)
    assert code == 1 and "ColumnWidth = 10" in err


def test_frozen_rows_must_leave_a_row(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = ROWS / "R1_hit_default.txt"
    rows = len(report.read_text(encoding="utf-8").splitlines()) - 1
    code, _, err = run(monkeypatch, capsys, report, "--frozen-rows", str(rows))
    assert code == 1 and "leaves no row" in err


def test_shifted_names_are_not_used_for_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # One name too many in front: read by name, SC_Alien.ElapsedSecs would be an X column.
    lines = (ROWS / "R1_hit_default.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    report = tmp_path / "r.txt"
    report.write_text("".join(["Extra " + lines[0], *lines[1:]]), encoding="utf-8")
    code, out, _ = run(monkeypatch, capsys, report, "--pursuer-cols", "8,9,10",
                       "--target-cols", "1,2,3")
    assert code == CHECK.SURFACE_UNJUDGED_EXIT and "first row     : t = 0.000000\n" in out


def test_a_header_less_aligned_report_gets_its_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Our certified R6 with its header line dropped; main gave the verdict here.
    report = tmp_path / "r6.txt"
    report.write_text("".join(R6.read_text(encoding="utf-8").splitlines(keepends=True)[1:]),
                      encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, report, "--write-headers", "false",
                         "--pursuer-cols", "9,10,11", "--target-cols", "2,3,4",
                         "--frozen-rows", "6")
    assert code == CHECK.SURFACE_UNJUDGED_EXIT and "VERDICT: the LAST row is inside" in out, err


def test_a_craft_without_an_epoch_needs_more_than_one_still_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # R1 at Precision 6 without SC_Alien.ElapsedSecs: one repeated row is no alignment.
    header, data = CHECK.read_report(str(ROWS / "R1_precision6.txt"))
    keep = [i for i, name in enumerate(header) if name != "SC_Alien.ElapsedSecs"]
    report = _write(tmp_path / "p6.txt", [header[i] for i in keep], data[:, keep])
    # One clock named by hand: the ranking rules need the day's proof (#181), not tested here.
    code, out, err = run(monkeypatch, capsys, report, "--time-column", "SC_Human.ElapsedSecs",
                         "--rules", "preliminary", *SCORE)
    assert code == 0 and "holds still" not in err and "epoch alignment" not in err
    assert "-> score" in out


@pytest.mark.parametrize("named", [True, False])
def test_a_nan_is_refused_as_not_finite(
    named: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A NaN time or position fails every comparison, so no row could be inside.
    lines = (ROWS / "R1_hit_default.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    parts = lines[3].split()
    parts[0] = "nan"
    rows = [*lines[1:3], " ".join(parts) + "\n", *lines[4:]]
    report = tmp_path / "nan.txt"
    report.write_text("".join([lines[0], *rows] if named else rows), encoding="utf-8")
    flags = [] if named else ["--pursuer-cols", "8,9,10", "--target-cols", "1,2,3"]
    code, out, err = run(monkeypatch, capsys, report, *flags)
    assert code == 1 and "VERDICT" not in out and "Traceback" not in err
    # Named, the time candidates are read first, and the column is named.
    assert ("data row 3 holds a SC_Alien.ElapsedSecs that is not finite" if named
            else "data row 3 holds a time or position that is not finite") in err


def test_a_nan_position_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = (ROWS / "R1_hit_default.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    parts = lines[-1].split()
    parts[8] = "nan"
    report = tmp_path / "nan.txt"
    report.write_text("".join([*lines[:-1], " ".join(parts) + "\n"]), encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, report)
    assert code == 1 and "VERDICT" not in out and "not finite" in err


def test_a_report_without_its_alignment_rows_is_not_scored_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A Toggle around the alignment leaves no held rows, and only the later craft's
    # ElapsedSecs: scored, R6 gave 4111.53 s where T_team is 5911.53 s.
    header, data = CHECK.read_report(str(R6))
    keep = [i for i, name in enumerate(header) if name != "SC_Alien.ElapsedSecs"]
    report = _write(tmp_path / "toggled.txt", [header[i] for i in keep], data[6:, keep])
    code, out, err = run(monkeypatch, capsys, report, *SCORE)
    assert code == 0 and "VERDICT: the LAST row is inside" in out
    assert "-> score" not in out and "was found by name" in out
    assert "holds still" not in err


def test_two_modjulians_apart_only_on_alignment_rows_ask_for_frozen_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # R6 with each craft's ElapsedSecs written as a UTCModJulian-like absolute epoch:
    # the craft are one instant apart only while SC_Human waits.
    header, data = CHECK.read_report(str(R6))
    start = 31353.5625
    for craft, offset in (("SC_Alien", 0.0), ("SC_Human", 1800.0)):
        index = header.index(f"{craft}.ElapsedSecs")
        header[index] = f"{craft}.UTCModJulian"
        data[:, index] = start + (data[:, index] + offset) / 86400.0
    report = _write(tmp_path / "mjd.txt", header, data)
    code, _, err = run(monkeypatch, capsys, report)
    assert code == 1 and "pass --frozen-rows from certification.json" in err
    code, out, _ = run(monkeypatch, capsys, report, "--frozen-rows", "6")
    assert code == 0 and "VERDICT: the LAST row is inside" in out


def test_modjulians_are_compared_in_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1e-7 day is 8.6 ms: two instants, though under 1e-6 as a number of days.
    header, data = CHECK.read_report(str(ROWS / "R1_mjd_first.txt"))
    names = [name.replace("ElapsedSecs", "Count") for name in header]
    extra = np.column_stack([data, data[:, 0] + 1.0e-7])
    code, _, err = run(monkeypatch, capsys, _write(tmp_path / "r.txt",
                                                   [*names, "SC_Alien.A1ModJulian"], extra))
    assert code == 1 and "differ" in err


def test_two_gmat_modjulians_are_one_instant_once_frozen_rows_are_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT, R6 reporting both craft's UTCModJulian and no ElapsedSecs.
    report = ROWS / "R6_two_modjulians.txt"
    code, out, err = run(monkeypatch, capsys, report)
    assert code == 1 and "VERDICT" not in out and "pass --frozen-rows" in err
    code, out, _ = run(monkeypatch, capsys, report, "--frozen-rows", "6", *SCORE)
    assert code == 0 and "VERDICT: the LAST row is inside" in out and "-> score" not in out


def test_a_time_column_past_the_row_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(
        monkeypatch, capsys, ROWS / "R1_hit_noheader.txt", "--write-headers", "false",
        "--columns", ADD_LIST + ",Extra.ElapsedSecs", "--time-column", "Extra.ElapsedSecs",
        "--pursuer-cols", "8,9,10", "--target-cols", "1,2,3",
    )
    assert code == 1 and "but a row has 14 values" in err and "Traceback" not in err


def test_the_report_file_is_chosen_by_its_file_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = script_in(tmp_path, {})
    text = script.read_text(encoding="utf-8")
    script.write_text(
        text.replace(
            "BeginMissionSequence;",
            "Create ReportFile Extra;\nExtra.Filename = 'extra.txt';\nExtra.Delimiter = ',';\n"
            "BeginMissionSequence;",
        ),
        encoding="utf-8",
    )
    report = tmp_path / "extra.txt"
    report.write_bytes((ROWS / "R1_hit_comma.txt").read_bytes())
    # Extra is read, and refused: ReportFile1 is the report uploaded.
    code, _, err = run(monkeypatch, capsys, report, "--script", script)
    assert code == 2 and "Extra is not" in err and "official report, ReportFile1" in err
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    assert run(monkeypatch, capsys, report, "--delimiter", ",") == default


def test_no_names_and_no_headers_on_a_headed_file_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt",
                         "--write-headers", "false", "--pursuer-cols", "8,9,10",
                         "--target-cols", "1,2,3")
    assert code == 1 and "VERDICT" not in out and "line 1 is not numeric" in err


def test_a_blank_line_in_a_comma_report_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = (ROWS / "R1_hit_noheader_comma_nofixed.txt").read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    report = tmp_path / "blank.txt"
    report.write_text("".join([*rows[:2], "\n", *rows[2:]]), encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, report, *FLAGS["noheader_comma_nofixed"])
    assert code == 0 and "VERDICT: the LAST row is inside" in out, err


def test_elapsed_days_alone_ask_for_a_time_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    header, data = CHECK.read_report(str(ROWS / "R1_hit_default.txt"))
    for craft, shift in (("SC_Alien", 0.0), ("SC_Human", 0.5)):
        index = header.index(f"{craft}.ElapsedSecs")
        header[index] = f"{craft}.ElapsedDays"
        data[:, index] = (data[:, index] + shift) / 86400.0
    code, _, err = run(monkeypatch, capsys, _write(tmp_path / "days.txt", header, data))
    assert code == 1 and "pass --time-column" in err and "--frozen-rows" not in err


def test_a_craft_held_for_one_row_is_an_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # SC_Human's position and epoch repeat on row 2 while SC_Alien moves.
    header, data = CHECK.read_report(str(ROWS / "R1_hit_default.txt"))
    human = [i for i, name in enumerate(header) if name.startswith("SC_Human.")]
    data[1, human] = data[0, human]
    report = _write(tmp_path / "held.txt", header, data)
    _, _, err = run(monkeypatch, capsys, report, "--time-column", "SC_Alien.ElapsedSecs")
    assert "SC_Human holds still for 1 rows" in err


def test_a_few_last_digits_between_two_modjulians_are_one_instant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The last digit at Precision 16 is 1e-11 day, 0.9 us; ten of them are 8.6 us.
    header, data = CHECK.read_report(str(ROWS / "R1_mjd_first.txt"))
    names = [name.replace("ElapsedSecs", "Count") for name in header]
    extra = np.column_stack([data, data[:, 0] + 1.0e-10])
    code, out, err = run(monkeypatch, capsys, _write(tmp_path / "r.txt",
                                                     [*names, "SC_Alien.A1ModJulian"], extra))
    assert code == 0 and "VERDICT: the LAST row is inside" in out, err


def test_short_names_on_kept_alignment_rows_are_inertial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # SC.X is EarthMJ2000Eq (R2026a), so the alignment's rows may stay.
    lines = R6.read_text(encoding="utf-8").splitlines(keepends=True)
    report = tmp_path / "short.txt"
    report.write_text("".join([lines[0].replace(".EarthMJ2000Eq.", "."), *lines[1:]]),
                      encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, report, "--time-column", "SC_Alien.ElapsedSecs")
    assert code == 0 and "VERDICT: the LAST row is inside" in out, err


@pytest.mark.parametrize(
    "craft, renamed, flags, time",
    [
        ("SC_Human", "HumanSC", ["--pursuer-cols", "8,9,10"], "SC_Alien.ElapsedSecs"),
        ("SC_Alien", "AlienSC", ["--target-cols", "1,2,3"], "SC_Human.ElapsedSecs"),
    ],
)
def test_a_craft_by_flag_lets_the_other_be_found_alone(
    craft: str, renamed: str, flags: list[str], time: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # The organizer's example names its pursuer HumanSC; main gave the verdict here.
    lines = (ROWS / "R1_hit_default.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    report = tmp_path / "renamed.txt"
    report.write_text("".join([lines[0].replace(craft, renamed), *lines[1:]]), encoding="utf-8")
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    # One clock named by hand: the ranking rules need the day's proof (#181), not tested here.
    code, out, err = run(monkeypatch, capsys, report, *flags, "--time-column", time,
                         "--rules", "preliminary")
    assert (code, _verdict(out)) == (0, _verdict(default[1])), err


def test_columns_name_a_report_without_headers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_hit_noheader.txt", "--columns", ADD_LIST)
    assert (code, _verdict(out)) == (0, _verdict(default[1])), err


def test_an_add_list_one_short_of_a_text_row_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fields = dict(PROBES["greg_noheader_nofixed"])
    fields["Add"] = fields["Add"].replace(", SC_Alien.A1Gregorian", "")
    script = script_in(tmp_path, fields)
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_greg_noheader_nofixed.txt",
                         "--script", script)
    assert code == 1 and "VERDICT" not in out and "is not numeric" in err


def test_a_flagged_craft_s_frame_binds_the_other(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT: SC_Alien in EarthFixed only, SC_Human in both. Pairing SC_Human's
    # EarthMJ2000Eq with SC_Alien's EarthFixed read a hit as a 9925 km miss.
    report = ROWS / "R1_mixed_frames.txt"
    code, out, err = run(monkeypatch, capsys, report, "--pursuer-cols", "7,8,9",
                         "--time-column", "SC_Alien.ElapsedSecs")
    assert code == 1 and "VERDICT" not in out and "cannot locate SC_Alien" in err
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    code, out, err = run(monkeypatch, capsys, report, "--pursuer-cols", "4,5,6",
                         "--time-column", "SC_Alien.ElapsedSecs", "--rules", "preliminary")
    assert (code, _verdict(out)) == (0, _verdict(default[1])), err


def test_columns_place_a_gregorian_row_without_write_headers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Taken as a header, the first row was lost and --frozen-rows shifted by one.
    names = ",".join(n.strip() for n in PROBES["greg_noheader_nofixed"]["Add"].strip("{}").split(","))
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_greg_noheader_nofixed.txt",
                         "--fixed-width", "false", "--columns", names)
    assert (code, _verdict(out)) == (0, _verdict(default[1])), err
    assert "rows          : 7" in out


def test_only_position_names_give_a_flagged_frame() -> None:
    names = ["A.EarthFixed.VX", "A.EarthFixed.VY", "A.EarthFixed.VZ", "A.X", "A.Y", "A.Z"]
    assert CHECK.flagged_frame(names, 6, [0, 1, 2]) is None
    assert CHECK.flagged_frame(names, 6, [3, 4, 5]) == "EarthMJ2000Eq"


def test_negative_flags_are_the_same_columns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # They skipped the frame rule and read a hit as a 9925 km miss.
    report = ROWS / "R1_mixed_frames.txt"
    # One clock named by hand: the ranking rules need the day's proof (#181), not tested here.
    plain = run(monkeypatch, capsys, report, "--target-cols", "1,2,3",
                "--time-column", "SC_Alien.ElapsedSecs", "--rules", "preliminary")
    negative = run(monkeypatch, capsys, report, "--target-cols=-9,-8,-7",
                   "--time-column", "SC_Alien.ElapsedSecs", "--rules", "preliminary")
    assert plain[0] == 0 and _verdict(negative[1]) == _verdict(plain[1])


def test_a_flag_past_the_row_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--target-cols", "1,2,30")
    assert code == 1 and "fall outside the 14 values" in err and "Traceback" not in err


@pytest.mark.parametrize(
    "flags, why",
    [
        (["--target-cols", "8,9,10"], "the same columns"),
        (["--target-cols", "1,2,9"], "not one craft's X, Y and Z"),
    ],
)
def test_a_flag_must_name_one_other_craft_s_position(
    flags: list[str], why: str, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 8,9,10 are SC_Human's own: paired with itself it read every row as a hit.
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", *flags)
    assert code == 1 and "VERDICT" not in out and why in err


def test_two_crafts_on_the_same_columns_are_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_hit_noheader.txt", "--pursuer-cols",
                         "1,2,3", "--target-cols", "1,2,3")
    assert code == 1 and "VERDICT" not in out and "the same columns" in err


def test_both_flags_on_one_craft_are_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT, a miss with craft named HumanSC and AlienSC and AlienSC.X/Y/Z added:
    # 14-16 and 1-3 are both AlienSC, and main read a 1140 km miss as a hit.
    report = ROWS / "R1_renamed_miss.txt"
    code, out, err = run(monkeypatch, capsys, report, "--time-column", "AlienSC.ElapsedSecs",
                         "--pursuer-cols", "14,15,16", "--target-cols", "1,2,3")
    assert code == 1 and "VERDICT" not in out and "both craft's columns are AlienSC's" in err
    code, out, _ = run(monkeypatch, capsys, report, "--time-column", "AlienSC.ElapsedSecs",
                       "--pursuer", "HumanSC", "--target", "AlienSC")
    assert code == 1 and "this report is a miss" in out and "VERDICT: unchecked" in out


def test_both_flags_in_two_frames_are_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # GMAT: SC_Human's EarthMJ2000Eq against SC_Alien's EarthFixed read a hit as a 9925 km miss.
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_mixed_frames.txt",
                         "--pursuer-cols", "7,8,9", "--target-cols", "1,2,3",
                         "--time-column", "SC_Alien.ElapsedSecs")
    assert code == 1 and "VERDICT" not in out and "EarthMJ2000Eq and EarthFixed" in err


def test_a_flag_of_two_craft_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # 8 and 10 are SC_Human's X and Z, 2 is SC_Alien's Y.
    code, _, err = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--pursuer-cols", "8,2,10")
    assert code == 1 and "not one craft's X, Y and Z" in err


def test_the_row_width_is_past_the_last_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(monkeypatch, capsys, ROWS / "R1_miss_noheader.txt", "--write-headers",
                       "false", "--pursuer-cols", "14,9,10", "--target-cols", "1,2,3")
    assert code == 1 and "fall outside the 14 values" in err


@pytest.mark.parametrize("flag", ["1,2,3,", "a,b,c"])
def test_a_malformed_flag_is_a_message(
    flag: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--target-cols", flag)
    assert code == 1 and "column flags take three integers" in err and "Traceback" not in err


def test_swapped_flags_give_the_same_distance(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A distance is symmetric; main read this, and so must the check.
    default = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt",
                         "--pursuer-cols", "1,2,3", "--target-cols", "8,9,10")
    assert (code, _verdict(out)) == (0, _verdict(default[1])), err


def test_two_elapsed_secs_a_tenth_of_a_millisecond_apart_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    header, data = CHECK.read_report(str(ROWS / "R1_hit_default.txt"))
    data[:, header.index("SC_Human.ElapsedSecs")] += 1.0e-4
    code, _, err = run(monkeypatch, capsys, _write(tmp_path / "r.txt", header, data))
    assert code == 1 and "differ, as after an epoch alignment" in err


def test_frozen_rows_with_nothing_held_draw_a_note(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A Toggle around the alignment, say: the rows skipped are the mission's own.
    _, _, err = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--frozen-rows", "1",
                    "--time-column", "SC_Alien.ElapsedSecs")
    assert "but no craft holds still" in err


#: R1's T_max; the GMAT miss fixtures end at 6711.667000001762.
R1_HORIZON = "6711.667"


@pytest.mark.parametrize("fmt", sorted(FLAGS))
def test_a_ranking_miss_run_to_t_max_passes(
    fmt: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(monkeypatch, capsys, ROWS / f"R1_miss_{fmt}.txt", *FLAGS[fmt],
                       "--time-column", "SC_Human.ElapsedSecs", "--horizon-s", R1_HORIZON)
    assert code == 0, out
    assert "VERDICT: the LAST row is at T_max = 6711.667000 s: a scored miss" in out


def test_a_miss_is_unchecked_without_the_horizon(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt",
                       "--time-column", "SC_Human.ElapsedSecs")
    assert code == 1 and "VERDICT: unchecked. Pass --horizon-s" in out


@pytest.mark.parametrize(
    ("report", "horizon", "off"),
    [
        ("R1_miss_default.txt", "6800", "-88.333000"),  # run short of T_max
        ("R1_renamed_miss.txt", R1_HORIZON, "-2194.054171"),  # stopped where a hit would
    ],
)
def test_a_miss_that_does_not_end_at_t_max_fails(
    report: str, horizon: str, off: str,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    extra = ["--pursuer", "HumanSC", "--target", "AlienSC", "--time-column",
             "AlienSC.ElapsedSecs"] if "renamed" in report else ["--time-column",
                                                                 "SC_Human.ElapsedSecs"]
    code, out, _ = run(monkeypatch, capsys, ROWS / report, *extra, "--horizon-s", horizon)
    assert code == 1 and "A miss must run to T_max" in out and off in out, out


def test_a_preliminary_miss_is_still_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The preliminary submits interceptions only (constraints.interception_required)."""
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--rules",
                       "preliminary", "--time-column", "SC_Human.ElapsedSecs",
                       "--horizon-s", R1_HORIZON)
    assert code == 1
    assert "VERDICT: no row is inside the sphere. This report does not show an interception." in out
    assert "scored miss" not in out


@pytest.mark.parametrize("flag", ["--horizon-s", "--radius-km"])
@pytest.mark.parametrize("value", ["nan", "inf", "0", "-5"])
def test_a_meaningless_horizon_or_radius_is_refused(
    flag: str, value: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # nan passed any miss: abs(nan) > tol is False.
    code, _, _ = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--time-column",
                     "SC_Human.ElapsedSecs", flag, value)
    assert code == 2


def test_a_radius_override_does_not_turn_a_hit_into_a_scored_miss(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_over_default.txt", "--time-column",
                       "SC_Human.ElapsedSecs", "--radius-km", "0.1", "--horizon-s", R1_HORIZON)
    assert code == 1 and "only --radius-km makes this a miss" in out


def test_frozen_rows_that_skip_the_entry_leave_a_miss_unchecked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_over_default.txt", "--time-column",
                       "SC_Human.ElapsedSecs", "--frozen-rows", "30", "--horizon-s", R1_HORIZON)
    assert code == 1 and "VERDICT: unchecked. --frozen-rows does not match" in out


def test_an_entry_after_t_max_is_not_a_hit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--time-column",
                       "SC_Human.ElapsedSecs", "--horizon-s", "4000")
    assert code == 1 and "after T_max = 4000.000000 s" in out
    code, _, _ = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--time-column",
                     "SC_Human.ElapsedSecs", "--horizon-s", R1_HORIZON)
    assert code == 0


@pytest.mark.parametrize(
    ("report", "horizon"), [("R1_a1_utc.txt", R1_HORIZON), ("R1_days_a1.txt", "4000")]
)
def test_the_entry_is_held_to_t_max_only_on_the_mission_clock(
    report: str, horizon: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ModJulian column read 31353.6 against 6711.667 s, and days never reached 4000."""
    code, _, err = run(monkeypatch, capsys, ROWS / report, "--horizon-s", horizon)
    assert code == 0, err
    assert "the entry was not checked against T_max" in err


def test_the_horizon_tolerance_is_the_last_digit_precision_prints() -> None:
    # Precision 6 writes 6711.67 for 6711.667: certify accepts it, so must this.
    assert CHECK._horizon_tolerance(6711.667, 6) == pytest.approx(0.01)
    assert CHECK._horizon_tolerance(6711.667, 16) == CHECK.HORIZON_TOLERANCE_S
    assert CHECK._horizon_tolerance(6711.667, None) == CHECK.HORIZON_TOLERANCE_S


#: The coefficients by hand, as the 2026/11/04 revision gives them: kv per m/s, Cv in m/s.
SCORE_FLAGS = ["--kt", "0.003", "--ct", "5369.334", "--kv", "0.0012", "--cv", "2000"]


# R1_miss passes 14046 km off: worth points only at a C_r of that order.
@pytest.mark.parametrize("cr_km", [3000.0, 10000.0, 30000.0])
def test_a_scored_miss_is_priced_on_its_closest_row(
    cr_km: float, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import math

    report = ROWS / "R1_miss_default.txt"
    code, out, _ = run(monkeypatch, capsys, report, "--time-column", "SC_Human.ElapsedSecs",
                       "--horizon-s", R1_HORIZON, *SCORE_FLAGS, "--cr-km", str(cr_km),
                       "--total-dv", "0")
    assert code == 0, out
    closest = float(out.split("closest row")[1].split("dr =")[1].split("km")[0])
    distance = 100.0 / (1.0 + math.exp(((closest - 5.0) / cr_km) ** 2))
    expected = (distance
                + 25.0 / (1.0 + math.exp(0.003 * (6711.667 - 5369.334)))
                + 25.0 / (1.0 + math.exp(0.0012 * (0.0 - 2000.0))))
    assert f"distance {distance:.4f} +" in out and f"= {expected:.4f}" in out, out


def test_a_scored_miss_is_not_priced_without_c_r(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--time-column",
                       "SC_Human.ElapsedSecs", "--horizon-s", R1_HORIZON, *SCORE_FLAGS,
                       "--total-dv", "0")
    assert code == 0, out
    assert "not priced: no C_r" in out and "scored miss:" not in out


def test_c_r_is_refused_under_the_preliminary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--rules",
                       "preliminary", "--cr-km", "300")
    assert code == 2 and "--cr-km is the ranking rules' C_r" in err


def test_a_typed_c_r_that_prices_nothing_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--time-column",
                       "SC_Human.ElapsedSecs", "--cr-km", "300")
    assert code == 0, out
    assert "C_r not used: a hit's distance term is 50 at any C_r" in out
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--time-column",
                       "SC_Human.ElapsedSecs", "--horizon-s", R1_HORIZON, "--cr-km", "300")
    assert code == 0, out
    assert "C_r not used: the miss was not priced" in out
    # R1_hit enters at 4518 s: with T_max 4000 s it is neither a hit nor a priced miss.
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--time-column",
                       "SC_Human.ElapsedSecs", "--horizon-s", "4000", "--cr-km", "300")
    assert code == 1 and "after T_max = 4000.000000 s" in out
    assert "C_r not used: nothing was priced" in out
    # Priced, it is used, and says nothing.
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--time-column",
                       "SC_Human.ElapsedSecs", "--horizon-s", R1_HORIZON, *SCORE_FLAGS,
                       "--cr-km", "30000", "--total-dv", "0")
    assert code == 0 and "scored miss:" in out and "C_r not used" not in out


def test_c_r_is_only_taken_with_its_unit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--cr", "300")
    assert code == 2 and "unrecognized arguments: --cr" in err


@pytest.mark.parametrize("cr_km", ["0", "-300", "nan", "inf"])
def test_c_r_must_be_a_positive_number(
    cr_km: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--cr-km", cr_km)
    assert code == 2 and "--cr-km" in err


def _day_files(tmp_path: Path, report: str = "R1_hit_default.txt", *, unit: str = "m/s",
               cr_km: float | None = 300.0,
               origin: str | None = "earliest", window: str | None = None,
               t_team: float | None = None,
               total_km_s: float = 0.0994, script_text: str | None = None,
               rules: str = "ranking", drop: str | None = None,
               config_rules: str = "ranking", rule_sha256: str | None = None,
               stop_depth_km: float = 0.05, allow_shallow: bool | None = None,
               mission_end_policy: str | None = None, origin_craft: str | None = None,
               frozen_rows: int = 0, horizon_s: float = 6711.667,
               offset_s: float = 0.0,
               certified_score: dict[str, object] | None = None) -> tuple[Path, Path, Path]:
    """A day config as ranking_day_config writes it, and certify's submission folder:
    the report under its submitted name, the script, and certification.json.

    R1_hit was stopped 50 m inside, the stop depth of its day, hence the default.
    """
    import hashlib

    import yaml

    from orbitwin.rules import RANKING, interception_rules, rule_set_record

    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["rules"] = config_rules
    if interception_rules(config_rules) is not RANKING:
        raw["search"]["ranking_search"] = False
    raw["policy"]["horizon_s"] = horizon_s
    raw["policy"]["stop_depth_km"] = stop_depth_km
    if allow_shallow is not None:
        raw["policy"]["allow_shallow_stop_submission"] = allow_shallow
    if mission_end_policy is not None:
        raw["policy"]["mission_end_policy"] = mission_end_policy
    if origin is not None:
        raw["policy"]["mission_time_origin"] = origin
    if window is not None:
        raw["policy"]["closest_approach_window"] = window
    if origin == "earliest" and origin_craft is None:
        origin_craft = "SC_Human"
    scale = 1000.0 if unit == "m/s" else 1.0
    raw["score"] = {"kt": 0.003, "ct": 5369.334, "kv": 1.2 / scale, "cv": 2.0 * scale,
                    "time_unit": "s", "delta_v_unit": unit}
    if cr_km is not None:
        raw["score"]["cr_km"] = cr_km
    config = tmp_path / "day.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    folder = tmp_path / "submission"
    folder.mkdir()
    uploaded = folder / "ODC_2026_Team54.txt"
    uploaded.write_bytes((ROWS / report).read_bytes())
    script = folder / "public_gmat_candidate.script"
    script.write_text(script_text or (ROWS / "R1_hit.script").read_text(encoding="utf-8"),
                      encoding="utf-8")
    body = {
        "submission_script_name": script.name,
        "submission_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (uploaded, script)},
        "rule_set": {**rule_set_record(rules),
                     **({"sha256": rule_sha256} if rule_sha256 is not None else {})},
        "clock": {"mission_time_origin": origin, "time_origin_craft": origin_craft,
                  "alignment_offset_s": offset_s,
                  "t_team_s": _last_time(report) if t_team is None else t_team,
                  "time_origin_offset_s": offset_s},
        "candidate": {"total_dv_km_s": total_km_s,
                      "metadata": {"horizon_s": horizon_s - offset_s}},
        "frozen_rows": frozen_rows,
        "exact_runs": [{"script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}] * 3,
        # As certify records the config's score block, with any field overridden.
        "score_block": {**{"cr_km": None, **raw["score"]}, **(certified_score or {})},
    }
    if drop is not None:
        outer, _, inner = drop.partition(".")
        if inner:
            del body[outer][inner]
        else:
            del body[outer]
    certificate = folder / "certification.json"
    certificate.write_text(json.dumps(body), encoding="utf-8")
    return config, certificate, uploaded


def _last_time(name: str) -> float:
    return float((ROWS / name).read_text(encoding="utf-8").split()[-14])


def test_a_preliminary_day_reads_the_certified_total_in_km_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Cv in km/s, as the preliminary's was: the certificate's km/s total is not scaled."""
    config, certificate, report = _day_files(tmp_path, unit="km/s", cr_km=None,
                                             rules="preliminary", config_rules="preliminary")
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate, "--rules", "preliminary")
    assert code == 0, out
    by_hand = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--rules", "preliminary",
                  "--time-column", "SC_Human.ElapsedSecs", "--kt", "0.003", "--ct", "5369.334",
                  "--kv", "1.2", "--cv", "2.0", "--total-dv", "0.0994")[1]
    assert out.split("validator reads last row")[1].split("\n")[0] in by_hand


# --- §4 note 1 (review of the impact inventory) -----------------------------------------


def _burns(*plan: tuple[float, float]) -> list[dict[str, object]]:
    """Burns as a certificate's candidate holds them: (t s, |Δv| km/s), along x."""
    return [{"time_s": t, "dv_inertial_km_s": [size, 0.0, 0.0],
             "dv_script_km_s": [size, 0.0, 0.0]} for t, size in plan]


def _judged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
            burns: list[dict[str, object]], total_km_s: float = 0.0994,
            flags: tuple[str, ...] = (), **day: object) -> tuple[int, str, str]:
    config, certificate, report = _day_files(tmp_path, total_km_s=total_km_s, **day)  # type: ignore[arg-type]
    body = json.loads(certificate.read_text(encoding="utf-8"))
    body["candidate"]["burns"] = burns
    certificate.write_text(json.dumps(body), encoding="utf-8")
    return run(monkeypatch, capsys, report, "--config", config, "--certificate", certificate,
               *flags)


def _scores(out: str) -> list[float]:
    return [float(line.rsplit(maxsplit=1)[-1]) for line in out.splitlines() if "-> score" in line]


def test_an_impulse_over_the_limit_disqualifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The probe: a certificate with one 0.26 km/s burn passed and printed 96.2676."""
    code, out, _ = _judged(tmp_path, monkeypatch, capsys, _burns((0.001, 0.26)))
    assert code == 2 and "-> score" not in out, out
    assert ("DISQUALIFIED: the certificate's candidate: burn 1 at t = 0.001 s is 260.000000 m/s, "
            "over the ranking rules' 250 m/s per impulse plus ε_v (§4 note 1(a))") in out


def test_a_total_over_the_cap_disqualifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The probe: 3.05 km/s in all passed and printed 79.5487."""
    burns = _burns(*((100.0 * k, 0.2345) for k in range(13)))
    code, out, _ = _judged(tmp_path, monkeypatch, capsys, burns, total_km_s=3.0485)
    assert code == 2 and "-> score" not in out, out
    assert "Σ|Δv| is 3.0485 km/s, over the ranking rules' 3 km/s cap" in out
    assert "(§4 note 1(c))" in out


def test_a_recorded_total_over_the_cap_disqualifies_without_coefficients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Step 4a's unscored config fills no --total-dv: the certificate's own total is judged."""
    import yaml

    config, certificate, report = _day_files(tmp_path, total_km_s=3.05)
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    del raw["score"]
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2, out
    assert "DISQUALIFIED: the certificate's total_dv_km_s: Σ|Δv| is 3.05 km/s" in out


@pytest.mark.parametrize(("size_km_s", "code"), [(0.250000005, 0), (0.250000015, 2)])
def test_an_impulse_is_judged_with_the_rules_epsilon(
    size_km_s: float, code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§4 note 1(a): over 250 m/s plus ε_v = 1e-5 m/s, not over 250 m/s."""
    got, out, _ = _judged(tmp_path, monkeypatch, capsys, _burns((0.001, size_km_s)))
    assert got == code, out


def test_a_hand_total_over_the_cap_disqualifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The probe: --total-dv 3050 by hand passed."""
    config, _, report = _day_files(tmp_path)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config, "--time-column",
                       "SC_Human.ElapsedSecs", "--frozen-rows", "0", "--total-dv", "3050")
    assert code == 2 and "DISQUALIFIED: --total-dv: Σ|Δv| is 3.05 km/s" in out, out


def test_burns_closer_than_100_s_cost_ten_points_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The probe: two burns 99 s apart passed at 96.3880, with no -10."""
    (tmp_path / "one").mkdir()
    code, out, _ = _judged(tmp_path / "one", monkeypatch, capsys, _burns((0.001, 0.0994)))
    assert code == 0 and "WARNING" not in out, out
    alone = _scores(out)
    (tmp_path / "two").mkdir()
    code, out, _ = _judged(tmp_path / "two", monkeypatch, capsys,
                           _burns((0.001, 0.05), (99.001, 0.0494)))
    assert code == 0, out
    assert "burn 2 at t = 99.001 s is 99.000 s after the one before, under the 100 s" in out
    assert "WARNING: 1 violating maneuver: -10 points in the score printed below" in out
    assert _scores(out) == pytest.approx([score - 10.0 for score in alone], abs=1e-4)
    (tmp_path / "three").mkdir()
    _, out, _ = _judged(tmp_path / "three", monkeypatch, capsys,
                        _burns((0.001, 0.03), (50.0, 0.03), (99.0, 0.0394)))
    assert "WARNING: 2 violating maneuvers: -20 points" in out
    assert _scores(out) == pytest.approx([score - 20.0 for score in alone], abs=1e-4)


@pytest.mark.parametrize("total_km_s", [3.000001, 3.000000001])
def test_a_total_just_over_the_cap_disqualifies(
    total_km_s: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The cap has no tolerance (§4 note 1(c)), and the line shows the digits past it."""
    code, out, _ = _judged(tmp_path, monkeypatch, capsys, _burns((0.001, 0.0994)),
                           total_km_s=total_km_s)
    assert code == 2, out
    assert f"Σ|Δv| is {total_km_s:.12g} km/s, over the ranking rules' 3 km/s cap" in out


def test_a_scripts_total_over_the_cap_is_named_for_the_script(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Review of #191, L1: the script's Σ|Δv| was called --total-dv, a flag nobody gave."""
    import argparse

    from orbitwin.gmat_parser import MissionSchedule, ScheduledManeuver

    burns = tuple(ScheduledManeuver(f"B{k}", "SC_Human", 100.0 * k, (0.24, 0.0, 0.0),
                                    ("EarthMJ2000Eq", "Earth", "MJ2000Eq"), k)
                  for k in range(13))
    schedule = MissionSchedule("SC_Alien", burns, 2000.0, 60.0)
    args = argparse.Namespace(rules="ranking", script="day.script", certificate=None,
                              recorded=None, day_config=None, total_dv=None,
                              total_dv_from="--total-dv", judged=[], penalty_points=0.0)
    CHECK.total_from_script(args, schedule)
    assert args.total_dv == pytest.approx(3120.0)
    with pytest.raises(SystemExit) as stopped:
        CHECK.judge_rules(args, schedule)
    assert stopped.value.code == 2
    assert ("DISQUALIFIED: the burns of --script day.script: Σ|Δv| is 3.12 km/s, over the "
            "ranking rules' 3 km/s cap") in capsys.readouterr().out


def test_a_burn_99_9_s_after_the_last_costs_ten_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out, _ = _judged(tmp_path, monkeypatch, capsys, _burns((0.001, 0.05), (99.901, 0.0494)))
    assert "WARNING: 1 violating maneuver: -10 points" in out, out


def test_a_coefficient_off_by_one_part_in_a_million_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _, _ = _day_files(tmp_path)
    ct = 5369.334 * (1.0 + 1.0e-6)
    code, _, err = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--config", config,
                       "--total-dv", "0", "--ct", repr(ct))
    assert code == 2 and f"--ct is {ct!r} and the config's is 5369.334" in err, err


# --- §4 note 1(b): a crossing of the Earth before the interception (review of #191) ----


ALTITUDE = "SC_Human.Earth.Altitude"


def _r4(tmp_path: Path, name: str = "r4.txt", altitude: bool = True,
        dip: int | None = None, height_km: float | None = None, to_entry: bool = False,
        rename: str | None = None) -> tuple[Path, int]:
    """R4's measuring flight as GMAT wrote it, with GMAT's geodetic altitude or without, and
    data row `dip` (counted from the entry, negative before it) moved 20 km under the
    surface, or to `height_km`. `to_entry` ends it at the entry, a hit's report; `rename`
    gives GMAT's altitude column another name. Returns the report and the entry's row."""
    import lzma

    whole = tmp_path / "R4_diagnostic.txt"
    whole.write_bytes(lzma.decompress((ROWS / "R4_diagnostic.txt.xz").read_bytes()))
    header, data = CHECK.read_report(str(whole))
    xyz = {craft: [header.index(f"{craft}.EarthMJ2000Eq.{a}") for a in "XYZ"]
           for craft in ("SC_Human", "SC_Alien")}
    entry = int(np.flatnonzero(np.linalg.norm(
        data[:, xyz["SC_Human"]] - data[:, xyz["SC_Alien"]], axis=1) <= 5.0)[0])
    if dip is not None:
        row = entry + dip
        r = data[row, xyz["SC_Human"]]
        if height_km is None:
            # 6336 km: 20 km under the polar radius, under the surface at any latitude.
            data[row, xyz["SC_Human"]] = r / np.linalg.norm(r) * 6336.0
            data[row, header.index(ALTITUDE)] = -20.0
        else:
            # Along the radius, which moves the height by the same to 0.2 %.
            for _ in range(3):
                now = float(CHECK.geodetic_altitude_km(r[None, :])[0])
                r = r * (1.0 + (height_km - now) / float(np.linalg.norm(r)))
            data[row, xyz["SC_Human"]] = r
            data[row, header.index(ALTITUDE)] = height_km
    if to_entry:
        data = data[:entry + 1]
    keep = [k for k, column in enumerate(header) if altitude or column != ALTITUDE]
    names = [rename if rename and header[k] == ALTITUDE else header[k] for k in keep]
    return _write(tmp_path / name, names, data[:, keep]), entry


R4_HAND = ("--time-column", "SC_Human.ElapsedSecs", "--frozen-rows", "0", "--total-dv",
           "1820.3602294050023")


def test_r4_dips_to_110_km_and_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R4 perigee drags it to 109.613 km on GMAT's Earth.Altitude. Without the column the
    height from its EarthMJ2000Eq position is within 60 m of GMAT's (54 m worst, 8 m here)."""
    config, _, _ = _day_files(tmp_path)
    with_column, _ = _r4(tmp_path)
    code, out, err = run(monkeypatch, capsys, with_column, "--config", config, *R4_HAND)
    assert "DISQUALIFIED" not in out and code in (0, 1), out + err
    line = next(line for line in out.splitlines() if line.startswith("surface"))
    assert line.startswith("surface       : SC_Human at least 109.61") and ALTITUDE in line
    assert "(b) judged on the rows" in out
    gmat_km = float(line.split("at least ")[1].split()[0])
    without, _ = _r4(tmp_path, "r4-no-altitude.txt", altitude=False)
    _, out, _ = run(monkeypatch, capsys, without, "--config", config, *R4_HAND)
    line = next(line for line in out.splitlines() if line.startswith("surface"))
    assert "EarthMJ2000Eq position, WGS-84" in line
    assert float(line.split("at least ")[1].split()[0]) == pytest.approx(gmat_km, abs=0.06)
    assert "WARNING: within" not in out


def test_the_whole_r4_flight_agrees_with_gmats_altitude(tmp_path: Path) -> None:
    import lzma

    whole = tmp_path / "R4_diagnostic.txt"
    whole.write_bytes(lzma.decompress((ROWS / "R4_diagnostic.txt.xz").read_bytes()))
    header, data = CHECK.read_report(str(whole))
    ours = CHECK.geodetic_altitude_km(
        data[:, [header.index(f"SC_Human.EarthMJ2000Eq.{a}") for a in "XYZ"]])
    assert np.max(np.abs(ours - data[:, header.index(ALTITUDE)])) < 0.06


@pytest.mark.parametrize("height_km", [0.0, 0.05, 110.0, 400.0, 35786.0])
@pytest.mark.parametrize("latitude_deg", [0.0, 23.5, 45.0, 67.0, 89.9, -60.0])
def test_the_geodetic_height_comes_back_to_a_millimetre(
    height_km: float, latitude_deg: float
) -> None:
    """A point placed at a geodetic latitude and height above GMAT's ellipsoid."""
    a, f = CHECK.EARTH_EQUATORIAL_KM, CHECK.EARTH_FLATTENING
    e2 = f * (2.0 - f)
    lat, lon = np.radians(latitude_deg), np.radians(-130.0)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    r = np.array([[(n + height_km) * np.cos(lat) * np.cos(lon),
                   (n + height_km) * np.cos(lat) * np.sin(lon),
                   (n * (1.0 - e2) + height_km) * np.sin(lat)]])
    assert CHECK.geodetic_altitude_km(r)[0] == pytest.approx(height_km, abs=1.0e-6)


@pytest.mark.parametrize("altitude", [True, False])
def test_a_dip_under_the_surface_before_the_interception_disqualifies(
    altitude: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, _, _ = _day_files(tmp_path)
    report, entry = _r4(tmp_path, altitude=altitude, dip=-100)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config, *R4_HAND)
    assert code == 2 and "VERDICT" not in out, out
    assert (f"DISQUALIFIED: SC_Human is -{'20.000000' if altitude else ''}") in out
    assert f"below the surface on data row {entry - 100 + 1}, before the interception " \
           "(§4 note 1(b);" in out


@pytest.mark.parametrize("dip", [1, 2])
def test_a_dip_after_the_interception_is_not_judged(
    dip: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The rules disqualify a crossing before the interception only: the row after the
    entry is not judged, the entry's own is (below)."""
    config, _, _ = _day_files(tmp_path)
    report, _ = _r4(tmp_path, altitude=False, dip=dip)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config, *R4_HAND)
    assert "DISQUALIFIED" not in out and code != 2, out
    assert "SC_Human at least 109.6" in out


def test_a_dip_on_the_entry_row_disqualifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _, _ = _day_files(tmp_path)
    # GMAT's column only: moving the position would move the entry too.
    whole, entry = _r4(tmp_path)
    header, data = CHECK.read_report(str(whole))
    data[entry, header.index(ALTITUDE)] = -0.001
    report = _write(tmp_path / "entry.txt", header, data)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config, *R4_HAND)
    assert code == 2 and f"below the surface on data row {entry + 1}," in out, out


@pytest.mark.parametrize(("name", "code"), [("SC_Human.Altitude", 2), ("SC_Alien.Altitude", 0)])
def test_the_short_altitude_header_is_the_pursuers_only(
    name: str, code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Review of #191's delta, P2-1 (S17): GMAT writes SC_Human.Altitude for the short form,
    the same as Earth.Altitude. Altitude -1 m there passed, judged on the position."""
    config, _, _ = _day_files(tmp_path)
    report, _ = _r4(tmp_path, dip=-100, height_km=-0.001, to_entry=True, rename=name)
    got, out, _ = run(monkeypatch, capsys, report, "--config", config, *R4_HAND)
    assert got == (2 if code == 2 else CHECK.SURFACE_UNDECIDED_EXIT), out
    if code == 2:
        assert "DISQUALIFIED: SC_Human is -0.001000 km below" in out and name in out
    else:
        # The target's column is not the pursuer's: its position decides, 1 m is undecided.
        assert "SC_Human's EarthMJ2000Eq position" in out


def _r4_in(tmp_path: Path, frame: str, radius_km: float | None = None) -> tuple[Path, int]:
    """R4 to its entry with both craft's positions written in `frame` (EarthMJ2000Ec turned
    by the obliquity, any other name as is), without GMAT's altitude; the row 100 before
    the entry at `radius_km` from the centre."""
    report, entry = _r4(tmp_path, "r4-eq.txt", altitude=False, to_entry=True)
    header, data = CHECK.read_report(str(report))
    for craft in ("SC_Human", "SC_Alien"):
        cols = [header.index(f"{craft}.EarthMJ2000Eq.{a}") for a in "XYZ"]
        if radius_km is not None and craft == "SC_Human":
            r = data[entry - 100, cols]
            data[entry - 100, cols] = r / np.linalg.norm(r) * radius_km
        if frame == "EarthMJ2000Ec":
            c, s = np.cos(CHECK.OBLIQUITY_J2000_RAD), np.sin(CHECK.OBLIQUITY_J2000_RAD)
            x, y, z = data[:, cols[0]], data[:, cols[1]].copy(), data[:, cols[2]].copy()
            data[:, cols[1]], data[:, cols[2]] = c * y + s * z, -s * y + c * z
            assert np.allclose(x, data[:, cols[0]])
        for k, a in zip(cols, "XYZ", strict=True):
            header[k] = f"{craft}.{frame}.{a}"
    return _write(tmp_path / f"r4-{frame}.txt", header, data), entry


def test_ecliptic_positions_are_judged_through_the_obliquity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """S15: EarthMJ2000Ec at r 6000 km left (b) unjudged and exited 0."""
    config, _, _ = _day_files(tmp_path)
    (tmp_path / "ok").mkdir()
    report, _ = _r4_in(tmp_path / "ok", "EarthMJ2000Ec")
    code, out, _ = run(monkeypatch, capsys, report, "--config", config, *R4_HAND)
    assert code == 0, out
    line = next(line for line in out.splitlines() if line.startswith("surface"))
    assert "EarthMJ2000Ec position" in line
    assert float(line.split("at least ")[1].split()[0]) == pytest.approx(109.613, abs=0.06)
    (tmp_path / "low").mkdir()
    report, entry = _r4_in(tmp_path / "low", "EarthMJ2000Ec", radius_km=6000.0)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config, *R4_HAND)
    assert code == 2 and f"below the surface on data row {entry - 100 + 1}," in out, out


def test_a_frame_without_the_pole_is_not_judged_and_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _, _ = _day_files(tmp_path)
    report, _ = _r4_in(tmp_path, "MoonMJ2000Eq")
    code, out, _ = run(monkeypatch, capsys, report, "--config", config, *R4_HAND)
    assert code == CHECK.SURFACE_UNJUDGED_EXIT, out
    assert "SC_Human's position is in MoonMJ2000Eq, whose z is not the pole" in out
    assert out.splitlines()[-1].startswith("SURFACE NOT JUDGED (§4 note 1(b)): ")


@pytest.mark.parametrize(("height_km", "altitude", "code"), [
    (0.05, False, 6), (-0.05, False, 6), (-0.2, False, 2), (0.2, False, 0),
    (-0.001, True, 2), (0.001, True, 0),
])
def test_a_computed_height_near_the_surface_is_undecided(
    height_km: float, altitude: bool, code: int, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Review of #191's delta, L2: 0..+100 m only warned where -1 m disqualified. Computed
    from the position, within 100 m of 0 is undecided (exit 6); GMAT's column is strict."""
    config, _, _ = _day_files(tmp_path)
    report, _ = _r4(tmp_path, altitude=altitude, dip=-100, height_km=height_km,
                    to_entry=True)
    got, out, _ = run(monkeypatch, capsys, report, "--config", config, *R4_HAND)
    assert got == code, out
    if code == 6:
        assert out.splitlines()[-1].startswith("SURFACE UNDECIDED (§4 note 1(b)): SC_Human at ")
        assert "within the 100 m that SC_Human's EarthMJ2000Eq position, WGS-84 may be off by" \
            in out


def test_a_non_finite_altitude_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """L1 (S3b): argmin of a NaN hid every other row."""
    config, _, _ = _day_files(tmp_path)
    whole, entry = _r4(tmp_path, to_entry=True)
    header, data = CHECK.read_report(str(whole))
    data[entry - 50, header.index(ALTITUDE)] = float("nan")
    report = _write(tmp_path / "nan.txt", header, data)
    code, out, err = run(monkeypatch, capsys, report, "--config", config, *R4_HAND)
    assert code == 1 and "VERDICT" not in out, out
    assert f"data row {entry - 50 + 1} gives SC_Human an altitude that is not a finite" in err


def test_the_templates_own_state_below_the_surface_is_for_the_organizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """S14: RMAG 6371 km on the equator, -7.136 km, is the template's, not our design."""
    header, data = CHECK.read_report(str(ROWS / "R1_hit_default.txt"))
    cols = [header.index(f"SC_Human.EarthMJ2000Eq.{a}") for a in "XYZ"]
    data[0, cols] = [6371.0, 0.0, 0.0]
    report = _write(tmp_path / "r1.txt", header, data)
    code, out, _ = run(monkeypatch, capsys, report, "--time-column", "SC_Human.ElapsedSecs")
    assert code == CHECK.SURFACE_TEMPLATE_EXIT and "DISQUALIFIED" not in out, out
    assert out.splitlines()[-1] == (
        "SURFACE: the template's initial state is below the surface (§4 note 1(b)): SC_Human "
        "at -7.136300 km on data row 1, before any burn of ours (SC_Human's EarthMJ2000Eq "
        "position, WGS-84); ask the organizer")


def test_a_height_is_printed_however_small() -> None:
    assert CHECK._km(-1.0e-13) == "-1.000e-13" and CHECK._km(109.61) == "109.610000"


def test_a_miss_is_judged_to_its_last_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    header, data = CHECK.read_report(str(ROWS / "R1_miss_default.txt"))
    cols = [header.index(f"SC_Human.EarthMJ2000Eq.{a}") for a in "XYZ"]
    data[-1, cols] = data[-1, cols] / np.linalg.norm(data[-1, cols]) * 6336.0
    report = _write(tmp_path / "miss.txt", header, data)
    code, out, _ = run(monkeypatch, capsys, report, "--time-column", "SC_Human.ElapsedSecs")
    assert code == 2 and f"on data row {len(data)}, before the interception" in out, out


def test_without_the_pursuers_position_the_surface_is_not_judged_and_not_passed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review of #191's delta, P2-2: a report on which (b) cannot be judged passed silently."""
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_hit_noheader.txt", "--pursuer-cols",
                       "8,9,10", "--target-cols", "1,2,3")
    assert code == CHECK.SURFACE_UNJUDGED_EXIT == 4, out
    assert "surface       : not judged (§4 note 1(b)): the report has no " \
           "SC_Human.Earth.Altitude or SC_Human.Altitude, and no named position" in out
    assert out.splitlines()[-1].startswith("SURFACE NOT JUDGED (§4 note 1(b)): ")
    assert out.splitlines()[-1].endswith("by eye in GMAT up to the interception (step 6b.7), "
                                         "or ask the organizer")


def test_the_preliminary_does_not_rule_a_crossing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report, _ = _r4(tmp_path, altitude=False, dip=-100)
    code, out, _ = run(monkeypatch, capsys, report, "--rules", "preliminary", *R4_HAND)
    assert code != 2, out
    assert "surface" not in out and "DISQUALIFIED" not in out, out


def test_a_legal_plan_at_the_limits_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """249 m/s each, 2.99 km/s in all, exactly 100 s apart: the rules' own limits, not
    our safe margins, which are certify's."""
    plan = [(0.001 + 100.0 * k, 0.249) for k in range(12)] + [(1200.001, 0.002)]
    code, out, _ = _judged(tmp_path, monkeypatch, capsys, _burns(*plan), total_km_s=2.99)
    assert code == 0 and "WARNING" not in out and "DISQUALIFIED" not in out, out
    assert "(§4 note 1(a), (c), (d) judged on " in out and "the certificate's candidate" in out


def test_the_scripts_own_burns_are_judged_beside_a_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = (ROWS / "R1_hit.script").read_text(encoding="utf-8")
    old = "ImpulsiveBurn1.Element1 = 0.0607629102585799;"
    assert old in script
    config, certificate, report = _day_files(
        tmp_path, script_text=script.replace(old, "ImpulsiveBurn1.Element1 = 0.26;"))
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "DISQUALIFIED: " in out and "public_gmat_candidate.script: burn 1" in out


def test_the_preliminary_deducts_for_an_impulse_over_its_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§5: 每次違規機動扣 10 分, not a disqualification; both faults on one burn are one."""
    code, out, _ = _judged(tmp_path, monkeypatch, capsys,
                           _burns((0.001, 0.05), (50.0, 1.6)), total_km_s=1.65, unit="km/s",
                           cr_km=None, rules="preliminary", config_rules="preliminary",
                           flags=("--rules", "preliminary"))
    assert code == 0 and "DISQUALIFIED" not in out, out
    assert "1600.000000 m/s, over the preliminary rules' 1500 m/s per impulse (§2, §5)" in out
    assert "WARNING: 1 violating maneuver: -10 points" in out


def test_a_certificate_from_another_revision_of_the_rules_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The 2026/10/06 PDF priced misses on 50 e^{-(d-5)/100}; its replayed score is not ours."""
    config, certificate, report = _day_files(
        tmp_path, rule_sha256="ecd26af2cbb9befb17cb41a59ad200c20f6e1c467b87d46b589ba8a97e281ee3"
    )
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "certify again under the current rules" in err
    assert "ecd26af2" in err and "1ded3ebb" in err


def test_the_day_files_give_every_value_the_check_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    config, certificate, report = _day_files(tmp_path)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 0, out
    assert "time column   : SC_Human.ElapsedSecs" in out
    assert "The last row matches the certificate's T_team" in out
    # The same score as by hand, the certificate's km/s total read in m/s.
    by_hand = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--time-column",
                  "SC_Human.ElapsedSecs", *SCORE_FLAGS, "--total-dv", "99.4")[1]
    assert out.split("validator reads last row")[1].split("\n")[0] in by_hand


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unit": "km/s"}, "fix ΔV_team in m/s"),
        ({"cr_km": None}, "The score block has no cr_km"),
    ],
)
def test_a_day_config_outside_the_revised_rules_is_refused(
    kwargs: dict[str, object], message: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    config, certificate, report = _day_files(tmp_path, **kwargs)  # type: ignore[arg-type]
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and message in err, err


def test_the_day_config_gives_the_miss_its_c_r(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import math

    priced = {}
    for cr_km in (10000.0, 30000.0):
        (tmp_path / f"cr{cr_km:g}").mkdir()
        config, _, _ = _day_files(tmp_path / f"cr{cr_km:g}", cr_km=cr_km)
        code, out, _ = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--config", config,
                           "--total-dv", "0")
        assert code == 0, out
        closest = float(out.split("closest row")[1].split("dr =")[1].split("km")[0])
        distance = 100.0 / (1.0 + math.exp(((closest - 5.0) / cr_km) ** 2))
        assert f"distance {distance:.4f} +" in out and f"(C_r = {cr_km:g} km)" in out, out
        priced[cr_km] = distance
    assert priced[30000.0] > priced[10000.0] > 1.0
    # Review of the impact inventory: --kt 0.0003 --cr-km 3000 printed 81.7248 against the
    # certified 69.2921. A flag the config gives otherwise is refused, showing both.
    (tmp_path / "flag").mkdir()
    config, _, _ = _day_files(tmp_path / "flag", cr_km=10000.0)
    for flags, said in ((["--cr-km", "30000"], "--cr-km is 30000.0 and the config's is 10000.0"),
                        (["--kt", "0.0003"], "--kt is 0.0003 and the config's is 0.003"),
                        (["--rules", "preliminary"], "--rules preliminary, but ")):
        code, out, err = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--config",
                             config, "--total-dv", "0", *flags)
        assert code == 2 and said in err and "score" not in out, (flags, err)
    # The config's own value by hand agrees.
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--config", config,
                       "--total-dv", "0", "--cr-km", "10000", "--kt", "0.003")
    assert code == 0 and f"distance {priced[10000.0]:.4f} +" in out, out


R8 = ROWS / "R8_team54.txt"
#: SC_Human's ElapsedSecs on R8's last row: T_team from the earlier epoch, its own.
R8_T_TEAM_S = 6101.43773064483


def _r8_script() -> str:
    """R1_hit.script with the ReportFile of the R8 rehearsal, SC_Human's Altitude first."""
    text = (ROWS / "R1_hit.script").read_text(encoding="utf-8")
    old = "ReportFile1.Add = {SC_Alien.ElapsedSecs,"
    assert old in text
    return text.replace(old, "ReportFile1.Add = {SC_Human.Earth.Altitude, SC_Alien.ElapsedSecs,")


def _r8_day(tmp_path: Path, **kwargs: object) -> tuple[Path, Path, Path]:
    """R8 under the organizer's clock: T_max 8511.667 s from SC_Human's epoch, 1800 s first."""
    settings: dict[str, object] = {
        "report": "R8_team54.txt", "origin": "earliest", "window": "after_alignment",
        "origin_craft": "SC_Human",
        "t_team": R8_T_TEAM_S, "frozen_rows": 6, "horizon_s": 8511.667, "offset_s": 1800.0,
        "script_text": _r8_script(), "stop_depth_km": 1.5,
    }
    settings.update(kwargs)
    return _day_files(tmp_path, **settings)  # type: ignore[arg-type]


def test_an_r8_report_is_scored_on_the_pursuers_elapsed_secs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SC_Human starts first, so its ElapsedSecs is T_team (organizer, 2026-11-04)."""
    config, certificate, report = _r8_day(tmp_path)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 0, out
    assert "time column   : SC_Human.ElapsedSecs" in out
    assert f"LAST row      : t = {R8_T_TEAM_S:.6f} s" in out
    assert "The last row matches the certificate's T_team" in out
    assert f"Tteam = {R8_T_TEAM_S:11.4f} s -> score" in out


def test_without_a_certificate_the_report_says_which_craft_started_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _, report = _r8_day(tmp_path)
    code, out, err = run(monkeypatch, capsys, report, "--config", config, "--frozen-rows", "6",
                         "--total-dv", "0.217")
    assert code == 0, out
    assert "time column   : SC_Human.ElapsedSecs" in out
    assert "SC_Human.ElapsedSecs is T_team" in err and "runs ahead" in err
    assert f"Tteam = {R8_T_TEAM_S:11.4f} s -> score" in out
    # R6: SC_Alien starts first, and its column is read as before.
    (tmp_path / "r6").mkdir()
    config, _, report = _day_files(tmp_path / "r6", "R6_team54.txt", origin="earliest",
                                   window="after_alignment", horizon_s=8511.667)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config, "--frozen-rows", "6",
                       "--total-dv", "0.217")
    assert code == 0 and "time column   : SC_Alien.ElapsedSecs" in out
    assert f"Tteam = {R6_T_TEAM_S:11.4f} s -> score" in out
    # One start: the two agree and the pursuer's is read.
    (tmp_path / "r1").mkdir()
    config, _, report = _day_files(tmp_path / "r1", origin="earliest")
    code, out, err = run(monkeypatch, capsys, report, "--config", config, "--total-dv", "0.1")
    assert code == 0 and "time column   : SC_Human.ElapsedSecs" in out
    assert "agrees with the other's" in err and "-> score" in out


@pytest.mark.parametrize(
    ("given", "said"),
    [
        ({"origin_craft": "SC_Alien", "t_team": 4301.437730749603},
         "the certificate's clock.time_origin_craft gives SC_Alien.ElapsedSecs"),
        ({"time_column": "SC_Alien.ElapsedSecs"}, "--time-column is SC_Alien.ElapsedSecs"),
    ],
)
def test_the_later_crafts_elapsed_secs_is_refused_as_t_team(
    given: dict[str, object], said: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """R8 as main certified it: T_team read off SC_Alien, 1800 s short of the organizer's."""
    flags = ["--time-column", str(given.pop("time_column"))] if "time_column" in given else []
    config, certificate, report = _r8_day(tmp_path, **given)
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate, *flags)
    if flags:
        # By hand against the certificate's clock.time_origin_craft: refused before the rows.
        assert code == 2 and said in err, err
        assert "certificate's clock.time_origin_craft gives SC_Human.ElapsedSecs" in err, err
        return
    assert code == 1 and "SC_Human.ElapsedSecs runs ahead" in err and said in err, err


def test_an_earliest_certificate_must_name_the_origin_craft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _r8_day(tmp_path, drop="clock.time_origin_craft")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "written before T_team counted from the earlier" in err


def test_a_ranking_config_with_another_time_origin_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _r8_day(tmp_path, origin="target")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "從比較早開始的開始算" in err


def test_a_report_that_ends_elsewhere_than_certified_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _day_files(tmp_path,
                                             t_team=_last_time("R1_hit_default.txt") + 1.0)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 1 and "this is not the report certify measured" in out


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rules": "preliminary"}, "made under the preliminary rules"),
        ({"drop": "clock.t_team_s"}, "has no clock.t_team_s"),
        ({"drop": "rule_set"}, "has no rule_set.name"),
        ({"drop": "clock.mission_time_origin"}, "has no clock.mission_time_origin"),
        ({"drop": "submission_files"}, "has no submission_files"),
        ({"drop": "submission_script_name"}, "has no submission_script_name"),
    ],
)
def test_a_certificate_that_is_not_this_days_is_refused(
    kwargs: dict[str, object], message: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    config, certificate, report = _day_files(tmp_path, **kwargs)  # type: ignore[arg-type]
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and message in err, err


@pytest.mark.parametrize(
    ("certified", "message"),
    [
        # R2: certified at C_r 300, checked at 3000: 23.6716 and 31.5363 printed, exit 0.
        ({"cr_km": 3000.0}, "certified under C_r 3000 km, checking under C_r 300 km"),
        ({"ct": 4633.1048}, "certified under Ct 4633.1048 s, checking under Ct 5369.334 s"),
        ({"delta_v_unit": "km/s"}, "certified under delta-v unit 'km/s'"),
    ],
)
def test_a_certificate_priced_under_other_coefficients_is_refused(
    certified: dict[str, object], message: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    config, certificate, report = _day_files(tmp_path, certified_score=certified)
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and message in err, err
    assert "certify again under the announced coefficients" in err


def test_a_coefficient_given_by_hand_is_held_to_the_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _day_files(tmp_path)
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate, "--cr-km", "3000")
    # Held to the config first, which gives it (an unscored config: the test below).
    assert code == 2 and "--cr-km is 3000.0 and the config's is 300.0" in err, err


def test_a_certificate_from_before_score_block_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _day_files(tmp_path, drop="score_block")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "--certificate has no score_block" in err, err


def test_an_unscored_certificate_is_priced_by_the_check_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Step 4a certifies under run/day.yaml, before the coefficients: no score disagrees."""
    config, certificate, report = _day_files(tmp_path)
    body = json.loads(certificate.read_text(encoding="utf-8"))
    body["score_block"] = None
    certificate.write_text(json.dumps(body), encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, report, "--config", config,
                         "--certificate", certificate)
    assert code == 0, out
    assert "the certificate was made without coefficients" in err


def _hand(tmp_path: Path, block: object = "config", *, scored: bool = True,
          drop: bool = False, flown: str | None = None) -> tuple[list[str | Path], Path]:
    """The hand command of RUNBOOK step 7 for the GUI's report, with a verification.json
    of its script whose score_block is the config's unless given."""
    import hashlib

    import yaml

    config, certificate, report = _day_files(tmp_path)
    body = json.loads(certificate.read_text(encoding="utf-8"))
    if not scored:
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        del raw["score"]
        config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    verification = tmp_path / "verification.json"
    script = certificate.parent / "public_gmat_candidate.script"
    # Numbered as certify writes it, so --script binds the report beside it.
    script.write_text(script.read_text(encoding="utf-8").replace(
        "'ODC_2026_TeamXX.txt'", "'ODC_2026_Team54.txt'"), encoding="utf-8")
    record: dict[str, object] = {
        "candidate_script_sha256": flown or hashlib.sha256(script.read_bytes()).hexdigest()
    }
    if not drop:
        record["score_block"] = body["score_block"] if block == "config" else block
    verification.write_text(json.dumps(record), encoding="utf-8")
    return [report, "--config", config, "--script", script, *CUT], verification


def test_the_verification_must_be_of_the_script_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Step 4a's verification.json, or an earlier .refused one, passed on its block alone."""
    argv, verification = _hand(tmp_path, flown="0" * 64)
    code, _, err = run(monkeypatch, capsys, *argv, "--verification", verification)
    assert code == 2, err
    assert f"--verification verified a script with SHA-256 {'0' * 64}, and --script" in err
    assert "the refusal message names" in err
    record = json.loads(verification.read_text(encoding="utf-8"))
    del record["candidate_script_sha256"]
    verification.write_text(json.dumps(record), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, *argv, "--verification", verification)
    assert code == 2 and "has no candidate_script_sha256" in err and "Certify again" in err
    without = [arg for arg in argv if arg not in ("--script", *CUT)
               and not str(arg).endswith(".script")]
    code, _, err = run(monkeypatch, capsys, *without, "--verification", verification)
    assert code == 2 and "--verification needs --script" in err, err


def test_the_hand_check_says_when_it_checked_no_score_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv, verification = _hand(tmp_path)
    code, out, err = run(monkeypatch, capsys, *argv)
    assert code == 3, out + err
    assert "note: no --certificate or --verification" in err
    code, _, err = run(monkeypatch, capsys, *argv, "--verification", verification)
    assert code == 3 and "no --certificate or --verification" not in err


def test_the_hand_check_takes_the_verifications_score_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """RUNBOOK 6b/7 with plots: certify stops, no certification.json, the GUI wrote the
    report. Certified at C_r 300, hand-checked at 3000, it exited 0 and said nothing."""
    argv, verification = _hand(tmp_path)
    code, out, err = run(monkeypatch, capsys, *argv, "--verification", verification)
    assert code == 3, out + err
    other = json.loads(verification.read_text(encoding="utf-8"))
    other["score_block"]["cr_km"] = 3000.0
    verification.write_text(json.dumps(other), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, *argv, "--verification", verification)
    assert code == 2, err
    assert "verified under C_r 3000 km, checking under C_r 300 km" in err
    assert "certify again under the announced coefficients" in err


@pytest.mark.parametrize(("kwargs", "flags", "code", "message"), [
    ({"drop": True}, [], 2, "--verification has no score_block"),
    # 3: the cut fixture passes with the binding warned past (CUT).
    ({"block": None}, [], 3, "the verification was made without coefficients"),
    # An unscored config prices nothing but what is given by hand.
    ({"scored": False}, ["--cr-km", "3000"], 2,
     "verified under C_r 300 km, checking under C_r 3000 km"),
    ({"scored": False}, ["--cr-km", "300"], 3, ""),
])
def test_the_verifications_block_is_held_as_the_certificates_is(
    kwargs: dict[str, object], flags: list[str], code: int, message: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    argv, verification = _hand(tmp_path, **kwargs)  # type: ignore[arg-type]
    got, out, err = run(monkeypatch, capsys, *argv, "--verification", verification, *flags)
    assert got == code and message in err, out + err


def test_an_unscored_day_config_holds_hand_coefficients_to_the_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    config, certificate, report = _day_files(tmp_path)
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    del raw["score"]
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate, "--cr-km", "3000")
    assert code == 2 and "certified under C_r 300 km, checking under C_r 3000 km" in err, err


@pytest.mark.parametrize("pair", [["--certificate"], []])
def test_verification_needs_a_config_and_no_certificate(
    pair: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, certificate, report = _day_files(tmp_path)
    extra: list[str | Path] = ["--config", config, "--certificate", certificate] if pair else []
    code, _, err = run(monkeypatch, capsys, report, *extra, "--verification", certificate)
    assert code == 2 and "--verification goes with --config and without --certificate" in err


@pytest.mark.parametrize(("relative", "code"), [(1.0e-13, 0), (2.0e-12, 2)])
def test_a_kv_apart_only_past_12_digits_is_shown_in_full(
    relative: float, code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exact equality said "certified under kv 0.0012, checking under kv 0.0012"."""
    kv = 0.0012 * (1.0 + relative)
    config, certificate, report = _day_files(tmp_path, certified_score={"kv": kv})
    got, out, err = run(monkeypatch, capsys, report, "--config", config,
                        "--certificate", certificate)
    assert got == code, out + err
    if code:
        assert f"certified under kv {kv!r}, checking under kv 0.0012:" in err, err


def test_a_null_script_name_is_refused_as_a_missing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _day_files(tmp_path)
    body = json.loads(certificate.read_text(encoding="utf-8"))
    body["submission_script_name"] = None
    certificate.write_text(json.dumps(body), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "--certificate has no submission_script_name; certify" in err, err


@pytest.mark.parametrize("where", ["missing", "directory"])
def test_a_script_flag_that_names_no_file_is_refused(
    where: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, certificate, report = _day_files(tmp_path)
    script = tmp_path / "nowhere.script"
    if where == "directory":
        script.mkdir()
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate, "--script", script)
    assert code == 2 and f"--script {script}: no such file" in err, err


@pytest.mark.parametrize("flags", [(), ("--delimiter", ",")])
def test_a_missing_certified_script_gets_advice_main_takes(
    flags: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main() refuses --script beside the format flags, so the advice cannot be both."""
    config, certificate, report = _day_files(tmp_path)
    (certificate.parent / "public_gmat_candidate.script").unlink()
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate, *flags)
    assert code == 2 and "or pass a copy of it with --script" in err, err
    assert ("instead of the format flags" in err) == bool(flags), err


def test_a_certificate_with_another_time_origin_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "plain").mkdir()
    config, _, _ = _day_files(tmp_path / "plain")
    _, certificate, report = _day_files(tmp_path, origin=None)
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "written before T_team counted from the earlier" in err


def test_a_report_or_script_not_the_certified_bytes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _day_files(tmp_path)
    other = tmp_path / "other" / report.name
    other.parent.mkdir()
    other.write_bytes((ROWS / "R1_over_default.txt").read_bytes())
    code, _, err = run(monkeypatch, capsys, other, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "is not a report this certificate lists" in err
    script = certificate.parent / "public_gmat_candidate.script"
    script.write_text(script.read_text(encoding="utf-8") + "% edited\n", encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "differs from the script this certificate lists" in err


def test_the_certificates_script_gives_the_report_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = script_in(tmp_path, FORMATS["comma"]).read_text(encoding="utf-8")
    config, certificate, report = _day_files(tmp_path, "R1_hit_comma.txt", script_text=script,
                                             t_team=_last_time("R1_hit_default.txt"))
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 0, out


def test_a_certificate_needs_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, certificate, report = _day_files(tmp_path)
    code, _, err = run(monkeypatch, capsys, report, "--certificate", certificate)
    assert code == 2 and "--certificate needs --config" in err


def test_a_config_without_an_origin_refuses_an_aligned_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R6 and R7 read with R1's config passed, scored 1800 s off."""
    config, _, _ = _day_files(tmp_path)
    lines = (ROWS / "R1_hit_default.txt").read_text(encoding="utf-8").splitlines()
    shifted = [lines[0]] + [
        " ".join([f"{float(row.split()[0]) + 1800.0:.9f}", *row.split()[1:]]) for row in lines[1:]
    ]
    aligned = tmp_path / "aligned.txt"
    aligned.write_text("\n".join(shifted) + "\n", encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, aligned, "--config", config)
    assert code == 1 and "holds an epoch alignment" in err


def test_a_scored_miss_takes_t_max_from_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _, _ = _day_files(tmp_path)
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--config", config)
    assert code == 0 and "a scored miss" in out
    # Review of the impact inventory: an R6 miss cut short passed as scored under a hand
    # T_max. A --horizon-s other than the config's is refused.
    code, out, err = run(monkeypatch, capsys, ROWS / "R1_miss_default.txt", "--config", config,
                         "--horizon-s", "6800")
    assert code == 2 and "--horizon-s is 6800.0 and the config's is 6711.667" in err, err
    assert "scored miss" not in out


def _edit_json(path: Path, **changes: object) -> None:
    body = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        outer, _, inner = key.partition("__")
        if inner:
            body[outer][inner] = value
        else:
            body[outer] = value
    path.write_text(json.dumps(body), encoding="utf-8")


def test_a_certificate_on_another_t_max_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _day_files(tmp_path)
    _edit_json(certificate, candidate={"total_dv_km_s": 0.0994, "metadata": {"horizon_s": 9999.0}})
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "the config is not the one certify ran" in err


@pytest.mark.parametrize("name", ["../elsewhere.script", "/etc/hostname"])
def test_the_certificates_script_must_sit_beside_it(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, certificate, report = _day_files(tmp_path)
    _edit_json(certificate, submission_script_name=name)
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "is not a file name" in err


def test_a_renamed_script_is_still_held_to_the_certified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _day_files(tmp_path)
    renamed = tmp_path / "renamed.script"
    renamed.write_text((ROWS / "R1_hit.script").read_text(encoding="utf-8") + "% edited\n",
                       encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate, "--script", renamed)
    assert code == 2 and "differs from the script this certificate lists" in err


def test_a_hand_flag_the_certificate_contradicts_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review of the docs (RUNBOOK step 7): --total-dv 2999 beside a certificate printed
    78.9902 for its 95.8797 and exited 0; --frozen-rows and --radius-km went through too."""
    config, certificate, report = _day_files(tmp_path)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 0 and "skipped: 0" in out, out
    # The same values by hand agree, and pass.
    for flags in (["--frozen-rows", "0"], ["--total-dv", "99.4"], ["--radius-km", "5"]):
        code, out, err = run(monkeypatch, capsys, report, "--config", config,
                             "--certificate", certificate, *flags)
        assert code == 0, (flags, out, err)
    for flags, said in (
        (["--frozen-rows", "5"], "--frozen-rows is 5 and the certificate's frozen_rows is 0"),
        (["--total-dv", "2999"], "--total-dv is 2999.0 m/s and the certificate's "
                                 "candidate.total_dv_km_s is 99.4 m/s"),
        (["--total-dv", "99.41"], "--total-dv is 99.41 m/s"),
        (["--radius-km", "50"], "--radius-km is 50 and the ranking rules' sphere"),
    ):
        code, out, err = run(monkeypatch, capsys, report, "--config", config,
                             "--certificate", certificate, *flags)
        assert code == 2 and said in err and "score" not in out, (flags, out, err)
    # Format flags by hand keep the certificate's script out of the way.
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate, "--write-headers", "true")
    assert "do not also pass its fields" not in err


def test_a_missing_or_broken_day_file_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _day_files(tmp_path)
    for flags, message in (
        (["--config", tmp_path / "missing.yaml"], "--config:"),
        (["--config", config, "--certificate", tmp_path / "missing.json"], "--certificate:"),
    ):
        code, _, err = run(monkeypatch, capsys, report, *flags)
        assert code == 2 and message in err, err
    code, _, err = run(monkeypatch, capsys, tmp_path / "missing.txt", "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "no such report" in err


def test_a_held_still_craft_alone_shows_an_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SC_Human, its ElapsedSecs included, holds still for rows as an alignment writes them."""
    config, _, _ = _day_files(tmp_path)
    lines = (ROWS / "R1_hit_default.txt").read_text(encoding="utf-8").splitlines()
    header = lines[0].split()
    human = [k for k, name in enumerate(header) if name.startswith("SC_Human.")]
    first = lines[1].split()
    held = [lines[0]]
    for row in lines[1:]:
        values = row.split()
        if len(held) <= 4:
            for k in human:
                values[k] = first[k]
        held.append(" ".join(values))
    # One ElapsedSecs only, so the alignment shows in the held positions alone.
    drop = header.index("SC_Alien.ElapsedSecs")
    held = [" ".join(v for k, v in enumerate(row.split()) if k != drop) for row in held]
    aligned = tmp_path / "held.txt"
    aligned.write_text("\n".join(held) + "\n", encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, aligned, "--config", config)
    assert code == 1 and "holds an epoch alignment" in err


def test_a_last_row_short_of_the_days_stop_depth_fails_as_certify_would(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # R1_hit ends 50 m inside; the day's 1.5 km asks for 30 times that.
    config, certificate, report = _day_files(tmp_path, stop_depth_km=1.5)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 1, out
    assert "the last row is 50 m inside" in out
    assert "short of the config's policy.stop_depth_km = 1500 m" in out
    assert "without policy.allow_shallow_stop_submission" in out
    # One verdict, the refusal.
    assert [line for line in out.splitlines() if line.startswith("VERDICT")] == [
        "VERDICT: certify refuses such a hit without policy.allow_shallow_stop_submission."
    ]


def _ending(tmp_path: Path, report: Path, depth_km: float) -> Path:
    """`report` with its last row moved `depth_km` inside the 5 km sphere (negative: outside),
    as the organizer's GMAT ending our script shallower would write it."""
    header, data = CHECK.read_report(str(report))
    human = [header.index(f"SC_Human.EarthMJ2000Eq.{axis}") for axis in "XYZ"]
    alien = [header.index(f"SC_Alien.EarthMJ2000Eq.{axis}") for axis in "XYZ"]
    relative = data[-1, alien] - data[-1, human]
    data[-1, alien] = data[-1, human] + relative / np.linalg.norm(relative) * (5.0 - depth_km)
    return _write(tmp_path / f"gui-{depth_km}.txt", header, data)


GUI_HAND = ("--time-column", "SC_Human.ElapsedSecs", "--frozen-rows", "0", "--total-dv", "99.4")


@pytest.mark.parametrize("depth_km", [1.4, 0.5, 0.01, 0.0001])
def test_a_gui_report_short_of_the_stop_depth_passes_with_a_warning(
    depth_km: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Review of the docs, P1: the organizer's GMAT ending our script 15 m shallower exited 1,
    and RUNBOOK sent the hit to the next fallback, which drifts the same way. The stop depth
    is our margin for that drift; the rules ask only that the last row be inside."""
    config, _, report = _day_files(tmp_path, stop_depth_km=1.5)
    code, out, err = run(monkeypatch, capsys, _ending(tmp_path, report, depth_km),
                         "--config", config, *GUI_HAND)
    assert code == 0, out + err
    assert f"the last row is {depth_km * 1000.0:.4g} m inside" in out
    assert (f"WARNING: the last row is {depth_km * 1000.0:.4g} m inside, short of the config's "
            "policy.stop_depth_km = 1500 m") in out
    assert "VERDICT: the LAST row is inside the sphere." in out


@pytest.mark.parametrize("depth_km", [-0.001, 5.0e-7])
def test_a_gui_report_not_shown_inside_fails(
    depth_km: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """5.001 km is outside; 0.5 mm inside is within what reading the digits can move."""
    config, _, report = _day_files(tmp_path, stop_depth_km=1.5)
    code, out, err = run(monkeypatch, capsys, _ending(tmp_path, report, depth_km),
                         "--config", config, *GUI_HAND)
    assert code == 1, out + err
    assert "VERDICT: the LAST row is inside" not in out
    said = ("under the 1 mm that reading the printed digits can move it" if depth_km > 0
            else "rows within 5 km: 0")
    assert said in out, out


def test_the_certificate_still_holds_our_report_to_the_stop_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same 50 m: passed by hand, refused against certify's own report's certificate."""
    config, certificate, report = _day_files(tmp_path, stop_depth_km=1.5)
    assert run(monkeypatch, capsys, report, "--config", config, *GUI_HAND)[0] == 0
    assert run(monkeypatch, capsys, report, "--config", config,
               "--certificate", certificate)[0] == 1


def test_the_horizon_policy_judges_no_stop_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """verify stops no hit under it, so certify measures no depth either."""
    config, certificate, report = _day_files(tmp_path, stop_depth_km=1.5,
                                             mission_end_policy="horizon")
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 0, out
    assert "the last row is 50 m inside" in out and "WARNING" not in out


def test_a_shallow_last_row_passes_only_with_the_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _day_files(tmp_path, stop_depth_km=1.5, allow_shallow=True)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 0, out
    assert "WARNING: short of the config's policy.stop_depth_km" in out
    assert "policy.allow_shallow_stop_submission is set" in out


@pytest.mark.parametrize(("depth_km", "code"), [(0.0505, 0), (0.0506, 1)])
def test_the_stop_depth_is_judged_with_certifys_tolerance(
    depth_km: float, code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 49.998 m deep: inside 1 % of 50.5 m, outside 1 % of 50.6 m.
    config, certificate, report = _day_files(tmp_path, stop_depth_km=depth_km)
    assert run(monkeypatch, capsys, report, "--config", config,
               "--certificate", certificate)[0] == code


def test_without_a_config_the_depth_is_shown_not_judged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt")
    assert code == 0 and "the last row is 50 m inside" in out
    assert "WARNING" not in out


# --- the earliest origin without a certificate (review of #181) ---------------------------


def _without(tmp_path: Path, source: Path, drop: str, name: str) -> Path:
    header, data = CHECK.read_report(str(source))
    keep = [i for i, column in enumerate(header) if column != drop]
    return _write(tmp_path / name, [header[i] for i in keep], data[:, keep])


def test_an_aligned_report_without_the_earlier_crafts_elapsed_secs_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Main fell back to SC_Alien's column, found by name, and passed a late hit."""
    config, _, _ = _r8_day(tmp_path)
    report = _without(tmp_path, R8, "SC_Human.ElapsedSecs", "r8.txt")
    code, out, err = run(monkeypatch, capsys, report, "--config", config, "--frozen-rows", "6")
    assert code == 1 and "VERDICT" not in out
    assert "holds SC_Alien still, so SC_Human started first" in err
    assert "has no SC_Human.ElapsedSecs" in err


def test_a_late_hit_is_refused_on_the_earlier_crafts_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R8 enters at 6101.44 s on SC_Human's clock and 4301.44 s on SC_Alien's: T_max 6000 s."""
    config, _, report = _r8_day(tmp_path, horizon_s=6000.0)
    code, out, err = run(monkeypatch, capsys, report, "--config", config, "--frozen-rows", "6")
    assert code == 1 and "after T_max = 6000.000000 s" in out, out + err


def _r1_without_alien_clock(tmp_path: Path) -> tuple[Path, Path]:
    """R1's hit with only SC_Human's ElapsedSecs, and the script that writes it."""
    report = _without(tmp_path, ROWS / "R1_hit_default.txt", "SC_Alien.ElapsedSecs", "r1.txt")
    text = (ROWS / "R1_hit.script").read_text(encoding="utf-8")
    assert "{SC_Alien.ElapsedSecs, " in text and "ReportFile1.FixedWidth = true;" in text
    script = tmp_path / "r1_one_clock.script"
    script.write_text(text.replace("{SC_Alien.ElapsedSecs, ", "{").replace(
        "ReportFile1.FixedWidth = true;", "ReportFile1.FixedWidth = false;"), encoding="utf-8")
    return report, script


def test_one_elapsed_secs_reads_as_t_team_only_when_the_day_shows_one_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report, script = _r1_without_alien_clock(tmp_path)
    (tmp_path / "same").mkdir()
    config, _, _ = _day_files(tmp_path / "same")
    code, out, err = run(monkeypatch, capsys, uploaded(tmp_path, report), "--config", config,
                         "--script", script, *CUT)
    assert code == 3 and "time column   : SC_Human.ElapsedSecs" in out and "-> score" in out, err
    # Without the script nothing shows one start: the config alone only claims it.
    for flags in ([], ["--time-column", "SC_Human.ElapsedSecs"]):
        code, out, err = run(monkeypatch, capsys, report, "--config", config, *flags)
        assert code == 1 and "VERDICT" not in out, flags
        assert "does not say whether that craft started first" in err, flags


def test_the_later_crafts_clock_is_refused_without_a_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(monkeypatch, capsys, R8, "--frozen-rows", "6",
                         "--time-column", "SC_Alien.ElapsedSecs")
    assert code == 1 and "VERDICT" not in out
    assert "SC_Human.ElapsedSecs runs ahead" in err and "--time-column is SC_Alien" in err
    code, out, _ = run(monkeypatch, capsys, R8, "--frozen-rows", "6",
                       "--time-column", "SC_Human.ElapsedSecs")
    assert code == 0 and f"LAST row      : t = {R8_T_TEAM_S:.6f} s" in out


def test_an_earliest_config_without_a_window_claims_one_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _, report = _r8_day(tmp_path, window=None)
    code, _, err = run(monkeypatch, capsys, report, "--config", config, "--frozen-rows", "6")
    assert code == 1 and "another day's report" in err


def test_from_origin_counts_the_alignments_rows_in_d_min(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """As score_candidate and certify do; R8 moved 60000 km apart after t = 0."""
    import math

    header, data = CHECK.read_report(str(R8))
    data = data.copy()
    # From the first row SC_Alien moves on; it waits at its epoch through row 7, t = 0.
    data[7:, header.index("SC_Alien.EarthMJ2000Eq.X")] += 60000.0
    report = _write(tmp_path / "miss.txt", header, data)
    end_s = float(data[-1, header.index("SC_Human.ElapsedSecs")])
    pursuer = [header.index(f"SC_Human.EarthMJ2000Eq.{a}") for a in "XYZ"]
    target = [header.index(f"SC_Alien.EarthMJ2000Eq.{a}") for a in "XYZ"]
    gaps = np.linalg.norm(data[:, pursuer] - data[:, target], axis=1)
    assert gaps[:6].min() < gaps[6:].min()
    priced = {}
    for window in ("after_alignment", "from_origin"):
        (tmp_path / window).mkdir()
        config, _, _ = _r8_day(tmp_path / window, window=window, horizon_s=end_s, cr_km=30000.0)
        code, out, err = run(monkeypatch, capsys, report, "--config", config, "--frozen-rows",
                             "6", "--total-dv", "217")
        assert code == 0, out + err
        priced[window] = out
    for window, d_min in (("after_alignment", gaps[6:].min()), ("from_origin", gaps[:6].min())):
        term = 100.0 / (1.0 + math.exp(((d_min - 5.0) / 30000.0) ** 2))
        assert f"distance {term:.4f} +" in priced[window], priced[window]
    assert "counts the alignment's rows too" in priced["from_origin"]
    assert "counts the alignment's rows too" not in priced["after_alignment"]


def test_two_elapsed_secs_that_cross_say_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    header, data = CHECK.read_report(str(R8))
    data = data.copy()
    data[-1, header.index("SC_Alien.ElapsedSecs")] += 4000.0
    report = _write(tmp_path / "crossed.txt", header, data)
    config, _, _ = _r8_day(tmp_path)
    code, _, err = run(monkeypatch, capsys, report, "--config", config, "--frozen-rows", "6")
    assert code == 1 and "runs ahead of the other on every row" in err


def test_with_one_start_either_column_may_be_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _, report = _day_files(tmp_path)
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--time-column", "SC_Alien.ElapsedSecs", "--total-dv", "99.4")
    assert code == 0 and "time column   : SC_Alien.ElapsedSecs" in out


def test_the_earliest_resolution_leaves_no_doubtful_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _, report = _r8_day(tmp_path)
    code, _, err = run(monkeypatch, capsys, report, "--config", config, "--frozen-rows", "6")
    assert code == 0 and "only if its craft is the time origin" not in err


def test_a_nan_elapsed_secs_is_named_before_the_origin_is_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    header, data = CHECK.read_report(str(R8))
    data = data.copy()
    data[9, header.index("SC_Human.ElapsedSecs")] = np.nan
    report = _write(tmp_path / "nan.txt", header, data)
    config, _, _ = _r8_day(tmp_path)
    code, _, err = run(monkeypatch, capsys, report, "--config", config, "--frozen-rows", "6")
    assert code == 1 and "data row 10 holds a SC_Human.ElapsedSecs that is not finite" in err


def test_two_elapsed_secs_are_compared_to_their_printed_digits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """At Precision 6 an ElapsedSecs near 4300 s prints to 0.01 s, so 0.004 s is no lead."""
    header, data = CHECK.read_report(str(ROWS / "R1_hit_default.txt"))
    data = data.copy()
    data[1:, header.index("SC_Alien.ElapsedSecs")] += 0.004
    report = _write(tmp_path / "p6.txt", header, data)
    script = script_in(tmp_path, {"Precision": "6", "FixedWidth": "false", "Delimiter": "' '"})
    code, _, err = run(monkeypatch, capsys, uploaded(tmp_path, report), "--script", script, *CUT)
    assert code == 3 and "differ" not in err, err
    # Printed to 16 digits the same 0.004 s is two clocks.
    code, _, err = run(monkeypatch, capsys, report)
    assert code == 1 and "differ, as after an epoch alignment" in err


def test_the_script_is_held_to_its_own_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A --script that is another file the certificate lists is not the submitted script."""
    import hashlib

    config, certificate, report = _day_files(tmp_path)
    original = certificate.parent / "official_original.script"
    original.write_bytes(b"original")
    body = json.loads(certificate.read_text(encoding="utf-8"))
    body["submission_files"][original.name] = hashlib.sha256(b"original").hexdigest()
    certificate.write_text(json.dumps(body), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate, "--script", original)
    assert code == 2 and "differs from the script this certificate lists" in err


def test_the_final_script_hash_is_the_submission_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import hashlib

    config, certificate, report = _day_files(tmp_path)
    script = certificate.parent / "public_gmat_candidate.script"
    _edit_json(certificate, final_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 0, out
    _edit_json(certificate, final_script_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "final_script_sha256 is not its submission script's" in err


def test_format_flags_do_not_skip_the_script_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--delimiter reads the format by hand; the script beside the certificate is still held."""
    config, certificate, report = _day_files(tmp_path)
    script = certificate.parent / "public_gmat_candidate.script"
    script.write_text(script.read_text(encoding="utf-8") + "% edited\n", encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate, "--delimiter", " ")
    assert code == 2 and "differs from the script this certificate lists" in err


@pytest.mark.parametrize(
    "field, value",
    [("submission_files", ["x"]), ("submission_files", {"a": 3}), ("clock", ["x"]),
     ("candidate", "x"), ("final_script_parity", 3), ("rule_set", [1]),
     ("submission_script_name", 7), ("submission_script_name", "nowhere.script")],
)
def test_a_malformed_certificate_is_refused_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    field: str, value: object,
) -> None:
    config, certificate, report = _day_files(tmp_path)
    _edit_json(certificate, **{field: value})
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2, err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "field, value",
    [("clock__t_team_s", float("nan")), ("clock__t_team_s", float("inf")),
     ("clock__t_team_s", "6711"), ("clock__time_origin_offset_s", float("nan")),
     ("candidate__total_dv_km_s", float("nan")),
     ("final_script_parity", {"replayed_score": float("inf")})],
)
def test_a_non_finite_certificate_number_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    field: str, value: object,
) -> None:
    """JSON's NaN compares as nothing: a NaN T_team silently passed nothing and failed nothing."""
    config, certificate, report = _day_files(tmp_path)
    _edit_json(certificate, **{field: value})
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "not a finite number" in err



def test_a_certificate_that_is_not_an_object_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _day_files(tmp_path)
    certificate.write_text(json.dumps([1, 2]), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "is not a JSON object" in err


@pytest.mark.parametrize("value", [[1], 3, None])
def test_a_listed_hash_that_is_not_text_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    value: object,
) -> None:
    config, certificate, report = _day_files(tmp_path)
    body = json.loads(certificate.read_text(encoding="utf-8"))
    body["submission_files"]["extra.txt"] = value
    certificate.write_text(json.dumps(body), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "submission_files is malformed" in err


def test_a_missing_script_beside_the_certificate_is_called_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _day_files(tmp_path)
    (certificate.parent / "public_gmat_candidate.script").unlink()
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "public_gmat_candidate.script is missing" in err


def test_a_given_script_needs_nothing_beside_the_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--script naming a byte-identical copy is the script; the one beside may be gone."""
    import shutil

    config, certificate, report = _day_files(tmp_path)
    beside = certificate.parent / "public_gmat_candidate.script"
    elsewhere = tmp_path / "elsewhere.script"
    shutil.copyfile(beside, elsewhere)
    beside.unlink()
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate, "--script", elsewhere)
    assert code == 0, out


@pytest.mark.parametrize(
    "field, value, said",
    [
        ("candidate", {"total_dv_km_s": 0.0994, "metadata": [1]}, "candidate.metadata is not"),
        ("candidate", {"total_dv_km_s": 0.0994, "metadata": {"horizon_s": "6711.667"}},
         "not a finite number"),
        ("candidate", {"total_dv_km_s": 0.0994, "metadata": {"horizon_s": float("nan")}},
         "not a finite number"),
        ("candidate", {"total_dv_km_s": True, "metadata": {"horizon_s": 6711.667}},
         "not a finite number"),
        ("final_script_parity", {"replayed_score": True}, "not a finite number"),
        ("candidate__total_dv_km_s", 10**400, "not a finite number"),
        ("clock__time_origin_offset_s", 10**400, "not a finite number"),
    ],
    ids=["metadata-list", "horizon-text", "horizon-nan", "dv-bool", "score-bool",
         "dv-huge-int", "offset-huge-int"],
)
def test_each_certificate_number_is_held_to_a_finite_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    field: str, value: object, said: str,
) -> None:
    config, certificate, report = _day_files(tmp_path)
    _edit_json(certificate, **{field: value})
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and said in err, err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "field, value, said",
    [
        ("clock", [], "clock is not an object"), ("candidate", 0, "candidate is not an object"),
        ("final_script_parity", "", "final_script_parity is not an object"),
        ("rule_set", [], "rule_set is not an object"),
        ("candidate__metadata", [], "candidate.metadata is not an object"),
        ("frozen_rows", False, "frozen_rows is False, not a row count"),
        ("frozen_rows", -1, "not a row count"), ("frozen_rows", 2.5, "not a row count"),
        ("frozen_rows", float("inf"), "not a row count"),
        ("clock__time_origin_offset_s", None, "not a finite number"),
    ],
)
def test_an_empty_value_of_the_wrong_type_is_not_a_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    field: str, value: object, said: str,
) -> None:
    """`x or {}` let [] and 0 through; only an absent key takes the default."""
    config, certificate, report = _day_files(tmp_path)
    if field == "candidate__metadata":
        body = json.loads(certificate.read_text(encoding="utf-8"))
        body["candidate"]["metadata"] = value
        certificate.write_text(json.dumps(body), encoding="utf-8")
    else:
        _edit_json(certificate, **{field: value})
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and said in err, err


def test_a_config_without_craft_names_reads_the_default_time_column(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """config/standard.yaml names no craft: the column was None.ElapsedSecs."""
    _, out, err = run(monkeypatch, capsys, ROWS / "R1_hit_default.txt", "--config",
                      ROOT / "config" / "standard.yaml", "--rules", "preliminary")
    assert "None." not in out + err
    assert "time column   : SC_Human.ElapsedSecs" in out


# --- the delta review of #181 -------------------------------------------------------------


def _joint_rows(tmp_path: Path, drop: str | None = None) -> Path:
    """R8 without its alignment's rows: both clocks run, SC_Human's 1800 s ahead."""
    header, data = CHECK.read_report(str(R8))
    keep = [i for i, column in enumerate(header) if column != drop]
    return _write(tmp_path / f"joint-{drop}.txt", [header[i] for i in keep], data[6:][:, keep])


def test_without_alignment_rows_the_clock_that_runs_ahead_is_t_team(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Main's later column read T_team 4301 s where the organizer reads 6101 s."""
    report = _joint_rows(tmp_path)
    code, out, err = run(monkeypatch, capsys, report, "--time-column", "SC_Alien.ElapsedSecs")
    assert code == 1 and "VERDICT" not in out
    assert "SC_Human.ElapsedSecs runs ahead" in err and "--time-column is SC_Alien" in err
    code, out, _ = run(monkeypatch, capsys, report, "--time-column", "SC_Human.ElapsedSecs")
    assert code == 0 and f"LAST row      : t = {R8_T_TEAM_S:.6f} s" in out


def test_one_later_clock_without_alignment_rows_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _joint_rows(tmp_path, "SC_Human.ElapsedSecs")
    config, _, _ = _r8_day(tmp_path)
    for flags in ([], ["--time-column", "SC_Alien.ElapsedSecs"]):
        code, out, err = run(monkeypatch, capsys, report, "--config", config, *flags)
        assert code == 1 and "VERDICT" not in out, flags
        assert "does not say whether that craft started first" in err, flags


def test_the_frozen_crafts_only_clock_is_refused_without_a_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _without(tmp_path, R8, "SC_Human.ElapsedSecs", "r8.txt")
    code, out, err = run(monkeypatch, capsys, report, "--frozen-rows", "6",
                         "--time-column", "SC_Alien.ElapsedSecs")
    assert code == 1 and "VERDICT" not in out
    assert "holds SC_Alien still, so it started later" in err


def test_with_one_start_the_certificate_takes_either_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Matrix c0286: a same-epoch report that carries only SC_Alien's ElapsedSecs."""
    import hashlib

    script = (ROWS / "R1_hit.script").read_text(encoding="utf-8")
    assert "SC_Human.ElapsedSecs, " in script and "ReportFile1.FixedWidth = true;" in script
    script = script.replace("SC_Human.ElapsedSecs, ", "").replace(
        "ReportFile1.FixedWidth = true;", "ReportFile1.FixedWidth = false;")
    # The same instant written another way: only GMAT reads that as one start, so here
    # only the certificate's zero offset can.
    assert "SC_Alien.Epoch = '09 Nov 2026 02:00:00.000';" in script
    script = script.replace("SC_Alien.Epoch = '09 Nov 2026 02:00:00.000';",
                            "SC_Alien.Epoch = '09 Nov 2026 02:00:00';")
    # A window set in step 2 just in case: the config alone does not say one start.
    config, certificate, uploaded = _day_files(tmp_path, script_text=script,
                                               window="after_alignment")
    alone = _without(tmp_path, ROWS / "R1_hit_default.txt", "SC_Human.ElapsedSecs", "r1.txt")
    uploaded.write_bytes(alone.read_bytes())
    body = json.loads(certificate.read_text(encoding="utf-8"))
    body["submission_files"][uploaded.name] = hashlib.sha256(uploaded.read_bytes()).hexdigest()
    certificate.write_text(json.dumps(body), encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, uploaded, "--config", config,
                         "--certificate", certificate)
    assert code == 0, out + err
    assert "time column   : SC_Alien.ElapsedSecs" in out
    assert "The last row matches the certificate's T_team" in out
    # A certificate without the alignment's offset is not one certify wrote.
    body["clock"].pop("alignment_offset_s")
    certificate.write_text(json.dumps(body), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, uploaded, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "no clock.alignment_offset_s" in err


# --- the final review of #181 ---------------------------------------------------------------


def test_one_later_clock_is_refused_without_the_days_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Probes c0467 and c0481: R8 without its alignment's rows and SC_Human's clock read
    T_team 4301.4377 s where the organizer reads 6101.4377 s."""
    report = _joint_rows(tmp_path, "SC_Human.ElapsedSecs")
    code, out, err = run(monkeypatch, capsys, report, "--time-column", "SC_Alien.ElapsedSecs")
    assert code == 1 and "VERDICT" not in out
    assert "only SC_Alien.ElapsedSecs and no epoch alignment's rows" in err
    config, _, _ = _r8_day(tmp_path, window=None)
    for flags in ([], ["--script", str(ROOT / "examples" / "ranking" /
                                      "R8_PursuerFirst_Aligned.script")]):
        code, out, err = run(monkeypatch, capsys, report, "--config", config, *flags)
        assert code == 1 and "VERDICT" not in out, flags
        assert "does not say whether that craft started first" in err, flags


def test_a_certificate_from_before_the_earliest_rule_asks_to_certify_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for origin in ("target", None):
        base = tmp_path / str(origin)
        (base / "old").mkdir(parents=True)
        config, _, _ = _day_files(base)
        _, certificate, report = _day_files(base / "old", origin=origin)
        code, _, err = run(monkeypatch, capsys, report, "--config", config,
                           "--certificate", certificate)
        assert code == 2 and "written before T_team counted from the earlier" in err, err
        assert "Certify again" in err


def test_a_certificate_naming_a_clock_the_report_lacks_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The certificate names SC_Human; the report carries only SC_Alien's clock."""
    import hashlib

    script = _r8_script().replace("SC_Human.ElapsedSecs, ", "").replace(
        "ReportFile1.FixedWidth = true;", "ReportFile1.FixedWidth = false;")
    config, certificate, uploaded = _r8_day(tmp_path, script_text=script)
    uploaded.write_bytes(_joint_rows(tmp_path, "SC_Human.ElapsedSecs").read_bytes())
    body = json.loads(certificate.read_text(encoding="utf-8"))
    body["submission_files"][uploaded.name] = hashlib.sha256(uploaded.read_bytes()).hexdigest()
    body["frozen_rows"] = 0
    certificate.write_text(json.dumps(body), encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, uploaded, "--config", config,
                         "--certificate", certificate)
    assert code == 1 and "VERDICT" not in out, (out, err)
    assert "does not say which craft started first" in err
    code, out, err = run(monkeypatch, capsys, uploaded, "--config", config,
                         "--certificate", certificate, "--time-column", "SC_Alien.ElapsedSecs")
    assert code == 2 and "VERDICT" not in out and "leave --time-column out" in err, (out, err)


@pytest.mark.parametrize("precision", range(1, 17))
def test_a_second_run_is_refused_at_every_precision(
    precision: int, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The back-check allows a step back of 1 us or two printed digits, the instant
    R2026a ends an alignment on; a run appended after the first goes back much further."""
    header, data = CHECK.read_report(str(ROWS / "R1_hit_default.txt"))
    script = script_in(tmp_path, {"Precision": str(precision), "FixedWidth": "false",
                                  "Delimiter": "' '"})

    def printed(rows: np.ndarray, name: str) -> Path:
        """The rows as GMAT prints them at this Precision, under the uploaded name."""
        folder = tmp_path / name
        folder.mkdir()
        path = folder / "written.txt"
        path.write_text(" ".join(header) + "\n" + "\n".join(
            " ".join(f"{float(v):.{precision}g}" for v in row) for row in rows) + "\n",
            encoding="utf-8")
        return uploaded(folder, path)

    code, out, err = run(monkeypatch, capsys, printed(np.vstack([data, data]), "twice"),
                         "--script", script, *CUT)
    assert code == 1 and "VERDICT" not in out and "so the report holds more than one run" in err
    # Back by just over what the check allows: still two runs.
    seconds = header.index("SC_Alien.ElapsedSecs")
    digit = 10.0 ** (math.floor(math.log10(float(np.max(np.abs(data[:, seconds]))))) - precision + 1)
    allowed = max(1.0e-6, 2.0 * digit)
    back = data.copy()
    for clock in ("SC_Alien.ElapsedSecs", "SC_Human.ElapsedSecs"):
        back[-1, header.index(clock)] = back[-2, header.index(clock)] - 3.0 * allowed
    code, _, err = run(monkeypatch, capsys, printed(back, "back"), "--script", script, *CUT)
    assert code != 0 and "goes back" in err, err


#: The OpenFrames lines of R1_openframes.script, R1 with an OpenFrames view.
_PLOT_BLOCK = "".join(
    line for line in (ROOT / "tests" / "fixtures" / "report_bind" / "R1_openframes.script")
    .read_text(encoding="ascii").splitlines(keepends=True)
    if line.startswith(("Create OpenFrames", "DefaultOrbitView"))
)


def _copy_day_files(tmp_path: Path, change=None, *, plots: str = _PLOT_BLOCK):  # type: ignore[no-untyped-def]
    """As certify writes them when its exact runs ran the console copy (gmat.console_copy)."""
    from orbitwin.console_copy import build_console_copy, proven_copy
    from orbitwin.gmat_parser import GmatScriptDocument

    text = (ROWS / "R1_hit.script").read_text(encoding="utf-8")
    text = text.replace("BeginMissionSequence;", plots + "BeginMissionSequence;", 1)
    config, certificate, report = _day_files(tmp_path, script_text=text)
    body = json.loads(certificate.read_text(encoding="utf-8"))
    # The organizer's template, whose plot lines the script carries as they are.
    official = certificate.parent / "official_original.script"
    official.write_text(text, encoding="utf-8")
    sha = __import__("hashlib").sha256(official.read_bytes()).hexdigest()
    body["submission_files"][official.name] = body["original_script_sha256"] = sha
    document = GmatScriptDocument.read(certificate.parent / body["submission_script_name"])
    proven = proven_copy(document, build_console_copy(document) or b"", "copy")
    body["console_copy"] = {"script": str(tmp_path / "copy.script"), **proven.to_dict()}
    body["exact_runs"] = [{"script_sha256": proven.copy_sha256}] * 3
    if change is not None:
        change(body)
    certificate.write_text(json.dumps(body), encoding="utf-8")
    return config, certificate, report


def test_a_console_copys_report_passes_with_the_copy_proven_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _copy_day_files(tmp_path)
    code, out, err = run(monkeypatch, capsys, report, "--config", config,
                         "--certificate", certificate)
    assert code == 0, out + err
    assert "rebuilt and proven again here" in err
    assert "The last row matches the certificate's T_team" in out


def _set(path: str, value: object):  # type: ignore[no-untyped-def]
    def change(body: dict) -> None:  # type: ignore[type-arg]
        *outer, last = path.split(".")
        target = body
        for key in outer:
            target = target[key]
        target[last] = value
    return change


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda body: (_set("console_copy.copy_sha256", "0" * 64)(body),
                       _set("exact_runs", [{"script_sha256": "0" * 64}])(body)),
         "is not the one the certificate's runs ran"),
        (_set("console_copy.removed", ["OpenFramesView DefaultOrbitView_View"]),
         "is not the one the certificate's runs ran"),
        (_set("console_copy.deleted_lines", [True]), "console_copy is malformed"),
        (_set("console_copy.deleted_lines", [float("nan")]), "console_copy is malformed"),
        (_set("console_copy", None), "console_copy is malformed"),
        (_set("console_copy.submitted_sha256", "1" * 64), "is not of the script it certifies"),
        (_set("exact_runs", [{"script_sha256": "2" * 64}]), "did not all run its console_copy"),
        (_set("exact_runs", []), "did not all run its console_copy"),
    ],
    ids=["copy hash", "removed", "a bool line", "a NaN line", "null", "other script",
         "other run", "no runs"],
)
def test_a_console_copy_record_that_does_not_hold_is_refused(
    change, message: str, tmp_path: Path,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    config, certificate, report = _copy_day_files(tmp_path, change)
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and message in err, err


def test_a_script_whose_copy_cannot_be_proven_again_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _copy_day_files(tmp_path)
    script = certificate.parent / "public_gmat_candidate.script"
    # A kept line naming a plot the copy would leave out: no copy proves.
    script.write_text(script.read_text(encoding="ascii").replace(
        "BeginMissionSequence;", "Create CoordinateSystem NoSuch;\nNoSuch.Origin = "
        "DefaultOrbitView;\nBeginMissionSequence;", 1), encoding="ascii")
    body = json.loads(certificate.read_text(encoding="utf-8"))
    sha = __import__("hashlib").sha256(script.read_bytes()).hexdigest()
    body["submission_files"][script.name] = body["console_copy"]["submitted_sha256"] = sha
    certificate.write_text(json.dumps(body), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "cannot be proven again" in err and "NoSuch" in err, err


def test_a_console_copys_report_must_be_held_to_t_team(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Item 3: a time column that keeps T_team from being compared exits 2 under a copy;
    without one it stays a note."""
    flags = ("--time-column", "SC_Human.EarthMJ2000Eq.X")
    (tmp_path / "copy").mkdir()
    config, certificate, report = _copy_day_files(tmp_path / "copy")
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate, *flags)
    assert code == 2 and "cannot be compared, which a console copy's report needs" in out, out
    config, certificate, report = _day_files(tmp_path)
    code, out, err = run(monkeypatch, capsys, report, "--config", config,
                         "--certificate", certificate, *flags)
    assert code != 2 and "was not compared" in err, out + err


@pytest.mark.parametrize("runs", [None, [], [{"script_sha256": "3" * 64}]],
                         ids=["absent", "none", "another script"])
def test_exact_runs_of_another_script_are_refused_without_a_copy(
    runs: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """T5 of the review of #189: a copy-mode certificate with its console_copy deleted
    names exact runs of the copy, not of the script, and is refused."""
    config, certificate, report = _day_files(tmp_path)
    body = json.loads(certificate.read_text(encoding="utf-8"))
    if runs is None:
        del body["exact_runs"]
    else:
        body["exact_runs"] = runs
    certificate.write_text(json.dumps(body), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "did not all run the script it lists" in err, err


def test_a_copy_certificate_stripped_of_its_copy_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _copy_day_files(tmp_path, lambda body: body.pop("console_copy"))
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "did not all run the script it lists" in err, err


def _forge(certificate: Path, edit) -> None:  # type: ignore[no-untyped-def]
    """Edit the certified script and forge every hash the certificate holds of it."""
    import hashlib

    from orbitwin.console_copy import build_console_copy, proven_copy
    from orbitwin.gmat_parser import GmatScriptDocument

    body = json.loads(certificate.read_text(encoding="utf-8"))
    script = certificate.parent / body["submission_script_name"]
    script.write_text(edit(script.read_text(encoding="ascii")), encoding="ascii")
    sha = hashlib.sha256(script.read_bytes()).hexdigest()
    body["submission_files"][script.name] = sha
    document = GmatScriptDocument.read(script)
    proven = proven_copy(document, build_console_copy(document) or b"", "copy")
    body["console_copy"].update(proven.to_dict())
    body["exact_runs"] = [{"script_sha256": proven.copy_sha256}] * 3
    certificate.write_text(json.dumps(body), encoding="utf-8")


def test_step_7_holds_the_copys_plot_lines_to_the_official_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Low-1 of the delta review of #189: a changed plot value under forged hashes
    passed, since the copy leaves the line out. official_original.script is the bar."""
    config, certificate, report = _copy_day_files(tmp_path)
    _forge(certificate, lambda text: text.replace("DefaultOrbitView.Axes = On;",
                                                  "DefaultOrbitView.Axes = Off;"))
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "does not carry official_original.script's plot lines" in err, err


@pytest.mark.parametrize("how", ["missing", "unlisted", "another"])
def test_step_7_needs_the_official_script_the_certificate_names(
    how: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, certificate, report = _copy_day_files(tmp_path)
    official = certificate.parent / "official_original.script"
    body = json.loads(certificate.read_text(encoding="utf-8"))
    if how == "missing":
        official.unlink()
    elif how == "unlisted":
        del body["submission_files"][official.name]
    else:
        official.write_text(official.read_text(encoding="ascii") + "% other\n", encoding="ascii")
    certificate.write_text(json.dumps(body), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "not the official script the certificate names" in err, err


# --- the official report without the target's position (R3 T16) --------------------------

#: The organizer's workshop style: the pursuer's time and position, nothing of SC_Alien.
WORKSHOP_ADD = ("SC_Human.ElapsedSecs", "SC_Human.EarthMJ2000Eq.X", "SC_Human.EarthMJ2000Eq.Y",
                "SC_Human.EarthMJ2000Eq.Z")


def _hit_score(t_team: float, total_m_s: float = 99.4) -> float:
    """50 + the time and delta-v terms under _day_files' coefficients, written out."""
    return (50.0 + 25.0 / (1.0 + math.exp(0.003 * (t_team - 5369.334)))
            + 25.0 / (1.0 + math.exp(0.0012 * (total_m_s - 2000.0))))


def _workshop_day(tmp_path: Path, columns: tuple[str, ...] = WORKSHOP_ADD, *,
                  record: dict[str, object] | None = None, hit: bool = True,
                  parity_km: float = 4.95, **day: object) -> tuple[Path, Path, Path]:
    """_day_files with the report cut to ``columns`` and the script's Add list to match;
    ``record`` is certify's official_report_rows, the default one of such a hit."""
    import hashlib

    script_text = (ROWS / "R1_hit.script").read_text(encoding="utf-8")
    add = next(line for line in script_text.splitlines() if line.startswith("ReportFile1.Add"))
    script_text = script_text.replace(add, f"ReportFile1.Add = {{{', '.join(columns)}}};")
    script_text = script_text.replace("ODC_2026_TeamXX.txt", "ODC_2026_Team54.txt")
    config, certificate, uploaded = _day_files(tmp_path, script_text=script_text,  # type: ignore[arg-type]
                                               stop_depth_km=1.5, **day)
    lines = (ROWS / "R1_hit_default.txt").read_text(encoding="utf-8").splitlines()
    names = lines[0].split()
    kept = [names.index(column) for column in columns]
    uploaded.write_text("\n".join("   ".join(line.split()[i] for i in kept) for line in lines)
                        + "\n", encoding="utf-8")
    body = json.loads(certificate.read_text(encoding="utf-8"))
    body["submission_files"][uploaded.name] = hashlib.sha256(uploaded.read_bytes()).hexdigest()
    last = _last_time("R1_hit_default.txt")
    body["candidate"].update(successful=hit, metadata={
        **body["candidate"]["metadata"], "gmat_event": {"stop_depth_km": 1.52}})
    body["final_script_parity"] = {"first_entry_time_s": 4517.3 if hit else None,
                                   "minimum_distance_km": parity_km}
    body["official_report_rows"] = record if record is not None else {
        "carries_target_position": False, "scored_distance_km": 5.0,
        "score": _hit_score(last)}
    certificate.write_text(json.dumps(body), encoding="utf-8")
    return config, certificate, uploaded


def test_a_certified_report_without_the_target_passes_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R3 T16: every fallback ended on cannot locate SC_Human X/Y/Z, so no upload."""
    config, certificate, report = _workshop_day(tmp_path)
    code, out, err = run(monkeypatch, capsys, report, "--config", config,
                         "--certificate", certificate)
    assert code == 0, out + err
    last = _last_time("R1_hit_default.txt")
    assert out.splitlines()[-1] == (
        "WARNING: the official report carries no SC_Alien position; d_min and the interception "
        "were proven on certify's diagnostic flight of the same script, not on this report")
    assert (f"  validator reads last row   Tteam = {last:11.4f} s -> score "
            f"{_hit_score(last):.4f}") in out.splitlines()
    assert "surface       : SC_Human at least" in out
    assert "The score matches the certificate's" in out


def test_a_report_without_the_targets_position_or_the_pursuers_velocity_is_bound_by_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """By hand it binds what it can, then says only the certificate can judge it: exit 7."""
    config, _, report = _workshop_day(tmp_path)
    script = report.parent / "public_gmat_candidate.script"
    # The fixture keeps the first and last rows: --binding warn passes the cut over.
    hand = ("--config", config, "--script", script, "--binding", "warn")
    code, out, err = run(monkeypatch, capsys, report, *hand)
    assert code == 2 and "--accept-end-time-binding-only" in err, out + err
    code, out, err = run(monkeypatch, capsys, report, *hand, "--accept-end-time-binding-only")
    assert code == CHECK.TARGET_NOT_IN_REPORT_EXIT == 7, out + err
    assert out.splitlines()[-1].startswith(
        f"VERDICT: NOT JUDGED BY HAND: {report} carries no SC_Alien position")
    assert "step 7's --certificate form" in out.splitlines()[-1]
    # Without a script nothing is bound, and still nothing is judged.
    code, out, _ = run(monkeypatch, capsys, report, "--config", config)
    assert code == 7 and "NOT JUDGED BY HAND" in out


def test_a_report_with_neither_craft_s_position_is_not_judged_on_the_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _workshop_day(tmp_path, ("SC_Human.ElapsedSecs",))
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == CHECK.SURFACE_UNJUDGED_EXIT, out
    assert "WARNING: the official report carries no SC_Alien position" in out


def test_a_tampered_report_without_the_target_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, certificate, report = _workshop_day(tmp_path)
    lines = report.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1].replace(lines[-1].split()[0], "4517.7", 1)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "SHA-256 differs" in err


@pytest.mark.parametrize(
    ("change", "code", "said"),
    [
        # Written before certify proved such a report: certify again.
        ({"record": {"carries_target_position": True}}, 2, "certify again"),
        # The last row a microsecond off T_team, over the 16 digits' half a last digit.
        ({"t_team": 4517.612829012796 + 1.0e-6}, 1, "differ in the printed digits"),
        ({"record": {"carries_target_position": False, "scored_distance_km": 5.0,
                     "score": 90.0}}, 1, "the certificate's, priced on these rows, is 90.000000"),
    ],
)
def test_a_certificate_that_does_not_prove_the_report_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    change: dict[str, object], code: int, said: str,
) -> None:
    config, certificate, report = _workshop_day(tmp_path, **change)  # type: ignore[arg-type]
    got, out, err = run(monkeypatch, capsys, report, "--config", config,
                        "--certificate", certificate)
    assert got == code and said in out + err, out + err
    assert "WARNING: the official report carries no" not in out


def test_a_certified_miss_without_the_target_is_priced_on_the_diagnostic_d_min(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    last = _last_time("R1_hit_default.txt")
    distance = 100.0 / (1.0 + math.exp(((65.0 - 5.0) / 300.0) ** 2))
    score = _hit_score(last) - 50.0 + distance
    config, certificate, report = _workshop_day(
        tmp_path, hit=False, horizon_s=last, parity_km=65.0, record={
            "carries_target_position": False, "scored_distance_km": 65.0, "score": score})
    code, out, err = run(monkeypatch, capsys, report, "--config", config,
                         "--certificate", certificate)
    assert code == 0, out + err
    (line,) = [line for line in out.splitlines() if line.startswith("  scored miss:")]
    assert line.startswith(f"  scored miss: distance {distance:.4f} + time ")
    assert line.endswith(f" = {score:.4f}")
    assert out.splitlines()[-1].startswith("WARNING: the official report carries no SC_Alien")


def test_the_missing_craft_is_the_one_named() -> None:
    """R3 P2-4: SC_Alien's columns were missing, and the refusal named SC_Human."""
    header = ["SC_Human.ElapsedSecs", "SC_Human.X", "SC_Human.Y", "SC_Human.Z"]
    with pytest.raises(SystemExit, match="cannot locate SC_Alien X/Y/Z"):
        CHECK.position_columns(header, 4, "SC_Human", "SC_Alien")
    with pytest.raises(SystemExit, match="cannot locate SC_Human X/Y/Z"):
        CHECK.position_columns(["SC_Alien.X", "SC_Alien.Y", "SC_Alien.Z"], 3, "SC_Human",
                               "SC_Alien")


@pytest.mark.parametrize(("delta", "code"), [(0.0, 0), (1.0e-3, 1)])
def test_a_report_with_both_craft_is_held_to_the_certified_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    delta: float, code: int,
) -> None:
    """R3 T14: the certificate claimed 95.7635 and the printed rows gave 95.7623."""
    config, certificate, report = _day_files(tmp_path)
    body = json.loads(certificate.read_text(encoding="utf-8"))
    body["official_report_rows"] = {"carries_target_position": True,
                                    "score": _hit_score(_last_time("R1_hit_default.txt")) + delta}
    certificate.write_text(json.dumps(body), encoding="utf-8")
    got, out, _ = run(monkeypatch, capsys, report, "--config", config,
                      "--certificate", certificate)
    assert got == code, out
    assert ("VERDICT: the score here is" in out) is bool(code)


@pytest.mark.parametrize("parity_km", [64.0, 66.0, math.nan])
def test_a_no_target_miss_is_priced_only_on_its_final_replays_d_min(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    parity_km: float,
) -> None:
    """Review of #195, Low-1: official_report_rows.scored_distance_km was taken on its own."""
    last = _last_time("R1_hit_default.txt")
    config, certificate, report = _workshop_day(
        tmp_path, hit=False, horizon_s=last, parity_km=parity_km, record={
            "carries_target_position": False, "scored_distance_km": 65.0, "score": 70.0})
    code, out, err = run(monkeypatch, capsys, report, "--config", config,
                         "--certificate", certificate)
    assert code == 2 and "is not its final replay's minimum_distance_km" in err, out + err


def test_the_last_row_reads_back_as_gmat_printed_it() -> None:
    """Review of #195, P1: R8's T_team 5911.5072750952095 s prints at Precision 16 as
    5911.507275095209, the double below, 9.1e-13 s off where half a digit is 5e-13 s."""
    t_team = 5911.5072750952095
    assert CHECK._same_printed(5911.507275095209, t_team, 16)
    assert abs(5911.507275095209 - t_team) > 0.5e-12
    assert not CHECK._same_printed(5911.507275095207, t_team, 16)


def test_a_certificate_of_another_head_start_setting_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """certify records policy.head_start_burns; step 7's --config must agree with it."""
    config, certificate, report = _day_files(tmp_path)
    body = json.loads(certificate.read_text(encoding="utf-8"))
    body["clock"]["head_start_burns"] = False
    certificate.write_text(json.dumps(body), encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 2 and "certified with policy.head_start_burns False" in err
    body["clock"]["head_start_burns"] = True
    certificate.write_text(json.dumps(body), encoding="utf-8")
    code, out, _ = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 0, out


def test_a_config_naming_the_rules_by_id_passes_as_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The certificate records the rule set's name; a config may give its id instead."""
    config, certificate, report = _day_files(tmp_path, config_rules="tasa-odc-2026/ranking")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--certificate", certificate)
    assert code == 0, err[-600:]
    (tmp_path / "pre").mkdir()
    other, _, _ = _day_files(tmp_path / "pre", config_rules="tasa-odc-2026/preliminary",
                             cr_km=None)
    code, _, err = run(monkeypatch, capsys, report, "--config", other, "--rules", "preliminary",
                       "--certificate", certificate)
    assert code == 2 and "made under the ranking rules, not tasa-odc-2026/preliminary" in err
