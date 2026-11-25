"""What each ReportFile format writes, and that the reader splits it.

Every file in tests/fixtures/gmat_report_formats is the first 40 lines of a
report stock GmatConsole R2026a wrote for examples/two_burn_rendezvous.script
with its plots commented out (diagnostic._neutralise_plots), ReportFile1.Add
widened to both craft's positions and the case's ReportFile1 fields set as
MANIFEST.json records them (a field recorded as null left out); one GmatConsole
run per case. MANIFEST.json holds each case's ReportFile fields as script text
and the SHA-256 of the whole report. A through O are the cases of #70. K_tab
has '\\t' in the script, which GMAT writes as backslashes; P_tab_character has a
tab character there. Recorded again on 2026-11-15 for this template; the same
steps first reproduced five cases (A_default, K_tab, P_tab_character,
T_no_delimiter, O_noheader_narrow) of the template the fixtures were first
recorded on byte for byte.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from orbitwin.config import OrbitWinConfig
from orbitwin.errors import GmatRunError, ParseError
from orbitwin.gmat_parser import _report_file_spec, parse_gmat_script, report_file_specs
from orbitwin.models import ReportFileSpec
from orbitwin.report_format import ReportFormat, read_rows, report_format
from orbitwin.report_parser import parse_numeric_report

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "gmat_report_formats"
MANIFEST = json.loads((FIXTURES / "MANIFEST.json").read_text(encoding="utf-8"))
ADD_LIST = (
    "HumanSC.A1ModJulian",
    "HumanSC.EarthMJ2000Eq.X",
    "HumanSC.EarthMJ2000Eq.Y",
    "HumanSC.EarthMJ2000Eq.Z",
    "AlienSC.EarthMJ2000Eq.X",
    "AlienSC.EarthMJ2000Eq.Y",
    "AlienSC.EarthMJ2000Eq.Z",
)
#: Precision 6 writes other numbers; a narrow header-less column fuses them.
SAME_NUMBERS = sorted(set(MANIFEST) - {"J_lowprec", "O_noheader_narrow"})


def _spec(case: str) -> ReportFileSpec:
    fields = {key: value for key, value in MANIFEST[case]["settings"].items() if value is not None}
    return _report_file_spec("ReportFile1", fields)


def _read(case: str) -> np.ndarray:
    spec = _spec(case)
    return parse_numeric_report(FIXTURES / f"{case}.txt", report_format(spec), ADD_LIST).data


def _same_report(a: str, b: str) -> bool:
    return bool(MANIFEST[a]["full_sha256"] == MANIFEST[b]["full_sha256"])


@pytest.mark.parametrize("case", SAME_NUMBERS)
def test_every_format_reads_to_the_default_numbers(case: str) -> None:
    expected = _read("A_default")
    assert expected.shape == (39, 7)
    # 40 lines are 39 rows behind a header, 40 without one.
    np.testing.assert_array_equal(_read(case)[:39], expected)


def test_low_precision_reads_its_own_numbers() -> None:
    np.testing.assert_allclose(_read("J_lowprec"), _read("A_default"), rtol=1e-5)
    assert not np.array_equal(_read("J_lowprec"), _read("A_default"))


def test_a_narrow_column_without_headers_is_refused_with_its_cause() -> None:
    with pytest.raises(GmatRunError, match=r"values ran together.*ColumnWidth = 10"):
        _read("O_noheader_narrow")


@pytest.mark.parametrize(
    "case, written",
    [
        # Each pair below is what R2026a wrote for the script text on the left.
        ("A_default", " "),
        ("C_comma", ","),
        ("K_tab", "\\"),
        ("P_tab_character", "\t"),
        ("R_two_characters", ";"),
        ("S_empty", " "),
        ("T_no_delimiter", " "),
        ("U_unquoted_comma", ","),
    ],
)
def test_the_delimiter_is_the_character_gmat_writes(case: str, written: str) -> None:
    assert _spec(case).delimiter == written


def test_the_delimiter_rules_are_what_gmat_wrote() -> None:
    assert _same_report("S_empty", "A_default")
    assert _same_report("T_no_delimiter", "A_default")
    assert _same_report("U_unquoted_comma", "C_comma")
    assert not _same_report("K_tab", "P_tab_character")
    first = (FIXTURES / "K_tab.txt").read_bytes().splitlines()[1]
    assert b"\t" not in first and b"\\\\\\" in first
    first = (FIXTURES / "R_two_characters.txt").read_bytes().splitlines()[1]
    assert b";" in first and b"," not in first


def test_without_headers_fixed_width_ignores_the_delimiter() -> None:
    assert _same_report("G_noheader_comma", "B_noheaders")
    assert _same_report("N_noheader_tab", "B_noheaders")
    assert ReportFormat(write_headers=False, delimiter=",").separator is None


def test_without_fixed_width_the_delimiter_is_kept() -> None:
    for case in ("M_noheader_comma_nofixed", "W_noheader_semicolon_nofixed"):
        delimiter = _spec(case).delimiter
        assert delimiter.encode() in (FIXTURES / f"{case}.txt").read_bytes().splitlines()[0]
        assert report_format(_spec(case)).separator == delimiter


def test_double_quotes_are_refused_as_gmat_refuses_them() -> None:
    # R2026a: '"","" is not a valid RHS of assignment'.
    with pytest.raises(ParseError, match="double-quoted"):
        _report_file_spec("R", {"Delimiter": '","'})


@pytest.mark.parametrize("delimiter", ["-", "+", ".", "e", "x", "5", "_", "%"])
def test_a_delimiter_inside_numbers_or_names_cannot_be_read(delimiter: str) -> None:
    fmt = ReportFormat(delimiter=delimiter, fixed_width=False)
    assert fmt.unreadable() is not None
    with pytest.raises(GmatRunError, match="does not split"):
        fmt.split("0-4766.09--4264.33")


def test_a_header_less_report_needs_the_add_list() -> None:
    with pytest.raises(GmatRunError, match="Add list"):
        parse_numeric_report(FIXTURES / "B_noheaders.txt", report_format(_spec("B_noheaders")))


def test_the_default_format_reads_as_before() -> None:
    assert parse_numeric_report(FIXTURES / "A_default.txt").headers == ADD_LIST


def _script_with(tmp_path: Path, script: Path, replace: dict[str, str | None]) -> Path:
    """`script` with its one ReportFile's fields set, or dropped where None."""
    (name,) = [spec.name for spec in report_file_specs(script)]
    lines = script.read_text(encoding="utf-8").splitlines(keepends=True)
    present = {line.split("=", 1)[0].strip() for line in lines}
    out = []
    for line in lines:
        field = line.split("=", 1)[0].strip()
        key = field.removeprefix(f"{name}.")
        if field.startswith(f"{name}.") and key in replace:
            if replace[key] is not None:
                out.append(f"{field} = {replace[key]};\n")
            continue
        out.append(line)
        if line.startswith(f"Create ReportFile {name};"):
            out += [
                f"{name}.{key} = {value};\n"
                for key, value in replace.items()
                if value is not None and f"{name}.{key}" not in present
            ]
    path = tmp_path / script.name
    path.write_text("".join(out), encoding="utf-8")
    return path


def test_the_parser_carries_the_format(tmp_path: Path, different_plane_script: Path) -> None:
    plain = parse_gmat_script(different_plane_script, OrbitWinConfig())[0].report_files
    assert [spec.delimiter for spec in plain] == [" "] * len(plain)
    edited = _script_with(
        tmp_path,
        different_plane_script,
        {"Delimiter": "'\\t'", "WriteHeaders": "false", "FixedWidth": "false",
         "ColumnWidth": "30", "LeftJustify": "Off", "ZeroFill": "On", "WriteReport": "false"},
    )
    specs = parse_gmat_script(edited, OrbitWinConfig())[0].report_files
    assert specs == report_file_specs(edited)
    (spec,) = specs
    assert (spec.delimiter, spec.write_headers, spec.fixed_width, spec.column_width) == (
        "\\", False, False, 30
    )
    assert (spec.left_justify, spec.zero_fill, spec.write_report) == (False, True, False)


def test_a_missing_delimiter_is_a_space(different_plane_script: Path) -> None:
    # The synthetic scripts have no Delimiter line; #70 read that as ''.
    assert "Delimiter" not in different_plane_script.read_text(encoding="utf-8")
    (spec,) = report_file_specs(different_plane_script)
    assert spec.delimiter == " "
    assert report_format(spec).split("1.5   2.5 ") == ["1.5", "2.5"]


def test_a_duplicate_report_field_is_refused(tmp_path: Path, different_plane_script: Path) -> None:
    text = different_plane_script.read_text(encoding="utf-8")
    name = report_file_specs(different_plane_script)[0].name
    marker = f"{name}.Precision"
    head, tail = text.split(marker, 1)
    path = tmp_path / "dup.script"
    path.write_text(head + f"{name}.Delimiter = ',';\n{name}.Delimiter = ';';\n" + marker + tail,
                    encoding="utf-8")
    with pytest.raises(ParseError, match="Duplicate assignment"):
        report_file_specs(path)


@pytest.mark.parametrize("width", ["0", "2.5", "-3"])
def test_column_width_must_be_a_positive_integer(width: str) -> None:
    with pytest.raises(ParseError, match="positive integer"):
        _report_file_spec("R", {"ColumnWidth": width})


def test_a_report_field_the_parser_cannot_read_is_refused(
    tmp_path: Path, different_plane_script: Path
) -> None:
    # GMAT accepts a line without its semicolon; parse_gmat_script refuses it.
    text = different_plane_script.read_text(encoding="utf-8")
    path = tmp_path / "nosemi.script"
    path.write_text(
        text.replace("OfficialReport.Precision = 16;", "OfficialReport.Precision = 16\n"
                     "OfficialReport.Delimiter = ','"),
        encoding="utf-8",
    )
    with pytest.raises(ParseError, match="cannot read"):
        report_file_specs(path)


def test_column_width_is_blamed_only_for_values_run_together(tmp_path: Path) -> None:
    fmt = ReportFormat(write_headers=False)
    fused = tmp_path / "fused.txt"
    fused.write_text("1.5 2.5\n1.5 4766.09-4264.33\n", encoding="utf-8")
    with pytest.raises(GmatRunError, match="ColumnWidth"):
        read_rows(fused, fmt)
    other = tmp_path / "other.txt"
    other.write_text("1.5 2.5\n1.5 garbage\n", encoding="utf-8")
    with pytest.raises(GmatRunError) as caught:
        read_rows(other, fmt)
    assert "ColumnWidth" not in str(caught.value)


def test_a_text_column_must_be_a_gregorian_epoch(tmp_path: Path) -> None:
    names = ("A.ElapsedSecs", "A.UTCGregorian", "A.X")
    good = tmp_path / "good.txt"
    good.write_text("0 09 Nov 2026 02:00:00.000 1.5\n", encoding="utf-8")
    assert read_rows(good, ReportFormat(write_headers=False), names)[1].shape == (1, 3)
    bad = tmp_path / "bad.txt"
    bad.write_text("0 1.0 2.0 3.0 4.0 1.5\n", encoding="utf-8")
    with pytest.raises(GmatRunError, match="not numeric"):
        read_rows(bad, ReportFormat(write_headers=False), names)


def test_a_short_name_is_read_as_earthmj2000eq() -> None:
    from orbitwin.report_format import position_frames

    assert position_frames(["A.X", "A.Y", "A.Z", "A.EarthFixed.X"], "A") == {
        "EarthMJ2000Eq": (0, 1, 2)
    }


def test_positive_values_run_together_are_explained(tmp_path: Path) -> None:
    # O_noheader_narrow, R2026a: `31480.000428638682360.467326019827`.
    fused = tmp_path / "fused.txt"
    fused.write_text("1.5 2.5\n1.5 31480.000428638682360.467326019827\n", encoding="utf-8")
    with pytest.raises(GmatRunError, match="values ran together"):
        read_rows(fused, ReportFormat(write_headers=False))


@pytest.mark.parametrize(
    "name", ["SC.ElapsedSecs", "SC.ElapsedDays", "SC.A1ModJulian", "SC.UTCModJulian",
             "SC.TAIModJulian", "SC.TDBModJulian"]
)
def test_every_numeric_epoch_is_an_epoch(name: str) -> None:
    from orbitwin.report_format import is_epoch_parameter

    assert is_epoch_parameter(name)
    assert not is_epoch_parameter(name.replace("SC.", "SC.Earth.") + "Rate")


def test_a_percent_line_is_skipped(tmp_path: Path) -> None:
    report = tmp_path / "r.txt"
    report.write_text("A B\n1.0 2.0\n% a note\n3.0 4.0\n", encoding="utf-8")
    assert read_rows(report)[1].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_a_name_of_four_parts_is_not_a_position() -> None:
    from orbitwin.report_format import position_frames

    assert position_frames(["A.B.C.X", "A.B.C.Y", "A.B.C.Z"], "A") == {}


def test_a_delimiter_of_two_characters_is_unreadable() -> None:
    assert ReportFormat(delimiter=";;", fixed_width=False).unreadable() is not None


def test_a_mission_sequence_report_field_is_refused(
    tmp_path: Path, different_plane_script: Path
) -> None:
    text = different_plane_script.read_text(encoding="utf-8")
    path = tmp_path / "mission.script"
    path.write_text(text.replace("BeginMissionSequence;",
                                 "BeginMissionSequence;\nOfficialReport.WriteHeaders = false;"),
                    encoding="utf-8")
    with pytest.raises(ParseError, match="in the mission sequence"):
        report_file_specs(path)


def test_header_and_row_widths_must_agree(tmp_path: Path) -> None:
    report = tmp_path / "r.txt"
    report.write_text("A B C\n1.0 2.0\n3.0 4.0\n", encoding="utf-8")
    with pytest.raises(GmatRunError, match="2 values a row for 3 headers"):
        parse_numeric_report(report)


def test_solver_iterations_is_read_from_the_script(
    tmp_path: Path, different_plane_script: Path
) -> None:
    edited = _script_with(tmp_path, different_plane_script, {"SolverIterations": "All"})
    (spec,) = report_file_specs(edited)
    assert spec.solver_iterations == "All"
    assert parse_gmat_script(edited, OrbitWinConfig())[0].report_files == (spec,)
