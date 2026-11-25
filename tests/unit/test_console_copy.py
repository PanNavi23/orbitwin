"""The console copy differs from the submitted bytes only by whole plot lines left out.

GmatConsole refuses a script creating an OpenFrames object, and a GroundTrack
created before a ReportFile left it writing no report. Under gmat.console_copy
certify runs a copy less those lines, and prove_console_copy refuses anything
else. Only the GMAT GUI reads the lines left out, and the checker does not judge
whether it takes them: they must be the official template's own, verbatim and in
order (assert_plot_lines_unchanged). tests/fixtures/console_copy/gui_oracle_cases.json
holds what R2026a's GUI said of edits to them, run as a parse oracle (a test
oracle only); the ones it refused and the checker takes are marked checker_differs.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import GUI_ORACLE, bare_environment, gui_oracle_script, organizer_scripts

from orbitwin.console_copy import (
    COPIED_TYPES,
    assert_plot_lines_unchanged,
    build_console_copy,
    prove_console_copy,
    write_console_copy,
)
from orbitwin.errors import OrbitWinError, ParseError, PolicyError
from orbitwin.gmat_parser import GmatScriptDocument, created_types, report_file_specs

ROOT = Path(__file__).resolve().parents[2]
R1_PLOTS = ROOT / "tests" / "fixtures" / "report_bind" / "R1_openframes.script"
WORKSHOP = ROOT / "examples" / "two_burn_rendezvous.script"
ORACLE = GUI_ORACLE
#: A line the copy leaves out, as the templates write them.
PLOT_LINE = re.compile(
    r"^(?:Create (?:OpenFrames\w+|GroundTrack) |"
    r"(?:DefaultOrbitView|DefaultOrbitView_View|DefaultGroundTrackPlot)\.)"
)
BEGIN = "BeginMissionSequence;"


def _script(tmp_path: Path, text: str, name: str = "t.script") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode("ascii"))
    return path


def _r1(edit: tuple[str, str] | None = None, *, before_begin: str = "",
        after_begin: str = "") -> str:
    text = R1_PLOTS.read_text(encoding="ascii")
    if edit is not None:
        assert edit[0] in text, edit[0]
        text = text.replace(edit[0], edit[1], 1)
    return text.replace(BEGIN + "\n", before_begin + BEGIN + "\n" + after_begin, 1)


def _copy(tmp_path: Path, text: str) -> tuple[Path, Path]:
    script = _script(tmp_path, text)
    path, _ = write_console_copy(script, tmp_path / "copy")
    return script, path


#: Plot values as a GMAT GUI saves a view moved by hand: negative, and with negative
#: exponents. Our templates write round numbers, which leave _STRICT_ASSIGN_RE's value
#: grammar untested on these.
GUI_SAVED_VALUES = {
    "DefaultOrbitView.UpperLeft = [ 0.05 0.05 ];":
        "DefaultOrbitView.UpperLeft = [ 0.04918032786885246 -0.0125786163522013 ];",
    "DefaultGroundTrackPlot.UpperLeft = [ 0.5 0.05 ];":
        "DefaultGroundTrackPlot.UpperLeft = [ 0.5009107468123861 -0.1446540880503145 ];",
    "DefaultOrbitView_View.CurrentEye = [ 30000 0 0 ];":
        "DefaultOrbitView_View.CurrentEye = [ 21650.63509461097 -12500.00000000001 "
        "2.273736754432321e-12 ];",
    "DefaultOrbitView_View.CurrentCenter = [ 0 0 0 ];":
        "DefaultOrbitView_View.CurrentCenter = [ -2.273736754432321e-12 4.547473508864641e-13 "
        "1.13686837721616e-12 ];",
    "DefaultOrbitView_View.CurrentUp = [ 0 0 1 ];":
        "DefaultOrbitView_View.CurrentUp = [ -0.4330127018922193 -0.25 0.8660254037844387 ];",
}


def _gui_saved(tmp_path: Path) -> Path:
    text = WORKSHOP.read_text(encoding="ascii")
    for old, new in GUI_SAVED_VALUES.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return _script(tmp_path, text, "gui_saved.script")


@pytest.mark.parametrize("template", [R1_PLOTS, WORKSHOP, "crlf", "gui-saved",
                                      *organizer_scripts()],
                         ids=lambda p: getattr(p, "stem", str(p)))
def test_the_templates_copies_leave_out_their_plots_and_nothing_else(
    template: Path | str, tmp_path: Path
) -> None:
    """Ours, and the organizer's templates where this tree carries them."""
    if template == "crlf":
        # The two-burn template as a GUI on Windows saves it.
        template = _script(tmp_path, WORKSHOP.read_text(encoding="ascii").replace("\n", "\r\n"),
                           "crlf.script")
    elif template == "gui-saved":
        template = _gui_saved(tmp_path)
    assert isinstance(template, Path)
    path, copy = write_console_copy(template, tmp_path / "copy")
    lines = template.read_bytes().splitlines(keepends=True)
    kept = [line for number, line in enumerate(lines, 1) if number not in copy.deleted_lines]
    assert path.read_bytes() == b"".join(kept) == copy.script_bytes
    # Every line left out is a plot's, and every plot line is left out.
    assert all(PLOT_LINE.match(lines[n - 1].decode()) for n in copy.deleted_lines)
    assert not any(PLOT_LINE.match(line.decode()) for line in kept)
    plots = {n: k for n, k in created_types(GmatScriptDocument.read(template)).items()
             if k in COPIED_TYPES}
    assert sorted(item.split(" ", 1)[1] for item in copy.removed) == sorted(plots)
    assert report_file_specs(path) == report_file_specs(template)
    # CRLF stays CRLF: a template saved on Windows is one.
    assert (b"\r\n" in path.read_bytes()) == (b"\r\n" in template.read_bytes())


def _tampered(honest: bytes, how: str) -> bytes:
    lines = honest.splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line.startswith(b"Propagate "))
    report = next(i for i, line in enumerate(lines) if b"ReportFile1.Filename" in line)
    if how == "a digit":
        lines[index] = lines[index].replace(b"6711.667", b"6711.668", 1)
    elif how == "a kept line dropped":
        del lines[report]
    elif how == "a line ending":
        lines[report] = lines[report].replace(b"\n", b"\r\n")
    elif how == "a line added":
        lines.insert(report, b"ReportFile1.Precision = 17;\n")
    elif how == "a comment added":
        lines.insert(report, b"% a note\n")
    elif how == "two lines swapped":
        lines[report], lines[report + 1] = lines[report + 1], lines[report]
    elif how == "a trailing line":
        lines.append(b"% after\n")
    elif how == "a plot line kept":
        lines.insert(report - 2, b"Create OpenFramesView DefaultOrbitView_View;\n")
    return b"".join(lines)


@pytest.mark.parametrize("how", ["a digit", "a kept line dropped", "a line ending", "a line added",
                                 "a comment added", "two lines swapped", "a trailing line",
                                 "a plot line kept"])
def test_a_copy_differing_by_anything_else_is_refused(how: str, tmp_path: Path) -> None:
    script, honest = _copy(tmp_path, R1_PLOTS.read_text(encoding="ascii"))
    forged = tmp_path / "forged.script"
    forged.write_bytes(_tampered(honest.read_bytes(), how))
    with pytest.raises(PolicyError, match=r"differs from|has lines after"):
        prove_console_copy(script, forged)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        # A kept resource naming a plot.
        (_r1(before_begin="Create CoordinateSystem CS;\nCS.Origin = DefaultOrbitView;\n"),
         "names a plot the console copy leaves out"),
        # GMAT copies one object onto another: the alias stays live.
        (_r1(before_begin="Create OpenFramesView V2;\nV2 = DefaultOrbitView_View;\n"),
         "names a plot"),
        # Two statements on one line are not one it can read.
        (_r1(before_begin="Create OpenFramesView V2; V2.FOVy = 45;\n"), "names a plot"),
        # GMAT's Create is case-sensitive; this is no Create to it.
        (_r1(before_begin="create OpenFramesView V2;\n"), "names a plot"),
        (_r1(before_begin="Create OpenFramesView V2 SC_Alien;\n"), "creates SC_Alien again"),
        (_r1(before_begin="Create OpenFramesView DefaultOrbitView_View;\n"),
         "creates DefaultOrbitView_View again"),
        # A field on a plot GmatConsole runs, naming one the copy leaves out.
        (_r1(before_begin="Create XYPlot XY;\nXY.XVariable = DefaultOrbitView;\n"),
         "names a plot"),
        # Mission commands.
        (_r1(after_begin="Toggle DefaultOrbitView ReportFile1 Off;\n"), "plot command naming only"),
        (_r1(after_begin="Toggle 'pause' DefaultOrbitView Off;\n"), "plot command naming only"),
        (_r1(after_begin="Report ReportFile1 DefaultOrbitView;\n"), "plot command naming only"),
        (_r1(("BeginMissionSequence;", "BeginMissionSequence; Toggle DefaultOrbitView Off;")),
         "plot command naming only"),
    ],
    ids=["resource field", "object alias", "two statements", "lowercase create",
         "create of a craft", "created twice", "XYPlot field", "mixed Toggle",
         "labelled Toggle", "Report", "on the BeginMissionSequence line"],
)
def test_a_live_line_that_names_a_plot_gets_no_copy(
    text: str, message: str, tmp_path: Path
) -> None:
    with pytest.raises(PolicyError, match=message):
        _copy(tmp_path, text)


@pytest.mark.parametrize(
    ("after", "left_out"),
    [
        ("Toggle DefaultOrbitView Off;\n", True),
        ("Toggle DefaultOrbitView DefaultOrbitView_View On;;\n", True),
        ("PenUp DefaultOrbitView;\n", True),
        # A Toggle of the ReportFile names no plot, so it stays.
        ("Toggle ReportFile1 Off;\n", False),
    ],
)
def test_a_plot_command_naming_only_plots_is_left_out(
    after: str, left_out: bool, tmp_path: Path
) -> None:
    _, path = _copy(tmp_path, _r1(after_begin=after))
    assert (after.encode() in path.read_bytes()) is not left_out
    assert BEGIN.encode() in path.read_bytes()


def test_comment_lines_are_kept_and_a_plot_lines_comment_goes_with_it(tmp_path: Path) -> None:
    text = _r1(("DefaultOrbitView.MSAASamples = 2;", "DefaultOrbitView.MSAASamples = 2; % two"),
               before_begin="% DefaultOrbitView.Add = {Nothing};\n")
    _, path = _copy(tmp_path, text)
    copied = path.read_text(encoding="ascii")
    assert "% DefaultOrbitView.Add = {Nothing};\n" in copied
    assert "% two" not in copied


def test_a_continuation_is_refused_before_any_copy(tmp_path: Path) -> None:
    text = _r1(("DefaultOrbitView.Add = {SC_Human, SC_Alien, Earth};",
                "DefaultOrbitView.Add = {SC_Human, ...\n   SC_Alien, Earth};"))
    with pytest.raises(ParseError, match=r"\.\.\."):
        _copy(tmp_path, text)


def test_a_quoted_percent_is_text_on_a_plot_line(tmp_path: Path) -> None:
    """Item 6: GMAT's first % inside quotes is text (#173's line model; the GUI built
    vector_label_percent), and a comment after one is refused before any copy."""
    vector = ("Create OpenFramesVector V1;\nV1.SourceObject = SC_Human;\n"
              "V1.DestinationObject = SC_Alien;\nV1.VectorLabel = 'a%b';\n")
    _, path = _copy(tmp_path, _r1(before_begin=vector))
    assert b"V1" not in path.read_bytes()
    with pytest.raises(ParseError, match="first % is inside quotes"):
        _copy(tmp_path, _r1(before_begin=vector.replace("'a%b';", "'a%b'; % note")))


def test_a_double_semicolon_is_gmats_and_left_out(tmp_path: Path) -> None:
    """Item 11: R2026a and the GUI take `;;` (oracle double_semicolon)."""
    text = _r1(("DefaultOrbitView.Axes = On;", "DefaultOrbitView.Axes = On;;"))
    text = text.replace("Create OpenFramesView DefaultOrbitView_View;",
                        "Create OpenFramesView DefaultOrbitView_View;;", 1)
    _, path = _copy(tmp_path, text)
    assert b"OpenFrames" not in path.read_bytes()


def test_other_plots_stay_for_gmatconsole_to_run(tmp_path: Path) -> None:
    text = _r1(before_begin="Create XYPlot XY;\nXY.XVariable = SC_Human.ElapsedSecs;\n")
    _, path = _copy(tmp_path, text)
    assert b"Create XYPlot XY;\nXY.XVariable" in path.read_bytes()


def test_a_script_without_those_plots_has_no_copy(tmp_path: Path) -> None:
    plain = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
    assert build_console_copy(GmatScriptDocument.read(plain)) is None
    with pytest.raises(PolicyError, match="creates no OpenFrames or GroundTrack"):
        write_console_copy(plain, tmp_path)
    assert not list(tmp_path.iterdir()) or not list((tmp_path).glob("*_console_copy*"))


def test_a_copy_that_fails_its_proof_is_not_left_on_disk(tmp_path: Path) -> None:
    with pytest.raises(PolicyError):
        _copy(tmp_path, _r1(after_begin="Toggle DefaultOrbitView ReportFile1 Off;\n"))
    assert not list((tmp_path / "copy").iterdir())


def _oracle_cases() -> list[dict[str, object]]:
    return json.loads(ORACLE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _oracle_cases(), ids=lambda case: str(case["name"]))
def test_the_checker_does_not_judge_what_the_gui_takes(
    case: dict[str, object], tmp_path: Path
) -> None:
    """Items 1, 4 and 7, as R2026a's GUI settled them on our two-burn template and
    R1's: the copy is proven of any plot line, and each one the GUI refused is a
    documented checker_differs. Identity with the official template is what keeps
    such a line out of an upload the organizer's own template does not have."""
    text = gui_oracle_script(json.loads(ORACLE.read_text(encoding="utf-8")), case)
    gui_took = case["gui"] == "ran to the end"
    try:
        _copy(tmp_path, text)
        took = True
    except OrbitWinError:
        took = False
    if "checker_differs" in case:
        assert took and not gui_took, case["checker_differs"]
    else:
        assert took is gui_took


@pytest.mark.parametrize(
    ("pattern", "value"),
    [
        ("_STRICT_ASSIGN_RE", "DefaultOrbitView.DrawLineWidth = [ " + "1 " * 20000 + "x ]"),
        ("_STRICT_ASSIGN_RE", "DefaultOrbitView.X = " + "'a'" * 20000 + "'"),
        ("_STRICT_CREATE_RE", "Create OpenFramesView " + "V " * 20000 + "x"),
        ("_STRICT_CREATE_RE", "Create OpenFramesView " + " , " * 20000 + ";"),
        ("_STRICT_COMMAND_RE", "PenUp " + "P " * 20000 + "%"),
    ],
)
def test_a_long_plot_line_that_does_not_match_fails_fast(pattern: str, value: str) -> None:
    """Item 2: no separator can match empty or two ways, so the engine never tries the
    splits of a long line; a separate process so a hang cannot stall the suite."""
    code = (
        "import sys; import orbitwin.console_copy as c; "
        f"assert c.{pattern}.fullmatch(sys.stdin.read()) is None"
    )
    start = time.monotonic()
    done = subprocess.run([sys.executable, "-c", code], input=value, text=True, timeout=20,
                          capture_output=True, check=False,
                          env=bare_environment(PYTHONPATH=str(ROOT / "src")))
    assert done.returncode == 0, done.stderr
    assert time.monotonic() - start < 5.0


@pytest.mark.parametrize("how", ["a value", "a line ending", "a line dropped", "a line added",
                                 "two swapped", "a Toggle added"])
def test_the_submitted_scripts_plot_lines_must_be_the_officials(
    how: str, tmp_path: Path
) -> None:
    """The GUI's verdict on the upload is the template's only if every plot line is the
    template's, byte for byte and in order."""
    official = _script(tmp_path, R1_PLOTS.read_text(encoding="ascii"), "official.script")
    lines = official.read_text(encoding="ascii").splitlines(keepends=True)
    first = next(i for i, line in enumerate(lines) if line.startswith("DefaultOrbitView."))
    if how == "a value":
        lines[first] = lines[first].replace("Current", "All")
    elif how == "a line ending":
        lines[first] = lines[first].replace("\n", "\r\n")
    elif how == "a line dropped":
        del lines[first]
    elif how == "a line added":
        lines.insert(first, "DefaultOrbitView.Minimized = false;\n")
    elif how == "two swapped":
        lines[first], lines[first + 1] = lines[first + 1], lines[first]
    else:
        index = lines.index("BeginMissionSequence;\n")
        lines.insert(index + 1, "Toggle DefaultOrbitView Off;\n")
    submitted = _script(tmp_path, "".join(lines), "submitted.script")
    assert_plot_lines_unchanged(GmatScriptDocument.read(official),
                                GmatScriptDocument.read(official))
    with pytest.raises(PolicyError, match=r"does not carry official\.script's plot lines"):
        assert_plot_lines_unchanged(GmatScriptDocument.read(official),
                                    GmatScriptDocument.read(submitted))


def test_the_policy_guard_holds_the_plot_lines(tmp_path: Path) -> None:
    import dataclasses

    from orbitwin.config import load_config
    from orbitwin.script_renderer import assert_final_script_policy

    official = _script(tmp_path, R1_PLOTS.read_text(encoding="ascii"), "official.script")
    submitted = _script(tmp_path, R1_PLOTS.read_text(encoding="ascii").replace(
        "DefaultOrbitView.Axes = On;", "DefaultOrbitView.Axes = Off;"), "submitted.script")
    raw = load_config(ROOT / "config" / "ranking.yaml")
    config = dataclasses.replace(raw, policy=dataclasses.replace(raw.policy, horizon_s=6711.667))
    with pytest.raises(PolicyError, match=r"plot line 23 is 'DefaultOrbitView\.Axes = Off;\\n'"):
        assert_final_script_policy(official, submitted, config)


def test_a_changed_line_ending_shows_in_the_refusal(tmp_path: Path) -> None:
    """Low-2 of the delta review of #189: a stripped line hid the difference."""
    official = _script(tmp_path, R1_PLOTS.read_text(encoding="ascii"), "official.script")
    submitted = _script(tmp_path, R1_PLOTS.read_text(encoding="ascii").replace(
        "DefaultOrbitView.Axes = On;\n", "DefaultOrbitView.Axes = On; \r\n"), "submitted.script")
    with pytest.raises(PolicyError, match=re.escape(
            "is 'DefaultOrbitView.Axes = On; \\r\\n' where the official script has "
            "'DefaultOrbitView.Axes = On;\\n'")):
        assert_plot_lines_unchanged(GmatScriptDocument.read(official),
                                    GmatScriptDocument.read(submitted))


def test_a_plot_line_with_the_gmat_prefix_is_held_too(tmp_path: Path) -> None:
    """N10 of the delta review of #189: `GMAT X.F = v;` is a plot line like any other."""
    text = R1_PLOTS.read_text(encoding="ascii").replace(
        "DefaultOrbitView.Axes = On;", "GMAT DefaultOrbitView.Axes = On;")
    official = _script(tmp_path, text, "official.script")
    submitted = _script(tmp_path, text.replace("GMAT DefaultOrbitView.Axes = On;",
                                               "GMAT DefaultOrbitView.Axes = Off;"), "sub.script")
    with pytest.raises(PolicyError, match="Axes = Off"):
        assert_plot_lines_unchanged(GmatScriptDocument.read(official),
                                    GmatScriptDocument.read(submitted))
