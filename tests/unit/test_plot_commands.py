"""A Toggle, PenUp, PenDown, MarkPoint or ClearPlot of plots is read; any other is refused.

A day template saved from the GUI may turn a plot on or off, lift its pen, mark
a point or clear it between segments. That changes no flight and no report, so
the parser takes it, the private copies comment it out with its plot, and the
console copy leaves it out. One of anything else, a ReportFile above all, may
change what the uploaded report holds and is refused, as is any form R2026a's
GUI does not build (tests/fixtures/console_copy/gui_oracle_cases.json,
"command_cases").
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from conftest import GUI_ORACLE, gui_oracle_script, openframes_errors

from orbitwin.config import load_config
from orbitwin.console_copy import COPIED_TYPES, write_console_copy
from orbitwin.diagnostic import NEUTRALISED_PREFIX, add_diagnostic_report
from orbitwin.errors import OrbitWinError, ParseError, PolicyError
from orbitwin.gmat_parser import (
    GmatScriptDocument,
    _code_part,
    created_types,
    mission_schedule,
    parse_gmat_script,
    plot_command,
)
from orbitwin.gmat_runner import EXIT_STATUS_ERROR
from orbitwin.models import GmatRunResult
from orbitwin.verify import plot_refusal

ROOT = Path(__file__).resolve().parents[2]
R1_PLOTS = ROOT / "tests" / "fixtures" / "report_bind" / "R1_openframes.script"
R6 = ROOT / "examples" / "ranking" / "R6_InsertedLate_Aligned.script"
ORACLE = GUI_ORACLE
BEGIN = "BeginMissionSequence;\n"
PROPAGATE = ("Propagate DefaultProp(SC_Alien, SC_Human) {SC_Alien.ElapsedSecs = 6711.667, "
             "SC_Human.ElapsedSecs = 6711.667};\n")


def _r1(tmp_path: Path, mission: str) -> Path:
    text = R1_PLOTS.read_text(encoding="ascii")
    assert text.count(BEGIN + PROPAGATE) == 1
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "r1_toggle.script"
    path.write_text(text.replace(BEGIN + PROPAGATE, BEGIN + mission), encoding="ascii")
    return path


def _config():  # type: ignore[no-untyped-def]
    raw = load_config(ROOT / "config" / "ranking.yaml")
    return dataclasses.replace(raw, policy=dataclasses.replace(raw.policy, horizon_s=6711.667))


#: Two slots, so a Toggle can sit between the segments and the Maneuvers.
BURNS = "".join(
    f"Create ImpulsiveBurn IB{n};\nIB{n}.CoordinateSystem = Local;\nIB{n}.Origin = Earth;\n"
    f"IB{n}.Axes = VNB;\nIB{n}.Element1 = 0;\nIB{n}.Element2 = 0;\nIB{n}.Element3 = 0;\n"
    f"IB{n}.DecrementMass = false;\n"
    for n in (1, 2)
)


def _segment(seconds: str) -> str:
    return PROPAGATE.replace("6711.667", seconds)


SPLIT = (_segment("1000") + "Toggle DefaultOrbitView Off;\nManeuver IB1(SC_Human);\n"
         + _segment("1000") + "Maneuver IB2(SC_Human);\nToggle DefaultOrbitView On;\n"
         + _segment("4711.667"))


def _slotted(tmp_path: Path, mission: str) -> Path:
    path = _r1(tmp_path, mission)
    text = path.read_text(encoding="ascii")
    path.write_text(text.replace("Create OpenFramesInterface", BURNS + "Create OpenFramesInterface",
                                 1), encoding="ascii")
    return path


@pytest.mark.parametrize("mission", [
    "Toggle DefaultOrbitView Off;\n" + PROPAGATE,
    PROPAGATE + "Toggle DefaultOrbitView Off;\n",
    "Toggle DefaultOrbitView Off;; % quiet\n" + PROPAGATE,
    SPLIT,
], ids=["before", "after", "semicolons and a comment", "between segments"])
def test_a_toggle_of_plots_flies_as_the_script_without_it(
    mission: str, tmp_path: Path
) -> None:
    script = _slotted(tmp_path, mission)
    plain = _slotted(tmp_path / "plain", "".join(
        line for line in mission.splitlines(keepends=True) if not line.startswith("Toggle")))
    with_toggle, _ = parse_gmat_script(script, _config())
    without, _ = parse_gmat_script(plain, _config())
    assert [(s.duration_s, s.propagator_names) for s in with_toggle.propagate_segments] == [
        (s.duration_s, s.propagator_names) for s in without.propagate_segments]
    assert [slot.name for slot in with_toggle.burn_slots] == [
        slot.name for slot in without.burn_slots]
    def flown(path: Path) -> tuple[object, ...]:
        schedule = mission_schedule(path)
        return (schedule.end_s, schedule.alignment_craft, [
            (m.burn, m.time_s, m.components_km_s, m.frame) for m in schedule.maneuvers])

    assert flown(script) == flown(plain)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("Toggle ReportFile1 Off;", r"ReportFile1 \(ReportFile\)"),
        ("Toggle SC_Human Off;", r"SC_Human \(Spacecraft\)"),
        ("Toggle DefaultOrbitView ReportFile1 Off;", r"ReportFile1 \(ReportFile\)"),
        ("Toggle NoSuchPlot Off;", "NoSuchPlot \\(nothing the script creates\\)"),
        # Not a subscriber: the GUI refuses it ("not an object of type Subscriber").
        ("Toggle DefaultOrbitView_View Off;", r"DefaultOrbitView_View \(OpenFramesView\)"),
        # Forms the GUI refuses.
        ("Toggle DefaultOrbitView, DefaultOrbitView Off;", "is not one the parser reads"),
        ("Toggle DefaultOrbitView off;", "is not one the parser reads"),
        ("Toggle DefaultOrbitView;", "is not one the parser reads"),
        ("toggle DefaultOrbitView Off;", "is not one the parser reads"),
        # One it builds, but no saved script writes; refused here only.
        ("Toggle DefaultOrbitView Off", "is not one the parser reads"),
        ("Toggle 'pause' DefaultOrbitView Off;", "is not one the parser reads"),
        # The same of the other plot commands.
        ("PenUp ReportFile1;", r"ReportFile1 \(ReportFile\)"),
        ("MarkPoint SC_Human;", r"SC_Human \(Spacecraft\)"),
        ("PenDown DefaultOrbitView_View;", r"DefaultOrbitView_View \(OpenFramesView\)"),
        ("ClearPlot NoSuchPlot;", "NoSuchPlot \\(nothing the script creates\\)"),
        ("PenUp;", "is not one the parser reads"),
        ("PenUp DefaultOrbitView, DefaultOrbitView;", "is not one the parser reads"),
        ("penup DefaultOrbitView;", "is not one the parser reads"),
        ("PenUp DefaultOrbitView", "is not one the parser reads"),
        ("PenUp DefaultOrbitView Off;", r"Off \(nothing the script creates\)"),
        # R2026a's Sandbox refuses these of anything but an XYPlot, on a run to the end.
        ("MarkPoint DefaultOrbitView;",
         r"DefaultOrbitView \(OpenFramesInterface\): only a MarkPoint of XYPlots"),
        ("ClearPlot DefaultOrbitView;", r"only a ClearPlot of XYPlots"),
    ],
)
def test_any_other_plot_command_is_refused(line: str, message: str, tmp_path: Path) -> None:
    script = _r1(tmp_path, line + "\n" + PROPAGATE)
    with pytest.raises(ParseError, match=message):
        parse_gmat_script(script, _config())
    with pytest.raises(ParseError, match=message):
        mission_schedule(script)


@pytest.mark.parametrize("line", [
    "PenUp DefaultOrbitView;", "PenDown DefaultOrbitView;;  % between segments",
])
def test_the_other_plot_commands_are_read_and_left_out(line: str, tmp_path: Path) -> None:
    script = _slotted(tmp_path, SPLIT.replace("Toggle DefaultOrbitView Off;", line))
    plain = _slotted(tmp_path / "plain", SPLIT.replace("Toggle DefaultOrbitView Off;\n", ""))
    assert mission_schedule(script).end_s == mission_schedule(plain).end_s
    parse_gmat_script(script, _config())
    path, _ = write_console_copy(script, tmp_path / "copy")
    assert line not in path.read_text(encoding="ascii")
    scenario, document = parse_gmat_script(script, _config())
    sibling = add_diagnostic_report(document, scenario).script_bytes.decode("ascii")
    assert f"{NEUTRALISED_PREFIX}{line}\n" in sibling


def test_gmats_record_for_a_pen_command_is_the_plots(tmp_path: Path) -> None:
    """R2026a: 'Nonexistent Subscriber "DefaultOrbitView" referenced in the PenUp'."""
    script = _r1(tmp_path, "PenUp DefaultOrbitView;\nPenDown DefaultOrbitView;\n" + PROPAGATE)
    result = _refused(tmp_path, script)
    assert any("referenced in the PenDown" in line for line in openframes_errors(script))
    reason, other, complete = plot_refusal(GmatScriptDocument.read(script), result)
    assert reason and other == () and complete


def test_a_toggle_before_the_alignment_leaves_it_the_first_command(tmp_path: Path) -> None:
    text = R6.read_text(encoding="ascii")
    assert text.count(BEGIN) == 1
    plotted = tmp_path / "r6.script"
    plotted.write_text(text.replace(
        BEGIN, "Create XYPlot XY;\nXY.XVariable = SC_Human.ElapsedSecs;\n" + BEGIN
        + "Toggle XY Off;\n"), encoding="ascii")
    assert mission_schedule(plotted).alignment_craft == mission_schedule(R6).alignment_craft
    assert mission_schedule(plotted).alignment_craft is not None


def test_the_diagnostic_sibling_comments_the_toggle_out_with_its_plot(tmp_path: Path) -> None:
    script = _r1(tmp_path, "Toggle DefaultOrbitView Off;\n" + PROPAGATE)
    scenario, document = parse_gmat_script(script, _config())
    sibling = add_diagnostic_report(document, scenario).script_bytes.decode("ascii")
    assert f"{NEUTRALISED_PREFIX}Toggle DefaultOrbitView Off;\n" in sibling
    assert "\nToggle" not in sibling


def test_the_console_copy_leaves_the_toggle_out(tmp_path: Path) -> None:
    script = _slotted(tmp_path, SPLIT)
    path, copy = write_console_copy(script, tmp_path / "copy")
    assert "Toggle" not in path.read_text(encoding="ascii")
    assert len(copy.deleted_lines) == 2 + sum(
        1 for line in R1_PLOTS.read_text(encoding="ascii").splitlines()
        if line.startswith(("Create OpenFrames", "DefaultOrbitView")))


def _refused(tmp_path: Path, script: Path) -> GmatRunResult:
    """A run of ``script`` as R2026a refuses it: its records on stdout, exit 1."""
    stdout = tmp_path / "stdout.txt"
    stdout.write_text("".join(f"{line}\n" for line in openframes_errors(script)),
                      encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    return GmatRunResult(
        run_id="r", success=False, return_code=1, started_at_utc="", duration_s=0.0,
        workdir=str(tmp_path), script_path=str(script), script_sha256="", executable_sha256="",
        startup_sha256=None, stdout_path=str(stdout), stderr_path=str(empty), log_path=None,
        reports={}, report_sha256={}, error=EXIT_STATUS_ERROR.format(1),
    )


def test_gmats_record_for_the_toggle_is_the_plots(tmp_path: Path) -> None:
    """R2026a, R1 with the OpenFrames view and this Toggle: 'Nonexistent Subscriber
    "DefaultOrbitView" referenced in the Toggle', which the plots explain."""
    script = _r1(tmp_path, "Toggle DefaultOrbitView Off;\n" + PROPAGATE)
    result = _refused(tmp_path, script)
    assert any("referenced in the Toggle" in line for line in openframes_errors(script))
    reason, other, complete = plot_refusal(GmatScriptDocument.read(script), result)
    assert reason.startswith("GmatConsole cannot create") and other == () and complete


def test_a_toggle_record_for_another_object_is_not_the_plots(tmp_path: Path) -> None:
    script = _r1(tmp_path, "Toggle DefaultOrbitView Off;\n" + PROPAGATE)
    result = _refused(tmp_path, script)
    with open(result.stdout_path, "a", encoding="utf-8") as stdout:
        stdout.write('9: x.script: **** ERROR **** Interpreter Exception: Nonexistent Subscriber '
                     '"Elsewhere" referenced in the Toggle in line:\n'
                     '   " 178: Toggle DefaultOrbitView Off;"\n')
    _, other, _ = plot_refusal(GmatScriptDocument.read(script), result)
    assert other and "Elsewhere" in other[0]


def _command_cases() -> list[dict[str, object]]:
    return json.loads(ORACLE.read_text(encoding="utf-8"))["command_cases"]


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("case", _command_cases(), ids=lambda case: str(case["name"]))
def test_the_parser_takes_a_plot_command_only_where_the_gmat_gui_does(
    case: dict[str, object], newline: str, tmp_path: Path
) -> None:
    """The GUI's verdicts were measured on an LF template; an organizer's file may be CRLF,
    which the oracle's earlier base was, and the parser must read it the same."""
    text = gui_oracle_script(json.loads(ORACLE.read_text(encoding="utf-8")), case)
    text = text.replace("\r\n", "\n").replace("\n", newline)
    script = tmp_path / f"{case['name']}.script"
    script.write_bytes(text.encode("ascii"))
    try:
        mission_schedule(script)
        took = True
    except OrbitWinError:
        took = False
    gui_took = case["gui"] == "ran to the end"
    if "checker_differs" in case:
        assert gui_took and not took, case["checker_differs"]
        return
    assert took is gui_took
    if not took:
        return
    # The copy leaves out a command of OpenFrames and GroundTrack objects, keeps one of
    # an XYPlot for GmatConsole, and has no copy for one of both.
    document = GmatScriptDocument.read(script)
    line = next(line for line in document.lines
                if plot_command(_code_part(line.rstrip("\r\n")), document))
    names = plot_command(_code_part(line.rstrip("\r\n")), document) or ()
    copied = {created_types(document)[name] in COPIED_TYPES for name in names}
    if copied == {True, False}:
        with pytest.raises(PolicyError, match="not a plot command naming only"):
            write_console_copy(script, tmp_path / "copy")
        return
    path, _ = write_console_copy(script, tmp_path / "copy")
    assert (line.encode("ascii") in path.read_bytes()) is (copied == {False})


def test_a_toggle_of_a_kept_and_a_dropped_plot_is_refused_in_a_private_copy(
    tmp_path: Path,
) -> None:
    """The probe keeps what a craft names; a Toggle half kept, half commented, would run
    against an object that no longer exists, or leave one that does untoggled."""
    from orbitwin.diagnostic import _neutralise_plots
    from orbitwin.gmat_parser import PLOT_TYPES

    script = _r1(tmp_path, "Toggle DefaultOrbitView Off;\n" + PROPAGATE)
    text = script.read_text(encoding="ascii").replace(
        "Create ReportFile", "Create XYPlot XY;\nXY.XVariable = SC_Human.ElapsedSecs;\n"
        "Create ReportFile", 1).replace("Toggle DefaultOrbitView Off;", "Toggle DefaultOrbitView XY Off;")
    script.write_text(text, encoding="ascii")
    document = GmatScriptDocument.read(script)
    lines, _ = _neutralise_plots(document)
    assert f"{NEUTRALISED_PREFIX}Toggle DefaultOrbitView XY Off;\n" in lines
    with pytest.raises(PolicyError, match=r"toggles \['XY'\], kept"):
        _neutralise_plots(document, PLOT_TYPES, frozenset({"XY"}))
