"""What only stock GmatConsole can settle about the official template and its report."""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from orbitwin import gmat_runner
from orbitwin.config import OrbitWinConfig
from orbitwin.diagnostic import add_diagnostic_report
from orbitwin.errors import ParseError
from orbitwin.gmat_parser import (
    CODE_CONTROL_CHARACTERS,
    GmatScriptDocument,
    parse_gmat_script,
    report_file_specs,
)
from orbitwin.gmat_runner import gmat_errors, run_gmat
from orbitwin.verify import PlotRefusal, plot_refusal

pytestmark = pytest.mark.gmat


def _console_copy(script: Path, out: Path, replace: tuple[str, str] | None = None) -> Path:
    """The private copy verification runs: plots commented out, the official report kept."""
    scenario, document = parse_gmat_script(script, OrbitWinConfig())
    data = add_diagnostic_report(document, scenario).script_bytes
    if replace is not None:
        old, new = (part.encode("ascii") for part in replace)
        assert old in data
        data = data.replace(old, new)
    out.write_bytes(data)
    return out


def _official(script: Path) -> list[str]:
    scenario, _ = parse_gmat_script(script, OrbitWinConfig())
    names = [report.filename for report in scenario.report_files]
    assert names, "no official ReportFile: every check below would pass vacuously"
    return names


def test_the_template_is_refused_and_its_console_copy_runs(
    tmp_path: Path, gmat: tuple[str, str], workshop_script: Path
) -> None:
    """Console mode skips GUI plugins, so OpenFrames objects are unknown types."""
    console, startup = gmat
    names = _official(workshop_script)
    raw = run_gmat(console, workshop_script, tmp_path / "raw", names, startup_file=startup,
                   timeout_s=600)
    assert not raw.success and raw.return_code == 1
    assert any("OpenFramesInterface" in str(error) for error in gmat_errors(raw, 50).errors)
    # Every record GMAT wrote is about the OpenFrames objects: the advice and nothing more.
    assert plot_refusal(GmatScriptDocument.read(workshop_script), raw) == PlotRefusal(
        "GmatConsole cannot create OpenFramesInterface DefaultOrbitView, OpenFramesView "
        "DefaultOrbitView_View (OpenFrames)", ()
    )
    copy = _console_copy(workshop_script, tmp_path / "copy.script")
    ok = run_gmat(console, copy, tmp_path / "copy", names, startup_file=startup, timeout_s=600)
    assert ok.success, ok.error


def test_the_official_header_is_the_declared_columns_in_order(
    tmp_path: Path, gmat: tuple[str, str], workshop_script: Path
) -> None:
    """The report the validator reads; our own readers find columns by name."""
    console, startup = gmat
    scenario, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    copy = _console_copy(workshop_script, tmp_path / "copy.script")
    result = run_gmat(console, copy, tmp_path / "run", _official(workshop_script),
                      startup_file=startup, timeout_s=600)
    assert result.success, result.error
    for spec in scenario.report_files:
        header = Path(result.reports[spec.filename]).read_text(encoding="utf-8").splitlines()[0]
        assert header.split() == list(spec.parameters), header


def test_two_runs_at_once_keep_their_own_reports(
    tmp_path: Path, gmat: tuple[str, str], workshop_script: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verification runs gmat.workers at a time; the precision tells the two apart."""
    console, startup = gmat
    spans: list[tuple[float, float]] = []

    class Timed(subprocess.Popen):  # type: ignore[type-arg]
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.began = time.monotonic()
            super().__init__(*args, **kwargs)  # type: ignore[call-overload]

        def communicate(self, *args: object, **kwargs: object) -> tuple[object, object]:
            try:
                return super().communicate(*args, **kwargs)  # type: ignore[arg-type]
            finally:
                spans.append((self.began, time.monotonic()))

    monkeypatch.setattr(gmat_runner.subprocess, "Popen", Timed)
    names = _official(workshop_script)
    jobs = [
        (_console_copy(workshop_script, tmp_path / "a.script"), tmp_path / "a"),
        (
            _console_copy(workshop_script, tmp_path / "b.script",
                          ("ReportFile1.Precision = 16;", "ReportFile1.Precision = 9;")),
            tmp_path / "b",
        ),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(
            lambda job: run_gmat(console, job[0], job[1], names, startup_file=startup,
                                 timeout_s=600),
            jobs,
        )
    assert first.success and second.success, (first.error, second.error)
    # The consoles themselves overlapped, not just the calls around them.
    consoles = [span for span in spans if span[1] - span[0] > 0.5]
    assert len(consoles) == 2, spans
    assert consoles[0][0] < consoles[1][1] and consoles[1][0] < consoles[0][1], consoles
    for name in names:
        left = Path(first.reports[name]).read_text(encoding="utf-8").splitlines()[1].split()
        right = Path(second.reports[name]).read_text(encoding="utf-8").splitlines()[1].split()
        assert len(left[0].replace(".", "")) >= 15, left
        assert len(right[0].replace(".", "")) <= 9, right


def test_another_error_beside_the_openframes_ones_is_quoted(
    tmp_path: Path, gmat: tuple[str, str], workshop_script: Path
) -> None:
    """R2026a refuses ShowPlot = maybe on the GroundTrack, which it reads; say so first."""
    console, startup = gmat
    script = tmp_path / "maybe.script"
    script.write_text(workshop_script.read_text(encoding="ascii").replace(
        "DefaultGroundTrackPlot.ShowPlot = true;", "DefaultGroundTrackPlot.ShowPlot = maybe;"),
        encoding="ascii")
    raw = run_gmat(console, script, tmp_path / "raw", _official(workshop_script),
                   startup_file=startup, timeout_s=600)
    reason, other, complete = plot_refusal(GmatScriptDocument.read(script), raw)
    assert reason.startswith("GmatConsole cannot create OpenFramesInterface") and complete
    # Once, with the line GMAT printed after it.
    assert len(other) == 1 and 'The value of "maybe" for field "ShowPlot"' in other[0]
    assert other[0].endswith(": DefaultGroundTrackPlot.ShowPlot = maybe;)"), other


def test_a_ground_track_can_leave_no_report(tmp_path: Path, gmat: tuple[str, str]) -> None:
    """R2026a: R1 with the two-burn template's GroundTrack created before its ReportFile exits 0
    and writes no report; created after it, the report is written."""
    console, startup = gmat
    root = Path(__file__).resolve().parents[2]
    r1 = (root / "examples" / "ranking" / "R1_NearPlane_Reachable.script").read_text(
        encoding="ascii")
    ground = "".join(
        line.replace("HumanSC", "SC_Human").replace("AlienSC", "SC_Alien")
        for line in (root / "examples" / "two_burn_rendezvous.script").read_text(
            encoding="ascii").splitlines(keepends=True)
        if line.startswith(("Create GroundTrack ", "DefaultGroundTrackPlot."))
    )
    report = "Create ReportFile ReportFile1;"
    assert r1.count(report) == 1 and ground
    results = {}
    for name, text in (("before", r1.replace(report, ground + report)),
                       ("after", r1.replace("BeginMissionSequence;",
                                            ground + "BeginMissionSequence;", 1))):
        script = tmp_path / f"{name}.script"
        script.write_text(text, encoding="ascii")
        names = [spec.filename for spec in report_file_specs(script)]
        assert names
        results[name] = (script, run_gmat(console, script, tmp_path / name, names,
                                          startup_file=startup, timeout_s=600))
    script, before = results["before"]
    assert before.return_code == 0 and not before.success
    reason, other, _ = plot_refusal(GmatScriptDocument.read(script), before)
    assert reason.startswith("it creates GroundTrack DefaultGroundTrackPlot before that "
                             "ReportFile") and other == ()
    assert results["after"][1].success, results["after"][1].error


def test_gmat_reads_a_script_as_the_parser_does(
    tmp_path: Path, gmat: tuple[str, str], plain_workshop_script: Path
) -> None:
    """A control character in a comment changes nothing; a NUL silently drops the rest."""
    console, startup = gmat
    names = _official(plain_workshop_script)
    text = plain_workshop_script.read_text(encoding="ascii")

    def run(name: str, body: str):  # type: ignore[no-untyped-def]
        script = tmp_path / f"{name}.script"
        script.write_text(body, encoding="ascii")
        return script, run_gmat(console, script, tmp_path / name, names, startup_file=startup,
                                timeout_s=600)

    _, plain = run("plain", text)
    assert plain.success, plain.error
    # Each as `% note<c> this is not valid gmat`: a GMAT that split the line there
    # would read the rest as a command and refuse the script.
    for char in sorted(CODE_CONTROL_CHARACTERS):
        commented, ran = run(f"commented-{ord(char):02x}", text.replace(
            "BeginMissionSequence;",
            f"% note{char} this is not valid gmat\nBeginMissionSequence;", 1))
        assert ran.return_code == 0 and ran.report_sha256 == plain.report_sha256, (
            hex(ord(char)), ran.error)
        parse_gmat_script(commented, OrbitWinConfig())

    # NUL padding at the end, and Ctrl-Z in a last comment: the same report.
    for name, tail in (("nul-padding", "\x00\x00\x00"), ("ctrl-z-comment", "% end\x1a\n")):
        padded, ran = run(name, text + tail)
        assert ran.return_code == 0 and ran.report_sha256 == plain.report_sha256, (name, ran.error)
        parse_gmat_script(padded, OrbitWinConfig())
    # A bare Ctrl-Z is a command to R2026a on Linux, and refused: so it is here.
    bare, ran = run("ctrl-z-bare", text + "\x1a")
    assert ran.return_code == 1, ran.error
    with pytest.raises(ParseError, match="Ctrl-Z"):
        parse_gmat_script(bare, OrbitWinConfig())

    last = text.rindex("\nPropagate")
    nul, cut = run("nul", text[:last] + "\n% stop\x00 here" + text[last:])
    # rc 0 and a report: only shorter, because the last Propagate never ran.
    assert cut.success, cut.error
    rows = [len(Path(path).read_text(encoding="utf-8").splitlines())
            for path in (cut.reports[names[0]], plain.reports[names[0]])]
    assert rows[0] < rows[1], rows
    with pytest.raises(ParseError, match="holds a NUL byte"):
        parse_gmat_script(nul, OrbitWinConfig())


def test_a_comment_after_a_value_ends_it(
    tmp_path: Path, gmat: tuple[str, str], plain_workshop_script: Path
) -> None:
    """R2026a writes ReportFile1.txt for both; main read the second's file name through
    its comment. The parser reads the first and refuses the second, whose ; is in it."""
    console, startup = gmat
    text = plain_workshop_script.read_text(encoding="ascii")
    old = "ReportFile1.Filename = 'ReportFile1.txt';"
    assert text.count(old) == 1
    for name, line in (("closed", "ReportFile1.Filename = 'ReportFile1.txt'; % was 'old.txt';"),
                       ("open", "ReportFile1.Filename = 'ReportFile1.txt' % was 'old.txt';")):
        script = tmp_path / f"{name}.script"
        script.write_text(text.replace(old, line), encoding="ascii")
        ran = run_gmat(console, script, tmp_path / name, ["ReportFile1.txt"],
                       startup_file=startup, timeout_s=600)
        assert ran.success, (name, ran.error)
        if name == "closed":
            assert [spec.filename for spec in report_file_specs(script)] == ["ReportFile1.txt"]
        else:
            with pytest.raises(ParseError, match="neither a Create nor an assignment"):
                parse_gmat_script(script, OrbitWinConfig())


def test_gmat_reads_a_percent_as_the_parser_does(
    tmp_path: Path, gmat: tuple[str, str], plain_workshop_script: Path
) -> None:
    """R2026a, TextParser: a line whose first % is inside its first single-quote pair
    is all command; double quotes are not quotes. The parser reads the one GMAT names
    a file it can name, and refuses the two it cannot."""
    console, startup = gmat
    text = plain_workshop_script.read_text(encoding="ascii")
    old = "ReportFile1.Filename = 'ReportFile1.txt';"
    assert text.count(old) == 1

    def run(name: str, line: str, report: str):  # type: ignore[no-untyped-def]
        script = tmp_path / f"{name}.script"
        script.write_text(text.replace(old, line), encoding="ascii")
        return script, run_gmat(console, script, tmp_path / name, [report],
                                startup_file=startup, timeout_s=600)

    script, ran = run("semicolon", "ReportFile1.Filename = 'Report;%1.txt';", "Report;%1.txt")
    assert ran.success, ran.error
    assert [spec.filename for spec in report_file_specs(script)] == ["Report;%1.txt"]

    # The comment is part of the value, quotes and all.
    script, ran = run("commented", "ReportFile1.Filename = 'Report%1.txt'; % note",
                      "'Report%1.txt'; % note")
    assert ran.success, ran.error
    with pytest.raises(ParseError, match="its first % is inside quotes"):
        parse_gmat_script(script, OrbitWinConfig())

    # A double quote inside single quotes is text.
    script = tmp_path / "quoted-double.script"
    script.write_text(text.replace("HumanSC.Id = 'SatId';", "HumanSC.Id = 'Sat \"X\"';", 1),
                      encoding="ascii")
    ran = run_gmat(console, script, tmp_path / "quoted-double", ["ReportFile1.txt"],
                   startup_file=startup, timeout_s=600)
    assert ran.success, ran.error
    parse_gmat_script(script, OrbitWinConfig())

    script, ran = run("double", 'ReportFile1.Filename = "Report%1.txt";', "Report%1.txt")
    assert ran.return_code == 1
    assert any('"Report" is not a valid RHS' in str(error)
               for error in gmat_errors(ran, 10).errors), ran.error
    with pytest.raises(ParseError, match="holds a double quote outside single quotes"):
        parse_gmat_script(script, OrbitWinConfig())


def test_gmat_continues_a_line_that_ends_in_dots(
    tmp_path: Path, gmat: tuple[str, str], plain_workshop_script: Path
) -> None:
    """R2026a, ScriptReadWriter::HasEllipse: a line ending in `...`, comment and all,
    swallows the next; `...;` is refused. The parser refuses both."""
    console, startup = gmat
    text = plain_workshop_script.read_text(encoding="ascii")
    old = "ReportFile1.Maximized = false;\nReportFile1.Filename = 'ReportFile1.txt';"
    assert text.count(old) == 1

    def run(name: str, lines: str, report: str):  # type: ignore[no-untyped-def]
        script = tmp_path / f"{name}.script"
        script.write_text(text.replace(old, lines), encoding="ascii")
        return script, run_gmat(console, script, tmp_path / name, [report],
                                startup_file=startup, timeout_s=600)

    then = "\nReportFile1.Filename = 'Other.txt';"
    script, ran = run("control", "ReportFile1.Maximized = false; % window state" + then,
                      "Other.txt")
    assert ran.success, ran.error
    parse_gmat_script(script, OrbitWinConfig())
    # The naming line is part of the comment: the report keeps its default name.
    script, ran = run("dots", "ReportFile1.Maximized = false; % window state..." + then,
                      "ReportFile1.txt")
    assert ran.success, ran.error
    with pytest.raises(ParseError, match=r"ends in `\.\.\.`"):
        parse_gmat_script(script, OrbitWinConfig())
    script, ran = run("semicolon", "ReportFile1.Maximized = false; % window state...;" + then,
                      "Other.txt")
    assert ran.return_code == 1
    assert any("Ellipses must be at the end of the line" in str(error)
               for error in gmat_errors(ran, 10).errors), ran.error
    with pytest.raises(ParseError, match=r"ends in `\.\.\.`"):
        parse_gmat_script(script, OrbitWinConfig())
