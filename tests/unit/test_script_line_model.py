"""Every reader of a script sees the lines GmatConsole runs.

GmatConsole ends a line at CR, LF or CRLF, and stops reading a script at NUL.
str.splitlines also ends a line at VT, FF and \\x1c-\\x1e, so a comment holding
one used to split into a second line here. Our lines now split at CR and LF
only. Those characters pass in a comment, where neither side reads them, and
are refused in code. NUL is refused anywhere: GmatConsole drops what follows
it and still exits 0 (R2026a, tests/integration/test_gmat_contract.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import quick_config, stamped

from orbitwin.errors import ParseError
from orbitwin.gmat_parser import GmatScriptDocument, parse_gmat_script, script_lines
from orbitwin.io import candidate_from_dict
from orbitwin.models import file_sha256
from orbitwin.pipeline import solve_scenario
from orbitwin.search import zero_burn_candidate
from orbitwin.verify import certify_candidate

CONTROLS = ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f"]


def _variant(tmp_path: Path, workshop: Path, name: str, before_begin: str) -> Path:
    script = tmp_path / name
    script.write_text(
        workshop.read_text(encoding="ascii").replace(
            "BeginMissionSequence;", before_begin + "BeginMissionSequence;", 1
        ),
        encoding="ascii",
    )
    return script


def test_lines_end_at_cr_lf_and_crlf_only() -> None:
    assert script_lines("a\rb\nc\r\nd\x0be\x0cf\x1cg") == ("a\r", "b\n", "c\r\n", "d\x0be\x0cf\x1cg")
    assert script_lines("") == () and script_lines("x\n") == ("x\n",)


@pytest.mark.parametrize("character", CONTROLS)
@pytest.mark.parametrize(
    "comment", ["% note{c}\n", "% note{c} more\n", "%{c}\n", "% a{c}% b\n",
                "% note{c}Create ReportFile GhostReport;\n"],
    ids=["at-end", "words-after", "alone", "second-percent", "ghost-create"],
)
def test_a_control_character_in_a_comment_is_read_as_gmat_reads_it(
    tmp_path: Path, workshop_script: Path, character: str, comment: str
) -> None:
    """main parsed some of these, refused others and read the ghost's report; GMAT runs all."""
    script = _variant(tmp_path, workshop_script, "commented.script", comment.format(c=character))
    plain_scenario, plain = parse_gmat_script(workshop_script, quick_config())
    scenario, document = parse_gmat_script(script, quick_config())
    assert len(document.lines) == len(plain.lines) + 1
    assert [report.filename for report in scenario.report_files] == [
        report.filename for report in plain_scenario.report_files
    ]
    def flown(segments):  # type: ignore[no-untyped-def]
        return [(s.duration_s, s.spacecraft_names, s.line_index) for s in segments]

    # One line more, before the mission: each segment one line later, and the same.
    assert flown(scenario.propagate_segments) == [
        (duration, names, index + 1) for duration, names, index in flown(plain_scenario.propagate_segments)
    ]


@pytest.mark.parametrize("character", CONTROLS)
@pytest.mark.parametrize(
    "code", ["Create\x00Spacecraft Extra;\n", "HumanSC.DryMass = 850{c};\n", "{c}% lead\n",
             "AlienSC.Id = 'x%y'{c};\n"],
    ids=["nul-in-code", "in-value", "before-comment", "after-quoted-percent"],
)
def test_a_control_character_in_code_is_refused(
    tmp_path: Path, workshop_script: Path, character: str, code: str
) -> None:
    script = _variant(tmp_path, workshop_script, "code.script", code.format(c=character))
    with pytest.raises(ParseError, match=r"line \d+ holds (a NUL byte|the control character)"):
        GmatScriptDocument.read(script)


@pytest.mark.parametrize(
    "stop, name, where",
    [("\x00", "a NUL byte", "comment-before-commands"),
     ("\x00", "a NUL byte", "code-before-commands"),
     ("\x00", "a NUL byte", "own-line-before-commands"),
     ("\x1a", "Ctrl-Z", "comment-before-commands")],
)
def test_a_stop_character_before_a_command_is_refused(
    tmp_path: Path, workshop_script: Path, stop: str, name: str, where: str
) -> None:
    """GMAT stops reading at NUL, and a Windows GMAT at Ctrl-Z: what follows would not run.

    R2026a exits 0 and writes a short report (tests/integration/test_gmat_contract.py).
    """
    insert = {"comment-before-commands": f"% note{stop}\n",
              "code-before-commands": f"HumanSC.DryMass = 850;{stop}\n",
              "own-line-before-commands": f"{stop}\n"}[where]
    script = _variant(tmp_path, workshop_script, "stop.script", insert)
    with pytest.raises(ParseError, match=rf"holds {name}.*line \d+ has a command after it"):
        parse_gmat_script(script, quick_config())


@pytest.mark.parametrize(
    "stop, tail",
    [("\x00", "{s}"), ("\x00", "{s}{s}{s}\n{s}{s}"), ("\x00", "\n{s}\n  \n"),
     ("\x00", "% last comment{s}\n"), ("\x00", "% x{s} more\n\n% y\n"),
     ("\x1a", "% last comment{s}\n"), ("\x1a", "% x{s} more\n\n% y\n")],
    ids=["nul-one", "nul-padding", "nul-blank-lines-after", "nul-in-last-comment",
         "nul-comments-after", "ctrl-z-in-last-comment", "ctrl-z-comments-after"],
)
def test_a_stop_character_with_nothing_after_it_parses(
    tmp_path: Path, workshop_script: Path, stop: str, tail: str
) -> None:
    """R2026a runs trailing NUL padding with the same report; input/original is immutable."""
    script = tmp_path / "tail.script"
    script.write_text(workshop_script.read_text(encoding="ascii") + tail.format(s=stop),
                      encoding="ascii")
    plain, _ = parse_gmat_script(workshop_script, quick_config())
    scenario, _ = parse_gmat_script(script, quick_config())
    assert scenario.report_files == plain.report_files
    assert [s.duration_s for s in scenario.propagate_segments] == [
        s.duration_s for s in plain.propagate_segments
    ]


def test_every_character_python_reads_as_space_or_a_break_is_refused_in_code(
    tmp_path: Path, workshop_script: Path
) -> None:
    """Derived, not copied: every ASCII character str.isspace or str.splitlines treats as
    whitespace or a break, but GMAT as text, is refused in code; any other byte a script
    may hold is ASCII, as the parser refuses every byte above 127."""
    from orbitwin.gmat_parser import CODE_CONTROL_CHARACTERS

    found = {
        chr(point) for point in range(128)
        if chr(point).isspace() or len(f"a{chr(point)}b".splitlines()) > 1
    } - set(" \t\r\n")
    assert {"\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f"} == found == CODE_CONTROL_CHARACTERS
    script = tmp_path / "nel.script"
    script.write_bytes(workshop_script.read_bytes() + "% a\x85b\n".encode("latin-1"))
    with pytest.raises(ParseError, match="non-ASCII byte 0x85"):
        GmatScriptDocument.read(script)


@pytest.mark.parametrize(
    "line, filename",
    [
        ("ReportFile1.Filename = 'ReportFile1.txt'; % was 'old.txt';\n", "ReportFile1.txt"),
        # A first % between the line's first two quotes is text, and so is the whole line.
        ("ReportFile1.Filename = 'Report%1.txt';\n", "Report%1.txt"),
        ("ReportFile1.Filename = 'Report;%1.txt';\n", "Report;%1.txt"),
    ],
    ids=["comment-after", "quoted-percent", "quoted-semicolon-percent"],
)
def test_a_value_ends_where_its_comment_starts(
    tmp_path: Path, workshop_script: Path, line: str, filename: str
) -> None:
    """One comment model for every line regex: the value is read off the code part."""
    text = workshop_script.read_text(encoding="ascii")
    old = "ReportFile1.Filename = 'ReportFile1.txt';\n"
    assert text.count(old) == 1
    script = tmp_path / "commented.script"
    script.write_text(text.replace(old, line), encoding="ascii")
    scenario, _ = parse_gmat_script(script, quick_config())
    assert [report.filename for report in scenario.report_files] == [filename]


@pytest.mark.parametrize(
    "old, new, said",
    [
        # R2026a writes ReportFile1.txt; main read "'ReportFile1.txt' % was 'old.txt'".
        ("ReportFile1.Filename = 'ReportFile1.txt';",
         "ReportFile1.Filename = 'ReportFile1.txt' % was 'old.txt';", "form this parser cannot"),
        # e2e audit A4 X15b and R17: a value, a comment, the ; only in the comment.
        ("ReportFile1.Filename = 'ReportFile1.txt';",
         "ReportFile1.Filename = 'ReportFile1.txt' % file;", "form this parser cannot"),
        ("ReportFile1.Add =", "ReportFile1.Delimiter = ',' % comma;\nReportFile1.Add =",
         "form this parser cannot"),
        # R2026a writes a file named "'Report%1.txt'; % a quoted % is text".
        ("ReportFile1.Filename = 'ReportFile1.txt';",
         "ReportFile1.Filename = 'Report%1.txt'; % a quoted % is text",
         "its first % is inside quotes"),
        # Double quotes are not quotes to GMAT: R2026a reads `ReportFile1.Filename = "DQ`.
        ("ReportFile1.Filename = 'ReportFile1.txt';",
         'ReportFile1.Filename = "DQ%1.txt"; % c', "double quote"),
        # Read as the file `"Report` at bb4c71d; R2026a refuses that value.
        ("ReportFile1.Filename = 'ReportFile1.txt';",
         'ReportFile1.Filename = "Report;%1.txt";', "double quote"),
    ],
    ids=["semicolon-in-comment", "a4-x15b", "a4-r17", "quoted-percent-and-comment",
         "double-quoted-percent", "double-quoted-semicolon"],
)
def test_a_line_gmat_reads_otherwise_is_refused_by_every_reader(
    tmp_path: Path, workshop_script: Path, old: str, new: str, said: str
) -> None:
    """Never misread: report_file_specs, which check_report_rows uses, refuses too."""
    from orbitwin.gmat_parser import report_file_specs

    text = workshop_script.read_text(encoding="ascii")
    assert text.count(old) == 1
    script = tmp_path / "comment.script"
    script.write_text(text.replace(old, new), encoding="ascii")
    with pytest.raises(ParseError, match=said):
        report_file_specs(script)
    with pytest.raises(ParseError, match=said.replace(
            "form this parser cannot", "neither a Create nor an assignment")):
        parse_gmat_script(script, quick_config())


def test_a_commented_control_character_solves_and_certifies_as_without_it(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """input/original is immutable: a comment GMAT runs must not stop any command."""
    config = quick_config()
    # Words after the FF: main split the line there and refused " more".
    script = _variant(tmp_path, workshop_script, "commented.script", "% note\x0c more\n")
    solved = solve_scenario(script, config, tmp_path / "solve")
    assert solved["best_candidate"] is not None
    reports = []
    for name, template in (("plain", workshop_script), ("commented", script)):
        candidate = stamped(candidate_from_dict(solved["best_candidate"]), template, config)
        certificate = certify_candidate(template, candidate, config, fake_gmat, fake_startup,
                                        tmp_path / name, runs=1)
        report = certificate["official_report_files"][0]
        reports.append(file_sha256(tmp_path / name / "submission" / report))
        # The submitted script keeps the comment byte for byte.
        submitted = tmp_path / name / "submission" / certificate["submission_script_name"]
        assert (b"% note\x0c more\n" in submitted.read_bytes()) is (name == "commented")
    assert reports[0] == reports[1]


def test_the_coast_sibling_reads_the_same_lines(tmp_path: Path, workshop_script: Path) -> None:
    """ephemeris rewrites Propagate lines; a FF in a comment hid one from main's splitlines."""
    from orbitwin.ephemeris import _target_only_mission

    script = tmp_path / "coast.script"
    script.write_text(
        workshop_script.read_text(encoding="ascii").replace(
            "BeginMissionSequence;\n",
            "BeginMissionSequence;\n% x\x0cPropagate DefaultProp(AlienSC) {AlienSC.ElapsedSecs = 1};\n",
            1,
        ),
        encoding="ascii",
    )
    scenario, document = parse_gmat_script(script, quick_config())
    coast = _target_only_mission(document.raw_bytes, scenario).decode("ascii")
    assert "% x\x0cPropagate DefaultProp(AlienSC) {AlienSC.ElapsedSecs = 1};\n" in coast


@pytest.mark.parametrize("tail", ["\x00\n\x0c\n", "\x00\x00\n\x0b\n"])
def test_the_coast_sibling_reads_no_further_than_gmat(
    tmp_path: Path, workshop_script: Path, tail: str
) -> None:
    """Rewriting the last Propagate dropped its NUL, and GMAT then read the control
    character after it, which no build reads in the official script."""
    from orbitwin.ephemeris import _target_only_mission

    script = tmp_path / "coast.script"
    script.write_text(workshop_script.read_text(encoding="ascii").rstrip("\n") + tail,
                      encoding="ascii")
    scenario, document = parse_gmat_script(script, quick_config())
    sibling = tmp_path / "coast_diagnostic.script"
    sibling.write_bytes(_target_only_mission(document.raw_bytes, scenario))
    assert sibling.read_text(encoding="ascii").endswith("};")
    GmatScriptDocument.read(sibling)


def test_zero_runs_leave_no_output_behind(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    from orbitwin import cli

    config = quick_config()
    scenario, _ = parse_gmat_script(workshop_script, config)
    candidate = stamped(zero_burn_candidate(scenario, config), workshop_script, config)
    with pytest.raises(ValueError, match="at least one exact run"):
        certify_candidate(workshop_script, candidate, config, fake_gmat, fake_startup,
                          tmp_path / "certify", runs=0)
    assert not (tmp_path / "certify").exists()
    with pytest.raises(SystemExit) as raised:
        cli.main(["certify", str(workshop_script), "--candidates", "x.json",
                  "--candidate", "C-1", "--gmat-console", str(fake_gmat),
                  "--startup-file", str(fake_startup), "--runs", "0",
                  "--output", str(tmp_path / "cli")])
    assert raised.value.code == 2
    assert not (tmp_path / "cli").exists()


def test_the_fake_console_reads_lines_as_gmat_does(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """The test double ends lines at CR and LF only and skips comments, as GMAT does, so
    fake-backed tests can hold control characters in comments."""
    from orbitwin.gmat_runner import run_gmat

    script = _variant(tmp_path, workshop_script, "ghost.script",
                      "% note\x0cCreate ReportFile GhostReport;\n")
    ran = run_gmat(fake_gmat, script, tmp_path / "run", ["ReportFile1.txt"],
                   startup_file=fake_startup)
    assert ran.success, ran.error
    written = sorted(path.name for path in Path(ran.reports["ReportFile1.txt"]).parent.iterdir())
    assert "GhostReport.txt" not in written


def test_a_config_with_no_exact_runs_is_refused() -> None:
    import dataclasses

    from orbitwin.config import validate_config
    from orbitwin.errors import ConfigurationError

    config = quick_config()
    for reruns in (0, -1):
        broken = dataclasses.replace(config, gmat=dataclasses.replace(config.gmat, reruns=reruns))
        with pytest.raises(ConfigurationError, match=r"gmat\.reruns must be at least 1"):
            validate_config(broken)



@pytest.mark.parametrize("tail", ["\x1a", "HumanSC.DryMass = 850;\x1a\n", "\x1a% note\n"],
                         ids=["bare-at-end", "after-code", "before-comment"])
def test_ctrl_z_outside_a_comment_is_refused(
    tmp_path: Path, workshop_script: Path, tail: str
) -> None:
    """Measured on R2026a: a bare Ctrl-Z at the end is `Found invalid function or command
    name ""`, exit 1. A Windows GMAT would stop there instead."""
    script = tmp_path / "ctrl-z.script"
    script.write_text(workshop_script.read_text(encoding="ascii") + tail, encoding="ascii")
    with pytest.raises(ParseError, match=r"holds Ctrl-Z \(0x1A\) outside a comment"):
        parse_gmat_script(script, quick_config())



@pytest.mark.parametrize(
    "tail, refused",
    [("\x00\n\x0c\n", False), ("\x00\x0c\n", False), ("% end\x00\n\x0c\n", False),
     ("\x00\n\x0b x\n", True), ("% end\x1a\n\x0c\n", True),
     # Every build stops at the NUL, whatever stopped a Windows one before it.
     ("% end\x1a \x00\n\x0c\n", False), ("% a\x1a\n% b\x00\n\x0c\n", False)],
    ids=["nul-then-ff-line", "nul-ff-one-line", "nul-comment-then-ff", "nul-then-text",
         "ctrl-z-then-ff", "ctrl-z-nul-one-line", "ctrl-z-line-then-nul-line"],
)
def test_lines_past_a_nul_are_not_read(
    tmp_path: Path, workshop_script: Path, tail: str, refused: bool
) -> None:
    """GMAT never reads past a NUL, so no character there is checked; a Linux GMAT reads
    past a Ctrl-Z, so a control character after one still is."""
    script = tmp_path / "tail.script"
    script.write_text(workshop_script.read_text(encoding="ascii") + tail, encoding="ascii")
    if refused:
        with pytest.raises(ParseError):
            GmatScriptDocument.read(script)
    else:
        GmatScriptDocument.read(script)


def test_a_nul_in_the_last_propagate_label_hides_no_goal(
    tmp_path: Path, workshop_script: Path
) -> None:
    """R2026a stops at the NUL and refuses `Propagate 'coast` (rc 1); main read the goal
    after it as a 1000 s segment."""
    text = workshop_script.read_text(encoding="ascii").rstrip("\n")
    last = text.rindex("\nPropagate ") + 1
    script = tmp_path / "label.script"
    script.write_text(text[:last] + text[last:].replace("Propagate ", "Propagate 'coast\x00' ", 1)
                      + "\n", encoding="ascii")
    with pytest.raises(ParseError, match="no explicit ElapsedSecs"):
        parse_gmat_script(script, quick_config())


@pytest.mark.parametrize(
    "insert",
    ["% NUL\x00\nPropagate DefaultProp(AlienSC) {AlienSC.ElapsedSecs = 1};\n",
     "HumanSC.DryMass = 850\x0c;\n", "% Ctrl-Z\x1a\nHumanSC.DryMass = 850;\n",
     "HumanSC.Id = 'a\x0cb';\n", 'HumanSC.Id = "a";\n', "HumanSC.Id = 'a%b'; % c\n",
     "HumanSC.Id = 'a' ...\n;\n"],
    ids=["nul", "control", "ctrl-z", "control-quoted", "double-quote", "quoted-percent",
         "continuation"],
)
def test_a_refusal_of_the_input_says_to_stop_not_to_edit(
    tmp_path: Path, workshop_script: Path, insert: str
) -> None:
    """input/original may not be edited, so no refusal tells the operator to remove anything."""
    script = _variant(tmp_path, workshop_script, "official.script", insert)
    with pytest.raises(ParseError) as raised:
        parse_gmat_script(script, quick_config())
    message = str(raised.value)
    assert "remove it" not in message
    assert message.endswith("if this is the official script, it may not be edited: stop and "
                            "take it to the organizer (docs/RUNBOOK.md, Stop rules)")


def test_a_control_character_in_a_quoted_value_says_so(
    tmp_path: Path, workshop_script: Path
) -> None:
    """R2026a runs `Id = 'Sat<FF>Id'` with the same report; the parser does not model it."""
    script = _variant(tmp_path, workshop_script, "quoted.script", "HumanSC.Id = 'Sat\x0cId';\n")
    with pytest.raises(ParseError, match="in a quoted value, which GMAT keeps as text and this "
                       "parser does not model"):
        GmatScriptDocument.read(script)


@pytest.mark.parametrize(
    "old, new, refused",
    [("BeginMissionSequence;", "% to be continued...\nBeginMissionSequence;", False),
     ("'ReportFile1.txt';", "'Report...1.txt';", False),
     ("HumanSC.DryMass = 850;", "HumanSC.DryMass = ...\n  850;", True),
     # R2026a wrote ReportFile1.txt: the next line, which names the file, was swallowed.
     ("ReportFile1.Filename = 'ReportFile1.txt';",
      "ReportFile1.Maximized = false; % window state...\nReportFile1.Filename = 'Other.txt';",
      True),
     # R2026a: "Ellipses must be at the end of the line", exit status 1.
     ("BeginMissionSequence;", "HumanSC.DryMass = 850; % note...;\nBeginMissionSequence;",
      True),
     ("BeginMissionSequence;", "HumanSC.DryMass = 850 ...;\nBeginMissionSequence;", True)],
    ids=["comment-line", "quoted", "code", "inline-comment", "comment-then-semicolon",
         "semicolon"],
)
def test_a_line_continues_as_gmat_continues_it(
    tmp_path: Path, workshop_script: Path, old: str, new: str, refused: bool
) -> None:
    """ScriptReadWriter::HasEllipse: a line that is not a comment line, read up to a NUL
    and trimmed of blanks and then semicolons, continues when it ends in `...`."""
    text = workshop_script.read_text(encoding="ascii")
    assert text.count(old) == 1
    script = tmp_path / "dots.script"
    script.write_text(text.replace(old, new, 1), encoding="ascii")
    if refused:
        number = script.read_text(encoding="ascii").splitlines().index(new.split("\n")[0]) + 1
        with pytest.raises(ParseError, match=rf"dots\.script line {number} ends in `\.\.\.`"):
            parse_gmat_script(script, quick_config())
    else:
        parse_gmat_script(script, quick_config())


@pytest.mark.parametrize("tail", ["}; % coast...\x1a", "}; % coast...\x00", "}; % coast...\x1a\n"])
def test_a_stop_after_dots_continues_into_nothing(
    tmp_path: Path, workshop_script: Path, tail: str
) -> None:
    """Up to a NUL, or a Windows build's Ctrl-Z, the last line ends in `...`: GMAT reads
    on past it, so the line is refused as the others are."""
    text = workshop_script.read_text(encoding="ascii").rstrip("\n")
    assert text.endswith("};")
    script = tmp_path / "dots.script"
    script.write_text(text.removesuffix("};") + tail, encoding="ascii")
    with pytest.raises(ParseError, match=r"ends in `\.\.\.`"):
        GmatScriptDocument.read(script)


@pytest.mark.parametrize(
    "line, code",
    [
        ("X.Y = 1; % c", "X.Y = 1; "),
        ("X.Y = 'a%b';", "X.Y = 'a%b';"),
        ("X.Y = 'a%b'; % c", "X.Y = 'a%b'; % c"),
        ("X.Y = 'a'; % c 'd'", "X.Y = 'a'; "),
        ('X.Y = "a%b";', 'X.Y = "a'),
        ("% 'a%b'", ""),
        ("X.Y = 1;\x00 % c", "X.Y = 1;"),
        ("X.Y = 'a\x00%b';", "X.Y = 'a"),
        ("X.Y = 'unclosed %", "X.Y = 'unclosed "),
    ],
)
def test_the_code_part_is_gmats(line: str, code: str) -> None:
    """R2026a TextParser.cpp: only the first % counts; it is text between the line's first
    two single quotes, and then the whole line is the command; double quotes are not quotes."""
    from orbitwin.gmat_parser import _code_part

    assert _code_part(line) == code


def test_the_fake_console_reads_comments_as_the_parser_does(project_root: Path) -> None:
    """The double copies _code_part, since it imports nothing from orbitwin."""
    import importlib.util
    import random

    from orbitwin.gmat_parser import _code_part

    spec = importlib.util.spec_from_file_location(
        "fake_console", project_root / "scripts" / "fake_gmat_console.py"
    )
    assert spec is not None and spec.loader is not None
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    lines = [line for path in sorted(project_root.glob("examples/**/*.script"))
             for line in path.read_text(encoding="ascii").splitlines()]
    rng = random.Random(20261104)
    lines += ["".join(rng.choice("ab '%\";\x00") for _ in range(rng.randint(0, 12)))
              for _ in range(20000)]
    assert [fake._code(line) for line in lines] == [_code_part(line) for line in lines]


def test_a_non_ascii_byte_is_placed_on_gmats_line(tmp_path: Path, workshop_script: Path) -> None:
    """Lines end at CR too, as GMAT ends them, not only at LF."""
    script = tmp_path / "cr.script"
    text = workshop_script.read_text(encoding="ascii").replace("\n", "\r")
    script.write_bytes(text.encode("ascii") + "% \u00e9\r".encode())
    with pytest.raises(ParseError, match=rf"line {len(text.split(chr(13)))}, column 3"):
        GmatScriptDocument.read(script)


def test_a_burn_value_moves_and_its_comment_stays(tmp_path: Path, workshop_script: Path) -> None:
    """The renderer patches a line's code part and keeps the rest as written."""
    from orbitwin.models import Burn, Candidate
    from orbitwin.script_renderer import assert_final_script_policy, write_rendered_candidate

    old = "ImpulsiveBurn1.Element1 = 0.5983;"
    template = tmp_path / "commented.script"
    template.write_text(workshop_script.read_text(encoding="ascii").replace(
        old, f"{old} % along V, 'km/s'", 1), encoding="ascii")
    config = quick_config()
    scenario, document = parse_gmat_script(template, config)
    hit = Candidate(
        burns=(Burn(1000.0, (0.1, 0.2, 0.3), (0.11, 0.22, 0.33), 0),),
        intercept_time_s=5000.0, minimum_distance_km=0.1, distance_at_completion_km=5.0,
        successful=True, analytical_endpoint_error_km=0.1, branch="test",
        metadata={"arrival_time_s": 5000.0},
    )
    output = tmp_path / "rendered.script"
    write_rendered_candidate(output, tmp_path / "diff.txt", document, scenario, hit, config)
    assert_final_script_policy(template, output, config)
    line = next(line for line in output.read_text(encoding="ascii").splitlines()
                if line.startswith("ImpulsiveBurn1.Element1"))
    assert line == "ImpulsiveBurn1.Element1 = 0.11; % along V, 'km/s'"


@pytest.mark.parametrize(
    "line, refused",
    [("HumanSC.Id = 'Alien \"X\"';", False),
     ("HumanSC.Id = \"Alien\";", True),
     ("HumanSC.OrbitSpiceKernelName = {'a%b.bsp', 'c%d.bsp'};", False),
     ("HumanSC.OrbitSpiceKernelName = {'a%b.bsp'}; % note", True)],
    ids=["double-quote-in-single-quotes", "double-quotes", "percent-in-a-later-quote",
         "percent-then-comment"],
)
def test_only_single_quotes_are_quotes(
    tmp_path: Path, workshop_script: Path, line: str, refused: bool
) -> None:
    """GMAT's one quote character is ': a " or a % inside one is text (R2026a ran the
    first case with exit status 0); outside them the parser does not model either."""
    script = _variant(tmp_path, workshop_script, "quotes.script", line + "\n")
    if refused:
        with pytest.raises(ParseError, match="quotes"):
            GmatScriptDocument.read(script)
    else:
        GmatScriptDocument.read(script)


def test_a_report_gmat_does_not_write_is_not_asked_for(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """R2026a writes no file for WriteReport = false, not even a header; every run used
    to wait for one, so baseline, verification and certify all failed."""
    from orbitwin.verify import baseline_original_script

    config = quick_config()
    script = tmp_path / "debug.script"
    script.write_text(workshop_script.read_text(encoding="ascii").replace(
        "Create ReportFile ReportFile1;",
        "Create ReportFile DebugReport;\nDebugReport.Filename = 'debug.txt';\n"
        "DebugReport.Add = {HumanSC.A1ModJulian};\nDebugReport.WriteReport = false;\n"
        "Create ReportFile ReportFile1;", 1), encoding="ascii")
    baseline_original_script(script, config, fake_gmat, fake_startup, tmp_path / "baseline")
    solved = solve_scenario(script, config, tmp_path / "solve")
    candidate = stamped(candidate_from_dict(solved["best_candidate"]), script, config)
    certificate = certify_candidate(script, candidate, config, fake_gmat, fake_startup,
                                    tmp_path / "certify", runs=1)
    assert certificate["official_report_files"] == ["ReportFile1.txt"]
    assert not list((tmp_path / "certify").rglob("debug.txt"))


def test_the_fake_names_a_report_as_the_parser_does(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """R2026a writes `Report;%1.txt`; the double used to stop the value at its `;`."""
    from orbitwin.gmat_parser import report_file_specs
    from orbitwin.gmat_runner import run_gmat

    script = tmp_path / "semicolon.script"
    script.write_text(workshop_script.read_text(encoding="ascii").replace(
        "'ReportFile1.txt';", "'Report;%1.txt';", 1), encoding="ascii")
    names = [spec.filename for spec in report_file_specs(script)]
    assert names == ["Report;%1.txt"]
    ran = run_gmat(fake_gmat, script, tmp_path / "run", names, startup_file=fake_startup)
    assert ran.success, ran.error


@pytest.mark.parametrize(
    "tail, code",
    [(" % FAKE_GMAT_FAIL", 3), ("\x00\n% FAKE_GMAT_FAIL\n", 3), ("\x00\nnot read\n", 0)],
    ids=["after-code", "after-a-nul", "nothing-past-a-nul"],
)
def test_the_fake_reads_its_directives_wherever_they_sit(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    tail: str, code: int,
) -> None:
    """main's double read them anywhere in the file; past a NUL it reads nothing else."""
    import importlib.util

    from orbitwin.gmat_runner import run_gmat

    spec = importlib.util.spec_from_file_location("fake", fake_gmat)
    assert spec is not None and spec.loader is not None
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    assert fake.gmat_text("A = 1;\x00\nB = 2;\n") == "A = 1;\n"
    script = tmp_path / "directive.script"
    script.write_text(workshop_script.read_text(encoding="ascii").rstrip("\n") + tail,
                      encoding="ascii")
    ran = run_gmat(fake_gmat, script, tmp_path / "run", ["ReportFile1.txt"],
                   startup_file=fake_startup)
    assert ran.return_code == code, ran.error


def test_a_config_number_past_a_double_is_refused(tmp_path: Path) -> None:
    """float(10**400) raises OverflowError; the config refuses it as not finite."""
    from orbitwin.config import finite_number, load_config
    from orbitwin.errors import ConfigurationError

    assert finite_number(10**400) is None and finite_number(True) is None
    assert finite_number(2) == 2.0
    root = Path(__file__).resolve().parents[2]
    text = (root / "config" / "standard.yaml").read_text(encoding="utf-8")
    assert text.count("dv_limit_km_s: 1.5") == 1
    config = tmp_path / "huge.yaml"
    config.write_text(text.replace("dv_limit_km_s: 1.5", "dv_limit_km_s: 1" + "0" * 400),
                      encoding="utf-8")
    with pytest.raises(ConfigurationError, match="dv_limit_km_s must be a finite number"):
        load_config(config)


def test_the_resource_fields_are_read_before_their_comments(workshop_script: Path,
                                                            tmp_path: Path) -> None:
    from orbitwin.gmat_parser import _resource_assignments

    script = _variant(tmp_path, workshop_script, "fields.script", "")
    script.write_text(script.read_text(encoding="ascii").replace(
        "AlienSC.SMA = 9600;", "AlienSC.SMA = 9600; % km", 1),
        encoding="ascii")
    fields = _resource_assignments(GmatScriptDocument.read(script))
    assert fields["AlienSC"]["SMA"] == "9600"


def test_the_fake_reads_a_value_without_a_semicolon_and_off_as_false(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """GMAT runs an assignment with no `;`, and reads WriteReport = Off as false."""
    import importlib.util

    from orbitwin.gmat_runner import run_gmat

    spec = importlib.util.spec_from_file_location("fake", fake_gmat)
    assert spec is not None and spec.loader is not None
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    assert fake.ASSIGN.match("ReportFile1.Filename = 'a.txt'").group(3) == "'a.txt'"
    assert fake.ASSIGN.match("ReportFile1.Filename = 'R;%1.txt';").group(3) == "'R;%1.txt'"
    script = tmp_path / "off.script"
    script.write_text(workshop_script.read_text(encoding="ascii").replace(
        "ReportFile1.WriteReport = true;", "ReportFile1.WriteReport = Off;", 1), encoding="ascii")
    ran = run_gmat(fake_gmat, script, tmp_path / "run", ["ReportFile1.txt"],
                   startup_file=fake_startup)
    assert not ran.success and ran.error == "Expected fresh GMAT report was not created: " \
        "ReportFile1.txt"
