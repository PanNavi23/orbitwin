from __future__ import annotations

import re
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig
from orbitwin.diagnostic import (
    GUI_SUBSCRIBER_TYPES,
    NEUTRALISED_PREFIX,
    add_diagnostic_report,
)
from orbitwin.gmat_parser import parse_gmat_script

# Measured against GMAT R2026a. A script declaring an OpenFrames object is
# refused outright -- console skips GUI plugins and the Linux build ships no
# such library -- and on the organizer's own workshop template a GroundTrack
# leaves the run printing its success marker while writing no report at all.
# Either way nothing about the competition script could be measured on a
# machine without the GUI, which is every machine we have before the day.


def _sibling(script: Path) -> tuple[str, tuple[str, ...]]:
    scenario, document = parse_gmat_script(script, OrbitWinConfig())
    diagnostic = add_diagnostic_report(document, scenario)
    return diagnostic.script_bytes.decode("utf-8"), diagnostic.neutralised


def test_the_organizers_template_has_its_plots_commented_out(workshop_script: Path) -> None:
    text, neutralised = _sibling(workshop_script)
    assert set(neutralised) == {
        "DefaultOrbitView",
        "DefaultGroundTrackPlot",
        "DefaultOrbitView_View",
    }
    for name in neutralised:
        assert f"{NEUTRALISED_PREFIX}Create" in text
        assert not re.search(rf"^\s*{name}\s*\.", text, re.M), f"{name} still assigns"


def test_a_script_that_does_not_draw_is_left_alone(different_plane_script: Path) -> None:
    text, neutralised = _sibling(different_plane_script)
    assert neutralised == ()
    assert NEUTRALISED_PREFIX not in text


def test_the_report_files_survive(workshop_script: Path) -> None:
    """Neutralising must not take the thing being measured with it."""
    scenario, document = parse_gmat_script(workshop_script, OrbitWinConfig())
    diagnostic = add_diagnostic_report(document, scenario)
    text = diagnostic.script_bytes.decode("utf-8")
    for report in scenario.report_files:
        assert re.search(rf"^\s*Create ReportFile {report.name}\s*;", text, re.M)
        assert f"{NEUTRALISED_PREFIX}Create ReportFile" not in text
    assert f"Create ReportFile {diagnostic.resource_name};" in text


def test_the_mission_sequence_is_untouched(workshop_script: Path) -> None:
    """Only the drawing is removed. What the spacecraft do is the measurement."""
    original = workshop_script.read_text(encoding="utf-8-sig")
    text, _neutralised = _sibling(workshop_script)
    mission = re.compile(r"^(?:Propagate|Maneuver|BeginMissionSequence).*$", re.M)
    assert mission.findall(text) == mission.findall(original)


def test_only_plot_types_are_neutralised(workshop_script: Path) -> None:
    # EphemerisFile writes a file rather than drawing one, and console handles
    # it, so it is not on the list; nor is anything that is not a subscriber.
    assert "ReportFile" not in GUI_SUBSCRIBER_TYPES
    assert "EphemerisFile" not in GUI_SUBSCRIBER_TYPES
    assert {"OpenFramesInterface", "GroundTrack", "OrbitView"} <= GUI_SUBSCRIBER_TYPES

    text, _ = _sibling(workshop_script)
    for line in text.splitlines():
        if line.startswith(NEUTRALISED_PREFIX):
            body = line[len(NEUTRALISED_PREFIX):].lstrip()
            assert not body.startswith("Propagate"), line
            assert not body.startswith("Maneuver"), line
            assert not body.startswith("Create Spacecraft"), line


@pytest.mark.parametrize(
    "spelling", ["BeginMissionSequence; % mission", "BeginMissionSequence ;"]
)
def test_the_report_goes_in_whatever_the_begin_line_spelling(
    different_plane_script: Path, tmp_path: Path, spelling: str
) -> None:
    # The parser and GMAT accept both; a literal match failed every verification.
    text = different_plane_script.read_text(encoding="utf-8")
    script = tmp_path / "spelled.script"
    script.write_text(text.replace("BeginMissionSequence;", spelling, 1), encoding="utf-8")
    sibling, _ = _sibling(script)
    lines = sibling.splitlines()
    begin = lines.index(spelling)
    assert "Create ReportFile OrbitWinDiagnostic;" in lines[:begin]
