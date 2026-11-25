"""An announcement entered twice, as printed, and held to the day's script."""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from conftest import bare_environment

from orbitwin.announcement import (
    AnnouncementError,
    ScriptView,
    against_script,
    horizon_s,
    read_entry,
    read_pair,
    read_script,
)
from orbitwin.config import load_config
from orbitwin.rules import RANKING, interception_rules

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
ONSITE = interception_rules("ranking-onsite")
#: R1's SC_Alien as a slide would print it, rounded, with U+2212 and grouped thousands.
ALIEN = {"Epoch": "09 Nov 2026 02:00:00.000 UTC", "CoordinateSystem": "EarthMJ2000Eq",
         "DisplayStateType": "Keplerian", "SMA": "\N{MINUS SIGN}99,650.11045 km",
         "ECC": "1.06852111", "INC": "35.336 deg", "RAAN": "57.046 deg", "AOP": "202.341 deg",
         "TA": "241.212 deg"}
SCORE = {"C_r": "300 km", "kt": "3e-3 1/s", "Ct": "5369.334 s", "kv": "0.0012 1/(m/s)",
         "Cv": "2000 m/s"}


def _toml(table: dict[str, object], prefix: str = "") -> str:
    lines, tables = [], []
    for key, value in table.items():
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            lines.append(f'{key} = "{value}"')
    text = (f"[{prefix}]\n" if prefix and lines else "") + "".join(f"{line}\n" for line in lines)
    for key, value in tables:
        text += _toml(value, f"{prefix}.{key}" if prefix else key)
    return text


def _entry(path: Path, by: str, *, alien: dict[str, str] | None = None,
           human: dict[str, str] | None = None, mission: dict[str, str] | None = None,
           score: dict[str, str] | None = SCORE, **top: str) -> Path:
    body: dict[str, object] = {
        "format": "orbitwin-announcement/1", "entered_by": by, **top,
        "craft": {"SC_Alien": alien or ALIEN,
                  "SC_Human": human or {"Epoch": "09 Nov 2026 02:00:00 UTC"}},
        "mission": mission or {"T_max": "09 Nov 2026 03:51:51.667 UTC", "T_max_s": "6711.667 s"},
    }
    if score is not None:
        body["score"] = score
    path.write_text(_toml(body), encoding="utf-8")
    return path


def _pair(tmp_path: Path, **second: object):
    return read_pair(_entry(tmp_path / "a.toml", "Alice"),
                     _entry(tmp_path / "b.toml", "Bob", **second))  # type: ignore[arg-type]


# -- reading one entry --------------------------------------------------------------------


def test_an_entry_is_read_as_printed(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice"))
    sma = entry.states["SC_Alien"]["SMA"]
    assert (sma.value, sma.unit, sma.half_step) == (Decimal("-99650.11045"), "km",
                                                    Decimal("0.000005"))
    assert entry.score is not None and entry.score["kt"].value == Decimal("0.003")
    assert entry.score["kt"].half_step == Decimal("0.0005")
    assert entry.t_max.half_step_s == Decimal("0.0005")
    assert entry.epochs["SC_Human"].half_step_s == Decimal("0.5")


@pytest.mark.parametrize(("field", "text", "says"), [
    ("ECC", "1.068 deg", "in deg; it is announced in no unit"),
    ("INC", "35.336", "in no unit; it is announced in deg"),
    ("SMA", "-99650 m", "in m; it is announced in km"),
    ("TA", "about 241 deg", "does not start with a number"),
    ("TA", "241,21 deg", "does not start with a number"),
    ("Epoch", "09 Nov 2026 02:00:00.000", "is not 'DD Mon YYYY"),
    ("Epoch", "31 Nov 2026 02:00:00 UTC", "is no such date and time"),
    ("Epoch", "09 Nov 2026 24:00:00 UTC", "is no such date and time"),
    ("Epoch", "09 nov 2026 02:00:00 UTC", "is not 'DD Mon YYYY"),
    ("RCS", "1 m2", "RCS is not a field"),
    ("SMA", "0,123 km", "does not start with a number"),
    ("SMA", "1e1000001 km", "does not start with a number"),
    ("SMA", "1e999 km", "is out of range"),
    ("CoordinateSystem", "Earth Fixed", "is not a GMAT name"),
])
def test_a_value_out_of_form_is_refused(tmp_path: Path, field: str, text: str, says: str) -> None:
    with pytest.raises(AnnouncementError, match=says):
        read_entry(_entry(tmp_path / "a.toml", "Alice", alien={**ALIEN, field: text}))


@pytest.mark.parametrize(("score", "says"), [
    ({**SCORE, "Cv": "2 km/s"}, "in km/s; it is announced in m/s"),
    ({k: v for k, v in SCORE.items() if k != "C_r"}, "lacks C_r"),
    ({**SCORE, "Cr": "300 km"}, "Cr is not a field"),
])
def test_a_score_out_of_form_is_refused(tmp_path: Path, score: dict[str, str], says: str) -> None:
    with pytest.raises(AnnouncementError, match=says):
        read_entry(_entry(tmp_path / "a.toml", "Alice", score=score))


def test_a_degree_sign_is_read_as_deg(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", alien={**ALIEN, "INC": "35.336°"}))
    assert (entry.states["SC_Alien"]["INC"].value, entry.states["SC_Alien"]["INC"].unit) == (
        Decimal("35.336"), "deg")


def test_a_number_outside_quotes_is_refused(tmp_path: Path) -> None:
    path = _entry(tmp_path / "a.toml", "Alice")
    path.write_text(path.read_text(encoding="utf-8").replace('"1.06852111"', "1.06852111"),
                    encoding="utf-8")
    with pytest.raises(AnnouncementError, match="write it in quotes"):
        read_entry(path)


@pytest.mark.parametrize(("top", "says"), [
    ({"format": "orbitwin-announcement/0"}, "format is"),
    ({"entered_by": " "}, "names no one"),
    ({"notes": "x"}, "notes is not a field"),
])
def test_an_entry_without_its_frame_is_refused(tmp_path: Path, top: dict[str, str],
                                               says: str) -> None:
    path = _entry(tmp_path / "a.toml", "Alice")
    text = path.read_text(encoding="utf-8")
    for key, value in top.items():
        line = next((ln for ln in text.splitlines() if ln.startswith(f"{key} =")), None)
        text = (text.replace(line, f'{key} = "{value}"') if line
                else f'{key} = "{value}"\n' + text)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(AnnouncementError, match=says):
        read_entry(path)


# -- two people ---------------------------------------------------------------------------


def test_two_entries_agree_on_values_not_spellings(tmp_path: Path) -> None:
    first, second, agreed = _pair(
        tmp_path, score={**SCORE, "kt": "0.003 1/s"},
        alien={**ALIEN, "SMA": "-99650.110450 km"})
    assert first.score["kt"].text != second.score["kt"].text  # type: ignore[index]
    # The coarser printing of each value is the one held to the script, whoever wrote it.
    assert agreed.states["SC_Alien"]["SMA"].half_step == Decimal("0.000005")
    *_, agreed = read_pair(
        _entry(tmp_path / "c.toml", "Carol", alien={**ALIEN, "INC": "35.33600 deg"}),
        _entry(tmp_path / "d.toml", "Dan"))
    assert agreed.states["SC_Alien"]["INC"].half_step == Decimal("0.0005")


def test_the_days_typo_is_caught_by_the_second_entry(tmp_path: Path) -> None:
    """kt was typed 0.00095073 on 2026-11-09 against an announced 5.9073e-4."""
    with pytest.raises(AnnouncementError, match=r"score\.kt: '3e-3 1/s' by Alice, "
                                                r"'0\.0095 1/s' by Bob"):
        _pair(tmp_path, score={**SCORE, "kt": "0.0095 1/s"})


@pytest.mark.parametrize("second", [
    {"alien": {k: v for k, v in ALIEN.items() if k != "TA"}},
    {"score": None},
    {"mission": {"T_max": "09 Nov 2026 03:51:51.667 UTC"}},
])
def test_a_value_one_entry_leaves_out_is_a_difference(tmp_path: Path,
                                                      second: dict[str, object]) -> None:
    with pytest.raises(AnnouncementError, match="not given"):
        _pair(tmp_path, **second)


def test_one_person_cannot_enter_both(tmp_path: Path) -> None:
    with pytest.raises(AnnouncementError, match="both entered by 'Alice'"):
        read_pair(_entry(tmp_path / "a.toml", "Alice"), _entry(tmp_path / "b.toml", "alice "))


# -- T_max --------------------------------------------------------------------------------


def test_t_max_counts_from_the_earlier_epoch(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", human={
        "Epoch": "09 Nov 2026 01:00:00 UTC"}, mission={"T_max": "09 Nov 2026 03:51:51.667 UTC"}))
    assert horizon_s(entry, RANKING, "SC_Human", "SC_Alien") == 10311.667
    # Under the onsite rules SC_Alien's epoch is the origin and SC_Human's is chosen.
    assert horizon_s(entry, ONSITE, "SC_Human", "SC_Alien") == 6711.667


def test_under_fixed_epochs_both_are_needed(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", human={"RA": "1 deg"}))
    with pytest.raises(AnnouncementError, match="gives none for SC_Human"):
        horizon_s(entry, RANKING, "SC_Human", "SC_Alien")
    assert horizon_s(entry, ONSITE, "SC_Human", "SC_Alien") == 6711.667


def test_the_announced_seconds_must_be_t_max_less_the_origin(tmp_path: Path) -> None:
    """T_max was typed 96200 on 2026-11-09 against 10 Nov 2026 07:43:24 UTC, 96204 s."""
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", mission={
        "T_max": "09 Nov 2026 03:51:51.667 UTC", "T_max_s": "6707 s"}))
    with pytest.raises(AnnouncementError, match=r"which is 6711\.667 s"):
        horizon_s(entry, RANKING, "SC_Human", "SC_Alien")
    slide = read_entry(_entry(tmp_path / "b.toml", "Bob", mission={
        "T_max": "10 Nov 2026 07:43:24 UTC", "T_max_s": "96,204 s"}, alien={
        **ALIEN, "Epoch": "09 Nov 2026 05:00:00.000 UTC"}, human={
        "Epoch": "09 Nov 2026 05:00:00.000 UTC"}))
    assert horizon_s(slide, ONSITE, "SC_Human", "SC_Alien") == 96204.0


@pytest.mark.parametrize(("t_max", "seconds", "taken"), [
    # Each printed value is rounded where it was printed: T_max to the second here.
    ("09 Nov 2026 03:51:52 UTC", "6711.667 s", True),
    ("09 Nov 2026 03:51:51.667 UTC", "6712 s", True),
    ("09 Nov 2026 03:51:51.667 UTC", "6711 s", False),
    ("09 Nov 2026 03:51:51.667 UTC", "6711.6 s", False),
])
def test_the_seconds_are_held_to_the_rounding_of_all_three(
    tmp_path: Path, t_max: str, seconds: str, taken: bool
) -> None:
    # SC_Human's epoch to the second, SC_Alien's to the millisecond: the finer is the origin.
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", mission={
        "T_max": t_max, "T_max_s": seconds}))
    if taken:
        horizon_s(entry, RANKING, "SC_Human", "SC_Alien")
    else:
        with pytest.raises(AnnouncementError, match="read both again"):
            horizon_s(entry, RANKING, "SC_Human", "SC_Alien")


@pytest.mark.parametrize("t_max", ["09 Nov 2026 01:00:00 UTC", "09 Nov 2026 02:00:00 UTC"])
def test_t_max_not_after_the_origin_is_refused(tmp_path: Path, t_max: str) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", mission={"T_max": t_max}))
    with pytest.raises(AnnouncementError, match="is not after"):
        horizon_s(entry, RANKING, "SC_Human", "SC_Alien")


# -- the script ---------------------------------------------------------------------------


R1_CRAFTS = read_script(R1)


def _crafts(craft: str = "SC_Alien", **fields: str) -> ScriptView:
    """R1's craft with ``fields`` assigned as the script would write them: an existing
    field on its own line, a new one after every other."""
    lines = dict(R1_CRAFTS.lines[craft])
    for field in fields:
        lines.setdefault(field, 1000 + len(lines))
    return ScriptView({**R1_CRAFTS.fields, craft: {**R1_CRAFTS.fields[craft], **fields}},
                      {**R1_CRAFTS.lines, craft: lines})


def test_the_script_is_held_to_the_digits_printed(tmp_path: Path) -> None:
    *_, agreed = _pair(tmp_path)
    wrong, unchecked, agree = against_script(agreed, R1_CRAFTS, "SC_Human", RANKING)
    assert (wrong, unchecked) == ([], [])
    assert "SC_Alien.TA" in agree and "SC_Human.Epoch" in agree


def test_a_value_on_the_edge_of_its_printing_is_held(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", alien={**ALIEN, "INC": "35.3 deg"}))
    assert against_script(entry, _crafts(INC="35.35"), "SC_Human", RANKING)[0] == []
    assert against_script(entry, _crafts(INC="35.3500001"), "SC_Human", RANKING)[0] != []


def test_a_double_printed_to_its_last_digit_is_held(tmp_path: Path) -> None:
    """2026-11-09: the slide printed VMAG 7.175430940911812, the script 7.175430940911815."""
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", human={
        "Epoch": "09 Nov 2026 02:00:00 UTC", "VMAG": "7.175430940911812 km/s"}))
    crafts = _crafts("SC_Human", VMAG="7.175430940911815")
    assert against_script(entry, crafts, "SC_Human", RANKING)[0] == []
    crafts = _crafts("SC_Human", VMAG="7.175430940921812")
    assert against_script(entry, crafts, "SC_Human", RANKING)[0] != []


@pytest.mark.parametrize(("field", "text", "says"), [
    # The day's script against the slide (2026-11-09): ECC, RAAN and TA all differed.
    ("ECC", "1.02339", "SC_Alien.ECC: announced 1.02339, the script 1.0685211112076598"),
    ("TA", "241.214 deg", "SC_Alien.TA: announced 241.214 deg"),
    ("Epoch", "09 Nov 2026 02:00:01 UTC", "SC_Alien.Epoch: announced 09 Nov 2026 02:00:01 UTC,"),
    ("Epoch", "09 Nov 2026 02:00:00.4 UTC", "SC_Alien.Epoch: announced"),
])
def test_a_value_the_script_contradicts_is_named(
    tmp_path: Path, field: str, text: str, says: str
) -> None:
    *_, agreed = read_pair(_entry(tmp_path / "a.toml", "A", alien={**ALIEN, field: text}),
                           _entry(tmp_path / "b.toml", "B", alien={**ALIEN, field: text}))
    wrong, _, _ = against_script(agreed, R1_CRAFTS, "SC_Human", RANKING)
    assert len(wrong) == 1 and says in wrong[0]


def test_an_epoch_is_compared_as_an_instant_to_its_printing(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice"))
    late = _crafts(Epoch="'09 Nov 2026 02:00:00.400'")
    # Announced to the millisecond: 0.4 s off is wrong.
    assert against_script(entry, late, "SC_Human", RANKING)[0] != []
    to_second = read_entry(_entry(tmp_path / "b.toml", "Bob", alien={
        **ALIEN, "Epoch": "09 Nov 2026 02:00:00 UTC"}))
    assert against_script(to_second, late, "SC_Human", RANKING)[0] == []
    other = _crafts(Epoch="'9 Nov 2026 02:00:00.000'")
    assert "SC_Alien.Epoch" in against_script(entry, other, "SC_Human", RANKING)[1][0]


@pytest.mark.parametrize(("fields", "says"), [
    ({"DateFormat": "TAIModJulian", "Epoch": "31000.08333"},
     "SC_Alien.Epoch (the script writes '31000.08333', not in UTCGregorian)"),
    ({"TA": "SC_Human.TA"}, "SC_Alien.TA (the script writes 'SC_Human.TA')"),
])
def test_a_value_the_script_writes_another_way_is_not_compared(
    tmp_path: Path, fields: dict[str, str], says: str
) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice"))
    wrong, unchecked, agree = against_script(entry, _crafts(**fields), "SC_Human", RANKING)
    assert wrong == [] and unchecked == [says]
    assert says.split(" ")[0] not in agree


def test_a_field_the_script_does_not_set_is_not_compared(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", human={
        "Epoch": "09 Nov 2026 02:00:00 UTC", "VMAG": "7 km/s",
        "CoordinateSystem": "EarthMJ2000Eq", "DisplayStateType": "Keplerian"}))
    wrong, unchecked, agree = against_script(entry, R1_CRAFTS, "SC_Human", RANKING)
    assert wrong == [] and unchecked == ["SC_Human.VMAG (the script does not set it)"]
    assert "SC_Human.CoordinateSystem" in agree


def test_a_frame_the_script_leaves_unset_is_gmats_default(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", alien={
        **ALIEN, "CoordinateSystem": "EarthFixed"}))
    fields = {k: v for k, v in R1_CRAFTS.fields["SC_Alien"].items() if k != "CoordinateSystem"}
    view = ScriptView({**R1_CRAFTS.fields, "SC_Alien": fields}, R1_CRAFTS.lines)
    wrong, unchecked, _ = against_script(entry, view, "SC_Human", RANKING)
    assert wrong == []
    assert unchecked[0] == ("SC_Alien.CoordinateSystem (announced EarthFixed, the script reads "
                            "EarthMJ2000Eq)")
    assert "SC_Alien.TA (the script's state is in EarthMJ2000Eq)" in unchecked


def test_a_state_announced_without_its_frame_is_not_compared(tmp_path: Path) -> None:
    alien = {k: v for k, v in ALIEN.items() if k not in ("CoordinateSystem", "DisplayStateType")}
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", alien=alien))
    crafts = _crafts(CoordinateSystem="EarthFixed")
    wrong, unchecked, agree = against_script(entry, crafts, "SC_Human", RANKING)
    assert wrong == [] and unchecked == [
        "SC_Alien.CoordinateSystem (not announced; the script reads EarthFixed)",
        "SC_Alien.DisplayStateType (not announced; the script reads Keplerian)"]
    assert "SC_Alien.TA" in agree


def test_a_craft_the_script_lacks_is_wrong(tmp_path: Path) -> None:
    path = _entry(tmp_path / "a.toml", "Alice")
    path.write_text(path.read_text(encoding="utf-8").replace("[craft.SC_Human]",
                                                             "[craft.HumanSC]"), encoding="utf-8")
    wrong, _, _ = against_script(read_entry(path), R1_CRAFTS, "SC_Human", RANKING)
    assert wrong == ["HumanSC: the script has no such craft (it flies SC_Alien and SC_Human)"]


def test_a_chosen_epoch_is_held_to_the_announced_bound(tmp_path: Path) -> None:
    later = read_entry(_entry(tmp_path / "a.toml", "Alice",
                              human={"Epoch": "09 Nov 2026 01:00:00 UTC"}))
    assert against_script(later, R1_CRAFTS, "SC_Human", ONSITE)[0] == []
    assert against_script(later, R1_CRAFTS, "SC_Human", RANKING)[0] != []
    bound = read_entry(_entry(tmp_path / "b.toml", "Bob",
                              human={"Epoch": "09 Nov 2026 03:00:00 UTC"}))
    assert against_script(bound, R1_CRAFTS, "SC_Human", ONSITE)[0] == [
        "SC_Human.Epoch: announced 09 Nov 2026 03:00:00 UTC or later, the script "
        "09 Nov 2026 02:00:00.000"]


# -- the day's config ---------------------------------------------------------------------


def _tool(tmp_path: Path, *args: str, script: Path | None = None
          ) -> subprocess.CompletedProcess[str]:
    if script is None:
        script = tmp_path / "r1.script"
    if not script.exists():
        # Past its perigee at t = 0, so the T_max check takes any T_max.
        script.write_text(R1.read_text(encoding="utf-8").replace(
            "SC_Alien.TA = 241.21247020102464;", "SC_Alien.TA = 118.78752979897536;"),
            encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ranking_day_config.py"), "--script",
         str(script), *args], capture_output=True, text=True, check=False, timeout=300,
        env=bare_environment(PYTHONPATH=str(ROOT / "src"), PATH="/usr/bin:/bin"))


def test_the_days_config_takes_t_max_and_the_score_from_two_entries(tmp_path: Path) -> None:
    alien = {**ALIEN, "TA": "118.788 deg"}
    a, b = (_entry(tmp_path / f"{n}.toml", n, alien=alien) for n in ("A", "B"))
    first = tmp_path / "day.yaml"
    done = _tool(tmp_path, "--output", str(first), "--team-number", "54",
                 "--announcement", str(a), "--announcement", str(b))
    assert done.returncode == 0, done.stderr
    config = load_config(first)
    assert config.policy.horizon_s == 6711.667 and config.score is None
    assert "SC_Alien.TA, SC_Human.Epoch agree with the script" in done.stdout
    assert "written by step 5's config" in done.stdout
    text = first.read_text(encoding="utf-8")
    assert f"# Announcement entered by A: A.toml, SHA-256 {read_entry(a).sha256}" in text
    scored = tmp_path / "day-scored.yaml"
    done = _tool(tmp_path, "--output", str(scored), "--day-config", str(first),
                 "--announcement", str(a), "--announcement", str(b))
    assert done.returncode == 0, done.stderr
    score = load_config(scored).score
    assert score is not None
    assert (score.kt, score.ct, score.kv, score.cv, score.cr_km, score.delta_v_unit) == (
        0.003, 5369.334, 0.0012, 2000.0, 300.0, "m/s")


def test_a_script_that_is_not_the_announcement_writes_nothing(tmp_path: Path) -> None:
    a, b = (_entry(tmp_path / f"{n}.toml", n) for n in ("A", "B"))
    out = tmp_path / "day.yaml"
    done = _tool(tmp_path, "--output", str(out), "--team-number", "54",
                 "--announcement", str(a), "--announcement", str(b))
    assert done.returncode == 2 and not out.exists()
    assert ("the script is not what was announced: SC_Alien.TA: announced 241.212 deg, the "
            "script 118.78752979897536") in done.stderr


def test_a_value_not_compared_is_printed_as_a_warning(tmp_path: Path) -> None:
    alien = {**ALIEN, "TA": "118.788 deg"}
    human = {"Epoch": "09 Nov 2026 02:00:00 UTC", "RA": "1 deg"}
    a, b = (_entry(tmp_path / f"{n}.toml", n, alien=alien, human=human) for n in ("A", "B"))
    done = _tool(tmp_path, "--output", str(tmp_path / "day.yaml"), "--team-number", "54",
                 "--announcement", str(a), "--announcement", str(b))
    assert done.returncode == 0, done.stderr
    assert ("WARNING: SC_Human.RA (the script does not set it) was announced but not "
            "compared; compare it by hand") in done.stdout


def test_an_announced_boundary_is_written_and_held_to_sc_aliens_coast(tmp_path: Path) -> None:
    alien = {**ALIEN, "TA": "118.788 deg"}
    mission = {"T_max": "09 Nov 2026 03:51:51.667 UTC", "boundary": "58,640.063 km"}
    a, b = (_entry(tmp_path / f"{n}.toml", n, alien=alien, mission=mission) for n in ("A", "B"))
    out = tmp_path / "day.yaml"
    done = _tool(tmp_path, "--output", str(out), "--team-number", "54",
                 "--announcement", str(a), "--announcement", str(b))
    assert done.returncode == 0, done.stderr
    assert load_config(out).policy.mission_boundary_km == 58640.063
    far = {**mission, "boundary": "50000 km"}
    a, b = (_entry(tmp_path / f"{n}.toml", n, alien=alien, mission=far) for n in ("C", "D"))
    done = _tool(tmp_path, "--output", str(tmp_path / "far.yaml"), "--team-number", "54",
                 "--announcement", str(a), "--announcement", str(b))
    assert done.returncode == 2 and "+8640.063 km from the announced" in done.stderr


def test_the_announcement_is_held_to_the_script_before_t_max_is(tmp_path: Path) -> None:
    """The day's script had SC_Alien's perigee after T_max: the contradiction comes first."""
    a, b = (_entry(tmp_path / f"{n}.toml", n, alien={**ALIEN, "TA": "118.788 deg"}, mission={
        "T_max": "09 Nov 2026 02:10:00 UTC"}) for n in ("A", "B"))
    done = _tool(tmp_path, "--output", str(tmp_path / "day.yaml"), "--team-number", "54",
                 "--announcement", str(a), "--announcement", str(b), script=R1)
    assert done.returncode == 2
    assert "SC_Alien.TA: announced 118.788 deg, the script 241.21247020102464" in done.stderr
    assert "perigee" not in done.stderr


@pytest.fixture
def step2(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Step 2's config from two entries that agree with the script."""
    alien = {**ALIEN, "TA": "118.788 deg"}
    a, b = (_entry(tmp_path / f"{n}.toml", n, alien=alien) for n in ("A", "B"))
    first = tmp_path / "day.yaml"
    done = _tool(tmp_path, "--output", str(first), "--team-number", "54",
                 "--announcement", str(a), "--announcement", str(b))
    assert done.returncode == 0, done.stderr
    return first, a, b


def test_step_5_takes_the_score_as_step_2_took_t_max(tmp_path: Path, step2) -> None:
    first, _, _ = step2
    done = _tool(tmp_path, "--output", str(tmp_path / "s.yaml"), "--day-config", str(first),
                 "--horizon-s", "6711.667", "--kt-per-s", "0.0095", "--ct-s", "5369.334",
                 "--kv", "0.0012", "--cv", "2000", "--cr-km", "300", "--delta-v-unit", "m/s")
    assert done.returncode == 2 and "give the score the same way" in done.stderr


@pytest.mark.parametrize(("alien", "mission", "says"), [
    ({"TA": "241.212 deg"}, None, "the script is not what was announced"),
    (None, {"T_max": "09 Nov 2026 03:51:52.667 UTC"}, "--horizon-s 6712.667 here, 6711.667 in"),
])
def test_step_5s_entries_are_held_as_step_2s_were(
    tmp_path: Path, step2, alien: dict[str, str] | None, mission: dict[str, str] | None,
    says: str
) -> None:
    first, _, _ = step2
    alien = {**ALIEN, "TA": "118.788 deg", **(alien or {})}
    c, d = (_entry(tmp_path / f"{n}.toml", n, alien=alien, mission=mission)
            for n in ("C", "D"))
    out = tmp_path / "s.yaml"
    done = _tool(tmp_path, "--output", str(out), "--day-config", str(first),
                 "--announcement", str(c), "--announcement", str(d))
    assert done.returncode == 2 and says in done.stderr and not out.exists()


@pytest.mark.parametrize(("args", "says"), [
    (["--announcement", "a.toml", "--announcement", "b.toml", "--mission-boundary-km", "5e4"],
     "drop --horizon-s, --mission-boundary-km"),
    (["--announcement", "a.toml"], "is given twice"),
    (["--announcement", "a.toml", "--announcement", "b.toml", "--horizon-s", "6711.667"],
     "drop --horizon-s"),
    ([], "give --horizon-s, or two --announcement"),
])
def test_the_announcement_replaces_the_typed_flags(tmp_path: Path, args: list[str],
                                                   says: str) -> None:
    done = _tool(tmp_path, "--output", str(tmp_path / "day.yaml"), "--team-number", "54", *args)
    assert done.returncode == 2 and says in done.stderr


def test_a_scored_config_needs_the_score_in_the_entries(tmp_path: Path) -> None:
    alien = {**ALIEN, "TA": "118.788 deg"}
    a, b = (_entry(tmp_path / f"{n}.toml", n, alien=alien, score=None) for n in ("A", "B"))
    first = tmp_path / "day.yaml"
    assert _tool(tmp_path, "--output", str(first), "--team-number", "54", "--announcement",
                 str(a), "--announcement", str(b)).returncode == 0
    done = _tool(tmp_path, "--output", str(tmp_path / "s.yaml"), "--day-config", str(first),
                 "--announcement", str(a), "--announcement", str(b))
    assert done.returncode == 2 and "give no [score]" in done.stderr


# -- the max review of 2026-11-13 -------------------------------------------------------


@pytest.mark.parametrize(("fields", "says"), [
    # GMAT reads an Epoch in the DateFormat set when its line runs (R2026a: 37 s apart).
    ({"Epoch": "'09 Nov 2026 02:00:00.000'", "DateFormat": "UTCGregorian"},
     "SC_Alien.Epoch (the script sets '09 Nov 2026 02:00:00.000' before its DateFormat)"),
    ({"Epoch.TAIGregorian": "'09 Nov 2026 02:00:37.000'"},
     "SC_Alien.Epoch (the script also sets Epoch.TAIGregorian)"),
])
def test_an_epoch_gmat_reads_otherwise_is_not_compared(
    tmp_path: Path, fields: dict[str, str], says: str
) -> None:
    view = _crafts(**fields)
    if "DateFormat" in fields:
        lines = dict(view.lines["SC_Alien"])
        lines["DateFormat"] = lines["Epoch"] + 1
        view = ScriptView(view.fields, {**view.lines, "SC_Alien": lines})
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice"))
    wrong, unchecked, agree = against_script(entry, view, "SC_Human", RANKING)
    assert wrong == [] and says in unchecked and "SC_Alien.Epoch" not in agree


@pytest.mark.parametrize(("fields", "reason"), [
    ({"AnomalyType": "MA"}, "the script's AnomalyType is MA"),
    ({"Element1": "-90000"}, "the script also sets Element1"),
    ({"CartesianX": "7000"}, "the script also sets CartesianX"),
])
def test_a_state_gmat_reads_through_another_field_is_not_compared(
    tmp_path: Path, fields: dict[str, str], reason: str
) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice"))
    wrong, unchecked, agree = against_script(entry, _crafts(**fields), "SC_Human", RANKING)
    assert wrong == [] and f"SC_Alien.TA ({reason})" in unchecked
    assert "SC_Alien.TA" not in agree


def test_a_state_set_before_its_frame_is_not_compared(tmp_path: Path) -> None:
    view = _crafts()
    lines = dict(view.lines["SC_Alien"])
    lines["CoordinateSystem"] = max(lines.values()) + 1
    view = ScriptView(view.fields, {**view.lines, "SC_Alien": lines})
    wrong, unchecked, _ = against_script(read_entry(_entry(tmp_path / "a.toml", "Alice")), view,
                                         "SC_Human", RANKING)
    assert wrong == []
    assert "SC_Alien.SMA (the script sets its state before its CoordinateSystem)" in unchecked


def test_another_display_type_is_a_warning_and_the_numbers_still_count(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", alien={**ALIEN, "TA": "241.3 deg"}))
    wrong, unchecked, _ = against_script(entry, _crafts(DisplayStateType="Cartesian"),
                                         "SC_Human", RANKING)
    assert unchecked == ["SC_Alien.DisplayStateType (announced Keplerian, the script reads "
                         "Cartesian)"]
    assert wrong == ["SC_Alien.TA: announced 241.3 deg, the script 241.21247020102464"]


@pytest.mark.parametrize(("announced", "written"), [
    ("-118.788 deg", "241.21247020102464"),
    ("601.212 deg", "241.21247020102464"),
    ("359.9999 deg", "-0.00005"),
])
def test_an_angle_is_held_to_its_direction(tmp_path: Path, announced: str, written: str) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", alien={**ALIEN, "TA": announced}))
    assert against_script(entry, _crafts(TA=written), "SC_Human", RANKING)[0] == []
    off = read_entry(_entry(tmp_path / "b.toml", "Bob", alien={**ALIEN, "TA": "61.212 deg"}))
    assert against_script(off, _crafts(TA=written), "SC_Human", RANKING)[0] != []


def test_a_script_value_past_a_double_is_not_compared(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice"))
    wrong, unchecked, _ = against_script(entry, _crafts(ECC="1e1000000"), "SC_Human", RANKING)
    assert wrong == [] and "SC_Alien.ECC (the script writes '1e1000000')" in unchecked


def test_t_max_counts_from_the_scripts_own_epoch(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", human={
        "Epoch": "09 Nov 2026 02:00:00 UTC"}, alien={**ALIEN, "Epoch": "09 Nov 2026 02:00:00 UTC"},
        mission={"T_max": "09 Nov 2026 03:51:51.667 UTC"}))
    written = {"SC_Alien": datetime.datetime(2026, 11, 9, 2, 0, 0, 400_000),
               "SC_Human": datetime.datetime(2026, 11, 9, 2, 0, 0, 400_000)}
    assert horizon_s(entry, RANKING, "SC_Human", "SC_Alien", written) == 6711.267


def test_t_max_s_printed_finer_than_t_max_is_the_one_taken(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", mission={
        "T_max": "09 Nov 2026 03:51:52 UTC", "T_max_s": "6711.667 s"}, human={
        "Epoch": "09 Nov 2026 02:00:00.000 UTC"}))
    assert horizon_s(entry, RANKING, "SC_Human", "SC_Alien") == 6711.667
    seconds = read_entry(_entry(tmp_path / "b.toml", "Bob", mission={"T_max_s": "6711.667 s"}))
    assert seconds.t_max is None
    assert horizon_s(seconds, RANKING, "SC_Human", "SC_Alien") == 6711.667


def test_an_entry_of_the_score_alone_is_read(tmp_path: Path) -> None:
    path = tmp_path / "a.toml"
    path.write_text('format = "orbitwin-announcement/1"\nentered_by = "Alice"\n'
                    + _toml({"score": SCORE}), encoding="utf-8")
    entry = read_entry(path)
    assert entry.t_max is None and entry.score is not None and entry.epochs == {}
    path.write_text('format = "orbitwin-announcement/1"\nentered_by = "Alice"\n',
                    encoding="utf-8")
    with pytest.raises(AnnouncementError, match="neither"):
        read_entry(path)


@pytest.mark.parametrize(("score", "mission", "says"), [
    ({**SCORE, "C_r": "0 km"}, None, "score.C_r = '0 km' is not positive"),
    ({**SCORE, "C_r": "-300 km"}, None, "is not positive"),
    (SCORE, {"T_max": "09 Nov 2026 03:51:51.667 UTC", "boundary": "0 km"}, "boundary"),
])
def test_a_value_that_must_be_positive_is_refused_as_read(
    tmp_path: Path, score: dict[str, str], mission: dict[str, str] | None, says: str
) -> None:
    with pytest.raises(AnnouncementError, match=says):
        read_entry(_entry(tmp_path / "a.toml", "Alice", score=score, mission=mission))


@pytest.mark.parametrize("by", ["Alice\nscore", "Alice\N{LINE SEPARATOR}", "Ali\x85ce"])
def test_entered_by_is_one_line(tmp_path: Path, by: str) -> None:
    path = _entry(tmp_path / "a.toml", "PLACEHOLDER")
    path.write_text(path.read_text(encoding="utf-8").replace(
        '"PLACEHOLDER"', json.dumps(by)), encoding="utf-8")
    with pytest.raises(AnnouncementError, match="not one line"):
        read_entry(path)


def test_what_cannot_be_read_as_toml_is_refused(tmp_path: Path) -> None:
    path = _entry(tmp_path / "a.toml", "Alice")
    path.write_text(path.read_text(encoding="utf-8") + "x = " + "9" * 5000 + "\n",
                    encoding="utf-8")
    with pytest.raises(AnnouncementError, match="ValueError"):
        read_entry(path)
    with pytest.raises(AnnouncementError, match="is not a file"):
        read_entry(tmp_path)


def test_the_hash_is_of_the_bytes_read(tmp_path: Path) -> None:
    path = _entry(tmp_path / "a.toml", "Alice")
    import hashlib

    assert read_entry(path).sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_step_5_takes_a_zero_burn_distance_with_entries(tmp_path: Path, step2) -> None:
    first, a, b = step2
    done = _tool(tmp_path, "--output", str(tmp_path / "s.yaml"), "--day-config", str(first),
                 "--announcement", str(a), "--announcement", str(b),
                 "--zero-burn-dmin-km", "604.311")
    assert done.returncode == 0, done.stderr
    assert "the zero burn's closest approach" in done.stdout


def test_step_5_takes_entries_of_the_score_alone(tmp_path: Path, step2) -> None:
    first, _, _ = step2
    paths = []
    for who in ("C", "D"):
        path = tmp_path / f"{who}.toml"
        path.write_text(f'format = "orbitwin-announcement/1"\nentered_by = "{who}"\n'
                        + _toml({"score": SCORE}), encoding="utf-8")
        paths.append(path)
    out = tmp_path / "s.yaml"
    done = _tool(tmp_path, "--output", str(out), "--day-config", str(first),
                 "--announcement", str(paths[0]), "--announcement", str(paths[1]))
    assert done.returncode == 0, done.stderr
    assert load_config(out).policy.horizon_s == 6711.667
    alone = _tool(tmp_path, "--output", str(tmp_path / "t.yaml"), "--team-number", "54",
                  "--announcement", str(paths[0]), "--announcement", str(paths[1]))
    assert alone.returncode == 2 and "give no T_max" in alone.stderr


# -- the review of the max review's fixes ------------------------------------------------


@pytest.mark.parametrize(("left_out", "says"), [
    ("DisplayStateType", "SC_Alien.DisplayStateType (announced Keplerian, the script reads "
     "Cartesian)"),
    ("DateFormat", "SC_Alien.Epoch (the script writes '09 Nov 2026 02:00:00.000', not in "
     "UTCGregorian)"),
])
def test_a_field_left_unset_is_read_as_gmats_default(
    tmp_path: Path, left_out: str, says: str
) -> None:
    fields = {k: v for k, v in R1_CRAFTS.fields["SC_Alien"].items() if k != left_out}
    lines = {k: v for k, v in R1_CRAFTS.lines["SC_Alien"].items() if k != left_out}
    view = ScriptView({**R1_CRAFTS.fields, "SC_Alien": fields},
                      {**R1_CRAFTS.lines, "SC_Alien": lines})
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice"))
    assert says in against_script(entry, view, "SC_Human", RANKING)[1]


def test_t_max_s_printed_coarser_than_t_max_is_not_taken(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", mission={
        "T_max": "09 Nov 2026 03:51:51.667 UTC", "T_max_s": "6712 s"}, human={
        "Epoch": "09 Nov 2026 02:00:00.000 UTC"}))
    assert horizon_s(entry, RANKING, "SC_Human", "SC_Alien") == 6711.667


def test_the_origin_is_the_earlier_epoch_the_script_writes(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", human={
        "Epoch": "10 Nov 2026 00:00:00 UTC"}, alien={**ALIEN, "Epoch": "10 Nov 2026 00:00:00 UTC"},
        mission={"T_max": "11 Nov 2026 00:00:00.000 UTC"}))
    # SC_Human comes first among equal announced epochs; the script starts SC_Alien first.
    written = {"SC_Human": datetime.datetime(2026, 11, 10, 0, 0, 0, 40_000),
               "SC_Alien": datetime.datetime(2026, 11, 9, 23, 59, 59, 600_000)}
    assert horizon_s(entry, RANKING, "SC_Human", "SC_Alien", written) == 86400.4


def test_t_max_s_counted_by_the_organizer_holds_against_the_scripts_epoch(
    tmp_path: Path
) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", human={
        "Epoch": "09 Nov 2026 02:00:00 UTC"}, alien={**ALIEN, "Epoch": "09 Nov 2026 02:00:00 UTC"},
        mission={"T_max": "09 Nov 2026 03:51:51.667 UTC", "T_max_s": "6711.667 s"}))
    written = dict.fromkeys(("SC_Human", "SC_Alien"),
                            datetime.datetime(2026, 11, 9, 2, 0, 0, 400_000))
    assert horizon_s(entry, RANKING, "SC_Human", "SC_Alien", written) == 6711.267
    late = read_entry(_entry(tmp_path / "b.toml", "Bob", human={
        "Epoch": "09 Nov 2026 02:00:00 UTC"}, alien={**ALIEN, "Epoch": "09 Nov 2026 02:00:00 UTC"},
        mission={"T_max": "09 Nov 2026 03:51:51.667 UTC", "T_max_s": "6712.3 s"}))
    with pytest.raises(AnnouncementError, match="SC_Human's epoch in the script"):
        horizon_s(late, RANKING, "SC_Human", "SC_Alien", written)


def test_an_angle_too_far_to_be_a_rounding_is_wrong_not_an_error(tmp_path: Path) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice"))
    wrong, _, _ = against_script(entry, _crafts(RAAN="5.7046e31"), "SC_Human", RANKING)
    assert wrong == ["SC_Alien.RAAN: announced 57.046 deg, the script 5.7046e31"]


def test_the_scripts_fields_stop_at_the_mission_sequence(tmp_path: Path) -> None:
    text = R1.read_text(encoding="utf-8")
    script = tmp_path / "late.script"
    script.write_text(text.replace("BeginMissionSequence;",
                                   "BeginMissionSequence;\nSC_Alien.TA = 1;", 1),
                      encoding="utf-8")
    view = read_script(script)
    assert view.fields["SC_Alien"]["TA"] == R1_CRAFTS.fields["SC_Alien"]["TA"]
    assert view.lines["SC_Alien"]["TA"] == R1_CRAFTS.lines["SC_Alien"]["TA"]


def test_the_days_config_counts_t_max_from_the_scripts_epoch(tmp_path: Path) -> None:
    script = tmp_path / "r1-late.script"
    script.write_text(R1.read_text(encoding="utf-8").replace(
        "SC_Alien.TA = 241.21247020102464;", "SC_Alien.TA = 118.78752979897536;").replace(
        "'09 Nov 2026 02:00:00.000'", "'09 Nov 2026 02:00:00.400'"), encoding="utf-8")
    second = {"Epoch": "09 Nov 2026 02:00:00 UTC"}
    a, b = (_entry(tmp_path / f"{n}.toml", n, human=second,
                   alien={**ALIEN, "TA": "118.788 deg", **second}) for n in ("A", "B"))
    out = tmp_path / "day.yaml"
    done = _tool(tmp_path, "--output", str(out), "--team-number", "54",
                 "--announcement", str(a), "--announcement", str(b), script=script)
    assert done.returncode == 0, done.stderr
    assert load_config(out).policy.horizon_s == 6711.267


@pytest.mark.parametrize(("written", "holds"), [
    ("1090.0", True),            # three turns past 10
    ("3600000000010", True),     # 1e13 turns past, still a double's
    ("36000000000000000010", False),  # past 1e15: no longer a direction
])
def test_an_angle_wraps_only_while_a_double_can_tell(
    tmp_path: Path, written: str, holds: bool
) -> None:
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", alien={**ALIEN, "TA": "10 deg"}))
    wrong = against_script(entry, _crafts(TA=written), "SC_Human", RANKING)[0]
    assert (wrong == []) is holds


def test_t_max_s_finer_than_t_max_and_an_announced_epoch_is_taken(tmp_path: Path) -> None:
    """The script's epoch not read: the seconds from the announced one carry its 0.5 s."""
    entry = read_entry(_entry(tmp_path / "a.toml", "Alice", human={
        "Epoch": "09 Nov 2026 02:00:00 UTC"}, alien={**ALIEN, "Epoch": "09 Nov 2026 02:00:00 UTC"},
        mission={"T_max": "09 Nov 2026 03:51:51.667 UTC", "T_max_s": "6711.267 s"}))
    assert horizon_s(entry, RANKING, "SC_Human", "SC_Alien") == 6711.267
