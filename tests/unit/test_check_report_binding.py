"""check_report_rows holds a report to the script given with it by hand.

tests/fixtures/report_bind holds whole GMAT R2026a reports, each beside the
script that wrote it: R1's certified hit at MaxStep 1, the phase-review
rehearsals flown at MaxStep 60, R8 at 2700 and R3 printed to 6-8 digits, xz'd
where large (MANIFEST.json has each one's SHA-256). No rows are cut: a report
missing rows is refused. Step 7 on the GUI path has no certificate, so nothing
else says the report is the script's.
"""

from __future__ import annotations

import importlib.util
import json
import lzma
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import yaml

from orbitwin.errors import ParseError
from orbitwin.gmat_parser import mission_schedule

ROOT = Path(__file__).resolve().parents[2]
BIND = ROOT / "tests" / "fixtures" / "report_bind"
MANIFEST = json.loads((BIND / "MANIFEST.json").read_text(encoding="utf-8"))
UPLOADED = "ODC_2026_Team54.txt"
#: (T_max, Ct, the time origin's window) of each rehearsal's day config.
DAYS = {
    "R1": (6711.667, 5369.333600000001, None),
    "R3": (5092.815, 4074.252, None),
    "R5": (5791.747, 4633.3976, None),
    "R6": (8511.667, 6809.3336, "after_alignment"),
    "R8": (8511.667, 6809.3336, "after_alignment"),
}
DAYS["R1x"] = DAYS["R1"]
#: certification.json's final_script_parity.replayed_score of each hit; GMAT flew the
#: MaxStep 60 variants to the same last row.
CERTIFIED = {"R1": 96.39083960116564, "R3": 81.39182695241871, "R6": 95.06564141437596,
             "R8": 95.06564766961542}


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_report_rows", ROOT / "scripts" / "check_report_rows.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = _load()


def run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        *args: str | Path) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["check_report_rows.py", *map(str, args)])
    try:
        code = CHECK.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
    out, err = capsys.readouterr()
    return code, out, err


def day(tmp_path: Path, case: str) -> Path:
    """The rehearsal's day config as ranking_day_config wrote it."""
    horizon_s, ct, window = DAYS[case.split("_")[0]]
    raw = yaml.safe_load((ROOT / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"].update(horizon_s=horizon_s, mission_time_origin="earliest")
    if window is not None:
        raw["policy"]["closest_approach_window"] = window
    raw["score"] = {"kt": 0.003, "ct": ct, "kv": 0.0012, "cv": 2000.0, "time_unit": "s",
                    "delta_v_unit": "m/s", "cr_km": 300.0}
    path = tmp_path / f"{case}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def report_bytes(case: str) -> bytes:
    """A fixture report's bytes as GMAT wrote them."""
    plain = BIND / f"{case}.txt"
    return plain.read_bytes() if plain.is_file() else lzma.decompress(
        (BIND / f"{case}.txt.xz").read_bytes())


def report_text(case: str) -> str:
    return report_bytes(case).decode("utf-8")


def fixture(tmp_path: Path, case: str) -> Path:
    """A fixture report as a file under its fixture name, as a stray copy would be."""
    path = tmp_path / "fixtures" / f"{case}.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(report_bytes(case))
    return path


def upload(tmp_path: Path, case: str, name: str = UPLOADED) -> Path:
    """A fixture report as GMAT left it, under its ReportFile's name."""
    folder = tmp_path / f"usb-{case}-{name}"
    folder.mkdir()
    (folder / name).write_bytes(report_bytes(case))
    return folder / name


def script(tmp_path: Path, case: str, old: str | None = None, new: str = "") -> Path:
    """A fixture script, with `old` replaced by `new`."""
    text = (BIND / f"{case}.script").read_text(encoding="utf-8")
    if old is not None:
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / f"{case}-{len(new)}.script"
    path.write_text(text, encoding="utf-8")
    return path


def frozen(case: str) -> list[str]:
    """--frozen-rows for an aligned fixture: the waiting craft's rows before t = 0."""
    rows = MANIFEST[case]["frozen_rows"]
    return ["--frozen-rows", str(rows)] if rows else []


def _score(out: str) -> float:
    line = next(line for line in out.splitlines() if "-> score" in line or "scored miss:" in line)
    return float(line.rsplit(maxsplit=1)[-1])


@pytest.mark.parametrize(
    "case", ["R1_hit", "R1_hit_step60", "R3_hit_step60", "R6_hit_step60", "R8_hit_step60"])
def test_a_certified_report_passes_against_its_script(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R6 (SC_Alien first) and R8 (SC_Human first) are aligned; R3 has 12 burns."""
    code, out, err = run(monkeypatch, capsys, upload(tmp_path, case), "--config",
                         day(tmp_path, case), "--script", BIND / f"{case}.script", *frozen(case))
    assert code == 0, out + err
    burns = MANIFEST[case]["burns"]
    assert f"hold its {burns} burn{'' if burns == 1 else 's'} and no other" in out
    # Σ|Δv| from the script and T_team from the rows give certify's replayed score.
    assert _score(out) == pytest.approx(CERTIFIED[case.split("_")[0]], abs=5.0e-5)


def test_a_13_burn_miss_passes_against_its_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(monkeypatch, capsys, upload(tmp_path, "R5_miss_step60"), "--config",
                         day(tmp_path, "R5"), "--script", BIND / "R5_miss_step60.script")
    assert code == 0 and "hold its 13 burns and no other" in out, out + err
    assert "a scored miss" in out

@pytest.mark.parametrize(
    "case", ["R8_hit_step60", "R8_hit_step2700", "R3_hit_prec6", "R3_hit_prec7", "R3_hit_prec8"])
def test_a_report_the_review_of_185_refused_passes(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R8 at MaxStep 60 and 2700 ends the alignment on two rows, the pre-burn state and the
    post-burn one; R3 printed to 6-8 digits shifts each row's time by its last digit."""
    code, out, err = run(monkeypatch, capsys, upload(tmp_path, case), "--config",
                         day(tmp_path, case), "--script", BIND / f"{case}.script", *frozen(case))
    assert code == 0, out + err
    assert f"hold its {MANIFEST[case]['burns']} burns and no other" in out
    assert "VERDICT: the LAST row is inside the sphere." in out


def test_the_zero_burn_template_passes_against_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = upload(tmp_path, "R1_zero_step60", "ODC_2026_TeamXX.txt")
    code, out, err = run(monkeypatch, capsys, report, "--config", day(tmp_path, "R1"),
                         "--script", BIND / "R1_zero_step60.script")
    assert code == 0 and "a scored miss" in out and "delta-v 22.9207" in out, out + err


def test_another_candidates_report_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R1's second certified candidate, the same name and format: main passed it, 95.5624."""
    config = day(tmp_path, "R1")
    code, out, err = run(monkeypatch, capsys, upload(tmp_path, "R1_other_step60"), "--config",
                         config, "--script", BIND / "R1_hit_step60.script")
    assert code == 2 and "VERDICT" not in out
    assert "ends at 4518.038474" in err and "Propagate goals end at 4301.437731" in err
    # And the other way round.
    code, _, err = run(monkeypatch, capsys, upload(tmp_path, "R1_hit_step60"), "--config",
                       config, "--script", BIND / "R1_other_step60.script")
    assert code == 2 and "ends at 4301.437731" in err


def test_another_candidate_ending_at_the_same_time_is_refused_on_its_burn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R1_other's goals with R1_hit's burn: only the jump in the rows tells them apart."""
    other = (BIND / "R1_other_step60.script").read_text(encoding="utf-8")
    hit = (BIND / "R1_hit_step60.script").read_text(encoding="utf-8")
    elements = [line for line in hit.splitlines(keepends=True) if ".Element" in line]
    theirs = [line for line in other.splitlines(keepends=True) if ".Element" in line]
    for mine, own in zip(elements, theirs, strict=True):
        other = other.replace(own, mine)
    swapped = tmp_path / "swapped.script"
    swapped.write_text(other, encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, upload(tmp_path, "R1_other_step60"), "--config",
                       day(tmp_path, "R1"), "--script", swapped)
    assert code == 2 and "the velocity jumps 0.09941" in err, err
    assert "burns 0.217284394 km/s there" in err


def test_a_zero_burn_report_is_refused_against_a_burning_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The R1 template flown as it is, renamed as ours: main scored it a miss at 24.6193."""
    config = day(tmp_path, "R1")
    report = upload(tmp_path, "R1_zero_step60")
    code, _, err = run(monkeypatch, capsys, report, "--config", config,
                       "--script", BIND / "R1_hit_step60.script")
    assert code == 2 and "ends at 6711.667000" in err
    # Its goals made to end at T_max, the burn where the zero burn wrote no row ...
    goal = "SC_Alien.ElapsedSecs = 4301.436731235679, SC_Human.ElapsedSecs = 4301.436731235679"
    at_tmax = "SC_Alien.ElapsedSecs = 6711.666, SC_Human.ElapsedSecs = 6711.666"
    code, _, err = run(monkeypatch, capsys, report, "--config", config, "--script",
                       script(tmp_path, "R1_hit_step60", goal, at_tmax))
    assert code == 2 and "has no row at burn 1 (ImpulsiveBurn1, t = 0.001000" in err
    # ... and where it did, at 60 s.
    first = "SC_Alien.ElapsedSecs = 0.001, SC_Human.ElapsedSecs = 0.001"
    moved = script(tmp_path, "R1_hit_step60", goal, at_tmax).read_text(encoding="utf-8").replace(
        first, "SC_Alien.ElapsedSecs = 60, SC_Human.ElapsedSecs = 60").replace(
        "6711.666", "6651.667")
    path = tmp_path / "at_sixty.script"
    path.write_text(moved, encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, report, "--config", config, "--script", path)
    assert code == 2 and "the velocity jumps" in err and "burns 0.217284394 km/s" in err, err


def test_a_burning_report_is_refused_against_a_script_without_its_burns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R5's 13-burn miss against R5's template, which ends at T_max too."""
    text = (BIND / "R5_miss_step60.script").read_text(encoding="utf-8")
    begin = text.index("BeginMissionSequence;")
    template = tmp_path / "template.script"
    template.write_text(text[:begin] + "BeginMissionSequence;\nPropagate DefaultProp(SC_Alien, "
                        "SC_Human) {SC_Alien.ElapsedSecs = 5791.747, SC_Human.ElapsedSecs = "
                        "5791.747};\n", encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, upload(tmp_path, "R5_miss_step60"), "--config",
                         day(tmp_path, "R5"), "--script", template)
    assert code == 2 and "VERDICT" not in out
    assert "has no burn: not this script's report" in err, err


def test_a_renamed_report_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(monkeypatch, capsys, upload(tmp_path, "R1_hit", "ODC_2026_TeamXX.txt"),
                       "--config", day(tmp_path, "R1"), "--script", BIND / "R1_hit.script")
    assert code == 2 and "is named 'ODC_2026_TeamXX.txt'" in err
    assert "writes 'ODC_2026_Team54.txt'" in err


def test_a_total_dv_in_the_wrong_unit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The hand-typed km/s total main scored 96.9445 instead of 96.3908."""
    report, config = upload(tmp_path, "R1_hit"), day(tmp_path, "R1")
    flags = ["--config", config, "--script", BIND / "R1_hit.script"]
    code, out, err = run(monkeypatch, capsys, report, *flags, "--total-dv", "0.2173")
    assert code == 2 and "VERDICT" not in out
    assert "--total-dv is 0.2173 m/s" in err and "217.284394 m/s (0.217284394 km/s)" in err
    code, out, _ = run(monkeypatch, capsys, report, *flags, "--total-dv", "217.28439409131667")
    assert code == 0 and _score(out) == pytest.approx(96.3908, abs=5.0e-5)
    code, without, _ = run(monkeypatch, capsys, report, *flags)
    assert code == 0 and without == out


def test_an_alignment_must_be_in_both_or_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    alignment = "Propagate DefaultProp(SC_Human) {SC_Human.UTCModJulian = SC_Alien.UTCModJulian};\n"
    code, _, err = run(monkeypatch, capsys, upload(tmp_path, "R8_hit_step60"), "--config",
                       day(tmp_path, "R8"), "--script",
                       script(tmp_path, "R8_hit_step60", alignment), *frozen("R8_hit_step60"))
    assert code == 2 and "holds SC_Alien still for 33 rows" in err and "has none" in err


def _without_velocity(tmp_path: Path, case: str, name: str, text: str | None = None
                      ) -> tuple[Path, Path]:
    """A fixture and its script with SC_Human's VX renamed out of reach, as a report in
    another frame has it."""
    lines = report_text(case).splitlines(keepends=True)
    folder = tmp_path / f"usb-{case}"
    folder.mkdir()
    (folder / name).write_text(
        "".join([lines[0].replace("SC_Human.EarthMJ2000Eq.VX", "SC_Human.EarthMJ2000Eq.VR"),
                 *lines[1:]]), encoding="utf-8")
    text = text if text is not None else (BIND / f"{case}.script").read_text(encoding="utf-8")
    path = tmp_path / f"{case}-vr.script"
    path.write_text(text.replace("SC_Human.EarthMJ2000Eq.VX", "SC_Human.EarthMJ2000Eq.VR"),
                    encoding="utf-8")
    return folder / name, path


def test_a_report_without_the_pursuers_velocity_needs_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review of #185, P2-2: only its end binds it, so a burn script passes on that alone."""
    report, path = _without_velocity(tmp_path, "R1_hit", UPLOADED)
    config = day(tmp_path, "R1")
    code, out, err = run(monkeypatch, capsys, report, "--config", config, "--script", path)
    assert code == 2 and "VERDICT" not in out
    assert "its burns are not checked" in err and "--accept-end-time-binding-only" in err
    code, out, err = run(monkeypatch, capsys, report, "--config", config, "--script", path,
                         "--accept-end-time-binding-only")
    assert code == 3 and "WARNING: " in out and "6b.7's by-eye comparison is the only" in out
    assert "hold its" not in out, out + err
    assert out.splitlines()[-1] == CHECK.UNBOUND_LINE
    # --binding warn does not stand in for the flag.
    code, _, err = run(monkeypatch, capsys, report, "--config", config, "--script", path,
                       "--binding", "warn")
    assert code == 2 and "--accept-end-time-binding-only" in err


def test_without_velocity_a_burn_still_needs_its_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The zero-burn template, held to a burn at 0.001 s whose goals end at T_max."""
    text = (BIND / "R1_hit_step60.script").read_text(encoding="utf-8").replace(
        "4301.436731235679", "6711.666")
    report, path = _without_velocity(tmp_path, "R1_zero_step60", UPLOADED, text)
    code, _, err = run(monkeypatch, capsys, report, "--config", day(tmp_path, "R1"),
                       "--script", path, "--accept-end-time-binding-only")
    assert code == 2 and "has no row at burn 1 (ImpulsiveBurn1, t = 0.001000" in err, err


def test_a_windows_copy_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The organizer's GUI runs on Windows: the same rows with CRLF line ends."""
    folder = tmp_path / "usb"
    folder.mkdir()
    text = report_text("R1_hit")
    (folder / UPLOADED).write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    code, out, err = run(monkeypatch, capsys, folder / UPLOADED, "--config", day(tmp_path, "R1"),
                         "--script", BIND / "R1_hit.script")
    assert code == 0 and "hold its 1 burn and no other" in out, out + err


@pytest.mark.parametrize("case", sorted(MANIFEST))
def test_the_schedule_is_the_scripts(case: str) -> None:
    """Burn count, alignment and end as the rehearsal's certificate flew them."""
    schedule = mission_schedule(BIND / f"{case}.script")
    assert len(schedule.maneuvers) == MANIFEST[case]["burns"]
    assert schedule.alignment_craft == MANIFEST[case]["alignment_craft"]
    assert [m.time_s for m in schedule.maneuvers] == sorted(m.time_s for m in schedule.maneuvers)


def test_the_schedule_agrees_with_the_parser() -> None:
    """Where parse_gmat_script reads the states in-process, the two readings agree."""
    import dataclasses

    from orbitwin.burn_rows import maneuvers_flown
    from orbitwin.config import load_config
    from orbitwin.gmat_parser import parse_gmat_script

    config = load_config(ROOT / "config" / "ranking.yaml")
    for case, horizon_s in (("R1_hit", 6711.667), ("R3_hit_step60", 5092.815),
                            ("R5_miss_step60", 5791.747)):
        config = dataclasses.replace(
            config, policy=dataclasses.replace(config.policy, horizon_s=horizon_s))
        scenario, _ = parse_gmat_script(BIND / f"{case}.script", config)
        flown = maneuvers_flown(scenario)
        schedule = mission_schedule(BIND / f"{case}.script")
        assert [m.time_s for m in flown] == [m.time_s for m in schedule.maneuvers], case
        assert [tuple(m.components_km_s) for m in flown] == [
            m.components_km_s for m in schedule.maneuvers], case
        assert schedule.end_s == pytest.approx(
            sum(segment.duration_s for segment in scenario.propagate_segments), abs=1e-9)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("Report ReportFile1 SC_Human.ElapsedSecs;\n", "neither a Propagate nor a Maneuver"),
        ("Propagate DefaultProp(SC_Human) {SC_Human.ElapsedSecs = 5};\n", "alone after the first"),
        ("Propagate DefaultProp(SC_Alien, SC_Human) {SC_Human.Periapsis};\n", "not on elapsed"),
        ("Maneuver Nothing(SC_Human);\n", "undeclared ImpulsiveBurn Nothing"),
    ],
)
def test_the_schedule_fails_closed(line: str, message: str, tmp_path: Path) -> None:
    text = (BIND / "R1_hit.script").read_text(encoding="utf-8")
    path = tmp_path / "odd.script"
    path.write_text(text.rstrip("\n") + "\n" + line, encoding="utf-8")
    with pytest.raises(ParseError, match=message):
        mission_schedule(path)


def test_the_schedule_refuses_an_element_alias(tmp_path: Path) -> None:
    """E2E audit A4: a V after the Elements made GMAT fly 70.69 m/s, not 91.15."""
    text = (BIND / "R1_hit.script").read_text(encoding="utf-8")
    path = tmp_path / "alias.script"
    path.write_text(text.replace("BeginMissionSequence;", "ImpulsiveBurn1.V = 0;\n"
                                 "BeginMissionSequence;"), encoding="utf-8")
    with pytest.raises(ParseError, match="GMAT reads V as Element1"):
        mission_schedule(path)


def test_a_burn_of_the_right_size_in_another_direction_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    text = (BIND / "R1_hit.script").read_text(encoding="utf-8")
    values = {k: text.split(f"ImpulsiveBurn1.Element{k} = ")[1].split(";")[0] for k in "12"}
    swapped = text.replace(f"Element1 = {values['1']};", "Element1 = X;").replace(
        f"Element2 = {values['2']};", f"Element2 = {values['1']};").replace(
        "Element1 = X;", f"Element1 = {values['2']};")
    path = tmp_path / "swapped.script"
    path.write_text(swapped, encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, upload(tmp_path, "R1_hit"), "--config",
                       day(tmp_path, "R1"), "--script", path)
    assert code == 2 and "burns 0.217284394 km/s there" in err and "apart as a vector" in err


def test_rows_that_do_not_continue_after_a_burn_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The row after the burn moved 1 km: a splice, not a flight."""
    header, data = CHECK.read_report(str(fixture(tmp_path, "R1_hit")))
    data = data.copy()
    data[2, header.index("SC_Human.EarthMJ2000Eq.X")] += 1.0
    folder = tmp_path / "usb"
    folder.mkdir()
    lines = ["  ".join(header) + "\n", *("  ".join(repr(float(x)) for x in row) + "\n"
                                       for row in data)]
    (folder / UPLOADED).write_text("".join(lines), encoding="utf-8")
    text = (BIND / "R1_hit.script").read_text(encoding="utf-8").replace(
        "ReportFile1.FixedWidth = true;", "ReportFile1.FixedWidth = false;")
    path = tmp_path / "loose.script"
    path.write_text(text, encoding="utf-8")
    code, _, err = run(monkeypatch, capsys, folder / UPLOADED, "--config", day(tmp_path, "R1"),
                       "--script", path)
    assert code == 2 and "the rows do not continue" in err, err


def test_the_waiting_craft_is_not_the_one_the_alignment_flies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R6 holds SC_Human still; R8's script flies SC_Human through its alignment."""
    code, _, err = run(monkeypatch, capsys, upload(tmp_path, "R6_hit_step60"), "--config",
                       day(tmp_path, "R6"), "--script", BIND / "R8_hit_step60.script",
                       *frozen("R6_hit_step60"))
    assert code == 2 and "holds SC_Human still, and" in err and "alignment flies it" in err


@pytest.mark.parametrize(("cut", "frozen_rows", "said"), [
    # All 30 rows before t = 0: the evidence of the docs review, which passed with exit 0.
    (slice(0, 30), 0, "holds no craft still as its rows do"),
    (slice(0, 29), 1, "its first rows were cut"),
    (slice(5, 20), 15, "rows missing in the epoch alignment between t = 240.000000 and "
                       "t = 1200.000000 on SC_Alien.ElapsedSecs"),
    # One row and two in a row: 120 s and 180 s where MaxStep is 60 s (review of #191).
    (slice(10, 11), 29, "rows missing in the epoch alignment between t = 540.000000 and "
                        "t = 660.000000 on SC_Alien.ElapsedSecs, 120 s apart"),
    (slice(10, 12), 28, "rows missing in the epoch alignment between t = 540.000000 and "
                        "t = 720.000000 on SC_Alien.ElapsedSecs, 180 s apart"),
])
def test_a_report_cut_from_its_epoch_alignment_is_refused(
    cut: slice, frozen_rows: int, said: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Review of the docs (RUNBOOK appendix D): R6 with its alignment rows removed held to its
    script. Under from_origin those rows decide d_min."""
    lines = report_text("R6_hit_step60").splitlines(keepends=True)
    data = lines[1:]
    del data[cut]
    path = tmp_path / "usb" / UPLOADED
    path.parent.mkdir()
    path.write_text(lines[0] + "".join(data), encoding="utf-8")
    argv = [path, "--config", day(tmp_path, "R6"), "--script", BIND / "R6_hit_step60.script",
            "--frozen-rows", str(frozen_rows)]
    code, out, err = run(monkeypatch, capsys, *argv)
    assert code == 2 and said in err and "VERDICT" not in out, out + err
    # Not a burn check: --binding warn refuses it too. A cut alignment is not the flight.
    code, out, err = run(monkeypatch, capsys, *argv, "--binding", "warn")
    assert code == 2 and said in err and "VERDICT" not in out, out + err


def _altitude_set(tmp_path: Path, case: str, rows: range, km: float) -> Path:
    """`case`'s report with SC_Human.Earth.Altitude set to `km` on data `rows`."""
    lines = report_text(case).splitlines()
    names = lines[0].split()
    column = names.index("SC_Human.Earth.Altitude")
    out = [" ".join(names)]
    for k, line in enumerate(lines[1:]):
        values = line.split()
        if k in rows:
            values[column] = repr(km)
        out.append(" ".join(values))
    path = tmp_path / "usb" / UPLOADED
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(("case", "rows", "code"), [
    # R6 holds SC_Human at its template state through the alignment (S1).
    ("R6_hit_step60", range(0, 31), 5),
    # R8 flies SC_Human through its alignment: still the template's, before any burn.
    ("R8_hit_step60", range(10, 11), 5),
    ("R8_hit_step60", range(0, 1), 5),
    # After the alignment our burns fly it.
    ("R8_hit_step60", range(40, 41), 2),
])
def test_below_the_surface_in_the_alignment_is_the_templates(
    case: str, rows: range, code: int, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Review of #191's delta, P2-3: the template's own state before the pursuer's epoch was
    judged as our design, DISQUALIFIED at data row 1."""
    argv = [_altitude_set(tmp_path, case, rows, -0.001), "--config",
            day(tmp_path, case.split("_")[0]), *frozen(case)]
    got, out, err = run(monkeypatch, capsys, *argv)
    assert got == code, out + err
    if code == 5:
        assert "DISQUALIFIED" not in out
        assert out.splitlines()[-1].startswith(
            "SURFACE: the template's initial state is below the surface (§4 note 1(b)): "
            f"SC_Human at -0.001000 km on data row {rows.start + 1}, before any burn of ours")
        assert out.splitlines()[-1].endswith("; ask the organizer")
    else:
        assert f"below the surface on data row {rows.start + 1}, before the interception" in out
    # Untouched, the same report passes.
    (tmp_path / "plain").mkdir()
    assert run(monkeypatch, capsys, _altitude_set(tmp_path / "plain", case, range(0), 0.0),
               "--config", day(tmp_path, case.split("_")[0]), *frozen(case))[0] == 0


def test_a_script_that_maneuvers_the_target_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = script(tmp_path, "R1_hit", "Maneuver ImpulsiveBurn1(SC_Human);",
                  "Maneuver ImpulsiveBurn1(SC_Alien);")
    code, _, err = run(monkeypatch, capsys, upload(tmp_path, "R1_hit"), "--config",
                       day(tmp_path, "R1"), "--script", path)
    assert code == 2 and "maneuvers ['SC_Alien'], not the pursuer SC_Human" in err


def test_binding_warn_turns_a_burn_refusal_into_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The escape for a false refusal step 6's pre-check did not show; never over the end."""
    config = day(tmp_path, "R1")
    path = script(tmp_path, "R1_hit", "ImpulsiveBurn1.Element2 = 0.18946230065701242;",
                  "ImpulsiveBurn1.Element2 = -0.18946230065701242;")
    flags = ["--config", config, "--script", path]
    code, _, err = run(monkeypatch, capsys, upload(tmp_path, "R1_hit"), *flags)
    assert code == 2 and "apart as a vector" in err
    code, out, err = run(monkeypatch, capsys, fixture(tmp_path, "R1_hit"), *flags,
                         "--binding", "warn")
    assert code == 2 and "is named 'R1_hit.txt'" in err  # (c) still refuses
    report = tmp_path / "usb-warn"
    report.mkdir()
    (report / UPLOADED).write_bytes(report_bytes("R1_hit"))
    code, out, err = run(monkeypatch, capsys, report / UPLOADED, *flags, "--binding", "warn")
    assert code == 3 and "WARNING: " in out and "apart as a vector" in out, out + err
    assert "6b.7's by-eye comparison is now mandatory" in out and "hold its" not in out
    # After the score lines, the last line says the pass needs the eye.
    assert "-> score" in out and out.splitlines()[-1] == CHECK.UNBOUND_LINE
    # The end and the total are never warned past.
    code, _, err = run(monkeypatch, capsys, upload(tmp_path, "R1_other_step60"),
                       "--config", config, "--script", BIND / "R1_hit_step60.script",
                       "--binding", "warn")
    assert code == 2 and "ends at 4518.038474" in err
    code, _, err = run(monkeypatch, capsys, report / UPLOADED, *flags, "--binding", "warn",
                       "--total-dv", "0.2173")
    assert code == 2 and "--total-dv is 0.2173 m/s" in err


def test_the_binding_flags_need_a_hand_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for flag in (["--binding", "warn"], ["--accept-end-time-binding-only"]):
        code, _, err = run(monkeypatch, capsys, fixture(tmp_path, "R1_hit"), *flag)
        assert code == 2 and "held to --script by hand" in err, flag


def test_a_burn_compared_in_magnitude_only_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Local/Earth/MJ2000Eq is not built here, so only the jump's length is held to it."""
    path = script(tmp_path, "R1_hit", "ImpulsiveBurn1.Axes = VNB;", "ImpulsiveBurn1.Axes = MJ2000Eq;")
    code, out, err = run(monkeypatch, capsys, upload(tmp_path, "R1_hit"), "--config",
                         day(tmp_path, "R1"), "--script", path)
    assert code == 0, out + err
    assert "hold its 1 burn (1 in magnitude only, its frame not built here) and no other" in out


def test_a_clock_that_goes_back_after_the_alignment_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Read on SC_Alien's clock, R8's SC_Human.ElapsedSecs, the binding's, steps back once."""
    header, data = CHECK.read_report(str(fixture(tmp_path, "R8_hit_step60")))
    data = data.copy()
    column = header.index("SC_Human.ElapsedSecs")
    start = MANIFEST["R8_hit_step60"]["frozen_rows"]
    data[start + 4, column] = data[start + 3, column] - 0.5
    folder = tmp_path / "usb"
    folder.mkdir()
    lines = ["  ".join(header) + "\n",
             *("  ".join(repr(float(x)) for x in row) + "\n" for row in data)]
    (folder / UPLOADED).write_text("".join(lines), encoding="utf-8")
    path = script(tmp_path, "R8_hit_step60", "ReportFile1.FixedWidth = true;",
                  "ReportFile1.FixedWidth = false;")
    code, _, err = run(monkeypatch, capsys, folder / UPLOADED, "--script", path,
                       "--rules", "preliminary", "--time-column", "SC_Alien.ElapsedSecs",
                       *frozen("R8_hit_step60"))
    assert code == 2 and "SC_Human.ElapsedSecs goes back after the alignment" in err, err


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("SC_Human.ElapsedSecs = 0.001}", "SC_Human.ElapsedDays = 0.001}", "Mixed elapsed-time"),
        ("Propagate DefaultProp(SC_Alien, SC_Human) {SC_Alien.ElapsedSecs = 4301",
         "Maneuver ImpulsiveBurn1(SC_Human);\nPropagate DefaultProp(SC_Alien, SC_Human) "
         "{SC_Alien.ElapsedSecs = 4301", "maneuvered more than once"),
    ],
)
def test_the_schedule_fails_closed_as_the_parser_does(
    old: str, new: str, message: str, tmp_path: Path
) -> None:
    with pytest.raises(ParseError, match=message):
        mission_schedule(script(tmp_path, "R1_hit", old, new))


def test_the_schedule_knows_the_longest_step() -> None:
    """Steps longer than it are rows the report does not hold, and are not flown back."""
    assert mission_schedule(BIND / "R1_hit.script").max_step_s == 1.0
    assert mission_schedule(BIND / "R8_hit_step2700.script").max_step_s == 2700.0


def test_two_pre_burn_rows_at_the_alignments_end_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R8 at MaxStep 60 with its second alignment-end row pre-burn too: the jump then lies
    after the alignment's end, and that step is the burn's, not a coast's."""
    header, data = CHECK.read_report(str(fixture(tmp_path, "R8_hit_step60")))
    data = data.copy()
    # t = 0 is the first of the two rows; the second, post-burn, made pre-burn too.
    second = MANIFEST["R8_hit_step60"]["frozen_rows"] + 1
    velocity = [header.index(f"SC_Human.EarthMJ2000Eq.V{axis}") for axis in "XYZ"]
    data[second, velocity] = data[second - 1, velocity]
    folder = tmp_path / "usb"
    folder.mkdir()
    lines = ["  ".join(header) + "\n",
             *("  ".join(repr(float(x)) for x in row) + "\n" for row in data)]
    (folder / UPLOADED).write_text("".join(lines), encoding="utf-8")
    path = script(tmp_path, "R8_hit_step60", "ReportFile1.FixedWidth = true;",
                  "ReportFile1.FixedWidth = false;")
    code, out, err = run(monkeypatch, capsys, folder / UPLOADED, "--config",
                         day(tmp_path, "R8"), "--script", path, *frozen("R8_hit_step60"))
    assert code == 0 and "hold its 2 burns and no other" in out, out + err


def _kept(tmp_path: Path, case: str, flight: str, name: str) -> tuple[Path, Path]:
    """A certify stopped on plots, set aside as `final.refused-2`: its staging script, its
    verification and its own report, from a fixture; and the day config."""
    import hashlib

    from orbitwin.config import load_config
    from orbitwin.score import score_stamp

    config = day(tmp_path, case)
    kept = tmp_path / "run" / "final.refused-2"
    (kept / "staging").mkdir(parents=True)
    staging = kept / "staging" / "final.script"
    staging.write_bytes((BIND / f"{case}.script").read_bytes())
    output = kept / "diagnostic" / flight / "gmat-output"
    output.mkdir(parents=True)
    (output / name).write_bytes(report_bytes(case))
    (kept / "diagnostic" / "verification.json").write_text(json.dumps({
        "candidate_script_sha256": hashlib.sha256(staging.read_bytes()).hexdigest(),
        "score_block": score_stamp(load_config(config).score),
        "frozen_rows": MANIFEST[case]["frozen_rows"],
        # Written before certify set the run aside, as verify writes it.
        "diagnostic_report": str(tmp_path / "run" / "final" / "diagnostic" / flight
                                 / "gmat-output" / "orbitwin_diagnostic.txt"),
    }), encoding="utf-8")
    # An earlier attempt beside it, which a literal path in the RUNBOOK would have picked.
    (tmp_path / "run" / "final.refused").mkdir()
    return kept, config


@pytest.mark.parametrize(("case", "flight", "name"), [
    ("R1_hit", "diagnostic-run-at-stop", UPLOADED),
    ("R1_zero_step60", "diagnostic-run", "ODC_2026_TeamXX.txt"),
    # Review of #185, P2-C: without --frozen-rows R8 at MaxStep 2700 exited 1.
    ("R8_hit_step2700", "diagnostic-run-at-stop", UPLOADED),
    ("R8_hit_step60", "diagnostic-run-at-stop", UPLOADED),
])
def test_the_plots_stop_names_a_pre_check_that_passes_on_our_report(
    case: str, flight: str, name: str, tmp_path: Path
) -> None:
    import subprocess

    from orbitwin.cli import gui_precheck

    kept, config = _kept(tmp_path, case, flight, name)
    argv = gui_precheck(kept, str(config))
    assert argv is not None
    frozen_rows = MANIFEST[case]["frozen_rows"]
    assert (argv[argv.index("--frozen-rows") + 1] == str(frozen_rows)) if frozen_rows else (
        "--frozen-rows" not in argv)
    assert argv[2] == str(kept / "diagnostic" / flight / "gmat-output" / name)
    assert argv[argv.index("--script") + 1] == str(kept / "staging" / "final.script")
    assert argv[argv.index("--verification") + 1] == str(kept / "diagnostic" / "verification.json")
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "and no other" in done.stdout


def test_certify_prints_the_pre_check_when_it_stops_on_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from orbitwin import cli
    from orbitwin.errors import GmatRunError
    from orbitwin.io import candidate_from_dict, write_candidates

    kept, config = _kept(tmp_path, "R1_hit", "diagnostic-run-at-stop", UPLOADED)
    candidate = candidate_from_dict(
        json.loads((BIND / "R1_hit.candidate.json").read_text(encoding="utf-8")))
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate])

    def stopped(*_: object, **__: object) -> None:
        error = GmatRunError(f"final.script: GmatConsole cannot create OpenFramesInterface. "
                             f"{cli.GUI_STOP} and put the report it writes through ...")
        error.__dict__["kept_dir"] = kept
        raise error

    monkeypatch.setattr(cli, "certify_candidate", stopped)
    code = cli.main(["certify", "official.script", "--config", str(config), "--candidates",
                     str(candidates), "--candidate", candidate.candidate_id, "--gmat-console",
                     "gmat", "--startup-file", "startup.txt", "--output", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 2
    line = next(line for line in err.splitlines() if "before the GUI" in line)
    assert str(kept / "diagnostic" / "verification.json") in line
    assert "final.refused/" not in line


@pytest.mark.parametrize("error", ["orbitwin", "other"])
def test_a_refused_certify_says_where_it_kept_the_run(
    error: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second refusal of one --output is kept in final.refused-2, and says so."""
    from orbitwin import verify
    from orbitwin.errors import GmatRunError

    def refuse(*_: object, root: Path, **__: object) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "staging").mkdir()
        raise GmatRunError("stopped") if error == "orbitwin" else RuntimeError("stopped")

    monkeypatch.setattr(verify, "_certify", lambda *a, **k: refuse(root=a[5], **k))
    output = tmp_path / "final"
    (tmp_path / "final.refused").mkdir()
    with pytest.raises((GmatRunError, RuntimeError)) as raised:
        verify.certify_candidate("x.script", None, None, "gmat", None, output)  # type: ignore[arg-type]
    assert raised.value.__dict__["kept_dir"] == tmp_path / "final.refused-2"


Drop = Callable[[int, float], bool]


def _extra(tmp_path: Path, drop: Drop, case: str = "R1x_extra02_step60") -> Path:
    """The review of #185's B attacks: R1's hit with an unscripted 0.2 m/s burn at t = 2000 s
    (5 m/s in R1x_extra5), flown by GMAT at MaxStep 60, less the rows `drop` takes (data row,
    time)."""
    lines = report_text(case).splitlines(keepends=True)
    header = lines[0].split()
    clock = header.index("SC_Human.ElapsedSecs")
    kept = [line for i, line in enumerate(lines[1:]) if not drop(i, float(line.split()[clock]))]
    folder = tmp_path / "usb-extra"
    folder.mkdir()
    (folder / UPLOADED).write_text("".join([lines[0], *kept]), encoding="utf-8")
    return folder / UPLOADED


@pytest.mark.parametrize(("drop", "script_case", "gap"), [
    # B2: the row before the burn the script lacks.
    (lambda i, t: abs(t - 2000.001) < 1e-3, "R1_hit_step60", "rows missing between t = "),
    # B3: every other row.
    (lambda i, t: i % 2 == 1 and i > 3, "R1_hit_step60", "rows missing between"),
    # B4: the MaxStep 60 flight against R1's MaxStep 1 script.
    (lambda i, t: False, "R1_hit", "60 s apart where"),
])
def test_rows_missing_are_refused(
    drop: Drop, script_case: str, gap: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Skipped over, they hid the burn the script lacks: exit 0, "hold its 1 burn"."""
    report = _extra(tmp_path, drop)
    code, out, err = run(monkeypatch, capsys, report, "--config", day(tmp_path, "R1"),
                         "--script", BIND / f"{script_case}.script")
    assert code == 2 and "VERDICT" not in out, out + err
    assert "rows missing between" in err and gap in err, err
    assert "hold its" not in out


def test_a_burn_the_script_lacks_shows_down_to_the_scans_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Whole, against the hit's MaxStep 60 script: 5 m/s at t = 2000 s is refused; 0.2 m/s
    passes, under the 0.61 m/s a 60 s step allows (docs/RUNBOOK.md step 7, its limit)."""
    flags = ["--config", day(tmp_path, "R1"), "--script", BIND / "R1_hit_step60.script"]
    (tmp_path / "five").mkdir()
    code, _, err = run(monkeypatch, capsys,
                       _extra(tmp_path / "five", lambda i, t: False, "R1x_extra5_step60"), *flags)
    assert code == 2 and "t = 2000.001000 to 2060.001000" in err and "has no burn" in err, err
    code, out, _ = run(monkeypatch, capsys, _extra(tmp_path, lambda i, t: False), *flags)
    assert code == 0 and "hold its 1 burn and no other" in out


def test_certify_ends_on_the_pre_checks_verdict_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PRE-CHECK PASS on our own report, FAILED on another's; certify exits 0 both times."""
    import hashlib

    from orbitwin import cli
    from orbitwin.config import load_config
    from orbitwin.io import candidate_from_dict, write_candidates
    from orbitwin.score import score_stamp

    config = day(tmp_path, "R8")
    candidate = candidate_from_dict(
        json.loads((BIND / "R1_hit.candidate.json").read_text(encoding="utf-8")))
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate])
    console = tmp_path / "GmatConsole"
    console.write_bytes(b"\x7fELF not the test double")
    output = tmp_path / "final"
    flown = {"case": "R8_hit_step60"}

    def certified(*_: object, **__: object) -> dict[str, object]:
        submission = output / "submission"
        submission.mkdir(parents=True, exist_ok=True)
        script = submission / "public_gmat_candidate.script"
        script.write_bytes((BIND / "R8_hit_step60.script").read_bytes())
        (submission / UPLOADED).write_bytes(report_bytes(flown["case"]))
        (output / "diagnostic").mkdir(exist_ok=True)
        (output / "diagnostic" / "verification.json").write_text(json.dumps({
            "candidate_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "score_block": score_stamp(load_config(config).score)}), encoding="utf-8")
        result = {"submission_script_name": script.name, "official_report_files": [UPLOADED],
                  "stage": "public_gmat_verified", "conferred_stage": "public_gmat_verified",
                  "frozen_rows": MANIFEST["R8_hit_step60"]["frozen_rows"]}
        (submission / "certification.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(cli, "certify_candidate", certified)
    # Step 7 on this stand-in certificate is not what is tested here: it passes.
    monkeypatch.setattr(cli, "step7_command", lambda *_: [sys.executable, "-c", "",
                                                          "--certificate"])
    argv = ["certify", "official.script", "--config", str(config), "--candidates",
            str(candidates), "--candidate", candidate.candidate_id, "--gmat-console",
            str(console), "--startup-file", "startup.txt", "--output", str(output)]
    for case, verdict in (("R8_hit_step60", "PRE-CHECK PASS"),
                          ("R6_hit_step60", "WARNING: PRE-CHECK FAILED: ")):
        flown["case"] = case
        code = cli.main(argv)
        last = capsys.readouterr().err.splitlines()[-1]
        assert code == 0 and last.startswith(verdict), last
        recorded = json.loads((output / "submission" / "certification.json").read_text())
        assert recorded["binding_pre_check"]["result"] == ("pass" if "PASS" in verdict
                                                            else "failed")
    # The test double's reports are not GMAT's: skipped, and said so.
    argv[argv.index(str(console))] = str(ROOT / "scripts" / "fake_gmat_console.py")
    assert cli.main(argv) == 0
    assert capsys.readouterr().err.splitlines()[-1].endswith(
        "PRE-CHECK SKIPPED: the console is orbitwin's test double, whose reports are not GMAT's")


def test_the_pre_check_says_when_it_could_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orbitwin import cli

    console = tmp_path / "GmatConsole"
    console.write_bytes(b"not the test double")
    line, record = cli.binding_precheck([["x"]], str(console), None)
    assert line.startswith("WARNING: PRE-CHECK NOT RUN: no --config was given")
    # Without the checkout's script (an installed package) the check runs as a module.
    monkeypatch.setattr(cli, "_CHECK_REPORT_ROWS", tmp_path / "gone.py")
    assert cli._checker() == [sys.executable, "-m", "orbitwin.tools.check_report_rows"]
    line, record = cli.binding_precheck([], str(console), "day.yaml")
    assert "no report to check" in line and record["result"] == "not run"
    # A module command that cannot start names the module.
    missing = [str(tmp_path / "no-python"), "-m", "orbitwin.tools.check_report_rows"]
    line, record = cli.binding_precheck([missing], str(console), "day.yaml")
    assert "orbitwin.tools.check_report_rows did not run" in line and record["result"] == "not run"


def test_an_alignment_ending_on_two_rows_is_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R8 at MaxStep 2700: its alignment ends on two rows, the second 3e-7 s earlier, and
    certify counts the first as t = 0. Read from there it was "more than one run"."""
    certified_rows = MANIFEST["R8_hit_step2700"]["frozen_rows"]
    code, out, err = run(monkeypatch, capsys, upload(tmp_path, "R8_hit_step2700"), "--config",
                         day(tmp_path, "R8"), "--script", BIND / "R8_hit_step2700.script",
                         "--frozen-rows", str(certified_rows))
    assert code == 0 and "more than one run" not in err, out + err


@pytest.mark.parametrize(("case", "certified"), [("R8_miss_step60", 33), ("R6_miss_step60", 30)])
def test_a_miss_burning_right_after_the_alignment_passes_on_certifys_frozen_rows(
    case: str, certified: int, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """certification.json's frozen_rows for each (R2026a, MaxStep 60). R8 ends its alignment
    on two rows; counting the second as t = 0 left the miss "unchecked", exit 1, and step 6's
    pre-check FAILED on a valid candidate."""
    assert MANIFEST[case]["frozen_rows"] == certified
    code, out, err = run(monkeypatch, capsys, upload(tmp_path, case), "--config",
                         day(tmp_path, case), "--script", BIND / f"{case}.script",
                         "--frozen-rows", str(certified))
    assert code == 0 and "a scored miss" in out and "holds still" not in err, out + err
    assert "hold its 1 burn and no other" in out


def test_t_0_is_the_first_row_at_the_instant_the_alignment_ends(tmp_path: Path) -> None:
    """R8's miss at MaxStep 60 ends its alignment on rows 34 and 35 (data rows), one instant:
    t = 0 is the first, as certify counts it. A clock too coarse to tell the alignment's own
    steps apart moves nothing."""
    header, data = CHECK.read_report(str(fixture(tmp_path, "R8_miss_step60")))
    positions = {craft: [header.index(f"{craft}.EarthMJ2000Eq.{axis}") for axis in "XYZ"]
                 for craft in ("SC_Human", "SC_Alien")}
    assert CHECK.waiting_craft(data, positions, header) == ("SC_Alien", 33)
    clock = header.index("SC_Human.ElapsedSecs")
    assert data[33, clock] != data[32, clock] and abs(data[34, clock] - data[33, clock]) < 1e-6
    # Printed to 2 digits, every alignment row is "1800": nothing to go back over.
    assert CHECK.waiting_craft(data, positions, header, precision=2) == ("SC_Alien", 34)
    # With an A1ModJulian beside it, the finer of the early craft's clocks decides: at
    # Precision 8 the day count prints to 86 s and cannot tell the 60 s steps apart.
    epochs = 31353.0 + data[:, clock] / 86400.0
    widened = np.column_stack([data, epochs])
    named = [*header, "SC_Human.A1ModJulian"]
    assert CHECK.waiting_craft(widened, positions, named, precision=8) == ("SC_Alien", 33)


@pytest.mark.parametrize("second_pre_burn", [False, True])
def test_without_the_early_crafts_clock_the_burn_is_found_on_the_alignments_last_rows(
    second_pre_burn: bool, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """R8 at MaxStep 60 with only SC_Alien's ElapsedSecs, which reads 0 through the
    alignment: nothing tells its two end rows apart, so t = 0 stays the second and the
    burn's row is looked for from the row before it; made pre-burn too, the second's step
    is the burn's."""
    header, data = CHECK.read_report(str(fixture(tmp_path, "R8_hit_step60")))
    data = data.copy()
    first = MANIFEST["R8_hit_step60"]["frozen_rows"]
    if second_pre_burn:
        velocity = [header.index(f"SC_Human.EarthMJ2000Eq.V{axis}") for axis in "XYZ"]
        data[first + 1, velocity] = data[first, velocity]
    keep = [i for i, name in enumerate(header) if name != "SC_Human.ElapsedSecs"]
    folder = tmp_path / "usb"
    folder.mkdir()
    (folder / UPLOADED).write_text("  ".join(header[i] for i in keep) + "\n" + "".join(
        "  ".join(repr(float(row[i])) for i in keep) + "\n" for row in data), encoding="utf-8")
    text = (BIND / "R8_hit_step60.script").read_text(encoding="utf-8")
    assert "SC_Human.ElapsedSecs, " in text
    path = tmp_path / "one_clock.script"
    path.write_text(text.replace("SC_Human.ElapsedSecs, ", "").replace(
        "ReportFile1.FixedWidth = true;", "ReportFile1.FixedWidth = false;"), encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, folder / UPLOADED, "--script", path, "--rules",
                         "preliminary", "--time-column", "SC_Alien.ElapsedSecs",
                         "--frozen-rows", str(first + 1))
    assert code == 0 and "hold its 2 burns and no other" in out, out + err


def test_a_step_printed_past_max_step_by_its_last_digit_is_not_rows_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R1's second candidate at MaxStep 60, printed to 7 digits as GMAT prints them: one
    step reads 60.0005 s, past MaxStep by less than two of its last digits."""
    lines = report_text("R1_other_step60").splitlines()
    header = lines[0].split()
    rows = [[float(value) for value in line.split()] for line in lines[1:]]
    clock = header.index("SC_Human.ElapsedSecs")
    printed = [[float(f"{value:.7g}") for value in row] for row in rows]
    steps = np.diff([row[clock] for row in printed])
    assert 60.0001 < steps.max() <= 60.002
    folder = tmp_path / "usb"
    folder.mkdir()
    (folder / UPLOADED).write_text(" ".join(header) + "\n" + "".join(
        " ".join(f"{value:.7g}" for value in row) + "\n" for row in rows), encoding="utf-8")
    path = script(tmp_path, "R1_other_step60", "ReportFile1.Precision = 16;",
                  "ReportFile1.Precision = 7;").read_text(encoding="utf-8").replace(
        "ReportFile1.FixedWidth = true;", "ReportFile1.FixedWidth = false;")
    seven = tmp_path / "seven.script"
    seven.write_text(path, encoding="utf-8")
    code, out, err = run(monkeypatch, capsys, folder / UPLOADED, "--config", day(tmp_path, "R1"),
                         "--script", seven)
    assert code == 0 and "rows missing" not in err and "and no other" in out, out + err


def test_the_plots_stop_checks_the_flight_verification_scored(tmp_path: Path) -> None:
    """Review of #185: a planned hit GMAT flies as a miss is scored on its flight to T_max,
    diagnostic-run-to-horizon. The pre-check took diagnostic-run, the measuring flight that
    stops short, and FAILED a candidate that passes."""
    import subprocess

    from orbitwin.cli import gui_precheck

    kept, config = _kept(tmp_path, "R1_zero_step60", "diagnostic-run-to-horizon",
                         "ODC_2026_TeamXX.txt")
    decoy = kept / "diagnostic" / "diagnostic-run" / "gmat-output"
    decoy.mkdir(parents=True)
    (decoy / "ODC_2026_TeamXX.txt").write_bytes(report_bytes("R1_hit_step60"))
    argv = gui_precheck(kept, str(config))
    assert argv[2] == str(kept / "diagnostic" / "diagnostic-run-to-horizon" / "gmat-output"
                          / "ODC_2026_TeamXX.txt")
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_plots_stop_says_why_it_ran_no_pre_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from orbitwin import cli
    from orbitwin.errors import GmatRunError
    from orbitwin.io import candidate_from_dict, write_candidates

    kept, config = _kept(tmp_path, "R1_hit", "diagnostic-run-at-stop", UPLOADED)
    candidate = candidate_from_dict(
        json.loads((BIND / "R1_hit.candidate.json").read_text(encoding="utf-8")))
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate])
    console = tmp_path / "GmatConsole"
    console.write_bytes(b"not the test double")

    def stopped(*_: object, **__: object) -> None:
        error = GmatRunError(f"GmatConsole cannot create OpenFramesInterface. {cli.GUI_STOP}")
        error.__dict__["kept_dir"] = kept
        raise error

    monkeypatch.setattr(cli, "certify_candidate", stopped)
    argv = ["certify", "official.script", "--candidates", str(candidates), "--candidate",
            candidate.candidate_id, "--gmat-console", str(console), "--startup-file", "s.txt",
            "--output", str(tmp_path / "out")]
    assert cli.main([*argv, "--rules", "preliminary"]) == 2
    last = capsys.readouterr().err.splitlines()[-1]
    assert last.startswith("WARNING: PRE-CHECK NOT RUN: no --config was given"), last
    (kept / "diagnostic" / "diagnostic-run-at-stop" / "gmat-output" / UPLOADED).unlink()
    assert cli.main([*argv, "--config", str(config)]) == 2
    last = capsys.readouterr().err.splitlines()[-1]
    assert last.startswith("WARNING: PRE-CHECK NOT RUN: the flight verification.json names "
                           "wrote no"), last
    recorded = json.loads((kept / "diagnostic" / "verification.json").read_text())
    assert recorded["binding_pre_check"]["result"] == "not run"


FLOWN = "diagnostic-run-at-stop/gmat-output/orbitwin_diagnostic.txt"


@pytest.mark.parametrize("recorded", [
    "/srv/someone/else/final/diagnostic/" + FLOWN,
    "{tmp}/other/final/diagnostic/" + FLOWN,
    "{tmp}/run/final.refused/diagnostic/" + FLOWN,
    "{tmp}/run/final2/diagnostic/" + FLOWN,
    "diagnostic/" + FLOWN,
    "{tmp}/run/final/staging/" + FLOWN,
    "{tmp}/run/final/diagnostic/../gmat-output/orbitwin_diagnostic.txt",
    "{tmp}/run/final/diagnostic/staging/x/orbitwin_diagnostic.txt",
    "{tmp}/run/final/diagnostic/diagnostic-run-at-stop/staging/orbitwin_diagnostic.txt",
    "/orbitwin_diagnostic.txt",
])
def test_a_flight_of_another_run_is_not_remapped(recorded: str, tmp_path: Path) -> None:
    """Final delta review of #185: diagnostic_report was remapped by its flight folder's name
    alone, so a path from anywhere checked a report under kept. It must be the kept run's."""
    from orbitwin.cli import PrecheckUnavailable, gui_precheck

    kept, config = _kept(tmp_path, "R1_hit", "diagnostic-run-at-stop", UPLOADED)
    verification = kept / "diagnostic" / "verification.json"
    record = json.loads(verification.read_text(encoding="utf-8"))
    record["diagnostic_report"] = recorded.format(tmp=tmp_path)
    verification.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(PrecheckUnavailable, match="not a flight of "
                       + re.escape(str(tmp_path / "run" / "final")) + ", the run kept in"):
        gui_precheck(kept, str(config))


def test_a_relative_flight_is_not_the_kept_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """verify records an absolute path; a relative one is no run's, from any directory."""
    from orbitwin.cli import PrecheckUnavailable, gui_precheck

    kept, config = _kept(tmp_path, "R1_hit", "diagnostic-run-at-stop", UPLOADED)
    verification = kept / "diagnostic" / "verification.json"
    record = json.loads(verification.read_text(encoding="utf-8"))
    record["diagnostic_report"] = "final/diagnostic/" + FLOWN
    verification.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.chdir(tmp_path / "run")
    with pytest.raises(PrecheckUnavailable, match="not a flight of"):
        gui_precheck(kept, str(config))


@pytest.mark.parametrize("name", ["final.refused", "final.refused-2", "final.refused-12"])
def test_a_flight_recorded_through_a_link_is_the_kept_runs(name: str, tmp_path: Path) -> None:
    """verify records the resolved path; a run certified through a symlink still matches."""
    from orbitwin.cli import gui_precheck

    kept, config = _kept(tmp_path, "R1_hit", "diagnostic-run-at-stop", UPLOADED)
    if kept.name != name:
        kept = kept.rename(kept.with_name(name))
    (tmp_path / "link").symlink_to(tmp_path / "run")
    verification = kept / "diagnostic" / "verification.json"
    record = json.loads(verification.read_text(encoding="utf-8"))
    record["diagnostic_report"] = str(tmp_path / "link" / "final" / "diagnostic" / FLOWN)
    verification.write_text(json.dumps(record), encoding="utf-8")
    argv = gui_precheck(tmp_path / "link" / name, str(config))
    assert argv[2] == str(tmp_path / "link" / name / "diagnostic" / "diagnostic-run-at-stop"
                          / "gmat-output" / UPLOADED)


def test_the_plots_stop_says_a_flight_of_another_run_is_not_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from orbitwin import cli
    from orbitwin.errors import GmatRunError
    from orbitwin.io import candidate_from_dict, write_candidates

    kept, config = _kept(tmp_path, "R1_hit", "diagnostic-run-at-stop", UPLOADED)
    verification = kept / "diagnostic" / "verification.json"
    record = json.loads(verification.read_text(encoding="utf-8"))
    record["diagnostic_report"] = "/srv/someone/else/final/diagnostic/" + FLOWN
    verification.write_text(json.dumps(record), encoding="utf-8")
    candidate = candidate_from_dict(
        json.loads((BIND / "R1_hit.candidate.json").read_text(encoding="utf-8")))
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate])
    console = tmp_path / "GmatConsole"
    console.write_bytes(b"not the test double")

    def stopped(*_: object, **__: object) -> None:
        error = GmatRunError(f"GmatConsole cannot create OpenFramesInterface. {cli.GUI_STOP}")
        error.__dict__["kept_dir"] = kept
        raise error

    monkeypatch.setattr(cli, "certify_candidate", stopped)
    assert cli.main(["certify", "official.script", "--config", str(config), "--candidates",
                     str(candidates), "--candidate", candidate.candidate_id, "--gmat-console",
                     str(console), "--startup-file", "s.txt", "--output",
                     str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert "before the GUI" not in err
    last = err.splitlines()[-1]
    assert last.startswith(f"WARNING: PRE-CHECK NOT RUN: {verification} names "
                           "'/srv/someone/else/final/"), last
    assert "not a flight of" in last


def test_a_hand_flag_the_verification_contradicts_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R8's 6 frozen rows come from verification.json; another count, or another sphere, by
    hand exits 2. The same count passes."""
    kept, config = _kept(tmp_path, "R8_hit_step60", "diagnostic-run-at-stop", UPLOADED)
    report = kept / "diagnostic" / "diagnostic-run-at-stop" / "gmat-output" / UPLOADED
    rows = MANIFEST["R8_hit_step60"]["frozen_rows"]
    argv = [report, "--config", config, "--script", kept / "staging" / "final.script",
            "--verification", kept / "diagnostic" / "verification.json"]
    code, out, err = run(monkeypatch, capsys, *argv)
    assert code == 0 and f"skipped: {rows}" in out, out + err
    assert run(monkeypatch, capsys, *argv, "--frozen-rows", str(rows))[0] == 0
    code, _, err = run(monkeypatch, capsys, *argv, "--frozen-rows", str(rows + 1))
    assert code == 2 and (f"--frozen-rows is {rows + 1} and the verification's frozen_rows is "
                          f"{rows}") in err, err
    code, _, err = run(monkeypatch, capsys, *argv, "--radius-km", "50")
    assert code == 2 and "which the verification was judged on, is 5 km" in err, err


VELOCITY = [f"SC_Human.EarthMJ2000Eq.{axis}" for axis in ("VX", "VY", "VZ")]


def _kept_without_velocity(tmp_path: Path) -> tuple[Path, Path]:
    """R1's step-60 hit set aside on plots, its official report without SC_Human's velocity:
    the organizer's own example reports one coordinate of the pursuer."""
    import hashlib

    kept, config = _kept(tmp_path, "R1_hit_step60", "diagnostic-run-at-stop", UPLOADED)
    staging = kept / "staging" / "final.script"
    text = staging.read_text(encoding="utf-8")
    for name in VELOCITY:
        assert f", {name}" in text
        text = text.replace(f", {name}", "")
    staging.write_text(text.replace("ReportFile1.FixedWidth = true;",
                                    "ReportFile1.FixedWidth = false;"), encoding="utf-8")
    rows = [line.split() for line in report_text("R1_hit_step60").splitlines()]
    keep = [k for k, name in enumerate(rows[0]) if name not in VELOCITY]
    report = kept / "diagnostic" / "diagnostic-run-at-stop" / "gmat-output" / UPLOADED
    report.write_text("".join(" ".join(row[k] for k in keep) + "\n" for row in rows),
                      encoding="utf-8")
    verification = kept / "diagnostic" / "verification.json"
    record = json.loads(verification.read_text(encoding="utf-8"))
    record["candidate_script_sha256"] = hashlib.sha256(staging.read_bytes()).hexdigest()
    verification.write_text(json.dumps(record), encoding="utf-8")
    return kept, config


def test_a_report_that_cannot_show_burns_is_pre_checked_on_its_end(tmp_path: Path) -> None:
    """Review of the docs, P1: without the pursuer's velocity every candidate's pre-check
    FAILED, and RUNBOOK then turned down every fallback. It runs end-time only and says so."""
    from orbitwin import cli

    kept, config = _kept_without_velocity(tmp_path)
    argv = cli.gui_precheck(kept, str(config))
    assert argv[-1] == "--accept-end-time-binding-only"
    console = tmp_path / "GmatConsole"
    console.write_bytes(b"not the test double")
    line, record = cli.binding_precheck([argv], str(console), str(config))
    assert line == cli.PRECHECK_PASS_END_TIME == (
        "PRE-CHECK PASS (end time only): the 6b.7 by-eye comparison is mandatory")
    assert record["result"] == "pass-end-time-only"
    # A report that shows its burns is held to them, as before.
    (tmp_path / "whole").mkdir()
    whole, _ = _kept(tmp_path / "whole", "R1_hit_step60", "diagnostic-run-at-stop", UPLOADED)
    assert "--accept-end-time-binding-only" not in cli.gui_precheck(whole, str(config))


def test_exit_3_passes_only_a_pre_check_that_asked_for_the_end_alone(tmp_path: Path) -> None:
    import sys as system

    from orbitwin import cli

    console = tmp_path / "GmatConsole"
    console.write_bytes(b"not the test double")
    three = [system.executable, "-c", "import sys; print('VERDICT: x'); sys.exit(3)"]
    line, record = cli.binding_precheck([[*three, "--accept-end-time-binding-only"]],
                                        str(console), "day.yaml")
    assert line == cli.PRECHECK_PASS_END_TIME and record["result"] == "pass-end-time-only"
    line, record = cli.binding_precheck([three], str(console), "day.yaml")
    assert line.startswith("WARNING: PRE-CHECK FAILED: VERDICT: x") and record["result"] == "failed"
    two = [system.executable, "-c", "import sys; sys.exit(2)", "--accept-end-time-binding-only"]
    line, record = cli.binding_precheck([two], str(console), "day.yaml")
    assert line.startswith("WARNING: PRE-CHECK FAILED") and record["result"] == "failed"


def test_exit_7_by_hand_passes_only_beside_step_7_s_pass(tmp_path: Path) -> None:
    """R3 T16: a report with no target position is judged by step 7 alone."""
    import sys as system

    from orbitwin import cli

    console = tmp_path / "GmatConsole"
    console.write_bytes(b"not the test double")

    def exits(code: int, *flags: str) -> list[str]:
        return [system.executable, "-c", f"import sys; print('VERDICT: c{code}'); "
                f"sys.exit({code})", *flags]

    seven, step7 = exits(7), exits(0, "--certificate")
    line, record = cli.binding_precheck([seven, step7], str(console), "day.yaml")
    assert line == cli.PRECHECK_PASS_NO_TARGET and record["result"] == "pass-target-not-in-report"
    assert line.startswith("PRE-CHECK PASS (target not in report)")
    line, record = cli.binding_precheck(
        [exits(7, "--accept-end-time-binding-only"), step7], str(console), "day.yaml")
    assert line == cli.PRECHECK_PASS_NO_TARGET_END_TIME
    assert record["result"] == "pass-target-not-in-report-end-time-only"
    assert line.endswith("the 6b.7 by-eye comparison is mandatory")
    # Alone, as on the GUI path, or beside a step 7 that failed, it is no pass.
    for commands in ([seven], [seven, exits(1, "--certificate")]):
        line, record = cli.binding_precheck(commands, str(console), "day.yaml")
        assert line.startswith("WARNING: PRE-CHECK FAILED: VERDICT: c"), line
        assert record["result"] == "failed"
    # Step 7 itself exiting 7 is a failure too.
    line, _ = cli.binding_precheck([exits(0), exits(7, "--certificate")], str(console), "d")
    assert line.startswith("WARNING: PRE-CHECK FAILED: VERDICT: c7")
    # A step 7 that leaves the surface to a person makes it that outcome.
    line, record = cli.binding_precheck([seven, exits(4, "--certificate")], str(console), "d")
    assert line.startswith("WARNING: PRE-CHECK SURFACE NOT JUDGED") and (
        record["result"] == "surface-not-judged")


def test_the_step_7_exit_codes_agree() -> None:
    import importlib.util

    from orbitwin import cli

    spec = importlib.util.spec_from_file_location("crr", ROOT / "scripts" / "check_report_rows.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert cli._TARGET_NOT_IN_REPORT == module.TARGET_NOT_IN_REPORT_EXIT
    assert set(cli._SURFACE_RESULTS) == {module.SURFACE_UNJUDGED_EXIT,
                                         module.SURFACE_TEMPLATE_EXIT,
                                         module.SURFACE_UNDECIDED_EXIT}


@pytest.mark.parametrize(("code", "result", "what"), [
    (4, "surface-not-judged", "NOT JUDGED"), (5, "surface-template-below", "TEMPLATE BELOW"),
    (6, "surface-undecided", "UNDECIDED")])
def test_an_undecided_surface_is_named_by_the_pre_check(
    code: int, result: str, what: str, tmp_path: Path
) -> None:
    import sys as system

    from orbitwin import cli

    console = tmp_path / "GmatConsole"
    console.write_bytes(b"not the test double")
    command = [system.executable, "-c",
               f"import sys; print('VERDICT: x'); print('SURFACE line {code}'); sys.exit({code})"]
    line, record = cli.binding_precheck([command], str(console), "day.yaml")
    assert line == f"WARNING: PRE-CHECK SURFACE {what}: SURFACE line {code}"
    assert record["result"] == result
    # A refusal beside it is still a failure.
    two = [system.executable, "-c", "import sys; sys.exit(2)"]
    line, record = cli.binding_precheck([two, command], str(console), "day.yaml")
    assert line.startswith("WARNING: PRE-CHECK FAILED") and record["result"] == "failed"


def test_the_plots_stop_ends_on_an_end_time_only_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from orbitwin import cli
    from orbitwin.errors import GmatRunError
    from orbitwin.io import candidate_from_dict, write_candidates

    kept, config = _kept_without_velocity(tmp_path)
    candidate = candidate_from_dict(
        json.loads((BIND / "R1_hit.candidate.json").read_text(encoding="utf-8")))
    candidates = tmp_path / "candidates.json"
    write_candidates(candidates, [candidate])
    console = tmp_path / "GmatConsole"
    console.write_bytes(b"not the test double")

    def stopped(*_: object, **__: object) -> None:
        error = GmatRunError(f"GmatConsole cannot create OpenFramesInterface. {cli.GUI_STOP}")
        error.__dict__["kept_dir"] = kept
        raise error

    monkeypatch.setattr(cli, "certify_candidate", stopped)
    assert cli.main(["certify", "official.script", "--config", str(config), "--candidates",
                     str(candidates), "--candidate", candidate.candidate_id, "--gmat-console",
                     str(console), "--startup-file", "s.txt", "--output",
                     str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert err.splitlines()[-1] == cli.PRECHECK_PASS_END_TIME
    [line] = [line for line in err.splitlines() if "before the GUI" in line]
    assert "on ours, must exit 3 (end time only; the 6b.7 by-eye check is mandatory): " in line
    recorded = json.loads((kept / "diagnostic" / "verification.json").read_text())
    assert recorded["binding_pre_check"]["result"] == "pass-end-time-only"


def test_a_pre_check_that_prints_no_utf_8_still_says_why(tmp_path: Path) -> None:
    import sys as system

    from orbitwin import cli

    console = tmp_path / "GmatConsole"
    console.write_bytes(b"not the test double")
    command = [system.executable, "-c",
               "import sys; sys.stderr.buffer.write(b'check: error: \\xff bad\\n'); sys.exit(2)"]
    line, record = cli.binding_precheck([command], str(console), "day.yaml")
    assert line.startswith("WARNING: PRE-CHECK FAILED: check: error: \ufffd bad"), line
    assert record["result"] == "failed"


def test_the_pre_check_line_is_last_in_one_stream(tmp_path: Path) -> None:
    """certify ... 2>&1 | tee: the JSON on stdout, buffered on a pipe, came out after it."""
    import subprocess
    import sys as system

    driver = tmp_path / "driver.py"
    driver.write_text(f"""
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
from orbitwin import cli

class Named:
    candidate_id = "C-0"

cli.read_candidates = lambda path: [Named()]
cli.certify_candidate = lambda *a, **k: {{
    "submission_script_name": "public_gmat_candidate.script",
    "official_report_files": ["ODC_2026_Team54.txt"], "stage": "public_gmat_verified",
    "conferred_stage": "public_gmat_verified"}}
sys.exit(cli.main(["certify", "x.script", "--config", {str(ROOT / 'config' / 'quick.yaml')!r},
                   "--rules", "preliminary",
                   "--candidates", "c.json", "--candidate", "C-0", "--gmat-console",
                   {str(ROOT / 'scripts' / 'fake_gmat_console.py')!r}, "--startup-file", "s",
                   "--output", {str(tmp_path / 'final')!r}]))
""", encoding="utf-8")
    done = subprocess.run([system.executable, str(driver)], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, check=False, cwd=tmp_path)
    assert done.returncode == 0, done.stdout
    assert done.stdout.splitlines()[-1].endswith("PRE-CHECK SKIPPED: the console is orbitwin's "
                                                 "test double, whose reports are not GMAT's")
