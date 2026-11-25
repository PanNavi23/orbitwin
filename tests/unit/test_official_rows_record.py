"""certify reads its official report as printed, as step 7 does, and refuses a disagreement.

R3 (2026-11-06) T14: at Precision 4 the certificate claimed 95.7635 and its
step 7 printed 95.7623 and exited 1, the last row 1258 m inside against the
1500 m stop depth.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, PolicyConfig, ScoreConfig
from orbitwin.errors import GmatRunError, PolicyError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.io import candidate_from_dict
from orbitwin.official_report import printed_distance_margin_km, same_printed_value
from orbitwin.verify import NO_TARGET_PROOF, official_rows_record

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "tests" / "fixtures" / "report_rows"
#: R1's hit, stopped 50 m inside, at Precision 16.
REPORT = ROWS / "R1_hit_default.txt"
LAST_S = 4517.612829012796


def _config(stop_depth_km: float = 0.05) -> OrbitWinConfig:
    policy = dataclasses.replace(PolicyConfig(), rules="ranking", horizon_s=6711.667,
                                 team_number=54, dv_limit_km_s=0.25, dv_safe_limit_km_s=0.249,
                                 total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99,
                                 stop_depth_km=stop_depth_km)
    score = ScoreConfig(kt=0.003, ct=5369.334, kv=0.0012, cv=2000.0, delta_v_unit="m/s",
                        cr_km=300.0)
    return dataclasses.replace(OrbitWinConfig(), policy=policy, score=score)


def _record(tmp_path: Path, *, columns: tuple[str, ...] | None = None, hit: bool = True,
            t_team_s: float = LAST_S, **config: float) -> dict[str, object]:
    cfg = _config(**config)
    scenario, _ = parse_gmat_script(ROWS / "R1_hit.script", cfg)
    (spec,) = scenario.report_files
    path = REPORT
    if columns is not None:
        lines = REPORT.read_text(encoding="utf-8").splitlines()
        names = lines[0].split()
        kept = [names.index(column) for column in columns]
        path = tmp_path / "cut.txt"
        path.write_text("\n".join("   ".join(line.split()[i] for i in kept) for line in lines)
                        + "\n", encoding="utf-8")
        spec = dataclasses.replace(spec, parameters=columns)
    data = json.loads((ROOT / "tests" / "fixtures" / "report_bind" / "R1_hit.candidate.json")
                      .read_text(encoding="utf-8"))
    verified = candidate_from_dict(data)
    return official_rows_record(
        str(path), spec, scenario, cfg, verified, frozen_rows=0, t_team_s=t_team_s, hit=hit,
        completion_s=LAST_S if hit else None, diagnostic_distance_km=4.95)


def test_a_hit_is_scored_on_its_printed_last_row(tmp_path: Path) -> None:
    record = _record(tmp_path)
    assert record["carries_target_position"] is True and record["time_column"] == (
        "SC_Human.ElapsedSecs")
    assert record["last_row_time_s"] == LAST_S
    assert record["stop_depth_km"] == pytest.approx(0.05, abs=1.0e-3)
    assert record["proof"] == "the official report's rows"
    total_m_s = candidate_from_dict(json.loads(
        (ROOT / "tests" / "fixtures" / "report_bind" / "R1_hit.candidate.json")
        .read_text(encoding="utf-8"))).total_dv_km_s * 1000.0
    expected = (50.0 + 25.0 / (1.0 + math.exp(0.003 * (LAST_S - 5369.334)))
                + 25.0 / (1.0 + math.exp(0.0012 * (total_m_s - 2000.0))))
    assert record["score"] == pytest.approx(expected, abs=1.0e-9)


def test_a_last_row_short_of_the_stop_depth_on_its_digits_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match=r"50 m inside .* short of policy.stop_depth_km = "
                       r"1500 m: step 7 would refuse it"):
        _record(tmp_path, stop_depth_km=1.5)


def test_a_last_row_off_t_team_in_its_digits_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GmatRunError, match=re.escape("last row is at 4517.612829012796 s on "
                                                     "SC_Human.ElapsedSecs, and T_team is")):
        _record(tmp_path, t_team_s=LAST_S + 1.0e-6)


def test_a_miss_whose_printed_rows_read_as_a_hit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GmatRunError, match="GMAT flew a miss, and the official report's rows"):
        _record(tmp_path, hit=False, t_team_s=LAST_S)


def test_without_the_target_the_proof_is_the_diagnostic_flight(tmp_path: Path) -> None:
    record = _record(tmp_path, columns=("SC_Human.ElapsedSecs", "SC_Human.EarthMJ2000Eq.X",
                                        "SC_Human.EarthMJ2000Eq.Y", "SC_Human.EarthMJ2000Eq.Z"),
                     stop_depth_km=1.5)
    assert record["carries_target_position"] is False
    assert record["last_row_distance_km"] is None and record["stop_depth_km"] is None
    assert record["proof"] == NO_TARGET_PROOF.format(target="SC_Alien")
    assert record["scored_distance_km"] == 4.95


def test_the_printed_margin_is_one_last_digit_per_axis() -> None:
    cfg = _config()
    scenario, _ = parse_gmat_script(ROWS / "R1_hit.script", cfg)
    (spec,) = scenario.report_files
    at = {4: 1.0, 6: 0.01, 8: 1.0e-4, 16: 1.0e-12}
    for precision, digit_km in at.items():
        coarse = dataclasses.replace(spec, precision=precision)
        assert printed_distance_margin_km(coarse, "SC_Human", "SC_Alien", 6600.0) == (
            pytest.approx(math.sqrt(3.0) * digit_km))
    pursuer_only = dataclasses.replace(
        spec, parameters=tuple(p for p in spec.parameters if p.startswith("SC_Human")))
    assert printed_distance_margin_km(pursuer_only, "SC_Human", "SC_Alien", 6600.0) == 0.0


def test_a_printed_value_is_its_value_to_precision_digits() -> None:
    assert same_printed_value(4538.0, 4537.758, 4)
    assert not same_printed_value(4538.0, 4537.4, 4)
    assert same_printed_value(4537.75823089527, 4537.758230895270, 16)
    assert not same_printed_value(4537.75823089527, 4537.75823089627, 16)


#: The last rows of R8 as written, certified on R2026a (review of #195): T_team
#: 5911.5072750952095 s, which its Precision 16 prints as 5911.507275095209.
R8_TAIL = ROOT / "tests" / "fixtures" / "printed_rows" / "R8_hit_tail.txt"
R8_T_TEAM_S = 5911.5072750952095


def test_r8s_last_row_is_its_t_team_as_gmat_printed_it(tmp_path: Path) -> None:
    """Review of #195, P1: half the 16th digit, 5e-13 s, refused it; the double below is
    9.1e-13 s away. main certified it at 95.0656."""
    from orbitwin.gmat_parser import report_file_specs
    from orbitwin.official_report import read_official_rows

    (spec,) = [s for s in report_file_specs(ROOT / "examples" / "ranking"
                                            / "R8_PursuerFirst_Aligned.script")
               if s.name == "ReportFile1"]
    rows = read_official_rows(R8_TAIL, spec, "SC_Human", "SC_Alien", frozen_rows=0,
                              time_columns=["SC_Human.ElapsedSecs"])
    assert rows.last_row_time_s == 5911.507275095209 != R8_T_TEAM_S
    assert same_printed_value(rows.last_row_time_s, R8_T_TEAM_S, spec.precision)
    assert abs(rows.last_row_time_s - R8_T_TEAM_S) > 0.5 * 1.0e-12
    # The certify check itself, R8's report under R1's names and a T_team on its clock.
    cfg = _config(stop_depth_km=1.5)
    scenario, _ = parse_gmat_script(ROWS / "R1_hit.script", cfg)
    scenario = dataclasses.replace(scenario, time_origin_craft="SC_Human")
    verified = candidate_from_dict(json.loads(
        (ROOT / "tests" / "fixtures" / "report_bind" / "R1_hit.candidate.json")
        .read_text(encoding="utf-8")))
    record = official_rows_record(str(R8_TAIL), spec, scenario, cfg, verified, frozen_rows=0,
                                  t_team_s=R8_T_TEAM_S, hit=True, completion_s=R8_T_TEAM_S,
                                  diagnostic_distance_km=3.5)
    assert record["stop_depth_km"] >= 1.5 * 0.99
    with pytest.raises(GmatRunError, match="step 7 would refuse it"):
        official_rows_record(str(R8_TAIL), spec, scenario, cfg, verified, frozen_rows=0,
                             t_team_s=R8_T_TEAM_S + 3.0e-12, hit=True,
                             completion_s=R8_T_TEAM_S, diagnostic_distance_km=3.5)


def test_verify_stops_deeper_by_the_printed_margin(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Review of #195, Low-2: an official report at Precision 6 with both craft's positions
    rounds them to a last digit of 10 m below 10000 km and 100 m above; the stop moves
    sqrt(3) of that deeper. The test double flies above 10000 km here."""
    from conftest import capped_one_burn, ranking_copy, ranking_quick_config

    from orbitwin.official_report import last_digit
    from orbitwin.verify import verify_candidate_with_gmat

    text = ranking_copy(workshop_script, tmp_path).read_text(encoding="utf-8")
    old = "ReportFile1.Add = {HumanSC.A1ModJulian, HumanSC.EarthMJ2000Eq.X};"
    assert old in text and "ReportFile1.Precision = 16;" in text
    both = ", ".join(f"{craft}.EarthMJ2000Eq.{axis}" for craft in ("HumanSC", "AlienSC")
                     for axis in "XYZ")
    script = tmp_path / "precision6.script"
    script.write_text(text.replace(old, f"ReportFile1.Add = {{HumanSC.ElapsedSecs, {both}}};")
                      .replace("ReportFile1.Precision = 16;", "ReportFile1.Precision = 6;"),
                      encoding="utf-8")
    config = ranking_quick_config()
    scenario, document = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    verified, _, _ = verify_candidate_with_gmat(script, scenario, document, candidate, config,
                                                fake_gmat, fake_startup, tmp_path / "v")
    event = verified.metadata["gmat_event"]
    assert event["printed_margin_km"] == pytest.approx(math.sqrt(3.0) * last_digit(6, 12000.0))
    assert event["stop_depth_km"] >= (config.policy.stop_depth_km + event["printed_margin_km"]
                                      - 1.0e-6)
