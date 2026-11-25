"""inspect's check of each official ReportFile against the rulebook."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from orbitwin.cli import main
from orbitwin.config import load_config
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import ReportFileSpec, Scenario
from orbitwin.official_report import (
    epoch_resolution_s,
    official_report_warnings,
    position_resolution_km,
)
from orbitwin.rules import PRELIMINARY, RANKING

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
ALTITUDE = "SC_Human.Earth.Altitude"


@pytest.fixture(scope="module")
def r1_config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    text = (ROOT / "config" / "ranking-rehearsal.yaml").read_text(encoding="utf-8")
    path = tmp_path_factory.mktemp("config") / "r1.yaml"
    path.write_text(text.replace("horizon_s: null", "horizon_s: 6711.667"), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def r1(r1_config_path: Path) -> Scenario:
    return parse_gmat_script(R1, load_config(r1_config_path))[0]


def with_report(scenario: Scenario, **changes: object) -> Scenario:
    (spec,) = scenario.report_files
    return dataclasses.replace(scenario, report_files=(dataclasses.replace(spec, **changes),))


def with_altitude(scenario: Scenario, **changes: object) -> Scenario:
    (spec,) = scenario.report_files
    return with_report(scenario, parameters=(*spec.parameters, ALTITUDE), **changes)


def test_the_ranking_template_only_lacks_altitude(r1: Scenario) -> None:
    (note,) = official_report_warnings(r1, RANKING)
    assert note.startswith("note: ReportFile1 has no SC_Human Altitude column")
    assert RANKING.earth_crossing_clause is not None
    assert RANKING.earth_crossing_clause in note
    assert official_report_warnings(with_altitude(r1), RANKING) == []


def test_altitude_is_not_asked_for_where_crossing_is_not_ruled(r1: Scenario) -> None:
    assert PRELIMINARY.earth_crossing == "not-ruled"
    assert official_report_warnings(r1, PRELIMINARY) == []


def test_write_report_false_is_flagged(r1: Scenario) -> None:
    (warning,) = official_report_warnings(with_altitude(r1, write_report=False), RANKING)
    assert "WriteReport is false" in warning


def test_precision_is_judged_against_the_rulebooks_sphere(r1: Scenario) -> None:
    coarse = with_altitude(r1, precision=3)
    sphere = [w for w in official_report_warnings(coarse, RANKING) if "sphere" in w]
    assert len(sphere) == 1 and "Precision = 3" in sphere[0] and "5 km sphere" in sphere[0]
    wide = dataclasses.replace(RANKING, intercept_radius_km=50000.0)
    assert not [w for w in official_report_warnings(coarse, wide) if "sphere" in w]


def test_position_resolution_is_gmats_significant_digits() -> None:
    # R2026a, Precision 3, 6, 7: 4766.091413204233 km wrote 4.77e+03, 4766.09, 4766.091.
    assert position_resolution_km(3, 4766.09) == pytest.approx(10.0)
    assert position_resolution_km(6, 4766.09) == pytest.approx(0.01)
    assert position_resolution_km(7, 4766.09) == pytest.approx(0.001)


def test_both_craft_in_one_frame(r1: Scenario) -> None:
    (spec,) = r1.report_files
    alien_elsewhere = tuple(
        p.replace("SC_Alien.EarthMJ2000Eq.", "SC_Alien.EarthFixed.") for p in spec.parameters
    )
    (warning,) = official_report_warnings(
        with_report(r1, parameters=(*alien_elsewhere, ALTITUDE)), RANKING
    )
    assert "in one frame" in warning
    pursuer_only = tuple(p for p in spec.parameters if not p.startswith("SC_Alien."))
    (warning,) = official_report_warnings(
        with_report(r1, parameters=(*pursuer_only, ALTITUDE)), RANKING
    )
    # R3 T16: the workshop style, which certify proves on its diagnostic flight.
    assert warning.startswith("ReportFile1, the official report, carries no SC_Alien position")
    assert "PRE-CHECK PASS (target not in report)" in warning and "exits 7" in warning
    assert "ask it (docs/RUNBOOK.md step 3)" in warning


def test_an_epoch_column_is_needed(r1: Scenario) -> None:
    (spec,) = r1.report_files
    no_epoch = tuple(p for p in spec.parameters if not p.endswith("ElapsedSecs"))
    warnings = official_report_warnings(with_report(r1, parameters=(*no_epoch, ALTITUDE)), RANKING)
    assert any("no numeric epoch column" in w for w in warnings)
    modjulian = ("SC_Human.ElapsedSecs", *no_epoch, "SC_Human.A1ModJulian", ALTITUDE)
    assert official_report_warnings(with_report(r1, parameters=modjulian), RANKING) == []


def test_an_unsplittable_delimiter_is_flagged(r1: Scenario) -> None:
    (warning,) = official_report_warnings(
        with_altitude(r1, delimiter="-", fixed_width=False), RANKING
    )
    assert "can occur inside a number" in warning


def test_a_narrow_column_without_headers_is_flagged(r1: Scenario) -> None:
    # O_noheader_narrow: ColumnWidth 10 without headers fused the values.
    (warning,) = official_report_warnings(
        with_altitude(r1, write_headers=False, column_width=10), RANKING
    )
    assert "run together" in warning
    assert official_report_warnings(with_altitude(r1, write_headers=False), RANKING) == []


def test_inspect_prints_the_check_to_stderr(
    r1_config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect", str(R1), "--config", str(r1_config_path)]) == 0
    out, err = capsys.readouterr()
    (spec,) = json.loads(out)["report_files"]
    assert spec["delimiter"] == " " and spec["write_headers"] is True
    assert "orbitwin inspect: note: ReportFile1 has no SC_Human Altitude column" in err


def test_the_spec_fields_default_to_gmats() -> None:
    spec = ReportFileSpec("R", "R.txt", (), 16, False)
    assert (spec.write_report, spec.write_headers, spec.delimiter, spec.fixed_width) == (
        True, True, " ", True
    )
    assert (spec.column_width, spec.left_justify, spec.zero_fill) == (23, True, False)


def test_the_epoch_column_must_resolve_the_completion_time(r1: Scenario) -> None:
    (spec,) = r1.report_files
    a1 = tuple(
        p.replace("SC_Alien.ElapsedSecs", "SC_Alien.A1ModJulian").replace(
            "SC_Human.ElapsedSecs", "SC_Human.A1ModJulian"
        )
        for p in spec.parameters
    )
    warnings = official_report_warnings(
        with_report(r1, parameters=(*a1, ALTITUDE), precision=7), RANKING
    )
    assert any("resolves SC_Alien.A1ModJulian to 864 s" in w for w in warnings)
    assert any("has no ElapsedSecs, so check_report_rows prints no score" in w for w in warnings)
    at_16 = official_report_warnings(with_report(r1, parameters=(*a1, ALTITUDE)), RANKING)
    assert not any("resolves" in w for w in at_16)


def test_epoch_resolution_is_gmats_significant_digits() -> None:
    # R2026a, Precision 7: A1ModJulian 31217.70876197201 wrote 31217.71.
    assert epoch_resolution_s("A.A1ModJulian", 7, 6711.667) == pytest.approx(864.0)
    assert epoch_resolution_s("A.ElapsedSecs", 7, 6711.667) == pytest.approx(1.0e-3)
    assert epoch_resolution_s("A.ElapsedDays", 16, 6711.667) == pytest.approx(
        86400.0 * 1.0e-17
    )


def test_a_short_name_in_a_narrow_column_is_flagged(r1: Scenario) -> None:
    # E_narrow: with headers a column takes max(ColumnWidth, len(name)) + 3.
    assert official_report_warnings(with_altitude(r1, column_width=10), RANKING) == []
    (spec,) = r1.report_files
    short = with_report(r1, parameters=(*spec.parameters, "A.X"), column_width=10)
    assert any("column A.X 13 characters" in w for w in official_report_warnings(short, RANKING))


def test_append_to_existing_file_is_flagged(r1: Scenario) -> None:
    (warning,) = official_report_warnings(with_altitude(r1, append=True), RANKING)
    assert "AppendToExistingFile is true" in warning


def test_a_text_column_is_flagged(r1: Scenario) -> None:
    (spec,) = r1.report_files
    gregorian = (*spec.parameters, "SC_Human.UTCGregorian", ALTITUDE)
    (warning,) = official_report_warnings(with_report(r1, parameters=gregorian), RANKING)
    assert "text columns ['SC_Human.UTCGregorian']" in warning
    # GMAT, no headers, ColumnWidth 23: the 24-character value ran into the next.
    fused = official_report_warnings(
        with_report(r1, parameters=gregorian, write_headers=False), RANKING
    )
    assert any("run together" in w for w in fused)


def test_inspect_reads_frames_as_the_check_does(r1: Scenario) -> None:
    (spec,) = r1.report_files
    short = tuple(p.replace(".EarthMJ2000Eq.", ".") for p in spec.parameters)
    assert official_report_warnings(with_report(r1, parameters=(*short, ALTITUDE)), RANKING) == []
    fixed = tuple(p.replace(".EarthMJ2000Eq.", ".EarthFixed.") for p in spec.parameters)
    assert official_report_warnings(with_report(r1, parameters=(*fixed, ALTITUDE)), RANKING) == []


def test_every_epoch_column_is_judged(r1: Scenario) -> None:
    # R2026a, Precision 9: A1ModJulian wrote 31353.636 on three rows running,
    # while ElapsedSecs beside it resolved 1e-5 s.
    (spec,) = r1.report_files
    # After the fine ElapsedSecs columns, so judging only the first epoch misses it.
    mixed = (*spec.parameters, "SC_Human.A1ModJulian", ALTITUDE)
    warnings = official_report_warnings(with_report(r1, parameters=mixed, precision=9), RANKING)
    assert any("resolves SC_Human.A1ModJulian to 8.64 s" in w for w in warnings)
    assert not any("resolves SC_Alien.ElapsedSecs" in w for w in warnings)


def test_a_delimiter_inside_a_gregorian_epoch_is_flagged(r1: Scenario) -> None:
    # GMAT, Delimiter ':' with UTCGregorian: the check could not split the rows.
    (spec,) = r1.report_files
    gregorian = (*spec.parameters, "SC_Human.UTCGregorian", ALTITUDE)
    warnings = official_report_warnings(
        with_report(r1, parameters=gregorian, delimiter=":"), RANKING
    )
    assert any("':' occurs inside a Gregorian epoch" in w for w in warnings)
    comma = official_report_warnings(with_report(r1, parameters=gregorian, delimiter=","), RANKING)
    assert not any("inside a Gregorian" in w for w in comma)


def test_the_position_step_is_held_to_one_percent_of_the_sphere(r1: Scenario) -> None:
    # Precision 6 near 3e4 km is 0.1 km: over 1% of 5 km, under 5 km.
    assert position_resolution_km(6, 2.9e4) == pytest.approx(0.1)
    warnings = official_report_warnings(with_altitude(r1, precision=6), RANKING)
    assert any("coarser than 0.01 of the 5 km sphere" in w for w in warnings)


def test_solver_iterations_all_is_flagged(r1: Scenario) -> None:
    (warning,) = official_report_warnings(with_altitude(r1, solver_iterations="All"), RANKING)
    assert "SolverIterations is All" in warning
    assert official_report_warnings(with_altitude(r1, solver_iterations="None"), RANKING) == []


def test_a_gregorian_column_is_widened_by_gmat(r1: Scenario) -> None:
    # R2026a, ColumnWidth 20 with short names: the Gregorian column got 27, the rest 23.
    short = ("SC_Human.UTCGregorian", "SC_Alien.ElapsedSecs", "SC_Alien.X", "SC_Alien.Y",
             "SC_Alien.Z", "SC_Human.X", "SC_Human.Y", "SC_Human.Z", ALTITUDE)
    warnings = official_report_warnings(
        with_report(r1, parameters=short, column_width=20), RANKING
    )
    assert not any("run together" in w for w in warnings)


def test_a_short_gregorian_name_in_a_narrow_column_is_not_flagged(r1: Scenario) -> None:
    # GMAT widened a Gregorian column to 27 at ColumnWidth 1 and 20, whatever its name.
    (spec,) = r1.report_files
    names = (*spec.parameters, "A.UTCGregorian", ALTITUDE)
    warnings = official_report_warnings(with_report(r1, parameters=names, column_width=10), RANKING)
    assert not any("run together" in w for w in warnings)


@pytest.mark.parametrize("width, flagged", [(19, True), (20, False)])
def test_a_numeric_column_is_flagged_at_precision_plus_six(
    r1: Scenario, width: int, flagged: bool
) -> None:
    # A.X takes max(ColumnWidth, 3) + 3; a value at Precision 16 can take 22.
    (spec,) = r1.report_files
    names = (*spec.parameters, "A.X", ALTITUDE)
    warnings = official_report_warnings(
        with_report(r1, parameters=names, column_width=width), RANKING
    )
    assert any("column A.X" in w and "run together" in w for w in warnings) is flagged


def test_an_alignment_asks_for_frozen_rows_and_the_later_craft(r1: Scenario) -> None:
    from orbitwin.models import EpochAlignment

    aligned = dataclasses.replace(
        with_altitude(r1),
        alignment=EpochAlignment(
            early="SC_Alien", late="SC_Human", propagator="DefaultProp", form="epoch-equality",
            offset_s=1800.0, text="", line_index=None,
        ),
    )
    (note,) = official_report_warnings(aligned, RANKING)
    assert "--frozen-rows" in note and "--time-column SC_Human.ElapsedSecs" in note
    (spec,) = r1.report_files
    without = tuple(p for p in spec.parameters if p != "SC_Human.ElapsedSecs")
    notes = official_report_warnings(
        dataclasses.replace(aligned, report_files=(dataclasses.replace(
            spec, parameters=(*without, ALTITUDE)),)), RANKING
    )
    assert any("which ReportFile1 does not carry" in w for w in notes)


def test_the_earlier_craft_s_elapsed_secs_runs_past_t_max(r1: Scenario) -> None:
    # 9000 s plus an 1800 s alignment reaches five digits: Precision 6 is 0.1 s.
    from orbitwin.models import EpochAlignment

    aligned = dataclasses.replace(
        with_altitude(r1, precision=6), max_time_s=9000.0,
        alignment=EpochAlignment(
            early="SC_Alien", late="SC_Human", propagator="DefaultProp", form="epoch-equality",
            offset_s=1800.0, text="", line_index=None,
        ),
    )
    warnings = official_report_warnings(aligned, RANKING)
    assert any("resolves SC_Alien.ElapsedSecs to 0.1 s" in w for w in warnings)


def test_an_early_origin_asks_for_the_early_craft_s_elapsed_secs(r1: Scenario) -> None:
    # The parser records the craft the origin resolved to; earliest names the early one.
    from orbitwin.models import EpochAlignment

    aligned = dataclasses.replace(
        with_altitude(r1), time_origin_offset_s=1800.0, time_origin_craft="SC_Alien",
        alignment=EpochAlignment(
            early="SC_Alien", late="SC_Human", propagator="DefaultProp", form="epoch-equality",
            offset_s=1800.0, text="", line_index=None,
        ),
    )
    (note,) = official_report_warnings(aligned, RANKING)
    assert "--time-column SC_Alien.ElapsedSecs" in note


@pytest.mark.parametrize("width, flagged", [(22, True), (23, False)])
def test_a_header_less_column_is_flagged_up_to_precision_plus_six(
    r1: Scenario, width: int, flagged: bool
) -> None:
    # R2026a, no headers, Precision 16: ColumnWidth 22 fused one line of 4520, 23 none.
    warnings = official_report_warnings(
        with_altitude(r1, write_headers=False, column_width=width), RANKING
    )
    assert any("run together" in w for w in warnings) is flagged


@pytest.mark.parametrize("precision, flagged", [(6, False), (5, True)])
def test_an_epoch_step_of_exactly_0_01_s_passes(
    r1: Scenario, precision: int, flagged: bool
) -> None:
    # ElapsedSecs over 6711.667 s: Precision 6 is 0.01 s, Precision 5 0.1 s.
    warnings = official_report_warnings(with_altitude(r1, precision=precision), RANKING)
    assert any("resolves SC_Alien.ElapsedSecs" in w for w in warnings) is flagged


def test_the_apoapsis_sizes_the_position_step(r1: Scenario) -> None:
    import numpy as np

    from orbitwin.models import StateVector

    mu = r1.mu_km3_s2
    circular = StateVector((7000.0, 0.0, 0.0), (0.0, float(np.sqrt(mu / 7000.0)), 0.0))
    # Perigee 7000 km, apoapsis 20000 km.
    v_peri = float(np.sqrt(mu * (2 / 7000.0 - 2 / 27000.0)))
    elliptic = StateVector((7000.0, 0.0, 0.0), (0.0, v_peri, 0.0))
    for orbit, flagged in ((circular, False), (elliptic, True)):
        scenario = dataclasses.replace(
            with_altitude(r1, precision=6),
            pursuer=dataclasses.replace(r1.pursuer, initial_state=circular),
            target=dataclasses.replace(r1.target, initial_state=orbit),
        )
        warnings = official_report_warnings(scenario, RANKING)
        assert any("coarser than 0.01 of the 5 km sphere" in w for w in warnings) is flagged
