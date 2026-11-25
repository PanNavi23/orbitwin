"""The state probe on stock GmatConsole: EarthFixed insertion and the workshop template."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orbitwin.config import OrbitWinConfig
from orbitwin.errors import GmatRunError, ParseError
from orbitwin.gmat_parser import (
    PROBE_POSITION_TOLERANCE_KM,
    PROBE_VELOCITY_TOLERANCE_KM_S,
    parse_gmat_script,
)
from orbitwin.gmat_runner import run_gmat
from orbitwin.report_parser import parse_numeric_report
from orbitwin.state_probe import load_scenario, probe_initial_states

ROOT = Path(__file__).resolve().parents[2]
ASYNC_FIXTURE = ROOT / "tests" / "fixtures" / "async" / "insertion_and_async_start.script"
INSERTION = ("SC_Human", "SC_Alien")

pytestmark = pytest.mark.gmat


def _resources(script: Path) -> str:
    return script.read_text(encoding="utf-8").split("BeginMissionSequence;")[0]


def test_the_probed_insertion_state_is_the_first_row_gmat_propagates(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    console, startup = gmat
    probe = probe_initial_states(ASYNC_FIXTURE, console, startup, tmp_path / "probe", timeout_s=120)
    human, alien = (probe.craft_named(name) for name in INSERTION)
    assert human is not None and alien is not None
    # RMAG 6871 km, given in EarthFixed; the epochs are 15 h apart in A1 too.
    assert float(np.linalg.norm(human.state.r)) == pytest.approx(6871.0, abs=1.0e-9)
    assert (human.epoch.a1_mjd - alien.epoch.a1_mjd) * 86400.0 == pytest.approx(54000.0, abs=1e-6)
    assert alien.state.r_km == (7000.0, 0.0, 0.0)

    columns = ["SC_Human.A1ModJulian"] + [
        f"SC_Human.EarthMJ2000Eq.{axis}" for axis in ("X", "Y", "Z", "VX", "VY", "VZ")
    ]
    script = tmp_path / "first_row.script"
    script.write_text(
        _resources(ASYNC_FIXTURE)
        + "Create ReportFile First;\nFirst.Filename = 'first.txt';\nFirst.Precision = 17;\n"
        + "First.WriteHeaders = true;\nFirst.Add = {" + ", ".join(columns) + "};\n"
        + "BeginMissionSequence;\nPropagate DefaultProp(SC_Human) {SC_Human.ElapsedSecs = 10};\n",
        encoding="utf-8",
    )
    result = run_gmat(
        console, script, tmp_path / "run", ["first.txt"], startup_file=startup, timeout_s=120
    )
    assert result.success, result.error
    first = parse_numeric_report(result.reports["first.txt"]).data[0]
    # Measured bit-equal on R2026a: both are GMAT's conversion, printed at Precision 17.
    assert first[0] == human.epoch.a1_mjd
    assert tuple(first[1:]) == (*human.state.r_km, *human.state.v_km_s)


def test_an_insertion_on_one_epoch_parses_with_the_probe(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    console, startup = gmat
    script = tmp_path / "synchronous.script"
    script.write_text(
        _resources(ASYNC_FIXTURE).replace(
            "SC_Human.Epoch = '01 Jan 2000 15:00:00.000';",
            "SC_Human.Epoch = '01 Jan 2000 00:00:00.000';",
        )
        + "BeginMissionSequence;\n"
        + "Propagate DefaultProp(SC_Alien, SC_Human) {SC_Human.ElapsedSecs = 3000};\n",
        encoding="utf-8",
    )
    config = OrbitWinConfig(pursuer_name="SC_Human", target_name="SC_Alien")
    with pytest.raises(ParseError, match="--gmat-console"):
        parse_gmat_script(script, config)
    probe = probe_initial_states(script, console, startup, tmp_path / "probe", timeout_s=120)
    scenario, _ = parse_gmat_script(script, config, probe=probe)
    assert scenario.epoch_offset_s == 0.0
    assert scenario.pursuer.coordinate_system == "EarthFixed"
    assert scenario.pursuer.initial_state == probe.craft_named("SC_Human").state


def test_different_epochs_are_still_refused(tmp_path: Path, gmat: tuple[str, str]) -> None:
    console, startup = gmat
    script = tmp_path / "joint.script"
    script.write_text(
        _resources(ASYNC_FIXTURE)
        + "BeginMissionSequence;\n"
        + "Propagate DefaultProp(SC_Alien, SC_Human) {SC_Human.ElapsedSecs = 3000};\n",
        encoding="utf-8",
    )
    probe = probe_initial_states(script, console, startup, tmp_path / "probe", timeout_s=120)
    with pytest.raises(ParseError, match=r"54000 s apart.*no epoch alignment.*Q4"):
        parse_gmat_script(
            script, OrbitWinConfig(pursuer_name="SC_Human", target_name="SC_Alien"), probe=probe
        )


def test_the_workshop_template_probes_through_its_openframes_objects(
    tmp_path: Path, gmat: tuple[str, str], workshop_script: Path
) -> None:
    """GmatConsole refuses the template itself (58 errors, rc 1); the probe copy runs."""
    console, startup = gmat
    probe = probe_initial_states(workshop_script, console, startup, tmp_path / "probe", timeout_s=120)
    scenario, _ = parse_gmat_script(workshop_script, OrbitWinConfig(), probe=probe)
    own, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    assert scenario.epoch_offset_s == 0.0
    for probed, ours in ((scenario.pursuer, own.pursuer), (scenario.target, own.target)):
        assert probed.in_process_mismatch is not None
        position_km, velocity_km_s = probed.in_process_mismatch
        assert position_km <= PROBE_POSITION_TOLERANCE_KM
        assert velocity_km_s <= PROBE_VELOCITY_TOLERANCE_KM_S
        assert np.linalg.norm(probed.initial_state.r - ours.initial_state.r) == position_km
    epoch = scenario.pursuer.probed_epoch
    assert epoch is not None
    # 15 Mar 2027: each scale in its own field, by GMAT's own offsets.
    assert (epoch.tai_mjd - epoch.utc_mjd) * 86400.0 == pytest.approx(37.0, abs=1e-5)
    assert (epoch.a1_mjd - epoch.tai_mjd) * 86400.0 == pytest.approx(0.0343817, abs=1e-5)
    assert (epoch.tt_mjd - epoch.tai_mjd) * 86400.0 == pytest.approx(32.184, abs=1e-5)
    assert 0.0 < abs(epoch.tdb_mjd - epoch.tt_mjd) * 86400.0 < 2.0e-3
    hashed = probe.run.data_file_sha256
    for key in ("EOP_FILE", "LEAP_SECS_FILE", "EGM96_FILE", "CSSI_FLUX_FILE", "SPAD_SRP_FILE",
                "PLANETARY_SPK_FILE"):
        assert len(hashed[key]["sha256"]) == 64, key


def test_a_report_outside_the_run_survives_the_probe(
    tmp_path: Path, gmat: tuple[str, str], workshop_script: Path, monkeypatch
) -> None:
    """Before the fix one probe of this script deleted the sentinel on R2026a."""
    import orbitwin.state_probe as state_probe

    sentinel = tmp_path / "official_abs.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    script = tmp_path / "absolute.script"
    script.write_text(
        workshop_script.read_text(encoding="utf-8").replace(
            "ReportFile1.Filename = 'ReportFile1.txt';", f"ReportFile1.Filename = '{sentinel}';"
        ),
        encoding="utf-8",
    )
    console, startup = gmat
    with pytest.raises(ParseError, match="Unsafe output filename"):
        probe_initial_states(script, console, startup, tmp_path / "refused", timeout_s=120)
    # Neutralising alone must also keep it, with the path check out of the way.
    monkeypatch.setattr(state_probe, "assert_resources_supported", lambda *_args: None)
    probe_initial_states(script, console, startup, tmp_path / "probe", timeout_s=120)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_an_eop_file_the_script_names_is_in_the_manifest(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    """R2026a read Earth.EopFileName instead of the startup file's EOP."""
    import hashlib

    console, startup = gmat
    eop = tmp_path / "eop_copy.txt"
    shipped = Path(console).resolve().parent.parent / "data" / "planetary_coeff" / "eopc04_08.62-now"
    eop.write_bytes(shipped.read_bytes() + b"\n")
    script = tmp_path / "named_eop.script"
    script.write_text(
        ASYNC_FIXTURE.read_text(encoding="utf-8").replace(
            "Create ForceModel", f"Earth.EopFileName = '{eop}';\n\nCreate ForceModel", 1
        ),
        encoding="utf-8",
    )
    probe = probe_initial_states(script, console, startup, tmp_path / "probe", timeout_s=120)
    assert probe.run.data_file_sha256["script:Earth.EopFileName"] == {
        "path": str(eop.resolve()), "sha256": hashlib.sha256(eop.read_bytes()).hexdigest()
    }


def test_solve_and_certify_an_earth_fixed_insertion_end_to_end(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    """Every re-parse of a derived script (the renderer's guard, verify's flown scripts,
    the coast run, certify's replay) needs the probe: in-process they refuse EarthFixed."""
    import dataclasses

    from conftest import quick_config

    from orbitwin.config import load_config
    from orbitwin.ephemeris import target_ephemeris_from_gmat
    from orbitwin.io import read_candidates
    from orbitwin.pipeline import solve_scenario
    from orbitwin.state_probe import load_scenario
    from orbitwin.verify import certify_candidate

    console, startup = gmat
    script = tmp_path / "earth_fixed.script"
    script.write_text(
        _resources(ASYNC_FIXTURE)
        .replace(
            "SC_Human.Epoch = '01 Jan 2000 15:00:00.000';",
            "SC_Human.Epoch = '01 Jan 2000 00:00:00.000';",
        )
        # 7.2 km/s puts perigee near 6031 km, inside the Earth.
        .replace("SC_Human.VMAG = 7.2;", "SC_Human.VMAG = 7.4;")
        + "BeginMissionSequence;\n"
        + "Propagate DefaultProp(SC_Alien, SC_Human) {SC_Human.ElapsedSecs = 3000};\n",
        encoding="utf-8",
    )
    base = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    base = dataclasses.replace(base, policy=dataclasses.replace(base.policy, team_number=54))
    # Ranking rules, so a GMAT-replayed miss is verified and certifiable.
    config = dataclasses.replace(
        base,
        search=dataclasses.replace(quick_config(local_refine=True).search, workers=1),
        policy=dataclasses.replace(base.policy, horizon_s=3000.0),
        gmat=dataclasses.replace(base.gmat, verify_top=2, reruns=1, workers=1),
    )
    summary = solve_scenario(
        script, config, tmp_path / "solve", gmat_console=console, startup_file=startup
    )
    assert summary["gmat_verified_count"] >= 1
    best = next(
        candidate
        for candidate in read_candidates(tmp_path / "solve" / "candidates.json")
        if candidate.candidate_id == summary["best_candidate"]["candidate_id"]
    )
    certification = certify_candidate(
        script, best, config, console, startup, tmp_path / "certify", runs=1
    )
    assert certification["conferred_stage"] == "public_gmat_verified"
    assert certification["state_probe"]["craft"][1]["name"] == "SC_Human"
    assert "EOP_FILE" in certification["gmat_data_files"]

    scenario, document, _ = load_scenario(
        script, config, gmat_console=console, startup_file=startup
    )
    ephemeris = target_ephemeris_from_gmat(
        scenario, document, config, console, startup, tmp_path / "coast"
    )
    assert ephemeris.times_s[-1] == pytest.approx(3000.0)


#: GMAT resources measured writing their file wherever it points.
GMAT_WRITERS = {
    "ReportFile": "Create ReportFile W;\nW.Filename = '{0}';\nW.Add = {{AlienSC.A1ModJulian}};",
    "EphemerisFile": "Create EphemerisFile W;\nW.Spacecraft = AlienSC;\nW.Filename = '{0}';",
    "EclipseLocator": (
        "Create EclipseLocator W;\nW.Spacecraft = AlienSC;\nW.Filename = '{0}';\n"
        "W.RunMode = Automatic;\nW.OccultingBodies = {{Earth}};"
    ),
    "ContactLocator": (
        "Create GroundStation GS;\nCreate ContactLocator W;\nW.Target = AlienSC;\n"
        "W.Observers = {{GS}};\nW.Filename = '{0}';\nW.RunMode = Automatic;"
    ),
}


@pytest.mark.parametrize("kind", sorted(GMAT_WRITERS))
def test_a_file_writer_is_refused_before_gmat_and_would_have_written(
    tmp_path: Path, gmat: tuple[str, str], kind: str
) -> None:
    from orbitwin.gmat_runner import run_gmat as run

    console, startup = gmat
    workshop = ROOT / "examples" / "synthetic" / "DifferentPlane_Circular.script"
    sentinel = tmp_path / "precious.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    block = GMAT_WRITERS[kind].format(sentinel)
    included = tmp_path / "inc.script"
    included.write_text(block + "\n", encoding="utf-8")
    text = workshop.read_text(encoding="utf-8")
    for name, insert in (("direct", block), ("include", f"#Include '{included}'")):
        script = tmp_path / f"{name}.script"
        script.write_text(
            text.replace("BeginMissionSequence;", insert + "\nBeginMissionSequence;", 1),
            encoding="utf-8",
        )
        if name == "include":
            expected = "Include"
        elif kind == "ReportFile":
            expected = "Unsafe output filename"
        else:
            expected = "creates (ContactLocator|EclipseLocator)|Unsafe output filename"
        with pytest.raises((ParseError, GmatRunError), match=expected):
            load_scenario(script, OrbitWinConfig(), gmat_console=console, startup_file=startup)
        assert sentinel.read_text(encoding="utf-8") == "keep\n", name
    # Measured, not assumed: the same resource run as written reaches the sentinel.
    run(console, tmp_path / "include.script", tmp_path / "raw", [], startup_file=startup,
        timeout_s=120)
    assert sentinel.read_text(encoding="utf-8") != "keep\n"


def test_a_failed_probe_says_what_gmat_said(tmp_path: Path, gmat: tuple[str, str]) -> None:
    console, startup = gmat
    script = tmp_path / "missing_eop.script"
    script.write_text(
        ASYNC_FIXTURE.read_text(encoding="utf-8").replace(
            "Create ForceModel", "Earth.EopFileName = '/nonexistent/eop.txt';\n\nCreate ForceModel", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(GmatRunError, match=r"Error opening EopFile /nonexistent/eop\.txt"):
        load_scenario(
            script, OrbitWinConfig(pursuer_name="SC_Human"), gmat_console=console,
            startup_file=startup,
        )


def test_every_newly_allowed_type_probes_on_gmat(tmp_path: Path, gmat: tuple[str, str]) -> None:
    """A craft naming its tanks, thrusters and power system still probes: they stay."""
    from conftest import all_types_script

    from orbitwin.diagnostic import add_diagnostic_report
    from orbitwin.gmat_runner import run_gmat as run

    console, startup = gmat
    script = all_types_script(tmp_path)
    scenario, document, _ = load_scenario(
        script, OrbitWinConfig(), gmat_console=console, startup_file=startup
    )
    assert scenario.pursuer.in_process_mismatch is not None
    assert scenario.pursuer.in_process_mismatch[0] < 1.0e-6
    # The diagnostic sibling runs with GroundTrackPlot, the OpenFrames objects and
    # the EphemerisFile commented out.
    sibling = tmp_path / "sibling.script"
    sibling.write_bytes(add_diagnostic_report(document, scenario, "diag.txt").script_bytes)
    result = run(
        console, sibling, tmp_path / "run", ["OfficialReport.txt", "diag.txt"],
        startup_file=startup, timeout_s=120,
    )
    assert result.success, result.error


@pytest.mark.parametrize("form", ["relative", "absolute"])
def test_a_backslash_path_never_reaches_gmat(
    tmp_path: Path, gmat: tuple[str, str], form: str
) -> None:
    """R2026a reads a backslash as "/": through baseline these wrote outside the run."""
    from orbitwin.gmat_runner import run_gmat as run
    from orbitwin.verify import baseline_original_script

    console, startup = gmat
    place = tmp_path / "a" / "b"
    place.mkdir(parents=True)
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    name = "..\\..\\..\\..\\..\\sentinel.txt" if form == "relative" else str(sentinel).replace("/", "\\")
    script = tmp_path / "backslash.script"
    script.write_text(
        (ROOT / "examples" / "synthetic" / "DifferentPlane_Circular.script")
        .read_text(encoding="utf-8")
        .replace("OfficialReport.Filename = 'OfficialReport.txt';", f"OfficialReport.Filename = '{name}';"),
        encoding="utf-8",
    )
    with pytest.raises(ParseError, match="is not a bare file name"):
        baseline_original_script(script, OrbitWinConfig(), console, startup, place / "baseline")
    with pytest.raises(GmatRunError, match="is not a bare file name"):
        run(console, script, place / "run", [name], startup_file=startup, timeout_s=120)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert not list(tmp_path.rglob("baseline/run"))


def test_a_legacy_fuel_tank_and_thruster_probe_on_gmat(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    """R2026a still runs FuelTank and Thruster; the craft names them, so they stay."""
    from conftest import legacy_script

    from orbitwin.diagnostic import add_diagnostic_report
    from orbitwin.gmat_runner import run_gmat as run

    console, startup = gmat
    scenario, document, _ = load_scenario(
        legacy_script(tmp_path), OrbitWinConfig(), gmat_console=console, startup_file=startup
    )
    assert scenario.state_probe is not None
    sibling = tmp_path / "sibling.script"
    sibling.write_bytes(add_diagnostic_report(document, scenario, "diag.txt").script_bytes)
    result = run(
        console, sibling, tmp_path / "run", ["OfficialReport.txt", "diag.txt"],
        startup_file=startup, timeout_s=120,
    )
    assert result.success, result.error


@pytest.mark.parametrize("name", ["Report File1.txt", "ODC_2026_Team(07).txt"])
def test_a_name_with_a_space_or_parentheses_lands_in_the_run(
    tmp_path: Path, gmat: tuple[str, str], name: str
) -> None:
    from orbitwin.gmat_runner import run_gmat as run

    console, startup = gmat
    script = tmp_path / "named.script"
    script.write_text(
        (ROOT / "examples" / "synthetic" / "DifferentPlane_Circular.script")
        .read_text(encoding="utf-8")
        .replace("OfficialReport.Filename = 'OfficialReport.txt';", f"OfficialReport.Filename = '{name}';"),
        encoding="utf-8",
    )
    parse_gmat_script(script, OrbitWinConfig())
    result = run(console, script, tmp_path / "run", [name], startup_file=startup, timeout_s=120)
    assert result.success, result.error
    assert Path(result.reports[name]) == (tmp_path / "run" / "gmat-output" / name).resolve()
