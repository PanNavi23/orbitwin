"""Which GMAT resource types a template may create, and what private copies keep."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from conftest import DROPPED_NAMES, KEPT_NAMES, all_types_script, legacy_script, quick_config

from orbitwin.config import OrbitWinConfig, load_config, validate_config
from orbitwin.diagnostic import NEUTRALISED_PREFIX, add_diagnostic_report
from orbitwin.errors import ConfigurationError, ParseError
from orbitwin.gmat_parser import GmatScriptDocument, parse_gmat_script
from orbitwin.script_renderer import write_rendered_candidate
from orbitwin.search import zero_burn_candidate
from orbitwin.state_probe import build_probe_script, load_scenario


def _live(text: str, name: str) -> bool:
    return any(
        line.startswith("Create ") and line.split()[-1].rstrip(";") == name
        for line in text.splitlines()
    )


def test_every_newly_allowed_type_parses_and_probes(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    script = all_types_script(tmp_path)
    parse_gmat_script(script, OrbitWinConfig())
    scenario, _, _ = load_scenario(
        script, OrbitWinConfig(), gmat_console=fake_gmat, startup_file=fake_startup
    )
    assert scenario.state_probe is not None


def test_the_probe_copy_keeps_what_a_craft_names_and_drops_the_rest(tmp_path: Path) -> None:
    """R2026a refused `Pursuer.Tanks = {T1}` with T1 commented out: "Nonexistent Hardware"."""
    text = build_probe_script(GmatScriptDocument.read(all_types_script(tmp_path)))[0].decode()
    assert all(_live(text, name) for name in KEPT_NAMES)
    assert "\nPursuer.Tanks = {T1, ET1};" in text
    assert not any(_live(text, name) for name in DROPPED_NAMES)
    assert not _live(text, "FM") and not _live(text, "Prop")


def test_private_copies_drop_the_plots_and_ephemeris_and_the_submission_keeps_them(
    tmp_path: Path,
) -> None:
    config = quick_config()
    scenario, document = parse_gmat_script(all_types_script(tmp_path), config)
    sibling = add_diagnostic_report(document, scenario, "diag.txt").script_bytes.decode()
    for name in ("GT", "DDD", "EF", "OFV", "OFS"):
        assert not _live(sibling, name) and f"{NEUTRALISED_PREFIX}Create" in sibling, name
    assert _live(sibling, "T1") and _live(sibling, "DC")
    final = tmp_path / "final.script"
    write_rendered_candidate(
        final, tmp_path / "final.diff", document, scenario, zero_burn_candidate(scenario, config),
        config,
    )
    rendered = final.read_text(encoding="utf-8")
    assert all(_live(rendered, name) for name in (*KEPT_NAMES, *DROPPED_NAMES))


@pytest.mark.parametrize(
    "line",
    [
        "EF.Filename = '/tmp/alien.oem';",
        "EF.Filename = '../alien.oem';",
        "DC.ReportFile = '/tmp/dc.txt';",
    ],
)
def test_an_allowed_writer_outside_the_run_is_refused(tmp_path: Path, line: str) -> None:
    with pytest.raises(ParseError, match="Unsafe output filename"):
        parse_gmat_script(all_types_script(tmp_path, line + "\n"), OrbitWinConfig())


def test_a_type_a_person_admits_runs_only_in_the_submission(tmp_path: Path) -> None:
    script = all_types_script(tmp_path, "Create ConicalFOV FOV;\n")
    with pytest.raises(ParseError, match=r"policy\.extra_resource_types"):
        parse_gmat_script(script, OrbitWinConfig())
    base = quick_config()
    config = dataclasses.replace(
        base, policy=dataclasses.replace(base.policy, extra_resource_types=("ConicalFOV",))
    )
    validate_config(config)
    scenario, document = parse_gmat_script(script, config)
    assert not _live(build_probe_script(document, ("ConicalFOV",))[0].decode(), "FOV")
    sibling = add_diagnostic_report(document, scenario, "diag.txt").script_bytes.decode()
    assert not _live(sibling, "FOV")


@pytest.mark.parametrize("kind", ["EclipseLocator", "GmatFunction", "FileInterface"])
def test_the_parser_refuses_a_never_run_type_even_when_admitted(
    tmp_path: Path, kind: str
) -> None:
    """validate_config refuses such a setting first; this is the parser's own guard."""
    from orbitwin.gmat_parser import assert_resources_supported

    script = all_types_script(tmp_path, f"Create {kind} X;\n")
    with pytest.raises(ParseError, match=f"creates {kind} X, which is not run: it writes"):
        assert_resources_supported(GmatScriptDocument.read(script), [kind])


@pytest.mark.parametrize("kind", ["EclipseLocator", "GmatFunction", "FileInterface"])
def test_a_refused_type_cannot_be_admitted(tmp_path: Path, kind: str) -> None:
    config = tmp_path / "extra.yaml"
    config.write_text(f"policy:\n  extra_resource_types: [ConicalFOV, {kind}]\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=f"names \\['{kind}'\\]"):
        load_config(config)


def test_extra_types_load_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "extra.yaml"
    config.write_text("policy:\n  extra_resource_types: [ConicalFOV]\n", encoding="utf-8")
    assert list(load_config(config).policy.extra_resource_types) == ["ConicalFOV"]


def test_the_legacy_tank_and_thruster_names_are_kept_for_the_craft(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """R2026a: with FT commented out the probe failed, 'Nonexistent Hardware "FT"'."""
    script = legacy_script(tmp_path)
    scenario, document, _ = load_scenario(
        script, OrbitWinConfig(), gmat_console=fake_gmat, startup_file=fake_startup
    )
    probe = build_probe_script(document)[0].decode()
    sibling = add_diagnostic_report(document, scenario, "diag.txt").script_bytes.decode()
    for text in (probe, sibling):
        assert _live(text, "FT") and _live(text, "TH") and "\nPursuer.Tanks = {FT};" in text


def test_an_admitted_resource_the_craft_names_stays_in_both_private_copies(
    tmp_path: Path,
) -> None:
    script = legacy_script(
        tmp_path,
        "Create MyHardware MH;\nPursuer.AddHardware = {MH};\nCreate MyHardware Spare;\n",
    )
    base = quick_config()
    config = dataclasses.replace(
        base, policy=dataclasses.replace(base.policy, extra_resource_types=("MyHardware",))
    )
    scenario, document = parse_gmat_script(script, config)
    probe = build_probe_script(document, ("MyHardware",))[0].decode()
    sibling = add_diagnostic_report(document, scenario, "diag.txt").script_bytes.decode()
    for text in (probe, sibling):
        assert _live(text, "MH"), "the craft names it"
        assert not _live(text, "Spare"), "nothing names it"


def test_an_admitted_resource_that_reads_a_file_is_flown_by_every_run(
    tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """The exact runs read its file, so the sibling must too, or certify always refuses."""
    from orbitwin.verify import certify_candidate

    script = all_types_script(tmp_path, "Create ConicalFOV FOV;\nFOV.MaskFile = 'EGM96.cof';\n")
    base = quick_config()
    config = dataclasses.replace(
        base, policy=dataclasses.replace(base.policy, extra_resource_types=("ConicalFOV",))
    )
    scenario, document, _ = load_scenario(
        script, config, gmat_console=fake_gmat, startup_file=fake_startup
    )
    sibling = add_diagnostic_report(document, scenario, "diag.txt").script_bytes.decode()
    assert _live(sibling, "FOV")
    # The fake console reports a hit for any script, so the zero burn certifies.
    candidate = zero_burn_candidate(scenario, config)
    candidate.metadata.update(scenario_sha256=scenario.source_sha256, horizon_s=scenario.max_time_s)
    certification = certify_candidate(
        script, candidate, config, fake_gmat, fake_startup, tmp_path / "certify", runs=1
    )
    assert "script:FOV.MaskFile" in certification["gmat_data_files"]


def test_a_non_hardware_resource_named_as_hardware_stays_dropped(tmp_path: Path) -> None:
    """R2026a refuses the template itself: "EF" ... is not an object of Hardware."""
    script = all_types_script(tmp_path, "Pursuer.AddHardware = {EF};\n")
    scenario, document = parse_gmat_script(script, OrbitWinConfig())
    probe = build_probe_script(document)[0].decode()
    sibling = add_diagnostic_report(document, scenario, "diag.txt").script_bytes.decode()
    assert not _live(probe, "EF") and not _live(sibling, "EF")
    assert _live(probe, "T1")
