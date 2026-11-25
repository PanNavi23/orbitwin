"""Verification refuses a run where GMAT flew another impulse than the script claims.

The fake console reads `Burn.V`, `.N` and `.B` as R2026a does, as Element1-3
with the later assignment winning (E2E audit A4). The parser refuses those
fields, so the parser's guard is switched off here: this is the second layer,
which must hold on its own.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    capped_one_burn,
    ranking_copy,
    ranking_quick_config,
    stamped,
    startup_for_console_elsewhere,
)

from orbitwin import gmat_parser
from orbitwin import verify as verify_module
from orbitwin.config import OrbitWinConfig, PolicyConfig
from orbitwin.errors import GmatRunError
from orbitwin.frames import script_components_to_inertial
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Candidate, as_vector3
from orbitwin.state_probe import load_scenario
from orbitwin.verify import certify_candidate as _certify_candidate

# _case flies the plan's hit as a 6 km miss: certified as GMAT flies it, on purpose.
certify_candidate = partial(_certify_candidate, accept_gmat_loss=True)

ROW_STEP_S = 7.0
ALIASES = "ImpulsiveBurn1.V = 0;\nImpulsiveBurn2.V = 0;\n"
ANCHOR = "BeginMissionSequence;"


@pytest.fixture()
def workshop_script(workshop_script: Path, tmp_path: Path) -> Path:
    """The workshop with the ranking report name; certify refuses it without one."""
    return ranking_copy(workshop_script, tmp_path)


def _no_burn_guard(monkeypatch) -> None:
    monkeypatch.setattr(gmat_parser, "_refuse_unmodelled_burn_fields", lambda *args, **kwargs: None)


def _case(
    workshop: Path, tmp_path: Path, aliases: str, *, text: str | None = None,
    config: OrbitWinConfig | None = None, miss_km: float = 6.0,
) -> tuple[Path, OrbitWinConfig, Candidate]:
    """The workshop's one-burn plan, cut to 0.199 km/s, a flyby `miss_km` off after its burn."""
    config = config or ranking_quick_config()
    scenario, _ = parse_gmat_script(workshop, config)
    candidate = capped_one_burn(
        scenario, config, over_km_s=0.8 * config.policy.dv_safe_limit_km_s
    )
    closest_s = max(burn.time_s for burn in candidate.burns) + ROW_STEP_S
    text = text if text is not None else workshop.read_text(encoding="utf-8")
    assert text.count(ANCHOR) == 1
    script = tmp_path / "template.script"
    script.write_text(
        f"% FAKE_GMAT_FLYBY {closest_s!r} {ROW_STEP_S}\n% FAKE_GMAT_MISS_KM {miss_km!r}\n"
        + text.replace(ANCHOR, aliases + ANCHOR),
        encoding="utf-8",
    )
    return script, config, stamped(candidate, script, config)


def _site(path: object, tmp_path: Path) -> str:
    replay = path is not None and Path(str(path)).is_relative_to(tmp_path / "c" / "final-parity")
    return "replay" if replay else "flight"


def test_a_clean_script_is_certified_with_every_burn_checked(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    script, config, candidate = _case(workshop_script, tmp_path, "")
    checked: list[tuple[str, int]] = []
    real = verify_module.check_flown_impulses

    def counting(path, maneuvers, *, complete):
        checked.append((str(path), len(maneuvers)))
        real(path, maneuvers, complete=complete)

    monkeypatch.setattr(verify_module, "check_flown_impulses", counting)
    certify_candidate(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1)
    assert {_site(path, tmp_path) for path, _ in checked} == {"flight", "replay"}
    assert all(path.endswith("_burns.txt") and count == 2 for path, count in checked)


@pytest.mark.parametrize("kept", ["flight", "replay"])
def test_an_aliased_impulse_is_refused_by_either_run(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch, kept: str,
) -> None:
    """Each call site alone refuses it; the other is switched off as a mutant would be."""
    _no_burn_guard(monkeypatch)
    script, config, candidate = _case(workshop_script, tmp_path, ALIASES)
    real = verify_module.check_flown_impulses
    refused: list[str] = []

    def one_site(path, maneuvers, *, complete):
        site = _site(path, tmp_path)
        if site == kept:
            try:
                real(path, maneuvers, complete=complete)
            except GmatRunError:
                refused.append(site)
                raise

    monkeypatch.setattr(verify_module, "check_flown_impulses", one_site)
    with pytest.raises(GmatRunError, match="Something in the script changes the impulse GMAT flies"):
        certify_candidate(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1)
    assert refused == [kept]


def test_verification_alone_refuses_an_aliased_impulse(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    _no_burn_guard(monkeypatch)
    script, config, candidate = _case(workshop_script, tmp_path, ALIASES)
    scenario, document = parse_gmat_script(script, config)
    claimed = max(burn.magnitude_km_s for burn in candidate.burns)
    assert not math.isclose(claimed, 0.0)
    with pytest.raises(GmatRunError, match=r"GMAT flew Maneuver \d \(ImpulsiveBurn\d"):
        verify_module.verify_candidate_with_gmat(
            script, scenario, document, candidate, config, fake_gmat, fake_startup,
            tmp_path / "v",
        )


@pytest.mark.parametrize("site", ["flight", "replay"])
def test_a_finished_run_that_reports_too_few_maneuvers_is_refused(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    monkeypatch, site: str,
) -> None:
    """A run that finished flew every Maneuver, so each must be in its burn report."""
    script, config, candidate = _case(workshop_script, tmp_path, "")
    real = verify_module.run_gmat

    def losing_a_pair(console, sibling, workdir, names, **kwargs):
        result = real(console, sibling, workdir, names, **kwargs)
        burns = [name for name in names if name.endswith("_burns.txt")]
        if burns and _site(workdir, tmp_path) == site:
            path = Path(result.reports[burns[0]])
            path.write_text("\n".join(path.read_text().splitlines()[:2]) + "\n")
        return result

    monkeypatch.setattr(verify_module, "run_gmat", losing_a_pair)
    with pytest.raises(GmatRunError, match="has 2 rows for 2 Maneuvers; it needs two each"):
        certify_candidate(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1)


_DROPPING = '''import os, subprocess, sys
from pathlib import Path
argv = sys.argv[1:]
code = subprocess.run([sys.executable, {fake!r}, *argv]).returncode
run = argv[argv.index("--run") + 1]
only = os.environ.get("P2_DROP_ONLY", "")
if code == 0 and "state_probe" not in run and only in run:
    startup = Path(argv[argv.index("--startup_file") + 1])
    out = next(Path(line.split("=", 1)[1].strip()) for line in startup.read_text().splitlines()
               if line.strip().upper().startswith("OUTPUT_PATH"))
    for path in out.glob("*_burns.txt"):
        path.unlink() if os.environ["P2_DROP"] == "drop" else path.write_bytes(b"")
sys.exit(code)
'''


@pytest.mark.parametrize("mode", ["drop", "empty"])
@pytest.mark.parametrize(("end", "only"), [("horizon", ""), ("interception", "at_stop")])
def test_a_run_that_exits_0_without_its_burn_report_is_refused(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, monkeypatch, mode: str, end: str,
    only: str,
) -> None:
    """Review V1: GMAT exits 0 and every report but the burn one is there. The only
    flight under `horizon`, or the stop flight, used to accept the run unchecked."""
    console = tmp_path / "console" / "GmatConsole.py"
    console.parent.mkdir()
    console.write_text(f"#!{sys.executable}\n" + _DROPPING.format(fake=str(fake_gmat)))
    console.chmod(0o755)
    base = ranking_quick_config()
    config = dataclasses.replace(
        base, policy=dataclasses.replace(base.policy, mission_end_policy=end)
    )
    # A hit, so `interception` flies a stop flight after the measuring one.
    script, config, candidate = _case(workshop_script, tmp_path, "", config=config, miss_km=0.0)
    startup = startup_for_console_elsewhere(tmp_path)
    monkeypatch.setenv("P2_DROP", mode)
    monkeypatch.setenv("P2_DROP_ONLY", only)
    scenario, document = parse_gmat_script(script, config)
    with pytest.raises(GmatRunError, match=r"was not created: orbitwin_diagnostic_burns\.txt"):
        verify_module.verify_candidate_with_gmat(
            script, scenario, document, candidate, config, console, startup, tmp_path / "v"
        )


def test_a_measuring_flight_that_died_between_maneuvers_still_serves(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    """Review V5: the measuring flight's pairs are checked as far as it flew, and the
    stop flight decides. `complete=True` there refused this."""
    config = ranking_quick_config()
    real = verify_module.run_gmat

    def first_flight_dies(console, sibling, workdir, names, **kwargs):
        if Path(workdir).name == "diagnostic-run":
            flown = parse_gmat_script(
                Path(sibling).parent / "verification_candidate.script", config
            )[0]
            first, second = (segment.duration_s for segment in flown.propagate_segments[:2])
            text = Path(sibling).read_text(encoding="utf-8")
            Path(sibling).write_text(
                f"% FAKE_GMAT_ABORT_AT_S {first + 0.5 * second!r}\n" + text, encoding="utf-8"
            )
        return real(console, sibling, workdir, names, **kwargs)

    monkeypatch.setattr(verify_module, "run_gmat", first_flight_dies)
    scenario, document = parse_gmat_script(workshop_script, config)
    candidate = capped_one_burn(scenario, config)
    verified = verify_module.verify_candidate_with_gmat(
        workshop_script, scenario, document, candidate, config, fake_gmat, fake_startup,
        tmp_path / "v",
    )[0]
    assert verified.successful
    assert (tmp_path / "v" / "verification_candidate_at_stop.script").exists()
    pairs = (tmp_path / "v" / "diagnostic-run" / "gmat-output" / "orbitwin_diagnostic_burns.txt")
    assert len(pairs.read_text().split("\n")) - 1 == 2, "one Maneuver flown, then the abort"


def test_an_aborted_flight_is_still_checked_for_the_impulse_flown(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    """Review V5: a flight that aborts is checked on its partial burn report."""
    _no_burn_guard(monkeypatch)
    config = ranking_quick_config()
    text = workshop_script.read_text(encoding="utf-8")
    script = tmp_path / "alias.script"
    script.write_text(
        "% FAKE_GMAT_FLYBY 6000.0 7\n% FAKE_GMAT_ABORT_AT_S 4000\n"
        + text.replace(ANCHOR, ALIASES + ANCHOR),
        encoding="utf-8",
    )
    scenario, document = parse_gmat_script(script, config)
    candidate = capped_one_burn(scenario, config)
    with pytest.raises(GmatRunError, match="Something in the script changes the impulse GMAT flies"):
        verify_module.verify_candidate_with_gmat(
            script, scenario, document, candidate, config, fake_gmat, fake_startup,
            tmp_path / "v",
        )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("ImpulsiveBurn2.Axes = VNB;", "ImpulsiveBurn2.Axes = MJ2000Eq;"),
        ("ImpulsiveBurn2.Origin = Earth;", "ImpulsiveBurn2.Origin = Luna;"),
        ("ImpulsiveBurn2.CoordinateSystem = Local;", "ImpulsiveBurn2.CoordinateSystem = EarthFixed;"),
    ],
    ids=["Local-MJ2000Eq", "Local-Luna", "EarthFixed"],
)
def test_an_unused_slot_in_a_frame_not_modelled_is_certified(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path,
    old: str, new: str,
) -> None:
    """Review V2: its claim is exactly zero, so the jump's length is checked, in no frame."""
    text = workshop_script.read_text(encoding="utf-8")
    assert text.count(old) == 1
    script, config, candidate = _case(workshop_script, tmp_path, "", text=text.replace(old, new))
    assert {burn.slot_index for burn in candidate.burns} == {0}
    certify_candidate(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1)


def test_an_alias_on_an_unused_slot_is_still_refused(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    """The zero-claim path still sees an impulse flown where none is claimed."""
    _no_burn_guard(monkeypatch)
    script, config, candidate = _case(
        workshop_script, tmp_path, "ImpulsiveBurn2.V = 0.01;\n"
    )
    scenario, document = parse_gmat_script(script, config)
    with pytest.raises(GmatRunError, match=r"GMAT flew Maneuver 2 \(ImpulsiveBurn2.* EarthMJ2000Eq"):
        verify_module.verify_candidate_with_gmat(
            script, scenario, document, candidate, config, fake_gmat, fake_startup,
            tmp_path / "v",
        )


def test_labelled_maneuvers_are_bracketed_and_checked(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Review V10: `Maneuver 'label' Burn(SC);`, which GMAT and the parser accept."""
    text = workshop_script.read_text(encoding="utf-8")
    for burn in ("ImpulsiveBurn1", "ImpulsiveBurn2"):
        assert text.count(f"Maneuver {burn}(") == 1
        text = text.replace(f"Maneuver {burn}(", f"Maneuver 'Do {burn}' {burn}(")
    script, config, candidate = _case(workshop_script, tmp_path, "", text=text)
    certify_candidate(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1)
    burns = list((tmp_path / "c").rglob("*_burns.txt"))
    assert burns and all(len(path.read_text().splitlines()) == 4 for path in burns)


def test_a_lone_burn_at_t_zero_after_an_alignment_is_checked(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Review V4: the fake crashed on a lone t = 0 burn (no drift, r x v = 0)."""
    text = no_maneuver_script.read_text(encoding="utf-8")
    early = "AlienSC.Epoch = '26 Jun 2026 05:00:00.000';"
    assert early in text
    script = tmp_path / "aligned.script"
    script.write_text(
        text.split(ANCHOR)[0].replace(early, "AlienSC.Epoch = '25 Jun 2026 14:00:00.000';")
        + "BeginMissionSequence;\n"
        "Propagate Prop(AlienSC) {AlienSC.A1ModJulian = Pursuer.A1ModJulian};\n"
        "Propagate Prop(Pursuer, AlienSC) {Pursuer.ElapsedSecs = 30000, AlienSC.ElapsedSecs = 30000};\n",
        encoding="utf-8",
    )
    config = OrbitWinConfig(
        policy=dataclasses.replace(
            PolicyConfig(), mission_time_origin="earliest", closest_approach_window="after_alignment",
            horizon_s=86400.0, rules="ranking", dv_limit_km_s=0.25, dv_safe_limit_km_s=0.249,
            total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99,
        ),
        pursuer_name="Pursuer", target_name="AlienSC",
    )
    scenario, document, _ = load_scenario(
        script, config, gmat_console=fake_gmat, startup_file=fake_startup
    )
    frame = ("Local", "Earth", "VNB")
    impulse = script_components_to_inertial(
        np.array([0.001, 0.0, 0.0]), scenario.pursuer.initial_state, *frame
    )
    candidate = Candidate(
        burns=(Burn(0.0, as_vector3(impulse), (0.001, 0.0, 0.0), 0, frame),),
        intercept_time_s=None, minimum_distance_km=None, distance_at_completion_km=None,
        successful=False, analytical_endpoint_error_km=None, branch="test",
    )
    verify_module.verify_candidate_with_gmat(
        script, scenario, document, candidate, config, fake_gmat, fake_startup, tmp_path / "v",
        timeout_s=120,
    )
    burns = list((tmp_path / "v").rglob("*_burns.txt"))
    assert burns and all(len(path.read_text().splitlines()) == 2 for path in burns)


def test_the_burn_report_is_asked_for_before_the_diagnostic_report(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    """run_gmat stops at the first report missing, so a run without its burn report
    keeps no diagnostic report either and no flight can read it."""
    script, config, candidate = _case(workshop_script, tmp_path, "")
    real = verify_module.run_gmat
    orders: list[tuple[str, list[str]]] = []

    def recording(console, sibling, workdir, names, **kwargs):
        orders.append((_site(workdir, tmp_path), list(names)))
        return real(console, sibling, workdir, names, **kwargs)

    monkeypatch.setattr(verify_module, "run_gmat", recording)
    certify_candidate(script, candidate, config, fake_gmat, fake_startup, tmp_path / "c", runs=1)
    siblings = [(site, names) for site, names in orders if config.gmat.diagnostic_report_name in names]
    assert {site for site, _ in siblings} == {"flight", "replay"}
    for _, names in siblings:
        burns = [index for index, name in enumerate(names) if name.endswith("_burns.txt")]
        assert burns and burns[0] < names.index(config.gmat.diagnostic_report_name)


@pytest.mark.parametrize(("end", "flight"), [("horizon", ""), ("interception", "-at-stop")])
def test_a_deciding_flight_that_failed_after_writing_is_refused(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch,
    end: str, flight: str,
) -> None:
    """Only a measuring flight may be read after a failure: GMAT exiting 1 once every
    report is written leaves them all as partial reports."""
    base = ranking_quick_config()
    config = dataclasses.replace(
        base, policy=dataclasses.replace(base.policy, mission_end_policy=end)
    )
    script, config, candidate = _case(workshop_script, tmp_path, "", config=config, miss_km=0.0)
    real = verify_module.run_gmat

    def failing_late(console, sibling, workdir, names, **kwargs):
        result = real(console, sibling, workdir, names, **kwargs)
        if Path(workdir).name == f"diagnostic-run{flight}":
            return dataclasses.replace(
                result, success=False, error="forced: exit 1 after the last row",
                reports={}, partial_reports=dict(result.reports),
            )
        return result

    monkeypatch.setattr(verify_module, "run_gmat", failing_late)
    scenario, document = parse_gmat_script(script, config)
    with pytest.raises(GmatRunError, match="forced: exit 1 after the last row"):
        verify_module.verify_candidate_with_gmat(
            script, scenario, document, candidate, config, fake_gmat, fake_startup, tmp_path / "v"
        )


def test_the_fake_writes_the_columns_a_report_command_asks_for(
    tmp_path: Path, workshop_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """A column change on one side shows up in fake runs, as it would on GMAT (review V6)."""
    from orbitwin.burn_rows import BURN_REPORT_QUANTITIES
    from orbitwin.diagnostic import BURN_RESOURCE_NAME, add_diagnostic_report
    from orbitwin.gmat_runner import run_gmat

    config = ranking_quick_config()
    scenario, document = parse_gmat_script(workshop_script, config)
    sibling = add_diagnostic_report(document, scenario, "diag.txt", check_burns=True)
    asked = " ".join(f"HumanSC.{quantity}" for quantity in BURN_REPORT_QUANTITIES)
    reversed_ = " ".join(f"HumanSC.{quantity}" for quantity in reversed(BURN_REPORT_QUANTITIES))
    text = sibling.script_bytes.decode("utf-8")
    assert text.count(f"Report {BURN_RESOURCE_NAME} {asked};") == 4
    script = tmp_path / "reversed.script"
    script.write_text(text.replace(asked, reversed_), encoding="utf-8")
    run = run_gmat(fake_gmat, script, tmp_path / "run", ["diag_burns.txt"],
                   startup_file=fake_startup, template_dir=workshop_script.parent)
    assert run.success, run.error
    rows = np.array([
        [float(value) for value in line.split()]
        for line in Path(run.reports["diag_burns.txt"]).read_text().splitlines()
    ])
    assert rows.shape == (4, len(BURN_REPORT_QUANTITIES))
    assert np.all(rows[:, -1] > 20000.0), "A1ModJulian, asked for last"
    assert np.all(np.abs(rows[:, 0]) < 20.0), "a velocity component, asked for first"
