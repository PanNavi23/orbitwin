"""The impulse GMAT flew against the one the script claims (E2E audit A4).

The diagnostic sibling reports the pursuer on either side of each Maneuver;
verify takes the velocity jump between the two, in the burn's frame on the
pre-burn state, and refuses a run where it is not the script's Element1-3.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
from conftest import quick_config

from orbitwin.burn_rows import (
    BURN_REPORT_QUANTITIES,
    FLOWN_IMPULSE_TOLERANCE_KM_S,
    FlownManeuver,
    check_flown_impulses,
    maneuvers_flown,
)
from orbitwin.diagnostic import BURN_RESOURCE_NAME, add_diagnostic_report
from orbitwin.errors import GmatRunError, PolicyError
from orbitwin.frames import script_components_to_inertial
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import StateVector
from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, propagate_universal
from orbitwin.rules import RANKING

MU = 398600.4418
VNB = ("Local", "Earth", "VNB")
LVLH = ("Local", "Earth", "LVLH")
INERTIAL = ("EarthMJ2000Eq", "Earth", "MJ2000Eq")
BURNS = (
    FlownManeuver(240.0, (0.2, -0.05, 0.03), VNB, "B1"),
    FlownManeuver(400.0, (0.0, 0.0, 0.0), LVLH, "B2"),
    FlownManeuver(600.0, (-0.01, 0.1, 0.12), INERTIAL, "B3"),
)


def _report(
    path: Path,
    burns: tuple[FlownManeuver, ...] = BURNS,
    *,
    flown: dict[int, tuple[float, float, float]] | None = None,
    pairs: int | None = None,
) -> Path:
    """Rows as the sibling's Report commands write them; `flown` overrides a burn's components."""
    state = keplerian_to_cartesian(KeplerianElements(7000.0, 0.01, 28.0, 15.0, 30.0, 0.0), MU)
    now_s, lines = 0.0, []
    for index, burn in enumerate(burns[:pairs]):
        state = propagate_universal(state, burn.time_s - now_s, MU)
        now_s = burn.time_s
        components = np.asarray((flown or {}).get(index, burn.components_km_s), dtype=float)
        after = StateVector(state.r_km, tuple(state.v + script_components_to_inertial(
            components, state, *burn.frame)))
        a1 = 30000.0 + burn.time_s / 86400.0
        for row in (state, after):
            lines.append(" ".join(f"{value:.17g}" for value in (a1, *row.r, *row.v)))
        state = after
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_impulses_the_script_claims_pass(tmp_path: Path) -> None:
    check_flown_impulses(_report(tmp_path / "b.txt"), BURNS, complete=True)


@pytest.mark.parametrize("index", range(len(BURNS)))
def test_another_impulse_flown_is_refused_in_every_frame(tmp_path: Path, index: int) -> None:
    """A4's B17: GMAT flew Element1 = 0 for a claimed 0.0575 km/s."""
    claimed = np.asarray(BURNS[index].components_km_s)
    flown = tuple(claimed + np.array([0.0, 0.0, 1.0e-3]))
    path = _report(tmp_path / "b.txt", flown={index: flown})
    with pytest.raises(GmatRunError, match=rf"GMAT flew Maneuver {index + 1} \({BURNS[index].name}, "
                       r"t = .* Something in the script changes the impulse GMAT flies"):
        check_flown_impulses(path, BURNS, complete=True)


def test_the_tolerance_sits_between_rounding_and_the_rules_epsilon(tmp_path: Path) -> None:
    assert 1.0e-14 < FLOWN_IMPULSE_TOLERANCE_KM_S < RANKING.impulse_epsilon_km_s / 10.0
    claimed = np.asarray(BURNS[0].components_km_s)
    near = tuple(claimed + np.array([3.0e-11, 0.0, 0.0]))
    check_flown_impulses(_report(tmp_path / "a.txt", flown={0: near}), BURNS, complete=True)
    far = tuple(claimed + np.array([3.0e-10, 0.0, 0.0]))
    with pytest.raises(GmatRunError, match="apart, tolerance 1e-10"):
        check_flown_impulses(_report(tmp_path / "b.txt", flown={0: far}), BURNS, complete=True)


def test_a_complete_run_must_report_every_maneuver(tmp_path: Path) -> None:
    path = _report(tmp_path / "b.txt", pairs=2)
    with pytest.raises(GmatRunError, match="has 4 rows for 3 Maneuvers"):
        check_flown_impulses(path, BURNS, complete=True)
    with pytest.raises(GmatRunError, match="wrote no burn report"):
        check_flown_impulses(None, BURNS, complete=True)


def test_a_run_that_stopped_is_checked_on_what_it_flew(tmp_path: Path) -> None:
    check_flown_impulses(_report(tmp_path / "a.txt", pairs=2), BURNS, complete=False)
    check_flown_impulses(None, BURNS, complete=False)
    bad = _report(tmp_path / "b.txt", pairs=1, flown={0: (0.0, -0.05, 0.03)})
    with pytest.raises(GmatRunError, match="GMAT flew Maneuver 1"):
        check_flown_impulses(bad, BURNS, complete=False)


def test_extra_rows_are_refused(tmp_path: Path) -> None:
    path = _report(tmp_path / "b.txt")
    path.write_text(path.read_text() * 2, encoding="utf-8")
    with pytest.raises(GmatRunError, match="has 12 rows for 3 Maneuvers"):
        check_flown_impulses(path, BURNS, complete=False)


@pytest.mark.parametrize("column", [0, 2])
def test_rows_that_are_not_one_instant_and_place_are_refused(tmp_path: Path, column: int) -> None:
    path = _report(tmp_path / "b.txt")
    lines = path.read_text().splitlines()
    values = lines[1].split()
    values[column] = repr(float(values[column]) + 1.0e-3)
    lines[1] = " ".join(values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(GmatRunError, match="are not one instant and place"):
        check_flown_impulses(path, BURNS, complete=True)


def _sibling(workshop: Path, **kwargs: bool) -> tuple[str, str | None]:
    scenario, document = parse_gmat_script(workshop, quick_config())
    diagnostic = add_diagnostic_report(document, scenario, "diag.txt", **kwargs)
    return diagnostic.script_bytes.decode("utf-8"), diagnostic.burn_report_filename


def test_the_sibling_reports_the_pursuer_around_each_maneuver(workshop_script: Path) -> None:
    text, burn_report = _sibling(workshop_script, check_burns=True)
    assert burn_report == "diag_burns.txt"
    assert f"{BURN_RESOURCE_NAME}.Filename = 'diag_burns.txt';" in text
    report = (
        f"Report {BURN_RESOURCE_NAME} HumanSC.A1ModJulian HumanSC.EarthMJ2000Eq.X "
        "HumanSC.EarthMJ2000Eq.Y HumanSC.EarthMJ2000Eq.Z HumanSC.EarthMJ2000Eq.VX "
        "HumanSC.EarthMJ2000Eq.VY HumanSC.EarthMJ2000Eq.VZ;"
    )
    lines = text.splitlines()
    maneuvers = [index for index, line in enumerate(lines) if re.match(r"Maneuver ", line)]
    assert len(maneuvers) == 2
    for index in maneuvers:
        assert lines[index - 1] == report and lines[index + 1] == report
    assert text.count(f"Report {BURN_RESOURCE_NAME} ") == 4
    # Everything else is the plain sibling's.
    plain, none = _sibling(workshop_script)
    assert none is None
    assert [line for line in lines if BURN_RESOURCE_NAME not in line] == plain.splitlines()


def test_the_sibling_pairs_with_the_scripts_maneuvers(workshop_script: Path) -> None:
    scenario, _ = parse_gmat_script(workshop_script, quick_config())
    assert [maneuver.name for maneuver in maneuvers_flown(scenario)] == [
        "ImpulsiveBurn1", "ImpulsiveBurn2"
    ]


def test_nothing_maneuvered_needs_no_burn_report(no_maneuver_script: Path) -> None:
    config = dataclasses.replace(quick_config(), pursuer_name="Pursuer")
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    diagnostic = add_diagnostic_report(document, scenario, "diag.txt", check_burns=True)
    assert diagnostic.burn_report_filename is None
    assert BURN_RESOURCE_NAME not in diagnostic.script_bytes.decode("utf-8")


def test_the_burn_resource_name_is_reserved(workshop_script: Path, tmp_path: Path) -> None:
    path = tmp_path / "taken.script"
    path.write_text(
        workshop_script.read_text(encoding="utf-8").replace(
            "BeginMissionSequence;", f"Create Variable {BURN_RESOURCE_NAME};\nBeginMissionSequence;", 1
        ),
        encoding="utf-8",
    )
    scenario, document = parse_gmat_script(path, quick_config())
    with pytest.raises(PolicyError, match=f"reserved resource name {BURN_RESOURCE_NAME}"):
        add_diagnostic_report(document, scenario, "diag.txt", check_burns=True)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "burn_alias"


def test_zero_claims_in_frames_not_modelled_pass_on_gmats_own_rows() -> None:
    """Review V2, R2026a's burn report: two unused slots, EarthFixed and Local/Earth/MJ2000Eq,
    which frames.py cannot build. Their jump is checked by length, in no frame.

    The report is the burn report of add_diagnostic_report(check_burns=True) on
    examples/two_burn_rendezvous.script with both Elements zero, ImpulsiveBurn1 in
    EarthFixed, ImpulsiveBurn2 on MJ2000Eq axes and Propagates of 0.001, 101 and 1000 s."""
    maneuvers = (
        FlownManeuver(0.001, (0.0, 0.0, 0.0), ("EarthFixed", "Earth", "VNB"), "ImpulsiveBurn1"),
        FlownManeuver(101.001, (0.0, 0.0, 0.0), ("Local", "Earth", "MJ2000Eq"), "ImpulsiveBurn2"),
    )
    check_flown_impulses(
        FIXTURES / "r2026a_zero_slots_unmodelled_frames_burns.txt", maneuvers, complete=True
    )


def test_a_jump_on_a_zero_claim_is_refused_in_any_frame(tmp_path: Path) -> None:
    unmodelled = FlownManeuver(240.0, (0.0, 0.0, 0.0), ("EarthFixed", "Earth", "VNB"), "Spare")
    path = _report(tmp_path / "b.txt", (dataclasses.replace(unmodelled, frame=INERTIAL),),
                   flown={0: (0.0, 2.0e-10, 0.0)})
    with pytest.raises(GmatRunError, match=r"GMAT flew Maneuver 1 \(Spare.* in EarthMJ2000Eq"):
        check_flown_impulses(path, (unmodelled,), complete=True)


def test_the_rows_are_read_as_the_sibling_asks_for_them(workshop_script: Path) -> None:
    text, _ = _sibling(workshop_script, check_burns=True)
    request = next(line for line in text.splitlines() if line.startswith("Report "))
    assert request.rstrip(";").split()[2:] == [f"HumanSC.{q}" for q in BURN_REPORT_QUANTITIES]


def test_the_sibling_brackets_only_the_maneuver_it_is_told_of(
    workshop_script: Path, monkeypatch
) -> None:
    """Review V7/V9: a Maneuver line that is not the one maneuvers_flown names is refused."""
    from orbitwin import diagnostic

    scenario, document = parse_gmat_script(workshop_script, quick_config())
    real = diagnostic.maneuvers_flown

    def swapped(scenario):
        flown = real(scenario)
        return tuple(
            dataclasses.replace(m, name=other.name)
            for m, other in zip(flown, reversed(flown), strict=True)
        )

    monkeypatch.setattr(diagnostic, "maneuvers_flown", swapped)
    # Bracketed from the last: ImpulsiveBurn2's line now claims ImpulsiveBurn1.
    with pytest.raises(PolicyError, match="has no `Maneuver ImpulsiveBurn1` where"):
        add_diagnostic_report(document, scenario, "diag.txt", check_burns=True)


def test_a_zero_slot_is_bracketed_like_any_other(workshop_script: Path, tmp_path: Path) -> None:
    """Every Maneuver the check pairs rows with gets its two Report commands."""
    text = workshop_script.read_text(encoding="utf-8")
    assert text.count("ImpulsiveBurn2.Element1 = 0.5507;") == 1
    path = tmp_path / "zero.script"
    path.write_text(text.replace("ImpulsiveBurn2.Element1 = 0.5507;", "ImpulsiveBurn2.Element1 = 0;"))
    scenario, document = parse_gmat_script(path, quick_config())
    assert not np.any(maneuvers_flown(scenario)[1].components_km_s)
    sibling = add_diagnostic_report(document, scenario, "diag.txt", check_burns=True)
    text = sibling.script_bytes.decode("utf-8")
    assert text.count(f"Report {BURN_RESOURCE_NAME} ") == 2 * len(maneuvers_flown(scenario))
