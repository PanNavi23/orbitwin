"""Step 7 and the audit on head-start burns (COMPETITION_POLICY.md, 2026-11-08).

R8's pursuer flies the alignment alone; here it burns once in it, 900 s after its
epoch, and once 300 s after the alignment's end. The report is flown under point
mass and J2, the model check_report_rows flies rows back in, at R8's MaxStep 60 s.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from orbitwin.frames import script_components_to_inertial
from orbitwin.gmat_parser import mission_schedule
from orbitwin.models import StateVector

ROOT = Path(__file__).resolve().parents[2]
BIND = ROOT / "tests" / "fixtures" / "report_bind"
LEAD_S = 1800.0
FRAME = ("Local", "Earth", "VNB")
EARLY = (0.012, 0.004, -0.003)
LATER = (0.010, -0.02, 0.005)
HUMAN = (np.array([-4531.813000625929, -4915.913394421509, 1243.761257051035]),
         np.array([3.7890923476176033, -4.690335264612704, -4.723372003674307]))
ALIEN = (np.array([-18013.43061577812, 15570.96754366546, 16721.63060171485]),
         np.array([1.487841735461544, -4.684101321376503, -2.691639119511299]))


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = _load("check_report_rows")
AUDIT = _load("audit_burn_table")


def _accel(r: np.ndarray) -> np.ndarray:
    from orbitwin.pursuer_model import EGM96_J2, EGM96_RADIUS_KM

    mu = 398600.4418
    r2 = float(r @ r)
    zz = 5.0 * r[2] ** 2 / r2
    k = 1.5 * EGM96_J2 * mu * EGM96_RADIUS_KM**2 / r2**2.5
    return -mu * r / r2**1.5 + k * r * np.array([zz - 1.0, zz - 1.0, zz - 3.0])


def _coast(r: np.ndarray, v: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    steps = max(2, math.ceil(dt / 0.5))
    h = dt / steps
    for _ in range(steps):
        a1 = _accel(r)
        a2 = _accel(r + 0.5 * h * v)
        a3 = _accel(r + 0.5 * h * (v + 0.5 * h * a1))
        a4 = _accel(r + h * (v + 0.5 * h * a2))
        r, v = r + h * v + h * h / 6.0 * (a1 + a2 + a3), v + h / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
    return r, v


def _report(early_at: float | None, later_at: float, end_after: float,
            early: tuple[float, ...] = EARLY) -> tuple[list[str], np.ndarray, int]:
    """Rows as R2026a writes them: every 60 s from each Propagate's start, one at its end
    (the pre-burn state), none after a Maneuver. Returns (header, rows, frozen rows)."""
    segments: list[tuple[str, float, tuple[float, ...] | None]] = []
    if early_at is not None:
        segments.append(("alone", early_at, early))
    segments += [("alone", LEAD_S, None), ("joint", LEAD_S + later_at, LATER),
                 ("joint", LEAD_S + later_at + end_after, None)]
    human, alien = HUMAN, ALIEN
    now = 0.0
    rows = [[0.0, 0.0, *human[0], *human[1], *alien[0], *alien[1]]]
    for kind, end, burn in segments:
        while now < end - 1.0e-9:
            step = min(60.0, end - now)
            human = _coast(*human, step)
            if kind == "joint":
                alien = _coast(*alien, step)
            now += step
            rows.append([now, max(0.0, now - LEAD_S), *human[0], *human[1], *alien[0],
                         *alien[1]])
        if burn is not None:
            dv = script_components_to_inertial(
                np.array(burn), StateVector(tuple(human[0]), tuple(human[1])), *FRAME)
            human = (human[0], human[1] + dv)
    header = ["SC_Human.ElapsedSecs", "SC_Alien.ElapsedSecs",
              *(f"SC_Human.EarthMJ2000Eq.{a}" for a in ("X", "Y", "Z", "VX", "VY", "VZ")),
              *(f"SC_Alien.EarthMJ2000Eq.{a}" for a in ("X", "Y", "Z", "VX", "VY", "VZ"))]
    full = np.array(rows)
    return header, full, int(np.flatnonzero(full[:, 0] >= LEAD_S - 1.0e-9)[0])


def _script(tmp_path: Path, mission: str, early: tuple[float, ...] = EARLY) -> Path:
    text = (BIND / "R8_hit_step60.script").read_text(encoding="utf-8")
    head = text.split("BeginMissionSequence;")[0]
    for name, values in (("ImpulsiveBurn1", early), ("ImpulsiveBurn2", LATER)):
        for k, value in enumerate(values, 1):
            start = head.index(f"{name}.Element{k} = ")
            end = head.index(";", start)
            head = head[:start] + f"{name}.Element{k} = {value!r}" + head[end:]
    path = tmp_path / f"hs{len(list(tmp_path.glob('*.script')))}.script"
    path.write_text(head + "BeginMissionSequence;\n" + mission, encoding="utf-8")
    return path


HEAD_START = (
    "Propagate DefaultProp(SC_Human) {{SC_Human.ElapsedSecs = {early}}};\n"
    "Maneuver ImpulsiveBurn1(SC_Human);\n"
)
AFTER = (
    "Propagate DefaultProp(SC_Human) {{SC_Human.UTCModJulian = SC_Alien.UTCModJulian}};\n"
    "Propagate DefaultProp(SC_Alien, SC_Human) {{SC_Alien.ElapsedSecs = {later}, "
    "SC_Human.ElapsedSecs = {later}}};\n"
    "Maneuver ImpulsiveBurn2(SC_Human);\n"
    "Propagate DefaultProp(SC_Alien, SC_Human) {{SC_Alien.ElapsedSecs = {end}, "
    "SC_Human.ElapsedSecs = {end}}};\n"
)


def _mission(early: float | None, later: float, end: float) -> str:
    return (HEAD_START.format(early=early) if early is not None else "") + AFTER.format(
        later=later, end=end)


def _args(script: Path, **extra: object) -> argparse.Namespace:
    return argparse.Namespace(report="hs.txt", script=script, binding="enforce",
                              pursuer="SC_Human", accept_end_time_binding_only=False,
                              rules="ranking", recorded=None, total_dv=None, day_config=None,
                              penalty_points=0.0, certified_lead_s=None, **extra)


def _bind(script: Path, header: list[str], full: np.ndarray, frozen: int,
          args: argparse.Namespace | None = None) -> None:
    schedule = mission_schedule(script)
    args = args or _args(script)
    CHECK.bind_to_script(args, schedule, header, full, header.index("SC_Human.ElapsedSecs"),
                         ("SC_Alien", frozen), None)


def test_the_schedule_reads_the_head_start(tmp_path: Path) -> None:
    schedule = mission_schedule(_script(tmp_path, _mission(900, 300, 600)))
    assert schedule.alignment_craft == "SC_Human" and schedule.head_start_s == 900.0
    assert [(m.burn, m.time_s, m.head_start) for m in schedule.maneuvers] == [
        ("ImpulsiveBurn1", 900.0, True), ("ImpulsiveBurn2", 300.0, False)]
    # The end is the goals after the alignment; the head start is inside its length.
    assert schedule.end_s == 900.0
    # Σ|Δv| counts the head-start burn too.
    assert schedule.total_dv_km_s == pytest.approx(math.hypot(*EARLY) + math.hypot(*LATER))


def test_a_head_start_report_binds_to_its_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    header, full, frozen = _report(900, 300, 600)
    _bind(_script(tmp_path, _mission(900, 300, 600)), header, full, frozen)
    assert "hold its 2 burns and no other" in capsys.readouterr().out


def test_a_head_start_burn_of_another_size_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    header, full, frozen = _report(900, 300, 600)
    other = (0.012, 0.006, -0.003)  # 2 m/s off, over the 0.6 m/s a 60 s step allows
    with pytest.raises(SystemExit) as exc:
        _bind(_script(tmp_path, _mission(900, 300, 600), early=other), header, full, frozen)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "burn 1 (ImpulsiveBurn1, t = 900.000000" in err and "velocity jumps" in err, err


def test_a_burn_in_the_alignment_the_script_lacks_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before head-start burns nothing scanned the alignment's rows."""
    header, full, frozen = _report(900, 300, 600)
    script = _script(tmp_path, _mission(None, 300, 600))
    with pytest.raises(SystemExit) as exc:
        _bind(script, header, full, frozen)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "t = 900.000000 to 960.000000" in err and "has no burn" in err, err


def test_the_end_is_the_alignment_and_the_goals_after_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rows run to Δ + 900 s; adding the head start's 900 s to that is another flight."""
    script = _script(tmp_path, _mission(900, 300, 600))
    header, full, frozen = _report(900, 300, 600 + 900)
    with pytest.raises(SystemExit) as exc:
        _bind(script, header, full, frozen)
    assert exc.value.code == 2
    assert "ends at 3600.000000" in (err := capsys.readouterr().err) and (
        "end at 2700.000000 (1800.000000 + 900.000000" in err), err


def test_a_head_start_burn_on_the_waiting_crafts_clock_cannot_be_placed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    header, full, frozen = _report(900, 300, 600)
    keep = [i for i, name in enumerate(header) if name != "SC_Human.ElapsedSecs"]
    header, full = [header[i] for i in keep], full[:, keep]
    script = _script(tmp_path, _mission(900, 300, 600))
    with pytest.raises(SystemExit) as exc:
        CHECK.bind_to_script(_args(script), mission_schedule(script), header, full,
                             header.index("SC_Alien.ElapsedSecs"), ("SC_Alien", frozen), None)
    assert exc.value.code == 2
    assert "head-start burn(s) cannot be placed" in capsys.readouterr().err


def test_spacing_across_the_alignments_end_is_judged_on_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """1750 s into the head start and 30 s after it: 80 s apart."""
    script = _script(tmp_path, _mission(1750, 30, 600))
    args = _args(script)
    CHECK.judge_rules(args, mission_schedule(script))
    # Before the report nothing gives the lead: no pair across the end is judged.
    assert args.penalty_points == 0.0
    header, full, frozen = _report(1750, 30, 600)
    _bind(script, header, full, frozen, args)
    out = capsys.readouterr().out
    assert "is 80.000 s after the last head-start burn" in out and args.penalty_points == 10.0


def test_spacing_across_the_alignments_end_with_the_certified_lead(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _script(tmp_path, _mission(1750, 30, 600))
    args = _args(script)
    args.certified_lead_s = LEAD_S
    CHECK.judge_rules(args, mission_schedule(script))
    assert args.penalty_points == 10.0
    assert "burn 2 at t = 30.000 s is 80.000 s after the one before" in capsys.readouterr().out
    # The report charges it no twice.
    header, full, frozen = _report(1750, 30, 600)
    _bind(script, header, full, frozen, args)
    assert args.penalty_points == 10.0


def test_a_recorded_plan_with_head_start_times_is_judged_whole(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args(None)  # type: ignore[arg-type]
    args.recorded = ("certificate", {"burns": [
        {"time_s": -50.0, "dv_inertial_km_s": [0.01, 0.0, 0.0]},
        {"time_s": 30.0, "dv_inertial_km_s": [0.0, 0.2, 0.0]}]})
    CHECK.judge_rules(args, None)
    assert args.penalty_points == 10.0
    assert "burn 2 at t = 30.000 s is 80.000 s after" in capsys.readouterr().out


#: GMAT's aligned state less the two-body coast, as the parser records it.
DEFECT = StateVector((20.0, -5.0, 1.0), (0.02, 0.0, -0.01))


def _scenario(lead_s: float) -> SimpleNamespace:
    """What _replay reads of a scenario: an open template, R8's craft."""
    early = StateVector(tuple(HUMAN[0]), tuple(HUMAN[1]))
    alien = StateVector(tuple(ALIEN[0]), tuple(ALIEN[1]))
    return SimpleNamespace(
        burn_slots=(), head_start_s=lead_s, head_start_state=early, head_start_note=None,
        head_start_defect=DEFECT,
        pursuer=SimpleNamespace(initial_state=StateVector((7000.0, 0.0, 0.0), (0.0, 7.5, 0.0))),
        target=SimpleNamespace(initial_state=alien))


def test_the_audit_replays_a_head_start_from_the_probed_state() -> None:
    mu = 398600.4418
    burns = [(-900.0, np.array([0.01, 0.0, 0.0])), (300.0, np.array([0.0, 0.02, 0.0]))]
    times, radii, distances, alignment_km = AUDIT._replay(_scenario(LEAD_S), burns, mu, 600.0,
                                                          10.0)
    assert times[0] == -LEAD_S and np.isinf(distances[times < 0.0]).all()
    # The radius wall covers the head start.
    assert radii.size == times.size and math.isfinite(alignment_km)
    start = StateVector(tuple(HUMAN[0]), tuple(HUMAN[1]))
    from orbitwin.orbit import propagate_universal

    at_zero = propagate_universal(
        StateVector(tuple(propagate_universal(start, 900.0, mu).r),
                    tuple(propagate_universal(start, 900.0, mu).v + burns[0][1])), 900.0, mu)
    # The alignment's end takes on the defect: GMAT's aligned state plus the burns' effect.
    at_zero = StateVector(tuple(at_zero.r + DEFECT.r), tuple(at_zero.v + DEFECT.v))
    zero = int(np.flatnonzero(times >= 0.0)[0])
    assert distances[zero] == pytest.approx(
        float(np.linalg.norm(at_zero.r - ALIEN[0])), rel=1.0e-9)


def test_the_audit_refuses_a_head_start_where_none_is_open() -> None:
    from orbitwin.errors import DynamicsError

    with pytest.raises(DynamicsError, match=r"needs policy\.head_start_burns"):
        AUDIT._replay(_scenario(0.0), [(-900.0, np.zeros(3))], 398600.4418, 600.0, 10.0)


def test_step7_refuses_a_head_start_burn_the_days_config_turned_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from orbitwin.config import load_config

    on = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    off = type(on)(**{**{f: getattr(on, f) for f in on.__dataclass_fields__},
                      "policy": type(on.policy)(**{**{f: getattr(on.policy, f) for f in
                                                      on.policy.__dataclass_fields__},
                                                   "head_start_burns": False})})
    script = _script(tmp_path, _mission(900, 300, 600))
    for given, recorded in ((script, None), (None, ("certificate", {"burns": [
            {"time_s": -900.0, "dv_inertial_km_s": [0.01, 0.0, 0.0]}]}))):
        args = _args(given, config="day.yaml")  # type: ignore[arg-type]
        args.day_config, args.recorded = off, recorded
        with pytest.raises(SystemExit) as stop:
            CHECK.judge_rules(args, mission_schedule(script) if given else None)
        assert stop.value.code == 2
        assert "turns policy.head_start_burns off" in capsys.readouterr().err
    # On, or without a head-start burn, it judges as before.
    args = _args(script, config="day.yaml")
    args.day_config = on
    CHECK.judge_rules(args, mission_schedule(script))
    (tmp_path / "plain").mkdir()
    plain = _script(tmp_path / "plain", _mission(None, 300, 600))
    args = _args(plain, config="day.yaml")
    args.day_config = off
    CHECK.judge_rules(args, mission_schedule(plain))
