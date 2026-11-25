"""The GMAT-residual corrector, flown against a stand-in for GmatConsole.

The stand-in replaces run_gmat. It parses the diagnostic sibling the corrector
rendered, flies it two-body with its own mu and, when asked, a thruster that
scales or turns every impulse, and writes the report GMAT would. Being wrong
in known ways is what lets a test say whether the corrector found the answer.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orbitwin import gmat_shoot
from orbitwin.burn_rows import FlownManeuver, maneuvers_flown
from orbitwin.config import OrbitWinConfig, ScoreConfig
from orbitwin.errors import DynamicsError, ParseError, PolicyError
from orbitwin.frames import script_components_to_inertial
from orbitwin.gmat_parser import GmatScriptDocument, parse_gmat_script
from orbitwin.gmat_shoot import correct_with_gmat
from orbitwin.models import (
    Burn,
    Candidate,
    GmatRunResult,
    Scenario,
    StateVector,
    as_vector3,
    burn_frame,
)
from orbitwin.multi_burn import as_candidate
from orbitwin.orbit import KeplerianElements, keplerian_to_cartesian, propagate_universal
from orbitwin.score import score_candidate

MU = 398600.4418
STEP_S = 10.0
HUMAN = KeplerianElements(7000.0, 0.0, 28.0, 15.0, 0.0, 300.0)
VNB = ("Local", "Earth", "VNB")
INERTIAL = ("EarthMJ2000Eq", "Earth", "MJ2000Eq")


@dataclasses.dataclass
class Model:
    """How the stand-in differs from the two-body model the corrector plans in."""

    pursuer_mu_scale: float = 1.0
    target_mu_scale: float = 1.0
    impulse_scale: float = 1.0
    #: Applied to every impulse after the frame conversion.
    rotation: np.ndarray | None = None
    #: Write the post-burn row R2026a leaves out. Only with the script's own
    #: impulse: burn_rows takes a second row as post-burn only when its jump is
    #: that impulse, so a scaled or turned one would be discarded.
    post_burn_rows: bool = False
    target_offset_km: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Put one pursuer row at dip_radius_km, inside the Earth by default, at this epoch.
    dip_at_s: float | None = None
    dip_radius_km: float = 6300.0
    #: Put the target 4 km from the pursuer at one row at this epoch.
    graze_at_s: float | None = None
    #: How far from the pursuer the target is put at that row.
    graze_km: float = 4.0
    #: From this call on, fail as GmatConsole does when RK89 gives up.
    abort_from_call: int | None = None
    #: Succeed, but report nothing after this epoch.
    report_until_s: float | None = None
    #: Seconds each call holds, so concurrent calls overlap.
    hold_s: float = 0.0


def _rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )


def _applied(
    model: Model, components: np.ndarray, frame: tuple[str, str, str], state: StateVector
) -> np.ndarray:
    impulse = model.impulse_scale * script_components_to_inertial(components, state, *frame)
    return impulse if model.rotation is None else model.rotation @ impulse


def _rows(
    model: Model,
    pursuer: StateVector,
    target: StateVector,
    burns: list[tuple[float, np.ndarray, tuple[str, str, str]]],
    end_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GMAT-shaped rows: every STEP_S from each segment start, and each segment end.

    Each burn is (epoch, script components, frame), applied as GMAT would: in
    its frame on the state at the burn.
    """
    mu_p, mu_t = MU * model.pursuer_mu_scale, MU * model.target_mu_scale
    edges = [burn[0] for burn in burns] + [end_s]
    times: list[float] = [0.0]
    states: list[StateVector] = [pursuer]
    start_s, start = 0.0, pursuer
    for index, edge in enumerate(edges):
        offsets = [*np.arange(STEP_S, edge - start_s - 1.0e-9, STEP_S), edge - start_s]
        for offset in offsets:
            times.append(start_s + float(offset))
            states.append(propagate_universal(start, float(offset), mu_p))
        if index < len(burns):
            pre = states[-1]
            _, components, frame = burns[index]
            start = StateVector(
                pre.r_km, as_vector3(pre.v + _applied(model, components, frame, pre))
            )
            start_s = edge
            if model.post_burn_rows:
                times.append(edge)
                states.append(start)
    t = np.asarray(times)
    rp = np.array([state.r for state in states])
    vp = np.array([state.v for state in states])
    if model.dip_at_s is not None:
        row = int(np.argmin(np.abs(t - model.dip_at_s)))
        rp[row] *= model.dip_radius_km / float(np.linalg.norm(rp[row]))
    target_states = [propagate_universal(target, float(time_s), mu_t) for time_s in t]
    rt = np.array([state.r for state in target_states]) + np.asarray(model.target_offset_km)
    vt = np.array([state.v for state in target_states])
    if model.graze_at_s is not None:
        row = int(np.argmin(np.abs(t - model.graze_at_s)))
        rt[row], vt[row] = rp[row] + np.array([model.graze_km, 0.0, 0.0]), vp[row]
    return t, rp, vp, rt, vt


class FakeGmat:
    """Stands in for run_gmat; counts its calls."""

    def __init__(self, config: OrbitWinConfig, model: Model) -> None:
        self.config = config
        self.model = model
        self.calls = 0
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __call__(
        self,
        gmat_console: str | Path,
        script: str | Path,
        workdir: str | Path,
        report_filenames: list[str],
        *,
        startup_file: str | Path | None = None,
        timeout_s: float = 300.0,
        file_size_limit_bytes: int | None = None,
        on_started: Callable[[], None] | None = None,
        template_dir: str | Path | None = None,
    ) -> GmatRunResult:
        if on_started is not None:
            on_started()
        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
            call, run_id = self.calls, f"fake-{self.calls:03d}"
        try:
            time.sleep(self.model.hold_s)
            if self.model.abort_from_call is not None and call >= self.model.abort_from_call:
                return dataclasses.replace(
                    self._report(script, workdir, run_id), success=False, reports={},
                    error="GmatConsole returned exit status 1",
                )
            return self._report(script, workdir, run_id)
        finally:
            with self._lock:
                self.active -= 1

    def _report(self, script: str | Path, workdir: str | Path, run_id: str) -> GmatRunResult:
        scenario, _ = parse_gmat_script(script, self.config)
        # Each burn in the frame its ImpulsiveBurn declares, as GMAT reads it.
        burns = [
            (maneuver.time_s, np.asarray(maneuver.components_km_s), maneuver.frame)
            for maneuver in maneuvers_flown(scenario)
        ]
        elapsed = sum(segment.duration_s for segment in scenario.propagate_segments)
        t, rp, vp, rt, vt = _rows(
            self.model, scenario.pursuer.initial_state, scenario.target.initial_state,
            burns, elapsed,
        )
        if self.model.report_until_s is not None:
            kept = t <= self.model.report_until_s
            t, rp, vp, rt, vt = t[kept], rp[kept], vp[kept], rt[kept], vt[kept]
        report = next(r for r in scenario.report_files if r.name == "OrbitWinDiagnostic")
        p, q = scenario.pursuer.name, scenario.target.name
        columns = {f"{p}.ElapsedSecs": t, f"{q}.ElapsedSecs": t}
        columns[f"{p}.A1ModJulian"] = columns[f"{q}.A1ModJulian"] = 21545.0 + t / 86400.0
        # Geodetic is never under the sphere's figure; equal on the equator.
        columns[f"{p}.Earth.Altitude"] = np.linalg.norm(rp, axis=1) - 6378.1363
        for craft, r, v in ((p, rp, vp), (q, rt, vt)):
            for axis in range(3):
                columns[f"{craft}.EarthMJ2000Eq.{'XYZ'[axis]}"] = r[:, axis]
                columns[f"{craft}.EarthMJ2000Eq.V{'XYZ'[axis]}"] = v[:, axis]
        output = Path(workdir) / "gmat-output"
        output.mkdir(parents=True, exist_ok=True)
        path = output / report.filename
        lines = [" ".join(report.parameters)]
        lines += [
            " ".join(repr(float(columns[name][row])) for name in report.parameters)
            for row in range(len(t))
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return GmatRunResult(
            run_id=run_id, success=True, return_code=0, started_at_utc="", duration_s=0.01,
            workdir=str(workdir), script_path=str(script), script_sha256="",
            executable_sha256="", startup_sha256=None, stdout_path="", stderr_path="",
            log_path=None, reports={report.filename: str(path)}, report_sha256={},
        )


def _config(horizon_s: float, **policy: object) -> OrbitWinConfig:
    base = OrbitWinConfig()
    settings: dict[str, object] = {
        "rules": "ranking", "horizon_s": horizon_s, "dv_limit_km_s": 0.25,
        "dv_safe_limit_km_s": 0.249, "total_dv_limit_km_s": 3.0,
        "total_dv_safe_limit_km_s": 2.95,
    }
    settings.update(policy)
    return dataclasses.replace(
        base, policy=dataclasses.replace(base.policy, **settings),
        pursuer_name="Pursuer", target_name="AlienSC",
    )


def _schedule(
    pursuer: StateVector, vnb_burns: list[tuple[float, tuple[float, float, float]]]
) -> list[tuple[float, np.ndarray]]:
    """Inertial impulses for VNB components on the two-body flown state."""
    state, now_s = pursuer, 0.0
    schedule = []
    for time_s, components in vnb_burns:
        state = propagate_universal(state, time_s - now_s, MU)
        now_s = time_s
        impulse = script_components_to_inertial(np.asarray(components, dtype=float), state, *VNB)
        schedule.append((time_s, impulse))
        state = StateVector(state.r_km, as_vector3(state.v + impulse))
    return schedule


def _scenario(
    tmp_path: Path,
    vnb_burns: list[tuple[float, tuple[float, float, float]]],
    arrival_s: float,
    *,
    placed_by: Model | None = None,
    offset_km: tuple[float, float, float] = (0.0, 0.0, 0.0),
    declared_frame: tuple[str, str, str] | None = None,
    **policy: object,
) -> tuple[OrbitWinConfig, Scenario, GmatScriptDocument, Candidate]:
    """An open template whose target sits where the plan arrives.

    Two-body by default, so the plan hits exactly; with `placed_by`, where that
    model flies the plan, plus `offset_km`. The plan is given in VNB. With
    `declared_frame`, the template declares one unmaneuvered ImpulsiveBurn per
    burn in that frame, so the candidate's burns are written in it.
    """
    model = placed_by or Model()
    pursuer = keplerian_to_cartesian(HUMAN, MU)
    flown = _rows(
        model, pursuer, pursuer,
        [(time_s, np.asarray(components), VNB) for time_s, components in vnb_burns], arrival_s,
    )
    arrival = flown[1][-1] + np.asarray(offset_km)
    radial = arrival / np.linalg.norm(arrival)
    normal = np.cross(radial, [0.0, 0.0, 1.0]) + np.array([0.0, 0.0, 0.5])
    normal -= radial * float(normal @ radial)
    normal /= np.linalg.norm(normal)
    speed = 1.25 * math.sqrt(MU / float(np.linalg.norm(arrival)))
    target = propagate_universal(
        StateVector(as_vector3(arrival), as_vector3(speed * np.cross(normal, radial))),
        -arrival_s, MU * model.target_mu_scale,
    )
    horizon_s = arrival_s + 600.0
    text = "\n".join([
        "Create Spacecraft Pursuer;",
        "Pursuer.DateFormat = UTCGregorian;",
        "Pursuer.Epoch = '05 Dec 2026 00:00:00.000';",
        "Pursuer.CoordinateSystem = EarthMJ2000Eq;",
        "Pursuer.DisplayStateType = Keplerian;",
        f"Pursuer.SMA = {HUMAN.sma_km!r};",
        f"Pursuer.ECC = {HUMAN.ecc!r};",
        f"Pursuer.INC = {HUMAN.inc_deg!r};",
        f"Pursuer.RAAN = {HUMAN.raan_deg!r};",
        f"Pursuer.AOP = {HUMAN.aop_deg!r};",
        f"Pursuer.TA = {HUMAN.ta_deg!r};",
        "",
        "Create Spacecraft AlienSC;",
        "AlienSC.DateFormat = UTCGregorian;",
        "AlienSC.Epoch = '05 Dec 2026 00:00:00.000';",
        "AlienSC.CoordinateSystem = EarthMJ2000Eq;",
        "AlienSC.DisplayStateType = Cartesian;",
        *(f"AlienSC.{axis} = {float(value)!r};" for axis, value in zip("XYZ", target.r, strict=True)),
        *(f"AlienSC.V{axis} = {float(value)!r};" for axis, value in zip("XYZ", target.v, strict=True)),
        "",
        *_declared_burns(len(vnb_burns), declared_frame),
        "Create ForceModel FM;",
        "FM.CentralBody = Earth;",
        "FM.PrimaryBodies = {Earth};",
        "FM.Drag = None;",
        "FM.SRP = Off;",
        "FM.GravityField.Earth.Degree = 0;",
        "FM.GravityField.Earth.Order = 0;",
        "",
        "Create Propagator Prop;",
        "Prop.FM = FM;",
        "Prop.Type = RungeKutta89;",
        f"Prop.MaxStep = {STEP_S!r};",
        "",
        "BeginMissionSequence;",
        f"Propagate Prop(Pursuer, AlienSC) {{Pursuer.ElapsedSecs = {horizon_s!r}, "
        f"AlienSC.ElapsedSecs = {horizon_s!r}}};",
        "",
    ])
    path = tmp_path / "official.script"
    path.write_text(text, encoding="ascii")
    config = _config(horizon_s, **policy)
    scenario, document = parse_gmat_script(path, config)
    schedule = _schedule(pursuer, vnb_burns)
    return config, scenario, document, as_candidate(schedule, arrival_s, scenario, config)


Corrector = Callable[..., gmat_shoot.CorrectionResult]


def _declared_burns(count: int, frame: tuple[str, str, str] | None) -> list[str]:
    if frame is None:
        return []
    lines = []
    for index in range(1, count + 1):
        name = f"DeclaredBurn{index}"
        lines += [
            f"Create ImpulsiveBurn {name};",
            f"{name}.CoordinateSystem = {frame[0]};",
            f"{name}.Origin = {frame[1]};",
            f"{name}.Axes = {frame[2]};",
            f"{name}.Element1 = 0;",
            f"{name}.Element2 = 0;",
            f"{name}.Element3 = 0;",
            f"{name}.DecrementMass = false;",
            "",
        ]
    return lines


def _corrector(
    monkeypatch: pytest.MonkeyPatch, config: OrbitWinConfig, model: Model
) -> tuple[FakeGmat, Corrector]:
    fake = FakeGmat(config, model)
    monkeypatch.setattr(gmat_shoot, "run_gmat", fake)

    def run(scenario, document, candidate, workdir: Path, **options):
        return correct_with_gmat(
            document, scenario, candidate, config, "GmatConsole", "startup.txt", workdir,
            **options,
        )

    return fake, run


TWO_BURNS = [(300.0, (0.12, 0.05, 0.0)), (900.0, (0.08, -0.06, 0.02))]


def _assert_not_a_hit(result: gmat_shoot.CorrectionResult) -> None:
    """An unconverged result offers nothing that ranks or reads as a hit."""
    assert result.candidate is None
    flight = result.closest_flight
    if flight is not None:
        assert not flight.successful and flight.score is None
        assert flight.intercept_time_s is None and flight.distance_at_completion_km is None
        assert flight.metadata["gmat_correction"]["converged"] is False
        assert flight.metadata["event_refined"] is False
        assert flight.metadata["frontier"] == "robust"
        assert flight.stage.value == "analytical"


def _assert_frames_agree(candidate: Candidate) -> None:
    for burn in candidate.burns:
        assert abs(
            float(np.linalg.norm(burn.dv_script_km_s))
            - float(np.linalg.norm(burn.dv_inertial_km_s))
        ) <= 1.0e-9


def test_a_near_hit_becomes_a_hit_in_two_rounds(tmp_path: Path, monkeypatch) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    original = candidate.to_dict()
    fake, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    result = run(scenario, document, candidate, tmp_path / "shoot")

    assert result.converged and result.reason is None and result.closest_flight is None
    assert result.rounds == 2 and result.gmat_runs == fake.calls == 2
    first, second = result.log
    assert first.miss_at_arrival_km > 5.0
    assert second.closest_approach_km < 0.5 < first.closest_approach_km
    assert (first.mode, first.sensitivity) == ("last", "two-body")
    corrected = result.candidate
    assert corrected is not None and corrected.successful
    assert corrected.minimum_distance_km == result.closest_approach_km
    assert corrected.intercept_time_s == result.first_entry_time_s
    assert corrected.burns[0] is candidate.burns[0]
    assert corrected.burns[1].dv_inertial_km_s != candidate.burns[1].dv_inertial_km_s
    assert corrected.candidate_id != candidate.candidate_id
    assert corrected.metadata["corrected_from"] == candidate.candidate_id
    assert corrected.stage.value == "analytical"
    _assert_frames_agree(corrected)
    assert candidate.to_dict() == original
    assert (tmp_path / "shoot" / "correction.json").is_file()


def test_the_corrected_burn_is_written_on_the_two_body_flown_state(
    tmp_path: Path, monkeypatch
) -> None:
    """The convention as_candidate writes, so verification renders the same vector."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    corrected = run(scenario, document, candidate, tmp_path / "shoot").candidate
    assert corrected is not None
    rebuilt = as_candidate(
        [(burn.time_s, np.asarray(burn.dv_inertial_km_s)) for burn in corrected.burns],
        2400.0, scenario, config,
    )
    for got, want in zip(corrected.burns, rebuilt.burns, strict=True):
        assert np.allclose(got.dv_script_km_s, want.dv_script_km_s, rtol=0.0, atol=1.0e-12)


def test_a_hit_on_the_first_flight_is_returned_unchanged(tmp_path: Path, monkeypatch) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    _, run = _corrector(monkeypatch, config, Model())
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert result.converged and result.rounds == 1
    assert result.candidate is not None
    assert result.candidate.burns == candidate.burns
    assert result.closest_approach_km < 1.0e-3


def test_a_corrected_candidate_is_corrected_again_at_the_same_arrival(
    tmp_path: Path, monkeypatch
) -> None:
    """Its intercept_time_s is GMAT's first entry, not the t_f it was aimed at."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    first = run(scenario, document, candidate, tmp_path / "first").candidate
    assert first is not None and first.intercept_time_s is not None
    assert first.intercept_time_s < 2400.0
    again = run(scenario, document, first, tmp_path / "again")
    assert again.converged and again.rounds == 1
    assert again.candidate is not None and again.candidate.burns == first.burns
    assert again.candidate.metadata["gmat_correction"]["planned_arrival_s"] == 2400.0


def test_the_corrector_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    results = [run(scenario, document, candidate, tmp_path / f"shoot{k}") for k in range(2)]

    def numbers(result: gmat_shoot.CorrectionResult) -> list:
        steps = [
            {key: value for key, value in dataclasses.asdict(step).items()
             if key not in {"run_ids", "wall_s"}}
            for step in result.log
        ]
        assert result.candidate is not None
        return [steps, [burn.to_dict() for burn in result.candidate.burns]]

    assert numbers(results[0]) == numbers(results[1])
    assert results[0].candidate.candidate_id == results[1].candidate.candidate_id


def test_a_correction_over_the_cap_is_shared_with_the_impulse_before(
    tmp_path: Path, monkeypatch
) -> None:
    """A thruster 2% short wants the last impulse 2% longer, and it is at the cap."""
    at_cap = 0.249 * np.array([0.8, -0.6, 0.0])
    config, scenario, document, candidate = _scenario(
        tmp_path, [(300.0, (0.12, 0.05, 0.0)), (900.0, tuple(at_cap))], 2400.0
    )
    _, run = _corrector(monkeypatch, config, Model(impulse_scale=0.98))
    result = run(scenario, document, candidate, tmp_path / "shoot")

    assert result.converged, result.reason
    assert result.log[0].mode == "last-two"
    corrected = result.candidate
    assert corrected is not None
    for burn in corrected.burns:
        assert burn.magnitude_km_s <= 0.249 + 1.0e-12
    assert corrected.burns[0].magnitude_km_s > candidate.burns[0].magnitude_km_s
    _assert_frames_agree(corrected)


def test_the_total_safe_limit_binds_the_shared_correction(tmp_path: Path, monkeypatch) -> None:
    at_cap = 0.249 * np.array([0.8, -0.6, 0.0])
    config, scenario, document, candidate = _scenario(
        tmp_path, [(300.0, (0.12, 0.05, 0.0)), (900.0, tuple(at_cap))], 2400.0,
        total_dv_safe_limit_km_s=0.379,
    )
    _, run = _corrector(monkeypatch, config, Model(impulse_scale=0.98))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert result.converged, result.reason
    assert result.log[0].mode == "last-two"
    assert result.candidate is not None
    # Spent to the limit, less the solver's 1e-9 km/s, and not past it.
    assert 0.379 - 1.0e-6 < result.candidate.total_dv_km_s <= 0.379


def test_a_single_impulse_at_the_cap_is_refused_with_the_reason(
    tmp_path: Path, monkeypatch
) -> None:
    at_cap = tuple(0.249 * np.array([0.8, -0.6, 0.0]))
    config, scenario, document, candidate = _scenario(tmp_path, [(300.0, at_cap)], 2400.0)
    _, run = _corrector(monkeypatch, config, Model(impulse_scale=0.98))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert not result.converged
    assert result.reason is not None
    assert "safe cap" in result.reason and "no earlier impulse" in result.reason
    # The flight that was made is still the best legal one, uncorrected.
    _assert_not_a_hit(result)
    assert result.closest_flight is not None
    assert result.closest_flight.burns == candidate.burns


def test_a_miss_no_legal_change_can_remove_is_refused(tmp_path: Path, monkeypatch) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    _, run = _corrector(monkeypatch, config, Model(target_offset_km=(3000.0, 0.0, 0.0)))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert not result.converged and result.rounds == 1
    assert result.reason is not None and "no change to the last two" in result.reason
    assert result.log[0].mode == "last-two" and result.log[0].change_km_s is None


@pytest.mark.parametrize(
    ("burns", "probes"), [([(300.0, (0.12, 0.05, 0.0))], 3), (TWO_BURNS, 6)]
)
def test_a_wrong_two_body_sensitivity_falls_back_to_gmat_measured(
    tmp_path: Path, monkeypatch, burns: list, probes: int
) -> None:
    """Every impulse turned 90 degrees: the two-body J points the wrong way."""
    model = Model(rotation=_rotation_z(90.0))
    config, scenario, document, candidate = _scenario(
        tmp_path, burns, 2400.0, placed_by=model, offset_km=(6.0, -4.0, 3.0),
    )
    fake, run = _corrector(monkeypatch, config, model)
    result = run(scenario, document, candidate, tmp_path / "shoot")

    assert result.converged, result.reason
    assert [step.kind for step in result.log] == ["flight", "flight", "sensitivity", "flight"]
    assert len(result.log[2].run_ids) == probes
    assert result.gmat_runs == fake.calls == 3 + probes
    assert result.log[1].miss_at_arrival_km > 0.3 * result.log[0].miss_at_arrival_km
    assert result.log[1].sensitivity == "gmat"


def test_the_measured_sensitivity_needs_two_rounds_left(tmp_path: Path, monkeypatch) -> None:
    model = Model(rotation=_rotation_z(90.0))
    config, scenario, document, candidate = _scenario(
        tmp_path, [(300.0, (0.12, 0.05, 0.0))], 2400.0, placed_by=model,
        offset_km=(6.0, -4.0, 3.0),
    )
    _, run = _corrector(monkeypatch, config, model)
    result = run(scenario, document, candidate, tmp_path / "shoot", max_rounds=3)
    assert not result.converged and result.rounds == 2
    assert result.reason is not None and "two more rounds" in result.reason
    assert result.reason.startswith("round 2: ")
    # The first flight came closer than the second, and is the one returned.
    assert result.log[0].closest_approach_km < result.log[1].closest_approach_km
    assert result.closest_approach_km == result.log[0].closest_approach_km
    assert result.closest_flight is not None
    assert result.closest_flight.burns == candidate.burns


def test_running_out_of_rounds_returns_the_closest_flight(tmp_path: Path, monkeypatch) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    result = run(scenario, document, candidate, tmp_path / "shoot", max_rounds=1)
    assert not result.converged and result.rounds == 1
    assert result.reason is not None and "1 round(s) used" in result.reason
    _assert_not_a_hit(result)
    assert result.closest_flight is not None
    assert result.closest_flight.burns == candidate.burns
    assert result.closest_approach_km == result.log[0].closest_approach_km > 0.5


def test_a_flight_through_the_earth_ends_the_correction(tmp_path: Path, monkeypatch) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    model = Model(pursuer_mu_scale=1.0 + 3.0e-4, dip_at_s=1500.0)
    _, run = _corrector(monkeypatch, config, model)
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert not result.converged and result.rounds == 1
    assert result.reason is not None and "inside the" in result.reason
    assert result.candidate is None and result.closest_flight is None


@pytest.mark.parametrize("floor_km", [None, 100.0, 5000.0])
def test_a_flight_under_the_altitude_floor_ends_the_correction(
    tmp_path: Path, monkeypatch, floor_km: float | None
) -> None:
    config, scenario, document, candidate = _scenario(
        tmp_path, TWO_BURNS, 2400.0, minimum_flown_altitude_km=floor_km
    )
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    if floor_km == 5000.0:
        assert not result.converged and result.rounds == 1
        assert result.reason is not None
        assert "under the 5000 km policy.minimum_flown_altitude_km" in result.reason
    else:
        assert result.converged


@pytest.mark.parametrize("dip_at_s, stopped", [(1500.0, True), (2500.0, False)])
def test_the_floor_is_held_where_verification_holds_it(
    tmp_path: Path, monkeypatch, dip_at_s: float, stopped: bool
) -> None:
    """Up to the first entry, as report_parser cuts it; a dip after it is not refused."""
    config, scenario, document, candidate = _scenario(
        tmp_path, TWO_BURNS, 2400.0, minimum_flown_altitude_km=100.0
    )
    model = Model(
        pursuer_mu_scale=1.0 + 3.0e-4, dip_at_s=dip_at_s, dip_radius_km=6378.1363 + 50.0
    )
    _, run = _corrector(monkeypatch, config, model)
    result = run(scenario, document, candidate, tmp_path / "shoot")
    if stopped:
        assert not result.converged and result.rounds == 1
        assert result.reason is not None
        # The row sits at 50 km; between rows, as verification reads it, a little lower.
        flown = re.search(r"to (\d+\.\d+) km altitude", result.reason)
        assert flown is not None and 49.9 < float(flown.group(1)) <= 50.0
    else:
        assert result.converged
        assert result.first_entry_time_s is not None and result.first_entry_time_s < dip_at_s


def test_a_miss_is_refused_before_flying(tmp_path: Path, monkeypatch) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    miss = dataclasses.replace(candidate, successful=False)
    fake, run = _corrector(monkeypatch, config, Model())
    result = run(scenario, document, miss, tmp_path / "shoot")
    assert not result.converged and fake.calls == 0 and result.gmat_runs == 0
    assert result.reason == "the candidate is a miss, and only a hit is corrected"


def test_a_correction_keeps_the_stamps_and_records_each_frame(
    tmp_path: Path, monkeypatch
) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    candidate.metadata.update(scenario_sha256="script-hash", horizon_s=scenario.max_time_s)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    corrected = result.candidate
    assert corrected is not None
    assert corrected.metadata["scenario_sha256"] == "script-hash"
    assert corrected.metadata["horizon_s"] == scenario.max_time_s
    assert all(burn.frame is not None for burn in corrected.burns)
    assert corrected.burns[1].frame == candidate.burns[1].frame


def test_a_graze_before_the_arrival_is_the_closest_approach_counted(
    tmp_path: Path, monkeypatch
) -> None:
    """Verification measures a hit on its first passage, so the corrector does too."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    model = Model(pursuer_mu_scale=1.0 + 3.0e-4, graze_at_s=1500.0)
    _, run = _corrector(monkeypatch, config, model)
    result = run(scenario, document, candidate, tmp_path / "shoot", max_rounds=8)
    assert not result.converged
    _assert_not_a_hit(result)
    assert result.log[-1].miss_at_arrival_km < 0.5
    assert result.closest_approach_km is not None and 0.5 < result.closest_approach_km < 5.0
    assert result.closest_approach_time_s == pytest.approx(1500.0, abs=STEP_S)
    # Stopped once t_f was hit: further rounds cannot move the earlier pass.
    assert result.rounds == result.gmat_runs == 2
    assert result.reason is not None and "earlier pass" in result.reason
    assert "already intercepts there" in result.reason and result.reason.startswith("round 2:")
    assert result.closest_flight is not None
    saved = result.closest_flight.metadata["gmat_correction"]
    assert saved["round"] == 2
    assert saved["first_entry_time_s"] == result.first_entry_time_s is not None


def test_the_missing_post_burn_row_is_reconstructed(tmp_path: Path, monkeypatch) -> None:
    """R2026a writes the pre-burn row at a Maneuver and no post-burn one."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    outcomes = []
    for k, rows in enumerate((False, True)):
        _, run = _corrector(
            monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4, post_burn_rows=rows)
        )
        outcomes.append(run(scenario, document, candidate, tmp_path / f"shoot{k}"))
    for left, right in zip(outcomes[0].log, outcomes[1].log, strict=True):
        assert left.miss_at_arrival_km == pytest.approx(right.miss_at_arrival_km, abs=1.0e-9)
        assert left.minimum_radius_km == pytest.approx(right.minimum_radius_km, abs=1.0e-6)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"burns": ()}, "no impulse"),
        ({"metadata": {"arrival_time_s": 800.0}}, "not after the last impulse"),
        ({"metadata": {"arrival_time_s": 1.0e6}}, "after T_max"),
    ],
)
def test_a_candidate_that_cannot_be_corrected_is_refused_before_flying(
    tmp_path: Path, monkeypatch, change: dict, reason: str
) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    fake, run = _corrector(monkeypatch, config, Model())
    broken = dataclasses.replace(candidate, candidate_id="", **change)
    result = run(scenario, document, broken, tmp_path / "shoot")
    assert not result.converged and result.rounds == 0 and fake.calls == 0
    assert result.reason is not None and reason in result.reason
    saved = json.loads((tmp_path / "shoot" / "correction.json").read_text())
    assert saved["reason"] == result.reason and saved["rounds"] == 0


def test_a_candidate_over_the_safe_cap_is_refused_before_flying(
    tmp_path: Path, monkeypatch
) -> None:
    config, scenario, document, candidate = _scenario(
        tmp_path, [(300.0, (0.2495, 0.0, 0.0))], 2400.0
    )
    fake, run = _corrector(monkeypatch, config, Model())
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert fake.calls == 0 and result.reason is not None and "safe cap" in result.reason


def test_burns_whose_frames_disagree_are_refused(tmp_path: Path, monkeypatch) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    burn = candidate.burns[1]
    skewed = Burn(
        burn.time_s, burn.dv_inertial_km_s,
        as_vector3(np.asarray(burn.dv_script_km_s)[[1, 0, 2]]), burn.slot_index,
    )
    broken = dataclasses.replace(
        candidate, burns=(candidate.burns[0], skewed), candidate_id=""
    )
    fake, run = _corrector(monkeypatch, config, Model())
    result = run(scenario, document, broken, tmp_path / "shoot")
    assert fake.calls == 0 and result.rounds == 0 and not result.converged
    assert result.reason is not None and "two frames disagree" in result.reason
    assert (tmp_path / "shoot" / "correction.json").is_file()


@pytest.mark.parametrize("aim_km", [0.0, -1.0, 3.6, 5.5, math.nan])
def test_an_aim_outside_the_sphere_is_refused(
    tmp_path: Path, monkeypatch, aim_km: float
) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    _, run = _corrector(monkeypatch, config, Model())
    with pytest.raises(ValueError, match="aim_km"):
        run(scenario, document, candidate, tmp_path / "shoot", aim_km=aim_km)


def test_a_failed_gmat_run_keeps_the_correction_so_far(tmp_path: Path, monkeypatch) -> None:
    """GMAT exits 1 when drag stops RK89 low down; the round before is not lost."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    model = Model(pursuer_mu_scale=1.0 + 3.0e-4, abort_from_call=2)
    fake, run = _corrector(monkeypatch, config, model)
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert not result.converged and result.rounds == 2 and result.gmat_runs == fake.calls == 2
    assert result.reason is not None and "round 2 (flight) failed" in result.reason
    assert result.log[1].error is not None and "exit status 1" in result.log[1].error
    _assert_not_a_hit(result)
    assert result.closest_flight is not None
    assert result.closest_flight.burns == candidate.burns
    assert result.closest_approach_km == result.log[0].closest_approach_km
    assert (tmp_path / "shoot" / "correction.json").is_file()


def test_a_failed_sensitivity_probe_keeps_the_correction_so_far(
    tmp_path: Path, monkeypatch
) -> None:
    model = Model(rotation=_rotation_z(90.0), abort_from_call=3)
    config, scenario, document, candidate = _scenario(
        tmp_path, [(300.0, (0.12, 0.05, 0.0))], 2400.0, placed_by=model,
        offset_km=(6.0, -4.0, 3.0),
    )
    _, run = _corrector(monkeypatch, config, model)
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert not result.converged
    assert [step.kind for step in result.log] == ["flight", "flight", "sensitivity"]
    assert result.reason is not None and "(sensitivity) failed" in result.reason
    _assert_not_a_hit(result)
    assert result.closest_approach_km == result.log[0].closest_approach_km


@pytest.mark.parametrize(
    ("until_s", "reason"), [(2000.0, "outside the ephemeris"), (800.0, "increasing epochs")]
)
def test_a_report_that_stops_short_ends_the_correction(
    tmp_path: Path, monkeypatch, until_s: float, reason: str
) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    _, run = _corrector(monkeypatch, config, Model(report_until_s=until_s))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert not result.converged and result.rounds == 1
    assert result.reason is not None and reason in result.reason
    assert result.candidate is None and result.closest_flight is None


def test_the_total_limit_alone_refuses_a_longer_impulse(tmp_path: Path, monkeypatch) -> None:
    """Far under the per-impulse cap; only the total can say no."""
    config, scenario, document, candidate = _scenario(
        tmp_path, [(300.0, (0.12, 0.05, 0.0))], 2400.0, total_dv_safe_limit_km_s=0.1305
    )
    _, run = _corrector(monkeypatch, config, Model(impulse_scale=0.98))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert not result.converged
    assert result.reason is not None
    assert "the total would be" in result.reason and "no earlier impulse" in result.reason


def test_the_total_limit_sends_a_correction_to_the_two_impulse_solve(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "plain").mkdir()
    _, _, _, plain = _scenario(tmp_path / "plain", TWO_BURNS, 2400.0)
    limit = plain.total_dv_km_s + 0.002
    config, scenario, document, candidate = _scenario(
        tmp_path, TWO_BURNS, 2400.0, total_dv_safe_limit_km_s=limit
    )
    _, run = _corrector(monkeypatch, config, Model(impulse_scale=0.98))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert result.log[0].mode == "last-two"
    for step in result.log:
        assert step.total_dv_km_s is not None and step.total_dv_km_s <= limit + 1.0e-12
    for burn in candidate.burns:
        assert burn.magnitude_km_s < 0.2


def _two_impulses_at_the_cap() -> tuple[list, dict, np.ndarray, gmat_shoot._Limits]:
    """A shared correction whose least-norm start puts the last impulse over the cap.

    J = 1000 s for both, so the two changes must add to 4 mm/s along x; half
    each takes the last 2 mm/s over its 249 m/s cap.
    """
    schedule = [(300.0, np.array([0.1, 0.0, 0.0])), (900.0, np.array([0.249, 0.0, 0.0]))]
    jacobians = {0: 1000.0 * np.eye(3), 1: 1000.0 * np.eye(3)}
    return schedule, jacobians, np.array([-0.004, 0.0, 0.0]), gmat_shoot._Limits(0.249, 2.95)


def test_a_feasible_slsqp_point_is_taken_whatever_its_status(monkeypatch) -> None:
    schedule, jacobians, miss, limits = _two_impulses_at_the_cap()
    answer = SimpleNamespace(x=np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.0]), success=False, status=8)
    monkeypatch.setattr(gmat_shoot, "minimize", lambda *args, **kwargs: answer)
    stepped = gmat_shoot._least_norm_two(schedule, miss, jacobians[0], jacobians[1], limits)
    assert stepped is not None
    assert np.allclose(stepped[0][1], [0.100004, 0.0, 0.0], rtol=0.0, atol=1e-12)
    assert np.allclose(stepped[1][1], [0.249, 0.0, 0.0])


def test_an_slsqp_point_off_the_equations_is_refused(monkeypatch) -> None:
    schedule, jacobians, miss, limits = _two_impulses_at_the_cap()
    answer = SimpleNamespace(x=np.zeros(6), success=True, status=0)
    monkeypatch.setattr(gmat_shoot, "minimize", lambda *args, **kwargs: answer)
    assert gmat_shoot._least_norm_two(schedule, miss, jacobians[0], jacobians[1], limits) is None


def _reference_jacobians(scenario: Scenario, schedule: list) -> dict[int, np.ndarray]:
    """Richardson on central differences at 1 and 0.5 m/s."""
    out: dict[int, list[np.ndarray]] = {}
    for step in (1.0e-3, 5.0e-4):
        for index in range(len(schedule)):
            columns = []
            for axis in range(3):
                offset = np.zeros(3)
                offset[axis] = step
                plus = gmat_shoot._replace_impulse(schedule, index, schedule[index][1] + offset)
                minus = gmat_shoot._replace_impulse(schedule, index, schedule[index][1] - offset)
                columns.append(
                    (gmat_shoot._two_body_arrival(scenario, plus, 2400.0)
                     - gmat_shoot._two_body_arrival(scenario, minus, 2400.0)) / (2.0 * step)
                )
            out.setdefault(index, []).append(np.column_stack(columns))
    return {index: (4.0 * pair[1] - pair[0]) / 3.0 for index, pair in out.items()}


def test_the_two_body_sensitivity_matches_an_extrapolated_one(tmp_path: Path) -> None:
    """Measured: 1.4e-10 relative at the 1e-5 km/s step, 4.8e-5 at 3e-2."""
    _, scenario, _, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    schedule = [(burn.time_s, np.asarray(burn.dv_inertial_km_s)) for burn in candidate.burns]
    got = gmat_shoot._two_body_jacobians(scenario, schedule, 2400.0)
    for index, want in _reference_jacobians(scenario, schedule).items():
        assert np.max(np.abs(got[index] - want)) < 1.0e-8 * np.max(np.abs(want))


def test_the_measured_sensitivity_step_is_small_enough(tmp_path: Path) -> None:
    """Two-body standing in for GMAT: 2.0e-4 relative at the 1 m/s step, 2.0e-2 at 100 m/s."""
    _, scenario, _, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    schedule = [(burn.time_s, np.asarray(burn.dv_inertial_km_s)) for burn in candidate.burns]

    def fly(burns, path):
        flown = [(burn.time_s, np.asarray(burn.dv_inertial_km_s)) for burn in burns]
        miss = gmat_shoot._two_body_arrival(scenario, flown, 2400.0)
        return gmat_shoot._Flight(burns, miss, None, 0.0, 0.0, 0.0, 0.0, None)

    measured, flights = gmat_shoot._measure_sensitivity(
        fly, scenario, schedule, [burn.slot_index for burn in candidate.burns],
        candidate.burns, fly(candidate.burns, None), tmp_path / "probes", 2,
    )
    assert len(flights) == 6
    for index, want in _reference_jacobians(scenario, schedule).items():
        assert np.max(np.abs(measured[index] - want)) < 1.0e-3 * np.max(np.abs(want))


def test_sensitivity_probes_keep_to_the_configured_workers(tmp_path: Path, monkeypatch) -> None:
    model = Model(rotation=_rotation_z(90.0), hold_s=0.05)
    config, scenario, document, candidate = _scenario(
        tmp_path, TWO_BURNS, 2400.0, placed_by=model, offset_km=(6.0, -4.0, 3.0),
    )
    config = dataclasses.replace(config, gmat=dataclasses.replace(config.gmat, workers=2))
    fake, run = _corrector(monkeypatch, config, model)
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert result.converged, result.reason
    assert len(result.log[2].run_ids) == 6
    assert fake.peak == 2


class FlippingGmat(FakeGmat):
    """Turns every impulse 90 degrees, then -90 from one call on."""

    def __init__(self, config: OrbitWinConfig, model: Model, flip_from_call: int) -> None:
        super().__init__(config, model)
        self.flip_from_call = flip_from_call

    def __call__(self, *args, **kwargs) -> GmatRunResult:
        if self.calls + 1 >= self.flip_from_call:
            self.model.rotation = _rotation_z(-90.0)
        return super().__call__(*args, **kwargs)


def test_a_gmat_measured_sensitivity_that_fails_ends_the_correction(
    tmp_path: Path, monkeypatch
) -> None:
    model = Model(rotation=_rotation_z(90.0))
    config, scenario, document, candidate = _scenario(
        tmp_path, [(300.0, (0.12, 0.05, 0.0))], 2400.0, placed_by=model,
        offset_km=(6.0, -4.0, 3.0),
    )
    # Flights 1 and 2 and three probes are calls 1-5; flight 3 flips the thruster.
    fake = FlippingGmat(config, model, flip_from_call=6)
    monkeypatch.setattr(gmat_shoot, "run_gmat", fake)
    result = correct_with_gmat(
        document, scenario, candidate, config, "GmatConsole", "startup.txt", tmp_path / "shoot",
        max_rounds=6,
    )
    assert not result.converged
    assert [step.kind for step in result.log] == ["flight", "flight", "sensitivity", "flight"]
    assert result.reason is not None and "GMAT-measured sensitivity" in result.reason
    assert result.reason.startswith("round 4: ")
    assert result.log[1].sensitivity == "gmat" and result.log[3].sensitivity is None
    assert result.gmat_runs == fake.calls == 6
    _assert_not_a_hit(result)
    closest = min(step.closest_approach_km for step in result.log if step.kind == "flight")
    assert result.closest_approach_km == closest
    saved = json.loads((tmp_path / "shoot" / "correction.json").read_text())
    assert saved["converged"] is False and saved["candidate"] is None
    assert saved["reason"] == result.reason
    assert saved["closest_flight"]["metadata"]["gmat_correction"]["converged"] is False


def test_a_step_that_leaves_half_the_miss_falls_back_to_gmat_measured(
    tmp_path: Path, monkeypatch
) -> None:
    """A thruster at half strength: each two-body step leaves 0.51 of the miss."""
    model = Model(impulse_scale=0.5)
    config, scenario, document, candidate = _scenario(
        tmp_path, [(300.0, (0.12, 0.05, 0.0))], 2400.0, placed_by=model,
        offset_km=(6.0, -4.0, 3.0),
    )
    _, run = _corrector(monkeypatch, config, model)
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert result.converged, result.reason
    assert [step.kind for step in result.log] == ["flight", "flight", "sensitivity", "flight"]
    ratio = result.log[1].miss_at_arrival_km / result.log[0].miss_at_arrival_km
    assert 0.3 < ratio < 0.99


@pytest.mark.parametrize(
    ("burns", "policy", "reason"),
    [
        ([(300.0, (0.1, 0.0, 0.0)), (350.0, (0.1, 0.0, 0.0))], {}, "safe separation"),
        (TWO_BURNS, {"total_dv_safe_limit_km_s": 0.2}, "safe limit before any correction"),
    ],
)
def test_a_candidate_outside_the_safe_margins_is_refused_before_flying(
    tmp_path: Path, monkeypatch, burns: list, policy: dict, reason: str
) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, burns, 2400.0, **policy)
    fake, run = _corrector(monkeypatch, config, Model())
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert fake.calls == 0 and result.rounds == 0
    assert result.reason is not None and reason in result.reason


def test_the_other_impulses_count_toward_the_shared_total() -> None:
    """Three burns, the total 1 mm/s from its limit: only turning both impulses fits.

    J = 1000 s, so the change must add 4 mm/s along x. The middle impulse lies at
    45 degrees; lengthening it would cost 2.8 mm/s of total, turning it costs none.
    """
    diagonal = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    schedule = [
        (0.0, np.array([0.0, 0.0, 0.2])),
        (300.0, 0.1 * diagonal),
        (900.0, np.array([0.249, 0.0, 0.0])),
    ]
    total = sum(float(np.linalg.norm(impulse)) for _, impulse in schedule)
    limits = gmat_shoot._Limits(0.249, total + 1.0e-6)
    jacobian = 1000.0 * np.eye(3)
    stepped = gmat_shoot._least_norm_two(
        schedule, np.array([-0.004, 0.0, 0.0]), jacobian, jacobian, limits
    )
    assert stepped is not None
    assert sum(float(np.linalg.norm(impulse)) for _, impulse in stepped) <= total + 1.0e-6
    change = sum(new - old for (_, new), (_, old) in zip(stepped, schedule, strict=True))
    assert np.allclose(change, [4.0e-6, 0.0, 0.0], rtol=0.0, atol=1.0e-12)


def test_the_flight_runs_past_the_planned_arrival(tmp_path: Path, monkeypatch) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    run(scenario, document, candidate, tmp_path / "shoot")
    flown, _ = parse_gmat_script(
        tmp_path / "shoot" / "round-01" / "correction_candidate.script", config
    )
    end_s = sum(segment.duration_s for segment in flown.propagate_segments)
    assert end_s >= 2400.0 + gmat_shoot.ARRIVAL_MARGIN_S


def test_probes_never_lengthen_an_impulse(tmp_path: Path) -> None:
    """The last impulse lies on x, so a probe along y or z would lengthen it."""
    _, scenario, _, candidate = _scenario(
        tmp_path, [(300.0, (0.12, 0.05, 0.0)), (900.0, (0.249, 0.0, 0.0))], 2400.0
    )
    schedule = [(burn.time_s, np.asarray(burn.dv_inertial_km_s)) for burn in candidate.burns]
    schedule[1] = (900.0, np.array([0.249, 0.0, 0.0]))
    flown: list[tuple[Burn, ...]] = []

    def fly(burns, path):
        flown.append(burns)
        return gmat_shoot._Flight(burns, np.zeros(3), None, 0.0, 0.0, 0.0, 0.0, None)

    gmat_shoot._measure_sensitivity(
        fly, scenario, schedule, [0, 1], gmat_shoot._express(scenario, schedule, [0, 1],
                                                             candidate.burns),
        fly(candidate.burns, None), tmp_path / "probes", 1,
    )
    for burns in flown[1:]:
        for burn, (_, impulse) in zip(burns, schedule, strict=True):
            assert burn.magnitude_km_s <= float(np.linalg.norm(impulse)) + 1.0e-15


def test_a_converged_candidate_is_scored_and_constrained(tmp_path: Path, monkeypatch) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    config = dataclasses.replace(
        config, score=ScoreConfig(kt=0.002, ct=3000.0, kv=0.01, cv=300.0, cr_km=100.0)
    )
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    corrected = run(scenario, document, candidate, tmp_path / "shoot").candidate
    assert corrected is not None and corrected.score is not None
    assert corrected.score == score_candidate(corrected, scenario, config.score).raw_score
    assert corrected.metadata["frontier"] == "robust"


def test_an_impulse_that_changes_length_between_frames_ends_the_correction(
    tmp_path: Path, monkeypatch
) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    real = gmat_shoot.inertial_to_script_components
    calls = []

    def stretching(*args, **kwargs):
        # Right while the candidate's frames are checked, 1% long afterwards.
        calls.append(None)
        components = real(*args, **kwargs)
        return components if len(calls) <= len(candidate.burns) else 1.01 * components

    monkeypatch.setattr(gmat_shoot, "inertial_to_script_components", stretching)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert not result.converged and result.rounds == 1
    assert result.reason is not None and "changed length between frames" in result.reason
    assert (tmp_path / "shoot" / "correction.json").is_file()


def test_a_render_that_moves_a_burn_ends_the_correction(tmp_path: Path, monkeypatch) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    real = gmat_shoot.write_rendered_candidate

    def moved(*args, **kwargs):
        rendered = real(*args, **kwargs)
        times = (rendered.burn_times_s[0] + 1.0, *rendered.burn_times_s[1:])
        return dataclasses.replace(rendered, burn_times_s=times)

    monkeypatch.setattr(gmat_shoot, "write_rendered_candidate", moved)
    _, run = _corrector(monkeypatch, config, Model())
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert not result.converged and result.rounds == 1
    assert result.reason is not None and "The render put slot 0" in result.reason
    assert result.log[0].error is not None
    assert (tmp_path / "shoot" / "correction.json").is_file()


@pytest.mark.parametrize("frame", [VNB, INERTIAL], ids=["VNB", "EarthMJ2000Eq"])
def test_the_correction_holds_in_either_burn_frame(
    tmp_path: Path, monkeypatch, frame: tuple[str, str, str]
) -> None:
    """The ranking search may write burns in EarthMJ2000Eq; the double flies each as declared."""
    config, scenario, document, candidate = _scenario(
        tmp_path, TWO_BURNS, 2400.0, declared_frame=frame
    )
    assert all(burn_frame(scenario, burn.slot_index) == frame for burn in candidate.burns)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert result.converged and result.rounds == 2, result.reason
    corrected = result.candidate
    assert corrected is not None
    _assert_frames_agree(corrected)
    flown, _ = parse_gmat_script(
        tmp_path / "shoot" / "round-02" / "correction_candidate.script", config
    )
    for slot, burn in zip(flown.burn_slots, corrected.burns, strict=True):
        assert (slot.coordinate_system, slot.origin, slot.axes) == frame
        assert np.allclose(slot.vector, burn.dv_script_km_s, rtol=0.0, atol=1.0e-15)
        if frame == INERTIAL:
            assert burn.dv_script_km_s == burn.dv_inertial_km_s


@pytest.mark.parametrize("frame", [VNB, INERTIAL], ids=["VNB", "EarthMJ2000Eq"])
def test_sharing_and_measured_sensitivity_hold_in_either_burn_frame(
    tmp_path: Path, monkeypatch, frame: tuple[str, str, str]
) -> None:
    at_cap = 0.249 * np.array([0.8, -0.6, 0.0])
    (tmp_path / "cap").mkdir()
    config, scenario, document, candidate = _scenario(
        tmp_path / "cap", [(300.0, (0.12, 0.05, 0.0)), (900.0, tuple(at_cap))], 2400.0,
        declared_frame=frame,
    )
    _, run = _corrector(monkeypatch, config, Model(impulse_scale=0.98))
    shared = run(scenario, document, candidate, tmp_path / "cap" / "shoot")
    assert shared.converged and shared.log[0].mode == "last-two", shared.reason

    model = Model(rotation=_rotation_z(90.0))
    (tmp_path / "turned").mkdir()
    config, scenario, document, candidate = _scenario(
        tmp_path / "turned", TWO_BURNS, 2400.0, placed_by=model, offset_km=(6.0, -4.0, 3.0),
        declared_frame=frame,
    )
    _, run = _corrector(monkeypatch, config, model)
    measured = run(scenario, document, candidate, tmp_path / "turned" / "shoot")
    assert measured.converged, measured.reason
    assert [step.kind for step in measured.log] == ["flight", "flight", "sensitivity", "flight"]


def test_an_impulse_half_a_millimetre_per_second_over_the_safe_cap_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """The cap's slack is 1e-12 km/s: 5e-7 km/s over is over."""
    config, scenario, document, candidate = _scenario(
        tmp_path, [(300.0, (0.249 + 5.0e-7, 0.0, 0.0))], 2400.0
    )
    fake, run = _corrector(monkeypatch, config, Model())
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert fake.calls == 0 and result.reason is not None and "safe cap" in result.reason


class DeeperFirstGraze(FakeGmat):
    """The first flight's graze 1 km deeper than every later one's."""

    def __call__(self, *args, **kwargs) -> GmatRunResult:
        self.model.graze_km = 3.0 if self.calls == 0 else 4.0
        return super().__call__(*args, **kwargs)


def test_the_early_stop_returns_the_flight_its_reason_describes(
    tmp_path: Path, monkeypatch
) -> None:
    """Round 1's earlier pass is closer, but round 2 is the one that hits t_f."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    fake = DeeperFirstGraze(config, Model(pursuer_mu_scale=1.0 + 3.0e-4, graze_at_s=1500.0))
    monkeypatch.setattr(gmat_shoot, "run_gmat", fake)
    result = correct_with_gmat(
        document, scenario, candidate, config, "GmatConsole", "startup.txt", tmp_path / "shoot",
        max_rounds=8,
    )
    assert result.log[0].closest_approach_km < result.log[1].closest_approach_km
    assert result.reason is not None and result.reason.startswith("round 2:")
    assert result.closest_approach_km == result.log[1].closest_approach_km
    flight = result.closest_flight
    assert flight is not None and flight.metadata["gmat_correction"]["round"] == 2
    assert flight.metadata["gmat_correction"]["miss_at_arrival_km"] < 0.5


def test_the_early_stop_fires_on_a_miss_just_inside_the_aim(
    tmp_path: Path, monkeypatch
) -> None:
    """A 2% short thruster leaves 0.279 km at t_f after one step: inside the aim, stop."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    model = Model(impulse_scale=0.98, graze_at_s=1500.0)
    _, run = _corrector(monkeypatch, config, model)
    result = run(scenario, document, candidate, tmp_path / "shoot", max_rounds=8)
    assert result.rounds == 2
    assert 0.05 < result.log[1].miss_at_arrival_km < 0.5
    assert result.reason is not None and "earlier pass" in result.reason


def test_running_out_of_rounds_names_the_closest_flight(tmp_path: Path, monkeypatch) -> None:
    model = Model(rotation=_rotation_z(90.0))
    config, scenario, document, candidate = _scenario(
        tmp_path, [(300.0, (0.12, 0.05, 0.0))], 2400.0, placed_by=model,
        offset_km=(6.0, -4.0, 3.0),
    )
    _, run = _corrector(monkeypatch, config, model)
    result = run(scenario, document, candidate, tmp_path / "shoot", max_rounds=2)
    assert result.reason is not None and "the closest flight, round 1," in result.reason
    assert result.closest_flight is not None
    assert result.closest_flight.metadata["gmat_correction"]["round"] == 1


@pytest.mark.parametrize(
    "error",
    [
        DynamicsError("probe could not be expressed"),
        PolicyError("probe render refused"),
        ParseError("probe script unreadable"),
        np.linalg.LinAlgError("singular probe offsets"),
    ],
    ids=["DynamicsError", "PolicyError", "ParseError", "LinAlgError"],
)
def test_probes_that_never_start_end_the_correction_and_are_not_counted(
    tmp_path: Path, monkeypatch, error: Exception
) -> None:
    model = Model(rotation=_rotation_z(90.0))
    config, scenario, document, candidate = _scenario(
        tmp_path, TWO_BURNS, 2400.0, placed_by=model, offset_km=(6.0, -4.0, 3.0)
    )
    fake, run = _corrector(monkeypatch, config, model)

    def raising(*args, **kwargs):
        raise error

    monkeypatch.setattr(gmat_shoot, "_measure_sensitivity", raising)
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert not result.converged
    assert result.gmat_runs == fake.calls == 2
    assert result.rounds == len(result.log) == 3
    assert result.log[-1].kind == "sensitivity" and result.log[-1].round == 3
    assert result.reason is not None and result.reason.startswith("round 3 (sensitivity) failed")
    saved = json.loads((tmp_path / "shoot" / "correction.json").read_text())
    assert saved["gmat_runs"] == 2 and saved["rounds"] == 3


def test_a_probe_whose_render_is_refused_is_not_counted(tmp_path: Path, monkeypatch) -> None:
    model = Model(rotation=_rotation_z(90.0))
    config, scenario, document, candidate = _scenario(
        tmp_path, TWO_BURNS, 2400.0, placed_by=model, offset_km=(6.0, -4.0, 3.0)
    )
    fake, run = _corrector(monkeypatch, config, model)
    real = gmat_shoot.write_rendered_candidate

    def refusing(script, *args, **kwargs):
        if Path(script).parent.name == "burn1-axis0":
            raise PolicyError("refused for the test")
        return real(script, *args, **kwargs)

    monkeypatch.setattr(gmat_shoot, "write_rendered_candidate", refusing)
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert result.reason is not None and "refused for the test" in result.reason
    assert result.gmat_runs == fake.calls == 2 + 5


def test_a_script_that_will_not_parse_ends_the_correction(tmp_path: Path, monkeypatch) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)

    def unparseable(*args, **kwargs):
        raise ParseError("not parseable for the test")

    monkeypatch.setattr(gmat_shoot, "parse_gmat_script", unparseable)
    fake, run = _corrector(monkeypatch, config, Model())
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert fake.calls == 0 and result.gmat_runs == 0 and result.rounds == 1
    assert result.reason is not None and "not parseable for the test" in result.reason
    assert (tmp_path / "shoot" / "correction.json").is_file()


def test_an_arrival_just_before_t_max_flies_to_t_max(tmp_path: Path, monkeypatch) -> None:
    """Arrival 50 s before T_max: the 120 s margin stops at T_max."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    config = dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, horizon_s=2450.0)
    )
    scenario, document = parse_gmat_script(tmp_path / "official.script", config)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert result.converged, result.reason
    flown, _ = parse_gmat_script(
        tmp_path / "shoot" / "round-01" / "correction_candidate.script", config
    )
    assert sum(segment.duration_s for segment in flown.propagate_segments) == pytest.approx(
        2450.0, abs=1.0e-9
    )


@pytest.mark.parametrize(
    "error", [DynamicsError("no step"), np.linalg.LinAlgError("no step")],
    ids=["DynamicsError", "LinAlgError"],
)
def test_a_step_that_cannot_be_computed_ends_the_correction(
    tmp_path: Path, monkeypatch, error: Exception
) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)

    def raising(*args, **kwargs):
        raise error

    monkeypatch.setattr(gmat_shoot, "_two_body_jacobians", raising)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert not result.converged and result.rounds == 1
    assert result.reason == f"the step after round 1 failed: {type(error).__name__}: no step"
    saved = json.loads((tmp_path / "shoot" / "correction.json").read_text())
    assert "Traceback" in saved["traceback"] and "no step" in saved["traceback"]


class Switching(FakeGmat):
    """Changes the model from one call on: a later flight that behaves differently."""

    def __init__(self, config: OrbitWinConfig, model: Model, from_call: int, **changes) -> None:
        super().__init__(config, model)
        self.from_call = from_call
        self.changes = changes

    def __call__(self, *args, **kwargs) -> GmatRunResult:
        if self.calls + 1 >= self.from_call:
            for name, value in self.changes.items():
                setattr(self.model, name, value)
        return super().__call__(*args, **kwargs)


def _after_a_measured_j(
    tmp_path: Path, monkeypatch, *, max_rounds: int = 6, **changes
) -> tuple[FakeGmat, gmat_shoot.CorrectionResult]:
    """Flights 1-2, three probes (calls 3-5), then flight 4 changed as given."""
    model = Model(rotation=_rotation_z(90.0))
    config, scenario, document, candidate = _scenario(
        tmp_path, [(300.0, (0.12, 0.05, 0.0))], 2400.0, placed_by=model,
        offset_km=(6.0, -4.0, 3.0),
    )
    fake = Switching(config, model, 6, **changes)
    monkeypatch.setattr(gmat_shoot, "run_gmat", fake)
    result = correct_with_gmat(
        document, scenario, candidate, config, "GmatConsole", "startup.txt", tmp_path / "shoot",
        max_rounds=max_rounds,
    )
    return fake, result


def test_runs_out_of_rounds_after_a_measured_j_counts_every_run(
    tmp_path: Path, monkeypatch
) -> None:
    fake, result = _after_a_measured_j(
        tmp_path, monkeypatch, max_rounds=4, rotation=_rotation_z(-90.0)
    )
    assert result.reason is not None and result.reason.startswith("4 round(s) used")
    assert result.rounds == 4 and result.gmat_runs == fake.calls == 6


def test_a_dive_after_a_measured_j_counts_every_run(tmp_path: Path, monkeypatch) -> None:
    fake, result = _after_a_measured_j(tmp_path, monkeypatch, dip_at_s=1500.0)
    assert result.reason is not None and "in round 4, inside the" in result.reason
    assert result.rounds == 4 and result.gmat_runs == fake.calls == 6


def test_an_earlier_pass_after_a_measured_j_counts_every_run(
    tmp_path: Path, monkeypatch
) -> None:
    fake, result = _after_a_measured_j(tmp_path, monkeypatch, graze_at_s=1500.0)
    assert result.reason is not None and result.reason.startswith("round 4: the miss at t_f")
    assert result.rounds == 4 and result.gmat_runs == fake.calls == 6


def _second_step(monkeypatch, replacement) -> None:
    """The step after the first flight as it is, the next one replaced."""
    real = gmat_shoot._step
    calls = []

    def step(*args, **kwargs):
        calls.append(None)
        return real(*args, **kwargs) if len(calls) == 1 else replacement(*args, **kwargs)

    monkeypatch.setattr(gmat_shoot, "_step", step)


def test_a_step_that_fails_after_a_measured_j_counts_every_run(
    tmp_path: Path, monkeypatch
) -> None:
    def raising(*args, **kwargs):
        raise DynamicsError("no step")

    _second_step(monkeypatch, raising)
    fake, result = _after_a_measured_j(tmp_path, monkeypatch)
    assert result.reason == "the step after round 2 failed: DynamicsError: no step"
    assert result.rounds == 3 and result.gmat_runs == fake.calls == 5


def test_no_step_after_a_measured_j_names_its_round_and_counts_every_run(
    tmp_path: Path, monkeypatch
) -> None:
    _second_step(monkeypatch, lambda *args, **kwargs: (None, "last", "no step for the test"))
    fake, result = _after_a_measured_j(tmp_path, monkeypatch)
    assert result.reason == "after round 2: no step for the test"
    assert result.rounds == 3 and result.gmat_runs == fake.calls == 5


@pytest.mark.parametrize(
    ("error", "on_call"),
    [
        (PermissionError(13, "Permission denied", "GmatConsole"), 1),
        (OSError(28, "No space left on device"), 2),
        (np.linalg.LinAlgError("singular in the flight"), 2),
    ],
    ids=["not-executable", "disk-full", "LinAlgError"],
)
def test_a_flight_that_cannot_start_keeps_the_rounds_before(
    tmp_path: Path, monkeypatch, error: Exception, on_call: int
) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    fake = FakeGmat(config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    calls = []

    def runner(*args, **kwargs):
        calls.append(None)
        if len(calls) == on_call:
            raise error
        return fake(*args, **kwargs)

    monkeypatch.setattr(gmat_shoot, "run_gmat", runner)
    result = correct_with_gmat(
        document, scenario, candidate, config, "GmatConsole", "startup.txt", tmp_path / "shoot"
    )
    assert not result.converged and result.rounds == on_call
    assert result.reason is not None and f"round {on_call} (flight) failed" in result.reason
    assert result.gmat_runs == fake.calls == on_call - 1
    assert (result.closest_flight is not None) == (on_call > 1)
    saved = json.loads((tmp_path / "shoot" / "correction.json").read_text())
    assert saved["reason"] == result.reason


def test_a_run_refused_before_gmat_starts_is_not_counted(tmp_path: Path) -> None:
    """The real run_gmat with no startup file refuses before spawning anything."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    result = correct_with_gmat(
        document, scenario, candidate, config, tmp_path / "no-such-console", None,
        tmp_path / "shoot",
    )
    assert not result.converged and result.rounds == 1 and result.gmat_runs == 0
    assert result.reason is not None and "startup file" in result.reason


def test_an_exact_tie_on_the_first_passage_goes_to_the_smaller_t_f_miss(
    tmp_path: Path, monkeypatch
) -> None:
    """A first passage before the last burn is the same in every round, to the bit."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    _, run = _corrector(monkeypatch, config, Model(impulse_scale=0.9, graze_at_s=600.0))
    result = run(scenario, document, candidate, tmp_path / "shoot", max_rounds=2)
    approaches = [step.closest_approach_km for step in result.log if step.kind == "flight"]
    assert len(approaches) == 2 and approaches[0] == approaches[1]
    assert result.log[1].miss_at_arrival_km < result.log[0].miss_at_arrival_km
    assert result.reason is not None and "round 2," in result.reason
    assert result.closest_flight is not None
    assert result.closest_flight.metadata["gmat_correction"]["round"] == 2


def test_the_closest_first_passage_wins_over_the_smaller_t_f_miss(
    tmp_path: Path, monkeypatch
) -> None:
    """Round 1's earlier pass is deeper, round 2's t_f miss smaller; neither stops early."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    fake = DeeperFirstGraze(config, Model(impulse_scale=0.9, graze_at_s=1500.0))
    monkeypatch.setattr(gmat_shoot, "run_gmat", fake)
    result = correct_with_gmat(
        document, scenario, candidate, config, "GmatConsole", "startup.txt", tmp_path / "shoot",
        max_rounds=2,
    )
    assert result.log[0].closest_approach_km < result.log[1].closest_approach_km
    assert result.log[1].miss_at_arrival_km < result.log[0].miss_at_arrival_km
    assert result.reason is not None and "round 1," in result.reason
    assert result.closest_flight is not None
    assert result.closest_flight.metadata["gmat_correction"]["round"] == 1


def test_a_run_that_fails_after_gmat_ran_is_counted(tmp_path: Path, monkeypatch) -> None:
    """run_gmat can raise after the process exits, writing stdout: the run still happened."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    fake = FakeGmat(config, Model())

    def ran_then_failed(*args, **kwargs):
        fake(*args, **kwargs)
        raise IsADirectoryError(21, "Is a directory", "stdout.txt")

    monkeypatch.setattr(gmat_shoot, "run_gmat", ran_then_failed)
    result = correct_with_gmat(
        document, scenario, candidate, config, "GmatConsole", "startup.txt", tmp_path / "shoot"
    )
    assert result.gmat_runs == fake.calls == 1
    assert result.reason is not None
    assert result.reason.startswith("round 1 (flight) failed: IsADirectoryError: ")


def test_a_fault_in_this_code_says_what_it_is_and_keeps_its_traceback(
    tmp_path: Path, monkeypatch
) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    real = gmat_shoot.parse_gmat_script

    def misnamed(path, cfg, **options):
        return real(Path(str(path) + ".typo"), cfg, **options)

    monkeypatch.setattr(gmat_shoot, "parse_gmat_script", misnamed)
    _, run = _corrector(monkeypatch, config, Model())
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert result.reason is not None
    assert result.reason.startswith("round 1 (flight) failed: FileNotFoundError: ")
    assert result.log[-1].error is not None and result.log[-1].error.startswith("FileNotFoundError")
    saved = json.loads((tmp_path / "shoot" / "correction.json").read_text())
    assert "Traceback" in saved["traceback"] and "misnamed" in saved["traceback"]


@pytest.mark.parametrize("error", [TypeError("a bug"), KeyboardInterrupt()], ids=["TypeError", "KeyboardInterrupt"])
def test_anything_else_escapes(tmp_path: Path, monkeypatch, error: BaseException) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)

    def raising(*args, **kwargs):
        raise error

    monkeypatch.setattr(gmat_shoot, "run_gmat", raising)
    with pytest.raises(type(error)):
        correct_with_gmat(
            document, scenario, candidate, config, "GmatConsole", "startup.txt",
            tmp_path / "shoot",
        )


def test_a_legal_least_norm_start_never_calls_slsqp(monkeypatch) -> None:
    """Both impulses well under the cap: the unconstrained least-norm change is taken."""
    schedule = [(300.0, np.array([0.1, 0.0, 0.0])), (900.0, np.array([0.1, 0.0, 0.0]))]
    jacobian = 1000.0 * np.eye(3)

    def refused(*args, **kwargs):
        raise AssertionError("SLSQP called on a legal least-norm start")

    monkeypatch.setattr(gmat_shoot, "minimize", refused)
    stepped = gmat_shoot._least_norm_two(
        schedule, np.array([-0.004, 0.0, 0.0]), jacobian, jacobian,
        gmat_shoot._Limits(0.249, 2.95),
    )
    assert stepped is not None
    assert np.allclose(stepped[1][1], [0.100002, 0.0, 0.0], rtol=0.0, atol=1.0e-12)


def test_the_traceback_keeps_the_cause_and_drops_absolute_paths(
    tmp_path: Path, monkeypatch
) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)

    def raising(*args, **kwargs):
        try:
            raise ZeroDivisionError("the root cause")
        except ZeroDivisionError as cause:
            raise DynamicsError("no step") from cause

    monkeypatch.setattr(gmat_shoot, "_two_body_jacobians", raising)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert result.traceback is not None
    assert "ZeroDivisionError: the root cause" in result.traceback
    assert 'File "orbitwin/gmat_shoot.py", line ' in result.traceback
    assert str(Path(gmat_shoot.__file__).resolve()) not in result.traceback


def test_a_value_error_from_this_code_escapes(tmp_path: Path, monkeypatch) -> None:
    """Not GMAT's or the file system's: a fault to see, not a round that failed."""
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)

    def raising(*args, **kwargs):
        raise ValueError("a bug")

    monkeypatch.setattr(gmat_shoot, "_two_body_jacobians", raising)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    with pytest.raises(ValueError, match="a bug"):
        run(scenario, document, candidate, tmp_path / "shoot")


def _console(tmp_path: Path, body: str) -> tuple[Path, Path]:
    console = tmp_path / "console.sh"
    console.write_text("#!/bin/sh\n" + body)
    console.chmod(0o755)
    startup = tmp_path / "startup.txt"
    startup.write_text("OUTPUT_PATH = ./\n")
    return console, startup


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        # Exits 0, then stdout.txt cannot be written: GMAT ran.
        ('log="$4"; mkdir -p "$(dirname "$log")/stdout.txt"\nexit 0\n', "IsADirectoryError"),
        # Prints a byte that is not UTF-8 and writes no report.
        ("printf 'GMAT \\377\\376 output\\n'\nexit 0\n", "report was not created"),
    ],
    ids=["stdout-unwritable", "not-utf8"],
)
def test_a_real_run_that_fails_after_gmat_ran_is_counted(
    tmp_path: Path, body: str, reason: str
) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)
    console, startup = _console(tmp_path, body)
    result = correct_with_gmat(
        document, scenario, candidate, config, console, startup, tmp_path / "shoot"
    )
    assert result.gmat_runs == 1 and result.rounds == 1
    assert result.reason is not None and reason in result.reason
    assert (tmp_path / "shoot" / "correction.json").is_file()


def test_an_arrival_in_the_step_after_a_burn_row_microseconds_short_is_read(
    tmp_path: Path,
) -> None:
    """Issue #106's case: GMAT's burn row 2e-6 s short of the summed goals, arrival 10 s on.

    Keeping only rows from the burn epoch exactly drops that row and its rebuilt
    post-burn one, and t_f falls before the first row left.
    """
    pursuer = keplerian_to_cartesian(HUMAN, MU)
    components = np.array([0.2, 0.05, 0.0])
    pre = propagate_universal(pursuer, 240.0, MU)
    post = StateVector(
        pre.r_km, as_vector3(pre.v + script_components_to_inertial(components, pre, *VNB))
    )
    meet = propagate_universal(post, 10.0, MU)
    across = np.cross(meet.r, meet.v)
    target = StateVector(meet.r_km, as_vector3(meet.v + across / np.linalg.norm(across)))
    header = "P.ElapsedSecs T.ElapsedSecs " + " ".join(
        f"{craft}.EarthMJ2000Eq.{axis}" for craft in "PT" for axis in ("X", "Y", "Z", "VX", "VY", "VZ")
    )
    lines = [header]
    for t in (0.0, 240.0, 480.0):
        state = propagate_universal(pursuer, t, MU) if t <= 240.0 else propagate_universal(
            post, t - 240.0, MU
        )
        other = propagate_universal(target, t - 250.0, MU)
        written = t - 2.0e-6 if t == 240.0 else t
        lines.append(" ".join(repr(float(v)) for v in (written, written, *state.r, *state.v,
                                                       *other.r, *other.v)))
    path = tmp_path / "report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    scenario = SimpleNamespace(pursuer=SimpleNamespace(name="P"), target=SimpleNamespace(name="T"),
                               alignment=None)
    config = SimpleNamespace(
        policy=SimpleNamespace(
            intercept_radius_km=5.0, central_body_radius_km=6378.1363, stop_depth_km=1.5
        )
    )
    flight = gmat_shoot._read_flight(
        path, scenario, config, (SimpleNamespace(time_s=240.0),),
        [FlownManeuver(240.0, as_vector3(components), VNB)], 250.0, None,
    )
    assert float(np.linalg.norm(flight.miss_km)) < 0.05


def test_the_traceback_writes_the_interpreter_s_paths_as_python(
    tmp_path: Path, monkeypatch
) -> None:
    config, scenario, document, candidate = _scenario(tmp_path, TWO_BURNS, 2400.0)

    def reading_nothing(*args, **kwargs):
        return (tmp_path / "no-such-file").read_text()  # raises inside pathlib

    monkeypatch.setattr(gmat_shoot, "_two_body_jacobians", reading_nothing)
    _, run = _corrector(monkeypatch, config, Model(pursuer_mu_scale=1.0 + 3.0e-4))
    result = run(scenario, document, candidate, tmp_path / "shoot")
    assert result.traceback is not None
    assert 'File "<python>/' in result.traceback and "pathlib" in result.traceback
    assert sys.base_prefix not in result.traceback
