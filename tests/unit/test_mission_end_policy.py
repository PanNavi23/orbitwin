from __future__ import annotations

import dataclasses

import pytest
from conftest import quick_config

from orbitwin.config import validate_config
from orbitwin.errors import ConfigurationError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import SYNTHESIZED_BURN_FRAME, Burn, Candidate
from orbitwin.script_renderer import assert_final_script_policy, render_candidate

# 2026-09-05 preliminary, measured against the organizer's own validator: it
# returned C = 23815.43 s (= 4TA) as the completion time for a run whose D =
# 2.394 km confirmed the 5 km entry at 5982.99 s. It reads the completion time
# off the last row of the report. A submitted script that runs to the horizon is
# therefore scored as a mission that never intercepted -- 22.2266 points on that
# scenario. Ending the sequence at the interception is what makes the last row
# the interception; a failing candidate still runs to 4TA, because there the
# distance term is Δr(4TA) and an early stop discards up to 50 points.


def _config(**policy: object):
    """quick_config plus the pursuer name the open template cannot supply."""
    base = dataclasses.replace(quick_config(), pursuer_name="Pursuer", target_name="AlienSC")
    if policy:
        return dataclasses.replace(base, policy=dataclasses.replace(base.policy, **policy))
    return base


def _candidate(intercept_time_s: float | None, *, successful: bool) -> Candidate:
    burn = Burn(
        time_s=1000.0,
        dv_inertial_km_s=(0.05, 0.0, 0.0),
        dv_script_km_s=(0.05, 0.0, 0.0),
        slot_index=0,
        # Written as given, not derived; the recorded frame is what they claim.
        frame=SYNTHESIZED_BURN_FRAME,
    )
    return Candidate(
        burns=(burn,),
        intercept_time_s=intercept_time_s,
        minimum_distance_km=0.001 if successful else 900.0,
        distance_at_completion_km=5.0 if successful else 900.0,
        successful=successful,
        analytical_endpoint_error_km=0.001 if successful else 900.0,
        branch="test",
    )


def test_successful_candidate_ends_on_the_entry_row(no_maneuver_script) -> None:
    """The mission ends on the report row that first falls inside the sphere.

    GMAT phases its rows from the last Maneuver at Propagator.MaxStep, so the
    entry row sits at last_burn + ceil((tint - last_burn) / step) * step. Ending
    anywhere else leaves the last row outside: measured on the 6-burn 98.32
    candidate, a 1 s tail ended 0.45 s past the row and a 2 s tail 1.45 s past
    it, both outside a sphere only one row was ever inside.
    """
    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    candidate = _candidate(4000.0, successful=True)  # burn at 1000 s
    result = render_candidate(document, scenario, candidate, config)
    step = float(scenario.propagator_fields["Prop.MaxStep"])
    offset = (result.mission_end_s - 1000.0) / step
    assert offset == pytest.approx(round(offset)), "mission end is not on the report grid"
    assert result.mission_end_s >= 4000.0
    assert result.mission_end_s - 4000.0 < step
    assert result.mission_end_s < scenario.max_time_s
    assert sum(result.segment_durations_s) == pytest.approx(result.mission_end_s)


def test_failed_candidate_still_runs_to_the_horizon(no_maneuver_script) -> None:
    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    candidate = _candidate(None, successful=False)
    result = render_candidate(document, scenario, candidate, config)
    assert result.mission_end_s == pytest.approx(scenario.max_time_s)


def test_horizon_policy_restores_the_old_behaviour(no_maneuver_script) -> None:
    config = _config(mission_end_policy="horizon")
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    result = render_candidate(document, scenario, _candidate(4000.0, successful=True), config)
    assert result.mission_end_s == pytest.approx(scenario.max_time_s)


def test_interception_past_the_horizon_is_clamped(no_maneuver_script) -> None:
    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    late = scenario.max_time_s - 0.1
    result = render_candidate(document, scenario, _candidate(late, successful=True), config)
    assert result.mission_end_s == pytest.approx(scenario.max_time_s)


def test_a_shortened_mission_still_passes_the_policy_guard(tmp_path, no_maneuver_script) -> None:
    # The guard refuses a script that runs *past* Tmax. One that stops at the
    # interception is shorter, and has to stay acceptable -- it is the shape
    # every successful submission now takes.
    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    result = render_candidate(document, scenario, _candidate(4000.0, successful=True), config)
    generated = tmp_path / "candidate.script"
    generated.write_bytes(result.script_bytes)
    assert_final_script_policy(no_maneuver_script, generated, config)


def test_unknown_mission_end_policy_is_refused() -> None:
    broken = _config(mission_end_policy="whenever")
    with pytest.raises(ConfigurationError, match="mission_end_policy"):
        validate_config(broken)


def test_a_burn_after_the_interception_still_schedules(no_maneuver_script) -> None:
    """The tail runs from the later of the interception and the last maneuver.

    §5 counts the delta-v of a burn scheduled after tint, so such a candidate is
    legal and has to render. Measuring the tail from tint alone asks for a
    segment that ends before the maneuver opening it: on the fake-GMAT
    certification fixture, tint = 5 s against a burn at 1100 s produced a
    -1095 s segment and a PolicyError.
    """
    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    candidate = _candidate(500.0, successful=True)  # burn at 1000 s, interception at 500 s
    result = render_candidate(document, scenario, candidate, config)
    assert result.mission_end_s == pytest.approx(1000.0 + config.policy.interception_tail_s)
    assert all(duration > 0.0 for duration in result.segment_durations_s)
    assert sum(result.segment_durations_s) == pytest.approx(result.mission_end_s)


def test_the_tail_is_short_because_every_second_costs_score() -> None:
    # A validator that reads the completion time off the last row reads
    # t_int + interception_tail_s. On the 2026-09-05 coefficients 20 s costs
    # 0.2025 points, so the tail is its own setting rather than final_padding_s,
    # which answers a different question and defaults to 20.
    config = _config()
    assert config.policy.interception_tail_s < config.policy.final_padding_s
    assert config.policy.interception_tail_s >= config.policy.min_propagate_duration_s


def test_a_tail_below_the_propagate_floor_is_refused() -> None:
    broken = _config(interception_tail_s=1.0e-9)
    with pytest.raises(ConfigurationError, match="interception_tail_s"):
        validate_config(broken)


def test_the_tail_never_falls_below_the_propagate_floor(no_maneuver_script) -> None:
    # Aligning to the grid can ask for a very short final segment when the
    # interception already sits almost on a row; it must still clear the floor
    # that keeps GMAT from hanging.
    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    step = float(scenario.propagator_fields["Prop.MaxStep"])
    for offset in (0.0, 1e-6, 0.5 * step, step - 1e-6):
        candidate = _candidate(1000.0 + 3 * step + offset, successful=True)
        result = render_candidate(document, scenario, candidate, config)
        assert all(
            duration >= config.policy.min_propagate_duration_s
            for duration in result.segment_durations_s
        )


def test_the_report_step_comes_from_the_propagator(no_maneuver_script) -> None:
    """The row spacing is Propagator.MaxStep, read from the script.

    Verified against a real GMAT R2026a run: with MaxStep = 1 the rows were
    1.000000 s apart, and the entry row sat exactly 3120 steps after the last
    maneuver. Getting this wrong puts the last row outside the sphere.
    """
    from orbitwin.script_renderer import _report_step_s

    config = _config()
    scenario, _ = parse_gmat_script(no_maneuver_script, config)
    assert _report_step_s(scenario) == pytest.approx(
        float(scenario.propagator_fields["Prop.MaxStep"])
    )


def test_the_report_step_is_the_lead_propagators(no_maneuver_script) -> None:
    """A Synchronized line reports at its first propagator's steps (R2026a)."""
    import dataclasses

    from orbitwin.script_renderer import _report_step_s

    scenario, _ = parse_gmat_script(no_maneuver_script, _config())
    last = scenario.propagate_segments[-1]
    fields = {"PH.MaxStep": "1", "PA.MaxStep": "60"}
    for order, step in ((("PH", "PA"), 1.0), (("PA", "PH"), 60.0), (("pa", "PH"), 60.0)):
        segment = dataclasses.replace(last, propagator_names=order)
        split = dataclasses.replace(
            scenario, propagator_fields=fields, propagate_segments=(segment,)
        )
        assert _report_step_s(split) == step


def test_an_unreadable_step_falls_back_to_the_tail(no_maneuver_script) -> None:
    # A propagator whose MaxStep cannot be read must not take the renderer down;
    # it falls back to interception_tail_s, which is why that setting stays.
    import dataclasses

    from orbitwin.script_renderer import _report_step_s

    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    for broken in ("not-a-number", "0", "-1"):
        blinded = dataclasses.replace(
            scenario, propagator_fields={"Prop.MaxStep": broken}
        )
        assert _report_step_s(blinded) is None
    missing = dataclasses.replace(scenario, propagator_fields={})
    assert _report_step_s(missing) is None
    # and the renderer still produces a legal schedule on the real scenario
    result = render_candidate(document, scenario, _candidate(4000.0, successful=True), config)
    assert all(d >= config.policy.min_propagate_duration_s for d in result.segment_durations_s)


def test_alignment_lands_on_the_row_gmat_will_write(no_maneuver_script) -> None:
    """Reproduce the arithmetic GMAT was observed to follow.

    Rows are phased from the last maneuver, so the first one at or after the
    interception is last_burn + ceil((tint - last_burn) / step) * step.
    """
    import math

    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    step = float(scenario.propagator_fields["Prop.MaxStep"])
    last_burn = 1000.0
    for intercept in (3000.4, 3000.5, 3000.6, 4321.75, 5000.999):
        result = render_candidate(
            document, scenario, _candidate(intercept, successful=True), config
        )
        steps = math.ceil((intercept - last_burn) / step - 1.0e-8)
        assert result.mission_end_s == pytest.approx(last_burn + steps * step)


def test_a_gmat_measured_stop_replaces_the_row_prediction(no_maneuver_script) -> None:
    """GMAT writes a row where Propagate stops, so stopping inside needs no prediction.

    The prediction above fails when InitialStepSize differs from MaxStep: on
    R2026a it ended a hit at 163.001 s with its last row 5.88 km out.
    """
    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    candidate = _candidate(4000.0, successful=True)
    candidate.metadata["gmat_event"] = {"stop_time_s": 4000.137}
    result = render_candidate(document, scenario, candidate, config)
    assert result.mission_end_s == pytest.approx(4000.137)
    # Inside before the last burn: that burn still flies, then the fixed tail.
    last_burn = max(burn.time_s for burn in candidate.burns)
    candidate.metadata["gmat_event"] = {"stop_time_s": last_burn - 1.0}
    result = render_candidate(document, scenario, candidate, config)
    assert result.mission_end_s == pytest.approx(last_burn + config.policy.interception_tail_s)
