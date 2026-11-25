from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, SearchConfig
from orbitwin.errors import PolicyError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import Burn, Candidate
from orbitwin.script_renderer import (
    assert_final_script_policy,
    render_candidate,
    write_rendered_candidate,
)


def failing_candidate() -> Candidate:
    """No interception, so the sequence runs to 4TA."""
    return Candidate(
        burns=(Burn(1000.0, (0.1, 0.2, 0.3), (0.11, 0.22, 0.33), 0),),
        intercept_time_s=None,
        minimum_distance_km=900.0,
        distance_at_completion_km=900.0,
        successful=False,
        analytical_endpoint_error_km=0.1,
        branch="test",
    )


def simple_candidate() -> Candidate:
    return Candidate(
        burns=(Burn(1000.0, (0.1, 0.2, 0.3), (0.11, 0.22, 0.33), 0),),
        intercept_time_s=5000.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="test",
        metadata={"arrival_time_s": 5000.0},
    )


def test_renderer_changes_only_whitelisted_lines(tmp_path: Path, workshop_script: Path) -> None:
    config = OrbitWinConfig(search=SearchConfig(two_burn_enabled=False))
    scenario, document = parse_gmat_script(workshop_script, config)
    output = tmp_path / "rendered.script"
    diff = tmp_path / "diff.txt"
    result = write_rendered_candidate(output, diff, document, scenario, simple_candidate(), config)
    assert result.changed_lines
    assert_final_script_policy(workshop_script, output, config)
    assert "ImpulsiveBurn1.Element1" in diff.read_text(encoding="utf-8")
    assert workshop_script.read_bytes() != output.read_bytes()


def test_policy_rejects_spacecraft_change(tmp_path: Path, workshop_script: Path) -> None:
    modified = tmp_path / "bad.script"
    modified.write_text(
        workshop_script.read_text(encoding="utf-8").replace(
            "HumanSC.SMA = 6900;", "HumanSC.SMA = 7001;"
        ),
        encoding="utf-8",
    )
    with pytest.raises(PolicyError):
        assert_final_script_policy(workshop_script, modified, OrbitWinConfig())


def test_renderer_zeros_unused_burn_slot(tmp_path: Path, workshop_script: Path) -> None:
    config = OrbitWinConfig()
    scenario, document = parse_gmat_script(workshop_script, config)
    output = tmp_path / "one-burn.script"
    write_rendered_candidate(output, tmp_path / "diff", document, scenario, simple_candidate(), config)
    # Asserted through the parser rather than against a literal: the intent is
    # that the unused slot carries an exact zero vector, not that zero is
    # spelled a particular way.
    generated, _ = parse_gmat_script(output, config)
    unused = generated.burn_slots[1]
    assert unused.name == "ImpulsiveBurn2"
    assert (unused.element1_km_s, unused.element2_km_s, unused.element3_km_s) == (0.0, 0.0, 0.0)


def test_renderer_rejects_more_burns_than_slots(workshop_script: Path, tmp_path: Path) -> None:
    config = OrbitWinConfig()
    scenario, document = parse_gmat_script(workshop_script, config)
    burns = tuple(Burn(200.0 * i, (0.1, 0, 0), (0.1, 0, 0), i) for i in range(3))
    candidate = Candidate(
        burns=burns,
        intercept_time_s=1000.0,
        minimum_distance_km=0.0,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.0,
        branch="too-many",
    )
    with pytest.raises(PolicyError, match="needs 3 burns"):
        write_rendered_candidate(
            tmp_path / "bad.script", tmp_path / "bad.diff", document, scenario, candidate, config
        )


def _rewrite(path: Path, tmp_path: Path, name: str, old: str, new: str) -> Path:
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    target = tmp_path / name
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return target


def test_policy_rejects_a_burn_above_the_formal_limit(
    tmp_path: Path, workshop_script: Path
) -> None:
    # Only a whitelisted line changes, so every "what did not move" comparison
    # passes. The value itself is 66x the 1.5 km/s formal limit.
    generated = _rewrite(
        workshop_script, tmp_path, "over.script",
        "ImpulsiveBurn1.Element1 = 0.5983;", "ImpulsiveBurn1.Element1 = 99;",
    )
    with pytest.raises(PolicyError, match="exceeds the formal limit"):
        assert_final_script_policy(workshop_script, generated, OrbitWinConfig())


def test_policy_uses_the_formal_limit_not_the_safety_margin(
    tmp_path: Path, workshop_script: Path
) -> None:
    # 1.4999 km/s sits between the production margin (1.495) and the rule
    # boundary (1.5). The guard enforces the rules, so it must accept this.
    generated = _rewrite(
        workshop_script, tmp_path, "edge.script",
        "ImpulsiveBurn1.Element1 = 0.5983;", "ImpulsiveBurn1.Element1 = 1.4999;",
    )
    assert_final_script_policy(workshop_script, generated, OrbitWinConfig())


def test_policy_rejects_burns_closer_than_the_minimum_separation(
    tmp_path: Path, workshop_script: Path
) -> None:
    # The middle Propagate is the gap between ImpulsiveBurn1 and ImpulsiveBurn2.
    generated = _rewrite(
        workshop_script, tmp_path, "close.script",
        "HumanSC.ElapsedSecs = 3728.5, AlienSC.ElapsedSecs = 3728.5",
        "HumanSC.ElapsedSecs = 40, AlienSC.ElapsedSecs = 40",
    )
    with pytest.raises(PolicyError, match="below the formal"):
        assert_final_script_policy(workshop_script, generated, OrbitWinConfig())


def test_policy_rejects_propagation_beyond_the_mission_horizon(
    tmp_path: Path, workshop_script: Path
) -> None:
    scenario, _ = parse_gmat_script(workshop_script, OrbitWinConfig())
    generated = _rewrite(
        workshop_script, tmp_path, "long.script",
        "HumanSC.ElapsedSecs = 3728.5, AlienSC.ElapsedSecs = 3728.5",
        f"HumanSC.ElapsedSecs = {scenario.max_time_s:.6f}, "
        f"AlienSC.ElapsedSecs = {scenario.max_time_s:.6f}",
    )
    with pytest.raises(PolicyError, match="mission horizon"):
        assert_final_script_policy(workshop_script, generated, OrbitWinConfig())


def test_renderer_refuses_a_burn_at_exactly_mission_start(
    tmp_path: Path, workshop_script: Path
) -> None:
    # GMAT R2026a does not reject ElapsedSecs = 0. It hangs: two 60 s runs of a
    # script whose first Propagate goal is 0 never reached the mission sequence,
    # while 0.001 completed in 1 s. A submitted script that hangs stalls the
    # organizer's validator instead of failing cleanly.
    config = OrbitWinConfig()
    scenario, document = parse_gmat_script(workshop_script, config)
    at_start = Candidate(
        burns=(Burn(0.0, (0.3, 0.0, 0.0), (0.3, 0.0, 0.0), 0),),
        intercept_time_s=3000.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="burn-at-t0",
        metadata={"event_refined": True, "arrival_time_s": 3000.0},
    )
    with pytest.raises(PolicyError, match="hangs on a zero-length"):
        write_rendered_candidate(
            tmp_path / "t0.script", tmp_path / "t0.diff", document, scenario, at_start, config
        )


def test_policy_rejects_a_zero_length_propagate_in_the_final_script(
    tmp_path: Path, workshop_script: Path
) -> None:
    text = workshop_script.read_text(encoding="utf-8")
    generated = tmp_path / "zero.script"
    generated.write_text(
        text.replace(
            "HumanSC.ElapsedSecs = 1000, AlienSC.ElapsedSecs = 1000",
            "HumanSC.ElapsedSecs = 0, AlienSC.ElapsedSecs = 0",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(PolicyError, match="hangs GMAT"):
        assert_final_script_policy(workshop_script, generated, OrbitWinConfig())


def test_search_never_proposes_a_burn_the_template_cannot_express(
    workshop_script: Path,
) -> None:
    from orbitwin.search import _time_samples

    config = OrbitWinConfig(search=SearchConfig(one_burn_samples=64, one_burn_grid_burns=8))
    scenario, _ = parse_gmat_script(workshop_script, config)
    floor_s = config.policy.min_propagate_duration_s
    burn_times = [burn_time for burn_time, _ in _time_samples(scenario, config)]
    assert burn_times
    assert min(burn_times) >= floor_s - 1e-12, min(burn_times)


def test_candidate_identity_distinguishes_the_slot_assignment() -> None:
    # select_candidates deduplicates by candidate_id. With slot_index outside
    # the identity, the same impulse in two different slots collided and one
    # variant was silently dropped.
    def at(slot_index: int) -> Candidate:
        return Candidate(
            burns=(Burn(1000.0, (0.3, 0.0, 0.0), (0.3, 0.0, 0.0), slot_index),),
            intercept_time_s=3000.0,
            minimum_distance_km=0.1,
            distance_at_completion_km=5.0,
            successful=True,
            analytical_endpoint_error_km=0.1,
            branch="b",
            metadata={"arrival_time_s": 3000.0},
        )

    assert at(0).candidate_id != at(1).candidate_id


def test_a_late_burn_renders_in_a_later_slot_but_not_in_slot_zero(
    tmp_path: Path, workshop_script: Path
) -> None:
    # In slot 0 the remaining slot must be scheduled after it and runs past
    # Tmax. In slot 1, with slot 0 carrying an exact zero vector, the same
    # trajectory is expressible.
    from orbitwin.errors import PolicyError as _PolicyError
    from orbitwin.script_renderer import render_candidate
    from orbitwin.search import slot_assignment_variants

    config = OrbitWinConfig()
    scenario, document = parse_gmat_script(workshop_script, config)
    late = scenario.max_time_s - 100.0
    candidate = Candidate(
        burns=(Burn(late, (0.2, 0.0, 0.0), (0.2, 0.0, 0.0), 0),),
        intercept_time_s=scenario.max_time_s - 50.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="late",
        metadata={"event_refined": True, "arrival_time_s": scenario.max_time_s - 50.0},
    )
    with pytest.raises(_PolicyError):
        render_candidate(document, scenario, candidate, config)

    variants = slot_assignment_variants(scenario, config, [candidate])
    assert variants, "a two-slot template must offer an alternative assignment"
    rendered = render_candidate(document, scenario, variants[0], config)
    assert variants[0].burns[0].slot_index == 1
    # The unused leading slot takes the earliest legal epoch and a zero vector.
    assert rendered.burn_times_s[0] == pytest.approx(config.policy.min_propagate_duration_s)
    assert rendered.mission_end_s <= scenario.max_time_s + 1e-8


def test_time_samples_cover_the_whole_legal_window(workshop_script: Path) -> None:
    from orbitwin.search import _time_samples

    config = OrbitWinConfig(search=SearchConfig(one_burn_samples=0, one_burn_grid_burns=8))
    scenario, _ = parse_gmat_script(workshop_script, config)
    latest_legal = scenario.max_time_s - config.search.min_tof_s
    burn_times = [burn_time for burn_time, _ in _time_samples(scenario, config)]
    assert max(burn_times) == pytest.approx(latest_legal, rel=1e-9)


def test_slot_variant_does_not_inherit_verification_state(workshop_script: Path) -> None:
    # A variant is a different script. dataclasses.replace is a shallow copy, so
    # without an explicit reset a variant of a GMAT-verified candidate would
    # carry verified_by_gmat, its stage and its run id having never been run.
    from orbitwin.models import CandidateStage
    from orbitwin.search import slot_assignment_variants

    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    source = Candidate(
        burns=(Burn(2000.0, (0.3, 0.0, 0.0), (0.3, 0.0, 0.0), 0),),
        intercept_time_s=6000.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="x",
        metadata={
            "arrival_time_s": 6000.0,
            "event_refined": True,
            "gmat_event": {"stop_time_s": 6000.37},
        },
    )
    source.gmat_run_id = "RUN-1"
    source.promote(CandidateStage.PUBLIC_GMAT_VERIFIED, run_id="RUN-1")
    assert source.verified_by_gmat is True

    variant = slot_assignment_variants(scenario, config, [source])[0]
    assert variant.verified_by_gmat is False
    assert variant.gmat_run_id is None
    assert variant.stage is CandidateStage.ANALYTICAL
    # Its stop would end the variant on a trajectory GMAT never flew.
    assert "gmat_event" not in variant.metadata
    assert variant.metadata["derived_from"] == source.candidate_id

    # promote() mutates stage_evidence in place, so the dicts must not be shared.
    assert variant.stage_evidence is not source.stage_evidence
    assert variant.score_components is not source.score_components
    assert variant.metadata is not source.metadata
    variant.promote(CandidateStage.RENDERABLE, note="variant")
    assert "renderable" not in source.stage_evidence


def test_slot_variant_reproduces_the_same_inertial_impulse(workshop_script: Path) -> None:
    # The variant re-expresses the impulse in the other slot's frame. If that
    # conversion were wrong the variant would fly a different trajectory while
    # claiming the source's event data.
    import numpy as np

    from orbitwin.frames import script_components_to_inertial
    from orbitwin.search import slot_assignment_variants
    from orbitwin.trajectory import state_with_burns_at

    config = OrbitWinConfig()
    scenario, _ = parse_gmat_script(workshop_script, config)
    source = Candidate(
        burns=(Burn(2000.0, (0.31, -0.12, 0.07), (0.0, 0.0, 0.0), 0),),
        intercept_time_s=6000.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="x",
        metadata={"arrival_time_s": 6000.0, "event_refined": True},
    )
    for variant in slot_assignment_variants(scenario, config, [source]):
        for burn in variant.burns:
            slot = scenario.burn_slots[burn.slot_index]
            state = state_with_burns_at(
                scenario.pursuer.initial_state,
                variant.burns,
                burn.time_s,
                scenario.mu_km3_s2,
                include_burn_at_time=False,
            )
            recovered = script_components_to_inertial(
                np.asarray(burn.dv_script_km_s),
                state,
                slot.coordinate_system,
                slot.origin,
                slot.axes,
            )
            assert np.allclose(recovered, np.asarray(burn.dv_inertial_km_s), atol=1e-12)


def test_rendered_numbers_read_back_as_the_same_double() -> None:
    # ".16g" was not enough: 44.2% of random doubles came back different, and 12
    # of the 33 burn components from one real search did. The script would then
    # carry an impulse the candidate does not hold, and every number recorded
    # about that candidate would describe the other one.
    import random
    import struct

    from orbitwin.script_renderer import _format_float

    rng = random.Random(20260820)
    checked = 0
    for _ in range(20000):
        value = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if value != value or abs(value) in (float("inf"),) or abs(value) > 1e300:
            continue
        if abs(value) < 5.0e-16:
            continue
        checked += 1
        assert float(_format_float(value)) == value, value
    assert checked > 5000, "the sample should exercise a wide range of doubles"

    # Readable as well as exact.
    assert _format_float(0.6388) == "0.6388"
    assert _format_float(3899.5) == "3899.5"


def test_a_rendered_burn_survives_the_round_trip_through_the_script(
    tmp_path: Path, workshop_script: Path
) -> None:
    # End to end: what the candidate holds is what the script says.
    config = OrbitWinConfig()
    scenario, document = parse_gmat_script(workshop_script, config)
    awkward = Candidate(
        burns=(
            Burn(
                1000.0,
                (0.12345678901234566, -0.98765432109876543, 0.3333333333333333),
                (0.12345678901234566, -0.98765432109876543, 0.3333333333333333),
                0,
            ),
        ),
        intercept_time_s=5000.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="awkward",
        metadata={"arrival_time_s": 5000.0, "event_refined": True},
    )
    output = tmp_path / "awkward.script"
    write_rendered_candidate(output, tmp_path / "awkward.diff", document, scenario, awkward, config)
    generated, _ = parse_gmat_script(output, config)
    slot = generated.burn_slots[0]
    assert (slot.element1_km_s, slot.element2_km_s, slot.element3_km_s) == awkward.burns[0].dv_script_km_s


def test_the_diagnostic_sibling_never_drops_a_line(
    tmp_path: Path, workshop_script: Path
) -> None:
    """The sibling may add and disable. It may not delete.

    Certification proves interception by running the sibling and then argues
    the result carries over to the submission. That argument holds only if the
    sibling contains the submitted bytes -- so it used to assert the sibling was
    a pure insertion, which was true until the sibling had to comment out the
    plot subscribers that stop GmatConsole running the organizer's template at
    all.

    Commented out is not deleted, and that is the property now: every line of
    the submitted script appears in the sibling either verbatim or behind the
    neutralisation marker, and the only additions are the diagnostic block.
    """
    from orbitwin.diagnostic import NEUTRALISED_PREFIX, add_diagnostic_report

    config = OrbitWinConfig()
    scenario, document = parse_gmat_script(workshop_script, config)
    final = tmp_path / "final.script"
    write_rendered_candidate(final, tmp_path / "final.diff", document, scenario, simple_candidate(), config)

    final_lines = final.read_text(encoding="utf-8").splitlines(keepends=True)
    final_scenario, final_document = parse_gmat_script(final, config)
    diagnostic = add_diagnostic_report(
        final_document, final_scenario, config.gmat.diagnostic_report_name
    )
    sibling_lines = diagnostic.script_bytes.decode("utf-8").splitlines(keepends=True)
    assert diagnostic.neutralised, "the workshop template draws, so something must be disabled"

    # Every submitted line is still there, verbatim or commented, in order.
    position = 0
    for line in final_lines:
        while position < len(sibling_lines):
            candidate = sibling_lines[position]
            position += 1
            if candidate == line or candidate == NEUTRALISED_PREFIX + line:
                break
        else:
            raise AssertionError(f"the sibling dropped {line!r}")

    # And the only lines it adds are the diagnostic block.
    added = [
        line
        for line in sibling_lines
        if line not in final_lines and not line.startswith(NEUTRALISED_PREFIX)
    ]
    assert added, "the diagnostic block is the point of the sibling"
    for line in added:
        assert (
            "OrbitWinDiagnostic" in line
            or line.lstrip().startswith("% OrbitWin internal diagnostic")
            or not line.strip()
        ), line


def test_scientific_notation_components_render_and_reparse(
    tmp_path: Path, workshop_script: Path
) -> None:
    # repr() emits 1e-05 for small components. GMAT's own scripts use the same
    # form -- a GUI save writes an eccentricity like 3.8e-16 that way -- and stock
    # R2026a runs a script whose burn components are written this way.
    config = OrbitWinConfig()
    scenario, document = parse_gmat_script(workshop_script, config)
    components = (2.5e-9, -7.25e-12, 1.0e-5)
    tiny = Candidate(
        burns=(Burn(1000.0, components, components, 0),),
        intercept_time_s=5000.0,
        minimum_distance_km=0.1,
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=0.1,
        branch="scientific",
        metadata={"arrival_time_s": 5000.0, "event_refined": True},
    )
    output = tmp_path / "sci.script"
    write_rendered_candidate(output, tmp_path / "sci.diff", document, scenario, tiny, config)
    assert "Element1 = 2.5e-09;" in output.read_text(encoding="utf-8")
    generated, _ = parse_gmat_script(output, config)
    slot = generated.burn_slots[0]
    assert (slot.element1_km_s, slot.element2_km_s, slot.element3_km_s) == components


def test_a_failing_mission_runs_to_the_full_horizon(
    tmp_path: Path, workshop_script: Path
) -> None:
    """A mission that never intercepts has to carry Δr(4TA).

    初賽規則 §4: Tteam = min(tint, 4TA) and Δrmin = max(Δr(Tteam), 5). With no
    interception Tteam is 4TA and the distance term is Δr(4TA) -- up to 50
    points, the largest term in the formula -- which a report ending short does
    not contain. A *successful* candidate stops at the interception instead;
    test_mission_end_policy.py has why, and it is worth 22 points.
    """
    config = OrbitWinConfig()
    scenario, document = parse_gmat_script(workshop_script, config)
    result = write_rendered_candidate(
        tmp_path / "full.script",
        tmp_path / "full.diff",
        document,
        scenario,
        failing_candidate(),
        config,
    )
    assert result.mission_end_s == pytest.approx(scenario.max_time_s)
    assert sum(result.segment_durations_s) == pytest.approx(scenario.max_time_s)


def _searching_config() -> OrbitWinConfig:
    return OrbitWinConfig(
        search=SearchConfig(
            one_burn_samples=24, one_burn_grid_burns=4, one_burn_grid_arrivals=6,
            two_burn_enabled=False, event_refine_top=2, keep_candidates=8, workers=1,
        )
    )


def test_the_rendered_segments_sum_to_the_horizon_exactly(
    different_plane_script: Path, high_plane_script: Path
) -> None:
    """GMAT adds up the ElapsedSecs values, not our mission_end_s.

    Each Propagate advances from wherever the previous one left off, so the
    epoch the mission ends at is the left-to-right sum of the numbers in the
    file. 初賽規則 §4 scores Δr(4TA) when there is no interception, and a sum
    that lands short does not contain it. Floating point does not promise that
    a telescoping sum is exact, so it is measured rather than assumed -- and on
    both bundled scenarios it is exact. Even where it is not, one ulp of 34000 s
    is 3.6e-12 s, which at 7.5 km/s is 27 picometres of separation.
    """
    import re

    from orbitwin.search import zero_burn_candidate

    config = OrbitWinConfig(search=SearchConfig(one_burn_samples=0, two_burn_enabled=False))
    cases: list[tuple[str, Path, object]] = []
    for script in (different_plane_script, high_plane_script):
        scenario, document = parse_gmat_script(script, config)
        # The horizon is set the same way whatever the candidate holds, so the
        # baseline exercises it without a search, and simple_candidate() adds
        # the shape where a real impulse splits the segments.
        cases.append((script.name, scenario, document, zero_burn_candidate(scenario, config)))
    scenario, document = parse_gmat_script(different_plane_script, config)
    cases.append((different_plane_script.name, scenario, document, simple_candidate()))

    for name, scenario, document, candidate in cases:
        rendered = render_candidate(document, scenario, candidate, config)
        durations = [
            float(value)
            for value in re.findall(
                rf"{scenario.pursuer.name}\.ElapsedSecs\s*=\s*([-+0-9.eE]+)",
                rendered.script_bytes.decode("utf-8"),
            )
        ]
        assert len(durations) == len(scenario.propagate_segments)
        total = 0.0
        for duration in durations:
            total += duration
        assert total == rendered.mission_end_s, f"{name}: {total!r} against {scenario.max_time_s!r}"


def test_the_policy_guard_counts_a_zero_vector_slot_as_a_maneuver(
    tmp_path: Path, different_plane_script: Path
) -> None:
    """An unused slot still carries a Maneuver command.

    初賽規則 §2 requires 100 s between maneuvers and says nothing about their
    magnitude, so a slot holding an exact zero vector is one for the purposes of
    that rule until the organizer says otherwise. validate_candidate cannot see
    it -- it reads Candidate.burns, which holds only the impulses the search
    placed -- so the guard that reads the rendered script is what covers it.
    """
    import re

    from orbitwin.search import search_one_burn

    config = _searching_config()
    scenario, document = parse_gmat_script(different_plane_script, config)
    found = [c for c in search_one_burn(scenario, config) if c.successful]
    assert found
    candidate = found[0]
    assert len(candidate.burns) == 1, "the second slot is the zero-vector one"
    rendered = render_candidate(document, scenario, candidate, config)
    text = rendered.script_bytes.decode("utf-8")

    clean = tmp_path / "clean.script"
    clean.write_bytes(rendered.script_bytes)
    assert_final_script_policy(different_plane_script, clean, config)

    # The segment between the two Maneuver commands is the separation, and one
    # of them fires an exact zero vector. Replace the whole number, not its
    # leading digits.
    middle = rendered.segment_durations_s[1]
    tampered = tmp_path / "tampered.script"
    tampered.write_bytes(
        re.sub(
            r"ElapsedSecs = " + re.escape(repr(middle)) + r"(?![0-9.])",
            "ElapsedSecs = 99.0",
            text,
        ).encode("utf-8")
    )
    assert "99.0" in tampered.read_text(encoding="utf-8"), "the tamper did not apply"
    with pytest.raises(PolicyError, match="separation"):
        assert_final_script_policy(different_plane_script, tampered, config)


def test_the_only_value_that_does_not_round_trip_is_the_one_that_is_meant_not_to(
    tmp_path: Path, workshop_script: Path
) -> None:
    """_format_float promises the script reads back as the same double.

    It does, except under 5e-16, which is flushed to zero. A Lambert solve
    leaves a component of that size wherever the transfer is very nearly
    planar, and 1e-16 km/s is 1e-13 m/s -- below anything §5 sums or any report
    resolves, but printed in a submitted script it looks like a number someone
    chose. The exception is pinned here so it stays deliberate.
    """
    config = OrbitWinConfig(search=SearchConfig(one_burn_samples=0, two_burn_enabled=False))
    scenario, document = parse_gmat_script(workshop_script, config)

    def _round_trip(component: float) -> float:
        impulse = (0.5, component, 0.0)
        candidate = Candidate(
            burns=(Burn(1000.0, impulse, impulse, 0),),
            intercept_time_s=4000.0, minimum_distance_km=0.1,
            distance_at_completion_km=5.0, successful=True,
            analytical_endpoint_error_km=None, branch="round-trip",
            metadata={"arrival_time_s": 4000.0},
        )
        path = tmp_path / f"r{component!r}.script"
        path.write_bytes(render_candidate(document, scenario, candidate, config).script_bytes)
        return parse_gmat_script(path, config)[0].burn_slots[0].element2_km_s

    # Above the floor, exactly. Including values .16g would have mangled.
    for component in (1.0e-15, 5.0e-16, 0.16377637148975283, -0.04052702449617837, 0.6388):
        assert _round_trip(component) == component, repr(component)

    # Below it, deliberately not.
    for component in (1.0e-16, -4.9e-16, 1.0e-30):
        assert _round_trip(component) == 0.0, repr(component)


@pytest.mark.parametrize("limit_km_s, refused", [(0.41, True), (0.42, False)])
def test_the_policy_guard_checks_the_total_delta_v(
    tmp_path: Path, workshop_script: Path, limit_km_s: float, refused: bool
) -> None:
    """The last gate before upload: §4 note 1(c) disqualifies over 3 km/s."""
    config = OrbitWinConfig(search=SearchConfig(two_burn_enabled=False))
    scenario, document = parse_gmat_script(workshop_script, config)
    output = tmp_path / "rendered.script"
    write_rendered_candidate(
        output, tmp_path / "diff.txt", document, scenario, simple_candidate(), config
    )
    capped = dataclasses.replace(
        config,
        policy=dataclasses.replace(
            config.policy, total_dv_limit_km_s=limit_km_s, total_dv_safe_limit_km_s=limit_km_s
        ),
    )
    if refused:
        with pytest.raises(PolicyError, match="in total"):
            assert_final_script_policy(workshop_script, output, capped)
    else:
        assert_final_script_policy(workshop_script, output, capped)
