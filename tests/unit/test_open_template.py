from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, SearchConfig
from orbitwin.errors import ParseError, PolicyError
from orbitwin.gmat_parser import GMAT_RESERVED_RESOURCE_NAMES, parse_gmat_script
from orbitwin.models import SYNTHESIZED_BURN_FRAME, Burn, Candidate
from orbitwin.script_renderer import (
    SYNTHESIS_MARKER,
    assert_final_script_policy,
    render_candidate,
)
from orbitwin.search import search_one_burn, zero_burn_candidate
from orbitwin.two_burn import search_two_burn_global

# 初賽規則 §2 says B 可進行多次瞬時脈衝機動 and 說明會簡報 p.8 says 由參賽者對
# Spacecraft B 進行機動. A template whose mission sequence is a single Propagate
# offers no way to do either, so the maneuvers are ours to write -- not a
# liberty taken with the organizer's file, but the only reading under which the
# task they set can be performed at all.


def _config(**search: object) -> OrbitWinConfig:
    base = dataclasses.replace(
        OrbitWinConfig(),
        pursuer_name="Pursuer",
        target_name="AlienSC",
        search=SearchConfig(
            one_burn_samples=24, one_burn_grid_burns=4, one_burn_grid_arrivals=6,
            two_burn_enabled=False, event_refine_top=2, keep_candidates=8, workers=1,
        ),
    )
    if search:
        return dataclasses.replace(base, search=dataclasses.replace(base.search, **search))
    return base


def _burn(time_s: float, dv: tuple[float, float, float], slot: int) -> Burn:
    # These tests read the rendered text, so the components are written as
    # given rather than derived; the recorded frame says which frame they
    # claim. Without one the renderer recomputes them from the impulse.
    return Burn(
        time_s=time_s, dv_inertial_km_s=dv, dv_script_km_s=dv, slot_index=slot,
        frame=SYNTHESIZED_BURN_FRAME,
    )


def _failing_candidate(*burns: Burn) -> Candidate:
    """No interception, so the sequence runs to 4TA."""
    return Candidate(
        burns=burns, intercept_time_s=None, minimum_distance_km=800.0,
        distance_at_completion_km=800.0, successful=False, analytical_endpoint_error_km=None,
        branch="test", metadata={"event_refined": True},
    )


def _candidate(*burns: Burn, intercept: float = 20000.0) -> Candidate:
    return Candidate(
        burns=burns, intercept_time_s=intercept, minimum_distance_km=0.5,
        distance_at_completion_km=5.0, successful=True, analytical_endpoint_error_km=None,
        branch="test", metadata={"arrival_time_s": intercept, "event_refined": True},
    )


def test_a_template_with_no_maneuver_parses(no_maneuver_script: Path) -> None:
    scenario, _ = parse_gmat_script(no_maneuver_script, _config())
    assert scenario.burn_slots == ()
    assert len(scenario.propagate_segments) == 1
    assert scenario.max_time_s > 0.0


def test_it_refuses_to_guess_which_spacecraft_is_ours(no_maneuver_script: Path) -> None:
    """Which is which is read off the Maneuver commands, and there are none.

    Solving for the wrong spacecraft makes every answer confidently wrong, so
    the parser stops and names both candidates rather than picking one.
    """
    unnamed = dataclasses.replace(_config(), pursuer_name=None, target_name=None)
    with pytest.raises(ParseError, match="no Maneuver command") as raised:
        parse_gmat_script(no_maneuver_script, unnamed)
    message = str(raised.value)
    assert "Pursuer" in message and "AlienSC" in message
    assert "pursuer_name" in message


@pytest.mark.parametrize("burn_count", [0, 1, 2, 3])
def test_the_mission_is_split_into_one_more_segment_than_maneuvers(
    no_maneuver_script: Path, burn_count: int
) -> None:
    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    separation = config.policy.safe_burn_separation_s
    burns = tuple(
        _burn(1000.0 + index * separation, (0.1, 0.0, 0.0), index) for index in range(burn_count)
    )
    # A failing candidate, so the mission runs to 4TA: this test is about how
    # many segments an open template is split into, and the horizon is the case
    # where the last one is longest.
    rendered = render_candidate(document, scenario, _failing_candidate(*burns), config)
    text = rendered.script_bytes.decode("utf-8")

    assert len(re.findall(r"^Propagate ", text, re.M)) == burn_count + 1
    assert len(re.findall(r"^Maneuver ", text, re.M)) == burn_count
    assert len(re.findall(r"^Create ImpulsiveBurn ", text, re.M)) == burn_count
    assert len(rendered.segment_durations_s) == burn_count + 1

    # GMAT reaches the end by adding up the numbers in the file, so the sum is
    # what has to land on the epoch the renderer says the mission ends at.
    # These candidates carry no interception, so that epoch is 4TA -- §4 scores
    # Δr(4TA) when nothing intercepts, and a sum landing short omits it.
    total = 0.0
    for duration in rendered.segment_durations_s:
        total += duration
    assert total == rendered.mission_end_s
    assert rendered.mission_end_s == scenario.max_time_s


def test_the_synthesized_script_says_what_it_added(no_maneuver_script: Path) -> None:
    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    candidate = _candidate(_burn(1000.0, (0.4, -0.2, 0.1), 0))
    rendered = render_candidate(document, scenario, candidate, config)
    text = rendered.script_bytes.decode("utf-8")

    coordinate_system, origin, axes = SYNTHESIZED_BURN_FRAME
    assert f"ImpulsiveBurn1.CoordinateSystem = {coordinate_system};" in text
    assert f"ImpulsiveBurn1.Origin = {origin};" in text
    assert f"ImpulsiveBurn1.Axes = {axes};" in text
    assert "ImpulsiveBurn1.DecrementMass = false;" in text
    assert "Maneuver ImpulsiveBurn1(Pursuer);" in text
    # The components are the candidate's, written so they read back exactly.
    assert "ImpulsiveBurn1.Element1 = 0.4;" in text
    assert "+ Create ImpulsiveBurn ImpulsiveBurn1" in rendered.semantic_diff
    assert "+ Maneuver ImpulsiveBurn1(Pursuer)" in rendered.semantic_diff


def test_the_organizers_own_lines_are_untouched(no_maneuver_script: Path) -> None:
    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    rendered = render_candidate(
        document, scenario, _candidate(_burn(1000.0, (0.4, 0.0, 0.0), 0)), config
    )
    generated = rendered.script_bytes.decode("utf-8").splitlines(keepends=True)
    original = list(document.lines)
    # Everything except the one Propagate the split replaced is still present,
    # in order and byte for byte.
    survivors = [line for line in original if not line.lstrip().startswith("Propagate ")]
    index = 0
    for line in generated:
        if index < len(survivors) and line == survivors[index]:
            index += 1
    assert index == len(survivors), "an organizer line went missing or moved"


def test_synthesized_names_avoid_reserved_words_and_collisions(
    no_maneuver_script: Path, tmp_path: Path
) -> None:
    config = _config()
    # A script that already uses the name we would have picked first.
    text = no_maneuver_script.read_text(encoding="utf-8-sig")
    text = text.replace(
        "BeginMissionSequence;",
        "Create Variable ImpulsiveBurn1;\nBeginMissionSequence;",
        1,
    )
    taken = tmp_path / "taken.script"
    taken.write_text(text, encoding="utf-8")

    scenario, document = parse_gmat_script(taken, config)
    rendered = render_candidate(
        document, scenario, _candidate(_burn(1000.0, (0.4, 0.0, 0.0), 0)), config
    )
    generated = rendered.script_bytes.decode("utf-8")
    (name,) = re.findall(r"^Create ImpulsiveBurn (\w+);", generated, re.M)
    assert name != "ImpulsiveBurn1"
    assert name not in GMAT_RESERVED_RESOURCE_NAMES


def _rendered(no_maneuver_script: Path) -> tuple[OrbitWinConfig, str]:
    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    found = [c for c in search_one_burn(scenario, config) if c.successful]
    candidate = found[0] if found else _candidate(_burn(1000.0, (0.4, 0.0, 0.0), 0))
    return config, render_candidate(document, scenario, candidate, config).script_bytes.decode()


def test_the_guard_accepts_a_script_that_only_gained_maneuvers(
    no_maneuver_script: Path, tmp_path: Path
) -> None:
    config, text = _rendered(no_maneuver_script)
    generated = tmp_path / "final.script"
    generated.write_text(text, encoding="utf-8")
    assert_final_script_policy(no_maneuver_script, generated, config)


@pytest.mark.parametrize(
    ("what", "find", "replace"),
    [
        ("the target's orbit", "AlienSC.SMA = 9000;", "AlienSC.SMA = 9500;"),
        ("the force model", "FM.Drag = None;", "FM.Drag = MSISE90;"),
        ("the propagator", "Prop.Accuracy = 1e-12;", "Prop.Accuracy = 1e-9;"),
        ("a dropped organizer line", "FM.SRP = Off;\n", ""),
        ("an unrelated assignment", "BeginMissionSequence;", "Pursuer.DryMass = 1;\nBeginMissionSequence;"),
        ("a second ReportFile", "BeginMissionSequence;", "Create ReportFile Extra;\nBeginMissionSequence;"),
        ("the burn frame", "ImpulsiveBurn1.Axes = VNB;", "ImpulsiveBurn1.Axes = MJ2000Eq;"),
        ("mass decrement", "ImpulsiveBurn1.DecrementMass = false;", "ImpulsiveBurn1.DecrementMass = true;"),
    ],
)
def test_the_guard_still_refuses_everything_else(
    no_maneuver_script: Path, tmp_path: Path, what: str, find: str, replace: str
) -> None:
    config, text = _rendered(no_maneuver_script)
    assert find in text, f"the fixture no longer contains {find!r}"
    generated = tmp_path / "tampered.script"
    generated.write_text(text.replace(find, replace, 1), encoding="utf-8")
    with pytest.raises(PolicyError):
        assert_final_script_policy(no_maneuver_script, generated, config)


def test_a_maneuver_on_the_target_is_refused(no_maneuver_script: Path, tmp_path: Path) -> None:
    # §2 makes A passive: 僅受重力場影響,無主動機動.
    config, text = _rendered(no_maneuver_script)
    generated = tmp_path / "target.script"
    generated.write_text(
        text.replace("Maneuver ImpulsiveBurn1(Pursuer);", "Maneuver ImpulsiveBurn1(AlienSC);", 1),
        encoding="utf-8",
    )
    with pytest.raises((PolicyError, ParseError)):
        assert_final_script_policy(no_maneuver_script, generated, config)


def test_the_baseline_still_renders_without_adding_anything(no_maneuver_script: Path) -> None:
    # No burns means no maneuvers to write, so the script keeps its single
    # Propagate and gains only the comment marker.
    config = _config()
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    rendered = render_candidate(document, scenario, zero_burn_candidate(scenario, config), config)
    text = rendered.script_bytes.decode("utf-8")
    assert "Create ImpulsiveBurn" not in text
    assert re.findall(r"^Maneuver ", text, re.M) == []
    assert len(re.findall(r"^Propagate ", text, re.M)) == 1


def test_a_non_ascii_character_anywhere_is_refused(
    no_maneuver_script: Path, tmp_path: Path
) -> None:
    """Measured on GMAT R2026a, which refuses the file before interpreting it.

        Interpreter Exception: The script "..." contains characters outside of
        the ASCII character set; please fix the file before proceeding.

    Found by rendering a two-burn candidate against a fixture whose header
    comment quoted the rules in Chinese. The script was legal in every other
    respect -- the policy guard passed it -- and GMAT would not open it. A
    submission that cannot be opened scores nothing.

    Either error is a pass because the same refusal now happens twice, and the
    earlier one wins. `assert_final_script_policy` parses both scripts before it
    reaches `_assert_ascii`, and the parser rejects a high byte in a file it is
    handed -- see `test_source_script_encoding` for why the input needed checking
    too. What this test pins is unchanged: a script carrying a byte above 127
    does not get out of here, whichever gate stops it.
    """
    config, text = _rendered(no_maneuver_script)
    for insertion in ("% \u521d\u8cfd\u898f\u5247\n", "% section \u00a75\n", "% a\u00a0non-breaking space\n"):
        generated = tmp_path / "nonascii.script"
        generated.write_text(text.replace("BeginMissionSequence;", insertion + "BeginMissionSequence;", 1), encoding="utf-8")
        with pytest.raises((ParseError, PolicyError), match="non-ASCII"):
            assert_final_script_policy(no_maneuver_script, generated, config)


# The other shape the competition template can take: the burn resources are
# still declared, the mission sequence simply no longer maneuvers them.


def test_declared_but_unmaneuvered_burns_are_slots(unmaneuvered_burns_script: Path) -> None:
    from orbitwin.models import template_maneuvers

    scenario, _ = parse_gmat_script(unmaneuvered_burns_script, _config())
    assert [slot.name for slot in scenario.burn_slots] == ["Burn1", "Burn2"]
    assert all(slot.maneuver_line is None for slot in scenario.burn_slots)
    assert template_maneuvers(scenario) is False
    assert len(scenario.propagate_segments) == 1


def test_what_is_declared_is_used_before_anything_is_added(
    unmaneuvered_burns_script: Path,
) -> None:
    """A resource the organizer declared is one they left for us to use.

    Adding our own beside it is a larger edit than wiring theirs in, and it
    throws away the frame their declaration states.
    """
    config = _config()
    scenario, document = parse_gmat_script(unmaneuvered_burns_script, config)
    separation = config.policy.safe_burn_separation_s

    for count, expected_maneuvers, expected_new in (
        (1, ["Burn1"], 0),
        (2, ["Burn1", "Burn2"], 0),
        (3, ["Burn1", "Burn2", "ImpulsiveBurn1"], 1),
    ):
        burns = tuple(
            _burn(1000.0 + index * separation, (0.2, -0.1, 0.05), index)
            for index in range(count)
        )
        rendered = render_candidate(document, scenario, _candidate(*burns), config)
        text = rendered.script_bytes.decode("utf-8")
        assert re.findall(r"^Maneuver (\w+)\(", text, re.M) == expected_maneuvers
        created = re.findall(r"^Create ImpulsiveBurn (\w+);", text, re.M)
        assert len(created) == 2 + expected_new, count
        # Reuse patches components in place, so nothing is added for those.
        assert text.count(SYNTHESIS_MARKER) == (1 if expected_new else 0)


def test_a_reused_burn_keeps_the_frame_the_organizer_declared(
    unmaneuvered_burns_script: Path, tmp_path: Path
) -> None:
    from orbitwin.models import SYNTHESIZED_BURN_FRAME as ours

    config = _config()
    scenario, document = parse_gmat_script(unmaneuvered_burns_script, config)
    declared = scenario.burn_slots[0]
    rendered = render_candidate(
        document, scenario, _candidate(_burn(1000.0, (0.4, 0.0, 0.0), 0)), config
    )
    text = rendered.script_bytes.decode("utf-8")
    assert f"{declared.name}.Axes = {declared.axes};" in text
    assert f"{declared.name}.Element1 = 0.4;" in text

    generated = tmp_path / "final.script"
    generated.write_text(text, encoding="utf-8")
    assert_final_script_policy(unmaneuvered_burns_script, generated, config)

    # Changing it to the frame we would have chosen for a burn of our own is
    # refused. Two layers can catch it and the line-level diff gets there
    # first, naming the line rather than the resource: Axes is neither a burn
    # component nor a Propagate, so it is simply one of the organizer's lines
    # that moved. The frame check behind it states the same invariant for a
    # redeclaration that does not touch an original line.
    other = "MJ2000Eq" if declared.axes != "MJ2000Eq" else ours[2]
    tampered = tmp_path / "tampered.script"
    tampered.write_text(
        text.replace(f"{declared.name}.Axes = {declared.axes};", f"{declared.name}.Axes = {other};", 1),
        encoding="utf-8",
    )
    with pytest.raises(PolicyError, match="changed or dropped line"):
        assert_final_script_policy(unmaneuvered_burns_script, tampered, config)


def test_no_slot_variants_when_nothing_is_maneuvered_yet(
    unmaneuvered_burns_script: Path,
) -> None:
    # Two declared resources would otherwise produce a variant per assignment.
    # A burn goes wherever the Propagate is split, so which resource carries it
    # changes no renderability and the variants would be pure duplication.
    from orbitwin.search import slot_assignment_variants

    config = _config()
    scenario, _ = parse_gmat_script(unmaneuvered_burns_script, config)
    assert len(scenario.burn_slots) == 2
    candidate = _candidate(_burn(1000.0, (0.4, 0.0, 0.0), 0))
    assert slot_assignment_variants(scenario, config, [candidate]) == []


def test_the_baseline_renders_against_declared_but_unused_burns(
    unmaneuvered_burns_script: Path, tmp_path: Path
) -> None:
    """The zero-burn baseline on a template that declares resources.

    It crashed: the rendered script maneuvers nothing, so the parser reads both
    declared resources back as slots, while the mission has one segment and one
    burn epoch. The legality check walked the slots and indexed an epoch that
    does not exist. A resource nothing maneuvers has no epoch and no effect --
    §5 counts maneuvers and there is none -- so it is not one of the burns to
    be checked.
    """
    from orbitwin.search import zero_burn_candidate

    config = _config()
    scenario, document = parse_gmat_script(unmaneuvered_burns_script, config)
    rendered = render_candidate(
        document, scenario, zero_burn_candidate(scenario, config), config
    )
    text = rendered.script_bytes.decode("utf-8")
    assert re.findall(r"^Maneuver ", text, re.M) == []
    assert len(re.findall(r"^Create ImpulsiveBurn ", text, re.M)) == 2, "and none added"
    assert len(re.findall(r"^Propagate ", text, re.M)) == 1

    generated = tmp_path / "baseline.script"
    generated.write_text(text, encoding="utf-8")
    assert_final_script_policy(unmaneuvered_burns_script, generated, config)


def test_both_template_shapes_reach_the_same_answer(
    no_maneuver_script: Path, unmaneuvered_burns_script: Path
) -> None:
    # Whether the resources are declared changes how the script is written, not
    # what it does. The search should not be able to tell the difference.
    config = _config(one_burn_samples=48, one_burn_grid_burns=6, one_burn_grid_arrivals=10)
    answers = []
    for script in (no_maneuver_script, unmaneuvered_burns_script):
        scenario, _ = parse_gmat_script(script, config)
        found = [c for c in search_one_burn(scenario, config) if c.successful]
        assert found, script.name
        best = found[0]
        answers.append((best.intercept_time_s, best.total_dv_km_s, len(best.burns)))
    assert answers[0] == answers[1]


def _declaring(script: Path, tmp_path: Path, count: int) -> Path:
    """The same open template with `count` of its ImpulsiveBurn resources kept.

    Derived from the two-resource fixture rather than written out, so the three
    shapes are provably the same mission: same orbits, same epoch, same single
    Propagate. Only the number of resources the organizer happened to declare
    differs, and that is what the assertions are about.
    """
    lines = script.read_text(encoding="utf-8").splitlines(keepends=True)
    dropped = [f"Burn{index}" for index in range(count + 1, 3)]
    kept = [
        line
        for line in lines
        if not any(
            re.match(rf"\s*(?:GMAT\s+)?(?:Create ImpulsiveBurn\s+{name}\b|{name}\.)", line)
            for name in dropped
        )
    ]
    out = tmp_path / f"declaring{count}.script"
    out.write_text("".join(kept), encoding="utf-8")
    return out


def test_the_two_burn_search_ignores_how_many_resources_are_declared(
    unmaneuvered_burns_script: Path, tmp_path: Path
) -> None:
    """A count that stopped meaning what it did.

    search_two_burn_global returned [] when fewer than two ImpulsiveBurn
    resources were declared, which was right while a slot only existed because
    a Maneuver used it. Once a declared-but-unmaneuvered resource became a
    slot, the same test read as "how many did the organizer happen to declare"
    -- and a template whose mission is a single Propagate can be given two
    maneuvers whether it declares two resources, one, or none.

    Measured on three files that are the same mission:

        declared 0  ->  0 two-burn candidates
        declared 1  ->  0 two-burn candidates
        declared 2  ->  6 two-burn candidates

    Nothing failed and nothing warned; the two-burn search simply did not run.
    """
    config = _config(two_burn_enabled=True, two_burn_samples=128)
    counts = []
    for declared in (0, 1, 2):
        script = _declaring(unmaneuvered_burns_script, tmp_path, declared)
        scenario, _ = parse_gmat_script(script, config)
        assert len(scenario.burn_slots) == declared, "the fixture surgery went wrong"
        counts.append(len(search_two_burn_global(scenario, config)))
    assert counts[0] > 0, "a template that maneuvers nothing can still take two burns"
    assert len(set(counts)) == 1, f"same mission, different two-burn results: {counts}"


def test_a_slotted_template_still_needs_two_slots_for_two_burns(
    workshop_script: Path,
) -> None:
    # The other half. A template that already maneuvers can express exactly the
    # impulses it has slots for, so there the count is still the right test.
    config = dataclasses.replace(
        _config(two_burn_enabled=True, two_burn_samples=64),
        pursuer_name=None,
        target_name=None,
    )
    scenario, _ = parse_gmat_script(workshop_script, config)
    one_slot = dataclasses.replace(scenario, burn_slots=scenario.burn_slots[:1])
    assert search_two_burn_global(one_slot, config) == []


def test_a_space_separated_create_survives_the_policy_guard(
    unmaneuvered_burns_script: Path, tmp_path: Path
) -> None:
    """The half of the comma fix that was not applied.

    `Create ImpulsiveBurn Burn1 Burn2;` is ordinary GMAT -- R2026a runs it --
    and the parser learned to read it. The renderer's policy guard still split
    on commas alone, so it saw one resource called "Burn1 Burn2", concluded
    that Burn1 was not something we were allowed to maneuver, and refused the
    final script with a reason that says the opposite of the truth:

        Final script adds a Maneuver of Burn1, which the original mission
        already maneuvers

    The original mission maneuvers nothing at all. Both halves now split the
    same way, in one function.
    """
    text = unmaneuvered_burns_script.read_text(encoding="utf-8")
    spaced = re.sub(r"^Create ImpulsiveBurn Burn2;\n", "", text, flags=re.M).replace(
        "Create ImpulsiveBurn Burn1;", "Create ImpulsiveBurn Burn1 Burn2;"
    )
    assert "Create ImpulsiveBurn Burn1 Burn2;" in spaced
    original = tmp_path / "spaced.script"
    original.write_text(spaced, encoding="utf-8")

    config = _config()
    scenario, document = parse_gmat_script(original, config)
    assert [slot.name for slot in scenario.burn_slots] == ["Burn1", "Burn2"]

    candidate = next(c for c in search_one_burn(scenario, config) if c.successful)
    rendered = render_candidate(document, scenario, candidate, config)
    generated = tmp_path / "spaced.out.script"
    generated.write_bytes(rendered.script_bytes)
    assert_final_script_policy(original, generated, config)


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("Burn1, Burn2", ["Burn1", "Burn2"]),
        ("Burn1 Burn2", ["Burn1", "Burn2"]),
        ("Burn1,Burn2", ["Burn1", "Burn2"]),
        ("  Burn1 ,  Burn2 ", ["Burn1", "Burn2"]),
        ("Burn1", ["Burn1"]),
    ],
)
def test_the_two_halves_split_a_name_list_the_same_way(
    spelling: str, expected: list[str]
) -> None:
    # One function, called from the parser and from the policy guard, so a
    # spelling cannot be legal to one and not the other.
    from orbitwin.gmat_parser import split_create_names

    assert split_create_names(spelling) == expected


def test_a_slotted_template_cannot_gain_a_maneuver_through_a_spare_resource(
    workshop_script: Path, tmp_path: Path
) -> None:
    """The widened guard, checked where widening it would hurt.

    Allowing a Maneuver of a resource the original declares but never
    maneuvers is what lets an open template use what the organizer left. On a
    template that *does* maneuver, the same sentence would permit adding a
    command -- which COMPETITION_POLICY.md (Unresolved permissions) does not.

    It does not, because the burn contract is compared first and a slotted
    original contributes only its maneuvered resources to it. This pins that,
    so the two rules cannot be changed independently into a hole.
    """
    text = workshop_script.read_text(encoding="utf-8")
    spacecraft = re.search(r"^\s*Maneuver\s+\w+\((\w+)\)", text, re.M).group(1)
    spare = (
        "Create ImpulsiveBurn Spare;\nSpare.CoordinateSystem = Local;\n"
        "Spare.Origin = Earth;\nSpare.Axes = VNB;\nSpare.Element1 = 0;\n"
        "Spare.Element2 = 0;\nSpare.Element3 = 0;\nSpare.DecrementMass = false;\n"
    )
    # A resource, so before the mission sequence, where the parser reads them.
    cut = text.index("\nBeginMissionSequence")
    original = text[:cut] + "\n" + spare + text[cut:]
    original_path = tmp_path / "with_spare.script"
    original_path.write_text(original, encoding="utf-8")

    # Keep the Propagate/Maneuver alternation intact, so the forgery gets past
    # the parser and has to be refused on the rule rather than on the shape.
    last_propagate = list(re.finditer(r"^\s*Propagate\b[^\n]*$", original, re.M))[-1]
    forged = (
        original[: last_propagate.end()]
        + f"\nManeuver Spare({spacecraft});\n{last_propagate.group(0).strip()}"
        + original[last_propagate.end() :]
    )
    forged_path = tmp_path / "forged.script"
    forged_path.write_text(forged, encoding="utf-8")

    with pytest.raises(PolicyError, match="Burn resource contract changed"):
        assert_final_script_policy(original_path, forged_path, OrbitWinConfig())


def test_the_refusal_carries_what_is_needed_to_answer_it(
    no_maneuver_script: Path, unmaneuvered_burns_script: Path
) -> None:
    """This is the first command run on competition day, and it stops.

    Answering wrongly is not a small error: solve for the wrong spacecraft and
    every number afterwards is confidently about the other one. So the question
    arrives with the evidence -- both orbits, which of them is circular, and
    which one the official ReportFile tracks.
    """
    unnamed = dataclasses.replace(_config(), pursuer_name=None, target_name=None)
    for script in (no_maneuver_script, unmaneuvered_burns_script):
        with pytest.raises(ParseError) as raised:
            parse_gmat_script(script, unnamed)
        message = str(raised.value)
        for name in ("AlienSC", "Pursuer"):
            assert f"{name}: SMA " in message, script.name
        assert "circular" in message
        # The organizer's own example reports the spacecraft that maneuvers.
        assert "the official ReportFile tracks it" in message
        assert "pursuer_name" in message


def test_it_still_refuses_when_an_orbit_cannot_be_read(
    no_maneuver_script: Path, tmp_path: Path
) -> None:
    # Describing the choice must not become a second way to fail: a spacecraft
    # the description cannot read is reported as such, and the refusal is still
    # the refusal.
    text = no_maneuver_script.read_text(encoding="utf-8-sig")
    broken = tmp_path / "broken.script"
    broken.write_text(text.replace("AlienSC.ECC = 0;", "AlienSC.ECC = 1;"), encoding="ascii")
    unnamed = dataclasses.replace(_config(), pursuer_name=None, target_name=None)
    with pytest.raises(ParseError, match="no Maneuver command") as raised:
        parse_gmat_script(broken, unnamed)
    assert "AlienSC" in str(raised.value)


@pytest.mark.parametrize(
    "spelling", ["BeginMissionSequence; % mission", "BeginMissionSequence ;"]
)
def test_maneuvers_are_written_whatever_the_begin_line_spelling(
    no_maneuver_script: Path, tmp_path: Path, spelling: str
) -> None:
    # The parser and GMAT accept both; a literal match raised a bare StopIteration.
    text = no_maneuver_script.read_text(encoding="utf-8-sig")
    script = tmp_path / "spelled.script"
    script.write_text(text.replace("BeginMissionSequence;", spelling, 1), encoding="ascii")
    config = _config()
    scenario, document = parse_gmat_script(script, config)
    rendered = render_candidate(
        document, scenario, _candidate(_burn(1000.0, (0.4, 0.0, 0.0), 0)), config
    )
    generated = rendered.script_bytes.decode("utf-8")
    assert generated.index(spelling) < generated.index("Maneuver ImpulsiveBurn1(Pursuer);")
    final = tmp_path / "final.script"
    final.write_bytes(rendered.script_bytes)
    assert_final_script_policy(script, final, config)


def _without_final_newline(no_maneuver_script: Path, tmp_path: Path, newline: str) -> Path:
    text = no_maneuver_script.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    assert text.endswith("};\n"), "the fixture's last line is no longer its Propagate"
    script = tmp_path / "no-final-newline.script"
    script.write_bytes(text.removesuffix("\n").replace("\n", newline).encode("ascii"))
    return script


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_a_template_ending_without_a_newline_keeps_one_command_per_line(
    no_maneuver_script: Path, tmp_path: Path, newline: str
) -> None:
    # An editor may save the last Propagate with no line end; a copy of it once
    # took the next Maneuver onto its own line, and the guard refused every burn.
    script = _without_final_newline(no_maneuver_script, tmp_path, newline)
    config = _config()
    scenario, document = parse_gmat_script(script, config)
    burns = (_burn(1000.0, (0.1, 0.0, 0.0), 0), _burn(2000.0, (0.0, 0.1, 0.0), 1))
    rendered = render_candidate(document, scenario, _failing_candidate(*burns), config)
    lines = rendered.script_bytes.decode("ascii").split(newline)
    mission = lines[next(i for i, line in enumerate(lines) if line == "BeginMissionSequence;") + 1 :]
    assert [line.split(" ", 1)[0] for line in mission] == [
        "Propagate", "Maneuver", "Propagate", "Maneuver", "Propagate"
    ]
    # The last line keeps the template's own ending, none.
    assert not rendered.script_bytes.endswith(b"\n")
    final = tmp_path / "final.script"
    final.write_bytes(rendered.script_bytes)
    assert_final_script_policy(script, final, config)


def test_the_guard_still_refuses_a_maneuver_on_the_propagate_line(
    no_maneuver_script: Path, tmp_path: Path
) -> None:
    # What the renderer wrote before; GMAT's reading of it was never measured.
    script = _without_final_newline(no_maneuver_script, tmp_path, "\n")
    config = _config()
    scenario, document = parse_gmat_script(script, config)
    rendered = render_candidate(
        document, scenario, _failing_candidate(_burn(1000.0, (0.1, 0.0, 0.0), 0)), config
    )
    joined = re.sub(r"\};\nManeuver", "};Maneuver", rendered.script_bytes.decode("ascii"), count=1)
    assert "};Maneuver" in joined
    final = tmp_path / "joined.script"
    final.write_text(joined, encoding="ascii")
    with pytest.raises((PolicyError, ParseError)):
        assert_final_script_policy(script, final, config)


@pytest.mark.parametrize("tail", ["}; \n", "};\t\n", "};\r\n", "};\r"])
def test_a_copied_propagate_keeps_the_templates_own_line_end(
    no_maneuver_script: Path, tmp_path: Path, tail: str
) -> None:
    config, text = _rendered(no_maneuver_script)
    first = re.search(r"^Propagate .*\};\n", text, re.M)
    assert first is not None
    tampered = text[: first.end() - 3] + tail + text[first.end() :]
    generated = tmp_path / "tampered.script"
    generated.write_bytes(tampered.encode("ascii"))
    with pytest.raises(PolicyError, match="not the template's own"):
        assert_final_script_policy(no_maneuver_script, generated, config)


def test_a_newline_is_allowed_only_where_the_template_had_none(
    no_maneuver_script: Path, tmp_path: Path
) -> None:
    # The template ends with a newline, so its last copy may not drop it.
    config, text = _rendered(no_maneuver_script)
    assert text.endswith("};\n")
    generated = tmp_path / "dropped.script"
    generated.write_bytes(text.removesuffix("\n").encode("ascii"))
    with pytest.raises(PolicyError, match="not the template's own"):
        assert_final_script_policy(no_maneuver_script, generated, config)
