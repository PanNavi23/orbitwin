from __future__ import annotations

import pathlib
import re

import pytest

from orbitwin.config import (
    GmatConfig,
    OrbitWinConfig,
    PolicyConfig,
    ScoreConfig,
    SearchConfig,
)

# A knob nothing reads is worse than no knob, because a profile that sets it
# describes a search that is not happening.
#
# Found this way: `two_burn_seed_count` was 100 in deep.yaml, 40 in standard,
# 20 in quick, 50 in ood-test, and read nowhere -- the two-burn search uses
# three fixed offsets off search.seed regardless. `second_burn_fraction_min`
# and `_max` were superseded when the sampler started covering the whole legal
# second-burn window. `allow_force_model_approximation` was superseded by the
# CandidateStage ladder.
#
# `allow_zero_vector_unused_slots` was the one that mattered:
# ASSUMPTIONS_AND_GAPS.md records it as the switch to flip if the organizer
# forbids filling an unused official slot with an exact zero vector, and
# flipping it changed nothing at all. It is now enforced by the renderer.

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "src" / "orbitwin"


def _bodies_that_could_read(field_name: str) -> list[str]:
    """Every module's source, minus the line that declares this field.

    config.py counts, because refusing a bad value is using the field:
    score.time_unit is never converted by anything -- validate_config accepts
    only seconds, and that refusal is the whole point of it. What has to be
    caught is a name that appears nowhere except the line declaring it, which
    is exactly what all four removals above were.
    """
    bodies: list[str] = []
    declaration = re.compile(rf"\s+{re.escape(field_name)}\s*:")
    for path in sorted(SOURCE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if path.name == "config.py":
            lines = text.splitlines(keepends=True)
            keep = [not declaration.match(line) for line in lines]
            # Also drop the `#:` block immediately above the declaration.
            # Almost every field in this file documents itself that way and the
            # comment usually names the field, so stripping only the declaration
            # left the natural way to add a setting -- with a doc comment -- as
            # the way to defeat the guard. Demonstrated: a `zzz_dead_knob` that
            # nothing reads passed as soon as it had a `#:` line above it.
            for index, dropped in enumerate(keep):
                if dropped:
                    continue
                back = index - 1
                while back >= 0 and lines[back].lstrip().startswith("#:"):
                    keep[back] = False
                    back -= 1
            text = "".join(line for line, wanted in zip(lines, keep, strict=True) if wanted)
        bodies.append(text)
    return bodies


@pytest.mark.parametrize(
    "config_class",
    [SearchConfig, PolicyConfig, GmatConfig, ScoreConfig, OrbitWinConfig],
    ids=lambda cls: cls.__name__,
)
def test_every_config_field_is_read_somewhere(config_class: type) -> None:
    """Every declared setting has to appear somewhere other than its declaration.

    Deliberately a smoke test rather than a call-graph analysis: it matches the
    field name as a word, so a name that only appears in a comment would slip
    through. That is the right trade here -- it catches the failure that
    actually happened, a field added to the dataclass and to four profiles and
    then never wired up, and it cannot fail spuriously on an indirect read.
    """
    unread = [
        name
        for name in config_class.__dataclass_fields__
        if not any(
            re.search(rf"\b{re.escape(name)}\b", body)
            for body in _bodies_that_could_read(name)
        )
    ]
    assert not unread, (
        f"{config_class.__name__} declares {unread}, which nothing reads. "
        "Wire it up or remove it -- a setting that does nothing is a false claim "
        "about what the run did."
    )


def test_switching_off_zero_vector_slots_actually_refuses_them(workshop_script) -> None:
    """The flag ASSUMPTIONS_AND_GAPS.md tells us to flip if the answer is no.

    Before this it was decorative: the renderer filled every unused official
    slot with an exact zero vector whatever the policy said, so a team told
    "zero-vector maneuvers are not allowed" could set the flag, watch the run
    succeed, and submit the script the organizer had just forbidden.
    """
    import dataclasses

    from conftest import quick_config

    from orbitwin.errors import PolicyError
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.script_renderer import render_candidate
    from orbitwin.search import search_one_burn


    config = quick_config()
    assert config.policy.allow_zero_vector_unused_slots is True
    scenario, document = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    assert len(candidate.burns) < len(scenario.burn_slots), "needs a slot left over"

    render_candidate(document, scenario, candidate, config)

    forbidden = dataclasses.replace(
        config,
        policy=dataclasses.replace(config.policy, allow_zero_vector_unused_slots=False),
    )
    with pytest.raises(PolicyError, match="allow_zero_vector_unused_slots"):
        render_candidate(document, scenario, candidate, forbidden)


def test_the_final_script_gate_refuses_a_zero_vector_too(
    workshop_script, tmp_path
) -> None:
    """The renderer and the policy gate have to agree, and they did not.

    Measured: with the flag off, render_candidate raised while
    assert_final_script_policy passed a script an earlier run had produced with
    the flag on. Every final output must pass that gate, so the closed half was the
    half nobody uploads through.
    """
    import dataclasses

    from conftest import quick_config

    from orbitwin.errors import PolicyError
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.script_renderer import assert_final_script_policy, write_rendered_candidate
    from orbitwin.search import search_one_burn

    config = quick_config()
    scenario, document = parse_gmat_script(workshop_script, config)
    candidate = search_one_burn(scenario, config)[0]
    generated = tmp_path / "candidate.script"
    write_rendered_candidate(generated, tmp_path / "diff.txt", document, scenario, candidate, config)
    assert "ImpulsiveBurn2.Element1 = 0.0;" in generated.read_text(), "needs a zero-vector slot"

    # Permitted by default, which is the state that produced the file.
    assert_final_script_policy(workshop_script, generated, config)

    forbidden = dataclasses.replace(
        config,
        policy=dataclasses.replace(config.policy, allow_zero_vector_unused_slots=False),
    )
    with pytest.raises(PolicyError, match="allow_zero_vector_unused_slots"):
        assert_final_script_policy(workshop_script, generated, forbidden)
