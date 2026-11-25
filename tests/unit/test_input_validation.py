from __future__ import annotations

import csv
import dataclasses
import json
import math
import re
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, validate_config
from orbitwin.errors import ConfigurationError, DynamicsError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.io import candidate_from_dict, read_candidates, write_candidates, write_candidates_csv
from orbitwin.models import Burn, Candidate


def _policy(**overrides) -> OrbitWinConfig:
    base = OrbitWinConfig()
    return dataclasses.replace(base, policy=dataclasses.replace(base.policy, **overrides))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field",
    ["mu_km3_s2", "intercept_radius_km", "min_propagate_duration_s", "central_body_radius_km"],
)
def test_a_non_finite_policy_number_is_refused(field: str, value: float) -> None:
    # Every comparison against NaN is False, so the relational checks alone let
    # it past: `mu_km3_s2 <= 0` is False for NaN.
    with pytest.raises(ConfigurationError, match=field):
        validate_config(_policy(**{field: value}))


def test_a_non_finite_mu_would_otherwise_produce_a_nan_horizon(workshop_script: Path) -> None:
    """Under Tmax = nan every `t <= max_time_s` is False and every candidate
    passes, so both the parser and validate_config refuse mu = nan."""
    broken = _policy(mu_km3_s2=math.nan)
    with pytest.raises(DynamicsError, match="mu must be finite"):
        parse_gmat_script(workshop_script, broken)

    with pytest.raises(ConfigurationError):
        validate_config(broken)


def test_score_coefficients_must_be_finite_but_their_sign_is_the_organizers() -> None:
    from orbitwin.config import ScoreConfig

    base = OrbitWinConfig()
    with pytest.raises(ConfigurationError, match=r"score\.kt"):
        validate_config(dataclasses.replace(base, score=ScoreConfig(math.nan, 1.0, 1.0, 1.0)))
    # A negative coefficient is strange but it is announced, not inferred, and
    # COMPETITION_POLICY.md forbids guessing on the organizer's behalf.
    validate_config(dataclasses.replace(base, score=ScoreConfig(-1.0, 1.0, 1.0, 1.0)))


def _candidate() -> Candidate:
    burn = Burn(
        time_s=100.0, dv_inertial_km_s=(0.1, 0.0, 0.0),
        dv_script_km_s=(0.1, 0.0, 0.0), slot_index=0,
    )
    return Candidate(
        burns=(burn,), intercept_time_s=1000.0, minimum_distance_km=0.5,
        distance_at_completion_km=5.0, successful=True,
        analytical_endpoint_error_km=1e-9, branch="test",
        metadata={"event_refined": True, "minimum_pursuer_radius_km": 6995.5},
    )


def test_a_serialized_number_that_arrives_as_a_string_is_coerced() -> None:
    # certify reads candidates.json back. A string here reaches the first
    # comparison against it before anything notices -- three steps past the
    # file that was wrong.
    record = _candidate().to_dict()
    record["intercept_time_s"] = "1000.0"
    record["minimum_distance_km"] = "0.5"
    restored = candidate_from_dict(record)
    assert restored.intercept_time_s == 1000.0
    assert isinstance(restored.intercept_time_s, float)
    assert restored.minimum_distance_km == 0.5


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_a_success_flag_must_be_a_json_boolean(value: object) -> None:
    """bool("false") is True, so a hand-edited "false" loaded as a hit."""
    record = _candidate().to_dict()
    record["successful"] = value
    with pytest.raises(ValueError, match="successful must be true or false"):
        candidate_from_dict(record)
    record = _candidate().to_dict()
    record["score_is_stale"] = value
    with pytest.raises(ValueError, match="score_is_stale must be true or false"):
        candidate_from_dict(record)


@pytest.mark.parametrize(
    ("where", "value"),
    [
        ("dv_script_km_s", "123"),
        ("dv_inertial_km_s", "100"),
        ("penalties", "none"),
        ("warnings", "none"),
    ],
)
def test_a_string_is_not_read_character_by_character(where: str, value: str) -> None:
    """"100" loaded as a 1.0 km/s burn, and "none" as four penalties."""
    record = _candidate().to_dict()
    if where in ("penalties", "warnings"):
        record[where] = value
    else:
        record["burns"][0][where] = value
    with pytest.raises(ValueError, match="must be a list"):
        candidate_from_dict(record)


@pytest.mark.parametrize("value", [1.7, 1.0, True, -1, "1"])
def test_a_slot_index_is_not_truncated(value: object) -> None:
    """int(1.7) is 1: the burn would render into a slot the record never named."""
    record = _candidate().to_dict()
    record["burns"][0]["slot_index"] = value
    with pytest.raises(ValueError, match=r"burn\[0\]\.slot_index"):
        candidate_from_dict(record)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "NaN", True])
@pytest.mark.parametrize("where", ["intercept_time_s", "score", "time_s", "dv_script_km_s"])
def test_a_non_finite_or_boolean_number_is_refused(where: str, value: object) -> None:
    """json reads NaN and Infinity, and NaN passes every comparison against it."""
    record = _candidate().to_dict()
    if where == "time_s":
        record["burns"][0]["time_s"] = value
    elif where == "dv_script_km_s":
        record["burns"][0]["dv_script_km_s"] = [value, 0.0, 0.0]
    else:
        record[where] = value
    with pytest.raises(ValueError, match="number"):
        candidate_from_dict(record)


def test_the_two_frames_must_describe_one_impulse() -> None:
    """Limits read the inertial vector, the renderer writes the script one."""
    record = _candidate().to_dict()
    record["burns"][0]["dv_inertial_km_s"] = [0.3, 0.0, 0.0]
    record["burns"][0]["dv_script_km_s"] = [1.49, 0.0, 0.0]
    with pytest.raises(ValueError, match="one impulse cannot be both"):
        candidate_from_dict(record)
    # A rotation is fine: same length, different components.
    record["burns"][0]["dv_script_km_s"] = [0.0, 0.18, 0.24]
    candidate_from_dict(record)


def test_an_integer_too_large_for_a_float_is_refused_as_a_number() -> None:
    record = _candidate().to_dict()
    record["intercept_time_s"] = 10**400
    with pytest.raises(ValueError, match="finite number"):
        candidate_from_dict(record)


def test_a_record_of_the_wrong_shape_is_named_as_such() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        candidate_from_dict(["not", "a", "record"])  # type: ignore[arg-type]
    record = _candidate().to_dict()
    record["burns"] = {"time_s": 1.0}
    with pytest.raises(ValueError, match="burns must be a list"):
        candidate_from_dict(record)


def test_the_writers_create_the_directory_they_are_given(tmp_path: Path) -> None:
    candidates = [_candidate()]
    write_candidates(tmp_path / "does" / "not" / "exist" / "candidates.json", candidates)
    write_candidates_csv(tmp_path / "also" / "new" / "candidates.csv", candidates)
    assert (tmp_path / "does" / "not" / "exist" / "candidates.json").is_file()
    assert (tmp_path / "also" / "new" / "candidates.csv").is_file()


def test_the_json_round_trip_survives_and_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    write_candidates(path, [_candidate()])
    first = path.read_bytes()
    write_candidates(path, [_candidate()])
    assert path.read_bytes() == first, "the same candidates must serialize to the same bytes"
    assert first.endswith(b"\n")
    (restored,) = read_candidates(path)
    assert restored.candidate_id == _candidate().candidate_id
    assert restored.total_dv_km_s == pytest.approx(_candidate().total_dv_km_s)


def test_the_table_carries_the_quantity_the_rules_score(tmp_path: Path) -> None:
    # §4 scores Δrmin = max(Δr(Tteam), 5), and the table a human reads on the
    # day did not carry the distance at completion at all.
    path = tmp_path / "candidates.csv"
    write_candidates_csv(path, [_candidate()])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["distance_at_completion_km"] == "5.0"
    assert rows[0]["minimum_pursuer_radius_km"] == "6995.5"
    assert rows[0]["penalty_count"] == "0"
    assert rows[0]["warning_count"] == "0"


def test_a_hand_edited_file_still_loads_or_says_why(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        read_candidates(path)


def test_the_padding_cannot_be_shorter_than_the_propagate_floor() -> None:
    """The last Propagate segment is the padding, so it inherits the floor.

    _schedule_slots ends the mission at Tmax, so the final segment lasts
    max_time_s - last_burn and the only thing keeping it positive is the guard
    that refuses last_burn + final_padding_s > max_time_s. A padding below the
    floor lets that guard pass a schedule the renderer then refuses per
    segment, two steps from the setting that caused it -- and the goal it would
    otherwise have produced is one GMAT hangs on rather than rejects.
    """
    from orbitwin.config import SCHEDULE_TOLERANCE_S

    with pytest.raises(ConfigurationError, match="final_padding_s"):
        validate_config(_policy(final_padding_s=1.0e-9, min_propagate_duration_s=1.0e-3))

    # Equal is not enough, and the first version of this check said it was. The
    # schedule guard compares against the horizon with SCHEDULE_TOLERANCE_S of
    # slack, so it admits a burn late enough to leave a final segment that much
    # shorter than the padding. With the two set equal at 1e-3, the renderer
    # refused a segment of 0.0009999899994 s -- below the 0.001 s floor by
    # exactly the slack -- after validate_config had passed the profile.
    with pytest.raises(ConfigurationError, match="tolerance"):
        validate_config(_policy(final_padding_s=1.0e-3, min_propagate_duration_s=1.0e-3))
    validate_config(
        _policy(final_padding_s=1.0e-3 + SCHEDULE_TOLERANCE_S, min_propagate_duration_s=1.0e-3)
    )


def test_the_renderer_still_refuses_it_if_the_config_check_is_bypassed(
    workshop_script: Path,
) -> None:
    # Defence in depth, and the reason the config check is an addition rather
    # than a replacement: a Candidate can be built without validate_config ever
    # running, and the per-segment floor is what actually stops the zero-length
    # goal reaching GMAT.
    from orbitwin.errors import PolicyError
    from orbitwin.models import Burn, Candidate
    from orbitwin.script_renderer import render_candidate

    config = _policy(final_padding_s=0.0)
    scenario, document = parse_gmat_script(workshop_script, config)
    at_the_horizon = Burn(
        time_s=scenario.max_time_s,
        dv_inertial_km_s=(0.1, 0.0, 0.0),
        dv_script_km_s=(0.1, 0.0, 0.0),
        slot_index=1,
    )
    candidate = Candidate(
        burns=(at_the_horizon,), intercept_time_s=scenario.max_time_s,
        minimum_distance_km=1.0, distance_at_completion_km=5.0, successful=True,
        analytical_endpoint_error_km=None, branch="probe",
        metadata={"arrival_time_s": scenario.max_time_s, "event_refined": True},
    )
    with pytest.raises(PolicyError, match="below the"):
        render_candidate(document, scenario, candidate, config)


def test_every_field_declared_float_is_checked_for_finiteness() -> None:
    """The anti-drift test, and the reason the check stopped being a list.

    Written by hand it covered nineteen fields and missed two of them -- not
    because those two mattered more than the others but because a list that has
    to be extended whenever a field is added will be short again. This fails the
    day someone adds a float and the structural pass stops reaching it.

    The two it missed were search.second_burn_fraction_min and _max, since
    removed as settings nothing read: the second-burn sampler covers the whole
    legal window, so bounding it again would narrow the search rather than widen
    it. They are named here as history, not as something to look up.
    """
    import dataclasses as dc

    from orbitwin.config import GmatConfig, PolicyConfig, ScoreConfig, SearchConfig

    base = OrbitWinConfig(score=ScoreConfig(kt=0.001, ct=5000.0, kv=0.004, cv=800.0))
    sections = {
        "policy": (PolicyConfig, lambda value: dc.replace(base, policy=value)),
        "search": (SearchConfig, lambda value: dc.replace(base, search=value)),
        "gmat": (GmatConfig, lambda value: dc.replace(base, gmat=value)),
        "score": (ScoreConfig, lambda value: dc.replace(base, score=value)),
    }
    checked = 0
    for label, (cls, rebuild) in sections.items():
        current = getattr(base, label)
        for field in dc.fields(cls):
            if field.type not in {"float", "float | None"}:
                continue
            checked += 1
            broken = rebuild(dc.replace(current, **{field.name: math.nan}))
            with pytest.raises(ConfigurationError, match=re.escape(f"{label}.{field.name}")):
                validate_config(broken)
    # 19, not the 21 this was written against, for the reason in the docstring.
    # The floor is what carries the test: it fails the day the walk stops
    # reaching the fields, whatever the count happens to be.
    assert checked >= 19, f"only {checked} float fields found; the walk is not reaching them"

def test_a_quoted_false_does_not_switch_the_earth_dive_gate_off(tmp_path: Path) -> None:
    """'false' in quotes is a non-empty string, and a non-empty string is True."""
    from orbitwin.config import load_config

    path = tmp_path / "quoted.yaml"
    path.write_text("policy:\n  allow_body_intersecting_submission: 'false'\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="allow_body_intersecting_submission must be bool"):
        load_config(path)


def test_a_yaml_merge_key_still_loads(tmp_path: Path) -> None:
    """Overriding a merged key is YAML, not a duplicate."""
    from orbitwin.config import load_config

    path = tmp_path / "merge.yaml"
    path.write_text(
        "policy:\n  <<: &limits {dv_limit_km_s: 1.5, dv_safe_limit_km_s: 1.4}\n"
        "  dv_safe_limit_km_s: 1.45\n",
        encoding="utf-8",
    )
    assert load_config(path).policy.dv_safe_limit_km_s == 1.45


def test_a_merge_of_a_merge_may_still_be_overridden(tmp_path: Path) -> None:
    """Checking merge sources must still let each level override the one it merges."""
    from orbitwin.config import load_config

    path = tmp_path / "merge.yaml"
    path.write_text(
        "policy:\n  <<: &outer\n    <<: &inner {dv_limit_km_s: 1.5, dv_safe_limit_km_s: 1.4}\n"
        "    dv_safe_limit_km_s: 1.45\n  dv_safe_limit_km_s: 1.46\n",
        encoding="utf-8",
    )
    assert load_config(path).policy.dv_safe_limit_km_s == 1.46


def test_disjoint_merge_sources_load_and_may_be_overridden(tmp_path: Path) -> None:
    from orbitwin.config import load_config

    path = tmp_path / "merge.yaml"
    path.write_text(
        "policy:\n  <<: [{dv_limit_km_s: 1.5}, {dv_safe_limit_km_s: 1.4}]\n"
        "  dv_safe_limit_km_s: 1.45\n",
        encoding="utf-8",
    )
    assert load_config(path).policy.dv_safe_limit_km_s == 1.45


def test_every_bool_int_and_str_field_refuses_another_type() -> None:
    """The same walk as the float one, so a field added later is covered too."""
    import dataclasses as dc

    from orbitwin.config import GmatConfig, PolicyConfig, ScoreConfig, SearchConfig

    base = OrbitWinConfig(score=ScoreConfig(kt=0.001, ct=5000.0, kv=0.004, cv=800.0))
    sections = {
        "policy": (PolicyConfig, lambda value: dc.replace(base, policy=value)),
        "search": (SearchConfig, lambda value: dc.replace(base, search=value)),
        "gmat": (GmatConfig, lambda value: dc.replace(base, gmat=value)),
        "score": (ScoreConfig, lambda value: dc.replace(base, score=value)),
    }
    wrong = {
        "bool": ["false", 0],
        "int": [True, 2.0, "2"],
        "int | None": [True, 2.0, "2"],
        "str": [5],
        "str | None": [5],
        # A bare string iterates as characters; a list may hold only names.
        "tuple[str, ...]": ["ConicalFOV", [5], ["two words"]],
    }
    checked = 0
    for label, (cls, rebuild) in sections.items():
        current = getattr(base, label)
        for field in dc.fields(cls):
            for value in wrong.get(str(field.type), []):
                checked += 1
                broken = rebuild(dc.replace(current, **{field.name: value}))
                with pytest.raises(ConfigurationError, match=re.escape(f"{label}.{field.name}")):
                    validate_config(broken)
    for name in ("pursuer_name", "target_name"):
        with pytest.raises(ConfigurationError, match=f"^{name} must be"):
            validate_config(dc.replace(base, **{name: 7}))
    assert checked >= 40, f"only {checked} cases; the walk is not reaching the fields"
    # Both walks match declared types by name, so a new kind of type would be
    # skipped by both without a word.
    walked = set(wrong) | {"float", "float | None"}
    for cls in (PolicyConfig, SearchConfig, GmatConfig, ScoreConfig):
        for field in dc.fields(cls):
            assert str(field.type) in walked, f"{cls.__name__}.{field.name}: {field.type}"


@pytest.mark.parametrize(
    "text",
    [
        "policy: []\n",
        "search: 3\n",
        "score: []\n",
        "score: {}\n",
        "false\n",
        # PyYAML keeps the last of two equal keys.
        "policy:\n  allow_body_intersecting_submission: false\n"
        "  allow_body_intersecting_submission: true\n",
        # YAML 1.1 reads these as octal 64 and sexagesimal 90.
        "search:\n  workers: 0100\n",
        "search:\n  workers: 1:30\n",
        "policy:\n  min_burn_separation_s: 1:40.0\n",
        "policy:\n  [1, 2]: 3\n",
        # A duplicate inside a merge source loaded True and switched the Earth-dive gate off.
        "policy: {<<: &g {allow_body_intersecting_submission: false,"
        " allow_body_intersecting_submission: true}}\n",
        "policy: {<<: {<<: {min_burn_separation_s: 101.0, min_burn_separation_s: 100.0}}}\n",
        # Two << keys: PyYAML merges both and the later wins.
        "policy:\n  <<: {allow_body_intersecting_submission: false}\n"
        "  <<: {allow_body_intersecting_submission: true}\n",
        # Two sources of one << overlapping: PyYAML lets the first win.
        "policy:\n  <<: [{allow_body_intersecting_submission: false},"
        " {allow_body_intersecting_submission: true}]\n",
        "policy:\n  <<: [{<<: {min_burn_separation_s: 101.0}}, {min_burn_separation_s: 100.0}]\n",
    ],
)
def test_a_config_file_read_other_than_written_is_refused(tmp_path: Path, text: str) -> None:
    """`policy: []` loaded as every default, `score: {}` as no score, 0100 as 64."""
    from orbitwin.config import load_config

    path = tmp_path / "shape.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)


@pytest.mark.parametrize(
    "text",
    [
        "policy: {=: 1}\n",
        "policy: [\n",
        "policy: !!python/name:os.system {}\n",
        # Raised ConstructorError before keys were checked ahead of construction.
        "policy: &p {a: 1, extra: *p}\n",
    ],
)
def test_a_yaml_error_is_a_configuration_error(
    workshop_script: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str
) -> None:
    """`{=: 1}` raised ConstructorError, which the CLI does not catch: a traceback, exit 1."""
    from orbitwin.cli import main
    from orbitwin.config import load_config

    path = tmp_path / "broken.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)
    assert main(["inspect", str(workshop_script), "--config", str(path)]) == 2
    assert "orbitwin: ERROR [" in capsys.readouterr().err


def test_a_yaml_error_names_the_file(tmp_path: Path) -> None:
    from orbitwin.config import load_config

    path = tmp_path / "broken.yaml"
    path.write_text("policy: [\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=re.escape(str(path))):
        load_config(path)


def test_the_config_rule_covers_what_the_schedule_guard_admits(
    workshop_script: Path, tmp_path: Path
) -> None:
    """The config rule has to bound the worst case the renderer will accept.

    Not the typical case. The schedule guard is permissive by
    SCHEDULE_TOLERANCE_S, so the tightest profile validate_config allows must
    still produce a final segment above the floor when a burn lands exactly on
    the boundary the guard admits.
    """
    from orbitwin.config import SCHEDULE_TOLERANCE_S, SearchConfig
    from orbitwin.models import Burn, Candidate
    from orbitwin.script_renderer import render_candidate

    floor_s = 1.0e-3
    config = dataclasses.replace(
        _policy(final_padding_s=floor_s + SCHEDULE_TOLERANCE_S, min_propagate_duration_s=floor_s),
        search=SearchConfig(one_burn_samples=0, two_burn_enabled=False),
    )
    validate_config(config)
    scenario, document = parse_gmat_script(workshop_script, config)

    # Exactly the latest burn the guard will admit.
    latest = scenario.max_time_s - config.policy.final_padding_s + SCHEDULE_TOLERANCE_S
    candidate = Candidate(
        burns=(Burn(latest, (0.1, 0.0, 0.0), (0.1, 0.0, 0.0), 1),),
        intercept_time_s=latest, minimum_distance_km=1.0, distance_at_completion_km=5.0,
        successful=True, analytical_endpoint_error_km=None, branch="boundary",
        metadata={"arrival_time_s": latest, "event_refined": True},
    )
    rendered = render_candidate(document, scenario, candidate, config)
    assert rendered.segment_durations_s[-1] >= floor_s, rendered.segment_durations_s[-1]


SCORE_COEFFICIENTS = "score:\n  kt: 0.003\n  ct: 5000.0\n  kv: 1.2\n  cv: 2.0\n"


@pytest.mark.parametrize(
    ("units", "missing"),
    [
        ("", "score.time_unit and score.delta_v_unit"),
        ("  time_unit: s\n", "score.delta_v_unit"),
        ("  delta_v_unit: km/s\n", "score.time_unit"),
    ],
)
def test_a_score_block_without_its_units_is_refused(
    tmp_path: Path, units: str, missing: str
) -> None:
    """Cv = 2.0 announced in km/s loaded as 2 m/s, and the delta-v term went flat."""
    from orbitwin.config import load_config

    path = tmp_path / "score.yaml"
    path.write_text(SCORE_COEFFICIENTS + units, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=f"does not state {missing}\\. .*never guessed"):
        load_config(path)


@pytest.mark.parametrize(
    ("units", "message"),
    [
        # null is not a unit, and the key being there does not make it one.
        ("  time_unit: s\n  delta_v_unit:\n", "delta_v_unit must be str"),
        ("  time_unit: ~\n  delta_v_unit: m/s\n", "time_unit must be str"),
        ("  time_unit: s\n  delta_v_unit: ''\n", "delta_v_unit must be m/s or km/s"),
        ("  time_unit: s\n  delta_v_unit: M/S\n", "delta_v_unit must be m/s or km/s"),
        ("  time_unit: s\n  delta_v_unit: 'km/s '\n", "delta_v_unit must be m/s or km/s"),
        ("  time_unit: min\n  delta_v_unit: m/s\n", "Only seconds"),
    ],
)
def test_a_stated_unit_must_still_be_a_known_one(
    tmp_path: Path, units: str, message: str
) -> None:
    from orbitwin.config import load_config

    path = tmp_path / "score.yaml"
    path.write_text(SCORE_COEFFICIENTS + units, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=re.escape(message)):
        load_config(path)


@pytest.mark.parametrize("unit", ["m/s", "km/s"])
def test_stated_units_are_the_ones_used(tmp_path: Path, unit: str) -> None:
    from orbitwin.config import load_config

    path = tmp_path / "score.yaml"
    path.write_text(
        SCORE_COEFFICIENTS + f"  time_unit: s\n  delta_v_unit: {unit}\n", encoding="utf-8"
    )
    score = load_config(path).score
    assert score is not None and (score.time_unit, score.delta_v_unit) == ("s", unit)


def test_units_given_through_a_merge_key_are_stated(tmp_path: Path) -> None:
    from orbitwin.config import load_config

    path = tmp_path / "score.yaml"
    path.write_text(
        "score: {<<: {time_unit: s, delta_v_unit: km/s}, kt: 0.003, ct: 5000.0, kv: 1.2, "
        "cv: 2.0}\n",
        encoding="utf-8",
    )
    score = load_config(path).score
    assert score is not None and score.delta_v_unit == "km/s"


SHIPPED = Path(__file__).resolve().parents[2] / "config"


@pytest.mark.parametrize(
    "path", sorted(p for p in SHIPPED.rglob("*.yaml")
                   if p.name not in {"tolerances.yaml", "editable-fields.yaml"}),
    ids=lambda p: str(p.relative_to(SHIPPED)),
)
def test_every_shipped_config_still_loads(path: Path) -> None:
    from orbitwin.config import load_config

    load_config(path)
