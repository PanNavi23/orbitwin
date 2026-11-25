"""The parameter type: strict, mapped to DUEL's announced list, DUEL checked."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

import orbitwin.duel.params as params_module
from orbitwin.duel.params import (
    ANNOUNCED_FIELDS,
    Announced,
    DuelParamsError,
    check_rules,
    from_dict,
    load,
    to_dict,
)
from orbitwin.rules import DUEL

BASE: dict[str, Any] = {
    "announced": {
        "dv_lim_m_s": 250.0,
        "r_cap_km": 5.0,
        "t_round_s": 1800.0,
        "a_lo_km": 7000.0,
        "a_hi_km": 7100.0,
        "pursuer_r0_km": [6878.0, 0.0, 0.0],
        "pursuer_v0_km_s": [0.0, 7.6, 0.0],
        "pursuer_epoch": "09 Nov 2026 00:00:00.000",
        "evader_r0_km": [6880.0, 30.0, 0.0],
        "evader_v0_km_s": [0.0, 7.6, 0.1],
        "evader_epoch": "09 Nov 2026 00:00:00.000",
    }
}


def _with(section: str, **values: Any) -> dict[str, Any]:
    raw = copy.deepcopy(BASE)
    raw.setdefault(section, {}).update(values)
    return raw


@pytest.mark.parametrize(
    ("section", "name"),
    [
        ("announced", "total_dv_limit_km_s"),
        ("choices", "kt"),
        ("choices", "Ct"),
        ("choices", "kv"),
        ("choices", "Cv"),
        ("announced", "T_max"),
        ("announced", "horizon_s"),
        ("announced", "four_ta"),
        ("choices", "min_separation_s"),
        ("announced", "intercept_radius_km"),
        ("announced", "sphere_km"),
    ],
)
def test_fields_of_other_rule_sets_are_refused_by_name(section: str, name: str) -> None:
    with pytest.raises(DuelParamsError, match=f"'{name}' is not a duel parameter"):
        from_dict(_with(section, **{name: 1.0}))


@pytest.mark.parametrize("name", ["Cr", "cr_km"])
def test_the_ranking_c_r_is_named_as_such(name: str) -> None:
    with pytest.raises(DuelParamsError, match=r"\(a ranking score coefficient\)"):
        from_dict(_with("choices", **{name: 300.0}))


def test_the_announced_mapping_covers_duel_exactly_once() -> None:
    assert set(ANNOUNCED_FIELDS) == set(DUEL.announced)
    mapped = [name for names in ANNOUNCED_FIELDS.values() for name in names]
    assert sorted(mapped) == sorted(f.name for f in dataclasses.fields(Announced))
    assert len(mapped) == len(set(mapped))


def test_a_full_file_round_trips() -> None:
    params = from_dict(BASE)
    assert from_dict(to_dict(params)) == params
    assert params.dv_lim_km_s == pytest.approx(0.25)


def test_an_interval_containing_zero_is_refused() -> None:
    raw = copy.deepcopy(BASE)
    raw["announced"].update(a_lo_km=-100.0, a_hi_km=100.0)
    with pytest.raises(DuelParamsError, match="contains a = 0"):
        from_dict(raw)


def test_negative_and_one_sided_intervals_are_accepted() -> None:
    raw = copy.deepcopy(BASE)
    raw["announced"].update(a_lo_km=-50000.0, a_hi_km=-20000.0)
    assert from_dict(raw).announced.a_hi_km == -20000.0
    raw["announced"].update(a_lo_km=8000.0, a_hi_km=float("inf"))
    assert from_dict(raw).announced.a_hi_km == float("inf")


def test_check_rules_refuses_a_changed_duel(monkeypatch: pytest.MonkeyPatch) -> None:
    check_rules()
    monkeypatch.setattr(params_module, "DUEL", dataclasses.replace(DUEL, over_limit="reject"))
    with pytest.raises(DuelParamsError, match="over_limit"):
        check_rules()


@pytest.mark.parametrize(
    ("section", "values", "message"),
    [
        ("choices", {"row_gap_setting": "rk89-1e-8"}, "row_gap_setting must be one of"),
        ("choices", {"a_margin_min_km": 0.0}, "a_margin_min_km must be positive"),
        ("choices", {"file_position_km": 0.0}, "file_position_km must be positive"),
        ("choices", {"deadline_s": DUEL.decision_s}, "leaves no time"),
        ("choices", {"deadline_s": DUEL.decision_s + 60.0}, "leaves no time"),
        ("choices", {"a_integration_km": -1.0}, "must not be negative"),
        ("announced", {"pursuer_r0_km": [0.0, 0.0, 0.0]}, "zero vector"),
        ("announced", {"evader_r0_km": [0.0, 0.0, 0.0]}, "zero vector"),
        ("readings", {"early_entry": True}, "not a duel parameter"),
    ],
)
def test_choices_that_cannot_work_are_refused(
    section: str, values: dict[str, Any], message: str
) -> None:
    with pytest.raises(DuelParamsError, match=message):
        from_dict(_with(section, **values))


def test_the_a_integration_term_defaults_to_the_measured_row_and_may_be_set() -> None:
    assert from_dict(BASE).choices.a_integration_km is None
    assert from_dict(_with("choices", a_integration_km=0.01)).choices.a_integration_km == 0.01
    assert from_dict(_with("choices", a_integration_km=None)).choices.a_integration_km is None
    assert from_dict(_with("choices", row_gap_setting="pd45")).choices.row_gap_setting == "pd45"


def test_a_key_given_twice_in_a_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "match.json"
    text = json.dumps(BASE)
    path.write_text(text.replace('"r_cap_km": 5.0', '"r_cap_km": 5.0, "r_cap_km": 3.0'),
                    encoding="utf-8")
    with pytest.raises(DuelParamsError, match="given twice"):
        load(path)
    path.write_text(text, encoding="utf-8")
    assert load(path) == from_dict(BASE)


def test_a_zero_max_step_is_refused_and_a_zero_row_gap_accepted() -> None:
    with pytest.raises(DuelParamsError, match="max_step_s must be positive"):
        from_dict(_with("choices", max_step_s=0.0))
    assert from_dict(_with("choices", row_gap_s_override=0.0)).choices.row_gap_s_override == 0.0


@pytest.mark.parametrize(
    ("section", "values", "message"),
    [
        ("readings", {"early_entry_wins": "false"}, "true or false"),
        ("readings", {"boundary_rows_judged": 0}, "true or false"),
        ("announced", {"dv_lim_m_s": True}, "must be a number"),
        ("announced", {"r_cap_km": float("inf")}, "must be finite"),
        ("announced", {"t_round_s": float("nan")}, "must be finite"),
        ("announced", {"a_lo_km": float("-inf")}, "must be finite"),
        ("announced", {"pursuer_v0_km_s": [0.0, float("inf"), 0.0]}, "must be finite"),
        ("announced", {"a_lo_km": 7100.0, "a_hi_km": 7100.0}, "a_lo_km < a_hi_km"),
        ("choices", {"lambert_keep": True}, "positive integer"),
        ("choices", {"lambert_keep": 2.0}, "positive integer"),
        ("choices", {"leaf_window_actions": 0}, "positive integer"),
    ],
)
def test_values_of_the_wrong_kind_are_refused(
    section: str, values: dict[str, Any], message: str
) -> None:
    with pytest.raises(DuelParamsError, match=message):
        from_dict(_with(section, **values))


def test_check_rules_pins_the_submission_decimals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(params_module, "DUEL",
                        dataclasses.replace(DUEL, submission_min_decimals=0))
    with pytest.raises(DuelParamsError, match="submission_min_decimals"):
        check_rules()


def test_check_rules_pins_the_decision_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(params_module, "DUEL", dataclasses.replace(DUEL, decision_s=300.0))
    with pytest.raises(DuelParamsError, match="decision_s"):
        check_rules()


def test_an_own_margin_under_gmat_error_over_a_round_is_refused() -> None:
    with pytest.raises(DuelParamsError, match="own_margin_km"):
        from_dict(_with("choices", own_margin_km=0.0))
    assert from_dict(_with("choices", own_margin_km=0.01)).choices.own_margin_km == 0.01
