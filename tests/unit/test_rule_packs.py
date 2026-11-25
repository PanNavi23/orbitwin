"""Rule sets read from rule packs: the 2026 pack, the reader's refusals, and outside packs."""

from __future__ import annotations

import dataclasses
import json
import sys
import textwrap
import types
from pathlib import Path

import pytest

from orbitwin import rules as rules_module
from orbitwin.cli import main
from orbitwin.config import OrbitWinConfig, rulebook_refusal, validate_config
from orbitwin.errors import ConfigurationError
from orbitwin.rules import (
    DUEL,
    PRELIMINARY,
    RANKING,
    DistanceTerm,
    DuelRules,
    InterceptionRules,
    RulePackError,
    interception_rule_names,
    interception_rules,
    read_rule_set,
    rule_set_document,
)

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "rules" / "golden_2026.json"
#: Fields the rule sets gained when they moved into packs; the golden file predates them.
_ADDED = {"id", "time_origin", "report_must_carry_team", "separation_clause", "burn_clauses",
          "miss_ranked", "time_origin_source", "pursuer_epoch", "pursuer_epoch_clause"}


def _config(**policy: object) -> OrbitWinConfig:
    config = OrbitWinConfig()
    return dataclasses.replace(config, policy=dataclasses.replace(config.policy, **policy))


def _plain(value: object) -> object:
    if isinstance(value, DistanceTerm):
        return value.name
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


@pytest.mark.parametrize("built_in", [PRELIMINARY, RANKING, DUEL], ids=lambda r: r.name)
def test_the_2026_pack_reads_back_every_number_the_code_held(built_in) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[built_in.name]
    assert golden.pop("__type__") == type(built_in).__name__
    loaded = {key: _plain(value) for key, value in dataclasses.asdict(built_in).items()
              if key not in _ADDED}

    assert loaded == golden


def test_the_added_fields_say_what_the_names_used_to() -> None:
    assert (RANKING.time_origin, RANKING.report_must_carry_team) == ("earliest", True)
    assert (PRELIMINARY.time_origin, PRELIMINARY.report_must_carry_team) == ("not-ruled", False)
    assert (RANKING.separation_clause, PRELIMINARY.separation_clause) == ("§4 note 1(d)", "§5")
    assert (RANKING.miss_ranked, PRELIMINARY.miss_ranked) == (True, False)
    assert RANKING.burn_clauses == "§4 note 1(a), (c), (d)"
    assert PRELIMINARY.burn_clauses == "§2, §5 (impulses and spacing)"
    assert [r.id for r in (PRELIMINARY, RANKING, DUEL)] == [
        "tasa-odc-2026/preliminary", "tasa-odc-2026/ranking", "tasa-odc-2026/duel"]


def test_a_config_may_name_a_rule_set_by_id_or_name() -> None:
    assert interception_rules("ranking") is interception_rules("tasa-odc-2026/ranking") is RANKING
    assert interception_rule_names() == [
        "preliminary", "ranking", "ranking-onsite", "tasa-odc-2026/preliminary",
        "tasa-odc-2026/ranking", "tasa-odc-2026/ranking-onsite"]
    with pytest.raises(ValueError, match="the duel is a different game"):
        interception_rules("duel")
    config = _config(rules="tasa-odc-2026/preliminary")
    validate_config(config)
    assert rulebook_refusal(config, None, "preliminary") is None


@pytest.mark.parametrize("built_in", [PRELIMINARY, RANKING, DUEL], ids=lambda r: r.name)
def test_a_rule_set_document_builds_the_same_rule_set(built_in) -> None:
    document = rule_set_document(built_in)
    assert document["format"] == 1
    kind = document.pop("kind")
    document.pop("format")
    cls = {"interception": InterceptionRules, "duel": DuelRules}[str(kind)]

    assert rules_module._build(cls, document, "test") == built_in


_RANKING_TOML = (Path(rules_module.__file__).parent / "rulepacks" / "tasa_odc_2026"
                 / "ranking.toml").read_text(encoding="utf-8")


def _edited(old: str, new: str) -> str:
    assert _RANKING_TOML.count(old) == 1, old
    return _RANKING_TOML.replace(old, new)


@pytest.mark.parametrize(
    ("text", "refusal"),
    [
        (_edited("advance = 4", "advance = 4\nbonus = 1"), r"unknown key\(s\) \['bonus'\]"),
        (_edited('horizon = "announced"', ""), "horizon is missing"),
        (_edited('horizon = "announced"', 'horizon = "whenever"'), "horizon is 'whenever'"),
        (_edited("advance = 4", "advance = 4.0"), "advance must be an integer"),
        (_edited("advance = 4", "advance = true"), "advance must be an integer"),
        (_edited("miss_scored = true", "miss_scored = 1"), "miss_scored must be true or false"),
        (_edited("impulse_limit_km_s = 0.25", "impulse_limit_km_s = nan"), "finite number"),
        (_edited("impulse_limit_km_s = 0.25", 'impulse_limit_km_s = "0.25"'), "finite number"),
        (_edited('distance_term = "LOGISTIC"', 'distance_term = "CUBIC"'), "distance_term is"),
        (_edited('score_units = ["s", "m/s"]', 'score_units = ["s"]'), "must hold 2 values"),
        (_edited('received = "2026-11-04"', 'received = "2026-11-04"\nurl = "x"'),
         r"\.source: unknown key"),
        (_edited('received = "2026-11-04"', ""), "received is missing"),
        (_edited('target_forces = "every external force of §7"', 'target_forces = ""'),
         "non-empty string"),
        (_edited("format = 1", "format = 2"), "format must be the integer 1"),
        (_edited("format = 1", "format = true"), "format must be the integer 1"),
        (_edited("format = 1", "format = 1.0"), "format must be the integer 1"),
        (_edited('kind = "interception"', 'kind = "relay"'), "kind must be one of"),
        (_edited('id = "tasa-odc-2026/ranking"', 'id = "tasa-odc-2026/round-2"'),
         "must read '<pack>/ranking'"),
        (_edited("format = 1", "format = 1\n[[x]"), "test.toml"),
    ],
)
def test_the_reader_refuses_what_it_cannot_read_as_written(text: str, refusal: str) -> None:
    with pytest.raises(RulePackError, match=refusal):
        read_rule_set(text, "test.toml")


def test_a_duel_has_no_total_cap_to_give() -> None:
    duel = (Path(rules_module.__file__).parent / "rulepacks" / "tasa_odc_2026"
            / "duel.toml").read_text(encoding="utf-8")
    with pytest.raises(RulePackError, match="total_dv_limit_km_s must be left out"):
        read_rule_set(duel.replace("rounds = 6", "rounds = 6\ntotal_dv_limit_km_s = 3.0"), "d")


def _install_pack(monkeypatch, tmp_path: Path, files: dict[str, str]) -> None:
    """An outside pack, a package with *.toml files, behind an orbitwin.rulepacks entry point."""
    package = tmp_path / "acme_contest"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name, text in files.items():
        (package / name).write_text(text, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("acme_contest", None)
    point = types.SimpleNamespace(name="acme", value="acme_contest",
                                  load=lambda: __import__("acme_contest"))
    monkeypatch.setattr(rules_module.metadata, "entry_points",
                        lambda group: [point] if group == rules_module.ENTRY_POINT_GROUP else [])
    rules_module._all.cache_clear()
    rules_module._read_installed.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_index():
    rules_module._all.cache_clear()
    rules_module._read_installed.cache_clear()
    yield
    rules_module._all.cache_clear()
    rules_module._read_installed.cache_clear()


_ACME = (_RANKING_TOML.replace('id = "tasa-odc-2026/ranking"', 'id = "acme-2027/final"')
         .replace('name = "ranking"', 'name = "final"')
         .replace('time_origin = "earliest"', 'time_origin = "not-ruled"')
         .replace(_RANKING_TOML[_RANKING_TOML.index("time_origin_source"):].split("\n")[0] + "\n",
                  "")
         .replace("impulse_limit_km_s = 0.25", "impulse_limit_km_s = 0.3"))


def test_an_installed_pack_is_read_listed_and_usable(monkeypatch, tmp_path, capsys) -> None:
    _install_pack(monkeypatch, tmp_path, {"final.toml": _ACME})

    acme = interception_rules("acme-2027/final")
    assert interception_rules("final") is acme and acme.impulse_limit_km_s == 0.3
    assert main(["rules"]) == 0
    assert "acme-2027/final" in capsys.readouterr().out
    validate_config(_config(rules="final", horizon_s=9000.0, dv_limit_km_s=0.29,
                            dv_safe_limit_km_s=0.28, total_dv_limit_km_s=3.0,
                            total_dv_safe_limit_km_s=2.99, mission_time_origin="target"))


def _named(name: str, pack: str = "acme-2027") -> str:
    return _ACME.replace('id = "acme-2027/final"', f'id = "{pack}/{name}"').replace(
        'name = "final"', f'name = "{name}"')


def test_an_installed_set_sharing_a_shipped_name_is_reached_by_id(monkeypatch, tmp_path,
                                                                  capsys) -> None:
    """"ranking" stays the shipped set's; the installed one answers to its id."""
    _install_pack(monkeypatch, tmp_path, {"ranking.toml": _named("ranking")})

    assert interception_rules("ranking") is RANKING
    assert interception_rules("acme-2027/ranking").impulse_limit_km_s == 0.3
    assert main(["rules"]) == 0
    assert "acme-2027/ranking" in capsys.readouterr().out


def test_a_name_two_installed_sets_share_names_neither(monkeypatch, tmp_path,
                                                     capsys) -> None:
    _install_pack(monkeypatch, tmp_path, {"a.toml": _named("final"),
                                          "b.toml": _named("final", "bcme-2027")})

    with pytest.raises(ValueError, match=r"'final' is shared by \['acme-2027/final', "
                                         r"'bcme-2027/final'\]; name it by id"):
        interception_rules("final")
    assert interception_rules("bcme-2027/final").id == "bcme-2027/final"
    assert interception_rules("ranking") is RANKING
    with pytest.raises(ConfigurationError, match=r"policy.rules: rule set name 'final' is shared"):
        validate_config(_config(rules="final", horizon_s=9000.0))
    with pytest.raises(SystemExit):
        main(["rank", "x.json", "--rules", "final"])
    assert "rule set name 'final' is shared by" in capsys.readouterr().err


def test_an_id_two_sets_claim_is_refused(monkeypatch, tmp_path) -> None:
    _install_pack(monkeypatch, tmp_path, {"a.toml": _named("final"), "b.toml": _named("final")})

    with pytest.raises(RulePackError, match="two rule sets have the id 'acme-2027/final'"):
        interception_rule_names()


def test_an_entry_point_naming_a_module_is_refused(monkeypatch, tmp_path) -> None:
    (tmp_path / "loose_rules.py").write_text("", encoding="utf-8")
    (tmp_path / "stray.toml").write_text("not a rule set", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    point = types.SimpleNamespace(name="loose", value="loose_rules",
                                  load=lambda: __import__("loose_rules"))
    monkeypatch.setattr(rules_module.metadata, "entry_points",
                        lambda group: [point] if group == rules_module.ENTRY_POINT_GROUP else [])

    with pytest.raises(RulePackError, match="names a module, not a package"):
        interception_rule_names()


@pytest.mark.parametrize(
    ("old", "new", "refusal"),
    [
        ("impulse_limit_km_s = 0.25", "impulse_limit_km_s = -0.25", "must be positive"),
        ("min_separation_s = 100.0", "min_separation_s = -1.0", "cannot be negative"),
        ('earth_crossing_clause = "§4 note 1(b)"', "", "given exactly when"),
        ('time_origin = "earliest"', 'time_origin = "not-ruled"', "time_origin_source is given"),
        ("penalty_points = 10.0", "penalty_points = -10.0", "cannot be negative"),
        ("miss_scored = true", "miss_scored = false", "miss_ranked needs miss_scored"),
        ('total_clause = "§4 note 1(c)"', "", "go together"),
        ('id = "tasa-odc-2026/ranking"', 'id = "/ranking"', "must read '<pack>/ranking'"),
    ],
)
def test_the_reader_refuses_values_that_cannot_hold(old: str, new: str, refusal: str) -> None:
    with pytest.raises(RulePackError, match=refusal):
        read_rule_set(_edited(old, new), "test.toml")


def test_a_rulebook_may_rule_no_spacing_or_no_earth_crossing() -> None:
    assert read_rule_set(_edited("min_separation_s = 100.0", "min_separation_s = 0.0"),
                         "t").min_separation_s == 0.0
    unruled = _edited('earth_crossing = "disqualify-before-interception"',
                      'earth_crossing = "not-ruled"')
    with pytest.raises(RulePackError, match="given exactly when"):
        read_rule_set(unruled, "t")
    assert read_rule_set(unruled.replace('earth_crossing_clause = "§4 note 1(b)"\n', ""),
                         "t").earth_crossing_clause is None


def test_a_rule_pack_error_is_a_configuration_error() -> None:
    assert issubclass(RulePackError, ConfigurationError)


def test_a_broken_pack_names_itself(monkeypatch, tmp_path) -> None:
    _install_pack(monkeypatch, tmp_path, {"final.toml": _ACME.replace("advance = 4", "")})

    with pytest.raises(RulePackError, match=r"acme \(acme_contest\)/final.toml: advance is"):
        interception_rule_names()


def test_the_ranking_time_origin_comes_from_the_rules() -> None:
    config = _config(rules="ranking", horizon_s=9000.0, mission_time_origin="target")
    with pytest.raises(ConfigurationError, match="under the ranking rules it is 'earliest'"):
        validate_config(config)


def test_the_rules_command_shows_one_set_or_refuses(capsys) -> None:
    assert main(["rules", "tasa-odc-2026/duel"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert (shown["kind"], shown["rounds"], shown["outcome_order"][0]) == ("duel", 6, "capture")
    assert "total_dv_limit_km_s" not in shown
    assert main(["rules", "nope"]) == 2
    assert "no rule set is named 'nope'" in capsys.readouterr().err
    assert main(["rules", "--json"]) == 0
    assert [entry["id"] for entry in json.loads(capsys.readouterr().out)] == [
        "tasa-odc-2026/duel", "tasa-odc-2026/preliminary", "tasa-odc-2026/ranking",
        "tasa-odc-2026/ranking-onsite"]


def test_the_pack_file_needs_no_reader_it_lacks() -> None:
    text = textwrap.dedent("""
        format = 1
        kind = "duel"
    """)
    with pytest.raises(RulePackError, match="id is missing"):
        read_rule_set(text, "empty.toml")


def test_an_installed_pack_may_not_use_a_shipped_pack_name(monkeypatch, tmp_path) -> None:
    _install_pack(monkeypatch, tmp_path, {"final.toml": _ACME.replace(
        'id = "acme-2027/final"', 'id = "tasa-odc-2026/final"')})

    with pytest.raises(RulePackError, match="uses a pack name OrbitWin ships"):
        interception_rule_names()


def test_a_broken_pack_does_not_break_commands_that_name_no_rules(monkeypatch, tmp_path,
                                                                   capsys) -> None:
    """--rules is checked when given, so building the parser reads no installed pack."""
    _install_pack(monkeypatch, tmp_path, {"final.toml": "format = 1\n"})

    with pytest.raises(SystemExit):
        main(["certify", "--help"])
    assert "--rules" in capsys.readouterr().out
    with pytest.raises(SystemExit) as refused:
        main(["rank", "x.json", "--rules", "ranking"])
    assert refused.value.code == 2
    assert "kind must be one of" in capsys.readouterr().err
