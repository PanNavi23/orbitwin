"""Contest rulebooks, typed, each number with the clause it comes from.

A rulebook is a TOML file in a rule pack (``orbitwin.rulepacks``), read and
checked here: an unknown key, a missing one or a value of the wrong kind is
refused, never defaulted. OrbitWin ships TASA's 2026 contest. Its preliminary
is history, kept so its submission can be reproduced; the ranking round and the
gold/bronze duel are different games and share no rule type. A day's announced
values (T_max, coefficients, ΔV_lim, R_cap, ...) and our own margins are not
rules and live in the configs.
"""

from __future__ import annotations

import argparse
import enum
import functools
import math
import tomllib
import types
from dataclasses import asdict, dataclass, fields
from importlib import metadata, resources
from importlib.resources.abc import Traversable
from typing import Literal, TypeVar, Union, assert_never, get_args, get_origin, get_type_hints

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class RuleSource:
    #: The file as the organizer handed it over, and its SHA-256.
    document: str
    sha256: str
    #: The date printed in its header. The ranking PDF kept 2026/10/06 when rule 2 was
    #: corrected on 2026-11-01 and says 2026/11/04 since the scoring revision.
    header_date: str
    received: str


class DistanceTerm(enum.Enum):
    """The score's first term, of x = Δr_min - 5 km."""

    #: 50 e^{-x/100}.
    EXPONENTIAL = "50 exp(-(Δr_min - 5)/100)"
    #: 100/(1 + e^{(x/C_r)^2}), C_r announced in km.
    LOGISTIC = "100/(1 + exp[((Δr_min - 5)/C_r)^2])"

    @property
    def takes_cr(self) -> bool:
        """Whether the term needs an announced C_r."""
        match self:
            case DistanceTerm.EXPONENTIAL:
                return False
            case DistanceTerm.LOGISTIC:
                return True
            case _:
                assert_never(self)


@dataclass(frozen=True, slots=True)
class InterceptionRules:
    """An interception scored on time, delta-v and distance: preliminary or ranking."""

    #: Unique across packs, "<pack>/<name>"; a config may name it by either.
    id: str
    name: str
    historical: bool
    source: RuleSource
    impulse_limit_km_s: float
    #: Tolerance over the impulse limit before it counts as exceeded.
    impulse_epsilon_km_s: float
    over_impulse_limit: Literal["penalty", "disqualify"]
    impulse_clause: str
    min_separation_s: float
    #: Points deducted per violating maneuver, the total floored at zero.
    penalty_points: float
    total_dv_limit_km_s: float | None
    total_clause: str | None
    intercept_radius_km: float
    horizon: Literal["4*T_A", "announced"]
    #: Whether a miss is a scored answer, and on which distance.
    miss_scored: bool
    #: Whether a scored miss is ranked, so worth submitting; else only a hit is submitted.
    miss_ranked: bool
    miss_distance: Literal["at-completion", "closest-approach"]
    distance_term: DistanceTerm
    distance_clause: str
    #: The units the rules fix for (T_team, ΔV_team), or None.
    score_units: tuple[str, str] | None
    earth_crossing: Literal["not-ruled", "disqualify-before-interception"]
    earth_crossing_clause: str | None
    #: How the validator read the completion time (measured 2026-09-05).
    completion_time: Literal["last-report-row"]
    target_forces: str
    advance: int
    #: Where T_team, T_max and Ct count from when the craft start apart, and what says so.
    time_origin: Literal["not-ruled", "earliest"]
    time_origin_source: str | None
    #: Whether the official report must carry the team number for certify to submit.
    report_must_carry_team: bool
    #: The clause on burn spacing, and the clauses on burns that a report check judges.
    separation_clause: str
    burn_clauses: str
    #: Whether a team sets the pursuer's epoch, and the clause that lets it.
    pursuer_epoch: Literal["fixed", "at-or-after-target"]
    pursuer_epoch_clause: str | None


@dataclass(frozen=True, slots=True)
class DuelRules:
    """The gold/bronze pursuit-evasion match. Two-body; one impulse a round each."""

    id: str
    name: str
    source: RuleSource
    rounds: int
    decision_s: float
    impulses_per_round: int
    over_limit: Literal["scale"]
    total_dv_limit_km_s: None
    frame: str
    submission_unit: Literal["m/s"]
    submission_min_decimals: int
    late_submission: tuple[float, float, float]
    dynamics: Literal["two-body point mass"]
    #: In the order PDF §5 lists them. Whether that order also ranks two events in
    #: one round, or the earliest decides, is not stated.
    outcome_order: tuple[str, ...]
    #: What the organizer announces on the day; none of it is fixed here.
    announced: tuple[str, ...]
    role_choice: str
    network_and_generative_ai_banned: bool


#: The rule pack format this module reads.
PACK_FORMAT = 1
#: Packs outside OrbitWin: each entry point names a package holding *.toml rule sets.
ENTRY_POINT_GROUP = "orbitwin.rulepacks"
_KINDS: dict[str, type[InterceptionRules] | type[DuelRules]] = {
    "interception": InterceptionRules,
    "duel": DuelRules,
}


_T = TypeVar("_T")


class RulePackError(ConfigurationError, ValueError):
    """A rule set that cannot be read as written."""

    code = "rules"


def _value(where: str, hint: object, raw: object) -> object:
    """``raw`` as ``hint`` describes it, or RulePackError."""
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        (inner,) = (arg for arg in get_args(hint) if arg is not type(None))
        return _value(where, inner, raw)
    if origin is Literal:
        if raw not in get_args(hint):
            raise RulePackError(f"{where} is {raw!r}; it must be one of {list(get_args(hint))}")
        return raw
    if origin is tuple:
        args = get_args(hint)
        if not isinstance(raw, list):
            raise RulePackError(f"{where} must be an array")
        if args[-1] is not Ellipsis and len(raw) != len(args):
            raise RulePackError(f"{where} must hold {len(args)} values, not {len(raw)}")
        kinds = [args[0]] * len(raw) if args[-1] is Ellipsis else list(args)
        return tuple(_value(f"{where}[{i}]", kind, item) for i, (kind, item)
                     in enumerate(zip(kinds, raw, strict=True)))
    if hint is DistanceTerm:
        if not isinstance(raw, str) or raw not in DistanceTerm.__members__:
            raise RulePackError(
                f"{where} is {raw!r}; it must be one of {list(DistanceTerm.__members__)}"
            )
        return DistanceTerm[raw]
    if hint is RuleSource:
        if not isinstance(raw, dict):
            raise RulePackError(f"{where} must be a table")
        return _build(RuleSource, raw, where)
    if hint is bool:
        if not isinstance(raw, bool):
            raise RulePackError(f"{where} must be true or false, not {raw!r}")
        return raw
    if hint is int:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise RulePackError(f"{where} must be an integer, not {raw!r}")
        return raw
    if hint is float:
        if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
            raise RulePackError(f"{where} must be a finite number, not {raw!r}")
        return float(raw)
    if hint is str:
        if not isinstance(raw, str) or not raw:
            raise RulePackError(f"{where} must be a non-empty string, not {raw!r}")
        return raw
    raise RulePackError(f"{where}: no reader for {hint!r}")


def _build(cls: type[_T], table: dict[str, object], where: str) -> _T:
    """``cls`` from ``table``: every field present unless it may be None, nothing else."""
    hints = get_type_hints(cls)
    names = [field.name for field in fields(cls)]  # type: ignore[arg-type]
    unknown = sorted(set(table) - set(names))
    if unknown:
        raise RulePackError(f"{where}: unknown key(s) {unknown}")
    values: dict[str, object] = {}
    for name in names:
        hint = hints[name]
        optional = type(None) in get_args(hint) or hint is type(None)
        if name not in table:
            if not optional:
                raise RulePackError(f"{where}: {name} is missing")
            values[name] = None
        elif hint is type(None):
            raise RulePackError(f"{where}: {name} must be left out")
        else:
            values[name] = _value(f"{where}.{name}", hint, table[name])
    return cls(**values)


def read_rule_set(text: str, where: str) -> InterceptionRules | DuelRules:
    """One rule set from the TOML in ``text``; ``where`` names it in errors."""
    try:
        table = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RulePackError(f"{where}: {exc}") from None
    found = table.pop("format", None)
    if isinstance(found, bool) or not isinstance(found, int) or found != PACK_FORMAT:
        raise RulePackError(f"{where}: format must be the integer {PACK_FORMAT}, not {found!r}")
    kind = table.pop("kind", None)
    if kind not in _KINDS:
        raise RulePackError(f"{where}: kind must be one of {sorted(_KINDS)}, not {kind!r}")
    rules = _build(_KINDS[kind], table, where)
    pack, _, rest = rules.id.partition("/")
    if not pack or rest != rules.name or "/" in rules.name:
        raise RulePackError(f"{where}: id {rules.id!r} must read '<pack>/{rules.name}'")
    if isinstance(rules, InterceptionRules):
        _check_interception(rules, where)
    return rules


def _check_interception(rules: InterceptionRules, where: str) -> None:
    """What the types cannot say: limits are positive or not negative, and paired fields agree."""
    positive = {"impulse_limit_km_s": rules.impulse_limit_km_s,
                "intercept_radius_km": rules.intercept_radius_km,
                "total_dv_limit_km_s": rules.total_dv_limit_km_s}
    for name, value in positive.items():
        if value is not None and value <= 0.0:
            raise RulePackError(f"{where}.{name} must be positive, not {value}")
    for name, value in {"impulse_epsilon_km_s": rules.impulse_epsilon_km_s,
                        "min_separation_s": rules.min_separation_s,
                        "penalty_points": rules.penalty_points,
                        "advance": float(rules.advance)}.items():
        if value < 0.0:
            raise RulePackError(f"{where}.{name} cannot be negative, not {value}")
    if rules.miss_ranked and not rules.miss_scored:
        raise RulePackError(f"{where}: miss_ranked needs miss_scored")
    if (rules.total_dv_limit_km_s is None) != (rules.total_clause is None):
        raise RulePackError(f"{where}: total_dv_limit_km_s and total_clause go together")
    if (rules.time_origin == "not-ruled") != (rules.time_origin_source is None):
        raise RulePackError(f"{where}: time_origin_source is given exactly when "
                            "time_origin is ruled")
    if (rules.pursuer_epoch == "fixed") != (rules.pursuer_epoch_clause is None):
        raise RulePackError(f"{where}: pursuer_epoch_clause is given exactly when the "
                            "pursuer's epoch is not fixed")
    if (rules.earth_crossing == "not-ruled") != (rules.earth_crossing_clause is None):
        raise RulePackError(f"{where}: earth_crossing_clause is given exactly when "
                            "earth_crossing is ruled")


def _read_package(root: Traversable, origin: str) -> list[InterceptionRules | DuelRules]:
    found = sorted((item for item in root.iterdir() if item.name.endswith(".toml")),
                   key=lambda item: item.name)
    return [read_rule_set(item.read_text(encoding="utf-8"), f"{origin}/{item.name}")
            for item in found]


def _index(sets: list[InterceptionRules | DuelRules],
           into: dict[str, InterceptionRules | DuelRules],
           shared: dict[str, list[str]] | None = None) -> None:
    """Index by id and name. Two rule sets with one id are refused. Among the built-in
    sets a shared name is refused too; an installed set whose name another holds is
    reachable by its id only, and ``shared`` records the name and every id claiming it."""
    for rules in sets:
        if rules.id in into:
            raise RulePackError(f"two rule sets have the id {rules.id!r}")
        into[rules.id] = rules
        holder = into.get(rules.name)
        if holder is None and (shared is None or rules.name not in shared):
            into[rules.name] = rules
        elif shared is None:
            raise RulePackError(f"two rule sets are named {rules.name!r}: "
                                f"{holder.id if holder else '?'} and {rules.id}")
        else:
            claims = shared.setdefault(rules.name, [holder.id] if holder else [])
            claims.append(rules.id)


def _builtin() -> dict[str, InterceptionRules | DuelRules]:
    packs = resources.files("orbitwin.rulepacks")
    index: dict[str, InterceptionRules | DuelRules] = {}
    for pack in sorted((item for item in packs.iterdir() if item.is_dir()
                        and item.joinpath("__init__.py").is_file()), key=lambda item: item.name):
        _index(_read_package(pack, f"orbitwin.rulepacks.{pack.name}"), index)
    return index


_BUILTIN = _builtin()


@functools.cache
def _all() -> dict[str, InterceptionRules | DuelRules]:
    """The built-in rule sets and every installed pack's, read on first use."""
    return _read_installed()[0]


@functools.cache
def _read_installed() -> tuple[dict[str, InterceptionRules | DuelRules], dict[str, list[str]]]:
    index = dict(_BUILTIN)
    shared: dict[str, list[str]] = {}
    ours = {rules.id.split("/")[0] for rules in _BUILTIN.values()}
    for point in sorted(metadata.entry_points(group=ENTRY_POINT_GROUP), key=lambda ep: ep.name):
        where = f"{point.name} ({point.value})"
        try:
            package = point.load()
            if not hasattr(package, "__path__"):
                raise RulePackError("it names a module, not a package")
            root = resources.files(package)
        except Exception as exc:
            raise RulePackError(f"rule pack entry point {where}: {exc}") from exc
        found = _read_package(root, where)
        taken = sorted({r.id for r in found if r.id.split("/")[0] in ours})
        if taken:
            raise RulePackError(f"rule pack {where} uses a pack name OrbitWin ships: {taken}")
        _index(found, index, shared)
    for name in shared:
        # A built-in name stays the built-in set's; any other shared name names no one.
        if name in index and index[name].id not in {r.id for r in _BUILTIN.values()}:
            del index[name]
    return index, shared


def rule_set_ambiguity(name: str) -> str | None:
    """Why ``name`` names no rule set although some carry it, or None."""
    claims = _read_installed()[1].get(name)
    if claims is None or name in _BUILTIN:
        return None
    return f"rule set name {name!r} is shared by {sorted(claims)}; name it by id"


def rule_sets() -> list[InterceptionRules | DuelRules]:
    """Every rule set, built-in and installed, once each, by id."""
    return sorted({rules.id: rules for rules in _all().values()}.values(), key=lambda r: r.id)


def rule_set(name: str) -> InterceptionRules | DuelRules:
    """A rule set by id or name, of either kind."""
    shared = rule_set_ambiguity(name)
    if shared is not None:
        raise KeyError(shared)
    try:
        return _all()[name]
    except KeyError:
        known = sorted({key for key in _all()})
        raise KeyError(f"no rule set is named {name!r}; known: {known}") from None


def rule_set_document(rules: InterceptionRules | DuelRules) -> dict[str, object]:
    """``rules`` as plain JSON-ready data, keyed as its file is."""

    def plain(value: object) -> object:
        if isinstance(value, enum.Enum):
            return value.name
        if isinstance(value, tuple):
            return [plain(item) for item in value]
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        return value

    document = plain(asdict(rules))
    assert isinstance(document, dict)
    kind = next(key for key, cls in _KINDS.items() if isinstance(rules, cls))
    return {"format": PACK_FORMAT, "kind": kind,
            **{key: item for key, item in document.items() if item is not None}}


def rules_name(text: str) -> str:
    """argparse type for --rules: an interception rule set's id or name, checked when given."""
    try:
        interception_rules(text)
    except RulePackError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    except ValueError:
        raise argparse.ArgumentTypeError(
            rule_set_ambiguity(text)
            or f"{text!r} is not an interception rule set; one of {interception_rule_names()}"
        ) from None
    return text


def interception_rule_names() -> list[str]:
    """What policy.rules and --rules accept: each interception rule set's name and id."""
    return sorted(key for key, rules in _all().items() if isinstance(rules, InterceptionRules))


def _builtin_interception(name: str) -> InterceptionRules:
    rules = _BUILTIN[name]
    if not isinstance(rules, InterceptionRules):
        raise RulePackError(f"{name} is not an interception rule set")
    return rules


def _builtin_duel(name: str) -> DuelRules:
    rules = _BUILTIN[name]
    if not isinstance(rules, DuelRules):
        raise RulePackError(f"{name} is not a duel rule set")
    return rules


PRELIMINARY = _builtin_interception("tasa-odc-2026/preliminary")
RANKING = _builtin_interception("tasa-odc-2026/ranking")
DUEL = _builtin_duel("tasa-odc-2026/duel")

#: The built-in rule sets a config may name in policy.rules, by name and id.
INTERCEPTION_RULES: dict[str, InterceptionRules] = {
    key: rules for key, rules in _BUILTIN.items() if isinstance(rules, InterceptionRules)
}


def rule_set_record(name: str) -> dict[str, object]:
    """Which rulebook an output was made under, for its record."""
    rules = interception_rules(name)
    return {
        "name": rules.name,
        "id": rules.id,
        "document": rules.source.document,
        "sha256": rules.source.sha256,
        "header_date": rules.source.header_date,
        "historical": rules.historical,
    }


def interception_rules(name: str) -> InterceptionRules:
    shared = rule_set_ambiguity(name)
    if shared is not None:
        raise ValueError(shared)
    rules = _all().get(name)
    if not isinstance(rules, InterceptionRules):
        raise ValueError(
            f"policy.rules must be one of {interception_rule_names()}, got {name!r}; "
            "the duel is a different game, not a mission config"
        )
    return rules
