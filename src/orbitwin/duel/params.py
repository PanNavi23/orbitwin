"""A match's parameters: what was announced, what we chose, how we read open rules.

Rule constants are not here; they come from `orbitwin.rules.DUEL`. Loading is
strict: a field this type does not define is refused by name, and fields that
belong to the ranking or preliminary rules get a message saying so, because
copying a ranking config into a duel match is the mistake this guards against.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import OrbitWinError
from ..rules import DUEL
from .gmat_settings import SETTINGS
from .units import M_PER_KM

#: Earth's mu in km^3/s^2 as GMAT R2026a uses it for a point-mass Earth. All five
#: Earth potential files in data/gravity/earth carry 3.986004415e14 m^3/s^2, and
#: scripts/duel_gmat_check.py measured it on GmatConsole (2026-11-02): a day of
#: RK89 at 1e-12 under both `PrimaryBodies` degree 0 and `PointMasses = {Earth}`
#: fits 398600.44149991 and 398600.44149993, and agrees with our coast to
#: 2.0e-6 km, against 1.7e-3 km at 398600.4418.
GMAT_MU_KM3_S2 = 398600.4415
#: GMAT's Earth: equatorial radius and flattening, so R_polar = R_eq (1 - f).
R_EQ_KM = 6378.1363
FLATTENING = 0.0033527
R_POLAR_KM = R_EQ_KM * (1.0 - FLATTENING)


class DuelParamsError(OrbitWinError):
    """A parameter file this match cannot use."""

    code = "duel"


#: Fields that other rule sets use. Named here only to explain the refusal.
_FOREIGN = {
    "kt": "a ranking score coefficient",
    "ct": "a ranking score coefficient",
    "kv": "a ranking score coefficient",
    "cv": "a ranking score coefficient",
    "cr": "a ranking score coefficient",
    "cr_km": "a ranking score coefficient",
    "t_max": "the ranking horizon",
    "t_max_s": "the ranking horizon",
    "horizon_s": "a preliminary or ranking horizon",
    "four_ta": "the preliminary 4·T_A horizon",
    "4ta": "the preliminary 4·T_A horizon",
    "total_dv_limit_km_s": "a total Δv cap, which the duel does not have (PDF §4)",
    "total_dv_cap": "a total Δv cap, which the duel does not have (PDF §4)",
    "min_separation_s": "the 100 s burn separation of the ranking rules",
    "burn_separation_s": "the 100 s burn separation of the ranking rules",
    "intercept_radius_km": "the 5 km interception sphere; the duel uses the announced r_cap_km",
    "interception_radius_km": "the 5 km interception sphere; the duel uses the announced r_cap_km",
    "sphere_km": "the 5 km interception sphere; the duel uses the announced r_cap_km",
}


@dataclass(frozen=True, slots=True)
class Announced:
    """PDF §7: announced on the day, one set per match."""

    dv_lim_m_s: float
    r_cap_km: float
    t_round_s: float
    a_lo_km: float
    #: `inf` for a one-sided interval.
    a_hi_km: float
    pursuer_r0_km: tuple[float, float, float]
    pursuer_v0_km_s: tuple[float, float, float]
    pursuer_epoch: str
    evader_r0_km: tuple[float, float, float]
    evader_v0_km_s: tuple[float, float, float]
    evader_epoch: str


#: Each `DUEL.announced` entry and the fields that carry it.
ANNOUNCED_FIELDS: dict[str, tuple[str, ...]] = {
    "r0, v0 and epoch of each craft": (
        "pursuer_r0_km",
        "pursuer_v0_km_s",
        "pursuer_epoch",
        "evader_r0_km",
        "evader_v0_km_s",
        "evader_epoch",
    ),
    "safe semimajor-axis interval": ("a_lo_km", "a_hi_km"),
    "impulse limit ΔV_lim": ("dv_lim_m_s",),
    "capture radius R_cap": ("r_cap_km",),
    "round time T_round": ("t_round_s",),
}


@dataclass(frozen=True, slots=True)
class Choices:
    """Ours, not the organizer's. Defaults are argued in DUEL_ENGINE.md."""

    mu_km3_s2: float = GMAT_MU_KM3_S2
    own_margin_km: float = 10.0
    a_margin_min_km: float = 0.001
    #: Live floors from the after-round check (section 6.4); they only rise.
    a_margin_floor_km: float = 0.0
    dr_int_floor_km: float = 0.0
    #: GMAT's error in a over one round (section 2.5); None takes it from the
    #: measured row of `row_gap_setting`.
    a_integration_km: float | None = None
    #: Resolution of the published states; until a file is seen, these defaults.
    file_position_km: float = 1.0e-3
    file_velocity_km_s: float = 1.0e-6
    knot_spacing_s: float = 60.0
    #: The propagator setting assumed for the organizer's GMAT, a row of
    #: `gmat_settings.SETTINGS`: row gap, position error and a error.
    row_gap_setting: str = "cautious"
    #: A row gap the organizer states; 0 means every instant is judged.
    row_gap_s_override: float | None = None
    #: The organizer's Propagator MaxStep, when known: no row gap exceeds it.
    max_step_s: float | None = None
    lambert_keep: int = 4
    lambert_evader_targets: int = 8
    leaf_window_actions: int = 8
    deadline_s: float = DUEL.decision_s - 180.0


@dataclass(frozen=True, slots=True)
class Readings:
    """How we read an open rule; each is tied to its question for the organizer."""

    #: Q9: does reaching the interval before round 6 win?
    early_entry_wins: bool = False
    #: Q6: is the state at every round boundary judged? With true, an event
    #: that holds there counts whatever its dwell (the fourth exception).
    boundary_rows_judged: bool = True
    #: Q10: coast the earlier craft to the later epoch if the epochs differ.
    align_epochs: bool = False


@dataclass(frozen=True, slots=True)
class DuelParams:
    announced: Announced
    choices: Choices = field(default_factory=Choices)
    readings: Readings = field(default_factory=Readings)

    @property
    def dv_lim_km_s(self) -> float:
        return self.announced.dv_lim_m_s / M_PER_KM


def check_rules() -> None:
    """Refuse to run if `DUEL` no longer says what this engine implements."""
    expected = {
        "rounds": 6,
        "impulses_per_round": 1,
        "over_limit": "scale",
        "total_dv_limit_km_s": None,
        "frame": "EarthMJ2000Eq",
        "submission_unit": "m/s",
        "submission_min_decimals": 2,
        "decision_s": 600.0,
        "late_submission": (0.0, 0.0, 0.0),
        "dynamics": "two-body point mass",
        "outcome_order": ("capture", "earth-crossing", "terminal-semimajor-axis"),
    }
    changed = {name: getattr(DUEL, name) for name, value in expected.items()
               if getattr(DUEL, name) != value}
    if changed:
        raise DuelParamsError(f"orbitwin.rules.DUEL changed; the duel engine implements "
                              f"something else: {changed}")
    if set(ANNOUNCED_FIELDS) != set(DUEL.announced):
        raise DuelParamsError("DUEL.announced no longer matches the fields DuelParams maps")


def _finite(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DuelParamsError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if math.isnan(number) or (math.isinf(number) and name != "a_hi_km"):
        raise DuelParamsError(f"{name} must be finite, got {value!r}")
    if positive and not number > 0.0:
        raise DuelParamsError(f"{name} must be positive, got {value!r}")
    return number


def _vector(name: str, value: Any, *, nonzero: bool = False) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise DuelParamsError(f"{name} must be three numbers")
    x, y, z = (_finite(name, v) for v in value)
    if nonzero and x == y == z == 0.0:
        raise DuelParamsError(f"{name} is the zero vector; a craft there has no orbit")
    return (x, y, z)


def _refuse_unknown(group: str, given: dict[str, Any], known: set[str]) -> None:
    for name in given:
        if name in known:
            continue
        reason = _FOREIGN.get(name.lower())
        detail = f" ({reason})" if reason else ""
        raise DuelParamsError(f"{name!r} is not a duel parameter{detail}; group {group!r}")


#: Choices that must be strictly positive; the rest may be zero.
_POSITIVE = {"mu_km3_s2", "knot_spacing_s", "deadline_s", "a_margin_min_km",
             "file_position_km", "file_velocity_km_s"}


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise DuelParamsError(f"{name!r} must be a mapping")
    return value


def from_dict(raw: dict[str, Any]) -> DuelParams:
    if not isinstance(raw, dict):
        raise DuelParamsError("a parameter file is a mapping")
    _refuse_unknown("top level", raw, {"announced", "choices", "readings"})
    announced = _section(raw, "announced")
    choices = _section(raw, "choices")
    readings = _section(raw, "readings")
    _refuse_unknown("announced", announced, {f.name for f in dataclasses.fields(Announced)})
    _refuse_unknown("choices", choices, {f.name for f in dataclasses.fields(Choices)})
    _refuse_unknown("readings", readings, {f.name for f in dataclasses.fields(Readings)})
    missing = {f.name for f in dataclasses.fields(Announced)} - set(announced)
    if missing:
        raise DuelParamsError(f"announced values missing: {sorted(missing)}")

    a_lo = _finite("a_lo_km", announced["a_lo_km"])
    a_hi = _finite("a_hi_km", announced["a_hi_km"])
    if not a_lo < a_hi:
        raise DuelParamsError("the safe interval needs a_lo_km < a_hi_km")
    if a_lo <= 0.0 <= a_hi:
        raise DuelParamsError("a safe interval that contains a = 0 has no clear meaning (Q3)")
    for name in ("pursuer_epoch", "evader_epoch"):
        if not isinstance(announced[name], str) or not announced[name].strip():
            raise DuelParamsError(f"{name} must be the epoch as the organizer wrote it")
    parsed_announced = Announced(
        dv_lim_m_s=_finite("dv_lim_m_s", announced["dv_lim_m_s"], positive=True),
        r_cap_km=_finite("r_cap_km", announced["r_cap_km"], positive=True),
        t_round_s=_finite("t_round_s", announced["t_round_s"], positive=True),
        a_lo_km=a_lo,
        a_hi_km=a_hi,
        pursuer_r0_km=_vector("pursuer_r0_km", announced["pursuer_r0_km"], nonzero=True),
        pursuer_v0_km_s=_vector("pursuer_v0_km_s", announced["pursuer_v0_km_s"]),
        pursuer_epoch=announced["pursuer_epoch"].strip(),
        evader_r0_km=_vector("evader_r0_km", announced["evader_r0_km"], nonzero=True),
        evader_v0_km_s=_vector("evader_v0_km_s", announced["evader_v0_km_s"]),
        evader_epoch=announced["evader_epoch"].strip(),
    )
    defaults = Choices()
    values: dict[str, Any] = {}
    for f in dataclasses.fields(Choices):
        if f.name not in choices:
            continue
        value = choices[f.name]
        default = getattr(defaults, f.name)
        if f.name == "row_gap_setting":
            if not isinstance(value, str) or value not in SETTINGS:
                raise DuelParamsError(f"row_gap_setting must be one of {sorted(SETTINGS)}, "
                                      f"got {value!r}")
            values[f.name] = value
        elif f.name in {"row_gap_s_override", "a_integration_km", "max_step_s"}:
            values[f.name] = None if value is None else _finite(
                f.name, value, positive=f.name == "max_step_s")
            if values[f.name] is not None and values[f.name] < 0.0:
                raise DuelParamsError(f"{f.name} must not be negative, got {value!r}")
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DuelParamsError(f"{f.name} must be a positive integer")
            values[f.name] = value
        else:
            number = _finite(f.name, value)
            if number < 0.0 or (f.name in _POSITIVE and number == 0.0):
                raise DuelParamsError(f"{f.name} must be positive, got {value!r}")
            values[f.name] = number
    setting = SETTINGS[values.get("row_gap_setting", defaults.row_gap_setting)]
    own = values.get("own_margin_km", defaults.own_margin_km)
    error = setting.position_error_km(parsed_announced.t_round_s)
    if own < error:
        raise DuelParamsError(f"own_margin_km {own} is under GMAT's position error over a "
                              f"round, {error:.3g} km: GMAT could put us below the surface "
                              "while our coast stays above it")
    deadline = values.get("deadline_s", defaults.deadline_s)
    if deadline >= DUEL.decision_s:
        raise DuelParamsError(f"deadline_s {deadline} leaves no time to submit inside the "
                              f"{DUEL.decision_s} s decision window")
    reading_values: dict[str, bool] = {}
    for name, value in readings.items():
        if not isinstance(value, bool):
            raise DuelParamsError(f"reading {name} must be true or false")
        reading_values[name] = value
    return DuelParams(parsed_announced, Choices(**values), Readings(**reading_values))


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise DuelParamsError(f"{key!r} is given twice; which one counts is not guessed")
        seen[key] = value
    return seen


def load(path: str | Path) -> DuelParams:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"),
                         object_pairs_hook=_no_duplicates)
    except (OSError, json.JSONDecodeError) as error:
        raise DuelParamsError(f"cannot read {path}: {error}") from error
    return from_dict(raw)


def to_dict(params: DuelParams) -> dict[str, Any]:
    return {
        "announced": dataclasses.asdict(params.announced),
        "choices": dataclasses.asdict(params.choices),
        "readings": dataclasses.asdict(params.readings),
    }
