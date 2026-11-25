"""Craft and match state, in km and km/s, EarthMJ2000Eq, with the epoch kept."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..errors import OrbitWinError
from .events import coast_one
from .params import DuelParams


class StateError(OrbitWinError):
    """A state the engine cannot use."""

    code = "duel"


@dataclass(frozen=True, slots=True)
class CraftState:
    r_km: tuple[float, float, float]
    v_km_s: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name, value in (("r_km", self.r_km), ("v_km_s", self.v_km_s)):
            if len(value) != 3 or not all(math.isfinite(float(x)) for x in value):
                raise StateError(f"{name} must be three finite numbers")
        if float(np.linalg.norm(self.r_km)) == 0.0:
            raise StateError("a craft at the centre of the Earth has no orbit")

    @property
    def r(self) -> np.ndarray:
        return np.array(self.r_km, dtype=float)

    @property
    def v(self) -> np.ndarray:
        return np.array(self.v_km_s, dtype=float)

    @classmethod
    def of(cls, r: np.ndarray, v: np.ndarray) -> CraftState:
        return cls(
            (float(r[0]), float(r[1]), float(r[2])), (float(v[0]), float(v[1]), float(v[2]))
        )


@dataclass(frozen=True, slots=True)
class MatchState:
    """The start of a round: the organizer's file, never our prediction."""

    round_number: int
    #: As the organizer wrote it; the engine does not need it for two-body motion.
    epoch: str
    pursuer: CraftState
    evader: CraftState


def specific_energy(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    return float(np.dot(v, v)) / 2.0 - mu / float(np.linalg.norm(r))


def semi_major_axis(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    """PDF section 5 eq. (2): a = 1/(2/|r| - |v|^2/mu); inf if parabolic, < 0 if open."""
    inverse = 2.0 / float(np.linalg.norm(r)) - float(np.dot(v, v)) / mu
    if inverse == 0.0:
        return math.inf
    return 1.0 / inverse


def energy_band(a_lo_km: float, a_hi_km: float, mu: float) -> tuple[float, float]:
    """The interval of energy -mu/(2a) for a in [a_lo, a_hi], which excludes 0.

    The energy increases with a on each side of a = 0, so the ends map in
    order. An upper end of `inf` maps to 0, which the caller treats as open.
    """
    def energy(a: float) -> float:
        return 0.0 if math.isinf(a) else -mu / (2.0 * a)

    return energy(a_lo_km), energy(a_hi_km)


#: GMAT's ModJulian day zero, 05 Jan 1941 12:00:00.
_MJD_ZERO = datetime(1941, 1, 5, 12, 0, 0)
_GREGORIAN = "%d %b %Y %H:%M:%S.%f"
_MJD = re.compile(r"[+-]?[0-9]+(\.[0-9]+)?", re.ASCII)
#: Epochs closer than this are the same. Two different Gregorian strings are
#: at least 1 ms apart, which at 15 km/s is 15 m of separation, so only float
#: noise counts as equal.
EPOCH_TOLERANCE_S = 1.0e-6


def epoch_seconds(text: str) -> tuple[str, float] | None:
    """(form, seconds since GMAT's ModJulian zero) for a Gregorian or ModJulian epoch.

    No time scale is attached: the file does not say whether it is UTC or A1,
    which differ by about 37 s, so only epochs of the same form compare.
    Leap seconds are not counted. None when the text is neither form.
    """
    text = text.strip().strip("'\"")
    if _MJD.fullmatch(text):
        return "modjulian", float(text) * 86400.0
    try:
        moment = datetime.strptime(text, _GREGORIAN)
    except ValueError:
        return None
    return "gregorian", (moment - _MJD_ZERO).total_seconds()


def match_start(params: DuelParams) -> MatchState:
    """The round-1 state from the announced values (section 2.2).

    Unequal epochs stop here (Q10). With `align_epochs` the earlier craft is
    coasted to the later epoch, which is then the match's epoch.
    """
    a = params.announced
    pursuer = CraftState(a.pursuer_r0_km, a.pursuer_v0_km_s)
    evader = CraftState(a.evader_r0_km, a.evader_v0_km_s)
    if a.pursuer_epoch == a.evader_epoch:
        return MatchState(1, a.pursuer_epoch, pursuer, evader)
    read_p, read_e = epoch_seconds(a.pursuer_epoch), epoch_seconds(a.evader_epoch)
    if read_p is None or read_e is None or read_p[0] != read_e[0]:
        raise StateError(f"the epochs {a.pursuer_epoch!r} and {a.evader_epoch!r} differ "
                         "and cannot be compared (unknown form, or two forms whose time "
                         "scales may differ)")
    lag = read_e[1] - read_p[1]
    if abs(lag) <= EPOCH_TOLERANCE_S:
        return MatchState(1, a.pursuer_epoch, pursuer, evader)
    if not params.readings.align_epochs:
        raise StateError(f"the evader's epoch is {lag:+.3f} s from the pursuer's; how the "
                         "organizer aligns them is open (Q10), so the reader stops. "
                         "Set the reading align_epochs once they confirm a two-body coast.")
    mu = params.choices.mu_km3_s2
    if lag > 0.0:
        pursuer = CraftState.of(*coast_one(pursuer.r, pursuer.v, lag, mu))
        return MatchState(1, a.evader_epoch, pursuer, evader)
    evader = CraftState.of(*coast_one(evader.r, evader.v, -lag, mu))
    return MatchState(1, a.pursuer_epoch, pursuer, evader)
