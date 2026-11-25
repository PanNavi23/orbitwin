"""The one place that converts m/s to km/s, projects onto the cap, and writes lines.

Submissions are m/s (PDF §3) and everything else in the engine is km/s. Our own
actions are held as integer centi-m/s triples, so the line we write carries
exactly two decimals and means the same however the organizer treats further
digits (DUEL_ENGINE.md section 2.1). The opponent's lines are parsed as written.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass

import numpy as np

from ..errors import OrbitWinError
from ..rules import DUEL

M_PER_KM = 1000.0
CENTI_PER_M = 100
#: Within this of an integer, a centi-m/s value is taken to be that integer.
_ON_GRID = 1.0e-6

ROLES = {"pursuer": "追捕方", "evader": "逃脫方"}
_ROLE_BY_TOKEN = {token: role for role, token in ROLES.items()} | {r: r for r in ROLES}


class SubmissionError(OrbitWinError):
    """A line that cannot be read as one impulse for one role and round."""

    code = "duel"


def project_m_s(vector_m_s: np.ndarray, cap_m_s: float) -> np.ndarray:
    """PDF §4.2: an impulse over the cap is scaled onto it; zero stays zero."""
    vector = np.asarray(vector_m_s, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise SubmissionError("an impulse is three finite numbers")
    if not math.isfinite(cap_m_s) or cap_m_s <= 0.0:
        raise SubmissionError("the cap must be positive and finite")
    norm = float(np.linalg.norm(vector))
    if norm > cap_m_s:
        projected: np.ndarray = vector * (cap_m_s / norm)
        return projected
    return vector.copy()


def executed_km_s(submitted_m_s: np.ndarray, cap_m_s: float) -> np.ndarray:
    """What the organizer executes, in km/s: project in m/s, convert once."""
    executed: np.ndarray = project_m_s(submitted_m_s, cap_m_s) / M_PER_KM
    return executed


@dataclass(frozen=True, slots=True)
class Impulse:
    """One of our impulses, in whole centi-m/s per EarthMJ2000Eq component."""

    centi: tuple[int, int, int]

    @property
    def m_s(self) -> np.ndarray:
        return np.array(self.centi, dtype=float) / CENTI_PER_M

    @property
    def km_s(self) -> np.ndarray:
        km_s: np.ndarray = self.m_s / M_PER_KM
        return km_s

    def norm_centi_squared(self) -> int:
        return sum(c * c for c in self.centi)


def _candidates(value_centi: float) -> tuple[int, ...]:
    nearest = round(value_centi)
    if abs(value_centi - nearest) <= _ON_GRID:
        return (int(nearest),)
    low = math.floor(value_centi)
    return (int(low), int(low) + 1)


def snap(vector_m_s: np.ndarray, cap_m_s: float) -> Impulse:
    """The grid corner nearest a projected vector whose norm stays within the cap.

    The vector is projected first. Each component's neighbours on the 0.01 m/s
    grid give at most eight corners; the corner towards zero never exceeds the
    projected norm, so at least one qualifies. A vector already on the grid maps
    to itself, which a float truncation does not guarantee.
    """
    projected = project_m_s(vector_m_s, cap_m_s) * CENTI_PER_M
    cap_centi = cap_m_s * CENTI_PER_M
    if abs(cap_centi - round(cap_centi)) <= 1.0e-9 * max(1.0, cap_centi):
        # A cap on the grid, such as 4.35 m/s, whose float times 100 is
        # 434.99999999999994: the grid point at the cap stays admissible.
        cap_centi = float(round(cap_centi))
    limit = cap_centi**2
    best: tuple[float, tuple[int, int, int]] | None = None
    for corner in itertools.product(*(_candidates(float(v)) for v in projected)):
        if sum(c * c for c in corner) > limit:
            continue
        distance = sum((c - float(v)) ** 2 for c, v in zip(corner, projected, strict=True))
        # Ties go to the smaller norm, then to the lexicographically first corner.
        key = (distance, sum(c * c for c in corner))
        if best is None or key < (best[0], sum(c * c for c in best[1])):
            best = (distance, (corner[0], corner[1], corner[2]))
    if best is None:
        # Only reachable if float noise put every corner a hair over the cap.
        toward_zero = tuple(math.trunc(float(v)) for v in projected)
        return Impulse((toward_zero[0], toward_zero[1], toward_zero[2]))
    return Impulse(best[1])


def _two_decimals(centi: int) -> str:
    sign = "-" if centi < 0 else ""
    whole, part = divmod(abs(centi), CENTI_PER_M)
    return f"{sign}{whole}.{part:02d}"


def format_line(role: str, round_number: int, impulse: Impulse) -> str:
    """`角色, 回合, ΔvX, ΔvY, ΔvZ` with exactly two decimals; zero is `0.00`."""
    if role not in ROLES:
        raise SubmissionError(f"unknown role {role!r}")
    if not 1 <= round_number <= DUEL.rounds:
        raise SubmissionError(f"round must be 1 to {DUEL.rounds}")
    return ", ".join([ROLES[role], str(round_number), *map(_two_decimals, impulse.centi)])


@dataclass(frozen=True, slots=True)
class ParsedLine:
    role: str
    round_number: int
    #: As written, in m/s.
    vector_m_s: tuple[float, float, float]
    #: The same, as an Impulse, when every component has at most two decimals.
    impulse: Impulse | None


#: ASCII digits only: `\d` and `str.isdigit` also take full-width, Arabic-Indic
#: and superscript digits, which the organizer's parser need not.
_NUMBER = re.compile(r"[+-]?[0-9]+(\.[0-9]+)?", re.ASCII)
_ROUND = re.compile(r"[0-9]+", re.ASCII)


def parse_line(text: str) -> ParsedLine:
    """Read a line back: role, round and three m/s components.

    ASCII and full-width commas are both accepted. Each component needs at least
    two decimals (PDF §3); a line that has none is refused rather than guessed.
    """
    fields = [field.strip() for field in re.split("[,\uff0c]", text.strip())]
    if len(fields) != 5:
        raise SubmissionError(f"expected 5 fields, got {len(fields)}: {text!r}")
    role = _ROLE_BY_TOKEN.get(fields[0])
    if role is None:
        raise SubmissionError(f"unknown role {fields[0]!r}")
    if not _ROUND.fullmatch(fields[1]) or not 1 <= int(fields[1]) <= DUEL.rounds:
        raise SubmissionError(f"round must be 1 to {DUEL.rounds}, got {fields[1]!r}")
    values: list[float] = []
    centi: list[int] = []
    on_grid = True
    for field in fields[2:]:
        if not _NUMBER.fullmatch(field):
            raise SubmissionError(f"not a number: {field!r}")
        decimals = len(field.split(".")[1]) if "." in field else 0
        if decimals < DUEL.submission_min_decimals:
            raise SubmissionError(
                f"{field!r} has {decimals} decimals; the rules ask for at least "
                f"{DUEL.submission_min_decimals}"
            )
        values.append(float(field))
        if decimals > 2 and field.split(".")[1][2:].strip("0"):
            on_grid = False
        else:
            negative = field.startswith("-")
            whole, part = field.lstrip("+-").split(".")
            magnitude = int(whole) * CENTI_PER_M + int((part + "00")[:2])
            centi.append(-magnitude if negative else magnitude)
    impulse = Impulse((centi[0], centi[1], centi[2])) if on_grid else None
    return ParsedLine(role, int(fields[1]), (values[0], values[1], values[2]), impulse)


def collect_round(lines: list[str], round_number: int) -> dict[str, ParsedLine]:
    """One line per role for one round (PDF §4.1: one impulse per round).

    A second line for the same role, or a line for another round, is refused.
    """
    collected: dict[str, ParsedLine] = {}
    for text in lines:
        parsed = parse_line(text)
        if parsed.round_number != round_number:
            raise SubmissionError(f"line for round {parsed.round_number} in round {round_number}")
        if parsed.role in collected:
            raise SubmissionError(f"a second {parsed.role} line for round {round_number}")
        collected[parsed.role] = parsed
    return collected


def executed_for(parsed: ParsedLine | None, cap_m_s: float) -> np.ndarray:
    """What runs for one role: its line projected, or the late default (PDF §3)."""
    if parsed is None:
        late: np.ndarray = np.array(DUEL.late_submission, dtype=float) / M_PER_KM
        return late
    return executed_km_s(np.array(parsed.vector_m_s), cap_m_s)
