"""GMAT corrections and verifications that raised, named so none is lost silently.

A GmatConsole crash, a timeout under load or a report cut short ends that plan's
GMAT run, and solve then chooses among the others. GMAT's verdicts on a flight
(an altitude floor, a constraint, a stop outside the sphere) are not listed. The plan may have been the best one, so solve and
rank both list what raised, ordered by the score each plan was chosen by.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .models import Candidate

#: An error shown in the warning is cut to this many characters; the record keeps it whole.
SHOWN_ERROR_CHARS = 240


def pre_gmat_score(candidate: Candidate) -> float | None:
    """The score the candidate was chosen by, before a GMAT flight rewrote it."""
    before = candidate.metadata.get("analytical_before_gmat")
    if isinstance(before, Mapping) and "score" in before:
        recorded = before["score"]
        return None if recorded is None else float(recorded)
    # A run that died before verify recorded it never reached apply_score either.
    return candidate.score


def verification_error(candidate: Candidate) -> str | None:
    """What solve's GMAT verification of the candidate raised, or None, also when
    what it raised is GMAT's verdict on the flight (errors.GmatVerdictError)."""
    message = candidate.metadata.get("gmat_verification_error")
    if not message or candidate.metadata.get("gmat_verification_verdict") is True:
        return None
    kind = candidate.metadata.get("gmat_verification_error_type")
    return f"{kind}: {message}" if kind else str(message)


def _shown(score: float | None) -> str:
    return "unscored" if score is None else f"{score:.6g}"


def gmat_failures(
    failed: Iterable[tuple[str, Candidate, str]],
    best: Candidate | None,
    *,
    archived: set[str],
    who: str,
    only_above: bool = False,
    score_of: Callable[[Candidate], float | None] = pre_gmat_score,
) -> dict[str, Any]:
    """Each (kind, candidate, error) that raised, highest pre-GMAT score first.

    ``above_best`` marks one that scored above ``best`` before GMAT, which
    every one does when there is no best. ``who`` names the choice the warning
    speaks of; ``only_above`` drops the rest from it; ``score_of`` reads the
    pre-GMAT score. The warning is None when there is nothing to say.
    """
    best_score = score_of(best) if best is not None else None
    runs: list[dict[str, Any]] = []
    for kind, candidate, error in failed:
        score = score_of(candidate)
        above = best is None or (
            score is not None and best_score is not None and score > best_score
        )
        runs.append({
            "kind": kind,
            "candidate_id": candidate.candidate_id,
            "branch": candidate.branch,
            "pre_gmat_score": score,
            "above_best": above,
            "in_archive": candidate.candidate_id in archived,
            "error": error,
        })
    runs.sort(key=lambda run: (
        run["pre_gmat_score"] is None, -(run["pre_gmat_score"] or 0.0),
        run["candidate_id"], run["kind"],
    ))
    shown = [run for run in runs if run["above_best"] or not only_above]
    if not shown:
        return {"warning": None, "runs": runs}
    above_count = sum(1 for run in shown if run["above_best"])
    if best is None:
        chosen = f"{who} is none, so every one of them ranks above it"
    else:
        now = f", now {best.score:.6g}" if best.score is not None else ""
        chosen = (
            f"{who} is {best.candidate_id} (pre-GMAT score {_shown(best_score)}{now}), and "
            f"{above_count} of them scored above it before GMAT"
        )
    listed = "; ".join(
        ("ABOVE THE BEST: " if run["above_best"] else "")
        + f"{run['kind']} of {run['candidate_id']} ({run['branch']}, pre-GMAT score "
        + f"{_shown(run['pre_gmat_score'])}"
        + ("" if run["in_archive"] else ", not in candidates.json")
        + f"): {str(run['error'])[:SHOWN_ERROR_CHARS]}"
        for run in shown
    )
    warning = (
        f"{len(shown)} GMAT correction(s) or verification(s) raised, so the choice was made "
        f"without them; {chosen}: {listed}. A crash or a timeout need not recur: run solve "
        "again, or `orbitwin certify --candidate <id>` a listed candidate from candidates.json, "
        "which flies it afresh."
    )
    return {"warning": warning, "runs": runs}
