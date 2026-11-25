from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .errors import PolicyError
from .models import ArrayLike3, Burn, Candidate, CandidateStage, as_vector3, stage_rank


def prepare_empty_output(directory: Path, operation: str) -> None:
    """Create an output directory, but only when it is new or empty.

    Writing a second run into a directory that already holds one leaves both,
    and the survivor of a name collision is whichever run wrote last. The
    dangerous shape is a submission-named script: an earlier run that reached
    CERTIFIED leaves team_solution.script behind, a later run whose best
    candidate is only analytical writes analytical_candidate.script beside it,
    and the directory now offers a submission the current evidence does not
    support.

    Deleting would be worse -- the old run is evidence, and this project's
    whole argument is that evidence is not disposable. So it refuses and says
    what to do instead.
    """
    if directory.exists() and any(directory.iterdir()):
        raise PolicyError(
            f"{operation} output directory already holds a run: {directory}. "
            "Point at a new directory, or move the old evidence aside first; "
            "mixing two runs in one directory makes neither reproducible."
        )
    directory.mkdir(parents=True, exist_ok=True)


def _optional_float(value: Any) -> float | None:
    """Coerce a serialized number, keeping None as None.

    A candidates.json is read back by `certify`, and a value that arrives as
    the string "1000" instead of the number reaches the first comparison
    against it before anything notices -- three steps past the file that was
    wrong. Coercing at the boundary puts the error where the data is.
    """
    return None if value is None else _finite_float(value)


def _finite_float(value: Any) -> float:
    # json reads NaN and Infinity, and every comparison against NaN is False;
    # bool is an int subclass, so `true` would load as 1.0.
    if isinstance(value, bool):
        raise ValueError(f"expected a number, got {value!r}")
    try:
        number = float(value)
    except OverflowError as exc:  # an int too large for a float
        raise ValueError(f"expected a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"expected a finite number, got {value!r}")
    return number


def _boolean(data: dict[str, Any], key: str, default: bool | None = None) -> bool:
    # bool("false") is True: a hand-edited "false" turned a miss into a hit.
    value = data[key] if default is None else data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"candidate.{key} must be true or false, got {value!r}")
    return value


def _strings(data: dict[str, Any], key: str) -> tuple[str, ...]:
    # A string is iterable too: "none" became four penalties.
    value = data.get(key, [])
    if not isinstance(value, list | tuple):
        raise ValueError(f"candidate.{key} must be a list, got {value!r}")
    return tuple(str(item) for item in value)


def candidate_from_dict(data: dict[str, Any]) -> Candidate:
    if not isinstance(data, dict):
        raise ValueError("A candidate record must be a JSON object")
    burns_raw = data.get("burns", [])
    if not isinstance(burns_raw, list):
        raise ValueError("candidate.burns must be a list")
    def _vector(item: dict[str, Any], field: str, index: int) -> ArrayLike3:
        """Three components, checked here rather than trusted downstream.

        Measured on an unvalidated record: two components gave a plausible
        total_dv_km_s and then raised a bare IndexError from inside to_dict()
        while *writing* the archive, and four were worse and silent -- 9.0139
        km/s from total_dv_km_s against 0.5000 from vector_sum_dv_km_s, two
        delta-v numbers in one record computed over different data.

        models.as_vector3 coerces and counts identically and is what the rest of
        the solver builds these with, so it is what finally produces the value.
        The count is repeated here first only for the message: as_vector3 can
        say a vector was the wrong length, and only this frame can say which
        burn and which field of the archive it was, which is what sends an
        operator to the line in candidates.json.
        """
        raw = item[field]
        # A string is iterable: "123" read as (1.0, 2.0, 3.0) and rendered.
        if not isinstance(raw, list | tuple):
            raise ValueError(f"burn[{index}].{field} must be a list of numbers, got {raw!r}")
        values = tuple(_finite_float(value) for value in raw)
        if len(values) != 3:
            raise ValueError(
                f"burn[{index}].{field} needs exactly 3 components, got {len(values)}"
            )
        return as_vector3(values)

    def _slot(item: dict[str, Any], index: int) -> int:
        # int(1.7) is 1, which renders the burn into a slot the record never named.
        slot = item.get("slot_index", index)
        if type(slot) is not int or slot < 0:
            raise ValueError(
                f"burn[{index}].slot_index must be a non-negative integer, got {slot!r}"
            )
        return slot

    def _frame(item: dict[str, Any], index: int) -> tuple[str, str, str] | None:
        raw = item.get("frame")
        if raw is None:
            return None
        if not isinstance(raw, list | tuple) or len(raw) != 3 or not all(
            isinstance(part, str) for part in raw
        ):
            raise ValueError(
                f"burn[{index}].frame must be three strings "
                f"(CoordinateSystem, Origin, Axes), got {raw!r}"
            )
        return (raw[0], raw[1], raw[2])

    burns = tuple(
        Burn(
            time_s=_finite_float(item["time_s"]),
            dv_inertial_km_s=_vector(item, "dv_inertial_km_s", index),
            dv_script_km_s=_vector(item, "dv_script_km_s", index),
            slot_index=_slot(item, index),
            frame=_frame(item, index),
        )
        for index, item in enumerate(burns_raw)
    )
    for index, burn in enumerate(burns):
        # One impulse in two orthonormal frames, so one length. The limits read
        # the inertial vector and the renderer writes the script one: a 0.3 km/s
        # record rendered as 1.49 km/s and scored as 0.3.
        inertial, script = burn.magnitude_km_s, math.hypot(*burn.dv_script_km_s)
        if abs(inertial - script) > 1.0e-9:
            raise ValueError(
                f"burn[{index}] is {inertial!r} km/s in the inertial frame and {script!r} "
                "km/s in the script frame; one impulse cannot be both"
            )
    candidate = Candidate(
        burns=burns,
        intercept_time_s=_optional_float(data.get("intercept_time_s")),
        minimum_distance_km=_optional_float(data.get("minimum_distance_km")),
        distance_at_completion_km=_optional_float(data.get("distance_at_completion_km")),
        successful=_boolean(data, "successful"),
        analytical_endpoint_error_km=_optional_float(data.get("analytical_endpoint_error_km")),
        branch=str(data["branch"]),
        score=_optional_float(data.get("score")),
        score_components={
            str(key): _finite_float(value)
            for key, value in dict(data.get("score_components", {})).items()
        },
        penalties=_strings(data, "penalties"),
        warnings=_strings(data, "warnings"),
        gmat_run_id=data.get("gmat_run_id"),
        metadata=dict(data.get("metadata", {})),
        candidate_id=str(data.get("candidate_id", "")),
        # A file cannot confer a GMAT tier. Every stage above RENDERABLE means
        # a run happened in this process against this build, and JSON is not
        # evidence of that -- a candidates.json written on the organizer's
        # machine and certified on a laptop would otherwise carry their
        # authority onto our GmatConsole's output. Demonstrated: a record
        # claiming organizer_gmat_verified made certify produce
        # team_solution.script from a public R2026a run with
        # gmat.organizer_compatible unset.
        _stage=min(
            CandidateStage(data.get("stage", CandidateStage.ANALYTICAL.value)),
            CandidateStage.RENDERABLE,
            key=stage_rank,
        ),
        stage_evidence=dict(data.get("stage_evidence", {})),
    )
    # A serialized score was computed from the serialized fields, so restore the
    # basis and the record stays self-consistent. If it was already stale when
    # written, leave the basis unset so it still reads as stale.
    stale = _boolean(data, "score_is_stale", False)
    if candidate.score is not None and not stale:
        candidate.score_basis = candidate.current_score_basis
    claimed = str(data.get("stage", CandidateStage.ANALYTICAL.value))
    if claimed != candidate.stage.value:
        # Kept because it is worth reading, recorded as a claim because that is
        # all a file can be.
        candidate.metadata["stage_claimed_by_file"] = claimed
    return candidate


def write_candidates(path: str | Path, candidates: list[Candidate]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            [candidate.to_dict() for candidate in candidates],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_candidates(path: str | Path) -> list[Candidate]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Candidate file must contain a JSON list")
    return [candidate_from_dict(item) for item in raw]


def write_candidates_csv(path: str | Path, candidates: list[Candidate]) -> None:
    fields = [
        "candidate_id",
        "successful",
        "verified_by_gmat",
        "burn_count",
        "intercept_time_s",
        "minimum_distance_km",
        # 初賽規則 §4 scores the distance at the completion time, and the table
        # a human reads on the day did not carry it.
        "distance_at_completion_km",
        "minimum_pursuer_radius_km",
        "total_dv_km_s",
        "score",
        "score_is_stale",
        "branch",
        "penalty_count",
        "warning_count",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "candidate_id": candidate.candidate_id,
                    "successful": candidate.successful,
                    "verified_by_gmat": candidate.verified_by_gmat,
                    "burn_count": len(candidate.burns),
                    "intercept_time_s": candidate.intercept_time_s,
                    "minimum_distance_km": candidate.minimum_distance_km,
                    "distance_at_completion_km": candidate.distance_at_completion_km,
                    "minimum_pursuer_radius_km": candidate.metadata.get(
                        "minimum_pursuer_radius_km"
                    ),
                    "total_dv_km_s": candidate.total_dv_km_s,
                    "score": candidate.score,
                    "score_is_stale": candidate.score_is_stale,
                    "branch": candidate.branch,
                    "penalty_count": len(candidate.penalties),
                    "warning_count": len(candidate.warnings),
                }
            )
