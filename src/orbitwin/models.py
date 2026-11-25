from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .errors import DynamicsError

ArrayLike3 = tuple[float, float, float]


def as_vector3(values: Iterable[Any]) -> ArrayLike3:
    """The three components of a position, velocity or impulse, as three floats.

    Everything that produces one of these -- a numpy row, a comprehension over
    a Lambert solution, a list parsed out of JSON -- produces something typed
    `tuple[float, ...]` or looser, and every consumer declares ArrayLike3.
    Twenty-odd of the type errors in this package are that one mismatch, and
    each was being fixed at its own call site or not at all.

    The length check is not decoration. A record carrying two components gave a
    plausible total_dv_km_s and then raised a bare IndexError from inside
    to_dict() while *writing* the archive; four were worse and silent, because
    total_dv_km_s summed all four while vector_sum_dv_km_s read the first
    three, putting two delta-v numbers in one record over different data. The
    type says three and now so does the value.
    """
    if isinstance(values, np.ndarray):
        # Shape, not length: NumPy's float() of a 1-element row differs between versions.
        if values.shape != (3,):
            raise ValueError(f"expected 3 components, got an array of shape {values.shape}")
        if type(values) is np.ndarray and values.dtype.type is np.float64:
            x, y, z = values.tolist()  # the floats float() gives, without a generator per element
            return (x, y, z)
    items = tuple(values)
    for item in items:
        if isinstance(item, np.ndarray) and item.ndim > 0:
            raise TypeError(f"expected 3 scalar components, got an array of shape {item.shape}")
    result = tuple(float(item) for item in items)
    if len(result) != 3:
        raise ValueError(f"expected 3 components, got {len(result)}: {result!r}")
    return (result[0], result[1], result[2])


@dataclass(frozen=True, slots=True)
class StateVector:
    """Cartesian inertial state in km and km/s."""

    r_km: ArrayLike3
    v_km_s: ArrayLike3

    @property
    def r(self) -> np.ndarray:
        return np.asarray(self.r_km, dtype=float)

    @property
    def v(self) -> np.ndarray:
        return np.asarray(self.v_km_s, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProbedEpoch:
    """One craft's initial epoch as GMAT reads it, in each scale, as ModJulian days."""

    a1_mjd: float
    utc_mjd: float
    tai_mjd: float
    tt_mjd: float
    tdb_mjd: float


@dataclass(frozen=True, slots=True)
class ProbedCraft:
    name: str
    epoch: ProbedEpoch
    #: EarthMJ2000Eq Cartesian at `epoch`, converted by GMAT from whatever the script gives.
    state: StateVector
    #: spacecraft_binding_sha256: binds the probe to scripts that keep what placed the craft.
    binding_sha256: str


@dataclass(frozen=True, slots=True)
class AlignedStates:
    """Both craft where GMAT's own run of the epoch alignment leaves them: t = 0.

    The early craft's state is GMAT's propagation, never two-body. See
    docs/ASYNC_START_AND_INSERTION.md §3.
    """

    #: EarthMJ2000Eq Cartesian at t = 0, by craft name.
    states: dict[str, StateVector]
    #: Both craft's A1 epoch at t = 0, ModJulian; equal within 1e-6 s.
    a1_mjd: float
    #: Lowest geocentric radius the early craft flies through the alignment, on rows.
    early_minimum_radius_km: float
    #: Closest the moving early craft comes to the waiting late one, on rows.
    minimum_distance_km: float
    #: alignment_binding_sha256: the alignment line, the force models, the
    #: propagators and what places each craft.
    binding_sha256: str
    run: GmatRunResult
    #: The early craft's rows, (t s on the solver clock, x, y, z, vx, vy, vz):
    #: GMAT's unburned head start. Left out of to_dict, as bulk.
    early_track: tuple[tuple[float, ...], ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class StateProbe:
    """What a GmatConsole `Report` before any propagation says each craft starts at.

    See docs/ASYNC_START_AND_INSERTION.md §1.
    """

    craft: tuple[ProbedCraft, ...]
    run: GmatRunResult
    #: After an alignment with Δ > 0; None until load_scenario has run it.
    aligned: AlignedStates | None = None

    def craft_named(self, name: str) -> ProbedCraft | None:
        return next((item for item in self.craft if item.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return _without_tracks(asdict(self))


def _without_tracks(data: dict[str, Any]) -> dict[str, Any]:
    """A probe record less AlignedStates.early_track."""
    aligned = data.get("aligned")
    if isinstance(aligned, dict):
        aligned.pop("early_track", None)
    return data


@dataclass(frozen=True, slots=True)
class SpacecraftSpec:
    name: str
    epoch_text: str
    coordinate_system: str
    display_state_type: str
    fields: dict[str, str]
    initial_state: StateVector
    #: From the state probe; None when parsed in-process without a console.
    probed_epoch: ProbedEpoch | None = None
    #: |Δr| km and |Δv| km/s between the probed state and our own conversion of
    #: the same fields; None when either is missing.
    in_process_mismatch: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(frozen=True, slots=True)
class BurnSlot:
    name: str
    spacecraft: str
    coordinate_system: str
    origin: str
    axes: str
    element1_km_s: float
    element2_km_s: float
    element3_km_s: float
    assignment_lines: tuple[int, int, int]
    #: None when the resource is declared but no Maneuver command uses it.
    #: The template then leaves the maneuvering to the contestant.
    maneuver_line: int | None

    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [self.element1_km_s, self.element2_km_s, self.element3_km_s], dtype=float
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Frame for a maneuver this project adds to a template that declares none.
#: Matches the organizer's own workshop example -- Local/Earth/VNB with mass
#: decrement off, so the impulse is exactly its vector length and §5's ΔVteam is
#: the sum of those lengths.
SYNTHESIZED_BURN_FRAME = ("Local", "Earth", "VNB")


def synthesized_burn_frame(scenario: Scenario) -> tuple[str, str, str]:
    """The frame for a maneuver this project adds, as configured."""
    return scenario.synthesized_burn_frame


def template_maneuvers(scenario: Scenario) -> bool:
    """True when the mission sequence already performs the maneuvers.

    A template that does is filled in: the impulses have slots and the segments
    around them are fixed. One that does not leaves both to be written, whether
    it declares the burn resources or not.
    """
    return any(slot.maneuver_line is not None for slot in scenario.burn_slots)


def burn_may_follow_alignment(scenario: Scenario) -> bool:
    """Whether a burn may fire at t = 0 itself, a Maneuver right after the alignment.

    Only where the maneuvers are ours to write; a template that maneuvers fixes
    its own first segment (docs/ASYNC_START_AND_INSERTION.md §2).
    """
    return scenario.alignment is not None and not template_maneuvers(scenario)


def head_start_lead_s(scenario: Scenario) -> float:
    """How long before t = 0 the pursuer may burn: Scenario.head_start_s on an
    open template, else 0. Burns there have t in [-lead, 0)."""
    return 0.0 if template_maneuvers(scenario) else scenario.head_start_s


def head_start_status(scenario: Scenario) -> str | None:
    """Why the head start is closed on a template policy.head_start_burns asked it of."""
    if scenario.head_start_note is not None or scenario.head_start_s <= 0.0:
        return scenario.head_start_note
    if template_maneuvers(scenario):
        return "the template maneuvers already, so no command may be added before its alignment"
    return None


def uses_head_start(burns: Iterable[Burn]) -> bool:
    """Whether any burn fires before t = 0, in the alignment."""
    return any(burn.time_s < 0.0 for burn in burns)


def head_start_jump(
    scenario: Scenario, times_s: Iterable[float]
) -> tuple[np.ndarray, np.ndarray] | None:
    """(Δr, Δv) a two-body flight under burns at `times_s` takes on at t = 0, the
    alignment's end, so that it leaves there from GMAT's aligned state plus what its
    head-start burns changed; None without head-start burns."""
    if not any(time_s < 0.0 for time_s in times_s):
        return None
    defect = scenario.head_start_defect
    if defect is None:
        raise DynamicsError("a head-start plan needs the alignment's two-body defect")
    return defect.r, defect.v


def pursuer_start(scenario: Scenario, times_s: Iterable[float]) -> tuple[float, StateVector]:
    """Where a two-body flight of the pursuer under burns at `times_s` starts.

    t = 0 at the aligned state, or with a burn before t = 0 the probed state at
    the pursuer's own epoch: GMAT's, never a two-body flight back from t = 0.
    """
    if not any(time_s < 0.0 for time_s in times_s):
        return 0.0, scenario.pursuer.initial_state
    lead = head_start_lead_s(scenario)
    if lead <= 0.0 or scenario.head_start_state is None:
        raise DynamicsError(
            "a burn before t = 0 needs policy.head_start_burns on an open template whose "
            f"pursuer starts first ({head_start_status(scenario) or 'it is off'})"
        )
    return -lead, scenario.head_start_state


def burn_frame(scenario: Scenario, index: int) -> tuple[str, str, str]:
    """The (coordinate system, origin, axes) burn ``index`` is expressed in.

    A declared resource dictates the frame, whether or not the mission
    maneuvers it: declaring one is the organizer saying how they expect the
    impulse written. Only past the last declared resource is the choice ours:
    policy.synthesized_burn_frame, their workshop convention by default.
    Either way the impulse is the same vector -- the frame decides only how
    it is written down.
    """
    if index < len(scenario.burn_slots):
        slot = scenario.burn_slots[index]
        return slot.coordinate_system, slot.origin, slot.axes
    return scenario.synthesized_burn_frame


#: What holds a Propagate line's craft to one epoch; "equal-goals" only while their
#: steps match, "instant" for the zero-length segment that steps nothing.
EpochLock = Literal["propagator", "synchronized", "equal-goals", "instant"]


@dataclass(frozen=True, slots=True)
class PropagateSegment:
    line_index: int
    duration_s: float
    unit: Literal["ElapsedSecs", "ElapsedDays"]
    #: From the propagator calls, not the goals.
    spacecraft_names: tuple[str, ...]
    #: In call order; the first drives a Synchronized line and its report rows.
    propagator_names: tuple[str, ...]
    #: One per goal, which may be fewer than the craft.
    duration_spans: tuple[tuple[int, int], ...]
    epoch_lock: EpochLock | None
    #: The zero-length segment between an epoch alignment and a Maneuver that
    #: directly follows it. It has no line of its own; line_index is that
    #: Maneuver's and duration_spans is empty.
    implicit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: How an alignment's stop condition names the later epoch.
AlignmentForm = Literal["epoch-equality", "epoch-value", "elapsed"]


@dataclass(frozen=True, slots=True)
class EpochAlignment:
    """The first mission command, flying the earlier craft alone to the later one's epoch.

    Not a PropagateSegment: segments feed the burn epochs and the horizon, and
    this ends at t = 0. See docs/ASYNC_START_AND_INSERTION.md §2.
    """

    early: str
    late: str
    propagator: str
    form: AlignmentForm
    #: Late minus early probed A1 epoch, s; 0.0 only when the two are bit-equal.
    offset_s: float
    #: The line exactly as the script holds it, line ending included.
    text: str
    #: None when the template has none and the renderer adds ``text``
    #: (policy.add_epoch_alignment).
    line_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReportFileSpec:
    name: str
    filename: str
    parameters: tuple[str, ...]
    precision: int
    append: bool
    # Defaults are GMAT's. delimiter is the character GMAT writes, not the
    # script text: R2026a writes the first character, so '\t' writes '\'.
    write_report: bool = True
    write_headers: bool = True
    delimiter: str = " "
    fixed_width: bool = True
    column_width: int = 23
    left_justify: bool = True
    zero_fill: bool = False
    #: GMAT's default Current writes only a solver's final pass.
    solver_iterations: str = "Current"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScriptEncoding:
    newline: str
    has_utf8_bom: bool


#: The osculating eccentricity above which SC_Alien's two-body perigee places
#: T_max. J2 alone gives a circular LEO orbit up to 2.95e-3 (6700-8000 km, 0-98
#: deg, measured 2026-11-05), whose osculating "perigee" then lands anywhere in
#: one period; §3's SC_Alien passes perigee and climbs out to a boundary, and
#: R1-R8's are hyperbolic.
CLEARLY_ECCENTRIC = 0.05


@dataclass(frozen=True, slots=True)
class HorizonCheck:
    """SC_Alien's two-body coast from t = 0 against an announced T_max (§3)."""

    #: T_max on the solver clock.
    max_time_s: float
    #: SC_Alien's osculating eccentricity at t = 0.
    ecc: float
    #: Its next two-body perigee on the solver clock; None when climbing at t = 0.
    perigee_s: float | None
    radius_km: float
    radial_rate_km_s: float
    boundary_km: float | None
    #: How far radius_km may sit from boundary_km; None without a boundary.
    tolerance_km: float | None

    @property
    def near_circular(self) -> bool:
        return self.ecc <= CLEARLY_ECCENTRIC

    @property
    def before_perigee(self) -> bool:
        return (
            not self.near_circular and self.perigee_s is not None
            and self.max_time_s < self.perigee_s
        )

    @property
    def before_osculating_perigee(self) -> bool:
        """Before the perigee, which says nothing for a near-circular SC_Alien."""
        return self.perigee_s is not None and self.max_time_s < self.perigee_s

    @property
    def off_boundary(self) -> bool:
        if self.boundary_km is None or self.tolerance_km is None:
            return False
        return (
            abs(self.radius_km - self.boundary_km) > self.tolerance_km
            or self.radial_rate_km_s <= 0.0
        )


@dataclass(frozen=True, slots=True)
class Scenario:
    source_path: str
    source_sha256: str
    pursuer: SpacecraftSpec
    target: SpacecraftSpec
    burn_slots: tuple[BurnSlot, ...]
    propagate_segments: tuple[PropagateSegment, ...]
    report_files: tuple[ReportFileSpec, ...]
    force_model_fields: dict[str, str]
    propagator_fields: dict[str, str]
    mu_km3_s2: float
    #: None when the target's orbit is open.
    target_period_s: float | None
    max_time_s: float
    encoding: ScriptEncoding
    warnings: tuple[str, ...] = ()
    #: Forces the flown ForceModels carry that two-body leaves out; also in warnings.
    two_body_gaps: tuple[str, ...] = ()
    #: The rulebook it was parsed under, policy.rules; scoring follows it.
    rules: str = "preliminary"
    #: Frame of the maneuvers we add, from policy.synthesized_burn_frame.
    synthesized_burn_frame: tuple[str, str, str] = SYNTHESIZED_BURN_FRAME
    #: None when parsed without a console; then the initial states are our own conversion.
    state_probe: StateProbe | None = None
    #: Target minus pursuer probed A1 epoch; None without a probe.
    epoch_offset_s: float | None = None
    #: The epoch alignment, the template's or the one the renderer adds.
    alignment: EpochAlignment | None = None
    #: The equatorial radius GMAT's Earth.Altitude is measured from: its own
    #: 6378.1363 km unless the script sets Earth.EquatorialRadius. None when the
    #: central body is not Earth, so the column is not the pursuer's altitude.
    geodetic_radius_km: float | None = 6378.1363
    #: Solver t = 0 less mission_time_origin's epoch: Δ when that craft flies the
    #: alignment, else 0. T_team and T_max are scored on the solver clock plus this.
    time_origin_offset_s: float = 0.0
    #: The craft policy.mission_time_origin resolves to, whose ElapsedSecs is
    #: T_team; the pursuer when the craft start together.
    time_origin_craft: str | None = None
    #: The announced T_max against SC_Alien's two-body coast; None without one.
    horizon_check: HorizonCheck | None = None
    #: Whether policy.accept_announced_horizon_before_perigee_s took a T_max
    #: horizon_check puts before SC_Alien's perigee.
    horizon_before_perigee_accepted: bool = False
    #: Closest the craft come during the alignment, on GMAT's rows; set only
    #: under policy.closest_approach_window "from_origin", where d_min counts it.
    alignment_minimum_distance_km: float | None = None
    #: The alignment's length when policy.head_start_burns lets the pursuer burn
    #: in it: the pursuer starts first and the alignment has an epoch goal. 0.0
    #: otherwise. Read it through head_start_lead_s, which also wants an open template.
    head_start_s: float = 0.0
    #: The pursuer's state at its own epoch, t = -head_start_s, from the probe.
    head_start_state: StateVector | None = None
    #: GMAT's aligned state at t = 0 less head_start_state flown two-body to it
    #: (Δr km, Δv km/s): what a two-body head-start plan adds at the alignment's end.
    head_start_defect: StateVector | None = None
    #: Why policy.head_start_burns finds no head start here; None when it does or is off.
    head_start_note: str | None = None
    #: A rendered script's pursuer-alone segments before the alignment, each
    #: followed by its Maneuver (COMPETITION_POLICY.md, head-start burns).
    head_start_segments: tuple[PropagateSegment, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if isinstance(data.get("state_probe"), dict):
            _without_tracks(data["state_probe"])
        return data


@dataclass(frozen=True, slots=True)
class Burn:
    time_s: float
    dv_inertial_km_s: ArrayLike3
    dv_script_km_s: ArrayLike3
    slot_index: int = 0
    #: (CoordinateSystem, Origin, Axes) dv_script_km_s is written in; None in
    #: records older than the field. The renderer refuses a mismatch.
    frame: tuple[str, str, str] | None = None

    @property
    def magnitude_km_s(self) -> float:
        return float(np.linalg.norm(np.asarray(self.dv_inertial_km_s, dtype=float)))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["magnitude_km_s"] = self.magnitude_km_s
        return data


class CandidateStage(StrEnum):
    """How much authority stands behind a candidate.

    The authorities rank the organizer validator, then the organizer's GMAT, then
    this project's analytical model (COMPETITION_POLICY.md, Order of authority;
    ASSUMPTIONS_AND_GAPS.md, Release status meaning). Two booleans cannot
    express that ordering -- in particular `verified_by_gmat` is true whether the
    run used the organizer's build or a copy of public R2026a downloaded this
    morning, and those are not the same claim.
    """

    ANALYTICAL = "analytical"
    RENDERABLE = "renderable"
    PUBLIC_GMAT_VERIFIED = "public_gmat_verified"
    ORGANIZER_GMAT_VERIFIED = "organizer_gmat_verified"
    CERTIFIED = "certified"


_STAGE_ORDER: tuple[CandidateStage, ...] = (
    CandidateStage.ANALYTICAL,
    CandidateStage.RENDERABLE,
    CandidateStage.PUBLIC_GMAT_VERIFIED,
    CandidateStage.ORGANIZER_GMAT_VERIFIED,
    CandidateStage.CERTIFIED,
)

#: Artifact name permitted at each stage. Only CERTIFIED may use the submission
#: name, so a filename cannot claim more than the evidence behind it.
STAGE_SCRIPT_NAME: dict[CandidateStage, str] = {
    CandidateStage.ANALYTICAL: "analytical_candidate.script",
    CandidateStage.RENDERABLE: "analytical_candidate.script",
    CandidateStage.PUBLIC_GMAT_VERIFIED: "public_gmat_candidate.script",
    CandidateStage.ORGANIZER_GMAT_VERIFIED: "organizer_gmat_candidate.script",
    CandidateStage.CERTIFIED: "team_solution.script",
}


def stage_rank(stage: CandidateStage) -> int:
    return _STAGE_ORDER.index(stage)


@dataclass(slots=True)
class Candidate:
    burns: tuple[Burn, ...]
    intercept_time_s: float | None
    minimum_distance_km: float | None
    distance_at_completion_km: float | None
    successful: bool
    analytical_endpoint_error_km: float | None
    branch: str
    score: float | None = None
    score_components: dict[str, float] = field(default_factory=dict)
    penalties: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    gmat_run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    candidate_id: str = ""
    #: Read through the `stage` property. Named with a leading underscore so
    #: `candidate.stage = CandidateStage.CERTIFIED` raises instead of silently
    #: awarding an authority level nothing earned.
    _stage: CandidateStage = CandidateStage.ANALYTICAL
    stage_evidence: dict[str, Any] = field(default_factory=dict)
    #: Snapshot of the score inputs taken when `score` was last computed.
    score_basis: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            # slot_index is part of the identity: the same impulse carried by a
            # different ImpulsiveBurn resource renders to a different script and
            # can differ in renderability. Omitting it made the two collide, and
            # select_candidates deduplicates by id.
            payload = repr(
                (
                    tuple(
                        (b.time_s, b.dv_inertial_km_s, b.dv_script_km_s, b.slot_index)
                        for b in self.burns
                    ),
                    self.branch,
                    self.metadata.get("arrival_time_s"),
                )
            ).encode("utf-8")
            self.candidate_id = "C-" + sha256(payload).hexdigest()[:16]

    @property
    def total_dv_km_s(self) -> float:
        """The sum of the impulse magnitudes, derived from the burns.

        This used to be a stored number that nothing reconciled with the burns.
        A candidate could declare 99 km/s while carrying a single 0.5 km/s
        impulse and pass validate_candidate, the renderer and
        assert_final_script_policy without a word, while the official score, the
        Pareto pruning and the tie-break all read the declared figure. The
        dangerous direction is the other one: declaring less than the burns
        carry ranks a candidate too high and hands the organizer a total the
        submission does not match.
        """
        return float(sum(burn.magnitude_km_s for burn in self.burns))

    @property
    def vector_sum_dv_km_s(self) -> float:
        """The magnitude of the inertial vector sum of the impulses.

        Reported, never scored. 初賽規則 §5 eq. 1 writes ΔVteam = Σk ΔVk without
        defining ΔVk, so it can be read as the sum of the impulse magnitudes --
        what total_dv_km_s returns, what fuel actually costs, and the only
        reading consistent with §2 bounding ‖Δvk‖ per impulse -- or as the
        magnitude of the vector sum. Docs Q14 asks the organizer which.

        By the triangle inequality this is never larger than total_dv_km_s, so
        the number we score ourselves on is an upper bound on the number the
        organizer would compute under the other reading: we can under-rate our
        own submission, never over-rate it. For a single-burn candidate the two
        are identical, which is why the distinction only reaches multi-burn
        candidates at all.
        """
        total = [
            sum(burn.dv_inertial_km_s[axis] for burn in self.burns) for axis in range(3)
        ]
        return float(math.sqrt(sum(value * value for value in total)))

    @property
    def current_score_basis(self) -> tuple[Any, ...]:
        """The score inputs as they stand right now."""
        return (
            self.intercept_time_s,
            self.distance_at_completion_km,
            self.minimum_distance_km,
            self.total_dv_km_s,
            len(self.penalties),
        )

    @property
    def score_is_stale(self) -> bool:
        """True when `score` describes inputs the candidate no longer has.

        Mutating an input without recomputing leaves a number that ranking will
        still read. Detecting it costs nothing and turns a silent wrong answer
        into a refusal at the gate.
        """
        return self.score is not None and self.score_basis != self.current_score_basis

    @property
    def stage(self) -> CandidateStage:
        """How much authority stands behind this candidate.

        Read-only on purpose. The previous version was a plain field, so any
        caller could write `candidate.stage = CandidateStage.CERTIFIED` and the
        emission gate would hand it the submission filename. promote() is the
        only way in, and it requires evidence for every tier above renderable.
        """
        return self._stage

    @property
    def verified_by_gmat(self) -> bool:
        """Derived, not stored.

        Keeping a separate boolean alongside the stage meant the two could
        disagree, and both were writable. This one cannot be set at all: it is
        true exactly when a GMAT tier was reached and the candidate is still
        holding a success.
        """
        return stage_rank(self._stage) >= stage_rank(CandidateStage.PUBLIC_GMAT_VERIFIED) and (
            # verify sets scored_miss when the ranking rules score a GMAT-replayed miss.
            self.successful or self.metadata.get("scored_miss") is True
        )

    def promote(self, stage: CandidateStage, **evidence: Any) -> None:
        """Raise the authority level, recording what justified it.

        Promotion is monotonic. A candidate that GMAT later rejects does not go
        backwards -- the evidence was still obtained -- it loses `successful`,
        which is what the gates read alongside the stage.

        Every tier above RENDERABLE requires evidence. A bare
        promote(CERTIFIED) is the shape a mistaken refactor takes, and refusing
        it costs nothing when the real callers already have run ids and hashes
        to hand.
        """
        if stage_rank(stage) >= stage_rank(CandidateStage.PUBLIC_GMAT_VERIFIED) and not evidence:
            raise ValueError(
                f"Promotion to {stage.value} requires evidence; nothing may reach a GMAT "
                "tier without a record of the run that put it there"
            )
        if stage_rank(stage) < stage_rank(self._stage):
            return
        self._stage = stage
        if evidence:
            self.stage_evidence[stage.value] = evidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage.value,
            "stage_evidence": self.stage_evidence,
            "score_is_stale": self.score_is_stale,
            "burns": [b.to_dict() for b in self.burns],
            "intercept_time_s": self.intercept_time_s,
            "minimum_distance_km": self.minimum_distance_km,
            "distance_at_completion_km": self.distance_at_completion_km,
            "total_dv_km_s": self.total_dv_km_s,
            "vector_sum_dv_km_s": self.vector_sum_dv_km_s,
            "successful": self.successful,
            "analytical_endpoint_error_km": self.analytical_endpoint_error_km,
            "branch": self.branch,
            "score": self.score,
            "score_components": self.score_components,
            "penalties": list(self.penalties),
            "warnings": list(self.warnings),
            "verified_by_gmat": self.verified_by_gmat,
            "gmat_run_id": self.gmat_run_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class GmatRunResult:
    run_id: str
    success: bool
    return_code: int
    started_at_utc: str
    duration_s: float
    workdir: str
    script_path: str
    script_sha256: str
    executable_sha256: str
    #: The startup file as supplied -- the organizer's, on their machine. This
    #: is the one that can be compared against anything.
    startup_sha256: str | None
    stdout_path: str
    stderr_path: str
    log_path: str | None
    reports: dict[str, str]
    report_sha256: dict[str, str]
    error: str | None = None
    #: The copy this run generated from the startup file, with its own
    #: OUTPUT_PATH and LOG_FILE so runs cannot read each other's reports.
    #: Different on every run by construction, which is why it is not the
    #: fingerprint and why startup_sha256 above is.
    derived_startup_sha256: str | None = None
    #: Fresh reports a failed run wrote before it stopped. Never evidence of
    #: success; a caller may only use them to decide what to fly next.
    partial_reports: dict[str, str] = field(default_factory=dict)
    #: Startup-file key -> {"path", "sha256"} for each data file it names that
    #: exists: EOP, leap seconds, gravity, ephemerides. They change the physics.
    data_file_sha256: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Each `**** ERROR ****` record GMAT logged, read only from a run that succeeded.
    log_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def spacecraft_binding_sha256(
    name: str,
    assignments: Mapping[str, Mapping[str, str]],
    created: Mapping[str, str],
) -> str:
    """What binds a state probe to a script: what GMAT reads to place this craft.

    The craft's assignments, those of a CoordinateSystem resource it names, and
    those of every object the script does not create (Earth, SolarSystem, ...).
    ``created`` maps each created name to its lower-cased type. The renderer
    edits none of these.
    """
    own = assignments.get(name, {})
    frame = own.get("CoordinateSystem", "").strip().strip("'\"")
    related = [frame] if created.get(frame) == "coordinatesystem" else []
    related += sorted(other for other in assignments if other not in created)
    payload = json.dumps(
        [[item, sorted(assignments.get(item, {}).items())] for item in [name, *related]],
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def alignment_binding_sha256(
    alignment_text: str,
    craft_bindings: Iterable[str],
    force_model_fields: Mapping[str, str],
    propagator_fields: Mapping[str, str],
) -> str:
    """What binds aligned states to a script: everything that moves the early craft."""
    payload = json.dumps(
        [
            alignment_text.rstrip("\r\n"),
            sorted(craft_bindings),
            sorted(force_model_fields.items()),
            sorted(propagator_fields.items()),
        ],
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def clock_stamps(scenario: Scenario) -> dict[str, float | None]:
    """What an archive carries of the clock it was scored on, beside horizon_s."""
    return {
        "time_origin_offset_s": scenario.time_origin_offset_s,
        "alignment_minimum_distance_km": scenario.alignment_minimum_distance_km,
    }


def clock_stamp_mismatch(metadata: Mapping[str, Any], scenario: Scenario) -> str | None:
    """Why a candidate's clock stamps are not this scenario's, or None.

    A candidate without them reads as the synchronous clock, which is refused
    only where the craft start apart and the origin is the early one.
    """
    if "time_origin_offset_s" not in metadata and scenario.time_origin_offset_s:
        return (
            "it carries no clock stamp, so it was solved before the craft's epochs were "
            "read apart"
        )
    offset = float(metadata.get("time_origin_offset_s", 0.0))
    if abs(offset - scenario.time_origin_offset_s) > 1.0e-6:
        return (
            f"T_team {offset!r} s before t = 0 against {scenario.time_origin_offset_s!r} s; "
            "policy.mission_time_origin differs"
        )
    window = metadata.get("alignment_minimum_distance_km")
    if (window is None) != (scenario.alignment_minimum_distance_km is None) or (
        window is not None
        and abs(float(window) - float(scenario.alignment_minimum_distance_km or 0.0)) > 1.0e-9
    ):
        return (
            f"the alignment's closest approach {window!r} km counted against "
            f"{scenario.alignment_minimum_distance_km!r} km; policy.closest_approach_window "
            "differs, or GMAT flew the alignment differently (another build or data files)"
        )
    return None


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
