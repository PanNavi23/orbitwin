from __future__ import annotations

import argparse
import dataclasses
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, assert_never

import yaml

from .errors import ConfigurationError
from .layout import CHECKOUT, PACKAGE
from .rules import (
    DistanceTerm,
    InterceptionRules,
    interception_rule_names,
    interception_rules,
    rule_set_ambiguity,
    rules_name,
)

#: Slack the burn schedule allows when comparing an epoch against the horizon.
#: It exists so a burn placed exactly at the boundary is not rejected by a
#: representation error, and it is permissive: a schedule it lets through can
#: leave a final segment SCHEDULE_TOLERANCE_S shorter than final_padding_s.
#: validate_config accounts for that, which is why the number is shared rather
#: than written twice.
SCHEDULE_TOLERANCE_S = 1.0e-8

#: Never run, whatever policy.extra_resource_types says: they write files
#: (measured on R2026a: EclipseLocator and ContactLocator over a sentinel) or run code.
REFUSED_RESOURCE_TYPES = frozenset(
    {
        "EclipseLocator",
        "ContactLocator",
        "IntrusionLocator",
        "Simulator",
        "BatchEstimator",
        "ExtendedKalmanFilter",
        "Smoother",
        "TrackingFileSet",
        "Transmitter",
        "Receiver",
        "Transponder",
        "Antenna",
        "ErrorModel",
        "AcceptFilter",
        "RejectFilter",
        "ProcessNoiseModel",
        "GmatFunction",
        "MatlabFunction",
        "Python",
        "PythonInterface",
        "FileInterface",
        "ThrustHistoryFile",
    }
)
#: policy.mission_time_origin and policy.closest_approach_window values.
#: "earliest" is the craft with the earlier epoch; the ranking rules take no other.
MISSION_TIME_ORIGINS = ("earliest", "target", "pursuer")
CLOSEST_APPROACH_WINDOWS = ("after_alignment", "from_origin")
#: What fixes the 2026 ranking round's time origin: the organizer's written answer
#: (the rule pack's time_origin_source).
ORGANIZER_TIME_ORIGIN_ANSWER = str(interception_rules("tasa-odc-2026/ranking").time_origin_source)
#: policy.synthesized_burn_frame values, as (CoordinateSystem, Origin, Axes).
SYNTHESIZED_BURN_FRAMES: dict[str, tuple[str, str, str]] = {
    "VNB": ("Local", "Earth", "VNB"),
    "EarthMJ2000Eq": ("EarthMJ2000Eq", "Earth", "MJ2000Eq"),
}



@dataclass(frozen=True, slots=True)
class ScoreConfig:
    kt: float
    ct: float
    kv: float
    cv: float
    time_unit: str = "s"
    delta_v_unit: str = "m/s"
    #: C_r in km, the ranking rules' only; the 2026-11-04 notice gives its unit.
    cr_km: float | None = None


@dataclass(frozen=True, slots=True)
class SearchConfig:
    seed: int = 20260818
    one_burn_samples: int = 512
    one_burn_grid_burns: int = 12
    one_burn_grid_arrivals: int = 24
    max_revolutions: int = 1
    min_tof_s: float = 120.0
    #: Where the densely sampled part of the burn-time window ends, as a
    #: fraction of the horizon. Nothing beyond it is excluded -- the tail is
    #: still gridded, just more coarsely -- so this controls density, not reach.
    dense_window_fraction: float = 0.85
    #: How many of the cheapest swept candidates get a closer look at the
    #: (burn epoch, arrival epoch) pair they came from. Zero disables it.
    local_refine_top: int = 12
    #: Samples per axis in that closer look, across one grid cell either side.
    #: Odd, so the pair that was found is itself resampled.
    local_refine_steps: int = 9
    #: How many independent Sobol draws the sweep is repeated with, unioned.
    #:
    #: Which solutions the sweep finds depends on where its samples land -- the
    #: basins are narrower than a grid cell -- so drawing again and keeping both
    #: sets widens the frontier. The first seed is always the configured one, so
    #: the sample set is a strict superset of the single-seed run. That is a
    #: property of the sampling, not of the answer: the pipeline's prune, event
    #: refinement and local refinement all act on a top-N, and which candidates
    #: are in that N changes when more are found, so a wider draw is not
    #: guaranteed to end better. Measured, it did on every case tried.
    #:
    #: Measure it on the frontier, not on one number. On the 12x24 grid, five
    #: seeds left the cheapest candidate exactly where one seed left it, 543.9
    #: m/s, which reads as "no gain" -- while taking the frontier from 4 points
    #: covering three intercept times to 10 points covering 9208 s to 26031 s.
    #: Scored over 625 (kt, Ct, kv, Cv) combinations that was +2.18 points on
    #: average, +24.97 at most, 0 at worst, winning 357 and losing none.
    #:
    #: It costs one full sweep per extra seed. 1 reproduces the single-seed run
    #: exactly, byte for byte.
    seed_count: int = 1
    #: Worker processes for the one-burn Lambert sweep and its event
    #: refinement. 1 keeps everything in this process; 0 or less means
    #: os.cpu_count() - 1. Work is split into contiguous chunks and collected in
    #: order, so any setting produces the identical candidate list.
    workers: int = 1
    keep_candidates: int = 300
    render_top: int = 20
    endpoint_tolerance_km: float = 0.05
    event_step_s: float = 2.0
    event_refine_top: int = 5
    two_burn_enabled: bool = True
    two_burn_samples: int = 1024
    #: Under rules: ranking, solve also runs ranking_search.py, aimed at GMAT's
    #: target track when it has a console.
    ranking_search: bool = False
    ranking_transfer_starts: int = 8
    ranking_random_starts: int = 4
    ranking_polish_top: int = 12
    ranking_screen_iterations: int = 80
    ranking_polish_iterations: int = 400
    #: Added to the altitude floor inside SLSQP; the hard wall stays the floor.
    ranking_force_model_margin_km: float = 10.0
    #: Also search with no margin, and let GMAT verification choose.
    ranking_search_without_margin: bool = False
    #: Taken off each cap and the total, as headroom for the GMAT correction.
    ranking_correction_margin_km_s: float = 0.0
    #: Burn counts the ranking search seeds by default, 1 up to this: a compute
    #: budget, not a rule, which allows as many as fit 100 s apart.
    ranking_max_burns: int = 13
    #: The pursuer's dynamics in the ranking search: two-body, or a model from
    #: pursuer_model.py ("j2", "j2-spad-drag") that re-polishes the two-body
    #: search's best.
    ranking_pursuer_model: str = "two-body"
    #: Two-body solutions per margin the model polishes. The two-body ones are
    #: offered as well: on R3 the model's alone scored 81.3081 against 81.3918.
    ranking_pursuer_model_top: int = 20


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    mu_km3_s2: float = 398600.4418
    intercept_radius_km: float = 5.0
    dv_limit_km_s: float = 1.5
    dv_safe_limit_km_s: float = 1.495
    #: Sum of impulse magnitudes allowed, km/s; None when the rules set none, as
    #: the preliminary's did. Over it is disqualification, not a ten-point
    #: deduction, and has no tolerance (ranking rules 2026/10/06 §4 note 1(c)).
    total_dv_limit_km_s: float | None = None
    #: Our margin under it: what the search may spend. Set with the limit or not at all.
    total_dv_safe_limit_km_s: float | None = None
    min_burn_separation_s: float = 100.0
    safe_burn_separation_s: float = 101.0
    final_padding_s: float = 20.0
    #: Where the mission sequence stops.
    #:
    #: "interception" ends it one `final_padding_s` after the first entry;
    #: "horizon" runs the whole 4*TA.
    #:
    #: Measured on the 2026-09-05 preliminary, and it cost 22.2266 points.
    #: 初賽規則 §4 defines Tteam = min(tint, 4TA), so a report that runs to the
    #: horizon still contains the interception and the rule reads the same
    #: either way -- that was the reasoning, and it was wrong about the thing
    #: that decides the score. The organizer's validator returned C = 23815.43
    #: (= 4TA) as the completion time while its own D = 2.394 km confirmed the
    #: 5 km entry it had just been handed: it takes the completion time from the
    #: last row of the report, not from the first row inside the sphere. Ending
    #: the sequence at the interception is what turns that into tint.
    #:
    #: "horizon" is still right when the mission is expected to fail, because
    #: then Tteam is 4TA and the distance term is Δr(4TA), worth up to 50 points
    #: and absent from a report that stopped early.
    mission_end_policy: str = "interception"
    #: How far inside the sphere a verified hit's submitted script stops, km:
    #: margin for the organizer's rerun reading other data files. R2026a,
    #: 2026-11-02: EOP/UT1 moved the stop <= 0.11 m and EarthFixed insertion
    #: <= 25.5 m, so 50 m covered every constant-flux case; R4's drag on the
    #: CSSI file at +30 % F10.7/Ap moved it 665 m. 1.5 km is twice that.
    #: Completion comes at least (stop_depth_km - 0.05) / closing speed later
    #: than at 50 m, more off head-on: 0.40 s on R1 at 3.6 km/s. A 13.2 m/s
    #: aligned crossing stopped 121.5 s after entry.
    #: `orbitwin inspect` names each drag model's weather source.
    stop_depth_km: float = 1.5
    #: How long the sequence runs past the first entry under "interception",
    #: where GMAT has not measured the stop: an unverified render, or a hit
    #: inside before its last burn. A verified hit stops stop_depth_km inside.
    #:
    #: Not `final_padding_s`: that is the gap the last *maneuver* needs before
    #: the horizon, a different question. A validator that reads the completion
    #: time off the last row reads t_int plus this, so every second here is a
    #: second added to Tteam. On the 2026-09-05 coefficients 20 s costs 0.2025
    #: points and 1 s costs 0.0098.
    #:
    #: It cannot be zero: a goal of 0 s is the zero-length segment that hangs
    #: R2026a. Nor can it be one reporting step. GMAT phases its rows from the
    #: last Maneuver, not from our interception, so the row that first falls
    #: inside the sphere sits up to one step after the epoch this tail is
    #: measured from. Observed on the 6-burn 98.32 candidate: analytical t_int
    #: 5676.2438 s, GMAT's entry row 5676.7897 s, and a 1.0 s tail ended at
    #: 5677.2438 -- 0.45 s past that row, between two samples, so the last row
    #: was 7.57 km outside. Two seconds clears a whole 1 s step either way.
    #:
    #: Scale it with the propagator: it has to exceed the report interval, which
    #: is `Propagator.MaxStep` on a two-body arc. Check the result with
    #: scripts/check_report_rows.py rather than trusting the arithmetic.
    interception_tail_s: float = 2.0
    #: Shortest Propagate segment the official template may carry. GMAT R2026a
    #: does not reject ElapsedSecs = 0 -- it hangs, which would stall the
    #: organizer's validator rather than fail cleanly. Measured on that build:
    #: 0.001 s completes normally. This is an observed floor, not a rule.
    min_propagate_duration_s: float = 1.0e-3
    #: Whether a candidate outside the production margins may be rendered and
    #: submitted. Off by default: official burns keep the
    #: safety margin rather than sit on the 1.5 km/s and 100 s boundaries. The
    #: formal frontier is still searched and archived either way; this only
    #: decides whether one of its candidates can leave the archive.
    allow_formal_frontier_submission: bool = False
    #: Whether certify may certify a hit whose passage never reached stop_depth_km.
    #: Off: the drift stop_depth_km covers could make it a miss on the organizer's
    #: machine. solve still names one when no deeper hit is verified, and says so.
    allow_shallow_stop_submission: bool = False
    #: How far the target's eccentricity may drift from circular before the
    #: parser says so. The briefing deck states the preliminary target is
    #: circular (`在初賽中,Spacecraft A 將以一圓軌道繞行地球`), so a target that
    #: is not is a sign the scenario is not the one that was described. It is a
    #: warning and not a refusal: TA = 2*pi*sqrt(a^3/mu) is just as well defined
    #: for an ellipse, so nothing is ambiguous, and refusing on a tolerance we
    #: guessed would mean producing nothing on competition day if the official
    #: state carries a little numerical eccentricity.
    target_circular_tolerance: float = 1.0e-6
    #: T_max as announced on the day (ranking rules 2026/10/06 §3). None means the
    #: preliminary's 4*T_A, which a target on an open orbit does not have.
    horizon_s: float | None = None
    #: Which rulebook: "preliminary" (2026/05/20), "ranking" (2026/10/06), "ranking-onsite"
    #: (2026/11/08) or another rule pack's id (docs/RULE_PACKS.md).
    #: Under the ranking rules a missing horizon_s is refused, not replaced by 4*T_A.
    rules: str = "preliminary"
    #: Our team number. It replaces each TeamXX placeholder in the official
    #: ReportFiles' Filenames, which the organizer requires
    #: (docs/LESSONS_2026-09-05.md §7). Under "ranking" certify refuses without
    #: it, and unless at least one official report then carries it.
    team_number: int | None = None
    #: The official report's Filename exactly as the template writes it, set only
    #: after the organizer answers in writing to keep a name with no team
    #: placeholder. Certify refuses any other value, and records it.
    report_filename_as_written: str | None = None
    #: The frame of a maneuver this project adds to a template that declares no
    #: burns: "VNB" (Local/Earth/VNB, the organizer's workshop convention) or
    #: "EarthMJ2000Eq". GMAT applies VNB on its own state, which under a rich
    #: force model is not the two-body state the impulse was planned on: on the
    #: R4 rehearsal the flown impulse was 0.431 m/s off the plan in VNB and
    #: 0.020 m/s inertially. Off by default until the organizer answers
    #: SCORING_STRATEGY_AND_OPEN_QUESTIONS.md Q7 in writing.
    synthesized_burn_frame: str = "VNB"
    #: Resource types to accept beyond gmat_parser.ALLOWED_RESOURCE_TYPES, set by
    #: a person on the day after checking each writes and runs nothing. Never one
    #: of REFUSED_RESOURCE_TYPES; commented out of every private copy.
    extra_resource_types: tuple[str, ...] = ()
    #: The body mu_km3_s2 and central_body_radius_km describe. The script has
    #: to agree: those two numbers are the whole two-body model, so a force
    #: model about a different body makes every derived quantity -- the period,
    #: the horizon, every Lambert arc, the surface the trajectory is checked
    #: against -- quietly wrong.
    central_body_name: str = "Earth"
    #: Equatorial radius of the central body, used to notice a candidate whose
    #: trajectory passes through it. GMAT R2026a's Earth value.
    central_body_radius_km: float = 6378.1363
    #: Whether a candidate that reaches the interception by flying through the
    #: central body may be submitted. Off by default. The preliminary rules say
    #: nothing about it and point-mass gravity integrates straight through, so
    #: such a candidate is legal on the page and impossible in fact; §8 leaves
    #: the final reading to the organizer and §6 ends in a five-minute design
    #: presentation. It stays in the archive either way -- this only decides
    #: whether it can leave. The ranking rules disqualify it (§4 note 1(b)),
    #: so a ranking config cannot set it.
    allow_body_intersecting_submission: bool = False
    #: Lowest altitude GMAT may fly the pursuer for verify, certify, solve and
    #: rank to accept a candidate, measured over every row GMAT flies it in:
    #: an epoch alignment that flies it, and on past an interception to the end
    #: of the report. None keeps the line at the central
    #: body, which is the rules' and the user's absolute line. The ranking
    #: profiles set 100 km, the search's own floor, as our margin against drag
    #: differences between GMAT builds: GMAT flew R4's margin-0 winners to
    #: 95.4 km under drag where the two-body plan held 100.
    minimum_flown_altitude_km: float | None = None
    allow_zero_vector_unused_slots: bool = True
    require_synchronized_propagation: bool = True
    #: Whether the renderer may add an epoch alignment to a template whose
    #: craft start on different epochs and that has none. Off by default; the
    #: ranking profiles set it since the 2026-11-03 sign-off (COMPETITION_POLICY.md).
    add_epoch_alignment: bool = False
    #: Whether an open template whose pursuer starts first may burn in the epoch
    #: alignment, before the target's epoch. Off by default; the ranking profiles
    #: set it since the 2026-11-08 sign-off (COMPETITION_POLICY.md).
    head_start_burns: bool = False
    #: How much a head-start candidate must beat the best certifiable one without
    #: head-start burns by for rank to pick it: the organizer has not said how its
    #: validator reads a burn in the alignment's rows (ASYNC_START_AND_INSERTION.md Q5).
    head_start_min_gain_points: float = 0.0
    #: Whose epoch is t = 0 for T_team, T_max and Ct when the craft start apart:
    #: "earliest", "target" or "pursuer". Required then. The ranking rules take
    #: only "earliest" (ORGANIZER_TIME_ORIGIN_ANSWER).
    mission_time_origin: str | None = None
    #: Where d_min's window starts when the craft start apart:
    #: "after_alignment" or "from_origin". Required then; still asked of the organizer.
    closest_approach_window: str | None = None
    #: The mission boundary of rules §3 as a geocentric radius, if announced.
    #: Read by the parser's T_max check and the time-origin evidence; warns only.
    mission_boundary_km: float | None = None
    #: An announced T_max taken although it falls before SC_Alien's perigee, which
    #: rules §3 rule out. It must equal horizon_s, so it covers that one value. Set
    #: only by ranking_day_config.py's flag of the same name, after the organizer
    #: confirmed the value in writing (COMPETITION_POLICY.md, Order of authority); recorded wherever the
    #: clock is.
    accept_announced_horizon_before_perigee_s: float | None = None
    #: An announced T_max taken although SC_Alien is off mission_boundary_km there
    #: (two-body), as a T_max announced rounded to the minute is. It must equal
    #: horizon_s. Set only by ranking_day_config.py --tmax-as-announced, which refuses
    #: an off-boundary T_max without it, so that a typo is caught.
    tmax_as_announced_s: float | None = None


@dataclass(frozen=True, slots=True)
class GmatConfig:
    verify_top: int = 0
    timeout_s: float = 300.0
    reruns: int = 3
    diagnostic_report_name: str = "orbitwin_diagnostic.txt"
    compare_report_bytes: bool = True
    #: Set only once the organizer has confirmed the build, startup file and
    #: data files. Left false, a successful run yields PUBLIC_GMAT_VERIFIED,
    #: which is not final authority (COMPETITION_POLICY.md, Order of authority).
    organizer_compatible: bool = False
    #: SHA-256 of the organizer GmatConsole. Required whenever
    #: organizer_compatible is set: that flag is what lets a candidate reach
    #: CERTIFIED and be named team_solution.script, and a bare boolean in a
    #: YAML file is not evidence that the run happened on the organizer's
    #: build. `orbitwin baseline` writes the hash to baseline.json under
    #: run.executable_sha256, and docs/RUNBOOK.md already runs that baseline in
    #: step 3, so the fingerprint costs nothing to obtain.
    organizer_executable_sha256: str | None = None
    #: SHA-256 of the organizer's startup file, optionally. It selects the
    #: gravity, leap-second and EOP data, so it changes the physics as surely
    #: as the binary does. Left unset, the tier rests on the executable alone.
    organizer_startup_sha256: str | None = None
    #: Concurrent GmatConsole runs while correcting and verifying. Each run has an
    #: isolated working and output directory, and a run is spent waiting on a
    #: subprocess, so threads are the right tool. 1 keeps it sequential.
    workers: int = 1
    #: Hits solve corrects on GMAT's residuals (gmat_shoot.py) before verifying
    #: them; 0 corrects none. Each takes 0 to 9 GmatConsole runs.
    correct_top: int = 0
    correct_rounds: int = 4
    correct_aim_km: float = 0.5
    #: Analytic misses solve verifies when a miss scores (ranking rules), the
    #: best-scored above the zero burn; a budget beside the hits', not from it.
    verify_misses: int = 0
    #: When GmatConsole refuses the submitted bytes for their OpenFrames or
    #: GroundTrack plots alone, run certify's and baseline's exact runs on their
    #: proven copy less them, and package that copy's report. Off, the refusal
    #: stands. The ranking profiles set it (signed off 2026-11-06).
    console_copy: bool = False


@dataclass(frozen=True, slots=True)
class OrbitWinConfig:
    search: SearchConfig = field(default_factory=SearchConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    gmat: GmatConfig = field(default_factory=GmatConfig)
    score: ScoreConfig | None = None
    pursuer_name: str | None = None
    target_name: str | None = None


_MERGE_TAG = "tag:yaml.org,2002:merge"


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader without the YAML 1.1 readings nobody means in a config."""

    def construct_document(self, node: yaml.Node) -> Any:
        # Checked before construction: flatten_mapping rewrites merged nodes in place.
        _check_keys(self, node, set())
        return super().construct_document(node)


def _check_keys(loader: _StrictLoader, node: yaml.Node, visited: set[int]) -> None:
    if id(node) in visited:  # an alias, or a recursive anchor
        return
    visited.add(id(node))
    if isinstance(node, yaml.SequenceNode):
        for item in node.value:
            _check_keys(loader, item, visited)
    elif isinstance(node, yaml.MappingNode):
        _unique_mapping(loader, node)
        for key_node, value_node in node.value:
            _check_keys(loader, key_node, visited)
            _check_keys(loader, value_node, visited)


def _unique_mapping(loader: _StrictLoader, node: yaml.MappingNode) -> None:
    # PyYAML keeps the last of two equal keys, so a second
    # allow_body_intersecting_submission silently overrode the first.
    # A << merge key may be overridden by an explicit one; that is YAML, not a duplicate.
    seen: set[Any] = set()
    merged = False
    for key_node, value_node in node.value:
        line = key_node.start_mark.line + 1
        if key_node.tag == _MERGE_TAG:
            # PyYAML merges both, so the later silently wins where they overlap.
            if merged:
                raise ConfigurationError(f"Merge key '<<' appears twice, again on line {line}")
            merged = True
            _disjoint_sources(loader, value_node, line)
            continue
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ConfigurationError(f"Key {key!r} on line {line} is not a name")
        if key in seen:
            raise ConfigurationError(f"Key {key!r} appears twice, again on line {line}")
        seen.add(key)


def _disjoint_sources(loader: _StrictLoader, value_node: yaml.Node, line: int) -> None:
    # In `<<: [a, b]` the first source silently wins, the same overlap as two << keys.
    if not isinstance(value_node, yaml.SequenceNode):
        return
    merged: set[str] = set()
    for source in value_node.value:
        if isinstance(source, yaml.MappingNode):
            keys = _merged_keys(loader, source, set())
            overlap = merged & keys
            if overlap:
                raise ConfigurationError(
                    f"Key {sorted(overlap)[0]!r} comes from two merge sources on line {line}"
                )
            merged |= keys


def _merged_keys(loader: _StrictLoader, node: yaml.MappingNode, active: set[int]) -> set[str]:
    """The names a mapping contributes when merged, its own merges included."""
    if id(node) in active:  # a recursive anchor adds nothing new
        return set()
    active.add(id(node))
    keys: set[str] = set()
    for key_node, value_node in node.value:
        if key_node.tag != _MERGE_TAG:
            key = loader.construct_object(key_node, deep=True)
            if isinstance(key, str):  # anything else is refused as not a name
                keys.add(key)
            continue
        sources = value_node.value if isinstance(value_node, yaml.SequenceNode) else [value_node]
        for source in sources:
            if isinstance(source, yaml.MappingNode):
                keys |= _merged_keys(loader, source, active)
    active.discard(id(node))
    return keys


_DECIMAL_INT = re.compile(r"[-+]?(?:0|[1-9][0-9_]*)")


def _decimal_int(loader: _StrictLoader, node: yaml.ScalarNode) -> int:
    # YAML 1.1 reads 0100 as octal 64 and 1:30 as 90.
    text = str(loader.construct_scalar(node))
    if not _DECIMAL_INT.fullmatch(text):
        raise ConfigurationError(
            f"{text!r} on line {node.start_mark.line + 1} is not a decimal integer; "
            "YAML reads it in another base"
        )
    return int(text.replace("_", ""))


def _decimal_float(loader: _StrictLoader, node: yaml.ScalarNode) -> float:
    # YAML 1.1 also reads 1:30.0 as 90.0.
    text = str(loader.construct_scalar(node))
    if ":" in text:
        raise ConfigurationError(
            f"{text!r} on line {node.start_mark.line + 1} is not a decimal number; "
            "YAML reads it in base 60"
        )
    return float(loader.construct_yaml_float(node))


_StrictLoader.add_constructor("tag:yaml.org,2002:int", _decimal_int)
_StrictLoader.add_constructor("tag:yaml.org,2002:float", _decimal_float)


def _construct(cls: type[Any], raw: dict[str, Any] | None) -> Any:
    # `raw or {}` turned `policy: []` into every default without a word.
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"{cls.__name__} must be a mapping, got {type(raw).__name__} {raw!r}"
        )
    known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise ConfigurationError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise ConfigurationError(f"Invalid {cls.__name__}: {exc}") from exc


#: Units a score block must state; ScoreConfig's defaults are for code, never for a file.
SCORE_UNIT_KEYS = ("time_unit", "delta_v_unit")


def _require_score_units(score_raw: dict[str, Any]) -> None:
    # A Cv announced in km/s and read as m/s flattens the delta-v term without an error.
    missing = [key for key in SCORE_UNIT_KEYS if key not in score_raw]
    if missing:
        raise ConfigurationError(
            f"The score block does not state {' and '.join(f'score.{k}' for k in missing)}. "
            "Write the units the organizer announced with kt, Ct, kv and Cv "
            "(time_unit: s; delta_v_unit: m/s, or km/s under the preliminary rules); "
            "they are never guessed. "
            "scripts/ranking_day_config.py writes them for you."
        )


#: The ranking profile beside the package in a checkout; an installed package carries a copy.
_CHECKOUT_PROFILE = None if CHECKOUT is None else CHECKOUT / "config" / "ranking.yaml"
_PACKAGED_PROFILE = PACKAGE / "profiles" / "ranking.yaml"


def ranking_profile() -> Path:
    """config/ranking.yaml in a checkout, else the copy the package carries (the same bytes)."""
    if _CHECKOUT_PROFILE is not None and _CHECKOUT_PROFILE.is_file():
        return _CHECKOUT_PROFILE
    return _PACKAGED_PROFILE


def load_config(path: str | Path | None) -> OrbitWinConfig:
    if path is None:
        config = OrbitWinConfig()
        validate_config(config)
        return config
    try:
        raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        # The CLI reports ConfigurationError; a YAMLError escaped it as a traceback.
        raise ConfigurationError(f"{path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError("Top-level YAML must be a mapping")
    known = {"search", "policy", "gmat", "score", "pursuer_name", "target_name"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigurationError(f"Unknown top-level keys: {sorted(unknown)}")
    score_raw = raw.get("score")
    if isinstance(score_raw, dict):
        _require_score_units(score_raw)
    score = _construct(ScoreConfig, score_raw) if score_raw is not None else None
    config = OrbitWinConfig(
        search=_construct(SearchConfig, raw.get("search")),
        policy=_construct(PolicyConfig, raw.get("policy")),
        gmat=_construct(GmatConfig, raw.get("gmat")),
        score=score,
        pursuer_name=raw.get("pursuer_name"),
        target_name=raw.get("target_name"),
    )
    validate_config(config)
    return config


#: The rulebook every day-path tool runs unless told otherwise on its command line.
DAY_RULES = "ranking"


def add_rules_argument(parser: argparse.ArgumentParser) -> None:
    """--rules, which a day-path tool needs for any config not under the ranking rules."""
    parser.add_argument(
        "--rules", type=rules_name,
        help=f"the rulebook the config is under; required unless it is {DAY_RULES}, "
        "as the day's config from scripts/ranking_day_config.py is",
    )


def rulebook_refusal(
    config: OrbitWinConfig, config_path: str | Path | None, chosen: str | None
) -> str | None:
    """Why a day-path tool must not run on ``config``, or None when it may.

    policy.rules defaults to the preliminary, so a missing --config, or a config
    with no rules key, ran the preliminary rules with only a note on stderr.
    Any rulebook but the ranking one now has to be named with --rules.
    """
    rules = config.policy.rules
    source = (
        f"no --config was given, so this would run the {rules} rules" if config_path is None
        else f"{config_path} is under the {rules} rules (policy.rules defaults to preliminary)"
    )
    if chosen is None and interception_rules(rules) is interception_rules(DAY_RULES):
        return None
    if chosen is None:
        return (
            f"{source}. The ranking day needs the day's config, written by "
            "scripts/ranking_day_config.py (docs/RUNBOOK.md step 2); to run the "
            f"{rules} rules on purpose, pass --rules {rules}"
        )
    if interception_rules(chosen) is not interception_rules(rules):
        return f"--rules {chosen}, but {source}; give a config under the rules you mean"
    return None


def finite_number(value: object) -> float | None:
    """``value`` as a float if it is a real, finite number; else None.

    bool is an int, and an int past a double's range does not convert.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _finite(value: object, label: str, *, positive: bool = True, allow_zero: bool = False) -> None:
    """Reject a configured number that is not a real, usable quantity.

    The relational checks below cannot do this on their own. Every comparison
    against NaN is False, so `mu_km3_s2 <= 0` waves NaN straight through, and
    then TA = 2*pi*sqrt(a^3/mu) is NaN, Tmax is NaN, and every later
    `t <= max_time_s` is False -- constraints pass, candidates are accepted, and
    the run reports a horizon of nan without one complaint. Measured on the
    workshop script before the parser checked: mu = nan and mu = inf both parsed
    and both gave Tmax = nan.
    """
    # Type before value. YAML gives back whatever was written, so `mu_km3_s2:
    # "398600.4418"` arrives as a str; math.isfinite then raises TypeError from
    # inside the validator, which names neither the field nor the fix. bool is
    # an int subclass, so `kt: true` would otherwise pass as 1.0.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(
            f"{label} must be a number, got {type(value).__name__} {value!r}"
        )
    if finite_number(value) is None:
        raise ConfigurationError(f"{label} must be a finite number, got {str(value)[:40]}")
    if not positive:
        return
    if allow_zero:
        if value < 0.0:
            raise ConfigurationError(f"{label} cannot be negative")
    elif value <= 0.0:
        raise ConfigurationError(f"{label} must be positive")


def _finite_declared_floats(section: object, label: str) -> None:
    """Check every field declared `float`, whatever it is called.

    The per-field checks below say what each number means -- positive, or
    allowed to be zero, or bounded. This says only that it is a number and a
    finite one, which is true of all of them and is the part that goes missing.
    It went missing: written as a hand-kept list it covered nineteen fields and
    missed two. A list that has to be extended whenever a field is added will be
    short again. (The two were search.second_burn_fraction_min and _max, both
    since removed as settings nothing read.)
    """
    for entry in dataclasses.fields(section):  # type: ignore[arg-type]
        if entry.type not in {"float", "float | None"}:
            continue
        value = getattr(section, entry.name)
        if value is None and entry.type == "float | None":
            continue
        _finite(value, f"{label}.{entry.name}", positive=False)


#: Declared types YAML can fill with another one that still reads as valid.
_EXACT_TYPES: dict[str, tuple[type, ...]] = {
    "bool": (bool,),
    "int": (int,),
    "int | None": (int, type(None)),
    "str": (str,),
    "str | None": (str, type(None)),
}


def _exact_declared_types(section: object, label: str) -> None:
    """Check every bool, int and str field holds exactly its declared type.

    A quoted `allow_body_intersecting_submission: 'false'` is a non-empty str,
    so it read as True and switched the Earth-dive gate off. bool is an int
    subclass, so the exact type is compared rather than isinstance.
    """
    prefix = f"{label}." if label else ""
    for entry in dataclasses.fields(section):  # type: ignore[arg-type]
        allowed = _EXACT_TYPES.get(str(entry.type))
        value = getattr(section, entry.name)
        if allowed is not None and type(value) not in allowed:
            raise ConfigurationError(
                f"{prefix}{entry.name} must be {entry.type}, "
                f"got {type(value).__name__} {value!r}"
            )


def _over_limit_verb(rules: InterceptionRules) -> str:
    return "disqualify" if rules.over_impulse_limit == "disqualify" else "penalise"


def validate_config(config: OrbitWinConfig) -> None:
    p = config.policy
    s = config.search
    _exact_declared_types(config, "")
    _exact_declared_types(p, "policy")
    _exact_declared_types(s, "search")
    _exact_declared_types(config.gmat, "gmat")
    if config.score:
        _exact_declared_types(config.score, "score")
    # Finiteness first: a NaN slips through every relational check below.
    _finite_declared_floats(p, "policy")
    _finite_declared_floats(s, "search")
    _finite_declared_floats(config.gmat, "gmat")
    if config.score:
        _finite_declared_floats(config.score, "score")
    _finite(p.mu_km3_s2, "policy.mu_km3_s2")
    _finite(p.intercept_radius_km, "policy.intercept_radius_km")
    _finite(p.dv_limit_km_s, "policy.dv_limit_km_s")
    _finite(p.dv_safe_limit_km_s, "policy.dv_safe_limit_km_s")
    _finite(p.min_burn_separation_s, "policy.min_burn_separation_s")
    _finite(p.safe_burn_separation_s, "policy.safe_burn_separation_s")
    _finite(p.min_propagate_duration_s, "policy.min_propagate_duration_s")
    _finite(p.central_body_radius_km, "policy.central_body_radius_km")
    _finite(p.final_padding_s, "policy.final_padding_s", allow_zero=True)
    _finite(p.target_circular_tolerance, "policy.target_circular_tolerance", allow_zero=True)
    _finite(p.head_start_min_gain_points, "policy.head_start_min_gain_points", allow_zero=True)
    if p.minimum_flown_altitude_km is not None:
        _finite(
            p.minimum_flown_altitude_km, "policy.minimum_flown_altitude_km", allow_zero=True
        )
    if p.horizon_s is not None:
        _finite(p.horizon_s, "policy.horizon_s")
        if p.horizon_s <= p.final_padding_s:
            raise ConfigurationError(
                f"policy.horizon_s {p.horizon_s!r} s leaves no mission after the "
                f"{p.final_padding_s!r} s final padding"
            )
    _finite(s.min_tof_s, "search.min_tof_s", allow_zero=True)
    _finite(s.endpoint_tolerance_km, "search.endpoint_tolerance_km")
    _finite(s.event_step_s, "search.event_step_s")
    _finite(s.dense_window_fraction, "search.dense_window_fraction")
    if s.seed_count < 1:
        raise ConfigurationError("search.seed_count must be at least 1")
    _finite(config.gmat.timeout_s, "gmat.timeout_s")
    if config.score:
        # Finite only. The four coefficients are announced by the organizer and
        # guessing a sign convention on their behalf is exactly what
        # COMPETITION_POLICY.md forbids.
        for name in ("kt", "ct", "kv", "cv"):
            # Not float(...): coercing here made the check pass for a value that
            # stayed a string, and scoring then died with "can't multiply
            # sequence by non-int of type 'float'" in the middle of a search.
            _finite(getattr(config.score, name), f"score.{name}", positive=False)
    if not 0.0 < p.stop_depth_km <= 0.5 * p.intercept_radius_km:
        raise ConfigurationError(
            f"policy.stop_depth_km is {p.stop_depth_km!r} km; it must be in (0, "
            f"{0.5 * p.intercept_radius_km:g}], half policy.intercept_radius_km, so a "
            "smaller radius needs it set too"
        )
    if p.interception_tail_s < p.min_propagate_duration_s:
        raise ConfigurationError(
            "policy.interception_tail_s cannot be below min_propagate_duration_s: the segment "
            "after the last maneuver would be shorter than the floor that keeps GMAT from hanging"
        )
    if p.report_filename_as_written is not None and (
        not isinstance(p.report_filename_as_written, str) or not p.report_filename_as_written
    ):
        raise ConfigurationError(
            "policy.report_filename_as_written must be the official report's Filename, "
            f"got {p.report_filename_as_written!r}"
        )
    if p.team_number is not None and p.team_number < 1:
        raise ConfigurationError(
            f"policy.team_number must be a positive integer, got {p.team_number!r}"
        )
    shared = rule_set_ambiguity(p.rules) if isinstance(p.rules, str) else None
    if shared is not None:
        raise ConfigurationError(f"policy.rules: {shared}")
    if not isinstance(p.rules, str) or p.rules not in interception_rule_names():
        raise ConfigurationError(
            f"policy.rules must be one of {interception_rule_names()}, got {p.rules!r}; "
            "the duel is played with `orbitwin duel`, not a mission config"
        )
    if not isinstance(p.synthesized_burn_frame, str) or (
        p.synthesized_burn_frame not in SYNTHESIZED_BURN_FRAMES
    ):
        raise ConfigurationError(
            f"policy.synthesized_burn_frame must be one of {sorted(SYNTHESIZED_BURN_FRAMES)}, "
            f"got {p.synthesized_burn_frame!r}"
        )
    extra = p.extra_resource_types
    if not isinstance(extra, list | tuple) or not all(
        isinstance(item, str) and re.fullmatch(r"[A-Za-z_]\w*", item) for item in extra
    ):
        raise ConfigurationError(
            f"policy.extra_resource_types must be a list of GMAT type names, got {extra!r}"
        )
    refused = sorted(set(extra) & REFUSED_RESOURCE_TYPES)
    if refused:
        raise ConfigurationError(
            f"policy.extra_resource_types names {refused}, which write files outside the run "
            "or run code and are never accepted"
        )
    if p.mission_time_origin not in (*MISSION_TIME_ORIGINS, None):
        raise ConfigurationError(
            f"policy.mission_time_origin must be one of {list(MISSION_TIME_ORIGINS)} or "
            f"unset, got {p.mission_time_origin!r}"
        )
    if (interception_rules(p.rules).time_origin == "earliest"
            and p.mission_time_origin not in ("earliest", None)):
        raise ConfigurationError(
            f"policy.mission_time_origin is {p.mission_time_origin!r}, but under the "
            f"{interception_rules(p.rules).name} rules it is 'earliest', by "
            f"{interception_rules(p.rules).time_origin_source}"
        )
    accepted = p.accept_announced_horizon_before_perigee_s
    if accepted is not None and accepted != p.horizon_s:
        raise ConfigurationError(
            f"policy.accept_announced_horizon_before_perigee_s is {accepted!r} and "
            f"policy.horizon_s {p.horizon_s!r}: it accepts only the T_max it names. Ask the "
            "organizer about the new value, and write the config again"
        )
    as_announced = p.tmax_as_announced_s
    if as_announced is not None and as_announced != p.horizon_s:
        raise ConfigurationError(
            f"policy.tmax_as_announced_s is {as_announced!r} and policy.horizon_s "
            f"{p.horizon_s!r}: it takes only the T_max it names. Write the config again with "
            "scripts/ranking_day_config.py"
        )
    if p.closest_approach_window not in (*CLOSEST_APPROACH_WINDOWS, None):
        raise ConfigurationError(
            f"policy.closest_approach_window must be one of {list(CLOSEST_APPROACH_WINDOWS)} "
            f"or unset, got {p.closest_approach_window!r}"
        )
    if p.mission_boundary_km is not None:
        _finite(p.mission_boundary_km, "policy.mission_boundary_km")
        if p.mission_boundary_km <= p.central_body_radius_km:
            raise ConfigurationError(
                f"policy.mission_boundary_km {p.mission_boundary_km!r} km is inside the "
                f"{p.central_body_radius_km!r} km central body"
            )
    if p.mission_end_policy not in {"interception", "horizon"}:
        raise ConfigurationError(
            "policy.mission_end_policy must be 'interception' or 'horizon', got "
            f"{p.mission_end_policy!r}"
        )
    if not 0 < p.dv_safe_limit_km_s <= p.dv_limit_km_s:
        raise ConfigurationError("dv_safe_limit_km_s must be in (0, dv_limit_km_s]")
    if (p.total_dv_limit_km_s is None) != (p.total_dv_safe_limit_km_s is None):
        raise ConfigurationError(
            "policy.total_dv_limit_km_s and policy.total_dv_safe_limit_km_s are set together"
        )
    if p.total_dv_limit_km_s is not None and p.total_dv_safe_limit_km_s is not None:
        _finite(p.total_dv_limit_km_s, "policy.total_dv_limit_km_s")
        _finite(p.total_dv_safe_limit_km_s, "policy.total_dv_safe_limit_km_s")
        if p.total_dv_safe_limit_km_s > p.total_dv_limit_km_s:
            raise ConfigurationError(
                "total_dv_safe_limit_km_s must be in (0, total_dv_limit_km_s]"
            )
    rules = interception_rules(p.rules)
    # A config may stay inside its rulebook's limits, never outside them. The
    # defaults are the preliminary's 1.5 km/s and no total.
    if p.dv_limit_km_s > rules.impulse_limit_km_s + rules.impulse_epsilon_km_s:
        raise ConfigurationError(
            f"policy.dv_limit_km_s is {p.dv_limit_km_s!r} km/s; the {rules.name} rules "
            f"({rules.source.header_date} {rules.impulse_clause}) {_over_limit_verb(rules)} "
            "an impulse over "
            f"{rules.impulse_limit_km_s:g} km/s"
            + (" + 1e-5 m/s" if rules.impulse_epsilon_km_s else "")
        )
    if rules.total_dv_limit_km_s is not None:
        if p.total_dv_limit_km_s is None:
            raise ConfigurationError(
                f"policy.total_dv_limit_km_s is unset; the {rules.name} rules "
                f"({rules.source.header_date} {rules.total_clause}) disqualify a total over "
                f"{rules.total_dv_limit_km_s:g} km/s"
            )
        if p.total_dv_limit_km_s > rules.total_dv_limit_km_s:
            raise ConfigurationError(
                f"policy.total_dv_limit_km_s is {p.total_dv_limit_km_s!r} km/s; the "
                f"{rules.name} rules ({rules.source.header_date} {rules.total_clause}) "
                "disqualify a total over "
                f"{rules.total_dv_limit_km_s:g} km/s"
            )
    if p.min_burn_separation_s < rules.min_separation_s:
        raise ConfigurationError(
            f"policy.min_burn_separation_s is {p.min_burn_separation_s!r} s; the "
            f"{rules.name} rules ({rules.source.header_date}) penalise maneuvers closer "
            f"than {rules.min_separation_s:g} s"
        )
    if p.intercept_radius_km > rules.intercept_radius_km:
        raise ConfigurationError(
            f"policy.intercept_radius_km is {p.intercept_radius_km!r} km; the {rules.name} "
            f"rules ({rules.source.header_date}) intercept at {rules.intercept_radius_km:g} km"
        )
    if p.allow_body_intersecting_submission and rules.earth_crossing_clause is not None:
        raise ConfigurationError(
            "policy.allow_body_intersecting_submission is true; the "
            f"{rules.name} rules ({rules.source.header_date} {rules.earth_crossing_clause}) "
            "disqualify a trajectory that crosses the Earth before the interception"
        )
    if p.safe_burn_separation_s < p.min_burn_separation_s:
        raise ConfigurationError("safe burn separation cannot be below the formal limit")
    if p.min_propagate_duration_s <= 0:
        raise ConfigurationError("min_propagate_duration_s must be positive")
    if p.min_propagate_duration_s > p.min_burn_separation_s:
        raise ConfigurationError("min_propagate_duration_s cannot exceed the burn separation")
    if p.final_padding_s - SCHEDULE_TOLERANCE_S < p.min_propagate_duration_s:
        # _schedule_slots ends the mission at Tmax, so the last segment lasts
        # max_time_s - last_burn, and the only thing keeping that above zero is
        # the guard that refuses last_burn + final_padding_s > max_time_s. If
        # the padding is smaller than the floor, that guard passes a schedule
        # the renderer then refuses per segment -- a failure two steps from the
        # setting that caused it, and GMAT hangs rather than errors on the
        # zero-length goal it would have produced.
        raise ConfigurationError(
            "policy.final_padding_s cannot be below policy.min_propagate_duration_s plus "
            f"the {SCHEDULE_TOLERANCE_S:g} s schedule tolerance; it is the last Propagate "
            "segment's whole length, and the schedule guard is permissive by that much"
        )
    if p.intercept_radius_km <= 0 or p.mu_km3_s2 <= 0:
        raise ConfigurationError("intercept radius and mu must be positive")
    if s.one_burn_samples < 0 or s.two_burn_samples < 0:
        raise ConfigurationError("sample counts cannot be negative")
    if not 0 < s.dense_window_fraction <= 1:
        raise ConfigurationError("dense_window_fraction must be in (0,1]")
    if s.local_refine_top < 0:
        raise ConfigurationError("local_refine_top cannot be negative")
    if s.local_refine_steps < 1:
        raise ConfigurationError("local_refine_steps must be at least 1")
    if s.event_refine_top < 0:
        raise ConfigurationError("event_refine_top cannot be negative")
    if s.max_revolutions < 0 or s.max_revolutions > 6:
        raise ConfigurationError("max_revolutions must be between 0 and 6")
    if config.gmat.reruns < 1:
        raise ConfigurationError("gmat.reruns must be at least 1: certify runs the script that often")
    if config.gmat.workers < 1:
        raise ConfigurationError("gmat.workers must be at least 1")
    searched = interception_rules(p.rules)
    if s.ranking_search and (searched.horizon, searched.miss_distance) != (
        "announced", "closest-approach"
    ):
        raise ConfigurationError(
            "search.ranking_search needs a rulebook with an announced T_max that scores a miss "
            f"on its closest approach, as the ranking rules do; policy.rules is {p.rules}"
        )
    for name in ("ranking_transfer_starts", "ranking_random_starts", "ranking_polish_top"):
        if getattr(s, name) < 0:
            raise ConfigurationError(f"search.{name} cannot be negative")
    if s.ranking_screen_iterations < 1 or s.ranking_polish_iterations < 1:
        raise ConfigurationError("search.ranking_*_iterations must be at least 1")
    if s.ranking_search and s.ranking_transfer_starts + s.ranking_random_starts == 0:
        raise ConfigurationError(
            "search.ranking_search with no transfer or random starts would search nothing"
        )
    if not 0 <= s.ranking_correction_margin_km_s < p.dv_safe_limit_km_s:
        raise ConfigurationError(
            "search.ranking_correction_margin_km_s must be in [0, policy.dv_safe_limit_km_s)"
        )
    if s.ranking_force_model_margin_km < 0:
        raise ConfigurationError("search.ranking_force_model_margin_km cannot be negative")
    if config.gmat.correct_top < 0:
        raise ConfigurationError("gmat.correct_top cannot be negative")
    if config.gmat.verify_misses < 0:
        raise ConfigurationError("gmat.verify_misses cannot be negative")
    if config.gmat.correct_rounds < 1:
        raise ConfigurationError("gmat.correct_rounds must be at least 1")
    if not 0 < config.gmat.correct_aim_km <= p.intercept_radius_km - p.stop_depth_km:
        # A closest approach at the aim reaches the stop depth; one past it never does.
        raise ConfigurationError(
            "gmat.correct_aim_km must be in (0, policy.intercept_radius_km - "
            "policy.stop_depth_km]"
        )
    if s.ranking_max_burns < 1:
        raise ConfigurationError("search.ranking_max_burns must be at least 1")
    if s.ranking_pursuer_model not in {"two-body", "j2", "j2-spad-drag"}:
        raise ConfigurationError(
            "search.ranking_pursuer_model must be two-body, j2 or j2-spad-drag"
        )
    if s.ranking_pursuer_model_top < 1:
        raise ConfigurationError("search.ranking_pursuer_model_top must be at least 1")
    if config.gmat.organizer_compatible and not config.gmat.organizer_executable_sha256:
        # ORGANIZER_GMAT_VERIFIED is the strongest claim this project makes
        # short of certification, and without a fingerprint a single typo in a
        # profile promotes whatever binary happens to be on PATH.
        raise ConfigurationError(
            "gmat.organizer_compatible requires gmat.organizer_executable_sha256; a declaration "
            "with nothing to check it against would promote any GmatConsole"
        )
    for field_name in ("organizer_executable_sha256", "organizer_startup_sha256"):
        value = getattr(config.gmat, field_name)
        if value is None:
            continue
        digest = value.strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ConfigurationError(f"gmat.{field_name} must be 64 hex characters")
    if config.score:
        if config.score.time_unit not in {"s", "sec", "seconds"}:
            raise ConfigurationError("Only seconds are accepted for score.time_unit")
        if config.score.delta_v_unit not in {"m/s", "km/s"}:
            raise ConfigurationError("score.delta_v_unit must be m/s or km/s")
        validate_score_for_rules(config.score, interception_rules(p.rules))


def validate_score_for_rules(score: ScoreConfig, rules: InterceptionRules) -> None:
    """C_r and the units, which the 2026/11/04 revision fixed for the ranking rules only."""
    notice = f"the {rules.name} rules ({rules.distance_clause})"
    match rules.distance_term:
        case DistanceTerm.EXPONENTIAL:
            if score.cr_km is not None:
                raise ConfigurationError(
                    f"score.cr_km is set, but the {rules.name} rules' distance term "
                    f"({rules.distance_clause}) is {rules.distance_term.value} and has no C_r"
                )
        case DistanceTerm.LOGISTIC:
            if score.cr_km is None:
                raise ConfigurationError(
                    f"The score block has no cr_km. Under {notice} the distance term is "
                    f"{rules.distance_term.value} with C_r announced on the day, in km; it is "
                    "never guessed. scripts/ranking_day_config.py --cr-km writes it."
                )
            if not (math.isfinite(score.cr_km) and score.cr_km > 0.0):
                raise ConfigurationError(f"score.cr_km must be positive, got {score.cr_km!r} km")
        case _:
            assert_never(rules.distance_term)
    if rules.score_units is None:
        return
    # Seconds are checked for every rule set above. A delta-v unit slip moves a
    # 25-point term and raises nothing anywhere else.
    delta_v_unit = rules.score_units[1]
    if score.delta_v_unit != delta_v_unit:
        raise ConfigurationError(
            f"score.delta_v_unit is {score.delta_v_unit!r}; {notice} fix ΔV_team in "
            f"{delta_v_unit}, so kv is per {delta_v_unit} and Cv in {delta_v_unit}"
        )
