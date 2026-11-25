#!/usr/bin/env python3
"""Write the ranking day's config from what the organizer announces.

    python scripts/ranking_day_config.py --output run/day.yaml --horizon-s 6711.667
    python scripts/ranking_day_config.py --output run/day-scored.yaml --horizon-s 6711.667 \
        --kt-per-s 0.003 --ct-s 5369.334 --kv 0.0012 --cv 2000 --delta-v-unit m/s --cr-km 300 \
        --day-config run/day.yaml

    python scripts/ranking_day_config.py --output run/day.yaml \
        --announcement run/announced-A.toml --announcement run/announced-B.toml

The first copies config/ranking.yaml and adds T_max. The second copies the
day's first config, which --day-config names, and adds kt, Ct, kv, Cv and C_r
once they are announced; T_max is typed again and must equal the first's. A
first config written from --announcement entries takes the score from two
entries too, which may give the score alone.
Each loads the result the way every command will, and prints it back for a
second person to check against the announcement, and flies SC_Alien's two-body
coast in the day's script to check T_max as inspect does: one before its
perigee, or off an announced boundary, is refused. It reads the script through
GMAT when --gmat-console and --startup-file, or else GMAT_CONSOLE and
GMAT_STARTUP, name a pair, as they must for craft that start apart or are
inserted EarthFixed. Two --announcement files, entered apart by two people
from the announcement as printed (docs/ANNOUNCEMENT.md), take the place of
--horizon-s, --mission-boundary-km and the score flags: they must agree, T_max is
counted from their epochs, and every value they give for a craft is held to the
script. The rules as revised on 2026/11/04 give ΔV_team in m/s and C_r in km, so the config refuses km/s. The
time origin is always "earliest". It never overwrites a file.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

import yaml

from orbitwin.announcement import (
    Entry,
    against_script,
    horizon_s,
    read_pair,
    read_script,
    written_epochs,
)
from orbitwin.argtypes import positive_finite, positive_int
from orbitwin.config import (
    CLOSEST_APPROACH_WINDOWS,
    ORGANIZER_TIME_ORIGIN_ANSWER,
    OrbitWinConfig,
    load_config,
    ranking_profile,
)
from orbitwin.errors import ConfigurationError, OrbitWinError
from orbitwin.mission_clock import OVERRIDE_FLAG, clock_lines, horizon_warning
from orbitwin.pinned import assert_pinned_environment
from orbitwin.rank import FLAT_TERM_POINTS
from orbitwin.ranking_search import build_setup
from orbitwin.rules import RANKING, InterceptionRules, interception_rules
from orbitwin.score import (
    _dv_for_score,
    _logistic_descending,
    ceilings,
    distance_for_term_km,
    distance_term,
)
from orbitwin.state_probe import add_probe_arguments, load_scenario

#: How a config written from two entries names each in its first lines.
_ENTERED = "# Announcement entered by "
#: The flag for policy.tmax_as_announced_s.
AS_ANNOUNCED_FLAG = "--tmax-as-announced"
_SCORE_FLAGS = ("kt_per_s", "ct_s", "kv", "cv", "cr_km", "delta_v_unit")
#: A term that goes from 22.5 to 2.5 points within this share of its range is a step.
_STEP_SHARE = 0.01
#: Misses the distance term is printed at, besides its 49 and 25 points.
_PROBE_DISTANCES_KM = (10.0, 100.0, 1000.0, 10000.0)
#: Where Linux describes each CPU's core.
_CPU_SYSFS = Path("/sys/devices/system/cpu")
#: Ct under this share of T_max, or Cv of the 3 km/s cap, is read as a unit slip.
_SMALL_SHARE = 0.01
#: A term worth under this share of its 25 points at zero time or delta-v is dead.
_SATURATED_SHARE = 0.01
#: C_r outside this range, in km, is no scenario's distance scale.
_CR_RANGE_KM = (1.0, 1.0e5)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _cpus() -> list[int]:
    """The CPUs this process may run on, which is what solve gets."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def _cores(cpus: list[int], sysfs: Path) -> int | None:
    """Physical cores under `cpus`, or None when the topology cannot be read.

    A core is the set of CPUs that share it. core_id is unique only within a
    package, and on some kernels and architectures only within a die or cluster.
    """
    cores = set()
    for cpu in cpus:
        topology = sysfs / f"cpu{cpu}" / "topology"
        for name in ("core_cpus_list", "thread_siblings_list"):  # the second before 5.3
            try:
                cores.add((topology / name).read_text(encoding="ascii").strip())
                break
            except OSError:
                continue
        else:
            return None
    return len(cores) or None


def _clock(seconds: float) -> str:
    millis = round(abs(seconds) * 1000.0)
    minutes, millis = divmod(millis, 60_000)
    hours, minutes = divmod(minutes, 60)
    sign = "-" if seconds < 0 else ""
    return f"{sign}{hours} h {minutes:02d} min {millis / 1000.0:06.3f} s"


def _step_warning(term: str, k: float, span: float, unit: str) -> list[str]:
    """25/(1+e^{k(x-c)}) falls from 22.5 to 2.5 points over a width 2 ln 9 / |k|."""
    width = 4.394 / abs(k) if k else float("inf")
    if width >= _STEP_SHARE * span:
        return []
    return [f"WARNING: the {term} term drops from 22.5 to 2.5 points within {width:.3g} {unit}, "
            f"under 1% of 0 to {span:g} {unit}: a step, not a slope. A unit slip looks like this."]


def _plausibility(kt: float, ct: float, kv: float, cv: float, cr_km: float, horizon_s: float,
                  cap: float, unit: str) -> list[str]:
    """What no announcement would say, each a WARNING; ``cap`` is the 3 km/s cap in ``unit``."""
    lines = []
    for name, k in (("kt", kt), ("kv", kv)):
        if not k > 0.0:
            lines.append(f"WARNING: {name} = {k!r} is not positive: the term would reward a later "
                         "or costlier answer. Read it back against the announcement.")
    if ct < _SMALL_SHARE * horizon_s:
        lines.append(f"WARNING: Ct = {ct!r} s is under {_SMALL_SHARE:.0%} of T_max = "
                     f"{horizon_s!r} s. Ct is in seconds; check the unit.")
    if cv < _SMALL_SHARE * cap:
        lines.append(f"WARNING: Cv = {cv!r} {unit} is under {_SMALL_SHARE:.0%} of the "
                     f"{cap:g} {unit} cap. Cv is in {unit}; check the unit.")
    elif cv > cap:
        # The preliminary's own Cv was 9064.3 m/s: not a slip, but nothing reaches it.
        lines.append(f"NOTE: Cv = {cv!r} {unit} is above the {cap:g} {unit} cap no answer may "
                     "exceed, so delta-v costs little; read it back.")
    for term, k, c, x in (("time", kt, ct, "zero time"), ("delta-v", kv, cv, "zero delta-v")):
        at_zero = _logistic_descending(25.0, k * (0.0 - c))
        if at_zero < _SATURATED_SHARE * 25.0:
            lines.append(f"WARNING: the {term} term is {at_zero:.4g} points at {x} "
                         f"({'kt*Ct' if term == 'time' else 'kv*Cv'} = {k * c:.4g}): "
                         "saturated at zero before any answer. Check the unit.")
    low, high = _CR_RANGE_KM
    if not low <= cr_km <= high:
        lines.append(f"WARNING: C_r = {cr_km!r} km is outside {low:g} to {high:g} km, no "
                     "scenario's distance scale. Read it back against the announcement.")
    return lines


def _distance_lines(cr_km: float, zero_burn_km: float | None,
                    rules: InterceptionRules = RANKING) -> list[str]:
    """The term where two people can check C_r against the scenario, and its range."""
    rows = [(d, "") for d in _PROBE_DISTANCES_KM] + [
        (distance_for_term_km(49.0, rules, cr_km), "  (D = 49, 5 + 0.2 C_r)"),
        (distance_for_term_km(25.0, rules, cr_km), "  (D = 25, 5 + 1.048 C_r)"),
    ]
    if zero_burn_km is not None:
        rows.append((zero_burn_km, "  (the zero burn's closest approach)"))
    lines = [f"  distance d = {d:10.3f} km   -> {distance_term(d, rules, cr_km):7.4f}{label}"
             for d, label in sorted(rows)]
    # Range, not a diagnosis: C_r alone cannot show which unit it was typed in. A small
    # C_r is plausible, so it is a note; only a C_r no scenario could use warns.
    if distance_term(500.0, rules, cr_km) < 1.0:
        lines.append(
            f"NOTE: at C_r = {cr_km:g} km a miss beyond "
            f"{distance_for_term_km(1.0, rules, cr_km):.0f} km scores under 1 point on "
            f"distance. The rules say the term falls smoothly with distance ({rules.distance_clause}: "
            "平滑遞減), and the organizer's revision email of 2026-11-04 that it does not fall "
            "to zero within hundreds of km (不會在數百公里內即降為零分); read C_r back against "
            "the announcement."
        )
    if distance_term(_PROBE_DISTANCES_KM[-1], rules, cr_km) > 49.0:
        lines.append(
            f"WARNING: at C_r = {cr_km:g} km the distance term is still above 49 at "
            f"{_PROBE_DISTANCES_KM[-1]:g} km, so it hardly separates any two misses. Read C_r "
            "back against the announcement."
        )
    return lines


def _readback(config_path: Path, zero_burn_km: float | None = None) -> list[str]:
    config = load_config(config_path)
    rules = interception_rules(config.policy.rules)
    policy, score = config.policy, config.score
    assert policy.horizon_s is not None
    search_aim_km = inspect.signature(build_setup).parameters["aim_radius_km"].default
    flown = policy.minimum_flown_altitude_km
    cpus = _cpus()
    cores = _cores(cpus, _CPU_SYSFS)
    may_use = ("cores unknown" if cores is None else _plural(cores, "core")) + (
        f", {_plural(len(cpus), 'thread')}"
    )
    # Each GmatConsole run keeps one core busy, so more runs than cores only
    # slow each other. The profile's 8 was measured on 14 cores.
    limit, cpu_unit = (len(cpus), "thread") if cores is None else (cores, "core")
    lines = [
        f"config      : {config_path}",
        f"rules       : {policy.rules}",
        f"craft       : maneuvering {config.pursuer_name}, target {config.target_name}",
        f"T_max       : {policy.horizon_s!r} s = {_clock(policy.horizon_s)}",
    ]
    if policy.mission_boundary_km is not None:
        lines.append(
            f"boundary    : {policy.mission_boundary_km!r} km; this tool refuses, and "
            "inspect warns, when SC_Alien is not there at T_max"
        )
    if policy.tmax_as_announced_s is not None:
        lines.append(
            f"as announced: T_max {policy.tmax_as_announced_s!r} s is taken although SC_Alien "
            f"may be off the boundary there ({AS_ANNOUNCED_FLAG}, policy.tmax_as_announced_s)"
        )
    if policy.accept_announced_horizon_before_perigee_s is not None:
        lines.append(
            f"override    : T_max {policy.accept_announced_horizon_before_perigee_s!r} s is "
            "taken if inspect finds it before SC_Alien's perigee, as the organizer confirmed "
            f"in writing ({OVERRIDE_FLAG}); inspect warns, and every output records it, only "
            "when it is used"
        )
    window = policy.closest_approach_window
    lines += [
        f"time origin : {policy.mission_time_origin}: T_team, T_max and Ct count from the "
        "earlier craft's epoch (organizer, 2026-11-04)",
        "              inspect names that craft, the head start and T_max on the solver clock",
        "d_min window: " + (
            f"{window.replace('_', ' ')} (not yet answered)" if window is not None
            else "not set; inspect refuses craft that start apart until "
            "--closest-approach-window is given (not yet answered)"
        ),
    ]
    if policy.add_epoch_alignment:
        lines.append(
            "alignment   : added as the first command when the craft start apart without one "
            "(signed off 2026-11-03); its rows lead the uploaded report"
        )
    lines.append(
        "head start  : policy.head_start_burns on: when the pursuer starts first, an open "
        "template's script may burn before the target's epoch (signed off 2026-11-08); rank "
        f"takes such a plan only {policy.head_start_min_gain_points:g} points or more above the "
        "best without (policy.head_start_min_gain_points); inspect says whether it is open"
        if policy.head_start_burns
        else "head start  : policy.head_start_burns off: no burn before the later craft's "
        "epoch (--no-head-start-burns, or the base's setting)"
    )
    lines.append(
        "plots       : gmat.console_copy on: a script GmatConsole refuses for its plots alone "
        "is certified through its proven copy less them, whose report is packaged "
        "(signed off 2026-11-06)"
        if config.gmat.console_copy
        else "plots       : gmat.console_copy off: a script GmatConsole refuses for its plots "
        "stops certify; its report comes from the GMAT GUI (RUNBOOK step 6)"
    )
    lines += [
        "",
    ]
    limits = [
        ("limit", f"the rules ({rules.source.header_date})", "ours (engineering, changeable)"),
        ("per impulse", f"{rules.impulse_limit_km_s * 1000:g} m/s + "
         f"{rules.impulse_epsilon_km_s * 1000:g} m/s ({rules.impulse_clause})",
         f"{policy.dv_safe_limit_km_s * 1000:g} m/s"),
        ("total delta-v", f"{rules.total_dv_limit_km_s:g} km/s ({rules.total_clause})",
         f"{policy.total_dv_safe_limit_km_s:g} km/s"),
        ("burn separation", f"{rules.min_separation_s:g} s, else "
         f"-{rules.penalty_points:g} each ({rules.separation_clause})",
         f"{policy.safe_burn_separation_s:g} s"),
        ("burns searched", "none; the window allows them",
         f"search.ranking_max_burns = {config.search.ranking_max_burns}"),
        ("altitude", f"surface, before intercept ({rules.earth_crossing_clause})",
         f"{flown:g} km on GMAT's flight, whole arc" if flown is not None else "none"),
        ("search aim", f"{rules.intercept_radius_km:g} km sphere",
         f"{search_aim_km:g} km two-body, {config.gmat.correct_aim_km:g} km on GMAT"),
        ("force-model margin", "none",
         f"{config.search.ranking_force_model_margin_km:g} km of altitude"
         + (", and a 0 km search" if config.search.ranking_search_without_margin else "")),
        ("report name", "TeamXX replaced by the team number (organizer)",
         f"policy.team_number = {policy.team_number}: each TeamXX placeholder takes it; "
         f"certify refuses unless the official report then carries Team{policy.team_number}"
         + (f"; KEPT AS WRITTEN: {policy.report_filename_as_written!r} (organizer's answer)"
            if policy.report_filename_as_written is not None else "")),
        ("where a hit ends", "not stated",
         f"{policy.stop_depth_km * 1000:g} m inside the sphere, policy.stop_depth_km (the "
         "preliminary's validator read the last row)"),
        ("GMAT runs at once", "not stated",
         f"gmat.workers = {config.gmat.workers}; this process may use {may_use}"),
    ]
    lines += [f"{name:<20}  {rule:<44}  {ours}" for name, rule, ours in limits]
    if config.gmat.workers > limit:
        lines.append(f"WARNING: gmat.workers {config.gmat.workers} is more than the "
                     f"{_plural(limit, cpu_unit)} this process may use; write a new config with "
                     f"--gmat-workers {limit} or fewer.")
    lines.append("")
    if score is None:
        lines.append("score       : none yet; solve puts GMAT-verified candidates first and hits "
                     "before misses, then sorts by delta-v, then closest approach")
        return lines
    unit = score.delta_v_unit
    lines += [
        f"kt          : {score.kt!r} per s",
        f"Ct          : {score.ct!r} s = {_clock(score.ct)}",
        f"kv          : {score.kv!r} per {unit}",
        f"Cv          : {score.cv!r} {unit}",
        f"C_r         : {score.cr_km!r} km",
        # 25/(1+e^{k(x-c)}) is steepest at x = c, at 25 |k| / 4 points per unit of x.
        f"steepest    : {6.25 * abs(score.kt):.4g} pt/s at T = Ct, "
        f"{6.25 * abs(score.kv) * _dv_for_score(1.0, unit):.4g} pt per km/s at ΔV = Cv",
        "",
        "the terms over what a team can reach:",
    ]
    horizon = policy.horizon_s
    total_dv = policy.total_dv_limit_km_s
    # The ranking rules refuse a config without either (validated above).
    assert horizon is not None and total_dv is not None
    # Only times a team can reach: a Ct outside them would add a 12.5 no answer gets.
    times = sorted({0.0, min(max(score.ct, 0.0), horizon), horizon})
    time_points = [(t, _logistic_descending(25.0, score.kt * (t - score.ct))) for t in times]
    dv_points = [(dv, _logistic_descending(25.0, score.kv * (_dv_for_score(dv, unit) - score.cv)))
                 for dv in (0.0, 0.5 * total_dv, total_dv)]
    lines += [f"  time    T = {t:10.3f} s      -> {points:7.4f}" for t, points in time_points]
    lines += [f"  delta-v ΔV = {dv:6.3f} km/s   -> {points:7.4f}" for dv, points in dv_points]
    for term, points in (("time", time_points), ("delta-v", dv_points)):
        values = [value for _, value in points]
        if max(values) - min(values) < FLAT_TERM_POINTS:
            lines.append(
                f"WARNING: the {term} term moves under {FLAT_TERM_POINTS} points over its whole "
                f"range, so it cannot separate answers. Check the unit; if it is right, the "
                f"{term} costs nothing that matters."
            )
    total_in_unit = _dv_for_score(total_dv, unit)
    lines += _step_warning("time", score.kt, horizon, "s")
    lines += _step_warning("delta-v", score.kv, total_in_unit, unit)
    # Validated above: the ranking rules refuse a score block without it.
    assert score.cr_km is not None
    lines += _plausibility(score.kt, score.ct, score.kv, score.cv, score.cr_km, horizon,
                           total_in_unit, unit)
    lines += _distance_lines(score.cr_km, zero_burn_km, rules)
    bound = ceilings(score, horizon, total_dv)
    lines += [
        "",
        f"no answer scores above {bound.any_answer:.4f}",
        f"no miss scores above   {bound.any_miss:.4f} (T_team = T_max); a verified hit above "
        "that cannot be beaten by any miss",
    ]
    return lines


def _announced_lines(entry: Entry, script: Path, config: OrbitWinConfig) -> list[str]:
    """The agreed entry against the script's own fields, before anything is flown from them;
    raises on any value the script contradicts."""
    assert config.pursuer_name is not None
    wrong, unchecked, agree = against_script(entry, read_script(script), config.pursuer_name,
                                             interception_rules(config.policy.rules))
    if wrong:
        raise ConfigurationError(
            "the script is not what was announced: " + "; ".join(wrong) + ". Stop and ask "
            "the organizer which is right")
    lines = [f"announced   : {', '.join(agree)} agree with the script, to the digits printed"
             if agree else "announced   : no craft value given to hold the script to"]
    lines += [f"WARNING: {item} was announced but not compared; compare it by hand"
              for item in unchecked]
    return lines


def _horizon_lines(
    config_path: Path, script: Path, console: str | None, startup: str | None,
    pair_source: str = "--gmat-console", announced: Entry | None = None,
) -> list[str]:
    """T_max against SC_Alien's coast in ``script``, as inspect checks it; raises on one it
    cannot have: before its perigee (the parser's refusal) or off an announced boundary."""
    config = load_config(config_path)
    held = [] if announced is None else _announced_lines(announced, script, config)
    scenario, _, _ = load_scenario(script, config, gmat_console=console, startup_file=startup)
    check = scenario.horizon_check
    horizon_s = config.policy.horizon_s
    if check is None or horizon_s is None:
        raise ConfigurationError(f"{script} gave no T_max check under {config_path}")
    taken: list[str] = []
    if check.off_boundary and config.policy.tmax_as_announced_s != horizon_s:
        raise ConfigurationError(
            horizon_warning(check, horizon_s).removeprefix("WARNING: ")
            + f". Read the announcement again; if this is the announced value, rerun with "
            f"{AS_ANNOUNCED_FLAG}"
        )
    if check.off_boundary:
        assert check.boundary_km is not None
        off_km = check.radius_km - check.boundary_km
        rate = check.radial_rate_km_s
        off_s = f"{off_km / rate:+.1f} s at {rate:+.4f} km/s" if rate > 0.0 else "not climbing"
        taken.append(
            f"WARNING: T_max {horizon_s!r} s taken as announced ({AS_ANNOUNCED_FLAG}): SC_Alien "
            f"is {off_km:+.3f} km off the {check.boundary_km!r} km boundary there ({off_s}, "
            "two-body)"
        )
    read = ("in-process: no GMAT pair given or exported" if console is None
            else f"through GMAT {console} ({pair_source})")
    return ["", f"script      : {script}, read {read}", *clock_lines(scenario, config.policy),
            *taken, *held]


def main() -> int:
    # No abbreviations: --cr would otherwise stand for --cr-km and drop the unit.
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0],
                                     allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--script", type=Path, default=Path("input/original/official.script"),
                        help="the day's official script, whose SC_Alien T_max is checked "
                        "against (default: step 1's input/original/official.script)")
    # Default: GMAT_CONSOLE and GMAT_STARTUP, as RUNBOOK step 0 exports them.
    add_probe_arguments(parser)
    parser.add_argument("--base", type=Path, default=ranking_profile())
    parser.add_argument("--horizon-s", type=float, help="T_max, as announced")
    parser.add_argument("--announcement", type=Path, action="append",
                        help="an entry of the announcement, given twice: one by each of two "
                        "people; in place of --horizon-s and the score flags")
    parser.add_argument("--kt-per-s", type=float)
    parser.add_argument("--ct-s", type=float)
    parser.add_argument("--kv", type=float, help="per --delta-v-unit")
    parser.add_argument("--cv", type=float, help="in --delta-v-unit")
    parser.add_argument("--delta-v-unit", choices=("m/s",),
                        help="m/s, as the 2026/11/04 revision fixes kv and Cv; typed to be read")
    parser.add_argument("--cr-km", type=positive_finite, help="C_r, in km (2026/11/04 revision)")
    parser.add_argument("--zero-burn-dmin-km", type=positive_finite,
                        help="the zero burn's closest approach, as step 4's solve printed it, "
                        "to price it at C_r")
    parser.add_argument("--team-number", type=positive_int,
                        help="the team's number, which replaces TeamXX in the official report's "
                        "file name; needed unless the base config sets policy.team_number")
    parser.add_argument("--pursuer-name", help="if the template does not use SC_Human")
    parser.add_argument("--target-name", help="if the template does not use SC_Alien")
    parser.add_argument("--mission-time-origin", default="earliest",
                        help="only 'earliest', always written (organizer, 2026-11-04)")
    parser.add_argument("--closest-approach-window", choices=CLOSEST_APPROACH_WINDOWS,
                        help="where d_min's window starts, when the craft start apart")
    parser.add_argument("--day-config", type=Path,
                        help="the day's first config: a scored config is it plus the score, "
                        "and --horizon-s must equal its T_max")
    parser.add_argument(OVERRIDE_FLAG, type=float, metavar="T_MAX_S",
                        help="take this announced T_max although it is before SC_Alien's "
                        "perigee; only after the organizer confirmed it in writing")
    parser.add_argument("--gmat-workers", type=int,
                        help="GmatConsole runs at once, if not the base's")
    parser.add_argument("--report-filename-as-written",
                        help="the official report's Filename, kept as the template writes it; "
                        "only after the organizer answers in writing to keep a name with no "
                        "team placeholder")
    parser.add_argument("--mission-boundary-km", type=float,
                        help="the mission boundary as a geocentric radius, if announced")
    parser.add_argument("--no-head-start-burns", action="store_true",
                        help="policy.head_start_burns off: no burn before the later craft's "
                        "epoch, if the organizer says so on the day")
    parser.add_argument(AS_ANNOUNCED_FLAG, action="store_true",
                        help="take --horizon-s although SC_Alien is off --mission-boundary-km "
                        "there, as a T_max announced rounded is; only after reading the "
                        "announcement again")
    args = parser.parse_args()
    try:
        assert_pinned_environment()
    except OrbitWinError as error:
        print(f"refused, nothing written: {error}", file=sys.stderr)
        return 2

    given = [getattr(args, name) is not None for name in _SCORE_FLAGS]
    if args.announcement is not None:
        if len(args.announcement) != 2:
            parser.error("--announcement is given twice, once for each person's entry")
        if args.horizon_s is not None or args.mission_boundary_km is not None or any(given):
            parser.error("--announcement gives T_max, the boundary and the score; drop "
                         "--horizon-s, --mission-boundary-km and the score flags")
    elif args.horizon_s is None:
        parser.error("give --horizon-s, or two --announcement entries")
    if any(given) and not all(given):
        missing = [f"--{name.replace('_', '-')}" for name, ok in zip(_SCORE_FLAGS, given,
                                                                      strict=True) if not ok]
        parser.error(f"the score needs all {len(_SCORE_FLAGS)} of its flags; "
                     f"missing {' '.join(missing)}")
    if args.zero_burn_dmin_km is not None and not (all(given) or (
            args.announcement is not None and args.day_config is not None)):
        parser.error("--zero-burn-dmin-km is priced with the score; give the score flags too")
    if args.output.exists():
        parser.error(f"{args.output} exists; write the day's config to a new file")
    if not args.script.is_file():
        parser.error(f"--script {args.script} is not a file: T_max is checked against the "
                     "day's script, step 1's input/original/official.script")
    # An empty flag is an unset variable expanded; no flags take the pair step 0 exports.
    args.gmat_console, args.startup_file = args.gmat_console or None, args.startup_file or None
    pair_source = "--gmat-console"
    if args.gmat_console is None and args.startup_file is None:
        args.gmat_console = os.environ.get("GMAT_CONSOLE") or None
        args.startup_file = os.environ.get("GMAT_STARTUP") or None
        pair_source = "GMAT_CONSOLE"
    if (args.gmat_console is None) != (args.startup_file is None):
        parser.error("--gmat-console and --startup-file, or else GMAT_CONSOLE and GMAT_STARTUP, "
                     "must be given together")
    if args.mission_time_origin != "earliest":
        parser.error(f"--mission-time-origin is {args.mission_time_origin!r}; it is 'earliest', "
                     f"by {ORGANIZER_TIME_ORIGIN_ANSWER}")
    if all(given) and args.day_config is None:
        parser.error("a scored config needs --day-config, the day's first config, so its "
                     "T_max and time origin cannot differ from it unseen")
    source = args.day_config or args.base
    try:
        rules = load_config(source).policy.rules
        source_text = source.read_text(encoding="utf-8")
        raw = yaml.safe_load(source_text)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ConfigurationError) as exc:
        flag = "--day-config" if args.day_config else "--base"
        print(f"refused, nothing written: {flag} {source}: {exc}", file=sys.stderr)
        return 2
    chosen = interception_rules(rules)
    # The ranking round's shape: an announced T_max, ranked misses, the earlier epoch's clock.
    if not (chosen.horizon == "announced" and chosen.miss_ranked
            and chosen.time_origin == "earliest"):
        parser.error(f"{source} is not under the ranking rules but {chosen.id}")
    entries: tuple[Entry, Entry, Entry] | None = None
    if args.announcement is not None:
        pursuer = args.pursuer_name or raw.get("pursuer_name")
        target = args.target_name or raw.get("target_name")
        if not pursuer or not target:
            parser.error(f"{source} does not name both craft; pass --pursuer-name and "
                         "--target-name")
        try:
            entries = read_pair(*args.announcement)
            if entries[2].t_max is None and entries[2].t_max_s is None:
                # The score alone, announced after T_max: step 2's T_max stands.
                if args.day_config is None:
                    raise ConfigurationError("the entries give no T_max; step 2's need it")
                args.horizon_s = raw["policy"].get("horizon_s")
            else:
                args.horizon_s = horizon_s(entries[2], chosen, pursuer, target,
                                           written_epochs(read_script(args.script)))
            if entries[2].boundary is not None:
                args.mission_boundary_km = float(entries[2].boundary.value)
        except (OrbitWinError, OSError) as exc:
            print(f"refused, nothing written: {exc}", file=sys.stderr)
            return 2
        score = entries[2].score
        if args.day_config is not None:
            if score is None:
                parser.error("the entries give no [score]; a scored config needs it")
            args.kt_per_s, args.ct_s = float(score["kt"].value), float(score["Ct"].value)
            args.kv, args.cv = float(score["kv"].value), float(score["Cv"].value)
            args.cr_km, args.delta_v_unit = float(score["C_r"].value), "m/s"
            given = [True] * len(_SCORE_FLAGS)
    override = args.accept_announced_horizon_before_perigee_s
    if override is not None and override != args.horizon_s:
        parser.error(f"{OVERRIDE_FLAG} {override!r} is not --horizon-s {args.horizon_s!r}: it "
                     "names the one announced T_max it accepts")
    if args.day_config is not None:
        # The day's config is the base: only what is typed again must agree with it.
        typed = [
            ("--horizon-s", args.horizon_s, raw["policy"].get("horizon_s")),
            ("--team-number", args.team_number, raw["policy"].get("team_number")),
            ("--pursuer-name", args.pursuer_name, raw.get("pursuer_name")),
            ("--target-name", args.target_name, raw.get("target_name")),
            ("--closest-approach-window", args.closest_approach_window,
             raw["policy"].get("closest_approach_window")),
            ("--mission-boundary-km", args.mission_boundary_km,
             raw["policy"].get("mission_boundary_km")),
            (OVERRIDE_FLAG, override,
             raw["policy"].get("accept_announced_horizon_before_perigee_s")),
            (AS_ANNOUNCED_FLAG, args.horizon_s if args.tmax_as_announced else None,
             raw["policy"].get("tmax_as_announced_s")),
            ("--report-filename-as-written", args.report_filename_as_written,
             raw["policy"].get("report_filename_as_written")),
            ("--no-head-start-burns", False if args.no_head_start_burns else None,
             raw["policy"].get("head_start_burns", False)),
        ]
        differ = [f"{flag} {mine!r} here, {theirs!r} in {args.day_config}"
                  for flag, mine, theirs in typed if mine is not None and mine != theirs]
        added = [flag for flag, mine, theirs in typed if mine is not None and theirs is None]
        if differ:
            advice = (
                f"Write step 2's config again with {' '.join(added)}, then this one from it"
                if added and len(added) == len(differ)
                else "Check both against the announcement and write the wrong one again"
            )
            print(f"refused, nothing written: {'; '.join(differ)}. {advice}", file=sys.stderr)
            return 2
        if raw.get("score") is not None:
            parser.error(f"{args.day_config} is scored already; name the day's first config")
        if args.announcement is None and _ENTERED in source_text:
            parser.error(f"{args.day_config} took T_max from two entries of the announcement; "
                         "give the score the same way, --announcement twice")
    written = raw["policy"].get("team_number")
    if args.team_number is not None and written not in (None, args.team_number):
        parser.error(f"--team-number {args.team_number}, but {source} sets "
                     f"policy.team_number {written}; give one of them")
    if args.team_number is not None:
        raw["policy"]["team_number"] = args.team_number
    if not raw["policy"].get("team_number"):
        parser.error(f"{source} sets no policy.team_number; pass --team-number, which the "
                     "ranking rules need for the official report's file name")
    raw["policy"]["horizon_s"] = args.horizon_s
    for key, value in (("pursuer_name", args.pursuer_name), ("target_name", args.target_name)):
        if value is not None:
            raw[key] = value
        if not raw.get(key):
            parser.error(f"{source} names no {key}; pass --{key.replace('_', '-')}")
    raw["policy"]["mission_time_origin"] = "earliest"
    if args.closest_approach_window is not None:
        raw["policy"]["closest_approach_window"] = args.closest_approach_window
    if args.mission_boundary_km is not None:
        raw["policy"]["mission_boundary_km"] = args.mission_boundary_km
    if override is not None:
        raw["policy"]["accept_announced_horizon_before_perigee_s"] = override
    if args.tmax_as_announced:
        raw["policy"]["tmax_as_announced_s"] = args.horizon_s
    if args.report_filename_as_written is not None:
        raw["policy"]["report_filename_as_written"] = args.report_filename_as_written
    if args.no_head_start_burns:
        raw["policy"]["head_start_burns"] = False
    if args.gmat_workers is not None:
        # A gmat: key with every field commented out loads as None.
        raw["gmat"] = {**(raw.get("gmat") or {}), "workers": args.gmat_workers}
    if all(given):
        raw["score"] = {
            "kt": args.kt_per_s, "ct": args.ct_s, "kv": args.kv, "cv": args.cv,
            "time_unit": "s", "delta_v_unit": args.delta_v_unit, "cr_km": args.cr_km,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = f"# Written by scripts/ranking_day_config.py from {source.name}.\n"
    if entries is not None:
        header += "".join(f"{_ENTERED}{entry.entered_by}: {entry.path.name}, "
                          f"SHA-256 {entry.sha256}\n" for entry in entries[:2])
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(header + yaml.safe_dump(raw, sort_keys=False))
    try:
        lines = _readback(args.output, args.zero_burn_dmin_km)
        lines += _horizon_lines(args.output, args.script, args.gmat_console, args.startup_file,
                                pair_source, None if entries is None else entries[2])
        if entries is not None and entries[2].score is not None and args.day_config is None:
            lines.append("score       : in the entries, and written by step 5's config, which "
                         "names this one with --day-config")
        print("\n".join(lines))
    except (OrbitWinError, OSError) as exc:
        args.output.unlink()
        print(f"refused, nothing written: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
