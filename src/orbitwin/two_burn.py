from __future__ import annotations

import math

import numpy as np
from scipy.stats import qmc

from .config import OrbitWinConfig
from .constraints import apply_constraints
from .frames import inertial_to_script_components, script_components_to_inertial
from .lambert import solve_lambert_all
from .models import Burn, Candidate, Scenario, as_vector3, burn_frame, template_maneuvers
from .orbit import propagate_universal
from .score import apply_score
from .search import (
    drop_post_intercept_burns,
    record_event,
    refine_events,
    resolve_workers,
)
from .trajectory import apply_burn


def _unit_ball_vector(u_radius: float, u_cos: float, u_azimuth: float, radius: float) -> np.ndarray:
    magnitude = radius * float(u_radius) ** (1.0 / 3.0)
    return _direct(magnitude, u_cos, u_azimuth)


def _shell_vector(
    u_radius: float, u_cos: float, u_azimuth: float, inner: float, outer: float
) -> np.ndarray:
    """A direction-uniform sample from the shell between two magnitudes.

    Widening the ball would have been simpler, but the magnitude map is
    ``radius * u**(1/3)``: scaling the radius moves every existing sample rather
    than adding to them. Measured on the different-plane scenario, that alone
    changed the robust two-burn set and made its best total delta-v worse,
    2.06893 -> 2.07135 km/s. Sampling the shell separately leaves the production
    ball untouched.
    """
    lower = inner**3
    magnitude = (lower + float(u_radius) * (outer**3 - lower)) ** (1.0 / 3.0)
    return _direct(magnitude, u_cos, u_azimuth)


def _direct(magnitude: float, u_cos: float, u_azimuth: float) -> np.ndarray:
    cos_theta = 2.0 * float(u_cos) - 1.0
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * math.pi * float(u_azimuth)
    return magnitude * np.array([sin_theta * math.cos(phi), cos_theta, sin_theta * math.sin(phi)])


def search_two_burn_global(scenario: Scenario, config: OrbitWinConfig) -> list[Candidate]:
    # A template that already maneuvers can express exactly the impulses it has
    # slots for. One that does not has its maneuvers written by us, so two are
    # always expressible -- however many ImpulsiveBurn resources it happens to
    # declare. Gating on the resource count made this vanish for a template
    # declaring one, silently, on a mission identical in every other way:
    #
    #   declared 0  ->  0 two-burn candidates
    #   declared 1  ->  0 two-burn candidates
    #   declared 2  ->  6 two-burn candidates
    if template_maneuvers(scenario) and len(scenario.burn_slots) < 2:
        return []
    if not config.search.two_burn_enabled:
        return []
    count = config.search.two_burn_samples
    if count <= 0:
        return []
    policy = config.policy
    # Three passes. The first reproduces the production search exactly -- same
    # ball radius, same separation floor, same seed, same sample count -- so its
    # results are unchanged. The other two reach the parts of the legal space
    # that lie outside the production margins, each on its own seed so it cannot
    # perturb the first.
    passes: list[tuple[int, int, tuple[float, float], tuple[float, float]]] = [
        (count, 17, (0.0, policy.dv_safe_limit_km_s), (policy.safe_burn_separation_s, -1.0)),
    ]
    shell = max(1, count // 16)
    if policy.dv_limit_km_s > policy.dv_safe_limit_km_s:
        passes.append(
            (
                shell,
                4013,
                (policy.dv_safe_limit_km_s, policy.dv_limit_km_s),
                (policy.safe_burn_separation_s, -1.0),
            )
        )
    if policy.safe_burn_separation_s > policy.min_burn_separation_s:
        passes.append(
            (
                shell,
                8017,
                (0.0, policy.dv_safe_limit_km_s),
                (policy.min_burn_separation_s, policy.safe_burn_separation_s),
            )
        )

    tmax = scenario.max_time_s
    min_tof = config.search.min_tof_s
    # burn_frame is what knows that a position past the declared resources
    # takes this project's own frame; reading burn_slots directly was the same
    # assumption that there are always two of them.
    frame1_cs, frame1_origin, frame1_axes = burn_frame(scenario, 0)
    frame2_cs, frame2_origin, frame2_axes = burn_frame(scenario, 1)
    candidates: list[Candidate] = []

    for sample_count, seed_offset, (dv_inner, dv_outer), (gap_floor, gap_ceiling) in passes:
      min_gap = gap_floor
      engine = qmc.Sobol(d=6, scramble=True, seed=config.search.seed + seed_offset)
      samples = engine.random_base2(math.ceil(math.log2(sample_count)))[:sample_count]
      for sample in samples:
          minimum_total = min_gap + min_tof
          if tmax <= minimum_total:
              break
          # A first burn at t=0 renders a zero-length Propagate, which hangs GMAT.
          # Above that floor the whole legal window for the first burn is sampled.
          floor_s = config.policy.min_propagate_duration_s
          span = max(0.0, tmax - minimum_total - floor_s)
          t1 = floor_s + float(sample[0]) * span
          remaining_after_t1 = tmax - t1 - minimum_total
          # The 0.75 factor that used to shrink this span excluded the last quarter
          # of the legal second-burn window for no stated reason.
          gap_span = (
              max(0.0, remaining_after_t1)
              if gap_ceiling < 0.0
              else max(0.0, min(gap_ceiling - gap_floor, remaining_after_t1))
          )
          t2 = t1 + min_gap + float(sample[1]) * gap_span
          if t2 + min_tof > tmax:
              continue
          arrival = t2 + min_tof + float(sample[2]) * (tmax - t2 - min_tof)

          state1 = propagate_universal(scenario.pursuer.initial_state, t1, scenario.mu_km3_s2)
          components1 = (
              _unit_ball_vector(sample[3], sample[4], sample[5], dv_outer)
              if dv_inner <= 0.0
              else _shell_vector(sample[3], sample[4], sample[5], dv_inner, dv_outer)
          )
          dv1 = script_components_to_inertial(
              components1, state1, frame1_cs, frame1_origin, frame1_axes
          )
          after1 = apply_burn(state1, as_vector3(dv1))
          state2_pre = propagate_universal(after1, t2 - t1, scenario.mu_km3_s2)
          target_arrival = propagate_universal(
              scenario.target.initial_state, arrival, scenario.mu_km3_s2
          )
          solutions = solve_lambert_all(
              state2_pre.r,
              target_arrival.r,
              arrival - t2,
              scenario.mu_km3_s2,
              max_revolutions=config.search.max_revolutions,
              endpoint_tolerance_km=config.search.endpoint_tolerance_km,
              reference_normal=np.cross(state2_pre.r, state2_pre.v),
          )
          for solution in solutions:
              dv2 = solution.v1 - state2_pre.v
              magnitude2 = float(np.linalg.norm(dv2))
              if magnitude2 > config.policy.dv_limit_km_s + 1.0e-12:
                  continue
              components2 = inertial_to_script_components(
                  dv2, state2_pre, frame2_cs, frame2_origin, frame2_axes
              )
              burn1 = Burn(
                  time_s=t1,
                  dv_inertial_km_s=as_vector3(dv1),
                  dv_script_km_s=as_vector3(components1),
                  slot_index=0,
                  frame=(frame1_cs, frame1_origin, frame1_axes),
              )
              burn2 = Burn(
                  time_s=t2,
                  dv_inertial_km_s=as_vector3(dv2),
                  dv_script_km_s=as_vector3(components2),
                  slot_index=1,
                  frame=(frame2_cs, frame2_origin, frame2_axes),
              )
              candidate = Candidate(
                  burns=(burn1, burn2),
                  intercept_time_s=arrival,
                  minimum_distance_km=solution.endpoint_error_km,
                  distance_at_completion_km=min(
                      config.policy.intercept_radius_km, solution.endpoint_error_km
                  ),
                  # Never assert success by fiat. As in the one-burn branch this is
                  # the Lambert endpoint residual until refinement replaces it.
                  successful=solution.endpoint_error_km <= config.policy.intercept_radius_km,
                  analytical_endpoint_error_km=solution.endpoint_error_km,
                  branch="two-burn-global/" + solution.branch,
                  metadata={
                      "burn_time_1_s": t1,
                      "burn_time_2_s": t2,
                      "arrival_time_s": arrival,
                      "lambert_z": solution.z,
                      "event_refined": False,
                  },
              )
              if not apply_constraints(candidate, scenario, config).valid:
                  continue
              apply_score(candidate, scenario, config.score)
              candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            -candidate.score if candidate.score is not None else candidate.total_dv_km_s,
            candidate.total_dv_km_s,
            candidate.intercept_time_s or math.inf,
            candidate.candidate_id,
        )
    )
    candidates = candidates[: config.search.keep_candidates]
    targets = candidates[: min(config.search.event_refine_top, len(candidates))]
    workers = resolve_workers(config.search.workers)
    events = refine_events(scenario, config, targets, workers)
    for candidate, event in zip(targets, events, strict=True):
        record_event(candidate, event, config)
        apply_score(candidate, scenario, config.score)
    candidates.extend(drop_post_intercept_burns(scenario, config, targets, workers))
    return [candidate for candidate in candidates if candidate.successful]
