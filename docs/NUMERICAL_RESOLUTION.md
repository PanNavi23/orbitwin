# Numerical resolution of the candidate ranking

Measured 2026-08-27 with `scripts/bit_probe.py` and the per-branch solves
recorded in `docs/STACK_ANSWER_MAP.md`.

## What was being checked

`docs/STACK_ANSWER_MAP.md` found ten pull requests that move the candidate set.
One of them, `#13 perf/event-and-derivative-hot-paths`, is titled *"Cut event
detection and propagation cost without changing any answer"* and its
`_two_body_pair_derivative` docstring states that "the expression order is kept
identical to the array form, so results are bit-for-bit unchanged".

That claim is invisible to the test suite. Every numerical assertion in it
compares with a tolerance, so a rewrite can move the last bits of every
derivative evaluation and stay green.

## #13 moves the bits, and it is the only layer that does

`scripts/bit_probe.py` evaluates `_two_body_pair_derivative` on 4,000
deterministic LEO-to-GEO pair states and dumps the exact bits:

| Comparison | Doubles differing | ULP distance |
| --- | --- | --- |
| `main` → #12 head | **0 of 48,000** | — |
| #12 head → #13 head | **2,538 of 48,000 (5.29%)** | min 1, max 7, mean 3.06 |

Layers #1 through #12 never touch the derivative. #13 does, so the docstring
claim is false as written.

The cause is the rewrite of `np.linalg.norm(position)` into
`math.sqrt(x*x + y*y + z*z)`. These are not the same operation: numpy reaches
BLAS, which is free to use a different summation order and fused multiply-add,
while the Python expression is strictly left-associative with no FMA. One ULP
is enough, because the value feeds the adaptive step controller in
`scipy.integrate.solve_ivp` -- a last-bit change moves which steps are accepted,
and the whole trajectory follows.

## The screening bound in #13 is sound

The same layer adds a per-segment screening bound to `evaluate_hermite_samples`
so segments that provably cannot approach the sphere skip root finding. That is
the change with real potential to lose a crossing. It does not:

| Metric | Result |
| --- | --- |
| Paths probed | 1,500 |
| Paths that entered the 5 km sphere | 509 |
| `first_entry_time_s` differing | **0** |
| `minimum_distance_km` differing | **0** |
| `minimum_distance_time_s` differing | **0** |
| `samples_inside_radius` differing | **0** |

Bit-identical to the exhaustive path on every case. The bound is safe; the
docstring is what is wrong.

## Downstream effect, measured

| Field | Largest change #12 → #13 | Relative to its own gate |
| --- | --- | --- |
| `intercept_time_s` | 1.8e-9 s | 5.4e-14 |
| `minimum_distance_km` | 2.9e-9 km | 5.8e-10 of the 5 km radius |
| `distance_at_completion_km` | 6.7e-11 km | 4.4e-15 |
| `total_dv_km_s` | **0** | — |
| `score` | **0** | — |

Solving three times on #13's tree gives a bit-identical candidate set each time,
so the difference is systematic, not run-to-run noise.

## Why the noise cannot reorder the ranking

A 2.9e-9 km perturbation is alarming next to `docs/COMPETITION_POLICY.md`,
which ranks ties by "actual minimum distance, total delta-v, completion time".
Read as policy, minimum distance is the first tie-break, and candidates in this
set sit 1.2e-11 km apart. Read as code, it does not work that way, and the
distinction matters enough to record.

`pipeline._sort_key` orders by evidence tier first, then:

- with no score configured, by `total_dv_km_s`, then measured minimum
  distance, then intercept time, then `candidate_id`;
- with a score configured, by `score.ranking_key`: score, then measured
  minimum distance, then delta-v, then intercept time, then `candidate_id`.

`config/quick.yaml` carries no `kt/Ct/kv/Cv`, so every candidate scores `None`
and **total delta-v is the primary key**, not minimum distance. Adjacent
delta-v values are 6.2e-3 km/s apart -- nine orders of magnitude above the
noise -- and #13 does not move delta-v at all.

Two further guards make the scored path no more sensitive:

1. `score.measured_minimum_distance_km`, added in #7, returns infinity for any
   candidate that has not been event-refined, "because before event refinement
   `minimum_distance_km` holds the Lambert endpoint residual, which is around
   1e-10 km for any converged arc". The failure mode of a residual winning a
   distance tie-break was found and closed before this measurement.
2. `score_candidate` computes `distance_score` from
   `max(distance_at_completion, 5.0)`, so it saturates at 50.0 for every
   successful intercept. Score is a function of completion time and delta-v
   only; a score tie *is* a time-and-delta-v tie.

Measured against the 25-candidate set rather than argued: there are twelve
groups of candidates that tie on delta-v, and **all twelve are decided by
`candidate_id`** -- each pair ties on delta-v, on minimum distance to the last
bit, and on intercept time. Minimum distance never breaks a tie between two
distinct finite values anywhere in this set, so there is nothing for a 2.9e-9 km
perturbation to reorder. That is a property of the key ordering, not luck.

The 1.2e-11 km figure is real but belongs to candidates in *different* delta-v
groups, where the comparison was already settled before minimum distance was
read.

## What the twelve tied pairs do show

Twelve of twenty-five candidates are duplicates of another candidate on every
ranked field. They are distinguished only by a content hash. That wastes half
of any `render_top` or `gmat.verify_top` budget on trajectories already
represented -- which is what #31, "Spend the GMAT budget on distinct
trajectories first", exists to fix. No new defect here either.

## The other nine

Re-diffing all ten `ANSWER-CHANGED` layers field by field, against each one's
own base:

| Layer | What actually changed | Numeric effect |
| --- | --- | --- |
| #4 | `metadata` gains `event_refined`, `event_samples_inside_radius` | none |
| #8 | records gain `stage`, `score_is_stale`, `stage_evidence` | none |
| #15 | `metadata` gains `frontier` | none |
| #22 | 3 of 25 records move `analytical` → `renderable`, warnings cleared | none |
| #24 | `metadata` gains `minimum_pursuer_radius_km` | none |
| #36 | same field, 1 of 25 records | none |
| #13 | floating-point noise, above | 5.8e-10 of the gate |
| #11 | 13 → 12 candidates | real, matches its title |
| #12 | 12 → 25 candidates | real, matches its title |
| #47 | 25 → 126 candidates | real, matches its title |

For #11, #12 and #47 the **best** candidate is bit-identical across the change
on every physical field -- delta-v, intercept time, minimum distance, burn
vectors, branch. Only the derived `candidate_id` hash moves. These layers widen
the search without changing what would be submitted.

## Reproducing

```bash
git worktree add --detach /tmp/a origin/<base-branch>
git worktree add --detach /tmp/b origin/<head-branch>
( cd /tmp/a && PYTHONPATH=src python scripts/bit_probe.py /tmp/a.json )
( cd /tmp/b && PYTHONPATH=src python scripts/bit_probe.py /tmp/b.json )
python scripts/bit_probe.py --compare /tmp/a.json /tmp/b.json
```

Exit status is 1 when anything differs, so it can gate a layer that claims not
to.
