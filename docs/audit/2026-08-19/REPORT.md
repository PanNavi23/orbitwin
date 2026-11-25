# OrbitWin 0.3.0 re-audit, 2026-08-19

## Revised release status

- Archive integrity: PASS.
- Research-prototype analytical tests: PASS within the tested scope.
- Real organizer-compatible GMAT scientific validation: NOT EXECUTED.
- Competition operational readiness: FAIL / WITHDRAWN.

## Reproduced P0 defects

1. `src/orbitwin/pipeline.py:116-120`: after GMAT verification only `candidates` is re-sorted; `renderable_candidates` remains stale, and the selected `best_candidate` can be a GMAT-rejected candidate. See `selection_bug.json`.
2. `src/orbitwin/verify.py:68-90`: GMAT-updated intercept and distance values do not trigger recomputation of score. See `score_stale.txt`.
3. `src/orbitwin/pipeline.py:120-135`: analytical success alone may emit `best_candidate/team_solution.script`; real GMAT verification and certification are not required.
4. `src/orbitwin/event_detection.py:106-123`: a shallow tangent contact can have minimum distance exactly equal to the radius but no `first_entry_time_s`. See `tangent_event.txt`.

## Material limitations

- One-burn search uses only burn slot 0.
- Default search discards the formally legal 1.495–1.500 km/s region.
- Two-burn search enforces 101 s rather than retaining a separate formal 100 s frontier.
- Burn-time search excludes the final 10–20% of the legal window by configuration.
- Pareto pruning can happen before true first-entry refinement.
- Lambert fuzz replays the same implementation; it does not independently prove branch completeness.
- The parser generalizes only within a deliberately narrow competition-template family.
- The diagnostic sibling proves a modified script, while exact final reruns only prove official report reproducibility; exact-final interception parity remains to be demonstrated with real GMAT.

## Supported claim

The ZIP is intact and useful as an audit/research prototype. It must not be used as the competition-day solver until the P0 defects are fixed and a real organizer-compatible GMAT and validator differential campaign passes.
