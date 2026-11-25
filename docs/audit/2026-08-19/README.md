# Audit snapshot of 2026-08-19 (history, not the current code)

`REPORT.md` is the re-audit that judged OrbitWin 0.3.0 not ready and listed four
P0 defects. The other five files are the evidence it cites, captured at the
moment it was written.

**All four P0s are fixed in `src/orbitwin/`.** Nothing here describes the current
implementation, and the two `.py.snapshot` files are the *defective* versions:

| P0 | Evidence | Where it is fixed now |
|---|---|---|
| `best_candidate` could be a GMAT-rejected candidate | `selection_bug.json` | `pipeline.py` sorts `renderable_candidates` with the same key |
| GMAT-updated values did not trigger a score recompute | `score_stale.txt` | `verify.py` calls `apply_score`; `score_is_stale` refuses a stale one |
| Analytical success alone could emit `team_solution.script` | — | `CandidateStage` reserves that name for `certify` |
| A tangent contact gave no `first_entry_time_s` | `tangent_event.txt` | `test_event_detection.py::test_exact_tangential_contact_is_an_interception` |

## Why the snapshots are renamed

They were `verify.py` and `pipeline.py` at the repository root, next to a package
exporting modules of exactly those names. They are not importable: every import
in them is relative, and `pyproject.toml` packages only `src/`. So they were
dead either way. The hazard was to readers: `rg` finds them first, they look like
the implementation, and they are the version the report is complaining about.
The `.snapshot` suffix removes them from the namespace and says what they are.

Read them only to follow `REPORT.md`'s line numbers, and note that it cites them
under their `src/orbitwin/` paths -- `src/orbitwin/pipeline.py:116-120`, for
instance. Those line numbers address the snapshots, not the files living there
now. Checked: snapshot `pipeline.py:116` is the `candidates.sort(key=_sort_key)`
that leaves `renderable_candidates` stale, and line 118 is the `best = next(...)`
that then reads it; snapshot `verify.py:68-90` is the block that overwrites
`intercept_time_s`, `minimum_distance_km` and `successful` from the GMAT run
without recomputing the score. Open the live files at those numbers and you land
somewhere unrelated.
