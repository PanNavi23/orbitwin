# Architecture

```text
immutable official.script ──> GMAT state probe (epochs and states, any frame)
        |
        v
fail-closed parser ──> scenario (solver clock, epoch alignment, T_max checks)
        |
        v
3-D two-body seeds
  - zero burn (flown in GMAT)
  - zero/multi-rev Lambert
  - two-burn split seeds and global Sobol fallback
  - ranking_search: capped multi-impulse, score-driven, hard walls
        |
        v
constraint + Pareto + official score layer
        |
        v
line-preserving renderer (each candidate GMAT flies is rendered first)
        |
        v
GMAT correction of the top hits (gmat_loop, gmat_shoot) + GMAT verification
  - stock GmatConsole, isolated process and output directory
  - diagnostic sibling (never submitted), fresh report enforcement
  - Hermite first-entry / minimum-distance verifier
        |
        v
rank (re-score, ceilings, levers, best_submittable)
        |
        v
line-preserving renderer + policy guard
        |
        v
certify: exact final Script reruns + official reports + hashes, then PRE-CHECK
  - no plots: exact runs, certification.json and the submission (script + report)
  - plots: exact-run-01 refused for them alone, the exact runs fly the proven
    plot-free copy (console_copy, gmat.console_copy); the report is the copy's
    and the script keeps its plots
        |
        v
check_report_rows --config --certificate on the submitted report
  (a report from a GMAT GUI: --script --verification instead)
```

## Trust boundaries

- The organizer's material, `input/original/` and `vendor/gmat/` are immutable inputs.
- The analytical engine may be wrong when the GMAT model is richer; it never confers final status.
- The diagnostic sibling is allowed to add ReportFiles, and a `Report` command on each side of every `Maneuver`, only to a private verification copy. Verification refuses a run whose velocity jump at a Maneuver is not the impulse the script's Element1-3 claim. The submission script is generated separately and policy checked. Its name follows its evidence tier: `public_gmat_candidate.script` on the day, `team_solution.script` only for a GMAT declared and fingerprinted as the organizer's (`models.STAGE_SCRIPT_NAME`).
- Every GmatConsole invocation uses an isolated output directory to prevent stale or cross-candidate reports.

## Parser boundary

The parser intentionally supports a narrow contest-oriented subset:

- explicit `Create` commands;
- one assignment per object field;
- exactly two spacecraft;
- explicit `ImpulsiveBurn.Element1/2/3`, and no ImpulsiveBurn field outside `gmat_parser.BURN_FIELDS`; GMAT's `V`/`N`/`B` aliases for the Elements are refused on every burn that becomes a slot, and left to GMAT on one never maneuvered;
- single-line `Propagate` commands with elapsed-time goals, synchronized or split with equal goals on every craft, plus the template's own epoch alignment (an A1ModJulian goal) as the first command;
- for a slotted template, an alternating `Propagate/Maneuver` sequence; an open template has no `Maneuver`, and its maneuvers are added (COMPETITION_POLICY.md, Unresolved permissions);
- a `Toggle`, `PenUp` or `PenDown` of plot subscribers alone, or a `MarkPoint` or `ClearPlot` of XYPlots alone, anywhere in the mission sequence (`gmat_parser.plot_command`), in the forms R2026a's GUI runs to the end; it changes no flight or report, and the private copies and the console copy leave it out with its plots. One of anything else is refused.

Unknown or ambiguous syntax fails closed instead of being silently rewritten.
