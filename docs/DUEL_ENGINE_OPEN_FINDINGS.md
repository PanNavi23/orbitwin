# Duel engine design: open findings

Status: 2026-11-02. `DUEL_ENGINE.md` went through fifteen adversarial review
rounds. Rounds 1 to 14 were fixed in the document as they came. With the
finals expected on 2026-11-09, the design review stopped after round 15,
which found no High finding, and its Medium and Low findings were listed
here. They have since been fixed in the design. A sixteenth review, of
only those fixes, found no High finding and three Medium findings, which
are fixed; its Low findings are listed below.

Each entry names the design section it touches and the phase that
implements it. An entry is closed by a commit that changes the design (or
the code, where the phase is already under way) and adds the test named.

## Closed

Round 15, each closed by the commit named:

- M1, the lock's `/proc` reads and the audit record file on the audit
  allowlist (sections 5.6, 6.2): f5728d9.
- L1, one rounding share, √3/2 of the resolution per vector (sections 2.1,
  2.5, 6.4): 6cf555b.
- L2, a crashed child falls back to the best incumbent with a matching
  input hash (section 6.2): 59d3400.
- L3, the same 1e-6 km floor band in the solver and the parent, and a
  verdict disagreement is a mismatch (sections 2.4, 6.2): f276a4d.
- L4, the deadline anchored to the round in `round_<k>_u.json` (sections
  5.8, 6.2): b98c34a.
- L5, the greedy witness gets the window filter's dwell rule and pair bound
  (sections 2.6, 5.3): 1d1563e.
- L6, counted events timed by the qualifying passage, capped at the round
  end (sections 2.4, 2.6, 3.2, 3.4, 5.3): 2646e16.
- L7, the method offset and the drift add in `a_margin_floor_km` (section
  6.4): 359941a.
- L8, the round-boundary capture stated as an assumption about the
  validator and asked in Q6 (sections 2.6, 8, 9, 10): c758b56.

Clarifications made in bec3cc7, not from a review: E = 0 in
`a_margin_floor_km` as pursuer; each `choose-role` search measures its
deadline from its own anchor; a configured deadline at or above
`DUEL.decision_s` is refused. That commit's message calls them the
findings of a review of the round-15 fixes. They were not: that review's
result had not arrived when the commit was made.

Round 16, the review of the round-15 fixes, closed in the commit that
follows bec3cc7:

- The round-boundary reading favours us in an open question (Q6) and had
  no switch and no flag. It is now `boundary_rows_judged` (default true),
  covers both Δr and each craft's radius, is flagged when a recommended
  action relies on it, is shown off in the role choice, and is the fourth
  documented exception (sections 2.0, 2.6, 6.5, 8, 10).
- The deadline anchor came from the first `recommend`, so time lost to a
  refused round file did not count, an organizer restart of the clock had
  no remedy, a run past `decision_s` printed nothing, and a past-deadline
  restart with no matching incumbent printed an unsafe zero. The anchor
  now comes from the first command that touches the round, with
  `--window-opened` and `--reanchor`; past the deadline the parent
  computes the rescue itself, capped at 20 s, and a run past the window
  still prints, with a warning (sections 5.8, 6.2).
- The "0.3 + 0.2 = 0.5 km" floor test also passed on the old
  largest-of-three rule as evader when M and D share a sign. It now runs
  as pursuer and as evader with opposite signs (section 7, phase 4).

## Low

### L9. 0.0087 m/s and 238 m do not match

- Section: 2.5.
- Phase: docs.
- Finding: the impulse-rounding example says an error "of up to 0.0087
  m/s" moves a by 238 m at 42164 km. With 0.0087 it is 238.6 m, so 239 m;
  238 m holds only for the exact share (√3/2)·0.01 = 0.00866 m/s (237.5 m).
- Fix: write the share as (√3/2)·0.01 m/s there, or keep 239 m.

### L10. The greedy witness's dwell rule is incomplete and its bias unstated

- Section: 5.3 item 1, 2.6, 7 (phase 3).
- Phase: 3.
- Finding: under the pursuer convention item 1 omits the rule of section
  2.6 that a dip holding at the round end counts (now under
  `boundary_rows_judged`); the omission is against us, so it is only an
  inconsistency. Holding the pursuer's part of the pair bound at its
  next-round value rejects more witnesses, raising T and leaning towards
  the pursuer, that is us; the text calls it heuristic without naming the
  direction. The phase-3 test says "any dip below R_f" where the floor
  rule of section 2.4 (R_f + 1e-6 km) is meant.
- Fix: add the round-end clause, state the direction, word the test by the
  floor rule; or bound later rounds' r_max in closed form from the largest
  apoapsis reachable with k·U.

### L11. The floor band can still split the two verdicts at its edge

- Section: 2.4, 6.2.
- Phase: 4.
- Finding: with one rule on both sides, a radius within the batch and
  scalar difference of R_f + 1e-6 km can still be judged differently and
  start the walk-back. No library action is optimised against the floor
  (the rescue maximises the minimum radius), so it should be rare.
- Fix: if a floor-hugging optimiser is ever added, have the solver call an
  action safe only above R_f + 2e-6 km (hysteresis).

### L12. The audit's access modes should be judged by flags

- Section: 5.6.
- Phase: 3 and 4.
- Finding: measured on CPython 3.11 to 3.13, built-in `open`, `read_text`
  and `read_bytes` report the `open` event as `(path, 'r', flags)`, also
  for `rb`, and `os.open` reports mode `None`, so `rb` never appears. A
  hook that opens its record file itself re-enters itself. The test must
  also say that the read ban on the record file applies to the `recommend`
  parent and child, not to the test that reads it afterwards.
- Fix: judge both opens by flags (read-only: `flags & O_ACCMODE ==
  O_RDONLY` without `O_CREAT` or `O_TRUNC`; append: `O_APPEND`), open the
  record file's descriptor before installing the hook or guard against
  re-entry, and scope the read ban to the parent and child.
