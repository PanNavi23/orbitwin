# Competition policy encoded by OrbitWin

## Source-derived rules

This document follows the uploaded **2026/05/20 preliminary rules**, the May 10 briefing deck, and the **ranking rules of 2026/10/06 as revised on 2026/11/04** (`rules.py`, `RANKING`), which `policy.rules: ranking` selects and every ranking profile sets. A clause is the preliminary's unless its sentence names the ranking rules. It does not replace later organizer instructions.

The solver encodes:

- Spacecraft A is passive and flies under the external forces the template's
  force model configures: gravity only in the preliminary, every external
  force of §7 under the ranking rules' rule 2 as corrected on 2026-11-01.
- The craft may start on different epochs (the organizer's tutorial of
  2026-11-01). The template's epoch alignment, the first mission command, is
  kept byte for byte (head-start burns, signed off below, go before it); the
  solver's t = 0 is its end, where GMAT's own run of it
  leaves both craft. T_team, T_max and Ct run from the earlier of the two
  epochs: the organizer's written answer to Team 54 of 2026-11-04 18:17,
  「從比較早開始的開始算」. Under the ranking rules `policy.mission_time_origin`
  is `earliest` and nothing else. `policy.closest_approach_window` must still
  be set whenever the craft start apart: whether the alignment's rows count
  toward d_min is not yet answered (docs/ASYNC_START_AND_INSERTION.md).
  What the choice changes: under `from_origin` a miss's scored distance is the
  smaller of its flight's closest approach and the alignment's
  (`Scenario.alignment_minimum_distance_km`; `score.scored_distance_km`), so
  when the alignment passes close, every miss ties on the distance term and
  the cheapest one, the zero burn, wins. Under `after_alignment` the
  alignment's rows are ignored. Hits score the same either way; only misses
  and the tie-break move. The parser computes the alignment's minimum only
  under `from_origin`, so run inspect once with it to see whether the choice
  matters. A candidate with head-start burns flies its own alignment rows, so
  under `from_origin` its own closest approach to SC_Alien waiting at its epoch
  is counted (`score.HEAD_START_ALIGNMENT_KEY`: two-body in the search, GMAT's
  rows in verify and certify); a head-start burn can aim at that point.
- When SC_Human starts first, ranking §2.2 and the 2026-11-04 answer allow a
  burn before SC_Alien's epoch. Under `policy.head_start_burns` (signed off
  below) an open template's script may burn there; the search places those
  burns at t in [-Δ, 0) on the solver clock beside every plan without them.
- Spacecraft B may perform multiple impulsive maneuvers.
- Each impulse vector norm must be at most **250 m/s** under the ranking rules
  (§2.1); the preliminary allowed **1500 m/s = 1.5 km/s**. Going over the
  250 m/s plus `ε_v = 1e-5 m/s` disqualifies
  rather than deducts (§4 note 1(a)); under `policy.rules: ranking` such a burn
  makes the candidate invalid and is not charged as a ten-point penalty.
  The ε only caps the config: `policy.dv_limit_km_s` may be set up to 250 m/s
  plus ε. The ranking profiles set 0.25 km/s, and an impulse over
  `policy.dv_limit_km_s` by more than a 1e-12 km/s comparison tolerance makes
  the candidate invalid, so under those profiles one in (250, 250.00001] m/s,
  inside the rule, is invalid too. The safe limit of 249 m/s keeps every
  submitted impulse under both.
- Consecutive maneuvers must be separated by at least **100 seconds**.
- The first maneuver may occur at mission start.
- Mission horizon: the preliminary's was `Tmax = 4 * T_A`. The ranking rules (2026/10/06 §3) replace
  it with a `T_max` announced at the start of the contest, entered as
  `policy.horizon_s`. It is when SC_Alien, past perigee, first climbs out to
  the mission boundary; the rules do not say whether its orbit is open. A T_max
  before the perigee of a clearly eccentric SC_Alien (osculating e above 0.05)
  is refused, and one that leaves SC_Alien off an announced boundary
  (`policy.mission_boundary_km`) is warned, both on its two-body coast. A
  refused T_max that was read correctly is a conflict between the announcement
  and §3: stop and ask the organizer. Only their written confirmation lets
  `policy.accept_announced_horizon_before_perigee_s`, which names that one
  T_max, take it, and every output records it.
- Interception succeeds at the **first** time relative position enters the 5 km sphere.
- Relative velocity is unrestricted, in the preliminary and in the ranking round
  (ranking §3: 本次排位賽僅考慮相對位置).
- Total delta-v is the sum of impulse magnitudes over all `N` maneuvers (`§5`, eq. 1;
  ranking §4 eq. (4)).
  The ranking rules cap it at 3 km/s with no tolerance, and over the cap is
  disqualification, not a deduction (§4 note 1(c)): `policy.total_dv_limit_km_s`,
  with our margin in `policy.total_dv_safe_limit_km_s`.
- Completion time is `Tteam = min(tint, 4TA)`, so a mission that never closes
  completes at the horizon rather than at the end of its script (`§4`). Under
  the ranking rules T_team is `tint` for a hit and the announced T_max for a
  miss (eq. (1)), both counted from the earlier epoch (above). The preliminary's
  validator read the completion time off the report's last row
  (docs/LESSONS_2026-09-05.md), so a hit's script ends inside the sphere;
  `rules.py` assumes the same of the ranking
  round's validator, unconfirmed.
- The scored distance is `Δrmin = max(Δr(Tteam), 5)`. For any success that is
  exactly 5 and the distance term is exactly 50, so nothing after the
  interception can change the score. For a failure it is the separation at
  `4TA`, which is worth up to 50 points and only exists in a report that runs
  that far (`§4`, `§5`). The ranking rules score a failure on its **closest
  approach** over `[0, T_max]` instead (eq. 2, 3); `policy.rules: ranking`
  selects that. Under those rules a miss that GMAT has replayed can also be
  verified, chosen as the best candidate and certified when it scores highest.
  Under the preliminary rules only an interception can.
- The ranking rules' distance term was revised on 2026-11-04, first by the
  organizer's email and then in the rules PDF itself
  (`排位賽規則-revised-2026-11-04.pdf`, header `2026/11/04 修訂版`, SHA-256
  `1ded3ebbaebb658477643c0e05391b7ba2bc040ac30dd18f0297beb4dbbc9e7e`, kept outside
  the repository like the earlier rule PDFs). §4 eq. (5) now reads
  `100 / (1 + exp[((Δr_min − 5)/C_r)²])` where it read
  `50·exp(−(Δr_min − 5)/100)`; the time and delta-v terms and the deductions are
  unchanged. A hit still scores exactly 50 (eq. 3). For a miss the term is flat
  at 5 km (zero slope where the old one fell 0.5 points per km), stays at or
  above 49 out to `5 + 0.2·C_r` and is 25 at `5 + 1.048·C_r`, so how much a near
  miss keeps depends on the C_r announced on the day. The preliminary keeps the
  old term (`rules.py`, `DistanceTerm`).
- A violation deducts ten points per violating **maneuver**, not per broken
  constraint, and the total score floors at zero (`§5`). Under the ranking rules
  only adjacent maneuvers closer than 100 s deduct, ten points per violating
  maneuver down to zero (§4 note 1(d)); an impulse over the limit, a total over
  3 km/s and a crossing of the Earth before the interception are mission
  failures that are not ranked (§4 note 1(a)-(c)). Which maneuver of a close
  pair the code charges is a reading, recorded in ASSUMPTIONS_AND_GAPS.md. No
  submission carries one: a gap under 100 s makes the candidate invalid, and
  one under 101 s keeps it from being submitted (Internal safety margin).
- Under the ranking rules only the top four are announced: first and second
  play the gold match, third and fourth the bronze, and every other team is
  佳作 without a place (§6; `rules.py`, `advance=4`). The top eight
  preliminary placings advanced (`§7`).
- Scenario-specific `kt, Ct, kv, Cv` are not guessed; they must be entered with units.
  Under the ranking rules `C_r` is announced with them (`score.cr_km`), and the
  2026/11/04 revision fixes the units: T_team in s, ΔV_team in m/s, Δr_min and
  C_r in km. A ranking score block without `cr_km`, or with `delta_v_unit:
  km/s`, is refused; a preliminary one with `cr_km` is refused too.
- Tie order is actual minimum distance, total delta-v, completion time, then design explanation.
- The organizer validator is authoritative, and unreproducible results may be disqualified.

## Order of authority

1. Instructions given on site, and the organizer's validator.
2. The latest rules on the organizer's site, and the organizer's written answers.
3. The day's official script.
4. This project's policy and config.
5. Mathematical approximations, test cases and AI suggestions.

On a conflict, stop at once; do not reason it away. A conflict with what the organizer
set (items 1 to 3) is theirs to settle: ask them, in writing.

## Executable safeguards

Natural-language policy is mirrored by:

- `constraints.py`: burn magnitude, separation, horizon and success gates.
- `gmat_parser.py`: passive target, synchronized A/B propagation, explicit burn slots.
- `script_renderer.py`: line-preserving, whitelist-only final edits.
- `verify.py`: stock-GMAT diagnostic replay and exact final reruns.
- `score.py`: unit-explicit official formula.

## Internal safety margin

The global defaults (`config.py`, and the preliminary profiles such as
`config/standard.yaml`) use 1.495 km/s per impulse against the preliminary's
1.5, and 101 seconds against 100. The ranking profiles (`config/ranking.yaml`,
`config/ranking-rehearsal.yaml`) use 0.249 km/s per impulse against 0.25,
2.99 km/s in total against 3.0, and 101 seconds against 100. These are not
additional organizer rules. They reduce floating-point and serialization
boundary risk. A candidate outside them but inside the rules is on the formal
frontier: it is archived, but rank does not list it as submittable and certify
refuses it, unless `policy.allow_formal_frontier_submission` is set (default
false, and no profile sets it).

## Unresolved permissions

Until the organizer answers in writing, final scripts must not:

- change initial states or epochs, except SC_Human's epoch under the onsite
  ranking rules (`ranking-onsite`, §4 入軌時刻), which a team sets with
  `orbitwin set-epoch` (docs/DESIGN_VARIABLES.md);
- change A;
- change force models or propagators;
- add or delete maneuver commands in a slotted template, one whose mission
  sequence already has a `Maneuver`: there only the existing burns' Element1-3
  and the Propagate durations move, and the policy guard
  (`assert_final_script_policy`) refuses any change to its burn resources, or
  to its Propagate commands beyond their durations. An open template, one with
  no `Maneuver`, is the exception the rules require: the
  ranking rules let
  SC_Human make 多次瞬時脈衝機動 (§2) and count `N`, 總機動次數, in ΔV_team
  (§4 eq. (4)), and such a template has no other way to make one. The renderer
  adds one `ImpulsiveBurn` and one `Maneuver` per impulse, using any
  `ImpulsiveBurn` the template declares first. The guard holds every burn to the
  pursuer, a declared one to the frame the template gives it, and an added one
  to `policy.synthesized_burn_frame`. Whether a slotted
  template's slot count bounds `多次` is still open (ASSUMPTIONS_AND_GAPS.md);
- add an epoch alignment to a template that lacks one, other than the single line
  `policy.add_epoch_alignment` adds (signed off below);
- add any command before the epoch alignment other than the head-start pairs
  `policy.head_start_burns` adds to an open template (signed off below);
- alter the official ReportFile contract, beyond replacing each team-number
  placeholder in a Filename with `policy.team_number`, which the organizer requires
  (under the ranking rules the official report, ReportFile1 among the reports GMAT
  writes, must then carry it
  unless the organizer has answered in writing to keep its name, recorded as
  `policy.report_filename_as_written`; any other report with no placeholder
  keeps the organizer's name);
- require Python, AI services, cloud services, custom plugins or custom GMAT binaries.

### Console copy of a script that draws: signed off

GmatConsole refuses a script creating an OpenFrames object (R2026a: 58 errors,
exit 1 on the workshop template), and a GroundTrack created before its
ReportFile left it writing no report. The preliminary's template carried both,
and the GMAT GUI segfaults on this workstation. `gmat.console_copy` (default
false; the ranking profiles set it true since the sign-off below) lets certify
and baseline run, when GmatConsole has refused the submitted bytes for those plots alone, a
copy that leaves out only whole lines: each Create of an OpenFrames or
GroundTrack object, each assignment to one, and each Toggle, PenUp, PenDown,
ClearPlot or MarkPoint naming only them. The copy is proven against the
submitted bytes from disk before it runs (`console_copy.prove_console_copy`):
every other line byte for byte, and nothing kept naming a plot. Anything else
gets no copy and the refusal stands. Whether the GMAT GUI takes the plot lines,
which only it reads, is not judged: they are copied verbatim from the official
template, and the policy guard (`console_copy.assert_plot_lines_unchanged`)
proves them identical, the same lines in the same order, so the GUI runs the
upload exactly when it runs the organizer's template. Tests record what
R2026a's GUI, run as a parse oracle, said of edits to those lines, and mark
each it refused that the checker takes.

The submitted script is unchanged by this: it keeps every plot line of the
official script. The report uploaded is the copy's, from this workstation;
the diagnostic sibling, run off the submitted bytes, must write it byte for
byte. `certification.json` records the copy as `console_copy`, binds the
script and that report by SHA-256, and `check_report_rows --certificate`
rebuilds and proves the copy again. Such a run never writes
`team_solution.script`. This is not an organizer permission: the organizer
has not said whether a report may come from a copy less the plots.

Decision: thc1006, 2026-11-06: certify and baseline run the console copy; the
uploaded script keeps the official plots unchanged; the uploaded report comes
from the copy. Confirmed by thc1006 on 2026-11-06 for the PR that adds it (#189).

Signed off by: thc1006 <84045975+thc1006@users.noreply.github.com>   Date: 2026-11-06

### Adding an epoch alignment: signed off

A template whose craft start on different epochs and has no alignment cannot be
flown as written: GMAT aborts a joint Propagate of craft on different epochs
("Spacecraft epochs do not match"), and a split one reports them their whole
head start apart on every row. Nothing licensed adding this line until the sign-off
below; since then the "add an epoch alignment" item of Unresolved permissions allows it,
for open and slotted templates alike. `policy.add_epoch_alignment` (default false) adds exactly one line, the
first mission command but for the head-start pairs signed off below,
`Propagate <P>(<early craft>) {<early craft>.A1ModJulian = <late craft>.A1ModJulian};`,
where `<P>` is the propagator the template's first Propagate flies the early
craft with. The policy guard refuses any other alignment line; burns are added
to an open template as before. The line also changes the uploaded report: it
starts with the alignment's rows, one per propagator step while the later craft
waits, at least Δ / MaxStep of them (1800 on R7). Whether the validator skips those rows
is Q5 of docs/ASYNC_START_AND_INSERTION.md. The default stays false; the ranking profiles (`config/ranking.yaml`,
`config/ranking-rehearsal.yaml`, and `config/ranking-onsite.yaml`, where a
set-epoch copy starts SC_Human later) set it true since the sign-off below. A template that carries its own
alignment, or starts both craft together, is unaffected: the guard refuses an
added line there.

Signed off by: thc1006 <84045975+thc1006@users.noreply.github.com>   Date: 2026-11-03

### Burns in the head start: signed off

When SC_Human starts first, ranking §2.2 lets the first maneuver come at
mission start, and the organizer's written answer of 2026-11-04 puts mission
start at the earlier epoch; so a burn may come before SC_Alien's epoch.
Forbidding it costs points: in the measurement of 2026-11-08
(`output/phase-review/final/headstart/`, two-body, R8's physics with the
first burn barred before L) 0.055 points at L = 2100 s, 0.31 at 2700, 1.08 at
3600 and 3.81 at 4200. On R8 itself, L = 1800 s, GMAT gave head-start burns
+0.0019. GmatConsole R2026a runs the shape below (rc 0).

`policy.head_start_burns` (default false; the ranking profiles set it true
since the sign-off below) allows, on an open template whose pursuer starts
first and whose alignment, its own or the one `policy.add_epoch_alignment`
adds, stops on an epoch goal, these lines before the alignment and nothing
else:

```
Propagate <P>(<pursuer>) {<pursuer>.ElapsedSecs = <s>};
Maneuver <burn>(<pursuer>);
```

one pair per head-start burn, `<P>` the propagator the alignment flies the
pursuer with. The alignment line follows unchanged, then the mission as
before. The parser and the policy guard refuse any other command there, a
burn on SC_Alien, a slotted template, SC_Alien first, an alignment that stops
on ElapsedSecs (a burn before it would carry it past the later epoch), a
segment under `policy.min_propagate_duration_s`, and a last burn less than
that before the later epoch: each burn is strictly before it. The solver's
t = 0 stays the alignment's end; a head-start burn is at t in [-Δ, 0). The
plan flies the pursuer two-body from its probed state at its own epoch, and
at t = 0 adds the alignment's defect, GMAT's aligned state less that two-body
coast unburned (`Scenario.head_start_defect`): it leaves t = 0 from GMAT's
aligned state plus what the head-start burns changed. The defect grows with
Δ: 20.3 km and 26.9 m/s on R8 (Δ = 1800 s), 66.45 km and 69.4 m/s on R9
(3600 s) under the section 7 forces. What the GMAT correction then removes,
its first flight's miss at the planned arrival (`miss_at_arrival_km` of the
first entry in each correction's log): on R9 83-88 km before the defect, 33.7-33.9
km (median 33.76) after it, against 16.33 km for plans without a head start; on
R8 20.67 km against 19.95. All converge in two rounds. A head-start burn's VNB
axes stay on the plan's two-body arc, 18.65 km and 25.5 m/s off GMAT's own row
at R9's burn: taking them from the alignment probe's GMAT rows instead took
R9's median to 39.92 km (2026-11-08, c83fdd9), so it was not kept. The search's
100 km wall is also two-body over the head start; GMAT's verify holds
`policy.minimum_flown_altitude_km` on every row it flies there. Under
`from_origin` the head start's pass by SC_Alien is read on the probe's GMAT rows,
moved by what the burns changed. T_team, T_max and Ct still count from the
earlier epoch; the 249 m/s, 2.99 km/s, 101 s (across the alignment's end too)
and 100 km walls hold over the head start as well.

How the organizer's validator reads a velocity jump in the rows written
before the later epoch is not answered (Q5 of docs/ASYNC_START_AND_INSERTION.md).
So the search keeps every plan without head-start burns, and rank takes a
plan with them only when it beats the best submittable plan without them by
`policy.head_start_min_gain_points` (0.05 in the ranking profiles); `orbitwin
day` then certifies that best plan as `final-03` beside it and uploads the
head-start plan only if its step-7 score keeps that margin. GMAT flies
head-start plans only once coefficients are announced and only when the best
one's two-body score leads the best plan without one by half that margin, 0.025
(`gmat_loop.head_start_flies`). Measured, that two-body lead against the lead
GMAT then gave: R9 1.0854 against 1.1015; R8 0.0032 against 0.0019 (every plan
flown to measure it) and, before the defect, -0.0289 against -0.0166. In these
three, GMAT's lead was observed 0.016, -0.0013 and 0.012 above the two-body one,
so a plan that wins by 0.05 in GMAT would lead by about 0.034 two-body, above
the 0.025 gate. Three points bound nothing: GMAT's best plan of either kind can
be another than two-body's, and the per-plan spread of GMAT over two-body
(+0.04 for most head-start plans, -0.07 to -0.09 for one; R1's plans, with no
head start, -0.26 to +0.05) could move the lead further. The gate errs the
conservative way, keeping a head-start plan out of GMAT rather than uploading
one that does not win; the 0.05 margin itself is held on GMAT's scores. Step
4a's nets, final-01 and final-02, never burn in the head start. Under `closest_approach_window: from_origin` a head-start
miss whose d_min is its own pass by SC_Alien waiting at its epoch makes rank
print a WARNING, which the day stops on until two people accept it. If the
organizer says on the day that no burn may come before the later epoch,
`orbitwin day --no-head-start-burns` (`ranking_day_config.py
--no-head-start-burns`) turns it off, recorded in the readback and beside the
upload.

Decision: thc1006, 2026-11-08, after seeing the measurement above.

Signed off by: thc1006 <84045975+thc1006@users.noreply.github.com>   Date: 2026-11-08

`policy.minimum_flown_altitude_km` refuses, in verify, certify, solve and
rank, a candidate GMAT flies under that altitude, on every row GMAT flies it
in: an alignment it flies, and on to the end of the report. The rules' line, and ours, is the surface: the
global default is None, which keeps it there. The ranking profiles set 100 km,
the ranking search's own floor, as margin against drag differences between
GMAT builds; under the section 7 model GMAT flew R4's margin-0 best to
95.4 km where the two-body plan held 100.

`policy.synthesized_burn_frame` can write the maneuvers we add to an open
template in EarthMJ2000Eq instead of the workshop's Local/Earth/VNB. It is
off by default until the organizer answers Q7 of
docs/SCORING_STRATEGY_AND_OPEN_QUESTIONS.md in writing. On the R4
rehearsal it brought the impulse GMAT flew from 0.431 m/s off the plan to
at most 0.020 m/s. With VNB that error is left to the GMAT correction
(step 3 of GMAT_IN_THE_LOOP.md, `gmat_shoot.py`), which solve now runs
(step 4, `gmat_loop.py`): the first `gmat.correct_top` hits inside the
production margins, 40 in the ranking profiles, have their last impulse
corrected on GMAT's own miss (the last two share the correction when the last
alone would break a safe cap), so for those the frame error is flown out. A
miss is replayed as planned and keeps it.
