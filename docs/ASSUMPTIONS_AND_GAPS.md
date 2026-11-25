# Assumptions and blocking gaps

## Assumptions used only for analytical seeds

1. The central-body gravitational parameter is Earth `398600.4418 km^3/s^2`.
2. Initial states and epochs come from GMAT itself, through the state probe, in any
   coordinate system and state type GMAT reads (EarthFixed SphericalAZFPA
   included). Without a console only `EarthMJ2000Eq` Keplerian or Cartesian
   states on one epoch text are read in-process. When the craft start apart the
   solver's t = 0 is the end of the template's epoch alignment, and both craft
   start from where GMAT's own run of it leaves them
   (docs/ASYNC_START_AND_INSERTION.md).
3. The two-body model is a seed generator. Non-spherical gravity, drag, SRP and other configured effects remain the responsibility of GMAT refinement/replay.
4. In a slotted template, one whose mission sequence has a `Maneuver`, the existing `ImpulsiveBurn` resources and alternating `Propagate/Maneuver` slots are the legal edit surface. An open template, one with no `Maneuver`, gets one `ImpulsiveBurn` and one `Maneuver` per impulse added, using any `ImpulsiveBurn` it declares first, as COMPETITION_POLICY.md (Unresolved permissions) requires. In both, the team number goes into the official report's Filename, and when the craft start apart without an alignment, the single alignment line is added (COMPETITION_POLICY.md).
5. Unused official burn slots may be assigned an exact zero vector and retained. Obtain organizer confirmation; if zero-vector maneuvers are disallowed, the renderer policy must be changed only after written permission.

## Reading of the rules that the code encodes

The preliminary rules use two different symbols for two different distances and
the project treats them as different quantities.

`§4` defines `Δrmin = max(Δr(Tteam), 5)`, the relative distance **at the
completion time**, floored at 5 km. For any successful interception `tint` is by
definition the first instant with `Δr ≤ 5`, so `Δrmin` is exactly 5 and the
score's distance term is exactly 50. That is what `distance_at_completion_km`
holds and what `score_candidate` uses under the preliminary rules. Under the
ranking rules `score_candidate` prices the closest approach over [0, T_max]
instead: `minimum_distance_km`, with the alignment's rows when
`closest_approach_window` is `from_origin` (eq. (2)-(3);
`score.scored_distance_km`).

`§6` breaks ties on `最小相對距離 dmin,team`. Read as `§4`'s `Δrmin` that would
be 5 for every successful team and the tie-break would do nothing, so it is read
as the actual closest approach over the trajectory. That is what
`minimum_distance_km` holds and what `ranking_key` uses. In the preliminary the
reading was an inference from the two symbols being distinct and from the
tie-break being useful. The ranking rules state it: eq. (2) defines
`dmin,team = min Δr(t)` over `0 ≤ t ≤ Tmax`, eq. (3) floors it at 5 for the
distance term, and §5 breaks ties on `dmin,team` first.

Also from `§5`: the deduction is `每次違規機動扣 10 分` -- ten points per
violating *maneuver*, not per broken constraint. A burn that is both over the
1500 m/s limit and closer than 100 s to its predecessor is one violating
maneuver. `candidate.penalties` therefore carries one entry per maneuver, each
listing that maneuver's reasons. Under the ranking rules an impulse over the
limit is not a deduction but a mission failure (§4 note 1(a)), so only the
gap is charged there (§4 note 1(d)).

**Which maneuver of a close pair is charged.** Neither rulebook says.
Ranking §4 note 1(d) reads `若每組相鄰機動時間 ∆t < 100 秒，每次違規機動扣 10 分`:
the violation belongs to a pair of adjacent maneuvers, and the deduction to
each violating maneuver. The code charges the later burn of the pair, the one
placed too close (`constraints.validate_candidate`), so a pair costs 10 points.
A reading that charges both burns would cost 20. The reading cannot reach an
upload: a gap under 100 s makes the candidate invalid, so verify, certify and
solve's final gate refuse it, and a gap under `policy.safe_burn_separation_s`
(101 s in every profile) puts it on the formal frontier, which rank, solve
and certify withhold unless `policy.allow_formal_frontier_submission` is set
(default false). The reading changes only the score rank prints for such a
row read from a file; the searches drop an invalid candidate and never keep one.

## Measured limits of the official template

A maneuver at exactly mission start is permitted by both rulebooks (ranking
§2.2), but in a template whose craft share an epoch the fixed
`Propagate`/`Maneuver` command order cannot express it. Placing a burn
at `t = 0` requires a first `Propagate` with `ElapsedSecs = 0`, and GMAT R2026a
does not reject that goal -- it hangs. Two 60-second runs never reached the
mission sequence, while `ElapsedSecs = 0.001` completed in 1 second. A submitted
script of that shape would stall the organizer's validator rather than fail
cleanly, so `policy.min_propagate_duration_s` (default 1e-3 s) is enforced by the
renderer, by the final-script policy guard, and by the search. The floor is an
observation about that build, not a rule, and should be re-measured against the
organizer's GMAT.

After an epoch alignment a burn can fire at the alignment's end, the solver's
`t = 0`, with no `Propagate` before it (docs/ASYNC_START_AND_INSERTION.md §2).
An open template may place its first burn there
(`models.burn_may_follow_alignment`); a slotted template whose first `Maneuver`
follows the alignment directly fires slot 0 there, and the parser records the
implicit zero-length segment in front of it. That instant is the later craft's
epoch, not the mission start the earlier epoch defines; whether a burn may come
at the later craft's epoch is one of the open questions below.

## What the briefing deck answers

The deck was previously recorded here as unreadable: its CJK font carries no
ToUnicode map, so `pdftotext` recovers the ASCII and nothing else. Rendering it
at 300 dpi and running `tesseract -l chi_tra` recovers all eleven pages, and the
sentences quoted below were then checked character by character against the
rendered page images.

- **The submission is the Script and the Report, both** (p.8): `將你所設計的
  Script 及產生的 Report 上傳至規定網址`, preceded by `請將你的 Script 存檔,並至少
  模擬一次,以產生 Report`. The uploaded report is the evidence, not a supporting
  artifact. So a submitted miss propagates to the end of the window, where its
  distance is scored: `4TA` in the preliminary, whose `§4` scores `Δr(4TA)`, and
  the announced T_max under the ranking rules, whose eq. (2) scores the closest
  approach over `[0, T_max]`. A submitted hit stops where GMAT measured it
  first `policy.stop_depth_km` inside the sphere (`policy.mission_end_policy:
  interception`, the default), because on 2026-09-05 the validator took the
  completion time from the last row of a report that ran to `4TA` (docs/LESSONS_2026-09-05.md).
- **Spacecraft A is circular in the preliminary** (p.8): `在初賽中,Spacecraft A
  將以一圓軌道繞行地球`. `TA` is therefore unambiguous up to the force model.
- **The deck says commands are issued and simulated on the organizer's computer** (p.10):
  `競賽過程中對太空船的指令下達與模擬,需使用主辦/承辦單位所準備的電腦。但可自備
  電腦 / 筆電做先期模擬與運算`. Private preliminary computation is explicitly
  allowed. This is the source of the rule that no submitted artifact may depend
  on Python, an AI service, a cloud service or a custom GMAT build. For the ranking
  round the user decided on 2026-11-06 that the uploaded script and the Team54
  report are produced on our workstation, as in the preliminary (docs/RUNBOOK.md, Who runs what).
- The scoring parameters are announced separately (p.8): `詳細競賽與計分規則擬定後,
  將公告於 TASA 官網上`.

The organizer's workshop template has the shape the parser must accept: two
`ImpulsiveBurn` slots, three `Propagate` segments, alternating,
`Local`/`Earth`/`VNB` axes and `DecrementMass = false`. It also mixes both
`Propagate P(A, B)` and `Propagate P(A) P(B)` in one file, and the parser
accepts both. Our `examples/two_burn_rendezvous.script` has the same shape, and
the tests parse it in the template's place.

The second form does not by itself keep A and B on one epoch. Without
`Synchronized`, GMAT stops every craft at the first goal reached, wherever its
own steps take it. Measured on R2026a, the workshop template's split Propagate
ended with both craft 0 s apart, because its `MaxStep = 1` held both steppers in
lockstep; the same form with adaptive 60 s steps ended 167 s apart. The parser records it as `equal-goals`
and refuses a split line that lacks both the keyword and a goal on every craft.
The GMAT replay checks it: the diagnostic report records both craft's
ElapsedSecs, and a row where they differ by more than 1e-6 s rejects the
candidate. When the craft start apart the check is on both craft's
A1ModJulian instead, and only the alignment's leading rows may differ
(`report_parser.py`). On R2026a the form above was rejected at 167.125 s apart and the
Synchronized form accepted.

## Blocking gaps before official competition use

Ordered by how much of the score depends on the answer. `docs/SCORING_STRATEGY_AND_OPEN_QUESTIONS.md`
states each one as a question with the consequence of either answer.

- **With the craft on different epochs** (docs/ASYNC_START_AND_INSERTION.md, Q1-Q6): T_team, T_max and Ct count from the earlier epoch, answered in writing on 2026-11-04 18:17 (「從比較早開始的開始算」) and encoded as `policy.mission_time_origin: earliest`; whether d_min's window includes the alignment's rows (`policy.closest_approach_window`, required and a guess until answered); whether the template carries the epoch alignment or teams add it (`policy.add_epoch_alignment`: default false, set in the ranking profiles since the 2026-11-03 sign-off in COMPETITION_POLICY.md); whether the first burn may be at the later craft's epoch itself; and which report column the validator reads as time, whether it skips the rows written before the later craft's epoch, and how it reads a head-start burn's velocity jump in them (`policy.head_start_burns`, signed off 2026-11-08; rank holds such a plan to `policy.head_start_min_gain_points`).
- **Which columns the official `ReportFile` carries.** The organizer's own
  example writes `{HumanSC.A1ModJulian, HumanSC.EarthMJ2000Eq.X}` -- an epoch and
  one coordinate of the pursuer, with nothing about the target. No relative
  distance can be computed from that, so either the competition-day script
  differs, or contestants are expected to add the columns, or the validator
  re-propagates from the script and ignores the report. The three answers imply
  three different renderer policies.
- Exact values **and units** of `kt, Ct, kv, Cv`. A `Cv` in km/s rather than m/s
  saturates the delta-v term into a constant. For the ranking round the
  2026/11/04 revision answers the units (T_team in s, ΔV_team in m/s, Δr_min and
  the new `C_r` in km); the values still come on the day.
- Whether the validator tests report rows or interpolates between them, and the
  report sampling interval of the official script. This decides whether a
  grazing approach counts as an interception at all.
- Whether all burns in the complete Script or only burns up to first interception
  count toward total delta-v, and whether an exact-zero maneuver counts as a
  maneuver for the 100 s separation rule.
- Preliminary only: whether a failed mission receives the literal continuous
  score or is ranked separately. The two answers make opposite strategies
  correct: under the first, the zero-burn baseline is the right submission when
  interception is impossible. The ranking rules answer it: a miss is scored by
  the formula, with T_team = T_max and the distance of its closest approach
  (eq. (1)-(3)), and ranked; only §4 note 1(a)-(c) failures are not. The best
  miss is the highest-scoring one, not necessarily the zero burn: on R2 at
  C_r 3000 a re-searched miss scored 45.82 against the zero burn's 31.54.
- Whether `Propagate` may run to exactly `4TA` (preliminary) or to exactly the
  announced T_max (ranking, where a scored miss ends there), and any size limit
  on the uploaded report.
- Preliminary only: how `TA` is defined if the force model is not point-mass,
  and which `mu` it uses. The ranking rules have no `4TA`: T_max is announced
  on the day (§3).
- Exact organizer GMAT version, build, OS and startup file, including gravity,
  leap-second and EOP file versions.
- Exact allowed Script fields and whether commands may be added or deleted --
  in particular whether `多次` maneuvers is bounded by the slot count.
- Floating-point tolerance and score rounding used by the validator. For
  impulses the ranking rules answer it: ε_v = 1e-5 m/s (§4 note 1(a)).
- Whether AI assistance is permitted for the private preliminary computation that
  p.10 allows.

## Release status meaning

A candidate's stage (`models.CandidateStage`) names the evidence behind it, and
its script is named for that stage (`models.STAGE_SCRIPT_NAME`):

- `analytical` / `renderable`: solved and self-replayed under two-body dynamics
  (`analytical_candidate.script`).
- `public_gmat_verified`: a successful run under a stock GmatConsole that the
  config has not declared as the organizer's (`public_gmat_candidate.script`).
  This is the stage the competition-day path reaches: `config/ranking.yaml`
  leaves `gmat.organizer_compatible` false and `ranking_day_config.py` never
  sets it, so certify, when it completes, writes `public_gmat_candidate.script`
  (docs/RUNBOOK.md appendix C). On a template with plots, once GmatConsole has
  refused the script for its plots alone at exact-run-01, certify flies the
  plot-free `gmat.console_copy` (set in the ranking profiles, signed off on
  2026-11-06) and writes the submission, plots kept in the uploaded script
  (docs/RUNBOOK.md step 6c). With it off, or when no copy can be proven, certify stops
  there and the report has to come from a GMAT GUI.
- `organizer_gmat_verified`: the same, under a build declared with
  `gmat.organizer_compatible` and matching `gmat.organizer_executable_sha256`
  (and `gmat.organizer_startup_sha256` when set), which `config.py` says to set
  only once the organizer has confirmed the build, startup file and data files
  (`organizer_gmat_candidate.script`).
- `certified`: an organizer-verified run plus repeated exact final Script runs
  and fresh official reports (`team_solution.script`).

`fake-GMAT integration` is not a stage: it describes orchestration tested
against the deterministic test double (`scripts/fake_gmat_console.py`).

The packaged release contains no claim that fake-GMAT evidence is scientific validation.
