# GMAT in the loop: design

Status: proposal, 2026-10-31. Steps 1 to 4 are implemented (2026-11-01);
see their sections.

## The problem

The search plans transfers in two-body dynamics and GMAT checks the result.
That worked in the preliminary, whose force model was point mass. The
ranking rules add EGM96 20x20, tides, drag, Sun and Moon, SRP and
relativity (section 7). Measured on R2026a against the two-body model:

| | after 30 min | after 2 h |
|---|---:|---:|
| SC_Human, LEO, full model | 20 km | 98 km |
| SC_Alien, gravity only | 4.5 km | 18 km |

The interception sphere is 5 km. Today GMAT can only accept or reject what
the search proposes, so under the ranking model it would reject nearly every
analytical hit. Nothing in the pipeline turns a near hit into a hit.

## What was measured before writing this

All on GMAT R2026a, SC_Human on the full ranking model, SC_Alien on a
gravity-only propagator and a hyperbola (e = 1.5), one `Synchronized`
Propagate, 2700 s mission, one impulse at 300 s.

- One GmatConsole run, start to finish: 0.95 s. Target alone: 0.63 s.
- Concurrent runs: 1 run in 0.96 s, 4 in 0.96 s, 8 in 1.35 s, 16 in
  1.83 s, about 8.7 runs per second at 16.
- Newton iteration on the impulse, with the residual taken from GMAT and the
  sensitivity d r(t_f) / d v(t_b) taken from the two-body model:

  | iteration | miss | impulse |
  |---:|---:|---:|
  | 0 (two-body Lambert) | 18.565 km | 4667.473 m/s |
  | 1 | 9.5 m | 4672.444 m/s |
  | 2 | 6 mm | 4672.441 m/s |

  The two-body sensitivity is close enough to GMAT's that two corrections
  reach millimetres, and the correction itself was 5 m/s.

## Design

### 1. Target ephemeris from GMAT

SC_Alien does not maneuver, so its trajectory does not depend on anything
we choose. Run the organizer's script once with no burns and a report of
the target's state every few seconds over [0, T_max], and interpolate it
(cubic Hermite on position and velocity, the same method the verifier
already uses). The search then aims at where GMAT says the target is,
which removes the target's share of the error (18 km over 2 h above) for
the cost of one run.

Implemented in `src/orbitwin/ephemeris.py`. On the ranking force model with
60 s report rows, the interpolated target was within 0.067 m of GMAT stopped
exactly at 1234.5, 3333.3 and 7000.25 s. On two-body rows through a
hyperbolic perigee the interpolation error is 1.75 m at 60 s spacing and
1.4 mm at 10 s.

### 2. Search stays two-body for the pursuer

The pursuer's trajectory depends on the burns, so it cannot be computed
once. The search keeps propagating it in two-body, which is what makes it
fast, but aims at the GMAT ephemeris and leaves headroom for the
correction:

- each impulse below the safe cap by a correction margin, starting at
  15 m/s (three times the 5 m/s measured above) and to be tuned on more
  cases;
- the total delta-v below its safe limit by the same margin;
- the altitude floor unchanged, since the corrected trajectory is checked
  again on replay.

A search of this shape is `src/orbitwin/ranking_search.py`, run by
`scripts/hunt_ranking_score.py` (`--target-from-gmat`,
`--correction-margin-km-s`, default 0). It keeps every impulse free and
capped and maximises the ranking score itself, so a near miss competes
with a hit. On the rehearsal scripts in `examples/ranking/`, with the
target from one GMAT run, the 0.249 / 2.99 km/s caps, a 100 km altitude
floor to T_max, and placeholder coefficients kt = 0.003 /s, Ct = 0.8 T_max,
kv = 1.2 per km/s, Cv = 2.0 km/s (analytical, 156 starts, 18 workers):

| | zero burn | best | burns | delta-v | first entry | wall |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 23.50 (589 km) | 96.34 | 1 | 0.234 km/s | 4303.4 s | 14 s |
| R2 | 23.67 (4718 km) | the zero burn | 0 | 0 | miss | 15 s |
| R3 | 24.05 (3028 km) | 81.40 | 12 | 2.872 km/s | 2241.2 s | 15 s |
| R4 | 23.97 (2495 km) | 87.30 | 12 | 1.734 km/s | 3393.4 s | 16 s |

R2's closest approach with the whole 2.99 km/s was 1973 km, worth 1e-7
distance points, so spending nothing scores best. That was the distance term
before the 2026/11/04 revision; under the revised one it depends on C_r
(examples/ranking/README.md, R2). The R3 and R4 hits
both run along the 100 km floor, and GMAT does not fly them as planned
(replays with VNB burns, no force-model margin):

| | floor | analytical | GMAT replay | GMAT lowest altitude |
|---|---:|---|---|---:|
| R1 | 100 km | hit, 96.34 | 21.1 km miss, 65.31 | 421 km |
| R3 | 100 km | hit, 81.40 | 43.1 km miss, 41.78 | 95.5 km |
| R4 | 100 km | hit, 87.30 | 194.1 km miss, 23.07 | 95.4 km |
| R4 | 200 km | hit, 82.64 (2.432 km/s) | 30.7 km miss, 49.04 | 192.7 km |

With a 200 km floor R3 has no hit and its best is the zero burn. Drag
took the three low arcs 4.5 to 7.3 km under the floor the two-body plan
held, so a floor needs margin for the force model as well as for the
correction. The search adds 10 km to the floor in SLSQP's constraint
(`FORCE_MODEL_MARGINS_KM`); the hard wall stays the 100 km floor. With VNB
burns, the default frame, R4's best is then 86.86 analytically and a
46.5 km miss worth 48.04 in GMAT, lowest altitude 104.7 km, where its
margin-0 winners replay at 193.7 km / 23.11 and 95.4 km altitude. On R5
the 10 km margin leaves only the zero burn (23.67). The margin-0 best
there, a 13-burn 88.3 km miss (28.32 analytically), replays at 113.7 km /
23.46 with VNB burns, below the zero burn, and at 108.2 km / 24.40 with
EarthMJ2000Eq burns. A margin of 0 is opt-in
(`--force-model-margins-km 0 10`) and costs 1.7 times the search time.
In solve it is `search.ranking_search_without_margin`, and a miss it keeps
above the zero burn is replayed through `gmat.verify_misses` (section 4), so
GMAT's score decides between it and the zero burn.

Maneuvers the project adds to an open template are written in VNB, the
organizer's workshop convention, unless `policy.synthesized_burn_frame`
is set to `EarthMJ2000Eq`. That setting is off by default until the
organizer answers Q7 in `docs/SCORING_STRATEGY_AND_OPEN_QUESTIONS.md`.
GMAT applies VNB on its own state, which here is not the two-body state
the impulse was planned on: on the R4 schedule above the impulse GMAT
flew differed from the plan by up to 0.431 m/s in VNB and by at most
0.020 m/s inertially, the floor of the row-difference estimate. The
correction step (`gmat_shoot.py`) absorbs that error along with the
rest, but nothing calls it yet; until the pipeline does (step 4), a GMAT
replay carries it.
Declared slots keep the organizer's frame. The R1 miss is the pursuer's
two-body drift that step 3 is for.

#### 2a. Where the two-body miss comes from, and a pursuer model

Measured 2026-11-02 on GMAT R2026a: the best uncorrected R4 and R3 plans,
each flown in private copies of its script with the pursuer's forces switched
on one at a time, against the target track the search aimed at.
Deviation of the pursuer from its two-body plan at the planned arrival, in km:

| forces | R4 | step | R3 | step |
|---|---:|---:|---:|---:|
| point mass | 0.00 | | 0.00 | |
| + J2 (degree 2, order 0) | 41.81 | 41.81 | 30.25 | 30.25 |
| + EGM96 20x20 | 42.35 | 0.58 | 30.74 | 0.56 |
| + SolidAndPole tides | 42.35 | 0.00 | 30.74 | 0.00 |
| + NRLMSISE00 / SPAD drag | 47.33 | 65.55 | 27.37 | 10.87 |
| + Sun, Luna | 47.32 | 0.01 | 27.37 | 0.00 |
| + SRP, relativity (full) | 47.32 | 0.00 | 27.37 | 0.00 |
| J2 and drag only | 46.86 | | 26.94 | |

J2 lowers the plans' 110 km perigee to 102.6 and 105.4 km, and the drag
there turns J2's along-track error radial (R4: [-2.3, 37.9, 18.8] km RTN
becomes [42.0, -10.4, 19.1]). GMAT's SPADFile drag with SphericalModel.spo
pushes along the relative wind: orbital energy rises through perigee, from
-29.3363 to -29.3070 km^2/s^2 on R4's point-mass-plus-drag flight.

The sign comes from the file. SphericalModel.spo states azimuth and
elevation in the reverse of the convention GMAT reads them in, and GMAT Jira
[GMT-6588](https://gmat.atlassian.net/browse/GMT-6588) was closed in R2019a
by documenting that convention and keeping the file, so every build since
then flies SPADFile drag with it this way. Another .spo would fly
differently. `solve`, `inspect` and `certify` therefore resolve each SPAD
file the template's drag or SRP reads as GMAT's FileManager::FindPath does:
an absolute name as written, else the script's folder (GMAT's working
directory), then GmatConsole's folder for a `./` or `../` name or a bare drag
name, and SPAD_SRP_FILE's folder for a bare SRP name. A blank SPADSRPFile is
SPAD_SRP_FILE itself. Our GMAT runs copy the script into a folder of their
own, with every SPAD, gravity or tide file it names by a relative name without
`..` that is found beside the template, so each file is resolved both
from there and from the template's own folder, where the organizer may run it.
A file GMAT would miss from either folder
is refused; each file's SHA-256 goes under `spad_files` in the solve summary,
the inspect output and certification.json. A file whose hash differs from
R2026a's stock SphericalModel.spo (`5168dea0...e531c`), or that differs
between the two folders, gets a loud warning and the run goes on, since the
organizer's template decides; #118 Q3 asks which file it names.

`pursuer_model.py` flies the pursuer under point mass, J2 about the IAU 1976
pole of date, and a table of rho * B against altitude read off GMAT's own
residual accelerations (8602 rows, 109 to 420 km, 7 % rms), with GMAT's sign.
It lands 1.1 km (R4) and 0.5 km (R3) from GMAT's full model; J2 about J2000's
z instead put R4 0.41 km off GMAT's J2 alone. The table was measured at the
rehearsal epoch, 2026-11-09 02:00 UTC; GMAT flew the same R4 plan 0.7 to
0.9 km differently with both epochs moved 6 to 18 h, 2.5 km at 30 days and
5.5 km at 182 days. The drag sign is R2026a's, and a build that signs it the
other way would put the model off by twice the drag term. RK4 on a 15 s
grid counted from t = 0, with the 12 central-difference copies of every leg
flown alongside, keeps the search's gradients. `search.ranking_pursuer_model:
j2-spad-drag` re-polishes the best `ranking_pursuer_model_top` two-body
solutions in it and offers them beside the two-body ones. They are kept out
of the prune (all of them stay) and are corrected, verified and rendered on
budgets of their own, so GMAT still sees every plan main makes. A script whose pursuer
force model or craft differs from what the table was measured on falls back
to two-body with a warning.

Full solve, rehearsal coefficients (kt 0.003 /s, Ct 0.8 T_max, kv 1.2 per
km/s, Cv 2.0 km/s), 4 GMAT workers, `config/ranking.yaml` otherwise:

| | main | with the model |
|---|---|---|
| R4 GMAT-verified best | 86.7273 | 86.8741 |
| R4 first flight, closest approach | 45.2-46.5 km | 2.2-2.4 km |
| R4 correction delta-v | +17.7 to +20.5 m/s | -1.1 m/s |
| R4 corrections converged | 8/15 | 15/15, beside main's 8/15 |
| R4 best's lowest GMAT radius less 6378.1363 km (geodetic) | 104.6 (109.6) km | 109.8 (114.3) km |
| R3 GMAT-verified best | 81.3918 | 81.3918 (model plans 81.3081) |
| R3 first flight, closest approach | 23.4-23.6 km | 0.90-0.96 km |
| R1 GMAT-verified best | 96.3908 | 96.3908 (model plans 96.3907) |
| GmatConsole runs, R4 / R3 / R1 | 42 / 63 / 164 | 102 / 82 / 184 |
| solve wall R4 / R3 / R1, s | 126 / 232 / 392 | 412 / 310 / 451 |
| of which ranking search, s | 16 / 21 / 16 | 84 / 41 / 63 |

Other jobs loaded the machine to about 13 on 20 cores during the runs with
the model and not during main's. With both at 22-28, R4 took 305 s on main
(42 s searching) and 580 s with the model as first committed (157 s).

The model's plans hold the 10 km force-model margin in the model, so GMAT
flies them about 5 km higher than main's, which it flies into the margin
(104.6 km on R4, 105.3 km on R3). On R3 that costs more than the correction
it saves, which is why the two-body plans stay in.
`scripts/audit_burn_table.py` replays a plan two-body: R4's best model plan
replays there at 103.5 km, 3.5 km above the 100 km wall, against 109.8 km in
GMAT. Another scenario could put a model plan GMAT flies safely under the
audit's wall.

### 3. Correction on GMAT residuals

New module, `gmat_shoot.py`. For each of the best K candidates by score:

1. Render the candidate and fly it in GMAT, reporting both craft at the
   planned arrival time t_f.
2. Miss vector = r_Human(t_f) - r_Alien(t_f), both from GMAT.
3. Change the last impulse by -J^-1 times the miss, where J is the two-body
   d r(t_f) / d v of that impulse, evaluated by finite differences with
   `propagate_universal`.
4. Repeat until the miss is under 0.5 km, at most 4 GMAT rounds (see below:
   up to 9 GmatConsole runs when the measured J is needed).

The target is 0.5 km rather than 5 km on purpose. The organizer's GMAT may
not be our build, and its data files (Earth orientation, space weather)
may differ. A closest approach deep inside the sphere survives small
differences that a 4.9 km pass would not.

After converging, recheck every cap. If the last impulse is now over its
cap, solve for the last two impulses together (six unknowns, least-norm
correction) once; if that also fails, drop the candidate.

When the two-body J does not converge (long arcs, several revolutions),
fall back to a J from GMAT itself: three perturbed runs in parallel, which
costs about one extra second of wall time.

Implemented in `src/orbitwin/gmat_shoot.py` as `correct_with_gmat`. Where
it differs from the plan above:

- The cap is checked before each step, not after converging: a step that
  would put the last impulse over its safe cap, or the total over its safe
  limit, is replaced by the two-impulse solve (SLSQP, caps 1e-9 km/s inside).
  A point that meets the equations and the caps is taken whatever SLSQP's
  status: two of six cap geometries ended with status 8 on such a point.
- The budget counts rounds. A round is one run, or the three (six with two
  or more impulses) concurrent runs that measure J; `gmat_runs` counts processes.
  J is measured when a two-body step leaves more than 30% of the miss.
- The miss is taken at t_f, as above, not at the closest approach: that
  residual has a rank-2 sensitivity and lets the arrival epoch drift. The
  stopping test is the closest approach of the first passage through the
  sphere, as verification measures a hit.
- R2026a writes the pre-burn row at a Maneuver and no post-burn one. The
  post-burn row is rebuilt from the script components on GMAT's own state;
  two-body from it to the next row agreed with GMAT to 21 m over 60 s.
  `burn_rows.py` does this for verification as well (#106).
- Any flight that goes inside the central body, at any point of the arc
  flown, ends the correction, and so does a run GMAT cannot finish: with
  drag on, a 249 m/s retrograde impulse at 6700 km made R2026a exit 1 at
  1274.8 s ("RungeKutta89: Accuracy settings will be violated") about 80 km
  up, well before the surface. The rounds before are kept.
- Only a converged correction returns a candidate, scored. Otherwise the
  closest legal flight comes back separately, unscored and not successful;
  after the earlier-pass stop it is the flight that hit t_f, not the closest.
- The sensitivity probes run at most `gmat.workers` at a time. Solve runs
  corrections and verifications `gmat.workers` at once in all, and each
  correction flies its probes one at a time.

Measured on R2026a, ranking force model, 2026-11-01, aim 0.5 km. "Miss" is
at t_f; "after" is the closest approach once converged. Run times are with
the machine at a load average of 20.

| case | first miss | after | runs | GmatConsole per run |
|---|---:|---:|---:|---:|
| probe above, burn in VNB | 21.470 km | 10.8 m | 2 | 1.2-1.5 s |
| probe, aimed at the two-body target | 10.200 km | 6.9 m | 2 | 1.2 s |
| 2 burns, 170 and 143 m/s | 37.149 km | 146 m | 2 | 1.6 s |
| 2 burns, last at 249 m/s along B (last two) | 36.600 km | 170 m | 2 | 1.6 s |
| 5 burns at 249 m/s (last shrank to 238 m/s) | 45.293 km | 164 m | 2 | 2.8 s |
| 5 burns at 249 m/s, last along B (last two) | 44.643 km | 157 m | 2 | 2.8-3.6 s |
| 2 burns, 9500 s arc (1.6 rev) | 135.6 km | 61 m | 3 | 1.7-2.3 s |

On the R1 rehearsal scenario (test/ranking-rehearsal-scenarios: one
propagator for both craft, full model, 1 s report rows), the five cheapest
analytical hits from `orbitwin solve`, one impulse of 89.7 to 91.8 m/s
arriving at 4519 to 4558 s, all went from 19.90-19.93 km at t_f to a closest
approach of 164-203 m in two runs, 3.6-7.5 s of GmatConsole each. Each
correction was 21.1-21.5 m/s, more than the 15 m/s margin step 2 proposes.

The probe's first miss is 21.470 km rather than 18.565 km because GMAT
builds VNB on its own state at the burn, 0.56 km and 3.7 m/s from two-body,
which turns a 4.67 km/s impulse by 2.0 m/s. Six two-burn geometries with the
last impulse at 249 m/s all converged in two runs, three of them through the
two-impulse solve (170 to 436 m). No case needed the GMAT-measured J; the
1.6-revolution arc contracted 0.037 per step.

### 4. Then the existing gates

The corrected candidate goes through the verification that exists today:
the diagnostic replay, the epoch check from #82, the radius check before
interception, Hermite first entry, the scored-miss rules from #81 and
#80, and the report-row alignment that ends the mission on the first row
inside the sphere.

As built (`gmat_loop.py`, called from `solve`): with `search.ranking_search`
the ranking search runs beside the existing one, aimed at GMAT's target
track when solve has a console (the conic, with a warning, if that run
fails), and keeps 15 solutions per margin. Under the ranking rules the zero
burn and those solutions stay in the pool however full the archive is. With
`gmat.correct_top` = K the first K hits inside the production margins are
corrected, each failing alone. Every converged correction is replayed as
soon as it converges, and the zero burn when a miss scores; the records and
lists come out in the order the hits were chosen. A replay waits only when
the candidate is an original whose own correction is still running (its
first flight is stamped first), or when an earlier-chosen correction of the
same burn epochs, slots, branch and arrival could still land on the same
schedule. Corrections and these replays share one pool of
`gmat.workers` GmatConsole processes. A replay goes before the corrections
still waiting while more than `gmat.workers` of them wait, and after them
once they fit in one round, so the last corrections do not run alone at the
end; the zero burn and then the scored misses below go last. Which runs
first changes only when results arrive, not what they are. A candidate two
corrections resolve to is replayed once; before, it was replayed twice into one directory, and the
second run's refusal un-verified it. The "GMAT verified none of the N"
warning counts replays run, so N no longer counts it twice. `verify_top` others follow once all of
them have finished, not counting originals a correction replaced. The
ranking profiles turn both on, with K = 40.

Under the ranking rules a miss scores on its closest approach, and with
`verify_top` 0 nothing else would replay one, so it could never be chosen.
`gmat.verify_misses` = M replays the first M misses, in rank order, whose
analytic score beats the zero burn's, inside the production margins and not
through the Earth (the altitude floor is judged on GMAT's flight), with M
more for plans flown in a pursuer model. Plans within 0.1 s and 5e-5 km/s in
every burn component count as one: on R5 the search kept 13 copies of one
13-burn miss from different starts, up to 6.4e-6 km/s and 3.1 ms apart, and
with M = 8 GMAT flew eight of them between 113.6517 and 113.6522 km. R5's two
distinct plans differ by 58 s, so they do not test the 5e-5 km/s side. Only GMAT's
score decides. The ranking profiles set M = 4.

Measured 2026-11-03 on R2026a (rehearsal scripts, the ranking day config
from `scripts/ranking_day_config.py`, placeholder coefficients as in step 2,
`gmat.workers` 4, another GMAT job on the machine):

| | margins | misses above the zero burn | replayed | GMAT | best | GMAT runs, before / after |
|---|---|---:|---:|---|---:|---:|
| R2 | 10 km | 0 | 0 | | 23.67, zero burn | 3 / 3 |
| R5 | 10 km | 0 | 0 | | 23.67, zero burn | 3 / 3 |
| R2 | 0, 10 km | 0 | 0 | | 23.67, zero burn | 3 / 3 |
| R5 | 0, 10 km | 15, two plans | 2 | 113.65 km / 23.46, 113.07 km / 23.56 | 23.67, zero burn | 3 / 5 |

The 0 km margin was searched on R2 and R5 only. The 10 km rows, and R1, R3,
R4 and R6 below, are this change's first version with M = 8, before the
near-duplicate rule; nothing was chosen in them, so the code path was the
one before. The two R5 replays took 20.7 s of GmatConsole time and ended
5.9 s after the rest; whole-solve wall time (43 s before, 45 s after) moved
more on R2, where nothing changed (40 s, 35 s).

The 0 km plans were 88.3 km and 91.3 km two-body (28.32 and 27.69) and both
fly about 113 km, under the zero burn, so on these scripts the budget changes
no answer. On R1, R3, R4 and R6 the ranking search keeps no miss with burns,
nothing is chosen, and best scores (96.39, 81.39, 86.73, 95.07) and GMAT run
counts (165, 63, 42, 177) are the same as before.

Measured on R2026a on 2026-11-02 with `scripts/rehearse_ranking.py`: rehearsal
scripts, placeholder coefficients as in step 2, `verify_top` 8, `gmat.workers`
8, 19 search workers; every best is `public_gmat_verified`:

| | best | from | corrections converged | GMAT runs correcting | wall |
|---|---:|---|---:|---:|---:|
| R1 | 96.39 | one-burn sweep | 40 of 40 | 87 | 267 s |
| R2 | 23.67 | zero burn | none tried | 0 | 47 s |
| R3 | 81.39 | ranking search, 12 burns, corrected | 15 of 15 | 30 | 164 s |
| R4 | 86.72 | ranking search, 11 burns, corrected | 7 of 15 | 22 | 154 s |
| R5 | 23.67 | zero burn | none tried | 0 | 49 s |

Solve reads the initial states through the state probe (#119), which moves
them by up to 1.1e-11 km, so the searches start from different seeds. Before
it, on the same day, R4's best was a 12-burn correction at 86.73 with 11 of 15
converging. Wall times include the probe's run and a machine shared with
other jobs.

Before this step GMAT flew the R3 and R4 hits wide: at the 10 km margin the
originals missed by 23.4-23.6 km on R3 and 45.0-46.2 km on R4. With
the 0 km margin searched as well, R3 to R5 ended on the same bests in 1.4 to
1.9 times the wall time; most of its hits fly under the 100 km floor in GMAT
and stop in their first correction round.

### 5. Budget

As built there is no time or run budget. K = `gmat.correct_top` candidates
are corrected, sharing `gmat.workers` with the replays, each in at most `gmat.correct_rounds`
rounds (4). A round is one GmatConsole run except the one that measures J:
3 runs for a one-impulse schedule, 6 for any with two or more. A correction
takes 0 to 9 runs: a miss is refused before flying, an original flown under
the floor stops after its first run, and the bound is 1 + 1 + 6 + 1. On R1-R5 the 40 corrections took
0 to 81 runs per scenario, 26 to 30 where the ranking search won.
Scored misses add at most 2 × `gmat.verify_misses` replays (two-body and
pursuer-model plans), one or two GmatConsole runs each.
`candidates.json` is written once every correction and verification has
finished, so a run that is stopped early leaves no answer.

## Not covered here

The search itself is still slow: a cold run on the preliminary scenario
took 34 minutes to reach 98.3154 (with other jobs on the machine). That is a
separate piece of work, to be measured again once the search aims at the
ephemeris.

## Steps, one PR each

1. Target ephemeris: one GMAT run, parse, interpolate. Tests against a
   real report and the fake console.
2. Search aims at the ephemeris, behind a setting, with the correction
   margins.
3. `gmat_shoot.py`: the corrector, with the probe above as a test case
   that runs when GMAT is available.
4. Pipeline: search, correct, verify.
5. Rehearsal: ranking-shaped synthetic scenarios with random coefficients,
   measuring time to the first verified answer and time to the best one.

## Open questions

- Whether drag and SRP apply to SC_Alien. The design does not depend on
  the answer, because the target ephemeris comes from the organizer's own
  script whatever it says.
- How far the organizer's GMAT build and data differ from ours. The 0.5 km
  aim is a guess until we can compare against one of their reports.
