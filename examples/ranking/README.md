# Ranking round rehearsal scenarios

Five synthetic scripts for rehearsing the ranking round (rules 2026/10/06).
None of them comes from the organizer. Each is shaped like the organizer's
second, replacement preliminary script (SHA-256 `1f378c65…`, as
`docs/LESSONS_2026-09-05.md` records): the same craft names, the same
`DefaultProp` (RungeKutta89, `MaxStep = 1`, so report rows are 1 s apart), the
same `ReportFile1` columns, and one `Propagate` with no burns.
The spacecraft blocks keep that script's mass, area and SPAD scale fields but
drop its display-only ones (orbit colours, covariance, 3-D model), and add
`SPADDragFile` and `SPADSRPFile` (see below). The plot subscribers are left out
because GmatConsole refuses a script that declares OpenFrames objects.

What differs from the preliminary is what the ranking rules change. SC_Alien
flies a hyperbola, passes perigee and climbs back out, and the force model is
the one in section 7. The epoch is 09 Nov 2026 02:00:00 UTC, the day of the
final. SC_Human is the preliminary's orbit (circular, SMA 6800 km, 421.9 km
altitude, INC 40, RAAN 60); only its starting TA changes between scripts.

## Scenarios

T_max is the first time after perigee that SC_Alien's geocentric distance
reaches the mission boundary. The boundary is ours (the real one is announced
on the day), chosen between 2 and 3 perigee radii. T_max was read off a
zero-burn GMAT run with the radius interpolated between rows, and each script's
`Propagate` goal is set to it. `scenarios.yaml` holds the same numbers for the
tests.

| Script | Boundary (km) | T_max (s) | Perigee at (s) | Perigee alt. (km) | v_inf (km/s) | Plane angle | SC_Alien at t = 0 | Zero-burn closest (km) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `R1_NearPlane_Reachable` | 17000 (2.49 r_p) | 6711.667 | 4500.238 | 448.995 | 2.0 | 5 deg | outside, 29096 km | 604.3 |
| `R2_SteepPlane_NearMiss` | 21000 (3.01 r_p) | 5791.381 | 2999.826 | 598.910 | 3.0 | 75 deg | outside, 22198 km | 4711.2 |
| `R3_FastHighPerigee_TwoPropagators` | 26000 (2.93 r_p) | 5092.815 | 2399.945 | 2501.053 | 7.0 | 30 deg | inside, 23669 km | 3014.2 |
| `R4_LowPerigee_Drag` | 13500 (2.02 r_p) | 5219.966 | 3600.236 | 321.374 | 1.5 | 20 deg | outside, 24166 km | 2468.9 |
| `R5_TiltedPlane_CloseMiss` | 21000 (3.01 r_p) | 5791.747 | 3000.036 | 601.635 | 3.0 | 60 deg | outside, 22198 km | 3049.3 |

Perigee time and altitude are GMAT's. v_inf and the plane angle (between the
two craft's angular momenta) are osculating at epoch.

How each was built: SC_Alien's plane is SC_Human's rotated by the plane angle
about a line through the centre of the Earth in SC_Human's plane, so the two
planes cross along that line. SC_Alien's perigee lies on that line (R1, R4) or
past it along SC_Alien's orbit (R2 by 60 deg, R3 by 20 deg, R5 by 40 deg).

### What the caps can reach

Two-body search results, not GMAT. A ranking search was run on each script at
three altitude floors, each a floor SC_Human must stay above over the whole arc
to T_max, with the 0.249 km/s per impulse and 2.99 km/s total caps and 101 s
between impulses. With no `score` block the search minimises the closest
approach first and the delta-v second, and the target is two-body. The search
is `scripts/hunt_ranking_score.py` as it was at 92533a8, before it was
merged; the numbers have not been measured again since. The script adds a
10 km force-model margin to the floor by default, which these runs turn off.
With `output/R5.yaml` made by the `sed` line under "Running one":

```bash
PYTHONPATH=src python scripts/hunt_ranking_score.py \
    examples/ranking/R5_TiltedPlane_CloseMiss.script --config output/R5.yaml \
    --altitude-floor-km 100 --force-model-margins-km 0 --workers 12 --top 5 \
    --output output/R5_hunt.json
```

| Script | floor 0 km | floor 100 km | floor 200 km |
|---|---|---|---|
| R1 | hit, 9 impulses, 0.084 km/s | hit, 2 impulses, 0.086 km/s | hit, 2 impulses, 0.086 km/s |
| R2 | miss by 1778 km, 2.99 km/s | miss by 1973 km | miss by 2181 km |
| R3 | hit, 11 impulses, 2.640 km/s | hit, 12 impulses, 2.872 km/s | miss by 140 km |
| R4 | hit, 10 impulses, 1.120 km/s | hit, 12 impulses, 1.724 km/s | hit, 12 impulses, 2.427 km/s |
| R5 | hit, 11 impulses, 2.707 km/s | miss by 89.5 km | miss by 396 km |

Every miss spends the whole 2.99 km/s. A single impulse is no shortcut. On a
60 s grid of burn and arrival times the cheapest one-impulse two-body hits on
R2, R3 and R4 (4.04 km/s, 2.38 km/s and 345 m/s) pass 1431, 345 and 225 km
below the Earth's surface. The cheapest that stay above it cost 6.20, 2.85 and
2.77 km/s and skim it at 3 to 15 km altitude.

- R1: the planes are 5 deg apart and the perigee, about 28 km above SC_Human's
  altitude, lies on the line where they cross. SC_Human is 10 deg ahead of
  that point at perigee time. In GMAT, under the script's own force model, one
  impulse at t = 0.001 s of 99.435 m/s, VNB (60.630660, -43.230759,
  65.896693) m/s, reaches SC_Alien: first entry at 4518.456 s, closest 0.1 m
  at 4519.92 s, and SC_Human stays at or above 421.9 km until first entry.
  Flown on to T_max it dips to 407.3 km at 5288 s; the floors in the table
  above apply to the whole arc to T_max, this one only up to the hit. The
  vector must be entered to the digits shown: rounded to 1e-3 m/s it enters at
  4518.454 s and passes 2.2 m away. The impulse came from a Newton correction
  on GMAT residuals made while reviewing this branch, and was flown again in
  GMAT to get the figures here. The two-body answer `orbitwin solve` gives, one 89.7 m/s
  impulse at the same time, misses by 19.79 km when GMAT flies it. A first
  version of R1 had SC_Human 12 deg behind; its cheapest single-impulse hit,
  150 m/s, dived to 107 km altitude to catch up.
- R2: the planes are 75 deg apart and the perigee is 60 deg past the line
  where they cross. R2 is a "spend nothing" check. The closest the caps reach
  is 1778 to 2181 km, where the distance term before the 2026/11/04 revision,
  `50 exp(-(d - 5)/100)`, is worth under 1e-6 points, so any delta-v costs
  more than it buys and the zero-burn script should score best. Under the
  revised term, `100/(1 + exp[((d - 5)/C_r)^2])`, that holds only for a small
  enough C_r. Two-body search, 100 km floor, no force-model margin, kt 0.003
  /s, Ct 0.8 T_max, kv 0.0012 per m/s, Cv 2000 m/s: the zero burn (4718 km,
  23.67) is best at C_r 100, 300 and 1000 km; at 3000 km a 13-impulse,
  2.99 km/s miss at 1973 km scores 46.00 against the zero burn's 31.48.
- R3: a fast pass, v_inf 7 km/s, 2500 km perigee. SC_Alien has a propagator
  of its own, on the same force model, and the two craft are stepped by
  `Propagate Synchronized`. Whether it is a hit depends on how low SC_Human is
  allowed to go.
- R4: the planes are 20 deg apart and the perigee, at 321 km and below
  SC_Human, lies on the line where they cross. The search's hits use 10 to 12
  impulses.
- R5: the planes are 60 deg apart and the perigee is 40 deg past the line
  where they cross. In the two-body search a hit needs SC_Human to skim the
  surface, about 0.05 km up. GMAT drifts 8 to 57 km from two-body within the
  hour (the drift table above), so a two-body skim gives no assurance that
  the flown arc stays above the surface; at a
  100 km floor the best is a miss by 89.5 km, worth 21.5 of the 50 distance
  points against 0 for the zero-burn 3049 km. Flown in GMAT, that 13-impulse,
  2.99 km/s schedule passes 107.3 km away at 2651.7 s (worth 18.0 points), with
  SC_Human no lower than 100.2 km, so the miss is where the score is decided.
  Those points are the term before the 2026/11/04 revision. Under the revised
  term the 89.5 km miss is worth 32.9 at C_r 100 km and 48.0 at 300 km
  (two-body search as for R2 above, total 39.46 and 54.61 against the zero
  burn's 23.67). At 1000 km the search stops at 202.9 km on 2.80 km/s, just
  inside 5 + 0.2 C_r where the term is still 49, and scores 56.69; at 3000 km
  it stops at 1376 km on 1.08 km/s and scores 64.34 against the zero burn's
  49.93.
  An earlier R5 had SC_Human 2 deg further back; its best miss at a 100 km
  floor was 304 km, worth 2.5 points, which made it a second "spend nothing"
  case.

## Force model

All five use the section 7 model for SC_Human. Choices where the rules leave
room, or where GMAT R2026a does not accept the rules' wording:

- Gravity `EGM96.cof`, degree and order 20.
- `TideModel = 'SolidAndPole'`. The 2026/10/06 PDF's `SolidandPole` stops
  R2026a with `The selected tide option "SolidandPole" is invalid`; the
  2026/11/04 revision spells it `SolidAndPole`.
- Drag `NRLMSISE00` and SRP `On`, both with the `SPADFile` model as the rules
  say. We do not have the organizer's SPAD file, so both point at GMAT's own
  `../data/vehicle/spad/SphericalModel.spo` (a sphere, 8.945 m^2), the file
  the stock startup file already names as `SPAD_SRP_FILE`. The path is
  relative to GMAT's `bin` directory, which is where GmatConsole runs under
  `orbitwin`, and is how GMAT's own samples write it. A bare file name, or no
  file at all, is refused: `SPAD drag data requested for Spacecraft SC_Human
  but no SPAD file has been provided`. If the organizer supplies a `.spo`, its
  areas replace these. `Cd`, `DragArea` and the other area fields stay as in
  the template; `SPADFile` does not read them.
- Space weather `ConstantFluxAndGeoMag` with F10.7 = F10.7A = 150 and Kp 3,
  GMAT's defaults when drag is switched on in the GUI. The run then does not
  depend on how far the space-weather file's predictions reach. An organizer
  script that reads the CSSI file instead will see different densities.
- Sun and Luna point masses, relativistic correction on.

R1, R2, R4 and R5 put both craft on one propagator. R3 gives SC_Alien its own
propagator with the same model, since rule 2 as corrected on 2026-11-01 puts it
under every section 7 force (`僅受外界力場影響,無主動機動`). Before the
correction rule 2 read `僅受重力場影響` and R3 kept SC_Alien off drag and SRP;
adding them moves SC_Alien 0.78 m by T_max, its most over [0, T_max], so every
R3 figure in scenarios.yaml and the drift table below is unchanged to the digits
shown. Only the run takes longer.

Drag hardly moves SC_Alien either way. In R4, with the lowest perigee, turning
drag off for SC_Alien moves it 3.6 m by T_max and turning SRP off as well 1.2 m
more; R1 differs from a gravity-only SC_Alien by 3.4 m. On a single flyby the
target spends a few hundred seconds low enough for drag to act.

## Measured on GMAT R2026a

Distance between a two-body propagation of each craft's initial state (what the
search plans with) and GMAT's report row, and the wall time of one GmatConsole
run of the script as written, zero burns, to T_max. Runs were one at a time on a
20-thread (14-core) machine that other jobs were also using.

| Script | SC_Human 30 min | SC_Human 60 min | SC_Alien 30 min | SC_Alien 60 min | GmatConsole wall |
|---|---:|---:|---:|---:|---:|
| R1 | 8.3 km | 19.0 km | 0.09 km | 0.98 km | 5.0 s |
| R2 | 8.9 km | 24.2 km | 0.42 km | 5.65 km | 4.5 s |
| R3 | 18.2 km | 56.8 km | 0.67 km | 3.19 km | 4.4 s |
| R4 | 14.3 km | 42.8 km | 0.36 km | 2.80 km | 3.8 s |
| R5 | 11.9 km | 19.4 km | 0.57 km | 4.51 km | 4.2 s |

SC_Human's orbit is the same in all five and only its starting point differs,
yet its drift ranges from 8 to 18 km at 30 minutes. The osculating elements GMAT
is handed differ from the mean ones by an amount that depends on where on the
orbit the state is given, and that sets how fast the along-track error grows.
Every one of these is far outside the 5 km sphere, which is what
`docs/GMAT_IN_THE_LOOP.md` is about.

## Running one

`config/ranking-rehearsal.yaml` is `config/ranking.yaml` with a 600 s
GmatConsole timeout: the ranking limits (`rules: ranking`, 0.25/0.249 km/s per
impulse, 3.0/2.99 km/s in total), `pursuer_name: SC_Human` and `target_name:
SC_Alien` (the open template names neither), the ranking search and
correction, and 8 GmatConsole runs at once. It leaves `horizon_s` null, so a
scenario cannot be parsed until its T_max is filled in, as on the day:

```bash
mkdir -p output
sed 's/^  horizon_s: null$/  horizon_s: 6711.667/' config/ranking-rehearsal.yaml > output/R1.yaml
PYTHONPATH=src python -m orbitwin solve examples/ranking/R1_NearPlane_Reachable.script \
    --config output/R1.yaml --output output/R1
```

`scripts/measure_ranking_scenario.py` recomputes, from the script and GMAT,
T_max, the perigee time and altitude, v_inf, the plane angle, the zero-burn
closest approach, SC_Human's final position, the drift table and the wall time.
It fails if the measured T_max and the script's goal differ by more than 5 ms
(pass `--provisional` while choosing a goal). It does not recompute the reach
table, the R1 GMAT correction or the drag and SRP figures, which came from
one-off runs.

```bash
PYTHONPATH=src python scripts/measure_ranking_scenario.py \
    examples/ranking/R1_NearPlane_Reachable.script --boundary-km 17000 \
    --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" \
    --output output/measure-R1
```

Tests: `tests/unit/test_ranking_rehearsal.py` parses every scenario under the
rehearsal config and pins its force model and propagators field by field;
`tests/integration/test_ranking_rehearsal_gmat.py` (marker `gmat`; it fails
without `GMAT_CONSOLE` and `GMAT_STARTUP`, as the other `gmat` test does) runs
each one with the diagnostic report and checks T_max, perigee, the zero-burn
closest approach and SC_Human's final position to 10 m.

## Asynchronous-start rehearsals R6-R9

The ranking round starts the two craft on different epochs (the organizer's
tutorial of 2026-11-01). `scripts/build_async_rehearsals.py` builds these
from R1 and R2 on GMAT R2026a: one craft's epoch moves 1800 s earlier, to
01:30:00 UTC, and its state becomes where GMAT's section 7 model flies it back
to in that time. At 02:00:00 the geometry is therefore R1's or R2's; GMAT's
alignment returns the early craft there to within 5e-9 km.
`async_scenarios.yaml` holds each one's T_max, the policy it needs, and the
hashes of the data files the build read: Earth's orientation moves R6's
EarthFixed insertion, so rebuild after refreshing the EOP file. ReportFile1
also carries `SC_Human.Earth.Altitude` first, the column the organizer said it
will add.

| Script | Built from | Early craft | Alignment | T_max from the earlier epoch (s) |
|---|---|---|---|---:|
| `R6_InsertedLate_Aligned` | R1 | SC_Alien | the tutorial's UTC equality form | 8511.667 |
| `R7_NoAlignment_NearMiss` | R2 | SC_Alien | none: `policy.add_epoch_alignment` adds the A1 form | 7591.381 |
| `R8_PursuerFirst_Aligned` | R1 | SC_Human | UTC equality, SC_Human flying alone 421 km up | 8511.667 |
| `R9_PursuerFirst_LongHeadStart` | R8 (R1) | SC_Human | as R8, 3600 s long | 8511.667 |

- R9 is R8 with SC_Alien moved 1800 s later, to 02:30:00, where GMAT flies
  R1's on to: SC_Human has 3600 s alone, the case for head-start burns
  (`policy.head_start_burns`). Two-body, forbidding burns before 3600 s cost
  about 1.08 points on this physics (output/phase-review/final/headstart,
  arm F3600); before 1800 s, R8's case, 0.003. `--only R9` rebuilds it alone.

- R6 gives SC_Human's state as an EarthFixed SphericalAZFPA insertion (RMAG,
  RA, DEC, VMAG, AZI, FPA), the tutorial's form, at R1's state. That puts it at
  longitude -95.3 deg and latitude -39.2 deg, not over Taiwan: the state, not
  the place, is what is rehearsed. Only GMAT reads it, so it needs a console.
- T_max runs from the earlier craft's epoch in all four
  (`mission_time_origin: earliest`), as the organizer answered on 2026-11-04.
  On the solver clock, which starts at the later epoch, R6-R8 keep their
  source's T_max and R9 has R1's less 1800 s, 4911.667 s. R8 counted from SC_Alien's epoch, 6711.667 s, before the answer.
- Each joint Propagate's goal counts from that Propagate, so it is R1's or R2's
  T_max; R9's is that solver-clock T_max.
- Each row carries its source's `mission_boundary_km`.
  `scripts/rehearse_ranking.py` writes it into the config it runs, where the
  time-origin evidence of inspect and solve reads it with the goal above
  (docs/ASYNC_START_AND_INSERTION.md, "Time-origin evidence").

`scripts/rehearse_ranking.py --certify --team-number 54 R6 R7 R8 R9` solves and certifies them.
`scripts/check_report_rows.py` with `--config` reads T_team from the earlier
craft's ElapsedSecs, the one that runs ahead: `SC_Alien.ElapsedSecs` on R6 and
R7, `SC_Human.ElapsedSecs` on R8 and R9. By hand, without `--config`, pass that
`--time-column`: without one it refuses them, since the two craft's
ElapsedSecs differ by the alignment.
