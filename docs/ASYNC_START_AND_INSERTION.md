# Insertion states and asynchronous start: design

Status: steps 2 to 7 are implemented; see the "as built" sections below. Δ > 0 is accepted. `policy.add_epoch_alignment` defaults to false; the ranking profiles set it since the 2026-11-03 sign-off in COMPETITION_POLICY.md.

## What the organizer announced

On 2026-11-01 the organizer told contestants that the ranking round "需要進行入軌影響模擬與非同步起飛的模擬". The same day it posted a tutorial (https://youtu.be/DAfod6-xQEI, "GMAT 教育訓練課程 (4) -- 入軌模擬與非同步起飛"). The tutorial is a GMAT GUI demonstration with no script on screen. Everything below that it shows was read off the dialogs.

- **Insertion.** A spacecraft's initial state is given relative to the rotating Earth, as `CoordinateSystem = EarthFixed` with `DisplayStateType = SphericalAZFPA`:
  - `RMAG` is the geocentric radius.
  - `RA` and `DEC` are longitude and latitude.
  - `VMAG` is the Earth-relative speed.
  - `AZI` is the azimuth (90 = east).
  - `FPA` is measured from the local vertical (90 = horizontal).

  The competition uses this spherical form, not `Planetodetic`. "Script 會直接設定好": the organizer's script carries the state.
- **Asynchronous start.** The two craft have different `Epoch`s: 00:00 and 15:00 in the demo. The earlier one is propagated alone until the epochs match, then both are propagated together:

  ```
  Propagate DefaultProp(A) {A.UTCModJulian = B.UTCModJulian};
  Propagate DefaultProp(A, B) {A.ElapsedSecs = 3000, B.ElapsedSecs = 3000};
  ```

  This is our reading of the dialogs; the script text was not shown.
- **Surface crossing** is judged on the report column `Altitude`. Below 0 is a crossing; above 0, however small, is a graze. "我們之後會多輸出這個指標."

Measured on GMAT R2026a with a script of exactly this shape (`async_insertion.script` in the session scratchpad):

- GmatConsole runs it.
- The first segment propagates the earlier craft alone by the epoch difference (54000 s). The later craft's rows stay at its epoch, with its ElapsedSecs 0.
- `Earth.Altitude` is geodetic. At RMAG 6871 km and latitude 23.5° it reads 496.28 km, against 492.86 km above a 6378.1363 km sphere.
- A `Report` as the first mission command reports each craft at its own initial epoch. On the same resources it printed SC_Alien at A1ModJulian 21544.50037 and SC_Human at 21545.12537, 54000 s later, with SC_Human's EarthFixed insertion state converted by GMAT to EarthMJ2000Eq Cartesian at radius 6871 km. That is the state probe in §1.

## What main does with it

Each of these refuses the template, or would mis-measure it, on the day:

1. **Stop condition.** `gmat_parser` refuses the stop condition `A.UTCModJulian = B.UTCModJulian`. #94 accepts only ElapsedSecs/ElapsedDays goals.
2. **Coordinate system.** `_build_spacecraft` refuses `CoordinateSystem = EarthFixed`: "analytical solver only accepts EarthMJ2000Eq".
3. **State type.** It also refuses `DisplayStateType = SphericalAZFPA`.
4. **Epochs.** The parser refuses different epochs: "Pursuer and target epochs differ; epoch alignment must be explicitly implemented".
5. **Report sync.** `report_parser` holds the craft together by comparing their `ElapsedSecs`. Asynchronous craft differ by the epoch offset on every row, so every verification would fail.
6. **Surface check.** It uses the geocentric radius against a sphere. That is stricter than the organizer's geodetic `Altitude` everywhere except at the equator. It is safe, but it can refuse legal arcs.

## Prerequisite

`policy.minimum_flown_altitude_km`, the 100 km floor in the ranking profiles and the altitude gates in verify, certify and rank exist only on `feat/search-the-ranking-score`. That branch must merge first.

## Design

### 1. Initial states and epochs come from GMAT

We do not convert EarthFixed states ourselves. That would need GMAT's Earth orientation model: precession, nutation, polar motion and EOP data.

A state probe runs a private copy of the organizer's script. Its resources are unchanged except that GUI subscribers are neutralised, as the diagnostic sibling already does: GmatConsole refuses any script that declares OpenFrames objects, which on the workshop template means 58 errors and rc 1. It has its own ReportFile in an isolated run directory, and its mission sequence is replaced by a single `Report` of each craft before any propagation. The probe uses `Precision = 17`, which gives bit-exact doubles. For each craft it reports:
- `A1ModJulian`, `UTCModJulian`, `TAIModJulian`, `TTModJulian`, `TDBModJulian`;
- `EarthMJ2000Eq.X/Y/Z/VX/VY/VZ`.

GMAT reports each craft at its own initial epoch. The probe gives every inertial state, every absolute epoch, and the epoch difference Δ.

- **Δ comes from the probed A1 epochs**, never from the Epoch text. Across the 2016-12-31 leap second, one UTC day spanned 86401 s of A1 time.
- **The same instant in two formats can differ by one double step.** UTCGregorian, UTCModJulian, TAIGregorian and TAIModJulian probed to 21545.125370768306, and TTGregorian to 21545.12537076831, 0.31 µs apart. GMAT runs a joint Propagate on that pair with no alignment, rc 0 and no message. So a template with |Δ| ≤ 1e-6 s and no alignment is synchronous; the A1 synchronisation check in §4 covers it. An alignment whose Δ is non-zero but within 1e-6 s is refused as ambiguous.
- **With an alignment, Δ = 0 means bit-equal probed A1 epochs.** At Precision 16, A1ModJulian prints in 1e-11-day steps (0.86 µs). A reversed alignment off by one such step hung GMAT: it simulated 1.5e8 s and wrote 2.5 million rows (457 MB) in 40 s.
- **The probe replaces the parser's epoch-text and DateFormat-equality checks** (`gmat_parser.py` about 701-726), because it compares absolute instants.
- **Fast tests:** the fake console answers the probe for EarthMJ2000Eq scripts.
- **gmat-marked tests:** run the real probe on EarthFixed and SphericalAZFPA.
- **Without a console:** an EarthMJ2000Eq template with equal epoch texts keeps today's in-process parsing. Anything else is refused with a message saying a GMAT console is required.

**Data dependence.** The probe and every GMAT run read GMAT's data files. A stale EOP file moved SC_Human's probed state by 6-25 m (12-50 % of the 50 m stop depth then; `policy.stop_depth_km` is 1.5 km now), and a 0.05 s shift in UT1-UTC by 23 m.

The certificate manifest therefore records the hash of every file the startup file points at: EOP, leap seconds, EGM96, the tide file, space weather for NRLMSISE00, the SPAD files and the planetary ephemeris. Our EOP file is R2026a's own and ends 2026-09-14. Keep it unless the organizer says they updated theirs: the Windows R2026a download carries the same file (identical after CRLF to LF, checked 2026-11-04), so refreshing ours would make our runs differ from the organizer's.

GMAT facts this relies on, measured on R2026a:
- VMAG is Earth-relative.
- AZI 90 is east.
- FPA 90 is horizontal. GMAT's prose for FPA contradicts this; its HFPA = 90 − VFPA relation does not.
- DEC is geocentric latitude.

### 2. The alignment is not a Propagate segment

`PropagateSegment` durations feed the burn epochs, the horizon check and the "one more segment than burns" rule in `assert_final_script_policy`. The alignment is therefore a separate field, `Scenario.alignment`, recording:
- the early and late craft;
- its propagator;
- its line index and exact text;
- Δ.

GMAT aborts a joint `Propagate P(A, B)` whose craft are on different epochs, at 15 h and at 0.5 s apart, with "Command Exception: Spacecraft epochs do not match" and rc 1. A joint Propagate with no alignment ran at Δ = 1.57 µs and at 3.14 µs (rc 0, no message; GMAT silently made the epochs equal) and aborted at 10.4 µs, so GMAT's abort threshold lies between 3.1 and 10.4 µs. We do not rely on it: a template with 1e-6 s < Δ and no alignment is refused, even where GMAT would run it.

The parser accepts the alignment only as the first mission command, in one of three forms:
- `Propagate <Prop>(<early>) {<early>.<E> = <late>.<E>}`, with `<E>` one of A1/TAI/UTC/TT/TDBModJulian, the same on both sides;
- `{<early>.ElapsedSecs = <x>}` or `{<early>.ElapsedDays = <x>}`, with `<x>` equal to Δ;
- `{<early>.<E> = <number>}`, with the number equal to the late craft's probed epoch in that scale.

Tolerances rest on measurement. GMAT's stops land 0.08-0.39 µs from the goal. Its own joint Propagate runs at a 1.57 µs mismatch and aborts at 10.4 µs. So the accepted forms must equal the probed value within 1e-6 s, which is meaningful only because the probe is bit-exact.

- Numeric UTCModJulian, A1/TAI goals and `ElapsedDays` all end exactly at Δ across a leap second.
- The equality form stops 1 s short when the late epoch is written inside a leap second (`23:59:60.500`). Both the A1 sync check in §4 and GMAT's own abort catch that.

Every other non-elapsed stop condition is still refused. The early craft must be the one with the earlier probed epoch, otherwise the template is refused, because GMAT never reaches the goal. Δ = 0, bit-equal, is a zero-length alignment that GMAT ends at once. Unlike an `ElapsedSecs = 0` goal, it does not hang. A burn right after it fires at t = 0, which rule 2.2 allows, and is counted by the rule below.

The renderer never edits the alignment line, and the final-script policy guard checks it byte for byte. One counting rule applies:

- Segments and burns are counted after the alignment.
- When the alignment is followed directly by a `Maneuver`, there is one implicit segment of duration 0 between them. That segment is exempt from `policy.min_propagate_duration_s` in the renderer, the policy guard and the search. Such a script runs on R2026a with rc 0, both at Δ = 54000 s and at bit-equal Δ = 0.

**A template with Δ > 0 and no alignment.** A joint Propagate in split form `P(A) P(B)` with Δ > 0 and no alignment is refused. GMAT runs it with both ElapsedSecs reading 600 while the A1 epochs stay 54000 s apart, which is not a simultaneous state.

Adding an alignment line was not covered by the open-template rule, which licenses burn resources and their commands. thc1006 signed it off on 2026-11-03 (COMPETITION_POLICY.md), and the ranking profiles turn it on, for open and slotted templates alike:
- with `policy.add_epoch_alignment` false (the default), such a template is refused with a message naming the question;
- set to true, the renderer adds `Propagate <the template's propagator>(<early>) {<early>.A1ModJulian = <late>.A1ModJulian};` as the first mission command, where the parser accepts an alignment. This is the tutorial's form in the A1 scale, because the UTC equality form stops 1 s short when the late epoch falls inside a leap second.

**The pursuer earlier than the target.** When the pursuer is the early craft, it flies alone during the alignment.
- Since the 2026-11-08 sign-off (COMPETITION_POLICY.md, burns in the head start), `policy.head_start_burns` lets an open template's script burn before the alignment ends: pursuer-alone `Propagate <P>(<pursuer>) {<pursuer>.ElapsedSecs = <s>};` and `Maneuver` pairs before the alignment line, which stays as it was. Ranking §2.2 and the 2026-11-04 answer put mission start at the earlier epoch. On the solver clock such a burn is at t in [-Δ, 0); the search flies it from the pursuer's probed state at its own epoch, and at t = 0 adds the alignment's defect (GMAT's aligned state less the unburned two-body coast: 20.3 km on R8, 66.45 km and 69.4 m/s on R9), so the plan leaves t = 0 from GMAT's state plus what its burns changed. How the validator reads a burn in those rows is Q5, so rank takes one only by `policy.head_start_min_gain_points`. Without the policy, or on a slotted template, none is placed (Q2).
- Its state at t = 0 comes from GMAT propagating the alignment, never from two-body.
- Its altitude is checked over the alignment too (§5). If it crosses the surface there without a burn, the template is reported infeasible: the plans without head-start burns cannot prevent it. The tutorial's own insertion state has a perigee near 6031 km, below the surface.

### 3. One solver clock

The solver's t = 0 is the end of the alignment: the late craft's epoch, when both craft fly together. The solver clock is the late craft's own `ElapsedSecs`. GMAT derives it from A1 day doubles, so it resolves to 0.314 µs (a 600.25 s goal printed as 600.25000059977). Δ taken from two A1 values can be off by up to about 0.6 µs, inside the 1e-6 s tolerance. Burn epochs, Propagate durations, the analytical search, the GMAT corrector and the target ephemeris all use it, and the target's state at t = 0 comes from the GMAT ephemeris.

The scoring origin was Q1. The organizer answered Team 54 in writing on 2026-11-04 18:17: 「從比較早開始的開始算」, the mission starts at the earlier of the two epochs, for T_team, T_max and Ct alike. So:
- `policy.mission_time_origin` is required whenever Δ > 0, and under the ranking rules it is `earliest`, the craft with the earlier probed A1 epoch; `target` and `pursuer` are refused there and stay for the preliminary rules. `Scenario.time_origin_craft` records the craft it resolves to, the pursuer when the craft start together, and inspect prints it with Δ and T_max on the solver clock.
- `policy.closest_approach_window` (`after_alignment` or `from_origin`) sets d_min's window, and also the scored-miss window from #99. Whether the alignment's rows count is still asked of the organizer, so it stays required and has no default.
- Scores, the horizon, rank's horizon stamp and reported times convert between the solver clock and the origin by Δ. A template whose Δ reaches the announced T_max is refused.
- The parser refuses a T_max before SC_Alien's perigee on its two-body coast from t = 0 when SC_Alien is clearly eccentric (osculating e above 0.05; J2 alone gives a circular LEO orbit up to 2.95e-3, and a perigee anywhere in a period). Otherwise T_max is unchecked, and every command warns so: §3's premise does not hold, and the organizer confirms T_max. After the organizer confirms such a T_max in writing, `policy.accept_announced_horizon_before_perigee_s` (ranking_day_config.py's `--accept-announced-horizon-before-perigee-s`), which must equal T_max, takes it; inspect, solve's summary, rank and the certificate then record the perigee it contradicts under `announced_horizon_before_perigee`. It warns when `policy.mission_boundary_km` is set and SC_Alien is more than 10 km plus the roundings off it at T_max, or not climbing. On R1-R5 under the §7 model the two-body coast was within 0.24 s of GMAT's perigee and 1.81 km of its radius at T_max.

### 4. Reports: time columns and frozen rows

Measured on R2026a:
- `<craft>.ElapsedSecs` counts from that craft's own epoch, so the late craft reads 0 through the alignment.
- `ElapsedSecs` stop goals restart with each `Propagate`.
- Every alignment row carries the late craft frozen at its initial state. Exactly one row sits at the late epoch, with no duplicate at the start of the joint segment. That row is t = 0 and is kept.
- The official report is byte-identical between the final script and the diagnostic sibling, and both have the same number of rows (920/920 with fixed steps, 576/576 adaptive). An official ReportFile listing only the late craft still writes every frozen row.
- "Epochs do not match" appears zero, one or more times in successful runs: twice when two joint segments followed an alignment that ended one double step off, and never when the epochs were bit-equal. It appears again just before the fatal abort when a joint Propagate is truly out of step. It must never be a fatal marker or be counted; rc ≠ 0 already catches the fatal case.

Therefore:
- **Diagnostic sibling.** It adds both craft's `A1ModJulian` and the pursuer's `Earth.Altitude`, at Precision 17. `report_parser` checks synchronisation on the A1 columns within 1e-6 s, about three double steps at these epochs.
  - Relative quantities (entry, closest approach, d_min) ignore the alignment rows.
  - The surface checks run on every row in which the pursuer is propagated (§5).
- **Ephemeris and coast run.** `ephemeris_from_report` and the target coast run map rows onto the solver clock and bounds-check it. The coast run replays the alignment line verbatim.
- **Last-row readers.** `scripts/check_report_rows.py` takes the frozen-row count from certify, which reads it off the diagnostic sibling, and skips those rows. It says which time column it reads. certify also compares the official row counts of the sibling and the final rerun, because the sibling comments out plots.
- **Size cap.** A run that never reaches its goal wrote 20 MB in about 2 s. `gmat_runner` limits the file size GmatConsole may write, sized from horizon ÷ MaxStep × row width with a generous factor, and also sets `RLIMIT_CORE = 0`, because apport took a core dump.
  - Not through `preexec_fn`, which Python documents as unsafe with threads; `pipeline.py` runs verifications in a ThreadPoolExecutor. The console runs through an exec shim (`prlimit`), so the cap holds before it starts; `resource.prlimit(pid, …)` after the start is only the fallback, and misses what the console runs first.
  - The kernel then ends such a run with SIGXFSZ. `subprocess` reports that as `returncode == -signal.SIGXFSZ` (−25), not 153, which was a shell wrapper's code. It gets its own message.
  - Hangs that write nothing still rely on the timeout.

### 5. Surface check: never looser

Crossing the Earth before interception disqualifies (rules §4 note 1(b)), and our own absolute rule is never below the surface, ever.

- **The gate.** The sphere check stays: geocentric radius ≥ 6378.1363 km plus `policy.minimum_flown_altitude_km`. It covers every row in which the pursuer is propagated, including the alignment when the pursuer is early, through interception and on to the end of the flown arc. Today `report_parser` stops the radius check at interception; it must cover the whole arc.
- **The geodetic Altitude** (the organizer's criterion) is recorded and reported as a **cross-check**, not a gate. Geodetic altitude is always ≥ r − 6378.1363 km, so it can never fire on a row the sphere check passed.
- **Between rows.** Both checks read rows. With the tutorial's DefaultProp, rows near a 150 km perigee came 85 s apart, and the true minimum was 1.63 km below the lowest row at e = 0.3. The 100 km floor covers this.
- **Relaxation.** Judging only by geodetic altitude would need the user's explicit decision and the organizer's answer to Q3. It is not part of this design.

### 6. Rehearsal scenarios, tools and documents

- **R6:** SC_Alien earlier, in EarthMJ2000Eq. SC_Human inserted later from an EarthFixed SphericalAZFPA state over Taiwan, with an alignment line. Reachable on the caps.
- **R7:** an open template without the alignment line (needs `add_epoch_alignment: true`), with only a near miss reachable.
- **R8:** SC_Human earlier than SC_Alien. Its alignment arc stays above the floor.
- **R3:** SC_Alien moves to the full §7 model (corrected rule 2).
- **Tools:** `audit_burn_table.py` and `ood_fuzz.py` learn asynchronous cases. rank stamps Δ, `mission_time_origin` and the probe's hash.
- **Documents:**
  - `COMPETITION_POLICY.md`: "Spacecraft A is affected by gravity only" becomes "external forces as configured in §7".
  - `ASSUMPTIONS_AND_GAPS.md`: assumption 2 becomes this design.
  - Both gain the `add_epoch_alignment` permission question, for a person to sign off.

## Step 2 as built

- **API.** `state_probe.load_scenario(script, config, gmat_console=, startup_file=)` is the one way in for an entry point that may hold a console. With a console it runs `probe_initial_states` in a temporary directory and then `parse_gmat_script(script, config, probe=...)`; without one it is today's in-process parse. It returns the scenario, the document and the probe's files as `ProbeEvidence`, whose record carries no paths until `ProbeEvidence.place` writes them somewhere.
  - solve, baseline and certify read the script and make every refusal first, then create their output directory and place the probe in `<output>/state_probe`. A refusal leaves nothing behind, so a retry into the same directory works.
  - inspect, rank and verify-policy take an optional `--gmat-console`/`--startup-file` pair and never place the probe, so they record no path to a deleted directory. So do `hunt_multi_burn.py`, `audit_burn_table.py`, `safe_delta_v_floor.py` and `hunt_ranking_score.py` (whose `--target-from-gmat` console also probes).
  - Every re-parse of a derived script (the renderer's policy guard, verify's flown scripts, the coast run, certify's final replay) passes the original's `Scenario.state_probe`. A probe is bound by hash to what places each craft: its assignments, those of a CoordinateSystem resource it names, and those of every object the script does not create (Earth, SolarSystem, ...). A script that changes any of them is refused.
- **Before GMAT runs anything**, `assert_resources_supported` refuses Include, line continuation, resource lines the parser cannot read, an output file (ReportFile or EphemerisFile Filename, DifferentialCorrector ReportFile) that holds `/`, `\`, `:` or a non-printing character, or is empty or all dots (GMAT on Linux reads a backslash as `/`, so `..\..\x.txt` climbed out of the run; a space or parentheses land in `gmat-output/`), and any resource type outside the allowlist or `policy.extra_resource_types`. Some types are never run, whatever that setting says: the event locators, the estimation resources and GmatFunction, MatlabFunction, Python, FileInterface and ThrustHistoryFile. On R2026a an EclipseLocator and a ContactLocator wrote over files outside the run, and an `#Include`d ReportFile deleted one before the old parser's refusal arrived. A relative EphemerisFile Filename lands in the run's `gmat-output/`.
- **The probe copy** keeps what places a craft or is named by one (Spacecraft, CoordinateSystem, Variable, Array, String, tanks and thrusters including the legacy FuelTank and Thruster, power systems, Plate, GroundStation, Barycenter, LibrationPoint), plus any hardware, or admitted type, one of those names through Tanks, Thrusters, PowerSystem, AddHardware or Tank. Everything else is commented out: GMAT refuses a craft that names a dropped tank. The diagnostic sibling comments out the plots, any EphemerisFile and any type admitted by the setting, unless a craft names it or it reads a data file. A failed probe's exception carries GMAT's own ERROR lines, since its directory is deleted.
- **Model.** `SpacecraftSpec.initial_state` is the probed state; `probed_epoch` holds A1/UTC/TAI/TT/TDB ModJulian and `in_process_mismatch` the distance to our own conversion where one exists. `Scenario.state_probe` holds the run, `Scenario.epoch_offset_s` the target-minus-pursuer A1 difference.
- **Epochs.** With a probe the epoch-text, DateFormat and `Epoch.<scale>` checks are replaced by the A1 comparison: |Δ| ≤ 1e-6 s is one start, anything more is refused until step 6. Without a probe they stay, and their messages say a console is needed.
- **Agreement on EarthMJ2000Eq templates.** On the organizer's workshop template, the four synthetic scripts, R1 to R5 and the organizer's first, withdrawn preliminary script, GMAT's state and our conversion differ by at most 1.1e-11 km and 3.5e-9 km/s. The velocity part is our mu 398600.4418 against GMAT's 398600.4415. Tolerances are 1e-6 km and 1e-8 km/s; beyond them the probed state is still used and a warning says how far apart they are. 4 T_A follows the probed state, 45 µs from the in-process value on the organizer's workshop template, so certify with a console refuses an archive solved without one.
- **The fixture.** SC_Human's probed state equals, bit for bit, the first row GMAT writes when it propagates SC_Human, at Precision 17. |r| is 6871 km and Δ is 54000 s.
- **Data files.** `GmatRunResult.data_file_sha256` hashes every existing file a `*_FILE` startup entry names, leaving out the 21 that only the GUI reads (27 remain on R2026a, 153 MB, hashed once per process). It also hashes every file a script field ending in File, FileName, Filename or Kernel names (`script:<Object>.<Field>`), looked up as GMAT does: absolute, then relative to the executable's directory, then in the startup paths. R2026a read `Earth.EopFileName` that way and refused a bare name. A name found nowhere is recorded as "not found" and warned. certify writes them as `gmat_data_files`. It refuses unless every run behind one certificate names the same files with the same hashes; the probe may name fewer.
- **Fake console.** It answers the probe for EarthMJ2000Eq Keplerian/Cartesian craft, with GMAT's Earth mu and a leap-second table from 1999, clears live ReportFiles as GMAT does, and fails for anything else. `% FAKE_GMAT_PROBE_FAIL` makes the probe fail.

## Step 3 as built

- **Model.** `Scenario.alignment` is an `EpochAlignment`: the early and late craft, the propagator, the form (`epoch-equality`, `epoch-value` or `elapsed`), the offset late minus early in probed A1 seconds, the line as written with its line ending, and its line index. The index is None for the alignment `policy.add_epoch_alignment` adds.
- **Reading it.** Only the first mission command is read as an alignment. A Propagate with an epoch goal anywhere else is refused, as is one that flies two craft or carries a second goal. OrbitColor and StopTolerance are allowed beside the goal, as on any other line. The goal must be on the craft it flies. The equality form must name the other craft in the same scale. The value form must be within 1e-6 s of the late craft's probed epoch in its scale, and bit-equal to it when the two start together, since a goal behind the craft is chased for ever. The elapsed form is an ordinary single-craft segment until the probed epochs say the craft start apart, and must then last Δ within 1e-6 s.
- **Δ.** Bit-equal A1 epochs with an alignment are one start and a zero-length alignment. Epochs within 1e-6 s but not equal are one start without an alignment and ambiguous with one. Apart without an alignment is refused, naming `policy.add_epoch_alignment` and Q4, with the consequence GMAT shows for the template's own Propagate form. Apart with one needs `policy.mission_time_origin` and `policy.closest_approach_window`, then stopped at `gmat_parser.ASYNCHRONOUS_START_ACCEPTED` until step 6 removed it.
- **The implicit segment.** A Maneuver directly after the alignment gives a `PropagateSegment` with `implicit` true, duration 0, no spans, the Maneuver's line index and epoch lock `instant`. The renderer fixes slot 0 at t = 0 on a template written that way and refuses a candidate that moves it. On an open template with an alignment it may write a burn at exactly t = 0 as a Maneuver right after the alignment; a burn inside `min_propagate_duration_s` but not at 0 is still refused. The policy guard exempts the implicit segment from the floor, and on a template that maneuvers it refuses a script whose segments are not implicit where the original's are, since a dropped Propagate would otherwise read as one.
- **The guard.** The original's alignment must reach the final script byte for byte; an added one must be exactly the A1 equality form on the propagator that flies the early craft in the first Propagate, as the first mission command, and only under `policy.add_epoch_alignment`. A script that adds an alignment the original does not need is refused.
- **The coast run** keeps the alignment line verbatim and skips the implicit segment (#112).
- **Measured on R2026a** (`output/async/s3forms`, `output/async/s3t0`): the value forms in A1 and UTC at Δ = 0 end at once, with rc 0 and 601 rows for a 600 s joint segment, as does the A1 equality form. At Δ = 54000 s the A1 equality form wrote 54601 rows with one row at the late epoch, followed by a Maneuver or not. With a Maneuver right after the alignment, the one row at t = 0 is the pre-burn state, at Δ = 0 and at Δ = 54000 s, which is what `burn_rows.with_post_burn_rows` assumes. A Maneuver as the very first command, with no alignment, writes a post-burn row at t = 0 instead; the parser does not accept that shape.
- **Left for later steps.** The search does not yet propose a burn at t = 0 after an alignment (step 6); `slot_assignment_variants` already moves a single burn off a pinned slot 0. Δ > 0 is refused to every caller outside the tests.

## Step 4 as built

- **Diagnostic columns.** The sibling reports both craft's `A1ModJulian` after their ElapsedSecs, at Precision 17. The fake console writes them, bit-exact at 17 digits, and on a script whose first command is an alignment it writes four rows first with the late craft waiting at ElapsedSecs 0.
- **The clock.** `report_parser.report_clock` checks the two A1 columns within 1e-6 s on every row. Only leading rows may differ, and only after an alignment with Δ > 0, during which the late craft's ElapsedSecs and A1 epoch must not move. Those are the frozen rows. Time is the late craft's ElapsedSecs from the first row on one epoch; before it, the early craft's less its ElapsedSecs there, so negative. Without the A1 columns it falls back to comparing ElapsedSecs, and refuses a report of an alignment with Δ > 0.
- **What reads which rows.** Entry, closest approach and the stop ignore the frozen rows, in verify and in the corrector. The geocentric radius check covers the frozen rows when the pursuer is the early craft, and every row to the end of the report: it no longer stops at the interception. `ephemeris_from_report` starts at t = 0. The coast run, which flies the target alone after the alignment, starts its rows where the target's A1 epoch reaches the waiting pursuer's.
- **Certify** records `frozen_rows` in certification.json. It refuses when an official report has a different number of rows from the final script and from its diagnostic sibling, or from the sibling's diagnostic report, whose rows `frozen_rows` counts. Rows are counted as non-empty lines less the header that `WriteHeaders` (GMAT's default true) says is there; a Variable column's header is a bare name, so the header cannot be told from its tokens. `scripts/check_report_rows.py` takes `--frozen-rows` and `--time-column`, and prints the time column it read.
- **The size cap.** `run_gmat` starts GmatConsole with Popen in its own session, through `setpriv --pdeathsig KILL` so it dies with the Python that ran it, and through `prlimit`, which sets RLIMIT_FSIZE and RLIMIT_CORE = 0 before exec, never above a hard limit already in force. Set on the pid after Popen returned, the cap missed a wrapper's child in 24 of 200 runs of a probe. `prlimit` is probed once; without it the cap goes on the pid as before and a warning says a child forked first has none. The cap is 20 times the rows the scenario's span takes at its smallest MaxStep or InitialStepSize, between 256 MiB and 2 GiB, or 2 GiB where no scenario is at hand. A run the kernel stops returns −SIGXFSZ (153 through a shell), the error says so, and its torn report is not kept as partial. A timeout kills the whole session and reads the pipes with their own bound. `setpriv` is probed once; without a working `--pdeathsig` a warning says a killed run outlives us. `prlimit`, `resource.prlimit` and `setpriv` are Linux's, and so is this runner. An official report with a whole Array column is written on several lines per row, so its rows are not compared; the byte comparison still holds.
- **One start, not bit-equal.** A template whose epochs are within 1e-6 s but not bit-equal is one start to the parser. On a split `P(A) P(B)` Propagate the A1 gap is that offset plus up to about 0.3 µs of stepping, so its report can now be refused where the ElapsedSecs check passed it. That is the A1 check doing its job.
- **"Epochs do not match"** is not a fatal marker. Measured on R2026a: an alignment that ended 1, 2, 3 or 6 A1 double steps past the late epoch, followed by two joint segments, printed it with rc 0. The A1 check refuses the 6-step case (1.9 µs); the others are inside the tolerance.
- **End to end on R2026a** (`tests/fixtures/async/aligned_crossing.script`, SC_Alien 5400 s before SC_Human, a UTC equality alignment, tracks crossing 1500 s in at 13.2 m/s): verify measures the zero burn's entry at 1109.25 s on the solver clock, with 540 frozen rows at MaxStep 10 s. certify replays the final script three times, the official report has the same rows from the final script and the sibling, and `check_report_rows.py --frozen-rows 540` finds the last row inside, at 3.5 km: the default 1.5 km `policy.stop_depth_km`, reached 121.5 s after the entry. The test set `ASYNCHRONOUS_START_ACCEPTED`, which step 6 removed; at this step every other caller still refused Δ > 0.

## Step 5 as built

- The diagnostic sibling also reports the pursuer's `Earth.Altitude`, GMAT's geodetic altitude, on the rows the sphere check covers.
- `EventResult.minimum_geodetic_altitude_km` is the lowest of them, read on rows only, as the organizer reads them. verify stores it in the candidate's metadata and its `gmat_event`; certify writes the final replay's as `flown_minimum_geodetic_altitude_km` beside `flown_minimum_altitude_km`.
- It gates nothing. A row whose geodetic altitude is under r less GMAT's equatorial radius by more than 1 mm is refused as a report whose columns are not what they say, since geodetic altitude is never lower than that. The radius is GMAT's 6378.1363 km, or `Earth.EquatorialRadius` where the script sets it, never `policy.central_body_radius_km`: with that at 6371 km the check refused every sound row. With a central body other than Earth the column is not the pursuer's altitude and nothing is checked.
- Measured on R2026a, on the aligned crossing at 51.6°: the certified replay flew down to 615.46 km above the sphere and 617.60 km geodetic.
- The fake console writes r − 6378.1363 km, which is the geodetic altitude on the equator where it flies.

## Step 6 as built

- **t = 0 from GMAT.** When Δ > 0 the parser raises `AlignedStatesNeeded` until the probe carries the alignment's end. `load_scenario` catches it and runs `state_probe.probe_alignment`: a private copy with the plots, ReportFiles and burns commented out, the force models and propagators kept, a ReportFile of both craft's A1 epoch and EarthMJ2000Eq state at Precision 17, and a mission that is the alignment line alone. Its last row is t = 0. The parse after it gives both craft that state as `initial_state`, so the search, the corrector's two-body seed, the renderer's frames and 4 T_A all start there. The early craft's state is GMAT's propagation, never two-body. On the aligned crossing it matches point-mass two-body to 1.8e-10 km.
- **Bound to the script.** `AlignedStates` carries `alignment_binding_sha256`: the alignment line, both craft's bindings, the force models and the propagators. A re-parse of a script that changes any of them is refused. The probe's files land in `state_probe/alignment/`.
- **The early pursuer.** No burn may come before t = 0 (Q2), so an alignment the pursuer flies below the surface makes the template infeasible and is refused. So is one under `policy.minimum_flown_altitude_km`, since verify measures every row from the first and would refuse each candidate.
- **The origin.** `Scenario.time_origin_offset_s` is Δ when the origin is the early craft, always so under `earliest`, else 0. `max_time_s` is T_max less it, and a Δ that reaches T_max from an early origin is refused, since nothing is left to fly. Scores read T_team as the solver time plus the offset: `score.on_solver_clock` moves Ct by it in `score_candidate`, `audit_burn_table.py` and `safe_delta_v_floor.py`. The ranking search adds the offset to its own time terms and leaves Ct alone, because its ordering goes through `score_candidate`, which moves it; moving both counted Δ twice and tied a 300 s hit with a 600 s one. `check_report_rows.py` does not add the offset: it reads the uploaded report's own time column. Only the later craft's ElapsedSecs is the solver clock, while the early craft's already counts from the origin. With a config whose origin is `earliest` it reads the certificate's `clock.time_origin_craft`, or without a certificate the ElapsedSecs that runs ahead of the other's, and refuses a time column naming the later craft. With neither, without `--time-column` it refuses an aligned report that carries both craft's ElapsedSecs, and prints no score lines at all. With `--time-column` it reads that column as T_team, and when the alignment's rows are in the report it notes that this holds only if it is the origin craft's.
- **The window.** Under `policy.closest_approach_window: from_origin` with the origin the early craft, the closest the moving early craft comes to the waiting late one on the alignment's rows, `Scenario.alignment_minimum_distance_km`, counts toward the scored closest approach; with head-start burns, the candidate's own (`score.HEAD_START_ALIGNMENT_KEY`). Under `after_alignment` d_min is the joint rows' alone, from t = 0, with or without head-start burns. With the origin the later craft those rows come before t = 0 and `from_origin` is `after_alignment`. Whether a craft still waiting at its insertion epoch is anywhere to be close to is for the organizer (Q1, Q5). An approach there only counts as distance: a miss within 5 km of it scores the full distance term, but completes at T_max, not as an interception.
- **Stamps.** solve and the ranking search stamp `time_origin_offset_s` and `alignment_minimum_distance_km` beside `horizon_s` (`models.clock_stamps`); rank and certify refuse an archive stamped with others, or with none when the origin is the early craft. On a template that maneuvers right after its alignment the ranking search pins slot 0 at t = 0, in its bounds, its repair and its walls, as the renderer fixes it there. With `policy.allow_zero_vector_unused_slots` on, a later slot the optimizer empties stays as an exact zero in its slot rather than being dropped, which would move the burn at t = 0 to a later slot; when the schedule is trimmed at the entry, the empty slots take the earliest places a separation clear of their neighbours before it, no kept burn moves, and a schedule with no room is refused. The GMAT corrector moves the last two impulses that carry one, never an empty slot nearer arrival. With the setting off the renderer refuses any exact zero, as the final-script guard does. rank's report and the certificate carry a `clock` block (Δ, the origin, the offset; rank also the probe's report hash, the certificate T_team).
- **The searches** may place the first burn at t = 0 after an alignment on an open template (`models.burn_may_follow_alignment`): the ranking and multi-burn searches drop their first-burn floor to 0 there. That reads rule 2.2, a first maneuver at mission start, as allowed at the later craft's insertion; Q2 asks. The ranking search's repair and the multi-burn search's unpacking put a first burn there at 0 or past `min_propagate_duration_s` plus 1 µs, never between, and the ranking search's walls refuse one between.
- **The fake console** answers the alignment probe with the early craft held where it started, so its t = 0 state is its initial one; only real GMAT moves it, and only the gmat-marked tests check the geometry at t = 0. The certificate's `clock.t_team_s` is T_max for a miss. `measure_ranking_scenario.py` measures synchronous templates and says so for an asynchronous one.
- **End to end on R2026a** (`tests/integration/test_solver_clock_gmat.py`): solve and certify on the aligned crossing, SC_Alien first and, with the geometry mirrored, SC_Human first. On the first, solve's best burned at 0.001 s, hit at 71.44 s on the solver clock (T_team 5471.44 s from SC_Alien's epoch), certified with 540 frozen rows and the same score on replay.

## Step 7 as built

- **Rehearsals R6-R8** (`examples/ranking/`, table `async_scenarios.yaml`, README section), built by `scripts/build_async_rehearsals.py` from R1 and R2 on GMAT R2026a: one craft's epoch 1800 s earlier, its state where the section 7 model flies it back to. At the later epoch each is its source's geometry to within 5e-9 km as measured; `tests/integration/test_async_rehearsals_gmat.py` holds it to 1e-6 km and 1e-9 km/s. That holds for the data files the build read, recorded in `built_with`: R6's EarthFixed insertion and R8's flight back move with Earth's orientation (3.8 m for R6 with UT1-UTC 10 ms off), so the test refuses other files and asks for a rebuild. Their official ReportFile1 also carries `SC_Human.Earth.Altitude` first, the column the organizer said it will add.
  - R6: SC_Alien first; SC_Human inserted from an EarthFixed SphericalAZFPA state, R1's state written the tutorial's way. It lies at longitude -95.3° and latitude -39.2°, not over Taiwan as §6 planned; the form is what is rehearsed.
  - R7: R2 with SC_Alien first and no alignment; `policy.add_epoch_alignment` adds the A1 form. Only a near miss is reachable, as in R2.
  - R8: SC_Human first, flying the alignment alone 421 km up.
- **`scripts/rehearse_ranking.py`** runs R1-R8, R6-R8 with their rows' policy, and `--certify --team-number <n>` certifies each best.
- **`scripts/ood_fuzz.py`** adds `--alignment-cases` (default 300): random alignment lines in the three forms, ElapsedDays included, against random probed epochs on the workshop template, judged by the rules of §2 as they read and by the reason given. Offsets include 1 to 4 A1 double steps either side of 1e-6 s; value-form errors are 0 to 5 double steps either way (three, 0.94 µs, are within 1e-6 s and four are not), elapsed-form errors ±0.4, ±0.9, ±1.1 and ±3 µs. 2000 cases on each of three seeds: about a third accepted, no disagreement.
- **Documents.** COMPETITION_POLICY.md: Spacecraft A under the configured external forces, the alignment kept byte for byte, and the `add_epoch_alignment` note for a person to sign. ASSUMPTIONS_AND_GAPS.md: assumption 2 is the probe and the alignment, and the asynchronous questions join the blocking gaps.
- **Measured on R2026a** on this code at the 1.5 km default `policy.stop_depth_km` (`output/async/reh678d`), `rehearse_ranking.py --certify R1 R6 R7 R8`, placeholder coefficients (kt 0.003 /s, Ct 0.8 T_max, kv 1.2 per km/s, Cv 2.0 km/s), every best `public_gmat_verified` and certified, replayed score equal; wall times on a machine shared with other jobs:

  | | best | from | frozen rows | T_team from SC_Alien's epoch | wall |
  |---|---:|---|---:|---:|---:|
  | R1 | 96.39 | one-burn sweep, corrected | 0 | 4301.44 s | 213 s |
  | R6 | 95.07 | ranking search, corrected | 1800 | 5911.53 s | 296 s |
  | R7 | 23.18 | zero burn, scored miss | 1800 | 7591.38 s (T_max) | 41 s |
  | R8 | 96.39 | one-burn sweep, corrected | 1800 | 4301.44 s | 297 s |

  At the old 50 m depth (`output/async/reh678c`) the same bests completed 0.40 s (R1, R8) and 0.34 s (R6) sooner and scored 0.0011 and 0.0015 more; R7, a miss, is unchanged. R8 is R1 to 1e-11 points (96.39083960116564 against 96.39083960116974): the same geometry on the same clock, the 1800 s alignment flown by the pursuer alone. R6 scores under R1 only because Ct is 0.8 of a T_max 1800 s longer. These runs took R8's T_team from SC_Alien's epoch, the later one, which the organizer's answer rules out: on SC_Human's clock the same certified report ends at 6101.44 s and scores 74.87 under the same coefficients (E2E audit A2, 2026-11-04).

## Time-origin evidence

Under the ranking rules the origin is `earliest`, so the evidence now checks T_max: a verdict naming the later craft's reading says T_max was most likely entered from that epoch. When the craft start apart, `orbitwin inspect` prints `time origin` lines on stderr and writes them, with the figures behind them, under `time_origin_evidence`; solve keeps the same record in summary.json, prints the lines, and adds the verdict to its warnings unless it names the reading the configured origin resolves to, and the track line when the coast failed. Nothing switches the setting (`time_origin.py`). The evidence speaks to T_max's origin; C_t and T_team follow the same setting, on the reading that the organizer counts all three from one epoch. If T_max is announced as an absolute epoch, converting it to `policy.horizon_s` already picks an origin, and the evidence then only checks that arithmetic.

- **Template.** The joint Propagate's goal counts from that Propagate (§4), so the template's goals after the alignment, summed, end Δ later from the early craft's epoch than from the late one's. Each is compared with `policy.horizon_s`, the announced T_max. The tolerance is the rounding of the numbers as written: half the last digit of each goal as the script writes it and of T_max as the config holds it (`8512.000` reads as `8512`), each held to at most 0.5 s, plus 1 µs for Δ (1.001 ms on R6). Holding a coarser number to a second only narrows the tolerance, so it can lose a match and never makes one; whether the check can tell readings Δ apart at all is judged on the numbers as written, so a goal of `6.0e3` (to the hundred seconds) picks nothing when Δ is 100 s. When that rounding reaches Δ/2 the line says the match is not counted. A goal unrelated to T_max, like the preliminary template's fixed 12000 s, matches neither. Only a match on the early craft's reading counts. A goal of T_max itself is what an author writes who counts from the later epoch, and also one who counts from the early epoch and does not know the goal counts from its own Propagate; only a goal short of T_max by Δ says which was meant. When the template has no alignment and `policy.add_epoch_alignment` writes ours (R7), the goal was written for a script without one, so where it ends against our line is our arithmetic: the line reports it and says it is not counted.
- **Track.** One more GMAT run flies SC_Alien alone, as the coast run does, from its own epoch (the alignment's rows kept when it is the early craft) to Δ/2 past T_max from the later craft's epoch, the later of the two readings' ends. Each reading's mission starts at its own craft's epoch, so its perigee and its climb out are looked for from there on, or from SC_Alien's epoch when that is later: a closed SC_Alien with a period under Δ passes both during the alignment too, and one already beyond the boundary there climbs out again only after its next perigee. A crossing that picks a reading lies within its tolerance, under Δ/2, of its T_max, so the extra Δ/2 holds every one (one cut off reads as matching neither, which only gives undetermined); T_max is announced rounded, and on R8 GMAT's SC_Alien reached the boundary 0.2 ms after 6711.667 s, past the end of a coast that stopped there. Nothing is two-body; the craft starting apart already needs a console, and a coast that fails leaves the track unread, which the line says. Perigee is where the radial rate turns from negative to positive, Hermite between rows. The crossing is found the same way, so its error grows with the row spacing: on R6's track at 17000 km, rows 600 s apart put it up to 51 ms off and rows 2700 s apart (DefaultProp's MaxStep) up to 21 s, where GMAT's RK89 writes them about 450 s apart. An error past the tolerance reads as matching neither, so undetermined, as long as it stays under Δ less the tolerance; with a head start of tens of seconds and rows that coarse it could match the other reading. At each reading's T_max the line gives SC_Alien's geocentric distance, whether it is past perigee and its radial rate. Rules §3 make T_max a climb past perigee, so a reading that lands before perigee or while SC_Alien is not climbing is ruled out, as is one whose T_max ends before the later craft's epoch, which the parser refuses as an origin anyway.
- **Boundary.** `policy.mission_boundary_km` (optional, `ranking_day_config.py --mission-boundary-km`; the parser's two-body T_max check reads it too) is the radius if the organizer announces it. The track gives the time SC_Alien, past perigee, first climbs through it; counted from each craft's epoch it is compared with T_max. The tolerance is T_max's rounding, plus the radius's rounding and 1 km over the radial rate there, plus 1 µs. T_max and the radius reach the check as numbers from the config, exact to their last digit, or to the unit when whole: a radius announced to the thousand kilometres is still read to half a kilometre. It takes T_max as exact to its last written digit; one read off report rows a step late reads as matching neither, which gives undetermined, as long as the step is well under Δ (MaxStep is 1 s on the rehearsals). The 1 km covers another GMAT build or data set: on R4, the rehearsal with the lowest perigee, drag moves SC_Alien 3.6 m by T_max and SRP 1.2 m more.
- **Verdict.** A reading the template or the boundary picks, and the track does not rule out. The two picking different readings, the picked one ruled out, or a boundary set and read that matches neither reading or only the other is `undetermined`, as is no pick at all. A boundary whose tolerance reaches Δ/2 picks nothing but can still veto this way. With no pick and one reading ruled out, the other is the verdict "by elimination". The line ends with Δ, T_max and each reading's figures.

Measured on R2026a with `orbitwin inspect` (`output/fix/origin/runs/snap9`), 7 to 9 s a run, with the probe, the alignment probe and the coast. "From SC_Human" announces the same scenario's T_max from SC_Human's epoch instead, Δ = 1800 s away; R6-12000 is R6 with its joint goal set to 12000 s.

| template | T_max (s) | boundary (km) | template | boundary, crossing from SC_Alien's epoch | verdict |
|---|---:|---:|---|---|---|
| R6 | 8511.667 | none | target | not set | target |
| R6 | 8511.667 | 17000 | target | target, 8511.6672 s (tolerance 0.268 s) | target |
| R7, alignment added | 7591.381 | 21000 | target, not counted | target, 7591.3813 s (0.26 s) | target |
| R8 | 6711.667 | 17000 | target, not counted | target, 6711.6672 s | target |
| R6 from SC_Human | 6711.667 | 17000 | pursuer, not counted | pursuer, 8511.6672 s | pursuer |
| R8 from SC_Human | 8511.667 | 17000 | pursuer | pursuer, 6711.6672 s | pursuer |
| R6-12000 | 8511.667 | 17000 | neither | target | target |

With `policy.mission_time_origin: target` on R6 from SC_Human the verdict line says the setting is not supported and that nothing was switched. A quick solve of R6 (20 s, no search) carried the same lines and record in summary.json.

## Other state types and epoch formats

The tutorial's SphericalAZFPA is spoken, not a rule. Measured on GMAT R2026a
(`tests/integration/test_state_types_gmat.py`): GMAT writes R6's two craft, at
02:00:00 UTC, in each CoordinateSystem and DisplayStateType at Precision 17; the
rewritten script is probed and compared with the original's probe.

- **Read, and parsed with the probe.** EarthMJ2000Eq and EarthFixed, for both
  craft wherever GMAT defines the type: Cartesian, Keplerian, ModifiedKeplerian,
  SphericalAZFPA, SphericalRADEC, Equinoctial, AlternateEquinoctial,
  ModifiedEquinoctial, Delaunay, Outgoing/IncomingAsymptote and BrouwerMeanShort/Long;
  Planetodetic in EarthFixed. The epoch is unchanged and the state lands within
  1.4e-11 km for SC_Human and 2.6e-8 km, 2.3e-12 km/s for SC_Alien, a 1e5 km
  hyperbola, through EarthFixed Keplerian; at the 16 digits GMAT writes in a
  saved script, within 5.5e-8 km. GMAT's iterative conversions round trip less
  well: Planetodetic 1.06e-4 km, BrouwerMeanLong 2.9e-5 km, BrouwerMeanShort
  1.0e-8 km. That is GMAT writing a value and reading it back; the probe is what
  GMAT flies from the template's own digits either way.
- **GMAT reads AlternateEquinoctial by line order.** The six values GMAT writes,
  set in alphabetical order with SMA last, put SC_Human 262 km from where GMAT's
  own order puts it. The probe still equals GMAT's first propagated row, bit for
  bit: it reports what GMAT flies from the lines in the order written, and the
  policy guard keeps every craft line as written.
  Keplerian and Cartesian read the same in any order, so the in-process reader,
  which ignores it, agrees; it also agrees within 2e-12 km on circular and
  equatorial elements, where AOP or RAAN is undefined.
- **`AnomalyType = MA`** is accepted by GMAT and puts SC_Alien's six Keplerian
  fields 5.6e5 km from where TA puts them; `AnomalyType = TA` changes nothing.
  `Element1` and `CartesianX` beside SC_Alien's Keplerian six are read into the
  state too, moving it 1.5e4 and 2.6e4 km.
- **GMAT refuses, so the probe fails with GMAT's words.** Planetodetic without
  body-fixed axes ("only defined for coordinate systems with BodyFixed axes");
  fields of two state types before BeginMissionSequence ("not contained in the
  same state type"); MA, EA or HA beside or instead of TA ("anomaly of type other
  than True Anomaly"); the epoch fields `A1ModJulian` ... `TDBGregorian` and
  `CurrA1MJD` set before BeginMissionSequence ("is not permitted"), although
  SpacecraftOrbitState.html sets `UTCGregorian` there in an example; a state
  type in the wrong case ("Unknown state element representation: keplerian").
- **GMAT cannot write.** Equinoctial, AlternateEquinoctial, Delaunay and
  Brouwer for the hyperbolic SC_Alien, and the asymptotes for SC_Human in
  EarthMJ2000Eq, where it is circular. A template cannot carry what GMAT does
  not define.
- **Epoch formats.** A1/UTC/TAI/TT/TDB ModJulian, quoted, unquoted or as
  `Epoch.<scale>ModJulian`, give back the probed A1 epoch and state bit for bit
  from the shortest repr. The 16-digit UTCModJulian lands one A1 double step
  (0.31 µs) off, which moves EarthFixed SC_Human 1.2e-7 km, and TAI/TT Gregorian
  likewise: one start. A1 and TDB Gregorian are not whole milliseconds from UTC,
  so the printed text lands 0.38 and 0.37 ms off and the craft start apart: the
  parser treats the pair as any asynchronous start, not as one start.
- **The in-process reader** (no console) read only its DisplayStateType's six
  fields and ignored the rest. It read `AnomalyType = MA`, `Element1` and
  `CartesianX` far from GMAT (it now refuses all of `Element1..6` and
  `CartesianX..VZ`), and accepted craft GMAT refuses: a second type's
  field, an MA beside TA, an epoch field set before BeginMissionSequence,
  `keplerian`. It now refuses each. It already refused the Keplerian or
  Cartesian fields of a craft whose DisplayStateType names another type, which
  GMAT reads as their own type, as a missing field; it now says so. It also
  refuses a state or epoch field, or `AnomalyType`, `DisplayStateType`,
  `CoordinateSystem`, `DateFormat` or `Epoch`, spelled in the wrong case: GMAT
  refused `ta`, `Sma`, `element1`, `utcmodjulian` and those five in lower case.
  It still ignores a field it does not know; GMAT refused the ones measured
  (`ElapsedSecs`, and a value for `EarthMJ2000Eq.X`), and then so does any GMAT
  run of the script. A craft field GMAT reads into the state and that is not
  listed here would be read wrong without a console.
- **Downstream.** Nothing past the parser reads the state type: the policy guard
  compares each craft's assignments, and the diagnostic sibling and reports carry
  EarthMJ2000Eq columns. One end-to-end run solves and verifies an EarthFixed
  Planetodetic SC_Human beside a ModifiedEquinoctial SC_Alien, and certify takes
  it to `public_gmat_verified`.

## Steps, one PR each, each reviewed adversarially until clean

1. **This document**, revised, then re-reviewed until clean.
2. **State probe:**
   - Precision 17, all time-scale columns, bit-exact Δ;
   - `SpacecraftSpec` carrying probed epoch and inertial state;
   - EarthFixed and SphericalAZFPA accepted;
   - fake-console probe support;
   - data-file hashes in the manifest.

   The epoch-text checks are replaced, and |Δ| ≤ 1e-6 s without an alignment counts as synchronous. Δ > 0 is still refused until step 6 lands. A test runs the probe on the workshop template, which has OpenFrames objects.
3. **Alignment:**
   - `Scenario.alignment` with its propagator and its three forms;
   - the direction check;
   - bit-equal Δ = 0, and refusing a non-zero Δ within 1e-6 s;
   - the counting rule for a burn right after the alignment, with its implicit zero-length segment exempt from `min_propagate_duration_s`;
   - refusing a split-form joint Propagate with Δ > 0 and no alignment;
   - the renderer leaving the alignment untouched;
   - `add_epoch_alignment`, default off;
   - the policy guard;
   - `mission_time_origin` and `closest_approach_window`, still refused when missing.
4. **Reports:**
   - the diagnostic columns at Precision 17;
   - A1 synchronisation;
   - relative quantities ignoring alignment rows;
   - the sphere check over the whole flown arc;
   - the fake console writing frozen rows;
   - `check_report_rows.py` with the frozen-row count from certify;
   - row-count comparison;
   - the report-size cap;
   - the non-fatal message.

   Then verify and certify end to end on an asynchronous template on real GMAT, with Δ > 0 still refused to callers outside the tests.
5. **Geodetic Altitude** as a cross-check, recorded and reported.
6. **Shared solver clock** for the search, ephemeris, corrector, audit, rank and scoring, with time conversion by `mission_time_origin`. End-to-end hunt and certify on R6/R7/R8. Only now is Δ > 0 accepted, because before this step an early target would be seeded from a state Δ stale and scored on the wrong clock.
7. **Scenarios, fuzz and documents**, including the permission note for a person to sign off.

## Questions for the organizer

1. With different epochs, which instant is t = 0 for T_team, T_max and the window of d_min: SC_Alien's epoch, SC_Human's insertion, or something else? Answered for T_team, T_max and Ct (§3 above): the earlier epoch. Still open: whether d_min's window includes the alignment, whether T_max is announced as a duration or an absolute epoch, and whether the mission boundary is announced, as a geocentric radius.
2. May the first maneuver be at SC_Human's insertion epoch itself? If SC_Human starts earlier than SC_Alien:
   - may it maneuver before SC_Alien's epoch?
   - does an Earth crossing in that stretch disqualify?
3. Will the official ReportFile carry `Altitude`, and is it GMAT's default geodetic altitude? Is a crossing judged only on the report's rows, at the organizer's own step size?
4. Does the template include the epoch alignment, or do teams add it? If teams add it, which exact form and which propagator must they use?
5. Which report column does the validator read as time, and does it skip the rows written before the later craft's epoch?
6. Which GMAT version, and which data files, does the validator use: EOP, leap seconds, EGM96, the tide file, space weather, the SPAD files and the planetary ephemeris?
