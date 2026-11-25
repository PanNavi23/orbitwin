# Scoring logic and questions for you to confirm

[繁體中文](SCORING_STRATEGY_AND_OPEN_QUESTIONS.zh-TW.md)

The document has these parts: Part 0, the facts this pass recovered from the organizer's
original documents; Part 1, what the score formula decides; Part 2, the state of the design
decisions that follow from it; Part 3, the questions for you to answer or to confirm with the
organizer.

Each question says how many points it is worth and what we change if the answer goes the
other way. Write your answer after **Your answer:**.

---

## Part 0: facts newly confirmed in this pass

`ASSUMPTIONS_AND_GAPS.md` used to list the briefing deck as unreadable ("CJK cannot be
extracted"). In this pass, rendering the pages at 300 dpi and running `tesseract chi_tra` OCR
recovered all 11 pages. The text below quotes the deck; none of it is inferred.

### F1: the submission is two files, the script and the report (deck p.8, 「競賽流程」)

The slide title 「競賽流程」 means "competition procedure".

> a. 下載題目 Script。(Download the problem script.)
> b. 於 GMAT 中透過修改 Script 的 Spacecraft B 設定,對 Spacecraft B 進行軌道設計、機動與模擬。
> (In GMAT, design, maneuver and simulate Spacecraft B's orbit by changing Spacecraft B's
> settings in the script.)
> c. 軌道設計完成後,請將你的 Script 存檔,並至少模擬一次,以產生 Report。(When the orbit
> design is done, save your script and simulate it at least once to produce the report.)
> d. 將你所設計的 Script 及產生的 Report 上傳至規定網址。(Upload the script you designed and
> the report it produced to the given URL.)

You also confirmed that the submission is the .txt report together with the script file.

This changes what the report is for. The `OfficialReport.txt` we upload is the evidence the
score is computed from. For a failed mission `§4` scores `Δr(4TA)`; if the report covers only
77% of the window, that row does not exist. So PR #22 (propagate over the whole window) went
from a precaution to a requirement.

(This changed later. In the preliminary on 2026-09-05 the validator took the completion time
from the report's last row, so a hit whose script ran to 4TA was scored as completing at 4TA;
see Q3. Now only a miss runs to the end of the window: 4TA in the preliminary, the announced
T_max in the ranking round. A hit stops at the moment GMAT measures it first
`policy.stop_depth_km` inside the sphere. See Q10.)

### F2: in the preliminary, A is on a circular orbit (deck p.8, 「規則說明 b」)

「規則說明 b」 is item b of the rule notes.

> b. 在初賽中,Spacecraft A 將以一圓軌道繞行地球。(In the preliminary, Spacecraft A orbits the
> Earth on a circular orbit.)

So `TA` is well defined (see Q8). Deck p.9 says that in the ranking round A changes to a
hyperbolic flyby and that the final is a pursuit-evasion game. Neither affects the preliminary.

### F3: on site, the organizer's computers are used, and your own only for preliminary computation (deck p.10, 「其他 3」)

「其他 3」 is item 3 of the "other" notes.

> 3. 競賽過程中對太空船的指令下達與模擬,需使用主辦/承辦單位所準備的電腦。
>    但可自備電腦 / 筆電做先期模擬與運算。
> (During the contest, commands to the spacecraft and their simulation must use the computers
> the organizer or host prepares. Teams may bring their own computer or laptop for preliminary
> simulation and computation.)

This supports our policy that the final script must not depend on Python, an AI service or a
cloud service.

(Ranking round: on 2026-11-06 thc1006 decided that, as in the preliminary, all computation runs
on our workstation, and the uploaded script and the Team54 report are produced there as well.
No path copies numbers onto the organizer's computer by hand. See RUNBOOK.md, "Who runs
what".)

### F4: schedule (deck p.6, checked word by word against the original image)

| Date | Event |
|---|---|
| 2026.05.10 | Online briefing (mission deck released) |
| 05.11 – 08.21 | Online team registration (戰術小隊) |
| 06.27 | In-person GMAT training (Taipei Tech, 北科大) |
| 08.29 | Preliminary (Tamkang University, 淡江大學), TOP 8 ADVANCE |
| 11.09 | Final (Greater Tainan Convention and Exhibition Center, 大臺南會展中心) |

This part was written on 2026-08-20: the preliminary was 9 days away, and registration closed
the next day, 8/21.

### F5: the shape of the organizer's workshop template

```
BeginMissionSequence;
Propagate DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000, AlienSC.ElapsedSecs = 1000, ...};
Maneuver ImpulsiveBurn1(HumanSC);
Propagate DefaultProp(HumanSC) DefaultProp(AlienSC) {HumanSC.ElapsedSecs = 3899.5, AlienSC.ElapsedSecs = 3899.5, ...};
Maneuver ImpulsiveBurn2(HumanSC);
Propagate DefaultProp(HumanSC, AlienSC) {HumanSC.ElapsedSecs = 1000, AlienSC.ElapsedSecs = 1000, ...};
```

- Two burn slots and three propagate segments, alternating. This is what our parser assumes.
  Our parser was run on this organizer file and parsed it correctly (pursuer=HumanSC,
  target=AlienSC, slots=2, segs=3, Tmax=39808.1 s).
- The one file mixes two Propagate forms, `P(A, B)` and `P(A) P(B)`. Our parser accepts both.
- `ImpulsiveBurn*.CoordinateSystem = Local; .Origin = Earth; .Axes = VNB;` and
  `DecrementMass = false`, so ΔV is only the length of the vector, as in `§5`.
- The example is a Hohmann transfer (0.6388 / 0.5841 km/s, both in Element1 only) of 5899.5 s
  in total, and it stops short of 4TA. That fits the reading that on a success `Δrmin` is
  always 5 and nothing after the entry changes the score, but the example does not cover a
  failure.

### F6: the ReportFile columns of the organizer's example (see Q1; this is the largest open problem)

```
ReportFile1.Add = {HumanSC.A1ModJulian, HumanSC.EarthMJ2000Eq.X};
```

The report has the time and one X coordinate of the pursuer. It has no Y or Z, no velocity,
and nothing about the target craft. No relative distance can be computed from it, so a report
like this cannot show `Δr(t) ≤ 5 km`.

---

## Part 1: what the score formula decides

```
Score = 50·exp(−(Δrmin−5)/100) + 25/(1+e^{kt(Tteam−Ct)}) + 25/(1+e^{kv(ΔVteam−Cv)}) − ΣPn
Tteam = min(tint, 4TA)      Δrmin = max(Δr(Tteam), 5)
```

This is the preliminary's formula. For the ranking round's distance term, see 1.1a. The
ranking round's time and ΔV terms have the same form. There `Tteam` is `tint` for a hit and the
announced `Tmax` for a miss (eq. (1)), and `Δrmin` is the closest approach over `[0, Tmax]`
(eq. (2), (3)).

### 1.1 The distance term is nearly binary

On a success, `tint` is by definition the first time `Δr ≤ 5`, so `Δrmin ≡ 5` and the distance
term is always 50. On a failure the term reads `Δr(4TA)`:

| Δr(4TA) km | 5 | 10 | 25 | 50 | 74.3 | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| Distance points (of 50) | 50.00 | 47.56 | 40.94 | 31.88 | 25.00 | 19.34 | 7.11 | 0.35 | 0.00 |

For a failure to keep half the distance points, `Δr(4TA)` must be at most 74.3 km, which is
already a near interception. So a plan either intercepts, or its distance term is close to
zero. (This is the preliminary; for the ranking round see 1.1a.)

### 1.1a The ranking round's distance term (2026/11/04 revision)

On 2026-11-04 the organizer changed the distance term of §4 eq. (5), first by email and then in
the revised rules (`排位賽規則-revised-2026-11-04.pdf`, page header 「2026/11/04 修訂版」
(2026/11/04 revision), SHA-256
`1ded3ebbaebb658477643c0e05391b7ba2bc040ac30dd18f0297beb4dbbc9e7e`), to

```
D = 100 / (1 + exp[((Δr_min − 5)/C_r)²])      Δr_min = max(d_min,team, 5) (eq. 3)
```

The time term, the ΔV term and the deductions are unchanged. `C_r` is announced at the contest
together with `kt, Ct, kv, Cv`, and the revision states the units: T_team in s, ΔV_team in m/s,
Δr_min and C_r in km. Properties of D:

- On a hit D is exactly 50, as with the old term. `dD/dd = −D(1 − D/100)·2u/C_r` with
  `u = (d − 5)/C_r`, which is 0 at 5 km. The top of the curve is flat there, where the old term
  lost 0.5 points per km.
- D ≥ 49 for `d ≤ 5 + 0.2·C_r`, and D = 25 at `d = 5 + C_r·√ln3 ≈ 5 + 1.048·C_r`.
- For `u² > 700`, D is 0 in floating point, and the code returns 0 there.

| d \ C_r | 50 | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|---|
| 25 km | 46.01 | 49.00 | 49.89 | 49.99 | 50.00 |
| 100 km | 2.63 | 28.85 | 47.50 | 49.77 | 49.97 |
| 500 km | 0.00 | 0.00 | 6.17 | 43.90 | 49.32 |
| 1000 km | 0 | 0 | 0 | 27.09 | 47.25 |
| 4711 km | 0 | 0 | 0 | 0 | 7.87 |

This changes the conclusion of 1.1. With C_r of a few hundred km or more, a near miss keeps
almost all 50 points, and a miss differs from a hit only in the time term (a miss has
T_team = T_max) and the ΔV term. R5's 88 km miss was worth about 22 points under the old term
and 48 at C_r = 300 km. With a small C_r the term comes close to the old rule of intercept or
get nearly nothing.

So two people check C_r. The flag is `--cr-km`, with the unit in its name.
`ranking_day_config.py` prints D at 10, 100, 1000 and 10000 km, at the two distances where D is
49 and 25, and, when `--zero-burn-dmin-km` is given, at the zero burn's closest approach. The
value of C_r alone does not show which unit it was typed in, so the tool checks only its range.
It prints a NOTE when D at 500 km is under 1 point (the organizer's revision email of
2026-11-04 says the distance term 「不會在數百公里內即降為零分」, does not fall to zero within a
few hundred km), and a WARNING when D at 10000 km is still over 49 points. The preliminary
rules keep the old term (`DistanceTerm` in `rules.py`).

### 1.2 A failure loses two terms at once

On a failure `Tteam = 4TA`, the largest value it can take, so the time term is at its lowest
too. Only the ΔV term is left, and it pays more the less ΔV a plan spends.

It follows that in the preliminary, when no interception is possible, the best plan spends no
ΔV at all. The zero-burn baseline scores about
`25/(1+e^{kv(0−Cv)}) + 50·exp(−(Δr₀(4TA)−5)/100) + time term`, where the ΔV term is close to
its full 25 because `Cv > 0`. A failed plan that spends ΔV loses on all three terms.

The ranking round differs. A miss has `Tteam = Tmax` and is scored on its closest approach
over `[0, Tmax]` (section 1.1a), so ΔV that takes it closer can pay for itself: on R2 at
C_r = 3000 km a searched miss scored 45.82 and the zero burn 31.54
(`ASSUMPTIONS_AND_GAPS.md`).

Our pipeline already ranks the zero-burn baseline as a regular candidate: `solve` in
`pipeline.py` starts its candidate list with `zero_burn_candidate`. In the preliminary, when
the scenario has no interception at all, that baseline is the only right submission. Under
the ranking rules it is one miss among others, and solve also replays in GMAT the best-scored
misses above it (`gmat.verify_misses`). This is implemented.

### 1.3 After a success, the announced coefficients set the exchange rate between time and ΔV

Both terms are logistic and both saturate. With the example values in
`config/score.example.yaml` (kt=0.001, Ct=5000 s, kv=0.004, Cv=800 m/s; an example, not a
forecast):

| Operating point | Points per 1 s | Points per 1 m/s | Exchange rate |
|---|---|---|---|
| T=5000 s, ΔV=800 m/s (both midpoints) | 0.00625 | 0.0250 | 1 m/s ≈ 4 s |
| T=2000 s, ΔV=400 m/s | 0.00113 | 0.0140 | 1 m/s ≈ 12.4 s |

Along one Pareto frontier the exchange rate can differ by a factor of 3 or more, and a
saturated term does not count at all. So until the coefficients are announced, the only correct
approach is to keep the whole (time, ΔV) frontier and to pick no point with made-up
coefficients.

`_prune_per_frontier` calls `prune_dominated`, which prunes by Pareto dominance and does not
read the score. Pruning on made-up scores would throw away the real winner. This is
implemented.

### 1.4 A penalty is a cliff

One violating maneuver costs 10 points, 40% of a whole logistic term. What a plan gains by
sitting closer to the 1500 m/s or 100 s limit is far less than what a wrong call at the limit
costs. The preliminary's production settings, 1.495 km/s per impulse and 101 s between burns,
are right.

In the ranking round the cliff is steeper. An impulse over 250 m/s (+ ε_v), or a total over
3 km/s, is a mission failure that is not ranked (§4 note 1(a), (c)); no deduction applies to it.
Only adjacent maneuvers less than 100 s apart deduct, 10 points per maneuver (note 1(d)).
`config/ranking.yaml` uses 249 m/s, 2.99 km/s and 101 s.

### 1.5 The first entry decides, and the closest approach only breaks ties

`§4` defines `tint` as the first time `Δr ≤ 5` holds. A graze at 4.9 km at t=1000 s beats a
pass at 0.1 km at t=2000 s: both get 50 on the distance term, and the first gets more on the
time term. The closest approach is used only to break a tie under `§6`. `ranking_key` sorts on
`-score` first, then in the `§6` order (minimum distance, ΔV, completion time). This is
implemented.

### 1.6 Going deeper is free, and it is the only way to lower the risk

Since the distance term is always 50 on a success, going deeper adds no points. Its only value
is lowering the risk that the organizer's validator does not see the entry (see Q3). A plan
that grazes the sphere at 4.999 km and one that reaches 1 km score the same and carry very
different risk.

The code now prefers depth explicitly, through `policy.stop_depth_km` (default 1.5 km). verify
stops a hit at the moment GMAT measures it first that deep inside the sphere. When any hit
reaches that depth, `prefer_deep_hits` (`constraints.py`, used by solve and rank) passes over
the hits that do not. certify, and `check_report_rows.py --config --certificate`, refuse a hit
whose last row is not that deep, unless `policy.allow_shallow_stop_submission` is set. The hand
form for a GUI report (`--script --verification`) prints a `WARNING:` for a shallower last row,
and fails it only when it is less than 1 mm inside. `samples_inside_radius` in
`event_detection.py` (the number of distinct samples inside the sphere) plays no part in the
ordering, and verify only warns when fewer than two samples are inside.

---

## Part 2: one design decision that follows (implemented)

### D1: a maneuver after the interception wastes its ΔV

`§5` eq. (1) is `ΔVteam = Σ_{k=1}^{N} ΔVk`, where N is 「總機動次數」 (the total number of
maneuvers). Read literally, every maneuver in the script counts, including those executed after
`tint`.

The two-burn branch first sets `intercept_time_s = arrival`, and its second burn comes before
`arrival`, so by construction there is no problem. After `refine_events`, though,
`intercept_time_s` is replaced by the actual first-entry time. If B has already swept into the
5 km sphere before its second burn, that burn fires after the interception.

Setting that burn to zero then:
- leaves `tint` unchanged (the trajectory up to `tint` is the same);
- leaves `Δrmin` unchanged (still 5);
- makes `ΔVteam` strictly smaller, so the score is strictly higher.

Under the opposite reading, where only maneuvers before the interception count, the zero
changes nothing. The trimmed plan is never worse under either reading and strictly better under
one, so it dominates.

This is implemented in the one- and two-burn searches and always applied. For a candidate whose
events were refined and that still fires after the interception,
`search.drop_post_intercept_burns` (the function in `search.py`) adds a variant without those
burns, whose branch name carries `post-intercept-trimmed`, and keeps the original candidate. The
variant is no worse under either reading, so it does not wait for the answer to Q4.

---

## Part 3: questions to confirm

### Q1: which columns will the official ReportFile write? (affects: whether the submission can be verified at all)

Rules: `§8` says 「所有結果以主辦單位驗證程式為準」 (all results are as the organizer's validator
decides). Deck p.8 asks for the script and the report both.

Current state: the organizer's example has
`ReportFile1.Add = {HumanSC.A1ModJulian, HumanSC.EarthMJ2000Eq.X}`, the time and the pursuer's
X only. `Δr(t)` cannot be computed from this report, because it has no Y or Z and nothing about
A.

There are three possibilities, and each needs a different approach from us:

1. The official script on the day has a fuller `.Add` (most likely). Then we change nothing and
   keep the current policy of not altering the official ReportFile contract.
2. Contestants are expected to add the columns they need to the ReportFile (if changing
   Spacecraft B's settings in the script, 「修改 Script 的 Spacecraft B 設定」 in deck p.8 b,
   covers the output settings). Then our design, which keeps the diagnostic columns out of
   the submitted script on purpose (`public_gmat_candidate.script` on the day), is the wrong
   way round, and the official report should carry the full A and B positions.
3. The organizer ignores our report and reruns our script (the validator, 「驗證程式」, in
   `§8` is their own). Then the report's content does not matter, and only the script's
   reproducibility counts.

**Your answer:**

---

### Q2: the values and units of `kt, Ct, kv, Cv` (affects: the 50 points that order the successful candidates)

Rules: `§5` says 「等參數數值與軌道分布狀況有關,在每次競賽前會公告」 (these parameter values
depend on the orbits and are announced before each contest).

Current state: we do not guess them. `config/score.example.yaml` is marked as an example.
Without a score block, the ordering falls back to the Pareto frontier and uses no made-up score,
which is correct.

We need the four values with their units. Is `Ct` in seconds or minutes? Is `Cv` in m/s or
km/s? `kt` is in 1 over that time unit. The example uses m/s. If the real unit is km/s,
`kv·ΔV` is off by a factor of 1000, the ΔV term saturates into a constant, and the term stops
ranking anything.

How many points these four values move, measured: one real solve of `DifferentPlane_Circular`
gave 120 successful candidates, 8 of them on the Pareto frontier. We swept 625 sets of
`kt ∈ {1e-4 … 1e-2}`, `Ct ∈ {2000 … 30000} s`, `kv ∈ {5e-4 … 2e-2}` and
`Cv ∈ {200 … 2000} m/s`:

| Frontier point | T | ΔV | Wins under how many coefficient sets |
| --- | --- | --- | --- |
| `C-6603b843…` | 26030.7 s | 543.9 m/s | 349/625 (55.8%) |
| `C-c917fb14…` | 13208.2 s | 785.3 m/s | 276/625 (44.2%) |
| The other 6 points | | | 0 |

- Betting on one candidate in advance loses up to 25 points in the worst case, the weight of a
  whole logistic term.
- The best single bet still loses 10.53 points in its worst case.
- Keeping the whole frontier and choosing after the announcement has a regret of 0 in every
  case.

So solve keeps the whole frontier, and `orbitwin rank` re-scores it in seconds under any
`(kt, Ct, kv, Cv)` announced on the day. The re-ranking does not replace a search after the
announcement. The preliminary's one- and two-burn enumeration finds transfers without reading
the coefficients, and on 2026-09-05 the announced coefficients made burns nearly free: the
best of the built-in search scored 74.997, and a search run again with the burn count and
spacing left free scored 98.319 (`LESSONS_2026-09-05.md` §3).

Ranking round (2026/11/04 revision): the units are stated. ΔV is in m/s and time in s, and
`C_r` is added, in km. Under the ranking rules the config refuses a score block with
`delta_v_unit: km/s` or without `cr_km`; under the preliminary rules it refuses `cr_km`. The
ranking search also picks its starting points by the coefficients, so it runs again after the
announcement (RUNBOOK step 5).

The command that re-ranks an archive (`orbitwin rank`; it does not search, and takes seconds):

```bash
orbitwin rank official.script \
  --config config/announced.yaml \
  --candidates output/gmat/candidates.json \
  --top 10
```

It prints the ranking and the winner's score term by term, and reports two problems that would
otherwise cost points without any message:

1. A term that separates no candidates. `term_spread_points` is a term's spread in points over
   the whole archive. A spread under 0.5 points means the term has no effect on the order. The
   most likely cause is a wrong unit: a `Cv` announced in km/s and entered as m/s puts every
   candidate on the same saturated tail, and no error message appears.
   Measured: with `config/score.example.yaml` (`Ct = 5000 s`) on a real archive, the time term
   spans only 0.007 points, so that example ranks by ΔV alone.
2. A winner on a knife edge. rank moves each announced coefficient by ±10% (four in the
   preliminary; five in the ranking round, with `C_r`) and, when the winner changes, names the
   coefficient. This checks the transcription. It does not forecast the announced values, which
   are exact. In one measured set of coefficients two candidates were 0.0003 points apart, and
   moving any one coefficient by 20% changed the winner.

`rank` only does arithmetic and does not write trial coefficients back onto the candidates. It
also prints the winner's `stage`, because a high score does not mean GMAT can replay the
candidate.

**Your answer:**

---

### Q3: how does the validator decide `Δr ≤ 5`, row by row or by interpolating between rows? (affects: success or failure itself)

Why it matters: this decides whether a grazing plan counts as a success at all. Our trajectory
may dip to 4.99 km as a continuous curve. If no report row falls inside the sphere at the
report's sampling interval, a row-by-row validator calls the plan a failure, and the distance
term drops from 50 points to near 0.

Our `event_detection.py` finds the first entry by cubic Hermite interpolation, so our test
accepts more than a row-by-row one. Where the two disagree, the disagreement goes against us.

What the answer changes: if the validator compares rows, the depth preference (section 1.6)
becomes a hard condition at once, with at least N rows required inside the sphere. If it
interpolates, the current code is enough.

A related question: what is the report sampling interval of the official script? Our example
has `Prop.MaxStep = 2`, which writes about 17000 rows over 33989 s. A larger MaxStep in the
official script, or a stride on its ReportFile, gives far fewer rows.

Partial answer (measured in the preliminary, 2026-09-05): the validator does read the report's
rows, and its distance is right. It returned D = 2.394 km, which matches the smallest row of our
report (t = 5982.992523 s) to the third decimal, and only one row of that run was inside the
sphere. So the row-by-row half holds: the validator finds a row inside the sphere without
interpolating.

It did not take the completion time from that row. The same reply gave
C = 23815.43 s = 4·T_A, the report's last row (the script ran the whole window). The only
combination of the three numbers that adds up is 50 + 0 + 23.0203 = 73.0203: the full distance
term and a time term of zero. The validator takes Tteam from the last row and the minimum
distance from all rows.

This disagrees with `Tteam = min(tint, 4TA)` in §4, but it is what the validator did, and §8
says 「所有結果以主辦單位驗證程式為準」. The response is `policy.mission_end_policy` (default
`"interception"`): a successful candidate's Mission Sequence ends on a row inside the sphere,
the first one GMAT measures `policy.stop_depth_km` deep (section 1.6), so the last row
itself is inside.

GMAT measures where the tail ends. Verify flies the plan past the predicted hit, finds where
GMAT's flight first reaches `policy.stop_depth_km` inside the sphere (deeper by what the
report's printed digits can move), and flies the submitted script again stopped there; the
last row of that flight must be inside the sphere. Before a measurement, the tail comes from
the row grid: GMAT writes a row every `Propagator.MaxStep` seconds from the last maneuver, and
the phase of that grid has nothing to do with the analytical `t_int`, so the tail extends to
the first row boundary after `t_int`, and to at least `policy.min_propagate_duration_s`.
`policy.interception_tail_s` is used only when MaxStep cannot be read, or when the hit comes
before the last burn. A fixed tail lands after the entry row: on the measured 6-burn
candidate, 1.0 s put the last row at 7.57 km and 2.0 s at 20.13 km, both outside the sphere.

A real GMAT R2026a replay confirmed that under the new policy the last two rows are both within
5 km (2.394 and 4.272 km), and both readings give 95.24–95.25. Before uploading, check that
`.txt` with `scripts/check_report_rows.py`.

The organizer's scoring also settled the units. They returned E = 6.996 and a total of 73.0203,
and `25/(1+e^{1.1862(6.996−9.0643)}) = 23.0203`, with `50 + 0 + 23.0203` exactly 73.0203. Had
Cv been read in m/s, the exponent would overflow, the term would be 0 and the total 50.0000.
In the preliminary, ΔV was in km/s and time in s.

The ranking round scores a miss on its closest approach (eq. 2, 3). So verify and certify take
a miss's distance as the smallest value over the report rows, the same reading as D above and
as `check_report_rows.py`. The Hermite minimum between rows, which is never larger, is kept in
`gmat_event.interpolated_minimum_distance_km` for diagnosis. With widely spaced rows the two
can differ by tens of km. Rows 60 s apart, a relative speed of 1 km/s and a 10 km graze give
31.6 against 10 km: 9.2 points apart under the old term, and 1.7 points under the 2026/11/04
revision at C_r = 100 km.
Hits are unaffected. Their distance term is always 50, and `minimum_distance_km` stays the
interpolated value, used only to break ties. Whether a tie reads the rows or the interpolation
is still unanswered; see the next paragraph.

Still unanswered: whether, at a lower sampling density, the validator requires several
consecutive rows inside (in that preliminary the official `MaxStep = 1`, the relative speed at
interception was 9.27 km/s, rows were 9.3 km apart, and only one row was inside); and whether
the `dmin,team` of the §6 tie-break is the minimum over the report rows or the interpolated
minimum.

**Your answer:**

---

### Q4: does `ΔVteam` count maneuvers executed after the interception, and does a zero-vector maneuver count as a maneuver? (affects: up to 25 points)

Rules: `§5` eq. (1) sums over `k=1..N`, and `N` is 「總機動次數」. Read literally, every maneuver
counts.

Two sub-questions:
1. Does a burn fired after `tint` count? (See D1: the variant without those burns is
   implemented and is no worse under either answer.)
2. Does an unused slot filled with an exact zero vector count as 「一次機動」 (one maneuver)? If
   it does, then
   (a) its `‖Δv‖ = 0 ≤ 1500` breaks no magnitude limit;
   (b) it is still a `Maneuver` command, so does 「兩次機動間需間隔至少 100 秒」 (two maneuvers
   must be at least 100 s apart) apply to it?
   Our renderer spaces unused slots as well, so we are safe for now, but that spacing is our
   own constraint, and the rules do not state it.

`ASSUMPTIONS_AND_GAPS.md` item 5 already records that if zero-vector maneuvers are disallowed,
the renderer policy changes only after written permission.

**Your answer:**

---

### Q5: what is `dmin,team` in `§6`? (affects: the first tie-break)

Rules: `§4` defines `Δrmin = max(Δr(Tteam), 5)`, while the first tie-break of `§6` is
「最小相對距離 `dmin,team`」 (minimum relative distance `dmin,team`). These are two different
symbols.

Our reading: if `dmin,team` were `Δrmin`, it would be 5 for every successful team, and the
first tie-break would decide nothing. So we read it as the actual closest approach over the
whole trajectory.

This reading is very likely right (the symbols differ, and a tie-break has to decide
something), but in the preliminary it is an inference; the rules do not say it. It is recorded
in `ASSUMPTIONS_AND_GAPS.md`. The ranking rules state it: eq. (2) defines
`dmin,team = min Δr(t)` (`0 ≤ t ≤ Tmax`), and it is the first tie-break of §5.

If the reading is right, the free depth (section 1.6) has one more real use: a deeper pass
also improves the tie-break.

**Your answer:**

---

### Q6: is a failed mission scored by the formula, or ranked separately? (affects: the choice of strategy)

Rules: the `§5` formula is defined for a failure (`Tteam = 4TA`), and `§7` says only
「依據初賽排名取前 8 名」 (the top 8 of the preliminary ranking advance).

Why it matters: in the preliminary, if a failure is scored by the formula, 1.2 holds, and the
zero-burn baseline is the best plan when no interception is possible. If every failure ranks
below every success, the zero-burn baseline is worth nothing, and we should put everything on
a plan that might succeed. The two strategies are opposite.

The ranking rules answer it: eq. (1)–(3) score a miss by the formula (`Tteam = Tmax`, with the
distance taken as the closest approach over `[0, Tmax]`), so the question is open for the
preliminary only. Under the ranking rules the best miss is the highest-scoring one, which need
not be the zero burn: on R2 at C_r = 3000 km a searched miss scored 45.82 against the zero
burn's 31.54 (section 1.2).

**Your answer:**

---

### Q7: how far does changing Spacecraft B's settings in the script reach? (affects: the solution space we may use)

Rules: deck p.8 b says
「於 GMAT 中透過修改 Script 的 Spacecraft B 設定,對 Spacecraft B 進行軌道設計、機動與模擬」
(glossed in F1). `§2` says only that B may make several impulsive maneuvers.

Current state (conservative): in a slotted template (its Mission Sequence already has a
`Maneuver`) we change only the existing `ImpulsiveBurn.Element1/2/3` and the existing
Propagates' `ElapsedSecs / ElapsedDays`, with two edits that apply to every template: the
team-number placeholder in the official report's Filename becomes the team number, and when
the craft start on different epochs and the template has no epoch alignment, one alignment
line is added as the first mission command (`policy.add_epoch_alignment`, signed off on
2026-11-03 in COMPETITION_POLICY.md). We add or delete no other burn or command and change no
coordinate system or Axes. An open template (no `Maneuver`) has no other way to maneuver, so
each impulse adds one `ImpulsiveBurn` and one `Maneuver` (COMPETITION_POLICY.md, "Unresolved
permissions").

To confirm:
- May we add `ImpulsiveBurn` resources and `Maneuver` commands? (`§2` allows 「多次」 (several)
  maneuvers, but the official script has only 2 slots. Is the slot count the limit on
  「多次」?)
- May we change `ImpulsiveBurn.Axes` (for example VNB → MJ2000Eq)? The physics stays the same
  and only the numbers that express a burn change, but our script would then differ in shape
  from the official example.
  The code has the option `policy.synthesized_burn_frame: EarthMJ2000Eq`, which affects only
  the burns we add and leaves the slots the organizer declares alone. It is off by default
  (VNB) until the organizer answers in writing. What it would gain: GMAT applies VNB on its own
  state, which differs from the state of the two-body plan. On the R4 rehearsal the impulse
  GMAT flew was 0.431 m/s off the plan in VNB, and at most 0.020 m/s in the inertial frame.
  Under VNB the GMAT correction step absorbs this direction error (GMAT_IN_THE_LOOP.md step 3,
  `gmat_shoot.py`), and solve already calls it (step 4, `gmat_loop.py`). The first
  `gmat.correct_top` hits inside the production margins (40 in the ranking profiles) have
  their last impulse corrected on GMAT's own miss. When changing the last impulse alone would
  break a safe cap, the last two share the correction. Those hits therefore lose the direction
  error. A miss is replayed as planned and keeps it.
- B's initial state and epoch cannot be changed (`§2`: the organizer sets them); this is not in
  question. The one exception is SC_Human's epoch under the onsite ranking rules
  (`ranking-onsite`), which a team sets with `orbitwin set-epoch` (COMPETITION_POLICY.md,
  "Unresolved permissions").

**Your answer:**

---

### Q8: the exact definition of `TA` (affects: `Tmax` itself, and where a failure is measured)

Rules: `§3` says 「設太空船 A 之軌道週期為 TA」 (let TA be the orbital period of spacecraft A).
Deck p.8 b: in the preliminary, A is on a circular orbit.

Current state: we use the two-body period of A's initial state, `2π√(a³/μ)` with
`μ = 398600.4418 km³/s²`.

Still uncertain:
- If the day's ForceModel includes J2 (our example is a pure point mass,
  `Degree = 0, Order = 0`), "period" can mean the osculating, the nodal or the mean period, and
  these differ by tens of seconds.
- Is `μ` GMAT's Earth `mu` or a value the rules give? The `mu` in GMAT R2026a's `GravityFile`
  need not equal 398600.4418. A Tmax off by 1 s measures a failure's `Δr(4TA)` at a different
  point.

This question is for the preliminary only. The ranking round has no `4TA`: `Tmax` is announced
at the start of the contest (§3). Under the ranking rules the code refuses to run without
`policy.horizon_s` and does not fall back to `4TA`.

**Your answer:**

---

### Q9: what the penalty `Pn` counts (affects: 10 points per misjudgment)

Rules: `§5` says 「每次違規機動扣 10 分,亦即 Pn = 10,直至總成績為 0 分」 (each violating
maneuver deducts 10 points, that is Pn = 10, until the total score is 0).

Current state: the code counts per violating maneuver. It used to count per violated
constraint, which charged a maneuver that was both over 1500 m/s and less than 100 s after the
previous one 20 points instead of 10. `ConstraintResult.rule_violations` now has one entry per
violating maneuver, listing all of that maneuver's reasons.

Still uncertain: does 「總成績為 0 分」 (the total score is 0) mean deducting down to 0 (our
implementation: `max(0, ...)`), or a score of 0 on any violation? We read it as the first.

Ranking round: only adjacent maneuvers less than 100 s apart deduct (§4 note 1(d),
「每次違規機動扣 10 分，直至總成績為 0 分」). Going over the per-impulse or the total limit is a
mission failure (note 1(a), (c)). The rules do not say which maneuver of a close pair is the
violating one. The code charges the later one; see `ASSUMPTIONS_AND_GAPS.md`. A candidate with
a gap under 101 s is never submitted, so this reading cannot reach an upload.

**Your answer:**

---

### Q10: may the script run to or past `4TA`, and is there a limit on the report's length? (affects: whether PR #22 is right)

Rules: `§3` says 「參賽者必須在此視窗內完成所有機動與攔截動作」 (contestants must complete all
maneuvers and the interception within this window). It constrains the maneuvers and the
interception, and says nothing about how long the propagation runs.

Current state: only a miss runs to the end of the window, to exactly `scenario.max_time_s` and
no further: `4TA` in the preliminary, the announced `Tmax` (`policy.horizon_s`) in the ranking
round. A hit stops at the moment GMAT measures it first `policy.stop_depth_km` inside the
sphere (`policy.mission_end_policy`, default `interception`). The reason: on 2026-09-05 the
validator took the completion time from the last row, and a hit that ran to 4TA lost 22.2266
points (Q3).

Before (PR #22): every submitted script propagated to exactly `4TA`. The reason was that `§4`
needs `Δr(4TA)` on a failure, and a report that stops at 77% has no such row (measured: cut at
26297.0 s against the full 33988.7 s). The cost was +0.76 s per GMAT run and 16998 report rows
instead of 13153. That reason now holds for a miss only.

The organizer's own example runs only to 5899.5 s (F5), short of 4TA. That fits the reading
that the end does not matter on a success, but it means our script's shape differs from the
example.

To confirm: (a) Is there a file size or row limit on the uploaded report? (b) Is a run to
exactly 4TA (ranking round: exactly Tmax) accepted?

**Your answer:**

---

### Q11: the GMAT version, OS and startup file on the organizer's computer (affects: reproducibility; under `§8` the organizer may cancel a result)

Rules: `§8` says 「若結果無法重現,主辦單位得取消其成績」 (if a result cannot be reproduced, the
organizer may cancel it). Deck p.10 item 3: on site, the computers the organizer prepares are
used.

Current state: we run `GmatConsole-R2026a` from GMAT R2026a locally.

To confirm: the version, build and OS, and the versions of `GravityFile` / `LeapSecondFile` /
`EOPFile` in the startup file. Each of these changes the numbers directly.

**Your answer:**

---

### Q12: floating-point tolerance and score rounding (affects: decisions at the limits)

Is `Δr = 5.0000001 km` a success? Is `‖Δv‖ = 1500.0000001 m/s` a violation? To how many
decimals is the score rounded?

Our safety margins (preliminary 1.495 km/s and 101 s; ranking round 249 m/s, 2.99 km/s and
101 s) are there so that this question decides nothing, but a tie-break uses exact values.
Ranking §4 note 1(a) gives the per-impulse tolerance `ε_v = 1e-5 m/s`. The code uses it only
to cap the config (`policy.dv_limit_km_s` may be at most 250 m/s + ε). The ranking profiles set
0.25 km/s, so under them an impulse over 250 m/s (with a comparison tolerance of 1e-12 km/s)
is invalid.

**Your answer:**

---

### Q13: how far may preliminary computation on our own computers go? (affects: whether our toolset is allowed at all)

Deck p.10 item 3 explicitly allows 「自備電腦 / 筆電做先期模擬與運算」 (preliminary simulation and
computation on your own computer or laptop), which covers our solver.

Still uncertain: may we bring our own computers on site and connect them? Is AI assistance
allowed? (Neither the rules nor the deck mention AI. Our policy is that the final script has no
runtime dependency on Python, an AI service or a cloud service, which is safe under any
answer.)

**Your answer:**

---

### Q14: is `Σ_k ΔV_k` in `§5` eq. (1) the sum of the impulse magnitudes, or the magnitude of the vector sum? (measured: at most 2.38 points for us)

Rules: `§5` eq. (1) writes `ΔV_team = Σ_k ΔV_k`, `k = 1..N`. The symbol `ΔV_k` is not defined
explicitly as `‖Δv_k‖`.

Our reading (reading A): `Candidate.total_dv_km_s` returns
`sum(burn.magnitude_km_s for burn in self.burns)`, the sum of the impulse magnitudes. That is
what propellant use means physically. It is also the only reading consistent with `§2`,
「每次脈衝的向量模長不得超過 1500 m/s」 (each impulse's vector magnitude must not exceed
1500 m/s), which already uses the magnitude of a single impulse.

The other reading (reading B): `‖Σ_k Δv_k‖`. Two maneuvers in opposite directions cancel.

How much it changes, measured (one real solve of `DifferentPlane_Circular`, 121 candidates):

- Only 6 of the 121 candidates have more than one impulse. A single-impulse candidate has one
  vector, and the two readings give the same number for it.
- All 8 points of the Pareto frontier are single-impulse, so under reading A the ambiguity has
  no effect on any plan we would submit now.
- For the 6 multi-impulse plans, though, `‖ΣΔv‖ / Σ‖Δv‖` is between 0.213 and 0.839. The
  extreme one, `C-d1600f339ae48d65`, goes from 2761.2 m/s to 588.3 m/s and would join reading
  B's frontier at `T = 21928.9 s`, in the gap between our frontier's points at 13208 s and
  26031 s.
- Re-ranked over the 625 `(kt, Ct, kv, Cv)` sets: if the organizer uses reading B, the best
  score from this archive rises by +0.06 points on average, +2.38 at most and +0.00 at least.

The error goes the safe way, and this can be proven. By the triangle inequality
`‖Σ Δv_k‖ ≤ Σ ‖Δv_k‖`, the ΔV we score with is never less than the organizer's under reading B.
So we can only underestimate our own score, never overestimate it. A submission is legal under
both readings: `§2` limits each impulse's magnitude, and the largest single impulse of the 6
multi-impulse plans was 1462.3 m/s, within 1500. No submission can break a rule through this
misreading.

The remaining risk is larger than those 2.38 points, which are what re-ranking the existing
archive is worth. If the organizer does use reading B, a solver that looks on purpose for
nearly opposite impulse pairs that cancel could find plans far cheaper than 588.3 m/s. Our
588.3 m/s was found by chance, because our two-burn search minimizes `Σ‖Δv‖`. The existing
archive cannot measure this exposure.

Our judgment and what the code does: the `‖Σ Δv_k‖` reading makes no physical sense (spent
propellant does not come back), so it is unlikely, and we do not change the search or the
pruning for it in advance. `Candidate.vector_sum_dv_km_s` is already computed and written to
`candidates.json`. If the answer is B, deciding whether to rerun the search is then a table
lookup, with no new measurement.

**Your answer:**

---

## Appendix: the evidence behind this document

| Source | How it was read | Reliability |
|---|---|---|
| The organizer's preliminary rules PDF (header 2026/05/20, received 2026-06-05) | Extracted directly with `pdftotext`, CJK complete | Original text |
| The organizer's 0510 online briefing deck PDF | Rendered at 300 dpi + `tesseract chi_tra` OCR; the schedule page also checked word by word against the original image | OCR, key sentences checked by hand |
| The organizer's workshop template | Read directly, and parsed successfully by this project's parser | Original text |

OCR can introduce wrong characters. The key sentences quoted above (F1 c/d, F2, F3, F5) were
checked character by character against the page images. Before you answer, check deck p.8 and
p.10 yourself once more.
