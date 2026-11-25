# Duel engine for the gold and bronze finals: design

Status: design only, 2026-11-02. Nothing here is implemented yet. Phases 1
to 4 below build it in order; each phase ends with its tests and a review.
The shared work phase 1 builds on has landed on main and is merged into
this branch:

- `orbitwin.rules.DUEL` (PR #115), where the rule constants come from
  (section 2.0), with `tests/cross_profile/test_import_boundaries.py`;
- the Stumpff fix (PR #116, section 2.3);
- `orbitwin.conic` (PR #117), the two-body kernels moved out of
  `ranking_search`, classed as core by the boundary test.

Source: `四強賽規則.pdf`, header `2026/10/29 初版`, 3 pages, sha256
`78c09cf72e7deb6ac3fcbc5f061964c9ac2c11f8f3bceb9497e616961ef94df4`. It is not in
the repository. Its text layer is clean (XeTeX output) and agrees with the
pages rendered at 110 dpi, which were read to check the formulas. The file's
CreationDate is 2026-11-01 10:41 CST, three days after the date in its
header, and PDF §6 says the latest revision on the official site governs. The
site must be checked for a newer version before the match.

The PDF numbers only its sections. "PDF §4.2" below means the second numbered
item of PDF §4. Plain "section 4.2" refers to this document.

The engine produces no GMAT script. The project's script rules
(`team_solution.script`, `assert_final_script_policy`, slotted burns) do not
apply to it. Its offline rule does, and PDF §6 makes it a condition of play.

## 1. The rules, clause by clause

| PDF § | PDF text | What the engine does with it |
|---|---|---|
| 1 | 金牌賽與銅牌賽採用相同規則，兩隊一對一對戰，以結果決定勝負，不計分。 | Outcome is binary. The game value is pursuer win = 1, evader win = 0. No score. |
| 1.2, 1.3 | 兩隊有 10 min 分析時間。排位賽成績較優的隊伍選擇角色 | Section 6.5 gives the role-choice procedure. |
| 2 | 逃脫方：全程不被攔截，並在比賽結束時抵達安全空域。 | Terminal condition on the evader, judged once (PDF §5.3). |
| 2 | 追捕方：在逃脫方抵達安全空域之前將其攔截。 | Together with the PDF §5 preamble below, this can be read as an early win for the evader. Section 3.6 explains why the engine follows PDF §5.3 instead, and how it shows the other reading. Q9. |
| 5 | 本賽以結果決定勝負，不計分，也不設同分判定。追捕方攔截成功即獲勝；逃脫方全程未被攔截且抵達安全空域即獲勝。 | No tie-break exists. The second sentence is the other half of the early-win reading (section 3.6). |
| 3 | 比賽共 6 回合，每回合策略擬定時間為 10 min，每回合模擬時間 Tround（當場公告）。 | Rounds and decision time come from `orbitwin.rules.DUEL`; `t_round_s` is a day parameter. Default compute deadline 7 min per round, our choice. |
| 3.1 | 雙方以目前的位置與速度為起點（第 1 回合為初始狀態，之後為上一回合結束時公告的狀態） | Every round starts from the announced state, never from our prediction. |
| 3.2 | 各自以書面提交本回合機動指令，彼此看不到對方的指令。 | Simultaneous moves. The recommender has no input for the opponent's current action (section 5.6). |
| 3.3 | 主辦單位於回合起點同時執行雙方機動，並模擬 Tround 時間。 | Both impulses at t = 0 of the round, then one coast of `t_round_s` each. |
| 3.4 | 公告本回合結果：雙方實際執行的 Δv、最近距離與發生時刻、是否攔截、逃脫方的半長軸，以及雙方太空船於回合結束時的位置與速度 | We learn both executed impulses, so the state is public and the game has perfect monitoring between rounds. The announced closest approach and the evader's a are compared with ours every round (section 6.4). The unit of the announced Δv is not stated (Q11). |
| 3.5 | 未分出勝負則進入下一回合，直到第 6 回合結束。 | The match ends at a capture, an Earth crossing, or after round 6. |
| 3 | 提交格式（每隊一行，單位 m/s，至少小數點後兩位）：角色, 回合, ΔvX, ΔvY, ΔvZ。 | Every action is snapped to the 0.01 m/s grid before it is evaluated and printed with exactly two decimals, so the line means the same however digits are treated (section 2.1). Role token and separator are asked in Q7. |
| 3 | 逾時未提交視為 [0, 0, 0]。 | The referee treats a missing submission as zero. The recommender holds a valid answer from its first seconds (section 6.2). |
| 4 | 雙方適用相同的機動約束 … 總速度增量不設上限。 | One `dv_lim` for both sides, no budget across rounds. |
| 4.1 | 每回合僅能在回合起點進行一次瞬時脈衝。 | One impulse per side per round, at the round start. |
| 4.2 | ‖Δv_k‖ ≤ ΔV_lim … 超過上限不視為違規，主辦單位依下式等比例縮至上限後照常執行 | Radial projection Π(u) = u · min(1, ΔV_lim/‖u‖), applied to every action and to every parsed submission. Never per-component clipping. |
| 4.3 | Δv 以 EarthMJ2000Eq 慣性座標表示 … 與公告的位置、速度使用同一座標系。 | Everything internal is EarthMJ2000Eq. Frame directions (V, N, B, R) are converted before projection. |
| 4.4 | 太空船軌跡不得穿越地球。 | Closed-form minimum radius per arc (section 3.3). |
| 4 | 本賽採用純二體模型：僅考慮地球點質量重力 … 賽事以 GMAT 進行模擬，相關參數請參閱 GMAT 的設定。 | Kepler propagation with GMAT's μ, measured in phase 1. |
| 5 | Δr(t) = ‖r_P(t) − r_E(t)‖ (1)，a = (2/‖r_E‖ − ‖v_E‖²/μ)⁻¹ (2) | Capture is on (1). The terminal test is on (2). |
| 5 | 主辦單位依下列順序判定 | Two readings of "order"; section 3.4. |
| 5.1 | 任一時刻 Δr(t) ≤ Rcap，追捕方獲勝，比賽立即結束。相對速度不設限。 | `≤`, at any time, no velocity condition. Whether "any time" is checked continuously is unknown (section 2.6). |
| 5.2 | 軌跡穿越地球的一方判失敗，比賽立即結束。 | The crossing side loses. Both sides crossing is not covered (Q4). |
| 5.3 | 第 6 回合結束時逃脫方未被攔截，且半長軸 a 落在安全空域內，逃脫方獲勝；否則追捕方獲勝。 | Judged once, on the evader's state at the end of round 6. |
| 6 | 每回合結果以 USB 提供電子檔，內含雙方位置（km）與速度（km/s） … 下一回合起點以電子檔數值為準。 | State reader with unit echo (section 6.3). The file's numbers are authoritative even where they differ from our prediction. |
| 6 | 比賽期間禁止使用網路及任何生成式 AI 工具，違者由主辦單位判定該場比賽失敗。賽前準備之程式與資料不在此限。 | A violation loses the match. Offline, numpy and scipy only, and the laptop itself prepared as in section 6.7. |
| 6 | 參賽隊員之手機及其他具通訊功能之個人裝置由主辦單位統一保管 | Phones are handed in. A laptop has radios too; section 6.7 and Q12. |
| 6 | 可使用主辦單位提供之電腦或自備筆電。 | Runs on our own laptop, with a second machine prepared as a spare. |
| 6 | 比賽時參賽隊員需全員到齊 | The whole team is present, which the paper read-back of section 6.6 relies on. |
| 6 | 所有判定以主辦單位驗證程式為準。 | Our referee predicts. It never overrides an announcement. |
| 6 | 若本競賽規則有修訂 … 以最新修訂結果為主。主辦單位保有最終解釋權。 | Re-check the rules before the match; the organizer's interpretation governs every open question below. |
| 7 | r0, v0 (EarthMJ2000Eq，含 epoch)，a 的範圍，ΔV_lim（雙方相同），Rcap (km)，Tround (s) | The parameter file, one per match since gold and bronze may differ. Units are stated only for R_cap and T_round; ΔV_lim and the interval are entered with an explicit unit and echoed (Q11). |

## 2. Model

### 2.0 Where the constants live

The three rule sets (preliminary, ranking, duel) are kept apart. What the
PDF fixes is defined once, in `orbitwin.rules.DUEL` (a frozen `DuelRules`,
each field with its clause), and the engine reads it from there with no
copies: `source` (document, sha256, header date), `rounds` = 6,
`decision_s` = 600, `impulses_per_round` = 1, `over_limit` = "scale",
`total_dv_limit_km_s` = None, `frame`, `submission_unit` = "m/s",
`submission_min_decimals` = 2, `late_submission` = (0, 0, 0), `dynamics`
= "two-body point mass", `outcome_order` = (capture, earth-crossing,
terminal-semimajor-axis), `announced` (five entries), `role_choice` and
`network_and_generative_ai_banned`.

`duel/params.py` holds the rest, in one `DuelParams` with two groups:

- the day's announced values: ΔV_lim, R_cap, T_round, the a interval, r0, v0
  and the epoch. `DUEL` lists the announced quantities as text, about five
  entries for about ten typed fields (two interval ends; position, velocity
  and epoch per craft), so `duel/params.py` holds an explicit mapping from
  each `DUEL` entry to its fields, and a test checks that the mapping covers
  every entry and every announced field exactly once;
- our own choices: μ (as measured in phase 1), `own_margin_km`,
  `a_margin_km` terms (`a_integration_km` defaults to the measured row of
  `row_gap_setting`) and the live floors `a_margin_floor_km` and
  `dr_int_floor_km` (section 6.4),
  `leaf_window_actions` (section 3.6), `row_gap_setting` and
  `row_gap_s_override`
  (section 2.6), knot spacing, `lambert_keep` and `lambert_evader_targets`
  (section 4.4), deadlines (the default compute deadline is derived from
  `DUEL`'s decision time);
- our readings of open rules, each tied to its question:
  `early_entry_wins` (section 3.6, Q9), `boundary_rows_judged` (section
  2.6, Q6) and `align_epochs` (section 2.2, Q10). These are not rule constants and live here, not in `DUEL`; the run
  log prints every reading in force.

`DuelParams` is strict. Loading a parameter file or CLI input with any field
it does not define fails with a message that the field is not a duel
parameter, naming the field. Fields that belong to other rule sets are the
point of this: a total Δv cap, score coefficients `kt`, `Ct`, `kv`, `Cv`,
`T_max` or a `4·T_A` horizon, a 100 s burn separation, a 5 km interception
sphere under any name other than the announced `r_cap_km`. Each of these has
a test.

The engine checks at import that `DUEL` still says what it implements
(scaling rather than rejection, no total cap, two-body, the order of the win
list) and refuses to run if an edit to `DUEL` changes any of them. A revised
PDF on the official site changes nothing in code by itself; it is caught
only by re-checking the site before the match (see the header) and updating
`DUEL`. Every run log records `DUEL`'s source sha256.

Dependencies run one way: `orbitwin.duel` imports `orbitwin.rules` and core
modules only. Which modules are core is decided by
`tests/cross_profile/test_import_boundaries.py`, not by this document. It
classes every top-level module as core, ranking or entry (`cli`,
`__main__`), and once `src/orbitwin/duel` holds Python files it checks,
statically and in a fresh interpreter, that the duel imports no ranking or
entry module. The duel engine needs `orbitwin.conic`,
`orbit`, `frames`, `models`, `errors`, `lambert` and, for the phase-1 GMAT
checks, `gmat_runner`. PR #115's test lists `report_parser`, `burn_rows`,
`event_detection` and `trajectory` as core too; the engine does not need
them (its own detector replaces `event_detection`, and the GMAT checks read
their two-column report with a small reader in `duel/`), so it leaves them
out to keep the duel release small. It never
imports a ranking module or `config`, whatever that test lists.

The conic helpers this design reuses come from `orbitwin.conic`, which
exports `coast`, `coast_many`, `gravity`, `periapsis_radius_km`,
`periapsis_radius_gradient`, `seconds_to_periapsis` and
`arc_minimum_radius`; the Stumpff functions they use live in core `orbit`.
The duel's own tests add a second check, which imports the duel package and its CLI in a fresh
interpreter and checks that no ranking module, `yaml` or network module is
loaded at import time. The banned network set is `socket`, `ssl`,
`http.client` and `urllib.request`; `scipy.optimize` itself loads
`urllib.parse` and `email`, which open no connection and are allowed.
Worker pools (`multiprocessing` loads `socket`) are therefore imported
inside the function that starts them, never at module level.

The duel release is its own distribution, designed in phase 4: a separate
`pyproject.toml` that packages `orbitwin.rules`, the core modules above and
`orbitwin.duel`, depends on numpy and scipy only, and installs the
`orbitwin-duel` script. It carries copies of the core modules, so it is
never installed in the same environment as `orbitwin`. Its offline
wheel bundle is built for the match laptop's OS and architecture, which
phase 4 records; the current `vendor/wheels/` is manylinux x86_64 only.
Development and the test suite keep running from `src/` on `PYTHONPATH`.

Core code returns facts and the duel referee gives verdicts. Facts are things
like the first entry within a given radius with its time uncertainty, the
global minimum distance and its time, the dwell inside a radius, and the
first time below a given Earth radius. Who wins, which floor applies to whom
and which window is checked are decided in `duel/referee.py`. No predicate
has a default radius or floor; there is no parameterless "is safe".

### 2.1 Units and the one conversion layer

Internally: km, km/s, s, rad. Submissions are m/s (PDF §3). One module,
`duel/units.py`, owns the factor 1000 and the projection:

    executed_km_s(submitted_m_s, dv_lim_m_s) = Π(submitted_m_s) / 1000

The projection is applied in m/s, as PDF §4.2 writes it, then converted once.
No other module multiplies or divides by 1000. A zero vector stays zero.
Every quantity whose unit the PDF leaves open (ΔV_lim, the a interval, the
announced Δv, the announced a and the announced closest approach) is
entered with an explicit unit and echoed back in both units.

Our own actions live on the submission grid. PDF §3 asks for at least two
decimals, and whether the organizer uses more digits than that, rounds or
truncates is unknown (Q7). A component written with exactly two decimals is
executed the same way under every one of those readings. So every action of
ours is held as an integer triple in centi-m/s (0.01 m/s) from the moment it
is created, and converted to km/s once, by `duel/units.py`, for evaluation.
The vector the solver judged is then exactly the vector the organizer
executes, and the rounding terms drop out of every margin (sections 2.5,
5.3).

Snapping a continuous vector (a Lambert intercept, an optimiser result) to
the grid picks, among the eight grid corners of the cell around it, the one
nearest to it whose norm is at most ΔV_lim; the corner towards zero always
qualifies, so one exists. Snapping is done on the integer side, so a vector
already on the grid maps to itself, and two actions are compared as integer
triples, never as floats. The snap moves an action by at most 0.0087 m/s
(0.0173 m/s in the rare case that only the corner towards zero fits). That
first figure is one instance of the convention this document uses for
every rounding term: a vector whose components are each rounded to the
nearest multiple of δ is off by at most (√3/2)·δ in norm, its rounding
share, here (√3/2)·0.01 m/s. A source that truncates instead would be off
by up to √3·δ; sections 2.5 and 6.4 say how each use covers that. The
snap is not negligible on long rounds: in a review, LEO at 51.6° over 86400 s,
it moved intercepts by a median 0.54 km with round-to-nearest (1.08 km
towards zero). Cells are evaluated on the snapped vector, so the verdicts
stay exact; what is lost is only some aim, and the library and the oracle
work on grid vectors throughout. The line prints the integers as two
decimals, with a zero always written `0.00`, never `-0.00`. Opponent actions
are samples of what they might do and are not snapped.

### 2.2 State

`CraftState(r_km, v_km_s)` and `MatchState(round, epoch, pursuer, evader)`.
The epoch is kept as the organizer wrote it, plus seconds since the round-1
epoch when it parses. Two-body motion does not use it. It is carried to
cross-check the file (round k should start at epoch_0 + (k−1)·T_round). The
engine's own Earth rules (section 2.4) do not need it; an ellipsoid surface
model would, if Q5 asks for one.

The two initial epochs may differ: the ranking tutorial video started its
craft at different epochs. How the organizer would align them is unknown
(Q10), so the state reader stops on unequal epochs. The reading
`align_epochs` (default false) lets it coast the earlier craft forward to
the later epoch under two-body motion, for use once the organizer confirms
that is what they do. Epochs are read in GMAT's Gregorian or ModJulian form
and are the same only within 1e-6 s, float noise: two different Gregorian
strings are at least 1 ms apart, 15 m of separation at 15 km/s. Two epochs in
different forms are not compared, because the file does not name a time
scale and UTC and A1 differ by about 37 s.

### 2.3 Dynamics and μ

A round is: both impulses at t = 0, then each craft coasts on its own conic
for `t_round_s`. Coasting uses `orbitwin.conic.coast` and `coast_many`
(universal variables, Laguerre-Conway; moved from `ranking_search` in PR
#117, with tests that pin `coast` to `orbit.propagate_universal` at 1e-8 km).

Before PR #116 they failed on high-eccentricity orbits, irregularly. From
periapsis 6878 km, `coast` raised "universal-variable coast did not
converge" for e = 1.000001
at 600 s, and in review sweeps for e = 0.9999 at 3600 s, e = 0.99999 at
600 s, e = 1.0001 at 60 s and e = 0.999999 at 3600 s and 86400 s. Random
surveys found failures from e = 0.969 to e = 1.042 (14 of 40000 states in
e ∈ [0.9, 1.1] fell outside [0.999, 1.002]), and whether a given state
fails depends on its last digits. `propagate_universal` succeeds on every
one. `coast_many` raises for the
whole batch when one row fails, so one escape-speed action would abort a
node. An evader near escape speed is a plausible action, since an escaping
evader loses only at the end of round 6.

The coordinator traced the cause to cancellation in `orbit.stumpff_c` and
`stumpff_s`: up to 1.7e-8 and 4.7e-8 relative just above their series
threshold, which makes the Kepler residual noisier than the 1e-13 step
tolerance. The core fix, PR #116, computes C in the half-angle form
C(z) = 2 sin²(√z/2)/z and S by series for |z| < 1. It leaves the
universal-variable starting guess as it is. With it, all 8192
near-parabolic samples and a 40000-state survey over e ∈ [0.9, 1.1]
converge, and `coast` agrees with `propagate_universal` to 2.6e-14 relative
and with DOP853 to 3.7e-9 km (coordinator's measurements); a review sweep
of 60000 near-parabolic states out to 6 days also converged, and on this
branch after merging main the six cases listed above all converge. An
independent implementation the user supplied uses the same half-angle form
(section 11).

The duel engine carries no integrator of its own. `coast_many` still raises
for a whole batch when any row fails, so the rollout (`events.coast_rows`)
catches that, re-runs the batch's rows one at a time, and marks any row that
still fails as an unusable action, logged, rather than losing the node.

A row can still fail. The first phase-1 code review found an outbound
hyperbola far from Earth (r = 738 595 km, e = 1.28) on which `coast` does
not converge at t = 180 s, while 120 s and 240 s converge; its hyperbolic
starting guess is the likely cause, which is core code and is reported
there. Such a state can be the opponent's announced state, which cannot be
discarded as an unusable action, so each row falls back to the core's other
solver of the same equations, `orbit.propagate_universal` (Newton-Raphson,
then brentq if Newton fails), before it is marked unusable (`events.coast_one`). The
referee's own coasts take the same path. The phase-1 tests in section 7 run
against the fixed core.

Every GmatConsole run made for this work goes through
`orbitwin.gmat_runner.run_gmat` with `$GMAT_STARTUP`; nothing calls
GmatConsole directly, not even for `--help`. It derives a per-run
startup file whose `OUTPUT_PATH` and `LOG_FILE` point into the run
directory, under `output/duel/`, and refuses to run without one. Nothing is
written inside the GMAT install.

μ is a parameter with GMAT's value as the default. What GMAT uses depends on
how the organizer writes the ForceModel, which PDF §4 leaves to "GMAT 的設定":

- `PrimaryBodies = {Earth}` with `GravityField.Earth.Degree = 0`, as the
  preliminary template did with `JGM2.cof`: μ comes from the potential file.
  All five Earth files in our R2026a (`JGM2.cof`, `JGM3.cof`, `EGM96.cof`,
  `EGM96low.cof`, `JGM2F70.cof`) carry `3.986004415e14` m³/s², that is
  398600.4415 km³/s².
- `PointMasses = {Earth}`: μ comes from the Earth body. A review run on
  R2026a found GMAT's `SMA` matching eq. (2) at 398600.4415 to 1e-9 km;
  phase 1 repeats the measurement and records it.

The repository's analytical default is 398600.4418. The difference is
7.5e-10 relative: about 5 mm in a at 7000 km and about 4 cm of along-track
drift per hour of coast (the secular term counts the change of a as well as
of n). That is far below any plausible R_cap, so μ matters for correctness at
interval edges and for the after-round check, not for strategy.

A larger risk is a ForceModel left at GMAT's defaults. A new ForceModel's
gravity field defaults to degree and order 4 (JGM-2); a review run on R2026a
saw a bare `Create ForceModel` swing a by 11.9 km in 2 h. That is not
two-body. With a J2 model at 7000 km and 50° inclination the osculating a
swings by 11 km per orbit, while the position after 600 s differs from
two-body by only 1.9 km. A position check alone would miss it; comparing the announced a with
ours catches it (section 6.4). Q2 asks for the exact ForceModel.

Phase 1 measures μ: GmatConsole runs of an eccentric orbit under both
point-mass forms, RK89 at 1e-12, states reported at 16 digits over several
days; μ is fitted by least squares against `coast`, and `Sat.Earth.SMA` is
compared with eq. (2) at both candidate values. The number and the file it
comes from go into `duel/params.py` next to the default. The same runs
validate the rollout against GMAT at several states and durations, to a
stated tolerance. The integration error that the margins use is measured
separately, for each propagator setting of the row-gap table (sections
2.5, 2.6).
GMAT output goes under `output/duel/`, never under the GMAT install or `/tmp`.

### 2.4 The Earth

PDF §4.4 forbids crossing the Earth but does not say which surface. Candidates:

| Surface | Radius | Note |
|---|---|---|
| GMAT sphere, equatorial | 6378.1363 km | Same as the potential files' reference radius. |
| GMAT ellipsoid | 6378.1363 km, flattening 0.0033527 | Polar radius 6356.7523 km. The ranking tutorial video (2026-11-01) judged the surface by the ReportFile column `Altitude`, which we measured to be geodetic in R2026a. |

The engine uses two floors and resolves the gap against us:

- our own craft going below `R_eq + own_margin_km` (default 10 km), judged
  continuously, counts as our crossing at the moment it goes below. The
  judgement carries a band g = 1e-6 km, the radius tolerance of the
  parent's final check (section 6.2), and one function of
  `duel/referee.py` makes it wherever our floor R_f is judged: the solver's
  cells, the leaf, the child's stored facts and the parent's re-flight. A
  minimum radius below R_f + g is our crossing, timed at the first moment
  the arc comes down to R_f + g. A radius within g of the floor, on either
  side, is therefore a crossing for the solver as it is for the parent,
  and the time is never later than at R_f itself. This is
  judged per cell, not by dropping actions. As pursuer, a capture still wins
  the cell if it is established before that moment, since PDF §5.1 ends the
  match at once under either reading. "Established" means the moment the
  capture would count on rows (section 2.6): the entry of the first
  passage that qualifies plus `row_step_s`, capped at the round end. That
  is neither the first entry alone nor the first entry of a shorter
  passage before the qualifying one. A capture that would only count at
  the round end is established at the round end and does not beat a
  crossing earlier in the round. As evader, going below our floor is a loss in every cell,
  also under `early_entry_wins` (section 3.6);
- the opponent is counted as having crossed only below
  R_opp = `R_polar` − e(T_round), one craft's integration error over the
  round from section 2.5 (GMAT could have it that much higher; this
  document writes R_opp wherever the opponent's floor is meant), which is below the surface under
  either reading, and only if it stays below long enough to show in sampled
  rows (section 2.6). Anything less earns us nothing.

The referee subcommand reports the minimum radius, the time below each floor
and both judgements, so a human sees the margin. The ellipsoid in
EarthMJ2000Eq ignores the motion of the pole since J2000 (about 0.15°), which
changes geodetic altitude by under 0.06 km; the polar-radius rule does not
depend on it.

### 2.5 The terminal test

In two-body motion a changes only at impulses. The evader's a at the end of
round 6 is therefore fixed by its round-6 impulse:

    a6 = (2/‖r6‖ − ‖v6 + u6‖²/μ)⁻¹

where (r6, v6) is the announced round-6 start state. For a positive interval
[a_lo, a_hi] the energy band is ε ∈ [−μ/(2a_lo), −μ/(2a_hi)], with
ε = v²/2 − μ/r. A hyperbolic or parabolic evader has a ≤ 0 or no finite a and
is outside any bounded positive interval. `a_hi = inf` (a one-sided
interval) maps to ε < 0 under the literal eq. (2); whether an escaping
evader, the limit a → ∞, counts as inside [a_lo, ∞) is open (Q3) and is
resolved by role: as pursuer we count it inside, as evader we do not. ε = −μ/(2a) is increasing on each side of a = 0, so a negative
interval (an escape zone, a plausible reading of 逃脫) maps the same way to
positive energies and is supported. Only an interval that contains a = 0 is
refused, since its meaning is unclear; Q3 asks.

Both sides start round 6 from the same file numbers, so the file's rounding
is not an error in a6, and our own impulse is on the submission grid
(section 2.1), so the organizer executes exactly the vector we judged. Had
we printed more digits and the organizer rounded them to two decimals, an
error of up to 0.0087 m/s would have moved a by (2a²‖v⁺‖/μ)·δv: 16 m at
a = 7000 km, 119 m at 26560 km and 238 m at 42164 km. The grid removes
that. What can still differ between our a6 and the organizer's:

- GMAT's integration error over one round (phase 1 measured it for each
  propagator setting of the row-gap table, section 2.6, and the default is
  the most cautious one, as for the row gap). Over the fit grid the largest
  |SMA − a₀| over two days was 4.9e-5 km for RK89 at its default accuracy
  and 4.9e-3 km at 1e-9, both on the e = 0.99 orbit; the stored value is 1.5
  times the sum of this and the next term, scaled up linearly for a round
  longer than the two days measured, and `a_integration_km` overrides it;
- a computed from GMAT's `SMA` rather than eq. (2) (Q3). Measured: GMAT's
  `SMA` equals eq. (2) on its own state to 1.6e-7 km or better under every
  setting;
- a computed from the published, rounded end state rather than GMAT's full
  precision one (Q3): with position and velocity components at resolutions
  δr and δv, each vector is off by at most its rounding share (section 2.1),
  which moves a by up to (√3/2)·((2a²/r²)·δr + (2a²‖v‖/μ)·δv): 24 m at GEO
  and 1.6 m at 7000 km for δv = 1e-6 km/s, with δr = 1e-6 km adding under
  2e-6 km. The resolutions are those of the files actually seen, the
  announced initial state before round 1;
- whatever the after-round check (section 6.4) shows on the day.

Whether the endpoints count is also unknown (Q3). The margin is computed per
state as twice the sum of the first three terms (the factor two also covers
a file that truncates its digits, whose share is twice the rounding share),
never less than 0.001 km so that a closed or open endpoint makes no
difference, and never less than the live floor that the after-round check
sets. It takes the larger of the post-impulse state and the round's end
state (the rounding term depends on where on the orbit the published state
is: from apoapsis to periapsis of a = 10 000 km, e = 0.3 it grows by 43%),
and in round 1 under early entry also the state before the impulse, which
is the one judged there.

The rounding term is evaluated at each band edge, not at the state's own a:
in energy it is (√3/2)(‖v‖·δv + μ/r²·δr), and a moves by 2a²/μ times that,
a being the edge's a since only an evader near the edge can be misjudged.
At the state's own a, the fourth code review found a barely hyperbolic
evader (a = −1e10 km) given a margin of 1.6e10 km and counted inside
[9000, 9500].

Every edge moves in energy, ε = −μ/(2a), by the edge's margin times
μ/(2a_edge²), and the moved energy is turned back into a. For a finite edge
that is the margin above to first order; it stays right for a huge or open
upper edge, where moving a linearly does not. The open edge of [a_lo, ∞)
has energy 0 and takes the km terms (integration, floors) at a_lo, the
smallest a in the band and so the most energy per km: for us as evader it
becomes a ≤ μ/(2·(2·(√3/2)(‖v‖·δv + μ/r²·δr) + km terms)), or the
published, rounded end state could read as escaping, which Q3 resolves
against us. The fifth code review found an evader at a = 1e12 km counted
inside although 34% of roundings of its end state give a < 0, and the
sixth that the open edge had left out the live floor (with a 12 km floor
the edge is a_lo²/12, 3.3e7 km at a_lo = 20000 km). The km terms are converted
at the edge, not at the a where they were measured. That is an open
choice: if GMAT's error is a fixed error in energy, δa grows as a² and a
fixed km floor understates it at a high finite edge, while an open edge,
converted at a_lo, overstates it. On the day it is dwarfed by the rounding
term (about 1e4 times larger at 1e-6 km/s), and the after-round check of
section 6.4 measures the floor at the evader's own orbit. For us as pursuer, a
hyperbolic band widened past zero energy also takes in the parabola and the
bound orbits down to the widened energy (the seventh review found those
counted outside, which favoured us).
Which band is used where, always the one worse for us:

- as evader, everything (our terminal impulse, the leaf's doom test, the
  window) uses the band narrowed by the margin. If the narrowed band is
  empty, which a narrow interval at high orbit or a large live margin from
  section 6.4 can cause, it uses instead the core of the real band: its
  centre ± 10% of its half-width, both taken in energy so that an open band
  [a_lo, ∞) has a core too (about 1.8 to 2.2 a_lo), and the tool flags it. That is a second
  documented exception to resolving against us (section 10): an empty band
  would make every state look lost and leave the solver nothing to choose
  between. Because terminal cells are 0 or 1, using the whole real band
  would let the solver pick an edge as readily as the centre; the core keeps
  it aiming at the centre;
- as pursuer, everything uses the band widened by the margin;
- the role-choice hard fact "the evader cannot reach the band" (section 6.5)
  is a claim in favour of choosing pursuer, so it uses the widened band and
  the looser evader floor R_opp.

The same holds for the capture radius. With our impulse on the grid, the
difference between our relative distance and the organizer's comes from
integration. Let e(t) be the position error GMAT accumulates on one craft
over t seconds, measured in phase 1 for each propagator setting of the
row-gap table (default: the most cautious, as for the row gap, so the
margin before the first after-round floor exists is not taken from our own
tight RK89 at 1e-12), and given a live floor on the day by the after-round
check (section 6.4). The relative distance can be off by the errors of both
craft, so the margin is δr_int(t) = e_P(t) + e_E(t). Phase 1 measured e(t)
against our coast from the same state over the fit grid of section 2.6 and
bounds it as a line, e(t) ≤ floor + rate·t, the floor being the largest
error in the first 600 s and the rate the largest (error − floor)/t after
that, both stored times 1.5. Every orbit of the fit is flown for two days,
so the line holds over two days for all of them; past that it is grown as
t² rather than extended, since an error in a makes the along-track error
grow quadratically (the second code review found that the first fit, which
stopped low orbits after two periods, gave a rate nine times too small for
RK89 at 1e-9). For the cautious row the line is 3.0e-6 km + 6.9e-8 km/s·t,
so δr_int is 2.5e-4 km after 1800 s and 1.2e-2 km after a day. The line is
the worst over the grid's orbits, so both craft use the same one; each is
raised to `dr_int_floor_km` when the live floor is larger. As pursuer a
capture counts for us only within R_cap − δr_int(t) (never below 0); as
evader a capture of us counts within R_cap + δr_int(t); refereeing a round
that has happened uses R_cap itself. `simulate_round` takes the view as a
required argument and records the radius it used at both ends of the round,
and `decide` refuses facts flown for another view. δr_int is non-decreasing
in t, so the detector of section 3.2 takes the radius as a function of time
and, on each piece, proves outside only beyond the larger of the piece's two
end values and inside only within the smaller. (Off the grid, rounding
alone would have moved each craft by ‖Φ_rv(t)‖·0.0087 m/s: 0.15 km after
5400 s and 2.2 km after a day in LEO, enough to wipe out a small R_cap.)

### 2.6 Sampled judgements

PDF §5.1 says "任一時刻", but PDF §6 makes the organizer's validator the
judge, and the ranking tutorial judged the surface from report rows. If the
validator only looks at rows every Δ seconds, a short passage between rows is
invisible to it. A periapsis d below the surface keeps a craft below it for
about 2·sqrt(2·d·r_p²/(μe)): for d = 1 km, 286 s at e = 0.01, 90 s at
e = 0.1, 29 s at e = 1 and 13 s at e = 5. Shallow dips on fast or
hyperbolic passes can therefore fall between rows. Captures are shorter
still. A pass at relative speed v and miss distance b stays inside R_cap
for 2·sqrt(R_cap² − b²)/v; the 2° crossing of section 3.2 (0.265 km/s) spends
13.6 s inside 3 km.

How far apart the rows are is not known, and it scales with the orbit.
GMAT's documented `MaxStep` default is 2700 s. Review runs on R2026a,
two-body with RK89 at default accuracy, wrote rows at about 1/61 of the
period on circular orbits (largest gap 95.7 s at a = 7000 km, 1414.6 s at
GEO) and up to 1790.9 s on a = 26560 km, e = 0.7, near apoapsis.

The engine handles this with one parameter, `row_step_s`, and one reading,
`boundary_rows_judged` (default true): that the state at every round
boundary, both Δr and each craft's radius, is judged. Under the continuous
reading that is part of "任一時刻". Under the row reading it holds if the
organizer's rows include each Propagate's end state; GMAT's ReportFile
writes a row at both ends of every Propagate, and both end states are
announced
(PDF §3.4, §6). Phase 1 measured the rows on R2026a
(`scripts/duel_gmat_boundary.py`, a `gmat` test): two craft, six rounds of
1234.567 s under RK89 at its default accuracy, which ends every round
between integrator steps, with a 10 m/s burn on one craft between rounds,
in both `Propagate P(A, B)` and `Synchronized` forms. Each round end had a
row within 1.3e-6 s of it (the epoch column's resolution) for both craft,
with positions within 6.3e-6 km of our coast, and the first row is the
initial state.
Which instants the validator judges is part of the open question Q6, and this reading takes the side that favours us: it lets a
capture that holds at a round boundary count for us as pursuer whatever
its dwell, the pursuer's main path to a capture when the row gap is long,
and an opponent's crossing that holds there count for us in either role.
So it is the fourth documented exception to resolving against us (section
10). To keep it visible, `decide` sets `via_boundary` whenever its verdict
differs from the one reached with the reading off, under either event
order, and a recommended action whose winning cells are won
only through such a boundary event (a boundary capture or crossing whose
passage is shorter than the bound) is flagged every round, and the
role-choice output also shows V_P and V_E with `boundary_rows_judged =
false`. With false, boundary events take the dwell rule like any other;
if Q6's answer contradicts the reading, the switch is set and nothing has
to be written. Within a round, a single contiguous
passage that lasts at least Δ contains a row whatever the rows' phase. So:

- an event that favours us counts if one contiguous passage lasts at least
  `row_step_s` (dwell is per passage, never summed over separate passages),
  or, under `boundary_rows_judged`, if it holds at a round boundary: our
  capture as pursuer (inside R_cap),
  the opponent's crossing (below R_opp). A passage that runs across a
  boundary holds at that boundary and therefore counts;
- an event that hurts us counts if it happens at all: a capture of us as
  evader, our own crossing.

An event that favours us counts at the moment rows would surely show it:
min(entry + Δ, round end), with the entry of the first passage that
qualifies. A passage that qualifies by its dwell ends no earlier than
entry + Δ, so the cap only matters for one that qualifies by holding at
the round end. An event that hurts us counts at its first entry.

This is against us only if `row_step_s` is at least the organizer's largest
row gap, which we do not know and which changes whenever a craft burns into a
new orbit. So it is not one number per match. An adaptive integrator's step
follows the local dynamical time r^1.5/√μ: the measured circular gaps are
0.103 r^1.5/√μ at both LEO and GEO, and eccentric orbits reach 0.12 to 0.127
near apoapsis. The engine therefore bounds the gap for each craft on each
arc as

    gap ≤ c · r_max^1.5 / √μ

with r_max the craft's largest radius over the round (closed form, like the
minimum), which also covers hyperbolic arcs that have no period, and capped
by `MaxStep` when that is known (`max_step_s`). Review runs confirmed the form for RK89 at
R2026a defaults: 0.103 on circles, 0.118 at e = 0.7, up to 0.127 at e = 0.78
depending on the starting phase, and 0.05 to 0.098 on hyperbolic arcs
(e = 1.01 to 5). With two craft in one Propagate the rows depend on its
form: `Propagate P(A, B)` with a LEO and a GEO craft wrote rows every
93.2 s, following the LEO craft, while `Synchronized` with GEO listed first
wrote them 1414.6 s apart, following the first-listed craft. The form and
the order are unknown (Q6), so a pair uses the larger of its two bounds.
Rows belong to the shared Propagate, so this pair bound also governs a
single craft's crossing: in a cell, an opponent's dip below R_opp counts
for us only if one passage lasts the cell's pair bound or holds at the
round end, and it counts at that passage's entry below R_opp plus the
bound, capped at the round end, as above.
Two places judge the opponent evader's crossing without enumerating our
arcs, both when we are the pursuer: the window filter of section 4.1 and
the greedy witness of section 5.3 item 1. In both the bound is the larger
of the evader's own and our craft's upper bound over its whole impulse
ball (the r_max bound used for the leaf below), since our burn can raise
the pair's gap; the witness holds our part at that value for its later
rounds, whose arcs of ours are not known.

c depends on the propagator settings, strongly. Measured on a LEO circle
in the design reviews: 0.103 for RK89 at accuracy 1e-11 (the default), 0.138
at 1e-10, 0.184 at 1e-9, 0.140 for PrinceDormand78, 0.020 for RK56 and 0.014
for PD45. The preliminary template used RK89 with `MaxStep = 1`, which would
make rows one second apart (`ASSUMPTIONS_AND_GAPS.md`), but the finals
template is not known. So phase 1 fitted c with GmatConsole for each of
these settings (`scripts/duel_gmat_fit.py`, 2026-11-02): periapsis 7000 km,
e ∈ {0, 0.3, 0.5, 0.7, 0.78, 0.9, 0.95, 0.99, 1.2, 2.0}, true anomaly 0°,
120° and 240°, two days each, with no cap on the step (MaxStep 1e6 s; the
ninth code review found GMAT's default cap of 2700 s cutting the gaps of the
eccentric orbits where c is largest). Each gap is normalised by the larger
radius of its two rows. The largest values per setting:

| setting | c measured | stored (× 1.5, rounded up) |
|---|---|---|
| RK89, accuracy 1e-11 (default) | 0.1274 | 0.1911 |
| RK89, 1e-10 | 0.1577 | 0.2366 |
| RK89, 1e-9 | 0.2102 | 0.3154 |
| PrinceDormand78 | 0.1396 | 0.2095 |
| RK56 | 0.0249 | 0.03738 |
| PD45 | 0.0246 | 0.03698 |

The table lives in `duel/gmat_settings.py`, and a `gmat` test re-measures
it and checks that every measured value stays under its stored row. The
match laptop needs no GMAT: `row_gap_setting` picks a row of the table and
`row_gap_s_override` sets a fixed gap when the organizer states one (0 for
continuous judging), and `max_step_s` caps the bound at a known `MaxStep`
(2700 s once the organizer confirms GMAT's default). The
default row, "cautious", takes the worst of each column (c = 0.3154, from
RK89 at 1e-9). It is against us among
the settings measured; a looser tolerance or another integrator than any in
the table could still exceed it, which is why Q2 and Q6 matter. The
RK89-default row is one choice among the others, used only when the
organizer confirms those settings. The
bound is recomputed for every cell, since each action changes the orbit;
`row_step_s` in this document means that per-cell value. The value used is
printed with its source. On the day, the announced time of closest
approach (whether it falls on a row grid) shows how they sample.

This is the largest single strategic unknown. With the cautious default the
bound at 7000 km on a circle is 293 s (c = 0.3154), so with R_cap = 5 km a
mid-round capture counts for us only at relative speeds under 0.034 km/s;
with the RK89-default row it is 177 s and 0.057 km/s. Either way
the pursuer's task turns from interception into rendezvous, or into ending
a round inside R_cap. Higher orbits make it stricter. Q6 asks first.

At a leaf the next round's arcs are not enumerated, so its Δ uses an upper
bound on r_max over the next round for each craft. The orbit after an
impulse depends only on the new speed and on the split of the new velocity
into radial and transverse parts at the known radius r, so the largest
apoapsis over the impulse ball is a two-dimensional maximisation over the
region of reachable (radial, transverse) pairs. It is solved numerically, a
grid over the region refined by local optimisation, and the result is
raised by 1% because a numerical maximum can fall short of the true one;
the phase-3 test checks it against dense sampling of the ball. The bound is
the smaller of that apoapsis (when every reachable orbit is closed) and
r + (‖v‖ + U)·T_round, which holds because the speed above radius r never
exceeds the post-impulse speed. On long rounds in LEO the apoapsis term is
the smaller one; on short rounds at high orbit the linear term is. Using
2·a_max instead of the apoapsis, or the linear term on its own everywhere,
overstated Δ by 2.1 to 2.7 times in LEO in a review, enough to deny the
pursuer's leaf all credit for mid-round captures.

The default row is the worst among the settings measured, not the worst
possible: GMAT's own `MaxStep` default is 2700 s and nothing bounds the
organizer's tolerance. So it is the third documented exception to
resolving against us (section 10), in both roles: as pursuer it governs our
captures, and as evader the opponent's crossings that count for us. To
keep it visible, every value the tool reports (cells, V, the role-choice
numbers) is also shown in two more views: with `row_step_s = 0`, and with
a pessimistic gap of max(2700 s, 2 × the per-cell bound), which stays
harsher than the default at every altitude (the default bound alone passes
2700 s above about 33,700 km). A human sees what the convention costs in
both directions. The extra views reuse the same rollouts; only the cell
verdicts, the leaves and one LP solve per view are repeated, and phase 3
includes them in the time it measures. The referee reports dwell times so a
human can see how close a call was.

## 3. Referee (phase 1)

### 3.1 Round rollout

For each craft: apply the executed impulse, coast to the knot times
`0 = t_0 < … < t_K = T_round` (default spacing 60 s), keep r and v at every
knot. For m pursuer and n evader actions at one node this is m + n rollouts;
the m·n pairs only subtract stored arrays (section 5.7).

### 3.2 Capture between knots, with a proof

Samples alone miss captures. Measured on 2026-11-01 with `coast_many`: two
circular orbits at 6878 km, planes 2° apart, meeting near the node, knots
every 60 s. The knots' closest approach is 4.70 km; the true one is 2.40 km.
With R_cap = 3 km sampling says no capture.

The bound used instead. Let ρ(t) = r_P(t) − r_E(t) and w = ρ'. Both craft
coast, so ρ'' = g(r_P) − g(r_E) with g(r) = −μ r/‖r‖³.

1. Crude bound. ‖ρ''‖ ≤ μ/r_P,min² + μ/r_E,min² = A₀, where r_X,min is the
   craft's minimum radius over the round, in closed form from
   `orbitwin.conic.arc_minimum_radius`.
2. Gravity-gradient bound. Dg(r) has eigenvalues 2μ/r³ and −μ/r³ (twice), so
   ‖g(x) − g(y)‖ ≤ ‖x − y‖ · max over the chord of 2μ/‖z‖³. Every point of the
   chord satisfies ‖z‖ ≥ min(‖x‖, ‖y‖) − ‖x − y‖/2. On a segment of length h,
   ‖ρ‖ ≤ ρ_hi = ‖ρ₀‖ + ‖w₀‖h + A₀h²/2, so with r_lo = min(r_P,min, r_E,min) and
   r_lo > ρ_hi/2:

       ‖ρ''‖ ≤ A₁ = 2μ ρ_hi / (r_lo − ρ_hi/2)³

   A = min(A₀, A₁).
3. Taylor with remainder, from both ends of the segment over half its length:

       min ‖ρ‖ on the segment ≥ LB = min( min_{τ∈[0,h/2]} ‖ρ₀ + w₀τ‖,
                                         min_{σ∈[0,h/2]} ‖ρ₁ − w₁σ‖ ) − A(h/2)²/2

   Each inner minimum is a closed-form point-to-segment distance.
4. Secant. Linear interpolation between ρ₀ and ρ₁ is off by at most
   A·h²/8 when ‖ρ''‖ ≤ A, so min ‖ρ‖ ≥ min over the chord from ρ₀ to ρ₁ of
   ‖·‖ − A·h²/8. The independent implementation uses this form with
   A = 2μ/(R − ε)². Its remainder equals ours (A(h/2)²/2 = A·h²/8), it needs
   no velocities, and either linear part can be the closer one, so the
   detector takes LB as the larger of the two bounds, both with
   A = min(A₀, A₁).

The same remainder gives an upper bound. ‖ρ₀ + w₀τ‖ is convex in τ, so its
maximum over a half segment is at an end, and

    max ‖ρ‖ on the segment ≤ UB = max( ‖ρ₀‖, ‖ρ₀ + w₀h/2‖, ‖ρ₁ − w₁h/2‖, ‖ρ₁‖ ) + A(h/2)²/2

A piece with LB > R_cap is proved outside, one with UB ≤ R_cap proved
inside; neither needs more work. Any other piece is bisected, with exact
`coast` states at each new knot, left half first, until it is proved one way
or is narrower than 1e-6 s (undecided). At 1e-3 s width the remainder is
already below 3e-9 km in LEO with the crude A₀, so undecided time is only
where the path grazes R_cap. The procedure therefore ends for long captures
as well as short ones.

One geometry has no graze to resolve: a pair that holds exactly at R_cap
for a long stretch (two craft on one circle, 4.0 km apart, judged at 4.0
km). Every piece stays undecided down to 1e-6 s, and the first review
measured 9.1 s for one round under a budget of 200 000 exact coasts. The
budget is now 20 000 per detector call (about 1 s). When it runs out, the
rest of the round is left undecided, which counts against us in either role
(no capture for us as pursuer, a capture of us as evader), and `decide`
flags the outcome `detector_exhausted`.

From the proved pieces:

- the first entry lies in the leftmost piece that is not proved outside.
  After the recursion that piece is either proved inside (then the entry is
  at its left edge, within the width of the undecided sliver before it) or an
  undecided sliver under 1e-6 s wide. A sliver with no exact state inside is
  a graze and is reported as one, and the search moves to the next piece.
  Its verdict follows the role: a capture of us when we are the evader, no
  capture when we are the pursuer. A
  path that enters, leaves and enters again reports the first entry, because
  nothing to the left is skipped;
- dwell is measured per passage. For an event that hurts us, a passage is a
  maximal run of pieces not proved outside, and its dwell is the run's
  length (an upper bound). For an event that favours us, a passage is a
  maximal run of pieces proved inside, so an undecided sliver splits it,
  and its dwell is the run's length (a lower bound on one contiguous
  stay). Section 2.6 never adds passages together.

LB also prunes the search for the round's closest approach.

Measured tightness, 60 s knots, 1800 s, LEO, against a 0.05 s dense
reference: on the segment holding the closest approach, LB is short of the
true minimum by 0.051 km (co-orbital, 14.57 km), 0.057 km (the 2° crossing,
2.40 km) and 0.106 km (an eccentric pursuer through perigee, 29.5 km). On
segments far from any approach the slack grows (up to 7.6 km at 1194 km
separation), which costs nothing because those segments are cleared anyway.
No segment had LB above the true minimum, and an independent review found
none in 620 random close-approach segments.

The ranking search's `measure` uses Hermite cubics with a reach heuristic. It
works well in practice but is not a proof, and it is a ranking module, so the
duel engine neither reuses nor imports it.

The detector lives in `duel/events.py` and returns facts only: first entry
within the radius it is given, with the width of the piece it was resolved
to; the global minimum distance and its time; and every passage inside the
radius with its entry, its dwell and whether it holds at the round end, so
the referee can time an event by the passage that qualifies.
`duel/referee.py` applies R_cap, the dwell rule of section 2.6 and the win
order. If the ranking side later wants the same proof, the fact function can
move to core unchanged.

### 3.3 Earth crossing per craft

Per arc: the minimum radius in closed form (periapsis if a passage falls
inside the arc, else an endpoint), compared with the floors of section 2.4.
A round longer than one period holds several periapsis passages; they are
enumerated from the first passage time and the period, and each is judged
on its own. Between consecutive turning points (periapses and apoapses) the
radius is monotone, so each crossing of r = R is bracketed there and found
by a root search on the exact coast. The closed forms through the eccentric
or hyperbolic anomaly lose accuracy as e → 1 and are not used for crossings.
Nor is the core's closed-form periapsis time within 1e-6 of e = 1: the third
code review measured it 0.13 s off at 1 − e = 1e-10, 11 s at 1e-12 and up
to 8e4 s nearer a parabola, and a periapsis it put after the round end hid a
real Earth crossing. There the turning points are where r·v changes sign
(− to + at a periapsis, + to − at an apoapsis), bracketed on a grid of at
most min(60 s, P/8), since they are half a period apart, and found by a
root search on the exact coast. The grid also covers a near-radial ellipse
with a short period (a craft released almost at rest, 1 − e about 1e-24),
which the sixth code review showed falling through the Earth unreported.
An arc with no angular momentum at all (|r × v| ≤ 1e-9·|r||v|) has no conic
shape and is refused as unusable, as is a pair whose minimum radius is 0.
The arc's minimum radius is the closed-form periapsis radius when a
periapsis is inside, else the smaller end.

The turning times are built from the first periapsis time by adding whole
periods, never by stepping back a period first. The second code review
found that `(first − P) + P` loses `first` to rounding when the period
dwarfs the round: at 1 − e = 1e-9 the period is 1.6e17 s, the periapsis
came out 7 s early on a 32 s grid, the bracket missed the dip, and a dip
0.01 km below our floor went unreported (56 of 3000 random near-parabolic
arcs). As a second guard, an arc whose closed-form minimum is below R but
whose brackets found no crossing (a near-circle under e = 1e-10 is treated
as having no turning points) is re-bracketed on a 4001-point grid plus its
minimum. For a near-circle the minimum is only a lower bound (the smaller
end less 2ae), so finding nothing is accepted; for any other arc the
minimum is exact and an arc that still shows none is refused as unusable.

The first inbound time is needed for event ordering, and each passage's
dwell below R is what section 2.6 judges. A test compares these times with
`scipy.integrate.solve_ivp` (DOP853, at most 2 s steps, since its events
step over a shallow dip otherwise) on random arcs, elliptic, near-parabolic
and hyperbolic: 24 in the fast suite and 256 in the slow one.

### 3.4 Event ordering

PDF §5 says "依下列順序判定" and both events "比賽立即結束". `DUEL` records
the order of the list; how the list applies within one round is not settled
by it and stays Q4; PR #115's docstring now says the same, and the engine's
import-time check reads only the order of the entries. Two readings:

- priority: a capture anywhere in the round wins for the pursuer, even after a
  crossing earlier in the same round;
- chronological: the first event in time decides; ties go to the listed order.

They differ when a crossing precedes a capture in the same round, and when
both craft cross (chronological then names the first to cross; priority
says nothing):

- the evader crosses, then is captured: the pursuer wins under both;
- the pursuer crosses, then captures: the pursuer wins under priority and
  loses under chronological.

For decisions each cell is judged under both readings, with the events as
the conventions of sections 2.4 and 2.6 count them (our crossing at our
floor, a capture of ours once established, an opponent's crossing below
R_polar once it counts, both timed by the qualifying passage as section
2.6 says), and the cell takes the reading worse
for us. Under the chronological reading a double crossing goes against the
first to cross; under the priority reading a capture in the round decides,
and a double crossing with no capture says nothing, which is resolved as a
loss for us (Q4). Some consequences:

- as evader, a pursuer that crosses and then captures us wins (priority);
- as pursuer, our own crossing ends the match against us before any capture
  not yet established by then (chronological);
- as pursuer, when the evader's counted crossing comes before ours and the
  round also holds a capture of ours, both readings give us the cell
  (priority through the capture, chronological through the evader crossing
  first), so it is ours; with no capture the priority reading says nothing
  about the double crossing, so the cell is a loss.

For refereeing the engine reports the outcome under each reading: the
chronological one names the first to cross, and the priority one marks a
double crossing without a capture as unresolved.

A capture exactly at a round boundary belongs to the round that ends there;
the next round's start state is then never used. An initial state already
inside R_cap is a capture at t = 0 of round 1, reported before any search.

### 3.5 Terminal judgement

After round 6, with no capture and no decisive crossing, the evader wins iff
a6 is inside the interval (with the margin of section 2.5 when deciding). The
referee computes ε at the end of the round and right after the impulse and
requires |Δε| ≤ 1e-9 · max(μ/‖r‖, |ε|), failing loudly otherwise. The
tolerance scales with the potential, or with the energy itself on a far
open arc, where |ε| dwarfs μ/r and round-off in v² alone tripped the
potential-only form (fifth code review: 21 of 3000 arcs at 1e6 to 1e7 km
over 1e6 s). It was sized at 1e-8 when a near-parabolic fallback
still existed (2.1e-8 km²/s² measured at μ/‖r‖ = 27 km²/s², against under
7e-11 for `coast`); on the fixed core the second code review found the worst
|Δε| at 0.021 of that over 30000 states, so phase 1 tightened it tenfold.
A relative check on a would raise false alarms near parabolic orbits, where
a is ill-conditioned.

### 3.6 Reaching the interval early

PDF §5.3 judges the interval once, "第 6 回合結束時", and PDF §3.5 continues
"直到第 6 回合結束". PDF §2 ("在逃脫方抵達安全空域之前") and the PDF §5
preamble ("逃脫方全程未被攔截且抵達安全空域即獲勝") can be read as an
evader win the moment its a enters the interval.

This open question is not resolved against us in both roles (section 10
lists the others), for these reasons, all from the PDF:

- the evader's own objective in PDF §2 is "全程不被攔截，並在比賽結束時抵達
  安全空域": never captured over the whole match ("全程"), and at the
  interval at its end. The §5 preamble repeats "全程未被攔截", which an
  evader that wins in round 2 could not yet have shown;
- the numbered list in PDF §5 is what the organizer says it applies
  ("依下列順序判定"), and its third item names the end of round 6;
- "比賽立即結束" appears only in PDF §5.1 (capture) and §5.2 (crossing),
  never for an evader win.

These favour the default reading without settling it. So the engine
follows PDF §5.3 by default, and the other reading is built and tested
before the day as `early_entry_wins` (default false, one of our readings in
`DuelParams`, section 2.0). When true:

- a changes only at impulses, so entry is judged at t = 0⁺ of each round,
  on the state after the impulse (a zero or missing submission still gives
  a post-impulse state). Whether an evader that starts inside the interval
  has already "抵達" before its round-1 impulse is open too and resolved by
  role: as pursuer, an evader inside at the initial state has entered; as
  evader, only the post-impulse state counts;
- whether an entry at t = 0⁺ also beats a capture or a crossing later in
  the same round is a further open point: the organizer announces a only
  after simulating the round (PDF §3.4), judges capture and crossing first
  (PDF §5), and asks for "全程未被攔截". It is resolved against us by role.
  As pursuer, the entry wins for the evader whatever happens later in that
  round. As evader, our entry counts only if the rest of that round has no
  capture of us and no crossing of ours (our floor, section 2.4); a capture
  at t = 0 itself (initial separation inside R_cap) comes first in either
  role;
- our margins apply as to the terminal test (section 2.5);
- in the solver, a cell whose evader action lands inside the interval is an
  evader win under the pursuer convention, and under the evader convention
  only if that cell also has no capture and no crossing of the evader;
- at a leaf (the start of a round with n impulses left), if the exact
  one-impulse window of section 4.1 at the leaf state meets the band (chosen
  by role, section 2.5), the evader can enter at t = 0⁺ of that round. Under
  the pursuer convention V = 0. Under the evader convention a later capture
  or our own crossing still beats the entry, so entering is only one option.
  The leaf samples `leaf_window_actions` (default 8) terminal-energy actions
  of section 4.3 and keeps those whose arc stays above the evader's floor
  for the round; if none is left, the option does not exist. The entry value
  is C computed with the evader's dodge restricted to those impulses: the
  smallest cone of impulse directions that contains the non-zero ones sets
  θ, and σ₁σ₂ is scaled as in section 5.3 item 3 (sin²θ up to 90°, 1 beyond;
  a zero impulse among them has no direction and is ignored). V is the smaller of that and the
  value of waiting computed below; the extra samples are part of the leaf
  cost phase 3 measures. Otherwise, and for the value of waiting, V is
  computed as in section 5.3: T from the outer bound
  over the n remaining impulses (the evader can still enter at any later
  round start, so missing OB_1 alone is not doom), C as there, and the
  forced-evader cone of item 3 still applies, since the evader must enter by
  its last impulse.

To keep the risk visible:

- whenever the evader's a is inside the interval at a round end, the tool
  says so in large type and names the reading under which that ends the
  match;
- the role-choice output (section 6.5) shows whether the evader starts inside
  the interval or can reach it in round 1, which under the other reading
  decides the match for the evader under the pursuer convention and is one
  option among others under the evader convention, and V_P and V_E computed
  with `early_entry_wins = true` next to the default;
- Q9 is asked before anything else on the day; if the organizer answers
  that entry wins, the parameter is set and nothing has to be written.

## 4. Action library (phase 2)

Every action is an EarthMJ2000Eq vector with norm at most ΔV_lim. Ours are
integer centi-m/s triples (section 2.1), and duplicates among them are
merged by comparing the integers; the opponent's are float samples in
km/s, merged when within 1e-9 km/s. Either way the first in the library's
fixed order is kept. Near-duplicates are kept: at U = 100 m/s a 0.1 m/s
difference moves the arrival by up to a kilometre after an hour of coast,
which is the scale of R_cap. The one known redundancy is handled directly:
±B is left out when |B·r̂| > 0.99, since B is then nearly ±R.

### 4.1 Exact final-round window

With s = ‖v6‖, r = ‖r6‖ and U = ΔV_lim, one impulse reaches exactly the
speeds [max(0, s − U), s + U]. The band needs

    s_lo = sqrt(max(0, μ(2/r − 1/a_lo))),  s_hi = sqrt(μ(2/r − 1/a_hi))

(with no solution if 2/r − 1/a_hi < 0). The speed window is the intersection.
Example: r = 6878 km circular (s = 7.612684 km/s), U = 100 m/s, band
[7000, 7100] km. Band speeds [7.678737, 7.730783], reach [7.512684, 7.712684],
window [7.678737, 7.712684], least impulse 66.05 m/s. At that radius a 100 m/s
tangential burn moves a from 6878 to 7064.8 km (linear estimate 180.7 km).

The speed window ignores the Earth. An orbit of semi-major axis a through
radius r has periapsis at most min(r, 2a − r), so a low band can put the
periapsis under the surface: lowering a from a 7000 km circle to 6600 km
gives a 6200 km periapsis. Only the round-6 arc matters (nothing after the
match is judged), so window actions are filtered by the arc floor over
T_round. The floor follows the role: our own floor R_f when we are the
evader; when we model the opponent evader as pursuer, R_polar with the
dwell rule of section 2.6, since a shallower dip does not count against
it. Two consequences:

- as evader we need a sampled window action that passes the filter;
- as pursuer we count the evader as unable to reach the band only when the
  speed window itself is empty, since a sampled filter cannot prove that no
  passing action exists.

### 4.2 Equal-energy rotations

‖v + u‖ = ‖v‖ iff v·u = −‖u‖²/2. These are rotations of v by θ about an axis
k ⊥ v, with ‖u‖ = 2‖v‖ sin(θ/2), so θ ≤ 2 arcsin(min(1, U/(2‖v‖))). They
change the orbit's direction without moving a. An impulse perpendicular to v
is not one of them: it adds ‖u‖²/2 to the energy. Sampled over k at several θ.

### 4.3 Terminal-energy actions

For a target speed s* inside the window, the impulses with ‖v + u‖ = s* and
‖u‖ ≤ U point v + u into a cone about v with
cos φ ≥ (s² + s*² − U²)/(2 s s*), the right side clamped to [−1, 1]. Sampled
at s* across the window (centre and near both ends, margins applied) and φ
across the cone. In earlier rounds the same construction aims at
intermediate energies on the way to the band.

### 4.4 Pursuer intercepts

Lambert (`lambert.solve_lambert_all`, both directions of motion) from the
pursuer's start to the evader's position at several times inside the round,
always including t = T_round, where a capture counts whatever its dwell,
along the evader's coast and along sampled evader actions. Impulses over the cap are projected and kept as ordinary candidates;
a projected intercept no longer intercepts and may point nowhere useful (a
retrograde branch projects to a mostly braking burn), so it earns its place
in the matrix like any other action.

Cost and count grow with revolutions. Measured per call: 1.2 to 2.3 ms at 0
revolutions; 4.0, 11.9 and 15.5 ms at up to 3, 10 and 14, returning 14, 42
and 58 branches. A fixed cap on revolutions would be wrong: on a long round
the cheap intercepts are the many-revolution ones. Measured, for a pursuer
on a 6878 km circle and an evader 200 km ahead in a plane 0.3° off: at
T_round = 86400 s the cheapest branch is 39.9 m/s at 15 revolutions, while
the cheapest with at most 3 is 3296 m/s. (Coplanar, the same geometry gives
1.56 m/s at 15 revolutions.) The revolution count belongs to the transfer
orbit, not to either craft, so no window around the pursuer's period is
safe: with the evader on a 12000 km circle and T_round = 86400 s, all the
branches returned have 0 to 9 revolutions while t/P_P is 15.2. Instead each
call passes the largest count any transfer can have. Every ellipse through
both endpoints has a ≥ a_m = s/2 (s the semi-perimeter of the triangle
formed by the two position vectors and their chord), so its period is at
least P(a_m). When the chord is short a_m falls towards r/2, and the bound
grows to about 2.8·t/P_P (41 revolutions and 63 ms for a target 200 km
away at t = 15·P_P, in review), but most of those transfers dive under the
Earth. A transfer of one revolution or more passes its periapsis, so for it
to stay above a floor R its periapsis is at least R and its apoapsis at
least max(r1, r2), giving a ≥ a_f = (R + max(r1, r2))/2. The cap follows
the role:

- as pursuer, the library is ours and R is our floor R_f: the call passes
  max_revolutions = ⌊t / P(max(a_m, a_f))⌋. The dropped transfers would put
  us below our floor before the intercept, a loss for us (section 2.4).
  Their projected versions are dropped too, which only removes candidates,
  never a judgement;
- as evader, the library models the opponent, and a pursuer that dives,
  even through the Earth, and then captures us still wins (section 3.4).
  So no floor applies and the call passes ⌊t / P(a_m)⌋.

`solve_lambert_all` scans every count from 0 to that maximum and the call
pays for the whole scan, about 17 ms at 16 revolutions. As pursuer, the
branches whose periapsis passage before the intercept lies below our floor
are removed first, since they are certain losses (a cap on revolutions
cannot remove them all: in the review's 86400 s case the 16-revolution
branch is within the cap and dives to 6184 km). Of what remains, the
library keeps the `lambert_keep` (default 4) branches with the smallest
impulse before projection. A minimum-revolution argument would
cut the cost but changes a shared core module, so it is proposed to the
module's owners rather than assumed.

Targets: the evader's coast, plus at most `lambert_evader_targets`
(default 8) sampled evader actions (the highest-weight ones of the current
mixture once one exists, otherwise spread over the library), each at a few
times including T_round. With the defaults that is at most 9 × 3 × 4 = 108
Lambert actions; the pursuer library is sized around them.

### 4.5 Frame, random and rescue actions

±V, ±N, ±B (VNB from `frames.vnb_basis`) and ±R (radial) at 1, 1/2 and 1/4
of the cap, plus zero. Random directions are uniform on the sphere
(normalised Gaussians) at full cap and at radii uniform in the ball, from a
recorded seed.

For our own craft a rescue action is always added at the root: the impulse
that maximises the arc's minimum radius over the round, found by local
optimisation from the best of ±V and ±R. Actions that go below our floor
stay in the library and are judged per cell (section 2.4); as pursuer some
of them can win by capturing first. Only when every cell of every action of
ours is a loss for us, so that nothing is left to choose between, is the
rescue action the recommendation, and the tool says so loudly.
A sampled search cannot prove that no action clears the floor, so Earth doom
is only ever assumed against us (section 5.3 item 4).

## 5. Game solver (phase 3)

### 5.1 Structure

The state is public after every round (PDF §3.4) and moves are simultaneous
within a round (PDF §3.2). This is a finite-horizon stochastic game with
perfect monitoring, so the value is computed stage by stage (Shapley): at
each state, the matrix game whose cells are the values of the successor
states. At round 6 the successors are terminal and every cell is exact.

### 5.2 Payoff cells

Cell (i, j) for pursuer action i and evader action j at round k:

- the round's events as sections 2.4 and 2.6 count them (a capture that
  counts, crossings of either craft, and under `early_entry_wins` an entry,
  section 3.6), judged under both event-order readings of section 3.4 with
  the worse for us taken. When either reading decides the round the cell
  is 1 or 0;
- at k = 6, the terminal judgement;
- otherwise the value of the successor state: a sub-game at depth 2, or the
  leaf value V at the depth limit.

The values are game values on sampled action sets with a heuristic leaf. They
are not probabilities of winning, and the tool never prints them as such.

### 5.3 Leaf value

V(x, n) for the state x at the start of a round with n impulses left per side.
It has to order states that are one or two rounds away from an exact
decision, cheaply, in the m·n batch. Its parts:

1. Terminal reachability. An outer bound on the energy the evader can reach
   in n impulses. The current impulse is applied at the known radius and
   speed s, so it reaches exactly the energies of speeds [max(0, s − U), s + U]
   (section 4.1). Later impulses happen at unknown radii, but above the
   evader's floor R_e: R_polar under the pursuer convention (the opponent is
   only out once below it, section 2.4) and our own floor R_f under the
   evader convention. Its speed is then at most v̄(ε) = sqrt(2(ε + μ/R_e)).
   R_e by role makes the bound looser as pursuer and tighter as evader, both
   against us. One such impulse adds at most
   U·v̄(ε) + U²/2 to ε and removes at most U·v̄(ε) (the exact loss is
   U‖v‖ − U²/2 when U ≤ ‖v‖ and ‖v‖²/2 otherwise, both under U·v̄). Applying
   the exact first step and then n − 1 bounded steps, with v̄ updated each
   step, gives an interval OB_n; if it misses the band (narrowed when we are
   the evader, widened when we are the pursuer, section 2.5) the evader cannot
   win (rigorous in two-body, ignoring the pursuer). At
   n = 1 this is the exact window. v̄ is loose (8.21 against 7.61 km/s in LEO,
   10.77 against 3.07 km/s at GEO, with R_e = R_polar), so at higher orbits
   the bound says little.
   A greedy plan (tangential, min(U, needed), coast T_round, repeat) gives a
   witness. Its arcs are checked against R_e as the window filter of
   section 4.1 checks its arcs. Under the evader convention the plan is
   ours, R_e = R_f, and any dip below it, under the floor rule of section
   2.4, rejects the plan. Under the pursuer convention R_e = R_opp with
   the dwell rule of section 2.6: a dip rejects the plan only if one
   passage below R_opp lasts the pair bound, the larger of the arc's own
   bound and the pursuer's r_max bound over its impulse ball (section 2.6),
   held at its next-round value for the plan's later rounds. Holding it is
   part of the heuristic: a pursuer that raised its orbit later would raise
   the gap and hide longer dips. Greedy ignores timing
   such as burning near periapsis, so it is a witness only, never a proof
   that the evader is forced.
   Doomed evader (outer bound misses): T = 1. Witness found: T = 0. Neither:
   T rises linearly from 0 to 1 as the energy gap goes from what the greedy
   plan covers to what the outer bound covers.
2. Capture threat. The impulse the pursuer needs to reach the evader's coast
   at time t of the next round, linearised: ‖Φ_rv^P(t)⁻¹ (r_E(t) − r_P(t))‖,
   where Φ_rv^P is the pursuer arc's position-velocity block of the two-body
   state transition matrix. Knots where Φ_rv is near singular (t near 0, and
   some directions at whole and half periods) are skipped. Let m be the
   minimum over t divided by U, reached at t*.
   Motion along the relative velocity only shifts when the craft meet; it
   does not open the miss distance. So the area the evader can hide in is its
   dodge ellipsoid U·Φ_rv^E(t*)·ball projected onto the plane normal to the
   relative velocity at t*. With σ₁, σ₂ the semi-axes of that projection, and
   v_rel the linearised relative speed at t*:

       C = s(m) · min(1, κ R_eff² / (σ₁σ₂)),   R_eff² = max(0, R_c² − (v_rel·Δ/2)²)

   with R_c the capture radius after the margin of section 2.5 at t*
   (max(0, R_cap − δr_int) as pursuer, R_cap + δr_int as evader; if the
   margin ever wipes out R_cap for us as pursuer the tool flags it, as it
   does an empty band), s(m) = 1
   for m ≤ 1 and exp(−(m − 1)/w) above it, and Δ the dwell required by
   section 2.6 (`row_step_s` when we are the pursuer, 0 when we are the
   evader). A straight pass at miss distance b spends 2·sqrt(R_c² − b²)/v_rel
   inside R_c, so R_eff is the largest miss that still dwells Δ; a fast pass
   gives R_eff = 0 and no credit, as the exact cells would. A capture that
   holds at the end of the next round counts whatever its dwell (section
   2.6), so C is the larger of the value at t* and the value at
   t = T_round, the latter with m(T_round) and the full R_c (no dwell). The
   second factor is the capture cross-section over the
   hiding area, which is how a covering game of this kind scales. It ignores
   that parts of the hiding area may lie outside the pursuer's reach, which
   makes C an overestimate there; m carries that only through s(m). The
   linearisation is poor at large separations, where m ≫ 1 and C is
   negligible anyway, and for slow rendezvous, where the straight-pass
   picture is crude; calibration (below) has to cover that case.
3. Forced evader. The current impulse must leave the evader inside what the
   remaining n − 1 impulses can still fix: after it, ε must lie in
   OB_{n−1}⁻¹(band), the energies from which n − 1 bounded impulses can still
   reach the band. When that set excludes some of the current impulse's
   reach, the usable directions form a cone. With D the least energy change
   the current impulse must make and s the current speed: when raising, full
   magnitude is best and the cone is cos θ ≥ (D − U²/2)/(sU) about +v. When
   lowering, a smaller impulse can remove more energy, so the condition is
   max over λ ∈ [0, U] of (λ s cos θ − λ²/2) ≥ |D| about −v: cos θ ≥ √(2|D|)/s
   where s cos θ ≤ U, and cos θ ≥ (|D| + U²/2)/(sU) otherwise. The largest
   possible loss (θ = 0) is s²/2 when U ≥ s and sU − U²/2 when U < s; if |D|
   exceeds it no direction works, which item 1 already reports as doomed.
   σ₁σ₂ is scaled by sin²θ for θ ≤ 90° and left as it is beyond (a cone
   wider than a hemisphere does not narrow the dodge). This is the only way
   the terminal condition feeds the capture term.
4. Earth. Doom from the Earth is used only against us. If no sampled action
   of ours keeps our next arc above our floor, V is
   computed as if we crash: as pursuer, V = C counted only over captures
   that would be established (the qualifying passage's entry plus Δ,
   capped at the round end, section 2.4)
   before our crossing; as evader, V = 1. Leaves sample ±V, ±R and the frame
   directions only; the rescue optimisation of section 4.5 runs at the root.
   A leaf may therefore call us doomed when a rescue existed, which errs
   against us. The opponent is never assumed to crash, since a sampled
   search cannot prove it must.

Combined: V = 1 − (1 − T)(1 − C). The two pursuer win paths are alternatives.

Why this form: the only exact information about later rounds is energy
reachability, which two-body motion lets us bound rigorously because a moves
only at impulses. Capture in a one-shot simultaneous round is a covering game,
which the reach ratio and the cross-section ratio measure. Everything nearer
than the depth limit is computed by the exact referee.

Its limits: C looks one round ahead while T looks to the end, so with several
rounds left V understates the pursuer, who has that many more chances; a
plan that opens the distance before closing it is invisible past the depth
limit; the witness's pair bound after its first round is held, not
bounded (item 1); κ and w are fitted, not derived.

Calibration in phase 3 uses ground truth where it exists. At the start of
round 6 the matrix is exact on its samples, so V(x, 1) is compared with that
exact value on many random round-6 states. For n ≥ 2 there is no exact value;
V(x, n) is compared with the outcomes of whole self-play matches started from
x between engine strategies with depth 2, with seeds, which is costly but
done before the day. Since V differs by convention (Δ, the band), κ and w
are fitted once per convention, and the self-play used for each fit is
refereed under that same convention. This calibration runs on the self-play
loop built in phase 3; the `selfplay` CLI of phase 4 only wraps it.
Agreement between depth 1 and depth 2 is reported but only measures
consistency. Monotonicity tests pin the direction of each term:
V rises as the energy gap grows, as R_cap grows, as m falls and as the
dwell Δ it is given falls (Δ is an input of V, so the test varies it
directly).

Cost per pair: Φ_rv^P(t) depends only on the pursuer action and Φ_rv^E(t) only
on the evader action, so both are computed m + n times; the pair step is a
batched 3×3 solve and a 2×2 projection per knot.

### 5.4 Matrix game

The pursuer maximises. `scipy.optimize.linprog(method="highs")` solves
max z s.t. Aᵀp ≥ z1, Σp = 1, p ≥ 0, and the dual LP for the evader's q. By
weak duality min(Aᵀp) ≤ v* ≤ max(Aq) holds for any feasible p and q, so it
proves nothing; the check is the duality gap max(Aq) − min(Aᵀp) ≤ 1e-7, and
the solve fails loudly above it. (An independent run of 300 random matrices
from 16 to 256 actions found gaps up to 5.7e-10.) We take p or q depending on
our role. Values hold for that matrix only.

### 5.5 Double oracle and lookahead

1. Seed both libraries (section 4), build the matrix, solve.
2. Best response for each side to the other's mixture, over the continuous
   ball: start from the best library actions and random points, optimise a
   smoothed payoff (the capture indicator replaced by a logistic in
   R_eff − Δr_min, with R_eff from the pass's relative speed as in section
   5.3, and its larger counterpart in R_c − Δr(T_round) for a capture that
   holds at the round end; widths shrink), with u = U·tanh(‖z‖)·z/‖z‖ so the
   optimiser is unconstrained. Accept a response only if its exact payoff
   against the mixture improves by more than 1e-4.
3. Add accepted responses, re-solve, repeat until neither side improves or the
   deadline is near.

The final improvement found by each oracle is reported as a lower bound on
how exploitable the mixture is. It is not a certificate.

Lookahead: depth 1 is the matrix at the current round with V at the
successors. Depth 2 replaces V by a sub-game at the successors in the support
of the current mixture, re-solves, and repeats on cells that newly enter the
support, until the support is fully refined or time runs out. Sub-games use
fixed small libraries with no Lambert and no double oracle: frame,
rotation, terminal-energy and random actions for both sides, plus for the
pursuer a few aimed actions u = Φ_rv^P(t)⁻¹ (r_E(t) − r_P(t)) at several
t, which the
leaf computes anyway. Without aimed actions depth 2 would understate the
pursuer against depth 1, whose C assumes aiming.

Refining only the support mixes two scales in one matrix: refined cells
carry depth-2 values, the rest keep the leaf, which understates a pursuer
with rounds left. A pursuer row kept out of the support by pessimistic leaf
values may never be refined, and the evader is drawn towards unrefined
columns. To limit this, refinement covers whole rows and columns of the
support (every cell of a support row or column), and the tool reports the
fraction of cells refined. The bias that remains is listed in section 10.

Measured costs, one core: `coast_many` for 128 actions × 61 knots in 3 to
6 ms; pair distances 128 × 128 × 61 in 23 to 33 ms; STMs for 32 actions × 30
knots in 6 to 7 ms; a 16 × 16 `linprog` in 2 to 5 ms. A sub-game of 16 × 16
is about 15 ms before its leaf values; the leaf adds its STMs and the
greedy witness per successor, and its cost is measured in phase 3, not
assumed. With Lambert in each sub-game it would be about 150 ms each, and
the 4096 successors of a 64 × 64 root would break the 7-minute budget.
Phase 3 sets the default sizes from measured times, leaf included, on the
competition laptop and uses worker processes. Depth 2 runs only after a depth-1 answer exists.

### 5.6 What the solver may not see

The recommender's inputs are the parameters, the round number, our role, the
announced start state, a deadline and a seed. There is no input for the
opponent's action in the current round, and the only function that takes both
actions is the referee. Tests check the signature. The isolation test runs
`recommend` with a scrubbed environment and stdin closed, inside a match
directory that also holds the opponent's line, with one search worker (no
pool). Section 6.2 runs the search in a child interpreter, so both the
parent and that child record every Python-level file open through an audit
hook (`sys.addaudithook`, events `open` and `open_code`), installed as the
first statement of each entry point when the test sets an environment
variable naming the record file; each process appends to it. Allowed: the
interpreter's installation (stdlib and site-packages), the package's source
or install location and its `__pycache__`, `/dev/null`, integer file
descriptors, the parameter and state files it was given, its own log, and
its own working files in the match directory: `round_<k>_u.json`, the
fallback file, the round's lock file, the numbered incumbent files
`round_<k>_inc_<i>.json` with their temporary siblings, and the
`round_<k>_runs/` archive (section 6.2). Two more are allowed by name,
each only in one access mode: the audit record file named by the
environment variable, opened for appending only, since the hook writes
to it after it is installed; and `/proc/<pid>/stat` for a numeric pid,
opened for reading only (mode `r` or `rb`, or `os.open` flags whose
access mode is `O_RDONLY`, without `O_CREAT` or `O_TRUNC`), which the lock
of section 6.2 reads for the start times of its own processes and of the
ones a lock file records. Any other path under `/proc` (`/proc/self/...`,
`/proc/<pid>/environ`, `cmdline`, `fd`), a `/proc/<pid>/stat` opened for
writing, and the record file opened for reading fail the test, as does
anything else, the opponent's line above all. Opens made in C code are not visible to audit
hooks, so this is a check on our Python code, not a sandbox. Pool workers
get their inputs as arrays from the parent; a second test inspects the
pickled task payloads and fails if any carries a path or file name.

Matching pennies ([[1, 0], [0, 1]]) must solve to 0.5, and a solver that let
the pursuer see the evader's column would get 1. That test checks the LP and
the simultaneous formulation, not isolation; isolation is the audit test's
job.

Past rounds' announced impulses are logged. An opponent model built from them
is possible but off by default, because exploiting a model opens us to being
exploited.

### 5.7 Batched evaluation

Per node: m + n rollouts with `coast_many` (one call per side, all knots),
per-craft quantities once (minimum radius, Earth floors and dwell, the
evader's a, the STMs), then the m·n pairs as array differences in chunks that
fit memory (128 × 128 × 61 × 3 doubles is 24 MB). Pairs are screened as
below and the rest go to the scalar refinement of section 3.2. The next
state of pair (i, j) is (P_i(T), E_j(T)), already computed.

The two hot loops sit behind a small backend module, `duel/backend.py`,
written against an array module `xp`:

- `propagate_batch(r0[N,3], v0[N,3], t[K]) -> (r[N,K,3], v[N,K,3])`, the
  universal-variable coast;
- `pairwise_screen(rp, vp, re, ve, a_bound) -> (lb[m,n], d_knot[m,n])`, tiled
  over pairs: the smallest segment LB of section 3.2 per pair, and the
  smallest knot distance. A plain minimum over samples overestimates the true
  minimum (4.70 against 2.40 km in the 2° example), so the screen's decision
  quantity is LB, never the sample minimum.

NumPy is the default and the only hard dependency. JAX (`jax[cuda12]==0.6.2`)
is the only optional GPU backend, used only when selected explicitly
(`--backend jax`). At start-up the backend sets `jax_enable_x64`, runs the
kernels on a small fixed input and checks that every output is float64; if
not, it refuses to use JAX. Batch shapes are fixed and padded so a round
never recompiles, and the warm-up (1 to 2 s of compilation, measured) runs
when the match starts, outside any round's deadline; if it ever runs inside
one, its time is charged to that deadline. If JAX is missing, refuses, or
fails at any step, the run falls back to NumPy and says so in its output and
log. No other GPU package is used.

Measured by the coordinator in FP64: NumPy on this box 9.6e7 pair-samples/s
with a tiled kernel (the untiled distance computation timed in section 5.5
runs at 3 to 4e7);
JAX 7.5e9 on the RTX 4060 Ti and 1.1e10 on one P100 (4096 × 4096 pairs × 512
epochs in 0.79 s), with propagation of 2048 × 512 in 18 ms and 6.4 ms, and
checksums equal to NumPy's to 1e-13 relative. The JAX dependency stays out
of the numpy-and-scipy offline release; it is an optional extra, and if the
match laptop is to use it, its jax and CUDA wheels go into that laptop's
offline bundle too.

Neither of those GPUs can be used during a match. PDF §6 allows only the
organizer's computer or our own laptop, with no network, so the desktop and
the remote P100s are out. JAX helps the match only if the match laptop has
an NVIDIA GPU, which phase 4 checks and records. Otherwise the match runs
on NumPy, and the GPU backend serves the work before the day: calibrating
the leaf (section 5.3) and self-play. The NumPy path must meet the time
budget on the match laptop on its own.

The screen only clears. Clearing is itself a verdict (no capture in this
pair), so it is made conservatively: a pair is cleared only when its LB
exceeds R_cap plus its margin (section 2.5) plus a guard (default 1 km).
Every other pair, and every craft whose minimum radius comes within the
same guard of a floor, goes through the continuous check of sections 3.2
and 3.3 on the CPU in NumPy, which alone decides that a capture, a crossing
or a dwell happened. On the JAX backend, a random sample of cleared pairs
(64 per node) is also recomputed in NumPy; any disagreement switches the
match to NumPy and is logged.

### 5.8 Sampling the mixture

One uniform number u in [0, 1) is drawn per round, when the round starts,
from OS entropy, and written to `round_<k>_u.json` in the match directory.
The same file records the round's anchor, from which section 6.2 measures
the deadline; `state` may create the file with the anchor alone, and the
first `recommend` adds u. A restarted `recommend` for the same round reads that file
and reuses u and the anchor; only a new round draws again. u comes from its own generator, separate from
the seeded one that builds the libraries. Every mixture the search produces
(each incumbent of section 6.2, and the final one) is turned into an action
by the same u: weights normalised to sum 1, actions in their fixed library
order, the first whose cumulative weight exceeds u, and the last action
with positive weight if rounding leaves u above the final cumulative sum. So there is exactly one
draw per round, and crash-safe incumbents do not add draws. Which incumbent
ends up final depends on the deadline, not on u, so the final action is a
fair draw from the final mixture, unless the parent's last check of section
6.2 replaces it; that happens only when re-flying our craft disagrees with
what the child stored, or the file is damaged, and is logged. The console shows only the final line; incumbent
files are read only by the parent at the deadline (section 6.2), never by
a person during the round.
Drawing again until an action looks better would change the strategy, so
the tool does not offer it. `--seed` fixes the draw for rehearsals and tests. Libraries and
solves are deterministic given the inputs; how far the search gets depends on
the deadline and the machine, so the log records the stage reached.

## 6. Operations (phase 4)

The duel submission flow gets its own runbook, `docs/runbook/duel.md`,
written in phase 4 and kept apart from the ranking runbook. This section is
the design it follows.

### 6.1 CLI

The duel CLI is `orbitwin/duel/cli.py`, run as `python -m orbitwin.duel` and
as its own console script `orbitwin-duel`; those are the entry points of the
numpy-and-scipy duel release. `orbitwin/cli.py` imports `config` (PyYAML),
`gmat_parser`, `io`, `pipeline`, `rank`, `script_renderer` and `verify` at
module level, all of
which run before `main()`. `main()` itself then calls `load_config`, which
reads YAML, and prints the historical-rulebook warning before it looks at
the command. So `orbitwin duel` reaches the duel CLI without them only if
those imports move inside the branches that use them and the `duel` branch
is taken first, before `load_config`. Phase 4 does both (no behaviour change
for the other commands). The import-boundary test runs all three entry points
in a fresh interpreter.

Subcommands, all offline:

- `params`: set and show the announced parameters for a match; writes
  `params.json` and echoes every value with its unit.
- `state`: read a round file or take manual entry; echo positions in km and
  velocities in km/s with derived radius, speed, altitude and a; write
  `round_<k>_state.json`. From round 2 on it also takes the announced
  executed impulses, the announced a and the announced closest approach
  (from the file or typed in) and runs the after-round check of section
  6.4, using the line we handed in for the previous round as `line` recorded
  it (below); if none was recorded, it asks for that line to be typed.
- `recommend --role {pursuer,evader} --round k`: anytime search; prints the
  line at the deadline.
- `choose-role`: the round-1 comparison of section 6.5, in its own
  directory.
- `line --role --round --dv x y z`: snap the vector to the 0.01 m/s grid,
  print the exact submission line with two decimals, parse it back, project
  it, and re-simulate the round against the
  opponent's coast, showing our a, our minimum radius and the closest
  approach. `line --parse "<text>"` checks a line read back from paper, and
  once it passes, records it as the line handed in for that round in
  `round_<k>_submitted.json`, which the next round's `state` uses.
- `referee`: one round from a state and the submitted lines. It takes at
  most one line per role for the round (PDF §4.1 allows one impulse per
  round) and refuses a second line for the same role and round, a line for
  another round, or an unknown role; a missing line is zero.
- `selfplay`: whole matches between named strategies (zero, greedy, random,
  engine) from a parameter file, with seeds.

### 6.2 Time budget

The organizer gives 10 minutes per round for everything, including reading
the USB and writing the line by hand. Default deadline for computation 7
minutes after the round's anchor (section 5.8), configurable. The anchor
is the wall-clock time (UTC) at which the first command to touch round k
started, `state` for round k or `recommend --round k`, whether or not that
command succeeded, so time lost to a refused file or a correction counts
against the round. Round 1 has no round file, so its first `recommend`
sets it. On the first write, `--window-opened HH:MM:SS` sets it earlier,
to when the organizer opened the window. After that it is never moved by
a restart, for any input hash: PDF §3 executes a late line as [0, 0, 0],
so a run restarted at minute 6 gets the minute that is left, not a fresh 7.
Only `--reanchor`, for an organizer who restarts the clock, moves it, and
the log records the old and new value. A restart that finds the deadline
already passed starts no search and goes straight to the deadline's steps
below, printing from the incumbents on disk; if none with its input hash
passes and the zero vector goes below our floor, the parent computes the
rescue action of section 4.5 itself, capped at 20 s, and prints that. An
anchor older than `DUEL.decision_s` still prints the same way, with a
loud warning that the window is over. An anchor in the future means the
clock moved back; the run warns loudly and measures the deadline from its
own start. A configured deadline at or above `DUEL.decision_s` is refused
when the parameters are loaded. Tests inject the clock. The recommender
always holds an answer:

- immediately: the zero vector, checked for Earth safety and flagged if not
  safe (it is what a late submission becomes anyway);
- within 60 s: the best worst-case pure action of a small matrix (about
  16 × 16, depth 1), with our floor judged per cell as in section 2.4;
- then the full libraries, the LP, the double oracle, and depth 2 on the
  support, replacing the incumbent each time a solve completes.

The deadline is enforced from outside the search. Each `recommend` run gets
a run id (time plus random suffix) and an input hash over the parameter
file, the state file, our role, the round and every reading in force. At
start the parent takes a lock file for the round, `round_<k>.lock`, which
records its pid, the child's process group and both processes' start times
(from `/proc/<pid>/stat`, so a reused pid is not mistaken for the original;
always by numeric pid, never `/proc/self`, and read-only, as section 5.6
allows);
a second `recommend` for the same round while the first is alive refuses to
start. If the lock is stale (its parent is dead), the new parent kills the
recorded process group, after checking that its leader's start time still
matches, so an orphaned child cannot keep writing incumbents. The child also exits
on its own as soon as its parent is gone (it checks `os.getppid()` between
solves). The new parent then moves into `round_<k>_runs/<old run id>/` every incumbent
and fallback file of that round whose input hash differs from its own, so
a restart after an input was corrected can never pick up the old run's
actions. Incumbents with the same input hash stay and remain eligible,
since they answer the same question. u always stays (section 5.8). The
parent then draws or reuses the round's u, writes the zero-vector fallback,
starts the search in a child process in its own process group
(`os.setsid`) and waits. The child first checks the zero vector against our floor; only if
the zero vector goes below it does the child compute the rescue action of
section 4.5 and write it to the fallback file in its place, since the
rescue changes the orbit and could, for example, move an evader's a out of
the band in round 6. It then writes each incumbent i as its own file,
`round_<k>_inc_<i>.json`, through a temporary file renamed into place
(`os.replace`), so every file on disk is a complete incumbent and earlier
ones stay available. Numbering continues from the highest index already on
disk. An incumbent holds the run id and input hash, the search stage it
came from (the 60 s small matrix, the full depth-1 solve, the double
oracle, depth 2), the mixture, the action u selects, its executed vector,
its line, and the facts the parent will check: our minimum radius over the
round and the time we would go below our floor, if we do. "Best" means the
latest stage reached and, within a stage, the highest index, so a restarted
run's quick incumbent never displaces an earlier run's deeper one.

At the deadline the parent kills the whole process group and goes straight
on. On Ctrl-C it kills the group, says the line is ready, and waits 3 s
before printing; a second Ctrl-C in that time aborts without printing, for
when the inputs turn out to be wrong. It then reads the best incumbent
whose input hash matches its own (any other is ignored and logged), formats
the line, parses it back and projects it, and checks only what it can check
alone: that the parsed integer triple equals the stored one, and that
re-flying our own craft with it gives the stored minimum radius to within
1e-6 km, the same crossing verdict under the floor rule of section 2.4,
and, when both radii are more than 1e-6 km below our floor, the stored
floor time to within 1e-3 s (the child computes them in batch, the parent
in scalar, so exact equality is not expected). When the minimum radius is
within 1e-6 km of our floor, the crossing time is ill-conditioned, so the
time is not compared; the action is a crossing under that rule for the
child as for the parent, the reading against us. A verdict that differs
between the stored facts and the re-flight, in either direction, is a
mismatch like a radius or time outside its tolerance, and starts the
walk-back below; with one rule on both sides it can happen only when the
radius lies within the batch and scalar difference of R_f + 1e-6 km. A
pursuer action that goes below our
floor after a capture is a legitimate choice and passes, since the solver
chose it knowingly (section 2.4). Only then does it print the line. A search
that hangs inside a library call cannot hold the line past the deadline. If
the child exits with an error, the parent says so and goes on exactly as at
the deadline: the best incumbent whose input hash matches its own, in the
order defined above (not the last one written, which after a restart with
unchanged inputs can be the new run's 60 s incumbent while an earlier
run's depth-2 one is on disk), the same check, and the same walk-back. If
the check fails, the
parent walks back through the incumbents with its input hash, in that
same order, and prints the best one that passes, else the fallback file,
with a warning in either case. The parent and child files (the lock, fallback,
numbered incumbents and their temporary siblings, the round's u, and the
`round_<k>_runs/` archive) are the working files section 5.6 allows, and
the `/proc/<pid>/stat` reads above are on its list as well. The independent implementation the user
supplied uses the same parent and child structure, which is where it comes
from.

`os.setsid` and `killpg` are POSIX. The design targets Linux; phase 4
records the match laptop's OS, and on another OS implements the
equivalent before the day (on Windows a new process group with
`CREATE_NEW_PROCESS_GROUP` and a job object that is terminated as a whole)
with the same deadline test.

### 6.3 Reading the round file

The format is not published (Q1). The reader accepts: a GMAT ReportFile
(whitespace columns with a header naming `X, Y, Z, VX, VY, VZ` per craft),
CSV with a header, JSON, and `key = value` text. It refuses anything it cannot
map to exactly two craft and six components each, and asks for manual entry.
Units are never converted silently. A radius under 6000 means the file is
in other units or columns are swapped, and the reader stops. A radius above
1e6 or a speed above 20 is possible (an escaping evader after long rounds;
a fast arc when ΔV_lim is large and there is no total cap), so the reader
asks a human to confirm it rather than refusing; a speed above 1000 is
certainly m/s and stops the reader. The
echo shows what it understood next to the raw text.

### 6.4 After-round check

Run by the `state` subcommand. With the previous round's file state and the
announced executed impulses, it re-simulates the round with the referee and
compares with the new file state and the announcement:

- the evader's announced a against eq. (2) on the new file state and against
  our prediction. This is the most sensitive check: a non-two-body
  ForceModel moves a by kilometres (section 2.3);
- the end positions;
- the announced closest approach, its time and the capture flag. A capture
  they report that we did not predict, or the reverse, points at how they
  sample (Q6).

Thresholds come from the digits actually in the file, through the
rounding shares of section 2.1: with position and velocity components at
resolutions δr and δv, a can be off by (√3/2)·((2a²/r²)·δr + (2a²‖v‖/μ)·δv)
from rounding alone, the third term of section 2.5 (0.024 km at GEO and
0.0016 km at 7000 km for δv = 1e-6 km/s). The tool warns at three times
that, and at three times the propagated rounding share for positions. For
the opponent's craft the announced impulse's own digits add a share,
‖Φ_rv(t)‖·(√3/2)·δ_Δv for an announced resolution δ_Δv (0.15 km after
5400 s and 2.2 km after a day on a 6878 km circle at two decimals in m/s),
which is included before any warning. Every residual below is taken
beyond these same shares. Using the round-to-nearest share errs towards
caution here: a source that truncates has a larger share, and a smaller
share leaves a larger residual, so a higher floor and an earlier warning.

Our own craft is re-flown with the line we submitted, not with the
announced impulse, and the two are compared first. If the announced
impulse for our craft differs from our line by more than the announcement's
digits can explain (a misread line, a unit slip on their side, a scaled
vector we did not expect), the tool says so loudly. Our executed impulse
is then not known, so that round contributes nothing that depends on it:
no position residual, and neither the drift nor the evader's own
comparison below. Its impulse-free comparison of the announced a with the
new file state still holds and still counts.

The residuals set two live floors:

- `dr_int_floor_km`, the floor of each craft's error envelope e(t) of
  section 2.5, from our own craft's position residual beyond the file's
  rounding share. Only our craft counts, because only its executed impulse
  is known exactly (it is on the grid); the opponent's is known only to the
  digits announced;
- `a_margin_floor_km`, from three comparisons, each beyond its rounding
  share. First, the method offset M: the evader's announced a against
  eq. (2) on the new file state. It involves no impulse and measures how
  the organizer computes a (μ, `SMA` against eq. (2)), in either role.
  Second, the drift D, in either role: our own craft's a predicted from the
  previous file state and our submitted line against eq. (2) on our new
  file state. Both inputs are exact, so this measures drift over the
  round, including a force model that is not two-body. Third, when we are
  the evader, E: our predicted a against the announced one, which already
  holds both effects. The organizer's a for the evader at the end of
  round 6 carries the method offset and that round's drift together, so
  the two add, and the floor is max(M + D, E), with E = 0 when we are
  the pursuer, so that the pursuer's floor is M + D.

The `state` subcommand writes both into the match's
`params.json` (our choices, section 2.0), which is how `recommend` receives
them; no other file carries them. Each round's contributions, one per
comparison, are stored with the hash of the inputs they were computed
from. M, D, E and the position residual are each the largest of their
contributions over the rounds as they now stand, and the floors are formed
from those, so M + D is never less than any one round's sum. So within a
match a floor
only rises as rounds are added, a later round with smaller residuals does
not lower it, and the `params` subcommand carries the floors over when it
rewrites the file. But if a round's inputs are corrected (a mistyped
T_round, a Δv in the wrong unit), `state` recomputes that round's
contribution and replaces it, and the log records the old and new value,
so an input error does not leave a floor behind for the rest of the
match.

### 6.5 Choosing the role

The better-ranked team has 10 minutes before round 1. The two roles see the
same game, but the conventions that resolve ambiguities against us flip with
the role, so the round-1 search runs in parallel once per convention, and
again for each with `early_entry_wins = true`: four searches at once with a
4-minute deadline each, which the 20-core box handles and the match laptop
must be checked for in phase 4 (on fewer cores, the two default-reading
searches come first). This is its own subcommand, `choose-role`. Each of its
searches works in its own directory under `role_choice/<convention>/` with
its own u and incumbents, measures its 4-minute deadline from the anchor in
its own u file as section 6.2 does for a round, and never writes the round files that `recommend`
uses, so the real round-1 `recommend` starts clean. The tool then shows,
side by side:

- hard facts first: whether the outer bound, on the band widened by the
  margin, says the evader cannot reach it at all (choose pursuer, provided
  we can keep our own craft above our floor; the tool shows whether the
  zero vector alone does so for all six rounds, and says when it does not),
  and whether the initial separation is already a capture. Under the default
  reading there is no matching rigorous fact for choosing evader: a cheap
  proof that the pursuer can never reach the evader in six rounds is not
  available. Under `early_entry_wins = true` the tool reports whether the
  exact round-1 window meets the narrowed band. That is not a proof either:
  the window ignores the Earth, and with the entry ordering resolved against
  us as evader, a capture later in round 1 still beats the entry, so it is
  shown with the round-1 values, not as a hard fact;
- the pursuer-convention value V_P and the evader-convention value V_E, and
  our prospects 1 − V_E as evader against V_P as pursuer, each also with
  `row_step_s = 0` and with the pessimistic row gap of section 2.6;
- whether the evader starts inside the interval or can reach it with its
  round-1 impulse, which under the early-win reading of section 3.6 decides
  the match for the evader in the pursuer convention and is one option in
  the evader convention, and V_P and V_E with `early_entry_wins = true`;
- V_P and V_E with `boundary_rows_judged = false` (section 2.6).

The comparison is biased, and the tool prints the known biases next to the
numbers. Towards the evader: the pursuer convention discounts captures
shorter than `row_step_s`, and the leaf understates a pursuer with several
rounds left (section 5.3). Towards the pursuer: C overestimates where the
evader's hiding area lies outside the pursuer's reach. The team decides. The
values are heuristic game values, not odds.

### 6.6 Writing the line

The line is copied by hand. All team members must be present (PDF §6), so a
second one is always there: they read it back from the
paper into `line --parse`, and the tool confirms that the parsed and
projected vector equals the recommendation digit for digit before it is
handed in.

### 6.7 The laptop

A violation of the network and AI rule loses the match. Before the match:
Wi-Fi and Bluetooth off (on Linux `rfkill block all`; on another OS its
own controls) and checked, no AI tool running
or installed in the session used (coding assistants, IDE completion), phones handed
in. The duel distribution's wheel and its offline bundle (section 2.0) are
installed on the match laptop and on a second laptop as a spare, each in an
environment of its own, never alongside the full `orbitwin` package,
whose modules it duplicates. The whole procedure is rehearsed offline.

## 7. Tests, by phase

Phase 1:
- [300, 400, 0] m/s with a 250 m/s cap executes as [150, 200, 0] m/s; zero
  stays zero; under the cap is unchanged; m/s to km/s and back round-trips.
- Capture between knots: the 2° crossing above, with its initial states and
  knot times written into the test so the segment's LB is fixed, is caught
  with R_cap = 3 km. With the radius set halfway between that segment's LB
  and the true minimum (about 2.34 and 2.40 km; the test computes both and
  asserts the gap), the refinement path must run and clear it. An
  implementation that calls LB ≤ R_cap a capture fails this case. These
  tests call the detector of section 3.2 with an explicit radius, not the
  referee, whose margins would move the radius.
- A long slow capture (dwell of minutes) terminates, with dwell bounds that
  bracket a dense exact measurement.
- Enter, leave and re-enter within one knot segment: the three passes of
  the 2° crossing inside a single 7000 s segment are each found, and the
  first entry is reported. (The passes are equally deep; the claim that
  the first entry, not the deepest point, is reported rests on the
  left-first recursion.)
- Dwell is per passage: two separate 0.6·`row_step_s` passages do not count
  for us as pursuer; one passage of 1.2·`row_step_s` does; a pass that ends
  the round inside R_cap counts whatever its dwell.
- The R_cap margin of section 2.5 shrinks the radius for us as pursuer and
  grows it for us as evader by the sum of both craft's error envelopes,
  using their end-of-piece values; a live floor from the after-round check raises it and a later,
  smaller residual does not lower it (phase 4, where the after-round check
  lives).
- Every action of ours is an integer centi-m/s triple with norm ≤ ΔV_lim;
  snapping picks the nearest admissible cell corner, maps a grid vector to
  itself (checked on all values 0.00 to 1000.00 m/s per component, where a
  float truncation changes several per cent of them), and works for a
  ΔV_lim that is not a multiple of 0.01; the printed line has exactly two
  decimals, never `-0.00`, and parses back to the same integers.
- The row-gap bound is recomputed per cell: a pursuer action that raises
  its orbit raises the bound for that cell; a hyperbolic arc gets a finite
  bound; a pair uses the larger of its two craft's bounds, and an
  opponent's dip below R_opp counts for us only if it lasts the pair
  bound, not its own smaller one.
- Regression shapes after the independent implementation's tests, written
  afresh: a fast capture hidden between knots; a narrow double pass inside
  one segment; a tangent graze and a near miss just outside R_cap; a
  one-metre Earth penetration.
- LB is the larger of the Taylor and secant bounds, and neither is ever
  above the dense exact minimum on random segments.
- Earth crossing between knots: an arc whose periapsis is 1 m under the floor
  while both knots are well above it is caught; a round of three periods
  judges each passage on its own.
- Asymmetric floors: an opponent between R_polar and R_eq has not crossed;
  our action dipping to R_eq + 5 km counts as our crossing at the moment it
  goes below R_eq + 10 km plus the 1e-6 km band of section 2.4. A minimum radius of R_f + 5e-7 km is our
  crossing, timed where the arc reaches R_f + 1e-6 km, and one of
  R_f + 2e-6 km is not; in phase 3 the solver's cell, the child's stored facts and the
  parent's re-flight give the same verdict on both. As pursuer, a capture whose first entry precedes
  that moment but whose `row_step_s` of dwell completes after it loses the
  cell; one whose dwell completes before it wins.
- Unequal initial epochs stop the reader; with `align_epochs = true` the
  earlier craft is coasted to the later epoch.
- Dwell rules: a capture lasting less than `row_step_s` does not count for us
  as pursuer and does count against us as evader; an opponent's crossing
  shorter than `row_step_s` below R_opp does not count for us.
- Counted events are timed by the passage that qualifies: an opponent that
  dips below R_opp first for 0.5·Δ and later for 1.2·Δ counts at the
  second passage's entry plus Δ; a capture with the same two passages is
  established at the second entry plus Δ, so our crossing between the
  first entry plus Δ and the second entry plus Δ beats it under the
  chronological reading; an opponent's dip that begins 0.3·Δ before the
  round end and holds there counts at the round end, not at its entry
  plus Δ, and so does a capture that does the same.
- Both event-order readings on a pursuer that crosses, then captures. As
  pursuer: the evader crosses, we cross later, and we capture in the round:
  the cell is ours; the same without the capture: a loss. The referee names
  the first to cross under the chronological reading and marks a double
  crossing without a capture unresolved under the priority reading.
- A capture at the round boundary; an initial state already inside R_cap.
- A `coast_many` batch with one row made to fail (an injected error) is
  re-run row by row; only that row is marked unusable and logged.
- An undecided graze sliver is a capture of us as evader and no capture as
  pursuer.
- High-eccentricity coasting through the duel's rollout succeeds and agrees
  with `propagate_universal` on every failing case listed in section 2.3, on
  a sweep e ∈ [0.999, 1.002] × durations from 60 s to 1 day, and on random
  states with e ∈ [0.9, 1.1] with unrounded components. (The core fix has
  its own tests; these pin that the duel path uses it.)
- The ε consistency check of section 3.5 trips on a corrupted end state and
  passes on the near-parabolic cases above.
- An evader whose a is inside the interval at the end of round 3 decides
  nothing, and the match continues (phase 1); phase 4 shows it with the
  early-win warning of section 3.6. With
  `early_entry_wins = true` the same match ends at t = 0⁺ of the round whose
  impulse put the evader inside. A capture later in that round loses to the
  entry under the pursuer convention and beats it under the evader
  convention. An evader that starts inside but burns out of the interval in
  round 1 has entered under the pursuer convention (the initial state
  counts) and has not under the evader convention (only the post-impulse
  state counts); one that starts inside and stays inside enters at t = 0⁺ of
  round 1 unless the initial separation is already a capture, and under the
  evader convention the entry counts only if round 1 then has no capture of
  the evader and no crossing of its own.
- The row-gap bound uses the selected table row, `row_gap_s_override`
  replaces it, and the c table is checked against fresh GmatConsole runs
  (marked `gmat`) for each stored setting.
- Round-boundary rows (marked `gmat`): a GmatConsole run whose Propagate
  ends between integrator steps writes a report row at its end state, in
  both `Propagate P(A, B)` and `Synchronized` forms, which is the GMAT side
  of `boundary_rows_judged` (section 2.6).
- `boundary_rows_judged`: with true, a capture inside R_cap at the round
  end for less than `row_step_s` counts for us as pursuer, and an
  opponent's dip below R_opp that holds at the round end counts for us;
  with false neither counts unless its passage lasts the bound; an action
  whose winning cells rest only on such an event is flagged.
- Terminal a with ‖r‖ ≠ a: an eccentric evader whose radius lies outside the
  interval while its a lies inside wins, and the reverse loses. One-sided
  interval. Hyperbolic evader loses a positive interval and wins a negative
  one that holds its a. An interval containing a = 0 is refused. Under a
  one-sided interval [a_lo, ∞), an escaping evader counts as inside for us
  as pursuer and outside for us as evader.
- The a margin narrows the band for us as evader and widens it for us as
  pursuer in the terminal judgement; a narrowed band that is empty falls
  back to the core of the real band (centre ± 10% of its half-width, in
  energy) with a
  flag, and (phase 2) the evader's terminal impulse then lands in that core. The
  margin's rounding term is the √3/2-share expression of sections 2.5 and
  6.4, 0.024 km at GEO for velocity components given to 1e-6 km/s.
- Reaching the interval before round 6 and leaving it in round 6 loses.
- A missing submission is executed as zero.
- Rollout against GmatConsole (marked `gmat`): several states, stated tolerance.
- `DuelParams` refuses foreign fields by name: total Δv cap, `kt`/`Ct`/`kv`/
  `Cv`, `T_max`, a 4·T_A horizon, a 100 s separation, a 5 km sphere not given
  as `r_cap_km`. The mapping from `DUEL`'s announced list to its fields
  covers every entry and every announced field exactly once.
- Import boundary: no import from `orbitwin.duel` into a module the boundary
  test calls ranking, or `config`; a fresh interpreter importing the duel
  package loads no ranking module, `yaml` or network module at import time
  (the CLI joins this test in phase 4).
- The referee refuses a second line for the same role and round.
- `tests/cross_profile/`, duel side: 300 m/s with ΔV_lim 250 executes at
  250 m/s; six rounds at a cap whose sum exceeds 3 km/s all execute in full;
  a second impulse for the same role and round is refused; reaching the safe
  a early and leaving it does not win; a 4 km closest approach is a capture
  with R_cap = 5 and not with R_cap = 3 (continuous checking,
  `row_step_s = 0`); the preliminary 4·T_A horizon is
  refused as a duel parameter.

Phase 2: the window against brute-force sampling of the ball, including a
band whose window exists in speed but whose arcs all cross the Earth;
rotations keep ‖v‖ to 1e-12 before snapping (after it, to within the
snap's effect), including U > 2‖v‖; every library action has
norm ≤ U; our terminal impulse as evader lands in the narrowed band; at
T_round = 86400 s the library keeps the 15-revolution intercept of section
4.4, and against an evader on a 12000 km circle at a fixed phase written
into the test it keeps the cheapest branch returned (about 1108 m/s before
projection); as pursuer, a call with a larger maximum returns no branch
above the bound that stays above our floor, and no kept branch dives below
our floor before the intercept; as evader, on a geometry chosen in the
test so that a branch with more revolutions than the pursuer-role cap
⌊t/P(max(a_m, a_f))⌋ is among the `lambert_keep` cheapest of an uncapped
call (the test asserts this precondition), the opponent's library keeps
that branch; the Lambert action count stays within its stated bound.

Phase 3: matching pennies 0.5 versus 1 with peeking; LP against known small
games, and the duality-gap check rejects a non-optimal pair; the file-access
audit of section 5.6; monotonicity of V; V(x, 1) against the exact round-6
value; the energy outer bound against brute-force sampled plans whose
impulses all happen above the role's floor R_e; under the pursuer
convention a greedy witness whose only dip below R_opp is one passage
shorter than the pair bound is kept, one whose passage lasts longer is
rejected, and under the evader convention any dip below R_f rejects it;
C at least its
end-of-round value when the evader's coast is reachable at T_round, and no
capture credit when the margin wipes out R_cap; the
forced-evader cone against brute force; the leaf's doom test uses the band
chosen by role, and an empty narrowed band falls back rather than calling
us doomed; the role-choice hard facts; one u per round selects every
incumbent's action, and a u above a cumulative sum that rounds short of 1
selects the last action with positive weight, never a zero-weight one;
depth-2 refinement covers every cell of each support
row and column. With `early_entry_wins = true`: a leaf whose one-impulse
window meets the band has V = 0 under the pursuer convention; under the
evader convention V is the smaller of the cone-restricted entry value and
the value of waiting, and with no window action above the evader's floor
only waiting counts; a leaf whose window misses the band but whose outer
bound reaches it is not doomed. The leaf's r_max bound is never below the
largest radius reached over the round by dense sampling of the impulse
ball (each sample flown); its apoapsis term is never below the largest
sampled apoapsis and within 2% of it on the review's LEO and GEO cases. The
leaf's floor R_e and capture radius R_c follow the role.
Backend: the same inputs through NumPy and JAX agree to 1e-9 km (skipped
when JAX is not importable; on CPU only, the NumPy path against scalar
`coast`); JAX without x64 is refused; a second call with the same padded
shape does not recompile; a failing JAX import falls back with a message; a
pair whose sample minimum is above R_cap but whose LB is not is never
cleared; every capture verdict comes from the CPU check.

Phase 4: the time budget (a 2 s deadline returns within a stated slack, and
the 60 s fallback exists), a `recommend` restarted 360 s after the round's
anchor printing at anchor plus the deadline rather than 7 minutes after the
restart, also when the restart carries a new input hash, an anchor set by
a `state` run that was refused and then corrected, `--window-opened`
setting the anchor earlier on its first write and ignored after it,
`--reanchor` moving it with a log entry, one restarted after anchor plus
the deadline printing at once from the incumbents on disk without
starting a search, the same restart with no matching incumbent and an
unsafe zero vector printing the parent's rescue action within 20 s, an
anchor older than `decision_s` printing with a warning, an anchor in the
future warned about and replaced by the run's own start, a configured
deadline of `decision_s` or more refused, a restarted `choose-role` search
keeping its own anchor, the line round-trip and paper read-back, state
files in each layout and unit refusals, the entry-point import test, a
restarted `recommend` reusing the round's u, a restart after the opponent's
state was corrected that leaves more old incumbents on disk than the new
run writes and still prints the new run's action (the old ones archived
and ignored), a child that raises after a restart with unchanged inputs
with the parent then printing the earlier run's depth-2 incumbent, not the
new run's later 60 s one, a final check that fails on the latest
incumbent and falls back to the next best incumbent with the same input
hash that passes (section 6.2), a restarted run's 60 s incumbent not
displacing an earlier run's depth-2 one, a stale lock whose pid was reused
not killing the unrelated process, a minimum radius within 1e-6 km of our
floor treated as crossing without a time comparison, an incumbent
stored as not crossing whose re-flight crosses (and the reverse) treated
as a mismatch that starts the walk-back, `choose-role` leaving no file that
the round-1 `recommend` reads, `params` keeping the live floors when it
rewrites the file, a round's floor contribution replaced (not kept) when
that round's inputs are corrected, rounding at its worst corner (every
component off by half a unit, so each vector off by its full √3/2 share)
in the file's positions and velocities and in the opponent's announced
impulse never raising a floor or a warning while a residual just beyond
that share does, an announced impulse for our craft that
differs from our line flagged, with that round contributing its method
offset M and nothing else, a round whose method offset is 0.3 km and
whose drift is 0.2 km beyond their shares setting `a_margin_floor_km` to
0.5 km rather than 0.3 km, both as pursuer (no E) and as evader with M and
D of opposite sign (so E is about 0.1 km and the largest single comparison
would give 0.3 km), one round's method offset adding to another round's
drift,
a stale lock whose orphaned child is killed before the new run starts,
the file-access audit of section 5.6 run across a start, a refused second
start and a restart over a stale lock, passing with the `/proc/<pid>/stat`
reads and record-file appends those make and failing on an injected open
of `/proc/<pid>/environ`, of `/proc/<pid>/stat` for writing, or of the
record file for reading, a
second Ctrl-C within the 3 s hold after the first aborting without a line, a second `recommend` for a locked round
refused, a restart with unchanged inputs keeping its earlier incumbents,
the `duel` branch of `orbitwin` taken before
`load_config`, and self-play sanity. Self-play
here is refereed with one fixed set of readings, because the conventions
shape our choices, not the outcome of a simulated match: continuous
checking (`row_step_s = 0`), no margins, the priority event order,
`early_entry_wins = false`, and a double crossing recorded as unresolved
and counted apart from wins and losses. In it, the engine pursuer beats a zero-burn evader that starts
inside its reach; the engine evader reaches the band against a distant
zero-burn pursuer; a zero-burn evader outside the band loses at round 6.

## 8. What is unknown

| Unknown | Where it matters | Default until answered |
|---|---|---|
| Round file format and digits | Reader | Several layouts, manual entry, unit checks |
| μ and the ForceModel, including whether the gravity field is degree 0 | All predictions | GMAT's measured μ; the announced a checked every round |
| GMAT integrator and tolerance | a margin, after-round check | Measure on R2026a, watch the residuals |
| Interval endpoints, how a is computed | Round 6 | Margin against us |
| Whether digits beyond two decimals are used, rounded or truncated | Submission | Moot: our lines carry exactly two decimals on the grid (section 2.1) |
| Capture with a crossing in the same round; both crossing | Event ordering | Against us; referee shows both |
| Earth surface | Crossing | Own floor R_eq + 10 km; opponent below R_polar |
| Capture and crossing judged continuously or on rows, and the step | PDF §5.1, §5.2 | Per-cell row-gap bound c·r_max^1.5/√μ from a pre-fitted table (× 1.5), most cautious row by default; override available; values also shown at 0 and at max(2700 s, 2 × the bound) |
| Whether the state at every round boundary is judged | Boundary captures and crossings, PDF §5.1, §5.2 | `boundary_rows_judged = true`: an event holding there counts whatever its dwell; flagged, and shown with false (section 2.6); Q6 |
| Whether reaching the interval early wins, and if so whether an entry beats a later capture or crossing in the same round and whether the initial state counts | Whole match | PDF §5.3 followed; the switch resolves the sub-questions by role; risk shown each round (section 3.6) |
| How a malformed or illegible line is executed | Submission | Read-back check (section 6.6) |
| Units of ΔV_lim, the interval and the announced Δv | Parameters | Entered with explicit units, echoed |
| Role token, separator, decimals accepted | Submission | `追捕方` / `逃脫方`, ASCII commas, exactly two decimals on the 0.01 m/s grid |
| Whether the two initial epochs are equal | Round 1 | The reader stops on unequal epochs; a flag coasts the earlier craft forward once the organizer confirms that is what they do |
| Whether a laptop with radios switched off is acceptable | Conduct | Radios off, checked by the team |
| U, R_cap, T_round, interval, initial states | Everything | Announced on the day |

## 9. Questions for the organizer

1. What is the round file on the USB: format, column order and headers, digits,
   whether it carries the epoch and in which time system? PDF §6 says a
   scaled Δv is listed with its executed value; is an unscaled one listed
   too, and in the same file? Can a sample be shared before the match?
2. Which μ, and which ForceModel exactly: `PointMasses = {Earth}`, or
   `PrimaryBodies = {Earth}` with degree and order 0 (GMAT's default is 4) and
   which potential file? Which integrator and accuracy, and which GMAT version?
3. Is the safe interval closed or open, and can it be negative (an escape
   zone)? If it is one-sided, [a_lo, ∞), does an evader on an escape orbit
   count as inside? Is a computed by eq. (2) from GMAT's
   final state, or read from GMAT's `SMA` parameter, and from the full
   precision state or the rounded one published on the USB? To what
   precision is it compared?
4. If a capture and an Earth crossing happen in the same round, does the
   earlier event decide, or does capture always come first as the numbered
   list suggests? If both craft cross, what is the result?
5. Which Earth surface defines a crossing: a sphere (which radius) or GMAT's
   ellipsoid, judged by `Altitude < 0` as in the ranking tutorial?
6. Are "任一時刻 Δr(t) ≤ Rcap" and the Earth crossing checked continuously or
   on report rows, and at what step (the propagator's `MaxStep` and
   accuracy)? Are both craft in one `Propagate`, in which form
   (`Synchronized` or not), and in which order? Is the state at the end of
   every round, the one announced, always checked for capture, so that a
   capture that holds at a round end counts however briefly Δr was inside
   R_cap? Is the announced closest approach interpolated? This decides
   whether a fast pass through R_cap counts. Asked first, with Q9.
7. Exactly how should the line be written: role in Chinese (`追捕方`,
   `逃脫方`) or a code, round numbered from 1, which comma? (We write exactly
   two decimals, so how further digits are treated does not affect us.)
8. When do the 10 minutes start, and who decides what is late? How is a
   line executed that is malformed, illegible or has the wrong round
   number?
9. PDF §2 says the pursuer must capture "在逃脫方抵達安全空域之前", and the
   PDF §5 preamble says the evader "抵達安全空域即獲勝". We read PDF §5.3 as
   judging the interval only at the end of round 6, so reaching it earlier
   changes nothing. Is that right? If entry does win, does an entry at the
   start of a round still win when a capture or an Earth crossing happens
   later in that round? Asked first, with Q6.
10. Do both craft share one initial epoch?
11. In which units are ΔV_lim, the safe interval, the announced executed Δv,
    the announced a and the announced closest approach given?
12. May we use our own laptop with Wi-Fi and Bluetooth switched off, and will
    it be inspected?

## 10. What the engine does not claim

It does not compute a probability of winning, and it does not find a global
equilibrium of the continuous game. Its matrix values hold for the sampled
action sets and the leaf heuristic. Depth 2 refines only part of the matrix,
so refined and unrefined cells sit on different scales and the result is not
a depth-2 solution. Four choices are not resolved against us: early entry
into the interval follows PDF §5.3 when we are the pursuer (section 3.6);
when the margin empties the band, the evader aims at the core of the real
band (section 2.5); and the propagator-setting defaults (the row gap of
section 2.6 and the integration margins of section 2.5) are the worst of
the settings measured, not the worst possible, in both roles; and
`boundary_rows_judged` counts an event that holds at a round boundary
whatever its dwell, a capture of ours as pursuer or an opponent's crossing
in either role (section 2.6). Each is flagged or shown
alongside its worst case. The double oracle's last improvement is a lower
bound on exploitability, nothing more. GMAT on the organizer's machine decides every outcome; our referee
predicts it and is checked against the announcement every round.

## 11. Cross-check against an independent implementation

On 2026-11-01 the user supplied a second duel implementation, package
`orbitwin_duel` 0.1.0, built with another assistant, tested on CPU only and
never run on a GPU or against GMAT. Its rules PDF hash matches ours. It is
not imported or vendored. It was read as a second opinion and these ideas
are taken from it, rewritten here and credited where they are used:

- its universal-variable solve (half-angle C, a doubling bracket on χ,
  safeguarded Newton and bisection, a Lagrange-identity check, explicit
  failure codes with a DOP853 fallback; its stress run reports no failures
  on 20000 states including e = 1 ± 1e-12 to 1e-4), as cross-checks for the
  core coast fix (section 2.3);
- the secant bound, combined with ours (section 3.2);
- closed-form crossing times with a numeric path near e = 1, and the
  comparison against DOP853 events (section 3.3);
- the parent and child process structure with atomic incumbent files
  (section 6.2);
- the shapes of four regression tests (section 7).

Where it differs and this design goes further:

- pursuer intercept actions: its library is zero, axis shells, equal-energy
  rotations, interval targets and Sobol samples. Its self-play record, a run
  it labels a correctness smoke test with 16 actions and no oracle
  iterations, has the pursuer burning in round 1 and coasting through
  rounds 2 to 6 while the evader wins; that is weak evidence of strength
  either way, but nothing in its library aims at the evader. Section 4.4
  adds Lambert and aimed actions;
- the leaf: its rounds 1 to 5 use a hand-set continuation value,
  0.15 + 0.5·pressure + 0.25·burden, where pressure comes from the sampled
  minimum distance and burden from an energy-shortfall term; lookahead is
  off by default and, when enabled, covers a configurable number of support
  pairs (sections 5.3, 5.5);
- conventions. It already treats uncertain cells as losses for the acting
  side and marks double crossings, Earth grazes and cases at the a boundary
  as unresolved. It counts continuous capture with no dwell rule (section
  2.6), checks the opponent's crossing at the equatorial sphere (2.4), has
  no a margin beyond a 1e-6 km tolerance (2.5), no early-entry reading
  (3.6), one common epoch (2.2, Q10), no role-choice analysis (6.5) and no
  after-round comparison with the announcement (6.4);
- a GPU stack the second machine can run: it pins `cupy-cuda12x==13.6.0`,
  which conflicts with that machine's fixed CuPy 14.2.0; our optional GPU
  backend is JAX in float64 (section 5.7).
