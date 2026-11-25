# docs/

[繁體中文](README.zh-TW.md)

Ordered by what you are about to do. Files with a `.zh-TW.md` copy beside them have a
Traditional Chinese version; the English one is the reference.

## On the contest day

| Document | What it answers |
|---|---|
| [`RUNBOOK.md`](RUNBOOK.md) | Every step from installation to upload, with the three gates before submission and the stop rules |
| [`WORKSPACE.md`](WORKSPACE.md) | The workspace: `orbitwin init`, `run` and `status`, the announced values in `announce.toml`, error codes |
| [`ERROR_CODES.md`](ERROR_CODES.md) | Each code of `orbitwin: ERROR [<code>]:` and its exit status |
| [`ANNOUNCEMENT.md`](ANNOUNCEMENT.md) | Two people each type the announced values as written: the format, the comparison of the two copies, the comparison with the script |
| [`DESIGN_VARIABLES.md`](DESIGN_VARIABLES.md) | Values the rules let a team choose (SC_Human's epoch): `orbitwin set-epoch` and the record it leaves |
| [`LESSONS_2026-09-05.md`](LESSONS_2026-09-05.md) | Eight things **measured** in the preliminary of 2026-09-05: the validator reads the last row, the units, how the coefficients reverse the search strategy, the reachable upper bound |
| [`SCORING_STRATEGY_AND_OPEN_QUESTIONS.md`](SCORING_STRATEGY_AND_OPEN_QUESTIONS.md) | What the score formula decides, and the questions still waiting for the organizer's written answer |

## Gold and bronze matches

| Document | What it answers |
|---|---|
| [`DUEL_ENGINE.md`](DUEL_ENGINE.md) | The six-round pursuit-evasion game: the rules clause by clause, the referee and the game solver, the unknowns and the questions for the organizer |
| [`DUEL_ENGINE_OPEN_FINDINGS.md`](DUEL_ENGINE_OPEN_FINDINGS.md) | Review findings on the duel engine's design still open, each with the section and the phase it touches |

## Rules and assumptions

| Document | What it answers |
|---|---|
| [`COMPETITION_POLICY.md`](COMPETITION_POLICY.md) | Which rules the code encodes, and which module enforces each |
| [`ASSUMPTIONS_AND_GAPS.md`](ASSUMPTIONS_AND_GAPS.md) | Assumptions used only for analytical seeds, and the permissions not yet granted |
| [`SOURCES.md`](SOURCES.md) | What the organizer's preliminary rules, 0510 briefing slides and workshop template are and where each came from, and GMAT's download links |
| [`ASYNC_START_AND_INSERTION.md`](ASYNC_START_AND_INSERTION.md) | Craft on different epochs and EarthFixed insertion: the epoch alignment, the time origin, the d_min window |
| [`RULE_PACKS.md`](RULE_PACKS.md) | Each round's rulebook as a TOML file in a rule pack, and `orbitwin rules` to list them |

## Mathematics and numerics

| Document | What it answers |
|---|---|
| [`DIFFERENT_PLANES_MATH.md`](DIFFERENT_PLANES_MATH.md) | The 3-D derivation of an interception between different planes, and why it cannot be reduced to a coplanar Hohmann transfer |
| [`NUMERICAL_RESOLUTION.md`](NUMERICAL_RESOLUTION.md) | The numerical resolution of candidate ranking: how small a difference is still real |
| [`OOD_STRATEGY.md`](OOD_STRATEGY.md) | How far the out-of-distribution tests reach, and what they do **not** promise |

## Engineering

| Document | What it answers |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The data flow and the trust boundaries; why the parser supports a narrow subset on purpose |
| [`GMAT_IN_THE_LOOP.md`](GMAT_IN_THE_LOOP.md) | How `solve` corrects the best plans on GMAT's own flights |
| [`GMAT_SETUP_LINUX.md`](GMAT_SETUP_LINUX.md) | Installing the public R2026a on Linux, and how it differs from the organizer's build |
| [`STACK_ANSWER_MAP.md`](STACK_ANSWER_MAP.md) | Which layers change the answer when changed, and which do not (measured 2026-08-26) |
| [`PERFORMANCE.md`](PERFORMANCE.md) | Where the time goes, and which speedups measured **not worth it** (numba, GPU) |
| [`ADVERSARIAL_REVIEW.md`](ADVERSARIAL_REVIEW.md) | The adversarial review of 0.3.0 and the gates open at the time |
| [`audit/2026-08-19/`](audit/2026-08-19/) | The four P0 findings fixed then, with snapshots of their evidence (history, not the current code) |
