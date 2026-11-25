# Which layers of the stack change the answer

Measured 2026-08-26 against the 47 open pull requests, at `origin/main` = `0312277`.

Reading 9,101 lines of diff tells you what each pull request *intends*. Running the
same search on every branch and comparing what comes out tells you what it *did*.
This document records the second, produced by `scripts/pr_answer_delta.py`.

## The stack is a tree, not a list

```
main
 └ #1 → #2 → … → #32 → #33            33-layer linear spine
         #33 ┬ #34 → #36 → #38 → #40   leaf A
             └ #35 → #37 ┬ #39 → #42   leaf B
                         └ #41 ┬ #43 → #45 → #46 → #47   leaf C
                               └ #44                     leaf D
```

Three fork points (`#33`, `#37`, `#41`) and four leaves. Merge order along the spine
is forced; the only integration decisions are at the forks.

## Method

1. Every distinct branch head is solved exactly once: `orbitwin solve` on
   the organizer's workshop template, each branch's own copy, with that
   branch's own `config/quick.yaml`, 48 branches in total. Since #223
   `scripts/pr_answer_delta.py` solves our `examples/two_burn_rendezvous.script`
   instead, so a rerun measures another scenario.
2. `candidates.json` and `summary.json` are hashed after per-run provenance is
   stripped (`gmat_run_id` carries a timestamp and a nonce; `startup_sha256` hashes a
   derived file embedding a per-run output path, so two identical runs disagree on
   them by design).
3. Each pull request is compared against **its own base branch**, not against the
   previous layer of some linearisation. A per-tip walk cannot express a fork and
   recomputes the shared spine once per leaf.

## Result

| Verdict | Count | Pull requests |
| --- | --- | --- |
| `ANSWER-CHANGED` | 10 | #4, #8, #11, #12, #13, #15, #22, #24, #36, #47 |
| `SUMMARY-ONLY` | 0 | — |
| `no-op` | 37 | the remainder |
| `BROKEN` | 0 | — |

The run behind this table is kept in git history as
`9f40d2d:validation/pr_answer_verdicts.json` (`git show` it). The command under
"Reproducing" reads the pull requests open when it runs, so today it judges
other ones; it writes into `output/answer_map/`, which is not committed.

## What `no-op` does and does not mean

`config/quick.yaml` sets `gmat.verify_top: 0`, and `vendor/gmat` holds a README and no
binary. The probe therefore never enters the GMAT verification path, which the
project's policy then named the only authority. A `no-op` verdict on a pull request that edits
that path is **unverified**, not **verified unchanged**:

| Class | Count | Pull requests |
| --- | --- | --- |
| `no-op`, no GMAT surface; confirmed unchanged | 16 | #2, #6, #7, #10, #21, #23, #26, #28, #29, #30, #31, #33, #34, #39, #40, #42 |
| `no-op`, edits `verify.py` / `gmat_runner.py` / `gmat_parser.py` / `report_parser.py`; untested here | 19 | #1, #3, #5, #9, #16, #17, #19, #20, #25, #27, #32, #35, #37, #38, #41, #43, #44, #45, #46 |
| `no-op`, GMAT behaviour orchestrated in `pipeline.py`; the branch is not taken at `verify_top: 0` | 2 | #14, #18 |

The first two classes are assigned by changed-file path. `#14` and `#18` are listed
separately because that proxy places them outside the GMAT surface while their stated
purpose ("Run the search sweep and GMAT verification concurrently", "Say what the GMAT
verification actually did") is squarely inside it. Treat the proxy as a starting point,
not as a finding.

## The one direct contradiction

`#13 perf/event-and-derivative-hot-paths`, titled *"Cut event detection and propagation cost
without changing any answer"*, changes the candidate set relative to its base.
Either the title is wrong or the optimisation is. A safety margin is never lowered to
make a test pass, and wrong physics, frame, unit or time semantics come first in review,
so a performance layer that silently moves the answer is the highest-severity item in
the stack.

`#14 perf/parallel-search-and-verification` is the mirror image: 8 files, +324/−84,
and no change to the answer at all. That is consistent with a pure concurrency change,
but its GMAT half is behind the gate this probe cannot open.

## Baselines measured at the four leaves

| Leaf | Tests (`not slow and not gmat`) | ruff | mypy |
| --- | --- | --- | --- |
| #47 `feat/look-around-what-the-sweep-found` | 299 passed, 0 failed | 5 errors | blocked |
| #44 `fix/the-count-no-longer-means-what-it-did` | 293 passed, 0 failed | 5 errors | blocked |
| #42 `fix/the-heredoc-classifier-reads-the-wrong-command` | 302 passed, 0 failed | 5 errors | blocked |
| #40 `fix/the-section-list-is-a-hand-list-too` | 258 passed, 0 failed | 5 errors | blocked |

The ruff and mypy results are identical at all four leaves, so neither is a regression
introduced by any pull request. `make typecheck` currently reports
`numpy/__init__.pyi:737: error: Type statement is only supported in Python 3.12 and
greater` and stops before reaching `src/orbitwin`, which means the type gate in
`Makefile` is presently inert. That is a toolchain configuration fault, not a code
fault, and it is worth its own layer.

## Standing gaps

1. The repository has no `.github/workflows`. All 47 pull requests carry zero checks,
   zero reviews and zero comments; nothing mechanical has ever verified any layer.
2. There are no `gmat`-marked tests, so the GMAT verification path has no
   automated coverage of any kind.
3. `make typecheck` cannot reach the source tree (above).

## Reproducing

```bash
git worktree add --detach /tmp/answermap origin/main
python scripts/pr_answer_delta.py \
    --worktree /tmp/answermap \
    --out output/answer_map \
    --python "$(command -v python)"
```

Requires `gh` authenticated against the repository. The run solves 48 branches at
roughly 8 s each.
