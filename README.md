# OrbitWin

[![CI](https://github.com/PanNavi23/orbitwin/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PanNavi23/orbitwin/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

[繁體中文](README.zh-TW.md)

OrbitWin plans and verifies multi-impulse spacecraft interceptions for GMAT
scripts. Given the organizer's script for a two-craft interception problem, it
searches for burn schedules under the contest's delta-v caps, replays the best of
them in a stock GmatConsole, and writes the script and report to upload. It was
built for the 2026 Orbit Design Competition of the Taiwan Space Agency (TASA); each
contest's rules are a rule pack ([docs/RULE_PACKS.md](docs/RULE_PACKS.md)).

The search runs on a two-body model and only proposes candidates. GmatConsole
replays decide which ones work, and the organizer's validator decides the score.
OrbitWin does not claim that any answer is optimal.

## Install

You need Python 3.11 or newer and GMAT R2026a
([download](https://sourceforge.net/projects/gmat/files/GMAT/)).

```bash
git clone https://github.com/PanNavi23/orbitwin && cd orbitwin
uv venv && . .venv/bin/activate
uv pip install -r requirements-dev.txt && uv pip install --no-deps -e .

make test          # about 5000 tests in one process; see "Development" for the time it takes
orbitwin --version
```

Without `uv`, use `python3 -m venv .venv` and
`pip install -r requirements-dev.txt && pip install --no-deps -e .`.
Install from `requirements-dev.txt`, not from `pyproject.toml`: the package metadata
gives version ranges, and the tests and the contest-day tools expect the pinned
versions. `requirements-dev.txt` reads `vendor/wheels/constraints.txt` as
constraints, which also pin the packages those depend on: the set the offline
bundle and CI install. `orbitwin` and the contest-day scripts refuse to run when numpy, SciPy or
PyYAML is not the version `requirements.txt` pins. Set `ORBITWIN_ALLOW_UNPINNED=1`
to run anyway. An install from a wheel has no `requirements.txt` and is not checked
([docs/WORKSPACE.md](docs/WORKSPACE.md)).

The contest machine may have no network. Fetch the offline bundle beforehand on a
machine that has one, and carry it with the repository:

```bash
bash scripts/fetch_offline_wheels.sh   # 39 wheels for Python 3.11 to 3.14
bash scripts/bootstrap_offline.sh      # on the contest machine: installs only the wheels SHA256SUMS.txt lists, with --require-hashes
. .venv/bin/activate                   # in every new terminal
```

## Contest day

The full procedure is [docs/RUNBOOK.md](docs/RUNBOOK.md).

The shortest path is a workspace ([docs/WORKSPACE.md](docs/WORKSPACE.md)): one
folder holds the team number, the GMAT paths, the values announced on the day and
the official script, and `orbitwin run` hands them to `orbitwin day`, which runs
steps 1 to 7.

```bash
orbitwin init contest --team-number <team> --expect-commit <the commit chosen the day before> \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP"
cp OFFICIAL.script contest/input/
$EDITOR contest/announce.toml          # T_max first; the coefficients once they are announced
orbitwin run contest                   # stops at each readback; after two people check it, add --confirm-readback <step>
orbitwin status contest                # what is missing, how far it got, which two files to upload
```

The same steps by hand, which are the commands `orbitwin day` runs, for the ranking
round (rules of 2026-10-06, distance term revised 2026-11-04). The preliminary's
configs are kept in `config/history/preliminary/`.

```bash
. .venv/bin/activate   # python and orbitwin below must be the ones in .venv
export GMAT_CONSOLE=/opt/GMAT/bin/GmatConsole
export GMAT_STARTUP=/opt/GMAT/bin/gmat_startup_file.txt

# 1. Keep the original input, read-only
cp OFFICIAL.script input/original/official.script && chmod a-w input/original/official.script

# 2. Write the announced T_max into the day's config (do not edit the YAML by hand);
#    two people check the values it prints. If it warns that gmat.workers exceeds the
#    cores, write a new file with --gmat-workers <at most the number it gives> and use it
#    in place of run/day.yaml in steps 3 and 4.
python scripts/ranking_day_config.py --output run/day.yaml --horizon-s <T_max> --team-number <team>

# 3. Inspect, then the first search (run/search-01)
orbitwin inspect input/original/official.script --config run/day.yaml \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" --output run/scenario.json
bash scripts/competition_day.sh input/original/official.script run run/day.yaml

# 4. Once the coefficients are announced: a new config (delta-v in m/s and C_r in km,
#    as the 2026-11-04 revision fixes them), a second search (run/latest), then rank
python scripts/ranking_day_config.py --output run/day-scored.yaml --horizon-s <T_max> \
  --kt-per-s <kt> --ct-s <Ct> --kv <kv> --cv <Cv> --delta-v-unit m/s --cr-km <C_r> \
  --day-config run/day.yaml
bash scripts/competition_day.sh input/original/official.script run run/day-scored.yaml
orbitwin rank input/original/official.script --config run/day-scored.yaml \
  --candidates run/latest/candidates.json --top 10 --output run/rank-scored.json \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP"

# 5. Certify the chosen candidate
orbitwin certify input/original/official.script --config run/day-scored.yaml \
  --candidates run/latest/candidates.json --candidate C-xxxxxxxxxxxxxxxx \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" --runs 3 --output run/final
```

With a console, every command first runs a state probe, so the initial states are
the ones GMAT reads. A template with an EarthFixed insertion state cannot be read
without one.

Search again after the coefficients are announced: the ranking search picks its
starting points from them. `orbitwin rank` re-ranks any saved archive in seconds and
warns when a term has gone flat, usually because a coefficient was typed in the
wrong unit. `ranking_day_config.py` also prints two ceilings, the most any plan can
score and the most any miss can score, and the distance term at 10, 100, 1000 and
10000 km and where it falls to 49 and 25 points (about `5 + 0.2·C_r` and
`5 + 1.048·C_r`). Given `--zero-burn-dmin-km`, the zero-burn closest approach that
the first solve prints, it prints that distance's score too, so two people can check
C_r. A miss is scored on its closest approach with
`100/(1 + e^{((d − 5)/C_r)²})` and can beat an expensive hit.

## Checks before upload

On 2026-09-05 every check the repository had passed, and the upload still scored
22 points less than it should have. These three tools came out of that day.

```bash
# Terrain: which arrival times are reachable and how many impulses to plan for
# (the delta-v column is an upper bound, not a floor)
python scripts/safe_delta_v_floor.py official.script --config run/day-scored.yaml \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP"

# Independent audit: replays the plan densely from the organizer's initial states and
# does not trust what the solver reported (the replay is two-body; under a richer force
# model it does not judge the interception or the score, and its AUDIT line says so)
python scripts/audit_burn_table.py official.script burns.json --config run/day-scored.yaml \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP"

# Before upload: reads the .txt you will upload and checks that its last row is inside
# 5 km. The report is named after the template's placeholder, ODC_2026_TeamXX.txt
# becoming ODC_2026_Team54.txt for team 54; certify prints the name.
python scripts/check_report_rows.py run/final/submission/ODC_2026_Team54.txt \
  --config run/day-scored.yaml --certificate run/final/submission/certification.json
```

The last check is the lesson of those 22 points: the organizer's validator took the
completion time from the report's last row, not from the first row inside the
sphere. A script that ran on to `4TA` was scored as never intercepting
([docs/LESSONS_2026-09-05.md](docs/LESSONS_2026-09-05.md)).

`check_report_rows.py` requires the report to be the bytes the certificate lists
and the script to be the certificate's own submitted script, also when format flags
such as `--delimiter` are given. A certificate field of the wrong type, or a NaN or
infinite number, exits 2. `--script` makes it split columns by that ReportFile's
Delimiter, WriteHeaders, FixedWidth and Add list. The script runs inside this
checkout (it reads `src/`) and needs Python 3.11 or newer and the project's venv.
With numpy, SciPy or PyYAML off the pinned versions it refuses and exits 2.

Two points differ in the ranking round (step 7 of the runbook):

- `audit_burn_table.py` replays two-body. Under the rules' section 7 perturbation
  model it still judges the impulse limits, the total, the separation and the time
  window, and the surface and altitude walls up to T_max. It does not judge the
  interception or the score: on R1 it called a GMAT-certified hit a 19.97 km miss.
- A scored miss is held to T_max. With `--config`, as above, T_max is the config's
  `policy.horizon_s`; leave `--horizon-s` out, because a value other than the config's
  exits 2. Without `--config`, pass `--horizon-s` with the day config's
  `policy.horizon_s` (not the solver-clock `horizon_s` in certification.json); without
  it the check prints `VERDICT: unchecked` and exits 1. When the last row is at T_max,
  `check_report_rows.py` exits 0 and prices the closest row.

## Output names follow the evidence

You do not choose the name of the script a run writes:

| Candidate stage | File name | Evidence needed |
|---|---|---|
| `analytical` / `renderable` | `analytical_candidate.script` | two-body model and event refinement |
| `public_gmat_verified` | `public_gmat_candidate.script` | a successful replay in any stock GMAT |
| `organizer_gmat_verified` | `organizer_gmat_candidate.script` | `gmat.organizer_compatible` and a matching fingerprint |
| `certified` | `team_solution.script` | the row above, an exact-script replay, report parity and repeated exact runs |

`solve` never writes `team_solution.script`. `summary.json` records what the run did
and, when it produced nothing, why.

## Layout

```text
src/orbitwin/    the package: parser, Lambert, event detection, renderer, GMAT runner, CLI
config/          quick, standard, deep, OOD and ranking profiles
scripts/         checks before upload, fuzzing, release validation, the GMAT test double
docs/            math, policy, runbook, measurements, reviews (index: docs/README.md)
tests/           unit, integration and slow OOD tests
examples/        our two-burn template, ranking rehearsals R1 to R9, different-plane scenes
notebooks/       a Jupyter front end, plots only; not in the release archive
input/original/  where the official script goes on the day (read-only after step 1; gitignored)
vendor/          the offline wheel bundle and GMAT install notes (no binaries in git)
output/          run outputs (gitignored)
```

## Development

```bash
make test          # about 5000 fast unit and integration tests, one process
make test-isolated # the same, one process per file (7% slower, 2026-08-21) -> output/make/
make test-slow     # the slow searches: OOD, multi-burn, post-intercept reachability
make fuzz          # 1000 seeded Lambert OOD cases
make lint          # ruff; any finding fails CI
make typecheck     # mypy; any error fails CI
make coverage      # 96% of statements (2026-11-04, workers and subprocesses included) -> output/make/
make validate      # the release validation campaign -> output/validation/ (not in git)
make package       # manifest, ZIP, and the fast tests run again on the unpacked copy -> dist/
```

CI runs `test` on Python 3.11, 3.12, 3.13 and 3.14 and `checks` (`make lint`, then
`make typecheck`, which runs even when lint fails) on every push to main and every pull
request.
Both jobs run on GitHub's `ubuntu-latest`. A private repository with its own runners
names them in repository variables, each a JSON list of runner labels:
`ORBITWIN_RUNS_ON_TEST` for `test` and `ORBITWIN_RUNS_ON_CHECKS` for `checks`, for
example `gh variable set ORBITWIN_RUNS_ON_TEST --body '["linux","x64","my-runner"]'`.
`ORBITWIN_RUNS_ON_TEST` must name only the repository's own runners; to use GitHub's
runners, unset it. A public repository ignores the variables. In
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), the comment above `jobs:` lists
the organization and repository settings that keep a fork's pull request off your
runners and what a re-run reads, and the comment on the `test` job gives a test leg's
time limit on each kind of runner.

`make validate` writes its evidence to `output/validation/` on the machine that runs
it. The evidence is not committed and not in the release archive; its
`environment.json` records the commit it ran on (`commit`) and whether tracked files
had changes (`tracked_changes`). `make package` refuses a tree with changed tracked
files, and a HEAD that is neither origin/main (fetched first) nor a `v<version>` tag.
So the order is: `git fetch`, check out origin/main, `make validate`, read
`output/validation/`, `make package`. Validation takes 30 to 80 minutes; if main
moves meanwhile, `make package` refuses, and you validate again on the new
origin/main so that `commit` is the commit you package.

`make validate` checks first that the interpreter `PYTHON` names imports the
project's dependencies at the pinned versions. `PYTEST_ARGS` reaches only its
coverage step and may hold only `-n`, `--numprocesses`, `--maxprocesses`, `--dist`,
`--maxschedchunk` and `--durations` (for example `PYTEST_ARGS="-n 16"` with pytest-xdist): an argument
that selects tests would shrink the evidence, so it is refused, as is a set
`PYTEST_ADDOPTS` or `PYTEST_PLUGINS`. One run per checkout at a time
(`output/validate/lock`). A run that fails or is interrupted puts
`output/validation/` back as it was and moves its own output to
`output/validate/failed-<time>-<pid>`. Only SIGKILL leaves `output/validate/previous`,
the evidence from before the run; the next run refuses until someone compares it,
restores what they need and deletes it. On 2026-11-04 a whole
`make validate PYTEST_ARGS="-n 16"` took 1794 s on an i5-14500 (coverage 399 s).
The offline bundle has no pytest-xdist, so a run installed from it, offline
(`bwrap --unshare-net`), took 4711 s the same day.

`test-isolated` and `coverage` write their logs, XML, coverage data and pytest cache
to the gitignored `output/make/` (`REPORT_DIR` moves it). `make clean` runs
`scripts/clean.sh`, which needs no Python and removes what `make package`
(`build_release.clean()`) removes: caches, coverage data, `src/*.egg-info` and
bytecode, but not `dist/`, `build/`, `output/`, `.venv` or another checkout. A test
holds the two implementations to the same result.

In CI, `make test` runs under pytest-xdist: 164, 158 and 257 s on Python 3.11, 3.13
and 3.14 on a runner of our own pinned to 16 CPUs (main, 2026-11-02), and about
6 minutes per leg from start to end on the same runner on 2026-11-12. GitHub's runners
for public repositories have 4 vCPUs and take several times longer. In one process it
is much slower: on 2026-11-03 the 2476 tests main had then took 1024 s on an
i5-14500 (14 cores, 20 threads) with CPython 3.13.14, at a one-minute load average of
5.5. `make package` stops the tests of the unpacked copy when their output stalls for
10 minutes or the run passes 4 hours (faulthandler writes a test's stack to the log
once it has run for 300 s, so a hung test is stopped within 900 s). It sends SIGINT,
then ends every process in the group, and leaves the output so far beside the
archive. Ctrl-C, SIGTERM and SIGHUP do the same.

## Known limits

- The parser accepts a narrow subset of GMAT syntax, the one contest templates use
  ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), "Parser boundary"): explicit `Create`
  commands, one assignment per object field, exactly two spacecraft, explicit
  `ImpulsiveBurn.Element1/2/3`, single-line `Propagate` commands with elapsed-time goals
  (synchronized, or split with equal goals on every craft) plus the template's own epoch
  alignment, an alternating `Propagate`/`Maneuver` sequence in a slotted template (an open
  template has no `Maneuver`, and its maneuvers are added), and `Toggle`, `PenUp` or
  `PenDown` of plot subscribers alone and `MarkPoint` or `ClearPlot` of XYPlots alone. On
  anything else it stops instead of guessing.
- The search is two-body. If the template's force model has J2, drag, SRP or third
  bodies, only the GMAT replay tells whether a plan works.
- Exactly collinear or antipodal Lambert geometry has no unique transfer plane. A
  Cartesian shooting fallback covers it, and the GMAT replay still decides.
- Stock GmatConsole refuses a script that creates an `OpenFramesInterface`.
  `certify` and `baseline` then fly a copy that leaves out only the plot lines,
  proven line by line against the submitted script (`gmat.console_copy`, on in the
  ranking profiles), and the uploaded script keeps its plots. The diagnostic copies
  `solve` flies leave the plots out too. With `gmat.console_copy` off, or when no copy
  can be proven, `certify` stops at `exact-run-01` and the report has to come from the
  GMAT GUI ([docs/RUNBOOK.md](docs/RUNBOOK.md) step 6b).
- Nothing proves an answer globally optimal. Several Lambert branches, Sobol starts,
  multi-impulse fallbacks, Pareto archives and OOD tests lower the risk of missing
  one.
- Which fields the organizer allows to change, how a failed mission is scored, score
  rounding and the validator's tolerances are not all confirmed
  ([docs/SCORING_STRATEGY_AND_OPEN_QUESTIONS.md](docs/SCORING_STRATEGY_AND_OPEN_QUESTIONS.md)).

## License

Apache-2.0 ([LICENSE](LICENSE)). The license does not cover the competition
organizer's problems, rules and templates ([NOTICE](NOTICE)).

The package was named `tasa-orbitwin`; the rename to `orbitwin` kept the version, 0.3.0
([RELEASE_NOTES.md](RELEASE_NOTES.md)), so only the name tells the two apart. If an
environment has the old name installed, run `uv pip uninstall tasa-orbitwin` first,
then install as above. With both names in one environment, uninstalling the old one
deletes the `orbitwin` command (and, for a non-editable install, `orbitwin/*.py` as
well) while `uv pip list` still lists `orbitwin`;
`uv pip install --reinstall --no-deps -e .` repairs that.
