# Contributing to OrbitWin

[繁體中文](CONTRIBUTING.zh-TW.md)

This file says how a change gets into OrbitWin: what happens to a pull request, how
to set up and test, and the rules a change is held to. What OrbitWin does is in
[README.md](README.md).

## How a pull request reaches the code

This repository is exported from a private mainline, and pull requests are not
merged on GitHub. Open yours against `main`. A change is applied only once CI passes
on the pull request. A maintainer then applies it to the private mainline, adjusting
it if the export needs that. The export publishes a file tree, not the mainline's
commits, so the next export commit carries your change and credits you with a
`Co-authored-by:` line. The maintainer closes the pull request with a link to that
commit, and closes the issue it links: GitHub does not close an issue named by
`Fixes #` in a pull request that is closed without a merge.

## Set up

You need Python 3.11 to 3.14 (CI tests each of the four), [uv](https://docs.astral.sh/uv/)
and `make`. Install as [README.md](README.md), "Install", says.

CI installs the same, plus what the Install steps of
[.github/workflows/ci.yml](.github/workflows/ci.yml) name. CI does not install
OrbitWin itself: it imports it from `src` (`PYTHONPATH: src` in that file).
The editable install, `uv pip install --no-deps -e .`, gives you the `orbitwin`
command, which CI does not have. A test that runs OrbitWin runs
`sys.executable -m orbitwin` or calls `orbitwin.cli.main`; it does not call
`orbitwin` by name. A test that runs it in a subprocess sets `PYTHONPATH` to the
absolute path of `src`: CI's `PYTHONPATH` is the relative `src`, which a
subprocess in another working directory does not find.

The fast suite runs programs it writes to `TMPDIR`, so `TMPDIR` must not be on a
filesystem mounted `noexec`.

### Change a pinned version

1. Change the pin in `requirements.txt` or `requirements-dev.txt`, whichever has it.
2. Change the same package's line in `vendor/wheels/constraints.txt`.
3. Refetch the offline bundle as [vendor/wheels/README.md](vendor/wheels/README.md)
   says. The fetch runs `$PYTHON -m pip download`, so name an interpreter that has
   pip: `PYTHON=/path/to/python3 bash scripts/fetch_offline_wheels.sh`. A venv made
   by `uv venv` has no pip. If the refetch fails on a conflict with a line of
   `constraints.txt`, because the new version needs another version of a
   dependency, change that line too and refetch. Commit `constraints.txt` with the
   `SHA256SUMS.txt` the fetch writes.

The fast suite checks the pins, the bundle's manifest and the wheel counts the docs
quote against one another: see `tests/unit/test_wheel_constraints.py`.

## Run the checks

[README.md](README.md), "Development", lists the `make` targets and what CI runs.
CI does not run `make test-slow`, `make fuzz` or `make test-gmat`: run the ones your
change touches, and say in the pull request which you ran.

`make test` runs in one process. To run the fast suite on several CPUs, run it as
the Fast suite step of [.github/workflows/ci.yml](.github/workflows/ci.yml) does,
with the packages that file's Install step adds and the variables its `env` sets.

Plain `pytest` is not `make test`. `pyproject.toml` deselects no marker, so plain
`pytest` also collects the `slow` and `gmat` tests, and the `gmat` tests fail without
GMAT.

### Tests against GMAT

```bash
export GMAT_CONSOLE=/opt/GMAT-R2026a/bin/GmatConsole
export GMAT_STARTUP=/opt/GMAT-R2026a/bin/gmat_startup_file.txt
make test-gmat
```

The `gmat`-marked tests run a stock GmatConsole R2026a. README.md, "Install", links
the download. [docs/GMAT_SETUP_LINUX.md](docs/GMAT_SETUP_LINUX.md) gives the layout
these paths assume, a preflight, and the `TMPDIR` the tests need. Mark a new GMAT test
`gmat` and give it the `gmat` fixture of `tests/conftest.py`, whose `gmat_install`
checks the two variables.

The other suites need no GMAT; they run a test double, `scripts/fake_gmat_console.py`.

## Rules for a change

- Add a regression test with every behaviour change: a test that fails without the
  change and passes with it. Add adversarial tests too: the inputs near the change
  that must still be refused. Do not delete an adversarial test, for any reason.
  Do not lower a safety margin to make a test pass.
- Fail closed on GMAT syntax the parser does not know. The parser accepts a narrow
  subset on purpose ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), "Parser
  boundary"). On anything else it raises `ParseError`, printed as
  `orbitwin: ERROR [parse]:` ([docs/ERROR_CODES.md](docs/ERROR_CODES.md)), and
  rewrites nothing. A change that widens the subset adds tests for the new form and
  for the nearby forms that must still be refused.
- Do not catch an exception and then report success.
- Do not label an analytical (two-body) estimate as GMAT verified.
- Give every GMAT run its own run directory. Two runs never share a ReportFile path.
- Keep diagnostic subscribers, the ReportFiles and `Report` commands added to check
  a run, out of a submission script.
- Put units in names: `inc_deg`, `dt_s`, `policy.dv_limit_km_s`.
- Annotate every function in `src/orbitwin`, the package `make typecheck` runs mypy
  on, with `disallow_untyped_defs = true` and `python_version = "3.11"`
  (`pyproject.toml`), so mypy also flags a call that Python 3.11 lacks.
- Keep seeds fixed and orderings stable, so that a run gives the same answer twice.
- Keep ruff at zero findings without a new `noqa`. The ruff tests of
  [tests/unit/test_policy_docs_match_code.py](tests/unit/test_policy_docs_match_code.py)
  hold the tree to the suppressions it keeps on purpose (`KEPT_NOQA`) and to the
  reviewed `[tool.ruff]` table (`RUFF_CONFIG`). A new suppression or a change to
  the table goes into that file, with the reason, in the same pull request.
- Keep every `.script` file ASCII: GMAT R2026a refuses a script with any byte above
  127 (`tests/unit/test_source_script_encoding.py`).
- Keep each module in `src/orbitwin/` on its side: the core both games share imports
  only the core, and the ranking modules and the duel (`src/orbitwin/duel/`) do not
  import each other. A new module goes into its set in
  `tests/cross_profile/test_import_boundaries.py`.
- Never commit the competition organizer's material ([NOTICE](NOTICE)). The tests
  that need it skip in the public repository (`organizer_file` in
  `tests/conftest.py`). Write a new test on a script of your own or one in
  `examples/`.

## New contests

A contest's rulebook is a rule set in a rule pack:
[docs/RULE_PACKS.md](docs/RULE_PACKS.md) says how to write one, what needs code,
and which commands take which rule sets. A new rule set in the shipped pack changes
the list `orbitwin rules --json` prints, which `tests/unit/test_rule_packs.py`
pins, so update that list too. The contest-day commands `orbitwin init` and
`orbitwin run` take only `DAY_RULES` in `src/orbitwin/workspace.py`.

## Documentation

The docs are in English. A doc with a `.zh-TW.md` file beside it has a Traditional
Chinese copy with the same sections, and the English one is the reference
([docs/README.md](docs/README.md)). When you change a doc that has a copy, change the
copy in the same pull request, section for section.

If you cannot write Traditional Chinese, say so in the pull request. Some values in
the copies are checked against the code: when one needs your new value, the fast
suite fails and names the file; change the line there too.

## Commits

Write the subject line as an English sentence in the imperative that says what the
change does.

Sign off every commit with `git commit -s`. The `Signed-off-by:` line it adds
certifies the [Developer Certificate of Origin 1.1](https://developercertificate.org/)
for the commit. Unless you state otherwise, a contribution you intentionally submit
for inclusion is under the Apache License 2.0 ([LICENSE](LICENSE), section 5).

## Pull requests

- Link the issue the pull request resolves (`Fixes #123`).
- Say what changed and why, and which checks you ran with their results. The pull
  request template asks for both.
