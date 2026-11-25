# GMAT on Linux

NASA's public R2026a package provides Ubuntu and RHEL Linux archives and runs `GmatConsole` from the release `bin` directory. Public R2026a is useful for development, but the competition must use the organizer's exact build.

## Recommended layout

```text
/opt/GMAT-R2026a/
  bin/GmatConsole
  bin/gmat_startup_file.txt
  lib/
  data/
```

## Preflight

```bash
export GMAT_CONSOLE=/opt/GMAT-R2026a/bin/GmatConsole
export GMAT_STARTUP=/opt/GMAT-R2026a/bin/gmat_startup_file.txt
bash scripts/preflight.sh
```

Run it with the project's venv active; it calls `python`. It runs `baseline` on
`examples/synthetic/DifferentPlane_Circular.script` and writes its results,
including GmatConsole's own log, under `output/preflight/<time>`. The test run
also uses `$TMPDIR` and `.pytest_cache/`. Pass another script only if it has
no OpenFrames objects; the workshop template is refused (see below).

OrbitWin creates a derived startup file per run. It changes only `OUTPUT_PATH` and `LOG_FILE` to absolute isolated paths. The original startup file is not modified. Relative GMAT data/plugin paths continue to be resolved by the stock executable.

## Tests against GmatConsole

```bash
export GMAT_CONSOLE=/opt/GMAT-R2026a/bin/GmatConsole
export GMAT_STARTUP=/opt/GMAT-R2026a/bin/gmat_startup_file.txt
make test-gmat
```

Every `gmat`-marked test runs the stock console, each run in its own output
directory, and fails unless the two variables name a GMAT install
(`gmat_install` in `tests/conftest.py`). `GMAT_CONSOLE` is a path, as `run_gmat`
takes it, and is not looked up on `PATH`.
Point `TMPDIR` at an existing directory with room: Python falls back to `/tmp` when it does not exist, and a full `/tmp` makes GMAT's file
writers fail.

GmatConsole skips GUI plugins, so a script that creates OpenFrames objects is
refused whole (58 errors and exit status 1 on the workshop template).
Verification runs a private copy with the plots commented out; `orbitwin
baseline` and certify's exact reruns run the bytes as written and fail on such
a template. `tests/integration/test_gmat_contract.py` pins this, the official
report's header column for column, and two runs at once keeping their own
reports.

## Required differential campaign

Before competition day, run at least:

1. baseline on a plot-free copy of the Workshop Script (GmatConsole refuses the original, see above);
2. different-plane synthetic Script;
3. high-plane-change two-burn Script;
4. 100 seeded random candidates;
5. three exact reruns of the final candidate.

Compare analytical and GMAT states, burn-frame conversion, first-entry time, minimum distance and total delta-v. Any unexplained systematic offset blocks release.
