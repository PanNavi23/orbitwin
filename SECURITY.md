# Security policy

## Report a vulnerability

Report it through GitHub's private vulnerability reporting: on this repository's
**Security** tab, choose **Report a vulnerability**. GitHub opens a draft security
advisory that only you and the repository's maintainers can see. Do not report a
vulnerability in a public issue, pull request or discussion.

Give in the report:

- what is affected: the file, the command or the function;
- the version: the output of `orbitwin --version`, and from a clone the output of
  `git rev-parse HEAD`;
- the steps that show the problem, with the smallest input that does;
- what an attacker can do with it.

Do not include the competition organizer's material ([NOTICE](NOTICE)). If the
problem needs a GMAT script to show, use one of your own or one in `examples/`.

## Scope

In scope, the places where OrbitWin reads input it does not trust:

- GMAT script parsing (`src/orbitwin/gmat_parser.py`): a script the parser accepts
  and then renders or runs in a way it should have refused;
- the output-file fields of a script (`gmat_runner.OUTPUT_FILE_FIELDS`), which must
  be bare file names: a script that makes GMAT write outside the run's directory;
- the YAML file given to `--config`;
- the release and install tools: `make package` (`scripts/build_release.py`,
  `scripts/make_manifest.py`), `scripts/fetch_offline_wheels.sh` and
  `scripts/bootstrap_offline.sh`. The bootstrap checks every wheel against
  `vendor/wheels/SHA256SUMS.txt` and installs them with `--require-hashes`. pip
  itself is not hash-checked when the interpreter has `ensurepip`: the new venv gets
  that interpreter's own pip. Only an interpreter without `ensurepip` gets pip from
  the bundle's wheel, installed with `--require-hashes`;
- `.github/workflows/`: GitHub Actions runs a pull request's code. Which runners
  it may run on is set in [.github/workflows/ci.yml](.github/workflows/ci.yml) and
  in the GitHub Actions settings of the repository and its organization. The
  comment above `jobs:` in ci.yml says how ci.yml picks them.

For example: a GMAT script or a config that makes `orbitwin` run a command or write
a file outside its output directory, or an offline install that takes a wheel that
`SHA256SUMS.txt` does not list.

Trusted, so out of scope. Whoever can write one of these can run code as you:

- installed rule packs: they are Python packages, and `orbitwin` imports every
  package registered under the `orbitwin.rulepacks` entry point group
  (`src/orbitwin/rules.py`);
- the GmatConsole that `--gmat-console`, `[gmat] console` in `orbitwin.toml` or
  `GMAT_CONSOLE` names: `orbitwin` runs that program;
- the GMAT startup file (`--startup-file`, `[gmat] startup_file`, `GMAT_STARTUP`) and
  every file and `PLUGIN` it names: GMAT loads them, and the copy the runner writes
  for each run changes only `OUTPUT_PATH` and `LOG_FILE`
  (`gmat_runner.derive_startup_file`);
- a day folder's `state.json`: `orbitwin day --resume` runs the console and startup
  file it records, checked only against the SHA-256 values in that same file.

Also out of scope:

- GMAT itself, and the Python packages OrbitWin depends on (`requirements.txt`):
  report those to their own projects;
- a wrong orbit, score or verdict that gives an attacker nothing: open a bug report.
