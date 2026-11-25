# OrbitWin 0.4.0 release notes

The first public release. Since 0.3.0:

- The contest's rules come from rule packs (`orbitwin rules`, docs/RULE_PACKS.md).
  OrbitWin ships `tasa-odc-2026`: the preliminary, the ranking round as revised
  on 2026-11-04, its onsite version of 2026-11-08, and the duel.
- Ranking-round scoring: the announced T_max and C_r, misses priced on their
  closest approach, and an impulse over 250 m/s, a total over 3 km/s or a
  crossing of the Earth treated as a failure rather than a deduction.
- A ranking search that places up to `search.ranking_max_burns` capped impulses
  and prices hits and misses by the announced score; `solve` then corrects the
  best hits on GMAT's own flights and replays the misses worth more than the
  zero burn (docs/GMAT_IN_THE_LOOP.md).
- Craft that start on different epochs, and initial states in any frame GMAT
  reads: a GMAT state probe, the epoch alignment line, and burns in the head
  start (docs/ASYNC_START_AND_INSERTION.md). `orbitwin set-epoch` sets
  SC_Human's epoch where the onsite rules let a team choose it.
- `certify` flies a copy without the plots when GmatConsole refuses a template
  for its OpenFrames or GroundTrack objects alone, ends with a PRE-CHECK line,
  and binds the uploaded script and report by SHA-256 in certification.json;
  `scripts/check_report_rows.py --config --certificate` checks the upload.
- `orbitwin day` runs the runbook's steps 1 to 7 in one command, stopping at
  each readback for two people to confirm; `orbitwin init`, `run` and `status`
  keep a contest day in a workspace, and the announced values are typed by two
  people separately and compared (docs/WORKSPACE.md, docs/ANNOUNCEMENT.md).
- Every refusal names an error code (docs/ERROR_CODES.md), and
  `orbitwin --version` names the commit an installed package came from.
- `orbit.propagate_universal` runs its Newton loop without scipy's per-step
  overhead (#227). The tests' quick search of
  `examples/two_burn_rendezvous.script`, with the local refinement, took 6.59 s
  instead of 13.53 s in the pull request's measurement (median of 5 on 8 CPUs
  of an i5-14500, 2026-11-24), and wrote the same files but for the script's
  path in `scenario.json` and `summary.json`.
- CI's fast suite runs with `--dist load --maxschedchunk 1` instead of
  `--dist worksteal` (#227): 181-183 s instead of 210-212 s at 20 workers on an
  i5-14500 (2026-11-24), and 164.5 s instead of 185.9 s at 16 workers in a
  simulation on CI's recorded test durations.
- In a public repository CI runs on GitHub's `ubuntu-latest` runners whatever
  `ORBITWIN_RUNS_ON_TEST` and `ORBITWIN_RUNS_ON_CHECKS` say (#237).
- CONTRIBUTING.md says how a pull request reaches the code and which checks to
  run, and SECURITY.md how to report a vulnerability privately (#236).
- The README and the main docs are in English, with Traditional Chinese copies
  beside them.
- Licensed under Apache-2.0 instead of MIT. Every contributor agreed, on the
  pull request that made the change.
  NOTICE says what of the competition organizer's material it does not cover.
- The distribution is now `orbitwin` (was `tasa-orbitwin`, at the same version
  0.3.0); the import name and the CLI are unchanged. Uninstall `tasa-orbitwin`
  from an existing environment before installing `orbitwin`: with both
  installed, uninstalling the old name deletes the `orbitwin` command (and, in
  an install that is not editable, the package's modules) while `uv pip list`
  still shows `orbitwin`.
  `uv pip install --reinstall --no-deps -e .` repairs it.
- The package ships a PEP 561 `py.typed` marker.
- CI tests Python 3.12 as well.

# OrbitWin 0.3.0 release notes

- Reconstructed the previously missing project payload from the surviving README/policy files.
- Added self-contained universal-variable propagation and 3-D Lambert enumeration.
- Added multi-revolution interval scanning and independently propagated endpoint validation.
- Added a Cartesian shooting fallback for exact antipodal/collinear geometry.
- Added VNB, LVLH and inertial burn-frame conversion.
- Added zero-burn, one-burn, and global Sobol two-burn search.
- Added cubic-Hermite first-entry/minimum-distance event detection.
- Added fail-closed GMAT Script parsing, rendering and immutable-field policy checks.
- Added isolated GmatConsole runner, output-path startup derivation, diagnostic sibling, baseline and certification.
- Added deterministic fake-GMAT integration tests, 5,000-case Lambert fuzz and random different-plane Script matrix.
- Added a normalized-timestamp release ZIP, payload SHA-256 manifest, archive sidecar hash, extraction verification and extracted fast-test smoke run. Validation evidence itself contains real run timestamps, so repeated archives are not claimed to be byte-identical.

A real organizer-compatible GMAT differential campaign remains mandatory before scientific/competition certification.

## Packaged validation

- 46 fast unit/integration tests passed.
- 2 slow OOD search tests passed.
- Fast-suite statement coverage: 79%.
- 5,000 seeded 3-D/multi-revolution Lambert fuzz cases completed with 0 failures.
- 20 random different-plane Script cases completed with 0 pipeline exceptions.
- Exact final-run orchestration repeated three times with stable Report hashes under the deterministic test double.
- Real organizer-compatible GMAT validation remains an explicit external blocking gate.
