# Workspaces

A workspace is one folder for one contest day. It holds what `orbitwin day`
otherwise takes as flags, so the day runs from files a person fills in rather
than from a long command line or a site-specific wrapper script.

```text
contest/
  orbitwin.toml   the team, the rule set, GMAT (written by orbitwin init)
  announce.toml   the values announced on the day (filled in by hand)
  input/          the organizer's .script, and every file it reads (names starting
                  with "." and links that lead nowhere are left out; the script
                  itself, a link to a folder, or a folder that cannot be read is
                  refused)
  days/           day folders, one per run of the day (docs/RUNBOOK.md)
  upload/         the uploads; upload/latest points at the one to send
```

## Commands

```bash
orbitwin init contest --team-number 54 --expect-commit "$(git rev-parse HEAD)" \
  --gmat-console /opt/GMAT/bin/GmatConsole --startup-file /opt/GMAT/bin/gmat_startup_file.txt
cp OFFICIAL.script contest/input/
$EDITOR contest/announce.toml     # T_max first; the coefficients when announced
orbitwin run contest              # stops at each readback: read it, then
orbitwin run contest --confirm-readback 2
orbitwin status contest
```

`orbitwin init` refuses a folder that already holds `orbitwin.toml` or
`announce.toml`, and leaves neither behind when it fails; a folder under the
checkout's `output/`, which `orbitwin day` refuses; and a rule set `orbitwin day` does not run (only
`tasa-odc-2026/ranking` so far). Without `--gmat-console` and `--startup-file`
the run takes `GMAT_CONSOLE` and `GMAT_STARTUP` from the environment.
`--expect-commit` is the commit chosen the day before (docs/RUNBOOK.md step 0;
`orbitwin --version` names it for a release): a day starts only with it, set
here or later in `orbitwin.toml`, or with `--allow-non-main`, which the upload's
note records.

`orbitwin run` starts a day folder under `days/` when there is none, and
otherwise goes on with the newest one (by its UTC name; other folders there are
not day folders; one without a readable `state.json` is refused,
`day-unreadable`). When that one has delivered and `upload/latest` points at its
upload, `run` says so and starts nothing (`day-delivered`). It passes the per-run flags of `orbitwin day`
(`--confirm-readback`, `--accept-config-warnings`, `--accept-rank-warnings`,
`--exclude`, `--include`, `--by-eye-passed`, `--force-latest`,
`--allow-non-main`, `--min-free-gb`) through unchanged. Where the day prints
`orbitwin day --resume <day folder> <flags>`, `orbitwin run <workspace> <flags>`
does the same; the coefficients go into `announce.toml` instead of flags.

A day folder keeps the values it started with: T_max, the boundary, the
coefficients once given, the script and the files beside it, the team, the
craft names, `base`, `expect_commit`, the GMAT pair and workers, the decisions
and the upload folder (so a
copied or moved workspace never resumes a day folder that uploads into the old
one, unless `upload/` is a link both copies share). When the workspace now
says otherwise, `run` lists each difference and starts nothing
(`day-inputs-changed`); put the value back, or start a new day folder with
`orbitwin run --new <workspace>`. The new folder runs every step again.

`orbitwin status` prints the script it found, what `announce.toml` still lacks,
each day folder's steps, confirmed readbacks and uploads, where `upload/latest`
points, and what to do next: the readback to confirm, the warning flags a stop
asked for, the coefficients the day waits for, or the command to go on with.
Whatever `orbitwin run` would refuse is listed as a problem. It exits 0 whenever it could read `orbitwin.toml` and list `days/`;
problems with `announce.toml`, `input/`, a day folder's `state.json` or the note
beside the latest upload are listed, not raised.

## From an installed package

The commands the day runs ship inside the package with the ranking profile:
`ranking_day_config`, `check_report_rows` and `audit_burn_table` as
`python -m orbitwin.tools.<name>`, and step 4's search script as
`orbitwin/tools/competition_day.sh`, which stops unless the `python` on PATH
imports that same package. `scripts/<name>` runs the same code in a checkout. A day records which commit ran it: from a git
checkout, an unpacked release archive, or an installed package's own record.
Installed from git (`uv pip install git+https://github.com/PanNavi23/orbitwin@<commit>`),
the package records that commit in its `direct_url.json`; `orbitwin --version`
names it, `expect_commit` is checked against it, and every file under
`orbitwin/` is held to the sha256 the installer recorded (RECORD): an edited,
missing or added file, a linked folder, or a `.pyc` outside `__pycache__`
stops the day, and so do a `direct_url.json` or an `orbitwin` command in `bin/`
that is not the file the installer wrote. Bytecode cannot be held to its source,
so `orbitwin day` and `orbitwin run` start themselves again with an empty
bytecode cache, and each step compiles into one emptied when the run starts: no
`.pyc` found beside the sources is run, and provenance refuses a process that was
not started that way. The few lines that do the restart run before anything
checks them; they are the one part this cannot hold. The rest of the environment is not judged: other packages, and the
`.pth` files and `sitecustomize` Python runs at startup, which provenance lists
beside the upload (`startup_files`). Installed from an index or a wheel file, it records no commit, and
`orbitwin run` there needs `--allow-non-main`, which the upload's note records. Such a package does not check that numpy, SciPy and PyYAML are
the versions requirements.txt pins either: that file is the checkout's. A bare
`orbitwin day` from it puts `run/day/` and `run/upload/` under the directory it
is started in (a workspace names its own).

## announce.toml

Every value starts commented out. Delete the leading `# ` of a line to give it.

| key | unit | `orbitwin day` flag |
|---|---|---|
| `horizon_s` | s | `--horizon-s` (needed to start) |
| `mission_boundary_km` | km, from the Earth's centre | `--mission-boundary-km` |
| `[score] kt_per_s, ct_s, kv_per_m_s, cv_m_s, cr_km` | 1/s, s, 1/(m/s), m/s, km | the five score flags; all or none |
| `[decisions] closest_approach_window` | `after_alignment` or `from_origin` | `--closest-approach-window` |
| `[decisions] no_head_start_burns` | true or false | `--no-head-start-burns` |
| `[decisions] tmax_as_announced` | true or false | `--tmax-as-announced` |
| `[decisions] report_filename_as_written` | text | `--report-filename-as-written` |
| `[decisions] accept_announced_horizon_before_perigee_s` | s | `--accept-announced-horizon-before-perigee-s` |

An unknown key, a value of the wrong type, a number that is not finite or not
positive where it must be, or some of the five coefficients without the rest
is refused before anything runs. The day itself still checks every value as
it always has.

## orbitwin.toml

`format`, `rules`, `team_number`, and optionally `pursuer_name`, `target_name`,
`base` (the profile step 2 copies), `expect_commit` (docs/RUNBOOK.md step 0),
`min_free_gb` (the free disk every run needs, 20 GB unless set here or given
with `--min-free-gb`) and `[gmat] console`, `startup_file`, `workers`. A relative path is read from
the workspace folder.

## --json

`init`, `status` and `run` take `--json`. A refusal, a usage error among them,
prints `{"error": {"code": ..., "message": ...}}` on stdout and exits 2. `run --json`
sends the day's own output to stderr and prints, on stdout, the exit status
with the workspace's status after the run. The other commands' codes are in
docs/ERROR_CODES.md. The workspace's:

| code | meaning |
|---|---|
| `workspace-missing`, `workspace-invalid`, `workspace-exists` | `orbitwin.toml` absent, unreadable or off; or init would overwrite one |
| `announce-missing`, `announce-invalid`, `announce-incomplete` | `announce.toml` absent or off; or T_max, or some coefficients, not given |
| `template-missing`, `template-ambiguous`, `input-name` | not exactly one `.script` in `input/`; a file name that is not UTF-8, a linked script, or a link to a folder |
| `rules-unknown`, `rules-unsupported` | init named a rule set that does not exist, or one the day does not run |
| `gmat-missing` | init named a GMAT file that does not exist |
| `usage` | a flag the command does not take or a value it refuses; init given half the GMAT pair |
| `environment`, `io` | `run` off the pinned packages; a file that could not be read or written |
| `day-delivered`, `day-inputs-changed`, `day-unreadable` | the newest day folder delivered, started with other values, or has no readable state.json |
| `workspace-busy` | another `orbitwin run` holds the workspace (status says `running`) |
| `commit-unset` | a new day folder with no `expect_commit` and no `--allow-non-main` |

Exit codes of `run` are those of `orbitwin day`: 0 the upload is in place, 2 a
stop or a refusal, 3 paused for a person.
