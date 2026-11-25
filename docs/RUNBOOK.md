# Competition-Day Runbook

## One command

A workspace (docs/WORKSPACE.md) gives the same run from files: `orbitwin init`
writes it, `announce.toml` takes the announced values, and `orbitwin run
<workspace>` runs `orbitwin day` below with its day folders in the workspace.

`orbitwin day` runs steps 1 to 7 below, in order, with their own commands, on
this workstation, and stops at the first gate that does not pass. In the
activated environment, from the checkout (origin/main, no local change):

```text
orbitwin day <the official script> --horizon-s <T_max> [--tmax-as-announced] \
  --team-number <your team's number> [--mission-boundary-km <km>] [--closest-approach-window after_alignment|from_origin] \
  [--companion <each file that came with the script, relative to its folder>] \
  --kt-per-s <kt> --ct-s <Ct> --kv-per-m-s <kv> --cv-m-s <Cv> --cr-km <C_r> \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" \
  --expect-commit <the commit written down the day before (step 0)>
```

Inputs: the official script as the organizer gave it; T_max as announced
(`--tmax-as-announced` when that is the announced value, step 2); the boundary
radius if announced; the five coefficients, each in the unit its flag names
(a negative kt or kv, or one not finite, is refused); the GMAT pair, both
flags or both from the environment, never half of each; and the commit chosen
the day before (step 0). HEAD must be that commit with no local change, or the
unpacked archive's `BUILD_INFO.json` must name it and every file under `src/`,
`scripts/` and `config/` must be the one its `SHA256SUMS.txt` lists, with none
added; nothing is fetched, so a merge to origin/main on the day changes nothing
here. When the craft start
on different epochs it stops at step 2 and names `--closest-approach-window`:
a person chooses it (step 2). Left out, the coefficients pause the day after
step 4a (exit 3), naming the 4a candidate that would be the upload without
them; once announced, give them to `orbitwin day --resume <day folder>`.

Two people can instead each type the announcement into an entry of their own
(docs/ANNOUNCEMENT.md) and give both as `--announcement <entry> --announcement
<entry>` in place of `--horizon-s`, `--mission-boundary-km` and the five
coefficients. The day refuses entries that differ before it makes a day folder,
copies both into `input/` (read-only, hashed and checked again on every
resume) and hands them to steps 2 and 5. A score announced later comes as two
more entries on the resume, which may give the score alone; the flags are then
refused. The day holds every entry to the official script before it records it.

Two people read each readback before the day goes on, as steps 2, 3 and 5
say. The day prints step 2's readback of `day.yaml` and pauses (exit 3); once
both have checked every value against the announcement, `orbitwin day --resume
<day folder> --confirm-readback 2` goes on. Step 3 pauses the same way on
inspect's `mission clock:` line (the craft that starts first) and T_max check,
for `--confirm-readback 3`, before anything searches; step 5 on the readback of
`day-scored.yaml` (T_max, kt, Ct, kv, Cv, C_r, the score table), for
`--confirm-readback 5`. A `WARNING:` or `NOTE:` line in a readback, and a
`WARNING:` from inspect, stops the day (exit 2) until `--accept-config-warnings`
is given with that `--confirm-readback`; the lines `--tmax-as-announced`
makes for the T_max and boundary given are not counted, at step 2, 3 or 5. Each
pause also says what a wrong value costs: a new day folder with the corrected
input, and the steps done so far again. rank's `warnings` are printed with its
`ceilings` and `levers`; those that can mean a wrong input (a term flat over the
archive, a coefficient of zero, another score block, another script, no GMAT
run) stop the day until `--accept-rank-warnings`, and those every archive gets
(the winner's 10% sensitivity, the stage a file is read at, the choice of the
best submittable row, solve's own GMAT failures) are shown only. Each of these
flags accepts only what an earlier stop printed, and `state.json` and the note
beside the upload record it.

What it does: copies the script into a new `run/day/<UTC>/input/` (read-only,
hashed; step 1), writes `run/day.yaml` (2), inspects and baselines (3),
searches (4), certifies `final-01` and, when named, `final-02` (4a), writes
`day-scored.yaml`, searches again and ranks (5), certifies `best_submittable`
into `final` (6; copy mode when the plots need it), and, when `final` burns in
the head start (`policy.head_start_burns`), rank's best without head-start
burns into `final-03`, runs step 7 on every
certified candidate with `day-scored.yaml` and its certificate, and runs
`audit_burn_table.py` (step 7) on the one chosen. Each step's output is in
`<day folder>/logs/`; on the screen each line a step prints is marked with it
(`[certify-final] ...`), so certify's own "the upload is" lines name its
submission folder, not the upload: only the day's last lines do.

Output: `run/upload/<UTC>/` holds the upload, the submission script and
the Team54 report of the highest-scoring candidate whose certify ended on
`PRE-CHECK PASS`, whose step 7 exited 0 and whose audit passed (one that
burns in the head start only if its step-7 score is
`policy.head_start_min_gain_points` above the best that does not), beside
`SHA256SUMS` and `README_NOT_FOR_UPLOAD.md` (the candidate, score, T_team, ΔV,
d_min, PRE-CHECK, step 7, the audit, the GMAT console and startup file, the
confirmations, and the commit and provenance of each run that made the files).
The copy is made in `<UTC>.partial/`, step 7 runs again on it, and only then is
it renamed `<UTC>/` and `run/upload/latest` moved to it (a symlink renamed
into place). The last line names the two files to upload, exactly. Exit 0
means all of it happened.

A candidate whose report cannot show a burn (no EarthMJ2000Eq velocity of the
pursuer in the official ReportFile) ends certify on `PRE-CHECK PASS (end time
only)`. The day runs step 7 on it and stops, printing side by side what 6b.7
compares: the report's full path and the column to read (the ElapsedSecs of the
craft that starts first), the certificate's T_team, and step 7's reading of the
last row and its score (a step 4a candidate's score is step 7's alone, since it
was certified before the coefficients). Once they agree, `orbitwin day --resume
<day folder> --by-eye-passed final` (or `final-01`, `final-02`, `final-03`) lets it compete
as a pass, after a full PASS on an equal score: step 6's binding runs again and
must still exit 3 on every run, and both are recorded. A certify that stops on
the plots with no console copy names step 6b and the `staging/final.script` in
the folder certify said it kept.

A candidate whose official report carries no SC_Alien position (step 3) ends
certify on `PRE-CHECK PASS (target not in report)`; the day runs step 7 on it,
prints "step 7 (<candidate>): PRE-CHECK PASS (target not in report): d_min and
the interception were proven on certify's diagnostic flight ...", and lets it
compete as a pass. Without the pursuer's velocity too it is the end-time-only
case above, `PRE-CHECK PASS (target not in report; end time only)`, except that
the binding run again must exit 7, not 3. The step 7 run certify's pre-check
recorded is step 7's own and is not run again as a binding.

A stop prints every resume command with `--allow-non-main` when the day folder
was started or resumed with it. A stop or crash between the copy being proven
and `latest` moving says so, and the next resume moves `latest` to that proven
copy instead of making another.

Any other outcome stops before `latest` moves: a step that does not exit 0, a
PRE-CHECK other than PASS, a step 7 exit other than 0 (3 to 6 included), a
failed audit, a disk under 20 GiB free or `output/DISK_LOW` (the workstation guard's; in an
unpacked release only the free space is checked unless `ORBITWIN_DISK_LOW_FILE`
names the guard's file), a final root that
is not a folder, a work or final root under `output/` (the disk guard's), a
checkout that is not the expected commit or has local changes
(`--allow-non-main` runs it anyway, recorded), and a `latest` that another day run delivered after this one began
(`--force-latest` moves it anyway). It prints the step below to follow, what a
copy cut short left (`<UTC>.partial` or `<UTC>.not-proven`, never an upload),
and where `latest` points now, with that upload's T_max, official script hash
and time. Each candidate's verdict is a person's to read: to leave one out,
`orbitwin day --resume <day folder> --exclude final-02` (or `final`,
`final-01`, `final-03`); `--include` takes that back and runs a refused certify again,
which a resume otherwise does not. A refusal (certify exit 2) prints both
forms and says that `--exclude` applies to it, and `--include` only when its
log shows GMAT did not finish a run. A certify that got no answer is no
verdict: GmatConsole died of a signal, timed out or wrote no report, without
an error GMAT wrote, or this machine failed under it (a full disk, a
permission, no memory), and certify exits 75 instead of 2 and keeps that run
as `<output>.no-answer`, not `.refused`. A failure GMAT gave a reason for, or
a file named wrong, stays exit 2. The day
runs it again once at once; if that ends the same way, it stops with "That is
no verdict on final" and the plain `orbitwin day --resume <day folder>`, which
runs it again. Do not `--exclude` it on the first such stop: on R4's R1 the
`--exclude final` a crash stop then printed delivered `final-01`, 0.513 points
down, and a delivered day folder takes no `--include` after. A certify killed
or cut short (any exit but 0, 2 and 75) is no verdict either, and a resume
runs it again. A day folder that has delivered says where
and changes nothing on a resume.

A failure no gate names, such as a full disk under the day folder or a bug,
ends with `FAILED while <the step>`, the error, and the resume to type once
it is dealt with; it is never a bare traceback (a bug's traceback is printed
above those lines). The day keeps 1 MiB beside `state.json`
(`.state.reserve`) and gives it up when the disk fills, so the record of the
step that was running is still written and a resume does not repeat what had
finished.

The command a readback pause, an interrupt, a failure, a disk stop or a
candidate's stop asks for is printed alone on its own line, with a blank line
on each side, to be copied whole.

A resume reruns only what did not end 0
(a search that verified nothing in GMAT, and a rank of an archive searched
under other coefficients, count as not ended 0), after checking every file the
earlier steps wrote against the SHA-256 recorded in `state.json`; a rerun without `--resume` starts a new day folder, and a new
upload folder never overwrites an older one.

The steps below remain the reference and the fallback: the command runs them,
and when it stops, the step it names is done by hand from there.

### If the screen, the ssh session or the machine goes

Run the day inside tmux: `tmux new -s day`, then the command. A dropped ssh
session or a closed laptop lid does not stop it; `tmux attach -t day` brings
the screen back, still running.

Write the day folder down when the first line prints it (`day folder
.../run/day/<UTC>`). If the screen is lost, it is the newest folder under
`run/day/` of the release: `ls -d run/day/2* | tail -1`.

If the day itself was killed (a closed terminal without tmux, kill, the OOM
killer), the step it was running goes on without it: each step runs in a
session of its own, so a search or a certify does not stop with the day.
Wait for it before resuming. A resume refuses while one is still running,
names its step and its process ids, and gives the command that stops them
all (`kill -TERM -- -<its process group>`); wait for it to end or run that,
then type the resume it prints. If the refusal names no step ("another
orbitwin day is running ... or something an earlier run started still holds
its lock"), first make sure no `orbitwin day` runs on that folder in another
terminal; then `ps -eo pid,args | rg <the day folder's UTC stamp>` lists what
is left: stop every process it lists (`kill <their pids>`) and resume.
`orbitwin day --resume <day folder>` skips every finished step.

When a step itself is killed while the day runs (the OOM killer taking a
solve, say), the day kills what that step left in its process group, its
workers among them, reports the step's exit and prints the resume.

After a reboot, open a new terminal (in tmux), `cd` to the release, and
start it as step 0 does (the exports are for the steps done by hand):

```sh
cd <the unpacked tasa_orbitwin> && . .venv/bin/activate
export GMAT_CONSOLE=<laptop GMAT>/bin/GmatConsole GMAT_STARTUP=<laptop GMAT>/bin/gmat_startup_file.txt
orbitwin day --resume <day folder>
```

A resume flies the GMAT pair the day folder recorded, whatever GMAT_CONSOLE
and GMAT_STARTUP say, and refuses to go on if either of those two files has
changed since (its SHA-256).

## Who runs what

| Machine | Runs |
|---|---|
| Our workstation: offline, x86-64 Linux, this repository, GMAT R2026a for Linux | steps 0 to 7, and the files to upload |
| The organizer's computer: Windows, GMAT GUI | the alternative only: our script in their GUI (step 6b) |

The upload is certify's submission from our workstation, the script and the
Team54 report in `run/final/submission/`, as in the preliminary (thc1006,
2026-11-06). A template with plots (OpenFrames, GroundTrack), the expected
case, gets no report from GmatConsole as written; certify then runs its
proven copy less the plots (`gmat.console_copy`, on in the ranking profiles)
and packages that copy's report (step 6c). Running our script in the
organizer's GMAT GUI (step 6b) is the alternative, when no copy can run.
Nothing from this repository runs on their computer.

On USB:

- organizer to laptop, at the start: the official script and every file that came with it (step 1);
- on the GUI path only (step 6b): laptop to organizer, our script, one file in a USB folder of its own per candidate; organizer to laptop, in the same folder, the report their GUI wrote, the copy of our script it ran (`<folder>-gui.script`) and their hashes, for step 7, and from there to the upload.

## Time budget

Measured by an end-to-end rehearsal on 2026-11-04 on the 14-core workstation
(gmat.workers 8, other jobs sharing it), on R1 and R6. A laptop with fewer
cores is slower: the day before, run steps 1 to 7 on
`examples/ranking/R1_NearPlane_Reachable.script` on it and write its times
next to these.

| Step | Time |
|---|---|
| 0 install (`bootstrap_offline.sh`) | 11-17 s |
| 0 fast test suite | 62 min serial for 4472 tests on a loaded machine (R4, 2026-11-07; 33:17 for 3711 on 2026-11-05); the day before, not on the day |
| 1, 2 | under 1 s each, plus the second person's check |
| 3 inspect / baseline | 1 s (9 s when the craft start apart) / 6-8 s |
| 4 first search | 4.3-4.6 min |
| 4a safety-net certify | 107 s for one certify on R5 (2026-11-06), 298 s on a loaded machine; two certifies when search-01 has no hit, and step 4 flies up to `gmat.verify_misses` (4) more misses |
| 5 config | under 1 s, plus the two-person check |
| 5 second search / rank | 3.9-4.8 min / 2 s |
| 6 certify, `--runs 3` | 25-33 s; on a template with plots it stops in 10 s, as expected |
| 6b the organizer's GUI | not measured |
| 7 check / audit | 1 s / about 26 s (R1, measured apart from the rehearsal) |
| `orbitwin day`, steps 1 to 7 | R1 from the release archive (R4, 2026-11-07): 8.4 min on an idle machine, 17.4 min on a loaded one; 6 min 8 s on R1 and on R1 with OpenFrames from the checkout earlier the same day (console copy on the second). The steps above, plus 1 s per step 7 check |
| the same, pursuer first (head start open, #202) | R8 from the checkout, 2026-11-08: 11.0 min before head-start burns; 17.7 min when GMAT also corrected 40 head-start plans; 18.4 min (step 4 349 s, step 5 652 s) with none flown, on a machine at load 22. The head-start search adds 10-30 s; GMAT flies head-start plans only when one leads by half `policy.head_start_min_gain_points` (R8: none). R9 (3600 s head start), where they do: 9.2 min (step 4 185 s, step 5 238 s with 15 head-start corrections, then certify `final` and `final-03`) |

## 0. Install on our laptop, the day before

Do this at home, the day before. On the day the laptop is offline;
`scripts/bootstrap_linux.sh` installs from PyPI and cannot be used then.

```bash
sha256sum --check --quiet SHA256SUMS.txt &&   # every file is the one the archive was built with
  python3 -I scripts/check_release_tree.py &&   # and there is no other
  bash scripts/bootstrap_offline.sh && . .venv/bin/activate &&
  orbitwin --version | grep -F 'commit XXXXXXXXXXXX' &&   # the commit written down when the archive was built
  PYTHONPATH=src python -m pytest -m 'not slow and not gmat'
```

Unzip the archive into a new, empty directory, never over an older tree: the
older tree's files stay beside the new ones, and `sha256sum` checks only the
files it lists. `check_release_tree.py` refuses a file the archive's
`SHA256SUMS.txt` does not list, outside `.venv/`, `run/`, `output/`, `dist/`,
`build/`, `input/original/`, `GmatLog.txt`, `src/*.egg-info`, caches, coverage
data and bytecode.

The block is one command, run in the unpacked archive: the first failure stops it.
First replace `XXXXXXXXXXXX` with the first 12 characters of the commit written
down when the archive was built; `make package` prints it as `commit=` and puts
the same 12 in the archive's name, `tasa_orbitwin-<version>-<commit>.zip`.
`orbitwin --version` prints the commit the archive's `BUILD_INFO.json` names.
`make package` refuses a HEAD that is neither origin/main, fetched first, nor
the release tag `v<version>`; one built with `--allow-non-main` says `NOT
origin/main` there and must not be used on the day.
If `grep` prints nothing, the placeholder is still there or this is another
build than the one tested: stop.

That commit is the day's: `orbitwin day` takes it as `--expect-commit` and
refuses any other, from the checkout or the archive, and fetches nothing. On
the workstation, write down `git rev-parse HEAD` of the origin/main checkout
tested the day before; a merge on the day does not move it.

The bootstrap refuses a wheel bundle without `vendor/wheels/SHA256SUMS.txt`,
with any file besides the wheels it lists, `constraints.txt` and `README.md`,
or without a wheel it lists, and pip installs
with `--require-hashes` from the hashes it lists. `orbitwin: command not found`
means the environment is not active; the `&&` keeps the suite from running on
whatever Python the machine has.

The suite runs serially: 4472 passed and 14 skipped in 62 minutes on the
workstation with other jobs on it (R4, 2026-11-07, from the archive of commit
27f7278; 3711 in 33:17 on 2026-11-05). Start it early. The offline bundle has no pytest-xdist, so there
is no parallel run on the laptop. Run it the day before; on the day,
`orbitwin --version | grep -F 'commit XXXXXXXXXXXX'`, with the commit filled
in, is enough if nothing on the laptop changed since.

Steps 2 to 7 run `python`, `orbitwin` or `scripts/competition_day.sh` (which
runs `python -m orbitwin`) and need that environment; steps 2 to 7 also need
the GMAT paths: step 2 reads a template whose craft start apart or are
inserted EarthFixed only through GMAT, and step 7's audit reads them.
`ranking_day_config.py` takes `GMAT_CONSOLE` and `GMAT_STARTUP` from the
environment when it is not given the pair, so whatever exports them (this
terminal, or a `BASH_ENV` file) decides which GMAT step 2 and step 5 read the
script with; the readback's `script` line names it. They are the laptop's R2026a
GmatConsole and its stock startup file, both in its `bin/`. Write the exact
paths here before the day:

- `GMAT_CONSOLE` = `______________________________/bin/GmatConsole`
- `GMAT_STARTUP` = `______________________________/bin/gmat_startup_file.txt`

Start every terminal used for them with:

```bash
cd <the unpacked tasa_orbitwin> && . .venv/bin/activate
export GMAT_CONSOLE=<laptop GMAT>/bin/GmatConsole GMAT_STARTUP=<laptop GMAT>/bin/gmat_startup_file.txt
```

Without the activation `python` is whatever the machine has. On the
development workstation (Ubuntu 26.04, apt's NumPy and PyYAML but no SciPy)
step 2 stopped with `ModuleNotFoundError: No module named 'scipy'` and step 3
with `orbitwin: command not found`. On a machine that has SciPy and PyYAML,
`orbitwin` and `ranking_day_config.py` refuse NumPy, SciPy or PyYAML other
than the versions `requirements.txt` pins (`ORBITWIN_ALLOW_UNPINNED=1`
overrides that), and so do `audit_burn_table.py`, `safe_delta_v_floor.py`,
`hunt_ranking_score.py` and `hunt_multi_burn.py`. `check_report_rows.py`
refuses them with or without SciPy, which it imports only for `--script`.
The line is `.`, not `source`, because a plain `sh` (dash) has no `source`
and goes on without the venv.

`orbitwin` (also with no `--config`), those four scripts, `check_report_rows.py
--config` and `competition_day.sh` exit 2 on a config not under the ranking
rules unless `--rules preliminary` (`--preliminary` for `competition_day.sh`)
says so. On the day that refusal means the wrong file was named: name step 2's.

The bundle must be fetched beforehand, at home, with
`bash scripts/fetch_offline_wheels.sh`, and carried with the repository. It
covers Python 3.11 to 3.14 on x86-64 Linux (an x86-64-v2 CPU, glibc 2.27 or
newer: `ldd --version`), so the laptop's Python need not be one version.
Before it touches `.venv`, `bootstrap_offline.sh` has the
bundle's own pip wheel resolve the install as a dry run, and refuses an
interpreter the bundle cannot serve: another version, a free-threaded build
(`python3.14t`) or another machine. The error names the wheel tag the
interpreter wants. A dry run that fails for another reason, such as the
shell's own settings, only warns. A bundle with no pip wheel gets the version
check only. An interpreter without `ensurepip` (on Debian and Ubuntu that is
the `python3.X-venv` package, which needs the network) gets a venv without pip
and pip from the bundle's own wheel. If `python3` is outside 3.11 to 3.14 but
such an interpreter is installed, name it:
`PYTHON=python3.11 bash scripts/bootstrap_offline.sh`. Run again with the same
Python, it keeps `.venv` and its packages but reinstalls orbitwin itself (4 s
here), and for a few milliseconds of that `.venv/bin/orbitwin` and its path
file are gone. Rerun it while no other terminal is starting a step.

The laptop must be x86-64 Linux with Python 3.11 to 3.14; check that when
packing. Any other target needs the bundle re-fetched with `WHEEL_PLATFORMS` /
`WHEEL_PYTHON_VERSIONS` set for it, on a machine that has internet. There is
no recovery from this on the day. The organizer's computer (Windows) needs
nothing from here: it only runs GMAT.

`make package` puts the bundle in the release archive, and `BUILD_INFO.json`,
which names the commit, its time and `"dirty": false` (a tree with uncommitted
changes is refused). It refuses to build
when a wheel that `vendor/wheels/SHA256SUMS.txt` lists is missing, a link, or
does not match its hash, or when that manifest is missing, empty or malformed.
It then runs the fast suite from the extracted archive in a session of its
own, so Ctrl-Z pauses the build and not the suite: the suite keeps the CPU
busy, and `fg` finds it finished or still running. To stop both, `fg` and then
Ctrl-C, or `kill %1`. Not `kill -9`: the suite runs on alone, and its
extracted copy (about 300 MB) stays in `$TMPDIR/orbitwin-release-check-*`
until someone removes it.

## 1. Preserve the input

```bash
mkdir -p input/original
cp OFFICIAL.script input/original/official.script
chmod a-w input/original/official.script
sha256sum input/original/official.script
```

If the script came with other files, such as a `.tide`, a `.spo` or a gravity
file, copy them into `input/original/` too, under the names and subfolders the
script gives them. GMAT reads a gravity, tide or SPAD file named by a relative
name from the script's folder first, and every GMAT run here copies those from
that folder beside its own copy of the script. A Windows path such as
`'C:\data\Earth.tide'` is a relative name to GMAT on Linux: put the file at
`input/original/C:/data/Earth.tide` (measured on R2026a). A name with `..`
cannot be copied that way; GMAT then reads what it finds from GmatConsole's
folder, as it does for `'../data/vehicle/spad/SphericalModel.spo'`. R2026a
never looked beside the script for a space-weather or EOP file; one of those
belongs in GMAT's data folders. `inspect` refuses a force-model or SPAD file
GMAT would find nowhere, and names it.

## 2. Write the day's config

The ranking rules (2026/10/06, revised 2026/11/04) announce T_max at the start
(§3) and C_r, kt, Ct, kv, Cv during the contest (§4). Write them into a new
config; never edit
`config/ranking.yaml` or type YAML by hand:

```bash
python scripts/ranking_day_config.py --output run/day.yaml --horizon-s <T_max> \
  --team-number <your team's number> \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP"
```

It reads step 1's `input/original/official.script` (`--script` names another)
and checks T_max against SC_Alien's two-body coast in it, as `inspect` does: a
T_max before SC_Alien's perigee, or one that leaves it off an announced
`--mission-boundary-km`, is refused and nothing is written. The GMAT pair lets
it read a template whose craft start apart or are inserted EarthFixed; without
the flags it takes `GMAT_CONSOLE` and `GMAT_STARTUP` as step 0 exports them, and
with neither such a template is refused. The readback's `script` line says
which GMAT read it. Step 5 passes the same pair.

Run by hand, the step can take the announcement entered twice, by two people
apart, in place of `--horizon-s`, `--mission-boundary-km` and the score flags:
`--announcement <first> --announcement <second>` (docs/ANNOUNCEMENT.md). It
refuses two entries that differ and a script that contradicts them.
`orbitwin day` takes them the same way ("One command" above).

T_team, T_max and Ct count from the earlier of the two craft's epochs: the
organizer answered Q1 of docs/ASYNC_START_AND_INSERTION.md in writing on
2026-11-04 18:17, 「從比較早開始的開始算」. There is no time-origin flag to pass.
The tool writes `mission_time_origin: earliest`, the ranking rules refuse any
other value, and step 3 prints which craft that is from the epochs GMAT reads;
the operator checks that craft against the announcement there. Enter T_max and
Ct as announced, counted from that epoch. If the craft start on different
epochs, with or without an alignment as the template's first mission command,
add `--closest-approach-window after_alignment|from_origin` (its trade-off:
docs/ASYNC_START_AND_INSERTION.md, "Step 6 as built", "The window"): which rows
d_min counts is still unanswered, so a person decides and writes it down.
`inspect` refuses Δ > 0 without it, naming the YAML key
`policy.closest_approach_window`; do not type that, write a new config with
the flag. If the organizer announces the mission boundary radius, add
`--mission-boundary-km`; this step then refuses a T_max where SC_Alien is not
at it, and `inspect` warns. The tolerance is about 13 km for a T_max given to
the second, while SC_Alien climbs 5.6 to 8 km/s, so a T_max announced rounded
to 10 s or to the minute is refused too. Read the announcement again; if this
is the announced value, rerun with `--tmax-as-announced`, which takes it with
a `WARNING:` giving the offset in km and s, writes `policy.tmax_as_announced_s`,
and is read back on its own line; step 5 then needs no flag. A T_max before
SC_Alien's perigee stays refused with it. A template whose
craft start apart and that carries no alignment gets one added: the ranking
profiles set `policy.add_epoch_alignment` (signed off 2026-11-03 in
docs/COMPETITION_POLICY.md), which writes
`Propagate <P>(<early>) {<early>.A1ModJulian = <late>.A1ModJulian};` as the
first mission command. The uploaded report then starts with that line's rows,
at least Δ / MaxStep of them, before the later craft moves. Baseline (step 3)
runs the official script with that line added too.

If the organizer says that no burn may come before the later craft's epoch, add
`--no-head-start-burns` (`orbitwin day --no-head-start-burns`): it turns
`policy.head_start_burns` off in this config and the scored one, and the readback
and `README_NOT_FOR_UPLOAD.md` say so. Never edit `config/ranking.yaml` for it.

A second team member checks the printed values against the announcement. The
craft are the rules' `SC_Human` (ours) and `SC_Alien` (§1). If the template
names them differently, `inspect` refuses it: write the config again with
`--pursuer-name` and `--target-name`, and pass the same names to
`check_report_rows.py` as `--pursuer` and `--target` in step 7.

The readback also prints `gmat.workers`, how many GmatConsole processes solve
runs at once, beside the cores and threads this process may use. Each run
keeps one core busy. The profile's 8 comes from our 14-core workstation: on
R1, 12 and 16 were no faster, and on R6, 12 was no faster. It warns when the
count is more than the cores (or the threads, when it cannot read the cores).
That is a ceiling, not the best count: on 14 cores, 12 was no faster than 8.
If it warns, write `run/day-2.yaml` with `--gmat-workers` at most the number
it gives, use that file in steps 3 and 4, and pass the same count in step 5. A
container's CPU quota is not seen; count it yourself.

## 3. Inspect

```bash
export GMAT_CONSOLE=<laptop GMAT>/bin/GmatConsole GMAT_STARTUP=<laptop GMAT>/bin/gmat_startup_file.txt
orbitwin inspect input/original/official.script --config run/day.yaml \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" \
  --output run/scenario.json
```

Stop immediately if roles, epochs, coordinates, burn slots, force model or reports differ from expectations.

When the official report (ReportFile1) carries no SC_Alien position, as the
organizer's workshop template writes it (`ReportFile1.Add = {SC_Human.ElapsedSecs,
SC_Human.EarthMJ2000Eq.X, ...Y, ...Z}`), inspect says so on stderr ("ReportFile1,
the official report, carries no SC_Alien position ..."). That is not a stop: no
distance can be read from such a report, by us or by hand, so certify proves the
interception and d_min on its diagnostic flight of the same script (both craft,
Precision 17), binds the report by its SHA-256 and its last row's time to the
certified T_team, and ends on `PRE-CHECK PASS (target not in report)` (step 6);
step 7 passes the report with a `WARNING:` saying so. A report from a GMAT GUI
cannot be judged at all (step 7, exit 7), so with such a template the upload is
certify's. How the organizer's validator computes d_min from data its report
does not hold is an open question: ask it, in writing if possible, before the
upload.

inspect prints on stderr a `mission clock:` line naming the craft `earliest`
resolves to, the head start Δ, and T_max on the solver clock, which starts at
the later craft's epoch: T_max less Δ when the craft start apart. Check that
craft against the announcement: it is the one that starts first. If they
differ, stop and ask. When SC_Alien's orbit is clearly eccentric (osculating
e above 0.05), every command refuses a T_max before its perigee (§3 puts T_max
after it; 111.86, T_max typed in minutes, is one). For a near-circular SC_Alien
§3's premise does not hold and J2 alone moves its osculating perigee anywhere in a
period, so T_max is unchecked: every command warns, and T_max must be confirmed
with the organizer. With `--mission-boundary-km` they warn when the two-body coast puts
SC_Alien off that radius at T_max (another scenario's T_max, or a typo); the
`T_max check` line shows the figures. Read a warning as a T_max to check against
the announcement before searching. If a refused T_max was read correctly from the
announcement, the announcement and §3 conflict: STOP and ask the organizer
whether that T_max is right (COMPETITION_POLICY.md, Order of authority). Only after they confirm it in
writing, write the config again with
`--accept-announced-horizon-before-perigee-s <that T_max>`; it covers that value
only, and inspect, solve, rank and the certificate record the override and the
perigee it contradicts.

When the craft start apart, inspect also prints `time origin` lines on stderr
and writes them under `time_origin_evidence`: where the template's Propagates
end counted from each craft's epoch, where GMAT's SC_Alien is at T_max under
each reading, when it reaches the boundary radius if one is set, and a last
line, `time origin verdict:`, naming the reading the evidence supports or
`undetermined`, with the numbers. It flies SC_Alien once more in GMAT for this.
Nothing is switched, and the organizer's answer outranks this evidence. A
verdict naming the later craft's epoch says T_max was most likely entered from
that epoch: check it against the announcement, and if it was announced that
way, stop and ask. The
template counts only when its goal ends short of T_max by the head start, which
an author writes only on purpose; a goal of T_max itself, or one in a template
with no alignment of its own, is reported and not counted. The boundary line
reads SC_Alien. solve repeats the lines, adds the verdict to its warnings unless
it names the earlier craft's reading, and warns when SC_Alien's track could not
be read.

```bash
orbitwin baseline input/original/official.script --config run/day.yaml \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" --output run/baseline
```

Baseline runs the script exactly as given; its report is the zero-burn
answer (7 s on R2). It is never an upload: the script is the official one, so
its report is still named `ODC_2026_TeamXX.txt`. If the script creates
OpenFrames objects, GmatConsole
refuses it and baseline fails; on the workshop template a GroundTrack also left
it writing no report. The organizer's GUI-saved templates carry both. `solve`
still works on its private copy, which leaves them out; the zero-burn report
then needs the GMAT GUI, running `input/original/official.script` itself.
When baseline made the copy below, the GUI runs that copy instead, from a
scratch folder (`run/baseline-gui/`) that holds a copy of everything in
`input/original/` beside it: GMAT looks for a relative tide, gravity or SPAD
file in the script's folder first (step 1, step 6b.2), and never in
`input/original/` itself, which stays as copied. Step 6 says what certify does
with such a script.

With `gmat.console_copy` on (the day config; see step 6c), baseline does not
fail on such a script: it writes `run/baseline/console_copy/`, the script less
its OpenFrames and GroundTrack lines, proven to differ by nothing else, and
runs that in `run/baseline/console-run/`. It says so on stderr ("orbitwin
baseline: GmatConsole refused ... for its plots alone, so this ran ..."), and
`baseline.json` names the copy under `console_copy`. That report is the zero
burn's, from this workstation; it is still not an upload.

The one exception to "exactly as given" is a template whose craft start apart
with no alignment (step 2). GMAT aborts its joint Propagate (`P(A, B)`) as
written ("Spacecraft epochs do not match"), and runs a split one (`P(A) P(B)`)
with both ElapsedSecs equal while the epochs stay apart, which is not one
instant. So baseline writes `run/baseline/official_with_alignment.script`, the
official script with only the line solve and certify add, and runs that. It
says so on stderr (`orbitwin baseline: baseline ran <full path>, the official
script with the epoch alignment ...`), naming which of the two the template
does, and `baseline.json` records the line as `epoch_alignment_added`, that
note as `epoch_alignment_note` and the copy as `ran_script`. That is expected,
not a stop. On R7 (2026-11-05) it took 7 s, and its report's closest
row, 4711.157 km, is the zero burn solve prints in step 4.

If baseline, or any later step, refuses the official script for GMAT's own
errors or for how GMAT would read it (the parser), stop and take it to the
organizer (Stop rules). Appendix A says what each message means.

## 4. First search

```bash
bash scripts/competition_day.sh input/original/official.script run run/day.yaml
```

It runs inspect and solve into `run/search-01`, and points `run/latest` at
it. It refuses to run unless GMAT_CONSOLE and GMAT_STARTUP are both set;
`--analysis-only` searches without them into `run/analysis-NN`, which nothing
below reads, except on a template only GMAT can read (craft on different epochs,
an EarthFixed insertion). Without coefficients, solve puts GMAT-verified candidates first and
hits before misses, then sorts by delta-v, then closest approach. Measured
on this workstation (14 cores, R2026a) with gmat.workers 4, on an earlier
build and without coefficients: 5 min on R1, 32 s on R2. On 2026-11-02, with
coefficients and other jobs on the machine, solve on R1 took 215-230 s at 4
and 171-174 s at 8. Every later run gets its own `search-NN` and none is
overwritten.
`run/latest` is removed before each solve and set after it, so a run that
fails leaves no `run/latest` and the commands below stop rather than read an
older search.

Under the ranking rules solve flies the zero burn in GMAT and ends by printing
on stderr `orbitwin solve: the zero burn passes <d> km off (GMAT); ...`, the
number step 5 takes as `--zero-burn-dmin-km`. It is also
`zero_burn_closest_approach` in `run/latest/summary.json`, with `measured_by`
saying whether GMAT flew it or it is still the two-body estimate.

A GmatConsole run that crashes, times out (`gmat.timeout_s`) or leaves a report
that stops short of the script's end loses that plan, and solve still exits 0
with the best of the rest. GMAT's own verdicts on a flight (under the altitude
floor, a broken constraint, a stop outside the sphere) are answers, not lost
plans, and are not listed. It then prints
`orbitwin solve: WARNING: <n> GMAT correction(s) or verification(s) raised ...`,
also in `warnings` and as `gmat_failures` in `summary.json`. Each one is listed
with the score it was chosen by, and `ABOVE THE BEST` marks one that scored above
the chosen best before GMAT. For one marked so, run the search again, or certify
it directly with `--candidate <its id>`: certify flies it afresh and, as for any
row, refuses one GMAT flies worse than the archive says unless
`--accept-gmat-loss`. Step 5's rank prints the same `WARNING:` for a
verification that raised on a row that scored above `best_submittable`.

## 4a. Safety net: certify search-01 now

As soon as step 4 finishes, before step 5, certify search-01's best into its
own folder, then take it through steps 6b and 7 like `run/final`. A checked
report is then ready before the coefficients are announced. That means using
the organizer's computer more than once (here and again after step 6): confirm
on the day that it is allowed. If it is not, stop 4a after certify and carry
only the one candidate you finally choose.

```bash
python -c 'import json; s = json.load(open("run/search-01/summary.json")); print(s["best_candidate_is_gmat_verified"], s["best_certifiable_without_shallow_opt_in"], s["closest_certifiable_miss"], s["best_certifiable_uses_head_start"], s["best_certifiable_without_head_start"])'
orbitwin certify input/original/official.script --config run/day.yaml \
  --candidates run/search-01/candidates.json --candidate <the first id it printed> \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" \
  --runs 3 --output run/final-01
```

The fourth value says whether the first id burns in the head start
(`policy.head_start_burns`), the fifth is the best without. Before the
coefficients solve flies no head-start plan in GMAT (nothing can price
`policy.head_start_min_gain_points` yet), so at step 4 the fourth is `False` and
the first id is the one to certify. Were it `True`, certify the fifth instead.

The third value is the verified miss that passes closest, nearer than the
zero burn on the distance the term prices (with `closest_approach_window:
from_origin` the alignment's minimum caps every miss's, so a miss that only
ties the zero burn there is not named). Solve also prints it: `orbitwin
solve: the closest verified miss is ...`. Without coefficients solve ranks
misses by delta-v first, so when search-01 has no hit its best is the zero
burn, which spends nothing and passes widest. The closest miss is the other
end of that trade, and only the announced coefficients can choose between
them. When it is not `None` and not the first id, certify it too, the same
way, into `run/final-02`. When the first id is a hit and the closest miss
spends at least its delta-v (`total_dv_km_s` in `run/search-01/candidates.json`),
`run/final-02` may be skipped: the hit scores at least as much on every term,
50 on distance, an earlier completion and no more delta-v. When the zero
burn failed its GMAT verification, the third value is the closest verified
miss at any distance, and the first id may be another miss: certify both.

```bash
orbitwin certify input/original/official.script --config run/day.yaml \
  --candidates run/search-01/candidates.json --candidate <the third id it printed> \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" \
  --runs 3 --output run/final-02
```

Measured on the phase-review archives with kt 0.003/s, Ct 0.8·T_max, kv
0.0012 per m/s and Cv 2000 m/s: on R5 the zero burn scores 23.67 at C_r 300
and the closest miss 53.11 (GMAT certified it at 117.02 km, 2.99 km/s, in
1 min 47 s); at C_r 3000 they score 49.98 and 56.56. On R2 and R7 the
closest miss passes 1993 km off: at C_r 300 it scores 6.59 and 6.10 against
the zero burn's 23.67 and 23.18, at C_r 3000 about 45.8 and 45.3 against
31.54 and 31.05 (two-body distances, not certified). Neither one is always
the better, so certify both. To find the closest misses, step 4's solve
flies up to `gmat.verify_misses` of them (4) in GMAT beside the zero burn.

Certify only if it printed `True` and an id. `False` means search-01 verified
nothing in GMAT, so check `GMAT_CONSOLE` and `GMAT_STARTUP` first. `True None`
means its best is a shallow hit and no other verified row is certifiable
without the opt-in (the third value is then `None` too, since every verified
miss is): go to fallback 4. Name
`run/search-01`, not `run/latest`, which step 5 moves. On a template with
plots certify stops at `exact-run-01`, as in step 6; that is expected only
when its message names OpenFrames or GroundTrack and contains "Run it in the
GMAT GUI" (step 6).

Upload the highest-scoring of these that passed step 7 (step 7 with
`--config run/day-scored.yaml` prints each one's score); before the
coefficients, the first that passed. `run/final-01` and `run/final-02` are
both misses only when search-01 found no hit, and then no order between them
holds without the coefficients: price both with step 7 once they are
announced and upload the higher.

1. `run/final`: step 6, the best under the announced coefficients.
2. `run/final-01`: this step.
3. `run/final-02`: this step's closest miss.
4. search-01's zero burn, certified the same way into `run/final-00`: a miss,
   scored on its closest approach. Skip it when `run/final-01` is the zero
   burn already. Its id:
   `python -c 'import json; print([c["candidate_id"] for c in json.load(open("run/search-01/candidates.json")) if c["branch"] == "zero-burn"])'`

Never the baseline (step 3), an uncertified candidate, or a report step 7 did
not pass. With a template that has plots the baseline does not even run.

## 5. Enter the coefficients and search again

```bash
python scripts/ranking_day_config.py --output run/day-scored.yaml --horizon-s <T_max> \
  --kt-per-s <kt> --ct-s <Ct> --kv <kv> --cv <Cv> --delta-v-unit m/s --cr-km <C_r> \
  --zero-burn-dmin-km <the km step 4's solve printed for the zero burn> \
  --day-config run/day.yaml \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP"
```

It checks T_max against the script again, as step 2 did, and reads it through
GMAT the same way.

`--day-config` names step 2's config (or the `run/day-2.yaml` that replaced it).
The scored config is that file with the score block added, so the window, the
boundary, the craft and `gmat.workers` come through as step 2 wrote them; pass
`--gmat-workers` again only to change the count. T_max is typed again from the
announcement and must equal step 2's, or the tool refuses and writes nothing, so
a slip in either file shows.

When step 2 took the announcement from two entries, this step does too: the
score flags are refused, and `--announcement <first> --announcement <second>`
takes their place, entries that may give the `[score]` alone
(docs/ANNOUNCEMENT.md).

Stop here. The tool exits 0 even when it prints `WARNING:` lines, so a `&&`
after it would search on a config nobody read. Two people read the readback
against the announcement: T_max, kt, Ct, kv, Cv, C_r, the unit, and every
`WARNING:` and `NOTE` line. Only then:

```bash
bash scripts/competition_day.sh input/original/official.script run run/day-scored.yaml &&
orbitwin rank input/original/official.script --config run/day-scored.yaml \
  --candidates run/latest/candidates.json --top 10 --output run/rank-scored.json \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP"
```

A value read wrong goes into a new file, `run/day-scored-2.yaml`, used in steps 5 and 6
from then on; the tool never overwrites one. Run the search above again with it
before ranking: an archive searched under the wrong value was optimised for that
value, and ranking it under the right one only re-scores rows chosen for the wrong
one (R2: C_r 300 re-ranked at 3000 chose the zero burn at 31.54, a search at 3000
found 45.82). Once a verified hit scores above
the readback's second ceiling (no miss scores above it), stop looking for
misses. What the units, terms, ceilings and levers mean: appendix B.

The second search aims at the announced score, and only a search does. `rank`
re-scores a saved `candidates.json` in seconds, but it cannot find a row the
search never looked for: each candidate records the score block it was searched
under (`searched_score`, also in `summary.json`), and when that is not the block
rank scores with, or the search ran without one, rank prints a `WARNING:` on
stderr, adds it to `warnings` and sets `best_submittable_note`. Then search again
under these coefficients and rank that archive; never certify from the old one
alone. Read what it chose:

```bash
python -c 'import json; r = json.load(open("run/rank-scored.json")); w = r["winner"]; c = r["ceilings"] or {}; print("winner", w["candidate_id"], "submittable", w["submittable"], "best_submittable", c.get("best_submittable"), "note", r["best_submittable_note"])'
```

`note None` means the archive was searched under these coefficients, whether or
not any row is submittable.

Certify `best_submittable` (step 6's `--candidate`). The winner may not be
submittable; `best_submittable` is the best row that is. If it prints `None`,
nothing is: read the winner's `blockers` and the `warnings` in
`run/rank-scored.json`. Under the ranking rules a miss is scored on its
closest approach and can win. There `submittable` also means solve flew the row
in GMAT: a row it never verified is blocked as `never verified by GMAT`, and an
archive solved without GMAT has no submittable row, which rank says first in
`warnings` and on stderr.

When the pursuer starts first, `best_submittable` may burn in the head start
(`policy.head_start_burns`); rank takes such a row only when it beats the best
submittable row without one by `policy.head_start_min_gain_points` (0.05), and
prints `orbitwin rank: head start: ...` with the margin. Read it:

```bash
python -c 'import json; h = json.load(open("run/rank-scored.json"))["head_start"]; print(h["chosen"], h["best_with"], h["best_with_score"], h["best_without"], h["best_without_score"], h["gain_points"])'
```

When `chosen` is `head-start`, certify `best_with` into `run/final` as above and
`best_without` into `run/final-03` the same way (`--output run/final-03`), take
both through step 7, and upload the head-start one only if its step-7 score is
still 0.05 or more above the best step-7 score of every certified candidate
without head-start burns, final-01, final-02 and final-03 alike; else upload
that best one. `orbitwin day` does this itself.

## 6. Certify

```bash
orbitwin certify input/original/official.script --config run/day-scored.yaml \
  --candidates run/latest/candidates.json --candidate C-... \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" \
  --runs 3 --output run/final
```

Certify refuses with exit 2 a candidate GMAT flies worse than the archive says:
a planned hit flown as a miss, a flight whose submitted script scores more than
half a point below the archive under the day's coefficients (a later entry, a
wider miss), or, without coefficients, a miss flown more than 0.1 km wider or an entry over 60 s
late. It prints both. Certify a submittable row instead, or run the same command
with `--accept-gmat-loss` to certify the candidate as GMAT flies it. A loss above
0.05 points is printed even when certify goes on: a verified row flies
bit-identically, so that is data or build drift. `certification.json` records
the comparison under `gmat_comparison`.

Any certify refusal after `run/final` exists, the stop on plots below included,
moves it aside to `run/final.refused`, or `.refused-2` and on when that is taken,
so the same `--output` works again. The refusal message names the directory and
the full paths of its `staging/final.script` and `diagnostic/verification.json`:
use those, never a `.refused` folder from an earlier attempt. An interrupted
certify (Ctrl-C) is not moved: give the next one a fresh `--output`.

The upload is certify's submission, from this workstation (thc1006,
2026-11-06): our script and the official report in `run/final/submission/`,
once step 7 exits 0 on that report. Never `official_original.script`,
`certification.json`, the `.diff.txt`, anything from `run/final/diagnostic/` or
internal logs. Only on the GUI path (step 6b) does the upload come from the
organizer's computer instead.

When it does not refuse, certify ends one of two ways. With no plots in the template, two scripts land
in `run/final/submission/`. `official_original.script` is the organizer's
input, kept for the record: never upload it. Ours is the one
`certification.json` names as `submission_script_name`. Certify prints on
stderr, with full paths, the step 7 command for the official report, which must
exit 0, and the upload ("... the upload is these files from this machine, as
they are"): our script and that official report. Another ReportFile's report is
named as neither uploaded nor checked; when the script does not say which report
is official, certify prints why and no step 7 command. With plots, the expected
case, it runs the console copy (step 6c); with `gmat.console_copy` off it stops
at `exact-run-01`: see the paragraph on the GUI below.

Our script is named for the evidence the run established
(`public_gmat_candidate.script` in the rehearsals): do not rename it
(appendix C).

The report's `TeamXX` placeholder becomes our number, from
`policy.team_number` (ours is 54, given to the config tool in step 2 as
`--team-number` and read back as "report name"; no profile sets it). Only the run of X's or N's is replaced,
in any case, after an optional space, `_` or `-`, and never right after a
letter: `TeamXX`, `Team_xx` and `TEAMNN` become `Team54`, `Team_54` and
`TEAM54`; `SteamXX` stays. The official report is `ReportFile1` among the
reports GMAT writes, else the first one GMAT writes; a ReportFile with
`WriteReport = false` writes no file, so it is neither uploaded nor judged.
Under the ranking rules certify refuses with the number unset, refuses unless the official report carries `Team54` once
numbered, refuses a placeholder that cannot take it (`TeamXX_TeamXXL`) or that
only another report has, and refuses another team's number in any report. Any
other report with no placeholder stays as the organizer wrote it, and certify
names it in a warning. Two names that are one file once case is ignored are
refused, as Windows reads them. `inspect`, `solve` and `rank` say the same
before the search.

When the official report has no placeholder at all (GMAT's default
`ReportFile1.txt`, or `ODC_2026_XX.txt`), certify says "the official script
names no team placeholder: stop and ask the organizer how to name the report".
The official script may not be edited. Only if the organizer answers in
writing to keep the name, write the day config again with
`--report-filename-as-written '<the exact name>'`; it must equal the template's
official Filename, the readback shows it as KEPT AS WRITTEN, and
certification.json records it.

The report to upload is named as certify prints it: the template's name with
its placeholder numbered (`ODC_2026_Team54.txt` for `ODC_2026_TeamXX.txt`,
`ODC_2026_Team_54.txt` for `ODC_2026_Team_XX.txt`). Step 7 reads that file.

With `gmat.console_copy` on, a template with plots does not stop certify:
step 6c. The rest of this paragraph is the GUI path, the alternative when it
is off, or when certify says "No console copy was run".

When GmatConsole refuses the final script for its plots (step 3), certify stops
at `exact-run-01` ("GmatConsole returned exit status 1. final.script:
GmatConsole cannot create OpenFramesInterface DefaultOrbitView, ..."), so its
reruns and its final check against the official report never ran, and there is
no certification.json. The report must then come from GMAT with the GUI, as in
the preliminary (docs/LESSONS_2026-09-05.md §8): load the `staging/final.script`
the refusal message names, plots and all, run it, and put the report it
writes through step 7's `check_report_rows`, with the flags given by hand. On
the day that GUI is the organizer's: step 6b. Only a message that names
OpenFrames or GroundTrack and contains "Run it in the GMAT GUI" means that;
after it come the name to upload the report under, when the day config keeps
the organizer's, and where certify kept the run. One that quotes GMAT's other
errors first, says the GUI would reject the script too and names the plots
only after them ("...; besides, GmatConsole cannot create ..." or "...;
besides, it creates GroundTrack ...") is a stop (Stop rules). Any other failure, such as the refusals above (all exit 2, as this
stop does too), is a failed certification: go to the next fallback (step 4a),
never to staging. Exit 75, with the last line `orbitwin certify: NO ANSWER:
...`, is not a refusal: a GmatConsole run died of a signal, timed out or
wrote no report with no error GMAT wrote, or this machine failed (a full
disk); the run is kept as `<output>.no-answer`. Run the same certify again
before going to a fallback. Step 7
says what the `diagnostic/verification.json` the message names gives in place
of the certificate.

Certify's last line must read `PRE-CHECK PASS`, or `PRE-CHECK PASS (end time
only)` for a template whose report cannot show a burn. Before it ends, certify runs the
binding step 7 holds their report to on our own report, with this run's own
paths and frozen rows, and records the result as `binding_pre_check` in
certification.json, or in the `diagnostic/verification.json` of the folder a
plots stop set aside (`.refused` or `.refused-N`). It prints the command
first ("before the GUI, the binding step 7 holds their report to, on ours,
must exit 0", or "must exit 3 (end time only; the 6b.7 by-eye check is
mandatory)" when it carries `--accept-end-time-binding-only`), on a plots stop naming the report of the flight
verification.json scored (stopped at the hit, the whole flight for a miss, or
flown on to T_max when a planned hit missed) and the `staging/final.script` the GUI
will run. After a certified run it then runs step 7 itself, the `--certificate`
command certify printed, on the report certify packaged, and passes only when
that exits 0 (or leaves §4 note 1(b) to a person, below): certify's own step 7
cannot fail after a `PRE-CHECK PASS`. The pre-check never changes certify's exit
status.

Certify also reads its official report as printed, at its ReportFile's
Precision, as step 7 does, and refuses (exit 2, nothing submitted) one whose
printed rows disagree with the certificate: a last row other than T_team in its
printed digits, a hit's last row outside the sphere or short of
`policy.stop_depth_km`, a miss with a row inside. Verify stops a hit deeper by
what that rounding can take (`printed_margin_km` in the candidate's
`gmat_event`: √3 last digits at the craft's radius, 1.7 m at Precision 8 near
6600 km, 17 m at 6, 1.7 km at 4, capped at half the sphere), so the refusal
means a Precision no stop can survive: ask the organizer. The certificate's
`replayed_score`, and `official_report_rows.score`, are the printed rows' score,
which step 7 prints; the diagnostic flight's is `diagnostic_replayed_score`. On
R3's T14 at Precision 4 the certificate claimed 95.7635 where the printed rows
gave 95.7623, and its step 7 exited 1.

Certify refuses (exit 2) a run behind the certificate whose GmatLog records a
`**** ERROR ****` although GmatConsole exited 0, unless the record is on
`gmat_runner.BENIGN_LOG_ERRORS`. That list is empty: none of the 19,945 completed
R2026a runs under `output/` on 2026-11-07, the R1-R8 rehearsals among them, logged
one. Each such record is kept in the run's result (`log_errors`),
`verification.json` and `certification.json` (`gmat_log_errors`), and solve
prints a `WARNING:` for it. Read the GmatLog.txt the refusal names; take the next
fallback (step 4a).

- `PRE-CHECK PASS`: go to the GUI with this candidate.
- `PRE-CHECK PASS (target not in report): ...`, or the same ending "its burns
  were bound by the end time only: the 6b.7 by-eye comparison is mandatory":
  the official report carries no target position (step 3). The hand binding
  exited 7, "NOT JUDGED BY HAND", and step 7 itself passed on certify's report.
  Upload certify's submission, the only report anything can judge; a GUI's
  cannot be (step 7). Recorded as `pass-target-not-in-report` or
  `pass-target-not-in-report-end-time-only`.
- `PRE-CHECK PASS (end time only): the 6b.7 by-eye comparison is mandatory`:
  the official report has no EarthMJ2000Eq position and velocity of the
  pursuer, so no report of this template can show a burn, and certify ran the
  pre-check with `--accept-end-time-binding-only`. It exited 3: the end and a
  row at each burn epoch bind it, nothing binds the burns. Go to the GUI, and
  at step 7 upload only after 6b.7's by-eye comparison matches. Recorded as
  `pass-end-time-only`.
- `WARNING: PRE-CHECK SURFACE NOT JUDGED: ...`, `... TEMPLATE BELOW: ...` or
  `... UNDECIDED: ...`: §4 note 1(b) could not be decided on our report (step
  7: exit 4, 5 or 6), recorded as `surface-not-judged`,
  `surface-template-below` or `surface-undecided`. Not judged or undecided:
  read the pursuer's lowest altitude off GMAT by eye up to the interception
  (step 6b.7), and go on only if it stays above 0; or ask the organizer.
  Template below: ask the organizer before going on.
- `WARNING: PRE-CHECK FAILED: <why>`: do not go to the GUI with this candidate.
  Take the next fallback (step 4a). Never `--binding warn` for it.
- `WARNING: PRE-CHECK NOT RUN: <why>`: the line says why (no `--config`, a
  missing file, a report it could not find, a verification.json naming a
  flight of another run). Run the command certify printed by hand, or when it
  printed none, step 7's hand command (below) on our own report from the
  folder the line names, with that run's `verification.json`; it must exit 0,
  or 3 when the command carries `--accept-end-time-binding-only` (6b.7 is then
  mandatory), and anything else is a failure as above.
- `PRE-CHECK SKIPPED: the console is orbitwin's test double`: only in tests;
  on the day the console is GMAT's.

It checks the official report alone (ReportFile1 among those GMAT writes, else
the first): the one uploaded. Another ReportFile's report is not checked.
Certify's own stdout is flushed first, so with `2>&1 | tee` the line is still
the last.

`--runs` is at least 1; `--runs 0` is refused before anything is written.

Certify, solve and rank also refuse a candidate that GMAT flies under 100 km
altitude (`policy.minimum_flown_altitude_km`). The rules' line is the surface;
100 km is our margin for drag differences between GMAT builds. A refusal names
the altitude it flew. Lower the setting only by a decision, not to get a
submission through.

## 6b. Run our script in the organizer's GMAT GUI

Every candidate writes a report of the same name, the report name certify
prints (the official report's Filename, numbered, or kept as written;
`<report>` below), so each one gets its own folder on the USB, named as its
run folder: `final/`, `final-01/`, `final-02/`, `final-00/`. Everything below stays in
that folder. Into it goes our script: `run/<folder>/submission/<submission_script_name>`,
or, when certify stopped on plots (step 6), the `staging/final.script` its
message names, in `run/<folder>.refused` or a later `.refused-N`.

On the organizer's computer, open cmd.exe, not PowerShell: in PowerShell `>`
writes UTF-16, which the laptop's `grep` cannot read, and `fc` is
Format-Custom. Every Windows command below is cmd syntax.

1. Delete any `<report>` in GMAT's output folder: `OUTPUT_PATH` in
   the GUI's startup file, in a stock R2026a install the `output` folder
   beside `bin` (`OUTPUT_PATH = ../output/`). A report left by an earlier run
   looks exactly like a new one.
2. Copy the script into the folder that holds the official script and the
   files that came with it as `<folder>-gui.script`: `final-gui.script`,
   `final-01-gui.script`, `final-02-gui.script`, `final-00-gui.script`. That name is never our
   script's own (`final.script` on the plots path, else the
   `submission_script_name`), so the copy coming back in 4 cannot land on
   it. If it is already in the organizer's folder, ask before going on;
   never copy over the official script. GMAT looks for a relative tide, gravity or
   SPAD file in the script's folder first (step 1). Open that copy in the GUI
   and run it to the end, plots and all. Never save it from the GUI; answer No
   if it asks: a GUI save writes the file out again in GMAT's own form.
3. Check that `<report>` is back in the output folder, dated now.
4. Copy that report and the script copy the GUI ran into the candidate's USB
   folder. In cmd.exe, in that USB folder, save both hashes there:
   `certutil -hashfile <report> SHA256 > report.sha256.txt` and
   `certutil -hashfile <the script copy> SHA256 > script.sha256.txt`. Do not
   copy hashes by hand.

On the laptop, in that USB folder:

5. `cmp <the script copy> <our script>` must print nothing (on Windows,
   `fc.exe /b` in cmd.exe says "FC: no differences encountered"). Any difference means the report
   is not from our script: start again at 1 with a fresh copy. `orbitwin
   verify-policy` is no substitute: an edited burn still passes it.
6. Step 7, by hand, on `<report>`, with `--script` our script. It refuses
   (exit 2) a report that script did not write: one that ends elsewhere
   than its Propagate goals, lacks one of its burns or holds one it lacks,
   or is named other than its official ReportFile writes. A report without
   the pursuer's EarthMJ2000Eq position and velocity cannot show its burns:
   only its end and a row at each burn epoch bind it, it exits 2 unless
   `--accept-end-time-binding-only` is given, and with that flag it exits 3
   (below). Then step 7 below is the only binding of the burns.
7. Compare by eye, with or without plots: the report's last-row time
   (`validator reads last row Tteam`) must equal `t_team_s` from step 7's
   one-liner, and its score line the one-liner's `score`, to the digits
   printed. Step 6 checks the report against the script, not against the
   candidate certify measured; this checks that.
8. `grep -ic "$(sha256sum <report> | cut -c1-64)" report.sha256.txt`
   must print 1, and the same for the script and `script.sha256.txt`.

Back on the organizer's computer, right before uploading, in cmd.exe in the
same USB folder, hash the two files again with the same commands into
`report-again.sha256.txt` and `script-again.sha256.txt`, then
`fc.exe report.sha256.txt report-again.sha256.txt` and
`fc.exe script.sha256.txt script-again.sha256.txt` must each say "FC: no
differences encountered". Then upload those two files, from that folder:
`<report>`, under the report name certify prints, and `<folder>-gui.script`,
the copy the GUI ran (the file `script.sha256.txt` hashes), under that name.
Not our script beside them: it is the same bytes (5), but the hashes are of
the copy. Neither the ranking rules (2026/11/04 revision) nor
anything the organizer has said or written to us so far names the uploaded script; the
briefing (p.8) says only to upload the Script and the Report, and only the
report's name carries our number. If the organizer names the script file, give
the copy that name in 2 instead, before anything is hashed: `certutil` writes
the file's name into its hash file, so a rename after 4 makes the `fc` check
fail. Never a report named `TeamXX`. A template whose
report name the organizer told us in writing to keep, written into the day
config with `--report-filename-as-written`, uploads under that exact name.

## 6c. Copy mode: certify and upload from this workstation

When the day config has `gmat.console_copy: true` (the ranking profiles set it;
thc1006 signed it off on 2026-11-06 in docs/COMPETITION_POLICY.md, and the
`plots` line of step 2's readback says "gmat.console_copy on"), a template with plots is
certified here and its upload comes from here, as in the preliminary. This
replaces 6b on this workstation; 6b stays the alternative.

Certify runs the final script itself first, in `exact-run-01`. When
GmatConsole refuses it for its OpenFrames or GroundTrack plots alone, certify
writes `run/final/console_copy/final_console_copy.script`: the final script
with whole lines left out, each a Create of an OpenFrames or GroundTrack
object, an assignment to one, or a Toggle, PenUp, PenDown, ClearPlot or
MarkPoint naming only them. Before it runs, the copy is proven from disk:
every other line is byte for byte the final script's, line ending included,
and no kept line names a plot. Certify does not judge whether the GMAT GUI
takes the plot lines, which only the GUI reads: the policy guard requires them
to be the official script's own, copied verbatim, the same lines in the same
order, so the GUI takes the upload exactly when it takes the organizer's
template. Every exact run then runs the copy, in `console-exact-run-01`
and on. The diagnostic sibling, run off the final script itself, must write
the same official report byte for byte, as in step 6.

`certification.json` then has `console_copy` (the copy's path and SHA-256,
the final script's SHA-256, the objects and lines left out, and the refused
run), every exact run's `script_sha256` is the copy's, and `submission_files`
binds both files to upload: our script, every plot line of the official
script kept, and the copy's report under the report name. The script is never
named `team_solution.script` on this path. Certify still ends on step 6's
pre-check, on that report, and its last line must read `PRE-CHECK PASS`.

Certify prints the upload on stderr ("... the upload is these two files from
this machine, as they are (docs/RUNBOOK.md step 6c)"):

- `run/final/submission/<report>`, under that name;
- `run/final/submission/<submission_script_name>`.

Step 7 with `--certificate` checks that report directly: it rebuilds the copy
from the script, proves it again, and requires the copy's SHA-256 to be the
one every exact run ran; the last row must match T_team. It must exit 0.
Never upload `console_copy/`, `official_original.script`, `certification.json`
or the `.diff.txt`.

Certify refuses with "No console copy was run (gmat.console_copy): <why>"
after the step 6 message when:

- GMAT reported errors besides the plots', or more than it read: the message
  says the GUI would refuse the script too. Stop (Stop rules).
- The copy cannot be proven (the line is named, such as an OpenFrames field
  naming an object nothing creates). The message still says "Run it in the
  GMAT GUI": that is 6b, and only there.

A copy that runs and fails ("That was a run of final_console_copy.script, ...
with the plots left out it still fails") is a failed certification: take the
next fallback (step 4a).

## 7. Read the report you are about to upload

```bash
python scripts/check_report_rows.py run/final/submission/<report> \
  --config run/day-scored.yaml --certificate run/final/submission/certification.json
```

`--config` gives the rules, the craft, T_max (`--horizon-s`), the time origin
and kt, Ct, kv, Cv and C_r. `--certificate` gives `--frozen-rows`, the total delta-v,
the script beside it (`--script`), the time column (the ElapsedSecs of
`clock.time_origin_craft`, the craft that started first; SC_Human's when the
craft share an epoch) and T_team, which the last row must match. The report must
be bytes the certificate lists (SHA-256), and the script, given or beside the
certificate, the submission script's own entry there, whatever format flags are
given. A certificate field of the wrong type, or a number that is NaN or
infinite, exits 2. The certificate must agree with the config on the rules, the
time origin and T_max, and a config that says the craft start together (no
time origin, or `earliest` with no `--closest-approach-window`) refuses a
report that holds an epoch alignment. It must agree on the coefficients too: a
certificate priced under another score block (`score_block`; C_r 300 checked
under 3000 printed 23.6716 beside 31.5363 and passed) exits 2, so certify again
under the announced ones. One certified without coefficients, as in step 4a, is
priced by the check alone. When the report carries both craft's ElapsedSecs, the
one that runs ahead is the earlier craft's, and a time column naming the other,
from the certificate or by hand, is refused: it is T_team short by the head
start (R8, SC_Human first: 6101.4377 s read as 4301.4377 s). A value given by hand
that `--certificate` or `--verification` also gives must be theirs, or the check
exits 2 naming both: `--total-dv`, `--frozen-rows`, the coefficients,
`--horizon-s` (T_max), `--radius-km` (the rules' sphere) and `--time-column`
(`clock.time_origin_craft`). `--total-dv 2999` beside a certificate printed
78.9902 for its 95.8797 and exited 0 before this; give none of them by hand. The
same holds for `--config`: a flag it gives otherwise (`--rules`, `--pursuer`,
`--target`, `--horizon-s`, the coefficients, and `--time-column` when the time
origin is not `earliest`) exits 2 showing both, so only what the config leaves
unset is given by hand (an R6 miss cut at 7369.13 s passed as scored under
`--horizon-s 7369.13`; `--kt 0.0003 --cr-km 3000` printed 81.7248 for 69.2921). Each
certificate's own replayed score is the one to match. The certified rehearsals
of 2026-11-02 gave R1 96.3908, R6 95.0657 and R7 a scored miss at 23.1809; R6
certified again on 2026-11-04 gave 95.0656. All were under the distance term
before the 2026/11/04 revision.

With a certificate, the score printed must be the certificate's
(`official_report_rows.score`, priced on the same printed rows), or the check
exits 1: certify and step 7 read one report one way.

An official report with no SC_Alien position (step 3) cannot show a distance.
With `--certificate` the check then binds the report by its SHA-256, requires
its last row to be the certified T_team in the digits it prints (T_max for a
miss), judges §4 note 1(b) on the pursuer's position to the last row if the
report has it (exit 4 if not), prices the hit, or the miss at the d_min of
certify's diagnostic flight, holds that score to the certificate's, and exits 0
with its last line "WARNING: the official report carries no SC_Alien position;
d_min and the interception were proven on certify's diagnostic flight of the
same script, not on this report". A certificate written before certify recorded
that (`official_report_rows`) exits 2: certify again. Without a certificate,
the hand command below binds what it can and then exits 7, its last line
"VERDICT: NOT JUDGED BY HAND: ..."; 7 is never a pass on its own. It means: use
the `--certificate` command on certify's report. A GUI's report of such a
template cannot be passed by anything here (step 6b), and whether the
organizer's validator can judge it, from its own data, is the open question of
step 3.

In copy mode (step 6c) the report to upload is the one certify hashed, so the
command above, with `--certificate`, is the whole check; it also proves the
console copy again. The report from the organizer's GUI (step 6b) is not the file certify hashed,
so `--certificate` refuses it ("SHA-256 differs"). Check it by hand, with our
script and the values certify recorded. The one-liner reads
`run/final/submission/certification.json`, or
the `diagnostic/verification.json` the refusal message names when certify
stopped on plots (pass `--script` the `staging/final.script` it names then);
use `run/final-01/` and so
on for the other candidates:

```bash
python -c '
import json, sys
d = json.load(open(sys.argv[1])); c = d["candidate"]
if "clock" in d:
    t_team, score = d["clock"]["t_team_s"], (d.get("final_script_parity") or {}).get("replayed_score")
else:
    stop = d["event"].get("stop_time_s")
    t_team = None if stop is None else stop + (c["metadata"].get("time_origin_offset_s") or 0)
    score = c.get("score")
print("total_dv_km_s", c["total_dv_km_s"], "frozen_rows", d.get("frozen_rows"), "t_team_s", t_team, "score", score)
print("score_block", d.get("score_block", "absent: made before it was recorded, certify again"))
' run/final/submission/certification.json
python scripts/check_report_rows.py <usb>/final/<report> --config run/day-scored.yaml \
  --script run/final/submission/<submission_script_name> \
  --frozen-rows <frozen_rows; leave the flag out if it is None or 0> \
  --verification <the diagnostic/verification.json of the same run>
```

A report without the pursuer's EarthMJ2000Eq position and velocity (no `VX`,
`VY`, `VZ` of it in the official ReportFile's `Add`) cannot show a burn: add
`--accept-end-time-binding-only` to the command above. It then exits 3 where it
would exit 0, and 3 is its pass, upload only after 6b.7's by-eye comparison
matches. Without the flag it exits 2 for every candidate alike, which is not a
reason to take the next fallback.

`--verification` is the `diagnostic/verification.json` of the same run: when
certify stopped on plots, the one the refusal message names, never an earlier
`.refused` folder. It must have verified the `--script` file (its
`candidate_script_sha256`), and its `score_block` must be the config's, or the
check exits 2 as with `--certificate`: `score` above was priced under it. A
`score_block` of None, a run certified before the coefficients as in step 4a,
passes with a note, and the score printed is the check's alone. Without
`--verification` a note says the coefficients were not checked.

Without a certificate `--script` binds the report to the script, and any
mismatch exits 2 naming what differs:

- Σ|Δv| is the script's: the norm of each Maneuver's Element1-3, in km/s,
  scored in the config's unit. `--total-dv` is not needed; given, it must be
  that total to 1 mm/s in Cv's unit (a km/s figure typed for m/s is refused).
- The last row must be the script's Propagate goals summed from the epoch
  alignment's end, the row where the waiting craft starts, to the larger of
  0.1 ms and the digits Precision prints. An alignment in the report and not
  the script, or the other way round, is refused, and so is a report whose
  alignment rows were cut: the aligning craft's ElapsedSecs must start at 0 on
  the first row, a craft must hold still on them, and no step there may be
  longer than MaxStep. A cut alignment is not this script's flight, and
  `--binding warn` does not touch it. Under `from_origin` those rows decide
  d_min. The step check has a limit: a cut is seen only when it leaves a step
  longer than MaxStep. At MaxStep 2700 s an R6-R8 alignment of 1800 s has a
  handful of rows, steps shorter than MaxStep, so rows cut from inside it
  leave no such step and pass; at MaxStep 1 s and 60 s any cut row shows.
- Each burn must have a row at its epoch, and the next row, flown back to it
  under point mass and J2, must show the burn's impulse as a vector, and must
  continue from that row. R2026a writes no row after a Maneuver, so that row
  is one step on. Elsewhere the rows must follow one another the same way: a
  burn the script lacks is refused. The allowance is 1 cm/s plus what the
  neighbouring steps show the other forces pulling (drag near 100 km on R3-R5).
  A report without the pursuer's EarthMJ2000Eq position and velocity cannot
  show its burns: it must still hold a row at each burn epoch, and it exits 2
  unless `--accept-end-time-binding-only` is given (step 6b.6), which
  `--binding warn` does not stand in for.
- No step may be longer than the script's propagators take (MaxStep, or
  InitialStepSize for a fixed step), past the last printed digit: longer is
  rows missing ("rows missing between t = ... and t = ..."), where a burn
  could hide. Every row GMAT wrote must be in the report.
- The report's file name must be the one the script's official ReportFile
  (ReportFile1 among those GMAT writes, else the first) writes.

- At MaxStep 60 s and more R2026a may end the alignment on two rows at one
  instant: its own last row and the next Propagate's first, at the same
  position, 3e-7 s apart and the second sometimes earlier. With a burn right
  after the alignment the second is post-burn (R8); it happens with the first
  burn later too (R8 with it at t = 60 s, the review of #185). t = 0 is the first of them, as certify counts `frozen_rows`, so the
  certificate's or verification's count is the one to pass; a check counting
  the second made a valid miss "unchecked" and failed step 6's pre-check.
- Each row's time is good to the last digit Precision prints, so the
  allowances grow by twice that digit times |v| (position) and |a|
  (velocity): at Precision 6-8 R3 would otherwise be refused.

The burn check has a limit: a burn the script lacks shows only above what
the other forces may pull over one step, about 1 cm/s plus 1e-5 km/s^2 times
the step. At MaxStep 1 s that is 2 cm/s; at 60 s an extra 0.2 m/s burn
passes (the review of #185 flew it); at 300 s about 3 m/s does. The burns
the script has are checked as vectors against the same allowance, and the
end and the total bind whatever the step.

`--binding warn` turns a refusal over the burns (no row at a burn, a jump
that is not the burn, rows that do not continue or are missing after the
alignment, a burn the script lacks) into `WARNING:` lines. It never touches
the end, the alignment (there, from 0, no rows missing), the name or the
total. Use it only here, at step 7, on the
organizer's report, when certify ended on `PRE-CHECK PASS` and the refusal
is one of those burn checks; never after a failed pre-check (step 6).

Exit 3 means the report passes only with the by-eye check done. Under
`--binding warn` with a warning, or with `--accept-end-time-binding-only`,
the check exits 3 where it would exit 0, and its last line is "WARNING:
burns not bound by this check: the 6b.7 by-eye comparison is mandatory".
Do step 6b.7, and upload only if the last-row time and the score match the
one-liner's.

`--certificate` binds by SHA-256 instead, and does none of this; the two
flags above are refused with it.

In verification.json `t_team_s` is None for a miss: its T_team is T_max
(certification.json gives it). `score` is None before the
coefficients. verification.json records `frozen_rows` as certify counts them; one
written before it did has none: after an epoch
alignment, run the check once without `--frozen-rows`; a note on stderr says
"holds still for N rows ... --frozen-rows is 0" (or, when those rows are not
EarthMJ2000Eq, the refusal names `--frozen-rows N`). Run it again with
`--frozen-rows N`. Then compare by eye (step 6b, 7).
The binding does not need `--frozen-rows`: it finds the alignment's end
itself, the first row at the instant the waiting craft starts from.

With `--config` and no certificate, as in the hand command above, the time
column is the ElapsedSecs that runs ahead when the report carries both craft's.
A report with one craft's ElapsedSecs and no alignment rows cannot show whether
that craft started first: under the ranking rules it is read as T_team only
with `--certificate`, or with `--config` and `--script` (the submitted script),
whose epochs then show the craft start together; otherwise it is refused. So
the hand command always passes `--script`. A check run by hand without
`--config` needs `--time-column`: without it the script takes the ElapsedSecs by
name, whatever column 0 holds (`SC_Human.Earth.Altitude` in the R6-R8
rehearsals), and refuses a report whose two craft's ElapsedSecs differ, as they
do after an epoch alignment. With both craft on one epoch, either craft's
ElapsedSecs will do. With an epoch alignment, pass `--frozen-rows` with `frozen_rows` from
`run/final/submission/certification.json`, and `--time-column` naming the
ElapsedSecs of the craft that started first (`clock.time_origin_craft`). Each craft's
ElapsedSecs counts from its own epoch, so that column is T_team as scored;
the certificate's `clock.t_team_s` holds the same number. If the official
report has no ElapsedSecs column for that craft, read T_team from
`clock.t_team_s` and take only the row checks from this script: the other
craft's column is short by the head start.

For a hit it must exit 0: the **last row** of the report has to be inside the 5 km sphere.
It prints how deep (`depth : the last row is ... m inside`). By hand, on the
organizer's report, that is all the rules ask: a last row at least 1 mm inside
(what reading the printed digits can move it) passes. One short of the
config's `policy.stop_depth_km` passes with a WARNING naming both depths: their
GMAT ended our script shallower than ours, which is what the stop depth is the
margin for (25.5 m to 665 m measured between builds and data files). With
`--certificate`, on certify's own report, the stop depth is still required, as
certify requires it.

For a scored miss it exits 0 only if the last row is at `--horizon-s`, the
announced T_max (within 1 ms), and prints the miss priced on its closest row.
By hand the price needs `--cr-km <C_r>` beside `--kt --ct --kv --cv --total-dv`;
without it the miss is left unpriced, and `--cr-km` under `--rules preliminary` is
refused. A certificate made under another revision of the rules than the
config's (its `rule_set.sha256`) is refused too: certify again.
`--horizon-s` is the day config's `policy.horizon_s`, on the same clock as the
time column. certification.json's candidate `horizon_s` is the solver's clock
and differs from it by `clock.time_origin_offset_s`, the head start when the
craft start apart (R7: 7591.381 s against 5791.381 s). A hit whose first row inside comes after `--horizon-s` fails
(§3), a `--radius-km` that turns a pass inside 5 km into a miss is refused, and
a `--frozen-rows` that does not match the alignment leaves a miss unchecked.
§4 note 1 is judged on every plan that gives the burns: the script (`--script`,
or the one the certificate lists), the certificate's or verification's
candidate, and the total (`--total-dv` included), against the rulebook's own
limits, not our safe margins. An impulse over 250 m/s plus ε_v (note 1(a)) or a
total over 3 km/s (note 1(c)) prints `DISQUALIFIED:` naming the rule and exits 2:
do not upload it. A burn less than 100 s after the one before warns and takes
10 points per violating maneuver off the score printed (note 1(d)); under the
preliminary rules an impulse over 1.5 km/s does the same (§5).

Note 1(b), a crossing of the Earth before the interception, is judged on the
report's own rows, on every row up to and including the first inside the sphere
(a miss: to the last row), the alignment's included. The organizer judges GMAT's
geodetic `Earth.Altitude` below 0. The check takes the pursuer's
`Earth.Altitude` column, or the short form `Altitude` GMAT writes the same
value under, and never the target's. Without one it takes the height above
the WGS-84 ellipsoid (GMAT's 6378.1363 km) from the pursuer's EarthFixed,
EarthMJ2000Eq or EarthICRF position, or EarthMJ2000Ec turned through J2000's
obliquity. From EarthFixed that is within 0.3 m of GMAT's; the others leave out
precession and nutation since J2000, 54.6 m at most measured (62.7 m at
Precision 6). It prints the lowest (`surface : ... at least ... km`), and:

- below 0 on a row our burns flew: `DISQUALIFIED:` naming note 1(b), exit 2.
  Do not upload it. Below the surface after the interception the rules allow,
  and it is not judged;
- below 0 only on the rows the template fixes (the alignment's and the one at
  t = 0, row 1 without one): exit 5, "the template's initial state is below
  the surface; ask the organizer". The rules do not say whether that counts;
- a height computed from the position within 100 m of 0 (1 m from EarthFixed,
  plus the printed digits): exit 6, `SURFACE UNDECIDED`. GMAT's column decides:
  read the pursuer's lowest altitude off GMAT by eye (step 6b.7), or ask the
  organizer. With GMAT's column the judgment is strictly below 0;
- no altitude column and no position it can use (no names, or another
  frame): exit 4, `SURFACE NOT JUDGED`. Check the lowest altitude by eye in
  GMAT up to the interception, or ask the organizer.

Exits 4, 5 and 6 are not passes and not refusals: they come after the other
verdicts, as the last line, only when those passed; a person decides. The
check sees the rows only: a dip between two rows, shorter than a step, does
not show. Certify's 100 km floor (`policy.minimum_flown_altitude_km`, on every
row GMAT flies, and the ranking search's own wall) is the guard against that.
The score line says which clauses were judged, and on what.
Without `--horizon-s` it says the miss is unchecked and exits 1. The closest
row is the one verify and certify score a miss on, rank's `minimum_distance_km`;
the Hermite minimum between rows is `gmat_event.interpolated_minimum_distance_km`
(R2: 4711.157 km on the rows, 4711.155 km between them).

`check_report_rows.py` refuses a report it cannot read one way only (exit 1,
no VERDICT line); appendix D lists the cases.

`orbitwin inspect` warns on stderr about each official ReportFile that cannot
show what the rulebook scores (see `src/orbitwin/official_report.py`).

`scripts/audit_burn_table.py` replays the burns in two-body. Under the ranking
force model it judges section 1 (magnitudes, total, separation, window) and the
surface and altitude walls of section 3, held through T_max. Section 4 checks
only the replay against itself. It does not judge the interception or the score,
and says so on its `AUDIT:` line: on R1 it called a GMAT-certified hit a 19.97 km
miss. Take those two from GMAT's report. It reads the burns from a file of
their own, made from the certificate (or from verification.json when certify
stopped on plots):

```bash
python -c 'import json, sys; b = json.load(open(sys.argv[1]))["candidate"]["burns"]; open(sys.argv[2], "x").write(json.dumps({"burns": b}))' \
  run/final/submission/certification.json run/final/burns.json
python scripts/audit_burn_table.py input/original/official.script run/final/burns.json \
  --config run/day-scored.yaml --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP"
```

The one-liner refuses to overwrite: give each candidate its own path
(`run/final-01/burns.json`), and pick a new name to make one again.

It audits those burns, not the script's bytes; that the script carries them is
what certify's hash of the flown script and step 6b's `cmp` show.

Measured 2026-09-05. The submission passed every gate above -- the diagnostic
replay showed the entry, the exact reruns reproduced byte for byte, the policy
guard was green -- and scored 73.0203 instead of 95.2470. The organizer's
validator returned D = 2.394 km, its own confirmation of the 5 km entry, next
to C = 23815.43 s: it takes the completion time from the last row of the report,
and the script ran to 4TA. The 22.2266-point time term went to zero.

Run it on the file that is actually going to be uploaded, not on a local replay
of it. Upload on exit 0, or on exit 3 after 6b.7's by-eye comparison matches. A
hit that exits 1 (its last row is not inside the sphere, or its entry is after
T_max) or 2 (it is not our script's report, a file disagrees, or the rules
disqualify it) is not uploaded: take the next fallback (step 4a). One exit 2 is
not a fallback: a report that cannot show a burn without
`--accept-end-time-binding-only`; give the flag (above). Exit 4, 5 or 6 is §4
note 1(b) left to a person (below): upload only once that is settled. A last row short of the stop depth
is a WARNING, not an exit 1 (above).
`policy.mission_end_policy` is already `interception` in every ranking config
(the default; no profile or flag of `ranking_day_config.py` changes it), so
certifying again changes nothing. A successful candidate ends where GMAT
measured it inside the sphere: verification finds the first point 1.5 km inside
(`policy.stop_depth_km`), flies the script stopped there, and refuses it unless that
flight's last row is inside. The hit is scored at that row's time, which is
what the validator reads.

The stop depth (`policy.stop_depth_km`, 1.5 km), the aim, the shallow-stop
opt-in and the mission end are changed only by a decision; how, and why the
defaults are what they are: appendix E.

A candidate that misses runs to T_max: there T_team is T_max and the distance
term is `100/(1 + e^{((d_min − 5)/C_r)²})` of the closest approach, worth up to
50 points.

## Stop rules

- No unverified new candidate in the final 40 minutes.
- Never hand-edit a Report `.txt`, and never save the script from the GMAT GUI (step 6b).
- Never accept a candidate whose Script/report did not come from the same clean run.
- When Python and GMAT differ, GMAT wins and the discrepancy is investigated.
- Never upload a hit whose last row is outside the sphere, or a miss whose last row is not at T_max (step 7).
- When GMAT rejects the official script itself for errors beyond its plots (step 3), neither GmatConsole nor the GUI runs it, and it may not be edited: stop and take it to the organizer. First rule out a file missing from this machine's GMAT install or run folder, which is ours to fix.
- When the parser refuses the official script itself for how GMAT would read it (a stop character, a control character, a `%`, a double quote or a `...` it does not model; step 3), it may not be edited: stop and take it to the organizer.

## Appendix: design notes behind the steps

Read these before the day, not during it.

### A. What GmatConsole, the state probe and the parser refuse

With a console, every command first runs the state probe: GMAT reports each
craft's initial epoch and EarthMJ2000Eq state, and those are what the solver
uses. An EarthFixed or SphericalAZFPA insertion state, or epochs written in two
formats, can only be read that way. `scenario.json` records the probe, the
epoch offset and how far GMAT's states are from our own conversion.

The error names those objects whenever GmatConsole exits non-zero on a script
that creates OpenFrames objects. It names a GroundTrack only in the case
measured on R2026a: the missing report is one the script writes, and a
GroundTrack is created before that ReportFile (after it, the report was
written). Any other missing report is reported as just that. A timeout or the
file-size stop is never put down to the plots. A ReportFile with
`WriteReport = false` is never written, by GmatConsole or the GUI (R2026a
writes no file, not even a header), so no run waits for it.

An error is put down to those objects by the script line GMAT prints after it:
the `Create` or a field of an OpenFrames object the script creates. Only an
error without a line falls back on R2026a's wording. GMAT's other errors, its
`**** ERROR ****` records and the `... Exception:` lines it writes without that
marker, are counted once each, and the first three are quoted first, with that line
(`GMAT reported: ... (line 220: ...)`), the rest as "(and N more)". Then the GUI is not offered: it shares
that interpreter and refuses the script for them too, so neither GmatConsole nor
the GUI can run it as written. The official script may not be edited, so that
is a stop: take it to the organizer (Stop rules below). When some of those
errors name a file GMAT could not open or find, the error says to check this
machine first: a data file missing from our GMAT install, its startup file or
the run folder is our fault, not the script's, and the organizer's GMAT may run
it. If GMAT wrote more than 10 000 distinct errors, the error says so first, quotes up to three of the ones read, counts the rest and does
not offer the GUI either: what GMAT said is incomplete, so whether the GUI would
run the script is unknown. That is a stop too.

The parser reads a script's lines as GmatConsole does, ending them at CR and LF
only. GMAT stops reading at a NUL byte, and a Windows GMAT also at Ctrl-Z
(0x1A), so a command after either would not run while the run still exits 0: such
a script is refused. NUL padding, or a comment or blank lines after the stop,
are accepted, as GMAT runs them with the same report; a render keeps the stop on
the last copy of an open template's Propagate only, and the earlier copies carry
its command alone. Every build stops at a NUL, whatever stopped a Windows one
before it, so nothing after a NUL is checked. Ctrl-Z is accepted only in
a comment: outside one, R2026a on Linux refuses it as a command. VT, FF, `\x1c` to
`\x1f` are read as GMAT reads them in a comment and refused outside one, a quoted
value included, where GMAT keeps them as text.

A line's comment starts at its first `%`, as GMAT's TextParser reads it, so
`X = 'a.txt' % was 'b.txt';` is refused (its `;` is in the comment), not read as
the file name `'a.txt' % was 'b.txt'`. When that first `%` is inside the line's
first pair of single quotes, GMAT reads the whole line as the command, any later
`%` included: `'Report;%1.txt';` is the file `Report;%1.txt` to both, while
`'Report%1.txt'; % note` made R2026a write a file named
`'Report%1.txt'; % note`, quotes and all, and the parser refuses it. A `%` in a
later single-quoted value is text to both. Double quotes are not quotes to GMAT:
`"Report%1.txt"` ends at its `%`, and GMAT refuses the value; the parser refuses
a double quote outside single quotes and outside a comment, and reads one inside
them as text, as GMAT does.

A line that is not a comment line and ends in `...` continues onto the next
(ScriptReadWriter::HasEllipse): GMAT reads it up to a NUL, trims trailing blanks
and then semicolons, and looks only at the end, comment included. On R2026a
`X = false; % window state...` swallowed the line after it, and
`X = false; % note...;` was refused ("Ellipses must be at the end of the line").
The parser refuses such a line by its number, reading it also as a Windows GMAT
does, up to a Ctrl-Z. `...` anywhere else, in a comment line or a value, is text.

The parser's refusals in the three paragraphs above are of the script as written.
If it is the official script, it may not be edited: stop and take it to the
organizer (Stop rules below).

### B. The score terms, ceilings and levers (step 5)

The 2026/11/04 revision fixes the units
(§4, under eq. (5)): T_team in s, ΔV_team in m/s, so kv is per m/s and Cv in m/s,
and Δr_min and C_r in km. The config refuses `--delta-v-unit km/s` and a score
without `--cr-km`, whose name carries the unit and has no abbreviation; a value
typed in the wrong unit still raises no error. A score
block written by hand must state `time_unit` and `delta_v_unit` too, and
`cr_km`: a config without them is refused rather than guessed. The printout shows
each term over its whole range and warns
when one cannot move by half a point, or when it drops from 22.5 to 2.5
points within 1 % of its range, which is what km/s values entered as m/s look
like. A flat term can be real: the preliminary's Cv of 9.06 km/s would make
delta-v cost nothing under the ranking's 3 km/s cap. The distance term is
printed at 10, 100, 1000 and 10000 km, where it reaches 49 points (about
`5 + 0.2·C_r`; it is flat inside) and 25 points (`5 + 1.048·C_r`), and, with
`--zero-burn-dmin-km`, at the zero burn's closest approach from step 4, which
says whether misses matter on this scenario at all. The flag is optional;
without it that line is left out. Read those against the scenario's
distances. The tool cannot tell from C_r alone which unit it was typed in, so
it only checks the range. A `NOTE` says where a miss falls under 1 point when
that is within 500 km, which the organizer's revision email of 2026-11-04 says
the term does not do (「不會在數百公里內即降為零分」); a C_r of 100 km gives one.
A `WARNING` means the term is still above 49 at 10000 km, which no scenario
could use. `--cr-km` and `--zero-burn-dmin-km` refuse anything but a positive
finite number, and `--delta-v-unit` takes only m/s. It also prints a `WARNING`
for a coefficient no announcement would give: kt or kv not positive, Ct under 1 %
of T_max, Cv under 1 % of the 3 km/s cap (a km/s value typed as m/s), a time or
delta-v term under 1 % of its 25 points at zero time or zero delta-v, and a C_r
outside 1 to 1e5 km; and a `NOTE` for a Cv above the cap, which is not a slip
(the preliminary's was 9064.3 m/s) but makes delta-v cost little. Each of these,
like every `WARNING` and `NOTE` above, stops `orbitwin day` at step 5 until
`--accept-config-warnings`. Ct is not held to SC_Alien's perigee: a hit can come
before it (R1's does). Two team members check
it. A value read wrong goes into a new file, `run/day-scored-2.yaml`, used in
steps 5 and 6 from then on; the tool never overwrites one. It also
prints two ceilings: no answer scores above the first, and no miss above the
second. Once a verified hit scores above the second, stop looking for misses.
`rank` prints the ceilings under `ceilings`, on the solver's clock, so after an
epoch alignment they can be tighter than the readback's. With them come the best
submittable row (`best_submittable`, `best`) and `misses_that_beat_best`: `none`,
`any`, or `closer`, with `miss_must_be_within_km` the distance a miss with the best
delta-v term must beat. It is a number only for `closer`. On R2 that depends on
C_r. With the zero burn the best row, its 4711.155 km keeps a term above 0 from
C_r of about 178 km up, and `closer` gives 4711.155. Below that its term is
exactly 0, and so is any miss's past `5 + C_r·√700` (2650.75 km at 100 km): such
a miss only ties, so `closer` gives that edge.
`levers`, beside it, says where the score moves for that row: `headroom`
holds what each term, and the deductions, leave below its best. `marginal` holds
the derivatives: the points one second earlier, one km/s saved (per km/s, though
kv and Cv are in m/s) and, for a miss, one km closer are worth at the row. At
0.02 pt/s and 0.5 pt/(km/s), `break_even_s_per_km_s` is 25: a little more delta-v
pays for itself if it buys time at 25 s per km/s or better. That is a slope, not a
price. Over one 249 m/s impulse the delta-v term bends by up to 11 %, and over
1000 m/s by up to 45 %. So `spent` gives what the score itself loses for one more
impulse at the safe per-impulse limit and for the rest of the safe total budget.
On the ranking day `orbitwin solve` spends it (step 5).
`safe_delta_v_floor.py` and `hunt_multi_burn.py` are the preliminary round's
tools (docs/LESSONS_2026-09-05.md) and the ranking day does not use them:
measured on R1, `hunt_multi_burn.py` timed out at 900 s where solve takes
211-250 s. A miss has no time rate, since only becoming a hit
moves it off T_max. Its distance rate is 0 at 5 km under the revised term, so a
near miss loses little by being a few km further off. With kt at or below 0 an
earlier arrival does not pay, and there is no break-even. A score floored at 0
has no rates at all (`floored`).

Whether R2's best is its zero burn depends on C_r: in the two-body rehearsal
search of examples/ranking/README.md (R2) it is at C_r 100, 300 and 1000 km,
and at 3000 km a 2.99 km/s miss at 1973 km scores 46.00 against the zero
burn's 31.48.

### C. The submission script's name (step 6)

Its name follows the evidence tier the run established, and is not to be
changed. It is `team_solution.script` only when the run conferred `certified`,
for a GMAT declared and fingerprinted as the organizer's:
`gmat.organizer_compatible: true` with `gmat.organizer_executable_sha256`, and
optionally `gmat.organizer_startup_sha256`, which
`src/orbitwin/config.py` says to set only once the organizer has confirmed the
build, startup file and data files. Step 3's `run/baseline/baseline.json`
records the hash of the GmatConsole that ran (`run.executable_sha256`). That
hash is for comparing against what the organizer confirms, in writing; it is
not the confirmation. `ranking_day_config.py` writes neither. In the R1 and R6 rehearsals of 2026-11-02 the script was
`public_gmat_candidate.script`.

### D. How check_report_rows reads a report (step 7)

`--script` makes it split the rows the way that script's ReportFile wrote them:
`Delimiter`, `WriteHeaders`, `FixedWidth` and the `Add` list, which names the
columns of a header-less report and places any Gregorian (text) column. Without
it the report must be in GMAT's default format; a comma or header-less report is
then refused, never misread. A header-less one can be read with `--pursuer-cols`
and `--target-cols`; column 0 is then shown as t and nothing is scored. Measured on R2026a (`tests/fixtures/gmat_report_formats`):
fixed width pads with the delimiter repeated, fixed width without headers pads
with spaces whatever the delimiter, and a script's `'\t'` writes a backslash.
Positions are read in any frame both craft share; with one craft's columns given
by flag, in the frame their names say. On a named report the two sides must be
two craft in one frame, whether found by flag or by name. Like every step it
runs in the project's venv, and refuses NumPy, SciPy or PyYAML off their pins,
with exit status 2.

A line that does not read as numbers after the first row is refused, where it
used to be skipped, and so is an epoch column that goes back: either means the
file holds more than one run (`AppendToExistingFile`, or `SolverIterations = All`).
Refusals exit 1 too, but print no VERDICT line. Exit 2 is for these alone:

- flags argparse rejects, and combinations the check refuses: `--certificate`
  without `--config`; `--verification` with `--certificate`, or without
  `--config` or `--script`; `--binding` or `--accept-end-time-binding-only`
  without a `--script` held by hand; `--script` with the format flags;
  `--report-file` without `--script`; `--cr-km` under rules without C_r;
- NumPy, SciPy or PyYAML off their pins;
- a `--config` it cannot load;
- a certificate it refuses: not one certify wrote (a field missing or of the
  wrong type, a number not finite), of another day (the rules or their
  revision, the time origin, T_max), priced under another score block, or
  listing other bytes than the report and the script given;
- a verification.json it refuses: not one verify wrote, of another script, or
  priced under another score block;
- a value given by hand that the config, the certificate or verification.json
  gives otherwise (step 7): only a value the config leaves unset may be given;
- a plan the rules disqualify (§4 note 1(a), (b) on a row our burns flew, (c)),
  with a `DISQUALIFIED:` line;
- by hand with `--script`, a report its script did not write (step 7): another
  official report name, an end off the Propagate goals, an alignment missing,
  cut or not the script's, a burn missing or not the script's impulse, a jump
  the script lacks, rows missing, a clock that goes back, the target
  maneuvered, a report that cannot show its burns without
  `--accept-end-time-binding-only`, or a `--total-dv` other than the script's.

The other exit statuses:

| Exit | Means | What to do |
|---|---|---|
| 0 | passes | upload (step 7) |
| 1 | a VERDICT that fails (last row outside, entry after T_max, a miss not at T_max, short of the stop depth with `--certificate`), or a report it cannot read one way only (no VERDICT line) | not uploaded: next fallback |
| 3 | passes, burns not bound (`--binding warn`, `--accept-end-time-binding-only`) | upload only after 6b.7's by-eye comparison matches |
| 4 | `SURFACE NOT JUDGED`: §4 note 1(b) cannot be judged on the report | read the pursuer's lowest altitude off GMAT by eye, or ask the organizer |
| 5 | `SURFACE: the template's initial state is below the surface` | ask the organizer |
| 6 | `SURFACE UNDECIDED`: a computed height within its error of 0 | read GMAT's `Earth.Altitude` by eye, or ask the organizer |
| 7 | `NOT JUDGED BY HAND`: the report carries no target position, and no `--certificate` was given | never a pass: run step 7's `--certificate` form on certify's report; a GMAT GUI's report of such a template cannot be passed (step 3) |

7 comes after the binding by hand passed, and overrides 3: its line is last.
4, 5 and 6 come only when every other verdict passed, and override 0 and 3:
the last line says which, and after a 3 the burns line is just above it.

Without `--time-column` the check takes the ElapsedSecs by name, else a
numeric epoch, never column 0 as such. It refuses when the two craft's
ElapsedSecs differ, as after an alignment; when a report with no ElapsedSecs
has two ModJulians of one kind that differ, as they do on an alignment's rows
until `--frozen-rows` skips them; or
when there is no time column at all. Otherwise it judges the rows. Score lines are printed only for an ElapsedSecs named
with `--time-column`: found by name, it may be the wrong craft's, and a Toggle
can keep an alignment's rows out of the report. Positions on alignment rows that `--frozen-rows` keeps
must be EarthMJ2000Eq: the craft are at two epochs there.

### E. The stop depth, shallow stops and the tail (step 7)

The depth is a reproducibility margin. When the organizer reruns the script,
their data files may differ from ours. Measured on R2026a: EOP/UT1 changes moved
the stop at most 0.11 m and EarthFixed insertion 25.5 m, so 50 m covered every
constant-flux case; R4's drag, reading the CSSI space-weather file with F10.7
and Ap 30 % high, moved SC_Human 665 m, and a 50 m stop ended at 5.25 km. The
default 1.5 km is twice that. Over a 50 m stop it costs at least
(1.5 - 0.05) km / closing speed of completion time, more for a pass that is not
head-on: 0.40 s on R1 at 3.6 km/s (4301.036 to 4301.438 s), about 0.3 s on R4
at 5.1 km/s, while a 13.2 m/s aligned crossing stopped 121.5 s after its entry.
`orbitwin inspect` names each drag
model's weather source and recommends 1.5 km when one reads a weather file;
the depth is never changed for you. Choose `policy.stop_depth_km` on the day,
up to half the radius, and keep `gmat.correct_aim_km` within the radius less it.

No flag of `ranking_day_config.py` sets `policy.stop_depth_km`,
`gmat.correct_aim_km`, `policy.allow_shallow_stop_submission` or
`policy.mission_end_policy`. To change one, copy `config/ranking.yaml` to
`run/base-2.yaml` and add that one line, indented two spaces, inside its
section: `stop_depth_km`, `allow_shallow_stop_submission` and
`mission_end_policy` under `policy:` (right after `  rules: ranking`),
`correct_aim_km` under `gmat:` (right after `gmat:`). A `policy:` key
appended at the end of the file lands in `gmat:`, the last section, and the
tool stops with "Unknown GmatConfig keys". Have the second person read
`diff config/ranking.yaml run/base-2.yaml`, and write a new day config from
it with `--base run/base-2.yaml` and the same flags as before. The readback
shows the depth ("where a hit ends") and the GMAT aim ("search aim"), and
neither of the other two.

A hit whose passage never gets that deep (a graze, or an entry just before
T_max) is stopped at its deepest point, and that is the hit the margin exists
for: a few metres of drift turn it into a miss. solve passes it over for any
verified hit that reached the depth, whatever the scores, and names it only
when none did, with `best_candidate_shallow_stop` in summary.json, a warning
saying so, and `best_certifiable_without_shallow_opt_in` naming the best
candidate certify takes as things stand. A stop measured under an older, smaller
`policy.stop_depth_km` says nothing about how deep the passage goes; rank
treats such a hit as neither deep nor shallow, and certify measures it again. rank lists it as not submittable, and certify and this step refuse
it, until `policy.allow_shallow_stop_submission` is set; certification.json
then records it under `shallow_stop`. A person decides that, knowing a
miss scores on its closest approach.

Nothing about the tail is predicted any more. The analytical estimate (a row
every `Propagator.MaxStep` from the last maneuver) only decides how far the
first measuring flight goes, and `policy.interception_tail_s` is used only
where GMAT has not measured the stop. A predicted tail failed when
InitialStepSize differed from MaxStep: on R2026a it ended a hit with its last
row 5.88 km out.
