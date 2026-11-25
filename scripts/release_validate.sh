#!/usr/bin/env bash
set -euo pipefail
# `make validate` passes its $(PYTHON). A path is made absolute before the cd.
PYTHON="${PYTHON:-python}"
case "$PYTHON" in
  */*)
    # Its own statement: inside the assignment below the last $(...) sets the status.
    if ! python_dir="$(cd "$(dirname "$PYTHON")" 2>/dev/null && pwd)"; then
      echo "error: PYTHON=$PYTHON: no directory $(dirname "$PYTHON"); nothing was changed" >&2
      exit 2
    fi
    PYTHON="$python_dir/$(basename "$PYTHON")"
    ;;
esac

# Extra arguments for the coverage run only, e.g. "-n 16" with pytest-xdist.
# xdist's options and --durations only: -k, -m, a path or -x would shrink the
# evidence and still end in PASS, and -p no:cov would drop it. Values cannot
# hold spaces, so quoting is refused rather than split.
# PYTEST_ADDOPTS and PYTEST_PLUGINS reach every pytest here unscreened, so
# they are refused outright.
for variable in PYTEST_ADDOPTS PYTEST_PLUGINS; do
  if [ -n "${!variable:-}" ]; then
    echo "error: $variable is set (\"${!variable}\"); it would reach every pytest this" \
      "runs, selection included. Unset it. Nothing was changed." >&2
    exit 2
  fi
done
read -r -a pytest_args <<< "${PYTEST_ARGS:-}"
index=0
while [ "$index" -lt "${#pytest_args[@]}" ]; do
  option="${pytest_args[$index]}"
  case "$option" in
    -n|--numprocesses|--maxprocesses|--dist|--maxschedchunk|--durations)
      value="${pytest_args[$((index + 1))]:-}"
      index=$((index + 2))
      ;;
    -n*|--numprocesses=*|--maxprocesses=*|--dist=*|--maxschedchunk=*|--durations=*)
      value="${option#-n}"; value="${value#*=}"
      index=$((index + 1))
      ;;
    *)
      value=""
      ;;
  esac
  if ! [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9_.:-]*$ ]]; then
    echo "error: PYTEST_ARGS=\"${PYTEST_ARGS:-}\": only -n, --numprocesses, --maxprocesses," \
      "--dist, --maxschedchunk and --durations with a plain value are taken, so the coverage" \
      "run keeps the whole suite; refused at \"$option\". Nothing was changed." >&2
    exit 2
  fi
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"

# Everything after this check deletes or rewrites the evidence, so the
# interpreter is checked first: a wrong one stops here with the tree untouched.
dependency_check="$(mktemp)"
trap 'rm -f "$dependency_check"' EXIT
"$PYTHON" - > "$dependency_check" <<'PYDEP'
import importlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

# Imported, not just looked up: metadata can outlive a broken install.
for module in ('numpy', 'scipy', 'yaml', 'pytest', 'pytest_cov', 'packaging', 'orbitwin'):
    try:
        importlib.import_module(module)
    except ImportError as error:
        raise SystemExit(
            f'{sys.executable} cannot import {module} ({error}). Nothing was changed; '
            'run with PYTHON=<the project venv python>.'
        ) from None
from packaging.specifiers import SpecifierSet

# Read the pins rather than repeating them. A hand-copied '>=1.26' here would
# keep passing while the machine ran something else entirely, which is the exact
# failure this check exists to catch.
requirements = {}
for source in ('requirements.txt', 'requirements-dev.txt'):
    for line in Path(source).read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if not line or line.startswith('-'):
            continue
        for operator in ('==', '>=', '~=', '<='):
            if operator in line:
                name, _, spec = line.partition(operator)
                requirements[name.strip()] = operator + spec.strip()
                break
assert requirements, 'no pins parsed from requirements files'
rows = {}
failed = []
for name, spec in requirements.items():
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = None
    ok = version is not None and version in SpecifierSet(spec)
    rows[name] = {'installed': version, 'required': spec, 'satisfied': ok}
    if not ok:
        failed.append(name)
print(json.dumps({'status': 'PASS' if not failed else 'FAIL', 'packages': rows},
                 indent=2, sort_keys=True))
if failed:
    raise SystemExit(
        f'{sys.executable}: unsatisfied project dependencies: {", ".join(failed)}. '
        'Nothing was changed.'
    )
PYDEP
# Arguments pytest would refuse (-n without xdist) fail here, not after the rm.
if [ "${#pytest_args[@]}" -gt 0 ]; then
  "$PYTHON" -m pytest -p no:cacheprovider --collect-only -q "${pytest_args[@]}" \
    tests/unit/test_make_targets.py > /dev/null
fi

# Never tracked by git, so no checkout writes over it or deletes it.
EVIDENCE=output/validation
export EVIDENCE
# A run that fails leaves $EVIDENCE as it found it: the evidence is copied
# aside first and put back on any failure, Ctrl-C included, with what the run
# wrote moved to output/validate/failed-<time>.<ns>-<pid>. A copy left by a run
# that was killed outright (SIGKILL) stops the next one.
snapshot=output/validate/previous
mkdir -p output/validate
# One run per checkout, for the whole run: a second one would take the first
# one's copy for its own, and the two restores lose the evidence.
if ! command -v flock > /dev/null; then
  echo "error: flock (util-linux) is not on PATH; nothing was changed" >&2
  exit 2
fi
exec 9> output/validate/lock
if ! flock -n 9; then
  echo "error: another make validate is running in this checkout (output/validate/lock);" \
    "nothing was changed" >&2
  exit 2
fi
if [ -e "$snapshot" ]; then
  echo "error: $snapshot is left from an interrupted run and holds $EVIDENCE/ from" \
    "before it. Compare it with $EVIDENCE/, restore what you need, then remove it." >&2
  exit 2
fi
# cp -a would copy the link and rm would delete through it.
if [ -L "$EVIDENCE" ]; then
  echo "error: $EVIDENCE is a symbolic link; make it a directory. Nothing was changed." >&2
  exit 2
fi
mkdir -p "$EVIDENCE"
# Nothing it would have to rewrite may be read-only: rm stopped part-way there.
while IFS= read -r -d '' entry; do
  if [ ! -w "$entry" ]; then
    echo "error: $entry is not writable; nothing was changed" >&2
    exit 2
  fi
done < <(find "$EVIDENCE" -print0)
# Runs once, from whichever comes first. A failing step and a signal can both
# be pending when the step ends: the EXIT trap then starts and the signal's
# trap fires inside it, and an `exit` there cut the restore short with no
# message (seen under load). So a signal's trap does the restore itself, and
# one that fires while a restore is under way returns to it.
#
# phase says what a stop has to undo. copying: the evidence is untouched, and
# any copy is this run's, so it goes. changing: the copy is whole and
# the evidence may differ from it, so the copy is put back.
phase=""
finishing=""
finish() {
  status=$1
  # A second Ctrl-C must not stop the restore half-way.
  trap '' INT TERM HUP
  rm -f "$dependency_check"
  if [ "$status" -eq 0 ]; then
    rm -rf "$snapshot"
    return
  fi
  case "$phase" in
    copying)
      rm -rf "$snapshot.partial" "$snapshot"
      echo "release validation: FAILED (exit $status) before $EVIDENCE/ was changed;" \
        "it is untouched." >&2
      return
      ;;
    changing) ;;
    *) return ;;
  esac
  # Unique, and -T: an existing name fails here rather than nesting into it.
  failed="output/validate/failed-$(date +%Y%m%dT%H%M%S.%N)-$$"
  if mv -T "$EVIDENCE" "$failed" && mv -T "$snapshot" "$EVIDENCE"; then
    echo "release validation: FAILED (exit $status). $EVIDENCE/ is restored;" \
      "this run's output is in $failed." >&2
  else
    echo "release validation: FAILED (exit $status), and $EVIDENCE/ could not be" \
      "restored: the evidence from before the run is in $snapshot." >&2
  fi
}
on_exit() {
  status=$?
  if [ -z "$finishing" ]; then
    finishing=1
    finish "$status"
  fi
}
on_signal() {
  if [ -z "$finishing" ]; then
    finishing=1
    finish "$1"
    exit "$1"
  fi
}
trap on_exit EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP

# Copied under another name and renamed when whole: a copy cut short by a kill
# or a full disk is never taken for the evidence. One left that way by a run
# killed outright was made before anything changed, so it is only removed.
phase=copying
rm -rf "$snapshot.partial"
cp -a "$EVIDENCE" "$snapshot.partial"
mv -T "$snapshot.partial" "$snapshot"
phase=changing

rm -rf "$EVIDENCE/fake_baseline" "$EVIDENCE/fake_solve" "$EVIDENCE/fake_certification"
# What release_verdict.py reads, so it can only read this run's.
rm -f "$EVIDENCE/ood_fuzz_5000.json" "$EVIDENCE/scenario_ood_matrix_20.json" \
  "$EVIDENCE/pip_check.exit_code" "$EVIDENCE/pip_check.log"
cp "$dependency_check" "$EVIDENCE"/project_dependency_check.json

"$PYTHON" - <<'PY' > "$EVIDENCE/environment.json"
import importlib.util, json, os, platform, subprocess, sys
import numpy, scipy, yaml

def git(*arguments):
    try:
        done = subprocess.run(["git", *arguments], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    return done.stdout.strip() if done.returncode == 0 else None

# The commit the evidence is for: make package builds HEAD, which may have moved since.
# Only this tree's own repository: git looks upward from a tree without .git.
top = git("rev-parse", "--show-toplevel")
own = top is not None and os.path.realpath(top) == os.path.realpath(os.getcwd())
dirty = git("status", "--porcelain", "--untracked-files=no") if own else None
print(json.dumps({
  "commit": git("rev-parse", "--verify", "-q", "HEAD") if own else None,
  "tracked_changes": None if dirty is None else bool(dirty),
  "python": sys.version,
  "platform": platform.platform(),
  "machine": platform.machine(),
  "numpy": numpy.__version__,
  "scipy": scipy.__version__,
  "pyyaml": yaml.__version__,
  "tools": {
    "ruff": bool(importlib.util.find_spec("ruff")),
    "mypy": bool(importlib.util.find_spec("mypy")),
  },
}, indent=2, sort_keys=True))
PY

"$PYTHON" -m compileall -q src tests scripts
"$PYTHON" - <<'PY' > "$EVIDENCE"/ast_parse.log
import ast
from pathlib import Path
files = sorted([*Path('src').rglob('*.py'), *Path('tests').rglob('*.py'), *Path('scripts').glob('*.py')])
for path in files:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print(f'AST parse: PASS ({len(files)} Python files)')
PY

# Run the expensive OOD tests first. In this hosted build image, running them
# immediately after the full fast suite can trigger severe CPU throttling.
"$PYTHON" -m pytest -m slow | tee "$EVIDENCE"/pytest_slow.log
# The release gate runs each file in its own process. `make test` stays the
# quick one; here the wall clock is not the point.
"$PYTHON" scripts/run_isolated_test_files.py --log "$EVIDENCE"/pytest_isolated.log \
  | tee "$EVIDENCE"/pytest_isolated_summary.log
"$PYTHON" scripts/ood_fuzz.py --cases 5000 --seed 20260818 --output "$EVIDENCE"/ood_fuzz_5000.json > "$EVIDENCE"/ood_fuzz_5000.log
"$PYTHON" scripts/scenario_ood_matrix.py --cases 20 --seed 20260818 --output "$EVIDENCE"/scenario_ood_matrix_20.json > "$EVIDENCE"/scenario_ood_matrix_20.log
"$PYTHON" -m pytest -m 'not slow and not gmat' --cov --cov-report=term-missing \
  --cov-report=xml:"$EVIDENCE"/coverage.xml ${pytest_args[@]+"${pytest_args[@]}"} | tee "$EVIDENCE"/coverage.log

PYTHON="$PYTHON" bash scripts/dependency_check.sh "$EVIDENCE"

# Run optional static tools when present. A disconnected build environment may
# not be able to install them; the skip is recorded and never presented as PASS.
"$PYTHON" - <<'PY'
import importlib.util, json, os, subprocess, sys
from pathlib import Path
results = {}
commands = {
    'ruff': [sys.executable, '-m', 'ruff', 'check', '--config', 'pyproject.toml', 'src', 'tests', 'scripts'],
    'mypy': [sys.executable, '-m', 'mypy', 'src/orbitwin'],
}
for name, command in commands.items():
    if importlib.util.find_spec(name) is None:
        results[name] = {'status': 'SKIPPED', 'reason': f'{name} is not installed in the build environment'}
        Path(os.environ['EVIDENCE'], f'{name}.log').write_text(results[name]['reason'] + '\n', encoding='utf-8')
        continue
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    Path(os.environ['EVIDENCE'], f'{name}.log').write_text(completed.stdout + '\n--- STDERR ---\n' + completed.stderr, encoding='utf-8')
    results[name] = {'status': 'PASS' if completed.returncode == 0 else 'FAIL', 'return_code': completed.returncode}
    if completed.returncode != 0:
        raise SystemExit(f'{name} failed')
Path(os.environ['EVIDENCE'], 'static_tools.json').write_text(json.dumps(results, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

# Exercise exact CLI orchestration paths with the deterministic test double.
PYTHON="$PYTHON" bash scripts/fake_cli_evidence.sh "$EVIDENCE"

"$PYTHON" scripts/release_verdict.py "$EVIDENCE"
