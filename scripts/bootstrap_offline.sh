#!/usr/bin/env bash
# Install the toolchain on a machine with no network.
#
# scripts/bootstrap_linux.sh reaches PyPI, which is not available on competition
# day and which nothing on the day may depend on. This installs only from
# vendor/wheels/, which scripts/fetch_offline_wheels.sh fills beforehand.
#
# Nothing here upgrades pip: that would need the network. The interpreter's
# bundled pip is enough to install wheels from a directory.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
WHEELS="${WHEELS_DIR:-vendor/wheels}"
VENV="${VENV_DIR:-.venv}"
# The versions the bundle was fetched at; with them pip cannot pick a stray
# newer wheel that happens to sit in the directory.
CONSTRAINTS=vendor/wheels/constraints.txt

if [ ! -d "$WHEELS" ] || [ -z "$(find "$WHEELS" -name '*.whl' -print -quit)" ]; then
  echo "error: no wheels in ${WHEELS}." >&2
  echo "Run scripts/fetch_offline_wheels.sh on a machine with internet first," >&2
  echo "and carry ${WHEELS}/ to this machine." >&2
  exit 1
fi

# Only the wheels the manifest lists, each with its hash. pip installed a
# tampered wheel when the manifest was missing, and an unlisted one beside a
# listed wheel when its build tag made it the newer.
echo "==> verifying wheel bundle"
"$PYTHON" -I scripts/wheel_requirements.py "$WHEELS" "$CONSTRAINTS" requirements-dev.txt
if ! ( cd "$WHEELS" && sha256sum --check --strict --quiet SHA256SUMS.txt ); then
  echo "error: a wheel in ${WHEELS} is not the one SHA256SUMS.txt lists." >&2
  echo "Bring ${WHEELS}/ again from the release archive, unchanged (docs/RUNBOOK.md step 0)." >&2
  exit 1
fi

echo "==> $("$PYTHON" -V)"
# An interpreter the bundle has no wheels for fails the install below with
# "No matching distribution found for numpy", which reads like a broken
# bundle and sends the operator to the runbook's stop rule.
# The last line: a sitecustomize that prints would otherwise be read as the tag.
tag="$("$PYTHON" -c 'import sys; print("cp%d%d" % sys.version_info[:2])' | tail -n 1)"
if [ -z "$(find "$WHEELS" -name "numpy-*-${tag}-*.whl" -print -quit)" ]; then
  echo "error: $("$PYTHON" -V) has no wheels in ${WHEELS} (no ${tag} numpy)." >&2
  echo "Name an interpreter the bundle covers: PYTHON=python3.11 bash $0" >&2
  exit 1
fi
# The version alone let a free-threaded 3.14t, and a machine other than
# x86-64, through to that same failure, after an existing venv was cleared.
# So the bundle's own pip resolves the install below as a dry run, offline,
# before anything is touched. Into an empty scratch --target: into the
# interpreter itself, PEP 668 refuses it (Debian's, uv's), dry run or not.
# requirements-dev.txt names the constraints too; naming them here does not depend on it.
# Wheels only: a dry run still runs an sdist's setup.py to read its metadata.
bundled_pip="$(find "$WHEELS" -maxdepth 1 -name 'pip-*-py3-none-any.whl' -print -quit)"
if [ -z "$bundled_pip" ]; then
  echo "==> no pip wheel in ${WHEELS}: checked the Python version only"
elif ! resolved="$(target="$(mktemp -d)" && trap 'rm -rf "$target"' EXIT &&
    PIP_REQUIRE_VIRTUALENV=0 "$PYTHON" -I "$bundled_pip/pip" install --dry-run --target "$target" \
    --no-index --find-links "$WHEELS" --only-binary :all: --requirement requirements-dev.txt \
    --constraint "$CONSTRAINTS" --no-cache-dir --disable-pip-version-check --quiet \
    setuptools wheel 2>&1)"; then
  # Why, in the terms of the wheels: this interpreter's most specific tag, and
  # the pinned projects with no wheel for any tag it takes.
  read_tags='
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import (
    InvalidWheelFilename, canonicalize_name, parse_wheel_filename)
supported = list(sys_tags())
lines = [line.split("#")[0] for line in Path(sys.argv[3]).read_text().splitlines()]
covered = {canonicalize_name(line.split("==")[0].strip()): False for line in lines if "==" in line}
for wheel in Path(sys.argv[2]).glob("*.whl"):
    try:
        project, _, _, tags = parse_wheel_filename(wheel.name)
    except InvalidWheelFilename:
        continue
    if project in covered and not tags.isdisjoint(supported):
        covered[project] = True
print(supported[0], *sorted(p for p, ok in covered.items() if not ok))
'
  found="$("$PYTHON" -I -c "$read_tags" "$bundled_pip" "$WHEELS" "$CONSTRAINTS" 2>/dev/null \
    | tail -n 1 || true)"
  read -r best uncovered <<<"$found" || true
  # Refused only for wheels this interpreter cannot take. A dry run outside
  # the venv can also fail on the shell's own settings (a TMPDIR that is
  # gone); the install below decides those, as it did before.
  if [ -z "$uncovered" ]; then
    echo "==> warning: the dry run failed, not for want of wheels; the install decides:"
    printf '%s\n' "$resolved" | tail -n 3
  else
    echo "error: $("$PYTHON" -V) installs ${best} wheels; ${WHEELS} has none for: ${uncovered}." >&2
    echo "Name an interpreter the bundle covers: PYTHON=python3.11 bash $0" >&2
    echo "On a machine the bundle was not fetched for, stop (docs/RUNBOOK.md step 0)." >&2
    exit 1
  fi
fi
# Debian and Ubuntu ship ensurepip as python3.X-venv, which needs the
# network. Without it the venv is made without pip, and pip installs itself
# from its own wheel in the bundle; with neither, refuse before touching
# anything.
pip_wheel=""
if ! "$PYTHON" -c 'import ensurepip' 2>/dev/null; then
  pip_wheel="$(find "$WHEELS" -maxdepth 1 -name 'pip-*-py3-none-any.whl' -print -quit)"
  if [ -z "$pip_wheel" ]; then
    echo "error: $("$PYTHON" -V) has no ensurepip, and ${WHEELS} holds no pip wheel." >&2
    echo "Name an interpreter that has ensurepip: PYTHON=python3.X bash $0" >&2
    exit 1
  fi
fi
# A venv left by a failed run, or carried from another machine, keeps
# pointing at that interpreter, and a retry failed the same way: remake it
# when its Python is another, and only then, since a solve running in another
# terminal imports from it. Anything there that is not a venv is refused.
# Which interpreter a venv runs is asked of the venv itself: pyvenv.cfg's
# version line is "3.11" from uv, "3.11.15" from CPython, and says nothing
# about an interpreter that has since gone.
identify='import os, sys; print("%d.%d.%d" % sys.version_info[:3], os.path.realpath(getattr(sys, "_base_executable", sys.executable)))'
venv_args=()
if [ -e "$VENV" ]; then
  if ! grep -qs '^home *=' "$VENV/pyvenv.cfg"; then
    echo "error: ${VENV} exists and is not a virtual environment; move it away first." >&2
    exit 1
  fi
  # The last line: a sitecustomize that prints would otherwise read as another Python.
  want="$("$PYTHON" -c "$identify" | tail -n 1)"
  have="$("$VENV/bin/python" -c "$identify" 2>/dev/null | tail -n 1 || true)"
  if [ "$have" != "$want" ]; then
    echo "==> ${VENV} runs ${have:-no working interpreter}; making it again for ${want}"
    venv_args+=(--clear)
  fi
fi
if [ -z "$pip_wheel" ]; then
  "$PYTHON" -m venv ${venv_args[@]+"${venv_args[@]}"} "$VENV"
else
  "$PYTHON" -m venv ${venv_args[@]+"${venv_args[@]}"} --without-pip "$VENV"
fi
# Every pin in the constraints, with the hashes of its listed wheels, so pip
# itself refuses any other file it finds in the bundle.
"$PYTHON" -I scripts/wheel_requirements.py "$WHEELS" "$CONSTRAINTS" requirements-dev.txt \
  --write "$VENV"
hashed=(--no-index --find-links "$WHEELS" --only-binary :all: --require-hashes)
if [ -n "$pip_wheel" ]; then
  echo "==> no ensurepip in $("$PYTHON" -V): installing pip from ${pip_wheel}"
  "$VENV/bin/python" "$pip_wheel/pip" install --quiet "${hashed[@]}" \
    --requirement "$VENV/bundle-pip.txt"
fi

# --no-index is the whole point: if a wheel is missing, fail here and loudly,
# rather than quietly reaching for a network that is not supposed to be used.
# setuptools and wheel come with the rest: Python 3.12 dropped setuptools from
# ensurepip, and the editable install below needs a build backend.
"$VENV/bin/python" -m pip install --quiet "${hashed[@]}" \
  --requirement "$VENV/bundle-requirements.txt"

# --no-build-isolation so the backend comes from the venv we just populated
# rather than from a fresh download.
"$VENV/bin/python" -m pip install --quiet --no-index --find-links "$WHEELS" \
  --no-deps --no-build-isolation --editable .

echo "==> installed"
"$VENV/bin/python" - <<'PY'
import numpy, scipy, yaml
print(f"  numpy {numpy.__version__}  scipy {scipy.__version__}  PyYAML {yaml.__version__}")
PY
"$VENV/bin/orbitwin" --version

echo
# The runbook's later steps call python and orbitwin by name.
echo "Next, in this and every terminal used for the runbook:"
# Single quotes, which dash reads too; printf %q writes $'...' for non-ASCII
# paths in the C locale, which dash does not.
quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
echo "  cd $(quote "$(pwd)") && . $(quote "$VENV")/bin/activate"
echo "and once, here:"
echo "  PYTHONPATH=src python -m pytest -m 'not slow and not gmat'"
