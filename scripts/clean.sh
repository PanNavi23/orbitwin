#!/bin/sh
# `make clean`: build_release.clean() without Python, for a checkout whose
# interpreter is broken. tests/unit/test_make_targets.py keeps the two equal.
#
#   sh scripts/clean.sh [project-root]
#
# Without an argument the root is this file's real location, links resolved:
# through a symlink elsewhere, on stdin or sourced, $0 names something else.
set -eu
if [ "$#" -gt 0 ]; then
  root=$1
else
  root=$(dirname -- "$(readlink -f -- "$0")")/..
fi
cd -- "$root"
if [ ! -f pyproject.toml ] || [ ! -d src/orbitwin ]; then
  echo "clean.sh: $(pwd) is not an OrbitWin checkout; nothing removed" >&2
  exit 2
fi
# Links are removed, never followed; a matching name goes whatever it is.
rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov src/*.egg-info .coverage .coverage.*
# Not searched: _NOT_WALKED in build_release.py, at any depth, and another
# checkout, a directory holding .git.
find . -mindepth 1 \( -name .git -o -name .venv -o -name output \
  -o -name dist -o -name build \) -prune \
  -o -name __pycache__ -prune -exec rm -rf {} + \
  -o -type d -exec sh -c '[ -e "$1/.git" ] || [ -L "$1/.git" ]' sh {} \; -prune \
  -o ! -type d \( -name '*.pyc' -o -name '*.pyo' \) -exec rm -f {} +
