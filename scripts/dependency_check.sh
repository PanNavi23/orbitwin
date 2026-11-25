#!/usr/bin/env bash
# Check that the packages in $PYTHON's environment agree with each other, and
# record the result in <dir>/pip_check.log and <dir>/pip_check.exit_code.
#
# The bootstrap's venv has pip; a uv venv has none, and `python -m pip check`
# there only says pip is missing. So pip when the interpreter has it, else uv,
# else SKIPPED. The exit code file holds the checker's status or SKIPPED, and
# release_verdict.py fails the campaign on anything but 0 or SKIPPED.
#
#   PYTHON=.venv/bin/python bash scripts/dependency_check.sh [dir]
set -euo pipefail

PYTHON="${PYTHON:-python}"
OUT="${1:-output/validation}"
mkdir -p "$OUT"

if "$PYTHON" -c 'import pip' > /dev/null 2>&1; then
  checker=("$PYTHON" -m pip check)
elif command -v uv > /dev/null 2>&1; then
  checker=(uv pip check --python "$PYTHON")
else
  printf 'SKIPPED: %s has no pip and uv is not on PATH\n' "$PYTHON" > "$OUT/pip_check.log"
  printf 'SKIPPED\n' > "$OUT/pip_check.exit_code"
  exit 0
fi

set +e
{
  printf '$ %s\n' "${checker[*]}"
  "${checker[@]}"
} > "$OUT/pip_check.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$OUT/pip_check.exit_code"
