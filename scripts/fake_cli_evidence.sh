#!/usr/bin/env bash
# Baseline, solve, verify-policy and certify through the CLI against the
# deterministic fake GmatConsole. release_validate.sh runs it into output/validation/;
# tests run it into a temporary directory. Never scientific validation.
#
#   PYTHON=.venv/bin/python bash scripts/fake_cli_evidence.sh [output-dir]
set -euo pipefail

PYTHON="${PYTHON:-python}"
case "$PYTHON" in
  */*)
    if ! python_dir="$(cd "$(dirname "$PYTHON")" 2>/dev/null && pwd)"; then
      echo "error: PYTHON=$PYTHON: no directory $(dirname "$PYTHON")" >&2
      exit 2
    fi
    PYTHON="$python_dir/$(basename "$PYTHON")"
    ;;
esac
# Relative to the project root, so the evidence records repo-relative paths.
OUT="${1:-output/validation}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/src"

SCRIPT=examples/two_burn_rendezvous.script
# solve takes the console too: certify reads the initial state through it, and
# 4 T_A from that state differs from the in-process one by 42 us, which
# certify refuses as another horizon.
FAKE=(--gmat-console scripts/fake_gmat_console.py --startup-file tests/fixtures/gmat/fake_startup.txt)

"$PYTHON" -m orbitwin baseline "$SCRIPT" --config config/quick.yaml --rules preliminary "${FAKE[@]}" \
  --output "$OUT/fake_baseline" > "$OUT/fake_baseline_cli.log"
"$PYTHON" -m orbitwin solve "$SCRIPT" --config config/quick.yaml --rules preliminary "${FAKE[@]}" \
  --output "$OUT/fake_solve" --render-top 3 > "$OUT/fake_solve_cli.log"
"$PYTHON" - "$OUT" <<'PY' > "$OUT/fake_candidate_id.txt"
import json, sys
from pathlib import Path
items = json.loads((Path(sys.argv[1]) / 'fake_solve' / 'candidates.json').read_text())
print(next(item for item in items if item['successful'])['candidate_id'])
PY
CANDIDATE_ID="$(tr -d '\r\n' < "$OUT/fake_candidate_id.txt")"
# The filename follows the stage the candidate actually reached, so it is read
# back from summary.json rather than written here. A literal
# `team_solution.script` was correct only before CandidateStage existed; solve
# has since been unable to produce that name at all, and the hardcoded path
# took this gate down instead of reporting anything.
"$PYTHON" - "$OUT" <<'PY' > "$OUT/fake_best_script.txt"
import json, sys
from pathlib import Path
print(json.loads((Path(sys.argv[1]) / 'fake_solve' / 'summary.json').read_text())['best_candidate_script'])
PY
BEST_SCRIPT="$(tr -d '\r\n' < "$OUT/fake_best_script.txt")"
"$PYTHON" -m orbitwin verify-policy "$SCRIPT" "$OUT/fake_solve/best_candidate/${BEST_SCRIPT}" \
  --config config/quick.yaml --rules preliminary > "$OUT/fake_policy_cli.log"
"$PYTHON" -m orbitwin certify "$SCRIPT" --config config/quick.yaml --rules preliminary \
  --candidates "$OUT/fake_solve/candidates.json" --candidate "$CANDIDATE_ID" \
  "${FAKE[@]}" --runs 3 \
  --output "$OUT/fake_certification" > "$OUT/fake_certification_cli.log"
