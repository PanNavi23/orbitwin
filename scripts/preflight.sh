#!/usr/bin/env bash
set -euo pipefail
: "${GMAT_CONSOLE:?Set GMAT_CONSOLE to the organizer-compatible GmatConsole path}"
: "${GMAT_STARTUP:?Set GMAT_STARTUP to the organizer-compatible startup file path}"
# GmatConsole refuses scripts with OpenFrames plots, so the default has none.
SCRIPT="$(realpath "${1:-$(dirname "$0")/../examples/synthetic/DifferentPlane_Circular.script}")"
cd "$(dirname "$0")/.."
OUT="$PWD/output/preflight/$(date -u +%Y%m%dT%H%M%SZ)"
export PYTHONPATH="${PYTHONPATH:-src}"
mkdir -p "$OUT"
# Without --logfile, --version rewrites GmatLog.txt inside the install.
"$GMAT_CONSOLE" --logfile "$OUT/version-GmatLog.txt" --version
python -m orbitwin inspect "$SCRIPT" --config config/quick.yaml --rules preliminary >"$OUT/scenario.json"
python -m orbitwin baseline "$SCRIPT" --config config/quick.yaml --rules preliminary \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" \
  --output "$OUT/baseline"
python -m pytest -m 'not slow and not gmat'
echo "preflight: PASS ($OUT)"
