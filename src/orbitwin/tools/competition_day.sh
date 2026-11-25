#!/usr/bin/env bash
set -euo pipefail
usage="Usage: competition_day.sh [--analysis-only] [--preliminary] OFFICIAL.script OUTPUT_DIR CONFIG (written by scripts/ranking_day_config.py)"
# --analysis-only, anywhere on the line, searches without GMAT apart from the day flow (below).
analysis_only=0
# --preliminary takes a config under the preliminary rules; without it only ranking is run.
preliminary=0
positional=()
for arg in "$@"; do
  case "$arg" in
    --analysis-only) analysis_only=1 ;;
    --preliminary) preliminary=1 ;;
    "") echo "An empty argument. $usage" >&2; exit 2 ;;
    -*) echo "Unknown option $arg. $usage" >&2; exit 2 ;;
    *) positional+=("$arg") ;;
  esac
done
# No default config: the old one ran the preliminary rulebook.
if [ "${#positional[@]}" -ne 3 ]; then
  echo "$usage" >&2
  exit 2
fi
SCRIPT="${positional[0]}"
OUTPUT="${positional[1]}"
CONFIG="${positional[2]}"
# In a checkout, its src/ (links followed); an installed package is found as installed.
here=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../.." && pwd)
if [ -z "${PYTHONPATH:-}" ] && [ "$(basename -- "$here")" = src ]; then
  export PYTHONPATH="$here"
fi
# The python on PATH must import the orbitwin this script belongs to, not another venv's.
imported=$(python -c 'import os, orbitwin; print(os.path.realpath(os.path.dirname(orbitwin.__file__)))' 2>/dev/null) || imported=""
if [ "$imported" != "$(readlink -f -- "$here/orbitwin")" ]; then
  echo "python on PATH imports orbitwin from '${imported:-nowhere}', not $here/orbitwin," \
    "which this script belongs to; activate that environment, or run it through orbitwin day." >&2
  exit 2
fi

# Before anything is run or removed: orbitwin's own refusal would come after run/latest went.
rules=$(python -c 'import sys
from orbitwin.config import load_config
from orbitwin.rules import interception_rules
try:
    print(interception_rules(load_config(sys.argv[1]).policy.rules).name)
except Exception as exc:
    sys.exit(f"{sys.argv[1]}: {exc}")' "$CONFIG") || exit 2
rules_flag=()
if [ "$rules" != ranking ] && [ "$preliminary" = 0 ]; then
  echo "$CONFIG is under the $rules rules (policy.rules defaults to preliminary), not ranking." \
    "The ranking day needs the day's config, written by scripts/ranking_day_config.py" \
    "(docs/RUNBOOK.md step 2); --preliminary runs the preliminary rules on purpose." >&2
  exit 2
fi
if [ "$preliminary" = 1 ]; then
  if [ "$rules" != preliminary ]; then
    echo "--preliminary, but $CONFIG is under the $rules rules; drop the flag." >&2
    exit 2
  fi
  rules_flag=(--rules preliminary)
fi

# solve flies its candidates in GMAT only with both set, and the state probe
# reads an EarthFixed insertion or a second epoch through them. Half the pair is
# refused, as orbitwin refuses --gmat-console without --startup-file.
console="${GMAT_CONSOLE:-}"
startup="${GMAT_STARTUP:-}"
if [ -n "$console$startup" ] && { [ -z "$console" ] || [ -z "$startup" ]; }; then
  echo "GMAT_CONSOLE and GMAT_STARTUP must be set together; GMAT_CONSOLE='$console'," \
    "GMAT_STARTUP='$startup'." >&2
  exit 2
fi
if [ "$analysis_only" = 1 ] && [ -n "$console" ]; then
  echo "--analysis-only runs no GMAT; unset GMAT_CONSOLE and GMAT_STARTUP, or drop it" \
    "for the day flow." >&2
  exit 2
fi
if [ "$analysis_only" = 0 ] && [ -z "$console" ]; then
  echo "GMAT_CONSOLE and GMAT_STARTUP are not set. Without them solve flies nothing in" \
    "GMAT, so nothing it writes has been flown as it will be scored, and certify may fly" \
    "any of it as a miss; export both (docs/RUNBOOK.md, before step 2). --analysis-only" \
    "searches without them, apart from the day flow, but not on a template whose craft" \
    "start on different epochs or are inserted EarthFixed: only GMAT reads those." >&2
  exit 2
fi

# solve refuses a directory that already holds a run, so that an older run's
# artifacts cannot be mistaken for this one's. That has to not stand in the way
# here: the documented flow is to look at the table, enter the announced score
# coefficients and run again, and it is run under time pressure on someone
# else's machine. So each attempt gets its own numbered directory, and nothing
# is overwritten or lost.
prefix=search
if [ "$analysis_only" = 1 ]; then
  prefix=analysis
fi
attempt=1
while [ -e "$OUTPUT/$prefix-$(printf '%02d' "$attempt")" ]; do
  attempt=$((attempt + 1))
done
SEARCH="$OUTPUT/$prefix-$(printf '%02d' "$attempt")"

# Kept away from run/latest and run/scenario.json, which the day flow reads.
if [ "$analysis_only" = 1 ]; then
  python -m orbitwin inspect "$SCRIPT" --config "$CONFIG" "${rules_flag[@]}" > /dev/null
  python -m orbitwin solve "$SCRIPT" --config "$CONFIG" "${rules_flag[@]}" --output "$SEARCH"
  echo
  echo "Analysis $attempt is in $SEARCH, without GMAT: not the day flow. Nothing in it"
  echo "is submittable, and $OUTPUT/latest, which rank and certify read, is unchanged."
  exit 0
fi

gmat=(--gmat-console "$console" --startup-file "$startup")
# rank and certify read run/latest. It is dropped before the solve and set only
# after it, so a failed run leaves no latest rather than an older search.
if [ -e "$OUTPUT/latest" ] && [ ! -L "$OUTPUT/latest" ]; then
  echo "$OUTPUT/latest is not a symlink; move it away first." >&2
  exit 1
fi
rm -f "$OUTPUT/latest"
python -m orbitwin inspect "$SCRIPT" --config "$CONFIG" "${rules_flag[@]}" --output "$OUTPUT/scenario.json" "${gmat[@]}"
python -m orbitwin solve "$SCRIPT" --config "$CONFIG" "${rules_flag[@]}" --output "$SEARCH" "${gmat[@]}"
ln -sfnT "$(basename "$SEARCH")" "$OUTPUT/latest"

echo
echo "Search $attempt is in $SEARCH; $OUTPUT/latest points at it."
scored=$(python -c 'import sys; from orbitwin.config import load_config; print(load_config(sys.argv[1]).score is not None)' "$CONFIG")
if [ "$scored" = "True" ]; then
  echo "Rank and certify $OUTPUT/latest/candidates.json (docs/RUNBOOK.md steps 5-7)."
else
  echo "No coefficients in $CONFIG. Once C_r/kt/Ct/kv/Cv are announced, write a new config"
  echo "with scripts/ranking_day_config.py and run this again with it."
fi
