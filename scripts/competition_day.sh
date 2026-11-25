#!/usr/bin/env bash
# Runs src/orbitwin/tools/competition_day.sh, which an installed package carries.
exec bash "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../src/orbitwin/tools/competition_day.sh" "$@"
