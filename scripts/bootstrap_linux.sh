#!/usr/bin/env bash
# Development setup. This reaches PyPI, so it is not the competition-day path:
# use scripts/bootstrap_offline.sh there, and fetch the bundle beforehand with
# scripts/fetch_offline_wheels.sh.
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# The pinned set first, as CI and the offline bundle have it. There is no
# '.[dev]': pyproject's ranges resolve newer numpy and SciPy, which orbitwin
# refuses (src/orbitwin/pinned.py).
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
echo 'Set GMAT_CONSOLE and GMAT_STARTUP before running scripts/preflight.sh.'
