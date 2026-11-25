#!/usr/bin/env python3
"""Runs orbitwin.tools.ranking_day_config (src/orbitwin/tools/ranking_day_config.py), which an installed package carries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitwin.tools import run_as_script

run_as_script("ranking_day_config", globals())
