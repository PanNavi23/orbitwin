#!/usr/bin/env python3
"""Runs orbitwin.tools.check_report_rows (src/orbitwin/tools/check_report_rows.py), which an installed package carries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitwin.tools import run_as_script

run_as_script("check_report_rows", globals())
