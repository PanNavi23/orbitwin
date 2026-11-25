#!/usr/bin/env python3
"""Runs orbitwin.tools.audit_burn_table (src/orbitwin/tools/audit_burn_table.py), which an installed package carries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitwin.tools import run_as_script

run_as_script("audit_burn_table", globals())
