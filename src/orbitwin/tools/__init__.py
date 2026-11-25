"""The RUNBOOK's command-line tools: step 2's config, step 7's check, the audit, the search.

Each runs as `python -m orbitwin.tools.<name>`. In a checkout, scripts/<name>.py runs the
same code: it executes this package's file in its own namespace (`run_as_script`), so a
test that loads the script by path and patches it patches the code that runs.
"""

from __future__ import annotations

import importlib.util
from typing import Any


def run_as_script(name: str, namespace: dict[str, Any]) -> None:
    """Execute orbitwin.tools.<name>'s source in ``namespace``, a script's globals."""
    spec = importlib.util.find_spec(f"orbitwin.tools.{name}")
    if spec is None or spec.origin is None or spec.loader is None:
        raise ImportError(f"orbitwin.tools.{name} is not installed")
    with open(spec.origin, encoding="utf-8") as handle:
        source = handle.read()
    namespace.setdefault("__doc__", None)
    exec(compile(source, spec.origin, "exec"), namespace)
