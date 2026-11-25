"""The search and audit tools read a script only GMAT can read once given a console.

The script writes one start instant in two DateFormats: the in-process reader
refuses it and the state probe (the fake console here) accepts it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from orbitwin.errors import ParseError

RANKING_YAML = """\
policy:
  rules: ranking
  horizon_s: 20000.0
  dv_limit_km_s: 0.25
  dv_safe_limit_km_s: 0.249
  total_dv_limit_km_s: 3.0
  total_dv_safe_limit_km_s: 2.99
score: {kt: 0.001, ct: 5000.0, kv: 0.001, cv: 1500.0, time_unit: s, delta_v_unit: m/s,
        cr_km: 100.0}
"""


class Parsed(Exception):
    """Raised by the first thing each tool does after reading the script."""


def _load(project_root: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"orbitwin_tool_{name}", project_root / "scripts" / f"{name}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Stop:
    """Stands in for a module, a class or a function; any use of it raises Parsed."""

    def __getattr__(self, _name: str) -> Any:
        raise Parsed

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise Parsed


def _two_formats(workshop_script: Path, tmp_path: Path) -> Path:
    text = workshop_script.read_text(encoding="utf-8")
    edited = text.replace(
        "AlienSC.DateFormat = UTCGregorian;\nAlienSC.Epoch = '15 Mar 2027 12:00:00.000';",
        "AlienSC.DateFormat = UTCModJulian;\nAlienSC.Epoch = '31480.0';",
    )
    assert edited != text
    path = tmp_path / "formats.script"
    path.write_text(edited, encoding="utf-8")
    return path


#: tool, its arguments after the script, and what it calls once it has the scenario.
TOOLS = [
    ("hunt_multi_burn", ["--config", "{config}"], "MultiBurnLimits"),
    ("audit_burn_table", ["{tmp}/burns.json", "--config", "{config}"], "json"),
    ("safe_delta_v_floor", ["--config", "{config}"], "_grid"),
    ("hunt_ranking_score", ["--config", "{config}"], "build_setup"),
]


@pytest.mark.parametrize(("name", "argv", "after"), TOOLS, ids=[tool[0] for tool in TOOLS])
def test_a_tool_reads_through_the_probe_when_given_a_console(
    project_root: Path,
    workshop_script: Path,
    fake_gmat: Path,
    fake_startup: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    argv: list[str],
    after: str,
) -> None:
    module = _load(project_root, name)
    monkeypatch.setattr(module, after, _Stop())
    config = tmp_path / "ranking.yaml"
    config.write_text(RANKING_YAML, encoding="utf-8")
    script = _two_formats(workshop_script, tmp_path)
    args = [arg.format(config=config, tmp=tmp_path) for arg in argv]

    monkeypatch.setattr(sys, "argv", [f"{name}.py", str(script), *args])
    with pytest.raises(ParseError, match="--gmat-console"):
        module.main()
    monkeypatch.setattr(
        sys, "argv",
        [f"{name}.py", str(script), *args, "--gmat-console", str(fake_gmat),
         "--startup-file", str(fake_startup)],
    )
    with pytest.raises(Parsed):
        module.main()
