"""The public policy docs' key facts, held to the code they describe.

Each expectation is built from the code (config, rules, parsers, constants)
and looked up in the doc, so a change on either side turns the test red.
"""

from __future__ import annotations

import functools
import io
import json
import os
import re
import subprocess
import sys
import tokenize
import tomllib
from collections import Counter
from pathlib import Path

import pytest
from conftest import ignored_files, repo_files
from test_make_targets import _recipe

from orbitwin.config import OrbitWinConfig, load_config
from orbitwin.rules import PRELIMINARY, RANKING

ROOT = Path(__file__).resolve().parents[2]


def _text(name: str) -> str:
    """The doc with runs of whitespace collapsed, so line wrapping does not matter."""
    return " ".join((ROOT / name).read_text(encoding="utf-8").split())


POLICY = _text("docs/COMPETITION_POLICY.md")
PRE = OrbitWinConfig().policy
RANKING_CONFIG = load_config(ROOT / "config" / "ranking.yaml")
RANK = RANKING_CONFIG.policy
#: Our team's number, which the docs use where they record what we did; no profile sets it.
OUR_TEAM = 54


def _m_s(km_s: float) -> str:
    return f"{km_s * 1000.0:g}"


def test_margins_in_competition_policy_are_the_configs() -> None:
    assert (
        f"use {PRE.dv_safe_limit_km_s} km/s per impulse against the preliminary's "
        f"{PRE.dv_limit_km_s}, and {PRE.safe_burn_separation_s:g} seconds against "
        f"{PRE.min_burn_separation_s:g}."
    ) in POLICY
    assert (
        f"use {RANK.dv_safe_limit_km_s} km/s per impulse against {RANK.dv_limit_km_s}, "
        f"{RANK.total_dv_safe_limit_km_s} km/s in total against {RANK.total_dv_limit_km_s}, "
        f"and {RANK.safe_burn_separation_s:g} seconds against {RANK.min_burn_separation_s:g}."
    ) in POLICY


def test_rule_limits_in_competition_policy_are_the_rulebooks() -> None:
    assert (
        f"**{_m_s(PRELIMINARY.impulse_limit_km_s)} m/s = {PRELIMINARY.impulse_limit_km_s:g} km/s**"
    ) in POLICY
    assert (f"at most **{_m_s(RANKING.impulse_limit_km_s)} m/s** under the ranking rules"
            in POLICY)
    assert f"Going over the {_m_s(RANKING.impulse_limit_km_s)} m/s plus" in POLICY
    assert RANKING.total_dv_limit_km_s is not None
    assert f"cap it at {RANKING.total_dv_limit_km_s:g} km/s" in POLICY
    assert PRELIMINARY.min_separation_s == RANKING.min_separation_s
    assert f"at least **{RANKING.min_separation_s:g} seconds**" in POLICY
    # The ranking profile sits on the rulebook's limits, as the doc says.
    assert RANK.dv_limit_km_s == RANKING.impulse_limit_km_s
    assert RANK.total_dv_limit_km_s == RANKING.total_dv_limit_km_s
    assert RANK.min_burn_separation_s == RANKING.min_separation_s


def test_correction_count_in_competition_policy_is_the_profiles() -> None:
    counts = {load_config(ROOT / "config" / name).gmat.correct_top
              for name in ("ranking.yaml", "ranking-rehearsal.yaml")}
    assert len(counts) == 1
    assert f"{counts.pop()} in the ranking profiles" in POLICY


#: A tool the docs name: the script or module that builds its parser, and the
#: subcommand whose options it means ("" for the top-level parser).
TOOLS: dict[str, tuple[str, str, str]] = {
    "check_report_rows": ("path", "scripts/check_report_rows.py", ""),
    "ranking_day_config": ("path", "scripts/ranking_day_config.py", ""),
    "audit_burn_table": ("path", "scripts/audit_burn_table.py", ""),
    "ood_fuzz": ("path", "scripts/ood_fuzz.py", ""),
    "hunt_multi_burn": ("path", "scripts/hunt_multi_burn.py", ""),
    "certify": ("module", "orbitwin", "certify"),
    "orbitwin solve": ("module", "orbitwin", "solve"),
    "orbitwin rank": ("module", "orbitwin", "rank"),
    "orbitwin day": ("module", "orbitwin", "day"),
}

#: Runs a tool up to its parse_args and prints every parser's option strings and
#: choices, the top-level one under "" and each subcommand under its name.
_DUMP_PARSERS = """
import argparse, json, runpy, sys
kind, target = sys.argv[1], sys.argv[2]
found = {}
def walk(parser, name):
    options = {}
    for action in parser._actions:
        for option in action.option_strings:
            options[option] = sorted(map(str, action.choices)) if action.choices else None
        if isinstance(action, argparse._SubParsersAction):
            for sub, subparser in action.choices.items():
                walk(subparser, sub)
    found[name] = options
def dump(self, *args, **kwargs):
    walk(self, "")
    print(json.dumps(found))
    raise SystemExit(0)
argparse.ArgumentParser.parse_args = dump
argparse.ArgumentParser.parse_known_args = dump
sys.argv = [target]
if kind == "path":
    runpy.run_path(target, run_name="__main__")
else:
    runpy.run_module(target, run_name="__main__", alter_sys=True)
"""


@functools.cache
def _options(tool: str) -> dict[str, list[str] | None]:
    """The tool's argparse option strings, read off its parser objects."""
    kind, target, sub = TOOLS[tool]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for name in ("BASH_ENV", "GMAT_CONSOLE", "GMAT_STARTUP"):
        env.pop(name, None)
    done = subprocess.run([sys.executable, "-c", _DUMP_PARSERS, kind, target], cwd=ROOT,
                          env=env, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    parsers: dict[str, dict[str, list[str] | None]] = json.loads(done.stdout.splitlines()[-1])
    assert sub in parsers, f"{target} has no subcommand {sub!r}: {sorted(parsers)}"
    return parsers[sub]


def _flags(tool: str) -> frozenset[str]:
    return frozenset(_options(tool))


# Every copy of a fact, not one: a doc that states a number twice drifts one copy at a time.
ASSUMPTIONS = _text("docs/ASSUMPTIONS_AND_GAPS.md")
SCORING = _text("docs/SCORING_STRATEGY_AND_OPEN_QUESTIONS.md")
SCORING_ZH = _text("docs/SCORING_STRATEGY_AND_OPEN_QUESTIONS.zh-TW.md")
ARCHITECTURE = _text("docs/ARCHITECTURE.md")
RUNBOOK = (ROOT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")


def _all(pattern: str, text: str, at_least: int = 1) -> list[str]:
    found = re.findall(pattern, text)
    assert len(found) >= at_least, f"{pattern!r} found {len(found)} times"
    return found


def _eps_m_s() -> str:
    return f"{RANKING.impulse_epsilon_km_s * 1000.0:.0e}".replace("e-0", "e-")


def test_epsilon_in_every_policy_doc_is_the_rulebooks() -> None:
    eps = _eps_m_s()
    for name, text in (("COMPETITION_POLICY", POLICY), ("ASSUMPTIONS_AND_GAPS", ASSUMPTIONS),
                       ("SCORING_STRATEGY", SCORING), ("SCORING_STRATEGY.zh-TW", SCORING_ZH)):
        assert set(_all(r"ε_v = ([\d.e-]+) m/s", text)) == {eps}, name


def test_margins_in_scoring_strategy_are_the_configs() -> None:
    dv, total, gap = (_m_s(RANK.dv_safe_limit_km_s), f"{RANK.total_dv_safe_limit_km_s:g}",
                      f"{RANK.safe_burn_separation_s:g}")
    pre_dv, pre_gap = f"{PRE.dv_safe_limit_km_s:g}", f"{PRE.safe_burn_separation_s:g}"
    assert f"ranking round {dv} m/s, {total} km/s and {gap} s" in SCORING
    assert f"`config/ranking.yaml` uses {dv} m/s, {total} km/s and {gap} s" in SCORING
    assert f"preliminary {pre_dv} km/s and {pre_gap} s" in SCORING
    ranking = f"{dv} m/s、{total} km/s、{gap} s"
    assert f"排位賽 {ranking}" in SCORING_ZH
    assert f"`config/ranking.yaml` 用 {ranking}" in SCORING_ZH
    assert f"初賽 {pre_dv} km/s、{pre_gap} s" in SCORING_ZH
    correct_top = {str(RANKING_CONFIG.gmat.correct_top)}
    # Whole numbers only: "0.05 in the ranking profiles" is another setting.
    assert set(_all(r"(?<![\d.])(\d+) in the ranking profiles", SCORING)) == correct_top
    assert set(_all(r"ranking 設定檔為 (\d+)", SCORING_ZH)) == correct_top
    for text in (SCORING, SCORING_ZH):
        assert set(_all(r"Team(\d+)", text)) == {str(OUR_TEAM)}


def test_numbers_in_assumptions_and_gaps_are_the_codes() -> None:
    assert set(_all(r"Team(\d+)", ASSUMPTIONS)) == {str(OUR_TEAM)}
    floor = f"{PRE.min_propagate_duration_s:.0e}".replace("e-0", "e-")
    assert f"(default {floor} s)" in ASSUMPTIONS
    separations = {load_config(path).policy.safe_burn_separation_s
                   for path in sorted((ROOT / "config").glob("*.yaml"))
                   if re.search(r"^policy:", path.read_text(encoding="utf-8"), re.MULTILINE)}
    assert len(separations) == 1
    assert f"({separations.pop():g} s in every profile)" in ASSUMPTIONS


#: Identifiers in ARCHITECTURE's diagram that are not modules, and what they are.
DIAGRAM_NON_MODULES = {
    "best_submittable": "rank's output field",
}


def test_every_module_architecture_md_draws_exists() -> None:
    diagram = re.search(r"```text(.*?)```", (ROOT / "docs" / "ARCHITECTURE.md").read_text(
        encoding="utf-8"), re.DOTALL)
    assert diagram is not None
    names = set(re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", diagram.group(1)))
    assert {"gmat_loop", "gmat_shoot", "ranking_search"} <= names
    for name in sorted(names - set(DIAGRAM_NON_MODULES)):
        assert ((ROOT / "src" / "orbitwin" / f"{name}.py").is_file()
                or (ROOT / "scripts" / f"{name}.py").is_file()), name
    verify = (ROOT / "src" / "orbitwin" / "verify.py").read_text(encoding="utf-8")
    assert 'f"exact-run-{index:02d}"' in verify and 'f"console-exact-run-{index:02d}"' in verify
    assert "best_submittable" in (ROOT / "src" / "orbitwin" / "rank.py").read_text(
        encoding="utf-8")


#: What no doc may still say about a change already made.
STALE_PHRASES = ("合併之後", "合併之前", "after the PR that adds", "not on main",
                 "not merged", "until the console-copy", "pending sign-off")


def test_console_copy_is_described_as_on() -> None:
    """gmat.console_copy exists and is off by default, the ranking profiles turn it on,
    the sign-off is in COMPETITION_POLICY.md, the docs say it is on, and ASSUMPTIONS
    names RUNBOOK step 6c, where the copy-mode path is."""
    from orbitwin.config import GmatConfig

    assert GmatConfig.__dataclass_fields__["console_copy"].default is False
    for profile in ("ranking.yaml", "ranking-rehearsal.yaml"):
        assert load_config(ROOT / "config" / profile).gmat.console_copy, profile
    assert ("### Console copy of a script that draws: signed off" in POLICY
            and "Signed off by: thc1006 <84045975+thc1006@users.noreply.github.com> "
            "Date: 2026-11-06" in POLICY)
    assert re.search(r"^## 6c\. Copy mode", RUNBOOK, re.MULTILINE)
    six_c = RUNBOOK.split("## 6c.", 1)[1].split("\n## ", 1)[0]
    assert "the ranking profiles set it" in " ".join(six_c.split())
    # Nor may any of them still say it waits.
    for name in ("docs/ASSUMPTIONS_AND_GAPS.md", "docs/ARCHITECTURE.md",
                 "docs/SCORING_STRATEGY_AND_OPEN_QUESTIONS.md",
                 "docs/SCORING_STRATEGY_AND_OPEN_QUESTIONS.zh-TW.md", "docs/RUNBOOK.md",
                 "docs/COMPETITION_POLICY.md"):
        text = _text(name)
        for stale in STALE_PHRASES:
            assert stale not in text, (name, stale)
    assert ("plot-free `gmat.console_copy` (set in the ranking profiles, signed off on "
            "2026-11-06)" in ASSUMPTIONS)
    assert "plot-free copy (console_copy, gmat.console_copy)" in _text("docs/ARCHITECTURE.md")
    assert "(docs/RUNBOOK.md step 6c)" in ASSUMPTIONS


def test_the_runbooks_one_command_is_orbitwin_days() -> None:
    """Its flags are the parser's, and it names where the upload goes."""
    from orbitwin.day import DEFAULT_FINAL_ROOT, NOTE_NAME, READBACKS

    block = RUNBOOK.split("## One command", 1)[1].split("\n## ", 1)[0]
    command = block.split("```text\n", 1)[1].split("```", 1)[0]
    assert command.startswith("orbitwin day ")
    named = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", block))
    assert named <= set(_flags("orbitwin day")), sorted(named - set(_flags("orbitwin day")))
    assert {"--horizon-s", "--kt-per-s", "--ct-s", "--kv-per-m-s", "--cv-m-s", "--cr-km",
            "--resume", "--exclude"} <= named
    final = DEFAULT_FINAL_ROOT.relative_to(Path(__file__).resolve().parents[2])
    assert f"`{final}/<UTC>/`" in block and f"`{NOTE_NAME}`" in block
    assert "--team-number" in named
    assert all(f"--confirm-readback {step}`" in block for step in READBACKS)
    assert {"--accept-config-warnings", "--accept-rank-warnings", "--include",
            "--force-latest", "--allow-non-main", "--expect-commit", "--by-eye-passed"} <= named
    assert "`--expect-commit`" in RUNBOOK.split("## 1. Preserve the input", 1)[0]


def test_the_sections_cited_from_code_and_docs_exist() -> None:
    """The day's T_max refusal and the docs cite these headings: a rename left them dangling."""
    policy = (ROOT / "docs" / "COMPETITION_POLICY.md").read_text(encoding="utf-8")
    order = policy.split("## Order of authority\n", 1)[1].split("\n## ", 1)[0]
    assert re.findall(r"^(\d)\. ", order, re.MULTILINE) == ["1", "2", "3", "4", "5"]
    assert "\n## Unresolved permissions\n" in policy
    assert "\n## Release status meaning\n" in (ROOT / "docs" / "ASSUMPTIONS_AND_GAPS.md").read_text(
        encoding="utf-8")
    assert re.search(r"^## Who runs what$", RUNBOOK, re.MULTILINE)
    clock = (ROOT / "src" / "orbitwin" / "mission_clock.py").read_text(encoding="utf-8")
    assert "docs/COMPETITION_POLICY.md, Order of authority" in clock


#: The [tool.ruff] table of pyproject.toml as reviewed; any change there fails until
#: this changes with it.
RUFF_CONFIG = {
    "line-length": 100,
    "target-version": "py311",
    "lint": {"select": ["E", "F", "I", "B", "UP", "SIM", "RUF"], "ignore": ["E501"]},
}
#: The noqa the tree keeps on purpose: how many findings of each code each file
#: hides. A new one is argued for first.
KEPT_NOQA: dict[str, dict[str, int]] = {
    "F401": {"tests/unit/test_seed_union.py": 1},
    "SIM108": {"src/orbitwin/orbit.py": 1},
}
#: What `make lint` checks.
LINTED = ("src", "tests", "scripts")
#: How `make lint` and `make validate` run ruff: --config, so it reads no other config file.
LINT = ("check", "--config", "pyproject.toml", *LINTED)
#: A comment holding an action of ruff's import sorter, which --ignore-noqa leaves in
#: force; a noqa naming RUF100 or M001, its old code, which also silences the report of
#: that noqa; or a file-level noqa in ruff's syntax.
_REFUSED_COMMENT = re.compile(
    r"(?i:isort\s*:|noqa\s*:.*(?:RUF100|M001))|#\s*(?:ruff|flake8)\s*:\s*(?i:noqa)")
#: The names of ruff's config files.
_CONFIG_NAMES = ("ruff.toml", ".ruff.toml", "pyproject.toml")


def _ruff(root: Path, *flags: str) -> str:
    """ruff's output under root, run as `make lint` runs it, with no RUFF_* setting."""
    try:
        from ruff import find_ruff_bin  # the binary `python -m ruff` execs

        binary = find_ruff_bin()
    except (ImportError, FileNotFoundError):
        pytest.fail("the ruff gate needs ruff from requirements-dev.txt", pytrace=False)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("RUFF_")}
    done = subprocess.run([binary, *LINT, "--no-cache", *flags], cwd=root,
                          capture_output=True, text=True, check=False, env=environment)
    assert done.returncode == 0, done.stderr
    return done.stdout


def _findings(root: Path, *flags: str) -> Counter[tuple[str, str]]:
    found = json.loads(_ruff(root, "--exit-zero", "--output-format", "json", *flags))
    return Counter((Path(item["filename"]).relative_to(root).as_posix(), str(item["code"]))
                   for item in found)


def hidden_noqa(root: Path) -> dict[str, dict[str, int]]:
    """What ruff's suppressions hide under root, counted by code and file as in KEPT_NOQA."""
    hidden = _findings(root, "--ignore-noqa") - _findings(root)
    counts: dict[str, dict[str, int]] = {}
    for (path, code), count in sorted(hidden.items()):
        counts.setdefault(code, {})[path] = count
    return counts


def ruff_config(root: Path) -> object:
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["tool"].get("ruff")


def unchecked_files(root: Path) -> list[str]:
    """Files the gate cannot check: another ruff config, which plain `ruff check` would read
    (git-ignored too) and `make lint` would not, and under LINTED a notebook or a name that
    is not UTF-8."""
    found = []
    for path in [*repo_files(root, "*"), *ignored_files(root, _CONFIG_NAMES)]:
        name = path.relative_to(root).as_posix()
        problem = _config_problem(path) if name != "pyproject.toml" else None
        if problem:
            found.append(f"{name}: {problem}")
        elif name.split("/", 1)[0] not in LINTED:
            continue
        elif name.endswith(".ipynb"):
            found.append(f"{name}: a notebook")
        elif not _utf8(name):
            found.append(f"{name!r}: not UTF-8, which ruff prints with U+FFFD")
    return found


def _config_problem(path: Path) -> str | None:
    """Why ruff would read path as its config, or None."""
    if path.name not in _CONFIG_NAMES:
        return None
    if path.name != "pyproject.toml":
        return "a ruff config other than pyproject.toml's"
    try:
        tool = tomllib.loads(path.read_text(encoding="utf-8")).get("tool", {})
    except ValueError as exc:  # TOMLDecodeError, UnicodeDecodeError
        return f"does not parse: {exc}"
    return "a ruff config other than pyproject.toml's" if "ruff" in tool else None


def _utf8(name: str) -> bool:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def linted_files(root: Path) -> list[str]:
    """The files `ruff check --show-files` lists, relative to root."""
    return sorted(Path(name).relative_to(root).as_posix()
                  for name in _ruff(root, "--show-files").split("\n") if name)


def unlinted_python(root: Path, linted: list[str]) -> set[str]:
    """The Python files git lists under LINTED that ruff does not lint."""
    python = {path.relative_to(root).as_posix() for path in repo_files(root, "*.py")}
    return {name for name in python if name.split("/", 1)[0] in LINTED} - set(linted)


def refused_comments(root: Path, linted: list[str]) -> list[str]:
    """Where a comment in a Python file ruff lints matches _REFUSED_COMMENT."""
    found = []
    for name in linted:
        text = (root / name).read_text(encoding="utf-8") if name.endswith((".py", ".pyi")) else ""
        if not _REFUSED_COMMENT.search(text):
            continue
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (SyntaxError, tokenize.TokenError) as exc:
            found.append(f"{name}: {exc}")
            continue
        found += [f"{name}:{token.start[0]}" for token in tokens
                  if token.type == tokenize.COMMENT and _REFUSED_COMMENT.search(token.string)]
    return found


def test_ruff_suppresses_only_the_kept_noqa() -> None:
    assert hidden_noqa(ROOT) == KEPT_NOQA


def test_ruff_reads_only_the_reviewed_config() -> None:
    assert ruff_config(ROOT) == RUFF_CONFIG
    assert unchecked_files(ROOT) == []
    linted = linted_files(ROOT)
    assert unlinted_python(ROOT, linted) == set()
    assert refused_comments(ROOT, linted) == []


def test_make_lint_and_make_validate_run_ruff_as_the_gate_does() -> None:
    assert _recipe(ROOT, "lint") == f"\t$(PYTHON) -m ruff {' '.join(LINT)}\n"
    validate = (ROOT / "scripts" / "release_validate.sh").read_text(encoding="utf-8")
    command = ", ".join(repr(arg) for arg in ("-m", "ruff", *LINT))
    assert f"\n    'ruff': [sys.executable, {command}],\n" in validate


def test_rule_packs_names_the_rule_sets_the_day_commands_take() -> None:
    from orbitwin import config, workspace

    text = _text("docs/RULE_PACKS.md")
    assert f"`orbitwin day` runs only `{config.DAY_RULES}` itself" in text
    assert (f"`orbitwin init` and `orbitwin run` take only `{workspace.DAY_RULES}` "
            "(`workspace.DAY_RULES`)") in text
