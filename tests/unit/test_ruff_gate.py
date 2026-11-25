"""The ruff gate of test_policy_docs_match_code.py on small trees: what it must refuse."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

# Imported here: an inline run that imports it first takes scipy with it on the way out,
# as pytester puts sys.modules back.
import test_source_script_encoding
from conftest import (
    drop_repository_variables,
    inline_run,
    own_git,
    repo_files,
    scratch_git_environment,
)
from test_policy_docs_match_code import (
    ROOT,
    RUFF_CONFIG,
    hidden_noqa,
    linted_files,
    refused_comments,
    ruff_config,
    unchecked_files,
    unlinted_python,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

KEPT = {"F401": {"src/pkg/a.py": 1}}
FILES = {
    "src/pkg/a.py": "import os  # noqa: F401\n",
    "scripts/b.py": 'print("b")\n',
    "tests/test_c.py": "def test_c() -> None:\n    pass\n",
}


def _edit(name: str, old: str, new: str) -> Callable[[Path], None]:
    def edit(root: Path) -> None:
        text = (root / name).read_text(encoding="utf-8") if (root / name).exists() else ""
        assert old in text, (name, old)
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text.replace(old, new, 1), encoding="utf-8")

    return edit


def _move(old: str, new: str) -> Callable[[Path], None]:
    def move(root: Path) -> None:
        (root / new).parent.mkdir(parents=True, exist_ok=True)
        (root / old).rename(root / new)

    return move


@pytest.fixture(autouse=True)
def _own_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    own_git(monkeypatch, tmp_path)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    shutil.copy2(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    for name, text in FILES.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    return tmp_path


def _set(name: str, value: str | None = None) -> bool:
    """Whether name is in the environment (with value, if given). A bool, so that a failed
    assert prints no environment."""
    return name in os.environ and value in (None, os.environ[name])


def test_the_unchanged_tree_passes(tree: Path) -> None:
    assert hidden_noqa(tree) == KEPT
    assert ruff_config(tree) == RUFF_CONFIG
    assert unchecked_files(tree) == []
    assert unlinted_python(tree, linted_files(tree)) == set()
    assert refused_comments(tree, linted_files(tree)) == []


def test_a_ruff_setting_in_the_environment_is_not_read(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With RUFF_OUTPUT_FILE, ruff wrote the gate's output to that file and printed nothing."""
    monkeypatch.setenv("RUFF_OUTPUT_FILE", str(tmp_path / "report"))
    assert hidden_noqa(tree) == KEPT
    assert not (tmp_path / "report").exists()


def test_a_kept_place_that_hides_nothing_is_refused(tree: Path) -> None:
    assert hidden_noqa(tree) != {**KEPT, "SIM108": {"src/pkg/a.py": 1}}


@pytest.mark.parametrize("edit", [
    pytest.param(_edit("src/pkg/a.py", "\n", "\nl = 1  # noqa: E741\n"), id="new-code"),
    pytest.param(_edit("scripts/b.py", "\n", "\nimport re  # noqa: F401\n"),
                 id="kept-code-in-another-file"),
    pytest.param(_edit("src/pkg/a.py", "\n", "\nimport re  # noqa: F401\n"),
                 id="one-more-of-a-kept-code"),
    pytest.param(_edit("tests/test_c.py", "", "# ruff: noqa: F401\nimport os\n\n\n"),
                 id="file-level-noqa"),
    pytest.param(_move("src/pkg/a.py", "src/pkg/sub/a.py"), id="kept-code-moved"),
])
def test_a_suppression_beyond_the_kept_is_refused(
    tree: Path, edit: Callable[[Path], None]
) -> None:
    edit(tree)
    assert hidden_noqa(tree) != KEPT


@pytest.mark.parametrize(("old", "new"), [
    ("line-length = 100", "line-length = 120"),
    ("line-length = 100", 'line-length = 100\nextend = "other.toml"'),
    ("line-length = 100", 'line-length = 100\nextend-exclude = ["scripts"]'),
    ("line-length = 100", 'line-length = 100\ninclude = ["src/**/*.py"]'),
    ('ignore = ["E501"]', 'ignore = ["E501"]\nextend-ignore = ["E402"]'),
    ('ignore = ["E501"]', 'ignore = ["E501"]\nper-file-ignores = {"x.py" = ["F401"]}'),
    ('ignore = ["E501"]', 'ignore = ["E501"]\npyflakes.extend-generics = ["x.Y"]'),
])
def test_a_config_change_is_refused(tree: Path, old: str, new: str) -> None:
    _edit("pyproject.toml", f"{old}\n", f"{new}\n")(tree)
    assert ruff_config(tree) != RUFF_CONFIG


@pytest.mark.parametrize(
    "name", ["scripts/ruff.toml", "scripts/.ruff.toml", "ruff.toml", "scripts/pyproject.toml",
             "docs/pyproject.toml"])
def test_another_config_file_is_refused(tree: Path, name: str) -> None:
    """Plain `ruff check` reads it and `make lint` does not."""
    table = "[tool.ruff.lint]" if name.endswith("pyproject.toml") else "[lint]"
    _edit(name, "", f'{table}\nignore = ["E402", "F401"]\n')(tree)
    assert unchecked_files(tree) == [f"{name}: a ruff config other than pyproject.toml's"]


@pytest.mark.parametrize("name", ["ruff.toml", "scripts/ruff.toml", "scripts/.ruff.toml",
                                  "scripts/pyproject.toml"])
def test_an_ignored_config_file_is_refused(tree: Path, name: str) -> None:
    """ruff reads a config file that git ignores; one in an ignored folder it never reaches."""
    _edit(".gitignore", "", f"/{name}\n/ignored/\n")(tree)
    _edit("ignored/ruff.toml", "", '[lint]\nignore = ["E402"]\n')(tree)
    table = "[tool.ruff.lint]" if name.endswith("pyproject.toml") else "[lint]"
    _edit(name, "", f'{table}\nignore = ["E402"]\n')(tree)
    assert unchecked_files(tree) == [f"{name}: a ruff config other than pyproject.toml's"]


def test_a_pyproject_without_ruff_is_not_a_config(tree: Path) -> None:
    _edit("scripts/pyproject.toml", "", '[project]\nname = "x"\n')(tree)
    assert unchecked_files(tree) == []


def test_a_pyproject_that_does_not_parse_is_named(tree: Path) -> None:
    _edit("scripts/pyproject.toml", "", "[tool.ruff\n")(tree)
    (found,) = unchecked_files(tree)
    assert found.startswith("scripts/pyproject.toml: does not parse: "), found


def test_without_the_ruff_package_the_gate_says_what_to_install(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "ruff", None)
    with pytest.raises(pytest.fail.Exception,
                       match=r"^the ruff gate needs ruff from requirements-dev\.txt$"):
        hidden_noqa(tree)


def test_a_notebook_is_refused(tree: Path) -> None:
    """ruff lints it, and honors an isort action in a cell that the comment check never reads."""
    _edit("tests/demo.ipynb", "", "{}\n")(tree)
    _edit("notebooks/demo.ipynb", "", "{}\n")(tree)
    assert unchecked_files(tree) == ["tests/demo.ipynb: a notebook"]


@pytest.mark.parametrize("comment", [
    pytest.param("import os  # isort: skip\n", id="skip"),
    pytest.param("import os  # keep isort: skip\n", id="after-words"),
    pytest.param("import os  # isort:skip\n", id="no-space"),
    pytest.param("# isort: skip_file\nimport os\n", id="skip-file"),
    pytest.param("# ruff: isort: skip_file\nimport os\n", id="ruff-prefix"),
    pytest.param("x = 1  # noqa: E501, RUF100\n", id="noqa-ruf100"),
    pytest.param("x = 1  # NOQA:RUF100\n", id="noqa-ruf100-upper"),
    pytest.param("x = 1  # noqa: M001\n", id="noqa-m001"),
    pytest.param("import os  # noqa: F401E402RUF100\n", id="noqa-ruf100-joined"),
    pytest.param("# ruff: noqa\n", id="file-noqa"),
    pytest.param("#ruff:NOQA\n", id="file-noqa-tight"),
    pytest.param("# flake8: noqa\n", id="file-noqa-flake8"),
    pytest.param("# ruff: noqa: E501\n", id="file-noqa-codes"),
    pytest.param("# flake8: noqa: E501\n", id="file-noqa-flake8-codes"),
    pytest.param("# ruff: noqa: RUF100\n", id="file-noqa-ruf100"),
])
def test_a_refused_comment_is_refused(tree: Path, comment: str) -> None:
    _edit("tests/test_c.py", "", comment)(tree)
    assert refused_comments(tree, linted_files(tree)) == ["tests/test_c.py:1"]


@pytest.mark.parametrize("text", [
    pytest.param('"""# isort: skip_file  # noqa: RUF100  # ruff: noqa"""\n', id="in-a-string"),
    pytest.param("# Ruff: noqa\n", id="not-ruffs-file-noqa"),
    pytest.param("# ruff noqa\n", id="no-colon"),
])
def test_a_comment_ruff_does_not_read_is_not_refused(tree: Path, text: str) -> None:
    _edit("tests/test_c.py", "", text)(tree)
    assert refused_comments(tree, linted_files(tree)) == []


def test_a_name_with_a_line_break_python_knows_is_one_file(tree: Path) -> None:
    """str.splitlines also breaks at \\x1c, which ruff prints as it is."""
    _edit("tests/test_\x1c.py", "", "x = 1\n")(tree)
    assert "tests/test_\x1c.py" in linted_files(tree)


def test_a_file_ruff_skips_is_refused(tree: Path) -> None:
    """ruff reads .ignore files, which git does not; a .pyi ruff lints and git's *.py misses is not."""
    _edit("src/pkg/a.pyi", "", "x: int\n")(tree)
    assert unlinted_python(tree, linted_files(tree)) == set()
    _edit(".ignore", "", "tests/test_c.py\n")(tree)
    assert unlinted_python(tree, linted_files(tree)) == {"tests/test_c.py"}


def _victim(tmp_path: Path) -> Path:
    victim = tmp_path / "victim"
    (victim / "kept.txt").parent.mkdir()
    (victim / "kept.txt").write_text("kept\n", encoding="utf-8")
    for command in (["init", "-q"], ["add", "kept.txt"],
                    ["-c", "user.name=v", "-c", "user.email=v@v", "commit", "-qm", "v"]):
        subprocess.run(["git", "-C", str(victim), *command], check=True)
    return victim


def _state(victim: Path) -> dict[str, bytes]:
    return {path.relative_to(victim).as_posix(): path.read_bytes()
            for path in sorted(victim.rglob("*")) if path.is_file()}


def _point_git_at(victim: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """What git gives a commit hook in a linked worktree: the repository and its index."""
    monkeypatch.setenv("GIT_DIR", str(victim / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(victim / ".git" / "index"))


@pytest.mark.parametrize("test", [
    f"{Path(__file__).resolve()}::test_the_unchanged_tree_passes",
    f"{Path(test_source_script_encoding.__file__).resolve()}"
    "::test_the_scan_reads_what_git_would_ship",
], ids=["ruff-gate-tree", "script-scan-tree"])
def test_a_scratch_repository_leaves_the_callers_alone(
    test: str, tmp_path: Path, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under such a hook, git init and add in a test re-made the caller's repository."""
    victim = _victim(tmp_path)
    before = _state(victim)
    _point_git_at(victim, monkeypatch)
    inline_run(pytester, "-o", "addopts=", test).assertoutcome(passed=1)
    assert _state(victim) == before


COMMAND_SCOPE = {"GIT_CONFIG_PARAMETERS": "'core.quotepath'='false'", "GIT_CONFIG_COUNT": "1",
                 "GIT_CONFIG_KEY_0": "core.quotepath", "GIT_CONFIG_VALUE_0": "false"}


def test_only_the_repository_variables_are_dropped(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim = _victim(tmp_path)
    (victim / ".git" / "info").mkdir(exist_ok=True)
    (victim / ".git" / "info" / "exclude").write_text("*.py\n", encoding="utf-8")
    _point_git_at(victim, monkeypatch)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    for name, value in COMMAND_SCOPE.items():
        monkeypatch.setenv(name, value)
    drop_repository_variables()
    assert not _set("GIT_DIR")
    assert not _set("GIT_INDEX_FILE")
    assert _set("GIT_CEILING_DIRECTORIES", str(tmp_path))
    for name, value in COMMAND_SCOPE.items():
        assert _set(name, value), name
    assert [path.relative_to(tree).as_posix() for path in repo_files(tree, "*.py")] == sorted(
        name for name in FILES if name.endswith(".py"))


def test_a_scratch_repository_reads_no_git_variable_of_the_callers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    leaked = "GIT_WORK_TREE" in scratch_git_environment(tmp_path)
    assert not leaked
    own_git(monkeypatch, tmp_path)
    assert not _set("GIT_WORK_TREE")
    assert _set("GIT_CONFIG_GLOBAL", os.devnull)


def test_a_broken_global_configuration_does_not_stop_the_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing reads no user configuration, which git refuses to run past."""
    (tmp_path / ".gitconfig").write_text("[broken\n", encoding="utf-8")  # own_git's HOME
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "index"))
    drop_repository_variables()
    assert not _set("GIT_INDEX_FILE")


def test_a_failed_listing_exits_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "git").write_text("#!/bin/sh\necho no >&2\nexit 1\n", encoding="ascii")
    (tmp_path / "bin" / "git").chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    with pytest.raises(pytest.exit.Exception, match=r"GIT_INDEX_FILE stay set: no$"):
        drop_repository_variables()


def test_repo_files_reads_the_users_git_configuration(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As the user's git and ruff do: a global core.excludesFile hides an untracked file."""
    (tmp_path / "ignore").write_text("local.py\n", encoding="utf-8")
    (tmp_path / "config").write_text(f"[core]\n\texcludesFile = {tmp_path / 'ignore'}\n",
                                     encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "config"))
    _edit("src/pkg/local.py", "", "x = 1\n")(tree)
    assert tree / "src" / "pkg" / "local.py" not in repo_files(tree, "*.py")


def test_repo_files_tracked_only_leaves_out_untracked_files(tree: Path) -> None:
    _edit("src/pkg/new.py", "", "x = 1\n")(tree)
    assert tree / "src" / "pkg" / "new.py" in repo_files(tree, "*.py")
    assert tree / "src" / "pkg" / "new.py" not in repo_files(tree, "*.py", tracked_only=True)


def test_repo_files_keeps_the_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Below the ceiling there is no repository, so the walk lists the files, not git."""
    outer = tmp_path / "outer"
    for name in ("inner/a.py", "inner/output/b.py"):
        _edit(name, "", "x = 1\n")(outer)
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(outer))
    assert repo_files(outer / "inner", "*.py") == [outer / "inner" / "a.py"]


def test_a_name_that_is_not_utf8_is_refused(tree: Path) -> None:
    name = os.fsdecode(b"src/pkg/\xff.py")
    (tree / name).write_text("x = 1\n", encoding="utf-8")
    assert tree / name in repo_files(tree, "*.py")
    assert unchecked_files(tree) == [f"{name!r}: not UTF-8, which ruff prints with U+FFFD"]


@pytest.fixture
def no_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tree outside any repository, as an unpacked release, with git not on PATH."""
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    root = tmp_path / "release"
    for name in ("a.script", "sub/b.script", "._a.script", ".hidden/c.script",
                 "run/final/d.script", "output/e.script", "src/x.egg-info/f.script",
                 "sub/__pycache__/g.script", "input/original/h.script"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x\n", encoding="ascii")
    return root


def test_without_git_only_what_ships_is_listed(no_git: Path) -> None:
    """Not hidden files (AppleDouble ._ copies among them), nor what a release makes after."""
    assert [path.relative_to(no_git).as_posix() for path in repo_files(no_git, "*.script")] == [
        "a.script", "sub/b.script"]


def test_without_git_a_hidden_config_file_is_refused(no_git: Path) -> None:
    """ruff reads .ruff.toml, which the walk leaves out as a hidden file."""
    for name in (".ruff.toml", "sub/.ruff.toml", ".hidden/.ruff.toml", "output/.ruff.toml"):
        (no_git / name).write_text('[lint]\nignore = ["E402"]\n', encoding="utf-8")
    assert unchecked_files(no_git) == [
        f"{name}: a ruff config other than pyproject.toml's"
        for name in (".ruff.toml", "sub/.ruff.toml")]
