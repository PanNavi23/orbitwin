"""tests/conftest.py's reader of ORGANIZER_MATERIAL.txt, on lists of its own.

Without the list, as in the public export, a test of the organizer's material
skips. With it, a key it does not name or a path not in the tree fails: a list
that has drifted from the tree must not turn those tests into skips.
"""

from __future__ import annotations

from pathlib import Path

import conftest
import pytest
from conftest import organizer_file, organizer_scripts, read_organizer_material


def _list(tmp_path: Path, text: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    listed = tmp_path / "ORGANIZER_MATERIAL.txt"
    listed.write_text(text, encoding="utf-8")
    monkeypatch.setattr(conftest, "ORGANIZER_MATERIAL", listed)
    return listed


def test_without_the_list_a_file_skips_and_there_are_no_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conftest, "ORGANIZER_MATERIAL", tmp_path / "ORGANIZER_MATERIAL.txt")
    assert read_organizer_material() is None
    assert organizer_scripts() == []
    with pytest.raises(pytest.skip.Exception, match="not in this tree"):
        organizer_file("problem")


def test_comments_blanks_and_tabs_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "theirs").mkdir()
    (tmp_path / "theirs" / "p.script").write_text("x\n", encoding="utf-8")
    _list(tmp_path, "# a comment\n"
                    "\n"
                    "   # an indented one\n"
                    "pattern\ttheirs/*   # a comment after a blank\n"
                    "  pattern   top.script\t\n"
                    "header \t%Created: 2026-01-01  12:00:00\n"
                    "file\tproblem \t theirs/p.script\n"
                    "file folder theirs\n", monkeypatch)
    material = read_organizer_material()
    assert material is not None
    assert material.patterns == ("theirs/*", "top.script")
    # Only the blanks around a value go; the header keeps its own.
    assert material.headers == ("%Created: 2026-01-01  12:00:00",)
    assert material.files == {"problem": "theirs/p.script", "folder": "theirs"}
    assert organizer_file("problem") == tmp_path / "theirs" / "p.script"
    assert organizer_file("folder") == tmp_path / "theirs"


def test_a_key_the_list_does_not_name_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _list(tmp_path, "pattern theirs/*\nfile other theirs/p.script\n", monkeypatch)
    with pytest.raises(pytest.fail.Exception, match="no `file problem <path>` line"):
        organizer_file("problem")


def test_a_path_not_in_the_tree_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _list(tmp_path, "file problem theirs/gone.script\n", monkeypatch)
    with pytest.raises(pytest.fail.Exception, match=r"names theirs/gone\.script for problem"):
        organizer_file("problem")


@pytest.mark.parametrize("line", ["patterns theirs/*", "pattern", "header   ",
                                  "file problem", "theirs/*"])
def test_a_line_it_cannot_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str
) -> None:
    _list(tmp_path, f"pattern ok/*\n{line}\n", monkeypatch)
    with pytest.raises(pytest.fail.Exception, match=":2 is not a pattern, header or"):
        read_organizer_material()


def test_a_key_named_twice_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _list(tmp_path, "file problem a.script\nfile problem b.script\n", monkeypatch)
    with pytest.raises(pytest.fail.Exception, match="names the file problem a second time"):
        read_organizer_material()


def test_the_scripts_are_found_from_each_patterns_folder_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("theirs/a.script", "theirs/deep/b.script", "theirs/notes.md", "top.script",
                 "ours/c.script", "ours/theirs/d.script"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    _list(tmp_path, "pattern theirs/*\npattern top.script\npattern gone/*.script\n", monkeypatch)
    walked: list[Path] = []
    rglob = Path.rglob

    def recording(self: Path, pattern: str):  # type: ignore[no-untyped-def]
        walked.append(self)
        return rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", recording)
    # * crosses /, as in fnmatch; ours/ is never walked.
    assert organizer_scripts() == [tmp_path / "theirs" / "a.script",
                                   tmp_path / "theirs" / "deep" / "b.script",
                                   tmp_path / "top.script"]
    assert walked == [tmp_path / "theirs"]
