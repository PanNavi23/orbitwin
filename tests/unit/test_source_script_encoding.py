"""The organizer's own script has to be ASCII too, and nothing checked it.

GMAT R2026a refuses to interpret a script containing any byte above 127, before
anything in it is read:

    Interpreter Exception: The script "..." contains characters outside of the
    ASCII character set; please fix the file before proceeding.

`script_renderer._assert_ascii` said that about the script we generate. On
competition day the input is a file nobody has seen, and every byte of it that is
not a burn value or an elapsed time is copied into the submission verbatim. What
each of the four ways a high byte can arrive did before the parse-time guard
existed is recorded on the test that covers it.

Two conventions here are deliberate. Every broken asset is built at run time from
a bundled ASCII script, because committing a file of deliberately broken bytes
invites an editor to quietly repair it and a repaired fixture is a test that
passes for the wrong reason. And this file is itself pure ASCII, spelling every
high character as an escape, so that nothing in a test about stray bytes depends
on how a tool chose to save it.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import own_git, repo_files

from orbitwin.config import OrbitWinConfig
from orbitwin.errors import ParseError
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.script_renderer import _assert_ascii

UTF8_BOM = b"\xef\xbb\xbf"


def _write(tmp_path: Path, name: str, payload: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def _ascii_source(script: Path) -> bytes:
    raw = script.read_bytes()
    assert all(byte < 0x80 for byte in raw), f"{script} is expected to be pure ASCII"
    return raw


def test_a_non_ascii_comment_is_refused(different_plane_script: Path, tmp_path: Path) -> None:
    """Before the guard: parsed silently, rendered through, caught only at the end.

    The bytes reached the rendered script untouched and the run died in
    `assert_final_script_policy`, which reported them against the *final* script.
    True, and it sends whoever reads it hunting for our bug in a defect that was
    in the input all along -- after a full search has already been spent.
    """
    raw = _ascii_source(different_plane_script)
    payload = raw.replace(b"circular orbits", "\u5713\u8ecc\u9053".encode(), 1)
    path = _write(tmp_path, "comment.script", payload)
    with pytest.raises(ParseError, match="non-ASCII"):
        parse_gmat_script(path, OrbitWinConfig())


def test_a_non_ascii_resource_name_is_refused(
    different_plane_script: Path, tmp_path: Path
) -> None:
    """Before the guard: parsed, and the accented name became the pursuer.

    `_ASSIGN_RE` matches an object as `[A-Za-z_]\\w*`, and Python's `\\w` is
    Unicode-aware, so an assignment to `Purs<e-acute>er` matched and the scenario
    carried a non-ASCII identifier through the whole pipeline.
    """
    raw = _ascii_source(different_plane_script)
    payload = raw.replace(b"Pursuer", "Purs\u00e9er".encode())
    path = _write(tmp_path, "name.script", payload)
    with pytest.raises(ParseError, match="non-ASCII"):
        parse_gmat_script(path, OrbitWinConfig())


def test_a_resource_name_starting_with_a_non_ascii_byte_is_refused(
    different_plane_script: Path, tmp_path: Path
) -> None:
    """Before the guard: refused, but for a reason that was not true.

    `Create` takes any name, while `_ASSIGN_RE` requires an ASCII first
    character, so every assignment line for the renamed object was skipped and
    the object looked fieldless. The parser stopped with "Missing required
    assignment <name>.Epoch" about an Epoch line sitting two lines further down.
    """
    raw = _ascii_source(different_plane_script)
    payload = raw.replace(b"Pursuer", "\u00c9ursuer".encode())
    path = _write(tmp_path, "leading.script", payload)
    with pytest.raises(ParseError, match="non-ASCII") as caught:
        parse_gmat_script(path, OrbitWinConfig())
    assert "Missing required assignment" not in str(caught.value)


def test_a_utf8_bom_is_refused(different_plane_script: Path, tmp_path: Path) -> None:
    """Before the guard: nothing caught it, at any stage.

    `ScriptEncoding.has_utf8_bom` records the mark and the renderer puts it back
    byte-for-byte, which is the right instinct for a file we must not mutate and
    which here carries three bytes above 127 into the submission. It was measured
    passing `assert_final_script_policy` end to end. The parse boundary is the
    only place left that can see it, so it refuses and says how to fix it.
    """
    raw = _ascii_source(different_plane_script)
    path = _write(tmp_path, "bom.script", UTF8_BOM + raw)
    with pytest.raises(ParseError, match="byte-order mark") as caught:
        parse_gmat_script(path, OrbitWinConfig())
    message = str(caught.value)
    assert "non-ASCII" in message
    assert "EF BB BF" in message


def test_the_final_script_guard_cannot_see_a_bom(different_plane_script: Path) -> None:
    """Why the byte-order mark is refused rather than merely warned about.

    `_assert_ascii` inspects decoded lines, and `utf-8-sig` has eaten the mark
    long before it looks. This pins the blind spot: if a later change ever makes
    the renderer's guard catch a BOM, this test fails and the parse-time refusal
    can be revisited.
    """
    raw = _ascii_source(different_plane_script)
    decoded = (UTF8_BOM + raw).decode("utf-8-sig")
    document = SimpleNamespace(lines=tuple(decoded.splitlines(keepends=True)))
    _assert_ascii(document)  # does not raise, and that is the problem


def test_a_byte_that_is_not_utf8_at_all_fails_as_an_orbitwin_error(
    different_plane_script: Path, tmp_path: Path
) -> None:
    """Before the guard: UnicodeDecodeError escaped `GmatScriptDocument.read`.

    Not an `OrbitWinError`, and its position was a byte index into the decode
    stream with no line number in it. 0xFF is never legal UTF-8 anywhere, so the
    check has to run on the raw bytes, ahead of the decode.
    """
    raw = _ascii_source(different_plane_script)
    payload = raw.replace(b"Prop.Accuracy", b"\xffProp.Accuracy", 1)
    path = _write(tmp_path, "invalid.script", payload)
    with pytest.raises(ParseError, match="not even UTF-8") as caught:
        parse_gmat_script(path, OrbitWinConfig())
    assert "0xFF" in str(caught.value)


def test_the_message_locates_the_byte(different_plane_script: Path, tmp_path: Path) -> None:
    """Naming the byte is not enough; it has to be findable in seconds.

    Line, column and absolute offset, so it can be reached by any of the tools
    someone has to hand under a clock, plus the line itself. The character is a
    non-breaking space, which is the one that arrives by paste and is invisible
    in the editor the operator will open next.
    """
    raw = _ascii_source(different_plane_script)
    marker = b"Prop.MaxStepAttempts = 50;"
    assert raw.count(marker) == 1
    inserted = "\u00a0".encode()
    payload = raw.replace(marker, b"Prop.MaxStepAttempts =" + inserted + b" 50;", 1)
    path = _write(tmp_path, "located.script", payload)
    with pytest.raises(ParseError) as caught:
        parse_gmat_script(path, OrbitWinConfig())
    message = str(caught.value)

    offset = payload.index(inserted)
    line_number = payload.count(b"\n", 0, offset) + 1
    column = offset - (payload.rfind(b"\n", 0, offset) + 1) + 1
    assert f"line {line_number}" in message
    assert f"column {column}" in message
    assert f"byte offset {offset}" in message
    assert "0xC2" in message
    assert "U+00A0" in message
    assert "Prop.MaxStepAttempts" in message


def test_a_clean_script_is_untouched(different_plane_script: Path) -> None:
    """The guard has to be invisible to every script that is already ASCII."""
    scenario, document = parse_gmat_script(different_plane_script, OrbitWinConfig())
    assert document.raw_bytes == different_plane_script.read_bytes()
    assert scenario.encoding.has_utf8_bom is False
    assert scenario.warnings == ()


def _bundled_scripts(root: Path) -> list[Path]:
    """The `.script` files under root that git tracks, or would (`repo_files`).

    output/ and agent worktrees hold hundreds of deliberately broken scripts nobody
    ships; git ignores the first and does not descend into the second, a repository
    of its own.
    """
    return repo_files(root, "*.script")


def test_every_bundled_script_asset_is_ascii(project_root: Path) -> None:
    """No bundled `.script` may carry a high byte, including any added later.

    Reproducing this bug meant writing scripts that are deliberately broken, and
    they are built at run time precisely so that none of them can end up here.
    """
    scripts = _bundled_scripts(project_root)
    assert scripts, "expected bundled .script assets"
    offenders = [
        str(path.relative_to(project_root))
        for path in scripts
        if any(byte > 0x7F for byte in path.read_bytes())
    ]
    assert offenders == []


def _tree(root: Path) -> None:
    for name in ("examples/a.script", "output/b.script", ".worktrees/w/c.script"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"% \xc3\xa9\n")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_scan_reads_what_git_would_ship(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tracked or untracked, never ignored, never inside another checkout.

    Reading tracked files only let a new non-ASCII script pass locally until
    someone ran `git add`.
    """
    own_git(monkeypatch, tmp_path)
    _tree(tmp_path)
    (tmp_path / "examples" / "new.script").write_bytes(b"% \xc2\xa0\n")
    (tmp_path / ".gitignore").write_text("/output/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "init", "-q", str(tmp_path / ".worktrees" / "w")],
                   check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "examples/a.script"], check=True)
    assert _bundled_scripts(tmp_path) == [
        tmp_path / "examples" / "a.script", tmp_path / "examples" / "new.script"
    ]


def test_without_git_the_scan_skips_hidden_and_ignored_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree(tmp_path)
    (tmp_path / "build.script").write_bytes(b"%\n")

    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert _bundled_scripts(tmp_path) == [
        tmp_path / "build.script", tmp_path / "examples" / "a.script"
    ]


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_outside_a_repository_the_scan_falls_back_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unpacked release: git is installed, the tree is no repository, and git exits 128."""
    _tree(tmp_path)
    (tmp_path / "build.script").write_bytes(b"%\n")
    # Stops git from finding a repository above tmp_path, as under a checkout's output/.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert subprocess.run(["git", "-C", str(tmp_path), "rev-parse"],
                          capture_output=True, check=False).returncode != 0

    assert _bundled_scripts(tmp_path) == [
        tmp_path / "build.script", tmp_path / "examples" / "a.script"
    ]


@pytest.mark.parametrize(
    ("fixture_name", "pursuer"),
    [
        ("workshop_script", None),
        ("different_plane_script", None),
        ("high_plane_script", None),
        ("no_maneuver_script", "Pursuer"),
        ("unmaneuvered_burns_script", "Pursuer"),
    ],
)
def test_every_bundled_example_still_parses(
    request: pytest.FixtureRequest, fixture_name: str, pursuer: str | None
) -> None:
    """A fail-closed check that rejects the organizer's own examples is a bug."""
    script: Path = request.getfixturevalue(fixture_name)
    config = OrbitWinConfig()
    if pursuer is not None:
        config = dataclasses.replace(config, pursuer_name=pursuer, target_name="AlienSC")
    scenario, _ = parse_gmat_script(script, config)
    assert scenario.max_time_s > 0.0


def test_the_guard_reads_bytes_not_characters(
    different_plane_script: Path, tmp_path: Path
) -> None:
    """A high byte hidden inside a quoted value is still a high byte.

    Nothing about the refusal depends on where in the line the byte sits, which
    is the point: GMAT refuses the file, not the statement.
    """
    raw = _ascii_source(different_plane_script)
    payload = raw.replace(
        b"'OfficialReport.txt'", "'OfficialReport\u2013.txt'".encode(), 1
    )
    path = _write(tmp_path, "quoted.script", payload)
    with pytest.raises(ParseError, match="non-ASCII") as caught:
        parse_gmat_script(path, OrbitWinConfig())
    assert re.search(r"line \d+, column \d+", str(caught.value))
