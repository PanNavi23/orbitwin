"""`orbitwin --version` names the commit the unpacked archive was built from."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbitwin import __version__, cli

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _info(path: Path, **changes: object) -> Path:
    info = {"artifact": "tasa_orbitwin", "version": __version__, "commit": COMMIT,
            "committed_utc": "2026-11-04T10:11:12Z", "dirty": False,
            "manifest": "SHA256SUMS.txt", **changes}
    path.write_text(json.dumps(info), encoding="utf-8")
    return path


def test_version_prints_the_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """It printed `orbitwin 0.3.0` from any archive, an older one included."""
    monkeypatch.setattr(cli, "BUILD_INFO", _info(tmp_path / "BUILD_INFO.json"))

    with pytest.raises(SystemExit) as stopped:
        cli.main(["--version"])

    assert stopped.value.code == 0
    assert capsys.readouterr().out == (
        f"orbitwin {__version__}, commit {COMMIT}, committed 2026-11-04T10:11:12Z\n"
    )


def test_the_build_info_is_read_beside_src() -> None:
    assert Path(cli.__file__).resolve().parents[2] / "BUILD_INFO.json" == cli.BUILD_INFO


@pytest.mark.parametrize(
    ("contents", "said"),
    [
        (None, f"orbitwin {__version__}, no BUILD_INFO.json: not an unpacked release archive"),
        ("{not json", "is unreadable (JSONDecodeError"),
        ("[]", "is unreadable (TypeError"),
        ('{"commit": "x"}', "is unreadable (KeyError"),
        ({"version": "0.0.1"}, f"commit {COMMIT}, committed 2026-11-04T10:11:12Z, but "
                               "BUILD_INFO.json names version 0.0.1"),
        ({"dirty": True}, ", built with uncommitted changes"),
        ({"dirty": "no"}, ", built with uncommitted changes"),
    ],
)
def test_a_build_that_is_not_the_archives_says_so(
    tmp_path: Path, contents: str | dict[str, object] | None, said: str
) -> None:
    path = tmp_path / "BUILD_INFO.json"
    if isinstance(contents, dict):
        _info(path, **contents)
    elif contents is not None:
        path.write_text(contents, encoding="utf-8")

    line = cli.describe_build(path)

    assert said in line
    assert line.startswith(f"orbitwin {__version__}, ")
