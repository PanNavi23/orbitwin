"""An installed orbitwin names the commit it was installed from, and the day holds to it."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import scratch_git_environment
from test_installed_tools import PACKAGE, _shipped

from orbitwin import __version__
from orbitwin.installed import FRESH_PYCACHE

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _site(tmp_path: Path, direct: dict[str, object] | None) -> Path:
    """A site directory as an installer leaves it: the package, RECORD and direct_url.json."""
    site = tmp_path / "site"
    lines = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and _shipped(path):
            name = "orbitwin/" + path.relative_to(PACKAGE).as_posix()
            target = site / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
            lines.append(f"{name},sha256={digest.rstrip(b'=').decode()},{path.stat().st_size}")
    info = site / f"orbitwin-{__version__}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: orbitwin\nVersion: "
                                   f"{__version__}\n", encoding="utf-8")
    names = ["METADATA", "RECORD"]
    if direct is not None:
        (info / "direct_url.json").write_text(json.dumps(direct), encoding="utf-8")
        names.append("direct_url.json")
    for name in names:
        if name == "RECORD":
            lines.append(f"{info.name}/RECORD,,")
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256((info / name).read_bytes()).digest())
        lines.append(f"{info.name}/{name},sha256={digest.rstrip(b'=').decode()},1")
    (info / "RECORD").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return site


def _env(path: str, *, fresh: Path | None) -> dict[str, str]:
    """The environment of a process run from ``path``; ``fresh``, as `orbitwin day` re-runs
    itself, with an empty bytecode cache there."""
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "BASH_ENV", "GMAT_CONSOLE", "GMAT_STARTUP",
                          "PYTHONPYCACHEPREFIX", FRESH_PYCACHE)}
    env["PYTHONPATH"] = path
    if fresh is not None:
        fresh.mkdir(parents=True, exist_ok=True)
        fresh.chmod(0o700)
        env.update(PYTHONPYCACHEPREFIX=str(fresh), **{FRESH_PYCACHE: str(fresh)})
    return env


def _ask(site: Path, code: str, *, fresh: bool = True) -> subprocess.CompletedProcess[str]:
    env = _env(str(site), fresh=site.parent / f"pyc-{os.urandom(4).hex()}" if fresh else None)
    return subprocess.run([sys.executable, "-s", "-c", code], cwd=site.parent, env=env,
                          capture_output=True, text=True, check=False, timeout=300)


_PROVENANCE = ("import json\nfrom orbitwin.day import ROOT, DayStop, provenance\n"
               "try:\n    print(json.dumps(provenance(ROOT, {expect!r}, False)))\n"
               "except DayStop as stop:\n    print('STOP', stop)\n")


@pytest.fixture
def from_git(tmp_path: Path) -> Path:
    return _site(tmp_path, {"url": "https://github.com/PanNavi23/orbitwin",
                            "vcs_info": {"vcs": "git", "commit_id": COMMIT}})


def test_a_package_installed_from_git_runs_its_commit(from_git: Path) -> None:
    done = _ask(from_git, _PROVENANCE.format(expect=COMMIT[:12]))
    assert done.returncode == 0, done.stderr
    record = json.loads(done.stdout)
    assert (record["source"], record["commit"], record["on_main"]) == ("installed", COMMIT, True)
    assert record["local_changes"] == [] and record["version"] == __version__


def test_another_commit_is_refused(from_git: Path) -> None:
    done = _ask(from_git, _PROVENANCE.format(expect="fedcba9"))
    assert done.stdout.startswith("STOP"), done.stdout + done.stderr
    assert f"this is commit {COMMIT[:12]}, not the expected fedcba9" in done.stdout


def test_an_edited_installed_file_is_refused(from_git: Path) -> None:
    score = from_git / "orbitwin" / "score.py"
    score.write_text(score.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    (from_git / "orbitwin" / "extra.py").write_text("", encoding="utf-8")
    done = _ask(from_git, _PROVENANCE.format(expect=COMMIT))
    assert done.stdout.startswith("STOP"), done.stdout + done.stderr
    assert "orbitwin/score.py (changed)" in done.stdout
    assert "orbitwin/extra.py (not installed)" in done.stdout


def test_a_package_from_an_index_records_no_commit(tmp_path: Path) -> None:
    site = _site(tmp_path, None)
    done = _ask(site, _PROVENANCE.format(expect=COMMIT))
    assert done.stdout.startswith("STOP"), done.stdout + done.stderr
    assert f"installed orbitwin {__version__} records no commit" in done.stdout
    allowed = _ask(site, "import json\nfrom orbitwin.day import ROOT, provenance\n"
                   f"print(json.dumps(provenance(ROOT, {COMMIT!r}, True)))")
    record = json.loads(allowed.stdout)
    assert record["on_main"] is False and record["source"] == "installed"
    assert any("records no commit" in note for note in record["notes"])


def test_version_names_the_installed_commit(from_git: Path) -> None:
    done = _ask(from_git, "from orbitwin.cli import describe_build\nprint(describe_build())")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == (f"orbitwin {__version__}, installed, commit {COMMIT}, "
                                   "from https://github.com/PanNavi23/orbitwin")


def test_a_checkout_is_never_judged_as_installed() -> None:
    from orbitwin.layout import CHECKOUT

    assert CHECKOUT is not None


# -- review findings ----------------------------------------------------------------------


def _record_line(site: Path, name: str, line: str | None) -> None:
    """Replace RECORD's line for ``name`` with ``line`` (dropped when None)."""
    record = next(site.glob("orbitwin-*.dist-info")) / "RECORD"
    lines = [entry for entry in record.read_text(encoding="utf-8").splitlines()
             if not entry.startswith(name + ",")]
    record.write_text("\n".join([*lines, *([line] if line else [])]) + "\n", encoding="utf-8")


def _stops(site: Path, says: str) -> None:
    done = _ask(site, _PROVENANCE.format(expect=COMMIT))
    assert done.stdout.startswith("STOP"), done.stdout + done.stderr
    assert says in done.stdout, done.stdout


def test_a_missing_file_stops_the_day(from_git: Path) -> None:
    (from_git / "orbitwin" / "pareto.py").unlink()
    _stops(from_git, "orbitwin/pareto.py (missing)")


@pytest.mark.parametrize(("line", "says"), [
    ("orbitwin/pareto.py,,", "orbitwin/pareto.py (no hash in RECORD)"),
    ("orbitwin/pareto.py,md5=AAAA,1", "orbitwin/pareto.py (md5 in RECORD, not sha256"),
    ("orbitwin/pareto.py,sha999=AAAA,1", "orbitwin/pareto.py (sha999 in RECORD"),
])
def test_a_record_line_that_proves_nothing_stops_the_day(from_git: Path, line: str,
                                                         says: str) -> None:
    _record_line(from_git, "orbitwin/pareto.py", line)
    _stops(from_git, says)


def test_bytecode_outside_the_cache_stops_the_day(from_git: Path) -> None:
    """A sourceless package directory is imported before score.py."""
    (from_git / "orbitwin" / "score").mkdir()
    (from_git / "orbitwin" / "score" / "__init__.pyc").write_bytes(b"\0" * 16)
    _stops(from_git, "orbitwin/score/__init__.pyc (not installed)")


def test_a_linked_folder_stops_the_day(from_git: Path, tmp_path: Path) -> None:
    (tmp_path / "elsewhere").mkdir()
    (from_git / "orbitwin" / "score2").symlink_to(tmp_path / "elsewhere")
    _stops(from_git, "orbitwin/score2 (a link, not installed)")


def test_a_commit_id_that_is_not_a_commit_is_no_commit(tmp_path: Path) -> None:
    site = _site(tmp_path, {"url": "x", "vcs_info": {"vcs": "git",
                                                      "commit_id": COMMIT + "-and-more"}})
    _stops(site, "records no commit")


def test_a_git_repository_around_the_install_does_not_speak_for_it(from_git: Path) -> None:
    """--target into a repository: its HEAD is not what runs."""
    repo = from_git.parent
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    score = from_git / "orbitwin" / "score.py"
    score.write_text(score.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    _stops(from_git, "orbitwin/score.py (changed)")


def test_another_environments_record_is_not_this_ones(tmp_path: Path) -> None:
    """The package on the path, the only record another site's: no record of this one."""
    other = _site(tmp_path / "other", {"url": "x", "vcs_info": {"vcs": "git",
                                                                "commit_id": COMMIT}})
    shutil.rmtree(other / "orbitwin")
    bare = tmp_path / "bare"
    shutil.copytree(_site(tmp_path / "pkg", None) / "orbitwin", bare / "orbitwin")
    env = _env(f"{bare}{os.pathsep}{other}", fresh=tmp_path / "pyc")
    done = subprocess.run([sys.executable, "-s", "-c", _PROVENANCE.format(expect=COMMIT)],
                          cwd=tmp_path, env=env, capture_output=True, text=True,
                          check=False, timeout=300)
    assert "carries no installer's record of itself" in done.stdout, done.stdout + done.stderr


def test_a_stale_record_earlier_on_the_path_does_not_hide_this_one(from_git: Path,
                                                                   tmp_path: Path) -> None:
    stale = tmp_path / "stale"
    info = stale / "orbitwin-0.0.1.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: orbitwin\nVersion: 0.0.1\n",
                                   encoding="utf-8")
    (info / "RECORD").write_text("", encoding="utf-8")
    env = _env(f"{stale}{os.pathsep}{from_git}", fresh=tmp_path / "pyc")
    done = subprocess.run([sys.executable, "-s", "-c", _PROVENANCE.format(expect=COMMIT)],
                          cwd=tmp_path, env=env, capture_output=True, text=True,
                          check=False, timeout=300)
    assert json.loads(done.stdout)["on_main"] is True, done.stdout + done.stderr


# -- second review round ----------------------------------------------------------------


_TAG = sys.implementation.cache_tag


@pytest.mark.parametrize("mode", ["TIMESTAMP", "UNCHECKED_HASH"])
def test_planted_bytecode_never_runs_in_a_day_and_the_sources_are_proven(
    from_git: Path, mode: str,
) -> None:
    """A .pyc compiled from an edited source, the source put back: a fresh process
    compiles the source, so the plant is not run, and the files on disk are the record's."""
    import py_compile

    score = from_git / "orbitwin" / "score.py"
    original = score.read_bytes()
    stat = score.stat()
    score.write_bytes(original + b"\nPLANTED = 1\n")
    py_compile.compile(str(score), cfile=str(from_git / "orbitwin" / "__pycache__" /
                                              f"score.{_TAG}.pyc"), doraise=True,
                       invalidation_mode=getattr(py_compile.PycInvalidationMode, mode))
    score.write_bytes(original)
    os.utime(score, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    if mode == "TIMESTAMP":
        # The header Python checks a timestamp .pyc by: the source's mtime and size.
        pyc = from_git / "orbitwin" / "__pycache__" / f"score.{_TAG}.pyc"
        data = bytearray(pyc.read_bytes())
        data[8:12] = (int(stat.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little")
        data[12:16] = (len(original) & 0xFFFFFFFF).to_bytes(4, "little")
        pyc.write_bytes(bytes(data))
    # Outside a fresh cache Python runs the plant: what the check exists for.
    stale = _ask(from_git, "import orbitwin.score as s; print(hasattr(s, 'PLANTED'))",
                 fresh=False)
    assert stale.stdout.strip() == "True", stale.stdout + stale.stderr
    done = _ask(from_git, "import orbitwin.score as s; print(hasattr(s, 'PLANTED'))\n"
                + _PROVENANCE.format(expect=COMMIT))
    planted, record = done.stdout.split("\n", 1)
    assert planted == "False", done.stdout + done.stderr
    assert json.loads(record)["on_main"] is True


def test_a_process_not_compiled_afresh_is_refused(from_git: Path) -> None:
    done = _ask(from_git, _PROVENANCE.format(expect=COMMIT), fresh=False)
    assert done.stdout.startswith("STOP") and "compile afresh" in done.stdout, (
        done.stdout + done.stderr)


def test_direct_url_must_be_the_file_the_installer_wrote(tmp_path: Path) -> None:
    site = _site(tmp_path, {"url": "x", "vcs_info": {"vcs": "git", "commit_id": COMMIT}})
    info = next(site.glob("orbitwin-*.dist-info"))
    _record_line(site, f"{info.name}/direct_url.json", f"{info.name}/direct_url.json,,")
    _stops(site, "direct_url.json (no hash in RECORD)")
    digest = base64.urlsafe_b64encode(
        hashlib.sha256((info / "direct_url.json").read_bytes()).digest()).rstrip(b"=")
    _record_line(site, f"{info.name}/direct_url.json",
                 f"{info.name}/direct_url.json,sha256={digest.decode()},1")
    done = _ask(site, _PROVENANCE.format(expect=COMMIT))
    assert json.loads(done.stdout)["on_main"] is True, done.stdout + done.stderr
    (info / "direct_url.json").write_text(json.dumps(
        {"url": "x", "vcs_info": {"vcs": "git", "commit_id": "f" * 40}}), encoding="utf-8")
    _stops(site, "direct_url.json (changed)")


def test_a_clean_repository_at_the_expected_commit_around_the_install_proves_nothing(
    from_git: Path,
) -> None:
    """The case the old order passed: the outer repository's HEAD is the expected one."""
    repo = from_git.parent
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    environment = scratch_git_environment(repo)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=environment)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True,
                   env=environment)
    head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True, env=environment).stdout.strip()
    (repo / "BUILD_INFO.json").write_text(json.dumps({"commit": head}), encoding="utf-8")
    done = _ask(from_git, _PROVENANCE.format(expect=head))
    assert done.stdout.startswith("STOP"), done.stdout + done.stderr
    assert f"not the expected {head[:12]}" in done.stdout


def test_two_records_of_one_package_stop_the_day(from_git: Path) -> None:
    info = next(from_git.glob("orbitwin-*.dist-info"))
    shutil.copytree(info, from_git / "orbitwin-0.0.9.dist-info")
    _stops(from_git, "2 installer records for this one package")


def test_startup_files_beside_the_package_are_recorded(from_git: Path) -> None:
    (from_git / "zz.pth").write_text("import os\n", encoding="utf-8")
    done = _ask(from_git, _PROVENANCE.format(expect=COMMIT))
    assert "zz.pth" in json.loads(done.stdout)["startup_files"], done.stdout + done.stderr


def test_an_installed_day_runs_itself_again_with_an_empty_bytecode_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orbitwin import cli

    seen: list[tuple[str, list[str], dict[str, str]]] = []

    def execve(path: str, argv: list[str], env: dict[str, str]) -> None:
        seen.append((path, argv, env))
        raise SystemExit(0)

    monkeypatch.setattr(cli, "CHECKOUT", None)
    monkeypatch.delenv(FRESH_PYCACHE, raising=False)
    monkeypatch.setattr(cli.os, "execve", execve)
    for command in (["rules"], ["status", "x"], ["--version"]):
        cli.compile_afresh(command)
    assert seen == []
    with pytest.raises(SystemExit):
        cli.compile_afresh(["run", "contest", "--confirm-readback", "2"])
    path, argv, env = seen[0]
    assert path == sys.executable
    assert argv == [sys.executable, "-P", "-m", "orbitwin", "run", "contest",
                    "--confirm-readback", "2"]
    fresh = Path(env["PYTHONPYCACHEPREFIX"])
    assert env[FRESH_PYCACHE] == str(fresh) and fresh.is_dir() and not any(fresh.iterdir())
    fresh.rmdir()



# -- fourth review round ----------------------------------------------------------------


def test_a_stray_fresh_cache_variable_removes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from orbitwin import cli

    victim = tmp_path / "victim"
    (victim / "sub").mkdir(parents=True)
    registered: list[object] = []
    monkeypatch.setattr(cli, "CHECKOUT", None)
    monkeypatch.setenv(FRESH_PYCACHE, str(victim))
    monkeypatch.setattr(cli.atexit, "register", lambda *call: registered.append(call))
    cli.compile_afresh(["day", "--help"])
    assert registered == [] and (victim / "sub").is_dir()


def test_a_cache_that_cannot_be_made_is_a_clean_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from orbitwin import cli

    def refuse(prefix: str) -> str:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cli, "CHECKOUT", None)
    monkeypatch.delenv(FRESH_PYCACHE, raising=False)
    monkeypatch.setattr(cli.tempfile, "mkdtemp", lambda prefix: refuse(prefix))
    monkeypatch.setattr(cli, "_sweep_stale_caches", lambda: None)
    with pytest.raises(SystemExit) as stopped:
        cli.compile_afresh(["run", "contest", "--json"])
    assert stopped.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "io"


def test_only_this_users_stale_caches_are_swept(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: Path) -> None:
    from orbitwin import cli

    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))
    old, new, other = (tmp_path / f"orbitwin-pyc-{name}" for name in ("old", "new", "x"))
    for path in (old, new):
        path.mkdir()
    (tmp_path / "elsewhere").mkdir()
    other.symlink_to(tmp_path / "elsewhere")
    os.utime(old, (1, 1))
    os.utime(other, (1, 1), follow_symlinks=False)
    cli._sweep_stale_caches()
    assert not old.exists() and new.is_dir() and other.is_symlink()
    assert (tmp_path / "elsewhere").is_dir()


def test_version_of_a_clean_install_names_no_difference(from_git: Path) -> None:
    done = _ask(from_git, "from orbitwin.cli import describe_build\nprint(describe_build())",
                fresh=False)
    assert "differ" not in done.stdout, done.stdout + done.stderr


# -- fifth review round ------------------------------------------------------------------


def test_the_sweep_passes_by_another_users_cache_and_a_live_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from orbitwin import cli

    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))
    theirs, live = tmp_path / "orbitwin-pyc-theirs", tmp_path / "orbitwin-pyc-live"
    for path in (theirs, live):
        path.mkdir()
    held = cli._hold(live)
    for path in (theirs, live):
        os.utime(path, (1, 1))
    real = os.getuid()
    monkeypatch.setattr(cli.os, "getuid", lambda: real + 1)
    cli._sweep_stale_caches()
    assert theirs.is_dir()
    monkeypatch.setattr(cli.os, "getuid", lambda: real)
    cli._sweep_stale_caches()
    assert live.is_dir() and not theirs.exists()
    held.close()
    os.utime(live, (1, 1))
    cli._sweep_stale_caches()
    assert not live.exists()


def test_a_cache_others_can_write_is_not_fresh(from_git: Path, tmp_path: Path) -> None:
    cache = tmp_path / "open"
    cache.mkdir()
    cache.chmod(0o777)
    env = _env(str(from_git), fresh=None)
    env.update(PYTHONPYCACHEPREFIX=str(cache), **{FRESH_PYCACHE: str(cache)})
    done = subprocess.run([sys.executable, "-s", "-c", _PROVENANCE.format(expect=COMMIT)],
                          cwd=tmp_path, env=env, capture_output=True, text=True,
                          check=False, timeout=300)
    assert done.stdout.startswith("STOP"), done.stdout + done.stderr
    assert "unset it" in done.stdout
