"""The runbook's later steps call python and orbitwin by name.

Followed as written from an unpacked release with no network, step 0 installed
into .venv and never activated it, so step 2's `python` was the system
interpreter (ModuleNotFoundError: scipy) and step 3's `orbitwin` was not found.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import repo_files

#: `.`, which dash has; `source` is a bash word, and dash goes on without the venv.
ACTIVATE = re.compile(r"(?:^|&&|;)\s*\.\s+(?:\./)?\.venv/bin/activate\s*(?:&&|;|$)")
#: python, python3, orbitwin or make run from PATH, after any VAR=value words.
_FROM_PATH = re.compile(r"(?:^|&&|;|\|)\s*(?:[A-Za-z_]+=\S*\s+)*(?:python3?|orbitwin|make)\s")
#: The two that run before there is an environment: one at home, one making it.
_BEFORE = {"scripts/fetch_offline_wheels.sh", "scripts/bootstrap_offline.sh"}


def _code(line: str) -> str:
    return line.split(" #", 1)[0].rstrip()


def _bash_blocks(text: str) -> list[list[str]]:
    return [[_code(line) for line in block.splitlines()]
            for block in re.findall(r"```bash\n(.*?)```", text, re.S)]


def _needs_the_environment(command: str) -> bool:
    if _FROM_PATH.search(command):
        return True
    return any(name not in _BEFORE for name in re.findall(r"scripts/\w+\.(?:sh|py)", command))


def _commands(line: str) -> list[str]:
    return [part for part in re.split(r"&&|;", line) if part.strip()]


def _check_order(block: list[str]) -> None:
    """Nothing that needs the venv between the bootstrap and the activation."""
    commands = [command for line in block for command in _commands(line)]
    bootstrap = next(i for i, c in enumerate(commands) if "bootstrap_offline.sh" in c)
    activate = next(i for i, c in enumerate(commands) if ACTIVATE.search(c))
    assert bootstrap < activate, "activate after the bootstrap"
    for command in commands[bootstrap + 1:activate]:
        assert not _needs_the_environment(command), f"runs before the activation: {command}"
    for command in commands[activate + 1:]:
        if _needs_the_environment(command):
            return
    pytest.fail("nothing after the activation uses it")


def _runbook(project_root: Path) -> list[list[str]]:
    return _bash_blocks((project_root / "docs" / "RUNBOOK.md").read_text(encoding="utf-8"))


def test_step_zero_activates_right_after_the_bootstrap(project_root: Path) -> None:
    install = _runbook(project_root)[0]
    assert any("bootstrap_offline.sh" in line for line in install), "step 0 is the first block"

    _check_order(install)
    line = next(line for line in install if "bootstrap_offline.sh" in line)
    assert re.search(r"bootstrap_offline\.sh\s*&&\s*\.\s+(?:\./)?\.venv/bin/activate", line), (
        "activate only if the bootstrap succeeded"
    )


def test_step_zero_runs_the_suite_only_inside_the_environment(project_root: Path) -> None:
    install = "\n".join(_runbook(project_root)[0])

    assert re.search(r"orbitwin --version[^\n]*&&\s*PYTHONPATH=src python -m pytest", install)


#: Stand-ins for what step 0 runs, each leaving a mark; `orbitwin` names $COMMIT.
_STEP_ZERO_STUBS = {
    "bin/sha256sum": 'touch "$MARKS/sums"; exit "${SUMS_RC:-0}"',
    "bin/python3": 'touch "$MARKS/tree"; exit "${TREE_RC:-0}"',
    "root/scripts/bootstrap_offline.sh": 'touch "$MARKS/bootstrap"; exit "${BOOT_RC:-0}"',
    "root/.venv/bin/activate": 'PATH="$ROOT/.venv/bin:$PATH"',
    "root/.venv/bin/orbitwin": 'echo "orbitwin 0.3.0, commit $COMMIT, committed 2026-11-04T00:00:00Z"',
    "root/.venv/bin/python": 'touch "$MARKS/suite"',
}


@pytest.mark.parametrize(
    ("failure", "reached"),
    [
        (None, {"sums", "tree", "bootstrap", "suite"}),
        ("manifest", {"sums"}),
        ("unlisted file", {"sums", "tree"}),
        ("bootstrap", {"sums", "tree", "bootstrap"}),
        ("another commit", {"sums", "tree", "bootstrap"}),
        ("placeholder left", {"sums", "tree", "bootstrap"}),
    ],
)
def test_step_zero_stops_at_the_first_failure(
    project_root: Path, tmp_path: Path, failure: str | None, reached: set[str]
) -> None:
    """Its lines ran one after another: a failed check went on to install and test anyway,
    and nothing compared the build with the one tested."""
    expected = "0123456789ab"
    for name, body in _STEP_ZERO_STUBS.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        path.chmod(0o755)
    (tmp_path / "root" / "vendor" / "wheels").mkdir(parents=True)
    (tmp_path / "marks").mkdir()
    block = "\n".join(_runbook(project_root)[0])
    assert block.count("XXXXXXXXXXXX") == 1, "one placeholder, for the commit"
    if failure != "placeholder left":
        block = block.replace("XXXXXXXXXXXX", expected)

    subprocess.run(
        ["bash", "-c", block], cwd=tmp_path / "root", capture_output=True, text=True, check=False,
        env={"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "MARKS": str(tmp_path / "marks"),
             "ROOT": str(tmp_path / "root"),
             "SUMS_RC": "1" if failure == "manifest" else "0",
             "TREE_RC": "1" if failure == "unlisted file" else "0",
             "BOOT_RC": "1" if failure == "bootstrap" else "0",
             "COMMIT": ("fedcba987654" if failure == "another commit" else expected) + "0" * 28},
    )

    assert {path.name for path in (tmp_path / "marks").iterdir()} == reached


def test_the_runbook_says_how_to_start_a_terminal(project_root: Path) -> None:
    blocks = _runbook(project_root)[1:]
    setup = [block for block in blocks if any(ACTIVATE.search(line) for line in block)]
    assert setup, "a block that starts a new terminal"
    first = next(line for line in setup[0] if line.strip())
    assert re.match(r"cd \S.*&&\s*\.\s+(?:\./)?\.venv/bin/activate", first)
    assert any(line.startswith("export GMAT_CONSOLE=") for line in setup[0])


@pytest.mark.parametrize("readme_name", ["README.md", "README.zh-TW.md"])
@pytest.mark.parametrize("marker", ["bootstrap_offline.sh", "ranking_day_config.py"])
def test_the_readme_activates_too(project_root: Path, marker: str, readme_name: str) -> None:
    readme = (project_root / readme_name).read_text(encoding="utf-8")
    block = next(block for block in _bash_blocks(readme) if any(marker in x for x in block))
    commands = [command for line in block for command in _commands(line)]
    activate = next(i for i, c in enumerate(commands) if ACTIVATE.search(c))
    if marker == "bootstrap_offline.sh":
        bootstrap = next(i for i, c in enumerate(commands) if marker in c)
        assert bootstrap < activate
    for command in commands[:activate]:
        assert not _needs_the_environment(command), f"README runs before activating: {command}"


def test_bootstrap_installs_where_the_runbook_activates_and_says_so(project_root: Path) -> None:
    script = (project_root / "scripts" / "bootstrap_offline.sh").read_text(encoding="utf-8")
    assert 'VENV="${VENV_DIR:-.venv}"' in script
    assert 'PYTHON="${PYTHON:-python3}"' in script, "the runbook names it PYTHON="
    assert 'cd "$(dirname "$0")/.."' in script
    assert 'echo "  cd $(quote "$(pwd)") && . $(quote "$VENV")/bin/activate"' in script


@pytest.mark.parametrize("name", ["with space", "it's", "\u4e0b\u8f09"])
def test_the_printed_activation_survives_dash(
    project_root: Path, tmp_path: Path, name: str
) -> None:
    """A path with a space broke the cd, and the && then skipped the activation.

    printf %q fixed that for bash only: in the C locale it writes $'...' for a
    non-ASCII path, which dash cannot read.
    """
    if shutil.which("dash") is None:
        pytest.skip("dash is not installed")
    script = (project_root / "scripts" / "bootstrap_offline.sh").read_text(encoding="utf-8")
    quote = next(line for line in script.splitlines() if line.startswith("quote() {"))
    directory = tmp_path / name
    directory.mkdir()
    printed = subprocess.run(
        ["bash", "-c", f'{quote}\nquote "$1"', "bash", str(directory)],
        capture_output=True, text=True, check=True, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    ).stdout
    landed = subprocess.run(
        ["dash", "-c", f"cd {printed} && pwd"], capture_output=True, text=True, check=False,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    assert landed.returncode == 0, landed.stderr
    assert landed.stdout.strip() == str(directory)


@pytest.mark.parametrize(
    ("path", "lines"),
    [("README.md", ("uv pip install -r requirements-dev.txt && uv pip install --no-deps -e .",)),
     ("README.zh-TW.md",
      ("uv pip install -r requirements-dev.txt && uv pip install --no-deps -e .",)),
     ("scripts/bootstrap_linux.sh",
      ("python -m pip install -r requirements-dev.txt", "python -m pip install --no-deps -e ."))],
)
def test_the_development_installs_take_the_pins(
    project_root: Path, path: str, lines: tuple[str, ...]
) -> None:
    """`-e '.[dev]'` resolved numpy 2.5.4 and SciPy 1.18.1 on 3.13, which orbitwin refuses."""
    text = (project_root / path).read_text(encoding="utf-8")
    assert all(line in text for line in lines)
    assert "-e '.[dev]'" not in text


def _versions(project_root: Path) -> list[str]:
    fetch = (project_root / "scripts" / "fetch_offline_wheels.sh").read_text(encoding="utf-8")
    found = re.search(r'VERSIONS="\$\{WHEEL_PYTHON_VERSIONS:-([0-9. ]+)\}"', fetch)
    assert found
    return found.group(1).split()


#: Each names a range; any other Markdown file at the top level is held to it when it does.
_RANGED = ("README.md", "README.zh-TW.md", "docs/RUNBOOK.md", "vendor/wheels/README.md")


def _top_level_markdown() -> list[str]:
    """What git tracks there, or in an unpacked archive the files there: never a stray note."""
    root = Path(__file__).resolve().parents[2]
    return [path.name for path in repo_files(root, "*.md", tracked_only=True)
            if path.parent == root]


@pytest.mark.parametrize("document", sorted({*_RANGED, *_top_level_markdown()}))
def test_every_python_range_ends_where_the_bundle_does(project_root: Path, document: str) -> None:
    covered = _versions(project_root)
    text = (project_root / document).read_text(encoding="utf-8")
    ranges = re.findall(r"3\.11\s*(?:-|\u2013|to|through|\u5230)\s*3\.(\d+)", text)
    ranges += re.findall(r"3\.11, 3\.12 and 3\.(\d+)", text)

    assert ranges or document not in _RANGED
    assert {f"3.{minor}" for minor in ranges} <= {covered[-1]}


def test_the_runbook_names_the_pythons_the_bundle_covers(project_root: Path) -> None:
    covered = _versions(project_root)
    runbook = (project_root / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    assert f"Python {covered[0]} to {covered[-1]}" in runbook


def _wheel(
    directory: Path, name: str, version: str, tags: str, build: str = "", summary: str = ""
) -> Path:
    """A wheel with metadata and nothing else: enough for pip to resolve and install."""
    import zipfile

    info = f"{name}-{version}.dist-info"
    files = {
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                            f"Summary: {summary}\n",
        f"{info}/WHEEL": f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: {tags}\n",
    }
    path = directory / f"{name}-{version}-{build + '-' if build else ''}{tags}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member, text in files.items():
            archive.writestr(member, text)
        archive.writestr(f"{info}/RECORD", "".join(f"{member},,\n" for member in [*files, f"{info}/RECORD"]))
    return path


def _fake_bundle(
    tmp_path: Path, project_root: Path, numpy_tag: str, numpy_tags: str | None = None
) -> Path:
    """numpy, setuptools and wheel as metadata-only wheels, pinned and required as the real ones.

    Installing them succeeds; the editable install of a tree with no project
    then fails, after the venv step.
    """
    root = tmp_path / "tree"
    (root / "scripts").mkdir(parents=True)
    for name in ("bootstrap_offline.sh", "wheel_requirements.py"):
        shutil.copy2(project_root / "scripts" / name, root / "scripts")
    wheels = root / "vendor" / "wheels"
    wheels.mkdir(parents=True)
    _wheel(wheels, "numpy", "2.4.6", numpy_tags or f"{numpy_tag}-{numpy_tag}-linux_x86_64")
    _wheel(wheels, "setuptools", "84.2.1", "py3-none-any")
    _wheel(wheels, "wheel", "0.48.0", "py3-none-any")
    (wheels / "constraints.txt").write_text(
        "numpy==2.4.6  # pinned\nsetuptools==84.2.1\nwheel==0.48.0\n", encoding="utf-8"
    )
    (root / "requirements-dev.txt").write_text(
        "-c vendor/wheels/constraints.txt\nnumpy==2.4.6\n", encoding="utf-8")
    _list_bundle(root)
    return root


def _list_bundle(root: Path) -> None:
    """SHA256SUMS.txt for every wheel there, as the fetch writes it; a pip wheel is pinned too."""
    import hashlib

    wheels = root / "vendor" / "wheels"
    found = sorted(wheels.glob("*.whl"))
    (wheels / "SHA256SUMS.txt").write_text(
        "".join(f"{hashlib.sha256(w.read_bytes()).hexdigest()}  ./{w.name}\n" for w in found),
        encoding="utf-8",
    )
    constraints = wheels / "constraints.txt"
    pip = [w for w in found if w.name.startswith("pip-")]
    if pip and "pip==" not in constraints.read_text(encoding="utf-8"):
        with constraints.open("a", encoding="utf-8") as handle:
            handle.write(f"pip=={pip[0].name.split('-')[1]}\n")


def test_bootstrap_refuses_a_python_the_bundle_has_no_wheels_for(
    project_root: Path, tmp_path: Path
) -> None:
    """It used to make the venv and then fail on numpy, which reads as a broken bundle."""
    import sys

    root = _fake_bundle(tmp_path, project_root, "cp39")
    completed = subprocess.run(
        ["bash", "scripts/bootstrap_offline.sh"], cwd=root, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable}, check=False,
    )

    assert completed.returncode == 1
    tag = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    assert f"no {tag} numpy" in completed.stderr
    assert "PYTHON=python3.11" in completed.stderr
    assert not (root / ".venv").exists(), "refused before making anything"


def _without_ensurepip(tmp_path: Path) -> Path:
    """This interpreter, behaving as Debian's does without python3.X-venv.

    `import ensurepip` fails, and so does any venv that would need it: one
    made without --without-pip.
    """
    import sys

    stripped = tmp_path / "python"
    stripped.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ] && [ "$2" = "import ensurepip" ]; then exit 1; fi\n'
        'if [ "$1" = "-m" ] && [ "$2" = "ensurepip" ]; then exit 1; fi\n'
        'if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then\n'
        '  case " $* " in *" --without-pip "*) ;; *) echo "ensurepip is not available" >&2; exit 1;; esac\n'
        "fi\n"
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    stripped.chmod(0o755)
    return stripped


def _bundled_pip_wheel(project_root: Path) -> Path | None:
    """A pip wheel for the fallback: ensurepip's own, or the bundle's."""
    import importlib.util

    spec = importlib.util.find_spec("ensurepip")
    places = [Path(spec.origin).parent / "_bundled"] if spec and spec.origin else []
    places.append(project_root / "vendor" / "wheels")
    found = [wheel for place in places for wheel in place.glob("pip-*-py3-none-any.whl")]
    return max(found, key=_pip_version, default=None)


def _pip_version(wheel: Path) -> tuple[int, ...]:
    return tuple(int(part) for part in wheel.name.split("-")[1].split(".") if part.isdigit())


def _venv_ready(root: Path, project_root: Path) -> None:
    """Make the fake bundle able to make a venv with this interpreter, whatever it lacks.

    On a Python without ensurepip (Debian's without python3.X-venv) the
    bootstrap needs a pip wheel, and these tests are about what happens after.
    """
    import importlib.util

    if importlib.util.find_spec("ensurepip") is not None:
        return
    wheel = _bundled_pip_wheel(project_root)
    if wheel is None:
        pytest.skip("this Python has no ensurepip and no pip wheel is at hand")
    shutil.copy2(wheel, root / "vendor" / "wheels")


def _bootstrap(
    root: Path, python: Path | str, listed: bool = True, **environment: str
) -> subprocess.CompletedProcess[str]:
    """Run it; `listed` first lists every wheel the test put in the bundle."""
    if listed:
        _list_bundle(root)
    return subprocess.run(
        ["bash", "scripts/bootstrap_offline.sh"], cwd=root, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHON": str(python), **environment}, check=False,
    )


def _platform_tag(other: bool = False) -> str:
    import sysconfig

    tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    if other:
        tag = tag.replace("x86_64", "aarch64") if "x86_64" in tag else "linux_x86_64"
    return tag


@pytest.mark.parametrize(
    ("wheel", "refused"),
    [("{cp}-{other_abi}-{platform}", True), ("{cp}-{abi}-{elsewhere}", True),
     ("{cp}-{abi}-{platform}", False)],
    ids=["free-threaded-wheel", "another-machine", "this-python"],
)
def test_bootstrap_refuses_wheels_this_python_cannot_install(
    project_root: Path, tmp_path: Path, wheel: str, refused: bool
) -> None:
    """The version matched, so 3.14t and an aarch64 machine reached pip's failure.

    Measured with the real bundle: python3.14t cleared a working venv first.
    The interpreter cannot be swapped here, so the wheels are: the same
    mismatch, seen from the other side.
    """
    import sys
    import sysconfig

    pip_wheel = _bundled_pip_wheel(project_root)
    if pip_wheel is None:
        pytest.skip("no pip wheel at hand")
    cp = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    tags = wheel.format(
        cp=cp, abi=cp + "t" * threaded, other_abi=cp + "t" * (not threaded),
        platform=_platform_tag(), elsewhere=_platform_tag(other=True),
    )
    root = _fake_bundle(tmp_path, project_root, cp, numpy_tags=tags)
    shutil.copy2(pip_wheel, root / "vendor" / "wheels")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root / ".venv")],
                   check=True, capture_output=True)
    marker = root / ".venv" / "marker"
    marker.write_text("x\n", encoding="utf-8")

    completed = _bootstrap(root, sys.executable)

    assert ("has none for: numpy." in completed.stderr) is refused, completed.stderr
    assert "cannot install from" not in completed.stderr, completed.stderr
    assert "checked the Python version only" not in completed.stdout
    if refused:
        assert completed.returncode == 1
        assert f"installs {cp}-" in completed.stderr
        assert "docs/RUNBOOK.md step 0" in completed.stderr
        assert marker.exists(), "refused before the venv"


def test_a_listed_wheel_the_install_never_reads_is_not_refused(
    project_root: Path, tmp_path: Path
) -> None:
    """The bundle holds wheels for four Pythons; the tag check refuses only for wanted ones."""
    import sys

    pip_wheel = _bundled_pip_wheel(project_root)
    if pip_wheel is None:
        pytest.skip("no pip wheel at hand")
    cp = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    root = _fake_bundle(tmp_path, project_root, cp, numpy_tags=f"{cp}-{cp}-{_platform_tag()}")
    shutil.copy2(pip_wheel, root / "vendor" / "wheels")
    _wheel(root / "vendor" / "wheels", "zzz_unused", "1.0", f"cp30-cp30-{_platform_tag(other=True)}")

    completed = _bootstrap(root, sys.executable)

    assert "has none for" not in completed.stderr, completed.stderr
    assert "cannot install from" not in completed.stderr, completed.stderr
    assert (root / ".venv").exists(), "went on to the venv"


@pytest.mark.parametrize("damage", ["no manifest", "unlisted", "sdist", "absent", "changed"])
def test_bootstrap_installs_only_the_listed_wheels(
    project_root: Path, tmp_path: Path, damage: str
) -> None:
    """With no SHA256SUMS.txt, or an unlisted numpy beside the listed one, a tampered wheel was
    installed: pip prefers a build tag. Every damage is refused before the venv."""
    import sys

    cp = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    root = _fake_bundle(tmp_path, project_root, cp, numpy_tags=f"{cp}-{cp}-{_platform_tag()}")
    _venv_ready(root, project_root)
    _list_bundle(root)
    wheels = root / "vendor" / "wheels"
    listed = next(wheels.glob("numpy-*.whl"))
    if damage == "no manifest":
        (wheels / "SHA256SUMS.txt").unlink()
    elif damage == "unlisted":
        _wheel(wheels, "numpy", "2.4.6", f"{cp}-{cp}-{_platform_tag()}", build="1",
               summary="tampered")
    elif damage == "sdist":
        (wheels / "numpy-2.4.6.tar.gz").write_bytes(b"an sdist")
    elif damage == "absent":
        listed.unlink()
    else:
        listed.write_bytes(listed.read_bytes() + b"tampered")

    completed = _bootstrap(root, sys.executable, listed=False)

    assert completed.returncode == 1, completed.stdout + completed.stderr
    message = {
        "no manifest": "vendor/wheels/SHA256SUMS.txt is missing",
        "unlisted": "does not list, which pip could install in place of the listed ones: "
                    f"numpy-2.4.6-1-{cp}",
        "sdist": "in place of the listed ones: numpy-2.4.6.tar.gz",
        "absent": f"lacks wheels SHA256SUMS.txt lists: {listed.name}",
        "changed": "is not the one SHA256SUMS.txt lists",
    }[damage]
    assert message in completed.stderr
    assert not (root / ".venv").exists(), "refused before making anything"


def test_pip_refuses_a_wheel_whose_hash_is_not_listed(project_root: Path, tmp_path: Path) -> None:
    """pip checks the hashes itself: a numpy that appears after the bundle was checked,
    under a build tag pip prefers, is refused, not installed."""
    import sys

    cp = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    root = _fake_bundle(tmp_path, project_root, cp, numpy_tags=f"{cp}-{cp}-{_platform_tag()}")
    _venv_ready(root, project_root)
    _list_bundle(root)
    late = _wheel(tmp_path, "numpy", "2.4.6", f"{cp}-{cp}-{_platform_tag()}", build="1",
                  summary="tampered")
    # The interpreter, which drops the wheel in once the hashed requirements are written.
    python = tmp_path / "python"
    python.write_text(
        f'#!/bin/sh\n"{sys.executable}" "$@" || exit $?\n'
        f'case " $* " in *" --write "*) cp "{late}" vendor/wheels/ ;; esac\n',
        encoding="utf-8",
    )
    python.chmod(0o755)

    completed = _bootstrap(root, python, listed=False)

    assert completed.returncode != 0
    assert "DO NOT MATCH THE HASHES" in completed.stderr, completed.stderr
    installed = list((root / ".venv").glob("lib/python*/site-packages/numpy-*.dist-info/METADATA"))
    assert not [path for path in installed if "tampered" in path.read_text(encoding="utf-8")]


@pytest.mark.parametrize(
    "setting", [{"PIP_REQUIRE_VIRTUALENV": "1"}, {"TMPDIR": "/nonexistent/tmp"}],
    ids=["require-virtualenv", "tmpdir-gone"],
)
def test_a_dry_run_that_fails_on_the_shells_settings_does_not_refuse(
    project_root: Path, tmp_path: Path, setting: dict[str, str]
) -> None:
    """Outside the venv the dry run met settings the install inside it does not mind.

    Both refused a Python the install on main accepted, with the runbook's stop rule.
    """
    import sys

    pip_wheel = _bundled_pip_wheel(project_root)
    if pip_wheel is None:
        pytest.skip("no pip wheel at hand")
    cp = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    root = _fake_bundle(tmp_path, project_root, cp, numpy_tags=f"{cp}-{cp}-{_platform_tag()}")
    shutil.copy2(pip_wheel, root / "vendor" / "wheels")

    completed = _bootstrap(root, sys.executable, **setting)

    assert "docs/RUNBOOK.md step 0" not in completed.stderr, completed.stderr
    assert (root / ".venv").exists(), "went on to the venv"


def test_a_sitecustomize_that_prints_is_not_read_as_the_tag(
    project_root: Path, tmp_path: Path
) -> None:
    import sys

    root = _fake_bundle(tmp_path, project_root, f"cp{sys.version_info[0]}{sys.version_info[1]}")
    noisy = tmp_path / "python"
    noisy.write_text(f'#!/bin/sh\necho "a sitecustomize line"\nexec "{sys.executable}" "$@"\n',
                     encoding="utf-8")
    noisy.chmod(0o755)

    completed = _bootstrap(root, noisy)

    assert "has no wheels in" not in completed.stderr, completed.stderr


def test_bootstrap_says_when_it_cannot_check_the_tags(project_root: Path, tmp_path: Path) -> None:
    import sys

    root = _fake_bundle(tmp_path, project_root, f"cp{sys.version_info[0]}{sys.version_info[1]}")

    completed = _bootstrap(root, sys.executable)

    assert "no pip wheel in vendor/wheels: checked the Python version only" in completed.stdout


def test_a_free_threaded_python_is_refused(project_root: Path, tmp_path: Path) -> None:
    """With a real free-threaded build, when one is installed.

    Needs a pip whose packaging knows the t ABI (24.1 on; 3.11's ensurepip has
    24.0). The bundle's is 26.2.2.
    """
    found = [shutil.which(f"python3.{minor}t") for minor in (13, 14)]
    if shutil.which("uv") is not None:
        located = subprocess.run(
            ["uv", "python", "find", "--no-python-downloads", "cpython-3.14+freethreaded"],
            capture_output=True, text=True, check=False,
        )
        found.append(located.stdout.strip() if located.returncode == 0 else None)
    python = next((path for path in found if path), None)
    pip_wheel = _bundled_pip_wheel(project_root)
    if python is None or pip_wheel is None:
        pytest.skip("no free-threaded Python or no pip wheel at hand")
    if _pip_version(pip_wheel) < (24, 1):
        pytest.skip(f"{pip_wheel.name} predates free-threaded wheel tags")
    minor = subprocess.run([python, "-c", "import sys; print(sys.version_info[1])"],
                           capture_output=True, text=True, check=True).stdout.strip()
    root = _fake_bundle(tmp_path, project_root, f"cp3{minor}")
    shutil.copy2(pip_wheel, root / "vendor" / "wheels")

    completed = _bootstrap(root, python)

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert f"installs cp3{minor}-cp3{minor}t-" in completed.stderr
    assert not (root / ".venv").exists()


def test_bootstrap_refuses_no_ensurepip_only_without_a_wheel_to_replace_it(
    project_root: Path, tmp_path: Path
) -> None:
    """Debian and Ubuntu ship ensurepip as python3.X-venv; without it venv fails half-made."""
    import sys

    tag = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    root = _fake_bundle(tmp_path, project_root, tag)

    completed = _bootstrap(root, _without_ensurepip(tmp_path))

    assert completed.returncode == 1
    assert "has no ensurepip, and vendor/wheels holds no" in completed.stderr
    assert not (root / ".venv").exists()


def test_a_directory_that_is_not_a_venv_is_refused_not_cleared(
    project_root: Path, tmp_path: Path
) -> None:
    import sys

    root = _fake_bundle(tmp_path, project_root, f"cp{sys.version_info[0]}{sys.version_info[1]}")
    _venv_ready(root, project_root)
    kept = root / ".venv" / "run" / "candidates.json"
    kept.parent.mkdir(parents=True)
    kept.write_text("{}\n", encoding="utf-8")

    completed = _bootstrap(root, sys.executable)

    assert completed.returncode == 1
    assert "is not a virtual environment" in completed.stderr
    assert kept.read_text(encoding="utf-8") == "{}\n"


@pytest.mark.parametrize(
    "state",
    ["same-python", "same-python-by-symlink", "uv-made", "interpreter-gone", "another-python",
     "noisy"],
)
def test_a_venv_is_made_again_only_when_it_cannot_run_this_python(
    project_root: Path, tmp_path: Path, state: str
) -> None:
    """A running solve imports from .venv; a venv on another or a missing interpreter is remade.

    Asked of the venv's own interpreter: uv writes `version_info = 3.13` where
    CPython writes `version = 3.13.14`, and neither says whether the
    interpreter is still there. "noisy" prints a line before answering, as a
    sitecustomize can. The fake bundle makes the install fail after the venv
    step, which is all this looks at.
    """
    import sys

    root = _fake_bundle(tmp_path, project_root, f"cp{sys.version_info[0]}{sys.version_info[1]}")
    _venv_ready(root, project_root)
    venv = root / ".venv"
    if state == "uv-made":
        if shutil.which("uv") is None:
            pytest.skip("uv is not installed")
        subprocess.run(["uv", "venv", "--quiet", "--python", sys.executable, str(venv)],
                       check=True, capture_output=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)],
                       check=True, capture_output=True)
    replacement = {
        "interpreter-gone": None,
        "another-python": '#!/bin/sh\necho "3.10.0 /opt/other/bin/python3.10"\n',
        "noisy": f'#!/bin/sh\necho "a sitecustomize line"\nexec "{sys.executable}" "$@"\n',
    }
    if state in replacement:
        for name in ("python", "python3", f"python3.{sys.version_info[1]}"):
            link = venv / "bin" / name
            if link.is_symlink() or link.exists():
                link.unlink()
                if replacement[state] is None:
                    link.symlink_to(tmp_path / "gone" / "python")
                else:
                    link.write_text(replacement[state], encoding="utf-8")
                    link.chmod(0o755)
    marker = venv / "marker"
    marker.write_text("x\n", encoding="utf-8")
    kept = state in {"same-python", "same-python-by-symlink", "uv-made", "noisy"}
    python: Path | str = sys.executable
    if state == "same-python-by-symlink":
        # As Debian's python3 -> python3.14: the same interpreter by another path.
        python = tmp_path / "python3"
        python.symlink_to(getattr(sys, "_base_executable", sys.executable))

    completed = _bootstrap(root, python)

    assert marker.exists() is kept, completed.stdout + completed.stderr
    assert ("making it again for" in completed.stdout) is not kept
    if state == "interpreter-gone":
        assert "no working interpreter" in completed.stdout
    if state == "another-python":
        assert "runs 3.10.0 /opt/other/bin/python3.10" in completed.stdout


def test_a_pyvenv_cfg_without_home_is_refused_not_cleared(
    project_root: Path, tmp_path: Path
) -> None:
    import sys

    root = _fake_bundle(tmp_path, project_root, f"cp{sys.version_info[0]}{sys.version_info[1]}")
    _venv_ready(root, project_root)
    kept = root / ".venv" / "runs" / "candidates.json"
    kept.parent.mkdir(parents=True)
    kept.write_text("{}\n", encoding="utf-8")
    (root / ".venv" / "pyvenv.cfg").write_text("include-system-site-packages = false\n",
                                               encoding="utf-8")

    completed = _bootstrap(root, sys.executable)

    assert completed.returncode == 1
    assert "is not a virtual environment" in completed.stderr
    assert kept.read_text(encoding="utf-8") == "{}\n"


def test_without_ensurepip_pip_comes_from_a_wheel(project_root: Path, tmp_path: Path) -> None:
    """A venv without pip, then pip from its own wheel, offline.

    The wheel is ensurepip's own when this Python has one, else the bundle's,
    so this runs in CI too.
    """
    import sys

    wheel = _bundled_pip_wheel(project_root)
    if wheel is None:
        pytest.skip("no pip wheel at hand")
    tag = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    root = _fake_bundle(tmp_path, project_root, tag)
    shutil.copy2(wheel, root / "vendor" / "wheels")

    completed = _bootstrap(root, _without_ensurepip(tmp_path))

    assert "no ensurepip in" in completed.stdout, completed.stderr
    pip = root / ".venv" / "bin" / "pip"
    assert pip.exists(), completed.stderr
    # The fake numpy wheel then stops the install; pip itself is in place.
    version = subprocess.run(
        [str(root / ".venv" / "bin" / "python"), "-m", "pip", "--version"],
        capture_output=True, text=True, check=False,
    )
    assert version.returncode == 0, version.stderr
