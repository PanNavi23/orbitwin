"""scripts/wheel_requirements.py: the bundle is its manifest, and pip gets the hashes."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

NUMPY = ("numpy-2.4.6-cp311-cp311-manylinux_2_28_x86_64.whl",
         "numpy-2.4.6-cp314-cp314-manylinux_2_28_x86_64.whl")
PIP = "pip-26.2.2-py3-none-any.whl"
SETUPTOOLS = "setuptools-84.2.1-py3-none-any.whl"


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def _bundle(tmp_path: Path, names: tuple[str, ...] = (*NUMPY, PIP, SETUPTOOLS)) -> Path:
    """Wheels whose bytes are their own names, listed as the fetch lists them."""
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name in names:
        (wheels / name).write_text(name, encoding="utf-8")
    (wheels / "SHA256SUMS.txt").write_text(
        "".join(f"{_digest(name)}  ./{name}\n" for name in names), encoding="utf-8"
    )
    (wheels / "constraints.txt").write_text(
        "# pinned\nnumpy==2.4.6\npip==26.2.2\nsetuptools==84.2.1  # backend\n", encoding="utf-8"
    )
    (wheels / "README.md").write_text("the bundle\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy==2.4.6\n", encoding="utf-8")
    # As the shipped file has it: the constraints line is read past, not as a pin.
    (tmp_path / "requirements-dev.txt").write_text(
        "-r requirements.txt\n-c wheels/constraints.txt\n", encoding="utf-8"
    )
    return wheels


def _append(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _run(project_root: Path, tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(project_root / "scripts" / "wheel_requirements.py"),
         str(tmp_path / "wheels"), str(tmp_path / "wheels" / "constraints.txt"),
         str(tmp_path / "requirements-dev.txt"), *extra],
        capture_output=True, text=True, check=False,
    )


def test_every_pin_gets_the_hash_of_every_listed_wheel(project_root: Path, tmp_path: Path) -> None:
    _bundle(tmp_path)
    out = tmp_path / "venv"
    out.mkdir()

    completed = _run(project_root, tmp_path, "--write", str(out))

    assert completed.returncode == 0, completed.stderr
    assert (out / "bundle-requirements.txt").read_text(encoding="utf-8").splitlines() == [
        f"numpy==2.4.6 --hash=sha256:{_digest(NUMPY[0])} --hash=sha256:{_digest(NUMPY[1])}",
        f"setuptools==84.2.1 --hash=sha256:{_digest(SETUPTOOLS)}",
    ]
    assert (out / "bundle-pip.txt").read_text(encoding="utf-8") == (
        f"pip==26.2.2 --hash=sha256:{_digest(PIP)}\n"
    )


def test_a_check_writes_nothing(project_root: Path, tmp_path: Path) -> None:
    _bundle(tmp_path)

    completed = _run(project_root, tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "requirements-dev.txt", "requirements.txt", "wheels",
    ]


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("no manifest", "SHA256SUMS.txt is missing"),
        ("empty manifest", "SHA256SUMS.txt lists no wheels"),
        ("comment", "line 1 is not `<sha256>  ./<name>.whl`"),
        ("CRLF", "line 1 is not `<sha256>  ./<name>.whl` (it ends in a CR"),
        ("listed twice", f"lists {PIP} twice"),
        ("unlisted", "does not list, which pip could install in place of the listed ones: "
                     "numpy-2.4.6-1-cp311"),
        ("unlisted upper case", "does not list, which pip could install in place of the listed "
                                "ones: X.WHL"),
        ("sdist", "does not list, which pip could install in place of the listed ones: "
                  "pyyaml-6.0.3.tar.gz"),
        ("stray file", "does not list, which pip could install in place of the listed ones: "
                       "notes.txt, sub"),
        ("absent", f"lacks wheels SHA256SUMS.txt lists: {SETUPTOOLS}"),
        ("pin without a wheel", "pins wheel==0.48.0 and"),
        ("requirement off the pin", "requirements-dev.txt pins numpy==2.5.0, "),
        ("requirement not pinned", "pins scipy==1.17.1, "),
        ("not an exact pin", "not an exact pin: numpy>=2"),
        ("other constraints", "names constraints other than "),
        ("extra index", "not an exact pin: --extra-index-url https://example.invalid/simple"),
        ("editable", "not an exact pin: -e git+https://example.invalid/x.git"),
        ("constraints name constraints", "a constraints file names another: -c constraints.txt"),
    ],
)
def test_a_bundle_other_than_its_manifest_is_refused(
    project_root: Path, tmp_path: Path, damage: str, message: str
) -> None:
    wheels = _bundle(tmp_path)
    manifest = wheels / "SHA256SUMS.txt"
    text = manifest.read_text(encoding="utf-8")
    if damage == "no manifest":
        manifest.unlink()
    elif damage == "empty manifest":
        manifest.write_text("", encoding="utf-8")
    elif damage == "comment":
        manifest.write_text("# fetched\n" + text, encoding="utf-8")
    elif damage == "CRLF":
        manifest.write_bytes(text.replace("\n", "\r\n").encode())
    elif damage == "listed twice":
        manifest.write_text(text + f"{_digest(PIP)}  ./{PIP}\n", encoding="utf-8")
    elif damage == "unlisted":
        (wheels / NUMPY[0].replace("2.4.6-", "2.4.6-1-")).write_text("x", encoding="utf-8")
    elif damage == "unlisted upper case":
        (wheels / "X.WHL").write_text("x", encoding="utf-8")
    elif damage == "sdist":
        (wheels / "pyyaml-6.0.3.tar.gz").write_bytes(b"an sdist, whose setup.py pip runs")
    elif damage == "stray file":
        (wheels / "notes.txt").write_text("x", encoding="utf-8")
        (wheels / "sub").mkdir()
    elif damage == "absent":
        (wheels / SETUPTOOLS).unlink()
    elif damage == "pin without a wheel":
        _append(wheels / "constraints.txt", "wheel==0.48.0\n")
    elif damage == "requirement off the pin":
        (tmp_path / "requirements.txt").write_text("numpy==2.5.0\n", encoding="utf-8")
    elif damage == "requirement not pinned":
        (tmp_path / "requirements.txt").write_text("numpy==2.4.6\nscipy==1.17.1\n",
                                                   encoding="utf-8")
    elif damage == "other constraints":
        (tmp_path / "other.txt").write_text("numpy==2.4.6\n", encoding="utf-8")
        _append(tmp_path / "requirements-dev.txt", "-c other.txt\n")
    elif damage == "extra index":
        _append(tmp_path / "requirements.txt", "--extra-index-url https://example.invalid/simple\n")
    elif damage == "editable":
        _append(tmp_path / "requirements.txt", "-e git+https://example.invalid/x.git#egg=x\n")
    elif damage == "constraints name constraints":
        _append(wheels / "constraints.txt", "-c constraints.txt\n")
    else:
        (tmp_path / "requirements.txt").write_text("numpy>=2\n", encoding="utf-8")
    out = tmp_path / "venv"
    out.mkdir()

    completed = _run(project_root, tmp_path, "--write", str(out))

    assert completed.returncode == 1
    assert message in completed.stderr, completed.stderr
    assert "docs/RUNBOOK.md step 0" in completed.stderr
    assert list(out.iterdir()) == [], "nothing to install from"


def test_the_shipped_manifest_gives_every_pin_a_hash(project_root: Path, tmp_path: Path) -> None:
    """The committed manifest, constraints and requirements, with stand-ins for the wheels."""
    spec = importlib.util.spec_from_file_location(
        "orbitwin_wheel_requirements", project_root / "scripts" / "wheel_requirements.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shipped = project_root / "vendor" / "wheels"
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    manifest = (shipped / "SHA256SUMS.txt").read_text(encoding="utf-8")
    (wheels / "SHA256SUMS.txt").write_text(manifest, encoding="utf-8")
    for line in manifest.splitlines():
        (wheels / line.split()[-1].lstrip("*").removeprefix("./")).write_bytes(b"")

    install, pip = module.hashed_lines(
        wheels, shipped / "constraints.txt", project_root / "requirements-dev.txt"
    )

    pins = module.read_pins(shipped / "constraints.txt", [])
    assert [line.split(" ")[0] for line in install] == [
        f"{name}=={version}" for name, version in sorted(pins.items()) if name != "pip"
    ]
    assert [line.split(" ")[0] for line in pip] == [f"pip=={pins['pip']}"]
    assert sum(line.count("--hash=sha256:") for line in install + pip) == len(manifest.splitlines())
