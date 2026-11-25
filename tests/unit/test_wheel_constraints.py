"""vendor/wheels/constraints.txt and the bundle's manifest describe one set.

Without it the unpinned dependencies moved on every refetch (six between
2026-09-05 and 2026-11-02), and CI resolved its own newest set besides.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FETCH = Path("scripts") / "fetch_offline_wheels.sh"


def _name(raw: str) -> str:
    return re.sub(r"[-_.]+", "-", raw).lower()


def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name, separator, version = line.partition("==")
        assert separator, f"{path.name}: not an exact pin: {line}"
        assert _name(name) not in pins, f"{path.name}: {name} twice"
        pins[_name(name)] = version.strip()
    return pins


def _wheels(project_root: Path) -> list[str]:
    manifest = project_root / "vendor" / "wheels" / "SHA256SUMS.txt"
    if not manifest.is_file():
        # A release archive built before the bundle's manifest was shipped.
        pytest.skip("vendor/wheels/SHA256SUMS.txt is not in this tree")
    return [
        line.split()[-1].lstrip("*").removeprefix("./")
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_the_bundle_holds_exactly_the_constrained_versions(project_root: Path) -> None:
    pins = _pins(project_root / "vendor" / "wheels" / "constraints.txt")
    bundle = {(_name(wheel.split("-")[0]), wheel.split("-")[1]) for wheel in _wheels(project_root)}

    assert {name for name, _ in bundle} == set(pins), "a package in one and not the other"
    assert {(name, version) for name, version in bundle if pins[name] != version} == set()


@pytest.mark.parametrize("document", ["README.md", "README.zh-TW.md", "vendor/wheels/README.md"])
def test_the_wheel_count_quoted_is_the_manifests(project_root: Path, document: str) -> None:
    """The README kept 38 after the refetch of 2026-11-02 made the manifest 39."""
    text = (project_root / document).read_text(encoding="utf-8")
    quoted = {int(count) for count in re.findall(r"\b(\d+) (?:個 )?wheels?\b", text)}

    assert quoted == {len(_wheels(project_root))}


def test_every_compiled_package_has_a_wheel_for_every_python(project_root: Path) -> None:
    fetch = (project_root / FETCH).read_text(encoding="utf-8")
    found = re.search(r'VERSIONS="\$\{WHEEL_PYTHON_VERSIONS:-([0-9. ]+)\}"', fetch)
    assert found
    tags = {"cp" + version.replace(".", "") for version in found.group(1).split()}
    by_package: dict[str, set[str]] = {}
    for wheel in _wheels(project_root):
        parts = wheel[: -len(".whl")].split("-")
        by_package.setdefault(_name(parts[0]), set()).add(parts[-3])
    for package, pythons in by_package.items():
        if pythons <= {"py3", "py2.py3"} or pythons == {"cp39"}:  # pure, or one abi3 wheel
            continue
        assert tags <= pythons, f"{package} has no wheel for {sorted(tags - pythons)}"


def test_the_direct_pins_are_the_constrained_ones(project_root: Path) -> None:
    pins = _pins(project_root / "vendor" / "wheels" / "constraints.txt")
    for source in ("requirements.txt", "requirements-dev.txt"):
        for name, version in _pins(project_root / source).items():
            assert pins.get(name) == version, f"{source}: {name}=={version}"


def test_requirements_dev_names_the_constraints(project_root: Path) -> None:
    """So `-r requirements-dev.txt` alone, in any recipe, installs the bundle's set."""
    lines = (project_root / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    assert "-c vendor/wheels/constraints.txt" in lines


def test_bootstrap_and_ci_read_the_constraints(project_root: Path) -> None:
    bootstrap = (project_root / "scripts" / "bootstrap_offline.sh").read_text(encoding="utf-8")
    ci = (project_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    fetch = (project_root / FETCH).read_text(encoding="utf-8")
    for script in (bootstrap, fetch):
        assert "\nCONSTRAINTS=vendor/wheels/constraints.txt\n" in script
        assert "WHEEL_CONSTRAINTS" not in script
    # The dry run; the installs take hashed pins made from them.
    assert bootstrap.count('--constraint "$CONSTRAINTS"') == 1
    assert bootstrap.count('scripts/wheel_requirements.py "$WHEELS" "$CONSTRAINTS"') == 2
    assert bootstrap.count('"${hashed[@]}"') == 2
    assert "--require-hashes)" in bootstrap
    # The dry run too: without it pip ran an sdist's setup.py to read its metadata.
    dry_run = bootstrap[bootstrap.index("install --dry-run"):bootstrap.index("2>&1)\"; then")]
    assert "--only-binary :all:" in dry_run
    assert "--requirement requirements-dev.txt" in dry_run
    assert '--constraint "$CONSTRAINTS"' in dry_run
    # Every install of requirements-dev.txt, the continuation lines joined:
    # the test legs' and the checks job's.
    commands = ci.replace("\\\n", " ").splitlines()
    installs = [line for line in commands if re.search(r"(-r|--requirement) requirements-dev", line)]
    assert len(installs) == 2
    assert all('"${constraints[@]}"' in line for line in installs)
    assert ci.count("constraints=(-c vendor/wheels/constraints.txt)") == 2


STUB = r'''
"""Stands in for `python -m pip download`: writes a wheel per call, logs argv."""
import os, sys
from pathlib import Path
args = sys.argv[1:]
log = Path(os.environ["STUB_LOG"])
calls = log.read_text().count("\n") if log.exists() else 0
with log.open("a") as handle:
    handle.write(repr(args) + "\n")
if str(calls + 1) == os.environ.get("STUB_FAIL_ON"):
    sys.exit(1)
dest = Path(args[args.index("--dest") + 1])
version = args[args.index("--python-version") + 1] if "--python-version" in args else "py3"
tag = "cp" + version.replace(".", "") if version != "py3" else "py3"
(dest / f"demo-1.0-{{tag}}-none-any.whl").write_bytes(tag.encode())
'''


def _fetch(project_root: Path, tmp_path: Path, fail_on: int | None = None):
    if shutil.which("sha256sum") is None or shutil.which("mktemp") is None:
        pytest.skip("needs sha256sum and mktemp")
    root = tmp_path / "tree"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(project_root / FETCH, root / FETCH)
    (root / "requirements-dev.txt").write_text(
        "-c vendor/wheels/constraints.txt\ndemo==1.0\n", encoding="utf-8")
    # A shell wrapper, not a #! line naming the interpreter: a #! path cannot
    # hold a space, and the unpacked release may well sit under one.
    (tmp_path / "stub.py").write_text(STUB.format(), encoding="utf-8")
    stub = tmp_path / "python"
    stub.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path / "stub.py"}" "$@"\n', encoding="utf-8"
    )
    stub.chmod(0o755)
    (tmp_path / "tmp").mkdir(exist_ok=True)
    environment = {
        "PATH": os.environ["PATH"], "PYTHON": str(stub), "TMPDIR": str(tmp_path / "tmp"),
        "WHEELS_DIR": "vendor/wheels",
        "WHEEL_PYTHON_VERSIONS": "3.11 3.14", "STUB_LOG": str(tmp_path / "calls.log"),
    }
    if fail_on is not None:
        environment["STUB_FAIL_ON"] = str(fail_on)
    return root, subprocess.run(
        ["bash", str(root / FETCH)], cwd=root, env=environment, capture_output=True, text=True,
        check=False,
    )


def _old_bundle(root: Path) -> Path:
    wheels = root / "vendor" / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "demo-0.9-cp311-none-any.whl").write_bytes(b"old")
    (wheels / "SHA256SUMS.txt").write_text("old manifest\n", encoding="utf-8")
    return wheels


def test_a_refetch_replaces_the_old_wheels_and_constrains_every_call(
    project_root: Path, tmp_path: Path
) -> None:
    """Fetched beside the old ones, a superseded wheel stayed and the newest was installed."""
    _old_bundle(tmp_path / "tree")

    root, completed = _fetch(project_root, tmp_path)

    assert completed.returncode == 0, completed.stderr
    wheels = root / "vendor" / "wheels"
    assert sorted(path.name for path in wheels.glob("*.whl")) == [
        "demo-1.0-cp311-none-any.whl", "demo-1.0-cp314-none-any.whl", "demo-1.0-py3-none-any.whl",
    ]
    manifest = (wheels / "SHA256SUMS.txt").read_text(encoding="utf-8")
    assert "0.9" not in manifest and manifest.count(".whl") == 3
    calls = (tmp_path / "calls.log").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    assert all("'--constraint', 'vendor/wheels/constraints.txt'" in call for call in calls)
    assert not list((root / "vendor").glob(".wheels-staging.*")), "staging removed"


def test_a_fetch_that_fails_leaves_the_old_bundle(project_root: Path, tmp_path: Path) -> None:
    _old_bundle(tmp_path / "tree")

    root, completed = _fetch(project_root, tmp_path, fail_on=2)

    assert completed.returncode != 0
    wheels = root / "vendor" / "wheels"
    assert [path.name for path in wheels.glob("*.whl")] == ["demo-0.9-cp311-none-any.whl"]
    assert (wheels / "SHA256SUMS.txt").read_text(encoding="utf-8") == "old manifest\n"
    assert not list((root / "vendor").glob(".wheels-staging.*")), "staging removed"
    assert not list((tmp_path / "tmp").iterdir())
