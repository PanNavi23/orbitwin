#!/usr/bin/env python3
"""Check the offline wheel bundle against its manifest and write pip's hashed requirements.

bootstrap_offline.sh runs this before it touches anything, and again once the
venv exists, to write the files it installs from:

    python -I scripts/wheel_requirements.py WHEELS CONSTRAINTS REQUIREMENTS [--write DIR]

The bundle is refused unless WHEELS/SHA256SUMS.txt is there, every wheel it
lists is in WHEELS, WHEELS holds nothing else but that file, constraints.txt
and README.md, and every pin in
REQUIREMENTS (requirements-dev.txt and what it includes) is the one
CONSTRAINTS holds; their only option besides `-r` is a `-c` naming CONSTRAINTS. With --write, DIR/bundle-requirements.txt pins every
constrained project except pip, each with the sha256 of every listed wheel of
that version, for `pip install --require-hashes`; DIR/bundle-pip.txt does the
same for pip, for an interpreter without ensurepip.

pip installed a wheel the manifest did not list: with no manifest at all, and
with one beside the listed wheels under a build tag, which pip prefers. The
hashes make pip itself refuse anything else, whatever lies in WHEELS.

Standard library only, and no syntax past 3.8: it runs on whatever PYTHON
names, before the bootstrap has checked that interpreter.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

MANIFEST = "SHA256SUMS.txt"
#: The bundle's own files besides the wheels; anything else in it is refused.
EXPECTED = frozenset({MANIFEST, "constraints.txt", "README.md"})
#: As fetch_offline_wheels.sh writes it, or `sha256sum -b`; build_release.py
#: takes the same lines.
_LINE = re.compile(r"([0-9a-f]{64}) [ *](?:\./)?([^/\s*]+\.whl)")


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_manifest(wheels: Path, problems: list[str]) -> dict[str, str]:
    """{wheel file name: sha256} as the manifest lists them."""
    path = wheels / MANIFEST
    if not path.is_file():
        problems.append(f"{path} is missing, so nothing shows these wheels are the ones tested")
        return {}
    listed: dict[str, str] = {}
    lines = path.read_bytes().decode("utf-8", errors="replace").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for number, line in enumerate(lines, start=1):
        match = _LINE.fullmatch(line)
        if match is None:
            # A CRLF file reads as a CR at the end of every name.
            cr = " (it ends in a CR: write the file with LF line endings)" * line.endswith("\r")
            problems.append(
                f"{path} line {number} is not `<sha256>  ./<name>.whl`{cr}: {line!r}"
            )
        elif match.group(2) in listed:
            problems.append(f"{path} lists {match.group(2)} twice")
        else:
            listed[match.group(2)] = match.group(1)
    if not lines:
        problems.append(f"{path} lists no wheels")
    return listed


def read_pins(
    path: Path,
    problems: list[str],
    seen: set[Path] | None = None,
    constraints: Path | None = None,
) -> dict[str, str]:
    """{canonical name: version} for every `name==version`, following `-r` includes.

    A `-c` line is passed over only when it names constraints, which is read on its
    own; every other option is refused.
    """
    seen = set() if seen is None else seen
    if path.resolve() in seen:
        return {}
    seen.add(path.resolve())
    pins: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        problems.append(f"cannot read {path}: {error.strerror}")
        return pins
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            included = path.parent / line.split(None, 1)[1]
            pins.update(read_pins(included, problems, seen, constraints))
            continue
        if line.startswith(("-c ", "--constraint ")):
            named = (path.parent / line.split(None, 1)[1]).resolve()
            if constraints is None:
                problems.append(f"{path}: a constraints file names another: {line}")
            elif named != constraints.resolve():
                problems.append(f"{path}: names constraints other than {constraints}: {line}")
            continue
        name, separator, version = line.partition("==")
        if not separator or not name.strip() or not version.strip() or line.startswith("-"):
            problems.append(f"{path}: not an exact pin: {line}")
            continue
        pins[canonical(name.strip())] = version.strip()
    return pins


def hashed_lines(
    wheels: Path, constraints: Path, requirements: Path
) -> tuple[list[str], list[str]]:
    """The checks, then (pip's hashed lines without pip, pip's own line); exits on a problem."""
    problems: list[str] = []
    listed = read_manifest(wheels, problems)
    present = set(os.listdir(wheels))
    if listed:
        # An sdist too: pip reads every file here, and a dry run without
        # --only-binary ran one's setup.py.
        unlisted = sorted(present - set(listed) - EXPECTED)
        absent = sorted(set(listed) - present)
        if unlisted:
            problems.append(
                f"{wheels} holds files {MANIFEST} does not list, which pip could install in "
                "place of the listed ones: " + ", ".join(unlisted)
            )
        if absent:
            problems.append(f"{wheels} lacks wheels {MANIFEST} lists: " + ", ".join(absent))
    pins = read_pins(constraints, problems)
    for name, version in sorted(read_pins(requirements, problems, constraints=constraints).items()):
        if pins.get(name) != version:
            problems.append(
                f"{requirements} pins {name}=={version}, {constraints} "
                + (f"pins {pins[name]}" if name in pins else "does not pin it")
            )
    digests: dict[tuple[str, str], list[str]] = {}
    for wheel, digest in sorted(listed.items()):
        parts = wheel[: -len(".whl")].split("-")
        if len(parts) in (5, 6):
            digests.setdefault((canonical(parts[0]), parts[1]), []).append(digest)
        else:
            problems.append(f"{wheel} is not a wheel file name")
    install: list[str] = []
    pip: list[str] = []
    for name, version in sorted(pins.items()):
        found = digests.get((name, version))
        if not found:
            # With no usable manifest, that says it already.
            if listed:
                problems.append(
                    f"{constraints} pins {name}=={version} and {wheels} has no such wheel"
                )
            continue
        line = f"{name}=={version} " + " ".join(f"--hash=sha256:{d}" for d in found)
        (pip if name == "pip" else install).append(line)
    if problems:
        print("error: the wheel bundle is not the one its manifest describes:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "Bring vendor/wheels/ again from the release archive, unchanged "
            "(docs/RUNBOOK.md step 0). On the day there is no other remedy.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return install, pip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheels", type=Path)
    parser.add_argument("constraints", type=Path)
    parser.add_argument("requirements", type=Path)
    parser.add_argument("--write", type=Path, metavar="DIR")
    args = parser.parse_args()
    install, pip = hashed_lines(args.wheels, args.constraints, args.requirements)
    if args.write is not None:
        (args.write / "bundle-requirements.txt").write_text(
            "".join(line + "\n" for line in install), encoding="utf-8"
        )
        (args.write / "bundle-pip.txt").write_text(
            "".join(line + "\n" for line in pip), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
