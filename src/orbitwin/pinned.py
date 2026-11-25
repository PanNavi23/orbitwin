"""Refuse numpy, SciPy or PyYAML other than the versions requirements.txt pins.

docs/RUNBOOK.md step 0 activates the offline bundle's environment, and every
later step finds python and orbitwin through it. In a terminal where that was
not done, a machine with SciPy installed ran the steps on whatever versions it
had, with no error. This makes that an error, in orbitwin and in each script
a competition day runs.
"""

from __future__ import annotations

import os
import sys
from importlib import metadata
from pathlib import Path

from .errors import OrbitWinError
from .layout import CHECKOUT

#: What the numerics run on. The test and lint tools are not checked.
PINNED = ("numpy", "scipy", "PyYAML")
#: Set to 1 to run anyway, on purpose.
OVERRIDE = "ORBITWIN_ALLOW_UNPINNED"
#: requirements.txt in the checkout this package was loaded from; an installed package has none.
REQUIREMENTS = (CHECKOUT / "requirements.txt" if CHECKOUT is not None
                else Path(__file__).resolve().parent / "no-requirements.txt")


class UnpinnedError(OrbitWinError):
    """numpy, SciPy or PyYAML is not the version requirements.txt pins."""

    code = "environment"


def _pins(requirements: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        name, separator, version = line.partition("==")
        if separator:
            pins[name.strip().lower()] = version.strip()
    return pins


def unpinned(requirements: Path = REQUIREMENTS) -> list[str]:
    """Each pinned package whose installed version is not its pin, described."""
    pins = _pins(requirements)
    wrong: list[str] = []
    for name in PINNED:
        wanted = pins.get(name.lower())
        if wanted is None:
            continue
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "missing"
        if installed != wanted:
            wrong.append(f"{name} {installed} (pinned {wanted})")
    return wrong


def assert_pinned_environment(requirements: Path = REQUIREMENTS) -> None:
    """Raise unless the pinned packages are the installed ones, or the override is set.

    Outside a checkout (no requirements.txt next to src/) there is nothing to
    compare against, and nothing is checked.
    """
    if os.environ.get(OVERRIDE) == "1" or not requirements.is_file():
        return
    wrong = unpinned(requirements)
    if wrong:
        raise UnpinnedError(
            f"{sys.executable} has {', '.join(wrong)}, not the validated set in "
            f"{requirements.name}. On the competition machine, activate the bundle's "
            "environment (docs/RUNBOOK.md step 0: . .venv/bin/activate). Elsewhere, install "
            "the pins into this one (uv pip install -r requirements-dev.txt), or set "
            f"{OVERRIDE}=1 to run on these versions anyway."
        )


def refuse_unpinned() -> None:
    """For a script's main(), before it reads anything: exit 2 off the pins."""
    try:
        assert_pinned_environment()
    except OrbitWinError as error:
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(2) from None
