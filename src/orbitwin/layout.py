"""Where this package runs from: a checkout of the repository, or an installed package."""

from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def _checkout() -> Path | None:
    """The repository root when the package is its src/orbitwin, else None (installed)."""
    root = PACKAGE.parents[1]
    if PACKAGE.parent.name == "src" and (root / "pyproject.toml").is_file():
        return root
    return None


#: The checkout's root, or None for an installed package.
CHECKOUT = _checkout()
