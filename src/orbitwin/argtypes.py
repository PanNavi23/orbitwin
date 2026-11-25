"""argparse types the command-line scripts share."""

from __future__ import annotations

import argparse
import math


def positive_finite(text: str) -> float:
    """A number above 0 and below infinity; anything else is a usage error."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not (math.isfinite(value) and value > 0.0):
        raise argparse.ArgumentTypeError(f"not a positive finite number: {text!r}")
    return value


def positive_int(text: str) -> int:
    """A whole number of 1 or more; anything else is a usage error."""
    if not (text.isascii() and text.isdigit()):
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}")
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"not 1 or more: {text!r}")
    return value
