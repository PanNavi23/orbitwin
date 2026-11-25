from __future__ import annotations


class OrbitWinError(Exception):
    """Base exception for expected OrbitWin failures."""

    #: What `orbitwin: ERROR [<code>]:` names (docs/ERROR_CODES.md); an instance may set its own.
    code: str = "error"


class ParseError(OrbitWinError):
    """The GMAT script could not be interpreted unambiguously."""

    code = "parse"


class PolicyError(OrbitWinError):
    """A generated script violates immutable-field policy."""

    code = "policy"


class DynamicsError(OrbitWinError):
    """An orbital mechanics calculation failed."""

    code = "dynamics"


class GmatVerdictError(OrbitWinError):
    """GMAT flew the candidate, and its flight refuses it: an answer, not a failed run."""

    code = "gmat-verdict"


class AltitudeFloorError(PolicyError, GmatVerdictError):
    """GMAT flew the pursuer under policy.minimum_flown_altitude_km."""

    code = "gmat-verdict"


class ConstraintVerdictError(PolicyError, GmatVerdictError):
    """The candidate as GMAT flew it breaks a constraint."""

    code = "gmat-verdict"


class LambertError(DynamicsError):
    """No numerically valid Lambert solution could be produced."""


class GmatRunError(OrbitWinError):
    """GmatConsole failed, timed out, or produced stale/missing outputs."""

    code = "gmat-run"


class StopVerdictError(GmatRunError, GmatVerdictError):
    """The submitted script, stopped where GMAT measured the hit, does not end inside."""

    code = "gmat-verdict"


class ConfigurationError(OrbitWinError):
    """Configuration is missing, inconsistent, or dimensionally unsafe."""

    code = "config"


def error_code(exc: BaseException) -> str:
    """The code `orbitwin: ERROR [<code>]:` names for ``exc`` (docs/ERROR_CODES.md)."""
    if isinstance(exc, OrbitWinError):
        return str(getattr(exc, "code", None) or OrbitWinError.code)
    if isinstance(exc, OSError):
        return "io"
    if isinstance(exc, ValueError):
        return "value"
    return "error"
