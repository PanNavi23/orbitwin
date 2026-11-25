from __future__ import annotations

import numpy as np

from .errors import DynamicsError
from .models import StateVector


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-14:
        raise DynamicsError(f"Cannot construct {label} axis from a zero vector")
    return vector / norm


def vnb_basis(state: StateVector) -> np.ndarray:
    """Return a 3x3 matrix whose columns are GMAT-style V, N, B axes."""
    v_axis = _unit(state.v, "V")
    n_axis = _unit(np.cross(state.r, state.v), "N")
    b_axis = _unit(np.cross(v_axis, n_axis), "B")
    basis = np.column_stack((v_axis, n_axis, b_axis))
    if np.linalg.det(basis) < 0.999999:
        raise DynamicsError("VNB basis is not right handed")
    return basis


def lvlh_basis(state: StateVector) -> np.ndarray:
    """Return GMAT-style LVLH axes: X radial, Z orbit-normal, Y completes RHS."""
    x_axis = _unit(state.r, "LVLH X")
    z_axis = _unit(np.cross(state.r, state.v), "LVLH Z")
    y_axis = _unit(np.cross(z_axis, x_axis), "LVLH Y")
    basis = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(basis) < 0.999999:
        raise DynamicsError("LVLH basis is not right handed")
    return basis


def _require_earth_origin(coordinate_system: str, origin: str, axes: str) -> None:
    """A Local frame is built about a body, and here that body is Earth.

    vnb_basis and lvlh_basis are built from the state this project propagates,
    which is Earth-centred, so a Local burn declared about anything else is
    written in a basis GMAT will not reconstruct -- the components would be
    right for the wrong frame, which is the worst shape an error can take here.

    Origin was the one field of the (system, origin, axes) triple that was
    parsed, stored, compared in the burn contract and written into synthesized
    resources without ever being read. An unsupported system and unsupported
    axes both failed closed; `Local/Luna/VNB` was accepted and converted as if
    it said Earth.
    """
    if origin.strip().lower() != "earth":
        raise DynamicsError(
            f"Unsupported burn origin: {origin}. A {coordinate_system}/{origin}/{axes} "
            "burn is expressed about that body, and this project propagates about Earth."
        )


def inertial_to_script_components(
    dv_inertial_km_s: np.ndarray,
    state_at_burn: StateVector,
    coordinate_system: str,
    origin: str,
    axes: str,
) -> np.ndarray:
    cs = coordinate_system.strip().lower()
    ax = axes.strip().lower()
    vector = np.asarray(dv_inertial_km_s, dtype=float)
    if cs in {"earthmj2000eq", "mj2000eq"}:
        # A named coordinate system defines the frame outright. GMAT reads
        # Origin and Axes only for Local, so neither says anything here.
        return vector.copy()
    if cs != "local":
        raise DynamicsError(f"Unsupported burn coordinate system: {coordinate_system}")
    _require_earth_origin(coordinate_system, origin, axes)
    if ax == "vnb":
        vnb_components_km_s: np.ndarray = vnb_basis(state_at_burn).T @ vector
        return vnb_components_km_s
    if ax == "lvlh":
        lvlh_components_km_s: np.ndarray = lvlh_basis(state_at_burn).T @ vector
        return lvlh_components_km_s
    raise DynamicsError(f"Unsupported Local burn axes: {axes}")


def script_components_to_inertial(
    components_km_s: np.ndarray,
    state_at_burn: StateVector,
    coordinate_system: str,
    origin: str,
    axes: str,
) -> np.ndarray:
    cs = coordinate_system.strip().lower()
    ax = axes.strip().lower()
    vector = np.asarray(components_km_s, dtype=float)
    if cs in {"earthmj2000eq", "mj2000eq"}:
        return vector.copy()
    if cs != "local":
        raise DynamicsError(f"Unsupported burn coordinate system: {coordinate_system}")
    _require_earth_origin(coordinate_system, origin, axes)
    if ax == "vnb":
        vnb_inertial_km_s: np.ndarray = vnb_basis(state_at_burn) @ vector
        return vnb_inertial_km_s
    if ax == "lvlh":
        lvlh_inertial_km_s: np.ndarray = lvlh_basis(state_at_burn) @ vector
        return lvlh_inertial_km_s
    raise DynamicsError(f"Unsupported Local burn axes: {axes}")
