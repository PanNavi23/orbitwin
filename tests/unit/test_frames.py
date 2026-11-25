from __future__ import annotations

import numpy as np

from orbitwin.frames import (
    inertial_to_script_components,
    lvlh_basis,
    script_components_to_inertial,
    vnb_basis,
)
from orbitwin.models import StateVector


def test_vnb_basis_orthonormal_and_right_handed() -> None:
    state = StateVector((7000, 100, 300), (-0.2, 7.4, 1.2))
    basis = vnb_basis(state)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    assert np.linalg.det(basis) > 0.999999999


def test_lvlh_basis_orthonormal_and_right_handed() -> None:
    state = StateVector((7000, 100, 300), (-0.2, 7.4, 1.2))
    basis = lvlh_basis(state)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    assert np.linalg.det(basis) > 0.999999999


def test_vnb_conversion_roundtrip() -> None:
    state = StateVector((7000, 100, 300), (-0.2, 7.4, 1.2))
    vector = np.array([0.2, -0.3, 0.4])
    components = inertial_to_script_components(vector, state, "Local", "Earth", "VNB")
    restored = script_components_to_inertial(components, state, "Local", "Earth", "VNB")
    assert np.allclose(restored, vector, atol=1e-14)


def test_inertial_frame_is_identity() -> None:
    state = StateVector((7000, 0, 0), (0, 7.5, 0))
    vector = np.array([0.1, 0.2, 0.3])
    assert np.array_equal(
        inertial_to_script_components(vector, state, "EarthMJ2000Eq", "Earth", "VNB"), vector
    )


def test_a_local_burn_about_another_body_is_refused() -> None:
    """Origin was the one frame field nothing read.

    An ImpulsiveBurn's (CoordinateSystem, Origin, Axes) was parsed, stored,
    compared in the burn contract and written into synthesized resources.
    Unsupported systems and unsupported axes both failed closed; Origin was
    carried the whole way and never looked at, so a `Local/Luna/VNB` resource
    was converted with an Earth-centred VNB basis and written out as if it were
    the Moon's. The components would have been right for the wrong frame, which
    is worse than being refused.

    Only Local is affected: GMAT reads Origin and Axes for Local frames, and a
    named coordinate system defines the frame outright.
    """
    import pytest

    from orbitwin.errors import DynamicsError

    state = StateVector((7000, 100, 300), (-0.2, 7.4, 1.2))
    vector = np.array([0.2, -0.3, 0.4])
    for axes in ("VNB", "LVLH"):
        with pytest.raises(DynamicsError, match="Luna"):
            inertial_to_script_components(vector, state, "Local", "Luna", axes)
        with pytest.raises(DynamicsError, match="Luna"):
            script_components_to_inertial(vector, state, "Local", "Luna", axes)
    # Earth in any spelling is the one this project propagates about.
    for origin in ("Earth", "earth", " Earth "):
        inertial_to_script_components(vector, state, "Local", origin, "VNB")
    # A named coordinate system is the frame, so Origin says nothing there.
    assert np.array_equal(
        inertial_to_script_components(vector, state, "EarthMJ2000Eq", "Luna", "VNB"), vector
    )
