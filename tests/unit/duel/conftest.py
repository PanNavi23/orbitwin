from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

MU = 398600.4415

Orbit = Callable[[float, float, float, float], tuple[np.ndarray, np.ndarray]]


def circular_state(
    radius_km: float, inclination_deg: float, raan_deg: float, u_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """A circular orbit's state at argument of latitude u."""
    i, node, u = (math.radians(x) for x in (inclination_deg, raan_deg, u_deg))
    speed = math.sqrt(MU / radius_km)
    r = np.array([math.cos(u), math.sin(u), 0.0]) * radius_km
    v = np.array([-math.sin(u), math.cos(u), 0.0]) * speed
    rx = np.array([[1, 0, 0], [0, math.cos(i), -math.sin(i)], [0, math.sin(i), math.cos(i)]])
    rz = np.array([[math.cos(node), -math.sin(node), 0], [math.sin(node), math.cos(node), 0],
                   [0, 0, 1]])
    rotation = rz @ rx
    return rotation @ r, rotation @ v


@pytest.fixture
def circular() -> Orbit:
    return circular_state
