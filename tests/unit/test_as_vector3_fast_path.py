"""as_vector3's ndarray path gives what its generic path gives, float for float."""

from __future__ import annotations

import math

import numpy as np
import pytest

from orbitwin.models import as_vector3


def _exact(values) -> list[tuple[type, str]]:
    return [(type(item), float(item).hex()) for item in values]


def _generic(values) -> list[tuple[type, str]]:
    """A list never takes the fast path."""
    return _exact(as_vector3(list(values)))


@pytest.mark.parametrize("values", [
    [1.0, -2.5, 3e300], [0.0, -0.0, 5e-324], [math.inf, -math.inf, math.nan],
    [6378.137, 7.0e3, -1.0e-15], [2.0**53 + 1, 1 / 3, -2 / 7],
])
def test_a_float64_array_gives_the_generic_floats(values: list[float]) -> None:
    array = np.array(values, dtype=np.float64)
    assert _exact(as_vector3(array)) == _generic(array)
    assert all(kind is float for kind, _ in _generic(array))


def test_random_rows_match_bit_for_bit() -> None:
    rng = np.random.default_rng(7)
    rows = rng.standard_normal((2000, 3)) * 10.0 ** rng.uniform(-6.0, 6.0, (2000, 1))
    for row in rows:
        assert _exact(as_vector3(row)) == _generic(row)


@pytest.mark.parametrize("values", [
    np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0], dtype=np.float32),
    np.array([0.1, 0.2, 0.3], dtype=np.longdouble), np.array([True, False, True]),
    np.arange(6.0)[::2], np.ma.masked_array([1.5, 2.5, 3.5]),
    # tolist() gives None for the masked element; float() gives nan.
    pytest.param(np.ma.masked_array([1.5, 2.5, 3.5], mask=[False, True, False]),
                 marks=pytest.mark.filterwarnings("ignore:Warning. converting a masked element")),
    [1, 2, 3], (1.5, 2.5, 3.5),
], ids=["int64", "float32", "longdouble", "bool", "strided", "subclass", "masked", "list",
        "tuple"])
def test_other_inputs_give_the_generic_floats(values) -> None:
    assert _exact(as_vector3(values)) == _generic(values)
    assert all(kind is float for kind, _ in _exact(as_vector3(values)))


@pytest.mark.parametrize("values", [
    np.zeros(2), np.zeros(4), np.zeros((1, 3)), np.zeros((3, 1)), np.zeros((3, 3)),
    np.zeros((3, 1), dtype=np.float32), np.ma.masked_array(np.zeros((3, 1))), np.zeros(()),
    [1.0, 2.0],
], ids=["2", "4", "1x3", "3x1", "3x3", "3x1-float32", "3x1-masked", "0-d", "list-2"])
def test_a_wrong_shape_is_refused(values) -> None:
    """(3, 1) passed on NumPy 2.3, whose float() takes a 1-element row, and failed on 2.4."""
    with pytest.raises(ValueError, match="expected 3 components"):
        as_vector3(values)


def _object_rows() -> np.ndarray:
    rows = np.empty(3, dtype=object)
    for k in range(3):
        rows[k] = np.array([float(k)])
    return rows


@pytest.mark.parametrize("values", [
    lambda: list(np.zeros((3, 1))), lambda: iter(np.zeros((3, 1))), _object_rows,
    lambda: [1.0, 2.0, np.zeros(1)], lambda: (np.zeros(3), 1.0, 2.0),
], ids=["list-of-rows", "iter-of-rows", "object-array", "one-row", "first-of-three"])
def test_an_array_component_is_refused(values) -> None:
    """NumPy 2.3's float() takes a 1-element row and 2.4's refuses it."""
    with pytest.raises(TypeError, match="expected 3 scalar components"):
        as_vector3(values())


def test_a_0_d_array_component_is_a_scalar() -> None:
    assert as_vector3([np.array(1.5), 2.5, np.float64(3.5)]) == (1.5, 2.5, 3.5)
