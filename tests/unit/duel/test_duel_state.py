"""The round-1 state: epochs compared, unequal ones stopped unless aligned (Q10)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from orbitwin.conic import coast
from orbitwin.duel.params import DuelParams, from_dict
from orbitwin.duel.state import StateError, epoch_seconds, match_start

from .conftest import MU


def _params(pursuer_epoch: str, evader_epoch: str, **readings: Any) -> DuelParams:
    return from_dict({
        "announced": {
            "dv_lim_m_s": 250.0, "r_cap_km": 5.0, "t_round_s": 1800.0,
            "a_lo_km": 7000.0, "a_hi_km": 7100.0,
            "pursuer_r0_km": [6878.0, 0.0, 0.0], "pursuer_v0_km_s": [0.0, 7.6, 0.0],
            "pursuer_epoch": pursuer_epoch,
            "evader_r0_km": [0.0, 6900.0, 0.0], "evader_v0_km_s": [-7.59, 0.0, 0.0],
            "evader_epoch": evader_epoch,
        },
        "readings": readings,
    })


def test_epochs_read_in_both_gmat_forms() -> None:
    assert epoch_seconds("05 Jan 1941 12:00:00.000") == ("gregorian", 0.0)
    assert epoch_seconds("06 Jan 1941 12:00:00.000") == ("gregorian", 86400.0)
    gregorian = epoch_seconds("'09 Nov 2026 00:00:00.000'")
    modjulian = epoch_seconds("31353.5")
    assert gregorian is not None and modjulian is not None
    assert gregorian[1] == modjulian[1] == pytest.approx(31353.5 * 86400.0)
    assert epoch_seconds("next Tuesday") is None
    assert epoch_seconds("".join(chr(0xFF10 + int(c)) if c.isdigit() else c
                                 for c in "31353.5")) is None


def test_equal_epochs_start_the_match_as_announced() -> None:
    start = match_start(_params("09 Nov 2026 00:00:00.000", "09 Nov 2026 00:00:00.000"))
    assert start.round_number == 1 and start.epoch == "09 Nov 2026 00:00:00.000"
    assert start.pursuer.r_km == (6878.0, 0.0, 0.0)
    # The same instant written with more digits is one epoch.
    same = match_start(_params("31353.5", "31353.50000"))
    assert same.evader.r_km == (0.0, 6900.0, 0.0)


def test_unequal_epochs_stop_the_reader() -> None:
    with pytest.raises(StateError, match="Q10"):
        match_start(_params("09 Nov 2026 00:00:00.000", "09 Nov 2026 00:01:00.000"))
    with pytest.raises(StateError, match="Q10"):
        match_start(_params("09 Nov 2026 00:00:00.000", "09 Nov 2026 00:00:02.000"))
    # One millisecond is 15 m of separation at 15 km/s: not the same epoch.
    with pytest.raises(StateError, match="Q10"):
        match_start(_params("09 Nov 2026 00:00:00.000", "09 Nov 2026 00:00:00.001"))
    with pytest.raises(StateError, match="Q10"):
        match_start(_params("31353.5", "31353.500000001"))
    with pytest.raises(StateError, match="cannot be compared"):
        match_start(_params("09 Nov 2026 00:00:00.000", "the evader's epoch"))
    # The same instant in two forms: their time scales may differ, so no.
    with pytest.raises(StateError, match="cannot be compared"):
        match_start(_params("09 Nov 2026 00:00:00.000", "31353.5"))


def test_aligned_epochs_coast_the_earlier_craft_to_the_later() -> None:
    later_evader = match_start(_params("09 Nov 2026 00:00:00.000", "09 Nov 2026 00:01:00.000",
                                       align_epochs=True))
    r, v = coast(np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.6, 0.0]), 60.0, MU)
    assert np.allclose(later_evader.pursuer.r, r, atol=1e-9)
    assert np.allclose(later_evader.pursuer.v, v, atol=1e-12)
    assert later_evader.evader.r_km == (0.0, 6900.0, 0.0)
    assert later_evader.epoch == "09 Nov 2026 00:01:00.000"
    later_pursuer = match_start(_params("09 Nov 2026 00:01:00.000", "09 Nov 2026 00:00:00.000",
                                        align_epochs=True))
    r, _ = coast(np.array([0.0, 6900.0, 0.0]), np.array([-7.59, 0.0, 0.0]), 60.0, MU)
    assert np.allclose(later_pursuer.evader.r, r, atol=1e-9)
    assert later_pursuer.pursuer.r_km == (6878.0, 0.0, 0.0)
    assert later_pursuer.epoch == "09 Nov 2026 00:01:00.000"
