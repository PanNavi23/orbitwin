"""The duel side: what the duel rules allow that the ranking rules would not, and back."""

from __future__ import annotations

import copy
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from orbitwin.duel.params import DuelParamsError, from_dict
from orbitwin.duel.referee import Band, Reading, judge, simulate_round
from orbitwin.duel.state import CraftState, MatchState
from orbitwin.duel.units import SubmissionError, collect_round, executed_km_s, project_m_s
from orbitwin.rules import DUEL

MU = 398600.4415
SOURCE = Path(__file__).resolve().parents[2] / "src"

ANNOUNCED: dict[str, Any] = {
    "dv_lim_m_s": 250.0,
    "r_cap_km": 5.0,
    "t_round_s": 600.0,
    "a_lo_km": 6800.0,
    "a_hi_km": 7000.0,
    "pursuer_r0_km": [6878.0, 0.0, 0.0],
    "pursuer_v0_km_s": [0.0, 7.6, 0.0],
    "pursuer_epoch": "e",
    "evader_r0_km": [6878.0, 4.0, 0.0],
    "evader_v0_km_s": [0.0, 7.6, 0.0],
    "evader_epoch": "e",
}


def _params(**announced: Any) -> Any:
    raw = {"announced": {**copy.deepcopy(ANNOUNCED), **announced}}
    return from_dict(raw)


def _reading() -> Reading:
    return Reading(order="priority", capture_dwell_s=0.0, pursuer_crossing_dwell_s=0.0,
                   evader_crossing_dwell_s=0.0, pursuer_surface="equatorial",
                   evader_surface="equatorial", capture_passages="favour")


def test_300_m_s_with_a_250_cap_is_scaled_to_250() -> None:
    executed = project_m_s(np.array([300.0, 0.0, 0.0]), 250.0)
    assert executed.tolist() == [250.0, 0.0, 0.0]


def test_six_rounds_at_a_cap_summing_past_3_km_s_all_run_in_full() -> None:
    cap = 600.0
    total = 0.0
    for _ in range(DUEL.rounds):
        total += float(np.linalg.norm(executed_km_s(np.array([600.0, 0.0, 0.0]), cap)))
    assert total == pytest.approx(3.6)
    assert DUEL.total_dv_limit_km_s is None


def test_a_second_impulse_in_a_round_is_refused() -> None:
    with pytest.raises(SubmissionError):
        collect_round(["追捕方, 1, 1.00, 0.00, 0.00", "追捕方, 1, 2.00, 0.00, 0.00"], 1)


def test_reaching_the_safe_a_early_does_not_win() -> None:
    r = np.array([6878.0, 0.0, 0.0])
    v = np.array([0.0, math.sqrt(MU * (2.0 / 6878.0 - 1.0 / 6900.0)), 0.0])
    far = CraftState.of(np.array([0.0, 0.0, 7500.0]), np.array([7.29, 0.0, 0.0]))
    for round_number in (1, 3, 5):
        state = MatchState(round_number, "e", far, CraftState.of(r, v))
        facts = simulate_round(state, np.zeros(3), np.zeros(3), _params(), view="referee")
        assert judge(facts, _reading(), Band(6800.0, 7000.0)).winner is None


def test_a_4_km_closest_approach_depends_on_the_announced_radius() -> None:
    # Two craft on one circle, 4 km apart along it: the gap stays 4 km.
    radius = 6878.0
    speed = math.sqrt(MU / radius)
    angle = 2.0 * math.asin(2.0 / radius)
    lead = np.array([math.cos(angle), math.sin(angle), 0.0])
    state = MatchState(
        1, "e",
        CraftState.of(np.array([radius, 0.0, 0.0]), np.array([0.0, speed, 0.0])),
        CraftState.of(radius * lead, speed * np.array([-lead[1], lead[0], 0.0])),
    )
    for r_cap, expected in ((5.0, "pursuer"), (3.0, None)):
        facts = simulate_round(state, np.zeros(3), np.zeros(3), _params(r_cap_km=r_cap),
                               view="referee")
        assert facts.capture.minimum_distance_km == pytest.approx(4.0, abs=0.01)
        assert judge(facts, _reading(), Band(6800.0, 7000.0)).winner == expected


def test_the_preliminary_horizon_is_not_a_duel_parameter() -> None:
    with pytest.raises(DuelParamsError, match="not a duel parameter"):
        _params(horizon_s=4 * 5700.0)


def test_importing_the_duel_loads_no_yaml_or_network_client() -> None:
    # Not socket: on 3.11 scipy loads it through numpy.testing and email.utils.
    # Whether anything opens one is the next test's question.
    probe = (
        "import sys, orbitwin.duel, orbitwin.duel.referee, orbitwin.duel.units,"
        " orbitwin.duel.state;"
        "banned = {'yaml', 'ssl', 'http.client', 'urllib.request'};"
        "print(sorted(banned & set(sys.modules)));"
        "print(sorted(m for m in sys.modules if m.startswith('orbitwin.')"
        " and not m.startswith('orbitwin.duel')))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            env={"PYTHONPATH": str(SOURCE)}, check=True)
    banned, loaded = result.stdout.strip().splitlines()
    assert banned == "[]"
    # Core only: never config, a ranking module or the CLI. rulepacks is the rules' data.
    allowed = {"orbitwin.conic", "orbitwin.errors", "orbitwin.models", "orbitwin.orbit",
               "orbitwin.rules", "orbitwin.rulepacks"}
    assert set(eval(loaded)) <= allowed, loaded


def test_a_round_opens_no_socket() -> None:
    """The match laptop is offline (PDF §6); a socket opened here would fail there."""
    probe = (
        "import sys\n"
        "seen = []\n"
        "sys.addaudithook(lambda event, args: seen.append(event)"
        " if event.startswith('socket.') else None)\n"
        "import numpy as np\n"
        "from orbitwin.duel.referee import simulate_round\n"
        "from orbitwin.duel.state import CraftState, MatchState\n"
        "from orbitwin.duel.params import from_dict\n"
        f"params = from_dict({ {'announced': ANNOUNCED}!r})\n"
        "r = 7000.0; v = (398600.4415 / r) ** 0.5\n"
        "state = MatchState(1, 'e', CraftState.of(np.array([r, 0.0, 0.0]),"
        " np.array([0.0, v, 0.0])), CraftState.of(np.array([0.0, r, 0.0]),"
        " np.array([-v, 0.0, 0.0])))\n"
        "simulate_round(state, np.zeros(3), np.zeros(3), params, view='referee')\n"
        "print(sorted(set(seen)))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            env={"PYTHONPATH": str(SOURCE)}, check=True)
    assert result.stdout.strip() == "[]", result.stdout
