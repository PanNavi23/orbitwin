"""A4's B17 on real GMAT R2026a: `ImpulsiveBurn1.V = 0;` after the Element lines.

The parser refuses it. With that guard switched off, verification flies it and
must refuse on its own: measured 2026-11-04, GMAT flew (0, -0.0427, 0.0563)
km/s, |dv| 70.69 m/s, for the 91.15 m/s the script's Element lines claim.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from orbitwin import gmat_parser
from orbitwin.config import OrbitWinConfig
from orbitwin.errors import GmatRunError, ParseError
from orbitwin.frames import script_components_to_inertial
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.multi_burn import as_candidate
from orbitwin.orbit import propagate_universal
from orbitwin.verify import verify_candidate_with_gmat

pytestmark = pytest.mark.gmat

B17 = Path(__file__).resolve().parents[1] / "fixtures" / "burn_alias" / "B17_burn_alias_after.script"
BURN_S = 60.883029141
#: A4's rendered ImpulsiveBurn1, Local/Earth/VNB.
COMPONENTS = (0.05754250780132859, -0.04269037212546463, 0.0563481184225411)


def _config() -> OrbitWinConfig:
    # 600 s keeps the run short. It is before SC_Alien's perigee at 4500 s, which §3
    # rules out for an announced T_max, so the test names it as the override would.
    base = OrbitWinConfig()
    return dataclasses.replace(
        base,
        policy=dataclasses.replace(
            base.policy, rules="ranking", team_number=54, horizon_s=600.0,
            accept_announced_horizon_before_perigee_s=600.0,
            dv_limit_km_s=0.25, dv_safe_limit_km_s=0.249,
            total_dv_limit_km_s=3.0, total_dv_safe_limit_km_s=2.99,
        ),
        pursuer_name="SC_Human", target_name="SC_Alien",
    )


def test_b17_is_refused_by_the_parser_and_by_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gmat: tuple[str, str]
) -> None:
    console, startup = gmat
    config = _config()
    with pytest.raises(ParseError, match=r"Line 126 sets ImpulsiveBurn1\.V\."):
        parse_gmat_script(B17, config)

    monkeypatch.setattr(gmat_parser, "_refuse_unmodelled_burn_fields", lambda *args, **kwargs: None)
    scenario, document = parse_gmat_script(B17, config)
    state = propagate_universal(scenario.pursuer.initial_state, BURN_S, config.policy.mu_km3_s2)
    impulse = script_components_to_inertial(np.asarray(COMPONENTS), state, "Local", "Earth", "VNB")
    candidate = as_candidate([(BURN_S, impulse)], 300.0, scenario, config)
    with pytest.raises(GmatRunError) as caught:
        verify_candidate_with_gmat(
            B17, scenario, document, candidate, config, console, startup, tmp_path / "v",
            timeout_s=600.0,
        )
    message = str(caught.value)
    assert "GMAT flew Maneuver 1 (ImpulsiveBurn1, t = 60.883029141 s)" in message, message
    assert "|dv| 0.0706935" in message and "|dv| 0.0911521" in message, message
    assert "Something in the script changes the impulse GMAT flies" in message
