from __future__ import annotations

from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig, SearchConfig
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.search import search_one_burn
from orbitwin.two_burn import search_two_burn_global


@pytest.mark.slow
def test_different_plane_search_has_nonzero_out_of_plane_component(
    different_plane_script: Path,
) -> None:
    config = OrbitWinConfig(
        search=SearchConfig(
            one_burn_samples=64,
            one_burn_grid_burns=6,
            one_burn_grid_arrivals=12,
            max_revolutions=1,
            keep_candidates=50,
            event_step_s=5.0,
            two_burn_enabled=False,
        )
    )
    scenario, _ = parse_gmat_script(different_plane_script, config)
    candidates = search_one_burn(scenario, config)
    assert candidates
    assert any(abs(candidate.burns[0].dv_script_km_s[1]) > 1e-3 for candidate in candidates)


@pytest.mark.slow
def test_high_plane_change_two_burn_fallback(project_root: Path) -> None:
    config = OrbitWinConfig(
        search=SearchConfig(
            one_burn_samples=32,
            one_burn_grid_burns=4,
            one_burn_grid_arrivals=8,
            max_revolutions=1,
            keep_candidates=50,
            event_step_s=5.0,
            two_burn_enabled=True,
            two_burn_samples=128,
        )
    )
    script = project_root / "examples" / "synthetic" / "HighPlaneChange_Circular.script"
    scenario, _ = parse_gmat_script(script, config)
    candidates = search_two_burn_global(scenario, config)
    assert candidates
    assert all(len(candidate.burns) == 2 for candidate in candidates)
    assert all(burn.magnitude_km_s <= config.policy.dv_safe_limit_km_s for candidate in candidates for burn in candidate.burns)
