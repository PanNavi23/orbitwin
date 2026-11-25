from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from orbitwin.config import PolicyConfig

# The matrix generator measures the pipeline on random different-plane geometry
# and treats any exception as a finding. That only works if every scenario it
# writes is one the parser accepts, and the raw ranges do not guarantee that:
# a1 reaches 6800 km with e1 up to 0.3, which puts a pericenter at 4760 km,
# inside the 6378.14 km Earth. gmat_parser refuses that state -- correctly, and
# tests/unit/test_scenario_premises.py is where that refusal is tested -- so an
# underground draw turned a correct refusal into a reported pipeline exception
# and took `make validate` down with it.


@pytest.fixture
def scenario_ood_matrix(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "orbitwin_scenario_ood_matrix", project_root / "scripts" / "scenario_ood_matrix.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pericenters(values: dict[str, float]) -> tuple[float, float]:
    return values["a1"] * (1.0 - values["e1"]), values["a2"] * (1.0 - values["e2"])


def test_every_sampled_case_is_in_orbit(scenario_ood_matrix) -> None:
    radius_km = PolicyConfig().central_body_radius_km
    rng = np.random.default_rng(20260818)
    for _ in range(200):
        pursuer_km, target_km = _pericenters(scenario_ood_matrix.sample_case(rng, radius_km))
        assert pursuer_km > radius_km
        assert target_km > radius_km


def test_the_seed_that_used_to_produce_underground_draws_no_longer_does(
    scenario_ood_matrix,
) -> None:
    # Seed 20260818, 20 cases: cases 8, 14, 17 and 19 drew a pericenter between
    # 5550 and 6323 km. Redrawing them costs six extra draws.
    radius_km = PolicyConfig().central_body_radius_km
    rng = np.random.default_rng(20260818)
    cases = [scenario_ood_matrix.sample_case(rng, radius_km) for _ in range(20)]
    assert min(min(_pericenters(values)) for values in cases) > radius_km


def test_rejection_does_not_narrow_the_apocentre_range(scenario_ood_matrix) -> None:
    # Rejecting low-pericenter draws must not quietly collapse the geometry the
    # matrix is there to cover: both plane change and altitude spread survive.
    radius_km = PolicyConfig().central_body_radius_km
    rng = np.random.default_rng(4242)
    cases = [scenario_ood_matrix.sample_case(rng, radius_km) for _ in range(100)]
    plane_deltas = [abs(values["i1"] - values["i2"]) for values in cases]
    assert min(plane_deltas) < 10.0
    assert max(plane_deltas) > 60.0
    assert max(values["a2"] for values in cases) > 16000.0
    assert min(values["a1"] for values in cases) < 8000.0
