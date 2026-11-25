"""An open template whose last Propagate carries a NUL or a Ctrl-Z renders runnably.

The renderer copies that line into every segment. A stop character copied
into an earlier segment put the Maneuvers and segments after it past where
GMAT stops reading, and the re-read refused the render, so solve and certify
aborted after the whole search. Only the last copy keeps it now.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from conftest import quick_config

from orbitwin.config import OrbitWinConfig, load_config
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.models import SYNTHESIZED_BURN_FRAME, Burn, Candidate
from orbitwin.pipeline import solve_scenario
from orbitwin.script_renderer import assert_final_script_policy, render_candidate

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
NO_MANEUVER = ROOT / "examples" / "synthetic" / "NoManeuver_Circular.script"
TAILS = {
    "nul-glued": "\x00\x00\x00",
    "nul-in-comment": " % end\x00\n",
    "ctrl-z-in-comment": " % end\x1a\n",
}


def _r1_config() -> OrbitWinConfig:
    config = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    return dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, horizon_s=6711.667),
        search=dataclasses.replace(config.search, ranking_search=False),
    )


def _open_config() -> OrbitWinConfig:
    return dataclasses.replace(quick_config(), pursuer_name="Pursuer", target_name="AlienSC")


TEMPLATES = {"R1": (R1, _r1_config), "NoManeuver": (NO_MANEUVER, _open_config)}


def _with_tail(tmp_path: Path, template: Path, tail: str) -> Path:
    text = template.read_text(encoding="ascii")
    assert text.endswith("};\n")
    script = tmp_path / f"{template.stem}.script"
    script.write_text(text.removesuffix("\n") + tail, encoding="ascii")
    return script


def _two_burns() -> Candidate:
    burns = tuple(
        Burn(time_s=t, dv_inertial_km_s=dv, dv_script_km_s=dv, slot_index=i,
             frame=SYNTHESIZED_BURN_FRAME)
        for i, (t, dv) in enumerate([(10.0, (0.01, 0.0, 0.0)), (200.0, (0.0, 0.01, 0.0))])
    )
    return Candidate(burns=burns, intercept_time_s=None, minimum_distance_km=800.0,
                     distance_at_completion_km=800.0, successful=False,
                     analytical_endpoint_error_km=None, branch="test", metadata={})


@pytest.mark.parametrize("tail", sorted(TAILS))
@pytest.mark.parametrize("template", sorted(TEMPLATES))
def test_the_stop_stays_on_the_last_copy_only(tmp_path: Path, template: str, tail: str) -> None:
    path, make_config = TEMPLATES[template]
    config = make_config()
    original = _with_tail(tmp_path, path, TAILS[tail])
    scenario, document = parse_gmat_script(original, config)
    rendered = render_candidate(document, scenario, _two_burns(), config).script_bytes
    final = tmp_path / "final.script"
    final.write_bytes(rendered)
    # Read back as GMAT reads it: three segments, two Maneuvers, all before the stop.
    reread, _ = parse_gmat_script(final, config)
    assert len(reread.propagate_segments) == 3
    text = rendered.decode("ascii")
    assert text.count("Maneuver ") == 2
    stop = TAILS[tail].strip(" \n")
    assert text.count(stop) == 1 and text.rstrip("\n").endswith(stop)
    # The earlier copies are the command alone.
    copies = [line for line in text.splitlines() if line.lstrip().startswith("Propagate")]
    assert copies[:-1] and all(line.endswith("};") for line in copies[:-1])
    assert_final_script_policy(original, final, config)


def test_solve_completes_on_a_template_ending_in_a_stop(tmp_path: Path) -> None:
    """At #173's bb4c71d this raised ParseError after the search, from the render."""
    config = _r1_config()
    # A grid just wide enough for one-burn misses, which render two segments.
    search = dataclasses.replace(
        quick_config().search, one_burn_grid_burns=4, one_burn_grid_arrivals=4
    )
    config = dataclasses.replace(config, search=search)
    original = _with_tail(tmp_path, R1, TAILS["ctrl-z-in-comment"])
    with pytest.warns(RuntimeWarning, match="SPAD"):
        summary = solve_scenario(original, config, tmp_path / "solve")
    assert summary["best_candidate"]["burns"]
    segments = []
    for script in (tmp_path / "solve").rglob("candidate.script"):
        reread, _ = parse_gmat_script(script, config)
        segments.append(len(reread.propagate_segments))
    assert max(segments) >= 2


def test_an_earlier_copy_cannot_end_in_dots() -> None:
    """`% coast...` before the stop would continue the copy into the next Maneuver."""
    from orbitwin.script_renderer import _without_stop

    line = "Propagate P(A, B) {A.ElapsedSecs = 1, B.ElapsedSecs = 1}; % coast...\x1a"
    assert _without_stop(line, "\n") == "Propagate P(A, B) {A.ElapsedSecs = 1, B.ElapsedSecs = 1};\n"
