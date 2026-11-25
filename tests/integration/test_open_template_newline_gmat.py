"""On stock GmatConsole, burns rendered into a template with no final newline fly."""

from __future__ import annotations

import dataclasses
import itertools
import math
from pathlib import Path

import pytest

from orbitwin.config import load_config
from orbitwin.gmat_parser import parse_gmat_script
from orbitwin.gmat_runner import run_gmat, said_by_gmat
from orbitwin.models import SYNTHESIZED_BURN_FRAME, Burn, Candidate
from orbitwin.script_renderer import assert_final_script_policy, render_candidate

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
REPORT = "ODC_2026_Team54.txt"

pytestmark = pytest.mark.gmat


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_r1_without_a_final_newline_flies_its_burns(
    tmp_path: Path, newline: str, gmat: tuple[str, str]
) -> None:
    console, startup = gmat
    text = R1.read_text(encoding="utf-8").replace("6711.667", "600").removesuffix("\n")
    template = tmp_path / "official.script"
    template.write_bytes(text.replace("\n", newline).encode("ascii"))
    config = load_config(ROOT / "config" / "ranking.yaml")
    config = dataclasses.replace(config, policy=dataclasses.replace(config.policy, team_number=54))
    # 600 s keeps the run short. It is before SC_Alien's perigee at 4500 s, which §3
    # rules out for an announced T_max, so the test names it as the override would.
    config = dataclasses.replace(
        config, policy=dataclasses.replace(
            config.policy, horizon_s=600.0, accept_announced_horizon_before_perigee_s=600.0
        )
    )
    scenario, document = parse_gmat_script(template, config)
    burns = tuple(
        Burn(time_s=t, dv_inertial_km_s=dv, dv_script_km_s=dv, slot_index=i,
             frame=SYNTHESIZED_BURN_FRAME)
        for i, (t, dv) in enumerate([(10.0, (0.01, 0.0, 0.0)), (200.0, (0.0, 0.01, 0.0))])
    )
    miss = Candidate(
        burns=burns, intercept_time_s=None, minimum_distance_km=800.0,
        distance_at_completion_km=800.0, successful=False, analytical_endpoint_error_km=None,
        branch="test", metadata={"event_refined": True},
    )
    final = tmp_path / "final.script"
    final.write_bytes(render_candidate(document, scenario, miss, config).script_bytes)
    assert_final_script_policy(template, final, config)

    run = run_gmat(console, final, tmp_path / "run", [REPORT], startup_file=startup)
    assert run.success, said_by_gmat(run)
    rows = Path(run.reports[REPORT]).read_text(encoding="utf-8").splitlines()
    assert float(rows[-1].split()[0]) == pytest.approx(600.0, abs=1e-3)


@pytest.mark.parametrize("tail", [b"\x00\x00\x00", b" % coast\x1a\n"], ids=["nul", "ctrl-z"])
def test_r1_ending_in_nul_padding_flies_every_segment(
    tmp_path: Path, tail: bytes, gmat: tuple[str, str]
) -> None:
    """The stop stays on the last copy of the Propagate, and the earlier copies carry
    the command alone, so GMAT reads both burns and all three segments."""
    console, startup = gmat
    text = R1.read_text(encoding="utf-8").replace("6711.667", "600").removesuffix("\n")
    template = tmp_path / "official.script"
    template.write_bytes(text.encode("ascii") + tail)
    config = load_config(ROOT / "config" / "ranking.yaml")
    config = dataclasses.replace(config, policy=dataclasses.replace(config.policy, team_number=54))
    # Before R1's perigee at 4500 s, so named as the override would; see above.
    config = dataclasses.replace(
        config, policy=dataclasses.replace(
            config.policy, horizon_s=600.0, accept_announced_horizon_before_perigee_s=600.0
        )
    )
    scenario, document = parse_gmat_script(template, config)
    burns = tuple(
        Burn(time_s=t, dv_inertial_km_s=dv, dv_script_km_s=dv, slot_index=i,
             frame=SYNTHESIZED_BURN_FRAME)
        for i, (t, dv) in enumerate([(10.0, (0.01, 0.0, 0.0)), (200.0, (0.0, 0.01, 0.0))])
    )
    miss = Candidate(
        burns=burns, intercept_time_s=None, minimum_distance_km=800.0,
        distance_at_completion_km=800.0, successful=False, analytical_endpoint_error_km=None,
        branch="test", metadata={"event_refined": True},
    )
    final = tmp_path / "final.script"
    final.write_bytes(render_candidate(document, scenario, miss, config).script_bytes)
    assert final.read_bytes().count(tail.strip(b" \n")) == 1
    assert_final_script_policy(template, final, config)

    run = run_gmat(console, final, tmp_path / "run", [REPORT], startup_file=startup)
    assert run.success, said_by_gmat(run)
    rows = [
        [float(value) for value in row.split()]
        for row in Path(run.reports[REPORT]).read_text(encoding="utf-8").splitlines()[1:]
    ]
    assert rows[-1][0] == pytest.approx(600.0, abs=1e-3)
    # SC_Human's velocity, columns 12-14, changes smoothly under gravity from one 1 s
    # row to the next; a burn kicks the step it falls in and, against it, the next.
    times = [row[7] for row in rows]
    change = [
        [a - b for a, b in zip(after[11:14], before[11:14], strict=True)]
        for before, after in itertools.pairwise(rows)
    ]
    kicks = [(times[k], math.dist(change[k], change[k - 1])) for k in range(1, len(change))]
    burns_seen = [(round(t), dv) for t, dv in kicks if dv > 0.005]
    assert [t for t, _ in burns_seen] == [10, 11, 200, 201], burns_seen
    assert all(dv == pytest.approx(0.01, rel=0.05) for _, dv in burns_seen), burns_seen
