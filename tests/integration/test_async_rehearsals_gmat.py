"""R6-R9 on stock GmatConsole: at the later epoch each is R1's or R2's geometry,
R9's flown on by its late_moved_s."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from orbitwin.config import load_config
from orbitwin.errors import ParseError
from orbitwin.gmat_runner import run_gmat
from orbitwin.report_parser import parse_numeric_report
from orbitwin.state_probe import load_scenario

ROOT = Path(__file__).resolve().parents[2]
RANKING = ROOT / "examples" / "ranking"
_TABLE = yaml.safe_load((RANKING / "async_scenarios.yaml").read_text(encoding="utf-8"))
_ROWS = _TABLE["scenarios"]
_SYNC = {
    row["script"]: row
    for row in yaml.safe_load((RANKING / "scenarios.yaml").read_text(encoding="utf-8"))["scenarios"]
}

pytestmark = pytest.mark.gmat


def _config(horizon_s: float, policy: dict[str, Any]):
    config = load_config(ROOT / "config" / "ranking-rehearsal.yaml")
    config = dataclasses.replace(config, policy=dataclasses.replace(config.policy, team_number=54))
    return dataclasses.replace(
        config, policy=dataclasses.replace(config.policy, horizon_s=horizon_s, **policy)
    )


def _flown_on(
    script: Path, seconds: float, console: str, startup: str, workdir: Path
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each craft of ``script`` flown alone ``seconds`` by GMAT, as the builder flies them."""
    resources = script.read_text(encoding="utf-8").split("BeginMissionSequence;")[0]
    columns = [f"{craft}.EarthMJ2000Eq.{axis}" for craft in ("SC_Alien", "SC_Human")
               for axis in ("X", "Y", "Z", "VX", "VY", "VZ")]
    workdir.mkdir(parents=True)
    flown = workdir / "flown_on.script"
    flown.write_text(
        resources + "Create ReportFile On;\nOn.Filename = 'on.txt';\nOn.Precision = 17;\n"
        "On.WriteHeaders = true;\nOn.Add = {" + ", ".join(columns) + "};\n\n"
        "BeginMissionSequence;\n"
        + "".join(f"Propagate DefaultProp({craft}) {{{craft}.ElapsedSecs = {seconds!r}}};\n"
                  for craft in ("SC_Alien", "SC_Human")),
        encoding="utf-8",
    )
    run = run_gmat(console, flown, workdir / "run", ["on.txt"], startup_file=startup,
                   timeout_s=600.0, template_dir=script.parent)
    assert run.success, run.error
    last = parse_numeric_report(run.reports["on.txt"]).data[-1]
    return {craft: (last[6 * k:6 * k + 3], last[6 * k + 3:6 * k + 6])
            for k, craft in enumerate(("SC_Alien", "SC_Human"))}


@pytest.mark.parametrize("row", _ROWS, ids=lambda row: row["script"])
def test_at_the_later_epoch_each_is_its_source_geometry(
    row: dict[str, Any], tmp_path: Path, gmat: tuple[str, str]
) -> None:
    console, startup = gmat
    scenario, _, _ = load_scenario(
        RANKING / row["script"], _config(row["horizon_s"], row["policy"]),
        gmat_console=console, startup_file=startup,
    )
    source = _SYNC[row["from"]]
    original, _, _ = load_scenario(
        RANKING / row["from"], _config(source["horizon_s"], {}),
        gmat_console=console, startup_file=startup,
    )
    probe = scenario.state_probe
    assert probe is not None and probe.aligned is not None
    read = probe.aligned.run.data_file_sha256
    moved = sorted(
        key for key, sha in _TABLE["built_with"].items() if read.get(key, {}).get("sha256") != sha
    )
    if moved:
        pytest.fail(
            f"GMAT's {', '.join(moved)} differ from the ones R6-R9 were built with; "
            "rebuild them with scripts/build_async_rehearsals.py and update built_with"
        )
    alignment = scenario.alignment
    assert alignment is not None and alignment.early == row["early"]
    assert alignment.offset_s == pytest.approx(row["lead_s"], abs=1.0e-6)
    # Measured 2026-11-02: within 5e-9 km after GMAT flies the 1800 s back and forth.
    moved_s = float(row.get("late_moved_s", 0.0))
    expected = {
        craft.name: (craft.initial_state.r, craft.initial_state.v)
        for craft in (original.pursuer, original.target)
    }
    if moved_s:
        expected = _flown_on(RANKING / row["from"], moved_s, console, startup, tmp_path / "on")
    for mine in (scenario.pursuer, scenario.target):
        r, v = expected[mine.name]
        assert float(np.linalg.norm(mine.initial_state.r - r)) < 1.0e-6, mine.name
        assert float(np.linalg.norm(mine.initial_state.v - v)) < 1.0e-9, mine.name
    assert scenario.max_time_s == pytest.approx(source["horizon_s"] - moved_s, abs=1.0e-6)
    if row["policy"].get("add_epoch_alignment"):
        assert alignment.line_index is None
        with pytest.raises(ParseError, match="add_epoch_alignment"):
            load_scenario(
                RANKING / row["script"],
                _config(row["horizon_s"], {**row["policy"], "add_epoch_alignment": False}),
                gmat_console=console, startup_file=startup,
            )


@pytest.mark.parametrize("announced", ["earlier", "later"])
@pytest.mark.parametrize("row", _ROWS, ids=lambda row: row["script"])
def test_the_time_origin_evidence_reads_each_epoch_back(
    row: dict[str, Any], announced: str, tmp_path: Path, gmat: tuple[str, str]
) -> None:
    """Each row's T_max counts from the earlier epoch, as the organizer answered, and reads
    back as the early craft's reading; the same T_max announced from the later epoch (Δ
    less) reads back as the other, and the verdict line says so. Each flies its own coast,
    as inspect does."""
    from orbitwin.time_origin import time_origin_evidence

    console, startup = gmat
    lead = float(row["lead_s"]) if announced == "later" else 0.0
    config = _config(row["horizon_s"] - lead, row["policy"])
    early = "target" if row["early"] == "SC_Alien" else "pursuer"
    reading = early if announced == "earlier" else ("pursuer" if early == "target" else "target")
    if announced == "later" and row.get("late_moved_s"):
        # R9: read from the later epoch that T_max falls before SC_Alien's perigee.
        with pytest.raises(ParseError, match="before SC_Alien's perigee"):
            load_scenario(RANKING / row["script"], config, gmat_console=console,
                          startup_file=startup)
        return
    scenario, document, _ = load_scenario(
        RANKING / row["script"], config, gmat_console=console, startup_file=startup
    )
    evidence = time_origin_evidence(scenario, document, config, console, startup, tmp_path / "c")
    assert evidence is not None and evidence.track_error is None, evidence
    assert evidence.configured_reading == early
    assert evidence.verdict == reading, evidence.lines
    supported = "T_max may have been entered from the other craft's epoch" in evidence.lines[-1]
    assert supported is (announced == "later"), evidence.lines[-1]
    # R7's alignment is ours, so its goal is not counted; nor is a goal of T_max
    # itself, which is what the later craft's reading reads.
    added = row["policy"].get("add_epoch_alignment") is True
    assert evidence.template_counted is not added
    counted = not added and evidence.readings[0 if reading == "target" else 1].craft == row["early"]
    assert evidence.template_supports == (reading if counted else None), evidence.lines
    assert evidence.reason == (
        "the template and the boundary support it" if counted else "the boundary supports it"
    ), evidence.lines


@pytest.mark.parametrize(
    "row", [row for row in _ROWS if row["policy"].get("add_epoch_alignment")],
    ids=lambda row: row["script"],
)
def test_baseline_flies_a_template_without_an_alignment(
    row: dict[str, Any], tmp_path: Path, gmat: tuple[str, str]
) -> None:
    """R2026a aborts R7 as written; baseline runs it with the line solve adds (2026-11-05)."""
    from orbitwin.verify import baseline_original_script

    console, startup = gmat
    evidence = baseline_original_script(
        RANKING / row["script"], _config(row["horizon_s"], row["policy"]), console, startup,
        tmp_path / "baseline",
    )
    added = evidence["epoch_alignment_added"]
    assert added is not None and f"{row['early']}.A1ModJulian" in added
    assert evidence["run"]["success"] is True
    assert evidence["run"]["script_sha256"] == evidence["ran_sha256"]
    assert evidence["original_sha256_before"] == evidence["original_sha256_after"]
    # Never numbered: baseline's report is not an upload.
    (report,) = (tmp_path / "baseline" / "run").rglob("ODC_2026_TeamXX.txt")
    assert len(report.read_text(encoding="utf-8").splitlines()) > row["lead_s"]
