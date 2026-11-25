"""A verified report in each ReportFile format gets the default's verdict on GmatConsole."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from orbitwin.gmat_runner import run_gmat

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "tests" / "fixtures" / "report_rows"
HIT = ROWS / "R1_hit.script"
MISS = ROOT / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
REPORT = "ODC_2026_TeamXX.txt"
GOAL = "4517.611829571932"
#: 30 s on, past the sphere: the last row is outside.
OVERRUN = "4547.611829571932"
FORMATS: dict[str, dict[str, str]] = {
    "default": {},
    "comma": {"Delimiter": "','"},
    "semicolon": {"Delimiter": "';'"},
    "noheader": {"WriteHeaders": "false"},
    "nofixed": {"FixedWidth": "false"},
    "noheader_comma_nofixed": {"WriteHeaders": "false", "Delimiter": "','", "FixedWidth": "false"},
    "backslash_t": {"Delimiter": "'\\t'"},
}
KINDS = {"hit": 0, "over": 1, "miss": 1}


def _script(base: Path, fields: dict[str, str], goal: str | None) -> str:
    lines = []
    for line in base.read_text(encoding="utf-8").splitlines(keepends=True):
        key = line.split("=", 1)[0].strip().removeprefix("ReportFile1.")
        if key in fields:
            line = f"ReportFile1.{key} = {fields[key]};\n"
        elif goal is not None and line.startswith("Propagate") and GOAL in line:
            line = line.replace(GOAL, goal)
        lines.append(line)
    return "".join(lines)


def _verdict(tmp_path: Path, gmat: tuple[str, str], kind: str, fmt: str) -> tuple[int, str]:
    console, startup = gmat
    run_dir = tmp_path / f"{kind}-{fmt}"
    run_dir.mkdir()
    script = run_dir / "final.script"
    base = MISS if kind == "miss" else HIT
    script.write_text(
        _script(base, FORMATS[fmt], OVERRUN if kind == "over" else None), encoding="utf-8"
    )
    result = run_gmat(
        console, script, run_dir / "run", ["ODC_2026_TeamXX.txt"],
        startup_file=startup, timeout_s=600,
    )
    assert result.success, result.error
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_report_rows.py"),
         result.reports["ODC_2026_TeamXX.txt"], "--script", str(script)],
        capture_output=True, text=True, check=False,
    )
    # The file and script lines name the run's own paths.
    out = "\n".join(line for line in done.stdout.splitlines()
                    if not line.startswith(("file ", "script ")))
    return done.returncode, out + done.stderr


@pytest.mark.gmat
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_format_gets_the_default_verdict(
    kind: str, tmp_path: Path, gmat: tuple[str, str]
) -> None:
    with ThreadPoolExecutor(4) as pool:
        verdicts = dict(
            zip(FORMATS, pool.map(lambda fmt: _verdict(tmp_path, gmat, kind, fmt), FORMATS),
                strict=True)
        )
    default = verdicts.pop("default")
    assert default[0] == KINDS[kind], default
    assert "VERDICT" in default[1]
    for fmt, verdict in verdicts.items():
        assert verdict == default, fmt


PROBES: dict[str, dict[str, str]] = {
    name: probe.get("fields", {})
    for name, probe in json.loads((ROWS / "MANIFEST.json").read_text(encoding="utf-8"))[
        "probes"
    ].items()
}


def _check(report: Path | str, script: Path) -> tuple[int, str]:
    # The probes are formats; most carry no velocity, so they bind on their end.
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_report_rows.py"), str(report),
         "--script", str(script), "--accept-end-time-binding-only"],
        capture_output=True, text=True, check=False,
    )
    # Exit 3: a pass bound on its end only, which the flag accepts.
    code = 0 if done.returncode == 3 else done.returncode
    return code, done.stdout + done.stderr


@pytest.mark.gmat
def test_text_columns_frames_and_a_second_run_on_gmatconsole(
    tmp_path: Path, gmat: tuple[str, str]
) -> None:
    console, startup = gmat

    def fly(name: str, again: bool = False) -> tuple[int, str]:
        run_dir = tmp_path / name
        run_dir.mkdir()
        script = run_dir / "final.script"
        script.write_text(_script(HIT, PROBES[name], None), encoding="utf-8")
        result = run_gmat(console, script, run_dir / "run", [REPORT], startup_file=startup,
                          timeout_s=600)
        assert result.success, result.error
        if again:
            # Into the same output directory, keeping the report.
            assert run_gmat(console, script, run_dir / "run", [], startup_file=startup,
                            timeout_s=600).success
        return _check(result.reports[REPORT], script)

    flown = ["greg_default", "greg_comma", "greg_noheader_nofixed",
             "greg_noheader_comma_nofixed", "short_names", "earthfixed", "greg_noheader"]
    with ThreadPoolExecutor(4) as pool:
        verdicts = dict(zip(flown, pool.map(fly, flown), strict=True))
    for name in flown[:-1]:
        code, out = verdicts[name]
        assert code == 0 and "LAST row      : t = 4517.612829 s   dr = 4.950002 km" in out, name
    assert verdicts["greg_noheader"][0] == 1 and "VERDICT" not in verdicts["greg_noheader"][1]
    code, out = fly("altitude_first")
    assert code == 0 and "dr = 4.950002 km" in out and "VERDICT: the LAST row is inside" in out
    for name in ("append_noheader", "altitude_first_append_noheader"):
        code, out = fly(name, again=True)
        assert code == 1 and "goes back from" in out and "VERDICT" not in out, name
