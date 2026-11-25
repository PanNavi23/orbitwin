from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import bare_environment

from orbitwin.constraints import DV_COMPARISON_TOLERANCE_KM_S

#: The preliminary's score block, and the ranking rules' as revised 2026/11/04: m/s and C_r.
SCORE_YAML = {
    "preliminary": "score:\n  kt: 0.003\n  ct: 3000\n  kv: 1.0\n  cv: 1.5\n  time_unit: s\n"
                   "  delta_v_unit: km/s\n",
    "ranking": "score:\n  kt: 0.003\n  ct: 3000\n  kv: 0.001\n  cv: 1500\n  time_unit: s\n"
               "  delta_v_unit: m/s\n  cr_km: 100\n",
}


def _near_miss(tmp_path: Path, project_root: Path, workshop_script: Path, rules: str,
               *extra: str, score: bool = True) -> subprocess.CompletedProcess[str]:
    """AlienSC is moved onto a 6950 km orbit just ahead of HumanSC, so with no
    burns they pass 50 km apart at 1127 s and are some 500 km apart at 7200 s."""
    script = tmp_path / "near.script"
    text = workshop_script.read_text(encoding="utf-8")
    for old, new in (
        ("AlienSC.SMA = 9600;", "AlienSC.SMA = 6950;"),
        ("AlienSC.TA = 311.265;", "AlienSC.TA = 250.766;"),
    ):
        assert old in text
        text = text.replace(old, new)
    script.write_text(text, encoding="utf-8")
    burns = tmp_path / "burns.json"
    burns.write_text('{"burns": []}', encoding="utf-8")
    config = tmp_path / "config.yaml"
    limits = (
        "  dv_limit_km_s: 0.25\n  dv_safe_limit_km_s: 0.249\n"
        "  total_dv_limit_km_s: 3.0\n  total_dv_safe_limit_km_s: 2.99\n"
        if rules == "ranking" else ""
    )
    config.write_text(f"policy:\n  rules: {rules}\n  horizon_s: 7200\n{limits}"
                      + (SCORE_YAML[rules] if score else ""), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(project_root / "scripts" / "audit_burn_table.py"),
         str(script), str(burns), "--config", str(config), "--step-s", "10",
         *(["--rules", rules] if rules != "ranking" else []), *extra],
        capture_output=True, text=True, check=False,
        env=bare_environment(PYTHONPATH=str(project_root / "src"), PATH="/usr/bin:/bin"),
    )


def test_c_r_is_read_from_the_config_and_the_flag_wins(
    tmp_path: Path, project_root: Path, workshop_script: Path
) -> None:
    def priced(done: subprocess.CompletedProcess[str]) -> tuple[float, float]:
        closest = float(re.search(r"closest approach: (\S+) km", done.stdout).group(1))
        return closest, float(re.search(r"m/s: distance (\S+) \+.*<- config",
                                        done.stdout).group(1))

    (tmp_path / "config").mkdir()
    (tmp_path / "flag").mkdir()
    closest, from_config = priced(_near_miss(tmp_path / "config", project_root,
                                             workshop_script, "ranking"))
    flag = _near_miss(tmp_path / "flag", project_root, workshop_script, "ranking", "--cr-km", "30")
    assert "C_r           = 30 km" in flag.stdout
    _, from_flag = priced(flag)
    for cr_km, points in ((100.0, from_config), (30.0, from_flag)):
        assert points == pytest.approx(
            100.0 / (1.0 + math.exp(((closest - 5.0) / cr_km) ** 2)), abs=1e-4
        )
    assert from_flag < from_config - 10.0


def test_c_r_is_refused_under_the_preliminary(
    tmp_path: Path, project_root: Path, workshop_script: Path
) -> None:
    done = _near_miss(tmp_path, project_root, workshop_script, "preliminary", "--cr-km", "100")
    assert done.returncode == 2 and "--cr-km is the ranking rules' C_r" in done.stderr


@pytest.mark.parametrize("cr_km", ["-30", "0", "inf", "nan"])
def test_c_r_must_be_a_positive_number_of_km(
    tmp_path: Path, project_root: Path, workshop_script: Path, cr_km: str
) -> None:
    done = _near_miss(tmp_path, project_root, workshop_script, "ranking", "--cr-km", cr_km)
    assert done.returncode == 2
    assert f"argument --cr-km: not a positive finite number: '{cr_km}'" in done.stderr


def test_c_r_is_only_taken_with_its_unit(
    tmp_path: Path, project_root: Path, workshop_script: Path
) -> None:
    done = _near_miss(tmp_path, project_root, workshop_script, "ranking", "--cr", "100")
    assert done.returncode == 2 and "unrecognized arguments: --cr" in done.stderr


def test_an_unused_c_r_is_said_to_be_unused(
    tmp_path: Path, project_root: Path, workshop_script: Path
) -> None:
    (tmp_path / "bare").mkdir()
    bare = _near_miss(tmp_path / "bare", project_root, workshop_script, "ranking", "--cr-km", "30",
                      score=False)
    assert bare.returncode == 0, bare.stdout
    assert "--cr-km not used: without the other coefficients nothing is scored" in bare.stdout
    # A force model two-body leaves out: nothing is scored at all.
    richer = _audit_r1(tmp_path, project_root, [(0.001, [0.1, 0.0, 0.0])], "--cr-km", "30")
    assert "--cr-km not used: nothing is scored here" in richer.stdout


@pytest.mark.parametrize("rules", ["preliminary", "ranking"])
def test_a_miss_is_audited_under_the_rulebook_it_was_set(
    tmp_path: Path, project_root: Path, workshop_script: Path, rules: str
) -> None:
    """The preliminary scores Δr at T_max, the ranking rules the closest approach."""
    done = _near_miss(tmp_path, project_root, workshop_script, rules)
    closest = float(re.search(r"closest approach: (\S+) km", done.stdout).group(1))
    at_end = float(re.search(r"separation at T_max: (\S+) km", done.stdout).group(1))
    distance = float(re.search(r"m/s: distance (\S+) \+.*<- config", done.stdout).group(1))
    assert closest < 60.0 < 400.0 < at_end
    if rules == "ranking":  # §4 eq. (5) as revised 2026/11/04, C_r 100 km
        expected = 100.0 / (1.0 + math.exp(((closest - 5.0) / 100.0) ** 2))
    else:
        expected = 50.0 * math.exp(-(at_end - 5.0) / 100.0)
    assert distance == pytest.approx(expected, abs=1e-4)
    # A ranking miss is an answer; only the walls are judged, over the whole run.
    checks = _checks(done.stdout)
    # Priced on report rows, which this replay does not have.
    assert ("scored on GMAT's closest report row" in done.stdout) is (rules == "ranking")
    if rules == "ranking":
        assert "enters the sphere" not in checks
        assert checks["outside the central body through T_max"] == "PASS"
        assert done.returncode == 0, done.stdout
    else:
        assert checks["enters the sphere"] == "FAIL"
        assert done.returncode == 1


def _audit_ranking_table(
    tmp_path: Path, project_root: Path, workshop_script: Path, magnitudes_km_s: list[float]
) -> subprocess.CompletedProcess[str]:
    burns = tmp_path / "burns.json"
    burns.write_text(json.dumps({"burns": [
        {"time_s": 101.0 * index, "dv_inertial_km_s": [magnitude, 0.0, 0.0]}
        for index, magnitude in enumerate(magnitudes_km_s)
    ]}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "policy:\n  rules: ranking\n  horizon_s: 7200\n"
        "  dv_limit_km_s: 0.25\n  dv_safe_limit_km_s: 0.249\n"
        "  total_dv_limit_km_s: 3.0\n  total_dv_safe_limit_km_s: 2.99\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, str(project_root / "scripts" / "audit_burn_table.py"),
         str(workshop_script), str(burns), "--config", str(config), "--step-s", "10"],
        capture_output=True, text=True, check=False,
        env=bare_environment(PYTHONPATH=str(project_root / "src"), PATH="/usr/bin:/bin"),
    )


def test_a_ranking_miss_that_dives_fails_both_walls(
    tmp_path: Path, project_root: Path, workshop_script: Path
) -> None:
    """Ten retrograde 249 m/s burns: a miss, judged on the walls over the whole run."""
    import numpy as np

    from orbitwin.config import load_config
    from orbitwin.gmat_parser import parse_gmat_script
    from orbitwin.models import Burn, as_vector3
    from orbitwin.trajectory import state_with_burns_at

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "policy:\n  rules: ranking\n  horizon_s: 7200\n"
        "  dv_limit_km_s: 0.25\n  dv_safe_limit_km_s: 0.249\n"
        "  total_dv_limit_km_s: 3.0\n  total_dv_safe_limit_km_s: 2.99\n",
        encoding="utf-8",
    )
    scenario, _ = parse_gmat_script(workshop_script, load_config(config_path))
    burns: list[Burn] = []
    for index in range(10):
        time_s = 1.0 + 101.0 * index
        state = state_with_burns_at(
            scenario.pursuer.initial_state, tuple(burns), time_s, scenario.mu_km3_s2
        )
        retro = -np.asarray(state.v) / float(np.linalg.norm(state.v)) * 0.249
        burns.append(Burn(time_s, as_vector3(retro), as_vector3(retro)))
    table = tmp_path / "burns.json"
    table.write_text(json.dumps({"burns": [
        {"time_s": burn.time_s, "dv_inertial_km_s": list(burn.dv_inertial_km_s)}
        for burn in burns
    ]}), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "audit_burn_table.py"),
         str(workshop_script), str(table), "--config", str(config_path), "--step-s", "10"],
        capture_output=True, text=True, check=False,
        env=bare_environment(PYTHONPATH=str(project_root / "src"), PATH="/usr/bin:/bin"),
    )
    checks = _checks(done.stdout)
    assert "enters the sphere" not in checks, done.stdout
    assert checks["outside the central body through T_max"] == "FAIL"
    assert checks["100 km altitude margin through T_max"] == "FAIL"
    assert "post-interception" not in done.stdout
    assert done.returncode == 1


def _checks(stdout: str) -> dict[str, str]:
    return dict(
        (label, verdict)
        for verdict, label in re.findall(r"\[(PASS|FAIL)\] ([^:]+):", stdout)
    )


def test_a_table_over_the_total_fails_the_audit(
    tmp_path: Path, project_root: Path, workshop_script: Path
) -> None:
    """Thirteen legal 0.25 km/s burns spend 3.25 km/s, over the 3 km/s that disqualifies."""
    done = _audit_ranking_table(tmp_path, project_root, workshop_script, [0.25] * 13)
    checks = _checks(done.stdout)
    assert {checks[f"burn {index} magnitude"] for index in range(1, 14)} == {"PASS"}
    assert checks["total delta-v"] == "FAIL"
    assert done.returncode == 1


@pytest.mark.parametrize(
    ("last_km_s", "verdict"),
    [
        (0.25, "PASS"),  # eleven 0.25 burns and this one sum to exactly 3.0
        (0.25 + 8 * math.ulp(0.25), "FAIL"),  # sums to one ulp over 3.0
    ],
)
def test_the_total_is_held_with_no_slack(
    tmp_path: Path, project_root: Path, workshop_script: Path, last_km_s: float, verdict: str
) -> None:
    assert (sum([0.25] * 11 + [last_km_s]) == 3.0) is (verdict == "PASS")
    done = _audit_ranking_table(tmp_path, project_root, workshop_script, [0.25] * 11 + [last_km_s])
    checks = _checks(done.stdout)
    assert checks["burn 12 magnitude"] == "PASS"
    assert checks["total delta-v"] == verdict


def test_the_per_burn_check_uses_the_constraint_tolerance(
    tmp_path: Path, project_root: Path, workshop_script: Path
) -> None:
    over = 0.25 + 2 * DV_COMPARISON_TOLERANCE_KM_S
    done = _audit_ranking_table(tmp_path, project_root, workshop_script, [0.25, over])
    checks = _checks(done.stdout)
    assert (checks["burn 1 magnitude"], checks["burn 2 magnitude"]) == ("PASS", "FAIL")


def _audit_r1(
    tmp_path: Path, project_root: Path, burns: list[tuple[float, list[float]]], *extra: str
) -> subprocess.CompletedProcess[str]:
    """R1 flies the ranking §7 force model, which a two-body replay leaves out."""
    table = tmp_path / "burns.json"
    table.write_text(json.dumps({"burns": [
        {"time_s": time_s, "dv_inertial_km_s": dv} for time_s, dv in burns
    ]}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "pursuer_name: SC_Human\ntarget_name: SC_Alien\n"
        "policy:\n  rules: ranking\n  horizon_s: 6711.667\n"
        "  dv_limit_km_s: 0.25\n  dv_safe_limit_km_s: 0.249\n"
        "  total_dv_limit_km_s: 3.0\n  total_dv_safe_limit_km_s: 2.99\n"
        + SCORE_YAML["ranking"],
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, str(project_root / "scripts" / "audit_burn_table.py"),
         str(project_root / "examples" / "ranking" / "R1_NearPlane_Reachable.script"),
         str(table), "--config", str(config), "--step-s", "10", *extra],
        capture_output=True, text=True, check=False,
        env=bare_environment(PYTHONPATH=str(project_root / "src"), PATH="/usr/bin:/bin"),
    )


@pytest.mark.parametrize(
    ("dv", "two_body"),
    [
        # R1's certified one-burn hit (GMAT 96.39); two-body misses by 19.97 km.
        ([0.15872224131162743, 0.031197795515318275, 0.1450739659057696], "fail"),
        # A two-body Lambert hit at 4500 s, whose interception time is not GMAT's.
        ([0.07808053418311323, 0.041284848835042354, -0.039418636552082886], "pass"),
    ],
)
def test_a_two_body_replay_of_a_richer_model_does_not_judge_the_interception(
    tmp_path: Path, project_root: Path, dv: list[float], two_body: str
) -> None:
    done = _audit_r1(tmp_path, project_root, [(0.001, dv)])
    checks = _checks(done.stdout)
    assert "drag" in done.stdout
    assert "enters the sphere" not in checks
    assert f"(two-body {two_body}, not judged)" in done.stdout
    assert not any("up to the interception" in label for label in checks)
    assert "Cv read as" not in done.stdout and "T_team        =" not in done.stdout
    # Nor is the distance: the score is GMAT's.
    assert "closest report row" not in done.stdout
    assert checks["burn 1 magnitude"] == "PASS"
    assert checks["outside the central body through T_max"] == "PASS"
    assert checks["100 km altitude margin through T_max"] == "PASS"
    assert checks["burn 1 moves velocity only"] == "PASS"
    assert checks["energy conserved on the final coast"] == "PASS"
    assert done.returncode == 0, done.stdout
    assert "AUDIT: PASS (interception and score not judged" in done.stdout


@pytest.mark.parametrize("extra", [(), ("--require-safe-through-horizon",)])
def test_a_dive_under_a_richer_model_fails_the_walls(
    tmp_path: Path, project_root: Path, extra: tuple[str, ...]
) -> None:
    """Ten burns two-body takes to 4199 km; GMAT R2026a flew it to 73.6 km altitude and aborted."""
    dives = [(1.0 + 101.0 * k, [-0.2, -0.1, 0.0]) for k in range(10)]
    done = _audit_r1(tmp_path, project_root, dives, *extra)
    checks = _checks(done.stdout)
    assert checks["outside the central body through T_max"] == "FAIL"
    assert checks["100 km altitude margin through T_max"] == "FAIL"
    assert done.returncode == 1, done.stdout
    assert "AUDIT: FAIL (interception and score not judged" in done.stdout


def test_legality_is_still_judged_under_a_richer_model(
    tmp_path: Path, project_root: Path
) -> None:
    done = _audit_r1(tmp_path, project_root, [(0.001, [0.2501, 0.0, 0.0])])
    assert _checks(done.stdout)["burn 1 magnitude"] == "FAIL"
    assert done.returncode == 1


def test_a_non_force_warning_leaves_a_point_mass_template_judged(
    tmp_path: Path, project_root: Path, workshop_script: Path
) -> None:
    """An eccentric target draws the briefing warning, which is not a force gap."""
    script = tmp_path / "eccentric.script"
    text = workshop_script.read_text(encoding="utf-8")
    old = "AlienSC.ECC = "
    assert text.count(old) == 1
    head, tail = text.split(old)
    script.write_text(f"{head}{old}0.05;{tail.split(';', 1)[1]}", encoding="utf-8")
    burns = tmp_path / "burns.json"
    burns.write_text('{"burns": []}', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("policy:\n  rules: preliminary\n", encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "audit_burn_table.py"),
         str(script), str(burns), "--config", str(config), "--step-s", "10",
         "--rules", "preliminary"],
        capture_output=True, text=True, check=False,
        env=bare_environment(PYTHONPATH=str(project_root / "src"), PATH="/usr/bin:/bin"),
    )
    from orbitwin.config import load_config
    from orbitwin.gmat_parser import parse_gmat_script

    scenario, _ = parse_gmat_script(script, load_config(config))
    assert scenario.warnings and scenario.two_body_gaps == ()
    assert _checks(done.stdout)["enters the sphere"] == "FAIL"
    assert "not judged" not in done.stdout
    assert done.returncode == 1
