"""scripts/ranking_day_config.py writes the day's config and nothing else."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from conftest import TEAM_NUMBER, tmax_as_announced_cases

from orbitwin.config import GmatConfig, OrbitWinConfig, load_config
from orbitwin.gmat_parser import parse_gmat_script

SCORE = ["--kt-per-s", "0.003", "--ct-s", "5369.334", "--kv", "0.0012", "--cv", "2000",
         "--cr-km", "300"]
#: Fits any machine, so a test about the score warnings sees no gmat.workers warning.
_ONE_GMAT = ["--gmat-workers", "1"]


#: Where the module's scripts are written: R1, and R1 with SC_Alien past its
#: perigee at t = 0, which any T_max fits; the T_max check has tests of its own.
_SCRIPTS: list[Path] = []


@pytest.fixture(autouse=True, scope="module")
def _script_dir(tmp_path_factory: pytest.TempPathFactory) -> None:
    _SCRIPTS[:] = [tmp_path_factory.mktemp("scripts")]


def _flag(args: tuple[str, ...], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args else None


def _script_for(project_root: Path, args: tuple[str, ...]) -> Path:
    """R1 under the craft names the config will carry; past its perigee unless a boundary is
    announced, where R1 itself is at 17000 km at 6711.667 s."""
    import yaml

    day = _flag(args, "--day-config")
    try:
        raw = yaml.safe_load(Path(day).read_text(encoding="utf-8")) if day else None
    except (OSError, UnicodeDecodeError, yaml.YAMLError):  # the tool's to refuse
        raw = None
    if not isinstance(raw, dict) or not isinstance(raw.get("policy"), dict):
        raw = {"policy": {}}
    pursuer = _flag(args, "--pursuer-name") or raw.get("pursuer_name") or "SC_Human"
    target = _flag(args, "--target-name") or raw.get("target_name") or "SC_Alien"
    bounded = "--mission-boundary-km" in args or raw["policy"].get("mission_boundary_km")
    text = (project_root / "examples" / "ranking" / "R1_NearPlane_Reachable.script").read_text(
        encoding="utf-8")
    if not bounded:
        old = "SC_Alien.TA = 241.21247020102464;"
        assert old in text
        text = text.replace(old, "SC_Alien.TA = 118.78752979897536;")
    text = text.replace("SC_Human", "@pursuer@").replace("SC_Alien", "@target@")
    text = text.replace("@pursuer@", pursuer).replace("@target@", target)
    path = _SCRIPTS[0] / f"R1-{pursuer}-{target}-{'bounded' if bounded else 'climbing'}.script"
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return path


def _environment(**gmat: str) -> dict[str, str]:
    """This process's environment, with only the GMAT variables in ``gmat``: BASH_ENV
    exports the real pair, which the tool now reads."""
    kept = {k: v for k, v in os.environ.items() if not k.startswith("GMAT_") and k != "BASH_ENV"}
    return {**kept, **gmat}


def _run(project_root: Path, *args: str, first: bool = True,
         gmat: dict[str, str] | None = None,
         team: str | None = str(TEAM_NUMBER)) -> subprocess.CompletedProcess[str]:
    """A scored run names a first config for the day, written from its own --horizon-s and
    --base, unless it names one or ``first`` is False. Without --script, _script_for's.
    A first config gets --team-number ``team`` unless it names one or ``team`` is None."""
    args = tuple(args)
    if team is not None and "--team-number" not in args and "--day-config" not in args \
            and not (first and "--kt-per-s" in args):
        args = (*args, "--team-number", team)
    if first and "--kt-per-s" in args and "--day-config" not in args:
        out = Path(args[args.index("--output") + 1])
        day = out.parent / f"first-for-{out.stem}.yaml"
        unscored = ["--output", str(day), "--horizon-s", args[args.index("--horizon-s") + 1]]
        if "--base" in args:
            unscored += ["--base", args[args.index("--base") + 1]]
        if "--team-number" in args:
            unscored += ["--team-number", args[args.index("--team-number") + 1]]
        if not day.exists():
            done = _run(project_root, *unscored, team=team)
            assert done.returncode == 0, done.stderr
        args = (*args, "--day-config", str(day))
    if "--script" not in args:
        args = ("--script", str(_script_for(project_root, args)), *args)
    return subprocess.run(
        [sys.executable, str(project_root / "scripts" / "ranking_day_config.py"), *args],
        capture_output=True, text=True, check=False, env=_environment(**(gmat or {})),
    )


def _first(project_root: Path, tmp_path: Path, horizon: str = "6711.667", *extra: str) -> list[str]:
    """--day-config naming a first config for the day, as step 2 writes it."""
    first = tmp_path / f"first-{len(list(tmp_path.glob('first-*')))}.yaml"
    done = _run(project_root, "--output", str(first), "--horizon-s", horizon, *extra)
    assert done.returncode == 0, done.stderr
    return ["--day-config", str(first)]


@pytest.fixture
def r1(project_root: Path) -> Path:
    return project_root / "examples" / "ranking" / "R1_NearPlane_Reachable.script"


def test_the_horizon_alone_is_a_config_solve_can_run(
    project_root: Path, r1: Path, tmp_path: Path
) -> None:
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667")
    assert result.returncode == 0, result.stderr
    config = load_config(out)
    assert config.policy.rules == "ranking" and config.score is None
    assert (config.pursuer_name, config.target_name) == ("SC_Human", "SC_Alien")
    assert config.gmat.workers == 8  # the profile's, unless --gmat-workers says otherwise
    scenario, _ = parse_gmat_script(r1, config)
    assert scenario.max_time_s == 6711.667
    assert scenario.pursuer.name == "SC_Human"
    assert "T_max       : 6711.667 s = 1 h 51 min 51.667 s" in result.stdout


def test_the_rules_and_our_margins_are_printed_apart(project_root: Path, tmp_path: Path) -> None:
    result = _run(project_root, "--output", str(tmp_path / "day.yaml"), "--horizon-s", "6711.667")
    assert result.returncode == 0, result.stderr
    rows = {line.split("  ")[0]: line for line in result.stdout.splitlines()}
    for name, rule, ours in (
        ("per impulse", "250 m/s + 1e-05 m/s (§4 note 1(a))", "249 m/s"),
        ("total delta-v", "3 km/s (§4 note 1(c))", "2.99 km/s"),
        ("burn separation", "100 s, else -10 each (§4 note 1(d))", "101 s"),
        ("burns searched", "none", "search.ranking_max_burns = 13"),
        ("altitude", "surface, before intercept (§4 note 1(b))", "100 km on GMAT's flight"),
        ("search aim", "5 km sphere ", "1 km two-body, 0.5 km on GMAT"),
        ("where a hit ends", "not stated", "1500 m inside the sphere, policy.stop_depth_km"),
        ("GMAT runs at once", "not stated", "gmat.workers = 8; this process may use"),
    ):
        assert rule in rows[name] and ours in rows[name], rows[name]
    # The cores are the topology's, counted here without the script's code.
    assert rows["GMAT runs at once"].endswith(f"this process may use {_may_use(project_root)}")


def test_the_gmat_worker_count_can_be_set_for_the_machine(
    project_root: Path, tmp_path: Path
) -> None:
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667",
                  "--gmat-workers", "1")
    assert result.returncode == 0, result.stderr
    # Only the count changes: correct_top 0 would turn the correction off.
    base = load_config(project_root / "config" / "ranking.yaml").gmat
    assert load_config(out).gmat == dataclasses.replace(base, workers=1)
    assert "gmat.workers = 1; this process may use" in result.stdout
    assert "WARNING: gmat.workers" not in result.stdout


def test_the_gmat_worker_count_fills_a_base_with_an_empty_gmat_block(
    project_root: Path, tmp_path: Path
) -> None:
    import yaml

    raw = yaml.safe_load((project_root / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["gmat"] = None  # every field commented out
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667",
                  "--base", str(base), *_ONE_GMAT)
    assert result.returncode == 0, result.stderr
    assert load_config(out).gmat == dataclasses.replace(GmatConfig(), workers=1)


def test_more_gmat_workers_than_cores_is_warned(project_root: Path, tmp_path: Path) -> None:
    result = _run(project_root, "--output", str(tmp_path / "day.yaml"), "--horizon-s", "6711.667",
                  "--gmat-workers", "100000")
    assert result.returncode == 0, result.stderr
    assert "WARNING: gmat.workers 100000 is more than the" in result.stdout


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="needs CPU affinity")
def test_one_cpu_warns_on_the_profile_and_not_at_one(project_root: Path, tmp_path: Path) -> None:
    cpu = min(os.sched_getaffinity(0))
    # Pinned before exec rather than in preexec_fn, which is unsafe with threads.
    pin = (f"import os, sys; os.sched_setaffinity(0, {{{cpu}}}); "
           "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])")

    def on_one_cpu(name: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", pin, str(project_root / "scripts" / "ranking_day_config.py"),
             "--output", str(tmp_path / name), "--horizon-s", "6711.667",
             "--team-number", str(TEAM_NUMBER),
             "--script", str(_script_for(project_root, ())), *args],
            capture_output=True, text=True, check=False, env=_environment(),
        )

    profile = on_one_cpu("a.yaml")
    assert profile.returncode == 0, profile.stderr
    assert f"gmat.workers = 8; this process may use {_may_use(project_root, [cpu])}\n" \
        in profile.stdout
    assert "WARNING: gmat.workers 8 is more than the 1 " in profile.stdout
    assert "write a new config with --gmat-workers 1 or fewer." in profile.stdout
    fits = on_one_cpu("b.yaml", *_ONE_GMAT)
    assert fits.returncode == 0, fits.stderr
    assert "WARNING" not in fits.stdout


def _day_config_module(project_root: Path) -> types.ModuleType:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ranking_day_config", project_root / "scripts" / "ranking_day_config.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _may_use(project_root: Path, cpus: list[int] | None = None) -> str:
    """What the readback should say, read from sysfs here rather than by the script."""
    cpus = sorted(os.sched_getaffinity(0)) if cpus is None else cpus
    shared: set[str] | None = set()
    for cpu in cpus:
        folder = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        found = [folder / n for n in ("core_cpus_list", "thread_siblings_list")
                 if (folder / n).is_file()]
        if not found or shared is None:
            shared = None
            break
        shared.add(found[0].read_text(encoding="ascii").strip())
    def plural(count: int, noun: str) -> str:
        return f"{count} {noun}{'' if count == 1 else 's'}"

    cores = "cores unknown" if not shared else plural(len(shared), "core")
    return f"{cores}, {plural(len(cpus), 'thread')}"


def _topology(root: Path, sharing: list[str], name: str = "core_cpus_list") -> None:
    """sharing[cpu] lists the CPUs that share cpu's core ("0-1" or "0,1")."""
    for cpu, siblings in enumerate(sharing):
        folder = root / f"cpu{cpu}" / "topology"
        folder.mkdir(parents=True)
        (folder / name).write_text(f"{siblings}\n", encoding="ascii")
        (folder / "core_id").write_text("0\n", encoding="ascii")  # repeats; never read


@pytest.mark.parametrize(
    ("sharing", "cores"),
    [
        (["0-1", "0-1", "2-3", "2-3"], 2),  # two threads per core
        (["0", "1", "2", "3"], 4),  # one thread per core
        # This workstation's i5-14500: 6 cores of two threads, then 8 of one.
        ([f"{2 * (c // 2)}-{2 * (c // 2) + 1}" for c in range(12)]
         + [str(c) for c in range(12, 20)], 14),
    ],
)
def test_cores_are_the_sets_of_cpus_that_share_one(
    project_root: Path, tmp_path: Path, sharing: list[str], cores: int
) -> None:
    _topology(tmp_path, sharing)
    module = _day_config_module(project_root)
    assert module._cores(list(range(len(sharing))), tmp_path) == cores


def test_kernels_before_5_3_are_read_from_thread_siblings(
    project_root: Path, tmp_path: Path
) -> None:
    _topology(tmp_path, ["0-1", "0-1", "2", "3"], name="thread_siblings_list")
    assert _day_config_module(project_root)._cores([0, 1, 2, 3], tmp_path) == 3


def test_an_unreadable_topology_gives_no_core_count(project_root: Path, tmp_path: Path) -> None:
    _topology(tmp_path, ["0"])  # cpu1 has no topology
    module = _day_config_module(project_root)
    assert module._cores([0, 1], tmp_path) is None
    assert module._cores([0], tmp_path) == 1


def test_the_warning_counts_cores_not_threads(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _day_config_module(project_root)
    cpus = module._cpus()
    if len(cpus) < 2:
        pytest.skip("needs two CPUs")
    # Every two of this process's CPUs share a core.
    for index, cpu in enumerate(cpus):
        pair = cpus[index - index % 2: index - index % 2 + 2]
        folder = tmp_path / "sysfs" / f"cpu{cpu}" / "topology"
        folder.mkdir(parents=True)
        (folder / "core_cpus_list").write_text(",".join(map(str, pair)) + "\n", encoding="ascii")
    cores = (len(cpus) + 1) // 2
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667",
                  "--gmat-workers", str(cores + 1))
    assert result.returncode == 0, result.stderr
    monkeypatch.setattr(module, "_CPU_SYSFS", tmp_path / "sysfs")
    lines = "\n".join(module._readback(out))
    assert (f"WARNING: gmat.workers {cores + 1} is more than the "
            f"{module._plural(cores, 'core')} this process may use; write a new config with "
            f"--gmat-workers {cores} or fewer.") in lines


def test_without_a_topology_the_readback_says_so_and_counts_threads(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _day_config_module(project_root)
    threads = len(module._cpus())
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667",
                  "--gmat-workers", str(threads + 1))
    assert result.returncode == 0, result.stderr
    monkeypatch.setattr(module, "_CPU_SYSFS", tmp_path / "no-sysfs")
    lines = module._readback(out)
    row = next(line for line in lines if line.startswith("GMAT runs at once"))
    assert row.endswith(f"this process may use cores unknown, {module._plural(threads, 'thread')}")
    assert (f"WARNING: gmat.workers {threads + 1} is more than the "
            f"{module._plural(threads, 'thread')} this process may use") in "\n".join(lines)


def test_the_coefficients_are_written_in_the_units_the_rules_fix(
    project_root: Path, tmp_path: Path
) -> None:
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667",
                  *SCORE, "--delta-v-unit", "m/s")
    assert result.returncode == 0, result.stderr
    score = load_config(out).score
    assert score is not None
    assert (score.kt, score.ct, score.kv, score.cv, score.cr_km) == (
        0.003, 5369.334, 0.0012, 2000.0, 300.0
    )
    assert (score.time_unit, score.delta_v_unit) == ("s", "m/s")
    # The 2026/11/04 revision gives ΔV_team in m/s.
    refused = tmp_path / "km.yaml"
    result = _run(project_root, "--output", str(refused), "--horizon-s", "6711.667",
                  "--kt-per-s", "0.003", "--ct-s", "5369.334", "--kv", "1.2", "--cv", "2.0",
                  "--cr-km", "300", "--delta-v-unit", "km/s")
    assert result.returncode == 2 and not refused.exists()
    assert "argument --delta-v-unit: invalid choice: 'km/s'" in result.stderr


def test_the_readback_prices_misses_at_the_distances_two_people_check(
    project_root: Path, tmp_path: Path
) -> None:
    """C_r 300 km: 100/(1 + e^{((d - 5)/300)^2}), by hand."""
    result = _run(project_root, "--output", str(tmp_path / "day.yaml"), "--horizon-s", "6711.667",
                  *SCORE, "--delta-v-unit", "m/s", *_ONE_GMAT)
    assert result.returncode == 0, result.stderr
    assert "C_r         : 300.0 km" in result.stdout
    shown = [line for line in result.stdout.splitlines() if line.startswith("  distance")]
    assert shown == [
        "  distance d =     10.000 km   -> 49.9931",
        "  distance d =     65.004 km   -> 49.0000  (D = 49, 5 + 0.2 C_r)",
        "  distance d =    100.000 km   -> 47.4952",
        "  distance d =    319.444 km   -> 25.0000  (D = 25, 5 + 1.048 C_r)",
        "  distance d =   1000.000 km   ->  0.0017",
        "  distance d =  10000.000 km   ->  0.0000",
    ]
    assert "WARNING" not in result.stdout


def test_the_readback_prices_the_zero_burn_when_given(project_root: Path, tmp_path: Path) -> None:
    """R2's zero burn passes 4711.155 km off: 7.86 points at C_r 3000 km."""
    result = _run(project_root, "--output", str(tmp_path / "day.yaml"), "--horizon-s", "5791.381",
                  *SCORE[:-1], "3000", "--delta-v-unit", "m/s", "--zero-burn-dmin-km", "4711.155",
                  *_ONE_GMAT)
    assert result.returncode == 0, result.stderr
    assert ("  distance d =   4711.155 km   ->  7.8647  (the zero burn's closest approach)"
            in result.stdout)
    alone = _run(project_root, "--output", str(tmp_path / "x.yaml"), "--horizon-s", "5791.381",
                 "--zero-burn-dmin-km", "4711.155")
    assert alone.returncode == 2 and "give the score flags too" in alone.stderr
    assert not (tmp_path / "x.yaml").exists()


@pytest.mark.parametrize(
    ("cr_km", "note"),
    [("5", "NOTE: at C_r = 5 km a miss beyond 16 km scores under 1 point on distance."),
     ("100", "NOTE: at C_r = 100 km a miss beyond 219 km scores under 1 point on distance.")],
)
def test_a_small_c_r_is_a_note_against_the_organizers_words(
    project_root: Path, tmp_path: Path, cr_km: str, note: str
) -> None:
    """Plausible, and what a rehearsal uses, so not a WARNING operators learn to skip."""
    result = _run(project_root, "--output", str(tmp_path / "day.yaml"), "--horizon-s", "6711.667",
                  *SCORE[:-1], cr_km, "--delta-v-unit", "m/s", *_ONE_GMAT)
    assert result.returncode == 0, result.stderr
    assert "WARNING" not in result.stdout
    (shown,) = [line for line in result.stdout.splitlines() if line.startswith("NOTE")]
    assert shown.startswith(note)
    assert "平滑遞減" in shown and "不會在數百公里內即降為零分" in shown
    assert "revision email of 2026-11-04" in shown


def test_a_c_r_no_scenario_could_use_warns_without_naming_a_unit(
    project_root: Path, tmp_path: Path
) -> None:
    result = _run(project_root, "--output", str(tmp_path / "day.yaml"), "--horizon-s", "6711.667",
                  *SCORE[:-1], "300000", "--delta-v-unit", "m/s", *_ONE_GMAT)
    assert result.returncode == 0, result.stderr
    warned = [line for line in result.stdout.splitlines() if line.startswith("WARNING")]
    # Past 1e5 km the range check warns too, also without naming a unit.
    assert len(warned) == 2 and warned[0].startswith("WARNING: C_r = 300000.0 km is outside")
    shown = warned[1]
    assert shown.startswith("WARNING: at C_r = 300000 km the distance term is still above 49 "
                            "at 10000 km")
    assert all(" in m" not in line and "unit" not in line for line in warned)


@pytest.mark.parametrize("cr_km", ["232", "300", "3000", "5000", "7036", "49000"])
def test_a_c_r_in_range_raises_no_warning(project_root: Path, tmp_path: Path, cr_km: str) -> None:
    """7036 km is in range; the old 10-to-1000 km check called it a C_r entered in m."""
    result = _run(project_root, "--output", str(tmp_path / "day.yaml"), "--horizon-s", "6711.667",
                  *SCORE[:-1], cr_km, "--delta-v-unit", "m/s", *_ONE_GMAT)
    assert result.returncode == 0, result.stderr
    assert "WARNING" not in result.stdout


def test_c_r_is_only_taken_with_its_unit(project_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667",
                  *SCORE[:-2], "--cr", "300", "--delta-v-unit", "m/s")
    assert result.returncode == 2 and "unrecognized arguments: --cr" in result.stderr
    assert not out.exists()


def test_a_base_the_flags_cannot_load_is_refused_without_a_traceback(
    project_root: Path, tmp_path: Path
) -> None:
    import yaml

    raw = yaml.safe_load((project_root / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["score"] = {"kt": 0.003, "ct": 5000.0, "kv": 0.0012, "cv": 2000.0, "time_unit": "s",
                    "delta_v_unit": "m/s"}
    base = tmp_path / "stale.yaml"
    base.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667",
                  "--base", str(base))
    assert result.returncode == 2 and "Traceback" not in result.stderr
    assert result.stderr.startswith(f"refused, nothing written: --base {base}: ")
    assert "no cr_km" in result.stderr and not out.exists()


def test_the_ceilings_are_the_formula_at_the_ends_of_its_ranges(
    project_root: Path, tmp_path: Path
) -> None:
    # kt 0.01 /s, Ct 500 s, kv 2 per km/s, Cv 1.5 km/s, T_max 1200 s:
    # 50 + 25/(1+e^-5) + 25/(1+e^-3) and 50 + 25/(1+e^7) + 25/(1+e^-3).
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "1200",
                  "--kt-per-s", "0.01", "--ct-s", "500", "--kv", "0.002", "--cv", "1500",
                  "--cr-km", "300", "--delta-v-unit", "m/s")
    assert result.returncode == 0, result.stderr
    assert "no answer scores above 98.6470" in result.stdout
    assert "no miss scores above   73.8371" in result.stdout


@pytest.mark.parametrize("kt, kv", [("0.01", "0.0005"), ("-0.01", "-0.0005")])
def test_the_readback_says_how_steep_each_term_gets(
    project_root: Path, tmp_path: Path, kt: str, kv: str
) -> None:
    # 25 |k| / 4 at the centre: 0.0625 pt/s, and 0.003125 pt per m/s is 3.125 pt per km/s.
    result = _run(project_root, "--output", str(tmp_path / "day.yaml"), "--horizon-s", "1200",
                  "--kt-per-s", kt, "--ct-s", "500", "--kv", kv, "--cv", "1500",
                  "--cr-km", "100", "--delta-v-unit", "m/s")
    assert result.returncode == 0, result.stderr
    assert ("steepest    : 0.0625 pt/s at T = Ct, 3.125 pt per km/s at ΔV = Cv"
            in result.stdout.splitlines())


def test_a_unit_slip_is_shown_as_a_flat_term(project_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667",
                  "--kt-per-s", "0.003", "--ct-s", "5369.334", "--kv", "0.0012",
                  "--cv", "2000000", "--cr-km", "300", "--delta-v-unit", "m/s")
    assert result.returncode == 0, result.stderr
    assert "WARNING: the delta-v term" in result.stdout
    assert "WARNING: the time term" not in result.stdout


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--horizon-s", "6711.667", "--kt-per-s", "0.003"], "missing --ct-s --kv --cv --cr-km"),
        (["--horizon-s", "6711.667", *SCORE], "missing --delta-v-unit"),
        (["--horizon-s", "6711.667", *SCORE[:-2], "--delta-v-unit", "m/s"], "missing --cr-km"),
        (["--horizon-s", "6711.667", *SCORE[:-1], "0", "--delta-v-unit", "m/s"],
         "argument --cr-km: not a positive finite number: '0'"),
        (["--horizon-s", "6711.667", *SCORE[:-1], "nan", "--delta-v-unit", "m/s"],
         "argument --cr-km: not a positive finite number: 'nan'"),
        (["--horizon-s", "6711.667", *SCORE, "--delta-v-unit", "m/s", "--zero-burn-dmin-km",
          "-1"], "argument --zero-burn-dmin-km: not a positive finite number: '-1'"),
        (["--horizon-s", "-5"], "policy.horizon_s must be positive"),
        (["--horizon-s", "6711.667", "--base", "config/quick.yaml"], "not under the ranking"),
        (["--horizon-s", "6711.667", "--gmat-workers", "0"], "gmat.workers must be at least 1"),
    ],
)
def test_a_refusal_writes_nothing(
    project_root: Path, tmp_path: Path, args: list[str], message: str
) -> None:
    out = tmp_path / "day.yaml"
    base = [str(project_root / a) if a.startswith("config/") else a for a in args]
    result = _run(project_root, "--output", str(out), *base)
    assert result.returncode == 2
    assert message in result.stderr
    assert not out.exists()


def test_an_existing_file_is_never_overwritten(project_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "day.yaml"
    out.write_text("kept\n", encoding="utf-8")
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667")
    assert result.returncode == 2 and "exists" in result.stderr
    assert out.read_text(encoding="utf-8") == "kept\n"


def test_the_rules_column_is_the_rulebook_not_the_base(project_root: Path, tmp_path: Path) -> None:
    import yaml

    raw = yaml.safe_load((project_root / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"].update(dv_limit_km_s=0.2, dv_safe_limit_km_s=0.19, min_burn_separation_s=120.0,
                         safe_burn_separation_s=121.0)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(raw), encoding="utf-8")
    result = _run(project_root, "--output", str(tmp_path / "day.yaml"), "--horizon-s", "6711.667",
                  "--base", str(base))
    assert result.returncode == 0, result.stderr
    rows = {line.split("  ")[0]: line for line in result.stdout.splitlines()}
    assert "250 m/s" in rows["per impulse"] and "190 m/s" in rows["per impulse"]
    assert "100 s, else" in rows["burn separation"] and "121 s" in rows["burn separation"]


def test_the_stop_depth_row_reads_the_configured_depth(project_root: Path, tmp_path: Path) -> None:
    import yaml

    raw = yaml.safe_load((project_root / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["stop_depth_km"] = 0.6
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667", "--base", str(base))
    assert result.returncode == 0, result.stderr
    rows = {line.split("  ")[0]: line for line in result.stdout.splitlines()}
    assert " 600 m inside the sphere, policy.stop_depth_km" in rows["where a hit ends"]
    assert "1500 m" not in rows["where a hit ends"]
    assert load_config(out).policy.stop_depth_km == 0.6


@pytest.mark.parametrize(
    ("score", "warning"),
    [
        # Ct with an extra digit: the time term is 25 over the whole window.
        (["--kt-per-s", "0.003", "--ct-s", "53693.34", "--kv", "0.0012", "--cv", "2000",
          "--cr-km", "300", "--delta-v-unit", "m/s"], "the time term moves under 0.5 points"),
        # km/s values entered as m/s: a step at 2 m/s, not a flat line.
        (["--kt-per-s", "0.003", "--ct-s", "5369.334", "--kv", "1.2", "--cv", "2.0",
          "--cr-km", "300", "--delta-v-unit", "m/s"],
         "the delta-v term drops from 22.5 to 2.5 points within"),
    ],
)
def test_the_unit_slips_a_flat_check_misses_are_shown(
    project_root: Path, tmp_path: Path, score: list[str], warning: str
) -> None:
    result = _run(project_root, "--output", str(tmp_path / "day.yaml"), "--horizon-s", "6711.667",
                  *score, *_first(project_root, tmp_path))
    assert result.returncode == 0, result.stderr
    assert warning in result.stdout


def test_the_preliminarys_real_coefficients_are_read_as_they_were(
    project_root: Path, tmp_path: Path
) -> None:
    """kt 0.003982 /s, Ct 6505.65 s, kv 1.1862 per km/s, Cv 9.0643 km/s: delta-v was free."""
    result = _run(project_root, "--output", str(tmp_path / "day.yaml"), "--horizon-s", "5791.381",
                  "--kt-per-s", "0.003982", "--ct-s", "6505.65", "--kv", "0.0011862",
                  "--cv", "9064.3", "--cr-km", "300", "--delta-v-unit", "m/s", *_ONE_GMAT)
    assert result.returncode == 0, result.stderr
    warnings = [line for line in result.stdout.splitlines() if line.startswith("WARNING")]
    assert len(warnings) == 1 and "the delta-v term moves under" in warnings[0], warnings
    # Above the ranking rules' 3 km/s cap: a NOTE, which orbitwin day stops on too.
    assert "NOTE: Cv = 9064.3 m/s is above the 3000 m/s cap" in result.stdout


def test_the_announced_values_raise_no_warning(project_root: Path, tmp_path: Path) -> None:
    result = _run(project_root, "--output", str(tmp_path / "day.yaml"), "--horizon-s", "6711.667",
                  *SCORE, "--delta-v-unit", "m/s", *_ONE_GMAT)
    assert result.returncode == 0, result.stderr
    assert "WARNING" not in result.stdout


def test_the_craft_names_come_from_the_base_or_the_flags(
    project_root: Path, tmp_path: Path
) -> None:
    import yaml

    raw = yaml.safe_load((project_root / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    del raw["pursuer_name"], raw["target_name"]
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = tmp_path / "day.yaml"
    refused = _run(project_root, "--output", str(out), "--horizon-s", "6711.667", "--base", str(base))
    assert refused.returncode == 2 and "--pursuer-name" in refused.stderr
    assert not out.exists()
    named = _run(project_root, "--output", str(out), "--horizon-s", "6711.667", "--base", str(base),
                 "--pursuer-name", "HumanSC", "--target-name", "AlienSC")
    assert named.returncode == 0, named.stderr
    config = load_config(out)
    assert (config.pursuer_name, config.target_name) == ("HumanSC", "AlienSC")


def test_the_clock_reads_negative_and_rounded_times(project_root: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ranking_day_config", project_root / "scripts" / "ranking_day_config.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._clock(-30.0) == "-0 h 00 min 30.000 s"
    assert module._clock(119.9996) == "0 h 02 min 00.000 s"
    assert module._clock(6711.667) == "1 h 51 min 51.667 s"


def test_the_time_origin_is_always_the_earliest(project_root: Path, tmp_path: Path) -> None:
    """The organizer's answer of 2026-11-04: no flag is needed, and no other is taken."""
    import yaml

    plain = tmp_path / "day.yaml"
    done = _run(project_root, "--output", str(plain), "--horizon-s", "8511.667")
    assert done.returncode == 0, done.stderr
    policy = yaml.safe_load(plain.read_text(encoding="utf-8"))["policy"]
    assert policy["mission_time_origin"] == "earliest"
    assert "closest_approach_window" not in policy
    assert "time origin : earliest: T_team, T_max and Ct count from the earlier" in done.stdout
    assert "d_min window: not set; inspect refuses craft that start apart" in done.stdout
    window = tmp_path / "window.yaml"
    done = _run(project_root, "--output", str(window), "--horizon-s", "8511.667",
                "--mission-time-origin", "earliest", "--closest-approach-window", "from_origin")
    assert done.returncode == 0, done.stderr
    assert load_config(window).policy.closest_approach_window == "from_origin"
    assert "d_min window: from origin (not yet answered)" in done.stdout
    for other in ("target", "pursuer"):
        out = tmp_path / f"{other}.yaml"
        refused = _run(project_root, "--output", str(out), "--horizon-s", "8511.667",
                       "--mission-time-origin", other)
        assert refused.returncode == 2
        assert "2026-11-04 18:17" in refused.stderr and "從比較早開始的開始算" in refused.stderr
        assert not out.exists()


def test_the_boundary_radius_goes_in_and_is_read_back(project_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "day.yaml"
    done = _run(project_root, "--output", str(out), "--horizon-s", "6711.667",
                "--mission-boundary-km", "17000")
    assert done.returncode == 0, done.stderr
    assert load_config(out).policy.mission_boundary_km == 17000.0
    assert ("boundary    : 17000.0 km; this tool refuses, and inspect warns, when "
            "SC_Alien is not there at T_max") in done.stdout
    inside = _run(project_root, "--output", str(tmp_path / "x.yaml"), "--horizon-s", "1000",
                  "--mission-boundary-km", "6000")
    assert inside.returncode == 2 and "inside the" in inside.stderr
    assert not (tmp_path / "x.yaml").exists()
    plain = _run(project_root, "--output", str(tmp_path / "y.yaml"), "--horizon-s", "6711.667")
    assert not [line for line in plain.stdout.splitlines() if line.startswith("boundary")]


def test_a_base_with_another_time_origin_is_refused(project_root: Path, tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        (project_root / "config" / "ranking.yaml").read_text(encoding="utf-8").replace(
            "policy:\n", "policy:\n  mission_time_origin: target\n", 1
        ),
        encoding="utf-8",
    )
    out = tmp_path / "day.yaml"
    done = _run(project_root, "--base", str(base), "--output", str(out), "--horizon-s", "100")
    assert done.returncode == 2 and "從比較早開始的開始算" in done.stderr
    assert "Traceback" not in done.stderr and not out.exists()


def test_a_scored_config_names_the_days_first(project_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "day-scored.yaml"
    done = _run(project_root, "--output", str(out), "--horizon-s", "6711.667", *SCORE,
                "--delta-v-unit", "m/s", first=False)
    assert done.returncode == 2 and "needs --day-config" in done.stderr
    assert not out.exists()


def test_a_scored_config_is_the_first_one_plus_the_score(
    project_root: Path, tmp_path: Path
) -> None:
    """RUNBOOK step 5's command after a step 2 that set every same-day flag."""
    step2 = ["--closest-approach-window", "after_alignment", "--mission-boundary-km", "17000",
             "--pursuer-name", "HumanSC", "--target-name", "AlienSC", "--gmat-workers", "3"]
    day = _first(project_root, tmp_path, "6711.667", *step2)
    out = tmp_path / "day-scored.yaml"
    done = _run(project_root, "--output", str(out), "--horizon-s", "6711.667", *SCORE,
                "--delta-v-unit", "m/s", "--zero-burn-dmin-km", "4711.155", *day)
    assert done.returncode == 0, done.stderr
    first, scored = load_config(Path(day[1])), load_config(out)
    assert first.score is None and scored.score is not None
    assert dataclasses.replace(scored, score=None) == first
    assert scored.gmat.workers == 3
    rescored = _run(project_root, "--output", str(tmp_path / "again.yaml"), "--horizon-s",
                    "6711.667", *SCORE, "--delta-v-unit", "m/s", "--day-config", str(out))
    assert rescored.returncode == 2 and "is scored already" in rescored.stderr


@pytest.mark.parametrize(
    ("first", "scored", "said"),
    [
        # A1's attack: step 5 typed 6771.667 where step 2 had 6711.667.
        ([], ["--horizon-s", "6771.667"], "--horizon-s 6771.667 here, 6711.667 in"),
        (["--closest-approach-window", "after_alignment"],
         ["--horizon-s", "6711.667", "--closest-approach-window", "from_origin"],
         "--closest-approach-window 'from_origin' here, 'after_alignment' in"),
        (["--mission-boundary-km", "17000"],
         ["--horizon-s", "6711.667", "--mission-boundary-km", "21000"],
         "--mission-boundary-km 21000.0 here, 17000.0 in"),
        (["--pursuer-name", "HumanSC"], ["--horizon-s", "6711.667", "--pursuer-name", "SC_Human"],
         "--pursuer-name 'SC_Human' here, 'HumanSC' in"),
    ],
)
def test_what_is_typed_again_must_match_the_first_config(
    project_root: Path, tmp_path: Path, first: list[str], scored: list[str], said: str
) -> None:
    out = tmp_path / "day-scored.yaml"
    done = _run(project_root, "--output", str(out), *scored, *SCORE, "--delta-v-unit", "m/s",
                *_first(project_root, tmp_path, "6711.667", *first))
    assert done.returncode == 2, done.stdout
    assert said in done.stderr and "nothing written" in done.stderr
    assert not out.exists()


def test_a_first_config_with_another_time_origin_is_refused(
    project_root: Path, tmp_path: Path
) -> None:
    """A step 2 config written before the organizer's answer must be written again."""
    import yaml

    first = tmp_path / "day.yaml"
    raw = yaml.safe_load((project_root / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"].update(horizon_s=6711.667, mission_time_origin="target",
                         closest_approach_window="after_alignment")
    first.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = tmp_path / "day-scored.yaml"
    done = _run(project_root, "--output", str(out), "--horizon-s", "6711.667", *SCORE,
                "--delta-v-unit", "m/s", "--day-config", str(first))
    assert done.returncode == 2
    assert done.stderr.startswith(f"refused, nothing written: --day-config {first}: ")
    assert "從比較早開始的開始算" in done.stderr
    assert not out.exists()


@pytest.mark.parametrize("content", [b"policy:\n  rules: ranking\n  team: \xff\xfe\n",
                                     b"policy: [unclosed\n"])
def test_an_unreadable_first_config_is_refused_cleanly(
    project_root: Path, tmp_path: Path, content: bytes
) -> None:
    first = tmp_path / "day.yaml"
    first.write_bytes(content)
    out = tmp_path / "day-scored.yaml"
    done = _run(project_root, "--output", str(out), "--horizon-s", "6711.667", *SCORE,
                "--delta-v-unit", "m/s", "--day-config", str(first))
    assert done.returncode == 2 and "Traceback" not in done.stderr
    assert done.stderr.startswith(f"refused, nothing written: --day-config {first}: ")
    assert not out.exists()


def test_the_team_number_comes_through_and_is_read_back(
    project_root: Path, tmp_path: Path
) -> None:
    import yaml

    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667",
                  "--team-number", "54")
    assert result.returncode == 0, result.stderr
    assert load_config(out).policy.team_number == 54
    rows = {line.split("  ")[0]: line for line in result.stdout.splitlines()}
    assert "policy.team_number = 54: each TeamXX placeholder takes it" in rows["report name"]

    # The profile carries no team: without --team-number nothing is written.
    unset = _run(project_root, "--output", str(tmp_path / "unset.yaml"), "--horizon-s",
                 "6711.667", team=None)
    assert unset.returncode == 2, unset.stderr
    assert "sets no policy.team_number; pass --team-number" in unset.stderr
    assert not (tmp_path / "unset.yaml").exists()

    raw = yaml.safe_load((project_root / "config" / "ranking.yaml").read_text(encoding="utf-8"))
    raw["policy"]["team_number"] = 7
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(raw), encoding="utf-8")
    from_base = _run(project_root, "--output", str(tmp_path / "base-day.yaml"), "--horizon-s",
                     "6711.667", "--base", str(base), team=None)
    assert from_base.returncode == 0, from_base.stderr
    assert load_config(tmp_path / "base-day.yaml").policy.team_number == 7
    clash = _run(project_root, "--output", str(tmp_path / "clash.yaml"), "--horizon-s",
                 "6711.667", "--base", str(base), "--team-number", "54")
    assert clash.returncode == 2 and "--team-number 54, but" in clash.stderr
    assert "policy.team_number 7; give one of them" in clash.stderr
    for bad in ("0", "-3", "5.5", "x"):
        refused = _run(project_root, "--output", str(tmp_path / "bad.yaml"), "--horizon-s",
                       "6711.667", "--team-number", bad)
        assert refused.returncode == 2 and "--team-number" in refused.stderr
    assert not (tmp_path / "clash.yaml").exists() and not (tmp_path / "bad.yaml").exists()


def test_the_organizer_s_answer_to_keep_a_name_is_written_and_read_back(
    project_root: Path, tmp_path: Path
) -> None:
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667",
                  "--report-filename-as-written", "ReportFile1.txt")
    assert result.returncode == 0, result.stderr
    assert load_config(out).policy.report_filename_as_written == "ReportFile1.txt"
    rows = {line.split("  ")[0]: line for line in result.stdout.splitlines()}
    assert "KEPT AS WRITTEN: 'ReportFile1.txt'" in rows["report name"]
    plain = tmp_path / "plain.yaml"
    assert _run(project_root, "--output", str(plain), "--horizon-s", "6711.667").returncode == 0
    assert load_config(plain).policy.report_filename_as_written is None


def test_the_readback_shows_the_signed_off_alignment(project_root: Path, tmp_path: Path) -> None:
    module = _day_config_module(project_root)
    out = tmp_path / "day.yaml"
    result = _run(project_root, "--output", str(out), "--horizon-s", "6711.667")
    assert result.returncode == 0, result.stderr
    assert any(line.startswith("alignment   : added as the first command")
               for line in module._readback(out))


@pytest.mark.parametrize(
    ("step2", "step5", "said"),
    [
        # The reviewer's c5: step 5 types another name than step 2 kept.
        (["--report-filename-as-written", "ODC_2026_Report.txt"],
         ["--report-filename-as-written", "ODC_2026_Team54.txt"],
         "--report-filename-as-written 'ODC_2026_Team54.txt' here, 'ODC_2026_Report.txt' in"),
        # d5: step 5 adds one step 2 did not have.
        ([], ["--report-filename-as-written", "ODC_2026_Report.txt"],
         "Write step 2's config again with --report-filename-as-written"),
    ],
)
def test_the_report_name_kept_as_written_is_step_2s(
    project_root: Path, tmp_path: Path, step2: list[str], step5: list[str], said: str
) -> None:
    out = tmp_path / "day-scored.yaml"
    done = _run(project_root, "--output", str(out), "--horizon-s", "6711.667", *SCORE,
                "--delta-v-unit", "m/s", *step5,
                *_first(project_root, tmp_path, "6711.667", *step2))
    assert done.returncode == 2 and said in done.stderr, done.stderr
    assert not out.exists()


def test_the_readback_checks_t_max_against_sc_aliens_coast(
    project_root: Path, r1: Path, tmp_path: Path
) -> None:
    """A mistyped T_max used to show only in inspect, and off the boundary only with it set."""
    good = tmp_path / "good.yaml"
    done = _run(project_root, "--script", str(r1), "--output", str(good), "--horizon-s",
                "6711.667", "--mission-boundary-km", "17000")
    assert done.returncode == 0, done.stderr
    assert f"script      : {r1}" in done.stdout
    assert ("T_max check (two-body, solver clock): SC_Alien passes perigee at t = 4500.000 s"
            in done.stdout)
    assert "at the 17000.0 km boundary" in done.stdout

    # A digit slipped: before SC_Alien's perigee, which §3 rules out.
    slipped = tmp_path / "slipped.yaml"
    done = _run(project_root, "--script", str(r1), "--output", str(slipped), "--horizon-s",
                "671.1667")
    assert done.returncode == 2 and "before SC_Alien's perigee" in done.stderr
    assert not slipped.exists()

    # After the perigee but nowhere near the announced boundary.
    off = tmp_path / "off.yaml"
    done = _run(project_root, "--script", str(r1), "--output", str(off), "--horizon-s", "6000",
                "--mission-boundary-km", "17000")
    assert done.returncode == 2, done.stdout
    assert "from the announced boundary 17000.0 km" in done.stderr
    assert "if this is the announced value, rerun with --tmax-as-announced" in done.stderr
    assert not off.exists()

    # The scored config is checked again, on the same script.
    scored = tmp_path / "scored.yaml"
    done = _run(project_root, "--script", str(r1), "--output", str(scored), "--horizon-s",
                "6711.667", "--mission-boundary-km", "17000", *SCORE, "--delta-v-unit", "m/s",
                "--day-config", str(good))
    assert done.returncode == 0, done.stderr
    assert "T_max check (two-body, solver clock)" in done.stdout


def test_the_readback_needs_the_script(project_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "day.yaml"
    # Step 1's copy by default, from where the day's commands run.
    done = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "ranking_day_config.py"),
         "--output", str(out), "--horizon-s", "6711.667"],
        capture_output=True, text=True, check=False, cwd=tmp_path, env=_environment(),
    )
    assert done.returncode == 2 and not out.exists()
    assert "--script input/original/official.script is not a file" in done.stderr
    done = _run(project_root, "--script", str(tmp_path / "missing.script"), "--output", str(out),
                "--horizon-s", "6711.667")
    assert done.returncode == 2 and "is not a file" in done.stderr and not out.exists()


def test_the_gmat_pair_comes_from_the_flags_or_else_the_environment(
    project_root: Path, r1: Path, tmp_path: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    """Step 5's command once lacked the pair, and an EarthFixed template then exited 2."""
    pair = {"GMAT_CONSOLE": str(fake_gmat), "GMAT_STARTUP": str(fake_startup)}
    base = ("--script", str(r1), "--horizon-s", "6711.667")
    exported = _run(project_root, *base, "--output", str(tmp_path / "a.yaml"), gmat=pair)
    assert exported.returncode == 0, exported.stderr
    assert f"read through GMAT {fake_gmat} (GMAT_CONSOLE)" in exported.stdout
    flagged = _run(project_root, *base, "--output", str(tmp_path / "b.yaml"),
                   "--gmat-console", str(fake_gmat), "--startup-file", str(fake_startup))
    assert flagged.returncode == 0, flagged.stderr
    assert f"read through GMAT {fake_gmat} (--gmat-console)" in flagged.stdout
    # "$GMAT_CONSOLE" unset expands to nothing: read in-process, as with no flags at all.
    empty = _run(project_root, *base, "--output", str(tmp_path / "c.yaml"),
                 "--gmat-console", "", "--startup-file", "")
    assert empty.returncode == 0, empty.stderr
    assert "read in-process: no GMAT pair given or exported" in empty.stdout
    half = _run(project_root, *base, "--output", str(tmp_path / "d.yaml"),
                gmat={"GMAT_CONSOLE": str(fake_gmat)})
    assert half.returncode == 2 and "must be given together" in half.stderr
    assert not (tmp_path / "d.yaml").exists()
    slipped = _run(project_root, "--script", str(r1), "--horizon-s", "671.1667",
                   "--output", str(tmp_path / "e.yaml"), gmat=pair)
    assert slipped.returncode == 2 and "before SC_Alien's perigee" in slipped.stderr
    assert not (tmp_path / "e.yaml").exists()


@pytest.mark.parametrize("name", ["R1", "R2", "R3", "R4", "R5"])
def test_a_t_max_announced_rounded_is_taken_with_the_flag(
    project_root: Path, tmp_path: Path, name: str
) -> None:
    def run(script: str, horizon: str, boundary: int, out: Path, *flags: str):
        return _run(project_root, "--script", str(project_root / "examples" / "ranking" / script),
                    "--output", str(out), "--horizon-s", horizon,
                    "--mission-boundary-km", str(boundary), *flags)

    tmax_as_announced_cases(run, name, tmp_path)


def test_the_flag_comes_through_step_5_and_names_its_t_max(
    project_root: Path, r1: Path, tmp_path: Path
) -> None:
    base = ("--script", str(r1), "--mission-boundary-km", "17000")
    plain, flagged = tmp_path / "plain.yaml", tmp_path / "flagged.yaml"
    assert _run(project_root, *base, "--output", str(plain), "--horizon-s",
                "6711.667").returncode == 0
    assert _run(project_root, *base, "--output", str(flagged), "--horizon-s", "6720",
                "--tmax-as-announced").returncode == 0
    scored = ("--kt-per-s", "0.003", "--ct-s", "5369.334", "--kv", "0.0012", "--cv", "2000",
              "--cr-km", "300", "--delta-v-unit", "m/s")
    # Step 5 need not repeat it: the day config carries it.
    carried = _run(project_root, *base, "--output", str(tmp_path / "s1.yaml"), "--horizon-s",
                   "6720", *scored, "--day-config", str(flagged))
    assert carried.returncode == 0, carried.stderr
    assert load_config(tmp_path / "s1.yaml").policy.tmax_as_announced_s == 6720.0
    # Added only at step 5, it is step 2's to write.
    added = _run(project_root, *base, "--output", str(tmp_path / "s2.yaml"), "--horizon-s",
                 "6711.667", *scored, "--tmax-as-announced", "--day-config", str(plain))
    assert added.returncode == 2
    assert "Write step 2's config again with --tmax-as-announced" in added.stderr


def test_the_flag_names_one_t_max(tmp_path: Path) -> None:
    import dataclasses as dc

    from orbitwin.config import PolicyConfig, validate_config
    from orbitwin.errors import ConfigurationError

    policy = dc.replace(PolicyConfig(), horizon_s=6720.0, tmax_as_announced_s=6720.0)
    validate_config(dc.replace(OrbitWinConfig(), policy=policy))
    with pytest.raises(ConfigurationError, match="takes only the T_max it names"):
        validate_config(dc.replace(OrbitWinConfig(), policy=dc.replace(
            policy, horizon_s=6711.667)))


#: (kt, Ct, kv, Cv, C_r) as announced in the tests above; T_max 6711.667 s, cap 3000 m/s.
_PLAUSIBLE = (0.003, 5369.334, 0.0012, 2000.0, 300.0)


@pytest.mark.parametrize(
    ("change", "warning"),
    [
        ({0: 0.0}, "WARNING: kt = 0.0 is not positive"),
        ({0: -0.003}, "WARNING: kt = -0.003 is not positive"),
        ({2: 0.0}, "WARNING: kv = 0.0 is not positive"),
        ({2: -0.0012}, "WARNING: kv = -0.0012 is not positive"),
        # Ct typed in minutes, 89.5 for 5369.334 s.
        ({1: 67.0}, "WARNING: Ct = 67.0 s is under 1% of T_max = 6711.667 s"),
        # Cv typed in km/s.
        ({3: 2.0}, "WARNING: Cv = 2.0 m/s is under 1% of the 3000 m/s cap"),
        ({3: 3000.5}, "NOTE: Cv = 3000.5 m/s is above the 3000 m/s cap"),
        ({0: 0.003, 1: -2000.0}, "WARNING: the time term is 0.06182 points at zero time "
         "(kt*Ct = -6)"),
        ({3: -4000.0}, "WARNING: the delta-v term is 0.2041 points at zero delta-v "
         "(kv*Cv = -4.8)"),
        ({4: 0.5}, "WARNING: C_r = 0.5 km is outside 1 to 100000 km"),
        ({4: 100001.0}, "WARNING: C_r = 100001.0 km is outside 1 to 100000 km"),
    ],
)
def test_each_implausible_coefficient_warns(
    project_root: Path, change: dict[int, float], warning: str
) -> None:
    module = _day_config_module(project_root)
    values = [change.get(i, value) for i, value in enumerate(_PLAUSIBLE)]
    lines = module._plausibility(*values, 6711.667, 3000.0, "m/s")
    assert any(line.startswith(warning) for line in lines), lines


@pytest.mark.parametrize(
    "values",
    [_PLAUSIBLE, (0.003, 67.2, 0.0012, 30.0, 1.0), (0.003, 5369.334, 0.0012, 3000.0, 1.0e5)],
)
def test_the_plausibility_bounds_themselves_pass(
    project_root: Path, values: tuple[float, ...]
) -> None:
    assert _day_config_module(project_root)._plausibility(*values, 6711.667, 3000.0, "m/s") == []


def test_the_readback_prints_the_plausibility_warnings(project_root: Path, tmp_path: Path) -> None:
    """Review of #195: Ct at 0.27 and 0.5 T_max is plausible on R1, before its perigee or not,
    and does not warn; kv -0.0012 does."""
    script = project_root / "examples" / "ranking" / "R1_NearPlane_Reachable.script"
    first = _run(project_root, "--output", str(tmp_path / "first.yaml"), "--horizon-s",
                 "6711.667", "--mission-boundary-km", "17000", "--script", str(script))
    assert first.returncode == 0, first.stderr
    for name, ct, kv in (("a", "1812.15", "0.0012"), ("b", "3355.8", "0.0012"),
                         ("c", "3355.8", "-0.0012")):
        result = _run(project_root, "--output", str(tmp_path / f"{name}.yaml"), "--horizon-s",
                      "6711.667", "--kt-per-s", "0.003", "--ct-s", ct, "--kv", kv, "--cv",
                      "2000", "--cr-km", "300", "--delta-v-unit", "m/s", "--script",
                      str(script), "--day-config", str(tmp_path / "first.yaml"), *_ONE_GMAT)
        assert result.returncode == 0, result.stderr
        warned = [line for line in result.stdout.splitlines()
                  if line.startswith(("WARNING", "NOTE"))]
        if kv == "0.0012":
            assert warned == [], warned
        else:
            assert len(warned) == 1 and warned[0].startswith("WARNING: kv = -0.0012"), warned


def test_head_start_burns_can_be_turned_off_on_the_day(project_root: Path, tmp_path: Path) -> None:
    on = tmp_path / "on.yaml"
    done = _run(project_root, "--output", str(on), "--horizon-s", "6711.667")
    assert done.returncode == 0, done.stderr
    assert load_config(on).policy.head_start_burns
    assert "head start  : policy.head_start_burns on" in done.stdout
    off = tmp_path / "off.yaml"
    done = _run(project_root, "--output", str(off), "--horizon-s", "6711.667",
                "--no-head-start-burns")
    assert done.returncode == 0, done.stderr
    assert not load_config(off).policy.head_start_burns
    assert "--no-head-start-burns" in done.stdout
    # The scored config keeps it, and refuses to turn it off unseen on a day it was on.
    scored = tmp_path / "scored.yaml"
    done = _run(project_root, "--output", str(scored), "--horizon-s", "6711.667", *SCORE, "--delta-v-unit", "m/s",
                "--day-config", str(off))
    assert done.returncode == 0, done.stderr
    assert not load_config(scored).policy.head_start_burns
    done = _run(project_root, "--output", str(tmp_path / "x.yaml"), "--horizon-s", "6711.667",
                *SCORE, "--delta-v-unit", "m/s", "--delta-v-unit", "m/s", "--day-config", str(on), "--no-head-start-burns")
    assert done.returncode != 0 and "--no-head-start-burns" in done.stderr
