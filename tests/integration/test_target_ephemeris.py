from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orbitwin.config import OrbitWinConfig
from orbitwin.ephemeris import target_ephemeris_from_gmat
from orbitwin.errors import GmatRunError
from orbitwin.gmat_parser import parse_gmat_script


def _config(horizon_s: float) -> OrbitWinConfig:
    base = OrbitWinConfig()
    policy = dataclasses.replace(base.policy, horizon_s=horizon_s, final_padding_s=5.0)
    return dataclasses.replace(base, policy=policy, pursuer_name="Pursuer", target_name="AlienSC")


def test_a_coast_run_covering_the_horizon_gives_the_target(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path
) -> None:
    # The fake console reports t = 0, 10 and 20 s with the target at x = 7000 km.
    config = _config(20.0)
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    ephemeris = target_ephemeris_from_gmat(
        scenario, document, config, fake_gmat, fake_startup, tmp_path / "coast"
    )
    assert list(ephemeris.times_s) == [0.0, 10.0, 20.0]
    assert ephemeris.state_at(15.0).r == pytest.approx((7000.0, 0.0, 0.0))


def test_a_coast_run_that_stops_short_of_the_horizon_is_refused(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    """A run that succeeds but whose report ends at 20 s, as a stop condition met early would."""
    import orbitwin.ephemeris as ephemeris_module

    real = ephemeris_module.run_gmat

    def stops_short(*args, **kwargs):
        run = real(*args, **kwargs)
        for path in run.reports.values():
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            kept = [lines[0], *(line for line in lines[1:] if float(line.split()[0]) <= 20.0)]
            Path(path).write_text("\n".join(kept) + "\n", encoding="utf-8")
        return run

    monkeypatch.setattr(ephemeris_module, "run_gmat", stops_short)
    config = _config(100.0)
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    with pytest.raises(GmatRunError, match=r"not over \[0, 100"):
        target_ephemeris_from_gmat(
            scenario, document, config, fake_gmat, fake_startup, tmp_path / "coast"
        )


def test_a_coast_that_intercepts_is_still_flown_to_the_horizon(
    tmp_path: Path, no_maneuver_script: Path, fake_gmat: Path, fake_startup: Path, monkeypatch
) -> None:
    """With mission_end_policy left at "interception" the coast would stop there."""
    import orbitwin.ephemeris as ephemeris_module

    real = ephemeris_module.zero_burn_candidate

    def intercepting(scenario, config):
        candidate = real(scenario, config)
        candidate.successful, candidate.intercept_time_s = True, 5.0
        return candidate

    monkeypatch.setattr(ephemeris_module, "zero_burn_candidate", intercepting)
    config = _config(20.0)
    scenario, document = parse_gmat_script(no_maneuver_script, config)
    target_ephemeris_from_gmat(scenario, document, config, fake_gmat, fake_startup, tmp_path / "c")
    coast = parse_gmat_script(tmp_path / "c" / "coast.script", config)[0]
    assert sum(segment.duration_s for segment in coast.propagate_segments) == pytest.approx(20.0)


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Propagate DefaultProp(SC_Alien, SC_Human) {SC_Alien.ElapsedSecs = 100};",
         "Propagate DefaultProp(SC_Alien) {SC_Alien.ElapsedSecs = 100.0};"),
        ("Propagate Synchronized DefaultProp(SC_Human) AlienProp(SC_Alien) "
         "{SC_Human.ElapsedSecs = 100};",
         "Propagate AlienProp(SC_Alien) {SC_Alien.ElapsedSecs = 100.0};"),
        ("Propagate 'coast (1)' DefaultProp(SC_Human, SC_Alien) {SC_Human.ElapsedSecs = 100};",
         "Propagate DefaultProp(SC_Alien) {SC_Alien.ElapsedSecs = 100.0};"),
    ],
    ids=["joint", "split", "labelled"],
)
def test_the_coast_flies_the_target_alone(line: str, expected: str) -> None:
    from types import SimpleNamespace

    from orbitwin.ephemeris import _target_only_mission

    scenario = SimpleNamespace(
        target=SimpleNamespace(name="SC_Alien"),
        propagate_segments=(SimpleNamespace(duration_s=100.0, implicit=False),),
        alignment=None,
    )
    script = (
        "Create Spacecraft SC_Alien;\r\nBeginMissionSequence;\r\n"
        f"{line}\r\nManeuver Burn1(SC_Human);\r\n"
    ).encode()
    out = _target_only_mission(script, scenario).decode("utf-8").split("\r\n")  # type: ignore[arg-type]
    assert out[0] == "Create Spacecraft SC_Alien;"
    assert out[2] == expected
    assert out[3] == "% Maneuver Burn1(SC_Human);"


def test_a_coast_that_cannot_name_the_targets_propagator_is_refused() -> None:
    from types import SimpleNamespace

    from orbitwin.ephemeris import _target_only_mission
    from orbitwin.errors import ParseError

    scenario = SimpleNamespace(
        target=SimpleNamespace(name="SC_Alien"),
        propagate_segments=(SimpleNamespace(duration_s=100.0, implicit=False),),
        alignment=None,
    )
    script = b"BeginMissionSequence;\nPropagate DefaultProp(SC_Human) {SC_Human.ElapsedSecs = 1};\n"
    with pytest.raises(ParseError, match="which propagator flies SC_Alien"):
        _target_only_mission(script, scenario)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "line, expected",
    [
        # A label that reads as syntax, which the parser accepts and GMAT runs.
        ("Propagate 'coast (1) {x}' DefaultProp(SC_Human, SC_Alien) {SC_Human.ElapsedSecs = 100};",
         "Propagate DefaultProp(SC_Alien) {SC_Alien.ElapsedSecs = 100.0};"),
        # The per-call stop condition of a split line.
        ("Propagate Synchronized DefaultProp(SC_Human, {SC_Human.ElapsedSecs = 100}) "
         "AlienProp(SC_Alien);",
         "Propagate AlienProp(SC_Alien) {SC_Alien.ElapsedSecs = 100.0};"),
    ],
    ids=["label-with-syntax", "goal-inside-call"],
)
def test_the_coast_reads_every_propagate_form_the_parser_does(line: str, expected: str) -> None:
    from types import SimpleNamespace

    from orbitwin.ephemeris import _target_only_mission

    scenario = SimpleNamespace(
        target=SimpleNamespace(name="SC_Alien"),
        propagate_segments=(SimpleNamespace(duration_s=100.0, implicit=False),),
        alignment=None,
    )
    out = _target_only_mission(f"BeginMissionSequence;\n{line}\n".encode(), scenario)  # type: ignore[arg-type]
    assert out.decode().splitlines()[1] == expected


def test_a_target_in_two_calls_is_refused() -> None:
    from types import SimpleNamespace

    from orbitwin.ephemeris import _target_only_mission
    from orbitwin.errors import ParseError

    scenario = SimpleNamespace(
        target=SimpleNamespace(name="SC_Alien"),
        propagate_segments=(SimpleNamespace(duration_s=100.0, implicit=False),),
        alignment=None,
    )
    script = b"BeginMissionSequence;\nPropagate P1(SC_Alien) P2(SC_Alien, SC_Human) {SC_Human.ElapsedSecs = 1};\n"
    with pytest.raises(ParseError, match="which propagator flies SC_Alien"):
        _target_only_mission(script, scenario)  # type: ignore[arg-type]
