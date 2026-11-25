"""`orbitwin rank` driven end to end through the command line.

初賽規則 §5's four coefficients are announced on competition day, and `rank` is
what turns a saved archive into a submission decision once they arrive. Its unit
tests call `rank_report` with objects already in memory; nothing until now
exercised the argument parsing, the config load, the candidates round trip
through `orbitwin.io`, the exit code or the two output channels. The command is
run by hand, under time pressure, on someone else's machine, so those are the
parts most worth pinning.

`main` is called with an argv list rather than spawned as a subprocess: the
point is to observe the exit code and both streams, not to prove that argv
reaches Python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import without_rulebook_note

from orbitwin.cli import main
from orbitwin.io import write_candidates
from orbitwin.models import Burn, Candidate

# (intercept time in s, total delta-v in km/s). FAST and CHEAP are measured: the
# two points of a real DifferentPlane_Circular frontier that ever won across a
# 625-combination coefficient sweep, and the same pair tests/unit/test_rank.py
# uses. MID is constructed, not measured -- it sits between them and wins under
# neither announcement here, so a ranking has three distinguishable rows and a
# --top of 1 is visibly a truncation.
FAST = (13208.2, 0.7853)
CHEAP = (26030.7, 0.5439)
MID = (19500.0, 0.6400)

# An announcement whose Ct sits just under the fast intercept: time dominates,
# `fast` wins, and a 10% nudge of any coefficient leaves it winning.
DECIDES_ON_TIME = """\
score:
  kt: 0.0005
  ct: 13000.0
  kv: 0.004
  cv: 800.0
  time_unit: s
  delta_v_unit: m/s
"""

# The same archive under a Ct past every intercept and a tighter Cv: all three
# arrive early enough that the time term barely separates them, so delta-v
# decides and `cheap` wins instead.
DECIDES_ON_DELTA_V = """\
score:
  kt: 0.001
  ct: 30000.0
  kv: 0.004
  cv: 600.0
  time_unit: s
  delta_v_unit: m/s
"""

# Cv announced as 0.8 km/s and entered against m/s. Every candidate lands on the
# same saturated tail, the delta-v term returns one constant, and nothing about
# the run fails.
CV_IN_THE_WRONG_UNIT = """\
score:
  kt: 0.001
  ct: 20000.0
  kv: 4.0
  cv: 0.8
  time_unit: s
  delta_v_unit: m/s
"""

# A perfectly valid profile that simply has not had the announcement entered.
NO_SCORE_BLOCK = """\
search:
  one_burn_samples: 0
"""

STAGE_WARNING = (
    "The top-scoring candidate stands at stage analytical. "
    "A score is arithmetic; only certify establishes that GMAT reproduces it."
)


def _workshop_sha256() -> str:
    """The stamp solve writes, computed the same way the parser does."""
    from hashlib import sha256

    root = Path(__file__).resolve().parents[2]
    return sha256((root / "examples" / "two_burn_rendezvous.script").read_bytes()).hexdigest()


def _candidate(intercept_s: float, dv_km_s: float, branch: str) -> Candidate:
    burn = Burn(
        time_s=100.0,
        dv_inertial_km_s=(dv_km_s, 0.0, 0.0),
        dv_script_km_s=(dv_km_s, 0.0, 0.0),
    )
    return Candidate(
        burns=(burn,),
        intercept_time_s=intercept_s,
        minimum_distance_km=1.0,
        # 初賽規則 §4: a successful interception is at 5 km by definition, so
        # the distance term is exactly 50 and the logistics decide the round.
        distance_at_completion_km=5.0,
        successful=True,
        analytical_endpoint_error_km=None,
        branch=branch,
        # What a real archived candidate carries. `event_refined` is load
        # bearing: without it the official minimum-distance tie-break reads the
        # Lambert endpoint residual instead of a measured closest approach. The
        # radius is here so the fixture is a candidate the pipeline could have
        # produced rather than the minimum that happens to rank today.
        metadata={
            "event_refined": True,
            "minimum_pursuer_radius_km": 6900.0,
            # solve stamps this so rank can refuse an archive built from a
            # different script -- the horizon is every failure's completion
            # time, and ranking across two scenarios was measured 25.000 points
            # out. An unstamped archive earns its own warning, which
            # tests/unit/test_rank.py covers; these fixtures are about the
            # ranking, so they look like what solve writes.
            "scenario_sha256": _workshop_sha256(),
        },
    )


@pytest.fixture
def archive() -> dict[str, Candidate]:
    return {
        "fast": _candidate(*FAST, "fast"),
        "cheap": _candidate(*CHEAP, "cheap"),
        "mid": _candidate(*MID, "mid"),
    }


@pytest.fixture
def archive_path(tmp_path: Path, archive: dict[str, Candidate]) -> Path:
    """The archive as `solve` leaves it on disk.

    Written with the project's own writer so that a change to the on-disk shape
    breaks these tests rather than leaving them agreeing with a format nothing
    produces any more.
    """
    path = tmp_path / "solve" / "candidates.json"
    write_candidates(path, list(archive.values()))
    return path


def _config(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _searched_under(archive_path: Path, config: Path) -> None:
    """Stamp the archive as a solve under `config` would: searched under its score block."""
    from orbitwin.config import load_config
    from orbitwin.io import read_candidates
    from orbitwin.score import score_stamp

    candidates = read_candidates(archive_path)
    for candidate in candidates:
        candidate.metadata["searched_score"] = score_stamp(load_config(config).score)
    write_candidates(archive_path, candidates)


def _argv(script: Path, candidates: Path, config: Path | None, *extra: str) -> list[str]:
    """A config not under the ranking rules is named the preliminary's, as rank requires."""
    argv = ["rank", str(script), "--candidates", str(candidates)]
    if config is None or "rules: ranking" not in config.read_text(encoding="utf-8"):
        argv += ["--rules", "preliminary"]
    if config is not None:
        argv += ["--config", str(config)]
    return argv + list(extra)


def test_rank_prints_the_ranking_the_announced_coefficients_produce(
    tmp_path: Path,
    workshop_script: Path,
    archive: dict[str, Candidate],
    archive_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole command, from argv to a report on stdout."""
    config = _config(tmp_path, "announced.yaml", DECIDES_ON_TIME)
    _searched_under(archive_path, config)
    assert main(_argv(workshop_script, archive_path, config)) == 0

    captured = capsys.readouterr()
    assert without_rulebook_note(captured.err) == ""
    report = json.loads(captured.out)

    # The announcement is echoed with its units, so the report says what it was
    # scored under rather than leaving that to whoever kept the config.
    assert report["coefficients"] == {
        "kt": 0.0005,
        "ct": 13000.0,
        "kv": 0.004,
        "cv": 800.0,
        "time_unit": "s",
        "delta_v_unit": "m/s",
    }
    assert report["candidate_count"] == 3
    assert report["successful_count"] == 3
    assert [row["candidate_id"] for row in report["ranking"]] == [
        archive["fast"].candidate_id,
        archive["cheap"].candidate_id,
        archive["mid"].candidate_id,
    ]
    scores = [row["score"] for row in report["ranking"]]
    assert scores == sorted(scores, reverse=True)

    winner = report["winner"]
    assert winner["candidate_id"] == archive["fast"].candidate_id
    assert winner["intercept_time_s"] == pytest.approx(FAST[0])
    assert winner["total_dv_km_s"] == pytest.approx(FAST[1])
    assert winner["penalty_count"] == 0
    components = winner["score_components"]
    assert components["distance_score"] == pytest.approx(50.0)
    assert components["time_score"] == pytest.approx(11.849962, abs=1e-6)
    assert components["delta_v_score"] == pytest.approx(12.867394, abs=1e-6)
    assert components["penalty_score"] == pytest.approx(0.0)
    assert winner["score"] == pytest.approx(74.717356, abs=1e-6)
    assert components["raw_score"] == pytest.approx(winner["score"])

    # Both logistics separate the field, so neither flat-term warning fires and
    # the winner survives a 10% nudge of all four coefficients. The one warning
    # left is the one that always applies to a file-borne archive: read_candidates
    # caps a claimed stage at renderable, so nothing here carries GMAT authority.
    assert report["term_spread_points"]["time"] > 0.5
    assert report["term_spread_points"]["delta_v"] > 0.5
    assert report["term_spread_scope"] == "successful"
    assert report["winner_stability"]["stable"] is True
    assert report["winner_stability"]["changes"] == []
    assert report["winner_stability"]["not_tested"] == []
    assert winner["stage"] == "analytical"
    assert report["warnings"] == [STAGE_WARNING]


def test_output_writes_the_printed_report_into_a_directory_that_does_not_exist(
    tmp_path: Path,
    workshop_script: Path,
    archive_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Competition day runs `rank` again per announcement, into a fresh directory.

    The file has to be the report that was printed, byte for byte, or the
    evidence kept afterwards is not what the decision was made on.
    """
    destination = tmp_path / "day-of" / "attempt-3" / "rank.json"
    assert not destination.parent.exists()

    config = _config(tmp_path, "announced.yaml", DECIDES_ON_TIME)
    argv = _argv(workshop_script, archive_path, config, "--output", str(destination))
    assert main(argv) == 0

    printed = capsys.readouterr().out
    assert destination.is_file()
    written = destination.read_text(encoding="utf-8")
    assert printed == written + "\n"
    assert json.loads(written)["winner"]["score"] == pytest.approx(74.717356, abs=1e-6)


def test_ranking_without_the_announced_coefficients_is_refused(
    tmp_path: Path,
    workshop_script: Path,
    archive_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The refusal that matters most.

    COMPETITION_POLICY.md forbids guessing kt, Ct, kv and Cv. A profile that
    simply has not had them entered must stop the command, not fall back to a
    default that would rank the archive on invented numbers and name a
    submission.
    """
    config = _config(tmp_path, "no-score.yaml", NO_SCORE_BLOCK)
    assert main(_argv(workshop_script, archive_path, config)) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert without_rulebook_note(captured.err).startswith("orbitwin: ERROR [")
    assert "score.kt, score.ct, score.kv and score.cv with their units" in captured.err
    assert "announced by the organizer and must not be guessed" in captured.err

    # The built-in default profile carries no score block either, so omitting
    # --config is the same refusal rather than a silent set of numbers.
    assert main(_argv(workshop_script, archive_path, None)) == 2
    fallback = capsys.readouterr()
    assert fallback.out == ""
    assert "must not be guessed" in fallback.err


def test_top_limits_the_listing_without_dropping_the_winner(
    tmp_path: Path,
    workshop_script: Path,
    archive: dict[str, Candidate],
    archive_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--top shortens what is printed and nothing else.

    The counts, the term spreads and the stability sweep still have to speak
    for the whole archive: a truncated listing that also truncated the analysis
    would understate how close the decision was.
    """
    config = _config(tmp_path, "announced.yaml", DECIDES_ON_TIME)
    assert main(_argv(workshop_script, archive_path, config)) == 0
    full = json.loads(capsys.readouterr().out)

    assert main(_argv(workshop_script, archive_path, config, "--top", "1")) == 0
    limited = json.loads(capsys.readouterr().out)

    assert len(full["ranking"]) == 3
    assert len(limited["ranking"]) == 1
    assert limited["ranking"][0] == full["ranking"][0]
    assert limited["winner"] == full["winner"]
    assert limited["winner"]["candidate_id"] == archive["fast"].candidate_id
    assert limited["candidate_count"] == 3
    assert limited["successful_count"] == 3
    assert limited["term_spread_points"] == full["term_spread_points"]
    assert limited["winner_stability"] == full["winner_stability"]


def test_a_top_below_one_is_refused_rather_than_listing_the_wrong_candidates(
    tmp_path: Path,
    workshop_script: Path,
    archive_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`ranked[:top]` accepts any int and says nothing about a useless one.

    `--top 0` printed an empty ranking beside a populated winner. `--top -1`
    was worse: it printed every candidate except the lowest-scoring one, in
    order, under the name of a top-N listing -- so an operator comparing the
    frontier would have seen a shorter frontier and no indication of it. Same
    shape as the zero tolerance below, and refused the same way.
    """
    config = _config(tmp_path, "announced.yaml", DECIDES_ON_TIME)
    for value in ("0", "-1"):
        assert main(_argv(workshop_script, archive_path, config, "--top", value)) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert without_rulebook_note(captured.err).startswith("orbitwin: ERROR [")
        assert "top" in captured.err

    assert main(_argv(workshop_script, archive_path, config, "--top", "1")) == 0


def test_a_delta_v_centre_in_the_wrong_unit_reaches_the_operator(
    tmp_path: Path,
    workshop_script: Path,
    archive: dict[str, Candidate],
    archive_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A wrong unit does not raise anywhere. The flat term is the only symptom.

    Cv announced as 0.8 km/s and entered against m/s puts 785.3, 640.0 and
    543.9 m/s all past the far tail of the logistic, so the delta-v term
    returns the same constant for every candidate and the ranking is silently
    made by time alone. That is worth up to 25 points, the run exits 0, and the
    winner is stable under a 10% nudge of all four coefficients -- so the
    stability sweep does not catch it either.
    """
    config = _config(tmp_path, "wrong-unit.yaml", CV_IN_THE_WRONG_UNIT)
    # The slip went into the search's config too, so nothing else warns.
    _searched_under(archive_path, config)
    assert main(_argv(workshop_script, archive_path, config)) == 0

    captured = capsys.readouterr()
    assert without_rulebook_note(captured.err) == ""
    report = json.loads(captured.out)

    assert report["term_spread_points"]["delta_v"] == pytest.approx(0.0)
    assert [row["score_components"]["delta_v_score"] for row in report["ranking"]] == [
        0.0,
        0.0,
        0.0,
    ]
    assert report["warnings"] == [
        "The delta_v term spans 0.000 points across the whole archive, so it cannot "
        "affect the ranking. Either the announced centre lies far outside the range "
        "this archive covers, or the unit is wrong: check kv=4.0, Cv=0.8 and "
        "score.delta_v_unit='m/s' against the announcement.",
        STAGE_WARNING,
    ]
    assert report["winner_stability"]["stable"] is True
    assert report["winner"]["candidate_id"] == archive["fast"].candidate_id


def test_a_tolerance_that_nudges_nothing_is_refused(
    tmp_path: Path,
    workshop_script: Path,
    archive_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A tolerance that cannot move a coefficient must be refused, not obeyed.

    A 0% nudge moves nothing, so every trial config equals the announced one,
    every trial winner matches, and the report says the winner is stable --
    which reads as "checked and fine" about a check that never ran. 0 and a
    negative value are two members of that class rather than the whole of it,
    and the class is still being enumerated, so this asserts the shape of the
    refusal rather than its current wording: exit 2, an error on stderr, and no
    report on stdout for anyone to act on.
    """
    config = _config(tmp_path, "announced.yaml", DECIDES_ON_TIME)
    for value in ("0", "-0.1"):
        assert main(_argv(workshop_script, archive_path, config, "--tolerance", value)) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert without_rulebook_note(captured.err).startswith("orbitwin: ERROR [")
        assert "tolerance" in captured.err

    # A tolerance that does move all four coefficients is reported as used.
    assert main(_argv(workshop_script, archive_path, config, "--tolerance", "0.25")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["winner_stability"]["tolerance"] == pytest.approx(0.25)
    assert report["winner_stability"]["not_tested"] == []


def test_the_same_archive_names_a_different_submission_under_a_different_announcement(
    tmp_path: Path,
    workshop_script: Path,
    archive: dict[str, Candidate],
    archive_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Why the subcommand exists at all.

    One search, one candidates.json, two announcements, two different scripts
    to submit. Committing to one of them before the coefficients arrive gives
    up the difference; measured on the real 120-candidate archive that reached
    25 points, the whole weight of a logistic term.
    """
    on_time = _config(tmp_path, "on-time.yaml", DECIDES_ON_TIME)
    on_delta_v = _config(tmp_path, "on-delta-v.yaml", DECIDES_ON_DELTA_V)

    assert main(_argv(workshop_script, archive_path, on_time)) == 0
    time_report = json.loads(capsys.readouterr().out)
    assert main(_argv(workshop_script, archive_path, on_delta_v)) == 0
    delta_v_report = json.loads(capsys.readouterr().out)

    assert time_report["winner"]["candidate_id"] == archive["fast"].candidate_id
    assert delta_v_report["winner"]["candidate_id"] == archive["cheap"].candidate_id

    def _score(report: dict[str, Any], candidate_id: str) -> float:
        rows: list[dict[str, Any]] = report["ranking"]
        return next(float(row["score"]) for row in rows if row["candidate_id"] == candidate_id)

    fast_id = archive["fast"].candidate_id
    cheap_id = archive["cheap"].candidate_id
    # Each announcement's loser gives up real points, not a rounding difference.
    forgone_by_committing_to_cheap = _score(time_report, fast_id) - _score(time_report, cheap_id)
    forgone_by_committing_to_fast = (
        _score(delta_v_report, cheap_id) - _score(delta_v_report, fast_id)
    )
    assert forgone_by_committing_to_cheap > 5.0
    assert forgone_by_committing_to_fast > 5.0


RANKING_RULES = """\
policy:
  rules: ranking
  horizon_s: 30000.0
  dv_limit_km_s: 0.25
  dv_safe_limit_km_s: 0.249
  total_dv_limit_km_s: 3.0
  total_dv_safe_limit_km_s: 2.99
  team_number: 54
  mission_time_origin: earliest
""" + DECIDES_ON_TIME + "  cr_km: 100.0\n"


def test_rank_says_on_stderr_that_an_archive_solved_without_gmat_has_nothing_to_submit(
    tmp_path: Path,
    workshop_script: Path,
    archive_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from orbitwin.io import read_candidates
    from orbitwin.rank import NO_GMAT_ARCHIVE, NO_GMAT_HUNT

    config = _config(tmp_path, "ranking.yaml", RANKING_RULES)
    assert main(_argv(workshop_script, archive_path, config)) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["submittable_count"] == 0
    assert report["warnings"][0] == NO_GMAT_HUNT
    assert f"orbitwin rank: {NO_GMAT_HUNT}\n" in captured.err
    # As solve writes them.
    solved = read_candidates(archive_path)
    for candidate in solved:
        candidate.metadata["renderable"] = True
    write_candidates(archive_path, solved)
    assert main(_argv(workshop_script, archive_path, config)) == 0
    assert f"orbitwin rank: {NO_GMAT_ARCHIVE}\n" in capsys.readouterr().err


def test_rank_warns_on_stderr_when_the_announcement_is_not_what_was_searched(
    tmp_path: Path,
    workshop_script: Path,
    archive_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R2: an archive searched under one block and ranked under another chose the
    zero burn at 31.54 where a search under the new block found 45.82."""
    searched = _config(tmp_path, "searched.yaml", DECIDES_ON_DELTA_V)
    _searched_under(archive_path, searched)
    announced = _config(tmp_path, "announced.yaml", DECIDES_ON_TIME)
    assert main(_argv(workshop_script, archive_path, announced)) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    warning = report["searched_score"]["warning"]
    assert warning in report["warnings"]
    assert warning.startswith(
        "3 of 3 candidates were searched under kt 0.001, Ct 30000 s, Cv 600 m/s, ranked under "
        "kt 0.0005, Ct 13000 s, Cv 800 m/s: the search optimised for the old coefficients"
    )
    assert f"orbitwin rank: WARNING: {warning}\n" in captured.err
    # Ranked anyway: the ranking is still worth reading.
    assert report["winner"]["candidate_id"]
