"""GmatQueue: the one bounded pool corrections and verifications share."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from conftest import FAKE_STARTUP

from orbitwin.gmat_loop import GmatQueue


def test_never_more_than_workers_at_once() -> None:
    lock = threading.Lock()
    live = {"now": 0, "max": 0}
    # The first three meet, so the peak reaches the bound however loaded the host.
    meet = threading.Barrier(3, timeout=10)
    started = iter(range(12))

    def job() -> None:
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
            first = next(started) < 3
        if first:
            meet.wait()
        time.sleep(0.01)
        with lock:
            live["now"] -= 1

    queue = GmatQueue(3)
    for _ in range(12):
        queue.add(0, job)
    queue.run()
    assert live["max"] == 3


def test_a_lower_tier_goes_first_and_done_may_add() -> None:
    order: list[str] = []
    main = threading.get_ident()
    queue = GmatQueue(1)

    def done(name: str) -> None:
        assert threading.get_ident() == main
        if name == "correct-a":
            queue.add(0, lambda: order.append("verify-a"))

    for name in ("correct-a", "correct-b"):
        queue.add(1, lambda name=name: order.append(name) or name, done)
    queue.add(2, lambda: order.append("zero"))
    queue.run()
    # The verification added on correct-a's completion overtakes correct-b.
    assert order == ["correct-a", "verify-a", "correct-b", "zero"]


def test_a_job_that_raises_lets_the_running_finish_and_starts_nothing_more() -> None:
    finished: list[str] = []
    started = threading.Event()
    failed = threading.Event()

    def slow() -> None:
        started.set()
        failed.wait(5)  # still running when the other raises, however loaded the host
        time.sleep(0.5)
        finished.append("slow")

    def fails() -> None:
        started.wait(5)
        failed.set()
        raise RuntimeError("boom")

    queue = GmatQueue(2)
    queue.add(0, slow)
    queue.add(0, fails)
    queue.add(1, lambda: finished.append("never"))
    with pytest.raises(RuntimeError, match="boom"):
        queue.run()
    assert finished == ["slow"]


def test_workers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="workers"):
        GmatQueue(0)


def test_waiting_counts_one_tier() -> None:
    queue = GmatQueue(1)
    for tier in (0, 1, 1, 2):
        queue.add(tier, lambda: None)
    assert [queue.waiting(t) for t in (0, 1, 2, 3)] == [1, 2, 1, 0]
    queue.run()
    assert queue.waiting(1) == 0


def test_an_abort_kills_the_running_gmat_and_starts_no_more(tmp_path: Path) -> None:
    """GmatConsole has its own session: without this, a Ctrl-C waited out every
    round of every running correction."""
    from orbitwin.errors import GmatRunError
    from orbitwin.gmat_runner import run_gmat

    pids = tmp_path / "pids"
    console = tmp_path / "bin" / "GmatConsole"
    console.parent.mkdir()
    console.write_text(f'#!/bin/sh\necho $$ >> "{pids}"\nsleep 30\n', encoding="ascii")
    console.chmod(0o755)
    script = tmp_path / "s.script"
    script.write_text("% nothing\n", encoding="ascii")
    startup = FAKE_STARTUP
    refused: list[str] = []

    def two_rounds() -> None:
        for k in range(2):
            try:
                run_gmat(console, script, tmp_path / f"r{k}", ["r.txt"], timeout_s=60,
                         startup_file=startup)
            except GmatRunError as exc:
                refused.append(str(exc))

    def interrupt(_: object) -> None:
        raise KeyboardInterrupt

    def first_run_started() -> None:
        deadline = time.monotonic() + 10
        while not pids.exists() and time.monotonic() < deadline:
            time.sleep(0.02)

    queue = GmatQueue(2)
    queue.add(0, two_rounds)
    queue.add(0, first_run_started, interrupt)
    began = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        queue.run()
    assert time.monotonic() - began < 10
    (pid,) = pids.read_text().split()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid), 0)  # killed and reaped
    assert refused == ["GmatConsole runs are being stopped"]
    # And runs start again afterwards.
    from orbitwin.gmat_runner import _STOPPERS

    assert not _STOPPERS


def test_a_second_interrupt_keeps_runs_refused_until_every_job_returns() -> None:
    from orbitwin import gmat_runner

    seen: list[bool] = []
    release = threading.Event()

    def job() -> None:
        release.wait(5)
        seen.append(bool(gmat_runner._STOPPERS))

    calls = {"n": 0}
    real = gmat_runner.stop_runs

    def stop() -> None:
        real()
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt  # the second Ctrl-C, inside the abort
        release.set()

    def interrupt(_: object) -> None:
        raise KeyboardInterrupt

    import orbitwin.gmat_loop as gmat_loop

    queue = GmatQueue(2)
    queue.add(0, job)
    queue.add(0, lambda: None, interrupt)
    original = gmat_loop.stop_runs
    gmat_loop.stop_runs = stop
    try:
        with pytest.raises(KeyboardInterrupt):
            queue.run()
    finally:
        gmat_loop.stop_runs = original
    assert seen == [True] and calls["n"] == 2
    assert not gmat_runner._STOPPERS
