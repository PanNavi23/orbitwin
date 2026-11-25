#!/usr/bin/env python3
"""Does a rewrite of the numerical core really leave the bits alone?

A performance layer that claims "results are bit-for-bit unchanged" is making a
claim the test suite cannot see: every assertion in it compares with a
tolerance, so a rewrite can move the last few bits of every derivative
evaluation and stay green. Downstream of an adaptive integrator those bits are
not small -- they change which steps `solve_ivp` accepts.

This dumps the exact bits of the two functions in the hot path for a fixed
deterministic set of inputs. Run it once under each source tree and diff:

    cd <tree-a> && PYTHONPATH=src python scripts/bit_probe.py a.json
    cd <tree-b> && PYTHONPATH=src python scripts/bit_probe.py b.json
    python scripts/bit_probe.py --compare a.json b.json

`--compare` reports the share of differing doubles and the ULP distance, which
is the difference between "identical" and "identical to within a tolerance".
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

import numpy as np

MU_KM3_S2 = 398600.4418
SEED = 20260827


def bits(value: float | None) -> str | None:
    return None if value is None else struct.pack("<d", float(value)).hex()


def unbits(hexed: str | None) -> float | None:
    return None if hexed is None else struct.unpack("<d", bytes.fromhex(hexed))[0]


def ulp_distance(a: float, b: float) -> int:
    """Doubles ordered as sign-magnitude integers; the gap counts representable steps."""
    ia, ib = (struct.unpack("<q", struct.pack("<d", v))[0] for v in (a, b))
    if ia < 0:
        ia = -9223372036854775808 - ia
    if ib < 0:
        ib = -9223372036854775808 - ib
    return abs(ia - ib)


def probe_derivative(count: int) -> list[list[str]]:
    """Circular-ish states from LEO to GEO, both spacecraft, exactly reproducible."""
    from orbitwin.trajectory import _two_body_pair_derivative

    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(count):
        state = np.empty(12)
        for offset in (0, 6):
            radius = rng.uniform(6600.0, 42200.0)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            state[offset : offset + 3] = radius * direction
            tangent = np.cross(direction, rng.normal(size=3))
            tangent /= np.linalg.norm(tangent)
            state[offset + 3 : offset + 6] = np.sqrt(MU_KM3_S2 / radius) * tangent
        derivative = np.asarray(_two_body_pair_derivative(0.0, state, MU_KM3_S2)).ravel()
        rows.append([bits(v) for v in derivative])
    return rows


def probe_events(count: int) -> list[dict]:
    """Paths aimed near the origin, so a useful share actually enter the sphere."""
    from orbitwin.event_detection import evaluate_hermite_samples

    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(count):
        samples = int(rng.integers(3, 40))
        times = np.cumsum(np.concatenate([[0.0], rng.uniform(0.5, 60.0, samples - 1)]))
        start = rng.normal(scale=rng.choice([20.0, 200.0, 3000.0]), size=3)
        end = rng.normal(scale=rng.choice([1.0, 20.0, 500.0]), size=3)
        fraction = np.linspace(0.0, 1.0, samples)[:, None]
        positions = start * (1 - fraction) + end * fraction
        positions = positions + rng.normal(scale=0.5, size=(samples, 3))
        span = max(float(times[-1] - times[0]), 1.0e-9)
        velocities = np.gradient(positions, axis=0) / (span / max(samples - 1, 1))
        result = evaluate_hermite_samples(times, positions, velocities, 5.0)
        # EventResult grew fields as the stack grew: samples_inside_radius and
        # first_entry_segment_dt_s are absent on older layers. A probe that only
        # runs on the newest tree cannot compare two of them, so every field is
        # read defensively and a field missing on one side compares as absent.
        rows.append(
            {
                "first_entry": bits(getattr(result, "first_entry_time_s", None)),
                "min_distance": bits(getattr(result, "minimum_distance_km", None)),
                "min_distance_time": bits(getattr(result, "minimum_distance_time_s", None)),
                "end_distance": bits(getattr(result, "distance_at_end_km", None)),
                "entry_segment_dt": bits(getattr(result, "first_entry_segment_dt_s", None)),
                "inside": getattr(result, "samples_inside_radius", None),
            }
        )
    return rows


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def compare(left: str, right: str) -> int:
    a = _load(left)
    b = _load(right)

    differing = [
        (i, j)
        for i, (x, y) in enumerate(zip(a["derivative"], b["derivative"], strict=True))
        for j, (p, q) in enumerate(zip(x, y, strict=True))
        if p != q
    ]
    total = len(a["derivative"]) * 12
    print(f"_two_body_pair_derivative: {len(differing)} of {total} doubles differ", end="")
    if differing:
        gaps = [
            ulp_distance(unbits(a["derivative"][i][j]), unbits(b["derivative"][i][j]))
            for i, j in differing
        ]
        print(
            f" ({len(differing) / total:.2%}), "
            f"ULP min={min(gaps)} max={max(gaps)} mean={sum(gaps) / len(gaps):.2f}"
        )
    else:
        print(" -- identical")

    keys = ("first_entry", "min_distance", "min_distance_time", "end_distance", "entry_segment_dt", "inside")
    counts = {k: 0 for k in keys}
    for x, y in zip(a["events"], b["events"], strict=True):
        for k in keys:
            if x[k] != y[k]:
                counts[k] += 1
    entered = sum(1 for e in a["events"] if e["first_entry"] is not None)
    print(f"evaluate_hermite_samples: {len(a['events'])} paths, {entered} entered the sphere")
    for k in keys:
        print(f"  {k:18} {counts[k]:>5} differ")
    return 1 if differing or any(counts.values()) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", help="write a probe here")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    parser.add_argument("--derivative-cases", type=int, default=4000)
    parser.add_argument("--event-cases", type=int, default=1500)
    args = parser.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not args.output:
        parser.error("give an output path, or --compare A B")

    payload = {
        "derivative": probe_derivative(args.derivative_cases),
        "events": probe_events(args.event_cases),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
