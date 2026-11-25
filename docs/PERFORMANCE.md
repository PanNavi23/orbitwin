# Where the time goes, and what has been measured not to help

Numbers from `config/history/preliminary/odc2026.yaml` on the 2026-09-05 scenario, an i5-14500
(6 P-core + 8 E-core, 20 threads). Each is a measurement, not an estimate; the
point of writing them down is so the next optimisation starts from evidence
instead of from where the time *feels* like it goes.

## A full `orbitwin solve`

Profiled with `cProfile`, `--workers 1`, sorted by own time:

| | own | cumulative | calls |
|---|---:|---:|---:|
| scipy `rk_step` (DOP853) | 6.89 s | 16.35 s | 262,219 |
| `trajectory._two_body_pair_derivative` | 3.46 s | 6.32 s | 3,146,742 |
| scipy `_minimize_scalar_bounded` | 2.13 s | 4.55 s | 17,892 |
| `numpy.array` | 1.62 s | — | 3,622,775 |
| `np.roots` + `eigvals` | 1.77 s | 4.22 s | 114,427 |
| `lambert.function` (scalar residual) | 0.89 s | 2.20 s | 694,250 |

The integration dominates. Lambert, which reads like the expensive part, is not.

## Parallelism

`search.workers` scales the one-burn sweep. Measured end to end, same scenario,
before and after the local refinement stopped running its pairs on one core:

| workers | before (s) | speedup | after (s) | speedup |
|---:|---:|---:|---:|---:|
| 1 | 23.34 | 1.00x | 23.39 | 1.00x |
| 4 | 9.68 | 2.41x | 8.17 | 2.86x |
| 8 | 8.07 | 2.89x | 6.68 | 3.50x |
| 14 (physical cores) | 7.59 | 3.07x | 5.75 | 4.07x |
| 19 (`cpu_count() - 1`, the default) | 7.47 | 3.12x | 5.75 | **4.07x** |

The old curve saturated at 3.12x and the reason was countable rather than
mysterious. A solve evaluates 2,044 `(burn, arrival)` pairs: the sweep's 1,081
went through a process pool and `_look_around`'s 963 ran in a plain loop on one
core. Both go through `_evaluate_pairs` now.

The knee is at 8, which reaches 86% of the ceiling on 57% of the machine.

**14 and 19 are now the same number**, and that settles an argument the earlier
measurement seemed to lose. `resolve_workers`'s docstring says the useful
ceiling is physical cores and that hyperthreads add contention rather than
throughput; before the change, 19 logical measured 1.6% *faster* than 14
physical, which read as evidence against it. It was the serial fraction
talking. With that gone the five extra logical CPUs buy nothing, so a caller
who wants the rest of the box can set 14 by hand at no cost.

Capping the BLAS thread pool to one per worker changed the total by about 1%.
The arrays here are three and six elements long, so OpenBLAS never engages its
threads and the oversubscription that setting exists to prevent does not
happen.

Process pools are not the cost either: creating one with 19 workers and
pickling the scenario and config to each takes 22 ms, and a solve enters one
three or four times. That is about 1% of a run.

## What was measured *not* to be worth it

### numba on the derivative: 1.14x end to end

A numba-compiled `_two_body_pair_derivative` is bit-identical to the current
one -- genuinely so, unlike the Lambert case above, because it uses only
`sqrt`, multiply, divide and negate, all of which IEEE-754 requires to be
correctly rounded; there is no libm transcendental for a SIMD kernel to round
differently -- and produces the same `nfev`. In isolation it is **1.97x** -- 0.745 us per
call against 0.378. End to end through `solve_ivp` it is **1.14x**.

The reason it is not more: the derivative is already the product of one round
of measurement -- its docstring records the `tolist()` change that took it from
5.28 to 2.71 us -- and what remains is the Python call itself. scipy invokes it
once per stage through three wrapper frames, and numba's own dispatch across
that boundary costs about as much as it saves.

The 368x that numba reaches on this kernel in a microbenchmark is available
only when the **whole integration loop** compiles, which means replacing
scipy's DOP853. That is a change of numerical method, and everything this
project claims rests on results reproducing under GMAT, so it would have to be
re-validated there rather than assumed. It has not been attempted.

So numba is not a dependency this project has a reason to take on. Reconsider
only with a number that beats 1.14x.

### The GPU: wrong shape, not wrong size

An RTX 4060 Ti was measured on a batched fixed-step propagation of the same
dynamics:

| | float64 steps/s | vs this CPU, all 14 cores |
|---|---:|---:|
| numpy per step (as the code does it) | 66,427 | 0.27x |
| numba, one core | 24,435,787 | 0.10x |
| numba, 14 cores | 246,040,940 | 1.00x |
| **GPU float64** | **1,271,937,041** | **5.17x** |
| GPU float32 | 64,434,932,294 | 261.9x |

float32 is not usable for the answer: over one 5,676 s arc it drifts 31 to 105
metres, and the flown solution's closest approach is 4.15 m with the tie-break
in §6 resting on exactly that. Consumer Ada runs float64 at 1/64 of float32 --
measured here as 50.7x -- and NVIDIA's own Blackwell whitepaper says those
units exist for correctness rather than throughput.

The batch has to be large before any of that applies. Measured crossovers, same
kernel: float64 beats one CPU core above a batch of ~52 and the whole CPU above
~850; float32 beats the whole CPU above ~50. Below a batch of 50 the GPU loses
to a single core, because the kernel launch is the whole runtime.

The searches here run batches of 1,081 (`odc2026.yaml`, 23x47) and 4,185
(`odc2026-deep.yaml`, 45x93). At 1,081 the GPU is 1.21x. And the work would
have to be rewritten as fixed-step branch-free kernels first, which is the same
change of numerical method as above, on a code path CI has no hardware to test.

`docs/LESSONS_2026-09-05.md` §6 also records that 66 jobs from 37 starting
points all converged to the same family, and that the 1.68-point gap to the
scoring bound is the 84.31° plane geometry rather than insufficient search. A
faster search would have found the same answer sooner.

## What was measured to help

| change | effect |
|---|---|
| One evaluation per point in the tangent-root guard | residual evaluations -10.2%, wall time -3% |
| Vectorised residual sampling and bracket search | residual evaluations -88%, Lambert 1.40x |
| Local refinement joined the pool the sweep uses | 19 workers 7.47 s to 5.75 s, ceiling 3.12x to 4.07x |

Both leave the answer alone. Every Lambert branch over 770 geometries -- 3,496
of them -- compared equal as raw float64 bits before and after **on this
machine**, which is the strongest form of that check and not a portable one:
numpy selects its float64 transcendental kernels by CPU at run time, so
`np.cos` and `math.cos` are identical here and differ in their last place
elsewhere. CI proved it, passing a bit-identity assertion on one runner and
failing it on another with numpy pinned to the same 2.4.6.

What is portable, and what the tests require, is that the sampled values agree
to a rounding difference and the **roots** they lead to are identical. brentq
refines every bracket with the scalar function, so a last-place difference in a
sample has nowhere to land.

## What is next, on this evidence

The two scipy loops the profile is dominated by: `rk_step`, with 3,146,742
derivative calls under it, and `_minimize_scalar_bounded` in the tangent-root
guard. Neither is reachable with a faster kernel, and the numbers above say a
faster kernel is not on offer anyway. Both are questions about doing less work
rather than doing it faster -- fewer integration steps, or a guard that runs
only where a tangent root is possible.

The serial fraction that capped the old curve at 3.12x is gone. What remains
between 4.07x and 14 is worth measuring again before it is worth attacking.
