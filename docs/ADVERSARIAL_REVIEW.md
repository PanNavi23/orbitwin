# Adversarial review of OrbitWin 0.3.0

Date: 2026-08-19  
Build status: **PASS WITH EXTERNAL BLOCKING GATES**

This report records what was actually executed in the artifact build environment. It deliberately separates analytical evidence, test-double orchestration evidence, and real-GMAT scientific validation.

## Executive assessment

The packaged code passed all executable tests available in this environment, the seeded Lambert fuzz campaign, random different-plane Script ingestion/search, Script mutation policy checks, and repeated exact-run orchestration against a deterministic GmatConsole test double.

It is **not yet competition-certified**, because the exact organizer-approved stock GMAT build and validator were not available. A candidate is not allowed to become `GMAT verified` or `certified` on analytical or fake-console evidence alone.

## Final evidence

| Gate | Result |
|---|---:|
| Python compileall | PASS |
| AST parse | PASS, 40 Python files |
| Source hygiene scan | PASS |
| Fast unit/integration tests | PASS, 46 |
| Slow OOD search tests | PASS, 2 |
| Fast-suite statement coverage | 79% |
| Seeded 3-D / multi-revolution Lambert fuzz | PASS, 5,000 cases, 0 failures |
| Worst best-branch endpoint replay error | 0.000333147433 km |
| Random different-plane Script matrix | 20 cases, 0 pipeline exceptions |
| Matrix cases with a legal one-burn candidate under the test budget | 10/20 |
| Final Script mutation policy | PASS |
| Fake-GMAT original baseline | PASS |
| Fake-GMAT exact final reruns | PASS, 3 |
| Fake official Report byte hashes stable | PASS |
| Project-scoped dependency constraints | PASS |
| Real organizer-compatible GMAT replay | **NOT EXECUTED, BLOCKING** |
| Organizer validator parity | **NOT EXECUTED, BLOCKING** |

The `10/20` one-burn result is not a failure rate. The random matrix intentionally includes geometries that may require more search budget, two burns, or may be infeasible under the 1.5 km/s per-burn bound. The asserted property is that all 20 Scripts were parsed and searched without an unexplained pipeline exception.

## Analytical OOD examples

### Different-plane circular scenario

- candidates retained: 11
- analytically successful: 10
- renderable under the fixed template: 10
- selected analytical total ΔV: 0.744078358546 km/s
- selected burn count: 1
- GMAT verified: **false**

### High-plane-change scenario

- candidates retained: 4
- analytically successful: 3
- renderable under the fixed template: 3
- selected analytical total ΔV: 1.427302563964 km/s
- selected burn count: 1
- GMAT verified: **false**

These numbers demonstrate code-path coverage, not official scores or optimality.

## Adversarial findings fixed during release hardening

### F-01: O(N × trajectory length) repeated propagation

**Severity:** High for competition-time performance.

The original event verifier independently replayed both spacecraft from the initial epoch for every sample. Repeated candidate evaluation became excessively expensive on long `4*T_A` windows.

**Fix:** Added piecewise sequential DOP853 propagation of the paired 12-state system. Impulsive epochs are represented by duplicate samples with identical positions and one-sided velocities. Cubic-Hermite root enumeration remains responsible for hidden crossings and minimum-distance refinement.

**Regression evidence:** Sequential and independent universal-variable trajectories agree within the test tolerances; fast and slow suites pass.

### F-02: An unrenderable candidate could abort the entire solve

**Severity:** High.

A mathematically valid late one-burn candidate may not fit an organizer template containing additional immutable burn slots plus required padding. The pipeline previously attempted to render the top mathematical candidate directly and could terminate before preserving a legal fallback.

**Fix:** Renderability is now evaluated before submission Pareto pruning. Unrenderable candidates are retained in a bounded diagnostic archive, but only renderable candidates may be rendered, GMAT-verified, or selected as `best_candidate`.

**Regression evidence:** Added an integration test in which an unrenderable low-ΔV candidate coexists with a legal two-burn fallback. The legal candidate is selected and the pipeline completes.

### F-03: Report path isolation could be defeated by GMAT startup semantics

**Severity:** Critical for stale/cross-candidate result contamination.

GMAT ReportFile paths are resolved through startup-file output settings, not solely the calling process working directory.

**Fix:** Every run derives a private startup file that changes only `OUTPUT_PATH` and `LOG_FILE` to absolute per-run locations. The organizer startup file remains unchanged and is hashed.

### F-04: Hidden between-sample 5 km crossing

**Severity:** Critical for false mission failure or wrong first-entry time.

Checking Report rows alone can miss a high-speed pass whose endpoints are both outside the 5 km sphere.

**Fix:** For each coast interval, the verifier builds a cubic-Hermite relative-position polynomial, enumerates stationary points and all real unit-interval roots of `||r_rel||² - R²`, and handles duplicate impulsive epochs without interpolating through the velocity discontinuity.

### F-05: Exact antipodal Lambert singularity

**Severity:** Medium/High for OOD geometry.

Universal-variable Lambert reconstruction loses a unique transfer plane for exact collinear endpoints.

**Fix:** Added a deterministic Cartesian shooting fallback using an explicit/reference normal, multiple bounded velocity seeds, and independent endpoint replay.

### F-06: Untrusted Script expansion

**Severity:** Critical for rule compliance.

Include macros, duplicate assignments, target maneuvers, unknown timing semantics, or resource changes could hide forbidden mutations.

**Fix:** Parser and renderer fail closed. Final policy compares byte-level changed lines plus semantic signatures for spacecraft, force model, propagator, burn contract, and official ReportFile contract.

## Remaining risks and blocking gates

### B-01: Exact organizer GMAT build unavailable

Public GMAT R2026a is useful for development, but it cannot be assumed to match the competition build. Before use, run the exact organizer binary, startup file, OS, plugins, gravity/ephemeris data and Report contract.

### B-02: No analytical-versus-real-GMAT differential campaign

Required before competition release:

1. Workshop baseline;
2. different-plane synthetic scene;
3. high-plane-change scene;
4. at least 100 seeded random candidates;
5. burn-frame component comparison;
6. first-entry/minimum-distance comparison;
7. three exact final reruns.

Any systematic discrepancy blocks submission.

### B-03: Organizer permission gaps

Unconfirmed items include:

- exact editable Script fields;
- whether unused existing burn slots may remain as zero-vector Maneuvers;
- whether private diagnostic siblings may be used on a personal laptop;
- score coefficient units and rounding;
- validator floating-point tolerance;
- whether all Script burns or only pre-interception burns count;
- AI/cloud/remote-assistance permissions.

### B-04: Static tools unavailable in the build image

`ruff` and `mypy` could not be installed because the artifact environment had no package-index network access. They are recorded as **SKIPPED**, not PASS. Syntax, AST, source-hygiene and runtime tests did pass. Run `make lint` and `make typecheck` in the team's connected development virtual environment before operational freeze.

### B-05: Global environment dependency conflict

Global `pip check` reports an unrelated pre-existing `moviepy`/`pillow` conflict in the hosted environment. The project-scoped required packages and versions pass. Use a clean virtual environment on the competition laptop.

### B-06: No global-optimality proof

The search is broad on purpose (multiple Lambert branches, multiple revolutions, deterministic grids, Sobol coverage, two-burn fallback and Pareto retention), but it is still a finite numerical search. Do not describe the result as mathematically proven global optimum.

## Release decision

The source package is suitable for continued team development, offline rehearsal, parser/policy testing, analytical candidate generation, and immediate organizer-GMAT integration.

The following statement is the strongest supported claim:

> OrbitWin 0.3.0 passed its packaged analytical, policy, OOD, orchestration and reproducibility test campaign. Competition certification remains blocked on exact organizer-compatible GMAT and validator replay.
