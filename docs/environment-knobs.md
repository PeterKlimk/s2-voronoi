# Environment-variable ownership

**Status:** authoritative inventory; QUAL-001H completed 2026-07-18

**Audited:** 2026-07-19

This document owns the environment-variable inventory read by compiled library, tool, campaign,
and test code. Shell-only orchestration controls are documented in each script's header or
`--help`; general build/toolchain inputs such as `RUSTFLAGS` and allocator controls such as
`MALLOC_ARENA_MAX` remain external contracts rather than crate runtime knobs.

Environment variables are process-global: library code must read them only at documented cold
seams, and tests that mutate them must restore the exact prior value even during unwinding.

Each Rust integration-test target runs in its own process. Isolation is therefore required between
parallel tests within a target, not between different files in `tests/`. Library unit tests share
one process with every other unit test and should prefer explicit internal inputs over environment
mutation.

## Library and tool runtime knobs

| Variable | Class | Owner / read boundary | Contract |
|---|---|---|---|
| `RAYON_NUM_THREADS` | supported operational | Rayon, before pool initialization | External Rayon contract; used to pin concurrency. |
| `VORONOI_MESH_BIN_COUNT` | supported tuning | `live_dedup::binning::target_bin_count`, once per computation | Integer shard target, clamped to `[6, 96]`; tests and benchmark invocations may override it. |
| `VORONOI_MESH_GRID_DENSITY` | benchmark tuning | `policy::knn_grid_target_density`, first use via `OnceLock` | Parsed `f64` at least 1; intended for grid-density sweeps, not per-computation mutation. |
| `VORONOI_MESH_VERIFY` | supported verification | `validation::verify_enabled`, ordinary compute return gate | Exact value `1` enables the fast verifier with an O(E) strict-validation fallback. |
| `VORONOI_MESH_TIMING_KV` | instrumentation | `timing::real::PhaseTimings::report` | Presence emits machine-readable timing output when the `timing` feature is enabled. |
| `VORONOI_MESH_RECONCILE_TELEMETRY` | correctness diagnostic | defect-scoped `ReconcileOptions` snapshot | Presence repeats a read-only primary-round analysis and emits `RECONCILE_KV`. Read once with the other reconciliation options only after mismatch records exist. |
| `VORONOI_MESH_RESOLUTION_KV` | correctness diagnostic | terminal output-resolution pass | Presence emits exact-zero resolution statistics. |
| `VORONOI_MESH_EDGE_MISMATCH_ORIGINS` | correctness diagnostic | live assembly, mismatch-record path only | Presence prints mismatch-origin counts. Clean assemblies return before the lookup; this replaces the stale internal name `VORONOI_MESH_UNPAIRED_ORIGINS`. |
| `VORONOI_MESH_VERIFY_TRACE` | correctness diagnostic | verification gate, only after fast verification rejects | Exact value `1` prints the fast-verifier fallback reason. |
| `VORONOI_MESH_RECONCILE_REBUILD` | differential oracle | defect-scoped `ReconcileOptions` snapshot | Exact value `1` selects the whole-buffer rebuild oracle instead of production in-place application. Read once only after mismatch records exist. |
| `VORONOI_MESH_RECONCILE_GLOBAL_DUPSCAN` | differential safety valve | defect-scoped `ReconcileOptions` snapshot | Exact value `1` substitutes the O(V) global duplicate scan for localized traversal. Read once only after mismatch records exist. |
| `VORONOI_MESH_LOCAL_REBUILD_DEBUG` | correctness diagnostic | attempt-scoped local-rebuild snapshot | Presence prints rebuild phase and acceptance diagnostics. Read once only after a rebuild trigger; disabled and clean computations perform no lookup. |

`VORONOI_MESH_PLANE_GRID_DENSITY` had no reader or backend in this repository. Its stale
performance-documentation entry was removed by QUAL-001H; reintroducing it requires a current
planar backend rather than a compatibility-only environment name.

## Campaign, benchmark, and manual-probe inputs

These variables are consumed by repository binaries or ignored/manual test targets, not by the
ordinary library API unless also listed above.

| Variables | Owner | Purpose |
|---|---|---|
| `VORONOI_MESH_CASE_DIST`, `VORONOI_MESH_CASE_N`, `VORONOI_MESH_CASE_SEED`, `VORONOI_MESH_CASE_PARAM` | `manual_probes` robustness and `tools` fidelity campaigns | Select one externally orchestrated distribution case. |
| `VORONOI_MESH_FIDELITY_CELLS`, `VORONOI_MESH_FIDELITY_EDGE_SAMPLES`, `VORONOI_MESH_LOCAL_REBUILD_MODE` | `tools` fidelity campaign | Configure sampling and the campaign's explicit `VoronoiConfig`; the mode variable is not a production config reader. |
| `VORONOI_MESH_BENCH_CAP_CENTER` | `bench_voronoi` | Choose the dense-cap placement for a benchmark distribution. |
| `VORONOI_MESH_BENCH_TARGET_MS`, `VORONOI_MESH_BENCH_SAMPLES`, `VORONOI_MESH_BENCH_HP_POOL`, `VORONOI_MESH_BENCH_CASE` | clipping microbench | Control sample sizing and case selection. |
| `VORONOI_MESH_PROBE_TARGETS`, `VORONOI_MESH_PROBE_LARGE`, `VORONOI_MESH_PROBE_N` | ignored cell-build unit probes | Select manual fallback/exhaustion probe scale and targets. |
| `VORONOI_SMALL_N_MAX`, `VORONOI_SMALL_N_SEEDS` | ignored small-N campaign | Bound an extended deterministic geometry campaign. Historical names lack the `MESH` component; they remain manual inputs until probe reorganization. |

## Private test-process variables

These names are test implementation details, not runtime knobs:

| Variable | Owner | Purpose |
|---|---|---|
| `VORONOI_MESH_VERIFY_GATE_CHILD` | `validation` unit test | Selects the enabled/disabled child-process branch used to test the real verification reader without mutating the parent unit-test process. |
| `VORONOI_MESH_TEST_ENV_RESTORE` | `tests/env_isolation.rs` | Private sentinel used only to prove exact restoration after panic and lock poisoning. |

## Test mutation policy and current writers

- `tests/support/env.rs` is the only helper for active integration-test mutation. It serializes
  mutations within that target, snapshots exact `OsString` values, restores them in reverse order,
  recovers a poisoned lock, and restores during panic unwinding.
- `tests/env_isolation.rs` directly seeds one private test-only variable so it can prove exact
  restoration after a caught panic; it restores the process's original value before asserting.
- `tests/edge_reconciliation.rs` uses it for `VORONOI_MESH_BIN_COUNT` and
  `VORONOI_MESH_RECONCILE_REBUILD`. Every active computation in that target participates in the
  same lock.
- `tests/local_rebuild_contract.rs` uses it for both the forced-bin case and ordinary control
  cases, preventing a forced bin count from leaking across parallel tests.
- The library verification-gate unit test no longer mutates `VORONOI_MESH_VERIFY` in the shared
  unit-test process. It runs filtered child-test processes for enabled and disabled cases, testing
  the real environment reader and error mapping without a production-only injection seam.
- Cargo excludes the wholly ignored `coincidence_probes` and `robustness_campaign` targets unless
  the internal `manual_probes` feature is selected, and excludes `fidelity_campaign` unless
  `tools` is selected. Campaign variables are read-only process inputs set before each isolated
  case; they do not require an in-process mutation guard.
