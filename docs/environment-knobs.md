# Environment-variable ownership

**Status:** active inventory for QUAL-001H

**Audited:** 2026-07-18

This document owns the process-environment inventory for the library, repository tools, campaigns,
and tests. Environment variables are process-global: library code must read them only at documented
cold seams, and tests that mutate them must restore the exact prior value even during unwinding.

Each Rust integration-test target runs in its own process. Isolation is therefore required between
parallel tests within a target, not between different files in `tests/`. Library unit tests share
one process with every other unit test and should prefer explicit internal inputs over environment
mutation.

## Library and tool runtime knobs

| Variable | Class | Owner / read boundary | Contract |
|---|---|---|---|
| `RAYON_NUM_THREADS` | supported operational | Rayon, before pool initialization | External Rayon contract; used to pin concurrency. |
| `VORONOI_MESH_BIN_COUNT` | supported tuning | `live_dedup::binning::target_bin_count`, once per computation | Integer shard target, clamped to `[6, 96]`; tests and `bench_bins` are current writers. |
| `VORONOI_MESH_GRID_DENSITY` | benchmark tuning | `policy::knn_grid_target_density`, first use via `OnceLock` | Parsed `f64` at least 1; intended for grid-density sweeps, not per-computation mutation. |
| `VORONOI_MESH_VERIFY` | supported verification | `validation::verify_enabled`, ordinary compute return gate | Exact value `1` enables the O(E) strict-validation gate. |
| `VORONOI_MESH_TIMING_KV` | instrumentation | `timing::real::PhaseTimings::report` | Presence emits machine-readable timing output when the `timing` feature is enabled. |
| `VORONOI_MESH_RECONCILE_TELEMETRY` | correctness diagnostic | reconciliation telemetry, defect path only | Presence repeats a read-only primary-round analysis and emits `RECONCILE_KV`; the clean path avoids even the lookup. |
| `VORONOI_MESH_RESOLUTION_KV` | correctness diagnostic | terminal output-resolution pass | Presence emits exact-zero resolution statistics. |
| `VORONOI_MESH_UNPAIRED_ORIGINS` | correctness diagnostic | live assembly, after mismatch collection | Presence prints mismatch-origin counts. |
| `VORONOI_MESH_VERIFY_TRACE` | correctness diagnostic | verification gate, only after fast verification rejects | Exact value `1` prints the fast-verifier fallback reason. |
| `VORONOI_MESH_RECONCILE_REBUILD` | differential oracle | reconciliation apply-mode selection | Exact value `1` selects the whole-buffer rebuild oracle instead of production in-place application. |
| `VORONOI_MESH_RECONCILE_GLOBAL_DUPSCAN` | differential safety valve | reconciliation defect path | Exact value `1` substitutes the O(V) global duplicate scan for localized traversal. |
| `VORONOI_MESH_LOCAL_REBUILD_DEBUG` | correctness diagnostic | local-rebuild orchestration/growth | Presence prints rebuild phase and acceptance diagnostics. |
| `VORONOI_MESH_LOCAL_REBUILD_GLOBAL_DELAUNAY` | feature probe | `local_rebuild_probe`-gated rebuild branch | Presence selects the global projected-Delaunay oracle. Internal feature only. |

`VORONOI_MESH_PLANE_GRID_DENSITY` had no reader or backend in this repository. Its stale
performance-documentation entry was removed by QUAL-001H; reintroducing it requires a current
planar backend rather than a compatibility-only environment name.

`VORONOI_MESH_LOCAL_REBUILD_PROBE_A0` was retired by QUAL-001H. A0 snapshot capture is now an
explicit `local_rebuild_probe` feature API scope. The scope is thread-local, composes when nested,
and restores its prior state during panic unwinding; it adds no ordinary production-path state or
environment lookup.

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
| `VORONOI_MESH_LOCAL_REBUILD_DIST`, `VORONOI_MESH_LOCAL_REBUILD_N`, `VORONOI_MESH_LOCAL_REBUILD_SEED`, `VORONOI_MESH_LOCAL_REBUILD_K` | ignored local-rebuild integration probes | Select manual defect/oracle workloads. |
| `VORONOI_MESH_CGAL_HULL3_BIN`, `VORONOI_MESH_NORM3D_FLAG_BANDS` | ignored external-oracle probes | Locate the CGAL helper and configure conditioning bands. |
| `VORONOI_SMALL_N_MAX`, `VORONOI_SMALL_N_SEEDS` | ignored small-N campaign | Bound an extended deterministic geometry campaign. Historical names lack the `MESH` component; they remain manual inputs until probe reorganization. |

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
- `src/bin/bench_bins.rs` mutates `VORONOI_MESH_BIN_COUNT` as a standalone process before running
  its workload; no test-process guard is required.
- Ignored `local_rebuild_probe` cases share one `stash_fast_triples` helper. Its explicit A0 scope
  is thread-local and panic-safe, so these probes do not mutate the process environment or a
  process-global rebuild switch. Cargo marks the all-ignored target as requiring the internal
  `local_rebuild_probe` feature; each research workload remains selected by test name.
- Cargo excludes the wholly ignored `coincidence_probes` and `robustness_campaign` targets unless
  the internal `manual_probes` feature is selected, and excludes `fidelity_campaign` unless
  `tools` is selected. Campaign variables are read-only process inputs set before each isolated
  case; they do not require an in-process mutation guard.
