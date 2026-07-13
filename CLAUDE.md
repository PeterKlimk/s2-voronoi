# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) working in the `voronoi-mesh` crate.
For workspace-level guidance, see the repo-root `CLAUDE.md`.

For user-facing crate docs, see `README.md` and `docs/`.

## Toolchain / Constraints

- Stable Rust (MSRV 1.88); explicit SIMD via the `wide` crate behind the `src/fp.rs` backend seam.
- Run heavy checks in release mode where possible.
- Input points are assumed to be unit-normalized.

## Build & Test

```bash
cargo test --release
cargo clippy
cargo fmt
```

Useful targeted checks:

```bash
# API/correctness suites only
cargo test --release --test api --test correctness
```

## Benchmarking

```bash
# Large-scale benchmark driver
cargo run --release --features tools --bin bench_voronoi -- 100k 500k 1m

# Detailed sub-phase timing
VORONOI_MESH_TIMING_KV=1 cargo run --release --features tools,timing --bin bench_voronoi -- 500k --no-preprocess

# Inter-commit perf comparisons (interleaved, paired, single-thread)
./scripts/bench_build.sh --chain 6
./scripts/bench_run.sh -s 500k -r 20 -m total

# Distribution + size matrix with CSV (bench_voronoi --dist: fib uniform
# clustered bimodal gradient outlier splittable mega; --dist-param tunes
# gradient k / mega fraction). Clustered inputs need explicit dists — uniform
# alone misses density-contrast regressions (see docs/performance.md).
./scripts/bench_run.sh -s "500k 2m" -d "uniform mega" --seeds "1 2 3" --csv /tmp/bench.csv
```

## Environment Knobs

- `RAYON_NUM_THREADS=1`: force single-threaded mode (stable perf comparisons).
- `VORONOI_MESH_BIN_COUNT=<n>`: override sharded bin count (defaults to about 2x threads).
- `VORONOI_MESH_TIMING_KV=1`: emit machine-readable timing lines (`timing` feature).
- `VORONOI_MESH_VERIFY=1`: run the full topological validator after every build on the plain `compute` fast path (which otherwise skips it) and return an error on any strict-validity failure. Off by default; O(E) cost per call. Belt-and-braces for callers wanting output validity machine-checked regardless of detection bookkeeping.
- `VORONOI_MESH_EDGE_REPAIR_REBUILD=1`: select the full-rewrite repair backend (differential oracle for the in-place default).

## Crate Overview

`voronoi-mesh` computes spherical Voronoi diagrams on the unit sphere (S2) using kNN-driven half-space clipping. (A planar/toroidal backend over the same dedup engine lives on the `parked/planar-backend` branch, removed from the release surface.)

High-level flow per generator:

1. Find candidate neighbors via cube-map spatial index.
2. Build bisector great-circle constraints.
3. Clip cell in local gnomonic/topological 2D representation.
4. Deduplicate/assemble shared vertices across cells.

## Documentation Map

- `README.md`: user-facing overview and API summary.
- `docs/architecture.md`: the algorithm (per-cell construction + stitching) and module map.
- `docs/correctness.md`: guarantees, outcome classes, and limits.
- `docs/performance.md`: benchmark guidance and perf knobs.

## Module Map (Current)

```text
src/
├── lib.rs                         # Public API and feature-gated internal exports
├── types.rs                       # UnitVec3 / UnitVec3Like
├── diagram.rs                     # SphericalVoronoi storage
├── validation.rs                  # Topology checks
├── locate.rs                      # Point-location API (SphereLocator)
├── error.rs                       # VoronoiError
├── fp.rs                          # Numeric helper ops + OrdF32
├── tolerances.rs                  # Centralized numerical tolerances
├── policy.rs                      # Grid sizing and neighbor policy knobs
├── timing/                        # Timing feature plumbing (crate-wide)
├── live_dedup/                    # Geometry-agnostic dedup/assembly engine
│   ├── cell_output.rs             # VertexKey / CellOutputBuffer<P> vocabulary
│   ├── emit.rs                    # Shared per-cell shard emission seam
│   ├── binning.rs                 # CSR bin assignment core + layouts
│   ├── edge_checks.rs             # Edge-check resolve/forward
│   └── assemble.rs                # Shard concat + deferred patching
├── knn_clipping/                  # Spherical backend
│   ├── compute.rs                 # End-to-end orchestration
│   ├── driver.rs                  # Per-bin cell-build driver (sphere)
│   ├── preprocess.rs              # Near-coincident weld pass
│   ├── edge_reconcile.rs          # Post-assembly edge reconciliation (shared)
│   ├── cell_build/                # Single-cell construction loop
│   └── topo2d/                    # Gnomonic/topological clipping
├── cube_grid/                     # Spherical spatial index + query stack
│   ├── build.rs                   # Grid construction
│   ├── projection.rs              # Face/uv/st conversion helpers
│   ├── query/                     # Directed eligibility + shell frontier
│   └── packed_knn/                # Packed batched directed kNN
├── generated/
│   └── sort_nets.rs               # Auto-generated sorting network code
└── sort.rs                        # Internal small-sort utilities (feature/test use)
```

## Features

- `parallel` (default): rayon-based parallel cell construction.
- `glam`: public `UnitVec3Like` impl/conversions for `glam::Vec3`.
- `serde`: Serialize/Deserialize for diagram types.
- Internal: `timing` (instrumentation), `profiling` (inline control),
  `microbench` (harnesses), `simd_scalar` (non-`wide` 8-lane fallback),
  `fma` (mul_add; off by default, see ledger), `tools` (bench binaries),
  `p5_shadow` (P5 stage-1 canonical-vs-local clip decision audit).

## Tests

Primary integration tests in `tests/`:

- `api.rs`: public API behavior.
- `correctness.rs`: geometric/topological invariants.
- `validation.rs`: validation report checks.
- `adversarial.rs`: stress and pathological distributions.

## Git Workflow Policy For Agents

- Agents may edit files in workspace as needed for requested tasks.
- Agents may commit without explicit per-turn approval when the requested change is complete, scoped, and validated.
- Do not auto-commit exploratory, partial, or uncertain work; ask before committing in those cases.
- For substantial work, prefer a topic branch like `agent/<short-topic>` unless user says otherwise.
- Keep commits scoped to a single logical change.
- Run relevant validation before commit and report what was run.
- Report the commit hash and message after committing.
- Do not push, force-push, amend, rebase, or reset unless explicitly requested.
- Do not include unrelated file churn just to satisfy formatting/linting unless requested.

## Change Checklist (Recommended)

1. Implement minimal coherent change.
2. Run `cargo fmt`.
3. Run focused tests/checks relevant to changed modules.
4. If broad behavior changed, run `cargo test --release`.
5. Summarize results and residual risks.

## Known Limitations

- Inputs should be unit-normalized.
- Numerical edge cases can still produce degeneracies; use `validation::validate`.
