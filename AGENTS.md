# AGENTS.md

This file provides guidance for coding agents working in the `s2-voronoi` crate.
For workspace-level guidance, see the repo-root `AGENTS.md`.

For user-facing crate docs, see `README.md` and `docs/`.

## Toolchain / Constraints

- Nightly Rust is required (`#![feature(portable_simd)]`).
- Run heavy checks in release mode where possible.
- Input points are assumed to be unit-normalized.

## Build & Test

```bash
cargo test --release
cargo test --release --features qhull
cargo clippy
cargo fmt
```

Useful targeted checks:

```bash
# Validate packed small-sort path compiles
cargo check --release --features packed_knn_sort_small

# API/correctness suites only
cargo test --release --test api --test correctness
```

## Benchmarking

```bash
# Large-scale benchmark driver
cargo run --release --features tools --bin bench_voronoi -- 100k 500k 1m

# Detailed sub-phase timing
S2_VORONOI_TIMING_KV=1 cargo run --release --features tools,timing --bin bench_voronoi -- 500k --no-preprocess

# Inter-commit perf comparisons
./scripts/bench_build.sh --chain 6
./scripts/bench_run.sh -s 500k -r 20 -m total
```

## Environment Knobs

- `RAYON_NUM_THREADS=1`: force single-threaded mode (stable perf comparisons).
- `S2_BIN_COUNT=<n>`: override sharded bin count (defaults to about 2x threads).
- `S2_VORONOI_TIMING_KV=1`: emit machine-readable timing lines (`timing` feature).

## Crate Overview

`s2-voronoi` computes spherical Voronoi diagrams on the unit sphere (S2) using kNN-driven half-space clipping.

High-level flow per generator:

1. Find candidate neighbors via cube-map spatial index.
2. Build bisector great-circle constraints.
3. Clip cell in local gnomonic/topological 2D representation.
4. Deduplicate/assemble shared vertices across cells.

The `qhull` backend is for comparison/testing, not primary production path.

## Documentation Map

- `README.md`: user-facing overview and API summary.
- `docs/architecture.md`: algorithm and module map.
- `docs/performance.md`: benchmark guidance and perf knobs.
- `docs/live_dedup.md`: live dedup and edge-check design.

## Module Map (Current)

```text
src/
├── lib.rs                         # Public API and feature-gated internal exports
├── types.rs                       # UnitVec3 / UnitVec3Like
├── diagram.rs                     # SphericalVoronoi storage
├── validation.rs                  # Topology/consistency checks
├── error.rs                       # VoronoiError
├── fp.rs                          # Numeric helper ops
├── knn_clipping/                  # Main backend
│   ├── compute.rs                 # End-to-end backend orchestration
│   ├── preprocess.rs              # Near-coincident merge pass
│   ├── edge_repair.rs             # Post-assembly edge repairs
│   ├── live_dedup/                # Sharded dedup + assembly
│   ├── topo2d/                    # Gnomonic/topological clipping
│   └── timing/                    # Timing feature plumbing
├── cube_grid/                     # Spatial index + query stack
│   ├── build.rs                   # Grid construction
│   ├── projection.rs              # Face/uv/st conversion helpers
│   ├── query/                     # Directed resumable kNN query path
│   └── packed_knn/                # Packed batched directed kNN
├── generated/
│   └── sort_nets.rs               # Auto-generated sorting network code
├── sort.rs                        # Internal small-sort utilities (feature/test use)
└── convex_hull.rs                 # qhull backend (feature: qhull)
```

## Features

- `parallel` (default): rayon-based parallel cell construction.
- `glam`: public `UnitVec3Like` impl/conversions for `glam::Vec3`.
- `timing`: detailed timing instrumentation.
- `profiling`: helpers for profiling runs (e.g. inline control).
- `microbench`: internal microbench harnesses.
- `qhull`: convex-hull backend for comparison tests/bench.
- `simd_clip`: SIMD small-N clippers in clipper dispatch.
- `fma`: prefer fused multiply-add (`mul_add`) where used.
- `packed_knn_sort_small`: enable internal small-sort path in packed kNN.
- `tools`: benchmark/utility binaries.

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

- Nightly Rust required.
- Inputs should be unit-normalized.
- Numerical edge cases can still produce degeneracies; use `validation::validate`.
