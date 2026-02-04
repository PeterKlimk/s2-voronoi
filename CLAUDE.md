# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the s2-voronoi crate.
For hex3 app/workspace guidance, see `CLAUDE.md` at the repo root.

For user-facing crate docs, see `README.md` and `docs/`.

## Build & Test Commands

```bash
cargo test --release   # Release mode recommended (debug is slow)
cargo test --release --features qhull  # Include qhull comparison tests
cargo clippy
cargo fmt
```

## Benchmarking

```bash
# Large-scale benchmark driver
cargo run --release --features tools --bin bench_voronoi -- 100k 500k 1m

# Detailed sub-phase timing
S2_VORONOI_TIMING_KV=1 cargo run --release --features tools,timing --bin bench_voronoi -- 500k --no-preprocess

# Inter-commit perf comparisons (build then run interleaved)
./scripts/bench_build.sh --chain 6
./scripts/bench_run.sh -s 500k -r 20 -m total
```

## Environment knobs

- `RAYON_NUM_THREADS=1`: force single-threaded mode (useful for stable perf comparisons).
- `S2_BIN_COUNT=<n>`: override sharded bin count (defaults to ~2x threads).
- `S2_VORONOI_TIMING_KV=1`: emit machine-readable `TIMING_KV ...` output (requires `timing`).

## Crate Overview

s2-voronoi computes spherical Voronoi diagrams on the unit sphere (S2) using kNN-driven half-space clipping. For each generator point, the algorithm finds k nearest neighbors, constructs bisector great-circle planes, and iteratively clips to produce the cell polygon. Vertices occur where 3+ bisector planes intersect.

The qhull backend (convex hull duality) is provided as ground truth for testing only.

## Documentation map

- `README.md`: user-facing overview, API summary, feature list.
- `docs/architecture.md`: algorithm + module map + glossary.
- `docs/performance.md`: benchmarking and perf knobs.
- `docs/live_dedup.md`: live vertex dedup and edge checks.

## Features

- `parallel` (default): rayon parallelism in cell construction
- `glam`: public `UnitVec3Like` impl + conversions for `glam::Vec3`
- `timing`: detailed timing instrumentation
- `profiling`: profiling helpers (e.g. disable inlining on hot functions)
- `microbench`: internal microbench harnesses (nightly `test` crate)
- `qhull`: convex hull backend for tests/benchmarks only
- `simd_clip`: use SIMD small-N clippers in the main clipper dispatch
- `fma`: prefer fused multiply-add via mul_add (may change results; can be slow without HW FMA)

Note: glam is always an internal dependency; the `glam` feature only gates the public API.

## Module Structure

- `src/lib.rs` - Public API: `compute()`, `validation::validate`
- `src/types.rs` - `UnitVec3`, `UnitVec3Like` trait
- `src/diagram.rs` - `SphericalVoronoi`, `CellView` diagram storage
- `src/error.rs` - `VoronoiError`
- `src/knn_clipping/` - kNN half-space clipping backend
- `src/cube_grid/` - Cube-map spatial index for fast kNN queries
- `src/convex_hull.rs` - qhull backend (feature-gated)

## Tests

Integration tests in `tests/`:
- `api.rs` - Public API tests
- `correctness.rs` - Geometric invariants (Euler characteristic, vertices on sphere, etc.)

## Known Limitations

- Requires nightly Rust (`#![feature(portable_simd)]`)
- Inputs should be normalized to unit length (not enforced, but assumed)
- Some cells may be degenerate near numerical edge cases; use `validation::validate`
