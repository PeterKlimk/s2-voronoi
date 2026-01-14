# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the s2-voronoi crate.
For hex3 app/workspace guidance, see `CLAUDE.md` at the repo root.

## Build & Test Commands

```bash
cargo test --release   # Release mode recommended (debug is slow)
cargo test --release --features qhull  # Include qhull comparison tests
cargo clippy
cargo fmt
```

## Crate Overview

s2-voronoi computes spherical Voronoi diagrams on the unit sphere (S2) using kNN-driven half-space clipping. For each generator point, the algorithm finds k nearest neighbors, constructs bisector great-circle planes, and iteratively clips to produce the cell polygon. Vertices occur where 3+ bisector planes intersect.

The qhull backend (convex hull duality) is provided as ground truth for testing only.

## Features

- `parallel` (default): rayon parallelism in cell construction
- `glam`: public `UnitVec3Like` impl + conversions for `glam::Vec3`
- `timing`: detailed timing instrumentation
- `qhull`: convex hull backend for tests/benchmarks only

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
