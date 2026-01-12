# AGENTS.md

This file provides context for LLM coding assistants working with the s2-voronoi crate.
For hex3 app/workspace guidance, see `AGENTS.md` at the repo root.

## Build & Test

```bash
cargo test --release   # Release mode recommended (debug is slow)
cargo test --release --features qhull  # Include qhull comparison tests
cargo clippy
cargo fmt
```

## Crate Overview

s2-voronoi computes spherical Voronoi diagrams on the unit sphere (S2) using kNN-driven half-space clipping.

**Algorithm:** For each generator point, find k nearest neighbors using a cube-map spatial index, construct bisector great-circle planes, and iteratively clip to produce the cell polygon. Vertices occur where 3+ bisector planes intersect. All cells are computed independently (embarrassingly parallel).

The qhull backend (convex hull duality) is provided as ground truth for testing only.

## Module Structure

```
src/
├── lib.rs            # Public API: compute(), VoronoiOutput, VoronoiDiagnostics
├── types.rs          # UnitVec3, UnitVec3Like trait
├── diagram.rs        # SphericalVoronoi, CellView diagram storage
├── error.rs          # VoronoiError
├── knn_clipping/     # kNN half-space clipping backend
├── cube_grid/        # Cube-map spatial index for fast kNN queries
└── convex_hull.rs    # qhull backend (feature: qhull)
```

## Features

- `parallel` (default): rayon parallelism in cell construction
- `glam`: public `UnitVec3Like` impl + conversions for `glam::Vec3`
- `timing`: detailed timing instrumentation
- `qhull`: convex hull backend for tests/benchmarks only

Note: glam is always an internal dependency; the `glam` feature only gates the public API.

## Public API

- `UnitVec3`, `UnitVec3Like` - Input point types
- `compute(&[P]) -> Result<VoronoiOutput, VoronoiError>` - Main entry point
- `VoronoiOutput` - Contains `diagram: SphericalVoronoi` and `diagnostics: VoronoiDiagnostics`
- `SphericalVoronoi` - Diagram with `generators`, `vertices`, `iter_cells()`, `cell(i)`
- `CellView` - Cell accessor with `vertex_indices`, `generator_index`, `len()`

## Tests

Integration tests in `tests/`:
- `api.rs` - Public API tests (10 tests)
- `correctness.rs` - Geometric invariants (8 tests)

## Known Limitations

- Requires nightly Rust (`#![feature(portable_simd)]`)
- Inputs should be normalized to unit length (not enforced, but assumed)
- Some cells may be degenerate near numerical edge cases; check `VoronoiDiagnostics`
