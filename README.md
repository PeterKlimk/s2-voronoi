# s2-voronoi

Spherical Voronoi diagrams on the unit sphere (S2).

This crate computes a Voronoi diagram for points on the unit sphere using a kNN-driven half-space
clipping algorithm. Cells are built independently (embarrassingly parallel) and then stitched via
sharded “live” vertex deduplication.

## Status / requirements

- **Nightly Rust required** (`#![feature(portable_simd)]`).
- Inputs are **assumed** to be unit-normalized (not enforced; may debug-assert in hot paths).
- Some degenerate cells can occur for near-coincident points or numerical edge cases; use
  `validation::validate`.

## Quickstart

```rust
use s2_voronoi::{compute, UnitVec3};

let points = vec![
    UnitVec3::new(1.0, 0.0, 0.0),
    UnitVec3::new(0.0, 1.0, 0.0),
    UnitVec3::new(0.0, 0.0, 1.0),
    UnitVec3::new(-1.0, 0.0, 0.0),
    UnitVec3::new(0.0, -1.0, 0.0),
    UnitVec3::new(0.0, 0.0, -1.0),
];

let diagram = compute(&points)?;
for cell in diagram.iter_cells() {
    let g = diagram.generator(cell.generator_index);
    let boundary = cell.vertex_indices.iter().map(|&i| diagram.vertex(i as usize));
    let _ = (g, boundary);
}
# Ok::<(), s2_voronoi::VoronoiError>(())
```

## API overview

- `compute(&[P]) -> Result<SphericalVoronoi, VoronoiError>`
- `compute_with(&[P], VoronoiConfig)`
- `validation::validate(&SphericalVoronoi) -> ValidationReport`
- `SphericalVoronoi`: `generators`, `vertices`, `iter_cells()`, `cell(i)`
- `CellView`: `vertex_indices`, `generator_index`, `len()`

## Configuration

`VoronoiConfig` controls preprocessing and (optional) termination fallback:

- `preprocess`: merges near-coincident generators for robustness (adds overhead).
- `preprocess_threshold`: override merge distance threshold (otherwise density-based default).
- `termination_max_k`: cap k growth if termination fallback keeps requesting neighbors.

## Features

- `parallel` (default): rayon parallelism in cell construction.
- `glam`: public `UnitVec3Like` impl + conversions for `glam::Vec3` (glam is always an internal dep).
- `timing`: detailed phase/sub-phase timing reports.
- `profiling`: profiling helpers (e.g. selectively disable inlining on hot functions).
- `microbench`: internal microbench harnesses (requires nightly `test` crate).
- `qhull`: convex hull backend used for tests/bench comparisons only.
- `simd_clip`: use SIMD small-N clippers in the main clipper dispatch.
- `fma`: prefer fused multiply-add via `mul_add` (may change results; can be slower without HW FMA).

## Documentation

- Design / algorithm notes: `docs/architecture.md`
- Performance + benchmarking: `docs/performance.md`
- Live vertex dedup and edge checks: `docs/live_dedup.md`

## License

MIT OR Apache-2.0

