# s2-voronoi

Spherical Voronoi diagrams on the unit sphere (S2).

This crate computes a Voronoi diagram for points on the unit sphere using a kNN-driven half-space
clipping algorithm. Cells are built independently (embarrassingly parallel) and then stitched via
sharded “live” vertex deduplication.

## Status / requirements

- **Nightly Rust required** (`#![feature(portable_simd)]`).
- Inputs are **assumed** to be unit-normalized (not enforced; may debug-assert in hot paths).
- For strict subdivision/invariant checks on computed output, use `validation::validate`.

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
- `compute_with_report(&[P], VoronoiConfig) -> Result<ComputeOutput, VoronoiError>`
- `validation::validate(&SphericalVoronoi) -> ValidationReport`
  Use `ValidationReport::is_strictly_valid()` and the explicit issue summaries.
- `SphericalVoronoi`: `generators()`, `vertices()`, `iter_cells()`, `cell(i)`
- `CellView`: `vertex_indices`, `generator_index`, `len()`

## Configuration

`VoronoiConfig` controls preprocessing and (optional) termination fallback:

- `preprocess_mode`: explicit preprocessing contract:
  - `PreprocessMode::MergeDensity` (default): merge near-coincident generators using a
    density-based threshold
  - `PreprocessMode::MergeWithin(threshold)`: merge using an explicit threshold
  - `PreprocessMode::Disabled`: do not merge
- `termination_max_k`: cap k growth if termination fallback keeps requesting neighbors.

`compute_with_report` exposes:
- whether preprocessing actually merged generators and remapped cells
- strict validation of the returned diagram
- when preprocessing changed the solved generator set, strict validation of the effective diagram
  actually solved by the backend
- when preprocessing changed the solved generator set, the `effective_diagram` itself

If preprocessing is enabled and merges occur, `report.preferred_validation()` is usually the
validation result you want first; it selects effective validation when available. Likewise,
`output.preferred_diagram()` selects the effective diagram when available.

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
- Supported input / failure contract: `docs/supported-envelope.md`
- Active engineering issues / roadmap findings: `docs/engineering-findings.md`

## License

MIT OR Apache-2.0
