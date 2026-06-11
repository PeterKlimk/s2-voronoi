# s2-voronoi

Fast spherical Voronoi diagrams on the unit sphere (S2).

Most spherical Voronoi implementations go through a 3D convex hull (qhull, scipy) and slow down
sharply past tens of thousands of points. This crate instead builds every cell independently by
kNN-driven half-space clipping — the construction usually reserved for GPU implementations — and
stitches the per-cell results into a single consistent graph on the CPU. Cell construction is
embarrassingly parallel; cross-cell work is limited to sharded "live" vertex deduplication and a
narrow reconciliation pass.

Design target: 2.5M points in under 500ms on a 6-core desktop CPU (Ryzen 3600 class) — roughly an
order of magnitude faster than convex-hull approaches at that scale.

## The correctness contract, in one paragraph

The output is *essentially Voronoi*: the crate aims for a **hard topological guarantee** (the
returned graph is a strictly valid subdivision of the sphere — checkable via
`validation::validate`) and a **soft geometric guarantee** (vertex positions accurate to
floating-point working precision; features at the resolution floor are handled by policy).
Exactness in the mathematical sense is not promised — no floating-point implementation can — but
graph validity is treated as non-negotiable and is fuzz-tested at multi-million point counts. See
`docs/correctness-contract.md` for the precise statement, the coincident-input (weld) policy, and
the measured safety margins behind it.

## Status / requirements

- Pre-release (0.1). The supported envelope is documented and test-backed, but the API is not yet
  stable; see `docs/todo.md` for the path to release.
- **Stable Rust** (MSRV 1.88). Explicit SIMD via the `wide` crate; nightly is no longer required.
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
- `SphericalVoronoi`: `generators()`, `vertices()`, `iter_cells()`, `cell(i)`,
  `build_adjacency()`, `cell_area(i)`, `cell_centroid(i)`, `weld_map()`, `compact_vertices()`
- `CellView`: `vertex_indices`, `generator_index`, `len()`
- `CellAdjacency`: per-cell Voronoi neighbors aligned with boundary edges (`neighbors_of(i)`);
  the neighbor pairs are the Delaunay edges of the generator set
- Lloyd relaxation (centroidal Voronoi tessellation) is one loop:
  `points = (0..n).map(|i| diagram.cell_centroid(i)).collect()` and recompute.

## Configuration

`VoronoiConfig` controls preprocessing and (optional) termination fallback:

- `preprocess_mode`: coincident-generator handling:
  - `PreprocessMode::MergeDensity` (current default): merge near-coincident generators using a
    density-based threshold. Scheduled to be replaced by a fixed-radius weld with corrected
    output semantics — see `docs/correctness-contract.md`.
  - `PreprocessMode::MergeWithin(threshold)`: merge using an explicit threshold
  - `PreprocessMode::Disabled`: do not merge (caller certifies generator separation)
- `termination_max_k`: cap k growth if termination fallback keeps requesting neighbors.

`compute_with_report` exposes whether preprocessing merged generators, the effective diagram the
backend actually solved, and strict validation of both views; `report.preferred_validation()` and
`output.preferred_diagram()` select the right view after a merged solve.

## Features

- `parallel` (default): rayon parallelism in cell construction.
- `glam`: public `UnitVec3Like` impl + conversions for `glam::Vec3` (glam is always an internal dep).
- `timing`: detailed phase/sub-phase timing reports.
- `qhull`: convex hull backend used for tests/bench comparisons only.
- Internal/research flags (`profiling`, `microbench`, `fma`,
  `packed_knn_sort_small`): not part of the public surface; subject to consolidation.

## Documentation

- Correctness contract + coincidence policy: `docs/correctness-contract.md`
- Design / algorithm notes (including the parallel-stitching consistency model):
  `docs/architecture.md`
- Supported input / failure contract: `docs/supported-envelope.md`
- Active roadmap / prioritized next steps: `docs/todo.md`
- Performance + benchmarking: `docs/performance.md`
- Live vertex dedup and edge checks: `docs/live_dedup.md`
- Engineering issues / findings log: `docs/engineering-findings.md`

## License

MIT OR Apache-2.0
