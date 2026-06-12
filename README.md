# s2-voronoi

Fast spherical Voronoi diagrams on the unit sphere (S2) — and planar Voronoi diagrams over a
bounded rectangle, through the same parallel engine.

Most spherical Voronoi implementations go through a 3D convex hull (qhull, scipy) and slow down
sharply past tens of thousands of points. This crate instead builds every cell independently by
kNN-driven half-space clipping — the construction usually reserved for GPU implementations — and
stitches the per-cell results into a single consistent graph on the CPU. Cell construction is
embarrassingly parallel; cross-cell work is limited to sharded "live" vertex deduplication and a
narrow reconciliation pass. The dedup/stitching engine is geometry-agnostic: the spherical and
planar backends are two drivers over one core.

Design target: 2.5M spherical points in under 500ms on a 6-core desktop CPU (Ryzen 3600 class) —
roughly an order of magnitude faster than convex-hull approaches at that scale. The planar
backend computes 2M cells in about a second on the same hardware, 3–4x faster than
delaunator-based planar crates at that size (and the gap grows with n: per-point cost is near
constant here versus their O(n log n)).

## The correctness contract, in one paragraph

The output is *essentially Voronoi*: the crate aims for a **hard topological guarantee** (the
returned graph is a strictly valid subdivision of the sphere — checkable via
`validation::validate` — or of the rectangle, via `validation::validate_plane`) and a **soft
geometric guarantee** (vertex positions accurate to
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

## Planar Voronoi

`compute_plane` produces a Voronoi diagram of a bounded axis-aligned rectangle: every cell is a
convex polygon, hull cells are clipped to the rectangle (its walls act as virtual generators in
the engine), and cell areas partition the rect exactly.

```rust
use s2_voronoi::{compute_plane, PlaneRect};

let points = vec![[0.25f32, 0.25], [0.75, 0.25], [0.5, 0.8]];
let diagram = compute_plane(&points, PlaneRect::unit())?;
for i in 0..diagram.num_cells() {
    let boundary = diagram.cell(i).iter().map(|&v| diagram.vertex(v as usize));
    let _ = boundary;
}
# Ok::<(), s2_voronoi::VoronoiError>(())
```

- Accepts any point count >= 1 (a single generator owns the whole rect).
- Generators within the planar weld radius (~1e-6 of the longer rect side) share one cell, exposed
  via `PlanarVoronoi::weld_map()` — required for graph validity, like the sphere's weld; see
  `docs/correctness-contract.md` for the probe data behind the radius.
- The domain transform is uniform per axis (Voronoi structure is not invariant under anisotropic
  scaling); high-aspect rects are supported, with grid sizing that follows the occupied band.

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
- Planar: `compute_plane(&[P], PlaneRect) -> Result<PlanarVoronoi, VoronoiError>`;
  `validation::validate_plane(&PlanarVoronoi) -> PlaneValidationReport`;
  `PlanarVoronoi`: `generators()`, `vertices()`, `iter_cells()`, `cell(i)`,
  `build_adjacency()`, `cell_area(i)`, `cell_centroid(i)`, `weld_map()`, `rect()` —
  planar Lloyd relaxation is the same one-loop recipe as the sphere's
- Lloyd relaxation (centroidal Voronoi tessellation) is one loop:
  `points = (0..n).map(|i| diagram.cell_centroid(i)).collect()` and recompute.

## Configuration

`VoronoiConfig` (spherical pipeline) controls preprocessing and the optional packed expansion
stage:

- `preprocess_mode`: coincident-generator handling:
  - `PreprocessMode::Weld` (default): weld generators within the fixed weld radius (~1.4e-6
    chord, derived from f32 rounding with measured margin — see
    `docs/correctness-contract.md`). Welded inputs share one cell, exposed via
    `SphericalVoronoi::weld_map()`.
  - `PreprocessMode::MergeWithin(threshold)`: weld within an explicit threshold
  - `PreprocessMode::Disabled`: no welding (caller certifies generator separation above the
    weld radius)
- `packed_knn_expand_r2`: enable the cold ring-2 packed expansion stage (off by default).

The planar pipeline currently takes no configuration; its weld radius is fixed (see above).

`compute_with_report` exposes whether preprocessing merged generators, the effective diagram the
backend actually solved, and strict validation of both views; `report.preferred_validation()` and
`output.preferred_diagram()` select the right view after a merged solve.

## Features

- `parallel` (default): rayon parallelism in cell construction.
- `glam`: public `UnitVec3Like` impl + conversions for `glam::Vec3` (glam is always an internal dep).
- `serde`: Serialize/Deserialize for `UnitVec3`, `SphericalVoronoi` (weld map included), and
  `CellAdjacency`.
- `timing`: detailed phase/sub-phase timing reports.
- `qhull`: convex hull backend used for tests/bench comparisons only.
- Internal/research flags (`profiling`, `microbench`, `fma`): not part of the public
  surface; subject to consolidation.

## Documentation

- Correctness contract + coincidence policy: `docs/correctness-contract.md`
- Design / algorithm notes (including the parallel-stitching consistency model):
  `docs/architecture.md`
- Supported input / failure contract: `docs/supported-envelope.md`
- Active roadmap / prioritized next steps: `docs/todo.md`
- Performance + benchmarking: `docs/performance.md`
- Optimization ideas ledger (incl. negative results): `docs/optimization-ideas.md`
- Live vertex dedup and edge checks: `docs/live_dedup.md`
- Engineering issues / findings log: `docs/engineering-findings.md`

## License

MIT OR Apache-2.0
