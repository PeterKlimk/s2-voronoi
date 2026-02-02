# Architecture / algorithm notes

s2-voronoi computes a spherical Voronoi diagram on the unit sphere (S2). The implementation is
optimized for building many independent cells efficiently.

## High-level pipeline

For each generator point `g`:

1. Find candidate neighbors via a cube-map spatial index (`cube_grid/`).
2. For each neighbor `n`, form a bisector great-circle plane (a half-space constraint).
3. Clip an initially-unbounded polygon in a **gnomonic projection** (local tangent plane) to
   produce the cell boundary in 2D (`knn_clipping/topo2d/`).
4. Extract spherical vertices (unit vectors) from the final 2D polygon and deduplicate them across
   cells (“live dedup”) to build the global vertex list and per-cell index lists
   (`knn_clipping/live_dedup/`).

Cells are computed independently; “global” work is limited to dedup/overflow handling and optional
edge repair.

## Why gnomonic projection?

The cell boundary is an intersection of spherical half-spaces (great-circle constraints). In a
gnomonic projection centered at `g`, every great circle becomes a line, so the clipping problem
reduces to repeated convex polygon clipping by half-planes in 2D.

This projection is **different per generator**, which is why most 2D geometric quantities cannot
be shared between cells, even when an underlying 3D plane is shared.

## Module map

- `src/lib.rs`: public API (`compute`, `compute_with`, `validation`).
- `src/diagram.rs`: storage (`SphericalVoronoi`, `CellView`).
- `src/types.rs`: `UnitVec3`, `UnitVec3Like`.
- `src/knn_clipping/`: kNN + clipping backend.
  - `topo2d/`: gnomonic projection, half-planes, and convex clipping.
  - `live_dedup/`: sharded vertex ownership + edge-check propagation.
  - `edge_repair.rs`: best-effort repairs for missing/mismatched edges.
  - `preprocess.rs`: merge near-coincident points.
  - `timing.rs`: optional timing + histograms.
- `src/cube_grid/`: cube-map spatial index + packed-kNN helpers.
- `src/convex_hull.rs` (`qhull` feature): convex-hull dual backend (tests/bench comparisons).

## Glossary

- **Generator**: input point that owns one Voronoi cell.
- **Neighbor**: candidate generator used to constrain a cell during clipping.
- **Half-plane / half-space**: bisector constraint induced by neighbor `n`.
- **Gnomonic projection**: maps great circles to lines in a local tangent plane at `g`.
- **Bin / shard**: a partition of generators used for parallel cell construction and localized
  communication.
- **Edge check**: a compact record used to propagate per-edge vertex indices between adjacent
  cells (see `docs/live_dedup.md`).
