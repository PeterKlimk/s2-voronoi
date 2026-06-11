# Changelog

## 0.1.0 (unreleased)

Initial release.

- Planar Voronoi diagrams over a bounded rectangle (`compute_plane`,
  `PlanarVoronoi`, `PlaneRect`): the same kNN-clipping engine on a flat 2D
  grid, with rect walls handled as virtual generators so hull cells clip to
  the domain and the strict-subdivision contract carries over (disk-topology
  validation via `validation::validate_plane`; cell areas partition the
  rect). Exact-bit duplicate inputs are welded; no near-coincidence radius
  is needed on the plane.
- Spherical Voronoi diagrams on the unit sphere via kNN-driven half-space
  clipping: per-cell parallel construction stitched into one consistent graph
  (see `docs/architecture.md` and the stitching invariant in
  `docs/live_dedup.md`).
- Correctness contract ("essentially Voronoi"): a hard topological guarantee
  (strictly valid subdivision, fuzz-asserted at 2-4.5M points in CI) over a
  soft geometric one; see `docs/correctness-contract.md` for the precise
  statement and the measured coincidence margins behind the default weld.
- API: `compute` / `compute_with` / `compute_with_report`, cell views,
  edge-aligned neighbor adjacency (Delaunay edges), spherical cell areas and
  centroids (Lloyd relaxation in three lines), strict validation, vertex
  compaction, weld introspection, optional serde.
- Stable Rust (MSRV 1.88); explicit SIMD via `wide`.
