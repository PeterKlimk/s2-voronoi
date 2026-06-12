# Changelog

## 0.1.0 (unreleased)

Initial release.

- Planar Voronoi diagrams over a bounded rectangle (`compute_plane`,
  `PlanarVoronoi`, `PlaneRect`): the same kNN-clipping engine on a flat 2D
  grid, with rect walls handled as virtual generators so hull cells clip to
  the domain and the strict-subdivision contract carries over (disk-topology
  validation via `validation::validate_plane`; cell areas partition the
  rect). Generators within the planar weld radius (~1e-6 of the longer
  rect side, always including exact duplicates) are welded — required for
  graph validity, like the sphere's weld; detection reuses the kNN grid so
  duplicate-free inputs pay only a scan.
- Periodic (toroidal) planar Voronoi (`compute_plane_periodic`): the rect's
  opposite edges are identified; minimum-image clipping with a half-period
  exactness guard, canonically wrapped vertex storage with a `cell_polygon`
  unwrap helper, torus validation (V−E+F=0, every edge paired), and
  topology-aware measures enabling periodic Lloyd relaxation.
- Spherical Voronoi diagrams on the unit sphere via kNN-driven half-space
  clipping: per-cell parallel construction stitched into one consistent graph
  (see `docs/architecture.md` and the stitching invariant in
  `docs/live_dedup.md`).
- Correctness contract ("essentially Voronoi"): a hard topological guarantee
  (strictly valid subdivision, fuzz-asserted at 2-4.5M points in CI) over a
  soft geometric one; see `docs/correctness-contract.md` for the precise
  statement and the measured coincidence margins behind the default weld.
- Delaunay export (`delaunay_triangles()` on both diagram types): the dual
  triangulation read off the stored graph — CCW winding (delaunator/CGAL
  convention), canonical indices, complete on the sphere (2c−4 triangles)
  and the torus (2c), circumcenter-in-rect subset on the bounded rect,
  cocircular ties fan-triangulated.
- Point location (`build_locator()` → `SphereLocator` / `PlaneLocator`):
  reusable nearest-generator queries in near-constant time per query on all
  three topologies; periodic queries wrap, bounded-plane queries may lie
  outside the rect, welded inputs resolve to canonical cells.
- API: `compute` / `compute_with` / `compute_with_report`, cell views,
  edge-aligned neighbor adjacency (Delaunay edges), spherical and planar
  cell areas and centroids with a one-call `lloyd_step()` (Lloyd relaxation
  on sphere, rect, and torus), strict validation, vertex compaction, weld
  introspection, optional serde for all diagram types.
- Stable Rust (MSRV 1.88); explicit SIMD via `wide`.
