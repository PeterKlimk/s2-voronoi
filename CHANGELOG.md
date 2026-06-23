# Changelog

## 0.1.0 (unreleased)

Initial release.

- Edge-repair observability and coverage: `ComputeReport::unresolved_edge_pairs`
  reports each shared-edge mismatch that reached post-assembly reconciliation,
  tagged with an `UnresolvedEdgeOrigin` naming the detection path; a
  deterministic net (`tests/edge_repair_net.rs`) pins a real 2M-scale defect
  site down to a ~1.7k-point fixture and exercises the in-bin and cross-bin
  detection/repair paths, asserting strict post-repair validity (see
  engineering-findings #13).
- Micro-optimization batch from a screened 17-branch matrix:
  paired-proven stack (~-36ms total
  at 500k ST, -120ms cell construction at 2M) plus eight prior-better
  merges (allocation removal in the frontier cache, integer-compare
  OrdF32 with the -0.0/NaN Eq/Ord fix, fixed-size SIMD chunk types,
  invariant-load hoists, a sqrt-free projection bound).
- Fixed an input-reachable panic in the bitmask clipper: with a polygon at
  the full 64-vertex budget, a transition-mask shift overflowed and indexed
  past the vertex arrays (first reachable via cap-edge cells of a 1M-point
  clustered input). Such inputs now take the clean vertex-budget
  `ComputationFailed` path; regression-tested with a full 64-gon clip.
- Input canonicalization (P5 stage 0): sphere inputs are renormalized once
  at compute entry (f64-normalize, rounded back to f32) and the per-builder
  renormalization is gone, so every pipeline stage consumes identical bits
  per generator; the returned diagram's generators are the canonicalized
  points (within ~1 ulp of the input). The bisector construction now
  carries the exact two-sided chord compensation (free: one cached scale
  per cell). Effect: natural unresolved-edge defects at 2M dropped to zero
  across ten fuzz seeds (previously ~1 site per ~4 seeds) — the dominant
  source of cross-cell epsilon disagreements was the renormalization
  asymmetry. Note: inputs differing only radially (off-unit ulps) now
  collapse to the same generator at entry.
- The sphere weld is grid-integrated: the query grid is built on the raw
  points and doubles as the coincidence detector, and welds compact the
  grid's point arrays in place (bit-identical to a rebuild on the effective
  points) instead of paying a standalone quantized-key sort pass. At 2M
  points single-threaded the weld preprocess drops from ~378ms to ~45ms
  (paired interleaved A/B vs week-start: -7.6% median total at 2M, -7.2%
  at 500k, default mode).
  `MergeWithin` radii too large for grid adjacency fall back to the
  standalone detector. (`TIMING_KV` note: `knn_build_ms` now precedes
  `preprocess_ms` in pipeline order, and preprocess no longer includes any
  grid work.)
- Edge reconciliation is O(defects) instead of O(diagram): merges collect
  into a sparse union-find (no per-run O(V) init) and apply by patching only
  the cells that can reference a merged vertex (located via vertex-key
  triplets), in place. On a defect-bearing 2M single-threaded run the repair
  drops from ~382ms to ~0.06ms; the original full rebuild is retained as a
  differential oracle behind `S2_EDGE_REPAIR_REBUILD=1`, with tests pinning
  identical per-cell output between backends.
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
  outside the rect, welded inputs resolve to canonical cells; `locate_many`
  batches queries across all cores.
- Performance: the periodic pipeline runs the same packed SIMD kNN stage as
  the bounded plane (~2x at 500k single-threaded, to within ~10% of the
  bounded pipeline), and both planar pipelines emit the sphere's TIMING_KV
  profiling schema under the `timing` feature.
- API: `compute` / `compute_with` / `compute_with_report`, cell views,
  edge-aligned neighbor adjacency (Delaunay edges), spherical and planar
  cell areas and centroids with a one-call `lloyd_step()` (Lloyd relaxation
  on sphere, rect, and torus), strict validation, vertex compaction, weld
  introspection, optional serde for all diagram types.
- Stable Rust (MSRV 1.88); explicit SIMD via `wide`.
