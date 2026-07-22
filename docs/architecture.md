# Architecture

voronoi-mesh computes Voronoi diagrams on the unit sphere. The design builds every cell
independently and in parallel, then stitches the cells into one shared graph. This document
describes how that works and where the code lives.

## World-space embedding

`SphereEmbedding` keeps translation and uniform scale outside the geometric backend. World-space
sites are converted in f64 to directions from the declared center, normalized with scale-safe
arithmetic, rounded to the backend's canonical f32 unit directions, and then processed by the
unchanged pipeline below. `EmbeddedSphericalVoronoi` stores that unit diagram plus the center and
radius; world positions and physical areas are derived on demand.

Radial distance from the center is intentionally not part of site identity. Translation and
uniform scaling preserve spherical Voronoi topology, while non-uniform transforms, weighted sites,
and sites whose different radii affect distance do not. Those are different geometric problems,
not embedding modes of this backend.

## Input adaptation

The unit-sphere API has one owned f32 backend buffer. `UnitVec3Like` inputs and the `compute_by`
family both write directly into that allocation before the common validation/canonicalization
pipeline starts. Closure ingest lets application records and foreign or version-mismatched math
types participate without an orphan-rule-sensitive trait implementation or a caller-allocated
coordinate vector. It does not make the diagram generic over the caller's point type, and it does
not provide an f64 path: the closure deliberately returns `[f32; 3]` under the existing storage and
search model.

## Pipeline stages and vocabulary

The computation uses the following stage names consistently. “Stitching” is a useful umbrella for
the post-construction stages, but it is not a synonym for reconciliation or local rebuilding.

| Stage | Contract | Primary owners |
|---|---|---|
| Input adaptation and canonicalization | Convert caller inputs into the owned, finite f32 unit-direction buffer used by the backend. | `types.rs`, `lib.rs`, `knn_clipping/compute.rs` |
| Preprocessing and grid build | Optionally weld near-coincident canonical generators into the effective generator set and build its cube-map index. | `knn_clipping/preprocess.rs`, `cube_grid/`, `knn_clipping/compute.rs` |
| Cell construction | Stream certified nearest candidates and clip one independent cell per effective generator, emitting sharded cell/vertex records. | `cube_grid/query/`, `cube_grid/packed_knn/`, `knn_clipping/driver.rs`, `knn_clipping/cell_build/`, `knn_clipping/topo2d/` |
| Assembly | Resolve sharded vertex ownership, patch deferred cross-bin slots, assemble global arrays, and record shared-edge mismatches. | `live_dedup/` |
| Edge reconciliation | Apply bounded, transactional identity/cycle changes to recorded mismatches and identify regions that still require rebuilding. | `knn_clipping/edge_reconcile.rs` |
| Local rebuild and acceptance | Rebuild defect-bearing neighborhoods with Hull3d; commit a candidate only after the strict whole-diagram gate accepts it. | `knn_clipping/local_rebuild.rs`, `knn_clipping/local_hull.rs`, `knn_clipping/compute.rs` |
| Output resolution | Canonicalize exact zero-length edges in the final stored-f32 realization without silently deleting a cell, and report cell-killing components to the return-policy gate. | `knn_clipping/output_resolution.rs`, `knn_clipping/compute.rs` |
| Original-index remap and return | Expand effective cells back to original generator indices, attach the weld map, and construct the returned diagram/report. | `knn_clipping/compute.rs`, `diagram.rs` |

Validation and derived views are consumers of the returned representation, not hidden repair
stages. `validation.rs` reports subdivision facts; `cell_mesh.rs`, `adjacency.rs`, `delaunay.rs`,
`locate.rs`, and `measures.rs` expose explicit transformations or queries.

## The per-cell construction

A Voronoi cell is the intersection of half-spaces: one per neighbor, bounded by the perpendicular
bisector between the generator and that neighbor. So a cell can be built by starting from
"everything" and clipping against the bisectors of nearby points, nearest first.

On the sphere, bisectors are great circles. In a **gnomonic projection** centered at the
generator — the tangent plane at that point — every great circle maps to a straight line, so
spherical cell construction reduces to clipping a convex polygon by half-planes in 2D. The
projection is different for every generator, which is why most 2D quantities cannot be shared
between cells even when the underlying 3D bisector plane is shared.

The gnomonic polygon is the fast path, not the final numerical authority. If projection range,
polygon capacity, or a rounded `ClippedAway` decision prevents it from continuing, construction
replays the accepted bisectors into a cold spherical fallback. That fallback keeps normalized
bisector planes and polygon vertices in f64, resumes the same nearest-first candidate stream, and
rounds only the final emitted vertex positions to the public f32 storage type. If the replayed
constraints are genuinely infeasible (for example, indistinguishable generators), the original
failure is retained rather than fabricating a cell.

### The security radius

Clipping against every point would be O(n) per cell. It isn't necessary. After some clipping, let
`max_r` be the distance from the generator to its farthest current cell vertex. A neighbor at
distance `d` puts its bisector at distance `d/2`; if `d > 2 * max_r`, that bisector lies beyond
every vertex of the current polygon and cannot cut it — and neither can anything farther away. So
if candidates arrive nearest-first, the cell is provably complete the moment the next candidate's
distance crosses twice the farthest-vertex radius. This is the classical radius of security (Lévy
& Bonneel 2013; also used by voro++).

For uniform input the certificate fires after clipping ~6-7 neighbors per cell, independent of n.
That is why per-point cost is near constant.

Candidates come from a spatial index walked outward in rings, each ring carrying a bound on
everything not yet seen, so the termination test is one comparison per candidate and there is no
`k` to choose. A packed SIMD stage runs in front of the ring walk: distances are evaluated 8 wide
(f32 lanes via the `wide` crate, sorting networks for the small sorts), staged so the common case
— cell finished after the first chunk — touches the least memory. The clipping kernels are tuned
at the same grain: branchless 8-wide signed-distance masks for the small polygons (~6 edges) that
dominate.

Cell construction is embarrassingly parallel and runs on all cores via rayon.

## Stitching cells into one graph

Ray et al., *Meshless Voronoi on the GPU* (2018), is the source of the per-cell construction
above. They stop at independent cells — each thread holds its own polyhedron and nothing is
shared, which is the right contract for Lloyd loops and flux integrals. Most uses of a Voronoi
*diagram*, though, want a single graph: each vertex stored once, each edge knowing both its cells.
Producing that from independently-built cells is the main work this crate adds. The stitching
pipeline is geometry-light but intentionally spherical: it stores `Vec3` positions, measures
spherical chord distances during reconciliation, and has no boundary-edge policy. Its ownership,
deduplication, and assembly mechanisms remain separated from cell construction, but the crate does
not carry a generic position abstraction for a backend that is not present. Generality should be
reintroduced only with a second current backend and shared contract tests.

Three mechanisms make it work without a global lock:

**Combinatorial vertex identity.** A Voronoi vertex is where three cells meet, so it is keyed by
the sorted triple of generator indices whose bisectors define it — never by floating-point
position. Two cells that both decide vertex `[a, b, c]` exists agree on its identity exactly,
regardless of rounding. Deduplication is integer key matching, with one representative position
kept per key.

**Sharded live dedup.** Generators are partitioned into bins. Each bin builds its cells
sequentially while bins run in parallel. Every vertex key has a deterministic owner bin: local
keys deduplicate immediately in a bin-local table; foreign keys leave a deferred slot patched in a
sort-and-match pass after construction. No locks, no global map, no synchronization in the hot
loop.

**Directed build order with edge forwarding.** Fully independent construction discovers and clips
every shared edge twice, once per owning cell. Within a bin, cells build in a fixed order, which
makes a sequential optimization legal: for two same-bin cells sharing an edge, only the earlier
cell discovers the pair, clips it, and forwards a compact *edge check* (the edge key plus its
endpoint vertex identities) to the later cell, which replays it as a seed. The pair is processed
once, and the forwarded record coordinates the shared vertex indices. Across bins, where no
ordering is assumed, both cells clip the shared bisector independently and the assembly pass
matches the two sides afterward. A coverage contract — each ordered pair is supplied to each side
by exactly one mechanism — makes the hybrid sound.

A final **edge reconciliation** pass handles the residue: where two cells made *different*
combinatorial decisions about an epsilon-scale feature (one kept a sliver edge, the other
collapsed it — each evaluates predicates in its own chart), the disagreement is detected and
reconciled. Positional merges are transactional: the complete component, including aliases from
earlier rounds, must have f64-measured diameter no larger than the reconciliation epsilon. A
component that would grow through a chain past that bound is left untouched and explicitly seeds
a Hull3d local rebuild. The disputed feature is epsilon-scale, so both paths remain local.

The fast-path validity argument is the central design idea: combinatorial identity plus directed
edge checks certify exact multiplicity and opposite orientation without rebuilding and sorting a
second global edge list. Cross-bin equal-key runs enforce one record per side; in-bin checks enforce
one consumption per side and reverse endpoint order. Bounded reconciliation runs only on recorded
defects and rechecks complete edge agreement over its touched region. The already-required
incidence pass supplies `V` and the live half-edge count `H`, so `V - H/2 + F = 2` adds no second
topology traversal. Full connectivity and diagnostic validation remain optional/testing checks.

Geometric accuracy is best-effort (f64 internally, f32 storage); topological coherence only needs
adjacent cells' combinatorial decisions to agree. Keep that distinction explicit when touching any
of the mechanisms above.

Final generators and vertices are stored as `SpherePoint`: a private-field, 12-byte `[f32; 3]`
wrapper enforcing the common finite squared-norm envelope. Backend `glam::Vec3` allocations are
transferred directly into this layout after construction; public packed xyz views borrow the same
memory. Raw inputs remain unchecked until entry canonicalization.

## Module map

- `lib.rs` owns the public compute/configuration/report surface. `types.rs`, `diagram.rs`, and
  `error.rs` own checked point storage, diagram storage/views, and public errors respectively.
- `cell_layout.rs` pairs internal cell records with their shared index buffer and owns live-span
  access, including the stale-tail rule used after in-place reconciliation shrinkage.
- `cell_mesh.rs` owns explicitly simplified spherical cell meshes and provenance. `adjacency.rs`
  and `delaunay.rs` derive graph views; `measures.rs` and `spherical_arc.rs` own area, centroid,
  Lloyd, and owner-conditioned edge geometry.
- `embedding.rs` owns f64 world-coordinate projection and delegates unit-sphere geometry to the
  common backend. `locate.rs` owns checked point location. `validation.rs` owns strict
  subdivision checks.
- `fp.rs` is the scalar/SIMD numerical backend seam. `tolerances.rs` owns numerical slack;
  `policy.rs` owns construction, query, and performance policy. `packed_layout.rs` and
  `spatial_order.rs` own packed memory-layout and deterministic ordering helpers.
- `cube_grid/` owns the cube-map spatial index. `query/` implements certified shell traversal and
  resumable directed queries; `packed_knn/` implements batched packed selection.
- `knn_clipping/` owns end-to-end spherical construction. `compute.rs` orchestrates the backend;
  `driver.rs` builds bins; `cell_build/` runs one cell; `topo2d/` performs gnomonic clipping;
  `preprocess.rs` merges near-coincident inputs; `edge_reconcile.rs` reconciles assembled cycles;
  `local_rebuild.rs` and `local_hull.rs` own cold neighborhood rebuilding;
  `positive_simplification.rs` owns the explicit post-compute stored-chord fixed point, quotient
  certificates, work accounting, and Elide suppression provenance. It is not called by ordinary
  construction; and `output_resolution.rs` owns terminal stored-zero policy.
- `live_dedup/` owns spherical sharded vertex ownership, forwarded edge checks, deferred-slot
  patching, and global assembly. It is specialized to `Vec3`; a future second geometry backend
  must earn any shared abstraction with a current consumer and tests.
- `timing/` contains real and zero-sized timing backends. `quality.rs` is the `tools`-only quality
  surface; `point_audit.rs` is the `profiling`-only storage-envelope audit. `sort.rs` is the
  small-sort facade used by packed production selection and microbench code.
- `generated/sort_nets.rs` is generated by `scripts/gen_sort_nets.py`. Change the generator, then
  regenerate the checked-in file; do not edit the generated network body manually.
