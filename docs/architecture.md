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
Producing that from independently-built cells is the main work this crate adds, and the rest of
the pipeline is geometry-agnostic — it is generic over the vertex position type, with the sphere
contributing a thin driver (spatial index with certified distance bounds, bisector construction,
termination predicate).

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
repaired. Positional merges are transactional: the complete component, including aliases from
earlier rounds, must have f64-measured diameter no larger than the reconciliation epsilon. A
component that would grow through a chain past that bound is left untouched and explicitly seeds
Local3d repair. The disputed feature is epsilon-scale, so both paths remain local.

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

## Module map

- `lib.rs` — public API. `diagram.rs` — Voronoi diagram storage and views. `cell_mesh.rs` — dense
  explicitly simplified spherical cell meshes, provenance, and generic mesh validation.
- `types.rs` — `UnitVec3`, `UnitVec3Like`. `tolerances.rs` — numerical slack, with per-constant
  justification. `policy.rs` — performance heuristics (grid density, packed sizing, termination
  cadence), kept separate from tolerances.
- `embedding.rs` — f64 world-coordinate projection, `SphereEmbedding`, embedded diagram/report
  wrappers, and world-space point location; delegates all geometry to the unit backend.
- `live_dedup/` — the geometry-agnostic core: sharded vertex ownership, deferred-slot patching,
  edge-check propagation, assembly. Generic over the vertex position type.
- `knn_clipping/` — the spherical backend: per-bin `driver.rs`, single-cell `cell_build/`,
  gnomonic clipping in `topo2d/`, `preprocess.rs` (weld), `edge_reconcile.rs`, and cold-path
  topology repair in `escalate.rs` / `local_hull.rs`; `output_resolution.rs` owns terminal
  exact-zero canonicalization and the explicit cell-elision quotient.
- `cube_grid/` — cube-map spatial index and packed-kNN stage: dot-product distance with
  conservative cap/plane upper bounds, ring walk with per-ring certificates.
- `locate.rs` — point location, reusing the grid shell frontiers to answer nearest-generator
  queries. `validation.rs` — strict subdivision checks.
- `timing/` — optional instrumentation.
