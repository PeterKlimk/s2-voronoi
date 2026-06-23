# Architecture

s2-voronoi computes Voronoi diagrams on the unit sphere and, with the same engine, on a bounded
rectangle and a rectangular torus. The design builds every cell independently and in parallel,
then stitches the cells into one shared graph. This document describes how that works and where
the code lives.

## The per-cell construction

A Voronoi cell is the intersection of half-spaces: one per neighbor, bounded by the perpendicular
bisector between the generator and that neighbor. So a cell can be built by starting from
"everything" and clipping against the bisectors of nearby points, nearest first.

On the sphere, bisectors are great circles. In a **gnomonic projection** centered at the
generator — the tangent plane at that point — every great circle maps to a straight line, so
spherical cell construction reduces to clipping a convex polygon by half-planes in 2D. The
projection is different for every generator, which is why most 2D quantities cannot be shared
between cells even when the underlying 3D bisector plane is shared. In the plane no projection is
needed; clipping happens directly.

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
the pipeline is geometry-agnostic — it is generic over the vertex position type and shared by all
three geometries.

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
repaired. The disputed feature is epsilon-scale, so the repair is local.

The validity argument is the central design idea: combinatorial identity plus the directed order
plus bounded repair yields a strictly valid subdivision. Geometric accuracy is best-effort (f64
internally); validity only needs adjacent cells' combinatorial decisions to agree. Keep that
explicit when touching any of the three mechanisms.

## The three geometries

Everything after clipping is geometry-agnostic. Each geometry contributes a thin driver: a spatial
index with certified distance bounds, a bisector construction, and a termination predicate.

**Sphere.** Cube-map grid; dot-product distance with conservative cap/plane upper bounds; gnomonic
charts for clipping.

**Bounded rectangle.** Flat grid; exact squared-distance bounds; no chart. The user rect is
normalized with a uniform scale (Voronoi structure is not invariant under anisotropic scaling).
The four walls enter as half-planes owned by virtual wall generators, so every cell is bounded
from the seed, boundary vertices are ordinary three-generator keys, and the dedup/validation
machinery never learns the domain has a boundary. Distance semantics are inverted relative to the
sphere — squared Euclidean with lower-bound certificates rather than dot products with upper
bounds — but the eligibility rules, packed layout, assembly, and reconciliation are shared.

**Rectangular torus.** Same flat grid with rings that wrap modulo the resolution, minimum-image
distances, and bisectors to each neighbor's nearest image (the wrap is bit-exactly antisymmetric,
so both cells of a shared edge build the identical line). A half-period guard — every cell must be
provably smaller than a quarter period — makes nearest-image clipping exact and keeps the vertex
keys sound (without it three generators can define more than one circumcenter on a torus).
Underpopulated domains fail loudly rather than wrap wrongly.

## Module map

- `lib.rs` — public API. `diagram.rs`, `plane_diagram.rs` — diagram storage and views.
- `types.rs` — `UnitVec3`, `UnitVec3Like`. `tolerances.rs` — numerical slack, with per-constant
  justification. `policy.rs` — performance heuristics (grid density, packed sizing, termination
  cadence), kept separate from tolerances.
- `live_dedup/` — the geometry-agnostic core: sharded vertex ownership, deferred-slot patching,
  edge-check propagation, assembly. Generic over the vertex position type.
- `knn_clipping/` — the spherical backend: per-bin `driver.rs`, single-cell `cell_build/`,
  gnomonic clipping in `topo2d/` (the 2D clip cores are shared with the plane), `preprocess.rs`
  (weld), `edge_reconcile.rs` (shared), and cold-path topology repair in `escalate.rs` /
  `local_hull.rs`.
- `plane_clipping/` — the planar backend: domain normalization and grid-integrated weld, per-bin
  driver, rect-seeded builder, and the periodic variants.
- `cube_grid/` — cube-map spatial index and packed-kNN stage (sphere). `plane_grid/` — flat
  index, packed-kNN stage, and shell frontier (plane).
- `locate.rs` — point location, reusing the grid shell frontiers to answer nearest-generator
  queries. `validation.rs` — strict subdivision checks for all three topologies.
- `timing/` — optional instrumentation. `convex_hull.rs` — qhull dual backend, for comparison
  tests only.
