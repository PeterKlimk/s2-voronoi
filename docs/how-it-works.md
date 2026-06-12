# How it works

This is the reader-oriented tour of the algorithm: why the crate is fast, why the output is a
single consistent graph even though every cell is built independently, and which ideas come from
prior work versus this crate. The maintainer-oriented material (module maps, policy seams,
invariant fine print) lives in [architecture.md](architecture.md) and
[live_dedup.md](live_dedup.md).

## The standard approach, and why it doesn't scale

Most spherical Voronoi implementations (scipy, anything qhull-based) compute the 3D convex hull
of the input points and read the diagram off its dual. The hull is a global structure: it is
hard to parallelize, and hull algorithms are O(n log n) with significant constants. Past tens of
thousands of points this dominates everything; at millions of points the gap to this crate is
roughly an order of magnitude and still widening with n. Planar crates built on Delaunay
triangulation (delaunator and its descendants) share the shape of the problem: a global
triangulation first, the diagram second.

## Build every cell independently

This crate never builds a hull or a triangulation. It follows the *meshless* construction of
[Ray, Sokolov, Lefebvre & Lévy, "Meshless Voronoi on the GPU", ACM TOG 37(6),
2018](https://doi.org/10.1145/3272127.3275092): each generator computes its own cell directly,
in parallel, from nothing but its nearby neighbors.

A Voronoi cell is the intersection of half-spaces — one per neighbor, bounded by the
perpendicular bisector between the generator and that neighbor. So a cell can be built by
starting from "everything" and clipping by bisectors of candidate neighbors, nearest first. On
the sphere the bisectors are great circles; in a **gnomonic projection** (the tangent plane at
the generator) every great circle becomes a straight line, so spherical cell construction
reduces to 2D convex polygon clipping. In the plane no projection is needed at all.

The catch is the word *candidate*: correctness seems to require clipping against everyone, which
would be O(n) per cell. It isn't, because of the next idea.

## The security radius: how a cell knows it is finished

The termination criterion is the classical **"radius of security"** (Ray et al. credit it to
[Lévy & Bonneel 2013](https://doi.org/10.1007/978-3-642-33573-0_21); voro++ uses the same idea):
after some clipping, let `max_r` be the distance from the generator to its farthest current cell
vertex. A neighbor at distance `d` has its bisector at distance `d/2`. If `d > 2 * max_r`, the
bisector passes beyond every vertex of the current polygon — it cannot cut anything, and neither
can any point farther away. So if candidates arrive in increasing distance order, the cell is
**provably complete** the moment the next-candidate distance crosses twice the farthest-vertex
radius. No global structure ever certified anything; the cell carries its own proof.

For uniformly distributed points the certificate fires after clipping ~6-7 neighbors per cell
(measured: 6.4 mean at 500k, a few dozen candidates examined, and only ~30 cells in 500,000
needed more than the first packed stage) — whether the input has ten thousand points or ten
million. That is the entire reason per-point cost is near constant, and it is why the gap to
hull/triangulation approaches *grows* with n.

What this crate builds around the certificate:

- **A streamed, certified candidate source instead of fixed-k kNN.** Ray et al. precompute a
  k-nearest-neighbors array per point and report failure when k candidates weren't enough. Here
  the spatial index — a cube-map grid on the sphere, a flat grid in the plane — is walked
  outward in rings from the generator, and each ring carries a bound on everything not yet seen
  (conservative cap bounds on the sphere, exact box distances in the plane). The termination
  test is a single comparison per candidate, there is no k to choose, and a cell that needs more
  neighbors simply keeps streaming instead of failing and retrying.
- **A packed SIMD stage in front of the ring walk.** Candidate distances are evaluated 8 wide
  (f32 lanes, explicit SIMD via `wide`, sorting networks for the small sorts), staged so the
  common case — cell done after the first chunk — touches the minimum of memory. The ring
  expansion takes over only for the cells that outlive the packed budget.
- **Half-space clipping kernels** tuned at the same grain: branchless 8-wide signed-distance
  masks with bit-twiddled entry/exit transitions for the small polygons that dominate (a typical
  cell has ~6 edges).

Cell construction is embarrassingly parallel and runs on all cores via rayon. The design target
is 2.5M spherical points in under 500ms on a 6-core desktop CPU; see
[performance.md](performance.md) for measured numbers and the benchmarking discipline behind
them.

## The part the GPU paper doesn't do: one consistent graph

Ray et al. stop at independent cells — each thread holds its own polyhedron, integrals are
computed on the fly, nothing is shared (voro++ has the same output shape). They are explicit
that this is the point: their applications "do not make use of the global combinatorics," so
"it is no longer necessary to ensure that geometric predicates are globally coherent." That is
the right contract for Lloyd loops and flux integrals, but most downstream uses of a Voronoi
*diagram* — adjacency queries, topology validation, rendering with shared vertex buffers,
serialization — want a single graph: each vertex stored once, each edge knowing both of its
cells. Which means taking on exactly the burden the meshless construction was designed to shed:
making the combinatorial decisions of independently-built cells globally coherent.

Stitching independently-built cells into one graph is this crate's main piece of engineering on
top of the paper. Naively it is miserable: adjacent cells compute the "same" vertex through
different projections and different rounding, so positions never match exactly, and a global
concurrent hash map in the hot loop destroys the parallelism you just won. Three mechanisms
carry the burden instead:

1. **Vertex identity is combinatorial, not geometric.** A Voronoi vertex is where three cells
   meet, so it is keyed by the sorted triple of generator indices whose bisectors define it —
   never by floating-point position. Two cells that both decide vertex `[a, b, c]` exists agree
   on its identity *exactly*, regardless of rounding. Deduplication becomes integer key
   matching; one representative position is kept per key.

2. **Sharded live dedup.** Generators are partitioned into bins; each bin builds its cells
   sequentially while bins run in parallel. Each vertex key has a deterministic owner bin: local
   keys deduplicate immediately in bin-local tables, foreign keys leave a deferred slot that is
   patched in a sort-and-match pass after construction. No locks, no global map, no
   synchronization inside the hot loop.

3. **A directed build order with edge forwarding — two regimes on one diagram.** Fully
   independent construction pays a hidden tax: every shared edge is discovered and clipped
   *twice*, once by each owning cell. A sequential algorithm wouldn't pay it — having built
   cell g, it already knows the g–h edge when it reaches h. This crate runs both regimes at
   once. Within a bin, cells build sequentially in a fixed order, which makes a sequential-only
   optimization legal: for two same-bin cells sharing an edge, only the earlier cell discovers
   the pair from its candidate stream; it clips, then forwards a compact *edge check* (the edge
   key plus its endpoint vertex identities) to the later cell, which replays the constraint as
   a seed before consuming its own stream — the pair is processed once, not twice, and the
   forwarded record is exactly what coordinates the shared vertex indices. Across bins, where
   no ordering can be assumed (bins run in parallel), both cells clip the shared bisector
   independently, GPU-paper style, and the assembly pass matches the two sides up afterwards.
   The *eligibility rules* — which candidates a cell's stream emits versus treats as
   transit-only — are what switch between the regimes per pair, and the coverage contract (each
   ordered pair is supplied to each side by exactly one mechanism) is what makes the hybrid
   sound. The full argument, including why the earlier side always delivers and the
   epsilon-scale caveat where it can't, is written out in [live_dedup.md](live_dedup.md), "The
   stitching invariant".

A narrow **edge reconciliation** pass runs last: where two cells made *different* combinatorial
decisions about an epsilon-scale feature (one kept a sliver edge, the other collapsed it —
possible because each evaluates predicates in its own chart), the disagreement is detected and
repaired. The disputed feature is itself epsilon-scale, so the repair is local and bounded.

## Degenerate inputs, and what is actually promised

Coincident and nearly-coincident generators would otherwise manufacture contradictions faster
than reconciliation can repair them, so inputs are **welded** first: generators within a fixed
radius (~1.4e-6 chord on the sphere, ~1e-6 of the longer side in the plane — both derived from
f32 rounding with measured margins) share one cell, exposed via `weld_map()`. In the plane the
weld detector *is* the production spatial grid: the no-weld case (almost always) is a read-only
scan of a structure the pipeline needed anyway.

The output contract is *essentially Voronoi*: a **hard topological guarantee** — the returned
graph is a strictly valid subdivision of the sphere / rectangle / torus, checkable via the
`validation` module and fuzz-tested at multi-million point counts — and a **soft geometric
guarantee** — positions accurate to floating-point working precision, features at the resolution
floor handled by policy. No floating-point implementation can promise mathematical exactness;
this one promises that the graph it returns is never nonsense, and tells you precisely what was
welded or repaired. The full statement, with the probe data behind the constants, is in
[correctness-contract.md](correctness-contract.md).

## One engine, three geometries

Everything after clipping — sharded dedup, edge checks, assembly, reconciliation, validation —
is geometry-agnostic (generic over the vertex position type). Each geometry contributes a thin
driver: a spatial index with certified bounds, a bisector construction, and a termination
predicate.

- **Sphere**: cube-map grid, dot-product distance semantics with conservative upper bounds,
  gnomonic charts for clipping.
- **Bounded rectangle**: flat grid, exact squared-distance bounds, no chart. The rect walls
  enter as half-planes owned by four *virtual generators*, so boundary vertices are ordinary
  three-generator keys and the dedup/validation machinery never learns the domain has a
  boundary.
- **Rectangular torus** (periodic): same flat grid with wrapping rings and minimum-image
  distances; cells clip against each neighbor's nearest image. A **half-period guard** — every
  cell must be provably smaller than a quarter period — makes nearest-image clipping exact,
  keeps the wrapped vertex storage well-defined, and keeps the combinatorial vertex keys sound
  on a torus (where, without it, three generators can define more than one circumcenter).
  Underpopulated domains fail loudly rather than wrap wrongly.

## Prior art, honestly

The meshless per-cell construction is from [Ray et al.
2018](https://doi.org/10.1145/3272127.3275092) (their setting is volumetric 3D on the GPU; this
crate is, deliberately, a CPU descendant of the idea for the sphere and the plane). The
security-radius criterion predates it — Ray et al. credit Lévy & Bonneel 2013, and voro++
(Rycroft 2009) uses the same bound. The stitching layer — combinatorial vertex identity, the
directed order with edge forwarding, sharded dedup with deferred patching, bounded
reconciliation, and the validated single-graph output contract — was developed independently
for this crate. Within it, the part we believe is most likely to be genuinely novel is the
**hybrid build regime**: exploiting sequential construction order *within* a shard to do
work-sharing that fully-parallel formulations cannot express (each same-shard edge clipped
once and forwarded, rather than discovered twice), while degrading gracefully to independent
both-sides construction *across* shards — with one coverage contract proving the two regimes
compose into a single coherent diagram. Other ingredients echo known techniques (keying
vertices by generator triples is folklore in exact-geometry circles; sharding with deferred
cross-shard patching is a classic parallel pattern). We claim no more than that; the value is
that the combination is implemented, tested, and fast.
