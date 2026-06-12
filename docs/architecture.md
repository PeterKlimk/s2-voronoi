# Architecture / algorithm notes

s2-voronoi computes Voronoi diagrams on the unit sphere (S2) and, with the same engine, on a
bounded planar rectangle. The implementation is optimized for building many independent cells
efficiently; the sharded dedup/stitching core is geometry-agnostic and each geometry contributes
a driver (spatial index + cell builder).

## High-level pipeline

For each generator point `g`:

1. Find candidate neighbors via a cube-map spatial index (`cube_grid/`).
2. For each neighbor `n`, form a bisector great-circle plane (a half-space constraint).
3. Clip an initially-unbounded polygon in a **gnomonic projection** (local tangent plane) to
   produce the cell boundary in 2D (`knn_clipping/topo2d/`).
4. Extract spherical vertices (unit vectors) from the final 2D polygon and deduplicate them across
   cells (“live dedup”) to build the global vertex list and per-cell index lists
   (`src/live_dedup/`, shared with the planar backend).

Cells are computed independently; “global” work is limited to dedup/overflow handling and narrow
post-pass edge reconciliation.

## Why gnomonic projection?

The cell boundary is an intersection of spherical half-spaces (great-circle constraints). In a
gnomonic projection centered at `g`, every great circle becomes a line, so the clipping problem
reduces to repeated convex polygon clipping by half-planes in 2D.

This projection is **different per generator**, which is why most 2D geometric quantities cannot
be shared between cells, even when an underlying 3D plane is shared.

The current `topo2d` builder now has an internal wrapper seam around the gnomonic implementation
so future non-hemispheric handling can be introduced as an alternate builder rather than as
special-case logic in `cell_build/`. See `docs/fallback_builder.md`.

## The planar backend

`compute_plane` runs the identical pipeline shape over a flat `res x res` grid (`plane_grid/`):

1. The user rect is normalized with a **uniform** scale (Voronoi structure is not invariant under
   anisotropic scaling); the longer side maps to 1.
2. Generators within the planar weld radius are welded using the grid itself as the detector
   (points can only weld within a cell or across a radius-thin wall band) — the no-weld case is a
   read-only scan of the grid that the kNN queries then reuse.
3. Candidate neighbors come from the same staged stream as the sphere: a packed SIMD stage
   (per-cell query groups, 8-wide squared distances, Chunk0/Tail/ExpandR2 with lazy cold paths)
   followed by a shell-expansion takeover that re-covers everything (the consumer dedups).
4. Cells clip in the plane directly — **no projection layer at all**; the gnomonic chart's job is
   already done by geometry. The polygon is seeded with the four rect walls as half-planes owned
   by *virtual wall generators* (ids `n..n+4`), so every cell is bounded from the seed, boundary
   vertices are ordinary `[gen, gen, wall]` key triples, and dedup/validation work unchanged.
5. Termination is a single comparison: a neighbor at squared distance `d2` cannot cut the cell
   once `d2 > 4 * max_r2` (its bisector passes beyond every vertex). All grid certificates are
   exact box distances (with a documented wall-classification slack), where the sphere needs
   conservative cap/plane bounds.

Distance semantics are inverted relative to the sphere throughout the planar stack: squared
Euclidean distance with *lower*-bound certificates ("nothing unseen is closer than b"), versus
the sphere's dot products with upper bounds. The directed-eligibility rules, packed slot layout,
emission seam, assembly, and edge reconciliation are shared, not duplicated.

### Periodic (toroidal) domains

`compute_plane_periodic` runs the planar pipeline with three substitutions: the grid's Chebyshev
rings wrap modulo the resolution (visited-stamped, since rings self-collide once they span the
grid), all distances are minimum-image, and the cell builder seeds unbounded (no walls) with
bisectors to each neighbor's nearest image — `wrap_half` is bit-exactly antisymmetric, so the
two cells of a shared edge construct the identical line. The half-period guard
(`max_r2 < (min_period/4)^2` at extraction) makes nearest-image clipping exact, makes the
canonical-wrap + unwrap storage convention well-defined, and keeps the image-agnostic vertex
keys sound (it excludes the torus's multi-circumcenter hazard). Validation checks the torus
Euler relation (`V - E + F = 0`) with every edge paired.

## Module map

- `src/lib.rs`: public API (`compute`, `compute_with`, `validation`).
- `src/diagram.rs`: storage (`SphericalVoronoi`, `CellView`).
- `src/policy.rs`: internal packed/termination policy and heuristic decisions.
- `src/types.rs`: `UnitVec3`, `UnitVec3Like`.
- `src/live_dedup/`: the geometry-agnostic engine — sharded vertex ownership, deferred-slot
  patching, edge-check propagation, assembly, bin layout, and the per-cell emission seam. Generic
  over the vertex position type (`Vec3` on the sphere, `Vec2` on the plane).
- `src/knn_clipping/`: the spherical backend.
  - `driver.rs`: per-bin parallel cell-build driver (the planar sibling is
    `plane_clipping/driver.rs`).
  - `cell_build/`: single-cell neighbor seeding, stream consumption, clipping, and extraction.
  - `topo2d/`: gnomonic projection, half-planes, and convex clipping (the 2D clip cores are
    shared with the plane).
  - `edge_reconcile.rs`: narrow post-pass reconciliation for unresolved shared-edge mismatches
    (shared with the plane; each geometry passes its own degenerate-length epsilon).
  - `preprocess.rs`: weld near-coincident generators (see docs/correctness-contract.md).
- `src/plane_clipping/`: the planar backend — domain normalization + grid-integrated weld
  (`compute.rs`), per-bin driver (`driver.rs`), rect-seeded cell builder (`builder.rs`).
- `src/plane_diagram.rs`: `PlanarVoronoi`, `PlanePoint`, `PlaneRect`.
- `src/cube_grid/`: cube-map spatial index + packed-kNN stage (sphere).
- `src/plane_grid/`: flat spatial index + packed-kNN stage + shell frontier (plane).
- `src/timing/`: optional timing + histograms (crate-wide).
- `src/convex_hull.rs` (`qhull` feature): convex-hull dual backend (tests/bench comparisons).

## Policy layer

The current heuristic and tuning decisions are intentionally centralized in `src/policy.rs`.

That layer currently owns:

- query-grid density and occupancy-feedback resolution
- packed chunk sizing
- packed `r=2` enablement
- packed count-model constants

Numerical tolerances live separately in `src/tolerances.rs` (empirical values with per-constant
justification); policy is for performance heuristics, tolerances for correctness slack.

If a future change adds dynamic heuristic activation, it should be expressed through this policy
layer and the neighbor-source boundary, not by reintroducing local fallback logic into
single-cell build orchestration.

See `docs/policy.md`.

## Fallback builder seam

`src/knn_clipping/topo2d/builder.rs` owns the builder-implementation seam.

Today:

- `Topo2DBuilder` dispatches to `GnomonicBuilder`
- projection-limit failure is classified as a fallback handoff request internally
- `cell_build/` only sees builder outcomes, not gnomonic projection details

That seam is intentionally small so a future non-hemispheric fallback builder can slot in without
re-entangling clip representation, stream consumption, and terminal-failure classification.

See `docs/fallback_builder.md`.

## Consistency model: why independently-built cells form one valid graph

Cells are built in parallel with no shared geometric state, yet the output must be a single
consistent subdivision. Three mechanisms carry that burden:

1. **Combinatorial vertex identity.** A Voronoi vertex is identified by the sorted triple of
   generator indices whose bisectors meet there — not by its floating-point position. Two cells
   that both decide vertex `[a,b,c]` exists agree on its identity exactly, regardless of rounding.
   Live dedup canonicalizes one representative position per key.
2. **Directed neighbor ordering.** The bin/local eligibility ordering ensures each shared edge is
   discovered in a coordinated way across the two owning cells and lets edge-check records
   propagate vertex indices from earlier-built to later-built cells (see `docs/live_dedup.md`).
3. **Reconciliation as a bounded safety net.** Where the two owning cells make *different*
   combinatorial decisions (one keeps an epsilon-scale edge, the other collapses it — possible
   because each cell evaluates clip predicates in its own gnomonic chart), edge checks and
   `edge_reconcile` repair the disagreement after assembly.

The important property: geometric *accuracy* is best-effort (f64 internally), but graph *validity*
only requires the combinatorial decisions of adjacent cells to agree. Today that agreement is
empirical (per-chart epsilon decisions agree except in adversarial regimes — see the seam finding
in `docs/engineering-findings.md` and the margin data in `docs/correctness-contract.md`). The
long-term plan (`docs/todo.md` P5) is to make shared decisions canonical — evaluated once in a
frame chosen by sorted generator index — so agreement holds by construction and reconciliation
shrinks to a true edge case.

The full invariant — order, coverage contract, the epsilon caveat — is written out in
`docs/live_dedup.md` ("The stitching invariant").

This argument — that the directed ordering plus combinatorial identity plus bounded repair yields
a strictly valid subdivision — is the crate's central design idea and should be kept explicit when
modifying any of the three mechanisms.

## Future geometries

The pipeline separates into geometry-specific layers and a geometry-agnostic core:

- geometry-specific: `cube_grid/` (spatial index with conservative cell bounds), `topo2d/`
  (chart, bisector construction, termination certificate)
- geometry-agnostic: directed stitching order, sharded live dedup, edge reconciliation, assembly

A planar (or other-domain) backend would swap the first group: a flat grid index, direct 2D
clipping (no chart at all — the hemisphere/projection-validity machinery is sphere-specific), and
a Euclidean distance bound. The plane adds one problem the sphere never has: unbounded cells need
a boundary policy. Keep geometry-specific code behind these two module boundaries; extract a
`Geometry` trait only when a second geometry is actually being built.

## Glossary

- **Generator**: input point that owns one Voronoi cell.
- **Neighbor**: candidate generator used to constrain a cell during clipping.
- **Half-plane / half-space**: bisector constraint induced by neighbor `n`.
- **Gnomonic projection**: maps great circles to lines in a local tangent plane at `g`.
- **Bin / shard**: a partition of generators used for parallel cell construction and localized
  communication.
- **Edge check**: a compact record used to propagate per-edge vertex indices between adjacent
  cells (see `docs/live_dedup.md`).
