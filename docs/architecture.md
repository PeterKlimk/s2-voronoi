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

## Module map

- `src/lib.rs`: public API (`compute`, `compute_with`, `validation`).
- `src/diagram.rs`: storage (`SphericalVoronoi`, `CellView`).
- `src/policy.rs`: internal packed/termination policy and heuristic decisions.
- `src/types.rs`: `UnitVec3`, `UnitVec3Like`.
- `src/knn_clipping/`: kNN + clipping backend.
  - `cell_build/`: single-cell neighbor seeding, stream consumption, clipping, and extraction.
  - `topo2d/`: gnomonic projection, half-planes, and convex clipping.
  - `live_dedup/`: sharded vertex ownership, deferred-slot patching, and edge-check propagation.
  - `edge_reconcile.rs`: narrow post-pass reconciliation for unresolved shared-edge mismatches.
  - `preprocess.rs`: merge near-coincident points.
  - `timing.rs`: optional timing + histograms.
- `src/cube_grid/`: cube-map spatial index + packed-kNN helpers.
- `src/convex_hull.rs` (`qhull` feature): convex-hull dual backend (tests/bench comparisons).

## Policy layer

The current heuristic and tuning decisions are intentionally centralized in `src/policy.rs`.

That layer currently owns:

- packed chunk sizing
- packed `r=2` enablement
- termination cadence
- packed count-model constants

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
