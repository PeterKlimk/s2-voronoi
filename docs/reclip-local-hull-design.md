# Tier-2 Re-clip — Local Hull Resolver (design)

Status (2026-06-20): **design agreed, implementation starting.** Replaces the C
resolver's fragile assembly (boundary-recovery + "exactly-2-endpoints" rule +
recursive `Expand`) — which fails by runaway `Expand` on dense mega — with a
**local constrained convex hull / Delaunay** whose dual yields the contested
cells directly. Supersedes roadmap item 2 in `reclip-repair-design.md`.

## Why (measured)

The exact-predicate swap was prototyped and **refuted by measurement** (mega
200k/500k, single-thread A/B): exact `in_circle==0` ties never fire on mega, the
exact predicate *regressed* recovery (jitter was helping the assembly), and the
real bail is **runaway `Expand`** — boundary recovery leaves degree-1 endpoints,
so the component grows 8→15→26→39→49 cells until the budget trips. Three reviews
agree: *the resolver has no single local topology.* The fix is to build that one
topology. On the unit sphere, **3D convex hull = Delaunay** (hull faces are
Delaunay triangles; a face's outward direction is its Voronoi vertex), so the
local hull's dual is the local Voronoi — consistent by construction, no
boundary-recovery, no `Expand`.

## The local point set (Codex-checked)

`S = C ∪ secured(C)`, where:

- **`C`** = the contested connected component (from `identify_components`).
  Re-meshed.
- **`secured(g)`** for each `g ∈ C` = the neighbors from a **fresh kNN stream**
  through the *bounded-cell certificate* (stop when `g`'s farthest current
  cell-vertex is provably closer to `g` than any not-yet-seen generator — the
  same certificate the main clipper uses; once unseen candidates are beyond
  `2·max_r` none can cut the cell). `secured(g)` is **not** stored in build
  artifacts — it needs a fresh stream.
- **`R = S \ C`** = the securing ring. **Constraints only** — its generators
  participate in the hull so the boundary faces form, but `R`'s own cells are
  NOT re-emitted. This is what kills the `Expand` recursion: you never have to
  bound `R`'s cells.

**Sufficiency (proved):** you do *not* need `a`/`b`'s certificates for a face
`(g,a,b)` with `g ∈ C` — any outside generator that could invalidate that face
is closer at its Voronoi vertex, so it would cut `g`'s cell and violate `g`'s
certificate. Per-`g` certificates cover every emitted face, including
`(contested, ring, ring)` boundary vertices. This set is *tight* — the actual
bounding neighborhood (tens–low-hundreds), **not** the old ~9k grid filter.

## Algorithm

1. `C` from the residual; for each `g ∈ C` stream `secured(g)` (budgeted; revert
   to residual on overflow — the fallback builder never early-terminates, so a
   hard per-cell/per-component cap is required).
2. Build the incremental 3D convex hull of `{points[s] : s ∈ S}` with exact
   `robust::orient3d` visibility (no jitter). Small `n` → naive `O(n²)` insertion
   with directed-edge horizon; no point-location structure.
3. **Emit only `C`'s cells** (the fan of hull faces around each `g ∈ C`,
   circumcenter per face = Voronoi vertex). Ignore pure-`R` faces (sound: every
   emitted face touches a certified `g`).
4. **Degenerate high-degree faces** (an exact-cocircular set = a coplanar quad/
   n-gon hull face, `orient3d==0`) get *explicit* handling — deterministically
   triangulate (id-order) or keep as one high-degree Voronoi vertex (the "merge"
   option). Not arbitrary hull triangulation.
5. **Boundary stitch:** pin each `C↔R` boundary vertex to the existing ring cell
   vid by key. Sound, with the existing validate-or-revert gate as backstop.
6. **validate-or-revert** gate (`verify_sphere_effective_strict`) as the final
   check — revert the whole repair on any strict-invalidity.

## The one subtlety — boundary fidelity (Codex item E)

"Non-contested" means *topologically agreed*, not *geometrically correct*: a ring
cell `r` and old-`g` can agree on the same slightly-wrong boundary vertex
(consistently wrong → no residual). Pinning by key then preserves that original,
tolerance-level error. This is **sound** under our bar: pin-by-key yields a
topologically-valid diagram whose seam is no worse than the original cells
(within "essentially Voronoi"); a genuine key/topology disagreement (firewall
gap) fails to pair → the gate reverts loudly. **Never silent-invalid.** Note B's
boundary handling is the *same* as C's (both pin by key); B's win is entirely the
**interior** (the hull replaces the assembly that was the bail root), so E is
orthogonal to what B fixes.

**Optional fidelity upgrade (deferred):** geometric-check the pin (key *and*
position/equidistance); on mismatch, promote `r` into `C` with its own
certificate (a *controlled, geometrically-justified* expansion that terminates at
the true geometric firewall, bounded by the cap) or revert. Adds seam accuracy;
only needed if revert-rate or fidelity demands it. Start with pin-by-key.

## Build order

1. **Incremental hull core** (`local_hull.rs`): `build(points) -> faces`,
   `face_circumcenter`, per-point ordered face fan (the dual). Self-contained,
   tested against tetra/octa/cube (exact face counts; closed surface).
2. **`secured(g)` stream**: fresh kNN + bounded-cell certificate, budgeted.
3. **Wire into `repair`**: build `S`, hull, emit `C` cells, pin boundary, gate.
   Behind `S2_RECLIP_REPAIR`, validate-or-revert keeps every step sound.
4. Fixtures: exact degree-4/5 cliques (incl. boundary-pinned); mega recovery
   sweep before/after.

## Open items / risks
- `identify_components` can over-merge `C` (not proven Delaunay-connected) — safe
  given per-`g` certificates, but can bail a bit eagerly; revisit if it hurts.
- Hard `secured`/hull-size budget + revert is mandatory (no unbounded streams).
- Degenerate-face policy (merge vs split) — default deterministic split; merge is
  the faithful option, rare on mega (measured `in_circle==0` = 0).
