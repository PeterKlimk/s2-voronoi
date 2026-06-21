# Tier-2 Re-clip Repair — State of Affairs (2026-06-21)

Branch: `agent/fallback-incremental-clip`. A running record of where the Tier-2
"repair the repair" work stands, what was tried, what was measured, and the open
decision — so the next session (or reader) doesn't re-derive the dead ends.

Companion docs: `reclip-repair-design.md` (the original C resolver),
`reclip-local-hull-design.md` (the B / local-hull design + its measured
boundary-stitch failure), `optimization-ideas.md` (the structured-degeneracy
note).

## TL;DR

- **The correctness bar is already met.** `compute` is *valid-or-clean-error*:
  the strict validator gates every repair, and any unrepaired residual fires the
  loud `residual_error`. Never silent-invalid. This holds with the existing
  **jitter** resolver, which recovers **4/6** of the mega A/B seeds.
- The remaining work is purely **recovery rate** on the `mega` distribution (a
  rare, opt-in `S2_RECLIP_REPAIR` path) — turning "clean error" into "valid
  graph" more often. It is *not* a correctness gap.
- A measurement campaign (below) **killed the cheap shortcuts**: pin-by-key bails
  on all mega, and "just collapse the strandings" cannot beat jitter either.
  Beating 4/6 requires the full overlay+collapse arrangement engine, whose payoff
  the diagnostic **could not confirm** (the natural diagnostic is confounded by
  the very free-hull distortion it tries to measure).
- **Open decision** (see end): stop at jitter+gate and document the limit
  (recommended), build the arrangement engine anyway, or run one more
  de-confounding measurement first.

## What "the repair" is

`mega` = a huge fraction of points packed into a tiny spherical cap → many
**near-cocircular** generators → high-degree degenerate Voronoi vertices that
per-cell clipping resolves inconsistently across cells. Pipeline of defenses:

1. **Tier-1 reconcile** (`edge_reconcile`): eps-proximity merge / 1-1
   force-merge / collinear-drop. Handles most degeneracies.
2. **Tier-2 re-clip** (`reclip_repair.rs`, opt-in `S2_RECLIP_REPAIR`): re-resolve
   the *contested components* Tier-1 couldn't pair, then stitch back to the
   surrounding valid ("ring") cells.
3. **Loud backstop**: any surviving unpaired interior edge → `residual_error`.

A *contested component* `C` is a connected cluster of cells that disagree about a
degenerate vertex. The hard part has always been **stitching a re-resolved `C`
back to the existing diagram across a near-degenerate seam.**

## What has LANDED (committed, on this branch)

- **Gate = full validator** (`1373a7b`): the repair's accept test is
  `verify_sphere_effective_strict` (edge-pairing + degree≥3 + antipodal +
  off-sphere + duplicate-cell + Euler), run over the re-stitched region, with
  byte-identical revert on failure. Closed a **silent-invalid** hole where the
  old re-detect gate was a strict *subset* of the validator (`S2_RECLIP_REPAIR`
  on + `S2_VORONOI_VERIFY` off could ship invalid topology). This is what makes
  the system valid-or-error regardless of recovery rate.
- **Rollback fixes** (`edbfa46`): first-snapshot-only revert map (duplicate-touch
  hazard), deterministic interior-vid order, OOB guards.
- **Fuzzing** (`92113fa`): mega + repair-contract tests (`tests/reclip_repair.rs`,
  `tests/high_degree.rs`, robustness campaign mega/grid cases). 0 silent-invalid
  observed; the strategy is **fuzz-with-validation in tests**, not always-on
  runtime validation (which would defeat the "blazing fast" goal).
- **Incremental hull core** (`dd5d39f`, `src/knn_clipping/local_hull.rs`): a
  self-contained, tested 3D convex hull (= spherical Delaunay) — exact
  `robust::orient3d` visibility, directed-edge horizon, `build` / `faces` /
  `face_circumcenter` / `cell_faces` (the dual). Tests: tetra=4, octa=8 (degree-4
  dual cells), cube=12, random closed/outward/equidistant, coplanar→None.
  **Reusable and unaffected by the integration outcome.** Currently
  `#[allow(dead_code)]` until a consumer lands.

## What was TRIED and REVERTED (with the reason)

### 1. Exact-predicate resolver — REFUTED by measurement
Hypothesis: an exact in-circle predicate would fix mega. Prototyped, A/B'd: exact
`in_circle==0` ties **never fire** on mega, and the exact predicate *regressed*
recovery (jitter was helping the assembly). The real bail root was **runaway
`Expand`** in the C resolver's assembly (boundary-recovery leaves degree-1
endpoints → the component grows 8→15→26→39→49 until the cap trips).

### 2. Free local hull + pin-by-key — BAILS ON ALL MEGA (0/6)
Build a free hull over `S = C ∪ secured(C)` (secured = nearest-128 grid
neighbors), read each contested cell off the dual, pin each boundary vertex to
the existing ring vid **by key**. Bails on every mega case. Root cause: the
**exact** hull picks a different third-generator at a near-degenerate seam vertex
than the **approximate** (gnomonic) original diagram did, so the key exists in no
ring cell → no pin → revert. The seam is itself near-degenerate; exact vs
approximate disagree *exactly there*. (Detail in `reclip-local-hull-design.md`,
"MEASURED 2026-06-20".)

### 3. Constrained boundary ("fix the seam, re-mesh only the interior")
The right *direction* (a free hull is the wrong tool — it recomputes the
boundary). But Codex found the **interface hole**: a C–C–V seam vertex `{g,h,r}`
is a *port* that demands an interior `g–h` Voronoi edge; if the exact interior
disagrees, you re-introduce C's degree-1 failure one ring deeper. "Fixed
enclosing loop + exact interior" is *not* sufficient — the boundary must be
honored as constraints, not just a containing loop.

## The current design idea: OVERLAY with create-and-attribute (+ collapse)

Treat the contested region's boundary as a **fixed arbitrary loop** (the ring
cells' inner edges, each segment carrying its valid owner `v`). Compute the
**free Voronoi of only the contested generators**, then **overlay** it onto the
loop. Every new contested-edge endpoint resolves by one of (originally) two
cases:

- **lands on a loop-edge interior** → split it; new degree-3 vertex `{c,c',v}`
  (`v` read off the loop owner, `c,c'` are the bisector you're drawing).
- **lands on an existing loop vertex** → snap to it → a **high-degree (4+)
  vertex** = the mega degeneracy represented *faithfully* (one honest vertex
  instead of warring degree-3 vertices).

Key realization: you never *match* the original's keys — you *create* the seam
vertex and *attribute* its third generator from the loop. That sidesteps the
pin-by-key disagreement entirely. Map-overlays of two valid subdivisions are
always valid → it doesn't bail on key disagreement.

**The "collapse" refinement (user's, good):** a stranded loop corner left at
degree-2 should be **deleted** (merge the two edges), not force-inherited — at a
near-flat (near-cocircular) corner that's a tolerance-level simplification, and
it *removes* a degeneracy where inheritance would *manufacture* a near-zero-angle
one. Generalizes to "collapse the degeneracy" (delete degree-2 / merge
near-coincident into high-degree) instead of "exactly arrange it" — more robust,
since merge/delete degrade gracefully where split/snap are brittle.

### Codex's adversarial verdict on the overlay (2026-06-21)
- **Not total / two cases not exhaustive.** Third case: a loop corner hit by *no*
  contested bisector must still be inherited by the inside contested cell, or its
  ring vertex strands at degree-2 (→ handle by clipping to the loop *polygon*,
  corners included). Plus interior C–C–C vertices, arc overlaps, multi-loop
  components.
- **The snap decision is the crux (HOLE).** No scalar tolerance is robust for all
  near-degenerate inputs; needs exact/topological classification + snap-rounding
  + the gate, else it just relocates pin-by-key's fragility into a threshold.
- **High-degree representation is fine.** Final storage uses vids, not keys
  (`diagram.rs`); degree-4+ is explicitly valid (`validation.rs:915`,
  `high_degree.rs`). No key-type change needed: case-2 is a triple `{c,c',v}`
  (fits `VertexKey3`); case-1 reuses an existing vid.
- **New caveat — `delaunay_triangles()` contract.** Case-2 seam vertices are
  bisector-loop *crossings*, **not circumcenters** of their triple, so the public
  Delaunay-dual API would be wrong at repaired seams. Inherent trade of a fixed
  loop (circumcenter seams = move the boundary = the free hull = bail). Must be
  documented if built.
- **Construction:** the tractable form is *clip the C-only Voronoi to the fixed
  loop polygon + 1D boundary overlay with midpoint owner attribution*, not a
  general 2D spherical arrangement engine.

### Code-grounded implementation delta (from reading `reclip_repair.rs:797`)
The current re-stitch is key→vid and only rewrites **contested** cells. The
overlay needs: (a) poly entries that can carry a **direct existing vid** (case-1
snap, bypassing key-match — exactly what failed before); (b) **rewriting RING
cells** to insert case-2 split vertices (the genuinely new capability — today the
repair never touches valid cells). Splits per ring edge must be batched + sorted
along the edge.

## The DIAGNOSTIC (2026-06-21) — the decisive measurement

`diagnose_seam` (behind `S2_RECLIP_DIAG`, in `reclip_repair.rs`) rebuilds the
free hull per component and classifies each seam face: `match` (exact hull
reproduces the key → pin would succeed) vs `mismatch`, split C–V–V / C–C–V, and
for each mismatch the angular distance to the nearest existing ring vertex (the
collapsibility proxy). Run on the 6 mega A/B seeds (`mega`, frac 0.8, ST):

**Aggregate (K=128, the design's secured set): 1012 seam faces →**
- **match 662 (65%)**, mismatch 350 (35%)
- mismatches: **C–V–V 84%**, C–C–V 16% (note: *not* C–C–V-dominated as theorized)
- mismatch distance: **coincident <1e-4: 44%**, near <1e-2: 23%,
  **far ≥1e-2 (~0.5°): 33%**

Per-seed (K=128): `200k s1` 4 comps seam=210 match=130 mm=80 (far=33); `200k s2`
1 comp seam=36 match=15 mm=21 (far=15); `200k s3` 3 comps seam=134 match=80 mm=54
(far=17); `500k s1` 3 comps (1 bailed) seam=71 match=47 mm=24 (far=13); `500k s2`
4 comps (1 bailed) seam=205 match=135 mm=70 (far=19); `500k s3` 5 comps seam=356
match=255 mm=101 (far=18).

**Artifact test (inconclusive/confounded):** bumping K 128→384 to see if "far"
was under-securing made far *worse* (33→48, 17→24, 18→25). That rules out "too
few neighbors," but a bigger `S` grows the free hull's outer boundary, which
manufactures its own spurious seam faces — so the free hull **distorts the very
measurement**. Cannot cleanly certify "far" as real-disagreement vs
hull-artifact with this setup.

### What the diagnostic decided
1. **Pin-by-key is dead**: 65% match, but one mismatch bails a whole component,
   and every component has several → 0/6, confirmed.
2. **Collapse alone cannot beat jitter**: every component has *far* mismatches
   (not coincident/near), so a pin+collapse+revert-on-far hybrid would recover
   *zero* (every component reverts) — worse than jitter's 4/6. **To beat the
   baseline you must handle the far mismatches**, which is the full overlay
   create-and-attribute (or promotion) — the expensive part. The cheap part
   doesn't move the needle.
3. The real-vs-artifact status of "far" — which would size the overlay's actual
   payoff — is **confounded** and unresolved.

## Working-tree state (uncommitted)

- `src/knn_clipping/reclip_repair.rs`: the `diagnose_seam` diagnostic + `diag()`
  + the `S2_RECLIP_DIAG` early-return hook are **uncommitted** (currently with
  `SECURED_K=384` from the artifact test). The production resolver is unchanged
  (still the committed jitter path). Decide whether to keep the diagnostic
  (gated, useful) or revert it.
- The free-hull *integration* (`resolve_component_hull`, pin-by-key) was already
  reverted (`git checkout`) after the 0/6 result; only the tested hull *core*
  remains committed.

## The OPEN DECISION

Correctness is met; this is a recovery-rate call on a rare opt-in path.

1. **Stop at jitter + gate** *(recommended)*: 4/6 mega, valid-or-error, document
   "mega recovery beyond jitter" as a known limitation. Revert the diagnostic
   scaffolding; keep the tested hull core. Lowest risk, no API caveat.
2. **Build the overlay+collapse arrangement engine anyway**: clip-to-loop-polygon
   + 1D boundary overlay + snap-rounding + ring re-tessellation + the
   `delaunay_triangles()` caveat + handling the far mismatches. Multi-day build,
   permanent API caveat, payoff the diagnostic could not size.
3. **One more measurement to de-confound "far"**: per far mismatch, does the
   original ring cell even have the `g–r` *edge* (real disagreement) vs lack the
   pair entirely (likely free-hull artifact)? A focused hour that would settle
   option 2's payoff before committing to it.

Recommendation: **(1)**, unless higher mega recovery is specifically worth the
engineering + the Delaunay-exactness caveat — in which case do **(3)** first.
