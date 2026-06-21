# Hull + Snap-to-Loop Resolver — Experiment Results (2026-06-21)

Behind `S2_RECLIP_HULL` (selects an experimental Tier-2 resolver instead of the
committed jitter path; opt-in, under `S2_RECLIP_REPAIR`). The goal was the
near-Voronoi totality contract: rebuild a contested component `C` from the exact
local Delaunay (`local_hull`) and attach it by **snapping** seam vertices to the
existing loop (bypassing the pin-by-key match that bails on near-degenerate
seams). Measured on the 6 mega A/B seeds (200k/500k, seeds 1-3, single-thread).

## What the audit established first (`S2_CANON_AUDIT`)

- Oracle validated on uniform: ~0.8% of cells disagree with the exact Delaunay,
  each a single near-cocircular diagonal flip at ~1e-5 rad.
- On mega the exact-vs-fast disagreement **percolates the whole cap**: ~99% of
  resolvable cap cells disagree, ~5 disagreeing neighbors/cell. Only ~7-31 cells
  are actually *invalid* (unpaired). So the **validity** boundary (the contested
  component) is tiny, but the **exact-agreement** boundary is the whole cap →
  "expand until exact agreement" is O(cap), the cap rim is the only firewall.
- Displacement of the disagreements is bounded (≤~1e-3 rad ≈ one cap cell-edge);
  the cap is combinatorially recoverable (f64 predicates), only the vertex
  *positions* are f32-ill-conditioned.

## The snap experiment — three sub-problems, all observed

**1. Naive snap → degeneracy (every component).** Snapping each boundary face to
its nearest existing loop vid independently produces consecutive duplicate vids
(`dup_edges` = 4/9/11/13/0/5 across seeds) — the near-cocircular case where the
exact hull splits one existing seam vertex into two faces. Confirms snap is *not*
strictly safe.

**2. Snap + collapse → C-internally valid, still 0/6.** Collapsing consecutive
snaps to the same loop vid (merge the split) drove `dup_edges` and `digons` to 0.
But the gate still reverts on all 6 seeds. **Decisive:** 200k s1 (comps=1,
resolved=1, all bails 0, dup=0, digon=0) and 200k s3 (resolved=2, all clean)
revert anyway. A fully clean component still reverts ⇒ the failure is the
**un-rewritten ring**: C's collapsed boundary no longer matches the ring cells'
edges → unpaired / degree-2 at the seam. The repair today never touches valid
cells; snapping only C cannot close the seam.

**3. Broken fans on the near-coplanar cap (`bail_fan`).** `local_hull.cell_faces`
returns an open fan for some `C` cells (bail_fan = 1-2 on several seeds), so the
exact hull doesn't even give those cells a clean Voronoi cell. The cap patch is
nearly coplanar; the hull's cycle-walk around such a generator does not always
close. Needs degenerate-face / securing hardening before the hull is a reliable
engine here.

## Conclusion

Snap-to-loop that rewrites only `C` is a dead end (0/6, proven by clean
components reverting). A valid hull-based repair needs the genuinely-new
capability flagged in `reclip-tier2-state-2026-06.md`: **bilateral seam edits —
rewrite the ring cells too** (split their edges / collapse their vertices to
match C's new boundary, batched and sorted along each ring edge). Plus fan
hardening for sub-problem 3. That is the full overlay engine; multi-day.

The dichotomy is now empirical: a hull rebuild must either reproduce the existing
seam exactly (pin-by-key — fails on the percolating disagreement) or rewrite both
sides (bilateral overlay — the real work). There is no rewrite-one-side middle.

## Open options (unchanged contract: valid-or-error; near-Voronoi accepted)

1. **Build the bilateral overlay** (rewrite ring cells + fan hardening). Highest
   recovery, multi-day, touches valid cells for the first time.
2. **Loop-fan-fill totality fallback** (does NOT use the exact hull): fill the
   contested loop using ONLY existing loop vertices (no new vertices, no snap
   collisions, no ring rewrite — the ring already references those vertices, so
   the seam pairs by construction). Crude geometry, but guaranteed valid and
   bounded. Sidesteps all three sub-problems. Needs a small loop-partition among
   C's generators that tiles validly.
3. **Stop at jitter (4/6) + document**, as `reclip-tier2-state` recommended.

The near-Voronoi reframe makes (2) newly attractive: it trades geometric fidelity
(which the audit shows is already only cell-scale on mega) for a simple, total,
always-valid construction.

## Codex adversarial review (2026-06-21) — verdict + two refinements

Independent review (gpt-5.5) agreed with the conclusion and the decision (stop at
jitter 4/6 + document; keep only gated, inert, documented measurement code; do
not build the bilateral overlay now). Two refinements that sharpen the spec:

- **Bilateral-bounded requires a strict invariant.** "Rewrite the ring" is bounded
  *only if* the overlay keeps a **fixed outer frontier**: ring cells' OUTER chains
  (edges to cells beyond the ring) stay **byte-identical**, and *every* cell
  incident to a replaced seam vertex is rewritten — including ring cells touching
  a seam **corner**, not just edge-neighbors. If the exact hull is allowed to move
  the seam through ring-cell interiors or alter ring-to-outside edges, it strands
  old vertices / creates split points the outer cells must also learn → unbounded
  (the percolation again). So the build's load-bearing rule is: replace only the
  ring's *inner* chain against `C`; never touch its outer chain.
- **Loop-fan-fill is NOT a true near-Voronoi guarantee.** A disk with `m` boundary
  vertices and no interior vertices supports only a bounded number of valid faces;
  `cell_count == input` (an invariant) can exceed that. To guarantee topology you
  must add synthetic interior vertices and triangulate/fan — which is no longer
  meaningfully Voronoi. So option 2 is only an **emergency valid-graph fallback**
  ("inside this tiny loop, geometry is approximate"), not a near-Voronoi repair.
- Sub-problem 3 (broken `cell_faces`) is confirmed an implementation/degeneracy
  issue (no coplanar policy; `cell_faces` assumes one clean cycle), not a
  fundamental obstacle to the exact hull.

## Parallelism inflates the mess (2026-06-21) — cross-bin edge resolution is weaker

Stress-testing the totality goal surfaced a more fundamental issue than the
backstop. The dedup engine's **cross-bin** edge resolution
(`resolve_edge_check_overflow`) pairs **fewer** near-degenerate edges than the
**within-bin** path (`collect_and_resolve_cell_edges`), so splitting a dense
degenerate region across bins leaves a **bigger** contested residual.

Measured (mega 1m, repair off, unpaired count; `S2_BIN_COUNT` only changes bin
assignment, so the delta is purely within-vs-cross-bin):

| | bins=1 (≈serial) | default (parallel) |
|---|---|---|
| pole s1 | 36 | **61** (×1.7) |
| pole s2 | 130 | 141 |
| pole s3 | 95 | 108 |
| edge s2 | 109 | 127 |

Deterministic (1m edge default → 127 three times; not a race). Shows up at the
pole too, not just face seams — at 1m the cap spans multiple intra-face bins.
(At 200k/500k bins=1==default *only* because the smaller cap fits inside one bin;
do not generalize bin-independence from small n.) The `S2_BENCH_CAP_CENTER`
env knob (pole/edge/corner, tools-only) was added to drive this.

Implication: this does **not** threaten correctness (extra unpaired edges are
still detected — valid-or-error holds), but it makes "never error" **harder**:
parallelism inflates the mess, it scales with n, so any Tier-2 repair faces a
bigger, n-growing component and is likelier to hit `MAX_COMPONENT_CELLS=128` /
`MAX_LOCAL_SET=3072` and bail. (UPDATE 2026-06-21: first read as a fixable dedup
inconsistency / "highest-value lever", but the origin tally + Codex review
concluded it is an inherent **artifact of parallel construction** — conservative
cross-bin pairing of degenerate >2-record groups vs optimistic within-bin
last-write-wins — NOT a bug and not to be fixed. See the retraction in the
SUPERSEDING DIRECTION section. The n-growing residual is a constraint the totality
repair must absorb, not something to shrink first.)

## Decision (2026-06-21)

Stop at jitter (4/6 mega) + document. Correctness is already met (valid-or-error),
this is a rare opt-in path, and the overlay's payoff stays unconfirmed against a
multi-day build with a likely `delaunay_triangles()` caveat. Keep the audit
(`S2_CANON_AUDIT`) and the hull-snap experiment (`S2_RECLIP_HULL`) as gated,
inert-by-default measurement code; this doc is the spec if the overlay is ever
revisited.

## SUPERSEDING DIRECTION (2026-06-21, later) — totality, never error

User raised the bar: `compute` should be TOTAL — a valid graph for any welded
input, never the `residual_error`. Near-Voronoi geometry accepted; the regime is
ill-posed (non-exact libs also produce nonsense here). This supersedes "stop at
jitter."

### Rim test (decisive) — no local firewall
BFS outward from a contested component, per-ring exact-vs-fast agree-rate
(`S2_RIM_PROBE`). The interior is uniformly ~99% disagree all the way out; the
first clean ring (zero disagreements) is reached only after engulfing
**~160,024 cells = the whole cap** (200k mega). So a DT-rebuild that must pin
where exact==fast is a WHOLE-CAP rebuild, not local. "Robust local Delaunay" is
really "robust whole-cap Delaunay."

### Consequence — the cheap totality path
For totality you only need to fix the RESIDUAL component (the ~tens that don't
pair); the ~99% disagree-but-PAIRED cap cells already ship valid. A **patch
synthesizer** on the residual component reproduces the existing boundary loop
verbatim (loop edges are paired-by-construction → ring cells unchanged → seam
pairs, no firewall, no whole-cap cost). Interior fill spectrum (drop-in, same
skeleton): centroid star [floor] → fan center at the cocircular **circumcenter**
= one honest high-degree vertex [near-free] → loop-constrained local DT of the M
component generators [multi-vertex]. Whole-cap exact DT = rejected as default
(rim kills locality; wrong cost).

### Final Codex review — the boundary extractor is the product
"Reproduce the loop verbatim ⇒ clean seam" is sound only CONDITIONALLY, and
`identify_components` does not prove it. The real, totality-critical work is a
**boundary extractor** that proves a manifold boundary and decomposes it into
oriented simple cycles. Holes to close:
- boundary may be a GRAPH not a cycle (pinch vertices, multi-loop, holes, one
  ring cell contributing multiple inner chains) → naive fan fails / dups
  vertices (`validation.rs:601`);
- the fill must reference EVERY boundary vid or a ring vertex drops below
  incidence 3 (`validation.rs:663`);
- `Resolved` must become first-class `ExistingVid | Synthetic`, not fake triples;
- remove every bail source (`MAX_COMPONENT_CELLS`, resolve-failure,
  unpinnable-key); treat **`u16` cell-length as the one legitimate hard cap**
  (`diagram.rs:308`) — or widen to u32;
- keep the whole-diagram all-or-nothing gate (per-component commit is unsafe with
  shared loop vertices);
- local-DT fidelity only with boundary constrained to loop vids; `local_hull`
  lacks a coplanar policy so it is not a total engine as-is.

Process steer RETRACTED (2026-06-21): the cross-bin gap is NOT a fixable
inconsistency. After the origin tally + Codex review it is an inherent **artifact
of parallel construction** — the cross-bin path is conservative about degenerate
>2-record edge groups (`CrossBinSingleSided`/`DuplicateSide`) where within-bin is
optimistic (last-write-wins). Symmetric 1:1 edges never go single-sided; only
near-cocircular degenerate groups do. The serial path papers over real ambiguity;
matching it would mean making the parallel path non-deterministic. **Parked, not a
bug.** The n-growing residual is therefore a *constraint the totality repair must
absorb* (remove the `MAX_COMPONENT_CELLS`/`MAX_LOCAL_SET` bails), not something to
shrink first.

### Refined plan
1. ~~Cross-bin resolution fix~~ — DROPPED (not a bug; parallel artifact, see above).
2. Boundary extractor (the centerpiece): manifold proof + oriented-cycle decomp,
   pinch/multi-loop/hole/multi-chain handling.
3. Fill: fan per region, reference every boundary vid, fresh first-class synthetic
   interior vids; special-case M<3 / M>K.
4. Plumbing: first-class vertex refs; remove bails; u16 cap as the only limit;
   keep whole-diagram gate.
5. Fidelity later inside the skeleton: circumcenter center, then loop-constrained
   local DT.

All three Codex reviews converge: direction confirmed, whole-cap DT rejected, the
work is **boundary extraction + bail removal**, not the fill.

## Boundary extractor built + measured (2026-06-21) — grow-until-clean is the unlock

Step 2 (the boundary extractor) is implemented and unit-tested in
`src/knn_clipping/boundary.rs`: `collect_boundary_edges` (inside-on-left directed
edges, reversed from valid ring cells), `decompose_cycles` (oriented simple
cycles, with angular-order pinch handling at shared vertices), `extract_boundary`,
and a `diagnose_boundary` for the probe. Unit fixtures cover single loop, two
disjoint loops, figure-eight pinch (genuinely exercised — the walk must pass
*through* the pinch, not close at it), unbalanced/missing-position errors, and a
hand-built single-triangle collect.

Measured on real contested components via `S2_BOUNDARY_PROBE` (mega 1m, repair on,
parallel, seeds 1-3 × pole/edge):

- **The boundary is NOT a clean manifold as-extracted.** Most components fail with
  `UnbalancedVertex`: a small, *balanced* set (2-5 per component) of dangling rim
  ends — vids with in/out degree (1,0) paired with (0,1). Not on residual edges;
  every background generator at every dangling end is already in the gathered ring
  (`bg_in_ring` all true), so it is **genuine rim non-manifoldness** (the
  near-cocircular percolation reaching the rim: adjacent ring cells reference
  slightly different rim vertices), NOT a gathering gap. This empirically refutes
  the bare "reproduce the boundary verbatim, `identify_components` suffices" claim
  (exactly as Codex flagged).
- **Multi-loop boundaries are real** (up to 3 loops/component, loop sizes 5-202),
  so the fill must handle several oriented loops, not one. Pinch never occurred on
  real data (`miss_pos=0`), though the extractor handles it.
- **Grow-until-clean converges, cheaply (the unlock).** Absorbing the background
  cells at each dangling rim end into `C` and re-extracting makes the boundary a
  clean manifold — in **one growth step** for almost every component (one took
  two), final component size ≤ 88, across all 6 seeds × both centers, every
  component, `extract_ok=true`. This is the principled alternative to bilaterally
  healing the rim (which would touch ring cells and reopen the firewall): instead
  push the component boundary *outward* past the non-manifold rim until it lands
  where the ring agrees. It reuses the spirit of the existing `Expand` mechanism.

**Refined pipeline (measured):** `identify_components` → **grow-until-extractable**
(absorb dangling-rim cells, 1-2 steps, bounded) → `extract_boundary` (clean
oriented loops) → fill. The grow step is the missing piece the extractor surfaced,
and it is cheap. Step 3 (the fill) now has a firm input contract: one or more
clean oriented loops of existing vids, total component size ≤ ~90, ring untouched.

Harness: `S2_BOUNDARY_PROBE` (extract every component, no fill),
`S2_BOUNDARY_DETAIL=<n>` (dump first n unbalanced components + grow convergence),
`S2_BOUNDARY_GROW_ITERS` / `S2_BOUNDARY_GROW_MAX` (grow caps). All inert by
default.

## Edge-domain pivot — extract the topological cut, not a key-derived adjacency (2026-06-21)

User hypothesis: "non-loop boundaries should be surfaced by edge mismatches."
Tested it with an UNRESTRICTED local edge-pairing check (`local_unpaired_incident`)
at every imbalanced vid. Result: the key-domain extractor's imbalances are at vids
that are **locally edge-paired CLEAN** (`local_unpaired=0`) — so they are NOT real
edge defects; they are a **classification artifact**: `key_common_pair` drops a
boundary edge when two consecutive ring vertices don't share a clean 2-generator
edge (near-degenerate vertex), even though the actual vid-pair topology is fine.
The literal hypothesis is false (nothing to detect); the deeper instinct — work in
the edge domain — is right.

Codex (gpt-5.5) review agreed and tightened it. **The correct primitive:**
`Boundary(C) = actual paired undirected edges whose two incident owners contain
exactly ONE cell in C` — the topological cut of the face set, not a key recognizer.
Enforce the full paired-edge contract: exactly two uses, opposite orientation,
exactly one owner in C; reject duplicates; take orientation from the **ring
(trusted) side, reversed** (C cells are the suspect side — never source orientation
from them). Keys are kept for pinning/fill metadata only.

Implemented as `CollectMode::Paired` (now the default; `Keys` kept for A/B). Unit
test `paired_mode_survives_degenerate_key` locks the win: Paired reconstructs the
loop where Keys drops the edge and fails.

Measured (mega 1m, parallel, all 6 seeds × pole/edge):
- **Paired flips the result:** most components now extract cleanly first-try
  (e.g. pole s2 2/7→7/7, pole s3 1/5→5/5); residual failures drop from ~all to 0-2
  per run. Multi-loop confirmed real (loops up to 339 vids).
- The **remaining 0-2 imbalances per run are GENUINE** (here the user's hypothesis
  IS right): the imbalanced vids are now `on_residual=true`, `local_unpaired=2`,
  all-C keys (`gens_in_C=3`) — interior splits of a degenerate RIM vertex (two
  near-coincident vids sharing a C generator, joined only by an unpaired residual
  edge), so the cut enters one split and exits the other and cannot close.
- grow-by-KEY is STUCK on these (all-C keys → nothing to absorb). The fix is to
  broaden along the OTHER axis: absorb the **ring cells incident to the imbalanced
  vids**. With that fallback, every stuck case CONVERGES in one step (1-2 cells),
  `extract_ok=true`, across all seeds.

**Final measured pipeline:** `identify_components → extract_boundary
(CollectMode::Paired, full contract) → on a classified failure (genuine non-
manifold cut / interior rim split) broaden by incident ring cells, re-extract
(1 step) → clean oriented loops → fill`. Per Codex, broaden ONLY behind specific
classified failures (singleton-near-C, missing partner, use-count≠2), fail loud in
probe / broaden in release; do NOT keep grow as a quiet catch-all. A global
vid-pair incidence map (built once on the cold repair path) would make
`C ∪ ring` completeness a proof rather than an empirical observation — optional
hardening for "any welded input".

Verdict (Codex, unchanged): boundary-extractor + synthetic-interior fill is the
right totality mechanism; whole-cap exact repair is the wrong blast radius. The
correction was making extraction a cut over actual edge uses.
