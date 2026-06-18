# Dense-cell sub-index ("punch 1") — design note

Status (2026-06-18): **axis-sort center-cell band prune implemented**. This
document is now the compact design background; the implementation ledger is
`docs/punch1-center-cell-integration.md`.

Punch 1 is the local half of the "1-2 punch" for non-uniform inputs. The global
half is the occupancy-feedback rebuild (see `docs/optimization-ideas.md`,
"Occupancy Rebuild Re-Calibration"). The two are **synergistic, not
alternatives**: rebuild handles majority-density; the dense side index handles
residual over-full cells after rebuild.

## Problem

The query grid (`CubeMapGrid`) is flat by design: CSR `cell_offsets` +
contiguous SoA `cell_points_{x,y,z}`, precomputed flat neighborhoods, and a
SIMD scan of each cell's full point range (`packed_knn`). A cell holding
`occ` points is scanned in full by every query that reaches it — and via
ring expansion that includes every neighbor cell's queries, not just the
`occ` homed in it. So a giant cell costs **O(occ²) regionally**, and the flat
grid has no intra-cell structure to prune it. At extreme concentration this
goes from slow to infeasible. The implemented Punch 1 path targets the worst
center-cell gather first, where cap-like inputs previously spent almost all
time.

### Why this is distinct from the rebuild

The occupancy rebuild raises the *global* resolution. It is the right tool
when the dense region is the **majority** of points (background negligible,
giant cells otherwise infeasible). It is the *wrong* tool for a minority
hotspot — a global re-grid de-tunes the whole grid's mean density (measured
1.5–9× slower). And a single global resolution, memory-capped, cannot always
fully tame the densest cells (mega at 1M leaves residual over-full cells).

The sub-index fixes the **residual / local** problem the rebuild can't:
- residual over-full cells a memory-capped global resolution leaves,
- cap-like post-rebuild cells where the packed center pass would scan/select
  O(occ²),
- without de-tuning the background (the flat grid stays as-is everywhere else).

Production currently clears the dense index when no rebuild fired. That keeps
minority hotspots/moderate clusters on the baseline path, because the
axis-sort band was measured as overhead for fast-closing clustered/outlier
cases.

## Hard requirement: a costless fast path

Large cells are rare, so the machinery must be a **predictably-not-taken
branch** that is effectively free when no cell is dense (the overwhelming
common case).

This needs no hot per-cell flag. Scan paths already have the cell's occupancy
as a live value (`end - start` from `cell_offsets`), and the side map is only
consulted when the dense branch is taken:

```rust
let occ = end - start;
if occ > DENSE_CELL_THRESHOLD {   // rare → branch predicted not-taken
    // consult this cell's side index
} else {
    // the existing linear / SIMD scan, verbatim
}
```

- One compare against a constant, on a register-resident value. For a run
  with no dense cells the outcome is constant → perfectly predicted →
  disappears into speculation. No flags array, no extra cache line.
- A side map (`dense_cell_id → sub-index`) is consulted **only** when the
  branch is taken. Normal cells never touch it.
- Sub-indexes are **built only for over-threshold cells at grid construction**,
  so a uniform input builds zero of them and the side map is empty. Production
  additionally drops the side index unless the occupancy rebuild fired.

`DENSE_CELL_THRESHOLD` is a *per-cell* "linear-scan vs sub-index crossover"
(the `occ` where O(occ) scanning loses to the structure's lookup/traversal
overhead — hundreds to low-thousands; below it the flat SIMD scan wins, so do
NOT sub-index). This is a different knob from the rebuild's `Σocc²/n` trigger,
which decides "is the whole grid mis-sized." They compose: post-rebuild, the
residual capped cells are exactly those that still clear this per-cell bar.

## Ambient-3D structure — no frame special-casing

Cell points are unit 3-vectors and distance is chord (`dot3`). So the
sub-index is built over the dense cell's raw 3D coords, and a query is *just
another 3D point* — a query expanding in from a neighbor cell is served
identically to one homed in the cell. **No cell-local 2D frame, no
cross-face/wraparound handling.** This is what makes a "parallel structure,
cube grid untouched" approach clean.

### Implementation findings (2026-06-14, from scoping the build)

Two constraints discovered while scoping, which revise the options below:

1. **Sub-indexes must be SIDE structures — do NOT permute the main SoA.**
   `slot_gen_map` is indexed by slot (SoA position) and binning assigns
   local-ids in slot order, so reordering a dense cell's points in place
   changes local-ids and ripples through dedup / packed-kNN / vertex emission.
   So the "in-place axis-sort" framing below is wrong; the axis-sort must be a
   *separate* sorted list of the cell's slot indices, leaving the grid SoA
   untouched. (This unifies the variants: one scaffold = a side map of
   per-dense-cell sub-indexes; the variants differ only in structure type.)
2. **No prune radius exists at `scan_cell` time.** The directed query
   accumulates a whole ring's candidates into `pending`, sorts by dot, and the
   consumer applies the `pending_bound` certificate ring-by-ring — there is no
   per-query current-best threaded into `scan_cell`. So a side index gives *no*
   speedup until the producer/consumer integration (radius-plumb or
   lazy-stream) lands. That integration is the load-bearing work, not the
   structure.

### Structure options

Three candidates were considered. The first one was implemented.

**1. Side axis-sort** *(implemented).*
At build time, store each over-full cell's slot ids sorted by dominant spread
axis. The query binary-searches the axis band `[q-r, q+r]` and only touches
that slab.
- Pros: leaves the main SoA and slot order untouched; cheap to build; the band
  is a superset of every point within radius `r`, so it supports a conservative
  completeness bound.
- Cons: **prunes one axis only** — points inside the slab but far on the other
  two axes are still scanned, so it's a slab not a ball. A partial win: good
  for moderately dense cells, weakening as concentration grows (the slab holds
  ~occ^{2/3} points for an isotropic cluster).
- Status: implemented for the packed center-cell pass. It is enough to remove
  the measured cap center-gather cliff; see
  `docs/punch1-center-cell-integration.md`.

**2. Per-cell mini-grid** *(fuller pruning, side storage).*
Give an over-full cell its own small bucket grid (2D in the cell's tangent
plane, or 3D).
- Pros: prunes all axes (closer to O(1) per query); conceptually a *local
  re-grid* of just that cell — the local analogue of the global rebuild.
- Cons: needs side storage; **breaks SoA contiguity for those cells** (loses
  SIMD there — acceptable, since those are exactly the cells where the linear
  scan was the problem); another structure to build/validate.

**3. Per-cell 3D kd-tree** *(full pruning + free lazy stream).*
- Pros: full pruning, and best-first (priority-queue) traversal yields points
  **in increasing distance** — which is exactly the lazy stream the consumer
  wants (integration model B below), so structure and integration align. Best
  fit for the *extreme residual* cells punch 1 targets (post-rebuild, capped,
  very dense), where full pruning beats a one-axis slab.
- Cons: most machinery (per-cell tree build + scalar best-first traversal);
  heaviest of the three.

Recommendation after implementation: keep the side axis-sort as the baseline.
Escalate to the kd-tree only if real dense workloads remain dominated by
within-cell candidate gathering rather than by certificate depth / shell
takeover.

## Producer/consumer integration

The hard part was not the structure or the fast-path branch. It was that the
candidate producer wants to dump all candidates, while the pruning radius /
k-th-best lives in the consumer. The implemented center-cell path bridges this
with a radius-bounded band:

- Pick `r` from cell extent and `DENSE_BAND_TARGET_COUNT`.
- Gather a side-axis band widened slightly for fp error.
- Keep points above `band_bound = 1 - r²/2`.
- Report completeness only down to `band_bound`; the existing shell takeover
  remains the backstop below it.

This is intentionally narrower than the original all-scan-path plan: the
packed center-cell read was the measured O(occ²) cliff. A kd-tree/lazy stream
for shell cells remains a possible later escalation, not current baseline.

## Correctness traps (must design around)

1. **Never truncate by a fixed "nearest-K cap."** That can *silently* drop a
   true neighbor and corrupt the diagram. The query must be radius-bounded
   (provably returns everything within the certificate radius) or the stream
   must run until the ring `pending_bound` certifies closure. The ring cap
   bounds (`cell_min_dist_sq`) supply the safe radius.
2. **Honor the directed-eligibility filter.** When the dense cell is the
   query's start cell, `scan_cell` applies `allows_center_slot` /
   `DirectedCellMode` (the stitching-ownership contract). The sub-index must
   filter by eligibility, not just distance, or it will double-emit / break
   ownership.
3. **Re-validate the NN certificate.** The termination soundness (NN-contract
   suite) is currently argued over the flat full-cell scan. Replacing it with
   a bounded sub-index query needs that argument re-checked for the dense
   path. Start with the radius-bounded form where the bound is obvious.

## Shape, end to end

1. At grid build: for each cell with `occ > DENSE_CELL_THRESHOLD`, build a side
   axis-sort index over that cell's SoA slots. Uniform input builds none.
2. During production grid setup: clear the dense index unless the
   occupancy-feedback rebuild fired.
3. In packed center-cell prep: if a dense radius exists, gather a conservative
   band, emit it as chunk0, set the center completeness bound to `band_bound`,
   and let shell takeover cover anything below.
4. Fast path cost: no dense index and no side lookup on normal grids. Dense
   path cost: band-size work instead of full center-cell gather/select.

## Implementation State

- `src/cube_grid/dense.rs`: side axis-sort index and band query.
- `src/cube_grid/packed_knn/scratch/prepare.rs`: dense center-cell band mode,
  `center_bound`, and shell-takeover backstop.
- `src/knn_clipping/compute.rs`: production gate that clears the dense index
  when no rebuild fired.
- `src/cube_grid/tests/nn_contract.rs`: dense-cell certificate contract
  coverage.

## Open Questions

- `DENSE_CELL_THRESHOLD` and `DENSE_BAND_TARGET_COUNT` still deserve quiet-box
  calibration.
- Whether shell-cell dense indexing is ever needed now that the center-cell
  cliff is gone.
- Whether a 3D kd-tree/lazy stream beats side axis-sort on real dense inputs,
  or whether remaining cost is mostly certificate depth.
