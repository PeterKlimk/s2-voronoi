# Dense-cell sub-index ("punch 1") — design note

Status (2026-06-14): **designed, not built.** Deferred until extreme local
density is a real workload. This is the local half of the "1-2 punch" for
non-uniform inputs; the global half (the occupancy-feedback rebuild) shipped
re-calibrated on 2026-06-14 (see docs/optimization-ideas.md, "Occupancy
rebuild re-calibration"). The two are **synergistic, not alternatives**.

## Problem

The query grid (`CubeMapGrid`) is flat by design: CSR `cell_offsets` +
contiguous SoA `cell_points_{x,y,z}`, precomputed flat neighborhoods, and a
SIMD scan of each cell's full point range (`packed_knn`). A cell holding
`occ` points is scanned in full by every query that reaches it — and via
ring expansion that includes every neighbor cell's queries, not just the
`occ` homed in it. So a giant cell costs **O(occ²) regionally**, and the flat
grid has no intra-cell structure to prune it. At extreme concentration this
goes from slow to infeasible (measured: 80% of points in one cap never
finishes a 300k build).

### Why this is distinct from the rebuild

The occupancy rebuild raises the *global* resolution. It is the right tool
when the dense region is the **majority** of points (background negligible,
giant cells otherwise infeasible). It is the *wrong* tool for a minority
hotspot — a global re-grid de-tunes the whole grid's mean density (measured
1.5–9× slower). And a single global resolution, memory-capped, cannot always
fully tame the densest cells (mega at 1M leaves residual over-full cells).

The sub-index fixes the **residual / local** problem the rebuild can't:
- minority hotspots the rebuild now correctly skips,
- the residual over-full cells a memory-capped global resolution leaves,
- without de-tuning the background (the flat grid stays as-is everywhere else).

Rebuild alone fails those; sub-index alone would wastefully refine most cells
on a majority-dense input where one cheap global re-grid is better. They
compose on different axes.

## Hard requirement: a costless fast path

Large cells are rare, so the machinery must be a **predictably-not-taken
branch** that is effectively free when no cell is dense (the overwhelming
common case).

This needs **no new per-cell data structure**. Both scan paths already have
the cell's occupancy as a live value (`end - start` from `cell_offsets` —
`shells.rs::scan_cell`, `packed_knn/mod.rs`). The guard is two lines:

```rust
let occ = end - start;
if occ > DENSE_CELL_THRESHOLD {   // rare → branch predicted not-taken
    // consult this cell's sub-index (best-first stream; see below)
} else {
    // the existing linear / SIMD scan, verbatim
}
```

- One compare against a constant, on a register-resident value. For a run
  with no dense cells the outcome is constant → perfectly predicted →
  disappears into speculation. No flags array, no extra cache line.
- A side map (`dense_cell_id → sub-index`) is consulted **only** when the
  branch is taken. Normal cells never touch it.
- Sub-indexes are **built only for over-threshold cells at grid
  construction**, so a uniform input builds zero of them and the side map is
  empty. Cost scales strictly with the (rare) problem.

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

### Structure options (mostly orthogonal to the integration below)

- **Intra-cell axis-sort** — permute the dense cell's points within their
  existing SoA range along one axis; scan binary-searches the axis slab.
  Least invasive (SoA/SIMD preserved), but prunes one axis only.
- **Per-cell mini-grid** — small bucket grid; fuller pruning, side storage.
- **Per-cell 3D kd-tree** — full pruning, best-first (priority-queue)
  traversal yields points in increasing distance — which is exactly the lazy
  stream the consumer wants (below). Best fit for the *extreme residual*
  cells punch 1 targets (post-rebuild, capped, very dense), where full
  pruning beats a one-axis slab. Recommended starting point.

## The real work: producer/consumer integration

The hard part is not the structure or the fast-path branch — it is that
`scan_cell` is a **"dump ALL candidates" producer**, while the pruning radius
/ k-th-best lives in the **consumer** (packed_knn's chunked early-termination
plus the ring `pending_bound`). A sub-index must bridge that one of two ways:

- **(A) Range query** — plumb the consumer's current radius down so the scan
  asks the structure for "dense points within R of q." Correct, but threads a
  radius through the producer.
- **(B) Lazy distance-ordered stream** *(recommended)* — the dense cell
  yields its points in increasing distance to q; the consumer's existing
  early-termination pulls until its certificate closes. A kd-tree's
  best-first traversal *is* this stream, so the structure and integration
  align.

There are **two scan paths** to intercept: the directed `shells.rs::scan_cell`
and `packed_knn`'s own center-cell range read. Both need the guard.

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

1. At grid build: for each cell with `occ > DENSE_CELL_THRESHOLD` (rare),
   build a 3D kd-tree over its points; store in a side map. Uniform input
   builds none.
2. In both scan paths: `if occ > DENSE_CELL_THRESHOLD` → best-first stream
   from the cell's kd-tree, bounded by the ring certificate and filtered by
   eligibility; else the existing scan, verbatim.
3. Fast path cost: one predicted-not-taken compare. Dense path cost: O(log
   occ)-ish per query instead of O(occ), paid only by the rare dense cells.

## Open questions for implementation

- Exact `DENSE_CELL_THRESHOLD` (crossover where kd-tree query beats linear
  SIMD scan) — measure on a quiet box.
- Integration model (A vs B) and whether the packed_knn chunked consumer can
  pull from a lazy stream without a large refactor.
- Whether the axis-sort (in-place, SIMD-preserving) is "good enough" for the
  realistic residual cells, deferring the kd-tree.
