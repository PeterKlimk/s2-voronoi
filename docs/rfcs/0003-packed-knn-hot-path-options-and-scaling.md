# RFC 0003: Packed-kNN Hot Path Options and Scaling Notes

This note consolidates discussion and next-step ideas around the current packed-kNN hot
path (`cube_grid::packed_knn` + `knn_clipping::live_dedup`) with emphasis on:

- Why the current profile is dominated by `packed_knn` at large `N`.
- What correctness constraints exist around ordering and termination bounds.
- Alternatives to the current two-pass `select_nth_unstable + sort_unstable` scheme,
  including radix/bucketing.
- How to scale heuristics (`k0`, `k1`, grid density, neighborhood radius) with point
  density and with the directed intra-bin + edgecheck protocol.

This document is not a finalized plan; it is a menu of options plus suggested
instrumentation to decide among them. It complements RFC 0001 and RFC 0002.

## Background: Current Packed Pipeline (Directed)

### Call flow

In the live-dedup backend, generators are processed per-bin, grouped by their center grid
cell:

- Group preparation: `PackedKnnCellScratch::prepare_group_directed` builds per-query candidate
  state for the group (center cell + neighbor cells).
- Per-cell consumption: `process_cell` calls `packed_scratch.next_chunk` repeatedly (chunk0,
  then optional tail) and clips against emitted neighbors.

Relevant entry points:

- `src/cube_grid/packed_knn.rs`: `PackedKnnCellScratch::{prepare_group_directed,next_chunk,ensure_tail_directed}`
- `src/knn_clipping/live_dedup/build.rs`: `process_cell` packed loop.

### Safety model (r=1 / 3×3 neighborhood)

Packed emits only candidates with `dot > security`, where `security` is a conservative upper
bound on `max dot` for points outside the 3×3 neighborhood (computed via either interior
planes or a ring-2 cap bound).

Packed additionally partitions ring candidates into:

- **chunk0 keys**: candidates with `dot > threshold`
- **tail keys**: candidates with `security < dot <= threshold` (built lazily)

Today, `threshold` is derived from the minimum dot among center-cell hits (if any), so ring
points must beat the worst center hit to be considered in chunk0.

### Ordering and termination coupling

`next_chunk` emits neighbors in strict descending-dot order by storing a key
`(desc_dot_bits, slot)` and applying:

- `select_nth_unstable(n-1)` on the remaining keys (if needed),
- `sort_unstable()` on the first `n`,
- then scattering slot indices.

This ordering feeds two termination checks in `process_cell`:

1. *Within-chunk*: occasionally uses `next_in_chunk` dot as a bound.
2. *After chunk*: uses `chunk.unseen_bound`, derived from the emitted `n`-th neighbor’s dot
   (or `threshold`/`security` at boundaries).

Any change that removes per-neighbor ordering must replace these bounds with equally
conservative alternatives (often less tight), or adjust the termination scheme.

## Observations From Timing/Stage Data

### 2.5M points (density target ~16)

```
cell_stages: pk0=2187462 (87.5%) pk_tail=300495 (12.0%) k18=12002 (0.5%) K24=39 (0.0%) K48=2 (0.0%) exhausted=0 (0.0%)
packed: tail_used=302562 (12.1%) safe_exhausted=12043 (0.5%)
pk_tail_possible: q=2284509 (91.4%)
pk_tail_build: groups=139296 (89.6%)
knn: used=12043 (0.5%)
```

Notes:

- Packed covers essentially all cells; resumable kNN is rare.
- Tail usage is modest (~12%), but tail build is invoked for ~90% of groups (see bug).

### 4M points

```
cell_stages: pk0=2881033 (72.0%) pk_tail=1021201 (25.5%) k18=63700 (1.6%) K24=13228 (0.3%) K48=20818 (0.5%) K96=20 (0.0%) exhausted=0 (0.0%)
packed: tail_used=1067323 (26.7%) safe_exhausted=97766 (2.4%)
pk_tail_possible: q=3577091 (89.4%)
pk_tail_build: groups=230204 (92.2%)
knn: used=97766 (2.4%)
```

Notes:

- Packed chunk0 is less dominant; tail usage more than doubles.
- `safe_exhausted` (and thus resumable kNN) increases ~5× (0.5% → 2.4%).
- Neighbor-count-to-prove-termination distribution shifts upward (p90/p99 larger), which
  usually indicates more work is needed to *prove* termination, not that true Voronoi degree
  grows.

## Note: Tail Build Overreach and Its Fix

Today, tail is built at group granularity:

- When any cell/query transitions from `Chunk0` to `Tail`, `process_cell` calls
  `ensure_tail_directed(...)` once for the whole group.
- `ensure_tail_directed` then scans ring ranges and constructs tail lists for *all* queries
  with `tail_possible == true`, not only the query that actually needs tail.

This can cause near-group-wide double scanning (ring pass + tail pass) even if only a
minority of queries consume tail. Fixing this is high leverage and also reduces noise in
timings for subsequent experiments.

Status: fixed by switching to per-query tail construction via
`PackedKnnCellScratch::ensure_tail_directed_for` (`src/cube_grid/packed_knn.rs:1518`), so the
fallback scan cost scales with tail consumers rather than all `tail_possible` queries.

## First-Principles Constraints

- **Conservative unseen bound is required for correctness.** If we claim `unseen_bound` is
  small while higher-dot candidates remain unseen, we can incorrectly terminate early and
  miss necessary clipping planes.
- **Approximate “k” is OK, approximate “bound” is not.** Emitting 4 or 6 neighbors instead
  of exactly 5 can be fine; claiming we have exhausted everything above some dot threshold
  without actually doing so is not.
- **Directed intra-bin filtering changes the candidate universe.** Earlier same-bin points
  are excluded from kNN/packed; later cells must obtain true earlier-neighbor adjacency via
  edgechecks. Edgecheck-provided neighbors are not distance-ordered and should not tighten
  distance-based termination bounds.

## Option Space (Possible Paths)

### Path A: Fix tail overreach (build tail only for consumers)

**Idea:** make tail building query-targeted (or for a bitset of qids), not group-wide.

Implementation shapes:

- Replace `tail_ready: bool` with per-query readiness (`tail_ready[qi]` or a bitset), so
  `ensure_tail_directed_for(qi)` only populates `tail_keys[qi]`.
- Or: keep the group scan but only compute tail keys for a provided “tail-needed” bitset.

Expected impact:

- Directly reduces `pk_fallback` scanning cost.
- Makes further experiments (radix/bucketing, thresholding) measurable by removing noise.

### Path B: Adaptive chunk sizing (scale `k1`, keep `k0` small)

Observed at 4M: tail usage is ~27%, but `k1` is fixed at 8, implying many tail-consuming
cells iterate multiple times in Tail and pay repeated selection/sort overhead.

**Idea:** keep small first chunk to preserve the dominant fast path, but increase the subsequent
chunk size when the cell is “already hard”:

- `k0` remains modest (e.g. 24).
- `k1` becomes larger (e.g. 16 or 24) at higher densities, or after the first chunk fails to
  terminate.

This adds negligible work for the ~72–88% that finish in pk0 but reduces overhead for the 25%
that spill to tail.

### Path C: Improve the `threshold` heuristic (avoid ring/tail entry inflation)

Current `threshold` is tied to the minimum center-dot among all center hits, which makes
`tail_possible` extremely common (~90% of queries at both 2.5M and 4M).

Potential replacements:

- If center provides at least `k0` safe candidates, set `threshold = security` and skip tail
  entirely (ring candidates only matter if they beat the worst of the *top-k*, not the worst
  of *all* center hits).
- Use a cheap estimate of the k-th dot among center candidates to set `threshold`: sampling,
  partial selection, or a tiny fixed buffer used only for cutoff estimation.

Goal: keep correctness (still only emit `dot > security`) but reduce ring hits and tail frequency
at high density.

### Path D: Cell-level upper-bound pruning inside the ring scan

Instead of deciding “ring candidate passes threshold” only after computing the dot for each
point, prune whole ring cells for a query if no point in that cell can beat the query’s cutoff.

Mechanism:

- For each query and ring cell, compute `ub(cell)` = conservative max dot to any point in that
  cell using the precomputed spherical cap for the cell.
- If `ub(cell) <= threshold[qi]` skip scanning that cell for that query in the ring pass.
- Similarly for tail: if `ub(cell) <= security[qi]` skip entirely.

This reduces `pk_ring` work and shrinks candidate lists, which indirectly reduces selection/sort.

### Path E: Tail sketch instead of full tail vectors (top-M tail)

Observation: tail exists to supply “safe but below threshold” candidates, but many queries may
only need a handful to prove termination.

**Idea:** during the first ring scan, maintain a tiny “best tail” buffer per query:

- Track top `M` tail candidates (e.g. `M = k1` or `k1 + slack`).
- Record whether tail was truncated.
- If a query consumes tail and `M` is enough, no second scan is needed.
- Only if a query needs deeper tail and `tail_truncated` is true, do a per-query full tail build.

This aligns with “approximate k is fine”: we are not trying to list the entire tail, just enough
to finish the typical cell.

### Path F: Region-level ordering (cell/bucket ordering) instead of per-point sorting

**Idea:** accept less granular termination checks and reduce sorting:

- **Cell ordering:** process the 9 cells (center + neighbors) in decreasing order of their per-query
  `ub(cell)`. Tighten `unseen_bound` at cell boundaries rather than per-point.
- **Dot bucketing:** bucket candidates by quantized dot (using ordered-u32 dot bits, not just
  exponent). Process buckets high→low; terminate at bucket boundaries.

Tradeoff:

- Potentially more clipping work (since you lose “next neighbor” tightness).
- Potentially much less `select_nth + sort` work and better cache behavior.

This requires re-deriving the termination bounds used in `process_cell`:

- while inside a region/bucket, bound must be conservative (e.g. bucket’s max),
- after exhausting a region/bucket, bound becomes the next region’s max.

### Path G: Radix sort / bucket sort as a replacement for select+sort

This is a more specific version of Path F:

- Store candidate keys (already include ordered dot bits).
- Radix/bucket them by a slice of bits chosen to preserve entropy near `dot≈1.0` (mantissa-heavy),
  not by only “high bits”.
- Emit buckets in order; optionally sort within a bucket only when needed.

Important considerations:

- “Emit unsorted buckets” is only viable if termination bounds are adjusted to terminate at bucket
  boundaries (or if per-neighbor termination checks are removed).
- If many candidates collapse into a single bucket (common near 1.0), you still end up
  sorting/partitioning that bucket, limiting the win.

### Path H: Approximate selection (“4 or 6 is fine”) and what it *can* mean safely

Approximate selection can help in two safe ways:

- **Approximate neighbor count:** we can choose to clip a few more or fewer neighbors than a fixed
  `k`; termination/correctness still relies on conservative unseen bounds.
- **Approximate thresholding:** we can use a heuristic cutoff to reduce candidate volume, as long
  as failure to terminate causes a conservative fallback (more packed work or kNN).

Approximate selection is *not* safe if it causes us to claim a too-strong `unseen_bound` without
actually exhausting candidates above that bound.

## Scaling With Density (2.5M → 4M)

### What changed and why it matters

From stage counts:

- Tail usage doubles (12% → 26.7%).
- Safe exhaustion and kNN fallback increase 5× (0.5% → 2.4%).
- Termination requires more neighbor planes to prove in many cells (histogram shifts up).

This suggests the main scaling risk is not “sorting is slower”, but “the algorithm spends more
time in tail and/or falls back to resumable kNN more often”.

### Recommended scaling levers (ordered by likely ROI)

1. **Eliminate wasted tail builds** (Path A) to stabilize `pk_fallback`.
2. **Increase `k1` (tail chunk size) at higher density** (Path B) to reduce small-chunk iteration.
3. **Reduce tail entry frequency by tightening the cutoff heuristic** (Path C/D/E).
4. **Reduce `safe_exhausted`**:
   - improve the r=1 packing so it finds enough safe candidates more often,
   - and/or conditionally expand the packed neighborhood when close to exhausting safe candidates.

### Grid density vs. work

Let `d` be points-per-cell (target density). For a fixed 3×3 neighborhood:

- Dot work per group is roughly `O(d^2)` (each query vs each candidate).
- Number of groups is `O(n/d)`.
- Total packed dot work is therefore `O(n*d)`.

Practical implication:

- Prefer tuning chunk sizes and pruning bounds before increasing density.
- If `safe_exhausted` rises with `n`, consider conditional neighborhood expansion (ring-2 / larger
  r) before globally increasing density.

## Directed kNN + Edgechecks: Using LocalId / Incoming Checks to Adapt Budgets

Directed filtering excludes earlier same-bin candidates from kNN and packed selection; later cells
obtain true earlier-neighbor adjacency via incoming edgechecks.

### What can be used safely as a signal

- `incoming_checks.len()` is an observed, per-cell signal for “how many earlier neighbors are
  already known”. This is a better control variable than `LocalId` alone.
- `LocalId` provides a prior: later locals are likely to have more incoming edges, but it is not a
  guarantee.

### Safe ways to exploit it

- **Budget adaptation:** reduce “how many additional ordered candidates to request” (packed and/or
  resumable kNN) based on incoming edge count, since seeds already contribute planes for bounding.
- **Chunk sizing:** if a cell is already seeded by many incoming edges and still not terminated,
  request a larger chunk next (to avoid many small iterations).

### What not to do

- Do not use edgecheck seeds to update distance-order-based termination bounds (`worst_cos`), since
  their dot products are not ordered and may be arbitrarily far.

## Recommended Next Experiments (Minimal-Risk Order)

1. **Instrumentation** (to guide tuning):
   - distribution of `chunk0_keys[qi].len()` and (when built) `tail_keys[qi].len()`
   - number of `next_chunk` iterations per cell in chunk0 and tail
   - correlation of tail usage with `incoming_checks.len()` and/or `LocalId` percentile
   - `safe_exhausted` breakdown by region (boundary cells vs interior, bins, etc.)

2. **Fix tail overreach** (Path A) and re-measure `pk_fallback` vs `tail_used`.

3. **Adaptive `k1` at 4M** (Path B) and re-measure:
   - `pk_partition/pk_sort` totals,
   - total packed time,
   - effect on `cells_used_knn` (should not worsen).

4. **Threshold tightening / ring pruning** (Path C + D):
   - choose a less ring-expanding cutoff,
   - add cell-cap upper-bound pruning in ring scans,
   - compare tail usage and candidate counts.

5. **Tail sketch** (Path E) to eliminate the second scan for most tail consumers.

6. Only after the above, evaluate **region/bucket termination** and **bucketing/radix** (Path F/G)
   as larger rewrites.

## Open Questions

- Is `safe_exhausted` at 4M primarily due to the strict r=1 neighborhood being insufficient, or due
  to directed filtering excluding near same-bin candidates that would otherwise be present?
- How often do tail-consuming cells need >`k1` tail candidates (deep tail), vs. just a few?
- How much does per-neighbor termination (within-chunk) contribute compared to per-chunk termination?
  (This informs whether region/bucket termination is viable.)
