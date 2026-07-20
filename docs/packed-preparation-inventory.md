# Packed Group Preparation Inventory

**Status:** source-shaped range-setup extraction accepted, 2026-07-20

This inventory covers `PackedKnnCellScratch::prepare_group_directed`. The routine is the hot
directed packed-query preparation path used by live cell construction. Its output deliberately
borrows the reusable scratch object mutably through `PreparedPackedGroup`, preventing another group
from overwriting prepared keys or generation state while a caller is consuming them.

## Program and ownership eras

The current routine has six ordered eras.

1. **Entry and generation.** Clear per-call timings, reject an invalid grid-cell id before any
   indexing or debug contract, validate the complete center-cell group in checked builds, and
   advance the nonzero generation stamp. Generation wrap clears all retained tail-ready stamps.
2. **Directed neighborhood setup.** Rebuild the center-plus-neighbor range list in grid neighbor
   order, classify each nonempty neighbor by bin and cell order, count eligible candidate work,
   reject hard or aggregate over-budget groups, and retain both eligible and all-ring counts for
   threshold policy.
3. **Security and reusable storage.** Borrow the center cell's SoA coordinate slices, compute one
   security floor per query through the interior-plane or boundary-cap path, then resize/clear the
   reusable key, cursor, tail, dense-band, and generation-stamp storage without shrinking retained
   inner allocations.
4. **Threshold selection and center scan.** Select the per-query high threshold from the security
   floor and candidate-count model. Scan the directed triangular center cell either through the
   ordinary SIMD chunks or the dense-band path. Dense-band takeover can lower the shared ring
   threshold, disable the ordinary tail, and raise the certified center bound independently for
   each query.
5. **Ring scan.** Traverse eligible neighbor ranges in their stored order, process full SIMD chunks
   in pairs, handle an odd full chunk and a padded remainder, and append candidates above each
   query's possibly dense-adjusted threshold.
6. **Observation and handoff.** Record key-storage and tail-demand facts, then return a
   `PreparedPackedGroup` that owns the exclusive scratch borrow, original group descriptor,
   generation stamp, and lazy-tail usage flag.

These eras are not freely reorderable. Candidate budgets depend on the directed range
classification. Security thresholds must exist before count-model selection. Threshold selection
must precede the center scan so high and tail candidates are split without a demotion pass. Dense
center handling can change the thresholds consumed by the ring scan. The final prepared borrow is
valid only after both scans and their observation reductions complete.

## Existing useful boundaries

- `PackedGroupInput` already correlates the center cell, bin, slot/local run, query count, and
  packed layout, and its checked-only contract verifies the complete slot-ordered center run.
- `PreparedPackedGroupStatus` makes whole-group fallback explicit; `PreparedPackedGroup` owns the
  reusable scratch borrow during query emission.
- Interior security calculation is already split into scalar finishing and eight-lane append
  helpers. Boundary-cap security remains a distinct helper.
- Dense-band radius and slot discovery belong to `CubeMapGrid`; lazy tail materialization and
  prefix emission already live in `scratch/emit.rs`.
- The center and ring loops encode distinct directed masks, chunk/remainder shapes, and threshold
  semantics. Their current flattening is intentional codegen, not missing generic iteration.

## Selected extraction

Extract only directed neighborhood range discovery and budget classification behind one private
method and one private summary:

```text
DirectedRangeSummary {
    center_soa_start: usize,
    center_soa_end: usize,
    ring_candidates_eligible: usize,
    ring_candidates_all: usize,
}
```

`collect_directed_ranges` receives the reusable scratch object, grid, and copyable group
descriptor. It clears and repopulates `cell_ranges`, preserving center-first and grid-neighbor
order. It performs the existing hard candidate and checked aggregate-work gates and returns `None`
for either budget rejection; otherwise it returns the center bounds and the two correlated ring
counts needed downstream.

The caller retains the invalid-cell gate, checked group contract, query count, generation-stamp
transition, timer creation, setup lap, and `SlowPath` conversion. In particular, a budget-rejected
valid group still consumes a generation exactly as it does now, and failed setup time remains
charged to the same timing field.

This boundary is useful because it owns one complete classification invariant:

- the first stored range is always the center cell;
- every later range represents one nonempty non-sentinel neighbor in grid order;
- same-bin earlier ranges remain recorded for the all-ring count model but are excluded from
  eligible work and scans;
- `ring_candidates_all` and `ring_candidates_eligible` are derived from the same finalized range
  list; and
- no later phase needs the temporary total-candidate count or repeats bin/local decoding.

Use default compiler inlining first. Add no `inline` attribute unless matched artifact and counter
evidence supports one isolated variant.

## Shapes deliberately excluded

- Do not create a broad preparation context borrowing the grid, group, timers, coordinate slices,
  thresholds, key vectors, and dense state. Their mutable lifetimes differ, and such a record would
  enlarge the alias surface across the SIMD loops.
- Do not extract or rewrite the indexed threshold-selection loop. A prior zip-based cleanup saved
  retired instructions and branches but repeatably worsened native cycles on Fibonacci and uniform;
  its compact indexed form is source-pinned.
- Do not extract, genericize, or unify the center and ring scans. The triangular directed mask,
  dense-band takeover, paired ring chunks, and two different remainder strategies are materially
  different kernels.
- Do not move dense-band selection into the range helper. It depends on security floors and mutates
  per-query thresholds, tail possibility, center bounds, key output, and takeover behavior.
- Do not combine the interior and boundary security paths. Their fallback and vectorization shapes
  are already isolated at the appropriate numerical seam.
- Do not change eager active-buffer clearing, retained inner capacities, or generation stamps.
  Generation-gated tail-buffer clearing was previously measured worse on ordinary inputs.
- Do not add a reason-bearing public fallback enum. The only caller action remains whole-group
  `SlowPath`; budget reasons are internal policy, not a new runtime state for downstream code.
- Do not extract final timing reductions alone. They enforce no new ownership or validity boundary
  and would only move a short observation epilogue.

## Semantic gate

The extraction must preserve:

- invalid-cell rejection before grid indexing, checked group validation, generation advancement,
  and timer start;
- generation advancement and setup timing for both valid accepted and budget-rejected groups;
- center-first range storage with `CrossBin` kind and exact grid-neighbor iteration order;
- skipping sentinel, self, and empty neighbor cells;
- one first-slot bin/local decode per retained neighbor cell;
- classification of same-bin cells strictly by neighbor cell id relative to the center;
- exclusion of same-bin earlier ranges from hard/aggregate eligible work while retaining them in
  the stored range list and all-ring count;
- hard-cap rejection before checked aggregate-work rejection, including overflow rejection;
- identical center bounds, ring counts, scratch capacities, later thresholds, candidate key order,
  frontiers, fallback decisions, and final diagrams; and
- the current zero-cost timing backend and feature-enabled timing schema.

Add a direct helper regression whose layout places every slot in one bin and whose selected center
has nonempty earlier and later neighbors. Pin center-first ordering, neighbor kinds, retained earlier
ranges, and the distinction between all-ring and eligible-ring counts. Existing invalid-cell,
aggregate-budget, brute-force ordering/bounds, dense, and pipeline suites remain the behavioral gate.

## Activity characterization

Timing-enabled single-thread 100k runs selected counter workloads. Their wall times are advisory on
the busy shared host and are not acceptance measurements.

| Workload | Grid state | Setup | Packed total | `cells_used_knn` |
|---|---|---:|---:|---:|
| Fibonacci seed 1 | res 26, max occupancy 36, not rebuilt | 0.9 ms | 41.6 ms | 0 |
| Uniform seed 1 | res 26, max occupancy 51, not rebuilt | 0.7 ms | 46.9 ms | 170 |
| Clustered seed 1 | res 26, max occupancy 1,366, not rebuilt | 0.7 ms | 335.7 ms | 12,026 |
| Mega seed 1 | res 268, max occupancy 315, rebuilt | 2.7 ms | 456.4 ms | 20,332 |
| Uniform seed 12345, 96 bins | res 26, max occupancy 47, not rebuilt | 1.0 ms | 67.9 ms | 161 |

`cells_used_knn` is a downstream mixed fallback count, not a direct count of preparation budget
rejections. The existing aggregate-work unit fixture supplies the exact active rejection contract;
clustered and mega provide broad dense/fallback codegen guardrails.

## Validation and performance gate

Run formatting and focused packed scratch/query tests, then the complete release, checked,
no-default-feature, and all-target/all-feature Clippy gates.

Build the dirty candidate and immediate parent together with native `tools` artifacts. In the
production build `prepare_group_directed` is currently folded into the cell-driver code shape, so
inspect the new helper, the relevant `build_cells_sharded_live_dedup` closure/body symbols,
aggregate sections, and file size. Then run seven interleaved single-thread hardware-counter pairs
for:

1. 500k Fibonacci, default bins — ordinary interior security and directed ranges;
2. 500k uniform seed 12345, default bins — scrambled ordinary path;
3. the same uniform workload with `VORONOI_MESH_BIN_COUNT=96` — altered cross-/same-bin range mix;
4. 100k clustered seed 1 — dense occupancy and mixed fallback pressure; and
5. 100k mega seed 1 — rebuilt grid, dense-band work, and budget/fallback pressure.

Use `--no-preprocess` for all pairs. Retired instructions and branches are primary; retain
switch/migration telemetry. Reject a material repeatable whole-build loss in any ordinary,
high-bin, dense, or rebuilt-grid regime. Differences around one basis point are practical noise
when opposite-direction counters, artifact size, and cross-regime evidence show no meaningful
performance loss. Run a quiet wall-clock confirmation only for a strong unexplained signal.

## Measured result and decision

Two extraction shapes were implemented and validated against source baseline `eb56662`. The first
was rejected; the source-shaped second form is retained. Its direct range-classification regression
and complete release, checked, no-default-feature, and all-target/all-feature Clippy gates pass.

The first shape used the returned center bounds for both query slices and the later center scan.
LLVM fully inlined the helper; the main cell-driver closure remained `0x1199` bytes, aggregate
`.text` shrank by 512 bytes, data and actual `.bss` were unchanged, and file size shrank by 720
bytes. Seven interleaved pairs produced these candidate/parent instruction and branch means:

| Workload | Instructions | Branches |
|---|---:|---:|
| 500k Fibonacci | `0.999834538` | `0.999676382` |
| 500k uniform seed 12345 | `0.999976891` | `0.999712105` |
| 500k uniform seed 12345, 96 bins | `0.999936075` | `0.999728022` |
| 100k clustered seed 1 | `1.001397086` | `0.999952846` |
| 100k mega seed 1 | `1.000127213` | `0.999910407` |

Every sample had zero context switches and CPU migrations. `#[inline(always)]` produced the same
artifact shape and reproduced the split: clustered measured `1.001398595` instructions and
`0.999956567` branches, while Fibonacci and mega remained at `0.999832850` / `0.999678887` and
`1.000126525` / `0.999910138`. The loss was therefore not call overhead.

The retained shape restores the original later `self.cell_ranges[0]` read while keeping the helper
and summary. That removes the clustered and mega instruction losses. Seven-pair instruction/branch
means are `1.000102288` / `0.999942805` for 500k Fibonacci, `1.000099301` / `0.999959540` for 500k
uniform seed 12345, `1.000091140` / `0.999953536` for the same uniform input at 96 bins,
`1.000010707` / `0.999988071` for 100k clustered, and `1.000004870` / `0.999983614` for 100k mega.
It adds 64 bytes of `.text` and 88 file bytes while leaving data, actual `.bss`, and the `0x1199`
driver closure unchanged. Every sample has zero switches/migrations.

The roughly 0.01% ordinary instruction increase is practically negligible, is paired with fewer
branches, and disappears in the dense and rebuilt-grid regimes. The named phase and direct
classification contract are worth that codegen displacement. The rebuilt retained artifact is
byte-identical to the measured source-shaped artifact, so no fresh wall-clock inference is needed.
Keep the original later center-range read as part of the accepted shape; the more compact first
form remains rejected.
