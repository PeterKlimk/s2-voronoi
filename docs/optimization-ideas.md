# Optimization ideas ledger

Performance ideas with status and evidence — including negative results, so
they are not re-tried. Roadmap priorities live in `docs/todo.md`; this is
the perf-specific backlog. Measurement rules: paired interleaved runs
(`scripts/bench_build.sh` + `bench_run.sh`), single-threaded for stability,
on a quieted machine; multithreaded numbers from different sessions are not
comparable (observed drift exceeds 30%).

## Open ideas

### Live within-bin edge repair (structural) — assessed 2026-06, deprioritized

Today every unresolved shared-edge mismatch — including those between two
cells of the *same bin* — accumulates into the shard output and is repaired
after assembly by `edge_reconcile` (a global pass over the final arrays with
a full-size union-find). Within-bin mismatches could instead be repaired
*during* the bin's build, while both cells' data is cache-hot and indices
are still shard-local; the post-pass would then handle only cross-bin
cases.

Assessment (2026-06 discussion; see engineering-findings #13): deprioritized
in favor of cheaper/safer steps, because (a) the repair would mutate
already-emitted cells in the shard CSR, (b) disputed endpoint vertices can
be owned by *other* bins (DEFERRED slots, no index or settled position
mid-build), so part of the repair moves to assembly anyway, and (c) a
two-phase repair needs a composition proof with the cross-bin pass — exactly
the proof P5 (consistency by construction) would make moot by removing the
mismatches at the source. Sequence agreed instead:

1. ~~Deterministic coverage net for all detection/repair paths~~ **Done**
   (`tests/edge_repair_net.rs`, origin-tagged records).
2. ~~Sparse union-find in `reconcile_unresolved_edges`~~ **Done** (dense
   init was 3-4.7ms at 2M ST on every run incl. defect-free; now ~0).
3. ~~Surgical O(defects) in-place patching~~ **Done**: affected cells found
   via vertex-key triplets (a vertex `(A,B,T)` appears only in cells A, B,
   T), spans shrink in place, stale index-buffer slots never read. The old
   rebuild survives as the differential oracle (`RepairApply::Rebuild`,
   env `S2_EDGE_REPAIR_REBUILD=1`), with unit-level and full-pipeline
   differential tests asserting identical per-cell sequences. Measured on
   the defect-bearing 2M seed-1 run (ST): rebuild 382ms -> in-place
   0.06ms; defect-free runs unchanged (~0).

Correction to the old note: the "8x8 cocircular lattice / ~112 unresolved"
figure presumably referred to the planar exact-lattice fixture (the only
8x8 lattice in the tests; the planar pipeline has no defect report to
re-verify against). On the sphere, synthetic degeneracy produces zero
unresolved edges and real defects appear only at multi-million scale
(~1-20 per run).

### Weld redesign: detect during scatter + in-place compaction

The planar weld already reuses the spatial grid as the detector
(`PlaneGrid::collect_pairs_within`, ~30ms at 1M vs the sphere's old ~110ms
standalone pass).

**Sphere port done (2026-06)**: the query grid is built on the raw points
and doubles as the weld detector (`CubeMapGrid::collect_weld_pairs` —
per-cell pairwise scan plus wall-plane-gated 3x3 neighbor checks, verified
against brute force across resolutions), and on welds the grid's point
arrays compact in place (`compact_welded`, bit-identical to a fresh build
on the effective points, test-pinned). The standalone quantized-key
detector survives only for `MergeWithin` radii above the grid-adjacency
bound (1/(16*res)). Measured at 2M ST: preprocess 378ms -> ~45ms (~8x; the
par-sort is gone). Resolution policy now sees the raw count (welds are too
few to shift it).

Remaining (planar side):

- Detect during the grid's scatter pass (compare each point against the
  already-scattered prefix of its cell) instead of a separate scan — note
  the sphere's parallel scatter writes cell segments concurrently, so for
  the sphere the post-build per-cell scan IS the parallel-friendly form;
  this refinement only fits the planar sequential scatter.
- Port the in-place CSR compaction to `PlaneGrid`/`PeriodicGrid` (they
  still rebuild on welds, and the rebuild path is the norm — uniform f32
  data contains sub-radius pairs at production scales: ~3 at 1M, ~15 at
  2M, birthday effect).

### HalfPlane epsilon without the per-clip sqrt

`HalfPlane::new_unnormalized` computes `eps = CLIP_EPS_INSIDE * sqrt(ab2)`
per clip attempt (~one sqrt per neighbor per cell). The epsilon only needs
to be scale-accurate, not exact — `|a| + |b|` is within sqrt(2) of the true
norm. Semantics-affecting (epsilon decisions shift; hp_eps is shared across
edge-check seeds), so it needs the same care as any tolerance change.

Measured 2026-06 (2M sphere ST, 3+3 pairs in both orders): inconclusive —
forward-order pairs showed 3-7% "wins" that reversed-order pairs exposed
as thermal drift; the combined data is a wash. Static estimate caps the
sqrt at <1% of total. Only worth retrying on a quiet box, and the eps
semantics shift (up to sqrt(2) larger) would need fuzz revalidation
regardless. Low expected value.

## Done (2026-06, micro-opt matrix screen — paired-confirmed)

- **Micro-opt stack merged** (~-36ms total at 500k ST, -120ms
  cell_construction at 2M, 12/12 paired rounds): extract-inline-checks
  (individually proven; deletes an O(vertices) per-cell diagnostic
  pre-pass) plus shell-frontier-scratch, packed-frontier-no-sentinel-fill,
  point-face-reciprocal, packed-tail-hoist (proven collectively as a
  stack increment). Full protocol and per-branch verdicts:
  docs/micro-optimization-matrix.md.
- **Methodological finding worth keeping**: per-binary code-layout offsets
  on this codebase are ~±10-15ms (1.3-2%) at 500k ST and are STABLE and
  SIGN-CONSISTENT across rounds — paired interleaving cancels machine
  drift but not layout luck; sign consistency alone cannot validate a
  micro-opt. Use diff-disjoint control branches to calibrate the floor,
  and stack sub-floor candidates to test their sum.
- Rejected with measurement: periodic-conditional-wrap (+11.8ms on its own
  periodic pipeline, 2/13), binning-cache-fuse, cell-to-face-u32,
  directed-cell-mode, frontier-cache-ordf32. Unproven (inside layout
  floor, not merged): chunk-array-loaders, clip-batch-slice,
  packed-center-tail-simd, packed-query-dot-cache, projection-max-r2,
  signed-dists-array-refs, preprocess-touched-reps.

## Done (2026-06, edge-repair / weld / stage-0 week — paired-confirmed)

Paired interleaved A/B (12-16 rounds, ST, pinned core, order rotated per
round) of week-start (8ee131c) vs post-stage-0 HEAD (c579966), on a BUSY box
— identical-config runs swung up to 2.5x, and the paired medians still
converged (the protocol works under noise; n must be large):

- **Default mode total: -287ms median at 2M (-7.6%, faster in 11/12 pairs);
  -94ms at 500k (-7.2%)** — the grid-integrated weld confirmed end-to-end.
- **Hot path (--no-preprocess): +18ms median at 2M (+0.5%), +10ms at 500k
  (+1.2%)** — the cost is the stage-0 canonicalization pass itself
  (~10ns/point scalar f64 sqrt+div), minus ~3ms from the sparse union-find.
  Parallelized after measurement (default builds pay ~nothing; ST keeps
  ~20ms at 2M).
- Repair pass (defect-bearing runs only): 382ms -> 0.06ms at 2M ST
  (phase-timing measurement, see edge-repair entries).

## Done (2026-06, batch round 2)

- **Planar TIMING_KV plumbing**: `PlanePackedTimings` is now the sphere's
  `PackedKnnTimings` (same stages, same `CellSubAccum` breakdown seam);
  both plane drivers lap knn/clip per batch, classify per-cell stages
  (chunk0/tail/expand/shell via batch sources, now carried by the periodic
  stream too), and both compute paths report phase timings + the KV line
  through the shared `TimingBuilder`. First yield: uniform 500k clips ~6.4
  neighbors/cell mean and only ~30 cells in 500k reach the shell takeover.

- **Vectorized N>8 bitmask clipper** (~2% sphere ST, torus
  neutral-to-positive; bit-identical, fingerprint held): 8-lane
  `fp::signed_dists_mask8` chunks replace the scalar distance loop.
  Measurement cautionary tale: short `-n 4` runs at 500k during a noisy box
  phase first measured this as a consistent ~1% loss and it was briefly
  rejected; re-measured with `-n 6` at 500k AND 2M (six clean pairs) it is
  a consistent win at both sizes. Borderline calls need big runs.

## Done (2026-06, periodic packed port)

- **Packed SIMD stage for the periodic pipeline** (~2x at 500k ST: 1600ms ->
  ~830ms, bringing the torus to within ~10% of the bounded plane): the
  bounded packed stage was genericized over a `PackedGeometry` trait (box
  enumeration, distance metric, outside-box certificate — monomorphized, the
  bounded instantiation compiles to the pre-trait code) and the periodic
  grid supplies wrapped boxes, an 8-lane minimum-image kernel
  (`fp::dist_sqs_wrapped`, lane math identical to `wrap_abs`), and
  wrapped-wall security bounds. Found and fixed in the process: re-clipping
  a bit-identical plane is NOT a no-op for the unbounded-seeded periodic
  builder (1e6 sentinel coordinates make clip-lerp drift exceed the clip
  epsilon; the packed -> takeover overlap re-emits, and the re-clip cut
  phantom slivers). The periodic driver now tracks clipped ids in the
  sorted seed-skip set. The bounded/sphere builders are seeded bounded
  (rect walls / gnomonic), where the no-op argument actually holds.

## Done (2026-06, "kernel contest" round; all bit-identical, paired-verified)

- **Small-N sorting networks always-on** (~4% total at 500k ST): existed as
  the off-by-default `packed_knn_sort_small` feature; promoted after 8/8
  paired rounds confirmed.
- **Vectorized small-clipper distance pass** (~4.5%): the per-clip f64
  signed-distance + inside test (paid on every clip attempt, including
  Unchanged) became two `f64x4` fmas via `fp::signed_dists_mask8`, and the
  entry/exit transition scan became two bit-tricks on the inside mask.
- **Fused hi/tail split** (~1%, plus deleted code): the threshold count
  model never used center-pass results, so thresholds are computed first
  and the center pass splits chunk0/tail directly; the post-hoc demotion
  pass and dead `min_center_dot` bookkeeping are gone.
- Net: ~9% at 500k ST, ~10-11% at 2M ST over the pre-contest baseline,
  fingerprint unchanged throughout. The plane inherits the clipper and
  network wins through shared code.

## Tried and rejected (do not re-try without new information)

- **Packed partial selection (unsorted batches + suffix-min bounds)**
  (2026-06, plane): dropping the per-chunk sort (select_nth only, consumers
  derive mid-batch bounds from suffix minima — sound for any order) measured
  a consistent 7-14% LOSS at 2M bounded ST across three paired rounds. The
  sort is not packaging: nearest-first clip order shrinks the polygon
  fastest, terminates earlier, and keeps every intermediate clip cheaper —
  worth far more than the sort costs (small-N networks made sorts cheap).
  Kept from the attempt: the NN contract harnesses now verify every
  reported batch distance against brute force per slot.

- **`fma` feature** (2026-06, measured on the Ryzen 3600): without
  `-C target-feature=+fma` codegen, `mul_add` lowers to a libm call and the
  feature is a 25-35% LOSS at 500k ST. With `-C target-cpu=native` it is
  statistically indistinguishable from the non-fma native build (mins 697.9
  vs 700.6ms over 4 paired rounds). Verdict: keep off by default; not worth
  a fingerprint move even on FMA hardware. Side-finding worth advertising:
  `target-cpu=native` alone is ~6% over the default build (745 -> 703ms)
  with no semantic change — documented in performance.md.

- **Ring-pass cap pruning** (per ring-cell × query cap bound to skip chunk
  dots): net loss — ring cells are *adjacent* cells, whose caps almost
  always straddle the tightened threshold, so the prune rarely fires and
  its per-(cell, query) cap evaluations are pure overhead.
- **PACKED_HI_BUDGET retune** (12/16/24 vs 32): 12 collapses (tail-build
  explosion); 16-32 flat within noise. Keep 32.
- **Planar `S2_BIN_COUNT` increase** (24/48/96 at 2M): no shard-balance win;
  the apparent planar MT-scaling gap was a measurement-window artifact
  (see docs/performance.md pairing note).
