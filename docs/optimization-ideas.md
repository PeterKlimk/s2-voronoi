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

### HalfPlane epsilon without the per-clip sqrt — DONE (2026-06-14, free after the strict flip)

`HalfPlane::new_unnormalized` computed `eps = base_eps * sqrt(ab2)` per clip
attempt (~one sqrt per neighbor per cell). The original idea was to
approximate the norm by `|a| + |b|` (within sqrt(2)); that was rejected as
low-EV — semantics-affecting (eps shifts up to sqrt(2), needs fuzz
revalidation; hp_eps is shared across edge-check seeds) and measured a wash
at 2M sphere ST (forward-order 3-7% "wins" were thermal drift; static
estimate <1% of total).

The strict-rule flip changed the calculus. With `CLIP_EPS_INSIDE = 0.0` and
`PLANE_CLIP_EPS_INSIDE = 0.0`, `base_eps` is 0 in production, so
`eps = 0.0 * sqrt(ab2) = 0.0` *identically* — the sqrt is provably dead, not
approximated. `new_unnormalized_base_eps` now skips it when `base_eps == 0.0`
(the only nonzero base_eps is the p5_shadow eps-override probe, which keeps
the exact path). Multiply-by-zero is exact, so output is bit-identical (full
correctness suite + p5_shadow lib green); it also removes a `0.0 * inf = NaN`
trap on a pathological huge-normal plane. Same small (<1%) prize as before
but now ZERO risk and no revalidation — batched for the next quiet-box
perf-test of the commit log.

### Optional sub-index for very dense grid cells — assessed 2026-06, low priority

When a grid cell holds far more generators than the uniform target, the
per-query neighbor scan inside that cell does more work. The idea: detect
over-occupied cells and give them an optional finer (nested) index so dense
clusters query as cheaply as uniform regions.

The motivating measurement is also the argument against it — degradation
with local density ratio is strongly sub-linear and bounded:

| density ratio | tess time | vs uniform |
|---|---|---|
| 1:1 (uniform) | 2,892 ms | 1.00x |
| 6.7:1 | 3,772 ms | 1.30x |
| 23.9:1 | 3,907 ms | 1.35x |
| 50:1 | 4,216 ms | 1.46x |

Even a 50x density ratio costs only 1.46x. The existing grid (plus the
occupancy-feedback rebuild, one step, memory-bounded) already absorbs
density gracefully, so a sub-index would add a hot-path branch and a build
cost to reclaim at most ~30-46% on inputs that are (a) rare and (b) often
limited by something else first — the 64-vertex clip budget and welding
both bite in very dense clusters before query cost dominates. Worth
revisiting only if a real target workload is dominated by extreme,
persistent local-density contrast; otherwise the complexity is not paid
for. Lean reject.

Blunter alternative (also lean reject): a globally denser grid selected
from a user "is uniform / is clustered" hint, instead of a per-cell
sub-index. Cheaper to build but pays everywhere — non-uniform inputs have
sparse regions too, so a uniformly denser grid wastes build time and
memory on empty cells to help a few hot spots.

Caveat: the SPHERE has an occupancy-feedback rebuild
(`grid_occupancy_rebuild_resolution`, policy.rs) — a one-shot global re-grid,
memory-capped to O(n) cells — plus the `S2_VORONOI_GRID_DENSITY` knob. Its
trigger was **re-calibrated 2026-06-14** (see "Occupancy rebuild
re-calibration" below): it now fires only on catastrophic concentration
(`Σocc²/n` over threshold), not on any modest cluster. The PLANE has no
feedback rebuild. **Porting it to the plane is now LOWER priority**: the
evaluation showed a global re-grid only ever helps when the dense region is
the majority of points (where it rescues an otherwise-infeasible build);
for the moderate-density cases a plane port would target, OFF degrades
gracefully and a re-grid is a net pessimization. So the plane port is only
worth it if a real majority-concentration planar workload appears. The
sub-index ("punch 1" local refinement) is SYNERGISTIC with the rebuild, not a
replacement — it handles local hotspots and the residual over-full cells a
global re-grid can't tame, while the rebuild handles globally-dense inputs;
build both if dense-cell cost ever becomes a real target.

## Occupancy rebuild re-calibration — DONE (2026-06-14)

The occupancy-feedback rebuild (added 2026-06-11, `1e1dfcc`) was validated on
the wrong metric — its integration test pinned *occupancy reduction*, never
*wall-time*. A deterministic occupancy + feasibility sweep across density-
contrast distributions showed the original `max_occ > 16×target (=384)`
trigger fires far too eagerly and is a **net pessimization in every case it
fired** on non-extreme inputs:

| distribution | max_occ | pts-in-over frac | Σocc²/n | rebuild ON vs OFF |
|---|---|---|---|---|
| outlier (1 pile) | 503 | 0.0005 | 26 | 1.5× SLOWER |
| fewclusters (8) | 632 | 0.005 | 28 | 2.4× SLOWER |
| splittable (387 cells) | 1125 | 0.28 | 244 | 9× SLOWER |
| mega-f0.2 | 7079 | 0.20 | 1237 | 1.6× SLOWER (OFF 4.5s) |
| mega-f0.3 | 10621 | 0.30 | 2753 | RESCUE (OFF >60s → ON 10s) |
| mega-f0.8 | 28205 | 0.80 | 19449 | RESCUE (OFF ∞ → ON 12s) |

Root cause: a single *global* resolution can't refine a hotspot without
de-tuning the background — the rebuild collapsed mean density 24→4 to chase a
minority of hot cells. It only wins when the dense region is the **majority**
(background negligible, giant cells otherwise O(occ²)-infeasible).

**Fix**: trigger on `Σocc²/n` (the candidate-scan work proxy = cost of NOT
rebuilding), not max occupancy. Crossover is sharp and single-variable:
rebuild is harmful at Σocc²/n=1237, essential at 2753 (point-mass fraction
does NOT separate these — splittable 0.28 fine vs mega-f0.3 0.30 hangs).
Threshold set to **2000** (`GRID_REBUILD_SUMSQ_PER_N`). Naturally skips the
single-giant-cell-in-uniform case (low Σocc²/n → that's a local-index
problem, not a re-grid one). Output is unaffected (the grid is only a
candidate index; the kNN certificate finds the same neighbors at any
resolution) — this is purely a build-time change. Timing magnitudes measured
on a noisy box; the *deterministic* occupancy crossover and the
feasibility (finishes / times out) signal carry the conclusion.

Follow-up still open — "punch 1", SYNERGISTIC with this rebuild (not a
replacement): a local sub-index for over-full cells (a local re-grid that
never de-tunes the background). The two compose along different axes — the
rebuild gets the *global* resolution right when the bulk is dense; punch 1
mops up the *residual* over-full cells a single global resolution can't tame,
including the memory-capped case (e.g. mega at 1M, where even the rebuild
can't fully fix the giant cells) and minority hotspots the rebuild now
correctly skips. Rebuild alone fails those; punch 1 alone would sub-index
most cells on a majority-dense input where one global re-grid is cheaper.
**Full design: `docs/dense-cell-subindex-design.md`** (costless occupancy-
gated fast-path branch, ambient-3D kd-tree, best-first stream bounded by the
ring certificate). Build it when dense clusters become a real workload.

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
