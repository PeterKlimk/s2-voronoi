# Optimization ideas ledger

Performance ideas with status and evidence — including negative results, so
they are not re-tried. Roadmap priorities live in `docs/todo.md`; this is
the perf-specific backlog. Measurement rules: paired interleaved runs
(`scripts/bench_build.sh` + `bench_run.sh`), single-threaded for stability,
on a quieted machine; multithreaded numbers from different sessions are not
comparable (observed drift exceeds 30%).

## Open ideas

### Live within-bin edge repair (structural)

Today every unresolved shared-edge mismatch — including those between two
cells of the *same bin* — accumulates into the shard output and is repaired
after assembly by `edge_reconcile` (a global pass over the final arrays with
a full-size union-find). Within-bin mismatches could instead be repaired
*during* the bin's build, while both cells' data is cache-hot and indices
are still shard-local; the post-pass would then handle only cross-bin
cases.

Value today is structural rather than hot (uniform inputs produce ~0
unresolved edges; the 8x8 cocircular lattice test produces ~112), but it
removes a serial global pass and becomes load-bearing if epsilon-repair
volume grows (clustered / near-cocircular inputs), and it is a prerequisite
for shrinking the post-assembly stage in general.

### Weld redesign: detect during scatter + in-place compaction

The planar weld already reuses the spatial grid as the detector
(`PlaneGrid::collect_pairs_within`, ~30ms at 1M vs the sphere's ~110ms
standalone pass). Two refinements remain:

- Detect during the grid's scatter pass (compare each point against the
  already-scattered prefix of its cell) instead of a separate scan.
- On welds, compact the existing CSR arrays with one linear sweep instead of
  rebuilding the grid (relevant because uniform random f32 data contains
  sub-radius pairs at production scales — ~3 at 1M, ~15 at 2M, birthday
  effect — so the rebuild path is the norm, not the exception).
- Port the grid-integrated design to the sphere, replacing the standalone
  quantized-key pass (~110ms at 2M per earlier measurements).

### Packed select: partial selection instead of full sort

`select_sort` sorts every emitted chunk so the consumer can use the next
emission as a mid-batch termination bound. Most cells terminate after ~8
neighbors, so sorted tails are wasted work. A partial-selection scheme
(`select_nth` + use the batch *minimum* as the certificate for the whole
batch) halves the sort work at the cost of a weaker mid-batch bound (later
termination, more clips). Semantics-affecting: changes emission order, so
the fingerprint moves and the tradeoff needs paired measurement plus a
policy decision.

### Vectorize the N>8 bitmask clipper

`clippers/bitmask.rs` (polygons above 8 vertices — rare, large-cell path)
still computes scalar distances; `fp::signed_dists_mask8` now exists and
chunks of 8 apply directly. Low priority: small share of clips.

### HalfPlane epsilon without the per-clip sqrt

`HalfPlane::new_unnormalized` computes `eps = CLIP_EPS_INSIDE * sqrt(ab2)`
per clip attempt (~one sqrt per neighbor per cell). The epsilon only needs
to be scale-accurate, not exact — `|a| + |b|` is within sqrt(2) of the true
norm. Semantics-affecting (epsilon decisions shift; hp_eps is shared across
edge-check seeds), so it needs the same care as any tolerance change.

### Planar TIMING_KV plumbing

`PlanePackedTimings` is a no-op shell with the sphere's call surface
already in place; the swap to real timing is mechanical and unlocks
planar sub-phase profiling (currently only the sphere has it).

### `fma` feature evaluation on FMA hardware

The `fma` feature is off by default ("may be slower on some CPUs") but the
reference Ryzen 3600 has hardware FMA. Changes results bit-wise (fingerprint
moves), so it is a declared-policy experiment, not a free win.

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

- **Ring-pass cap pruning** (per ring-cell × query cap bound to skip chunk
  dots): net loss — ring cells are *adjacent* cells, whose caps almost
  always straddle the tightened threshold, so the prune rarely fires and
  its per-(cell, query) cap evaluations are pure overhead.
- **PACKED_HI_BUDGET retune** (12/16/24 vs 32): 12 collapses (tail-build
  explosion); 16-32 flat within noise. Keep 32.
- **Planar `S2_BIN_COUNT` increase** (24/48/96 at 2M): no shard-balance win;
  the apparent planar MT-scaling gap was a measurement-window artifact
  (see docs/performance.md pairing note).
