# Performance

Per-cell construction is embarrassingly parallel; the only cross-cell work is vertex
deduplication. Per-point cost is near constant in n, so the advantage over hull- and
triangulation-based libraries grows with input size.

## Reference numbers

Ryzen 3600 (6 cores), uniform input, minimum of repeats:

| n  | multithreaded |
|----|---------------|
| 1M | ~330ms |
| 2M | ~720ms |

Single-threaded, ~1.8s at 1M.

Peak resident memory per build is roughly linear, ~0.65 KB/point (500k ≈ 320 MB, 1M ≈ 660 MB,
2M ≈ 1.3 GB). The working set frees when the diagram drops. A process that builds many diagrams in
a loop will accumulate high-water RSS — glibc does not return freed arenas to the OS between
builds, amplified by per-thread rayon arenas. This is allocator behavior, not a leak; build in a
child process per job, set `MALLOC_ARENA_MAX=2`, or link jemalloc/mimalloc if it matters.

## Building for speed

`RUSTFLAGS="-C target-cpu=native"` is worth ~6% on the reference machine and is the main build
flag that matters. Run benchmarks in release.

## Running the benchmarks

The benchmark binaries need the `tools` feature:

```bash
cargo run --release --features tools --bin bench_voronoi -- 100k 500k 1m
```

Useful flags:

- `--dist {fib|uniform|clustered|bimodal|gradient|outlier|splittable|mega}` and `--dist-param` —
  non-uniform distributions exercise the density-adaptive paths that uniform input never reaches.
  Benchmark across a few of these, not uniform alone.
- `--validate` — compare against the convex-hull ground truth (slow; capped at 100k).
- `--no-preprocess` — skip welding (isolates construction cost).

## Knobs

- `RAYON_NUM_THREADS=1` — single-threaded, for stable comparisons.
- `VORONOI_MESH_BIN_COUNT=<n>` — shard count (default ~2x threads).
- `VORONOI_MESH_TIMING_KV=1` with `--features timing` — machine-readable phase timing.
- `VORONOI_MESH_GRID_DENSITY=<f>` / `VORONOI_MESH_PLANE_GRID_DENSITY=<f>` — spatial-grid target
  density (points per cell) for sweeps.

## Repair cold path

The post-assembly local repair fires only on defect-bearing inputs (rare; near-cocircular
clusters, e.g. the `mega` distribution) and is fast in the common case. One known cold path
remains: when the defect sits on the boundary of an extremely dense cluster (most points inside a
single grid cell), the repair's neighbor gather expands into the cluster and the repair can take
seconds to minutes at millions of points. The output contract is unaffected — the result is still
strictly valid or a clean error — only latency degrades. `RepairMode::Disabled` skips repair
entirely (residual defects then fail plain `compute` loudly).

## Comparing commits

The machine is noisy — per-binary code-layout shifts alone are ~1-2% at 500k single-threaded — so
compare commits with interleaved paired runs, not back-to-back batches, and treat sub-1% deltas as
noise:

```bash
./scripts/bench_build.sh --chain 6
./scripts/bench_run.sh -s "500k 2m" -d "uniform mega" --seeds "1 2 3" --csv out.csv
```

`bench_run.sh` sweeps sizes x distributions x seeds, runs the commits interleaved, and emits a
CSV. Prefer hardware counters (`perf stat`) over wall time for behavior decisions; wall time on
this class of machine drifts more than most single optimizations are worth.

For a first-pass hardware-counter comparison of the same built artifacts:

```bash
./scripts/bench_build.sh --chain 2
./scripts/bench_perf.sh -s 500k -d fib -r 9 --csv /tmp/bench_perf.csv
```

The counter runner rotates version order, defaults to one Rayon thread pinned to CPU 0, and emits
raw tidy CSV rather than prematurely aggregating results. Compare instructions first, then cycles,
branches/misses, and cache behavior. Use the included context-switch and CPU-migration counts to
identify contaminated samples. If counters do not resolve the decision, use `bench_run.sh
--converge` for longer paired wall-time runs.

### Counter calibration on the noisy reference box

A 2026-07 control run compared two 500k single-thread binaries with behaviorally identical release
code over nine paired rounds. Retired instructions and branches were stable to a few parts per
million. The paired cycles ratio had an approximately ±1.2% 95% interval; hardware cache counters
were noisier. Hardware branch misses showed a false-looking ~0.8% separation, while Cachegrind's
deterministic branch simulator differed by only 18 of ~704k predicted misses on a 10k run.

Accordingly, use retired instructions/branches as the primary structural signal, paired cycles as
the primary hardware-cost signal, and Cachegrind (`--branch-sim=yes`) or `llvm-mca`/`cargo asm` for
attribution. Treat hardware branch/cache misses as corroborating evidence rather than a standalone
verdict on this machine. Effects below the cycles noise floor should proceed to longer paired
`bench_run.sh --converge` runs.

### Resource-bound calibration

The timing feature reports `weld_pairs`, `weld_pair_capacity`,
`packed_keys_materialized`, `packed_key_capacity_peak`, tail possible/requested counts, ring-tail
rescan/dot counts, total/unrequested center-tail candidates, and total/unused high-threshold
`chunk0_keys` in `TIMING_KV`. Measurements leading to
the initial packed aggregate-work bound found a 500k uniform peak capacity of 6,464 keys versus
2,220,652 keys (~17.8 MiB of `u64` payload in one worker) for the clustered distribution. The 1M
query×candidate budget reduced the clustered peak to 1,188,540 keys (~9.5 MiB of allocator
capacity) and routes larger groups to the bounded shell fallback.

That fallback is intentionally a reliability tradeoff: at 100k clustered it cost approximately
23% more instructions and 15% more cycles; uniform work remained structurally neutral. The weld
pair budget added approximately 0.8% instructions to a 500k normal-preprocessing control, while the
paired cycles interval remained unresolved around no change. Revisit these thresholds only with
both peak-storage telemetry and end-to-end counter measurements; optimizing away the fallback must
not restore unbounded retained work.

The normal 100k uniform packed-bound comparison remained unresolved after the maximum 160 paired
wall-time rounds: candidate/base geometric mean `+0.3%`, 95% interval `[-1.1%, +1.8%]`, with the
candidate faster in 82/160 rounds. This is below the 1% decision resolution on the reference host;
do not describe it as either a performance win or a demonstrated regression.

Incremental shell-layer emission keeps whole sorting for layers up to 128 entries and otherwise
partition-sorts 64-entry prefixes on demand. At 100k single-threaded it reduced clustered retired
instructions by 3.05%, branches by 1.76%, and cycles by 3.52%; bimodal instructions fell 0.91% with
neutral cycles. Fibonacci instructions/branches were structurally neutral-to-lower, while its
0.83% cycle increase remained inside the reference host's noise interval.

Center-tail candidates are counted during the initial SIMD pass but their keys are materialized only
when that query requests the tail. At 100k clustered this reduced key materialization from 24.19M to
18.67M and peak capacity from 1,018,656 to 782,400 keys (about 1.9 MiB), while reducing instructions
2.05%, branches 2.88%, and cycles 2.83%. Fibonacci improved by 1.20% instructions, 1.29% branches,
and 3.55% cycles after the requested-query rescan was vectorized.

Shell-frontier pending entries use one packed `u64` whose high word orders the dot descending and
whose low word orders the slot ascending. This preserves the previous
`(Reverse(total-order dot), slot)` selection policy while avoiding tuple-key construction and
comparison. In a 20k mega run it reduced `knn_query` time by 18.4% and total time by 7.0% over 12
paired interleaved rounds; the query phase won all 12 pairs. A 10k mega counter pair reduced retired
instructions from 7.37B to 5.70B and cycles from 2.40B to 1.92B. At 50k clustered, `knn_query`
improved 23.6% over eight pairs. The 100k Fibonacci control has negligible shell time; its wall-time
result was unresolved, while counters improved about 1.1% instructions and 1.2% cycles.

Packed center passes test an emission mask before extracting SIMD lanes to a scalar array. The
change is deliberately small: at 2M clustered it reduced retired instructions by 0.07–0.08% in
three of three paired runs, while cycles remained neutral. A 1M Fibonacci control was structurally
neutral. Splitting the experiment showed that the center-prepare guards account for nearly all of
the saving; the requested-tail rescan guard alone was neutral. Retain this as a low-risk structural
cleanup, not as a demonstrated wall-time improvement.

The large-polygon clip output writer uses an increment-and-wrap step instead of `(i + 1) % n`.
Release code previously emitted an integer divide for every retained input vertex. An isolated
mixed-clip microbenchmark improved from 48.12 to 43.27 ns/call at N=9, 59.36 to 49.73 ns/call at
N=12, and 82.61 to 62.62 ns/call at N=20 (10%, 16%, and 24%). Fixed-work counters at 500k and 1M
mega were neutral because large-polygon output is rare relative to the full build. Retain this as a
clear local latency win; whole-program measurements describe its incidence, not its intrinsic cost.

### Open optimization queue

These are code-specific hypotheses from a 2026-07 subsystem scan. Each item is an isolated
experiment: preserve its stated semantics, measure the named regime, and move it either into the
measured results above or the retired list below. Do not bundle candidates before attribution.

High-priority, low-risk experiments:

- **B1 — fuse inverse-slot and AoS construction:** build `point_slots` while constructing
  `SlotPoint` records, eliminating one unconditional pass over `point_indices`. Measure grid-build
  instructions at 500k–2M across ordinary distributions.
- **O1 — fuse finite validation and canonicalization:** combine the two input sweeps while
  preserving the first-invalid index and error payload exactly. Measure single-threaded 500k–2M
  with preprocessing disabled, plus invalid inputs at the beginning, middle, and end.
- **O2 — non-atomic low-incidence scan for one worker:** use plain `u32` counts when Rayon has one
  worker and retain the atomic path for parallel builds. Measure the always-on repair gate in
  single-threaded ordinary and defect-bearing inputs.

Promising workload-specific experiments:

- **K2 — specialize shell scanning by cell mode:** remove per-slot global-id loads and mode checks
  from directed center and non-center loops. Re-run the full directed/unrestricted nearest-neighbor
  contracts and measure shell-heavy clustered/mega inputs.
- **K3 — vectorize interior security thresholds:** evaluate the four cell-wall planes in eight-query
  chunks while preserving dot association and nonfinite fallback behavior. Measure
  `packed_security` and compare thresholds bit-for-bit.
- **C2 — avoid padded small-clip stores:** the ubiquitous N=3/4 path currently writes four unused
  f64 zeros before escalation examines only live lanes. Any uninitialized representation must prove
  all feature combinations, including `p5_shadow`, read only initialized lanes.
- **C3 — use a four-lane large-clip tail:** for polygon lengths with a remainder at most four, avoid
  evaluating dead lanes. Measure N=9–12 and N=17–20 microbench cases before end-to-end testing.
- **C4 — unswitch batch-source handling:** hoist the invariant packed-tail/chunk/shell source match
  out of the per-neighbor loop. Keep bounds checks unless a separate safety argument and counter
  result justify removing them.
- **D1 — indexed incoming edge-check lookup:** retain linear search for small cells and test a
  reusable tiny map or sort/merge above a measured incoming-count threshold. Preserve first-index
  and duplicate semantics; measure high-degree, clustered, mega, and bin-count sweeps.
- **D2 — reserve aggregate assembly vectors:** sum shard lengths before appending unresolved edges,
  overflow checks, and deferred slots. Measure allocation counts, cycles, and peak RSS at high bin
  counts; reject if the reservation merely increases simultaneous memory.
- **D3 — compact high-degree consumed flags:** replace the per-cell byte vector above 64 incoming
  checks with reusable bit words. First instrument activation frequency; ordinary cells should stay
  on the existing inline mask.
- **B2 — branchless `uv_to_st` sign selection:** the sign branch is balanced and remains in release
  assembly; an `abs` plus bit-select prototype was bit-identical for finite f32 inputs. Require
  paired branch, instruction, and cycle results because the added arithmetic may cancel the win.
- **B3 — collapse dense-index build passes:** accumulate all axis ranges in one scan, hoist the
  chosen coordinate slice, and materialize sorted slots/coordinates together. This is dense-only;
  measure outlier/mega and synthetic cells above the 512-point threshold.
- **O3 — reuse reconciliation segment scratch:** two temporary vectors are allocated per unresolved
  record. Reuse two buffers per reconciliation round and measure allocations on high-degree defect
  fixtures; the clean path must remain unchanged.

Lower-confidence cleanup candidates, to attempt only with structural counters or activation data:

- Remove the redundant per-cell output-buffer clear when all success writers clear before use.
- Reserve live-dedup owned vertices nearer the observed ~2 vertices/generator instead of 6, while
  tracking reallocations and RSS on irregular inputs.
- Unroll weld wall-proximity tests without changing f64 arithmetic or pair-budget behavior.
- Precompute standalone large-`MergeWithin` wall proximity; this does not affect default welding.
- Replace the high-degree edge-check spill byte vector only if instrumentation shows it is live.
- Consider unchecked endpoint/owner-bin access only after a measurable bounds-check cost and a
  complete invariant audit; a sub-noise win does not justify converting a panic into possible UB.

### Retired experiments

Do not broadly retry these without a materially different design or workload:

- Per-(ring cell, query) spherical-cap pruning: adjacent caps rarely prune; measured net loss.
- Packed-to-shell attempted-slot filtering: low duplicate coverage and extra branching.
- Scalar shell dot-only SIMD: measured 6.5–8.5% slower.
- Lower grid target density: density 24 beat 16 by 4.8–7.1%.
- Packed partial-selection rewrite: measured 7–14% loss at 2M.
- Whole-ring packed bound skipping: neutral or worse outside a narrow dense case.
- Local packed radius-2 optimization: no winning regime; removed end to end.
- Eager/adaptive local ring-tail batching: only 3.8–9.6% of queries requested tails across 100k
  fib/uniform/clustered/bimodal, while productive lazy rescans were about 1.5% of 500k clustered
  runtime. Batching only useful requests requires a traversal redesign.
- Lazy recomputation of retained high-threshold `chunk0_keys`: despite 86.4% unused keys on 100k
  clustered, 75,653 later requests rebuilt 15.75M keys, costing 28.3% instructions, 65.1% branches,
  and 29.8% cycles. Keep the retained keys.
- Dense-band eligibility before the raw candidate cap: admitting actual band work within the same
  budget regressed a 5k cap by 0.7% instructions, 1.6% branches, and 1.4% cycles; a 10k cap
  wall-time check was about 7% slower.

Group-wide shell takeover batching is not an isolated query optimization in the current pipeline.
Same-bin cells are serialized because earlier cells emit live edge checks that seed and reconcile
later cells. Sharing traversal and emission across a group would therefore require a corresponding
stitching/scheduling redesign; revisit it only as that larger architectural change.
