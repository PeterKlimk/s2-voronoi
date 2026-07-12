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

Generator finiteness validation and unit-vector canonicalization share one chunked traversal. The
fused pass leaves invalid values untouched, reduces chunk-local first-invalid indices to the same
minimum global index as the old validation pass, and uses the identical per-point f64 normalization
arithmetic. At 2M single-threaded it reduced retired instructions by 0.18% on Fibonacci and 0.16%
on uniform input in all three paired runs. Cycles improved 3.5–4.4% on Fibonacci and 2.3–3.9% on
uniform. The same instruction reduction was visible in the default parallel build.

The always-on low-incidence repair trigger uses plain `u32` counters when the active Rayon pool has
one worker, avoiding atomic increments and the parallel final scan. Pools with more than one worker
retain the atomic implementation. At 2M single-threaded it reduced retired instructions by 0.067%
on Fibonacci and 0.061% on uniform input, with effectively identical counts in all three paired
runs. Cycles were too noisy to resolve; the default parallel control was structurally neutral.

The N=3/4 small clipper keeps its four SIMD distances in a four-element array instead of padding an
eight-element array with four zero stores. Escalation receives only the initialized slice; N=5–8
retains the full eight-lane representation. Release code shrank by 208 bytes and no longer contains
the two padded 16-byte stores. At 2M single-threaded, retired instructions fell about 0.062% on
Fibonacci and 0.061% on uniform input in all three pairs. Fibonacci cycles improved in all pairs;
uniform cycles were neutral. The default, microbench, `p5_shadow`, and escalation feature paths all
share the initialized-slice invariant.

Interior packed-kNN security thresholds evaluate each of the four boundary planes across eight
queries with `PointChunk8`. Lane minima remain scalar and follow the original plane order, so the
wide and scalar backends produce identical threshold bits; remainder and boundary-cell paths remain
scalar. At 2M single-threaded, retired instructions fell about 0.058% on Fibonacci and 0.053% on
uniform input in all three pairs. Cycles were unresolved. The wide and `simd_scalar` 100k backend
fingerprints remained identical.

Cell construction dispatches each exact neighbor batch once to a packed or shell-specialized loop.
The packed loop marks every occurrence; the shell loop performs insertion-based deduplication. This
removes the invariant source match from every candidate while preserving source-specific bounds,
timing, tracing, and termination. At 2M single-threaded Fibonacci it reduced retired instructions by
about 0.096%; at 500k clustered it reduced them by about 0.206%, consistently across three pairs.
Cycles were neutral-to-lower overall.

The quadratic cube projection computes one absolute-value square root and branchlessly selects the
positive or negative result. It is bit-identical to the previous finite-input branches, including
signed zero, subnormals, and grid-boundary-adjacent values. At 2M single-threaded it added about
0.05% retired instructions but removed about 0.06% branches. Cycles improved in all three Fibonacci
pairs and two of three uniform pairs, with the third uniform pair neutral; hardware branch misses
were lower in all six pairs. This is a latency win from removing a balanced sign branch, not an
instruction-count optimization.

Assembly pre-sums shard bookkeeping lengths and reserves exact aggregate capacity before appending
overflow checks and deferred slots; unresolved edges reserve only when nonempty. At 500k Fibonacci
with 96 bins this removed eight growth reallocations from each active vector, eliminated about
3.14 MB and 1.70 MB of recopied overflow/deferred payload, and reduced final capacities from
79,936/54,656 to the exact 61,504/42,360 entries. Fixed-work instructions and peak RSS were neutral;
cycles were neutral-to-lower. This is an allocation/capacity win, not a demonstrated total-time win.

Dense-cell index construction computes all three coordinate ranges in one traversal, selects the
sort coordinate slice once outside the comparator, and materializes sorted slots and coordinates
together. Exact baseline-equivalence tests cover axis ties, equal and nonfinite coordinates, float
bits, ordering, and band queries. Fixed-work 500k outlier and mega counters were neutral because
dense-index construction is a small cold portion of the build. Retain this as a strictly local
pass/branch reduction, not as a demonstrated end-to-end improvement.

### Open optimization queue

These are code-specific hypotheses from a 2026-07 subsystem scan. Each item is an isolated
experiment: preserve its stated semantics, measure the named regime, and move it either into the
measured results above or the retired list below. Do not bundle candidates before attribution.

Promising workload-specific experiments:

- **D3 — compact high-degree consumed flags:** replace the per-cell byte vector above 64 incoming
  checks with reusable bit words. First instrument activation frequency; ordinary cells should stay
  on the existing inline mask.
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
- Fusing the inverse point-to-slot map with `SlotPoint` AoS construction: at 2M it did not reduce
  retired instructions on Fibonacci or uniform input and was slightly higher in all Fibonacci
  pairs. The separate pass likely keeps the random inverse-map stores out of the multi-stream AoS
  construction loop; the candidate was reverted.
- Specializing shell scans by cell mode and slot order: the full version reduced 500k clustered
  instructions by 1.04% but increased 500k mega instructions by 0.99%. Restricting specialization
  to the directed center suffix still split −0.95% / +0.80%; requiring at least eight rejected
  prefix slots split −0.94% / +0.80%. Candidate counts and timing telemetry were identical, so this
  is a genuine path/codegen tradeoff rather than changed work. All variants were reverted rather
  than introduce a distribution-sensitive heuristic.
- Using a four-lane tail for large-clip remainders 1–4: N=9/12/20 mixed-clip microbench results were
  neutral, and 500k mega instructions improved only about 0.005% while cycles were slightly worse
  in all three pairs. The saved dead-lane arithmetic does not repay the remainder dispatch in the
  production path; the candidate was reverted.
- Replacing linear incoming edge-check lookup with a tiny index: a 500k sweep over Fibonacci,
  clustered, and mega at 6 and 96 bins found 99.55–99.93% of cells had at most eight incoming
  checks. Linear scans averaged only 2.4–2.7 contiguous comparisons per lookup. Cells above eight
  accounted for 0.49–2.70% of comparisons; above sixteen accounted for at most 1,533 comparisons
  in an entire run. No duplicate incoming keys occurred. Map setup cannot repay that activation, so
  D1 was retired without implementation. The same sweep saw no cells above 64 incoming checks, so
  the compact high-degree spill candidate remains fixture-only rather than production-motivated.

Group-wide shell takeover batching is not an isolated query optimization in the current pipeline.
Same-bin cells are serialized because earlier cells emit live edge checks that seed and reconcile
later cells. Sharing traversal and emission across a group would therefore require a corresponding
stitching/scheduling redesign; revisit it only as that larger architectural change.
