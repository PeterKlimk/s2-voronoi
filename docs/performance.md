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
