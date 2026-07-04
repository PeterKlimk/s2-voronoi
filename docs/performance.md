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
