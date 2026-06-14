# Performance / benchmarking

This crate is structured so that per-cell work is embarrassingly parallel, and most cross-cell
coordination happens only during vertex deduplication.

## Useful features

- `--features timing`: prints timing breakdowns and optionally emits machine-readable `TIMING_KV`
  lines when `S2_VORONOI_TIMING_KV` is set.
- `--features qhull`: enables the convex hull backend for comparisons (slow).
- `--features simd_clip`: uses SIMD small-N clippers in the main dispatch.
- `--features fma`: prefers `mul_add` (changes results bit-wise). Measured a wash on FMA
  hardware under `-C target-cpu=native`, and a large LOSS without `+fma` codegen (libm
  fallback) — see the rejected list in optimization-ideas.md. Prefer plain
  `RUSTFLAGS="-C target-cpu=native"`, which alone is worth ~6% on the reference Ryzen 3600.
- `--features tools`: enables benchmark/utility binaries (they are not built by default).

## Environment knobs

- `RAYON_NUM_THREADS=1`: force single-threaded mode (useful for stable perf comparisons).
- `S2_BIN_COUNT=<n>`: override bin/shard count (defaults to ~2x threads).
- `S2_VORONOI_TIMING_KV=1`: enable `TIMING_KV ...` output (requires `timing` feature).
- `S2_VORONOI_PLANE_GRID_DENSITY=<f>`: override the planar grid's points-per-cell target
  (default 16; sweep data in `src/policy.rs`).

## Bench binaries

- `cargo run --release --features tools --bin bench_voronoi -- 100k 500k 1m`
- `cargo run --release --features tools,timing --bin bench_voronoi -- 500k --no-preprocess`
- `cargo run --release --features tools,qhull --bin bench_voronoi -- 50k --validate`
- `cargo run --release --features tools --bin bench_plane -- 100k 1m 2m -n 6 --validate`
- `cargo run --release --features bench_voronoice --bin bench_plane -- 1m 2m -n 6 --voronoice`
  (head-to-head against the `voronoice` crate on identical inputs)

## Planar reference numbers (Ryzen 3600, min of repeats, uniform)

| n | plane (MT) | sphere (MT) | voronoice |
|----|-----------|-------------|-----------|
| 1M | ~430ms | ~330ms | ~1.4s |
| 2M | ~1.0s | ~720ms | ~3.5s |

Single-threaded the two geometries are at parity per point (~1.8-1.9s at 1M); multithreaded
they are within ~5-10% of each other when measured in interleaved A/B rounds (long benchmark
sessions drift the machine by far more than that — always pair the runs you compare). The
voronoice gap grows with n (their delaunator core is O(n log n); per-point cost here is near
constant).

## Comparison scripts

For inter-commit comparisons:

- `./scripts/bench_build.sh --chain 6`
- `./scripts/bench_run.sh -s 500k -r 20 -m total`

The scripts default to pinned + single-threaded runs where possible.

## Policy profiling

When evaluating heuristic changes, do not rely only on total time. Prefer timing-enabled runs that
also show how work moved between the packed stages and the shell-expansion takeover.

Useful commands:

- `S2_VORONOI_TIMING_KV=1 cargo run --release --features tools,timing --bin bench_voronoi -- 100k --no-preprocess`
- `S2_VORONOI_TIMING_KV=1 cargo run --release --features tools,timing --bin bench_voronoi -- 100k --no-preprocess --packed-expand-r2`
- `S2_VORONOI_TIMING_KV=1 cargo run --release --features tools,timing --bin bench_plane -- 500k`
  (the planar pipelines emit the same KV schema; add `--periodic` for the torus)
- `./scripts/bench_build.sh --timing HEAD`
- `./scripts/bench_run.sh -s 100k -r 5 -c 1 -m total -- --packed-expand-r2`

Counters to watch:

- `cells_used_knn`
- `cells_packed_tail_used`
- `cells_packed_expand_r2_used`
- `packed_tail_builds`
- `packed_expand_r2_builds`
- `packed_expand_r2_cap_skips`
- `packed_expand_r2_scan_ms`
- `packed_expand_r2_select_ms`

For the current policy surface and change rules, see `docs/policy.md`.

## Grid density tuning

- `S2_VORONOI_GRID_DENSITY=<f>`: override the query-grid target density
  (points per cell) for sweeps. The default is a known-imperfect constant; the
  optimum varies with point count and distribution (see docs/todo.md P3.2).
- `./scripts/sweep_grid_density.sh`: density x size sweep emitting `TIMING_KV`
  lines. Watch `neighbors_total` (mean neighbors before termination =
  total / n), `grid_res`, `grid_max_occ`, and `grid_rebuilt` alongside
  `total_ms` when fitting.
- Catastrophically concentrated inputs (Σocc²/n over `GRID_REBUILD_SUMSQ_PER_N`,
  i.e. the dense region is the majority of points) trigger a one-step,
  memory-bounded occupancy-feedback rebuild; `grid_rebuilt=1` in `TIMING_KV`
  marks affected runs. Modest clusters do NOT trigger it — a global re-grid
  de-tunes the background and is a net pessimization there (see the occupancy
  rebuild re-calibration in docs/optimization-ideas.md).

## Memory

- Per-build peak RSS is roughly linear in point count, ~0.65 KB/point on
  the Ryzen 3600 reference (measured: 500k≈320 MB, 1M≈660 MB, 2M≈1.3 GB,
  3M≈2.3 GB). A single build's working set is freed when the diagram is
  dropped.
- **Long-running processes that build many diagrams**: glibc does not
  return freed arenas to the OS between builds, so a process that loops
  over many builds accumulates a high-water RSS (amplified by per-thread
  arenas under rayon and fragmentation across mixed allocation sizes) that
  can climb far above any single build's peak. This is a glibc allocator
  behavior, not a leak in the crate. Mitigations, in order of preference:
  build in a child process per job (the OS reclaims fully on exit — see
  `scripts/robustness_campaign.sh` for the pattern); set
  `MALLOC_ARENA_MAX=2`; or link a trimming allocator (jemalloc/mimalloc).
  One-shot CLI usage (one build, then exit) is unaffected.
