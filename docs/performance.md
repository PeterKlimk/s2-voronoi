# Performance / benchmarking

This crate is structured so that per-cell work is embarrassingly parallel, and most cross-cell
coordination happens only during vertex deduplication.

## Useful features

- `--features timing`: prints timing breakdowns and optionally emits machine-readable `TIMING_KV`
  lines when `S2_VORONOI_TIMING_KV` is set.
- `--features qhull`: enables the convex hull backend for comparisons (slow).
- `--features simd_clip`: uses SIMD small-N clippers in the main dispatch.
- `--features fma`: prefers `mul_add` (can change results; may be slower on some CPUs).

## Environment knobs

- `RAYON_NUM_THREADS=1`: force single-threaded mode (useful for stable perf comparisons).
- `S2_BIN_COUNT=<n>`: override bin/shard count (defaults to ~2x threads).
- `S2_VORONOI_TIMING_KV=1`: enable `TIMING_KV ...` output (requires `timing` feature).

## Bench binaries

- `cargo run --release --bin bench_voronoi -- 100k 500k 1m`
- `cargo run --release --features timing --bin bench_voronoi -- 500k --no-preprocess`
- `cargo run --release --features qhull --bin bench_voronoi -- 50k --validate`

## Comparison scripts

For inter-commit comparisons:

- `./scripts/bench_build.sh --chain 6`
- `./scripts/bench_run.sh -s 500k -r 20 -m total`

The scripts default to pinned + single-threaded runs where possible.

