#!/usr/bin/env bash
# Sweep the query-grid target density across input sizes to fit the density
# curve for docs/todo.md P3.2. Emits TIMING_KV lines (note neighbors_total /
# neighbors_max / grid_res / grid_max_occ for the explanatory model).
#
# Run on the reference machine (Ryzen 3600, target-cpu=native) for numbers
# that ship; interleaves densities within each rep to decorrelate drift.
#
#   DENSITIES="4 8 16 32" SIZES="100k 500k 2m" REPS=3 ./scripts/sweep_grid_density.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DENSITIES="${DENSITIES:-4 8 12 16 24 32 48}"
SIZES="${SIZES:-100k 500k 2m}"
REPS="${REPS:-3}"
BIN=target/release/bench_voronoi

RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" \
    cargo build --release --features tools,timing --bin bench_voronoi >/dev/null

for rep in $(seq 1 "$REPS"); do
    for d in $DENSITIES; do
        for s in $SIZES; do
            RAYON_NUM_THREADS=1 S2_VORONOI_TIMING_KV=1 S2_VORONOI_GRID_DENSITY="$d" \
                "$BIN" "$s" --no-preprocess 2>&1 |
                grep TIMING_KV | sed "s/^/density=$d rep=$rep /"
        done
    done
done
