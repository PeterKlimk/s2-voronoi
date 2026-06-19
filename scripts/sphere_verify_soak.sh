#!/usr/bin/env bash
#
# Endless high-N spherical correctness soak.
#
# Builds bench_voronoi once, then runs a size x distribution matrix with
# S2_VORONOI_VERIFY=1 until cancelled or the first failing case. This uses
# Rayon normally; set RAYON_NUM_THREADS outside the script if you want a fixed
# thread count.
#
# Defaults favor the current production path: multi-threaded, no preprocess,
# 2M points, and a small mix of structured/random sphere distributions.
#
# Examples:
#   ./scripts/sphere_verify_soak.sh
#   SIZES="2m 3m" DISTS="fib uniform mega cap" ./scripts/sphere_verify_soak.sh
#   NO_PREPROCESS=0 SIZES=1m DISTS="clustered bimodal gradient" ./scripts/sphere_verify_soak.sh
#   MAX_CASES=4 SIZES=50k DISTS=fib ./scripts/sphere_verify_soak.sh
#   DIST_PARAM=0.4 DISTS=mega ./scripts/sphere_verify_soak.sh
#   ./scripts/sphere_verify_soak.sh --repeat 2
#
# Environment:
#   SIZES           space-separated bench sizes (default: "2m")
#   DISTS           bench_voronoi --dist values (default: "fib uniform")
#                   known values: fib uniform clustered bimodal gradient
#                   outlier splittable mega cap
#   SEED_START      first generated seed (default: 1)
#   MAX_CASES       stop after this many cases; 0 means forever (default: 0)
#   NO_PREPROCESS   1 passes --no-preprocess, 0 uses weld preprocess (default: 1)
#   DIST_PARAM      optional bench_voronoi --dist-param value
#   FEATURES        cargo feature set for bench_voronoi (default: tools)
#   CARGO           cargo executable (default: cargo)

set -euo pipefail
cd "$(dirname "$0")/.."

SIZES="${SIZES:-2m}"
DISTS="${DISTS:-fib uniform}"
SEED_START="${SEED_START:-1}"
MAX_CASES="${MAX_CASES:-0}"
NO_PREPROCESS="${NO_PREPROCESS:-1}"
DIST_PARAM="${DIST_PARAM:-}"
FEATURES="${FEATURES:-tools}"
CARGO="${CARGO:-cargo}"

if ! [[ "$SEED_START" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED_START must be an integer, got '$SEED_START'" >&2
    exit 2
fi
if ! [[ "$MAX_CASES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MAX_CASES must be an integer, got '$MAX_CASES'" >&2
    exit 2
fi
case "$NO_PREPROCESS" in
    0|1) ;;
    *) echo "ERROR: NO_PREPROCESS must be 0 or 1, got '$NO_PREPROCESS'" >&2; exit 2 ;;
esac

echo "Building bench_voronoi --release --features '$FEATURES'..."
"$CARGO" build --release --features "$FEATURES" --bin bench_voronoi
BIN="target/release/bench_voronoi"
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: missing executable $BIN" >&2
    exit 1
fi

echo
echo "Starting strict sphere validation soak."
echo "  sizes:         $SIZES"
echo "  dists:         $DISTS"
echo "  seed_start:    $SEED_START"
echo "  max_cases:     $MAX_CASES (0 = forever)"
echo "  no_preprocess: $NO_PREPROCESS"
echo "  dist_param:    ${DIST_PARAM:-<default>}"
echo "  rayon threads: ${RAYON_NUM_THREADS:-rayon default}"
echo "  extra args:    ${*:-<none>}"
echo

case_idx=0
round=0
while :; do
    round=$((round + 1))
    for size in $SIZES; do
        for dist in $DISTS; do
            seed=$((SEED_START + case_idx))
            args=("$size" "--dist" "$dist" "--seed" "$seed")
            if [[ "$NO_PREPROCESS" == "1" ]]; then
                args+=("--no-preprocess")
            fi
            if [[ -n "$DIST_PARAM" ]]; then
                args+=("--dist-param" "$DIST_PARAM")
            fi
            args+=("$@")

            printf '[%s] round=%d case=%d size=%s dist=%s seed=%d\n' \
                "$(date -Is)" "$round" "$case_idx" "$size" "$dist" "$seed"
            S2_VORONOI_VERIFY=1 "$BIN" "${args[@]}"
            printf '[%s] PASS round=%d case=%d size=%s dist=%s seed=%d\n\n' \
                "$(date -Is)" "$round" "$case_idx" "$size" "$dist" "$seed"

            case_idx=$((case_idx + 1))
            if [[ "$MAX_CASES" != "0" && "$case_idx" -ge "$MAX_CASES" ]]; then
                echo "Completed MAX_CASES=$MAX_CASES without failure."
                exit 0
            fi
        done
    done
done
