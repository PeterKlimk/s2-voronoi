#!/usr/bin/env bash
# Build a native bench_voronoi binary with profile-guided optimization.
#
# Usage:
#   ./scripts/pgo_build.sh balanced
#   ./scripts/pgo_build.sh fib
#
# The balanced preset trains Fibonacci, uniform, clustered, and mega paths.
# The fib preset maximizes the common well-distributed Fibonacci path, at the
# cost of some performance on the adversarial mega distribution.

set -euo pipefail

PRESET="${1:-balanced}"
case "$PRESET" in
    balanced|fib) ;;
    *)
        echo "usage: $0 [balanced|fib]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HOST="$(rustc -vV | sed -n 's/^host: //p')"
SYSROOT="$(rustc --print sysroot)"
LLVM_PROFDATA="$SYSROOT/lib/rustlib/$HOST/bin/llvm-profdata"

if [[ ! -x "$LLVM_PROFDATA" ]]; then
    echo "missing $LLVM_PROFDATA" >&2
    echo "install it with: rustup component add llvm-tools-preview" >&2
    exit 1
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
OUTPUT_ROOT="${PGO_OUTPUT_ROOT:-$PROJECT_DIR/target/pgo}"
RUN_DIR="$OUTPUT_ROOT/$PRESET-$RUN_ID"
PROFILE_DIR="$RUN_DIR/profiles"
GENERATE_TARGET="$RUN_DIR/generate"
USE_TARGET="$RUN_DIR/use"
OUTPUT_BIN="$RUN_DIR/bench_voronoi-$PRESET"
mkdir -p "$PROFILE_DIR"

BASE_RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native -C force-frame-pointers=yes"
GENERATE_RUSTFLAGS="$BASE_RUSTFLAGS -C profile-generate=$PROFILE_DIR"
USE_RUSTFLAGS="$BASE_RUSTFLAGS -C profile-use=$PROFILE_DIR/merged.profdata"

cd "$PROJECT_DIR"

echo "Building instrumented binary ($PRESET)..."
CARGO_TARGET_DIR="$GENERATE_TARGET" RUSTFLAGS="$GENERATE_RUSTFLAGS" \
    cargo build --release --features tools --bin bench_voronoi
GENERATE_BIN="$GENERATE_TARGET/release/bench_voronoi"

train() {
    echo "Training: bench_voronoi $*"
    LLVM_PROFILE_FILE="$PROFILE_DIR/default_%m.profraw" "$GENERATE_BIN" "$@"
}

train 2.5m --no-preprocess
if [[ "$PRESET" == balanced ]]; then
    train 1m --dist uniform --no-preprocess
    train 500k --dist clustered --no-preprocess
    train 500k --dist mega --no-preprocess
fi

"$LLVM_PROFDATA" merge -o "$PROFILE_DIR/merged.profdata" "$PROFILE_DIR"

echo "Building profile-optimized binary..."
CARGO_TARGET_DIR="$USE_TARGET" RUSTFLAGS="$USE_RUSTFLAGS" \
    cargo build --release --features tools --bin bench_voronoi
cp "$USE_TARGET/release/bench_voronoi" "$OUTPUT_BIN"

{
    echo "preset=$PRESET"
    echo "rustc=$(rustc -Vv | tr '\n' ' ')"
    echo "rustflags=$USE_RUSTFLAGS"
    echo "profile=$PROFILE_DIR/merged.profdata"
} > "$RUN_DIR/manifest.txt"

echo "PGO binary: $OUTPUT_BIN"
