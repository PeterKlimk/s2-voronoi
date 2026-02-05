#!/bin/bash
# Build benchmark binaries for one or more commits/refs.
# Use `bench_run.sh` to execute interleaved runs of the built artifacts.
#
# Usage:
#   bench_build.sh HEAD main~1 main~2     # Explicit refs
#   bench_build.sh HEAD main              # Working tree vs branch tip
#   bench_build.sh --chain 10             # Last 10 commits from HEAD
#   bench_build.sh --chain 5 main         # Last 5 commits from main
#   bench_build.sh HEAD --chain 3 main    # Working tree + last 3 from main

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TMP_DIR="/tmp/bench_compare"

# Build configuration (recorded in manifest).
USE_NATIVE=true
FEATURES="tools"
EXTRA_RUSTFLAGS=""

# Parse arguments.
COMMITS=()
CHAIN_COUNT=0
CHAIN_BASE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --features)
            FEATURES="${2:-}"
            shift 2
            ;;
        --timing)
            FEATURES="${FEATURES:+$FEATURES,}timing"
            shift
            ;;
        --no-native)
            USE_NATIVE=false
            shift
            ;;
        --native)
            USE_NATIVE=true
            shift
            ;;
        --rustflags)
            EXTRA_RUSTFLAGS="${2:-}"
            shift 2
            ;;
        --chain)
            CHAIN_COUNT="$2"
            shift 2
            # Next arg (if exists and not a flag) is the base
            if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
                CHAIN_BASE="$1"
                shift
            else
                CHAIN_BASE="HEAD"
            fi
            # Expand chain into commits
            for i in $(seq 0 $((CHAIN_COUNT - 1))); do
                if [[ $i -eq 0 ]]; then
                    COMMITS+=("$CHAIN_BASE")
                else
                    COMMITS+=("${CHAIN_BASE}~${i}")
                fi
            done
            ;;
        -h|--help)
            echo "Usage: $0 [commits...] [--chain N [base]]"
            echo ""
            echo "Examples:"
            echo "  $0 HEAD main~1 main~2      # Specific commits"
            echo "  $0 HEAD main               # Working tree vs main"
            echo "  $0 --chain 10              # Last 10 commits from HEAD"
            echo "  $0 --chain 5 main          # Last 5 commits from main"
            echo "  $0 HEAD --chain 3 main     # Working tree + last 3 from main"
            echo ""
            echo "Build options:"
            echo "  --native/--no-native       Toggle -C target-cpu=native (default: on)"
            echo "  --features \"a,b\"           Cargo features to enable (default: tools)"
            echo "  --timing                   Shorthand for --features timing"
            echo "  --rustflags \"...\"          Extra RUSTFLAGS (appended)"
            echo ""
            echo "Note: HEAD builds the current working tree (including uncommitted changes)."
            exit 0
            ;;
        *)
            COMMITS+=("$1")
            shift
            ;;
    esac
done

# bench_voronoi requires the `tools` feature.
if [[ -n "${FEATURES}" && "${FEATURES}" != *tools* ]]; then
    FEATURES="${FEATURES},tools"
fi

# Default to HEAD if no commits specified
if [[ ${#COMMITS[@]} -eq 0 ]]; then
    echo "No commits specified. Use --help for usage."
    exit 1
fi

mkdir -p "$TMP_DIR"

cd "$PROJECT_DIR"

# Use a consistent codegen baseline for comparisons.
RUSTFLAGS_FINAL="${RUSTFLAGS:-}"
if $USE_NATIVE; then
    RUSTFLAGS_FINAL="${RUSTFLAGS_FINAL} -C target-cpu=native"
fi
if [[ -n "$EXTRA_RUSTFLAGS" ]]; then
    RUSTFLAGS_FINAL="${RUSTFLAGS_FINAL} ${EXTRA_RUSTFLAGS}"
fi
export RUSTFLAGS="$RUSTFLAGS_FINAL"

# Detect dirty working tree.
DIRTY=false
if ! git diff --quiet || ! git diff --cached --quiet; then
    DIRTY=true
fi

# Save current state.
ORIGINAL_REF=$(git rev-parse HEAD)
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
STASHED=false

cleanup() {
    cd "$PROJECT_DIR"
    # Return to original state.
    if [[ -n "$ORIGINAL_BRANCH" && "$ORIGINAL_BRANCH" != "HEAD" ]]; then
        git checkout "$ORIGINAL_BRANCH" --quiet 2>/dev/null || git checkout "$ORIGINAL_REF" --quiet
    else
        git checkout "$ORIGINAL_REF" --quiet 2>/dev/null || true
    fi
    if $STASHED; then
        echo "Restoring stashed changes..."
        git stash pop --quiet || true
    fi
}
trap cleanup EXIT

echo "=== Building ${#COMMITS[@]} versions ==="
echo "Output: $TMP_DIR"
echo ""

# Clear old manifest
> "$TMP_DIR/manifest.txt"
{
    echo "# project_dir=$PROJECT_DIR"
    echo "# built_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true)"
    echo "# rustc=$(rustc -V 2>/dev/null || true)"
    echo "# cargo=$(cargo -V 2>/dev/null || true)"
    echo "# features=${FEATURES:-<none>}"
    echo "# use_native=$USE_NATIVE"
    echo "# rustflags=${RUSTFLAGS:-<unset>}"
} >> "$TMP_DIR/manifest.txt"
echo "" >> "$TMP_DIR/manifest.txt"

build_cmd=(cargo build --release --bin bench_voronoi)
if [[ -n "${FEATURES}" ]]; then
    build_cmd+=(--features "${FEATURES}")
fi

has_head_ref=false
for commit in "${COMMITS[@]}"; do
    if [[ "$commit" == "HEAD" ]]; then
        has_head_ref=true
        break
    fi
done

head_dirty_snapshot=""
if $DIRTY && $has_head_ref; then
    echo "Preparing deterministic dirty-HEAD snapshot..."
    "${build_cmd[@]}"
    head_dirty_snapshot="$TMP_DIR/bench_head_dirty_snapshot"
    cp target/release/bench_voronoi "$head_dirty_snapshot"
fi

for i in "${!COMMITS[@]}"; do
    commit="${COMMITS[$i]}"

    # Handle dirty HEAD deterministically, independent of argument order.
    if [[ "$commit" == "HEAD" && -n "$head_dirty_snapshot" ]]; then
        label="HEAD (dirty)"
        echo "[$((i+1))/${#COMMITS[@]}] Using $label snapshot..."
        cp "$head_dirty_snapshot" "$TMP_DIR/bench_$i"
        echo "$i:$label" >> "$TMP_DIR/manifest.txt"
        continue
    fi

    # For non-dirty-HEAD refs, use a clean checkout.
    if $DIRTY && ! $STASHED; then
        echo "Stashing uncommitted changes..."
        git stash push -m "bench_build temp stash" --quiet
        STASHED=true
        DIRTY=false
    fi

    # Resolve commit to short sha and create a readable label.
    sha=$(git rev-parse --short "$commit" 2>/dev/null || echo "")
    if [[ -z "$sha" ]]; then
        echo "[$((i+1))/${#COMMITS[@]}] Skipping $commit (not found)"
        continue
    fi

    if [[ "$commit" == "HEAD" ]]; then
        label="HEAD ($sha)"
    elif [[ "$commit" =~ ~[0-9]+$ ]]; then
        # It's a relative ref like main~2
        label="$commit ($sha)"
    else
        # Check if it's a branch name
        if git show-ref --verify --quiet "refs/heads/$commit" 2>/dev/null; then
            label="$commit ($sha)"
        else
            label="$sha"
        fi
    fi

    echo "[$((i+1))/${#COMMITS[@]}] Building $label..."
    git checkout "$commit" --quiet 2>/dev/null
    "${build_cmd[@]}"
    cp target/release/bench_voronoi "$TMP_DIR/bench_$i"
    echo "$i:$label" >> "$TMP_DIR/manifest.txt"
done

echo ""
echo "=== Build complete ==="
echo "Binaries:"
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^# ]] && continue
    IFS=: read -r idx label <<< "$line"
echo "  $TMP_DIR/bench_$idx  ($label)"
done < "$TMP_DIR/manifest.txt"
echo ""
echo "Then run:"
echo "  ./scripts/bench_run.sh"
