#!/bin/bash
# Build benchmark binaries for specified commits
# Run this, let laptop cool, then run bench_run.sh
#
# Usage:
#   bench_build.sh HEAD main~1 main~2     # Specific commits
#   bench_build.sh HEAD main              # Compare working tree vs main
#   bench_build.sh --chain 10             # Last 10 commits from HEAD
#   bench_build.sh --chain 5 main         # Last 5 commits from main
#   bench_build.sh HEAD --chain 3 main    # Working tree + last 3 from main

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TMP_DIR="/tmp/bench_compare"

# Parse arguments - collect commits and handle --chain
COMMITS=()
CHAIN_COUNT=0
CHAIN_BASE=""

while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "HEAD builds current working tree (including uncommitted changes)"
            exit 0
            ;;
        *)
            COMMITS+=("$1")
            shift
            ;;
    esac
done

# Default to HEAD if no commits specified
if [[ ${#COMMITS[@]} -eq 0 ]]; then
    echo "No commits specified. Use --help for usage."
    exit 1
fi

mkdir -p "$TMP_DIR"

cd "$PROJECT_DIR"

# Check if working tree is dirty
DIRTY=false
if ! git diff --quiet || ! git diff --cached --quiet; then
    DIRTY=true
fi

# Save current state
ORIGINAL_REF=$(git rev-parse HEAD)
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
STASHED=false

cleanup() {
    cd "$PROJECT_DIR"
    # Return to original state
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

for i in "${!COMMITS[@]}"; do
    commit="${COMMITS[$i]}"

    # Handle HEAD specially - build current working tree
    if [[ "$commit" == "HEAD" && $DIRTY == true ]]; then
        label="HEAD (dirty)"
        echo "[$((i+1))/${#COMMITS[@]}] Building $label (working tree with uncommitted changes)..."
        # Don't checkout, just build current state
        cargo build --release --bin bench_voronoi 2>&1 | grep -E "(Compiling|Finished)" || true
        cp target/release/bench_voronoi "$TMP_DIR/bench_$i"
        echo "$i:$label" >> "$TMP_DIR/manifest.txt"
        continue
    fi

    # For non-HEAD or clean HEAD, we need to checkout
    # Stash if dirty and not yet stashed
    if $DIRTY && ! $STASHED; then
        echo "Stashing uncommitted changes..."
        git stash push -m "bench_build temp stash" --quiet
        STASHED=true
        DIRTY=false  # Working tree is now clean
    fi

    # Resolve commit to short sha and get a nice label
    sha=$(git rev-parse --short "$commit" 2>/dev/null || echo "")
    if [[ -z "$sha" ]]; then
        echo "[$((i+1))/${#COMMITS[@]}] Skipping $commit (not found)"
        continue
    fi

    # Create readable label
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
    cargo build --release --bin bench_voronoi 2>&1 | grep -E "(Compiling|Finished)" || true
    cp target/release/bench_voronoi "$TMP_DIR/bench_$i"
    echo "$i:$label" >> "$TMP_DIR/manifest.txt"
done

echo ""
echo "=== Build complete ==="
echo "Binaries:"
while IFS=: read -r idx label; do
    echo "  $TMP_DIR/bench_$idx  ($label)"
done < "$TMP_DIR/manifest.txt"
echo ""
echo "Now let your laptop cool down, then run:"
echo "  ./scripts/bench_run.sh"
