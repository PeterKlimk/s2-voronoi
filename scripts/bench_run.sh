#!/bin/bash
# Run interleaved benchmarks on pre-built binaries
# First run bench_build.sh, let laptop cool, then run this
#
# Defaults to pinned (core 0) + single-threaded for consistent results

set -e

TMP_DIR="/tmp/bench_compare"
MANIFEST="$TMP_DIR/manifest.txt"

# Defaults: pinned + single-threaded
ROUNDS=5
SIZE="100k"
COOLDOWN=5
CPU_PIN="0"
SINGLE_THREAD=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rounds) ROUNDS="$2"; shift 2 ;;
        -s|--size) SIZE="$2"; shift 2 ;;
        -c|--cooldown) COOLDOWN="$2"; shift 2 ;;
        -p|--pin) CPU_PIN="$2"; shift 2 ;;
        --no-pin) CPU_PIN=""; shift ;;
        -1|--single) SINGLE_THREAD=true; shift ;;
        --multi) SINGLE_THREAD=false; shift ;;
        -h|--help)
            echo "Usage: $0 [-r rounds] [-s size] [-c cooldown] [-p cpu] [--no-pin] [--multi]"
            echo ""
            echo "Options:"
            echo "  -r, --rounds    Number of rounds (default: 5)"
            echo "  -s, --size      Benchmark size (default: 100k)"
            echo "  -c, --cooldown  Seconds between rounds (default: 5)"
            echo "  -p, --pin       Pin to CPU core (default: 0)"
            echo "  --no-pin        Disable CPU pinning"
            echo "  -1, --single    Force single-threaded (default)"
            echo "  --multi         Allow multi-threaded"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check manifest exists
if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: $MANIFEST not found"
    echo "Run ./scripts/bench_build.sh first"
    exit 1
fi

# Read manifest into arrays
INDICES=()
LABELS=()
while IFS=: read -r idx label; do
    INDICES+=("$idx")
    LABELS+=("$label")
done < "$MANIFEST"

NUM_VERSIONS=${#INDICES[@]}
if [[ $NUM_VERSIONS -eq 0 ]]; then
    echo "Error: No versions found in manifest"
    exit 1
fi

# Check binaries exist
for idx in "${INDICES[@]}"; do
    if [[ ! -x "$TMP_DIR/bench_$idx" ]]; then
        echo "Error: $TMP_DIR/bench_$idx not found"
        echo "Run ./scripts/bench_build.sh first"
        exit 1
    fi
done

# Build taskset prefix if pinning requested
TASKSET=""
if [[ -n "$CPU_PIN" ]]; then
    if command -v taskset &> /dev/null; then
        TASKSET="taskset -c $CPU_PIN"
    else
        echo "Warning: taskset not found, running without CPU pinning"
    fi
fi

# Set single-threaded mode
if $SINGLE_THREAD; then
    export RAYON_NUM_THREADS=1
fi

echo "=== Interleaved Benchmark ==="
echo "Versions: $NUM_VERSIONS"
echo "Rounds: $ROUNDS, Size: $SIZE, Cooldown: ${COOLDOWN}s"
[[ -n "$CPU_PIN" ]] && echo "CPU pin: core $CPU_PIN"
$SINGLE_THREAD && echo "Mode: single-threaded"
echo ""
echo "Testing:"
for i in "${!INDICES[@]}"; do
    echo "  ${LABELS[$i]}"
done
echo ""

# Create result files
for idx in "${INDICES[@]}"; do
    > "$TMP_DIR/times_$idx.txt"
done

# Run interleaved with rotating start position
for round in $(seq 1 $ROUNDS); do
    echo "--- Round $round/$ROUNDS ---"

    # Rotate starting position each round to avoid order bias
    start=$(( (round - 1) % NUM_VERSIONS ))

    # Warmup run (discarded) - use the first binary in rotation
    warmup_idx="${INDICES[$start]}"
    warmup_output=$($TASKSET "$TMP_DIR/bench_$warmup_idx" "$SIZE" 2>&1)
    warmup_ms=$(echo "$warmup_output" | grep "Total time:" | awk '{print $3}' | tr -d 'ms')
    echo "  (warmup: ${warmup_ms}ms)"

    for offset in $(seq 0 $((NUM_VERSIONS - 1))); do
        i=$(( (start + offset) % NUM_VERSIONS ))
        idx="${INDICES[$i]}"
        label="${LABELS[$i]}"
        output=$($TASKSET "$TMP_DIR/bench_$idx" "$SIZE" 2>&1)
        time_ms=$(echo "$output" | grep "Total time:" | awk '{print $3}' | tr -d 'ms')

        if [[ -z "$time_ms" ]]; then
            echo "  $label: FAILED"
            continue
        fi

        echo "  $label: ${time_ms}ms"
        echo "$time_ms" >> "$TMP_DIR/times_$idx.txt"
    done

    if [[ $round -lt $ROUNDS ]]; then
        sleep "$COOLDOWN"
    fi
done

echo ""
echo "=== Results ==="
echo ""

calc_stats() {
    local file="$1"
    [[ ! -s "$file" ]] && { echo "0 0 0 0"; return; }
    sort -n "$file" | awk '
    { a[NR] = $1; sum += $1 }
    END {
        n = NR; min = a[1]; max = a[n]; avg = sum / n
        median = (n % 2 == 1) ? a[int(n/2) + 1] : (a[n/2] + a[n/2 + 1]) / 2
        printf "%.1f %.1f %.1f %.1f", min, median, avg, max
    }'
}

# Find max label length for formatting
max_len=7  # minimum "Version"
for label in "${LABELS[@]}"; do
    len=${#label}
    (( len > max_len )) && max_len=$len
done

# Print header
printf "%-${max_len}s %10s %10s %10s %10s %10s\n" "Version" "Min" "Median" "Avg" "Max" "Spread"
printf "%-${max_len}s %10s %10s %10s %10s %10s\n" "$(printf '%*s' $max_len '' | tr ' ' '-')" "---" "------" "---" "---" "------"

# Store mins for comparison
declare -A MINS

for i in "${!INDICES[@]}"; do
    idx="${INDICES[$i]}"
    label="${LABELS[$i]}"
    stats=$(calc_stats "$TMP_DIR/times_$idx.txt")
    read -r min median avg max <<< "$stats"
    MINS[$idx]="$min"
    if (( $(echo "$min > 0" | bc -l) )); then
        spread=$(echo "$max $min" | awk '{printf "%.1f%%", ($1 - $2) / $2 * 100}')
    else
        spread="N/A"
    fi
    printf "%-${max_len}s %9.1fms %9.1fms %9.1fms %9.1fms %10s\n" "$label" "$min" "$median" "$avg" "$max" "$spread"
done

echo ""
echo "=== Relative Performance (min times, lower is better) ==="
echo ""

# Find the fastest (lowest min)
best_idx=""
best_min=999999999
for i in "${!INDICES[@]}"; do
    idx="${INDICES[$i]}"
    min="${MINS[$idx]}"
    if (( $(echo "$min > 0 && $min < $best_min" | bc -l) )); then
        best_min="$min"
        best_idx="$idx"
    fi
done

if [[ -n "$best_idx" ]]; then
    for i in "${!INDICES[@]}"; do
        idx="${INDICES[$i]}"
        label="${LABELS[$i]}"
        min="${MINS[$idx]}"
        if [[ -n "$min" && "$min" != "0" ]]; then
            pct=$(echo "$min $best_min" | awk '{printf "%.1f", ($1 / $2 - 1) * 100}')
            if [[ "$idx" == "$best_idx" ]]; then
                verdict="FASTEST"
            elif (( $(echo "$pct > 0.5" | bc -l) )); then
                verdict="+${pct}%"
            else
                verdict="~same"
            fi
            printf "%-${max_len}s %s\n" "$label" "$verdict"
        fi
    done
fi

echo ""
