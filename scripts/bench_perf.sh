#!/bin/bash
# Interleaved `perf stat` comparison for artifacts built by bench_build.sh.
# Emits one tidy CSV row per (round, version, event), retaining scheduling
# counters so contaminated samples can be filtered during analysis.

set -euo pipefail

TMP_DIR="/tmp/bench_compare"
MANIFEST="$TMP_DIR/manifest.txt"
ROUNDS=7
SIZE="500k"
DIST="fib"
SEED=""
DIST_PARAM=""
CPU_PIN="0"
SINGLE_THREAD=true
NO_PREPROCESS=true
CSV="/tmp/bench_perf.csv"
EVENTS="task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses,context-switches,cpu-migrations"
EXTRA_BENCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rounds) ROUNDS="$2"; shift 2 ;;
        -s|--size) SIZE="$2"; shift 2 ;;
        -d|--dist) DIST="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dist-param) DIST_PARAM="$2"; shift 2 ;;
        -p|--pin) CPU_PIN="$2"; shift 2 ;;
        --no-pin) CPU_PIN=""; shift ;;
        -1|--single) SINGLE_THREAD=true; shift ;;
        --multi) SINGLE_THREAD=false; shift ;;
        --preprocess) NO_PREPROCESS=false; shift ;;
        --no-preprocess) NO_PREPROCESS=true; shift ;;
        -e|--events) EVENTS="$2"; shift 2 ;;
        --csv) CSV="$2"; shift 2 ;;
        --) shift; EXTRA_BENCH_ARGS+=("$@"); break ;;
        -h|--help)
            cat <<'EOF'
Usage: bench_perf.sh [opts] [-- bench_voronoi_args...]
  -r, --rounds N       Interleaved rounds (default: 7)
  -s, --size SIZE      Benchmark size (default: 500k)
  -d, --dist DIST      Distribution (default: fib)
      --seed N         Explicit distribution seed
      --dist-param X   Distribution shape parameter
  -p, --pin CORE       Pin benchmark to a CPU (default: 0)
      --no-pin         Disable CPU pinning
  -1, --single         Set RAYON_NUM_THREADS=1 (default)
      --multi          Leave Rayon thread count unrestricted
      --preprocess     Include preprocessing (default is --no-preprocess)
  -e, --events LIST    Comma-separated perf events
      --csv FILE       Raw tidy output (default: /tmp/bench_perf.csv)
EOF
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

command -v perf >/dev/null || { echo "perf not found" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "Run scripts/bench_build.sh first" >&2; exit 1; }

INDICES=(); LABELS=()
while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    IFS=: read -r idx label <<< "$line"
    INDICES+=("$idx"); LABELS+=("$label")
done < "$MANIFEST"
(( ${#INDICES[@]} > 0 )) || { echo "Empty benchmark manifest" >&2; exit 1; }

for idx in "${INDICES[@]}"; do
    [[ -x "$TMP_DIR/bench_$idx" ]] || { echo "Missing bench_$idx; rebuild" >&2; exit 1; }
done

$SINGLE_THREAD && export RAYON_NUM_THREADS=1
BENCH_ARGS=("$SIZE")
$NO_PREPROCESS && BENCH_ARGS+=("--no-preprocess")
BENCH_ARGS+=("--dist" "$DIST")
[[ -n "$SEED" ]] && BENCH_ARGS+=("--seed" "$SEED")
[[ -n "$DIST_PARAM" ]] && BENCH_ARGS+=("--dist-param" "$DIST_PARAM")
BENCH_ARGS+=("${EXTRA_BENCH_ARGS[@]}")

run_bench() {
    local binary="$1" stat_file="$2"
    local command=("$binary" "${BENCH_ARGS[@]}")
    if [[ -n "$CPU_PIN" ]]; then
        command=(taskset -c "$CPU_PIN" "${command[@]}")
    fi
    perf stat --no-big-num -x, -o "$stat_file" -e "$EVENTS" -- "${command[@]}" >/dev/null
}

echo "round,order,index,label,event,value,unit,runtime_pct,size,dist,seed,cpu" > "$CSV"
echo "=== perf counter matrix: rounds=$ROUNDS size=$SIZE dist=$DIST pin=${CPU_PIN:-none} ==="

for round in $(seq 1 "$ROUNDS"); do
    start=$(( (round - 1) % ${#INDICES[@]} ))
    for offset in $(seq 0 $(( ${#INDICES[@]} - 1 ))); do
        i=$(( (start + offset) % ${#INDICES[@]} ))
        idx="${INDICES[$i]}" label="${LABELS[$i]}"
        stat_file=$(mktemp /tmp/bench_perf_stat.XXXXXX)
        if (( round == 1 )); then
            warm_file=$(mktemp /tmp/bench_perf_warm.XXXXXX)
            run_bench "$TMP_DIR/bench_$idx" "$warm_file"
            rm -f "$warm_file"
        fi
        run_bench "$TMP_DIR/bench_$idx" "$stat_file"
        awk -F, -v round="$round" -v order="$offset" -v idx="$idx" -v label="$label" \
            -v size="$SIZE" -v dist="$DIST" -v seed="${SEED:-default}" -v cpu="${CPU_PIN:-none}" '
            $1 ~ /^#/ || NF < 3 || $1 ~ /not (counted|supported)/ { next }
            { gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3);
              printf "%s,%s,%s,\"%s\",%s,%s,%s,%s,%s,%s,%s,%s\n",
                     round,order,idx,label,$3,$1,$2,$5,size,dist,seed,cpu }
        ' "$stat_file" >> "$CSV"
        rm -f "$stat_file"
        echo "  round $round: $label"
    done
done

echo "Raw counters written to $CSV"
