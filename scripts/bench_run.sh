#!/bin/bash
# Run interleaved benchmarks for binaries built by `bench_build.sh`.
#
# Sweeps a matrix of sizes x distributions x seeds. For each cell it runs the
# commits interleaved with rotating start order (the paired protocol that
# cancels slow-drift), and reports per-commit median/spread plus a relative
# verdict. Optionally emits a structured CSV across the whole matrix.
#
# Defaults favor reproducibility (single-threaded + CPU pinning). The box is
# noisy: per-binary code-layout offsets alone are ~1-2% at 500k ST (see
# docs/micro-optimization-matrix.md), so treat sub-1% deltas as noise and use
# a control commit when a result is close.
#
# Examples:
#   ./scripts/bench_run.sh -s 500k -r 20 -m total
#   ./scripts/bench_run.sh -s "500k 2m" -d "uniform mega" --seeds "1 2 3" --csv /tmp/bench.csv
#   ./scripts/bench_run.sh -s 1m -d mega --dist-param 0.4

set -euo pipefail

TMP_DIR="/tmp/bench_compare"
MANIFEST="$TMP_DIR/manifest.txt"

# Defaults (single values are back-compatible; space-separated lists sweep).
ROUNDS=5
SIZES="100k"
DISTS="fib"
SEEDS=""           # empty => bench_voronoi's default seed (single run)
DIST_PARAM=""
COOLDOWN=5
CPU_PIN="0"
SINGLE_THREAD=true
METRIC="total"
NO_PREPROCESS=true
CSV=""
EXTRA_BENCH_ARGS=()
# Convergence mode: run paired-interleaved rounds until the per-round-ratio CI
# settles (decision-grade on a NOISY box for effects above the ~layout floor),
# then stop. First commit listed = baseline; others reported relative to it.
CONVERGE=false
MAX_ROUNDS=160
MIN_ROUNDS=12
RESOLUTION=0.01    # ± band (1%) treated as "neutral"; ~the code-layout floor

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rounds) ROUNDS="$2"; shift 2 ;;
        -s|--size|--sizes) SIZES="$2"; shift 2 ;;
        -d|--dist|--dists) DISTS="$2"; shift 2 ;;
        --seeds|--seed) SEEDS="$2"; shift 2 ;;
        --dist-param) DIST_PARAM="$2"; shift 2 ;;
        --csv) CSV="$2"; shift 2 ;;
        --converge) CONVERGE=true; shift ;;
        --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
        --min-rounds) MIN_ROUNDS="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        -c|--cooldown) COOLDOWN="$2"; shift 2 ;;
        -p|--pin) CPU_PIN="$2"; shift 2 ;;
        --no-pin) CPU_PIN=""; shift ;;
        -1|--single) SINGLE_THREAD=true; shift ;;
        --multi) SINGLE_THREAD=false; shift ;;
        -m|--metric) METRIC="$2"; shift 2 ;;
        --no-preprocess) NO_PREPROCESS=true; shift ;;
        --preprocess) NO_PREPROCESS=false; shift ;;
        --) shift; EXTRA_BENCH_ARGS+=("$@"); break ;;
        -h|--help)
            cat <<'EOF'
Usage: bench_run.sh [opts] [-- bench_voronoi_args...]
  -r, --rounds N      Rounds per cell (default: 5)
  -s, --sizes "..."   Size(s), space-separated to sweep (default: 100k)
  -d, --dists "..."   Distribution(s) to sweep: fib uniform clustered bimodal
                      gradient outlier splittable mega (default: fib)
      --seeds "..."   Seed(s) to sweep (default: bench_voronoi default, one run)
      --dist-param X  Distribution shape knob (gradient k / mega fraction)
      --csv FILE      Write a structured CSV across the whole matrix
      --converge      Run paired-interleaved rounds until the per-round-ratio
                      95% CI settles, then stop (decision-grade on a noisy box
                      for effects above ~--resolution). First commit = baseline.
      --max-rounds N  Cap for --converge (default 160)
      --min-rounds N  Min rounds before checking convergence (default 12)
      --resolution R  Neutral band / target resolution (default 0.01 = 1%)
  -c, --cooldown N    Seconds between rounds (default: 5)
  -p, --pin CORE      Pin to CPU core (default: 0); --no-pin to disable
  -1, --single        Single-threaded (default); --multi to allow rayon
  -m, --metric M      total|timing_total|preprocess|knn_build|cell_construction|
                      dedup|edge_reconcile|assemble (default: total)
      --no-preprocess Pass --no-preprocess (default); --preprocess to disable
      --              Forward remaining args to bench_voronoi
EOF
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

[[ -f "$MANIFEST" ]] || { echo "Error: $MANIFEST not found; run bench_build.sh first"; exit 1; }

INDICES=(); LABELS=()
while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    IFS=: read -r idx label <<< "$line"
    INDICES+=("$idx"); LABELS+=("$label")
done < "$MANIFEST"
NUM_VERSIONS=${#INDICES[@]}
[[ $NUM_VERSIONS -eq 0 ]] && { echo "Error: no versions in manifest"; exit 1; }
for idx in "${INDICES[@]}"; do
    [[ -x "$TMP_DIR/bench_$idx" ]] || { echo "Error: $TMP_DIR/bench_$idx missing; rebuild"; exit 1; }
done

TASKSET=""
if [[ -n "$CPU_PIN" ]]; then
    command -v taskset &>/dev/null && TASKSET="taskset -c $CPU_PIN" \
        || echo "Warning: taskset not found, no CPU pinning"
fi
$SINGLE_THREAD && export RAYON_NUM_THREADS=1

extract_metric_ms() {
    local output="$1" metric="$2" line=""
    if [[ "$metric" != "total" ]]; then
        local kv; kv="$(echo "$output" | grep -m1 -E "^TIMING_KV " || true)"
        if [[ -n "$kv" ]]; then
            local key; [[ "$metric" == "timing_total" ]] && key="total_ms" || key="${metric}_ms"
            local val
            if val="$(echo "$kv" | awk -v k="${key}=" '{for(i=1;i<=NF;i++){if(index($i,k)==1){print substr($i,length(k)+1);exit}}exit 1}')"; then
                [[ -n "$val" ]] && { echo "$val"; return 0; }
            fi
        fi
    fi
    case "$metric" in
        total) line="$(echo "$output" | grep -m1 "Total time:" || true)" ;;
        timing_total) line="$(echo "$output" | grep -m1 -E "^[[:space:]]*total:" || true)" ;;
        *) line="$(echo "$output" | grep -m1 -E "[[:space:]]${metric}:" || true)" ;;
    esac
    [[ -z "$line" ]] && return 1
    echo "$line" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+(\.[0-9]+)?ms$/){gsub(/ms$/,"",$i);print $i;exit}}}'
}
float_gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>b)}'; }
float_lt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a<b)}'; }
calc_stats() {
    local file="$1"; [[ ! -s "$file" ]] && { echo "0 0 0 0"; return; }
    sort -n "$file" | awk '{a[NR]=$1;s+=$1} END{n=NR;med=(n%2==1)?a[int(n/2)+1]:(a[n/2]+a[n/2+1])/2;printf "%.1f %.1f %.1f %.1f",a[1],med,s/n,a[n]}'
}

# Paired verdict of `target` vs `baseline` over rounds (paired by line number,
# so common-mode noise cancels). Prints: VERDICT geomean ci_lo ci_hi k n
# VERDICT in {FASTER,SLOWER,NEUTRAL,UNRESOLVED,INSUF}. The CI is on the
# log-ratio (geometric); NEUTRAL = CI fully inside [1-res,1+res]; FASTER =
# CI fully below 1-res (target faster than baseline); SLOWER = fully above.
paired_verdict() {
    local baseline="$1" target="$2" res="$3"
    awk -v res="$res" '
        NR==FNR { b[FNR]=$1; nb=FNR; next }
        { if (FNR<=nb && b[FNR]>0 && $1>0) { r=$1/b[FNR]; lr=log(r); s+=lr; s2+=lr*lr; n++; if($1<b[FNR]) k++ } }
        END {
            if (n<3) { printf "INSUF 0 0 0 0 %d", n+0; exit }
            m=s/n; var=(s2-n*m*m)/(n-1); if(var<0)var=0; se=sqrt(var/n);
            gm=exp(m); lo=exp(m-1.96*se); hi=exp(m+1.96*se);
            # Direction by the paired SIGN test (robust to the heavy-tailed
            # per-round bursts on a busy box that inflate the parametric CI);
            # magnitude by the geometric mean. k = rounds target beat baseline.
            z=(2*k-n)/sqrt(n); sig=(z*z>3.8416);   # |z|>1.96 ~ p<0.05, two-sided
            v="UNRESOLVED";
            if (sig && gm < 1-res) v="FASTER";
            else if (sig && gm > 1+res) v="SLOWER";
            else if ((lo>=1-res && hi<=1+res) || (sig && gm>=1-res && gm<=1+res)) v="NEUTRAL";
            printf "%s %.4f %.4f %.4f %d %d", v, gm, lo, hi, k, n;
        }' "$baseline" "$target"
}

SEED_LIST="${SEEDS:-_}"   # "_" sentinel = no explicit seed
if [[ -n "$CSV" ]]; then
    if $CONVERGE; then
        echo "commit,size,dist,seed,metric,verdict,geomean,ci_lo,ci_hi,k_faster,n,rounds" > "$CSV"
    else
        echo "commit,size,dist,seed,metric,min_ms,median_ms,avg_ms,max_ms,spread_pct" > "$CSV"
    fi
fi

echo "=== Interleaved benchmark matrix ==="
echo "Versions: $NUM_VERSIONS | rounds=$ROUNDS metric=$METRIC ${SINGLE_THREAD:+ST}${CPU_PIN:+ pin=$CPU_PIN}"
echo "Sizes: [$SIZES]  Dists: [$DISTS]  Seeds: [${SEEDS:-default}]${DIST_PARAM:+  dist-param=$DIST_PARAM}"
for l in "${LABELS[@]}"; do echo "  $l"; done

run_cell() { # size dist seed
    local size="$1" dist="$2" seed="$3"
    local bench_args=("$size")
    $NO_PREPROCESS && bench_args+=("--no-preprocess")
    bench_args+=("--dist" "$dist")
    [[ "$seed" != "_" ]] && bench_args+=("--seed" "$seed")
    [[ -n "$DIST_PARAM" ]] && bench_args+=("--dist-param" "$DIST_PARAM")
    bench_args+=("${EXTRA_BENCH_ARGS[@]}")

    echo ""; echo "### cell: size=$size dist=$dist seed=${seed/_/default} ###"
    for idx in "${INDICES[@]}"; do > "$TMP_DIR/times_$idx.txt"; done

    local cap="$ROUNDS"; $CONVERGE && cap="$MAX_ROUNDS"
    local done_rounds=0
    for round in $(seq 1 "$cap"); do
        local start=$(( (round - 1) % NUM_VERSIONS ))
        $TASKSET "$TMP_DIR/bench_${INDICES[$start]}" "${bench_args[@]}" >/dev/null 2>&1 || true  # warmup
        for offset in $(seq 0 $((NUM_VERSIONS - 1))); do
            local i=$(( (start + offset) % NUM_VERSIONS )) idx label out ms
            idx="${INDICES[$i]}"; label="${LABELS[$i]}"
            out=$($TASKSET "$TMP_DIR/bench_$idx" "${bench_args[@]}" 2>&1)
            if ! ms="$(extract_metric_ms "$out" "$METRIC")"; then
                echo "  $label: FAILED (metric '$METRIC' not found)"; continue
            fi
            echo "$ms" >> "$TMP_DIR/times_$idx.txt"
        done
        done_rounds=$round
        # Convergence: stop once every non-baseline pair's CI has settled.
        if $CONVERGE && (( round >= MIN_ROUNDS )); then
            local all_resolved=true j
            for j in "${!INDICES[@]}"; do
                (( j == 0 )) && continue
                local vv; vv=$(paired_verdict "$TMP_DIR/times_${INDICES[0]}.txt" "$TMP_DIR/times_${INDICES[$j]}.txt" "$RESOLUTION" | awk '{print $1}')
                [[ "$vv" == "UNRESOLVED" || "$vv" == "INSUF" ]] && all_resolved=false
            done
            $all_resolved && break
        fi
        [[ $round -lt $cap ]] && sleep "$COOLDOWN"
    done

    if $CONVERGE; then
        local conv="converged"; (( done_rounds >= MAX_ROUNDS )) && conv="HIT MAX (unresolved at ${RESOLUTION} resolution)"
        echo "  baseline: ${LABELS[0]}   rounds=$done_rounds ($conv)"
        for i in "${!INDICES[@]}"; do
            (( i == 0 )) && continue
            local v gm lo hi k n
            read -r v gm lo hi k n <<< "$(paired_verdict "$TMP_DIR/times_${INDICES[0]}.txt" "$TMP_DIR/times_${INDICES[$i]}.txt" "$RESOLUTION")"
            local pct ci
            pct=$(awk -v g="$gm" 'BEGIN{printf "%+.1f%%",(g-1)*100}')
            ci=$(awk -v a="$lo" -v b="$hi" 'BEGIN{printf "[%+.1f%%, %+.1f%%]",(a-1)*100,(b-1)*100}')
            printf "    %-26.26s %-11s %7s vs base  CI %-20s (%d/%d faster)\n" "${LABELS[$i]}" "$v" "$pct" "$ci" "$k" "$n"
            [[ -n "$CSV" ]] && echo "\"${LABELS[$i]}\",$size,$dist,${seed/_/default},$METRIC,$v,$gm,$lo,$hi,$k,$n,$done_rounds" >> "$CSV"
        done
        return
    fi

    # Per-commit stats + verdict for this cell.
    local best_median=999999999 best_idx=""
    declare -A MED
    printf "  %-28s %9s %9s %9s\n" "version" "median" "min" "spread"
    for i in "${!INDICES[@]}"; do
        local idx="${INDICES[$i]}" label="${LABELS[$i]}" min med avg max spread
        read -r min med avg max <<< "$(calc_stats "$TMP_DIR/times_${INDICES[$i]}.txt")"
        MED[$idx]="$med"
        if float_gt "$min" "0"; then spread="$(awk -v a="$max" -v b="$min" 'BEGIN{printf "%.1f%%",(a-b)/b*100}')"; else spread="N/A"; fi
        printf "  %-28.28s %8.1fms %8.1fms %9s\n" "$label" "$med" "$min" "$spread"
        [[ -n "$CSV" ]] && echo "\"$label\",$size,$dist,${seed/_/default},$METRIC,$min,$med,$avg,$max,${spread/\%/}" >> "$CSV"
        if float_gt "$med" "0" && float_lt "$med" "$best_median"; then best_median="$med"; best_idx="$idx"; fi
    done
    if [[ -n "$best_idx" ]]; then
        for i in "${!INDICES[@]}"; do
            local idx="${INDICES[$i]}" med="${MED[${INDICES[$i]}]}"
            [[ -z "$med" || "$med" == "0" ]] && continue
            local pct; pct="$(awk -v a="$med" -v b="$best_median" 'BEGIN{printf "%.1f",(a/b-1)*100}')"
            local v; [[ "$idx" == "$best_idx" ]] && v="FASTEST" || { float_gt "$pct" "0.5" && v="+${pct}%" || v="~same"; }
            printf "    %-28.28s %s\n" "${LABELS[$i]}" "$v"
        done
    fi
}

for size in $SIZES; do
    for dist in $DISTS; do
        for seed in $SEED_LIST; do
            run_cell "$size" "$dist" "$seed"
        done
    done
done

echo ""
[[ -n "$CSV" ]] && echo "CSV written: $CSV"
echo "Done."
