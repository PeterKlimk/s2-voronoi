#!/usr/bin/env bash
#
# Robustness evidence campaign driver: one process per case.
#
# Sweeps seeds / sizes / distributions through the `campaign_case` test,
# forking a fresh process per case so the OS reclaims memory fully between
# builds. Peak RSS is bounded to a single build (~2.3 GB at 3M), so this is
# safe on small-RAM boxes where the in-process sweep OOMs. See the module
# docs in tests/robustness_campaign.rs for the why.
#
# Usage:
#   ./scripts/robustness_campaign.sh [output.csv]
#
# Environment:
#   MAX_N            cap on case size (default 3000000 — set higher only on
#                    a box with RAM headroom; peak is ~0.65 KB/point)
#   RAYON_NUM_THREADS  intra-build parallelism (default 6)
#   VORONOI_MESH_VERIFY  set to 1 for the always-on validator gate
#
# Each matrix case must succeed, clear all repair residuals, and validate
# strictly. Errors, surviving residuals, invalid output, and process death all
# fail the campaign.

set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-/tmp/robustness_campaign.csv}"
MAX_N="${MAX_N:-3000000}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-6}"

echo "Building test binary..."
cargo test --release --test robustness_campaign --no-run 2>&1 | tail -1
BIN="$(find target/release/deps -maxdepth 1 -type f -name 'robustness_campaign-*' ! -name '*.d' \
  -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
if [ -z "${BIN:-}" ] || [ ! -x "$BIN" ]; then
  echo "ERROR: could not locate compiled test binary" >&2
  exit 1
fi
echo "Binary: $BIN"
echo "Output: $OUT  (MAX_N=$MAX_N, RAYON_NUM_THREADS=$RAYON_NUM_THREADS)"
echo

echo "dist,n,seed,param,result,defects,post_repair,no_chain,valid,peak_mb,origins" > "$OUT"

invalid=0
errored=0
total=0

run_case() { # dist n seed param
  local dist="$1" n="$2" seed="$3" param="$4"
  if [ "$n" -gt "$MAX_N" ]; then
    echo "  skip $dist n=$n seed=$seed (> MAX_N=$MAX_N)"
    return
  fi
  total=$((total + 1))
  local line
  # libtest with --nocapture prefixes the line ("test campaign_case ...
  # CASERESULT ..."), so match anywhere and extract from CASERESULT on.
  line="$(VORONOI_MESH_CASE_DIST="$dist" VORONOI_MESH_CASE_N="$n" VORONOI_MESH_CASE_SEED="$seed" VORONOI_MESH_CASE_PARAM="$param" \
    "$BIN" --ignored --exact campaign_case --nocapture --test-threads=1 2>&1 \
    | grep -o 'CASERESULT .*' | head -1)"
  if [ -z "$line" ]; then
    echo "  FAIL $dist n=$n seed=$seed (no CASERESULT — process died, likely OOM)"
    echo "$dist,$n,$seed,$param,died,-,-,-,-,-,-" >> "$OUT"
    invalid=$((invalid + 1))
    return
  fi
  # Parse "key=value" tokens.
  local result defects post_repair no_chain valid peak origins
  result="$(sed -n 's/.* result=\([^ ]*\).*/\1/p' <<<"$line")"
  defects="$(sed -n 's/.* defects=\([^ ]*\).*/\1/p' <<<"$line")"
  post_repair="$(sed -n 's/.* post_repair=\([^ ]*\).*/\1/p' <<<"$line")"
  post_repair="${post_repair:--}"
  no_chain="$(sed -n 's/.* no_chain=\([^ ]*\).*/\1/p' <<<"$line")"
  no_chain="${no_chain:--}"
  valid="$(sed -n 's/.* valid=\([^ ]*\).*/\1/p' <<<"$line")"
  peak="$(sed -n 's/.* peak_mb=\([^ ]*\).*/\1/p' <<<"$line")"
  origins="$(sed -n 's/.* origins=\(.*\)$/\1/p' <<<"$line")"
  echo "$dist,$n,$seed,$param,$result,$defects,$post_repair,$no_chain,$valid,$peak,$origins" >> "$OUT"
  printf "  %-11s n=%-8s seed=%-3s result=%-3s defects=%-4s post_repair=%-4s no_chain=%-3s valid=%-5s peak=%sMB\n" \
    "$dist" "$n" "$seed" "$result" "$defects" "$post_repair" "$no_chain" "$valid" "$peak"
  if [ "$result" = "err" ]; then
    errored=$((errored + 1))
    invalid=$((invalid + 1))
  elif [ "$valid" != "true" ] || [ "$post_repair" != "0" ]; then
    invalid=$((invalid + 1))
  fi
}

# Uniform sweep matrix is env-overridable for targeted campaigns (e.g. a
# large-tier run hunting natural backstop triggers):
#   UNIFORM_SIZES="3000000 4500000 5000000" SEEDS="$(seq 1 15)" ...
UNIFORM_SIZES="${UNIFORM_SIZES:-100000 500000 1000000 2000000 3000000}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
echo "== uniform seed sweep =="
for n in $UNIFORM_SIZES; do
  for seed in $SEEDS; do
    run_case uniform "$n" "$seed" 0
  done
done

# Structured / adversarial distributions tuned to BUILD (stress
# near-degenerate geometry without exceeding the 64-vertex clip budget, so
# the result is real evidence about the net rather than an out-of-envelope
# error). `clustered_cap` is intentionally omitted: its sharp cap boundary
# overflows the vertex budget at any useful density (run it manually to
# confirm the out-of-envelope error path).
#   cocircular: 4*N near-cocircular points — the parity-defect class.
#   cube:       points clustered toward the 8 cube vertices.
#   bimodal:    two density regimes.
#   fibonacci:  structured near-uniform lattice with jitter.
echo "== structured / adversarial =="
for seed in 1 2 3 4 5; do
  run_case cocircular 50000  "$seed" 0.0001
  run_case cube       100000 "$seed" 0.01
  run_case bimodal    300000 "$seed" 0.5
  run_case fibonacci  500000 "$seed" 0.001
done

# Mega (a `param` fraction of points in a tiny cap) exercises the default
# Local3d escalation path. It has the same clean-success requirement as every
# other supported matrix case. Override sizes with MEGA_SIZES.
MEGA_SIZES="${MEGA_SIZES:-100000 500000 1000000}"
echo "== mega (default Local3d repair) =="
for n in $MEGA_SIZES; do
  for seed in 1 2 3 4 5; do
    run_case mega "$n" "$seed" 0.8
  done
done

echo
echo "SUMMARY: $total cases run, $invalid failed, $errored errors"
echo "Per-case rows in $OUT"
if [ "$invalid" -gt 0 ]; then
  echo "!! $invalid case(s) produced an invalid diagram or died — investigate" >&2
  exit 1
fi
