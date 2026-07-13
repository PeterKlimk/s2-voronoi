#!/usr/bin/env bash
#
# Geometric-fidelity evidence campaign: one release-test process per case.
# Output is a line-oriented key/value ledger rather than a correctness oracle.
# Every case must satisfy strict topology; angular/residual values are recorded
# for AUD-011 analysis and are intentionally not thresholded here.
#
# Usage:
#   ./scripts/fidelity_campaign.sh [output.txt]
#
# Environment:
#   SEEDS="1 2 3"                  random seeds (default: 1 2 3)
#   FIDELITY_CELLS=1024            sampled cells per case
#   FIDELITY_EDGE_SAMPLES=3        interior samples per edge (endpoints always included)
#   RAYON_NUM_THREADS=6            intra-case parallelism

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-/tmp/fidelity_campaign.txt}"
SEEDS="${SEEDS:-1 2 3}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-6}"
export VORONOI_MESH_FIDELITY_CELLS="${FIDELITY_CELLS:-1024}"
export VORONOI_MESH_FIDELITY_EDGE_SAMPLES="${FIDELITY_EDGE_SAMPLES:-3}"

echo "Building fidelity campaign binary..."
build_output="$(cargo test --release --features tools --test fidelity_campaign --no-run 2>&1)"
tail -1 <<< "$build_output"
BIN="$(sed -n 's/.*Executable tests\/fidelity_campaign.rs (\(.*\))/\1/p' \
  <<< "$build_output" | tail -1)"
if [[ -z "${BIN:-}" || ! -x "$BIN" ]]; then
  echo "ERROR: could not locate fidelity campaign binary" >&2
  exit 1
fi

: > "$OUT"
failures=0
total=0

run_case() {
  local dist="$1" n="$2" seed="$3" param="$4"
  total=$((total + 1))
  local raw status=0 line
  raw="$(VORONOI_MESH_CASE_DIST="$dist" VORONOI_MESH_CASE_N="$n" \
    VORONOI_MESH_CASE_SEED="$seed" VORONOI_MESH_CASE_PARAM="$param" \
    "$BIN" --ignored --exact fidelity_case --nocapture --test-threads=1 2>&1)" || status=$?
  line="$(grep -o 'FIDELITYRESULT .*' <<< "$raw" | head -1 || true)"
  if [[ -z "$line" ]]; then
    echo "FAIL dist=$dist n=$n seed=$seed status=$status (no FIDELITYRESULT)" | tee -a "$OUT"
    tail -12 <<< "$raw"
    failures=$((failures + 1))
    return
  fi

  # Keep the complete machine-readable record in the ledger, but make live
  # progress human-sized. The full records contain all conditioning buckets
  # and are deliberately much too wide for a terminal status stream.
  echo "$line" >> "$OUT"
  local merged pre_defects repair_attempted repair_accepted ownership edge_max
  merged="$(sed -n 's/.* merged=\([^ ]*\).*/\1/p' <<< "$line")"
  pre_defects="$(sed -n 's/.* pre_defects=\([^ ]*\).*/\1/p' <<< "$line")"
  repair_attempted="$(sed -n 's/.* repair_attempted=\([^ ]*\).*/\1/p' <<< "$line")"
  repair_accepted="$(sed -n 's/.* repair_accepted=\([^ ]*\).*/\1/p' <<< "$line")"
  ownership="$(sed -n 's/.* ownership_mismatches=\([^ ]*\).*/\1/p' <<< "$line")"
  edge_max="$(sed -n 's/.* edge_cross_max_rad=\([^ ]*\).*/\1/p' <<< "$line")"
  echo "OK dist=$dist n=$n seed=$seed merged=$merged pre_defects=$pre_defects repair=$repair_attempted/$repair_accepted ownership=$ownership edge_max_rad=$edge_max"
}

for seed in $SEEDS; do
  run_case uniform      100000 "$seed" 0
  run_case fibonacci    100000 "$seed" 0.001
  run_case clustered    100000 "$seed" 0.05
  run_case bimodal      100000 "$seed" 0.1
  run_case cube         100000 "$seed" 0.01
  run_case cocircular    25000 "$seed" 0.0001
  run_case mega         100000 "$seed" 0.8
  run_case hemisphere   100000 "$seed" 0
  run_case cap          100000 "$seed" 0.05
  run_case welded       100000 "$seed" 0
done

# Fully deterministic structured degeneracy needs only one run. Exact
# great-circle input is small because it exercises the explicit perturbation
# policy rather than a scaling property.
run_case grid         100000 0 0
run_case great_circle   1000 1 0

echo "SUMMARY total=$total failures=$failures output=$OUT"
if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
