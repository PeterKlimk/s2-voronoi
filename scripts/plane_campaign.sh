#!/usr/bin/env bash
#
# Planar robustness evidence campaign driver: one process per case.
#
# Twin of robustness_campaign.sh for the plane backend. Sweeps seeds / sizes
# / distributions through the `plane_campaign_case` test, forking a fresh
# process per case so the OS reclaims memory fully between builds.
#
# RULE-AGNOSTIC: the clip rule in force is whatever the binary was COMPILED
# with (`PLANE_CLIP_EPS_INSIDE` in src/tolerances.rs). To run the strict
# campaign — the gate before flipping that default — set the constant to 0.0,
# rebuild (this script rebuilds), run, then revert. The script reads the
# constant and prints which rule is active so the evidence is self-labelling.
#
# Usage:
#   ./scripts/plane_campaign.sh [output.csv]
#
# Environment:
#   MAX_N              cap on case size (default 2000000)
#   RAYON_NUM_THREADS  intra-build parallelism (default 6)
#   UNIFORM_SIZES      override uniform sweep sizes
#   SEEDS              override seed list
#
# Each built case is asserted strictly valid; an invalid build fails that
# case (flagged INVALID). Out-of-envelope errors are recorded as `err`.

set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-/tmp/plane_campaign.csv}"
MAX_N="${MAX_N:-2000000}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-6}"

# Self-label the active clip rule from the source constant.
RULE_LINE="$(grep -n 'const PLANE_CLIP_EPS_INSIDE' src/tolerances.rs | head -1)"
RULE_VAL="$(sed -n 's/.*PLANE_CLIP_EPS_INSIDE: f64 = \([^;]*\);.*/\1/p' src/tolerances.rs | head -1)"
if [ "$RULE_VAL" = "0.0" ]; then
  RULE="STRICT (d >= 0)"
else
  RULE="BIAS (eps=$RULE_VAL)"
fi
echo "Clip rule compiled in: $RULE   [$RULE_LINE]"

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

echo "rule,dist,n,seed,param,result,defects,valid,peak_mb" > "$OUT"

invalid=0
errored=0
total=0

run_case() { # dist n seed param [rect]
  local dist="$1" n="$2" seed="$3" param="$4" rect="${5:-}"
  if [ "$n" -gt "$MAX_N" ]; then
    echo "  skip $dist n=$n seed=$seed (> MAX_N=$MAX_N)"
    return
  fi
  total=$((total + 1))
  local line
  # NB: route through `env` so the optional, expansion-produced
  # S2_PLANE_RECT token is taken as an assignment (a KEY=VALUE word produced
  # by expansion is otherwise treated as the command name, not a var prefix).
  line="$(env S2_PLANE_DIST="$dist" S2_CASE_N="$n" S2_CASE_SEED="$seed" S2_CASE_PARAM="$param" \
    ${rect:+S2_PLANE_RECT="$rect"} \
    "$BIN" --ignored --exact plane_campaign_case --nocapture --test-threads=1 2>&1 \
    | grep -o 'PLANECASE .*' | head -1)"
  if [ -z "$line" ]; then
    echo "  FAIL $dist n=$n seed=$seed (no PLANECASE — process died, likely OOM)"
    echo "$RULE,$dist,$n,$seed,$param,died,-,-,-" >> "$OUT"
    invalid=$((invalid + 1))
    return
  fi
  local result defects valid peak
  result="$(sed -n 's/.* result=\([^ ]*\).*/\1/p' <<<"$line")"
  defects="$(sed -n 's/.* defects=\([^ ]*\).*/\1/p' <<<"$line")"
  valid="$(sed -n 's/.* valid=\([^ ]*\).*/\1/p' <<<"$line")"
  peak="$(sed -n 's/.* peak_mb=\([^ ]*\).*/\1/p' <<<"$line")"
  echo "$RULE,$dist,$n,$seed,$param,$result,$defects,$valid,$peak" >> "$OUT"
  printf "  %-10s n=%-8s seed=%-3s result=%-3s defects=%-4s valid=%-5s peak=%sMB\n" \
    "$dist" "$n" "$seed" "$result" "$defects" "$valid" "$peak"
  [ "$result" = "err" ] && errored=$((errored + 1))
  [ "$valid" = "false" ] && invalid=$((invalid + 1))
}

UNIFORM_SIZES="${UNIFORM_SIZES:-2000 20000 80000 200000 500000 1000000}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
echo "== uniform seed sweep =="
for n in $UNIFORM_SIZES; do
  for seed in $SEEDS; do
    run_case uniform "$n" "$seed" 0.0
  done
done

# Periodic torus sweep (the override-uncovered path; only the compiled
# constant flips it, so this is the periodic strict evidence).
PERIODIC_SEEDS="${PERIODIC_SEEDS:-1 2 3 4 5}"
echo "== periodic torus sweep =="
for n in 5000 50000 200000; do
  for seed in $PERIODIC_SEEDS; do
    run_case periodic "$n" "$seed" 0.0
  done
done

# Sliver / tie / parity stressors — the classes the strict rule historically
# broke on (this is where strict-plane risk concentrates, so STRUCT_SEEDS is
# where to put campaign volume). clustered jitter is swept tight-to-loose;
# lattice is exact ties; cocircular is the planar parity-defect class.
STRUCT_SEEDS="${STRUCT_SEEDS:-1 2 3 4 5}"
echo "== structured / adversarial =="
for seed in $STRUCT_SEEDS; do
  run_case clustered  400    "$seed" 0.01
  run_case clustered  400    "$seed" 0.001
  run_case clustered  400    "$seed" 0.0005
  run_case clustered  2000   "$seed" 0.005
  run_case clustered  2000   "$seed" 0.001
  run_case lattice    40000  "$seed" 0.0
  run_case cocircular 20000  "$seed" 0.0001
  run_case cocircular 20000  "$seed" 0.00001
done
# A few non-unit rects: aspect + offset (domain-transform interplay).
for seed in 1 2 3; do
  run_case uniform 80000 "$seed" 0.0 "0,0,200,1"
  run_case uniform 80000 "$seed" 0.0 "-7.3,2.9,11.1,5.7"
done

echo
echo "SUMMARY [$RULE]: $total cases run, $invalid invalid, $errored out-of-envelope errors"
echo "Per-case rows in $OUT"
if [ "$invalid" -gt 0 ]; then
  echo "!! $invalid case(s) produced an invalid diagram or died — investigate" >&2
  exit 1
fi
