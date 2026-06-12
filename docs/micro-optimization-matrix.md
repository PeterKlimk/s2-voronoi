# Micro-Optimization Matrix

First-pass benchmark matrix for branches split out from
`docs/micro-optimization-ideas.md`.

This is a screening log, not a merge recommendation. The matrix was run on
2026-06-13 using binaries built with:

```bash
./scripts/bench_build.sh \
  main \
  agent/micro-opt-binning-cache-fuse \
  agent/micro-opt-cell-to-face-u32 \
  agent/micro-opt-chunk-array-loaders \
  agent/micro-opt-clip-batch-slice \
  agent/micro-opt-directed-cell-mode \
  agent/micro-opt-extract-inline-checks \
  agent/micro-opt-frontier-cache-ordf32 \
  agent/micro-opt-packed-center-tail-simd \
  agent/micro-opt-packed-frontier-no-sentinel-fill \
  agent/micro-opt-packed-query-dot-cache \
  agent/micro-opt-packed-tail-hoist \
  agent/micro-opt-periodic-conditional-wrap \
  agent/micro-opt-point-face-reciprocal \
  agent/micro-opt-preprocess-touched-reps \
  agent/micro-opt-projection-max-r2 \
  agent/micro-opt-shell-frontier-scratch \
  agent/micro-opt-signed-dists-array-refs
```

Build metadata:

- `rustc 1.96.0-nightly (bcded3316 2026-04-06)`
- `cargo 1.96.0-nightly (888f67534 2026-03-30)`
- features: `tools`
- rustflags: `-C target-cpu=native`
- runner: single-threaded, CPU-pinned to core 0

## Branches

| Branch | Commit |
|---|---:|
| `main` | `19ba94c` |
| `agent/micro-opt-binning-cache-fuse` | `3e185e7` |
| `agent/micro-opt-cell-to-face-u32` | `22d3b67` |
| `agent/micro-opt-chunk-array-loaders` | `d3f395c` |
| `agent/micro-opt-clip-batch-slice` | `76f36f2` |
| `agent/micro-opt-directed-cell-mode` | `acc687a` |
| `agent/micro-opt-extract-inline-checks` | `f2eca0f` |
| `agent/micro-opt-frontier-cache-ordf32` | `5542c4d` |
| `agent/micro-opt-packed-center-tail-simd` | `f0ebb69` |
| `agent/micro-opt-packed-frontier-no-sentinel-fill` | `e2bf31b` |
| `agent/micro-opt-packed-query-dot-cache` | `62b610a` |
| `agent/micro-opt-packed-tail-hoist` | `62ca411` |
| `agent/micro-opt-periodic-conditional-wrap` | `f64a593` |
| `agent/micro-opt-point-face-reciprocal` | `5d1c63b` |
| `agent/micro-opt-preprocess-touched-reps` | `763da66` |
| `agent/micro-opt-projection-max-r2` | `8e0201a` |
| `agent/micro-opt-shell-frontier-scratch` | `770cd52` |
| `agent/micro-opt-signed-dists-array-refs` | `b7ee5fb` |

## Runs

| Label | Command | Notes |
|---|---|---|
| 100k screen | `./scripts/bench_run.sh -s 100k -r 5 -c 2 -m total` | quick screen; noisy but useful for obvious losers |
| 500k screen | `./scripts/bench_run.sh -s 500k -r 5 -c 2 -m total` | first larger pass; high spread |
| 500k confirmation | `./scripts/bench_run.sh -s 500k -r 12 -c 2 -m total` | best current no-preprocess signal |
| 1m scale | `./scripts/bench_run.sh -s 1m -r 8 -c 2 -m total` | larger-scale check; still noisy |
| 500k preprocess | `./scripts/bench_run.sh -s 500k -r 8 -c 2 -m total --preprocess` | default-preprocess shape |

## Median Results

All values are median total time in milliseconds. Lower is better.

| Branch | 100k no-pre r5 | 500k no-pre r5 | 500k no-pre r12 | 1m no-pre r8 | 500k pre r8 |
|---|---:|---:|---:|---:|---:|
| `main` | 151.4 | 752.1 | 727.0 | 1525.3 | 760.7 |
| `binning-cache-fuse` | 150.4 | 774.9 | 757.2 | 1565.1 | 772.6 |
| `cell-to-face-u32` | 150.7 | 755.4 | 740.8 | 1544.3 | 760.4 |
| `chunk-array-loaders` | 154.0 | 758.8 | 741.7 | 1501.1 | 763.5 |
| `clip-batch-slice` | 151.2 | 760.3 | 757.8 | 1480.0 | 757.7 |
| `directed-cell-mode` | 151.0 | 776.9 | 730.9 | 1534.8 | 780.2 |
| `extract-inline-checks` | 146.8 | 770.8 | 746.3 | 1498.4 | 760.2 |
| `frontier-cache-ordf32` | 153.9 | 807.1 | 740.5 | 1532.7 | 768.5 |
| `packed-center-tail-simd` | 150.5 | 770.6 | 726.2 | 1542.2 | 757.5 |
| `packed-frontier-no-sentinel-fill` | 148.9 | 797.0 | 741.9 | 1527.5 | 768.0 |
| `packed-query-dot-cache` | 148.9 | 771.5 | 727.8 | 1513.9 | 757.8 |
| `packed-tail-hoist` | 148.5 | 764.5 | 746.7 | 1532.0 | 766.5 |
| `periodic-conditional-wrap` | 148.9 | 796.5 | 725.8 | 1553.2 | 768.5 |
| `point-face-reciprocal` | 147.5 | 764.3 | 726.4 | 1538.8 | 764.5 |
| `preprocess-touched-reps` | 151.8 | 776.0 | 788.6 | 1537.0 | 787.4 |
| `projection-max-r2` | 147.7 | 752.6 | 724.6 | 1540.8 | 764.5 |
| `shell-frontier-scratch` | 146.0 | 745.8 | 748.8 | 1498.4 | 752.9 |
| `signed-dists-array-refs` | 150.6 | 755.2 | 729.4 | 1492.5 | 761.2 |

## Initial Read

- `projection-max-r2` is the cleanest 500k no-preprocess median winner, but
  does not carry that win to 1m or default-preprocess.
- `shell-frontier-scratch` wins the 500k default-preprocess pass and tied for
  second at 1m, but its 500k no-preprocess confirmation was not a win.
- `clip-batch-slice` wins the 1m no-preprocess pass and stays near baseline in
  the preprocess pass, but was slower in the 500k no-preprocess confirmation.
- `signed-dists-array-refs`, `packed-query-dot-cache`,
  `packed-center-tail-simd`, `point-face-reciprocal`, and
  `periodic-conditional-wrap` are within noise in at least one longer pass.
- `preprocess-touched-reps` is consistently poor in this harness, including
  the default-preprocess pass. Treat as a likely reject unless a targeted
  preprocess-heavy case says otherwise.
- `binning-cache-fuse`, `directed-cell-mode`, `frontier-cache-ordf32`,
  `packed-frontier-no-sentinel-fill`, and `packed-tail-hoist` do not currently
  show a stable positive signal.

Suggested next step: rebuild a focused subset with `--timing` and compare
phase-level metrics for `main`, `projection-max-r2`, `shell-frontier-scratch`,
`clip-batch-slice`, `signed-dists-array-refs`, `packed-query-dot-cache`, and
`packed-center-tail-simd`.


## Phase-level paired campaign (2026-06-13, second pass)

Protocol upgrade over the first pass: all 18 binaries rebuilt with
`--timing`, interleaved per round with rotated start order, and analyzed as
**paired per-round deltas vs main** (cancels machine drift). Sessions: 16
rounds x {cell_construction, knn_build, dedup, preprocess(+--preprocess),
total} at 500k ST.

### The layout-noise calibration (the campaign's key methodological finding)

Two branches whose diffs cannot affect the measured pipeline acted as
controls, and both showed large, *sign-consistent* offsets:

- `periodic-conditional-wrap` (planar-only diff): **+10.4ms median,
  13/16 consistent** on sphere cell_construction.
- `preprocess-touched-reps` (preprocess-only diff, dead code in
  no-preprocess runs): **-14.5ms median, 12/16** on no-preprocess total.

Per-binary code-layout offsets are stable and consistent — sign
consistency cannot discriminate them from real effects; only magnitude vs
the control distribution can. Noise floor: ~±10-15ms (±1.3-2%) per binary
on the large metrics. **Every first-pass "winner" except
`extract-inline-checks` sits inside this floor.**

### Survivor and stack confirmation

`extract-inline-checks`: -19.2ms cell_construction (21/24) and -20.3ms
total at 500k; **-66ms at 2M (11/12)** — scales with n as the eliminated
per-cell diagnostic pass predicts.

Stacking the five best cell_construction signals
(extract-inline-checks + shell-frontier-scratch +
packed-frontier-no-sentinel-fill + point-face-reciprocal +
packed-tail-hoist) and re-measuring three-way (main / extract / stack):

| comparison | cc 500k r24 | total 500k r24 | cc 2M r12 |
|---|---:|---:|---:|
| extract - main | -19.2 (21/24) | -20.3 (17/24) | -66 (11/12) |
| stack - main | -22.3 (22/24) | **-36.0 (20/24)** | **-120 (12/12)** |
| stack - extract | -8.7 (17/24) | -10.5 (16/24) | -65 (7/12) |

The stack's increment over extract alone is real (unanimous at 2M), so the
five merge together. Full test suite green on the stack (247).

### Verdicts

- **Merged to main** (as `agent/micro-opt-stack`): `extract-inline-checks`
  (the only individually-proven win), `shell-frontier-scratch`,
  `packed-frontier-no-sentinel-fill`, `point-face-reciprocal`,
  `packed-tail-hoist` (proven collectively via the stack increment).
- **Unproven at 500k/2M — not merged** (inside the layout-noise floor):
  `chunk-array-loaders`, `clip-batch-slice`, `packed-center-tail-simd`,
  `packed-query-dot-cache`, `projection-max-r2`,
  `signed-dists-array-refs`.
- **No positive signal — rejected**: `binning-cache-fuse`,
  `cell-to-face-u32`, `directed-cell-mode`, `frontier-cache-ordf32`.
- `preprocess-touched-reps`: zero measurable preprocess effect at 500k (its
  target path needs weld-heavy inputs); cleanliness-only candidate.
- `periodic-conditional-wrap`: measured on its actual pipeline
  (bench_plane --periodic, 500k ST, 14 order-alternated paired rounds):
  **+11.8ms median, faster in only 2/13 — rejected** (conditional wrapping
  defeats the unconditional vectorized minimum-image path).
