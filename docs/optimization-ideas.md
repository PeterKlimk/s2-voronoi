# Optimization Ideas Ledger

Status: compact performance ledger, 2026-06-18.

This file is the short index of measured optimization state. It intentionally
does not keep full experiment transcripts; use git history for archaeology.

Measurement rules:

- Prefer hardware counters (`perf stat`) for behavior decisions; time alone is
  noisy on this machine.
- Use paired/interleaved runs for sub-percent changes.
- Validate release builds and relevant correctness tests before treating a
  branch as parked or promising.
- Do not retry rejected ideas without a new theorem, workload, or measurement
  flaw.

## Where Ideas Live

| topic | source |
|---|---|
| Algorithmic ST headroom, edge-passed information, candidate reuse | `docs/st-headroom-and-fade-comparison.md` |
| Dense/sparse/non-uniform regime strategy | `docs/multi-regime-perf.md` |
| Dense-cell local index summary | `docs/dense-cell-subindex-design.md` |
| Punch 1 implemented center-cell band prune | `docs/punch1-center-cell-integration.md` |
| Paired micro-optimization evidence | `docs/micro-optimization-matrix.md` |
| Raw micro-idea backlog | `docs/micro-optimization-ideas.md` |
| Edge-check matching (parked: near-optimal for sane regime) | `docs/edgecheck-matching-optimization.md` |
| Post-clipping pipeline (cert + edge_emit: both rejected) | `docs/post-clipping-pipeline-optimization.md` |
| Research-level: incremental support cert, dot reuse, batched clip | `docs/research-level-optimization.md` |
| Roadmap priorities | `docs/todo.md` |

## Current High-Value Threads

| idea | status | note |
|---|---|---|
| Dense-cell local index (`punch 1`) | implemented, review/merge-gated | Axis-sort center-cell band prune exists. It gives a large cap-pathology win and is kept neutral elsewhere by the rebuild gate. See `docs/punch1-center-cell-integration.md`. |
| Directional shell/cap certificate | parked promising | Branch `agent/directional-certificates` (`f51523a`) / gated follow-up `agent/directional-cell-cap-gate-audit` (`3eaf1c4`) is roughly equal on fib and faster in mega; keep default-off until productized. |
| Candidate-production certificate work | open, narrow | The remaining ST prize is avoiding candidate examination with already-available metadata, not another late per-candidate support probe. |
| Regime-aware candidate engine | synthesis idea | Long-term shape where packed, shell, dense local index, and certificates share one frontier contract. See `docs/multi-regime-perf.md`. |

## Occupancy Rebuild Re-Calibration

The sphere grid has a one-shot occupancy-feedback rebuild. The old max-occupancy
trigger fired too eagerly and hurt moderate clusters. The current trigger uses
the candidate-scan work proxy:

```text
sum_sq_per_n = Σ(cell_occupancy²) / n
```

Current policy:

- Rebuild when `sum_sq_per_n > 500`.
- Aim the fullest rebuilt cell at about `GRID_REBUILD_TARGET_MAX_OCC = 192`.
- Cap total grid cells at `GRID_MAX_CELLS_PER_POINT = 8.0`.

Quiet-box calibration found the beneficial crossover near `sum_sq_per_n ≈ 450`
across cluster shapes:

| input | sum_sq_per_n | rebuild verdict |
|---|---:|---|
| mega frac0.05 | 102 | hurts |
| splittable 1m | 274 | hurts |
| mega frac0.1 | 331 | hurts |
| splittable 500k | 536 | helps |
| mega frac0.15 | 712 | helps |
| mega frac0.2 | 1244 | helps |

Why this matters: global re-gridding is only right when dense regions dominate
candidate-scan work. Minority hotspots want a local dense-cell index, not a
global resolution shift that de-tunes sparse background cells.

## Dense-Cell Local Index

Assessed state:

- A per-dense-cell local index is synergistic with the occupancy rebuild, not a
  replacement.
- The rebuild handles globally dense majority inputs cheaply.
- The implemented axis-sort center-cell band prune handles residual over-full
  center cells after rebuild. It attacks the cap-style O(occ²) packed center
  gather while the shell takeover remains the correctness backstop.
- Production currently clears the dense index when no rebuild fired, keeping
  clustered/outlier cases on the baseline path where the band was measured as
  overhead.
- The kd-tree/lazy stream version remains an escalation option only if the
  axis-sort band is not enough for real dense workloads.

See `docs/punch1-center-cell-integration.md` for the implemented behavior and
`docs/dense-cell-subindex-design.md` for the compact design background.

## Completed Wins Worth Remembering

| area | result |
|---|---|
| Integrated sphere weld detection/compaction | Reused the query grid as the weld detector and compacted welded grids in place. Preprocess dropped from about 378 ms to 45 ms at 2M ST in the measured run. |
| Sparse edge repair | Replaced dense union-find/rebuild repair with sparse/in-place repair. Defect-bearing 2M seed-1 repair dropped from about 382 ms to 0.06 ms. |
| Strict zero clip epsilon sqrt removal | `HalfPlane::new_unnormalized_base_eps` skips the norm sqrt when `base_eps == 0.0`; semantics are identical in production. |
| Micro-opt stack | Extract-inline-checks, shell-frontier scratch, no sentinel fill, reciprocal face projection, packed-tail hoist. See `docs/micro-optimization-matrix.md`. |
| Vectorized clip distance pass | `fp::signed_dists_mask8` replaced scalar distance loops in clipper hot paths; measured positive after clean paired reruns. |
| Small-N sorting networks | Promoted always-on after paired confirmation. |
| Fused hi/tail split | Thresholds are computed before center pass; dead demotion/min-center bookkeeping removed. |
| Periodic packed port | Packed SIMD stage roughly halved periodic 500k ST time in the measured round. |
| Punch 1 axis-sort center-cell band prune | Replaces full dense center-cell gather with a certificate-safe band on rebuilt dense grids. Cap 25k measured about 106s to 6.2s; uniform/regime matrix appears neutral with the rebuild gate. |

## Deprioritized But Not Forgotten

### Live within-bin edge repair

Deprioritized. It would mutate already-emitted shard CSR, disputed endpoint
vertices can be owned by other bins, and the proof overlaps with broader P5
consistency-by-construction work. Existing edgecheck coverage audits found no
same-bin handoff leak in tested regimes.

### Plane occupancy rebuild

Lower priority. The sphere rebuild only helps when dense regions dominate
candidate-scan work; moderate cases degrade gracefully and global re-grid can
hurt. Porting to the plane is only worth it for a real majority-concentration
planar workload.

### Angular-sweep clipper

Counter-probed on `codex/angular-sweep-clipping` (2026-06-18) with an opt-in
timing audit. The audit used the existing 64-sector angular support envelope
before each bounded stream clip, then compared against the real clip result.
It is software counters only; wall time was intentionally ignored.

| dist | neighbors | audited candidates | exact unchanged | support hits | support misses | false positives |
|---|---:|---:|---:|---:|---:|---:|
| fib | 819,121 | 381,139 | 203,530 | 176,608 | 26,922 | 0 |
| uniform | 991,724 | 528,853 | 354,369 | 315,050 | 39,319 | 0 |
| splittable | 1,571,515 | 1,076,591 | 902,641 | 847,575 | 55,066 | 0 |
| mega 0.8 | 3,709,805 | 3,133,815 | 2,946,563 | 2,883,619 | 62,944 | 0 |

Signal: the angular envelope can conservatively identify 21.6% / 31.8% /
53.9% / 77.7% of all processed-neighbor clips as no-op in these regimes, and
captures 86.8% / 88.9% / 93.9% / 97.9% of the exact unchanged clips. The
opportunity is real, especially on dense inputs.

Actual 64-sector skip, however, was rejected. `codex/angular-sweep-clipping`
commit `a1236a3` turned support hits into real skips; same-binary `perf stat
-r 3`, 100k ST, env off vs env on:

| dist | skipped clips | instructions | cycles | branches | branch misses |
|---|---:|---:|---:|---:|---:|
| fib | 176,608 | +160.3% | +139.5% | +139.9% | +28.6% |
| uniform | 315,050 | +121.8% | +89.4% | +106.5% | +20.3% |
| splittable | 847,575 | +66.6% | +70.0% | +63.7% | +20.2% |
| mega 0.8 | 2,883,619 | +33.9% | +43.9% | +26.6% | +24.4% |

Verdict: reject the 64-sector support-cache implementation as a runtime skip.
The skipped-clip signal is real, but the per-candidate support test (bisector
projection, sectoring, and frequent support-cache rebuilds) overwhelms the
savings even in the dense mega case.

Follow-up cheap variant on `codex/cheap-angular-skip` commit `c2b2ef9`:
`S2_VORONOI_RADIUS_SKIP=1` uses the existing polygon radius bound before
entering the clipper, gated to cells with at least 4 accepted constraints.
Same-binary `perf stat -r 5`, 100k ST, env off vs env on:

| dist | radius tests | radius hits | instructions | cycles | branches |
|---|---:|---:|---:|---:|---:|
| fib | 373,110 | 48,637 | +0.07% | -0.96% | +0.60% |
| uniform | 528,567 | 48,256 | +0.16% | +0.22% | +0.77% |
| splittable | 1,076,941 | 139,692 | +0.08% | -4.64% | +0.82% |
| mega 0.8 | 3,129,204 | 1,364,073 | -0.85% | +2.11% | +0.05% |

Verdict: plausible but marginal, not a clear production win. This is far more
competitive than the 64-sector support-cache skip because it has no rebuilds
and reuses the cheap radius certificate, but the upside is only visible in the
mega dense case and is still below 1% instructions. Keep it parked as an
env-gated micro-probe unless a broader dense-regime gate can make the win
decision-grade.

## Tried And Rejected

| idea | verdict |
|---|---|
| Packed partial selection with unsorted batches and suffix-min bounds | 7-14% loss at 2M bounded ST. Nearest-first order shrinks polygons faster and is worth the sort. |
| `fma` feature by default | 25-35% loss without native FMA codegen; indistinguishable from non-FMA under `target-cpu=native`. Keep off. |
| Ring-pass cap pruning | Net loss. Adjacent ring cells almost always straddle the threshold, so cap tests rarely skip work. |
| `PACKED_HI_BUDGET` retune | 12 collapses due to tail builds; 16-32 are flat. Keep 32. |
| Morton cell traversal order | Branch `codex/locality-morton-perf` commit `2f275b2` added `S2_VORONOI_CELL_ORDER=morton`, assigning within-bin locals by Morton cell order while leaving grid storage row-major. Same-binary perf rejected it: 2M fib ST default rerun 17.76B instructions / 9.57B cycles / 142.8M L1D misses vs Morton 18.03B / 10.12B / 145.0M; 2M fib MT 17.67B / 13.44B / 276.6M vs Morton 18.13B / 14.38B / 285.8M; 500k splittable ST 10.39B / 5.04B / 81.0M vs Morton 10.86B / 6.27B / 113.4M. Mechanism: processing order changes, but the cell/slot arrays remain row-major, so Morton often defeats sequential prefetch without reducing candidates. |
| Morton cell storage order | Branch `codex/locality-morton-perf` commit `73049b8` added `S2_VORONOI_CELL_STORAGE=morton`, making the cube cell id itself Morton-ranked within each face so `cell_offsets`, point slots, and cell-indexed arrays are physically reordered. Also rejected: 2M fib ST default 18.00B instructions / 9.50B cycles / 135.0M L1D misses vs Morton-storage 18.09B / 10.08B / 153.1M; 2M fib MT 17.92B / 13.38B / 270.8M vs 18.30B / 13.62B / 273.8M; 500k splittable ST 10.53B / 5.14B / 83.6M vs 10.86B / 5.21B / 88.6M. Candidate counts stayed flat, and the storage-order decode/build overhead plus loss of row-major prefetch outweighed any 2D-locality gain. |
| Tiled cell storage order | Branch `codex/locality-tiled-storage` commit `41fb3e8` generalized `S2_VORONOI_CELL_STORAGE=tile{N}` / `tile{U}x{V}`. The best-looking timing probe was `tile4`, but non-timing counters rejected it: 2M fib ST default 18.10B instructions / 9.47B cycles / 141.1M L1D misses vs tile4 18.27B / 9.62B / 145.0M; 2M fib MT default 18.19B / 13.68B / 280.9M vs tile4 18.22B / 13.44B / 275.2M (noise-sized cycle/cache shift, no instruction win); 500k splittable ST default 10.60B / 5.09B / 79.2M vs tile4 10.96B / 5.12B / 77.7M. `tile2` also lost at 2M fib ST (18.28B / 9.57B / 144.5M). Tiling can trim some cache misses in clustered cases, but the extra decode/order overhead and reduced row prefetch erase it. |
| Planar `S2_BIN_COUNT` increase | No shard-balance win; earlier apparent MT gap was a measurement-window artifact. |
| Plane scatter/detect weld fusion | Rejected on analysis. Cross-cell wall-band pairs still require a pass, and fusing would lose parallelism. |
| Candidate prefix/seed reuse | Negative or audit-negative. See `docs/st-headroom-and-fade-comparison.md`. |
| Edgecheck endpoint/anchored seed ordering | Negative despite real geometric signal. See `docs/st-headroom-and-fade-comparison.md`. |
| Distance-symmetry certificate seeding | No termination signal in tested regimes. |
| Late known-batch directional support probing | The shadow signal is real, but the late support-probe path was too costly. The parked shell/cap certificate branches are the useful continuation point. |

## Benchmarks To Keep In Rotation

Use these so changes do not optimize only the uniform case:

```bash
RAYON_NUM_THREADS=1 perf stat -r 5 -e instructions,cycles,branches,branch-misses \
  target/release/bench_voronoi 100k --no-preprocess

RAYON_NUM_THREADS=1 perf stat -r 5 -e instructions,cycles,branches,branch-misses \
  target/release/bench_voronoi 100k --no-preprocess --dist splittable

RAYON_NUM_THREADS=1 perf stat -r 5 -e instructions,cycles,branches,branch-misses \
  target/release/bench_voronoi 100k --no-preprocess --dist mega --dist-param 0.8
```

For timing counters:

```bash
RAYON_NUM_THREADS=1 S2_VORONOI_TIMING_KV=1 \
  cargo run --release --features tools,timing --bin bench_voronoi -- \
  100k --no-preprocess --dist mega --dist-param 0.8
```

## One-Line Summary

Keep uniform fast, fix dense regimes behind measured gates, and spend new
algorithmic effort on avoiding candidate examination rather than rearranging
where the same clips happen.
