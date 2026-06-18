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

Still conceptually valid but demoted. Same-bin handoff already removes much of
the duplicate fresh clipping it would optimize, and certificate/candidate work
is a better target.

## Tried And Rejected

| idea | verdict |
|---|---|
| Packed partial selection with unsorted batches and suffix-min bounds | 7-14% loss at 2M bounded ST. Nearest-first order shrinks polygons faster and is worth the sort. |
| `fma` feature by default | 25-35% loss without native FMA codegen; indistinguishable from non-FMA under `target-cpu=native`. Keep off. |
| Ring-pass cap pruning | Net loss. Adjacent ring cells almost always straddle the threshold, so cap tests rarely skip work. |
| `PACKED_HI_BUDGET` retune | 12 collapses due to tail builds; 16-32 are flat. Keep 32. |
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
