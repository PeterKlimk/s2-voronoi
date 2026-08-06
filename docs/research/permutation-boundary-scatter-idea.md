# Permutation-boundary scatter: the third option

**Status:** retired after measurement (2026-07-26). The prototype passed its isolated phase gate
with 128 destination windows, but regressed the correlated Fibonacci and `cubed` controls and added
retired work in every ordinary regime. The implementation and its private mode switch were
removed. See [Experiment result](#experiment-result).

Originally written after a read-only pass over `performance.md`,
`algorithmic-performance-ideas.md`, `memory-layout-ideas.md`, and the kernel experiment log.

[`work-log.md`](../work-log.md) remains the authoritative queue. This document should move a candidate
into the narrow experiment queue in [`performance.md`](../performance.md#open-optimization-queue), or
into the retired list, only after the gate below has run.

## Why this boundary and not the kernel

The July 2026 kernel pass is closed
([`kernel-optimization-experiment-log.md`](../internal/kernel-optimization-experiment-log.md#pass-closeout)):
every shortlisted production hypothesis was rejected, and the surviving ceilings are all around one
percent. Demand-sized prefixes have a real 44–63% first-batch abandonment premise but a fixed
8-slot request cost +2.777% instructions on uniform. Regional local hull did not reach 74% of even
the optimistic pair-count floor. Threshold correction was rejected twice and is explicitly closed to
new sampling geometries.

Meanwhile the two largest recent accepted wins were not kernel work at all. Adaptive grid
coordinate materialization reduced 1M uniform cycles 8.27% and cache misses 26.67%; adaptive final
index scatter improved 2M multithreaded uniform 1.12% and 1M single-threaded uniform 2.72%. Both are
*address-order* decisions at a phase boundary. That is where the remaining headroom is, and the
claim of this document is that the family was only half-explored.

## The framing gap

[`performance.md`](../performance.md#open-optimization-queue) states the accepted policy as a choice
between two traversals:

> Generator-order traversal gives sequential destination writes but can jump among shard source
> streams; shard-order traversal gives sequential source reads but scatters the final writes.

The code implements exactly that exclusive-or. `live_dedup/assemble.rs:529` is the shard-order
branch and `live_dedup/assemble.rs:571` the generator-order branch; they differ only in which side
loses locality. `cube_grid/build.rs:409` versus `cube_grid/build.rs:427` is the same shape at the
grid boundary — scatter-fused versus scatter-ids-then-gather-coordinates.

The transform at both boundaries is a genuine permutation, not a mergeable ordering.
`live_dedup/binning.rs` assigns `LocalId` in cell-major order:

> locals are assigned in cell-major order, the invariant the directed edge-check scheduling relies
> on

so shard-local rank and generator id are uncorrelated for any input that is not already spatially
sorted. That is precisely what `prefer_shard_order_scatter` (`live_dedup/assemble.rs:81`) measures
and what the `grid_order_abs_delta` / `shard_order_abs_delta` telemetry reports.

For a true permutation, one pass cannot make both sides sequential. The standard answer is two
passes: **bucket-partition by destination, then fill each bucket inside a cache-resident window.**
Neither the policy audit nor the retired list considers it. The accepted adaptive policy chooses
the cheaper loss; it never removes the loss.

## Size of the prize

From the existing attribution in [`performance.md`](../performance.md#open-optimization-queue), at 2M
on the eight-thread Intel Mac:

| Phase | Fibonacci | Uniform |
|---|---:|---:|
| Attributed final index scatter | 16–18 ms | 42–48 ms |
| Vertex concatenation (sequential) | 10–12 ms | 10–12 ms |

Against roughly 790 ms total for 2M multithreaded uniform, the index scatter is 5.3–6.1% of the
build, and the accepted adaptive policy recovered about 1.1% of it. The 4x gap between the index
scatter and vertex concatenation is the random side: `cell_indices` at 2M is about
`6 x 2M x 4B = 48 MB`, whose sequential-write cost is single-digit milliseconds. Most of the
remaining ~35 ms is random-access penalty, not bandwidth.

Two structural facts make this worse than the single-thread numbers suggest:

- 48 MB of destination exceeds the reference Ryzen 3600's 16 MiB L3 outright at 2M, and the working
  set is shared by all workers, so the effective per-thread residency is far below that even at 1M.
- The shard-order branch pays *two* irregular streams, not one: the random write into
  `cell_indices`, plus a random read of `cells_ref[gen_idx]` (about 16 MB at 2M) to obtain
  `dst_start`. Only the payload write appears in the current framing.

## Design

`scatter_local_indices` (`live_dedup/assemble.rs:101`) copies a short run — about six `u32`, 24
bytes — per cell. Replace the branch pair with a two-pass partition. `dst_start` is monotone in
`gen_idx`, so a destination bucket *is* a contiguous generator range; no sorting is required, only
a shift.

```text
pass 1  (parallel over shards; both reads sequential)
    for each local rank in bin_generators[bin]:
        read shard.output.cell_starts / cell_counts / cell_indices   sequential
        append {gen_idx, count, payload} to bucket (gen_idx >> shift)

pass 2  (parallel over buckets; source sequential, destination window-resident)
    for each record in bucket:
        dst_start = cells[gen_idx].vertex_start()      // cells window in cache
        write payload into cell_indices[dst_start..]   // 1/K of the output
```

Properties:

- No irregular access on either side. Pass 1 keeps `K` write heads live (K ≈ 64 gives about 4 KB of
  cache lines per worker); pass 2 confines both the `cells` metadata reads and the `cell_indices`
  writes to one `1/K` window.
- The random `cells_ref[gen_idx]` read disappears entirely — it becomes a windowed read in pass 2.
- Public cell order, index values, contiguity, and the disjoint-span invariant are unchanged. This
  is a traversal reorder, not a format or contract change.
- Determinism is unaffected: each cell's destination span is fixed by the prefix sum, and buckets
  are disjoint, so worker interleaving cannot change any written value.

Cost is roughly 28 bytes per cell of staged traffic, written and re-read, all streaming. That is
the thing the experiment must weigh against the removed random access.

If the primary gate passes, the same primitive applies to the `point_indices` scatter in
`cube_grid/build.rs:165` and, less obviously, to vertex concatenation. Build it as one shared
helper rather than three site-specific rewrites, but measure the assembly site alone first.

## Relationship to prior work

- **Parent, accepted:** adaptive final index scatter and adaptive grid coordinate materialization.
  This proposal does not replace their spatial-order classifier; it changes what the classifier
  chooses *between*. If it wins, the classifier's shard/generator branch collapses into a single
  path and the sampler is retained only for the already-cell-major fast case (`cubed`, Fibonacci),
  where the fused one-pass form should still win.
- **Adjacent, retired:** "Fusing the inverse point-to-slot map with `SlotPoint` AoS construction"
  was rejected because it merged random stores into a multi-stream loop. This proposal moves in the
  opposite direction — it separates irregular work into its own bounded pass — so the retirement
  does not cover it.
- **Adjacent, retired:** reusing the backend's `Vec<VoronoiCell>` allocation reduced retired work by
  0.19–0.22% but raised cache references 9.3–11.9% and cycles 2.8–4.2%. It is the standing warning
  that output-materialization changes are judged on cache behavior, not instruction counts.
- **Adjacent, closed neutral:** the adaptive cell-count/prefix candidate. Its lesson — do not add
  another A/B materialization path with no retired-work or outcome benefit — applies directly.
  Bucketing must be a single unconditional path if it wins, not a fourth mode.
- **Interacting, untried:** the TLB / huge-page probe. Bucketing reduces TLB reach pressure, so if
  `MADV_HUGEPAGE` lands first this win shrinks. Measure them in that order and do not bundle them.

## Gate

The old `dedup_indices_ms` hot-path clock was removed because fine-grained timing distorted the
workload. Use a production-shaped `perf` profile for attribution, and
`scatter_by_shard`, `shard_order_pairs`, `shard_order_descents`, and `shard_order_abs_delta` already
report which branch fired and why.

1. Confirm the attributed phase cost on the reference host at 2M and 1M, uniform and Fibonacci,
   before writing any code. The Mac numbers above have never been reproduced on Linux.
2. Prototype pass 1 / pass 2 behind a build-mode switch; keep the existing branches available for
   interleaved pairing.
3. A/B on the established matrix: 2M and 1M, uniform (scrambled, the target), Fibonacci
   (correlated control), clustered, and `cubed` (cell-major guard — the fused path must not
   regress). Prefer paired `perf` cycles and cache counters over wall time, per the host calibration
   section.

Falsification: if a production-shaped profile still attributes roughly 30 ms or more to the
scatter/prefix work at 2M uniform, the staged traffic is not repaying the removed random access and the whole family is dead —
retire it with the same finality as the vertex-concatenation-drop experiment. A win confined to
uniform with a `cubed` or Fibonacci regression is also a rejection: a distribution-sensitive
heuristic here would repeat the rejected shell-scan specialization.

## Experiment result

The gate ran on the reference Ryzen 5 3600 Linux environment (12 logical CPUs, 16 MiB L3). The
prototype staged each cell as one `u32` generator id followed by its already-globalized payload,
using per-shard bucket vectors and a parallel destination-window fill. The existing adaptive path
remained available in the same binary through a private experiment switch.

Before implementation, three ordinary all-core timing samples reproduced the target phase cost:

| Input | 1M `dedup_indices_ms` | 2M `dedup_indices_ms` |
|---|---:|---:|
| Uniform | 18.60–20.37 ms | 38.59–43.70 ms |
| Fibonacci | 6.73–11.09 ms | 21.93–30.96 ms |

A five-round 2M uniform sweep with preprocessing disabled found that window count mattered:

| Path | Mean | Range |
|---|---:|---:|
| Existing adaptive scatter | 36.11 ms | 29.96–45.04 ms |
| 32 windows | 36.56 ms | 35.56–38.62 ms |
| 64 windows | 32.32 ms | 21.63–49.90 ms |
| 128 windows | 27.66 ms | 21.89–34.68 ms |
| 256 windows | 29.89 ms | 27.38–33.48 ms |

The 128-window form therefore passed the isolated `~30 ms` gate and advanced to seven paired,
interleaved all-core `perf stat` rounds at 1M and 2M. The comparison used the same release binary,
changed only the private runtime mode, disabled preprocessing, and reported no context switches or
CPU migrations. Percentages below are candidate relative to the existing adaptive scatter:

| Size / input | Cycles | Instructions | Branches | Cache misses |
|---|---:|---:|---:|---:|
| 1M uniform | -1.60% | +0.49% | +0.77% | -4.05% |
| 2M uniform | -2.19% | +0.46% | +0.71% | -0.63% |
| 1M Fibonacci | -1.99% | +0.45% | +0.81% | -8.90% |
| 2M Fibonacci | +1.88% | +0.40% | +0.72% | +10.87% |
| 1M clustered | +0.98% | +0.22% | +0.35% | +4.07% |
| 2M clustered | -1.74% | +0.26% | +0.42% | -8.01% |
| 1M `cubed` | +0.58% | +0.46% | +0.79% | +1.17% |
| 2M `cubed` | +0.75% | +0.47% | +0.81% | +1.96% |

The primary uniform target improved, but the 2M Fibonacci regression exceeded the host's
approximately 1.2% paired-cycle noise interval, both `cubed` scales moved adversely, and the
candidate added roughly 0.4–0.5% instructions plus 0.7–0.8% branches across the normal controls.
This is the explicit correlated-control falsifier above. Do not add a fourth distribution-sensitive
scatter mode, and do not transfer this primitive to grid point-index scatter or vertex
concatenation without a materially different staging design. The prototype was reverted.

## Secondary, lower confidence

The demand-prefix adversarial review asked for "a cheap deterministic predictor" of when a cell
should request fewer exact slots, and none was built; the rejected experiment tested only a fixed
global 8-versus-16. A free, deterministic, spatially correlated predictor exists: the consumed depth
of the *previous cell in the same group*. Cells in a group are processed sequentially in a fixed
order (`knn_clipping/driver.rs`, `emit_generator_group`), so reading the prior cell's consumed count
adds one integer of state and cannot introduce nondeterminism.

This is recorded for completeness, not recommended. The fixed-8 probe put the whole ceiling at about
0.315% on Fibonacci with a large uniform regression, so even a perfect policy is worth well under a
percent, and the adversarial review's other objection — that a smaller ask can add a second
full-remainder partition pass — is untouched by a better predictor. Do not spend a pass on it ahead
of the assembly boundary.
