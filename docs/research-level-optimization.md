# Research-Level Optimization Ideas

Status: open, 2026-06-18. Three algorithmic ideas after exhausting
micro-optimization headroom in the post-clipping pipeline. All attack
the examine-and-reject ratio (8.3 neighbors examined per 6 final edges)
rather than per-operation cost.

## The Prize

From the shadow audit in the timing output (500k uniform):

```
dir_shadow: checks=436431 tests=1547913 hits=144544 saved=1259194
```

30-35% of candidate examinations are provably skippable — the shadow
audit proves it. Every prior attempt to *use* that signal for pruning
has been too costly to net a win. The prize is ~1.26M clips at 500k
(~290ms), but the cost of the check has always exceeded the savings.

The Fade2D comparison (`docs/st-headroom-and-fade-comparison.md:35-55`)
confirms why: incremental Delaunay "touches real edges only" (6 per
cell), while this algorithm examines-and-rejects 8.3-14.3 candidates to
prove completeness. That gap is the entire ST headroom.

Examine-and-reject ratio, fixed seed, ST, no preprocess
(`st-headroom-and-fade-comparison.md:94-101`):

| distribution | neighbors/cell | final edges/cell | examine/edge |
|---|---:|---:|---:|
| fib | 8.60 | 6.00 | 1.434 |
| uniform | 9.89 | 6.00 | 1.648 |
| gradient k=4 | 9.92 | 6.00 | 1.653 |
| splittable | 14.32 | 6.00 | 2.387 |

Known-batch directional shadow, timing-only lower bound
(`st-headroom-and-fade-comparison.md:103-110`):

| distribution | saved / neighbors |
|---|---:|
| fib | 30.6% |
| uniform | 32.5% |
| gradient k=4 | 32.5% |
| splittable | 34.6% |

Support-envelope shadow recovered 75-85% of exact shadow hits with zero
false positives in the timing probe, but the real late-batch behavior
was too costly.

## What Has Been Tried (And Failed)

See `docs/st-headroom-and-fade-comparison.md:67-90` for the full closed
list. Key lessons:

- **Candidate reuse via seeds** (`agent/candidate-set-reuse`, `82e6ff8`):
  Adding seed clips *increases* work. Reuse must remove
  production/select work, not add clips.
- **Conservative reject before clip** (`agent/clip-conservative-reject-prototype`,
  `2d9f5bb`): Tail-radius reject was near-neutral; support reject was
  worse. The support cache rebuild cost (O(64×N) per clip) killed it.
- **Directional certificates** (`agent/directional-certificates`,
  `f51523a`): Real value in dense regimes, neutral on fib. The
  per-candidate check (`candidate_would_be_unchanged`) is too expensive
  — it computes bisector coefficients and evaluates all polygon vertices.
- **Packed prefix reuse** (`agent/packed-select-reuse-audit`, `e895f1e`):
  Previous packed first-prefix overlap is too sparse to skip selection.
- **Frontier prune** (`agent/frontier-prune-prototype`, `5e835da`):
  Scalar shell-layer bound had zero skip hits.

## Idea 1: Incremental Support Cache + Directional Termination

**Branch:** `agent/incremental-support-cert`

**The two-part idea:** The support envelope
(`candidate_would_be_unchanged_support`, `clip.rs:186-209`) recovers
75-85% of the exact shadow signal with zero false positives. It was
rejected because `rebuild_support_cache` (`clip.rs:167-179`) is
O(64×N) per clip. This idea fixes the rebuild cost and moves the check
from per-candidate (expensive) to certificate-level (cheap).

### Part A: Incremental cache maintenance

`rebuild_support_cache` (`clip.rs:167-179`) recomputes all 64 sectors
after every clip:

```rust
fn rebuild_support_cache(&mut self) {
    const K: usize = 64;
    let poly = self.current_poly().clone();
    for sector in 0..K {
        let angle = (sector as f64) * TAU / K as f64;
        let (sin, cos) = angle.sin_cos();
        let mut min_proj = f64::INFINITY;
        for i in 0..poly.len {
            min_proj = min_proj.min(cos * poly.us[i] + sin * poly.vs[i]);
        }
        self.support_min_proj[sector] = min_proj;
    }
}
```

A clip changes at most 2 edges of the polygon (entry and exit
intersections). So at most 2 angular sectors change. Track which sectors
are dirty and only recompute those — O(2) per clip instead of O(64×N).

Concretely: each polygon edge has an angular range `[θ_start, θ_end]`
that it "covers" in the support function. When a clip replaces edges
`E_old1, E_old2` with `E_new1, E_new2`, invalidate only the sectors
overlapping the old edges' angular ranges, then recompute those sectors
from the new edges. With 64 sectors and ~6 edges, each edge covers ~11
sectors on average, so worst case ~22 sectors to recompute — but only
when the clip changes the polygon (Changed), not on Unchanged clips.

### Part B: Directional termination bound

The current `can_terminate` (`projection.rs:220-240`) uses a single
scalar bound: `max_r2` → `min_cos` → `cos_2max`. Every unseen neighbor
is bounded by the *same* threshold regardless of direction. This is why
2.3 extra neighbors survive — they're close enough in dot-product
distance to pass the scalar bound, but their bisector doesn't actually
intersect the polygon because the polygon is already tight in their
direction.

The directional certificate should work at the **certificate level**,
not the per-candidate level:

Instead of one scalar `term_threshold_cache`, compute a *piecewise*
bound: for each angular sector (the same 64 sectors from the support
cache), compute the local termination threshold from the polygon's
support in that sector. An unseen neighbor at angle θ is bounded by
`threshold(θ)`, not `max(threshold(θ) for all θ)`.

The certificate then becomes: for each unseen neighbor, check
`dot(neighbor) < threshold(angle(neighbor))`. This is O(1) per neighbor
(one sector lookup + one comparison), with no per-candidate bisector
computation.

The threshold computation: for sector `s`, the polygon's support is
`min_proj[s]` (already computed by the support cache). The termination
threshold for that sector is derived from `min_proj[s]` the same way
`term_threshold_cache` is derived from `min_cos` — via the double-angle
cosine bound. This is O(64) per clip (or O(2) with incremental cache
from Part A).

**Why this could work where prior attempts didn't:**
- Prior `agent/directional-certificates`: per-candidate bisector
  evaluation → O(N) per candidate → too costly.
- This: per-candidate dot+compare → O(1) per candidate → cheap.
- Prior `agent/clip-conservative-reject-prototype`: support cache
  rebuild O(64×N) per clip → killed the win.
- This: incremental cache O(2) per clip → makes the support envelope
  cheap enough to maintain continuously.

**Interaction:** Parts A and B share the same 64-sector table.
Implementing both together gives: cheap per-clip cache maintenance
(A) + cheap per-candidate directional bound (B). The combined effect:
the 30% skip rate becomes a 30% reduction in clips, at the cost of
O(2) per-clip cache updates and O(1) per-candidate lookups.

**Where it fires:** All regimes. Unlike the parked directional
certificate (which was neutral on fib), this attacks the 30% skip rate
that exists even in fib. The shadow audit proves the signal is there;
the cost was the implementation barrier.

**Where it doesn't fire:** The directional bound is conservative — it
won't skip a candidate whose bisector actually intersects the polygon.
The 75-85% recovery rate of the support envelope means ~20-25% of the
30% skippable clips still get examined. So the realistic clip reduction
is ~22-25%, not 30%.

**Estimated saving:** ~22-25% of 956ms clipping = ~210-240ms at 500k
uniform. This is the largest single prize in the codebase.

**Risk:** The incremental cache must be correct — a stale sector entry
could skip a needed clip, producing an invalid cell. The correctness
guard is the existing validation suite + the `dir_shadow` audit (which
should show reduced `tests` and `hits` proportional to the skip rate).

**Measurement plan:**
- `perf stat` counters: reduced `instructions` (fewer clip dispatches),
  reduced `cycles`. `L1-dcache-load-misses` may rise slightly (64-sector
  table = 512 bytes, fits in L1 but adds pressure).
- `S2_VORONOI_TIMING_KV=1`: clipping time should drop; the new
  per-clip cache update and per-candidate lookup should appear as a
  small new slice (instrument separately).
- `dir_shadow` audit: `saved` should increase, `hits` should stay
  proportional (the audit measures exact skips; the certificate should
  recover most of them).
- Correctness: `cargo test --release --test correctness --test validation
  --test adversarial` must stay green. The `dir_shadow` audit can also
  be used as a differential check — run it with the certificate on and
  off and verify the skip set is a subset of the audit's skip set.

### Implementation outline

1. Add `support_min_proj: [f64; 64]` and `support_cache_valid: bool` to
   `GnomonicBuilder` (already exist under `#[cfg(feature = "timing")]` —
   promote to always-on).
2. In `commit_clip` (`clip.rs:48`), after the polygon is updated,
   incrementally update the dirty sectors (the ones whose angular range
   overlaps the removed/added edges).
3. Add `directional_threshold: [f64; 64]` computed from
   `support_min_proj` (double-angle cosine bound per sector).
4. In `can_terminate` (`projection.rs:220-240`), replace the scalar
   `term_threshold_cache` check with: for the batch's `unseen_bound`,
   check against `max(directional_threshold)` (conservative) — or, for
   per-candidate checks, `directional_threshold[sector(neighbor)]`.
5. In `clip_batch` (`run.rs:427-535`), the mid-batch termination check
   (`can_terminate(bound)`) uses the directional bound. The per-candidate
   early-out would go in the clip loop, before `clip_with_slot_result`:
   if `dot(neighbor) < directional_threshold[sector(neighbor)]`, skip
   the clip (mark as Unchanged without dispatching).

## Idea 2: Ring-Pass Dot Reuse

**Branch:** `agent/ring-pass-dot-reuse`

**What was tried:** `agent/candidate-set-reuse` (`82e6ff8`) reused
*constraints* (seed clips) — failed because it added work. This reuses
*computation* (dot products) — removes work.

**The idea:** Adjacent cells share most of their neighbor sets. The
packed kNN already processes a *group* of queries (all generators in one
grid cell) together with SIMD-8 dots. But the ring_pass dot products are
lost when the group finishes — the next group (adjacent grid cell) must
recompute dots against the same ring cells.

The ring cells overlap heavily between adjacent grid cells in the same
bin. If group N's ring_pass already computed `dot(neighbor_slot,
query_slot)` for slots that group N+1 also needs, skip the
recomputation.

**Concrete design:**
- A per-bin dot-product cache keyed by (neighbor_slot, query_slot)
- Populated during ring_pass, consumed by the next group's ring_pass
- Direct-indexed by neighbor_slot (one `f32` per slot, stamp-cleared)
  for O(1) lookup — works for the 8 ring cells' worth of slots
  (~hundreds), not the full population
- On cache hit: skip the SIMD dot AND the candidate collection
  (`make_desc_key` + push to `chunk0_keys`)

**What it saves:** The dot products are cheap (~4ns each), but the
candidate collection (`make_desc_key` + push) adds overhead. Skipping
both for known-neighbors saves ~10-15ns per hit. With ~30% hit rate
(rough estimate from grid-cell adjacency), that's ~1.5ms at 500k —
small.

**Why it's lower priority:** The packed_knn slice (193ms) is already
the smaller of the two big slices. The hit rate is uncertain and the
cache lookup overhead (`stamp check + array read`) might eat the
savings. The bigger value is skipping the *candidate collection*, not
the dots.

**Estimated saving:** ~5-15ms at 500k (uncertain). Small but attacks a
different slice from Idea 1.

**Risk:** The cache must be nearly free to net a win. A stamp-cleared
direct-indexed array (like `AttemptedNeighbors`, `run.rs:29-67`) is the
cheapest possible structure. The main risk is that the hit rate is too
low to amortize the cache maintenance cost.

## Idea 3: Multi-Plane Batched Clipping (Loop Fusion)

**Branch:** `agent/multi-plane-batch-clip`

**What was tried:** Nothing directly. The closest is "angular-sweep
clipper" (demoted, not tried).

**The idea:** Currently each clip evaluates N vertices against 1
half-plane via SIMD-8. What if we evaluated N vertices against K
half-planes simultaneously? Load 4 half-planes, evaluate each against
the 4-8 vertices, and build the combined output polygon in one pass.

The advantage: fewer passes over the polygon data, better register
utilization. The challenge: clip order matters for floating-point
results (A then B ≠ B then A at the margin), so the batch must preserve
kNN order. But within a sorted batch, the order is already known —
process the batch's half-planes in order, but share the vertex loads.

This is closer to "loop fusion" than true multi-plane clipping: instead
of `for hp in batch { for v in poly { eval(hp, v) } }`, do `for v in poly
{ for hp in batch { eval(hp, v) } }`. The vertex data stays in registers
across all half-planes in the batch.

**Why it's speculative:** The per-clip cost is already low (230ns with
SIMD), so the savings are bounded. The main benefit would be reduced
polygon data traffic — each clip currently reloads the polygon's `us`,
`vs` arrays from L1; fusing K clips would load them once. For N=6 and
K=4, that's 4× less polygon data traffic. But the polygon is only 48
bytes (6 × f64 × 2), which fits in 1 cache line — already in L1.

**Estimated saving:** ~20-40ms at 500k (speculative). The savings are
in instruction count (fewer loop iterations, fewer poly loads), not
cache misses (already L1-resident).

**Risk:** The fused loop is significantly more complex — the output
polygon changes shape after each half-plane, so the K half-planes can't
all be evaluated against the same vertex set. Only the *distance
evaluation* can be fused; the clip logic (which vertices survive, where
intersections are) must still be sequential. This limits the benefit to
the distance-evaluation portion of the clip (~40% of per-clip cost).

**Why it's lowest priority:** The per-clip cost is already
well-optimized, and the biggest prize is in *fewer clips* (Idea 1), not
*cheaper clips*. Only worth trying if the certificate work doesn't
deliver.

## Suggested Branch Order

1. **`agent/incremental-support-cert`** (Idea 1) — highest potential
   (~210-240ms), attacks the 30% skip rate with a concrete fix for the
   cost problem that sank prior attempts. The incremental cache (Part A)
   should be implemented and measured first in isolation, then the
   directional termination bound (Part B) layered on top.
2. **`agent/ring-pass-dot-reuse`** (Idea 2) — lower priority, smaller
   win, but attacks a different slice (packed_knn, not clipping).
3. **`agent/multi-plane-batch-clip`** (Idea 3) — speculative, only worth
   trying if the certificate work doesn't deliver.

## Measurement Protocol

Same as prior branches — hardware counters as primary evidence:

```bash
RAYON_NUM_THREADS=1 perf stat -e cycles,instructions,\
cache-references,cache-misses,\
L1-dcache-load-misses,L1-dcache-loads,\
branch-misses,branches \
  cargo run --release --features tools --bin bench_voronoi -- 500k --no-preprocess

S2_VORONOI_TIMING_KV=1 RAYON_NUM_THREADS=1 \
  cargo run --release --features tools,timing --bin bench_voronoi -- 500k --no-preprocess

./scripts/bench_build.sh --chain 6
./scripts/bench_run.sh -s 500k -r 20 -m total
```

Correctness guards (every branch):
```bash
cargo test --release --test api --test correctness --test validation
cargo test --release --test adversarial
```

For Idea 1 specifically, the `dir_shadow` audit provides an additional
differential check: run with `--features timing` and verify that the
certificate's skip set is a subset of the audit's skip set. The audit's
`saved` counter should increase; `hits` (exact skips) should stay
proportional to the audit's measurement.

Test across distributions (fib, uniform, splittable, mega) — Idea 1
should help all regimes, not just dense. Prior directional certificate
work was neutral on fib; this should be different because the check is
at certificate level (cheap) not per-candidate (expensive).
