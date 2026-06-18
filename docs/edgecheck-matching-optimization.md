# Edge-Check Matching Optimization

Status: **parked**, 2026-06-18. All candidates are a flop for the sane regime.

## Verdict

With mean incoming edges ≈3 (edges only come from preceding same-bin
neighbors, roughly half the ~6 planar-bounded cell edges), the matching scan
is ~3 register-resident u64 compares per edge. That is already near-optimal
for the sane regime, and no candidate beats it there:

- **(1) u32 SOA scan:** halves the compare width but 3 u32 compares vs 3 u64
  compares is invisible. Not worth the diff.
- **(2) SIMD batched equality:** needs k > ~8 to fill lanes. Mean k≈3 never
  fills them. Two code paths for a case that doesn't fire.
- **(3) Direct-indexed table:** L2-latency lookup (~10 cycles) is *slower*
  than 3 register compares (~3 cycles). Wins only at k ≫ 6, which is rare.
  The geometry works against it: large k correlates with dense bins (large
  `local_population`, cold table).
- **Cross-cutting refinement (store `neighbor_local`):** only an enabler for
  (3), which is weak. Not worth doing standalone.

Measured timing confirms the slice is small:

```
edge_collect:   27.7ms ( 4.8%)
edge_resolve:   27.7ms ( 4.8%)
edge_emit:      42.1ms ( 7.3%)
```

`edge_resolve` (which contains the matching) is 4.8% of cell construction.
Even halving it would save ~14ms out of 580ms. Not worth the complexity or
the correctness risk on a path that propagates vertex indices.

## Original Analysis (Retained For Reference)

The analysis below is retained in case a future workload changes the
regime (e.g., a use case with consistently high k from dense same-bin
clustering). Do not retry without such a workload.

### Context

After each cell is built, `collect_and_resolve_cell_edges`
(`src/live_dedup/edge_checks.rs:111-297`) walks the cell's edges once and, for
each edge to an **earlier same-bin** neighbor, searches the incoming
`Vec<EdgeCheck>` (forwarded by that earlier neighbor) for a matching
`check.key == edge_key`. Matched checks patch vertex indices inline; unmatched
checks become `InBinUnconsumedCheck` records for `edge_reconcile`.

This same-bin matching is the only place in the hot loop with real algorithmic
headroom. Cross-bin overflow matching (`resolve_edge_check_overflow`,
`edge_checks.rs:312-404`) is a serial assembly-point sort+pair pass that is
small in practice and out of scope here.

## Constraints Noted In Discussion

1. **Bisector recompute in `clip_with_slot_edgecheck` stays.** Recomputing
   `(a,b,c)` from the neighbor position (rather than reusing the earlier
   cell's half-plane) improves numerical accuracy by avoiding drift through
   the seed-forwarding channel. Not a target.
2. **Linear scan beats a hashmap here.** `incoming_count` is debug-asserted
   ≤ 64 and is typically small (k≈3 uniform); hashing overhead dominates.
   Alternatives must be in the linear / direct-indexed family.
3. **Cross-bin work is very small.** Not worth optimizing.

## Headroom Observation

Every key on both sides of the match already contains `cell_idx` as one half:

- Incoming checks are stored *at* this cell's `LocalId` slot
  (`ShardDedup::edge_checks[local]`, `edge_checks.rs:71-99`), so every
  `check.key` is `(cell_idx, neighbor)` for some neighbor.
- Cell edges are *this* cell's edges, so every `edge_key` is
  `(cell_idx, neighbor)` (`edge_checks.rs:193`).

The distinguisher is the *other* u32 — the neighbor id — which the consume
side already holds as `edge_neighbor_globals[i]` (`edge_checks.rs:187`). The
current `check.key == edge_key` is a u64 compare where half the bits are
tautological. Reducing the match to a u32 compare halves bandwidth and
doubles SIMD lane width for free.

## Candidates

Listed cheapest first. All four keep the linear / direct-indexed family
per constraint (2).

### 1. u32-keyed SOA scan

Extract `incoming_neighbor_ids: &[u32]` once per cell (one O(k) pass over
`incoming_checks`), then linear scan comparing `neighbor` (already in a
register) against u32s instead of u64 keys.

- Same algorithm, ~2× less data touched per compare, needle is free.
- Trivial diff, low risk.
- Likely a wash for typical k≈3 but a clean basis for (2).
- Back-of-envelope: ~40ms at 1M uniform — may not be a measurable slice at
  all. Worth doing first as a *probe* to confirm whether matching is on the
  critical path before investing in (2)/(3).

### 2. SIMD batched equality on the u32 array

With (1) in place, load `incoming_neighbor_ids` as `u32x8`/`u32x16`, broadcast
`neighbor`, compare, horizontal-any. Branch-free, 8–16× fewer compare insns.

- Uses the crate's existing `portable_simd` path (`#![feature(portable_simd)]`).
- Wins clearly once k > ~8; for k ≤ 4 the scalar scan likely still wins.
- → threshold hybrid (candidate 4).

### 3. Direct-indexed table by neighbor local id, stamp-cleared

Replace the search with O(1). A per-shard array of size `local_population`
(the number of generators in this bin) keyed by the neighbor's *local id* —
which both sides already compute via `layout.bin_local(slot)`
(`edge_checks.rs:196`):

- Push writes `table[local_b]`.
- Take reads `table[edge_neighbor_local]`.
- Stamp-based clearing reuses the crate's `AttemptedNeighbors` pattern
  (`src/knn_clipping/cell_build/run.rs:29-67`).

Properties:

- Eliminates the scan entirely.
- Handles the duplicate-side case naturally (both edges hit the same slot).
- Cost: `local_population` entries per shard. With `num_bins` clamped to
  `[6, 96]` (`binning.rs:51`) and `local` being the sequential in-bin index
  (`binning.rs:173`), `local_population ≈ n / num_bins`: ~10k for 1M/96
  bins, ~31k for 1M/32 bins, ~167k for 1M/6 bins. A stamp+index table at
  those sizes is ~40KB–670KB — **L2-sized, not L1**, and competes with other
  L2-resident data (the `edge_checks` vecs, the cell output buffer). The
  cache-miss trade must be measured, not assumed.
- No overflow fallback needed: `local_population` is the actual population,
  and `validate_local_capacity` (`binning.rs:71-88`) already guarantees
  `local_population <= local_mask + 1` at assignment time.
- Biggest algorithmic win, biggest diff, and the cache analysis is the
  critical unknown.

### 4. Threshold hybrid of (1) + (2)

Scalar for k ≤ 4, SIMD above. Tuned by the k-distribution from the existing
`incoming_seed_neighbors` / `edgecheck_seed_clips` counters
(`src/knn_clipping/cell_build/run.rs:154-155, 711-712`).

## Cross-Cutting Refinement: Narrowed Keys Within Bins

There are three levels of key narrowing for identifying a point or edge
within a bin:

1. **Global index (u32):** distinguishes any point in the whole input.
   This is what the current u64 edge keys carry.
2. **Bin local id (u32, u16-valued in practice):** distinguishes any point
   within a bin. The *type* is u32 (masked to `local_mask` bits,
   `binning.rs:137-138`), but the *values* are dense sequential indices
   0..local_population-1 (`binning.rs:173`). With `num_bins` clamped to
   `[6, 96]`, `local_population ≈ n/num_bins`: ~10k for 1M/96 bins (fits
   u16), ~167k for 1M/6 bins (needs u18, overflows u16). Both sides already
   compute this via `layout.bin_local(slot)` (`edge_checks.rs:196`).
3. **Per-cell neighborhood id (ceil(log2(k)) bits, k≈3–20):** distinguishes
   only the neighbors a single cell actually sees.

A Voronoi polygon in a sane regime is made of spatially close neighbors, not
points scattered across the whole bin. So the *effective* key space for a
single cell's matching is much narrower than level 1, and the values at
level 2 are narrower than the u32 type suggests. Candidate (3) already
operates at level 2 (local-id table); candidates (1) and (2) as written
still operate at level 1 (u32 global ids).

**The practical refinement:** store `neighbor_local` directly in `EdgeCheck`
(`src/live_dedup/types.rs:172-184`). The push side (earlier cell) already
computes `local_b` at `edge_checks.rs:196` and discards it; the take side
(later cell) currently would need a `point_index_to_slot` +
`layout.bin_local` lookup to recover it. Storing it inline eliminates that
lookup and lets the SOA scan (1) and SIMD scan (2) operate on the local id
instead of the global id.

Struct-layout fit: `EdgeCheck` is 28 bytes padded to 32
(`key:u64` + `hp_eps:f32` + `thirds:[u32;2]` + `indices:[u32;2]`). There are
4 bytes of tail padding. A `u32` field for `neighbor_local` fits without
enlarging the struct — zero footprint cost per check. Storing as `u16` would
also fit (2 bytes) but is not safe for the degenerate case (1M/6 bins →
167k > 65535); a `u32` field is always safe and still benefits a separate
SOA scan array where the array can be u16-typed when `local_population <=
65535` and u32-typed otherwise.

**Where the narrowing pays off:**

- Cache footprint: 2× less data per element in a u16 SOA scan array (u16 vs
  u32). Matters only when the scan array is large enough that footprint
  affects other accesses — i.e., clustered input with a heavy k tail.
- SIMD lane width: `u16x16` vs `u32x8` = 2× more lanes for candidate (2).
  Matters only when k is large enough to fill those lanes.
- Bandwidth: 2× less data to load from the scan array.
- The `u32`-in-struct variant pays none of these (the struct doesn't
  shrink), but still eliminates the `point_index_to_slot` + `bin_local`
  lookup on the take side. The SOA-array narrowing is where the
  footprint/bandwidth/SIMD wins live, and it requires extracting the local
  ids into a separate compact array.

**Where it does not pay off (the catches):**

- Scalar u16 compares are not faster than u32 on modern CPUs — the ALU
  widens them. The win is purely footprint/bandwidth/SIMD-width, not scalar
  instruction count. For k≈3 (uniform input) the scan array is already in
  registers and one cache line; narrowing changes nothing measurable.
- Level 3 (per-cell neighborhood id below the bin's local-id range) is the
  seductive version: a cell with k neighbors needs only ~4–5 bits. But the
  push and take sides are different cells with different neighborhoods, so
  the remap must be built on the take side alone — O(k) setup to save an
  O(k) scan, no asymptotic win. The constant-factor trade (SIMD width vs
  setup cost) only favors it when k is large and the scan is the bottleneck.
  Not recommended unless (2) proves insufficient at large k.
- Adversarial/clustered inputs *do* scatter neighbors across the full
  `local_population` range. The key space must still cover
  `[0, local_population)`; narrowing below that requires the level-3
  per-cell remap with its setup cost.
- The u16 SOA array is only safe when `local_population <= 65535`; a
  data-dependent u16/u32 split adds a branch and two code paths. The u32
  SOA array is always safe but only halves bandwidth if the hardware
  compresses u32 loads (it doesn't — the win is from the separate compact
  array being smaller, not from per-element width).

**Relationship to candidates:**

- Refines (1): u32 global-id SOA scan → u16/u32 local-id SOA scan. The
  local-id variant also eliminates the `bin_local` lookup on take.
- Refines (2): `u32x8` SIMD → `u16x16` SIMD (2× wider lanes) when
  `local_population` fits u16.
- Already implicit in (3): the local-id table is level-2 by construction.
  Storing `neighbor_local` in `EdgeCheck` just makes the push→take handoff
  explicit instead of requiring a layout lookup on take.

This refinement can be applied to candidate (1) at low risk (the u32-in-
struct variant is always safe; the u16 SOA array needs a population check),
and makes (2) more effective if we get there.

## Suggested First Branch

**(1) alone.** It is near-zero-risk and doubles as a probe: if `perf stat`
shows `collect_and_resolve_cell_edges` is not a measurable slice at 500k/1M
under both uniform and clustered inputs, we park the whole thread. If it is
measurable *and* the k-distribution has a heavy tail, (2) or (3) become worth
the complexity.

Branch name (matches existing `agent/<short-topic>` convention):
`agent/edgecheck-u32-soa-scan`.

## Measurement Plan

Per branch, identical harness so numbers are comparable:

- **Timing:**
  ```bash
  S2_VORONOI_TIMING_KV=1 \
    cargo run --release --features tools,timing --bin bench_voronoi -- 500k --no-preprocess
  ```
  Run on both uniform and clustered inputs.
- **Hardware counters:**
  ```bash
  perf stat -e cycles,instructions,cache-references,cache-misses,\
  L1-dcache-load-misses,LLC-load-misses,branch-misses \
    cargo run --release --features tools --bin bench_voronoi -- 500k --no-preprocess
  ```
- **Inter-branch comparison:**
  ```bash
  ./scripts/bench_build.sh --chain 6
  ./scripts/bench_run.sh -s 500k -r 20 -m total
  ```
- **Single-threaded stable perf:** `RAYON_NUM_THREADS=1` for the timing runs.

## Correctness Guards

Behavior must not move across branches. Assert unchanged across branches:

- `incoming_seed_neighbors` and `edgecheck_seed_clips` counters
  (`run.rs:154-155`).
- The set and origins of `unresolved_edges` records
  (`edge_checks.rs:260-291`): `InBinThirdsMismatch`, `InBinMissingCheck`,
  `InBinUnconsumedCheck` counts must be identical for a fixed input.
- `cargo test --release --test api --test correctness --test validation`
  and the adversarial battery (`cargo test --release --test adversarial`)
  must stay green.

The matching path is correctness-sensitive (it propagates vertex indices
between cells), so any change must preserve the exact set of resolved /
unresolved records, not just the final diagram validity — the repair pass can
mask a changed matching set by fixing different mismatches, which would
silently shift work between hot loop and repair.

## Open Questions

- Is `collect_and_resolve_cell_edges` actually a measurable slice at 500k/1M?
  (1) answers this as a side effect.
- What is the k-distribution of `incoming_seed_neighbors` across uniform vs
  clustered inputs? Determines whether (2) or (3) is the better follow-up.
- Does the direct-indexed table in (3) interact badly with the shard's L2
  footprint? The table is `local_population` entries (~10–167k depending on
  n/num_bins), which is L2-sized and competes with other L2-resident data.
  Needs a cache-miss comparison against (1)/(2), not just instruction counts.
  The clustered regime (large `local_population` in dense bins) is the
  worst case for the table and must be measured specifically.
