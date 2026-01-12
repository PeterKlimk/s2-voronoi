# Packed kNN / live_dedup / termination refactor plan

## Scope
Performance-first refactor of the knn_clipping backend with focus on:
- Simplifying live_dedup control flow
- Using packed kNN for resumable, ring-expanding neighbor batches
- Reducing can_terminate cost and ensuring correct bounds
- Eliminating per-query fallback when packed kNN already computed candidates

## Issues observed
1. live_dedup is overly complex with repeated loops and many can_terminate/is_bounded checks.
2. can_terminate is expensive (per-vertex scan + trig), called frequently.
3. Packed kNN already computes dot products; live_dedup recomputes them.
4. Packed kNN currently returns only top-K; when a cell needs more neighbors it falls back to per-query kNN.
5. Packed kNN should return an explicit bound (kth dot or security bound).
6. Termination should use the peeked neighbor bound (next unseen), not the current one.

## Goals
- Make packed kNN the main neighbor provider, capable of expanding rings indefinitely.
- Provide dot reuse + explicit bounds so termination is cheap and correct.
- Centralize neighbor consumption in a single batch loop.
- Enable deferred ring expansion for subsets of queries while retaining SIMD benefits.

## High-level approach
- Introduce a ring-expanding packed stream that scans ring0 (3x3), emits neighbors in batches
  (k_step = 24), then expands to ring1 (5x5 - 3x3), ring2, ... as needed.
- For each query, keep a candidate slab for ring0 and append ringN candidates only when
  needed (for the deferred subset). No full per-ring storage for all queries.
- Use peeked neighbor dot as the max_unseen bound inside the batch loop; when the batch
  is exhausted, use an explicit bound derived from the next ring.
- Make can_terminate O(1) by caching max_r2 in the polygon builder.

## Proposed APIs and structs (sketches)

### Packed ring stream (cube_grid/packed_knn.rs)

```rust
pub struct PackedRingStream<'a> {
    grid: &'a CubeMapGrid,
    points: &'a [Vec3],
    cell: usize,
    queries: &'a [u32],
    k_step: usize,
    scratch: &'a mut PackedKnnCellScratch,
}

pub enum PackedStreamEvent<'a> {
    Batch(PackedBatchView<'a>),
    Done(u32),
    NeedsNextRing(u32),
}

pub struct PackedBatchView<'a> {
    pub query_idx: u32,
    pub neighbors: &'a [u32],
    pub dots: &'a [f32],
    pub bound: f32,      // max unseen dot
    pub exhausted: bool, // exhausted current candidate set
}
```

### Packed scratch additions

```rust
pub struct PackedKnnCellScratch {
    // candidate slab per query
    keys_slab: Vec<MaybeUninit<u64>>,
    lens: Vec<usize>,

    // per-query cursor
    requested_k: Vec<usize>,
    query_state: Vec<QueryState>,

    // ring expansion state
    ring_frontier: Vec<u32>,
    ring_next: Vec<u32>,
    ring_visited: Vec<u32>, // stamp/bitmap
}

enum QueryState { Active, Done, NeedsRing }
```

### live_dedup consumption (knn_clipping/live_dedup.rs)

```rust
struct CellState {
    builder: Topo2DBuilder,
    global: usize,
    local: u32,
    neighbors_processed: usize,
    terminated: bool,
}

struct DeferredCell {
    state: CellState,
    cell_start: u32,
}

fn consume_batch(
    state: &mut CellState,
    batch: PackedBatchView<'_>,
    termination: TerminationConfig,
) -> PackedBatchDecision;
```

## Ring expansion helper (sketch)

```rust
struct RingExpander {
    visited_stamp: Vec<u32>,
    stamp: u32,
    frontier: Vec<u32>,
    next: Vec<u32>,
}

impl RingExpander {
    fn begin(&mut self, grid: &CubeMapGrid, center_cell: usize) -> &[u32];
    fn advance(&mut self, grid: &CubeMapGrid) -> &[u32];
}
```

- ring0 = center + 8 neighbors (3x3)
- ring1 = 5x5 - 3x3
- ring2 = 7x7 - 5x5
- ... until all cells visited

## Bound computation for termination
For each batch:
- Use peeked dot if available: dots[i+1]
- Otherwise use batch.bound

batch.bound = max(kth_dot, security_next_ring)

security_next_ring is computed from the next ring's cell caps:

```rust
fn security_bound_for_ring(qx, qy, qz, ring_next, grid) -> f32
```

This generalizes the existing ring2 security bound to arbitrary ring depth.

## Termination optimization (Topo2DBuilder)
Cache max_r2 (max u^2 + v^2) in the polygon and compute min_cos as:

```
min_cos = 1 / sqrt(1 + max_r2)
```

This makes can_terminate O(1) instead of scanning vertices each time.

## Phased action plan

### Phase 1: Packed ring stream and chunked selection
- Implement PackedRingStream with ring expansion and batch emission (k_step = 24).
- Dense 3x3 path: keep candidate slab and allow repeated select_nth_unstable for batches.
- RingN pass: run SIMD only for deferred queries, append candidates, and resume batches.
- Expose dots (or decode from keys) and explicit bounds in PackedBatchView.

### Phase 2: live_dedup refactor to consume batches
- Replace repeated kNN resume/restart loops with consume_batch.
- Pause cells that need more neighbors; resume automatically on next ring pass.
- Remove per-query fallback for packed cells.

### Phase 3: can_terminate optimization
- Track max_r2 in polygon updates.
- Update can_terminate to use cached min_cos and peeked bound.

### Phase 4: cleanup
- Remove redundant is_bounded/can_terminate checks and simplify live_dedup.
- Keep a small slow-path for pathological packed_knn slow cells.

## Notes
- Memory impact: small; candidate slab for ring0 plus ringN append for deferred subset.
- SIMD is preserved for all ring scans; per-query fallback becomes unnecessary.
- This plan assumes packed kNN can expand rings indefinitely and replaces the single-query API.

