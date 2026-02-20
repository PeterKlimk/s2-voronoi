 # Packed kNN Ring-Expansion (r=2 Band) With Zero Hot-Path Cost

  ## Summary

  Add an optional, cold-path packed-kNN ring expansion that runs only when a query exhausts the existing packed r=1 (3x3) candidates and clipping is still unproven, before falling back to per-query resumable kNN. The expansion is a single additional band that covers
  the full radius-2 neighborhood (3x3 + ring2 cells) and emits candidates in a dot range that preserves the packed ordering and unseen_bound correctness contract.

  Default behavior stays identical unless the new runtime knob is enabled, and the expansion is triggered.

  ## Goals / Success Criteria

  - No measurable regression on the existing fast path when expansion is not triggered:
      - No extra scanning, allocations, vector resizes, or per-query bookkeeping in PackedKnnCellScratch::prepare_group_directed.
  - When enabled, reduce time spent in find_k_nearest_resumable_* for clustered/hard regions by satisfying termination/clipping earlier.
  - Preserve correctness invariants:
      - Packed emission remains in descending dot order within each stage.
      - Returned unseen_bound is always a conservative upper bound on dot for any not-yet-emitted candidates under the stage schedule.

  ## Non-Goals

  - General r=3..R iterative expansion in the first implementation.
  - Replacing resumable kNN entirely.
  - Changing topology/clipping logic; only change neighbor sourcing.

  ## Configuration / Runtime Control

  ### Public knob (runtime)

  - Add to VoronoiConfig (default false):
      - packed_knn_expand_r2: bool
  - Wire it into knn_clipping build path by extending internal TerminationConfig (still internal to crate since knn_clipping is pub(crate)), or by passing a separate internal flag down into live_dedup::build_cells_sharded_live_dedup.

  Rationale: You selected “runtime knob”; default off mitigates regression risk.

  ### Internal safety caps (not exposed initially)

  - PACKED_EXPAND_R2_MAX_CANDIDATES_PER_QUERY: usize (e.g. 8192)
  - If exceeded while building the expansion band for a query, abort expansion for that query and immediately fall back to resumable kNN (correctness preserved).

  ## Design Overview

  ### Current packed r=1 behavior (baseline)

  - r=1 scans center cell + 8 neighbors (3x3), emits:
      - Chunk0: candidates with dot > threshold (per-query hi threshold)
      - Tail: candidates with security1 < dot <= threshold (built lazily per-query)
  - security1 is the “outside 3x3” cap bound computed from cell_ring2.

  ### New behavior: one extra packed stage ExpandR2Band

  Add one stage that emits only candidates with:

  - security2 < dot <= security1
    where:
  - security2 is a conservative bound for any point outside the radius-2 neighborhood (computed from a “ring3” boundary cap list).

  This preserves ordering and bounds:

  - We already emitted everything with dot > security1.
  - ExpandR2Band emits the next band down to security2.
  - After this stage is exhausted, unseen_bound becomes security2.

  ## Implementation Details (Decision Complete)

  ### 1) Add a new stage enum

  File: src/cube_grid/packed_knn/mod.rs

  - Extend PackedStage:
      - Chunk0
      - Tail
      - ExpandR2Band (new)

  ### 2) Extend packed scratch with cold-path expansion storage

  File: src/cube_grid/packed_knn/scratch.rs

  Add new fields to PackedKnnCellScratch, all lazily initialized to avoid hot-path work:

  - Group-level cached lists (computed only if any query requests expansion in the group):
      - expand_r2_cells: Option<Vec<PackedCellRange>>
          - Contains all radius-2 cells to scan, classified into PackedCellRangeKind like the existing neighbor list.
          - Must include:
              - Center cell
              - 8 neighbors (3x3)
              - All ring2 cells
          - Deduplicate cells (use a small stamp/bitset local to this build; group-only).
      - ring3_cells: Option<Vec<u32>>
          - Boundary cells at Chebyshev/neighbor-graph distance exactly 3 from the center cell.
          - Computed via BFS over grid.cell_neighbors(cell) for 3 steps, collecting depth-3 cells (deduped).
  - Per-query, built on demand:
      - expand2_keys: Option<Vec<Vec<u64>>>
      - expand2_pos: Option<Vec<usize>>
      - expand2_ready_gen: Option<Vec<u32>> (generation stamps)
      - security2: Option<Vec<f32>> (per-query outside-r2 bound)

  Do not touch these in prepare_group_directed unless expansion is triggered.

  ### 3) Compute security2 lazily (outside radius-2 bound)

  Add a helper method on scratch (cold path):

  - ensure_security2_for(qi, grid, timings) -> Option<f32>
    Behavior:
  - Ensure ring3_cells exists for group_cell.
  - Compute security2[qi] as:
      - max over cell in ring3_cells of max_dot_to_cap_xyz(q, cell_center, cell_radius)
  - Store into security2[qi].
  - If ring3 computation fails (e.g., bad cell index / degenerate grid), return None and skip expansion (fall back to resumable).

  Correctness assumption (to be validated with tests):

  - The maximum possible dot for points outside the radius-2 cell neighborhood is achieved by a point within the ring3 boundary, so a max over ring3 caps is a conservative outside bound (same rationale as existing outside_max_dot_xyz using ring2 to bound outside 3x3).

  ### 4) Build the r=2 expansion band for a single query

  Add method (cold path):

  - ensure_expand_r2_band_directed_for(qi, grid, slot_gen_map, local_shift, local_mask, timings) -> bool
    Steps:

  1. Ensure security2(qi) exists; if not, return false.
  2. Ensure expand_r2_cells exists (radius-2 scan list).
  3. Scan each cell range in expand_r2_cells and collect candidates satisfying:
      - security2 < dot && dot <= security1
      - Directed constraints:
          - Skip SameBinEarlier cell ranges entirely.
          - Apply the same “center cell triangular” filter (skip earlier slots in the center cell; skip self).
          - Always skip slot == query_slot.
  4. Push keys with make_desc_key(dot, slot) into expand2_keys[qi].
  5. If key count exceeds PACKED_EXPAND_R2_MAX_CANDIDATES_PER_QUERY, abort and return false.
  6. If empty, mark as “no expansion available” (so we don’t retry) and return true (meaning stage is exhausted immediately).

  Implementation note:

  - Use the existing SIMD dot approach for scanning ring/cell ranges where convenient, but correctness is more important than vectorization since it’s cold-path. Keep code isolated so it doesn’t contaminate hot-path.

  ### 5) Emit expansion candidates via next_chunk

  Extend PackedKnnCellScratch::next_chunk:

  - Add match arm for PackedStage::ExpandR2Band:
      - Ensure expansion is ready for qi (via expand2_ready_gen).
      - Partition + sort top n keys identical to Tail.
      - unseen_bound:
          - If more keys remain: last_dot
          - Else: security2[qi]

  ### 6) Integrate stage progression in cell building

  File: src/knn_clipping/live_dedup/build/process_cell.rs

  Modify phase_3_packed_knn_seeds loop:

      1. Run Chunk0 as today.
      3. If Tail exhausted (or tail not possible) and still not terminated:
          - If packed_knn_expand_r2 is true:
              - Call ensure_expand_r2_band_directed_for(qi, ...).
              - If it returns true and there are candidates, switch stage = ExpandR2Band, set k_cur = packed_k1.
              - If it returns false (cap exceeded / couldn’t compute security2), do not expand and proceed to resumable kNN.
      4. When ExpandR2Band exhausts successfully, set packed_safe_exhausted = true and store packed_security_final = security2 for this query.

  Also update the later termination bound choice in phase_4_resumable_knn:

  - Today it uses packed_security when packed exhausted.
  - Change to use packed_security_final (security2 if expansion ran and exhausted; else security1 as before).

  ### 7) Timing/Instrumentation (optional but recommended)

  Files: src/cube_grid/packed_knn/timing.rs, src/knn_clipping/timing/* (feature-gated)

  - Add counters/durations:
      - expand_r2_builds count
      - expand_r2_scan duration
      - expand_r2_select duration
      - expand_r2_skipped_cap count
  - Ensure these are behind cfg(feature="timing") to keep baseline overhead minimal.

  ## Testing Plan

  ### Unit tests (correctness + invariants)

  Add #[cfg(test)] tests near packed_knn scratch:

  - Band ordering + unseen_bound monotonicity
      - Construct a small grid (small res) with deterministic points.
      - Run packed stages and assert:
          - All emitted candidates across stages are globally non-increasing by dot (allow ties).
          - unseen_bound never increases as stages progress.
  - security2 conservativeness (randomized)
      - For random queries in a cell, brute force compute:
          - The maximum dot among points whose cells are outside radius-2 neighborhood.
      - Assert max_dot_outside <= security2 + eps.
      - Keep sizes small to make the brute force feasible in tests.

  ### Integration tests / acceptance

  - Run cargo test --release --test correctness --test validation with knob both off and on (with a small dataset).
  - Add at least one adversarial-style case (clustered points) where knob-on reduces resumable kNN calls (can be validated via timing counters when timing feature is enabled).

  ## Performance Validation / Rollout

  - Baseline benchmarks with knob off must match current performance within noise:
      - cargo run --release --features tools --bin bench_voronoi -- 500k --no-preprocess (and a smaller size for iteration)
  - Then knob on, compare:
      - Total time
      - Packed timings + count of expansions
      - Resumable kNN time reduction
  - Ship with default packed_knn_expand_r2 = false.
  - If results are positive, consider a follow-up plan:
      - Group-triggered expansion when multiple queries in a cell fail r=1, to amortize the build.

  ## Assumptions / Defaults Chosen

  - First implementation is only r=2 band expansion.
  - Trigger is per-query only, and only after r=1 packed is exhausted and clipping is still unproven.
  - Controlled via a runtime knob on VoronoiConfig, default false.
  - If any expansion step becomes “too big” (candidate cap exceeded or security2 unavailable), the system falls back to resumable kNN; correctness does not depend on expansion.