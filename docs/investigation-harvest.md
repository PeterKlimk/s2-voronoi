# S2 Voronoi: Current Investigation Harvest

Verified read-only against current HEAD `92c80b6` (`agent/polar-cert-fix`). No tests or benchmarks
were run. This is a candidate backlog, not a claim that every item should be implemented.

## Highest-value correctness and reliability items

### 1. Bound coincident-cluster preprocessing

**Current code:** `src/cube_grid/weld.rs:39-93`, `src/knn_clipping/preprocess.rs:141-173`.

Default welding materializes every close pair before union-find. A cluster of `k` identical points
retains `k(k-1)/2` pairs and can allocation-abort on supported ordinary input.

**First validation:** instrument retained pair count/bytes on growing duplicate clusters. Consider a
pair budget, direct dense-run union, controlled fallback, or explicit error rather than unbounded
materialization.

### 2. Bound aggregate packed-key work, not only candidate slots

**Current code:** cap at `src/cube_grid/packed_knn/scratch/prepare.rs:94-106`; per-query center/ring key
materialization later in the same file.

`MAX_CANDIDATES_HARD` limits the 3x3 candidate population, but not aggregate
`queries × passing candidates` work or `chunk0_keys`/`tail_keys` storage. A dense accepted group can
therefore remain quadratic below the nominal cap.

**First validation:** record total keys and peak capacity per prepared group; create a concentrated
group where the dense index is unavailable or ineffective. Decide an aggregate work/memory budget
and route over-budget groups to the bounded fallback.

### 3. Fix `SphereLocator` above `2^24` slots

**Current code:** `src/locate.rs:25-32,77-84`.

`all_emit_map` stores identity slot values, but the directed packed layout interprets bits above 23
as a bin. Above 16,777,216 generators, some supposedly all-emit slots decode as query-bin 1 and may
be suppressed, allowing a non-nearest result.

**First validation:** synthetic layout/eligibility test around slots `2^24-1`, `2^24`, and `2^24+1`;
no huge diagram allocation is required.

### 4. Reject non-finite vertices in strict validation

**Current code:** `src/validation.rs:523-527,818-822,1095-1100`; serde structure checking in
`src/diagram.rs` does not enforce coordinate semantics.

For NaN, `(len_sq - 1).abs() > eps` is false. A topologically valid serialized diagram with a NaN
vertex can therefore evade the sphere check and appear strictly valid.

**First validation:** deserialize an otherwise valid diagram with one NaN component and require all
strict/report verification paths to reject it.

### 5. Define and validate `MergeWithin` thresholds

**Current code:** public option `src/lib.rs:171-187`; dispatch in
`src/knn_clipping/compute.rs:968-1002`; squaring/comparison in `cube_grid/weld.rs` and
`knn_clipping/preprocess.rs`.

- NaN follows an expensive standalone path and silently merges nothing.
- Infinity can produce all-pair behavior/collapse.
- A positive subnormal squares to zero, so even exact duplicates fail the strict-radius contract.

**First validation:** tiny release tests for NaN, ±infinity, zero, negative values, and
`f32::from_bits(1)`. Prefer a clean configuration/input error for unsupported thresholds.

### 6. Make adjacency completeness inspect live spans

**Current code:** `src/adjacency.rs:103-109,121-140`.

In-place reconciliation can shrink a cell while retaining gaps in the shared backing buffer.
Adjacency populates live positions but `is_complete()` scans the entire buffer, so unreachable
`NO_NEIGHBOR` holes can produce a false negative for a strictly valid diagram.

**First validation:** synthetic non-contiguous live spans with every live edge paired; compare
per-cell neighbor slices with `is_complete()`.

### 7. Align `has_post_repair_residuals` with plain `compute`

**Current code:** `src/lib.rs:337-343`; plain rejection in
`src/knn_clipping/compute.rs:139-156`.

The helper checks only unpaired edges, while plain compute also rejects unrepaired low-incidence
vertices. Report mode can therefore say there are no residuals for an output plain compute rejects.

**First validation:** construct a report state with a low-incidence defect and no unpaired edges;
assert helper and plain-path semantics agree.

### 8. Correct the cocircular Delaunay/adjacency contract

**Current code:** fan triangulation `src/delaunay.rs:61-67,100-108`; promise at `:69-79`.

A degree-4 Voronoi vertex has a four-edge adjacency cycle. Fan triangulation necessarily adds a
diagonal, so triangle edges cannot be “exactly” the adjacency pairs for supported cocircular
degeneracies.

**First action:** add a degree-4/cocircular fixture and clarify the documentation—fan diagonals are
tie-breaking triangulation edges, not Voronoi boundary adjacencies.

## Lower-frequency but real items

- **Out-of-range packed cells return `Ready` with uninitialized/stale scratch:**
  `packed_knn/scratch/prepare.rs:42-49` detects `cell >= num_cells` but returns a usable-looking
  `PreparedPackedGroupStatus::Ready` without initializing per-query state. The production driver
  currently supplies valid cells, but a future/malformed internal caller can observe prior-group
  candidates or panic. Return `SlowPath`/invalid instead; test a valid group followed by an
  out-of-range group under release.
- **Fallback raw-corner attribution:** `topo2d/builder/extract.rs:394-418` explicitly retains a raw
  corner/key when exact pair recovery fails. Rare geometry-accuracy defect that topology validation
  may miss; needs a targeted fallback fixture.
- **32-bit serde span overflow:** `diagram.rs:435-445` computes `start + len` unchecked. Use
  `checked_add`; validate on an i686 target or extracted boundary unit.
- **Extreme regrid cell-ID overflow:** `policy.rs:91-115`, `compute.rs:1062-1088`, and
  `cube_grid/build.rs:72-83` can permit more than `u32::MAX` cells for accepted `n > ~537M`.
  Likely OOM-preempted, but a pure arithmetic policy test is cheap.

## Test and contract hardening

1. **Exercise the production bin/local layout in NN contracts.** Current synthetic harness uses
   contiguous/global locals; production uses `src/live_dedup/binning.rs`. Build a real assignment and
   rerun the existing brute-force frontier contracts.
2. **Add dense seam/corner coverage.** Existing dense tests emphasize an interior single cell. Add
   >512-point clusters crossing cube-face edges/corners with populated neighbor cells.
3. **Harden packed-group shape in release.** The complete center-run/slot-order contract is mostly
   `debug_assert`-only. Prefer typed construction or a cheap `SlowPath` fallback; test partial,
   reordered, and layout-mismatched groups.
4. **Document the caller-owned frontier buffer protocol.** Exact frontier caching retains metadata,
   not slot contents. A caller that clears/mutates the buffer before a repeated probe can observe
   stale or impossible state. Encode the precondition or add misuse tests.
5. **Resolve shell determinism/comment drift.** `cube_grid/mod.rs` promises descending dot then slot,
   while the shell comparator uses dot only. Either add the slot tie-break and an equal-dot test or
   weaken the comment; remove the duplicate dead exhaustion field if still unused.

## Still-plausible performance experiments

These are experiments, not predicted wins. Require paired end-to-end measurement and correctness
fingerprints.

1. **Adaptive/batched ring-tail classification.** Current ring-hi scan is in
   `packed_knn/scratch/prepare.rs`; lazy per-query tail rescan is in `scratch/emit.rs`. Measure tail
   possible/requested, empty rescans, repeated dot evaluations, and unused stored bytes. Avoid
   unconditional eager tails.
2. **Incremental/resumable shell-layer emission.** `cube_grid/query/shells.rs` gathers, sorts, and
   copies the complete layer even when clipping terminates after a prefix. Measure layer size versus
   consumed prefix and mid-layer closure before designing a conservative next-dot bound.
3. **Defer dense-index construction until the retained grid is known.** `cube_grid/build.rs` builds
   and sorts dense entries before `compute.rs` may discard/rebuild the grid or clear the index.
   Count discarded builds, time, allocations, and peak RSS on clustered workloads.
4. **Bound the first packed chunk and materialize the remainder lazily.** Potentially reduces retained
   keys for first-chunk closures, but can lose badly when later chunks are common. Measure retained
   keys, later requests, recomputation, and peak bytes.
5. **Apply dense-band eligibility before the global candidate cap.** Narrow cap/mega opportunity:
   allow a proven bounded dense band to avoid premature SlowPath, while retaining an aggregate key
   budget and shell certificate.
6. **Batch shell takeover across same-cell queries.** Only as a whole-pipeline traversal/emission
   experiment; isolated dot-only SIMD was already a measured regression.

## Do not broadly retry

- per-(ring cell, query) spherical-cap pruning — adjacent caps rarely prune; measured net loss;
- packed-to-shell attempted-slot filtering — low duplicate coverage, extra branch/instruction cost;
- scalar shell dot-only SIMD — measured 6.5-8.5% slower;
- lower grid target density — measured sweep found density 24 faster than 16 by 4.8-7.1%;
- packed partial-selection rewrite — measured 7-14% loss at 2M;
- whole-ring packed bound skip — neutral/worse outside a narrow dense case;
- any local optimization of packed radius-2 expansion — the stage had no winning regime and was
  deleted end-to-end.

## Already harvested into the current implementation

- fused pre-center threshold selection and center hi/tail classification;
- duplicate frontier-cache copy removal;
- dense-band center pruning with clustered guard;
- emitted-key dot reuse;
- lazy dense-band state;
- density 24 tuning;
- deferred shell-frontier setup.

## Suggested validation order

1. Tiny deterministic tests: locator layout, NaN validation, threshold edge values, Delaunay contract.
2. Synthetic internal fixtures: residual helper and adjacency live-span completeness.
3. Reliability counters: duplicate-cluster pair bytes and aggregate packed key bytes.
4. Production-layout NN and dense seam/corner test expansion.
5. Only then run the ranked performance experiments with stable single-thread paired benches.

## Coverage of the original quality experiment

All seven normalized quality families from the very first experiment are retained here:

| Original finding | Current harvest location |
|---|---|
| Cached frontier does not restore caller buffer | Test/contract hardening #4 |
| Packed-group shape/layout is debug-only | Test/contract hardening #3 |
| Out-of-range packed cell returns `Ready` | Lower-frequency items, explicit bullet |
| Candidate cap permits quadratic center-key storage | Highest-value item #2 |
| Directed-layout tests use synthetic binning | Test/contract hardening #1 |
| Shell equal-dot order contradicts documentation | Test/contract hardening #5 |
| Scratch exhaustion state is dead | Test/contract hardening #5 |
