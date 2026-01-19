# RFC 0001: Directed Intra-Bin kNN + Packed-kNN Dominance

## Summary

This RFC proposes two related performance/complexity changes to the spherical Voronoi construction pipeline:

1. **Directed intra-bin neighbor elimination:** avoid testing kNN candidates that are in the same shard/bin and have a smaller `LocalId` than the current generator, and instead **consume earlier neighbors from incoming edgechecks** as mandatory seeds for geometry.
2. **Make packed-kNN the dominant kNN path:** replace the current “packed seed → per-cell resumable kNN retries → brute force” ladder with a packed-first hierarchy (including ring expansion) that shares work across queries in the same grid cell group; keep resumable/bruteforce only as a cold escape hatch (or remove it entirely once confidence is high).

The intent is to (a) remove redundant symmetric work (the dominant within-bin interactions), and (b) make fallback behavior more coherent and less per-query pathological at large `N`.

## Goals

- Reduce neighbor-testing cost by eliminating redundant within-bin “earlier generator” candidates.
- Preserve (and ideally improve) topological consistency by using the existing edgecheck protocol as a synchronization mechanism.
- Simplify `process_cell` by making the dominant kNN behavior batched and monotonic (no repeated restarts per query).
- Improve scaling behavior as `N` increases (especially when termination requires more neighbors to prove boundedness).

## Non-Goal

- *Exact* Voronoi diagrams. The target remains “~99.9% geometrically reasonable + topological consistency,” acknowledging floating point and near-degenerate cases.

## Current Architecture (Relevant Pieces)

### Ordering and identifiers

- Each generator is assigned to a **bin** (`BinId`) and a bin-local **processing rank** (`LocalId`).
- `LocalId` is assigned in per-bin cell-major order (the grid’s cell traversal order), preserving the `(grid.point_index_to_cell(g), g)` order.
- Within a single kNN grid cell group, the local id is effectively:

```
LocalId = group_start (cell-local offset within bin) + position within group
```

### Edgechecks: directed earlier → later within a bin

- During cell construction, edges are classified as:
  - cross-bin → overflow bookkeeping
  - same-bin + `local < local_b` → **emit edgecheck to later neighbor**
  - same-bin + `local > local_b` → resolve against **incoming edgechecks**
- Edgechecks carry:
  - `EdgeKey` for `(min(cell_idx), max(cell_idx))`
  - endpoint “thirds” for validation
  - per-endpoint vertex indices for dedup propagation

### kNN: packed seed + resumable fallback

- Packed-kNN is invoked per grid cell group (batched), but is currently used as a **seed** only (3×3 cells).
- If packed results don’t terminate/bound the cell, `process_cell` runs multiple stages of per-query resumable kNN (and may escalate k repeatedly).
- If still not bounded, the current implementation may run a full scan fallback.

## Proposal A: Directed Intra-Bin Neighbor Elimination

### Key idea

Voronoi adjacency is symmetric, but the system already has a **directed within-bin channel** (edgechecks from earlier → later) to reconcile topology and propagate indices. We can reuse that channel to eliminate redundant “discover earlier locals” work.

### Change in meaning of edgechecks

Edgechecks are currently consumed after geometry is built (during edge resolution). Under this proposal, incoming edgechecks are additionally treated as a **source of mandatory earlier-neighbor candidates for geometry**.

Concretely:

- For generator `g` with `(bin, local)`, gather incoming edgechecks addressed to `local`.
- Extract the neighbor generator id from each `EdgeKey`:
  - `(a, b) = unpack_edge(edge_key)`
  - if `a == g` then neighbor is `b` else neighbor is `a`
- De-duplicate the resulting neighbor list.

These neighbors are “mandatory” in the sense that we will **clip against them** (or otherwise ensure they are included as planes) before/alongside packed-kNN candidates.

Rationale: within a bin, most “earlier local” candidates are *not* true Voronoi neighbors and would eventually be clipped away (or shown irrelevant) if we tested them, so spending kNN budget on them is mostly wasted work. The edgecheck protocol already identifies the sparse subset of earlier generators that actually formed edges on the earlier side. By consuming incoming edgechecks as geometric seeds, the later cell can include the earlier **true** neighbors without having to enumerate and test the much larger set of earlier same-bin candidates that would be discarded anyway.

This does not change the termination model: when termination is proven, the result is still exact with respect to the conservative bound used by `can_terminate()`. The change is purely about how we source candidate planes—prefer “earlier true neighbors via edgechecks” over “scan earlier locals and clip most of them away.”

### Directed kNN filtering rule (within-bin only)

When consuming kNN candidate slots (from packed or any other source), skip candidates that decode to:

- `bin_b == bin` and `local_b < local`

This is the core “avoid testing same-bin generators with a smaller LocalId” optimization.

### Expected effect

- Eliminates most redundant within-bin symmetric candidate testing (dominant interaction).
- Reduces work most in dense within-bin areas (where earlier locals are abundant in the query’s local neighborhood).
- Replaces “search earlier locals” with “consume earlier locals that already proved adjacency from the earlier side,” via edgechecks.

### Invariants / debug assertions to retain confidence

- Every edgecheck consumed for geometry must refer to an endpoint equal to the current cell index.
- Incoming edgechecks should be bounded (per cell degree is typically small); if not, treat as suspicious and keep instrumentation.
- When skipping `local_b < local`, ensure the cell still clips against all edgecheck-derived neighbors.
- Avoid hard correctness dependencies on debug-only caps (e.g. matching logic that assumes `incoming_checks <= 64` or `cell_vertices <= 64`). If these are exceeded, prefer a slow but safe matching strategy (e.g. `Vec<bool>` matched flags) and/or disable directed skipping for that cell.

### Failure modes and how we accept them

- **Earlier-side omission bugs:** If an earlier cell fails to emit an edgecheck for a "true" neighbor and later-side scanning is suppressed, the later cell may omit that neighbor plane and become too large. This is treated as a normal “bug makes output wrong” category; we rely on assumptions + debug asserts rather than trying to harden against all logic bugs. The present policy of collapsing single-sided edges is close to equivalent to skipping them in the first place anyway.
- **Numerical one-sided edges:** Still expected and unavoidable (near-degeneracy and f64). The system already collapses one-sided edges in repair; the proposal does not attempt to eliminate these entirely.

### Implementation sketch (high level)

- Move the “take incoming edgechecks” concept to the start of cell construction:
  - Use it to build a `Vec<usize>` (global neighbor indices) for mandatory clipping.
  - Retain the same incoming checks for later edge resolution (avoid double-take).
- Clip against these neighbors first, then continue with packed-kNN candidate consumption (filtering out earlier locals).
- Keep the existing edgecheck validation and bad-edge reporting for the remaining edges (still useful for diagnostics).

## Proposal B: Packed-kNN Dominance (and correlated fallback)

### Problem statement

The current pipeline often looks like:

1. Packed seed (3×3) → not enough candidates / termination not proven
2. Per-query resumable kNN stage(s)
3. Potentially escalate k repeatedly
4. Potentially brute force

This has two major issues:

- **Correlation:** If packed seed is insufficient in a sparse region, many nearby queries will fail similarly.
- **Complexity & wasted work:** `process_cell` becomes a control-flow ladder and the “resumable” path is effectively restarted per stage, causing repeated scanning and bookkeeping.

### Proposed hierarchy

Make packed-kNN the primary driver by expanding the packed query scope as needed, in a way that shares work across all queries in the same grid cell group:

1. **Stage A (current):** packed 3×3 candidate collection + top-k selection.
2. **Stage B (new):** adaptive **ring expansion** (increase candidate cell ranges beyond 3×3) when:
   - `count < k`, or
   - termination cannot be proven using the current “outside bound” (`security`)
3. **Stage C (new):** keep packed behavior but switch internal algorithmic mode as needed:
   - dense slab for small candidate sets
   - streaming top-k for large candidate sets
4. **Stage D (cold escape hatch):** brute force (ideally batched per group), used only when packed expansion hits a cap or fails to converge.

Resumable kNN becomes optional:

- either remove it entirely once packed expansion is robust, or
- keep it behind a debug/feature flag as a correctness/perf backstop during rollout.

### Why this fits termination behavior at large N

Termination proof can require more neighbor planes as `N` increases (tighter geometry, smaller cells). A packed-first approach that can expand its candidate scope monotonically avoids the “nearly everyone falls back to resumable” failure mode by making “ask for more” a shared group-level operation.

### API sketch (conceptual)

Introduce a packed entry point that can expand search radius and report an “outside-of-radius bound” suitable for termination checks:

- Input: center cell, query slots, `k`, and a `PackedExpansionState` (per cell group)
- Output per query: neighbor slots, count, and `outside_dot_bound` for unseen candidates

The expansion state is reused for subsequent queries in the same group so that once a region “learns” it needs a larger ring, later queries don’t thrash through the fast path again.

#### Outside-of-radius bound (proposed definition)

Let `N(r)` be the set of cells within neighbor-graph radius `r` of the center cell (radius measured by repeated application of the precomputed 3×3 `cell_neighbors`, i.e. the same connectivity that already defines “3×3” on stitched cube faces).

When packed has fully scanned all candidate points in `N(r)`, define the **frontier ring** as:

- `F(r+1) = neighbors(N(r)) \\ N(r)`

Then define the termination/security bound for unseen candidates as:

- `outside_dot_bound(q, r) = max_{c in F(r+1)} max_dot_to_cell_cap(q, c)`

where `max_dot_to_cell_cap` is the existing spherical-cap upper bound computed from `cell_center[c]` and `cell_cos/sin_radius[c]`.

This matches the current `r = 1` behavior (3×3 neighborhood + “ring2” frontier) and relies on the same empirical invariant: under the cap model, the maximum possible dot among all cells outside `N(r)` occurs on the immediate frontier `F(r+1)`. For ring expansion, we should validate this invariant for `r <= r_max` at a few grid resolutions via randomized sampling; if it fails for any configuration, fall back to a more conservative (slower) bound computation for that case (e.g. scan additional rings, or use a best-first cell-cap search).

#### Optional refinement: bounded chunks (to keep future selection options open)

To preserve early termination without requiring a total sort of all candidates, an alternate contract is:

- Packed emits candidates in **bounded chunks** `C0, C1, ...` such that all dots in `C{i+1}` are `<=` all dots in `C{i}` (e.g. by consuming dot buckets from high→low, or by per-chunk partial sorts).
- After each chunk, packed provides a conservative `unseen_dot_bound_i` for any not-yet-emitted candidate (including outside the current ring radius during expansion).
- The cell builder clips chunk-by-chunk and checks `can_terminate(unseen_dot_bound_i)` between chunks.

This keeps the dominant behavior compatible with later optimizations (bucketing, smaller batches, different selection strategies) while maintaining a sound termination proof model.

### Instrumentation/metrics to guide tuning

- Distribution of `packed_count` vs `k` per group and overall.
- Distribution of required ring radius to reach `k` and/or to terminate.
- Fraction of cells that require escalation beyond the initial packed radius.
- Total time share of:
  - packed setup / scanning / select
  - clipping
  - (any remaining) fallback paths

## Heuristic: choosing grid resolution and initial k

Today grid resolution is driven by a fixed target density (expected points per cell), and initial packed `k` is tied to the resumable schedule’s first stage.

As `N` grows, two levers matter:

- **Grid density:** higher density increases candidates in the initial packed window (good for sparse regions), but increases per-query scan cost and can push dense areas into slower internal modes.
- **Initial k:** larger k increases initial neighbor plane coverage but also increases per-cell work if the candidate supply is available.

Recommendation (high level):

- Treat `(grid target density, initial packed k, max ring radius)` as a coupled configuration.
- Prefer keeping target density in a range that keeps the initial packed window “comfortably above k” on average, and rely on adaptive ring expansion for outliers.
- Consider making target density (or max ring radius) a function of `N` if empirical data shows a consistent drift in required neighbor counts for termination.

This RFC does not lock in a specific formula; it calls for measurement-driven tuning using the instrumentation above.

## Rollout plan (phased)

1. **Phase 1: consume edgechecks for geometry**
   - Extract earlier-neighbor candidates from incoming edgechecks and clip against them.
   - Keep existing kNN behavior unchanged.
2. **Phase 2: within-bin directed filtering**
   - Filter out same-bin `local_b < local` candidates when consuming packed seeds and any fallback candidates.
   - Add debug asserts to ensure edgecheck-derived neighbors are always included.
3. **Phase 3: packed ring expansion**
   - Implement adaptive packed expansion for groups that fail to reach `k` / termination.
   - Keep existing per-query fallback only as a cold escape hatch.
4. **Phase 4: deprecate resumable**
   - Remove resumable from the dominant path, optionally feature-flag it.

## Open questions

- What exact “outside-of-radius” bound should be propagated for termination once radius > ring2? (See proposed frontier definition above; still needs validation + a conservative fallback.)
- What caps should exist on max ring radius / candidates to avoid worst-case blowups?
- Do we require fully sorted top-k output from packed, or is a bounded-chunk contract sufficient (and preferable for future bucketing)?
- Should we ever temporarily re-enable earlier-local scanning as a debug-only “consistency detector,” or is repair + asserts sufficient?
