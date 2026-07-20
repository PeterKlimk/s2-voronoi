# Live Assembly Phase Inventory

**Status:** first QUAL-001D post-scatter extraction accepted, 2026-07-20

This inventory covers `live_dedup::assemble::assemble_sharded_live_dedup`. It is independent of
the closed QUAL-001B handoff question: `AssemblyResult` still moves its global geometry arrays
directly into `EffectiveGeometry`, and this slice does not introduce another geometry owner or a
cell-layout wrapper.

## Program and ownership eras

The current assembly function contains four sequential ownership eras.

1. **Mutable shard repair.** Drain mismatch, overflow, and deferred-slot bookkeeping; reconcile
   cross-shard edge checks; emit mismatch-origin diagnostics only for a nonempty mismatch set;
   resolve deferred references or mint fallback representatives; update owner incidence and
   representative-drift evidence; then discard the temporary override lookup.
2. **Finalized shards.** Convert `ShardState` to `ShardFinal`, dropping dedup-only queues and pools.
   Compute checked global vertex offsets, copy positions into one global allocation, and move each
   vertex-key allocation into `ShardedVertexKeys` without concatenating it.
3. **Global cell materialization.** Emit generator-ordered checked cell prefixes, reduce the
   shard-owner incidence summary, choose generator- or shard-order traversal from the existing
   spatial-correlation classifier, and scatter every primary local vertex id into uninitialized
   disjoint final spans. Only after all writers join is the final index length published. Sparse
   foreign-reference overrides then patch their final cell/offset destinations.
4. **Post-scatter confirmation.** Gather the globally identified cells selected by construction's
   widened exact-zero hint, inspect their final post-override cycles and global positions, normalize
   confirmed distinct-id pairs, then sort and deduplicate the candidate set. Assemble timing and
   return all global arrays, provenance, evidence, and topology facts.

These eras cannot be freely reordered. Overflow and deferred resolution can change references,
mint positions/keys, and incidence. Vertex offsets are valid only after that mutation stops.
Primary scatter deliberately writes placeholder values for foreign references, so exact-zero
confirmation cannot read cell cycles until sparse patching completes. Timing-only reductions still
borrow `finals` near the return, extending that owner even after geometry is materialized.

## Existing useful boundaries

The mutable repair era is already divided at meaningful mutation seams:

- `collect_shard_bookkeeping` drains and capacity-plans its three transient streams;
- `resolve_edge_check_overflow` owns sorted cross-shard matching;
- `patch_deferred_slots_with_fallback` owns logical-reference lookup, fallback minting, incidence,
  and drift evidence; and
- `ShardState::into_final` makes dedup-state destruction explicit.

The scatter kernel is also already isolated as `scatter_local_indices`, with `inline(always)` and
an explicit unsafe contract. The surrounding two-mode traversal remains flattened so its Rayon
closures capture raw destinations and immutable source slices by value. Those are deliberate
codegen and locality decisions, not missing generic abstractions.

## Selected extraction

Extract only post-scatter exact-zero confirmation behind one private helper and one private result:

```text
ConfirmedZeroEdgeHints {
    candidates: Vec<(u32, u32)>,
    hinted_cell_count: usize,
}
```

`confirm_exact_zero_edge_hints` receives finalized shards plus raw immutable slices of the global
positions, cells, and index buffer. It performs exactly the existing hint-cell gather, final-cycle
scan, distinct-id check, exact final-position comparison, normalized-pair insertion, sort, and
dedup. The caller starts and stops the existing timing span around the helper, then moves the two
result fields into `AssemblyResult`.

This is a real phase boundary rather than a generic layout abstraction:

- all mutable and unsafe assembly work is complete before entry;
- its four inputs are immutable observations with the same lifetime;
- its two outputs are correlated evidence from one scan;
- no output can affect materialization, repair, or scatter policy; and
- output-resolution already consumes the candidate vector and hint count as one discovery source.

Use default compiler inlining first. Add no `inline`, `cold`, or outlining attribute unless matched
artifact and counter evidence supports one isolated variant.

## Shapes deliberately excluded

- Do not extract a broad `AssemblyContext` borrowing assignment, shards, offsets, final arrays,
  timers, and outputs. Their mutation phases and lifetimes differ, and the wrapper would obscure
  rather than enforce those transitions.
- Do not introduce `OwnedCellLayout`, `FinalArrays`, or an assembly-specific geometry owner. The
  QUAL-001B handoff inventory already established that such a value would be immediately unpacked
  into `EffectiveGeometry` or propagated through unrelated downstream policy.
- Do not route cell prefixes, primary scatter, sparse patching, or hint confirmation through
  `LiveCellLayout`. The arrays are incomplete or mutable during the first three operations; adding
  accessors to the final read-only scan alone creates no new lifetime guarantee.
- Do not extract or genericize the generator-order/shard-order scatter dispatch. Its raw pointer,
  by-value capture, source ordering, destination partition, and `set_len` publication sequence are
  source-pinned performance and safety decisions.
- Do not combine vertex offsets, position copying, and key transfer in this slice. They share the
  `finals` owner but have different allocation strategies under parallel and scalar builds.
- Do not move sparse reference patching into the selected helper. It mutates the final index
  buffer and must remain visibly ordered after joined primary scatter and before every final-cycle
  read.
- Do not alter the construction hint predicate, representative-drift certificate, output-
  resolution policy, candidate pair representation, numeric comparison, or timing schema.
- Do not move timing-feature-only reductions out of the assembly body concurrently. Their
  configuration-dependent fields would broaden the result solely to shorten the function.

## Semantic gate

The extraction must preserve:

- shard-order concatenation of hinted global cell ids and the pre-scan hint-cell count;
- use of final generator-ordered cell spans only after sparse overrides are applied;
- skipping equal vertex ids before position comparison;
- the current f64 coordinate-difference sum and exact `== 0.0` decision;
- canonical `(min, max)` candidate pairs followed by unstable sort and dedup;
- empty-vector allocation behavior when no cells or edges qualify;
- the existing `dedup_zero_hints` timing boundary and `AssemblyResult` fields;
- debug sentinel and final-layout assertions outside the helper;
- unchanged representative-drift fallback selection; and
- byte-identical final diagram and output-resolution reports.

Add a direct helper regression with multiple hinted cells that rediscover the same equal-position,
distinct-id pair, pinning both the uncollapsed hint count and deduplicated normalized candidate.
Existing assembly, output-resolution, reconciliation, and strict-validation suites remain the
pipeline-level gate.

## Activity characterization

Timing-enabled single-thread runs selected counter workloads; their wall times are not acceptance
measurements on the busy shared host.

| Workload | Scatter mode | Hint cells | Confirmed candidates |
|---|---|---:|---:|
| 500k Fibonacci | generator order | 6,827 | 5 |
| 500k uniform, seed 12345 | shard order | 8,466 | 1 |
| 500k clustered, seed 1 | shard order | 29,579 | 11 |
| 1M clustered, seed 1 | shard order | 60,793 | 27 |

The 1M clustered check also reproduced the expected three assembly mismatches, four-cell
reconciliation footprint, and certified discovery path. Treat these current counts as
workload-selection observations, not replacements for the older closed campaign record; the
selected helper preserves current evidence rather than pinning historical totals.

## Validation and performance gate

Run formatting and focused `live_dedup::assemble`, output-resolution, edge-reconciliation, and
validation tests, then the complete release, checked, no-default-feature, and all-target/all-feature
Clippy gates.

Build the dirty candidate and immediate parent together with native `tools` artifacts. Inspect
`assemble_sharded_live_dedup`, the helper/result symbol shape, aggregate sections, and file size.
Then run interleaved single-thread hardware counters for:

1. 500k Fibonacci, default bins — generator-order scatter plus active hints;
2. 500k uniform seed 12345, default bins — shard-order scatter plus active hints;
3. the same two workloads with `VORONOI_MESH_BIN_COUNT=96`; and
4. 500k clustered seed 1 — the denser active-hint path.

Retired instructions and branches are primary; cache counters help attribute a change that affects
the hinted-cell gather. Every sample retains switch/migration telemetry. Reject a repeatable loss
in either scatter regime or the dense-hint workload. A quiet wall-clock run is needed only for a
strong unexplained signal; readability alone does not justify an assembly regression.

## Accepted result

The selected extraction was accepted on 2026-07-20 against immediate parent `62b7851`.

- Private `ConfirmedZeroEdgeHints` now owns the candidate vector and pre-scan hint-cell count;
  `confirm_exact_zero_edge_hints` owns only the final read-only gather, scan, normalization, sort,
  and dedup phase. The existing timer remains around the helper call.
- A direct regression pins two hinted cells rediscovering one equal-position, distinct-id pair:
  the result retains hint count two and returns one normalized candidate. Complete release,
  checked, no-default-feature, and all-target/all-feature Clippy gates passed.
- LLVM fully inlined the helper. `assemble_sharded_live_dedup` shrank from `0x2fee` to `0x2fbc`
  bytes; aggregate `.text` shrank by 48 bytes, data and actual `.bss` were unchanged, and file size
  shrank by 72 bytes. No inline attribute was needed.
- Seven interleaved single-thread counter pairs for each gate were neutral. Candidate/parent mean
  instruction and branch ratios were respectively `1.00000131` / `0.99999811` for default-bin
  Fibonacci, `0.99999901` / `0.99999471` for default-bin uniform seed 12345, `0.99999881` /
  `0.99999586` for 96-bin Fibonacci, `1.00000183` / `0.99999503` for 96-bin uniform seed 12345,
  and `1.00000211` / `1.00000062` for clustered seed 1.
- Every sample recorded zero context switches and CPU migrations. The artifact and primary
  counters were unambiguous, so neither cache-counter attribution nor a quiet wall-clock rerun was
  justified.

The remaining assembly materialization and unsafe scatter shapes stay deliberately flattened.
Revisit them only when a new ownership invariant or consumer creates a narrower natural seam.
