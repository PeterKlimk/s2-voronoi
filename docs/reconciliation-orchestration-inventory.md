# Reconciliation Orchestration Inventory

**Status:** first QUAL-001D round-state extraction accepted, 2026-07-20

This inventory covers reconciliation orchestration and state carried across its primary and
backstop passes. It does not change reconciliation evidence, numeric policy, cell mutation,
local-rebuild policy, or any raw cell-layout signature retained by earlier counter gates.

## Current boundaries

Reconciliation has three orchestration layers:

1. `compute::reconcile_edges` snapshots diagnostic/oracle options only when assembly supplied at
   least one mismatch, optionally emits pre-mutation telemetry, converts mismatch records to the
   narrow `EdgeRecord` input, invokes reconciliation, and records timing.
2. `reconcile_edge_mismatches` owns the zero-record fast return, the primary pass, the direct
   output-invariant scan, optional synthesized backstop pass, residual scan, and result
   normalization.
3. `run_reconciliation_rounds` owns one capped fixpoint: localized collinear drops, merge proposal,
   diameter/face-safety acceptance, rejected-component seeding, selected apply backend, and
   convergence detection.

The lower-level merge collection, component bounding, mutation backends, localized residual scan,
and telemetry are already separate semantic operations. The readability problem is the mutable
state threaded manually across the two calls to `run_reconciliation_rounds` and assembled by an
inline finalizing closure.

## Clean-path boundary

An empty `edge_records` slice returns `ReconcileResult::default()` before any release-mode scan,
state construction, or allocation. Debug builds run the global unpaired-edge oracle first to
continuously audit assembly's detection-completeness claim.

This branch is load-bearing. Production normally has no mismatch records, and multiple nominally
cold signature changes elsewhere in this function have perturbed ordinary code generation. The
first extraction must leave the entry signature, empty check, debug oracle, and immediate return in
place. It must not insert a context object, helper call, option lookup, layout audit, or allocation
before that return.

## Defect-run phase sequence

After the fast return, the current order is:

1. Audit the cell/index pairing in checked builds.
2. Derive the primary record-cell cover.
3. Run primary reconciliation to a fixpoint.
4. Record the primary cover for output-resolution rescanning only if a cell span changed.
5. Scan the primary region and its verified partners for bad interior edges.
6. Finish immediately if that scan is clean.
7. Synthesize deduplicated proximity-only records from the surviving endpoint keys.
8. Run the backstop records to a fixpoint, sharing the primary pass's merge ledger.
9. Add the backstop cover to the resolution footprint only if that pass changed a span.
10. Scan the union of both record covers, convert residual vertex edges to cell pairs, and finish.

The primary scan must precede synthesis, and synthesis must use the post-primary cell state.
Residual ordering comes from the sorted unpaired scan and must remain unchanged.

## State carried across passes

| State | Updated by | Final treatment | Why it persists |
|---|---|---|---|
| `MergeLedger` | accepted merge components | not returned | Later rounds measure component diameter against every original member, including ids retired by an earlier pass. |
| `local_rebuild_seed_pairs` | rejected diameter/face-safety components | sort and deduplicate | Rejected identity edits must remain explicit Hull3d seeds even if the unchanged diagram happens to scan clean. |
| `merge_affected_cells` | accepted proposals and rejected components | sort and deduplicate | Later localization must include cells whose references may no longer follow current vertex-key ownership. |
| `resolution_scan_cells` | each pass that actually changes a span | conditionally extend, then sort and deduplicate | This is a mutation certificate, not a defect-region inventory. A rejected proposal alone must not trigger an output-resolution rescan. |
| `MergeSafetyStats` | every bounded-component round | copied to result counters | Timing reports the total localized cell cover and global fallbacks across both passes. |

The finalization order is significant. `merge_affected_cells` is added to
`resolution_scan_cells` only when the latter is already nonempty. This includes every key-owner
cell after a real mutation without converting rejected-only evidence into a mutation footprint.

## Selected first extraction

Introduce one private, default-constructed `ReconcileRunState` after the empty-record return and
checked-build audit. It owns exactly the five values above.

- Pass `&mut ReconcileRunState` to `run_reconciliation_rounds` instead of four independently
  pairable mutable accumulators.
- Keep vertices, cells, indices, vertex keys, epsilon, options, records, and `MergeMode` as their
  existing explicit arguments. In particular, do not introduce a broad borrowed context or change
  the measured raw cell/index signatures.
- Record changed primary/backstop covers through a small state operation that preserves the current
  clone/extend timing and allocation behavior.
- Replace the local `done` closure with a consuming state operation that performs the exact current
  normalization order and constructs the unchanged `ReconcileResult`.
- Keep the primary/backstop control flow flattened in `reconcile_edge_mismatches`. Moving that body
  behind another function boundary is a later experiment, conditional on this smaller state owner
  passing its gates.

The state is defect-local orchestration, not geometry ownership. It must not contain vertices,
cells, indices, options, input records, synthesized records, or residual scan output.

## Retained and rejected shapes

- Do not move the zero-record path into a helper or construct state before it.
- Do not combine `ReconcileOptions` with mutable run state; options are an immutable snapshot made
  by the caller only for a defect-bearing run.
- Do not replace raw cell/index arguments with `LiveCellLayout` or a mutable owner. The semantic
  comparison, unpaired scan, and collinear-drop forms already failed the counter gate.
- Do not combine primary and synthesized records. `MergeMode`, first-primary-round duplicate
  discovery, and post-primary evidence synthesis give them different semantics.
- Do not deduplicate assembly mismatch records as part of this cleanup. The report retains every
  origin, and the performance idea needs separate activation and equivalence evidence.
- Do not move merge inference, component safety, application, or residual scanning into new modules
  in the same change. That would prevent attribution of any codegen movement.
- Do not change the eight-round cap, epsilon arithmetic, component ordering, representative choice,
  face visit order, convergence condition, or rebuild/in-place oracle switch.

## Semantic gate

Focused coverage must retain:

- the empty-record early return and checked detection-completeness oracle;
- primary and proximity-only behavior, including one-shared-endpoint and distant-endpoint cases;
- transactional diameter rejection across multiple rounds;
- localized merge-safety coverage and missing-provenance global fallback counters;
- rejected-component rebuild seeds;
- exact mutation footprints, including rejected-only versus actually changed runs;
- in-place/full-rebuild per-cell sequence agreement; and
- strict validity and mismatch origins on the deterministic in-bin and cross-bin integration net.

Run the focused reconciliation, output-resolution, and local-rebuild suites, then the complete
release, checked, no-default-feature, and all-feature Clippy gates.

## Performance gate

Compare the immediate parent and candidate artifacts, then run interleaved Linux perf counters:

- seven 500k single-threaded Fibonacci pairs with preprocessing disabled for the ordinary clean
  path;
- seven approximately 100k single-threaded `cubed` pairs for an active reconciliation path; and
- a smaller confirmation matrix at approximately 500k `cubed` if the first active-path result is
  close or favorable.

Retired instructions and branches are the primary signal on the shared host. Record context
switches and CPU migrations and ignore contaminated samples. Artifact sections/file size and the
`edge_reconcile` timing phase provide attribution; wall clock alone is not a decision criterion.
Any repeatable clean-path regression rejects or narrows the extraction even if defect-path timing
improves.

## Result

`ReconcileRunState` now owns the five cross-pass values selected above. Both primary and
proximity-only fixpoint calls receive one mutable state reference instead of four independent
accumulators. The state records changed-region cells and consumes itself to normalize the unchanged
`ReconcileResult`. The zero-record branch and debug oracle remain textually before state
construction, while the primary/backstop sequence and raw geometry arguments remain flattened.

The direct rejected-component test now also consumes the run state through production
finalization. Existing tests continue to pin changed and rejected-only resolution footprints, and
the deterministic integration net retained strict validity, mismatch origins, and in-place/rebuild
cycle agreement. The complete release, checked, no-default-feature, and all-feature Clippy gates
passed.

Against parent `37b0f65`, the matched release artifact removed 544 text bytes and 3,552 BSS bytes,
kept data size unchanged, reduced aggregate section accounting by 4,096 bytes, and reduced file
size by 616 bytes. Although that resembled an earlier optimizer-cliff fingerprint, the counter
gates were neutral:

- Seven 500k single-threaded Fibonacci pairs produced mean candidate/parent ratios of
  `1.000003220` instructions and `1.000004729` branches, with ranges
  `0.999993432..=1.000006944` and `0.999990748..=1.000013778`.
- Seven approximately 100k single-threaded `cubed` pairs produced `1.000010264` instructions and
  `1.000007825` branches, with ranges `0.999999243..=1.000032918` and
  `0.999988724..=1.000041528`.
- Five approximately 500k single-threaded `cubed` confirmation pairs produced `1.000003216`
  instructions and `1.000004527` branches, with ranges `0.999999901..=1.000006197` and
  `0.999998370..=1.000009988`.

Every measured sample recorded zero context switches and CPU migrations. Wall clock was ignored on
the busy host. The state owner is accepted; any further defect-body function extraction remains a
separate measured slice.
