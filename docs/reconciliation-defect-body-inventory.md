# Reconciliation Defect-Body Boundary Inventory

**Status:** one QUAL-001D helper extraction selected, 2026-07-20

This inventory covers the next reconciliation-orchestration boundary after
`ReconcileRunState` grouped state shared by the primary and synthesized-backstop passes. It does
not change evidence, mutations, error behavior, numeric policy, or previously measured cell-layout
signatures.

## Boundary today

`reconcile_edge_mismatches` currently owns two different programs:

1. The detection-completeness gate checks whether assembly emitted any mismatch record. Release
   builds return an empty result immediately; checked builds first prove the global topology scan is
   also clean.
2. The nonempty-record program audits the cell/index pairing in checked builds, constructs
   `ReconcileRunState`, runs primary reconciliation, scans the resulting topology, optionally runs
   a synthesized proximity-only backstop, scans residuals, and normalizes the result.

The gate is the common production path. The second program is defect-only and now has one coherent
state owner, but its roughly fifty lines remain embedded after the gate. This makes the entry's
critical early-return contract visually compete with the uncommon mutation program and prevents
the latter from having a direct nonempty-record precondition.

## Inputs and mutation contract

The defect body needs exactly the current entry arguments:

- nonempty `edge_records` as immutable evidence;
- immutable vertex positions and vertex-key provenance;
- mutable `Vec<VoronoiCell>` and `Vec<u32>` storage, because the oracle apply backend may replace
  both allocations while the production backend shrinks live spans in place;
- the spherical chord epsilon; and
- immutable `ReconcileOptions` selecting diagnostic/oracle behavior.

It returns the unchanged `Result<ReconcileResult, VoronoiError>`. Reconciliation is transactional
per proposed merge component, not across the whole function: an earlier productive round may have
mutated cell spans before a later malformed-state error is discovered. The extraction must not add
cloning, rollback, deferred commit, or a new intermediate result.

## Selected extraction

Keep `reconcile_edge_mismatches` as the crate-visible contract boundary:

1. Preserve its exact arguments and documentation.
2. Keep the `edge_records.is_empty()` branch, checked global oracle, and immediate default return
   in the entry.
3. Keep the defect-bearing `LiveCellLayout::debug_assert_valid` audit in the entry, after the empty
   return and before any mutation.
4. Delegate the remaining nonempty-record program to a private
   `reconcile_recorded_mismatches` helper with the same seven explicit arguments.

The helper may debug-assert its nonempty-record precondition, constructs `ReconcileRunState`, and
retains the current primary/backstop statements in their current order. Use the compiler's default
inlining decision in the first candidate; add no `inline`, `cold`, or optimization attribute.

This extraction gives the two programs names without changing data representation. The entry owns
whether reconciliation should run and whether the input layout is structurally trusted; the helper
owns what a recorded mismatch run does.

## Shapes deliberately excluded

- Do not pass a `ReconcileContext` containing borrowed geometry, keys, epsilon, or options. It would
  widen aliasing and lifetime changes while hiding which inputs mutate.
- Do not absorb inputs into `ReconcileRunState`; that type owns cross-pass bookkeeping, not borrowed
  geometry or policy.
- Do not replace `cells`/`cell_indices` with a read-only or mutable layout owner. Those signatures
  have separate rejected counter evidence.
- Do not move the checked layout audit into the helper. The entry should establish its defect-run
  precondition before delegating.
- Do not move `compute::reconcile_edges` option lookup, telemetry, timing, or mismatch-record
  conversion. Those are caller orchestration with intentionally different lifetimes.
- Do not split primary and backstop phases further, create a pass-result record, deduplicate input
  records, or move helpers into another module in the same change.
- Do not add whole-stage rollback. That is a semantic redesign, not function extraction.
- Do not mark the helper `cold`: deterministic `cubed` inputs make reconciliation substantial, and
  branch-frequency policy requires its own evidence.

If the default helper changes clean-path counters adversely, test at most one source-preserving
fallback at a time. `inline(always)` may retain the named source boundary while restoring the
previous flattened optimizer shape; `inline(never)` is appropriate only if evidence shows unwanted
reinlining or code duplication. Either is a new measured variant, not an unreported tweak.

## Semantic gate

The extraction must preserve:

- empty-record release behavior and the checked detection-completeness oracle;
- malformed-layout error timing before mutation in checked builds;
- primary-before-scan and scan-before-synthesis ordering;
- shared merge-ledger state across primary and backstop rounds;
- partial-mutation-on-later-error behavior where currently observable;
- rejected-only versus changed mutation footprints;
- residual pair ordering, rebuild seeds, affected cells, and merge-safety counters;
- in-place/full-rebuild per-cell cycle agreement; and
- deterministic integration-net mismatch origins and strict output validity.

Run focused reconciliation, assembly-to-reconciliation, output-resolution, and local-rebuild
coverage, followed by the complete release, checked, no-default-feature, and all-feature Clippy
gates.

## Performance gate

Build matched immediate-parent and dirty-candidate artifacts. Compare sections, file size, and
symbol layout, then run interleaved Linux perf counters:

- seven 500k single-threaded Fibonacci pairs with preprocessing disabled for the empty-record
  common path;
- seven approximately 100k single-threaded `cubed` pairs for active reconciliation; and
- approximately 500k `cubed` confirmation if the active result or artifact shape needs
  disambiguation.

Retired instructions and branches are primary on the busy shared host; context switches and CPU
migrations identify contaminated samples. Wall clock is advisory. A repeatable clean-path loss
rejects or reshapes the helper even if the defect path improves.
