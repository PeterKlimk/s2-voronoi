# Live Cell Layout Boundary Inventory

**Status:** local-rebuild overlay/materialization slice selected, 2026-07-19

This inventory records the next QUAL-001B ownership boundary after `EffectiveGeometry` made the
terminal position/cell/index arrays one coherent owner. It does not reopen the reconciliation
signatures that already failed the release counter gate.

## Existing measured boundary

`LiveCellLayout` is already accepted at topology-summary and selected reconciliation reader seams.
The segment-reader and localized duplicate-key families carry one paired view. Checked builds audit
the cell-id/index capacities and declared live spans on reconciliation's defect path.

The following expansions were tested and reverted after reproducing the same optimizer cliff:

- the semantic old/new span comparison;
- the localized unpaired-edge scan family; and
- a mutable collinear-drop rewrite owner.

Those raw signatures remain explicit measured fallbacks. The overlay migration must not route
through them or use its success to claim that their codegen has changed.

## Local-rebuild overlay

`WorkingDiagram` currently stores `base_cells` and `base_cell_indices` as independent references.
They are borrowed together from reconciled effective geometry and never mutate during an overlay;
spliced boundaries live in the separate `overrides` map.

All grow-loop base-boundary reads pass through `WorkingDiagram::boundary`. That method selects an
override when present and otherwise repeats the cell-record/index-buffer range expression. It is
used by:

- two-ring neighbor gathering;
- triple-to-vertex lookup;
- splice footprint and winding reconciliation;
- global winding selection;
- localized and debug-oracle residual scans; and
- touched-vertex/owner expansion across grow rounds.

`WorkingDiagram::into_flat` separately repeats the same base-span expression for unspliced cells
and reads the backing index length to reserve the replacement buffer. The constructor therefore
admits a mismatched cell/index pair even though every reader relies on their coherence.

This is distinct from `EffectiveGeometry`: the geometry owner is mutable pipeline storage, whereas
`LiveCellLayout` is the immutable paired view borrowed by one rebuild attempt. Vertex positions,
assembly keys, overrides, and minted vertex side arrays are not part of the layout.

## Selected migration

Add two representation-preserving `LiveCellLayout` accessors:

- direct-index `span(cell)`, using the same cell indexing and slice expression as the current
  trusted internal reader; and
- `index_count()`, used only for materialization capacity.

Do not implement `span` through `slice.get`, `checked_span`, an iterator adapter, or a new error
path. Those forms change the trusted grow-loop contract and the first has already demonstrated a
repeatable codegen regression elsewhere.

Replace `WorkingDiagram`'s two base fields with one `LiveCellLayout`. Rename its constructor from
`from_assembled` to `from_reconciled` and require the caller to pass an already paired layout. The
constructor may run `debug_assert_valid` in checked builds; release construction remains the same
two-reference move with no scan.

`boundary`, `num_cells`, the residual-scan reservation, and `into_flat` then read through the view.
Keep override selection, minted ids/positions/keys, flattening order, capacity choice, and returned
tuple exactly unchanged.

## Gate

Focused coverage must retain:

- live-span behavior with stale backing-buffer tails;
- overlay reads switching from base boundaries to overrides;
- deterministic overridden-cell ordering and flattening;
- accepted local-rebuild geometry and output-resolution footprints; and
- rejected/disabled/no-trigger behavior through the existing contract suites.

Run the complete release, checked, no-default-feature, and all-feature Clippy gates. Compare the
matched release artifact and the usual seven 500k single-thread Fibonacci instruction/branch
pairs. Because the new accessor is active only during a rebuild attempt, also run a structural
counter comparison on a deterministic mega workload if it actually triggers the overlay; wall
clock remains advisory on the shared host.
