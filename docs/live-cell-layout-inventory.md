# Live Cell Layout Boundary Inventory

**Status:** selective boundary complete; semantic comparison and unpaired readers accepted after
layout retests, 2026-07-20

This inventory records the selective QUAL-001B ownership boundary after `EffectiveGeometry` made
the terminal position/cell/index arrays one coherent owner. A post-closeout audit reopened the
semantic old/new span comparison after isolating its prior counter signal to codegen partitioning.
The unpaired-reader family was then reopened after its earlier cliff disappeared; the mutable
reconciliation signature remains closed.

## Existing measured boundary

`LiveCellLayout` is already accepted at topology-summary and selected reconciliation reader seams.
The segment-reader and localized duplicate-key families carry one paired view. Checked builds audit
the cell-id/index capacities and declared live spans on reconciliation's defect path.

The following expansions originally reproduced the same optimizer cliff:

- the semantic old/new span comparison, now accepted after the controlled retest below;
- the localized unpaired-edge scan family, now accepted after a neutral default-build retest; and
- a mutable collinear-drop rewrite owner.

The mutable-layout raw signature remains an explicit measured fallback. The overlay migration does
not route through it or claim that its default codegen has changed.

## Semantic-comparison retest

`cell_spans_differ` now receives an old and a new `LiveCellLayout`, preventing its cell records and
backing index buffers from being cross-paired at the semantic fixpoint boundary. The rebuild and
in-place reconciliation oracle continues to pin the active behavior.

The default benchmark cannot execute this function because it selects `ReconcileApply::InPlace`;
the comparison exists only in the diagnostic rebuild backend. Nevertheless, the ordinary default
artifact repeated a smaller version of the earlier counter fingerprint. The displacement varied
substantially by workload, from `+0.0996%` instructions / `+1.3600%` branches on 500k Fibonacci to
`+0.0076%` / `+0.0731%` on 100k mega. Cycles were noisy and directionally favorable in Fibonacci,
uniform, clustered, and mega matrices, with no scheduling contamination.

As a controlled codegen-partition test, rebuilding candidate and parent with
`-C codegen-units=1` produced byte-identical `.text`, `.rodata`, exception, and unwind sections,
identical aggregate/file sizes, and neutral seven-pair Fibonacci counters (`0.999999795`
instructions, `0.999998433` branches). The default-build movement is therefore an optimizer/layout
artifact from inactive code, not additional comparison work. The typed boundary is retained; the
other formerly rejected readers are not implicitly accepted by this result.

## Unpaired-reader retest

The whole unpaired-edge reader family now carries one `LiveCellLayout` through localized region
expansion, partner-cell edge counting, and the checked-only global differential. Each scan forms
its view after the preceding reconciliation round, preserving the existing mutation boundary.

Against parent `add4409`, seven-pair instruction/branch means were neutral on 500k Fibonacci
(`0.999998856` / `0.999999338`), 500k uniform seed 12345 (`0.999999804` / `1.000000368`), 100k
clustered seed 1 (`0.999998424` / `0.999997622`), and 100k mega seed 1 (`0.999995289` /
`0.999999346`). Cycles were unresolved; every sample had zero context switches/migrations. The
artifact added 168 text bytes, removed 160 BSS bytes, and grew by 120 file bytes. The historical
optimizer cliff is absent, so the paired reader boundary is retained.

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

## Result

`WorkingDiagram` now stores one `LiveCellLayout` and its `from_reconciled` constructor requires the
caller to form that pairing explicitly. Base-boundary traversal, cell count, residual-scan
reservation, and flattening all read through the view. Override selection, minted vertex storage,
flattening order, and the returned materialization tuple are unchanged. Checked construction audits
the layout; release construction remains a two-reference move with no scan.

Focused coverage pins direct live-span reads through stale backing slots, override substitution,
and flattened output. The complete release, checked, no-default-feature, and all-feature Clippy
gates passed, including the production local-rebuild contract suites.

Against parent `72a681d`, the matched release artifact removed 64 text bytes and 48 data bytes,
added 96 BSS bytes, reduced aggregate accounting by 16 bytes, and reduced file size by 48 bytes.
Seven interleaved 500k single-threaded Fibonacci pairs produced mean candidate/parent ratios of
`1.000000228` instructions and `1.000001841` branches, with pair ranges
`0.999997202..=1.000004276` and `0.999996770..=1.000010283`. Every sample recorded zero context
switches and CPU migrations.

A separate overlay counter run was not claimed. The deterministic 100k mega fixtures at fraction
`0.8`, seeds 1, 2, and 15, are now strict-valid with rebuilding disabled, and a debug run of that
test did not enter the overlay. Existing direct overlay tests and the release/checked local-rebuild
suites therefore provide the active-path semantic evidence until a natural accepted splice fixture
reappears.
