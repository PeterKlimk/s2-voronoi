# Effective Validation Layout Boundary Inventory

**Status:** one representation-preserving QUAL-001B migration selected, 2026-07-20

This inventory covers the fourth QUAL-001B migration stage: the live cell layout consumed by the
whole-effective-diagram strict gate. It does not reopen QUAL-001C's closed validation-fact
extractions or attempt to make the three validation consumers share one traversal.

## Boundary and ownership

`verify_sphere_effective_strict` has one production caller. After a local-rebuild overlay is
materialized, `maybe_rebuild_effective` canonicalizes the candidate cell cycles, temporarily
appends minted positions to the effective geometry, and validates the candidate before deciding
whether to commit its cell arrays or truncate the appended positions.

The gate currently receives four independent slices:

- effective generator positions;
- candidate vertex positions;
- candidate `VoronoiCell` records; and
- the candidate backing index buffer.

Only the final two are one representation. Generator/cell cardinality must remain independently
checkable, and candidate vertices have a distinct transaction lifetime: the base vector has
already been extended, while candidate cell arrays have not yet replaced the reconciled geometry.
Passing `EffectiveGeometry` would therefore describe the wrong state and would couple validation
to the backend's private owner.

Tests are the other callers. They deliberately construct raw malformed arrays to pin exact failure
ordering, compare the effective gate with the independent diagram gate, and check terminal
output-resolution results. These are contract fixtures, not alternate production owners.

## Cell-scan use of the pair

`verify_sphere_effective_strict` uses the cell/index pair in only three ways:

1. `cells.len()` determines generator cardinality, parallel chunking, signature capacity,
   connectivity, and Euler face count.
2. `cell_indices.len()` bounds incidence work and reserves the final directed-edge vector. The
   capacity deliberately includes stale backing slots, matching the current allocation choice.
3. `scan_cells_strict` obtains one checked live span per cell, then performs the ranked per-cell
   checks and emits signatures and edge uses from that span.

The parallel closure copies only slice references today. `LiveCellLayout` is likewise a copyable
pair of shared slices, so carrying it into each chunk adds no allocation, locking, or ownership
change.

## Semantic constraints

The effective gate is intentionally not a trusted-layout reader. Unlike topology summary and the
local-rebuild overlay, it must reject malformed raw arrays in release mode with the exact
`"invalid cell span"` reason at rank zero. Its current lookup uses `checked_add` before
`slice.get(start..end)`, so an overflowing `usize` end is rejected rather than panicking.

`LiveCellLayout::checked_span` currently distinguishes an invalid cell id from an out-of-buffer
span, but forms `start + count` directly. The addition cannot overflow on the measured 64-bit
target because the stored fields are only `u32` plus `u16`; it can overflow a 32-bit `usize`.
Routing validation through that method without first closing this difference would weaken the
portable malformed-input contract.

The migration must also preserve:

- generator/cell mismatch as the first input error;
- off-sphere vertices before all per-cell errors;
- lexicographic `(cell, check_rank)` selection across parallel chunks;
- the rank order span → vertex/duplicate → degeneracy → duplicate cell;
- signature and edge emission points;
- stale-tail exclusion with backing-length capacity reservation;
- every static failure string; and
- whole-diagram validation before the local-rebuild transaction commits.

Do not call `debug_assert_valid` at this boundary. Malformed spans are supported verifier inputs,
not internal assertions, and must return an error in checked and release builds alike.

## Relationship to QUAL-001C

QUAL-001C established that the fast diagram gate, effective-array gate, and accumulating public
report intentionally have different storage assumptions, stopping policies, ordering, and weld
semantics. A shared strict-error enum and a shared weld predicate both reproduced the clean-path
optimizer cliff and were reverted.

This QUAL-001B slice changes none of those facts. It affects only the effective gate's cell/index
input representation and its existing private parallel scan. `verify_sphere_fast` continues to
iterate `SphericalVoronoi` cells sequentially, and `validate_impl` continues its accumulating
diagram traversal. `EdgeUse`, `EdgeUseClass`, sorting, connectivity, and reason strings remain
unchanged.

## Rejected shapes

- Do not pass `EffectiveGeometry`: the candidate cells are intentionally not installed in that
  owner until the gate succeeds, and `validation.rs` should not depend on backend orchestration.
- Do not wrap generators or vertices with the layout: their independent cardinality and
  transaction lifetimes are meaningful.
- Do not construct a temporary `SphericalVoronoi`: cloning the candidate arrays was the dominant
  cost that the in-place effective gate removed.
- Do not trust the layout with a debug-only audit: release-mode malformed-span rejection is part of
  the gate's contract.
- Do not extract a common scan with the other validators: QUAL-001C already measured and closed
  that direction.
- Do not migrate the assembly handoff in the same change; it is the final, separately measured
  QUAL-001B stage.

## Selected migration

Make the checked layout operation portable before using it here:

1. Add a typed `CellSpanError` outcome for `start + count` overflow.
2. Make `LiveCellLayout::checked_span` and its checked-build structural audit use checked addition.
3. Map the new outcome to reconciliation's existing controlled-state error without changing valid
   traversal; add a 32-bit-only regression for the representable overflowing record.

Then replace the effective verifier's `cells`/`cell_indices` parameters with one
`LiveCellLayout`:

- derive cell and backing-index counts through existing accessors;
- pass the copyable layout into `scan_cells_strict`;
- map every checked-span error to the existing rank-zero `"invalid cell span"` result;
- use the successful span length for the existing stack/spill and edge loops; and
- construct the view explicitly at the local-rebuild candidate gate and raw-array tests.

The trusted direct `span` and `span_for` accessors remain unchanged. This keeps their accepted
release expressions separate from the malformed-input path.

## Acceptance gate

Focused tests must pin the existing valid/malformed layout errors, the effective gate's raw span
reason, every fast/effective differential reason, output-resolution terminal validation, and
accepted/rejected local-rebuild transactions. Run the complete release, checked,
no-default-feature, and all-feature Clippy gates.

Build the candidate and parent together and run the usual seven 500k single-threaded Fibonacci
instruction/branch pairs even though the production call is cold; earlier nominally cold signature
changes perturbed clean-path codegen. Compare release section and file sizes. An active-gate counter
run is required only if a deterministic workload actually enters local rebuilding; the current
100k mega fixtures do not. Any repeatable clean-path regression rejects the migration or requires
a smaller seam.
