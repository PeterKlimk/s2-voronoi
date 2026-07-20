# Effective Validation Layout Boundary Inventory

**Status:** migration accepted after controlled layout retest, 2026-07-20

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

## Measured result

The original production changes were reverted. Three forms were measured against parent `7c70983`; every
seven-pair matrix reproduced the same clean-path optimizer cliff despite the effective gate being
cold on the Fibonacci workload.

### Full boundary

The selected form passed `LiveCellLayout` from the local-rebuild candidate transaction through
`verify_sphere_effective_strict` and its private parallel scan. It also made checked span addition
overflow-safe as specified above.

- Mean candidate/parent ratios were `1.001290682` instructions and `1.013601256` branches.
- Pair ranges were `1.001285718..=1.001295949` and `1.013587754..=1.013614888`; every pair
  regressed.
- The release artifact added 332 text bytes, removed 24 data bytes and 312 BSS bytes, reduced
  aggregate accounting by four bytes, and grew the file by 848 bytes.

### Internal-only layout

The outer verifier ABI was restored to the original four slices. The verifier constructed one
layout internally and passed it only through `scan_cells_strict`.

- Mean ratios were `1.001293695` instructions and `1.013603930` branches.
- Pair ranges were `1.001289474..=1.001299176` and `1.013599255..=1.013611676`; every pair
  regressed.
- The artifact added 356 text bytes, removed 24 data bytes and 328 BSS bytes, grew aggregate
  accounting by four bytes, and grew the file by 928 bytes.

This rules out the outer call signature as the cause. Routing the effective scan through the
checked layout accessor is sufficient to perturb ordinary codegen.

### Overflow-safe checked accessor only

The validator and scan were restored completely. The remaining candidate only added the typed
span-end-overflow outcome, checked addition in `checked_span`/the checked-build audit, controlled
reconciliation error mapping, and 32-bit-only coverage.

- Mean ratios were `1.001294912` instructions and `1.013604706` branches.
- Pair ranges were `1.001284480..=1.001299841` and `1.013583502..=1.013616981`; every pair
  regressed.
- The artifact removed 364 text bytes and 3,744 BSS bytes, reduced aggregate accounting by 4,108
  bytes, and grew the file by 368 bytes—the characteristic section-layout fingerprint seen in
  earlier rejected layout experiments.

All samples across all forms recorded zero context switches and CPU migrations. Wall clock was
ignored on the busy host.

The original effective validator already retains portable `checked_add` malformed-span rejection.
`LiveCellLayout::checked_span` remains appropriate only for its current internal layouts, whose
u32/u16 span arithmetic is representable on the measured 64-bit target; it must not replace the
effective validator's raw checked expression. Revisit only after a material compiler or surrounding
codegen change. The final QUAL-001B assembly-handoff stage remains independent and may still be
inventoried.

## Controlled retest

The full selected migration was retested on 2026-07-20 against parent `b8bd22e` after the semantic
comparison and unpaired-reader boundaries changed the surrounding codegen. It is now retained.

- `verify_sphere_effective_strict` receives generator and vertex slices plus one
  `LiveCellLayout`. Its parallel chunks copy that view and obtain every span through
  `checked_span`; generator cardinality, temporarily appended vertices, ranked error ordering,
  allocation points, and static reasons remain independent and unchanged.
- `CellSpanError::SpanEndOverflow` and checked span-end addition preserve the former raw
  validator's malformed-input behavior on 32-bit targets. Reconciliation maps the new outcome to
  its existing controlled-state error; trusted direct accessors remain unchanged.
- Default-build seven-pair instruction/branch means were `1.000296110` / `0.999999107` on 500k
  Fibonacci, `1.000310428` / `1.000001768` on 500k uniform seed 12345, `1.000176884` /
  `0.999996530` on 100k clustered seed 1, and `1.000119413` / `0.999999036` on 100k mega seed 1.
  Cycles showed no adverse signal; every sample recorded zero context switches/migrations.
- A `-C codegen-units=1` control reduced the Fibonacci displacement to `1.000002647`
  instructions and `1.000002939` branches, isolating the small default instruction increase to
  codegen partitioning. Under the retained default build, text fell 44 bytes, data fell 24 bytes,
  BSS grew 72 bytes, aggregate accounting grew four bytes, and file size grew 832 bytes.

The current boundary therefore supplies a portable malformed-layout contract and prevents the
candidate cell records/backing indices from being independently paired, without a measured
throughput regression.
