# Assembly Handoff Layout Boundary Inventory

**Status:** closed without a production migration, 2026-07-20

This inventory covers the fifth and final QUAL-001B migration stage: the cell-layout handoff from
sharded live-dedup assembly into the effective-geometry pipeline. It does not reopen the measured
reconciliation, effective-validation, or mutable-layout signatures.

## Producer and consumer

`live_dedup::assemble_sharded_live_dedup` is the only production producer of `AssemblyResult`.
After resolving deferred references and concatenating vertex storage, it creates the global cell
layout in three ordered steps:

1. A checked u32 prefix sum creates one `VoronoiCell` per effective generator.
2. A direct scatter fills every slot in the eventual exact-length `cell_indices` buffer, followed
   by sparse foreign reference overrides.
3. Exact-zero hint discovery reads the completed cell windows before the two vectors are returned.

The assembly layout is stronger than the general live-layout contract. Every cell window is
contiguous in generator order, the windows partition the entire backing buffer, and no stale tail
slots exist. Reconciliation may later shrink live windows and introduce the stale-tail state that
`LiveCellLayout` exists to make explicit.

`run_core_pipeline` is the only production consumer. It immediately destructures `AssemblyResult`
and moves `vertices`, `cells`, and `cell_indices` into `EffectiveGeometry`; no branch, fallible
operation, mutation, or independent consumer occurs between those operations. `EffectiveGeometry`
then remains the coherent owner through reconciliation, local rebuilding, output resolution,
report cloning, and final remapping.

Assembly unit tests inspect `cells` and `cell_indices` directly, but they do not create another
runtime ownership boundary.

## Candidate shapes considered

### Nest an owned cell layout in `AssemblyResult`

An `OwnedCellLayout { cells, indices }` (or assembly-specific equivalent) would prevent those two
fields from being independently replaced while the result exists. In production that lifetime is
only the immediate move into `EffectiveGeometry`, after which the wrapper would either be unpacked
again or propagated through the entire post-assembly pipeline.

Unpacking immediately adds a type and conversion without protecting a consumer. Propagating it
would duplicate or replace `EffectiveGeometry` ownership, require mutable accessors throughout
reconciliation and output resolution, and reopen signatures whose narrower paired forms already
failed the release counter gate.

### Return `EffectiveGeometry` from live dedup

`EffectiveGeometry` owns post-assembly lifecycle semantics, including later mutation and minted
vertices. Moving it into `live_dedup` would make the assembly module own downstream orchestration
policy. A second assembly-specific geometry record followed by conversion would merely move the
same three vectors between two nominal wrappers.

### Borrow `LiveCellLayout` inside assembly

The scatter and sparse override phases mutate the backing buffer, so a persistent read-only view
cannot span them. Constructing a view only for exact-zero hint discovery would replace one local
trusted span expression but would not protect the return handoff. It would also put a measured
codegen-sensitive accessor into the hot assembly function for no new invariant.

## Decision

Close the assembly-handoff stage without a production change. The current boundary already has:

- one producer and one immediate consumer;
- checked prefix construction and exact allocation coverage;
- debug sentinels and range/coverage assertions around unsafe direct fill;
- a stronger freshly compacted invariant inside assembly; and
- one accepted `EffectiveGeometry` owner as soon as the data acquires a multi-phase lifetime.

An additional owner would not make a sustained invalid state unrepresentable. It would instead
overlap the existing geometry owner or add a wrapper/unwrapper pair at a hot ABI. Avoiding that
change is the Pareto-frontier outcome, not deferred implementation work.

No runtime benchmark is warranted because no production candidate was selected. Documentation
formatting and link checks are the gate for this inventory.

## Retry condition

Revisit only if assembly gains a second consumer, cell records and their backing indices remain
separate across a meaningful lifetime, or a later phase extraction creates a natural owned layout
that both assembly and effective geometry can use without conversion. Any such change is an
assembly-sensitive migration and must compare ordered Fibonacci and scrambled uniform workloads
at default and high bin counts, using retired instructions and branches as the primary signal on
the shared host.
