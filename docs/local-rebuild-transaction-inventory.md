# Local-Rebuild Transaction Inventory

**Status:** one QUAL-001D ownership/extraction slice selected, 2026-07-20

This inventory covers the local-rebuild phase after the accepted reconciliation orchestration
boundaries. It selects one representation-preserving cleanup; trigger policy, oracle algorithms,
numeric policy, and the grow loop remain unchanged.

## Boundary today

`compute::maybe_rebuild_effective` owns three different programs:

1. **Entry policy:** derive low-incidence and Euler facts, honor the probe-only A0 interception,
   return when rebuilding is disabled, normalize reconciliation evidence into defect pairs, and
   return when no rebuild trigger exists.
2. **Overlay growth:** read diagnostic knobs only after a real trigger, create grid scratch and a
   borrowed `WorkingDiagram`, select the configured oracle, and grow/splice until clean or stuck.
3. **Candidate transaction:** reject a zero-splice overlay, materialize and canonicalize the
   proposed arrays, append minted positions, run the strict whole-diagram gate, then either truncate
   the appended positions or commit the replacement cell arrays.

The ordinary enabled path returns during the first program. It performs no diagnostic environment
read, scratch allocation, overlay construction, flattening, or strict validation. The disabled
path also returns before defect-pair allocation. Those orderings are performance and behavior
contracts, not incidental control flow.

## Ownership and mutation contract

`WorkingDiagram` borrows the reconciled base positions and `LiveCellLayout` immutably. All growth
mutations live in its overlay maps and minted-vertex side arrays. After a productive grow pass,
four values describe one proposed replacement:

- minted vertex positions, whose ids start at the current base vertex count;
- one complete replacement `Vec<VoronoiCell>`;
- its paired replacement `Vec<u32>` index buffer; and
- the sorted cells whose final cycles were overridden, used by terminal exact-zero discovery.

These values are currently independent locals even though none is meaningful as a committed
result without the others. The replacement cell/index arrays are not installed until strict
validation succeeds. Minted positions are appended before validation so the gate can read them;
rejection truncates the position vector to its exact prior length while leaving the original cell
and index allocations untouched. Acceptance retains the append and swaps both replacement arrays.

The borrow boundary is load-bearing. A helper cannot safely receive both `WorkingDiagram` and
`&mut EffectiveGeometry`: the overlay still borrows the geometry. Consuming the overlay into an
owned candidate must happen before the mutable geometry transaction begins.

## Selected extraction

Add a private `LocalRebuildCandidate` in `compute.rs` owning the four correlated values above.

1. `LocalRebuildCandidate::from_work` consumes `WorkingDiagram`, captures the sorted overridden
   cells, calls the existing `into_flat`, and canonicalizes replacement cycle starts before the
   strict gate.
2. A consuming `try_commit` method receives the effective generators, mutable
   `EffectiveGeometry`, the existing debug flag, and the already-started materialization timer.
   It preserves the current append, timing, whole-diagram validation, diagnostic printing,
   truncate-on-rejection, and two-array commit order. It returns the resolution-scan cells only on
   acceptance.
3. `maybe_rebuild_effective` retains every entry gate, defect-pair normalization, environment-read
   boundary, scratch/overlay construction, oracle dispatch, and zero-splice rejection. After a
   productive overlay it prepares one candidate and maps `try_commit`'s result to the existing
   `Rejected` or `Accepted` outcome.

Use default compiler inlining for the first candidate. Add no `inline`, `cold`, or optimization
attribute unless matched artifact evidence justifies one isolated follow-up variant.

This is an ownership boundary rather than a new geometry representation. The candidate owns only
new material; it does not copy, move out, or wrap the reconciled base position allocation.

## Shapes deliberately excluded

- Do not extract the entire active attempt behind a broad borrowed context. Effective points,
  grid, provenance, mutable geometry, evidence, topology facts, policy, and diagnostics have
  different lifetimes and mutation roles.
- Do not move trigger policy or public `LocalRebuildMode` interpretation into `local_rebuild.rs`.
  That module owns overlay/oracle mechanics; `compute.rs` owns pipeline policy and commit validity.
- Do not make `WorkingDiagram` own or mutably borrow `EffectiveGeometry`. Its cheap borrowed overlay
  and delayed transaction are source-pinned performance decisions.
- Do not clone the base vertices or build a complete candidate vertex vector. Preserve the
  append/validate/truncate strategy.
- Do not replace the whole-diagram strict gate with a local validation. Minted triple identities
  can affect cells outside the immediate splice footprint.
- Do not change `run_rebuild_growth`, oracle dispatch, residual localization, winding parity,
  gather policy, or `LocalRebuildStats` in the same slice.
- Do not add generalized rollback or panic guards. The selected change names the existing explicit
  transaction; broader failure-atomicity semantics are a separate design decision.
- Do not move diagnostic environment reads above the trigger or change the printed timing spans.

## Semantic gate

The extraction must preserve:

- A0 diagnostic capture before ordinary mode/trigger handling;
- disabled, not-triggered, rejected, and accepted status distinctions;
- independent pre-rebuild low-incidence and Euler facts;
- defect-pair normalization and deduplication;
- zero diagnostic reads and allocations on the ordinary return;
- exact Hull3d/projected/probe oracle selection and growth statistics;
- zero-splice rejection without touching `EffectiveGeometry`;
- cycle-start canonicalization before the strict gate;
- append-only minted vertex ids during validation;
- exact base-position length restoration and unchanged cell/index allocations on rejection;
- paired cell/index replacement and the complete override footprint on acceptance; and
- whole-effective-diagram strict validation before any accepted return.

Add direct transaction coverage for accepted and rejected candidates if existing integration
fixtures do not enter the candidate path deterministically. Run focused local-rebuild,
local-rebuild-contract, reconciliation, output-resolution, and validation coverage, then the
complete release, checked, no-default-feature, and all-feature Clippy gates.

## Performance gate

Compare immediate-parent and dirty-candidate release artifacts, including the emitted
`maybe_rebuild_effective`/candidate symbol shape. Run seven interleaved 500k single-threaded
Fibonacci counter pairs to detect whole-binary layout movement on the ordinary no-trigger path.
Retired instructions and branches remain primary; reject repeatable clean-path loss.

There is currently no honest natural active-path counter fixture. On exact `4d15619`, twenty 100k
single-threaded `mega` runs at fraction `0.8` (seeds 1 through 20, preprocessing disabled) emitted
no local-rebuild trigger. The existing production tests likewise note that their historical mega
fixtures now resolve before rebuilding. Do not label these workloads active or infer transaction
performance from their total time.

If a deterministic accepted/rejected end-to-end trigger is found, add paired counters for it. If
none is found, acceptance requires direct transaction semantics plus artifact evidence that the
default-inline extraction did not introduce an out-of-line active-path boundary or materially
change the surrounding emitted shape. Otherwise defer the source change until an active fixture or
more faithful probe exists. Wall clock remains advisory on the busy shared host.
