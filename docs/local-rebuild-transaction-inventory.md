# Local-Rebuild Transaction Inventory

**Status:** selected QUAL-001D ownership/extraction slice accepted, 2026-07-20

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

A stable rejected-transaction fixture is now available: 100k `mega`, corner-centered cap, fraction
`0.8`, seed `224`, preprocessing disabled, and one Rayon thread. It deterministically performs two
growth rounds, splices seven generators, materializes the full candidate, and reaches the strict
gate before rejection and rollback. The search and exact fingerprint are recorded in
[`local-rebuild-trigger-hunt.md`](local-rebuild-trigger-hunt.md).

Use paired counters on that expected-error fixture with a harness that verifies completion of the
transaction before accepting the sample. No natural accepted transaction was found; pin the
accepted append/swap/footprint path with a direct deterministic transaction test. Artifact evidence
must still show that default inlining introduced no unexplained boundary. Wall clock remains
advisory on the busy shared host.

## Accepted result

`LocalRebuildCandidate` now owns the minted positions, complete replacement cell/index arrays, and
sorted override footprint produced by a productive `WorkingDiagram`. `from_work` consumes the
borrowed overlay and canonicalizes cycle starts before any mutable geometry borrow. Its consuming
`try_commit` method preserves the append, whole-diagram strict gate, debug timing and diagnostics,
truncate-on-rejection, and paired-array installation. `maybe_rebuild_effective` retains every
entry gate, environment-read boundary, oracle decision, growth call, and zero-splice return.

Two direct tests pin both sides of the transaction. The accepted fixture replaces every use of one
live vertex id with an appended equal-position vertex and verifies that the candidate's two array
allocations and complete footprint move into `EffectiveGeometry` together. The rejected fixture
appends a position, supplies an invalid candidate reference, and verifies exact base-position
restoration while the original cell and index allocations remain installed. Existing
accepted-rebuild integration fixtures also remain green.

Complete release, checked, no-default-feature, and all-target/all-feature Clippy gates passed.
LLVM emitted no standalone candidate or `maybe_rebuild_effective` body; the boundary remains folded
into `run_core_pipeline`, whose symbol grew from `0x4a16` to `0x4b96` bytes. GNU aggregate size
accounting added 1,308 text bytes and 2,784 alignment-accounted BSS bytes with unchanged data;
the file grew by 992 bytes. Actual `.bss` remained 291 bytes, and the aggregate BSS delta is wholly
the corresponding RELRO-padding shift rather than new mutable storage.

Seven interleaved 500k single-threaded Fibonacci pairs measured candidate/parent ratios of
`0.999999187` instructions and `0.999999792` branches (ranges
`0.999990973..=1.000007501` and `0.999985103..=1.000010080`). Seven interleaved seed-224 active
rejection pairs each verified the full two-round/seven-splice/strict-rejection fingerprint and
measured `0.999991525` instructions and `1.000000202` branches (ranges
`0.999987191..=0.999994591` and `0.999997381..=1.000003300`). Every sample recorded zero context
switches and CPU migrations. The default-inline source boundary is accepted; no optimization
attribute or quiet wall-clock run is warranted.
