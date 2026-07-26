# July 2026 code-quality boundary closeout

**Status:** consolidated closeout, 2026-07-27

This document replaces the ten closed `*-inventory.md` worksheets from QUAL-001. It preserves the
decision needed to maintain each boundary, the decisive validation or performance evidence, and
the condition for reopening it. The detailed source-shaped worksheets remain available in Git
history through revision `f98bf6e`.

[`code-quality-plan.md`](code-quality-plan.md) records the full program and
[`../work-log.md`](../work-log.md) remains the authoritative queue.

## Global decision

QUAL-001 stopped at selective ownership and phase boundaries rather than optimizing for short
functions or uniform signatures. Cold correlated state gained named owners. Read-only layout and
validation facts were shared only where semantics matched. Hot assembly, reconciliation, and packed
query loops remained flat when extraction changed code generation or lacked an ownership seam.

Reopen a closed boundary only for a new consumer or ownership invariant, a correctness defect, or
materially changed compiler/codegen evidence. Function length alone is not a reopening condition.

## Lifecycle and effective-state ownership

**Decision.** Four correlated-state migrations were accepted:

- `LocalRebuildStatus` represents not-triggered, disabled, rejected, and accepted outcomes while
  low-incidence and Euler observations remain independent defect facts.
- `ResolutionDiscoveryMode` replaces inverse certified-hint/drift-fallback booleans.
- `EffectiveInput` owns either identity input or a complete `MergeResult`; representative points no
  longer travel beside independently optional merge metadata.
- `EffectiveGeometry` owns positions, cell spans, and the live index buffer from assembly through
  reconciliation, rebuilding, output resolution, and remapping.

Assembly vertex keys remain separate, intentionally partial provenance. Residual records, mutation
footprints, and historical action status also retain their independent lifetimes. Accepted local
rebuilding keeps its append-and-rollback position strategy rather than copying base positions.

**Evidence.** All four migrations passed focused contracts and complete validation. Seven paired
release counter runs were neutral for each migration. The effective-input artifact kept aggregate
size unchanged and reduced file size by 664 bytes; the effective-geometry migration reduced file
size by 1,040 bytes. Public and machine-readable attempted/accepted semantics remained unchanged.

**Reopen when.** Revisit only if a new stage needs shared ownership, assembly provenance becomes
complete geometry state, or the local-rebuild transaction changes its append/rollback contract.

## Live cell layout

**Decision.** `LiveCellLayout` pairs cell records with their backing index buffer at selected
read-only seams: topology summaries, shared-edge segment reads, localized duplicate-key traversal,
semantic old/new span comparison, localized unpaired-edge scanning, and the local-rebuild overlay.
Checked construction audits capacity and declared spans where the defect path already pays for it.

The explicit cell-bound/end-bound check followed by ordinary slicing is retained. A mutable layout
owner for collinear rewrite remains rejected; the rewrite stays visible at its call site.

**Evidence.** The overlay migration reduced the executable by 48 bytes and seven 500k Fibonacci
pairs were neutral (`1.000000228` instructions, `1.000001841` branches). Earlier semantic-comparison
and unpaired-reader forms reproduced a roughly `+0.16%` instruction / `+1.66%` branch displacement;
controlled retests after surrounding code changed showed neutral cycles and a byte-identical
one-codegen-unit control, identifying codegen partitioning rather than added work. Replacing the
explicit checked-span sequence with `slice.get(start..end)` added `0.1337%` instructions and
`1.6620%` branches and remains rejected.

**Reopen when.** Add another reader only when it benefits from enforced cell/index pairing. Revisit
mutable ownership only if mutation becomes a reusable operation with a single invariant and neutral
codegen evidence.

## Effective-validation layout

**Decision.** The whole-effective-diagram strict gate carries `LiveCellLayout` through its private
parallel cell scan. Generators retain an independent cardinality check, and temporarily appended
vertices retain their transaction lifetime. Traversal policy, allocation, ordering, and typed
failure reasons remain distinct from the fast and public-report validators.

**Evidence.** The original migration repeatedly produced about `+0.129%` instructions and
`+1.36%` branches, even when narrowed to the private scan or to overflow-safe checked access, so it
was reverted. A controlled 2026-07-20 retest after surrounding codegen changes added only
`0.012%..=0.031%` instructions across four regimes, with neutral branches, no adverse cycle signal,
and a neutral one-codegen-unit control. Full release, checked, feature, and transaction suites
passed before retaining the boundary.

**Reopen when.** Do not merge this traversal with other validators unless their input, failure,
allocation, and ordering policies converge. Reassess only after material compiler or surrounding
codegen changes.

## Assembly handoff

**Decision.** No production migration was made. `AssemblyResult` is produced once and immediately
moved into `EffectiveGeometry`; nesting another owned layout would only add a wrapper/unwrapper, and
returning `EffectiveGeometry` directly would couple live dedup to a later pipeline owner.

**Evidence.** Producer/consumer and lifetime analysis found no second consumer or shared operation,
so no benchmark candidate was justified.

**Reopen when.** Revisit only if assembly gains another consumer or cell records and indices acquire
a natural shared lifetime before `EffectiveGeometry`.

## Shared validation facts

**Decision.** Validation shares edge-use classification and typed strict reasons while retaining
three consumers: allocation-conscious fail-fast verification, effective-array acceptance, and
accumulating public diagnostics. Their cell/edge traversals, weld handling, ordering, and reporting
remain separate.

**Evidence.** `EdgeUseClass` added 12 text bytes, removed 16 BSS bytes, and was neutral across seven
500k pairs. `StrictValidationIssue` and differential negative controls were neutral. A shared weld
predicate reproduced `+0.1604%` instructions and `+1.6618%` branches and was reverted. The fast gate
must not allocate a `ValidationReport`, maps, strings, or stored-position telemetry on success.

**Reopen when.** Share another fact only when semantics and traversal policy are identical and the
fast success path remains allocation-conscious.

## Reconciliation orchestration

**Decision.** `ReconcileRunState` owns the merge ledger, rejected-component rebuild seeds,
merge-affected cells, mutation scan cells, and merge-safety counters across primary and synthesized
backstop rounds. The empty-record return remains before state construction; pass order, raw geometry
arguments, and residual ordering remain flat and explicit.

**Evidence.** Complete validation passed. Seven 500k Fibonacci, seven active 100k `cubed`, and five
500k `cubed` counter sets were neutral. The artifact removed 544 text bytes, 3,552 alignment-counted
BSS bytes, and 616 file bytes; all measured samples had zero context switches and migrations.

**Reopen when.** Extend the owner only for state genuinely shared across reconciliation rounds. Do
not bury the clean return or pass sequence in a general transaction object.

## Reconciliation defect body

**Decision.** `reconcile_edge_mismatches` retains the empty-record gate and checked structural audit,
then delegates the nonempty program to a private helper with the same explicit inputs. No universal
context bundle or shared clean/defect path was introduced.

**Evidence.** LLVM inlined the helper and retained the primary function sizes. Seven clean
Fibonacci and seven active `cubed` pairs were neutral; the artifact added 8 text bytes and 24 file
bytes while removing 16 alignment-counted BSS bytes.

**Reopen when.** Extract further only around a new ownership boundary or independently reusable
defect operation, with active-defect and clean-path evidence.

## Local-rebuild transaction

**Decision.** `LocalRebuildCandidate` owns minted positions, replacement cell/index arrays, and the
sorted mutation footprint after a productive overlay. Its consuming commit owns append, strict
whole-diagram validation, diagnostics, rollback, and paired swap. Trigger policy, oracle selection,
and growth remain in `maybe_rebuild_effective`.

**Evidence.** Direct accepted and rejected tests pin paired commit and exact rollback. LLVM emitted
no standalone boundary. Seven 500k Fibonacci pairs and seven seed-224 productive-rejection pairs
were neutral; actual BSS stayed unchanged. The accepted source boundary added 1,308 text bytes and
992 file bytes without measurable runtime work.

**Reopen when.** Move more logic only if trigger/growth state gains its own transaction invariant.
Do not combine policy selection with candidate commit merely to shorten the caller.

## Live assembly phase

**Decision.** `ConfirmedZeroEdgeHints` and `confirm_exact_zero_edge_hints` own the final read-only
post-patch exact-zero scan and its correlated candidate/count result. Mutable shard repair, vertex
and cell materialization, unsafe scatter, sparse overrides, timing, and output policy remain in
their current performance-shaped order.

**Evidence.** The helper fully inlined, reduced the assembly body by 50 bytes, reduced aggregate text
by 48 bytes, and passed all semantic suites. Default/high-bin Fibonacci and uniform plus clustered
seed 1 were neutral; mean instruction ratios ranged `0.99999881..=1.00000211` and branch ratios
`0.99999471..=1.00000062`.

**Reopen when.** Revisit materialization or scatter only when a new owner/consumer creates a narrower
seam. Existing timer boundaries alone are not extraction boundaries.

## Packed group preparation

**Decision.** `collect_directed_ranges` and its four-field summary own center-plus-neighbor ordering,
same-bin eligibility, and hard/aggregate work gates. Scratch reset, threshold selection, dense
takeover, and center/ring SIMD kernels remain flat. The retained source-shaped extraction preserves
the later center-range read.

**Evidence.** The compact first form added `0.1397%` instructions on clustered input and was rejected.
The retained form adds about `0.009%..=0.010%` instructions while removing about
`0.004%..=0.006%` branches on ordinary inputs and is neutral on clustered and mega. It adds 64 text
bytes and 88 file bytes; direct invariants and complete validation passed.

**Reopen when.** Revisit the compact form only with changed codegen evidence. Extract SIMD or dense
paths only when they acquire independent state or a second consumer, not to reduce function length.
