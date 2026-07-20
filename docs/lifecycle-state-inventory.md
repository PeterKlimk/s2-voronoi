# Lifecycle State Inventory

**Status:** archived pre-migration inventory; superseded by the final 2026-07-20 pruning pass

Names and reachability statements below describe the audited pre-migration tree, not the current
API. The resulting model is recorded in [`code-quality-plan.md`](code-quality-plan.md).

This inventory identifies the first correlated cold state to replace with an enum. It records the
current behavior before changing representation; geometry, trigger policy, validation, and report
semantics are out of scope for the migration.

## Local-rebuild action state

`knn_clipping::compute::LocalRebuildOutcome` and the public `LocalRebuildReport` both carry
`attempted` and `accepted` booleans. Production constructs only three combinations:

| `attempted` | `accepted` | Meaning | Reachable |
|---|---|---|---|
| false | false | Rebuild did not run | Yes |
| true | false | Rebuild ran but committed nothing | Yes |
| true | true | Rebuild ran, passed the strict gate, and committed | Yes |
| false | true | Accepted without an attempt | No; representable invalid state |

The reachable `false/false` state currently combines three materially different reasons:

1. the configured mode is enabled but no residual-pair or low-incidence trigger exists;
2. `LocalRebuildMode::Disabled` prevents an attempt, even if defect facts exist; and
3. the `local_rebuild_probe` A0 capture intercepts the assembled state before ordinary trigger
   evaluation.

An enabled run with an Euler-only defect also takes the ordinary no-trigger path. Euler is an
independent fail-loud fact, not a rebuild trigger, so the status name must describe whether the
stage ran rather than claiming the full output needed no repair.

## Independent defect facts

`low_incidence_defect` and `euler_defect` are not action states. They describe the pre-rebuild
assembled topology and remain necessary when rebuilding is disabled, not triggered, or rejected.
An accepted rebuild supersedes them because the committed diagram already passed whole-diagram
strict validation.

Keep these facts as separate booleans in the internal outcome. Do not encode combinations of them
as status variants and do not infer them from the rebuild action.

## Consumers and compatibility

- `check_plain_return_signals` needs only “accepted” plus the two defect facts.
- report assembly clears residual records only for an accepted commit.
- `ComputeReport::has_output_residuals` treats a rejected attempt as a residual signal; disabled or
  no-trigger cases are covered by residual records and validation facts instead.
- `bench_voronoi`, the fidelity and robustness campaigns, and tests print or inspect the two public
  booleans. The machine-readable field names `local_rebuild_attempted` and
  `local_rebuild_accepted` are active repository interfaces and must retain their values.
- No repository code constructs `LocalRebuildReport` outside the backend. There are no external
  users, so the public Rust field migration can be atomic without deprecated compatibility fields.

## Implemented first enum boundary

The public, non-exhaustive `LocalRebuildStatus` contains only user-meaningful outcomes:

- `NotTriggered` — ordinary enabled pipeline did not run the stage;
- `Disabled` — policy prevented the stage from running;
- `Rejected` — the stage ran but committed nothing, including the zero-splice and failed-gate paths;
  and
- `Accepted` — the strict gate passed and the rebuilt state committed.

Private `LocalRebuildExecution` distinguishes an ordinary completed status from the feature-gated
A0 diagnostic interception. Probe callers consume the existing capture side channel and discard
the compute result. If their callback nevertheless completes report construction, the conversion
boundary reports `NotTriggered`; the public status taxonomy does not expose repository test
control.

`LocalRebuildReport` contains the status and exposes derived `attempted()` and `accepted()` methods.
Repository consumers migrated atomically to those methods while retaining the KV field names and
values. No public compatibility booleans remain, so the invalid false/true combination is no longer
representable.

The public status truth table, ordinary clean/disabled paths, and feature-gated capture conversion
are pinned directly. Existing local-rebuild contract and fault-injection tests continue to own
accepted/rejected and fail-loud behavior. Seven release counter pairs for the original enum
migration were neutral; the attempted/accepted KV semantics remain unchanged.

## Resolution discovery state

The former `ResolutionDiscoveryDecision` stored `certified_hint` and `drift_fallback`, exact
inverses constructed from one drift boolean. Timing repeated both fields and accepted the impossible
equal-value combinations.

`ResolutionDiscoveryMode` now has exactly two states: `CertifiedHint` and
`ExhaustiveDriftFallback`. Candidate discovery branches on the enum. Timing retains only the
fallback bit and derives both existing `resolution_certified_hint` and `resolution_fallback_drift`
KV values, so their names and output are unchanged. The exhaustive-fallback and timing-finish tests
pin both behavior and telemetry.

The release artifact exchanged 16 BSS bytes for 16 text bytes without changing aggregate or file
size. Seven instruction/branch counter pairs were neutral.

## Effective-input ownership

`PipelineState::effective_points` and `PipelineState::merge_result` describe one preprocessing
decision but are stored as independent `Option`s. Production constructs only two combinations:

| `effective_points` | `merge_result` | Meaning | Reachable |
|---|---|---|---|
| `None` | `None` | Use the original points directly | Yes |
| `Some` | `Some` | Use welded representatives and retain the original-to-effective map | Yes |
| `Some` | `None` | Effective geometry without a way to remap it | No |
| `None` | `Some` | Remap metadata without its effective geometry | No |

Disabled preprocessing and a weld pass that finds no pairs both retain the identity state. In
particular, the steady pipeline must continue to borrow `PipelineState::points` in that state; the
cleanup must not retain a duplicate point vector or identity remap merely to simplify the type.
Only an actual merge enters the welded state.

The current split is sharper than the two fields suggest. `MergeResult` initially owns
`effective_points`, `original_to_effective`, and `num_merged`. `prepare_points_and_grid` moves the
point vector out with `mem::take`, stores it in the first `Option`, and stores the now-empty result
in the second. The invariant therefore depends on coordinated mutation as well as coordinated
presence.

### Consumers and lifetime

- grid compaction/rebuilding and all effective-cell construction use the representative points;
- construction error reporting uses the merge map to translate effective generator ids back to
  original ids;
- report mode clones the effective arrays into `effective_diagram` only after an actual merge;
- final assembly expands effective cells back to original generators and installs the weld map;
- output-resolution mesh tests need both the effective geometry and original-to-effective map; and
- coplanar perturbation retry creates a fresh core pipeline for each attempt, so no prepared-input
  state is shared across attempts.

`PreprocessReport::effective_points` and `num_merged` are observations of the same decision. They
must be derived from its owner rather than used as another source of truth.

### Implemented boundary

One cold orchestration enum, `EffectiveInput`, now has two variants:

```rust,ignore
enum EffectiveInput {
    Identity,
    Merged(MergeResult),
}
```

`MergeResult` retains its effective point vector. Methods on the enum select the original or
representative slice, expose optional merge metadata, and derive the effective length and merge
count. `PipelineState` owns this enum instead of the two `Option`s.

The four-element `PreparedPointsAndGrid` tuple is now a named phase record containing this enum,
the preprocessing report, and the grid. This keeps preparation ownership explicit without moving a
larger record through the per-cell hot path. Construction continues to receive the same point slice
and optional map it did before the migration.

A direct contract test pins all three policy outcomes—disabled, weld-with-no-merge, and actual
merge—including borrowed identity storage and report counts. Existing release API tests pin the
effective diagram, standalone large-threshold merge, error-index, and final-remap behavior.

The matched release artifact kept aggregate size unchanged, moved 656 bytes from text to BSS, and
reduced file size by 664 bytes. Seven interleaved counter pairs were neutral: mean
candidate/parent ratios were `0.999998159` instructions and `0.999998587` branches, with zero
context switches and migrations.

## Effective-geometry ownership

Three vectors formerly existed as separate state throughout the post-assembly pipeline:

- `vertices: Vec<Vec3>` owns the effective-space vertex positions;
- `eff_cells: Vec<VoronoiCell>` owns one live span descriptor per effective generator; and
- `eff_cell_indices: Vec<u32>` owns the flattened boundaries addressed by those spans.

Every consumer requires a coherent triple, but the former representation permitted cells from one
phase to be paired with positions or indices from another.

### Mutation sequence

1. Assembly creates all three arrays together. Its incidence summary is exact for these initial
   live spans.
2. Reconciliation mutates cell spans and vertex ids in the index buffer. Vertex positions remain
   allocated in place, and replaced ids may become unreferenced.
3. Local rebuilding overlays the reconciled arrays. A rejected candidate appends minted positions
   only temporarily, then truncates back to the exact base length without changing the live cell
   arrays. An accepted candidate keeps those appended positions and replaces the complete cell and
   index arrays together.
4. Output resolution may contract vertex ids in the live boundaries. It deliberately retains the
   position allocation, including any vertices made unreferenced by reconciliation, rebuilding, or
   contraction.
5. Report mode may clone the terminal effective arrays; final remapping consumes cells and indices
   while reusing the same positions for the returned diagram.

This is one mutable geometry object across phases, not separate assembled and rebuilt allocations.
Representing the phases as an enum would either duplicate the same storage shapes or pressure the
accepted local-rebuild path to copy all base positions, losing its current append-only behavior.

### State that must remain outside

`ShardedVertexKeys` is assembly provenance, not a fourth parallel geometry vector. It initially
describes assembly vertex ids, but local rebuilding can append minted positions without extending
the sharded key store. That partial provenance is intentional: local rebuilding resolves minted
keys inside its overlay, while terminal output-resolution localization treats a missing key as a
certificate failure and falls back conservatively. The corrected `live_dedup` ownership comment
now records all three consumers and the missing-provenance behavior.

Assembly mismatch records, reconciliation residuals, rejected-component seed pairs, and the
local-rebuild status are historical diagnostic facts. An accepted rebuild supersedes some of them
for fail-loud/output validity while report mode still retains assembly evidence. They must not be
inferred from the terminal geometry or absorbed into its owner.

Construction exact-zero candidates and reconciliation/rebuild scan-cell lists are mutation
certificates. They select which terminal cycles must be rescanned; they are neither terminal
geometry nor independently authoritative evidence about its contents. The construction incidence
summary similarly expires as soon as a later phase changes a live span.

### Implemented boundary

One private cold-orchestration record now owns the geometry:

```rust,ignore
struct EffectiveGeometry {
    vertices: Vec<Vec3>,
    cells: Vec<VoronoiCell>,
    cell_indices: Vec<u32>,
}
```

It is created immediately after assembly. Reconciliation mutates it and returns only its
diagnostic result; local rebuilding receives it as one mutable object; output resolution mutates
its live boundaries; and `PipelineState` owns it as one field. The
`ReconciledWithResiduals` tuple is gone, without a second wrapper duplicating the same ownership.

`assembly_vertex_keys` remains a clearly named, separately borrowed provenance store until output
resolution finishes. It is neither flattened nor extended merely to make lengths match.

The implementation is representation-only: no compaction, provenance expansion, candidate
ownership change, or local-rebuild transaction redesign. Existing focused suites cover accepted
commits, effective-report cloning, welded remapping, reconciliation, and output-resolution
footprints; the append/validate/truncate rejection sequence is unchanged within the new owner. The
complete release, checked, no-default-feature, and all-feature Clippy gates passed.

The matched release artifact removed 1,484 text bytes, added 1,488 BSS bytes, retained data size,
grew aggregate section accounting by four bytes, and reduced file size by 1,040 bytes. Seven
interleaved counter pairs were neutral: mean candidate/parent ratios were `1.000000038`
instructions and `0.999998878` branches, with zero context switches and migrations.
