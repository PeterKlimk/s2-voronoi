# Lifecycle State Inventory

**Status:** QUAL-001A first two state-model migrations implemented; effective-input boundary
inventoried, 2026-07-19

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

One public, non-exhaustive `LocalRebuildStatus` now flows directly through
`LocalRebuildOutcome`:

- `NotTriggered` — ordinary enabled pipeline did not run the stage;
- `Disabled` — policy prevented the stage from running;
- `Rejected` — the stage ran but committed nothing, including the zero-splice and failed-gate paths;
- `Accepted` — the strict gate passed and the rebuilt state committed; and
- a doc-hidden diagnostic-capture status for the feature-gated A0 interception path.

`LocalRebuildReport` contains the status and exposes derived `attempted()` and `accepted()` methods.
Repository consumers migrated atomically to those methods while retaining the KV field names and
values. No public compatibility booleans remain, so the invalid false/true combination is no longer
representable.

The status truth table and ordinary clean/disabled paths are pinned directly. Existing
local-rebuild contract and fault-injection tests continue to own accepted/rejected and fail-loud
behavior. Seven release counter pairs were neutral; the attempted/accepted KV semantics are
unchanged.

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

### Selected boundary

Use one cold orchestration enum, provisionally `EffectiveInput`, with two variants:

```rust,ignore
enum EffectiveInput {
    Identity,
    Merged(MergeResult),
}
```

`MergeResult` will retain its effective point vector. Methods on the enum will select the original
or representative slice, expose optional merge metadata, and derive the effective length and merge
count. `PipelineState` will own this enum instead of the two `Option`s.

Replace the four-element `PreparedPointsAndGrid` tuple with a named phase record containing this
enum, the preprocessing report, and the grid. This keeps preparation ownership explicit without
moving a larger record through the per-cell hot path. Construction should continue to receive the
same point slice and optional map it does today.

The implementation gate must pin all three policy outcomes—disabled, weld-with-no-merge, and
actual merge—plus unchanged report, effective-diagram, error-index, and final-remap behavior.
Because representative selection feeds the full construction pipeline, release artifact size and
interleaved instruction/branch counters remain mandatory even though the new state is cold.
