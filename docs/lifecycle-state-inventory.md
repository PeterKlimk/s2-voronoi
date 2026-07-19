# Lifecycle State Inventory

**Status:** QUAL-001A state-model inventory, 2026-07-19

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

## Selected first enum boundary

Introduce one public, non-exhaustive `LocalRebuildStatus` and use it directly inside
`LocalRebuildOutcome`:

- `NotTriggered` — ordinary enabled pipeline did not run the stage;
- `Disabled` — policy prevented the stage from running;
- `Rejected` — the stage ran but committed nothing, including the zero-splice and failed-gate paths;
- `Accepted` — the strict gate passed and the rebuilt state committed; and
- a doc-hidden diagnostic-capture status for the feature-gated A0 interception path.

`LocalRebuildReport` should contain the status and expose derived `attempted()` and `accepted()`
methods. Repository consumers should migrate atomically to those methods while retaining the KV
field names and values. Do not keep public boolean fields: doing so would preserve the invalid
state the enum is intended to remove.

The first implementation must pin the status truth table and the ordinary clean/disabled paths.
The existing local-rebuild contract and fault-injection tests continue to own accepted/rejected and
fail-loud behavior. Since the status is carried through the main pipeline, acceptance requires the
usual release artifact comparison and hardware-counter gate.
