# TODO

## Heuristics

- Revisit packed `r=2` expansion activation heuristics.
  The current design is intentionally simple: expansion is enabled by default and triggers whenever a query exhausts packed `r=1` without proving termination. Bench results suggest this is worthwhile on rougher distributions, but it can be slightly negative on smoother `--lloyd`-style inputs. If heuristic gating is added later, keep it in the neighbor-source layer rather than spreading policy across `process_cell`.

- Consolidate other heuristic decisions in one place.
  There are now multiple heuristic knobs across packed kNN, termination cadence, and fallback behavior. Future work should track these together rather than adding one-off local rules.

- Add explicit policy profiling baselines for common benchmark distributions.
  The code now has a policy layer and timing counters, but we do not yet keep a small documented
  matrix of expected structural outcomes across `fibonacci+jitter`, `--lloyd`, and any future
  adversarial distributions. That would make future heuristic changes easier to review.

- Consider policy-level tests for timing-surface stability.
  Not benchmark assertions in CI, but small tests that pin which counters/stages are expected to be
  exposed when policy changes are made intentionally.

- Benchmark the current frontier-certificate stream contract before further query-path cleanup.
  The stream now exposes "exact batch / conservative bound / exhausted" directly, and cell
  construction terminates against that contract. Future work here should be driven by measured
  regressions or simplification wins, not by another abstraction pass for its own sake.

## Reconciliation

- Consider moving some current `edge_reconcile` reconciliation into the live dedup/assembly path.
  The current post-pass is intentionally narrow and handles unresolved shared-edge mismatches after
  assembly. A future design pass can explore doing some of that agreement earlier, but only if the
  extra hot-path complexity remains clearly justified.

## Correctness

- Tighten behavior for cells that extend beyond the generator hemisphere.
  The current clipping path relies on a gnomonic projection centered at the generator, so cells that
  extend past 90 degrees are outside the model. The explicit failure boundary now exists; the
  remaining work is to decide whether to keep that as a clean error only or add a fallback
  projection/model that can represent the cell.

- Continue narrowing ambiguous terminal failures where repro evidence exists.
  The broad supported-failure taxonomy is in place, but cases like
  `UnboundedAfterExhaustion` remain deliberately conservative. If a reproducible family appears,
  tighten it from `ComputationFailed` into a sharper classification rather than leaving it vague.

- Audit remaining panic-only invariant paths now that the phase/query boundaries are cleaner.
  The main current hotspots are:
  - `topo2d/builder.rs`
  - `edge_reconcile.rs`
  - any residual unchecked layout/index assumptions in live-dedup assembly
  The goal is to keep true bug-only invariants as panics, but make diagnostics sharper and avoid
  leaving obviously user-reachable states in the panic bucket by accident.
