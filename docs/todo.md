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

## Reconciliation

- Consider moving some current `edge_reconcile` reconciliation into the live dedup/assembly path.
  The current post-pass is intentionally narrow and handles unresolved shared-edge mismatches after
  assembly. A future design pass can explore doing some of that agreement earlier, but only if the
  extra hot-path complexity remains clearly justified.
