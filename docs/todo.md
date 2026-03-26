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

- Rework the neighbor-stream frontier API around "exact next or conservative bound".
  Termination is fundamentally based on the best possible unseen neighbor, not on the neighbor that
  was just clipped. Inside a materialized batch we often know the exact next neighbor, but across a
  stage boundary we may only know an upper bound until we expand `r=2` or enter directed cursor
  fallback. The stream should expose that directly so cell construction can terminate against the
  frontier bound before paying expansion/fallback costs.

- Avoid forced expansion when the frontier bound already proves termination.
  Today the stream can still eagerly materialize `r=2` or deeper fallback work in cases where the
  builder could already stop based on a conservative unseen bound. After the frontier API cleanup,
  revisit stage progression so expansion only happens when the current frontier bound is not enough
  to terminate.

## Reconciliation

- Consider moving some current `edge_reconcile` reconciliation into the live dedup/assembly path.
  The current post-pass is intentionally narrow and handles unresolved shared-edge mismatches after
  assembly. A future design pass can explore doing some of that agreement earlier, but only if the
  extra hot-path complexity remains clearly justified.

## Correctness

- Tighten behavior for cells that extend beyond the generator hemisphere.
  The current clipping path relies on a gnomonic projection centered at the generator, so cells that
  extend past 90 degrees are outside the model. This should not remain an accidental failure mode.
  Detect the boundary explicitly, then either fail cleanly with a real error or route to a fallback
  that uses a projection/model that can represent the cell.

- Make unsupported geometry a real contract boundary instead of a panic path.
  The hemisphere / >90 degree case is the clearest current example, but the broader goal is to
  separate unsupported geometry from internal bugs. Expected unsupported/pathological outcomes
  should become explicit internal failure states and eventually surface as defined errors.
