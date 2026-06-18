# Packed r=2 Expansion Plan - Archived

Status: superseded by measurement and current regime strategy.

This file used to contain the detailed implementation plan for adding a packed
`r=2` neighbor-expansion stage after the packed 3x3 stage and before directed
cursor fallback. The prototype and later measurements showed that the idea is
not a production win in that form, so the long plan has been removed from the
active docs.

## What The Plan Proposed

- Add a cold packed `r=2` stage after packed tail exhaustion.
- Use a conservative outside-radius-2 bound (`security2`) for unseen
  candidates.
- Scan the radius-2 band, emit ordered candidates, then fall through to the
  directed cursor if the stage did not prove termination.
- Keep stage progression in the neighbor stream rather than in cell clipping.

## Measured Outcome

The runtime toggle was measured across regimes on 2026-06-14:

| regime | result |
|---|---|
| uniform | roughly neutral (`-1.7%`) |
| splittable | catastrophic (`+645%`, about 7.45x slower) |
| mega | catastrophic (more than 9x slower in the recorded run) |

Verdict: do not revive the old scalar packed `r=2` band as a default or simple
toggle. It adds work exactly where dense inputs are already candidate-heavy.

## Current Guidance

- Treat the old `expand_r2` design as a cautionary example: locality or staged
  expansion only helps if it avoids candidate examination or is gated by a
  strong workload signal.
- If this family is revisited, fold it into a grouped SIMD/resumable cursor or
  a broader regime-aware candidate engine instead of restoring the special
  one-off stage.
- Re-measure against `uniform`, `splittable`, and `mega`; uniform alone is not a
  meaningful acceptance test for candidate-production policy.

See:

- `docs/multi-regime-perf.md` for the current dense/sparse strategy.
- `docs/optimization-ideas.md` for the measured rejection entry.
- `docs/perf-profiling-plan.md` for the measurement workflow.
