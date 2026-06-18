# Perf-Testing Timeline - Archived

Status: compact archive of the 2026-06 strict-plane / occupancy profiling arc.

The old version of this file was a per-commit checklist for one historical
batch of performance work. The active measurement workflow now lives in
`docs/perf-profiling-plan.md`, and the compact measured ledger lives in
`docs/optimization-ideas.md`.

## Retained Facts

- Use paired, interleaved, order-rotated comparisons for timing decisions.
- Treat sub-1% wall-time deltas as noise unless hardware counters and a
  diff-disjoint control make the mechanism clear.
- For density/occupancy/candidate-source changes, always include clustered or
  concentrated workloads (`splittable`, `mega`, sometimes `gradient`); uniform
  alone missed earlier 1.5-9x failures.
- Skip docs, tests/scripts, `#[cfg(debug_assertions)]`-only changes, off-by
  default instrumentation, and codegen-identical refactors when building an A/B
  queue.

## Historical Anchor Commits

| commit | change | archived verdict |
|---|---|---|
| `ff3528b` | plane strict clip rule | Needed a plane no-regression check; see current queue if still pending. |
| `0def19e` | skip per-clip epsilon sqrt when `base_eps == 0` | Tiny/sub-resolution effect when bundled with later binning work. |
| `33b4962` | occupancy rebuild trigger based on `Σocc²/n` | The important axis was clustered/concentrated workloads, not uniform. |
| `24b8df8` stack | micro-opt stack | Recorded as a real win on noisy hardware; see `docs/micro-optimization-matrix.md`. |
| `e659655` | stage-0 entry canonicalization | Real correctness cost to keep in mind when profiling preprocess. |
| `5dbfd5c` / grid weld | integrated weld redesign | Large preprocess win on welded runs. |

## Current Pointers

- Use `docs/perf-profiling-plan.md` for how to run and record new A/Bs.
- Use `docs/optimization-ideas.md` for current accepted/rejected performance
  state.
- Use git history before this compaction for the old full per-commit skip list.
