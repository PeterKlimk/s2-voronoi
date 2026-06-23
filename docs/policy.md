# Policy / heuristic notes

This document describes the crate's current internal policy layer in `src/policy.rs`.

The goal is to keep behavior-changing heuristics in one place, with profiling and tests around the
decision points, instead of spreading one-off thresholds across neighbor-query code, cell
construction, and reconciliation.

## Scope

The current policy layer covers:

- packed neighbor chunk sizing
- termination check cadence
- packed count-model constants used to tighten packed thresholds

It no longer owns packed `r=2` expansion. That stage was measured and removed
from the live neighbor-source policy; see `docs/optimization-ideas.md` for the
rejection note (the archived expansion design lives in git history).

## Current decisions

### Packed chunk sizing

`PackedNeighborPolicy` currently chooses:

- `chunk0 = min(16, n - 1)`
- `chunk = min(8, n - 1)`

This keeps the current hot path stable while still allowing tiny point sets to clamp naturally.

### Packed threshold tightening

Packed kNN still uses the count-model heuristics in `PackedKnnCellScratch`, but the controlling
constants now live in `src/policy.rs`:

- `PACKED_HI_BUDGET`
- `PACKED_COUNT_MODEL_IGNORE_DIRECTED_CENTER`
- `PACKED_COUNT_MODEL_INCLUDE_SAME_BIN_EARLIER`

Those constants affect packed threshold tightening. They should be treated as
policy, not as local implementation detail.

### Termination cadence

`TerminationPolicy` currently uses:

- `check_start = 8`
- `check_step = 1`

This policy only affects directed cursor termination cadence. Packed stages still use their own
ordered-batch termination behavior.

## Profiling workflow

When adjusting policy, use timing-driven comparisons instead of only total wall time.

Useful commands:

```bash
cargo test --release --lib
cargo test --release --test api --test correctness
S2_VORONOI_TIMING_KV=1 cargo run --release --features tools,timing --bin bench_voronoi -- 100k --no-preprocess
./scripts/bench_build.sh --timing HEAD
./scripts/bench_run.sh -s 100k -r 5 -c 1 -m total
```

Counters to watch:

- `cells_used_knn`
- `cells_packed_tail_used`
- `packed_tail_builds`
- `neighbors_total`
- `neighbors_max`

Interpretation rules:

- Prefer structural counters over tiny wall-time deltas on noisy runs.
- Treat changes to packed thresholds and activation heuristics as policy changes, not micro-edits.

## Change rules

Before changing policy:

1. Update `src/policy.rs` first.
2. Add or update a policy test that pins the intended decision.
3. Run the targeted benchmark/timing commands above.
4. Update this document and `docs/todo.md` if the heuristic story changed.

Avoid:

- adding heuristic branches directly in `process_cell`
- duplicating thresholds in packed/query/assembly code
- introducing new policy without timing counters or a profiling plan
