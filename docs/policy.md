# Policy / heuristic notes

This document describes the crate's current internal policy layer in `src/policy.rs`.

The goal is to keep behavior-changing heuristics in one place, with profiling and tests around the
decision points, instead of spreading one-off thresholds across neighbor-query code, cell
construction, and reconciliation.

## Scope

The current policy layer covers:

- packed neighbor chunk sizing
- packed `r=2` expansion enablement
- termination check cadence
- packed count-model constants used to tighten packed thresholds
- cold-path caps for packed `r=2` expansion

It does not currently choose a dynamic `r=2` activation heuristic. Expansion is still a simple
policy flag: enabled by default, then triggered whenever packed `r=1` is exhausted without proving
termination.

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
- `PACKED_MAX_EXPAND_R2_CANDIDATES_PER_QUERY`

Those constants affect threshold tightening and cold-path `r=2` expansion caps. They should be
treated as policy, not as local implementation detail.

### Expansion activation

The crate's current default is:

- `packed_knn_expand_r2 = true`

The runtime behavior is:

- do packed `r=1`
- if packed `r=1` is exhausted without proving termination, try packed `r=2`
- if packed `r=2` is exhausted or skipped by cap, fall back to directed cursor

Future smart activation heuristics should stay in the neighbor-source / policy layer. They should
not be added back into `process_cell`.

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
S2_VORONOI_TIMING_KV=1 cargo run --release --features tools,timing --bin bench_voronoi -- 100k --no-preprocess --packed-expand-r2
./scripts/bench_build.sh --timing HEAD
./scripts/bench_run.sh -s 100k -r 5 -c 1 -m total -- --packed-expand-r2
```

Counters to watch:

- `cells_used_knn`
- `cells_packed_tail_used`
- `cells_packed_expand_r2_used`
- `packed_tail_builds`
- `packed_expand_r2_builds`
- `packed_expand_r2_cap_skips`
- `packed_expand_r2_scan_ms`
- `packed_expand_r2_select_ms`

Interpretation rules:

- Prefer structural counters over tiny wall-time deltas on noisy runs.
- Compare knob-off and knob-on behavior on more than one distribution.
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
