# Deterministic Parallel Cross-Bin Overflow Resolution

**Status:** reviewed design; telemetry gate precedes implementation

## 1. Objective

Parallelize cross-bin edge matching while preserving the current serial resolver exactly, including
its defect behavior. The first design uses dependency levels: pair buckets that touch no common
shard run concurrently, while every shard observes its incident pair buckets in their current
canonical order.

This is preferable to staging patch commands unless dependency depth makes it ineffective. It
requires no new conflict semantics, provenance representation, global duplicate scan, error path,
or large temporary patch stream.

The phase starts after `EdgeCheckOverflow` records have been grouped and sorted by unordered shard
pair, and ends before deferred-slot fallback reads logical references.

## 2. Current observable order

The resolver visits nonempty pair buckets in ascending pair id:

```text
pair_id = min(source_bin, target_bin) * num_bins
        + max(source_bin, target_bin)
```

Within a bucket it visits equal-`EdgeKey` runs in sorted order and preserves the current endpoint
reconciliation order. `ShardOutput::patch_reference` appends every proposal and makes the latest
proposal the logical winner. Consequently, operations on one shard are order-sensitive when
proposals disagree. Operations on two disjoint pairs commute: they read immutable overflow records,
mutate disjoint `ShardOutput`s, and have no other shared mutable state.

The parallel design must preserve:

- complete `reference_overrides` contents and order per shard;
- final override lookup values;
- unambiguous partial patches from malformed/thirds-mismatch edges;
- which fully reconciled edge observes `CrossBinSlotConflict`;
- the exact ordered mismatch vector and report; and
- final assembled and reconciled output.

## 3. Dependency-level construction

Enumerate active pair buckets in the same ascending pair-id order as today. Maintain
`last_level[bin]`, initially zero. For each pair `(a, b)`:

```text
level = 1 + max(last_level[a], last_level[b])
last_level[a] = level
last_level[b] = level
```

Append the pair task to `levels[level]`. Store its canonical active-pair ordinal so results can be
merged in original order.

A direct container is sufficient:

```text
PairWork { pair_id, bucket, mismatches }
levels: Vec<Vec<PairWork>>
last_level: Vec<u16>
```

Consume nonempty buckets into `PairWork` in ascending pair id. After execution, order completed
work by `pair_id` and append its local mismatches. Endpoints can be recovered from `pair_id` rather
than stored redundantly.

Before scheduling, release-validate that both bins are in range and distinct. A same-bin pair would
self-deadlock in the proposed safe access mechanism and indicates a producer invariant failure.

### 3.1 Proof of shard disjointness

If a later pair shares shard `s` with an earlier pair, `last_level[s]` includes the earlier pair's
level, so the later pair receives a strictly greater level. Therefore no two tasks in one level
share a shard.

### 3.2 Proof of serial equivalence

For every shard, its incident pairs occur in strictly increasing levels and in exactly their
canonical serial subsequence. Each pair retains the existing within-bucket run and endpoint order.
Pairs reordered relative to the serial loop are necessarily shard-disjoint, and their mutations
commute. Thus level execution is observationally equivalent to the current serial pair loop for all
shard state.

Mismatch collection is the only other ordering concern. Each pair writes a task-local mismatch
vector. After all levels complete, concatenate these vectors in canonical active-pair order. This
reconstructs the exact current global mismatch order.

## 4. Execution model

Execute levels sequentially in ascending order, with a hard completion barrier between them.
Execute all pair tasks within one level through Rayon.

For safe mutable shard access, temporarily wrap each exclusive `&mut ShardState` (or the narrower
mutable output state) in `std::sync::Mutex`. Each task:

1. locks its two shard wrappers in ascending `BinId` order;
2. holds both guards for the complete pair bucket;
3. runs the existing bucket match/patch body unchanged, except that mismatches go to its local
   vector; and
4. releases both guards before completing.

The level proof means the locks are uncontended; they exist to express safe ownership without raw
pointer aliasing and to provide a defensive backstop. No task may acquire a third shard. Poisoning
is not recoverable: a worker panic unwinds the computation, and later lock attempts use `expect`
rather than continuing with potentially inconsistent vector/map state. Drop all guards and mutex
wrappers before deferred fallback resumes exclusive access to the shard slice.

Do not use unsafe disjoint-pointer access merely to remove roughly two uncontended acquisitions per
active pair. Consider that only after a measured mutex cost justifies a separately audited helper.

## 5. Pair-local behavior remains unchanged

Extract the current inner bucket loop into a helper, but retain its semantics verbatim:

- one-record, same-side, and three-or-more-record runs keep their taxonomy;
- malformed endpoints and ordinary thirds mismatches keep their taxonomy;
- each individually well-formed matching endpoint still performs its unambiguous patches even if
  full thirds reconciliation fails;
- complete reconciliation reports `CrossBinSlotConflict` exactly when the current serial code
  would; and
- the `a`/`b` patch call order within each reconciled endpoint remains unchanged.

No canonical conflict winner is introduced. No proposal provenance is added. No duplicate override
is collapsed. Defect behavior remains serial-equivalent rather than merely schedule-invariant.

## 6. Required telemetry gate

The existing greedy conflict-free-round census is not the dependency depth. Ordinary edge coloring
may use few rounds while violating per-shard serial order. Before implementation, measure the actual
level schedule described in section 3:

- level count;
- tasks and overflow records per level;
- maximum tasks and records in one level;
- number and record weight of single-task/skinny levels;
- `sum(max_bucket_records_per_level)` as a simple weighted critical-path proxy; and
- total records divided by that proxy as an upper-bound parallelism signal.

The measured graph has roughly 282 active pairs and 133k overflow records; its unrelated greedy
edge-coloring census found nine rounds. Do not use nine as the expected dependency-level count.
Lexicographic pair order can propagate a longer chain, with a worst case linear in active pairs.

Proceed to the prototype if the level width and weighted critical path leave plausible headroom
against the current 5--9 ms serial match phase. Several tens of barriers may still be acceptable,
but that is an empirical question.

## 7. Tests required before performance evaluation

### Scheduler tests

1. Every level contains unique shard endpoints.
2. For every shard, flattening its incident tasks by level exactly equals canonical pair order.
3. Paths, cycles, stars, disconnected graphs, and adversarial lexicographic graphs satisfy both
   properties.
4. Same-bin and out-of-range pair metadata triggers an unconditional producer-invariant assertion
   before any lock is acquired.

### Serial/parallel differential tests

5. Compare complete override vectors and logical lookup values, not only final diagrams.
6. Compare exact ordered mismatch vectors for one-record, duplicate-side, three-plus-record,
   half-malformed, and thirds-mismatch fixtures.
7. Exercise two- and three-edge slot conflicts spanning multiple pair buckets; compare provisional
   winners and exact `CrossBinSlotConflict` attribution.
8. Compare incidence, deferred fallback, resolution-drift state, final cell indices, and reconciled
   diagrams.
9. Run checked representation and semantic fingerprints at 1, 16, and 32 threads.
10. Randomize worker timing repeatedly while holding the canonical schedule fixed; outputs must
    remain byte-identical to the serial oracle.

The serial bucket helper should remain directly callable by tests so the differential does not
duplicate its logic.

## 8. Performance gate

Measure on the post-PBO stock configuration with separate native target directories.

- Primary: 2.5M Fibonacci and uniform, no preprocessing, 16 and 32 threads, rotated pairs.
- Guardrail: 1M single-threaded Fibonacci and uniform.
- Attribution: pair sort, schedule build, per-level match wall time, barrier time if measurable,
  mutex acquisition, mismatch merge, and whole overflow phase.
- Correctness workloads: `cubed`, `mega`, and existing synthetic overflow defects.

The parallel path should retain the current size threshold and fall back to the existing serial
loop for small overflow streams or one Rayon worker. Keep the design only if whole-build time
improves or the phase reduction is clear without measurable single-thread or small-input regression.

## 9. Fallback if dependency levels are too narrow

If telemetry or benchmarking shows excessive dependency depth, use full pair-local patch staging:

1. match pair buckets read-only and emit local attempts plus local mismatches;
2. keep two attempt vectors per pair, one for each source bin;
3. concatenate them into per-source-bin buffers in canonical pair order without a global sort;
4. group by source slot within each source buffer and apply source shards independently; and
5. specify and test canonical conflict semantics before replacing current last-writer behavior.

At the measured scale, up to roughly 266k attempts would cost approximately 5--9 MiB depending on
layout, before vector overhead. A compact attempt may recover its `EdgeKey` from an original
overflow index rather than storing the key. This fallback offers more parallelism but carries a
larger correctness surface: complete provenance, existing-local conflicts, same-key validation,
different-key failure policy, diagnostic attribution, and reconciliation coverage.

Do not implement concurrent per-source mailboxes, a dense array over the roughly 15M source slots,
or a two-pass matcher before profiling demonstrates a benefit. They add synchronization, cache
traffic, or repeated work without simplifying the proof.

## 10. Non-goals

- No reliance on spatial adjacency or cube-face coloring for correctness.
- No ordinary greedy edge coloring that reorders incident pairs.
- No scheduler-selected defect behavior.
- No change to public error, report, or diagram contracts.
- No patch-command staging unless the dependency schedule fails its empirical gate.
