# Performance

Per-cell construction is embarrassingly parallel; the only cross-cell work is vertex
deduplication. Per-point cost is near constant in n, so the advantage over hull- and
triangulation-based libraries grows with input size.

## Reference numbers

Ryzen 3600 (6 cores), uniform input, minimum of repeats:

| n  | multithreaded |
|----|---------------|
| 1M | ~330ms |
| 2M | ~720ms |

Single-threaded, ~1.8s at 1M.

Peak resident memory per build is roughly linear, ~0.65 KB/point (500k ≈ 320 MB, 1M ≈ 660 MB,
2M ≈ 1.3 GB). The working set frees when the diagram drops. A process that builds many diagrams in
a loop will accumulate high-water RSS — glibc does not return freed arenas to the OS between
builds, amplified by per-thread rayon arenas. This is allocator behavior, not a leak; build in a
child process per job, set `MALLOC_ARENA_MAX=2`, or link jemalloc/mimalloc if it matters.

For repeated similarly sized builds, `VoronoiWorkspace` is the portable way to trade retained
memory for throughput. It keeps the existing per-worker cell-construction contexts, including one
point-count-sized `u32` visitation table per active context, but no input or output geometry. A
subsequent computation with an incompatible effective point count or grid shape discards the old
contexts rather than accumulating multiple sizes. Call `clear()` after the batch to release the
retained scratch; use ordinary `compute` for one-shot work.

On the 16-core Ryzen test host, four interleaved counter pairs at 2.5M points (four builds per
process, native code, no preprocessing) found the workspace reduced aggregate cycles by 2.53% on
uniform input and 3.20% on Fibonacci input. Task-clock fell 2.74% and 2.77%, respectively, while
retired instructions changed by -0.09% and +0.004%. That pattern supports avoided allocation,
page-initialization, and cache/TLB disruption rather than an arithmetic shortcut. At 2.5M points
and 16 active contexts, the retained visitation tables alone are approximately 160 MB.
The same mechanism is useful without parallelism: four pinned 1M Fibonacci pairs reduced cycles
and task-clock by 0.76% and instructions by 0.20%, while retaining one approximately 4 MB table.

## Point-location queries

`SphereLocator::locate` accepts arbitrary finite, nonzero f32 directions and therefore validates
and normalizes each raw query in f64. `locate_many` first materializes a canonical 12-byte direction
per query so the subsequent grid search can run in parallel and invalid batches can report the
lowest input index deterministically. At one million queries, that buffer is about 12 MB and lives
alongside the returned index vector.

When queries are reused or already canonical, construct `SpherePoint` values once and call
`locate_point` or `locate_many_points`. These infallible paths skip the normalization and canonical
input buffer; the batched form allocates only its returned indices. In the 2026-07 point-API review,
raw-query normalization cost about 93 retired instructions/query and 2.4% native single-threaded
locator time. The serial prepass made the raw multithreaded batch about 5.1% slower; the checked
point path exists for throughput-sensitive repeated queries.

## Building for speed

`RUSTFLAGS="-C target-cpu=native"` is worth ~6% on the reference machine and is the main build
flag that matters. Run benchmarks in release.

PGO/BOLT is worth a further ~5% (measured 2026-07, minutes to set up). It is a code-layout win,
not a code improvement: the gain is brittle across source changes and profiles, so treat it as an
opt-in recipe for benchmark headlines and deployment binaries rather than something the source
should be tuned against. The 1–2% per-binary layout noise documented under
[Comparing commits](#comparing-commits) is the same effect uncontrolled.

The archived LLVM PGO experiment also established a real workload-policy tradeoff. A balanced
profile trained on Fibonacci, uniform, clustered, and mega reduced binary text by 6.8%; at 2.5M
multithreaded Fibonacci it reduced cycles 4.16%, instructions 3.36%, and branches 3.85%. Clustered
cycles improved 2.1% despite 0.8% more branches, while mega was approximately cycle-neutral
(+0.7%) with 0.6% fewer instructions and 4.3% more branches. Fibonacci-only training reached about
6.3% fewer Fibonacci cycles but regressed mega by roughly 3.5%. Profile choice must therefore be
explicit; the old build-helper scripts were not retained on main.

## Running the benchmarks

The benchmark binaries need the `tools` feature:

```bash
cargo run --release --features tools --bin bench_voronoi -- 100k 500k 1m
```

Useful flags:

- `--dist {fib|uniform|clustered|bimodal|gradient|outlier|splittable|mega|cubed|great-circle}`
  and `--dist-param` —
  non-uniform distributions exercise the density-adaptive paths that uniform input never reaches.
  Benchmark across a few of these, not uniform alone. `cubed` is a deterministic projected quad
  grid with O(n) degree-4 vertices, intended specifically for reconciliation benchmarks.
  `great-circle` is a narrow perturbed band intended for successful high-candidate-work benchmarks;
  its coordinate jitter defaults to 0.01 and is controlled by `--dist-param`.
- `fib` evaluates the unbounded phase and latitude in f64 before storing f32 coordinates.
- `--validate` — compare against the convex-hull ground truth (slow; capped at 100k).
- `--no-preprocess` — skip welding (isolates construction cost).
- `--reuse-workspace` — retain construction scratch across `--repeat` iterations. The first
  iteration is cold; compare aggregate repeated runs and account for the retained memory described
  above.
- `--edge-queue-audit` with `--features profiling` — emit queue-length, capacity, growth, pool
  reuse, and per-shard high-water telemetry for the internal live-dedup edge-check queues.

## Knobs

- `RAYON_NUM_THREADS=1` — single-threaded, for stable comparisons.
- `VORONOI_MESH_BIN_COUNT=<n>` — explicit shard target, quantized and capped at 96. Without an
  override, the default is 2x workers through 8 workers and a 96-bin coarse layout at 9+; severely
  imbalanced coarse layouts may refine to 216 bins.
- `VORONOI_MESH_TIMING_KV=1` with `--features timing` — machine-readable phase timing.
- `VORONOI_MESH_RECONCILE_TELEMETRY=1` — on defect-bearing builds, emit a read-only
  `RECONCILE_KV` simulation of the primary reconciliation round before mutation. It reports
  inferred endpoint-pair distances, origin-specific histograms, and the diameter of the vertex
  equivalence components the current policy would create, plus the number rejected by the
  no-chain diameter gate. This intentionally repeats cold-path reconciliation work and is for
  correctness audits, not performance measurements; clean builds skip even the environment lookup.
- `VORONOI_MESH_GRID_DENSITY=<f>` — spatial-grid target density (points per cell) for sweeps. It is
  snapshotted on first use; run each density in a separate process.

The complete supported, diagnostic, campaign, and manual-probe inventory is maintained in
[`environment-knobs.md`](environment-knobs.md).

Timing builds also expose scale-relative construction-work telemetry in both the human report and
`TIMING_KV`:

- `candidate_work_*` describes all examined candidates per generator;
- `no_progress_tail_*` describes candidates examined after the final polygon-changing constraint;
- p50/p90/p99/p999 values are exact below 256 and power-of-two bucket lower bounds above it; and
- `*_ge4x_median_lb`, `*_ge16x_median_lb`, and `*_ge64x_median_lb` are conservative counts relative
  to that same run's median (floored at one), rather than fixed global thresholds. The exact base
  used is emitted as `*_relative_base`.

`no_progress_tail_excluded` counts cells whose batched exhaustion recovery did not preserve
per-candidate clip outcomes. Those cells still contribute to total candidate work. Use raw
quantiles across input sizes to measure natural scaling, then use the run-relative tail counts to
distinguish a few exceptional cells from a broadly expensive distribution.

## Local-rebuild cold path

Post-assembly local rebuilding runs only on defect-bearing inputs (rare; near-cocircular
clusters, e.g. the `mega` distribution) and is fast in the common case. One known cold path
remains: when the defect sits on the boundary of an extremely dense cluster (most points inside a
single grid cell), the rebuild's neighbor gather expands into the cluster and the rebuild can take
seconds to minutes at millions of points. The output contract is unaffected — the result is still
strictly valid or a clean error — only latency degrades. `LocalRebuildMode::Disabled` skips local
rebuilding entirely (residual defects then fail plain `compute` loudly).

## Comparing commits

The machine is noisy — per-binary code-layout shifts alone are ~1-2% at 500k single-threaded — so
compare commits with interleaved paired runs, not back-to-back batches, and treat sub-1% deltas as
noise:

```bash
./scripts/bench_build.sh --chain 6
./scripts/bench_run.sh -s "500k 2m" -d "uniform mega" --seeds "1 2 3" --csv out.csv
```

`bench_run.sh` sweeps sizes x distributions x seeds, runs the commits interleaved, and emits a
CSV. Prefer hardware counters (`perf stat`) over wall time for behavior decisions; wall time on
this class of machine drifts more than most single optimizations are worth.

For a first-pass hardware-counter comparison of the same built artifacts:

```bash
./scripts/bench_build.sh --chain 2
./scripts/bench_perf.sh -s 500k -d fib -r 9 --csv /tmp/bench_perf.csv
```

The counter runner rotates version order, defaults to one Rayon thread pinned to CPU 0, and emits
raw tidy CSV rather than prematurely aggregating results. Compare instructions first, then cycles,
branches/misses, and cache behavior. Use the included context-switch and CPU-migration counts to
identify contaminated samples. If counters do not resolve the decision, use `bench_run.sh
--converge` for longer paired wall-time runs.

### Counter calibration on the noisy reference box

A 2026-07 control run compared two 500k single-thread binaries with behaviorally identical release
code over nine paired rounds. Retired instructions and branches were stable to a few parts per
million. The paired cycles ratio had an approximately ±1.2% 95% interval; hardware cache counters
were noisier. Hardware branch misses showed a false-looking ~0.8% separation, while Cachegrind's
deterministic branch simulator differed by only 18 of ~704k predicted misses on a 10k run.

Accordingly, use retired instructions/branches as the primary structural signal, paired cycles as
the primary hardware-cost signal, and Cachegrind (`--branch-sim=yes`) or `llvm-mca`/`cargo asm` for
attribution. Treat hardware branch/cache misses as corroborating evidence rather than a standalone
verdict on this machine. Effects below the cycles noise floor should proceed to longer paired
`bench_run.sh --converge` runs.

### Resource-bound calibration

The timing feature reports `weld_pairs`, `weld_pair_capacity`,
`packed_keys_materialized`, `packed_key_capacity_peak`, tail possible/requested counts, ring-tail
rescan/dot counts, total/unrequested center-tail candidates, and total/unused high-threshold
`chunk0_keys` in `TIMING_KV`. Measurements leading to
the initial packed aggregate-work bound found a 500k uniform peak capacity of 6,464 keys versus
2,220,652 keys (~17.8 MiB of `u64` payload in one worker) for the clustered distribution. The 1M
query×candidate budget reduced the clustered peak to 1,188,540 keys (~9.5 MiB of allocator
capacity) and routes larger groups to the bounded shell fallback.

Occupancy feedback also changes the packed prefix's cost balance. Once it rebuilds the grid,
same-cell groups of at least the dense-band work unit (128 queries) use a two-unit (256-candidate)
high prefix; smaller groups and grids that were not rebuilt retain 32. Alternating native pairs on
the 16-core host improved 500k mega by 15.96% at 16 workers and 15.27% at 32, and 1M bimodal by
11.07% and 8.93%. Gradient, clustered, outlier, splittable, Fibonacci, and uniform guardrails were
unresolved around zero. The 100k cube-vertices stress case improved 6.38% at 16 workers and was
unresolved at 32. Peak RSS was essentially unchanged for mega and bimodal and fell in that cube
stress run. The rule is deliberately conditioned on the spatial feedback signal rather than an
input distribution or platform.

A post-change production-counter pass confirmed that this is removed work rather than timing
movement. Four 16-worker pairs reduced mega instructions/cycles by 24.90%/15.52% and bimodal by
10.67%/8.79%; Fibonacci changed by -0.01%/+0.04%. Sampling the retained build still placed
`select_nth_unstable` at 13.8% of mega samples. Sorting the remaining eager prefix after a query
consumed two batches tested that ceiling: eight pairs improved mega by 3.53% (bootstrap interval
2.08--5.41%) and left bimodal unresolved at +0.37% candidate/base (-0.53--+1.14%). The prototype
was nevertheless rejected because its per-query sorted-state path added about 0.24% instructions
and 0.47% branches to Fibonacci and about 0.25% instructions to bimodal. Revisit repeated
partitioning only with group-level specialization that leaves the ordinary packed emitter
unchanged; observed continuation depth is the demonstrated discriminator.

The follow-up shell profile did not expose another keeper. Caching the takeover query's normalized
`DVec3` across layers regressed mega by 1.77% and great-circle by 8.21%; the sampled square roots
were primarily per-cell conservative cap bounds, while the larger hot frontier state harmed code
shape. A renewed attempted-slot census found shell duplicate shares of about 1.9% on mega, 16.9%
on bimodal, 12.9% on clustered, and 0.4% on great-circle. Filtering those slots before shell dot/key
construction saved only 0.17% instructions on bimodal while adding 1.12% cycles and 3.8% cache
misses. Clustered saved 0.38% instructions but added 1.76% branches; mega and great-circle added
instructions/branches, and mega cache misses rose about 5.2%. The earlier rejection therefore
still holds after the wider prefix: full re-coverage is visible, but consulting the large attempted
table during spatial scanning costs more than the avoided keys.

That fallback is intentionally a reliability tradeoff: at 100k clustered it cost approximately
23% more instructions and 15% more cycles; uniform work remained structurally neutral. The weld
pair budget added approximately 0.8% instructions to a 500k normal-preprocessing control, while the
paired cycles interval remained unresolved around no change. Revisit these thresholds only with
both peak-storage telemetry and end-to-end counter measurements; optimizing away the fallback must
not restore unbounded retained work.

A refreshed native 2.5M scaling/profile pass on the 16-core Linux host found no packed-specific
parallel bottleneck. Fibonacci's reported packed phase fell from 615ms at one thread to 58ms at
sixteen (10.6x), while uniform fell from 743ms to 73ms (10.2x); total cell construction scaled by
essentially the same factors. Frame-pointer sampling attributed 7.8% of whole-build cycles to
`prepare_group_directed`, but its descendants remained distributed: small sorting networks were
about 1.3%, partitioning 0.9%, and tail materialization 1.2%. The current loop already shares each
candidate coordinate chunk across the group's queries, counts rather than materializes unused
center-tail keys, and builds tails only on demand. Combined with the retired smaller-prefix,
streaming-heap, seed-first, and lazy-high-key results, further packed-preparation changes need a
similarly explicit work-removal mechanism and must preserve the occupancy-feedback guardrails
above.

The normal 100k uniform packed-bound comparison remained unresolved after the maximum 160 paired
wall-time rounds: candidate/base geometric mean `+0.3%`, 95% interval `[-1.1%, +1.8%]`, with the
candidate faster in 82/160 rounds. This is below the 1% decision resolution on the reference host;
do not describe it as either a performance win or a demonstrated regression.

Incremental shell-layer emission keeps whole sorting for layers up to 128 entries and otherwise
partition-sorts 64-entry prefixes on demand. At 100k single-threaded it reduced clustered retired
instructions by 3.05%, branches by 1.76%, and cycles by 3.52%; bimodal instructions fell 0.91% with
neutral cycles. Fibonacci instructions/branches were structurally neutral-to-lower, while its
0.83% cycle increase remained inside the reference host's noise interval.

Center-tail candidates are counted during the initial SIMD pass but their keys are materialized only
when that query requests the tail. At 100k clustered this reduced key materialization from 24.19M to
18.67M and peak capacity from 1,018,656 to 782,400 keys (about 1.9 MiB), while reducing instructions
2.05%, branches 2.88%, and cycles 2.83%. Fibonacci improved by 1.20% instructions, 1.29% branches,
and 3.55% cycles after the requested-query rescan was vectorized.

Shell-frontier pending entries use one packed `u64` whose high word orders the dot descending and
whose low word orders the slot ascending. This preserves the previous
`(Reverse(total-order dot), slot)` selection policy while avoiding tuple-key construction and
comparison. In a 20k mega run it reduced `knn_query` time by 18.4% and total time by 7.0% over 12
paired interleaved rounds; the query phase won all 12 pairs. A 10k mega counter pair reduced retired
instructions from 7.37B to 5.70B and cycles from 2.40B to 1.92B. At 50k clustered, `knn_query`
improved 23.6% over eight pairs. The 100k Fibonacci control has negligible shell time; its wall-time
result was unresolved, while counters improved about 1.1% instructions and 1.2% cycles.

Packed center passes test an emission mask before extracting SIMD lanes to a scalar array. The
change is deliberately small: at 2M clustered it reduced retired instructions by 0.07–0.08% in
three of three paired runs, while cycles remained neutral. A 1M Fibonacci control was structurally
neutral. Splitting the experiment showed that the center-prepare guards account for nearly all of
the saving; the requested-tail rescan guard alone was neutral. Retain this as a low-risk structural
cleanup, not as a demonstrated wall-time improvement.

The large-polygon clip output writer uses an increment-and-wrap step instead of `(i + 1) % n`.
Release code previously emitted an integer divide for every retained input vertex. An isolated
mixed-clip microbenchmark improved from 48.12 to 43.27 ns/call at N=9, 59.36 to 49.73 ns/call at
N=12, and 82.61 to 62.62 ns/call at N=20 (10%, 16%, and 24%). Fixed-work counters at 500k and 1M
mega were neutral because large-polygon output is rare relative to the full build. Retain this as a
clear local latency win; whole-program measurements describe its incidence, not its intrinsic cost.

Generator finiteness validation and unit-vector canonicalization share one chunked traversal. The
fused pass leaves invalid values untouched, reduces chunk-local first-invalid indices to the same
minimum global index as the old validation pass, and uses the identical per-point f64 normalization
arithmetic. At 2M single-threaded it reduced retired instructions by 0.18% on Fibonacci and 0.16%
on uniform input in all three paired runs. Cycles improved 3.5–4.4% on Fibonacci and 2.3–3.9% on
uniform. The same instruction reduction was visible in the default parallel build.

The low-incidence local-rebuild trigger uses the construction-time owner-local summary on clean
builds. If reconciliation changes cell cycles, it captures only their original spans and updates
incidence from the sparse old/new delta; exact old counts come from the at-most-three owner cells in
each vertex key. On mismatch-bearing builds, assembly also retains the sparse ids behind an
owner-local degree-1/2 signal so the delta path can resolve real defects versus mismatch-bookkeeping
false positives. Clean builds keep the original summary loop and do not collect those ids. Missing
provenance falls back to the whole-diagram scan. That fallback uses plain `u32` counters with one
Rayon worker and compact atomics with multiple workers.

On the 10M uniform/no-preprocess case, four reconciled cells previously triggered the fallback.
Removing it reduced Linux native instructions/branches by 0.20%/1.11% and portable instructions/
branches by 1.90%/3.60% over three interleaved runs. Ten Windows native pairs improved 3.31% with a
95% paired CI of [2.74%, 3.87%] and ten portable pairs improved 2.13% directionally (CI crossed zero
at [-0.25%, 4.45%]). A checked 10M run pins the local result equal to the exhaustive scalar oracle.

A follow-up native Windows ring at 2.5M Fibonacci/no-preprocess isolated a clean-path regression to
the sparse low-id collection in the first version of this optimization: `2cea6fb -> 1cb94d6` was
+1.26% with a 95% paired CI of [+0.33%, +2.19%]. Gating that collection on nonempty mismatch
records removes 0.49% retired instructions and 0.88% branches on the clean Fibonacci path while
preserving the 10M four-cell reconciliation result. The direct Windows fix pair was directionally
favorable but unresolved (-0.44%, CI [-1.25%, +0.39%]); retain the gate on structural evidence.

The N=3/4 small clipper keeps its four SIMD distances in a four-element array instead of padding an
eight-element array with four zero stores; N=5–8 retains the full eight-lane representation.
Release code shrank by 208 bytes and no longer contains
the two padded 16-byte stores. At 2M single-threaded, retired instructions fell about 0.062% on
Fibonacci and 0.061% on uniform input in all three pairs. Fibonacci cycles improved in all pairs;
uniform cycles were neutral. The default and microbench paths share the initialized-slice invariant.

Interior packed-kNN security thresholds evaluate each of the four boundary planes across eight
queries with `PointChunk8`. Lane minima remain scalar and follow the original plane order, so the
wide and scalar backends produce identical threshold bits; remainder and boundary-cell paths remain
scalar. At 2M single-threaded, retired instructions fell about 0.058% on Fibonacci and 0.053% on
uniform input in all three pairs. Cycles were unresolved. The wide and `simd_scalar` 100k backend
fingerprints remained identical.

Cell construction dispatches each exact neighbor batch once to a packed or shell-specialized loop.
The packed loop marks every occurrence; the shell loop performs insertion-based deduplication. This
removes the invariant source match from every candidate while preserving source-specific bounds,
timing, tracing, and termination. At 2M single-threaded Fibonacci it reduced retired instructions by
about 0.096%; at 500k clustered it reduced them by about 0.206%, consistently across three pairs.
Cycles were neutral-to-lower overall.

The same source specialization elides self-neighbor filtering from packed batches: packed center
and ring construction already exclude the query slot, while shell takeover can rediscover it and
retains the check. This directly removes the hottest distinct sampled instruction in
`clip_batch_source<false>`. At 1M single-threaded native Fibonacci, nine pairs reduced retired
instructions by 0.178% and branches by 1.824% in every run; cycles favored the change in six of nine
pairs but remained noisy. Generic-target 500k counters reduced instructions by 0.159% and branches
by 1.805%. Portable Cachegrind at 100k independently showed 0.169% fewer instruction references,
0.478% fewer conditional branches, and 2.88% fewer conditional branch mispredicts. Fifteen Windows
native 2.5M pairs were unresolved (8/15 favorable, about -0.4% median) amid large host noise.

The quadratic cube projection computes one absolute-value square root and branchlessly selects the
positive or negative result. It is bit-identical to the previous finite-input branches, including
signed zero, subnormals, and grid-boundary-adjacent values. At 2M single-threaded it added about
0.05% retired instructions but removed about 0.06% branches. Cycles improved in all three Fibonacci
pairs and two of three uniform pairs, with the third uniform pair neutral; hardware branch misses
were lower in all six pairs. This is a latency win from removing a balanced sign branch, not an
instruction-count optimization.

Assembly pre-sums shard bookkeeping lengths and reserves exact aggregate capacity before appending
overflow checks and deferred slots; unresolved edges reserve only when nonempty. At 500k Fibonacci
with 96 bins this removed eight growth reallocations from each active vector, eliminated about
3.14 MB and 1.70 MB of recopied overflow/deferred payload, and reduced final capacities from
79,936/54,656 to the exact 61,504/42,360 entries. Fixed-work instructions and peak RSS were neutral;
cycles were neutral-to-lower. This is an allocation/capacity win, not a demonstrated total-time win.

Dense-cell index construction computes all three coordinate ranges in one traversal, selects the
sort coordinate slice once outside the comparator, and materializes sorted slots and coordinates
together. Exact baseline-equivalence tests cover axis ties, equal and nonfinite coordinates, float
bits, ordering, and band queries. Fixed-work 500k outlier and mega counters were neutral because
dense-index construction is a small cold portion of the build. Retain this as a strictly local
pass/branch reduction, not as a demonstrated end-to-end improvement.

Edge reconciliation reuses two unrestricted segment vectors across unresolved records within each
reconciliation round. The previous path created two vectors per record; an 86,400-point high-degree fixture
with 6,626 records therefore exposed 13,252 allocation opportunities. The clean path still returns
before allocating scratch, and each reconciliation round receives fresh buffers. Same-session defect-heavy
timing moved from 13.884 ms to 12.804 ms; repeated candidate timing was noisy, so retain this for the
certain structural allocation removal rather than the provisional 7.8% phase result.

Tolerance reconciliation retains a sparse membership ledger only on defect-bearing runs. Proposed
threshold-graph components are checked transactionally in f64 against every original member, so a
later round cannot hide a transitive diameter violation behind an earlier representative. Rejected
components seed Hull3d. Clean builds allocate neither the ledger nor proposal state.

Merge-safety face validation uses that ledger's original vertex ids to derive a complete local cell
cover from their key triples. This includes cells inherited by a surviving representative across
earlier reconciliation rounds. Missing or invalid provenance falls back to the prior global scan, and
checked builds compare localized component decisions against a global oracle. The `timing` feature
reports `merge_safety_scan_cells` and `merge_safety_global_fallbacks`.

The deterministic `cubed` reconciliation workload showed strong scaling. At 99,846 sites, the
local cover scanned 2,650 cells with zero fallbacks; twelve interleaved single-threaded rounds cut
median `edge_reconcile` time from 19.6 ms to 8.0 ms. At 501,126 sites, it scanned 2,084 cells and
seven rounds cut the phase from 55.3 ms to 6.4 ms. Nine 100k Linux perf pairs on normal non-timing
builds reduced whole-build retired instructions by 13.47%, branches by 21.80%, and cycles by 9.74%,
with every pair favorable. Task-clock improved 3.96% but only 6/9 pairs favored the candidate, so
the structural counters are the stronger evidence on the busy host. A nine-pair 500k Fibonacci
clean-path guardrail reduced instructions by 0.16% and branches by 0.49%, again with every pair
favorable; cycles were noise-dominated with only 4/9 candidate wins.

Cell construction no longer clears its reusable output buffer before building. Every successful
writer—gnomonic, spherical fallback, and all-constraints recovery—clears before writing, while an
error returns before the driver can consume the buffer. Poison-buffer tests cover all three writers
and a terminal error retaining stale poison. At 2M single-threaded this reduced retired instructions
by about 0.041% on Fibonacci and 0.038% on uniform input in all three pairs; cycles were unresolved.

The timing-only directional-support audit caches its 64 unit directions once instead of recomputing
their sines and cosines whenever a polygon invalidates the support cache. This does not affect
production builds. On 500k single-threaded Fibonacci with native instructions, the diagnostic build
dropped from 9.39B to 7.07B instructions and from 5.09B to 3.75B cycles. All audit counters were
unchanged, including 1,300,400 support tests, 112,559 hits, and zero false positives.

Gnomonic extraction writes its four parallel outputs into reserved spare capacity and publishes all
four vector lengths once after every vertex validates. This removes four capacity branches and four
length updates per output vertex while keeping partially initialized output unobservable on error.
At 500k single-threaded with native instructions, Fibonacci retired instructions fell 1.59% in all
nine pairs and cycles fell 1.13% in seven of nine; uniform instructions fell 1.49% in all seven
pairs and cycles fell 2.50% in six of seven. Cachegrind independently measured 1.58% fewer
instructions at 20k Fibonacci. Without native instructions, Fibonacci instructions fell 1.75% in
all seven pairs while cycles were neutral. Valgrind Memcheck reported no errors end to end.

Convex clipping computes its two intersection parameters together. Zero-epsilon spherical clips
return the raw, already-bracketed divisions; propagated-epsilon edge checks and the diagnostic
fallback retain finite checking and clamping. On 500k single-threaded native Fibonacci this
reduced instructions by 1.11% and branches by 0.138% in all nine pairs with neutral cycles; native
uniform reduced instructions by 1.06% and branches by 0.138% in all nine pairs. The generic-target
build improved by 1.27% instructions, 0.36% branches, and 2.75% cycles on Fibonacci.

Edge collection validates the four parallel cell-output lengths and vertex-index scratch length once
per cell, then uses that proof to avoid repeated bounds checks in the hot edge loop. At 500k
single-threaded with native instructions, Fibonacci reduced instructions by 1.06% and branches by
2.62% in all nine pairs; cycles were noisy and are treated as neutral. Uniform reduced instructions
by 1.00%, branches by 2.42%, and cycles by 1.04%. Cachegrind reported 1.04% fewer instructions at
20k Fibonacci. The generic-target build reduced Fibonacci instructions by 0.71%, branches by 2.69%,
and cycles by 3.86% in all seven pairs.

Cell construction consumes incoming edge checks directly when seeding the clipper instead of first
copying each check into a temporary `SeedNeighbor` vector. Key orientation, seed order, slot lookup,
and epsilon bits are unchanged. At 500k single-threaded with native instructions, Fibonacci reduced
instructions by 0.99% and branches by 0.53% in all nine pairs; cycles fell 2.06% in eight of nine.
Uniform reduced instructions by 0.94% and branches by 0.47% in all nine pairs, with neutral cycles.
The 100k mega case reduced instructions by 0.22% in all seven pairs. The generic-target Fibonacci
build reduced instructions by 0.80% and branches by 0.41% in all seven pairs, with neutral cycles.

For AVX2 builds using the `wide` backend, packed interior-cell security thresholds finalize eight
positive finite plane distances together; exceptional lanes retain the existing scalar geometric
fallback. At 1M single-threaded native Fibonacci over 90 pairs, instructions fell 0.368% and
branches 0.274% in every pair, while cycles were neutral (-0.055%, 53/90 favorable). At 1M uniform
over 60 pairs, instructions fell 0.343% and branches 0.251% in every pair, and cycles fell 0.376%.
The ungated generic-target form reduced retired work but regressed cycles 3.76%, so generic and
`simd_scalar` builds deliberately retain the original scalar finalization; their structural counters
were unchanged within 0.00002% after gating.

Edge emission iterates its `Copy`-only per-cell scratch records by reference and clears each buffer
after successful forwarding, avoiding `Vec::drain` state and unwind bookkeeping. At 1M
single-threaded native Fibonacci over 30 pairs, instructions fell 0.667% and branches 0.102% in
every pair; cycles fell 3.22% in 26 of 30, though the magnitude remained layout-sensitive. At 500k
native uniform, instructions fell 0.631% and branches 0.086% in all nine pairs. The generic-target
Fibonacci build reduced instructions 0.642% in all nine pairs with neutral cycles.

The same edge-emission path validates its vertex-key and resolved-index lengths once per cell, then
uses the producer's cyclic-local invariant to avoid four repeated endpoint bounds checks per
forwarded edge. At 500k single-threaded native Fibonacci, instructions fell 0.413% and branches
1.106% in all nine pairs; uniform fell 0.382% instructions and 1.025% branches. The generic-target
Fibonacci build reduced instructions 0.378% and branches 1.090% in all nine pairs, with neutral
cycles.

Final assembly captures its immutable scatter inputs as references and slices by value in the Rayon
closure. In particular, this lets optimized code retain the vertex-offset slice's data pointer and
length outside the packed-reference loop while preserving checked indexing and its release panic on
corrupt input. At 1M single-threaded native, retired instructions fell 0.291% on Fibonacci and
0.270% on uniform, while branches fell 0.124% and 0.112%; all six pairs agreed. The generic-target
build reduced instructions 0.280%/0.260% and branches 0.121%/0.109% across four agreeing pairs. The
hot Rayon helper grew by about 37 bytes, but the repeated closure-environment loads disappeared
without inner-loop spills.

The same assembly pass writes each final cell's `(start, count)` metadata during its checked prefix
sum and then reads that immutable metadata during parallel scatter. This removes the separate
`num_cells + 1` start vector, a prefix stream, duplicate cell-metadata stores, and the scatter-time
random shard count load. Release initializes `VoronoiCell` spare capacity and publishes its length
only after the checked prefix completes; debug retains full-coverage sentinels. At 1M
single-threaded native, retired instructions fell 0.167% on Fibonacci and 0.155% on uniform, while
branches fell 0.125% and 0.113%; all six pairs agreed. Generic-target instructions fell
0.159%/0.148% and branches 0.121%/0.110% across four agreeing pairs. Cycles favored the candidate in
four of six native pairs per distribution and were mixed in generic trials.

Cross-bin overflow checks sort by their scalar edge key rather than `(key, side)`. Resolution only
needs contiguous equal-key runs: side equality and reverse-winding patching are symmetric for the
normal two-record run, while duplicate runs of three or more are deferred without selecting a pair.
At 1M, ordinary 6-bin Fibonacci/uniform inputs contained 18.9k/17.8k overflow records and reduced
native retired instructions by 0.032%/0.026%. With 96 bins, 85.8k/83.8k records increased the gain to
0.163%/0.142%. All four structural pairs agreed in each regime; generic-target builds showed the
same scaling. Branch reductions were smaller, and cycles ranged from mixed to favorable on the noisy
host.

Overflow resolution then narrows sort movement further by sorting 16-byte `(key, record index)`
handles while leaving the 40-byte records immutable. Handle construction and allocation are included
in the measured path; resolution pays one indirect record read after grouping. Relative to sorting
the records directly by scalar key, 1M native instructions fell another 0.021%/0.017% on 6-bin
Fibonacci/uniform and 0.116%/0.099% at 96 bins; branches fell 0.011%/0.009% and 0.061%/0.050%.
Generic-target gains were 0.030%/0.025% at 6 bins and 0.165%/0.141% at 96 bins. All structural pairs
agreed; default-bin native cycles improved in all eight pairs, while high-bin results ranged from
mixed to favorable. The private resolver accepts an immutable record slice, making its independence
from record permutation explicit.

Within-bin forwarded edge checks store the earlier neighbor's 32-bit generator id instead of the
canonical 64-bit edge key. The destination generator is already known at both consumers, so defect
records reconstruct the exact key with `pack_edge(destination, neighbor)`; seed clipping also avoids
decoding the key. This shrinks the hot queue record from 32 to 24 bytes without changing enqueue
order, matching, diagnostics, or reconciliation inputs. At 1M single-threaded native, retired instructions
fell 0.189% on Fibonacci and 0.187% on uniform across six agreeing pairs; branches were neutral.
Generic-target instructions fell 0.178%/0.175% and branches 0.366%/0.326% across four agreeing
pairs. Longer native cycle matrices remained below the host's decision floor: Fibonacci measured
+0.78% with 10/24 candidate wins and a roughly -1.44% to +3.06% interval; uniform measured +0.45%
with 5/16 wins and a roughly -0.05% to +0.97% interval. The lossless record shrink and consistent
structural reductions were retained rather than treating unresolved layout-sensitive cycles as a
regression.

Cell construction now takes the generator position from the group's known contiguous grid slot
instead of gathering `points[generator_idx]`. The slot record retains the same global generator id,
and checked builds verify that its position is bit-identical to the canonical point. The same value
is forwarded through builder reset, shell-frontier initialization, mid-batch termination bounds,
and exhaustion recovery, so the ordinary path no longer reintroduces the scattered generator load.
On the 2M, twelve-thread native reference run, nine paired Fibonacci rounds reduced cycles 8.41%,
cache references 6.89%, and cache misses 17.04%; nine uniform rounds reduced them 8.93%, 5.10%, and
9.19%. Every cycle and cache-miss pair favored the candidate. Retired instructions fell only
0.12--0.23%, identifying locality rather than changed query work as the cause. A seven-pair 2M
uniform run with preprocessing and 96 bins retained an 8.86% cycle and 13.16% cache-miss reduction.
The pinned one-thread Fibonacci guardrail also remained favorable (4.87% fewer cycles over seven
pairs), and the full checked suite passed. On an Intel i5-1038NG7 MacBook Pro using the MSRV
toolchain (Rust 1.88 / LLVM 20), 2M eight-thread wall time improved 2.9% on Fibonacci (95% interval
1.4--4.3%, 14/16 pairs) and 2.6% on uniform (2.0--3.2%, 16/16 pairs).

Forwarded in-bin edge checks now carry the earlier generator's grid slot rather than its global id.
Seed clipping uses that slot directly and gets position plus global id from one `SlotPoint` load,
removing the scattered global-to-slot inverse lookup. At 2M/12T, nine Linux Fibonacci pairs reduced
cycles 2.28% in eight of nine despite adding 0.50% instructions and 0.67% branches. On the quieter
2M/8T Intel Mac, Fibonacci wall time improved 2.8% (95% interval 1.4--4.2%, 13/16 pairs) and uniform
improved 3.8% (3.3--4.4%, 16/16 pairs). Because weld compaction is then the inverse slot map's final
consumer, the map is released before construction, saving about 8 MB at 2M. That follow-up was
instruction/cycle neutral, reduced Linux hardware cache references 2.05%, and was wall-time neutral
on both Mac distributions over 32 pairs each.

The default same-cell weld scan now rejects a candidate after one face-tangent coordinate whenever
that component's square alone is not below the strict weld threshold. Because the remaining squared
components are non-negative, this is an exact early-out from the existing computed-f32 predicate;
threshold-adjacent and brute-force pair-set tests pin the behavior. At 1M single-threaded native
with preprocessing, fifteen paired Fibonacci runs reduced whole-build retired instructions by
2.10%, branches by 2.69%, and cycles by 10.26%, with every pair favoring the candidate for all three
counters. Nine 96-bin uniform pairs reduced instructions by 1.96%, branches by 2.41%, and cycles by
7.60%, again with every pair favorable. Portable-codegen Cachegrind at 20k reported 2.08% fewer
instruction references, 2.57% fewer data references, and 3.01% fewer branches; simulated
mispredictions rose 5.13%, a layout-sensitive result not reproduced by the native cycle counters.

The same-cell weld scan now evaluates that exact tangent-component rejection gate eight pairs at a
time on the `wide` backend. Lanes that survive still run the original strict computed-f32 XYZ
predicate, and the scalar tail preserves pair order. At 1M single-threaded native with
preprocessing, nine paired Fibonacci runs reduced whole-build instructions by 2.04% and branches by
3.54%; uniform reduced them by 1.99% and 3.16%. A 2.5M multithreaded Fibonacci guardrail retained
2.07% fewer instructions and 3.58% fewer branches in all seven pairs. On native Windows at 2.5M
multithreaded, fifty phase-timed Fibonacci pairs reduced the exact preprocessing phase by 34.92%
(95% interval 33.49--36.31%, 50/50 favorable); total wall time remained unresolved because the
phase takes only about 7ms of the build.

Packed ring preparation now processes adjacent eight-lane point chunks as a pair. The two chunks
share the query-coordinate SIMD broadcasts, compute both security masks before one combined-empty
test, and still emit chunk A before chunk B; odd chunks and scalar remainders retain the old path.
Relative to the vectorized-weld baseline, 1M single-threaded Linux Fibonacci reduced instructions
by 0.168% and branches by 0.771%, while uniform reduced them by 0.211% and 0.628%; clustered and
mega guardrails also reduced both counters. Isolated directly on main, native Windows 2.5M
multithreaded phase timing reduced `ring_pass` by 6.97% on Fibonacci (95% interval 5.20--8.70%) and
9.28% on uniform (6.80--11.70%). The enclosing packed-kNN phase improved directionally by
1.69--1.95%; uninstrumented whole-build estimates were 0.50% faster on Fibonacci and 0.79% faster
on uniform but remained below the machine's resolution.

On AVX2, the packed ring path extends that microbatch to three chunks, sharing each query broadcast
across 24 candidates while preserving A/B/C emission order. The portable path deliberately remains
two chunks: splitting each `wide::f32x8` across two registers made the triple version execute 0.62%
more instructions inside `prepare_group_directed`, while the gated portable function is byte-for-byte
identical to its baseline codegen. Native 50k Cachegrind reduced that function's instructions by
0.95% and conditional branches by 3.97%; whole-build instructions were effectively flat (+0.035%,
consistent with layout noise) and branches fell 0.59%. At 2M/6T, phase-timed medians reduced
`ring_pass` by 7.2% on uniform (11/12 pairs) and 6.1% on Fibonacci (10/10 pairs). Uninstrumented
whole-build timing was neutral, as expected for a phase accounting for about 4% of the build.

Lazy ring-tail rescans evaluate a non-empty remainder by overlapping the final stored eight-point
chunk and masking away lanes that belonged to the preceding full chunk. Ranges shorter than eight
points retain the scalar path, and the retained high lanes preserve the original ascending slot
order. At 2M/6T native, nine uniform hardware-counter pairs reduced whole-build instructions by
0.074% and branches by 0.185%, with every pair favorable; Fibonacci's lighter fallback incidence
reduced branches by 0.024% in every pair while instructions were nearly neutral (-0.007%). Twenty
phase-timed uniform pairs reduced `ring_fallback` by 14.69% (95% interval 13.51--15.86%, 20/20
favorable). A deterministic 100k single-threaded portable Cachegrind guardrail reduced whole-build
instruction references by 0.152%, conditional branches by 0.178%, and simulated branch misses by
0.806%; inside `ensure_tail_directed_for` the reductions were 12.34%, 13.09%, and 24.90%.

Ordinary and edge-check clipping share one out-of-line small-polygon specialization dispatcher.
Inlining the table into both entry points duplicated about 6.0 KiB of native text and 7.6 KiB of
portable text. The retained wrappers tail-transfer to the shared body; at 1M single-threaded native,
nine Fibonacci pairs reduced whole-build instructions by 0.205% and hardware cache references by
18.62%, while uniform reduced them by 0.255% and 17.11%, with every structural pair favorable.
A same-condition twelve-pair uniform confirmation retained 0.255% fewer instructions and 11.49%
fewer cache references; cycles were neutral at +0.26% with a -0.28% to +0.80% interval. At 2M/6T,
instructions fell 0.207% on Fibonacci and 0.267% on uniform, and cache references fell 8.58--8.79%
in every pair; cycles were neutral. Portable 50k single-threaded Cachegrind reduced instruction
references by 0.172%/0.227% and simulated I1 misses by 19.24%/23.08% on Fibonacci/uniform.
Conditional branches rise about 1% because of the shared transfer. Forcing the two tiny wrappers
back inline reduced that retired branch cost but produced a repeatable 2.94% uniform cycle
regression over fifteen pairs; retain the layout-stable out-of-line wrappers.

A refreshed timing-only census on the 16-core native Linux host confirmed that the remaining clip
cost is broad rather than one missed specialization. At 2.5M, Fibonacci made 18.3M clip attempts:
82.1% changed the polygon, and sizes 3/4/5 accounted for 25.9%/34.3%/23.5%. Uniform made 25.1M
attempts: 61.8% changed, with sizes 3/4/5 at 22.2%/27.4%/22.4%. A 500k clustered run made 19.2M
attempts, 83.6% unchanged, spread most heavily across sizes 5--9+. The retained radius certificate
avoided full dispatch for 1.85M Fibonacci, 1.41M uniform, and 8.12M clustered attempts—respectively
56.4%, 14.7%, and 50.6% of all unchanged results. Native cycle annotation distributed samples
across SIMD distance evaluation, interpolation divides, survivor/metadata writes, radius tracking,
and dispatch overhead. This reinforces the existing individual negative results: no single size,
result class, or assembly sequence now justifies another narrow clipper variant.

The AVX2 small-sort dispatcher keeps its eight-element network out of line. Inlining that leaf at
both dispatch sites made every sort preserve two additional callee-saved registers; outlining it
reduced the dispatcher from 1,538 to 847 bytes and the combined dispatcher/leaf fixed work by
2.44%. At 500k single-threaded, twelve uniform hardware-counter pairs reduced whole-build retired
instructions by 0.036% in every pair while adding 0.021% branches; cycles were neutral. The change
is intentionally AVX2-only because the outlined portable experiment added 0.51% whole-build
branches. Gating restores byte-identical portable `sort_small` codegen.

### Source-pinned performance decisions

Source comments retain the invariant that constrains a local implementation choice and link here
for its historical measurement. This keeps host-specific timing narratives out of production
modules without making the non-obvious code shape look accidental.

- **Numeric backend and grid policy.** The stable `wide` backend matched the retired nightly
  `portable_simd` path within roughly 1–2% on the June 2026 Ryzen 3600 campaign while preserving
  backend fingerprints. On the same reference family, query-grid density 24 beat 16 by 4.8–7.1%
  across 100k, 500k, and 2M uniform inputs. The quiet 2026-06-14 occupancy-feedback sweep placed
  the beneficial `Σocc²/n` crossover near 450: measured 274/331 cases lost 19%/8%, while 536/712
  cases gained 18%/29%. The selected threshold 500 separates all recorded cases; the superseded
  noisy-host value 2000 placed the crossover 4–7× too high.
- **Non-fused arithmetic contract.** The internal `fma` feature was retired after isolated
  measurements showed no dependable throughput benefit. Native Windows 2.5M Fibonacci runs were
  neutral in multithreaded mode and neutral/slightly adverse single-threaded. On an older Ryzen,
  pinned 1M single-threaded counters retired about 2.18% fewer instructions but used about 0.73%
  more cycles. On a Ryzen 5900XT, nine 1M single-threaded pairs favored FMA by 0.51% with native
  codegen and 0.89% when fairly isolated as AVX versus AVX+FMA, but twenty-one 2.5M/16-thread pairs
  were unresolved at -0.51% in both builds (95% intervals -1.34% to +0.33% native and -1.41% to
  +0.39% isolated). Across the latter host, roughly 2.0--2.1% fewer instructions were offset by
  1.2--1.9% higher CPI. Retain one non-fused evaluation order and fingerprint instead of a
  hardware-dependent numerical variant with no runtime dispatch.
- **Pinned hot-path codegen.** Keeping worker setup folded into the shard driver avoided about 1%
  more retired instructions after unrelated cold growth caused LLVM outlining. Verified-cell XOR
  extraction avoided a 1.3% whole-build instruction increase from repeated endpoint membership
  checks. Small-N sorting networks beat `sort_unstable` by about 5% total time at 500k in their
  measured regime, while extracting the shared packed emit sequence out of line added 0.6%
  whole-build instructions; these helpers therefore retain their explicit inline boundaries.
- **Canonicalization and dense-query gating.** The scalar f64-normalize/store pass measured about
  20 ms at 2M single-threaded (roughly 0.5–0.8% of total) and is chunk-parallel by default. The
  dense-cell band plus takeover lost about 13% on the 500k moderate-cluster control, so it remains
  gated on occupancy feedback having rebuilt the grid. The one-percent final-scatter classifier
  lies between the measured spatial-order signals for Fibonacci (about 0.2% of `n`) and
  shuffled/uniform input (about 7%).
- **Shared shell layer schedule.** A shell frontier retains the query-independent BFS cell order
  and layer offsets while consecutive queries have the same start cell. Each query still computes
  its own bounds, resident dots, sorting, clipping, and directed forwarding. Three native
  single-threaded Linux perf pairs reduced instructions/branches by 0.043%/0.129% on 100k
  Fibonacci, 0.066%/0.172% on uniform, 0.392%/0.893% on clustered, 3.048%/5.271% on mega,
  0.655%/1.921% on great-circle, and 0.455%/1.014% on 500k clustered. Every structural pair
  agreed; cycles were noisy except for mega, where all pairs improved by about 3.06%. One-shot
  peak-RSS checks changed by +1.2 MiB on 100k single-threaded mega, +0.3 MiB on great-circle,
  +0.1 MiB on 500k clustered, and +5.6 MiB on default-thread mega. Retain the full discovered
  schedule: the measured memory increase is modest and bounding it would discard the strongest
  reuse regimes.
  A later 12-core guardrail confirmed the decision. Instructions/branches changed by
  -0.158%/-0.350% on 500k Fibonacci, -0.052%/-0.156% on uniform, -0.493%/-1.071% on clustered,
  -2.582%/-4.649% on 100k mega, -0.583%/-1.657% on great-circle, and -2.148%/-4.201% on 500k
  mega. The large mega case added 26.2 MiB peak RSS (about 4.0%) and 3.18% cache references, while
  cache misses were neutral (-0.51%). Its aggregate cycles improved 2.67%, but individual pairs
  ranged from +1.84% to -8.36% on the busy host; retain the optimization on structural counters
  and use a future quiet run only to resolve all-core mega throughput.
- **Defect-local reconciliation.** Scanning a stale cell-index tail could create a phantom
  low-incidence trigger whose acceptance work cost about 13 seconds at 2.5M; topology scans must
  use live cell windows. In-place merge application saved about 382 ms at 2M single-threaded.
  Candidate-local collinear cleanup avoided seconds of whole-vertex work on a three-defect 2.5M
  run, and the localized unpaired-edge scan avoids a roughly 17-second global scan in the same
  scale regime. Strict validation sorts about six million edge-use records at one million cells,
  which is why the available parallel sort owns that stage. Centralizing paired/boundary/overused/
  same-direction classification was neutral in seven 500k single-threaded pairs (mean ratios
  `0.9999960` instructions and `0.9999938` branches), so the typed classifier is retained.
  Reconciliation now returns sparse original cell cycles to the topology safety gate; exact
  key-owner incidence deltas replace the prior whole-diagram atomic rescan when provenance is
  complete, while checked builds compare the result with the exhaustive live-window oracle.
- **Local-rebuild cold path.** Reusing the construction grid replaces an all-generator neighbor
  scan that became minutes-long for thousand-generator closures on dense defects. Returning before
  flatten/clone/validation when no splice occurred removed about 12.6 seconds from a 15-second
  2.5M tail. The borrowed overlay avoids about one second of eager diagram-wide setup at 2.5M, and
  its sorted residual scan replaced an unreserved, per-round ~2E hash map that cost about 1.3
  seconds per round at 1M; the sorted form was roughly ten times cheaper and parallelizable.

### Open optimization queue

These are code-specific hypotheses from a 2026-07 subsystem scan. Each item is an isolated
experiment: preserve its stated semantics, measure the named regime, and move it either into the
measured results above or the retired list below. Do not bundle candidates before attribution.

The broader memory-layout and memory-traffic backlog, including regime-dependent tradeoffs,
hybrid fast-path/fallback designs, and a shared experiment matrix, lives in the repository-only
[`memory-layout-ideas.md`](https://github.com/PeterKlimk/voronoi-mesh/blob/main/docs/research/memory-layout-ideas.md).

Larger changes to scheduling, local-rebuild scope, pathological-work handoff, and repeated-build reuse are
kept in the repository-only, non-authoritative
[`algorithmic performance ideas`](https://github.com/PeterKlimk/voronoi-mesh/blob/main/docs/research/algorithmic-performance-ideas.md)
catalogue.

Recently accepted optimizations:

- **Shrinking-suffix incoming edge-check matching:** partition each cell's already-matched in-bin
  checks to the front, search the shrinking unmatched suffix first, and search the prefix only for
  the handled duplicate-side case. This also makes the final unmatched suffix explicit and removes
  the high-degree consumed-mask spill. The checked edge-check suite and the release high-degree,
  edge-reconciliation, adversarial, and correctness suites pass. At 1M single-threaded native Linux, nine
  interleaved Fibonacci pairs reduced instructions by 0.42% and branches by 1.83% in every pair;
  uniform reduced them by 0.44% and 1.76%, also in every pair. Cycles were directionally favorable
  but noisy. On the quiet eight-thread Intel Mac at 2M, twenty multithreaded pairs were neutral:
  Fibonacci changed +0.10% (95% interval -0.67% to +0.87%, 8/20 favorable) and uniform +0.03%
  (-0.48% to +0.56%, 9/20 favorable). Single-threaded Mac validation at 1M supplied a clear signal:
  thirty Fibonacci pairs were 0.86% faster (95% interval 0.28--1.43%, 25/30 favorable), and thirty
  uniform pairs were 1.63% faster (0.79--2.46%, 26/30 favorable). The repeatable retired-work
  reduction and quiet-host single-threaded wins justify the default change. Treat the neutral Mac
  multithreaded result as evidence that this work is not limiting there, not as a regression; the
  measured intervals rule out a material loss on that host.

  A follow-up avoids swapping a 20-byte check with itself when the next match is already at the
  unmatched-prefix boundary. Cachegrind at 50k Fibonacci reduced whole-build instructions by
  0.17% and data references by 0.48%; 1M native Linux counters retained about 0.15--0.16% fewer
  instructions on Fibonacci and uniform, though the added predictable guard raised total branches
  by 0.45--0.53%. On the quiet Mac at 1M single-threaded, thirty Fibonacci pairs were 1.07% faster
  (95% interval 0.24--1.89%, 25/30 favorable); uniform was directionally 0.38% faster but unresolved
  (-0.30% to +1.06% when expressed as candidate speedup, 18/30 favorable). The clear Fibonacci win,
  reduced retired/data work, and absence of a material uniform loss justify keeping the guard.

- **Fused weld point-view finalization:** preprocessing-enabled grid builds defer the
  global-id-to-slot inverse and slot-ordered `SlotPoint` stream until occupancy feedback selects
  the retained grid, then initialize the required slot stream inside the existing same-cell weld
  loop. The inverse remains absent on the ordinary zero-weld path; actual compaction builds its
  effective cell and slot maps directly in the existing survivor-copy pass, removing the old
  dropped-slot list and original-id follow-up pass. Disabled preprocessing keeps the original
  grid-build path, while a `MergeWithin` radius too large for grid adjacency retains the
  standalone detector and finalizes the selected slot stream separately. Differential tests pin
  the weld-pair set and slot stream against a normal grid, and compacted maps against a fresh grid.
  At 1M single-threaded native Linux with preprocessing, Fibonacci added 0.21% instructions and
  0.24% branches but reduced hardware cache references 5.06%; uniform at 96 bins added 0.17% and
  0.20% while reducing cache references 7.23%. Cycles were unresolved. Cachegrind at 20k
  Fibonacci independently measured 2.22% fewer D1 misses and 10.48% fewer I1 misses, alongside
  0.13--0.14% more instruction/data references. The quiet eight-thread Intel Mac supplied the
  throughput result at 2M: twenty multithreaded Fibonacci pairs were 1.98% faster (95% interval
  1.16--2.80%, 16/20 favorable), and uniform was 2.31% faster (1.57--3.04%, 19/20). A 1M
  single-threaded Fibonacci guardrail was also 0.93% faster (0.31--1.55%, 13/20). This is an
  accepted locality win despite slightly more scalar retired work.

  Making the inverse truly weld-only and rebuilding maps in the survivor pass reduced another
  0.078--0.080% of Linux instructions in every Fibonacci and 96-bin uniform pair, with branches
  essentially unchanged. Cachegrind confirmed 0.06% fewer instruction references, 0.07% fewer
  data references, and 0.86% fewer D1 misses, but recorded 4.4% more I1 misses and 6.2% more
  simulated mispredicts from layout movement. A direct quiet-Mac comparison against the accepted
  fused baseline was neutral-to-favorable: 2M multithreaded Fibonacci was directionally 0.48%
  faster (paired interval -0.23% to +1.18% when expressed as speedup, 12/20 favorable), while
  uniform changed -0.05% (-0.59% to +0.49%, 11/20). Keep the lower-work lifecycle and named build
  modes; the intervals rule out the noisy Linux Fibonacci cycle regression on the outcome host.

- **Fused eager point-view materialization:** when preprocessing is disabled and input order
  requires slot-coordinate materialization, the existing parallel coordinate pass now emits the
  slot-ordered `SlotPoint` stream at the same time. This removes the separate sequential AoS pass
  without widening the direct-scatter or preprocessing-enabled weld paths. On a native Ryzen
  5900XT, fifteen rotated pairs improved 2.5M Fibonacci/16-thread construction by 5.55% (95%
  bootstrap interval 4.26--6.73%, 14/15 favorable) and 1M uniform/16-thread construction by 4.75%
  (2.96--6.55%, 14/15). The 1M Fibonacci single-thread guardrail improved 0.65% (0.38--1.01%,
  14/15). Seven counter pairs confirmed slightly fewer instructions in both thread regimes;
  aggregate multithreaded cycles were not an outcome metric because the candidate deliberately
  replaces serial first-touch with concurrent work. Output fingerprints and topology were
  unchanged.

- **Remove the stale point-to-slot inverse:** the grid formerly allocated and sentinel-filled a
  `u32` per point, then scattered every spatial slot back to its global generator id. No production
  code still read that inverse: weld compaction had evolved to rebuild its surviving maps directly
  in slot order, and ordinary construction released the array before cell building. Removing the
  field and its construction/compaction lifecycle deletes one complete random-write pass and 4
  transient bytes per point while simplifying the grid invariant.

  At 1M single-threaded with preprocessing disabled, seven native counter pairs reduced Fibonacci
  cycles by 1.33% and uniform by 0.69%; retired instructions fell 0.10%/0.08%, cache references
  21.8%/20.1%, and cache misses 2.2%/5.7%. Twenty rotated 2.5M/16-worker pairs improved construction
  by 1.93% on Fibonacci (95% bootstrap interval 0.42--3.40%, 16/20 favorable) and 1.45% on uniform
  (0.53--2.37%, 14/20). The normal preprocessing path already deferred the removed inverse and was
  structurally neutral: pinned 1M instructions changed by about +0.01%, and fifteen 2.5M/16-worker
  timing pairs were unresolved around neutral. Checked weld-compaction tests, the complete release
  suite, and all-target clippy passed.

- **Overlap immutable grid topology at high worker counts:** cube-face neighbors, ring-2 cells,
  cap bounds, and wall planes depend only on grid resolution. At twelve or more Rayon workers and
  at least 16,384 cells, their construction now runs independently while the main build classifies,
  counts, and spatially permutes points. Lower worker counts retain the original sequential schedule:
  a 1M sweep found that overlap lengthened the grid phase 5--9% at four workers and was weak/mixed
  at eight, while the intended 16-worker regime had enough concurrency to hide the independent work.

  On the native 16-core Ryzen at 2.5M without preprocessing, fifteen rotated timing-build pairs
  reduced grid wall time 17.2% on Fibonacci (15/15 favorable) and 14.3% on uniform (15/15).
  Whole-build time improved 3.4% on Fibonacci (95% paired log interval 0.3--6.5%, 12/15) and 1.5%
  on uniform (0.5--2.5%, 12/15); cell construction remained neutral. A separate twenty-pair
  non-instrumented run with normal preprocessing kept uniform favorable at 1.71% (0.47--2.94%,
  14/20) and left Fibonacci directionally 0.59% faster but unresolved. The schedule changes no
  point or topology ordering and the one-thread/old six-core regime never enters it.

- **Reuse immutable topology in `VoronoiWorkspace`:** repeated workspace computations now retain
  reference-counted cube-grid topology for up to two resolutions, covering the ordinary resolution
  plus a possible occupancy-feedback regrid. Point classification, occupancy, spatial permutation,
  and every mutable/dense side structure are still rebuilt per input. Ordinary `compute()` calls
  retain nothing, and `VoronoiWorkspace::clear()` releases both topology and construction scratch.

  On the native 16-core Ryzen at 2.5M without preprocessing, eight rotated processes with five
  iterations each (discarding each process's cold first iteration) reduced steady-state grid wall
  time 14.6% on Fibonacci and 10.0% on uniform, with 31/32 samples favorable in both cases. Whole
  builds improved 2.1% on Fibonacci (95% log interval 0.2--4.0%) and 1.1% on uniform (interval
  -0.1--2.3%). Fifteen-pair one-shot controls were unresolved around neutral. Seven pinned 1M
  one-thread counter pairs found no added work from shared topology ownership: instructions changed
  by +0.005%/-0.043% and branches fell about 0.11% on Fibonacci/uniform. Workspace and non-workspace
  outputs remain exact because both consume the same immutable tables.

- **Cap high-core grid construction at eight chunks:** worker-local counting and scatter use two
  full-cell rows per chunk, while the prefix phase visits every row for every grid cell. At 16 or
  more Rayon workers, grid construction now uses eight input chunks; lower-worker schedules are
  unchanged. This reduces transient histogram/cursor storage by about 6.3 MiB at resolution 131
  with 16 workers and about 18.9 MiB relative to 32 rows, while leaving the remaining workers free
  for independently scheduled topology work.

  Repeated 2.5M/16-worker workspace runs reduced grid time 8--10% versus one chunk per worker; the
  prefix phase fell roughly 34--47%. Production non-timing binaries were neutral overall at 16
  physical workers, where grid construction is only about 5% of the build. At the machine's normal
  32-thread Rayon setting, fifteen no-preprocess pairs improved Fibonacci by 0.96% (95% log interval
  0.15--1.81%) and uniform directionally by 1.43% (interval -1.31--3.95%). Twelve normal-preprocess
  pairs improved uniform by 3.14% (interval 1.83--4.44%, 11/12 favorable) and Fibonacci
  directionally by 0.88% (interval -0.76--2.59%). A 12-worker guardrail showed that eight chunks
  can under-parallelize Fibonacci scatter, which is why the existing lower-worker schedule is
  preserved rather than applying a global cap.

  A follow-up crossover sweep covered 100k, 250k, 500k, 1M, and 2.5M at both 16 and 32 workers.
  The prefix phase improved at every point; aggregate grid time improved in 18/20
  size/distribution/worker regimes. At 32 workers, grid improvements were 12--23% at 100k and
  generally 9--21% across uniform sizes. Production whole-build results showed no repeatable
  small-input loss, so an additional grid-size threshold would discard measured small-scale wins
  without protecting a demonstrated adverse regime.

- **Recycled per-bin cell-build contexts:** the default keeps about twice as many spatial bins as
  workers for load balance, but each bin formerly allocated and zeroed its own full-input
  attempted-neighbor stamp table. Parallel builds now recycle the complete cell-build context
  after a bin finishes, bounding live/full initialization work by executing workers while retaining
  the existing bin count and exact candidate behavior. This is distinct from the retired lazy-stamp
  experiment: ordinary candidate insertion remains one direct spatial stamp access, with no added
  per-cell vector or transition branch. The one-thread path deliberately bypasses pooling.

  On a native 16-core Ryzen 5900XT at 2.5M with preprocessing disabled, ten rotated pairs improved
  Fibonacci by 5.80% (95% bootstrap interval 4.86--6.71%, 10/10 favorable) and uniform by 4.97%
  (2.83--7.14%, 9/10). A separate 15-pair exploratory form produced the same direction before the
  scalar bypass. Five-run Fibonacci counters attributed the retained form to 3.7% fewer cycles,
  0.7% fewer instructions, 5.7% fewer data-TLB misses, and 10.5% fewer page faults. Twelve pinned
  1M single-threaded uniform pairs favored the bypassed candidate by roughly 0.3%; Fibonacci was
  previously neutral-to-favorable. Correctness and checked fingerprint-support suites passed.

- **Packed-kNN scratch follows the recycled context:** packed query preparation owns reusable
  per-query key vectors. They formerly survived groups within one spatial bin but were dropped and
  regrown when a worker took its next bin. Moving that scratch into the existing build-context pool
  extends its lifetime only to the next task on the same build; it does not add another live
  context or increase the ordinary high-water memory model.

  Heaptrack at 500k uniform/16 threads measured 67,232 allocations before and 38,476 after
  (-42.8%), with temporary allocations falling from 12,323 to 3,676. Peak heap was effectively
  flat (219.34 versus 220.28 MB). Seven pinned 1M single-threaded pairs improved Fibonacci cycles
  by 0.69% and uniform by 0.83%. At 2.5M/16 threads, seven pairs improved Fibonacci cycles by
  1.00%; uniform was neutral (+0.04%). Cachegrind at 20k Fibonacci measured 0.32% more instruction
  references and 0.87% more simulated conditional mispredicts, consistent with the native counters'
  roughly 0.3% instruction increase. The measured cycle wins therefore come from avoided allocator
  calls and growth copies, not reduced geometric work; the flat uniform all-core result is the
  retained guardrail.

- **Parallel cell-metadata prefix:** final assembly formerly gathered every shard-local cell count
  and emitted the global cell prefix in one serial generator-order loop. Large parallel builds now
  use a two-level scan: disjoint chunks gather counts once while writing chunk-local prefixes, a
  tiny serial scan computes chunk bases, and a contiguous parallel pass applies those bases. The
  existing loop remains the one-thread and small-input path. Unlike the retired adaptive
  cell-count scatter, this removes the serial prefix bottleneck instead of adding a locality mode
  in front of it.

  On a native 16-core Ryzen 5900XT at 2.5M without preprocessing, fifteen rotated pairs improved
  Fibonacci by 3.89% (95% bootstrap interval 1.22--7.46%, 12/15 favorable) and uniform by 6.87%
  (4.55--9.40%, 14/15). With normal preprocessing, ten pairs improved Fibonacci by 3.34%
  (2.67--3.99%, 10/10) and uniform by 4.58% (2.36--6.83%, 9/10). The attributed 1M prefix phase
  fell from roughly 4--5ms to about 1ms; the candidate's 2.5M phase was about 2ms. Aggregate
  multithreaded cycles rose 0.6% because the latency reduction deliberately spends concurrent CPU,
  while instructions were neutral and branches fell 0.7%. Pinned 1M single-thread counters were
  neutral in task-clock/cycles (+0.07%/+0.05%) with 0.17% fewer instructions and 1.1% fewer
  branches. Strict 100k validation passed.

Spatial-order materialization policy candidate:

- **Adaptive final index scatter:** phase attribution showed that the apparent final typed point
  conversion was already an allocation-preserving ownership transfer. An isolated cell-metadata
  conversion removal lowered retired work but hurt ordinary uniform throughput and remains retired.
  The material attainable component was instead the shard-local to global cell-index scatter.
  Generator-order traversal gives sequential destination writes but can jump among shard source
  streams; shard-order traversal gives sequential source reads but scatters the final writes. The
  assembly path now samples up to 32 adjacent generator-id pairs per shard and selects shard order
  when their mean absolute delta exceeds 1% of the input size. Fibonacci and cubed-sphere inputs
  remain generator ordered; uniform, clustered, and mega inputs select shard order. Public cell
  order, contiguous storage, and index values are unchanged.

  At 2M on the eight-thread Intel Mac, the attributed index phase was about 16--18ms on Fibonacci
  and 42--48ms on uniform, versus roughly 10--12ms for vertex concatenation. Twenty interleaved
  multithreaded pairs left Fibonacci neutral (645.8ms versus 645.2ms median) and improved uniform
  by 1.12% (paired 95% interval 0.58--1.66%, 17/20 favorable). At 1M single-threaded, uniform
  improved 2.72% (1.25--4.17%, 18/20); Fibonacci was unresolved at about 0.3% slower by median and
  a paired interval spanning -2.52% to +0.34% speedup. Clustered was directionally 1.58% faster and
  mega neutral. A separate 12-pair 2M multithreaded no-preprocess guardrail was neutral/favorable:
  Fibonacci moved from 629.6ms to 626.8ms and uniform from 790.9ms to 785.7ms. Linux fixed-work
  counters reduced retired instructions by 0.25% on Fibonacci and 0.55% on uniform in every one of
  nine pairs. This is a locality trade selected from the input's existing spatial-order signal,
  not a public-format change.

- **Adaptive grid coordinate materialization:** the same one-percent mean-delta policy now also
  classifies the grid-build boundary. The grid samples 32 adjacent input point-to-cell addresses.
  A correlated, already cell-major input retains the fused pass that scatters ids and XYZ together.
  A scrambled input first scatters only ids, then traverses spatial slots to gather each input point
  and write the three coordinate arrays sequentially. This adds a pass and about 0.19--0.21% whole-
  build Linux instructions, but the final readable implementation reduced 1M uniform cycles by
  8.27%, cache references by 16.11%, and cache misses by 26.67% in all five confirmation pairs;
  Fibonacci cycles were 0.80% lower with four of five pairs favorable and cache traffic neutral.

  The quiet eight-thread Intel Mac supplied the outcome signal against the accepted output-scatter
  baseline. In the final 20-pair 2M multithreaded build, Fibonacci was neutral (643.2ms versus
  642.8ms median), while uniform improved from 790.4ms to 769.8ms; the uniform geometric ratio was
  0.9754 with a 0.9685--0.9824 interval and all 20 pairs favorable. A focused 30-pair Fibonacci
  run on the pre-readability form was likewise neutral (ratio 0.9958, interval 0.9897--1.0020). The
  final code was also neutral single-threaded at 1M (Fibonacci 1003.7ms versus 1008.5ms; uniform
  1264.2ms versus 1261.0ms). With preprocessing disabled at 2M multithreaded, Fibonacci remained
  neutral (643.7ms versus 650.0ms) and uniform improved from 802.5ms to 777.8ms. The cell-major
  `cubed` guard retained the fused path and was neutral over 20 pairs. This makes the final-scatter
  choice part of one cross-pipeline address-order policy rather than a distribution-named assembly
  exception.

Policy audit: these are the two largest measured identity/spatial conversion boundaries, not every
possible use of the signal. The next credible candidate was final cell metadata: generator-order
prefix construction writes globally ordered cells sequentially but gathers counts from shard-local
streams. A candidate can scatter counts in shard order and then run the required sequential prefix.
Local 500k attribution measured the current prefix phase at 6.1ms for Fibonacci and 7.8ms for
uniform, giving it a smaller but nontrivial ceiling. Bin-assignment inverse maps are another
boundary, but changing their scattered global writes requires an added pass or inverse map and has
a lower prior. The prior point-to-slot fusion is already retired, and the slot-native construction
core has no competing identity-order traversal left to select.

The abstraction follow-up kept the conceptual split between measurement and boundary-specific
choice, but rejected a generalized `LocalitySample` implementation. Distributed short sample
blocks fixed a periodic alias in the old fixed-position sampler, while moving the grid probe to
actual CSR slot offsets improved the metric itself; together the larger sampling path perturbed hot
scatter codegen by a repeatable +0.228% whole-build instructions. A flattened generator-to-shard
source-distance probe also failed to distinguish Fibonacci from uniform. Keep the compact
site-specific samplers and the pre-prefix numeric-cell proxy. The shared classifier now adds only a
one-element floor for domains below 100. Rebuild telemetry reports the retained grid and exports the
decision through `TIMING_KV`.

That minimal retained form reduced 1M Linux instructions by 0.162% on Fibonacci and 0.147% on
uniform, and branches by 0.382%/0.339%, in all three confirmation pairs. On the quiet Mac at 2M
multithreaded it was neutral: Fibonacci medians were 641.4ms/641.0ms and uniform's paired ratio was
0.9972 with a 0.9885--1.0059 interval (12/20 favorable).

The adaptive cell-count/prefix candidate was closed neutral. Its one-byte form scattered counts in
shard order before the mandatory sequential global prefix when the existing policy selected shard
order. At 1M single-threaded Linux, instructions and branches were effectively unchanged
(-0.008%/-0.013%); cycles, cache references, and misses were directionally 1.47%, 8.18%, and 5.28%
lower over five pairs. The quiet-Mac outcome gate did not confirm a throughput win: 20 paired 2M
multithreaded uniform rounds produced a 0.9985 ratio with a 0.9906--1.0065 interval and 11/20
favorable, while Fibonacci was neutral. Retain the fused generator-order cell-prefix loop rather
than another A/B materialization path with no retired-work or outcome benefit.

The grid-build prefix was subsequently parallelized without changing count or scatter order. For
large parallel grids (at least 16,384 cells), workers first sum the existing per-chunk histograms by
cell, a short serial scan produces global offsets, and workers then fill disjoint cell columns of
the per-chunk cursor rows. The original fused loop remains the explicit path for one thread and
small grids. On the native 16-core Linux bench at 2.5M, ten rotated pairs per case improved
Fibonacci by 3.06% without preprocessing (95% bootstrap interval 1.80--4.38%, 9/10 favorable) and
4.01% with preprocessing (2.94--5.05%, 10/10). Uniform improved 2.33% without preprocessing
(1.10--3.56%, 9/10) and 2.86% with it (1.64--4.07%, 9/10). The isolated prefix phase fell from
about 5.2ms to 1.5--1.8ms at 1M and measured 3.5--4.0ms at 2.5M. A pinned one-thread 1M guardrail
never executes the parallel helper: twelve pairs improved Fibonacci 0.71% and regressed uniform
0.25%, a small net improvement across the two distributions. Strict 100k Fibonacci and uniform
validation passed.

The default spatial-shard target now increases from 2x to 4x workers at 16 or more workers, while
retaining the 96-bin ceiling and the existing 2x policy below that threshold. Cube-face
quantization means the 16-core case moves from 54 to 96 actual bins. This is deliberately not a
blanket multiplier increase: on the same host, 6-thread targets of 12 and 24 both quantized to the
same 24-bin layout, while an 8-thread jump from 24 to 54 actual bins was slower. The high-core
change provides enough construction tasks to reduce the Rayon tail while accepting more cross-bin
assembly work. At 2.5M on the native 16-core Linux bench, ten rotated no-preprocessing pairs
improved Fibonacci by 2.29% (95% bootstrap interval 0.94--3.71%, 8/10 favorable) and uniform by
3.05% (1.85--4.24%, 9/10). With normal preprocessing, Fibonacci improved 2.92%
(2.18--3.70%, 10/10) and uniform 1.74% (0.32--3.17%, 6/10). Timing attribution reduced cell
construction by about 15ms/29ms on Fibonacci/uniform while adding about 5--6ms of dedup work.
Peak RSS remained within roughly 2MiB around a 704--710MiB envelope. At eight threads, where the
selected layout is unchanged, uniform wall time was neutral and seven Fibonacci counter pairs
were structurally neutral (slightly fewer instructions and branches), rejecting a small noisy wall
regression as changed work.

A post-optimization scaling census first extended the same 96-bin layout down to twelve workers.
After whole shards began entering a largest-first dynamic queue, a quieter 16-core follow-up found
that the useful crossover had moved to nine workers. At 2.5M without preprocessing, twelve paired
rotated comparisons of 96 versus 24 shards improved Fibonacci/uniform by 3.15%/5.33% at nine
workers, 7.51%/7.88% at ten, and 10.55%/11.31% at eleven; all corresponding 95% bootstrap
intervals excluded neutral. At eight workers, six-pair guards left Fibonacci neutral and made
uniform 2.01% slower. Timing attribution showed why: aggregate shard work stayed similar, but at
nine and ten workers the 96-bin layout shortened cell construction by roughly 30--75ms and sharply
reduced the longest shard task; at eight workers the 24-bin layout already supplies exactly three
tasks per worker, so extra shard boundaries only add overhead. The default therefore retains 2x
workers through eight and uses the 96-bin cube layout at nine or more: this is a minimum queue-depth
rule, not a hardware-size cutoff.

Severely imbalanced default layouts now refine adaptively from 96 to 216 spatial bins. The gate
uses only exact integer population counts: the heaviest coarse bin must hold at least seven times
the mean population, the fine layout must reduce that absolute maximum by at least 10%, and the
fine local-id capacity must remain sufficient. Explicit `VORONOI_MESH_BIN_COUNT` overrides disable
the policy. Ordinary inputs get the maximum coarse population from the assignment already being
built and perform no additional grid scan; only the overloaded cases scan the fine population and
rebuild assignment when accepted.

A timing-only 96-bin census found 92--94% task efficiency on 2.5M Fibonacci/uniform, but only 61%
at 16 workers and 40% at 32 on 1M gradient. The gradient closing wave occupied 65--84% of cell
construction, and coarse generator count predicted bin elapsed time with correlation 0.996--0.998.
Fixed 216-bin tests improved gradient by 15--18% and bimodal by 9--12%, but regressed clustered by
4--6% and great-circle by up to 26%, motivating both halves of the adaptive gate rather than a
generic variance threshold.

With the staged production gate and normal preprocessing, ten native pairs improved 1M gradient
by 7.50% at 16 workers and 11.05% at 32, every pair favorable. Bimodal 500k was neutral at 16 and
5.15% faster at 32. Without preprocessing, twelve 32-thread pairs improved gradient by 10.13%
(95% log interval 8.46--11.87%) and bimodal by 3.19% (2.14--4.28%); the 16-thread results were
directionally 0.93% and significantly 2.60% faster. Mega, Fibonacci, and uniform controls retained
the coarse layout and were unresolved around neutral. Great-circle also retained 96 bins; a large
favorable code-placement movement was not attributed to the adaptive mechanism.

A broader scheduling census tested the cube-face layouts with 6, 24, 54, 96, 150, and 216 total
shards across 4--32 workers and both ordinary and adversarial distributions. Population-only
makespan models were useful for smooth density gradients but did not safely replace the retained
policy. In particular, the odd 3x3 and 5x5 per-face layouts could place a concentrated cap wholly
inside one central shard rather than across a boundary; `mega` then took roughly three times as
long as with the even layouts. Great-circle was the decisive model counterexample: at 32 workers
the 216-shard population schedule looked substantially better than 96, but measured total time was
slightly worse. Shard layout changes construction work and ordering as well as task balance, so a
single predicted-efficiency threshold was rejected instead of fitting another machine-specific
rule. Timing output retains the selected shard count, maximum shard population, summed shard-task
time, and maximum shard-task time for future scheduling diagnosis.

Construction now admits whole-shard tasks in descending generator-population order through a
small dynamic Rayon queue, then restores completed results to canonical shard-id order before
assembly. This preserves the one-writer-per-shard guarantee and generator order within each shard;
it changes scheduling only, without adding ownership boundaries. Against the preceding native
production binary, eight rotated no-preprocessing pairs improved 1M gradient by 11.64% at 16
workers and 4.26% at 32. At 16 workers, 500k bimodal improved 6.99%, mega 2.45%, and 100k
cube-vertices 3.55%, while great-circle was neutral; the corresponding 32-worker controls were
neutral to 1.85% faster. Ten-pair 2.5M Fibonacci/uniform guardrails across 8, 12, 16, and 32 workers
were neutral to favorable overall. An apparent 12-worker uniform regression did not reproduce in a
focused 30-pair confirmation (2.37% directionally faster, interval spanning neutral). Strict 100k
validation passed for Fibonacci, gradient, and great-circle at 12 and 32 workers.

The adaptive 216-shard refinement was rechecked after largest-first scheduling because both target
the closing construction tail. Ten rotated native 1M pairs against forced 96 shards retained clear
wins: gradient improved 8.39% at 16 workers and 11.20% at 32, while bimodal improved 4.79% and
3.57%; all four 95% bootstrap intervals excluded neutral. Timing runs confirmed that the default
selected 216 shards, reduced maximum shard population and longest-task time, and paid modestly more
aggregate task and overflow work. Fibonacci, uniform, mega, great-circle, and clustered 1M controls
all retained 96 shards. Largest-first therefore complements rather than supersedes the gated finer
ownership layout; unconditional 216 shards remains unjustified.

Cross-bin overflow resolution now groups records by unordered shard pair before sorting and
matching. The opposite-bin byte fits existing padding in both the per-cell and assembled records,
so neither representation grows. Pair-local sorting shortens each sort and, more importantly,
keeps the serial endpoint-patching pass on the same two shard outputs instead of jumping globally
in generator-key order. At 96 bins and 2.5M sites without preprocessing, the post-PBO native
16-core host emitted about 133k records. Three final-form timing pairs reduced sorting from roughly
2.8--3.0ms to 1.8ms; pair-local match/patch samples were also favorable but noisier, ranging from
5.0--6.5ms versus 6.6ms on Fibonacci and 6.2--8.3ms versus 8.0--8.6ms on uniform. Twenty-four
ordinary-build pairs improved Fibonacci by 1.61% (approximate 95% paired interval 0.63--2.58%,
18/24 favorable) and uniform by 0.83% (-1.14--2.76%, 19/24). Twelve pinned one-thread 1M pairs
were neutral/favorable at 0.13% and 0.20%. Checked one- and 16-thread fingerprints and the full
release suite passed.

The post-PBO 16-core Zen 3 host was also profiled at 16 physical workers versus all 32 SMT
threads on the retained native 2.5M path. Work was essentially fixed: 32 threads retired only
about 1.6% more instructions on both Fibonacci and uniform. Aggregate cycles nevertheless rose
54.5%/50.8%, reducing IPC from 1.81/1.63 to 1.19/1.10. Cache misses rose a much smaller
13.1%/11.3%; data-TLB misses rose 78.3%/41.3%. Miss sampling attributed translation and cache
traffic broadly to grid coordinate materialization/prefixes, bin assignment, final index scatter,
edge collection/emission, and the cell driver rather than one dominant clipping load.

Zen dispatch-token counters supplied the stronger classification. From 16 to 32 workers,
store-queue resource stalls grew about 2.36x/1.93x and integer-scheduler stalls about 9.1x/25.6x
on Fibonacci/uniform, while load-queue stalls grew only about 1.24x/1.75x. These multiplexed
counters are diagnostic rather than exact accounting, but they reject a load-latency explanation:
SMT is primarily increasing shared backend/scheduler pressure. Flat cycle profiles remained broad
across clipping, the cell driver, batch clipping, edge collection/emission, and scatter. Partial-SMT
16/20/24/28/32 sweeps were timing-noisy and produced no stable intermediate optimum. Retain the
ordinary Rayon policy and do not use this profile to justify phase-specific thread pools.

Untried probes and candidates (2026-07-17 triage; each begins with a cheap measurement gate):

- **TLB / huge-page probe (native Linux only):** the measured memory wall is dedup and grid build
  streaming large arrays with scattered access; every locality experiment so far targeted cache
  lines, none TLB reach. WSL reported millions of data-TLB misses at 2M, but its page and memory
  behavior is not representative enough to gate an allocator/THP change. Repeat the counter probe
  on native Linux before trying `madvise(MADV_HUGEPAGE)` on slot arrays, shards, or grid storage.
- **Dense-gated directional termination certificate:** the archived certificate (d9d0975) closed
  negative on fib/uniform (+3.9–4.5% instructions), but the residual dense-cap cost after
  band-prune is certificate depth, and mega runs ~11x candidate inflation versus ~1.6x normal.
  Gate it on `grid_rebuilt` exactly like band-prune so the normal path is untouched, and A/B on
  mega/great-circle only. The gated exact-preflight follow-up is now closed negative. Its timing
  oracle found a real ceiling at 500k mega (2.79M potentially skipped clips, 5.9% of examined
  candidates), while great-circle exposed only 0.29%. A minimal production prototype activated
  only when the dense side index existed and only with at least four batch candidates remaining.
  Even at 100k mega, where the oracle ceiling was higher at 8.6%, seven native 16-core pairs added
  about 0.5% retired instructions and cycles were worse in five pairs. Exact all-vertex preflight
  repeats the dominant clip classification work; the avoided mutation bookkeeping cannot repay it.
  Do not promote the much larger 64-direction support cache without a new free support statistic.
- **Cell-interleaving ILP probe:** software-pipelining 2–4 independent cells through the scalar
  clipper only pays if the clip loop is load-latency-bound. The current taxonomy says cell
  construction is compute-bound. The native 16-versus-32-worker gate found sharply increasing
  integer-scheduler and store-queue pressure but much smaller load-queue growth, with no single
  memory-latency hotspot. The gate is closed negative: interleaving more independent cells would
  increase the already-constrained in-flight backend state. Degree-bucketed scheduling remains a
  companion to this rejected item only — Morton ordering was already neutral, so it is not worth
  trying alone.

Promising workload-specific experiments:

- **D3 — compact high-degree consumed flags:** replace the per-cell byte vector above 64 incoming
  checks with reusable bit words. First instrument activation frequency; ordinary cells should stay
  on the existing inline mask.

Assembly/live-dedup swarm backlog (2026-07-13):

- **Flatten per-local edge-check queues only as a memory redesign:** `Vec<Vec<EdgeCheck>>` pays a
  `Vec` header per local generator. A node arena plus head/tail arrays could reduce empty-queue
  metadata, but it loses the current zero-copy transfer and may add traversal/copy work. Require
  queue-count telemetry and preserve exact directed enqueue order, mismatch origins, and high-degree
  behavior before prototyping.
- **Deduplicate reconciliation work by unresolved edge key:** defect inputs can report multiple origins for
  one edge. A reconciliation-only unique-key view may avoid repeated work while retaining the full
  diagnostic origin list. Existing large probes found only a few mismatch records, so this remains
  a cold-path robustness idea rather than a production-speed candidate.

Lower-confidence cleanup candidates, to attempt only with structural counters or activation data:

- Unroll weld wall-proximity tests without changing f64 arithmetic or pair-budget behavior.
- Precompute standalone large-`MergeWithin` wall proximity; this does not affect default welding.
- Consider unchecked endpoint/owner-bin access only after a measurable bounds-check cost and a
  complete invariant audit; a sub-noise win does not justify converting a panic into possible UB.

### Retired experiments

Do not broadly retry these without a materially different design or workload:

- **Uninitialized bin-assignment inverse arrays:** native 2.5M timing on the 16-core Ryzen showed
  that cell construction still scales about 11.1--11.3x from one to sixteen workers, while grid
  build scales only 3.8--4.0x and dedup 3.5--4.3x. From eight to sixteen workers, dedup rose from
  31.8 to 35.3ms on Fibonacci and 32.9 to 38.6ms on uniform, but the complete profile did not show
  a DRAM-bandwidth wall: DRAM-sourced fills were flat-to-lower, while cross-CCD fills and aggregate
  cycles rose. An eight-worker affinity control confirmed a regime tradeoff rather than a universal
  communication bottleneck: splitting workers across the two L3 complexes slowed Fibonacci by
  3.25% geometrically but improved uniform by 4.68%.

  Call-chain sampling attributed part of the growing `memset` share to `assign_bins`, where
  point-count-sized `generator_bin` and `generator_layout` sentinel arrays are subsequently
  overwritten through the grid permutation. A release-only uninitialized-capacity prototype kept
  debug sentinels and published lengths only after complete initialization. At 1M single-threaded,
  seven counter pairs reduced cycles by about 0.36--0.56% and hardware cache references by
  17.7--19.6%, but repeatably added 0.41%/0.67% instructions on Fibonacci/uniform. Fifteen rotated
  2.5M/16-worker pairs were unresolved: both distributions were directionally about 0.5% faster,
  with candidate/base intervals of 0.9854--1.0039 and 0.9837--1.0061 and only 7/15 and 8/15
  favorable. Do not add an unsafe initialization invariant for this sub-noise result. The broader
  profile also found no hidden lock hotspot: clipping, cell emission, and edge-check collection
  remained dominant at both worker counts; the residual scaling loss is distributed across lower
  all-core frequency, cell work, grid construction, and final assembly.

- Extending recycled per-bin state beyond `CellBuildContext` to retain packed-kNN and edge-emission
  scratch was neutral/noisy in 2.5M throughput and adverse structurally. Seven-run Fibonacci
  counters increased cycles 0.39%, instructions 0.09%, cache references 9.3%, data-TLB loads 4.9%,
  and data-TLB misses 2.9%. Recycle only the full-input attempted-neighbor table; allowing the
  smaller scratch allocations to die between bins preserves better cache behavior.

- Parallel sorting of cross-bin overflow handles had only a roughly 2ms phase ceiling at 2.5M;
  serial matching and patching consumed another 4.8--5.6ms. Twenty rotated 16-core pairs split by
  distribution: Fibonacci improved 0.67% (95% bootstrap interval 0.01--1.28%, 14/20 favorable),
  while uniform changed -0.27% (-1.47--0.89%, 9/20). Seven-run Fibonacci counters added 0.28%
  instructions and 6.5% cache references for only 0.19% fewer cycles. Retain the serial unstable
  sort; parallelizing the matcher requires a different patch representation and must justify that
  larger redesign independently.

- Unconditional high-core bin scheduling did not improve on the retained 96-bin coarse policy. A timing-only
  per-bin census measured exposed construction tails of about 15ms at 2.5M Fibonacci, 12ms at
  2.5M uniform, and 18ms at 500k clustered. Globally sorting bins by generator count to start
  predicted-heavy work first destroyed spatial/context locality and slowed all three timing runs
  by roughly 15--20%. Raising the cube-face ceiling to 216 bins also failed the ordinary-workload
  gate: ten rotated 96-versus-216 pairs made Fibonacci 0.58% slower (only 2/10 favorable) and were
  neutral on uniform (+0.07%, 4/10 favorable). Retain spatial bin order; finer task splitting must
  be gated on a measured, reducible population imbalance so ordinary inputs do not pay its extra
  cross-shard boundaries. The later adaptive 216-bin policy above satisfies that narrower gate.

- Fusing point-to-cell classification with each worker's grid histogram removed a complete reread
  of the temporary cell-id array, but made the grid builder slower. In ten alternating native
  16-core pairs at 2.5M without preprocessing, `knn_build` regressed by 5.94% on Fibonacci
  (approximate 95% paired interval 1.74--10.32%) and 7.93% on uniform (3.76--12.26%). Whole-build
  time remained unresolved (-0.86% and +0.81%, respectively). Interleaving scattered histogram
  increments with coordinate projection therefore costs more than the separate regular reread,
  and the fused implementation also required unsafe parallel initialization. Retain the two-pass
  classification/count structure.

- Assigning scrambled-input cell-index backing spans in shard order made both sides of final index
  assembly sequential while leaving the public cell array in generator order. On the native
  16-core host at 2.5M uniform, the index-scatter phase fell 47.6% and the complete dedup phase fell
  13.7% in all ten timing pairs. Production builds improved uniform by 1.87% over twenty pairs
  (approximate 95% paired interval 1.06--2.68%, 18/20 favorable), by 2.25% with preprocessing, and
  by 1.52% at eight threads. Initial default-codegen arrangements appeared to regress the inactive
  correlated-input path: the final outlined/cold form made 16-core Fibonacci 1.87% slower
  (0.37--3.39%, only 3/12 favorable), while scalar variants ranged from roughly 0.3--0.7% adverse.
  A one-codegen-unit causal control removed that signal: twelve alternating pairs left 1M
  single-threaded Fibonacci neutral (-0.09%, interval -0.22--+0.05%) and improved 16-thread 2.5M
  Fibonacci by 1.71% (0.66--2.75%, 10/12 favorable). The corresponding uniform controls improved
  1M single-threaded time by 0.48% (0.22--0.75%, 11/12) and were directionally 2.22% faster at
  16 threads despite one system timing spike widening the interval. This identifies the earlier
  inactive-path movement as code placement rather than executed work, so the adaptive span layout
  is retained. Checked one- and 16-thread fingerprints preserved the semantic topology hash; the
  representation hash may differ because backing-span order is intentionally internal.

- **Permutation-boundary two-pass scatter:** staging each cell as its generator id plus
  already-globalized index payload, then filling cache-sized destination windows, passed its
  isolated 2M uniform phase gate only after increasing the partition from the proposed 64 windows
  to 128. The 128-window form reduced seven-pair all-core cycles by 1.60% at 1M uniform and 2.19%
  at 2M uniform, but added about 0.4--0.5% instructions and 0.7--0.8% branches across ordinary
  regimes. It regressed 2M Fibonacci cycles 1.88%, `cubed` cycles 0.58--0.75%, and 1M clustered
  cycles 0.98%. This is the predeclared correlated-control falsifier, not a case for a fourth
  distribution-sensitive scatter mode. The prototype and private switch were removed; the full
  phase sweep and counter matrix are recorded in the repository-only
  [`permutation-boundary-scatter-idea.md`](https://github.com/PeterKlimk/voronoi-mesh/blob/main/docs/research/permutation-boundary-scatter-idea.md#experiment-result).
- Implementing `LiveCellLayout` checked spans with `slice.get(start..end)` added a redundant range
  validity branch to clean-path reconciliation traversal. Seven interleaved 500k single-threaded
  Fibonacci pairs showed +0.1337% instructions and +1.6620% branches. Preserve the accepted
  explicit cell-bound/end-bound checks followed by ordinary slicing; that form improved
  instructions by 0.0262% with neutral branches while retaining typed malformed-layout errors.
  Threading that accepted view through the shared-edge segment reader was subsequently neutral
  (-0.000097% instructions, -0.000004% branches) and reduced the executable file by 48 bytes. The
  adjacent localized duplicate-key BFS was also neutral (+0.000153% instructions, +0.000209%
  branches), with identical aggregate section sizes and a 32-byte-smaller executable. Its later
  checked-only structural audit leaves release `.text`, `.rodata`, and unwind sections
  byte-identical to the parent, so it requires no runtime counter gate.
- Replacing `cell_spans_differ`'s four raw slices with two `LiveCellLayout` values originally
  perturbed clean-path codegen despite making the executable eight bytes smaller. Default,
  never-inline, and always-inline forms all repeated approximately +0.1597% instructions and
  +1.6620% branches in every one of seven pairs, so the first implementation was reverted. A
  2026-07-20 retest after the surrounding reconciliation code changed retained the typed boundary:
  the changed diagnostic-rebuild function is unreachable under the ordinary in-place benchmark,
  default-build movement varied from +0.0996% instructions/+1.3600% branches on Fibonacci to
  +0.0076%/+0.0731% on mega with no cycle regression, and a `-C codegen-units=1` control produced
  byte-identical executable code and neutral counters. The historical clean-path signal was a
  codegen-partition/layout artifact rather than work added by the comparison.
- Carrying `LiveCellLayout` through the localized unpaired-edge scan originally reproduced the same
  optimizer cliff. Typing the entry, localized scan, partner lookup, and debug oracle produced
  +0.1598% instructions and +1.6619% branches; retaining the raw entry and typing only the internal
  family produced +0.1600% and +1.6625%. Both were initially reverted. A 2026-07-20 retest after
  surrounding reconciliation changes was neutral across Fibonacci, uniform, clustered, and mega:
  instruction means ranged from `0.999995289` to `0.999999804`, and branch means ranged from
  `0.999997622` to `1.000000368`. The whole paired reader family is now retained.
- Introducing `LiveCellLayoutMut::rewrite_and_shrink` for the defect-only collinear-drop mutation
  reproduced the same cliff despite full inlining and an unchanged outer reconciliation signature.
  Seven pairs averaged +0.15987% instructions and +1.66186% branches, with every pair regressing
  and no context switches or migrations. The helper and its mutable view were reverted; keep this
  small prefix-write/count-shrink operation flattened until surrounding codegen changes materially.
- Passing one overflow-safe `LiveCellLayout` through the effective-array validator and its private
  parallel scan originally reproduced the optimizer cliff (+0.129% instructions, +1.360%
  branches) even though the gate was inactive, so three forms were reverted. A 2026-07-20 retest
  after surrounding codegen changed retained the full boundary: default instructions moved
  +0.012% to +0.031% across Fibonacci, uniform, clustered, and mega with neutral branches and no
  adverse cycle signal; a `-C codegen-units=1` Fibonacci control was neutral at roughly three ppm.
  The remaining default displacement is codegen-partition noise rather than validation work.
- Replacing duplicated fail-fast validation strings with a 13-variant typed issue taxonomy
  originally reproduced the optimizer cliff and was reverted. The current retest is retained:
  default Fibonacci/uniform instructions fall about 0.12% while branches rise about 0.12%,
  clustered is closer to neutral, and mega changes by roughly one basis point. Cycles show no
  resolved loss, a one-codegen-unit control is neutral, and the default artifact shrinks 12 KiB
  aggregate. This is a compiler-layout tradeoff with a favorable code-size/ordinary-instruction
  side, not validation work added to ordinary construction.
- Carrying reconciliation-produced local-rebuild seed pairs through pipeline state as a typed
  owner perturbed clean-path codegen despite retaining tuple storage. Seven interleaved 500k
  single-threaded Fibonacci counter pairs showed +0.1602% instructions and +1.6619% branches in
  every pair. The implementation was reverted. The accepted narrower `CellId` boundary exists
  only at the cold overlay splice mutation; the same counter matrix was neutral (mean changes
  -0.000066% instructions and +0.000082% branches). Extending the same local pattern to
  `VertexId` position/key lookups was also neutral in seven pairs (-0.000058% instructions and
  -0.000133% branches) with identical section sizes. Carrying `VertexId` across adjacent vertex
  creation and owner lookup was likewise neutral (+0.000307% instructions and -0.000152%
  branches); it added 32 `.text` bytes and four unwind bytes, offset by 32 fewer padding bytes.
- Reusing the backend's final `Vec<VoronoiCell>` allocation as the diagram's
  layout-identical cell storage removed the per-cell conversion allocation and copy. Both a shared
  internal record and a `repr(C)` ownership-transfer implementation reduced 1M native uniform
  instructions by 0.19--0.22% and branches by 1.15% in all nine pairs. Both also perturbed the hot
  build's cache layout badly: the shared-record form increased cache references 9.3% and cycles
  2.8%, while the ownership-transfer form increased them 11.9% and 4.2%. Keep the explicit final
  conversion until output construction can change more holistically; removing this isolated copy
  is a clear retired-work win but a throughput loss on the ordinary uniform workload.
- Borrowing a matched incoming edge check in place instead of copying its 20-byte record made the
  shrinking-suffix loop worse. At 1M single-threaded native, Fibonacci instructions/branches rose
  0.073%/0.397% and uniform rose 0.066%/0.347%, with every one of nine pairs unfavorable for both
  counters. LLVM handles the short-lived value copy better than the indexed borrow; keep the copy.
- Reusing the gnomonic extraction direction's f64 squared norm for both validity and canonical
  normalization removed exactly 12 instructions and 6 branches per emitted vertex (600k/300k at
  50k Fibonacci), but the smaller extractor perturbed code layout badly: Cachegrind I1 misses rose
  from 1.23M to 3.19M. At 1M native uniform, seven interleaved pairs retained 0.19% fewer
  instructions and 0.73% fewer branches but increased cache references 16.7% in every pair and
  cycles 3.1% overall. Expressing the arithmetic locally instead of through a shared helper
  produced identical codegen. Retain the separate rounded-f32 extraction guard; arithmetic work is
  not the limiting cost if removing it makes the instruction footprint less favorable.
- Passing the generator's already-evaluated squared norm into tangent-basis construction was a
  compiler-level no-op. Cachegrind attributed exactly 7,850,000 instructions to 50k builder resets
  before and after; whole-build movement was -0.005% with unfavorable layout noise. LLVM already
  eliminates the repeated dot products, so keep the clearer independent calculations.
- Releasing shard position buffers during global vertex concatenation did not materially reduce
  the observed peak. Dropping sources inside the parallel scatter left the source-plus-destination
  overlap intact and measured about 8--9 MiB more RSS at 2M. Copying shards serially before each
  drop changed the median peak from 562,672 KiB to 560,338 KiB across six alternating measurements,
  only 0.4% amid 14--17 MiB run ranges, while adding 1.42% whole-build branches at 1M; instructions
  were neutral (+0.016%). Retain the parallel copy: this concatenation overlap is not the governing
  RSS peak, and serializing it has a structural cost without a useful memory-envelope gain.
- Accumulating an owner-local incidence bit mask during edge collection duplicated enough work to
  lose despite removing a later classification pass. At 1M single-threaded Fibonacci it added
  2.68% instructions, 2.51% branches, and 3.03% cycles. Keep the existing pass unless the mask can
  be produced without widening the hot collection loop.
- Replacing the per-cell resolved vertex-index `Vec` with an inline `[u32; 24]` scratch plus heap
  fallback slightly reduced instructions and branches but increased cache references by about 4%
  and regressed cycles by 1.05% on Fibonacci and 0.67% on uniform. The larger always-live stack
  frame is not a throughput win; retain the reusable vector.
- Reducing the live-dedup position/key reserve from six to three or four entries per local generator
  is a valid memory/performance tradeoff, but not a default speed win. A 1M sweep over Fibonacci,
  uniform, clustered, and a successful 500k mega case found no shard above 2.03 owned vertices per
  local generator, so neither factor reallocated. At 96 bins, 3x reduced peak RSS by 29--33 MB and
  minor faults by 7--10%, but added 0.025% instructions/0.030% branches on Fibonacci and about
  0.005%/0.004% on uniform. The 4x form saved only 12--13 MB and 2--4% of faults while still adding
  roughly 0.006% Fibonacci instructions/branches; uniform was neutral. Six bins showed essentially
  no RSS or counter effect. Retain 6x for the speed-oriented default; revisit a smaller factor only
  as an explicit memory-mode policy.
- Packing `DeferredSlot`'s `(source_bin: u8, source_slot: u32)` into a `u64` does not shrink the
  32-byte record: the key and position consume 24 bytes and the packed field raises alignment to
  eight. Packing both into `u32` could reach 28 bytes but would reduce the source-slot range to 24
  bits, adding a representation limit to save only four bytes. Do not trade supported capacity for
  this cold fallback-record layout.
- A same-owner-bin fast path in final scatter has an exceptionally high hit rate but still loses to
  the unconditional indexed offset load. Same-owner references accounted for 99.78--99.88% at six
  bins and 99.02--99.42% at 96 bins across Fibonacci, uniform, clustered, and mega. Loading the
  current bin's offset once per cell and branching per packed reference nevertheless added about
  0.195% native instructions and 0.25% branches on both Fibonacci and uniform, consistently at both
  bin counts. On a quieter machine, 40 rotated 1M Fibonacci rounds with three inner builds measured
  3.21% worse cycles (95% interval +1.79% to +4.65%, candidate won 6/40); candidate-first and
  candidate-second subsets independently regressed about 3%. Thirty equivalent uniform rounds were
  0.93% worse overall (10/30 wins), though an execution-order split remained. Neither run recorded
  context switches or CPU migrations. A branchless select would still load the indexed offset and
  cannot realize the intended saving; retain the simple lookup.
- Per-(ring cell, query) spherical-cap pruning: adjacent caps rarely prune; measured net loss.
- Packed-to-shell attempted-slot filtering: low duplicate coverage and extra branching.
- Scalar shell dot-only SIMD: measured 6.5–8.5% slower.
- Lower grid target density: density 24 beat 16 by 4.8–7.1%.
- Packed partial-selection rewrite: measured 7–14% loss at 2M.
- Whole-ring packed bound skipping: neutral or worse outside a narrow dense case.
- Local packed radius-2 optimization: no winning regime; removed end to end.
- Eager/adaptive local ring-tail batching: only 3.8–9.6% of queries requested tails across 100k
  fib/uniform/clustered/bimodal, while productive lazy rescans were about 1.5% of 500k clustered
  runtime. Batching only useful requests requires a traversal redesign.
- Lazy recomputation of retained high-threshold `chunk0_keys`: despite 86.4% unused keys on 100k
  clustered, 75,653 later requests rebuilt 15.75M keys, costing 28.3% instructions, 65.1% branches,
  and 29.8% cycles. Keep the retained keys.
- Clearing per-query packed tail buffers only when a generation actually requests its tail added
  0.045% instructions and 0.059% branches on 500k native Fibonacci, and 0.085% instructions and
  0.076% branches on uniform; every structural pairing regressed. Eagerly clearing the mostly-empty
  inner vectors is cheaper than moving that work into the requested-tail path.
- Dense-band eligibility before the raw candidate cap: admitting actual band work within the same
  budget regressed a 5k cap by 0.7% instructions, 1.6% branches, and 1.4% cycles; a 10k cap
  wall-time check was about 7% slower.
- Fusing the inverse point-to-slot map with `SlotPoint` AoS construction: at 2M it did not reduce
  retired instructions on Fibonacci or uniform input and was slightly higher in all Fibonacci
  pairs. The separate pass likely keeps the random inverse-map stores out of the multi-stream AoS
  construction loop; the candidate was reverted.
- Specializing shell scans by cell mode and slot order: the full version reduced 500k clustered
  instructions by 1.04% but increased 500k mega instructions by 0.99%. Restricting specialization
  to the directed center suffix still split −0.95% / +0.80%; requiring at least eight rejected
  prefix slots split −0.94% / +0.80%. Candidate counts and timing telemetry were identical, so this
  is a genuine path/codegen tradeoff rather than changed work. All variants were reverted rather
  than introduce a distribution-sensitive heuristic.
- Using a four-lane tail for large-clip remainders 1–4: N=9/12/20 mixed-clip microbench results were
  neutral, and 500k mega instructions improved only about 0.005% while cycles were slightly worse
  in all three pairs. The saved dead-lane arithmetic does not repay the remainder dispatch in the
  production path; the candidate was reverted.
- Replacing linear incoming edge-check lookup with a tiny index: a 500k sweep over Fibonacci,
  clustered, and mega at 6 and 96 bins found 99.55–99.93% of cells had at most eight incoming
  checks. Linear scans averaged only 2.4–2.7 contiguous comparisons per lookup. Cells above eight
  accounted for 0.49–2.70% of comparisons; above sixteen accounted for at most 1,533 comparisons
  in an entire run. No duplicate incoming keys occurred. Map setup cannot repay that activation, so
  D1 was retired without implementation. The same sweep saw no cells above 64 incoming checks, so
  the compact high-degree spill candidate remains fixture-only rather than production-motivated.
- Combining the gnomonic builder's parallel half-plane, neighbor-index, and neighbor-slot vectors
  into one accepted-constraint record reduced retired instructions by 0.775% and branches by 1.285%
  on 500k single-threaded native Fibonacci (all nine pairs); Cachegrind independently measured a
  0.754% instruction reduction at 20k. However, cache references rose 19.8%, cache misses rose 28.0%,
  and cycles regressed 2.28% in eight of nine pairs. Uniform showed none of that cache penalty, but
  the ordinary Fibonacci regression rejects the wider AoS record; keep the hot half-plane stream
  separate from extraction metadata.
- Publishing those same three vectors with one explicit capacity test and unchecked stores reduced
  500k native Fibonacci instructions by 0.16%, but added 0.79% branches and 2.69% branch misses;
  cycles regressed 0.66%. An earlier short-circuit form added 2.12% branches. LLVM's ordinary
  `Vec::push` paths are predicted better than the combined invariant machinery, so keep the three
  safe pushes.
- Combining clip polygon size and bounding-reference state into one tuple match produced equivalent
  native codegen (instructions +0.00011%, branches +0.00030%, mixed pair signs). A 1M Fibonacci
  audit found N=3/4/5 account for 24.9%/31.4%/23.7% of clips, while bounded incidence falls from
  86% at N=3 to 65% at N=4 and 34% at N=5. LLVM already optimizes the nested dispatch; future work
  should target the N=3-5 kernels rather than rearranging the match.
- Hoisting exit-intersection coordinate calculation beside the already-paired entry calculation in
  the N=3-8 small kernels also produced equivalent native codegen (instructions and branches both
  +0.00004%, mixed pair signs). LLVM already schedules the independent interpolation chains across
  the survivor-copy loop; source-level reordering adds nothing.
- Replacing N=3/N=4 cyclic-mask transition decoding with byte lookup tables reduced instructions
  0.129% but added 0.456% branches. N=3 alone still added 0.215% branches; packing its eight entries
  into one branchless-looking `u64` added 0.017% instructions, 0.215% branches, and 0.39% cycles.
  The original rotate/AND/two-`tzcnt` sequence is superior. A 1M mask audit also found no single
  dominant mixed mask, so a narrow pattern fast path is not justified.
- Evaluating only three scalar signed distances in the N=3 production clip kernel, while retaining
  its existing mask, fallback, guarded interpolation, and output paths, regressed the native
  clip microbench in both regimes: mixed clips moved from about 20.6 to 23.5 ns/call and unchanged
  clips from about 4.7 to 6.2 ns/call. AVX2's four-lane evaluation is cheaper even with one dead
  lane. The faster retained scalar reference omits other production work and is not a valid kernel
  substitute.
- Caching the promoted generator norm once per cell removed repeated square roots from termination
  cache rebuilds and reduced native instructions 0.114% in all 60 pairs at 1M Fibonacci. However,
  the added builder field regressed cycles 1.98% (thirds +0.26%/+3.44%/+2.26%; both execution orders
  worse). Generic-target instructions/cycles improved, but the primary native layout rejects the
  field; retain the recomputation.
- Specializing incoming edge-check lookup for slice lengths 0 through 4 made the short common case
  explicit but added dispatcher branches around the already-small linear search. At 1M native
  Fibonacci it regressed instructions 0.095% and branches 0.479% in all 15 pairs, with cycles 0.95%
  worse; retain the generic iterator search.
- Deriving final assembly's generator loop length from the packed assignment array, rather than the
  parallel byte bin array, did not remove measurable bounds-check work: 1M native Fibonacci differed
  by only +0.00014% instructions and +0.00017% branches over 15 pairs. LLVM already optimizes the
  packed lookup equivalently; retain the more direct cell-count expression.
- Replacing the paired vertex/resolved-index `zip` in key dedup with one length assertion plus an
  indexed unchecked loop regressed 1M native Fibonacci instructions 0.511% and branches 0.688% in
  all 15 pairs. LLVM's slice-zip lowering is better than the manual traversal here; retain `zip`.
- Native assembly already hoists the packed cell-bin prefix, reducing `pack_ref(bin, local)` on the
  resolved vertex path to an OR. It also eagerly loads the 24-byte vertex record before that path,
  but sinking the load by changing copied-item zip to reference zip regressed instructions 0.054%
  and branches 0.339% in all 15 native 1M Fibonacci pairs. The altered loop control outweighs the
  saved loads; retain copied-item zip and the compiler-hoisted packing.
- Outlining `lerp_t_pair`'s nonzero-epsilon finite/clamp guards into a cold helper enlarged the final
  binary text by about 880 bytes and regressed 1M native Fibonacci instructions 0.851% and branches
  0.691% in all 15 pairs (cycles +3.99%). The guarded path is not cold enough across real ordinary
  and edge-check clips, and LLVM's inline cross-specialization layout is superior; retain it inline.
- Compile-time strict-epsilon specialization removed 1.13% whole-build instructions and 1.79%
  branches on native 1M Fibonacci, but duplicated about 26 KiB of clip dispatch text. Over 45 pairs,
  cache references rose 31.85% in every pair and cycles regressed 1.51% (all thirds and both orders
  worse). Restricting strict arithmetic to N=3/4 did not narrow the binary because forced inlining
  still copied the full dispatch and regressed cycles 2.51%; retain the compact runtime epsilon guard.
- **Promising, not rejected:** AVX2 `rsqrtss` plus one Newton refinement for extracted vertex
  normalization improved 1M Fibonacci cycles 3.46% (14/15) but added 0.34% instructions. A 45-pair
  uniform run was cycle-neutral (-0.20%, candidate lower 16/45) while adding 0.31% instructions in
  every pair. Targeted correctness/validation/adversarial suites passed, but the changed vertex
  rounding can affect proximity-based reconciliation/local rebuilding as well as public geometry.
  Revisit only with a stronger accuracy/topology audit or a workload showing broader latency benefit; retain exact
  `sqrt().recip()` for now.
- Deferring `PolyBuffer::max_r2` maintenance while synthetic bounding vertices remain removed the
  per-survivor radius arithmetic from bounded clips, then recomputed the exact radius once when the
  final bounding reference disappeared. Native 1M Fibonacci instructions improved 0.103%, but the
  required transition test added 1.89% branches; branch misses were neutral, and a 20-pair 100k
  cycle run was neutral/slightly worse (11/20 candidate wins, about +0.2% mean cycles). The saved
  arithmetic does not repay the predictable per-clip branch, so retain eager radius maintenance.
  A related correctness refactor remains independently worthwhile: make the cached radius private,
  expose an exact getter that debug-asserts the polygon is free of synthetic bounding references,
  and have cold diagnostics recompute radius directly from live coordinates. This would enforce a
  clearer invariant at compile/checked time without weakening diagnostics, but should be evaluated
  and justified separately rather than bundled with the rejected lazy-radius optimization.
- Extending the conservative early-unchanged radius certificate from polygon sizes >=5 to N=4
  added 0.12–0.13% retired instructions and 0.28–0.31% branches on 500k single-threaded native
  Fibonacci and uniform, with every one of seven pairs worse on both structural counters. Cycles
  were noisy and distribution-dependent. Keep the >=5 cutoff: four-lane classification is cheap
  enough that rare N=4 certificate hits do not repay the scalar precheck.
- Rewriting the incoming edge-check linear search from `position` plus indexed copy to an
  enumerated copied `find` produced identical retired instructions and branches (ratios
  0.999999–1.000000) on 500k native Fibonacci and uniform. LLVM already eliminates the apparent
  redundant lookup; cycle movement split by distribution and was layout noise.
- Tightening packed prefix selection so every remainder larger than the requested prefix uses
  `select_nth_unstable` (instead of whole-sorting remainders up to 2x the request) added 1.06–1.22%
  instructions, 3.15–3.46% branches, and 8.2–9.3% branch misses on 500k native Fibonacci and
  uniform. Clustered also regressed slightly. Keep the 2x whole-sort threshold: partitioning these
  small 9–16 element remainders is substantially more branch-heavy than the sorting networks.
- Fusing exact-zero endpoint comparison into the mandatory topology-summary halfedge pass avoided
  a separate traversal but added irregular vertex loads, comparisons, and per-worker candidate
  collection. Against the hardened sparse certificate at 500k single-threaded native, Fibonacci
  added 2.91% instructions, 3.32% branches, and about 6.0% branch misses; uniform added 2.65%,
  3.25%, and about 5.58%. Keep the degree-local necessary-coordinate hint plus representative-
  drift certificate, and reserve exhaustive scanning for uncertified builds.
- Accumulating the existing x-only zero-edge hint while gnomonic extraction positions were live
  removed the later tiny-buffer scan, but added 0.192% native instructions and 0.084% branches at
  1M Fibonacci, with all fifteen pairs worse on both counters; cycles trended about 2% worse.
  Portable Cachegrind instead showed 0.12% fewer instruction references and 1.81% fewer branches,
  but 28.8% more simulated I1 misses and 0.59% more D1 misses. Keep the outlined scan: on the native
  production target its hot-buffer reread is cheaper than extending extraction's live state.
- Replacing the two 16-byte outgoing-edge scratch streams with one u32 tag per cell edge and
  forwarding each edge as soon as vertex dedup produced its second endpoint removed the final
  scratch-emission pass, but mixed queue/overflow dispatch into the vertex loop. At 1M native
  Fibonacci it added 2.32% instructions and 4.16% branches in all nine pairs, with neutral cycles.
  Portable Cachegrind confirmed 2.43% more instruction references, 3.44% more branches, 25.8% more
  I1 misses, and 3.61% more mispredicts. Keep the compact dedicated outgoing passes; their regular
  iteration is much cheaper than interleaving the two state machines.
- Short-circuiting packed tail SIMD chunks when their security mask is empty was neutral on
  Fibonacci (instructions -0.007%, branches +0.009%) but regressed 500k native clustered by 0.048%
  instructions, 0.106% branches, and 1.74% cycles. Most activated tail-rescan chunks have at least
  one security-safe lane, so keep computing the high-threshold mask without an extra branch.
- Fusing the full and overlapping remainder chunks for 9--15 point packed ring ranges shared the
  query broadcasts and one combined-empty test, directly targeting the hottest sampled branch in
  `prepare_group_directed`. A rotated Windows 2.5M Fibonacci phase ring was directionally favorable
  but unresolved (`ring_pass` paired median -4.02%, 10/16 favorable). Nine pinned 1M native Linux
  counter pairs decisively rejected the dispatch/code-footprint cost: whole-build instructions rose
  0.385% and branches 0.972% in every pair; branch misses were neutral (+0.078%) and noisy cycles
  favored the candidate by 0.87%. Keep the separate single-chunk and overlapping-remainder loops.
- Instrumenting the hottest sampled owner-routing branch in `emit_cell_output` on Windows 2.5M
  Fibonacci found that 66.381% of vertex keys take the resolved-index fast exit, 33.333% create a
  local vertex, and only 0.286% defer to another shard. Of keys that reach the owner-bin test,
  99.149% are therefore local, matching the existing native layout's local fallthrough. Outlining
  the rare deferred arm reduced the native hot function by 417 bytes, but added 0.522% whole-build
  instructions while removing 1.058% branches on pinned 1M Linux. Nine rotated Windows pairs were
  unresolved/slightly adverse (total-time median +0.57%, 5/9 worse; dedup also 5/9 worse). Keep the
  inline deferred arm and the current resolved-first/local-fallthrough native layout.
- Zipping the three equal-length threshold-selection streams saved about 0.19% instructions and
  0.49% branches on native 1M Fibonacci (and comparable structural work on uniform), but native
  cycles were worse in 6/8 rotated-order pairs on both distributions, often by several percent.
  Generic-target cycles were also worse in 5/6 initial pairs. Keep the compact indexed loop: its
  layout/register allocation is materially better despite the extra retired work.
- Extracting packed directed-range discovery and budget classification into a fully inlined helper
  improved 500k Fibonacci and uniform retired work but added 0.1397% instructions on 100k clustered
  and 0.0127% on 100k mega in every seven-pair gate; forced inlining did not change the split.
  Restoring the original later center-range read removed the dense losses but added 0.0102%
  instructions on Fibonacci and 64 text bytes. The compact shape was reverted. The source-shaped
  form was retained as practically neutral after default/high-bin uniform reproduced the same
  roughly one-basis-point instruction displacement with fewer branches, while clustered and mega
  remained neutral. The retained decision and reopening condition are summarized in the
  repository-only [`code-quality closeout`](https://github.com/PeterKlimk/voronoi-mesh/blob/main/docs/internal/code-quality-closeout.md#packed-group-preparation).
- Hoisting shard-local `usize` to `u32` validation from every generator to once per grid-cell group
  saved about 0.05% native instructions, but slightly increased native branches. Rotated 1M cycles
  split by distribution: Fibonacci favored the candidate in 3/4 pairs while uniform rejected it in
  3/4. Generic structural counters improved, but the primary native signal is too small and mixed
  to justify extra group-validation machinery; retain the direct checked conversion per cell.
- An assembly-guided rewrite of the clean-path representative-drift predicate replaced the explicit
  non-finite classification plus epsilon comparison with one negated ordered comparison. The delta
  is an absolute coordinate difference, so this accepts exactly finite in-range values and still
  rejects NaN and infinity. Native codegen removes the integer bit classification, two flag
  materializations, and their OR. Twelve interleaved 1M single-thread Fibonacci pairs reduced
  retired instructions by 0.3604% with unchanged branches; ten portable-codegen pairs reduced them
  by 0.2981%, also with unchanged branches. A deliberately noisy 30-round Windows native 2.5M
  multithreaded run was directionally favorable (-4.53%, interval -9.12% to +0.29%), but that
  magnitude is code-layout interaction rather than a causal estimate of the small rewrite.
- Two nearby assembly-driven controls were rejected. Explicitly keeping the resolved vertex index
  live removed two reloads but perturbed register allocation enough to add 0.084% instructions and
  0.375% branches in every 1M pair. Forcing `build_cell_into` out of line shrank the caller but
  created a 10.6 KiB generic body, adding 1.398% instructions and 0.335% branches. Retain the reloads
  and forced inline specialization.
- A PGO-guided attempt to inline the edge resolver into cell emission saved 0.177% native
  instructions but added 0.310% branches, increased L1I misses about fivefold, and regressed cycles
  1.48%; PGO's coordinated global layout cannot be reproduced by that annotation alone. Inlining
  the single-call-site clip-batch wrapper was independently favorable: native Fibonacci reduced
  instructions 0.427% and branches 0.578%, while portable codegen reduced them 0.168% and 0.559%.
  Native uniform, clustered, and mega all improved both counters as well; total text shrank by 8.5
  KiB. Cycle samples were host-noise dominated, so acceptance rests on the cross-target,
  cross-regime structural reduction and the simpler one-call-site codegen shape.
- A Windows 2.5M Fibonacci incidence audit found N=3/N=4 clips contribute 4.75M/6.28M of 16.45M
  small-kernel dispatches, with changed clips against the synthetic bounding polygon dominating
  both sizes (4.21M/4.31M). Packing the entry and exit `(u, v)` interpolations into one `f64x4`
  reduced native 1M instructions by 0.086%, but enlarged `dispatch_clip` by 731 bytes and regressed
  cycles in seven of nine pairs because the exit coordinates remained live across the survivor
  loop. A lifetime-preserving two-coordinate helper compiled to the original scalar operations and
  was counter-neutral. Keep the separately scheduled scalar interpolations.
- Reusing the generator norm reciprocal in tangent-basis reset removes one scalar divide per cell:
  the already-required `0.5 / |g|^2` is doubled exactly for the basis projection's `1 / |g|^2`.
  Native 1M Fibonacci retired 0.031% fewer instructions and portable 500k retired 0.026% fewer,
  with branches effectively unchanged. Short Linux cycle sets were neutral/slightly adverse, and
  Windows 2.5M multithreaded timings remained scheduler/layout dominated (roughly 434--981 ms), so
  acceptance rests on the removed divide, cross-target structural reduction, and a direct
  bit-equivalence regression over normalized-f32 generators.
- Gnomonic vertex extraction formerly converted each unnormalized f64 chart source to f32 and
  computed a second squared length solely for its finite/degenerate guard, then recomputed the f64
  squared length for mandatory normalization. Reusing the f64 value removes 0.616% of native 1M
  Fibonacci instructions and 0.701% of portable 500k instructions, with branches effectively
  unchanged. Native cycles favored the candidate in five of nine noisy pairs. A subsequent quiet
  Windows native 2.5M ring split exactly 12/24, with a -0.11% paired median and +0.04% geometric
  mean (approximate 95% interval -1.03% to +1.13%). The projection-limit and tangency invariants
  keep live sources inside the same f32 validity envelope; checked builds recompute the former
  predicate and assert agreement on every extracted vertex.

Group-wide shell takeover batching is not an isolated query optimization in the current pipeline.
Same-bin cells are serialized because earlier cells emit live edge checks that seed and reconcile
later cells. Sharing traversal and emission across a group would therefore require a corresponding
stitching/scheduling redesign; revisit it only as that larger architectural change.
## Positive simplification

Construction-aware positive simplification has a separate probe:

```bash
RAYON_NUM_THREADS=1 cargo run --release --features tools --bin bench_simplify -- \
  10000 1e-4 10
```

Arguments are point count, unit-sphere chord threshold, and timed rounds. The probe measures the
complete construction-aware computation. Its output distinguishes hot-hinted cells, confirmed
terminal candidates, accepted and declined contractions, newly exposed edges, removed vertices,
and the maximum representative-displacement bound.
