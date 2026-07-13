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

## Building for speed

`RUSTFLAGS="-C target-cpu=native"` is worth ~6% on the reference machine and is the main build
flag that matters. Run benchmarks in release.

## Running the benchmarks

The benchmark binaries need the `tools` feature:

```bash
cargo run --release --features tools --bin bench_voronoi -- 100k 500k 1m
```

Useful flags:

- `--dist {fib|uniform|clustered|bimodal|gradient|outlier|splittable|mega}` and `--dist-param` —
  non-uniform distributions exercise the density-adaptive paths that uniform input never reaches.
  Benchmark across a few of these, not uniform alone.
- `--validate` — compare against the convex-hull ground truth (slow; capped at 100k).
- `--no-preprocess` — skip welding (isolates construction cost).

## Knobs

- `RAYON_NUM_THREADS=1` — single-threaded, for stable comparisons.
- `VORONOI_MESH_BIN_COUNT=<n>` — shard count (default ~2x threads).
- `VORONOI_MESH_TIMING_KV=1` with `--features timing` — machine-readable phase timing.
- `VORONOI_MESH_GRID_DENSITY=<f>` / `VORONOI_MESH_PLANE_GRID_DENSITY=<f>` — spatial-grid target
  density (points per cell) for sweeps.

## Repair cold path

The post-assembly local repair fires only on defect-bearing inputs (rare; near-cocircular
clusters, e.g. the `mega` distribution) and is fast in the common case. One known cold path
remains: when the defect sits on the boundary of an extremely dense cluster (most points inside a
single grid cell), the repair's neighbor gather expands into the cluster and the repair can take
seconds to minutes at millions of points. The output contract is unaffected — the result is still
strictly valid or a clean error — only latency degrades. `RepairMode::Disabled` skips repair
entirely (residual defects then fail plain `compute` loudly).

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

That fallback is intentionally a reliability tradeoff: at 100k clustered it cost approximately
23% more instructions and 15% more cycles; uniform work remained structurally neutral. The weld
pair budget added approximately 0.8% instructions to a 500k normal-preprocessing control, while the
paired cycles interval remained unresolved around no change. Revisit these thresholds only with
both peak-storage telemetry and end-to-end counter measurements; optimizing away the fallback must
not restore unbounded retained work.

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

The always-on low-incidence repair trigger uses plain `u32` counters when the active Rayon pool has
one worker, avoiding atomic increments and the parallel final scan. Pools with more than one worker
retain the atomic implementation. At 2M single-threaded it reduced retired instructions by 0.067%
on Fibonacci and 0.061% on uniform input, with effectively identical counts in all three paired
runs. Cycles were too noisy to resolve; the default parallel control was structurally neutral.

The N=3/4 small clipper keeps its four SIMD distances in a four-element array instead of padding an
eight-element array with four zero stores. Escalation receives only the initialized slice; N=5–8
retains the full eight-lane representation. Release code shrank by 208 bytes and no longer contains
the two padded 16-byte stores. At 2M single-threaded, retired instructions fell about 0.062% on
Fibonacci and 0.061% on uniform input in all three pairs. Fibonacci cycles improved in all pairs;
uniform cycles were neutral. The default, microbench, `p5_shadow`, and escalation feature paths all
share the initialized-slice invariant.

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
repair round. The previous path created two vectors per record; an 86,400-point high-degree fixture
with 6,626 records therefore exposed 13,252 allocation opportunities. The clean path still returns
before allocating scratch, and each repair round receives fresh buffers. Same-session defect-heavy
timing moved from 13.884 ms to 12.804 ms; repeated candidate timing was noisy, so retain this for the
certain structural allocation removal rather than the provisional 7.8% phase result.

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
return the raw, already-bracketed divisions; propagated-epsilon edge checks and diagnostic
escalation retain finite checking and clamping. On 500k single-threaded native Fibonacci this
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

### Open optimization queue

These are code-specific hypotheses from a 2026-07 subsystem scan. Each item is an isolated
experiment: preserve its stated semantics, measure the named regime, and move it either into the
measured results above or the retired list below. Do not bundle candidates before attribution.

Promising workload-specific experiments:

- **D3 — compact high-degree consumed flags:** replace the per-cell byte vector above 64 incoming
  checks with reusable bit words. First instrument activation frequency; ordinary cells should stay
  on the existing inline mask.

Assembly/live-dedup swarm backlog (2026-07-13):

- **Release shard vertex buffers during the global copy:** `all_vertices` is populated while all
  per-shard position vectors remain live. Taking/dropping each source after its disjoint copy could
  lower transient live heap by about 12 bytes per output vertex. Retain per-shard spans for debug
  bounds checks and measure allocator live bytes as well as peak RSS, since the allocator may retain
  freed pages. This is principally a memory-envelope experiment.
- **Instrument a same-owner-bin scatter fast path:** using the current cell's known vertex offset can
  avoid unpack/lookup when a packed vertex owner bin equals the cell bin. Measure the hit fraction by
  distribution and input ordering before adding a branch; do not reorder cells by bin because that
  would turn sequential destination writes into scattered writes.
- **Flatten per-local edge-check queues only as a memory redesign:** `Vec<Vec<EdgeCheck>>` pays a
  `Vec` header per local generator. A node arena plus head/tail arrays could reduce empty-queue
  metadata, but it loses the current zero-copy transfer and may add traversal/copy work. Require
  queue-count telemetry and preserve exact directed enqueue order, repair origins, and high-degree
  behavior before prototyping.
- **Deduplicate repair work by unresolved edge key:** defect inputs can report multiple origins for
  one edge. A repair-only unique-key view may avoid repeated reconciliation while retaining the full
  diagnostic origin list. Existing large probes found only a few mismatch records, so this remains
  a cold-path robustness idea rather than a production-speed candidate.

Lower-confidence cleanup candidates, to attempt only with structural counters or activation data:

- Unroll weld wall-proximity tests without changing f64 arithmetic or pair-budget behavior.
- Precompute standalone large-`MergeWithin` wall proximity; this does not affect default welding.
- Consider unchecked endpoint/owner-bin access only after a measurable bounds-check cost and a
  complete invariant audit; a sub-noise win does not justify converting a panic into possible UB.

### Retired experiments

Do not broadly retry these without a materially different design or workload:

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
  its existing mask, escalation, guarded interpolation, and output paths, regressed the native
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
  rounding can affect proximity-based defect repair as well as public geometry. Revisit only with a
  stronger accuracy/repair audit or a workload showing broader latency benefit; retain exact
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
- Short-circuiting packed tail SIMD chunks when their security mask is empty was neutral on
  Fibonacci (instructions -0.007%, branches +0.009%) but regressed 500k native clustered by 0.048%
  instructions, 0.106% branches, and 1.74% cycles. Most activated tail-rescan chunks have at least
  one security-safe lane, so keep computing the high-threshold mask without an extra branch.
- Zipping the three equal-length threshold-selection streams saved about 0.19% instructions and
  0.49% branches on native 1M Fibonacci (and comparable structural work on uniform), but native
  cycles were worse in 6/8 rotated-order pairs on both distributions, often by several percent.
  Generic-target cycles were also worse in 5/6 initial pairs. Keep the compact indexed loop: its
  layout/register allocation is materially better despite the extra retired work.
- Hoisting shard-local `usize` to `u32` validation from every generator to once per grid-cell group
  saved about 0.05% native instructions, but slightly increased native branches. Rotated 1M cycles
  split by distribution: Fibonacci favored the candidate in 3/4 pairs while uniform rejected it in
  3/4. Generic structural counters improved, but the primary native signal is too small and mixed
  to justify extra group-validation machinery; retain the direct checked conversion per cell.

Group-wide shell takeover batching is not an isolated query optimization in the current pipeline.
Same-bin cells are serialized because earlier cells emit live edge checks that seed and reconcile
later cells. Sharing traversal and emission across a group would therefore require a corresponding
stitching/scheduling redesign; revisit it only as that larger architectural change.
