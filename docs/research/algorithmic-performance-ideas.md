# Algorithmic Performance Ideas

**Status:** research catalogue, not an implementation queue

This document collects larger algorithmic ideas that do not fit the code-specific experiment queue
in [`performance.md`](../performance.md) or the representation and memory-traffic backlog in
[`memory-layout-ideas.md`](memory-layout-ideas.md). It records hypotheses and the evidence needed to
promote them; it does not assign priority or imply a commitment. [`work-log.md`](../work-log.md) remains
the authoritative queue.

The current ordinary path is already fast enough that a more sophisticated algorithm is not a win
merely because it has better asymptotic language attached to it. Comparisons need quiet, optimized
builds, interleaved workloads, and hardware counters. In particular, alternatives with larger
constants must establish a measured crossover rather than compete with the normal path by default.

## Implemented from this catalogue

### Defect-local merge-safety validation — implemented 2026-07-16

Edge reconciliation now validates proposed positional merges over a certified cell cover instead of
revisiting the complete diagram on every defect-bearing round. The cover is the union of generator
cells in the vertex keys of every original member retained by the cross-round merge ledger. This
also covers cells that reference a surviving representative whose own key no longer names them.

All components proposed in a round are still simulated jointly and in the original cell order.
Missing or invalid key provenance falls back to the global scan. Checked builds run the global scan
as a differential oracle and assert identical component decisions; focused tests cover sparse
localization, the fallback, and references inherited through an earlier merge.

The deterministic `cubed` benchmark distribution exercises O(n) degree-4 reconciliation defects.
At approximately 100k sites, the local cover examined 2,650 of 99,846 cells with no fallback;
twelve interleaved single-threaded rounds reduced median reconciliation time from 19.6 ms to 8.0 ms.
At approximately 500k, it examined 2,084 of 501,126 cells and seven rounds reduced the phase from
55.3 ms to 6.4 ms. Nine 100k Linux perf pairs on normal non-timing builds reduced whole-build
retired instructions by 13.47%, branches by 21.80%, and cycles by 9.74%, with every pair favorable.
The measured result is also recorded in [`performance.md`](../performance.md).

### Group-shared shell schedule — implemented 2026-07-21

Shell frontiers now retain the query-independent BFS cell order and layer offsets across
consecutive queries with the same start cell. Queries remain sequential and still compute their
own bounds, resident dots, key ordering, clipping, and directed edge forwarding; a later query only
avoids rediscovering layers already reached by an earlier group member.

Three native single-threaded Linux perf pairs reduced instructions and branches on every workload:
0.043%/0.129% on 100k Fibonacci, 0.066%/0.172% on uniform, 0.392%/0.893% on clustered,
3.048%/5.271% on mega, 0.655%/1.921% on great-circle, and 0.455%/1.014% on 500k clustered. Mega
also improved cycles by 3.06% in every pair; other cycle results were noisy. The full retained
schedule added about 1.2 MiB peak RSS on single-threaded 100k mega and 5.6 MiB with default
threading. Detailed census, counter, and memory evidence is in the
[`kernel optimization experiment log`](../internal/kernel-optimization-experiment-log.md#group-shared-shell-traversal-census).

## Candidate ideas

### Post-review kernel hypotheses

The original July 2026 multi-model kernel shortlist has been measured and closed. Its tried branches,
counter results, and retained oracles are summarized in the
[`kernel optimization experiment log`](../internal/kernel-optimization-experiment-log.md#pass-closeout).
The first three hypotheses below were derived from those failures and closed by their read-only
gates. The later group-shared shell traversal passed its gate and is now listed in the implemented
section above.

#### 1. Center-informed one-shot high-threshold correction — rejected and closed

The packed high threshold is currently selected before candidate dots are observed, using a count
model whose distributional assumption substantially over-retains keys on clustered inputs. The
failed compact-overflow experiments tried to repair that overshoot by maintaining an exact top 64
while all high keys streamed past. The per-key heap comparison and repair was decisively more
expensive than the append-then-partition work it removed.

A different cost model is to correct the threshold once, before the dominant ring scan. Use the
already-computed directed center-cell dots to estimate a group-level normalized dot distribution,
then raise (never lower) the per-query high threshold for occupancy-rebuilt, ordinary non-band
groups predicted to overshoot. Ring dots below the corrected threshold become part of the existing
lazy tail. Exact candidate order, the security floor, and fallback behavior remain unchanged; a
query that needs the demoted band reconstructs it through the existing exact tail path.

This would leave candidate dot count unchanged but could remove ring-key construction and writes,
allocator growth, partition depth, and sorting without a data-dependent selection operation on
every high key. Its first experiment is timing-only: build a small center-derived histogram,
simulate several corrected thresholds against the exact ring keys already produced, and record
retained keys, the number of queries whose observed consumed depth crosses each threshold, and the
corresponding exact tail-rescan dots. Reject it if the center sample does not predict ring
overshoot, if productive rescans recover most saved work, or if the eligible key volume is too
small to repay even gated histogram maintenance.

That gate rejected the center-only estimator: it treated systematically nearer center residents as
representative of the ring and grossly overraised thresholds on Fibonacci and uniform controls. A
stratified ring-prefix refinement correctly exposed large key ceilings on clustered and splittable
inputs, but a per-cell eight-resident sample performed far more extra dot work than the keys it
saved on mega, bimodal, gradient, and outlier cases. Full census values are in the
[`kernel optimization experiment log`](../internal/kernel-optimization-experiment-log.md#center-informed-threshold-shadow-branch).

The final narrower gate also ran. It pre-gated individual queries from exact center high-key counts,
used a cheap maximum-possible-saving test before one whole-ring SIMD sample, and charged keys rebuilt
by tail use rather than calling every eager demotion a saving. Strict 4x and 8x policies isolated a
small clustered opportunity at 100k, but it collapsed at 500k: the 4x form permanently avoided
179,853 keys while paying 57,208 sample-vector evaluations, 63,442 selected-center key visits, and
182,768 rescan dots; the 8x form sampled 24,950 queries to accept 58 and avoided only 9,158 keys.
Splittable was negative and the other density-contrast controls produced no useful strict-margin
work. Full counters are in the
[`kernel optimization experiment log`](../internal/kernel-optimization-experiment-log.md#one-shot-threshold-refinement-oracle-branch).

Do not revive this family with another sampling geometry. Revisit only if a future representation
provides a suitable ring-distribution statistic for free or makes threshold changes avoid both
center selection and lazy-tail reconstruction.

#### 2. Seed-first micro-batched packed preparation — rejected with current bounds

Incoming edge checks are real neighboring constraints and are clipped before the ordinary query
stream, but a complete same-grid-cell group is prepared before any cell in that group is emitted.
Consequently, later generators pay their full row of the group query-candidate matrix even when
constraints forwarded from earlier work may already make the cell certifiable or sharply reduce
its remaining demand.

A possible restructuring would process a group in small blocks, apply every currently available
forwarded constraint first, and prepare packed ring work only for unresolved queries. A four- or
eight-query block may retain enough of the current SIMD sharing while allowing newly completed
cells to seed the next block. The upside is deletion of complete query rows rather than shorter
already-produced batches; the main risk is that lost group-wide SIMD and more preparation
boundaries cost more than the skipped rows.

Do not restructure the driver first. Add an exact timing-only oracle after seed clipping and count
cells that would need zero packed candidates if the already-prepared initial exact frontier bound
were available. If that ceiling is material, test how much survives replacement by a cheap
conservative grid-cell bound. Only then compare micro-block sizes and account for dot rows saved,
extra preparation calls, forwarded-edge volume, and lost SIMD occupancy.

Both gates have now run. The exact prepared-frontier oracle covers only 3--6% of ordinary row dots
(and 0.04% on mega), while production visits exactly one later packed candidate per exact-batch
hit. Existing whole-cell caps retain at most 1.43% of row dots on Fibonacci and only 0.06--0.23%
on the density-contrast targets, with effectively no high-key savings. The center-cell cap contains
the generator and becomes useful almost exclusively when the directed center suffix is empty.
Full counters are in the
[`kernel optimization experiment log`](../internal/kernel-optimization-experiment-log.md#seed-first-packed-preparation-oracle-branch).

Do not introduce micro-batch boundaries for this ceiling. Revisit only if another workload makes
row preparation dominant enough to justify finer center-suffix cap metadata, and charge that
metadata plus lost group-wide SIMD against the exact 3--6% upper bound.

#### 3. Certified dense-region local-hull handoff — same-cell form rejected

For `mega`, great-circle, and isolated high-work regions, many generators can repeatedly search and
clip against nearly the same large candidate set. Instead, run the existing robust local 3D hull
once over a dense region plus a guard ring, derive cells for several interior generators together,
and accept only cells certified independent of points outside the region. Boundary or uncertified
cells remain on the current path. This is a concrete regional form of the progress-aware handoff
described below: it attempts to replace repeated query-by-candidate work rather than accelerate
one query.

The first, cheaper gate counted exact attempted-neighbor overlap for every existing same-grid-cell
group before attempting an offline replay. It compared each group's repeated candidate attempts
with the pair-count floor for running the current naive `O(p²)` local hull over the smallest
possible union. No group qualified across 100k Fibonacci, uniform, clustered, splittable,
gradient, mega, or great-circle inputs, nor at 500k clustered. The best case reached 74.4% of that
optimistic floor before guard expansion, robust orientation cost, certification, cell extraction,
or stitching. Mega's most expensive group was a single 98,444-attempt row, so it had no regional
work to amortize. Full counters are in the
[`kernel optimization experiment log`](../internal/kernel-optimization-experiment-log.md#same-cell-regional-local-hull-oracle-branch).

Do not proceed to guard-region replay with the current `LocalHull` and same-grid-cell regions.
Larger or differently shaped regions remain a research possibility, not a queued implementation:
they first need either a subquadratic regional triangulation, reuse of an already available
triangulation, or a certificate that produces a much smaller candidate union. Any promoted design
still requires a deterministic boundary certificate and a stitching plan that cannot turn an
ordinary successful construction into a failure.

#### 4. Group-shared shell traversal — implemented

Shell takeover currently creates an independent Chebyshev BFS frontier for every query. Queries
in an existing same-grid-cell generator group frequently revisit the same grid cells and resident
point ranges, even when their termination depths differ. A timing-only trace measured 93.5--99.1%
group-local cell-visit redundancy on uniform, clustered, mega, great-circle, and 500k clustered
inputs; at least 95% of resident/query work in each positive workload occurred at an active width
of 16 or more. Fibonacci performed no shell traversal and remains a clean control. Full counters
and the deliberately optimistic position-load ceiling are in the
[`kernel optimization experiment log`](../internal/kernel-optimization-experiment-log.md#group-shared-shell-traversal-census).

Keep two mechanisms distinct. The accepted first form is a shared immutable layer schedule consumed by cells in
their current sequential order. It can remove repeated visited-stamp and neighbor-enumeration work
without changing candidate dots, bounds, ordering, clipping, or edge forwarding. It passed native
counter and memory gates; results are summarized in the implemented section above. The much larger
resident ratio is not its saving.

The second is a tiled resident-by-query kernel that loads a point range once and evaluates it for
multiple active queries. It retains every query-specific dot and key but may reduce point-load and
loop overhead and expose SIMD across queries. It is also a dataflow change: later cells currently
receive constraints forwarded by earlier cells, traces are not uniformly lockstep, and eager
cross-query preparation can compute and store rows that sequential termination would avoid.
Before implementing it, specify how block boundaries preserve directed eligibility and forwarding,
then charge lost same-block seeds, speculative rows, per-query masks, and key sorting/storage.

`PERF-003` in [`work-log.md`](../work-log.md) records the completed implementation. Do not treat the
second mechanism as automatic follow-up work; it needs its own dependency design and gate.

### Selected-neighbor constraint batches — closed negative 2026-07-16

The directed neighbor pipeline collapses selected, dot-bearing batches to slot ids before the
gnomonic builder reconstructs constraints one at a time. A backend-private selected-constraint
stage was proposed to preserve the exact batch, prepare a rolling window of generator-local
bisectors, and reject redundant constraints together before ordered polygon mutation.

Every staged form was built, verified behavior-identical, and rejected on 500k pinned native
counters: rolling `f64x4` coefficient preparation (+3.0% instructions), four-lane radial
preclassification (+3.6–4.5%), 64-sector support-envelope classification (+17.0%), and a fused
exact classifier in eager (+11.9%) and adaptive (+5.15%) forms. The shared cause is that the
scalar width-one path is already near the cost floor — an unchanged clip is ~40–60 instructions
with a pre-vertex radial early exit — while short streams make ~20% of window preparation
speculative. Full results are recorded in the
[`selected-neighbor constraint batch plan`](constraint-batch-pipeline-idea.md). Do not reopen
without a workload with materially longer per-cell constraint streams.

### Work-balanced construction for non-uniform inputs

The current spatial bins provide locality and deterministic directed build order, but equal spatial
area does not imply equal construction work for clustered, gradient, or adversarial distributions.
First record work per group: generator count, candidates examined, shell work, clipping work, and
forwarded edge checks. Use those measurements to test whether a cheap pre-build estimate predicts
the actual tail.

Possible designs include weighted contiguous spatial partitions and a hot-group escape path. Any
escape path must preserve the edge-coverage contract: splitting a group across workers turns its
cross-task relationships into independently built edges unless a new deterministic forwarding rule
is proved. Measure load balance together with lost locality, duplicate edge work, foreign-key
volume, and assembly cost. Keep the present scheduler for distributions whose work is already
balanced.

Group-wide shell batching is part of the same design space, not an isolated loop optimization. The
current same-bin sequential order is what makes edge forwarding cheap; batching queries across a
group should be considered only alongside a scheduling and stitching design that retains that
benefit or demonstrates a better whole-pipeline tradeoff.

The August 2026 native 16/32-worker profile does not by itself motivate cell-interleaved scheduling.
SMT added little algorithmic work but sharply increased integer-scheduler and store-queue resource
stalls; load-queue growth was much smaller, and cache/TLB miss samples were distributed across grid,
assembly, and construction routines. Treat that as evidence against software-pipelining multiple
cell clippers on this design. Work balancing remains distinct and should be reopened only for a
measured non-uniform task tail, not as a response to ordinary SMT inefficiency.

### Progress-aware high-work handoff

Some pathological cells can remain valid while examining nearly every generator. A handoff should
be based on measured work and lack of geometric progress, not a fixed neighbor count. There are
likely several useful regimes:

- a few expensive cells can replay through unrestricted spherical construction or Hull3d rebuilding;
- a compact defect region can justify one local hull or triangulation shared by that region; and
- only a sufficiently large affected fraction can amortize a global hull or triangulation.

The existing exhaustive path is a correctness backstop, not a claim that its cold-path algorithm is
optimal. Candidate replacements must establish crossover points for the affected-cell count,
candidate work, and repair-region size, and must prove that handoff cannot turn a valid ordinary
success into failure. The authoritative tracked form of this idea is `PERF-001` in
[`work-log.md`](../work-log.md#perf-001--total-query-work-circuit-breaker).

Timing builds now record two per-cell distributions: all examined candidates and candidates after
the final polygon-changing constraint. They report raw quantiles together with counts at least
4x/16x/64x each run's own median. This normalization is essential: candidate work per generator can
grow naturally with total input size and density contrast, so an absolute count is not a pathology
certificate. Values below 256 are exact; larger quantiles are power-of-two bucket lower bounds.
Cells entering batched exhaustion recovery remain in total work but are explicitly excluded from
the progress-tail distribution because that path does not retain individual clip outcomes.

An apparent broad Fibonacci tail beyond 2.5M was traced to the benchmark generator rather than the
query algorithm. Its historical implementation multiplied an unbounded phase by the site index in
f32. Above index roughly 2.16M the phase crosses `2^23` radians and its representable spacing reaches
one radian, quantizing the nominal golden-angle lattice before `sin`/`cos`. Adjacent grid resolutions,
different jitter seeds, native instructions, and FMA all reproduced the artifact. Promoting phase
and latitude generation to f64 removed it: at 3M, candidate p50/p90/p99/p999/max changed from
9/15/26/116/167 to 7/9/10/11/14, examine-per-edge from 1.709 to 1.222, and unrestricted shell
takeovers from 5,775 cells to zero. The legacy generator was subsequently removed; this measured
record remains because it explains why the case is not evidence for a production handoff or
query-policy change.

The `mega` dense-cap workload grew from median 21 at 100k to 70 at 500k, confirming separately that
a fixed candidate threshold would misclassify scale growth within some distributions. Despite that
shift, the 500k run still isolated 252 cells at or above 64x its own median and a maximum of 492,283
candidates. The directly successful `great-circle` benchmark at 1k sites (default 0.01 coordinate
jitter) had median 24, p90 at least 512, and maximum 999. These counters characterize opportunity;
they do not yet select a handoff or establish its crossover.

### High-core-count scaling validation

Single-thread per-op cost is near its floor; the differentiating claim is multithreaded scaling
(measured ~4x on 6 cores against tuned planar Delaunay's ~2.4x). That claim has never been tested
beyond the reference machine. A short rented many-core run (64+ cores) would upgrade it from a
benchmark observation to an architecture result — or find the ceiling.

Cheap preparation before paying for the run, so it measures the architecture rather than an
allocation artifact:

- justify the `VORONOI_MESH_BIN_COUNT` default (about 2x threads) at 64+ cores, or sweep it;
- check that shard and scratch allocation is not all first-touch on one NUMA node; and
- watch fallback/repair queue growth and assembly-phase serialization at high parallelism.

### Naive-baseline ablation

Build a same-architecture, untuned variant (or a configuration that disables the tuned paths) and
compare it with both the tuned build and a naive incremental construction. This separates the
algorithmic contribution from implementation tuning for any write-up, and occasionally reveals
what a tuning was actually buying. Evaluation infrastructure, not a production optimization.

### Reusable regional dependency information

Construction already produces facts that could make cold repair and repeated computations cheaper:
cell adjacency, query coverage, affected-cell footprints, and spatial-index occupancy. Investigate
which of these can be retained internally at low cost and reused by repair or a subsequent build.
The first experiment should measure retained bytes and reuse hit rates; keeping a second graph or
certificate structure for every ordinary build is unlikely to be free.

Any caller-visible reuse contract belongs in the
[`feature/API wishlist`](../internal/feature-api-wishlist.md#temporal-topology-hints). This entry is limited to
the internal cost model and representation question.

## Already addressed: dense-cap lookup

Dense grid cells already have a side index: over-full cells are sorted along their dominant axis,
queries use a conservative coordinate band, and shell takeover covers everything below the band's
certificate. This is the current replacement for the earlier per-cell kd-tree or mini-grid ideas;
it is absent on normal uniform cells and is rebuilt after slot-changing weld compaction.

Do not reopen a generic kd-tree or hierarchy as a standalone optimization. Revisit dense-cell
indexing only when measurements identify a residual problem in band selectivity, certificate depth,
index construction, or a specific non-uniform workload that the current axis-band design handles
poorly.

## Promotion rule

An idea moves from this catalogue into [`work-log.md`](../work-log.md) only when it has a motivating
workload, an explicit semantic invariant, a measurement plan, and a scoped next decision or
experiment. Narrow implementation hypotheses may instead move into the open optimization queue in
[`performance.md`](../performance.md#open-optimization-queue).
