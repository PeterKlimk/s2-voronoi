# Algorithmic Performance Ideas

**Status:** research catalogue, not an implementation queue

This document collects larger algorithmic ideas that do not fit the code-specific experiment queue
in [`performance.md`](performance.md) or the representation and memory-traffic backlog in
[`memory-layout-ideas.md`](memory-layout-ideas.md). It records hypotheses and the evidence needed to
promote them; it does not assign priority or imply a commitment. [`work-log.md`](work-log.md) remains
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
The measured result is also recorded in [`performance.md`](performance.md).

## Candidate ideas

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

### Progress-aware high-work handoff

Some pathological cells can remain valid while examining nearly every generator. A handoff should
be based on measured work and lack of geometric progress, not a fixed neighbor count. There are
likely several useful regimes:

- a few expensive cells can replay through unrestricted spherical construction or Local3d;
- a compact defect region can justify one local hull or triangulation shared by that region; and
- only a sufficiently large affected fraction can amortize a global hull or triangulation.

The existing exhaustive path is a correctness backstop, not a claim that its cold-path algorithm is
optimal. Candidate replacements must establish crossover points for the affected-cell count,
candidate work, and repair-region size, and must prove that handoff cannot turn a valid ordinary
success into failure. The authoritative tracked form of this idea is `PERF-001` in
[`work-log.md`](work-log.md#perf-001--total-query-work-circuit-breaker).

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
takeovers from 5,775 cells to zero. `fib-legacy` retains the old input for historical comparisons.
This case is not evidence for a production handoff or query-policy change.

The `mega` dense-cap workload grew from median 21 at 100k to 70 at 500k, confirming separately that
a fixed candidate threshold would misclassify scale growth within some distributions. Despite that
shift, the 500k run still isolated 252 cells at or above 64x its own median and a maximum of 492,283
candidates. The directly successful `great-circle` benchmark at 1k sites (default 0.01 coordinate
jitter) had median 24, p90 at least 512, and maximum 999. These counters characterize opportunity;
they do not yet select a handoff or establish its crossover.

### Reusable regional dependency information

Construction already produces facts that could make cold repair and repeated computations cheaper:
cell adjacency, query coverage, affected-cell footprints, and spatial-index occupancy. Investigate
which of these can be retained internally at low cost and reused by repair or a subsequent build.
The first experiment should measure retained bytes and reuse hit rates; keeping a second graph or
certificate structure for every ordinary build is unlikely to be free.

Any caller-visible reuse contract belongs in the
[`feature/API wishlist`](feature-api-wishlist.md#temporal-topology-hints). This entry is limited to
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

An idea moves from this catalogue into [`work-log.md`](work-log.md) only when it has a motivating
workload, an explicit semantic invariant, a measurement plan, and a scoped next decision or
experiment. Narrow implementation hypotheses may instead move into the open optimization queue in
[`performance.md`](performance.md#open-optimization-queue).
