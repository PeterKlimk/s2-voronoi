# Memory-layout optimization ideas

This document records prospective memory-layout and memory-traffic experiments for the
multithreaded spherical backend. These are hypotheses, not established wins. The current pipeline
becomes increasingly limited by shared-cache and memory-system pressure as worker count rises, so
an optimization that removes instructions in a single-threaded run may still lose when it widens a
hot stream, increases cache-line traffic, or introduces cross-core ownership transfers.

The main lesson from earlier layout experiments is to reduce bytes that are actually touched, not
to apply AoS or SoA mechanically. For example, combining the gnomonic builder's parallel
half-plane and extraction-metadata arrays reduced instructions but increased cache references by
19.8%, cache misses by 28.0%, and cycles by 2.28% on Fibonacci. The existing point-coordinate SoA
and selected-neighbor `SlotPoint` AoS serve different access patterns and should not be combined
without evidence that the duplicated representation, rather than its access locality, is the
bottleneck.

## Experiment policy

Implement and measure each idea independently. Test multiple regimes, including the obvious edge
case for the proposed representation. At minimum, cover:

- Fibonacci and uniform inputs at large enough `n` to expose the multithreaded throughput ceiling.
- Clustered or bimodal input when the proposal changes packed-query behavior.
- `mega`, outlier, or another shell/dense-path input when the proposal changes candidate tracking.
- Default and high bin counts when the proposal changes shard-local versus cross-bin storage.
- A defect-bearing or reconciliation fixture when the proposal maintains incremental topology
  metadata.
- One-thread and physical-core/SMT multithreaded runs, because a layout can trade local instruction
  count for better aggregate cache or memory behavior.

Measure total cycles/time as the outcome, with retired instructions, cache behavior, peak/live
memory, and phase timings used for attribution. Preserve exact work counters and output
fingerprints where applicable. Do not bundle candidates before establishing attribution.

If an idea wins substantially on regular regimes such as Fibonacci or uniform but loses on its
obvious edge case, a hybrid is likely the right production design. Keep the common representation
small and cheap, then activate the current general representation or another bounded fallback only
when telemetry shows that the exceptional regime has begun. The fallback must preserve the current
correctness and representation limits; a fast common path is not justification for an unbounded or
inexact pathological path.

## 1. Lazy attempted-neighbor stamps

**Status: retired after experiment (2026-07-15).**

### Current cost

Every bin's `CellBuildContext` constructs an `N`-entry `u32` generation-stamp table. At one million
points this is 4 MB per active context and, with twelve active workers, can occupy about 48 MB
against a 16 MB shared L3 on the reference machine. Construction also zeroes `4 * N * num_bins`
bytes over all bin tasks: about 96 MB at 24 bins and 384 MB at 96 bins for a one-million-point
build.

The packed path writes a stamp for every processed candidate so a later shell takeover can suppress
the candidates it re-covers. A measured one-million-point Fibonacci run processed about 8.05
million neighbors, while only 19 cells entered shell takeover. Thus almost all normal-path stamp
writes prepare for a fallback that never occurs.

### Candidate design

- Keep a small reusable per-cell vector of slots actually processed from seed and packed batches.
- Do not touch an `N`-entry table on packed-only cells.
- When shell takeover first occurs, seed a lazily allocated generation-stamp table from that vector
  and retain the existing O(1) insertion behavior for the rest of the shell traversal.
- Reuse the lazy table across later shell-taking cells in the same context; advance its generation
  only for those cells.

This is deliberately a hybrid. Linear search alone is attractive for the ordinary eight-ish
attempted neighbors but can become quadratic on a shell-heavy dense cell. Eager dense stamps are
robust there but wasteful on the packed common path.

### Regime and correctness checks

- Expected win: packed-dominated Fibonacci/uniform, especially with many bins or workers.
- Obvious edge case: shell-heavy `mega`, dense, or adversarial cells with long candidate streams.
- Record only candidates actually processed before mid-batch termination or failure.
- Preserve deduplication across seed, packed chunk-zero, packed tail, shell re-coverage, and fallback
  handoff.
- Compare the lazy transition, table allocation count, attempted-list maximum, and number of cells
  that materialize the table.

### Experiment result

The lazy transition was implemented in three forms: a reusable `Vec<u32>`, an adaptive
shell-pressure fallback, and a 32-slot inline buffer with a reusable overflow vector. Exact work
counters and output sizes were unchanged. On one-million-point Fibonacci, only 18 cells
materialized dense state, 14 of 24 contexts allocated a table, and the maximum pre-shell list was
23 slots, confirming the sparsity hypothesis.

The bookkeeping cost nevertheless outweighed the avoided spatial stamp stores. For the inline
variant, deterministic one-thread Cachegrind at 20k reported 2.30% more instruction references,
2.19% more branches, 5.84% more branch mispredicts, and 44.3% more L1 instruction misses. It did
reduce data writes by 1.23% and D1 misses by 2.97%, but last-level data misses were effectively
flat. Hardware retired-instruction counts at scale likewise increased for both the vector and
inline forms. The pressure fallback did not materially improve the shell-heavy shape and added
ordinary per-cell work.

Wall time and cycle counters were too noisy during this experiment to support a throughput claim.
The deterministic counters are already unfavorable enough to retire the design: do not replace
the packed path's unconditional spatial stamp store with a per-candidate growable/inline stream
unless a future stream representation removes its capacity and recording control flow.

## 2. Owner-local vertex-incidence accounting

**Status: accepted as a multithreaded throughput candidate (2026-07-15).**

### Current cost

Every build performs a post-assembly low-incidence scan over the live cell windows. It is part of
the plain-return safety gate as well as the repair trigger, so disabling repair no longer disables
the scan. The parallel path allocates one `AtomicU32` per assembled vertex and performs roughly six
random atomic increments per input point. On a two-million-point reference run the scalar scan took
about 22.7 ms and the twelve-thread atomic scan about 25.0 ms: the pass did not scale and became
slightly slower in parallel.

### Candidate design

- Store a small saturating incidence counter alongside every shard-owned vertex.
- Increment it when an emitted cell resolves an on-shard vertex reference.
- Route off-shard incidence through the existing deferred/foreign-owner resolution data and apply
  it at the owner during assembly.
- Saturate at three if the counter remains private to the current predicate, which only
  distinguishes incidence one or two from incidence at least three.
- On the clean path, reduce the owner-local counters without a global random atomic histogram.
- If edge reconciliation or repair changes live cell windows, discard the incremental answer and
  run the existing exact scan.

### Regime and correctness checks

- Expected win: clean multithreaded builds with repair enabled.
- Obvious edge cases: reconciliation mutations, repair acceptance/rejection, welded inputs, and
  high cross-bin incidence.
- Keep the accounting active under `LocalRebuildMode::Disabled`; the topology summary is an independent
  plain-return safety signal.
- Measure whether moving one cheap increment per incidence into the dominant construction/dedup
  phase offsets the removed tail pass.
- A saturating byte is valid only for the boolean repair trigger. Do not reuse it for exact degree
  reporting without changing the representation and tests.

### Provisional experiment result

The candidate stores a saturating byte parallel to each shard's vertices, initializes newly
created vertices at incidence one, increments resolved on-shard references during emission, and
applies every deferred reference once at its final owner. Clean assembly reduces these private
counters. Any reconciliation that reports a changed live-cell footprint discards the summary and
runs the existing exact scan. Checked builds recompute the scalar live-window summary on the clean
path and assert equality.

Deterministic one-thread Cachegrind at 20k Fibonacci was close to instruction-neutral (+0.01%
instruction references) while reducing branches by 2.19%, D1 misses by 2.76%, and last-level data
misses by 1.73%; data references increased 0.31% and branch mispredicts increased 1.12%. At one
million uniform points and 96 bins, seven-round hardware-counter means showed 0.32% fewer retired
instructions, 1.72% fewer branches, 1.65% fewer branch misses, 3.45% fewer cache references, and
10.4% fewer cache misses. One-thread 500k Fibonacci remained near neutral in instructions (+0.13%)
with mixed cache movement. A 2M Fibonacci peak-RSS probe measured about 1.5 MiB more RSS, so this is
not currently a memory-envelope win.

The full `checked` test profile passes, including reconciliation and Hull3d rebuild fixtures.

Windows-native paired wall-time measurements supplied the missing acceptance signal. At two
million generators, owner-local incidence was 2.42% faster on Fibonacci, 2.87% faster on
default-bin uniform, and 3.72% faster on 96-bin uniform; all three 20-round 95% intervals excluded
zero. A separate 40-round focused Fibonacci run measured 1.47% faster, again just excluding zero.
A 30-round single-thread Fibonacci guardrail was directionally 0.69% faster with a -1.42% to +0.04%
interval, providing no evidence of a scalar regression.

Portable Windows codegen (release without `-C target-cpu=native`) also passed the acceptance
guardrails. A 30-round two-million-point Fibonacci multithreaded run was neutral at 0.60% faster
(-2.35% to +1.17%); uniform with 96 bins was 2.10% faster (1.15% to 3.05%); and a one-million-point
single-thread Fibonacci run was 0.71% faster (0.38% to 1.03%). Promote this branch to the primary
default-path candidate for both native and portable builds.

## 3. Compact shard-local cell-reference stream — implemented 2026-07-16

### Current cost

Each shard retains cell references as packed `u64` `(bin, local_vertex)` values and later converts
them to final `u32` vertex indices. At about six incidences per point, the temporary stream occupies
48 bytes per input point. It is written during emission and read during final assembly.

Measured same-owner incidence is 99.78--99.88% at six bins and 99.02--99.42% at 96 bins over the
ordinary benchmark distributions. All concrete references are initially shard-local; foreign-owner
slots arise through the deferred patch path. Storing an owner bin in every entry is consequently a
wide representation of a sparse exception.

### Candidate design

- Store the primary shard stream as `u32` local vertex ids.
- Record foreign-owner references in a sparse sidecar carrying enough information to identify both
  the source/destination cell slot and the foreign `(bin, local)` vertex.
- Assemble local references in one branch-free bulk pass using the current shard's vertex offset.
- Apply foreign references in a separate sparse patch pass.

This differs materially from the retired same-owner branch experiment. That experiment retained
the `u64` stream and added a branch per incidence, so it paid both the wide load and the branch. The
layout proposal realizes the width reduction and keeps the primary loop unconditional.

### Regime and correctness checks

- Expected win: ordinary spatial bins where foreign incidence remains near the measured 0.1--1%.
- Obvious edge cases: 96 bins, concentrated generators on bin boundaries, extremely imbalanced
  ownership, and any future less-spatial ownership policy.
- A 12--16-byte sidecar record remains storage-positive while foreign incidence is below roughly
  25--33%, but random patch stores and destination mapping may become costly much earlier.
- Preserve the full `u32` local range; do not introduce a smaller packed-local representation limit.
- Measure foreign incidence, sidecar bytes, patch locality, live memory, and assembly cycles by
  distribution and bin count.

### Rejected predecessor: source-slot sidecar

An earlier branch narrowed the primary stream but identified foreign corrections only by their
shard-local source slot. Deferred resolution rebuilt per-shard hash maps from those records, and
final assembly recovered the destination cell with a binary search over shard-local cell starts.
That version saved roughly 68 MiB of peak RSS at two million generators but retained too much cold
mapping work. Its initial form added about 1.5% instructions, 2.5--2.9% branches, 2.0% Cachegrind
instruction references, and 5.0% simulated mispredicts. After rebasing onto the accepted
slot-native path, Linux instructions still rose 0.59% and branches 1.46% in every Fibonacci pair.
On the eight-thread Intel Mac it was 0.8% slower on Fibonacci (paired 95% interval +0.1% to +1.5%)
and neutral on uniform.

Controlled current-baseline ablations on 2026-07-17 identified the cause rather than attributing
the change to wall-time noise. Each ablation retained the accepted 12-byte override and changed
one part of the predecessor's cold mapping lifecycle; Linux figures are retired hardware counters
at one million generators, with wall time discarded:

| Reintroduced predecessor behavior | Fibonacci instructions / branches | Uniform, 96 bins | Cachegrind, 20k Fibonacci |
| --- | ---: | ---: | ---: |
| Recover the final destination from `source_slot` and monotonic cell starts | +0.16% / +0.22% | +0.29% / +0.41% | +0.22% instruction refs / +0.29% branches |
| Rebuild per-shard resolution maps before deferred fallback | -0.12% / +1.29% | +0.01% / +1.20% | -0.02% instruction refs / +0.84% branches |
| Sort the sparse 12-byte sidecars before final assembly | +0.19% / +1.50% | +0.31% / +1.37% | +0.32% instruction refs / +0.89% branches |

The factors are not additive because each changes inlining and code layout, but the attribution is
clear. The mandatory sidecar sort alone reproduces the refreshed predecessor's +1.46% branch
penalty; Cachegrind also recorded 4.0% more simulated mispredicts. Source-slot recovery adds a
smaller repeatable cost and couples final assembly to complete monotonic starts even for empty
shard-local cells. Rebuilding the map is instruction-neutral but branch-heavy; it discards work
that the current single allocation-lazy lookup can reuse.

The sort-only ablation was also 1.4% slower on two-million-generator Fibonacci and 0.7% slower on
uniform on the eight-thread Intel Mac (16 interleaved rounds, median comparison). This reproduces
the earlier branch's wall-time shape closely enough to close the discrepancy: the old result was
negative because its source-slot design bundled a mandatory sort, a second sparse lookup
lifecycle, and destination recovery. The accepted design became positive by removing those costs,
not because the same implementation happened to receive a friendlier benchmark sample.

That negative result applies to the source-slot mapping design, not to compact references as a
class. It motivated the accepted implementation below, which carries final source cell/offset
provenance directly, uses an allocation-lazy lookup only while overwrite semantics are live, and
drops that lookup before branch-free assembly.

### Candidate result

The branch implementation stores one shard-local `u32` in the primary stream and appends a
12-byte override only when reconciliation resolves a reference to another shard. A temporary
allocation-lazy map preserves the previous overwrite/conflict semantics on the cold reconciliation
path and is dropped before assembly. The ordinary final scatter is unconditional; a subsequent
sparse pass writes foreign references directly to their final cell/offset destinations. Checked
builds assert that every foreign placeholder has an override before the temporary map is dropped.

Timing telemetry at 100k recorded 5,600 overrides for 599,988 references (0.933%) on Fibonacci with
the default six bins and 26,164 overrides (4.361%) on uniform with 96 bins. Before transient map
overhead, that reduces retained reference storage by about 23.33 and 20.86 bytes per input point,
respectively, relative to the packed-`u64` stream.

Matched native Linux counters at one million points were positive in both regimes. Fibonacci used
0.53% fewer retired instructions, 1.53% fewer branches, and 0.83% fewer branch misses. Uniform with
96 bins used 0.28% fewer instructions, 1.31% fewer branches, and 4.05% fewer branch misses. Host
wall time varied by more than an order of magnitude and was discarded. In a one-thread 20k
Cachegrind run, the final assembly loop performed 3.24% fewer data reads, 36.4% fewer D1 read misses,
22.5% fewer branches, and 45.7% fewer simulated branch mispredicts; whole-program D1 misses fell
1.51%. Simulated last-level movement was mixed under Cachegrind's direct-mapped model and is not an
acceptance claim.

Two reversed-order 2M Fibonacci RSS probes reproduced a reduction from about 625 MiB to 544 MiB,
roughly 81 MiB or 13%. The deliberately unfavorable 96-bin uniform probe remained positive but
smaller, falling from 751.3 MiB to 738.7 MiB because the transient sparse lookup is proportionally
larger. The full release suite and checked assembly tests pass.

The cross-platform gate also passed on the eight-thread Intel i5-1038NG7 Mac using Rust 1.88. In
20-round interleaved 2M runs without preprocessing, Fibonacci fell from a 644.0 ms median to
613.8 ms (4.7% lower latency), default-bin uniform from 818.4 ms to 782.0 ms (4.4% lower), and
96-bin uniform from 870.3 ms to 843.0 ms (3.1% lower). Per-cell spreads were 3.6--5.8%, but all
three paired medians independently favored the candidate by materially more than the Linux
instruction-count improvements alone predicted. The candidate was accepted and merged.

## 4. Slot-native packed groups and cell construction

### Current cost

The driver builds a `packed_queries_all: Vec<u32>` by mapping every bin generator back to its grid
slot, even though each packed group is already one complete, contiguous grid-cell slot range.
Construction also retains global-index-to-slot and global-index-to-cell maps for operations that can
often receive the slot or cell directly from their group.

### Candidate progression

1. Represent a packed group's queries as a contiguous slot start and length; derive query slot as
   `start + query_index` rather than reading a materialized slice.
2. Fetch the generator coordinates from the known slot-ordered storage instead of gathering
   `points[global_index]` in the common path.
3. Store a neighbor slot in forwarded edge checks and recover its global generator id from
   `SlotPoint` where needed.
4. Pass the already-known group cell into shell takeover.
5. If the remaining users are preprocessing or cold repair only, release inverse maps before cell
   construction or compute cold values on demand.

### Regime and correctness checks

- Expected win: packed, cell-major Fibonacci/uniform construction.
- Obvious edge cases: welding/compaction, shell takeover, repair queries, and disabled packed mode.
- This makes the complete-cell, contiguous-slot, and local-id ordering invariants more explicit and
  more deeply load-bearing. Preserve debug assertions at the abstraction boundary.
- Do not replace a 12-byte coordinate load with an unmeasured wider load merely to avoid an index;
  compare SoA query coordinates with `SlotPoint` access.
- Track separately the eliminated allocation/pass, inverse-map lifetime, and scattered generator
  gathers.

### Experimental result (2026-07-15)

The first progression step is implemented on `agent/slot-native-packed-groups`. A packed group now
stores its contiguous slot start and query count instead of borrowing a materialized `Vec<u32>`.
The hot ring pass does not reconstruct slots it only used for a redundant debug assertion; the
group-boundary assertions still verify complete-cell coverage, slot order, and bin/local mapping.

At one million Fibonacci generators, retired instructions fell about 0.36% and branches about
0.64%. The same instruction reduction held for uniform input with 96 bins. Cachegrind at 20k showed
0.44% fewer instruction references, 0.31% fewer data references, 2.6% fewer D1 misses, and 2.1%
fewer last-level data misses. It also showed about 11% more L1 instruction misses and 1.8% more
simulated branch mispredictions, so quiet-machine cycle and wall-time measurements remain required.

Peak RSS at two million Fibonacci generators was repeatably about 548,000 KiB versus 575,000 KiB,
a reduction of roughly 27 MiB. The full checked test suite passes. Keep later progression steps on
separate branches so their gather and inverse-map effects remain attributable.

### Layered recheck after owner-local promotion (2026-07-15)

`agent/slot-groups-on-owner` reapplies this change directly to the accepted owner-local incidence
baseline. Two Windows-native 60-round, two-million-point multithreaded comparisons measured it
0.50% faster on Fibonacci (-1.31% to +0.32%) and 0.53% faster on uniform with 96 bins (-1.11% to
+0.05%). Pooling all 120 paired log-ratios gives a 0.513% improvement with a -1.008% to -0.015%
interval, but neither regime independently excludes zero. A 40-round portable-codegen Fibonacci
guardrail was also directionally 0.44% faster, with a -1.48% to +0.62% interval.

Promote this layer despite its marginal wall-time signal. The stronger prior is favorable: it
removes a per-shard allocation and fill pass, reduces retired instructions and cache traffic, and
represents each packed group using the contiguous slot range already required by its grid-cell
invariant. The resulting production path is smaller and more direct, and the portable guardrail
shows no contrary signal. Treat the measured speedup as supportive rather than precise because
code-layout effects are comparable to an effect of this size.

### Slot-native generator-position result (2026-07-15)

The second progression step is accepted. The driver derives each generator's slot from the known
group start plus its local offset, loads the generator position from the spatially ordered
`SlotPoint` stream, and forwards that value through cell construction. The global generator id
remains unchanged. Checked builds assert both the slot's id and the position's exact f32 bits
against the canonical point, covering packed, packed-slow-path, and packed-disabled groups.

The production profile had placed the scattered `points[generator_idx]` load at the dominant
sampled source location inside cell construction. Reusing the slot-native position removes that
gather from builder reset, shell-frontier creation, and mid-batch bounds. At 2M with twelve threads,
native hardware counters improved materially across both ordinary distributions: Fibonacci cycles
fell 8.41% and cache misses 17.04% over nine agreeing pairs; uniform cycles fell 8.93% and cache
misses 9.19%, also with every pair agreeing. Instructions changed by only -0.12% and -0.23%, which
supports reduced memory stalls rather than changed geometric work. The benefit persisted with
preprocessing and 96 bins (uniform cycles -8.86%, cache misses -13.16% over seven pairs), while a
pinned one-thread Fibonacci guardrail remained favorable. An Intel i5-1038NG7 MacBook Pro on Rust
1.88 independently measured 2M eight-thread wall-time improvements of 2.9% on Fibonacci (95%
interval 1.4--4.3%, 14/16 pairs) and 2.6% on uniform (2.0--3.2%, 16/16 pairs). This completes
progression step 2.

An earlier isolated form reconstructed the generator from the three slot-ordered coordinate SoA
arrays while retaining a packed-path selection. It regressed 1M Fibonacci by about 0.4% retired
instructions and 0.7% branches; Cachegrind also measured 0.4% more instruction references, 0.6%
more data references, and roughly 20% more I1 misses, with data-cache behavior effectively
unchanged. The accepted result above depends on loading the already-fused `SlotPoint` record and
forwarding that position through construction. Do not infer that replacing one scattered point
load with three separate SoA loads is independently beneficial.

### Slot-forwarded edge-check and inverse-map result (2026-07-16)

Progression step 3 is accepted. An in-bin `EdgeCheck` now carries the earlier generator's grid slot
instead of its global id. Seed clipping consumes the slot directly and obtains both position and
global id from the same `SlotPoint` load; cold unmatched-check diagnostics recover their edge key
from that record. The queue record remains 20 bytes, and checked builds assert at the forwarding
boundary that the slot identifies the source generator.

At 2M with twelve threads, nine paired Linux Fibonacci runs reduced cycles 2.28% in eight of nine
pairs despite increasing retired instructions 0.50% and branches 0.67%; this is a locality win, not
an arithmetic-work reduction. The quieter eight-thread Intel Mac independently measured 2.8%
faster Fibonacci wall time (95% interval 1.4--4.2%, 13/16 pairs) and 3.8% faster uniform wall time
(3.3--4.4%, 16/16 pairs).

Progression step 5 is also accepted as a separate lifetime reduction. Weld compaction is the final
consumer of the global-id-to-slot inverse, so construction releases its four-byte-per-generator
buffer before building cells (about 8 MB at 2M). Relative to slot-forwarded checks alone, Linux
instructions and branches changed by less than 0.01%, cycles were neutral (-0.22%), and hardware
cache references/misses fell 2.05%/4.33%. The Mac measured exactly neutral geometric-mean wall time
for both Fibonacci and uniform over 32 pairs each (both 95% intervals -0.5% to +0.6%). The full
checked suite passes with both steps.

The first isolated slot-forwarding prototype predated the surrounding slot-native position and
inverse-map lifetime changes. On that baseline it added about 0.52% retired instructions, 0.50%
branches, and 1.6% branch misses at 1M Fibonacci. Cachegrind reported 0.51% more instruction
references, 0.59% more data references, 5.8% more I1 misses, and 3.9% more mispredicts, despite
2.4% fewer D1 misses. The later accepted composition is therefore a locality/lifetime result, not
evidence that changing the identifier carried by `EdgeCheck` is a standalone arithmetic win.

### Known-cell takeover result — closed neutral 2026-07-15

Progression step 4 was isolated separately. Only when a packed query exhausted did the stream copy
its already-known group cell into the shell frontier; ordinary packed-only cells and non-packed
slow paths were unchanged. Whole-run instructions and branches at one million Fibonacci generators
were identical to measurement precision. Cachegrind recorded only 2,688 additional instruction
references and 1,302 data references across the run, while layout movement increased simulated
branch mispredictions by about 5%. The transition is too rare to justify tighter packed/shell state
coupling, so this remains attribution rather than a production candidate.

### Point-cell inverse-map resurfacing (2026-07-16)

`agent/drop-point-cell-map-v2` reapplies the old point-cell-map experiment to the current
slot-native baseline without its replacement-allocation drawback. After preprocessing, the
per-generator `point_cells` vector is taken from the grid and its allocation is reused as one flat,
bin-ordered stream of non-empty cell ids. A small offset table indexes each bin's run. Construction
gets each group length from the grid CSR offsets, then drops the recycled stream before assembly.
The number of non-empty cells cannot exceed the number of points, so the reused allocation already
has sufficient capacity.

At 1M single-threaded native Fibonacci, nine paired Linux runs reduced retired instructions by
0.483% and branches by 1.432%, with every pair agreeing; cycles remained unresolved. A 96-bin
uniform guardrail reduced instructions by 0.424% and branches by 1.179%, again in every pair, with
unresolved cycles. Portable-codegen Cachegrind at 20k reported 0.441% fewer instruction references,
0.321% fewer data references, 7.26% fewer D1 misses, and 2.15% fewer last-level data misses. The
simulated branch-mispredict count rose 7.71%, while hardware branch misses were noisy and did not
reproduce a stable loss. A 2.5M/12-thread RSS probe showed no recurrence of the old nested-vector
branch's roughly 30 MiB peak regression. The full release suite passes; quiet-Mac wall-time
validation remains pending.

### Slot-derived edge-neighbor global experiment (2026-07-16)

`agent/drop-edge-neighbor-globals` stops writing the hot gnomonic
`CellOutputBuffer::edge_neighbor_globals` stream. Edge collection already needs each neighbor slot
for bin/local classification, so it recovers the global id from that slot's `SlotPoint` record.
Checked/test builds retain the parallel stream for the extraction consistency assertion, and the
rare spherical fallback retains it for its release-mode malformed-attribution check; ordinary
release extraction no longer reserves or writes it.

At 1M single-threaded native Fibonacci, fifteen paired Linux runs reduced retired instructions by
0.494% and branches by 0.389%, with every pair agreeing; cycles were directionally 1.40% lower in
eleven pairs but unresolved. A 96-bin uniform guardrail reduced instructions by 0.441% and branches
by 0.333% in every pair. Portable-codegen Cachegrind was instruction-neutral (+0.003%), reduced data
references by 0.427%, I1 misses by 4.24%, and branches by 0.124%, but increased simulated branch
mispredictions by 3.74%. At 2.5M/12 threads, instructions and branches retained the same reductions
(-0.496%/-0.394% in all nine pairs) while cycles remained unresolved. Quiet-Mac wall-time
validation is required before promotion.

### Combined candidate result (2026-07-16)

`agent/perf-candidates-combined` layers the recycled point-cell runs, the exact tangent-component
weld gate, and slot-derived edge-neighbor ids. Their structural gains compose without visible
interference. At 1M single-threaded native Fibonacci, nine pairs reduced instructions by 3.02% and
branches by 4.35%. At 2.5M/12 threads with preprocessing, Fibonacci reduced instructions by 3.01%
and branches by 4.34% in all nine pairs, cycles by 2.89% in eight, and hardware cache misses by
13.95% in eight. The matching 96-bin uniform run reduced instructions by 2.80% and branches by
3.84% in every pair, cycles by 5.75% in seven, and cache misses by 11.16% in eight. API,
correctness, targeted weld, and the component branches' full-release checks pass. On the quiet
eight-thread Intel Mac at 2.5M with preprocessing, sixteen paired Fibonacci runs measured the
combination 1.3% faster (95% interval 0.4--2.2%, 13/16 favorable); uniform with 96 bins measured
1.6% faster (0.8--2.3%, 13/16 favorable). The combined candidate clears its wall-time promotion
gate.

## 5. Thin per-local edge-check queues

### Current cost

`Vec<Vec<EdgeCheck>>` pays a 24-byte `Vec` header per local generator before payload: about 24 MB per
million generators. The current representation, however, gives each populated queue contiguous
storage, cache-friendly linear lookup, pooled capacity reuse, and zero-copy transfer into the cell
builder.

### Candidate designs

- A thin-vector queue whose slot stores one pointer and whose allocation stores length/capacity.
- A pooled small contiguous allocation that preserves the current take/recycle behavior.
- An arena with compact head/tail metadata only if queue telemetry shows the header saving can repay
  wider nodes and pointer traversal.

Avoid fixed inline payload per generator: even a few inline 24-byte checks multiply into a much
larger always-live array.

### Regime and correctness checks

- Expected win: many empty or tiny queues, especially when metadata dominates live payload.
- Obvious edge cases: high-degree cells, `mega`, few-bin runs with more within-bin forwarding, and
  long-lived queues to far-later generators.
- More bins reduce within-bin checks but increase empty queue metadata and cross-bin overflow; fewer
  bins do the reverse. Sweep both.
- First instrument queue-count, maximum/percentile length, capacity, active lifetime, pool reuse,
  and allocation count. This is primarily a memory-envelope proposal until a design also preserves
  or improves traversal cycles.

### Experimental result — rejected for the default path 2026-07-15

The thin-queue prototype replaced each local slot with one nullable pointer to a boxed
`Vec<EdgeCheck>`, moving queue headers and payload capacity together through the existing
take/recycle path. At two million Fibonacci generators, peak RSS fell repeatably from about
575,000 KiB to 554,600 KiB, roughly 20 MiB.

The indirection cost outweighs that saving for the default throughput path. At one million
Fibonacci generators, instructions rose about 0.9% and branches 2.1%. Cachegrind reported 1.0%
more instruction references, 1.4% more data references, 1.6% more branches, 8.6% more simulated
mispredictions, and 22% more I1 misses. D1 misses fell 1.0% and last-level data misses 6.1%, but
not enough to offset the deterministic front-end work. Retain this only as evidence for an explicit
memory mode or a lower-overhead custom thin allocation.

### Fixed inline-block follow-up — rejected 2026-08-02

A profiling-only census tested whether a lower-overhead thin allocation was now justified. At
2.5M points, all but 3 Fibonacci queues fit eight records; uniform had 8,275 queues above eight
(0.33%) and none above 13. Across 500k clustered, mega, and cubed-sphere cases, only eight queues
exceeded sixteen records. The existing `Vec` capacities observed when queues were taken totaled
2.3--2.9 times the records used, while their 24-byte headers cost 12 MB per 500k generators.

The follow-up replaced each slot with an optional pointer to a pooled block containing eight inline
`EdgeCheck` records and a spill `Vec`. It therefore preserved contiguous iteration and the existing
take/recycle lifecycle while avoiding a second payload allocation for ordinary queues. At 500k
uniform, allocations fell from 38,476 to 29,143, peak heap from 220.28 to 212.68 MB, and
heaptrack-observed RSS from 167.44 to 148.48 MB.

Throughput still rejected the representation. Seven pinned 1M pairs added about 2.2% retired
instructions; Fibonacci cycles were neutral (-0.08%), while uniform regressed 0.72%. At 2.5M/16
threads, Fibonacci was cycle-neutral and uniform appeared 1.17% favorable amid noisy task-clock,
but both retained the roughly 2.1% instruction penalty. This confirms that pointer ownership and
queue-state dispatch, rather than `Vec` growth alone, dominate the access cost. Do not revisit a
pointer-owned block queue for the default path. A materially different candidate must keep queue
metadata and ordinary payload access direct, or target an explicit memory mode.

A smaller follow-up retained `Vec<Vec<EdgeCheck>>` and merely gave newly allocated queues capacity
eight instead of allowing the first `push` to choose capacity four. At 500k uniform points this cut
heaptrack's allocation count from 38,476 to 27,611 (-28.2%), with peak heap unchanged and observed
RSS lower. It still failed the throughput guardrail: Cachegrind at 20k Fibonacci reported 0.42%
more instruction references, and seven pinned 1M single-thread pairs were slightly adverse in both
Fibonacci and uniform runs (roughly 0.1--0.3% more cycles). The larger initial allocation is not a
free replacement for geometric `Vec` growth; retain the allocator traffic rather than spend extra
work on every ordinary queue.

## 6. Lower-priority local layout experiments

These may remove load uops or L1 traffic but are less likely to move a true multithreaded memory
ceiling:

- Split `CellOutputBuffer` vertex keys and positions so resolved vertices can read keys without
  eagerly reading positions. About two-thirds of cell-vertex incidences refer to an already-created
  global vertex, but the per-worker buffer has at most 24 entries and normally remains hot in L1.
  The July 2026 split-stream experiment confirmed this is not a useful default layout: at 1M
  Fibonacci it added about 0.35% instructions and 0.25% branches. Cachegrind showed 5,200 fewer D1
  misses but roughly 497k more instruction reads, 215k more I1 misses, and 49k more branch
  mispredicts. The small data-locality gain does not repay the extraction and iteration machinery.
- Narrow sphere-only edge-check seed data if the shared planar/spherical engine can retain one
  coherent API. Splitting seed fields from endpoint-reconciliation payload risks adding queue
  headers or allocations and may not reduce DRAM traffic because both passes occur close together.
  A copy-based split materialized forwarding-generator ids in a reusable `u32` stream before
  spherical clipping while retaining full checks for reconciliation. It added 0.50% instructions,
  0.85% branches, and 2.9% branch misses at 1M Fibonacci without reducing hardware cache misses.
  Cachegrind likewise added 0.47% instruction references, 0.45% data references, 0.72% branches,
  and 5.6% mispredicts; D1 misses fell only 1.3%. Ordinary seed queues are too small to repay the
  copy, so retire this form.
- Make shell-grid visited stamps lazy. Their table scales with grid cells rather than input points
  and is much smaller than attempted-neighbor stamps, so evaluate only after the larger table is
  addressed.

### Lazy shell-stamp experiment — rejected 2026-07-15

Leaving the shell visitation table empty until a frontier first initialized only delayed the
allocation: large Fibonacci and uniform builds eventually entered shell takeover in active
contexts, and 2M peak RSS remained about 575 MiB. Instructions and branches were neutral at the
default bin count and about 0.02% higher with 96 bins. Cachegrind added 0.03% instruction
references, 0.09% data references, 0.36% branches, and 5.0% simulated mispredicts; D1 misses fell
about 1%, but last-level data misses rose about 1.5%. Retain eager allocation outside the rare
takeover path; this table is too small to create a useful memory-envelope win.

## 7. Final output materialization

**Status: shared spatial-order materialization policy candidate (2026-07-17).**

### Current cost and ceiling probe

Live dedup retains vertices and cell references in shard-owned buffers, then assembly allocates and
fills contiguous global vertex, cell-metadata, and cell-index buffers. Final diagram construction
transfers the certified backend `Vec3` generator and vertex allocations directly into packed
`SpherePoint` storage and maps `VoronoiCell` into the private cell-storage type. The public
representation requires contiguous vertex storage and contiguous index windows for `cell()`.

A profiling-only ablation measured the assembly destination writes without changing candidate or
topology work. Both modes ran normal preprocessing, construction, construction-time reconciliation,
source-buffer reads, vertex-offset and cell-prefix arithmetic, incidence reduction, sparse foreign
reference patch arithmetic, and matched checksum work. The write mode additionally allocated and
filled the three global buffers; the null mode stopped before post-assembly reconciliation and
diagram construction. Output counts and checksums matched at 100k and 1M.

At one million points on single-threaded Linux, the null mode removed 0.605% of retired instructions
and 1.233% of branches on Fibonacci. Uniform with 96 bins removed 0.557% and 1.055%, respectively.
The Fibonacci branch-miss result was mixed while uniform improved, consistent with a small compute
effect rather than eliminated algorithmic work. Cachegrind at 20k Fibonacci measured 0.82% fewer
instruction references, 2.32% fewer data references, 7.41% fewer D1 misses, and 15.19% fewer
last-level data misses in the null mode.

Native Mac wall time supplied the outcome gate. On an eight-thread Intel i5-1038NG7, 16 interleaved
2M rounds reduced the Fibonacci median from 687.1ms to 651.5ms (5.5%) and uniform from 864.2ms to
827.0ms (4.5%). This is a useful bandwidth/cache ceiling, not a forecast: a production diagram
still has to own the final bytes.

### Result

Attribution changed the target. Generator and vertex conversion in
`SphericalVoronoi::from_raw_parts` already transfers the `Vec3` allocations without copying. Two
layout-identical approaches to removing only the cell-metadata conversion reduced retired work but
regressed cache behavior and uniform throughput, so that isolated conversion remains retired.

On the quiet Mac at 2M, vertex concatenation took roughly 10--12ms. Cell-index scatter took about
16--18ms on Fibonacci and 42--48ms on uniform, making it the material component that could be
changed without redesigning the output. There are two opposing locality choices:

- generator order writes the final index vector sequentially but jumps among shard-local sources;
- shard order reads each shard stream sequentially but scatters writes into the already assigned
  final cell spans.

The accepted implementation samples up to 32 adjacent generator ids per shard. It uses shard order
when the sampled mean absolute id delta is greater than 1% of the input size, and otherwise retains
generator order. Measured means were about 0.2% for Fibonacci and 7% for uniform, with clustered
and mega also above the gate and cubed-sphere below it. The decision costs at most 32 comparisons
per shard; it neither scans all cells nor changes any public order, offset, or storage contract.

Twenty 2M multithreaded Mac pairs left Fibonacci neutral and improved uniform by 1.12% (paired 95%
interval 0.58--1.66%). At 1M single-threaded, uniform improved 2.72%; Fibonacci was unresolved with
a roughly 0.3% median regression and an interval spanning neutral. Linux fixed-work counters
reduced instructions by 0.25% on Fibonacci and 0.55% on uniform in all nine pairs. The larger null
write result remains only an upper bound: direct flat backing stores or segmented public storage
would add substantially more coupling for an unproven residual benefit.

The follow-up grid experiment established that this is broader than one final-scatter exception.
Both grid construction and final output now use the same classification rule: sample adjacent
addresses in the current traversal, and treat their order as scrambled when the mean absolute
delta exceeds 1% of the destination/source domain. The two boundaries apply that result differently:

- grid construction keeps its fused input-order scatter for correlated cell-major input, but for
  scrambled input scatters ids first and then writes XYZ sequentially in spatial-slot order;
- final assembly retains generator order when spatial shard sources correlate with generator ids,
  and otherwise streams each shard source while scattering into fixed generator-ordered spans.

On the quiet Mac, the final combined policy left 2M multithreaded Fibonacci neutral and improved
uniform by 2.46% geometrically (ratio 0.9754, interval 0.9685--0.9824, 20/20 favorable). A focused
30-pair Fibonacci control was neutral. The final code was also neutral single-threaded for both
Fibonacci and uniform; with preprocessing disabled, multithreaded Fibonacci remained neutral and
uniform improved by about 3.1% by median. The already cell-major `cubed` guard was neutral in the
preceding policy build. The grid half adds roughly 0.2% retired instructions to buy lower
multithreaded cache traffic; it is a measured locality win rather than a retired-work reduction.
The policy is based on address correlation, not benchmark distribution names, and leaves every
public order and storage contract unchanged.

### Policy audit and remaining boundaries

The two accepted candidates cover the largest measured conversions between spatial and identity
order, but they do not exhaust the pattern. Most cell construction is already unconditionally
slot/bin ordered, so detection is useful only where a global-order storage contract creates a
competing traversal. The strongest residual candidate was final cell metadata: the current
generator-order prefix loop writes `VoronoiCell` records sequentially while reading cell counts
through shard-local addresses. A shard-order count scatter followed by the mandatory sequential
prefix pass is the direct analogue of adaptive index assembly. On a local 500k timing run, the
prefix phase was 6.1ms on Fibonacci and 7.8ms on uniform, versus 16.5ms and 10.7ms for cell-index
assembly, respectively. This is a credible but smaller ceiling.

Bin assignment is a lower-confidence residual. Its spatial traversal appends `bin_generators` and
writes `slot_gen_map` sequentially, but scatters `generator_bin` and `generator_layout` by global
id. Making those inverse arrays global-order would require retaining or rebuilding another inverse
map or adding a pass, so the extra work may exceed the locality benefit. The point-to-slot inverse
fusion was first measured and reverted; the subsequent ownership audit found that no consumer
remained and removed the inverse entirely, including its obsolete compaction maintenance. Batched
locator query reordering is a separate API workload, not a continuation of compute-pipeline
materialization.

The first shared classifier is intentionally small, but should not be mistaken for a final cost
model. It shares only a normalized mean-delta threshold; each caller still owns its sampling, and
the same one-percent crossover stands in for different read/write widths and pass costs. Numeric
grid-cell deltas are only a proxy for slot addresses when cells are sparse or unevenly occupied,
and shard-to-global deltas do not detect a global traversal that round-robins among many otherwise
ordered shard streams. Before this audit, domains below about 100 entries could also classify
unit-stride order as scrambled because one percent was less than one element, while grid-rebuild
timing combined samples from different resolutions instead of describing the retained grid. The
minimal changes below close those two concrete defects; the broader cost-model limitations remain.

The abstraction experiment confirmed that measurement and choice should be conceptually separate,
but the generalized `LocalitySample` implementation was not retained. The former single fixed
position per stratum aliased a synthetic two-points-per-cell sequence; distributed short blocks
fixed that sampling defect, but the larger generic sampler perturbed hot codegen. Sampling actual
CSR slot offsets after the grid prefix was likewise semantically more precise than numeric cell ids
but added a repeatable 0.228% whole-build instructions (about 16 per generator) on Fibonacci and
uniform. A trial flattened generator-to-shard source address was also not useful: at 500k its mean
delta was about 146,813 for Fibonacci and 161,851 for uniform, failing to distinguish the regimes
that the shard-to-global destination sample separates cleanly (about 443 versus 36,979).

Keep site-specific sampling and boundary-specific choices. The shared classifier now only adds an
absolute one-element floor so unit-stride domains below 100 entries remain correlated. Grid rebuild
telemetry retains the selected final sample instead of combining resolutions, and the grid decision
is included in `TIMING_KV`. This minimal form reduced 1M Linux instructions by 0.162% on Fibonacci
and 0.147% on uniform, and branches by 0.382%/0.339%, in all three confirmation pairs. The quiet-Mac
2M multithreaded guardrail was neutral: Fibonacci medians were 641.4ms/641.0ms and uniform's paired
ratio was 0.9972 with a 0.9885--1.0059 interval (12/20 favorable).

The cell-count/prefix candidate was implemented and then retired. For scrambled input it scattered
the native one-byte shard counts into generator order, followed by the mandatory sequential global
prefix; correlated input kept the old fused gather/prefix. The one-byte form was structurally
neutral on 1M single-threaded Linux (instructions -0.008%, branches -0.013%) and directionally
reduced cycles 1.47%, cache references 8.18%, and cache misses 5.28% over five pairs. That signal
did not become a quiet-Mac outcome win: the final 20-pair 2M multithreaded uniform result was a
0.9985 candidate/base ratio with a 0.9906--1.0065 interval and 11/20 favorable; Fibonacci was
neutral. An earlier two-byte prototype produced one favorable Mac run but regressed Linux ST cache
traffic and did not survive the native-width refinement. With no retired-work reduction or
confirmed outcome benefit, another materialization path is not justified. Keep the simpler fused
cell-prefix loop.

Do not couple the clipper to final addresses merely to chase the full null-write percentage. Counts
and offsets are not known until shard construction completes, and a counting prepass or synchronized
global arena can easily repay the saved copy with duplicated geometry work, contention, or unstable
addresses.

## Ideas currently disfavored

- Do not replace the full cube-grid neighbor/ring-2 tables with a boundary-only
  exception representation. On the 16-core Ryzen at resolution 131, the prototype
  reduced retained topology from about 9.9 MiB to about 1 MiB, but its lookup and
  interior-array materialization added 0.11--0.14% retired instructions and branches.
  Pinned 1M runs reduced cache references 3.4--4.8% while increasing cache misses about
  3%; 2.5M multithreaded cell construction was neutral on Fibonacci and about 1.6%
  slower on uniform. The full tables are predictably traversed and cache-friendlier than
  their size alone suggests. Raw runs: `/tmp/s2-grid-boundary-mt.raw` and
  `/tmp/s2-grid-boundary-st-perf.raw` (2026-08-03).
- Do not fuse point classification with worker-local grid histogram counting in its
  straightforward form. It removes one input-sized classification read, but the native
  16-thread grid phase was about 3% slower on both ordinary distributions. A large
  uniform whole-build movement was downstream code-placement noise because the attributed
  grid phase moved in the opposite direction. Raw runs:
  `/tmp/s2-grid-fused-mt.raw` and `/tmp/s2-grid-fused-phase.raw` (2026-08-03).
- Do not narrow per-worker grid histograms to `u16` with an exact overflow fallback.
  On the 2.5M/16-worker grid, halving histogram storage saved only about 0.2ms in the
  prefix phase, while the required increment overflow guard added about 0.5ms on
  Fibonacci and 1.3ms on uniform. Fifteen-pair whole-build ratios were 1.014/1.015
  and unresolved-to-adverse. The histogram byte footprint is not the limiting part
  of this random increment stream. Raw run: `/tmp/s2-grid-narrow-counts.raw`
  (2026-08-03).
- Do not isolate overlapped topology onto a private one-worker Rayon pool. Preserving
  every global-pool worker for point permutation makes topology too slow to remain
  hidden: against the accepted shared-pool overlap, fifteen-pair grid time regressed
  29.4% on Fibonacci and 5.1% on uniform. The accepted high-core schedule relies on
  work stealing across the shared pool. Raw run: `/tmp/s2-grid-isolated-topology.raw`
  (2026-08-03).

- Do not merge the point-coordinate SoA and selected-neighbor `SlotPoint` AoS without a new access
  strategy; query SIMD and random selected-neighbor gathers want different layouts.
- Do not recombine the gnomonic half-plane and extraction metadata streams; the measured wider AoS
  record lost on cache traffic and cycles.
- Do not reduce the speed-oriented shard vertex/key reserve as a claimed throughput optimization.
  Smaller factors reduce RSS and page faults but have already failed to demonstrate a default speed
  win; they remain suitable for an explicit memory mode.
- Do not add a per-reference same-owner branch to the existing `u64` stream. The measured hit rate
  was extremely high and the branch still regressed; a successful compact-reference experiment must
  actually narrow the primary stream and isolate sparse exceptions.
