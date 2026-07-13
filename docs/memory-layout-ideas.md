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

## 2. Owner-local vertex-incidence accounting

### Current cost

With repair enabled, every build performs a post-assembly low-incidence scan over the live cell
windows. The parallel path allocates one `AtomicU32` per assembled vertex and performs roughly six
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
- Disable the added accounting when `RepairMode::Disabled`; that configuration currently pays no
  low-incidence scan.
- Measure whether moving one cheap increment per incidence into the dominant construction/dedup
  phase offsets the removed tail pass.
- A saturating byte is valid only for the boolean repair trigger. Do not reuse it for exact degree
  reporting without changing the representation and tests.

## 3. Compact shard-local cell-reference stream

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

## 6. Lower-priority local layout experiments

These may remove load uops or L1 traffic but are less likely to move a true multithreaded memory
ceiling:

- Split `CellOutputBuffer` vertex keys and positions so resolved vertices can read keys without
  eagerly reading positions. About two-thirds of cell-vertex incidences refer to an already-created
  global vertex, but the per-worker buffer has at most 24 entries and normally remains hot in L1.
- Narrow sphere-only edge-check seed data if the shared planar/spherical engine can retain one
  coherent API. Splitting seed fields from endpoint-reconciliation payload risks adding queue
  headers or allocations and may not reduce DRAM traffic because both passes occur close together.
- Make shell-grid visited stamps lazy. Their table scales with grid cells rather than input points
  and is much smaller than attempted-neighbor stamps, so evaluate only after the larger table is
  addressed.

## Ideas currently disfavored

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

