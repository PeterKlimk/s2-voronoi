# Claims and scope

This is a claim ledger, not manuscript prose. Labels indicate the present evidentiary status.

## Working thesis

`voronoi-mesh` is a high-throughput shared-memory CPU system for constructing explicit spherical
Voronoi meshes. It couples ownership-filtered nearest-neighbor search with same-shard edge
forwarding to avoid redundant work, while retaining cross-shard parallelism and producing either a
mesh satisfying stated topology checks or an explicit classified outcome.

Status: **candidate**. The mechanism is present in the implementation, but novelty and the exact
performance/correctness wording still require literature review and experiments.

## Candidate contributions

### Ownership-filtered neighbor search

Within a deterministically ordered shard, a later generator omits earlier same-shard generators
from its ordinary neighbor stream because those pairs are owned by the earlier generator.

Preferred term: **ownership-filtered nearest-neighbor search** or **order-restricted neighbor
search**. Avoid bare "directional nearest neighbor," which can be read as a geometric query type.

Status: **implemented; novelty unestablished**.

### Same-shard edge forwarding

When an earlier cell produces an edge shared with a later cell in its shard, it forwards a compact
record containing the constraint/neighbor and endpoint identities. The later cell consumes that
record as an initial construction seed and uses it to coordinate shared topology.

Status: **implemented; novelty unestablished**.

### Coupled ownership schedule

The two mechanisms above form one correctness-dependent contribution. Filtering without a
replacement source of constraints is incomplete; forwarding supplies that source. Cross-shard
pairs remain independently constructed and are matched later, preserving coarse parallelism.

Status: **strongest candidate algorithmic contribution**.

### High-performance explicit topology

The engine combines streamed candidate discovery, packed/SIMD query preparation, gnomonic cell
clipping, sharded live vertex ownership, bounded reconciliation, and local rebuilds into an
end-to-end implementation. A large reproducible speed improvement over established explicit-mesh
implementations can be a systems contribution even when many constituent optimizations are known.

Status: **requires same-machine comparator results and ablations**.

### Correctness contract

The public result is more than independent cell geometry: it contains shared indexed topology and
exposes checks, diagnostics, reconciliation, and classified outcomes for numerical degeneracies.

Status: **implemented; paper wording must match the exact API contract**. Prefer "satisfies the
specified invariants or reports a classified outcome" over an unconditional "always valid."

## Claims currently ruled out

- First CPU implementation of Ray et al.'s construction.
- First spherical half-space clipping implementation.
- First implementation producing an explicit mesh from independently clipped cells.
- First use of sorted incident-generator triples to identify/deduplicate Voronoi vertices.
- A direct speedup over a published result measured on different hardware.

Caplan et al./Vortex already cover the first three areas and their optional mesh merge uses sorted
generator triples. See [prior-work.md](prior-work.md).

## Questions the paper must answer

1. How much candidate discovery and clipping work does the coupled ownership schedule remove?
2. How much does explicit topology, reconciliation, and checking cost relative to cell geometry?
3. Does the performance advantage persist across point distributions, sizes, and thread counts?
4. Which individual optimizations materially affect end-to-end runtime or memory?
5. What invariants are guaranteed on success, and how are unresolved degeneracies represented?
6. Against which comparator does each output tier constitute an apples-to-apples comparison?
