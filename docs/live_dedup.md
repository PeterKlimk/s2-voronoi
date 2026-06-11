# Live vertex deduplication and edge checks

After single-cell construction extracts each clipped cell, multiple cells discover the same
Voronoi vertices. The live-dedup path deduplicates these vertices without requiring a global
concurrent hash map in the hot path.

## Sharded ownership

Generators are assigned to **bins**. Each bin builds all of its local cells (optionally in
parallel across bins).

Vertices (triplets of generators) are assigned an “owner bin” derived from a canonicalized triplet
key. If the owner is local, the bin can allocate/deduplicate the vertex immediately; otherwise, the
cell records a deferred vertex slot to be patched later.

The current live path is triplet-key only. Near-degenerate multi-generator disagreements are not
materialized as a separate vertex-owner class; they are carried forward as unresolved shared-edge
mismatches for `edge_reconcile`.

## The stitching invariant

This is the argument for why independently built cells compose into one consistent subdivision —
the crate's central design idea. It rests on a total order over generators and a coverage
contract per ordered pair.

**The order.** Every generator has a `(bin, local)` id; locals increase with grid-cell index
within a bin, and a grid cell maps to exactly one bin. This yields a total order in which "all of
my same-bin predecessors finished building before I start" holds, because bins build their cells
sequentially in slot order (parallelism is across bins).

**The coverage contract.** For each pair of generators (g earlier, h later in the order) whose
cells share a Voronoi edge, exactly one of these supplies the constraint to each side:

1. *Cross-bin pairs*: both streams emit the other side (`EmitAll`), so both cells clip the shared
   bisector independently; the overflow match reconciles the resulting vertex indices.
2. *Same-bin pairs*: only the earlier cell's stream emits the later one. The earlier cell clips
   against h, the shared edge becomes an active boundary edge of g's cell, and the edge check for
   it is forwarded to h, which clips the replayed constraint as a **seed** before consuming its
   own stream. The later cell's stream treats earlier same-bin cells as transit-only — that is
   the work-halving, and it is sound because of the next paragraph.

**Why the earlier side always delivers.** If h's true cell shares an edge with g's, then g's
bisector cuts g's cell; g's stream cannot terminate while an uncut relevant point is unseen
(certificate soundness — pinned in isolation by the NN contract suite), so g must have clipped h,
the constraint is active in g's final boundary, and the edge check is emitted. The seed therefore
arrives whenever the earlier side *agrees the edge exists*.

**The epsilon caveat — where reconciliation earns its place.** At epsilon scale the two cells can
disagree: g's clip decisions may conclude the g–h edge does not exist while h's would. Then no
seed is sent, and h never clips g (its stream will not emit an earlier same-bin cell). The
omission is bounded by near-symmetry of the clip decisions — the disputed feature is itself
epsilon-scale — so h's cell is wrong by at most an epsilon feature, and the resulting one-sided
edge is exactly the class of mismatch `edge_reconcile` repairs. This residual asymmetry is the
precise motivation for the eventual canonical-predicate refactor (docs/todo.md P5): decide each
shared edge once, and the caveat disappears.

**Vertex identity is combinatorial.** A vertex is keyed by its sorted generator triple, so cells
that agree a vertex exists agree on its identity exactly, independent of floating-point
positions; dedup picks one representative position per key.

Changing any leg of this — the bin/local order, the seed forwarding, the transit-only rule, the
certificate, or triplet keying — requires revisiting the whole argument, not just the touched
code.

## Why edge checks exist

Even with triplet-key ownership, adjacent cells need to agree on which global vertex index appears
at each edge endpoint. The live-dedup pipeline uses **edge checks** as a compact, cache-friendly way
to propagate indices between adjacent cells while staying shard-local most of the time.

An edge check for an undirected cell edge `(A, B)` stores:

- a canonical edge key `(A, B)`
- the two “third” generator ids `T0/T1` for the edge endpoints (each endpoint vertex is `(A, B, T)`)
- the (possibly deferred) global vertex indices for those endpoints

The earlier-local side emits the check to the later-local neighbor; the later side consumes it to
seed adjacency and to fill in any vertex indices it can prove consistent. The actual clipping and
neighbor-stream logic now lives in `knn_clipping/cell_build`; live dedup consumes the extracted
cell output plus these edge-check records.

## Overflow matching

If the adjacent cells are in different bins, both sides emit an overflow record. After cell
construction completes, the overflow records are sorted by edge key and matched, allowing the two
bins to patch each other’s deferred slots without global synchronization during the hot loop.

This assembly boundary has a narrow contract:

- cross-bin edge checks are matched first and patch any slots that can be reconciled exactly
- any still-deferred vertex slots are then patched via fallback ownership by canonical vertex key
- if an edge still cannot be reconciled cleanly, it is emitted as an unresolved shared-edge
  mismatch for the later `edge_reconcile` pass

So assembly is not a generic repair stage. It performs exact cross-bin agreement first, then a
key-based fallback for unresolved vertex ownership, and only carries forward the narrow edge
mismatches that survive those two steps.

See:

- `src/live_dedup/edge_checks.rs`
- `src/live_dedup/assemble.rs`
