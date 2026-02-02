# Live vertex deduplication and edge checks

During cell construction, multiple cells discover the same Voronoi vertices. The live-dedup path
deduplicates these vertices without requiring a global concurrent hash map in the hot path.

## Sharded ownership

Generators are assigned to **bins**. Each bin builds all of its local cells (optionally in
parallel across bins).

Vertices (triplets of generators) are assigned an “owner bin” derived from a canonicalized triplet
key. If the owner is local, the bin can allocate/deduplicate the vertex immediately; otherwise, the
cell records a deferred slot to be patched later.

## Why edge checks exist

Even with triplet-key ownership, adjacent cells need to agree on which global vertex index appears
at each edge endpoint. The live-dedup pipeline uses **edge checks** as a compact, cache-friendly way
to propagate indices between adjacent cells while staying shard-local most of the time.

An edge check for an undirected cell edge `(A, B)` stores:

- a canonical edge key `(A, B)`
- the two “third” generator ids `T0/T1` for the edge endpoints (each endpoint vertex is `(A, B, T)`)
- the (possibly deferred) global vertex indices for those endpoints

The earlier-local side emits the check to the later-local neighbor; the later side consumes it to
seed adjacency and to fill in any vertex indices it can prove consistent.

## Overflow matching

If the adjacent cells are in different bins, both sides emit an overflow record. After cell
construction completes, the overflow records are sorted by edge key and matched, allowing the two
bins to patch each other’s deferred slots without global synchronization during the hot loop.

See:

- `src/knn_clipping/live_dedup/edge_checks.rs`
- `src/knn_clipping/live_dedup/assemble.rs`

