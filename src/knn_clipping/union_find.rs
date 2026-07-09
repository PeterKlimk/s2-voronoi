//! Disjoint-set (union-find) with path compression.

/// Dense disjoint-set. No production caller since edge_reconcile moved to
/// [`SparseUnionFind`]; retained as the semantics oracle for the sparse
/// equivalence test.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
pub(crate) struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl UnionFind {
    pub fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n);
        for i in 0..n {
            parent.push(i as u32);
        }
        Self {
            parent,
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, x: u32) -> u32 {
        let idx = x as usize;
        let p = self.parent[idx];
        if p != x {
            let root = self.find(p);
            self.parent[idx] = root;
        }
        self.parent[idx]
    }

    /// Union by rank. Returns `true` if `a` and `b` were in different sets.
    ///
    /// No production caller since edge_reconcile moved to `SparseUnionFind`;
    /// retained as the semantics oracle for the sparse equivalence test.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn union(&mut self, a: u32, b: u32) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        let ra_idx = ra as usize;
        let rb_idx = rb as usize;
        if self.rank[ra_idx] < self.rank[rb_idx] {
            self.parent[ra_idx] = rb;
        } else if self.rank[ra_idx] > self.rank[rb_idx] {
            self.parent[rb_idx] = ra;
        } else {
            self.parent[rb_idx] = ra;
            self.rank[ra_idx] = self.rank[ra_idx].saturating_add(1);
        }
        true
    }
}

/// Sparse disjoint-set over a large implicit id space: ids not present in
/// the map are their own roots, so construction is O(1) regardless of the
/// universe size. Mirrors `UnionFind`'s union-by-rank semantics exactly
/// (same representatives for the same union sequence — pinned by the
/// equivalence test below), so swapping it in for a sparsely-used dense
/// instance is behavior-preserving; only the O(universe) init is avoided.
///
/// Lookups never iterate the map, so map ordering cannot leak into results.
#[derive(Debug, Default)]
pub(crate) struct SparseUnionFind {
    /// `(parent, rank)`; a missing key reads as `(self, 0)`.
    nodes: rustc_hash::FxHashMap<u32, (u32, u8)>,
}

impl SparseUnionFind {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn find(&mut self, x: u32) -> u32 {
        let Some(&(p, _)) = self.nodes.get(&x) else {
            return x;
        };
        if p == x {
            return x;
        }
        let root = self.find(p);
        self.nodes.get_mut(&x).expect("node exists").0 = root;
        root
    }

    fn rank(&self, x: u32) -> u8 {
        self.nodes.get(&x).map_or(0, |&(_, r)| r)
    }

    fn set_parent(&mut self, x: u32, p: u32) {
        self.nodes.entry(x).or_insert((x, 0)).0 = p;
    }

    /// All ids that have entered the structure, sorted for determinism.
    /// Every id that participated in a successful union is included (both
    /// representatives and merged-away ids).
    pub fn touched_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.nodes.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Order-dependent union: the smaller index always becomes the
    /// representative. Returns `true` if `a` and `b` were in different sets.
    pub fn union_keep_min(&mut self, a: u32, b: u32) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        let (min, max) = if ra <= rb { (ra, rb) } else { (rb, ra) };
        self.set_parent(max, min);
        true
    }

    /// Union by rank, with `UnionFind::union`'s exact tie-breaking.
    /// Returns `true` if `a` and `b` were in different sets.
    pub fn union(&mut self, a: u32, b: u32) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        let rank_a = self.rank(ra);
        let rank_b = self.rank(rb);
        if rank_a < rank_b {
            self.set_parent(ra, rb);
        } else {
            self.set_parent(rb, ra);
            if rank_a == rank_b {
                let entry = self.nodes.entry(ra).or_insert((ra, 0));
                entry.1 = entry.1.saturating_add(1);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sparse implementation must produce identical representatives to
    /// the dense one for any union sequence (edge_reconcile relies on this
    /// to make the dense->sparse swap behavior-preserving).
    #[test]
    fn sparse_matches_dense_representatives() {
        const N: u32 = 1000;
        // Deterministic pseudo-random union sequence (LCG).
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) % N
        };

        let mut dense = UnionFind::new(N as usize);
        let mut sparse = SparseUnionFind::new();
        for _ in 0..600 {
            let (a, b) = (next(), next());
            assert_eq!(dense.union(a, b), sparse.union(a, b));
        }
        for x in 0..N {
            assert_eq!(
                dense.find(x),
                sparse.find(x),
                "representative mismatch at {x}"
            );
        }
    }
}
