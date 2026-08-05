//! Disjoint-set (union-find) with path compression.

/// Sparse disjoint-set over a large implicit id space: ids not present in
/// the map are their own roots, so construction is O(1) regardless of the
/// universe size. Union-by-rank ties keep the first root, while
/// [`union_keep_min`](Self::union_keep_min) explicitly keeps the smaller id.
///
/// Lookups never iterate the map, so map ordering cannot leak into results.
#[derive(Debug, Default)]
pub(crate) struct SparseUnionFind {
    /// `(parent, rank)`; a missing key reads as `(self, 0)`.
    nodes: rustc_hash::FxHashMap<u32, (u32, u8)>,
}

impl SparseUnionFind {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn find(&mut self, x: u32) -> u32 {
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

    /// All ids with stored parent or rank state, sorted for determinism.
    ///
    /// This includes every id whose representative can differ from itself. A
    /// minimum representative used only as an implicit root can remain absent.
    pub(crate) fn touched_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.nodes.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// The smaller index always becomes the representative.
    /// Returns `true` if `a` and `b` were in different sets.
    pub(crate) fn union_keep_min(&mut self, a: u32, b: u32) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        let (min, max) = if ra <= rb { (ra, rb) } else { (rb, ra) };
        self.set_parent(max, min);
        true
    }

    /// Union by rank, keeping `a`'s root when the ranks are equal.
    /// Returns `true` if `a` and `b` were in different sets.
    pub(crate) fn union(&mut self, a: u32, b: u32) -> bool {
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

    #[test]
    fn union_modes_pin_representatives_and_touched_ids() {
        let mut ranked = SparseUnionFind::new();
        assert!(ranked.union(8, 3));
        assert!(ranked.union(4, 5));
        assert!(ranked.union(4, 8));
        assert!(!ranked.union(3, 5));
        for id in [3, 4, 5, 8] {
            assert_eq!(ranked.find(id), 4);
        }
        assert_eq!(ranked.touched_ids(), [3, 4, 5, 8]);
        assert_eq!(ranked.find(99), 99);
        assert_eq!(ranked.touched_ids(), [3, 4, 5, 8]);

        let mut minimum = SparseUnionFind::new();
        assert!(minimum.union_keep_min(8, 3));
        assert!(minimum.union_keep_min(7, 8));
        assert!(!minimum.union_keep_min(8, 7));
        for id in [3, 7, 8] {
            assert_eq!(minimum.find(id), 3);
        }
        assert_eq!(minimum.touched_ids(), [7, 8]);
    }
}
