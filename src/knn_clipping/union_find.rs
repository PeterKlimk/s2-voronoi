//! Disjoint-set (union-find) with path compression.

#[derive(Debug)]
pub(crate) struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

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

    /// Order-dependent union: the smaller index always becomes the representative.
    /// Returns `true` if `a` and `b` were in different sets.
    pub fn union_keep_min(&mut self, a: u32, b: u32) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        let (min, max) = if ra <= rb { (ra, rb) } else { (rb, ra) };
        self.parent[max as usize] = min;
        true
    }
}
