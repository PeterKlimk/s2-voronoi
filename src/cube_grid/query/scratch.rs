use std::cmp::Reverse;

use super::super::{CubeMapGridScratch, OrdF32};

impl CubeMapGridScratch {
    pub(in crate::cube_grid) const MAX_TRACK: usize = 128;

    pub fn new(num_cells: usize) -> Self {
        Self {
            visited_stamp: vec![0; num_cells],
            stamp: 0,
            cell_heap: std::collections::BinaryHeap::new(),
            track_limit: 0,
            candidates_fixed: [(f32::INFINITY, 0); Self::MAX_TRACK],
            candidates_len: 0,
            use_fixed: true,
            candidates_vec: Vec::new(),
            exhausted: false,
        }
    }

    #[inline]
    pub(super) fn begin_query(&mut self, k: usize, track_limit: usize) {
        self.cell_heap.clear();
        self.exhausted = false;
        self.track_limit = track_limit;
        self.candidates_len = 0;
        self.use_fixed = track_limit <= Self::MAX_TRACK;
        self.candidates_vec.clear();
        if !self.use_fixed {
            let reserve = track_limit.max(k);
            if self.candidates_vec.capacity() < reserve {
                self.candidates_vec
                    .reserve(reserve - self.candidates_vec.capacity());
            }
        }

        // Stamp 0 means "unvisited". Avoid ever using stamp 0 for a query.
        self.stamp = self.stamp.wrapping_add(1).max(1);
        if self.stamp == u32::MAX {
            self.visited_stamp.fill(0);
            self.stamp = 1;
        }
    }

    #[inline]
    pub(super) fn mark_visited(&mut self, cell: u32) -> bool {
        let idx = cell as usize;
        if self.visited_stamp[idx] == self.stamp {
            return false;
        }
        self.visited_stamp[idx] = self.stamp;
        true
    }

    #[inline]
    pub(super) fn push_cell(&mut self, cell: u32, bound_dist_sq: f32) {
        self.cell_heap
            .push(Reverse((OrdF32::new(bound_dist_sq), cell)));
    }

    #[inline]
    pub(super) fn peek_cell(&self) -> Option<(f32, u32)> {
        self.cell_heap
            .peek()
            .map(|Reverse((bound, cell))| (bound.get(), *cell))
    }

    #[inline]
    pub(super) fn pop_cell(&mut self) -> Option<(f32, u32)> {
        self.cell_heap
            .pop()
            .map(|Reverse((bound, cell))| (bound.get(), cell))
    }

    #[inline]
    pub(super) fn kth_dist_sq(&self, k: usize) -> f32 {
        if self.use_fixed {
            self.candidates_fixed[k - 1].0
        } else {
            self.candidates_vec[k - 1].0
        }
    }

    #[inline]
    pub(super) fn have_k(&self, k: usize) -> bool {
        if self.use_fixed {
            self.candidates_len >= k
        } else {
            self.candidates_vec.len() >= k
        }
    }

    pub(super) fn try_add_neighbor(&mut self, slot: u32, dist_sq: f32) {
        let len = if self.use_fixed {
            self.candidates_len
        } else {
            self.candidates_vec.len()
        };
        let k = self.track_limit;

        if len < k {
            if self.use_fixed {
                self.candidates_fixed[len] = (dist_sq, slot);
                self.candidates_len += 1;
            } else {
                self.candidates_vec.push((dist_sq, slot));
            }

            // Just reached k: sort in-place.
            if len + 1 == k {
                if self.use_fixed {
                    let slice = &mut self.candidates_fixed[..k];
                    slice.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                } else {
                    self.candidates_vec
                        .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                }
            }
            return;
        }

        if dist_sq >= self.kth_dist_sq(k) {
            return;
        }

        // Replace worst, then rescan to find new worst (k is small: 12/24/48).
        if self.use_fixed {
            self.candidates_fixed[k - 1] = (dist_sq, slot);
            let slice = &mut self.candidates_fixed[..k];
            slice.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        } else {
            self.candidates_vec[k - 1] = (dist_sq, slot);
            self.candidates_vec
                .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }
    }

    #[inline]
    pub(super) fn copy_k_slots_into(&self, k: usize, out: &mut Vec<u32>) {
        out.clear();

        let k = if self.use_fixed {
            k.min(self.candidates_len)
        } else {
            k.min(self.candidates_vec.len())
        };

        out.reserve(k);
        if self.use_fixed {
            out.extend(self.candidates_fixed[..k].iter().map(|&(_, slot)| slot));
        } else {
            out.extend(self.candidates_vec[..k].iter().map(|&(_, slot)| slot));
        }
    }
}
