//! Shell-expansion takeover frontier: Chebyshev BFS layers over the cube-map
//! cell adjacency, replacing the best-first heap cursor.
//!
//! Like the cursor, this is a full re-coverage traversal: when the packed
//! stages exhaust without proving safety, the takeover re-walks from the home
//! cell and may re-emit points the packed stages already served (the consumer
//! dedups). Emission is per-layer, sorted nearest-first within the layer; the
//! certificate after a layer is the conservative cap bound over the next BFS
//! ring, which dominates everything beyond it for the same geometric reason
//! the cursor's frontier-heap top does (cells behind the ring are farther
//! from the query than the ring cells that occlude them).

use glam::Vec3;

use super::super::{CubeMapGrid, CubeMapGridScratch};
use super::directed::{DirectedCellMode, DirectedEligibility};
use crate::fp::{self, OrdF32};

fn sort_pending(pending: &mut [(OrdF32, u32)]) {
    pending.sort_unstable_by_key(|&(dot, slot)| (std::cmp::Reverse(dot), slot));
}

pub(crate) struct ShellBatch {
    pub(crate) n: usize,
    pub(crate) first_dot: f32,
    pub(crate) unseen_bound: f32,
}

pub(crate) struct ShellFrontier<'a, 'b> {
    grid: &'a CubeMapGrid,
    scratch: &'a mut CubeMapGridScratch,
    query: Vec3,
    query_idx: usize,
    start_cell: u32,
    eligibility: DirectedEligibility<'b>,
    initialized: bool,
    pending_bound: f32,
    has_pending: bool,
    exhausted: bool,
}

impl<'a, 'b> ShellFrontier<'a, 'b> {
    pub(crate) fn new(
        grid: &'a CubeMapGrid,
        query: Vec3,
        query_idx: usize,
        scratch: &'a mut CubeMapGridScratch,
        eligibility: DirectedEligibility<'b>,
    ) -> Self {
        Self {
            grid,
            scratch,
            query,
            query_idx,
            start_cell: u32::MAX,
            eligibility,
            initialized: false,
            pending_bound: -1.0,
            has_pending: false,
            exhausted: false,
        }
    }

    #[inline]
    fn initialize(&mut self) {
        debug_assert!(!self.initialized);
        self.start_cell = if self.query_idx < self.grid.point_cells.len() {
            self.grid.point_cells[self.query_idx]
        } else {
            self.grid.point_to_cell(self.query) as u32
        };
        self.scratch.stamp = self.scratch.stamp.wrapping_add(1).max(1);
        if self.scratch.stamp == u32::MAX {
            self.scratch.visited_stamp.fill(0);
            self.scratch.stamp = 1;
        }
        self.scratch.mark_visited(self.start_cell);
        self.scratch.current.clear();
        self.scratch.current.push(self.start_cell);
        self.scratch.next.clear();
        self.scratch.pending.clear();
        self.initialized = true;
    }

    /// Gather the eligible points of one cell into `pending`.
    fn scan_cell(&mut self, cell: usize, mode: DirectedCellMode) {
        if mode == DirectedCellMode::TransitOnly {
            return;
        }
        let start = self.grid.cell_offsets[cell] as usize;
        let end = self.grid.cell_offsets[cell + 1] as usize;
        let xs = &self.grid.cell_points_x[start..end];
        let ys = &self.grid.cell_points_y[start..end];
        let zs = &self.grid.cell_points_z[start..end];
        let indices = &self.grid.point_indices[start..end];
        let (qx, qy, qz) = (self.query.x, self.query.y, self.query.z);

        for i in 0..xs.len() {
            if indices[i] as usize == self.query_idx {
                continue;
            }
            let slot = (start + i) as u32;
            if mode == DirectedCellMode::EmitCenterDirected
                && !self.eligibility.allows_center_slot(slot)
            {
                continue;
            }
            let dot = fp::dot3_f32(xs[i], ys[i], zs[i], qx, qy, qz);
            self.scratch.pending.push((OrdF32::new(dot), slot));
        }
    }

    /// Build the next non-empty layer batch, or mark exhaustion.
    fn build_pending(&mut self) {
        debug_assert!(!self.has_pending);
        while !self.scratch.current.is_empty() {
            // Discover the next ring and its certificate while scanning the
            // current ring's points.
            self.scratch.pending.clear();
            self.scratch.next.clear();
            let mut next_min_dist_sq = f32::INFINITY;

            for layer_idx in 0..self.scratch.current.len() {
                let cell = self.scratch.current[layer_idx];
                let mode = self.eligibility.cell_mode(
                    &self.grid.cell_offsets,
                    self.start_cell,
                    cell as usize,
                );
                self.scan_cell(cell as usize, mode);

                for &ncell in self.grid.cell_neighbors(cell as usize) {
                    if ncell == u32::MAX || !self.scratch.mark_visited(ncell) {
                        continue;
                    }
                    self.scratch.next.push(ncell);
                    let bound = self.grid.cell_min_dist_sq(self.query, ncell as usize);
                    next_min_dist_sq = next_min_dist_sq.min(bound);
                }
            }

            std::mem::swap(&mut self.scratch.current, &mut self.scratch.next);
            self.pending_bound = if self.scratch.current.is_empty() {
                -1.0
            } else {
                (1.0 - 0.5 * next_min_dist_sq).clamp(-1.0, 1.0)
                    + crate::tolerances::GRID_DOT_BOUND_PAD
            };

            if !self.scratch.pending.is_empty() {
                // Nearest-first within the layer.
                sort_pending(&mut self.scratch.pending);
                self.has_pending = true;
                return;
            }
        }
        self.exhausted = true;
    }

    /// Current frontier: fills `out` with the pending layer's slots.
    /// Returns `None` when the traversal is exhausted.
    pub(crate) fn frontier(&mut self, out: &mut Vec<u32>) -> Option<ShellBatch> {
        if !self.initialized {
            self.initialize();
        }
        if !self.has_pending && !self.exhausted {
            self.build_pending();
        }
        if self.exhausted {
            return None;
        }
        out.clear();
        out.extend(self.scratch.pending.iter().map(|&(_, slot)| slot));
        Some(ShellBatch {
            n: self.scratch.pending.len(),
            first_dot: self.scratch.pending[0].0.get(),
            unseen_bound: self.pending_bound,
        })
    }

    pub(crate) fn advance(&mut self) {
        self.has_pending = false;
        self.scratch.pending.clear();
    }

    #[inline]
    pub(crate) fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    #[cfg(test)]
    pub(super) fn is_initialized(&self) -> bool {
        self.initialized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_order_breaks_equal_dots_by_slot() {
        let mut pending = vec![
            (OrdF32::new(0.5), 7),
            (OrdF32::new(0.75), 9),
            (OrdF32::new(0.5), 2),
            (OrdF32::new(0.75), 3),
        ];
        sort_pending(&mut pending);
        let slots: Vec<u32> = pending.into_iter().map(|(_, slot)| slot).collect();
        assert_eq!(slots, vec![3, 9, 2, 7]);
    }
}

impl CubeMapGrid {
    #[inline]
    pub(crate) fn shell_frontier<'a, 'b>(
        &'a self,
        query: Vec3,
        query_idx: usize,
        scratch: &'a mut CubeMapGridScratch,
        eligibility: DirectedEligibility<'b>,
    ) -> ShellFrontier<'a, 'b> {
        ShellFrontier::new(self, query, query_idx, scratch, eligibility)
    }
}
