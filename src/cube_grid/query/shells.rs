//! Shell-expansion takeover frontier: Chebyshev BFS layers over the cube-map
//! cell adjacency, replacing the best-first heap cursor (see docs/todo.md P3).
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
    /// BFS layer whose points have not been emitted yet.
    current: Vec<u32>,
    /// Layer discovered while emitting `current`.
    next: Vec<u32>,
    /// Sorted (descending dot, slot) emission for the pending batch.
    pending: Vec<(OrdF32, u32)>,
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
        let start_cell = if query_idx < grid.point_cells.len() {
            grid.point_cells[query_idx]
        } else {
            grid.point_to_cell(query) as u32
        };

        scratch.exhausted = false;
        scratch.stamp = scratch.stamp.wrapping_add(1).max(1);
        if scratch.stamp == u32::MAX {
            scratch.visited_stamp.fill(0);
            scratch.stamp = 1;
        }
        scratch.mark_visited(start_cell);

        Self {
            grid,
            scratch,
            query,
            query_idx,
            start_cell,
            eligibility,
            current: vec![start_cell],
            next: Vec::new(),
            pending: Vec::new(),
            pending_bound: -1.0,
            has_pending: false,
            exhausted: false,
        }
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
            self.pending.push((OrdF32::new(dot), slot));
        }
    }

    /// Build the next non-empty layer batch, or mark exhaustion.
    fn build_pending(&mut self) {
        debug_assert!(!self.has_pending);
        while !self.current.is_empty() {
            // Discover the next ring and its certificate while scanning the
            // current ring's points.
            self.pending.clear();
            self.next.clear();
            let mut next_min_dist_sq = f32::INFINITY;

            for layer_idx in 0..self.current.len() {
                let cell = self.current[layer_idx];
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
                    self.next.push(ncell);
                    let bound = self.grid.cell_min_dist_sq(self.query, ncell as usize);
                    next_min_dist_sq = next_min_dist_sq.min(bound);
                }
            }

            std::mem::swap(&mut self.current, &mut self.next);
            self.pending_bound = if self.current.is_empty() {
                -1.0
            } else {
                (1.0 - 0.5 * next_min_dist_sq).clamp(-1.0, 1.0)
            };

            if !self.pending.is_empty() {
                // Nearest-first within the layer.
                self.pending
                    .sort_unstable_by_key(|&(dot, _)| std::cmp::Reverse(dot));
                self.has_pending = true;
                return;
            }
        }
        self.exhausted = true;
        self.scratch.exhausted = true;
    }

    /// Current frontier: fills `out` with the pending layer's slots.
    /// Returns `None` when the traversal is exhausted.
    pub(crate) fn frontier(&mut self, out: &mut Vec<u32>) -> Option<ShellBatch> {
        if !self.has_pending && !self.exhausted {
            self.build_pending();
        }
        if self.exhausted {
            return None;
        }
        out.clear();
        out.extend(self.pending.iter().map(|&(_, slot)| slot));
        Some(ShellBatch {
            n: self.pending.len(),
            first_dot: self.pending[0].0.get(),
            unseen_bound: self.pending_bound,
        })
    }

    pub(crate) fn advance(&mut self) {
        self.has_pending = false;
        self.pending.clear();
    }

    #[inline]
    pub(crate) fn is_exhausted(&self) -> bool {
        self.exhausted
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
