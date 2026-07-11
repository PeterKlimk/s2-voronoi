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
//! from the query than the ring cells that occlude them). Small layers are
//! sorted whole; large layers are partitioned into nearest-first prefixes so
//! mid-layer cell closure does not pay to sort an unused suffix.

use glam::Vec3;

use super::super::{CubeMapGrid, CubeMapGridScratch};
use super::directed::{DirectedCellMode, DirectedEligibility};
use crate::fp::{self, OrdF32};

pub(crate) trait ShellEligibility: Copy {
    fn cell_mode(self, cell_offsets: &[u32], start_cell: u32, cell: usize) -> DirectedCellMode;

    fn allows_center_slot(self, slot: u32) -> bool;
}

impl ShellEligibility for DirectedEligibility<'_> {
    #[inline]
    fn cell_mode(self, cell_offsets: &[u32], start_cell: u32, cell: usize) -> DirectedCellMode {
        DirectedEligibility::cell_mode(self, cell_offsets, start_cell, cell)
    }

    #[inline]
    fn allows_center_slot(self, slot: u32) -> bool {
        DirectedEligibility::allows_center_slot(self, slot)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct UnrestrictedEligibility;

impl ShellEligibility for UnrestrictedEligibility {
    #[inline]
    fn cell_mode(self, _cell_offsets: &[u32], _start_cell: u32, _cell: usize) -> DirectedCellMode {
        DirectedCellMode::EmitAll
    }

    #[inline]
    fn allows_center_slot(self, _slot: u32) -> bool {
        true
    }
}

fn sort_pending(pending: &mut [(OrdF32, u32)]) {
    pending.sort_unstable_by_key(|&(dot, slot)| (std::cmp::Reverse(dot), slot));
}

const INCREMENTAL_LAYER_CHUNK: usize = 64;

fn prepare_pending_prefix(pending: &mut [(OrdF32, u32)]) -> usize {
    if pending.len() <= 2 * INCREMENTAL_LAYER_CHUNK {
        sort_pending(pending);
        return pending.len();
    }

    pending.select_nth_unstable_by_key(INCREMENTAL_LAYER_CHUNK, |&(dot, slot)| {
        (std::cmp::Reverse(dot), slot)
    });
    let prefix_len = INCREMENTAL_LAYER_CHUNK;
    sort_pending(&mut pending[..prefix_len]);
    prefix_len
}

pub(crate) struct ShellBatch {
    pub(crate) n: usize,
    pub(crate) first_dot: f32,
    pub(crate) unseen_bound: f32,
}

pub(crate) struct ShellFrontier<'a, E: ShellEligibility> {
    grid: &'a CubeMapGrid,
    scratch: &'a mut CubeMapGridScratch,
    query: Vec3,
    query_idx: usize,
    start_cell: u32,
    eligibility: E,
    initialized: bool,
    pending_bound: f32,
    pending_pos: usize,
    pending_prefix_len: usize,
    has_pending: bool,
    exhausted: bool,
}

impl<'a, E: ShellEligibility> ShellFrontier<'a, E> {
    pub(crate) fn new(
        grid: &'a CubeMapGrid,
        query: Vec3,
        query_idx: usize,
        scratch: &'a mut CubeMapGridScratch,
        eligibility: E,
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
            pending_pos: 0,
            pending_prefix_len: 0,
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
                self.pending_pos = 0;
                self.pending_prefix_len = 0;
                self.has_pending = true;
                return;
            }
        }
        self.exhausted = true;
    }

    /// Current frontier: fills `out` with the pending layer's next sorted
    /// prefix (or the whole layer when small).
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
        if self.pending_prefix_len == 0 {
            self.pending_prefix_len =
                prepare_pending_prefix(&mut self.scratch.pending[self.pending_pos..]);
        }
        let end = self.pending_pos + self.pending_prefix_len;
        let prefix = &self.scratch.pending[self.pending_pos..end];
        out.clear();
        out.extend(prefix.iter().map(|&(_, slot)| slot));
        let same_layer_bound = if end < self.scratch.pending.len() {
            prefix[self.pending_prefix_len - 1].0.get() + crate::tolerances::GRID_DOT_BOUND_PAD
        } else {
            -1.0
        };
        Some(ShellBatch {
            n: self.pending_prefix_len,
            first_dot: prefix[0].0.get(),
            unseen_bound: self.pending_bound.max(same_layer_bound),
        })
    }

    pub(crate) fn advance(&mut self) {
        self.pending_pos += self.pending_prefix_len;
        self.pending_prefix_len = 0;
        if self.pending_pos >= self.scratch.pending.len() {
            self.pending_pos = 0;
            self.has_pending = false;
            self.scratch.pending.clear();
        }
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

impl CubeMapGrid {
    /// Shell traversal that emits every grid point except `query_idx`.
    /// Passing an out-of-range index treats `query` as external to the grid.
    #[inline]
    pub(crate) fn unrestricted_shell_frontier<'a>(
        &'a self,
        query: Vec3,
        query_idx: usize,
        scratch: &'a mut CubeMapGridScratch,
    ) -> ShellFrontier<'a, UnrestrictedEligibility> {
        ShellFrontier::new(self, query, query_idx, scratch, UnrestrictedEligibility)
    }

    /// Nearest grid slot to an external query, reusing `scratch` and `batch`.
    pub(crate) fn nearest_unrestricted_slot(
        &self,
        query: Vec3,
        scratch: &mut CubeMapGridScratch,
        batch: &mut Vec<u32>,
    ) -> Option<u32> {
        let mut frontier =
            self.unrestricted_shell_frontier(query, self.point_indices.len(), scratch);
        let mut best: Option<(f32, u32)> = None;
        while let Some(layer) = frontier.frontier(batch) {
            let candidate = (layer.first_dot, batch[0]);
            if best.is_none_or(|(dot, _)| candidate.0 > dot) {
                best = Some(candidate);
            }
            if best.is_some_and(|(dot, _)| dot >= layer.unseen_bound) {
                break;
            }
            frontier.advance();
        }
        best.map(|(_, slot)| slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_points() -> Vec<Vec3> {
        [
            (1.0, 0.2, 0.1),
            (-0.4, 1.0, 0.3),
            (0.1, -0.6, 1.0),
            (-1.0, -0.2, 0.4),
            (0.3, 0.7, -1.0),
            (0.8, -1.0, -0.5),
        ]
        .into_iter()
        .map(|(x, y, z)| Vec3::new(x, y, z).normalize())
        .collect()
    }

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

    #[test]
    fn incremental_prefixes_match_full_layer_sort() {
        let mut pending: Vec<(OrdF32, u32)> = (0..257u32)
            .map(|slot| {
                let dot = ((slot.wrapping_mul(73) % 41) as f32 - 20.0) / 20.0;
                (OrdF32::new(dot), slot)
            })
            .collect();
        let mut expected = pending.clone();
        sort_pending(&mut expected);

        let mut pos = 0usize;
        while pos < pending.len() {
            let n = prepare_pending_prefix(&mut pending[pos..]);
            assert!(n > 0);
            pos += n;
        }
        assert_eq!(pending, expected);
    }

    #[test]
    fn unrestricted_frontier_emits_every_other_point_with_safe_bounds() {
        let points = fixture_points();
        let grid = CubeMapGrid::new(&points, 3);
        let query_idx = 0usize;
        let query = points[query_idx];
        let mut scratch = grid.make_scratch();
        let mut frontier = grid.unrestricted_shell_frontier(query, query_idx, &mut scratch);
        let mut batch = Vec::new();
        let mut unseen = vec![true; points.len()];
        unseen[query_idx] = false;

        while let Some(layer) = frontier.frontier(&mut batch) {
            for &slot in &batch {
                let idx = grid.point_indices()[slot as usize] as usize;
                assert_ne!(idx, query_idx);
                assert!(std::mem::replace(&mut unseen[idx], false));
            }
            let max_unseen = unseen
                .iter()
                .enumerate()
                .filter(|&(_, &is_unseen)| is_unseen)
                .map(|(idx, _)| query.dot(points[idx]))
                .fold(-1.0f32, f32::max);
            assert!(max_unseen <= layer.unseen_bound + 1e-6);
            frontier.advance();
        }

        assert!(unseen.iter().all(|&is_unseen| !is_unseen));
    }

    #[test]
    fn nearest_unrestricted_slot_matches_brute_force() {
        let points = fixture_points();
        let grid = CubeMapGrid::new(&points, 3);
        let query = Vec3::new(0.37, -0.21, 0.91).normalize();
        let expected = points
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| query.dot(**a).total_cmp(&query.dot(**b)))
            .map(|(idx, _)| idx)
            .unwrap();
        let mut scratch = grid.make_scratch();
        let mut batch = Vec::new();
        let slot = grid
            .nearest_unrestricted_slot(query, &mut scratch, &mut batch)
            .unwrap();

        assert_eq!(grid.point_indices()[slot as usize] as usize, expected);
    }
}
