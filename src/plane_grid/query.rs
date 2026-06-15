//! Shell-expansion frontier and directed neighbor stream for [`PlaneGrid`].
//!
//! The planar counterpart of `cube_grid::query::shells` plus a thin stream
//! wrapper (no packed SIMD stage yet — the stream shape leaves room for one).
//! Chebyshev rings around the start cell are enumerated arithmetically: the
//! flat chart needs no BFS, no visited stamps, and no scratch. Emission is
//! per-ring, sorted nearest-first within the ring; the certificate after ring
//! `k` is the squared distance from the query to the boundary of the explored
//! `(2k+1) x (2k+1)` cell box (clipped at the domain edge, beyond which
//! nothing exists), shrunk by the wall-classification slack so it lower-bounds
//! every point classification can place outside the box.

use glam::Vec2;

use super::{PlaneGrid, PlaneGridScratch};
use crate::cube_grid::{DirectedCellMode, DirectedEligibility};
use crate::fp::OrdF32;

pub(crate) struct PlaneShellBatch {
    pub(crate) n: usize,
    pub(crate) first_dist_sq: f32,
    /// Lower bound on the squared distance of every eligible point not yet
    /// emitted (`f32::INFINITY` once nothing unseen remains).
    pub(crate) unseen_bound: f32,
}

pub(crate) struct PlaneShellFrontier<'a, 'b> {
    grid: &'a PlaneGrid,
    /// Borrowed per-query scratch holding the pending emission (sorted
    /// ascending (dist_sq, slot)); reused across cells by the driver.
    scratch: &'a mut PlaneGridScratch,
    query: Vec2,
    query_idx: usize,
    start_cell: u32,
    cx: usize,
    cy: usize,
    /// Next Chebyshev ring to scan.
    ring: usize,
    /// Largest ring with any in-grid cell.
    max_ring: usize,
    eligibility: DirectedEligibility<'b>,
    pending_bound: f32,
    has_pending: bool,
    exhausted: bool,
}

impl<'a, 'b> PlaneShellFrontier<'a, 'b> {
    pub(crate) fn new(
        grid: &'a PlaneGrid,
        query: Vec2,
        query_idx: usize,
        scratch: &'a mut PlaneGridScratch,
        eligibility: DirectedEligibility<'b>,
    ) -> Self {
        let start_cell = if query_idx < grid.point_cells.len() {
            grid.point_cells[query_idx]
        } else {
            grid.point_to_cell(query) as u32
        };
        let res = grid.res();
        let cx = start_cell as usize % res;
        let cy = start_cell as usize / res;
        let max_ring = cx.max(res - 1 - cx).max(cy).max(res - 1 - cy);
        scratch.pending.clear();

        Self {
            grid,
            scratch,
            query,
            query_idx,
            start_cell,
            cx,
            cy,
            ring: 0,
            max_ring,
            eligibility,
            pending_bound: f32::INFINITY,
            has_pending: false,
            exhausted: false,
        }
    }

    /// Gather the eligible points of one cell into `pending`.
    fn scan_cell(&mut self, cell: usize) {
        let mode = self
            .eligibility
            .cell_mode(&self.grid.cell_offsets, self.start_cell, cell);
        if mode == DirectedCellMode::TransitOnly {
            return;
        }
        let start = self.grid.cell_offsets[cell] as usize;
        let end = self.grid.cell_offsets[cell + 1] as usize;
        let xs = &self.grid.cell_points_x[start..end];
        let ys = &self.grid.cell_points_y[start..end];
        let indices = &self.grid.point_indices[start..end];
        let (qx, qy) = (self.query.x, self.query.y);

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
            let (dx, dy) = (xs[i] - qx, ys[i] - qy);
            let dist_sq = dx * dx + dy * dy;
            self.scratch.pending.push((OrdF32::new(dist_sq), slot));
        }
    }

    /// Visit every in-grid cell of Chebyshev ring `k` around the start cell.
    fn for_each_ring_cell(&mut self, k: usize) {
        let res = self.grid.res();
        if k == 0 {
            self.scan_cell(self.start_cell as usize);
            return;
        }
        let (cx, cy) = (self.cx as isize, self.cy as isize);
        let k = k as isize;
        let (x0, x1) = (cx - k, cx + k);
        let (y0, y1) = (cy - k, cy + k);
        let xa = x0.max(0) as usize;
        let xb = x1.min(res as isize - 1) as usize;

        if y0 >= 0 {
            let row = y0 as usize * res;
            for x in xa..=xb {
                self.scan_cell(row + x);
            }
        }
        if y1 < res as isize {
            let row = y1 as usize * res;
            for x in xa..=xb {
                self.scan_cell(row + x);
            }
        }
        let ya = (y0 + 1).max(0) as usize;
        let yb = (y1 - 1).min(res as isize - 1) as usize;
        if x0 >= 0 {
            for y in ya..=yb {
                self.scan_cell(y * res + x0 as usize);
            }
        }
        if x1 < res as isize {
            for y in ya..=yb {
                self.scan_cell(y * res + x1 as usize);
            }
        }
    }

    /// Lower bound on the squared distance from the query to anything
    /// outside the explored cell box of Chebyshev radius `k` (see
    /// [`super::outside_box_dist_sq`], the shared box-certificate owner).
    fn unseen_bound_after(&self, k: usize) -> f32 {
        super::outside_box_dist_sq(self.grid, self.cx, self.cy, k, self.query.x, self.query.y)
    }

    /// Build the next non-empty ring batch, or mark exhaustion.
    fn build_pending(&mut self) {
        debug_assert!(!self.has_pending);
        while self.ring <= self.max_ring {
            self.scratch.pending.clear();
            let k = self.ring;
            self.for_each_ring_cell(k);
            self.ring += 1;
            self.pending_bound = self.unseen_bound_after(k);

            if !self.scratch.pending.is_empty() {
                // Nearest-first within the ring.
                self.scratch
                    .pending
                    .sort_unstable_by_key(|&(dist_sq, _)| dist_sq);
                self.has_pending = true;
                return;
            }
        }
        self.exhausted = true;
    }

    /// Current frontier: fills `out` with the pending ring's slots and
    /// `dists` with their sorted squared distances (the consumer's
    /// per-emission termination bounds — already computed here, so it
    /// never re-derives them from point loads).
    /// Returns `None` when the traversal is exhausted.
    pub(crate) fn frontier(
        &mut self,
        out: &mut Vec<u32>,
        dists: &mut Vec<f32>,
    ) -> Option<PlaneShellBatch> {
        if !self.has_pending && !self.exhausted {
            self.build_pending();
        }
        if self.exhausted {
            return None;
        }
        out.clear();
        dists.clear();
        out.extend(self.scratch.pending.iter().map(|&(_, slot)| slot));
        dists.extend(self.scratch.pending.iter().map(|&(d, _)| d.get()));
        Some(PlaneShellBatch {
            n: self.scratch.pending.len(),
            first_dist_sq: self.scratch.pending[0].0.get(),
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlaneNeighborBatchSource {
    PackedChunk0,
    PackedTail,
    ShellExpand,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlaneNeighborBatch {
    pub(crate) n: usize,
    /// Read by contract tests; production consumers use per-emission bounds.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) first_dist_sq: f32,
    pub(crate) unseen_bound: f32,
    /// Single-source today; the packed stage adds variants.
    #[allow(dead_code)]
    pub(crate) source: PlaneNeighborBatchSource,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum PlaneNeighborFrontier {
    ExactBatch(PlaneNeighborBatch),
    /// No batch at this stage boundary, but everything unseen is at least
    /// this far away; the consumer may terminate against the bound or
    /// `advance_frontier` to the next stage.
    UnknownButBounded {
        dist_lower_bound: f32,
    },
    Exhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamStage {
    Packed,
    Takeover,
    Done,
}

/// Directed neighbor stream over the plane grid: the packed SIMD stages
/// (when a prepared group is supplied) followed by the shell-expansion
/// takeover, which re-covers everything the packed stages may have skipped
/// (the consumer dedups).
pub(crate) struct PlaneNeighborStream<'a, 'b, 'p, 'g> {
    grid: &'a PlaneGrid,
    takeover: PlaneShellFrontier<'a, 'b>,
    packed: Option<super::packed::PlanePackedQuery<'a, 'p, 'g>>,
    stage: StreamStage,
    packed_safe_exhausted: bool,
    knn_exhausted: bool,
}

impl<'a, 'b, 'p, 'g> PlaneNeighborStream<'a, 'b, 'p, 'g> {
    pub(crate) fn new(
        grid: &'a PlaneGrid,
        points: &[Vec2],
        query_idx: usize,
        scratch: &'a mut PlaneGridScratch,
        eligibility: DirectedEligibility<'b>,
        packed: Option<super::packed::PlanePackedQuery<'a, 'p, 'g>>,
    ) -> Self {
        let stage = if packed.is_some() {
            StreamStage::Packed
        } else {
            StreamStage::Takeover
        };
        Self {
            grid,
            takeover: PlaneShellFrontier::new(
                grid,
                points[query_idx],
                query_idx,
                scratch,
                eligibility,
            ),
            packed,
            stage,
            packed_safe_exhausted: false,
            knn_exhausted: false,
        }
    }

    /// Current frontier; repeated calls return the same batch until
    /// [`Self::advance_frontier`]. `dists` receives the batch's sorted
    /// squared distances, parallel to `out`.
    pub(crate) fn frontier(
        &mut self,
        out: &mut Vec<u32>,
        dists: &mut Vec<f32>,
    ) -> PlaneNeighborFrontier {
        out.clear();
        dists.clear();
        loop {
            match self.stage {
                StreamStage::Packed => {
                    let packed = self
                        .packed
                        .as_mut()
                        .expect("packed stage requires packed query state");
                    match packed.frontier(out, dists) {
                        super::packed::PlanePackedFrontier::ExactBatch(batch) => {
                            let source = match batch.source {
                                super::packed::PlanePackedBatchSource::Chunk0 => {
                                    PlaneNeighborBatchSource::PackedChunk0
                                }
                                super::packed::PlanePackedBatchSource::Tail => {
                                    PlaneNeighborBatchSource::PackedTail
                                }
                            };
                            return PlaneNeighborFrontier::ExactBatch(PlaneNeighborBatch {
                                n: batch.n,
                                first_dist_sq: batch.first_dist_sq,
                                unseen_bound: batch.unseen_bound,
                                source,
                            });
                        }
                        super::packed::PlanePackedFrontier::UnknownButBounded {
                            dist_lower_bound,
                        } => return PlaneNeighborFrontier::UnknownButBounded { dist_lower_bound },
                        super::packed::PlanePackedFrontier::Exhausted => {
                            self.packed_safe_exhausted |= packed.safe_exhausted();
                            self.stage = StreamStage::Takeover;
                            continue;
                        }
                    }
                }
                StreamStage::Takeover => {
                    if let Some(batch) = self.takeover.frontier(out, dists) {
                        return PlaneNeighborFrontier::ExactBatch(PlaneNeighborBatch {
                            n: batch.n,
                            first_dist_sq: batch.first_dist_sq,
                            unseen_bound: batch.unseen_bound,
                            source: PlaneNeighborBatchSource::ShellExpand,
                        });
                    }
                    self.knn_exhausted = self.takeover.is_exhausted();
                    self.stage = StreamStage::Done;
                    return PlaneNeighborFrontier::Exhausted;
                }
                StreamStage::Done => return PlaneNeighborFrontier::Exhausted,
            }
        }
    }

    pub(crate) fn advance_frontier(&mut self) {
        match self.stage {
            StreamStage::Packed => {
                let grid = self.grid;
                let packed = self
                    .packed
                    .as_mut()
                    .expect("packed stage requires packed query state");
                packed.advance_frontier(grid);
                if packed.is_exhausted() {
                    self.packed_safe_exhausted |= packed.safe_exhausted();
                    self.stage = StreamStage::Takeover;
                }
            }
            StreamStage::Takeover => self.takeover.advance(),
            StreamStage::Done => {}
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    #[inline]
    pub(crate) fn knn_exhausted(&self) -> bool {
        self.knn_exhausted
    }

    // Diagnostics surface (sphere parity); the planar stats reporting
    // grows into it.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn packed_safe_exhausted(&self) -> bool {
        self.packed_safe_exhausted
    }
}
