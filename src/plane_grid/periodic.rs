//! Periodic (toroidal) grid and neighbor stream for planar Voronoi over a
//! rectangular torus.
//!
//! The wrap-aware sibling of the bounded `PlaneGrid`: an `res x res` grid of
//! (possibly anisotropic) cells tiling the domain `[0, px) x [0, py)`
//! exactly, with Chebyshev rings that wrap modulo `res` on both axes and
//! minimum-image squared-distance semantics throughout. Rings can collide
//! with themselves once `2k + 1 >= res`, so traversal uses visited stamps
//! (the cube grid's proven scratch pattern) rather than arithmetic
//! enumeration. Certificates are wrapped box distances with the same
//! wall-classification slack as the bounded grid.
//!
//! The neighbor stream runs the shared packed SIMD stage first (the
//! [`super::packed::PackedGeometry`] seam supplies wrapped boxes,
//! minimum-image distances, and wrapped-wall certificates), then the
//! shell-expansion takeover re-covers everything. The driver must not
//! re-clip a re-emitted neighbor (see the periodic driver), because the
//! unbounded seed makes re-clips non-idempotent.

use glam::Vec2;

use crate::cube_grid::{DirectedCellMode, DirectedEligibility};
use crate::fp::OrdF32;

/// Minimum-image displacement magnitude along one axis of period `p`.
#[inline(always)]
pub(crate) fn wrap_abs(d: f32, p: f32) -> f32 {
    let d = d.abs();
    d.min(p - d)
}

/// Minimum-image squared distance on the torus with periods `(px, py)`.
#[inline(always)]
pub(crate) fn min_image_dist_sq(a: Vec2, b: Vec2, px: f32, py: f32) -> f32 {
    let dx = wrap_abs(a.x - b.x, px);
    let dy = wrap_abs(a.y - b.y, py);
    dx * dx + dy * dy
}

/// Uniform spatial grid over the torus `[0, px) x [0, py)`.
pub(crate) struct PeriodicGrid {
    res: usize,
    /// Domain periods (normalized: the longer side is 1).
    px: f32,
    py: f32,
    cell_offsets: Vec<u32>,
    point_indices: Vec<u32>,
    point_cells: Vec<u32>,
    point_slots: Vec<u32>,
    cell_points_x: Vec<f32>,
    cell_points_y: Vec<f32>,
    /// Wall coordinates per axis: `i * p / res` for `i` in `0..=res`,
    /// computed with the same rounding as classification.
    walls_x: Vec<f32>,
    walls_y: Vec<f32>,
}

/// Reusable per-query scratch: pending emission + visited stamps.
pub(crate) struct PeriodicGridScratch {
    pending: Vec<(OrdF32, u32)>,
    visited_stamp: Vec<u32>,
    stamp: u32,
    /// Current and next ring cell lists for the stamped BFS.
    current: Vec<u32>,
    next: Vec<u32>,
}

impl PeriodicGrid {
    pub(crate) fn new(points: &[Vec2], res: usize, px: f32, py: f32) -> Self {
        assert!(res >= 1, "periodic grid resolution must be at least 1");
        assert!(px > 0.0 && py > 0.0);
        let n = points.len();
        let num_cells = res * res;

        let mut point_cells = vec![0u32; n];
        let mut counts = vec![0u32; num_cells];
        for (i, p) in points.iter().enumerate() {
            debug_assert!(
                (0.0..px).contains(&p.x) && (0.0..py).contains(&p.y),
                "periodic grid input outside the domain: {p:?}"
            );
            let cell = Self::cell_of(p.x, p.y, res, px, py) as u32;
            point_cells[i] = cell;
            counts[cell as usize] += 1;
        }

        let mut cell_offsets = vec![0u32; num_cells + 1];
        let mut acc = 0u32;
        for (cell, &count) in counts.iter().enumerate() {
            cell_offsets[cell] = acc;
            acc += count;
        }
        cell_offsets[num_cells] = acc;

        let mut cursor = cell_offsets[..num_cells].to_vec();
        let mut point_indices = vec![0u32; n];
        let mut point_slots = vec![0u32; n];
        let mut cell_points_x = vec![0.0f32; n];
        let mut cell_points_y = vec![0.0f32; n];
        for (i, p) in points.iter().enumerate() {
            let cell = point_cells[i] as usize;
            let slot = cursor[cell];
            cursor[cell] += 1;
            point_indices[slot as usize] = i as u32;
            point_slots[i] = slot;
            cell_points_x[slot as usize] = p.x;
            cell_points_y[slot as usize] = p.y;
        }

        // Same expression as classification (`i * p / res`), so certificate
        // wall comparisons match classification rounding.
        let walls_x = (0..=res).map(|i| i as f32 * px / res as f32).collect();
        let walls_y = (0..=res).map(|i| i as f32 * py / res as f32).collect();

        Self {
            res,
            px,
            py,
            cell_offsets,
            point_indices,
            point_cells,
            point_slots,
            cell_points_x,
            cell_points_y,
            walls_x,
            walls_y,
        }
    }

    #[inline]
    fn cell_of(x: f32, y: f32, res: usize, px: f32, py: f32) -> usize {
        let ix = ((x * res as f32 / px) as usize).min(res - 1);
        let iy = ((y * res as f32 / py) as usize).min(res - 1);
        iy * res + ix
    }

    pub(crate) fn make_scratch(&self) -> PeriodicGridScratch {
        PeriodicGridScratch {
            pending: Vec::new(),
            visited_stamp: vec![0; self.res * self.res],
            stamp: 0,
            current: Vec::new(),
            next: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn res(&self) -> usize {
        self.res
    }

    #[inline]
    pub(crate) fn periods(&self) -> (f32, f32) {
        (self.px, self.py)
    }

    #[inline]
    pub(crate) fn cell_offsets(&self) -> &[u32] {
        &self.cell_offsets
    }

    #[inline]
    pub(crate) fn point_indices(&self) -> &[u32] {
        &self.point_indices
    }

    #[inline]
    pub(crate) fn point_index_to_slot(&self, idx: usize) -> u32 {
        self.point_slots[idx]
    }

    #[inline]
    pub(crate) fn point_index_to_cell(&self, idx: usize) -> usize {
        self.point_cells[idx] as usize
    }

    /// Lower bound on the minimum-image squared distance from `(qx, qy)`
    /// to anything outside the wrapped cell box of Chebyshev radius `k`
    /// around `(cx, cy)`; INFINITY once the box covers the torus. The
    /// wrapped sibling of the bounded grid's `outside_box_dist_sq`.
    #[inline]
    pub(crate) fn outside_wrapped_box_dist_sq(
        &self,
        cx: usize,
        cy: usize,
        k: usize,
        qx: f32,
        qy: f32,
    ) -> f32 {
        let res = self.res;
        if 2 * k + 1 >= res {
            return f32::INFINITY;
        }
        let mut d = f32::INFINITY;
        let lo_x = (cx + res - k) % res; // left wall index of the box
        let hi_x = (cx + k + 1) % res; // right wall index
        let lo_y = (cy + res - k) % res;
        let hi_y = (cy + k + 1) % res;
        d = d.min(self.wall_dist(qx, 0, lo_x));
        d = d.min(self.wall_dist(qx, 0, hi_x));
        d = d.min(self.wall_dist(qy, 1, lo_y));
        d = d.min(self.wall_dist(qy, 1, hi_y));
        let d = d.max(0.0);
        d * d
    }

    /// Minimum-image distance from `q` to the nearest point of the wrapped
    /// wall `i` on the given axis (0 = x, 1 = y), shrunk by the
    /// classification slack.
    #[inline]
    fn wall_dist(&self, q: f32, axis: usize, i: usize) -> f32 {
        const SLACK: f32 = crate::tolerances::PLANE_WALL_CLASSIFICATION_SLACK;
        let (walls, period) = if axis == 0 {
            (&self.walls_x, self.px)
        } else {
            (&self.walls_y, self.py)
        };
        let w = walls[i];
        wrap_abs(q - w, period) - w.abs() * SLACK - period * SLACK
    }
}

impl PeriodicGrid {
    /// Collect all pairs of point indices within minimum-image `radius` of
    /// each other (each unordered pair reported at least once; duplicates
    /// and self-pairs are harmless to the union-find consumer).
    ///
    /// Same grid-integrated design as the bounded grid, with wrapped
    /// E/N/NE/NW neighbor bands; at tiny resolutions wrapped neighbors may
    /// coincide with the cell itself, which only produces redundant pairs.
    pub(crate) fn collect_pairs_within(&self, radius: f32, out: &mut Vec<(u32, u32)>) {
        let res = self.res;
        let r_sq = radius * radius;
        for cell in 0..res * res {
            let start = self.cell_offsets[cell] as usize;
            let end = self.cell_offsets[cell + 1] as usize;
            let k = end - start;
            if k == 0 {
                continue;
            }
            // Within-cell pairs (min-image distances; cells are far larger
            // than the radius at every production resolution).
            for i in start..end {
                let pi = Vec2::new(self.cell_points_x[i], self.cell_points_y[i]);
                for j in (i + 1)..end {
                    let pj = Vec2::new(self.cell_points_x[j], self.cell_points_y[j]);
                    if min_image_dist_sq(pi, pj, self.px, self.py) <= r_sq {
                        out.push((self.point_indices[i], self.point_indices[j]));
                    }
                }
            }
            // Wrapped forward neighbors for points within `radius` of the
            // E/N walls (W/S covered by those cells' own forward checks).
            let (ix, iy) = (cell % res, cell / res);
            let e_wall = self.walls_x[(ix + 1) % res];
            let n_wall = self.walls_y[(iy + 1) % res];
            let w_wall = self.walls_x[ix];
            let e_cell = iy * res + (ix + 1) % res;
            let n_cell = ((iy + 1) % res) * res + ix;
            let ne_cell = ((iy + 1) % res) * res + (ix + 1) % res;
            let nw_cell = ((iy + 1) % res) * res + (ix + res - 1) % res;
            for i in start..end {
                let p = Vec2::new(self.cell_points_x[i], self.cell_points_y[i]);
                let idx = self.point_indices[i];
                let close_e = wrap_abs(e_wall - p.x, self.px) <= radius;
                let close_n = wrap_abs(n_wall - p.y, self.py) <= radius;
                let close_w = wrap_abs(p.x - w_wall, self.px) <= radius;
                if close_e {
                    self.pairs_against_cell(e_cell, p, idx, r_sq, out);
                }
                if close_n {
                    self.pairs_against_cell(n_cell, p, idx, r_sq, out);
                }
                if close_e && close_n {
                    self.pairs_against_cell(ne_cell, p, idx, r_sq, out);
                }
                if close_w && close_n {
                    self.pairs_against_cell(nw_cell, p, idx, r_sq, out);
                }
            }
        }
    }

    fn pairs_against_cell(
        &self,
        cell: usize,
        p: Vec2,
        index: u32,
        r_sq: f32,
        out: &mut Vec<(u32, u32)>,
    ) {
        let start = self.cell_offsets[cell] as usize;
        let end = self.cell_offsets[cell + 1] as usize;
        for slot in start..end {
            let q = Vec2::new(self.cell_points_x[slot], self.cell_points_y[slot]);
            let qi = self.point_indices[slot];
            if qi == index {
                continue;
            }
            if min_image_dist_sq(p, q, self.px, self.py) <= r_sq {
                out.push((index, qi));
            }
        }
    }
}

impl super::packed::PackedGeometry for PeriodicGrid {
    #[inline(always)]
    fn res(&self) -> usize {
        PeriodicGrid::res(self)
    }

    #[inline(always)]
    fn cell_offsets(&self) -> &[u32] {
        PeriodicGrid::cell_offsets(self)
    }

    #[inline(always)]
    fn points_x(&self) -> &[f32] {
        &self.cell_points_x
    }

    #[inline(always)]
    fn points_y(&self) -> &[f32] {
        &self.cell_points_y
    }

    /// Wrapped box: every cell of the (2r+1)^2 neighborhood modulo the
    /// resolution. Distinctness (no cell visited twice) is guaranteed by
    /// [`Self::box_radius_distinct`], which gates the packed stage.
    #[inline(always)]
    fn for_each_box_cell(&self, cx: usize, cy: usize, radius: usize, mut f: impl FnMut(usize)) {
        let res = self.res as isize;
        let r = radius as isize;
        let (cx, cy) = (cx as isize, cy as isize);
        for y in (cy - r)..=(cy + r) {
            let yw = y.rem_euclid(res) as usize;
            for x in (cx - r)..=(cx + r) {
                let xw = x.rem_euclid(res) as usize;
                f(yw * res as usize + xw);
            }
        }
    }

    #[inline(always)]
    fn box_radius_distinct(&self, radius: usize) -> bool {
        // The (2r+1)-wide wrapped box enumerates distinct cells iff it spans
        // at most the full resolution.
        2 * radius < self.res
    }

    #[inline(always)]
    fn chunk_dist_sqs(
        &self,
        chunk: &crate::fp::PlaneChunk8,
        qx: f32,
        qy: f32,
    ) -> crate::fp::Dists8 {
        chunk.dist_sqs_wrapped(qx, qy, self.px, self.py)
    }

    #[inline(always)]
    fn dist_sq(&self, x: f32, y: f32, qx: f32, qy: f32) -> f32 {
        min_image_dist_sq(Vec2::new(x, y), Vec2::new(qx, qy), self.px, self.py)
    }

    #[inline(always)]
    fn outside_box_dist_sq(&self, cx: usize, cy: usize, radius: usize, qx: f32, qy: f32) -> f32 {
        self.outside_wrapped_box_dist_sq(cx, cy, radius, qx, qy)
    }
}

pub(crate) struct PeriodicShellBatch {
    pub(crate) n: usize,
    pub(crate) unseen_bound: f32,
}

/// Shell-expansion frontier over the torus: stamped Chebyshev rings with
/// wrapped indices and minimum-image distances.
pub(crate) struct PeriodicShellFrontier<'a, 'b> {
    grid: &'a PeriodicGrid,
    scratch: &'a mut PeriodicGridScratch,
    query: Vec2,
    query_idx: usize,
    start_cell: u32,
    cx: usize,
    cy: usize,
    /// Next Chebyshev ring radius to expand.
    ring: usize,
    eligibility: DirectedEligibility<'b>,
    pending_bound: f32,
    has_pending: bool,
    exhausted: bool,
}

impl<'a, 'b> PeriodicShellFrontier<'a, 'b> {
    pub(crate) fn new(
        grid: &'a PeriodicGrid,
        query: Vec2,
        query_idx: usize,
        scratch: &'a mut PeriodicGridScratch,
        eligibility: DirectedEligibility<'b>,
    ) -> Self {
        let start_cell = if query_idx < grid.point_cells.len() {
            grid.point_cells[query_idx]
        } else {
            PeriodicGrid::cell_of(query.x, query.y, grid.res, grid.px, grid.py) as u32
        };
        let res = grid.res;
        let cx = start_cell as usize % res;
        let cy = start_cell as usize / res;

        scratch.stamp = scratch.stamp.wrapping_add(1).max(1);
        if scratch.stamp == u32::MAX {
            scratch.visited_stamp.fill(0);
            scratch.stamp = 1;
        }
        scratch.pending.clear();
        scratch.current.clear();
        scratch.next.clear();
        scratch.visited_stamp[start_cell as usize] = scratch.stamp;
        scratch.current.push(start_cell);

        Self {
            grid,
            scratch,
            query,
            query_idx,
            start_cell,
            cx,
            cy,
            ring: 0,
            eligibility,
            pending_bound: f32::INFINITY,
            has_pending: false,
            exhausted: false,
        }
    }

    fn scan_cell(&mut self, cell: usize) {
        let mode = self
            .eligibility
            .cell_mode(&self.grid.cell_offsets, self.start_cell, cell);
        if mode == DirectedCellMode::TransitOnly {
            return;
        }
        let start = self.grid.cell_offsets[cell] as usize;
        let end = self.grid.cell_offsets[cell + 1] as usize;
        let (px, py) = (self.grid.px, self.grid.py);
        for slot in start..end {
            if self.grid.point_indices[slot] as usize == self.query_idx {
                continue;
            }
            let slot_u32 = slot as u32;
            if mode == DirectedCellMode::EmitCenterDirected
                && !self.eligibility.allows_center_slot(slot_u32)
            {
                continue;
            }
            let p = Vec2::new(self.grid.cell_points_x[slot], self.grid.cell_points_y[slot]);
            let dist_sq = min_image_dist_sq(p, self.query, px, py);
            self.scratch.pending.push((OrdF32::new(dist_sq), slot_u32));
        }
    }

    /// Mark-and-collect the wrapped Chebyshev ring `k` cells into
    /// `scratch.next`; previously visited cells (ring self-collision once
    /// `2k + 1 >= res`) are skipped by the stamp.
    fn collect_ring(&mut self, k: usize) {
        let res = self.grid.res as isize;
        let (cx, cy) = (self.cx as isize, self.cy as isize);
        let k = k as isize;
        let stamp = self.scratch.stamp;
        let push = |scratch: &mut PeriodicGridScratch, x: isize, y: isize| {
            let xw = x.rem_euclid(res) as usize;
            let yw = y.rem_euclid(res) as usize;
            let cell = (yw * res as usize + xw) as u32;
            let v = &mut scratch.visited_stamp[cell as usize];
            if *v != stamp {
                *v = stamp;
                scratch.next.push(cell);
            }
        };
        for x in (cx - k)..=(cx + k) {
            push(self.scratch, x, cy - k);
            push(self.scratch, x, cy + k);
        }
        for y in (cy - k + 1)..(cy + k) {
            push(self.scratch, cx - k, y);
            push(self.scratch, cx + k, y);
        }
    }

    /// Lower bound on the minimum-image squared distance from the query to
    /// anything outside the wrapped box of Chebyshev radius `k`. INFINITY
    /// once the box covers the torus (ring collection is stamped, so once
    /// the box arithmetic covers both axes everything has been visited).
    fn unseen_bound_after(&self, k: usize) -> f32 {
        self.grid
            .outside_wrapped_box_dist_sq(self.cx, self.cy, k, self.query.x, self.query.y)
    }

    fn build_pending(&mut self) {
        debug_assert!(!self.has_pending);
        loop {
            if self.scratch.current.is_empty() {
                self.exhausted = true;
                return;
            }
            self.scratch.pending.clear();
            // Scan the current ring's cells.
            let cells = std::mem::take(&mut self.scratch.current);
            for &cell in &cells {
                self.scan_cell(cell as usize);
            }
            self.scratch.current = cells;
            self.scratch.current.clear();

            // Collect the next ring (stamped) and compute this ring's bound.
            let k = self.ring;
            self.ring += 1;
            self.scratch.next.clear();
            if 2 * k + 1 < 2 * self.grid.res {
                self.collect_ring(self.ring);
            }
            std::mem::swap(&mut self.scratch.current, &mut self.scratch.next);
            self.pending_bound = self.unseen_bound_after(k);

            if !self.scratch.pending.is_empty() {
                self.scratch
                    .pending
                    .sort_unstable_by_key(|&(dist_sq, _)| dist_sq);
                self.has_pending = true;
                return;
            }
        }
    }

    pub(crate) fn frontier(
        &mut self,
        out: &mut Vec<u32>,
        dists: &mut Vec<f32>,
    ) -> Option<PeriodicShellBatch> {
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
        Some(PeriodicShellBatch {
            n: self.scratch.pending.len(),
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

/// Directed neighbor stream over the periodic grid: the packed SIMD stages
/// (when a prepared group is supplied) followed by the shell-expansion
/// takeover, which re-covers everything the packed stages may have skipped
/// (the consumer dedups) — the wrapped twin of `PlaneNeighborStream`.
pub(crate) struct PeriodicNeighborStream<'a, 'b, 'p, 'g> {
    grid: &'a PeriodicGrid,
    takeover: PeriodicShellFrontier<'a, 'b>,
    packed: Option<super::packed::PlanePackedQuery<'a, 'p, 'g>>,
    in_packed: bool,
    knn_exhausted: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PeriodicNeighborBatch {
    pub(crate) n: usize,
    /// Lower bound on the minimum-image squared distance of every eligible
    /// point not yet emitted (`INFINITY` once nothing unseen remains).
    pub(crate) unseen_bound: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum PeriodicNeighborFrontier {
    ExactBatch(PeriodicNeighborBatch),
    /// No batch at this stage boundary, but everything unseen is at least
    /// this far away; the consumer may terminate against the bound or
    /// `advance_frontier` to the next stage.
    UnknownButBounded {
        dist_lower_bound: f32,
    },
    Exhausted,
}

impl<'a, 'b, 'p, 'g> PeriodicNeighborStream<'a, 'b, 'p, 'g> {
    pub(crate) fn new(
        grid: &'a PeriodicGrid,
        points: &[Vec2],
        query_idx: usize,
        scratch: &'a mut PeriodicGridScratch,
        eligibility: DirectedEligibility<'b>,
        packed: Option<super::packed::PlanePackedQuery<'a, 'p, 'g>>,
    ) -> Self {
        let in_packed = packed.is_some();
        Self {
            grid,
            takeover: PeriodicShellFrontier::new(
                grid,
                points[query_idx],
                query_idx,
                scratch,
                eligibility,
            ),
            packed,
            in_packed,
            knn_exhausted: false,
        }
    }

    pub(crate) fn frontier(
        &mut self,
        out: &mut Vec<u32>,
        dists: &mut Vec<f32>,
    ) -> PeriodicNeighborFrontier {
        out.clear();
        dists.clear();
        while self.in_packed {
            let packed = self
                .packed
                .as_mut()
                .expect("packed stage requires packed query state");
            match packed.frontier(out, dists) {
                super::packed::PlanePackedFrontier::ExactBatch(batch) => {
                    return PeriodicNeighborFrontier::ExactBatch(PeriodicNeighborBatch {
                        n: batch.n,
                        unseen_bound: batch.unseen_bound,
                    });
                }
                super::packed::PlanePackedFrontier::UnknownButBounded { dist_lower_bound } => {
                    return PeriodicNeighborFrontier::UnknownButBounded { dist_lower_bound };
                }
                super::packed::PlanePackedFrontier::Exhausted => {
                    self.in_packed = false;
                }
            }
        }
        if let Some(batch) = self.takeover.frontier(out, dists) {
            PeriodicNeighborFrontier::ExactBatch(PeriodicNeighborBatch {
                n: batch.n,
                unseen_bound: batch.unseen_bound,
            })
        } else {
            self.knn_exhausted = self.takeover.is_exhausted();
            PeriodicNeighborFrontier::Exhausted
        }
    }

    pub(crate) fn advance_frontier(&mut self) {
        if self.in_packed {
            let grid = self.grid;
            let packed = self
                .packed
                .as_mut()
                .expect("packed stage requires packed query state");
            packed.advance_frontier(grid);
            if packed.is_exhausted() {
                self.in_packed = false;
            }
            return;
        }
        self.takeover.advance();
    }

    // Diagnostics surface (sibling-stream parity).
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn knn_exhausted(&self) -> bool {
        self.knn_exhausted
    }
}

// Mirrors the other NN contract suites: the loop index addresses parallel
// per-cell arrays, and the frontier loop has a non-trivial Exhausted arm.
#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::while_let_loop)]
mod tests {
    use super::super::packed::{
        PlanePackedGroupInput, PlanePackedQuery, PlanePackedScratch, PlanePackedTimings,
        PlanePreparedGroupStatus,
    };
    use super::*;
    use crate::packed_layout::PackedSlotLayout;
    use crate::policy::PackedNeighborPolicy;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    const LOCAL_SHIFT: u32 = 24;
    const LOCAL_MASK: u32 = (1u32 << LOCAL_SHIFT) - 1;
    const DIST_SQ_TOL: f32 = 1e-6;

    struct Harness {
        points: Vec<Vec2>,
        grid: PeriodicGrid,
        slot_gen_map: Vec<u32>,
        cell_of_slot: Vec<usize>,
        bin_of_cell: Vec<u8>,
        px: f32,
        py: f32,
    }

    impl Harness {
        fn new(points: Vec<Vec2>, res: usize, num_bins: u8, px: f32, py: f32) -> Self {
            assert!(num_bins >= 1);
            let grid = PeriodicGrid::new(&points, res, px, py);
            let num_cells = grid.cell_offsets().len() - 1;
            let bin_of_cell: Vec<u8> = (0..num_cells)
                .map(|cell| ((cell * num_bins as usize) / num_cells) as u8)
                .collect();
            let mut cell_of_slot = vec![0usize; points.len()];
            let mut slot_gen_map = vec![0u32; points.len()];
            for cell in 0..num_cells {
                let start = grid.cell_offsets()[cell] as usize;
                let end = grid.cell_offsets()[cell + 1] as usize;
                for slot in start..end {
                    cell_of_slot[slot] = cell;
                    slot_gen_map[slot] = ((bin_of_cell[cell] as u32) << LOCAL_SHIFT) | slot as u32;
                }
            }
            Harness {
                points,
                grid,
                slot_gen_map,
                cell_of_slot,
                bin_of_cell,
                px,
                py,
            }
        }

        fn layout(&self) -> PackedSlotLayout<'_> {
            PackedSlotLayout::new(&self.slot_gen_map, LOCAL_SHIFT, LOCAL_MASK)
        }

        fn slot_dist_sq(&self, query_slot: u32, slot: u32) -> f32 {
            let qi = self.grid.point_indices()[query_slot as usize] as usize;
            let pi = self.grid.point_indices()[slot as usize] as usize;
            min_image_dist_sq(self.points[pi], self.points[qi], self.px, self.py)
        }

        fn brute_eligible(&self, query_slot: u32) -> Vec<u32> {
            let start_cell = self.cell_of_slot[query_slot as usize];
            let qbin = self.bin_of_cell[start_cell];
            (0..self.points.len() as u32)
                .filter(|&slot| {
                    if slot == query_slot {
                        return false;
                    }
                    let cell = self.cell_of_slot[slot as usize];
                    let bin = self.bin_of_cell[cell];
                    if bin != qbin {
                        return true;
                    }
                    match cell.cmp(&start_cell) {
                        std::cmp::Ordering::Less => false,
                        std::cmp::Ordering::Equal => slot >= query_slot,
                        std::cmp::Ordering::Greater => true,
                    }
                })
                .collect()
        }

        fn collect_and_check(
            &self,
            name: &str,
            query_slot: u32,
            mut stream: PeriodicNeighborStream<'_, '_, '_, '_>,
        ) {
            let eligible = self.brute_eligible(query_slot);
            let mut unseen: std::collections::HashSet<u32> = eligible.iter().copied().collect();
            let mut emitted: std::collections::HashSet<u32> = Default::default();
            let mut batch = Vec::new();
            let mut dists = Vec::new();

            let nearest_unseen = |unseen: &std::collections::HashSet<u32>| -> f32 {
                unseen
                    .iter()
                    .map(|&slot| self.slot_dist_sq(query_slot, slot))
                    .fold(f32::INFINITY, f32::min)
            };

            loop {
                match stream.frontier(&mut batch, &mut dists) {
                    PeriodicNeighborFrontier::ExactBatch(result) => {
                        assert!(
                            dists.windows(2).all(|w| w[0] <= w[1]),
                            "{name}: batch distances must be sorted ascending"
                        );
                        for &slot in &batch[..result.n] {
                            assert_ne!(slot, query_slot, "{name}: emitted the query itself");
                            // Packed -> takeover overlap may re-emit; the
                            // consumer dedups, so the harness tolerates it.
                            emitted.insert(slot);
                            unseen.remove(&slot);
                        }
                        assert!(
                            nearest_unseen(&unseen) >= result.unseen_bound - DIST_SQ_TOL,
                            "{name}: unseen_bound {} overstates an unseen point at {} \
                             (query slot {query_slot})",
                            result.unseen_bound,
                            nearest_unseen(&unseen)
                        );
                        stream.advance_frontier();
                    }
                    PeriodicNeighborFrontier::UnknownButBounded { dist_lower_bound } => {
                        assert!(
                            nearest_unseen(&unseen) >= dist_lower_bound - DIST_SQ_TOL,
                            "{name}: stage bound {} overstates an unseen point at {} \
                             (query slot {query_slot})",
                            dist_lower_bound,
                            nearest_unseen(&unseen)
                        );
                        stream.advance_frontier();
                    }
                    PeriodicNeighborFrontier::Exhausted => break,
                }
            }

            let mut expected = eligible.clone();
            expected.sort_unstable();
            let mut got: Vec<u32> = emitted.into_iter().collect();
            got.sort_unstable();
            assert_eq!(
                got, expected,
                "{name}: eligible set mismatch (slot {query_slot})"
            );
        }

        /// Run the contract for every (sampled) query through the
        /// takeover-only path and, per center cell, the packed path
        /// (mirrors the bounded harness).
        fn check_all(&self, name: &str) {
            let n = self.points.len();
            let stride = (n / 96).max(1);
            for slot in (0..n as u32).step_by(stride) {
                let query_idx = self.grid.point_indices()[slot as usize] as usize;
                let ctx = DirectedEligibility::from_layout(
                    self.bin_of_cell[self.cell_of_slot[slot as usize]],
                    slot,
                    self.layout(),
                );
                let mut scratch = self.grid.make_scratch();
                let stream = PeriodicNeighborStream::new(
                    &self.grid,
                    &self.points,
                    query_idx,
                    &mut scratch,
                    ctx,
                    None,
                );
                self.collect_and_check(&format!("{name}/takeover"), slot, stream);
            }

            let num_cells = self.grid.cell_offsets().len() - 1;
            for cell in 0..num_cells {
                let start = self.grid.cell_offsets()[cell] as usize;
                let end = self.grid.cell_offsets()[cell + 1] as usize;
                if start == end {
                    continue;
                }
                let queries: Vec<u32> = (start..end).map(|s| s as u32).collect();
                let group = PlanePackedGroupInput::new(
                    cell,
                    self.bin_of_cell[cell],
                    &queries,
                    start as u32,
                    self.layout(),
                );
                for &expand_r2 in &[false, true] {
                    let mut packed_scratch = PlanePackedScratch::new();
                    let mut timings = PlanePackedTimings;
                    let PlanePreparedGroupStatus::Ready(mut prepared) =
                        packed_scratch.prepare_group(&self.grid, group, &mut timings)
                    else {
                        // SlowPath groups (incl. tiny wrapped grids) are
                        // exercised by the takeover-only pass.
                        continue;
                    };
                    for (qi, &slot) in queries.iter().enumerate() {
                        let query_idx = self.grid.point_indices()[slot as usize] as usize;
                        let ctx = DirectedEligibility::from_layout(
                            self.bin_of_cell[cell],
                            slot,
                            self.layout(),
                        );
                        let mut scratch = self.grid.make_scratch();
                        let packed = PlanePackedQuery::new(
                            &mut prepared,
                            &mut timings,
                            qi,
                            PackedNeighborPolicy::for_point_count(self.points.len(), expand_r2),
                        );
                        let stream = PeriodicNeighborStream::new(
                            &self.grid,
                            &self.points,
                            query_idx,
                            &mut scratch,
                            ctx,
                            Some(packed),
                        );
                        self.collect_and_check(
                            &format!("{name}/packed_r2={expand_r2}"),
                            slot,
                            stream,
                        );
                    }
                }
            }
        }
    }

    fn rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    fn uniform(n: usize, seed: u64, px: f32, py: f32) -> Vec<Vec2> {
        let mut r = rng(seed);
        (0..n)
            .map(|_| Vec2::new(r.gen_range(0.0..px), r.gen_range(0.0..py)))
            .collect()
    }

    #[test]
    fn periodic_nn_uniform() {
        Harness::new(uniform(320, 5, 1.0, 1.0), 8, 1, 1.0, 1.0).check_all("uniform");
        Harness::new(uniform(320, 7, 1.0, 1.0), 8, 3, 1.0, 1.0).check_all("uniform_3bins");
    }

    #[test]
    fn periodic_nn_coarse_grids() {
        // res 1/2/3: every query's rings wrap immediately (max self-collision).
        Harness::new(uniform(40, 11, 1.0, 1.0), 1, 1, 1.0, 1.0).check_all("res1");
        Harness::new(uniform(120, 13, 1.0, 1.0), 2, 2, 1.0, 1.0).check_all("res2");
        Harness::new(uniform(150, 17, 1.0, 1.0), 3, 1, 1.0, 1.0).check_all("res3");
    }

    #[test]
    fn periodic_nn_anisotropic_domain() {
        // Rectangular torus: periods differ per axis.
        Harness::new(uniform(300, 19, 1.0, 0.35), 8, 2, 1.0, 0.35).check_all("aniso");
        Harness::new(uniform(200, 23, 0.5, 1.0), 6, 1, 0.5, 1.0).check_all("aniso_tall");
    }

    #[test]
    fn periodic_nn_seam_clusters() {
        // Clusters hugging the wrap seams and the (0,0) corner: nearest
        // neighbors live across the seam (the regime bounded grids never see).
        let mut r = rng(29);
        let mut points = uniform(200, 31, 1.0, 1.0);
        for &(cx, cy) in &[
            (0.001f32, 0.5f32),
            (0.999, 0.5),
            (0.5, 0.001),
            (0.001, 0.999),
        ] {
            for _ in 0..15 {
                points.push(Vec2::new(
                    (cx + r.gen_range(-0.004f32..0.004)).rem_euclid(1.0),
                    (cy + r.gen_range(-0.004f32..0.004)).rem_euclid(1.0),
                ));
            }
        }
        Harness::new(points, 10, 2, 1.0, 1.0).check_all("seam_clusters");
    }

    #[test]
    fn periodic_nn_tiny_and_duplicates() {
        Harness::new(uniform(2, 37, 1.0, 1.0), 4, 1, 1.0, 1.0).check_all("n2");
        Harness::new(uniform(3, 41, 1.0, 1.0), 4, 1, 1.0, 1.0).check_all("n3");
        let mut points = uniform(80, 43, 1.0, 1.0);
        for i in 0..5 {
            points.push(points[i * 11]);
        }
        Harness::new(points, 4, 1, 1.0, 1.0).check_all("duplicates");
    }

    #[test]
    fn periodic_nn_pipeline_repro_aniso_block_bins() {
        // Exact shape of the failing pipeline config: anisotropic torus
        // (py = 0.25), n = 800, res 7, 4x4 block bins of stride 2 —
        // every slot checked through both stream paths.
        // Reproduce the pipeline's exact normalized points: rect
        // (-3,2)-(5,4), seed 203, scale 1/8, wrapped into [0,p).
        let next_below = |p: f32| f32::from_bits(p.to_bits() - 1);
        let mut r = rng(203);
        let points: Vec<Vec2> = (0..800)
            .map(|_| {
                let x = r.gen_range(-3.0f32..5.0);
                let y = r.gen_range(2.0f32..4.0);
                let nx = (x - -3.0) * 0.125;
                let ny = (y - 2.0) * 0.125;
                Vec2::new(
                    nx.rem_euclid(1.0).min(next_below(1.0)),
                    ny.rem_euclid(0.25).min(next_below(0.25)),
                )
            })
            .collect();
        let res = 7usize;
        let grid = PeriodicGrid::new(&points, res, 1.0, 0.25);
        let num_cells = res * res;
        let bin_stride = 2usize;
        let bin_res = res.div_ceil(bin_stride);
        let bin_of_cell: Vec<u8> = (0..num_cells)
            .map(|cell| {
                let ix = cell % res;
                let iy = cell / res;
                let bu = (ix / bin_stride).min(bin_res - 1);
                let bv = (iy / bin_stride).min(bin_res - 1);
                (bv * bin_res + bu) as u8
            })
            .collect();
        let mut cell_of_slot = vec![0usize; points.len()];
        let mut slot_gen_map = vec![0u32; points.len()];
        for cell in 0..num_cells {
            let start = grid.cell_offsets()[cell] as usize;
            let end = grid.cell_offsets()[cell + 1] as usize;
            for slot in start..end {
                cell_of_slot[slot] = cell;
                slot_gen_map[slot] = ((bin_of_cell[cell] as u32) << LOCAL_SHIFT) | slot as u32;
            }
        }
        let h = Harness {
            points,
            grid,
            slot_gen_map,
            cell_of_slot,
            bin_of_cell,
            px: 1.0,
            py: 0.25,
        };
        // Check every slot (no stride): failures here are sparse.
        for slot in 0..h.points.len() as u32 {
            let query_idx = h.grid.point_indices()[slot as usize] as usize;
            let ctx = DirectedEligibility::from_layout(
                h.bin_of_cell[h.cell_of_slot[slot as usize]],
                slot,
                h.layout(),
            );
            let mut scratch = h.grid.make_scratch();
            let stream =
                PeriodicNeighborStream::new(&h.grid, &h.points, query_idx, &mut scratch, ctx, None);
            h.collect_and_check("repro/takeover", slot, stream);
        }
        let num_cells = h.grid.cell_offsets().len() - 1;
        for cell in 0..num_cells {
            let start = h.grid.cell_offsets()[cell] as usize;
            let end = h.grid.cell_offsets()[cell + 1] as usize;
            if start == end {
                continue;
            }
            let queries: Vec<u32> = (start..end).map(|s| s as u32).collect();
            let group = PlanePackedGroupInput::new(
                cell,
                h.bin_of_cell[cell],
                &queries,
                start as u32,
                h.layout(),
            );
            let mut packed_scratch = PlanePackedScratch::new();
            let mut timings = PlanePackedTimings;
            let PlanePreparedGroupStatus::Ready(mut prepared) =
                packed_scratch.prepare_group(&h.grid, group, &mut timings)
            else {
                continue;
            };
            for (qi, &slot) in queries.iter().enumerate() {
                let query_idx = h.grid.point_indices()[slot as usize] as usize;
                let ctx = DirectedEligibility::from_layout(h.bin_of_cell[cell], slot, h.layout());
                let mut scratch = h.grid.make_scratch();
                let packed = PlanePackedQuery::new(
                    &mut prepared,
                    &mut timings,
                    qi,
                    PackedNeighborPolicy::for_point_count(h.points.len(), false),
                );
                let stream = PeriodicNeighborStream::new(
                    &h.grid,
                    &h.points,
                    query_idx,
                    &mut scratch,
                    ctx,
                    Some(packed),
                );
                h.collect_and_check("repro/packed", slot, stream);
            }
        }
    }

    #[test]
    fn periodic_min_image_distance_basics() {
        // Wrap-around distances: points near opposite edges are close.
        let a = Vec2::new(0.02, 0.5);
        let b = Vec2::new(0.98, 0.5);
        let d = min_image_dist_sq(a, b, 1.0, 1.0);
        assert!((d - 0.04f32 * 0.04).abs() < 1e-8, "wrapped dx: {d}");
        // Anisotropic periods wrap per axis.
        let c = Vec2::new(0.02, 0.30);
        let e = Vec2::new(0.02, 0.04);
        let d2 = min_image_dist_sq(c, e, 1.0, 0.35);
        assert!((d2 - 0.09f32 * 0.09).abs() < 1e-8, "wrapped dy: {d2}");
    }
}
