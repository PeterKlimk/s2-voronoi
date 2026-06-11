//! Uniform 2D grid over the unit square for planar Voronoi neighbor queries.
//!
//! The planar analog of `cube_grid`: the same CSR layout (cell offsets plus
//! slot-ordered SoA point coordinates) and the same directed-eligibility
//! semantics (the shared [`DirectedEligibility`]), but one flat `res x res`
//! chart with no faces, no corners, and no seam wrap. Cells are exact
//! axis-aligned boxes, so min-distance certificates come from box geometry
//! instead of conservative spherical caps.
//!
//! Distance semantics are squared Euclidean (`dist_sq`, smaller is closer)
//! with *lower*-bound certificates on unseen points — the planar pipeline
//! never uses the sphere's dot-product ("bigger is closer") form.
//!
//! Input points are expected in the unit square `[0, 1]^2`; the pipeline
//! entry normalizes the user's bounding rect to it. Out-of-range coordinates
//! are clamped into the edge cells (and rejected upstream by validation).

mod query;

pub(crate) use query::{PlaneNeighborFrontier, PlaneNeighborStream};

#[cfg(test)]
mod tests;

use glam::Vec2;

/// Uniform spatial grid over the unit square.
pub(crate) struct PlaneGrid {
    res: usize,
    /// Start index into `point_indices` for each cell, plus final length.
    /// Length: res^2 + 1. Cells are row-major: `cell = iy * res + ix`.
    cell_offsets: Vec<u32>,
    /// Point indices grouped by cell. Length: n.
    point_indices: Vec<u32>,
    /// Precomputed cell index per point. Length: n.
    point_cells: Vec<u32>,
    /// Inverse mapping from point index to SoA slot. Length: n.
    point_slots: Vec<u32>,
    /// X coordinates of points, ordered by cell (use `cell_offsets` for ranges).
    cell_points_x: Vec<f32>,
    /// Y coordinates of points, ordered by cell.
    cell_points_y: Vec<f32>,
}

impl PlaneGrid {
    pub(crate) fn new(points: &[Vec2], res: usize) -> Self {
        assert!(res >= 1, "plane grid resolution must be at least 1");
        let n = points.len();
        let num_cells = res * res;

        let mut point_cells = vec![0u32; n];
        let mut counts = vec![0u32; num_cells];
        for (i, p) in points.iter().enumerate() {
            debug_assert!(
                (-1e-6..=1.0 + 1e-6).contains(&p.x) && (-1e-6..=1.0 + 1e-6).contains(&p.y),
                "plane grid input outside the unit square: {p:?}"
            );
            let cell = cell_of_point(*p, res) as u32;
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

        Self {
            res,
            cell_offsets,
            point_indices,
            point_cells,
            point_slots,
            cell_points_x,
            cell_points_y,
        }
    }

    /// Get cell index for a point.
    #[inline]
    pub(crate) fn point_to_cell(&self, p: Vec2) -> usize {
        cell_of_point(p, self.res)
    }

    /// Get the precomputed cell index for `points[idx]` used to build this grid.
    // Used by the packed-kNN stage when it lands (cell-grouped query runs).
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn point_index_to_cell(&self, idx: usize) -> usize {
        self.point_cells[idx] as usize
    }

    /// Get the SoA slot index for `points[idx]` used to build this grid.
    #[inline]
    pub(crate) fn point_index_to_slot(&self, idx: usize) -> u32 {
        self.point_slots[idx]
    }

    /// Grid resolution (cells per axis).
    #[inline]
    pub(crate) fn res(&self) -> usize {
        self.res
    }

    /// Cell offsets array (length = res^2 + 1).
    #[inline]
    pub(crate) fn cell_offsets(&self) -> &[u32] {
        &self.cell_offsets
    }

    /// Point indices array (SoA layout, length = n).
    #[inline]
    pub(crate) fn point_indices(&self) -> &[u32] {
        &self.point_indices
    }

    /// Grid-line coordinate of wall `i` (`i` in `0..=res`).
    ///
    /// Computed as `i / res` in f32 — the same rounding as a point constructed
    /// at the wall — rather than `i * (1/res)`, so wall comparisons in the
    /// certificates match point classification to within an ulp.
    #[inline]
    pub(crate) fn wall(&self, i: usize) -> f32 {
        i as f32 / self.res as f32
    }
}

/// Row-major cell index of a point, clamped into the grid.
#[inline]
fn cell_of_point(p: Vec2, res: usize) -> usize {
    let scale = res as f32;
    // `as usize` saturates at 0 for negative f32, so slight negatives clamp
    // to the edge cell without a branch.
    let ix = ((p.x * scale) as usize).min(res - 1);
    let iy = ((p.y * scale) as usize).min(res - 1);
    iy * res + ix
}
