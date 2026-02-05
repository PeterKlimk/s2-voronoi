//! Query helpers for CubeMapGrid.

mod directed;
mod scratch;
mod undirected;

pub(crate) use directed::DirectedCtx;

use crate::fp;
use glam::Vec3;

use super::{face_uv_to_cell, point_to_face_uv, CubeMapGrid, CubeMapGridScratch};

impl CubeMapGrid {
    /// Get cell index for a point.
    #[inline]
    pub fn point_to_cell(&self, p: Vec3) -> usize {
        let (face, u, v) = point_to_face_uv(p);
        face_uv_to_cell(face, u, v, self.res)
    }

    /// Get the precomputed cell index for `points[idx]` used to build this grid.
    #[inline]
    pub fn point_index_to_cell(&self, idx: usize) -> usize {
        self.point_cells[idx] as usize
    }

    /// Get the SOA slot index for `points[idx]` used to build this grid.
    #[inline]
    pub fn point_index_to_slot(&self, idx: usize) -> u32 {
        self.point_slots[idx]
    }

    /// Get grid resolution (cells per face).
    #[inline]
    pub fn res(&self) -> usize {
        self.res
    }

    /// Get cell offsets array (length = num_cells + 1).
    #[inline]
    pub fn cell_offsets(&self) -> &[u32] {
        &self.cell_offsets
    }

    /// Get point indices array (SOA layout, length = n).
    #[inline]
    pub fn point_indices(&self) -> &[u32] {
        &self.point_indices
    }

    /// Get points in a cell.
    #[inline]
    pub fn cell_points(&self, cell: usize) -> &[u32] {
        let start = self.cell_offsets[cell] as usize;
        let end = self.cell_offsets[cell + 1] as usize;
        &self.point_indices[start..end]
    }

    /// Get the 9 neighbor cells (including self) for a cell.
    #[inline]
    pub fn cell_neighbors(&self, cell: usize) -> &[u32; 9] {
        &self.neighbors[cell]
    }

    /// Get the ring-2 cells (Chebyshev distance 2) for a cell.
    #[inline]
    pub fn cell_ring2(&self, cell: usize) -> &[u32] {
        let len = self.ring2_lens[cell] as usize;
        &self.ring2[cell][..len]
    }

    /// Get the u-grid-line plane normal for a face at a given line index.
    ///
    /// Indexed by `face * (res + 1) + line`, where `line ∈ [0, res]`.
    #[inline]
    pub fn face_u_line_plane(&self, face: usize, line: usize) -> Vec3 {
        debug_assert!(face < 6, "invalid face {}", face);
        debug_assert!(
            line <= self.res,
            "invalid u line {} (res={})",
            line,
            self.res
        );
        self.u_line_planes[face * (self.res + 1) + line]
    }

    /// Get the v-grid-line plane normal for a face at a given line index.
    ///
    /// Indexed by `face * (res + 1) + line`, where `line ∈ [0, res]`.
    #[inline]
    pub fn face_v_line_plane(&self, face: usize, line: usize) -> Vec3 {
        debug_assert!(face < 6, "invalid face {}", face);
        debug_assert!(
            line <= self.res,
            "invalid v line {} (res={})",
            line,
            self.res
        );
        self.v_line_planes[face * (self.res + 1) + line]
    }

    /// Create a reusable scratch buffer for fast repeated queries.
    pub fn make_scratch(&self) -> CubeMapGridScratch {
        CubeMapGridScratch::new(6 * self.res * self.res)
    }

    /// Conservative lower bound on squared Euclidean distance from `query` to any point in `cell`.
    ///
    /// Uses a spherical cap that contains the cell and triangle inequality on the sphere.
    #[inline]
    fn cell_min_dist_sq(&self, query: Vec3, cell: usize) -> f32 {
        let center = self.cell_centers[cell];
        let mut cos_d = fp::dot3_f32(query.x, query.y, query.z, center.x, center.y, center.z);
        cos_d = cos_d.clamp(-1.0, 1.0);

        let cos_r = self.cell_cos_radius[cell];
        let sin_r = self.cell_sin_radius[cell];

        // If the query direction is within the cell's cap, the minimum distance can be 0.
        if cos_d > cos_r {
            return 0.0;
        }

        // cos(d - r) = cos d cos r + sin d sin r
        let sin_d = (1.0 - cos_d * cos_d).max(0.0).sqrt();
        let max_dot_upper = (cos_d * cos_r + sin_d * sin_r).clamp(-1.0, 1.0);
        2.0 - 2.0 * max_dot_upper
    }
}
