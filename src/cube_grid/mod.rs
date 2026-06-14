//! Cube-map based spatial grid for fast spatial queries on unit sphere.
//!
//! Projects sphere onto 6 cube faces, divides each into a regular grid.
//! O(n) build, O(1) cell lookup.
//!
//! Supports two query types:
//! - `knn_into`: k-nearest neighbors
//! - `within_cos_into`: all points within angular distance (range query)
//!
//! Queries use best-first expansion over neighboring cells with conservative
//! distance bounds. Typical uniform inputs terminate after visiting a handful
//! of cells; worst-case falls back to brute force.

mod build;
mod dense;
pub mod packed_knn;
mod projection;
mod query;
mod weld;

pub(crate) use packed_knn::PackedQuery;
pub(crate) use projection::{cell_to_face_ij, face_uv_to_3d, st_to_uv};
use projection::{face_uv_to_cell, point_to_face_uv};
pub(crate) use query::{
    DirectedCellMode, DirectedEligibility, DirectedNeighborBatch, DirectedNeighborBatchSource,
    DirectedNeighborFrontier, DirectedNeighborStream,
};

use glam::Vec3;
#[cfg(feature = "timing")]
use std::time::Duration;

/// Fine-grained timings for `CubeMapGrid::new`.
#[cfg(feature = "timing")]
#[derive(Debug, Clone, Default)]
pub(crate) struct CubeMapGridBuildTimings {
    pub count_cells: Duration,
    pub prefix_sum: Duration,
    pub scatter_soa: Duration,
    pub neighbors: Duration,
    pub ring2: Duration,
    pub cell_bounds: Duration,
    pub security_3x3: Duration,
}

/// Dummy timings when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CubeMapGridBuildTimings;

#[cfg(feature = "timing")]
impl CubeMapGridBuildTimings {
    #[inline]
    pub fn total(&self) -> Duration {
        self.count_cells
            + self.prefix_sum
            + self.scatter_soa
            + self.neighbors
            + self.ring2
            + self.cell_bounds
            + self.security_3x3
    }
}

// Ring-2 size is 16 in a planar 3×3 neighborhood, but can be slightly larger on the stitched cube
// surface near cube vertices where a "3×3" neighborhood has only 7 unique neighbors (one diagonal
// is missing because only 3 faces meet at a cube vertex).
const RING2_MAX: usize = 16;

/// Cube-map spatial grid for points on unit sphere.
pub struct CubeMapGrid {
    pub(super) res: usize,
    /// Start index into point_indices for each cell, plus final length.
    /// Length: 6 * res² + 1
    pub(super) cell_offsets: Vec<u32>,
    /// Point indices grouped by cell.
    /// Length: n (number of points)
    pub(super) point_indices: Vec<u32>,
    /// Precomputed cell index per point (for fast query start).
    /// Length: n (number of points)
    point_cells: Vec<u32>,
    /// Precomputed 3×3 neighborhood for each cell.
    /// 9 entries per cell (self + 8 neighbors).
    /// Length: 6 * res²
    neighbors: Vec<[u32; 9]>,
    /// Precomputed ring-2 (Chebyshev distance 2) cells for each cell.
    /// Entries are padded with u32::MAX past ring2_lens.
    ring2: Vec<[u32; RING2_MAX]>,
    ring2_lens: Vec<u8>,
    /// Unit vector at the center of each cell (on the sphere).
    pub(super) cell_centers: Vec<Vec3>,
    /// Spherical cap radius around `cell_centers[cell]` that conservatively contains the cell.
    /// Stored as cos/sin for fast per-query bounds.
    pub(super) cell_cos_radius: Vec<f32>,
    pub(super) cell_sin_radius: Vec<f32>,

    /// Precomputed normalized u-grid-line plane normals, indexed by `face * (res+1) + line`.
    ///
    /// These are great-circle boundary planes `u = const` in the cube-map parameterization
    /// used by `point_to_face_uv`.
    pub(super) u_line_planes: Vec<Vec3>,
    /// Precomputed normalized v-grid-line plane normals, indexed by `face * (res+1) + line`.
    ///
    /// These are great-circle boundary planes `v = const` in the cube-map parameterization
    /// used by `point_to_face_uv`.
    pub(super) v_line_planes: Vec<Vec3>,

    // === SoA layout: points stored contiguous by cell ===
    /// X coordinates of points, ordered by cell (use cell_offsets for ranges).
    pub(super) cell_points_x: Vec<f32>,
    /// Y coordinates of points, ordered by cell.
    pub(super) cell_points_y: Vec<f32>,
    /// Z coordinates of points, ordered by cell.
    pub(super) cell_points_z: Vec<f32>,

    /// Inverse mapping from point index to SOA slot index.
    ///
    /// `point_slots[point_idx]` gives the slot in `point_indices` / `cell_points_*`.
    point_slots: Vec<u32>,

    /// Dense-cell sub-index ("punch 1", axis-sort): present only when some cell
    /// exceeds `DENSE_CELL_THRESHOLD` (rare). Side structure; does not affect
    /// the SoA. Not yet consulted by the query path (integration pending).
    #[allow(dead_code)]
    dense_index: Option<dense::DenseCellIndex>,
}

#[derive(Clone, Copy, Debug)]
enum EdgeDir {
    Left,
    Right,
    Down,
    Up,
}

#[inline]
fn cross_face_edge(
    face: usize,
    iu: usize,
    iv: usize,
    dir: EdgeDir,
    res: usize,
) -> (usize, usize, usize) {
    let last = res - 1;
    let iu_flip = last - iu;
    let iv_flip = last - iv;

    match (face, dir) {
        (0, EdgeDir::Left) => (4, last, iv),
        (0, EdgeDir::Right) => (5, 0, iv),
        (0, EdgeDir::Down) => (3, last, iu_flip),
        (0, EdgeDir::Up) => (2, last, iu),
        (1, EdgeDir::Left) => (5, last, iv),
        (1, EdgeDir::Right) => (4, 0, iv),
        (1, EdgeDir::Down) => (3, 0, iu),
        (1, EdgeDir::Up) => (2, 0, iu_flip),
        (2, EdgeDir::Left) => (1, iv_flip, last),
        (2, EdgeDir::Right) => (0, iv, last),
        (2, EdgeDir::Down) => (4, iu, last),
        (2, EdgeDir::Up) => (5, iu_flip, last),
        (3, EdgeDir::Left) => (1, iv, 0),
        (3, EdgeDir::Right) => (0, iv_flip, 0),
        (3, EdgeDir::Down) => (5, iu_flip, 0),
        (3, EdgeDir::Up) => (4, iu, 0),
        (4, EdgeDir::Left) => (1, last, iv),
        (4, EdgeDir::Right) => (0, 0, iv),
        (4, EdgeDir::Down) => (3, iu, last),
        (4, EdgeDir::Up) => (2, iu, 0),
        (5, EdgeDir::Left) => (0, last, iv),
        (5, EdgeDir::Right) => (1, 0, iv),
        (5, EdgeDir::Down) => (3, iu_flip, 0),
        (5, EdgeDir::Up) => (2, iu_flip, last),
        _ => unreachable!("invalid cube face"),
    }
}

#[inline]
fn step_one(face: usize, iu: usize, iv: usize, dir: EdgeDir, res: usize) -> (usize, usize, usize) {
    match dir {
        EdgeDir::Left => {
            if iu > 0 {
                (face, iu - 1, iv)
            } else {
                cross_face_edge(face, iu, iv, dir, res)
            }
        }
        EdgeDir::Right => {
            if iu + 1 < res {
                (face, iu + 1, iv)
            } else {
                cross_face_edge(face, iu, iv, dir, res)
            }
        }
        EdgeDir::Down => {
            if iv > 0 {
                (face, iu, iv - 1)
            } else {
                cross_face_edge(face, iu, iv, dir, res)
            }
        }
        EdgeDir::Up => {
            if iv + 1 < res {
                (face, iu, iv + 1)
            } else {
                cross_face_edge(face, iu, iv, dir, res)
            }
        }
    }
}

#[inline]
fn diagonal_from_edge_neighbors(center: u32, a: u32, b: u32, res: usize) -> u32 {
    #[inline]
    fn step_cell(cell: u32, dir: EdgeDir, res: usize) -> u32 {
        let (face, iu, iv) = cell_to_face_ij(cell as usize, res);
        let (nf, nu, nv) = step_one(face, iu, iv, dir, res);
        (nf * res * res + nv * res + nu) as u32
    }

    let a_edges = [
        step_cell(a, EdgeDir::Left, res),
        step_cell(a, EdgeDir::Right, res),
        step_cell(a, EdgeDir::Down, res),
        step_cell(a, EdgeDir::Up, res),
    ];
    let b_edges = [
        step_cell(b, EdgeDir::Left, res),
        step_cell(b, EdgeDir::Right, res),
        step_cell(b, EdgeDir::Down, res),
        step_cell(b, EdgeDir::Up, res),
    ];

    let mut found: Option<u32> = None;
    for cand in a_edges {
        if cand == center || cand == a || cand == b {
            continue;
        }
        if b_edges.contains(&cand) {
            if let Some(prev) = found {
                if prev != cand {
                    return u32::MAX;
                }
            } else {
                found = Some(cand);
            }
        }
    }

    found.unwrap_or(u32::MAX)
}

/// Reusable per-query scratch buffers.
///
/// For performance (especially parallel queries), prefer `CubeMapGrid::make_scratch()`.
pub struct CubeMapGridScratch {
    /// Cell visitation stamps (avoids clearing between queries)
    visited_stamp: Vec<u32>,
    stamp: u32,
    /// BFS layer whose points have not been emitted yet.
    current: Vec<u32>,
    /// Layer discovered while emitting `current`.
    next: Vec<u32>,
    /// Sorted (descending dot, slot) emission for the pending shell batch.
    pending: Vec<(crate::fp::OrdF32, u32)>,
    /// If true, the current query has exhausted all unseen candidates.
    exhausted: bool,
}

#[cfg(test)]
mod tests;
