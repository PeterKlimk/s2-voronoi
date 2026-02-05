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
pub mod packed_knn;
mod projection;
mod query;

pub(crate) use projection::{cell_to_face_ij, face_uv_to_3d, st_to_uv};
use projection::{face_uv_to_cell, point_to_face_uv};
pub(crate) use query::DirectedCtx;

use glam::Vec3;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
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

/// A f32 wrapper that implements Ord using total_cmp.
/// Unlike NotNan, this doesn't check for NaN - it just orders NaN consistently.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl OrdF32 {
    #[inline]
    fn new(v: f32) -> Self {
        OrdF32(v)
    }

    #[inline]
    fn get(self) -> f32 {
        self.0
    }
}

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

/// Status of a resumable k-NN query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnnStatus {
    /// More neighbors may be available; query can be resumed with a larger k.
    CanResume,
    /// Search exhausted; no more neighbors available beyond what was returned.
    Exhausted,
}

/// Reusable per-query scratch buffers.
///
/// Uses a fixed-size sorted buffer for candidates when possible:
/// - Avoids Vec growth and bounds checks
/// - Keeps k-th distance O(1)
/// - Resume remains cheap (slice the existing buffer)
///
/// For performance (especially parallel queries), prefer `CubeMapGrid::make_scratch()`.
pub struct CubeMapGridScratch {
    /// Cell visitation stamps (avoids clearing between queries)
    visited_stamp: Vec<u32>,
    stamp: u32,
    /// Priority queue for cell expansion (min-heap by distance bound)
    cell_heap: BinaryHeap<Reverse<(OrdF32, u32)>>,
    /// Track limit for resumable queries (number of neighbors preserved across resume).
    track_limit: usize,
    /// Candidate buffer (sorted ascending by distance).
    /// (dist_sq, point_idx)
    candidates_fixed: [(f32, u32); Self::MAX_TRACK],
    candidates_len: usize,
    use_fixed: bool,
    candidates_vec: Vec<(f32, u32)>,

    /// Dot-product top-k buffer for non-resumable queries (unsorted).
    /// Stored as (dot, point_idx), where larger dot = closer for unit vectors.
    candidates_dot: Vec<(f32, u32)>,
    // worst_dot: f32, // Unused
    // worst_dot_pos: usize, // Unused
    /// If true, we've done a brute-force scan and have an exhaustive candidate set
    /// (up to `track_limit`).
    exhausted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_one_has_unique_inverse_direction() {
        for res in [4usize, 5, 8, 16] {
            let num_cells = 6 * res * res;
            for cell in 0..num_cells {
                let (face, iu, iv) = cell_to_face_ij(cell, res);
                for dir in [EdgeDir::Left, EdgeDir::Right, EdgeDir::Down, EdgeDir::Up] {
                    let (f1, u1, v1) = step_one(face, iu, iv, dir, res);
                    let mut count = 0usize;
                    for back in [EdgeDir::Left, EdgeDir::Right, EdgeDir::Down, EdgeDir::Up] {
                        let (fb, ub, vb) = step_one(f1, u1, v1, back, res);
                        if (fb, ub, vb) == (face, iu, iv) {
                            count += 1;
                        }
                    }
                    assert_eq!(
                        count, 1,
                        "step_one inverse not unique: res={}, cell={}, dir={:?}, step=({},{},{})",
                        res, cell, dir, f1, u1, v1
                    );
                }
            }
        }
    }

    #[test]
    fn test_cell_neighbors_unique() {
        for res in [4usize, 5, 8, 16] {
            let grid = CubeMapGrid::new(&[], res);
            let num_cells = 6 * res * res;
            for cell in 0..num_cells {
                let (face, iu, iv) = cell_to_face_ij(cell, res);
                let neighbors = grid.cell_neighbors(cell);
                let mut seen = std::collections::HashSet::<u32>::with_capacity(9);
                for &ncell in neighbors.iter() {
                    if ncell == u32::MAX {
                        continue;
                    }
                    assert!(
                        (ncell as usize) < num_cells,
                        "invalid neighbor cell: res={}, cell={}, neighbor={}",
                        res,
                        cell,
                        ncell
                    );
                    assert!(
                        seen.insert(ncell),
                        "duplicate neighbor cell: res={}, cell={}, neighbor={}, neighbors={:?}",
                        res,
                        cell,
                        ncell,
                        neighbors
                    );
                }
                assert!(
                    seen.contains(&(cell as u32)),
                    "center missing: res={}, cell={}",
                    res,
                    cell
                );

                // Face-corner cells sit on a cube vertex at one corner, where only 3 cells meet;
                // the "outer" diagonal doesn't exist, so we expect 7 neighbors (8 including self).
                let last = res - 1;
                let is_face_corner = (iu == 0 || iu == last) && (iv == 0 || iv == last);
                let expected = if is_face_corner { 8 } else { 9 };
                assert_eq!(
                    seen.len(),
                    expected,
                    "unexpected neighborhood size: res={}, cell={}, face={}, iu={}, iv={}, got={}",
                    res,
                    cell,
                    face,
                    iu,
                    iv,
                    seen.len()
                );
            }
        }
    }

    #[test]
    fn test_ring2_unique_non_empty() {
        for res in [4usize, 5, 8, 16] {
            let grid = CubeMapGrid::new(&[], res);
            let num_cells = 6 * res * res;
            for cell in 0..num_cells {
                let ring2 = grid.cell_ring2(cell);
                assert!(
                    !ring2.is_empty(),
                    "ring2 is empty: res={}, cell={}",
                    res,
                    cell
                );
                assert!(
                    ring2.len() <= RING2_MAX,
                    "ring2 too large: res={}, cell={}, len={}",
                    res,
                    cell,
                    ring2.len()
                );
                let mut seen = std::collections::HashSet::<u32>::with_capacity(ring2.len());
                for &ncell in ring2 {
                    assert!(
                        (ncell as usize) < num_cells,
                        "invalid ring2 cell: res={}, cell={}, ring_cell={}",
                        res,
                        cell,
                        ncell
                    );
                    assert!(
                        seen.insert(ncell),
                        "duplicate ring2 cell: res={}, cell={}, ring_cell={}",
                        res,
                        cell,
                        ncell
                    );
                }
            }
        }
    }

    #[test]
    fn test_security_ring2_captures_outside_cap_max() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        #[inline]
        fn max_dot_to_cap(q: Vec3, center: Vec3, cos_r: f32, sin_r: f32) -> f32 {
            let cos_d = q.dot(center).clamp(-1.0, 1.0);
            if cos_d > cos_r {
                return 1.0;
            }
            let sin_d = (1.0 - cos_d * cos_d).max(0.0).sqrt();
            (cos_d * cos_r + sin_d * sin_r).clamp(-1.0, 1.0)
        }

        fn sample_point_in_cell(cell: usize, res: usize, rng: &mut impl Rng) -> Vec3 {
            let (face, iu, iv) = cell_to_face_ij(cell, res);

            // Sample in ST space, away from boundaries to avoid tie-breaking artifacts.
            let eps = 1e-4f32;
            let fu = (rng.gen::<f32>() * (1.0 - 2.0 * eps) + eps) / res as f32;
            let fv = (rng.gen::<f32>() * (1.0 - 2.0 * eps) + eps) / res as f32;
            let su = iu as f32 / res as f32 + fu;
            let sv = iv as f32 / res as f32 + fv;

            let u = st_to_uv(su);
            let v = st_to_uv(sv);
            face_uv_to_3d(face, u, v)
        }

        let res = 8usize;
        let grid = CubeMapGrid::new(&[], res);
        let num_cells = 6 * res * res;
        let mut rng = ChaCha8Rng::seed_from_u64(12345);

        let samples_per_cell = 8usize;

        for cell in 0..num_cells {
            let neighbors = grid.cell_neighbors(cell);
            let mut in_neighborhood = vec![false; num_cells];
            for &c in neighbors.iter() {
                if c == u32::MAX {
                    continue;
                }
                in_neighborhood[c as usize] = true;
            }

            let ring2 = grid.cell_ring2(cell);
            for &c in ring2 {
                assert!(
                    !in_neighborhood[c as usize],
                    "ring2 cell is inside neighborhood: cell={}, ring2_cell={}",
                    cell, c
                );
            }

            for _ in 0..samples_per_cell {
                let q = sample_point_in_cell(cell, res, &mut rng);
                assert_eq!(
                    grid.point_to_cell(q),
                    cell,
                    "sample not inside cell: cell={}",
                    cell
                );

                let mut outside_max = f32::NEG_INFINITY;
                for other in 0..num_cells {
                    if in_neighborhood[other] {
                        continue;
                    }
                    let dot = max_dot_to_cap(
                        q,
                        grid.cell_centers[other],
                        grid.cell_cos_radius[other],
                        grid.cell_sin_radius[other],
                    );
                    outside_max = outside_max.max(dot);
                }

                let mut ring2_max = f32::NEG_INFINITY;
                for &other in ring2 {
                    let idx = other as usize;
                    let dot = max_dot_to_cap(
                        q,
                        grid.cell_centers[idx],
                        grid.cell_cos_radius[idx],
                        grid.cell_sin_radius[idx],
                    );
                    ring2_max = ring2_max.max(dot);
                }

                // For a correct "3×3 neighborhood" security bound based on caps, the maximum cap
                // dot among all outside cells should always occur in ring2.
                let diff = outside_max - ring2_max;
                assert!(
                    diff <= 1e-5,
                    "ring2 missed outside max: cell={}, outside_max={}, ring2_max={}, diff={}",
                    cell,
                    outside_max,
                    ring2_max,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_uv_line_planes_match_uv_rect_interior() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        // Pick a resolution with enough margin to sample "outside" points while staying
        // strictly inside [-1, 1] UV bounds.
        let res = 16usize;
        let grid = CubeMapGrid::new(&[], res);
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        // Pick a few comfortably interior cells per face (3×3 envelope stays on-face, and
        // the 3×3 envelope doesn't touch the face boundary).
        let samples = [(3usize, 4usize), (6, 6), (10, 9)];

        for face in 0..6usize {
            for &(iu, iv) in &samples {
                let cell = face * res * res + iv * res + iu;
                let center = grid.cell_centers[cell];

                let mut planes = [
                    grid.face_u_line_plane(face, iu - 1),
                    grid.face_u_line_plane(face, iu + 2),
                    grid.face_v_line_plane(face, iv - 1),
                    grid.face_v_line_plane(face, iv + 2),
                ];
                for n in &mut planes {
                    if n.dot(center) < 0.0 {
                        *n = -*n;
                    }
                }

                let umin = st_to_uv((iu - 1) as f32 / res as f32);
                let umax = st_to_uv((iu + 2) as f32 / res as f32);
                let vmin = st_to_uv((iv - 1) as f32 / res as f32);
                let vmax = st_to_uv((iv + 2) as f32 / res as f32);

                let eps = 5e-4f32;

                // Inside points: must satisfy all plane halfspaces and map back to the same face.
                for _ in 0..256 {
                    let u = rng.gen_range((umin + eps)..(umax - eps));
                    let v = rng.gen_range((vmin + eps)..(vmax - eps));
                    let p = face_uv_to_3d(face, u, v);
                    let (f2, u2, v2) = point_to_face_uv(p);
                    assert_eq!(f2, face);
                    assert!(u2 >= umin - 1e-5 && u2 <= umax + 1e-5);
                    assert!(v2 >= vmin - 1e-5 && v2 <= vmax + 1e-5);

                    for n in &planes {
                        assert!(
                            n.dot(p) >= -1e-5,
                            "inside point violates plane: face={}, iu={}, iv={}, n·p={}",
                            face,
                            iu,
                            iv,
                            n.dot(p)
                        );
                    }
                }

                // Outside points on the same face: violate at least one boundary plane.
                let delta = 2e-3f32;
                for &(u, v) in &[
                    (umin - delta, (vmin + vmax) * 0.5),
                    (umax + delta, (vmin + vmax) * 0.5),
                    ((umin + umax) * 0.5, vmin - delta),
                    ((umin + umax) * 0.5, vmax + delta),
                ] {
                    assert!(u > -1.0 && u < 1.0 && v > -1.0 && v < 1.0);
                    let p = face_uv_to_3d(face, u, v);
                    let (f2, ..) = point_to_face_uv(p);
                    assert_eq!(f2, face);

                    let ok = planes.iter().all(|n| n.dot(p) >= -1e-6);
                    assert!(
                        !ok,
                        "outside point unexpectedly inside: face={}, iu={}, iv={}, u={}, v={}",
                        face, iu, iv, u, v
                    );
                }
            }
        }
    }
}
