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

pub mod packed_knn;

use glam::{Vec3, Vec3A};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
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

trait UnitVec: Copy {
    fn dot(self, other: Self) -> f32;
    fn to_vec3(self) -> Vec3;
}

impl UnitVec for Vec3 {
    #[inline(always)]
    fn dot(self, other: Self) -> f32 {
        Vec3::dot(self, other)
    }

    #[inline(always)]
    fn to_vec3(self) -> Vec3 {
        self
    }
}

impl UnitVec for Vec3A {
    #[inline(always)]
    fn dot(self, other: Self) -> f32 {
        Vec3A::dot(self, other)
    }

    #[inline(always)]
    fn to_vec3(self) -> Vec3 {
        Vec3::from(self)
    }
}

#[inline(always)]
fn unit_vec_dist_sq_generic<P: UnitVec>(a: P, b: P) -> f32 {
    (2.0 - 2.0 * a.dot(b)).max(0.0)
}

// Ring-2 size is 16 in a planar 3×3 neighborhood, but can be slightly larger on the stitched cube
// surface near cube vertices where a "3×3" neighborhood has only 7 unique neighbors (one diagonal
// is missing because only 3 faces meet at a cube vertex).
const RING2_MAX: usize = 16;

// S2-style quadratic projection to reduce cube map distortion.
// Maps UV ∈ [-1, 1] to ST ∈ [0, 1] with area-equalizing transform.
// Corners get compressed (larger solid angle → fewer cells),
// centers get expanded (smaller solid angle → more cells).

/// S2 quadratic transform: UV [-1, 1] → ST [0, 1]
#[inline]
pub(crate) fn uv_to_st(u: f32) -> f32 {
    if u >= 0.0 {
        0.5 * (1.0 + 3.0 * u).sqrt()
    } else {
        1.0 - 0.5 * (1.0 - 3.0 * u).sqrt()
    }
}

/// S2 inverse transform: ST [0, 1] → UV [-1, 1]
#[inline]
pub(crate) fn st_to_uv(s: f32) -> f32 {
    if s >= 0.5 {
        (1.0 / 3.0) * (4.0 * s * s - 1.0)
    } else {
        (1.0 / 3.0) * (1.0 - 4.0 * (1.0 - s) * (1.0 - s))
    }
}

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
    /// Length: 6 * res² * 9
    neighbors: Vec<u32>,
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
    /// Precomputed security threshold for 3x3 neighborhood: max dot from any point in the cell
    /// to any ring-2 cell (Chebyshev distance 2). This is computed using ring caps,
    /// which overestimates by ~6.7% but can be precomputed once per cell.
    pub(super) security_3x3: Vec<f32>,

    // === SoA layout: points stored contiguous by cell ===
    /// X coordinates of points, ordered by cell (use cell_offsets for ranges).
    pub(super) cell_points_x: Vec<f32>,
    /// Y coordinates of points, ordered by cell.
    pub(super) cell_points_y: Vec<f32>,
    /// Z coordinates of points, ordered by cell.
    pub(super) cell_points_z: Vec<f32>,
}

/// Map a point on unit sphere to (face, u, v) where u,v ∈ [-1, 1].
#[inline]
fn point_to_face_uv(p: Vec3) -> (usize, f32, f32) {
    let (x, y, z) = (p.x, p.y, p.z);
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());

    if ax >= ay && ax >= az {
        // ±X
        if x >= 0.0 {
            (0, -z / ax, y / ax)
        } else {
            (1, z / ax, y / ax)
        }
    } else if ay >= ax && ay >= az {
        // ±Y
        if y >= 0.0 {
            (2, x / ay, -z / ay)
        } else {
            (3, x / ay, z / ay)
        }
    } else {
        // ±Z
        if z >= 0.0 {
            (4, x / az, y / az)
        } else {
            (5, -x / az, y / az)
        }
    }
}

/// Convert (face, u, v) to cell index.
#[inline]
fn face_uv_to_cell(face: usize, u: f32, v: f32, res: usize) -> usize {
    // Map UV [-1, 1] -> ST [0, 1] using the S2 quadratic transform.
    let su = uv_to_st(u);
    let sv = uv_to_st(v);
    let fu = (su * res as f32).max(0.0);
    let fv = (sv * res as f32).max(0.0);
    let iu = (fu as usize).min(res - 1);
    let iv = (fv as usize).min(res - 1);
    face * res * res + iv * res + iu
}

/// Convert (face, u, v) back to a 3D point (inverse of point_to_face_uv).
#[inline]
pub(crate) fn face_uv_to_3d(face: usize, u: f32, v: f32) -> Vec3 {
    // Project onto cube face, then normalize to sphere
    let p = match face {
        0 => Vec3::new(1.0, v, -u),  // +X: u = -z/x, v = y/x
        1 => Vec3::new(-1.0, v, u),  // -X: u = z/|x|, v = y/|x|
        2 => Vec3::new(u, 1.0, -v),  // +Y: u = x/y, v = -z/y
        3 => Vec3::new(u, -1.0, v),  // -Y: u = x/|y|, v = z/|y|
        4 => Vec3::new(u, v, 1.0),   // +Z: u = x/z, v = y/z
        5 => Vec3::new(-u, v, -1.0), // -Z: u = -x/|z|, v = y/|z|
        _ => unreachable!(),
    };
    p.normalize()
}

/// Convert cell index to (face, iu, iv).
#[inline]
pub(crate) fn cell_to_face_ij(cell: usize, res: usize) -> (usize, usize, usize) {
    let face = cell / (res * res);
    let rem = cell % (res * res);
    let iv = rem / res;
    let iu = rem % res;
    (face, iu, iv)
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
        if b_edges.iter().any(|&x| x == cand) {
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
    worst_dot: f32,
    worst_dot_pos: usize,

    /// If true, we've done a brute-force scan and have an exhaustive candidate set
    /// (up to `track_limit`).
    exhausted: bool,
}

impl CubeMapGridScratch {
    const MAX_TRACK: usize = 128;

    pub fn new(num_cells: usize) -> Self {
        Self {
            visited_stamp: vec![0; num_cells],
            stamp: 0,
            cell_heap: BinaryHeap::new(),
            track_limit: 0,
            candidates_fixed: [(f32::INFINITY, 0); Self::MAX_TRACK],
            candidates_len: 0,
            use_fixed: true,
            candidates_vec: Vec::new(),
            candidates_dot: Vec::new(),
            worst_dot: f32::NEG_INFINITY,
            worst_dot_pos: 0,
            exhausted: false,
        }
    }

    #[inline]
    fn begin_query(&mut self, k: usize, track_limit: usize) {
        self.cell_heap.clear();
        self.exhausted = false;
        self.track_limit = track_limit;
        self.candidates_len = 0;
        self.use_fixed = track_limit <= Self::MAX_TRACK;
        self.candidates_vec.clear();
        self.candidates_dot.clear();
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
    fn begin_query_dot(&mut self, k: usize) {
        self.cell_heap.clear();
        self.exhausted = false;
        self.track_limit = 0;
        self.candidates_len = 0;
        self.use_fixed = true;
        self.candidates_vec.clear();
        self.candidates_dot.clear();
        if self.candidates_dot.capacity() < k {
            self.candidates_dot
                .reserve(k - self.candidates_dot.capacity());
        }
        self.worst_dot = f32::NEG_INFINITY;
        self.worst_dot_pos = 0;

        // Stamp 0 means "unvisited". Avoid ever using stamp 0 for a query.
        self.stamp = self.stamp.wrapping_add(1).max(1);
        if self.stamp == u32::MAX {
            self.visited_stamp.fill(0);
            self.stamp = 1;
        }
    }

    #[inline]
    fn mark_visited(&mut self, cell: u32) -> bool {
        let idx = cell as usize;
        if self.visited_stamp[idx] == self.stamp {
            return false;
        }
        self.visited_stamp[idx] = self.stamp;
        true
    }

    #[inline]
    fn push_cell(&mut self, cell: u32, bound_dist_sq: f32) {
        self.cell_heap
            .push(Reverse((OrdF32::new(bound_dist_sq), cell)));
    }

    #[inline]
    fn peek_cell(&self) -> Option<(f32, u32)> {
        self.cell_heap
            .peek()
            .map(|Reverse((bound, cell))| (bound.get(), *cell))
    }

    #[inline]
    fn pop_cell(&mut self) -> Option<(f32, u32)> {
        self.cell_heap
            .pop()
            .map(|Reverse((bound, cell))| (bound.get(), cell))
    }

    /// Get the distance to the k-th candidate (for pruning).
    /// Returns infinity if we have fewer than k candidates.
    #[inline]
    fn kth_dist_sq(&self, k: usize) -> f32 {
        if k == 0 {
            return f32::INFINITY;
        }
        if self.use_fixed {
            if k > self.candidates_len {
                f32::INFINITY
            } else {
                self.candidates_fixed[k - 1].0
            }
        } else if k > self.candidates_vec.len() {
            f32::INFINITY
        } else {
            self.candidates_vec[k - 1].0
        }
    }

    #[inline]
    fn have_k(&self, k: usize) -> bool {
        if self.use_fixed {
            self.candidates_len >= k
        } else {
            self.candidates_vec.len() >= k
        }
    }

    #[inline]
    fn have_k_dot(&self, k: usize) -> bool {
        self.candidates_dot.len() >= k
    }

    /// Get distance to k-th candidate for dot-topk mode.
    #[inline]
    fn kth_dist_sq_dot(&self, k: usize) -> f32 {
        if k == 0 || self.candidates_dot.len() < k {
            return f32::INFINITY;
        }
        // dist_sq = 2 - 2*dot (unit vectors)
        let dot = self.worst_dot.clamp(-1.0, 1.0);
        (2.0 - 2.0 * dot).max(0.0)
    }

    /// Try to add a neighbor, tracking up to `track_limit` candidates.
    #[inline]
    fn try_add_neighbor(&mut self, idx: usize, dist_sq: f32) {
        if dist_sq.is_nan() || self.track_limit == 0 {
            return;
        }
        let limit = self.track_limit;
        let idx_u32 = idx as u32;

        if self.use_fixed {
            let len = self.candidates_len;
            if len >= limit && dist_sq >= self.candidates_fixed[limit - 1].0 {
                return;
            }
            if len < limit {
                let insert_pos =
                    self.candidates_fixed[..len].partition_point(|&(d, _)| d < dist_sq);
                if insert_pos < len {
                    self.candidates_fixed
                        .copy_within(insert_pos..len, insert_pos + 1);
                }
                self.candidates_fixed[insert_pos] = (dist_sq, idx_u32);
                self.candidates_len = len + 1;
            } else {
                let insert_pos =
                    self.candidates_fixed[..limit].partition_point(|&(d, _)| d < dist_sq);
                if insert_pos >= limit {
                    return;
                }
                self.candidates_fixed
                    .copy_within(insert_pos..(limit - 1), insert_pos + 1);
                self.candidates_fixed[insert_pos] = (dist_sq, idx_u32);
            }
        } else {
            let len = self.candidates_vec.len();
            if len >= limit && dist_sq >= self.candidates_vec[limit - 1].0 {
                return;
            }
            if len < limit {
                let insert_pos = self.candidates_vec[..len].partition_point(|&(d, _)| d < dist_sq);
                self.candidates_vec.push((dist_sq, idx_u32));
                if insert_pos < len {
                    self.candidates_vec
                        .copy_within(insert_pos..len, insert_pos + 1);
                    self.candidates_vec[insert_pos] = (dist_sq, idx_u32);
                }
            } else {
                let insert_pos =
                    self.candidates_vec[..limit].partition_point(|&(d, _)| d < dist_sq);
                if insert_pos >= limit {
                    return;
                }
                self.candidates_vec
                    .copy_within(insert_pos..(limit - 1), insert_pos + 1);
                self.candidates_vec[insert_pos] = (dist_sq, idx_u32);
            }
        }
    }

    /// Copy the first k candidate indices into output vec (sorted by distance).
    fn copy_k_indices_into(&self, k: usize, out: &mut Vec<usize>) {
        out.clear();
        if k == 0 {
            return;
        }
        let count = if self.use_fixed {
            k.min(self.candidates_len)
        } else {
            k.min(self.candidates_vec.len())
        };
        out.reserve(count);
        for i in 0..count {
            let idx = if self.use_fixed {
                self.candidates_fixed[i].1
            } else {
                self.candidates_vec[i].1
            };
            out.push(idx as usize);
        }
    }

    /// Append indices for `prev_k..new_k` into `out`, preserving the existing prefix.
    ///
    /// This is useful for resumable queries where the previous results are known to be a prefix
    /// of the new results (i.e. correctness was certified for the earlier k).
    fn append_k_indices_into(&self, prev_k: usize, new_k: usize, out: &mut Vec<usize>) {
        if new_k == 0 {
            out.clear();
            return;
        }

        let prev_k = prev_k.min(new_k);
        if out.len() > prev_k {
            out.truncate(prev_k);
        }

        if new_k <= prev_k {
            return;
        }

        let available = if self.use_fixed {
            self.candidates_len
        } else {
            self.candidates_vec.len()
        };
        let count = new_k.min(available);
        if count <= prev_k {
            return;
        }

        out.reserve(count - prev_k);
        for i in prev_k..count {
            let idx = if self.use_fixed {
                self.candidates_fixed[i].1
            } else {
                self.candidates_vec[i].1
            };
            out.push(idx as usize);
        }
    }

    fn copy_k_indices_dot_into(&mut self, k: usize, out: &mut Vec<usize>) {
        out.clear();
        if k == 0 {
            return;
        }
        let count = k.min(self.candidates_dot.len());
        if count == 0 {
            return;
        }

        // Sort by dot descending (closest first), tie-break by index for determinism.
        self.candidates_dot
            .sort_unstable_by(|(da, ia), (db, ib)| db.total_cmp(da).then_with(|| ia.cmp(ib)));

        out.reserve(count);
        for i in 0..count {
            out.push(self.candidates_dot[i].1 as usize);
        }
    }

    #[inline]
    fn try_add_neighbor_dot(&mut self, idx: usize, dot: f32, k: usize) {
        if dot.is_nan() || k == 0 {
            return;
        }
        let dot = dot.clamp(-1.0, 1.0);
        let idx_u32 = idx as u32;

        let len = self.candidates_dot.len();
        if len < k {
            self.candidates_dot.push((dot, idx_u32));
            if dot < self.worst_dot || len == 0 {
                self.worst_dot = dot;
                self.worst_dot_pos = len;
            } else if len + 1 == k {
                // Just reached k: compute worst in one pass.
                let mut worst_dot = self.candidates_dot[0].0;
                let mut worst_pos = 0usize;
                for (i, &(d, _)) in self.candidates_dot.iter().enumerate().skip(1) {
                    if d < worst_dot {
                        worst_dot = d;
                        worst_pos = i;
                    }
                }
                self.worst_dot = worst_dot;
                self.worst_dot_pos = worst_pos;
            }
            return;
        }

        if dot <= self.worst_dot {
            return;
        }

        // Replace current worst, then rescan to find new worst (k is small: 12/24/48).
        self.candidates_dot[self.worst_dot_pos] = (dot, idx_u32);
        let mut worst_dot = self.candidates_dot[0].0;
        let mut worst_pos = 0usize;
        for (i, &(d, _)) in self.candidates_dot.iter().enumerate().skip(1) {
            if d < worst_dot {
                worst_dot = d;
                worst_pos = i;
            }
        }
        self.worst_dot = worst_dot;
        self.worst_dot_pos = worst_pos;
    }
}

impl CubeMapGrid {
    /// Build a cube-map grid from points on unit sphere.
    ///
    /// `res` controls grid resolution: 6 * res² total cells.
    /// For n points, good choices:
    /// - res ≈ sqrt(n / 300) for ~50 points per cell
    /// - res ≈ sqrt(n / 600) for ~100 points per cell
    pub fn new(points: &[Vec3], res: usize) -> Self {
        #[cfg(feature = "timing")]
        let mut timings = CubeMapGridBuildTimings::default();
        #[cfg(feature = "timing")]
        return Self::new_impl(points, res, Some(&mut timings));
        #[cfg(not(feature = "timing"))]
        return Self::new_impl(points, res, None);
    }

    #[cfg(feature = "timing")]
    pub fn new_with_build_timings(
        points: &[Vec3],
        res: usize,
        timings: &mut CubeMapGridBuildTimings,
    ) -> Self {
        Self::new_impl(points, res, Some(timings))
    }

    fn new_impl(
        points: &[Vec3],
        res: usize,
        #[cfg(feature = "timing")] mut timings: Option<&mut CubeMapGridBuildTimings>,
        #[cfg(not(feature = "timing"))] _timings: Option<&mut CubeMapGridBuildTimings>,
    ) -> Self {
        assert!(res > 0, "CubeMapGrid requires res > 0");
        let num_cells = 6 * res * res;

        // Step 1: Count points per cell
        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let mut cell_counts = vec![0u32; num_cells];
        let mut point_cells = Vec::with_capacity(points.len());
        for p in points {
            let (face, u, v) = point_to_face_uv(*p);
            let cell = face_uv_to_cell(face, u, v, res);
            point_cells.push(cell as u32);
            cell_counts[cell] += 1;
        }
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.count_cells += t.elapsed();
        }

        // Step 2: Prefix sum to get offsets
        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let mut cell_offsets = Vec::with_capacity(num_cells + 1);
        cell_offsets.push(0);
        let mut sum = 0u32;
        for &count in &cell_counts {
            sum += count;
            cell_offsets.push(sum);
        }
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.prefix_sum += t.elapsed();
        }

        // Step 3: Scatter points into cells and build SoA layout (points stored contiguous by cell).
        //
        // This is a single pass that avoids an additional gather pass over a random permutation.
        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let n = points.len();
        debug_assert_eq!(cell_offsets[num_cells] as usize, n, "prefix sum mismatch");

        let mut point_indices = Vec::<u32>::with_capacity(n);
        let mut cell_points_x = Vec::<f32>::with_capacity(n);
        let mut cell_points_y = Vec::<f32>::with_capacity(n);
        let mut cell_points_z = Vec::<f32>::with_capacity(n);

        let indices_spare = point_indices.spare_capacity_mut();
        let x_spare = cell_points_x.spare_capacity_mut();
        let y_spare = cell_points_y.spare_capacity_mut();
        let z_spare = cell_points_z.spare_capacity_mut();

        let mut cell_cursors = cell_offsets[..num_cells].to_vec();
        for (i, cell_u32) in point_cells.iter().copied().enumerate() {
            let cell = cell_u32 as usize;
            let pos = cell_cursors[cell] as usize;
            let p = points[i];

            indices_spare[pos].write(i as u32);
            x_spare[pos].write(p.x);
            y_spare[pos].write(p.y);
            z_spare[pos].write(p.z);

            cell_cursors[cell] += 1;
        }

        debug_assert!(
            cell_cursors
                .iter()
                .zip(cell_offsets.iter().skip(1))
                .all(|(cursor, offset_end)| cursor == offset_end),
            "scatter cursors did not reach cell end offsets"
        );

        // SAFETY: The prefix-sum scatter visits each `pos` in `0..n` exactly once, so the first
        // `n` elements of each vec's spare capacity have been fully initialized.
        unsafe {
            point_indices.set_len(n);
            cell_points_x.set_len(n);
            cell_points_y.set_len(n);
            cell_points_z.set_len(n);
        }
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.scatter_soa += t.elapsed();
        }

        // Step 4: Precompute neighbors and ring-2 cells for each cell
        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let neighbors = Self::compute_all_neighbors(res);
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.neighbors += t.elapsed();
        }

        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let (ring2, ring2_lens) = Self::compute_ring2(res, &neighbors);
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.ring2 += t.elapsed();
        }

        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let (cell_centers, cell_cos_radius, cell_sin_radius) = Self::compute_cell_bounds(res);
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.cell_bounds += t.elapsed();
        }

        // Step 5: Precompute security_3x3 threshold per cell using ring-2 caps
        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let security_3x3 = Self::compute_security_3x3(
            res,
            &cell_centers,
            &cell_cos_radius,
            &cell_sin_radius,
            &ring2,
            &ring2_lens,
        );
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.security_3x3 += t.elapsed();
        }

        CubeMapGrid {
            res,
            cell_offsets,
            point_indices,
            point_cells,
            neighbors,
            ring2,
            ring2_lens,
            cell_centers,
            cell_cos_radius,
            cell_sin_radius,
            security_3x3,
            cell_points_x,
            cell_points_y,
            cell_points_z,
        }
    }

    /// Compute 3×3 neighborhood for all cells.
    fn compute_all_neighbors(res: usize) -> Vec<u32> {
        let num_cells = 6 * res * res;
        let mut neighbors = vec![u32::MAX; num_cells * 9];

        for cell in 0..num_cells {
            let (face, iu, iv) = cell_to_face_ij(cell, res);
            let base = cell * 9;

            let center = cell as u32;
            let (lf, lu, lv) = step_one(face, iu, iv, EdgeDir::Left, res);
            let left = (lf * res * res + lv * res + lu) as u32;
            let (rf, ru, rv) = step_one(face, iu, iv, EdgeDir::Right, res);
            let right = (rf * res * res + rv * res + ru) as u32;
            let (df, duu, dvv) = step_one(face, iu, iv, EdgeDir::Down, res);
            let down = (df * res * res + dvv * res + duu) as u32;
            let (uf, uu, uv) = step_one(face, iu, iv, EdgeDir::Up, res);
            let up = (uf * res * res + uv * res + uu) as u32;

            let down_left = diagonal_from_edge_neighbors(center, down, left, res);
            let down_right = diagonal_from_edge_neighbors(center, down, right, res);
            let up_left = diagonal_from_edge_neighbors(center, up, left, res);
            let up_right = diagonal_from_edge_neighbors(center, up, right, res);

            neighbors[base + 0] = down_left;
            neighbors[base + 1] = down;
            neighbors[base + 2] = down_right;
            neighbors[base + 3] = left;
            neighbors[base + 4] = center;
            neighbors[base + 5] = right;
            neighbors[base + 6] = up_left;
            neighbors[base + 7] = up;
            neighbors[base + 8] = up_right;

            debug_assert!(
                {
                    let ns = [
                        down_left, down, down_right, left, center, right, up_left, up, up_right,
                    ];
                    let mut ok = true;
                    for i in 0..9 {
                        for j in (i + 1)..9 {
                            if ns[i] == u32::MAX || ns[j] == u32::MAX {
                                continue;
                            }
                            ok &= ns[i] != ns[j];
                        }
                    }
                    ok
                },
                "duplicate neighbor cell: cell={}, neighbors={:?}",
                cell,
                &neighbors[base..base + 9]
            );
        }

        neighbors
    }

    fn compute_ring2(res: usize, neighbors: &[u32]) -> (Vec<[u32; RING2_MAX]>, Vec<u8>) {
        let num_cells = 6 * res * res;
        let mut ring2 = vec![[u32::MAX; RING2_MAX]; num_cells];
        let mut ring2_lens = vec![0u8; num_cells];

        for cell in 0..num_cells {
            let base = cell * 9;
            let near = &neighbors[base..base + 9];
            debug_assert_eq!(near.len(), 9);

            // Interior cells (>=2 away from any edge) have a fixed ring-2 pattern on the face.
            let (face, iu, iv) = cell_to_face_ij(cell, res);
            if iu >= 2 && iv >= 2 && iu + 2 < res && iv + 2 < res {
                let mut ring = [u32::MAX; RING2_MAX];
                let mut ring_len = 0usize;
                let face_base = face * res * res;
                let iu = iu as isize;
                let iv = iv as isize;
                for dv in -2isize..=2 {
                    for du in -2isize..=2 {
                        if du == 0 && dv == 0 {
                            continue;
                        }
                        if du.abs().max(dv.abs()) != 2 {
                            continue;
                        }
                        let u = (iu + du) as usize;
                        let v = (iv + dv) as usize;
                        ring[ring_len] = (face_base + v * res + u) as u32;
                        ring_len += 1;
                    }
                }
                debug_assert_eq!(
                    ring_len, RING2_MAX,
                    "interior ring2 size mismatch: {}",
                    ring_len
                );
                ring2[cell] = ring;
                ring2_lens[cell] = ring_len as u8;
                continue;
            }

            // Unroll near-membership checks: this is hot (called for every neighbor-of-neighbor).
            let near0 = near[0];
            let near1 = near[1];
            let near2 = near[2];
            let near3 = near[3];
            let near4 = near[4];
            let near5 = near[5];
            let near6 = near[6];
            let near7 = near[7];
            let near8 = near[8];

            #[inline(always)]
            fn ring_contains(ring: &[u32; RING2_MAX], ring_len: usize, v: u32) -> bool {
                let mut i = 0usize;
                while i < ring_len {
                    if ring[i] == v {
                        return true;
                    }
                    i += 1;
                }
                false
            }

            let mut ring = [u32::MAX; RING2_MAX];
            let mut ring_len = 0usize;
            for &n1_cell in near {
                if n1_cell == u32::MAX {
                    continue;
                }
                let n1_idx = n1_cell as usize;
                let b1 = n1_idx * 9;
                let near_of_n1 = &neighbors[b1..b1 + 9];
                for &cand in near_of_n1 {
                    if cand == u32::MAX {
                        continue;
                    }
                    // Skip the 3×3 neighborhood (including self).
                    if cand == near0
                        || cand == near1
                        || cand == near2
                        || cand == near3
                        || cand == near4
                        || cand == near5
                        || cand == near6
                        || cand == near7
                        || cand == near8
                    {
                        continue;
                    }
                    if ring_contains(&ring, ring_len, cand) {
                        continue;
                    }
                    debug_assert!(ring_len < RING2_MAX, "ring2 exceeded max size");
                    ring[ring_len] = cand;
                    ring_len += 1;
                    if ring_len == RING2_MAX {
                        break;
                    }
                }
                if ring_len == RING2_MAX {
                    break;
                }
            }

            debug_assert!(ring_len > 0, "ring2 must be non-empty");

            ring2[cell][..ring_len].copy_from_slice(&ring[..ring_len]);
            ring2_lens[cell] = ring_len as u8;
        }

        (ring2, ring2_lens)
    }

    /// Compute security_3x3 threshold for all cells using ring-2 caps.
    ///
    /// For each cell, finds the max dot product from any point in the cell to any point
    /// in the ring-2 set (Chebyshev distance 2). Uses spherical caps as a conservative bound.
    fn compute_security_3x3(
        res: usize,
        cell_centers: &[Vec3],
        cell_cos_radius: &[f32],
        cell_sin_radius: &[f32],
        ring2: &[[u32; RING2_MAX]],
        ring2_lens: &[u8],
    ) -> Vec<f32> {
        let num_cells = 6 * res * res;
        let mut security = vec![f32::NEG_INFINITY; num_cells];

        for cell in 0..num_cells {
            let ring = &ring2[cell];
            let ring_len = ring2_lens[cell] as usize;

            let center_a = cell_centers[cell];
            let cos_ra = cell_cos_radius[cell];
            let sin_ra = cell_sin_radius[cell];

            let mut max_dot = f32::NEG_INFINITY;
            for i in 0..ring_len {
                let ring_cell = ring[i] as usize;
                let center_b = cell_centers[ring_cell];
                let cos_rb = cell_cos_radius[ring_cell];
                let sin_rb = cell_sin_radius[ring_cell];

                // Compute max dot between two spherical caps
                let cos_d = center_a.dot(center_b).clamp(-1.0, 1.0);

                // If center cell is inside ring cell's cap (distance < radius), max_dot = 1.0
                if cos_d > cos_rb {
                    max_dot = 1.0;
                    break;
                }

                // Max dot = cos(d - ra - rb) where d is center-to-center angle
                // cos(d - r) = cos_d * cos_r + sin_d * sin_r
                let sin_d = (1.0 - cos_d * cos_d).max(0.0).sqrt();

                // First subtract ra: cos(d - ra)
                let cos_d_minus_ra = cos_d * cos_ra + sin_d * sin_ra;
                let sin_d_minus_ra = sin_d * cos_ra - cos_d * sin_ra;

                // Then subtract rb: cos(d - ra - rb)
                let cos_min_angle = cos_d_minus_ra * cos_rb + sin_d_minus_ra.abs() * sin_rb;

                max_dot = max_dot.max(cos_min_angle.clamp(-1.0, 1.0));
            }

            security[cell] = max_dot;
        }

        security
    }

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

    /// Get grid resolution (cells per face).
    #[inline]
    pub fn res(&self) -> usize {
        self.res
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
        let base = cell * 9;
        self.neighbors[base..base + 9].try_into().unwrap()
    }

    /// Get the ring-2 cells (Chebyshev distance 2) for a cell.
    #[inline]
    pub fn cell_ring2(&self, cell: usize) -> &[u32] {
        let len = self.ring2_lens[cell] as usize;
        &self.ring2[cell][..len]
    }

    /// Get precomputed security_3x3 threshold for a cell.
    /// This is the max dot from any point in the cell to any ring-2 cell.
    #[inline]
    pub fn cell_security_3x3(&self, cell: usize) -> f32 {
        self.security_3x3[cell]
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
        let mut cos_d = query.dot(center);
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

    /// Scratch-based k-NN query that writes results into `out_indices` (sorted closest-first).
    ///
    /// This is the preferred high-throughput API: it avoids per-query allocations.
    pub fn find_k_nearest_with_scratch_into(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        k: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) {
        self.find_k_nearest_with_scratch_into_impl(
            points,
            query,
            query_idx,
            k,
            scratch,
            out_indices,
        );
    }

    /// Non-resumable scratch-based k-NN query optimized for unit vectors:
    /// maintains an unsorted top-k by dot product and sorts once at the end.
    pub fn find_k_nearest_with_scratch_into_dot_topk(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        k: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) {
        self.find_k_nearest_with_scratch_into_dot_topk_impl(
            points,
            query,
            query_idx,
            k,
            scratch,
            out_indices,
        );
    }

    /// Start a resumable k-NN query.
    ///
    /// Unlike `find_k_nearest_with_scratch_into`, this preserves scratch state so the query
    /// can be resumed with `resume_k_nearest_into` to fetch additional neighbors.
    ///
    /// `track_limit` controls how many candidates we track internally. This bounds
    /// how far we can resume: if track_limit=48, we can resume up to k=48 without
    /// losing any neighbors.
    pub fn find_k_nearest_resumable_into(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) -> KnnStatus {
        self.find_k_nearest_resumable_into_impl(
            points,
            query,
            query_idx,
            k,
            track_limit,
            scratch,
            out_indices,
        )
    }

    /// Resume a k-NN query to fetch additional neighbors.
    ///
    /// Call this after `find_k_nearest_resumable_into` when you need more neighbors.
    /// `new_k` should be larger than the previous k but within the original `track_limit`.
    pub fn resume_k_nearest_into(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        new_k: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) -> KnnStatus {
        self.resume_k_nearest_into_impl(points, query, query_idx, new_k, scratch, out_indices)
    }

    /// Resume a k-NN query and append only the new neighbors to `out_indices`.
    ///
    /// `prev_k` is the number of neighbors previously produced into `out_indices` for this
    /// same scratch/query state. On success, this appends indices for the range
    /// `prev_k..new_k` (or less if exhausted).
    ///
    /// This is an optimization over `resume_k_nearest_into` when the caller wants to process
    /// only the newly discovered neighbors.
    pub fn resume_k_nearest_append_into(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        prev_k: usize,
        new_k: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) -> KnnStatus {
        self.resume_k_nearest_append_into_impl(
            points,
            query,
            query_idx,
            prev_k,
            new_k,
            scratch,
            out_indices,
        )
    }

    fn bruteforce_fill_impl<P: UnitVec>(
        &self,
        points: &[P],
        query: P,
        query_idx: usize,
        scratch: &mut CubeMapGridScratch,
    ) {
        if scratch.use_fixed {
            scratch.candidates_len = 0;
        } else {
            scratch.candidates_vec.clear();
        }

        for (idx, p) in points.iter().enumerate() {
            if idx == query_idx {
                continue;
            }
            let dist_sq = unit_vec_dist_sq_generic(*p, query);
            scratch.try_add_neighbor(idx, dist_sq);
        }
        scratch.exhausted = true;
    }

    fn bruteforce_fill_dot_impl<P: UnitVec>(
        &self,
        points: &[P],
        query: P,
        query_idx: usize,
        k: usize,
        scratch: &mut CubeMapGridScratch,
    ) {
        scratch.candidates_dot.clear();
        scratch.worst_dot = f32::NEG_INFINITY;
        scratch.worst_dot_pos = 0;

        for (idx, p) in points.iter().enumerate() {
            if idx == query_idx {
                continue;
            }
            let dot = p.dot(query);
            scratch.try_add_neighbor_dot(idx, dot, k);
        }
        scratch.exhausted = true;
    }

    #[inline]
    fn scan_cell_points_impl<P: UnitVec>(
        &self,
        _points: &[P],
        query: P,
        query_idx: usize,
        cell: usize,
        scratch: &mut CubeMapGridScratch,
    ) {
        let start = self.cell_offsets[cell] as usize;
        let end = self.cell_offsets[cell + 1] as usize;

        // Use SoA layout for contiguous memory access
        let xs = &self.cell_points_x[start..end];
        let ys = &self.cell_points_y[start..end];
        let zs = &self.cell_points_z[start..end];
        let indices = &self.point_indices[start..end];

        let qv = query.to_vec3();
        let (qx, qy, qz) = (qv.x, qv.y, qv.z);

        for i in 0..xs.len() {
            let pidx = indices[i] as usize;
            if pidx == query_idx {
                continue;
            }
            // Contiguous SoA access - should auto-vectorize well
            let dot = xs[i] * qx + ys[i] * qy + zs[i] * qz;
            let dist_sq = (2.0 - 2.0 * dot).max(0.0);
            scratch.try_add_neighbor(pidx, dist_sq);
        }
    }

    #[inline]
    fn scan_cell_points_dot_impl<P: UnitVec>(
        &self,
        _points: &[P],
        query: P,
        query_idx: usize,
        k: usize,
        cell: usize,
        scratch: &mut CubeMapGridScratch,
    ) {
        let start = self.cell_offsets[cell] as usize;
        let end = self.cell_offsets[cell + 1] as usize;

        // Use SoA layout for contiguous memory access
        let xs = &self.cell_points_x[start..end];
        let ys = &self.cell_points_y[start..end];
        let zs = &self.cell_points_z[start..end];
        let indices = &self.point_indices[start..end];

        let qv = query.to_vec3();
        let (qx, qy, qz) = (qv.x, qv.y, qv.z);

        for i in 0..xs.len() {
            let pidx = indices[i] as usize;
            if pidx == query_idx {
                continue;
            }
            // Contiguous SoA access - should auto-vectorize well
            let dot = xs[i] * qx + ys[i] * qy + zs[i] * qz;
            scratch.try_add_neighbor_dot(pidx, dot, k);
        }
    }

    fn find_k_nearest_with_scratch_into_impl<P: UnitVec>(
        &self,
        points: &[P],
        query: P,
        query_idx: usize,
        k: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) {
        let n = points.len();
        out_indices.clear();

        if k == 0 || n <= 1 {
            return;
        }
        let k = k.min(n - 1);

        let query_vec3 = query.to_vec3();
        let num_cells = 6 * self.res * self.res;
        scratch.begin_query(k, k);

        let start_cell = if query_idx < self.point_cells.len() {
            self.point_cells[query_idx]
        } else {
            self.point_to_cell(query_vec3) as u32
        };
        let mut visited_cells =
            self.seed_start_cell_impl(points, query, query_idx, start_cell, scratch);

        let max_cells_before_bruteforce = (num_cells / 2).max(64);

        if scratch.have_k(k) {
            let kth_dist = scratch.kth_dist_sq(k);
            if let Some((bound, _)) = scratch.peek_cell() {
                if bound >= kth_dist {
                    scratch.copy_k_indices_into(k, out_indices);
                    return;
                }
            } else {
                scratch.copy_k_indices_into(k, out_indices);
                return;
            }
        }

        while let Some((bound_dist_sq, cell_u32)) = scratch.pop_cell() {
            let kth_dist = scratch.kth_dist_sq(k);
            if scratch.have_k(k) && bound_dist_sq >= kth_dist {
                break;
            }

            visited_cells += 1;
            if visited_cells > max_cells_before_bruteforce {
                self.bruteforce_fill_impl(points, query, query_idx, scratch);
                scratch.copy_k_indices_into(k, out_indices);
                return;
            }

            let cell = cell_u32 as usize;
            debug_assert!(cell < num_cells);

            self.scan_cell_points_impl(points, query, query_idx, cell, scratch);

            let base = cell * 9;
            for &ncell in &self.neighbors[base..base + 9] {
                if ncell == u32::MAX || ncell == cell_u32 {
                    continue;
                }
                if !scratch.mark_visited(ncell) {
                    continue;
                }
                let nb = self.cell_min_dist_sq(query_vec3, ncell as usize);
                scratch.push_cell(ncell, nb);
            }
        }

        if !scratch.have_k(k) {
            self.bruteforce_fill_impl(points, query, query_idx, scratch);
        }

        scratch.copy_k_indices_into(k, out_indices);
    }

    fn find_k_nearest_with_scratch_into_dot_topk_impl<P: UnitVec>(
        &self,
        points: &[P],
        query: P,
        query_idx: usize,
        k: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) {
        let n = points.len();
        out_indices.clear();

        if k == 0 || n <= 1 {
            return;
        }
        let k = k.min(n - 1);

        let query_vec3 = query.to_vec3();
        let num_cells = 6 * self.res * self.res;
        scratch.begin_query_dot(k);

        let start_cell = if query_idx < self.point_cells.len() {
            self.point_cells[query_idx]
        } else {
            self.point_to_cell(query_vec3) as u32
        };
        scratch.mark_visited(start_cell);
        scratch.push_cell(start_cell, 0.0);

        let max_cells_before_bruteforce = (num_cells / 2).max(64);
        let mut visited_cells = 0usize;

        while let Some((bound_dist_sq, cell_u32)) = scratch.pop_cell() {
            let kth_dist = scratch.kth_dist_sq_dot(k);
            if scratch.have_k_dot(k) && bound_dist_sq >= kth_dist {
                break;
            }

            visited_cells += 1;
            if visited_cells > max_cells_before_bruteforce {
                self.bruteforce_fill_dot_impl(points, query, query_idx, k, scratch);
                scratch.copy_k_indices_dot_into(k, out_indices);
                return;
            }

            let cell = cell_u32 as usize;
            debug_assert!(cell < num_cells);
            self.scan_cell_points_dot_impl(points, query, query_idx, k, cell, scratch);

            let base = cell * 9;
            for &ncell in &self.neighbors[base..base + 9] {
                if ncell == u32::MAX || ncell == cell_u32 {
                    continue;
                }
                if !scratch.mark_visited(ncell) {
                    continue;
                }
                let nb = self.cell_min_dist_sq(query_vec3, ncell as usize);
                scratch.push_cell(ncell, nb);
            }
        }

        if !scratch.have_k_dot(k) {
            self.bruteforce_fill_dot_impl(points, query, query_idx, k, scratch);
        }

        scratch.copy_k_indices_dot_into(k, out_indices);
    }

    fn find_k_nearest_resumable_into_impl<P: UnitVec>(
        &self,
        points: &[P],
        query: P,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) -> KnnStatus {
        let n = points.len();
        out_indices.clear();

        if k == 0 || n <= 1 {
            return KnnStatus::Exhausted;
        }
        let k = k.min(n - 1);
        let track_limit = track_limit.max(k).min(n - 1);

        let query_vec3 = query.to_vec3();
        let num_cells = 6 * self.res * self.res;
        scratch.begin_query(k, track_limit);

        let start_cell = if query_idx < self.point_cells.len() {
            self.point_cells[query_idx]
        } else {
            self.point_to_cell(query_vec3) as u32
        };
        let seeded_visited_cells =
            self.seed_start_cell_impl(points, query, query_idx, start_cell, scratch);

        let exhausted = self.knn_search_loop_impl(
            points,
            query,
            query_idx,
            k,
            num_cells,
            seeded_visited_cells,
            scratch,
        );

        scratch.copy_k_indices_into(k, out_indices);
        if exhausted {
            KnnStatus::Exhausted
        } else {
            KnnStatus::CanResume
        }
    }

    fn resume_k_nearest_into_impl<P: UnitVec>(
        &self,
        points: &[P],
        query: P,
        query_idx: usize,
        new_k: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) -> KnnStatus {
        let n = points.len();
        out_indices.clear();

        if new_k == 0 || n <= 1 {
            return KnnStatus::Exhausted;
        }
        let new_k = new_k.min(n - 1);

        if new_k > scratch.track_limit {
            scratch.begin_query(new_k, new_k);
            self.bruteforce_fill_impl(points, query, query_idx, scratch);
            scratch.copy_k_indices_into(new_k, out_indices);
            return KnnStatus::Exhausted;
        }

        if scratch.exhausted {
            scratch.copy_k_indices_into(new_k, out_indices);
            return KnnStatus::Exhausted;
        }

        if scratch.have_k(new_k) {
            let kth_dist = scratch.kth_dist_sq(new_k);
            if let Some((bound, _)) = scratch.peek_cell() {
                if bound >= kth_dist {
                    scratch.copy_k_indices_into(new_k, out_indices);
                    return KnnStatus::CanResume;
                }
            } else {
                scratch.copy_k_indices_into(new_k, out_indices);
                return KnnStatus::Exhausted;
            }
        }

        let num_cells = 6 * self.res * self.res;
        let exhausted =
            self.knn_search_loop_impl(points, query, query_idx, new_k, num_cells, 0, scratch);

        scratch.copy_k_indices_into(new_k, out_indices);
        if exhausted {
            KnnStatus::Exhausted
        } else {
            KnnStatus::CanResume
        }
    }

    fn resume_k_nearest_append_into_impl<P: UnitVec>(
        &self,
        points: &[P],
        query: P,
        query_idx: usize,
        prev_k: usize,
        new_k: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) -> KnnStatus {
        let n = points.len();
        if new_k == 0 || n <= 1 {
            out_indices.clear();
            return KnnStatus::Exhausted;
        }

        let new_k = new_k.min(n - 1);
        let prev_k = prev_k.min(new_k);
        out_indices.truncate(prev_k);

        if new_k <= prev_k {
            return if scratch.exhausted {
                KnnStatus::Exhausted
            } else {
                KnnStatus::CanResume
            };
        }

        if new_k > scratch.track_limit {
            // Fallback: recompute exhaustively, but only append the missing suffix.
            scratch.begin_query(new_k, new_k);
            self.bruteforce_fill_impl(points, query, query_idx, scratch);
            scratch.append_k_indices_into(prev_k, new_k, out_indices);
            return KnnStatus::Exhausted;
        }

        if scratch.exhausted {
            scratch.append_k_indices_into(prev_k, new_k, out_indices);
            return KnnStatus::Exhausted;
        }

        if scratch.have_k(new_k) {
            let kth_dist = scratch.kth_dist_sq(new_k);
            if let Some((bound, _)) = scratch.peek_cell() {
                if bound >= kth_dist {
                    scratch.append_k_indices_into(prev_k, new_k, out_indices);
                    return KnnStatus::CanResume;
                }
            } else {
                scratch.append_k_indices_into(prev_k, new_k, out_indices);
                return KnnStatus::Exhausted;
            }
        }

        let num_cells = 6 * self.res * self.res;
        let exhausted =
            self.knn_search_loop_impl(points, query, query_idx, new_k, num_cells, 0, scratch);

        scratch.append_k_indices_into(prev_k, new_k, out_indices);
        if exhausted {
            KnnStatus::Exhausted
        } else {
            KnnStatus::CanResume
        }
    }

    fn knn_search_loop_impl<P: UnitVec>(
        &self,
        points: &[P],
        query: P,
        query_idx: usize,
        k: usize,
        num_cells: usize,
        seeded_visited_cells: usize,
        scratch: &mut CubeMapGridScratch,
    ) -> bool {
        let query_vec3 = query.to_vec3();
        let max_cells_before_bruteforce = (num_cells / 2).max(64);
        let mut visited_cells = seeded_visited_cells;

        loop {
            let Some((bound_dist_sq, cell_u32)) = scratch.peek_cell() else {
                return true;
            };

            let kth_dist = scratch.kth_dist_sq(k);
            if scratch.have_k(k) && bound_dist_sq >= kth_dist {
                return false;
            }

            scratch.pop_cell();

            visited_cells += 1;
            if visited_cells > max_cells_before_bruteforce {
                self.bruteforce_fill_impl(points, query, query_idx, scratch);
                return true;
            }

            let cell = cell_u32 as usize;
            debug_assert!(cell < num_cells);

            self.scan_cell_points_impl(points, query, query_idx, cell, scratch);

            let base = cell * 9;
            for &ncell in &self.neighbors[base..base + 9] {
                if ncell == u32::MAX || ncell == cell_u32 {
                    continue;
                }
                if !scratch.mark_visited(ncell) {
                    continue;
                }
                let nb = self.cell_min_dist_sq(query_vec3, ncell as usize);
                scratch.push_cell(ncell, nb);
            }
        }
    }

    fn seed_start_cell_impl<P: UnitVec>(
        &self,
        points: &[P],
        query: P,
        query_idx: usize,
        start_cell: u32,
        scratch: &mut CubeMapGridScratch,
    ) -> usize {
        scratch.mark_visited(start_cell);
        self.scan_cell_points_impl(points, query, query_idx, start_cell as usize, scratch);

        let query_vec3 = query.to_vec3();
        let neighbors = self.cell_neighbors(start_cell as usize);
        for &ncell in neighbors.iter() {
            if ncell == u32::MAX || ncell == start_cell {
                continue;
            }
            if !scratch.mark_visited(ncell) {
                continue;
            }
            let bound = self.cell_min_dist_sq(query_vec3, ncell as usize);
            scratch.push_cell(ncell, bound);
        }

        1
    }

    fn compute_cell_bounds(res: usize) -> (Vec<Vec3>, Vec<f32>, Vec<f32>) {
        let num_cells = 6 * res * res;
        let mut centers = Vec::with_capacity(num_cells);
        let mut cos_r = Vec::with_capacity(num_cells);
        let mut sin_r = Vec::with_capacity(num_cells);

        #[inline]
        fn plane_normal_u(face: usize, u: f32) -> Vec3 {
            let n = match face {
                0 | 1 => Vec3::new(u, 0.0, 1.0),  // z + u x = 0
                2 => Vec3::new(1.0, -u, 0.0),     // x - u y = 0
                3 => Vec3::new(1.0, u, 0.0),      // x + u y = 0
                4 | 5 => Vec3::new(1.0, 0.0, -u), // x - u z = 0
                _ => unreachable!(),
            };
            n.normalize()
        }

        #[inline]
        fn plane_normal_v(face: usize, v: f32) -> Vec3 {
            let n = match face {
                0 => Vec3::new(-v, 1.0, 0.0),    // y - v x = 0
                1 => Vec3::new(v, 1.0, 0.0),     // y + v x = 0
                2 | 3 => Vec3::new(0.0, v, 1.0), // z + v y = 0
                4 => Vec3::new(0.0, 1.0, -v),    // y - v z = 0
                5 => Vec3::new(0.0, 1.0, v),     // y + v z = 0
                _ => unreachable!(),
            };
            n.normalize()
        }

        #[inline]
        fn inside_all(p: Vec3, planes: &[Vec3; 4]) -> bool {
            planes.iter().all(|n| n.dot(p) >= -1e-6)
        }

        for cell in 0..num_cells {
            let (face, iu, iv) = cell_to_face_ij(cell, res);

            let s0 = (iu as f32) / res as f32;
            let s1 = ((iu + 1) as f32) / res as f32;
            let t0 = (iv as f32) / res as f32;
            let t1 = ((iv + 1) as f32) / res as f32;
            let u0 = st_to_uv(s0);
            let u1 = st_to_uv(s1);
            let v0 = st_to_uv(t0);
            let v1 = st_to_uv(t1);

            let sc = (s0 + s1) * 0.5;
            let tc = (t0 + t1) * 0.5;
            let uc = st_to_uv(sc);
            let vc = st_to_uv(tc);
            let center = face_uv_to_3d(face, uc, vc);

            let mut planes = [
                plane_normal_u(face, u0),
                plane_normal_u(face, u1),
                plane_normal_v(face, v0),
                plane_normal_v(face, v1),
            ];
            for n in planes.iter_mut() {
                if n.dot(center) < 0.0 {
                    *n = -*n;
                }
            }

            #[inline]
            fn update_min_dot(center: Vec3, p: Vec3, min_dot: &mut f32) {
                let dot = center.dot(p).clamp(-1.0, 1.0);
                *min_dot = (*min_dot).min(dot);
            }

            // Track min dot (= cos of max angle) using exact edge/vertex candidates.
            let mut min_dot = 1.0f32;

            // Corners: intersections of adjacent u/v planes.
            let corners = [
                (planes[0], planes[2]),
                (planes[0], planes[3]),
                (planes[1], planes[2]),
                (planes[1], planes[3]),
            ];
            for (a, b) in corners {
                let cross = a.cross(b);
                if cross.length_squared() <= 1e-12 {
                    continue;
                }
                let mut p = cross.normalize();
                if !inside_all(p, &planes) {
                    p = -p;
                }
                if inside_all(p, &planes) {
                    update_min_dot(center, p, &mut min_dot);
                }
            }

            // Edge interior candidates: farthest point on each boundary great circle.
            for n in planes.iter() {
                let proj = center - *n * center.dot(*n);
                let len2 = proj.length_squared();
                if len2 <= 1e-12 {
                    min_dot = min_dot.min(0.0);
                    continue;
                }
                let p = -proj / len2.sqrt();
                if inside_all(p, &planes) {
                    update_min_dot(center, p, &mut min_dot);
                }
            }

            // Inflate by epsilon using angle addition formulas (no trig needed).
            // cos(a + ε) = cos(a)·cos(ε) - sin(a)·sin(ε)
            // sin(a + ε) = sin(a)·cos(ε) + cos(a)·sin(ε)
            // For ε = 1e-4: cos(ε) ≈ 1.0, sin(ε) ≈ 1e-4
            const SIN_EPS: f32 = 1e-4;
            let sin_a = (1.0 - min_dot * min_dot).max(0.0).sqrt();
            let cos_radius = (min_dot - sin_a * SIN_EPS).clamp(-1.0, 1.0);
            let sin_radius = (sin_a + min_dot * SIN_EPS).clamp(0.0, 1.0);

            centers.push(center);
            cos_r.push(cos_radius);
            sin_r.push(sin_radius);
        }

        (centers, cos_r, sin_r)
    }

    /// Statistics about the grid.
    pub fn stats(&self) -> GridStats {
        let num_cells = 6 * self.res * self.res;
        let mut min_pts = u32::MAX;
        let mut max_pts = 0u32;
        let mut empty = 0usize;

        for cell in 0..num_cells {
            let count = self.cell_offsets[cell + 1] - self.cell_offsets[cell];
            min_pts = min_pts.min(count);
            max_pts = max_pts.max(count);
            if count == 0 {
                empty += 1;
            }
        }

        GridStats {
            num_cells,
            num_points: self.point_indices.len(),
            min_points_per_cell: min_pts as usize,
            max_points_per_cell: max_pts as usize,
            empty_cells: empty,
            avg_points_per_cell: self.point_indices.len() as f64 / num_cells as f64,
        }
    }
}

#[derive(Debug)]
pub struct GridStats {
    pub num_cells: usize,
    pub num_points: usize,
    pub min_points_per_cell: usize,
    pub max_points_per_cell: usize,
    pub empty_cells: usize,
    pub avg_points_per_cell: f64,
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
}
