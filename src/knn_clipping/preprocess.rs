//! Preprocessing helpers (merge close generators).

use glam::Vec3;

/// Result of merging coincident generators before Voronoi computation.
pub struct MergeResult {
    /// Points to use for Voronoi (representatives only, or all if no merges).
    pub effective_points: Vec<Vec3>,
    /// Maps original point index -> representative index in effective_points.
    /// If no merges occurred, this is just identity (0, 1, 2, ...).
    pub original_to_effective: Vec<usize>,
    /// Number of points that were merged (removed).
    pub num_merged: usize,
}

/// Find and merge coincident (near-identical) generators.
/// Uses strict radius-based merging to ensure no remaining pair is within threshold.
/// Returns effective points (representatives) and a mapping from original to effective indices.
///
/// NOTE: A potentially more efficient approach for the future would be to detect close
/// neighbors during cell construction and "borrow" the cell from the close neighbor
/// when a cell dies. This would avoid the preprocessing pass but requires more complex
/// recovery logic in the cell builder.
pub fn merge_close_points(points: &[Vec3], threshold: f32) -> MergeResult {
    let n = points.len();
    if n == 0 {
        return MergeResult {
            effective_points: Vec::new(),
            original_to_effective: Vec::new(),
            num_merged: 0,
        };
    }

    if threshold <= 0.0 {
        return MergeResult {
            effective_points: points.to_vec(),
            original_to_effective: (0..n).collect(),
            num_merged: 0,
        };
    }

    let threshold_sq = threshold * threshold;

    // Use a cube-map grid to restrict candidate pairs to small spatial bins.
    //
    // This avoids the very high constant factors of hashing ~N distinct 3D grid keys
    // (especially when `threshold` is tiny) while still checking all near pairs
    // by scanning the 3×3 neighborhood of each cell.
    //
    // Target fewer points per cell than the KNN grid to keep pairwise checks cheap.
    const PREPROCESS_TARGET_POINTS_PER_CELL: f64 = 12.0;
    let target = PREPROCESS_TARGET_POINTS_PER_CELL.max(1.0);
    let res = ((n as f64 / (6.0 * target)).sqrt() as usize).max(4);
    let grid = crate::cube_grid::CubeMapGrid::new(points, res);
    let num_cells = 6 * res * res;

    struct MergeDsu {
        parent: Vec<u32>,
    }

    impl MergeDsu {
        fn new(n: usize) -> Self {
            let mut parent = Vec::with_capacity(n);
            for i in 0..n {
                parent.push(i as u32);
            }
            Self { parent }
        }

        fn find(&mut self, x: u32) -> u32 {
            let idx = x as usize;
            let p = self.parent[idx];
            if p != x {
                let root = self.find(p);
                self.parent[idx] = root;
            }
            self.parent[idx]
        }

        // Order-dependent merge: keep the smaller representative as the parent.
        fn union_keep_min(&mut self, a: u32, b: u32) -> bool {
            let ra = self.find(a);
            let rb = self.find(b);
            if ra == rb {
                return false;
            }
            let (min, max) = if ra <= rb { (ra, rb) } else { (rb, ra) };
            self.parent[max as usize] = min;
            true
        }
    }

    let mut dsu = MergeDsu::new(n);

    for cell in 0..num_cells {
        let a_points = grid.cell_points(cell);
        if a_points.len() >= 2 {
            for i in 0..a_points.len() {
                let ai = a_points[i] as usize;
                for j in (i + 1)..a_points.len() {
                    let aj = a_points[j] as usize;
                    let dist_sq = (points[ai] - points[aj]).length_squared();
                    if dist_sq < threshold_sq {
                        let _ = dsu.union_keep_min(ai as u32, aj as u32);
                    }
                }
            }
        }

        let neighbors = grid.cell_neighbors(cell);
        for &nb_u32 in neighbors {
            if nb_u32 == u32::MAX {
                continue;
            }
            let nb = nb_u32 as usize;
            if nb <= cell {
                continue;
            }
            let b_points = grid.cell_points(nb);
            if a_points.is_empty() || b_points.is_empty() {
                continue;
            }
            for &ai_u32 in a_points {
                let ai = ai_u32 as usize;
                for &bj_u32 in b_points {
                    let bj = bj_u32 as usize;
                    let dist_sq = (points[ai] - points[bj]).length_squared();
                    if dist_sq < threshold_sq {
                        let _ = dsu.union_keep_min(ai as u32, bj as u32);
                    }
                }
            }
        }
    }

    // Count how many unique representatives we have
    let mut rep_to_effective: Vec<Option<usize>> = vec![None; n];
    let mut effective_points = Vec::new();
    let mut original_to_effective = vec![0usize; n];

    for i in 0..n {
        let rep = dsu.find(i as u32) as usize;
        if rep_to_effective[rep].is_none() {
            rep_to_effective[rep] = Some(effective_points.len());
            effective_points.push(points[rep]);
        }
        original_to_effective[i] = rep_to_effective[rep].unwrap();
    }

    let num_merged = n - effective_points.len();

    MergeResult {
        effective_points,
        original_to_effective,
        num_merged,
    }
}
