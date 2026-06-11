//! Preprocessing helpers (weld near-coincident generators).

use glam::Vec3;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Result of welding coincident generators before Voronoi computation.
pub struct MergeResult {
    /// Points to use for Voronoi (representatives only, or all if no merges).
    pub effective_points: Vec<Vec3>,
    /// Maps original point index -> representative index in effective_points.
    /// If no merges occurred, this is just identity (0, 1, 2, ...).
    pub original_to_effective: Vec<usize>,
    /// Number of points that were merged (removed).
    pub num_merged: usize,
}

/// Sparse quantized grid used to find candidate pairs within a weld threshold.
///
/// Cell size is at least 64x the threshold (clamped so indices fit 21 bits per
/// axis), so a pair within `threshold` is either in the same cell or split
/// across a wall that both endpoints lie within `threshold` of. Same-key runs
/// plus wall-adjacent neighbor lookups for boundary points therefore cover
/// every candidate pair.
struct WeldGrid {
    cell: f64,
    inv_cell: f64,
    /// Boundary-proximity padding: threshold plus a small relative guard for
    /// the floating-point error in the position computation.
    pad: f64,
}

/// Coordinate shift so quantized positions are strictly positive for inputs
/// in a generous [-1.2, 1.2] band around the unit sphere.
const COORD_SHIFT: f64 = 1.25;
const AXIS_BITS: u32 = 21;

impl WeldGrid {
    fn new(threshold: f32) -> Self {
        let thr = threshold as f64;
        let min_cell = 2.5 / (1u64 << AXIS_BITS) as f64;
        let cell = (thr * 64.0).max(min_cell);
        WeldGrid {
            cell,
            inv_cell: 1.0 / cell,
            pad: thr * (1.0 + 1e-6),
        }
    }

    #[inline]
    fn axis(&self, coord: f32) -> (u64, bool, bool) {
        let pos = ((coord as f64) + COORD_SHIFT) * self.inv_cell;
        let q = pos as u64;
        let low_dist = (pos - q as f64) * self.cell;
        let near_low = low_dist < self.pad && q > 0;
        let near_high = (self.cell - low_dist) < self.pad;
        (q, near_low, near_high)
    }

    #[inline]
    fn key(&self, p: Vec3) -> u64 {
        let (qx, _, _) = self.axis(p.x);
        let (qy, _, _) = self.axis(p.y);
        let (qz, _, _) = self.axis(p.z);
        (qx << (2 * AXIS_BITS)) | (qy << AXIS_BITS) | qz
    }

    /// Keys of wall-adjacent neighbor cells this point is within `pad` of.
    /// Empty for interior points (the overwhelmingly common case).
    fn boundary_neighbor_keys(&self, p: Vec3, out: &mut Vec<u64>) {
        out.clear();
        let ax = self.axis(p.x);
        let ay = self.axis(p.y);
        let az = self.axis(p.z);
        let options = |(q, near_low, near_high): (u64, bool, bool)| {
            [Some(q), near_low.then(|| q - 1), near_high.then(|| q + 1)]
        };
        let xs = options(ax);
        let ys = options(ay);
        let zs = options(az);
        for qx in xs.iter().flatten() {
            for qy in ys.iter().flatten() {
                for qz in zs.iter().flatten() {
                    if (*qx, *qy, *qz) == (ax.0, ay.0, az.0) {
                        continue;
                    }
                    out.push((qx << (2 * AXIS_BITS)) | (qy << AXIS_BITS) | qz);
                }
            }
        }
    }
}

fn identity_result(points: &[Vec3]) -> MergeResult {
    MergeResult {
        effective_points: points.to_vec(),
        original_to_effective: (0..points.len()).collect(),
        num_merged: 0,
    }
}

/// Find and weld coincident (near-identical) generators.
///
/// Strict radius-based welding: any pair within `threshold` ends up in the
/// same weld class. Welding is transitive (a chain of sub-threshold steps
/// collapses into one class). The class representative is the smallest
/// original index, and representatives keep their original relative order in
/// `effective_points`.
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
        return identity_result(points);
    }

    let grid = WeldGrid::new(threshold);

    // Sorted (cell key, point index) pairs form the sparse grid.
    #[cfg(feature = "parallel")]
    let mut keyed: Vec<(u64, u32)> = (0..n as u32)
        .into_par_iter()
        .map(|i| (grid.key(points[i as usize]), i))
        .collect();
    #[cfg(not(feature = "parallel"))]
    let mut keyed: Vec<(u64, u32)> = (0..n as u32)
        .map(|i| (grid.key(points[i as usize]), i))
        .collect();

    #[cfg(feature = "parallel")]
    keyed.par_sort_unstable();
    #[cfg(not(feature = "parallel"))]
    keyed.sort_unstable();

    let threshold_sq = threshold * threshold;
    let mut pairs: Vec<(u32, u32)> = Vec::new();

    // Same-cell candidates: pairwise within each equal-key run. Runs are
    // almost always length 1, so this is a near-free linear walk.
    let mut run_start = 0usize;
    while run_start < n {
        let key = keyed[run_start].0;
        let mut run_end = run_start + 1;
        while run_end < n && keyed[run_end].0 == key {
            run_end += 1;
        }
        if run_end - run_start >= 2 {
            for i in run_start..run_end {
                let ai = keyed[i].1;
                for &(_, aj) in &keyed[(i + 1)..run_end] {
                    let dist_sq = (points[ai as usize] - points[aj as usize]).length_squared();
                    if dist_sq < threshold_sq {
                        pairs.push((ai, aj));
                    }
                }
            }
        }
        run_start = run_end;
    }

    // Cross-wall candidates: points within `threshold` of a cell wall check
    // the cells across those walls. A pair split across a wall has both
    // endpoints within `threshold` of it, so gating on `nkey > key` makes
    // exactly one side do the check.
    let scan_boundary = |range: std::ops::Range<usize>| -> Vec<(u32, u32)> {
        let mut local: Vec<(u32, u32)> = Vec::new();
        let mut neighbor_keys: Vec<u64> = Vec::new();
        for &(key, idx) in &keyed[range] {
            let p = points[idx as usize];
            grid.boundary_neighbor_keys(p, &mut neighbor_keys);
            for &nkey in &neighbor_keys {
                if nkey <= key {
                    continue;
                }
                let start = keyed.partition_point(|&(k, _)| k < nkey);
                for &(k, other) in &keyed[start..] {
                    if k != nkey {
                        break;
                    }
                    let dist_sq = (p - points[other as usize]).length_squared();
                    if dist_sq < threshold_sq {
                        local.push((idx, other));
                    }
                }
            }
        }
        local
    };

    #[cfg(feature = "parallel")]
    {
        const CHUNK: usize = 1 << 16;
        let chunk_pairs: Vec<Vec<(u32, u32)>> = (0..n.div_ceil(CHUNK))
            .into_par_iter()
            .map(|c| scan_boundary(c * CHUNK..((c + 1) * CHUNK).min(n)))
            .collect();
        for chunk in chunk_pairs {
            pairs.extend(chunk);
        }
    }
    #[cfg(not(feature = "parallel"))]
    pairs.extend(scan_boundary(0..n));

    if pairs.is_empty() {
        return identity_result(points);
    }

    let mut dsu = super::union_find::UnionFind::new(n);
    for &(a, b) in &pairs {
        let _ = dsu.union_keep_min(a, b);
    }

    // Extract representatives, preserving first-occurrence order.
    let mut rep_to_effective: Vec<Option<usize>> = vec![None; n];
    let mut effective_points = Vec::new();
    let mut original_to_effective = vec![0usize; n];

    for (i, slot) in original_to_effective.iter_mut().enumerate() {
        let rep = dsu.find(i as u32) as usize;
        if rep_to_effective[rep].is_none() {
            rep_to_effective[rep] = Some(effective_points.len());
            effective_points.push(points[rep]);
        }
        *slot = rep_to_effective[rep].unwrap();
    }

    let num_merged = n - effective_points.len();

    MergeResult {
        effective_points,
        original_to_effective,
        num_merged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize()
    }

    #[test]
    fn no_welds_on_well_separated_points() {
        let points = vec![
            unit(1.0, 0.0, 0.0),
            unit(0.0, 1.0, 0.0),
            unit(0.0, 0.0, 1.0),
            unit(-1.0, 0.0, 0.0),
        ];
        let result = merge_close_points(&points, 1e-6);
        assert_eq!(result.num_merged, 0);
        assert_eq!(result.original_to_effective, vec![0, 1, 2, 3]);
        assert_eq!(result.effective_points.len(), 4);
    }

    #[test]
    fn welds_exact_duplicates() {
        let p = unit(0.3, -0.7, 0.2);
        let points = vec![unit(1.0, 0.0, 0.0), p, unit(0.0, 1.0, 0.0), p];
        let result = merge_close_points(&points, 1e-6);
        assert_eq!(result.num_merged, 1);
        assert_eq!(
            result.original_to_effective[1],
            result.original_to_effective[3]
        );
        assert_eq!(result.effective_points.len(), 3);
    }

    #[test]
    fn welds_pair_within_threshold() {
        let p = unit(0.3, -0.7, 0.2);
        let q = Vec3::new(p.x + 4e-7, p.y, p.z);
        let points = vec![p, q, unit(0.0, 1.0, 0.0)];
        let result = merge_close_points(&points, 1e-6);
        assert_eq!(result.num_merged, 1);
        assert_eq!(
            result.original_to_effective[0],
            result.original_to_effective[1]
        );
        // Representative is the smallest original index.
        assert_eq!(result.effective_points[result.original_to_effective[0]], p);
    }

    #[test]
    fn keeps_pair_outside_threshold() {
        let p = unit(0.3, -0.7, 0.2);
        let q = Vec3::new(p.x + 4e-6, p.y, p.z);
        let points = vec![p, q, unit(0.0, 1.0, 0.0)];
        let result = merge_close_points(&points, 1e-6);
        assert_eq!(result.num_merged, 0);
    }

    #[test]
    fn welds_transitive_chain() {
        let p = unit(0.3, -0.7, 0.2);
        let q = Vec3::new(p.x + 8e-7, p.y, p.z);
        let r = Vec3::new(p.x + 1.6e-6, p.y, p.z);
        // p-q and q-r are within 1e-6; p-r is not. One class regardless.
        let points = vec![p, q, r, unit(0.0, 1.0, 0.0)];
        let result = merge_close_points(&points, 1e-6);
        assert_eq!(result.num_merged, 2);
        let eff = result.original_to_effective[0];
        assert_eq!(result.original_to_effective[1], eff);
        assert_eq!(result.original_to_effective[2], eff);
    }

    #[test]
    fn welds_pairs_straddling_cell_walls() {
        // Construct pairs straddling quantization walls for a coarse threshold,
        // where the wall positions are easy to hit: cell = 64 * threshold.
        let threshold = 1e-3f32;
        let cell = (threshold as f64) * 64.0;
        // Choose a coordinate just below a wall so p and q land in different cells.
        let wall = (20.0 * cell - COORD_SHIFT) as f32;
        let p = Vec3::new(wall - 2e-4, 0.5, 0.5);
        let q = Vec3::new(wall + 2e-4, 0.5, 0.5);
        let far = Vec3::new(wall + 0.3, 0.5, 0.5);
        let points = vec![p, q, far];
        let result = merge_close_points(&points, threshold);
        assert_eq!(
            result.num_merged, 1,
            "pair straddling a quantization wall must still weld"
        );
        assert_eq!(
            result.original_to_effective[0],
            result.original_to_effective[1]
        );
    }

    #[test]
    fn empty_and_disabled_threshold() {
        assert_eq!(merge_close_points(&[], 1e-6).effective_points.len(), 0);
        let points = vec![unit(1.0, 0.0, 0.0), unit(1.0, 0.0, 0.0)];
        let result = merge_close_points(&points, 0.0);
        assert_eq!(result.num_merged, 0);
    }
}
