//! Preprocessing helpers (weld near-coincident generators).

use glam::Vec3;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Result of welding coincident generators before Voronoi computation.
pub struct MergeResult {
    /// Points to use for Voronoi (representatives only, or all if no merges).
    pub effective_points: Vec<Vec3>,
    /// Maps original point index -> representative index in effective_points.
    /// If no merges occurred, this is just identity (0, 1, 2, ...).
    pub original_to_effective: Vec<u32>,
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
        original_to_effective: (0..points.len() as u32).collect(),
        num_merged: 0,
    }
}

/// Collect the strict computed-f32 threshold-graph edges with the standalone
/// quantized-key detector.
pub(crate) fn try_collect_close_pairs(
    points: &[Vec3],
    threshold: f32,
) -> Result<Vec<(u32, u32)>, usize> {
    let n = points.len();
    if n == 0 || threshold <= 0.0 {
        return Ok(Vec::new());
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
                    if crate::cube_grid::is_weld_pair(dist_sq, threshold_sq) {
                        if pairs.len() == crate::cube_grid::MAX_RETAINED_WELD_PAIRS {
                            return Err(crate::cube_grid::MAX_RETAINED_WELD_PAIRS + 1);
                        }
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
    let retained = AtomicUsize::new(pairs.len());
    let exceeded = AtomicBool::new(false);
    let scan_boundary = |range: std::ops::Range<usize>| -> Vec<(u32, u32)> {
        let mut local: Vec<(u32, u32)> = Vec::new();
        let mut neighbor_keys: Vec<u64> = Vec::new();
        for &(key, idx) in &keyed[range] {
            if exceeded.load(Relaxed) {
                break;
            }
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
                    if crate::cube_grid::is_weld_pair(dist_sq, threshold_sq) {
                        if retained
                            .fetch_update(Relaxed, Relaxed, |n| {
                                (n < crate::cube_grid::MAX_RETAINED_WELD_PAIRS).then_some(n + 1)
                            })
                            .is_err()
                        {
                            exceeded.store(true, Relaxed);
                            break;
                        }
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

    if exceeded.load(Relaxed) {
        return Err(crate::cube_grid::MAX_RETAINED_WELD_PAIRS + 1);
    }

    Ok(pairs)
}

/// Find and weld coincident (near-identical) generators.
///
/// An edge exists when the computed f32 squared distance is strictly less
/// than the computed f32 squared `threshold`. Weld classes are transitive
/// connected components of that graph. The class representative is the
/// smallest original index, and representatives keep their original relative
/// order in `effective_points`.
pub fn try_merge_close_points(points: &[Vec3], threshold: f32) -> Result<MergeResult, usize> {
    let pairs = try_collect_close_pairs(points, threshold)?;
    if pairs.is_empty() {
        return Ok(identity_result(points));
    }

    Ok(merge_result_from_pairs(points, &pairs).0)
}

#[cfg(test)]
fn merge_close_points(points: &[Vec3], threshold: f32) -> MergeResult {
    try_merge_close_points(points, threshold).unwrap()
}

/// Build a `MergeResult` from detected sub-threshold pairs (any detector).
///
/// Welding is transitive over the pairs; the class representative is the
/// smallest original index, and representatives keep their original
/// relative order in `effective_points`. Also returns `kept[orig]` — true
/// when `orig` is its class representative (survives the weld) — which the
/// grid compaction consumes.
pub(crate) fn merge_result_from_pairs(
    points: &[Vec3],
    pairs: &[(u32, u32)],
) -> (MergeResult, Vec<bool>) {
    let n = points.len();
    let mut dsu = super::union_find::SparseUnionFind::new();
    for &(a, b) in pairs {
        let _ = dsu.union_keep_min(a, b);
    }

    // Only ids that appeared in a successful union can differ from their own
    // representative. Untouched ids map to themselves without a hash lookup.
    let touched_reps: Vec<(usize, usize)> = dsu
        .touched_ids()
        .into_iter()
        .map(|id| {
            let rep = dsu.find(id) as usize;
            (id as usize, rep)
        })
        .collect();
    // Extract representatives, preserving original-index order. Welds are
    // sparse, so `touched_reps` is tiny and the index space is dominated by
    // long untouched runs (every id maps to itself). Copy those runs in bulk
    // (`extend_from_slice` memcpy + a counter fill) instead of a per-element
    // push gated by a peek, and only branch at the few touched ids.
    let mut effective_points: Vec<Vec3> = Vec::with_capacity(n);
    let mut original_to_effective = vec![0u32; n];
    let mut kept = vec![true; n];

    let fill_identity_run = |effective_points: &mut Vec<Vec3>,
                             original_to_effective: &mut [u32],
                             run: std::ops::Range<usize>| {
        let base = effective_points.len() as u32;
        effective_points.extend_from_slice(&points[run.clone()]);
        for (k, oe) in original_to_effective[run].iter_mut().enumerate() {
            *oe = base + k as u32;
        }
    };

    let mut prev = 0usize;
    for (id, rep) in touched_reps {
        if id > prev {
            fill_identity_run(&mut effective_points, &mut original_to_effective, prev..id);
        }
        // `rep` is the class min and was assigned in an earlier run/iteration.
        if rep == id {
            original_to_effective[id] = effective_points.len() as u32;
            effective_points.push(points[id]);
        } else {
            kept[id] = false;
            original_to_effective[id] = original_to_effective[rep];
        }
        prev = id + 1;
    }
    if prev < n {
        fill_identity_run(&mut effective_points, &mut original_to_effective, prev..n);
    }

    let num_merged = n - effective_points.len();

    (
        MergeResult {
            effective_points,
            original_to_effective,
            num_merged,
        },
        kept,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeSet;

    fn unit(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize()
    }

    fn brute_force_pairs(points: &[Vec3], threshold: f32) -> BTreeSet<(u32, u32)> {
        let radius_squared = threshold * threshold;
        let mut pairs = BTreeSet::new();
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let distance_squared = (points[i] - points[j]).length_squared();
                if distance_squared < radius_squared {
                    pairs.insert((i as u32, j as u32));
                }
            }
        }
        pairs
    }

    fn standalone_pair_set(points: &[Vec3], threshold: f32) -> BTreeSet<(u32, u32)> {
        try_collect_close_pairs(points, threshold)
            .unwrap()
            .into_iter()
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect()
    }

    fn random_unit(rng: &mut ChaCha8Rng) -> Vec3 {
        loop {
            let p = Vec3::new(
                rng.gen_range(-1.0f32..1.0),
                rng.gen_range(-1.0f32..1.0),
                rng.gen_range(-1.0f32..1.0),
            );
            if p.length_squared() > 1e-6 {
                return p.normalize();
            }
        }
    }

    #[test]
    fn weld_radius_comparison_is_strict_at_adjacent_values() {
        let distance = 0.5f32;
        let points = [Vec3::ZERO, Vec3::new(distance, 0.0, 0.0)];
        assert!(standalone_pair_set(&points, distance.next_down()).is_empty());
        assert!(standalone_pair_set(&points, distance).is_empty());
        assert_eq!(
            standalone_pair_set(&points, distance.next_up()),
            BTreeSet::from([(0, 1)])
        );
    }

    #[test]
    fn standalone_detector_matches_computed_f32_oracle() {
        for threshold in [crate::tolerances::weld_radius(), 1e-3f32] {
            for seed in [1u64, 7, 42] {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let mut points: Vec<Vec3> = (0..240).map(|_| random_unit(&mut rng)).collect();
                for i in 0..30 {
                    let base = points[i * 7];
                    let direction = random_unit(&mut rng);
                    points.push((base + direction * (0.4 * threshold)).normalize());
                    points.push((base + direction * (1.4 * threshold)).normalize());
                }

                let expected = brute_force_pairs(&points, threshold);
                assert!(!expected.is_empty(), "fixture has no pairs");
                assert_eq!(
                    standalone_pair_set(&points, threshold),
                    expected,
                    "threshold={threshold:e} seed={seed}"
                );
            }
        }
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
        assert_eq!(
            result.effective_points[result.original_to_effective[0] as usize],
            p
        );
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
    fn long_weld_chain_uses_minimum_index_without_a_diameter_bound() {
        let threshold = 1e-3f32;
        let step = 0.75 * threshold;
        let mut points = vec![Vec3::new(-0.5, 0.5, 0.5)];
        // Put an interior chain point first: the representative policy is
        // original index, not a geometric endpoint or centroid choice.
        points.push(Vec3::new(5.0 * step, 0.5, 0.5));
        points.extend((0..12).map(|i| Vec3::new(i as f32 * step, 0.5, 0.5)));

        let result = merge_close_points(&points, threshold);
        assert_eq!(result.num_merged, 12);
        let representative = result.original_to_effective[1];
        for &mapped in &result.original_to_effective[1..] {
            assert_eq!(mapped, representative);
        }
        assert_eq!(result.effective_points[representative as usize], points[1]);
        assert!((points[2] - points[13]).length() > 8.0 * threshold);
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
