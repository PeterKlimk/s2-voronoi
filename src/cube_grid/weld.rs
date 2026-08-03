//! Grid-integrated weld detection and in-place compaction.
//!
//! Replaces the standalone quantized-key weld pass on the sphere: the query
//! grid is built on the raw points, sub-threshold pairs are detected with a
//! per-cell scan (the grid already groups near-coincident points), and on
//! welds the grid's point arrays are compacted in place instead of
//! rebuilding. The zero-weld common case pays only the detection scan.

use glam::Vec3;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{cell_to_face_ij, CubeMapGrid};
use crate::tolerances::GRID_WELD_WALL_ABS_PAD;

pub(crate) const MAX_RETAINED_WELD_PAIRS: usize = 1 << 20;

#[derive(Clone, Copy)]
struct SlotPointInit {
    cell_points_aos_addr: usize,
}

/// The exact computed-f32 edge predicate for the welding threshold graph.
#[inline(always)]
pub(crate) fn is_weld_pair(distance_squared: f32, radius_squared: f32) -> bool {
    distance_squared < radius_squared
}

impl CubeMapGrid {
    /// Conservative upper bound on a weld threshold detectable through 3x3
    /// cell adjacency: a sub-threshold pair must never span non-adjacent
    /// cells. Adjacent parallel wall planes are at least ~2/(3*res) apart
    /// (st spacing 1/res, d(uv)/d(st) >= 4/3, d(angle)/d(uv) >= 1/2), so
    /// 1/(16*res) keeps a ~10x margin. Thresholds above this must use the
    /// standalone detector (`merge_close_points`).
    pub(crate) fn max_grid_weld_threshold(&self) -> f32 {
        1.0 / (16.0 * self.res as f32)
    }

    /// Collect every point pair within `threshold` (Euclidean chord,
    /// strict `<` to match the standalone detector), as `(min, max)` global
    /// point-index pairs, duplicate-free.
    ///
    /// Coverage argument: a pair in the same cell is found by the pairwise
    /// scan. A pair split across cells has both endpoints within
    /// `threshold` of the separating wall plane (|n.p| <= |p - q| for p, q
    /// on opposite sides of plane n), so the endpoint in the lower-indexed
    /// cell is wall-flagged and scans its full 3x3 neighborhood — which
    /// contains the partner cell because `threshold` is far below the wall
    /// spacing (`max_grid_weld_threshold`). The `nc > cell` gate makes
    /// exactly one side of each cell pair do the scan.
    #[cfg(test)]
    pub(crate) fn collect_weld_pairs(&self, threshold: f32) -> Result<Vec<(u32, u32)>, usize> {
        self.collect_weld_pairs_impl::<false>(
            threshold,
            SlotPointInit {
                cell_points_aos_addr: 0,
            },
        )
    }

    /// Collect weld pairs while initializing the retained grid's slot-ordered
    /// point stream. Each cell owns a disjoint slot range, so the existing
    /// cell-parallel traversal can populate the destination without
    /// synchronization. The global-id-to-slot inverse remains absent unless
    /// actual weld compaction needs it.
    pub(crate) fn collect_weld_pairs_and_finalize_slot_points(
        &mut self,
        threshold: f32,
    ) -> Result<Vec<(u32, u32)>, usize> {
        assert!(
            self.cell_points_aos.is_empty(),
            "slot-point view already initialized"
        );
        let n = self.point_indices.len();
        let mut cell_points_aos = Vec::<super::SlotPoint>::with_capacity(n);
        let init = SlotPointInit {
            cell_points_aos_addr: cell_points_aos.spare_capacity_mut().as_mut_ptr() as usize,
        };
        let pairs = self.collect_weld_pairs_impl::<true>(threshold, init)?;
        // SAFETY: a successful traversal visits every nonempty cell, whose
        // slot ranges partition `0..n`, so `scan_cell` initialized every
        // element exactly once. On an early pair-budget error, the length
        // remains zero.
        unsafe {
            cell_points_aos.set_len(n);
        }
        self.cell_points_aos = cell_points_aos;
        Ok(pairs)
    }

    /// Initialize deferred slot-ordered points without weld detection. Used
    /// by the standalone large-radius `MergeWithin` path after it selects the
    /// grid that will be retained.
    pub(crate) fn finalize_slot_points(&mut self) {
        assert!(
            self.cell_points_aos.is_empty()
                || self.cell_points_aos.len() == self.point_indices.len(),
            "slot-point stream is partially initialized"
        );
        if !self.cell_points_aos.is_empty() {
            return;
        }
        self.cell_points_aos = super::build::build_pos_aos(
            &self.cell_points_x,
            &self.cell_points_y,
            &self.cell_points_z,
            &self.point_indices,
        );
    }

    fn collect_weld_pairs_impl<const FINALIZE_SLOT_POINTS: bool>(
        &self,
        threshold: f32,
        slot_point_init: SlotPointInit,
    ) -> Result<Vec<(u32, u32)>, usize> {
        debug_assert!(
            threshold <= self.max_grid_weld_threshold(),
            "weld threshold {} exceeds grid adjacency bound {}",
            threshold,
            self.max_grid_weld_threshold()
        );
        let num_cells = 6 * self.res * self.res;
        let thr_sq = threshold * threshold;
        // Wall-proximity pad: threshold plus a generous absolute guard for
        // the f32 plane-dot error (a few ulps of 1.0). False positives only
        // cost a redundant neighbor scan.
        let pad = threshold + GRID_WELD_WALL_ABS_PAD;
        let line_count = self.res + 1;
        let retained = AtomicUsize::new(0);
        let exceeded = AtomicBool::new(false);

        let scan_cell = |cell: usize, out: &mut Vec<(u32, u32)>| {
            if exceeded.load(Relaxed) {
                return;
            }
            let start = self.cell_offsets[cell] as usize;
            let end = self.cell_offsets[cell + 1] as usize;
            if start == end {
                return;
            }
            let (face, iu, iv) = cell_to_face_ij(cell, self.res);

            let push = |out: &mut Vec<(u32, u32)>, a: u32, b: u32| -> bool {
                if retained
                    .fetch_update(Relaxed, Relaxed, |n| {
                        (n < MAX_RETAINED_WELD_PAIRS).then_some(n + 1)
                    })
                    .is_err()
                {
                    exceeded.store(true, Relaxed);
                    return false;
                }
                out.push((a.min(b), a.max(b)));
                true
            };

            // Same-cell pairs.
            // Test one face-tangent component first. If its square alone is
            // not below the strict threshold, the non-negative three-term
            // squared distance cannot be below it either. Almost every normal
            // input pair exits here after loading one coordinate instead of
            // all three.
            let gate_points = match face {
                0 | 1 => &self.cell_points_z,
                _ => &self.cell_points_x,
            };
            let cell_points_aos = slot_point_init.cell_points_aos_addr as *mut super::SlotPoint;
            for i in start..end {
                let gate_i = if FINALIZE_SLOT_POINTS {
                    let global = self.point_indices[i];
                    let x = self.cell_points_x[i];
                    let y = self.cell_points_y[i];
                    let z = self.cell_points_z[i];
                    // SAFETY: cell ranges are disjoint across workers and
                    // together partition the slot-ordered destination.
                    unsafe {
                        cell_points_aos.add(i).write(super::SlotPoint {
                            pos: Vec3::new(x, y, z),
                            idx: global,
                        });
                    }
                    if matches!(face, 0 | 1) {
                        z
                    } else {
                        x
                    }
                } else {
                    gate_points[i]
                };
                let candidate_start = i + 1;
                let (candidate_chunks, candidate_tail) =
                    gate_points[candidate_start..end].as_chunks::<8>();

                for (chunk_idx, candidates) in candidate_chunks.iter().enumerate() {
                    let mut mask = crate::fp::squared_deltas_mask_lt8(candidates, gate_i, thr_sq);
                    while mask != 0 {
                        let lane = mask.trailing_zeros() as usize;
                        let j = candidate_start + chunk_idx * 8 + lane;
                        let dx = self.cell_points_x[i] - self.cell_points_x[j];
                        let dy = self.cell_points_y[i] - self.cell_points_y[j];
                        let dz = self.cell_points_z[i] - self.cell_points_z[j];
                        if is_weld_pair(dx * dx + dy * dy + dz * dz, thr_sq)
                            && !push(out, self.point_indices[i], self.point_indices[j])
                        {
                            return;
                        }
                        mask &= mask - 1;
                    }
                }

                let tail_start = end - candidate_tail.len();
                for (tail_offset, &gate_j) in gate_points[tail_start..end].iter().enumerate() {
                    let j = tail_start + tail_offset;
                    let gate_delta = gate_i - gate_j;
                    if !is_weld_pair(gate_delta * gate_delta, thr_sq) {
                        continue;
                    }
                    let dx = self.cell_points_x[i] - self.cell_points_x[j];
                    let dy = self.cell_points_y[i] - self.cell_points_y[j];
                    let dz = self.cell_points_z[i] - self.cell_points_z[j];
                    if is_weld_pair(dx * dx + dy * dy + dz * dz, thr_sq)
                        && !push(out, self.point_indices[i], self.point_indices[j])
                    {
                        return;
                    }
                }
            }

            // Cross-cell pairs: only points within `pad` of a wall plane can
            // have a partner in another cell (essentially never on real
            // input — this loop's body is cold).
            let walls = [
                self.u_line_planes[face * line_count + iu],
                self.u_line_planes[face * line_count + iu + 1],
                self.v_line_planes[face * line_count + iv],
                self.v_line_planes[face * line_count + iv + 1],
            ];
            for i in start..end {
                let p = Vec3::new(
                    self.cell_points_x[i],
                    self.cell_points_y[i],
                    self.cell_points_z[i],
                );
                if !walls.iter().any(|n| n.dot(p).abs() < pad) {
                    continue;
                }
                for &nc in &self.neighbors[cell] {
                    // Skips the u32::MAX padding and the center entry, and
                    // gates each unordered cell pair to one scanning side.
                    if nc == u32::MAX || (nc as usize) <= cell {
                        continue;
                    }
                    let ns = self.cell_offsets[nc as usize] as usize;
                    let ne = self.cell_offsets[nc as usize + 1] as usize;
                    for j in ns..ne {
                        let dx = p.x - self.cell_points_x[j];
                        let dy = p.y - self.cell_points_y[j];
                        let dz = p.z - self.cell_points_z[j];
                        if is_weld_pair(dx * dx + dy * dy + dz * dz, thr_sq)
                            && !push(out, self.point_indices[i], self.point_indices[j])
                        {
                            return;
                        }
                    }
                }
            }
        };

        #[cfg(feature = "parallel")]
        {
            const CHUNK: usize = 1 << 10;
            let chunk_pairs: Vec<Vec<(u32, u32)>> = (0..num_cells.div_ceil(CHUNK))
                .into_par_iter()
                .map(|c| {
                    let mut local = Vec::new();
                    for cell in c * CHUNK..((c + 1) * CHUNK).min(num_cells) {
                        scan_cell(cell, &mut local);
                    }
                    local
                })
                .collect();
            if exceeded.load(Relaxed) {
                Err(MAX_RETAINED_WELD_PAIRS + 1)
            } else {
                Ok(chunk_pairs.into_iter().flatten().collect())
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut pairs = Vec::new();
            for cell in 0..num_cells {
                scan_cell(cell, &mut pairs);
            }
            if exceeded.load(Relaxed) {
                Err(MAX_RETAINED_WELD_PAIRS + 1)
            } else {
                Ok(pairs)
            }
        }
    }

    /// Remove welded-away points and remap survivors to effective indices,
    /// in place. `kept[orig]` says whether original point `orig` survives
    /// (is its weld-class representative); `original_to_effective[orig]`
    /// gives its effective index; `n_eff` is the effective point count.
    ///
    /// Only the point-dependent arrays change (offsets, indices, SoA
    /// coordinates, AoS positions, and per-point cells); the per-cell
    /// geometry depends only on `res`. Survivors keep their relative (cell,
    /// slot) order, so the slot streams and effective-id maps are rebuilt in
    /// one forward pass without random source reads. The result is bit-identical
    /// to a fresh build on the effective points at the same resolution (pinned
    /// by a test below).
    pub(crate) fn compact_welded(
        &mut self,
        kept: &[bool],
        original_to_effective: &[u32],
        n_eff: usize,
    ) {
        let num_cells = 6 * self.res * self.res;

        assert_eq!(
            self.cell_points_aos.len(),
            self.point_indices.len(),
            "slot-point stream must be initialized before compaction"
        );
        assert_eq!(self.point_cells.len(), kept.len());
        // One slot-order pass compacts the SoA + AoS streams and directly
        // rebuilds the effective-id-to-cell map. The outer cell and forward write
        // cursor already are the surviving point's final cell and slot, so no
        // dropped-slot list or original-id follow-up pass is needed. Writes to
        // effective-id map entries cannot disturb the loop: neither map is a
        // source for this pass.
        let mut w = 0usize;
        let mut read_start = 0usize;
        for cell in 0..num_cells {
            let read_end = self.cell_offsets[cell + 1] as usize;
            for r in read_start..read_end {
                let orig = self.point_indices[r] as usize;
                if !kept[orig] {
                    continue;
                }
                let eff = original_to_effective[orig];
                let eff_usize = eff as usize;
                let (x, y, z) = (
                    self.cell_points_x[r],
                    self.cell_points_y[r],
                    self.cell_points_z[r],
                );
                self.point_indices[w] = eff;
                self.cell_points_x[w] = x;
                self.cell_points_y[w] = y;
                self.cell_points_z[w] = z;
                self.cell_points_aos[w] = super::SlotPoint {
                    pos: Vec3::new(x, y, z),
                    idx: eff,
                };
                self.point_cells[eff_usize] = cell as u32;
                w += 1;
            }
            self.cell_offsets[cell + 1] = w as u32;
            read_start = read_end;
        }
        debug_assert_eq!(w, n_eff, "compaction kept-count mismatch");
        self.point_indices.truncate(w);
        self.cell_points_x.truncate(w);
        self.cell_points_y.truncate(w);
        self.cell_points_z.truncate(w);
        self.cell_points_aos.truncate(w);
        self.point_cells.truncate(n_eff);

        // The dense-cell side index is keyed to slot order and cell ranges,
        // both of which compaction just rewrote. Rebuild only when this grid
        // already materialized the optional index; provisional compute grids
        // deliberately defer it until preprocessing selects the retained
        // layout. Compaction cannot increase a cell's occupancy, so an absent
        // index cannot become newly necessary here.
        if self.dense_index.is_some() {
            self.dense_index = super::dense::DenseCellIndex::build(
                &self.cell_offsets,
                &self.cell_points_x,
                &self.cell_points_y,
                &self.cell_points_z,
                crate::policy::DENSE_CELL_THRESHOLD,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn random_unit(rng: &mut ChaCha8Rng) -> Vec3 {
        loop {
            let v = Vec3::new(
                rng.gen_range(-1.0f32..1.0),
                rng.gen_range(-1.0f32..1.0),
                rng.gen_range(-1.0f32..1.0),
            );
            let len = v.length();
            if len > 1e-3 && len < 1.0 {
                return v / len;
            }
        }
    }

    /// Points with planted sub-threshold twins (offset in a random
    /// direction, so a fraction straddle cell walls at higher res).
    fn points_with_twins(n_base: usize, n_twins: usize, offset: f32, seed: u64) -> Vec<Vec3> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut pts: Vec<Vec3> = (0..n_base).map(|_| random_unit(&mut rng)).collect();
        for t in 0..n_twins {
            let base = pts[t * (n_base / n_twins.max(1)).max(1) % n_base];
            let dir = random_unit(&mut rng);
            pts.push((base + dir * offset).normalize());
        }
        pts
    }

    fn brute_force_pairs(
        points: &[Vec3],
        threshold: f32,
    ) -> std::collections::BTreeSet<(u32, u32)> {
        let thr_sq = threshold * threshold;
        let mut out = std::collections::BTreeSet::new();
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                if (points[i] - points[j]).length_squared() < thr_sq {
                    out.insert((i as u32, j as u32));
                }
            }
        }
        out
    }

    #[test]
    fn squared_weld_predicate_is_strict_at_adjacent_values() {
        let radius_squared = 0.25f32;
        assert!(is_weld_pair(radius_squared.next_down(), radius_squared));
        assert!(!is_weld_pair(radius_squared, radius_squared));
        assert!(!is_weld_pair(radius_squared.next_up(), radius_squared));
    }

    /// Grid detection must match brute force exactly, across resolutions
    /// (including res where twins straddle walls) and seeds.
    #[test]
    fn grid_pairs_match_brute_force() {
        let threshold = crate::tolerances::weld_radius();
        for seed in [1u64, 7, 42] {
            // Sub-threshold twins (offset below) and above-threshold near
            // misses (offset just above) in one input.
            let mut points = points_with_twins(400, 40, threshold * 0.5, seed);
            points.extend(points_with_twins(50, 10, threshold * 1.5, seed + 100));
            let expected = brute_force_pairs(&points, threshold);
            assert!(!expected.is_empty(), "fixture must contain pairs");
            for res in [1usize, 2, 4, 13, 64] {
                let grid = CubeMapGrid::new(&points, res);
                let mut got: Vec<(u32, u32)> = grid.collect_weld_pairs(threshold).unwrap();
                got.sort_unstable();
                got.dedup();
                assert_eq!(
                    got.iter()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>(),
                    expected,
                    "pair set mismatch at res={res} seed={seed}"
                );
            }
        }
    }

    #[test]
    fn fused_weld_scan_finalizes_exact_slot_points() {
        let threshold = crate::tolerances::weld_radius();
        let points = points_with_twins(400, 40, threshold * 0.5, 91);
        let expected = CubeMapGrid::new(&points, 13);

        #[cfg(feature = "timing")]
        let mut fused = {
            let mut timings = crate::cube_grid::CubeMapGridBuildTimings::default();
            CubeMapGrid::new_deferred_dense_and_point_views_with_build_timings(
                &points,
                13,
                &mut timings,
            )
        };
        #[cfg(not(feature = "timing"))]
        let mut fused = CubeMapGrid::new_deferred_dense_and_point_views(&points, 13);

        assert!(fused.cell_points_aos.is_empty());
        let mut expected_pairs = expected.collect_weld_pairs(threshold).unwrap();
        let mut fused_pairs = fused
            .collect_weld_pairs_and_finalize_slot_points(threshold)
            .unwrap();
        expected_pairs.sort_unstable();
        fused_pairs.sort_unstable();

        assert_eq!(fused_pairs, expected_pairs);
        assert_eq!(fused.cell_points_aos, expected.cell_points_aos);
    }

    /// Exact-duplicate pairs straddling nothing (same coordinates) must be
    /// found at any resolution — the degenerate same-cell case.
    #[test]
    fn grid_pairs_find_exact_duplicates() {
        let p = Vec3::new(0.3f32, -0.7, 0.2).normalize();
        let points = vec![Vec3::new(1.0, 0.0, 0.0), p, Vec3::new(0.0, 1.0, 0.0), p];
        for res in [1usize, 8] {
            let grid = CubeMapGrid::new(&points, res);
            assert_eq!(
                grid.collect_weld_pairs(crate::tolerances::weld_radius())
                    .unwrap(),
                vec![(1, 3)]
            );
        }
    }

    #[test]
    fn dense_duplicate_cluster_stops_at_pair_budget() {
        let points = vec![Vec3::Z; 1_450];
        let grid = CubeMapGrid::new(&points, 4);
        assert_eq!(
            grid.collect_weld_pairs(crate::tolerances::weld_radius()),
            Err(MAX_RETAINED_WELD_PAIRS + 1)
        );
    }

    /// Compaction must produce a grid bit-identical to a fresh build on the
    /// effective points at the same resolution.
    #[test]
    fn compacted_grid_matches_fresh_build() {
        let threshold = crate::tolerances::weld_radius();
        for seed in [3u64, 11] {
            let points = points_with_twins(300, 30, threshold * 0.4, seed);
            for res in [2usize, 9, 33] {
                let mut grid = CubeMapGrid::new(&points, res);
                let pairs = grid.collect_weld_pairs(threshold).unwrap();
                assert!(!pairs.is_empty());
                let (result, kept) =
                    crate::knn_clipping::preprocess::merge_result_from_pairs(&points, &pairs);
                grid.compact_welded(
                    &kept,
                    &result.original_to_effective,
                    result.effective_points.len(),
                );

                let fresh = CubeMapGrid::new(&result.effective_points, res);
                assert_eq!(
                    grid.cell_offsets, fresh.cell_offsets,
                    "res={res} seed={seed}"
                );
                assert_eq!(
                    grid.point_indices, fresh.point_indices,
                    "res={res} seed={seed}"
                );
                assert_eq!(grid.point_cells, fresh.point_cells, "res={res} seed={seed}");
                assert_eq!(
                    grid.cell_points_x, fresh.cell_points_x,
                    "res={res} seed={seed}"
                );
                assert_eq!(
                    grid.cell_points_y, fresh.cell_points_y,
                    "res={res} seed={seed}"
                );
                assert_eq!(
                    grid.cell_points_z, fresh.cell_points_z,
                    "res={res} seed={seed}"
                );
                // The slot-ordered AoS positions must also match a fresh build
                // (compact_welded rebuilds them; pins the sync invariant).
                assert_eq!(
                    grid.cell_points_aos, fresh.cell_points_aos,
                    "res={res} seed={seed}"
                );
            }
        }
    }
}
