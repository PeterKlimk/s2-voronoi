//! Grid-integrated weld detection and in-place compaction.
//!
//! Replaces the standalone quantized-key weld pass on the sphere: the query
//! grid is built on the raw points, sub-threshold pairs are detected with a
//! per-cell scan (the grid already groups near-coincident points), and on
//! welds the grid's point arrays are compacted in place instead of
//! rebuilding. The zero-weld common case pays only the detection scan.

use glam::Vec3;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{cell_to_face_ij, CubeMapGrid};

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
    pub(crate) fn collect_weld_pairs(&self, threshold: f32) -> Vec<(u32, u32)> {
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
        let pad = threshold + 1e-6;
        let line_count = self.res + 1;

        let scan_cell = |cell: usize, out: &mut Vec<(u32, u32)>| {
            let start = self.cell_offsets[cell] as usize;
            let end = self.cell_offsets[cell + 1] as usize;
            if start == end {
                return;
            }

            let push = |out: &mut Vec<(u32, u32)>, a: u32, b: u32| {
                out.push((a.min(b), a.max(b)));
            };

            // Same-cell pairs.
            for i in start..end {
                let (xi, yi, zi) = (
                    self.cell_points_x[i],
                    self.cell_points_y[i],
                    self.cell_points_z[i],
                );
                for j in (i + 1)..end {
                    let dx = xi - self.cell_points_x[j];
                    let dy = yi - self.cell_points_y[j];
                    let dz = zi - self.cell_points_z[j];
                    if dx * dx + dy * dy + dz * dz < thr_sq {
                        push(out, self.point_indices[i], self.point_indices[j]);
                    }
                }
            }

            // Cross-cell pairs: only points within `pad` of a wall plane can
            // have a partner in another cell (essentially never on real
            // input — this loop's body is cold).
            let (face, iu, iv) = cell_to_face_ij(cell, self.res);
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
                        if dx * dx + dy * dy + dz * dz < thr_sq {
                            push(out, self.point_indices[i], self.point_indices[j]);
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
            chunk_pairs.into_iter().flatten().collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut pairs = Vec::new();
            for cell in 0..num_cells {
                scan_cell(cell, &mut pairs);
            }
            pairs
        }
    }

    /// Remove welded-away points and remap survivors to effective indices,
    /// in place. `kept[orig]` says whether original point `orig` survives
    /// (is its weld-class representative); `original_to_effective[orig]`
    /// gives its effective index; `n_eff` is the effective point count.
    ///
    /// Only the point-dependent arrays change (offsets, indices, SoA
    /// coordinates, AoS positions, per-point cells/slots); the per-cell
    /// geometry depends only on `res`. Survivors keep their relative (cell,
    /// slot) order, so every array compacts with a single forward pass over its
    /// own buffer — no reallocation and no random scatter. The result is
    /// bit-identical to a fresh build on the effective points at the same
    /// resolution (pinned by a test below).
    pub(crate) fn compact_welded(
        &mut self,
        kept: &[bool],
        original_to_effective: &[usize],
        n_eff: usize,
    ) {
        let num_cells = 6 * self.res * self.res;

        // Slot-order stream compaction of the SoA + AoS arrays. A forward write
        // cursor `w` packs survivors densely and remaps each stored index to its
        // effective id; coordinates and the slot-ordered AoS are overwritten in
        // place (`w <= r`, so the source slot `r` is never one already written).
        // The dropped slots are recorded in ascending order for the slot
        // renumber below.
        let mut dropped_slots: Vec<u32> = Vec::new();
        let mut w = 0usize;
        let mut read_start = 0usize;
        for cell in 0..num_cells {
            let read_end = self.cell_offsets[cell + 1] as usize;
            for r in read_start..read_end {
                let orig = self.point_indices[r] as usize;
                if !kept[orig] {
                    dropped_slots.push(r as u32);
                    continue;
                }
                let eff = original_to_effective[orig] as u32;
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

        // Per-point arrays (indexed by point id). Cell membership is invariant
        // under welding, so `point_cells` compacts by original id in place. A
        // survivor's new slot is its old slot minus the dropped points ahead of
        // it — a `partition_point` into the ascending `dropped_slots` (a single
        // comparison in the common one-weld case). Both reuse their buffers; the
        // effective id `e` never overtakes the read cursor `orig`, so neither
        // pass reads a slot it has already overwritten.
        let mut e = 0usize;
        for (orig, &keep) in kept.iter().enumerate() {
            if !keep {
                continue;
            }
            let old_slot = self.point_slots[orig] as usize;
            let drops_before = dropped_slots.partition_point(|&s| (s as usize) < old_slot);
            self.point_cells[e] = self.point_cells[orig];
            self.point_slots[e] = (old_slot - drops_before) as u32;
            e += 1;
        }
        debug_assert_eq!(e, n_eff, "per-point compaction kept-count mismatch");
        self.point_cells.truncate(n_eff);
        self.point_slots.truncate(n_eff);

        // The dense-cell side index is keyed to slot order and cell ranges,
        // both of which compaction just rewrote — rebuild it from the compacted
        // arrays (matches a fresh build at this resolution).
        self.dense_index = super::dense::DenseCellIndex::build(
            &self.cell_offsets,
            &self.cell_points_x,
            &self.cell_points_y,
            &self.cell_points_z,
            crate::policy::DENSE_CELL_THRESHOLD,
        );
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
                let mut got: Vec<(u32, u32)> = grid.collect_weld_pairs(threshold);
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

    /// Exact-duplicate pairs straddling nothing (same coordinates) must be
    /// found at any resolution — the degenerate same-cell case.
    #[test]
    fn grid_pairs_find_exact_duplicates() {
        let p = Vec3::new(0.3f32, -0.7, 0.2).normalize();
        let points = vec![Vec3::new(1.0, 0.0, 0.0), p, Vec3::new(0.0, 1.0, 0.0), p];
        for res in [1usize, 8] {
            let grid = CubeMapGrid::new(&points, res);
            assert_eq!(
                grid.collect_weld_pairs(crate::tolerances::weld_radius()),
                vec![(1, 3)]
            );
        }
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
                let pairs = grid.collect_weld_pairs(threshold);
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
                assert_eq!(grid.point_slots, fresh.point_slots, "res={res} seed={seed}");
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
