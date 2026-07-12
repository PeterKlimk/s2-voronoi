//! Dense-cell sub-index ("punch 1", axis-sort variant). A SIDE structure: it
//! never permutes the grid SoA (slot order is load-bearing for binning
//! local-ids), it only stores, per over-full cell, that cell's slots sorted by
//! a dominant axis so a query can band-prune instead of scanning all `occ`
//! points.
//!
//! Built only for cells with `occ > DENSE_CELL_THRESHOLD` (rare), so it is
//! empty on uniform input and the fast path never touches it.
//!
//! The packed center-cell path consults this via a conservative band query and
//! reports coverage only down to the band's dot bound; shell takeover backstops
//! anything below that bound.

use std::collections::HashMap;

/// One dense cell's axis-sorted index.
struct DenseCellEntry {
    /// Dominant axis (0=x, 1=y, 2=z) — the one with the largest coordinate
    /// spread in this cell, so the band prunes the most.
    axis: u8,
    /// The cell's slots sorted ascending by their `axis` coordinate.
    sorted_slots: Vec<u32>,
    /// The corresponding sorted axis coordinates (parallel to `sorted_slots`),
    /// held separately for branch-light binary search.
    sorted_coord: Vec<f32>,
    /// Euclidean (chord) diagonal of the cell's coordinate bounding box: a
    /// cheap, safe upper bound on the cell's spatial extent, used to size the
    /// band radius for a target nearest-neighbor count from local density.
    diag: f32,
}

/// Side index over a grid's over-full cells. Empty when no cell is dense.
pub(crate) struct DenseCellIndex {
    cells: HashMap<u32, DenseCellEntry>,
}

impl DenseCellIndex {
    /// Build the index for every cell whose occupancy exceeds `threshold`.
    /// Returns `None` when no cell qualifies (the overwhelmingly common case),
    /// so the grid stores `Option<DenseCellIndex>` and pays nothing otherwise.
    pub(crate) fn build(
        cell_offsets: &[u32],
        cell_points_x: &[f32],
        cell_points_y: &[f32],
        cell_points_z: &[f32],
        threshold: usize,
    ) -> Option<DenseCellIndex> {
        let mut cells: HashMap<u32, DenseCellEntry> = HashMap::new();
        for cell in 0..cell_offsets.len().saturating_sub(1) {
            let start = cell_offsets[cell] as usize;
            let end = cell_offsets[cell + 1] as usize;
            if end - start <= threshold {
                continue;
            }
            let xs = &cell_points_x[start..end];
            let ys = &cell_points_y[start..end];
            let zs = &cell_points_z[start..end];

            // Dominant axis = largest coordinate range over the cell's points;
            // bounding-box diagonal = a safe extent for sizing the band radius.
            let (rx, ry, rz) = axis_ranges(xs, ys, zs);
            let axis = if rx >= ry && rx >= rz {
                0
            } else if ry >= rz {
                1
            } else {
                2
            };
            let diag = (rx * rx + ry * ry + rz * rz).sqrt();
            let coords = match axis {
                0 => xs,
                1 => ys,
                _ => zs,
            };

            let mut order: Vec<u32> = (0..(end - start) as u32).collect();
            order.sort_unstable_by(|&a, &b| {
                coords[a as usize]
                    .partial_cmp(&coords[b as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sorted_slots = Vec::with_capacity(order.len());
            let mut sorted_coord = Vec::with_capacity(order.len());
            for &i in &order {
                sorted_slots.push(start as u32 + i);
                sorted_coord.push(coords[i as usize]);
            }

            cells.insert(
                cell as u32,
                DenseCellEntry {
                    axis,
                    sorted_slots,
                    sorted_coord,
                    diag,
                },
            );
        }
        if cells.is_empty() {
            None
        } else {
            Some(DenseCellIndex { cells })
        }
    }

    /// Whether `cell` has a sub-index. The production query path gates via
    /// [`band_radius`](Self::band_radius) (which returns `None` when absent),
    /// so this is a test-only confirmation helper.
    #[cfg(test)]
    pub(crate) fn has(&self, cell: u32) -> bool {
        self.cells.contains_key(&cell)
    }

    /// Suggested band gather radius (Euclidean chord) for `cell` that captures
    /// roughly `target_count` nearest neighbors under a uniform-local-density
    /// model: `r ≈ (diag/2)·sqrt(target/occ)`. `None` when `cell` is not
    /// indexed. The radius is cell-level (independent of the query); the caller
    /// derives the completeness bound `1 - r²/2` and gathers a slightly wider
    /// band to absorb fp error (see the band-prune integration design doc).
    pub(crate) fn band_radius(&self, cell: u32, target_count: usize) -> Option<f32> {
        let entry = self.cells.get(&cell)?;
        let occ = entry.sorted_slots.len();
        if occ == 0 {
            return None;
        }
        let frac = (target_count as f32 / occ as f32).sqrt();
        Some(0.5 * entry.diag * frac)
    }

    /// Slots of `cell` whose `axis` coordinate lies within `radius` of the
    /// query's coordinate `(qx, qy, qz)` — the band that can contain any point
    /// within Euclidean `radius` of the query (a superset; the caller still
    /// distance-filters). Returns nothing if `cell` is not indexed.
    ///
    /// This is the certificate-safe prune: every point within `radius` is in
    /// the band, so no true neighbor is missed (vs a fixed-K cap). The band is
    /// found by two binary searches on the sorted axis coordinates.
    pub(crate) fn band_slots(
        &self,
        cell: u32,
        qx: f32,
        qy: f32,
        qz: f32,
        radius: f32,
        out: &mut Vec<u32>,
    ) {
        out.clear();
        let Some(entry) = self.cells.get(&cell) else {
            return;
        };
        let q = match entry.axis {
            0 => qx,
            1 => qy,
            _ => qz,
        };
        let lo = q - radius;
        let hi = q + radius;
        let from = entry.sorted_coord.partition_point(|&c| c < lo);
        let to = entry.sorted_coord.partition_point(|&c| c <= hi);
        out.extend_from_slice(&entry.sorted_slots[from..to]);
    }
}

/// Per-axis coordinate ranges (max - min) over the cell's points.
fn axis_ranges(xs: &[f32], ys: &[f32], zs: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(xs.len(), ys.len());
    debug_assert_eq!(xs.len(), zs.len());

    let (mut x_lo, mut y_lo, mut z_lo) = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let (mut x_hi, mut y_hi, mut z_hi) = (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for ((&x, &y), &z) in xs.iter().zip(ys).zip(zs) {
        x_lo = x_lo.min(x);
        x_hi = x_hi.max(x);
        y_lo = y_lo.min(y);
        y_hi = y_hi.max(y);
        z_lo = z_lo.min(z);
        z_hi = z_hi.max(z);
    }
    (x_hi - x_lo, y_hi - y_lo, z_hi - z_lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline_axis_ranges(xs: &[f32], ys: &[f32], zs: &[f32]) -> (f32, f32, f32) {
        let range = |s: &[f32]| -> f32 {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &v in s {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            hi - lo
        };
        (range(xs), range(ys), range(zs))
    }

    fn baseline_build(
        cell_offsets: &[u32],
        cell_points_x: &[f32],
        cell_points_y: &[f32],
        cell_points_z: &[f32],
        threshold: usize,
    ) -> Option<DenseCellIndex> {
        let mut cells = HashMap::new();
        for cell in 0..cell_offsets.len().saturating_sub(1) {
            let start = cell_offsets[cell] as usize;
            let end = cell_offsets[cell + 1] as usize;
            if end - start <= threshold {
                continue;
            }
            let xs = &cell_points_x[start..end];
            let ys = &cell_points_y[start..end];
            let zs = &cell_points_z[start..end];
            let (rx, ry, rz) = baseline_axis_ranges(xs, ys, zs);
            let axis = if rx >= ry && rx >= rz {
                0
            } else if ry >= rz {
                1
            } else {
                2
            };
            let diag = (rx * rx + ry * ry + rz * rz).sqrt();
            let coord = |i: usize| match axis {
                0 => xs[i],
                1 => ys[i],
                _ => zs[i],
            };
            let mut order: Vec<u32> = (0..(end - start) as u32).collect();
            order.sort_unstable_by(|&a, &b| {
                coord(a as usize)
                    .partial_cmp(&coord(b as usize))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let sorted_slots = order.iter().map(|&i| start as u32 + i).collect();
            let sorted_coord = order.iter().map(|&i| coord(i as usize)).collect();
            cells.insert(
                cell as u32,
                DenseCellEntry {
                    axis,
                    sorted_slots,
                    sorted_coord,
                    diag,
                },
            );
        }
        (!cells.is_empty()).then_some(DenseCellIndex { cells })
    }

    fn assert_matches_baseline(
        cell_offsets: &[u32],
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        threshold: usize,
    ) {
        let actual = DenseCellIndex::build(cell_offsets, xs, ys, zs, threshold);
        let expected = baseline_build(cell_offsets, xs, ys, zs, threshold);
        assert_eq!(actual.is_some(), expected.is_some());
        let (Some(actual), Some(expected)) = (actual, expected) else {
            return;
        };
        assert_eq!(actual.cells.len(), expected.cells.len());

        for cell in 0..cell_offsets.len().saturating_sub(1) as u32 {
            let actual_entry = actual.cells.get(&cell);
            let expected_entry = expected.cells.get(&cell);
            assert_eq!(
                actual_entry.is_some(),
                expected_entry.is_some(),
                "cell={cell}"
            );
            let (Some(actual_entry), Some(expected_entry)) = (actual_entry, expected_entry) else {
                continue;
            };
            assert_eq!(actual_entry.axis, expected_entry.axis, "cell={cell}");
            assert_eq!(
                actual_entry.sorted_slots, expected_entry.sorted_slots,
                "cell={cell}"
            );
            assert_eq!(
                actual_entry
                    .sorted_coord
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected_entry
                    .sorted_coord
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "cell={cell}"
            );
            assert_eq!(
                actual_entry.diag.to_bits(),
                expected_entry.diag.to_bits(),
                "cell={cell}"
            );

            for target_count in [0, 1, 4, 128] {
                assert_eq!(
                    actual.band_radius(cell, target_count).map(f32::to_bits),
                    expected.band_radius(cell, target_count).map(f32::to_bits),
                    "cell={cell} target_count={target_count}"
                );
            }
            for (qx, qy, qz, radius) in [
                (0.0, 0.0, 0.0, 0.0),
                (0.25, -0.5, 0.75, 0.2),
                (-1.0, 1.0, -1.0, 2.0),
            ] {
                let mut actual_band = Vec::new();
                let mut expected_band = Vec::new();
                actual.band_slots(cell, qx, qy, qz, radius, &mut actual_band);
                expected.band_slots(cell, qx, qy, qz, radius, &mut expected_band);
                assert_eq!(actual_band, expected_band, "cell={cell}");
            }
        }
    }

    #[test]
    fn dense_index_construction_and_queries_match_three_pass_baseline() {
        let cell_offsets = [0, 4, 12, 20, 27];
        let xs = [
            0.0, 0.1, 0.2, 0.3, // threshold-sized sparse cell
            -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5, // x/y range tie -> x
            0.0, 0.1, 0.0, -0.1, 0.0, 0.1, 0.0, -0.1, // y dominant
            0.0, 0.5, 0.0, -0.5, 0.25, -0.25, 0.0, // z dominant
        ];
        let ys = [
            0.0, 0.0, 0.0, 0.0, // sparse
            1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.5, 0.5, // tied with x
            -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.0, // y dominant + tie
            0.0, 0.1, 0.0, -0.1, 0.05, -0.05, 0.0, // z dominant
        ];
        let zs = [
            0.0, 0.0, 0.0, 0.0, // sparse
            0.0, 0.1, 0.0, -0.1, 0.0, 0.1, 0.0, -0.1, // x dominant
            0.0, 0.25, 0.0, -0.25, 0.1, -0.1, 0.0, 0.0, // y dominant
            -4.0, -2.0, 0.0, 2.0, 4.0, 4.0, 0.0, // z dominant + ties
        ];
        assert_matches_baseline(&cell_offsets, &xs, &ys, &zs, 4);
        assert_matches_baseline(&cell_offsets, &xs, &ys, &zs, 8);
    }

    #[test]
    fn dense_index_nonfinite_comparisons_match_baseline() {
        let offsets = [0, 8];
        let xs = [-1.0, -0.5, 0.0, 0.5, 1.0, f32::NAN, 0.0, -0.0];
        let ys = [
            f32::NAN,
            f32::NEG_INFINITY,
            -1.0,
            0.0,
            1.0,
            f32::INFINITY,
            f32::from_bits(0xffc0_0001),
            0.0,
        ];
        let zs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        assert_matches_baseline(&offsets, &xs, &ys, &zs, 4);
    }

    #[test]
    fn band_query_is_a_correct_superset() {
        // One dense cell (slots 0..n) + a sparse cell after it; threshold 3.
        let n = 50u32;
        let xs: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let ys: Vec<f32> = vec![0.0; n as usize];
        let zs: Vec<f32> = vec![0.0; n as usize];
        let cell_offsets = vec![0u32, n, n + 1];
        let mut xs2 = xs.clone();
        xs2.push(9.0);
        let mut ys2 = ys.clone();
        ys2.push(0.0);
        let mut zs2 = zs.clone();
        zs2.push(0.0);

        let idx =
            DenseCellIndex::build(&cell_offsets, &xs2, &ys2, &zs2, 3).expect("cell 0 is dense");
        assert!(idx.has(0) && !idx.has(1));

        // Band [q-r, q+r] must contain every slot within r on the x-axis.
        let (q, r) = (0.25f32, 0.05f32);
        let mut band = Vec::new();
        idx.band_slots(0, q, 0.0, 0.0, r, &mut band);
        for slot in 0..n {
            let within = (xs2[slot as usize] - q).abs() <= r;
            if within {
                assert!(
                    band.contains(&slot),
                    "slot {slot} within radius but not in band"
                );
            }
        }
        // And the band is a tight-ish superset (no slot far outside).
        for &slot in &band {
            assert!((xs2[slot as usize] - q).abs() <= r + 1e-6);
        }
    }
}
