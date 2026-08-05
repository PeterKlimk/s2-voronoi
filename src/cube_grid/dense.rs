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

    #[test]
    fn dense_index_build_pins_threshold_and_axis_selection() {
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

        let index = DenseCellIndex::build(&cell_offsets, &xs, &ys, &zs, 4).unwrap();
        assert_eq!(index.cells.len(), 3);
        assert!(!index.cells.contains_key(&0));
        for (cell, axis, expected_slots, ranges) in [
            (1, 0, 4..12, (2.0f32, 2.0f32, 0.2f32)),
            (2, 1, 12..20, (0.2f32, 6.0f32, 0.5f32)),
            (3, 2, 20..27, (1.0f32, 0.2f32, 8.0f32)),
        ] {
            let entry = &index.cells[&cell];
            assert_eq!(entry.axis, axis);
            assert!(entry.sorted_coord.windows(2).all(|w| w[0] <= w[1]));
            let mut slots = entry.sorted_slots.clone();
            slots.sort_unstable();
            assert_eq!(slots, expected_slots.collect::<Vec<_>>());
            let (rx, ry, rz) = ranges;
            assert_eq!(
                entry.diag.to_bits(),
                (rx * rx + ry * ry + rz * rz).sqrt().to_bits()
            );
        }
        assert!(index.band_radius(0, 4).is_none());
        assert!(index.band_radius(1, 4).is_some());
        assert!(DenseCellIndex::build(&cell_offsets, &xs, &ys, &zs, 8).is_none());
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

        let index =
            DenseCellIndex::build(&cell_offsets, &xs2, &ys2, &zs2, 3).expect("cell 0 is dense");
        assert!(index.band_radius(0, 8).is_some());
        assert!(index.band_radius(1, 8).is_none());

        // Band [q-r, q+r] must contain every slot within r on the x-axis.
        let (q, r) = (0.25f32, 0.05f32);
        let mut band = Vec::new();
        index.band_slots(0, q, 0.0, 0.0, r, &mut band);
        for slot in 0..n {
            let within = (xs2[slot as usize] - q).abs() <= r;
            assert!(
                !within || band.contains(&slot),
                "slot {slot} within radius but not in band"
            );
        }
        // And the band is a tight-ish superset (no slot far outside).
        for &slot in &band {
            assert!((xs2[slot as usize] - q).abs() <= r + 1e-6);
        }
    }
}
