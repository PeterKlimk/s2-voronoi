//! Uniform 2D grid over the unit square for planar Voronoi neighbor queries.
//!
//! The planar analog of `cube_grid`: the same CSR layout (cell offsets plus
//! slot-ordered SoA point coordinates) and the same directed-eligibility
//! semantics (the shared [`DirectedEligibility`]), but one flat `res x res`
//! chart with no faces, no corners, and no seam wrap. Cells are exact
//! axis-aligned boxes, so min-distance certificates come from box geometry
//! instead of conservative spherical caps.
//!
//! Distance semantics are squared Euclidean (`dist_sq`, smaller is closer)
//! with *lower*-bound certificates on unseen points — the planar pipeline
//! never uses the sphere's dot-product ("bigger is closer") form.
//!
//! Input points are expected in the unit square `[0, 1]^2`; the pipeline
//! entry normalizes the user's bounding rect to it. Out-of-range coordinates
//! are clamped into the edge cells (and rejected upstream by validation).

pub(crate) mod packed;
pub(crate) mod periodic;
mod query;

pub(crate) use query::{PlaneNeighborFrontier, PlaneNeighborStream, PlaneShellFrontier};
// (PlaneGridScratch is defined below and used by both.)

#[cfg(test)]
mod tests;

use glam::Vec2;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Uniform spatial grid over the unit square.
pub(crate) struct PlaneGrid {
    res: usize,
    /// Start index into `point_indices` for each cell, plus final length.
    /// Length: res^2 + 1. Cells are row-major: `cell = iy * res + ix`.
    cell_offsets: Vec<u32>,
    /// Point indices grouped by cell. Length: n.
    point_indices: Vec<u32>,
    /// Precomputed cell index per point. Length: n.
    point_cells: Vec<u32>,
    /// Inverse mapping from point index to SoA slot. Length: n.
    point_slots: Vec<u32>,
    /// X coordinates of points, ordered by cell (use `cell_offsets` for ranges).
    cell_points_x: Vec<f32>,
    /// Y coordinates of points, ordered by cell.
    cell_points_y: Vec<f32>,
    /// Wall coordinates `i / res` for `i` in `0..=res`, precomputed (the
    /// certificate reads 4 walls per ring advance; an f32 division each
    /// would dominate that path).
    walls: Vec<f32>,
}

/// Reusable per-query scratch for [`PlaneShellFrontier`]: avoids a heap
/// allocation per cell in the million-cell driver loop (mirrors the sphere's
/// `CubeMapGridScratch` pattern).
#[derive(Default)]
pub(crate) struct PlaneGridScratch {
    pub(super) pending: Vec<(crate::fp::OrdF32, u32)>,
}

impl PlaneGrid {
    pub(crate) fn new(points: &[Vec2], res: usize) -> Self {
        assert!(res >= 1, "plane grid resolution must be at least 1");
        let n = points.len();
        let num_cells = res * res;

        let mut point_cells = vec![0u32; n];
        let mut counts = vec![0u32; num_cells];
        for (i, p) in points.iter().enumerate() {
            debug_assert!(
                (-1e-6..=1.0 + 1e-6).contains(&p.x) && (-1e-6..=1.0 + 1e-6).contains(&p.y),
                "plane grid input outside the unit square: {p:?}"
            );
            let cell = cell_of_point(*p, res) as u32;
            point_cells[i] = cell;
            counts[cell as usize] += 1;
        }

        let mut cell_offsets = vec![0u32; num_cells + 1];
        let mut acc = 0u32;
        for (cell, &count) in counts.iter().enumerate() {
            cell_offsets[cell] = acc;
            acc += count;
        }
        cell_offsets[num_cells] = acc;

        let mut cursor = cell_offsets[..num_cells].to_vec();
        let mut point_indices = vec![0u32; n];
        let mut point_slots = vec![0u32; n];
        let mut cell_points_x = vec![0.0f32; n];
        let mut cell_points_y = vec![0.0f32; n];
        for (i, p) in points.iter().enumerate() {
            let cell = point_cells[i] as usize;
            let slot = cursor[cell];
            cursor[cell] += 1;
            point_indices[slot as usize] = i as u32;
            point_slots[i] = slot;
            cell_points_x[slot as usize] = p.x;
            cell_points_y[slot as usize] = p.y;
        }

        // Same expression as a point constructed at the wall (`i / res`),
        // so certificate comparisons match point classification rounding.
        let walls = (0..=res).map(|i| i as f32 / res as f32).collect();

        Self {
            res,
            cell_offsets,
            point_indices,
            point_cells,
            point_slots,
            cell_points_x,
            cell_points_y,
            walls,
        }
    }

    /// Create a reusable scratch for repeated queries.
    pub(crate) fn make_scratch(&self) -> PlaneGridScratch {
        PlaneGridScratch::default()
    }

    /// Get cell index for a point.
    #[inline]
    pub(crate) fn point_to_cell(&self, p: Vec2) -> usize {
        cell_of_point(p, self.res)
    }

    /// Get the precomputed cell index for `points[idx]` used to build this grid.
    // Used by the packed-kNN stage when it lands (cell-grouped query runs).
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn point_index_to_cell(&self, idx: usize) -> usize {
        self.point_cells[idx] as usize
    }

    /// Get the SoA slot index for `points[idx]` used to build this grid.
    #[inline]
    pub(crate) fn point_index_to_slot(&self, idx: usize) -> u32 {
        self.point_slots[idx]
    }

    /// Grid resolution (cells per axis).
    #[inline]
    pub(crate) fn res(&self) -> usize {
        self.res
    }

    /// Cell offsets array (length = res^2 + 1).
    #[inline]
    pub(crate) fn cell_offsets(&self) -> &[u32] {
        &self.cell_offsets
    }

    /// Point indices array (SoA layout, length = n).
    #[inline]
    pub(crate) fn point_indices(&self) -> &[u32] {
        &self.point_indices
    }

    /// Collect all pairs of point indices within `radius` of each other
    /// (each unordered pair reported at least once; duplicates possible).
    ///
    /// Grid-integrated proximity detection: a qualifying pair is either in
    /// one cell or straddles a wall with both members within `radius` of it
    /// (`radius` is far below the cell size at every production
    /// resolution), so the scan checks each cell's interior triangularly
    /// plus, for near-wall points only, the E/N/NE/NW neighbors (the W/S
    /// directions are covered by those cells' own forward checks). The
    /// common no-pair case is a pure read-only scan of the already-built
    /// grid — no hashing, no allocation.
    pub(crate) fn collect_pairs_within(&self, radius: f32, out: &mut Vec<(u32, u32)>) {
        let res = self.res;
        let num_cells = res * res;
        // Parallel over row bands; qualifying pairs are rare, so the
        // per-band vecs are almost always empty and the reduce is free.
        let bands: Vec<Vec<(u32, u32)>> = maybe_par_into_iter!(0..res)
            .map(|band| {
                let mut local = Vec::new();
                for cell in band * res..(band + 1) * res {
                    self.collect_pairs_for_cell(cell, radius, &mut local);
                }
                local
            })
            .collect();
        let _ = num_cells;
        for band in bands {
            out.extend(band);
        }
    }

    /// One cell's contribution to [`Self::collect_pairs_within`].
    fn collect_pairs_for_cell(&self, cell: usize, radius: f32, out: &mut Vec<(u32, u32)>) {
        let res = self.res;
        let r_sq = radius * radius;
        // Pathological cells (everything coincident) would make the
        // triangular scan quadratic; switch to a sorted quantized sweep.
        const FAT_CELL_LIMIT: usize = 256;
        let mut fat_scratch: Vec<(u64, u32)> = Vec::new();

        {
            let start = self.cell_offsets[cell] as usize;
            let end = self.cell_offsets[cell + 1] as usize;
            let k = end - start;
            if k == 0 {
                return;
            }
            let xs = &self.cell_points_x[start..end];
            let ys = &self.cell_points_y[start..end];
            let idx = &self.point_indices[start..end];

            // Within-cell pairs.
            if k <= FAT_CELL_LIMIT {
                for i in 0..k {
                    for j in (i + 1)..k {
                        let dx = xs[j] - xs[i];
                        let dy = ys[j] - ys[i];
                        if dx * dx + dy * dy <= r_sq {
                            out.push((idx[i], idx[j]));
                        }
                    }
                }
            } else {
                // Quantize to radius-pitch keys, sort, compare each point
                // against the 3x3 neighborhood of quanta via key ranges.
                let pitch = radius.max(f32::MIN_POSITIVE);
                fat_scratch.clear();
                fat_scratch.extend((0..k).map(|i| {
                    let qx = (xs[i] / pitch) as u32;
                    let qy = (ys[i] / pitch) as u32;
                    ((((qy as u64) << 32) | qx as u64), i as u32)
                }));
                fat_scratch.sort_unstable();
                for n in 0..k {
                    let (key, i_local) = fat_scratch[n];
                    let (qy, qx) = ((key >> 32) as u32, (key & 0xFFFF_FFFF) as u32);
                    for dqy in 0..=1u32 {
                        for dqx in -1i64..=1 {
                            if dqy == 0 && dqx < 0 {
                                continue; // forward half-space only
                            }
                            let nkey = (((qy + dqy) as u64) << 32)
                                | ((qx as i64 + dqx).max(0) as u64 & 0xFFFF_FFFF);
                            let from = if nkey == key {
                                n + 1
                            } else {
                                fat_scratch.partition_point(|&(kk, _)| kk < nkey)
                            };
                            for &(kk, j_local) in &fat_scratch[from..] {
                                if kk != nkey {
                                    break;
                                }
                                let (i, j) = (i_local as usize, j_local as usize);
                                let dx = xs[j] - xs[i];
                                let dy = ys[j] - ys[i];
                                if dx * dx + dy * dy <= r_sq {
                                    out.push((idx[i], idx[j]));
                                }
                            }
                        }
                    }
                }
            }

            // Cross-cell pairs: only points within `radius` of the E/N
            // walls (plus the NE/NW corners) can pair into those neighbors.
            let (ix, iy) = (cell % res, cell / res);
            let near_e = ix + 1 < res;
            let near_n = iy + 1 < res;
            let e_wall = if near_e { self.wall(ix + 1) } else { 0.0 };
            let n_wall = if near_n { self.wall(iy + 1) } else { 0.0 };
            let w_wall = self.wall(ix);
            for i in 0..k {
                let (x, y) = (xs[i], ys[i]);
                let close_e = near_e && e_wall - x <= radius;
                let close_n = near_n && n_wall - y <= radius;
                let close_w = ix > 0 && x - w_wall <= radius;
                if close_e {
                    self.pairs_against_cell(cell + 1, x, y, idx[i], r_sq, out);
                }
                if close_n {
                    self.pairs_against_cell(cell + res, x, y, idx[i], r_sq, out);
                }
                if close_e && close_n {
                    self.pairs_against_cell(cell + res + 1, x, y, idx[i], r_sq, out);
                }
                if close_w && close_n {
                    self.pairs_against_cell(cell + res - 1, x, y, idx[i], r_sq, out);
                }
            }
        }
    }

    /// Compare one point against every point of `cell`, pushing qualifying
    /// pairs. Cross-wall bands are radius-thin, so `cell` rarely has many
    /// qualifying members.
    fn pairs_against_cell(
        &self,
        cell: usize,
        x: f32,
        y: f32,
        index: u32,
        r_sq: f32,
        out: &mut Vec<(u32, u32)>,
    ) {
        let start = self.cell_offsets[cell] as usize;
        let end = self.cell_offsets[cell + 1] as usize;
        for slot in start..end {
            let dx = self.cell_points_x[slot] - x;
            let dy = self.cell_points_y[slot] - y;
            if dx * dx + dy * dy <= r_sq {
                out.push((index, self.point_indices[slot]));
            }
        }
    }

    /// Grid-line coordinate of wall `i` (`i` in `0..=res`), precomputed as
    /// `i / res` in f32 — the same rounding as a point constructed at the
    /// wall, so certificate comparisons match point classification to
    /// within an ulp.
    #[inline]
    pub(crate) fn wall(&self, i: usize) -> f32 {
        self.walls[i]
    }
}

/// Lower bound on the squared distance from `(qx, qy)` to anything outside
/// the Chebyshev-radius-`radius` cell box around `(cx, cy)`.
///
/// Sides clipped at the domain edge have nothing beyond them; if every side
/// is clipped the bound is `INFINITY` (nothing exists outside). Each exposed
/// side distance is shrunk by ulps of the wall coordinate
/// ([`crate::tolerances::PLANE_WALL_CLASSIFICATION_SLACK`]): point
/// classification and the wall coordinate round independently, so a point
/// classified outside the box can sit slightly inside the f32 wall value.
///
/// Single owner of the box-certificate logic: the shell frontier's per-ring
/// certificate and the packed stage's security thresholds both use it.
pub(crate) fn outside_box_dist_sq(
    grid: &PlaneGrid,
    cx: usize,
    cy: usize,
    radius: usize,
    qx: f32,
    qy: f32,
) -> f32 {
    const SLACK: f32 = crate::tolerances::PLANE_WALL_CLASSIFICATION_SLACK;
    let res = grid.res();
    let mut d = f32::INFINITY;
    if cx > radius {
        let w = grid.wall(cx - radius);
        d = d.min(qx - w - w * SLACK);
    }
    if cx + radius < res - 1 {
        let w = grid.wall(cx + radius + 1);
        d = d.min(w - w * SLACK - qx);
    }
    if cy > radius {
        let w = grid.wall(cy - radius);
        d = d.min(qy - w - w * SLACK);
    }
    if cy + radius < res - 1 {
        let w = grid.wall(cy + radius + 1);
        d = d.min(w - w * SLACK - qy);
    }
    if d == f32::INFINITY {
        return f32::INFINITY;
    }
    // A query outside the unit square (clamped into an edge cell) can sit
    // outside the box; 0 is the sound bound there.
    let d = d.max(0.0);
    d * d
}

/// Row-major cell index of a point, clamped into the grid.
#[inline]
fn cell_of_point(p: Vec2, res: usize) -> usize {
    let scale = res as f32;
    // `as usize` saturates at 0 for negative f32, so slight negatives clamp
    // to the edge cell without a branch.
    let ix = ((p.x * scale) as usize).min(res - 1);
    let iy = ((p.y * scale) as usize).min(res - 1);
    iy * res + ix
}
