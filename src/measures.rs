//! Geometric measures of cells: spherical area and centroid.
//!
//! Computed on demand in f64 from the stored f32 geometry. Welded twins
//! alias their canonical cell's boundary and therefore report the same
//! measures.

use crate::{SphericalVoronoi, UnitVec3};
use glam::DVec3;

#[inline]
fn dvec(v: UnitVec3) -> DVec3 {
    DVec3::new(v.x as f64, v.y as f64, v.z as f64)
}

impl SphericalVoronoi {
    /// Spherical (solid-angle) area of a cell, in steradians.
    ///
    /// The sum over canonical cells of a strictly valid diagram is `4π`.
    /// Computed as a signed triangle fan of solid angles
    /// (van Oosterom–Strackee) in f64; cells with fewer than 3 vertices
    /// report zero area.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.num_cells()` (see [`Self::cell`]).
    #[track_caller]
    pub fn cell_area(&self, index: usize) -> f64 {
        let cell = self.cell(index);
        let n = cell.len();
        if n < 3 {
            return 0.0;
        }
        let v0 = dvec(self.vertex(cell.vertex_indices[0] as usize));
        let mut total = 0.0f64;
        for k in 1..n - 1 {
            let v1 = dvec(self.vertex(cell.vertex_indices[k] as usize));
            let v2 = dvec(self.vertex(cell.vertex_indices[k + 1] as usize));
            // Solid angle of the spherical triangle (v0, v1, v2):
            // tan(omega/2) = det(v0, v1, v2) / (1 + dots), signed by orientation.
            let det = v0.dot(v1.cross(v2));
            let denom = 1.0 + v0.dot(v1) + v1.dot(v2) + v2.dot(v0);
            total += 2.0 * det.atan2(denom);
        }
        total.abs()
    }

    /// Spherical centroid of a cell: the direction of the integral of the
    /// position vector over the cell, projected back onto the sphere.
    ///
    /// This is the target point of Lloyd relaxation (centroidal Voronoi
    /// tessellation): move each generator to its cell centroid and recompute.
    /// Uses the exact boundary integral `∫ p dA = ½ Σ θ_k n̂_k` over the
    /// cell's edges (arc angle times unit edge-plane normal), in f64.
    /// Degenerate cells (fewer than 3 vertices, or a vanishing integral)
    /// fall back to the generator itself.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.num_cells()` (see [`Self::cell`]).
    #[track_caller]
    pub fn cell_centroid(&self, index: usize) -> UnitVec3 {
        let cell = self.cell(index);
        let n = cell.len();
        let generator = self.generator(index);
        if n < 3 {
            return generator;
        }

        let mut integral = DVec3::ZERO;
        for k in 0..n {
            let a = dvec(self.vertex(cell.vertex_indices[k] as usize));
            let b = dvec(self.vertex(cell.vertex_indices[(k + 1) % n] as usize));
            let cross = a.cross(b);
            let cross_len = cross.length();
            if cross_len <= f64::EPSILON {
                continue;
            }
            let arc_angle = cross_len.atan2(a.dot(b));
            integral += cross * (arc_angle / cross_len) * 0.5;
        }

        let len = integral.length();
        if len <= f64::EPSILON {
            return generator;
        }
        let mut centroid = integral / len;
        // The boundary integral's sign follows the stored winding; orient it
        // into the generator's hemisphere (the cell always contains its
        // generator and spans less than a hemisphere in the supported model).
        if centroid.dot(dvec(generator)) < 0.0 {
            centroid = -centroid;
        }
        UnitVec3::new(centroid.x as f32, centroid.y as f32, centroid.z as f32)
    }

    /// The next generator set of Lloyd relaxation: every cell's centroid,
    /// in input order. Recompute with [`crate::compute`] to complete the
    /// step:
    ///
    /// ```ignore
    /// for _ in 0..iters {
    ///     points = compute(&points)?.lloyd_step();
    /// }
    /// ```
    ///
    /// Welded twins report their canonical cell's centroid, so coincident
    /// inputs remain coincident (and re-weld) under relaxation.
    pub fn lloyd_step(&self) -> Vec<UnitVec3> {
        (0..self.num_cells())
            .map(|i| self.cell_centroid(i))
            .collect()
    }
}
