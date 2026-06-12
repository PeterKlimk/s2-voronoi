//! Geometric measures of planar cells: area and centroid.
//!
//! Computed on demand in f64 from the stored f32 geometry. Welded twins
//! alias their canonical cell's boundary and therefore report the same
//! measures.

use crate::{PlanarVoronoi, PlanePoint};

impl PlanarVoronoi {
    /// Area of a cell, in the rect's coordinate units squared.
    ///
    /// The sum over canonical cells of a strictly valid diagram is the rect
    /// area (for both topologies). Computed by the shoelace formula in f64
    /// over [`Self::cell_polygon`], so periodic seam-straddling cells are
    /// measured on their contiguous unwrapped polygon. Cells with fewer
    /// than 3 vertices report zero area.
    pub fn cell_area(&self, index: usize) -> f64 {
        let poly = self.cell_polygon(index);
        let n = poly.len();
        if n < 3 {
            return 0.0;
        }
        let mut acc = 0.0f64;
        for k in 0..n {
            let a = poly[k];
            let b = poly[(k + 1) % n];
            acc += (a.x as f64) * (b.y as f64) - (b.x as f64) * (a.y as f64);
        }
        (0.5 * acc).abs()
    }

    /// Centroid of a cell.
    ///
    /// This is the target point of Lloyd relaxation (centroidal Voronoi
    /// tessellation): move each generator to its cell centroid and
    /// recompute — on both topologies (periodic Lloyd / CVT in periodic
    /// domains is the classic physics use). Computed by the shoelace
    /// centroid formula in f64 over [`Self::cell_polygon`]; degenerate
    /// cells fall back to the generator.
    ///
    /// Bounded diagrams clamp the result into the rect. Periodic diagrams
    /// wrap it into the rect instead (a seam cell's centroid can land past
    /// the edge in unwrapped coordinates; its wrapped image is the same
    /// torus point), so it is always a valid input for the matching
    /// compute function.
    pub fn cell_centroid(&self, index: usize) -> PlanePoint {
        let poly = self.cell_polygon(index);
        let n = poly.len();
        let generator = self.generator(index);
        if n < 3 {
            return generator;
        }

        let mut area2 = 0.0f64;
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        for k in 0..n {
            let a = poly[k];
            let b = poly[(k + 1) % n];
            let (ax, ay) = (a.x as f64, a.y as f64);
            let (bx, by) = (b.x as f64, b.y as f64);
            let cross = ax * by - bx * ay;
            area2 += cross;
            cx += (ax + bx) * cross;
            cy += (ay + by) * cross;
        }
        if area2.abs() <= f64::EPSILON {
            return generator;
        }
        let scale = 1.0 / (3.0 * area2);
        let rect = self.rect();
        let (x, y) = ((cx * scale) as f32, (cy * scale) as f32);
        match self.topology() {
            crate::PlaneTopology::Bounded => PlanePoint::new(
                x.clamp(rect.min.x, rect.max.x),
                y.clamp(rect.min.y, rect.max.y),
            ),
            crate::PlaneTopology::Periodic => {
                let (w, h) = (rect.width(), rect.height());
                let wrapped_x = rect.min.x + (x - rect.min.x).rem_euclid(w);
                let wrapped_y = rect.min.y + (y - rect.min.y).rem_euclid(h);
                PlanePoint::new(
                    wrapped_x.clamp(rect.min.x, rect.max.x),
                    wrapped_y.clamp(rect.min.y, rect.max.y),
                )
            }
        }
    }

    /// The next generator set of Lloyd relaxation: every cell's centroid,
    /// in input order. Centroids are topology-aware (bounded diagrams
    /// clamp into the rect, periodic diagrams wrap), so the result is
    /// always a valid input for the matching compute function:
    ///
    /// ```ignore
    /// for _ in 0..iters {
    ///     points = compute_plane_periodic(&points, rect)?.lloyd_step();
    /// }
    /// ```
    ///
    /// Welded twins report their canonical cell's centroid, so coincident
    /// inputs remain coincident (and re-weld) under relaxation.
    pub fn lloyd_step(&self) -> Vec<PlanePoint> {
        (0..self.num_cells())
            .map(|i| self.cell_centroid(i))
            .collect()
    }
}
