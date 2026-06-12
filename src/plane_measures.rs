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
    /// area. Computed by the shoelace formula in f64; cells with fewer than
    /// 3 vertices report zero area.
    pub fn cell_area(&self, index: usize) -> f64 {
        let cell = self.cell(index);
        let n = cell.len();
        if n < 3 {
            return 0.0;
        }
        let mut acc = 0.0f64;
        for k in 0..n {
            let a = self.vertex(cell[k] as usize);
            let b = self.vertex(cell[(k + 1) % n] as usize);
            acc += (a.x as f64) * (b.y as f64) - (b.x as f64) * (a.y as f64);
        }
        (0.5 * acc).abs()
    }

    /// Centroid of a cell.
    ///
    /// This is the target point of Lloyd relaxation (centroidal Voronoi
    /// tessellation): move each generator to its cell centroid and
    /// recompute. Computed by the shoelace centroid formula in f64;
    /// degenerate cells (fewer than 3 vertices, or vanishing area) fall
    /// back to the generator. The result is clamped into the rect so it is
    /// always a valid `compute_plane` input.
    pub fn cell_centroid(&self, index: usize) -> PlanePoint {
        let cell = self.cell(index);
        let n = cell.len();
        let generator = self.generator(index);
        if n < 3 {
            return generator;
        }

        let mut area2 = 0.0f64;
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        for k in 0..n {
            let a = self.vertex(cell[k] as usize);
            let b = self.vertex(cell[(k + 1) % n] as usize);
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
        PlanePoint::new(
            ((cx * scale) as f32).clamp(rect.min.x, rect.max.x),
            ((cy * scale) as f32).clamp(rect.min.y, rect.max.y),
        )
    }
}
