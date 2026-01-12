//! Convex hull computation using qhull (test/benchmark only).
//!
//! This module provides a ground-truth Voronoi computation via convex hull duality.
//! It's slower than the knn_clipping backend but mathematically exact (within qhull's precision).

use glam::Vec3;
use qhull_enhanced::Qh;

use crate::{SphericalVoronoi, UnitVec3, VoronoiCell};
use std::collections::HashMap;

/// A triangular facet of the convex hull, with indices into the original point array.
/// Winding is counter-clockwise when viewed from outside.
#[derive(Debug, Clone)]
pub struct HullFacet {
    pub indices: [usize; 3],
}

/// Result of computing a 3D convex hull.
#[derive(Debug)]
pub struct ConvexHull {
    pub facets: Vec<HullFacet>,
}

impl ConvexHull {
    /// Compute the convex hull of a set of 3D points.
    pub fn compute(points: &[Vec3]) -> Self {
        // qhull expects iterables of [f64; N]
        let pts: Vec<[f64; 3]> = points
            .iter()
            .map(|p| [p.x as f64, p.y as f64, p.z as f64])
            .collect();

        // Optimize for sphere points: all points are on the hull (no interior),
        // uniformly distributed (not narrow)
        let qh = Qh::builder()
            .compute(true)
            .no_near_inside(true) // Q8: no interior points to handle
            .no_narrow(true) // Q10: not a narrow distribution
            .build_from_iter(pts)
            .expect("Failed to compute convex hull");

        let mut facets = Vec::new();
        for simplex in qh.simplices() {
            let vertices: Vec<usize> = simplex
                .vertices()
                .expect("Failed to get vertices")
                .iter()
                .map(|v| v.index(&qh).expect("Failed to get vertex index"))
                .collect();

            if vertices.len() == 3 {
                facets.push(HullFacet {
                    indices: [vertices[0], vertices[1], vertices[2]],
                });
            }
        }

        ConvexHull { facets }
    }
}

/// Compute circumcenter of a spherical triangle and project it to the sphere.
fn circumcenter_on_sphere(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let normal = ab.cross(ac);

    let center = normal.normalize();

    // Check if the center is on the correct side (same hemisphere as the triangle centroid)
    let centroid = (a + b + c).normalize();
    if center.dot(centroid) < 0.0 {
        -center
    } else {
        center
    }
}

/// Order vertex indices counter-clockwise around a generator point when viewed from outside.
fn order_vertices_ccw(generator: Vec3, vertex_indices: &[usize], vertices: &[Vec3]) -> Vec<usize> {
    if vertex_indices.len() <= 2 {
        return vertex_indices.to_vec();
    }

    // Project vertices to tangent plane at generator and sort by angle
    let mut indexed: Vec<(usize, f32)> = vertex_indices
        .iter()
        .map(|&idx| {
            let v = vertices[idx];
            let angle = angle_in_tangent_plane(generator, v);
            (idx, angle)
        })
        .collect();

    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    indexed.into_iter().map(|(idx, _)| idx).collect()
}

/// Compute the angle of a point in the tangent plane at the generator.
fn angle_in_tangent_plane(generator: Vec3, point: Vec3) -> f32 {
    let up = if generator.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };

    let tangent_x = generator.cross(up).normalize();
    let tangent_y = generator.cross(tangent_x).normalize();

    let to_point = point - generator * generator.dot(point);

    let x = to_point.dot(tangent_x);
    let y = to_point.dot(tangent_y);

    y.atan2(x)
}

/// Compute spherical Voronoi diagram using convex hull duality.
///
/// This is slower than knn_clipping but serves as ground truth for testing.
pub fn compute_voronoi_qhull(points: &[Vec3]) -> SphericalVoronoi {
    let hull = ConvexHull::compute(points);

    // Compute Voronoi vertices (circumcenters of hull facets)
    let vertices: Vec<Vec3> = hull
        .facets
        .iter()
        .map(|facet| {
            let a = points[facet.indices[0]];
            let b = points[facet.indices[1]];
            let c = points[facet.indices[2]];
            circumcenter_on_sphere(a, b, c)
        })
        .collect();

    // Build adjacency: for each point, find all facets containing it
    let mut point_to_facets: HashMap<usize, Vec<usize>> = HashMap::new();
    for (facet_idx, facet) in hull.facets.iter().enumerate() {
        for &point_idx in &facet.indices {
            point_to_facets
                .entry(point_idx)
                .or_default()
                .push(facet_idx);
        }
    }

    // Build cells by ordering vertices CCW around each generator
    let generators: Vec<UnitVec3> = points
        .iter()
        .map(|&p| UnitVec3::new(p.x, p.y, p.z))
        .collect();
    let voronoi_vertices: Vec<UnitVec3> = vertices
        .iter()
        .map(|&v| UnitVec3::new(v.x, v.y, v.z))
        .collect();

    let mut cells = Vec::with_capacity(points.len());
    let mut cell_indices: Vec<u32> = Vec::new();

    for point_idx in 0..points.len() {
        let facet_indices = point_to_facets.get(&point_idx).cloned().unwrap_or_default();
        let ordered = order_vertices_ccw(points[point_idx], &facet_indices, &vertices);

        let start = cell_indices.len() as u32;
        for idx in &ordered {
            cell_indices.push(*idx as u32);
        }
        let count = ordered.len() as u16;
        cells.push((start, count));
    }

    SphericalVoronoi::from_parts(generators, voronoi_vertices, cells, cell_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hull_tetrahedron() {
        let points = vec![
            Vec3::new(1.0, 0.0, -1.0 / 2.0_f32.sqrt()),
            Vec3::new(-1.0, 0.0, -1.0 / 2.0_f32.sqrt()),
            Vec3::new(0.0, 1.0, 1.0 / 2.0_f32.sqrt()),
            Vec3::new(0.0, -1.0, 1.0 / 2.0_f32.sqrt()),
        ];
        let hull = ConvexHull::compute(&points);
        assert_eq!(hull.facets.len(), 4);
    }

    #[test]
    fn test_voronoi_octahedron() {
        // 6 points on axes -> 6 cells
        let points = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let voronoi = compute_voronoi_qhull(&points);
        assert_eq!(voronoi.num_cells(), 6);
        // Each cell should have 4 vertices (square faces on cube dual)
        for cell in voronoi.iter_cells() {
            assert_eq!(cell.len(), 4);
        }
    }
}
