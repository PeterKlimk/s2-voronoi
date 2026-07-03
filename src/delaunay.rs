//! Delaunay triangulation export: the dual of a computed diagram.
//!
//! A Voronoi vertex is, combinatorially, a Delaunay triangle — the three
//! cells meeting at the vertex are the triangle's corners, and the vertex
//! position is its circumcenter. Since this crate identifies vertices by
//! their generator triple, the triangulation falls out of the stored graph
//! by reading each vertex's incident cells; no new geometry is computed.
//!
//! Triangles are wound **counterclockwise** (viewed from outside the
//! sphere, or in standard right-handed plane axes), matching the
//! delaunator/CGAL convention. Indices are **canonical** cell indices:
//! welded twins never appear (the same convention as `build_adjacency`).
//!
//! Degenerate (cocircular) inputs can leave a vertex with more than three
//! incident cells after epsilon-edge reconciliation; such a vertex dualizes
//! to a convex polygon and is fan-triangulated. The choice of fan apex is
//! the only tie-breaking freedom in the output.

use glam::DVec3;

use crate::SphericalVoronoi;

/// CSR incidence: for each vertex, the canonical cells whose boundary uses
/// it. Twin cells alias canonical storage, so they are skipped to avoid
/// double-counting.
fn incidence_from_cells<'a>(
    num_vertices: usize,
    num_cells: usize,
    weld_map: Option<&[u32]>,
    mut cell: impl FnMut(usize) -> &'a [u32],
) -> (Vec<u32>, Vec<u32>) {
    let is_canonical = |i: usize| weld_map.is_none_or(|m| m[i] as usize == i);
    let mut counts = vec![0u32; num_vertices + 1];
    for i in 0..num_cells {
        if !is_canonical(i) {
            continue;
        }
        for &v in cell(i) {
            counts[v as usize + 1] += 1;
        }
    }
    for k in 1..counts.len() {
        counts[k] += counts[k - 1];
    }
    let offsets = counts.clone();
    let mut incident = vec![0u32; *offsets.last().unwrap() as usize];
    let mut cursor = offsets.clone();
    for i in 0..num_cells {
        if !is_canonical(i) {
            continue;
        }
        for &v in cell(i) {
            let c = &mut cursor[v as usize];
            incident[*c as usize] = i as u32;
            *c += 1;
        }
    }
    (offsets, incident)
}

/// Fan-triangulate one vertex's incident cells, already sorted CCW around
/// the vertex.
fn emit_fan(sorted: &[u32], out: &mut Vec<[u32; 3]>) {
    for k in 1..sorted.len() - 1 {
        out.push([sorted[0], sorted[k], sorted[k + 1]]);
    }
}

impl SphericalVoronoi {
    /// The Delaunay triangulation of the (canonical) generators: one
    /// triangle per Voronoi vertex, as `[a, b, c]` cell indices wound
    /// counterclockwise viewed from outside the sphere.
    ///
    /// This is the complete dual of the diagram — for a strictly valid
    /// diagram with `c` canonical cells it has exactly `2c - 4` triangles
    /// (cocircular degeneracies are fan-triangulated, preserving the
    /// count). Welded twins do not appear; the triangle edges are exactly
    /// the neighbor pairs of [`Self::build_adjacency`].
    pub fn delaunay_triangles(&self) -> Vec<[u32; 3]> {
        let (offsets, incident) = incidence_from_cells(
            self.num_vertices(),
            self.num_cells(),
            self.weld_map(),
            |i| self.cell(i).vertex_indices,
        );

        let mut triangles = Vec::with_capacity(self.num_vertices());
        let mut ring: Vec<(f64, u32)> = Vec::new();
        for v in 0..self.num_vertices() {
            let cells = &incident[offsets[v] as usize..offsets[v + 1] as usize];
            if cells.len() < 3 {
                continue;
            }
            let p = self.vertex(v);
            let p = DVec3::new(p.x as f64, p.y as f64, p.z as f64);
            // Tangent basis at the vertex; generators sorted CCW around the
            // outward normal are CCW viewed from outside.
            let e1 = p.cross(reference_axis(p)).normalize();
            let e2 = p.cross(e1);
            ring.clear();
            for &c in cells {
                let g = self.generator(c as usize);
                let g = DVec3::new(g.x as f64, g.y as f64, g.z as f64);
                ring.push((g.dot(e2).atan2(g.dot(e1)), c));
            }
            ring.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let sorted: Vec<u32> = ring.iter().map(|&(_, c)| c).collect();
            emit_fan(&sorted, &mut triangles);
        }
        triangles
    }
}

/// Any axis not parallel to `p`, for building a tangent basis.
fn reference_axis(p: DVec3) -> DVec3 {
    if p.x.abs() < 0.9 {
        DVec3::X
    } else {
        DVec3::Y
    }
}
