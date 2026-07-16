//! Geometric measures of cells: spherical area and centroid.
//!
//! Computed on demand in f64 from the stored f32 geometry. Welded twins
//! alias their canonical cell's boundary and therefore report the same
//! measures.

use crate::spherical_arc::{resolve_owner_arc, OwnerArc};
use crate::tolerances::ANTIPODAL_DOT_EPS;
use crate::{SpherePoint, SphericalVoronoi};
use glam::DVec3;
use rustc_hash::{FxHashMap, FxHashSet};

type OwnerPairs = FxHashMap<u64, [u32; 2]>;

#[inline]
fn dvec(v: SpherePoint) -> DVec3 {
    DVec3::new(v.x() as f64, v.y() as f64, v.z() as f64)
}

#[inline]
fn edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | hi as u64
}

#[inline]
fn near_pi_dot(dot: f64) -> bool {
    dot <= -1.0 + ANTIPODAL_DOT_EPS as f64
}

#[inline]
fn solid_angle(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let det = a.dot(b.cross(c));
    let denom = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
    2.0 * det.atan2(denom)
}

#[inline]
fn solid_angle_with_bc(a: DVec3, b: DVec3, c: DVec3, bc: f64) -> f64 {
    let det = a.dot(b.cross(c));
    let denom = 1.0 + a.dot(b) + bc + c.dot(a);
    2.0 * det.atan2(denom)
}

/// Scan the diagram only after a near-pi edge has actually been observed.
/// The requested set is normally tiny, so this avoids retaining or sorting a
/// full halfedge adjacency solely for rare conditioned measure calls.
fn resolve_owner_pairs(diagram: &SphericalVoronoi, requested: &FxHashSet<u64>) -> OwnerPairs {
    let mut pairs: OwnerPairs = requested.iter().map(|&key| (key, [u32::MAX; 2])).collect();
    let mut unresolved = pairs.len();
    for cell_idx in 0..diagram.num_cells() {
        if diagram.canonical_cell_index(cell_idx) != cell_idx {
            continue;
        }
        let cell = diagram.cell(cell_idx);
        for k in 0..cell.len() {
            let key = edge_key(
                cell.vertex_indices[k],
                cell.vertex_indices[(k + 1) % cell.len()],
            );
            let Some(owners) = pairs.get_mut(&key) else {
                continue;
            };
            let cell_idx = cell_idx as u32;
            if owners[0] == u32::MAX {
                owners[0] = cell_idx;
            } else if owners[0] != cell_idx && owners[1] == u32::MAX {
                owners[1] = cell_idx;
                unresolved -= 1;
            }
        }
        if unresolved == 0 {
            break;
        }
    }
    pairs
}

fn owner_arc(
    diagram: &SphericalVoronoi,
    a: DVec3,
    b: DVec3,
    key: u64,
    owners: &OwnerPairs,
) -> Option<OwnerArc> {
    let [owner, neighbor] = *owners.get(&key)?;
    if owner == u32::MAX || neighbor == u32::MAX {
        return None;
    }
    resolve_owner_arc(
        a,
        b,
        dvec(diagram.generator(owner as usize)),
        dvec(diagram.generator(neighbor as usize)),
        ANTIPODAL_DOT_EPS as f64,
    )
    .ok()
}

fn collect_near_pi_keys(diagram: &SphericalVoronoi, cells: &[usize]) -> FxHashSet<u64> {
    let mut keys = FxHashSet::default();
    for &cell_idx in cells {
        let cell = diagram.cell(cell_idx);
        for k in 0..cell.len() {
            let ai = cell.vertex_indices[k];
            let bi = cell.vertex_indices[(k + 1) % cell.len()];
            if near_pi_dot(dvec(diagram.vertex(ai as usize)).dot(dvec(diagram.vertex(bi as usize))))
            {
                keys.insert(edge_key(ai, bi));
            }
        }
    }
    keys
}

fn cell_area_pass(
    diagram: &SphericalVoronoi,
    index: usize,
    owners: Option<&OwnerPairs>,
) -> (f64, bool) {
    let cell = diagram.cell(index);
    if cell.len() < 3 {
        return (0.0, false);
    }
    let generator = dvec(diagram.generator(index));
    let mut total = 0.0f64;
    let mut conditioned = false;
    for k in 0..cell.len() {
        let ai = cell.vertex_indices[k];
        let bi = cell.vertex_indices[(k + 1) % cell.len()];
        let a = dvec(diagram.vertex(ai as usize));
        let b = dvec(diagram.vertex(bi as usize));
        let edge_dot = a.dot(b);
        if near_pi_dot(edge_dot) {
            conditioned = true;
            if let Some(arc) =
                owners.and_then(|owners| owner_arc(diagram, a, b, edge_key(ai, bi), owners))
            {
                let midpoint = arc.sample(0.5);
                total += solid_angle(generator, a, midpoint);
                total += solid_angle(generator, midpoint, b);
                continue;
            }
        }
        total += solid_angle_with_bc(generator, a, b, edge_dot);
    }
    (total.abs(), conditioned)
}

fn cell_centroid_pass(
    diagram: &SphericalVoronoi,
    index: usize,
    owners: Option<&OwnerPairs>,
) -> (SpherePoint, bool) {
    let cell = diagram.cell(index);
    let generator = diagram.generator(index);
    if cell.len() < 3 {
        return (generator, false);
    }

    let mut integral = DVec3::ZERO;
    let mut conditioned = false;
    for k in 0..cell.len() {
        let ai = cell.vertex_indices[k];
        let bi = cell.vertex_indices[(k + 1) % cell.len()];
        let a = dvec(diagram.vertex(ai as usize));
        let b = dvec(diagram.vertex(bi as usize));
        let edge_dot = a.dot(b);
        if near_pi_dot(edge_dot) {
            conditioned = true;
            if let Some(arc) =
                owners.and_then(|owners| owner_arc(diagram, a, b, edge_key(ai, bi), owners))
            {
                integral += arc.oriented_normal * arc.angle * 0.5;
                continue;
            }
        }

        let cross = a.cross(b);
        let cross_len = cross.length();
        if cross_len <= f64::EPSILON {
            continue;
        }
        let arc_angle = cross_len.atan2(edge_dot);
        integral += cross * (arc_angle / cross_len) * 0.5;
    }

    let len = integral.length();
    if len <= f64::EPSILON {
        return (generator, conditioned);
    }
    let mut centroid = integral / len;
    if centroid.dot(dvec(generator)) < 0.0 {
        centroid = -centroid;
    }
    (SpherePoint::from_direction_dvec3(centroid), conditioned)
}

impl SphericalVoronoi {
    /// Spherical (solid-angle) area of a cell, in steradians.
    ///
    /// The sum over canonical cells of a strictly valid diagram is `4π`.
    /// Computed as a generator-centered fan of signed spherical triangles in
    /// f64. A near-semicircle boundary edge is split at its owner-plane
    /// midpoint so no triangle depends on an ill-conditioned endpoint cross.
    /// Cells with fewer than 3 vertices report zero area.
    /// Ordinary cells cost `O(cell degree)`; an observed near-semicircle edge
    /// triggers one cold `O(total halfedges)` owner lookup with sparse memory.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.num_cells()` (see [`Self::cell`]).
    #[track_caller]
    pub fn cell_area(&self, index: usize) -> f64 {
        let index = self.canonical_cell_index(index);
        let (area, conditioned) = cell_area_pass(self, index, None);
        if !conditioned {
            return area;
        }
        let keys = collect_near_pi_keys(self, &[index]);
        let owners = resolve_owner_pairs(self, &keys);
        cell_area_pass(self, index, Some(&owners)).0
    }

    /// Spherical centroid of a cell: the direction of the integral of the
    /// position vector over the cell, projected back onto the sphere.
    ///
    /// This is the target point of Lloyd relaxation (centroidal Voronoi
    /// tessellation): move each generator to its cell centroid and recompute.
    /// Uses the exact boundary integral `∫ p dA = ½ Σ θ_k n̂_k` over the
    /// cell's edges in f64. Near-semicircle edges recover `θ_k` and `n̂_k`
    /// from their two owning generators. Degenerate cells fall back to the
    /// generator itself. Ordinary cells cost `O(cell degree)`; an observed
    /// near-semicircle edge triggers one cold `O(total halfedges)` owner lookup
    /// with sparse memory.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.num_cells()` (see [`Self::cell`]).
    #[track_caller]
    pub fn cell_centroid(&self, index: usize) -> SpherePoint {
        let index = self.canonical_cell_index(index);
        let (centroid, conditioned) = cell_centroid_pass(self, index, None);
        if !conditioned {
            #[cfg(feature = "profiling")]
            crate::point_audit::record_sphere_point_f64_canonical(
                crate::point_audit::PointProducer::Centroid,
                centroid,
            );
            return centroid;
        }
        let keys = collect_near_pi_keys(self, &[index]);
        let owners = resolve_owner_pairs(self, &keys);
        let centroid = cell_centroid_pass(self, index, Some(&owners)).0;
        #[cfg(feature = "profiling")]
        crate::point_audit::record_sphere_point_f64_canonical(
            crate::point_audit::PointProducer::Centroid,
            centroid,
        );
        centroid
    }

    /// The next generator set of Lloyd relaxation: every cell's centroid,
    /// in input order. Recompute with [`crate::compute`] to complete the step:
    ///
    /// ```ignore
    /// for _ in 0..iters {
    ///     points = compute(&points)?.lloyd_step();
    /// }
    /// ```
    ///
    /// Welded twins report their canonical cell's centroid, so coincident
    /// inputs remain coincident (and re-weld) under relaxation. If any cell
    /// has a near-semicircle edge, the batch resolves all such owners in one
    /// cold linear scan rather than rebuilding adjacency per cell.
    pub fn lloyd_step(&self) -> Vec<SpherePoint> {
        let mut centroids = self.generators().to_vec();
        let mut conditioned_cells = Vec::new();
        for (index, slot) in centroids.iter_mut().enumerate() {
            if self.canonical_cell_index(index) != index {
                continue;
            }
            let (centroid, conditioned) = cell_centroid_pass(self, index, None);
            *slot = centroid;
            if conditioned {
                conditioned_cells.push(index);
            }
        }

        if !conditioned_cells.is_empty() {
            let keys = collect_near_pi_keys(self, &conditioned_cells);
            let owners = resolve_owner_pairs(self, &keys);
            for &index in &conditioned_cells {
                centroids[index] = cell_centroid_pass(self, index, Some(&owners)).0;
            }
        }

        for index in 0..centroids.len() {
            let canonical = self.canonical_cell_index(index);
            if canonical != index {
                centroids[index] = centroids[canonical];
            }
        }
        #[cfg(feature = "profiling")]
        for &centroid in &centroids {
            crate::point_audit::record_sphere_point_f64_canonical(
                crate::point_audit::PointProducer::Centroid,
                centroid,
            );
        }
        centroids
    }
}
