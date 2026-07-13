use super::projection::sort3_u32;
use super::{
    BuilderDebugState, BuilderImpl, ExtractionInvariantFailure, FallbackBuilder, GnomonicBuilder,
    Topo2DBuilder,
};
use crate::fp;
use crate::knn_clipping::cell_build::{CellFailure, CellOutputBuffer};
use crate::knn_clipping::topo2d::types::INVALID_PLANE_ID;
use glam::{DVec3, Vec3};

use crate::tolerances::{EXTRACT_DEGENERATE_LEN2, FALLBACK_PLANE_TOL};

#[derive(Clone, Copy)]
struct FallbackVertex {
    position: DVec3,
    plane_a: usize,
    plane_b: usize,
}

/// Whether every real edge's neighbor appears in BOTH endpoint vertex keys —
/// the emit engine's key/edge-consistency precondition (see
/// `CellOutputBuffer::edge_keys_verified`). The fallback extractors can
/// violate it: split-plane corner resolution pushes `(B, X)` + `(X, C)` for a
/// corner between edge planes B and C, and when position dedup collapses the
/// split micro-edge the surviving vertex keeps X in its key while the edge
/// sequence continues with C.
fn edge_keys_consistent(buffer: &CellOutputBuffer) -> bool {
    let n = buffer.vertices.len();
    (0..n).all(|i| {
        let en = buffer.edge_neighbor_globals[i];
        en == u32::MAX
            || (buffer.vertices[i].0.contains(&en)
                && buffer.vertices[if i + 1 == n { 0 } else { i + 1 }]
                    .0
                    .contains(&en))
    })
}

impl GnomonicBuilder {
    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn to_vertex_data_full(
        &self,
        buffer: &mut CellOutputBuffer,
    ) -> Result<(), CellFailure> {
        let poly = self.current_poly();
        if !self.is_bounded() || poly.len < 3 {
            return Err(CellFailure::NoValidSeed);
        }
        buffer.clear();
        buffer.vertices.reserve(poly.len);
        buffer.edge_neighbor_globals.reserve(poly.len);
        buffer.edge_neighbor_slots.reserve(poly.len);

        let vertices = buffer.vertices.spare_capacity_mut();
        let edge_neighbor_globals = buffer.edge_neighbor_globals.spare_capacity_mut();
        let edge_neighbor_slots = buffer.edge_neighbor_slots.spare_capacity_mut();
        debug_assert!(vertices.len() >= poly.len);
        debug_assert!(edge_neighbor_globals.len() >= poly.len);
        debug_assert!(edge_neighbor_slots.len() >= poly.len);

        let gen_idx = self.generator_idx as u32;
        for i in 0..poly.len {
            let u = poly.us[i];
            let v = poly.vs[i];
            let (plane_a, plane_b) = poly.vertex_planes[i];

            let dir = DVec3::new(
                fp::fma_f64(
                    u,
                    self.basis.t1.x,
                    fp::fma_f64(v, self.basis.t2.x, self.basis.g.x),
                ),
                fp::fma_f64(
                    u,
                    self.basis.t1.y,
                    fp::fma_f64(v, self.basis.t2.y, self.basis.g.y),
                ),
                fp::fma_f64(
                    u,
                    self.basis.t1.z,
                    fp::fma_f64(v, self.basis.t2.z, self.basis.g.z),
                ),
            );
            let dir = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32);
            let len2 = dir.length_squared();
            if !len2.is_finite() || len2 < EXTRACT_DEGENERATE_LEN2 {
                return Err(CellFailure::NoValidSeed);
            }
            let v_pos = dir * len2.sqrt().recip();

            let plane_a = plane_a as usize;
            let plane_b = plane_b as usize;
            let Some(n1) = self.constraints.get(plane_a).map(|c| c.neighbor_idx) else {
                return Err(CellFailure::NoValidSeed);
            };
            let Some(n2) = self.constraints.get(plane_b).map(|c| c.neighbor_idx) else {
                return Err(CellFailure::NoValidSeed);
            };
            let n1 = n1 as u32;
            let n2 = n2 as u32;
            let key = sort3_u32(gen_idx, n1, n2);
            // SAFETY: all three vectors were cleared and reserved for
            // `poly.len` immediately above; `i` is in `0..poly.len`.
            unsafe { vertices.get_unchecked_mut(i).write((key, v_pos)) };

            let edge_plane = poly.edge_planes[i];
            if edge_plane == INVALID_PLANE_ID {
                unsafe {
                    edge_neighbor_globals.get_unchecked_mut(i).write(u32::MAX);
                    edge_neighbor_slots.get_unchecked_mut(i).write(u32::MAX);
                }
            } else {
                let edge_plane = edge_plane as usize;
                let Some(constraint) = self.constraints.get(edge_plane) else {
                    return Err(CellFailure::NoValidSeed);
                };
                unsafe {
                    edge_neighbor_globals
                        .get_unchecked_mut(i)
                        .write(constraint.neighbor_idx as u32);
                    edge_neighbor_slots
                        .get_unchecked_mut(i)
                        .write(constraint.neighbor_slot);
                }
            }
        }
        // Every spare-capacity slot above is initialized on success. On any
        // earlier error the public lengths remain zero, so partial output is
        // neither observed nor dropped as initialized data.
        unsafe {
            buffer.vertices.set_len(poly.len);
            buffer.edge_neighbor_globals.set_len(poly.len);
            buffer.edge_neighbor_slots.set_len(poly.len);
        }
        // The incremental clip keeps vertex plane pairs and edge planes in
        // lockstep (a clip's entry/exit vertices carry the pair {crossed
        // edge's plane, new plane}; interior vertices and their edges are
        // copied together), so key/edge consistency holds by construction —
        // asserted rather than paid for per edge in release.
        debug_assert!(
            edge_keys_consistent(buffer),
            "gnomonic extract broke key/edge consistency (generator {gen_idx})"
        );
        buffer.edge_keys_verified = true;

        Ok(())
    }

    pub(super) fn count_active_planes(&self) -> (usize, usize) {
        let poly = self.current_poly();
        let mut active = vec![false; self.constraints.len()];

        for i in 0..poly.len {
            let (pa, pb) = poly.vertex_planes[i];
            let pa = pa as usize;
            let pb = pb as usize;
            if pa < active.len() {
                active[pa] = true;
            }
            if pb < active.len() {
                active[pb] = true;
            }
        }

        let active_count = active.iter().filter(|&&x| x).count();
        (active_count, self.constraints.len())
    }

    pub(crate) fn debug_state(&self) -> BuilderDebugState {
        let poly = self.current_poly();
        BuilderDebugState {
            bounded: !poly.has_bounding_ref(),
            poly_len: poly.len,
            has_bounding_ref: poly.has_bounding_ref(),
            min_cos: self.chart_min_cos_bound(poly.max_r2),
            half_plane_count: self.constraints.len(),
            neighbor_index_count: self.constraints.len(),
            neighbor_slot_count: self.constraints.len(),
        }
    }

    pub(crate) fn debug_extraction_failure(&self) -> Option<ExtractionInvariantFailure> {
        if !self.is_bounded() {
            return Some(ExtractionInvariantFailure::UnboundedPolygon);
        }

        let poly = self.current_poly();
        if poly.len < 3 {
            return Some(ExtractionInvariantFailure::TooFewVertices { poly_len: poly.len });
        }

        let half_plane_count = self.constraints.len();
        let neighbor_index_count = self.constraints.len();
        let neighbor_slot_count = self.constraints.len();
        if half_plane_count != neighbor_index_count || half_plane_count != neighbor_slot_count {
            return Some(ExtractionInvariantFailure::MetadataLengthMismatch {
                half_plane_count,
                neighbor_index_count,
                neighbor_slot_count,
            });
        }

        for i in 0..poly.len {
            let u = poly.us[i];
            let v = poly.vs[i];
            if !u.is_finite() || !v.is_finite() {
                return Some(ExtractionInvariantFailure::NonFiniteProjectedVertex {
                    vertex: i,
                    u,
                    v,
                });
            }

            let (plane_a, plane_b) = poly.vertex_planes[i];
            let plane_a = plane_a as usize;
            let plane_b = plane_b as usize;
            if plane_a >= neighbor_index_count || plane_b >= neighbor_index_count {
                return Some(ExtractionInvariantFailure::InvalidVertexPlane {
                    vertex: i,
                    plane_a,
                    plane_b,
                    neighbor_index_count,
                });
            }

            let dir = DVec3::new(
                fp::fma_f64(
                    u,
                    self.basis.t1.x,
                    fp::fma_f64(v, self.basis.t2.x, self.basis.g.x),
                ),
                fp::fma_f64(
                    u,
                    self.basis.t1.y,
                    fp::fma_f64(v, self.basis.t2.y, self.basis.g.y),
                ),
                fp::fma_f64(
                    u,
                    self.basis.t1.z,
                    fp::fma_f64(v, self.basis.t2.z, self.basis.g.z),
                ),
            );
            let dir = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32);
            let len2 = dir.length_squared();
            if !len2.is_finite() || len2 < EXTRACT_DEGENERATE_LEN2 {
                return Some(ExtractionInvariantFailure::DegenerateDirection { vertex: i, len2 });
            }

            let edge_plane = poly.edge_planes[i];
            if edge_plane != INVALID_PLANE_ID {
                let edge_plane = edge_plane as usize;
                if edge_plane >= half_plane_count
                    || edge_plane >= neighbor_index_count
                    || edge_plane >= neighbor_slot_count
                {
                    return Some(ExtractionInvariantFailure::InvalidEdgePlane {
                        vertex: i,
                        edge_plane,
                        half_plane_count,
                        neighbor_index_count,
                        neighbor_slot_count,
                    });
                }
            }
        }

        None
    }
}

impl FallbackBuilder {
    /// On-plane membership tolerance for the normalized f64 fallback planes.
    /// Shared with the incremental clip's `classify_vertex`.
    pub(super) const ON_PLANE_TOL: f64 = FALLBACK_PLANE_TOL;

    pub(super) fn computed_vertex_count(&self) -> usize {
        self.poly.len()
    }

    fn satisfies_all_constraints(&self, dir: DVec3) -> bool {
        self.constraints
            .iter()
            .all(|constraint| constraint.normal.dot(dir) >= -Self::ON_PLANE_TOL)
    }

    fn shared_edge_constraint(&self, a: FallbackVertex, b: FallbackVertex) -> Option<usize> {
        for plane in [a.plane_a, a.plane_b] {
            if plane == b.plane_a || plane == b.plane_b {
                return Some(plane);
            }
        }

        self.constraints
            .iter()
            .enumerate()
            .find_map(|(idx, constraint)| {
                let normal = constraint.normal;
                if normal.dot(a.position).abs() <= Self::ON_PLANE_TOL
                    && normal.dot(b.position).abs() <= Self::ON_PLANE_TOL
                {
                    Some(idx)
                } else {
                    None
                }
            })
    }

    fn active_candidate_planes(&self) -> Vec<usize> {
        let mut planes = Vec::with_capacity(self.poly.edge_planes.len() + 8);
        for &plane in &self.poly.edge_planes {
            if plane < self.constraints.len() {
                planes.push(plane);
            }
        }

        for (idx, constraint) in self.constraints.iter().enumerate() {
            let touches =
                self.poly.vertices.iter().any(|vertex| {
                    constraint.normal.dot(vertex.position).abs() <= Self::ON_PLANE_TOL
                });
            if touches {
                planes.push(idx);
            }
        }

        planes.sort_unstable();
        planes.dedup();
        planes
    }

    fn pair_vertex(&self, plane_a: usize, plane_b: usize) -> Option<FallbackVertex> {
        let cross = self.constraints[plane_a]
            .normal
            .cross(self.constraints[plane_b].normal);
        let len2 = cross.length_squared();
        if !len2.is_finite() || len2 <= 1e-24 {
            return None;
        }

        let inv_len = len2.sqrt().recip();
        for sign in [1.0, -1.0] {
            let dir = cross * (sign * inv_len);
            if self.satisfies_all_constraints(dir) {
                return Some(FallbackVertex {
                    position: dir,
                    plane_a,
                    plane_b,
                });
            }
        }
        None
    }

    fn push_fallback_vertex(vertices: &mut Vec<FallbackVertex>, vertex: FallbackVertex) {
        if vertices
            .last()
            .is_some_and(|last| (last.position - vertex.position).length_squared() <= 1e-24)
        {
            return;
        }
        vertices.push(vertex);
    }

    fn computed_active_vertices(&self) -> Vec<FallbackVertex> {
        if self.poly.vertices.len() < 3 || self.poly.vertices.len() != self.poly.edge_planes.len() {
            return Vec::new();
        }

        let candidates = self.active_candidate_planes();
        let mut vertices = Vec::new();
        let mut used_raw_corner = false;

        for i in 0..self.poly.vertices.len() {
            let plane_a = self.poly.edge_planes
                [(i + self.poly.edge_planes.len() - 1) % self.poly.edge_planes.len()];
            let plane_b = self.poly.edge_planes[i];
            if plane_a >= self.constraints.len() || plane_b >= self.constraints.len() {
                // A bounding-box pseudo-edge (usize::MAX sentinel) survived
                // clipping: the cell is not fully bounded by real bisectors.
                // Fail the whole cell cleanly (caller maps the empty result to
                // NoValidSeed) rather than dropping this corner and emitting an
                // open vertex chain as if it were closed.
                return Vec::new();
            }
            if plane_a == plane_b {
                continue;
            }

            let corner_pos = self.poly.vertices[i].position;
            let split = candidates.iter().copied().find(|&plane_c| {
                plane_c != plane_a
                    && plane_c != plane_b
                    && self.constraints[plane_c].normal.dot(corner_pos).abs() <= Self::ON_PLANE_TOL
                    && self.pair_vertex(plane_a, plane_c).is_some()
                    && self.pair_vertex(plane_c, plane_b).is_some()
            });

            if let Some(plane_c) = split {
                if let Some(vertex) = self.pair_vertex(plane_a, plane_c) {
                    Self::push_fallback_vertex(&mut vertices, vertex);
                }
                if let Some(vertex) = self.pair_vertex(plane_c, plane_b) {
                    Self::push_fallback_vertex(&mut vertices, vertex);
                }
            } else if let Some(vertex) = self.pair_vertex(plane_a, plane_b) {
                Self::push_fallback_vertex(&mut vertices, vertex);
            } else {
                // The locally labelled pair is outside another accepted
                // constraint and no touching split plane resolves it. Retain the
                // raw corner provisionally so extraction stays available if the
                // global reconstruction below cannot form a closed cell.
                used_raw_corner = true;
                Self::push_fallback_vertex(
                    &mut vertices,
                    FallbackVertex {
                        position: self.poly.vertices[i].position,
                        plane_a,
                        plane_b,
                    },
                );
            }
        }

        if used_raw_corner {
            // A raw corner can carry a stale (plane_a, plane_b) key even when
            // its cyclic edge labels remain self-consistent, so topology-only
            // stitching cannot reliably detect the attribution error. This is
            // already a cold defect path: rebuild from every accepted pair and
            // prefer that exact attribution when it forms a closed cell. Keep
            // the provisional raw result only when the rebuild itself is not
            // usable; that last-resort fallthrough is load-bearing for known
            // highly degenerate inputs.
            let rebuilt = all_constraints_vertices(self.generator, &self.constraints);
            let rebuilt_is_closed = rebuilt.len() >= 3
                && (0..rebuilt.len()).all(|i| {
                    shared_all_constraints_edge(
                        rebuilt[i],
                        rebuilt[(i + 1) % rebuilt.len()],
                        &self.constraints,
                    )
                    .is_some()
                });
            if rebuilt_is_closed {
                return rebuilt;
            }
        }

        if vertices.len() >= 2
            && (vertices[0].position - vertices[vertices.len() - 1].position).length_squared()
                <= 1e-24
        {
            vertices.pop();
        }
        vertices
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn to_vertex_data_full(
        &self,
        buffer: &mut CellOutputBuffer,
    ) -> Result<(), CellFailure> {
        let vertices = self.computed_active_vertices();
        if vertices.len() < 3 {
            return Err(CellFailure::NoValidSeed);
        }

        buffer.clear();

        let gen_idx = self.generator_idx as u32;
        for (i, vertex) in vertices.iter().copied().enumerate() {
            let plane_a = &self.constraints[vertex.plane_a];
            let plane_b = &self.constraints[vertex.plane_b];
            let key = sort3_u32(
                gen_idx,
                plane_a.neighbor_idx as u32,
                plane_b.neighbor_idx as u32,
            );
            let position = Vec3::new(
                vertex.position.x as f32,
                vertex.position.y as f32,
                vertex.position.z as f32,
            )
            .normalize();
            buffer.vertices.push((key, position));

            let next = vertices[(i + 1) % vertices.len()];
            let Some(edge_plane) = self.shared_edge_constraint(vertex, next) else {
                return Err(CellFailure::NoValidSeed);
            };
            let edge = &self.constraints[edge_plane];
            buffer.edge_neighbor_globals.push(edge.neighbor_idx as u32);
            buffer.edge_neighbor_slots.push(edge.neighbor_slot);
        }
        // Fallback cells are rare and defect-adjacent: verify key/edge
        // consistency here (cold) so emit can record any malformed
        // attribution deterministically instead of checking every cell.
        buffer.edge_keys_verified = edge_keys_consistent(buffer);

        Ok(())
    }

    pub(super) fn count_active_planes(&self) -> (usize, usize) {
        let mut active = vec![false; self.constraints.len()];
        for &edge_plane in &self.poly.edge_planes {
            if edge_plane < active.len() {
                active[edge_plane] = true;
            }
        }
        let active_count = active.iter().filter(|&&x| x).count();
        (active_count, self.constraints.len())
    }

    pub(crate) fn debug_state(&self) -> BuilderDebugState {
        BuilderDebugState {
            bounded: self.poly.vertices.len() >= 3,
            poly_len: self.poly.vertices.len(),
            has_bounding_ref: false,
            min_cos: f64::NAN,
            half_plane_count: self.constraints.len(),
            neighbor_index_count: self.constraints.len(),
            neighbor_slot_count: self.constraints.len(),
        }
    }

    pub(crate) fn debug_extraction_failure(&self) -> Option<ExtractionInvariantFailure> {
        let vertices = &self.poly.vertices;
        if vertices.len() < 3 {
            return Some(ExtractionInvariantFailure::UnboundedPolygon);
        }
        if vertices.len() != self.poly.edge_planes.len() {
            return Some(ExtractionInvariantFailure::MetadataLengthMismatch {
                half_plane_count: self.constraints.len(),
                neighbor_index_count: self.constraints.len(),
                neighbor_slot_count: self.poly.edge_planes.len(),
            });
        }

        for (i, _) in vertices.iter().enumerate() {
            let incoming_edge = self.poly.edge_planes
                [(i + self.poly.edge_planes.len() - 1) % self.poly.edge_planes.len()];
            if incoming_edge >= self.constraints.len() {
                return Some(ExtractionInvariantFailure::InvalidVertexPlane {
                    vertex: i,
                    plane_a: incoming_edge,
                    plane_b: self.poly.edge_planes[i],
                    neighbor_index_count: self.constraints.len(),
                });
            }
            let edge_plane = self.poly.edge_planes[i];
            if edge_plane >= self.constraints.len() {
                return Some(ExtractionInvariantFailure::InvalidEdgePlane {
                    vertex: i,
                    edge_plane,
                    half_plane_count: self.constraints.len(),
                    neighbor_index_count: self.constraints.len(),
                    neighbor_slot_count: self.constraints.len(),
                });
            }
        }

        None
    }
}

impl Topo2DBuilder {
    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn to_vertex_data_full(&self, buffer: &mut CellOutputBuffer) -> Result<(), CellFailure> {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.to_vertex_data_full(buffer),
            BuilderImpl::Fallback(builder) => builder.to_vertex_data_full(buffer),
        }
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(crate) fn to_vertex_data_from_all_constraints(
        &self,
        points: &[Vec3],
        buffer: &mut CellOutputBuffer,
    ) -> Result<(), CellFailure> {
        let constraints = self.accepted_spherical_constraints(points);
        extract_all_constraints_cell(self.generator_idx(), self.generator(), &constraints, buffer)
    }

    pub fn count_active_planes(&self) -> (usize, usize) {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.count_active_planes(),
            BuilderImpl::Fallback(builder) => builder.count_active_planes(),
        }
    }

    pub(crate) fn debug_state(&self) -> BuilderDebugState {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.debug_state(),
            BuilderImpl::Fallback(builder) => builder.debug_state(),
        }
    }

    pub(crate) fn debug_extraction_failure(&self) -> Option<ExtractionInvariantFailure> {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.debug_extraction_failure(),
            BuilderImpl::Fallback(builder) => builder.debug_extraction_failure(),
        }
    }
}

fn extract_all_constraints_cell(
    generator_idx: usize,
    generator: DVec3,
    constraints: &[super::FallbackConstraint],
    buffer: &mut CellOutputBuffer,
) -> Result<(), CellFailure> {
    let vertices = all_constraints_vertices(generator, constraints);
    if vertices.len() < 3 {
        return Err(CellFailure::NoValidSeed);
    }

    buffer.clear();
    buffer.vertices.reserve(vertices.len());
    buffer.edge_neighbor_globals.reserve(vertices.len());
    buffer.edge_neighbor_slots.reserve(vertices.len());

    let gen_idx = generator_idx as u32;
    for (i, vertex) in vertices.iter().copied().enumerate() {
        let plane_a = &constraints[vertex.plane_a];
        let plane_b = &constraints[vertex.plane_b];
        let position = Vec3::new(
            vertex.position.x as f32,
            vertex.position.y as f32,
            vertex.position.z as f32,
        )
        .normalize();
        buffer.vertices.push((
            sort3_u32(
                gen_idx,
                plane_a.neighbor_idx as u32,
                plane_b.neighbor_idx as u32,
            ),
            position,
        ));

        let next = vertices[(i + 1) % vertices.len()];
        let Some(edge_plane) = shared_all_constraints_edge(vertex, next, constraints) else {
            buffer.clear();
            return Err(CellFailure::NoValidSeed);
        };
        let edge = &constraints[edge_plane];
        buffer.edge_neighbor_globals.push(edge.neighbor_idx as u32);
        buffer.edge_neighbor_slots.push(edge.neighbor_slot);
    }
    // Same rare-path verification as the fallback extract (see
    // `edge_keys_consistent`): the all-pairs reconstruction shares the
    // position-dedup wrong-attribution hazard.
    buffer.edge_keys_verified = edge_keys_consistent(buffer);

    Ok(())
}

fn all_constraints_vertices(
    generator: DVec3,
    constraints: &[super::FallbackConstraint],
) -> Vec<FallbackVertex> {
    if constraints.len() < 3 {
        return Vec::new();
    }

    let mut vertices = Vec::new();
    for a in 0..constraints.len() {
        for b in a + 1..constraints.len() {
            let cross = constraints[a].normal.cross(constraints[b].normal);
            let len2 = cross.length_squared();
            if !len2.is_finite() || len2 <= 1e-24 {
                continue;
            }

            let inv_len = len2.sqrt().recip();
            for sign in [1.0, -1.0] {
                let position = cross * (sign * inv_len);
                if constraints.iter().all(|constraint| {
                    constraint.normal.dot(position) >= -FallbackBuilder::ON_PLANE_TOL
                }) {
                    push_all_constraints_vertex(
                        &mut vertices,
                        FallbackVertex {
                            position,
                            plane_a: a,
                            plane_b: b,
                        },
                    );
                }
            }
        }
    }

    let basis = super::TangentBasis::new(generator);
    vertices.sort_by(|a, b| {
        let aa = a.position.dot(basis.t2).atan2(a.position.dot(basis.t1));
        let ab = b.position.dot(basis.t2).atan2(b.position.dot(basis.t1));
        aa.total_cmp(&ab)
    });
    vertices
}

fn push_all_constraints_vertex(vertices: &mut Vec<FallbackVertex>, vertex: FallbackVertex) {
    if vertices
        .iter()
        .any(|existing| (existing.position - vertex.position).length_squared() <= 1e-24)
    {
        return;
    }
    vertices.push(vertex);
}

fn shared_all_constraints_edge(
    a: FallbackVertex,
    b: FallbackVertex,
    constraints: &[super::FallbackConstraint],
) -> Option<usize> {
    if let Some(plane) = [a.plane_a, a.plane_b]
        .into_iter()
        .find(|&plane| plane == b.plane_a || plane == b.plane_b)
    {
        return Some(plane);
    }

    constraints
        .iter()
        .enumerate()
        .find_map(|(idx, constraint)| {
            let normal = constraint.normal;
            if normal.dot(a.position).abs() <= FallbackBuilder::ON_PLANE_TOL
                && normal.dot(b.position).abs() <= FallbackBuilder::ON_PLANE_TOL
            {
                Some(idx)
            } else {
                None
            }
        })
}
