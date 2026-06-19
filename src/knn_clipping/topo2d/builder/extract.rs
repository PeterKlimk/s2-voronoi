use super::projection::sort3_u32;
use super::{
    BuilderDebugState, BuilderImpl, ExtractionInvariantFailure, FallbackBuilder, GnomonicBuilder,
    TangentBasis, Topo2DBuilder,
};
use crate::fp;
use crate::knn_clipping::cell_build::{CellFailure, CellOutputBuffer};
use crate::knn_clipping::topo2d::types::INVALID_PLANE_ID;
use glam::{DVec3, Vec3};

use crate::tolerances::{EXTRACT_DEGENERATE_LEN2, FALLBACK_DEDUP_DOT, FALLBACK_PLANE_TOL};

#[derive(Clone, Copy)]
struct FallbackVertex {
    position: Vec3,
    plane_a: usize,
    plane_b: usize,
    angle: f64,
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
        let half_plane_count = self.half_planes.len();
        let neighbor_index_count = self.neighbor_indices.len();
        let neighbor_slot_count = self.neighbor_slots.len();
        if half_plane_count != neighbor_index_count || half_plane_count != neighbor_slot_count {
            return Err(CellFailure::NoValidSeed);
        }

        buffer.clear();
        buffer.vertices.reserve(poly.len);
        buffer.edge_neighbor_globals.reserve(poly.len);
        buffer.edge_neighbor_slots.reserve(poly.len);
        buffer.edge_neighbor_eps.reserve(poly.len);

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
            let Some(&n1) = self.neighbor_indices.get(plane_a) else {
                return Err(CellFailure::NoValidSeed);
            };
            let Some(&n2) = self.neighbor_indices.get(plane_b) else {
                return Err(CellFailure::NoValidSeed);
            };
            let n1 = n1 as u32;
            let n2 = n2 as u32;
            let key = sort3_u32(gen_idx, n1, n2);
            buffer.vertices.push((key, v_pos));

            let edge_plane = poly.edge_planes[i];
            if edge_plane == INVALID_PLANE_ID {
                buffer.edge_neighbor_globals.push(u32::MAX);
                buffer.edge_neighbor_slots.push(u32::MAX);
                buffer.edge_neighbor_eps.push(0.0);
            } else {
                let edge_plane = edge_plane as usize;
                let Some(&edge_neighbor) = self.neighbor_indices.get(edge_plane) else {
                    return Err(CellFailure::NoValidSeed);
                };
                let Some(&edge_slot) = self.neighbor_slots.get(edge_plane) else {
                    return Err(CellFailure::NoValidSeed);
                };
                let Some(edge_hp) = self.half_planes.get(edge_plane) else {
                    return Err(CellFailure::NoValidSeed);
                };
                buffer.edge_neighbor_globals.push(edge_neighbor as u32);
                buffer.edge_neighbor_slots.push(edge_slot);
                buffer.edge_neighbor_eps.push(edge_hp.eps as f32);
            }
        }

        Ok(())
    }

    pub(super) fn count_active_planes(&self) -> (usize, usize) {
        let poly = self.current_poly();
        let mut active = vec![false; self.half_planes.len()];

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
        (active_count, self.half_planes.len())
    }

    pub(crate) fn debug_state(&self) -> BuilderDebugState {
        let poly = self.current_poly();
        BuilderDebugState {
            bounded: !poly.has_bounding_ref(),
            poly_len: poly.len,
            has_bounding_ref: poly.has_bounding_ref(),
            min_cos: poly.min_cos(),
            half_plane_count: self.half_planes.len(),
            neighbor_index_count: self.neighbor_indices.len(),
            neighbor_slot_count: self.neighbor_slots.len(),
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

        let half_plane_count = self.half_planes.len();
        let neighbor_index_count = self.neighbor_indices.len();
        let neighbor_slot_count = self.neighbor_slots.len();
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
    pub(super) fn computed_vertex_count(&self) -> usize {
        self.computed_vertices().len()
    }

    fn satisfies_all_constraints(&self, dir: DVec3) -> bool {
        self.constraints
            .iter()
            .all(|constraint| constraint.normal.dot(dir) >= -FALLBACK_PLANE_TOL)
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
                let pos_a = DVec3::new(
                    a.position.x as f64,
                    a.position.y as f64,
                    a.position.z as f64,
                );
                let pos_b = DVec3::new(
                    b.position.x as f64,
                    b.position.y as f64,
                    b.position.z as f64,
                );
                if normal.dot(pos_a).abs() <= FALLBACK_PLANE_TOL
                    && normal.dot(pos_b).abs() <= FALLBACK_PLANE_TOL
                {
                    Some(idx)
                } else {
                    None
                }
            })
    }

    fn computed_vertices(&self) -> Vec<FallbackVertex> {
        if self.constraints.len() < 3 {
            return Vec::new();
        }

        let basis = TangentBasis::new(self.generator);
        let mut vertices = Vec::new();

        for plane_a in 0..self.constraints.len() {
            for plane_b in plane_a + 1..self.constraints.len() {
                let cross = self.constraints[plane_a]
                    .normal
                    .cross(self.constraints[plane_b].normal);
                let len2 = cross.length_squared();
                if !len2.is_finite() || len2 <= 1e-24 {
                    continue;
                }

                let inv_len = len2.sqrt().recip();
                for sign in [1.0, -1.0] {
                    let dir = cross * (sign * inv_len);
                    if !self.satisfies_all_constraints(dir) {
                        continue;
                    }

                    let dir32 = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32).normalize();
                    if vertices.iter().any(|vertex: &FallbackVertex| {
                        vertex.position.dot(dir32) >= FALLBACK_DEDUP_DOT
                    }) {
                        continue;
                    }

                    let angle = dir.dot(basis.t2).atan2(dir.dot(basis.t1));
                    vertices.push(FallbackVertex {
                        position: dir32,
                        plane_a,
                        plane_b,
                        angle,
                    });
                }
            }
        }

        // total_cmp: NaN angles (degenerate directions) get a deterministic
        // order instead of an inconsistent comparator, which std::sort may
        // reject with a panic since Rust 1.81.
        vertices.sort_by(|a, b| a.angle.total_cmp(&b.angle));
        vertices
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn to_vertex_data_full(
        &self,
        buffer: &mut CellOutputBuffer,
    ) -> Result<(), CellFailure> {
        let vertices = self.computed_vertices();
        if vertices.len() < 3 {
            return Err(CellFailure::NoValidSeed);
        }

        buffer.clear();
        buffer.vertices.reserve(vertices.len());
        buffer.edge_neighbor_globals.reserve(vertices.len());
        buffer.edge_neighbor_slots.reserve(vertices.len());
        buffer.edge_neighbor_eps.reserve(vertices.len());

        let gen_idx = self.generator_idx as u32;
        for (i, vertex) in vertices.iter().copied().enumerate() {
            let plane_a = &self.constraints[vertex.plane_a];
            let plane_b = &self.constraints[vertex.plane_b];
            let key = sort3_u32(
                gen_idx,
                plane_a.neighbor_idx as u32,
                plane_b.neighbor_idx as u32,
            );
            buffer.vertices.push((key, vertex.position));

            let next = vertices[(i + 1) % vertices.len()];
            let Some(edge_plane) = self.shared_edge_constraint(vertex, next) else {
                return Err(CellFailure::NoValidSeed);
            };
            let edge = &self.constraints[edge_plane];
            buffer.edge_neighbor_globals.push(edge.neighbor_idx as u32);
            buffer.edge_neighbor_slots.push(edge.neighbor_slot);
            buffer.edge_neighbor_eps.push(edge.hp_eps.unwrap_or(0.0));
        }

        Ok(())
    }

    pub(super) fn count_active_planes(&self) -> (usize, usize) {
        let vertices = self.computed_vertices();
        let mut active = vec![false; self.constraints.len()];
        for vertex in vertices {
            active[vertex.plane_a] = true;
            active[vertex.plane_b] = true;
        }
        let active_count = active.iter().filter(|&&x| x).count();
        (active_count, self.constraints.len())
    }

    pub(crate) fn debug_state(&self) -> BuilderDebugState {
        let vertices = self.computed_vertices();
        BuilderDebugState {
            bounded: vertices.len() >= 3,
            poly_len: vertices.len(),
            has_bounding_ref: false,
            min_cos: f64::NAN,
            half_plane_count: self.constraints.len(),
            neighbor_index_count: self.constraints.len(),
            neighbor_slot_count: self.constraints.len(),
        }
    }

    pub(crate) fn debug_extraction_failure(&self) -> Option<ExtractionInvariantFailure> {
        let vertices = self.computed_vertices();
        if vertices.len() < 3 {
            return Some(ExtractionInvariantFailure::UnboundedPolygon);
        }

        for i in 0..vertices.len() {
            let next = vertices[(i + 1) % vertices.len()];
            if self.shared_edge_constraint(vertices[i], next).is_none() {
                return Some(ExtractionInvariantFailure::InvalidEdgePlane {
                    vertex: i,
                    edge_plane: usize::MAX,
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
