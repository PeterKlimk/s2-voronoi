use super::projection::sort3_u32;
use super::{BuilderDebugState, ExtractionInvariantFailure, Topo2DBuilder};
use crate::fp;
use crate::knn_clipping::cell_build::{CellFailure, CellOutputBuffer};
use glam::{DVec3, Vec3};

impl Topo2DBuilder {
    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn to_vertex_data_full(&self, buffer: &mut CellOutputBuffer) -> Result<(), CellFailure> {
        let poly = self.current_poly();
        if self.debug_extraction_failure().is_some() {
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
            if len2 < 1e-28 {
                return Err(CellFailure::NoValidSeed);
            }
            let v_pos = dir * len2.sqrt().recip();

            let n1 = self.neighbor_indices[plane_a] as u32;
            let n2 = self.neighbor_indices[plane_b] as u32;
            let key = sort3_u32(gen_idx, n1, n2);
            buffer.vertices.push((key, v_pos));

            let edge_plane = poly.edge_planes[i];
            if edge_plane == usize::MAX {
                buffer.edge_neighbor_globals.push(u32::MAX);
                buffer.edge_neighbor_slots.push(u32::MAX);
                buffer.edge_neighbor_eps.push(0.0);
            } else {
                buffer
                    .edge_neighbor_globals
                    .push(self.neighbor_indices[edge_plane] as u32);
                buffer
                    .edge_neighbor_slots
                    .push(self.neighbor_slots[edge_plane]);
                buffer
                    .edge_neighbor_eps
                    .push(self.half_planes[edge_plane].eps as f32);
            }
        }

        Ok(())
    }

    pub fn count_active_planes(&self) -> (usize, usize) {
        let poly = self.current_poly();
        let mut active = vec![false; self.half_planes.len()];

        for i in 0..poly.len {
            let (pa, pb) = poly.vertex_planes[i];
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
            if !len2.is_finite() || len2 < 1e-28 {
                return Some(ExtractionInvariantFailure::DegenerateDirection { vertex: i, len2 });
            }

            let edge_plane = poly.edge_planes[i];
            if edge_plane != usize::MAX
                && (edge_plane >= half_plane_count
                    || edge_plane >= neighbor_index_count
                    || edge_plane >= neighbor_slot_count)
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

        None
    }
}
