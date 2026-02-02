use super::clippers::{clip_convex, clip_convex_edgecheck};
use super::types::{ClipResult, HalfPlane, PolyBuffer};
use crate::fp;
use crate::knn_clipping::cell_builder::{CellFailure, VertexData};
use glam::{DVec3, Vec3};

/// Orthonormal tangent basis for gnomonic projection.
pub struct TangentBasis {
    pub t1: DVec3,
    pub t2: DVec3,
    pub g: DVec3,
}

impl TangentBasis {
    pub fn new(g: DVec3) -> Self {
        let arbitrary = if g.x.abs() <= g.y.abs() && g.x.abs() <= g.z.abs() {
            DVec3::X
        } else if g.y.abs() <= g.z.abs() {
            DVec3::Y
        } else {
            DVec3::Z
        };
        let t1 = g.cross(arbitrary).normalize();
        let t2 = g.cross(t1);
        TangentBasis { t1, t2, g }
    }

    #[inline]
    pub fn plane_to_line(&self, n: DVec3) -> (f64, f64, f64) {
        (n.dot(self.t1), n.dot(self.t2), n.dot(self.g))
    }
}

pub struct Topo2DBuilder {
    pub(crate) generator_idx: usize,
    pub(crate) generator: DVec3,
    pub(crate) basis: TangentBasis,

    half_planes: Vec<HalfPlane>,
    neighbor_indices: Vec<usize>,
    neighbor_slots: Vec<u32>,

    poly_a: PolyBuffer,
    poly_b: PolyBuffer,
    use_a: bool,

    failed: Option<CellFailure>,
    term_sin_pad: f64,
    term_cos_pad: f64,
}

impl Topo2DBuilder {
    pub fn new(generator_idx: usize, generator: Vec3) -> Self {
        let angle_pad = 8.0 * f32::EPSILON as f64;
        let (term_sin_pad, term_cos_pad) = angle_pad.sin_cos();
        let gen64 =
            DVec3::new(generator.x as f64, generator.y as f64, generator.z as f64).normalize();
        let basis = TangentBasis::new(gen64);

        let mut poly_a = PolyBuffer::new();
        poly_a.init_bounding(1e6);

        Self {
            generator_idx,
            generator: gen64,
            basis,
            half_planes: Vec::with_capacity(32),
            neighbor_indices: Vec::with_capacity(32),
            neighbor_slots: Vec::with_capacity(32),
            poly_a,
            poly_b: PolyBuffer::new(),
            use_a: true,
            failed: None,
            term_sin_pad,
            term_cos_pad,
        }
    }

    pub fn reset(&mut self, generator_idx: usize, generator: Vec3) {
        let gen64 =
            DVec3::new(generator.x as f64, generator.y as f64, generator.z as f64).normalize();
        self.generator_idx = generator_idx;
        self.generator = gen64;
        self.basis = TangentBasis::new(gen64);
        self.half_planes.clear();
        self.neighbor_indices.clear();
        self.neighbor_slots.clear();
        self.poly_a.init_bounding(1e6);
        self.poly_b.clear();
        self.use_a = true;
        self.failed = None;
    }

    pub fn clip_with_slot(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<(), CellFailure> {
        if let Some(f) = self.failed {
            return Err(f);
        }

        debug_assert!(
            (neighbor.length_squared() - 1.0).abs() < 1e-5,
            "neighbor not unit-normalized: |N|² = {}",
            neighbor.length_squared()
        );

        let n_raw = DVec3::new(neighbor.x as f64, neighbor.y as f64, neighbor.z as f64);
        let len_sq = n_raw.length_squared();
        let scale = fp::fma_f64(len_sq, 0.5, 0.5);

        let g = self.generator;
        let normal_unnorm = DVec3::new(
            fp::fma_f64(g.x, scale, -n_raw.x),
            fp::fma_f64(g.y, scale, -n_raw.y),
            fp::fma_f64(g.z, scale, -n_raw.z),
        );

        let (a, b, c) = self.basis.plane_to_line(normal_unnorm);
        let plane_idx = self.half_planes.len();
        let hp = HalfPlane::new_unnormalized(a, b, c, plane_idx);

        let clip_result = if self.use_a {
            clip_convex(&self.poly_a, &hp, &mut self.poly_b)
        } else {
            clip_convex(&self.poly_b, &hp, &mut self.poly_a)
        };

        match clip_result {
            ClipResult::TooManyVertices => {
                self.failed = Some(CellFailure::TooManyVertices);
                return Err(CellFailure::TooManyVertices);
            }
            ClipResult::Changed => {
                self.half_planes.push(hp);
                self.neighbor_indices.push(neighbor_idx);
                self.neighbor_slots.push(neighbor_slot);
                self.use_a = !self.use_a;
            }
            ClipResult::Unchanged => {}
        }

        let poly = self.current_poly();
        if poly.len < 3 {
            self.failed = Some(CellFailure::ClippedAway);
            return Err(CellFailure::ClippedAway);
        }

        Ok(())
    }

    pub fn clip_with_slot_edgecheck(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
        hp_eps: f32,
    ) -> Result<(), CellFailure> {
        if !hp_eps.is_finite() || hp_eps <= 0.0 {
            return self.clip_with_slot(neighbor_idx, neighbor_slot, neighbor);
        }
        if let Some(f) = self.failed {
            return Err(f);
        }

        debug_assert!(
            (neighbor.length_squared() - 1.0).abs() < 1e-5,
            "neighbor not unit-normalized: |N|² = {}",
            neighbor.length_squared()
        );

        let n_raw = DVec3::new(neighbor.x as f64, neighbor.y as f64, neighbor.z as f64);
        let len_sq = n_raw.length_squared();
        let scale = fp::fma_f64(len_sq, 0.5, 0.5);

        let g = self.generator;
        let normal_unnorm = DVec3::new(
            fp::fma_f64(g.x, scale, -n_raw.x),
            fp::fma_f64(g.y, scale, -n_raw.y),
            fp::fma_f64(g.z, scale, -n_raw.z),
        );

        let (a, b, c) = self.basis.plane_to_line(normal_unnorm);
        let plane_idx = self.half_planes.len();
        let hp = HalfPlane::new_unnormalized_with_eps(a, b, c, plane_idx, hp_eps as f64);

        let clip_result = if self.use_a {
            clip_convex_edgecheck(&self.poly_a, &hp, &mut self.poly_b)
        } else {
            clip_convex_edgecheck(&self.poly_b, &hp, &mut self.poly_a)
        };

        match clip_result {
            ClipResult::TooManyVertices => {
                self.failed = Some(CellFailure::TooManyVertices);
                return Err(CellFailure::TooManyVertices);
            }
            ClipResult::Changed => {
                self.half_planes.push(hp);
                self.neighbor_indices.push(neighbor_idx);
                self.neighbor_slots.push(neighbor_slot);
                self.use_a = !self.use_a;
            }
            ClipResult::Unchanged => {}
        }

        let poly = self.current_poly();
        if poly.len < 3 {
            self.failed = Some(CellFailure::ClippedAway);
            return Err(CellFailure::ClippedAway);
        }

        Ok(())
    }

    #[inline]
    fn current_poly(&self) -> &PolyBuffer {
        if self.use_a {
            &self.poly_a
        } else {
            &self.poly_b
        }
    }

    #[inline]
    pub fn is_bounded(&self) -> bool {
        !self.current_poly().has_bounding_ref()
    }

    #[inline]
    pub fn is_failed(&self) -> bool {
        self.failed.is_some()
    }

    #[inline]
    pub fn failure(&self) -> Option<CellFailure> {
        self.failed
    }

    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.current_poly().len
    }

    #[inline]
    pub fn has_neighbor(&self, neighbor_idx: usize) -> bool {
        self.neighbor_indices.contains(&neighbor_idx)
    }

    #[inline]
    pub fn neighbor_indices_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbor_indices.iter().copied()
    }

    pub fn can_terminate(&mut self, max_unseen_dot_bound: f32) -> bool {
        if !self.is_bounded() || self.vertex_count() < 3 {
            return false;
        }

        let min_cos = self.current_poly().min_cos();
        if min_cos <= 0.0 || min_cos > 1.0 {
            return false;
        }

        let sin_theta = (1.0 - min_cos * min_cos).max(0.0).sqrt();
        let cos_theta_pad = fp::fma_f64(min_cos, self.term_cos_pad, -sin_theta * self.term_sin_pad);
        let cos_2max = fp::fma_f64(2.0 * cos_theta_pad, cos_theta_pad, -1.0);
        let threshold = cos_2max - 3.0 * f32::EPSILON as f64;

        (max_unseen_dot_bound as f64) < threshold
    }

    pub fn to_vertex_data_full(
        &self,
        out: &mut Vec<VertexData>,
        edge_neighbors: &mut Vec<u32>,
        edge_neighbor_slots: &mut Vec<u32>,
        edge_neighbor_eps: &mut Vec<f32>,
    ) -> Result<(), CellFailure> {
        self.to_vertex_data_impl(
            out,
            Some(edge_neighbors),
            Some(edge_neighbor_slots),
            Some(edge_neighbor_eps),
        )
    }

    fn to_vertex_data_impl(
        &self,
        out: &mut Vec<VertexData>,
        mut edge_neighbors: Option<&mut Vec<u32>>,
        mut edge_neighbor_slots: Option<&mut Vec<u32>>,
        mut edge_neighbor_eps: Option<&mut Vec<f32>>,
    ) -> Result<(), CellFailure> {
        if !self.is_bounded() {
            return Err(CellFailure::NoValidSeed);
        }

        let poly = self.current_poly();
        if poly.len < 3 {
            return Err(CellFailure::NoValidSeed);
        }

        out.clear();
        out.reserve(poly.len);
        if let Some(edge_neighbors) = edge_neighbors.as_deref_mut() {
            edge_neighbors.clear();
            edge_neighbors.reserve(poly.len);
        }
        if let Some(edge_neighbor_slots) = edge_neighbor_slots.as_deref_mut() {
            edge_neighbor_slots.clear();
            edge_neighbor_slots.reserve(poly.len);
        }
        if let Some(edge_neighbor_eps) = edge_neighbor_eps.as_deref_mut() {
            edge_neighbor_eps.clear();
            edge_neighbor_eps.reserve(poly.len);
        }

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
            let len2 = dir.length_squared();
            if len2 < 1e-28 {
                return Err(CellFailure::NoValidSeed);
            }
            let inv_len = len2.sqrt().recip();
            let v_pos = dir * inv_len;

            let n1 = self.neighbor_indices[plane_a] as u32;
            let n2 = self.neighbor_indices[plane_b] as u32;

            let mut key = [gen_idx, n1, n2];
            key.sort();

            out.push((key, v_pos.as_vec3()));

            if let Some(edge_neighbors) = edge_neighbors.as_deref_mut() {
                let edge_plane = poly.edge_planes[i];
                let neighbor = if edge_plane == usize::MAX {
                    u32::MAX
                } else {
                    self.neighbor_indices[edge_plane] as u32
                };
                edge_neighbors.push(neighbor);
            }
            if let Some(edge_neighbor_slots) = edge_neighbor_slots.as_deref_mut() {
                let edge_plane = poly.edge_planes[i];
                let slot = if edge_plane == usize::MAX {
                    u32::MAX
                } else {
                    self.neighbor_slots[edge_plane]
                };
                edge_neighbor_slots.push(slot);
            }
            if let Some(edge_neighbor_eps) = edge_neighbor_eps.as_deref_mut() {
                let edge_plane = poly.edge_planes[i];
                let eps = if edge_plane == usize::MAX {
                    0.0
                } else {
                    self.half_planes[edge_plane].eps as f32
                };
                edge_neighbor_eps.push(eps);
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
}
