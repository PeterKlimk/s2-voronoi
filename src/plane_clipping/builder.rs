//! Planar cell builder: rect-seeded incremental half-plane clipping.

use glam::{DVec2, Vec2, Vec3};

use crate::fp;
use crate::knn_clipping::cell_build::{CellFailure, CellOutputBuffer, VertexKey};
use crate::knn_clipping::topo2d::builder::sort3_u32;
use crate::knn_clipping::topo2d::clippers::{clip_convex, clip_convex_edgecheck};
use crate::knn_clipping::topo2d::types::{ClipResult, HalfPlane, PolyBuffer};
use crate::tolerances::PLANE_TERMINATION_GUARD;

/// One extracted vertex: key (sorted generator/wall triple) and position in
/// the unit square.
// The Vec2-native extraction path is exercised by builder tests today and
// becomes the production path when live-dedup positions genericize (the
// pipeline currently uses the z=0 `to_vertex_data_v3` embedding). Kept
// compiled in all profiles to avoid dev-only rot.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) type PlaneVertexData = (VertexKey, Vec2);

/// Per-cell extraction output, mirroring `cell_build::CellOutputBuffer` with
/// planar positions.
#[derive(Default)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct PlaneCellOutputBuffer {
    pub(crate) vertices: Vec<PlaneVertexData>,
    pub(crate) edge_neighbor_globals: Vec<u32>,
    pub(crate) edge_neighbor_slots: Vec<u32>,
    pub(crate) edge_neighbor_eps: Vec<f32>,
}

impl PlaneCellOutputBuffer {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn clear(&mut self) {
        self.vertices.clear();
        self.edge_neighbor_globals.clear();
        self.edge_neighbor_slots.clear();
        self.edge_neighbor_eps.clear();
    }
}

/// Incremental planar Voronoi cell builder.
///
/// Chart coordinates are `(u, v) = point - generator` in f64 (f32 inputs,
/// f64 math). The polygon is seeded with the unit-square walls, so it is
/// bounded and `>= 3` vertices from the start; the only failure modes are
/// the vertex budget and a fully clipped-away cell (coincident generators).
pub(crate) struct PlaneCellBuilder {
    pub(crate) generator_idx: usize,
    generator: DVec2,
    /// First virtual wall id (= generator count); walls are
    /// `wall_base + WALL_*`.
    wall_base: u32,
    /// Domain extents: walls sit at x=0, x=domain.x, y=0, y=domain.y
    /// (normalized coordinates; the longer rect side maps to 1).
    domain: DVec2,

    half_planes: Vec<HalfPlane>,
    neighbor_indices: Vec<usize>,
    neighbor_slots: Vec<u32>,

    poly_a: PolyBuffer,
    poly_b: PolyBuffer,
    use_a: bool,

    failed: Option<CellFailure>,
    term_threshold_cache: f64,
    term_cache_valid: bool,
}

impl PlaneCellBuilder {
    pub(crate) fn new(generator_idx: usize, generator: Vec2, wall_base: u32, domain: Vec2) -> Self {
        let mut builder = Self {
            generator_idx,
            generator: DVec2::new(generator.x as f64, generator.y as f64),
            wall_base,
            domain: DVec2::new(domain.x as f64, domain.y as f64),
            half_planes: Vec::with_capacity(32),
            neighbor_indices: Vec::with_capacity(32),
            neighbor_slots: Vec::with_capacity(32),
            poly_a: PolyBuffer::new(),
            poly_b: PolyBuffer::new(),
            use_a: true,
            failed: None,
            term_threshold_cache: 0.0,
            term_cache_valid: false,
        };
        builder.seed_rect();
        builder
    }

    pub(crate) fn reset(&mut self, generator_idx: usize, generator: Vec2) {
        self.generator_idx = generator_idx;
        self.generator = DVec2::new(generator.x as f64, generator.y as f64);
        self.half_planes.clear();
        self.neighbor_indices.clear();
        self.neighbor_slots.clear();
        self.poly_b.clear();
        self.use_a = true;
        self.failed = None;
        self.term_cache_valid = false;
        self.seed_rect();
    }

    /// Seed the polygon with the unit-square walls (planes 0..4, neighbors
    /// `wall_base + WALL_*`), in generator-centered chart coordinates.
    fn seed_rect(&mut self) {
        let (gx, gy) = (self.generator.x, self.generator.y);

        let (sx, sy) = (self.domain.x, self.domain.y);
        // (a, b, c) with interior satisfying a*u + b*v + c >= 0:
        //   bottom (y >= 0):   v + gy >= 0
        //   right  (x <= sx): -u + (sx - gx) >= 0
        //   top    (y <= sy): -v + (sy - gy) >= 0
        //   left   (x >= 0):   u + gx >= 0
        use super::{WALL_BOTTOM, WALL_LEFT, WALL_RIGHT, WALL_TOP};
        let walls: [(u32, f64, f64, f64); 4] = [
            (WALL_BOTTOM, 0.0, 1.0, gy),
            (WALL_RIGHT, -1.0, 0.0, sx - gx),
            (WALL_TOP, 0.0, -1.0, sy - gy),
            (WALL_LEFT, 1.0, 0.0, gx),
        ];
        for (plane_idx, &(side, a, b, c)) in walls.iter().enumerate() {
            self.half_planes
                .push(HalfPlane::new_unnormalized(a, b, c, plane_idx));
            self.neighbor_indices.push((self.wall_base + side) as usize);
            self.neighbor_slots.push(u32::MAX);
        }

        // Corners CCW from bottom-left; vertex_planes = the two adjacent
        // walls, edge_planes[i] = wall of edge i -> i+1.
        let (u0, u1) = (-gx, sx - gx);
        let (v0, v1) = (-gy, sy - gy);
        let poly = &mut self.poly_a;
        poly.clear();
        poly.push_raw(u0, v0, (3, 0), 0);
        poly.push_raw(u1, v0, (0, 1), 1);
        poly.push_raw(u1, v1, (1, 2), 2);
        poly.push_raw(u0, v1, (2, 3), 3);
        let r2 = |u: f64, v: f64| fp::fma_f64(u, u, v * v);
        poly.max_r2 = r2(u0, v0).max(r2(u1, v0)).max(r2(u1, v1)).max(r2(u0, v1));
        poly.has_bounding_ref = false;
    }

    #[inline]
    fn current_poly(&self) -> &PolyBuffer {
        if self.use_a {
            &self.poly_a
        } else {
            &self.poly_b
        }
    }

    /// Bisector half-plane coefficients for a neighbor: points no farther
    /// from the generator than from the neighbor, in chart coordinates.
    #[inline]
    fn bisector_coefficients(&self, neighbor: Vec2) -> (f64, f64, f64) {
        let qu = neighbor.x as f64 - self.generator.x;
        let qv = neighbor.y as f64 - self.generator.y;
        // |p|^2 <= |p - q|^2  <=>  -qu*u - qv*v + (qu^2 + qv^2)/2 >= 0
        let c = 0.5 * fp::fma_f64(qu, qu, qv * qv);
        (-qu, -qv, c)
    }

    pub(crate) fn clip_with_slot_result(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec2,
    ) -> Result<ClipResult, CellFailure> {
        if let Some(f) = self.failed {
            return Err(f);
        }

        let (a, b, c) = self.bisector_coefficients(neighbor);
        let plane_idx = self.half_planes.len();
        let hp = HalfPlane::new_unnormalized(a, b, c, plane_idx);

        let clip_result = if self.use_a {
            clip_convex(&self.poly_a, &hp, &mut self.poly_b)
        } else {
            clip_convex(&self.poly_b, &hp, &mut self.poly_a)
        };

        self.commit_clip(clip_result, hp, neighbor_idx, neighbor_slot)
    }

    fn commit_clip(
        &mut self,
        clip_result: ClipResult,
        hp: HalfPlane,
        neighbor_idx: usize,
        neighbor_slot: u32,
    ) -> Result<ClipResult, CellFailure> {
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
                self.term_cache_valid = false;
            }
            ClipResult::Unchanged => {}
        }

        if self.current_poly().len < 3 {
            self.failed = Some(CellFailure::ClippedAway);
            return Err(CellFailure::ClippedAway);
        }

        Ok(clip_result)
    }

    pub(crate) fn clip_with_slot(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec2,
    ) -> Result<(), CellFailure> {
        self.clip_with_slot_result(neighbor_idx, neighbor_slot, neighbor)
            .map(|_| ())
    }

    /// Clip with an externally supplied epsilon (edge-check seeds): reusing
    /// the epsilon computed on the opposite side of the same undirected edge
    /// keeps both cells' near-degenerate decisions consistent.
    pub(crate) fn clip_with_slot_edgecheck(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec2,
        hp_eps: f32,
    ) -> Result<(), CellFailure> {
        if !hp_eps.is_finite() || hp_eps <= 0.0 {
            return self.clip_with_slot(neighbor_idx, neighbor_slot, neighbor);
        }
        if let Some(f) = self.failed {
            return Err(f);
        }

        let (a, b, c) = self.bisector_coefficients(neighbor);
        let plane_idx = self.half_planes.len();
        let hp = HalfPlane::new_unnormalized_with_eps(a, b, c, plane_idx, hp_eps as f64);

        let clip_result = if self.use_a {
            clip_convex_edgecheck(&self.poly_a, &hp, &mut self.poly_b)
        } else {
            clip_convex_edgecheck(&self.poly_b, &hp, &mut self.poly_a)
        };
        self.commit_clip(clip_result, hp, neighbor_idx, neighbor_slot)
            .map(|_| ())
    }

    // Failure-classification surface (sphere parity); the pipeline's
    // failure reporting grows into it.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn is_failed(&self) -> bool {
        self.failed.is_some()
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn failure(&self) -> Option<CellFailure> {
        self.failed
    }

    #[inline]
    pub(crate) fn vertex_count(&self) -> usize {
        self.current_poly().len
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn neighbor_indices_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbor_indices.iter().copied()
    }

    /// Sound termination: a neighbor at squared distance `d2` cuts the cell
    /// only if its bisector (at distance `sqrt(d2)/2` from the generator)
    /// reaches inside the polygon's vertex radius, i.e. `d2 <= 4 * max_r2`.
    /// `min_unseen_dist_sq_bound` must lower-bound the squared distance of
    /// every neighbor not yet clipped.
    pub(crate) fn can_terminate(&mut self, min_unseen_dist_sq_bound: f32) -> bool {
        if self.failed.is_some() || self.vertex_count() < 3 {
            return false;
        }

        if !self.term_cache_valid {
            let max_r2 = self.current_poly().max_r2;
            self.term_threshold_cache = 4.0 * max_r2 * (1.0 + PLANE_TERMINATION_GUARD);
            self.term_cache_valid = true;
        }

        (min_unseen_dist_sq_bound as f64) > self.term_threshold_cache
    }

    /// Extract vertices (key + unit-square position) and per-edge neighbor
    /// records, walking the polygon in order.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn to_vertex_data_full(
        &self,
        buffer: &mut PlaneCellOutputBuffer,
    ) -> Result<(), CellFailure> {
        let poly = self.current_poly();
        if self.failed.is_some() || poly.len < 3 {
            return Err(CellFailure::NoValidSeed);
        }

        buffer.clear();
        buffer.vertices.reserve(poly.len);
        buffer.edge_neighbor_globals.reserve(poly.len);
        buffer.edge_neighbor_slots.reserve(poly.len);
        buffer.edge_neighbor_eps.reserve(poly.len);

        let gen_idx = self.generator_idx as u32;
        let plane_count = self.half_planes.len();
        for i in 0..poly.len {
            let u = poly.us[i];
            let v = poly.vs[i];
            if !u.is_finite() || !v.is_finite() {
                return Err(CellFailure::NoValidSeed);
            }

            let (plane_a, plane_b) = poly.vertex_planes[i];
            let edge_plane = poly.edge_planes[i];
            // The rect seed registers the walls as planes 0..4, so every
            // vertex and edge of a planar polygon references a real plane.
            if plane_a >= plane_count || plane_b >= plane_count || edge_plane >= plane_count {
                return Err(CellFailure::NoValidSeed);
            }

            let pos = Vec2::new((u + self.generator.x) as f32, (v + self.generator.y) as f32);
            let n1 = self.neighbor_indices[plane_a] as u32;
            let n2 = self.neighbor_indices[plane_b] as u32;
            buffer.vertices.push((sort3_u32(gen_idx, n1, n2), pos));

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

        Ok(())
    }

    /// Extract into the shared `CellOutputBuffer` with positions embedded as
    /// `Vec3 {x, y, 0}` — the temporary z=0 embedding that lets the planar
    /// pipeline drive the (geometry-agnostic) live-dedup layer unchanged,
    /// pending position-type genericization.
    ///
    /// Wall edges are emitted with the `u32::MAX` "no stitchable neighbor"
    /// convention (edge checks skip them); wall ids still appear in vertex
    /// *keys*, where dedup treats them as ordinary generator indices.
    pub(crate) fn to_vertex_data_v3(
        &self,
        buffer: &mut CellOutputBuffer,
    ) -> Result<(), CellFailure> {
        let poly = self.current_poly();
        if self.failed.is_some() || poly.len < 3 {
            return Err(CellFailure::NoValidSeed);
        }

        buffer.clear();
        buffer.vertices.reserve(poly.len);
        buffer.edge_neighbor_globals.reserve(poly.len);
        buffer.edge_neighbor_slots.reserve(poly.len);
        buffer.edge_neighbor_eps.reserve(poly.len);

        let gen_idx = self.generator_idx as u32;
        let plane_count = self.half_planes.len();
        for i in 0..poly.len {
            let u = poly.us[i];
            let v = poly.vs[i];
            if !u.is_finite() || !v.is_finite() {
                return Err(CellFailure::NoValidSeed);
            }

            let (plane_a, plane_b) = poly.vertex_planes[i];
            let edge_plane = poly.edge_planes[i];
            if plane_a >= plane_count || plane_b >= plane_count || edge_plane >= plane_count {
                return Err(CellFailure::NoValidSeed);
            }

            let pos = Vec3::new(
                (u + self.generator.x) as f32,
                (v + self.generator.y) as f32,
                0.0,
            );
            let n1 = self.neighbor_indices[plane_a] as u32;
            let n2 = self.neighbor_indices[plane_b] as u32;
            buffer.vertices.push((sort3_u32(gen_idx, n1, n2), pos));

            let edge_neighbor = self.neighbor_indices[edge_plane] as u32;
            if edge_neighbor >= self.wall_base {
                buffer.edge_neighbor_globals.push(u32::MAX);
                buffer.edge_neighbor_slots.push(u32::MAX);
                buffer.edge_neighbor_eps.push(0.0);
            } else {
                buffer.edge_neighbor_globals.push(edge_neighbor);
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
}
