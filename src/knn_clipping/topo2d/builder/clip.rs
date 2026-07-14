use super::{
    BuilderClipOutcome, BuilderImpl, BuilderStepOutcome, FallbackBuilder, GnomonicBuilder,
    GnomonicConstraint, SphericalPoly, SphericalPolyVertex, Topo2DBuilder,
};
use crate::knn_clipping::cell_build::CellFailure;
use crate::knn_clipping::topo2d::clippers::{clip_convex, clip_convex_edgecheck};
use crate::knn_clipping::topo2d::types::{ClipResult, HalfPlane};
use glam::Vec3;

impl GnomonicBuilder {
    #[cfg(test)]
    pub(super) fn clip_with_slot(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<(), CellFailure> {
        self.clip_with_slot_result(neighbor_idx, neighbor_slot, neighbor)
            .map(|_| ())
    }

    /// Commit a clip result: update state on Changed, set failure on TooManyVertices/ClippedAway.
    #[cfg_attr(not(feature = "profiling"), inline(always))]
    pub(super) fn commit_clip(
        &mut self,
        clip_result: ClipResult,
        neighbor_idx: usize,
        neighbor_slot: u32,
    ) -> Result<ClipResult, CellFailure> {
        match clip_result {
            ClipResult::TooManyVertices => {
                self.constraints.push(GnomonicConstraint {
                    neighbor_idx,
                    neighbor_slot,
                });
                self.term_cache_valid = false;
                #[cfg(feature = "timing")]
                {
                    self.support_cache_valid = false;
                }
                self.failed = Some(CellFailure::TooManyVertices);
                return Err(CellFailure::TooManyVertices);
            }
            ClipResult::Changed => {
                self.constraints.push(GnomonicConstraint {
                    neighbor_idx,
                    neighbor_slot,
                });
                self.use_a = !self.use_a;
                self.term_cache_valid = false;
                #[cfg(feature = "timing")]
                {
                    self.support_cache_valid = false;
                }
            }
            ClipResult::Unchanged => return Ok(ClipResult::Unchanged),
        }

        let poly = self.current_poly();
        if poly.len < 3 {
            self.failed = Some(CellFailure::ClippedAway);
            return Err(CellFailure::ClippedAway);
        }
        if !poly.has_bounding_ref()
            && (!poly.max_r2.is_finite() || self.exceeds_projection_limit(poly.max_r2))
        {
            self.failed = Some(CellFailure::ProjectionInvalid);
            return Err(CellFailure::ProjectionInvalid);
        }

        Ok(clip_result)
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    #[cfg_attr(not(feature = "profiling"), inline(always))]
    pub(super) fn clip_with_slot_result(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<ClipResult, CellFailure> {
        if let Some(f) = self.failed {
            return Err(f);
        }

        let (a, b, c) = self.bisector_coefficients(neighbor);
        let plane_idx = self.constraints.len();
        let hp = HalfPlane::new_unnormalized(a, b, c, plane_idx);

        let clip_result = if self.use_a {
            clip_convex(&self.poly_a, &hp, &mut self.poly_b)
        } else {
            clip_convex(&self.poly_b, &hp, &mut self.poly_a)
        };

        if clip_result == ClipResult::Unchanged {
            return Ok(ClipResult::Unchanged);
        }
        self.commit_clip(clip_result, neighbor_idx, neighbor_slot)
    }

    /// Shadow/profiling helper: test whether a candidate's bisector would leave
    /// the current polygon unchanged. This mirrors the ordinary clipper's
    /// all-inside decision without mutating builder state.
    #[cfg(feature = "timing")]
    pub(super) fn candidate_would_be_unchanged(&self, neighbor: Vec3) -> bool {
        if !self.is_bounded() || self.vertex_count() < 3 {
            return false;
        }
        let (a, b, c) = self.bisector_coefficients(neighbor);
        let hp = HalfPlane::new_unnormalized(a, b, c, self.constraints.len());
        let poly = self.current_poly();
        for i in 0..poly.len {
            if hp.signed_dist(poly.us[i], poly.vs[i]) < 0.0 {
                return false;
            }
        }
        true
    }

    #[cfg(feature = "timing")]
    fn rebuild_support_cache(&mut self) {
        use std::sync::OnceLock;

        const K: usize = 64;
        static DIRECTIONS: OnceLock<[(f64, f64); K]> = OnceLock::new();
        let directions = DIRECTIONS.get_or_init(|| {
            std::array::from_fn(|sector| {
                let angle = (sector as f64) * std::f64::consts::TAU / K as f64;
                let (sin, cos) = angle.sin_cos();
                (sin, cos)
            })
        });
        let poly = self.current_poly().clone();
        for (sector, &(sin, cos)) in directions.iter().enumerate() {
            let mut min_proj = f64::INFINITY;
            for i in 0..poly.len {
                min_proj = min_proj.min(cos * poly.us[i] + sin * poly.vs[i]);
            }
            self.support_min_proj[sector] = min_proj;
        }
        self.support_cache_valid = true;
    }

    /// Conservative O(1) support-envelope version of
    /// `candidate_would_be_unchanged`. False means "unknown"; true should imply
    /// the exact all-vertices test is also true.
    #[cfg(feature = "timing")]
    pub(super) fn candidate_would_be_unchanged_support(&mut self, neighbor: Vec3) -> bool {
        const K: usize = 64;
        const SECTOR_PENALTY: f64 = 2.0 * 0.024_541_228_522_912_288_f64; // 2 * sin(pi / (2K))

        if !self.is_bounded() || self.vertex_count() < 3 {
            return false;
        }
        if !self.support_cache_valid {
            self.rebuild_support_cache();
        }

        let (a, b, c) = self.bisector_coefficients(neighbor);
        let hp = HalfPlane::new_unnormalized(a, b, c, self.constraints.len());
        if hp.ab2 <= 0.0 || !hp.ab2.is_finite() {
            return hp.c >= 0.0;
        }

        let angle = b.atan2(a).rem_euclid(std::f64::consts::TAU);
        let sector = ((angle * K as f64 / std::f64::consts::TAU).round() as usize) & (K - 1);
        let poly = self.current_poly();
        let radius = poly.max_r2.max(0.0).sqrt();
        let support_lb = self.support_min_proj[sector] - SECTOR_PENALTY * radius;
        let hp_len = hp.ab2.sqrt();
        hp_len * support_lb + hp.c >= 0.0
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn clip_with_slot_edgecheck(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<(), CellFailure> {
        if let Some(f) = self.failed {
            return Err(f);
        }

        let (a, b, c) = self.bisector_coefficients(neighbor);
        let plane_idx = self.constraints.len();
        let hp = HalfPlane::new_unnormalized(a, b, c, plane_idx);
        let clip_result = if self.use_a {
            clip_convex_edgecheck(&self.poly_a, &hp, &mut self.poly_b)
        } else {
            clip_convex_edgecheck(&self.poly_b, &hp, &mut self.poly_a)
        };

        if clip_result == ClipResult::Unchanged {
            return Ok(());
        }
        self.commit_clip(clip_result, neighbor_idx, neighbor_slot)?;
        Ok(())
    }
}

/// Squared-distance threshold for merging numerically identical f64 vertices
/// during the cold fallback clip. This must stay well below the spacing of
/// distinct f32 input generators; a larger f32-era threshold collapsed the
/// microscopic cells that the ClippedAway fallback exists to preserve.
const CLIP_DEDUP_LEN2: f64 = 1e-24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FallbackClipDisposition {
    /// The nominal halfspace contains every current vertex, so the constraint
    /// is redundant for this convex polygon and need not be retained.
    Redundant,
    /// Tolerant classification left the working polygon unchanged, but the
    /// nominal halfspace excludes at least one vertex. Retain the constraint
    /// so extraction and all-constraints reconstruction can enforce it.
    RetainedUnchanged,
    /// The working polygon changed and the constraint owns boundary geometry.
    Changed,
}

impl FallbackClipDisposition {
    #[inline]
    fn clip_result(self) -> ClipResult {
        match self {
            Self::Redundant | Self::RetainedUnchanged => ClipResult::Unchanged,
            Self::Changed => ClipResult::Changed,
        }
    }
}

impl FallbackBuilder {
    /// Inside test for the f64 fallback polygon.
    fn classify_vertex(constraint: &super::FallbackConstraint, position: glam::DVec3) -> bool {
        constraint.normal.dot(position) >= -Self::ON_PLANE_TOL
    }

    fn edge_intersection(
        a: SphericalPolyVertex,
        b: SphericalPolyVertex,
        retained_edge_normal: Option<glam::DVec3>,
        constraint: &super::FallbackConstraint,
    ) -> Option<SphericalPolyVertex> {
        let mut edge_normal = retained_edge_normal
            .unwrap_or_else(|| a.position.cross(b.position))
            .normalize_or_zero();
        let edge_len2 = edge_normal.length_squared();
        if !edge_len2.is_finite() || edge_len2 <= 1e-24 {
            return None;
        }

        // Orient the retained supporting plane from `a` toward `b`. Unlike
        // `a × b`, this remains conditioned when the edge approaches pi.
        let mut tangent = edge_normal.cross(a.position).normalize_or_zero();
        if tangent.dot(b.position) < 0.0 {
            edge_normal = -edge_normal;
            tangent = -tangent;
        }

        let cross = edge_normal.cross(constraint.normal);
        let len2 = cross.length_squared();
        if !len2.is_finite() || len2 <= 1e-24 {
            return None;
        }
        let candidate = cross * len2.sqrt().recip();
        let total_angle = tangent
            .dot(b.position)
            .max(0.0)
            .atan2(a.position.dot(b.position).clamp(-1.0, 1.0));
        let candidate_angle = tangent
            .dot(candidate)
            .atan2(a.position.dot(candidate).clamp(-1.0, 1.0))
            .rem_euclid(std::f64::consts::TAU);
        let dir = if candidate_angle <= total_angle + 1.0e-12 {
            candidate
        } else {
            -candidate
        };
        Some(SphericalPolyVertex { position: dir })
    }

    fn push_output_vertex(
        vertices: &mut Vec<SphericalPolyVertex>,
        edge_planes: &mut Vec<usize>,
        vertex: SphericalPolyVertex,
        outgoing_edge: usize,
    ) {
        if let Some(last) = vertices.last_mut() {
            if (last.position - vertex.position).length_squared() <= CLIP_DEDUP_LEN2 {
                if let Some(last_edge) = edge_planes.last_mut() {
                    *last_edge = outgoing_edge;
                }
                return;
            }
        }
        vertices.push(vertex);
        edge_planes.push(outgoing_edge);
    }

    pub(super) fn clip_poly_with_constraint(
        poly: &SphericalPoly,
        constraint: &super::FallbackConstraint,
        clip_plane: usize,
        constraints: &[super::FallbackConstraint],
    ) -> SphericalPoly {
        let n = poly.vertices.len();
        if n < 3 {
            return SphericalPoly::empty();
        }

        let mut out_vertices = Vec::with_capacity(n + 1);
        let mut out_edges = Vec::with_capacity(n + 1);
        for i in 0..n {
            let j = (i + 1) % n;
            let a = poly.vertices[i];
            let b = poly.vertices[j];
            let edge_plane = poly.edge_planes[i];
            let retained_edge_normal = constraints.get(edge_plane).map(|edge| edge.normal);
            let a_in = Self::classify_vertex(constraint, a.position);
            let b_in = Self::classify_vertex(constraint, b.position);

            match (a_in, b_in) {
                (true, true) => {
                    Self::push_output_vertex(&mut out_vertices, &mut out_edges, a, edge_plane);
                }
                (true, false) => {
                    Self::push_output_vertex(&mut out_vertices, &mut out_edges, a, edge_plane);
                    if let Some(x) = Self::edge_intersection(a, b, retained_edge_normal, constraint)
                    {
                        Self::push_output_vertex(&mut out_vertices, &mut out_edges, x, clip_plane);
                    }
                }
                (false, true) => {
                    if let Some(x) = Self::edge_intersection(a, b, retained_edge_normal, constraint)
                    {
                        Self::push_output_vertex(&mut out_vertices, &mut out_edges, x, edge_plane);
                    }
                }
                (false, false) => {}
            }
        }

        if out_vertices.len() >= 2
            && (out_vertices[0].position - out_vertices[out_vertices.len() - 1].position)
                .length_squared()
                <= CLIP_DEDUP_LEN2
        {
            out_vertices.pop();
            out_edges.pop();
        }

        if out_vertices.len() < 3 {
            return SphericalPoly::empty();
        }
        SphericalPoly {
            vertices: out_vertices,
            edge_planes: out_edges,
        }
    }

    fn push_constraint(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> FallbackClipDisposition {
        let constraint = super::FallbackConstraint::from_neighbor(
            self.generator,
            neighbor_idx,
            neighbor_slot,
            neighbor,
        );
        let plane_idx = self.constraints.len();
        self.constraints.push(constraint);
        let constraint = &self.constraints[plane_idx];
        let nominally_redundant = self
            .poly
            .vertices
            .iter()
            .all(|vertex| constraint.normal.dot(vertex.position) >= 0.0);
        let clipped =
            Self::clip_poly_with_constraint(&self.poly, constraint, plane_idx, &self.constraints);
        if clipped.len() == self.poly.len()
            && clipped.edge_planes == self.poly.edge_planes
            && clipped
                .vertices
                .iter()
                .zip(self.poly.vertices.iter())
                .all(|(a, b)| {
                    a.position.dot(b.position) >= crate::tolerances::FALLBACK_DEDUP_DOT as f64
                })
        {
            if nominally_redundant {
                // The nominal halfspace contains the convex polygon, so this
                // constraint cannot become active after later clipping.
                self.constraints.pop();
                FallbackClipDisposition::Redundant
            } else {
                // The tolerance is an uncertainty band around the nominal
                // great circle, not permission to forget that constraint.
                // It has no incremental edge yet, but extraction's candidate
                // scan sees the retained plane and can reconstruct it.
                FallbackClipDisposition::RetainedUnchanged
            }
        } else {
            self.poly = clipped;
            FallbackClipDisposition::Changed
        }
    }

    #[cfg(test)]
    #[inline]
    pub(super) fn clip_with_slot(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<(), CellFailure> {
        self.clip_with_slot_result(neighbor_idx, neighbor_slot, neighbor)
            .map(|_| ())
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn clip_with_slot_result(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<ClipResult, CellFailure> {
        Ok(self
            .push_constraint(neighbor_idx, neighbor_slot, neighbor)
            .clip_result())
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn clip_with_slot_edgecheck(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<(), CellFailure> {
        self.push_constraint(neighbor_idx, neighbor_slot, neighbor);
        Ok(())
    }

    #[inline]
    #[cfg(feature = "timing")]
    pub(super) fn candidate_would_be_unchanged(&self, _neighbor: Vec3) -> bool {
        false
    }

    #[inline]
    #[cfg(feature = "timing")]
    pub(super) fn candidate_would_be_unchanged_support(&mut self, _neighbor: Vec3) -> bool {
        false
    }
}

impl Topo2DBuilder {
    #[inline(always)]
    pub(crate) fn handle_step_result(
        result: Result<(), CellFailure>,
    ) -> Result<BuilderStepOutcome, CellFailure> {
        Self::classify_step_result(result)
    }

    #[inline(always)]
    pub(crate) fn handle_clip_result(
        result: Result<ClipResult, CellFailure>,
    ) -> Result<BuilderClipOutcome, CellFailure> {
        Self::classify_clip_result(result)
    }

    /// Test-only convenience: clip and surface a fallback request as its
    /// underlying failure (production code routes through the policy methods
    /// and handles fallback explicitly).
    #[cfg(test)]
    #[inline]
    pub(crate) fn clip_with_slot_policy(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<BuilderStepOutcome, CellFailure> {
        let result = match &mut self.inner {
            BuilderImpl::Gnomonic(builder) => {
                builder.clip_with_slot(neighbor_idx, neighbor_slot, neighbor)
            }
            BuilderImpl::Fallback(builder) => {
                builder.clip_with_slot(neighbor_idx, neighbor_slot, neighbor)
            }
        };
        Self::handle_step_result(result)
    }

    #[cfg(test)]
    pub(crate) fn clip_with_slot(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<(), CellFailure> {
        match self.clip_with_slot_policy(neighbor_idx, neighbor_slot, neighbor)? {
            BuilderStepOutcome::Applied => Ok(()),
            BuilderStepOutcome::NeedsFallback(request) => Err(request.as_cell_failure()),
        }
    }

    #[cfg(test)]
    pub(crate) fn clip_with_slot_result(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<ClipResult, CellFailure> {
        match self.clip_with_slot_result_policy(neighbor_idx, neighbor_slot, neighbor)? {
            BuilderClipOutcome::Applied(clip_result) => Ok(clip_result),
            BuilderClipOutcome::NeedsFallback(request) => Err(request.as_cell_failure()),
        }
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    #[cfg_attr(not(feature = "profiling"), inline(always))]
    pub(crate) fn clip_with_slot_result_policy(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<BuilderClipOutcome, CellFailure> {
        let result = match &mut self.inner {
            BuilderImpl::Gnomonic(builder) => {
                builder.clip_with_slot_result(neighbor_idx, neighbor_slot, neighbor)
            }
            BuilderImpl::Fallback(builder) => {
                builder.clip_with_slot_result(neighbor_idx, neighbor_slot, neighbor)
            }
        };
        Self::handle_clip_result(result)
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    #[cfg_attr(not(feature = "profiling"), inline(always))]
    pub(crate) fn clip_with_slot_edgecheck_policy(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<BuilderStepOutcome, CellFailure> {
        let result = match &mut self.inner {
            BuilderImpl::Gnomonic(builder) => {
                builder.clip_with_slot_edgecheck(neighbor_idx, neighbor_slot, neighbor)
            }
            BuilderImpl::Fallback(builder) => {
                builder.clip_with_slot_edgecheck(neighbor_idx, neighbor_slot, neighbor)
            }
        };
        Self::handle_step_result(result)
    }

    #[inline]
    #[cfg(feature = "timing")]
    pub(crate) fn candidate_would_be_unchanged(&self, neighbor: Vec3) -> bool {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.candidate_would_be_unchanged(neighbor),
            BuilderImpl::Fallback(builder) => builder.candidate_would_be_unchanged(neighbor),
        }
    }

    #[inline]
    #[cfg(feature = "timing")]
    pub(crate) fn candidate_would_be_unchanged_support(&mut self, neighbor: Vec3) -> bool {
        match &mut self.inner {
            BuilderImpl::Gnomonic(builder) => {
                builder.candidate_would_be_unchanged_support(neighbor)
            }
            BuilderImpl::Fallback(builder) => {
                builder.candidate_would_be_unchanged_support(neighbor)
            }
        }
    }
}

#[cfg(test)]
mod fallback_tolerance_tests {
    use super::*;
    use glam::DVec3;

    fn constraint(normal: DVec3) -> super::super::FallbackConstraint {
        super::super::FallbackConstraint {
            normal,
            neighbor_idx: 1,
            neighbor_slot: 1,
        }
    }

    #[test]
    fn fallback_membership_boundary_is_inclusive_at_negative_tolerance() {
        let constraint = constraint(DVec3::X);
        let boundary = -FallbackBuilder::ON_PLANE_TOL;
        for (margin, expected) in [
            (boundary.next_up(), true),
            (boundary, true),
            (boundary.next_down(), false),
        ] {
            assert_eq!(
                FallbackBuilder::classify_vertex(&constraint, DVec3::new(margin, 0.0, 0.0)),
                expected,
                "unexpected classification at margin {margin:.17e}"
            );
        }
    }

    #[test]
    fn near_pi_intersection_uses_retained_edge_plane() {
        // Endpoint cross is dominated by the endpoint z residuals,
        // while the retained boundary plane still identifies the intended
        // positive-Y semicircle and its x=0 intersection.
        let a = SphericalPolyVertex {
            position: DVec3::new(1.0, 0.0, 1.0e-8).normalize(),
        };
        let b = SphericalPolyVertex {
            position: DVec3::new(-1.0, 1.0e-10, 1.0e-8).normalize(),
        };
        let cut = constraint(DVec3::X);
        let intersection = FallbackBuilder::edge_intersection(a, b, Some(DVec3::Z), &cut)
            .expect("conditioned edge planes should intersect");
        assert!(intersection.position.dot(DVec3::Y) > 0.999_999);
        assert!(intersection.position.dot(DVec3::X).abs() < 1.0e-12);
    }

    fn point_with_margin(normal: DVec3, tangent: DVec3, margin: f64) -> DVec3 {
        normal * margin + tangent * (1.0 - margin * margin).sqrt()
    }

    fn fallback_for_neighbor(margins: [f64; 3], neighbor: Vec3) -> FallbackBuilder {
        let generator = DVec3::Z;
        let candidate = super::super::FallbackConstraint::from_neighbor(generator, 1, 1, neighbor);
        let normal = candidate.normal;
        let t1 = DVec3::Y;
        let t2 = normal.cross(t1).normalize();
        let tangents = [t1, (t2 - t1).normalize(), (-t2 - t1).normalize()];
        FallbackBuilder {
            generator_idx: 0,
            generator,
            constraints: Vec::new(),
            poly: SphericalPoly {
                vertices: margins
                    .into_iter()
                    .zip(tangents)
                    .map(|(margin, tangent)| SphericalPolyVertex {
                        position: point_with_margin(normal, tangent, margin),
                    })
                    .collect(),
                edge_planes: vec![usize::MAX; 3],
            },
            trigger: super::super::BuilderFallbackTrigger::ProjectionLimit,
        }
    }

    #[test]
    fn tolerance_kept_nominally_active_constraint_is_retained() {
        let neighbor = Vec3::X;
        let mut fallback =
            fallback_for_neighbor([-0.5 * FallbackBuilder::ON_PLANE_TOL, 0.1, 0.1], neighbor);

        assert_eq!(
            fallback.push_constraint(1, 1, neighbor),
            FallbackClipDisposition::RetainedUnchanged
        );
        assert_eq!(fallback.constraints.len(), 1);
    }

    #[test]
    fn nominally_redundant_constraint_is_discarded() {
        let neighbor = Vec3::X;
        let mut fallback = fallback_for_neighbor([0.05, 0.1, 0.1], neighbor);

        assert_eq!(
            fallback.push_constraint(1, 1, neighbor),
            FallbackClipDisposition::Redundant
        );
        assert!(fallback.constraints.is_empty());
    }
}
