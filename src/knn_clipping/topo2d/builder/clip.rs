use super::projection::MIN_PROJECTION_COS;
use super::{
    BuilderClipOutcome, BuilderImpl, BuilderStepOutcome, FallbackBuilder, GnomonicBuilder,
    SphericalPoly, SphericalPolyVertex, Topo2DBuilder,
};
use crate::knn_clipping::cell_build::CellFailure;
use crate::knn_clipping::topo2d::clippers::{clip_convex, clip_convex_edgecheck, EscalationCtx};
use crate::knn_clipping::topo2d::types::{ClipResult, HalfPlane};
use glam::Vec3;

const MAX_PROJECTION_R2: f64 = (1.0 / (MIN_PROJECTION_COS * MIN_PROJECTION_COS)) - 1.0;

impl GnomonicBuilder {
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
        hp: HalfPlane,
        neighbor_idx: usize,
        neighbor_slot: u32,
    ) -> Result<ClipResult, CellFailure> {
        match clip_result {
            ClipResult::TooManyVertices => {
                self.half_planes.push(hp);
                self.neighbor_indices.push(neighbor_idx);
                self.neighbor_slots.push(neighbor_slot);
                self.term_cache_valid = false;
                #[cfg(feature = "timing")]
                {
                    self.support_cache_valid = false;
                }
                self.failed = Some(CellFailure::TooManyVertices);
                return Err(CellFailure::TooManyVertices);
            }
            ClipResult::Changed => {
                self.half_planes.push(hp);
                self.neighbor_indices.push(neighbor_idx);
                self.neighbor_slots.push(neighbor_slot);
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
            && (!poly.max_r2.is_finite() || poly.max_r2 >= MAX_PROJECTION_R2)
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
        let plane_idx = self.half_planes.len();
        let hp = HalfPlane::new_unnormalized(a, b, c, plane_idx);

        #[cfg(feature = "p5_shadow")]
        self.shadow_audit(neighbor_idx, neighbor, &hp);

        let esc = EscalationCtx {
            generator_raw: self.generator_raw,
            neighbor_raw: neighbor,
            #[cfg(feature = "p5_shadow")]
            neighbor_positions: &self.neighbor_positions_raw,
            #[cfg(not(feature = "p5_shadow"))]
            neighbor_positions: &[],
        };
        let clip_result = if self.use_a {
            clip_convex(&self.poly_a, &hp, &mut self.poly_b, &esc)
        } else {
            clip_convex(&self.poly_b, &hp, &mut self.poly_a, &esc)
        };

        if clip_result == ClipResult::Unchanged {
            return Ok(ClipResult::Unchanged);
        }
        let committed = self.commit_clip(clip_result, hp, neighbor_idx, neighbor_slot);
        self.sync_neighbor_positions(neighbor);
        committed
    }

    /// Shadow audit (P5 stage 1): compare near-margin local decisions
    /// against the canonical exact predicate before the clip is applied.
    /// No behavior change; compiled out without the feature.
    #[cfg(feature = "p5_shadow")]
    fn shadow_audit(&self, neighbor_idx: usize, neighbor: Vec3, hp: &HalfPlane) {
        let poly = if self.use_a {
            &self.poly_a
        } else {
            &self.poly_b
        };
        crate::knn_clipping::p5_shadow::audit_clip(
            self.generator_idx,
            self.generator_raw,
            neighbor_idx,
            neighbor,
            &self.neighbor_indices,
            &self.neighbor_positions_raw,
            poly,
            hp,
        );
    }

    /// Keep the raw position list parallel to `neighbor_indices`
    /// (a `Changed` commit pushed a constraint — including commits that then
    /// fail with `ClippedAway`, so this runs on the error path too).
    ///
    /// Only the `p5_shadow` build reads `neighbor_positions_raw` (production
    /// keeps escalation dead), so this is a no-op otherwise.
    #[cfg_attr(not(feature = "p5_shadow"), allow(unused_variables))]
    #[inline]
    fn sync_neighbor_positions(&mut self, neighbor: Vec3) {
        #[cfg(feature = "p5_shadow")]
        {
            if self.neighbor_indices.len() > self.neighbor_positions_raw.len() {
                self.neighbor_positions_raw.push(neighbor);
            }
            debug_assert_eq!(
                self.neighbor_indices.len(),
                self.neighbor_positions_raw.len()
            );
        }
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
        let hp = HalfPlane::new_unnormalized(a, b, c, self.half_planes.len());
        let poly = self.current_poly();
        let neg_eps = -hp.eps;
        for i in 0..poly.len {
            if hp.signed_dist(poly.us[i], poly.vs[i]) < neg_eps {
                return false;
            }
        }
        true
    }

    #[cfg(feature = "timing")]
    fn rebuild_support_cache(&mut self) {
        const K: usize = 64;
        let poly = self.current_poly().clone();
        for sector in 0..K {
            let angle = (sector as f64) * std::f64::consts::TAU / K as f64;
            let (sin, cos) = angle.sin_cos();
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
        let hp = HalfPlane::new_unnormalized(a, b, c, self.half_planes.len());
        if hp.ab2 <= 0.0 || !hp.ab2.is_finite() {
            return hp.c >= -hp.eps;
        }

        let angle = b.atan2(a).rem_euclid(std::f64::consts::TAU);
        let sector = ((angle * K as f64 / std::f64::consts::TAU).round() as usize) & (K - 1);
        let poly = self.current_poly();
        let radius = poly.max_r2.max(0.0).sqrt();
        let support_lb = self.support_min_proj[sector] - SECTOR_PENALTY * radius;
        let hp_len = hp.ab2.sqrt();
        hp_len * support_lb + hp.c >= -hp.eps
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn clip_with_slot_edgecheck(
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

        let (a, b, c) = self.bisector_coefficients(neighbor);
        let plane_idx = self.half_planes.len();
        let hp = HalfPlane::new_unnormalized_with_eps(a, b, c, plane_idx, hp_eps as f64);

        #[cfg(feature = "p5_shadow")]
        self.shadow_audit(neighbor_idx, neighbor, &hp);

        let esc = EscalationCtx {
            generator_raw: self.generator_raw,
            neighbor_raw: neighbor,
            #[cfg(feature = "p5_shadow")]
            neighbor_positions: &self.neighbor_positions_raw,
            #[cfg(not(feature = "p5_shadow"))]
            neighbor_positions: &[],
        };
        let clip_result = if self.use_a {
            clip_convex_edgecheck(&self.poly_a, &hp, &mut self.poly_b, &esc)
        } else {
            clip_convex_edgecheck(&self.poly_b, &hp, &mut self.poly_a, &esc)
        };

        if clip_result == ClipResult::Unchanged {
            return Ok(());
        }
        let committed = self.commit_clip(clip_result, hp, neighbor_idx, neighbor_slot);
        self.sync_neighbor_positions(neighbor);
        committed?;
        Ok(())
    }
}

/// Squared-distance threshold for merging coincident vertices *during* the
/// incremental clip (~1e-6 rad, matching extraction's `push_fallback_vertex`).
/// Deliberately far tighter than `FALLBACK_DEDUP_DOT` (~4.5e-3 rad): that coarse
/// value is for the final output-vertex dedup over exact all-pairs points, but
/// applying it per-clip collapses short *real* edges and overwrites the dropped
/// edge's neighbor plane, silently losing a true neighbor of the cell.
const CLIP_DEDUP_LEN2: f32 = 1e-12;

impl FallbackBuilder {
    /// Inside test for the incremental clip. Operates on the f32 polygon
    /// vertices (~1e-7 noise at unit scale), so it uses the f32-noise-tolerant
    /// `ON_PLANE_TOL` band rather than the much tighter `FALLBACK_PLANE_TOL`
    /// (1e-9) that extraction's `satisfies_all_constraints` applies to *exact
    /// f64* pair-intersection directions. A 1e-9 slack here wrongly clips
    /// near-boundary f32 vertices and can collapse the polygon below 3 vertices
    /// (UnboundedAfterExhaustion). (`hp_eps` is gnomonic-chart scaled —
    /// meaningless against this chord-scale dot — and is retained only as output
    /// metadata.)
    fn classify_vertex(constraint: &super::FallbackConstraint, position: Vec3) -> bool {
        let p = glam::DVec3::new(position.x as f64, position.y as f64, position.z as f64);
        constraint.normal.dot(p) >= -Self::ON_PLANE_TOL
    }

    fn edge_intersection(
        a: SphericalPolyVertex,
        b: SphericalPolyVertex,
        constraint: &super::FallbackConstraint,
    ) -> Option<SphericalPolyVertex> {
        let a64 = glam::DVec3::new(
            a.position.x as f64,
            a.position.y as f64,
            a.position.z as f64,
        );
        let b64 = glam::DVec3::new(
            b.position.x as f64,
            b.position.y as f64,
            b.position.z as f64,
        );
        let edge_normal = a64.cross(b64);
        let edge_len2 = edge_normal.length_squared();
        if !edge_len2.is_finite() || edge_len2 <= 1e-24 {
            return None;
        }

        let cross = edge_normal.cross(constraint.normal);
        let len2 = cross.length_squared();
        if !len2.is_finite() || len2 <= 1e-24 {
            return None;
        }
        let candidate = cross * len2.sqrt().recip();
        let midpoint = (a64 + b64).normalize_or_zero();
        let dir = if candidate.dot(midpoint) >= 0.0 {
            candidate
        } else {
            -candidate
        };
        let dir32 = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32).normalize();
        Some(SphericalPolyVertex { position: dir32 })
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
            let a_in = Self::classify_vertex(constraint, a.position);
            let b_in = Self::classify_vertex(constraint, b.position);

            match (a_in, b_in) {
                (true, true) => {
                    Self::push_output_vertex(&mut out_vertices, &mut out_edges, a, edge_plane);
                }
                (true, false) => {
                    Self::push_output_vertex(&mut out_vertices, &mut out_edges, a, edge_plane);
                    if let Some(x) = Self::edge_intersection(a, b, constraint) {
                        Self::push_output_vertex(&mut out_vertices, &mut out_edges, x, clip_plane);
                    }
                }
                (false, true) => {
                    if let Some(x) = Self::edge_intersection(a, b, constraint) {
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
        hp_eps: Option<f32>,
    ) -> ClipResult {
        let constraint = super::FallbackConstraint::from_neighbor(
            self.generator,
            neighbor_idx,
            neighbor_slot,
            hp_eps,
            neighbor,
        );
        let plane_idx = self.constraints.len();
        self.constraints.push(constraint);
        let constraint = &self.constraints[plane_idx];
        let clipped = Self::clip_poly_with_constraint(&self.poly, constraint, plane_idx);
        if clipped.len() == self.poly.len()
            && clipped.edge_planes == self.poly.edge_planes
            && clipped
                .vertices
                .iter()
                .zip(self.poly.vertices.iter())
                .all(|(a, b)| a.position.dot(b.position) >= crate::tolerances::FALLBACK_DEDUP_DOT)
        {
            // The constraint cut nothing, so no edge references `plane_idx`;
            // drop it so it does not inflate the O(constraints) extraction
            // scans. Mirrors the gnomonic path, which never records an
            // `Unchanged` clip as an accepted half-plane.
            self.constraints.pop();
            ClipResult::Unchanged
        } else {
            self.poly = clipped;
            ClipResult::Changed
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
        Ok(self.push_constraint(neighbor_idx, neighbor_slot, neighbor, None))
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn clip_with_slot_edgecheck(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
        hp_eps: f32,
    ) -> Result<(), CellFailure> {
        let replay_hp_eps = if hp_eps.is_finite() && hp_eps > 0.0 {
            Some(hp_eps)
        } else {
            None
        };
        self.push_constraint(neighbor_idx, neighbor_slot, neighbor, replay_hp_eps);
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
        hp_eps: f32,
    ) -> Result<BuilderStepOutcome, CellFailure> {
        let result = match &mut self.inner {
            BuilderImpl::Gnomonic(builder) => {
                builder.clip_with_slot_edgecheck(neighbor_idx, neighbor_slot, neighbor, hp_eps)
            }
            BuilderImpl::Fallback(builder) => {
                builder.clip_with_slot_edgecheck(neighbor_idx, neighbor_slot, neighbor, hp_eps)
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
