use super::projection::MIN_PROJECTION_COS;
use super::{
    BuilderClipOutcome, BuilderImpl, BuilderStepOutcome, FallbackBuilder, GnomonicBuilder,
    Topo2DBuilder,
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
    pub(super) fn commit_clip(
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

        let poly = self.current_poly();
        if poly.len < 3 {
            self.failed = Some(CellFailure::ClippedAway);
            return Err(CellFailure::ClippedAway);
        }
        if !poly.has_bounding_ref() {
            if !poly.max_r2.is_finite() || poly.max_r2 >= MAX_PROJECTION_R2 {
                self.failed = Some(CellFailure::ProjectionInvalid);
                return Err(CellFailure::ProjectionInvalid);
            }
        }

        Ok(clip_result)
    }

    #[cfg_attr(feature = "profiling", inline(never))]
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
            neighbor_positions: &self.neighbor_positions_raw,
        };
        let clip_result = if self.use_a {
            clip_convex(&self.poly_a, &hp, &mut self.poly_b, &esc)
        } else {
            clip_convex(&self.poly_b, &hp, &mut self.poly_a, &esc)
        };

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
    fn sync_neighbor_positions(&mut self, neighbor: Vec3) {
        if self.neighbor_indices.len() > self.neighbor_positions_raw.len() {
            self.neighbor_positions_raw.push(neighbor);
        }
        debug_assert_eq!(
            self.neighbor_indices.len(),
            self.neighbor_positions_raw.len()
        );
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
            neighbor_positions: &self.neighbor_positions_raw,
        };
        let clip_result = if self.use_a {
            clip_convex_edgecheck(&self.poly_a, &hp, &mut self.poly_b, &esc)
        } else {
            clip_convex_edgecheck(&self.poly_b, &hp, &mut self.poly_a, &esc)
        };

        let committed = self.commit_clip(clip_result, hp, neighbor_idx, neighbor_slot);
        self.sync_neighbor_positions(neighbor);
        committed?;
        Ok(())
    }
}

impl FallbackBuilder {
    fn push_constraint(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
        hp_eps: Option<f32>,
    ) {
        self.constraints
            .push(super::FallbackConstraint::from_neighbor(
                self.generator,
                neighbor_idx,
                neighbor_slot,
                hp_eps,
                neighbor,
            ));
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
        self.push_constraint(neighbor_idx, neighbor_slot, neighbor, None);
        Ok(ClipResult::Changed)
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
}

impl Topo2DBuilder {
    pub(crate) fn handle_step_result(
        result: Result<(), CellFailure>,
    ) -> Result<BuilderStepOutcome, CellFailure> {
        Self::classify_step_result(result)
    }

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
}
