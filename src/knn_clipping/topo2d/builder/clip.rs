use super::projection::MIN_PROJECTION_COS;
use super::{
    BuilderClipOutcome, BuilderImpl, BuilderReplayNeighbor, BuilderStepOutcome, FallbackBuilder,
    GnomonicBuilder, Topo2DBuilder,
};
use crate::knn_clipping::cell_build::CellFailure;
use crate::knn_clipping::topo2d::clippers::{clip_convex, clip_convex_edgecheck};
use crate::knn_clipping::topo2d::types::{ClipResult, HalfPlane};
use glam::Vec3;

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
        replay_hp_eps: Option<f32>,
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
                self.replay_neighbors.push(BuilderReplayNeighbor {
                    neighbor_idx,
                    neighbor_slot,
                    hp_eps: replay_hp_eps,
                });
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
            let min_cos = poly.min_cos();
            if !min_cos.is_finite() || min_cos <= MIN_PROJECTION_COS {
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

        let clip_result = if self.use_a {
            clip_convex(&self.poly_a, &hp, &mut self.poly_b)
        } else {
            clip_convex(&self.poly_b, &hp, &mut self.poly_a)
        };

        self.commit_clip(clip_result, hp, neighbor_idx, neighbor_slot, None)
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

        let clip_result = if self.use_a {
            clip_convex_edgecheck(&self.poly_a, &hp, &mut self.poly_b)
        } else {
            clip_convex_edgecheck(&self.poly_b, &hp, &mut self.poly_a)
        };

        self.commit_clip(clip_result, hp, neighbor_idx, neighbor_slot, Some(hp_eps))?;
        Ok(())
    }
}

impl FallbackBuilder {
    #[inline]
    pub(super) fn clip_with_slot(
        &mut self,
        _neighbor_idx: usize,
        _neighbor_slot: u32,
        _neighbor: Vec3,
    ) -> Result<(), CellFailure> {
        Err(self.failure)
    }

    #[inline]
    pub(super) fn clip_with_slot_result(
        &mut self,
        _neighbor_idx: usize,
        _neighbor_slot: u32,
        _neighbor: Vec3,
    ) -> Result<ClipResult, CellFailure> {
        Err(self.failure)
    }

    #[inline]
    pub(super) fn clip_with_slot_edgecheck(
        &mut self,
        _neighbor_idx: usize,
        _neighbor_slot: u32,
        _neighbor: Vec3,
        _hp_eps: f32,
    ) -> Result<(), CellFailure> {
        Err(self.failure)
    }
}

impl Topo2DBuilder {
    pub(crate) fn handle_step_result(
        &mut self,
        result: Result<(), CellFailure>,
    ) -> Result<BuilderStepOutcome, CellFailure> {
        match Self::classify_step_result(result)? {
            BuilderStepOutcome::Applied => Ok(BuilderStepOutcome::Applied),
            BuilderStepOutcome::NeedsFallback(request) => {
                self.enter_fallback(request);
                Ok(BuilderStepOutcome::NeedsFallback(request))
            }
        }
    }

    pub(crate) fn handle_clip_result(
        &mut self,
        result: Result<ClipResult, CellFailure>,
    ) -> Result<BuilderClipOutcome, CellFailure> {
        match Self::classify_clip_result(result)? {
            BuilderClipOutcome::Applied(clip_result) => {
                Ok(BuilderClipOutcome::Applied(clip_result))
            }
            BuilderClipOutcome::NeedsFallback(request) => {
                self.enter_fallback(request);
                Ok(BuilderClipOutcome::NeedsFallback(request))
            }
        }
    }

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
        self.handle_step_result(result)
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
        self.handle_clip_result(result)
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
        self.handle_step_result(result)
    }

    pub fn clip_with_slot(
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

    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn clip_with_slot_result(
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
}
