use super::projection::MIN_PROJECTION_COS;
use super::{GnomonicBuilder, Topo2DBuilder};
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

        self.commit_clip(clip_result, hp, neighbor_idx, neighbor_slot)
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

        self.commit_clip(clip_result, hp, neighbor_idx, neighbor_slot)?;
        Ok(())
    }
}

impl Topo2DBuilder {
    pub fn clip_with_slot(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<(), CellFailure> {
        self.gnomonic_mut()
            .clip_with_slot(neighbor_idx, neighbor_slot, neighbor)
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn clip_with_slot_result(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<ClipResult, CellFailure> {
        self.gnomonic_mut()
            .clip_with_slot_result(neighbor_idx, neighbor_slot, neighbor)
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn clip_with_slot_edgecheck(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
        hp_eps: f32,
    ) -> Result<(), CellFailure> {
        self.gnomonic_mut()
            .clip_with_slot_edgecheck(neighbor_idx, neighbor_slot, neighbor, hp_eps)
    }
}
