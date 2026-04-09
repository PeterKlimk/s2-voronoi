mod clip;
mod extract;
mod projection;
#[cfg(test)]
mod tests;

use super::types::{HalfPlane, PolyBuffer};
use crate::knn_clipping::cell_build::CellFailure;
use crate::knn_clipping::topo2d::types::ClipResult;
use glam::DVec3;

pub use projection::TangentBasis;

enum BuilderImpl {
    Gnomonic(GnomonicBuilder),
}

pub struct Topo2DBuilder {
    inner: BuilderImpl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuilderFallbackTrigger {
    ProjectionLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BuilderFallbackRequest {
    pub(crate) trigger: BuilderFallbackTrigger,
}

impl BuilderFallbackRequest {
    #[inline]
    pub(crate) fn as_cell_failure(self) -> CellFailure {
        match self.trigger {
            BuilderFallbackTrigger::ProjectionLimit => CellFailure::ProjectionInvalid,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuilderStepOutcome {
    Applied,
    NeedsFallback(BuilderFallbackRequest),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuilderClipOutcome {
    Applied(ClipResult),
    NeedsFallback(BuilderFallbackRequest),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BuilderReplayNeighbor {
    pub(crate) neighbor_idx: usize,
    pub(crate) neighbor_slot: u32,
    pub(crate) hp_eps: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BuilderReplayPlan<'a> {
    pub(crate) generator_idx: usize,
    pub(crate) accepted_neighbors: &'a [BuilderReplayNeighbor],
}

pub(crate) struct GnomonicBuilder {
    pub(crate) generator_idx: usize,
    pub(crate) generator: DVec3,
    pub(crate) basis: TangentBasis,

    half_planes: Vec<HalfPlane>,
    neighbor_indices: Vec<usize>,
    neighbor_slots: Vec<u32>,
    replay_neighbors: Vec<BuilderReplayNeighbor>,

    poly_a: PolyBuffer,
    poly_b: PolyBuffer,
    use_a: bool,

    failed: Option<CellFailure>,
    term_sin_pad: f64,
    term_cos_pad: f64,
    term_threshold_cache: f64,
    term_cache_valid: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BuilderDebugState {
    pub(crate) bounded: bool,
    pub(crate) poly_len: usize,
    pub(crate) has_bounding_ref: bool,
    pub(crate) min_cos: f64,
    pub(crate) half_plane_count: usize,
    pub(crate) neighbor_index_count: usize,
    pub(crate) neighbor_slot_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ExtractionInvariantFailure {
    UnboundedPolygon,
    TooFewVertices {
        poly_len: usize,
    },
    MetadataLengthMismatch {
        half_plane_count: usize,
        neighbor_index_count: usize,
        neighbor_slot_count: usize,
    },
    NonFiniteProjectedVertex {
        vertex: usize,
        u: f64,
        v: f64,
    },
    InvalidVertexPlane {
        vertex: usize,
        plane_a: usize,
        plane_b: usize,
        neighbor_index_count: usize,
    },
    DegenerateDirection {
        vertex: usize,
        len2: f32,
    },
    InvalidEdgePlane {
        vertex: usize,
        edge_plane: usize,
        half_plane_count: usize,
        neighbor_index_count: usize,
        neighbor_slot_count: usize,
    },
}

impl Topo2DBuilder {
    #[inline]
    pub(crate) fn fallback_request_for_failure(
        failure: CellFailure,
    ) -> Option<BuilderFallbackRequest> {
        match failure {
            CellFailure::ProjectionInvalid => Some(BuilderFallbackRequest {
                trigger: BuilderFallbackTrigger::ProjectionLimit,
            }),
            CellFailure::TooManyVertices
            | CellFailure::ClippedAway
            | CellFailure::UnboundedAfterExhaustion
            | CellFailure::NoValidSeed => None,
        }
    }

    #[inline]
    pub(crate) fn classify_step_result(
        result: Result<(), CellFailure>,
    ) -> Result<BuilderStepOutcome, CellFailure> {
        match result {
            Ok(()) => Ok(BuilderStepOutcome::Applied),
            Err(failure) => match Self::fallback_request_for_failure(failure) {
                Some(request) => Ok(BuilderStepOutcome::NeedsFallback(request)),
                None => Err(failure),
            },
        }
    }

    #[inline]
    pub(crate) fn classify_clip_result(
        result: Result<ClipResult, CellFailure>,
    ) -> Result<BuilderClipOutcome, CellFailure> {
        match result {
            Ok(clip_result) => Ok(BuilderClipOutcome::Applied(clip_result)),
            Err(failure) => match Self::fallback_request_for_failure(failure) {
                Some(request) => Ok(BuilderClipOutcome::NeedsFallback(request)),
                None => Err(failure),
            },
        }
    }

    #[inline]
    pub(crate) fn gnomonic(&self) -> &GnomonicBuilder {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder,
        }
    }

    #[inline]
    pub(crate) fn gnomonic_mut(&mut self) -> &mut GnomonicBuilder {
        match &mut self.inner {
            BuilderImpl::Gnomonic(builder) => builder,
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn as_gnomonic(&self) -> &GnomonicBuilder {
        self.gnomonic()
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn as_gnomonic_mut(&mut self) -> &mut GnomonicBuilder {
        self.gnomonic_mut()
    }
}
