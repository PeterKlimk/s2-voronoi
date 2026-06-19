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
// Key construction helper shared with the planar builder.
pub(crate) use projection::sort3_u32;

// One long-lived instance per worker, reused via reset(); boxing the
// large gnomonic variant would put indirection on the hot clip path.
#[allow(clippy::large_enum_variant)]
enum BuilderImpl {
    Gnomonic(GnomonicBuilder),
    Fallback(FallbackBuilder),
}

pub struct Topo2DBuilder {
    inner: BuilderImpl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuilderFallbackTrigger {
    ProjectionLimit,
    PolygonVertexLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BuilderFallbackRequest {
    pub(crate) trigger: BuilderFallbackTrigger,
}

#[cfg(test)]
impl BuilderFallbackRequest {
    #[inline]
    pub(crate) fn as_cell_failure(self) -> CellFailure {
        match self.trigger {
            BuilderFallbackTrigger::ProjectionLimit => CellFailure::ProjectionInvalid,
            BuilderFallbackTrigger::PolygonVertexLimit => CellFailure::TooManyVertices,
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

pub(crate) struct GnomonicBuilder {
    pub(crate) generator_idx: usize,
    pub(crate) generator: DVec3,
    pub(crate) basis: TangentBasis,

    /// 0.5 / |generator|^2, cached per cell: the exact chord-bisector scale
    /// correction for a generator that is (post stage-0) the raw f32 bits
    /// rather than an f64-renormalized unit vector. Equal to 0.5 exactly for
    /// exactly-unit generators, reproducing the legacy formula bit-for-bit.
    inv_two_gg: f64,
    generator_dot_g: f64,

    half_planes: Vec<HalfPlane>,
    neighbor_indices: Vec<usize>,
    neighbor_slots: Vec<u32>,

    poly_a: PolyBuffer,
    poly_b: PolyBuffer,
    use_a: bool,

    failed: Option<CellFailure>,
    /// Raw f32 generator coordinates — the exact input bits the canonical
    /// escalation predicate consumes (`generator` above is the f64
    /// promotion; P5 stage 2).
    pub(crate) generator_raw: glam::Vec3,
    /// Raw f32 neighbor positions, parallel to `neighbor_indices` /
    /// `half_planes`; consumed by canonical escalation and the shadow audit.
    /// In production `CLIP_ESCALATION_FACTOR == 0.0`, so the escalation path is
    /// dead (see `maybe_escalate`) and this list is never read — it exists only
    /// for the `p5_shadow` audit/override build, so it (and its per-clip
    /// maintenance in `sync_neighbor_positions`) is gated off otherwise.
    #[cfg(feature = "p5_shadow")]
    pub(crate) neighbor_positions_raw: Vec<glam::Vec3>,
    term_sin_pad: f64,
    term_cos_pad: f64,
    term_threshold_cache: f64,
    term_cache_valid: bool,
    #[cfg(feature = "timing")]
    support_cache_valid: bool,
    #[cfg(feature = "timing")]
    support_min_proj: [f64; 64],
}

pub(crate) struct FallbackBuilder {
    pub(crate) generator_idx: usize,
    pub(crate) generator: DVec3,
    constraints: Vec<FallbackConstraint>,
    /// Which limit forced the fallback handoff; read by handoff tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) trigger: BuilderFallbackTrigger,
}

#[derive(Debug, Clone)]
pub(crate) struct FallbackConstraint {
    pub(crate) normal: DVec3,
    pub(crate) neighbor_idx: usize,
    pub(crate) neighbor_slot: u32,
    pub(crate) hp_eps: Option<f32>,
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
            CellFailure::TooManyVertices => Some(BuilderFallbackRequest {
                trigger: BuilderFallbackTrigger::PolygonVertexLimit,
            }),
            CellFailure::ClippedAway
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

    pub(crate) fn enter_fallback(
        &mut self,
        points: &[glam::Vec3],
        request: BuilderFallbackRequest,
    ) {
        let fallback = match &self.inner {
            BuilderImpl::Gnomonic(builder) => {
                FallbackBuilder::from_gnomonic(builder, points, request.trigger)
            }
            BuilderImpl::Fallback(builder) => {
                FallbackBuilder::from_fallback(builder, request.trigger)
            }
        };
        self.inner = BuilderImpl::Fallback(fallback);
    }

    #[inline]
    pub(crate) fn generator_idx(&self) -> usize {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.generator_idx,
            BuilderImpl::Fallback(builder) => builder.generator_idx,
        }
    }

    #[inline]
    pub(crate) fn accepted_constraint_count(&self) -> usize {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.neighbor_indices.len(),
            BuilderImpl::Fallback(builder) => builder.constraints.len(),
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn as_gnomonic(&self) -> &GnomonicBuilder {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder,
            BuilderImpl::Fallback(_) => {
                panic!("attempted to access gnomonic builder after fallback handoff")
            }
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn as_gnomonic_mut(&mut self) -> &mut GnomonicBuilder {
        match &mut self.inner {
            BuilderImpl::Gnomonic(builder) => builder,
            BuilderImpl::Fallback(_) => {
                panic!("attempted to mutably access gnomonic builder after fallback handoff")
            }
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn as_fallback(&self) -> &FallbackBuilder {
        match &self.inner {
            BuilderImpl::Fallback(builder) => builder,
            BuilderImpl::Gnomonic(_) => panic!("expected fallback builder"),
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn is_fallback(&self) -> bool {
        matches!(self.inner, BuilderImpl::Fallback(_))
    }
}

impl FallbackBuilder {
    fn from_gnomonic(
        builder: &GnomonicBuilder,
        points: &[glam::Vec3],
        trigger: BuilderFallbackTrigger,
    ) -> Self {
        let constraints = builder
            .neighbor_indices
            .iter()
            .copied()
            .zip(builder.neighbor_slots.iter().copied())
            .zip(builder.half_planes.iter())
            .map(|((neighbor_idx, neighbor_slot), plane)| {
                FallbackConstraint::from_neighbor(
                    builder.generator,
                    neighbor_idx,
                    neighbor_slot,
                    Some(plane.eps as f32),
                    points[neighbor_idx],
                )
            })
            .collect();

        Self {
            generator_idx: builder.generator_idx,
            generator: builder.generator,
            constraints,
            trigger,
        }
    }

    fn from_fallback(builder: &FallbackBuilder, trigger: BuilderFallbackTrigger) -> Self {
        Self {
            generator_idx: builder.generator_idx,
            generator: builder.generator,
            constraints: builder.constraints.clone(),
            trigger,
        }
    }
}
