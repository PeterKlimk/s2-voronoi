mod clip;
mod extract;
mod projection;
#[cfg(test)]
mod tests;

use super::types::{HalfPlane, PolyBuffer, INVALID_PLANE_ID};
use crate::knn_clipping::cell_build::CellFailure;
use crate::knn_clipping::topo2d::types::ClipResult;
use glam::DVec3;

pub use projection::TangentBasis;

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
    ClippedAway,
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
            BuilderFallbackTrigger::ClippedAway => CellFailure::ClippedAway,
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

    constraints: Vec<GnomonicConstraint>,

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
    /// Chart metric stretch bound folded into every angular-radius use
    /// (near-pole tangent-basis distortion; see `chart_metric_r2_bound`).
    chart_metric_r2_scale: f64,
    #[cfg(feature = "timing")]
    support_cache_valid: bool,
    #[cfg(feature = "timing")]
    support_min_proj: [f64; 64],
}

#[derive(Clone, Copy)]
struct GnomonicConstraint {
    half_plane: HalfPlane,
    neighbor_idx: usize,
    neighbor_slot: u32,
}

pub(crate) struct FallbackBuilder {
    pub(crate) generator_idx: usize,
    pub(crate) generator: DVec3,
    constraints: Vec<FallbackConstraint>,
    poly: SphericalPoly,
    /// Which limit forced the fallback handoff; read by handoff tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) trigger: BuilderFallbackTrigger,
}

#[derive(Clone)]
pub(crate) struct SphericalPoly {
    vertices: Vec<SphericalPolyVertex>,
    edge_planes: Vec<usize>,
}

#[derive(Clone, Copy)]
pub(crate) struct SphericalPolyVertex {
    pub(crate) position: DVec3,
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
            CellFailure::ClippedAway => Some(BuilderFallbackRequest {
                trigger: BuilderFallbackTrigger::ClippedAway,
            }),
            CellFailure::UnboundedAfterExhaustion | CellFailure::NoValidSeed => None,
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

    /// Try to replace the current builder with the requested fallback.
    /// Returns `false` when a `ClippedAway` replay is genuinely infeasible;
    /// the original failed builder is retained so its failure classification
    /// remains available to the caller.
    #[must_use = "fallback rejection must terminate the current clip phase"]
    pub(crate) fn try_enter_fallback(
        &mut self,
        points: &[glam::Vec3],
        request: BuilderFallbackRequest,
    ) -> bool {
        let fallback = match &self.inner {
            BuilderImpl::Gnomonic(builder) => {
                FallbackBuilder::from_gnomonic(builder, points, request.trigger)
            }
            BuilderImpl::Fallback(builder) => {
                FallbackBuilder::from_fallback(builder, request.trigger)
            }
        };
        // A gnomonic ClippedAway result has already committed the triggering
        // constraint and reduced its f32 chart polygon below three vertices.
        // Rebuild from the accepted f64 spherical constraints before switching.
        // If those constraints are genuinely infeasible (e.g. coincident
        // generators), keep the original failed builder so its ClippedAway
        // classification survives to the public error path.
        if request.trigger == BuilderFallbackTrigger::ClippedAway && fallback.poly.len() < 3 {
            return false;
        }
        self.inner = BuilderImpl::Fallback(fallback);
        true
    }

    #[inline]
    pub(crate) fn generator_idx(&self) -> usize {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.generator_idx,
            BuilderImpl::Fallback(builder) => builder.generator_idx,
        }
    }

    #[inline]
    pub(crate) fn generator(&self) -> DVec3 {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.generator,
            BuilderImpl::Fallback(builder) => builder.generator,
        }
    }

    #[inline]
    pub(crate) fn accepted_constraint_count(&self) -> usize {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.constraints.len(),
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

    pub(crate) fn accepted_spherical_constraints(
        &self,
        points: &[glam::Vec3],
    ) -> Vec<FallbackConstraint> {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder
                .constraints
                .iter()
                .map(|constraint| {
                    FallbackConstraint::from_neighbor(
                        builder.generator,
                        constraint.neighbor_idx,
                        constraint.neighbor_slot,
                        Some(constraint.half_plane.eps as f32),
                        points[constraint.neighbor_idx],
                    )
                })
                .collect(),
            BuilderImpl::Fallback(builder) => builder.constraints.clone(),
        }
    }
}

impl FallbackBuilder {
    fn from_gnomonic(
        builder: &GnomonicBuilder,
        points: &[glam::Vec3],
        trigger: BuilderFallbackTrigger,
    ) -> Self {
        let constraints = builder
            .constraints
            .iter()
            .map(|constraint| {
                FallbackConstraint::from_neighbor(
                    builder.generator,
                    constraint.neighbor_idx,
                    constraint.neighbor_slot,
                    Some(constraint.half_plane.eps as f32),
                    points[constraint.neighbor_idx],
                )
            })
            .collect();

        let mut fallback = Self {
            generator_idx: builder.generator_idx,
            generator: builder.generator,
            constraints,
            poly: SphericalPoly::from_gnomonic(builder),
            trigger,
        };

        if trigger == BuilderFallbackTrigger::ClippedAway {
            // The failed chart polygon cannot seed a handoff. Start again from
            // the same large bounding polygon in f64 spherical form and replay
            // every accepted constraint, including the one that spuriously
            // collapsed the gnomonic polygon. Sentinel bounding edges are
            // allowed here; extraction rejects them unless later real
            // bisectors fully bound the cell.
            let seed = GnomonicBuilder::new(builder.generator_idx, builder.generator_raw);
            fallback.poly = SphericalPoly::from_gnomonic(&seed);
            for plane_idx in 0..fallback.constraints.len() {
                fallback.poly = FallbackBuilder::clip_poly_with_constraint(
                    &fallback.poly,
                    &fallback.constraints[plane_idx],
                    plane_idx,
                );
                if fallback.poly.len() < 3 {
                    break;
                }
            }
        } else if trigger == BuilderFallbackTrigger::PolygonVertexLimit {
            if let Some((plane_idx, constraint)) = fallback
                .constraints
                .len()
                .checked_sub(1)
                .map(|idx| (idx, &fallback.constraints[idx]))
            {
                fallback.poly = FallbackBuilder::clip_poly_with_constraint(
                    &fallback.poly,
                    constraint,
                    plane_idx,
                );
            }
        }

        fallback
    }

    fn from_fallback(builder: &FallbackBuilder, trigger: BuilderFallbackTrigger) -> Self {
        Self {
            generator_idx: builder.generator_idx,
            generator: builder.generator,
            constraints: builder.constraints.clone(),
            poly: builder.poly.clone(),
            trigger,
        }
    }
}

impl SphericalPoly {
    fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            edge_planes: Vec::new(),
        }
    }

    fn from_gnomonic(builder: &GnomonicBuilder) -> Self {
        let poly = builder.current_poly();
        if poly.len < 3 {
            return Self::empty();
        }

        let mut out = Self {
            vertices: Vec::with_capacity(poly.len),
            edge_planes: Vec::with_capacity(poly.len),
        };

        for i in 0..poly.len {
            let edge_plane = poly.edge_planes[i];
            let u = poly.us[i];
            let v = poly.vs[i];
            let dir = DVec3::new(
                crate::fp::fma_f64(
                    u,
                    builder.basis.t1.x,
                    crate::fp::fma_f64(v, builder.basis.t2.x, builder.basis.g.x),
                ),
                crate::fp::fma_f64(
                    u,
                    builder.basis.t1.y,
                    crate::fp::fma_f64(v, builder.basis.t2.y, builder.basis.g.y),
                ),
                crate::fp::fma_f64(
                    u,
                    builder.basis.t1.z,
                    crate::fp::fma_f64(v, builder.basis.t2.z, builder.basis.g.z),
                ),
            );
            let len2 = dir.length_squared();
            if !len2.is_finite() || len2 < crate::tolerances::EXTRACT_DEGENERATE_LEN2 as f64 {
                return Self::empty();
            }

            out.vertices.push(SphericalPolyVertex {
                position: dir * len2.sqrt().recip(),
            });
            // Bounding-box pseudo-edges (INVALID_PLANE_ID) are kept as an
            // out-of-range sentinel so the incremental clip can still carry and
            // eventually remove the box vertices as real bisectors arrive. Any
            // sentinel that survives to extraction is caught there as an
            // unbounded-cell failure rather than silently dropping a corner.
            out.edge_planes.push(plane_id_to_fallback(edge_plane));
        }

        out
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.vertices.len()
    }
}

#[inline]
fn plane_id_to_fallback(plane: super::types::PlaneId) -> usize {
    if plane == INVALID_PLANE_ID {
        usize::MAX
    } else {
        plane as usize
    }
}
