mod clip;
mod extract;
mod projection;
#[cfg(test)]
mod tests;

use super::types::{HalfPlane, PolyBuffer};
use crate::knn_clipping::cell_build::CellFailure;
use glam::DVec3;

pub use projection::TangentBasis;

enum BuilderImpl {
    Gnomonic(GnomonicBuilder),
}

pub struct Topo2DBuilder {
    inner: BuilderImpl,
}

pub(crate) struct GnomonicBuilder {
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
