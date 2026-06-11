//! Error types for Voronoi computation.

use std::fmt;

/// Errors that can occur during Voronoi computation.
#[derive(Debug, Clone)]
pub enum VoronoiError {
    /// Not enough points to form a valid Voronoi diagram on the sphere.
    /// Need at least 4 points for a non-degenerate result.
    InsufficientPoints(usize),

    /// An input point is not usable (e.g. a non-finite component).
    InvalidInput { point_index: usize, message: String },

    /// The input induced geometry outside the currently supported clipping model.
    ///
    /// This is used for expected algorithm boundaries such as cells that extend beyond the
    /// generator hemisphere in the current gnomonic projection model.
    UnsupportedGeometry {
        generator_index: usize,
        message: String,
    },

    /// Too many coincident (or near-coincident) point pairs in the input.
    /// This indicates the input is degenerate and cannot be reliably computed.
    DegenerateInput {
        coincident_pairs: usize,
        message: String,
    },

    /// Computation exceeded a concrete representation or packing limit.
    ///
    /// This is not necessarily a logic bug; it means the current internal
    /// storage/layout could not represent the requested computation.
    RepresentationLimit(String),

    /// Internal or otherwise unclassified computation failure.
    ComputationFailed(String),
}

impl fmt::Display for VoronoiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoronoiError::InsufficientPoints(n) => {
                write!(f, "insufficient points: need at least 4, got {}", n)
            }
            VoronoiError::InvalidInput {
                point_index,
                message,
            } => {
                write!(f, "invalid input at point {}: {}", point_index, message)
            }
            VoronoiError::UnsupportedGeometry {
                generator_index,
                message,
            } => {
                write!(
                    f,
                    "unsupported geometry at generator {}: {}",
                    generator_index, message
                )
            }
            VoronoiError::DegenerateInput {
                coincident_pairs,
                message,
            } => {
                write!(
                    f,
                    "degenerate input: {} coincident point pairs ({})",
                    coincident_pairs, message
                )
            }
            VoronoiError::RepresentationLimit(msg) => {
                write!(f, "representation limit exceeded: {}", msg)
            }
            VoronoiError::ComputationFailed(msg) => {
                write!(f, "computation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for VoronoiError {}
