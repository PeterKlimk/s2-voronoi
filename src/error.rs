//! Error types for Voronoi computation.

use std::fmt;

/// Errors that can occur during Voronoi computation.
#[derive(Debug, Clone)]
pub enum VoronoiError {
    /// Not enough points to form a valid Voronoi diagram on the sphere.
    /// Need at least 4 points for a non-degenerate result.
    InsufficientPoints(usize),

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

    /// Internal computation failure.
    /// This should not happen with valid input and indicates a bug.
    ComputationFailed(String),
}

impl fmt::Display for VoronoiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoronoiError::InsufficientPoints(n) => {
                write!(f, "insufficient points: need at least 4, got {}", n)
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
            VoronoiError::ComputationFailed(msg) => {
                write!(f, "computation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for VoronoiError {}
