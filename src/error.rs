//! Error types for Voronoi computation.

use std::fmt;

/// Errors that can occur during Voronoi computation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum VoronoiError {
    /// Not enough points to form a valid Voronoi diagram on the sphere.
    /// Need at least 4 points for a non-degenerate result.
    InsufficientPoints(usize),

    /// An input point is not usable (e.g. a non-finite component).
    InvalidInput {
        /// Index of the offending point in the input slice.
        point_index: usize,
        /// What is wrong with it.
        message: String,
    },

    /// A computation option has an unsupported value.
    InvalidConfiguration(String),

    /// The input induced geometry outside the currently supported clipping model.
    ///
    /// This is used for expected algorithm boundaries such as cells that extend beyond the
    /// generator hemisphere in the current gnomonic projection model.
    UnsupportedGeometry {
        /// Generator whose cell hit the model boundary.
        generator_index: usize,
        /// Which boundary, and what to do about it.
        message: String,
    },

    /// Too many coincident (or near-coincident) point pairs in the input.
    /// This indicates the input is degenerate and cannot be reliably computed.
    DegenerateInput {
        /// Number of coincident generator pairs detected.
        coincident_pairs: usize,
        /// The offending generators and the suggested fix.
        message: String,
    },

    /// Computation exceeded a concrete representation or packing limit.
    ///
    /// This is not necessarily a logic bug; it means the current internal
    /// storage/layout could not represent the requested computation.
    RepresentationLimit(String),

    /// Exact-zero output resolution could not be completed without removing
    /// one or more effective generator cells.
    CellEliminationRequired {
        /// Original input generator indices whose effective cells would have
        /// to be removed. If preprocessing welded a class, every original
        /// generator in the affected class is included.
        generator_indices: Vec<usize>,
        /// Exact stored-zero edges left after all generator-preserving
        /// contractions were performed.
        remaining_exact_zero_edges: usize,
    },

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
            VoronoiError::InvalidConfiguration(msg) => {
                write!(f, "invalid configuration: {}", msg)
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
            VoronoiError::CellEliminationRequired {
                generator_indices,
                remaining_exact_zero_edges,
            } => {
                let preview_len = generator_indices.len().min(8);
                write!(
                    f,
                    "output resolution would eliminate one or more effective generator cells while resolving {} remaining exact-zero edge(s); {} affected input generator(s) {:?}",
                    remaining_exact_zero_edges,
                    generator_indices.len(),
                    &generator_indices[..preview_len],
                )?;
                if preview_len < generator_indices.len() {
                    write!(f, " (and {} more)", generator_indices.len() - preview_len)?;
                }
                Ok(())
            }
            VoronoiError::ComputationFailed(msg) => {
                write!(f, "computation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for VoronoiError {}
