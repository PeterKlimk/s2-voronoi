#![feature(portable_simd)]
#![cfg_attr(test, feature(test))]

//! Spherical Voronoi diagrams on the unit sphere (S2).
//!
//! This crate computes Voronoi diagrams for points on the unit sphere using
//! a kNN-driven half-space clipping algorithm.
//!
//! # Example
//!
//! ```
//! use s2_voronoi::{compute, UnitVec3};
//!
//! // Generate some points on the unit sphere
//! let points = vec![
//!     UnitVec3::new(1.0, 0.0, 0.0),
//!     UnitVec3::new(0.0, 1.0, 0.0),
//!     UnitVec3::new(0.0, 0.0, 1.0),
//!     UnitVec3::new(-1.0, 0.0, 0.0),
//!     UnitVec3::new(0.0, -1.0, 0.0),
//!     UnitVec3::new(0.0, 0.0, -1.0),
//! ];
//!
//! let diagram = compute(&points).expect("computation should succeed");
//! assert_eq!(diagram.num_cells(), 6);
//! ```

mod diagram;
mod error;
mod fp;
mod types;
pub mod validation;

// Internal modules
pub(crate) mod cube_grid;
pub(crate) mod knn_clipping;

#[cfg(feature = "microbench")]
pub fn run_clip_convex_microbench() {
    knn_clipping::topo2d::run_clip_convex_microbench();
}

#[cfg(feature = "microbench")]
pub fn run_batch_clip_microbench() {
    knn_clipping::topo2d::run_batch_clip_microbench();
}

// Optional qhull backend (test/benchmark only)
#[cfg(feature = "qhull")]
pub mod convex_hull;

pub use diagram::{CellView, SphericalVoronoi, VoronoiCell};
pub use error::VoronoiError;
pub use types::{UnitVec3, UnitVec3Like};

#[cfg(feature = "qhull")]
pub use convex_hull::compute_voronoi_qhull;

/// Configuration for Voronoi computation.
#[derive(Debug, Clone)]
pub struct VoronoiConfig {
    /// If true, run a preprocessing pass to merge near-coincident generators.
    ///
    /// This improves robustness for pathological inputs with duplicates/near-duplicates,
    /// but adds overhead for large point sets. For benchmarking or when inputs are known
    /// to be well-spaced, disabling this can substantially improve performance.
    pub preprocess: bool,
    /// Optional override for the merge threshold used during preprocessing.
    /// When None, uses a density-based default.
    pub preprocess_threshold: Option<f32>,
    /// Optional cap on k during termination fallback (None = no cap).
    pub termination_max_k: Option<usize>,
}

impl Default for VoronoiConfig {
    fn default() -> Self {
        Self {
            preprocess: true,
            preprocess_threshold: None,
            termination_max_k: None,
        }
    }
}

/// Compute a spherical Voronoi diagram with default settings.
///
/// Errors are reserved for invalid inputs (e.g., insufficient points) or
/// unrecoverable internal failures.
pub fn compute<P: UnitVec3Like>(points: &[P]) -> Result<SphericalVoronoi, VoronoiError> {
    compute_with(points, VoronoiConfig::default())
}

/// Compute a spherical Voronoi diagram with explicit configuration.
pub fn compute_with<P: UnitVec3Like>(
    points: &[P],
    config: VoronoiConfig,
) -> Result<SphericalVoronoi, VoronoiError> {
    use glam::Vec3;

    if points.len() < 4 {
        return Err(VoronoiError::InsufficientPoints(points.len()));
    }

    // Convert input points to Vec3 for the backend
    let vec3_points: Vec<Vec3> = points
        .iter()
        .map(|p| Vec3::new(p.x(), p.y(), p.z()))
        .collect();

    // Call knn_clipping backend
    Ok(knn_clipping::compute_voronoi_gpu_style_with_config(
        &vec3_points,
        &config,
    ))
}
