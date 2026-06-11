#![warn(missing_docs)]

//! Spherical Voronoi diagrams on the unit sphere (S2).
//!
//! This crate computes Voronoi diagrams for points on the unit sphere using
//! a kNN-driven half-space clipping algorithm.
//!
//! ## Requirements
//!
//! - Stable Rust (MSRV 1.88).
//! - Input points are **assumed** to be unit length (not normalized by the API).
//! - At least 4 points are required to form a non-degenerate diagram.
//!
//! ## Output
//!
//! The result is a [`SphericalVoronoi`] containing:
//! - generator points, accessible via [`SphericalVoronoi::generators`] / [`SphericalVoronoi::generator`]
//! - shared Voronoi vertices, accessible via [`SphericalVoronoi::vertices`] / [`SphericalVoronoi::vertex`]
//! - per-cell vertex index lists, accessible via [`SphericalVoronoi::cell`] / [`SphericalVoronoi::iter_cells`]
//!
//! For strict subdivision and exact-invariant checks on computed output, use
//! [`validation::validate`].
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

pub mod adjacency;
mod diagram;
mod error;
mod fp;
#[cfg(any(test, feature = "packed_knn_sort_small"))]
#[allow(clippy::too_many_arguments)] // generated sorting networks
pub(crate) mod generated;
mod measures;
mod packed_layout;
mod plane_diagram;
pub(crate) mod policy;
#[doc(hidden)]
pub mod quality;
#[cfg(any(test, feature = "packed_knn_sort_small"))]
pub(crate) mod sort;
pub(crate) mod timing;
pub(crate) mod tolerances;
mod types;
pub mod validation;

// Conditionally-parallel iteration macros, used by cube_grid and knn_clipping.
// Note: call sites must have `use rayon::prelude::*` in scope when `parallel` is enabled.

/// Conditionally parallel `into_iter` over an owned collection or range.
macro_rules! maybe_par_into_iter {
    ($v:expr) => {{
        #[cfg(feature = "parallel")]
        {
            $v.into_par_iter()
        }
        #[cfg(not(feature = "parallel"))]
        {
            $v.into_iter()
        }
    }};
}

/// Conditionally parallel `iter` over a slice reference.
macro_rules! maybe_par_iter {
    ($slice:expr) => {{
        #[cfg(feature = "parallel")]
        {
            $slice.par_iter()
        }
        #[cfg(not(feature = "parallel"))]
        {
            $slice.iter()
        }
    }};
}

// Internal modules
pub(crate) mod cube_grid;
pub(crate) mod knn_clipping;
pub(crate) mod live_dedup;
pub(crate) mod plane_clipping;
pub(crate) mod plane_grid;

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

pub use adjacency::CellAdjacency;
pub use diagram::{CellView, SphericalVoronoi};
pub use error::VoronoiError;
pub use plane_diagram::{PlanarVoronoi, PlanePoint, PlanePointLike, PlaneRect};
pub use types::{UnitVec3, UnitVec3Like};

#[cfg(feature = "qhull")]
pub use convex_hull::compute_voronoi_qhull;

/// Preprocessing mode applied before Voronoi computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreprocessMode {
    /// Do not weld near-coincident generators.
    ///
    /// Only safe when the caller certifies that no generators are closer than
    /// the weld radius; sub-radius clusters can otherwise fail construction.
    /// See `docs/correctness-contract.md`.
    Disabled,
    /// Weld generators within the library's fixed weld radius (default).
    ///
    /// The radius (~1.4e-6 chord distance) is derived from f32 input rounding
    /// with a measured safety margin; welding below it is required for graph
    /// validity, not input hygiene. Welded generators share one cell in the
    /// returned diagram (see [`SphericalVoronoi::weld_map`]).
    Weld,
    /// Weld generators within an explicit Euclidean (chord) threshold.
    MergeWithin(f32),
}

/// Observable preprocessing outcome for a computation run.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessReport {
    /// Requested preprocessing mode.
    pub requested_mode: PreprocessMode,
    /// Threshold actually used for merging, if preprocessing ran.
    pub threshold_used: Option<f32>,
    /// Number of generators passed into the computation.
    pub original_points: usize,
    /// Number of effective generators after preprocessing.
    pub effective_points: usize,
    /// Number of generators merged away before cell construction.
    pub num_merged: usize,
}

impl PreprocessReport {
    /// True when preprocessing welded at least one generator pair.
    #[inline]
    pub fn did_merge(&self) -> bool {
        self.num_merged > 0
    }

    /// True when the returned diagram's cells were remapped back to the
    /// original input indices (always the case when welds occurred).
    #[inline]
    pub fn did_remap_cells(&self) -> bool {
        self.did_merge()
    }
}

/// Observable per-run report for Voronoi computation.
#[derive(Debug, Clone)]
pub struct ComputeReport {
    /// What preprocessing did to the input.
    pub preprocess: PreprocessReport,
    /// Strict validation of the returned public diagram.
    pub returned_validation: validation::ValidationReport,
    /// Strict validation of the effective preprocessed diagram, when preprocessing
    /// changed the solved generator set.
    pub effective_validation: Option<validation::ValidationReport>,
}

impl ComputeReport {
    /// Preferred strict validation view for this computation.
    ///
    /// When preprocessing merged generators, this returns the validation result
    /// for the effective diagram actually solved by the backend. Otherwise it
    /// returns the validation result for the returned diagram.
    #[inline]
    pub fn preferred_validation(&self) -> &validation::ValidationReport {
        self.effective_validation
            .as_ref()
            .unwrap_or(&self.returned_validation)
    }
}

/// Output of `compute_with_report`.
#[derive(Debug, Clone)]
pub struct ComputeOutput {
    /// The returned diagram (one cell per input point; welded twins share
    /// their canonical cell).
    pub diagram: SphericalVoronoi,
    /// Effective diagram actually solved by the backend after preprocessing, if
    /// preprocessing merged generators.
    pub effective_diagram: Option<SphericalVoronoi>,
    /// Preprocessing and validation observability for this run.
    pub report: ComputeReport,
}

impl ComputeOutput {
    /// Preferred diagram view for this computation.
    ///
    /// When preprocessing merged generators, this returns the effective
    /// preprocessed diagram actually solved by the backend. Otherwise it
    /// returns the public remapped diagram.
    #[inline]
    pub fn preferred_diagram(&self) -> &SphericalVoronoi {
        self.effective_diagram.as_ref().unwrap_or(&self.diagram)
    }
}

/// Configuration for Voronoi computation.
#[derive(Debug, Clone)]
pub struct VoronoiConfig {
    /// Preprocessing applied before Voronoi computation.
    ///
    /// The default welds generators within a fixed sub-resolvability radius
    /// (~1.4e-6 chord). Welded input indices share one cell in the returned
    /// diagram rather than receiving duplicated boundaries, so strict
    /// validation passes whether or not welds occur.
    pub preprocess_mode: PreprocessMode,
    /// Enable a cold-path packed r=2 expansion band before the shell-expansion
    /// takeover.
    ///
    /// Default is `true`. The expansion is only built for queries that exhaust
    /// the existing packed r=1 path without proving termination.
    pub packed_knn_expand_r2: bool,
}

impl Default for VoronoiConfig {
    fn default() -> Self {
        Self {
            preprocess_mode: PreprocessMode::Weld,
            packed_knn_expand_r2: true,
        }
    }
}

/// Compute a spherical Voronoi diagram with default settings.
///
/// Errors are returned for invalid inputs, unsupported geometry in the current
/// clipping model, or unrecoverable internal failures.
pub fn compute<P: UnitVec3Like>(points: &[P]) -> Result<SphericalVoronoi, VoronoiError> {
    compute_with(points, VoronoiConfig::default())
}

/// Compute a planar Voronoi diagram over a bounded rectangle.
///
/// Every input point must lie inside `rect` (boundary inclusive); hull cells
/// are clipped to the rect, whose walls behave like virtual generators. The
/// result is a strict subdivision of the rect: cell areas sum to the rect
/// area and every interior edge is shared by exactly two cells. Use
/// [`validation::validate_plane`] for strict invariant checks.
///
/// Points that coincide after domain normalization are welded (see
/// [`PlanarVoronoi::weld_map`]). This always includes exact-bit duplicate
/// inputs, and may include inputs so close (sub-ulp at the rect's scale)
/// that mapping into the unit-square computation space rounds them to the
/// same coordinates. Unlike the sphere there is no near-coincidence weld
/// *radius*: points that remain distinct in normalized space always yield
/// well-formed bisectors in the f64 clip math.
///
/// Requires at least 1 point (a single generator owns the whole rect).
///
/// # Example
///
/// ```
/// use s2_voronoi::{compute_plane, PlaneRect};
///
/// let points = vec![[0.25f32, 0.25], [0.75, 0.25], [0.5, 0.8]];
/// let diagram = compute_plane(&points, PlaneRect::unit()).unwrap();
/// assert_eq!(diagram.num_cells(), 3);
/// ```
pub fn compute_plane<P: PlanePointLike>(
    points: &[P],
    rect: PlaneRect,
) -> Result<PlanarVoronoi, VoronoiError> {
    plane_clipping::compute::compute_plane_impl(points, rect)
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

    let vec3_points: Vec<Vec3> = points
        .iter()
        .map(|p| Vec3::new(p.x(), p.y(), p.z()))
        .collect();

    knn_clipping::compute_voronoi_knn_clipping_with_config_owned(vec3_points, &config)
}

/// Compute a spherical Voronoi diagram and return observable preprocessing and
/// validation metadata.
pub fn compute_with_report<P: UnitVec3Like>(
    points: &[P],
    config: VoronoiConfig,
) -> Result<ComputeOutput, VoronoiError> {
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
    knn_clipping::compute_voronoi_knn_clipping_with_report_owned(vec3_points, &config)
}
