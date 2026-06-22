#![deny(missing_docs)]

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
mod delaunay;
mod diagram;
mod error;
mod fp;
#[allow(clippy::too_many_arguments)] // generated sorting networks
pub(crate) mod generated;
mod measures;
mod packed_layout;
mod plane_diagram;
mod plane_measures;
pub(crate) mod policy;
#[doc(hidden)]
pub mod quality;
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
pub mod locate;
pub(crate) mod plane_clipping;
pub(crate) mod plane_grid;

/// Run the internal convex-clip microbench harness (feature: `microbench`).
#[cfg(feature = "microbench")]
pub fn run_clip_convex_microbench() {
    knn_clipping::topo2d::run_clip_convex_microbench();
}

/// Run the internal batch-clip microbench harness (feature: `microbench`).
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
pub use live_dedup::UnresolvedEdgeOrigin;

/// P5 stage-1 shadow audit counters (feature `p5_shadow`); diagnostic only.
#[cfg(feature = "p5_shadow")]
#[doc(hidden)]
pub mod p5_shadow {
    pub use crate::knn_clipping::p5_shadow::{
        paired_dump_involving, paired_quad_report, paired_question_summaries, paired_report,
        paired_reset, plane_unresolved, plane_unresolved_reset, report, reset, set_audit_cutoff,
        set_clip_eps_override, set_escalation_factor_override, set_pair_cutoff,
        set_pair_key_filter, set_plane_clip_eps_override, set_term_pad_override,
    };
}

/// Defect-driven escalation probe (feature `escalate_probe`); lets an
/// integration test drive the exact local rebuild over real defect sites.
/// Diagnostic only.
#[cfg(feature = "escalate_probe")]
#[doc(hidden)]
pub mod escalate_probe {
    pub use crate::knn_clipping::escalate::{
        check_cell_internally_paired, gather_local, rebuild_cells, set_escalation_enabled,
        shared_neighbor, take_a0_fast, RebuiltCell,
    };
    pub use crate::knn_clipping::proactive_audit::{
        reset as reset_proactive_audit, take as take_proactive_audit, CellAudit,
    };
}

/// Probe override for the dependency-free local repair pass.
///
/// Normal callers should use [`VoronoiConfig::repair_mode`]. This process-global
/// hook remains only for diagnostic tests that need to force the repair path
/// without rebuilding call sites.
#[doc(hidden)]
pub fn set_escalation_enabled(on: bool) {
    crate::knn_clipping::escalate::set_escalation_enabled(on);
}

pub use locate::{PlaneLocator, SphereLocator};
pub use plane_diagram::{PlanarVoronoi, PlanePoint, PlanePointLike, PlaneRect, PlaneTopology};
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

/// Post-assembly repair policy for rare near-degenerate topology defects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairMode {
    /// Do not run local repair. Residual unpaired edges remain
    /// observable through `compute_with_report` and make plain `compute` fail.
    Disabled,
    /// Rebuild residual near-degenerate neighborhoods with one normalized local
    /// 3D hull and accept only if whole-diagram validation passes.
    Local3d,
    /// Rebuild residual near-degenerate neighborhoods with one shared local
    /// stereographic Delaunay and accept only if whole-diagram validation passes.
    ///
    /// This remains available as a projected-oracle diagnostic path. The default
    /// uses [`RepairMode::Local3d`], which avoids the large-chart failure mode of
    /// projected repair in extreme closures.
    LocalProjected,
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
    /// Unresolved shared-edge mismatches that survived live dedup and were
    /// handed to post-assembly edge reconciliation, as pairs of
    /// effective-diagram generator indices plus the detection path that
    /// recorded each (empty means the repair pass had nothing to do).
    /// Diagnostic: tests use this to prove defect-forcing inputs actually
    /// exercise the repair paths (see `tests/edge_repair_net.rs`).
    pub unresolved_edge_pairs: Vec<(u32, u32, UnresolvedEdgeOrigin)>,
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
    /// Cold-path repair for rare near-degenerate clipping defects.
    ///
    /// The default tries normalized local 3D repair and accepts it only when strict
    /// validation succeeds. Disable this for diagnostics or to reproduce the raw
    /// fast-path residual/error behavior.
    pub repair_mode: RepairMode,
}

impl Default for VoronoiConfig {
    fn default() -> Self {
        Self {
            preprocess_mode: PreprocessMode::Weld,
            repair_mode: RepairMode::Local3d,
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
/// Generators within the planar weld radius (~1e-6 of the longer rect
/// side) are welded to one cell (see [`PlanarVoronoi::weld_map`]); this
/// always includes exact duplicates. As on the sphere, the radius weld is
/// required for graph validity, not input hygiene: probing shows clusters
/// of 3+ generators within ~1 ulp of the domain scale produce invalid
/// topology even though every individual bisector is well-formed. The weld
/// detection reuses the kNN spatial grid, so duplicate-free inputs pay
/// only a read-only scan.
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

/// Observability for a planar computation (bounded or periodic).
#[derive(Debug, Clone)]
pub struct PlaneComputeReport {
    /// Strict validation of the returned diagram.
    pub validation: validation::PlaneValidationReport,
    /// Post-repair unpaired interior edges as generator pairs (empty on a
    /// valid result). These are the residuals the plain `compute_plane` /
    /// `compute_plane_periodic` paths turn into an error; the report path
    /// surfaces them instead. Indices are effective-generator indices
    /// (equal to original input indices when no welding occurred).
    pub unresolved_edge_pairs: Vec<(u32, u32)>,
}

/// Output of [`compute_plane_with_report`] / [`compute_plane_periodic_with_report`].
#[derive(Debug, Clone)]
pub struct PlaneComputeOutput {
    /// The returned diagram (one cell per input point; welded twins share
    /// their canonical cell).
    pub diagram: PlanarVoronoi,
    /// Validation and residual observability for this run.
    pub report: PlaneComputeReport,
}

/// Compute a bounded planar Voronoi diagram and return observability
/// metadata instead of erroring on a post-repair residual.
///
/// [`compute_plane`] fails loud if reconciliation leaves an unpaired
/// interior edge (provably-invalid output with no other signal channel).
/// This variant returns the diagram together with a [`PlaneComputeReport`]
/// carrying strict validation and the residual generator pairs, so callers
/// that want to inspect (rather than be interrupted by) a defect can.
pub fn compute_plane_with_report<P: PlanePointLike>(
    points: &[P],
    rect: PlaneRect,
) -> Result<PlaneComputeOutput, VoronoiError> {
    plane_clipping::compute::compute_plane_with_report_impl(points, rect)
}

/// Compute a planar Voronoi diagram on a rectangular torus (periodic
/// boundary conditions): the rect's opposite edges are identified, so cells
/// wrap across them and the diagram has no boundary.
///
/// Vertex positions are stored canonically wrapped into the rect; use
/// [`PlanarVoronoi::cell_polygon`] to reconstruct a cell's polygon (it
/// unwraps each vertex to within half a period of the cell's generator).
/// The result is a strict subdivision of the torus: every edge is shared by
/// exactly two cells and `validation::validate_plane` checks the torus
/// Euler relation.
///
/// Every cell must be provably smaller than a quarter of the shorter
/// period for nearest-image clipping to be exact; underpopulated domains
/// fail with [`VoronoiError::UnsupportedGeometry`] instead of producing
/// wrong answers (roughly: dozens of reasonably spread generators are
/// enough; 3 points on a torus are not).
///
/// # Example
///
/// ```
/// use s2_voronoi::{compute_plane_periodic, PlaneRect};
///
/// let points: Vec<[f32; 2]> = (0..64)
///     .map(|i| [((i % 8) as f32 + 0.5) / 8.0, ((i / 8) as f32 + 0.37) / 8.0])
///     .collect();
/// let diagram = compute_plane_periodic(&points, PlaneRect::unit()).unwrap();
/// assert!(diagram.is_periodic());
/// assert_eq!(diagram.num_cells(), 64);
/// ```
pub fn compute_plane_periodic<P: PlanePointLike>(
    points: &[P],
    rect: PlaneRect,
) -> Result<PlanarVoronoi, VoronoiError> {
    plane_clipping::compute::compute_plane_periodic_impl(points, rect)
}

/// Periodic counterpart of [`compute_plane_with_report`]: returns the torus
/// diagram plus validation/residual observability instead of erroring on a
/// post-repair residual.
pub fn compute_plane_periodic_with_report<P: PlanePointLike>(
    points: &[P],
    rect: PlaneRect,
) -> Result<PlaneComputeOutput, VoronoiError> {
    plane_clipping::compute::compute_plane_periodic_with_report_impl(points, rect)
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
