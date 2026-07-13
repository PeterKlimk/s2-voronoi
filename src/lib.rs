#![deny(missing_docs)]

//! Spherical Voronoi diagrams on the unit sphere (S2).
//!
//! This crate computes Voronoi diagrams for points on the unit sphere using
//! a kNN-driven half-space clipping algorithm.
//!
//! ## Requirements
//!
//! - Stable Rust (MSRV 1.88).
//! - Input points are **assumed** to be unit length. They are canonicalized
//!   once at entry (renormalized in f64 and rounded back to f32), which
//!   absorbs mild off-unit drift; non-finite inputs are rejected with an
//!   error. Inputs far from unit length are outside the supported contract.
//!   For f64 world positions on a translated or scaled sphere, use
//!   [`compute_on_sphere`] with a [`SphereEmbedding`]; that API projects every
//!   non-center input radially before delegating to this unit-sphere backend.
//! - At least 4 points are required to form a non-degenerate diagram.
//!
//! ## Feature stability
//!
//! The documented features are `parallel` (default), `glam`, `serde`, and
//! `qhull`. Any other Cargo feature is an internal instrumentation or
//! benchmarking hook, exempt from semver: items they expose may change or
//! vanish in any release.
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
//! use voronoi_mesh::{compute, UnitVec3};
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
mod embedding;
mod error;
mod fp;
#[allow(clippy::too_many_arguments)] // generated sorting networks
pub(crate) mod generated;
mod measures;
mod packed_layout;
pub(crate) mod policy;
/// Diagnostic quality assessment (bench/comparison tooling only).
#[cfg(any(feature = "qhull", feature = "tools"))]
#[doc(hidden)]
pub mod quality;
pub(crate) mod sort;
/// Internal small-N sorting-network entry point, exposed for microbenchmarking only.
#[cfg(feature = "microbench")]
#[doc(hidden)]
pub use crate::sort::sort_small as bench_sort_small;
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
pub use embedding::{
    compute_on_sphere, compute_on_sphere_with, compute_on_sphere_with_report,
    EmbeddedComputeOutput, EmbeddedSphereLocator, EmbeddedSphericalVoronoi,
    IndexedSphereProjectionError, SphereEmbedding, SphereEmbeddingError, SphereProjectionError,
    WorldVec3Like,
};
pub use error::VoronoiError;
/// EXPERIMENTAL DIAGNOSTIC re-export — see the type's documentation; not
/// part of the supported API surface (taxonomy may change in patch releases).
#[doc(hidden)]
pub use live_dedup::UnresolvedEdgeOrigin;

/// P5 stage-1 shadow audit counters (feature `p5_shadow`); diagnostic only.
#[cfg(feature = "p5_shadow")]
#[doc(hidden)]
pub mod p5_shadow {
    pub use crate::knn_clipping::p5_shadow::{
        paired_dump_involving, paired_quad_report, paired_question_summaries, paired_report,
        paired_reset, report, reset, set_audit_cutoff, set_clip_eps_override,
        set_escalation_factor_override, set_pair_cutoff, set_pair_key_filter,
        set_term_pad_override,
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
}

pub use locate::SphereLocator;
pub use types::{UnitVec3, UnitVec3Like};

#[cfg(feature = "qhull")]
pub use convex_hull::compute_voronoi_qhull;

/// Preprocessing mode applied before Voronoi computation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum PreprocessMode {
    /// Do not weld near-coincident generators.
    ///
    /// Only safe when the caller certifies that no generators are closer than
    /// the weld radius; sub-radius clusters can otherwise fail construction.
    /// See `docs/correctness.md`.
    Disabled,
    /// Weld generators within the library's fixed weld radius (default).
    ///
    /// The radius (~1.4e-6 chord distance) is derived from f32 input rounding
    /// with a measured safety margin; welding below it is required for graph
    /// validity, not input hygiene. Welded generators share one cell in the
    /// returned diagram (see [`SphericalVoronoi::weld_map`]).
    Weld,
    /// Weld generators within an explicit Euclidean (chord) threshold.
    ///
    /// The threshold must be finite, positive, and large enough that its
    /// squared `f32` value is nonzero.
    MergeWithin(f32),
}

/// Post-assembly repair policy for rare near-degenerate topology defects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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

/// Policy for degenerate spherical inputs that do not have a stable
/// full-dimensional floating-point topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DegenerateMode {
    /// Return the backend's ordinary clean error for unsupported degenerate
    /// geometries. This is the default valid-or-error contract.
    Strict,
    /// After an initial failure, detect rank-2 great-circle inputs and retry
    /// once with a deterministic small off-plane perturbation. This returns a
    /// nearby full-dimensional diagram, not an exact lower-dimensional one.
    /// Welding does not imply this behavior; use this mode when rank-2
    /// great-circle inputs should produce an approximate valid diagram instead
    /// of a clean error.
    PerturbGreatCircle,
}

/// Observable preprocessing outcome for a computation run.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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

/// Observable degenerate-input handling outcome for a computation run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DegenerateReport {
    /// Requested degenerate-input policy.
    pub requested_mode: DegenerateMode,
    /// True when the computation retried with deterministic great-circle
    /// perturbation and returned that perturbed solved problem.
    pub perturbation_applied: bool,
}

/// Observable outcome of the optional post-assembly local repair pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RepairReport {
    /// A repair pass ran (defects were detected and the configured
    /// [`RepairMode`] is not `Disabled`). False on clean builds.
    pub attempted: bool,
    /// The repaired diagram passed whole-diagram strict validation and was
    /// committed. Always false when `attempted` is false.
    pub accepted: bool,
}

/// Observable per-run report for Voronoi computation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ComputeReport {
    /// What preprocessing did to the input.
    pub preprocess: PreprocessReport,
    /// What degenerate-input policy did to the input.
    pub degenerate: DegenerateReport,
    /// Strict validation of the returned public diagram.
    pub returned_validation: validation::ValidationReport,
    /// Strict validation of the effective preprocessed diagram, when preprocessing
    /// changed the solved generator set.
    pub effective_validation: Option<validation::ValidationReport>,
    /// Number of shared-edge mismatches observed during live dedup before the
    /// post-assembly reconciliation and optional local repair passes.
    ///
    /// A non-zero count proves a near-degenerate input exercised the repair
    /// machinery; it does not mean the returned diagram is invalid. Check
    /// [`ComputeReport::post_repair_unpaired_edges`] and
    /// [`ComputeReport::preferred_validation`] for the output contract.
    pub pre_repair_edge_mismatch_count: usize,
    /// What the local repair pass did this run.
    pub repair: RepairReport,
    /// Interior edges that remained unpaired after reconciliation and were not
    /// cleared by an accepted local repair. `compute` turns these into a loud
    /// error; `compute_with_report` surfaces them for diagnostics.
    pub post_repair_unpaired_edges: Vec<(u32, u32)>,
    /// EXPERIMENTAL DIAGNOSTIC (unsupported surface; taxonomy changes in
    /// patch releases): the pre-repair mismatches behind
    /// [`ComputeReport::pre_repair_edge_mismatch_count`], as effective-diagram
    /// generator pairs plus the detection path that recorded each.
    #[doc(hidden)]
    pub pre_repair_edge_mismatches: Vec<(u32, u32, UnresolvedEdgeOrigin)>,
    /// EXPERIMENTAL DIAGNOSTIC (unsupported surface; taxonomy changes in
    /// patch releases): unresolved shared-edge mismatches handed to
    /// post-assembly reconciliation, as effective-diagram generator pairs plus
    /// detection origins. Historical aggregate: contains
    /// `pre_repair_edge_mismatches` plus `PostRepairUnpaired` records when
    /// [`ComputeReport::post_repair_unpaired_edges`] is non-empty. Tests use
    /// this to prove defect-forcing inputs exercise each detection path (see
    /// `tests/edge_repair_net.rs`).
    #[doc(hidden)]
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

    /// True when the returned report contains output-invariant residuals that
    /// plain `compute` would reject with an error.
    #[inline]
    pub fn has_post_repair_residuals(&self) -> bool {
        !self.post_repair_unpaired_edges.is_empty()
            || self.preferred_validation().low_incidence_vertices > 0
    }
}

/// Output of `compute_with_report`.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
///
/// Construct with [`VoronoiConfig::default`] and adjust through the
/// `with_*` methods (or by assigning to the public fields); the struct is
/// `#[non_exhaustive]`, so it cannot be built with struct-literal syntax.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// Handling for rank-deficient great-circle inputs.
    ///
    /// The default is [`DegenerateMode::PerturbGreatCircle`]: rank-2
    /// great-circle failures retry as a deterministic nearby full-dimensional
    /// diagram and report that choice through [`ComputeReport::degenerate`].
    /// Use [`DegenerateMode::Strict`] to preserve the ordinary clean-error
    /// behavior for these lower-dimensional inputs.
    pub degenerate_mode: DegenerateMode,
}

impl Default for VoronoiConfig {
    fn default() -> Self {
        Self {
            preprocess_mode: PreprocessMode::Weld,
            repair_mode: RepairMode::Local3d,
            degenerate_mode: DegenerateMode::PerturbGreatCircle,
        }
    }
}

impl VoronoiConfig {
    /// Default config with the given [`PreprocessMode`].
    pub fn with_preprocess_mode(mut self, mode: PreprocessMode) -> Self {
        self.preprocess_mode = mode;
        self
    }

    /// Default config with the given [`RepairMode`].
    pub fn with_repair_mode(mut self, mode: RepairMode) -> Self {
        self.repair_mode = mode;
        self
    }

    /// Default config with the given [`DegenerateMode`].
    pub fn with_degenerate_mode(mut self, mode: DegenerateMode) -> Self {
        self.degenerate_mode = mode;
        self
    }
}

/// Compute a spherical Voronoi diagram with default settings.
///
/// Errors are returned for invalid inputs, unsupported geometry in the current
/// clipping model, or unrecoverable internal failures.
pub fn compute<P: UnitVec3Like>(points: &[P]) -> Result<SphericalVoronoi, VoronoiError> {
    compute_with(points, VoronoiConfig::default())
}

/// Shared entry preamble: reject sub-4 inputs and convert to the backend's
/// `Vec3` representation.
fn backend_points<P: UnitVec3Like>(points: &[P]) -> Result<Vec<glam::Vec3>, VoronoiError> {
    if points.len() < 4 {
        return Err(VoronoiError::InsufficientPoints(points.len()));
    }
    Ok(points
        .iter()
        .map(|p| glam::Vec3::new(p.x(), p.y(), p.z()))
        .collect())
}

/// Compute a spherical Voronoi diagram with explicit configuration.
pub fn compute_with<P: UnitVec3Like>(
    points: &[P],
    config: VoronoiConfig,
) -> Result<SphericalVoronoi, VoronoiError> {
    let vec3_points = backend_points(points)?;
    knn_clipping::compute_voronoi_knn_clipping_with_config_owned(vec3_points, &config)
}

/// Compute a spherical Voronoi diagram and return observable preprocessing and
/// validation metadata.
pub fn compute_with_report<P: UnitVec3Like>(
    points: &[P],
    config: VoronoiConfig,
) -> Result<ComputeOutput, VoronoiError> {
    let vec3_points = backend_points(points)?;
    knn_clipping::compute_voronoi_knn_clipping_with_report_owned(vec3_points, &config)
}
