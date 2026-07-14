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
//! The documented features are `parallel` (default), `glam`, and `serde`.
//! Any other Cargo feature is an internal instrumentation or benchmarking
//! hook, exempt from semver: items they expose may change or vanish in any
//! release.
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
//! A caller that explicitly accepts removal of effective cells whose geometry
//! collapses at f32 output resolution can consume [`ComputeOutput`] through
//! [`ComputeOutput::into_elided_cell_mesh`]. The resulting
//! [`SphericalCellMesh`] is separately typed and does not claim Voronoi
//! locator, Delaunay, or Lloyd semantics.
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
mod cell_mesh;
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
#[cfg(feature = "tools")]
#[doc(hidden)]
pub mod quality;
pub(crate) mod sort;
mod spherical_arc;
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

pub use adjacency::CellAdjacency;
pub use cell_mesh::{
    CellElisionError, CellElisionErrorKind, CellElisionReport, CellMeshCellView, CellMeshOutput,
    CellMeshValidationReport, SphericalCellMesh,
};
pub use diagram::{CellView, SphericalVoronoi};
pub use embedding::{
    compute_on_sphere, compute_on_sphere_with, compute_on_sphere_with_report,
    EmbeddedCellElisionError, EmbeddedCellMeshOutput, EmbeddedComputeOutput, EmbeddedSphereLocator,
    EmbeddedSphericalCellMesh, EmbeddedSphericalVoronoi, IndexedSphereProjectionError,
    SphereEmbedding, SphereEmbeddingError, SphereProjectionError, WorldVec3Like,
};
pub use error::VoronoiError;
/// EXPERIMENTAL DIAGNOSTIC re-export — see the type's documentation; not
/// part of the supported API surface (taxonomy may change in patch releases).
#[doc(hidden)]
pub use live_dedup::UnresolvedEdgeOrigin;

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
    /// validity, not input hygiene. Pairs are classified by a strict computed
    /// f32 squared-distance comparison; equality is not welded. Connected
    /// components are welded transitively to their lowest original index, so
    /// a chain's endpoints need not lie within one radius. Welded generators
    /// share one cell in the returned diagram (see
    /// [`SphericalVoronoi::weld_map`]).
    Weld,
    /// Weld generators within an explicit Euclidean (chord) threshold.
    ///
    /// The threshold must be finite, positive, and large enough that its
    /// squared `f32` value is nonzero. It uses the same strict, transitive
    /// threshold-graph semantics as [`PreprocessMode::Weld`].
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
    /// geometries without applying a perturbation.
    Strict,
    /// After an initial failure, detect affinely coplanar spherical inputs and
    /// retry once with a deterministic small off-plane perturbation. Exact
    /// affine coplanarity is certified over the canonical f32 generators;
    /// near-coplanar full great-circle inputs retain a conservative tolerance
    /// classifier. This returns a nearby full-dimensional diagram, not an
    /// exact lower-dimensional one.
    PerturbCoplanar,
}

/// Policy when resolving exact stored-zero edges would eliminate an effective
/// generator cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CellKillingPolicy {
    /// Retain the zero geometry so every effective generator keeps a cell.
    /// This is the default.
    Preserve,
    /// Return [`VoronoiError::CellEliminationRequired`] after performing every
    /// safe exact-zero contraction.
    Error,
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

/// Observable outcome of final exact-zero output canonicalization.
///
/// The current baseline preserves one effective cell per effective generator:
/// exact stored-zero components are contracted only when every cell remains
/// representable. Positive-length edge simplification is not included here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct OutputResolutionReport {
    /// Unique exact stored-zero edges found after reconciliation and optional
    /// Local3d repair.
    pub exact_zero_edges_detected: usize,
    /// Connected exact stored-zero vertex components found after repair.
    pub exact_zero_components_detected: usize,
    /// Exact stored-zero edges removed by committed transactions.
    pub exact_zero_edges_contracted: usize,
    /// Exact stored-zero components removed by committed transactions.
    pub exact_zero_components_contracted: usize,
    /// Components retained because contracting their interacting transaction
    /// group would eliminate at least one effective generator cell.
    pub cell_killing_components_preserved: usize,
    /// Components retained because the rewritten graph failed a structural or
    /// local quotient check for a reason other than cell elimination.
    pub topology_rejected_components: usize,
    /// Detected exact stored-zero edges still present under the
    /// generator-preserving policy after all accepted transactions. The
    /// independent validation report is authoritative for the final diagram.
    pub exact_zero_edges_remaining: usize,
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
    /// What final exact-zero output canonicalization did this run.
    pub output_resolution: OutputResolutionReport,
    /// Interior edges that remained unpaired, overused, or misoriented after
    /// reconciliation and were not cleared by an accepted local repair.
    /// Historical field name retained for API stability. `compute` turns these
    /// into a loud error; `compute_with_report` surfaces them for diagnostics.
    pub post_repair_unpaired_edges: Vec<(u32, u32)>,
    /// EXPERIMENTAL DIAGNOSTIC: reconciliation cell pairs rejected by the
    /// no-chain diameter policy and not cleared by an accepted Local3d repair.
    #[doc(hidden)]
    pub post_repair_escalation_pairs: Vec<(u32, u32)>,
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
            || !self.post_repair_escalation_pairs.is_empty()
            || !self.preferred_validation().is_strictly_valid()
            || (self.repair.attempted && !self.repair.accepted)
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
    /// Handling for rank-deficient coplanar inputs.
    ///
    /// The default is [`DegenerateMode::PerturbCoplanar`]: certified affine
    /// circle and conservatively detected full great-circle failures retry as
    /// a deterministic nearby full-dimensional diagram and report that choice
    /// through [`ComputeReport::degenerate`]. Use [`DegenerateMode::Strict`] to
    /// preserve the ordinary clean-error behavior for these lower-dimensional
    /// inputs.
    pub degenerate_mode: DegenerateMode,
    /// Handling for exact stored-zero contractions that would leave an
    /// effective generator with fewer than three boundary vertices.
    ///
    /// The default is [`CellKillingPolicy::Preserve`]. This is a separate
    /// output-stage control from preprocessing welding, but sufficient welding
    /// is intended to provide the separation floor that prevents whole-cell
    /// collapse. The policy remains the opt-out and residual safety net.
    pub cell_killing_policy: CellKillingPolicy,
}

impl Default for VoronoiConfig {
    fn default() -> Self {
        Self {
            preprocess_mode: PreprocessMode::Weld,
            repair_mode: RepairMode::Local3d,
            degenerate_mode: DegenerateMode::PerturbCoplanar,
            cell_killing_policy: CellKillingPolicy::Preserve,
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

    /// Default config with the given [`CellKillingPolicy`].
    pub fn with_cell_killing_policy(mut self, policy: CellKillingPolicy) -> Self {
        self.cell_killing_policy = policy;
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
