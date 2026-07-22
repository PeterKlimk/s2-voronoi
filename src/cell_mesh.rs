//! Spherical cell meshes produced by explicit output-resolution simplification.

use crate::{CellAdjacency, SpherePoint};
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

const NO_CELL: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct MeshCellData {
    start: u32,
    len: u16,
}

/// A connected, oriented cell decomposition of the unit sphere.
///
/// Unlike [`crate::SphericalVoronoi`], this type does not claim that its
/// boundaries are exact bisectors of the retained source sites. It is produced
/// by an explicit output-resolution operation which may remove cells that
/// cannot be represented with nonzero stored geometry.
///
/// Storage is dense: every stored vertex is referenced by at least one cell.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "CellMeshWire"))]
pub struct SphericalCellMesh {
    vertices: Vec<SpherePoint>,
    cells: Vec<MeshCellData>,
    cell_indices: Vec<u32>,
    cell_source_sites: Vec<SpherePoint>,
    cell_to_input: Vec<u32>,
    input_to_cell: Vec<u32>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct CellMeshWire {
    vertices: Vec<SpherePoint>,
    cells: Vec<MeshCellData>,
    cell_indices: Vec<u32>,
    cell_source_sites: Vec<SpherePoint>,
    cell_to_input: Vec<u32>,
    input_to_cell: Vec<u32>,
}

#[cfg(feature = "serde")]
impl TryFrom<CellMeshWire> for SphericalCellMesh {
    type Error = String;

    fn try_from(wire: CellMeshWire) -> Result<Self, Self::Error> {
        for (cell, data) in wire.cells.iter().enumerate() {
            let end = data
                .start
                .checked_add(data.len as u32)
                .ok_or_else(|| format!("cell {cell} boundary span overflows u32"))?;
            if end as usize > wire.cell_indices.len() {
                return Err(format!(
                    "cell {cell} boundary span exceeds index buffer length {}",
                    wire.cell_indices.len()
                ));
            }
        }
        let mesh = Self {
            vertices: wire.vertices,
            cells: wire.cells,
            cell_indices: wire.cell_indices,
            cell_source_sites: wire.cell_source_sites,
            cell_to_input: wire.cell_to_input,
            input_to_cell: wire.input_to_cell,
        };
        let validation = mesh.validate();
        if validation.is_strictly_valid() {
            Ok(mesh)
        } else {
            Err(validation.headline())
        }
    }
}

impl SphericalCellMesh {
    pub(crate) fn from_raw_parts(
        vertices: Vec<SpherePoint>,
        cell_cycles: Vec<Vec<u32>>,
        cell_source_sites: Vec<SpherePoint>,
        cell_to_input: Vec<u32>,
        input_to_cell: Vec<Option<u32>>,
    ) -> Self {
        debug_assert_eq!(cell_cycles.len(), cell_source_sites.len());
        debug_assert_eq!(cell_cycles.len(), cell_to_input.len());
        #[cfg(feature = "profiling")]
        {
            for &vertex in &vertices {
                crate::point_audit::record_sphere_point(
                    crate::point_audit::PointProducer::CellMeshVertex,
                    vertex,
                );
            }
            for &site in &cell_source_sites {
                crate::point_audit::record_sphere_point(
                    crate::point_audit::PointProducer::CellMeshSourceSite,
                    site,
                );
            }
        }
        let total_indices = cell_cycles.iter().map(Vec::len).sum();
        let mut cells = Vec::with_capacity(cell_cycles.len());
        let mut cell_indices = Vec::with_capacity(total_indices);
        for cycle in cell_cycles {
            cells.push(MeshCellData {
                start: cell_indices.len() as u32,
                len: cycle.len() as u16,
            });
            cell_indices.extend(cycle);
        }
        Self {
            vertices,
            cells,
            cell_indices,
            cell_source_sites,
            cell_to_input,
            input_to_cell: input_to_cell
                .into_iter()
                .map(|cell| cell.unwrap_or(NO_CELL))
                .collect(),
        }
    }

    /// Number of cells in the simplified mesh.
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// Number of densely stored mesh vertices.
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Borrow all mesh vertices.
    #[inline]
    pub fn vertices(&self) -> &[SpherePoint] {
        &self.vertices
    }

    /// Borrow mesh vertex coordinates as tightly packed xyz triples.
    #[inline]
    pub fn vertices_xyz(&self) -> &[[f32; 3]] {
        crate::types::sphere_points_as_xyz(&self.vertices)
    }

    /// Return a mesh vertex.
    ///
    /// # Panics
    ///
    /// Panics when `index >= self.num_vertices()`.
    #[inline]
    #[track_caller]
    pub fn vertex(&self, index: usize) -> SpherePoint {
        self.vertices[index]
    }

    /// Checked form of [`Self::vertex`].
    #[inline]
    pub fn get_vertex(&self, index: usize) -> Option<SpherePoint> {
        self.vertices.get(index).copied()
    }

    /// Return one ordered cell-boundary view.
    ///
    /// # Panics
    ///
    /// Panics when `index >= self.num_cells()`.
    #[inline]
    #[track_caller]
    pub fn cell(&self, index: usize) -> CellMeshCellView<'_> {
        let data = self.cells[index];
        let start = data.start as usize;
        CellMeshCellView {
            cell_index: index,
            vertex_indices: &self.cell_indices[start..start + data.len as usize],
        }
    }

    /// Checked form of [`Self::cell`].
    #[inline]
    pub fn get_cell(&self, index: usize) -> Option<CellMeshCellView<'_>> {
        (index < self.num_cells()).then(|| self.cell(index))
    }

    /// Iterate over all cells in compact cell-index order.
    pub fn iter_cells(&self) -> impl Iterator<Item = CellMeshCellView<'_>> {
        (0..self.num_cells()).map(|cell| self.cell(cell))
    }

    /// Number of original input indices represented by the provenance map.
    #[inline]
    pub fn num_source_inputs(&self) -> usize {
        self.input_to_cell.len()
    }

    /// Final cell for an original input, or `None` when that input's effective
    /// cell was elided.
    ///
    /// Welded original inputs return the same final cell. This method panics
    /// for an out-of-range input; use [`Self::get_cell_for_input`] for checked
    /// access.
    #[inline]
    #[track_caller]
    pub fn cell_for_input(&self, input: usize) -> Option<usize> {
        self.get_cell_for_input(input).unwrap_or_else(|| {
            panic!(
                "input index {input} out of bounds (num_source_inputs {})",
                self.num_source_inputs()
            )
        })
    }

    /// Checked form of [`Self::cell_for_input`]. The outer `Option` represents
    /// bounds checking; the inner `Option` distinguishes an elided input.
    #[inline]
    pub fn get_cell_for_input(&self, input: usize) -> Option<Option<usize>> {
        self.input_to_cell
            .get(input)
            .map(|&cell| (cell != NO_CELL).then_some(cell as usize))
    }

    /// Canonical original input index attributed to a final cell.
    ///
    /// This is provenance only, not a claim that every mesh edge remains a
    /// bisector for the source input. Panics for an out-of-range cell.
    #[inline]
    #[track_caller]
    pub fn source_input_index(&self, cell: usize) -> usize {
        self.cell_to_input[cell] as usize
    }

    /// Checked form of [`Self::source_input_index`].
    #[inline]
    pub fn get_source_input_index(&self, cell: usize) -> Option<usize> {
        self.cell_to_input.get(cell).map(|&input| input as usize)
    }

    /// Stored canonicalized source-site direction attributed to a final cell.
    ///
    /// The direction may reflect deterministic coplanar perturbation. It is
    /// retained for attribution and does not give this mesh Voronoi locator,
    /// Delaunay, or Lloyd semantics. Panics for an out-of-range cell.
    #[inline]
    #[track_caller]
    pub fn source_site(&self, cell: usize) -> SpherePoint {
        self.cell_source_sites[cell]
    }

    /// Checked form of [`Self::source_site`].
    #[inline]
    pub fn get_source_site(&self, cell: usize) -> Option<SpherePoint> {
        self.cell_source_sites.get(cell).copied()
    }

    /// Borrow source-site directions as tightly packed xyz triples.
    #[inline]
    pub fn source_sites_xyz(&self) -> &[[f32; 3]] {
        crate::types::sphere_points_as_xyz(&self.cell_source_sites)
    }

    /// Build combinatorial cell adjacency aligned with boundary edges.
    ///
    /// Entry `k` of the returned adjacency for cell `i` is the cell across the
    /// edge from boundary vertex `k` to `k + 1` (cyclic). Unlike adjacency on
    /// [`crate::SphericalVoronoi`], this has no Delaunay interpretation.
    pub fn build_adjacency(&self) -> CellAdjacency {
        crate::adjacency::build_adjacency_from_parts(
            self.num_cells(),
            |cell| {
                let data = self.cells[cell];
                (data.start, data.len)
            },
            &self.cell_indices,
            |cell| cell,
        )
    }

    /// Validate the generic oriented S2 cell-complex contract.
    pub fn validate(&self) -> CellMeshValidationReport {
        validate_cell_mesh(self)
    }
}

/// Borrowed view of one cell in a [`SphericalCellMesh`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct CellMeshCellView<'a> {
    /// Compact mesh cell index.
    pub cell_index: usize,
    /// Ordered vertex indices around the cell boundary.
    pub vertex_indices: &'a [u32],
}

impl CellMeshCellView<'_> {
    /// Number of boundary vertices.
    #[inline]
    pub fn len(&self) -> usize {
        self.vertex_indices.len()
    }

    /// Whether this boundary has no vertices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertex_indices.is_empty()
    }
}

/// Validation report for a generic spherical cell mesh.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CellMeshValidationReport {
    /// Number of cells.
    pub num_cells: usize,
    /// Number of stored vertices.
    pub num_vertices: usize,
    /// Number of unique undirected edges.
    pub num_edges: usize,
    /// Euler characteristic `V - E + F`.
    pub euler_characteristic: i32,
    /// Number of connected components in the cell adjacency graph.
    pub connected_components: usize,
    /// Cells with fewer than three distinct vertices.
    pub degenerate_cells: usize,
    /// Cells whose vertex records occupy fewer than three distinct exact
    /// stored directions.
    pub cells_with_fewer_than_three_stored_positions: usize,
    /// Cells containing a repeated vertex index.
    pub cells_with_duplicate_vertices: usize,
    /// Cells containing an out-of-range vertex reference.
    pub cells_with_invalid_references: usize,
    /// Duplicate cell-boundary vertex sets.
    pub duplicate_cells: usize,
    /// Vertices outside the unit-sphere storage tolerance.
    pub vertices_off_sphere: usize,
    /// Stored vertices referenced by no cell. Dense meshes contain none.
    pub orphan_vertices: usize,
    /// Referenced vertices incident to fewer than three cells.
    pub low_incidence_vertices: usize,
    /// Vertices whose incident face link is not one directed cycle.
    pub disconnected_vertex_links: usize,
    /// Edges with only one incident cell.
    pub boundary_edges: usize,
    /// Edges with more than two incident cells.
    pub overused_edges: usize,
    /// Twice-used edges whose owners traverse them in the same direction.
    pub same_direction_edge_pairs: usize,
    /// Edges whose distinct endpoint records have identical stored geometry.
    pub zero_length_edges: usize,
    /// Edges with exactly antipodal stored endpoints and therefore no unique
    /// shorter great-circle arc.
    pub antipodal_edges: usize,
    /// Inconsistencies in source-site and input/cell provenance mappings.
    pub provenance_issues: usize,
}

impl CellMeshValidationReport {
    /// Whether the mesh is a connected, oriented, closed S2 subdivision with
    /// dense valid storage and coherent provenance.
    pub fn is_strictly_valid(&self) -> bool {
        self.num_cells > 0
            && self.num_vertices > 0
            && self.euler_characteristic == 2
            && self.connected_components == 1
            && self.degenerate_cells == 0
            && self.cells_with_fewer_than_three_stored_positions == 0
            && self.cells_with_duplicate_vertices == 0
            && self.cells_with_invalid_references == 0
            && self.duplicate_cells == 0
            && self.vertices_off_sphere == 0
            && self.orphan_vertices == 0
            && self.low_incidence_vertices == 0
            && self.disconnected_vertex_links == 0
            && self.boundary_edges == 0
            && self.overused_edges == 0
            && self.same_direction_edge_pairs == 0
            && self.zero_length_edges == 0
            && self.antipodal_edges == 0
            && self.provenance_issues == 0
    }

    /// Short diagnostic summary. Wording is not a stable API contract.
    pub fn headline(&self) -> String {
        if self.is_strictly_valid() {
            return format!(
                "strictly valid spherical cell mesh (V={}, E={}, F={})",
                self.num_vertices, self.num_edges, self.num_cells
            );
        }
        format!(
            "invalid spherical cell mesh (chi={}, components={}, degenerate={}, stored_degenerate={}, edge_issues={}, link_issues={}, provenance_issues={})",
            self.euler_characteristic,
            self.connected_components,
            self.degenerate_cells,
            self.cells_with_fewer_than_three_stored_positions,
            self.boundary_edges
                + self.overused_edges
                + self.same_direction_edge_pairs
                + self.zero_length_edges
                + self.antipodal_edges,
            self.disconnected_vertex_links,
            self.provenance_issues,
        )
    }
}

/// Why a positive simplification chord threshold was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CellSimplificationThresholdError {
    /// The threshold is NaN or infinite.
    NonFinite,
    /// The threshold is zero or negative.
    NonPositive,
    /// A unit-sphere chord cannot exceed two.
    ExceedsSphereDiameter,
}

impl fmt::Display for CellSimplificationThresholdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "simplification chord threshold must be finite"),
            Self::NonPositive => write!(f, "simplification chord threshold must be positive"),
            Self::ExceedsSphereDiameter => {
                write!(f, "simplification chord threshold must be at most 2")
            }
        }
    }
}

impl std::error::Error for CellSimplificationThresholdError {}

/// What positive simplification should do when a requested contraction would
/// remove an effective generator cell.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SimplificationCellPolicy {
    /// Decline optional positive contractions which would remove a cell.
    #[default]
    Preserve,
    /// Stop at the first otherwise admissible contraction which would remove a cell.
    Error,
    /// Permit requested cell removal and return a non-Voronoi cell mesh.
    Elide,
}

/// Deterministic work limits for the cold simplification conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CellSimplificationLimits {
    diameter_pair_comparisons: u64,
    cell_index_visits: u64,
    provenance_member_checks: u64,
}

impl CellSimplificationLimits {
    /// Default maximum exact source-member distance comparisons.
    pub const DEFAULT_DIAMETER_PAIR_COMPARISONS: u64 = 100_000_000;
    /// Default maximum cumulative live cell-index visits.
    pub const DEFAULT_CELL_INDEX_VISITS: u64 = 100_000_000;
    /// Default maximum current/final suppressed-member geometry checks.
    pub const DEFAULT_PROVENANCE_MEMBER_CHECKS: u64 = 100_000_000;

    /// Construct explicit deterministic work limits.
    pub const fn new(
        diameter_pair_comparisons: u64,
        cell_index_visits: u64,
        provenance_member_checks: u64,
    ) -> Self {
        Self {
            diameter_pair_comparisons,
            cell_index_visits,
            provenance_member_checks,
        }
    }

    /// Maximum exact source-member distance comparisons.
    pub const fn diameter_pair_comparisons(self) -> u64 {
        self.diameter_pair_comparisons
    }

    /// Maximum cumulative live cell-index visits.
    pub const fn cell_index_visits(self) -> u64 {
        self.cell_index_visits
    }

    /// Maximum current/final suppressed-member geometry checks.
    pub const fn provenance_member_checks(self) -> u64 {
        self.provenance_member_checks
    }

    /// Replace the diameter-pair limit.
    pub const fn with_diameter_pair_comparisons(mut self, limit: u64) -> Self {
        self.diameter_pair_comparisons = limit;
        self
    }

    /// Replace the live cell-index visit limit.
    pub const fn with_cell_index_visits(mut self, limit: u64) -> Self {
        self.cell_index_visits = limit;
        self
    }

    /// Replace the suppressed-member geometry limit.
    pub const fn with_provenance_member_checks(mut self, limit: u64) -> Self {
        self.provenance_member_checks = limit;
        self
    }
}

impl Default for CellSimplificationLimits {
    fn default() -> Self {
        Self::new(
            Self::DEFAULT_DIAMETER_PAIR_COMPARISONS,
            Self::DEFAULT_CELL_INDEX_VISITS,
            Self::DEFAULT_PROVENANCE_MEMBER_CHECKS,
        )
    }
}

/// Validated options for explicit positive edge simplification.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct CellSimplificationOptions {
    chord_threshold: f32,
    policy: SimplificationCellPolicy,
    limits: CellSimplificationLimits,
}

impl CellSimplificationOptions {
    /// Create options from a positive unit-sphere chord threshold.
    pub fn from_chord_length(
        chord_threshold: f32,
    ) -> Result<Self, CellSimplificationThresholdError> {
        if !chord_threshold.is_finite() {
            return Err(CellSimplificationThresholdError::NonFinite);
        }
        if chord_threshold <= 0.0 {
            return Err(CellSimplificationThresholdError::NonPositive);
        }
        if chord_threshold > 2.0 {
            return Err(CellSimplificationThresholdError::ExceedsSphereDiameter);
        }
        Ok(Self {
            chord_threshold,
            policy: SimplificationCellPolicy::Preserve,
            limits: CellSimplificationLimits::default(),
        })
    }

    /// Replace the effective-cell outcome policy.
    pub const fn with_cell_policy(mut self, policy: SimplificationCellPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Replace deterministic work limits.
    pub const fn with_limits(mut self, limits: CellSimplificationLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Requested unit-sphere chord threshold.
    pub const fn chord_threshold(self) -> f32 {
        self.chord_threshold
    }

    /// Effective-cell outcome policy.
    pub const fn cell_policy(self) -> SimplificationCellPolicy {
        self.policy
    }

    /// Deterministic work limits.
    pub const fn limits(self) -> CellSimplificationLimits {
        self.limits
    }
}

/// Phase in which an explicit simplification failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CellSimplificationPhase {
    /// Source preparation has not completed.
    Preparation,
    /// Exact stored-position source preflight completed or failed.
    SourcePreflight,
    /// Mandatory exact-zero resolution.
    Exact,
    /// Optional positive-threshold contraction.
    Positive,
    /// Elision-created degree-two subdivision suppression.
    Suppression,
    /// Fixed-point provenance and topology certification.
    FinalCertification,
    /// Dense public mesh validation.
    Validation,
}

/// Stable top-level reason a positive simplification was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CellSimplificationErrorKind {
    /// The report-bearing computation was not a valid simplification source.
    InvalidSource,
    /// A source face has fewer than three stored positions without exact closure.
    UnsupportedStoredDegeneracy,
    /// A mandatory exact-zero group could not be represented safely.
    UnresolvedExactGroup,
    /// Error policy encountered a requested cell-killing contraction.
    CellEliminationRequired,
    /// The diameter pair-comparison limit was exhausted.
    DiameterPairLimitExceeded,
    /// The cumulative cell-index visit limit was exhausted.
    CellIndexLimitExceeded,
    /// The suppressed-member geometry-check limit was exhausted.
    ProvenanceMemberLimitExceeded,
    /// A checked cumulative work counter overflowed.
    CounterOverflow,
    /// A requested quotient failed its topology or geometry certificate.
    UnsafeQuotient,
    /// Suppression endpoints could not define a sufficiently conditioned arc.
    IllConditionedReplacementArc,
    /// A positive-caused suppressed member exceeded the requested arc bound.
    PositiveSuppressionDeviation,
    /// Compact public mesh storage cannot represent the result.
    RepresentationLimit,
    /// Final strict cell-mesh validation rejected the result.
    ValidationFailed,
}

/// Deterministic work consumed by simplification.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CellSimplificationWork {
    /// Exact source-member pair comparisons performed by diameter checks.
    pub diameter_pair_comparisons: u64,
    /// Cumulative live cell-index entries examined.
    pub cell_index_visits: u64,
    /// Suppressed-member current/final geometry checks performed.
    pub provenance_member_checks: u64,
    /// Largest unique candidate count retained by one phase attempt.
    pub candidate_high_water: u64,
}

/// Narrow diagnostic report returned with a failed conversion.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CellSimplificationFailureReport {
    /// Requested unit-sphere chord threshold.
    pub requested_chord_threshold: f32,
    /// Exact promoted-f64 squared threshold used for stored chords.
    pub stored_chord_threshold_squared: f64,
    /// Last phase reached by the failing operation.
    pub failure_phase: CellSimplificationPhase,
    /// Deterministic work successfully consumed before failure.
    pub work: CellSimplificationWork,
    /// Original input indices affected by the failing transaction, when known.
    pub affected_original_inputs: Vec<usize>,
}

/// Observable result of a successful positive simplification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CellSimplificationReport {
    /// Requested unit-sphere chord threshold.
    pub requested_chord_threshold: f32,
    /// Exact promoted-f64 squared threshold used for stored chords.
    pub stored_chord_threshold_squared: f64,
    /// Construction-time cells selected by the conservative hot hint.
    pub hinted_candidate_cells: usize,
    /// Positive source edges confirmed in the terminal pre-simplification mesh.
    pub confirmed_positive_edges: usize,
    /// Positive source-edge contraction proposals considered once.
    pub attempted_contractions: usize,
    /// Positive vertex contractions published by the batch.
    pub accepted_contractions: usize,
    /// Number of fixed-point attempts, including the terminal no-progress attempt.
    pub round_attempts: usize,
    /// Number of attempts which published one topology-changing transaction.
    pub productive_rounds: usize,
    /// Unique exact-zero candidate occurrences across phase attempts.
    pub exact_candidate_occurrences: u64,
    /// Unique positive candidate occurrences across phase attempts.
    pub positive_candidate_occurrences: u64,
    /// Candidate occurrences observed after the initial attempt.
    pub later_round_candidate_occurrences: u64,
    /// Fully certified transactions committed to the successful private buffer.
    pub committed_transactions: usize,
    /// Exact-zero component occurrences contained in committed transactions.
    pub exact_components_committed: usize,
    /// Positive component occurrences contained in committed transactions.
    pub positive_components_committed: usize,
    /// Positive component occurrences declined for excessive source diameter.
    pub positive_components_declined_diameter: u64,
    /// Positive interaction-group occurrences declined to preserve cells.
    pub positive_groups_declined_cell: u64,
    /// Positive interaction-group occurrences declined by topology/representation checks.
    pub positive_groups_declined_topology: u64,
    /// Positive threshold edges remaining at the final fixed point.
    pub remaining_positive_edges: usize,
    /// Exact stored-zero edges remaining at the final fixed point.
    pub remaining_exact_edges: usize,
    /// Sum of final positive-nonzero and exact-zero threshold edges.
    pub remaining_edges_at_or_below_threshold: usize,
    /// Positive edges first exposed by the single contraction batch.
    pub newly_exposed_positive_edges: usize,
    /// Effective generator cells removed by Elide.
    pub effective_cells_elided: usize,
    /// Original input indices mapped to no final cell.
    pub source_inputs_elided: usize,
    /// Live stable-id vertices retired by committed transactions.
    pub vertices_retired: usize,
    /// Source-buffer vertices absent after final compaction, including prior orphans.
    pub vertices_removed: usize,
    /// Largest retained source-member count of any considered component.
    pub max_component_members: usize,
    /// Largest stored-chord diameter of a committed positive component.
    pub max_component_diameter: f64,
    /// Largest stored-chord displacement to a selected representative.
    pub max_representative_displacement: f64,
    /// Suppressed vertices whose cause remained threshold-independent exact work.
    pub exact_suppression_members: usize,
    /// Largest acceptance-time exact suppression cross-track residual.
    pub max_exact_suppression_cross_track_radians: f64,
    /// Suppressed vertices finally carrying positive cause.
    pub positive_suppression_members: usize,
    /// Largest final positive suppressed-member deviation from its owner arc.
    pub max_positive_suppression_unit_arc_chord: f64,
    /// Deterministic work consumed by the successful conversion.
    pub work: CellSimplificationWork,
    /// Strict validation of the returned cell mesh.
    pub validation: CellMeshValidationReport,
}

/// Successful explicit positive simplification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SimplifiedCellMeshOutput {
    /// Dense simplified abstract S2 cell mesh.
    pub mesh: SphericalCellMesh,
    /// Original construction and repair report.
    pub compute_report: crate::ComputeReport,
    /// Positive simplification result telemetry.
    pub simplification_report: CellSimplificationReport,
}

/// Failure of [`crate::ComputeOutput::into_simplified_cell_mesh`].
#[derive(Debug)]
pub struct CellSimplificationError {
    kind: CellSimplificationErrorKind,
    message: String,
    report: CellSimplificationFailureReport,
    source_output: Box<crate::ComputeOutput>,
}

impl CellSimplificationError {
    fn new(
        kind: CellSimplificationErrorKind,
        message: impl Into<String>,
        report: CellSimplificationFailureReport,
        source_output: crate::ComputeOutput,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            report,
            source_output: Box::new(source_output),
        }
    }

    /// Stable top-level rejection category.
    pub fn kind(&self) -> CellSimplificationErrorKind {
        self.kind
    }

    /// Unstable diagnostic detail.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Borrow failure phase, work, and affected-input diagnostics.
    pub fn report(&self) -> &CellSimplificationFailureReport {
        &self.report
    }

    /// Borrow the original successful computation.
    pub fn source_output(&self) -> &crate::ComputeOutput {
        &self.source_output
    }

    /// Recover the original successful computation without cloning.
    pub fn into_source_output(self) -> crate::ComputeOutput {
        *self.source_output
    }
}

impl fmt::Display for CellSimplificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cell simplification {:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for CellSimplificationError {}

/// Stable top-level reason an explicit cell-elision conversion was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CellElisionErrorKind {
    /// The report-bearing source did not satisfy its strict output contract.
    InvalidSource,
    /// The requested quotient could not produce a valid spherical cell mesh.
    UnsafeQuotient,
    /// A compact index or storage representation limit was exceeded.
    RepresentationLimit,
}

/// Failure of [`crate::ComputeOutput::into_elided_cell_mesh`].
///
/// The successful Preserve result is retained inside the error, so a failed
/// optional simplification never destroys or silently substitutes the source
/// diagram. Use [`Self::into_source_output`] to recover it without cloning.
#[derive(Debug)]
pub struct CellElisionError {
    kind: CellElisionErrorKind,
    message: String,
    source_output: Box<crate::ComputeOutput>,
}

impl CellElisionError {
    fn new(
        kind: CellElisionErrorKind,
        message: impl Into<String>,
        source_output: crate::ComputeOutput,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            source_output: Box::new(source_output),
        }
    }

    /// Stable top-level rejection category.
    #[inline]
    pub fn kind(&self) -> CellElisionErrorKind {
        self.kind
    }

    /// Diagnostic detail. Wording is not a stable API contract.
    #[inline]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Borrow the original successful computation.
    #[inline]
    pub fn source_output(&self) -> &crate::ComputeOutput {
        &self.source_output
    }

    /// Recover the original successful computation without cloning.
    #[inline]
    pub fn into_source_output(self) -> crate::ComputeOutput {
        *self.source_output
    }
}

impl fmt::Display for CellElisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cell elision {:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for CellElisionError {}

#[derive(Debug, Clone, Copy)]
pub(crate) enum CellMeshPreparationErrorKind {
    InvalidSource,
    RepresentationLimit,
}

#[derive(Debug)]
pub(crate) struct CellMeshPreparationError {
    pub kind: CellMeshPreparationErrorKind,
    pub message: String,
}

#[derive(Debug)]
pub(crate) struct PreparedCellMeshSource {
    pub generators: Vec<glam::Vec3>,
    pub vertices: Vec<glam::Vec3>,
    pub cells: Vec<crate::diagram::VoronoiCell>,
    pub cell_indices: Vec<u32>,
    pub effective_source_sites: Vec<SpherePoint>,
    pub effective_to_input: Vec<u32>,
    pub input_to_effective: Vec<u32>,
}

pub(crate) fn prepare_cell_mesh_source(
    source: &crate::ComputeOutput,
) -> Result<PreparedCellMeshSource, CellMeshPreparationError> {
    let invalid = |message: &str| CellMeshPreparationError {
        kind: CellMeshPreparationErrorKind::InvalidSource,
        message: message.into(),
    };
    let representation = |message: &str| CellMeshPreparationError {
        kind: CellMeshPreparationErrorKind::RepresentationLimit,
        message: message.into(),
    };

    if source.report.has_output_residuals() {
        return Err(invalid(
            "source computation has output or strict-validation residuals",
        ));
    }

    let preferred = source.preferred_diagram();
    let generators: Vec<glam::Vec3> = preferred
        .generators()
        .iter()
        .map(|site| glam::Vec3::from_array(site.to_array()))
        .collect();
    let vertices: Vec<glam::Vec3> = preferred
        .vertices()
        .iter()
        .map(|vertex| glam::Vec3::from_array(vertex.to_array()))
        .collect();
    let effective_source_sites = preferred.generators().to_vec();
    let mut cells = Vec::with_capacity(preferred.num_cells());
    let mut cell_indices = Vec::new();
    for cell in preferred.iter_cells() {
        if cell.vertex_indices.len() > u16::MAX as usize || cell_indices.len() > u32::MAX as usize {
            return Err(representation(
                "source cell layout exceeds compact mesh index capacity",
            ));
        }
        cells.push(crate::diagram::VoronoiCell::new(
            cell_indices.len() as u32,
            cell.vertex_indices.len() as u16,
        ));
        cell_indices.extend_from_slice(cell.vertex_indices);
    }

    let original_cells = source.diagram.num_cells();
    if original_cells > u32::MAX as usize || preferred.num_cells() > u32::MAX as usize {
        return Err(representation("source input mapping exceeds u32 capacity"));
    }
    let mut effective_to_input = Vec::with_capacity(preferred.num_cells());
    let mut canonical_to_effective = vec![u32::MAX; original_cells];
    for (input, slot) in canonical_to_effective.iter_mut().enumerate() {
        if source.diagram.canonical_cell_index(input) == input {
            let effective = effective_to_input.len() as u32;
            *slot = effective;
            effective_to_input.push(input as u32);
        }
    }
    if effective_to_input.len() != preferred.num_cells() {
        return Err(invalid(
            "source weld mapping does not match the effective diagram",
        ));
    }
    let mut input_to_effective = Vec::with_capacity(original_cells);
    for input in 0..original_cells {
        let canonical = source.diagram.canonical_cell_index(input);
        let effective = canonical_to_effective[canonical];
        if effective == u32::MAX {
            return Err(invalid(
                "source weld mapping names a noncanonical effective cell",
            ));
        }
        input_to_effective.push(effective);
    }

    Ok(PreparedCellMeshSource {
        generators,
        vertices,
        cells,
        cell_indices,
        effective_source_sites,
        effective_to_input,
        input_to_effective,
    })
}

pub(crate) struct FinalizedCellMesh {
    pub mesh: SphericalCellMesh,
    pub source_inputs_elided: usize,
    pub validation: CellMeshValidationReport,
}

pub(crate) fn finalize_cell_mesh_source(
    prepared: &PreparedCellMeshSource,
    final_vertices: Vec<SpherePoint>,
    final_cycles: Vec<Vec<u32>>,
    effective_to_cell: &[Option<u32>],
    cell_to_effective: &[u32],
) -> Result<FinalizedCellMesh, String> {
    let input_to_cell: Vec<Option<u32>> = prepared
        .input_to_effective
        .iter()
        .map(|&effective| effective_to_cell[effective as usize])
        .collect();
    let source_inputs_elided = input_to_cell.iter().filter(|cell| cell.is_none()).count();
    let cell_to_input: Vec<u32> = cell_to_effective
        .iter()
        .map(|&effective| prepared.effective_to_input[effective as usize])
        .collect();
    let cell_source_sites: Vec<SpherePoint> = cell_to_effective
        .iter()
        .map(|&effective| prepared.effective_source_sites[effective as usize])
        .collect();
    let mesh = SphericalCellMesh::from_raw_parts(
        final_vertices,
        final_cycles,
        cell_source_sites,
        cell_to_input,
        input_to_cell,
    );
    let validation = mesh.validate();
    if !validation.is_strictly_valid() {
        return Err(validation.headline());
    }
    Ok(FinalizedCellMesh {
        mesh,
        source_inputs_elided,
        validation,
    })
}

pub(crate) fn finish_construction_simplification(
    source: crate::ComputeOutput,
    options: CellSimplificationOptions,
    internal: crate::knn_clipping::output_resolution::PositiveResolutionReport,
) -> Result<SimplifiedCellMeshOutput, crate::VoronoiError> {
    let prepared = prepare_cell_mesh_source(&source).map_err(|error| match error.kind {
        CellMeshPreparationErrorKind::InvalidSource => {
            crate::VoronoiError::ComputationFailed(error.message)
        }
        CellMeshPreparationErrorKind::RepresentationLimit => {
            crate::VoronoiError::RepresentationLimit(error.message)
        }
    })?;

    let mut used = vec![false; prepared.vertices.len()];
    for &vertex in &prepared.cell_indices {
        let Some(slot) = used.get_mut(vertex as usize) else {
            return Err(crate::VoronoiError::ComputationFailed(
                "simplified cell references an out-of-range vertex".into(),
            ));
        };
        *slot = true;
    }
    let mut dense_for = vec![u32::MAX; prepared.vertices.len()];
    let mut dense_vertices = Vec::with_capacity(used.iter().filter(|&&live| live).count());
    for (old, (&live, &position)) in used.iter().zip(&prepared.vertices).enumerate() {
        if live {
            dense_for[old] = dense_vertices.len() as u32;
            dense_vertices.push(position);
        }
    }
    let mut cycles = Vec::with_capacity(prepared.cells.len());
    for cell in &prepared.cells {
        let span =
            &prepared.cell_indices[cell.vertex_start()..cell.vertex_start() + cell.vertex_count()];
        cycles.push(
            span.iter()
                .map(|&vertex| dense_for[vertex as usize])
                .collect(),
        );
    }
    let effective_to_cell: Vec<Option<u32>> = (0..prepared.cells.len())
        .map(|cell| Some(cell as u32))
        .collect();
    let cell_to_effective: Vec<u32> = (0..prepared.cells.len() as u32).collect();
    let vertices_removed = prepared.vertices.len() - dense_vertices.len();
    // SAFETY: the construction pipeline only retains its checked unit-sphere
    // vertex records; this compaction neither edits nor creates positions.
    let dense_vertices = unsafe { crate::types::sphere_points_from_vec3(dense_vertices) };
    let finalized = finalize_cell_mesh_source(
        &prepared,
        dense_vertices,
        cycles,
        &effective_to_cell,
        &cell_to_effective,
    )
    .map_err(|message| {
        crate::VoronoiError::ComputationFailed(format!(
            "construction-aware simplification failed strict validation: {message}"
        ))
    })?;

    let threshold = f64::from(options.chord_threshold);
    let productive = usize::from(internal.accepted_contractions != 0);
    Ok(SimplifiedCellMeshOutput {
        mesh: finalized.mesh,
        compute_report: source.report,
        simplification_report: CellSimplificationReport {
            requested_chord_threshold: options.chord_threshold,
            stored_chord_threshold_squared: threshold * threshold,
            hinted_candidate_cells: internal.hinted_cells,
            confirmed_positive_edges: internal.confirmed_candidates,
            attempted_contractions: internal.attempted_contractions,
            accepted_contractions: internal.accepted_contractions,
            round_attempts: 1,
            productive_rounds: productive,
            exact_candidate_occurrences: 0,
            positive_candidate_occurrences: internal.confirmed_candidates as u64,
            later_round_candidate_occurrences: 0,
            committed_transactions: productive,
            exact_components_committed: 0,
            positive_components_committed: internal.accepted_contractions,
            positive_components_declined_diameter: internal.displacement_declines as u64,
            positive_groups_declined_cell: internal.cell_declined_components as u64,
            positive_groups_declined_topology: internal.topology_declined_components as u64,
            remaining_positive_edges: internal.remaining_positive_edges,
            remaining_exact_edges: 0,
            remaining_edges_at_or_below_threshold: internal.remaining_positive_edges,
            newly_exposed_positive_edges: internal.newly_exposed_positive_edges,
            effective_cells_elided: 0,
            source_inputs_elided: 0,
            vertices_retired: internal.vertices_retired,
            vertices_removed,
            max_component_members: internal.max_component_members,
            max_component_diameter: 0.0,
            max_representative_displacement: internal.max_representative_displacement_bound,
            exact_suppression_members: 0,
            max_exact_suppression_cross_track_radians: 0.0,
            positive_suppression_members: 0,
            max_positive_suppression_unit_arc_chord: 0.0,
            work: CellSimplificationWork::default(),
            validation: finalized.validation,
        },
    })
}

/// Observable result of exact stored-zero cell elision.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CellElisionReport {
    /// Remaining exact stored-zero edges considered by this postprocess.
    pub exact_zero_edges_detected: usize,
    /// Connected exact-zero vertex components considered by this postprocess.
    pub exact_zero_components_detected: usize,
    /// Effective generator cells removed by the quotient.
    pub effective_cells_elided: usize,
    /// Original input indices mapped to no final cell. This can exceed
    /// `effective_cells_elided` when preprocessing welded an elided class.
    pub source_inputs_elided: usize,
    /// Degree-two boundary subdivision vertices suppressed after face removal.
    pub degree_two_vertices_suppressed: usize,
    /// Stored source vertices absent from the dense final mesh.
    pub vertices_removed: usize,
    /// Maximum cross-track residual, in radians, of a suppressed vertex
    /// against its replacement great circle.
    ///
    /// This is transaction telemetry, not a global Hausdorff or Voronoi error
    /// bound.
    pub max_suppression_cross_track_radians: f64,
    /// Generic cell-mesh validation of the returned result.
    pub validation: CellMeshValidationReport,
}

/// Result of explicit exact stored-zero cell elision.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CellMeshOutput {
    /// Dense valid spherical cell subdivision.
    pub mesh: SphericalCellMesh,
    /// Original construction, reconciliation, local-rebuild, and
    /// output-resolution report.
    pub compute_report: crate::ComputeReport,
    /// Outcome of the explicit postprocessing transaction.
    pub elision_report: CellElisionReport,
}

impl crate::ComputeOutput {
    /// Consume a report-bearing Voronoi computation and explicitly remove
    /// effective cells whose geometry cannot survive exact stored-zero
    /// contraction.
    ///
    /// This is a requested cold `O(V + E + F)` postprocess. It never runs from
    /// [`crate::compute`] or [`crate::compute_with_report`] implicitly. The
    /// returned [`SphericalCellMesh`] is a valid spherical subdivision with
    /// source provenance, but is not promised to remain a Voronoi diagram of
    /// its surviving source sites.
    ///
    /// A rejected quotient returns [`CellElisionError`] containing this
    /// original successful computation. There is no partial output and no
    /// implicit Preserve fallback.
    pub fn into_elided_cell_mesh(self) -> Result<CellMeshOutput, CellElisionError> {
        match prepare_elided_cell_mesh(&self) {
            Ok((mesh, elision_report)) => {
                let crate::ComputeOutput { report, .. } = self;
                Ok(CellMeshOutput {
                    mesh,
                    compute_report: report,
                    elision_report,
                })
            }
            Err((kind, message)) => Err(CellElisionError::new(kind, message, self)),
        }
    }

    /// Consume a report-bearing computation and explicitly simplify every
    /// admissible live edge within a positive chord threshold.
    ///
    /// This is a cold postprocess. It does not change ordinary construction or
    /// exact-zero output resolution. The returned mesh is a valid abstract S2
    /// cell complex, not necessarily a Voronoi diagram of the source sites.
    pub fn into_simplified_cell_mesh(
        self,
        options: CellSimplificationOptions,
    ) -> Result<SimplifiedCellMeshOutput, CellSimplificationError> {
        match prepare_simplified_cell_mesh(&self, options) {
            Ok((mesh, simplification_report)) => {
                let crate::ComputeOutput { report, .. } = self;
                Ok(SimplifiedCellMeshOutput {
                    mesh,
                    compute_report: report,
                    simplification_report,
                })
            }
            Err((kind, message, report)) => {
                Err(CellSimplificationError::new(kind, message, report, self))
            }
        }
    }
}

fn public_simplification_phase(
    phase: crate::knn_clipping::positive_simplification::Phase,
) -> CellSimplificationPhase {
    use crate::knn_clipping::positive_simplification::Phase;
    match phase {
        Phase::Preparation => CellSimplificationPhase::Preparation,
        Phase::SourcePreflight => CellSimplificationPhase::SourcePreflight,
        Phase::Exact => CellSimplificationPhase::Exact,
        Phase::Positive => CellSimplificationPhase::Positive,
        Phase::Suppression => CellSimplificationPhase::Suppression,
        Phase::FinalCertification => CellSimplificationPhase::FinalCertification,
    }
}

fn public_simplification_work(
    work: crate::knn_clipping::positive_simplification::WorkStats,
) -> CellSimplificationWork {
    CellSimplificationWork {
        diameter_pair_comparisons: work.diameter_pair_comparisons,
        cell_index_visits: work.cell_index_visits,
        provenance_member_checks: work.provenance_member_checks,
        candidate_high_water: work.candidate_high_water,
    }
}

fn public_simplification_error_kind(
    kind: crate::knn_clipping::positive_simplification::FailureKind,
) -> CellSimplificationErrorKind {
    use crate::knn_clipping::positive_simplification::FailureKind;
    match kind {
        FailureKind::UnsupportedStoredDegeneracy => {
            CellSimplificationErrorKind::UnsupportedStoredDegeneracy
        }
        FailureKind::UnresolvedExactGroup => CellSimplificationErrorKind::UnresolvedExactGroup,
        FailureKind::CellEliminationRequired => {
            CellSimplificationErrorKind::CellEliminationRequired
        }
        FailureKind::DiameterLimit => CellSimplificationErrorKind::DiameterPairLimitExceeded,
        FailureKind::CellIndexLimit => CellSimplificationErrorKind::CellIndexLimitExceeded,
        FailureKind::ProvenanceLimit => CellSimplificationErrorKind::ProvenanceMemberLimitExceeded,
        FailureKind::CounterOverflow => CellSimplificationErrorKind::CounterOverflow,
        FailureKind::Validation => CellSimplificationErrorKind::ValidationFailed,
        FailureKind::UnsafeQuotient => CellSimplificationErrorKind::UnsafeQuotient,
        FailureKind::IllConditionedReplacementArc => {
            CellSimplificationErrorKind::IllConditionedReplacementArc
        }
        FailureKind::PositiveSuppressionDeviation => {
            CellSimplificationErrorKind::PositiveSuppressionDeviation
        }
    }
}

fn prepare_simplified_cell_mesh(
    source: &crate::ComputeOutput,
    options: CellSimplificationOptions,
) -> Result<
    (SphericalCellMesh, CellSimplificationReport),
    (
        CellSimplificationErrorKind,
        String,
        CellSimplificationFailureReport,
    ),
> {
    let threshold = options.chord_threshold as f64;
    let threshold_squared = threshold * threshold;
    let basic_failure = |failure_phase| CellSimplificationFailureReport {
        requested_chord_threshold: options.chord_threshold,
        stored_chord_threshold_squared: threshold_squared,
        failure_phase,
        work: CellSimplificationWork::default(),
        affected_original_inputs: Vec::new(),
    };
    let prepared = prepare_cell_mesh_source(source).map_err(|error| {
        let (kind, phase) = match error.kind {
            CellMeshPreparationErrorKind::InvalidSource => (
                CellSimplificationErrorKind::InvalidSource,
                CellSimplificationPhase::Preparation,
            ),
            CellMeshPreparationErrorKind::RepresentationLimit => (
                CellSimplificationErrorKind::RepresentationLimit,
                CellSimplificationPhase::Preparation,
            ),
        };
        (kind, error.message, basic_failure(phase))
    })?;
    let policy = match options.policy {
        SimplificationCellPolicy::Preserve => {
            crate::knn_clipping::positive_simplification::CellPolicy::Preserve
        }
        SimplificationCellPolicy::Error => {
            crate::knn_clipping::positive_simplification::CellPolicy::Error
        }
        SimplificationCellPolicy::Elide => {
            crate::knn_clipping::positive_simplification::CellPolicy::Elide
        }
    };
    let limits = crate::knn_clipping::positive_simplification::Limits {
        diameter_pair_comparisons: options.limits.diameter_pair_comparisons,
        cell_index_visits: options.limits.cell_index_visits,
        provenance_member_checks: options.limits.provenance_member_checks,
    };
    let outcome = crate::knn_clipping::positive_simplification::simplify(
        &prepared.vertices,
        &prepared.cells,
        &prepared.cell_indices,
        threshold,
        policy,
        limits,
    )
    .map_err(|failure| {
        let affected: FxHashSet<usize> = failure.affected_effective_cells.iter().copied().collect();
        let affected_original_inputs = prepared
            .input_to_effective
            .iter()
            .enumerate()
            .filter_map(|(input, &effective)| {
                affected.contains(&(effective as usize)).then_some(input)
            })
            .collect();
        (
            public_simplification_error_kind(failure.kind),
            failure.message,
            CellSimplificationFailureReport {
                requested_chord_threshold: options.chord_threshold,
                stored_chord_threshold_squared: threshold_squared,
                failure_phase: public_simplification_phase(failure.phase),
                work: public_simplification_work(failure.work),
                affected_original_inputs,
            },
        )
    })?;
    let vertices_removed = prepared.vertices.len() - outcome.vertices.len();
    // SAFETY: simplification only retains validated source vertex records; it
    // neither edits their coordinates nor creates unchecked directions.
    let final_vertices = unsafe { crate::types::sphere_points_from_vec3(outcome.vertices) };
    let finalized = finalize_cell_mesh_source(
        &prepared,
        final_vertices,
        outcome.cycles,
        &outcome.effective_to_cell,
        &outcome.cell_to_effective,
    )
    .map_err(|message| {
        (
            CellSimplificationErrorKind::ValidationFailed,
            message,
            CellSimplificationFailureReport {
                requested_chord_threshold: options.chord_threshold,
                stored_chord_threshold_squared: threshold_squared,
                failure_phase: CellSimplificationPhase::Validation,
                work: public_simplification_work(outcome.work),
                affected_original_inputs: Vec::new(),
            },
        )
    })?;
    let stats = outcome.stats;
    Ok((
        finalized.mesh,
        CellSimplificationReport {
            requested_chord_threshold: options.chord_threshold,
            stored_chord_threshold_squared: threshold_squared,
            hinted_candidate_cells: 0,
            confirmed_positive_edges: stats.positive_candidate_occurrences as usize,
            attempted_contractions: stats.positive_candidate_occurrences as usize,
            accepted_contractions: stats.positive_components_committed,
            round_attempts: stats.round_attempts,
            productive_rounds: stats.productive_rounds,
            exact_candidate_occurrences: stats.exact_candidate_occurrences,
            positive_candidate_occurrences: stats.positive_candidate_occurrences,
            later_round_candidate_occurrences: stats.later_round_candidate_occurrences,
            committed_transactions: stats.committed_transactions,
            exact_components_committed: stats.exact_components_committed,
            positive_components_committed: stats.positive_components_committed,
            positive_components_declined_diameter: stats.positive_components_declined_diameter,
            positive_groups_declined_cell: stats.positive_groups_declined_cell,
            positive_groups_declined_topology: stats.positive_groups_declined_topology,
            remaining_positive_edges: stats.final_positive_edges,
            remaining_exact_edges: stats.final_exact_edges,
            remaining_edges_at_or_below_threshold: stats.final_positive_edges
                + stats.final_exact_edges,
            newly_exposed_positive_edges: 0,
            effective_cells_elided: stats.effective_cells_elided,
            source_inputs_elided: finalized.source_inputs_elided,
            vertices_retired: stats.vertices_retired,
            vertices_removed,
            max_component_members: stats.max_component_members,
            max_component_diameter: stats.max_component_diameter,
            max_representative_displacement: stats.max_representative_displacement,
            exact_suppression_members: stats.exact_suppression_members,
            max_exact_suppression_cross_track_radians: stats
                .max_exact_suppression_cross_track_radians,
            positive_suppression_members: stats.positive_suppression_members,
            max_positive_suppression_unit_arc_chord: stats.max_positive_suppression_unit_arc_chord,
            work: public_simplification_work(outcome.work),
            validation: finalized.validation,
        },
    ))
}

fn prepare_elided_cell_mesh(
    source: &crate::ComputeOutput,
) -> Result<(SphericalCellMesh, CellElisionReport), (CellElisionErrorKind, String)> {
    let prepared = prepare_cell_mesh_source(source).map_err(|error| {
        let kind = match error.kind {
            CellMeshPreparationErrorKind::InvalidSource => CellElisionErrorKind::InvalidSource,
            CellMeshPreparationErrorKind::RepresentationLimit => {
                CellElisionErrorKind::RepresentationLimit
            }
        };
        (kind, error.message)
    })?;

    let elision = crate::knn_clipping::output_resolution::elide_exact_zero_cells_for_mesh(
        &prepared.generators,
        &prepared.vertices,
        &prepared.cells,
        &prepared.cell_indices,
    )
    .map_err(|error| (CellElisionErrorKind::UnsafeQuotient, error.to_string()))?;
    let final_vertices = elision.diagram.vertices().to_vec();
    let final_cycles: Vec<Vec<u32>> = elision
        .diagram
        .iter_cells()
        .map(|cell| cell.vertex_indices.to_vec())
        .collect();
    let vertices_removed = prepared.vertices.len() - final_vertices.len();
    let finalized = finalize_cell_mesh_source(
        &prepared,
        final_vertices,
        final_cycles,
        &elision.effective_to_cell,
        &elision.cell_to_effective,
    )
    .map_err(|message| (CellElisionErrorKind::UnsafeQuotient, message))?;

    Ok((
        finalized.mesh,
        CellElisionReport {
            exact_zero_edges_detected: elision.zero_edges_before,
            exact_zero_components_detected: elision.zero_components_before,
            effective_cells_elided: elision.effective_cells_elided,
            source_inputs_elided: finalized.source_inputs_elided,
            degree_two_vertices_suppressed: elision.degree_two_vertices_suppressed,
            vertices_removed,
            max_suppression_cross_track_radians: elision.max_suppression_cross_track_radians,
            validation: finalized.validation,
        },
    ))
}

#[derive(Debug, Clone, Copy)]
struct EdgeUse {
    cell: u32,
    forward: bool,
}

fn validate_cell_mesh(mesh: &SphericalCellMesh) -> CellMeshValidationReport {
    let mut vertices_off_sphere = 0usize;
    for vertex in &mesh.vertices {
        let len_sq = vertex.length_squared();
        if !len_sq.is_finite() || (len_sq - 1.0).abs() > crate::tolerances::VERTEX_ON_SPHERE_EPS {
            vertices_off_sphere += 1;
        }
    }

    let mut used = vec![false; mesh.num_vertices()];
    let mut incidence = vec![0usize; mesh.num_vertices()];
    let mut links = vec![Vec::<(u32, u32)>::new(); mesh.num_vertices()];
    let mut edge_uses = FxHashMap::<(u32, u32), Vec<EdgeUse>>::default();
    let mut cell_signatures = FxHashSet::<Vec<u32>>::default();
    let mut degenerate_cells = 0;
    let mut cells_with_duplicate_vertices = 0;
    let mut cells_with_invalid_references = 0;
    let mut duplicate_cells = 0;
    let mut cells_with_fewer_than_three_stored_positions = 0;

    for cell in mesh.iter_cells() {
        let cycle = cell.vertex_indices;
        if cycle.len() < 3 {
            degenerate_cells += 1;
            cells_with_fewer_than_three_stored_positions += 1;
            continue;
        }
        let mut unique = FxHashSet::default();
        let mut invalid = false;
        for &vertex in cycle {
            if vertex as usize >= mesh.num_vertices() {
                invalid = true;
            }
            unique.insert(vertex);
        }
        if unique.len() != cycle.len() {
            cells_with_duplicate_vertices += 1;
        }
        if invalid {
            cells_with_invalid_references += 1;
            continue;
        }
        let mut distinct_positions = [None; 3];
        let mut distinct_position_count = 0usize;
        for &vertex in cycle {
            let position = mesh.vertices[vertex as usize];
            if distinct_position_count < 3
                && !distinct_positions[..distinct_position_count].contains(&Some(position))
            {
                distinct_positions[distinct_position_count] = Some(position);
                distinct_position_count += 1;
            }
        }
        if distinct_position_count < 3 {
            cells_with_fewer_than_three_stored_positions += 1;
        }
        let mut signature = cycle.to_vec();
        signature.sort_unstable();
        if !cell_signatures.insert(signature) {
            duplicate_cells += 1;
        }

        for i in 0..cycle.len() {
            let vertex = cycle[i] as usize;
            let prev = cycle[(i + cycle.len() - 1) % cycle.len()];
            let next = cycle[(i + 1) % cycle.len()];
            used[vertex] = true;
            incidence[vertex] += 1;
            links[vertex].push((prev, next));

            let a = cycle[i];
            let b = cycle[(i + 1) % cycle.len()];
            let (lo, hi, forward) = if a < b { (a, b, true) } else { (b, a, false) };
            edge_uses.entry((lo, hi)).or_default().push(EdgeUse {
                cell: cell.cell_index as u32,
                forward,
            });
        }
    }

    let orphan_vertices = used.iter().filter(|&&is_used| !is_used).count();
    let low_incidence_vertices = incidence
        .iter()
        .filter(|&&degree| degree > 0 && degree < 3)
        .count();
    let mut disconnected_vertex_links = 0;
    for edges in links.iter().filter(|edges| !edges.is_empty()) {
        let mut next_for = FxHashMap::<u32, u32>::default();
        let mut incoming = FxHashSet::<u32>::default();
        let mut valid = true;
        for &(from, to) in edges {
            if next_for.insert(from, to).is_some() || !incoming.insert(to) {
                valid = false;
                break;
            }
        }
        if valid
            && (next_for.len() != incoming.len()
                || next_for.keys().any(|vertex| !incoming.contains(vertex)))
        {
            valid = false;
        }
        if valid {
            let start = edges[0].0;
            let mut current = start;
            let mut visited = FxHashSet::default();
            loop {
                if !visited.insert(current) {
                    valid = current == start;
                    break;
                }
                let Some(&next) = next_for.get(&current) else {
                    valid = false;
                    break;
                };
                current = next;
            }
            valid &= visited.len() == next_for.len();
        }
        if !valid {
            disconnected_vertex_links += 1;
        }
    }

    let mut boundary_edges = 0;
    let mut overused_edges = 0;
    let mut same_direction_edge_pairs = 0;
    let mut zero_length_edges = 0;
    let mut antipodal_edges = 0;
    let mut cell_neighbors = vec![Vec::<usize>::new(); mesh.num_cells()];
    for (&(a, b), uses) in &edge_uses {
        match uses.len() {
            1 => boundary_edges += 1,
            2 => {
                if uses[0].forward == uses[1].forward {
                    same_direction_edge_pairs += 1;
                }
                let ca = uses[0].cell as usize;
                let cb = uses[1].cell as usize;
                if ca != cb {
                    cell_neighbors[ca].push(cb);
                    cell_neighbors[cb].push(ca);
                }
            }
            _ => overused_edges += 1,
        }
        if a == b || mesh.vertices[a as usize] == mesh.vertices[b as usize] {
            zero_length_edges += 1;
        } else {
            let va = mesh.vertices[a as usize];
            let vb = mesh.vertices[b as usize];
            if va.x() == -vb.x() && va.y() == -vb.y() && va.z() == -vb.z() {
                antipodal_edges += 1;
            }
        }
    }

    let mut connected_components = 0;
    let mut seen_cells = vec![false; mesh.num_cells()];
    for start in 0..mesh.num_cells() {
        if seen_cells[start] {
            continue;
        }
        connected_components += 1;
        seen_cells[start] = true;
        let mut stack = vec![start];
        while let Some(cell) = stack.pop() {
            for &neighbor in &cell_neighbors[cell] {
                if !seen_cells[neighbor] {
                    seen_cells[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
    }

    let mut provenance_issues = 0;
    if mesh.cell_source_sites.len() != mesh.num_cells()
        || mesh.cell_to_input.len() != mesh.num_cells()
    {
        provenance_issues += 1;
    } else {
        provenance_issues += mesh
            .cell_source_sites
            .iter()
            .filter(|site| {
                let len_sq = site.length_squared();
                !len_sq.is_finite()
                    || (len_sq - 1.0).abs() > crate::tolerances::VERTEX_ON_SPHERE_EPS
            })
            .count();
        let mut cells_with_inputs = vec![false; mesh.num_cells()];
        for (input, &cell) in mesh.input_to_cell.iter().enumerate() {
            if cell == NO_CELL {
                continue;
            }
            let Some(slot) = cells_with_inputs.get_mut(cell as usize) else {
                provenance_issues += 1;
                continue;
            };
            *slot = true;
            if mesh.cell_to_input[cell as usize] as usize > input {
                provenance_issues += 1;
            }
        }
        for (cell, &source_input) in mesh.cell_to_input.iter().enumerate() {
            if source_input as usize >= mesh.input_to_cell.len()
                || mesh.input_to_cell[source_input as usize] != cell as u32
                || !cells_with_inputs[cell]
            {
                provenance_issues += 1;
            }
        }
    }

    CellMeshValidationReport {
        num_cells: mesh.num_cells(),
        num_vertices: mesh.num_vertices(),
        num_edges: edge_uses.len(),
        euler_characteristic: mesh.num_vertices() as i32 - edge_uses.len() as i32
            + mesh.num_cells() as i32,
        connected_components,
        degenerate_cells,
        cells_with_fewer_than_three_stored_positions,
        cells_with_duplicate_vertices,
        cells_with_invalid_references,
        duplicate_cells,
        vertices_off_sphere,
        orphan_vertices,
        low_incidence_vertices,
        disconnected_vertex_links,
        boundary_edges,
        overused_edges,
        same_direction_edge_pairs,
        zero_length_edges,
        antipodal_edges,
        provenance_issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn raw(x: f32, y: f32, z: f32) -> [f32; 3] {
        [x, y, z]
    }

    fn unit(x: f32, y: f32, z: f32) -> SpherePoint {
        SpherePoint::try_from_xyz([x, y, z]).unwrap()
    }

    fn fixture_mesh(vertices: Vec<SpherePoint>, cycles: &[&[u32]]) -> SphericalCellMesh {
        SphericalCellMesh::from_raw_parts(
            vertices,
            cycles.iter().map(|cycle| cycle.to_vec()).collect(),
            vec![unit(0.0, 0.0, 1.0); cycles.len()],
            (0..cycles.len() as u32).collect(),
            (0..cycles.len() as u32).map(Some).collect(),
        )
    }

    #[test]
    fn generic_validator_rejects_a_pinched_vertex_link() {
        let tetra_vertices = vec![
            unit(1.0, 1.0, 1.0),
            unit(1.0, -1.0, -1.0),
            unit(-1.0, 1.0, -1.0),
            unit(-1.0, -1.0, 1.0),
        ];
        let tetra_cycles: &[&[u32]] = &[&[0, 2, 1], &[0, 1, 3], &[0, 3, 2], &[1, 2, 3]];
        let tetra = fixture_mesh(tetra_vertices, tetra_cycles);
        assert!(tetra.validate().is_strictly_valid());

        let pinched = fixture_mesh(
            vec![
                unit(1.0, 1.0, 1.0),
                unit(1.0, -1.0, -1.0),
                unit(-1.0, 1.0, -1.0),
                unit(-1.0, -1.0, 1.0),
                unit(1.0, -1.0, 1.0),
                unit(-1.0, 1.0, 1.0),
                unit(-1.0, -1.0, -1.0),
            ],
            &[
                &[0, 2, 1],
                &[0, 1, 3],
                &[0, 3, 2],
                &[1, 2, 3],
                &[0, 5, 4],
                &[0, 4, 6],
                &[0, 6, 5],
                &[4, 5, 6],
            ],
        );
        let report = pinched.validate();
        assert!(!report.is_strictly_valid());
        assert_eq!(report.disconnected_vertex_links, 1);
    }

    #[test]
    fn rejected_conversion_returns_the_original_successful_output() {
        let points = [
            raw(1.0, 0.0, 0.0),
            raw(-1.0, 0.0, 0.0),
            raw(0.0, 1.0, 0.0),
            raw(0.0, -1.0, 0.0),
            raw(0.0, 0.0, 1.0),
            raw(0.0, 0.0, -1.0),
        ];
        let mut output = crate::compute_with_report(&points, crate::VoronoiConfig::default())
            .expect("octahedral source should compute");
        output.report.residual_unpaired_edges.push((0, 1));
        let cell_count = output.diagram.num_cells();

        let error = output
            .into_elided_cell_mesh()
            .expect_err("a source with explicit residuals must be rejected");
        assert_eq!(error.kind(), CellElisionErrorKind::InvalidSource);
        assert_eq!(error.source_output().diagram.num_cells(), cell_count);
        let recovered = error.into_source_output();
        assert_eq!(recovered.diagram.num_cells(), cell_count);
        assert_eq!(recovered.report.residual_unpaired_edges, [(0, 1)]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip_revalidates_dense_mesh() {
        let points = [
            raw(1.0, 0.0, 0.0),
            raw(-1.0, 0.0, 0.0),
            raw(0.0, 1.0, 0.0),
            raw(0.0, -1.0, 0.0),
            raw(0.0, 0.0, 1.0),
            raw(0.0, 0.0, -1.0),
        ];
        let mesh = crate::compute_with_report(&points, crate::VoronoiConfig::default())
            .unwrap()
            .into_elided_cell_mesh()
            .unwrap()
            .mesh;
        let encoded = serde_json::to_string(&mesh).unwrap();
        let decoded: SphericalCellMesh = serde_json::from_str(&encoded).unwrap();
        assert!(decoded.validate().is_strictly_valid());
        assert_eq!(decoded.num_cells(), mesh.num_cells());
        assert_eq!(decoded.num_vertices(), mesh.num_vertices());
        for input in 0..mesh.num_source_inputs() {
            assert_eq!(decoded.cell_for_input(input), mesh.cell_for_input(input));
        }
    }
}
