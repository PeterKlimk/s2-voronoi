//! Spherical cell meshes produced by explicit output-resolution simplification.

use crate::{CellAdjacency, UnitVec3};
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
    vertices: Vec<UnitVec3>,
    cells: Vec<MeshCellData>,
    cell_indices: Vec<u32>,
    cell_source_sites: Vec<UnitVec3>,
    cell_to_input: Vec<u32>,
    input_to_cell: Vec<u32>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct CellMeshWire {
    vertices: Vec<UnitVec3>,
    cells: Vec<MeshCellData>,
    cell_indices: Vec<u32>,
    cell_source_sites: Vec<UnitVec3>,
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
        vertices: Vec<UnitVec3>,
        cell_cycles: Vec<Vec<u32>>,
        cell_source_sites: Vec<UnitVec3>,
        cell_to_input: Vec<u32>,
        input_to_cell: Vec<Option<u32>>,
    ) -> Self {
        debug_assert_eq!(cell_cycles.len(), cell_source_sites.len());
        debug_assert_eq!(cell_cycles.len(), cell_to_input.len());
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
    pub fn vertices(&self) -> &[UnitVec3] {
        &self.vertices
    }

    /// Return a mesh vertex.
    ///
    /// # Panics
    ///
    /// Panics when `index >= self.num_vertices()`.
    #[inline]
    #[track_caller]
    pub fn vertex(&self, index: usize) -> UnitVec3 {
        self.vertices[index]
    }

    /// Checked form of [`Self::vertex`].
    #[inline]
    pub fn get_vertex(&self, index: usize) -> Option<UnitVec3> {
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
    pub fn source_site(&self, cell: usize) -> UnitVec3 {
        self.cell_source_sites[cell]
    }

    /// Checked form of [`Self::source_site`].
    #[inline]
    pub fn get_source_site(&self, cell: usize) -> Option<UnitVec3> {
        self.cell_source_sites.get(cell).copied()
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
            "invalid spherical cell mesh (chi={}, components={}, degenerate={}, edge_issues={}, link_issues={}, provenance_issues={})",
            self.euler_characteristic,
            self.connected_components,
            self.degenerate_cells,
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
    /// Original construction, repair, and output-resolution report.
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
}

fn prepare_elided_cell_mesh(
    source: &crate::ComputeOutput,
) -> Result<(SphericalCellMesh, CellElisionReport), (CellElisionErrorKind, String)> {
    if source.report.has_post_repair_residuals() {
        return Err((
            CellElisionErrorKind::InvalidSource,
            "source computation has post-repair or strict-validation residuals".into(),
        ));
    }

    let preferred = source.preferred_diagram();
    let generators: Vec<glam::Vec3> = preferred
        .generators()
        .iter()
        .map(|site| glam::Vec3::new(site.x, site.y, site.z))
        .collect();
    let vertices: Vec<glam::Vec3> = preferred
        .vertices()
        .iter()
        .map(|vertex| glam::Vec3::new(vertex.x, vertex.y, vertex.z))
        .collect();
    let mut cells = Vec::with_capacity(preferred.num_cells());
    let mut cell_indices = Vec::new();
    for cell in preferred.iter_cells() {
        if cell.vertex_indices.len() > u16::MAX as usize || cell_indices.len() > u32::MAX as usize {
            return Err((
                CellElisionErrorKind::RepresentationLimit,
                "source cell layout exceeds compact mesh index capacity".into(),
            ));
        }
        cells.push(crate::diagram::VoronoiCell::new(
            cell_indices.len() as u32,
            cell.vertex_indices.len() as u16,
        ));
        cell_indices.extend_from_slice(cell.vertex_indices);
    }

    let elision = crate::knn_clipping::output_resolution::elide_exact_zero_cells_for_mesh(
        &generators,
        &vertices,
        &cells,
        &cell_indices,
    )
    .map_err(|error| (CellElisionErrorKind::UnsafeQuotient, error.to_string()))?;

    let original_cells = source.diagram.num_cells();
    if original_cells > u32::MAX as usize || preferred.num_cells() > u32::MAX as usize {
        return Err((
            CellElisionErrorKind::RepresentationLimit,
            "source input mapping exceeds u32 capacity".into(),
        ));
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
        return Err((
            CellElisionErrorKind::InvalidSource,
            "source weld mapping does not match the effective diagram".into(),
        ));
    }
    let mut input_to_effective = Vec::with_capacity(original_cells);
    for input in 0..original_cells {
        let canonical = source.diagram.canonical_cell_index(input);
        let effective = canonical_to_effective[canonical];
        if effective == u32::MAX {
            return Err((
                CellElisionErrorKind::InvalidSource,
                "source weld mapping names a noncanonical effective cell".into(),
            ));
        }
        input_to_effective.push(effective);
    }

    let input_to_cell: Vec<Option<u32>> = input_to_effective
        .iter()
        .map(|&effective| elision.effective_to_cell[effective as usize])
        .collect();
    let source_inputs_elided = input_to_cell.iter().filter(|cell| cell.is_none()).count();
    let cell_to_input: Vec<u32> = elision
        .cell_to_effective
        .iter()
        .map(|&effective| effective_to_input[effective as usize])
        .collect();
    let cell_source_sites: Vec<UnitVec3> = elision
        .cell_to_effective
        .iter()
        .map(|&effective| preferred.generator(effective as usize))
        .collect();
    let final_vertices = elision.diagram.vertices().to_vec();
    let final_cycles: Vec<Vec<u32>> = elision
        .diagram
        .iter_cells()
        .map(|cell| cell.vertex_indices.to_vec())
        .collect();
    let vertices_removed = preferred.num_vertices() - final_vertices.len();
    let mesh = SphericalCellMesh::from_raw_parts(
        final_vertices,
        final_cycles,
        cell_source_sites,
        cell_to_input,
        input_to_cell,
    );
    let validation = mesh.validate();
    if !validation.is_strictly_valid() {
        return Err((CellElisionErrorKind::UnsafeQuotient, validation.headline()));
    }

    Ok((
        mesh,
        CellElisionReport {
            exact_zero_edges_detected: elision.zero_edges_before,
            exact_zero_components_detected: elision.zero_components_before,
            effective_cells_elided: elision.effective_cells_elided,
            source_inputs_elided,
            degree_two_vertices_suppressed: elision.degree_two_vertices_suppressed,
            vertices_removed,
            max_suppression_cross_track_radians: elision.max_suppression_cross_track_radians,
            validation,
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
        let len_sq = vertex.x * vertex.x + vertex.y * vertex.y + vertex.z * vertex.z;
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

    for cell in mesh.iter_cells() {
        let cycle = cell.vertex_indices;
        if cycle.len() < 3 {
            degenerate_cells += 1;
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
            if va.x == -vb.x && va.y == -vb.y && va.z == -vb.z {
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
                let len_sq = site.x * site.x + site.y * site.y + site.z * site.z;
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

    fn unit(x: f32, y: f32, z: f32) -> UnitVec3 {
        UnitVec3::new(x, y, z).normalize()
    }

    fn fixture_mesh(vertices: Vec<UnitVec3>, cycles: &[&[u32]]) -> SphericalCellMesh {
        SphericalCellMesh::from_raw_parts(
            vertices,
            cycles.iter().map(|cycle| cycle.to_vec()).collect(),
            vec![UnitVec3::new(0.0, 0.0, 1.0); cycles.len()],
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
            UnitVec3::new(1.0, 0.0, 0.0),
            UnitVec3::new(-1.0, 0.0, 0.0),
            UnitVec3::new(0.0, 1.0, 0.0),
            UnitVec3::new(0.0, -1.0, 0.0),
            UnitVec3::new(0.0, 0.0, 1.0),
            UnitVec3::new(0.0, 0.0, -1.0),
        ];
        let mut output = crate::compute_with_report(&points, crate::VoronoiConfig::default())
            .expect("octahedral source should compute");
        output.report.post_repair_unpaired_edges.push((0, 1));
        let cell_count = output.diagram.num_cells();

        let error = output
            .into_elided_cell_mesh()
            .expect_err("a source with explicit residuals must be rejected");
        assert_eq!(error.kind(), CellElisionErrorKind::InvalidSource);
        assert_eq!(error.source_output().diagram.num_cells(), cell_count);
        let recovered = error.into_source_output();
        assert_eq!(recovered.diagram.num_cells(), cell_count);
        assert_eq!(recovered.report.post_repair_unpaired_edges, [(0, 1)]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip_revalidates_dense_mesh() {
        let points = [
            UnitVec3::new(1.0, 0.0, 0.0),
            UnitVec3::new(-1.0, 0.0, 0.0),
            UnitVec3::new(0.0, 1.0, 0.0),
            UnitVec3::new(0.0, -1.0, 0.0),
            UnitVec3::new(0.0, 0.0, 1.0),
            UnitVec3::new(0.0, 0.0, -1.0),
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
