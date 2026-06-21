//! Strict validation for spherical Voronoi diagrams.
//!
//! This module checks whether a diagram is a coherent S2 subdivision and whether
//! exact representation invariants hold. It intentionally does not try to score
//! approximate Voronoi fidelity or generic-position heuristics.

use crate::SphericalVoronoi;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::tolerances::{ANTIPODAL_DOT_EPS, VERTEX_ON_SPHERE_EPS};

/// Detailed validation report for a spherical Voronoi diagram.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Number of cells (faces) in the diagram.
    pub num_cells: usize,
    /// Number of stored vertices in the diagram.
    pub num_vertices: usize,
    /// Number of referenced (non-orphan) vertices.
    pub used_vertices: usize,
    /// Number of undirected edges in the cell-boundary graph.
    pub num_edges: usize,

    /// Euler characteristic V - E + F using referenced vertices and all cells.
    pub euler_characteristic: i32,
    /// Number of connected components in the cell adjacency graph.
    pub connected_components: usize,

    /// Number of cells with fewer than 3 distinct vertices.
    pub degenerate_cells: usize,
    /// Number of cells with duplicate vertex indices in the boundary cycle.
    pub cells_with_duplicate_vertices: usize,
    /// Number of cells that reference at least one out-of-range vertex index.
    pub cells_with_invalid_references: usize,
    /// Total number of out-of-range vertex references across all cells.
    pub invalid_vertex_references: usize,
    /// Number of duplicate cells (identical sorted boundary signatures) among
    /// canonical cells. Welded twins are accounted separately and are not
    /// duplicates.
    pub duplicate_cells_count: usize,
    /// Number of topologically unique cells.
    pub unique_cells: usize,
    /// Number of cells that are welded twins aliasing a canonical cell.
    /// Informational: welded twins are part of the documented coincident-input
    /// contract, not a defect.
    pub welded_twin_cells: usize,
    /// Number of weld-map inconsistencies (twin not aliasing its canonical
    /// cell's boundary, or canonical index not itself canonical).
    pub weld_map_issues: usize,

    /// Sum of raw cell boundary lengths.
    pub total_cell_vertices: usize,

    /// Number of stored vertices not on the unit sphere within tolerance.
    pub vertices_off_sphere: usize,
    /// Number of vertices that are referenced by no cells.
    ///
    /// Representation note, not a defect: edge repair may leave unreferenced
    /// vertices behind rather than paying a compaction pass (they do not
    /// participate in the subdivision and the Euler count ignores them). Use
    /// [`crate::SphericalVoronoi::compact_vertices`] to remove them.
    pub orphan_vertices: usize,
    /// Number of referenced vertices with degree 1 or 2.
    pub low_incidence_vertices: usize,
    /// Vertex incidence histogram: [d0, d1, d2, d3, d4+].
    pub degree_counts: [usize; 5],

    /// Number of self-loop edges `(v, v)` in cell boundary cycles.
    pub self_loop_edges: usize,
    /// Number of edges whose endpoints are antipodal or near-antipodal.
    pub antipodal_edges: usize,
    /// Number of undirected edges referenced by fewer than 2 directed edges.
    pub boundary_edges: usize,
    /// Number of undirected edges referenced by more than 2 directed edges.
    pub overused_edges: usize,
    /// Number of undirected edges whose two directed uses do not appear with opposite orientation.
    pub same_direction_edge_pairs: usize,
}

impl ValidationReport {
    /// Returns true when the diagram is a coherent S2 subdivision and all exact
    /// representation invariants hold.
    pub fn is_strictly_valid(&self) -> bool {
        self.subdivision_issues().is_empty() && self.invariant_issues().is_empty()
    }

    /// Human-readable subdivision failures.
    pub fn subdivision_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.degenerate_cells > 0 {
            issues.push(format!("{} degenerate cells", self.degenerate_cells));
        }
        if self.cells_with_duplicate_vertices > 0 {
            issues.push(format!(
                "{} cells with duplicate boundary vertices",
                self.cells_with_duplicate_vertices
            ));
        }
        if self.duplicate_cells_count > 0 {
            issues.push(format!(
                "{} duplicate cells ({} unique of {})",
                self.duplicate_cells_count, self.unique_cells, self.num_cells
            ));
        }
        if self.boundary_edges > 0 {
            issues.push(format!("{} unpaired edges", self.boundary_edges));
        }
        if self.overused_edges > 0 {
            issues.push(format!("{} overused edges", self.overused_edges));
        }
        if self.same_direction_edge_pairs > 0 {
            issues.push(format!(
                "{} edges reused with the same orientation",
                self.same_direction_edge_pairs
            ));
        }
        if self.low_incidence_vertices > 0 {
            let [_d0, d1, d2, _d3, _d4p] = self.degree_counts;
            let mut parts = Vec::new();
            if d1 > 0 {
                parts.push(format!("d1:{}", d1));
            }
            if d2 > 0 {
                parts.push(format!("d2:{}", d2));
            }
            issues.push(format!(
                "{} low-incidence vertices ({})",
                self.low_incidence_vertices,
                parts.join(", ")
            ));
        }
        if self.connected_components != 1 {
            issues.push(format!(
                "{} connected components",
                self.connected_components
            ));
        }
        if self.euler_characteristic != 2 {
            issues.push(format!("Euler={} (expected 2)", self.euler_characteristic));
        }

        issues
    }

    /// Human-readable exact-invariant failures.
    pub fn invariant_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.cells_with_invalid_references > 0 {
            issues.push(format!(
                "{} invalid vertex references across {} cells",
                self.invalid_vertex_references, self.cells_with_invalid_references
            ));
        }
        if self.vertices_off_sphere > 0 {
            issues.push(format!("{} vertices off sphere", self.vertices_off_sphere));
        }
        if self.self_loop_edges > 0 {
            issues.push(format!("{} self-loop edges", self.self_loop_edges));
        }
        if self.antipodal_edges > 0 {
            issues.push(format!("{} antipodal edges", self.antipodal_edges));
        }
        if self.weld_map_issues > 0 {
            issues.push(format!("{} weld-map inconsistencies", self.weld_map_issues));
        }

        issues
    }

    /// Representation notes: properties of the stored representation that are
    /// part of the documented contract rather than defects.
    pub fn representation_notes(&self) -> Vec<String> {
        let mut notes = Vec::new();

        if self.orphan_vertices > 0 {
            notes.push(format!("{} orphan vertices", self.orphan_vertices));
        }
        if self.welded_twin_cells > 0 {
            notes.push(format!("{} welded twins", self.welded_twin_cells));
        }

        notes
    }

    /// Single-line headline for logs and debug output.
    pub fn headline(&self) -> String {
        let notes = self.representation_notes();
        let notes_suffix = if notes.is_empty() {
            String::new()
        } else {
            format!(" (notes: {})", notes.join(", "))
        };

        if self.is_strictly_valid() {
            return format!("Strictly valid{notes_suffix}");
        }

        let subdivision = self.subdivision_issues();
        let invariants = self.invariant_issues();
        let mut parts = Vec::new();

        if !subdivision.is_empty() {
            parts.push(format!("subdivision: {}", subdivision.join(", ")));
        }
        if !invariants.is_empty() {
            parts.push(format!("invariants: {}", invariants.join(", ")));
        }

        format!("{}{notes_suffix}", parts.join("; "))
    }
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ValidationReport {{ V={}, used_V={}, E={}, F={}, components={}, euler={}, {} }}",
            self.num_vertices,
            self.used_vertices,
            self.num_edges,
            self.num_cells,
            self.connected_components,
            self.euler_characteristic,
            self.headline()
        )
    }
}

#[derive(Debug, Clone, Default)]
struct EdgeStat {
    forward: usize,
    reverse: usize,
    cells: Vec<usize>,
}

const INLINE_CELL_SIGNATURE_VERTS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CellSignature {
    Inline {
        len: u8,
        vertices: [u32; INLINE_CELL_SIGNATURE_VERTS],
    },
    Heap(Vec<u32>),
}

fn cell_signature(vertices: &[u32]) -> Option<CellSignature> {
    if vertices.is_empty() {
        return None;
    }
    if vertices.len() <= INLINE_CELL_SIGNATURE_VERTS {
        let mut out = [0u32; INLINE_CELL_SIGNATURE_VERTS];
        out[..vertices.len()].copy_from_slice(vertices);
        out[..vertices.len()].sort_unstable();
        Some(CellSignature::Inline {
            len: vertices.len() as u8,
            vertices: out,
        })
    } else {
        let mut out = vertices.to_vec();
        out.sort_unstable();
        Some(CellSignature::Heap(out))
    }
}

#[derive(Debug, Clone)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let parent = self.parent[x];
        if parent != x {
            let root = self.find(parent);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
    }
}

/// Validate whether a diagram is a coherent S2 subdivision plus exact
/// representation invariants.
///
/// This intentionally excludes approximate Voronoi-fidelity or generic-position
/// heuristics.
pub fn validate(diagram: &SphericalVoronoi) -> ValidationReport {
    validate_impl(diagram)
}

/// Opt-in post-build verification gate (env `S2_VORONOI_VERIFY=1`).
///
/// The full topological validator is O(E) and is skipped by the plain
/// `compute` / `compute_plane` fast paths for speed (the report-returning
/// entry points already validate unconditionally). With the net's repair
/// scans gated on detection records, a defect that left no record would
/// ship silently on those paths. Enabling this runs the validator after
/// every build and turns any strict-validity failure into an error —
/// belt-and-braces for callers who want output validity machine-checked
/// regardless of cost.
pub(crate) fn verify_enabled() -> bool {
    matches!(std::env::var("S2_VORONOI_VERIFY"), Ok(v) if v == "1")
}

/// Run the sphere validator and map a strict-validity failure to an error
/// when [`verify_enabled`]. No-op otherwise.
pub(crate) fn verify_sphere_if_enabled(
    diagram: &SphericalVoronoi,
) -> Result<(), crate::VoronoiError> {
    if !verify_enabled() {
        return Ok(());
    }
    match verify_sphere_fast(diagram) {
        Ok(()) => return Ok(()),
        Err(reason) => {
            if matches!(std::env::var("S2_VORONOI_VERIFY_TRACE"), Ok(v) if v == "1") {
                eprintln!("S2_VORONOI_VERIFY fast path fell back: {reason}");
            }
        }
    }

    let report = validate_impl(diagram);
    if report.is_strictly_valid() {
        return Ok(());
    }
    let mut issues = report.subdivision_issues();
    issues.extend(report.invariant_issues());
    Err(crate::VoronoiError::ComputationFailed(format!(
        "S2_VORONOI_VERIFY: returned diagram failed strict validation: {}",
        issues.join("; ")
    )))
}

#[derive(Debug, Clone, Copy)]
struct EdgeUse {
    key: u64,
    forward: bool,
    cell: u32,
}

#[inline]
fn edge_key(lo: u32, hi: u32) -> u64 {
    ((lo as u64) << 32) | hi as u64
}

/// Fast success-path verifier for `S2_VORONOI_VERIFY`.
///
/// This checks the same strict sphere contract as [`validate_impl`], but does
/// not build the detailed diagnostic report. On failure the caller reruns the
/// full validator to produce the public error message.
fn verify_sphere_fast(diagram: &SphericalVoronoi) -> Result<(), &'static str> {
    let num_cells = diagram.num_cells();
    let num_vertices = diagram.num_vertices();

    let mut welded_twin_cells = 0usize;
    let mut is_welded_twin = vec![false; num_cells];
    for (i, twin_flag) in is_welded_twin.iter_mut().enumerate() {
        let canonical = diagram.canonical_cell_index(i);
        if canonical == i {
            continue;
        }
        welded_twin_cells += 1;
        *twin_flag = true;
        let canonical_is_canonical =
            canonical < num_cells && diagram.canonical_cell_index(canonical) == canonical;
        if !canonical_is_canonical
            || diagram.cell(i).vertex_indices != diagram.cell(canonical).vertex_indices
        {
            return Err("weld map");
        }
    }
    let num_faces = num_cells - welded_twin_cells;

    let estimated_directed_edges = diagram.cell_indices_raw().len();
    let mut unique_cell_signatures: FxHashSet<CellSignature> =
        FxHashSet::with_capacity_and_hasher(num_faces.max(1), Default::default());
    let mut vertex_cell_count = vec![0u8; num_vertices];
    let mut edge_uses = Vec::with_capacity(estimated_directed_edges);

    for cell in diagram.iter_cells() {
        if is_welded_twin[cell.generator_index] {
            continue;
        }
        let len = cell.len();
        let mut seen_stack = [0u32; 64];
        let mut seen_stack_len = 0usize;
        let mut seen_spill = if len > seen_stack.len() {
            Vec::with_capacity(len)
        } else {
            Vec::new()
        };
        let use_spill = len > seen_stack.len();

        for &vi in cell.vertex_indices {
            if (vi as usize) >= num_vertices {
                return Err("invalid vertex reference");
            }

            let is_duplicate = if use_spill {
                if seen_spill.contains(&vi) {
                    true
                } else {
                    seen_spill.push(vi);
                    false
                }
            } else if seen_stack[..seen_stack_len].contains(&vi) {
                true
            } else {
                seen_stack[seen_stack_len] = vi;
                seen_stack_len += 1;
                false
            };

            if is_duplicate {
                return Err("duplicate vertex in cell");
            }
            let count = &mut vertex_cell_count[vi as usize];
            *count = count.saturating_add(1);
        }

        let seen_valid_len = if use_spill {
            seen_spill.len()
        } else {
            seen_stack_len
        };
        if seen_valid_len < 3 {
            return Err("degenerate cell");
        }

        let signature = if use_spill {
            cell_signature(&seen_spill)
        } else {
            cell_signature(&seen_stack[..seen_stack_len])
        };
        if let Some(signature) = signature {
            if !unique_cell_signatures.insert(signature) {
                return Err("duplicate cell");
            }
        }

        for edge_idx in 0..len {
            let a = cell.vertex_indices[edge_idx];
            let b = cell.vertex_indices[(edge_idx + 1) % len];
            if a == b {
                return Err("self-loop edge");
            }

            let va = diagram.vertex(a as usize);
            let vb = diagram.vertex(b as usize);
            let dot = va.x * vb.x + va.y * vb.y + va.z * vb.z;
            if dot <= -1.0 + ANTIPODAL_DOT_EPS {
                return Err("antipodal edge");
            }

            let (lo, hi, forward) = if a < b { (a, b, true) } else { (b, a, false) };
            edge_uses.push(EdgeUse {
                key: edge_key(lo, hi),
                forward,
                cell: cell.generator_index as u32,
            });
        }
    }

    let mut used_vertices = 0usize;
    for &count in &vertex_cell_count {
        if count > 0 {
            used_vertices += 1;
            if count < 3 {
                return Err("low-incidence vertex");
            }
        }
    }

    for v in diagram.vertices() {
        let len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
        if (len_sq - 1.0).abs() > VERTEX_ON_SPHERE_EPS {
            return Err("off-sphere vertex");
        }
    }

    edge_uses.sort_unstable_by_key(|edge| edge.key);

    let mut dsu = DisjointSet::new(num_cells);
    let mut num_edges = 0usize;
    let mut i = 0usize;
    while i < edge_uses.len() {
        num_edges += 1;
        let first = edge_uses[i];
        let mut j = i + 1;
        while j < edge_uses.len() && edge_uses[j].key == first.key {
            j += 1;
        }

        let group = &edge_uses[i..j];
        if group.len() != 2 || group[0].forward == group[1].forward {
            return Err("unpaired, overused, or misoriented edge");
        }
        if group[0].cell != group[1].cell {
            dsu.union(group[0].cell as usize, group[1].cell as usize);
        }
        i = j;
    }

    let connected_components = if num_faces == 0 {
        0
    } else {
        let mut roots: FxHashSet<usize> =
            FxHashSet::with_capacity_and_hasher(num_faces, Default::default());
        for (cell_idx, &is_twin) in is_welded_twin.iter().enumerate() {
            if is_twin {
                continue;
            }
            roots.insert(dsu.find(cell_idx));
        }
        roots.len()
    };
    if connected_components != 1 {
        return Err("disconnected subdivision");
    }

    let euler_characteristic = used_vertices as i32 - num_edges as i32 + num_faces as i32;
    if euler_characteristic != 2 {
        return Err("bad euler characteristic");
    }

    Ok(())
}

/// Strict S2-subdivision check over raw effective arrays (no weld map),
/// enforcing the SAME contract as [`verify_sphere_fast`].
///
/// Retained (and covered by `effective_strict_matches_fast`) as a standalone
/// effective-space strict validator: it has no current production caller after the
/// Tier-2 re-clip repair was removed, but is the natural gate for a future
/// locally-resolved region (the planned hybrid exact-predicate path).
///
/// Was used by the Tier-2 re-clip repair to gate its output against the full
/// validator *inside the repair* — independent of the `S2_VORONOI_VERIFY` env
/// flag (so both the plain and report paths are covered) and without cloning the
/// diagram into a `SphericalVoronoi`. Effective index space has no welded twins,
/// so every cell is its own face (`num_faces == num_cells`).
///
/// Pinned to `verify_sphere_fast` by the differential test
/// `effective_strict_matches_fast`.
#[allow(dead_code)] // no production caller post-Tier-2-removal; kept for tests + future hybrid
pub(crate) fn verify_sphere_effective_strict(
    vertices: &[glam::Vec3],
    cells: &[crate::diagram::VoronoiCell],
    cell_indices: &[u32],
) -> Result<(), &'static str> {
    let num_cells = cells.len();
    let num_vertices = vertices.len();

    let mut unique_cell_signatures: FxHashSet<CellSignature> =
        FxHashSet::with_capacity_and_hasher(num_cells.max(1), Default::default());
    let mut vertex_cell_count = vec![0u8; num_vertices];
    let mut edge_uses = Vec::with_capacity(cell_indices.len());

    for (ci, cell) in cells.iter().enumerate() {
        let start = cell.vertex_start();
        let len = cell.vertex_count();
        let Some(span) = len
            .checked_add(start)
            .and_then(|end| cell_indices.get(start..end))
        else {
            return Err("invalid cell span");
        };

        let mut seen_stack = [0u32; 64];
        let mut seen_stack_len = 0usize;
        let mut seen_spill = if len > seen_stack.len() {
            Vec::with_capacity(len)
        } else {
            Vec::new()
        };
        let use_spill = len > seen_stack.len();

        for &vi in span {
            if (vi as usize) >= num_vertices {
                return Err("invalid vertex reference");
            }
            let is_duplicate = if use_spill {
                if seen_spill.contains(&vi) {
                    true
                } else {
                    seen_spill.push(vi);
                    false
                }
            } else if seen_stack[..seen_stack_len].contains(&vi) {
                true
            } else {
                seen_stack[seen_stack_len] = vi;
                seen_stack_len += 1;
                false
            };
            if is_duplicate {
                return Err("duplicate vertex in cell");
            }
            let count = &mut vertex_cell_count[vi as usize];
            *count = count.saturating_add(1);
        }

        let seen_valid_len = if use_spill {
            seen_spill.len()
        } else {
            seen_stack_len
        };
        if seen_valid_len < 3 {
            return Err("degenerate cell");
        }

        let signature = if use_spill {
            cell_signature(&seen_spill)
        } else {
            cell_signature(&seen_stack[..seen_stack_len])
        };
        if let Some(signature) = signature {
            if !unique_cell_signatures.insert(signature) {
                return Err("duplicate cell");
            }
        }

        for edge_idx in 0..len {
            let a = span[edge_idx];
            let b = span[(edge_idx + 1) % len];
            if a == b {
                return Err("self-loop edge");
            }
            let va = vertices[a as usize];
            let vb = vertices[b as usize];
            let dot = va.x * vb.x + va.y * vb.y + va.z * vb.z;
            if dot <= -1.0 + ANTIPODAL_DOT_EPS {
                return Err("antipodal edge");
            }
            let (lo, hi, forward) = if a < b { (a, b, true) } else { (b, a, false) };
            edge_uses.push(EdgeUse {
                key: edge_key(lo, hi),
                forward,
                cell: ci as u32,
            });
        }
    }

    let mut used_vertices = 0usize;
    for &count in &vertex_cell_count {
        if count > 0 {
            used_vertices += 1;
            if count < 3 {
                return Err("low-incidence vertex");
            }
        }
    }

    for v in vertices {
        let len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
        if (len_sq - 1.0).abs() > VERTEX_ON_SPHERE_EPS {
            return Err("off-sphere vertex");
        }
    }

    edge_uses.sort_unstable_by_key(|edge| edge.key);

    let mut dsu = DisjointSet::new(num_cells);
    let mut num_edges = 0usize;
    let mut i = 0usize;
    while i < edge_uses.len() {
        num_edges += 1;
        let first = edge_uses[i];
        let mut j = i + 1;
        while j < edge_uses.len() && edge_uses[j].key == first.key {
            j += 1;
        }
        let group = &edge_uses[i..j];
        if group.len() != 2 || group[0].forward == group[1].forward {
            return Err("unpaired, overused, or misoriented edge");
        }
        if group[0].cell != group[1].cell {
            dsu.union(group[0].cell as usize, group[1].cell as usize);
        }
        i = j;
    }

    let connected_components = if num_cells == 0 {
        0
    } else {
        let mut roots: FxHashSet<usize> =
            FxHashSet::with_capacity_and_hasher(num_cells, Default::default());
        for cell_idx in 0..num_cells {
            roots.insert(dsu.find(cell_idx));
        }
        roots.len()
    };
    if connected_components != 1 {
        return Err("disconnected subdivision");
    }

    let euler_characteristic = used_vertices as i32 - num_edges as i32 + num_cells as i32;
    if euler_characteristic != 2 {
        return Err("bad euler characteristic");
    }

    Ok(())
}

/// Run the plane validator and map a strict-validity failure to an error
/// when [`verify_enabled`]. No-op otherwise.
pub(crate) fn verify_plane_if_enabled(
    diagram: &crate::PlanarVoronoi,
) -> Result<(), crate::VoronoiError> {
    if !verify_enabled() {
        return Ok(());
    }
    let report = validate_plane(diagram);
    if report.is_strictly_valid() {
        return Ok(());
    }
    Err(crate::VoronoiError::ComputationFailed(format!(
        "S2_VORONOI_VERIFY: returned plane diagram failed strict validation: {report:?}"
    )))
}

fn validate_impl(diagram: &SphericalVoronoi) -> ValidationReport {
    let num_cells = diagram.num_cells();
    let num_vertices = diagram.num_vertices();
    let vertices = diagram.vertices();

    // Welded twins alias a canonical cell's boundary; the subdivision is
    // accounted over canonical cells only, after checking the aliasing holds.
    let mut welded_twin_cells = 0usize;
    let mut weld_map_issues = 0usize;
    let mut is_welded_twin = vec![false; num_cells];
    for (i, twin_flag) in is_welded_twin.iter_mut().enumerate() {
        let canonical = diagram.canonical_cell_index(i);
        if canonical == i {
            continue;
        }
        welded_twin_cells += 1;
        *twin_flag = true;
        let canonical_is_canonical =
            canonical < num_cells && diagram.canonical_cell_index(canonical) == canonical;
        if !canonical_is_canonical
            || diagram.cell(i).vertex_indices != diagram.cell(canonical).vertex_indices
        {
            weld_map_issues += 1;
        }
    }
    let num_faces = num_cells - welded_twin_cells;

    let estimated_directed_edges = diagram.cell_indices_raw().len();
    let estimated_undirected_edges = (estimated_directed_edges / 2).max(1);

    let mut unique_cell_signatures: FxHashSet<CellSignature> =
        FxHashSet::with_capacity_and_hasher(num_faces.max(1), Default::default());
    let mut duplicate_cells_count = 0usize;
    let mut vertex_cell_count: Vec<u32> = vec![0; num_vertices];

    let mut total_cell_vertices = 0usize;
    let mut degenerate_cells = 0usize;
    let mut cells_with_duplicate_vertices = 0usize;
    let mut cells_with_invalid_references = 0usize;
    let mut invalid_vertex_references = 0usize;
    let mut self_loop_edges = 0usize;
    let mut antipodal_edges = 0usize;

    let mut edges: FxHashMap<u64, EdgeStat> =
        FxHashMap::with_capacity_and_hasher(estimated_undirected_edges, Default::default());

    for cell in diagram.iter_cells() {
        if is_welded_twin[cell.generator_index] {
            continue;
        }
        let len = cell.len();
        total_cell_vertices += len;

        let mut seen_stack = [0u32; 64];
        let mut seen_stack_len = 0usize;
        let mut seen_spill = if len > seen_stack.len() {
            Vec::with_capacity(len)
        } else {
            Vec::new()
        };
        let use_spill = len > seen_stack.len();
        let mut cell_has_duplicate_vertices = false;
        let mut cell_has_invalid_reference = false;

        for &vi in cell.vertex_indices {
            if (vi as usize) >= num_vertices {
                invalid_vertex_references += 1;
                cell_has_invalid_reference = true;
                continue;
            }

            let is_duplicate = if use_spill {
                if seen_spill.contains(&vi) {
                    true
                } else {
                    seen_spill.push(vi);
                    false
                }
            } else if seen_stack[..seen_stack_len].contains(&vi) {
                true
            } else {
                seen_stack[seen_stack_len] = vi;
                seen_stack_len += 1;
                false
            };

            if is_duplicate {
                cell_has_duplicate_vertices = true;
            } else {
                vertex_cell_count[vi as usize] += 1;
            }
        }

        if cell_has_duplicate_vertices {
            cells_with_duplicate_vertices += 1;
        }
        if cell_has_invalid_reference {
            cells_with_invalid_references += 1;
        }
        let seen_valid_len = if use_spill {
            seen_spill.len()
        } else {
            seen_stack_len
        };
        if seen_valid_len < 3 {
            degenerate_cells += 1;
        }

        // Canonical duplicate-cell signature over valid references only.
        let signature = if use_spill {
            cell_signature(&seen_spill)
        } else {
            cell_signature(&seen_stack[..seen_stack_len])
        };
        if let Some(signature) = signature {
            if !unique_cell_signatures.insert(signature) {
                duplicate_cells_count += 1;
            }
        }

        if len < 2 {
            continue;
        }

        for edge_idx in 0..len {
            let a = cell.vertex_indices[edge_idx];
            let b = cell.vertex_indices[(edge_idx + 1) % len];

            if (a as usize) >= num_vertices || (b as usize) >= num_vertices {
                continue;
            }
            if a == b {
                self_loop_edges += 1;
                continue;
            }

            let va = diagram.vertex(a as usize);
            let vb = diagram.vertex(b as usize);
            let dot = va.x * vb.x + va.y * vb.y + va.z * vb.z;
            if dot <= -1.0 + ANTIPODAL_DOT_EPS {
                antipodal_edges += 1;
                continue;
            }

            let (lo, hi, forward) = if a < b { (a, b, true) } else { (b, a, false) };
            let stat = edges.entry(edge_key(lo, hi)).or_default();
            if forward {
                stat.forward += 1;
            } else {
                stat.reverse += 1;
            }
            stat.cells.push(cell.generator_index);
        }
    }

    let unique_cells = unique_cell_signatures.len();

    let mut orphan_vertices = 0usize;
    let mut low_incidence_vertices = 0usize;
    let mut degree_counts = [0usize; 5];
    for &count in &vertex_cell_count {
        match count {
            0 => {
                orphan_vertices += 1;
                degree_counts[0] += 1;
            }
            1 => {
                low_incidence_vertices += 1;
                degree_counts[1] += 1;
            }
            2 => {
                low_incidence_vertices += 1;
                degree_counts[2] += 1;
            }
            3 => degree_counts[3] += 1,
            _ => degree_counts[4] += 1,
        }
    }

    let mut boundary_edges = 0usize;
    let mut overused_edges = 0usize;
    let mut same_direction_edge_pairs = 0usize;
    let num_edges = edges.len();

    let mut dsu = DisjointSet::new(num_cells);
    for stat in edges.values() {
        let total = stat.forward + stat.reverse;
        if total < 2 {
            boundary_edges += 1;
        } else if total > 2 {
            overused_edges += 1;
        } else if stat.forward != 1 || stat.reverse != 1 {
            same_direction_edge_pairs += 1;
        }

        if let Some((&first, rest)) = stat.cells.split_first() {
            for (offset, &other) in rest.iter().enumerate() {
                if stat.cells[..=offset].contains(&other) {
                    continue;
                }
                dsu.union(first, other);
            }
        }
    }

    let connected_components = if num_faces == 0 {
        0
    } else {
        let mut roots: FxHashSet<usize> =
            FxHashSet::with_capacity_and_hasher(num_faces, Default::default());
        for (cell_idx, &is_twin) in is_welded_twin.iter().enumerate() {
            if is_twin {
                continue;
            }
            roots.insert(dsu.find(cell_idx));
        }
        roots.len()
    };

    let used_vertices = num_vertices.saturating_sub(orphan_vertices);
    let euler_characteristic = used_vertices as i32 - num_edges as i32 + num_faces as i32;

    let mut vertices_off_sphere = 0usize;
    for v in vertices {
        let len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
        if (len_sq - 1.0).abs() > VERTEX_ON_SPHERE_EPS {
            vertices_off_sphere += 1;
        }
    }

    ValidationReport {
        num_cells,
        num_vertices,
        used_vertices,
        num_edges,
        euler_characteristic,
        connected_components,
        degenerate_cells,
        cells_with_duplicate_vertices,
        cells_with_invalid_references,
        invalid_vertex_references,
        duplicate_cells_count,
        unique_cells,
        welded_twin_cells,
        weld_map_issues,
        total_cell_vertices,
        vertices_off_sphere,
        orphan_vertices,
        low_incidence_vertices,
        degree_counts,
        self_loop_edges,
        antipodal_edges,
        boundary_edges,
        overused_edges,
        same_direction_edge_pairs,
    }
}

// === Planar validation ===

/// Validation report for a [`crate::PlanarVoronoi`] diagram.
///
/// The strict contract for a bounded-rect planar diagram: canonical cells
/// form a subdivision of the rect — every interior edge shared by exactly
/// two cells with opposite orientations, every single-use edge lying on the
/// rect boundary, and disk topology (`V - E + (F + 1) = 2` counting the
/// outer face).
#[derive(Debug, Clone)]
pub struct PlaneValidationReport {
    /// Number of cells (one per input generator, including welded twins).
    pub num_cells: usize,
    /// Number of canonical (non-twin) cells.
    pub canonical_cells: usize,
    /// Number of welded twin cells (informational, part of the weld contract).
    pub welded_twin_cells: usize,
    /// Weld-map inconsistencies (twin not aliasing its canonical cell).
    pub weld_map_issues: usize,
    /// Number of stored vertices.
    pub num_vertices: usize,
    /// Number of vertices referenced by at least one canonical cell.
    pub used_vertices: usize,
    /// Number of undirected edges over canonical cells.
    pub num_edges: usize,
    /// Single-use edges that lie on the rect boundary (expected for hull cells).
    pub boundary_edges: usize,
    /// Normalized Euler characteristic: `V - E + F + 1` (outer face) for
    /// bounded diagrams, `V - E + F + 2` for periodic ones — 2 for a valid
    /// subdivision in both topologies.
    pub euler_characteristic: i32,
    /// Cells with fewer than 3 distinct vertices.
    pub degenerate_cells: usize,
    /// Cells with duplicate vertex indices in their boundary cycle.
    pub cells_with_duplicate_vertices: usize,
    /// Cells referencing out-of-range vertex indices.
    pub cells_with_invalid_references: usize,
    /// Total out-of-range vertex references.
    pub invalid_vertex_references: usize,
    /// Single-use edges NOT on the rect boundary (a strict-validity defect).
    pub unpaired_interior_edges: usize,
    /// Edges used by more than two cells.
    pub overused_edges: usize,
    /// Edges used twice with the same orientation (winding defect).
    pub misoriented_edges: usize,
    /// Vertices outside the rect beyond tolerance.
    pub off_domain_vertices: usize,
    /// Vertices with non-finite coordinates.
    pub non_finite_vertices: usize,
    /// Vertices referenced by no cell (representation note, not a defect).
    pub orphan_vertices: usize,
}

impl PlaneValidationReport {
    /// True when the diagram satisfies the strict subdivision contract.
    pub fn is_strictly_valid(&self) -> bool {
        self.degenerate_cells == 0
            && self.cells_with_duplicate_vertices == 0
            && self.cells_with_invalid_references == 0
            && self.invalid_vertex_references == 0
            && self.unpaired_interior_edges == 0
            && self.overused_edges == 0
            && self.misoriented_edges == 0
            && self.off_domain_vertices == 0
            && self.non_finite_vertices == 0
            && self.weld_map_issues == 0
            && self.euler_characteristic == 2
    }
}

/// Validate a planar Voronoi diagram against the strict subdivision contract.
///
/// Bounded diagrams: disk topology (`V - E + (F + 1) = 2`), interior edges
/// paired with opposite orientations, single-use edges only on the rect
/// boundary. Periodic diagrams: torus topology (`V - E + F = 0`), EVERY
/// edge paired — single-use edges are always defects and `boundary_edges`
/// stays zero.
pub fn validate_plane(diagram: &crate::PlanarVoronoi) -> PlaneValidationReport {
    use std::collections::HashMap;

    let rect = diagram.rect();
    let extent = rect.width().max(rect.height());
    // Cap at a quarter of the SHORT axis: for high-aspect rects the
    // extent-scaled tolerance would otherwise swallow the short axis
    // entirely, classifying every edge as "on" both short-axis walls and
    // blinding the unpaired-interior-edge and off-domain checks.
    let wall_eps =
        (crate::tolerances::PLANE_ON_WALL_EPS * extent).min(0.25 * rect.width().min(rect.height()));
    let num_cells = diagram.num_cells();
    let num_vertices = diagram.num_vertices();

    let weld_map = diagram.weld_map();
    let is_canonical = |i: usize| weld_map.map(|m| m[i] as usize == i).unwrap_or(true);

    let mut report = PlaneValidationReport {
        num_cells,
        canonical_cells: 0,
        welded_twin_cells: 0,
        weld_map_issues: 0,
        num_vertices,
        used_vertices: 0,
        num_edges: 0,
        boundary_edges: 0,
        euler_characteristic: 0,
        degenerate_cells: 0,
        cells_with_duplicate_vertices: 0,
        cells_with_invalid_references: 0,
        invalid_vertex_references: 0,
        unpaired_interior_edges: 0,
        overused_edges: 0,
        misoriented_edges: 0,
        off_domain_vertices: 0,
        non_finite_vertices: 0,
        orphan_vertices: 0,
    };

    for v in diagram.vertices() {
        if !v.x.is_finite() || !v.y.is_finite() {
            report.non_finite_vertices += 1;
        } else if v.x < rect.min.x - wall_eps
            || v.x > rect.max.x + wall_eps
            || v.y < rect.min.y - wall_eps
            || v.y > rect.max.y + wall_eps
        {
            report.off_domain_vertices += 1;
        }
    }

    // (undirected edge) -> (uses, orientation balance)
    let mut edge_uses: HashMap<(u32, u32), (u32, i32)> = HashMap::new();
    let mut vertex_used = vec![false; num_vertices];

    for i in 0..num_cells {
        if !is_canonical(i) {
            report.welded_twin_cells += 1;
            let canonical = weld_map.expect("twin implies weld map")[i] as usize;
            if !is_canonical(canonical) || diagram.cell(i) != diagram.cell(canonical) {
                report.weld_map_issues += 1;
            }
            continue;
        }
        report.canonical_cells += 1;

        let cell = diagram.cell(i);
        let mut invalid_refs = 0usize;
        for &v in cell {
            if (v as usize) < num_vertices {
                vertex_used[v as usize] = true;
            } else {
                invalid_refs += 1;
            }
        }
        if invalid_refs > 0 {
            report.cells_with_invalid_references += 1;
            report.invalid_vertex_references += invalid_refs;
            continue;
        }

        let mut distinct: Vec<u32> = cell.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() < cell.len() {
            report.cells_with_duplicate_vertices += 1;
        }
        if distinct.len() < 3 {
            report.degenerate_cells += 1;
        }

        for k in 0..cell.len() {
            let a = cell[k];
            let b = cell[(k + 1) % cell.len()];
            if a == b {
                continue;
            }
            let (key, dir) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
            let entry = edge_uses.entry(key).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += dir;
        }
    }

    report.used_vertices = vertex_used.iter().filter(|&&u| u).count();
    report.orphan_vertices = num_vertices - report.used_vertices;
    report.num_edges = edge_uses.len();

    let periodic = diagram.is_periodic();
    // A single-use edge must lie on the rect boundary: both endpoints within
    // tolerance of one common wall. (Bounded topology only; a torus has no
    // boundary, so on the periodic side every single-use edge is a defect.)
    let on_common_wall = |a: u32, b: u32| -> bool {
        let (pa, pb) = (diagram.vertex(a as usize), diagram.vertex(b as usize));
        let near = |u: f32, wall: f32| (u - wall).abs() <= wall_eps;
        (near(pa.x, rect.min.x) && near(pb.x, rect.min.x))
            || (near(pa.x, rect.max.x) && near(pb.x, rect.max.x))
            || (near(pa.y, rect.min.y) && near(pb.y, rect.min.y))
            || (near(pa.y, rect.max.y) && near(pb.y, rect.max.y))
    };

    for (&(a, b), &(uses, balance)) in &edge_uses {
        match uses {
            1 => {
                if !periodic && on_common_wall(a, b) {
                    report.boundary_edges += 1;
                } else {
                    report.unpaired_interior_edges += 1;
                }
            }
            2 => {
                if balance != 0 {
                    report.misoriented_edges += 1;
                }
            }
            _ => report.overused_edges += 1,
        }
    }

    // Disk: V - E + (F + 1) = 2 with the outer face. Torus: V - E + F = 0,
    // normalized here to the same "expected 2" scale (+2) so
    // is_strictly_valid keeps a single check.
    report.euler_characteristic = if periodic {
        report.used_vertices as i32 - report.num_edges as i32 + report.canonical_cells as i32 + 2
    } else {
        report.used_vertices as i32 - report.num_edges as i32 + (report.canonical_cells as i32 + 1)
    };

    report
}

#[cfg(test)]
mod verify_gate_tests {
    use super::*;
    use glam::Vec3;
    use std::sync::Mutex;

    // Serializes the process-global env-var mutation so the set/remove
    // window cannot leak into a parallel case in this test.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// A single triangular cell: its three edges are each used once, and the
    /// sphere has no boundary, so all three are unpaired interior edges —
    /// not strictly valid, with every vertex index in range.
    fn invalid_diagram() -> SphericalVoronoi {
        // One generator => one cell (num_cells mirrors generator count).
        SphericalVoronoi::from_raw_parts(
            vec![Vec3::new(0.0, 0.0, 1.0)],
            vec![
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            vec![crate::diagram::VoronoiCell::new(0, 3)],
            vec![0, 1, 2],
            None,
        )
    }

    #[test]
    fn verify_gate_errors_only_when_enabled_and_invalid() {
        let _guard = ENV_LOCK.lock().unwrap();
        let diagram = invalid_diagram();
        assert!(!validate(&diagram).is_strictly_valid());

        std::env::remove_var("S2_VORONOI_VERIFY");
        assert!(
            verify_sphere_if_enabled(&diagram).is_ok(),
            "disabled gate must not error even on an invalid diagram"
        );

        std::env::set_var("S2_VORONOI_VERIFY", "1");
        let res = verify_sphere_if_enabled(&diagram);
        std::env::remove_var("S2_VORONOI_VERIFY");
        let err = res.expect_err("enabled gate must error on an invalid diagram");
        match err {
            crate::VoronoiError::ComputationFailed(msg) => {
                assert!(msg.contains("S2_VORONOI_VERIFY"), "message: {msg}");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    fn fib_sphere(n: usize) -> Vec<[f32; 3]> {
        let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        (0..n)
            .map(|i| {
                let y = 1.0 - (i as f32 / (n as f32 - 1.0)) * 2.0;
                let r = (1.0 - y * y).max(0.0).sqrt();
                let theta = golden * i as f32;
                let v = Vec3::new(theta.cos() * r, y, theta.sin() * r).normalize();
                [v.x, v.y, v.z]
            })
            .collect()
    }

    /// Extract the effective-space arrays (`weld_map` is `None` for the diagrams
    /// used here, so the diagram *is* its own effective representation).
    fn effective_arrays(
        d: &SphericalVoronoi,
    ) -> (Vec<Vec3>, Vec<crate::diagram::VoronoiCell>, Vec<u32>) {
        let verts = d
            .vertices()
            .iter()
            .map(|v| Vec3::new(v.x, v.y, v.z))
            .collect();
        let cells = (0..d.num_cells())
            .map(|i| crate::diagram::VoronoiCell::new(d.cell_start(i), d.cell(i).len() as u16))
            .collect();
        (verts, cells, d.cell_indices_raw().to_vec())
    }

    /// The slice validator must reach the SAME verdict (and first error) as the
    /// canonical `verify_sphere_fast` it stands in for inside the re-clip repair.
    fn assert_agree(d: &SphericalVoronoi) {
        let (v, c, ci) = effective_arrays(d);
        let fast = verify_sphere_fast(d);
        let eff = verify_sphere_effective_strict(&v, &c, &ci);
        assert_eq!(fast, eff, "fast={fast:?} effective={eff:?}");
    }

    #[test]
    fn effective_strict_matches_fast() {
        // Valid: a real computed diagram (no coincident points => no weld map).
        let good = crate::compute(&fib_sphere(64)).expect("compute");
        assert!(
            verify_sphere_fast(&good).is_ok(),
            "compute output must be valid"
        );
        assert_agree(&good);

        // Invalid: three unpaired interior edges + degree-1 vertices.
        assert_agree(&invalid_diagram());

        // Invalid: a vertex repeated within one cell.
        assert_agree(&SphericalVoronoi::from_raw_parts(
            vec![Vec3::new(0.0, 0.0, 1.0)],
            vec![
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            vec![crate::diagram::VoronoiCell::new(0, 4)],
            vec![0, 1, 0, 2],
            None,
        ));

        // Invalid: an off-sphere vertex on an otherwise-valid diagram. (The
        // validator does not read generator positions, so a placeholder vec of
        // the right length suffices.)
        let (mut v, c, ci) = effective_arrays(&good);
        v[0] *= 2.0;
        let generators = vec![Vec3::new(0.0, 0.0, 1.0); c.len()];
        let corrupted = SphericalVoronoi::from_raw_parts(generators, v, c, ci, None);
        assert!(verify_sphere_fast(&corrupted).is_err());
        assert_agree(&corrupted);
    }
}
