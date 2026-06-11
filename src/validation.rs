//! Strict validation for spherical Voronoi diagrams.
//!
//! This module checks whether a diagram is a coherent S2 subdivision and whether
//! exact representation invariants hold. It intentionally does not try to score
//! approximate Voronoi fidelity or generic-position heuristics.

use crate::SphericalVoronoi;
use std::collections::{HashMap, HashSet};

const VERTEX_ON_SPHERE_EPS: f32 = 1e-4;
const ANTIPODAL_DOT_EPS: f32 = 1e-5;

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

    let mut unique_cell_signatures = HashSet::new();
    let mut duplicate_cells_count = 0usize;
    let mut vertex_cell_count: Vec<u32> = vec![0; num_vertices];

    let mut total_cell_vertices = 0usize;
    let mut degenerate_cells = 0usize;
    let mut cells_with_duplicate_vertices = 0usize;
    let mut cells_with_invalid_references = 0usize;
    let mut invalid_vertex_references = 0usize;
    let mut self_loop_edges = 0usize;
    let mut antipodal_edges = 0usize;

    let mut edges: HashMap<(u32, u32), EdgeStat> = HashMap::new();

    for cell in diagram.iter_cells() {
        if is_welded_twin[cell.generator_index] {
            continue;
        }
        let len = cell.len();
        total_cell_vertices += len;

        let mut seen_valid = HashSet::new();
        let mut cell_has_duplicate_vertices = false;
        let mut cell_has_invalid_reference = false;

        for &vi in cell.vertex_indices {
            if (vi as usize) >= num_vertices {
                invalid_vertex_references += 1;
                cell_has_invalid_reference = true;
                continue;
            }
            if !seen_valid.insert(vi) {
                cell_has_duplicate_vertices = true;
            }
        }

        if cell_has_duplicate_vertices {
            cells_with_duplicate_vertices += 1;
        }
        if cell_has_invalid_reference {
            cells_with_invalid_references += 1;
        }
        if seen_valid.len() < 3 {
            degenerate_cells += 1;
        }

        // Canonical duplicate-cell signature over valid references only.
        let mut signature: Vec<u32> = seen_valid.iter().copied().collect();
        signature.sort_unstable();
        if !signature.is_empty() {
            if !unique_cell_signatures.insert(signature) {
                duplicate_cells_count += 1;
            }
        }

        for &vi in &seen_valid {
            vertex_cell_count[vi as usize] += 1;
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
            let stat = edges.entry((lo, hi)).or_default();
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

        let mut unique_cells_for_edge = stat.cells.clone();
        unique_cells_for_edge.sort_unstable();
        unique_cells_for_edge.dedup();
        if let Some((&first, rest)) = unique_cells_for_edge.split_first() {
            for &other in rest {
                dsu.union(first, other);
            }
        }
    }

    let connected_components = if num_faces == 0 {
        0
    } else {
        let mut roots = HashSet::with_capacity(num_faces);
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
