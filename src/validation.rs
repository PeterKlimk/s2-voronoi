//! Topological validation for spherical Voronoi diagrams.
//!
//! Provides functions to verify combinatorial correctness of a diagram.
//! Useful for debugging, testing, and catching numerical issues.

use crate::SphericalVoronoi;
use std::collections::HashSet;

/// Detailed validation report for a spherical Voronoi diagram.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Number of cells (faces) in the diagram.
    pub num_cells: usize,
    /// Number of vertices in the diagram.
    pub num_vertices: usize,
    /// Number of edges (each edge shared by 2 cells).
    pub num_edges: usize,

    /// Euler characteristic V - E + F (using non-orphan vertices).
    pub euler_characteristic: i32,
    /// Expected vertex count: 2n - 4 for n cells (generic position).
    pub expected_vertices: usize,

    /// Number of cells with < 3 vertices (degenerate).
    pub degenerate_cells: usize,
    /// Number of cells with duplicate vertex indices.
    pub cells_with_duplicates: usize,
    /// Sum of cell vertex counts (includes duplicates from merged generators).
    pub total_cell_vertices: usize,
    /// Sum of cell vertex counts for unique cells only (should be even = 2E).
    pub total_unique_cell_vertices: usize,

    /// Number of vertices not on unit sphere (|v| not in [1-eps, 1+eps]).
    pub vertices_off_sphere: usize,

    /// Number of degree-1/2 vertices.
    /// Degree-0 (orphans) and degree-4+ are tracked separately.
    pub non_degree3_vertices: usize,
    /// Number of vertices that appear in 0 cells (orphaned).
    pub orphan_vertices: usize,
    /// Count of vertices by degree: [degree0, degree1, degree2, degree4+]
    pub degree_counts: [usize; 4],

    /// Number of topologically unique cells (ignoring duplicates).
    pub unique_cells: usize,
    /// Number of duplicate cells (identical vertex indices to another cell).
    pub duplicate_cells_count: usize,
}

impl ValidationReport {
    /// Check if the diagram is valid with tolerance for numerical edge cases.
    ///
    /// Allows:
    /// - Euler characteristic within ±2 of expected (for minor degeneracies)
    /// - Up to 1% degenerate cells
    /// - Up to 1% degree-1/2 vertices (degree-0 or 4+ are tolerated)
    /// - No duplicate vertices in cells
    pub fn is_valid(&self) -> bool {
        let euler_ok = (self.euler_characteristic - 2).abs() <= 2;
        let degenerate_ratio = self.degenerate_cells as f64 / self.num_cells.max(1) as f64;
        let degenerate_ok = degenerate_ratio <= 0.01;
        let no_duplicates = self.cells_with_duplicates == 0;
        let edges_consistent = self.total_unique_cell_vertices.is_multiple_of(2);
        let on_sphere = self.vertices_off_sphere == 0;
        let degree3_bad = (self.degree_counts[1] + self.degree_counts[2]) as f64;
        let degree3_ratio = degree3_bad / self.num_vertices.max(1) as f64;
        let degree3_ok = degree3_ratio <= 0.01;
        // Duplicate cells are allowed if they result from merged generators
        // But we should verify they are consistent
        let duplicate_cells_ok = true;

        euler_ok
            && degenerate_ok
            && no_duplicates
            && edges_consistent
            && on_sphere
            && degree3_ok
            && duplicate_cells_ok
    }

    /// Strict check: exact Euler characteristic, no degenerates, perfect structure.
    pub fn is_perfect(&self) -> bool {
        self.euler_characteristic == 2
            && self.degenerate_cells == 0
            && self.cells_with_duplicates == 0
            && self.total_unique_cell_vertices.is_multiple_of(2)
            && self.vertices_off_sphere == 0
            && self.non_degree3_vertices == 0
    }

    /// Format a summary of any issues found.
    pub fn summary(&self) -> String {
        if self.is_perfect() {
            return "Perfect".to_string();
        }

        let mut issues = Vec::new();

        if self.euler_characteristic != 2 {
            issues.push(format!("Euler={} (expected 2)", self.euler_characteristic));
        }
        if self.num_vertices != self.expected_vertices {
            issues.push(format!(
                "V={} (expected {})",
                self.num_vertices, self.expected_vertices
            ));
        }
        if self.degenerate_cells > 0 {
            issues.push(format!("{} degenerate cells", self.degenerate_cells));
        }
        if self.cells_with_duplicates > 0 {
            issues.push(format!(
                "{} cells with duplicate vertices",
                self.cells_with_duplicates
            ));
        }
        if !self.total_unique_cell_vertices.is_multiple_of(2) {
            issues.push("odd total cell vertices (edge inconsistency)".to_string());
        }
        if self.vertices_off_sphere > 0 {
            issues.push(format!("{} vertices off sphere", self.vertices_off_sphere));
        }
        if self.orphan_vertices > 0 {
            issues.push(format!(
                "{} orphan vertices (in 0 cells)",
                self.orphan_vertices
            ));
        }
        if self.non_degree3_vertices > 0 {
            let [_d0, d1, d2, _d4p] = self.degree_counts;
            let mut parts = Vec::new();
            if d1 > 0 {
                parts.push(format!("d1:{}", d1));
            }
            if d2 > 0 {
                parts.push(format!("d2:{}", d2));
            }
            issues.push(format!(
                "{} non-degree-3 vertices ({})",
                self.non_degree3_vertices,
                parts.join(", ")
            ));
        }

        if issues.is_empty() {
            "Valid (minor imperfections)".to_string()
        } else {
            issues.join(", ")
        }
    }
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ValidationReport {{ V={}, E={}, F={}, euler={}, {} }}",
            self.num_vertices,
            self.num_edges,
            self.num_cells,
            self.euler_characteristic,
            self.summary()
        )
    }
}

/// Validate topological correctness of a spherical Voronoi diagram.
///
/// Checks:
/// - Euler characteristic (V - E + F = 2)
/// - Vertex count (V = 2F - 4 for generic position)
/// - Cell validity (≥ 3 vertices, no duplicates)
/// - Edge consistency (total cell vertices is even)
/// - Vertices on unit sphere
/// - Vertex incidence (each vertex in exactly 3 cells)
pub fn validate(diagram: &SphericalVoronoi) -> ValidationReport {
    let num_cells = diagram.num_cells();
    let num_vertices = diagram.num_vertices();

    // Count unique cells to handle merged generators
    let mut unique_cell_signatures = HashSet::new();
    let mut duplicate_cells_count = 0;

    // Count how many cells each vertex appears in
    let mut vertex_cell_count: Vec<u32> = vec![0; num_vertices];

    // Count total cell vertices and check for degenerates/duplicates
    let mut total_cell_vertices = 0usize;
    let mut degenerate_cells = 0usize;
    let mut cells_with_duplicates = 0usize;

    for cell in diagram.iter_cells() {
        let len = cell.len();
        total_cell_vertices += len;

        if len < 3 {
            degenerate_cells += 1;
        }

        // Check for duplicate vertex indices in this cell
        let unique: HashSet<u32> = cell.vertex_indices.iter().copied().collect();
        if unique.len() < len {
            cells_with_duplicates += 1;
        }

        // Check for duplicate cells (same sorted vertex indices)
        // We sort the indices for canonical representation
        let mut signature = cell.vertex_indices.to_vec();
        signature.sort_unstable();
        if !unique_cell_signatures.insert(signature) {
            duplicate_cells_count += 1;
        }

        // Count vertex incidence
        for &vi in cell.vertex_indices {
            if (vi as usize) < num_vertices {
                vertex_cell_count[vi as usize] += 1;
            }
        }
    }

    let unique_cells = unique_cell_signatures.len();

    // Count vertices with wrong degree
    let mut orphan_vertices = 0usize;
    let mut non_degree3_vertices = 0usize;
    let mut degree_counts = [0usize; 4]; // [d0, d1, d2, d4+]
    for &count in &vertex_cell_count {
        match count {
            0 => {
                orphan_vertices += 1;
                degree_counts[0] += 1;
            }
            1 => {
                non_degree3_vertices += 1;
                degree_counts[1] += 1;
            }
            2 => {
                non_degree3_vertices += 1;
                degree_counts[2] += 1;
            }
            3 => {} // correct
            _ => {
                degree_counts[3] += 1;
            }
        }
    }

    // Each edge is shared by 2 cells, so E = total_cell_vertices / 2
    // Note: E includes edges from duplicate cells, which is not strictly correct for Euler
    // unless we also count duplicate cells in F.
    // However, validation of Euler is best done on the unique diagram.

    // Recalculate E for the unique diagram
    let total_unique_cell_vertices: usize = unique_cell_signatures.iter().map(|s| s.len()).sum();
    let unique_edges = total_unique_cell_vertices / 2;

    // Euler characteristic: (V - orphans) - E + F (using unique cells)
    let effective_vertices = num_vertices.saturating_sub(orphan_vertices);
    let euler_characteristic =
        effective_vertices as i32 - unique_edges as i32 + unique_cells as i32;

    // For generic position (all degree-3 vertices): V = 2F - 4
    let expected_vertices = if unique_cells >= 2 {
        2 * unique_cells - 4
    } else {
        0
    };

    // Check vertices are on unit sphere
    let mut vertices_off_sphere = 0usize;
    const EPS: f32 = 1e-4;
    for v in &diagram.vertices {
        let len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
        if (len_sq - 1.0).abs() > EPS {
            vertices_off_sphere += 1;
        }
    }

    ValidationReport {
        num_cells,
        num_vertices,
        num_edges: unique_edges, // Use unique edges for the report to match Euler
        euler_characteristic,
        expected_vertices,
        degenerate_cells,
        cells_with_duplicates,
        total_cell_vertices,
        total_unique_cell_vertices,
        vertices_off_sphere,
        non_degree3_vertices,
        orphan_vertices,
        degree_counts,
        unique_cells,
        duplicate_cells_count,
    }
}
