//! Strict validation for spherical Voronoi diagrams.
//!
//! This module checks whether a diagram is a coherent S2 subdivision and whether
//! exact representation invariants hold. It intentionally does not try to score
//! approximate Voronoi fidelity or generic-position heuristics.

use crate::SphericalVoronoi;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::tolerances::{ANTIPODAL_DOT_EPS, VERTEX_ON_SPHERE_EPS};

#[inline]
fn vertex_is_on_sphere(x: f32, y: f32, z: f32) -> bool {
    let len_sq = x * x + y * y + z * z;
    len_sq.is_finite() && (len_sq - 1.0).abs() <= VERTEX_ON_SPHERE_EPS
}

/// Detailed validation report for a spherical Voronoi diagram.
///
/// Stability: [`ValidationReport::is_strictly_valid`] is the authoritative
/// verdict and the stable contract. The individual counters expose the
/// current defect taxonomy for telemetry and tests; new fields may be added
/// in minor releases (the struct is `#[non_exhaustive]`). Human-readable
/// output ([`ValidationReport::headline`] and issue strings) is diagnostic
/// only — its wording is not stable.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// Number of canonical cells whose referenced vertices occupy fewer than
    /// three distinct exact stored directions.
    ///
    /// This is representation telemetry rather than an abstract-topology
    /// defect: `Preserve` may intentionally retain such geometry when fixed
    /// output precision cannot represent every effective cell injectively.
    pub cells_with_fewer_than_three_stored_positions: usize,
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
    /// Number of undirected edges whose distinct endpoint ids have exactly
    /// equal stored directions. This is representation telemetry rather than
    /// an abstract-topology defect.
    pub zero_length_edges: usize,
    /// Number of edges that are exactly antipodal or whose near-pi arc cannot
    /// be reconciled with the bisector plane of its two owning generators.
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
        if self.zero_length_edges > 0 {
            notes.push(format!("{} zero-length edges", self.zero_length_edges));
        }
        if self.cells_with_fewer_than_three_stored_positions > 0 {
            notes.push(format!(
                "{} cells with fewer than three stored positions",
                self.cells_with_fewer_than_three_stored_positions
            ));
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

/// Validate whether a simplified spherical cell mesh is a connected,
/// oriented, closed S2 subdivision with dense geometry and coherent source
/// provenance.
///
/// This checks generic mesh structure only. It deliberately makes no Voronoi,
/// nearest-site, or Delaunay claim about the mesh's source sites.
pub fn validate_cell_mesh(mesh: &crate::SphericalCellMesh) -> crate::CellMeshValidationReport {
    mesh.validate()
}

/// Opt-in post-build verification gate (env `VORONOI_MESH_VERIFY=1`).
///
/// The full topological validator is O(E) and is skipped by the plain
/// `compute` fast path for speed (the report-returning
/// entry points already validate unconditionally). With the net's repair
/// scans gated on detection records, a defect that left no record would
/// ship silently on those paths. Enabling this runs the validator after
/// every build and turns any strict-validity failure into an error —
/// belt-and-braces for callers who want output validity machine-checked
/// regardless of cost.
pub(crate) fn verify_enabled() -> bool {
    matches!(std::env::var("VORONOI_MESH_VERIFY"), Ok(v) if v == "1")
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
            if matches!(std::env::var("VORONOI_MESH_VERIFY_TRACE"), Ok(v) if v == "1") {
                eprintln!("VORONOI_MESH_VERIFY fast path fell back: {reason}");
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
        "VORONOI_MESH_VERIFY: returned diagram failed strict validation: {}",
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

#[inline]
fn edge_vertices(key: u64) -> (usize, usize) {
    ((key >> 32) as usize, key as u32 as usize)
}

#[inline]
fn owner_arc_class(
    start: glam::Vec3,
    end: glam::Vec3,
    owner: glam::Vec3,
    neighbor: glam::Vec3,
) -> crate::spherical_arc::OwnerArcClass {
    crate::spherical_arc::classify_owner_arc(start, end, owner, neighbor, ANTIPODAL_DOT_EPS)
}

/// Sort edge-use records by key — in parallel when available. This sort is the
/// dominant cost of the strict verifiers at scale (~6M records at 1M cells).
/// The downstream pairing scan only groups records by key and applies
/// order-symmetric checks within a group (`len == 2`, opposite `forward`,
/// commutative DSU union), so an unstable parallel sort is verdict-equivalent
/// to the sequential one.
fn sort_edge_uses(edge_uses: &mut [EdgeUse]) {
    #[cfg(feature = "parallel")]
    {
        use rayon::slice::ParallelSliceMut;
        edge_uses.par_sort_unstable_by_key(|edge| edge.key);
    }
    #[cfg(not(feature = "parallel"))]
    edge_uses.sort_unstable_by_key(|edge| edge.key);
}

/// Fast success-path verifier for `VORONOI_MESH_VERIFY`.
///
/// This checks the same strict sphere contract as [`validate_impl`], but does
/// not build the detailed diagnostic report. On failure the caller reruns the
/// full validator to produce the public error message.
fn verify_sphere_fast(diagram: &SphericalVoronoi) -> Result<(), &'static str> {
    let num_cells = diagram.num_cells();
    let num_vertices = diagram.num_vertices();

    if diagram
        .vertices()
        .iter()
        .any(|v| !vertex_is_on_sphere(v.x(), v.y(), v.z()))
    {
        return Err("off-sphere vertex");
    }

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

    sort_edge_uses(&mut edge_uses);

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
        let (a, b) = edge_vertices(first.key);
        let va = diagram.vertex(a);
        let vb = diagram.vertex(b);
        let dot = va.dot(vb);
        if dot <= -1.0 + ANTIPODAL_DOT_EPS {
            let owner = diagram.generator(group[0].cell as usize);
            let neighbor = diagram.generator(group[1].cell as usize);
            let class = owner_arc_class(
                glam::Vec3::from_array(va.to_array()),
                glam::Vec3::from_array(vb.to_array()),
                glam::Vec3::from_array(owner.to_array()),
                glam::Vec3::from_array(neighbor.to_array()),
            );
            if matches!(
                class,
                crate::spherical_arc::OwnerArcClass::ExactPi
                    | crate::spherical_arc::OwnerArcClass::Invalid
            ) {
                return Err("antipodal edge");
            }
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

/// One chunk's cell-scan output: edge-use records, cell signatures for the
/// cross-chunk duplicate-cell pass, and the chunk's lexicographically first
/// error as `(cell, check_rank, message)`.
struct CellScan {
    edge_uses: Vec<EdgeUse>,
    signatures: Vec<(CellSignature, u32)>,
    err: Option<(u32, u8, &'static str)>,
}

/// Check ranks mirror the sequential validator's within-cell check order, so
/// the lexicographic minimum over `(cell, rank)` reproduces the sequential
/// first error exactly: span(0) → vertex-ref/duplicate-vertex(1) →
/// degenerate(2) → duplicate-cell(3, needs cross-cell info) →
/// self-loop/arc(4).
const RANK_SPAN: u8 = 0;
const RANK_VERTEX: u8 = 1;
const RANK_DEGENERATE: u8 = 2;
const RANK_DUP_CELL: u8 = 3;
const RANK_EDGE: u8 = 4;

/// Scan `range` of cells: per-cell structural checks, exact vertex-incidence
/// counting into the shared atomics, and edge-use/signature collection. Stops
/// at the first erroring cell — later cells in the chunk can only produce
/// lexicographically larger errors, and dropped signatures/edge-uses are
/// irrelevant once any error exists (see the duplicate-cell argument on
/// `verify_sphere_effective_strict`).
fn scan_cells_strict(
    vertices: &[glam::Vec3],
    cells: &[crate::diagram::VoronoiCell],
    cell_indices: &[u32],
    range: std::ops::Range<usize>,
    vertex_cell_count: &[std::sync::atomic::AtomicU32],
) -> CellScan {
    use std::sync::atomic::Ordering::Relaxed;
    let num_vertices = vertices.len();
    let mut out = CellScan {
        edge_uses: Vec::new(),
        signatures: Vec::with_capacity(range.len()),
        err: None,
    };
    'cells: for ci in range {
        let cell = &cells[ci];
        let start = cell.vertex_start();
        let len = cell.vertex_count();
        let Some(span) = len
            .checked_add(start)
            .and_then(|end| cell_indices.get(start..end))
        else {
            out.err = Some((ci as u32, RANK_SPAN, "invalid cell span"));
            break 'cells;
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
                out.err = Some((ci as u32, RANK_VERTEX, "invalid vertex reference"));
                break 'cells;
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
                out.err = Some((ci as u32, RANK_VERTEX, "duplicate vertex in cell"));
                break 'cells;
            }
            vertex_cell_count[vi as usize].fetch_add(1, Relaxed);
        }

        let seen_valid_len = if use_spill {
            seen_spill.len()
        } else {
            seen_stack_len
        };
        if seen_valid_len < 3 {
            out.err = Some((ci as u32, RANK_DEGENERATE, "degenerate cell"));
            break 'cells;
        }

        // Signature emission mirrors the sequential insert point: after the
        // rank-0..2 checks, before the edge checks — a cell that fails an edge
        // check still participates in duplicate-cell detection.
        let signature = if use_spill {
            cell_signature(&seen_spill)
        } else {
            cell_signature(&seen_stack[..seen_stack_len])
        };
        if let Some(signature) = signature {
            out.signatures.push((signature, ci as u32));
        }

        for edge_idx in 0..len {
            let a = span[edge_idx];
            let b = span[(edge_idx + 1) % len];
            if a == b {
                out.err = Some((ci as u32, RANK_EDGE, "self-loop edge"));
                break 'cells;
            }
            let (lo, hi, forward) = if a < b { (a, b, true) } else { (b, a, false) };
            out.edge_uses.push(EdgeUse {
                key: edge_key(lo, hi),
                forward,
                cell: ci as u32,
            });
        }
    }
    out
}

/// Strict validation of effective arrays in place — the repair acceptance gate
/// (also the plain-path contract check), sharing the strict contract of
/// `verify_sphere_fast` without cloning the diagram into a `SphericalVoronoi`.
/// Effective index space has no welded twins, so every cell is its own face
/// (`num_faces == num_cells`).
///
/// The cell scan runs in parallel chunks (feature `parallel`); the verdict and
/// the reported first error are IDENTICAL to the sequential scan. Per-cell
/// checks are pure; vertex incidence uses exact shared atomics (read only
/// after the scan); and the first error is the lexicographic minimum over
/// `(cell, check_rank)`, which equals the sequential scan's first return.
/// Duplicate-cell detection moves to a sort-based pass over the emitted
/// signatures: a chunk stops emitting after its first error, but every
/// dropped signature belongs to a cell past that error, so any duplicate-cell
/// error it could have influenced is lexicographically dominated — the
/// minimum is unchanged. Pinned to `verify_sphere_fast` (kept sequential as
/// the independent reference) by the differential test
/// `effective_strict_matches_fast`.
pub(crate) fn verify_sphere_effective_strict(
    generators: &[glam::Vec3],
    vertices: &[glam::Vec3],
    cells: &[crate::diagram::VoronoiCell],
    cell_indices: &[u32],
) -> Result<(), &'static str> {
    use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
    let num_cells = cells.len();
    let num_vertices = vertices.len();

    if generators.len() != num_cells {
        return Err("generator/cell count mismatch");
    }

    if vertices.iter().any(|v| !vertex_is_on_sphere(v.x, v.y, v.z)) {
        return Err("off-sphere vertex");
    }

    // Exact incidence counters, shared across chunks; only read after the
    // scan completes. (u32: cannot saturate — total increments are bounded by
    // `cell_indices.len()`.)
    let vertex_cell_count: Vec<AtomicU32> = (0..num_vertices).map(|_| AtomicU32::new(0)).collect();

    #[cfg(feature = "parallel")]
    let scans: Vec<CellScan> = {
        use rayon::prelude::*;
        let chunk = num_cells
            .div_ceil(rayon::current_num_threads().max(1) * 4)
            .max(1024);
        (0..num_cells.div_ceil(chunk).max(1))
            .into_par_iter()
            .map(|i| {
                let lo = i * chunk;
                let hi = ((i + 1) * chunk).min(num_cells);
                scan_cells_strict(vertices, cells, cell_indices, lo..hi, &vertex_cell_count)
            })
            .collect()
    };
    #[cfg(not(feature = "parallel"))]
    let scans: Vec<CellScan> = vec![scan_cells_strict(
        vertices,
        cells,
        cell_indices,
        0..num_cells,
        &vertex_cell_count,
    )];

    // Lexicographic-first error across chunks (chunks are disjoint ascending
    // cell ranges, so per-chunk firsts merge by (cell, rank)).
    let mut first_err: Option<(u32, u8, &'static str)> = None;
    let lower = |cand: (u32, u8, &'static str), cur: &mut Option<(u32, u8, &'static str)>| {
        if cur.is_none_or(|c| (cand.0, cand.1) < (c.0, c.1)) {
            *cur = Some(cand);
        }
    };
    for scan in &scans {
        if let Some(e) = scan.err {
            lower(e, &mut first_err);
        }
    }

    // Duplicate-cell pass over the emitted signatures: sort by (signature,
    // cell), then each equal-signature run's SECOND cell is where the
    // sequential scan would have reported the duplicate.
    let mut signatures: Vec<(CellSignature, u32)> = Vec::with_capacity(num_cells);
    for scan in &scans {
        signatures.extend_from_slice(&scan.signatures);
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::slice::ParallelSliceMut;
        signatures.par_sort_unstable();
    }
    #[cfg(not(feature = "parallel"))]
    signatures.sort_unstable();
    for pair in signatures.windows(2) {
        if pair[0].0 == pair[1].0 {
            lower((pair[1].1, RANK_DUP_CELL, "duplicate cell"), &mut first_err);
            // Runs are cell-ascending, so the first adjacent duplicate in a
            // run is that run's minimal candidate; keep scanning other runs.
        }
    }
    if let Some((_, _, msg)) = first_err {
        return Err(msg);
    }

    let mut used_vertices = 0usize;
    for count in &vertex_cell_count {
        let count = count.load(Relaxed);
        if count > 0 {
            used_vertices += 1;
            if count < 3 {
                return Err("low-incidence vertex");
            }
        }
    }

    let mut edge_uses: Vec<EdgeUse> = Vec::with_capacity(cell_indices.len());
    for scan in &scans {
        edge_uses.extend_from_slice(&scan.edge_uses);
    }
    sort_edge_uses(&mut edge_uses);

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
        let (a, b) = edge_vertices(first.key);
        let va = vertices[a];
        let vb = vertices[b];
        if va.dot(vb) <= -1.0 + ANTIPODAL_DOT_EPS {
            let class = owner_arc_class(
                va,
                vb,
                generators[group[0].cell as usize],
                generators[group[1].cell as usize],
            );
            if matches!(
                class,
                crate::spherical_arc::OwnerArcClass::ExactPi
                    | crate::spherical_arc::OwnerArcClass::Invalid
            ) {
                return Err("antipodal edge");
            }
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
    let mut cells_with_fewer_than_three_stored_positions = 0usize;
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
        let mut distinct_positions = [None; 3];
        let mut distinct_position_count = 0usize;

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

            let position = vertices[vi as usize];
            if distinct_position_count < 3
                && !distinct_positions[..distinct_position_count].contains(&Some(position))
            {
                distinct_positions[distinct_position_count] = Some(position);
                distinct_position_count += 1;
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
        if distinct_position_count < 3 {
            cells_with_fewer_than_three_stored_positions += 1;
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
    let mut zero_length_edges = 0usize;
    let num_edges = edges.len();

    for (&key, stat) in &edges {
        let (a, b) = edge_vertices(key);
        let va = diagram.vertex(a);
        let vb = diagram.vertex(b);
        if va == vb {
            zero_length_edges += 1;
        }
        if stat.cells.len() != 2 {
            continue;
        }
        let dot = va.dot(vb);
        if dot > -1.0 + ANTIPODAL_DOT_EPS {
            continue;
        }
        let owner = diagram.generator(stat.cells[0]);
        let neighbor = diagram.generator(stat.cells[1]);
        let class = owner_arc_class(
            glam::Vec3::from_array(va.to_array()),
            glam::Vec3::from_array(vb.to_array()),
            glam::Vec3::from_array(owner.to_array()),
            glam::Vec3::from_array(neighbor.to_array()),
        );
        if matches!(
            class,
            crate::spherical_arc::OwnerArcClass::ExactPi
                | crate::spherical_arc::OwnerArcClass::Invalid
        ) {
            antipodal_edges += 1;
        }
    }

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
        if !vertex_is_on_sphere(v.x(), v.y(), v.z()) {
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
        cells_with_fewer_than_three_stored_positions,
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
        zero_length_edges,
        antipodal_edges,
        boundary_edges,
        overused_edges,
        same_direction_edge_pairs,
    }
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
    fn report_detects_two_position_cycle_without_an_adjacent_zero_edge() {
        let diagram = SphericalVoronoi::from_raw_parts(
            vec![Vec3::new(0.0, 0.0, 1.0)],
            vec![
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![crate::diagram::VoronoiCell::new(0, 4)],
            vec![0, 1, 2, 3],
            None,
        );
        let report = validate(&diagram);
        assert_eq!(report.zero_length_edges, 0);
        assert_eq!(report.cells_with_fewer_than_three_stored_positions, 1);
        assert!(report
            .representation_notes()
            .iter()
            .any(|note| note.contains("fewer than three stored positions")));
    }

    #[test]
    fn verify_gate_errors_only_when_enabled_and_invalid() {
        let _guard = ENV_LOCK.lock().unwrap();
        let diagram = invalid_diagram();
        assert!(!validate(&diagram).is_strictly_valid());

        std::env::remove_var("VORONOI_MESH_VERIFY");
        assert!(
            verify_sphere_if_enabled(&diagram).is_ok(),
            "disabled gate must not error even on an invalid diagram"
        );

        std::env::set_var("VORONOI_MESH_VERIFY", "1");
        let res = verify_sphere_if_enabled(&diagram);
        std::env::remove_var("VORONOI_MESH_VERIFY");
        let err = res.expect_err("enabled gate must error on an invalid diagram");
        match err {
            crate::VoronoiError::ComputationFailed(msg) => {
                assert!(msg.contains("VORONOI_MESH_VERIFY"), "message: {msg}");
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
            .map(|v| Vec3::from_array(v.to_array()))
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
        let generators: Vec<Vec3> = d
            .generators()
            .iter()
            .map(|g| Vec3::from_array(g.to_array()))
            .collect();
        let fast = verify_sphere_fast(d);
        let eff = verify_sphere_effective_strict(&generators, &v, &c, &ci);
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

        // Invalid: two cells with identical boundaries (duplicate cell) — pins
        // the sort-based duplicate-signature pass against the incremental
        // hashset the sequential reference uses.
        let (v, mut c, mut ci) = effective_arrays(&good);
        let first = c[0];
        ci.extend_from_within(first.vertex_start()..first.vertex_start() + first.vertex_count());
        c.push(crate::diagram::VoronoiCell::new(
            (ci.len() - first.vertex_count()) as u32,
            first.vertex_count() as u16,
        ));
        let generators = vec![Vec3::new(0.0, 0.0, 1.0); c.len()];
        let dup_cell = SphericalVoronoi::from_raw_parts(generators, v, c, ci, None);
        assert!(verify_sphere_fast(&dup_cell).is_err());
        assert_agree(&dup_cell);

        // Raw effective arrays can still be rejected before they become a
        // diagram. A checked SpherePoint-backed SphericalVoronoi cannot safely
        // contain these coordinates, so there is intentionally no
        // verify_sphere_fast comparison for these cases.
        let (mut v, c, ci) = effective_arrays(&good);
        v[0] *= 2.0;
        let generators = vec![Vec3::new(0.0, 0.0, 1.0); c.len()];
        assert_eq!(
            verify_sphere_effective_strict(&generators, &v, &c, &ci),
            Err("off-sphere vertex")
        );

        // Invalid: non-finite coordinates must not exploit NaN comparison
        // semantics in the effective-array gate.
        for coordinate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let (mut v, c, ci) = effective_arrays(&good);
            v[0].x = coordinate;
            let generators = vec![Vec3::new(0.0, 0.0, 1.0); c.len()];
            assert_eq!(
                verify_sphere_effective_strict(&generators, &v, &c, &ci),
                Err("off-sphere vertex")
            );
        }
    }
}
