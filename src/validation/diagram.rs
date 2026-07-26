use rustc_hash::{FxHashMap, FxHashSet};

use super::{
    cell_signature, classify_edge_uses, edge_key, edge_vertices, owner_arc_class,
    vertex_is_on_sphere, CellSignature, DisjointSet, EdgeStat, EdgeUseClass, ValidationReport,
    ANTIPODAL_DOT_EPS,
};
use crate::SphericalVoronoi;

struct WeldAudit {
    is_welded_twin: Vec<bool>,
    welded_twin_cells: usize,
    weld_map_issues: usize,
}

fn audit_weld_map(diagram: &SphericalVoronoi) -> WeldAudit {
    let num_cells = diagram.num_cells();
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
    WeldAudit {
        is_welded_twin,
        welded_twin_cells,
        weld_map_issues,
    }
}

struct CellScan {
    edges: FxHashMap<u64, EdgeStat>,
    vertex_cell_count: Vec<u32>,
    total_cell_vertices: usize,
    degenerate_cells: usize,
    cells_with_fewer_than_three_stored_positions: usize,
    cells_with_duplicate_vertices: usize,
    cells_with_invalid_references: usize,
    invalid_vertex_references: usize,
    duplicate_cells_count: usize,
    unique_cells: usize,
    self_loop_edges: usize,
}

fn scan_cells(diagram: &SphericalVoronoi, weld: &WeldAudit, num_faces: usize) -> CellScan {
    let num_vertices = diagram.num_vertices();
    let vertices = diagram.vertices();
    let estimated_directed_edges = diagram.cell_indices_raw().len();
    let estimated_undirected_edges = (estimated_directed_edges / 2).max(1);

    let mut unique_cell_signatures: FxHashSet<CellSignature> =
        FxHashSet::with_capacity_and_hasher(num_faces.max(1), Default::default());
    let mut duplicate_cells_count = 0usize;
    let mut vertex_cell_count = vec![0u32; num_vertices];
    let mut total_cell_vertices = 0usize;
    let mut degenerate_cells = 0usize;
    let mut cells_with_fewer_than_three_stored_positions = 0usize;
    let mut cells_with_duplicate_vertices = 0usize;
    let mut cells_with_invalid_references = 0usize;
    let mut invalid_vertex_references = 0usize;
    let mut self_loop_edges = 0usize;
    let mut edges: FxHashMap<u64, EdgeStat> =
        FxHashMap::with_capacity_and_hasher(estimated_undirected_edges, Default::default());

    for cell in diagram.iter_cells() {
        if weld.is_welded_twin[cell.generator_index] {
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

        cells_with_duplicate_vertices += usize::from(cell_has_duplicate_vertices);
        cells_with_invalid_references += usize::from(cell_has_invalid_reference);
        let seen_valid_len = if use_spill {
            seen_spill.len()
        } else {
            seen_stack_len
        };
        degenerate_cells += usize::from(seen_valid_len < 3);
        cells_with_fewer_than_three_stored_positions += usize::from(distinct_position_count < 3);

        // Canonical duplicate-cell signature over valid references only.
        let signature = if use_spill {
            cell_signature(&seen_spill)
        } else {
            cell_signature(&seen_stack[..seen_stack_len])
        };
        if let Some(signature) = signature {
            duplicate_cells_count += usize::from(!unique_cell_signatures.insert(signature));
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

    CellScan {
        edges,
        vertex_cell_count,
        total_cell_vertices,
        degenerate_cells,
        cells_with_fewer_than_three_stored_positions,
        cells_with_duplicate_vertices,
        cells_with_invalid_references,
        invalid_vertex_references,
        duplicate_cells_count,
        unique_cells: unique_cell_signatures.len(),
        self_loop_edges,
    }
}

struct VertexIncidence {
    used_vertices: usize,
    orphan_vertices: usize,
    low_incidence_vertices: usize,
    degree_counts: [usize; 5],
}

fn analyze_vertex_incidence(vertex_cell_count: &[u32]) -> VertexIncidence {
    let mut orphan_vertices = 0usize;
    let mut low_incidence_vertices = 0usize;
    let mut degree_counts = [0usize; 5];
    for &count in vertex_cell_count {
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
    VertexIncidence {
        used_vertices: vertex_cell_count.len().saturating_sub(orphan_vertices),
        orphan_vertices,
        low_incidence_vertices,
        degree_counts,
    }
}

struct EdgeGeometry {
    zero_length_edges: usize,
    antipodal_edges: usize,
}

fn analyze_edge_geometry(
    diagram: &SphericalVoronoi,
    edges: &FxHashMap<u64, EdgeStat>,
) -> EdgeGeometry {
    let mut zero_length_edges = 0usize;
    let mut antipodal_edges = 0usize;
    for (&key, stat) in edges {
        let (a, b) = edge_vertices(key);
        let va = diagram.vertex(a);
        let vb = diagram.vertex(b);
        zero_length_edges += usize::from(va == vb);
        if stat.cells.len() != 2 || va.dot(vb) > -1.0 + ANTIPODAL_DOT_EPS {
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
        antipodal_edges += usize::from(matches!(
            class,
            crate::spherical_arc::OwnerArcClass::ExactPi
                | crate::spherical_arc::OwnerArcClass::Invalid
        ));
    }
    EdgeGeometry {
        zero_length_edges,
        antipodal_edges,
    }
}

struct EdgeTopology {
    boundary_edges: usize,
    overused_edges: usize,
    same_direction_edge_pairs: usize,
    connected_components: usize,
}

fn analyze_edge_topology(
    num_cells: usize,
    num_faces: usize,
    is_welded_twin: &[bool],
    edges: &FxHashMap<u64, EdgeStat>,
) -> EdgeTopology {
    let mut boundary_edges = 0usize;
    let mut overused_edges = 0usize;
    let mut same_direction_edge_pairs = 0usize;
    let mut dsu = DisjointSet::new(num_cells);
    for stat in edges.values() {
        let total = stat.forward + stat.reverse;
        match classify_edge_uses(total, stat.forward == 1 && stat.reverse == 1) {
            EdgeUseClass::Boundary => boundary_edges += 1,
            EdgeUseClass::Overused => overused_edges += 1,
            EdgeUseClass::SameDirection => same_direction_edge_pairs += 1,
            EdgeUseClass::Paired => {}
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
            if !is_twin {
                roots.insert(dsu.find(cell_idx));
            }
        }
        roots.len()
    };
    EdgeTopology {
        boundary_edges,
        overused_edges,
        same_direction_edge_pairs,
        connected_components,
    }
}

pub(super) fn validate_impl(diagram: &SphericalVoronoi) -> ValidationReport {
    let num_cells = diagram.num_cells();
    let num_vertices = diagram.num_vertices();
    let weld = audit_weld_map(diagram);
    let num_faces = num_cells - weld.welded_twin_cells;
    let cells = scan_cells(diagram, &weld, num_faces);
    let incidence = analyze_vertex_incidence(&cells.vertex_cell_count);
    let edge_geometry = analyze_edge_geometry(diagram, &cells.edges);
    let edge_topology =
        analyze_edge_topology(num_cells, num_faces, &weld.is_welded_twin, &cells.edges);
    let num_edges = cells.edges.len();
    let euler_characteristic = incidence.used_vertices as i32 - num_edges as i32 + num_faces as i32;
    let vertices_off_sphere = diagram
        .vertices()
        .iter()
        .filter(|v| !vertex_is_on_sphere(v.x(), v.y(), v.z()))
        .count();

    ValidationReport {
        num_cells,
        num_vertices,
        used_vertices: incidence.used_vertices,
        num_edges,
        euler_characteristic,
        connected_components: edge_topology.connected_components,
        degenerate_cells: cells.degenerate_cells,
        cells_with_fewer_than_three_stored_positions: cells
            .cells_with_fewer_than_three_stored_positions,
        cells_with_duplicate_vertices: cells.cells_with_duplicate_vertices,
        cells_with_invalid_references: cells.cells_with_invalid_references,
        invalid_vertex_references: cells.invalid_vertex_references,
        duplicate_cells_count: cells.duplicate_cells_count,
        unique_cells: cells.unique_cells,
        welded_twin_cells: weld.welded_twin_cells,
        weld_map_issues: weld.weld_map_issues,
        total_cell_vertices: cells.total_cell_vertices,
        vertices_off_sphere,
        orphan_vertices: incidence.orphan_vertices,
        low_incidence_vertices: incidence.low_incidence_vertices,
        degree_counts: incidence.degree_counts,
        self_loop_edges: cells.self_loop_edges,
        zero_length_edges: edge_geometry.zero_length_edges,
        antipodal_edges: edge_geometry.antipodal_edges,
        boundary_edges: edge_topology.boundary_edges,
        overused_edges: edge_topology.overused_edges,
        same_direction_edge_pairs: edge_topology.same_direction_edge_pairs,
    }
}
