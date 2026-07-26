use rustc_hash::{FxHashMap, FxHashSet};

use super::{CellMeshValidationReport, SphericalCellMesh, NO_CELL};

#[derive(Debug, Clone, Copy)]
struct EdgeUse {
    cell: u32,
    forward: bool,
}

struct CellScan {
    used_vertices: Vec<bool>,
    vertex_incidence: Vec<usize>,
    vertex_links: Vec<Vec<(u32, u32)>>,
    edge_uses: FxHashMap<(u32, u32), Vec<EdgeUse>>,
    degenerate_cells: usize,
    cells_with_fewer_than_three_stored_positions: usize,
    cells_with_duplicate_vertices: usize,
    cells_with_invalid_references: usize,
    duplicate_cells: usize,
}

fn count_vertices_off_sphere(mesh: &SphericalCellMesh) -> usize {
    mesh.vertices
        .iter()
        .filter(|vertex| {
            let len_sq = vertex.length_squared();
            !len_sq.is_finite() || (len_sq - 1.0).abs() > crate::tolerances::VERTEX_ON_SPHERE_EPS
        })
        .count()
}

fn scan_cells(mesh: &SphericalCellMesh) -> CellScan {
    let mut used_vertices = vec![false; mesh.num_vertices()];
    let mut vertex_incidence = vec![0usize; mesh.num_vertices()];
    let mut vertex_links = vec![Vec::<(u32, u32)>::new(); mesh.num_vertices()];
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
            used_vertices[vertex] = true;
            vertex_incidence[vertex] += 1;
            vertex_links[vertex].push((prev, next));

            let a = cycle[i];
            let b = cycle[(i + 1) % cycle.len()];
            let (lo, hi, forward) = if a < b { (a, b, true) } else { (b, a, false) };
            edge_uses.entry((lo, hi)).or_default().push(EdgeUse {
                cell: cell.cell_index as u32,
                forward,
            });
        }
    }

    CellScan {
        used_vertices,
        vertex_incidence,
        vertex_links,
        edge_uses,
        degenerate_cells,
        cells_with_fewer_than_three_stored_positions,
        cells_with_duplicate_vertices,
        cells_with_invalid_references,
        duplicate_cells,
    }
}

fn count_disconnected_vertex_links(vertex_links: &[Vec<(u32, u32)>]) -> usize {
    let mut disconnected = 0;
    for edges in vertex_links.iter().filter(|edges| !edges.is_empty()) {
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
        disconnected += usize::from(!valid);
    }
    disconnected
}

struct EdgeAnalysis {
    num_edges: usize,
    boundary_edges: usize,
    overused_edges: usize,
    same_direction_edge_pairs: usize,
    zero_length_edges: usize,
    antipodal_edges: usize,
    cell_neighbors: Vec<Vec<usize>>,
}

fn analyze_edges(
    mesh: &SphericalCellMesh,
    edge_uses: &FxHashMap<(u32, u32), Vec<EdgeUse>>,
) -> EdgeAnalysis {
    let mut boundary_edges = 0;
    let mut overused_edges = 0;
    let mut same_direction_edge_pairs = 0;
    let mut zero_length_edges = 0;
    let mut antipodal_edges = 0;
    let mut cell_neighbors = vec![Vec::<usize>::new(); mesh.num_cells()];

    for (&(a, b), uses) in edge_uses {
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

    EdgeAnalysis {
        num_edges: edge_uses.len(),
        boundary_edges,
        overused_edges,
        same_direction_edge_pairs,
        zero_length_edges,
        antipodal_edges,
        cell_neighbors,
    }
}

fn count_connected_components(cell_neighbors: &[Vec<usize>]) -> usize {
    let mut connected_components = 0;
    let mut seen_cells = vec![false; cell_neighbors.len()];
    for start in 0..cell_neighbors.len() {
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
    connected_components
}

fn count_provenance_issues(mesh: &SphericalCellMesh) -> usize {
    if mesh.cell_source_sites.len() != mesh.num_cells()
        || mesh.cell_to_input.len() != mesh.num_cells()
    {
        return 1;
    }

    let mut issues = mesh
        .cell_source_sites
        .iter()
        .filter(|site| {
            let len_sq = site.length_squared();
            !len_sq.is_finite() || (len_sq - 1.0).abs() > crate::tolerances::VERTEX_ON_SPHERE_EPS
        })
        .count();
    let mut cells_with_inputs = vec![false; mesh.num_cells()];
    for (input, &cell) in mesh.input_to_cell.iter().enumerate() {
        if cell == NO_CELL {
            continue;
        }
        let Some(slot) = cells_with_inputs.get_mut(cell as usize) else {
            issues += 1;
            continue;
        };
        *slot = true;
        if mesh.cell_to_input[cell as usize] as usize > input {
            issues += 1;
        }
    }
    for (cell, &source_input) in mesh.cell_to_input.iter().enumerate() {
        if source_input as usize >= mesh.input_to_cell.len()
            || mesh.input_to_cell[source_input as usize] != cell as u32
            || !cells_with_inputs[cell]
        {
            issues += 1;
        }
    }
    issues
}

pub(super) fn validate_cell_mesh(mesh: &SphericalCellMesh) -> CellMeshValidationReport {
    let vertices_off_sphere = count_vertices_off_sphere(mesh);
    let cell_scan = scan_cells(mesh);
    let orphan_vertices = cell_scan
        .used_vertices
        .iter()
        .filter(|&&is_used| !is_used)
        .count();
    let low_incidence_vertices = cell_scan
        .vertex_incidence
        .iter()
        .filter(|&&degree| degree > 0 && degree < 3)
        .count();
    let disconnected_vertex_links = count_disconnected_vertex_links(&cell_scan.vertex_links);
    let edge_analysis = analyze_edges(mesh, &cell_scan.edge_uses);
    let connected_components = count_connected_components(&edge_analysis.cell_neighbors);
    let provenance_issues = count_provenance_issues(mesh);

    CellMeshValidationReport {
        num_cells: mesh.num_cells(),
        num_vertices: mesh.num_vertices(),
        num_edges: edge_analysis.num_edges,
        euler_characteristic: mesh.num_vertices() as i32 - edge_analysis.num_edges as i32
            + mesh.num_cells() as i32,
        connected_components,
        degenerate_cells: cell_scan.degenerate_cells,
        cells_with_fewer_than_three_stored_positions: cell_scan
            .cells_with_fewer_than_three_stored_positions,
        cells_with_duplicate_vertices: cell_scan.cells_with_duplicate_vertices,
        cells_with_invalid_references: cell_scan.cells_with_invalid_references,
        duplicate_cells: cell_scan.duplicate_cells,
        vertices_off_sphere,
        orphan_vertices,
        low_incidence_vertices,
        disconnected_vertex_links,
        boundary_edges: edge_analysis.boundary_edges,
        overused_edges: edge_analysis.overused_edges,
        same_direction_edge_pairs: edge_analysis.same_direction_edge_pairs,
        zero_length_edges: edge_analysis.zero_length_edges,
        antipodal_edges: edge_analysis.antipodal_edges,
        provenance_issues,
    }
}
