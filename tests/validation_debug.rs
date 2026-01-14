mod support;

use s2_voronoi::{compute, SphericalVoronoi};
use std::collections::{HashMap, HashSet};

/// For a d1 vertex V in cell A with triplet (A,B,C), check if cells B and C
/// have A as a neighbor (i.e., have any vertex involving generator A).
fn analyze_missing_neighbor_edges(diagram: &SphericalVoronoi) {
    let num_vertices = diagram.num_vertices();
    let num_cells = diagram.num_cells();

    // Count vertex degrees
    let mut vertex_degree: Vec<u32> = vec![0; num_vertices];
    for cell in diagram.iter_cells() {
        for &vi in cell.vertex_indices {
            if (vi as usize) < num_vertices {
                vertex_degree[vi as usize] += 1;
            }
        }
    }

    // For each cell, compute its "neighbor set" = generators that are equidistant
    // to any of its vertices. We find this by checking which generators are closest
    // to each vertex.
    let mut cell_neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); num_cells];

    for (cell_idx, cell) in diagram.iter_cells().enumerate() {
        for &vi in cell.vertex_indices {
            let v = diagram.vertices[vi as usize];

            // Find generators closest to this vertex (should be ~3 equidistant)
            let mut dists: Vec<(usize, f32)> = diagram
                .generators
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    let d = (v.x - g.x).powi(2) + (v.y - g.y).powi(2) + (v.z - g.z).powi(2);
                    (i, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // The first 3 should be nearly equidistant (the defining triplet)
            // Add them as neighbors (excluding self)
            for &(gen_idx, _) in dists.iter().take(3) {
                if gen_idx != cell_idx {
                    cell_neighbors[cell_idx].insert(gen_idx);
                }
            }
        }
    }

    // For each d1 vertex, find its defining triplet and check if B,C have A as neighbor
    let mut d1_both_have_a = 0usize;
    let mut d1_one_has_a = 0usize;
    let mut d1_neither_has_a = 0usize;
    let mut d1_total = 0usize;

    for (cell_a, cell) in diagram.iter_cells().enumerate() {
        for &vi in cell.vertex_indices {
            if (vi as usize) >= num_vertices || vertex_degree[vi as usize] != 1 {
                continue;
            }
            d1_total += 1;

            let v = diagram.vertices[vi as usize];

            // Find the 3 closest generators (the triplet A, B, C)
            let mut dists: Vec<(usize, f32)> = diagram
                .generators
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    let d = (v.x - g.x).powi(2) + (v.y - g.y).powi(2) + (v.z - g.z).powi(2);
                    (i, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let triplet: Vec<usize> = dists.iter().take(3).map(|&(i, _)| i).collect();

            // cell_a should be in the triplet
            if !triplet.contains(&cell_a) {
                // Weird - vertex in cell A but A is not in closest 3?
                continue;
            }

            // B and C are the other two
            let others: Vec<usize> = triplet.into_iter().filter(|&x| x != cell_a).collect();
            if others.len() != 2 {
                continue;
            }
            let (cell_b, cell_c) = (others[0], others[1]);

            // Check: does cell B have A as a neighbor?
            let b_has_a = cell_neighbors[cell_b].contains(&cell_a);
            // Check: does cell C have A as a neighbor?
            let c_has_a = cell_neighbors[cell_c].contains(&cell_a);

            match (b_has_a, c_has_a) {
                (true, true) => d1_both_have_a += 1,
                (true, false) | (false, true) => d1_one_has_a += 1,
                (false, false) => d1_neither_has_a += 1,
            }
        }
    }

    eprintln!("  d1 triplet neighbor analysis (n={}):", d1_total);
    eprintln!("    B and C both have A as neighbor: {}", d1_both_have_a);
    eprintln!("    only one has A as neighbor: {}", d1_one_has_a);
    eprintln!(
        "    neither B nor C has A as neighbor: {}",
        d1_neither_has_a
    );
}

/// Analyze edge lengths for low-degree vertices.
/// For each d1/d2 vertex, finds the min edge length in the cells that contain it.
fn analyze_bad_vertex_edges(diagram: &SphericalVoronoi) {
    let num_vertices = diagram.num_vertices();
    let num_cells = diagram.num_cells();

    // Count how many cells each vertex appears in
    let mut vertex_degree: Vec<u32> = vec![0; num_vertices];
    // Track which cells contain each vertex
    let mut vertex_cells: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];

    for (cell_idx, cell) in diagram.iter_cells().enumerate() {
        for &vi in cell.vertex_indices {
            let vi = vi as usize;
            if vi < num_vertices {
                vertex_degree[vi] += 1;
                vertex_cells[vi].push(cell_idx);
            }
        }
    }

    // For low-degree vertices, compute min edge length
    let mut d1_edge_lengths: Vec<f32> = Vec::new();
    let mut d2_edge_lengths: Vec<f32> = Vec::new();

    for (vi, &deg) in vertex_degree.iter().enumerate() {
        if deg != 1 && deg != 2 {
            continue;
        }

        let v = diagram.vertices[vi];
        let mut min_edge_len = f32::MAX;

        for &cell_idx in &vertex_cells[vi] {
            let cell = diagram.cell(cell_idx);
            let n = cell.len();
            if n < 2 {
                continue;
            }

            // Find position of this vertex in the cell
            let pos_in_cell = cell.vertex_indices.iter().position(|&x| x as usize == vi);
            if pos_in_cell.is_none() {
                continue;
            }
            let pos = pos_in_cell.unwrap();

            // Get prev and next vertices in the cell polygon
            let prev_vi = cell.vertex_indices[(pos + n - 1) % n] as usize;
            let next_vi = cell.vertex_indices[(pos + 1) % n] as usize;

            let prev_v = diagram.vertices[prev_vi];
            let dist_prev =
                ((v.x - prev_v.x).powi(2) + (v.y - prev_v.y).powi(2) + (v.z - prev_v.z).powi(2))
                    .sqrt();
            min_edge_len = min_edge_len.min(dist_prev);

            let next_v = diagram.vertices[next_vi];
            let dist_next =
                ((v.x - next_v.x).powi(2) + (v.y - next_v.y).powi(2) + (v.z - next_v.z).powi(2))
                    .sqrt();
            min_edge_len = min_edge_len.min(dist_next);
        }

        if min_edge_len < f32::MAX {
            if deg == 1 {
                d1_edge_lengths.push(min_edge_len);
            } else {
                d2_edge_lengths.push(min_edge_len);
            }
        }
    }

    // Sort and report percentiles
    d1_edge_lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d2_edge_lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let percentiles = [0, 10, 25, 50, 75, 90, 100];

    if !d1_edge_lengths.is_empty() {
        eprintln!(
            "  d1 min-edge-length percentiles (n={}):",
            d1_edge_lengths.len()
        );
        for p in percentiles {
            let idx = (p * d1_edge_lengths.len() / 100).min(d1_edge_lengths.len() - 1);
            eprintln!("    p{:3}: {:.2e}", p, d1_edge_lengths[idx]);
        }
    }

    if !d2_edge_lengths.is_empty() {
        eprintln!(
            "  d2 min-edge-length percentiles (n={}):",
            d2_edge_lengths.len()
        );
        for p in percentiles {
            let idx = (p * d2_edge_lengths.len() / 100).min(d2_edge_lengths.len() - 1);
            eprintln!("    p{:3}: {:.2e}", p, d2_edge_lengths[idx]);
        }
    }

    // Also report mean spacing for reference
    let mean_spacing = (4.0 * std::f32::consts::PI / num_cells as f32).sqrt();
    eprintln!("  (mean generator spacing: {:.2e})", mean_spacing);
}

/// Analyze how many low-degree vertices share positions with other vertices.
/// Tests multiple epsilon values and prints results.
fn analyze_position_duplicates(diagram: &SphericalVoronoi) -> (usize, usize, usize, usize) {
    let num_vertices = diagram.num_vertices();

    // Count how many cells each vertex appears in
    let mut vertex_degree: Vec<u32> = vec![0; num_vertices];
    for cell in diagram.iter_cells() {
        for &vi in cell.vertex_indices {
            if (vi as usize) < num_vertices {
                vertex_degree[vi as usize] += 1;
            }
        }
    }

    // Collect low-degree vertex indices and positions
    let mut low_degree: Vec<(usize, u32)> = Vec::new(); // (index, degree)
    for (i, &deg) in vertex_degree.iter().enumerate() {
        if deg == 1 || deg == 2 {
            low_degree.push((i, deg));
        }
    }

    if low_degree.is_empty() {
        return (0, 0, 0, 0);
    }

    // Build spatial grid for fast lookup (cell size = largest epsilon we'll test)
    let grid_size = 1e-4_f32;
    let inv_grid = 1.0 / grid_size;

    let grid_key = |v: &s2_voronoi::UnitVec3| -> (i32, i32, i32) {
        (
            (v.x * inv_grid) as i32,
            (v.y * inv_grid) as i32,
            (v.z * inv_grid) as i32,
        )
    };

    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, v) in diagram.vertices.iter().enumerate() {
        grid.entry(grid_key(v)).or_default().push(i);
    }

    // For each low-degree vertex, find min distance to any other vertex
    let mut min_dists: Vec<(usize, u32, f32)> = Vec::with_capacity(low_degree.len());

    for &(idx, deg) in &low_degree {
        let v = &diagram.vertices[idx];
        let (gx, gy, gz) = grid_key(v);

        let mut min_dist_sq = f32::MAX;

        // Check 27 neighboring cells
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = grid.get(&(gx + dx, gy + dy, gz + dz)) {
                        for &j in indices {
                            if j == idx {
                                continue;
                            }
                            let other = &diagram.vertices[j];
                            let d = (v.x - other.x).powi(2)
                                + (v.y - other.y).powi(2)
                                + (v.z - other.z).powi(2);
                            min_dist_sq = min_dist_sq.min(d);
                        }
                    }
                }
            }
        }

        min_dists.push((idx, deg, min_dist_sq.sqrt()));
    }

    // Report at multiple epsilon thresholds
    let epsilons = [0.0_f32, 1e-7, 1e-6, 1e-5, 1e-4];
    eprintln!("  Min-distance analysis for low-degree vertices:");
    for eps in epsilons {
        let d1_count = min_dists
            .iter()
            .filter(|&&(_, d, dist)| d == 1 && dist <= eps)
            .count();
        let d2_count = min_dists
            .iter()
            .filter(|&&(_, d, dist)| d == 2 && dist <= eps)
            .count();
        let d1_total = min_dists.iter().filter(|&&(_, d, _)| d == 1).count();
        let d2_total = min_dists.iter().filter(|&&(_, d, _)| d == 2).count();
        eprintln!(
            "    eps={:.0e}: d1={}/{} ({:.1}%), d2={}/{} ({:.1}%)",
            eps,
            d1_count,
            d1_total,
            if d1_total > 0 {
                d1_count as f64 / d1_total as f64 * 100.0
            } else {
                0.0
            },
            d2_count,
            d2_total,
            if d2_total > 0 {
                d2_count as f64 / d2_total as f64 * 100.0
            } else {
                0.0
            }
        );
    }

    // Return counts for eps=1e-6
    let d1_with = min_dists
        .iter()
        .filter(|&&(_, d, dist)| d == 1 && dist <= 1e-6)
        .count();
    let d2_with = min_dists
        .iter()
        .filter(|&&(_, d, dist)| d == 2 && dist <= 1e-6)
        .count();
    let d1_total = min_dists.iter().filter(|&&(_, d, _)| d == 1).count();
    let d2_total = min_dists.iter().filter(|&&(_, d, _)| d == 2).count();

    (d1_with, d1_total, d2_with, d2_total)
}

#[test]
#[ignore]
fn debug_validation_helpers() {
    let points = support::points::fibonacci_sphere_points(200, 0.1, 4242);
    let diagram = compute(&points).expect("compute should succeed");

    analyze_missing_neighbor_edges(&diagram);
    analyze_bad_vertex_edges(&diagram);
    let _ = analyze_position_duplicates(&diagram);
}
