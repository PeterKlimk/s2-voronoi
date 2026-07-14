//! Diagnostic: print representation and vertex-id-independent semantic
//! topology fingerprints so execution backends can be compared at the right
//! contract level.

mod support;

use support::points::random_sphere_points;
use voronoi_mesh::compute;

fn mix(h: &mut u64, value: u64) {
    *h ^= value;
    *h = h.wrapping_mul(0x100000001b3);
}

fn canonical_cycle(keys: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut best: Option<Vec<Vec<u32>>> = None;
    for reverse in [false, true] {
        for shift in 0..keys.len() {
            let candidate: Vec<Vec<u32>> = (0..keys.len())
                .map(|offset| {
                    let slot = if reverse {
                        (shift + keys.len() - offset) % keys.len()
                    } else {
                        (shift + offset) % keys.len()
                    };
                    keys[slot].clone()
                })
                .collect();
            if best.as_ref().is_none_or(|current| candidate < *current) {
                best = Some(candidate);
            }
        }
    }
    best.unwrap_or_default()
}

/// Fingerprint the abstract cell complex without depending on stored vertex
/// ids, vertex order, coordinates, cycle starts, or global winding. A
/// vertex's semantic identity is the sorted set of generator cells incident
/// to it; each generator cell is then hashed through its canonical cycle of
/// those identities.
fn semantic_topology_fingerprint(diagram: &voronoi_mesh::SphericalVoronoi) -> u64 {
    let mut incident_cells = vec![Vec::<u32>::new(); diagram.num_vertices()];
    for (cell_idx, cell) in diagram.iter_cells().enumerate() {
        for &vertex in cell.vertex_indices {
            incident_cells[vertex as usize].push(cell_idx as u32);
        }
    }
    for incident in &mut incident_cells {
        incident.sort_unstable();
        incident.dedup();
    }

    let mut h = 0xcbf29ce484222325;
    mix(&mut h, diagram.num_cells() as u64);
    for (cell_idx, cell) in diagram.iter_cells().enumerate() {
        mix(&mut h, cell_idx as u64);
        let keys: Vec<Vec<u32>> = cell
            .vertex_indices
            .iter()
            .map(|&vertex| incident_cells[vertex as usize].clone())
            .collect();
        let cycle = canonical_cycle(&keys);
        mix(&mut h, cycle.len() as u64);
        for key in cycle {
            mix(&mut h, key.len() as u64);
            for generator in key {
                mix(&mut h, generator as u64);
            }
        }
    }
    h
}

#[test]
fn semantic_cycle_canonicalization_ignores_start_and_winding() {
    let base = vec![vec![0, 2, 4], vec![0, 2, 5], vec![0, 3, 5], vec![0, 3, 4]];
    let expected = canonical_cycle(&base);
    for reverse in [false, true] {
        for shift in 0..base.len() {
            let mut variant = base.clone();
            if reverse {
                variant.reverse();
            }
            variant.rotate_left(shift);
            assert_eq!(canonical_cycle(&variant), expected);
        }
    }
}

#[test]
#[ignore = "cross-backend determinism check - run manually"]
fn backend_fingerprint() {
    let points = random_sphere_points(100_000, 7);
    let diagram = compute(&points).unwrap();

    let mut h: u64 = 0xcbf29ce484222325;
    for v in diagram.vertices() {
        mix(&mut h, v.x.to_bits() as u64);
        mix(&mut h, v.y.to_bits() as u64);
        mix(&mut h, v.z.to_bits() as u64);
    }
    for cell in diagram.iter_cells() {
        for &i in cell.vertex_indices {
            mix(&mut h, i as u64);
        }
    }
    let semantic = semantic_topology_fingerprint(&diagram);
    eprintln!(
        "fingerprint: {:016x} semantic_topology={:016x} vertices={} cells={}",
        h,
        semantic,
        diagram.num_vertices(),
        diagram.num_cells()
    );
}
