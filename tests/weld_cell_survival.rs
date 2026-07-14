//! Post-canonical f32-lattice campaign for the default weld/cell-survival
//! contract. This is deliberately self-contained: it checks stored coordinate
//! classes directly rather than treating topology or another backend as a
//! geometric oracle.

mod support;

use glam::{DQuat, DVec3};
use support::points::{near_cocircular_stress_points, random_sphere_points};
use voronoi_mesh::{compute_with_report, UnitVec3, VoronoiConfig};

const DEFAULT_WELD_DISTANCE_SQ: f32 = 128.0 * f32::EPSILON * f32::EPSILON;

fn canonical(mut point: DVec3) -> UnitVec3 {
    // Reach a fixed point of the public entry canonicalization so the pair's
    // measured lattice separation is the one construction actually receives.
    for _ in 0..4 {
        point = DVec3::new(
            f64::from(point.x as f32),
            f64::from(point.y as f32),
            f64::from(point.z as f32),
        )
        .normalize();
    }
    UnitVec3::new(point.x as f32, point.y as f32, point.z as f32)
}

fn chord_sq(a: UnitVec3, b: UnitVec3) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

/// Find the first representable canonical direction reached along one tangent
/// ray whose f32 chord predicate lies strictly outside the default weld.
fn threshold_adjacent_pair(center: DVec3) -> (UnitVec3, UnitVec3) {
    let p = canonical(center);
    let p64 = DVec3::new(p.x as f64, p.y as f64, p.z as f64);
    let reference = if p64.z.abs() < 0.8 {
        DVec3::Z
    } else {
        DVec3::X
    };
    let tangent = p64.cross(reference).normalize();
    let mut low = 0.0f64;
    let mut high = 4.0e-6f64;
    let mut best = canonical(p64 + tangent * high);
    assert!(chord_sq(p, best) > DEFAULT_WELD_DISTANCE_SQ);
    for _ in 0..80 {
        let mid = 0.5 * (low + high);
        let q = canonical(p64 + tangent * mid);
        if chord_sq(p, q) > DEFAULT_WELD_DISTANCE_SQ {
            high = mid;
            best = q;
        } else {
            low = mid;
        }
    }
    assert!(chord_sq(p, best) > DEFAULT_WELD_DISTANCE_SQ);
    (p, best)
}

fn exact_position_class_count(output: &voronoi_mesh::SphericalVoronoi, cell: usize) -> usize {
    let mut classes: Vec<(f32, f32, f32)> = Vec::new();
    for &vertex in output.cell(cell).vertex_indices {
        let p = output.vertex(vertex as usize);
        // Ordinary float equality deliberately identifies signed zero, as the
        // production stored-direction predicate does.
        let key = (p.x, p.y, p.z);
        if !classes.contains(&key) {
            classes.push(key);
        }
    }
    classes.len()
}

fn assert_no_stored_cell_collapse(name: &str, points: &[UnitVec3]) {
    let output = compute_with_report(points, VoronoiConfig::default())
        .unwrap_or_else(|error| panic!("{name}: computation failed: {error}"));
    assert_eq!(
        output.report.preprocess.num_merged, 0,
        "{name}: target unexpectedly fell inside the weld predicate"
    );
    assert!(
        output.report.preferred_validation().is_strictly_valid(),
        "{name}: {}",
        output.report.preferred_validation().headline()
    );
    let diagram = output.preferred_diagram();
    let collapsed: Vec<_> = (0..diagram.num_cells())
        .filter(|&cell| exact_position_class_count(diagram, cell) < 3)
        .collect();
    assert!(
        collapsed.is_empty(),
        "{name}: cells with fewer than three exact stored positions: {collapsed:?}"
    );
    assert_eq!(
        output
            .report
            .preferred_validation()
            .cells_with_fewer_than_three_stored_positions,
        0,
        "{name}: validator disagreed with the direct coordinate-class scan"
    );
}

fn rotated_pair_case(center: DVec3, rotation: DQuat, seed: u64) -> Vec<UnitVec3> {
    let rotated_center = rotation * center.normalize();
    let (p, q) = threshold_adjacent_pair(rotated_center);
    let mut points: Vec<_> = random_sphere_points(512, seed)
        .into_iter()
        .filter(|candidate| {
            let c = DVec3::new(candidate.x as f64, candidate.y as f64, candidate.z as f64);
            c.dot(rotated_center) < 0.9999
        })
        .collect();
    points.push(p);
    points.push(q);
    points
}

#[test]
fn default_weld_preserves_threshold_adjacent_cells_across_lattice_regimes() {
    let regimes = [
        ("axis", DVec3::X),
        ("face_seam", DVec3::new(1.0, 1.0, 0.0)),
        ("cube_corner", DVec3::new(1.0, 1.0, 1.0)),
        ("half_boundary", DVec3::new(0.5, 0.5, 0.5f64.sqrt())),
        (
            "quarter_boundary",
            DVec3::new(0.25, 15.0f64.sqrt() / 4.0, 0.0),
        ),
    ];
    let rotations = [
        DQuat::IDENTITY,
        DQuat::from_euler(glam::EulerRot::XYZ, 0.71, 1.13, 2.41),
    ];
    for (regime_idx, &(name, center)) in regimes.iter().enumerate() {
        for (rotation_idx, &rotation) in rotations.iter().enumerate() {
            let case = rotated_pair_case(
                center,
                rotation,
                0x51a7_0000 + (regime_idx * 17 + rotation_idx) as u64,
            );
            assert_no_stored_cell_collapse(&format!("{name}/rotation_{rotation_idx}"), &case);
        }
    }
}

#[test]
fn near_cocircular_cells_keep_three_stored_position_classes() {
    for (noise, seed) in [(1.0e-3, 42), (1.0e-4, 43), (1.0e-5, 44)] {
        let points = near_cocircular_stress_points(32, noise, seed);
        assert_no_stored_cell_collapse(&format!("near_cocircular/{noise:.0e}"), &points);
    }
}

#[test]
#[ignore = "extended f32-lattice rotation campaign; run manually in release mode"]
fn extended_threshold_adjacent_rotation_campaign() {
    let centers = [
        DVec3::X,
        DVec3::new(1.0, 1.0, 0.0),
        DVec3::new(1.0, 1.0, 1.0),
        DVec3::new(0.5, 0.5, 0.5f64.sqrt()),
    ];
    for rotation_idx in 0..64 {
        let t = rotation_idx as f64;
        let rotation = DQuat::from_euler(
            glam::EulerRot::XYZ,
            0.173 * t,
            0.311 * t + 0.07,
            0.419 * t + 0.13,
        );
        for (center_idx, &center) in centers.iter().enumerate() {
            let case = rotated_pair_case(
                center,
                rotation,
                0x51a8_0000 + (rotation_idx * centers.len() + center_idx) as u64,
            );
            assert_no_stored_cell_collapse(
                &format!("rotation_{rotation_idx}/center_{center_idx}"),
                &case,
            );
        }
    }
}
