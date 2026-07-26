use glam::Vec3;

use crate::diagram::VoronoiCell;

pub(crate) fn fib_sphere(n: usize) -> Vec<[f32; 3]> {
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

/// Extract effective-space arrays from a diagram without a weld map.
pub(crate) fn effective_arrays(
    diagram: &crate::SphericalVoronoi,
) -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<u32>) {
    let vertices = diagram
        .vertices()
        .iter()
        .map(|vertex| Vec3::from_array(vertex.to_array()))
        .collect();
    let cells = (0..diagram.num_cells())
        .map(|i| VoronoiCell::new(diagram.cell_start(i), diagram.cell(i).len() as u16))
        .collect();
    (vertices, cells, diagram.cell_indices_raw().to_vec())
}

pub(crate) fn effective_generators(diagram: &crate::SphericalVoronoi) -> Vec<Vec3> {
    diagram
        .generators()
        .iter()
        .map(|generator| Vec3::from_array(generator.to_array()))
        .collect()
}
