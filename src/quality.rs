//! Internal quality / fidelity metrics for spherical Voronoi diagrams.
//!
//! This module is intentionally separate from strict subdivision validation.
//! It reports sampled ownership consistency and Voronoi residuals, but does not
//! define pass/fail correctness.

use crate::cube_grid::{CubeMapGrid, CubeMapGridScratch, DirectedCtx};
use crate::SphericalVoronoi;
use glam::Vec3;
use std::collections::{HashMap, HashSet};

const GRID_TARGET_DENSITY: f64 = 16.0;
const LOW_DEGREE_DUPLICATE_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy)]
pub struct QualityConfig {
    pub max_sampled_cells: usize,
    pub edge_samples_per_edge: usize,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            max_sampled_cells: 128,
            edge_samples_per_edge: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ResidualStats {
    pub samples: usize,
    pub max_abs: f32,
    pub mean_abs: f32,
    pub p95_abs: f32,
}

impl ResidualStats {
    fn from_values(mut values: Vec<f32>) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let samples = values.len();
        let sum: f32 = values.iter().copied().sum();
        let max_abs = values
            .iter()
            .copied()
            .fold(0.0f32, |acc, v| acc.max(v.abs()));
        values.sort_by(|a, b| a.total_cmp(b));
        let p95_idx = ((samples - 1) * 95) / 100;
        Self {
            samples,
            max_abs,
            mean_abs: sum / samples as f32,
            p95_abs: values[p95_idx],
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SampledOwnershipStats {
    pub sampled_cells: usize,
    pub samples: usize,
    pub mismatches: usize,
    pub worst_margin_violation: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LowDegreeQualityStats {
    pub low_degree_vertices: usize,
    pub near_duplicate_vertices: usize,
    pub min_neighbor_distance: f32,
}

#[derive(Debug, Clone)]
pub struct QualityReport {
    pub ownership: SampledOwnershipStats,
    pub vertex_dot_residuals: ResidualStats,
    pub edge_dot_residuals: ResidualStats,
    pub low_degree: LowDegreeQualityStats,
}

impl QualityReport {
    pub fn headline(&self) -> String {
        format!(
            "ownership mismatches={}/{}, vertex residual max={:.2e} p95={:.2e}, edge residual max={:.2e} p95={:.2e}, low-degree near-dupes={}/{}",
            self.ownership.mismatches,
            self.ownership.samples,
            self.vertex_dot_residuals.max_abs,
            self.vertex_dot_residuals.p95_abs,
            self.edge_dot_residuals.max_abs,
            self.edge_dot_residuals.p95_abs,
            self.low_degree.near_duplicate_vertices,
            self.low_degree.low_degree_vertices,
        )
    }
}

#[cfg(feature = "qhull")]
#[derive(Debug, Clone, Copy, Default)]
pub struct QhullCellCountComparison {
    pub total_cells: usize,
    pub matching_cell_vertex_counts: usize,
    pub match_ratio: f32,
}

pub fn assess(diagram: &SphericalVoronoi) -> QualityReport {
    assess_with_config(diagram, QualityConfig::default())
}

pub fn assess_with_config(diagram: &SphericalVoronoi, config: QualityConfig) -> QualityReport {
    let generators: Vec<Vec3> = diagram
        .generators()
        .iter()
        .map(|g| Vec3::new(g.x, g.y, g.z))
        .collect();
    let vertices: Vec<Vec3> = diagram
        .vertices()
        .iter()
        .map(|v| Vec3::new(v.x, v.y, v.z))
        .collect();

    let sampled_cells = sampled_cell_indices(diagram.num_cells(), config.max_sampled_cells);
    let sampled_set: HashSet<usize> = sampled_cells.iter().copied().collect();

    let mut vertex_to_cells = vec![Vec::<usize>::new(); diagram.num_vertices()];
    let mut edge_to_cells: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    let mut sampled_vertices = HashSet::<u32>::new();
    let mut sampled_edges = HashSet::<(u32, u32)>::new();

    for cell in diagram.iter_cells() {
        for &vi in cell.vertex_indices {
            if (vi as usize) < diagram.num_vertices() {
                vertex_to_cells[vi as usize].push(cell.generator_index);
            }
        }
        let len = cell.len();
        if len >= 2 {
            for edge_idx in 0..len {
                let a = cell.vertex_indices[edge_idx];
                let b = cell.vertex_indices[(edge_idx + 1) % len];
                if a == b {
                    continue;
                }
                let edge = if a < b { (a, b) } else { (b, a) };
                edge_to_cells
                    .entry(edge)
                    .or_default()
                    .push(cell.generator_index);
                if sampled_set.contains(&cell.generator_index) {
                    sampled_edges.insert(edge);
                }
            }
        }
        if sampled_set.contains(&cell.generator_index) {
            for &vi in cell.vertex_indices {
                if (vi as usize) < diagram.num_vertices() {
                    sampled_vertices.insert(vi);
                }
            }
        }
    }

    let ownership = assess_sampled_ownership(diagram, &generators, &vertices, &sampled_cells);
    let vertex_dot_residuals =
        assess_vertex_residuals(&vertices, &generators, &vertex_to_cells, &sampled_vertices);
    let edge_dot_residuals = assess_edge_residuals(
        &vertices,
        &generators,
        &edge_to_cells,
        &sampled_edges,
        config.edge_samples_per_edge.max(1),
    );
    let low_degree = analyze_low_degree_vertices(diagram);

    QualityReport {
        ownership,
        vertex_dot_residuals,
        edge_dot_residuals,
        low_degree,
    }
}

#[cfg(feature = "qhull")]
pub fn compare_cell_vertex_counts(
    diagram: &SphericalVoronoi,
    reference: &SphericalVoronoi,
) -> QhullCellCountComparison {
    let total_cells = diagram.num_cells().min(reference.num_cells());
    if total_cells == 0 {
        return QhullCellCountComparison::default();
    }

    let mut matching = 0usize;
    for i in 0..total_cells {
        if diagram.cell(i).len() == reference.cell(i).len() {
            matching += 1;
        }
    }

    QhullCellCountComparison {
        total_cells,
        matching_cell_vertex_counts: matching,
        match_ratio: matching as f32 / total_cells as f32,
    }
}

fn assess_sampled_ownership(
    diagram: &SphericalVoronoi,
    generators: &[Vec3],
    vertices: &[Vec3],
    sampled_cells: &[usize],
) -> SampledOwnershipStats {
    if generators.is_empty() || sampled_cells.is_empty() {
        return SampledOwnershipStats::default();
    }

    let grid = build_generator_grid(generators);
    let fake_slot_map = vec![0u32; generators.len()];
    let ctx = DirectedCtx::new(u8::MAX, 0, &fake_slot_map, 0, 0);
    let mut scratch = grid.make_scratch();

    let mut samples = 0usize;
    let mut mismatches = 0usize;
    let mut worst_margin_violation = 0.0f32;

    for &cell_idx in sampled_cells {
        let cell = diagram.cell(cell_idx);
        let len = cell.len();
        if len < 2 {
            continue;
        }

        let g = generators[cell_idx];
        for edge_idx in 0..len {
            let a = cell.vertex_indices[edge_idx] as usize;
            let b = cell.vertex_indices[(edge_idx + 1) % len] as usize;
            if a >= vertices.len() || b >= vertices.len() {
                continue;
            }

            let sample = (g * 2.0 + vertices[a] + vertices[b]).normalize();
            let Some(nearest) =
                nearest_generator_index(&grid, generators, &mut scratch, ctx, sample)
            else {
                continue;
            };
            samples += 1;
            if nearest != cell_idx {
                mismatches += 1;
                let nearest_dot = sample.dot(generators[nearest]);
                let owner_dot = sample.dot(g);
                worst_margin_violation = worst_margin_violation.max(nearest_dot - owner_dot);
            }
        }
    }

    SampledOwnershipStats {
        sampled_cells: sampled_cells.len(),
        samples,
        mismatches,
        worst_margin_violation,
    }
}

fn assess_vertex_residuals(
    vertices: &[Vec3],
    generators: &[Vec3],
    vertex_to_cells: &[Vec<usize>],
    sampled_vertices: &HashSet<u32>,
) -> ResidualStats {
    let mut residuals = Vec::new();

    for &vi in sampled_vertices {
        let vi = vi as usize;
        if vi >= vertices.len() {
            continue;
        }
        let cells = &vertex_to_cells[vi];
        if cells.len() < 2 {
            continue;
        }
        let p = vertices[vi];
        let mut min_dot = f32::INFINITY;
        let mut max_dot = f32::NEG_INFINITY;
        for &cell_idx in cells {
            let dot = p.dot(generators[cell_idx]);
            min_dot = min_dot.min(dot);
            max_dot = max_dot.max(dot);
        }
        residuals.push((max_dot - min_dot).abs());
    }

    ResidualStats::from_values(residuals)
}

fn assess_edge_residuals(
    vertices: &[Vec3],
    generators: &[Vec3],
    edge_to_cells: &HashMap<(u32, u32), Vec<usize>>,
    sampled_edges: &HashSet<(u32, u32)>,
    samples_per_edge: usize,
) -> ResidualStats {
    let mut residuals = Vec::new();

    for &(a, b) in sampled_edges {
        let Some(cells) = edge_to_cells.get(&(a, b)) else {
            continue;
        };
        let mut uniq_cells = cells.clone();
        uniq_cells.sort_unstable();
        uniq_cells.dedup();
        if uniq_cells.len() != 2 {
            continue;
        }

        let a = a as usize;
        let b = b as usize;
        if a >= vertices.len() || b >= vertices.len() {
            continue;
        }

        let va = vertices[a];
        let vb = vertices[b];
        let ga = generators[uniq_cells[0]];
        let gb = generators[uniq_cells[1]];

        for k in 1..=samples_per_edge {
            let t = k as f32 / (samples_per_edge + 1) as f32;
            let p = (va * (1.0 - t) + vb * t).normalize();
            residuals.push((p.dot(ga) - p.dot(gb)).abs());
        }
    }

    ResidualStats::from_values(residuals)
}

fn analyze_low_degree_vertices(diagram: &SphericalVoronoi) -> LowDegreeQualityStats {
    let num_vertices = diagram.num_vertices();
    let diagram_vertices = diagram.vertices();
    if num_vertices == 0 {
        return LowDegreeQualityStats::default();
    }

    let mut vertex_degree: Vec<u32> = vec![0; num_vertices];
    for cell in diagram.iter_cells() {
        for &vi in cell.vertex_indices {
            if (vi as usize) < num_vertices {
                vertex_degree[vi as usize] += 1;
            }
        }
    }

    let low_degree: Vec<usize> = vertex_degree
        .iter()
        .enumerate()
        .filter_map(|(idx, &deg)| (deg == 1 || deg == 2).then_some(idx))
        .collect();

    if low_degree.is_empty() {
        return LowDegreeQualityStats::default();
    }

    let grid_size = 1e-4_f32;
    let inv_grid = 1.0 / grid_size;
    let grid_key = |v: &crate::UnitVec3| -> (i32, i32, i32) {
        (
            (v.x * inv_grid) as i32,
            (v.y * inv_grid) as i32,
            (v.z * inv_grid) as i32,
        )
    };

    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, v) in diagram_vertices.iter().enumerate() {
        grid.entry(grid_key(v)).or_default().push(i);
    }

    let mut near_duplicate_vertices = 0usize;
    let mut min_neighbor_distance = f32::MAX;

    for idx in low_degree {
        let v = &diagram_vertices[idx];
        let (gx, gy, gz) = grid_key(v);
        let mut min_dist_sq = f32::MAX;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = grid.get(&(gx + dx, gy + dy, gz + dz)) {
                        for &j in indices {
                            if j == idx {
                                continue;
                            }
                            let other = &diagram_vertices[j];
                            let d = (v.x - other.x).powi(2)
                                + (v.y - other.y).powi(2)
                                + (v.z - other.z).powi(2);
                            min_dist_sq = min_dist_sq.min(d);
                        }
                    }
                }
            }
        }

        if min_dist_sq.is_finite() {
            let min_dist = min_dist_sq.sqrt();
            min_neighbor_distance = min_neighbor_distance.min(min_dist);
            if min_dist <= LOW_DEGREE_DUPLICATE_EPS {
                near_duplicate_vertices += 1;
            }
        }
    }

    LowDegreeQualityStats {
        low_degree_vertices: vertex_degree
            .iter()
            .filter(|&&deg| deg == 1 || deg == 2)
            .count(),
        near_duplicate_vertices,
        min_neighbor_distance: if min_neighbor_distance.is_finite() {
            min_neighbor_distance
        } else {
            0.0
        },
    }
}

fn sampled_cell_indices(num_cells: usize, max_sampled_cells: usize) -> Vec<usize> {
    if num_cells == 0 || max_sampled_cells == 0 {
        return Vec::new();
    }
    if num_cells <= max_sampled_cells {
        return (0..num_cells).collect();
    }

    let mut out = Vec::with_capacity(max_sampled_cells);
    let mut last = None;
    for i in 0..max_sampled_cells {
        let idx = i * num_cells / max_sampled_cells;
        if last != Some(idx) {
            out.push(idx);
            last = Some(idx);
        }
    }
    out
}

fn build_generator_grid(generators: &[Vec3]) -> CubeMapGrid {
    let n = generators.len();
    let target = GRID_TARGET_DENSITY.max(1.0);
    let res = ((n as f64 / (6.0 * target)).sqrt() as usize).max(4);
    CubeMapGrid::new(generators, res)
}

fn nearest_generator_index(
    grid: &CubeMapGrid,
    generators: &[Vec3],
    scratch: &mut CubeMapGridScratch,
    ctx: DirectedCtx<'_>,
    query: Vec3,
) -> Option<usize> {
    let mut cursor = grid.directed_no_k_cursor(query, generators.len(), scratch, ctx);
    let slot = cursor.pop_next_proven_slot()?;
    Some(grid.point_indices()[slot as usize] as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compute, UnitVec3};

    fn fibonacci_sphere_points(n: usize, jitter: f32) -> Vec<UnitVec3> {
        let golden_angle = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        (0..n)
            .map(|i| {
                let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
                let radius = (1.0 - y * y).sqrt();
                let theta = golden_angle * i as f32;
                let x = radius * theta.cos() + (((i * 37 + 11) as f32) * 0.12345).sin() * jitter;
                let z = radius * theta.sin() + (((i * 53 + 7) as f32) * 0.23456).cos() * jitter;
                let p = Vec3::new(x, y, z).normalize();
                UnitVec3::new(p.x, p.y, p.z)
            })
            .collect()
    }

    fn clustered_cap_points(n: usize, cap_radius_rad: f32) -> Vec<UnitVec3> {
        let mut points = Vec::with_capacity(n);
        points.push(UnitVec3::new(1.0, 0.0, 0.0));
        points.push(UnitVec3::new(-1.0, 0.0, 0.0));
        points.push(UnitVec3::new(0.0, 1.0, 0.0));
        points.push(UnitVec3::new(0.0, -1.0, 0.0));
        points.push(UnitVec3::new(0.0, 0.0, 1.0));
        points.push(UnitVec3::new(0.0, 0.0, -1.0));

        let clustered = n.saturating_sub(6);
        for i in 0..clustered {
            let t = (i as f32 + 0.5) / clustered.max(1) as f32;
            let theta = (2.0 * std::f32::consts::PI * 1.618_034 * i as f32).fract()
                * 2.0
                * std::f32::consts::PI;
            let cos_theta = 1.0 - t * (1.0 - cap_radius_rad.cos());
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
            points.push(UnitVec3::new(
                sin_theta * theta.cos(),
                sin_theta * theta.sin(),
                cos_theta,
            ));
        }

        points
    }

    #[test]
    fn healthy_input_has_small_quality_residuals() {
        let points = fibonacci_sphere_points(100, 0.01);
        let diagram = compute(&points).expect("compute should succeed");
        let report = assess_with_config(
            &diagram,
            QualityConfig {
                max_sampled_cells: 64,
                edge_samples_per_edge: 3,
            },
        );

        assert_eq!(report.ownership.mismatches, 0, "{}", report.headline());
        assert!(
            report.vertex_dot_residuals.max_abs < 1e-3,
            "{}",
            report.headline()
        );
        assert!(
            report.edge_dot_residuals.max_abs < 1e-3,
            "{}",
            report.headline()
        );
    }

    #[test]
    fn clustered_cap_quality_is_worse_than_well_spaced_input() {
        let healthy_points = fibonacci_sphere_points(100, 0.01);
        let healthy = compute(&healthy_points).expect("healthy compute should succeed");
        let healthy_report = assess_with_config(
            &healthy,
            QualityConfig {
                max_sampled_cells: 64,
                edge_samples_per_edge: 3,
            },
        );

        let stressed_points = clustered_cap_points(100, 0.0175);
        let stressed = compute(&stressed_points).expect("clustered cap should compute");
        let stressed_report = assess_with_config(
            &stressed,
            QualityConfig {
                max_sampled_cells: 64,
                edge_samples_per_edge: 3,
            },
        );

        assert!(
            stressed_report.ownership.mismatches > healthy_report.ownership.mismatches
                || stressed_report.vertex_dot_residuals.max_abs
                    > healthy_report.vertex_dot_residuals.max_abs
                || stressed_report.edge_dot_residuals.max_abs
                    > healthy_report.edge_dot_residuals.max_abs
                || stressed_report.low_degree.near_duplicate_vertices
                    > healthy_report.low_degree.near_duplicate_vertices,
            "expected stressed quality report to show worse metrics\nhealthy={}\nstressed={}",
            healthy_report.headline(),
            stressed_report.headline()
        );
    }
}
