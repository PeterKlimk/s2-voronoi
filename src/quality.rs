//! Internal quality / fidelity metrics for spherical Voronoi diagrams.
//!
//! This module is intentionally separate from strict subdivision validation.
//! It reports sampled ownership consistency and Voronoi residuals, but does not
//! define pass/fail correctness.

use crate::cube_grid::{CubeMapGrid, CubeMapGridScratch};
use crate::spherical_arc::resolve_owner_arc;
use crate::tolerances::ANTIPODAL_DOT_EPS;
use crate::SphericalVoronoi;
use glam::{DVec3, Vec3};
use std::collections::{HashMap, HashSet};

const GRID_TARGET_DENSITY: f64 = 16.0;
const LOW_DEGREE_DUPLICATE_EPS: f32 = 1e-6;
const SITE_CHORD_BUCKET_UPPERS: [f64; 5] = [2e-6, 1e-5, 1e-4, 1e-3, f64::INFINITY];

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

#[derive(Debug, Clone, Copy, Default)]
pub struct F64Stats {
    pub samples: usize,
    pub max: f64,
    pub mean: f64,
    pub p95: f64,
    pub p99: f64,
}

impl F64Stats {
    fn from_values(mut values: Vec<f64>) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let samples = values.len();
        let sum: f64 = values.iter().copied().sum();
        let max = values.iter().copied().fold(0.0f64, f64::max);
        values.sort_by(f64::total_cmp);
        let percentile = |percent: usize| values[(samples - 1) * percent / 100];
        Self {
            samples,
            max,
            mean: sum / samples as f64,
            p95: percentile(95),
            p99: percentile(99),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AngularConditionBucket {
    pub site_chord_upper: f64,
    pub radians: F64Stats,
}

#[derive(Debug, Clone)]
pub struct ConditionedAngularStats {
    pub overall: F64Stats,
    pub buckets: [AngularConditionBucket; SITE_CHORD_BUCKET_UPPERS.len()],
}

impl Default for ConditionedAngularStats {
    fn default() -> Self {
        Self {
            overall: F64Stats::default(),
            buckets: SITE_CHORD_BUCKET_UPPERS.map(|site_chord_upper| AngularConditionBucket {
                site_chord_upper,
                radians: F64Stats::default(),
            }),
        }
    }
}

impl ConditionedAngularStats {
    pub fn aggregate(&self) -> F64Stats {
        self.overall
    }
}

#[derive(Default)]
struct ConditionedAngularValues {
    overall: Vec<f64>,
    buckets: [Vec<f64>; SITE_CHORD_BUCKET_UPPERS.len()],
}

impl ConditionedAngularValues {
    fn push(&mut self, site_chord: f64, radians: f64) {
        if !site_chord.is_finite() || !radians.is_finite() {
            return;
        }
        self.overall.push(radians);
        let bucket = SITE_CHORD_BUCKET_UPPERS
            .iter()
            .position(|&upper| site_chord < upper)
            .unwrap_or(SITE_CHORD_BUCKET_UPPERS.len() - 1);
        self.buckets[bucket].push(radians);
    }

    fn finish(self) -> ConditionedAngularStats {
        ConditionedAngularStats {
            overall: F64Stats::from_values(self.overall),
            buckets: std::array::from_fn(|i| AngularConditionBucket {
                site_chord_upper: SITE_CHORD_BUCKET_UPPERS[i],
                radians: F64Stats::from_values(self.buckets[i].clone()),
            }),
        }
    }
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
    pub worst_margin_violation: f64,
    pub worst_cross_track_radians: f64,
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
    pub canonicalization_angular_error: F64Stats,
    pub vertex_norm_error: F64Stats,
    pub vertex_cross_track_error: ConditionedAngularStats,
    pub edge_cross_track_error: ConditionedAngularStats,
    pub low_degree: LowDegreeQualityStats,
}

impl QualityReport {
    pub fn headline(&self) -> String {
        let vertex_angular = self.vertex_cross_track_error.aggregate();
        let edge_angular = self.edge_cross_track_error.aggregate();
        format!(
            "ownership mismatches={}/{}, vertex cross-track max={:.2e}rad, edge cross-track max={:.2e}rad, vertex norm max={:.2e}, low-degree near-dupes={}/{}",
            self.ownership.mismatches,
            self.ownership.samples,
            vertex_angular.max,
            edge_angular.max,
            self.vertex_norm_error.max,
            self.low_degree.near_duplicate_vertices,
            self.low_degree.low_degree_vertices,
        )
    }

    pub fn fidelity_kv_fields(&self) -> String {
        let mut fields = format!(
            "ownership_samples={} ownership_mismatches={} ownership_worst_margin={:.17e} ownership_worst_rad={:.17e} canonical_n={} canonical_max_rad={:.17e} canonical_p99_rad={:.17e} vertex_norm_n={} vertex_norm_max={:.17e} vertex_norm_p99={:.17e} vertex_cross_n={} vertex_cross_max_rad={:.17e} vertex_cross_p99_rad={:.17e} edge_cross_n={} edge_cross_max_rad={:.17e} edge_cross_p99_rad={:.17e}",
            self.ownership.samples,
            self.ownership.mismatches,
            self.ownership.worst_margin_violation,
            self.ownership.worst_cross_track_radians,
            self.canonicalization_angular_error.samples,
            self.canonicalization_angular_error.max,
            self.canonicalization_angular_error.p99,
            self.vertex_norm_error.samples,
            self.vertex_norm_error.max,
            self.vertex_norm_error.p99,
            self.vertex_cross_track_error.overall.samples,
            self.vertex_cross_track_error.overall.max,
            self.vertex_cross_track_error.overall.p99,
            self.edge_cross_track_error.overall.samples,
            self.edge_cross_track_error.overall.max,
            self.edge_cross_track_error.overall.p99,
        );
        for (i, bucket) in self.edge_cross_track_error.buckets.iter().enumerate() {
            use std::fmt::Write;
            write!(
                fields,
                " edge_b{i}_upper={:.17e} edge_b{i}_n={} edge_b{i}_max_rad={:.17e} edge_b{i}_p99_rad={:.17e}",
                bucket.site_chord_upper,
                bucket.radians.samples,
                bucket.radians.max,
                bucket.radians.p99,
            )
            .expect("writing to String cannot fail");
        }
        fields
    }
}

pub fn assess(diagram: &SphericalVoronoi) -> QualityReport {
    assess_with_config(diagram, QualityConfig::default())
}

pub fn assess_canonicalization<P: crate::UnitVec3Like>(
    input: &[P],
    returned_diagram: &SphericalVoronoi,
) -> F64Stats {
    let values = input
        .iter()
        .zip(returned_diagram.generators())
        .filter_map(|(before, after)| {
            let before = DVec3::new(before.x() as f64, before.y() as f64, before.z() as f64);
            let after = DVec3::new(after.x() as f64, after.y() as f64, after.z() as f64);
            (before.length_squared() > 0.0 && after.length_squared() > 0.0)
                .then(|| angular_separation(before.normalize(), after.normalize()))
        })
        .collect();
    F64Stats::from_values(values)
}

pub fn assess_with_config(diagram: &SphericalVoronoi, config: QualityConfig) -> QualityReport {
    let generators: Vec<Vec3> = diagram
        .generators()
        .iter()
        .map(|g| Vec3::from_array(g.to_array()))
        .collect();
    let vertices: Vec<Vec3> = diagram
        .vertices()
        .iter()
        .map(|v| Vec3::from_array(v.to_array()))
        .collect();
    let normalized_generators: Vec<DVec3> = generators.iter().copied().map(normalize_f64).collect();
    let normalized_vertices: Vec<DVec3> = vertices.iter().copied().map(normalize_f64).collect();

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

    let ownership = assess_sampled_ownership(
        diagram,
        &generators,
        &vertices,
        &normalized_generators,
        &sampled_cells,
    );
    let vertex_dot_residuals =
        assess_vertex_residuals(&vertices, &generators, &vertex_to_cells, &sampled_vertices);
    let edge_dot_residuals = assess_edge_residuals(
        &vertices,
        &generators,
        &edge_to_cells,
        &sampled_edges,
        config.edge_samples_per_edge.max(1),
    );
    let vertex_norm_error = assess_vertex_norm_error(&vertices, &sampled_vertices);
    let vertex_cross_track_error = assess_vertex_cross_track(
        &normalized_vertices,
        &normalized_generators,
        &vertex_to_cells,
        &sampled_vertices,
    );
    let edge_cross_track_error = assess_edge_cross_track(
        &normalized_vertices,
        &normalized_generators,
        &edge_to_cells,
        &sampled_edges,
        config.edge_samples_per_edge,
    );
    let low_degree = analyze_low_degree_vertices(diagram);

    QualityReport {
        ownership,
        vertex_dot_residuals,
        edge_dot_residuals,
        canonicalization_angular_error: F64Stats::default(),
        vertex_norm_error,
        vertex_cross_track_error,
        edge_cross_track_error,
        low_degree,
    }
}

fn assess_sampled_ownership(
    diagram: &SphericalVoronoi,
    generators: &[Vec3],
    vertices: &[Vec3],
    normalized_generators: &[DVec3],
    sampled_cells: &[usize],
) -> SampledOwnershipStats {
    if generators.is_empty() || sampled_cells.is_empty() {
        return SampledOwnershipStats::default();
    }

    let grid = build_generator_grid(generators);
    let mut scratch = grid.make_scratch();
    let mut batch = Vec::new();

    let mut samples = 0usize;
    let mut mismatches = 0usize;
    let mut worst_margin_violation = 0.0f64;
    let mut worst_cross_track_radians = 0.0f64;

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
            let Some(nearest) = nearest_generator_index(&grid, &mut scratch, &mut batch, sample)
            else {
                continue;
            };
            samples += 1;
            if nearest != cell_idx {
                let sample = normalize_f64(sample);
                let nearest_dot = sample.dot(normalized_generators[nearest]);
                let owner_dot = sample.dot(normalized_generators[cell_idx]);
                let violation = nearest_dot - owner_dot;
                if violation <= 0.0 {
                    continue;
                }
                mismatches += 1;
                worst_margin_violation = worst_margin_violation.max(violation);
                if let Some((_, radians)) = cross_track_radians(
                    sample,
                    normalized_generators[cell_idx],
                    normalized_generators[nearest],
                ) {
                    worst_cross_track_radians = worst_cross_track_radians.max(radians);
                }
            }
        }
    }

    SampledOwnershipStats {
        sampled_cells: sampled_cells.len(),
        samples,
        mismatches,
        worst_margin_violation,
        worst_cross_track_radians,
    }
}

fn normalize_f64(p: Vec3) -> DVec3 {
    DVec3::new(p.x as f64, p.y as f64, p.z as f64).normalize()
}

fn angular_separation(a: DVec3, b: DVec3) -> f64 {
    a.cross(b).length().atan2(a.dot(b).clamp(-1.0, 1.0))
}

fn cross_track_radians(p: DVec3, a: DVec3, b: DVec3) -> Option<(f64, f64)> {
    let normal = a - b;
    let site_chord = normal.length();
    if site_chord == 0.0 || !site_chord.is_finite() {
        return None;
    }
    let sine = (p.dot(normal) / site_chord).abs().min(1.0);
    Some((site_chord, sine.asin()))
}

fn assess_vertex_norm_error(vertices: &[Vec3], sampled_vertices: &HashSet<u32>) -> F64Stats {
    F64Stats::from_values(
        sampled_vertices
            .iter()
            .filter_map(|&vi| vertices.get(vi as usize))
            .map(|v| (DVec3::new(v.x as f64, v.y as f64, v.z as f64).length() - 1.0).abs())
            .collect(),
    )
}

fn assess_vertex_cross_track(
    vertices: &[DVec3],
    generators: &[DVec3],
    vertex_to_cells: &[Vec<usize>],
    sampled_vertices: &HashSet<u32>,
) -> ConditionedAngularStats {
    let mut values = ConditionedAngularValues::default();
    for &vi in sampled_vertices {
        let vi = vi as usize;
        let Some(&p) = vertices.get(vi) else {
            continue;
        };
        let mut cells = vertex_to_cells[vi].clone();
        cells.sort_unstable();
        cells.dedup();
        for i in 0..cells.len() {
            for j in i + 1..cells.len() {
                if let Some((site_chord, radians)) =
                    cross_track_radians(p, generators[cells[i]], generators[cells[j]])
                {
                    values.push(site_chord, radians);
                }
            }
        }
    }
    values.finish()
}

fn assess_edge_cross_track(
    vertices: &[DVec3],
    generators: &[DVec3],
    edge_to_cells: &HashMap<(u32, u32), Vec<usize>>,
    sampled_edges: &HashSet<(u32, u32)>,
    interior_samples_per_edge: usize,
) -> ConditionedAngularStats {
    let mut values = ConditionedAngularValues::default();
    for &(a, b) in sampled_edges {
        let Some(cells) = edge_to_cells.get(&(a, b)) else {
            continue;
        };
        let mut cells = cells.clone();
        cells.sort_unstable();
        cells.dedup();
        if cells.len() != 2 {
            continue;
        }
        let (Some(&va), Some(&vb)) = (vertices.get(a as usize), vertices.get(b as usize)) else {
            continue;
        };
        let arc = resolve_owner_arc(
            va,
            vb,
            generators[cells[0]],
            generators[cells[1]],
            ANTIPODAL_DOT_EPS as f64,
        )
        .ok();
        let denominator = (interior_samples_per_edge + 1) as f64;
        for k in 0..=interior_samples_per_edge + 1 {
            let t = k as f64 / denominator;
            let p = if k == 0 {
                va
            } else if k == interior_samples_per_edge + 1 {
                vb
            } else if let Some(arc) = arc {
                arc.sample(t)
            } else {
                (va * (1.0 - t) + vb * t).normalize()
            };
            if let Some((site_chord, radians)) =
                cross_track_radians(p, generators[cells[0]], generators[cells[1]])
            {
                values.push(site_chord, radians);
            }
        }
    }
    values.finish()
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
        let arc = resolve_owner_arc(
            va.as_dvec3(),
            vb.as_dvec3(),
            ga.as_dvec3(),
            gb.as_dvec3(),
            ANTIPODAL_DOT_EPS as f64,
        )
        .ok();

        for k in 1..=samples_per_edge {
            let t = k as f32 / (samples_per_edge + 1) as f32;
            let p = if let Some(arc) = arc {
                arc.sample(t as f64)
            } else {
                (va * (1.0 - t) + vb * t).normalize().as_dvec3()
            };
            residuals.push((p.dot(ga.as_dvec3()) - p.dot(gb.as_dvec3())).abs() as f32);
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
    let grid_key = |v: &crate::SpherePoint| -> (i32, i32, i32) {
        (
            (v.x() * inv_grid) as i32,
            (v.y() * inv_grid) as i32,
            (v.z() * inv_grid) as i32,
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
                            let d = (v.x() - other.x()).powi(2)
                                + (v.y() - other.y()).powi(2)
                                + (v.z() - other.z()).powi(2);
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
    scratch: &mut CubeMapGridScratch,
    batch: &mut Vec<u32>,
    query: Vec3,
) -> Option<usize> {
    grid.nearest_unrestricted_slot(query, scratch, batch)
        .map(|slot| grid.point_indices()[slot as usize] as usize)
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
    fn cross_track_formula_returns_angular_distance_to_bisector() {
        let a = DVec3::X;
        let b = DVec3::Y;
        let normal = (a - b).normalize();
        let delta = 2.5e-4_f64;
        let on_bisector = DVec3::Z;
        let p = (on_bisector * delta.cos() + normal * delta.sin()).normalize();
        let (site_chord, measured) = cross_track_radians(p, a, b).unwrap();
        assert!((site_chord - 2.0_f64.sqrt()).abs() < 1e-15);
        assert!((measured - delta).abs() < 1e-15);
        assert_eq!(cross_track_radians(on_bisector, a, a), None);
    }

    #[test]
    fn conditioning_buckets_are_disjoint_at_boundaries() {
        let mut values = ConditionedAngularValues::default();
        for (i, &upper) in SITE_CHORD_BUCKET_UPPERS[..4].iter().enumerate() {
            values.push(upper.next_down(), (i + 1) as f64);
            values.push(upper, (i + 11) as f64);
        }
        let stats = values.finish();
        assert_eq!(stats.overall.samples, 8);
        assert_eq!(stats.buckets[0].radians.samples, 1);
        assert_eq!(stats.buckets[1].radians.samples, 2);
        assert_eq!(stats.buckets[2].radians.samples, 2);
        assert_eq!(stats.buckets[3].radians.samples, 2);
        assert_eq!(stats.buckets[4].radians.samples, 1);
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
        assert!(
            report.vertex_cross_track_error.overall.max < 1e-5,
            "{}",
            report.headline()
        );
        assert!(
            report.edge_cross_track_error.overall.max < 1e-5,
            "{}",
            report.headline()
        );
    }

    #[test]
    fn near_pi_edge_sampling_uses_owner_plane() {
        let points = [
            UnitVec3::new(-0.346_064_27, -0.758_758, -0.551_838_64),
            UnitVec3::new(0.672_760_4, -0.217_307_08, -0.707_227_7),
            UnitVec3::new(-0.753_194_45, 0.368_890_3, 0.544_626_5),
            UnitVec3::new(-0.661_814_2, -0.681_742_25, -0.311_816_45),
        ];
        let diagram = compute(&points).expect("near-pi fixture should compute");
        let report = assess_with_config(
            &diagram,
            QualityConfig {
                max_sampled_cells: points.len(),
                edge_samples_per_edge: 4,
            },
        );
        assert!(
            report.edge_cross_track_error.overall.max < 1.0e-6,
            "{}",
            report.headline()
        );
        assert!(
            report.edge_dot_residuals.max_abs < 1.0e-6,
            "{}",
            report.headline()
        );
    }

    #[test]
    fn clustered_cap_quality_stays_bounded_under_default_preprocessing() {
        let stressed_points = clustered_cap_points(100, 0.0175);
        let stressed = compute(&stressed_points).expect("clustered cap should compute");
        let stressed_report = assess_with_config(
            &stressed,
            QualityConfig {
                max_sampled_cells: 64,
                edge_samples_per_edge: 3,
            },
        );

        assert_eq!(
            stressed_report.ownership.mismatches,
            0,
            "{}",
            stressed_report.headline()
        );
        assert!(
            stressed_report.vertex_dot_residuals.max_abs < 1e-3,
            "{}",
            stressed_report.headline()
        );
        assert!(
            stressed_report.edge_dot_residuals.max_abs < 1e-3,
            "{}",
            stressed_report.headline()
        );
        assert!(
            stressed_report.vertex_cross_track_error.overall.max < 1e-4,
            "{}",
            stressed_report.headline()
        );
        assert!(
            stressed_report.edge_cross_track_error.overall.max < 1e-4,
            "{}",
            stressed_report.headline()
        );
    }
}
