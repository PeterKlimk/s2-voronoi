//! Compute entry points for GPU-style Voronoi construction.

use glam::Vec3;

use super::debug;
use super::edge_repair;
use super::live_dedup;
use super::timing::{Timer, TimingBuilder};
use super::{
    constants, log_enabled, merge_close_points, CubeMapGridKnn, MergeResult, TerminationConfig,
};

pub(super) fn compute_voronoi_gpu_style_core(
    points: &[Vec3],
    termination: TerminationConfig,
    skip_preprocess: bool,
) -> crate::SphericalVoronoi {
    let mut tb = TimingBuilder::new();

    // Preprocessing: merge close points
    let t = Timer::start();
    let (effective_points, merge_result): (Vec<Vec3>, Option<MergeResult>) = if skip_preprocess {
        (points.to_vec(), None)
    } else {
        let threshold = std::env::var("S2V_PREPROCESS_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or_else(|| constants::merge_threshold_for_density(points.len()));
        let result = merge_close_points(points, threshold);
        let pts = if result.num_merged > 0 {
            result.effective_points.clone()
        } else {
            points.to_vec()
        };
        (pts, Some(result))
    };
    tb.set_preprocess(t.elapsed());
    let needs_remap = merge_result.as_ref().map_or(false, |r| r.num_merged > 0);
    if let Some(r) = &merge_result {
        if r.num_merged > 0 && log_enabled() {
            eprintln!(
                "preprocess: merged {} close generators ({} -> {})",
                r.num_merged,
                points.len(),
                r.effective_points.len()
            );
        }
    }

    // Build KNN on effective points (this is the timed grid build)
    let t = Timer::start();
    let knn = CubeMapGridKnn::new(&effective_points);
    tb.set_knn_build(t.elapsed());
    #[cfg(feature = "timing")]
    tb.set_knn_build_sub(knn.grid_build_timings().clone());

    // Build cells using sharded live dedup
    let t = Timer::start();
    let sharded = live_dedup::build_cells_sharded_live_dedup(&effective_points, &knn, termination);
    tb.set_cell_construction(t.elapsed(), sharded.cell_sub.clone().into_sub_phases());

    let t = Timer::start();
    let (all_vertices, all_vertex_keys, bad_edges, eff_cells, eff_cell_indices, dedup_sub) =
        live_dedup::assemble_sharded_live_dedup(sharded);
    tb.set_dedup(t.elapsed(), dedup_sub);

    let debug_pre: usize = std::env::var("S2V_DEBUG_LOW_DEGREE_PRE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    debug::debug_low_degree_vertices(
        "pre-repair",
        &effective_points,
        Some(&knn),
        &all_vertices,
        &all_vertex_keys,
        &eff_cells,
        &eff_cell_indices,
        debug_pre,
    );

    if log_enabled() && !bad_edges.is_empty() {
        eprintln!("edge checks (live): bad_edges={}", bad_edges.len());
        for record in &bad_edges {
            let (a, b) = edge_repair::unpack_edge(record.key.as_u64());
            eprintln!("  edge=({},{}) reason={:?}", a, b, record.reason);
        }
    }

    let repair_edges_storage: Vec<live_dedup::EdgeRecord> = bad_edges
        .iter()
        .map(|b| live_dedup::EdgeRecord { key: b.key })
        .collect();
    if log_enabled() {
        eprintln!(
            "edge repair: using live bad edges (count={})",
            repair_edges_storage.len()
        );
    }

    let t = Timer::start();
    let (eff_cells, eff_cell_indices) = if let Some((cells, indices)) =
        edge_repair::repair_bad_edges(
            &repair_edges_storage,
            &all_vertices,
            &eff_cells,
            &eff_cell_indices,
            &all_vertex_keys,
        ) {
        (cells, indices)
    } else {
        (eff_cells, eff_cell_indices)
    };
    tb.set_edge_repair(t.elapsed());

    let debug_post: usize = std::env::var("S2V_DEBUG_LOW_DEGREE_POST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    debug::debug_low_degree_vertices(
        "post-repair",
        &effective_points,
        Some(&knn),
        &all_vertices,
        &all_vertex_keys,
        &eff_cells,
        &eff_cell_indices,
        debug_post,
    );

    // Remap cells back to original point indices if we merged
    let t = Timer::start();
    let (cells, cell_indices) = if needs_remap {
        use crate::VoronoiCell;
        let merge_result = merge_result.as_ref().unwrap();

        // Each original point maps to an effective point's cell
        let mut new_cells = Vec::with_capacity(points.len());
        let mut new_cell_indices: Vec<u32> = Vec::new();

        for orig_idx in 0..points.len() {
            let eff_idx = merge_result.original_to_effective[orig_idx];
            let eff_cell = &eff_cells[eff_idx];

            let start = u32::try_from(new_cell_indices.len())
                .expect("cell index buffer exceeds u32 capacity");
            let eff_start = eff_cell.vertex_start();
            let eff_end = eff_start + eff_cell.vertex_count();
            new_cell_indices.extend_from_slice(&eff_cell_indices[eff_start..eff_end]);

            let count_u16 =
                u16::try_from(eff_cell.vertex_count()).expect("cell vertex count exceeds u16");
            new_cells.push(VoronoiCell::new(start, count_u16));
        }
        (new_cells, new_cell_indices)
    } else {
        (eff_cells, eff_cell_indices)
    };

    let voronoi =
        crate::SphericalVoronoi::from_raw_parts(points.to_vec(), all_vertices, cells, cell_indices);
    tb.set_assemble(t.elapsed());

    // Report timing if feature enabled
    let timings = tb.finish();
    timings.report(points.len());

    voronoi
}

/// Compute spherical Voronoi diagram using the GPU-style algorithm.
///
/// Uses adaptive k-NN (12→24→48→full) with early termination.
///
/// Timing output is controlled by the `timing` feature flag:
/// ```bash
/// cargo run --release --features timing
/// ```
pub fn compute_voronoi_gpu_style(points: &[Vec3]) -> crate::SphericalVoronoi {
    let termination = TerminationConfig::default();
    compute_voronoi_gpu_style_core(points, termination, false)
}

/// Compute spherical Voronoi with custom termination config (for benchmarks).
#[allow(dead_code)]
pub fn compute_voronoi_gpu_style_with_termination(
    points: &[Vec3],
    termination: TerminationConfig,
) -> crate::SphericalVoronoi {
    compute_voronoi_gpu_style_core(points, termination, false)
}

/// Compute spherical Voronoi WITHOUT preprocessing (merge close points).
/// For benchmarking only - assumes points are already well-spaced.
pub fn compute_voronoi_gpu_style_no_preprocess(points: &[Vec3]) -> crate::SphericalVoronoi {
    let termination = TerminationConfig::default();
    compute_voronoi_gpu_style_core(points, termination, true)
}
