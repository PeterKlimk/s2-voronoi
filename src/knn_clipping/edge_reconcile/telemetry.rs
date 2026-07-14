//! Opt-in, behavior-preserving diagnostics for the primary reconciliation pass.
//!
//! This deliberately analyzes the pre-mutation state and then discards its
//! simulated union-find. It must never influence reconciliation policy or the
//! returned diagram. Enable with `VORONOI_MESH_RECONCILE_TELEMETRY=1`.

use std::collections::BTreeMap;
use std::time::Instant;

use rustc_hash::FxHashMap;

use super::{
    bound_merge_components, collect_merges, dist_sq, edge_segments_for_neighbor_into, unpack_edge,
    vertex_pos, MergeLedger, MergeMode, VertexKeys,
};
use crate::diagram::VoronoiCell;
use crate::knn_clipping::live_dedup::{
    EdgeRecord, UnresolvedEdgeMismatch, UnresolvedEdgeOrigin, VertexPosition,
};
use crate::knn_clipping::union_find::SparseUnionFind;

// Inclusive upper bounds for the histogram. The final bucket is +inf.
// Chord distances are on the canonical unit sphere in production.
const DISTANCE_BOUNDS: [f32; 12] = [
    0.0, 1.0e-8, 3.0e-8, 1.0e-7, 3.0e-7, 1.0e-6, 3.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1,
];
const EXACT_DIAMETER_COMPONENT_LIMIT: usize = 128;

#[derive(Clone, Debug)]
struct DistanceStats {
    count: usize,
    within_eps: usize,
    min: f32,
    max: f32,
    histogram: [usize; DISTANCE_BOUNDS.len() + 1],
}

impl Default for DistanceStats {
    fn default() -> Self {
        Self {
            count: 0,
            within_eps: 0,
            min: f32::INFINITY,
            max: 0.0,
            histogram: [0; DISTANCE_BOUNDS.len() + 1],
        }
    }
}

impl DistanceStats {
    fn record(&mut self, distance: f32, eps: f32) {
        self.count += 1;
        if distance <= eps {
            self.within_eps += 1;
        }
        self.min = self.min.min(distance);
        self.max = self.max.max(distance);
        let bin = DISTANCE_BOUNDS
            .iter()
            .position(|&bound| distance <= bound)
            .unwrap_or(DISTANCE_BOUNDS.len());
        self.histogram[bin] += 1;
    }

    fn min_or_zero(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.min
        }
    }

    fn histogram_text(&self) -> String {
        self.histogram
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("|")
    }
}

#[derive(Default)]
struct OriginStats {
    records: usize,
    irregular: usize,
    one_sided: usize,
    already_matched: usize,
    one_shared_endpoint: usize,
    distinct_endpoints: usize,
    inferred_pairing: DistanceStats,
}

#[derive(Default)]
struct ComponentStats {
    touched_vertices: usize,
    components: usize,
    max_component_size: usize,
    approximate_components: usize,
    // Exact for components up to the cap. For larger components this is a
    // two-sweep metric lower bound.
    max_diameter_lower: f32,
    // Exact for small components. For larger components, twice the maximum
    // distance from an arbitrary member is a metric upper bound.
    max_diameter_upper: f32,
}

#[derive(Default)]
struct PrimaryTelemetry {
    records: usize,
    unique_edge_keys: usize,
    irregular: usize,
    one_sided: usize,
    already_matched: usize,
    one_shared_endpoint: usize,
    distinct_endpoints: usize,
    sum_minimax_disagreements: usize,
    inferred_pairing: DistanceStats,
    proximity_evaluations: DistanceStats,
    one_sided_edge: DistanceStats,
    origins: BTreeMap<UnresolvedEdgeOrigin, OriginStats>,
    simulated_unions: usize,
    rejected_components: usize,
    components: ComponentStats,
}

/// Emit a machine-readable description of what the *current* primary pass
/// would consider on its first round. The real pass still runs normally after
/// this function returns. Analysis errors are reported but never propagated.
#[allow(clippy::too_many_arguments)] // mirrors the read-only reconciliation seam
pub(crate) fn emit_primary_reconcile_telemetry<P: VertexPosition>(
    unresolved: &[UnresolvedEdgeMismatch],
    vertices: &[P],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: VertexKeys<'_>,
    eps: f32,
) {
    if std::env::var_os("VORONOI_MESH_RECONCILE_TELEMETRY").is_none() {
        return;
    }

    let started = Instant::now();
    match analyze_primary(unresolved, vertices, cells, cell_indices, vertex_keys, eps) {
        Ok(stats) => emit_stats(&stats, eps, started.elapsed().as_secs_f64() * 1_000.0),
        Err(err) => {
            eprintln!(
                "RECONCILE_KV status=analysis_error records={} analysis_ms={:.6}",
                unresolved.len(),
                started.elapsed().as_secs_f64() * 1_000.0,
            );
            eprintln!("reconciliation telemetry analysis failed: {err}");
        }
    }
}

fn analyze_primary<P: VertexPosition>(
    unresolved: &[UnresolvedEdgeMismatch],
    vertices: &[P],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: VertexKeys<'_>,
    eps: f32,
) -> Result<PrimaryTelemetry, crate::VoronoiError> {
    let mut stats = PrimaryTelemetry {
        records: unresolved.len(),
        ..PrimaryTelemetry::default()
    };
    let mut keys: Vec<u64> = unresolved
        .iter()
        .map(|record| record.key.as_u64())
        .collect();
    keys.sort_unstable();
    keys.dedup();
    stats.unique_edge_keys = keys.len();

    let mut seg_a = Vec::new();
    let mut seg_b = Vec::new();
    for record in unresolved {
        let origin = stats.origins.entry(record.origin).or_default();
        origin.records += 1;
        let (a, b) = unpack_edge(record.key.as_u64());
        edge_segments_for_neighbor_into(a, b, cells, cell_indices, vertex_keys, &mut seg_a)?;
        edge_segments_for_neighbor_into(b, a, cells, cell_indices, vertex_keys, &mut seg_b)?;

        if seg_a.len() != 1 || seg_b.len() != 1 {
            stats.irregular += 1;
            origin.irregular += 1;
            record_proximity_distances(
                &seg_a,
                &seg_b,
                vertices,
                eps,
                &mut stats.proximity_evaluations,
            )?;
            if (seg_a.len() == 1 && seg_b.is_empty()) || (seg_b.len() == 1 && seg_a.is_empty()) {
                stats.one_sided += 1;
                origin.one_sided += 1;
                let (v0, v1) = if seg_a.len() == 1 { seg_a[0] } else { seg_b[0] };
                let distance = vertex_distance(vertices, v0, v1)?;
                stats.one_sided_edge.record(distance, eps);
            }
            continue;
        }

        let (a0, a1) = seg_a[0];
        let (b0, b1) = seg_b[0];
        let share_a0 = a0 == b0 || a0 == b1;
        let share_a1 = a1 == b0 || a1 == b1;
        if share_a0 && share_a1 {
            stats.already_matched += 1;
            origin.already_matched += 1;
            continue;
        }
        if share_a0 || share_a1 {
            stats.one_shared_endpoint += 1;
            origin.one_shared_endpoint += 1;
            let (unmatched_a, unmatched_b) = if a0 == b0 {
                (a1, b1)
            } else if a0 == b1 {
                (a1, b0)
            } else if a1 == b0 {
                (a0, b1)
            } else {
                (a0, b0)
            };
            let distance = vertex_distance(vertices, unmatched_a, unmatched_b)?;
            stats.inferred_pairing.record(distance, eps);
            origin.inferred_pairing.record(distance, eps);
            continue;
        }

        stats.distinct_endpoints += 1;
        origin.distinct_endpoints += 1;
        let d00a = vertex_distance_sq(vertices, a0, b0)?;
        let d00b = vertex_distance_sq(vertices, a1, b1)?;
        let d01a = vertex_distance_sq(vertices, a0, b1)?;
        let d01b = vertex_distance_sq(vertices, a1, b0)?;
        let sum_00 = d00a + d00b;
        let sum_01 = d01a + d01b;
        let max_00 = d00a.max(d00b);
        let max_01 = d01a.max(d01b);
        let current_max_sq = if sum_00 <= sum_01 { max_00 } else { max_01 };
        let minimax_uses_00 = max_00 <= max_01;
        if (sum_00 <= sum_01) != minimax_uses_00 {
            stats.sum_minimax_disagreements += 1;
        }
        let distance = current_max_sq.sqrt();
        stats.inferred_pairing.record(distance, eps);
        origin.inferred_pairing.record(distance, eps);
    }

    if !unresolved.is_empty() {
        let records: Vec<EdgeRecord> = unresolved
            .iter()
            .map(|record| EdgeRecord { key: record.key })
            .collect();
        let (mut proposed, _) = collect_merges(
            &records,
            vertices,
            cells,
            cell_indices,
            vertex_keys,
            eps,
            MergeMode::Primary,
            true,
        )?;
        let (mut uf, simulated_unions, rejected_components) = bound_merge_components(
            &mut proposed,
            vertices,
            cells,
            cell_indices,
            &mut MergeLedger::default(),
            eps,
        )?;
        stats.simulated_unions = simulated_unions;
        stats.rejected_components = rejected_components.len();
        stats.components = component_stats(&mut uf, vertices)?;
    }

    Ok(stats)
}

fn record_proximity_distances<P: VertexPosition>(
    seg_a: &[(u32, u32)],
    seg_b: &[(u32, u32)],
    vertices: &[P],
    eps: f32,
    stats: &mut DistanceStats,
) -> Result<(), crate::VoronoiError> {
    let mut ids = Vec::with_capacity((seg_a.len() + seg_b.len()) * 2);
    for &(v0, v1) in seg_a.iter().chain(seg_b.iter()) {
        ids.push(v0);
        ids.push(v1);
    }
    ids.sort_unstable();
    ids.dedup();
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            stats.record(vertex_distance(vertices, ids[i], ids[j])?, eps);
        }
    }
    Ok(())
}

fn component_stats<P: VertexPosition>(
    uf: &mut SparseUnionFind,
    vertices: &[P],
) -> Result<ComponentStats, crate::VoronoiError> {
    let touched = uf.touched_ids();
    let mut groups: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for &id in &touched {
        groups.entry(uf.find(id)).or_default().push(id);
    }
    let mut stats = ComponentStats {
        touched_vertices: touched.len(),
        components: groups.len(),
        ..ComponentStats::default()
    };
    for ids in groups.values() {
        stats.max_component_size = stats.max_component_size.max(ids.len());
        let (lower, upper, exact) = component_diameter_bounds(ids, vertices)?;
        stats.max_diameter_lower = stats.max_diameter_lower.max(lower);
        stats.max_diameter_upper = stats.max_diameter_upper.max(upper);
        if !exact {
            stats.approximate_components += 1;
        }
    }
    Ok(stats)
}

fn component_diameter_bounds<P: VertexPosition>(
    ids: &[u32],
    vertices: &[P],
) -> Result<(f32, f32, bool), crate::VoronoiError> {
    if ids.len() <= 1 {
        return Ok((0.0, 0.0, true));
    }
    if ids.len() <= EXACT_DIAMETER_COMPONENT_LIMIT {
        let mut diameter_sq: f32 = 0.0;
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                diameter_sq = diameter_sq.max(vertex_distance_sq(vertices, ids[i], ids[j])?);
            }
        }
        let diameter = diameter_sq.sqrt();
        return Ok((diameter, diameter, true));
    }

    let pivot = ids[0];
    let mut farthest = pivot;
    let mut radius_sq: f32 = 0.0;
    for &id in &ids[1..] {
        let d = vertex_distance_sq(vertices, pivot, id)?;
        if d > radius_sq {
            radius_sq = d;
            farthest = id;
        }
    }
    let mut lower_sq: f32 = 0.0;
    for &id in ids {
        lower_sq = lower_sq.max(vertex_distance_sq(vertices, farthest, id)?);
    }
    Ok((lower_sq.sqrt(), 2.0 * radius_sq.sqrt(), false))
}

fn vertex_distance<P: VertexPosition>(
    vertices: &[P],
    a: u32,
    b: u32,
) -> Result<f32, crate::VoronoiError> {
    Ok(vertex_distance_sq(vertices, a, b)?.sqrt())
}

fn vertex_distance_sq<P: VertexPosition>(
    vertices: &[P],
    a: u32,
    b: u32,
) -> Result<f32, crate::VoronoiError> {
    Ok(dist_sq(vertex_pos(vertices, a)?, vertex_pos(vertices, b)?))
}

fn emit_stats(stats: &PrimaryTelemetry, eps: f32, analysis_ms: f64) {
    eprintln!(
        "RECONCILE_KV status=ok records={} unique_keys={} eps={eps:.9e} inferred_gate=epsilon component_gate=diameter irregular={} \
         one_sided={} already_matched={} one_shared={} distinct_endpoints={} \
         inferred_count={} inferred_within_eps={} inferred_over_eps={} \
         inferred_min={:.9e} inferred_max={:.9e} inferred_hist={} \
         proximity_evals={} proximity_within_eps={} proximity_max={:.9e} \
         one_sided_edge_count={} one_sided_edge_within_eps={} one_sided_edge_max={:.9e} \
         sum_minimax_disagree={} simulated_unions={} rejected_components={} touched_vertices={} components={} \
         max_component_size={} approximate_components={} component_diameter_lower={:.9e} \
         component_diameter_upper={:.9e} analysis_ms={analysis_ms:.6}",
        stats.records,
        stats.unique_edge_keys,
        stats.irregular,
        stats.one_sided,
        stats.already_matched,
        stats.one_shared_endpoint,
        stats.distinct_endpoints,
        stats.inferred_pairing.count,
        stats.inferred_pairing.within_eps,
        stats.inferred_pairing.count - stats.inferred_pairing.within_eps,
        stats.inferred_pairing.min_or_zero(),
        stats.inferred_pairing.max,
        stats.inferred_pairing.histogram_text(),
        stats.proximity_evaluations.count,
        stats.proximity_evaluations.within_eps,
        stats.proximity_evaluations.max,
        stats.one_sided_edge.count,
        stats.one_sided_edge.within_eps,
        stats.one_sided_edge.max,
        stats.sum_minimax_disagreements,
        stats.simulated_unions,
        stats.rejected_components,
        stats.components.touched_vertices,
        stats.components.components,
        stats.components.max_component_size,
        stats.components.approximate_components,
        stats.components.max_diameter_lower,
        stats.components.max_diameter_upper,
    );
    eprintln!(
        "RECONCILE_HIST_KV bounds=0|1e-8|3e-8|1e-7|3e-7|1e-6|3e-6|1e-5|1e-4|1e-3|1e-2|1e-1|inf"
    );
    for (origin, origin_stats) in &stats.origins {
        eprintln!(
            "RECONCILE_ORIGIN_KV origin={origin:?} records={} irregular={} one_sided={} \
             already_matched={} one_shared={} distinct_endpoints={} inferred_count={} \
             inferred_within_eps={} inferred_over_eps={} inferred_min={:.9e} \
             inferred_max={:.9e} inferred_hist={}",
            origin_stats.records,
            origin_stats.irregular,
            origin_stats.one_sided,
            origin_stats.already_matched,
            origin_stats.one_shared_endpoint,
            origin_stats.distinct_endpoints,
            origin_stats.inferred_pairing.count,
            origin_stats.inferred_pairing.within_eps,
            origin_stats.inferred_pairing.count - origin_stats.inferred_pairing.within_eps,
            origin_stats.inferred_pairing.min_or_zero(),
            origin_stats.inferred_pairing.max,
            origin_stats.inferred_pairing.histogram_text(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knn_clipping::live_dedup::EdgeKey;
    use glam::Vec3;

    #[test]
    fn distance_histogram_is_inclusive_at_reconcile_epsilon() {
        let mut stats = DistanceStats::default();
        for distance in [0.0, 1.0e-7, 1.0e-6, 1.1e-6] {
            stats.record(distance, 1.0e-6);
        }
        assert_eq!(stats.count, 4);
        assert_eq!(stats.within_eps, 3);
        assert_eq!(stats.histogram.iter().sum::<usize>(), 4);
        assert_eq!(stats.max, 1.1e-6);
    }

    #[test]
    fn component_diameter_detects_transitive_span() {
        let vertices = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.75e-6, 0.0, 0.0),
            Vec3::new(1.5e-6, 0.0, 0.0),
        ];
        let mut uf = SparseUnionFind::new();
        assert!(uf.union(0, 1));
        assert!(uf.union(1, 2));
        let stats = component_stats(&mut uf, &vertices).expect("component analysis");
        assert_eq!(stats.components, 1);
        assert_eq!(stats.max_component_size, 3);
        assert!((stats.max_diameter_lower - 1.5e-6).abs() < 1.0e-12);
        assert_eq!(stats.max_diameter_lower, stats.max_diameter_upper);
    }

    #[test]
    fn primary_analysis_measures_disjoint_endpoint_matching_by_origin() {
        let vertices = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0 + 1.0e-5, 2.0e-6, 0.0),
            Vec3::new(2.0e-6, 1.0 + 1.0e-5, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
        ];
        let vertex_keys = [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [0, 1, 4],
            [0, 1, 5],
            [1, 4, 5],
        ];
        let cells = [VoronoiCell::new(0, 3), VoronoiCell::new(3, 3)];
        let cell_indices = [0, 1, 2, 3, 4, 5];
        let records = [UnresolvedEdgeMismatch {
            key: EdgeKey::from(1_u64 << 32),
            origin: UnresolvedEdgeOrigin::CrossBinThirdsMismatch,
        }];

        let stats = analyze_primary(
            &records,
            &vertices,
            &cells,
            &cell_indices,
            VertexKeys::Flat(&vertex_keys),
            1.0e-6,
        )
        .expect("telemetry analysis");

        assert_eq!(stats.distinct_endpoints, 1);
        assert_eq!(stats.inferred_pairing.count, 1);
        assert_eq!(stats.inferred_pairing.within_eps, 0);
        assert!(stats.inferred_pairing.max > 1.0e-6);
        let origin = &stats.origins[&UnresolvedEdgeOrigin::CrossBinThirdsMismatch];
        assert_eq!(origin.records, 1);
        assert_eq!(origin.distinct_endpoints, 1);
        assert_eq!(origin.inferred_pairing.count, 1);
    }
}
