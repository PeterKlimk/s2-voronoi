//! Internal policy and heuristic configuration for neighbor search.
//!
//! This module centralizes the crate's behavior-preserving tuning knobs so the
//! query/stream code can depend on named policy decisions instead of scattered
//! constants.

/// Target mean points per query-grid cell.
///
/// Set from the 2026-06 reference-machine sweep (Ryzen 3600,
/// target-cpu=native, single-thread, uniform 100k/500k/2M): density 24 was
/// fastest at every size (4.8-7.1% over the previous 16), and the optimum is
/// flat across that range, so a constant suffices for now. Mean
/// neighbors-before-termination rises mildly with n (8.16 -> 8.44 from 100k
/// to 2M) and is density-independent; revisit beyond ~4M points or for
/// strongly non-uniform inputs. Use `S2_VORONOI_GRID_DENSITY` to override
/// for sweeps (scripts/sweep_grid_density.sh); the `neighbors_total` /
/// `grid_*` TIMING_KV fields are the model inputs.
pub(crate) const KNN_GRID_TARGET_DENSITY: f64 = 24.0;

/// Occupancy-feedback rebuild fires when the fullest grid cell exceeds this
/// multiple of the target density (clustered inputs produce mega-cells that
/// degrade candidate filtering quadratically).
pub(crate) const GRID_OCCUPANCY_REBUILD_FACTOR: f64 = 16.0;

/// Memory cap for the feedback rebuild: total grid cells stay O(n).
pub(crate) const GRID_MAX_CELLS_PER_POINT: f64 = 8.0;

/// Query-grid target density, with the sweep/tuning env override.
pub(crate) fn knn_grid_target_density() -> f64 {
    static OVERRIDE: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    OVERRIDE
        .get_or_init(|| {
            std::env::var("S2_VORONOI_GRID_DENSITY")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|d| *d >= 1.0)
        })
        .unwrap_or(KNN_GRID_TARGET_DENSITY)
}

/// Query-grid resolution for a point count at the target density.
pub(crate) fn knn_grid_resolution(num_points: usize) -> usize {
    let target = knn_grid_target_density().max(1.0);
    ((num_points as f64 / (6.0 * target)).sqrt() as usize).max(4)
}

/// Occupancy-feedback decision: given a built grid's max cell occupancy,
/// return a higher resolution to rebuild at, or `None` to keep the grid.
///
/// The new resolution aims the fullest cell back at half the rebuild
/// threshold (occupancy scales ~1/res^2 for a concentrated cluster), capped
/// by the memory budget. Single feedback step — concentration beyond what a
/// global resolution can fix within memory is left to the big-cell path.
pub(crate) fn grid_occupancy_rebuild_resolution(
    res: usize,
    num_points: usize,
    max_occupancy: usize,
) -> Option<usize> {
    let target = knn_grid_target_density().max(1.0);
    let threshold = GRID_OCCUPANCY_REBUILD_FACTOR * target;
    if (max_occupancy as f64) <= threshold {
        return None;
    }

    let desired_max = (threshold * 0.5).max(1.0);
    let scale = (max_occupancy as f64 / desired_max).sqrt();
    let new_res = ((res as f64 * scale).ceil() as usize).max(res + 1);

    let max_cells = (GRID_MAX_CELLS_PER_POINT * num_points as f64).max(6.0 * 16.0);
    let res_cap = ((max_cells / 6.0).sqrt() as usize).max(4);
    let new_res = new_res.min(res_cap);
    (new_res > res).then_some(new_res)
}

pub(crate) const DEFAULT_PACKED_CHUNK0_SIZE: usize = 16;
pub(crate) const DEFAULT_PACKED_CHUNK_SIZE: usize = 8;

/// Target size for the packed "hi" candidate list before selection gets expensive.
pub(crate) const PACKED_HI_BUDGET: usize = 32;
/// Count-model knob: ignore directed center eligibility when tightening packed thresholds.
pub(crate) const PACKED_COUNT_MODEL_IGNORE_DIRECTED_CENTER: bool = false;
/// Count-model knob: include same-bin-earlier cells when estimating packed candidate pressure.
pub(crate) const PACKED_COUNT_MODEL_INCLUDE_SAME_BIN_EARLIER: bool = false;
/// Hard cold-path cap for packed `r=2` expansion storage per query.
pub(crate) const PACKED_MAX_EXPAND_R2_CANDIDATES_PER_QUERY: usize = 8_192;

/// Policy decisions that affect packed neighbor sourcing before directed cursor fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedNeighborPolicy {
    chunk0_size: usize,
    chunk_size: usize,
    expand_r2_enabled: bool,
}

impl PackedNeighborPolicy {
    #[inline]
    pub(crate) fn for_point_count(num_points: usize, expand_r2_enabled: bool) -> Self {
        let max_neighbors = num_points.saturating_sub(1);
        Self {
            chunk0_size: DEFAULT_PACKED_CHUNK0_SIZE.min(max_neighbors),
            chunk_size: DEFAULT_PACKED_CHUNK_SIZE.min(max_neighbors),
            expand_r2_enabled,
        }
    }

    #[inline]
    pub(crate) fn enabled(self) -> bool {
        self.chunk0_size > 0
    }

    #[inline]
    pub(crate) fn chunk0_size(self) -> usize {
        self.chunk0_size
    }

    #[inline]
    pub(crate) fn chunk_size(self) -> usize {
        self.chunk_size
    }

    #[inline]
    pub(crate) fn expand_r2_enabled(self) -> bool {
        self.expand_r2_enabled
    }

    #[inline]
    pub(crate) fn scratch_chunk_capacity(self) -> usize {
        self.chunk0_size.max(self.chunk_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{knn_clipping::TerminationConfig, VoronoiConfig};

    #[test]
    fn grid_resolution_scales_with_point_count() {
        let small = knn_grid_resolution(100);
        let medium = knn_grid_resolution(100_000);
        let large = knn_grid_resolution(2_500_000);
        assert_eq!(small, 4, "resolution floor");
        assert!(medium > small);
        assert!(large > medium);
        // Mean density near the target for large n.
        let cells = 6.0 * (large as f64) * (large as f64);
        let density = 2_500_000.0 / cells;
        assert!(
            (density / KNN_GRID_TARGET_DENSITY - 1.0).abs() < 0.25,
            "mean density {density} should track the target"
        );
    }

    #[test]
    fn occupancy_rebuild_fires_only_above_threshold() {
        let target = knn_grid_target_density();
        let threshold = (GRID_OCCUPANCY_REBUILD_FACTOR * target) as usize;

        assert_eq!(
            grid_occupancy_rebuild_resolution(32, 100_000, target as usize),
            None
        );
        assert_eq!(
            grid_occupancy_rebuild_resolution(32, 100_000, threshold),
            None
        );

        let new_res = grid_occupancy_rebuild_resolution(32, 100_000, threshold * 16)
            .expect("16x over threshold must trigger a rebuild");
        assert!(new_res > 32);

        // Memory cap: total cells stay O(n).
        let capped = grid_occupancy_rebuild_resolution(32, 1_000, 1_000)
            .map(|r| 6 * r * r)
            .unwrap_or(0);
        assert!(
            capped as f64 <= (GRID_MAX_CELLS_PER_POINT * 1_000.0).max(96.0) * 1.1,
            "rebuild resolution must respect the memory budget, got {capped} cells"
        );
    }

    #[test]
    fn packed_policy_clamps_to_available_neighbors() {
        let zero = PackedNeighborPolicy::for_point_count(0, true);
        assert!(!zero.enabled());
        assert_eq!(zero.chunk0_size(), 0);
        assert_eq!(zero.chunk_size(), 0);

        let one = PackedNeighborPolicy::for_point_count(1, true);
        assert!(!one.enabled());
        assert_eq!(one.chunk0_size(), 0);
        assert_eq!(one.chunk_size(), 0);

        let small = PackedNeighborPolicy::for_point_count(5, true);
        assert!(small.enabled());
        assert_eq!(small.chunk0_size(), 4);
        assert_eq!(small.chunk_size(), 4);
        assert_eq!(small.scratch_chunk_capacity(), 4);

        let large = PackedNeighborPolicy::for_point_count(100, false);
        assert_eq!(large.chunk0_size(), 16);
        assert_eq!(large.chunk_size(), 8);
        assert_eq!(large.scratch_chunk_capacity(), 16);
        assert!(!large.expand_r2_enabled());
    }

    #[test]
    fn policy_defaults_match_public_and_internal_config_defaults() {
        let public = VoronoiConfig::default();
        let internal = TerminationConfig::default();
        let policy = internal.packed_policy(100);

        assert!(public.packed_knn_expand_r2);
        assert!(internal.packed_expand_r2);
        assert!(policy.expand_r2_enabled());
        assert_eq!(policy.chunk0_size(), DEFAULT_PACKED_CHUNK0_SIZE);
        assert_eq!(policy.chunk_size(), DEFAULT_PACKED_CHUNK_SIZE);
    }
}
