//! Internal policy and heuristic configuration for neighbor search.
//!
//! This module centralizes the crate's behavior-preserving tuning knobs so the
//! query/stream code can depend on named policy decisions instead of scattered
//! constants.

/// Target mean points per query-grid cell.
///
/// HONESTY NOTE: this value is tuned for large uniform inputs (the original
/// 2.5M-point target) and the optimum is known to vary with point count and
/// distribution — denser inputs need more neighbors before termination, so
/// the right density is a curve over n, not a constant (docs/todo.md P3.2).
/// Use `S2_VORONOI_GRID_DENSITY` to override for sweeps; the
/// `neighbors_total`/`neighbors_max` TIMING_KV fields provide the model
/// inputs for fitting the curve.
pub(crate) const KNN_GRID_TARGET_DENSITY: f64 = 16.0;

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
pub(crate) const DEFAULT_TERMINATION_CHECK_START: usize = 8;
pub(crate) const DEFAULT_TERMINATION_CHECK_STEP: usize = 1;

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

/// Policy decisions for directed termination cadence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TerminationPolicy {
    check_start: usize,
    check_step: usize,
    max_k_cap: Option<usize>,
}

impl Default for TerminationPolicy {
    fn default() -> Self {
        Self {
            check_start: DEFAULT_TERMINATION_CHECK_START,
            check_step: DEFAULT_TERMINATION_CHECK_STEP,
            max_k_cap: None,
        }
    }
}

impl TerminationPolicy {
    #[inline]
    pub(crate) fn new(check_start: usize, check_step: usize, max_k_cap: Option<usize>) -> Self {
        Self {
            check_start,
            check_step,
            max_k_cap,
        }
    }

    #[inline]
    pub(crate) fn should_check(self, neighbors_processed: usize) -> bool {
        self.check_step > 0
            && neighbors_processed >= self.check_start
            && (neighbors_processed - self.check_start).is_multiple_of(self.check_step)
    }

    #[inline]
    pub(crate) fn max_k_cap(self) -> Option<usize> {
        self.max_k_cap
    }
}

/// Combined neighbor-query policy derived from public/internal config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KnnPolicy {
    packed: PackedNeighborPolicy,
    termination: TerminationPolicy,
}

impl KnnPolicy {
    #[inline]
    pub(crate) fn new(packed: PackedNeighborPolicy, termination: TerminationPolicy) -> Self {
        Self {
            packed,
            termination,
        }
    }

    #[inline]
    pub(crate) fn for_point_count(
        num_points: usize,
        termination: TerminationPolicy,
        expand_r2_enabled: bool,
    ) -> Self {
        Self::new(
            PackedNeighborPolicy::for_point_count(num_points, expand_r2_enabled),
            termination,
        )
    }

    #[inline]
    pub(crate) fn packed(self) -> PackedNeighborPolicy {
        self.packed
    }

    #[inline]
    pub(crate) fn termination(self) -> TerminationPolicy {
        self.termination
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
    fn termination_policy_uses_expected_cadence() {
        let policy = TerminationPolicy::new(8, 3, None);
        for n in 0..8 {
            assert!(!policy.should_check(n));
        }
        assert!(policy.should_check(8));
        assert!(!policy.should_check(9));
        assert!(!policy.should_check(10));
        assert!(policy.should_check(11));
        assert!(policy.should_check(14));
    }

    #[test]
    fn knn_policy_keeps_packed_and_termination_together() {
        let termination = TerminationPolicy::new(12, 2, Some(64));
        let policy = KnnPolicy::for_point_count(20, termination, true);
        assert_eq!(policy.packed().chunk0_size(), 16);
        assert_eq!(policy.packed().chunk_size(), 8);
        assert!(policy.packed().expand_r2_enabled());
        assert_eq!(policy.termination().max_k_cap(), Some(64));
        assert!(policy.termination().should_check(12));
        assert!(policy.termination().should_check(14));
        assert!(!policy.termination().should_check(13));
    }

    #[test]
    fn policy_defaults_match_public_and_internal_config_defaults() {
        let public = VoronoiConfig::default();
        let internal = TerminationConfig::default();
        let policy = KnnPolicy::for_point_count(
            100,
            internal.termination_policy(),
            public.packed_knn_expand_r2,
        );

        assert!(public.packed_knn_expand_r2);
        assert!(policy.packed().expand_r2_enabled());
        assert_eq!(policy.packed().chunk0_size(), DEFAULT_PACKED_CHUNK0_SIZE);
        assert_eq!(policy.packed().chunk_size(), DEFAULT_PACKED_CHUNK_SIZE);
        assert!(policy
            .termination()
            .should_check(DEFAULT_TERMINATION_CHECK_START));
        assert_eq!(policy.termination().max_k_cap(), None);
    }
}
