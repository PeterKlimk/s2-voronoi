//! Internal behavior and performance policy configuration.
//!
//! This module centralizes the crate's behavior-preserving tuning knobs so
//! implementation code can depend on named policy decisions instead of
//! scattered constants.

/// Realized normal-direction joggle used by `DegenerateMode::PerturbCoplanar`.
///
/// This dimensionless `f64` coefficient scales a stable signed value before
/// adding the plane normal and renormalizing the point. Its maximum magnitude
/// is approximately the angular offset in radians. This is an output-changing
/// robust-mode policy, not a coplanarity acceptance tolerance.
pub(crate) const COPLANAR_PERTURBATION_SCALE: f64 = 1.0e-2;

/// Generator-z boundary below which the gnomonic tangent-basis builder uses
/// its explicit south-pole branch.
///
/// This `f64` conditioning policy avoids the `1 + z` denominator becoming too
/// small in the general closed-form basis. The alternate branch is selected
/// for `z <` this value; equality retains the general formula.
pub(crate) const GNOMONIC_TANGENT_BASIS_SOUTH_POLE_SWITCH_Z: f64 = -0.999_999_9;

/// Initial half-extent of the gnomonic chart's synthetic bounding square.
///
/// This dimensionless `f64` construction policy supplies a large finite seed
/// envelope for half-plane clipping. Projection-limit handoff and cell
/// constraints, not this synthetic square, determine accepted final geometry.
pub(crate) const GNOMONIC_INITIAL_BOUNDING_EXTENT: f64 = 1e6;

/// Absolute x-component below which an `f64` direction uses the X axis as a
/// well-separated reference; otherwise it uses Y. This is a deterministic
/// basis-conditioning policy, not a geometric acceptance tolerance.
pub(crate) const REFERENCE_AXIS_COMPONENT_SWITCH_F64: f64 = 0.9;

/// Target mean generators per point-locator grid cell.
///
/// This `f64` lookup-policy value is intentionally independent of the kNN
/// construction grid's tuned density and environment override.
pub(crate) const LOCATOR_GRID_TARGET_DENSITY: f64 = 16.0;

/// Target mean points per query-grid cell.
///
/// Density 24 is the calibrated default; use `VORONOI_MESH_GRID_DENSITY` to
/// override it for sweeps (`scripts/sweep_grid_density.sh`). Revisit the
/// policy for substantially larger or strongly non-uniform inputs. Calibration
/// is recorded in `docs/performance.md#source-pinned-performance-decisions`.
pub(crate) const KNN_GRID_TARGET_DENSITY: f64 = 24.0;

/// Largest cube-face resolution whose `6 * res²` cells fit the grid's `u32`
/// cell identifiers.
pub(crate) const MAX_GRID_RESOLUTION: usize = 26_754;

/// Occupancy-feedback rebuild fires when the candidate-scan work proxy
/// `Σocc²/n` exceeds this. `Σocc²` is the sum of squared per-cell
/// occupancies — exactly the O(occ²) cost of scanning each cell's candidates
/// for every query homed in it, i.e. the cost of NOT rebuilding. Dividing by
/// `n` makes it scale-free: it equals the target density (~24) for uniform
/// input and rises with concentration.
///
/// `Σocc²/n` is the right *variable*: it weights by total candidate-scan work,
/// so a single giant cell in a uniform sea stays low (rebuild correctly
/// skipped — that's a local-index problem) while genuine concentration drives
/// it high. The earlier `max_occ > 16×target` trigger fired far too eagerly
/// (re-grids on any cell over the bar, even when not rebuilding is faster).
///
/// The calibrated threshold is above ordinary uniform/gradient occupancy and
/// below the concentration regimes that benefit from rebuilding. Measurement
/// history is in `docs/performance.md#source-pinned-performance-decisions`.
pub(crate) const GRID_REBUILD_SUMSQ_PER_N: f64 = 500.0;

/// Post-rebuild target for the fullest cell (drives the new resolution):
/// `new_res = res · sqrt(max_occ / this)`. This leaves headroom below the
/// feedback trigger after rebuilding.
pub(crate) const GRID_REBUILD_TARGET_MAX_OCC: f64 = 192.0;

/// Memory cap for the feedback rebuild: total grid cells stay O(n).
pub(crate) const GRID_MAX_CELLS_PER_POINT: f64 = 8.0;

/// Per-cell occupancy above which a dense-cell sub-index ("punch 1") is worth
/// building for that cell — the linear-scan vs sub-index crossover (distinct
/// from the rebuild's Σocc²/n trigger). Placeholder pending the quiet-box
/// calibration.
pub(crate) const DENSE_CELL_THRESHOLD: usize = 512;

/// Target nearest-neighbor count the dense-cell band gather aims to capture per
/// query. Sizes the band radius (`r ≈ (diag/2)·sqrt(target/occ)`) and hence the
/// completeness bound; the shell takeover backstops queries whose cell needs
/// neighbors beyond the band, so this only trades band width (work per query)
/// against takeover frequency. Set comfortably above the typical
/// neighbors-before-termination (~8) so takeover stays rare.
pub(crate) const DENSE_BAND_TARGET_COUNT: usize = 128;

/// Fractional expansion applied to the dense-cell gather radius.
///
/// This dimensionless `f32` pad makes the gathered band a conservative
/// superset of the claimed chord-radius band despite floating-point error.
/// False positives only add candidates; the exact claim radius still defines
/// the strict `dot > band_bound` coverage boundary.
pub(crate) const DENSE_BAND_RADIUS_INFLATION: f32 = 1e-3;

/// Query-grid target density, with the sweep/tuning env override.
pub(crate) fn knn_grid_target_density() -> f64 {
    static OVERRIDE: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    OVERRIDE
        .get_or_init(|| {
            std::env::var("VORONOI_MESH_GRID_DENSITY")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|d| *d >= 1.0)
        })
        .unwrap_or(KNN_GRID_TARGET_DENSITY)
}

/// Query-grid resolution for a point count at the target density.
pub(crate) fn knn_grid_resolution(num_points: usize) -> usize {
    let target = knn_grid_target_density().max(1.0);
    ((num_points as f64 / (6.0 * target)).sqrt() as usize).clamp(4, MAX_GRID_RESOLUTION)
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
    sum_sq_per_n: f64,
) -> Option<usize> {
    // Fire only in the catastrophic-concentration regime where NOT rebuilding
    // is infeasible (see GRID_REBUILD_SUMSQ_PER_N). Below it, the flat grid
    // degrades gracefully and a global re-grid is a net pessimization.
    if sum_sq_per_n <= GRID_REBUILD_SUMSQ_PER_N {
        return None;
    }

    let desired_max = GRID_REBUILD_TARGET_MAX_OCC.max(1.0);
    let scale = (max_occupancy as f64 / desired_max).sqrt();
    let new_res = ((res as f64 * scale).ceil() as usize).max(res + 1);

    let max_cells = (GRID_MAX_CELLS_PER_POINT * num_points as f64).max(6.0 * 16.0);
    let res_cap = ((max_cells / 6.0).sqrt() as usize).clamp(4, MAX_GRID_RESOLUTION);
    let new_res = new_res.min(res_cap);
    (new_res > res).then_some(new_res)
}

pub(crate) const DEFAULT_PACKED_CHUNK0_SIZE: usize = 16;
pub(crate) const DEFAULT_PACKED_CHUNK_SIZE: usize = 8;

/// Target size for the packed "hi" candidate list before selection gets expensive.
pub(crate) const PACKED_HI_BUDGET: usize = 32;
/// After occupancy feedback has selected a finer spatial grid, large same-cell
/// groups amortize a wider eager prefix across many queries. Tie both sides of
/// that trade to the existing dense-band work unit instead of adding an
/// unrelated distribution-specific cutoff.
pub(crate) const CONCENTRATED_PACKED_HI_MIN_QUERIES: usize = DENSE_BAND_TARGET_COUNT;
pub(crate) const CONCENTRATED_PACKED_HI_BUDGET: usize = 2 * DENSE_BAND_TARGET_COUNT;
/// Count-model knob: ignore directed center eligibility when tightening packed thresholds.
pub(crate) const PACKED_COUNT_MODEL_IGNORE_DIRECTED_CENTER: bool = false;
/// Count-model knob: include same-bin-earlier cells when estimating packed candidate pressure.
pub(crate) const PACKED_COUNT_MODEL_INCLUDE_SAME_BIN_EARLIER: bool = false;

/// Policy decisions that affect packed neighbor sourcing before directed cursor fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedNeighborPolicy {
    chunk0_size: usize,
    chunk_size: usize,
    hi_budget: usize,
    hi_min_queries: usize,
}

impl PackedNeighborPolicy {
    #[inline]
    pub(crate) fn for_point_count(num_points: usize) -> Self {
        let max_neighbors = num_points.saturating_sub(1);
        Self {
            chunk0_size: DEFAULT_PACKED_CHUNK0_SIZE.min(max_neighbors),
            chunk_size: DEFAULT_PACKED_CHUNK_SIZE.min(max_neighbors),
            hi_budget: PACKED_HI_BUDGET,
            hi_min_queries: usize::MAX,
        }
    }

    #[inline]
    pub(crate) fn after_occupancy_rebuild(mut self) -> Self {
        self.hi_budget = CONCENTRATED_PACKED_HI_BUDGET;
        self.hi_min_queries = CONCENTRATED_PACKED_HI_MIN_QUERIES;
        self
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
    pub(crate) fn scratch_chunk_capacity(self) -> usize {
        self.chunk0_size.max(self.chunk_size)
    }

    #[inline]
    pub(crate) fn hi_budget(self) -> usize {
        self.hi_budget
    }

    #[inline]
    pub(crate) fn hi_min_queries(self) -> usize {
        self.hi_min_queries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn occupancy_rebuild_fires_only_when_work_is_catastrophic() {
        // The trigger is Σocc²/n, NOT max occupancy: a single big cell (high
        // max_occ, low Σocc²/n) must NOT fire — that is a local-index problem,
        // and a global re-grid would de-tune the whole grid (measured harmful).
        let below = GRID_REBUILD_SUMSQ_PER_N * 0.5;
        let above = GRID_REBUILD_SUMSQ_PER_N * 2.0;

        // Below the work threshold: no rebuild, even with a fat fullest cell.
        assert_eq!(
            grid_occupancy_rebuild_resolution(32, 100_000, 5_000, below),
            None,
            "high max_occ but low Σocc²/n must not rebuild (single-giant-cell case)"
        );
        // At the threshold: still no rebuild (strict >).
        assert_eq!(
            grid_occupancy_rebuild_resolution(32, 100_000, 5_000, GRID_REBUILD_SUMSQ_PER_N),
            None
        );
        // Above it: rebuild to a finer resolution.
        let new_res = grid_occupancy_rebuild_resolution(32, 100_000, 20_000, above)
            .expect("catastrophic Σocc²/n must trigger a rebuild");
        assert!(new_res > 32);

        // Memory cap: total cells stay O(n) even for an extreme concentration.
        let capped = grid_occupancy_rebuild_resolution(32, 1_000, 1_000, above)
            .map(|r| 6 * r * r)
            .unwrap_or(0);
        assert!(
            capped as f64 <= (GRID_MAX_CELLS_PER_POINT * 1_000.0).max(96.0) * 1.1,
            "rebuild resolution must respect the memory budget, got {capped} cells"
        );
    }

    #[test]
    fn grid_resolutions_fit_u32_cell_ids() {
        let max = MAX_GRID_RESOLUTION as u64;
        assert!(6 * max * max <= u32::MAX as u64);
        assert!(6 * (max + 1) * (max + 1) > u32::MAX as u64);
        assert_eq!(knn_grid_resolution(usize::MAX), MAX_GRID_RESOLUTION);
        assert_eq!(
            grid_occupancy_rebuild_resolution(
                4,
                usize::MAX,
                usize::MAX,
                GRID_REBUILD_SUMSQ_PER_N * 2.0,
            ),
            Some(MAX_GRID_RESOLUTION)
        );
    }

    #[test]
    fn packed_policy_clamps_to_available_neighbors() {
        let zero = PackedNeighborPolicy::for_point_count(0);
        assert!(!zero.enabled());
        assert_eq!(zero.chunk0_size(), 0);
        assert_eq!(zero.chunk_size(), 0);

        let one = PackedNeighborPolicy::for_point_count(1);
        assert!(!one.enabled());
        assert_eq!(one.chunk0_size(), 0);
        assert_eq!(one.chunk_size(), 0);

        let small = PackedNeighborPolicy::for_point_count(5);
        assert!(small.enabled());
        assert_eq!(small.chunk0_size(), 4);
        assert_eq!(small.chunk_size(), 4);
        assert_eq!(small.scratch_chunk_capacity(), 4);

        let large = PackedNeighborPolicy::for_point_count(100);
        assert_eq!(large.chunk0_size(), 16);
        assert_eq!(large.chunk_size(), 8);
        assert_eq!(large.scratch_chunk_capacity(), 16);
    }

    #[test]
    fn packed_policy_defaults_are_pinned() {
        let policy = PackedNeighborPolicy::for_point_count(100);

        assert_eq!(policy.chunk0_size(), DEFAULT_PACKED_CHUNK0_SIZE);
        assert_eq!(policy.chunk_size(), DEFAULT_PACKED_CHUNK_SIZE);
        assert_eq!(policy.hi_budget(), PACKED_HI_BUDGET);
        assert_eq!(policy.hi_min_queries(), usize::MAX);

        let concentrated = policy.after_occupancy_rebuild();
        assert_eq!(concentrated.hi_budget(), CONCENTRATED_PACKED_HI_BUDGET);
        assert_eq!(
            concentrated.hi_min_queries(),
            CONCENTRATED_PACKED_HI_MIN_QUERIES
        );
    }
}
