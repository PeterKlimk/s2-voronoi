//! K-nearest neighbor provider for Voronoi cell construction.

use glam::Vec3;

/// K-NN provider using CubeMapGrid - O(n) build time, good for large point sets.
///
/// Wraps a CubeMapGrid and provides confidence-based k-NN queries with const-generic
/// fixed buffers for zero per-query allocation.
pub struct CubeMapGridKnn<'a> {
    grid: crate::cube_grid::CubeMapGrid,
    points: &'a [Vec3],
    #[cfg(feature = "timing")]
    grid_build_timings: crate::cube_grid::CubeMapGridBuildTimings,
}

impl<'a> CubeMapGridKnn<'a> {
    pub fn new(points: &'a [Vec3]) -> Self {
        Self::new_with_target_density(points, super::KNN_GRID_TARGET_DENSITY)
    }

    pub fn new_with_target_density(points: &'a [Vec3], target_points_per_cell: f64) -> Self {
        let n = points.len();
        let target = target_points_per_cell.max(1.0);
        let res = ((n as f64 / (6.0 * target)).sqrt() as usize).max(4);
        #[cfg(feature = "timing")]
        {
            let mut grid_build_timings = crate::cube_grid::CubeMapGridBuildTimings::default();
            let grid = crate::cube_grid::CubeMapGrid::new_with_build_timings(
                points,
                res,
                &mut grid_build_timings,
            );
            Self {
                grid,
                points,
                grid_build_timings,
            }
        }
        #[cfg(not(feature = "timing"))]
        {
            let grid = crate::cube_grid::CubeMapGrid::new(points, res);
            Self { grid, points }
        }
    }

    /// Access the underlying cube-map grid.
    #[inline]
    pub fn grid(&self) -> &crate::cube_grid::CubeMapGrid {
        &self.grid
    }

    #[cfg(feature = "timing")]
    #[inline]
    pub fn grid_build_timings(&self) -> &crate::cube_grid::CubeMapGridBuildTimings {
        &self.grid_build_timings
    }

    /// Create a scratch buffer for k-NN queries.
    #[inline]
    pub fn make_scratch(&self) -> crate::cube_grid::CubeMapGridScratch {
        self.grid.make_scratch()
    }

    /// Start a resumable k-NN query returning slots, tracking up to `track_limit` neighbors.
    #[inline]
    pub fn knn_resumable_slots_into(
        &self,
        query: Vec3,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut crate::cube_grid::CubeMapGridScratch,
        out_slots: &mut Vec<u32>,
    ) -> crate::cube_grid::KnnStatus {
        self.grid.find_k_nearest_resumable_slots_into(
            self.points,
            query,
            query_idx,
            k,
            track_limit,
            scratch,
            out_slots,
        )
    }
}
