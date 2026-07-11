//! Point location: map a query point to the Voronoi cell that contains it
//! (equivalently, find its nearest generator).
//!
//! Locators are built on demand from a computed diagram (one grid build
//! over the generators) and answer repeated queries in near-constant time
//! via the same shell-expansion frontiers the construction pipeline uses:
//! collect ring candidates nearest-first until the ring certificate proves
//! no unseen generator can beat the best found. Queries take `&mut self`
//! (each locator carries its own scratch); clone the locator for use from
//! multiple threads.
//!
//! Returned indices are **canonical** cell indices: for welded inputs the
//! twin's canonical cell is returned (a query can never be strictly closer
//! to a twin than to its canonical, since welded generators coincide within
//! the weld radius).

use glam::Vec3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::cube_grid::{CubeMapGrid, CubeMapGridScratch};
use crate::{SphericalVoronoi, UnitVec3Like};

/// Run `f` over every query with a per-worker scratch from `mk` (parallel
/// with the `parallel` feature, plain loop otherwise).
fn map_with_scratch<P, S, MK, F>(queries: &[P], mk: MK, f: F) -> Vec<usize>
where
    P: Sync,
    S: Send,
    MK: Fn() -> S + Sync + Send,
    F: Fn(&mut S, &P) -> usize + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        queries
            .par_iter()
            .map_init(&mk, |scratch, p| f(scratch, p))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = mk();
        queries.iter().map(|p| f(&mut scratch, p)).collect()
    }
}

/// Point locator for a [`SphericalVoronoi`] diagram.
///
/// Build with [`SphericalVoronoi::build_locator`]; query with
/// [`Self::locate`].
pub struct SphereLocator {
    grid: CubeMapGrid,
    canonical: Option<Vec<u32>>,
    scratch: CubeMapGridScratch,
    batch: Vec<u32>,
}

impl SphericalVoronoi {
    /// Build a point locator over this diagram's generators (one spatial
    /// grid build; queries are then near-constant time).
    pub fn build_locator(&self) -> SphereLocator {
        let generators: Vec<Vec3> = self
            .generators()
            .iter()
            .map(|g| Vec3::new(g.x, g.y, g.z))
            .collect();
        let n = generators.len();
        let res = ((n as f64 / (6.0 * 16.0)).sqrt() as usize).max(4);
        let grid = CubeMapGrid::new(&generators, res);
        let scratch = grid.make_scratch();
        SphereLocator {
            grid,
            canonical: self.weld_map().map(|m| m.to_vec()),
            scratch,
            batch: Vec::new(),
        }
    }
}

impl SphereLocator {
    /// Cell containing the query direction (nearest generator by angle).
    ///
    /// The query is assumed unit-normalized, like the diagram's inputs.
    /// Returns the canonical cell index.
    pub fn locate<P: UnitVec3Like>(&mut self, p: &P) -> usize {
        let query = Vec3::new(p.x(), p.y(), p.z());
        sphere_locate_core(
            &self.grid,
            self.canonical.as_deref(),
            &mut self.scratch,
            &mut self.batch,
            query,
        )
    }

    /// Locate every query, in input order (parallel with the `parallel`
    /// feature — each worker gets its own scratch, so this takes `&self`).
    pub fn locate_many<P: UnitVec3Like + Sync>(&self, queries: &[P]) -> Vec<usize> {
        map_with_scratch(
            queries,
            || (self.grid.make_scratch(), Vec::new()),
            |scratch, p| {
                sphere_locate_core(
                    &self.grid,
                    self.canonical.as_deref(),
                    &mut scratch.0,
                    &mut scratch.1,
                    Vec3::new(p.x(), p.y(), p.z()),
                )
            },
        )
    }
}

fn sphere_locate_core(
    grid: &CubeMapGrid,
    canonical: Option<&[u32]>,
    scratch: &mut CubeMapGridScratch,
    batch: &mut Vec<u32>,
    query: Vec3,
) -> usize {
    let slot = grid
        .nearest_unrestricted_slot(query, scratch, batch)
        .expect("locator requires a non-empty diagram");
    let idx = grid.point_indices()[slot as usize] as usize;
    match canonical {
        Some(map) => map[idx] as usize,
        None => idx,
    }
}
