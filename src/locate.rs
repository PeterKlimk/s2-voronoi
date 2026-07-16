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

use std::fmt;

use glam::Vec3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::cube_grid::{CubeMapGrid, CubeMapGridScratch};
use crate::{SpherePoint, SpherePointError, SphericalVoronoi, UnitVec3Like};

/// Why a unit-sphere locator query could not define a direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SphereQueryError {
    /// One query coordinate is NaN or infinite.
    NonFinite {
        /// Component index: 0 = x, 1 = y, 2 = z.
        component: usize,
    },
    /// All query coordinates are zero.
    Directionless,
}

impl fmt::Display for SphereQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { component } => {
                write!(f, "sphere query component {component} is not finite")
            }
            Self::Directionless => write!(f, "sphere query does not define a direction"),
        }
    }
}

impl std::error::Error for SphereQueryError {}

/// An indexed unit-sphere locator-query error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexedSphereQueryError {
    query_index: usize,
    source: SphereQueryError,
}

impl IndexedSphereQueryError {
    /// Index of the invalid query in the input slice.
    #[inline]
    pub fn query_index(&self) -> usize {
        self.query_index
    }

    /// Underlying query validation error.
    #[inline]
    pub fn query_error(&self) -> SphereQueryError {
        self.source
    }
}

impl fmt::Display for IndexedSphereQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid sphere query at index {}: {}",
            self.query_index, self.source
        )
    }
}

impl std::error::Error for IndexedSphereQueryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

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
            .map(|g| Vec3::from_array(g.to_array()))
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
    /// Every finite, nonzero vector denotes its radial direction: the query
    /// is normalized in f64 and rounded once to f32 before ranking. Returns
    /// the canonical cell index.
    pub fn locate<P: UnitVec3Like + ?Sized>(&mut self, p: &P) -> Result<usize, SphereQueryError> {
        let query = normalized_query(p)?;
        Ok(sphere_locate_core(
            &self.grid,
            self.canonical.as_deref(),
            &mut self.scratch,
            &mut self.batch,
            query,
        ))
    }

    /// Locate every query, in input order (parallel with the `parallel`
    /// feature — each worker gets its own scratch, so this takes `&self`).
    ///
    /// Queries use the same radial normalization as [`Self::locate`]. An
    /// error identifies the lowest invalid input index before any location
    /// work starts.
    pub fn locate_many<P: UnitVec3Like + Sync>(
        &self,
        queries: &[P],
    ) -> Result<Vec<usize>, IndexedSphereQueryError> {
        let normalized: Result<Vec<Vec3>, IndexedSphereQueryError> = queries
            .iter()
            .enumerate()
            .map(|(query_index, query)| {
                normalized_query(query).map_err(|source| IndexedSphereQueryError {
                    query_index,
                    source,
                })
            })
            .collect();
        Ok(self.locate_many_canonical(&normalized?, |query| *query))
    }

    #[inline]
    pub(crate) fn locate_sphere_point(&mut self, p: SpherePoint) -> usize {
        sphere_locate_core(
            &self.grid,
            self.canonical.as_deref(),
            &mut self.scratch,
            &mut self.batch,
            Vec3::from_array(p.to_array()),
        )
    }

    pub(crate) fn locate_many_sphere_points(&self, queries: &[SpherePoint]) -> Vec<usize> {
        self.locate_many_canonical(queries, |query| Vec3::from_array(query.to_array()))
    }

    fn locate_many_canonical<P, F>(&self, queries: &[P], to_vec3: F) -> Vec<usize>
    where
        P: Sync,
        F: Fn(&P) -> Vec3 + Sync + Send,
    {
        map_with_scratch(
            queries,
            || (self.grid.make_scratch(), Vec::new()),
            |scratch, p| {
                sphere_locate_core(
                    &self.grid,
                    self.canonical.as_deref(),
                    &mut scratch.0,
                    &mut scratch.1,
                    to_vec3(p),
                )
            },
        )
    }
}

#[inline]
fn normalized_query<P: UnitVec3Like + ?Sized>(p: &P) -> Result<Vec3, SphereQueryError> {
    let point = SpherePoint::try_from_xyz([p.x(), p.y(), p.z()]).map_err(|error| match error {
        SpherePointError::NonFinite { component } => SphereQueryError::NonFinite { component },
        SpherePointError::Directionless => SphereQueryError::Directionless,
        SpherePointError::OutsideStoredEnvelope => {
            unreachable!("direction construction normalizes rather than validating stored bits")
        }
    })?;
    Ok(Vec3::from_array(point.to_array()))
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
