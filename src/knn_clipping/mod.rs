//! Spherical Voronoi computation via half-space (great circle) clipping.
//!
//! This module implements a "meshless" approach where each Voronoi cell is computed
//! independently from nearby neighbors. This structure is friendly to data-parallel CPU
//! implementations.

pub(crate) mod cell_build;
pub(crate) mod compute;
pub(crate) mod driver;
pub(crate) mod edge_reconcile;
pub(crate) mod local_hull;
pub(crate) mod local_rebuild;
pub(crate) mod output_resolution;
pub(crate) mod preprocess;
pub(crate) mod topo2d;
pub(crate) mod union_find;
