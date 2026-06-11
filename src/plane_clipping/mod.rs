//! Planar Voronoi cell construction over a bounded rectangle.
//!
//! The planar counterpart of `knn_clipping`: cells are clipped in the same
//! 2D convex-polygon machinery (`topo2d::clippers`, `HalfPlane`,
//! `PolyBuffer`) — but with no projection layer at all. The chart *is* the
//! Euclidean plane, translated so the generator sits at the origin (which is
//! what makes `PolyBuffer::max_r2` directly meaningful for termination).
//!
//! The domain boundary is handled by seeding every cell's polygon with the
//! four rectangle walls as half-planes whose "neighbors" are **virtual wall
//! generators** (ids `wall_base + side`, where `wall_base` is the generator
//! count and `side` is [`WALL_BOTTOM`]..[`WALL_LEFT`]). Vertex keys, edge
//! records, and (later) dedup/stitching then work unchanged: a boundary
//! vertex is an ordinary `[gen, gen, wall]` triple and a rect corner is
//! `[gen, wall, wall]`.
//!
//! Cells are therefore bounded from the seed onward — there is no unbounded
//! state, no projection-limit fallback, and termination is a single
//! Euclidean comparison (see `PlaneCellBuilder::can_terminate`).

// Only builder-level tests consume this module so far; the planar pipeline
// (stream orchestration / live dedup integration) lands next.
#![allow(dead_code)]

mod builder;
#[cfg(test)]
mod tests;

// Re-exported for the planar pipeline (next phase); only tests use them yet.
#[allow(unused_imports)]
pub(crate) use builder::{PlaneCellBuilder, PlaneCellOutputBuffer, PlaneVertexData};

/// Wall side ids, offset from `wall_base` (= generator count).
pub(crate) const WALL_BOTTOM: u32 = 0; // y = 0
pub(crate) const WALL_RIGHT: u32 = 1; // x = 1
pub(crate) const WALL_TOP: u32 = 2; // y = 1
pub(crate) const WALL_LEFT: u32 = 3; // x = 0
