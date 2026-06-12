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

mod builder;
pub(crate) mod compute;
pub(crate) mod driver;
pub(crate) mod periodic_builder;
#[cfg(test)]
mod tests;

pub(crate) use builder::PlaneCellBuilder;

/// Wall side ids, offset from `wall_base` (= generator count). Walls sit at
/// the normalized domain bounds: x in [0, domain.x], y in [0, domain.y]
/// (the longer rect side maps to 1; the shorter to < 1 for non-square rects).
pub(crate) const WALL_BOTTOM: u32 = 0; // y = 0
pub(crate) const WALL_RIGHT: u32 = 1; // x = domain.x
pub(crate) const WALL_TOP: u32 = 2; // y = domain.y
pub(crate) const WALL_LEFT: u32 = 3; // x = 0
