//! Planar Voronoi computation orchestration: domain normalization,
//! exact-duplicate welding, backend invocation, and output mapping.

use glam::Vec2;

use super::driver::compute_plane_cells;
use crate::plane_diagram::{PlanarVoronoi, PlanePoint, PlanePointLike, PlaneRect};
use crate::plane_grid::PlaneGrid;
use crate::policy::plane_grid_resolution;
use crate::VoronoiError;

/// Uniform rect -> normalized-domain transform (Voronoi structure is not
/// invariant under anisotropic scaling, so both axes share one scale; the
/// longer rect side maps to 1).
struct DomainTransform {
    rect: PlaneRect,
    /// Longer rect side; carried directly so the inverse map multiplies by
    /// `extent` instead of the double reciprocal `1/(1/extent)` (which
    /// overshoots the short axis for ~12% of width/height pairs).
    extent: f32,
    scale: f32,
    /// Normalized domain extents: walls at x=0, x=domain.x, y=0, y=domain.y.
    domain: Vec2,
}

impl DomainTransform {
    fn new(rect: PlaneRect) -> Self {
        let extent = rect.width().max(rect.height());
        let scale = 1.0 / extent;
        Self {
            rect,
            extent,
            scale,
            domain: Vec2::new(rect.width() * scale, rect.height() * scale),
        }
    }

    #[inline]
    fn to_normalized(&self, p: Vec2) -> Vec2 {
        (p - Vec2::new(self.rect.min.x, self.rect.min.y)) * self.scale
    }

    /// Map a normalized vertex back to rect coordinates.
    ///
    /// Clamped to the rect: the diagram is documented as a strict
    /// subdivision, and even a single rounding can push a wall vertex one
    /// ulp past the boundary (vertices are mathematically inside, so
    /// clamping only ever corrects rounding).
    #[inline]
    fn to_rect(&self, p: Vec2) -> PlanePoint {
        let r = self.rect;
        PlanePoint::new(
            (p.x * self.extent + r.min.x).clamp(r.min.x, r.max.x),
            (p.y * self.extent + r.min.y).clamp(r.min.y, r.max.y),
        )
    }
}

fn validate_rect(rect: PlaneRect) -> Result<(), VoronoiError> {
    let finite = rect.min.x.is_finite()
        && rect.min.y.is_finite()
        && rect.max.x.is_finite()
        && rect.max.y.is_finite();
    if !finite {
        return Err(VoronoiError::InvalidDomain {
            message: "rect has non-finite coordinates".to_string(),
        });
    }
    if !(rect.width() > 0.0 && rect.height() > 0.0) {
        return Err(VoronoiError::InvalidDomain {
            message: format!(
                "rect must have positive extent on both axes (width={}, height={})",
                rect.width(),
                rect.height()
            ),
        });
    }
    // Finite corners do not imply a finite extent (max - min can overflow
    // f32), and a subnormal extent makes the normalization scale overflow;
    // both would send inf/NaN through the pipeline.
    let extent = rect.width().max(rect.height());
    if !extent.is_finite() || !(1.0 / extent).is_finite() {
        return Err(VoronoiError::InvalidDomain {
            message: format!(
                "rect extent {} is not representable (must be finite and large \
                 enough that the normalization scale 1/extent is finite)",
                extent
            ),
        });
    }
    Ok(())
}

/// Weld points with identical NORMALIZED coordinates: the one coincidence
/// class that provably breaks half-plane clipping (the bisector degenerates
/// to the whole plane). This is deliberately the computation space, not the
/// input space — bit-distinct inputs that round together under the rect
/// normalization MUST weld (they would otherwise produce the degenerate
/// bisector), while points that stay distinct here have exactly
/// representable f64 coordinate differences and well-formed bisectors.
struct ExactWeld {
    /// Canonical (first-occurrence) original index per original index.
    weld_map: Option<Vec<u32>>,
    /// Effective (deduplicated) normalized points.
    effective: Vec<Vec2>,
    /// Effective index per original index.
    original_to_effective: Vec<u32>,
}

fn weld_exact_duplicates(points: &[Vec2]) -> ExactWeld {
    let mut first_seen: rustc_hash::FxHashMap<[u32; 2], u32> = rustc_hash::FxHashMap::default();
    first_seen.reserve(points.len());
    let mut weld_map: Vec<u32> = Vec::with_capacity(points.len());
    let mut effective: Vec<Vec2> = Vec::with_capacity(points.len());
    let mut original_to_effective: Vec<u32> = Vec::with_capacity(points.len());
    let mut any_weld = false;

    for (i, p) in points.iter().enumerate() {
        let bits = [p.x.to_bits(), p.y.to_bits()];
        match first_seen.get(&bits) {
            Some(&canonical) => {
                any_weld = true;
                weld_map.push(canonical);
                original_to_effective.push(original_to_effective[canonical as usize]);
            }
            None => {
                first_seen.insert(bits, i as u32);
                weld_map.push(i as u32);
                original_to_effective.push(effective.len() as u32);
                effective.push(*p);
            }
        }
    }

    ExactWeld {
        weld_map: any_weld.then_some(weld_map),
        effective,
        original_to_effective,
    }
}

pub(crate) fn compute_plane_impl<P: PlanePointLike>(
    points: &[P],
    rect: PlaneRect,
) -> Result<PlanarVoronoi, VoronoiError> {
    validate_rect(rect)?;
    if points.is_empty() {
        return Err(VoronoiError::InsufficientPoints(0));
    }
    if u32::try_from(points.len().saturating_add(4)).is_err() {
        return Err(VoronoiError::RepresentationLimit(format!(
            "generator count {} exceeds u32-backed index capacity",
            points.len()
        )));
    }

    let transform = DomainTransform::new(rect);
    let mut generators: Vec<PlanePoint> = Vec::with_capacity(points.len());
    let mut normalized: Vec<Vec2> = Vec::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        let (x, y) = (p.x(), p.y());
        if !x.is_finite() || !y.is_finite() {
            return Err(VoronoiError::InvalidInput {
                point_index: i,
                message: format!("non-finite coordinates ({x}, {y})"),
            });
        }
        if !rect.contains(x, y) {
            return Err(VoronoiError::InvalidInput {
                point_index: i,
                message: format!("point ({x}, {y}) lies outside the domain rect"),
            });
        }
        generators.push(PlanePoint::new(x, y));
        normalized.push(transform.to_normalized(Vec2::new(x, y)));
    }

    let weld = weld_exact_duplicates(&normalized);
    let effective = &weld.effective;

    let occupied = transform.domain.x as f64 * transform.domain.y as f64;
    let res = plane_grid_resolution(effective.len(), occupied);
    let grid = PlaneGrid::new(effective, res);
    let output = compute_plane_cells(effective, &grid, transform.domain)?;

    // Map vertices back to rect coordinates.
    let vertices: Vec<PlanePoint> = output
        .vertices
        .into_iter()
        .map(|v| transform.to_rect(v))
        .collect();

    // One cell per ORIGINAL generator; welded twins alias their canonical
    // cell's (start, len) range, exactly like the spherical weld contract.
    let cells: Vec<crate::diagram::VoronoiCell> = weld
        .original_to_effective
        .iter()
        .map(|&eff| output.cells[eff as usize])
        .collect();

    Ok(PlanarVoronoi::from_raw_parts(
        generators,
        vertices,
        cells,
        output.cell_indices,
        weld.weld_map,
        rect,
    ))
}
