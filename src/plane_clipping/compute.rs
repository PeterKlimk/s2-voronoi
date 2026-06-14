//! Planar Voronoi computation orchestration: domain normalization,
//! exact-duplicate welding, backend invocation, and output mapping.

use glam::Vec2;

use super::driver::compute_plane_cells;
use super::periodic_driver::compute_periodic_cells;
use crate::knn_clipping::timing::{Timer, TimingBuilder};
use crate::plane_diagram::{PlanarVoronoi, PlanePoint, PlanePointLike, PlaneRect, PlaneTopology};
use crate::plane_grid::periodic::PeriodicGrid;
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

/// Radius weld over the already-built grid (the grid-integrated design):
/// generators within [`crate::tolerances::PLANE_WELD_DIST`] (normalized
/// units) weld to one cell, the canonical member being the lowest original
/// index of each group.
///
/// Welding within a radius — not just exact coordinate equality — is
/// required for graph validity: probing (tests/plane_coincidence_probes.rs)
/// shows clusters of 3+ generators within ~1 ulp of unit scale produce
/// invalid topology even though each individual bisector is well-formed.
/// Pairs at any distinct-f32 separation resolve fine; the radius's margin
/// over the observed failure scale is documented at the constant.
///
/// The detection reuses the spatial grid built for the kNN queries (points
/// can only weld within a cell or across a radius-thin wall band), so the
/// common no-weld case is a read-only scan and the detection grid IS the
/// production grid — no hash pass, no copies, no rebuild.
struct WeldResult {
    /// Canonical (lowest) original index per original index.
    weld_map: Vec<u32>,
    /// Effective (canonical) normalized points, in original index order.
    effective: Vec<Vec2>,
    /// Effective index per original index.
    original_to_effective: Vec<u32>,
}

fn weld_within_radius(points: &[Vec2], grid: &PlaneGrid) -> Option<WeldResult> {
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    grid.collect_pairs_within(crate::tolerances::PLANE_WELD_DIST, &mut pairs);
    if pairs.is_empty() {
        return None;
    }

    let mut uf = crate::knn_clipping::union_find::UnionFind::new(points.len());
    for &(a, b) in &pairs {
        uf.union_keep_min(a, b);
    }

    let mut weld_map: Vec<u32> = Vec::with_capacity(points.len());
    let mut effective: Vec<Vec2> = Vec::new();
    let mut original_to_effective: Vec<u32> = vec![0; points.len()];
    for i in 0..points.len() {
        let canonical = uf.find(i as u32);
        weld_map.push(canonical);
        if canonical as usize == i {
            original_to_effective[i] = effective.len() as u32;
            effective.push(points[i]);
        }
    }
    for i in 0..points.len() {
        let canonical = weld_map[i] as usize;
        original_to_effective[i] = original_to_effective[canonical];
    }

    Some(WeldResult {
        weld_map,
        effective,
        original_to_effective,
    })
}

/// Plain bounded compute: a residual is provably-invalid output with no
/// report channel, so fail loud.
pub(crate) fn compute_plane_impl<P: PlanePointLike>(
    points: &[P],
    rect: PlaneRect,
) -> Result<PlanarVoronoi, VoronoiError> {
    let (diagram, residual) = compute_plane_built(points, rect)?;
    if !residual.is_empty() {
        return Err(crate::knn_clipping::edge_reconcile::residual_error(
            &residual,
        ));
    }
    crate::validation::verify_plane_if_enabled(&diagram)?;
    Ok(diagram)
}

/// Bounded compute with observability: returns the diagram plus a report
/// carrying strict validation and any post-repair unpaired-edge residuals
/// (effective-generator pairs; equal to original indices when no weld
/// occurred). Does NOT error on residuals — the report is the catch channel.
pub(crate) fn compute_plane_with_report_impl<P: PlanePointLike>(
    points: &[P],
    rect: PlaneRect,
) -> Result<crate::PlaneComputeOutput, VoronoiError> {
    let (diagram, residual) = compute_plane_built(points, rect)?;
    let report = crate::PlaneComputeReport {
        validation: crate::validation::validate_plane(&diagram),
        unresolved_edge_pairs: residual,
    };
    Ok(crate::PlaneComputeOutput { diagram, report })
}

fn compute_plane_built<P: PlanePointLike>(
    points: &[P],
    rect: PlaneRect,
) -> Result<(PlanarVoronoi, Vec<(u32, u32)>), VoronoiError> {
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

    let mut tb = TimingBuilder::new();
    let transform = DomainTransform::new(rect);
    let t = Timer::start();
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
    let mut preprocess = t.elapsed();

    let t = Timer::start();
    let occupied = transform.domain.x as f64 * transform.domain.y as f64;
    let res = plane_grid_resolution(normalized.len(), occupied);
    let mut grid = PlaneGrid::new(&normalized, res);
    tb.set_knn_build(t.elapsed());

    // Optimistic weld: the no-weld case (overwhelmingly common) reuses this
    // very grid for the computation; only actual welds pay a rebuild.
    let t = Timer::start();
    let weld = weld_within_radius(&normalized, &grid);
    preprocess += t.elapsed();
    tb.set_preprocess(preprocess);
    tb.set_grid_stats(res, 0, weld.is_some());
    let (output, weld_map, original_to_effective) = match &weld {
        None => (
            compute_plane_cells(&normalized, &grid, transform.domain, &mut tb)?,
            None,
            None,
        ),
        Some(weld) => {
            // Compact the existing grid in place at the same resolution rather
            // than rebuilding (the sphere's approach): welds are too few to
            // shift the resolution, and the compacted grid is bit-identical to
            // a fresh build on the effective points. Skips the count+scatter.
            let n_eff = weld.effective.len();
            let kept: Vec<bool> = (0..normalized.len())
                .map(|i| weld.weld_map[i] as usize == i)
                .collect();
            grid.compact_welded(&kept, &weld.original_to_effective, n_eff);
            (
                compute_plane_cells(&weld.effective, &grid, transform.domain, &mut tb)?,
                Some(weld.weld_map.clone()),
                Some(&weld.original_to_effective),
            )
        }
    };
    let residual = output.residual.clone();

    let t = Timer::start();
    // Map vertices back to rect coordinates.
    let vertices: Vec<PlanePoint> = output
        .vertices
        .into_iter()
        .map(|v| transform.to_rect(v))
        .collect();

    // One cell per ORIGINAL generator; welded twins alias their canonical
    // cell's (start, len) range, exactly like the spherical weld contract.
    let cells: Vec<crate::diagram::VoronoiCell> = match original_to_effective {
        None => output.cells,
        Some(map) => map.iter().map(|&eff| output.cells[eff as usize]).collect(),
    };

    let diagram = PlanarVoronoi::from_raw_parts(
        generators,
        vertices,
        cells,
        output.cell_indices,
        weld_map,
        rect,
        PlaneTopology::Bounded,
    );
    tb.set_assemble(t.elapsed());
    tb.finish().report(diagram.num_cells());
    Ok((diagram, residual))
}

fn weld_within_radius_periodic(points: &[Vec2], grid: &PeriodicGrid) -> Option<WeldResult> {
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    grid.collect_pairs_within(crate::tolerances::PLANE_WELD_DIST, &mut pairs);
    if pairs.is_empty() {
        return None;
    }
    let mut uf = crate::knn_clipping::union_find::UnionFind::new(points.len());
    for &(a, b) in &pairs {
        uf.union_keep_min(a, b);
    }
    let mut weld_map: Vec<u32> = Vec::with_capacity(points.len());
    let mut effective: Vec<Vec2> = Vec::new();
    let mut original_to_effective: Vec<u32> = vec![0; points.len()];
    for i in 0..points.len() {
        let canonical = uf.find(i as u32);
        weld_map.push(canonical);
        if canonical as usize == i {
            original_to_effective[i] = effective.len() as u32;
            effective.push(points[i]);
        }
    }
    for i in 0..points.len() {
        let canonical = weld_map[i] as usize;
        original_to_effective[i] = original_to_effective[canonical];
    }
    Some(WeldResult {
        weld_map,
        effective,
        original_to_effective,
    })
}

/// Plain periodic compute: fail loud on residuals (an unpaired edge on the
/// torus is an invalid subdivision).
pub(crate) fn compute_plane_periodic_impl<P: PlanePointLike>(
    points: &[P],
    rect: PlaneRect,
) -> Result<PlanarVoronoi, VoronoiError> {
    let (diagram, residual) = compute_plane_periodic_built(points, rect)?;
    if !residual.is_empty() {
        return Err(crate::knn_clipping::edge_reconcile::residual_error(
            &residual,
        ));
    }
    crate::validation::verify_plane_if_enabled(&diagram)?;
    Ok(diagram)
}

/// Periodic compute with observability (see [`compute_plane_with_report_impl`]).
pub(crate) fn compute_plane_periodic_with_report_impl<P: PlanePointLike>(
    points: &[P],
    rect: PlaneRect,
) -> Result<crate::PlaneComputeOutput, VoronoiError> {
    let (diagram, residual) = compute_plane_periodic_built(points, rect)?;
    let report = crate::PlaneComputeReport {
        validation: crate::validation::validate_plane(&diagram),
        unresolved_edge_pairs: residual,
    };
    Ok(crate::PlaneComputeOutput { diagram, report })
}

fn compute_plane_periodic_built<P: PlanePointLike>(
    points: &[P],
    rect: PlaneRect,
) -> Result<(PlanarVoronoi, Vec<(u32, u32)>), VoronoiError> {
    validate_rect(rect)?;
    if points.is_empty() {
        return Err(VoronoiError::InsufficientPoints(0));
    }
    if u32::try_from(points.len()).is_err() {
        return Err(VoronoiError::RepresentationLimit(format!(
            "generator count {} exceeds u32-backed index capacity",
            points.len()
        )));
    }

    let mut tb = TimingBuilder::new();
    let transform = DomainTransform::new(rect);
    // Normalized periods: the longer rect side maps to 1.
    let (px, py) = (transform.domain.x, transform.domain.y);
    let t = Timer::start();
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
        // On the torus, the rect's max edges are identified with its min
        // edges: wrap normalized coordinates into [0, p).
        let n = transform.to_normalized(Vec2::new(x, y));
        normalized.push(Vec2::new(
            n.x.rem_euclid(px).min(next_below(px)),
            n.y.rem_euclid(py).min(next_below(py)),
        ));
    }

    let mut preprocess = t.elapsed();

    // The torus is fully occupied: occupancy fraction 1 over res^2 cells.
    let t = Timer::start();
    let res = plane_grid_resolution(normalized.len(), 1.0);
    let mut grid = PeriodicGrid::new(&normalized, res, px, py);
    tb.set_knn_build(t.elapsed());

    let t = Timer::start();
    let weld = weld_within_radius_periodic(&normalized, &grid);
    preprocess += t.elapsed();
    tb.set_preprocess(preprocess);
    tb.set_grid_stats(res, 0, weld.is_some());
    let (output, weld_map, original_to_effective) = match &weld {
        None => (
            compute_periodic_cells(&normalized, &grid, &mut tb)?,
            None,
            None,
        ),
        Some(weld) => {
            // Compact in place at the same resolution instead of rebuilding
            // (see the bounded path); bit-identical to a fresh effective grid.
            let n_eff = weld.effective.len();
            let kept: Vec<bool> = (0..normalized.len())
                .map(|i| weld.weld_map[i] as usize == i)
                .collect();
            grid.compact_welded(&kept, &weld.original_to_effective, n_eff);
            (
                compute_periodic_cells(&weld.effective, &grid, &mut tb)?,
                Some(weld.weld_map.clone()),
                Some(&weld.original_to_effective),
            )
        }
    };
    let residual = output.residual.clone();

    let t = Timer::start();
    // Map canonical wrapped vertices back to rect coordinates (still
    // canonically wrapped; cell_polygon does the per-cell unwrap).
    let vertices: Vec<PlanePoint> = output
        .vertices
        .into_iter()
        .map(|v| transform.to_rect(v))
        .collect();

    let cells: Vec<crate::diagram::VoronoiCell> = match original_to_effective {
        None => output.cells,
        Some(map) => map.iter().map(|&eff| output.cells[eff as usize]).collect(),
    };

    let diagram = PlanarVoronoi::from_raw_parts(
        generators,
        vertices,
        cells,
        output.cell_indices,
        weld_map,
        rect,
        PlaneTopology::Periodic,
    );
    tb.set_assemble(t.elapsed());
    tb.finish().report(diagram.num_cells());
    Ok((diagram, residual))
}

/// Largest f32 strictly below `p` (clamps wrapped coordinates out of the
/// half-open domain's excluded endpoint).
#[inline]
fn next_below(p: f32) -> f32 {
    f32::from_bits(p.to_bits() - 1)
}
