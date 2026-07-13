//! Embed the canonical unit-sphere diagram in world coordinates.
//!
//! The clipping backend remains entirely in dimensionless unit-sphere space.
//! This module supplies the translation and uniform scale that place that
//! abstract sphere in a caller's coordinate system.

use std::fmt;

use crate::{
    ComputeOutput, ComputeReport, SphereLocator, SphericalVoronoi, UnitVec3, VoronoiConfig,
    VoronoiError,
};

/// A type whose components can be read as f64 world coordinates.
///
/// World-space conversion deliberately uses f64 even though the geometric
/// backend stores canonical directions as f32. This preserves translation and
/// scaling precision until the last possible point before computation.
pub trait WorldVec3Like {
    /// X component.
    fn x(&self) -> f64;
    /// Y component.
    fn y(&self) -> f64;
    /// Z component.
    fn z(&self) -> f64;
}

impl WorldVec3Like for [f64; 3] {
    #[inline]
    fn x(&self) -> f64 {
        self[0]
    }

    #[inline]
    fn y(&self) -> f64 {
        self[1]
    }

    #[inline]
    fn z(&self) -> f64 {
        self[2]
    }
}

impl WorldVec3Like for (f64, f64, f64) {
    #[inline]
    fn x(&self) -> f64 {
        self.0
    }

    #[inline]
    fn y(&self) -> f64 {
        self.1
    }

    #[inline]
    fn z(&self) -> f64 {
        self.2
    }
}

#[cfg(feature = "glam")]
impl WorldVec3Like for glam::DVec3 {
    #[inline]
    fn x(&self) -> f64 {
        self.x
    }

    #[inline]
    fn y(&self) -> f64 {
        self.y
    }

    #[inline]
    fn z(&self) -> f64 {
        self.z
    }
}

/// Why a [`SphereEmbedding`] could not be constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SphereEmbeddingError {
    /// A center component was NaN or infinite.
    NonFiniteCenter {
        /// Component index: 0 = x, 1 = y, 2 = z.
        component: usize,
    },
    /// The radius was zero, negative, NaN, or infinite.
    InvalidRadius,
    /// Some point on the embedded sphere would exceed finite f64 range.
    UnrepresentableExtent {
        /// Component index: 0 = x, 1 = y, 2 = z.
        component: usize,
    },
}

impl fmt::Display for SphereEmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteCenter { component } => {
                write!(f, "sphere center component {component} is not finite")
            }
            Self::InvalidRadius => write!(f, "sphere radius must be finite and positive"),
            Self::UnrepresentableExtent { component } => write!(
                f,
                "sphere extent along component {component} exceeds finite f64 range"
            ),
        }
    }
}

impl std::error::Error for SphereEmbeddingError {}

/// Why a world-space point could not be converted to a sphere direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SphereProjectionError {
    /// A point component was NaN or infinite.
    NonFinitePoint {
        /// Component index: 0 = x, 1 = y, 2 = z.
        component: usize,
    },
    /// The point is at, or is numerically indistinguishable from, the sphere center.
    PointAtCenter,
}

impl fmt::Display for SphereProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinitePoint { component } => {
                write!(f, "world point component {component} is not finite")
            }
            Self::PointAtCenter => write!(
                f,
                "world point does not define a direction from the sphere center"
            ),
        }
    }
}

impl std::error::Error for SphereProjectionError {}

/// An indexed world-query conversion error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexedSphereProjectionError {
    point_index: usize,
    source: SphereProjectionError,
}

impl IndexedSphereProjectionError {
    /// Index of the invalid point in the input query slice.
    #[inline]
    pub fn point_index(&self) -> usize {
        self.point_index
    }

    /// Underlying conversion error.
    #[inline]
    pub fn projection_error(&self) -> SphereProjectionError {
        self.source
    }
}

impl fmt::Display for IndexedSphereProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid world point at index {}: {}",
            self.point_index, self.source
        )
    }
}

impl std::error::Error for IndexedSphereProjectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Translation and uniform scale embedding a unit sphere in world coordinates.
///
/// Construction guarantees that every point `center + radius * direction`
/// with unit `direction` has finite f64 components. It cannot guarantee that a
/// tiny radius remains distinguishable from a very large center after f64
/// rounding.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "wire::SphereEmbeddingWire"))]
pub struct SphereEmbedding {
    center: [f64; 3],
    radius: f64,
}

impl SphereEmbedding {
    /// Construct a validated embedding.
    pub fn new(center: [f64; 3], radius: f64) -> Result<Self, SphereEmbeddingError> {
        for (component, &value) in center.iter().enumerate() {
            if !value.is_finite() {
                return Err(SphereEmbeddingError::NonFiniteCenter { component });
            }
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(SphereEmbeddingError::InvalidRadius);
        }
        for (component, &value) in center.iter().enumerate() {
            if radius > f64::MAX - value.abs() {
                return Err(SphereEmbeddingError::UnrepresentableExtent { component });
            }
        }
        Ok(Self { center, radius })
    }

    /// World-space center.
    #[inline]
    pub fn center(self) -> [f64; 3] {
        self.center
    }

    /// World-space radius.
    #[inline]
    pub fn radius(self) -> f64 {
        self.radius
    }

    /// Recover the unit direction from the center to a world-space point.
    ///
    /// The point's radial distance is deliberately discarded. Every finite
    /// point other than the center is projected radially onto the sphere.
    pub fn project_world_to_unit<P: WorldVec3Like + ?Sized>(
        self,
        point: &P,
    ) -> Result<[f64; 3], SphereProjectionError> {
        let world = world_components(point);
        self.validate_world_point(world)?;
        Ok(self.project_validated_world(world))
    }

    #[inline]
    fn validate_world_point(self, world: [f64; 3]) -> Result<(), SphereProjectionError> {
        for (component, &value) in world.iter().enumerate() {
            if !value.is_finite() {
                return Err(SphereProjectionError::NonFinitePoint { component });
            }
        }
        if world == self.center {
            return Err(SphereProjectionError::PointAtCenter);
        }
        Ok(())
    }

    #[inline]
    fn project_validated_world(self, world: [f64; 3]) -> [f64; 3] {
        let direct = [
            world[0] - self.center[0],
            world[1] - self.center[1],
            world[2] - self.center[2],
        ];
        if direct.iter().all(|v| v.is_finite()) {
            return normalize_scaled_valid(direct);
        }

        // Both operands are finite, but an opposite-sign subtraction can
        // overflow. Divide all six components by one common scale first; that
        // preserves the displacement direction while keeping subtraction in
        // [-2, 2].
        let operand_scale = world
            .iter()
            .chain(self.center.iter())
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);
        debug_assert!(operand_scale.is_finite() && operand_scale > 0.0);
        normalize_scaled_valid([
            world[0] / operand_scale - self.center[0] / operand_scale,
            world[1] / operand_scale - self.center[1] / operand_scale,
            world[2] / operand_scale - self.center[2] / operand_scale,
        ])
    }

    /// Embed a unit direction as a world-space point on the sphere.
    ///
    /// `direction` is assumed unit-normalized, following the [`UnitVec3`]
    /// contract. The embedding constructor guarantees finite output for such
    /// directions.
    #[inline]
    pub fn unit_to_world(self, direction: [f64; 3]) -> [f64; 3] {
        [
            direction[0].mul_add(self.radius, self.center[0]),
            direction[1].mul_add(self.radius, self.center[1]),
            direction[2].mul_add(self.radius, self.center[2]),
        ]
    }

    /// Convert a solid angle in steradians to physical surface area.
    ///
    /// Returns positive infinity when the mathematical area exceeds finite
    /// f64 range.
    #[inline]
    pub fn solid_angle_to_area(self, steradians: f64) -> f64 {
        steradians * self.radius * self.radius
    }

    #[inline]
    fn stored_unit_to_world(self, direction: UnitVec3) -> [f64; 3] {
        self.unit_to_world([direction.x as f64, direction.y as f64, direction.z as f64])
    }
}

#[inline]
fn world_components<P: WorldVec3Like + ?Sized>(point: &P) -> [f64; 3] {
    [point.x(), point.y(), point.z()]
}

#[inline]
fn normalize_scaled_valid(mut v: [f64; 3]) -> [f64; 3] {
    let scale = v.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    debug_assert!(scale.is_finite() && scale > 0.0);
    for value in &mut v {
        *value /= scale;
    }
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    debug_assert!(len.is_finite() && len >= 1.0);
    [v[0] / len, v[1] / len, v[2] / len]
}

#[inline]
fn projected_unit_f32<P: WorldVec3Like + ?Sized>(
    embedding: SphereEmbedding,
    point: &P,
) -> Result<UnitVec3, SphereProjectionError> {
    projected_components_f32(embedding, world_components(point))
}

#[inline]
fn projected_components_f32(
    embedding: SphereEmbedding,
    world: [f64; 3],
) -> Result<UnitVec3, SphereProjectionError> {
    embedding.validate_world_point(world)?;
    let u = embedding.project_validated_world(world);
    Ok(UnitVec3::new(u[0] as f32, u[1] as f32, u[2] as f32))
}

fn project_points<P: WorldVec3Like>(
    points: &[P],
    embedding: SphereEmbedding,
) -> Result<Vec<glam::Vec3>, VoronoiError> {
    if points.len() < 4 {
        return Err(VoronoiError::InsufficientPoints(points.len()));
    }

    #[cfg(feature = "parallel")]
    {
        use std::sync::mpsc;

        const PARALLEL_PROJECTION_MIN_POINTS: usize = 1 << 14;
        if points.len() >= PARALLEL_PROJECTION_MIN_POINTS && rayon::current_num_threads() > 1 {
            // Match the core API's unconstrained input trait: read caller data
            // once on this thread, then send owned chunks to the Rayon pool.
            // This supports non-Sync and stateful implementations without a
            // full-size snapshot or re-reading invalid points.
            const CHUNK_POINTS: usize = 1 << 14;
            let mut projected = vec![glam::Vec3::ZERO; points.len()];
            let (error_tx, error_rx) = mpsc::channel();
            let wave_points = CHUNK_POINTS * rayon::current_num_threads();
            let mut remaining_output = projected.as_mut_slice();
            for (wave_index, input_wave) in points.chunks(wave_points).enumerate() {
                let wave_start = wave_index * wave_points;
                let (output_wave, remaining) = remaining_output.split_at_mut(input_wave.len());
                remaining_output = remaining;
                rayon::in_place_scope(|scope| {
                    let mut remaining_wave_output = output_wave;
                    for (chunk_index, input_chunk) in input_wave.chunks(CHUNK_POINTS).enumerate() {
                        let (output_chunk, remaining) =
                            remaining_wave_output.split_at_mut(input_chunk.len());
                        remaining_wave_output = remaining;
                        let point_start = wave_start + chunk_index * CHUNK_POINTS;
                        let world_points: Vec<[f64; 3]> =
                            input_chunk.iter().map(world_components).collect();
                        let error_tx = error_tx.clone();
                        scope.spawn(move |_| {
                            let mut first_error = None;
                            for (offset, (&world, output)) in
                                world_points.iter().zip(output_chunk.iter_mut()).enumerate()
                            {
                                match projected_components_f32(embedding, world) {
                                    Ok(u) => *output = glam::Vec3::new(u.x, u.y, u.z),
                                    Err(err) => {
                                        if first_error.is_none() {
                                            first_error = Some((point_start + offset, err));
                                        }
                                    }
                                }
                            }
                            if let Some(error) = first_error {
                                let _ = error_tx.send(error);
                            }
                        });
                    }
                });
            }
            drop(error_tx);
            if let Some((point_index, err)) = error_rx.into_iter().min_by_key(|(index, _)| *index) {
                return Err(VoronoiError::InvalidInput {
                    point_index,
                    message: err.to_string(),
                });
            }

            return Ok(projected);
        }
    }

    points
        .iter()
        .enumerate()
        .map(|(point_index, point)| {
            projected_unit_f32(embedding, point)
                .map(|u| glam::Vec3::new(u.x, u.y, u.z))
                .map_err(|err| VoronoiError::InvalidInput {
                    point_index,
                    message: err.to_string(),
                })
        })
        .collect()
}

/// A unit-sphere Voronoi diagram together with its world-space embedding.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmbeddedSphericalVoronoi {
    diagram: SphericalVoronoi,
    embedding: SphereEmbedding,
}

impl EmbeddedSphericalVoronoi {
    /// Wrap an existing canonical unit-sphere diagram.
    #[inline]
    pub fn new(diagram: SphericalVoronoi, embedding: SphereEmbedding) -> Self {
        Self { diagram, embedding }
    }

    /// Canonical unit-sphere diagram.
    #[inline]
    pub fn diagram(&self) -> &SphericalVoronoi {
        &self.diagram
    }

    /// Mutable access to the canonical diagram, for operations such as vertex compaction.
    #[inline]
    pub fn diagram_mut(&mut self) -> &mut SphericalVoronoi {
        &mut self.diagram
    }

    /// World-space embedding.
    #[inline]
    pub fn embedding(&self) -> SphereEmbedding {
        self.embedding
    }

    /// Consume the wrapper and return its canonical diagram.
    #[inline]
    pub fn into_diagram(self) -> SphericalVoronoi {
        self.diagram
    }

    /// Consume the wrapper into its canonical diagram and embedding.
    #[inline]
    pub fn into_parts(self) -> (SphericalVoronoi, SphereEmbedding) {
        (self.diagram, self.embedding)
    }

    /// Canonicalized generator mapped onto the embedded sphere.
    ///
    /// This is not necessarily the original world-space input: radial distance
    /// is discarded and the backend canonicalizes each stored direction to f32.
    #[track_caller]
    pub fn generator_world(&self, index: usize) -> [f64; 3] {
        self.embedding
            .stored_unit_to_world(self.diagram.generator(index))
    }

    /// Voronoi vertex mapped onto the embedded sphere.
    #[track_caller]
    pub fn vertex_world(&self, index: usize) -> [f64; 3] {
        self.embedding
            .stored_unit_to_world(self.diagram.vertex(index))
    }

    /// Physical surface area of a cell in squared world-coordinate units.
    ///
    /// This is the core cell's solid angle multiplied by `radius²`. It can be
    /// positive infinity when the mathematical area exceeds finite f64 range.
    /// Welded twins report their canonical cell's area; sum only canonical
    /// cells when computing the sphere's total area.
    #[track_caller]
    pub fn cell_area_world(&self, index: usize) -> f64 {
        self.embedding
            .solid_angle_to_area(self.diagram.cell_area(index))
    }

    /// Spherical centroid/Lloyd target mapped onto the embedded sphere.
    ///
    /// This is the direction of the surface-position integral projected back
    /// to the shell, matching [`SphericalVoronoi::cell_centroid`]. It is not
    /// the unconstrained Euclidean center of mass of the curved patch.
    #[track_caller]
    pub fn cell_centroid_world(&self, index: usize) -> [f64; 3] {
        self.embedding
            .stored_unit_to_world(self.diagram.cell_centroid(index))
    }

    /// World-space Lloyd targets for all cells, in input order.
    pub fn lloyd_step_world(&self) -> Vec<[f64; 3]> {
        self.diagram
            .lloyd_step()
            .into_iter()
            .map(|direction| self.embedding.stored_unit_to_world(direction))
            .collect()
    }

    /// Build a point locator accepting world-space queries.
    pub fn build_locator_world(&self) -> EmbeddedSphereLocator {
        EmbeddedSphereLocator {
            locator: self.diagram.build_locator(),
            embedding: self.embedding,
        }
    }
}

/// Report-bearing result of an embedded computation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EmbeddedComputeOutput {
    /// Returned diagram, with one cell per original input point.
    pub diagram: EmbeddedSphericalVoronoi,
    /// Effective diagram actually solved after preprocessing, when welding changed the generator set.
    pub effective_diagram: Option<EmbeddedSphericalVoronoi>,
    /// Existing unit-backend preprocessing, repair, and validation report.
    pub report: ComputeReport,
}

impl EmbeddedComputeOutput {
    /// Preferred embedded diagram for interpreting the computation.
    #[inline]
    pub fn preferred_diagram(&self) -> &EmbeddedSphericalVoronoi {
        self.effective_diagram.as_ref().unwrap_or(&self.diagram)
    }
}

/// Compute an embedded spherical Voronoi diagram with default settings.
pub fn compute_on_sphere<P: WorldVec3Like>(
    points: &[P],
    embedding: SphereEmbedding,
) -> Result<EmbeddedSphericalVoronoi, VoronoiError> {
    compute_on_sphere_with(points, embedding, VoronoiConfig::default())
}

/// Compute an embedded spherical Voronoi diagram with explicit configuration.
pub fn compute_on_sphere_with<P: WorldVec3Like>(
    points: &[P],
    embedding: SphereEmbedding,
    config: VoronoiConfig,
) -> Result<EmbeddedSphericalVoronoi, VoronoiError> {
    let projected = project_points(points, embedding)?;
    let diagram =
        crate::knn_clipping::compute_voronoi_knn_clipping_with_config_owned(projected, &config)?;
    Ok(EmbeddedSphericalVoronoi::new(diagram, embedding))
}

/// Compute an embedded diagram and return preprocessing and validation metadata.
pub fn compute_on_sphere_with_report<P: WorldVec3Like>(
    points: &[P],
    embedding: SphereEmbedding,
    config: VoronoiConfig,
) -> Result<EmbeddedComputeOutput, VoronoiError> {
    let projected = project_points(points, embedding)?;
    let ComputeOutput {
        diagram,
        effective_diagram,
        report,
    } = crate::knn_clipping::compute_voronoi_knn_clipping_with_report_owned(projected, &config)?;
    Ok(EmbeddedComputeOutput {
        diagram: EmbeddedSphericalVoronoi::new(diagram, embedding),
        effective_diagram: effective_diagram
            .map(|diagram| EmbeddedSphericalVoronoi::new(diagram, embedding)),
        report,
    })
}

/// Point locator for an [`EmbeddedSphericalVoronoi`].
pub struct EmbeddedSphereLocator {
    locator: SphereLocator,
    embedding: SphereEmbedding,
}

impl EmbeddedSphereLocator {
    /// Locate one world-space query, returning its canonical cell index.
    pub fn locate_world<P: WorldVec3Like + ?Sized>(
        &mut self,
        query: &P,
    ) -> Result<usize, SphereProjectionError> {
        let direction = projected_unit_f32(self.embedding, query)?;
        Ok(self.locator.locate(&direction))
    }

    /// Locate world-space queries in input order.
    ///
    /// Conversion is performed deterministically before the existing parallel
    /// locator runs, so an error always identifies the lowest invalid index.
    pub fn locate_many_world<P: WorldVec3Like>(
        &self,
        queries: &[P],
    ) -> Result<Vec<usize>, IndexedSphereProjectionError> {
        let directions: Result<Vec<UnitVec3>, IndexedSphereProjectionError> = queries
            .iter()
            .enumerate()
            .map(|(point_index, query)| {
                projected_unit_f32(self.embedding, query).map_err(|source| {
                    IndexedSphereProjectionError {
                        point_index,
                        source,
                    }
                })
            })
            .collect();
        Ok(self.locator.locate_many(&directions?))
    }

    /// Embedding used to interpret world-space queries.
    #[inline]
    pub fn embedding(&self) -> SphereEmbedding {
        self.embedding
    }
}

#[cfg(feature = "serde")]
mod wire {
    use super::{SphereEmbedding, SphereEmbeddingError};

    #[derive(serde::Deserialize)]
    pub struct SphereEmbeddingWire {
        center: [f64; 3],
        radius: f64,
    }

    impl TryFrom<SphereEmbeddingWire> for SphereEmbedding {
        type Error = SphereEmbeddingError;

        fn try_from(value: SphereEmbeddingWire) -> Result<Self, Self::Error> {
            SphereEmbedding::new(value.center, value.radius)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn wire_conversion_rejects_non_json_f64_values() {
            assert!(matches!(
                SphereEmbedding::try_from(SphereEmbeddingWire {
                    center: [f64::NAN, 0.0, 0.0],
                    radius: 1.0,
                }),
                Err(SphereEmbeddingError::NonFiniteCenter { component: 0 })
            ));
            assert!(matches!(
                SphereEmbedding::try_from(SphereEmbeddingWire {
                    center: [0.0; 3],
                    radius: f64::INFINITY,
                }),
                Err(SphereEmbeddingError::InvalidRadius)
            ));
            assert!(matches!(
                SphereEmbedding::try_from(SphereEmbeddingWire {
                    center: [f64::MAX, 0.0, 0.0],
                    radius: 1.0,
                }),
                Err(SphereEmbeddingError::UnrepresentableExtent { component: 0 })
            ));
        }
    }
}

#[cfg(all(test, feature = "parallel"))]
mod projection_tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    fn in_two_thread_pool<R: Send>(op: impl FnOnce() -> R + Send) -> R {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(rayon::current_num_threads(), 2);
            op()
        })
    }

    fn non_axis_world_points(count: usize) -> Vec<[f64; 3]> {
        let golden_angle = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
        (0..count)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
                let radial = (1.0 - z * z).sqrt();
                let theta = i as f64 * golden_angle;
                [
                    7.0 + radial * theta.cos() * 3.0,
                    -11.0 + radial * theta.sin() * 3.0,
                    13.0 + z * 3.0,
                ]
            })
            .collect()
    }

    #[test]
    fn valid_parallel_projection_matches_scalar_order() {
        let embedding = SphereEmbedding::new([7.0, -11.0, 13.0], 3.0).unwrap();
        let world = non_axis_world_points(20_000);
        let expected: Vec<glam::Vec3> = world
            .iter()
            .map(|point| {
                let u = projected_unit_f32(embedding, point).unwrap();
                glam::Vec3::new(u.x, u.y, u.z)
            })
            .collect();

        let projected = in_two_thread_pool(|| project_points(&world, embedding).unwrap());
        assert_eq!(projected, expected);
    }

    struct StatefulPoint {
        xyz: [f64; 3],
        first_x_bits: AtomicU64,
    }

    impl WorldVec3Like for StatefulPoint {
        fn x(&self) -> f64 {
            f64::from_bits(
                self.first_x_bits
                    .swap(self.xyz[0].to_bits(), Ordering::Relaxed),
            )
        }

        fn y(&self) -> f64 {
            self.xyz[1]
        }

        fn z(&self) -> f64 {
            self.xyz[2]
        }
    }

    #[test]
    fn parallel_projection_snapshots_stateful_inputs_once() {
        let embedding = SphereEmbedding::new([0.0; 3], 1.0).unwrap();
        let world: Vec<StatefulPoint> = (0..20_000)
            .map(|i| StatefulPoint {
                xyz: [1.0, 0.0, 0.0],
                first_x_bits: AtomicU64::new(if i == 1_234 {
                    f64::NAN.to_bits()
                } else {
                    1.0f64.to_bits()
                }),
            })
            .collect();

        let result = in_two_thread_pool(|| project_points(&world, embedding));
        assert!(matches!(
            result,
            Err(VoronoiError::InvalidInput {
                point_index: 1_234,
                ..
            })
        ));
    }
}
