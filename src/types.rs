//! Core types for spherical Voronoi computation.

use bytemuck::{Pod, Zeroable};
use std::fmt;
use std::mem::{align_of, size_of, ManuallyDrop};

/// Maximum absolute error in the squared norm of a stored [`SpherePoint`].
///
/// A point is normalized in f64 and then rounded once to three f32
/// components. The bound deliberately includes rounding margin; stored
/// points are not claimed to have mathematically exact unit length.
pub const SPHERE_POINT_MAX_NORM_SQUARED_ERROR: f64 = 2.0 * f32::EPSILON as f64;

/// Why coordinates could not form or validate a [`SpherePoint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpherePointError {
    /// One coordinate is NaN or infinite.
    NonFinite {
        /// Component index: 0 = x, 1 = y, 2 = z.
        component: usize,
    },
    /// All coordinates are zero, so they do not define a direction.
    Directionless,
    /// Exact stored coordinates fall outside the documented norm envelope.
    OutsideStoredEnvelope,
}

impl fmt::Display for SpherePointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { component } => {
                write!(f, "sphere point component {component} is not finite")
            }
            Self::Directionless => write!(f, "sphere point coordinates do not define a direction"),
            Self::OutsideStoredEnvelope => write!(
                f,
                "stored sphere point lies outside the squared-norm envelope"
            ),
        }
    }
}

impl std::error::Error for SpherePointError {}

/// A checked, tightly packed direction on the unit sphere.
///
/// The three components are finite and their f64-evaluated squared norm is
/// within [`SPHERE_POINT_MAX_NORM_SQUARED_ERROR`] of one. Construction from a
/// direction normalizes in f64 and rounds once to f32. The private field keeps
/// arbitrary safe bit patterns—including zero and NaN—from masquerading as
/// certified spherical output.
///
/// `SpherePoint` is `#[repr(transparent)]` over `[f32; 3]`; this packed
/// 12-byte representation is a public layout commitment. It intentionally
/// does not implement `Pod`, `Zeroable`, or infallible conversion from raw
/// coordinates because those traits would invalidate the semantic contract.
///
/// ```compile_fail
/// use voronoi_mesh::SpherePoint;
///
/// // The packed field is private: raw coordinates must use checked construction.
/// let point = SpherePoint([1.0, 0.0, 0.0]);
/// # let _ = point;
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpherePoint([f32; 3]);

const _: () = assert!(size_of::<SpherePoint>() == size_of::<[f32; 3]>());
const _: () = assert!(align_of::<SpherePoint>() == align_of::<[f32; 3]>());
const _: () = assert!(size_of::<SpherePoint>() == size_of::<glam::Vec3>());
const _: () = assert!(align_of::<SpherePoint>() == align_of::<glam::Vec3>());

impl SpherePoint {
    /// Normalize finite, nonzero coordinates into a stored spherical point.
    pub fn try_from_xyz(xyz: [f32; 3]) -> Result<Self, SpherePointError> {
        for (component, value) in xyz.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(SpherePointError::NonFinite { component });
            }
        }
        let direction = glam::DVec3::new(xyz[0] as f64, xyz[1] as f64, xyz[2] as f64);
        let len_sq = direction.length_squared();
        if len_sq == 0.0 {
            return Err(SpherePointError::Directionless);
        }
        Ok(Self::from_direction_dvec3(direction))
    }

    /// Borrow the packed xyz components.
    #[inline]
    pub const fn as_array(&self) -> &[f32; 3] {
        &self.0
    }

    /// Return the packed xyz components.
    #[inline]
    pub const fn to_array(self) -> [f32; 3] {
        self.0
    }

    /// X component.
    #[inline]
    pub const fn x(self) -> f32 {
        self.0[0]
    }

    /// Y component.
    #[inline]
    pub const fn y(self) -> f32 {
        self.0[1]
    }

    /// Z component.
    #[inline]
    pub const fn z(self) -> f32 {
        self.0[2]
    }

    /// Dot product with another stored sphere point.
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }

    /// Squared length evaluated in f32.
    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Convert to a glam vector when the public `glam` feature is enabled.
    #[cfg(feature = "glam")]
    #[inline]
    pub fn to_glam(self) -> glam::Vec3 {
        glam::Vec3::from_array(self.0)
    }

    pub(crate) fn validate_stored(xyz: [f32; 3]) -> Result<Self, SpherePointError> {
        for (component, value) in xyz.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(SpherePointError::NonFinite { component });
            }
        }
        let x = xyz[0] as f64;
        let y = xyz[1] as f64;
        let z = xyz[2] as f64;
        let error = (x * x + y * y + z * z - 1.0).abs();
        if error <= SPHERE_POINT_MAX_NORM_SQUARED_ERROR {
            Ok(Self(xyz))
        } else {
            Err(SpherePointError::OutsideStoredEnvelope)
        }
    }

    #[inline]
    pub(crate) fn from_canonical_vec3(point: glam::Vec3) -> Self {
        Self::validate_stored(point.to_array())
            .expect("crate-produced sphere point must satisfy its stored envelope")
    }

    #[inline]
    pub(crate) fn from_direction_dvec3(direction: glam::DVec3) -> Self {
        Self::from_canonical_vec3(canonical_vec3_from_dvec3(direction))
    }
}

/// Normalize in f64 and round once into the backend's packed f32 vector.
#[inline]
pub(crate) fn canonical_vec3_from_dvec3(direction: glam::DVec3) -> glam::Vec3 {
    let normalized = direction / direction.length_squared().sqrt();
    glam::Vec3::new(
        normalized.x as f32,
        normalized.y as f32,
        normalized.z as f32,
    )
}

/// Transfer a certified backend allocation into checked public storage.
///
/// # Safety
///
/// Every vector must satisfy the `SpherePoint` finite norm envelope. Layout
/// compatibility is asserted above; the source vector must not be used after
/// this ownership transfer.
pub(crate) unsafe fn sphere_points_from_vec3(points: Vec<glam::Vec3>) -> Vec<SpherePoint> {
    debug_assert!(points
        .iter()
        .all(|point| SpherePoint::validate_stored(point.to_array()).is_ok()));
    let mut points = ManuallyDrop::new(points);
    let ptr = points.as_mut_ptr().cast::<SpherePoint>();
    let len = points.len();
    let capacity = points.capacity();
    // SAFETY: the caller certifies the semantic invariant, the compile-time
    // assertions establish identical element layout, and ManuallyDrop leaves
    // this allocation with exactly one owner.
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

#[inline]
pub(crate) fn sphere_points_as_xyz(points: &[SpherePoint]) -> &[[f32; 3]] {
    // SAFETY: SpherePoint is repr(transparent) over [f32; 3], and the returned
    // shared slice cannot manufacture an invalid SpherePoint.
    unsafe { std::slice::from_raw_parts(points.as_ptr().cast(), points.len()) }
}

impl From<SpherePoint> for [f32; 3] {
    #[inline]
    fn from(point: SpherePoint) -> Self {
        point.to_array()
    }
}

impl AsRef<[f32; 3]> for SpherePoint {
    #[inline]
    fn as_ref(&self) -> &[f32; 3] {
        self.as_array()
    }
}

impl TryFrom<[f32; 3]> for SpherePoint {
    type Error = SpherePointError;

    #[inline]
    fn try_from(xyz: [f32; 3]) -> Result<Self, Self::Error> {
        Self::try_from_xyz(xyz)
    }
}

#[cfg(feature = "glam")]
impl From<SpherePoint> for glam::Vec3 {
    #[inline]
    fn from(point: SpherePoint) -> Self {
        point.to_glam()
    }
}

#[cfg(feature = "glam")]
impl TryFrom<glam::Vec3> for SpherePoint {
    type Error = SpherePointError;

    #[inline]
    fn try_from(point: glam::Vec3) -> Result<Self, Self::Error> {
        Self::try_from_xyz(point.to_array())
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for SpherePoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde::Serialize::serialize(&self.0, serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SpherePoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let xyz = <[f32; 3] as serde::Deserialize>::deserialize(deserializer)?;
        Self::validate_stored(xyz).map_err(serde::de::Error::custom)
    }
}

/// An unchecked raw f32 input adapter for a point on the unit sphere.
///
/// This type provides a small `#[repr(C)]` representation with a stable layout.
/// Points are assumed to be on (or near) the unit sphere; the crate may
/// debug-assert normalization but does not normalize this value in place.
/// Computation canonicalizes supported input once and returns checked
/// [`SpherePoint`] values instead. Prefer `[f32; 3]` when a named input type is
/// unnecessary.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnitVec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl UnitVec3 {
    /// Create a new unit vector.
    ///
    /// Note: This does NOT normalize the input. The caller is responsible
    /// for ensuring the vector is on the unit sphere.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Create from any type implementing `UnitVec3Like`.
    #[inline]
    pub fn from_like<P: UnitVec3Like>(p: &P) -> Self {
        Self::new(p.x(), p.y(), p.z())
    }

    /// Convert to a glam::Vec3 (available when the `glam` feature is enabled).
    #[cfg(feature = "glam")]
    #[inline]
    pub fn to_glam(self) -> glam::Vec3 {
        glam::Vec3::new(self.x, self.y, self.z)
    }

    /// Create from a glam::Vec3 (available when the `glam` feature is enabled).
    #[cfg(feature = "glam")]
    #[inline]
    pub fn from_glam(v: glam::Vec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    /// Compute the dot product with another vector.
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Compute the squared length.
    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Compute the length.
    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Normalize the vector.
    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 0.0 {
            Self::new(self.x / len, self.y / len, self.z / len)
        } else {
            self
        }
    }
}

impl From<[f32; 3]> for UnitVec3 {
    #[inline]
    fn from([x, y, z]: [f32; 3]) -> Self {
        Self::new(x, y, z)
    }
}

impl From<UnitVec3> for [f32; 3] {
    #[inline]
    fn from(v: UnitVec3) -> Self {
        [v.x, v.y, v.z]
    }
}

#[cfg(feature = "glam")]
impl From<glam::Vec3> for UnitVec3 {
    #[inline]
    fn from(v: glam::Vec3) -> Self {
        Self::from_glam(v)
    }
}

#[cfg(feature = "glam")]
impl From<UnitVec3> for glam::Vec3 {
    #[inline]
    fn from(v: UnitVec3) -> glam::Vec3 {
        v.to_glam()
    }
}

/// Trait for types that can be used as input points.
///
/// This allows zero-copy input from various math libraries.
pub trait UnitVec3Like {
    /// X component.
    fn x(&self) -> f32;
    /// Y component.
    fn y(&self) -> f32;
    /// Z component.
    fn z(&self) -> f32;
}

impl UnitVec3Like for UnitVec3 {
    #[inline]
    fn x(&self) -> f32 {
        self.x
    }
    #[inline]
    fn y(&self) -> f32 {
        self.y
    }
    #[inline]
    fn z(&self) -> f32 {
        self.z
    }
}

impl UnitVec3Like for SpherePoint {
    #[inline]
    fn x(&self) -> f32 {
        (*self).x()
    }

    #[inline]
    fn y(&self) -> f32 {
        (*self).y()
    }

    #[inline]
    fn z(&self) -> f32 {
        (*self).z()
    }
}

impl UnitVec3Like for [f32; 3] {
    #[inline]
    fn x(&self) -> f32 {
        self[0]
    }
    #[inline]
    fn y(&self) -> f32 {
        self[1]
    }
    #[inline]
    fn z(&self) -> f32 {
        self[2]
    }
}

impl UnitVec3Like for (f32, f32, f32) {
    #[inline]
    fn x(&self) -> f32 {
        self.0
    }
    #[inline]
    fn y(&self) -> f32 {
        self.1
    }
    #[inline]
    fn z(&self) -> f32 {
        self.2
    }
}

// Implement for glam::Vec3 when the feature is enabled.
#[cfg(feature = "glam")]
impl UnitVec3Like for glam::Vec3 {
    #[inline]
    fn x(&self) -> f32 {
        self.x
    }
    #[inline]
    fn y(&self) -> f32 {
        self.y
    }
    #[inline]
    fn z(&self) -> f32 {
        self.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_point_normalizes_in_f64_and_rounds_once() {
        let point = SpherePoint::try_from_xyz([3.0, 4.0, 0.0]).unwrap();
        assert_eq!(point.to_array(), [0.6, 0.8, 0.0]);
        let [x, y, z] = point.to_array();
        let norm_squared =
            (x as f64).mul_add(x as f64, (y as f64).mul_add(y as f64, z as f64 * z as f64));
        assert!((norm_squared - 1.0).abs() <= SPHERE_POINT_MAX_NORM_SQUARED_ERROR);
    }

    #[test]
    fn sphere_point_rejects_zero_and_non_finite_directions() {
        assert_eq!(
            SpherePoint::try_from_xyz([0.0; 3]),
            Err(SpherePointError::Directionless)
        );
        assert_eq!(
            SpherePoint::try_from_xyz([1.0, f32::NAN, 0.0]),
            Err(SpherePointError::NonFinite { component: 1 })
        );
        assert_eq!(
            SpherePoint::try_from_xyz([1.0, 0.0, f32::INFINITY]),
            Err(SpherePointError::NonFinite { component: 2 })
        );
    }

    #[test]
    fn sphere_point_layout_and_xyz_view_are_exact() {
        assert_eq!(size_of::<SpherePoint>(), size_of::<[f32; 3]>());
        assert_eq!(align_of::<SpherePoint>(), align_of::<[f32; 3]>());

        let points = [
            SpherePoint::try_from_xyz([1.0, 2.0, 3.0]).unwrap(),
            SpherePoint::try_from_xyz([-4.0, 5.0, -6.0]).unwrap(),
        ];
        let xyz = sphere_points_as_xyz(&points);
        assert_eq!(xyz.as_ptr().cast::<u8>(), points.as_ptr().cast::<u8>());
        for (point, packed) in points.iter().zip(xyz) {
            assert_eq!(packed.map(f32::to_bits), point.to_array().map(f32::to_bits));
        }
    }

    #[test]
    fn vec3_allocation_transfer_preserves_pointer_capacity_and_bits() {
        let mut source = Vec::with_capacity(8);
        source.push(canonical_vec3_from_dvec3(glam::DVec3::new(1.0, 2.0, 3.0)));
        source.push(canonical_vec3_from_dvec3(glam::DVec3::new(-4.0, 5.0, -6.0)));
        let pointer = source.as_ptr().cast::<u8>();
        let capacity = source.capacity();
        let expected: Vec<[u32; 3]> = source
            .iter()
            .map(|point| point.to_array().map(f32::to_bits))
            .collect();

        // SAFETY: the shared canonicalizer produced every source element.
        let transferred = unsafe { sphere_points_from_vec3(source) };
        assert_eq!(transferred.as_ptr().cast::<u8>(), pointer);
        assert_eq!(transferred.capacity(), capacity);
        let actual: Vec<[u32; 3]> = transferred
            .iter()
            .map(|point| point.to_array().map(f32::to_bits))
            .collect();
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn sphere_point_serde_preserves_valid_bits_and_rejects_invalid_storage() {
        let point = SpherePoint::try_from_xyz([1.0, 2.0, 3.0]).unwrap();
        let encoded = serde_json::to_string(&point).unwrap();
        assert!(
            encoded.starts_with('['),
            "wire shape must be a packed xyz triple"
        );
        let decoded: SpherePoint = serde_json::from_str(&encoded).unwrap();
        assert_eq!(
            decoded.to_array().map(f32::to_bits),
            point.to_array().map(f32::to_bits)
        );

        let zero = serde_json::from_str::<SpherePoint>("[0.0,0.0,0.0]").unwrap_err();
        assert!(zero.to_string().contains("envelope"));
        let off_sphere = serde_json::from_str::<SpherePoint>("[0.5,0.5,0.5]").unwrap_err();
        assert!(off_sphere.to_string().contains("envelope"));
    }

    #[test]
    fn test_unit_vec3_basics() {
        let v = UnitVec3::new(1.0, 0.0, 0.0);
        assert_eq!(v.length(), 1.0);
        assert_eq!(v.dot(v), 1.0);
    }

    #[test]
    fn test_from_array() {
        let v: UnitVec3 = [0.0, 1.0, 0.0].into();
        assert_eq!(v.y, 1.0);
    }

    #[test]
    fn test_unit_vec3_like_trait() {
        fn accepts_like<P: UnitVec3Like>(p: &P) -> f32 {
            p.x() + p.y() + p.z()
        }

        let uv = UnitVec3::new(1.0, 2.0, 3.0);
        let arr = [1.0f32, 2.0, 3.0];
        let tuple = (1.0f32, 2.0f32, 3.0f32);

        assert_eq!(accepts_like(&uv), 6.0);
        assert_eq!(accepts_like(&arr), 6.0);
        assert_eq!(accepts_like(&tuple), 6.0);
    }

    #[test]
    #[cfg(feature = "glam")]
    fn test_unit_vec3_like_trait_glam() {
        fn accepts_like<P: UnitVec3Like>(p: &P) -> f32 {
            p.x() + p.y() + p.z()
        }

        let glam_v = glam::Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(accepts_like(&glam_v), 6.0);
    }
}
