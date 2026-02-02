//! Core types for spherical Voronoi computation.

use bytemuck::{Pod, Zeroable};

/// A point on the unit sphere, represented as a 3D unit vector.
///
/// This type provides a small `#[repr(C)]` representation with a stable layout.
/// Points are assumed to be on (or near) the unit sphere; the crate may
/// debug-assert normalization but does not normalize inputs.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct UnitVec3 {
    pub x: f32,
    pub y: f32,
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
    fn x(&self) -> f32;
    fn y(&self) -> f32;
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
