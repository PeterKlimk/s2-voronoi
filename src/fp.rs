//! Floating-point helpers and the 8-lane SIMD backend seam.
//!
//! All explicit SIMD in the crate goes through [`PointChunk8`] / [`Dots8`],
//! so the backend is swappable for benchmarking and portability:
//!
//! - default: the `wide` crate (explicit SIMD on stable Rust)
//! - `simd_scalar`: plain arrays relying on autovectorization (debugging /
//!   comparison; a few percent slower on generic targets)
//!
//! All backends are lane-wise only (no horizontal reductions), so results
//! are bit-identical across them (see tests/backend_fingerprint.rs).
//!
//! History: the original backend was nightly `portable_simd`. Benchmarks on
//! the reference Ryzen 3600 (2026-06) showed `wide` at parity (within ~1-2%,
//! winning some runs), so the nightly requirement was dropped. The portable
//! backend can be recovered from git history if `std::simd` stabilizes.

// The `fma` feature is only sound when the SIMD backend actually fuses. On
// x86/x86_64 the `wide` crate's `mul_add` lowers to a true FMA *only* when the
// `fma` target feature is enabled; without it, `wide` does a two-rounding
// `a*b+c`. The scalar `f32::mul_add` (below) always fuses, so the SIMD and
// scalar kNN-distance paths would round differently and desymmetrize neighbor
// selection (cell i admits neighbor j while j drops i -> unpaired interior
// edge). Require `+fma` so both paths fuse identically. The `simd_scalar`
// backend routes its SIMD dot through `f32::mul_add` too, so it is exempt.
#[cfg(all(
    feature = "fma",
    not(feature = "simd_scalar"),
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "fma")
))]
compile_error!(
    "the `fma` feature requires the `fma` target feature on x86/x86_64; build \
     with `RUSTFLAGS=\"-C target-feature=+fma\"` or `-C target-cpu=native`. \
     Without it the `wide` SIMD dot and the scalar dot round differently and \
     kNN neighbor selection can desymmetrize, producing invalid topology. Use \
     the `simd_scalar` backend if you need `fma` without HW FMA."
);

#[inline(always)]
pub(crate) fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(feature = "fma")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "fma"))]
    {
        a * b + c
    }
}

#[inline(always)]
pub(crate) fn fma_f64(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(feature = "fma")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "fma"))]
    {
        a * b + c
    }
}

#[inline(always)]
pub(crate) fn dot3_f32(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32 {
    // Match the baseline left-associative order:
    //   (ax*bx + ay*by) + az*bz
    let ab = fma_f32(ax, bx, ay * by);
    fma_f32(az, bz, ab)
}

/// Eight candidate points in SoA form, loaded once and dotted against many
/// queries (the load hoisting is load-bearing in the packed kNN hot loops).
pub(crate) struct PointChunk8 {
    x: backend::V,
    y: backend::V,
    z: backend::V,
}

/// Eight dot products; mask extraction is cheap and `to_array` is deferred
/// until a mask is known to be non-empty.
pub(crate) struct Dots8(backend::V);

impl PointChunk8 {
    #[inline(always)]
    pub(crate) fn from_arrays(x: [f32; 8], y: [f32; 8], z: [f32; 8]) -> Self {
        Self {
            x: backend::load_array(x),
            y: backend::load_array(y),
            z: backend::load_array(z),
        }
    }

    #[inline(always)]
    pub(crate) fn from_array_refs(x: &[f32; 8], y: &[f32; 8], z: &[f32; 8]) -> Self {
        Self::from_arrays(*x, *y, *z)
    }

    /// Lane-wise dot products against a broadcast query point.
    #[inline(always)]
    pub(crate) fn dots(&self, qx: f32, qy: f32, qz: f32) -> Dots8 {
        Dots8(backend::dot3(self.x, self.y, self.z, qx, qy, qz))
    }
}

impl Dots8 {
    /// Bitmask of lanes with dot > threshold (lane i -> bit i).
    #[inline(always)]
    pub(crate) fn mask_gt(&self, threshold: f32) -> u32 {
        backend::mask_gt(self.0, threshold)
    }

    #[inline(always)]
    pub(crate) fn to_array(&self) -> [f32; 8] {
        backend::to_array(self.0)
    }
}

/// Convert eight positive finite inward-plane distances to conservative
/// outside-dot thresholds. The returned mask marks lanes where this interior
/// formula is valid; callers retain their geometric fallback for other lanes.
#[inline(always)]
#[cfg(any(test, all(target_feature = "avx2", not(feature = "simd_scalar"))))]
pub(crate) fn interior_security_thresholds8(s_min: [f32; 8], pad: f32) -> ([f32; 8], u32) {
    backend::interior_security_thresholds8(backend::load_array(s_min), pad)
}

/// Signed distances of the first 8 polygon vertices to a half-plane
/// (`fma_f64(a, u, fma_f64(b, v, c))`, lane-exact with the scalar formula)
/// plus the inside bitmask (`d >= neg_eps`, lane i -> bit i).
///
/// Callers may pass slices longer than the live vertex count; lanes past it
/// compute on stale-but-finite data and must be masked off by the caller.
#[inline(always)]
pub(crate) fn signed_dists_mask8(
    a: f64,
    b: f64,
    c: f64,
    us: &[f64; 8],
    vs: &[f64; 8],
    neg_eps: f64,
) -> ([f64; 8], u32) {
    backend::signed_dists_mask8(a, b, c, us, vs, neg_eps)
}

/// Like `signed_dists_mask8` but only evaluates the first four lanes — used by
/// the N <= 4 clip kernels (triangles/quads) to halve the SIMD distance eval.
/// Reads `us[0..4]`/`vs[0..4]`; the result is bit-identical to lanes 0..4 of
/// the eight-lane path.
#[inline(always)]
pub(crate) fn signed_dists_mask4(
    a: f64,
    b: f64,
    c: f64,
    us: &[f64; 8],
    vs: &[f64; 8],
    neg_eps: f64,
) -> ([f64; 4], u32) {
    backend::signed_dists_mask4(a, b, c, us, vs, neg_eps)
}

#[cfg(not(feature = "simd_scalar"))]
mod backend {
    #[cfg(any(test, target_feature = "avx2"))]
    use wide::CmpLe;
    use wide::{f32x8, f64x4, CmpGe, CmpGt};

    pub(super) type V = f32x8;

    #[inline(always)]
    pub(super) fn load_array(a: [f32; 8]) -> V {
        f32x8::from(a)
    }

    #[inline(always)]
    pub(super) fn dot3(x: V, y: V, z: V, qx: f32, qy: f32, qz: f32) -> V {
        let qx = f32x8::splat(qx);
        let qy = f32x8::splat(qy);
        let qz = f32x8::splat(qz);
        #[cfg(feature = "fma")]
        {
            let ab = x.mul_add(qx, y * qy);
            z.mul_add(qz, ab)
        }
        #[cfg(not(feature = "fma"))]
        {
            (x * qx + y * qy) + z * qz
        }
    }

    #[inline(always)]
    pub(super) fn mask_gt(v: V, threshold: f32) -> u32 {
        v.cmp_gt(f32x8::splat(threshold)).move_mask() as u32 & 0xff
    }

    #[inline(always)]
    pub(super) fn to_array(v: V) -> [f32; 8] {
        v.to_array()
    }

    #[inline(always)]
    #[cfg(any(test, target_feature = "avx2"))]
    pub(super) fn interior_security_thresholds8(v: V, pad: f32) -> ([f32; 8], u32) {
        let zero = f32x8::ZERO;
        let valid = v.cmp_gt(zero) & v.abs().cmp_le(f32x8::splat(f32::MAX));
        let s = (v - f32x8::splat(pad)).max(zero).min(f32x8::ONE);
        let threshold = (f32x8::ONE - s * s).max(zero).sqrt();
        (threshold.to_array(), valid.move_mask() as u32 & 0xff)
    }

    #[inline(always)]
    fn dists4(a: f64, b: f64, c: f64, u: f64x4, v: f64x4) -> f64x4 {
        #[cfg(feature = "fma")]
        {
            u.mul_add(f64x4::splat(a), v.mul_add(f64x4::splat(b), f64x4::splat(c)))
        }
        #[cfg(not(feature = "fma"))]
        {
            f64x4::splat(a) * u + (f64x4::splat(b) * v + f64x4::splat(c))
        }
    }

    #[inline(always)]
    pub(super) fn signed_dists_mask8(
        a: f64,
        b: f64,
        c: f64,
        us: &[f64; 8],
        vs: &[f64; 8],
        neg_eps: f64,
    ) -> ([f64; 8], u32) {
        let u_lo = f64x4::from([us[0], us[1], us[2], us[3]]);
        let u_hi = f64x4::from([us[4], us[5], us[6], us[7]]);
        let v_lo = f64x4::from([vs[0], vs[1], vs[2], vs[3]]);
        let v_hi = f64x4::from([vs[4], vs[5], vs[6], vs[7]]);
        let d_lo = dists4(a, b, c, u_lo, v_lo);
        let d_hi = dists4(a, b, c, u_hi, v_hi);
        let eps4 = f64x4::splat(neg_eps);
        let mask = (d_lo.cmp_ge(eps4).move_mask() as u32 & 0xf)
            | ((d_hi.cmp_ge(eps4).move_mask() as u32 & 0xf) << 4);
        let lo = d_lo.to_array();
        let hi = d_hi.to_array();
        (
            [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]],
            mask,
        )
    }

    /// Four-lane twin of `signed_dists_mask8` for clipping polygons with
    /// N <= 4 (triangles/quads): one `f64x4` instead of two, bit-identical to
    /// lanes 0..4 of the eight-lane path. Only `us[0..4]`/`vs[0..4]` are read.
    #[inline(always)]
    pub(super) fn signed_dists_mask4(
        a: f64,
        b: f64,
        c: f64,
        us: &[f64; 8],
        vs: &[f64; 8],
        neg_eps: f64,
    ) -> ([f64; 4], u32) {
        let u = f64x4::from([us[0], us[1], us[2], us[3]]);
        let v = f64x4::from([vs[0], vs[1], vs[2], vs[3]]);
        let d = dists4(a, b, c, u, v);
        let mask = d.cmp_ge(f64x4::splat(neg_eps)).move_mask() as u32 & 0xf;
        (d.to_array(), mask)
    }
}

#[cfg(test)]
mod tests {
    use super::{dot3_f32, interior_security_thresholds8, PointChunk8};

    #[test]
    fn point_chunk_dot_and_strict_mask_match_scalar_at_adjacent_thresholds() {
        let xs = [
            1.0,
            -1.0,
            0.577_350_26,
            -0.577_350_26,
            0.999_999_94,
            1.0e-7,
            -0.75,
            0.3125,
        ];
        let ys = [
            0.0,
            1.0e-7,
            -0.577_350_26,
            0.577_350_26,
            -3.0e-4,
            -1.0,
            0.5,
            -0.9375,
        ];
        let zs = [
            1.0e-7,
            0.0,
            0.577_350_26,
            0.577_350_26,
            2.0e-4,
            3.0e-7,
            -0.433_012_7,
            0.15625,
        ];
        let (qx, qy, qz) = (0.577_350_26, -0.707_106_77, 0.408_248_3);
        let dots = PointChunk8::from_arrays(xs, ys, zs).dots(qx, qy, qz);
        let lanes = dots.to_array();

        for lane in 0..8 {
            let scalar = dot3_f32(xs[lane], ys[lane], zs[lane], qx, qy, qz);
            assert_eq!(lanes[lane].to_bits(), scalar.to_bits(), "lane {lane}");

            let bit = 1u32 << lane;
            assert_ne!(dots.mask_gt(scalar.next_down()) & bit, 0, "lane {lane}");
            assert_eq!(dots.mask_gt(scalar) & bit, 0, "lane {lane}");
            assert_eq!(dots.mask_gt(scalar.next_up()) & bit, 0, "lane {lane}");
        }
    }

    #[test]
    fn interior_security_simd_matches_scalar_and_marks_fallback_lanes() {
        let pad = crate::tolerances::GRID_PLANE_PAD;
        let input = [
            0.5,
            pad * 0.5,
            1.0,
            2.0,
            f32::MIN_POSITIVE,
            0.0,
            -0.25,
            f32::INFINITY,
        ];
        let (actual, mask) = interior_security_thresholds8(input, pad);
        assert_eq!(mask, 0b0001_1111);
        for lane in 0..5 {
            let s = (input[lane] - pad).clamp(0.0, 1.0);
            let expected = (1.0 - s * s).max(0.0).sqrt();
            assert_eq!(actual[lane].to_bits(), expected.to_bits(), "lane {lane}");
        }

        let (_, nan_mask) = interior_security_thresholds8([f32::NAN; 8], pad);
        assert_eq!(nan_mask, 0);
    }
}

#[cfg(feature = "simd_scalar")]
mod backend {
    pub(super) type V = [f32; 8];

    #[inline(always)]
    pub(super) fn load_array(a: [f32; 8]) -> V {
        a
    }

    #[inline(always)]
    pub(super) fn dot3(x: V, y: V, z: V, qx: f32, qy: f32, qz: f32) -> V {
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = super::fma_f32(z[i], qz, super::fma_f32(x[i], qx, y[i] * qy));
        }
        out
    }

    #[inline(always)]
    pub(super) fn mask_gt(v: V, threshold: f32) -> u32 {
        let mut bits = 0u32;
        for (i, &d) in v.iter().enumerate() {
            bits |= u32::from(d > threshold) << i;
        }
        bits
    }

    #[inline(always)]
    pub(super) fn to_array(v: V) -> [f32; 8] {
        v
    }

    #[inline(always)]
    #[cfg(test)]
    pub(super) fn interior_security_thresholds8(v: V, pad: f32) -> ([f32; 8], u32) {
        let mut out = [0.0; 8];
        let mut valid = 0u32;
        for i in 0..8 {
            if v[i] > 0.0 && v[i].is_finite() {
                let s = (v[i] - pad).clamp(0.0, 1.0);
                out[i] = (1.0 - s * s).max(0.0).sqrt();
                valid |= 1 << i;
            }
        }
        (out, valid)
    }

    #[inline(always)]
    pub(super) fn signed_dists_mask8(
        a: f64,
        b: f64,
        c: f64,
        us: &[f64; 8],
        vs: &[f64; 8],
        neg_eps: f64,
    ) -> ([f64; 8], u32) {
        let mut dists = [0.0f64; 8];
        let mut mask = 0u32;
        for i in 0..8 {
            let d = super::fma_f64(a, us[i], super::fma_f64(b, vs[i], c));
            dists[i] = d;
            mask |= u32::from(d >= neg_eps) << i;
        }
        (dists, mask)
    }

    /// Four-lane twin of `signed_dists_mask8` (scalar backend); see the wide
    /// backend's version. Bit-identical to lanes 0..4 of the eight-lane path.
    #[inline(always)]
    pub(super) fn signed_dists_mask4(
        a: f64,
        b: f64,
        c: f64,
        us: &[f64; 8],
        vs: &[f64; 8],
        neg_eps: f64,
    ) -> ([f64; 4], u32) {
        let mut dists = [0.0f64; 4];
        let mut mask = 0u32;
        for i in 0..4 {
            let d = super::fma_f64(a, us[i], super::fma_f64(b, vs[i], c));
            dists[i] = d;
            mask |= u32::from(d >= neg_eps) << i;
        }
        (dists, mask)
    }
}
