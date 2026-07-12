use glam::Vec3;
use std::hint::select_unpredictable;

// S2-style quadratic projection to reduce cube map distortion.
// Maps UV in [-1, 1] to ST in [0, 1] with area-equalizing transform.
// Corners get compressed (larger solid angle -> fewer cells),
// centers get expanded (smaller solid angle -> more cells).

/// S2 quadratic transform: UV [-1, 1] -> ST [0, 1]
///
/// For every finite `f32`, this is bit-identical to selecting between
/// `0.5 * sqrt(1 + 3u)` and `1 - 0.5 * sqrt(1 - 3u)`. Production generators
/// are validated as finite before grid construction; signed-NaN propagation
/// outside that contract is intentionally unspecified.
#[inline]
pub(crate) fn uv_to_st(u: f32) -> f32 {
    let positive = 0.5 * (1.0 + 3.0 * u.abs()).sqrt();
    let negative = 1.0 - positive;
    select_unpredictable(u >= 0.0, positive, negative)
}

/// S2 inverse transform: ST [0, 1] -> UV [-1, 1]
#[inline]
pub(crate) fn st_to_uv(s: f32) -> f32 {
    if s >= 0.5 {
        (1.0 / 3.0) * (4.0 * s * s - 1.0)
    } else {
        (1.0 / 3.0) * (1.0 - 4.0 * (1.0 - s) * (1.0 - s))
    }
}

/// Map a point on unit sphere to (face, u, v) where u,v in [-1, 1].
#[inline]
pub(crate) fn point_to_face_uv(p: Vec3) -> (usize, f32, f32) {
    let (x, y, z) = (p.x, p.y, p.z);
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());

    if ax >= ay && ax >= az {
        // +/-X
        let inv = 1.0 / ax;
        if x >= 0.0 {
            (0, -z * inv, y * inv)
        } else {
            (1, z * inv, y * inv)
        }
    } else if ay >= ax && ay >= az {
        // +/-Y
        let inv = 1.0 / ay;
        if y >= 0.0 {
            (2, x * inv, -z * inv)
        } else {
            (3, x * inv, z * inv)
        }
    } else {
        // +/-Z
        let inv = 1.0 / az;
        if z >= 0.0 {
            (4, x * inv, y * inv)
        } else {
            (5, -x * inv, y * inv)
        }
    }
}

/// Convert (face, u, v) to cell index.
#[inline]
pub(crate) fn face_uv_to_cell(face: usize, u: f32, v: f32, res: usize) -> usize {
    // Map UV [-1, 1] -> ST [0, 1] using the S2 quadratic transform.
    let su = uv_to_st(u);
    let sv = uv_to_st(v);
    let fu = (su * res as f32).max(0.0);
    let fv = (sv * res as f32).max(0.0);
    let iu = (fu as usize).min(res - 1);
    let iv = (fv as usize).min(res - 1);
    face * res * res + iv * res + iu
}

/// Convert (face, u, v) back to a 3D point (inverse of point_to_face_uv).
#[inline]
pub(crate) fn face_uv_to_3d(face: usize, u: f32, v: f32) -> Vec3 {
    // Project onto cube face, then normalize to sphere
    let p = match face {
        0 => Vec3::new(1.0, v, -u),  // +X: u = -z/x, v = y/x
        1 => Vec3::new(-1.0, v, u),  // -X: u = z/|x|, v = y/|x|
        2 => Vec3::new(u, 1.0, -v),  // +Y: u = x/y, v = -z/y
        3 => Vec3::new(u, -1.0, v),  // -Y: u = x/|y|, v = z/|y|
        4 => Vec3::new(u, v, 1.0),   // +Z: u = x/z, v = y/z
        5 => Vec3::new(-u, v, -1.0), // -Z: u = -x/|z|, v = y/|z|
        _ => unreachable!(),
    };
    p.normalize()
}

/// Convert cell index to (face, iu, iv).
#[inline]
pub(crate) fn cell_to_face_ij(cell: usize, res: usize) -> (usize, usize, usize) {
    let face = cell / (res * res);
    let rem = cell % (res * res);
    let iv = rem / res;
    let iu = rem % res;
    (face, iu, iv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn uv_to_st_branch_reference(u: f32) -> f32 {
        if u >= 0.0 {
            0.5 * (1.0 + 3.0 * u).sqrt()
        } else {
            1.0 - 0.5 * (1.0 - 3.0 * u).sqrt()
        }
    }

    #[track_caller]
    fn assert_uv_bits_match(u: f32) {
        assert_eq!(
            uv_to_st(u).to_bits(),
            uv_to_st_branch_reference(u).to_bits(),
            "uv_to_st bit mismatch for u={u:?} (bits={:#010x})",
            u.to_bits()
        );
    }

    #[test]
    fn branchless_uv_to_st_matches_branch_formula_for_finite_bits() {
        // Exhaust every positive and negative subnormal encoding, including
        // both signed zeros. This is the sign/absolute-value corner where a
        // branchless rewrite is easiest to get subtly wrong.
        for fraction in 0..=0x007f_ffffu32 {
            assert_uv_bits_match(f32::from_bits(fraction));
            assert_uv_bits_match(f32::from_bits(0x8000_0000 | fraction));
        }

        // Cover every finite exponent at the mantissa extrema and midpoint,
        // plus the values immediately adjacent to those representatives.
        const FRACTIONS: [u32; 5] = [0, 1, 0x003f_ffff, 0x007f_fffe, 0x007f_ffff];
        for exponent in 1..=254u32 {
            for fraction in FRACTIONS {
                let magnitude = (exponent << 23) | fraction;
                assert_uv_bits_match(f32::from_bits(magnitude));
                assert_uv_bits_match(f32::from_bits(0x8000_0000 | magnitude));
            }
        }

        // Valid-domain endpoints and their immediate finite neighbors.
        for bits in [
            (-1.0f32).to_bits() - 1,
            (-1.0f32).to_bits(),
            (-1.0f32).to_bits() + 1,
            1.0f32.to_bits() - 1,
            1.0f32.to_bits(),
            1.0f32.to_bits() + 1,
        ] {
            assert_uv_bits_match(f32::from_bits(bits));
        }
    }

    #[inline]
    fn next_down(v: f32) -> f32 {
        if v == 0.0 {
            return f32::from_bits(0x8000_0001);
        }
        let bits = v.to_bits();
        f32::from_bits(if v > 0.0 { bits - 1 } else { bits + 1 })
    }

    #[inline]
    fn next_up(v: f32) -> f32 {
        if v == 0.0 {
            return f32::from_bits(1);
        }
        let bits = v.to_bits();
        f32::from_bits(if v > 0.0 { bits + 1 } else { bits - 1 })
    }

    fn face_uv_to_cell_branch_reference(face: usize, u: f32, v: f32, res: usize) -> usize {
        let su = uv_to_st_branch_reference(u);
        let sv = uv_to_st_branch_reference(v);
        let fu = (su * res as f32).max(0.0);
        let fv = (sv * res as f32).max(0.0);
        let iu = (fu as usize).min(res - 1);
        let iv = (fv as usize).min(res - 1);
        face * res * res + iv * res + iu
    }

    #[test]
    fn cell_assignment_matches_branch_formula_around_grid_lines() {
        for res in [1usize, 2, 3, 4, 26, 58, 267, 1024, 26_754] {
            let res_f = res as f32;
            for line in 0..=res {
                let u = st_to_uv(line as f32 / res_f);
                for adjacent in [next_down(u), u, next_up(u)] {
                    for face in 0..6 {
                        for (sample_u, sample_v) in
                            [(adjacent, 0.0), (0.0, adjacent), (adjacent, -adjacent)]
                        {
                            assert_eq!(
                                face_uv_to_cell(face, sample_u, sample_v, res),
                                face_uv_to_cell_branch_reference(face, sample_u, sample_v, res),
                                "cell mismatch at face={face} res={res} line={line} \
                                 u={sample_u:?} v={sample_v:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}
