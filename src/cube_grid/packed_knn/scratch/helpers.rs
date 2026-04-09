use super::super::super::{cell_to_face_ij, CubeMapGrid};
#[cfg(feature = "packed_knn_sort_small")]
use crate::sort::sort_small as sort_small_u64;
use glam::Vec3;

#[inline(always)]
pub(super) fn unpack_bin_local(packed: u32, local_shift: u32, local_mask: u32) -> (u8, u32) {
    let bin = (packed >> local_shift) as u8;
    let local = packed & local_mask;
    (bin, local)
}

#[inline(always)]
fn f32_to_ordered_u32(val: f32) -> u32 {
    let b = val.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b ^ 0x8000_0000
    }
}

#[inline(always)]
pub(super) fn make_desc_key(dot: f32, idx: u32) -> u64 {
    // Sorting networks use `u64::MAX` as a padding sentinel (via `sort_small`). For finite
    // floats, `f32_to_ordered_u32` is never 0, so `desc` is never all-ones and the full key can
    // never be `u64::MAX` (even if `idx == u32::MAX`).
    debug_assert!(dot.is_finite());
    // Bigger dot = smaller key, so ascending sort gives descending dot.
    let ord = f32_to_ordered_u32(dot);
    let desc = !ord;
    ((desc as u64) << 32) | (idx as u64)
}

#[inline(always)]
pub(super) fn sort_keys_u64(keys: &mut [u64]) {
    #[cfg(feature = "packed_knn_sort_small")]
    {
        if keys.len() <= 35 {
            sort_small_u64(keys);
            return;
        }
    }
    keys.sort_unstable();
}

#[inline(always)]
pub(super) fn key_to_idx(key: u64) -> u32 {
    (key & 0xFFFF_FFFF) as u32
}

#[inline(always)]
fn ordered_u32_to_f32(val: u32) -> f32 {
    let b = if val & 0x8000_0000 != 0 {
        val ^ 0x8000_0000
    } else {
        !val
    };
    f32::from_bits(b)
}

#[inline(always)]
pub(super) fn key_to_dot(key: u64) -> f32 {
    let desc = (key >> 32) as u32;
    let ord = !desc;
    ordered_u32_to_f32(ord)
}

#[inline]
fn max_dot_to_cap_xyz(qx: f32, qy: f32, qz: f32, center: Vec3, cos_r: f32, sin_r: f32) -> f32 {
    let cos_d = (qx * center.x + qy * center.y + qz * center.z).clamp(-1.0, 1.0);
    if cos_d > cos_r {
        return 1.0;
    }

    let sin_d = (1.0 - cos_d * cos_d).max(0.0).sqrt();
    (cos_d * cos_r + sin_d * sin_r).clamp(-1.0, 1.0)
}

#[inline]
pub(super) fn security_planes_3x3_interior(cell: usize, grid: &CubeMapGrid) -> Option<[Vec3; 4]> {
    let res = grid.res;
    if res < 3 {
        return None;
    }

    // 3x3 neighborhood stays on a single face iff the center cell is not on the face boundary.
    let (face, iu, iv) = cell_to_face_ij(cell, res);
    if iu < 1 || iv < 1 || iu + 1 >= res || iv + 1 >= res {
        return None;
    }

    // Outer boundaries for the 3x3 envelope: lines at (iu-1, iu+2) and (iv-1, iv+2).
    let mut planes = [
        grid.face_u_line_plane(face, iu - 1),
        grid.face_u_line_plane(face, iu + 2),
        grid.face_v_line_plane(face, iv - 1),
        grid.face_v_line_plane(face, iv + 2),
    ];

    // Orient all planes so that the interior (containing the cell center) has `n·p >= 0`.
    let center = grid.cell_centers[cell];
    for n in &mut planes {
        if n.dot(center) < 0.0 {
            *n = -*n;
        }
    }

    Some(planes)
}

#[inline]
pub(super) fn outside_max_dot_xyz(
    qx: f32,
    qy: f32,
    qz: f32,
    ring2: &[u32],
    grid: &CubeMapGrid,
) -> f32 {
    debug_assert!(!ring2.is_empty(), "ring2 must be non-empty");
    let mut max_dot = f32::NEG_INFINITY;
    for &cell in ring2 {
        let idx = cell as usize;
        let center = grid.cell_centers[idx];
        let cos_r = grid.cell_cos_radius[idx];
        let sin_r = grid.cell_sin_radius[idx];
        let dot = max_dot_to_cap_xyz(qx, qy, qz, center, cos_r, sin_r);
        if dot > max_dot {
            max_dot = dot;
            if max_dot >= 1.0 {
                break;
            }
        }
    }
    max_dot
}
