/// General clipper for large polygons (N > 8) using u64 bitmask.
pub(super) fn clip_bitmask(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    let n = poly.len;
    debug_assert!(n <= 64, "Polygon too large for u64 bitmask");

    let neg_eps = -hp.eps;
    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    let mut dists: [core::mem::MaybeUninit<f64>; MAX_POLY_VERTICES] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let mut mask = 0u64;
    let full_mask: u64 = if n == 64 { !0u64 } else { (1u64 << n) - 1 };

    unsafe {
        let mut bit = 1u64;
        for i in 0..n {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            mask |= (0u64.wrapping_sub((d >= neg_eps) as u64)) & bit;
            bit <<= 1;
        }
    }

    if mask == 0 {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if mask == full_mask {
        return ClipResult::Unchanged;
    }

    let inside_count = mask.count_ones() as usize;
    if inside_count + 2 > MAX_POLY_VERTICES {
        return ClipResult::TooManyVertices;
    }

    let prev_mask = (mask << 1) | (mask >> (n - 1));
    let trans_mask = (mask ^ prev_mask) & ((1u64 << n) - 1);

    debug_assert_eq!(trans_mask.count_ones(), 2);

    let idx1 = trans_mask.trailing_zeros() as usize;
    let trans_mask_rem = trans_mask & !(1 << idx1);
    let idx2 = trans_mask_rem.trailing_zeros() as usize;

    let (entry_next, exit_next) = if (mask & (1 << idx1)) != 0 {
        (idx1, idx2)
    } else {
        (idx2, idx1)
    };

    let calc_transition = |next_idx: usize| -> (usize, usize, f64, f64) {
        let prev_idx = if next_idx == 0 { n - 1 } else { next_idx - 1 };
        unsafe {
            let d0 = dists.get_unchecked(prev_idx).assume_init_read();
            let d1 = dists.get_unchecked(next_idx).assume_init_read();
            (prev_idx, next_idx, d0, d1)
        }
    };

    let (entry_idx, entry_next_idx, entry_d0, entry_d1) = calc_transition(entry_next);
    let (exit_idx, exit_next_idx, exit_d0, exit_d1) = calc_transition(exit_next);

    let t_entry = entry_d0 / (entry_d0 - entry_d1);
    let entry_u = fp::fma_f64(
        t_entry,
        poly.us[entry_next_idx] - poly.us[entry_idx],
        poly.us[entry_idx],
    );
    let entry_v = fp::fma_f64(
        t_entry,
        poly.vs[entry_next_idx] - poly.vs[entry_idx],
        poly.vs[entry_idx],
    );
    let entry_edge_plane = poly.edge_planes[entry_idx];

    let t_exit = exit_d0 / (exit_d0 - exit_d1);
    let exit_u = fp::fma_f64(
        t_exit,
        poly.us[exit_next_idx] - poly.us[exit_idx],
        poly.us[exit_idx],
    );
    let exit_v = fp::fma_f64(
        t_exit,
        poly.vs[exit_next_idx] - poly.vs[exit_idx],
        poly.vs[exit_idx],
    );
    let exit_edge_plane = poly.edge_planes[exit_idx];

    super::output::build_output(
        poly,
        out,
        n,
        (entry_u, entry_v),
        entry_edge_plane,
        entry_next_idx,
        (exit_u, exit_v),
        exit_edge_plane,
        exit_idx,
        hp.plane_idx,
    );

    ClipResult::Changed
}
use super::super::types::{ClipResult, HalfPlane, PolyBuffer, MAX_POLY_VERTICES};
use crate::fp;
