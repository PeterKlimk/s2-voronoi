#[cfg(feature = "timing")]
use super::clip_stats;
use super::types::{ClipResult, HalfPlane, PolyBuffer, MAX_POLY_VERTICES};

/// Clip a convex polygon by a half-plane.
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn clip_convex(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    let n = poly.len;
    #[cfg(feature = "timing")]
    clip_stats::record_call(n);

    debug_assert!(n >= 3, "clip_convex expects poly.len >= 3, got {}", n);

    let has_bounding_ref = poly.has_bounding_ref;
    // Early-unchanged is only relevant for bounded polys. Empirically, it is very rare for small N
    // (e.g. triangles), so we skip the check below a small-N threshold to save overhead.
    const EARLY_UNCHANGED_MIN_N: usize = 5;
    if !has_bounding_ref && n >= EARLY_UNCHANGED_MIN_N {
        let max_r2 = poly.max_r2;
        if max_r2 > 0.0 {
            let t = hp.c + hp.eps;
            if t >= 0.0 && t * t >= hp.ab2 * max_r2 {
                #[cfg(feature = "timing")]
                clip_stats::record_early_unchanged(n, !has_bounding_ref);
                return ClipResult::Unchanged;
            }
        }
    }

    // Dispatch strategy (x86_64):
    // - N=4,8: `% N` is cheap (bitmask), so the plain ptr variant tends to win.
    // - N=3,5,6,7: avoid `% N` in the hot loop via the ptr_d variant.
    // We `match` on N first, then branch on bounded/unbounded per arm. This can improve branch
    // prediction when `has_bounding_ref` correlates with N, while still allowing the inner push
    // path to be specialized via `const TRACK_BOUNDING`.
    match n {
        3 => {
            if has_bounding_ref {
                clip_convex_small_bool_out_idx_ptr_d_impl::<3, true>(poly, hp, out)
            } else {
                clip_convex_small_bool_out_idx_ptr_d_impl::<3, false>(poly, hp, out)
            }
        }
        4 => {
            if has_bounding_ref {
                clip_convex_small_bool_out_idx_ptr_impl::<4, true>(poly, hp, out)
            } else {
                clip_convex_small_bool_out_idx_ptr_impl::<4, false>(poly, hp, out)
            }
        }
        5 => {
            if has_bounding_ref {
                clip_convex_small_bool_out_idx_ptr_d_impl::<5, true>(poly, hp, out)
            } else {
                clip_convex_small_bool_out_idx_ptr_d_impl::<5, false>(poly, hp, out)
            }
        }
        6 => {
            if has_bounding_ref {
                clip_convex_small_bool_out_idx_ptr_d_impl::<6, true>(poly, hp, out)
            } else {
                clip_convex_small_bool_out_idx_ptr_d_impl::<6, false>(poly, hp, out)
            }
        }
        7 => {
            if has_bounding_ref {
                clip_convex_small_bool_out_idx_ptr_d_impl::<7, true>(poly, hp, out)
            } else {
                clip_convex_small_bool_out_idx_ptr_d_impl::<7, false>(poly, hp, out)
            }
        }
        8 => {
            if has_bounding_ref {
                clip_convex_small_bool_out_idx_ptr_impl::<8, true>(poly, hp, out)
            } else {
                clip_convex_small_bool_out_idx_ptr_impl::<8, false>(poly, hp, out)
            }
        }
        _ => clip_convex_bitmask(poly, hp, out),
    }
}

pub(crate) fn clip_convex_small_bool<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            let is_inside = d >= neg_eps;
            inside[i] = is_inside;
            any_inside |= is_inside;
            all_inside &= is_inside;
        }
    }

    if !any_inside {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if all_inside {
        return ClipResult::Unchanged;
    }

    let mut entry_idx = None;
    let mut exit_idx = None;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if entry_idx.is_none() && !inside[prev] && inside[i] {
            entry_idx = Some((prev, i));
        } else if exit_idx.is_none() && inside[prev] && !inside[i] {
            exit_idx = Some((prev, i));
        }
        if entry_idx.is_some() && exit_idx.is_some() {
            break;
        }
    }

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    out.len = 0;

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            out.push_raw(u, v, vp, $ep);
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    let (d_entry, d_entry_next) = unsafe {
        (
            dists.get_unchecked(entry_idx).assume_init_read(),
            dists.get_unchecked(entry_next).assume_init_read(),
        )
    };
    let t_entry = d_entry / (d_entry - d_entry_next);
    let entry_u = t_entry.mul_add(poly.us[entry_next] - poly.us[entry_idx], poly.us[entry_idx]);
    let entry_v = t_entry.mul_add(poly.vs[entry_next] - poly.vs[entry_idx], poly.vs[entry_idx]);
    let entry_ep = poly.edge_planes[entry_idx];
    push!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

    let mut i = entry_next;
    loop {
        push!(
            poly.us[i],
            poly.vs[i],
            poly.vertex_planes[i],
            poly.edge_planes[i]
        );
        if i == exit_idx {
            break;
        }
        i = (i + 1) % N;
    }

    let (d_exit, d_exit_next) = unsafe {
        (
            dists.get_unchecked(exit_idx).assume_init_read(),
            dists.get_unchecked(exit_next).assume_init_read(),
        )
    };
    let t_exit = d_exit / (d_exit - d_exit_next);
    let exit_u = t_exit.mul_add(poly.us[exit_next] - poly.us[exit_idx], poly.us[exit_idx]);
    let exit_v = t_exit.mul_add(poly.vs[exit_next] - poly.vs[exit_idx], poly.vs[exit_idx]);
    let exit_ep = poly.edge_planes[exit_idx];
    push!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Variant of `clip_convex_small_bool` that writes output vertices using a local index and
/// unchecked stores (sets `out.len` once at the end).
#[cfg(any(test, feature = "microbench"))]
pub(crate) fn clip_convex_small_bool_out_idx<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            let is_inside = d >= neg_eps;
            inside[i] = is_inside;
            any_inside |= is_inside;
            all_inside &= is_inside;
        }
    }

    if !any_inside {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if all_inside {
        return ClipResult::Unchanged;
    }

    let mut entry_idx = None;
    let mut exit_idx = None;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if entry_idx.is_none() && !inside[prev] && inside[i] {
            entry_idx = Some((prev, i));
        } else if exit_idx.is_none() && inside[prev] && !inside[i] {
            exit_idx = Some((prev, i));
        }
        if entry_idx.is_some() && exit_idx.is_some() {
            break;
        }
    }

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            unsafe {
                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
            }
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    let (d_entry, d_entry_next) = unsafe {
        (
            dists.get_unchecked(entry_idx).assume_init_read(),
            dists.get_unchecked(entry_next).assume_init_read(),
        )
    };
    let t_entry = d_entry / (d_entry - d_entry_next);
    let entry_u = t_entry.mul_add(poly.us[entry_next] - poly.us[entry_idx], poly.us[entry_idx]);
    let entry_v = t_entry.mul_add(poly.vs[entry_next] - poly.vs[entry_idx], poly.vs[entry_idx]);
    let entry_ep = poly.edge_planes[entry_idx];
    push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

    let mut i = entry_next;
    loop {
        push_idx!(
            poly.us[i],
            poly.vs[i],
            poly.vertex_planes[i],
            poly.edge_planes[i]
        );
        if i == exit_idx {
            break;
        }
        i = (i + 1) % N;
    }

    let (d_exit, d_exit_next) = unsafe {
        (
            dists.get_unchecked(exit_idx).assume_init_read(),
            dists.get_unchecked(exit_next).assume_init_read(),
        )
    };
    let t_exit = d_exit / (d_exit - d_exit_next);
    let exit_u = t_exit.mul_add(poly.us[exit_next] - poly.us[exit_idx], poly.us[exit_idx]);
    let exit_v = t_exit.mul_add(poly.vs[exit_next] - poly.vs[exit_idx], poly.vs[exit_idx]);
    let exit_ep = poly.edge_planes[exit_idx];
    push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Like `clip_convex_small_bool_out_idx`, but uses a modulo-free vertex copy (two linear ranges).
#[cfg(any(test, feature = "microbench"))]
pub(crate) fn clip_convex_small_bool_out_idx_range<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            let is_inside = d >= neg_eps;
            inside[i] = is_inside;
            any_inside |= is_inside;
            all_inside &= is_inside;
        }
    }

    if !any_inside {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if all_inside {
        return ClipResult::Unchanged;
    }

    let mut entry_idx = None;
    let mut exit_idx = None;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if entry_idx.is_none() && !inside[prev] && inside[i] {
            entry_idx = Some((prev, i));
        } else if exit_idx.is_none() && inside[prev] && !inside[i] {
            exit_idx = Some((prev, i));
        }
        if entry_idx.is_some() && exit_idx.is_some() {
            break;
        }
    }

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            unsafe {
                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
            }
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    let (d_entry, d_entry_next) = unsafe {
        (
            dists.get_unchecked(entry_idx).assume_init_read(),
            dists.get_unchecked(entry_next).assume_init_read(),
        )
    };
    let t_entry = d_entry / (d_entry - d_entry_next);
    let entry_u = t_entry.mul_add(poly.us[entry_next] - poly.us[entry_idx], poly.us[entry_idx]);
    let entry_v = t_entry.mul_add(poly.vs[entry_next] - poly.vs[entry_idx], poly.vs[entry_idx]);
    let entry_ep = poly.edge_planes[entry_idx];
    push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

    if entry_next <= exit_idx {
        for i in entry_next..=exit_idx {
            push_idx!(
                poly.us[i],
                poly.vs[i],
                poly.vertex_planes[i],
                poly.edge_planes[i]
            );
        }
    } else {
        for i in entry_next..N {
            push_idx!(
                poly.us[i],
                poly.vs[i],
                poly.vertex_planes[i],
                poly.edge_planes[i]
            );
        }
        for i in 0..=exit_idx {
            push_idx!(
                poly.us[i],
                poly.vs[i],
                poly.vertex_planes[i],
                poly.edge_planes[i]
            );
        }
    }

    let (d_exit, d_exit_next) = unsafe {
        (
            dists.get_unchecked(exit_idx).assume_init_read(),
            dists.get_unchecked(exit_next).assume_init_read(),
        )
    };
    let t_exit = d_exit / (d_exit - d_exit_next);
    let exit_u = t_exit.mul_add(poly.us[exit_next] - poly.us[exit_idx], poly.us[exit_idx]);
    let exit_v = t_exit.mul_add(poly.vs[exit_next] - poly.vs[exit_idx], poly.vs[exit_idx]);
    let exit_ep = poly.edge_planes[exit_idx];
    push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Combined variant: `out_idx_range` vertex copying + raw-pointer input reads + interleaved divisions.
///
/// This tends to win for larger small-N (e.g. N=7..8) because it avoids per-vertex wrap/exit branches
/// in the mixed-case copy loop while keeping the interpolation path tight.
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn clip_convex_small_bool_out_idx_ptr_range_d<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();
    let vps = poly.vertex_planes.as_ptr();
    let eps = poly.edge_planes.as_ptr();

    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            let is_inside = d >= neg_eps;
            inside[i] = is_inside;
            any_inside |= is_inside;
            all_inside &= is_inside;
        }
    }

    if !any_inside {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if all_inside {
        return ClipResult::Unchanged;
    }

    let mut entry_idx = None;
    let mut exit_idx = None;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if entry_idx.is_none() && !inside[prev] && inside[i] {
            entry_idx = Some((prev, i));
        } else if exit_idx.is_none() && inside[prev] && !inside[i] {
            exit_idx = Some((prev, i));
        }
        if entry_idx.is_some() && exit_idx.is_some() {
            break;
        }
    }

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            unsafe {
                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
            }
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    unsafe {
        // Read all four distances first, then issue both divisions for ILP.
        let d_entry = dists.get_unchecked(entry_idx).assume_init_read();
        let d_entry_next = dists.get_unchecked(entry_next).assume_init_read();
        let d_exit = dists.get_unchecked(exit_idx).assume_init_read();
        let d_exit_next = dists.get_unchecked(exit_next).assume_init_read();

        let t_entry = d_entry / (d_entry - d_entry_next);
        let t_exit = d_exit / (d_exit - d_exit_next);

        let entry_u = t_entry.mul_add(*us.add(entry_next) - *us.add(entry_idx), *us.add(entry_idx));
        let entry_v = t_entry.mul_add(*vs.add(entry_next) - *vs.add(entry_idx), *vs.add(entry_idx));
        let entry_ep = *eps.add(entry_idx);
        push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        if entry_next <= exit_idx {
            for i in entry_next..=exit_idx {
                push_idx!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            }
        } else {
            for i in entry_next..N {
                push_idx!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            }
            for i in 0..=exit_idx {
                push_idx!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            }
        }

        let exit_u = t_exit.mul_add(*us.add(exit_next) - *us.add(exit_idx), *us.add(exit_idx));
        let exit_v = t_exit.mul_add(*vs.add(exit_next) - *vs.add(exit_idx), *vs.add(exit_idx));
        let exit_ep = *eps.add(exit_idx);
        push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);
    }

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Like `clip_convex_small_bool_out_idx`, but uses raw pointers for all input reads in the mixed
/// case (including intersection interpolation reads) to eliminate bounds checks on reads.
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn clip_convex_small_bool_out_idx_ptr<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    let vps = poly.vertex_planes.as_ptr();
    let eps = poly.edge_planes.as_ptr();

    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            let is_inside = d >= neg_eps;
            inside[i] = is_inside;
            any_inside |= is_inside;
            all_inside &= is_inside;
        }
    }

    if !any_inside {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if all_inside {
        return ClipResult::Unchanged;
    }

    let mut entry_idx = None;
    let mut exit_idx = None;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if entry_idx.is_none() && !inside[prev] && inside[i] {
            entry_idx = Some((prev, i));
        } else if exit_idx.is_none() && inside[prev] && !inside[i] {
            exit_idx = Some((prev, i));
        }
        if entry_idx.is_some() && exit_idx.is_some() {
            break;
        }
    }

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            unsafe {
                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
            }
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    unsafe {
        let d_entry = dists.get_unchecked(entry_idx).assume_init_read();
        let d_entry_next = dists.get_unchecked(entry_next).assume_init_read();
        let t_entry = d_entry / (d_entry - d_entry_next);
        let entry_u = t_entry.mul_add(*us.add(entry_next) - *us.add(entry_idx), *us.add(entry_idx));
        let entry_v = t_entry.mul_add(*vs.add(entry_next) - *vs.add(entry_idx), *vs.add(entry_idx));
        let entry_ep = *eps.add(entry_idx);
        push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        let mut i = entry_next;
        loop {
            push_idx!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            if i == exit_idx {
                break;
            }
            i = (i + 1) % N;
        }

        let d_exit = dists.get_unchecked(exit_idx).assume_init_read();
        let d_exit_next = dists.get_unchecked(exit_next).assume_init_read();
        let t_exit = d_exit / (d_exit - d_exit_next);
        let exit_u = t_exit.mul_add(*us.add(exit_next) - *us.add(exit_idx), *us.add(exit_idx));
        let exit_v = t_exit.mul_add(*vs.add(exit_next) - *vs.add(exit_idx), *vs.add(exit_idx));
        let exit_ep = *eps.add(exit_idx);
        push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);
    }

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Like `clip_convex_small_bool_out_idx_ptr`, but splits bounded/unbounded at entry and uses a
/// const generic to remove the per-vertex `track_bounding` branch in the hot `push` path.
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn clip_convex_small_bool_out_idx_ptr_split<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    if poly.has_bounding_ref {
        clip_convex_small_bool_out_idx_ptr_impl::<N, true>(poly, hp, out)
    } else {
        clip_convex_small_bool_out_idx_ptr_impl::<N, false>(poly, hp, out)
    }
}

#[cfg_attr(feature = "profiling", inline(never))]
fn clip_convex_small_bool_out_idx_ptr_impl<const N: usize, const TRACK_BOUNDING: bool>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();
    let vps = poly.vertex_planes.as_ptr();
    let eps = poly.edge_planes.as_ptr();

    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            let is_inside = d >= neg_eps;
            inside[i] = is_inside;
            any_inside |= is_inside;
            all_inside &= is_inside;
        }
    }

    if !any_inside {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if all_inside {
        return ClipResult::Unchanged;
    }

    let mut entry_idx = None;
    let mut exit_idx = None;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if entry_idx.is_none() && !inside[prev] && inside[i] {
            entry_idx = Some((prev, i));
        } else if exit_idx.is_none() && inside[prev] && !inside[i] {
            exit_idx = Some((prev, i));
        }
        if entry_idx.is_some() && exit_idx.is_some() {
            break;
        }
    }

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            unsafe {
                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
            }
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if TRACK_BOUNDING {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    unsafe {
        let d_entry = dists.get_unchecked(entry_idx).assume_init_read();
        let d_entry_next = dists.get_unchecked(entry_next).assume_init_read();
        let t_entry = d_entry / (d_entry - d_entry_next);
        let entry_u = t_entry.mul_add(*us.add(entry_next) - *us.add(entry_idx), *us.add(entry_idx));
        let entry_v = t_entry.mul_add(*vs.add(entry_next) - *vs.add(entry_idx), *vs.add(entry_idx));
        let entry_ep = *eps.add(entry_idx);
        push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        let mut i = entry_next;
        loop {
            push_idx!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            if i == exit_idx {
                break;
            }
            i = (i + 1) % N;
        }

        let d_exit = dists.get_unchecked(exit_idx).assume_init_read();
        let d_exit_next = dists.get_unchecked(exit_next).assume_init_read();
        let t_exit = d_exit / (d_exit - d_exit_next);
        let exit_u = t_exit.mul_add(*us.add(exit_next) - *us.add(exit_idx), *us.add(exit_idx));
        let exit_v = t_exit.mul_add(*vs.add(exit_next) - *vs.add(exit_idx), *vs.add(exit_idx));
        let exit_ep = *eps.add(exit_idx);
        push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);
    }

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if TRACK_BOUNDING { has_bounding } else { false };
    ClipResult::Changed
}

/// Like `clip_convex_small_bool_out_idx_ptr`, but issues both divisions before using results
/// for better instruction-level parallelism.
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn clip_convex_small_bool_out_idx_ptr_b<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    let vps = poly.vertex_planes.as_ptr();
    let eps = poly.edge_planes.as_ptr();

    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            let is_inside = d >= neg_eps;
            inside[i] = is_inside;
            any_inside |= is_inside;
            all_inside &= is_inside;
        }
    }

    if !any_inside {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if all_inside {
        return ClipResult::Unchanged;
    }

    let mut entry_idx = None;
    let mut exit_idx = None;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if entry_idx.is_none() && !inside[prev] && inside[i] {
            entry_idx = Some((prev, i));
        } else if exit_idx.is_none() && inside[prev] && !inside[i] {
            exit_idx = Some((prev, i));
        }
        if entry_idx.is_some() && exit_idx.is_some() {
            break;
        }
    }

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            unsafe {
                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
            }
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    unsafe {
        // Read all four distances first
        let d_entry = dists.get_unchecked(entry_idx).assume_init_read();
        let d_entry_next = dists.get_unchecked(entry_next).assume_init_read();
        let d_exit = dists.get_unchecked(exit_idx).assume_init_read();
        let d_exit_next = dists.get_unchecked(exit_next).assume_init_read();

        // Issue both divisions for ILP (division has high latency)
        let t_entry = d_entry / (d_entry - d_entry_next);
        let t_exit = d_exit / (d_exit - d_exit_next);

        // Now compute entry intersection using t_entry
        let entry_u = t_entry.mul_add(*us.add(entry_next) - *us.add(entry_idx), *us.add(entry_idx));
        let entry_v = t_entry.mul_add(*vs.add(entry_next) - *vs.add(entry_idx), *vs.add(entry_idx));
        let entry_ep = *eps.add(entry_idx);
        push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        let mut i = entry_next;
        loop {
            push_idx!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            if i == exit_idx {
                break;
            }
            i = (i + 1) % N;
        }

        // Compute exit intersection using t_exit
        let exit_u = t_exit.mul_add(*us.add(exit_next) - *us.add(exit_idx), *us.add(exit_idx));
        let exit_v = t_exit.mul_add(*vs.add(exit_next) - *vs.add(exit_idx), *vs.add(exit_idx));
        let exit_ep = *eps.add(exit_idx);
        push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);
    }

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Like `clip_convex_small_bool_out_idx_ptr`, but uses explicit branch instead of modulo
/// for vertex iteration: `if i == N { i = 0; }` instead of `i = (i + 1) % N`.
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn clip_convex_small_bool_out_idx_ptr_c<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    let vps = poly.vertex_planes.as_ptr();
    let eps = poly.edge_planes.as_ptr();

    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            let is_inside = d >= neg_eps;
            inside[i] = is_inside;
            any_inside |= is_inside;
            all_inside &= is_inside;
        }
    }

    if !any_inside {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if all_inside {
        return ClipResult::Unchanged;
    }

    let mut entry_idx = None;
    let mut exit_idx = None;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if entry_idx.is_none() && !inside[prev] && inside[i] {
            entry_idx = Some((prev, i));
        } else if exit_idx.is_none() && inside[prev] && !inside[i] {
            exit_idx = Some((prev, i));
        }
        if entry_idx.is_some() && exit_idx.is_some() {
            break;
        }
    }

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            unsafe {
                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
            }
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    unsafe {
        let d_entry = dists.get_unchecked(entry_idx).assume_init_read();
        let d_entry_next = dists.get_unchecked(entry_next).assume_init_read();
        let t_entry = d_entry / (d_entry - d_entry_next);
        let entry_u = t_entry.mul_add(*us.add(entry_next) - *us.add(entry_idx), *us.add(entry_idx));
        let entry_v = t_entry.mul_add(*vs.add(entry_next) - *vs.add(entry_idx), *vs.add(entry_idx));
        let entry_ep = *eps.add(entry_idx);
        push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        let mut i = entry_next;
        loop {
            push_idx!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            if i == exit_idx {
                break;
            }
            i += 1;
            if i == N {
                i = 0;
            }
        }

        let d_exit = dists.get_unchecked(exit_idx).assume_init_read();
        let d_exit_next = dists.get_unchecked(exit_next).assume_init_read();
        let t_exit = d_exit / (d_exit - d_exit_next);
        let exit_u = t_exit.mul_add(*us.add(exit_next) - *us.add(exit_idx), *us.add(exit_idx));
        let exit_v = t_exit.mul_add(*vs.add(exit_next) - *vs.add(exit_idx), *vs.add(exit_idx));
        let exit_ep = *eps.add(exit_idx);
        push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);
    }

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Hybrid variant: bitmask transition finding (from clip_convex_small) with raw pointer
/// reads and out_idx-style output writing.
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn clip_convex_small_bool_out_idx_ptr_mask<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();
    let vps = poly.vertex_planes.as_ptr();
    let eps = poly.edge_planes.as_ptr();

    let mut inside_mask: u8 = 0;
    let all_mask = (1u8 << N) - 1;

    // 1. Calculate Distances (Keep this vectorized!)
    // We avoid index-dependent operations inside the loop to ensure the compiler
    // emits AVX2 instructions (vfmadd) like it did in the original profile.
    unsafe {
        // Note: We rely on hp.signed_dist being inlinable.
        // If it's not, we would need to inline the math manually here.
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);

            // Check condition. We separate the mask construction slightly to ensure
            // the compiler doesn't de-optimize the math loop.
            // However, since N is small, a simple scalar branch here is acceptable
            // as long as the *math* itself was vectorized.
            if d >= neg_eps {
                inside_mask |= 1 << i;
            }
        }
    }

    // 2. Fast Path Checks using Bitmask
    if inside_mask == 0 {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if inside_mask == all_mask {
        return ClipResult::Unchanged;
    }

    // 3. Find Entry and Exit using Bitwise Logic (Eliminates the branchy search loop)
    // Rotate mask right by 1 to align previous vertices with current indices
    let rotated_mask = ((inside_mask << 1) | (inside_mask >> (N - 1))) & all_mask;

    // Entry: Outside (0) -> Inside (1). Pattern: !rotated & inside_mask
    let entry_mask = !rotated_mask & inside_mask;

    // Exit: Inside (1) -> Outside (0). Pattern: rotated & !inside_mask
    let exit_mask = rotated_mask & !inside_mask;

    // Find indices using trailing zeros (bsf/tzcnt instruction)
    let entry_next = entry_mask.trailing_zeros() as usize;
    let exit_idx = exit_mask.trailing_zeros() as usize;

    let entry_idx = if entry_next == 0 {
        N - 1
    } else {
        entry_next - 1
    };
    let exit_next = (exit_idx + 1) % N;

    // 4. Interpolation and Output (Same as original)
    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            unsafe {
                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
            }
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    unsafe {
        let d_entry = dists.get_unchecked(entry_idx).assume_init_read();
        let d_entry_next = dists.get_unchecked(entry_next).assume_init_read();
        let t_entry = d_entry / (d_entry - d_entry_next);
        let entry_u = t_entry.mul_add(*us.add(entry_next) - *us.add(entry_idx), *us.add(entry_idx));
        let entry_v = t_entry.mul_add(*vs.add(entry_next) - *vs.add(entry_idx), *vs.add(entry_idx));
        let entry_ep = *eps.add(entry_idx);
        push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        let mut i = entry_next;
        loop {
            push_idx!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            if i == exit_idx {
                break;
            }
            i = (i + 1) % N;
        }

        let d_exit = dists.get_unchecked(exit_idx).assume_init_read();
        let d_exit_next = dists.get_unchecked(exit_next).assume_init_read();
        let t_exit = d_exit / (d_exit - d_exit_next);
        let exit_u = t_exit.mul_add(*us.add(exit_next) - *us.add(exit_idx), *us.add(exit_idx));
        let exit_v = t_exit.mul_add(*vs.add(exit_next) - *vs.add(exit_idx), *vs.add(exit_idx));
        let exit_ep = *eps.add(exit_idx);
        push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);
    }

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Combined variant: out_idx_ptr + interleaved divisions + branch instead of modulo.
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn clip_convex_small_bool_out_idx_ptr_d<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    let vps = poly.vertex_planes.as_ptr();
    let eps = poly.edge_planes.as_ptr();

    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            let is_inside = d >= neg_eps;
            inside[i] = is_inside;
            any_inside |= is_inside;
            all_inside &= is_inside;
        }
    }

    if !any_inside {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if all_inside {
        return ClipResult::Unchanged;
    }

    let mut entry_idx = None;
    let mut exit_idx = None;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if entry_idx.is_none() && !inside[prev] && inside[i] {
            entry_idx = Some((prev, i));
        } else if exit_idx.is_none() && inside[prev] && !inside[i] {
            exit_idx = Some((prev, i));
        }
        if entry_idx.is_some() && exit_idx.is_some() {
            break;
        }
    }

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            unsafe {
                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
            }
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    unsafe {
        // Read all four distances first, then issue both divisions for ILP
        let d_entry = dists.get_unchecked(entry_idx).assume_init_read();
        let d_entry_next = dists.get_unchecked(entry_next).assume_init_read();
        let d_exit = dists.get_unchecked(exit_idx).assume_init_read();
        let d_exit_next = dists.get_unchecked(exit_next).assume_init_read();

        let t_entry = d_entry / (d_entry - d_entry_next);
        let t_exit = d_exit / (d_exit - d_exit_next);

        let entry_u = t_entry.mul_add(*us.add(entry_next) - *us.add(entry_idx), *us.add(entry_idx));
        let entry_v = t_entry.mul_add(*vs.add(entry_next) - *vs.add(entry_idx), *vs.add(entry_idx));
        let entry_ep = *eps.add(entry_idx);
        push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        let mut i = entry_next;
        loop {
            push_idx!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            if i == exit_idx {
                break;
            }
            i += 1;
            if i == N {
                i = 0;
            }
        }

        let exit_u = t_exit.mul_add(*us.add(exit_next) - *us.add(exit_idx), *us.add(exit_idx));
        let exit_v = t_exit.mul_add(*vs.add(exit_next) - *vs.add(exit_idx), *vs.add(exit_idx));
        let exit_ep = *eps.add(exit_idx);
        push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);
    }

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Like `clip_convex_small_bool_out_idx_ptr_d`, but splits bounded/unbounded at the entry and
/// uses a const generic to remove the per-vertex `track_bounding` branch in the hot `push` path.
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn clip_convex_small_bool_out_idx_ptr_d_split<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    if poly.has_bounding_ref {
        clip_convex_small_bool_out_idx_ptr_d_impl::<N, true>(poly, hp, out)
    } else {
        clip_convex_small_bool_out_idx_ptr_d_impl::<N, false>(poly, hp, out)
    }
}

#[cfg_attr(feature = "profiling", inline(never))]
fn clip_convex_small_bool_out_idx_ptr_d_impl<const N: usize, const TRACK_BOUNDING: bool>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();
    let vps = poly.vertex_planes.as_ptr();
    let eps = poly.edge_planes.as_ptr();

    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            let is_inside = d >= neg_eps;
            inside[i] = is_inside;
            any_inside |= is_inside;
            all_inside &= is_inside;
        }
    }

    if !any_inside {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if all_inside {
        return ClipResult::Unchanged;
    }

    let mut entry_idx = None;
    let mut exit_idx = None;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if entry_idx.is_none() && !inside[prev] && inside[i] {
            entry_idx = Some((prev, i));
        } else if exit_idx.is_none() && inside[prev] && !inside[i] {
            exit_idx = Some((prev, i));
        }
        if entry_idx.is_some() && exit_idx.is_some() {
            break;
        }
    }

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            unsafe {
                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
            }
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if TRACK_BOUNDING {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    unsafe {
        // Read all four distances first, then issue both divisions for ILP
        let d_entry = dists.get_unchecked(entry_idx).assume_init_read();
        let d_entry_next = dists.get_unchecked(entry_next).assume_init_read();
        let d_exit = dists.get_unchecked(exit_idx).assume_init_read();
        let d_exit_next = dists.get_unchecked(exit_next).assume_init_read();

        let t_entry = d_entry / (d_entry - d_entry_next);
        let t_exit = d_exit / (d_exit - d_exit_next);

        let entry_u = t_entry.mul_add(*us.add(entry_next) - *us.add(entry_idx), *us.add(entry_idx));
        let entry_v = t_entry.mul_add(*vs.add(entry_next) - *vs.add(entry_idx), *vs.add(entry_idx));
        let entry_ep = *eps.add(entry_idx);
        push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        let mut i = entry_next;
        loop {
            push_idx!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            if i == exit_idx {
                break;
            }
            i += 1;
            if i == N {
                i = 0;
            }
        }

        let exit_u = t_exit.mul_add(*us.add(exit_next) - *us.add(exit_idx), *us.add(exit_idx));
        let exit_v = t_exit.mul_add(*vs.add(exit_next) - *vs.add(exit_idx), *vs.add(exit_idx));
        let exit_ep = *eps.add(exit_idx);
        push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);
    }

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if TRACK_BOUNDING { has_bounding } else { false };
    ClipResult::Changed
}

pub(super) fn clip_convex_bitmask(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
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
    let entry_u = t_entry.mul_add(
        poly.us[entry_next_idx] - poly.us[entry_idx],
        poly.us[entry_idx],
    );
    let entry_v = t_entry.mul_add(
        poly.vs[entry_next_idx] - poly.vs[entry_idx],
        poly.vs[entry_idx],
    );
    let entry_edge_plane = poly.edge_planes[entry_idx];

    let t_exit = exit_d0 / (exit_d0 - exit_d1);
    let exit_u = t_exit.mul_add(
        poly.us[exit_next_idx] - poly.us[exit_idx],
        poly.us[exit_idx],
    );
    let exit_v = t_exit.mul_add(
        poly.vs[exit_next_idx] - poly.vs[exit_idx],
        poly.vs[exit_idx],
    );
    let exit_edge_plane = poly.edge_planes[exit_idx];

    build_output(
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

fn build_output(
    poly: &PolyBuffer,
    out: &mut PolyBuffer,
    n: usize,
    entry_pt: (f64, f64),
    entry_edge_plane: usize,
    entry_next: usize,
    exit_pt: (f64, f64),
    exit_edge_plane: usize,
    exit_idx: usize,
    hp_plane_idx: usize,
) {
    out.len = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            out.push_raw(u, v, vp, $ep);
            max_r2 = max_r2.max(u.mul_add(u, v * v));
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    push!(
        entry_pt.0,
        entry_pt.1,
        (entry_edge_plane, hp_plane_idx),
        entry_edge_plane
    );

    let mut i = entry_next;
    loop {
        push!(
            poly.us[i],
            poly.vs[i],
            poly.vertex_planes[i],
            poly.edge_planes[i]
        );
        if i == exit_idx {
            break;
        }
        i = (i + 1) % n;
    }

    push!(
        exit_pt.0,
        exit_pt.1,
        (exit_edge_plane, hp_plane_idx),
        hp_plane_idx
    );

    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
}
