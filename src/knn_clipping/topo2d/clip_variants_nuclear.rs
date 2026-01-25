#[cfg(all(target_arch = "x86_64", any(test, feature = "microbench")))]
pub mod clip_convex_nuclear_n4 {
    use crate::knn_clipping::topo2d::clippers::clip_convex_small_bool;
    use crate::knn_clipping::topo2d::types::{ClipResult, HalfPlane, PolyBuffer};

    pub fn clip_convex_nuclear_n4(
        poly: &PolyBuffer,
        hp: &HalfPlane,
        out: &mut PolyBuffer,
    ) -> ClipResult {
        #[cfg(target_feature = "avx2")]
        unsafe {
            clip_convex_nuclear_n4_avx2(poly, hp, out)
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            clip_convex_small_bool::<4>(poly, hp, out)
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(never)]
    unsafe fn clip_convex_nuclear_n4_avx2(
        poly: &PolyBuffer,
        hp: &HalfPlane,
        out: &mut PolyBuffer,
    ) -> ClipResult {
        use std::arch::x86_64::*;

        debug_assert_eq!(poly.len, 4);

        let us_ptr = poly.us.as_ptr();
        let vs_ptr = poly.vs.as_ptr();

        // Load Us, Vs
        let us = _mm256_loadu_pd(us_ptr);
        let vs = _mm256_loadu_pd(vs_ptr);

        let a = _mm256_set1_pd(hp.a);
        let b = _mm256_set1_pd(hp.b);
        let c = _mm256_set1_pd(hp.c);

        // dists = a*u + b*v + c
        let dists = _mm256_fmadd_pd(a, us, _mm256_fmadd_pd(b, vs, c));
        let neg_eps = _mm256_set1_pd(-hp.eps);

        // Mask
        let mask_pd = _mm256_cmp_pd(dists, neg_eps, _CMP_GE_OQ);
        let mask = _mm256_movemask_pd(mask_pd) as usize;

        if mask == 0 {
            out.len = 0;
            out.max_r2 = 0.0;
            out.has_bounding_ref = false;
            return ClipResult::Changed;
        }
        if mask == 15 {
            return ClipResult::Unchanged;
        }

        // Prepare dists for scalar usage
        // We only use 2, but storing 4 is simpler/safer
        let mut dist_arr = [0.0; 4];
        _mm256_storeu_pd(dist_arr.as_mut_ptr(), dists);

        // Lookup configuration
        // (entry_edge, exit_edge, start_vert, len)
        // Entry edge i means intersection on edge i->(i+1)%4
        // Exit edge i means intersection on edge i->(i+1)%4
        // We use 255 (0xFF) to indicate invalid (should not happen for valid mask)
        const T: [(u8, u8, u8, u8); 16] = [
            (0, 0, 0, 0),    // 0: empty (handled)
            (3, 0, 0, 1),    // 1: In 0. Entry 3->0, Exit 0->1. Copy 0.
            (0, 1, 1, 1),    // 2: In 1. Entry 0->1, Exit 1->2. Copy 1.
            (3, 1, 0, 2),    // 3: In 0,1. Entry 3->0, Exit 1->2. Copy 0,1.
            (1, 2, 2, 1),    // 4: In 2. Entry 1->2, Exit 2->3. Copy 2.
            (0xFF, 0, 0, 0), // 5: In 0,2 (Impossible)
            (0, 2, 1, 2),    // 6: In 1,2. Entry 0->1, Exit 2->3. Copy 1,2.
            (3, 2, 0, 3),    // 7: In 0,1,2. Entry 3->0, Exit 2->3. Copy 0,1,2.
            (2, 3, 3, 1),    // 8: In 3.
            (2, 0, 3, 2),    // 9: In 3,0. Wraps. Order 3,0. Entry 2->3, Exit 0->1.
            (0xFF, 0, 0, 0), // 10: In 1,3 (Impossible)
            (2, 1, 3, 3),    // 11: In 0,1,3. Wraps. Order 3,0,1. Entry 2->3, Exit 1->2.
            (1, 3, 2, 2),    // 12: In 2,3.
            (1, 0, 2, 3),    // 13: In 0,2,3. Wraps 2,3,0. Entry 1->2, Exit 0->1.
            (0, 3, 1, 3),    // 14: In 1,2,3. Order 1,2,3. Entry 0->1, Exit 3->0.
            (0, 0, 0, 0),    // 15: all in (handled)
        ];

        let (entry_edge, exit_edge, start_vert, len) = *T.get_unchecked(mask);
        debug_assert_ne!(entry_edge, 0xFF);

        let mut out_len = 0;
        let mut max_r2 = 0.0;
        let mut has_bounding = false;
        let track_bounding = poly.has_bounding_ref;

        // --- Entry Intersection ---
        {
            let idx = entry_edge as usize;
            let next = (idx + 1) & 3;

            let d1 = *dist_arr.get_unchecked(idx);
            let d2 = *dist_arr.get_unchecked(next);
            let t = d1 / (d1 - d2);

            let u1 = *us_ptr.add(idx);
            let v1 = *vs_ptr.add(idx);
            let u2 = *us_ptr.add(next);
            let v2 = *vs_ptr.add(next);

            let u = u1 + t * (u2 - u1);
            let v = v1 + t * (v2 - v1);

            let ep = *poly.edge_planes.get_unchecked(idx);

            *out.us.get_unchecked_mut(out_len) = u;
            *out.vs.get_unchecked_mut(out_len) = v;
            *out.vertex_planes.get_unchecked_mut(out_len) = (ep, hp.plane_idx);
            *out.edge_planes.get_unchecked_mut(out_len) = ep;
            out_len += 1;

            let r2 = u * u + v * v;
            if r2 > max_r2 {
                max_r2 = r2;
            }
        }

        // --- Inside Vertices ---
        {
            let mut v_idx = start_vert as usize;
            for _ in 0..len {
                let idx = v_idx & 3; // handle wrap if len > 1
                v_idx += 1;

                let u = *us_ptr.add(idx);
                let v = *vs_ptr.add(idx);
                let vp = *poly.vertex_planes.get_unchecked(idx);
                let ep = *poly.edge_planes.get_unchecked(idx);

                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
                out_len += 1;

                let r2 = u * u + v * v;
                if r2 > max_r2 {
                    max_r2 = r2;
                }
                if track_bounding {
                    has_bounding |= vp.0 == usize::MAX;
                }
            }
        }

        // --- Exit Intersection ---
        {
            let idx = exit_edge as usize;
            let next = (idx + 1) & 3;

            let d1 = *dist_arr.get_unchecked(idx);
            let d2 = *dist_arr.get_unchecked(next);
            let t = d1 / (d1 - d2);

            let u1 = *us_ptr.add(idx);
            let v1 = *vs_ptr.add(idx);
            let u2 = *us_ptr.add(next);
            let v2 = *vs_ptr.add(next);

            let u = u1 + t * (u2 - u1);
            let v = v1 + t * (v2 - v1);

            let ep = *poly.edge_planes.get_unchecked(idx);

            *out.us.get_unchecked_mut(out_len) = u;
            *out.vs.get_unchecked_mut(out_len) = v;
            *out.vertex_planes.get_unchecked_mut(out_len) = (ep, hp.plane_idx);
            *out.edge_planes.get_unchecked_mut(out_len) = hp.plane_idx;
            out_len += 1;

            let r2 = u * u + v * v;
            if r2 > max_r2 {
                max_r2 = r2;
            }
        }

        out.len = out_len;
        out.max_r2 = max_r2;
        out.has_bounding_ref = if track_bounding { has_bounding } else { false };
        ClipResult::Changed
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(never)]
    pub unsafe fn clip_convex_nuclear_n4_v5(
        poly: &PolyBuffer,
        hp: &HalfPlane,
        out: &mut PolyBuffer,
    ) -> ClipResult {
        use std::arch::x86_64::*;
        debug_assert_eq!(poly.len, 4);

        let us_ptr = poly.us.as_ptr();
        let vs_ptr = poly.vs.as_ptr();

        let us = _mm256_loadu_pd(us_ptr);
        let vs = _mm256_loadu_pd(vs_ptr);

        let a = _mm256_set1_pd(hp.a);
        let b = _mm256_set1_pd(hp.b);
        let c = _mm256_set1_pd(hp.c);

        // dists = a*u + b*v + c
        let dists = _mm256_fmadd_pd(a, us, _mm256_fmadd_pd(b, vs, c));
        let neg_eps = _mm256_set1_pd(-hp.eps);

        let mask_pd = _mm256_cmp_pd(dists, neg_eps, _CMP_GE_OQ);
        let mask = _mm256_movemask_pd(mask_pd) as u8;

        if mask == 0 {
            out.len = 0;
            out.max_r2 = 0.0;
            out.has_bounding_ref = false;
            return ClipResult::Changed;
        }
        if mask == 0b1111 {
            return ClipResult::Unchanged;
        }

        let dists_next = _mm256_permute4x64_pd::<0b00_11_10_01>(dists);
        let diff = _mm256_sub_pd(dists, dists_next);
        let t_vec = _mm256_div_pd(dists, diff);

        let us_next = _mm256_permute4x64_pd::<0b00_11_10_01>(us);
        let us_diff = _mm256_sub_pd(us_next, us);
        let interp_us = _mm256_fmadd_pd(t_vec, us_diff, us);

        let vs_next = _mm256_permute4x64_pd::<0b00_11_10_01>(vs);
        let vs_diff = _mm256_sub_pd(vs_next, vs);
        let interp_vs = _mm256_fmadd_pd(t_vec, vs_diff, vs);

        // Store intersection points, we'll selectively load them later or shuffle them
        let mut i_us = [0.0; 4];
        let mut i_vs = [0.0; 4];
        _mm256_storeu_pd(i_us.as_mut_ptr(), interp_us);
        _mm256_storeu_pd(i_vs.as_mut_ptr(), interp_vs);

        let mut out_len = 0;
        let mut max_r2 = 0.0;
        let mut has_bounding = false;
        let track_bounding = poly.has_bounding_ref;

        macro_rules! push_raw {
            ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
                *out.us.get_unchecked_mut(out_len) = $u;
                *out.vs.get_unchecked_mut(out_len) = $v;
                *out.vertex_planes.get_unchecked_mut(out_len) = $vp;
                *out.edge_planes.get_unchecked_mut(out_len) = $ep;
                out_len += 1;
                let r2 = $u * $u + $v * $v;
                if r2 > out.max_r2 {
                    out.max_r2 = r2;
                }
                if track_bounding && $vp.0 == usize::MAX {
                    has_bounding = true;
                }
            }};
        }

        out.max_r2 = 0.0;

        match mask {
            // 1 vertex inside
            0b0001 => {
                // In: 0. Entry: 3->0. Exit: 0->1
                push_raw!(
                    i_us[3],
                    i_vs[3],
                    (*poly.edge_planes.get_unchecked(3), hp.plane_idx),
                    hp.plane_idx
                ); // Interp 3
                push_raw!(
                    *us_ptr.add(0),
                    *vs_ptr.add(0),
                    *poly.vertex_planes.get_unchecked(0),
                    *poly.edge_planes.get_unchecked(0)
                ); // Input 0
                push_raw!(
                    i_us[0],
                    i_vs[0],
                    (*poly.edge_planes.get_unchecked(0), hp.plane_idx),
                    hp.plane_idx
                ); // Interp 0
            }
            0b0010 => {
                // In: 1. Entry: 0->1. Exit: 1->2
                push_raw!(
                    i_us[0],
                    i_vs[0],
                    (*poly.edge_planes.get_unchecked(0), hp.plane_idx),
                    hp.plane_idx
                );
                push_raw!(
                    *us_ptr.add(1),
                    *vs_ptr.add(1),
                    *poly.vertex_planes.get_unchecked(1),
                    *poly.edge_planes.get_unchecked(1)
                );
                push_raw!(
                    i_us[1],
                    i_vs[1],
                    (*poly.edge_planes.get_unchecked(1), hp.plane_idx),
                    hp.plane_idx
                );
            }
            0b0100 => {
                // In: 2. Entry: 1->2. Exit: 2->3
                push_raw!(
                    i_us[1],
                    i_vs[1],
                    (*poly.edge_planes.get_unchecked(1), hp.plane_idx),
                    hp.plane_idx
                );
                push_raw!(
                    *us_ptr.add(2),
                    *vs_ptr.add(2),
                    *poly.vertex_planes.get_unchecked(2),
                    *poly.edge_planes.get_unchecked(2)
                );
                push_raw!(
                    i_us[2],
                    i_vs[2],
                    (*poly.edge_planes.get_unchecked(2), hp.plane_idx),
                    hp.plane_idx
                );
            }
            0b1000 => {
                // In: 3. Entry: 2->3. Exit: 3->0
                push_raw!(
                    i_us[2],
                    i_vs[2],
                    (*poly.edge_planes.get_unchecked(2), hp.plane_idx),
                    hp.plane_idx
                );
                push_raw!(
                    *us_ptr.add(3),
                    *vs_ptr.add(3),
                    *poly.vertex_planes.get_unchecked(3),
                    *poly.edge_planes.get_unchecked(3)
                );
                push_raw!(
                    i_us[3],
                    i_vs[3],
                    (*poly.edge_planes.get_unchecked(3), hp.plane_idx),
                    hp.plane_idx
                );
            }

            // 2 vertices inside - Vectorized candidates
            0b0011 => {
                // In: 0, 1. Entry: 3->0. Exit: 1->2.  Order: I3, V0, V1, I1
                push_raw!(
                    i_us[3],
                    i_vs[3],
                    (*poly.edge_planes.get_unchecked(3), hp.plane_idx),
                    hp.plane_idx
                );
                push_raw!(
                    *us_ptr.add(0),
                    *vs_ptr.add(0),
                    *poly.vertex_planes.get_unchecked(0),
                    *poly.edge_planes.get_unchecked(0)
                );
                push_raw!(
                    *us_ptr.add(1),
                    *vs_ptr.add(1),
                    *poly.vertex_planes.get_unchecked(1),
                    *poly.edge_planes.get_unchecked(1)
                );
                push_raw!(
                    i_us[1],
                    i_vs[1],
                    (*poly.edge_planes.get_unchecked(1), hp.plane_idx),
                    hp.plane_idx
                );
            }
            0b0110 => {
                // In: 1, 2. Entry: 0->1. Exit: 2->3. Order: I0, V1, V2, I2
                push_raw!(
                    i_us[0],
                    i_vs[0],
                    (*poly.edge_planes.get_unchecked(0), hp.plane_idx),
                    hp.plane_idx
                );
                push_raw!(
                    *us_ptr.add(1),
                    *vs_ptr.add(1),
                    *poly.vertex_planes.get_unchecked(1),
                    *poly.edge_planes.get_unchecked(1)
                );
                push_raw!(
                    *us_ptr.add(2),
                    *vs_ptr.add(2),
                    *poly.vertex_planes.get_unchecked(2),
                    *poly.edge_planes.get_unchecked(2)
                );
                push_raw!(
                    i_us[2],
                    i_vs[2],
                    (*poly.edge_planes.get_unchecked(2), hp.plane_idx),
                    hp.plane_idx
                );
            }
            0b1100 => {
                // In: 2, 3. Entry: 1->2. Exit: 3->0. Order: I1, V2, V3, I3
                push_raw!(
                    i_us[1],
                    i_vs[1],
                    (*poly.edge_planes.get_unchecked(1), hp.plane_idx),
                    hp.plane_idx
                );
                push_raw!(
                    *us_ptr.add(2),
                    *vs_ptr.add(2),
                    *poly.vertex_planes.get_unchecked(2),
                    *poly.edge_planes.get_unchecked(2)
                );
                push_raw!(
                    *us_ptr.add(3),
                    *vs_ptr.add(3),
                    *poly.vertex_planes.get_unchecked(3),
                    *poly.edge_planes.get_unchecked(3)
                );
                push_raw!(
                    i_us[3],
                    i_vs[3],
                    (*poly.edge_planes.get_unchecked(3), hp.plane_idx),
                    hp.plane_idx
                );
            }
            0b1001 => {
                // In: 3, 0. Entry: 2->3. Exit: 0->1. Order: I2, V3, V0, I0
                push_raw!(
                    i_us[2],
                    i_vs[2],
                    (*poly.edge_planes.get_unchecked(2), hp.plane_idx),
                    hp.plane_idx
                );
                push_raw!(
                    *us_ptr.add(3),
                    *vs_ptr.add(3),
                    *poly.vertex_planes.get_unchecked(3),
                    *poly.edge_planes.get_unchecked(3)
                );
                push_raw!(
                    *us_ptr.add(0),
                    *vs_ptr.add(0),
                    *poly.vertex_planes.get_unchecked(0),
                    *poly.edge_planes.get_unchecked(0)
                );
                push_raw!(
                    i_us[0],
                    i_vs[0],
                    (*poly.edge_planes.get_unchecked(0), hp.plane_idx),
                    hp.plane_idx
                );
            }

            // 3 vertices inside
            0b0111 => {
                // In: 0, 1, 2. Entry: 3->0. Exit: 2->3. Order: I3, V0, V1, V2, I2
                push_raw!(
                    i_us[3],
                    i_vs[3],
                    (*poly.edge_planes.get_unchecked(3), hp.plane_idx),
                    hp.plane_idx
                );
                push_raw!(
                    *us_ptr.add(0),
                    *vs_ptr.add(0),
                    *poly.vertex_planes.get_unchecked(0),
                    *poly.edge_planes.get_unchecked(0)
                );
                push_raw!(
                    *us_ptr.add(1),
                    *vs_ptr.add(1),
                    *poly.vertex_planes.get_unchecked(1),
                    *poly.edge_planes.get_unchecked(1)
                );
                push_raw!(
                    *us_ptr.add(2),
                    *vs_ptr.add(2),
                    *poly.vertex_planes.get_unchecked(2),
                    *poly.edge_planes.get_unchecked(2)
                );
                push_raw!(
                    i_us[2],
                    i_vs[2],
                    (*poly.edge_planes.get_unchecked(2), hp.plane_idx),
                    hp.plane_idx
                );
            }
            0b1110 => {
                // In: 1, 2, 3. Entry: 0->1. Exit: 3->0. Order: I0, V1, V2, V3, I3
                push_raw!(
                    i_us[0],
                    i_vs[0],
                    (*poly.edge_planes.get_unchecked(0), hp.plane_idx),
                    hp.plane_idx
                );
                push_raw!(
                    *us_ptr.add(1),
                    *vs_ptr.add(1),
                    *poly.vertex_planes.get_unchecked(1),
                    *poly.edge_planes.get_unchecked(1)
                );
                push_raw!(
                    *us_ptr.add(2),
                    *vs_ptr.add(2),
                    *poly.vertex_planes.get_unchecked(2),
                    *poly.edge_planes.get_unchecked(2)
                );
                push_raw!(
                    *us_ptr.add(3),
                    *vs_ptr.add(3),
                    *poly.vertex_planes.get_unchecked(3),
                    *poly.edge_planes.get_unchecked(3)
                );
                push_raw!(
                    i_us[3],
                    i_vs[3],
                    (*poly.edge_planes.get_unchecked(3), hp.plane_idx),
                    hp.plane_idx
                );
            }
            0b1101 => {
                // In: 0, 2, 3 (wraps). Entry: 1->2. Exit: 0->1. Order: I1, V2, V3, V0, I0
                push_raw!(
                    i_us[1],
                    i_vs[1],
                    (*poly.edge_planes.get_unchecked(1), hp.plane_idx),
                    hp.plane_idx
                );
                push_raw!(
                    *us_ptr.add(2),
                    *vs_ptr.add(2),
                    *poly.vertex_planes.get_unchecked(2),
                    *poly.edge_planes.get_unchecked(2)
                );
                push_raw!(
                    *us_ptr.add(3),
                    *vs_ptr.add(3),
                    *poly.vertex_planes.get_unchecked(3),
                    *poly.edge_planes.get_unchecked(3)
                );
                push_raw!(
                    *us_ptr.add(0),
                    *vs_ptr.add(0),
                    *poly.vertex_planes.get_unchecked(0),
                    *poly.edge_planes.get_unchecked(0)
                );
                push_raw!(
                    i_us[0],
                    i_vs[0],
                    (*poly.edge_planes.get_unchecked(0), hp.plane_idx),
                    hp.plane_idx
                );
            }
            0b1011 => {
                // In: 0, 1, 3 (wraps). Entry: 2->3. Exit: 1->2. Order: I2, V3, V0, V1, I1
                push_raw!(
                    i_us[2],
                    i_vs[2],
                    (*poly.edge_planes.get_unchecked(2), hp.plane_idx),
                    hp.plane_idx
                );
                push_raw!(
                    *us_ptr.add(3),
                    *vs_ptr.add(3),
                    *poly.vertex_planes.get_unchecked(3),
                    *poly.edge_planes.get_unchecked(3)
                );
                push_raw!(
                    *us_ptr.add(0),
                    *vs_ptr.add(0),
                    *poly.vertex_planes.get_unchecked(0),
                    *poly.edge_planes.get_unchecked(0)
                );
                push_raw!(
                    *us_ptr.add(1),
                    *vs_ptr.add(1),
                    *poly.vertex_planes.get_unchecked(1),
                    *poly.edge_planes.get_unchecked(1)
                );
                push_raw!(
                    i_us[1],
                    i_vs[1],
                    (*poly.edge_planes.get_unchecked(1), hp.plane_idx),
                    hp.plane_idx
                );
            }

            _ => std::hint::unreachable_unchecked(),
        }

        out.len = out_len;
        out.has_bounding_ref = if track_bounding { has_bounding } else { false };
        ClipResult::Changed
    }
}
