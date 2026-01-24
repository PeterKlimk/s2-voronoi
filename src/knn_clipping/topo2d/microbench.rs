//! Internal microbench harnesses for `topo2d`/`clip_convex`.
//!
//! This is intentionally "close to the metal" (no Criterion) to keep iteration fast and
//! to make it easy to compare small variants while editing the clippers.

#[cfg(feature = "microbench")]
pub fn run_clip_convex_microbench() {
    use super::clippers::{
        clip_convex_small_bool, clip_convex_small_bool_out_idx, clip_convex_small_bool_out_idx_ptr,
    };
    use super::{ClipResult, HalfPlane, PolyBuffer};

    use std::hint::black_box;
    use std::time::{Duration, Instant};

    let target_ms: u64 = std::env::var("S2_VORONOI_BENCH_TARGET_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(200)
        .clamp(10, 10_000);
    let samples: usize = std::env::var("S2_VORONOI_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(9)
        .clamp(3, 101);
    let hp_pool_len: usize = std::env::var("S2_VORONOI_BENCH_HP_POOL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1024)
        .clamp(32, 1 << 20)
        .next_power_of_two();

    fn median(mut xs: Vec<f64>) -> f64 {
        if xs.is_empty() {
            return f64::NAN;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    }

    fn make_regular_poly_bounded<const N: usize>(radius: f64) -> PolyBuffer {
        let mut p = PolyBuffer::new();
        p.len = N;
        p.max_r2 = 0.0;
        p.has_bounding_ref = false;
        for i in 0..N {
            let theta = std::f64::consts::TAU * (i as f64) / (N as f64);
            let (s, c) = theta.sin_cos();
            let u = radius * c;
            let v = radius * s;
            p.us[i] = u;
            p.vs[i] = v;
            p.vertex_planes[i] = (i, (i + 1) % N);
            p.edge_planes[i] = i;
            p.max_r2 = p.max_r2.max(u * u + v * v);
        }
        p
    }

    fn make_regular_poly_unbounded<const N: usize>(
        radius: f64,
        bounding_verts: usize,
    ) -> PolyBuffer {
        debug_assert!(bounding_verts > 0 && bounding_verts <= N);

        let mut p = PolyBuffer::new();
        p.len = N;
        p.max_r2 = 0.0;
        p.has_bounding_ref = true;

        for i in 0..N {
            let theta = std::f64::consts::TAU * (i as f64) / (N as f64);
            let (s, c) = theta.sin_cos();
            let u = radius * c;
            let v = radius * s;
            p.us[i] = u;
            p.vs[i] = v;

            if i < bounding_verts {
                p.vertex_planes[i] = (usize::MAX, i);
            } else {
                p.vertex_planes[i] = (i, (i + 1) % N);
            }

            p.edge_planes[i] = i;
            p.max_r2 = p.max_r2.max(u * u + v * v);
        }

        p
    }

    fn calibrate(target: Duration, mut f: impl FnMut(u64)) -> u64 {
        let mut iters = 1u64;
        loop {
            let t0 = Instant::now();
            f(iters);
            let dt = t0.elapsed();
            if dt >= target {
                return iters;
            }
            iters = iters.saturating_mul(2);
            if iters == 0 {
                return u64::MAX;
            }
        }
    }

    fn bench_ns_per_call(
        label: &str,
        target: Duration,
        samples: usize,
        calls_per_iter: u64,
        mut f: impl FnMut(u64),
    ) -> (f64, f64) {
        let mut runs = Vec::with_capacity(samples);

        // Warmup.
        f(10_000);

        let iters = calibrate(target, |n| f(n));
        for _ in 0..samples {
            let t0 = Instant::now();
            f(iters);
            let dt = t0.elapsed();
            let total_calls = (iters as f64) * (calls_per_iter as f64);
            let ns = dt.as_secs_f64() * 1e9 / total_calls;
            runs.push(ns);
        }

        let med = median(runs.clone());
        let min = runs
            .into_iter()
            .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
        eprintln!("{label:<32} median {med:>10.2} ns/call  min {min:>10.2}");
        (med, min)
    }

    #[inline(always)]
    fn next_idx(state: &mut u64, mask: usize) -> usize {
        *state = state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        (*state as usize) & mask
    }

    fn build_hp_pools<const N: usize>(
        poly: &PolyBuffer,
        pool_len: usize,
    ) -> (Vec<HalfPlane>, Vec<HalfPlane>, Vec<HalfPlane>) {
        debug_assert!(pool_len.is_power_of_two());
        let full_mask_val: u8 = ((1u16 << N) - 1) as u8;

        #[derive(Clone, Copy)]
        struct Rng(u64);
        impl Rng {
            #[inline(always)]
            fn next_u64(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                self.0 = x;
                x.wrapping_mul(0x2545F4914F6CDD1D)
            }
            #[inline(always)]
            fn next_f64(&mut self) -> f64 {
                let bits = self.next_u64() >> 11;
                (bits as f64) * (1.0 / ((1u64 << 53) as f64))
            }
            #[inline(always)]
            fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
                lo + (hi - lo) * self.next_f64()
            }
        }

        #[inline(always)]
        fn mask_for<const N: usize>(poly: &PolyBuffer, hp: &HalfPlane) -> u8 {
            let mut mask: u8 = 0;
            let neg_eps = -hp.eps;
            for i in 0..N {
                let d = hp.signed_dist(poly.us[i], poly.vs[i]);
                mask |= ((d >= neg_eps) as u8) << i;
            }
            mask
        }

        let mut rng = Rng(0xD1CE_B00C_A55E_D00Du64 ^ (N as u64));

        let mut changed = Vec::with_capacity(pool_len);
        while changed.len() < pool_len {
            let theta = rng.range_f64(0.0, std::f64::consts::TAU);
            let scale = rng.range_f64(0.25, 4.0);
            let (s, c) = theta.sin_cos();
            let a = scale * c;
            let b = scale * s;

            let norm = ((a.mul_add(a, b * b)) as f32).sqrt() as f64;
            let cc = rng.range_f64(-0.6 * norm, 0.6 * norm);
            let hp = HalfPlane::new_unnormalized(a, b, cc, 1_000_000 + changed.len());
            let m = mask_for::<N>(poly, &hp);
            if m != 0 && m != full_mask_val {
                changed.push(hp);
            }
        }

        let mut unchanged = Vec::with_capacity(pool_len);
        while unchanged.len() < pool_len {
            let theta = rng.range_f64(0.0, std::f64::consts::TAU);
            let scale = rng.range_f64(0.25, 4.0);
            let (s, c) = theta.sin_cos();
            let a = scale * c;
            let b = scale * s;
            let norm = ((a.mul_add(a, b * b)) as f32).sqrt() as f64;

            let cc = rng.range_f64(0.1 * norm, 2.5 * norm);
            let hp = HalfPlane::new_unnormalized(a, b, cc, 2_000_000 + unchanged.len());
            if mask_for::<N>(poly, &hp) == full_mask_val {
                unchanged.push(hp);
            }
        }

        let mut combo = Vec::with_capacity(pool_len * 2);
        for i in 0..pool_len {
            combo.push(changed[i]);
            combo.push(unchanged[i]);
        }

        (changed, unchanged, combo)
    }

    fn build_hp_pools_unbounded<const N: usize>(
        poly: &PolyBuffer,
        bounding_verts: usize,
        pool_len: usize,
    ) -> (
        Vec<HalfPlane>,
        Vec<HalfPlane>,
        Vec<HalfPlane>,
        Vec<HalfPlane>,
    ) {
        debug_assert!(pool_len.is_power_of_two());
        debug_assert!(bounding_verts > 0 && bounding_verts < N);
        let full_mask_val: u8 = ((1u16 << N) - 1) as u8;
        let bounding_mask: u8 = ((1u16 << bounding_verts) - 1) as u8;

        #[inline(always)]
        fn mask_for<const N: usize>(poly: &PolyBuffer, hp: &HalfPlane) -> u8 {
            let mut mask: u8 = 0;
            let neg_eps = -hp.eps;
            for i in 0..N {
                let d = hp.signed_dist(poly.us[i], poly.vs[i]);
                mask |= ((d >= neg_eps) as u8) << i;
            }
            mask
        }

        let mut keep_base = Vec::new();
        let mut drop_base = Vec::new();
        let mut unchanged_base = Vec::new();

        let steps = N * 64;
        let mut plane_idx = 3_000_000usize;
        for j in 0..steps {
            let phi = std::f64::consts::TAU * (j as f64) / (steps as f64);
            let (s, c) = phi.sin_cos();
            let a = c;
            let b = s;

            let mut dots = [0.0f64; 8];
            for i in 0..N {
                dots[i] = a.mul_add(poly.us[i], b * poly.vs[i]);
            }

            if unchanged_base.len() < 64 {
                let mut min_dot = dots[0];
                for &d in &dots[1..N] {
                    if d < min_dot {
                        min_dot = d;
                    }
                }
                let t = min_dot - 1.0;
                let hp = HalfPlane::new_unnormalized(a, b, -t, plane_idx);
                plane_idx += 1;
                if mask_for::<N>(poly, &hp) == full_mask_val {
                    unchanged_base.push(hp);
                }
            }

            let mut idxs = [0usize; 8];
            for i in 0..N {
                idxs[i] = i;
            }
            idxs[..N].sort_by(|&i1, &i2| dots[i2].partial_cmp(&dots[i1]).unwrap());

            for k in 1..N {
                let t = 0.5 * (dots[idxs[k - 1]] + dots[idxs[k]]);
                let hp = HalfPlane::new_unnormalized(a, b, -t, plane_idx);
                plane_idx += 1;
                let m = mask_for::<N>(poly, &hp);
                if m == 0 || m == full_mask_val {
                    continue;
                }

                if (m & bounding_mask) != 0 {
                    keep_base.push(hp);
                } else {
                    drop_base.push(hp);
                }
            }
        }

        assert!(
            !keep_base.is_empty(),
            "failed to build keep_bounding base pool (N={N}, bounding_verts={bounding_verts})"
        );
        assert!(
            !drop_base.is_empty(),
            "failed to build drop_bounding base pool (N={N}, bounding_verts={bounding_verts})"
        );
        assert!(
            !unchanged_base.is_empty(),
            "failed to build unchanged base pool (N={N}, bounding_verts={bounding_verts})"
        );

        fn repeat_to_len(mut base: Vec<HalfPlane>, pool_len: usize) -> Vec<HalfPlane> {
            if base.len() >= pool_len {
                base.truncate(pool_len);
                return base;
            }
            let orig = base.clone();
            while base.len() < pool_len {
                base.push(orig[base.len() % orig.len()]);
            }
            base
        }

        let keep_bounding = repeat_to_len(keep_base, pool_len);
        let drop_bounding = repeat_to_len(drop_base, pool_len);
        let unchanged = repeat_to_len(unchanged_base, pool_len);

        let mut combo = Vec::with_capacity(pool_len * 2);
        for i in 0..pool_len {
            combo.push(keep_bounding[i]);
            combo.push(drop_bounding[i]);
        }

        (keep_bounding, drop_bounding, unchanged, combo)
    }

    fn run_for<const N: usize>(target: Duration, samples: usize, hp_pool_len: usize) {
        let poly = make_regular_poly_bounded::<N>(1.0);
        let (hps_changed, hps_unchanged, hps_combo) = build_hp_pools::<N>(&poly, hp_pool_len);

        // Pre-allocate output buffers.
        let mut out_base = PolyBuffer::new();
        let mut out_out_idx = PolyBuffer::new();
        let mut out_out_idx_ptr = PolyBuffer::new();

        // Sanity: ensure the intended regimes.
        assert!(matches!(
            clip_convex_small_bool::<N>(&poly, &hps_changed[0], &mut out_base),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex_small_bool_out_idx::<N>(&poly, &hps_changed[0], &mut out_out_idx),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex_small_bool_out_idx_ptr::<N>(&poly, &hps_changed[0], &mut out_out_idx_ptr),
            ClipResult::Changed
        ));

        // Poison outs; unchanged variants must not write.
        out_base.len = 13;
        out_base.us[0] = 123.0;
        out_out_idx.len = 13;
        out_out_idx.us[0] = 123.0;
        out_out_idx_ptr.len = 13;
        out_out_idx_ptr.us[0] = 123.0;

        assert!(matches!(
            clip_convex_small_bool::<N>(&poly, &hps_unchanged[0], &mut out_base),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex_small_bool_out_idx::<N>(&poly, &hps_unchanged[0], &mut out_out_idx),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex_small_bool_out_idx_ptr::<N>(&poly, &hps_unchanged[0], &mut out_out_idx_ptr),
            ClipResult::Unchanged
        ));

        eprintln!("\nclip_convex microbench (N={N})");

        bench_ns_per_call("small_bool mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_changed.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_base);
            let mut s = 0x1234_5678_9ABC_DEF0u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("out_idx mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_changed.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_out_idx);
            let mut s = 0x1234_5678_9ABC_DEF0u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_out_idx::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("out_idx_ptr mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_changed.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_out_idx_ptr);
            let mut s = 0x1234_5678_9ABC_DEF0u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_out_idx_ptr::<N>(poly, hp, out);
                black_box(r);
            }
        });

        // (Temporarily disabled) Alternating keep/cut can be misleading for batch-friendly ideas.
        let _ = hps_combo;

        bench_ns_per_call("small_bool unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_unchanged.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_base);
            let mut s = 0x0BAD_F00D_1234_5678u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("out_idx unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_unchanged.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_out_idx);
            let mut s = 0x0BAD_F00D_1234_5678u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_out_idx::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("out_idx_ptr unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_unchanged.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_out_idx_ptr);
            let mut s = 0x0BAD_F00D_1234_5678u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_out_idx_ptr::<N>(poly, hp, out);
                black_box(r);
            }
        });

        let bounding_verts: usize = if N <= 4 { 2 } else { 3 };
        let poly_u = make_regular_poly_unbounded::<N>(1.0, bounding_verts);
        let (hps_keep, hps_drop, hps_u_unchanged, hps_u_combo) =
            build_hp_pools_unbounded::<N>(&poly_u, bounding_verts, hp_pool_len);

        let mut out_u_base = PolyBuffer::new();
        let mut out_u_out_idx = PolyBuffer::new();
        let mut out_u_out_idx_ptr = PolyBuffer::new();

        assert!(matches!(
            clip_convex_small_bool::<N>(&poly_u, &hps_keep[0], &mut out_u_base),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex_small_bool_out_idx::<N>(&poly_u, &hps_keep[0], &mut out_u_out_idx),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex_small_bool_out_idx_ptr::<N>(&poly_u, &hps_keep[0], &mut out_u_out_idx_ptr),
            ClipResult::Changed
        ));

        out_u_base.len = 13;
        out_u_base.us[0] = 123.0;
        out_u_out_idx.len = 13;
        out_u_out_idx.us[0] = 123.0;
        out_u_out_idx_ptr.len = 13;
        out_u_out_idx_ptr.us[0] = 123.0;

        assert!(matches!(
            clip_convex_small_bool::<N>(&poly_u, &hps_u_unchanged[0], &mut out_u_base),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex_small_bool_out_idx::<N>(&poly_u, &hps_u_unchanged[0], &mut out_u_out_idx),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex_small_bool_out_idx_ptr::<N>(
                &poly_u,
                &hps_u_unchanged[0],
                &mut out_u_out_idx_ptr
            ),
            ClipResult::Unchanged
        ));

        eprintln!("\nclip_convex microbench unbounded (N={N})");

        bench_ns_per_call("small_bool keep_bounding", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_keep.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_base);
            let mut s = 0xA11C_E11E_A11C_E11Eu64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("out_idx keep_bounding", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_keep.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_out_idx);
            let mut s = 0xA11C_E11E_A11C_E11Eu64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_out_idx::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("out_idx_ptr keep_bounding", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_keep.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_out_idx_ptr);
            let mut s = 0xA11C_E11E_A11C_E11Eu64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_out_idx_ptr::<N>(poly, hp, out);
                black_box(r);
            }
        });

        bench_ns_per_call("small_bool drop_bounding", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_drop.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_base);
            let mut s = 0xB055_1E55_B055_1E55u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("out_idx drop_bounding", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_drop.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_out_idx);
            let mut s = 0xB055_1E55_B055_1E55u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_out_idx::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("out_idx_ptr drop_bounding", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_drop.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_out_idx_ptr);
            let mut s = 0xB055_1E55_B055_1E55u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_out_idx_ptr::<N>(poly, hp, out);
                black_box(r);
            }
        });

        // (Temporarily disabled) Alternating keep/drop is useful, but not while iterating on
        // per-call overhead in the scalar clippers.
        let _ = hps_u_combo;

        bench_ns_per_call("small_bool unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_u_unchanged.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_base);
            let mut s = 0x0BAD_F00D_1234_5678u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("out_idx unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_u_unchanged.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_out_idx);
            let mut s = 0x0BAD_F00D_1234_5678u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_out_idx::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("out_idx_ptr unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_u_unchanged.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_out_idx_ptr);
            let mut s = 0x0BAD_F00D_1234_5678u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_out_idx_ptr::<N>(poly, hp, out);
                black_box(r);
            }
        });
    }

    let target = Duration::from_millis(target_ms);
    run_for::<3>(target, samples, hp_pool_len);
    run_for::<4>(target, samples, hp_pool_len);
    run_for::<5>(target, samples, hp_pool_len);
    run_for::<6>(target, samples, hp_pool_len);
    run_for::<7>(target, samples, hp_pool_len);
    run_for::<8>(target, samples, hp_pool_len);
}

/// Batch clipping microbench: compare serial vs batched clip_convex.
#[cfg(feature = "microbench")]
pub fn run_batch_clip_microbench() {
    use super::clippers::clip_convex;
    use super::types::HalfPlane;
    use super::{ClipResult, PolyBuffer};
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    let target_ms: u64 = std::env::var("S2_VORONOI_BENCH_TARGET_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(200)
        .clamp(10, 10_000);
    let samples: usize = std::env::var("S2_VORONOI_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(9)
        .clamp(3, 101);

    fn median(mut xs: Vec<f64>) -> f64 {
        if xs.is_empty() {
            return f64::NAN;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    }

    fn make_regular_poly<const N: usize>(radius: f64) -> PolyBuffer {
        let mut p = PolyBuffer::new();
        p.len = N;
        p.max_r2 = 0.0;
        p.has_bounding_ref = false;
        for i in 0..N {
            let theta = std::f64::consts::TAU * (i as f64) / (N as f64);
            let (s, c) = theta.sin_cos();
            let u = radius * c;
            let v = radius * s;
            p.us[i] = u;
            p.vs[i] = v;
            p.vertex_planes[i] = (i, (i + 1) % N);
            p.edge_planes[i] = i;
            p.max_r2 = p.max_r2.max(u * u + v * v);
        }
        p
    }

    fn calibrate(target: Duration, mut f: impl FnMut(u64)) -> u64 {
        let mut iters = 1u64;
        loop {
            let t0 = Instant::now();
            f(iters);
            let dt = t0.elapsed();
            if dt >= target {
                return iters;
            }
            iters = iters.saturating_mul(2);
            if iters == 0 {
                return u64::MAX;
            }
        }
    }

    fn bench_ns_per_call(
        label: &str,
        target: Duration,
        samples: usize,
        mut f: impl FnMut(u64),
    ) -> f64 {
        let mut runs = Vec::with_capacity(samples);

        // Warmup.
        f(1000);

        let iters = calibrate(target, |n| f(n));
        for _ in 0..samples {
            let t0 = Instant::now();
            f(iters);
            let dt = t0.elapsed();
            let ns = dt.as_secs_f64() * 1e9 / (iters as f64);
            runs.push(ns);
        }

        let med = median(runs.clone());
        eprintln!("{label:<40} median {med:>10.2} ns/call");
        med
    }

    // Generate 64 half-planes with mixed results
    fn build_hp_pool<const N: usize>(_poly: &PolyBuffer) -> Vec<HalfPlane> {
        let mut hps = Vec::new();
        let mut rng = 0x1234_5678_9ABC_DEF0u64;

        for _ in 0..64 {
            rng = rng
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            let theta = (rng as f64) / (u64::MAX as f64) * std::f64::consts::TAU;
            let scale = 0.5 + ((rng >> 32) as f64) / (u32::MAX as f64) * 2.0;

            let (s, c) = theta.sin_cos();
            let a = scale * c;
            let b = scale * s;
            let norm = (a * a + b * b).sqrt();
            // Use only 16 bits for the offset term; shifting the full `u64` and dividing by
            // `u16::MAX` makes this enormous and causes nearly-every plane to be trivially
            // "unchanged" (bench becomes meaningless).
            let r16 = ((rng >> 16) & 0xFFFF) as u16;
            let cc = ((r16 as f64) / (u16::MAX as f64) - 0.5) * norm * 2.0;

            let hp = HalfPlane::new_unnormalized(a, b, cc, hps.len());
            hps.push(hp);
        }

        hps
    }

    eprintln!("\n=== Batch Clip Microbench ===\n");

    for n in [3, 4, 5, 6, 7, 8] {
        eprintln!("\n--- N = {n} ---");

        match n {
            3 => run_batch_bench::<3>(target_ms, samples),
            4 => run_batch_bench::<4>(target_ms, samples),
            5 => run_batch_bench::<5>(target_ms, samples),
            6 => run_batch_bench::<6>(target_ms, samples),
            7 => run_batch_bench::<7>(target_ms, samples),
            8 => run_batch_bench::<8>(target_ms, samples),
            _ => unreachable!(),
        }
    }

    eprintln!("\n=== End Batch Clip Microbench ===\n");

    fn run_batch_bench<const N: usize>(target_ms: u64, samples: usize) {
        use super::batch_clip::{clip_batch as clip_batch4, HpBatch as HpBatch4};
        use super::batch_clip8::{clip_batch as clip_batch8, HpBatch8};

        let poly = make_regular_poly::<N>(1.0);
        let hp_pool = build_hp_pool::<N>(&poly);

        let target = Duration::from_millis(target_ms);

        // Pre-allocate buffers
        let mut out1 = PolyBuffer::new();
        let mut out2 = PolyBuffer::new();
        let mut out3 = PolyBuffer::new();

        #[inline(always)]
        fn consume_poly(poly: &PolyBuffer) {
            black_box(poly.len);
            black_box(poly.max_r2);
        }

        // Quick workload characterization (not timed). This is useful because the
        // relative performance is dominated by how often we hit "Unchanged" and how
        // often polygons become empty before all planes are applied.
        {
            let iters = 20_000usize;

            // ---- serial (up to 8 scalar clips, early-break on empty) ----
            let mut serial_planes: u64 = 0;
            let mut serial_unchanged: u64 = 0;
            let mut serial_empty_iters: u64 = 0;

            // ---- batch4 (two calls; second call skipped if empty) ----
            let mut b4_second_called: u64 = 0;
            let mut b4_r0_unchanged: u64 = 0;
            let mut b4_r1_unchanged: u64 = 0;
            let mut b4_empty_after_r0: u64 = 0;
            let mut b4_empty_after_r1: u64 = 0;

            // ---- batch8 (one call) ----
            let mut b8_unchanged: u64 = 0;
            let mut b8_empty: u64 = 0;

            for iter in 0..iters {
                let start = iter % 56;
                let hps_slice = &hp_pool[start..start + 8];

                // serial
                let mut cur_slot: u8 = 0; // 0=poly,1=out1,2=out2
                for hp in hps_slice {
                    serial_planes += 1;
                    match cur_slot {
                        0 => match clip_convex(&poly, hp, &mut out1) {
                            ClipResult::Unchanged => serial_unchanged += 1,
                            ClipResult::Changed => {
                                cur_slot = 1;
                                if out1.len == 0 {
                                    serial_empty_iters += 1;
                                    break;
                                }
                            }
                            ClipResult::TooManyVertices => {}
                        },
                        1 => match clip_convex(&out1, hp, &mut out2) {
                            ClipResult::Unchanged => serial_unchanged += 1,
                            ClipResult::Changed => {
                                cur_slot = 2;
                                if out2.len == 0 {
                                    serial_empty_iters += 1;
                                    break;
                                }
                            }
                            ClipResult::TooManyVertices => {}
                        },
                        2 => match clip_convex(&out2, hp, &mut out1) {
                            ClipResult::Unchanged => serial_unchanged += 1,
                            ClipResult::Changed => {
                                cur_slot = 1;
                                if out1.len == 0 {
                                    serial_empty_iters += 1;
                                    break;
                                }
                            }
                            ClipResult::TooManyVertices => {}
                        },
                        _ => unreachable!(),
                    }
                }

                // batch4
                let b0 = HpBatch4::from_array([hps_slice[0], hps_slice[1], hps_slice[2], hps_slice[3]]);
                let r0 = clip_batch4(&poly, &b0, &mut out2);
                if matches!(r0, ClipResult::Unchanged) {
                    b4_r0_unchanged += 1;
                }
                if matches!(r0, ClipResult::Changed) && out2.len == 0 {
                    b4_empty_after_r0 += 1;
                } else {
                    b4_second_called += 1;
                    let input = if matches!(r0, ClipResult::Changed) { &out2 } else { &poly };
                    let b1 = HpBatch4::from_array([hps_slice[4], hps_slice[5], hps_slice[6], hps_slice[7]]);
                    let r1 = clip_batch4(input, &b1, &mut out3);
                    if matches!(r1, ClipResult::Unchanged) {
                        b4_r1_unchanged += 1;
                    }
                    if matches!(r1, ClipResult::Changed) && out3.len == 0 {
                        b4_empty_after_r1 += 1;
                    }
                }

                // batch8
                let b = HpBatch8::from_array([
                    hps_slice[0], hps_slice[1], hps_slice[2], hps_slice[3],
                    hps_slice[4], hps_slice[5], hps_slice[6], hps_slice[7],
                ]);
                let r = clip_batch8(&poly, &b, &mut out2);
                if matches!(r, ClipResult::Unchanged) {
                    b8_unchanged += 1;
                }
                if matches!(r, ClipResult::Changed) && out2.len == 0 {
                    b8_empty += 1;
                }
            }

            let iters_f = iters as f64;
            eprintln!(
                "workload: serial avg_planes={:.2} empty={:.1}% unchanged/call={:.1}%",
                (serial_planes as f64) / iters_f,
                (serial_empty_iters as f64) * 100.0 / iters_f,
                (serial_unchanged as f64) * 100.0 / (serial_planes as f64),
            );
            eprintln!(
                "workload: batch4 second_call={:.1}% r0_unchanged={:.1}% r1_unchanged={:.1}% empty_after_r0={:.1}% empty_after_r1={:.1}%",
                (b4_second_called as f64) * 100.0 / iters_f,
                (b4_r0_unchanged as f64) * 100.0 / iters_f,
                (b4_r1_unchanged as f64) * 100.0 / (b4_second_called as f64).max(1.0),
                (b4_empty_after_r0 as f64) * 100.0 / iters_f,
                (b4_empty_after_r1 as f64) * 100.0 / (b4_second_called as f64).max(1.0),
            );
            eprintln!(
                "workload: batch8 unchanged={:.1}% empty={:.1}%",
                (b8_unchanged as f64) * 100.0 / iters_f,
                (b8_empty as f64) * 100.0 / iters_f,
            );
        }

        // === Serial clipping (8 clips, one at a time) ===
        bench_ns_per_call("serial (8 clips)", target, samples, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hp_pool);
            let out1 = black_box(&mut out1);
            let out2 = black_box(&mut out2);
            let mut s = 0u64;

            for _ in 0..iters {
                s = s.wrapping_add(1);
                let start = (s % 56) as usize; // Ensure we have room for 8
                let hps_slice = &hps[start..start + 8];

                // Avoid cloning `PolyBuffer` (it's a large fixed-size struct). Instead,
                // treat the initial input as `poly` and then ping-pong between `out1/out2`.
                //
                // 0 = `poly`, 1 = `out1`, 2 = `out2`.
                let mut cur_slot: u8 = 0;

                for hp in hps_slice {
                    match cur_slot {
                        0 => {
                            let r = clip_convex(poly, hp, out1);
                            if matches!(r, ClipResult::Changed) {
                                cur_slot = 1;
                                if out1.len == 0 {
                                    break;
                                }
                            }
                        }
                        1 => {
                            let r = clip_convex(out1, hp, out2);
                            if matches!(r, ClipResult::Changed) {
                                cur_slot = 2;
                                if out2.len == 0 {
                                    break;
                                }
                            }
                        }
                        2 => {
                            let r = clip_convex(out2, hp, out1);
                            if matches!(r, ClipResult::Changed) {
                                cur_slot = 1;
                                if out1.len == 0 {
                                    break;
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                // Consume some output so the compiler can't DCE stores.
                match cur_slot {
                    0 => consume_poly(poly),
                    1 => consume_poly(out1),
                    2 => consume_poly(out2),
                    _ => unreachable!(),
                };
            }
        });

        // === Batch clipping with 4-lane batches (2 batches of 4) ===
        bench_ns_per_call("batch4 (2×4 at once)", target, samples, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hp_pool);
            let out2 = black_box(&mut out2);
            let out3 = black_box(&mut out3);
            let mut s = 0u64;

            for _ in 0..iters {
                s = s.wrapping_add(1);
                let start = (s % 56) as usize; // Ensure we have room for 8

                let batch0_hps = [hps[start], hps[start + 1], hps[start + 2], hps[start + 3]];
                let batch0 = HpBatch4::from_array(batch0_hps);

                // 0 = `poly`, 1 = `out2`, 2 = `out3`
                let mut cur_slot: u8 = 0;
                let mut current_poly = poly;

                // First batch of 4
                let r0 = clip_batch4(current_poly, &batch0, out2);
                if matches!(r0, ClipResult::Changed) {
                    if out2.len == 0 {
                        black_box(r0);
                        continue;
                    }
                    current_poly = out2;
                    cur_slot = 1;
                }

                // Second batch of 4
                let batch1_hps = [
                    hps[start + 4],
                    hps[start + 5],
                    hps[start + 6],
                    hps[start + 7],
                ];
                let batch1 = HpBatch4::from_array(batch1_hps);
                let r1 = clip_batch4(current_poly, &batch1, out3);
                if matches!(r1, ClipResult::Changed) {
                    cur_slot = 2;
                }

                black_box(r0);
                black_box(r1);
                match cur_slot {
                    0 => consume_poly(poly),
                    1 => consume_poly(out2),
                    2 => consume_poly(out3),
                    _ => unreachable!(),
                }
            }
        });

        // === Batch clipping with 8-lane batch (1 batch of 8) ===
        bench_ns_per_call("batch8 (1×8 at once)", target, samples, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hp_pool);
            let out2 = black_box(&mut out2);
            let mut s = 0u64;

            for _ in 0..iters {
                s = s.wrapping_add(1);
                let start = (s % 56) as usize; // Ensure we have room for 8

                let batch_hps = [
                    hps[start],
                    hps[start + 1],
                    hps[start + 2],
                    hps[start + 3],
                    hps[start + 4],
                    hps[start + 5],
                    hps[start + 6],
                    hps[start + 7],
                ];
                let batch = HpBatch8::from_array(batch_hps);

                let r = clip_batch8(poly, &batch, out2);
                black_box(r);
                if matches!(r, ClipResult::Changed) {
                    consume_poly(out2);
                } else {
                    consume_poly(poly);
                }
            }
        });
    }
}
