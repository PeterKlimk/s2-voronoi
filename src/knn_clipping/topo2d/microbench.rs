//! Internal microbench harnesses for `topo2d`/`clip_convex`.
//!
//! This is intentionally "close to the metal" (no Criterion) to keep iteration fast and
//! to make it easy to compare small variants while editing the clippers.

#[cfg(feature = "microbench")]
pub fn run_clip_convex_microbench() {
    use super::clip_variants::{clip_convex_small_bool_stream, clip_convex_small_bool_stream_ptr};
    use super::clippers::clip_convex_small_bool;
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
        let mut out_stream = PolyBuffer::new();
        let mut out_stream_ptr = PolyBuffer::new();

        // Sanity: ensure the intended regimes.
        assert!(matches!(
            clip_convex_small_bool::<N>(&poly, &hps_changed[0], &mut out_base),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex_small_bool_stream::<N>(&poly, &hps_changed[0], &mut out_stream),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex_small_bool_stream_ptr::<N>(&poly, &hps_changed[0], &mut out_stream_ptr),
            ClipResult::Changed
        ));

        // Poison outs; unchanged variants must not write.
        out_base.len = 13;
        out_base.us[0] = 123.0;
        out_stream.len = 13;
        out_stream.us[0] = 123.0;
        out_stream_ptr.len = 13;
        out_stream_ptr.us[0] = 123.0;

        assert!(matches!(
            clip_convex_small_bool::<N>(&poly, &hps_unchanged[0], &mut out_base),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex_small_bool_stream::<N>(&poly, &hps_unchanged[0], &mut out_stream),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex_small_bool_stream_ptr::<N>(&poly, &hps_unchanged[0], &mut out_stream_ptr),
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
        bench_ns_per_call("stream mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_changed.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_stream);
            let mut s = 0x1234_5678_9ABC_DEF0u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_stream::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("stream_ptr mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_changed.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_stream_ptr);
            let mut s = 0x1234_5678_9ABC_DEF0u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_stream_ptr::<N>(poly, hp, out);
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
        bench_ns_per_call("stream unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_unchanged.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_stream);
            let mut s = 0x0BAD_F00D_1234_5678u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_stream::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("stream_ptr unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_unchanged.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_stream_ptr);
            let mut s = 0x0BAD_F00D_1234_5678u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_stream_ptr::<N>(poly, hp, out);
                black_box(r);
            }
        });

        let bounding_verts: usize = if N <= 4 { 2 } else { 3 };
        let poly_u = make_regular_poly_unbounded::<N>(1.0, bounding_verts);
        let (hps_keep, hps_drop, hps_u_unchanged, hps_u_combo) =
            build_hp_pools_unbounded::<N>(&poly_u, bounding_verts, hp_pool_len);

        let mut out_u_base = PolyBuffer::new();
        let mut out_u_stream = PolyBuffer::new();
        let mut out_u_stream_ptr = PolyBuffer::new();

        assert!(matches!(
            clip_convex_small_bool::<N>(&poly_u, &hps_keep[0], &mut out_u_base),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex_small_bool_stream::<N>(&poly_u, &hps_keep[0], &mut out_u_stream),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex_small_bool_stream_ptr::<N>(&poly_u, &hps_keep[0], &mut out_u_stream_ptr),
            ClipResult::Changed
        ));

        out_u_base.len = 13;
        out_u_base.us[0] = 123.0;
        out_u_stream.len = 13;
        out_u_stream.us[0] = 123.0;
        out_u_stream_ptr.len = 13;
        out_u_stream_ptr.us[0] = 123.0;

        assert!(matches!(
            clip_convex_small_bool::<N>(&poly_u, &hps_u_unchanged[0], &mut out_u_base),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex_small_bool_stream::<N>(&poly_u, &hps_u_unchanged[0], &mut out_u_stream),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex_small_bool_stream_ptr::<N>(&poly_u, &hps_u_unchanged[0], &mut out_u_stream_ptr),
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
        bench_ns_per_call("stream keep_bounding", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_keep.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_stream);
            let mut s = 0xA11C_E11E_A11C_E11Eu64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_stream::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("stream_ptr keep_bounding", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_keep.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_stream_ptr);
            let mut s = 0xA11C_E11E_A11C_E11Eu64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_stream_ptr::<N>(poly, hp, out);
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
        bench_ns_per_call("stream drop_bounding", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_drop.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_stream);
            let mut s = 0xB055_1E55_B055_1E55u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_stream::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("stream_ptr drop_bounding", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_drop.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_stream_ptr);
            let mut s = 0xB055_1E55_B055_1E55u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_stream_ptr::<N>(poly, hp, out);
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
        bench_ns_per_call("stream unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_u_unchanged.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_stream);
            let mut s = 0x0BAD_F00D_1234_5678u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_stream::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("stream_ptr unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly_u);
            let hps = black_box(hps_u_unchanged.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_u_stream_ptr);
            let mut s = 0x0BAD_F00D_1234_5678u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool_stream_ptr::<N>(poly, hp, out);
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
