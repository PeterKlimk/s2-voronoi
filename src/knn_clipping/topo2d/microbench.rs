//! Internal microbench harnesses for `topo2d`/`clip_convex`.
//!
//! This is intentionally "close to the metal" (no Criterion) to keep iteration fast and
//! to make it easy to compare small variants while editing the clippers.

#[cfg(feature = "microbench")]
pub fn run_clip_convex_microbench() {
    use super::clippers::{clip_convex, clip_convex_small_bool};
    use super::types::{ClipResult, HalfPlane, PolyBuffer};

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

    #[inline(always)]
    fn approx_eq_f64(a: f64, b: f64) -> bool {
        if a == b {
            return true;
        }
        let diff = (a - b).abs();
        let scale = 1.0 + a.abs().max(b.abs());
        diff <= 1e-12 * scale
    }

    fn assert_same_poly(label: &str, expected: &PolyBuffer, got: &PolyBuffer) {
        assert_eq!(
            expected.len, got.len,
            "{label}: len mismatch (expected {}, got {})",
            expected.len, got.len
        );
        assert!(
            approx_eq_f64(expected.max_r2, got.max_r2),
            "{label}: max_r2 mismatch (expected {}, got {})",
            expected.max_r2,
            got.max_r2
        );
        assert_eq!(
            expected.has_bounding_ref, got.has_bounding_ref,
            "{label}: has_bounding_ref mismatch (expected {}, got {})",
            expected.has_bounding_ref, got.has_bounding_ref
        );

        for i in 0..expected.len {
            assert!(
                approx_eq_f64(expected.us[i], got.us[i]),
                "{label}: us[{i}] mismatch (expected {}, got {})",
                expected.us[i],
                got.us[i]
            );
            assert!(
                approx_eq_f64(expected.vs[i], got.vs[i]),
                "{label}: vs[{i}] mismatch (expected {}, got {})",
                expected.vs[i],
                got.vs[i]
            );
            assert_eq!(
                expected.vertex_planes[i], got.vertex_planes[i],
                "{label}: vertex_planes[{i}] mismatch (expected {:?}, got {:?})",
                expected.vertex_planes[i],
                got.vertex_planes[i]
            );
            assert_eq!(
                expected.edge_planes[i], got.edge_planes[i],
                "{label}: edge_planes[{i}] mismatch (expected {}, got {})",
                expected.edge_planes[i],
                got.edge_planes[i]
            );
        }
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

    fn run_for<const N: usize>(target: Duration, samples: usize, hp_pool_len: usize) {
        let poly = make_regular_poly_bounded::<N>(1.0);
        let (hps_changed, hps_unchanged, hps_combo) = build_hp_pools::<N>(&poly, hp_pool_len);

        // Pre-allocate output buffers.
        let mut out_baseline = PolyBuffer::new();
        let mut out_dispatch = PolyBuffer::new();

        // Sanity: ensure the intended regimes.
        assert!(matches!(
            clip_convex_small_bool::<N>(&poly, &hps_changed[0], &mut out_baseline),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex(&poly, &hps_changed[0], &mut out_dispatch),
            ClipResult::Changed
        ));

        // Poison outs; unchanged variants must not write.
        out_baseline.len = 13;
        out_baseline.us[0] = 123.0;
        out_dispatch.len = 13;
        out_dispatch.us[0] = 123.0;

        assert!(matches!(
            clip_convex_small_bool::<N>(&poly, &hps_unchanged[0], &mut out_baseline),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex(&poly, &hps_unchanged[0], &mut out_dispatch),
            ClipResult::Unchanged
        ));

        // Correctness: dispatch must match baseline output for changed half-planes.
        for (i, hp) in hps_changed.iter().take(64).enumerate() {
            assert!(
                matches!(clip_convex_small_bool::<N>(&poly, hp, &mut out_baseline), ClipResult::Changed),
                "baseline unexpectedly not Changed (N={N}, i={i})"
            );
            let baseline = &out_baseline;

            assert!(
                matches!(clip_convex(&poly, hp, &mut out_dispatch), ClipResult::Changed),
                "dispatch unexpectedly not Changed (N={N}, i={i})"
            );
            assert_same_poly("dispatch", baseline, &out_dispatch);
        }

        eprintln!("\nclip_convex microbench (N={N})");

        bench_ns_per_call("small_bool mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_changed.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_baseline);
            let mut s = 0x1234_5678_9ABC_DEF0u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool::<N>(poly, hp, out);
                black_box(r);
            }
        });

        bench_ns_per_call("dispatch mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_changed.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_dispatch);
            let mut s = 0x1234_5678_9ABC_DEF0u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex(poly, hp, out);
                black_box(r);
            }
        });

        // (Temporarily disabled) Alternating keep/cut can be misleading for batch-friendly ideas.
        let _ = hps_combo;

        bench_ns_per_call("small_bool unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_unchanged.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_baseline);
            let mut s = 0x0BAD_F00D_1234_5678u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex_small_bool::<N>(poly, hp, out);
                black_box(r);
            }
        });

        bench_ns_per_call("dispatch unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(hps_unchanged.as_slice());
            let hp_mask = hps.len() - 1;
            let out = black_box(&mut out_dispatch);
            let mut s = 0x0BAD_F00D_1234_5678u64;
            for _ in 0..iters {
                let hp = &hps[next_idx(&mut s, hp_mask)];
                let r = clip_convex(poly, hp, out);
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
    eprintln!("\n=== Batch Clip Microbench ===\n");
    eprintln!("Batch clipping has been removed.\n");
    eprintln!("=== End Batch Clip Microbench ===\n");
}
