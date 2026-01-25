//! Benchmark different bin counts with fixed thread count.
//! Set S2_BIN_COUNT env var to override default (threads * 2).

use s2_voronoi::{compute, UnitVec3};
use std::time::Instant;

fn generate_fibonacci_sphere(n: usize) -> Vec<UnitVec3> {
    let golden_ratio: f32 = (1.0 + 5.0f32.sqrt()) / 2.0;
    let mut points = Vec::with_capacity(n);

    for i in 0..n {
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / golden_ratio;
        let phi = (1.0 - 2.0 * (i as f32 + 0.5) / (n as f32)).acos();
        let x = phi.sin() * theta.cos();
        let y = phi.sin() * theta.sin();
        let z = phi.cos();
        points.push(UnitVec3::new(x, y, z));
    }
    points
}

struct Stats {
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
}

fn compute_stats(samples: &[f64]) -> Stats {
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    Stats {
        mean,
        std_dev: variance.sqrt(),
        min: samples.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        max: samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
    }
}

fn run_bench(n_points: usize, bin_count: usize, samples: usize) -> Stats {
    // Set bin count override
    std::env::set_var("S2_BIN_COUNT", bin_count.to_string());

    // Generate points once (reuse across samples)
    let points = generate_fibonacci_sphere(n_points);

    let mut times = Vec::with_capacity(samples);

    for _ in 0..samples {
        // Warmup
        drop(compute(&points));

        // Timed run
        let start = Instant::now();
        let _diagram = compute(&points).unwrap();
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64() * 1000.0);
    }

    compute_stats(&times)
}

fn main() {
    let n_points = 2_500_000;
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(12);
    let samples = 5;

    // Initialize rayon thread pool once
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    println!("Benchmarking {n_points} Voronoi cells");
    println!("Fixed thread count: {num_threads}, Samples per config: {samples}");
    println!();

    // Test various bins-per-thread ratios
    let bins_per_thread = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0];

    println!("{:>10} {:>10} {:>12} {:>10} {:>10} {:>10}",
        "Bins", "Bins/Thread", "Mean (ms)", "StdDev", "Min", "Max");
    println!("{}", "-".repeat(70));

    let mut results = Vec::new();

    for ratio in &bins_per_thread {
        let bin_count = (*ratio * num_threads as f32) as usize;
        let bin_count = bin_count.clamp(6, 96);

        let stats = run_bench(n_points, bin_count, samples);
        let effective_ratio = bin_count as f32 / num_threads as f32;
        let cells_per_sec = (n_points as f64) / (stats.mean / 1000.0);

        println!("{:>10} {:>10.2} {:>10.1} ms {:>8.2} ms {:>8.1} ms {:>8.1} ms ({:.0}k cells/sec)",
            bin_count, effective_ratio, stats.mean, stats.std_dev, stats.min, stats.max, cells_per_sec / 1000.0);
        results.push((bin_count, effective_ratio, stats, cells_per_sec));
    }

    println!();
    println!("Summary (by mean time):");
    let fastest = results.iter().min_by(|a, b| a.2.mean.partial_cmp(&b.2.mean).unwrap()).unwrap();
    println!("Fastest: {} bins ({:.2} bins/thread) - {:.1} ms (±{:.1})",
        fastest.0, fastest.1, fastest.2.mean, fastest.2.std_dev);

    let by_min = results.iter().min_by(|a, b| a.2.min.partial_cmp(&b.2.min).unwrap()).unwrap();
    println!("Best single run: {} bins ({:.2} bins/thread) - {:.1} ms",
        by_min.0, by_min.1, by_min.2.min);
}
