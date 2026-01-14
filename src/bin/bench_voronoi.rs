//! Benchmark s2-voronoi at large scales.
//!
//! Run with: cargo run --release --bin bench_voronoi
//!
//! Usage:
//!   bench_voronoi              Run default size (100k)
//!   bench_voronoi 100k 500k 1m Run multiple sizes
//!   bench_voronoi --lloyd      Use Lloyd-relaxed points
//!   bench_voronoi -n 10        Run 10 iterations (for profiling)
//!
//! For detailed sub-phase timing, build with: cargo run --release --features timing --bin bench_voronoi

use clap::Parser;
use glam::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use s2_voronoi::{UnitVec3, VoronoiConfig};
use std::io::{self, Write};
use std::time::Instant;

fn parse_count(s: &str) -> Result<usize, String> {
    let s = s.to_lowercase();
    let (num_str, multiplier) = if s.ends_with('m') {
        (&s[..s.len() - 1], 1_000_000)
    } else if s.ends_with('k') {
        (&s[..s.len() - 1], 1_000)
    } else {
        (s.as_str(), 1)
    };

    num_str
        .parse::<f64>()
        .map(|n| (n * multiplier as f64) as usize)
        .map_err(|e| format!("Invalid number '{}': {}", s, e))
}

fn mean_spacing(num_points: usize) -> f32 {
    if num_points == 0 {
        return 0.0;
    }
    (4.0 * std::f32::consts::PI / num_points as f32).sqrt()
}

const PHI: f32 = 1.618_034;

fn fibonacci_sphere_points_with_rng<R: Rng>(n: usize, jitter: f32, rng: &mut R) -> Vec<Vec3> {
    use std::f32::consts::TAU;

    (0..n)
        .map(|i| {
            let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
            let r = (1.0 - y * y).sqrt();
            let theta = TAU * i as f32 / PHI;

            let mut p = Vec3::new(r * theta.cos(), y, r * theta.sin());

            if jitter > 0.0 {
                let tangent = random_tangent_vector(p, rng);
                p = (p + tangent * jitter).normalize();
            }

            p
        })
        .collect()
}

fn random_tangent_vector<R: Rng>(p: Vec3, rng: &mut R) -> Vec3 {
    let arbitrary = if p.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };

    let u = p.cross(arbitrary).normalize();
    let v = p.cross(u);

    let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
    u * angle.cos() + v * angle.sin()
}

fn lloyd_relax_kmeans<R: Rng>(
    points: &mut [Vec3],
    iterations: usize,
    samples_per_site: usize,
    rng: &mut R,
) {
    use kiddo::{ImmutableKdTree, SquaredEuclidean};

    let n = points.len();
    if n < 2 {
        return;
    }

    let num_samples = n * samples_per_site;

    let samples: Vec<Vec3> = (0..num_samples)
        .map(|_| {
            let z: f32 = rng.gen_range(-1.0..1.0);
            let theta: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let r = (1.0 - z * z).sqrt();
            Vec3::new(r * theta.cos(), r * theta.sin(), z)
        })
        .collect();

    let mut entries: Vec<[f32; 3]> = vec![[0.0; 3]; n];

    for _ in 0..iterations {
        for (i, p) in points.iter().enumerate() {
            entries[i] = [p.x, p.y, p.z];
        }
        let tree: ImmutableKdTree<f32, 3> = ImmutableKdTree::new_from_slice(&entries);

        let (sums, counts) = {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                let num_threads = rayon::current_num_threads().max(1);
                let chunk_size = num_samples.div_ceil(num_threads);

                samples
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut local_sums = vec![Vec3::ZERO; n];
                        let mut local_counts = vec![0usize; n];
                        for sample in chunk {
                            let query = [sample.x, sample.y, sample.z];
                            let nearest = tree.approx_nearest_one::<SquaredEuclidean>(&query);
                            let site_idx = nearest.item as usize;
                            local_sums[site_idx] += *sample;
                            local_counts[site_idx] += 1;
                        }
                        (local_sums, local_counts)
                    })
                    .reduce(
                        || (vec![Vec3::ZERO; n], vec![0usize; n]),
                        |(mut sums_a, mut counts_a), (sums_b, counts_b)| {
                            for i in 0..n {
                                sums_a[i] += sums_b[i];
                                counts_a[i] += counts_b[i];
                            }
                            (sums_a, counts_a)
                        },
                    )
            }
            #[cfg(not(feature = "parallel"))]
            {
                let mut sums = vec![Vec3::ZERO; n];
                let mut counts = vec![0usize; n];
                for sample in &samples {
                    let query = [sample.x, sample.y, sample.z];
                    let nearest = tree.approx_nearest_one::<SquaredEuclidean>(&query);
                    let site_idx = nearest.item as usize;
                    sums[site_idx] += *sample;
                    counts[site_idx] += 1;
                }
                (sums, counts)
            }
        };

        for i in 0..n {
            if counts[i] > 0 {
                let centroid = sums[i] / counts[i] as f32;
                if centroid.length_squared() > 1e-10 {
                    points[i] = centroid.normalize();
                }
            }
        }
    }
}

#[derive(Parser)]
#[command(name = "bench_voronoi")]
#[command(about = "Benchmark s2-voronoi at various scales")]
struct Args {
    /// Cell counts to benchmark (e.g., 100k, 1m, 10M)
    #[arg(value_parser = parse_count)]
    sizes: Vec<usize>,

    /// Random seed
    #[arg(short, long, default_value_t = 12345)]
    seed: u64,

    /// Use Lloyd-relaxed points (well-behaved, like production)
    #[arg(long)]
    lloyd: bool,

    /// Compare against convex hull ground truth (slow, max 100k)
    #[arg(long)]
    validate: bool,

    /// Disable preprocessing (merge near-coincident points) for benchmarking.
    #[arg(long)]
    no_preprocess: bool,

    /// Number of iterations to run (useful for profiling)
    #[arg(short = 'n', long, default_value_t = 1)]
    repeat: usize,
}

fn generate_points(n: usize, seed: u64, lloyd: bool) -> Vec<Vec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let jitter_scale = if lloyd { 0.1 } else { 0.25 };
    let jitter = mean_spacing(n) * jitter_scale;
    let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);
    if lloyd {
        lloyd_relax_kmeans(&mut points, 2, 20, &mut rng);
    }
    points
}

fn format_rate(count: usize, ms: f64) -> String {
    if ms <= 0.0 {
        return "N/A".to_string();
    }
    let per_sec = count as f64 / (ms / 1000.0);
    if per_sec >= 1_000_000.0 {
        format!("{:.2}M/s", per_sec / 1_000_000.0)
    } else if per_sec >= 1_000.0 {
        format!("{:.1}k/s", per_sec / 1000.0)
    } else {
        format!("{:.0}/s", per_sec)
    }
}

fn format_num(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        format!("{}", n)
    }
}

#[cfg(feature = "qhull")]
fn validate_against_hull(points: &[Vec3], preprocess: bool) {
    println!("\nValidating against convex hull ground truth...");

    let t0 = Instant::now();
    let hull = s2_voronoi::convex_hull::compute_voronoi_qhull(points);
    let hull_time = t0.elapsed().as_secs_f64() * 1000.0;

    let unit_points: Vec<UnitVec3> = points
        .iter()
        .map(|p| UnitVec3::new(p.x, p.y, p.z))
        .collect();

    let t1 = Instant::now();
    let s2_output = s2_voronoi::compute_with(&unit_points, VoronoiConfig { preprocess })
        .expect("s2-voronoi should succeed");
    let s2_time = t1.elapsed().as_secs_f64() * 1000.0;

    let mut exact_match = 0usize;
    let mut bad_cells = 0usize;

    for i in 0..points.len() {
        let hull_count = hull.diagram.cell(i).len();
        let s2_count = s2_output.diagram.cell(i).len();

        if hull_count == s2_count {
            exact_match += 1;
        }
        if s2_count < 3 {
            bad_cells += 1;
        }
    }

    let match_pct = exact_match as f64 / points.len() as f64 * 100.0;

    println!("  Convex hull time: {:>8.1}ms", hull_time);
    println!(
        "  s2-voronoi time:  {:>8.1}ms ({:.1}x faster)",
        s2_time,
        hull_time / s2_time
    );
    println!(
        "  Exact matches:    {:>8} / {} ({:.2}%)",
        exact_match,
        points.len(),
        match_pct
    );
    if bad_cells > 0 {
        println!("  Invalid cells:    {:>8} (< 3 vertices)", bad_cells);
    }
    if !s2_output.diagnostics.is_clean() {
        println!(
            "  Diagnostics:      {:>8} bad, {} degenerate",
            s2_output.diagnostics.bad_cells.len(),
            s2_output.diagnostics.degenerate_cells.len()
        );
    }
}

#[cfg(not(feature = "qhull"))]
fn validate_against_hull(_points: &[Vec3], _preprocess: bool) {
    println!("\nValidation requires the `qhull` feature; rebuild with --features qhull.");
}

struct BenchResult {
    n: usize,
    time_ms: f64,
    num_vertices: usize,
    num_cells: usize,
}

fn run_benchmark_with_config(points: &[UnitVec3], config: VoronoiConfig) -> BenchResult {
    let n = points.len();

    let t0 = Instant::now();
    let output = s2_voronoi::compute_with(points, config).expect("s2-voronoi should succeed");
    let time_ms = t0.elapsed().as_secs_f64() * 1000.0;

    #[cfg(debug_assertions)]
    {
        use s2_voronoi::validation::validate;
        let report = validate(&output.diagram);
        if !report.is_perfect() {
            eprintln!("WARNING: Validation failed for n={}: {}", n, report);
        } else {
            println!("Validation passed for n={}", n);
        }
    }

    BenchResult {
        n,
        time_ms,
        num_vertices: output.diagram.vertices.len(),
        num_cells: output.diagram.num_cells(),
    }
}

fn main() {
    let args = Args::parse();

    println!("s2-voronoi Benchmark");
    println!("====================\n");

    let sizes: Vec<usize> = if args.sizes.is_empty() {
        vec![100_000]
    } else {
        args.sizes
    };

    let point_type = if args.lloyd {
        "Lloyd-relaxed"
    } else {
        "fibonacci+jitter"
    };

    println!("Configuration:");
    println!("  seed = {}", args.seed);
    println!("  point type = {}", point_type);
    println!(
        "  sizes = {:?}",
        sizes.iter().map(|&n| format_num(n)).collect::<Vec<_>>()
    );
    if args.no_preprocess {
        println!("  preprocess = disabled (no point merging)");
    }
    if args.repeat > 1 {
        println!("  repeat = {}", args.repeat);
    }

    #[cfg(feature = "timing")]
    println!("  timing = enabled (detailed sub-phase timing will be printed)");

    let mut results: Vec<BenchResult> = Vec::new();

    for n in &sizes {
        println!("\n{}", "=".repeat(60));
        println!("Benchmarking n = {}", format_num(*n));
        println!("{}", "=".repeat(60));

        let t_gen = Instant::now();
        let points = generate_points(*n, args.seed, args.lloyd);
        let unit_points: Vec<UnitVec3> = points
            .iter()
            .map(|p| UnitVec3::new(p.x, p.y, p.z))
            .collect();
        let gen_time = t_gen.elapsed().as_secs_f64() * 1000.0;
        println!("Point generation: {:.1}ms", gen_time);

        let mut times: Vec<f64> = Vec::with_capacity(args.repeat);
        let mut last_result: Option<BenchResult> = None;

        let config = VoronoiConfig {
            preprocess: !args.no_preprocess,
        };

        for iter in 0..args.repeat {
            if args.repeat > 1 {
                print!("  Iteration {}/{}... ", iter + 1, args.repeat);
                io::stdout().flush().unwrap();
            }

            let result = run_benchmark_with_config(&unit_points, config.clone());
            times.push(result.time_ms);

            if args.repeat > 1 {
                println!("{:.1}ms", result.time_ms);
            }

            last_result = Some(result);
        }

        let result = last_result.unwrap();

        println!("\nResults:");
        if args.repeat > 1 {
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let avg = times.iter().sum::<f64>() / times.len() as f64;
            println!("  Min time:      {:>8.1}ms", min);
            println!("  Max time:      {:>8.1}ms", max);
            println!("  Avg time:      {:>8.1}ms", avg);
            println!("  Throughput:    {:>8} (avg)", format_rate(result.n, avg));
        } else {
            println!("  Total time:    {:>8.1}ms", result.time_ms);
            println!(
                "  Throughput:    {:>8}",
                format_rate(result.n, result.time_ms)
            );
        }
        println!("  Vertices:      {:>8}", format_num(result.num_vertices));
        println!("  Cells:         {:>8}", format_num(result.num_cells));
        println!(
            "  Avg verts/cell:{:>8.2}",
            result.num_vertices as f64 * 3.0 / result.num_cells as f64
        );

        if args.validate && *n <= 100_000 {
            validate_against_hull(&points, !args.no_preprocess);
        } else if args.validate && *n > 100_000 {
            println!("\n  (skipping validation for n > 100k - convex hull is slow)");
        }

        results.push(result);
    }

    if results.len() > 1 {
        println!("\n\n{}", "=".repeat(60));
        println!("SUMMARY");
        println!("{}", "=".repeat(60));
        println!(
            "{:>10} | {:>10} | {:>12} | {:>10}",
            "n", "time", "throughput", "verts"
        );
        println!("{:-<10}-+-{:-<10}-+-{:-<12}-+-{:-<10}", "", "", "", "");

        for r in &results {
            println!(
                "{:>10} | {:>9.1}ms | {:>12} | {:>10}",
                format_num(r.n),
                r.time_ms,
                format_rate(r.n, r.time_ms),
                format_num(r.num_vertices)
            );
        }
    }

    println!("\nBenchmark complete.");
}
