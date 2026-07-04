//! Benchmark voronoi-mesh at large scales.
//!
//! Run with: cargo run --release --features tools --bin bench_voronoi
//!
//! Usage:
//!   bench_voronoi              Run default size (100k)
//!   bench_voronoi 100k 500k 1m Run multiple sizes
//!   bench_voronoi --lloyd      Use Lloyd-relaxed points
//!   bench_voronoi -n 10        Run 10 iterations (for profiling)
//!
//! For detailed sub-phase timing, build with:
//!   cargo run --release --features tools,timing --bin bench_voronoi

use clap::Parser;
use glam::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::io::{self, Write};
use std::time::Instant;
use voronoi_mesh::{PreprocessMode, RepairMode, UnitVec3, VoronoiConfig};

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
#[command(about = "Benchmark voronoi-mesh at various scales")]
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

    /// Point distribution. Well-distributed: fib (default), uniform (true
    /// random). Density-contrast: clustered (caps mixture), bimodal (one dense
    /// cap over a sparse bg), gradient (smooth density ~exp(k·z)), outlier
    /// (uniform plus one tiny pile), splittable (many cell-scale clusters),
    /// mega (one cap holding a fraction of all points). See --dist-param.
    #[arg(long, default_value = "fib")]
    dist: String,

    /// Distribution shape knob, meaning depends on --dist: gradient = k
    /// (steepness, default 4); mega = fraction in the cap (default 0.8);
    /// others ignore it. 0 = use the distribution's default.
    #[arg(long, default_value_t = 0.0)]
    dist_param: f64,

    /// Compare against convex hull ground truth (slow, max 100k)
    #[arg(long)]
    validate: bool,

    /// Disable preprocessing (merge near-coincident points) for benchmarking.
    #[arg(long)]
    no_preprocess: bool,

    /// Disable local repair for benchmarking raw fast-path behavior.
    #[arg(long)]
    no_repair: bool,

    /// Number of iterations to run (useful for profiling)
    #[arg(short = 'n', long, default_value_t = 1)]
    repeat: usize,
}

fn generate_points(n: usize, seed: u64, lloyd: bool, dist: &str, param: f64) -> Vec<Vec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    match dist {
        "uniform" => return (0..n).map(|_| random_unit(&mut rng)).collect(),
        "clustered" => return clustered_points(n, &mut rng),
        "bimodal" => return bimodal_points(n, &mut rng),
        "gradient" => return gradient_points(n, param, &mut rng),
        "outlier" => return outlier_points(n, &mut rng),
        "splittable" => return splittable_points(n, &mut rng),
        "mega" => return mega_points(n, param, &mut rng),
        // Almost everything inside a single tight cap (radius = dist-param,
        // default 0.002 ≈ tighter than one max-res grid cell) — the "everything
        // in one cell" extreme the occupancy rebuild's memory cap cannot split.
        // A ~2% uniform background bounds the rim cells (a fully-isolated cap
        // leaves rim cells spanning the empty hemisphere, which the chart cannot
        // build — a separate correctness limitation, not the perf case here).
        "cap" => {
            let r = if param > 0.0 { param as f32 } else { 0.002 };
            let center = Vec3::new(0.0, 0.0, 1.0);
            let bg = (n / 50).max(16);
            let bulk = n.saturating_sub(bg);
            let mut pts: Vec<Vec3> = (0..bg).map(|_| random_unit(&mut rng)).collect();
            pts.extend((0..bulk).map(|_| cap_point(center, r, &mut rng)));
            return pts;
        }
        _ => {}
    }
    let jitter_scale = if lloyd { 0.1 } else { 0.25 };
    let jitter = mean_spacing(n) * jitter_scale;
    let mut points = fibonacci_sphere_points_with_rng(n, jitter, &mut rng);
    if lloyd {
        lloyd_relax_kmeans(&mut points, 2, 20, &mut rng);
    }
    points
}

fn random_unit<R: Rng>(rng: &mut R) -> Vec3 {
    loop {
        let p = Vec3::new(
            rng.gen_range(-1.0f32..1.0),
            rng.gen_range(-1.0f32..1.0),
            rng.gen_range(-1.0f32..1.0),
        );
        let len_sq = p.length_squared();
        if len_sq > 1e-6 && len_sq <= 1.0 {
            return p / len_sq.sqrt();
        }
    }
}

/// Cap sample around `center` with angular radius `radius` (cosine-weighted
/// disc in the tangent plane, good enough for density stress).
fn cap_point<R: Rng>(center: Vec3, radius: f32, rng: &mut R) -> Vec3 {
    let r = radius * rng.gen_range(0.0f32..1.0).sqrt();
    let theta = rng.gen_range(0.0..std::f32::consts::TAU);
    let any = if center.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let t1 = center.cross(any).normalize();
    let t2 = center.cross(t1);
    (center + t1 * (r * theta.cos()) + t2 * (r * theta.sin())).normalize()
}

/// Caps mixture: ~90% of points in n/1000 clusters with log-uniform radii
/// (0.005..0.1 rad), 10% uniform background. "Game map regions" shape.
fn clustered_points<R: Rng>(n: usize, rng: &mut R) -> Vec<Vec3> {
    let num_clusters = (n / 1000).max(1);
    let centers: Vec<(Vec3, f32)> = (0..num_clusters)
        .map(|_| {
            let radius = 0.005f32 * (0.1f32 / 0.005).powf(rng.gen_range(0.0f32..1.0));
            (random_unit(rng), radius)
        })
        .collect();
    (0..n)
        .map(|_| {
            if rng.gen_range(0.0f32..1.0) < 0.1 {
                random_unit(rng)
            } else {
                let (c, r) = centers[rng.gen_range(0..centers.len())];
                cap_point(c, r, rng)
            }
        })
        .collect()
}

/// Bimodal: 80% of points in one 0.3 rad cap (~160x density contrast),
/// 20% uniform background.
fn bimodal_points<R: Rng>(n: usize, rng: &mut R) -> Vec<Vec3> {
    let center = Vec3::new(0.0, 0.0, 1.0);
    (0..n)
        .map(|_| {
            if rng.gen_range(0.0f32..1.0) < 0.8 {
                cap_point(center, 0.3, rng)
            } else {
                random_unit(rng)
            }
        })
        .collect()
}

/// Smooth density gradient: weight ~ exp(k·z), denser toward +z. The
/// realistic "one region wants high density, another low" case — no discrete
/// cluster. `k` = param (default 4); pole/anti-pole density ratio ~ exp(2k).
fn gradient_points<R: Rng>(n: usize, param: f64, rng: &mut R) -> Vec<Vec3> {
    let k = if param > 0.0 { param as f32 } else { 4.0 };
    let wmax = k.exp();
    let mut pts: Vec<Vec3> = Vec::with_capacity(n);
    while pts.len() < n {
        let z = rng.gen_range(-1.0f32..1.0);
        if rng.gen_range(0.0f32..wmax) < (k * z).exp() {
            let theta = rng.gen_range(0.0..std::f32::consts::TAU);
            let r = (1.0 - z * z).sqrt();
            pts.push(Vec3::new(r * theta.cos(), r * theta.sin(), z));
        }
    }
    pts
}

/// Uniform background plus one tiny sub-cell pile (~500 points in a 0.0025 rad
/// cap): the "one cell over by chance / local pile" case. A spatial grid can't
/// split a sub-cell pile, so it stresses the *detection*, not the fix.
fn outlier_points<R: Rng>(n: usize, rng: &mut R) -> Vec<Vec3> {
    let pile = 500.min(n / 2);
    let center = Vec3::new(0.3, 0.4, 0.866).normalize();
    let mut pts: Vec<Vec3> = (0..n - pile).map(|_| random_unit(rng)).collect();
    pts.extend((0..pile).map(|_| cap_point(center, 0.0025, rng)));
    pts
}

/// Many (40) cell-scale clusters (~0.03 rad, ~8k points each) on a uniform
/// background: dense enough to exceed per-cell targets but larger than a grid
/// cell, so a finer grid *can* subdivide them. The minority-but-spread case.
fn splittable_points<R: Rng>(n: usize, rng: &mut R) -> Vec<Vec3> {
    let k = 40usize;
    let per = 8000.min(n / (2 * k));
    let mut pts: Vec<Vec3> = (0..n - per * k).map(|_| random_unit(rng)).collect();
    for _ in 0..k {
        let c = random_unit(rng);
        pts.extend((0..per).map(|_| cap_point(c, 0.03, rng)));
    }
    pts
}

/// One 0.05 rad cap holding a `param` fraction (default 0.8) of all points,
/// rest uniform: extreme single concentration. The majority-concentration
/// case where a global re-grid is essential (the flat scan is O(occ²)).
fn mega_points<R: Rng>(n: usize, param: f64, rng: &mut R) -> Vec<Vec3> {
    let frac = if param > 0.0 { param } else { 0.8 };
    let bulk = ((n as f64) * frac) as usize;
    // VORONOI_MESH_BENCH_CAP_CENTER places the dense cap to stress bin-line straddling:
    // `pole` (default) = face +Z center, lands in one bin; `edge` = a cube-face
    // edge (two-face/bin seam through the cap); `corner` = a cube corner
    // (three-face straddle).
    let center = match std::env::var("VORONOI_MESH_BENCH_CAP_CENTER")
        .ok()
        .as_deref()
    {
        Some("edge") => Vec3::new(1.0, 0.0, 1.0).normalize(),
        Some("corner") => Vec3::new(1.0, 1.0, 1.0).normalize(),
        _ => Vec3::new(0.0, 0.0, 1.0),
    };
    let mut pts: Vec<Vec3> = (0..n - bulk).map(|_| random_unit(rng)).collect();
    pts.extend((0..bulk).map(|_| cap_point(center, 0.05, rng)));
    pts
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
    let hull = voronoi_mesh::convex_hull::compute_voronoi_qhull(points);
    let hull_time = t0.elapsed().as_secs_f64() * 1000.0;

    let unit_points: Vec<UnitVec3> = points
        .iter()
        .map(|p| UnitVec3::new(p.x, p.y, p.z))
        .collect();

    let t1 = Instant::now();
    let s2_diagram = voronoi_mesh::compute_with(
        &unit_points,
        VoronoiConfig::default().with_preprocess_mode(if preprocess {
            PreprocessMode::Weld
        } else {
            PreprocessMode::Disabled
        }),
    )
    .expect("voronoi-mesh should succeed");
    let s2_time = t1.elapsed().as_secs_f64() * 1000.0;
    let report = voronoi_mesh::validation::validate(&s2_diagram);
    let quality = voronoi_mesh::quality::assess(&s2_diagram);
    let comparison = voronoi_mesh::quality::compare_cell_vertex_counts(&s2_diagram, &hull);

    println!("  Convex hull time: {:>8.1}ms", hull_time);
    println!(
        "  voronoi-mesh time:  {:>8.1}ms ({:.1}x faster)",
        s2_time,
        hull_time / s2_time
    );
    println!(
        "  Exact matches:    {:>8} / {} ({:.2}%)",
        comparison.matching_cell_vertex_counts,
        comparison.total_cells,
        comparison.match_ratio as f64 * 100.0
    );
    println!("  Validation:       {}", report.headline());
    println!("  Quality:          {}", quality.headline());
    if report.degenerate_cells > 0 {
        println!(
            "  Invalid cells:    {:>8} (< 3 vertices)",
            report.degenerate_cells
        );
    }
    if report.cells_with_duplicate_vertices > 0 {
        println!(
            "  Duplicate verts:  {:>8} cells",
            report.cells_with_duplicate_vertices
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
    let diagram = voronoi_mesh::compute_with(points, config).expect("voronoi-mesh should succeed");
    let time_ms = t0.elapsed().as_secs_f64() * 1000.0;

    #[cfg(debug_assertions)]
    {
        use voronoi_mesh::validation::validate;
        let report = validate(&diagram);
        if !report.is_strictly_valid() {
            eprintln!("WARNING: Validation failed for n={}: {}", n, report);
        } else {
            println!("Validation passed for n={}", n);
        }
    }

    BenchResult {
        n,
        time_ms,
        num_vertices: diagram.num_vertices(),
        num_cells: diagram.num_cells(),
    }
}

fn main() {
    let args = Args::parse();

    println!("voronoi-mesh Benchmark");
    println!("====================\n");

    let sizes: Vec<usize> = if args.sizes.is_empty() {
        vec![100_000]
    } else {
        args.sizes
    };

    let point_type = if args.dist != "fib" {
        args.dist.as_str()
    } else if args.lloyd {
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
    if args.no_repair {
        println!("  repair = disabled");
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
        let points = generate_points(*n, args.seed, args.lloyd, &args.dist, args.dist_param);
        let unit_points: Vec<UnitVec3> = points
            .iter()
            .map(|p| UnitVec3::new(p.x, p.y, p.z))
            .collect();
        let gen_time = t_gen.elapsed().as_secs_f64() * 1000.0;
        println!("Point generation: {:.1}ms", gen_time);

        let mut times: Vec<f64> = Vec::with_capacity(args.repeat);
        let mut last_result: Option<BenchResult> = None;

        let config = VoronoiConfig::default()
            .with_preprocess_mode(if args.no_preprocess {
                PreprocessMode::Disabled
            } else {
                PreprocessMode::Weld
            })
            .with_repair_mode(if args.no_repair {
                RepairMode::Disabled
            } else {
                RepairMode::Local3d
            });

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
