//! Planar Voronoi benchmark driver (`tools` feature).
//!
//! Times `compute_plane` over uniform / clustered / grid distributions in
//! the unit square, optionally cross-checking with `validate_plane` and —
//! behind the `bench_voronoice` feature — benchmarking the `voronoice`
//! crate (delaunator-based) on identical inputs for a head-to-head.

use std::io::{self, Write};
use std::time::Instant;

use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use s2_voronoi::{compute_plane, validation, PlaneRect};

fn parse_count(s: &str) -> Result<usize, String> {
    let s = s.trim().to_lowercase();
    let (num, mult) = if let Some(stripped) = s.strip_suffix('m') {
        (stripped, 1_000_000)
    } else if let Some(stripped) = s.strip_suffix('k') {
        (stripped, 1_000)
    } else {
        (s.as_str(), 1)
    };
    num.parse::<f64>()
        .map(|v| (v * mult as f64) as usize)
        .map_err(|e| e.to_string())
}

fn format_num(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        n.to_string()
    }
}

#[derive(Parser)]
#[command(name = "bench_plane")]
#[command(about = "Benchmark s2-voronoi's planar pipeline at various scales")]
struct Args {
    /// Point counts to benchmark (e.g., 100k, 1m)
    #[arg(value_parser = parse_count)]
    sizes: Vec<usize>,

    /// Random seed
    #[arg(short, long, default_value_t = 12345)]
    seed: u64,

    /// Point distribution: uniform (default), clustered (gaussian blobs +
    /// uniform background), grid (jittered lattice, well-behaved like
    /// production inputs).
    #[arg(long, default_value = "uniform")]
    dist: String,

    /// Run validate_plane on the result and assert strict validity.
    #[arg(long)]
    validate: bool,

    /// Number of iterations to run (first iteration discarded as warmup
    /// when > 1)
    #[arg(short = 'n', long, default_value_t = 1)]
    repeat: usize,

    /// Also benchmark the `voronoice` crate on the same input
    /// (requires the `bench_voronoice` feature).
    #[arg(long)]
    voronoice: bool,
}

fn generate_points(n: usize, seed: u64, dist: &str) -> Vec<[f32; 2]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    match dist {
        "clustered" => {
            // ~90% of points in n/1000 gaussian blobs, 10% uniform.
            let num_clusters = (n / 1000).max(1);
            let centers: Vec<([f32; 2], f32)> = (0..num_clusters)
                .map(|_| {
                    let sigma = 0.002f32 * (0.03f32 / 0.002).powf(rng.gen_range(0.0f32..1.0));
                    (
                        [rng.gen_range(0.0f32..1.0), rng.gen_range(0.0f32..1.0)],
                        sigma,
                    )
                })
                .collect();
            (0..n)
                .map(|_| {
                    if rng.gen_range(0.0f32..1.0) < 0.1 {
                        [rng.gen_range(0.0f32..1.0), rng.gen_range(0.0f32..1.0)]
                    } else {
                        let (c, sigma) = centers[rng.gen_range(0..centers.len())];
                        // Box-Muller-ish: two uniforms -> approx gaussian.
                        let g = |r: &mut ChaCha8Rng| {
                            let sum: f32 = (0..4).map(|_| r.gen_range(-1.0f32..1.0)).sum();
                            sum * 0.5
                        };
                        [
                            (c[0] + g(&mut rng) * sigma).clamp(0.0, 1.0),
                            (c[1] + g(&mut rng) * sigma).clamp(0.0, 1.0),
                        ]
                    }
                })
                .collect()
        }
        "grid" => {
            let side = (n as f64).sqrt().ceil() as usize;
            let jitter = 0.25 / side as f32;
            let mut points = Vec::with_capacity(n);
            'outer: for j in 0..side {
                for i in 0..side {
                    if points.len() == n {
                        break 'outer;
                    }
                    points.push([
                        ((i as f32 + 0.5) / side as f32 + rng.gen_range(-jitter..jitter))
                            .clamp(0.0, 1.0),
                        ((j as f32 + 0.5) / side as f32 + rng.gen_range(-jitter..jitter))
                            .clamp(0.0, 1.0),
                    ]);
                }
            }
            points
        }
        _ => (0..n)
            .map(|_| [rng.gen_range(0.0f32..1.0), rng.gen_range(0.0f32..1.0)])
            .collect(),
    }
}

fn bench_once(points: &[[f32; 2]]) -> (f64, usize, usize) {
    let t = Instant::now();
    let diagram = compute_plane(points, PlaneRect::unit()).expect("compute_plane failed");
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    (ms, diagram.num_cells(), diagram.num_vertices())
}

#[cfg(feature = "bench_voronoice")]
fn bench_voronoice_once(points: &[[f32; 2]]) -> f64 {
    use voronoice::{BoundingBox, Point, VoronoiBuilder};
    let sites: Vec<Point> = points
        .iter()
        .map(|p| Point {
            x: p[0] as f64,
            y: p[1] as f64,
        })
        .collect();
    let t = Instant::now();
    let v = VoronoiBuilder::default()
        .set_sites(sites)
        .set_bounding_box(BoundingBox::new(Point { x: 0.5, y: 0.5 }, 1.0, 1.0))
        .build()
        .expect("voronoice build failed");
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    std::hint::black_box(v.cells().len());
    ms
}

#[cfg(not(feature = "bench_voronoice"))]
fn bench_voronoice_once(_points: &[[f32; 2]]) -> f64 {
    panic!("rebuild with --features tools,bench_voronoice for the voronoice comparison");
}

fn stats(times: &[f64]) -> (f64, f64) {
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    (min, mean)
}

fn main() {
    let args = Args::parse();

    println!("s2-voronoi planar benchmark");
    println!("===========================\n");

    let sizes: Vec<usize> = if args.sizes.is_empty() {
        vec![100_000]
    } else {
        args.sizes
    };

    println!("Configuration:");
    println!("  seed = {}", args.seed);
    println!("  dist = {}", args.dist);
    println!(
        "  sizes = {:?}",
        sizes.iter().map(|&n| format_num(n)).collect::<Vec<_>>()
    );
    if args.repeat > 1 {
        println!("  repeat = {} (first iteration discarded)", args.repeat);
    }

    for &n in &sizes {
        println!("\n{}", "=".repeat(60));
        println!("Benchmarking n = {}", format_num(n));
        println!("{}", "=".repeat(60));

        let t_gen = Instant::now();
        let points = generate_points(n, args.seed, &args.dist);
        println!(
            "Point generation: {:.1}ms",
            t_gen.elapsed().as_secs_f64() * 1000.0
        );

        let mut times = Vec::with_capacity(args.repeat);
        let mut cells = 0usize;
        let mut verts = 0usize;
        for iter in 0..args.repeat {
            if args.repeat > 1 {
                print!("  Iteration {}/{}... ", iter + 1, args.repeat);
                io::stdout().flush().unwrap();
            }
            let (ms, c, v) = bench_once(&points);
            if args.repeat > 1 {
                println!("{ms:.1}ms");
            }
            if iter > 0 || args.repeat == 1 {
                times.push(ms);
            }
            cells = c;
            verts = v;
        }
        let (min, mean) = stats(&times);
        println!("\ncompute_plane: min {min:.1}ms / mean {mean:.1}ms");
        println!(
            "  {} cells, {} vertices, {:.2}M cells/s (at min)",
            format_num(cells),
            format_num(verts),
            n as f64 / min / 1000.0
        );

        if args.validate {
            let diagram = compute_plane(&points, PlaneRect::unit()).unwrap();
            let report = validation::validate_plane(&diagram);
            println!(
                "  validate_plane: strictly_valid = {}",
                report.is_strictly_valid()
            );
            assert!(report.is_strictly_valid(), "validation failed: {report:#?}");
        }

        if args.voronoice {
            let mut vtimes = Vec::with_capacity(args.repeat);
            for iter in 0..args.repeat {
                let ms = bench_voronoice_once(&points);
                if iter > 0 || args.repeat == 1 {
                    vtimes.push(ms);
                }
            }
            let (vmin, vmean) = stats(&vtimes);
            println!("voronoice:     min {vmin:.1}ms / mean {vmean:.1}ms");
            println!("  speedup at min: {:.2}x", vmin / min);
        }
    }
}
