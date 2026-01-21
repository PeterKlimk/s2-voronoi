//! Microbenchmark for the packed-kNN hot path (CubeMapGrid + packed_knn).
//!
//! Build/run:
//!   cargo run --release --features bench-internals --bin bench_packed_knn -- 500k -n 20
//!   cargo run --release --features bench-internals,timing --bin bench_packed_knn -- 500k -n 20

use clap::Parser;
use glam::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use s2_voronoi::cube_grid::packed_knn::{
    PackedKnnCellScratch, PackedKnnCellStatus, PackedKnnTimings, PackedStage,
};
use s2_voronoi::cube_grid::CubeMapGrid;
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

fn auto_res(n: usize) -> usize {
    // Heuristic: aim for ~100 points/cell.
    if n == 0 {
        return 1;
    }
    let cells = (n as f64 / 100.0).max(1.0);
    let per_face = (cells / 6.0).max(1.0);
    (per_face.sqrt().round() as usize).max(1)
}

fn local_shift_for_n(n: usize) -> u32 {
    if n <= 1 {
        return 1;
    }
    let next = (n as u32).next_power_of_two();
    next.trailing_zeros().max(1).min(31)
}

#[derive(Parser)]
#[command(name = "bench_packed_knn")]
#[command(about = "Microbenchmark CubeMapGrid packed-kNN group preparation + chunk emission")]
struct Args {
    /// Number of points to generate (e.g., 100k, 1m).
    #[arg(value_parser = parse_count, default_value = "100k")]
    n: usize,

    /// Cube grid resolution. If omitted, uses a heuristic.
    #[arg(long)]
    res: Option<usize>,

    /// Random seed.
    #[arg(long, default_value_t = 12345)]
    seed: u64,

    /// Iterations to run (useful for profiling).
    #[arg(short = 'n', long, default_value_t = 10)]
    repeat: usize,

    /// Number of cell-groups to sample (densest cells first).
    #[arg(long, default_value_t = 64)]
    groups: usize,

    /// Max queries per group (points in the chosen cell).
    #[arg(long, default_value_t = 8)]
    queries_per_group: usize,

    /// First chunk size (like packed_k0_base).
    #[arg(long, default_value_t = 32)]
    k0: usize,

    /// Subsequent chunk size (like packed_k1).
    #[arg(long, default_value_t = 8)]
    k1: usize,

    /// Exhaust all candidates (otherwise consume one chunk0 + optional one tail chunk).
    #[arg(long)]
    exhaust: bool,

    /// Use key-prefix bucketing with this many bits (0 = disabled).
    #[arg(long, default_value_t = 0)]
    bucket_bits: u8,
}

#[cfg(feature = "timing")]
fn add_timings(dst: &mut PackedKnnTimings, src: &PackedKnnTimings) {
    dst.setup += src.setup;
    dst.query_cache += src.query_cache;
    dst.security_thresholds += src.security_thresholds;
    dst.center_pass += src.center_pass;
    dst.ring_thresholds += src.ring_thresholds;
    dst.ring_pass += src.ring_pass;
    dst.ring_fallback += src.ring_fallback;
    dst.select_prep += src.select_prep;
    dst.select_query_prep += src.select_query_prep;
    dst.select_partition += src.select_partition;
    dst.select_sort += src.select_sort;
    dst.select_scatter += src.select_scatter;
    dst.tail_builds += src.tail_builds;
}

#[cfg(not(feature = "timing"))]
fn add_timings(_dst: &mut PackedKnnTimings, _src: &PackedKnnTimings) {}

fn main() {
    let args = Args::parse();
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);

    let jitter = mean_spacing(args.n) * 0.15;
    let points = fibonacci_sphere_points_with_rng(args.n, jitter, &mut rng);

    let res = args.res.unwrap_or_else(|| auto_res(args.n));
    println!(
        "points={} res={} groups={} queries/group={} repeat={} exhaust={} (jitter={:.4})",
        args.n, res, args.groups, args.queries_per_group, args.repeat, args.exhaust, jitter
    );

    let grid = CubeMapGrid::new(&points, res);
    let num_cells = 6 * grid.res() * grid.res();

    // Pick densest cells as "groups".
    let mut cells: Vec<(usize, usize)> = (0..num_cells)
        .map(|cell| {
            let start = grid.cell_offsets()[cell] as usize;
            let end = grid.cell_offsets()[cell + 1] as usize;
            (end - start, cell)
        })
        .collect();
    cells.sort_by(|a, b| b.0.cmp(&a.0));
    let selected: Vec<usize> = cells
        .into_iter()
        .filter(|(len, _)| *len >= 2)
        .take(args.groups.max(1))
        .map(|(_, cell)| cell)
        .collect();

    if selected.is_empty() {
        eprintln!("no non-empty cells found");
        return;
    }

    let local_shift = local_shift_for_n(points.len());
    let local_mask = (1u32 << local_shift) - 1;
    let query_bin = 0u8;
    let slot_gen_map: Vec<u32> = (0..points.len() as u32).collect();

    let mut scratch = PackedKnnCellScratch::new();
    if args.bucket_bits > 0 {
        scratch.set_bucket_bits(args.bucket_bits);
    }
    let mut timings = PackedKnnTimings::default();
    let mut total_timings = PackedKnnTimings::default();

    let mut total_groups_ok = 0usize;
    let mut total_groups_slow = 0usize;
    let mut total_queries = 0usize;
    let mut total_chunks = 0usize;

    let t_all = Instant::now();
    for _ in 0..args.repeat.max(1) {
        for &cell in &selected {
            let start = grid.cell_offsets()[cell] as usize;
            let end = grid.cell_offsets()[cell + 1] as usize;
            let len = end - start;
            if len == 0 {
                continue;
            }

            let q_end = (start + args.queries_per_group).min(end);
            let queries: Vec<u32> = (start..q_end).map(|s| s as u32).collect();
            let query_locals: Vec<u32> = queries.iter().map(|&s| s & local_mask).collect();

            timings.clear();
            let status = scratch.prepare_group_directed(
                &grid,
                cell,
                &queries,
                &query_locals,
                query_bin,
                &slot_gen_map,
                local_shift,
                local_mask,
                &mut timings,
            );

            match status {
                PackedKnnCellStatus::Ok => total_groups_ok += 1,
                PackedKnnCellStatus::SlowPath => {
                    total_groups_slow += 1;
                    continue;
                }
            }

            add_timings(&mut total_timings, &timings);

            let k0 = args.k0.max(1);
            let k1 = args.k1.max(1);
            let mut out = vec![u32::MAX; k0.max(k1)];

            for qi in 0..queries.len() {
                total_queries += 1;
                let mut stage = PackedStage::Chunk0;
                let mut k_cur = k0;

                loop {
                    out.fill(u32::MAX);
                    timings.clear();
                    let Some(chunk) = scratch.next_chunk(qi, stage, k_cur, &mut out, &mut timings)
                    else {
                        if stage == PackedStage::Chunk0 && scratch.tail_possible(qi) {
                            scratch.ensure_tail_directed_for(
                                qi,
                                &grid,
                                &slot_gen_map,
                                local_shift,
                                local_mask,
                                &mut timings,
                            );
                            stage = PackedStage::Tail;
                            k_cur = k1;
                            continue;
                        }
                        break;
                    };
                    total_chunks += 1;
                    add_timings(&mut total_timings, &timings);

                    if !args.exhaust {
                        // Simulate a typical "seed" consumption: one chunk0 + optional one tail chunk.
                        if stage == PackedStage::Chunk0 {
                            if scratch.tail_possible(qi) {
                                scratch.ensure_tail_directed_for(
                                    qi,
                                    &grid,
                                    &slot_gen_map,
                                    local_shift,
                                    local_mask,
                                    &mut timings,
                                );
                                stage = PackedStage::Tail;
                                k_cur = k1;
                                continue;
                            }
                        }
                        let _ = chunk;
                        break;
                    }

                    // Exhaust mode: keep going in the current stage.
                    k_cur = k1;
                }
            }
        }
    }
    let ms = t_all.elapsed().as_secs_f64() * 1000.0;

    println!(
        "time={:.2}ms groups_ok={} groups_slow={} queries={} chunks={}",
        ms, total_groups_ok, total_groups_slow, total_queries, total_chunks
    );

    #[cfg(feature = "timing")]
    {
        let groups_ok = total_groups_ok.max(1) as f64;
        let queries = total_queries.max(1) as f64;
        println!(
            "timing avg/group: setup={:.3}ms ring_pass={:.3}ms select={:.3}ms tail_builds={}",
            total_timings.setup.as_secs_f64() * 1000.0 / groups_ok,
            total_timings.ring_pass.as_secs_f64() * 1000.0 / groups_ok,
            (total_timings.select_query_prep
                + total_timings.select_partition
                + total_timings.select_sort
                + total_timings.select_scatter)
                .as_secs_f64()
                * 1000.0
                / queries,
            total_timings.tail_builds
        );
        println!(
            "timing totals: setup={:.2}ms query_cache={:.2}ms security={:.2}ms center={:.2}ms ring_thresholds={:.2}ms ring_pass={:.2}ms select_partition={:.2}ms select_sort={:.2}ms select_scatter={:.2}ms",
            total_timings.setup.as_secs_f64() * 1000.0,
            total_timings.query_cache.as_secs_f64() * 1000.0,
            total_timings.security_thresholds.as_secs_f64() * 1000.0,
            total_timings.center_pass.as_secs_f64() * 1000.0,
            total_timings.ring_thresholds.as_secs_f64() * 1000.0,
            total_timings.ring_pass.as_secs_f64() * 1000.0,
            total_timings.select_partition.as_secs_f64() * 1000.0,
            total_timings.select_sort.as_secs_f64() * 1000.0,
            total_timings.select_scatter.as_secs_f64() * 1000.0,
        );
    }
}
