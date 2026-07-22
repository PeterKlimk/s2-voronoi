//! Cold positive-simplification timing probe.

use std::time::Instant;

use voronoi_mesh::{
    compute_simplified_with, compute_with_report, CellSimplificationOptions,
    SimplificationCellPolicy, VoronoiConfig,
};

fn fibonacci_points(count: usize) -> Vec<[f32; 3]> {
    let golden_angle = core::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    (0..count)
        .map(|index| {
            let z = 1.0 - 2.0 * (index as f64 + 0.5) / count as f64;
            let radial = (1.0 - z * z).sqrt();
            let theta = index as f64 * golden_angle;
            [
                (radial * theta.cos()) as f32,
                (radial * theta.sin()) as f32,
                z as f32,
            ]
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let count = args
        .next()
        .as_deref()
        .unwrap_or("100000")
        .parse::<usize>()
        .expect("point count must be an integer");
    let threshold = args
        .next()
        .as_deref()
        .unwrap_or("1e-10")
        .parse::<f32>()
        .expect("threshold must be an f32");
    let rounds = args
        .next()
        .as_deref()
        .unwrap_or("10")
        .parse::<usize>()
        .expect("round count must be an integer");
    assert!(rounds > 0, "round count must be positive");
    let mode = args.next().unwrap_or_else(|| "batched".into());
    let policy = match mode.as_str() {
        "batched" => SimplificationCellPolicy::Preserve,
        "preserve" => SimplificationCellPolicy::Preserve,
        "error" => SimplificationCellPolicy::Error,
        "elide" => SimplificationCellPolicy::Elide,
        value => panic!("unknown mode {value:?}; expected batched, preserve, error, or elide"),
    };

    let points = fibonacci_points(count);
    let output = (mode != "batched").then(|| {
        compute_with_report(&points, VoronoiConfig::default())
            .expect("benchmark source computation failed")
    });
    let options = CellSimplificationOptions::from_chord_length(threshold)
        .expect("invalid threshold")
        .with_cell_policy(policy);
    let mut elapsed = Vec::with_capacity(rounds);
    let mut last_report = None;
    for _ in 0..rounds {
        let start = Instant::now();
        let result = if mode == "batched" {
            compute_simplified_with(&points, VoronoiConfig::default(), options)
                .map_err(|error| error.to_string())
        } else {
            output
                .as_ref()
                .unwrap()
                .clone()
                .into_simplified_cell_mesh(options)
                .map_err(|error| format!("{:?}: {}", error.kind(), error))
        };
        match result {
            Ok(simplified) => {
                elapsed.push(start.elapsed().as_secs_f64() * 1_000.0);
                last_report = Some(simplified.simplification_report);
            }
            Err(error) => {
                let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
                eprintln!("simplification failed after {elapsed_ms:.3}ms: {error}");
                std::process::exit(1);
            }
        }
    }
    elapsed.sort_by(f64::total_cmp);
    let median = elapsed[elapsed.len() / 2];
    let report = last_report.unwrap();
    println!(
        "points={count} threshold={threshold:e} mode={mode} rounds={rounds} median_ms={median:.3} hinted_cells={} candidates={} attempts={} accepted={} newly_exposed={} remaining={} cell_visits={} pair_checks={} provenance_checks={}",
        report.hinted_candidate_cells,
        report.positive_candidate_occurrences,
        report.attempted_contractions,
        report.accepted_contractions,
        report.newly_exposed_positive_edges,
        report.remaining_positive_edges,
        report.work.cell_index_visits,
        report.work.diameter_pair_comparisons,
        report.work.provenance_member_checks,
    );
}
