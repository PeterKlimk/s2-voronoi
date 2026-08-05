//! Production-path regression for the local exact defect rebuild. The cell
//! fallback may resolve historical mega defects upstream; otherwise this proves
//! the Hull3d oracle reaches strict validity.
//!
//!   cargo test --release --test local_rebuild

mod support;

use support::points::*;
use voronoi_mesh::{compute, compute_with_report, LocalRebuildMode, VoronoiConfig};

#[test]
fn local_rebuild_makes_mega_strictly_valid() {
    let off = || VoronoiConfig::default().with_local_rebuild_mode(LocalRebuildMode::Disabled);
    let on = VoronoiConfig::default;
    for seed in [1u64, 2, 15] {
        let points = mega_points(100_000, 0.8, seed);

        let before = compute_with_report(&points, off()).expect("build");
        let before_valid = before.report.validation.is_strictly_valid();

        let after = compute_with_report(&points, on()).expect("build");
        let after_report = &after.report.validation;

        println!(
            "mega 100k s{seed}: off={} on={}",
            if before_valid { "VALID" } else { "INVALID" },
            if after_report.is_strictly_valid() {
                "VALID".to_string()
            } else {
                format!("{:?}", after_report.subdivision_issues())
            }
        );
        // The fallback path can now resolve some formerly rebuild-only defects
        // upstream. Whether rebuild runs or not, the meaningful contract is a
        // strictly valid final diagram.
        assert!(
            after_report.is_strictly_valid(),
            "seed {seed}: local rebuild did not produce a strictly valid diagram: {:?}",
            after_report.subdivision_issues()
        );
        assert_eq!(
            after_report.cells_with_fewer_than_three_stored_positions, 0,
            "seed {seed}: final Hull3d/fallback output lost a stored cell"
        );
    }
}

#[test]
fn default_compute_rebuilds_known_mega_defects() {
    for seed in [1u64, 2, 15] {
        let points = mega_points(100_000, 0.8, seed);
        let diagram = compute(&points)
            .unwrap_or_else(|e| panic!("mega 100k s{seed}: default compute failed: {e:?}"));
        let report = voronoi_mesh::validation::validate(&diagram);
        assert!(
            report.is_strictly_valid(),
            "mega 100k s{seed}: default compute returned invalid diagram: {}",
            report.headline()
        );
    }
}

#[test]
fn accepted_default_rebuild_clears_surviving_residual_report() {
    for seed in [1u64, 2, 15] {
        let points = mega_points(100_000, 0.8, seed);
        let out = compute_with_report(&points, VoronoiConfig::default())
            .unwrap_or_else(|e| panic!("mega 100k s{seed}: report build failed: {e:?}"));
        assert!(
            out.report.validation.is_strictly_valid(),
            "mega 100k s{seed}: default rebuild was not accepted"
        );
        let post = out.report.residual_unpaired_edges.len();
        assert_eq!(
            post, 0,
            "mega 100k s{seed}: accepted rebuild left surviving residual records"
        );
    }
}

/// Broader Hull3d regression sweep over historically defective inputs. Ignored
/// because the larger sizes take minutes; run with `--ignored --nocapture`.
#[test]
#[ignore = "broad local rebuilding sweep; run with --ignored --nocapture"]
fn local_rebuild_broad_sweep() {
    let off = || VoronoiConfig::default().with_local_rebuild_mode(LocalRebuildMode::Disabled);
    let on = VoronoiConfig::default;
    let mut cases: Vec<(String, Vec<_>)> = Vec::new();
    for seed in 1u64..=20 {
        cases.push((
            format!("mega 100k s{seed}"),
            mega_points(100_000, 0.8, seed),
        ));
    }
    for seed in 1u64..=3 {
        cases.push((
            format!("mega 300k s{seed}"),
            mega_points(300_000, 0.8, seed),
        ));
        cases.push((
            format!("mega 500k s{seed}"),
            mega_points(500_000, 0.8, seed),
        ));
    }
    cases.push(("mega 1m s1".into(), mega_points(1_000_000, 0.8, 1)));
    for seed in 1u64..=3 {
        cases.push((
            format!("clustered 200k s{seed}"),
            clustered_cap_points(200_000, 0.15, seed),
        ));
        cases.push((
            format!("bimodal 200k s{seed}"),
            bimodal_density_points(200_000, 0.1, seed),
        ));
    }

    let mut defects = 0usize;
    for (name, points) in &cases {
        let before = compute_with_report(points, off()).expect("build");
        let before_valid = before.report.validation.is_strictly_valid();

        let after = compute_with_report(points, on()).expect("build");
        let after_valid = after.report.validation.is_strictly_valid();

        if !before_valid {
            defects += 1;
        }
        println!(
            "{name}: off={} on={}",
            if before_valid { "VALID" } else { "INVALID" },
            if after_valid {
                "VALID".into()
            } else {
                format!("{:?}", after.report.validation.subdivision_issues())
            }
        );
        assert!(
            after_valid,
            "{name}: local rebuild did not reach strict validity"
        );
    }
    println!("defective inputs rebuilt: {defects}/{}", cases.len());
    assert!(defects > 0, "expected some defective inputs in the sweep");
}
