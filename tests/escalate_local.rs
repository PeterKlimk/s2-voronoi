//! Production-path regression for the dependency-free, local exact defect
//! repair (the escalation engine). Unlike `escalate.rs`, this file is NOT gated
//! on `escalate_probe` and pulls in NO `delaunator` crate — it exercises the
//! exact path a default build ships, proving the local oracle reaches strict
//! validity on the mega near-cocircular defects with no external dependency.
//!
//!   cargo test --release --test escalate_local

mod support;

use s2_voronoi::{compute_with_report, set_escalation_enabled, VoronoiConfig};
use support::points::*;

#[test]
fn local_escalation_makes_mega_strictly_valid() {
    let cfg = VoronoiConfig::default;
    let mut fixed_at_least_one = false;
    for seed in [1u64, 2, 15] {
        let points = mega_points(100_000, 0.8, seed);

        set_escalation_enabled(false);
        let before = compute_with_report(&points, cfg()).expect("build");
        let before_valid = before.report.returned_validation.is_strictly_valid();

        set_escalation_enabled(true);
        let after = compute_with_report(&points, cfg()).expect("build");
        set_escalation_enabled(false);
        let after_report = &after.report.returned_validation;

        println!(
            "mega 100k s{seed}: off={} on={}",
            if before_valid { "VALID" } else { "INVALID" },
            if after_report.is_strictly_valid() {
                "VALID".to_string()
            } else {
                format!("{:?}", after_report.subdivision_issues())
            }
        );
        // The valid-or-revert gate guarantees output is never worse than the fast
        // path; the meaningful bar is that the local repair actually reaches
        // strict validity on these known defects.
        assert!(
            after_report.is_strictly_valid(),
            "seed {seed}: local repair did not produce a strictly valid diagram: {:?}",
            after_report.subdivision_issues()
        );
        fixed_at_least_one |= !before_valid;
    }
    assert!(
        fixed_at_least_one,
        "expected at least one mega seed to be invalid without the repair"
    );
}

/// Broader parity sweep against the delaunator baseline: every defective input
/// the global oracle resolved must also resolve with the local engine. Ignored
/// (minutes at the larger sizes); run with `--ignored --nocapture`.
#[test]
#[ignore = "broad escalation sweep; run with --ignored --nocapture"]
fn local_escalation_broad_sweep() {
    let cfg = VoronoiConfig::default;
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
        set_escalation_enabled(false);
        let before = compute_with_report(points, cfg()).expect("build");
        let before_valid = before.report.returned_validation.is_strictly_valid();

        set_escalation_enabled(true);
        let after = compute_with_report(points, cfg()).expect("build");
        set_escalation_enabled(false);
        let after_valid = after.report.returned_validation.is_strictly_valid();

        if !before_valid {
            defects += 1;
        }
        println!(
            "{name}: off={} on={}",
            if before_valid { "VALID" } else { "INVALID" },
            if after_valid {
                "VALID".into()
            } else {
                format!(
                    "{:?}",
                    after.report.returned_validation.subdivision_issues()
                )
            }
        );
        assert!(
            after_valid,
            "{name}: local repair did not reach strict validity"
        );
    }
    println!("defective inputs repaired: {defects}/{}", cases.len());
    assert!(defects > 0, "expected some defective inputs in the sweep");
}
