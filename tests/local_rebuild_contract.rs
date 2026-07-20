//! Local-rebuild valid-or-loud-error contract tests.
//!
//! The hot `compute` path never validates — that is the whole point — so these
//! tests pin the pipeline's *valid-or-loud-error* contract on the one distribution
//! (`mega`) that actually drives it, using the report API (which validates
//! internally for diagnostics) plus the plain path.
//!
//! Cross-bin behavior is exercised deterministically via `VORONOI_MESH_BIN_COUNT`
//! rather than multi-threading (whose defect set varies run-to-run). Sizes are kept
//! modest (100k) so the suite stays CI-friendly; the full seed/size/param/MT
//! sweep lives in `scripts/robustness_campaign.sh`.

mod support;
use support::env::with_env_vars;
use support::points::mega_points;

use voronoi_mesh::{compute, compute_with_report, ComputeReport, VoronoiConfig};

/// Count residuals that survived to the returned diagram (what the hot path
/// loud-fails on; the report path returns them as diagnostics instead).
fn surviving_residual(report: &ComputeReport) -> usize {
    report.residual_unpaired_edges.len()
}

const SEEDS: [u64; 3] = [1, 2, 3];
// 100k is the smallest mega that reliably produces contested clusters for rebuild
// to act on (incl. cross-bin) while keeping the always-run suite CI-friendly; the
// full size/seed/param/MT sweep lives in `scripts/robustness_campaign.sh`.
const N: usize = 100_000;

/// THE silent-invalid guard: with rebuild on, a returned mega diagram that leaves
/// NO surviving residual must be strictly valid. (A diagram that left a residual
/// is out-of-envelope on the report API and is allowed to be invalid there — the
/// plain path loud-fails on it; see the next test.)
#[test]
fn mega_rebuild_no_residual_implies_strictly_valid() {
    with_env_vars(&[], || {
        for seed in SEEDS {
            let pts = mega_points(N, 0.8, seed);
            let out = compute_with_report(&pts, VoronoiConfig::default())
                .expect("report path must not loud-fail");
            let residual = surviving_residual(&out.report);
            let valid = out.report.preferred_validation().is_strictly_valid();
            assert!(
                valid || residual > 0,
                "mega seed {seed}: returned diagram is NOT strictly valid yet left NO \
                 surviving residual — a silent invalid-output defect"
            );
        }
    });
}

/// The report's `PostReconciliationUnpaired` residual is exactly what the hot `compute`
/// path turns into a loud error: residual present <=> `compute` returns `Err`,
/// and whenever `compute` returns `Ok` the diagram is strictly valid.
#[test]
fn mega_plain_compute_errs_iff_residual_remains() {
    with_env_vars(&[], || {
        for seed in SEEDS {
            let pts = mega_points(N, 0.8, seed);
            let residual = surviving_residual(
                &compute_with_report(&pts, VoronoiConfig::default())
                    .expect("report path must not loud-fail")
                    .report,
            );
            match compute(&pts) {
                Ok(diagram) => {
                    assert_eq!(
                        residual, 0,
                        "mega seed {seed}: plain compute returned Ok but report kept \
                         {residual} residual(s)"
                    );
                    assert!(
                        voronoi_mesh::validation::validate(&diagram).is_strictly_valid(),
                        "mega seed {seed}: plain compute Ok but diagram is not strictly valid"
                    );
                }
                Err(_) => assert!(
                    residual > 0,
                    "mega seed {seed}: plain compute returned Err but report kept no residual"
                ),
            }
        }
    });
}

/// Cross-bin rebuild on a forced high bin count still yields valid-or-residual
/// (exercises the cross-bin path deterministically, single process).
#[test]
fn mega_rebuild_cross_bin_contract() {
    with_env_vars(&[("VORONOI_MESH_BIN_COUNT", Some("48"))], || {
        let pts = mega_points(N, 0.8, 2);
        let out = compute_with_report(&pts, VoronoiConfig::default())
            .expect("report path must not loud-fail");
        let residual = surviving_residual(&out.report);
        let valid = out.report.preferred_validation().is_strictly_valid();
        assert!(
            valid || residual > 0,
            "mega cross-bin: returned diagram invalid with no surviving residual — silent-invalid"
        );
    });
}

/// The rebuilt diagram is deterministic single-threaded: the interior-vid
/// assignment is sorted for exactly this, so the same input yields the same
/// vertex layout and cell boundaries run-to-run. `#[ignore]`'d for runtime (two
/// single-threaded builds); run manually / in scheduled CI.
#[test]
#[ignore = "two single-threaded builds — run manually to pin rebuild determinism"]
fn mega_rebuild_is_deterministic_single_threaded() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    with_env_vars(&[], || {
        let pts = mega_points(N, 0.8, 2);
        let layout = |pts: &[support::points::TestPoint]| {
            let out =
                pool.install(|| compute_with_report(pts, VoronoiConfig::default()).expect("build"));
            let verts: Vec<(u32, u32, u32)> = out
                .diagram
                .vertices()
                .iter()
                .map(|v| (v.x().to_bits(), v.y().to_bits(), v.z().to_bits()))
                .collect();
            let cells: Vec<Vec<u32>> = (0..out.diagram.num_cells())
                .map(|i| out.diagram.cell(i).vertex_indices.to_vec())
                .collect();
            (verts, cells)
        };
        let a = layout(&pts);
        let b = layout(&pts);
        assert_eq!(
            a.0, b.0,
            "rebuilt vertex layout differs across single-threaded runs"
        );
        assert_eq!(
            a.1, b.1,
            "rebuilt cell boundaries differ across single-threaded runs"
        );
    });
}
