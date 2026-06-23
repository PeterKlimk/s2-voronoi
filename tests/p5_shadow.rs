//! P5 stage-1 shadow-audit probes (feature `p5_shadow`): run representative
//! inputs and print the margin/disagreement report. Diagnostic, not a
//! contract test — run with
//! `cargo test --release --features p5_shadow --test p5_shadow -- --ignored --nocapture`
#![cfg(feature = "p5_shadow")]

mod support;

use s2_voronoi::{compute_with_report, UnitVec3Like, VoronoiConfig};
use support::points::*;

/// The defect site from uniform 2M seed 1 (see tests/edge_repair_net.rs).
const SITE_CENTER: (f32, f32, f32) = (-0.419_580_3, -0.220_115_17, -0.880_625_7);

fn defect_fixture() -> Vec<s2_voronoi::UnitVec3> {
    let mean_spacing = (4.0 * std::f32::consts::PI / 2_000_000.0).sqrt();
    let cos_window = (10.0 * mean_spacing).cos();
    let cos_excl = (15.0 * mean_spacing).cos();
    let c = SITE_CENTER;
    let mut fixture: Vec<s2_voronoi::UnitVec3> = random_sphere_points(2_000_000, 1)
        .into_iter()
        .filter(|p| p.x() * c.0 + p.y() * c.1 + p.z() * c.2 >= cos_window)
        .collect();
    fixture.extend(
        random_sphere_points(2_000, 4242)
            .into_iter()
            .filter(|p| p.x() * c.0 + p.y() * c.1 + p.z() * c.2 < cos_excl),
    );
    fixture
}

fn run_case(name: &str, points: &[s2_voronoi::UnitVec3]) {
    s2_voronoi::p5_shadow::reset();
    let out = compute_with_report(points, VoronoiConfig::default()).expect(name);
    println!(
        "=== {name}: n={} defects={} ===",
        points.len(),
        out.report.unresolved_edge_pairs.len()
    );
    print!("{}", s2_voronoi::p5_shadow::report());
}

#[test]
#[ignore]
fn probe_shadow_audit() {
    run_case("defect_fixture", &defect_fixture());
    run_case("uniform_100k_s2", &random_sphere_points(100_000, 2));
    run_case("uniform_2m_s1", &random_sphere_points(2_000_000, 1));
}

/// GO/NO-GO for the adaptive canonical clip: the SUPERSET-BAND
/// measurement. Raise the exact-eval cutoff so EVERY keep/drop decision is
/// compared against the exact in-circle predicate, then read the smallest
/// margin BAND that contains all disagreements + ties (the band the cheap SIMD
/// prefilter would need) and its trip rate, across all regimes. A BAND with
/// the superset property at <1% trip => the cheap prefilter stands.
///
/// Run single-threaded for stable counts:
///   RAYON_NUM_THREADS=1 cargo test --release --features p5_shadow \
///     --test p5_shadow -- --ignored probe_superset_band --nocapture
#[test]
#[ignore]
fn probe_superset_band() {
    let run = |name: &str, points: &[s2_voronoi::UnitVec3]| {
        s2_voronoi::p5_shadow::reset();
        // Evaluate the exact predicate at every margin (not just < 1e-4), so a
        // large-margin disagreement cannot be missed.
        s2_voronoi::p5_shadow::set_audit_cutoff(Some(10.0));
        let out = compute_with_report(points, VoronoiConfig::default()).expect(name);
        s2_voronoi::p5_shadow::set_audit_cutoff(None);
        println!(
            "=== {name}: n={} defects={} ===",
            points.len(),
            out.report.unresolved_edge_pairs.len()
        );
        print!("{}", s2_voronoi::p5_shadow::report());
    };

    run("uniform_200k_s1", &random_sphere_points(200_000, 1));
    run("mega_200k_s3", &mega_points(200_000, 0.8, 3));
    run(
        "clustered_200k_r0.05",
        &clustered_cap_points(200_000, 0.05, 1),
    );
    run(
        "bimodal_200k_r0.1",
        &bimodal_density_points(200_000, 0.1, 1),
    );
    run("grid_cubed_~150k", &cubed_sphere_points(150_000, 0));
    run(
        "cube_stress_100k_s0.02",
        &cube_vertex_stress_points(100_000, 0.02, 1),
    );
    run(
        "cocircular_20k_groups",
        &near_cocircular_stress_points(20_000, 1e-4, 1),
    );
    run(
        "fibonacci_200k_j0.5",
        &fibonacci_sphere_points(200_000, 0.5, 1),
    );
}

/// The TRUE GO/NO-GO band: the CROSS-CELL conflict tail. Single-cell
/// chart-vs-exact disagreement (probe_superset_band) is a pessimistic proxy —
/// most such disagreements are benign because both incident cells drift the
/// same way and still agree with each other. A defect needs two cells to answer
/// the SAME question differently. `paired_quad_report` groups decisions by the
/// sorted 4-point question and reports cross-cell CONTRADICTIONS and their max
/// margin — the band the per-cell flag must actually cover.
///
///   RAYON_NUM_THREADS=1 cargo test --release --features p5_shadow \
///     --test p5_shadow -- --ignored probe_superset_paired --nocapture
#[test]
#[ignore]
fn probe_superset_paired() {
    let run = |name: &str, points: &[s2_voronoi::UnitVec3], cutoff: f64| {
        s2_voronoi::p5_shadow::reset();
        s2_voronoi::p5_shadow::paired_reset();
        s2_voronoi::p5_shadow::set_pair_cutoff(cutoff);
        let out = compute_with_report(points, VoronoiConfig::default()).expect(name);
        s2_voronoi::p5_shadow::set_pair_cutoff(0.0);
        println!(
            "=== {name}: n={} cutoff={cutoff:.0e} defects={} ===",
            points.len(),
            out.report.unresolved_edge_pairs.len()
        );
        print!("{}", s2_voronoi::p5_shadow::paired_quad_report());
    };

    // Clean baseline: expect ~0 cross-cell contradictions despite many
    // single-cell disagreements.
    run("uniform_200k_s1", &random_sphere_points(200_000, 1), 1e-4);
    // Defect-bearing: the contradiction margins ARE the band.
    run("mega_100k_s3", &mega_points(100_000, 0.8, 3), 1e-4);
    run(
        "clustered_100k_r0.05",
        &clustered_cap_points(100_000, 0.05, 1),
        1e-4,
    );
    run("grid_cubed_~150k", &cubed_sphere_points(150_000, 0), 1e-3);
}

/// Quick cost proxy for exact-clip-everywhere: the dominant overhead is one
/// adaptive in-circle (`orient3d`) per keep/drop decision instead of SIMD
/// `signed_dist`. Isolate it by timing the SAME build with the per-decision
/// exact predicate evaluated for EVERY vertex (`audit_cutoff=10`) vs NOT
/// (`audit_cutoff=0`) — both keep identical audit bookkeeping (the per-vertex
/// signed_dist + margin atomic), so the DELTA is purely the predicate cost.
/// UPPER BOUND: `in_circle_sphere_sign` does 2 `orient3d`, d2 recomputed each
/// call; a real exact clip caches d2 (~1 orient3d/decision) → halve the delta.
/// Also excludes the (smaller) circumcenter-vs-lerp vertex cost.
///   RAYON_NUM_THREADS=1 cargo test --release --features p5_shadow \
///     --test p5_shadow -- --ignored probe_predicate_cost --nocapture
#[test]
#[ignore]
fn probe_predicate_cost() {
    use std::time::Instant;
    let time_one = |points: &[s2_voronoi::UnitVec3], cutoff: f64| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            s2_voronoi::p5_shadow::reset();
            s2_voronoi::p5_shadow::set_audit_cutoff(Some(cutoff));
            let t = Instant::now();
            let out = compute_with_report(points, VoronoiConfig::default()).expect("build");
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            best = best.min(dt);
        }
        s2_voronoi::p5_shadow::set_audit_cutoff(None);
        best
    };
    let run = |name: &str, points: &[s2_voronoi::UnitVec3]| {
        // cutoff=0 => predicate never evaluated (bookkeeping only); cutoff=10 =>
        // evaluated for every decision.
        let base = time_one(points, 0.0);
        let full = time_one(points, 10.0);
        println!(
            "{name}: n={} bookkeeping_only={base:.3}s  +per-decision-incircle={full:.3}s  \
             delta={:.3}s ({:.1}% ; real~{:.1}% after d2 caching)",
            points.len(),
            full - base,
            100.0 * (full - base) / base,
            50.0 * (full - base) / base,
        );
    };
    run("uniform_200k_s1", &random_sphere_points(200_000, 1));
    run("mega_100k_s3", &mega_points(100_000, 0.8, 3));
}

/// Does mega have SAME-QUESTION cross-cell sign conflicts at a wider margin, or
/// is its divergence purely question-set (different fourth generators, which
/// the same-4-set report cannot see)? Sweep the collection cutoff on one mega
/// input. If contradictions stay 0 as the cutoff widens, mega's defects are
/// question-set divergence, not sign conflicts.
///   RAYON_NUM_THREADS=1 cargo test --release --features p5_shadow \
///     --test p5_shadow -- --ignored probe_mega_cutoff_sweep --nocapture
#[test]
#[ignore]
fn probe_mega_cutoff_sweep() {
    let pts = mega_points(100_000, 0.8, 3);
    for cutoff in [1e-4, 1e-3, 1e-2] {
        s2_voronoi::p5_shadow::reset();
        s2_voronoi::p5_shadow::paired_reset();
        s2_voronoi::p5_shadow::set_pair_cutoff(cutoff);
        let out = compute_with_report(&pts, VoronoiConfig::default()).expect("mega");
        s2_voronoi::p5_shadow::set_pair_cutoff(0.0);
        println!(
            "=== mega_100k_s3 cutoff={cutoff:.0e} defects={} ===",
            out.report.unresolved_edge_pairs.len()
        );
        print!("{}", s2_voronoi::p5_shadow::paired_quad_report());
    }
}

/// Paired two-cell audit (P5 stage-2 prerequisite): group near-margin
/// decisions by their abstract question — (sorted triple, opposing
/// generator) — and measure how often distinct cells answer the SAME
/// question with conflicting local signs, and at what margins. The conflict
/// tail is what EPS_FILTER must dominate. The 3M seed-3 input carries real
/// defects, so its conflicts are ground truth for the audit itself.
#[test]
#[ignore]
fn probe_paired_audit() {
    let run = |name: &str, points: &[s2_voronoi::UnitVec3], cutoff: f64| {
        s2_voronoi::p5_shadow::reset();
        s2_voronoi::p5_shadow::paired_reset();
        s2_voronoi::p5_shadow::set_pair_cutoff(cutoff);
        let out = compute_with_report(points, VoronoiConfig::default()).expect(name);
        s2_voronoi::p5_shadow::set_pair_cutoff(0.0);
        println!(
            "=== {name}: n={} cutoff={cutoff:.0e} defects={} ===",
            points.len(),
            out.report.unresolved_edge_pairs.len()
        );
        print!("{}", s2_voronoi::p5_shadow::paired_report());
    };

    run("uniform_500k_s2", &random_sphere_points(500_000, 2), 1e-3);
    run("uniform_2m_s1", &random_sphere_points(2_000_000, 1), 1e-4);
    run("uniform_3m_s3", &random_sphere_points(3_000_000, 3), 1e-4);
}

/// Defect-site anatomy: every paired record touching the 3M-seed-3 defect
/// cluster, to see HOW cross-cell divergence manifests when paired answers
/// never conflict (expected: divergent question sets / triple identities,
/// i.e. failure mode B, not conflicting answers).
#[test]
#[ignore]
fn probe_defect_anatomy() {
    let points = random_sphere_points(3_000_000, 3);
    s2_voronoi::p5_shadow::paired_reset();
    s2_voronoi::p5_shadow::set_pair_cutoff(1e-3);
    let out = compute_with_report(&points, VoronoiConfig::default()).expect("3m");
    s2_voronoi::p5_shadow::set_pair_cutoff(0.0);
    println!("defects: {:?}", out.report.unresolved_edge_pairs);
    let site = [1790353u32, 2327897, 2902347, 2992988];
    print!("{}", s2_voronoi::p5_shadow::paired_dump_involving(&site));
}

/// Two-pass de-censored audit: pass 1 collects question keys below a margin
/// cutoff; pass 2 re-runs the same input recording EVERY party's answer to
/// those questions at ANY margin. Splits pass-1 single-party questions into
/// "partner answered above the cutoff" (censoring — check for conflicts!)
/// vs "partner never posed it" (true question-set divergence), making the
/// zero-conflict claim rigorous or refuting it.
#[test]
#[ignore]
fn probe_two_pass_audit() {
    use std::collections::HashMap;

    let run_two_pass = |name: &str, points: &[s2_voronoi::UnitVec3], cutoff: f64| {
        // Pass 1: cutoff-based collection.
        s2_voronoi::p5_shadow::paired_reset();
        s2_voronoi::p5_shadow::set_pair_cutoff(cutoff);
        s2_voronoi::p5_shadow::set_pair_key_filter(None);
        let out = compute_with_report(points, VoronoiConfig::default()).expect(name);
        s2_voronoi::p5_shadow::set_pair_cutoff(0.0);
        let pass1 = s2_voronoi::p5_shadow::paired_question_summaries();
        let keys: Vec<[u32; 4]> = pass1.iter().map(|&(k, ..)| k).collect();

        // Pass 2: same input, key-filtered, no margin censoring.
        s2_voronoi::p5_shadow::paired_reset();
        s2_voronoi::p5_shadow::set_pair_key_filter(Some(keys));
        let _ = compute_with_report(points, VoronoiConfig::default()).expect(name);
        s2_voronoi::p5_shadow::set_pair_key_filter(None);
        let pass2 = s2_voronoi::p5_shadow::paired_question_summaries();
        let pass2_by_key: HashMap<[u32; 4], (u32, bool, f32)> = pass2
            .iter()
            .map(|&(k, c, conf, m)| (k, (c, conf, m)))
            .collect();

        let mut p1_single = 0u64;
        let mut censored_pairs = 0u64; // single in pass 1, multi in pass 2
        let mut true_one_sided = 0u64; // single in both passes
        let mut censored_conflicts = 0u64; // THE number
        let mut conflict_min_margins: Vec<f32> = Vec::new();
        for &(k, cells1, _conf1, _m1) in &pass1 {
            let (cells2, conf2, m2) = pass2_by_key[&k];
            if cells1 < 2 {
                p1_single += 1;
                if cells2 >= 2 {
                    censored_pairs += 1;
                    if conf2 {
                        censored_conflicts += 1;
                        conflict_min_margins.push(m2);
                    }
                } else {
                    true_one_sided += 1;
                }
            } else if conf2 {
                // Was multi-party in pass 1 already; pass 2 may add parties.
                censored_conflicts += 1;
                conflict_min_margins.push(m2);
            }
        }
        println!(
            "=== {name}: n={} cutoff={cutoff:.0e} defects={} ===",
            points.len(),
            out.report.unresolved_edge_pairs.len()
        );
        println!(
            "  pass1 questions={} single_party={p1_single}; pass2: censored_pairs={censored_pairs} \
             true_one_sided={true_one_sided} CONFLICTS={censored_conflicts}",
            pass1.len()
        );
        if !conflict_min_margins.is_empty() {
            conflict_min_margins.sort_by(f32::total_cmp);
            println!("  conflict min-margins: {conflict_min_margins:?}");
        }
    };

    run_two_pass("uniform_500k_s2", &random_sphere_points(500_000, 2), 1e-3);
    run_two_pass("uniform_2m_s1", &random_sphere_points(2_000_000, 1), 1e-4);
    run_two_pass("uniform_3m_s3", &random_sphere_points(3_000_000, 3), 1e-4);

    // Uncensored defect-site anatomy for the 3M site (filter still warm
    // from the last pass-2 run would be cleaner, but rerun explicitly).
    let points = random_sphere_points(3_000_000, 3);
    s2_voronoi::p5_shadow::paired_reset();
    s2_voronoi::p5_shadow::set_pair_cutoff(1e-3);
    let _ = compute_with_report(&points, VoronoiConfig::default()).expect("3m");
    s2_voronoi::p5_shadow::set_pair_cutoff(0.0);
    let site = [1790353u32, 2327897, 2902347, 2992988];
    let keys: Vec<[u32; 4]> = s2_voronoi::p5_shadow::paired_question_summaries()
        .into_iter()
        .filter(|(k, ..)| k.iter().any(|id| site.contains(id)))
        .map(|(k, ..)| k)
        .collect();
    s2_voronoi::p5_shadow::paired_reset();
    s2_voronoi::p5_shadow::set_pair_key_filter(Some(keys));
    let _ = compute_with_report(&points, VoronoiConfig::default()).expect("3m");
    s2_voronoi::p5_shadow::set_pair_key_filter(None);
    println!("=== uncensored defect-site anatomy (3m s3) ===");
    print!("{}", s2_voronoi::p5_shadow::paired_dump_involving(&site));
}

/// Gate-1 closure experiment: sweep the termination certificate's angle pad
/// (EPS_CERT candidates) over every known defect-bearing input and measure
/// whether wider delivery kills the defects, and what it costs in wall
/// time. The decisive stage-2 question: is certificate conservatism alone
/// sufficient, and at what pad?
#[test]
#[ignore]
fn probe_eps_cert_sweep() {
    let fixture_2k = defect_fixture();
    let u3m = random_sphere_points(3_000_000, 3);
    let u45m = random_sphere_points(4_500_000, 2);

    for pad in [None, Some(1e-6f64), Some(1e-5), Some(1e-4)] {
        s2_voronoi::p5_shadow::set_term_pad_override(pad);
        println!("=== pad={pad:?} ===");
        for (name, points) in [
            ("fixture_2k", &fixture_2k),
            ("uniform_3m_s3", &u3m),
            ("uniform_4500k_s2", &u45m),
        ] {
            let t = std::time::Instant::now();
            let out = compute_with_report(points, VoronoiConfig::default()).expect(name);
            let dt = t.elapsed().as_millis();
            println!(
                "  {name:18} defects={:?} valid={} wall={dt}ms",
                out.report.unresolved_edge_pairs,
                out.report.preferred_validation().is_strictly_valid()
            );
        }
    }
    s2_voronoi::p5_shadow::set_term_pad_override(None);
}

/// Quad-coherence probe: regroup the paired records by sorted 4-point set,
/// where the two opposite-parity phrasings of the same question become
/// comparable. Contradictions here are the real cross-cell conflicts the
/// (triple, x)-keyed audit structurally missed.
#[test]
#[ignore]
fn probe_quad_coherence() {
    let run = |name: &str, points: &[s2_voronoi::UnitVec3], cutoff: f64| {
        s2_voronoi::p5_shadow::paired_reset();
        s2_voronoi::p5_shadow::set_pair_cutoff(cutoff);
        let out = compute_with_report(points, VoronoiConfig::default()).expect(name);
        s2_voronoi::p5_shadow::set_pair_cutoff(0.0);
        println!(
            "=== {name}: n={} cutoff={cutoff:.0e} defects={} ===",
            points.len(),
            out.report.unresolved_edge_pairs.len()
        );
        print!("{}", s2_voronoi::p5_shadow::paired_quad_report());
    };
    run("fixture_2k", &defect_fixture(), 1e-3);
    run("uniform_500k_s2", &random_sphere_points(500_000, 2), 1e-3);
    run("uniform_3m_s3", &random_sphere_points(3_000_000, 3), 1e-3);
}

/// Antisymmetric-tie-rule sweep (successor candidate 1): replace the
/// keep-bias `d >= -eps` with strict `d >= 0` (eps override 0.0) and
/// measure defects + validity across every known defect-bearing input plus
/// clean controls. Intermediate eps values map where the bias starts/stops
/// mattering. Also re-runs the quad-coherence report under the strict rule
/// on the defect carriers: the 1:1 parity contradictions should vanish if
/// the mechanism diagnosis is right.
#[test]
#[ignore]
fn probe_tie_rule_sweep() {
    let fixture_2k = defect_fixture();
    let u500k = random_sphere_points(500_000, 2);
    let u2m = random_sphere_points(2_000_000, 1);
    let u3m = random_sphere_points(3_000_000, 3);
    let u45m = random_sphere_points(4_500_000, 2);

    for eps in [None, Some(0.0f64), Some(1e-14), Some(1e-13)] {
        s2_voronoi::p5_shadow::set_clip_eps_override(eps);
        println!("=== clip_eps={eps:?} ===");
        for (name, points) in [
            ("fixture_2k", &fixture_2k),
            ("uniform_500k_s2", &u500k),
            ("uniform_2m_s1", &u2m),
            ("uniform_3m_s3", &u3m),
            ("uniform_4500k_s2", &u45m),
        ] {
            let t = std::time::Instant::now();
            let out = compute_with_report(points, VoronoiConfig::default()).expect(name);
            let dt = t.elapsed().as_millis();
            println!(
                "  {name:16} defects={} valid={} wall={dt}ms",
                out.report.unresolved_edge_pairs.len(),
                out.report.preferred_validation().is_strictly_valid()
            );
        }
    }

    // Quad coherence under the strict rule, defect carriers only.
    s2_voronoi::p5_shadow::set_clip_eps_override(Some(0.0));
    for (name, points) in [("fixture_2k", &fixture_2k), ("uniform_3m_s3", &u3m)] {
        s2_voronoi::p5_shadow::paired_reset();
        s2_voronoi::p5_shadow::set_pair_cutoff(1e-3);
        let out = compute_with_report(points, VoronoiConfig::default()).expect(name);
        s2_voronoi::p5_shadow::set_pair_cutoff(0.0);
        println!(
            "=== strict-rule quads {name}: defects={} ===",
            out.report.unresolved_edge_pairs.len()
        );
        print!("{}", s2_voronoi::p5_shadow::paired_quad_report());
    }
    s2_voronoi::p5_shadow::set_clip_eps_override(None);
}

/// 4.5M-s2 defect anatomy: the only carrier the tie-rule sweep left
/// untouched, and the one input the quad-coherence analysis never
/// characterized. Quad report + defect-site record dump under the default
/// rule and the strict rule, to learn whether its defects are parity
/// contradictions at all (vs question-set divergence / margins above the
/// collection cutoff).
#[test]
#[ignore]
fn probe_45m_quad_anatomy() {
    let points = random_sphere_points(4_500_000, 2);
    for eps in [None, Some(0.0f64)] {
        s2_voronoi::p5_shadow::set_clip_eps_override(eps);
        s2_voronoi::p5_shadow::paired_reset();
        s2_voronoi::p5_shadow::set_pair_cutoff(1e-4);
        let out = compute_with_report(&points, VoronoiConfig::default()).expect("4.5m");
        s2_voronoi::p5_shadow::set_pair_cutoff(0.0);
        println!(
            "=== 4.5m s2 clip_eps={eps:?} defects={:?} ===",
            out.report.unresolved_edge_pairs
        );
        print!("{}", s2_voronoi::p5_shadow::paired_quad_report());
        let ids: Vec<u32> = out
            .report
            .unresolved_edge_pairs
            .iter()
            .flat_map(|&(a, b, _)| [a, b])
            .collect();
        println!("--- records involving defect generators {ids:?} ---");
        print!("{}", s2_voronoi::p5_shadow::paired_dump_involving(&ids));
    }
    s2_voronoi::p5_shadow::set_clip_eps_override(None);
}

/// Plane strict-rule anatomy: rerun the two planar fixtures that fail
/// under strict bisectors (`PLANE_CLIP_EPS_INSIDE` overridden to 0.0),
/// with the planar audit + detection collector, and report per fixture
/// and rule:
/// - externally recomputed unpaired interior edges (vertex pairs, owning
///   cells) — the ground-truth defect set;
/// - the DETECTED unresolved set (what entered the repair net) — the gap
///   between these two is the detection escape;
/// - quad-coherence of near-margin decisions (exact planar in-circle) —
///   whether the defects are parity contradictions, and in which regime.
#[test]
#[ignore]
fn probe_plane_strict_anatomy() {
    use rand::{Rng, SeedableRng};
    use s2_voronoi::{compute_plane, PlanarVoronoi, PlaneRect};

    let clustered: Vec<[f32; 2]> = {
        let mut r = rand_chacha::ChaCha8Rng::seed_from_u64(17);
        let mut v: Vec<[f32; 2]> = (0..400)
            .map(|_| {
                [
                    (0.5 + r.gen_range(-0.01f32..0.01)).clamp(0.0, 1.0),
                    (0.5 + r.gen_range(-0.01f32..0.01)).clamp(0.0, 1.0),
                ]
            })
            .collect();
        v.push([0.01, 0.01]);
        v.push([0.99, 0.99]);
        v
    };
    let uniform: Vec<[f32; 2]> = {
        let mut r = rand_chacha::ChaCha8Rng::seed_from_u64(31);
        (0..50_000)
            .map(|_| [r.gen_range(0.0f32..1.0), r.gen_range(0.0f32..1.0)])
            .collect()
    };

    // Single-use cell edges not on the rect boundary: (va, vb, owning cell).
    fn unpaired_interior(d: &PlanarVoronoi) -> Vec<(u32, u32, u32)> {
        use std::collections::HashMap;
        let mut uses: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (ci, cell) in d.iter_cells().enumerate() {
            if cell.len() < 2 {
                continue;
            }
            for k in 0..cell.len() {
                let a = cell[k];
                let b = cell[(k + 1) % cell.len()];
                uses.entry((a.min(b), a.max(b)))
                    .or_default()
                    .push(ci as u32);
            }
        }
        let r = d.rect();
        let on_wall = |v: u32| {
            let p = d.vertex(v as usize);
            p.x <= r.min.x + 1e-5
                || p.x >= r.max.x - 1e-5
                || p.y <= r.min.y + 1e-5
                || p.y >= r.max.y - 1e-5
        };
        let mut out: Vec<(u32, u32, u32)> = uses
            .into_iter()
            .filter(|((a, b), cells)| cells.len() == 1 && !(on_wall(*a) && on_wall(*b)))
            .map(|((a, b), cells)| (a, b, cells[0]))
            .collect();
        out.sort_unstable();
        out
    }

    for (name, points) in [("clustered_402", &clustered), ("uniform_50k", &uniform)] {
        for eps in [None, Some(0.0f64)] {
            s2_voronoi::p5_shadow::set_plane_clip_eps_override(eps);
            s2_voronoi::p5_shadow::paired_reset();
            s2_voronoi::p5_shadow::plane_unresolved_reset();
            s2_voronoi::p5_shadow::set_pair_cutoff(1e-3);
            // Under the strict override the plain path now ERRORS on
            // residuals (the production correctness contract). The collector
            // still captured detection + residual records before the error,
            // so the anatomy is recoverable either way.
            let built = compute_plane(points, PlaneRect::unit());
            s2_voronoi::p5_shadow::set_pair_cutoff(0.0);
            let detected = s2_voronoi::p5_shadow::plane_unresolved();
            println!("=== {name} plane_eps={eps:?} ===");
            println!("  detected/residual (cell pairs): {detected:?}");
            match &built {
                Ok(diagram) => {
                    let unpaired = unpaired_interior(diagram);
                    println!("  final unpaired interior edges: {}", unpaired.len());
                    for &(va, vb, cell) in unpaired.iter().take(20) {
                        let pa = diagram.vertex(va as usize);
                        let pb = diagram.vertex(vb as usize);
                        println!(
                            "    v{va}({:.6},{:.6})-v{vb}({:.6},{:.6}) cell={cell}",
                            pa.x, pa.y, pb.x, pb.y
                        );
                    }
                }
                Err(e) => println!("  compute_plane errored (residual): {e:?}"),
            }
            print!("{}", s2_voronoi::p5_shadow::paired_quad_report());
            let cells: Vec<u32> = detected.iter().flat_map(|&(a, b)| [a, b]).collect();
            if !cells.is_empty() {
                println!("--- records involving defect cells {cells:?} ---");
                print!("{}", s2_voronoi::p5_shadow::paired_dump_involving(&cells));
            }
        }
    }
    s2_voronoi::p5_shadow::set_plane_clip_eps_override(None);
}

/// Escalation-band sweep: with the (unsound) wide 1e-8 filter shown to
/// manufacture contradictions at its boundary, sweep the band downward
/// toward the CLIP_EPS_INSIDE scale where the observed parity
/// contradictions actually live. factor = filter / CLIP_EPS_INSIDE;
/// factor 0 disables escalation entirely (baseline).
#[test]
#[ignore]
fn probe_escalation_band_sweep() {
    let fixture_2k = defect_fixture();
    let u500k = random_sphere_points(500_000, 2);
    let u3m = random_sphere_points(3_000_000, 3);

    for factor in [0.0f64, 64.0] {
        s2_voronoi::p5_shadow::set_escalation_factor_override(Some(factor));
        println!("=== factor={factor:.0e} ===");
        let fixture_360k = {
            let mean_spacing = (4.0 * std::f32::consts::PI / 2_000_000.0).sqrt();
            let cos_excl = (15.0 * mean_spacing).cos();
            let c = SITE_CENTER;
            let mut f = fixture_2k.clone();
            f.truncate(fixture_2k.len()); // window+2k scaffold; extend below
            f.extend(
                random_sphere_points(360_000, 4242)
                    .into_iter()
                    .skip(2_000)
                    .filter(|p| p.x() * c.0 + p.y() * c.1 + p.z() * c.2 < cos_excl),
            );
            f
        };
        let u2m = random_sphere_points(2_000_000, 1);
        let u45m = random_sphere_points(4_500_000, 2);
        for (name, points) in [
            ("fixture_2k", &fixture_2k),
            ("fixture_360k_ish", &fixture_360k),
            ("uniform_500k_s2", &u500k),
            ("uniform_2m_s1", &u2m),
            ("uniform_3m_s3", &u3m),
            ("uniform_4500k_s2", &u45m),
        ] {
            let out = compute_with_report(points, VoronoiConfig::default()).expect(name);
            println!(
                "  {name:16} defects={} valid={}",
                out.report.unresolved_edge_pairs.len(),
                out.report.preferred_validation().is_strictly_valid()
            );
        }
    }
    s2_voronoi::p5_shadow::set_escalation_factor_override(None);
}
