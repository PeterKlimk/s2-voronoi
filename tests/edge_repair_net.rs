//! Deterministic coverage net for the edge-repair machinery.
//!
//! Unresolved shared-edge mismatches are rare (roughly 1-20 per multi-million
//! uniform run, none at smaller scales), and *cross-bin* mismatches are rarer
//! still — a conjunction that uniform fuzzing essentially never samples, so a
//! wrong repair could hide indefinitely. This suite makes the rare detection
//! and repair paths fire deterministically and asserts both that defects
//! actually occurred (so the net cannot silently rot into a no-op) and that
//! the repaired diagram still validates strictly.
//!
//! ## The fixture
//!
//! Synthetic degeneracy does not produce these defects: exact quantized
//! lattices, near-cocircular rings down to 1e-8 perturbation, cube-vertex /
//! great-circle / clustered-cap stress, and uniform runs up to 1M all yield
//! zero unresolved edges (probe history, 2026-06). Exact ties resolve
//! *identically* in both cells' charts; a defect needs a near-tie sitting in
//! the narrow cross-chart rounding gap, which only shows up by volume.
//!
//! The fixture is therefore a real defect site: uniform 2M seed 1 contains a
//! cluster around `SITE_CENTER` whose cell disagreements produce 3 unresolved
//! edges. A 10-mean-spacing cap of that input around the site, plus a sparse
//! scaffold elsewhere on the sphere (so boundary cells stay bounded),
//! reproduces the defects at ~1.7k points. The disagreement survives window
//! radius and scaffold-density changes, but NOT rotation (f32 re-rounding
//! erases the gap) and NOT input-order reversal — do not reorder the fixture
//! construction.
//!
//! ## Steering defects across bins
//!
//! Bin boundaries lie on grid-cell lines, and grid resolution is a function
//! of total point count — so the scaffold size steers bin boundaries across
//! the (fixed) defect site without touching its bits. Pinned by probe sweep
//! (re-pinned after P5 stage 0 moved the fingerprint): scaffold 280k+bins 12
//! shows InBinUnconsumedCheck; scaffold 360k (also 560k) + bins 36-54 puts
//! the one-sided defect cross-bin (CrossBinSingleSided).
//!
//! ## Post stage-0 landscape (input canonicalization, 2026-06)
//!
//! Entry canonicalization eliminated the natural defect population at 2M:
//! uniform seeds 1-10 all produce ZERO unresolved edges (pre stage-0, seed 1
//! had a 3-defect site). The windowed fixture still defects (different total
//! n -> different grid -> different epsilon decisions), making it the only
//! known deterministic sphere defect source — treat it as load-bearing.
//! CrossBinThirdsMismatch lost its deterministic pin (see the coverage-gap
//! note at its former test site).
//!
//! ## Bin layout and the defect set
//!
//! When the site is interior to one bin, the defect pair set is identical at
//! every bin count (only bookkeeping changes). When a bin boundary cuts the
//! site, the pair's evaluation path itself changes — same-bin pairs are
//! clipped once and seed-forwarded, cross-bin pairs are clipped
//! independently by both sides — so the epsilon disagreements (and hence the
//! defect set) can legitimately differ between layouts. Measured: scaffold
//! 280k yields 4 defects at bins=12 and 3 (one cross-bin) at bins=48. The
//! invariant that holds across all layouts is the contract itself: strict
//! validity after repair.
//!
//! ## Origin coverage
//!
//! Deterministically exercised: InBinThirdsMismatch, InBinUnconsumedCheck,
//! CrossBinSingleSided. CrossBinThirdsMismatch: deterministic pin lost at
//! stage 0 (coverage gap, see above). Not covered:
//! InBinMissingCheck (believed unreachable by construction — an edge to an
//! earlier same-bin neighbor only enters a cell via a replayed seed, and a
//! seed implies its check is present; the branch remains as a conservative
//! repair route) and CrossBinDuplicateSide (debug-asserted bug trap).
//!
//! If a numerics/tolerance/policy change shifts the defects and these tests
//! fail, re-run the discovery probes (`probe_*`, ignored by default) to find
//! replacement fixtures; see each probe's doc comment.

mod support;

use s2_voronoi::{
    compute_with_report, validation::validate, ComputeOutput, UnitVec3Like, UnresolvedEdgeOrigin,
    VoronoiConfig,
};
use support::points::*;

/// The defect site discovered in uniform 2M seed 1 (generator 1948042's
/// cluster): all three unresolved pairs share this generator. Only used as a
/// cap center, so the coordinates need not be bit-exact.
const SITE_CENTER: (f32, f32, f32) = (-0.419_580_3, -0.220_115_17, -0.880_625_7);

const SOURCE_N: usize = 2_000_000;
const WINDOW_RADIUS_MULT: f32 = 10.0;
const SCAFFOLD_EXCL_MULT: f32 = 15.0;

/// Serialize env-var mutation across tests in this binary.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn with_env<R>(bin_count: Option<usize>, repair_rebuild: bool, f: impl FnOnce() -> R) -> R {
    let _guard = ENV_LOCK.lock().unwrap();
    match bin_count {
        Some(n) => std::env::set_var("S2_BIN_COUNT", n.to_string()),
        None => std::env::remove_var("S2_BIN_COUNT"),
    }
    if repair_rebuild {
        std::env::set_var("S2_EDGE_REPAIR_REBUILD", "1");
    } else {
        std::env::remove_var("S2_EDGE_REPAIR_REBUILD");
    }
    let result = f();
    std::env::remove_var("S2_BIN_COUNT");
    std::env::remove_var("S2_EDGE_REPAIR_REBUILD");
    result
}

fn with_bin_count<R>(bin_count: Option<usize>, f: impl FnOnce() -> R) -> R {
    with_env(bin_count, false, f)
}

fn mean_spacing() -> f32 {
    (4.0 * std::f32::consts::PI / SOURCE_N as f32).sqrt()
}

/// The defect-site cap of the uniform-2M-seed-1 input.
fn defect_window() -> Vec<s2_voronoi::UnitVec3> {
    let cos_window = (WINDOW_RADIUS_MULT * mean_spacing()).cos();
    let c = SITE_CENTER;
    random_sphere_points(SOURCE_N, 1)
        .into_iter()
        .filter(|p| p.x() * c.0 + p.y() * c.1 + p.z() * c.2 >= cos_window)
        .collect()
}

/// Window plus a uniform scaffold outside the (slightly larger) exclusion
/// cap, so every cell is bounded and the defect neighborhood is untouched.
/// The scaffold size sets the grid resolution, which positions the
/// bin-boundary lines relative to the site. Order matters: window first,
/// scaffold second (the defect is input-order sensitive).
fn with_scaffold(window: &[s2_voronoi::UnitVec3], scaffold_n: usize) -> Vec<s2_voronoi::UnitVec3> {
    let cos_excl = (SCAFFOLD_EXCL_MULT * mean_spacing()).cos();
    let c = SITE_CENTER;
    let mut fixture = window.to_vec();
    fixture.extend(
        random_sphere_points(scaffold_n, 4242)
            .into_iter()
            .filter(|p| p.x() * c.0 + p.y() * c.1 + p.z() * c.2 < cos_excl),
    );
    fixture
}

/// Compute under a bin count and assert the contract: success + strictly
/// valid (both the effective and the returned diagram).
fn compute_strict(
    name: &str,
    points: &[s2_voronoi::UnitVec3],
    bins: Option<usize>,
) -> ComputeOutput {
    let out = with_bin_count(bins, || {
        compute_with_report(points, VoronoiConfig::default())
    })
    .unwrap_or_else(|err| panic!("{name} bins={bins:?}: compute failed: {err:?}"));

    let report = out.report.preferred_validation();
    assert!(
        report.is_strictly_valid(),
        "{name} bins={bins:?}: repaired diagram must validate strictly: {}",
        report.headline()
    );
    let returned = validate(&out.diagram);
    assert!(
        returned.is_strictly_valid(),
        "{name} bins={bins:?}: returned diagram must validate strictly: {}",
        returned.headline()
    );
    out
}

fn pair_set(out: &ComputeOutput) -> std::collections::BTreeSet<(u32, u32)> {
    out.report
        .unresolved_edge_pairs
        .iter()
        .map(|&(a, b, _)| (a, b))
        .collect()
}

fn origins(out: &ComputeOutput) -> Vec<UnresolvedEdgeOrigin> {
    let mut v: Vec<_> = out
        .report
        .unresolved_edge_pairs
        .iter()
        .map(|&(_, _, o)| o)
        .collect();
    v.sort();
    v
}

/// Within-bin thirds-mismatch detection: the small fixture must produce
/// defects through InBinThirdsMismatch at every bin count, with a
/// bin-invariant defect set, and repair must restore strict validity.
/// (Post stage-0 the small fixture's case mix is thirds-mismatch only;
/// the unconsumed-check case pins at scaffold 280k below.)
#[test]
fn net_in_bin_detection_and_repair() {
    let fixture = with_scaffold(&defect_window(), 2_000);

    let mut sets = Vec::new();
    for bins in [Some(6), None, Some(96)] {
        let out = compute_strict("in_bin_net", &fixture, bins);
        let os = origins(&out);
        assert!(
            os.contains(&UnresolvedEdgeOrigin::InBinThirdsMismatch),
            "expected InBinThirdsMismatch at bins={bins:?}, got {os:?}"
        );
        sets.push((bins, pair_set(&out)));
    }
    let (_, first) = &sets[0];
    assert!(!first.is_empty(), "defect fixture produced no defects");
    for (bins, set) in &sets[1..] {
        assert_eq!(
            set, first,
            "defect pair set must be independent of bin count (bins={bins:?})"
        );
    }
}

/// Within-bin unconsumed-check detection: at scaffold 280k with an in-bin
/// layout, the one-sided epsilon-edge defect is detected by the later cell
/// as an unconsumed incoming check.
#[test]
fn net_in_bin_unconsumed_check() {
    let fixture = with_scaffold(&defect_window(), 280_000);
    let out = compute_strict("in_bin_unconsumed", &fixture, Some(12));
    let os = origins(&out);
    assert!(
        os.contains(&UnresolvedEdgeOrigin::InBinUnconsumedCheck),
        "expected InBinUnconsumedCheck at scaffold 280k bins=12, got {os:?} \
         (layout may have moved; re-run probe_scaffold_sweep)"
    );
}

/// Cross-bin one-sided detection: at scaffold 360k with bins 36-54 a bin
/// boundary splits the site so the one-sided epsilon-edge defect is
/// detected by the overflow matcher (CrossBinSingleSided). No in-bin
/// control comparison: post stage-0, the in-bin layouts of this fixture
/// produce no defects at all (the defect set is layout-dependent when the
/// boundary cuts the site; see module docs).
#[test]
fn net_cross_bin_single_sided() {
    let fixture = with_scaffold(&defect_window(), 360_000);
    let split = compute_strict("cross_single_split", &fixture, Some(48));
    let os = origins(&split);
    assert!(
        os.contains(&UnresolvedEdgeOrigin::CrossBinSingleSided),
        "expected CrossBinSingleSided at scaffold 360k bins=48, got {os:?} \
         (bin layout may have moved; re-run probe_scaffold_sweep)"
    );
}

// Coverage gap (post stage-0): CrossBinThirdsMismatch lost its deterministic
// fixture — input canonicalization eliminated the configurations where the
// thirds-mismatch defect family survived a bin-boundary split (sweeps over
// scaffold 40k-640k x bins {12,36,48,54,96} found none; the family resolves
// consistently when evaluated via the independent EmitAll path). The
// detection branch remains; re-run probe_scaffold_sweep/probe_site_scan
// after numerics changes to look for a new pin.

/// Full-pipeline differential: the surgical in-place repair (production
/// default) and the original full-rebuild oracle (`S2_EDGE_REPAIR_REBUILD=1`)
/// must produce semantically identical diagrams — same defects detected,
/// same vertices, same per-cell vertex-id sequences — on every net fixture
/// (in-bin and cross-bin defect routings).
#[test]
fn net_repair_backends_agree() {
    let window = defect_window();
    for (scaffold_n, bins) in [(2_000usize, None), (280_000, Some(12)), (360_000, Some(48))] {
        let fixture = with_scaffold(&window, scaffold_n);
        let name = format!("differential s={scaffold_n} bins={bins:?}");

        let surgical = with_env(bins, false, || {
            compute_with_report(&fixture, VoronoiConfig::default())
        })
        .unwrap_or_else(|e| panic!("{name} surgical: {e:?}"));
        let rebuild = with_env(bins, true, || {
            compute_with_report(&fixture, VoronoiConfig::default())
        })
        .unwrap_or_else(|e| panic!("{name} rebuild: {e:?}"));

        assert!(
            !surgical.report.unresolved_edge_pairs.is_empty(),
            "{name}: fixture produced no defects, differential is vacuous"
        );
        assert_eq!(
            surgical.report.unresolved_edge_pairs, rebuild.report.unresolved_edge_pairs,
            "{name}: backends saw different defects (detection precedes repair; this \
             would mean nondeterminism upstream of the repair backends)"
        );

        let (ds, dr) = (&surgical.diagram, &rebuild.diagram);
        assert_eq!(ds.num_cells(), dr.num_cells(), "{name}: cell count differs");
        assert_eq!(
            ds.num_vertices(),
            dr.num_vertices(),
            "{name}: vertex count differs"
        );
        for i in 0..ds.num_cells() {
            assert_eq!(
                ds.cell(i).vertex_indices,
                dr.cell(i).vertex_indices,
                "{name}: cell {i} vertex sequence differs between repair backends"
            );
        }
        assert!(
            surgical.report.preferred_validation().is_strictly_valid(),
            "{name}: surgical result must validate strictly"
        );
    }
}

// ---------------------------------------------------------------------------
// Discovery probes (ignored): tools for re-finding fixtures after a numerics
// or policy change, kept with the findings that shaped the net.
// ---------------------------------------------------------------------------

/// Scan inputs for natural defect sites. Findings (2026-06): synthetic
/// degeneracy (quantized lattices, 1e-8 cocircular rings, cube-vertex /
/// great-circle / cap stress) produces zero unresolved edges; uniform runs
/// produce none up to 1M, and ~1 site per multi-million run (2M seed 1 has
/// one 3-defect site; seeds 2-4 have none). Run with
/// `cargo test --release --test edge_repair_net probe_site_scan -- --ignored --nocapture`
#[test]
#[ignore]
fn probe_site_scan() {
    let mut candidates: Vec<(String, Vec<s2_voronoi::UnitVec3>)> = Vec::new();
    for seed in 1..=10u64 {
        candidates.push((
            format!("uniform_2m_s{seed}"),
            random_sphere_points(2_000_000, seed),
        ));
    }
    for (name, points) in &candidates {
        let out = with_bin_count(None, || {
            compute_with_report(points, VoronoiConfig::default())
        })
        .expect("compute");
        let gens = out.preferred_diagram().generators().to_vec();
        print!("{name}: ");
        if out.report.unresolved_edge_pairs.is_empty() {
            println!("no defects");
        } else {
            println!();
            for &(a, b, origin) in &out.report.unresolved_edge_pairs {
                let g = gens[a as usize];
                println!(
                    "  ({a},{b},{origin:?}) site=({:.9},{:.9},{:.9})",
                    g.x(),
                    g.y(),
                    g.z()
                );
            }
        }
    }
}

/// Re-derive the windowed fixture from a defect site found by
/// `probe_site_scan` (update `SITE_CENTER`), checking the defects reproduce
/// across window radii. Findings (2026-06): defects survive windowing at
/// 5-40x mean spacing and scaffold changes, but not rotation (f32
/// re-rounding) and not input-order reversal. Run with
/// `cargo test --release --test edge_repair_net probe_window -- --ignored --nocapture`
#[test]
#[ignore]
fn probe_window() {
    let window = defect_window();
    for scaffold_n in [1_000usize, 2_000, 5_000] {
        let fixture = with_scaffold(&window, scaffold_n);
        let out = with_bin_count(None, || {
            compute_with_report(&fixture, VoronoiConfig::default())
        });
        match out {
            Ok(out) => println!(
                "scaffold={scaffold_n:6}: n={} unresolved={:?} strict_valid={}",
                fixture.len(),
                out.report.unresolved_edge_pairs,
                out.report.preferred_validation().is_strictly_valid()
            ),
            Err(err) => println!("scaffold={scaffold_n:6}: ERROR: {err:?}"),
        }
    }
}

/// Sweep scaffold size x bin count to find configurations where a bin
/// boundary splits the defect site (CrossBin* origins). Findings (2026-06):
/// scaffold 280k -> CrossBinSingleSided and 320k -> CrossBinThirdsMismatch,
/// each at 36-54 bins; the defect pair set is bin-count invariant while the
/// site stays interior to a bin, but can change when the boundary cuts the
/// site (see module docs). Run with
/// `cargo test --release --test edge_repair_net probe_scaffold_sweep -- --ignored --nocapture`
#[test]
#[ignore]
fn probe_scaffold_sweep() {
    let window = defect_window();
    for scaffold_n in (40_000usize..=640_000).step_by(40_000) {
        let fixture = with_scaffold(&window, scaffold_n);
        for bins in [12usize, 36, 48, 54, 96] {
            let out = with_bin_count(Some(bins), || {
                compute_with_report(&fixture, VoronoiConfig::default())
            });
            match out {
                Ok(out) => println!(
                    "scaffold={scaffold_n:7} bins={bins:2}: unresolved={:?} strict_valid={}",
                    out.report.unresolved_edge_pairs,
                    out.report.preferred_validation().is_strictly_valid()
                ),
                Err(err) => println!("scaffold={scaffold_n:7} bins={bins:2}: ERROR: {err:?}"),
            }
        }
    }
}
