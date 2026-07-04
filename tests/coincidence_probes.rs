//! Coincidence margin-mapping probes (diagnostic, run manually).
//!
//! These map the resolvability boundary of the raw pipeline (welding
//! disabled): pair-separation sweeps, ulp-scale clusters, cube-map seam
//! alignments, and the rotated control. They print validation outcomes
//! rather than asserting, because they intentionally explore the regime
//! *below* the supported envelope; the margin data they produce backs the
//! weld radius in docs/correctness.md. Re-run after changes to the
//! clipping/dedup numerics to detect boundary drift:
//!
//!   cargo test --release --test coincidence_probes -- --ignored --nocapture
//!
//! The asserting contract tests derived from these findings live in
//! tests/adversarial.rs (weld contract + resolvable-regime sections).

mod support;

use glam::DVec3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use support::points::random_sphere_points;
use voronoi_mesh::{compute_with, validation::validate, PreprocessMode, UnitVec3, VoronoiConfig};

fn run(name: &str, points: &[UnitVec3]) {
    let config = VoronoiConfig::default().with_preprocess_mode(PreprocessMode::Disabled);
    match compute_with(points, config) {
        Ok(diagram) => {
            let report = validate(&diagram);
            let status = if report.is_strictly_valid() {
                "STRICT_VALID"
            } else {
                "INVALID"
            };
            eprintln!(
                "{name}: {} cells, {} - {status}",
                diagram.num_cells(),
                report.headline()
            );
        }
        Err(e) => {
            let msg = format!("{e:?}");
            let short: String = msg.chars().take(160).collect();
            eprintln!("{name}: FAILED {short}");
        }
    }
}

fn dvec(p: UnitVec3) -> DVec3 {
    DVec3::new(p.x as f64, p.y as f64, p.z as f64)
}

/// Twin at target geodesic-ish separation `s`, built in f64 then cast to f32.
/// Returns None if it rounds back to the identical bit pattern.
fn offset_twin(p: UnitVec3, s: f64, rng: &mut ChaCha8Rng) -> Option<(UnitVec3, f64)> {
    let p64 = dvec(p);
    let r = DVec3::new(
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    );
    let t = (r - p64 * r.dot(p64)).normalize();
    let q64 = (p64 + t * s).normalize();
    let q = UnitVec3::new(q64.x as f32, q64.y as f32, q64.z as f32);
    if (q.x.to_bits(), q.y.to_bits(), q.z.to_bits())
        == (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())
    {
        return None;
    }
    let realized = (dvec(q) - p64).length();
    Some((q, realized))
}

#[test]
#[ignore = "margin-mapping diagnostic - run manually"]
fn probe_separation_sweep() {
    const N: usize = 20_000;
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let base = random_sphere_points(N, 11);

    for &s in &[1e-12f64, 1e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7] {
        let mut pts = base.clone();
        let mut realized_min = f64::MAX;
        let mut realized_max = 0.0f64;
        let mut placed = 0usize;
        let mut i = 0usize;
        while placed < 100 && i < 1000 {
            if let Some((q, realized)) = offset_twin(base[i * 7], s, &mut rng) {
                pts.push(q);
                realized_min = realized_min.min(realized);
                realized_max = realized_max.max(realized);
                placed += 1;
            }
            i += 1;
        }
        let name =
            format!("sep_{s:.0e}_x{placed} (realized {realized_min:.1e}..{realized_max:.1e})");
        run(&name, &pts);
    }
}

#[test]
#[ignore = "margin-mapping diagnostic - run manually"]
fn probe_seam_pairs_ulp_scale() {
    const N: usize = 20_000;
    let base = random_sphere_points(N, 11);

    // Seam points with twins nudged ONLY in dominant components (|v| > 0.5),
    // guaranteeing ulp-scale separation, never denormal-scale.
    let inv3 = 1.0f32 / 3.0f32.sqrt();
    let inv2 = 1.0f32 / 2.0f32.sqrt();
    let mut seam_points: Vec<UnitVec3> = Vec::new();
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                seam_points.push(UnitVec3::new(sx * inv3, sy * inv3, sz * inv3));
            }
        }
    }
    for s in [-1.0f32, 1.0] {
        for t in [-1.0f32, 1.0] {
            seam_points.push(UnitVec3::new(s * inv2, t * inv2, 0.0));
            seam_points.push(UnitVec3::new(s * inv2, 0.0, t * inv2));
            seam_points.push(UnitVec3::new(0.0, s * inv2, t * inv2));
        }
    }
    for s in [-1.0f32, 1.0] {
        seam_points.push(UnitVec3::new(s, 0.0, 0.0));
        seam_points.push(UnitVec3::new(0.0, s, 0.0));
        seam_points.push(UnitVec3::new(0.0, 0.0, s));
    }

    let mut pts = base.clone();
    for &p in &seam_points {
        let twin = UnitVec3::new(
            if p.x.abs() > 0.5 { p.x.next_up() } else { p.x },
            if p.y.abs() > 0.5 { p.y.next_up() } else { p.y },
            if p.z.abs() > 0.5 { p.z.next_up() } else { p.z },
        );
        pts.push(p);
        pts.push(twin);
    }
    run("seam_pairs_ulp_only_x26", &pts);
}

#[test]
#[ignore = "margin-mapping diagnostic - run manually"]
fn probe_seam_pairs_rotated() {
    const N: usize = 20_000;
    let base = random_sphere_points(N, 11);

    // Same 26 symmetric positions, but rotated by an arbitrary rotation first,
    // so nothing sits on a cube-map seam or has exact-zero/equal components.
    // Twins built the same way (next_up on dominant components, post-rotation).
    let rot = glam::DQuat::from_euler(glam::EulerRot::XYZ, 0.71, 1.13, 2.41);

    let inv3 = 1.0f64 / 3.0f64.sqrt();
    let inv2 = 1.0f64 / 2.0f64.sqrt();
    let mut seam_points64: Vec<DVec3> = Vec::new();
    for sx in [-1.0f64, 1.0] {
        for sy in [-1.0f64, 1.0] {
            for sz in [-1.0f64, 1.0] {
                seam_points64.push(DVec3::new(sx * inv3, sy * inv3, sz * inv3));
            }
        }
    }
    for s in [-1.0f64, 1.0] {
        for t in [-1.0f64, 1.0] {
            seam_points64.push(DVec3::new(s * inv2, t * inv2, 0.0));
            seam_points64.push(DVec3::new(s * inv2, 0.0, t * inv2));
            seam_points64.push(DVec3::new(0.0, s * inv2, t * inv2));
        }
    }
    for s in [-1.0f64, 1.0] {
        seam_points64.push(DVec3::new(s, 0.0, 0.0));
        seam_points64.push(DVec3::new(0.0, s, 0.0));
        seam_points64.push(DVec3::new(0.0, 0.0, s));
    }

    let mut pts = base.clone();
    for &p64 in &seam_points64 {
        let r = (rot * p64).normalize();
        let p = UnitVec3::new(r.x as f32, r.y as f32, r.z as f32);
        let twin = UnitVec3::new(
            if p.x.abs() > 0.5 { p.x.next_up() } else { p.x },
            if p.y.abs() > 0.5 { p.y.next_up() } else { p.y },
            if p.z.abs() > 0.5 { p.z.next_up() } else { p.z },
        );
        pts.push(p);
        pts.push(twin);
    }
    run("seam_pairs_rotated_x26", &pts);
}

/// k points scattered in a tangent disc of `radius` around `center`, cast to
/// f32, bitwise-distinct (re-rolled on collision). Returns the points plus the
/// realized min pairwise chord distance.
fn cluster_at(center: DVec3, k: usize, radius: f64, rng: &mut ChaCha8Rng) -> (Vec<UnitVec3>, f64) {
    let c = center.normalize();
    let mut pts: Vec<UnitVec3> = Vec::with_capacity(k);
    let mut tries = 0;
    while pts.len() < k && tries < 200 {
        tries += 1;
        let r = DVec3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        let t = (r - c * r.dot(c)).normalize();
        let q64 = (c + t * (radius * rng.gen_range(0.1..1.0))).normalize();
        let q = UnitVec3::new(q64.x as f32, q64.y as f32, q64.z as f32);
        let dup = pts.iter().any(|p| {
            (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())
                == (q.x.to_bits(), q.y.to_bits(), q.z.to_bits())
        });
        if !dup {
            pts.push(q);
        }
    }
    let mut min_sep = f64::MAX;
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            min_sep = min_sep.min((dvec(pts[i]) - dvec(pts[j])).length());
        }
    }
    (pts, min_sep)
}

fn seam_centers() -> Vec<DVec3> {
    let inv3 = 1.0f64 / 3.0f64.sqrt();
    let inv2 = 1.0f64 / 2.0f64.sqrt();
    let mut out: Vec<DVec3> = Vec::new();
    for sx in [-1.0f64, 1.0] {
        for sy in [-1.0f64, 1.0] {
            for sz in [-1.0f64, 1.0] {
                out.push(DVec3::new(sx * inv3, sy * inv3, sz * inv3));
            }
        }
    }
    for s in [-1.0f64, 1.0] {
        for t in [-1.0f64, 1.0] {
            out.push(DVec3::new(s * inv2, t * inv2, 0.0));
            out.push(DVec3::new(s * inv2, 0.0, t * inv2));
            out.push(DVec3::new(0.0, s * inv2, t * inv2));
        }
    }
    for s in [-1.0f64, 1.0] {
        out.push(DVec3::new(s, 0.0, 0.0));
        out.push(DVec3::new(0.0, s, 0.0));
        out.push(DVec3::new(0.0, 0.0, s));
    }
    out
}

#[test]
#[ignore = "margin-mapping diagnostic - run manually"]
fn probe_cluster_margin_sweep() {
    const N: usize = 20_000;
    let mut rng = ChaCha8Rng::seed_from_u64(404);
    let base = random_sphere_points(N, 11);

    // Random positions: cluster size x radius.
    for &k in &[3usize, 5, 9] {
        for &radius in &[1e-8f64, 1e-7, 1e-6, 1e-5] {
            let mut pts = base.clone();
            let mut worst_min = f64::MAX;
            let mut placed = 0;
            for i in 0..50 {
                let center = dvec(base[i * 31]);
                let (cluster, min_sep) = cluster_at(center, k, radius, &mut rng);
                if cluster.len() == k {
                    worst_min = worst_min.min(min_sep);
                    pts.extend(cluster);
                    placed += 1;
                }
            }
            let name = format!("rand_k{k}_r{radius:.0e}_x{placed} (min_sep {worst_min:.1e})");
            run(&name, &pts);
        }
    }

    // Seam positions: the worst known regime.
    for &k in &[3usize, 5] {
        for &radius in &[1e-7f64, 1e-6, 1e-5] {
            let mut pts = base.clone();
            let mut worst_min = f64::MAX;
            let mut placed = 0;
            for center in seam_centers() {
                let (cluster, min_sep) = cluster_at(center, k, radius, &mut rng);
                if cluster.len() == k {
                    worst_min = worst_min.min(min_sep);
                    pts.extend(cluster);
                    placed += 1;
                }
            }
            let name = format!("seam_k{k}_r{radius:.0e}_x{placed} (min_sep {worst_min:.1e})");
            run(&name, &pts);
        }
    }
}

#[test]
#[ignore = "margin-mapping diagnostic - run manually"]
fn probe_aligned_separated_pairs() {
    const N: usize = 20_000;
    let mut rng = ChaCha8Rng::seed_from_u64(505);
    let base = random_sphere_points(N, 11);

    // Pairs with exact z=0 (both points ON the equator great circle, which is
    // also a cube-map seam), at increasing along-circle separations. The
    // nastiest alignment we know, now with separation as the only variable.
    for &sep in &[3e-7f64, 1e-6, 3e-6, 1e-5, 1e-4] {
        let mut pts = base.clone();
        let mut realized_min = f64::MAX;
        for _ in 0..30 {
            let theta: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            let a = UnitVec3::new(theta.cos() as f32, theta.sin() as f32, 0.0);
            let t2 = theta + sep;
            let b = UnitVec3::new(t2.cos() as f32, t2.sin() as f32, 0.0);
            if (a.x.to_bits(), a.y.to_bits()) == (b.x.to_bits(), b.y.to_bits()) {
                continue;
            }
            realized_min = realized_min.min((dvec(a) - dvec(b)).length());
            pts.push(a);
            pts.push(b);
        }
        let name = format!("eq_aligned_sep_{sep:.0e} (realized_min {realized_min:.1e})");
        run(&name, &pts);
    }
}
