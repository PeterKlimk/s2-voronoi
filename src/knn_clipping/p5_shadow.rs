//! P5 stage-1 shadow audit (feature `p5_shadow`).
//!
//! Logs, with zero behavior change, how the chart-local clip decisions
//! compare against the canonical exact in-circle predicate that P5 would
//! escalate to (see `docs/p5-consistency-design.md`):
//!
//! - a margin histogram of every audited vertex-vs-bisector decision
//!   (normalized chart distance |d|/|n|, bucketed by decimal exponent),
//! - for near-margin decisions (below `CANONICAL_CUTOFF`), the exact
//!   canonical answer and whether the local decision disagrees,
//! - exact ties (true cocircularity) counted separately.
//!
//! This measures (a) escalation frequency for the P5 perf model and (b) how
//! often today's path disagrees with canonical — before any behavior changes.
//! Audits cover the gnomonic builder only (the fallback builder is the rare
//! `ProjectionLimit` path and defers clipping entirely).
//!
//! Note on permutation consistency: because the evaluator is *exact*, its
//! signs are automatically permutation-covariant — the sorted-row-order
//! discipline in the design doc only matters for rounded evaluators and for
//! the SoS cascade (stage 2).

use std::sync::atomic::{AtomicU64, Ordering};

use glam::Vec3;
use robust::Coord3D;

use crate::knn_clipping::topo2d::types::{HalfPlane, PolyBuffer};

/// Margin buckets by decimal exponent of the normalized distance:
/// bucket 0: nd >= 1e-1, bucket k: 1e-(k+1) <= nd < 1e-k, bucket 15: nd < 1e-15 (incl. 0).
const BUCKETS: usize = 16;

/// Normalized-margin cutoff below which the canonical predicate is evaluated.
/// Chart units are ~radians; cell sizes are ~2.5e-3 at 2M points, so 1e-4
/// catches the near-tie tail without evaluating everything.
const CANONICAL_CUTOFF: f64 = 1e-4;

static AUDITED: AtomicU64 = AtomicU64::new(0);
static SKIPPED_SYNTHETIC: AtomicU64 = AtomicU64::new(0);
static CANON_EVALS: AtomicU64 = AtomicU64::new(0);
static EXACT_TIES: AtomicU64 = AtomicU64::new(0);
#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static MARGIN_HIST: [AtomicU64; BUCKETS] = [ZERO; BUCKETS];
static DISAGREE_HIST: [AtomicU64; BUCKETS] = [ZERO; BUCKETS];

#[inline]
fn c3(p: Vec3) -> Coord3D<f64> {
    Coord3D {
        x: p.x as f64,
        y: p.y as f64,
        z: p.z as f64,
    }
}

/// Exact canonical in-circle on the unit sphere.
///
/// Returns +1 if `h` lies strictly inside the circumcircle of (g, a, b)
/// (the vertex (g,a,b) must be cut by bisector(g,h)), -1 if strictly
/// outside (the vertex is kept), 0 on an exact tie (4-cocircular, or a
/// degenerate triple).
///
/// Derivation: the Voronoi vertex v of (g,a,b) is the circumcircle pole
/// with v.g > 0; h inside the cap centered at v through g iff
/// (h-g).v > 0. With v ~ s*(a-g)x(b-g) and s = sign(((a-g)x(b-g)).g)
/// = sign(det[a,b,g]):
///
///   in_circle(g,a,b;h) = sign(det[a-g, b-g, h-g]) * sign(det[a, b, g])
///
/// Both determinants are evaluated exactly (Shewchuk adaptive orient3d;
/// f32 inputs are exactly representable in f64), so the sign is exact and
/// permutation-consistent by construction.
pub(crate) fn in_circle_sphere_sign(g: Vec3, a: Vec3, b: Vec3, h: Vec3) -> i8 {
    // orient3d(pa,pb,pc,pd) computes det of rows (pa-pd, pb-pd, pc-pd).
    let d1 = robust::orient3d(c3(a), c3(b), c3(h), c3(g));
    let d2 = robust::orient3d(
        c3(a),
        c3(b),
        c3(g),
        Coord3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
    );
    let prod_sign = (d1 > 0.0) as i8 - (d1 < 0.0) as i8;
    let pole_sign = (d2 > 0.0) as i8 - (d2 < 0.0) as i8;
    prod_sign * pole_sign
}

#[inline]
fn bucket_for(nd: f64) -> usize {
    if nd >= 1e-1 {
        return 0;
    }
    // nd == 0.0 -> inf -> saturating cast -> clamped to BUCKETS-1.
    let e = (-nd.log10()).floor() as usize;
    e.min(BUCKETS - 1)
}

/// Audit one clip attempt: for every current polygon vertex with real plane
/// attribution, classify its margin against the incoming half-plane and,
/// below the cutoff, compare the local decision with the canonical one.
pub(crate) fn audit_clip(
    generator_raw: Vec3,
    neighbor_raw: Vec3,
    neighbor_positions: &[Vec3],
    poly: &PolyBuffer,
    hp: &HalfPlane,
) {
    // NaN or zero-normal planes carry no geometric meaning to audit.
    if hp.ab2.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return;
    }
    let inv_norm = 1.0 / hp.ab2.sqrt();
    for i in 0..poly.len {
        let (pa, pb) = poly.vertex_planes[i];
        if pa == usize::MAX || pb == usize::MAX {
            SKIPPED_SYNTHETIC.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        debug_assert!(pa < neighbor_positions.len() && pb < neighbor_positions.len());
        let d = hp.signed_dist(poly.us[i], poly.vs[i]);
        let nd = d.abs() * inv_norm;
        AUDITED.fetch_add(1, Ordering::Relaxed);
        let bucket = bucket_for(nd);
        MARGIN_HIST[bucket].fetch_add(1, Ordering::Relaxed);

        if nd < CANONICAL_CUTOFF {
            CANON_EVALS.fetch_add(1, Ordering::Relaxed);
            let a = neighbor_positions[pa];
            let b = neighbor_positions[pb];
            let sign = in_circle_sphere_sign(generator_raw, a, b, neighbor_raw);
            if sign == 0 {
                EXACT_TIES.fetch_add(1, Ordering::Relaxed);
            } else {
                let local_keep = d >= -hp.eps;
                let canonical_keep = sign < 0;
                if local_keep != canonical_keep {
                    DISAGREE_HIST[bucket].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Reset all shadow counters.
pub fn reset() {
    AUDITED.store(0, Ordering::Relaxed);
    SKIPPED_SYNTHETIC.store(0, Ordering::Relaxed);
    CANON_EVALS.store(0, Ordering::Relaxed);
    EXACT_TIES.store(0, Ordering::Relaxed);
    for b in &MARGIN_HIST {
        b.store(0, Ordering::Relaxed);
    }
    for b in &DISAGREE_HIST {
        b.store(0, Ordering::Relaxed);
    }
}

/// Formatted dump of the shadow counters.
pub fn report() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let audited = AUDITED.load(Ordering::Relaxed);
    writeln!(
        out,
        "p5_shadow: audited={} skipped_synthetic={} canon_evals={} exact_ties={}",
        audited,
        SKIPPED_SYNTHETIC.load(Ordering::Relaxed),
        CANON_EVALS.load(Ordering::Relaxed),
        EXACT_TIES.load(Ordering::Relaxed),
    )
    .unwrap();
    writeln!(out, "  margin nd=|d|/|n|      decisions  disagreements").unwrap();
    for (k, (m, dis)) in MARGIN_HIST.iter().zip(DISAGREE_HIST.iter()).enumerate() {
        let m = m.load(Ordering::Relaxed);
        let dis = dis.load(Ordering::Relaxed);
        if m == 0 && dis == 0 {
            continue;
        }
        let label = if k == 0 {
            ">= 1e-1        ".to_string()
        } else if k == BUCKETS - 1 {
            format!("<  1e-{}        ", BUCKETS - 1)
        } else {
            format!("1e-{:<2} .. 1e-{:<2}", k + 1, k)
        };
        writeln!(out, "  {label} {m:>12}  {dis:>12}").unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_point(colat: f32, lon_deg: f32) -> Vec3 {
        let lon = lon_deg.to_radians();
        Vec3::new(
            colat.sin() * lon.cos(),
            colat.sin() * lon.sin(),
            colat.cos(),
        )
    }

    /// Known geometry: three points on the colatitude-theta circle around
    /// +z; the circumcircle pole is +z. Points at smaller colatitude are
    /// inside the cap, larger are outside.
    #[test]
    fn in_circle_sign_matches_known_geometry() {
        let theta = 0.3f32;
        let g = ring_point(theta, 0.0);
        let a = ring_point(theta, 120.0);
        let b = ring_point(theta, 240.0);

        let h_inside = ring_point(theta * 0.5, 60.0);
        let h_outside = ring_point(theta * 1.5, 60.0);

        assert_eq!(in_circle_sphere_sign(g, a, b, h_inside), 1);
        assert_eq!(in_circle_sphere_sign(g, a, b, h_outside), -1);
        // Swapping a/b (reversing triple orientation) must not change the
        // geometric answer.
        assert_eq!(in_circle_sphere_sign(g, b, a, h_inside), 1);
        assert_eq!(in_circle_sphere_sign(g, b, a, h_outside), -1);
        // The roles of g and the third ring point are symmetric.
        assert_eq!(in_circle_sphere_sign(a, g, b, h_inside), 1);
        // Exactly-on-circle fourth point: f32 ring points are not exactly
        // cocircular in general, so only check the planted exact tie below.
    }

    /// An exactly cocircular quadruple (shared z, symmetric lattice) must
    /// return 0.
    #[test]
    fn in_circle_sign_exact_tie() {
        let z = 0.5f32;
        let r = (1.0f32 - z * z).sqrt();
        let g = Vec3::new(r, 0.0, z);
        let a = Vec3::new(-r, 0.0, z);
        let b = Vec3::new(0.0, r, z);
        let h = Vec3::new(0.0, -r, z);
        assert_eq!(in_circle_sphere_sign(g, a, b, h), 0);
    }

    /// The exact sign must match a naive f64 evaluation on well-separated
    /// random quadruples (where naive arithmetic is reliable).
    #[test]
    fn in_circle_sign_matches_naive_on_clear_cases() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let random_unit = |rng: &mut rand_chacha::ChaCha8Rng| loop {
            let v = Vec3::new(
                rng.gen_range(-1.0f32..1.0),
                rng.gen_range(-1.0f32..1.0),
                rng.gen_range(-1.0f32..1.0),
            );
            let len = v.length();
            if len > 1e-2 && len < 1.0 {
                return v / len;
            }
        };
        let naive = |g: Vec3, a: Vec3, b: Vec3, h: Vec3| -> f64 {
            let gd = g.as_dvec3();
            let det3 = |r0: glam::DVec3, r1: glam::DVec3, r2: glam::DVec3| r0.cross(r1).dot(r2);
            let d1 = det3(a.as_dvec3() - gd, b.as_dvec3() - gd, h.as_dvec3() - gd);
            let d2 = det3(a.as_dvec3(), b.as_dvec3(), gd);
            d1.signum() * d2.signum()
        };
        let mut checked = 0;
        for _ in 0..2000 {
            let (g, a, b, h) = (
                random_unit(&mut rng),
                random_unit(&mut rng),
                random_unit(&mut rng),
                random_unit(&mut rng),
            );
            let gd = g.as_dvec3();
            let det3 = |r0: glam::DVec3, r1: glam::DVec3, r2: glam::DVec3| r0.cross(r1).dot(r2);
            let d1 = det3(a.as_dvec3() - gd, b.as_dvec3() - gd, h.as_dvec3() - gd);
            let d2 = det3(a.as_dvec3(), b.as_dvec3(), gd);
            // Only compare clear cases: naive f64 signs are trustworthy only
            // well away from zero (error ~1e-15 on O(1) inputs).
            if d1.abs() < 1e-9 || d2.abs() < 1e-9 {
                continue;
            }
            let n = naive(g, a, b, h);
            checked += 1;
            assert_eq!(
                in_circle_sphere_sign(g, a, b, h),
                n as i8,
                "exact vs naive mismatch for g={g:?} a={a:?} b={b:?} h={h:?}"
            );
        }
        assert!(checked > 1500, "too few clear cases: {checked}");
    }
}
