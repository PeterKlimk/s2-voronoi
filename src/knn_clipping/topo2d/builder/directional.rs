//! Direction-aware non-cutting certificate ("directional termination").
//!
//! `clip_convex` already carries the scalar coefficient-space early-unchanged
//! check: a half-plane `(a, b, c)` cannot cut the polygon when `c >= 0` and
//! `c² >= (a² + b²) · max_r2` — the candidate's bisector clears the polygon's
//! circumradius in every direction. This module is the per-direction
//! refinement of that same check: replace the global `max_r2` with a
//! per-octant window bound `W²[k]` on the polygon's support toward the
//! candidate's chart direction. A candidate close in dot terms but lying in a
//! direction where the cell is already tight then certifies as provably
//! `Unchanged` without running the clip at all — and when every remaining
//! known in-batch candidate certifies while the scalar termination
//! certificate covers everything beyond the batch, the whole cell terminates
//! early (see `cell_build/run.rs`).
//!
//! Soundness is pure 2D algebra over the exact same `(a, b, c)` the clipper
//! would compute (`bisector_coefficients`) and the exact same stored chart
//! vertices the clipper would evaluate — no unit-input or chart-orthonormality
//! assumption anywhere (a raw-dot-space variant of this certificate is
//! *unsound* near the pole-adjacent tangent-basis cancellation region; see the
//! 2026-07 design review). With the sphere's strict clip rule
//! (`CLIP_EPS_INSIDE = 0.0`), a certified candidate is one the clipper would
//! have returned `Unchanged` for, so skipping it leaves the output
//! byte-identical.

use super::GnomonicBuilder;
use glam::Vec3;

/// Number of direction windows (octants of the chart plane).
pub(crate) const DIR_SECTORS: usize = 8;

/// Relative inflation applied to the per-window support bound. This must
/// dominate the floating-point rounding of the table build, of `a² + b²` /
/// `c²`, and of the clipper's fma `signed_dist` evaluation the certificate
/// speaks for (each ~2⁻⁵² relative, chains of ≤ ~5 ops). 1e-9 leaves ~5
/// orders of magnitude of headroom while only refusing candidates within
/// ~5e-10 relative of the exact non-cutting boundary — exactly the
/// near-cocircular band where a conservative "not provably unchanged" answer
/// is the safe, cross-cell-consistent one.
const W2_MARGIN: f64 = 1.0 + 1e-9;

/// Cyclic window index of direction `(x, y)`: window `k` spans
/// `[k·45°, (k+1)·45°)`. Ties on window boundaries may classify to either
/// side; that is sound because each window's bound includes the support at
/// both of its boundary directions (see `rebuild_dir_table`).
#[inline]
pub(super) fn dir_window(x: f64, y: f64) -> usize {
    // bits: (y < 0) << 2 | (x < 0) << 1 | (|y| > |x|), remapped to be cyclic.
    const LUT: [u8; 8] = [0, 1, 3, 2, 7, 6, 4, 5];
    let bits =
        (((y < 0.0) as usize) << 2) | (((x < 0.0) as usize) << 1) | ((y.abs() > x.abs()) as usize);
    LUT[bits] as usize
}

impl GnomonicBuilder {
    /// Rebuild the per-window squared support bounds from the current polygon.
    ///
    /// The polygon's support function `h(φ) = max_i (u_i·cos φ + v_i·sin φ)`
    /// is a max of per-vertex cosine humps, so its supremum over a window is
    /// exactly `max(h(left boundary), h(right boundary), max r_i over
    /// vertices whose own direction lies in the window)` — a hump's max over
    /// an interval is its peak `r_i` if the peak direction is inside, else its
    /// value at the nearest boundary, and the boundary supports cover all
    /// out-of-window vertices. Everything stays in squared space (all terms
    /// are >= 0 once clamped: the polygon contains the chart origin), so the
    /// build is sqrt- and trig-free.
    ///
    /// Out-of-line: this runs at most once per polygon change, and only on
    /// cells where a directional attempt fires; the clip loop stays small.
    #[inline(never)]
    pub(super) fn rebuild_dir_table(&mut self) {
        let poly = if self.use_a {
            &self.poly_a
        } else {
            &self.poly_b
        };
        debug_assert!(
            !poly.has_bounding_ref() && poly.len >= 3,
            "directional table rebuilt on an unbounded or degenerate polygon"
        );

        let mut max_u = f64::NEG_INFINITY;
        let mut min_u = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        let mut min_v = f64::INFINITY;
        let mut max_d1 = f64::NEG_INFINITY;
        let mut min_d1 = f64::INFINITY;
        let mut max_d2 = f64::NEG_INFINITY;
        let mut min_d2 = f64::INFINITY;
        let mut r2max = [0.0f64; DIR_SECTORS];

        for i in 0..poly.len {
            let u = poly.us[i];
            let v = poly.vs[i];
            let d1 = u + v;
            let d2 = u - v;
            max_u = max_u.max(u);
            min_u = min_u.min(u);
            max_v = max_v.max(v);
            min_v = min_v.min(v);
            max_d1 = max_d1.max(d1);
            min_d1 = min_d1.min(d1);
            max_d2 = max_d2.max(d2);
            min_d2 = min_d2.min(d2);
            let r2 = u * u + v * v;
            let k = dir_window(u, v);
            if r2 > r2max[k] {
                r2max[k] = r2;
            }
        }

        // Support at the eight window-boundary directions (multiples of 45°),
        // read off the running extremes: e.g. h(135°) = max((v - u)/√2).
        const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
        let h = [
            max_u,
            max_d1 * INV_SQRT2,
            max_v,
            -min_d2 * INV_SQRT2,
            -min_u,
            -min_d1 * INV_SQRT2,
            -min_v,
            max_d2 * INV_SQRT2,
        ];

        for k in 0..DIR_SECTORS {
            let hk = h[k].max(0.0);
            let hk1 = h[(k + 1) & (DIR_SECTORS - 1)].max(0.0);
            let w2 = (hk * hk).max(hk1 * hk1).max(r2max[k]);
            self.dir_w2[k] = w2 * W2_MARGIN;
        }
        self.dir_table_valid = true;
    }

    /// Certify that `neighbor`'s bisector cannot cut the current polygon,
    /// using the candidate's chart direction. `true` means the clipper would
    /// provably return `Unchanged` for this candidate (skipping it is
    /// byte-identical); `false` means "not provably unchanged" — the candidate
    /// must be clipped normally.
    ///
    /// Callers gate on a bounded gnomonic polygon (`can_terminate`-style
    /// preconditions); the table itself is rebuilt lazily after any `Changed`
    /// clip.
    #[inline]
    pub(super) fn directional_reject(&mut self, neighbor: Vec3) -> bool {
        if !self.dir_table_valid {
            // A valid table implies a bounded polygon (any `Changed` clip
            // invalidates it), so this guard only needs to run on rebuild.
            {
                let poly = if self.use_a {
                    &self.poly_a
                } else {
                    &self.poly_b
                };
                if poly.has_bounding_ref() || poly.len < 3 {
                    return false;
                }
            }
            self.rebuild_dir_table();
        }
        let (a, b, c) = self.bisector_coefficients(neighbor);
        // `c <= 0` can never certify: the bisector passes on the generator's
        // side of the chart origin. (A NaN `c` falls through and fails the
        // final `>=` compare.)
        if c <= 0.0 {
            return false;
        }
        // The candidate's chart direction is ∝ (n·t1, n·t2) = (-a, -b).
        let k = dir_window(-a, -b);
        let ab2 = a * a + b * b;
        let certified = c * c >= ab2 * self.dir_w2[k];

        #[cfg(debug_assertions)]
        if certified {
            self.debug_assert_exactly_unchanged(a, b, c);
        }

        certified
    }

    /// Debug cross-check: a certified candidate must pass the clipper's exact
    /// strict all-vertices-inside test for the identical half-plane.
    #[cfg(debug_assertions)]
    fn debug_assert_exactly_unchanged(&self, a: f64, b: f64, c: f64) {
        use crate::knn_clipping::topo2d::types::HalfPlane;
        let hp = HalfPlane::new_unnormalized(a, b, c, self.half_planes.len());
        let poly = if self.use_a {
            &self.poly_a
        } else {
            &self.poly_b
        };
        let neg_eps = -hp.eps;
        for i in 0..poly.len {
            let d = hp.signed_dist(poly.us[i], poly.vs[i]);
            debug_assert!(
                d >= neg_eps,
                "directional certificate skipped a cutting candidate: \
                 vertex {i} signed_dist {d} < {neg_eps} (a={a}, b={b}, c={c})"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dir_window_is_cyclic_over_the_plane() {
        // Walk directions in 1° steps; the window index must be the angle's
        // 45° bucket everywhere except exactly on boundaries (either side ok).
        for deg in 0..360 {
            let phi = (deg as f64).to_radians();
            let (s, c) = phi.sin_cos();
            let k = dir_window(c, s);
            let expected = ((deg / 45) % 8) as usize;
            if deg % 45 == 0 {
                let prev = (expected + 7) % 8;
                assert!(
                    k == expected || k == prev,
                    "deg {deg}: window {k}, expected {expected} or {prev}"
                );
            } else {
                assert_eq!(k, expected, "deg {deg}");
            }
        }
    }
}
