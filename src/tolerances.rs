//! Central registry of the crate's numerical tolerances.
//!
//! Every value here is **empirical**: tuned against the validator, the
//! adversarial corpus, and the coincidence margin probes
//! (`tests/coincidence_probes.rs`), not derived from error analysis. That is
//! a deliberate stance (see `docs/correctness.md`): the topological
//! contract is enforced by strict validation over the supported envelope,
//! and these constants are implementation detail behind it. The margin data
//! backing the envelope: the worst adversarial construction that defeats the
//! raw pipeline does so at ~1.2e-7 chord separation; the weld radius below
//! carries ~10x margin over it.
//!
//! Grouped by pipeline stage. When changing any value, re-run the
//! coincidence probes to detect resolvability-boundary drift, and the
//! adversarial + fuzz suites.

// === Welding (coincident-generator preprocessing) ===

/// Treat generators as coincident when their dot product differs from 1 by
/// this amount. For unit vectors, `1 - dot ≈ (distance^2) / 2`; derived from
/// f32 rounding on normalized inputs with a small safety multiplier. This is
/// only for duplicate-generator handling; it does not address vertex
/// degeneracy.
pub(crate) const COINCIDENT_DOT_TOL: f32 = 64.0 * f32::EPSILON * f32::EPSILON;

/// Squared Euclidean distance threshold for coincident generators.
#[inline]
pub(crate) const fn coincident_distance_sq() -> f32 {
    2.0 * COINCIDENT_DOT_TOL
}

/// Euclidean distance threshold for coincident generators.
#[inline]
pub(crate) fn coincident_distance() -> f32 {
    coincident_distance_sq().sqrt()
}

/// Default weld radius (chord distance) for `PreprocessMode::Weld`.
///
/// Generators closer than this are welded into one cell before construction.
/// The value is the coincident-distance floor (~1.4e-6). Empirically (margin
/// map in `docs/correctness.md`), the worst adversarial
/// construction that breaks the raw pipeline does so at ~1.2e-7 minimum
/// pairwise separation, so this radius carries an order of magnitude of
/// safety margin while staying far below the point spacing of any realistic
/// well-spaced input (~1e-6 spacing requires ~1e13 uniform points).
#[inline]
pub(crate) fn weld_radius() -> f32 {
    coincident_distance()
}

// === Gnomonic clipping (per-cell chart) ===

/// Base inside-slack for half-plane clipping, scaled by the plane's normal
/// magnitude at use (`eps = CLIP_EPS_INSIDE * |n|`). **0.0 = the strict
/// antisymmetric keep rule `d >= 0` — the production setting.**
///
/// The historical value 1e-12 (hex3 import) kept epsilon-grazing vertices
/// inside, biasing both cells of a shared edge toward agreeing a marginal
/// vertex exists. The quad-coherence analysis showed that bias to be the
/// direct cause of tie-regime parity contradictions: for a near-cocircular
/// 4-set, both opposite-parity phrasings of the same in-circle question
/// keep when |d| < eps, defeating the antisymmetry that correlated
/// cross-chart errors otherwise provide. The 2026-06 tie-rule sweep
/// measured strict
/// `d >= 0` eliminating the 3M-seed-3 defects (4 -> 0, zero quad
/// contradictions across 8M quads) with no regressions, validity loss, or
/// wall-time change anywhere in the battery. The surviving defects are
/// error-regime contradictions (computed-sign errors at 1.5e-11..4.6e-11,
/// above any tie band) that no local rule can fix; edge checks and
/// reconciliation own those residuals.
pub(crate) const CLIP_EPS_INSIDE: f64 = 0.0;

/// Chart-validity floor: once the clipped polygon's minimum generator-dot
/// reaches this, the feasible region is effectively at the generator's
/// hemisphere boundary and the gnomonic model hands off to the spherical
/// fallback. Sized to f32 input granularity (inputs are f32; 8 ulps of
/// slack), conservative for the f64 chart math.
pub(crate) const MIN_PROJECTION_COS: f64 = 8.0 * f32::EPSILON as f64;

/// Canonical-escalation band width as a multiple of a half-plane's stored
/// eps (`CLIP_EPS_INSIDE * |n|`). **0.0 = escalation disabled — the
/// production setting.**
///
/// P5 stage 2a measured:
/// margin-gated escalation is unsound at ANY width, because the gate is
/// evaluated per cell on displaced computed margins (the same question's
/// margins differ ~100x across cells; local answers below ~1e-4 are ~50%
/// anti-correlated with canonical while near-perfectly correlated with
/// each other). Mixing canonical answers into the correlated-local system
/// manufactures cross-cell contradictions at the band boundary: factor
/// 1e4 turned 0/4 natural defects into 10/200 (500k/3M); factors 16..128
/// fixed the three watched sites but elevated 4.5M from 3 to 6-11. The
/// clipper machinery is kept (probe-overridable via
/// `p5_shadow::set_escalation_factor_override`) as the test rig for
/// successor designs (question-intrinsic gating / antisymmetric tie rule).
/// Note: with `CLIP_EPS_INSIDE = 0.0` the band `factor * hp.eps` is zero
/// regardless of factor; escalation experiments must also set
/// `p5_shadow::set_clip_eps_override` to a nonzero eps.
pub(crate) const CLIP_ESCALATION_FACTOR: f64 = 0.0;

/// Angular padding folded into the termination certificate (the polygon's
/// angular extent is widened by this before comparing against unseen-dot
/// bounds). Same f32-granularity scale as `MIN_PROJECTION_COS`.
pub(crate) const TERMINATION_ANGLE_PAD: f64 = 8.0 * f32::EPSILON as f64;

/// Absolute guard subtracted from the cached termination threshold, covering
/// rounding in the double-angle computation of the certificate itself.
pub(crate) const TERMINATION_THRESHOLD_GUARD: f64 = 3.0 * f32::EPSILON as f64;

/// Norm envelope for a point normalized in f64 and rounded once to f32 by
/// `canonicalize_unit_points`. Componentwise roundoff gives a tighter bound;
/// one f32 epsilon is retained here so the termination certificate can map a
/// normalized angular threshold back into the raw-dot space used by kNN.
pub(crate) const CANONICAL_UNIT_NORM_SLACK: f64 = f32::EPSILON as f64;

// === Spherical fallback (constraint replay past the chart limit) ===

/// Constraint-satisfaction slack for fallback vertex candidates (a candidate
/// direction must satisfy every half-space to within this dot tolerance).
/// Absolute, in dot units: ~1e-9 admits vertices displaced by ~1e-9 rad from
/// exact constraint intersections, far below the f32 output quantization.
pub(crate) const FALLBACK_PLANE_TOL: f64 = 1e-9;

/// Fallback vertex dedup: candidate directions with mutual dot above this
/// (within ~3e-3 rad) collapse to one vertex. Coarser than the clipping
/// tolerances on purpose - fallback cells sit at the hemisphere boundary
/// where intersection conditioning is poor.
pub(crate) const FALLBACK_DEDUP_DOT: f32 = 1.0 - 1e-5;

/// Squared-length floor for reconstructed vertex directions; rejects only
/// catastrophically cancelled directions (|v| < 1e-14) as
/// degenerate-extraction failures rather than normalizing noise.
pub(crate) const EXTRACT_DEGENERATE_LEN2: f32 = 1e-28;

// === Edge reconciliation (post-assembly repair) ===

/// Scale below which a reconciled edge's endpoints are considered the same
/// point (epsilon-edge collapse). Sits in the f32-lattice band at unit scale
/// (~1e-7..1e-6), i.e. the same family as the weld radius: features below
/// this are sub-resolvability and their collapse is the documented policy
/// for epsilon-scale geometry.
pub(crate) const RECONCILE_DEGENERATE_LEN_EPS: f32 = 1e-6;

// === Cube-grid conservative bounds ===

/// Slack subtracted from security-plane distances when deriving per-query
/// packed thresholds, covering f32 rounding in the plane evaluation so the
/// "safe" classification stays conservative.
pub(crate) const GRID_PLANE_PAD: f32 = 1e-6;

/// Inflation applied to per-cell cap sin-radii (scaled by the center dot)
/// when precomputing conservative cell bounds, covering rounding in the
/// bound construction.
pub(crate) const GRID_SIN_EPS: f32 = 1e-5;

/// Absolute inflation on dot bounds exported by the kNN stream. Some bounds
/// are spherical-cosine/chord bounds while the stream ranks the once-rounded
/// canonical f32 points by raw f32 dot. The norm conversion plus dot/bound
/// arithmetic can otherwise leave a theoretical frontier one ulp below an
/// unseen raw dot (caught by `nn_contract_dense_single_cell`). Four ulps keep
/// the public internal contract genuinely upper-bounding, rather than relying
/// on the termination certificate's separate guard.
pub(crate) const GRID_DOT_BOUND_PAD: f32 = 4.0 * f32::EPSILON;

// === Validation (exact-invariant checks) ===

/// On-sphere check for output vertices. Loose relative to working precision
/// because vertices pass through f32 storage and fallback-path
/// normalization; this bounds representation error, not algorithm error.
pub(crate) const VERTEX_ON_SPHERE_EPS: f32 = 1e-4;

/// Edge endpoints with dot below `-1 + this` are flagged antipodal (an
/// edge's great-circle arc is ill-defined near antipodality).
pub(crate) const ANTIPODAL_DOT_EPS: f32 = 1e-5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weld_radius_band() {
        let r = weld_radius();
        assert!(
            (1.0e-6..2.0e-6).contains(&r),
            "weld radius {r} should sit in the ~1e-6 band backed by the margin data"
        );
        assert!(r >= coincident_distance());
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // pinning the constant hierarchy is the point
    fn tolerance_ordering_sanity() {
        // The hierarchy the pipeline relies on: the strict antisymmetric
        // keep rule (clipping slack zero); the extraction floor sits below
        // the error-regime contradiction scale (~1e-11).
        assert!(CLIP_EPS_INSIDE == 0.0);
        assert!((EXTRACT_DEGENERATE_LEN2 as f64).sqrt() < 1e-11);
        assert!(CLIP_EPS_INSIDE < MIN_PROJECTION_COS);
        assert!(RECONCILE_DEGENERATE_LEN_EPS <= weld_radius());
        assert!(FALLBACK_DEDUP_DOT < 1.0);
    }
}
