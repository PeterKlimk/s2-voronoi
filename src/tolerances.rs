//! Central registry of the crate's numerical tolerances.
//!
//! Every value here is **empirical**: tuned against the validator, the
//! adversarial corpus, and the coincidence margin probes
//! (`tests/coincidence_probes.rs`), not derived from error analysis. That is
//! a deliberate stance (see `docs/correctness-contract.md`): the topological
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
/// map in `docs/correctness-contract.md`), the worst adversarial
/// construction that breaks the raw pipeline does so at ~1.2e-7 minimum
/// pairwise separation, so this radius carries an order of magnitude of
/// safety margin while staying far below the point spacing of any realistic
/// well-spaced input (~1e-6 spacing requires ~1e13 uniform points).
#[inline]
pub(crate) fn weld_radius() -> f32 {
    coincident_distance()
}

// === Gnomonic clipping (per-cell chart) ===

/// Base inside-slack for half-plane clipping on the SPHERICAL backend,
/// scaled by the plane's normal magnitude at use
/// (`eps = CLIP_EPS_INSIDE * |n|`). **0.0 = the strict antisymmetric keep
/// rule `d >= 0` — the production setting.** The planar backends keep the
/// bias (strict regresses plane perf via slivers); see [`PLANE_CLIP_EPS_INSIDE`].
///
/// The historical value 1e-12 (hex3 import) kept epsilon-grazing vertices
/// inside, biasing both cells of a shared edge toward agreeing a marginal
/// vertex exists. The quad-coherence analysis showed that bias to be the
/// direct cause of tie-regime parity contradictions: for a near-cocircular
/// 4-set, both opposite-parity phrasings of the same in-circle question
/// keep when |d| < eps, defeating the antisymmetry that correlated
/// cross-chart errors otherwise provide. The 2026-06 tie-rule sweep
/// (docs/p5-consistency-design.md, tie-rule findings) measured strict
/// `d >= 0` eliminating the 3M-seed-3 defects (4 -> 0, zero quad
/// contradictions across 8M quads) with no regressions, validity loss, or
/// wall-time change anywhere in the battery. The surviving defects are
/// error-regime contradictions (computed-sign errors at 1.5e-11..4.6e-11,
/// above any tie band) that no local rule can fix; edge checks and
/// reconciliation own those residuals.
pub(crate) const CLIP_EPS_INSIDE: f64 = 0.0;

/// Base inside-slack for half-plane clipping on the PLANAR backends (rect
/// walls and bisectors, bounded and periodic alike): the keep-bias
/// `d >= -eps`. **1e-12 (the bias) is the production setting** — the planar
/// backends deliberately do NOT use the sphere's strict rule. A separate
/// constant from [`CLIP_EPS_INSIDE`] so the two backends diverge here.
///
/// Why the plane keeps the bias and the sphere doesn't: the bias keeps
/// epsilon-grazing corners inside BOTH cells of a shared edge, holding vertex
/// keys/sequences identical, which **suppresses VERTEX-IDENTITY SLIVERS** —
/// the same corner committed under different triple attributions (duplicate
/// ids, zero-length sliver edges). The plane produces these orders denser
/// than the sphere (~1 site per millions), so on the plane they are not rare.
///
/// History: flipped to 0.0 (strict) on 2026-06-14 (`ff3528b`) for parsimony
/// after a 1,362-case correctness campaign came back clean. That flip was
/// **REVERTED on 2026-06-14 (quiet-box perf)**: the campaign validated
/// correctness but never measured perf, and strict is a major REGRESSION —
/// the slivers it admits are detected (uniform 2M: 212 unresolved edges,
/// zero on the bias) and trigger the O(n) `edge_reconcile` machinery on
/// EVERY plane build (~42% of build time, +73% total). The "zero repairs"
/// campaign claim was a misread of the post-repair residual count (0), not
/// the repair WORK. The bias's slivers-guard role is real and load-bearing
/// for plane perf. (A future option to reclaim strict-plane: make
/// `edge_reconcile` O(defects) instead of O(n) per round, so a few slivers
/// cost little; see docs/optimization-ideas.md.) See
/// docs/p5-consistency-design.md.
pub(crate) const PLANE_CLIP_EPS_INSIDE: f64 = 1e-12;

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
/// P5 stage 2a measured (docs/p5-consistency-design.md, stage-2a findings):
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

// === Planar termination (plane_clipping) ===

/// Relative widening of the planar termination threshold `4 * max_r2`.
///
/// A neighbor at squared distance `d2` from the generator cannot cut the cell
/// when `d2 > 4 * max_r2` (its bisector passes at distance `sqrt(d2)/2`,
/// beyond every vertex). The guard absorbs rounding RELATIVE to the
/// quantities compared: f32 squared-distance accumulation in emission and
/// f64 rounding of `max_r2` itself. 3 ulps relative mirrors the sphere's
/// `TERMINATION_THRESHOLD_GUARD` granularity.
///
/// Note: the grid certificate's wall-classification slack is ABSOLUTE
/// (ulps of the wall coordinate, not of `4 * max_r2`) and is therefore
/// handled inside the certificate itself via
/// [`PLANE_WALL_CLASSIFICATION_SLACK`], not here — a relative guard cannot
/// absorb it once cells are much smaller than the domain.
pub(crate) const PLANE_TERMINATION_GUARD: f64 = 3.0 * f32::EPSILON as f64;

/// Relative shrink applied to wall distances inside the plane grid's
/// unseen-distance certificate.
///
/// Point classification (`(p.x * res) as usize`) and the certificate's wall
/// coordinate (`fl(i / res)`) round independently, so a point classified
/// OUTSIDE the explored box can sit up to ~2 ulps of the wall coordinate
/// INSIDE the f32 wall value (one ulp from the classification product, one
/// from the wall division). Shrinking each exposed side distance by
/// `wall * PLANE_WALL_CLASSIFICATION_SLACK` (4 ulps relative — the bound
/// doubled for margin) restores a sound lower bound by construction. Cost:
/// certificates loosen by < 5e-7 absolute, negligible against cell sizes
/// down to ~1e-6 (n ~ 1e12 uniform points).
pub(crate) const PLANE_WALL_CLASSIFICATION_SLACK: f32 = 4.0 * f32::EPSILON;

/// Planar weld radius in normalized domain units (longer rect side = 1):
/// generators closer than this are welded to one cell.
///
/// Empirically required, not input hygiene — the same conclusion as the
/// sphere's weld, re-established for the plane by probing
/// (tests/plane_coincidence_probes.rs + margin probes 2026-06-12): pairs
/// resolve at any distinct-f32 separation (including straddling grid walls
/// and at rect corners), but CLUSTERS (k >= 3) within ~1 ulp of unit scale
/// produce invalid topology (degenerate cells, overused edges) — invalid at
/// min-separation 3e-8, valid from 6e-8 in every probed configuration,
/// including the subnormal-separation regime near the origin. 1e-6 gives
/// ~30x margin over the worst observed failure (the sphere shipped ~8x).
/// Features below this scale exist only for >1e12 uniform points.
pub(crate) const PLANE_WELD_DIST: f32 = 1e-6;

/// Degenerate edge-segment length for the planar edge reconcile post-pass,
/// in normalized domain units (longer rect side = 1).
///
/// Same value and same rationale as the sphere's
/// [`RECONCILE_DEGENERATE_LEN_EPS`], transferred deliberately: planar
/// coordinates are f32 normalized to unit scale, so the f32-lattice band is
/// ~1e-7..1e-6 here too. This governs *output repair* of one-sided
/// epsilon edges (cells that already topologically disagree), not input
/// welding — the planar pipeline welds only normalized-coordinate
/// duplicates.
pub(crate) const PLANE_RECONCILE_DEGENERATE_LEN_EPS: f32 = 1e-6;

/// Neighbor distance below which a planar `ClippedAway` failure is
/// classified as degenerate (near-coincident) input rather than an internal
/// error, in normalized domain units. Same f32-lattice scale as
/// [`PLANE_RECONCILE_DEGENERATE_LEN_EPS`].
pub(crate) const PLANE_COINCIDENT_DIST: f32 = 1e-6;

/// On-wall classification tolerance for planar validation, in normalized
/// domain units (longer rect side = 1). Wall vertices come from f64
/// intersections cast to f32 (~1e-7 relative); 1e-5 is generous headroom
/// while staying far below practical cell sizes.
pub(crate) const PLANE_ON_WALL_EPS: f32 = 1e-5;

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
        // The hierarchy the pipeline relies on: the SPHERE uses the strict
        // antisymmetric keep rule (clipping slack zero), the PLANE keeps the
        // bias (its dense slivers must be suppressed — strict is a perf
        // regression there); the extraction floor sits below the plane bias
        // and below the error-regime contradiction scale (~1e-11).
        assert!(CLIP_EPS_INSIDE == 0.0);
        assert!(PLANE_CLIP_EPS_INSIDE > 0.0);
        assert!((EXTRACT_DEGENERATE_LEN2 as f64).sqrt() < PLANE_CLIP_EPS_INSIDE);
        assert!((EXTRACT_DEGENERATE_LEN2 as f64).sqrt() < 1e-11);
        assert!(CLIP_EPS_INSIDE < MIN_PROJECTION_COS);
        assert!(PLANE_CLIP_EPS_INSIDE < MIN_PROJECTION_COS);
        assert!(RECONCILE_DEGENERATE_LEN_EPS <= weld_radius());
        assert!(FALLBACK_DEDUP_DOT < 1.0);
    }
}
