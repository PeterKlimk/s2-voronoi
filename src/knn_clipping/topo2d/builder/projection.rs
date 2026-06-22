use super::{
    BuilderImpl, FallbackBuilder, FallbackConstraint, GnomonicBuilder, PolyBuffer, Topo2DBuilder,
};
use crate::fp;
use crate::knn_clipping::cell_build::CellFailure;
use glam::{DVec3, Vec3};
use std::hint::select_unpredictable;

#[inline(always)]
fn cswap_u32(a: &mut u32, b: &mut u32) {
    let va = *a;
    let vb = *b;
    let cond = va <= vb;
    *a = select_unpredictable(cond, va, vb);
    *b = select_unpredictable(cond, vb, va);
}

#[inline(always)]
pub(crate) fn sort3_u32(a: u32, b: u32, c: u32) -> [u32; 3] {
    // Sorting network (3 elements): (0,1) (1,2) (0,1)
    let mut x0 = a;
    let mut x1 = b;
    let mut x2 = c;
    cswap_u32(&mut x0, &mut x1);
    cswap_u32(&mut x1, &mut x2);
    cswap_u32(&mut x0, &mut x1);
    [x0, x1, x2]
}

/// Orthonormal tangent basis for gnomonic projection.
pub struct TangentBasis {
    pub t1: DVec3,
    pub t2: DVec3,
    pub g: DVec3,
}

impl TangentBasis {
    pub fn new(g: DVec3) -> Self {
        let len_sq = g.length_squared();
        if len_sq == 0.0 || !len_sq.is_finite() {
            return TangentBasis {
                t1: DVec3::X,
                t2: DVec3::Y,
                g,
            };
        }

        let (mut t1, mut t2) = if g.z < -0.999_999_9 {
            (DVec3::NEG_Y, DVec3::NEG_X)
        } else {
            let a = 1.0 / (1.0 + g.z);
            let b = -g.x * g.y * a;
            (
                DVec3::new(1.0 - g.x * g.x * a, b, -g.x),
                DVec3::new(b, 1.0 - g.y * g.y * a, -g.y),
            )
        };
        // The closed-form ONB assumes |g| == 1, but production keeps the
        // exact f32-promoted generator. Project out the tiny radial component
        // so the chart axes are tangent to that promoted point set.
        let inv_len_sq = len_sq.recip();
        t1 -= g * (t1.dot(g) * inv_len_sq);
        t2 -= g * (t2.dot(g) * inv_len_sq);
        TangentBasis { t1, t2, g }
    }
}

pub(super) use crate::tolerances::MIN_PROJECTION_COS;

impl FallbackConstraint {
    #[inline]
    pub(super) fn from_neighbor(
        generator: DVec3,
        neighbor_idx: usize,
        neighbor_slot: u32,
        hp_eps: Option<f32>,
        neighbor: Vec3,
    ) -> Self {
        // The fallback builder is a separate algorithm (ProjectionLimit
        // path) whose plane math expects unit vectors; it keeps the legacy
        // f64 normalization of both sides (P5 canonicalization of this path
        // is deferred; see p5-consistency-design.md).
        let neighbor =
            DVec3::new(neighbor.x as f64, neighbor.y as f64, neighbor.z as f64).normalize();
        Self {
            normal: generator.normalize() - neighbor,
            neighbor_idx,
            neighbor_slot,
            hp_eps,
        }
    }
}

impl GnomonicBuilder {
    pub(super) fn new(generator_idx: usize, generator: Vec3) -> Self {
        #[cfg(feature = "p5_shadow")]
        let angle_pad = crate::knn_clipping::p5_shadow::term_pad_override()
            .unwrap_or(crate::tolerances::TERMINATION_ANGLE_PAD);
        #[cfg(not(feature = "p5_shadow"))]
        let angle_pad = crate::tolerances::TERMINATION_ANGLE_PAD;
        let (term_sin_pad, term_cos_pad) = angle_pad.sin_cos();
        // P5 stage 0: promote the canonicalized f32 bits exactly — no
        // renormalization (the old per-builder normalize made each chart
        // solve a ~1-ulp-perturbed point set; see p5-consistency-design.md).
        let gen64 = DVec3::new(generator.x as f64, generator.y as f64, generator.z as f64);
        let inv_two_gg = 0.5 / gen64.length_squared();
        let basis = TangentBasis::new(gen64);
        let generator_dot_g = gen64.dot(basis.g);

        let mut poly_a = PolyBuffer::new();
        poly_a.init_bounding(1e6);

        Self {
            generator_idx,
            generator: gen64,
            basis,
            inv_two_gg,
            generator_dot_g,
            half_planes: Vec::with_capacity(32),
            neighbor_indices: Vec::with_capacity(32),
            neighbor_slots: Vec::with_capacity(32),
            poly_a,
            poly_b: PolyBuffer::new(),
            use_a: true,
            failed: None,
            generator_raw: generator,
            #[cfg(feature = "p5_shadow")]
            neighbor_positions_raw: Vec::with_capacity(32),
            term_sin_pad,
            term_cos_pad,
            term_threshold_cache: 0.0,
            term_cache_valid: false,
            #[cfg(feature = "timing")]
            support_cache_valid: false,
            #[cfg(feature = "timing")]
            support_min_proj: [0.0; 64],
            #[cfg(feature = "escalate_probe")]
            proactive_min_transition_delta: None,
            #[cfg(feature = "escalate_probe")]
            proactive_min_early_unchanged_clearance: None,
        }
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn reset(&mut self, generator_idx: usize, generator: Vec3) {
        // P5 stage 0: exact f32 bits, no renormalization (see new()).
        let gen64 = DVec3::new(generator.x as f64, generator.y as f64, generator.z as f64);
        self.generator_idx = generator_idx;
        self.generator = gen64;
        self.inv_two_gg = 0.5 / gen64.length_squared();
        self.basis = TangentBasis::new(gen64);
        self.generator_dot_g = gen64.dot(self.basis.g);
        self.half_planes.clear();
        self.neighbor_indices.clear();
        self.neighbor_slots.clear();
        self.generator_raw = generator;
        #[cfg(feature = "p5_shadow")]
        self.neighbor_positions_raw.clear();
        self.poly_a.init_bounding(1e6);
        self.poly_b.clear();
        self.use_a = true;
        self.failed = None;
        self.term_cache_valid = false;
        #[cfg(feature = "timing")]
        {
            self.support_cache_valid = false;
        }
        #[cfg(feature = "escalate_probe")]
        {
            self.proactive_min_transition_delta = None;
            self.proactive_min_early_unchanged_clearance = None;
        }
    }

    /// Compute the bisector half-plane coefficients (a, b, c) for a neighbor.
    #[inline]
    pub(super) fn bisector_coefficients(&self, neighbor: Vec3) -> (f64, f64, f64) {
        debug_assert!(
            (neighbor.length_squared() - 1.0).abs() < 1e-5,
            "neighbor not unit-normalized: |N|² = {}",
            neighbor.length_squared()
        );

        let n_raw = DVec3::new(neighbor.x as f64, neighbor.y as f64, neighbor.z as f64);
        let len_sq = n_raw.length_squared();
        // Exact chord bisector of two f32-canonicalized points promoted to
        // f64. Do not collapse this to the unit-vector formula: the promoted
        // f32 inputs are slightly off-unit, and that correction is what keeps
        // near-twin bisectors in the same solved point set as the rest of the
        // pipeline.
        //
        // The plane value at chart points needs c = (|g|^2 + |h|^2)/2 - g.h,
        // achieved by w = g * (|g|^2 + |h|^2) / (2|g|^2) - h dotted with
        // (t1, t2, g). Cache `g`'s projection onto the chart normal so each
        // clip only projects the neighbor and applies the same scale algebra.
        let scale = fp::fma_f64(len_sq, self.inv_two_gg, 0.5);
        (
            -n_raw.dot(self.basis.t1),
            -n_raw.dot(self.basis.t2),
            fp::fma_f64(scale, self.generator_dot_g, -n_raw.dot(self.basis.g)),
        )
    }

    #[inline]
    pub(super) fn current_poly(&self) -> &PolyBuffer {
        if self.use_a {
            &self.poly_a
        } else {
            &self.poly_b
        }
    }

    #[inline]
    pub(super) fn is_bounded(&self) -> bool {
        !self.current_poly().has_bounding_ref()
    }

    #[inline]
    pub(super) fn is_failed(&self) -> bool {
        self.failed.is_some()
    }

    #[inline]
    pub(super) fn failure(&self) -> Option<crate::knn_clipping::cell_build::CellFailure> {
        self.failed
    }

    #[inline]
    pub(super) fn vertex_count(&self) -> usize {
        self.current_poly().len
    }

    #[inline]
    pub(super) fn neighbor_indices_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbor_indices.iter().copied()
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn can_terminate(&mut self, max_unseen_dot_bound: f32) -> bool {
        self.termination_clearance(max_unseen_dot_bound)
            .is_some_and(|clearance| clearance > 0.0)
    }

    pub(super) fn termination_clearance(&mut self, max_unseen_dot_bound: f32) -> Option<f64> {
        if !self.is_bounded() || self.vertex_count() < 3 {
            return None;
        }

        if !self.term_cache_valid {
            let min_cos = self.current_poly().min_cos();
            if min_cos <= 0.0 || min_cos > 1.0 {
                return None;
            }

            let sin_theta = (1.0 - min_cos * min_cos).max(0.0).sqrt();
            let cos_theta_pad =
                fp::fma_f64(min_cos, self.term_cos_pad, -sin_theta * self.term_sin_pad);
            let cos_2max = fp::fma_f64(2.0 * cos_theta_pad, cos_theta_pad, -1.0);
            self.term_threshold_cache = cos_2max - crate::tolerances::TERMINATION_THRESHOLD_GUARD;
            self.term_cache_valid = true;
        }

        Some(self.term_threshold_cache - max_unseen_dot_bound as f64)
    }
}

impl FallbackBuilder {
    #[inline]
    pub(super) fn is_bounded(&self) -> bool {
        self.computed_vertex_count() >= 3
    }

    #[inline]
    pub(super) fn is_failed(&self) -> bool {
        false
    }

    #[inline]
    pub(super) fn failure(&self) -> Option<CellFailure> {
        None
    }

    #[inline]
    pub(super) fn vertex_count(&self) -> usize {
        self.computed_vertex_count()
    }

    #[inline]
    pub(super) fn neighbor_indices_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.constraints
            .iter()
            .map(|constraint| constraint.neighbor_idx)
    }

    #[inline]
    pub(super) fn can_terminate(&mut self, _max_unseen_dot_bound: f32) -> bool {
        false
    }

    pub(super) fn termination_clearance(&mut self, _max_unseen_dot_bound: f32) -> Option<f64> {
        None
    }
}

impl Topo2DBuilder {
    pub fn new(generator_idx: usize, generator: Vec3) -> Self {
        Self {
            inner: BuilderImpl::Gnomonic(GnomonicBuilder::new(generator_idx, generator)),
        }
    }

    #[cold]
    #[inline(never)]
    fn reset_from_fallback(&mut self, generator_idx: usize, generator: Vec3) {
        self.inner = BuilderImpl::Gnomonic(GnomonicBuilder::new(generator_idx, generator));
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn reset(&mut self, generator_idx: usize, generator: Vec3) {
        if let BuilderImpl::Gnomonic(builder) = &mut self.inner {
            builder.reset(generator_idx, generator);
        } else {
            self.reset_from_fallback(generator_idx, generator);
        }
    }

    #[inline]
    pub fn is_bounded(&self) -> bool {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.is_bounded(),
            BuilderImpl::Fallback(builder) => builder.is_bounded(),
        }
    }

    #[inline]
    pub fn is_failed(&self) -> bool {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.is_failed(),
            BuilderImpl::Fallback(builder) => builder.is_failed(),
        }
    }

    #[inline]
    pub fn failure(&self) -> Option<crate::knn_clipping::cell_build::CellFailure> {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.failure(),
            BuilderImpl::Fallback(builder) => builder.failure(),
        }
    }

    #[inline]
    pub fn vertex_count(&self) -> usize {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => builder.vertex_count(),
            BuilderImpl::Fallback(builder) => builder.vertex_count(),
        }
    }

    #[inline]
    pub fn neighbor_indices_iter(&self) -> impl Iterator<Item = usize> + '_ {
        let iter: Box<dyn Iterator<Item = usize> + '_> = match &self.inner {
            BuilderImpl::Gnomonic(builder) => Box::new(builder.neighbor_indices_iter()),
            BuilderImpl::Fallback(builder) => Box::new(builder.neighbor_indices_iter()),
        };
        iter
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn can_terminate(&mut self, max_unseen_dot_bound: f32) -> bool {
        match &mut self.inner {
            BuilderImpl::Gnomonic(builder) => builder.can_terminate(max_unseen_dot_bound),
            BuilderImpl::Fallback(builder) => builder.can_terminate(max_unseen_dot_bound),
        }
    }

    pub fn termination_clearance(&mut self, max_unseen_dot_bound: f32) -> Option<f64> {
        match &mut self.inner {
            BuilderImpl::Gnomonic(builder) => builder.termination_clearance(max_unseen_dot_bound),
            BuilderImpl::Fallback(builder) => builder.termination_clearance(max_unseen_dot_bound),
        }
    }

    #[cfg(feature = "escalate_probe")]
    pub fn proactive_clip_metrics(&self) -> (Option<f64>, Option<f64>) {
        match &self.inner {
            BuilderImpl::Gnomonic(builder) => (
                builder.proactive_min_transition_delta,
                builder.proactive_min_early_unchanged_clearance,
            ),
            BuilderImpl::Fallback(_) => (None, None),
        }
    }
}
