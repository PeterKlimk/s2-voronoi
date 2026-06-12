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
        let arbitrary = if g.x.abs() <= g.y.abs() && g.x.abs() <= g.z.abs() {
            DVec3::X
        } else if g.y.abs() <= g.z.abs() {
            DVec3::Y
        } else {
            DVec3::Z
        };
        let t1 = g.cross(arbitrary).normalize();
        let t2 = g.cross(t1);
        TangentBasis { t1, t2, g }
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    #[cfg_attr(not(feature = "profiling"), inline)]
    pub fn plane_to_line(&self, n: DVec3) -> (f64, f64, f64) {
        (n.dot(self.t1), n.dot(self.t2), n.dot(self.g))
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

        let mut poly_a = PolyBuffer::new();
        poly_a.init_bounding(1e6);

        Self {
            generator_idx,
            generator: gen64,
            basis,
            inv_two_gg,
            half_planes: Vec::with_capacity(32),
            neighbor_indices: Vec::with_capacity(32),
            neighbor_slots: Vec::with_capacity(32),
            poly_a,
            poly_b: PolyBuffer::new(),
            use_a: true,
            failed: None,
            generator_raw: generator,
            neighbor_positions_raw: Vec::with_capacity(32),
            term_sin_pad,
            term_cos_pad,
            term_threshold_cache: 0.0,
            term_cache_valid: false,
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
        self.half_planes.clear();
        self.neighbor_indices.clear();
        self.neighbor_slots.clear();
        self.generator_raw = generator;
        self.neighbor_positions_raw.clear();
        self.poly_a.init_bounding(1e6);
        self.poly_b.clear();
        self.use_a = true;
        self.failed = None;
        self.term_cache_valid = false;
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
        // Exact chord bisector of two non-unit points: the plane value at
        // chart points needs c = (|g|^2 + |h|^2)/2 - g.h, achieved by
        // w = g * (|g|^2 + |h|^2) / (2|g|^2) - h dotted with (t1, t2, g)
        // (t1, t2 are exactly orthogonal to g, so a and b stay -t.h). For
        // |g| exactly 1 this is the legacy scale = (|h|^2 + 1)/2,
        // bit-for-bit. Without the |g|^2 correction, a ~1e-7 off-unit
        // generator displaces a 2e-6-separation twin's bisector by ~3% of
        // the chart (c error ~delta_g amplified by 1/|n|) — the
        // resolvable-regime killer the contract tests catch.
        let scale = fp::fma_f64(len_sq, self.inv_two_gg, 0.5);

        let g = self.generator;
        let normal_unnorm = DVec3::new(
            fp::fma_f64(g.x, scale, -n_raw.x),
            fp::fma_f64(g.y, scale, -n_raw.y),
            fp::fma_f64(g.z, scale, -n_raw.z),
        );

        self.basis.plane_to_line(normal_unnorm)
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
        if !self.is_bounded() || self.vertex_count() < 3 {
            return false;
        }

        if !self.term_cache_valid {
            let min_cos = self.current_poly().min_cos();
            if min_cos <= 0.0 || min_cos > 1.0 {
                return false;
            }

            let sin_theta = (1.0 - min_cos * min_cos).max(0.0).sqrt();
            let cos_theta_pad =
                fp::fma_f64(min_cos, self.term_cos_pad, -sin_theta * self.term_sin_pad);
            let cos_2max = fp::fma_f64(2.0 * cos_theta_pad, cos_theta_pad, -1.0);
            self.term_threshold_cache = cos_2max - crate::tolerances::TERMINATION_THRESHOLD_GUARD;
            self.term_cache_valid = true;
        }

        (max_unseen_dot_bound as f64) < self.term_threshold_cache
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
}

impl Topo2DBuilder {
    pub fn new(generator_idx: usize, generator: Vec3) -> Self {
        Self {
            inner: BuilderImpl::Gnomonic(GnomonicBuilder::new(generator_idx, generator)),
        }
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn reset(&mut self, generator_idx: usize, generator: Vec3) {
        match &mut self.inner {
            BuilderImpl::Gnomonic(builder) => builder.reset(generator_idx, generator),
            BuilderImpl::Fallback(_) => {
                self.inner = BuilderImpl::Gnomonic(GnomonicBuilder::new(generator_idx, generator));
            }
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
}
