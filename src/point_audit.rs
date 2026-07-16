//! Profiling-only attribution for stored/public spherical-direction envelopes.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const EPS: f64 = f32::EPSILON as f64;

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub(crate) enum PointProducer {
    CanonicalGenerator,
    GnomonicVertex,
    FallbackVertex,
    RepairVertex,
    FinalGenerator,
    FinalVertex,
    Centroid,
    EmbeddingProjection,
    CellMeshVertex,
    CellMeshSourceSite,
}

const PRODUCER_COUNT: usize = 10;

impl PointProducer {
    const ALL: [Self; PRODUCER_COUNT] = [
        Self::CanonicalGenerator,
        Self::GnomonicVertex,
        Self::FallbackVertex,
        Self::RepairVertex,
        Self::FinalGenerator,
        Self::FinalVertex,
        Self::Centroid,
        Self::EmbeddingProjection,
        Self::CellMeshVertex,
        Self::CellMeshSourceSite,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::CanonicalGenerator => "canonical_generator",
            Self::GnomonicVertex => "gnomonic_vertex",
            Self::FallbackVertex => "fallback_vertex",
            Self::RepairVertex => "repair_vertex",
            Self::FinalGenerator => "final_generator",
            Self::FinalVertex => "final_vertex",
            Self::Centroid => "centroid",
            Self::EmbeddingProjection => "embedding_projection",
            Self::CellMeshVertex => "cell_mesh_vertex",
            Self::CellMeshSourceSite => "cell_mesh_source_site",
        }
    }
}

struct AtomicEnvelope {
    count: AtomicU64,
    non_finite: AtomicU64,
    over_1eps: AtomicU64,
    over_2eps: AtomicU64,
    over_4eps: AtomicU64,
    over_8eps: AtomicU64,
    over_1e6: AtomicU64,
    over_1e5: AtomicU64,
    over_1e4: AtomicU64,
    max_abs_error_bits: AtomicU64,
    f32_rule_changed: AtomicU64,
    f64_rule_changed: AtomicU64,
    rules_differ: AtomicU64,
    f32_rule_max_abs_error_bits: AtomicU64,
    f64_rule_max_abs_error_bits: AtomicU64,
}

impl AtomicEnvelope {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            non_finite: AtomicU64::new(0),
            over_1eps: AtomicU64::new(0),
            over_2eps: AtomicU64::new(0),
            over_4eps: AtomicU64::new(0),
            over_8eps: AtomicU64::new(0),
            over_1e6: AtomicU64::new(0),
            over_1e5: AtomicU64::new(0),
            over_1e4: AtomicU64::new(0),
            max_abs_error_bits: AtomicU64::new(0),
            f32_rule_changed: AtomicU64::new(0),
            f64_rule_changed: AtomicU64::new(0),
            rules_differ: AtomicU64::new(0),
            f32_rule_max_abs_error_bits: AtomicU64::new(0),
            f64_rule_max_abs_error_bits: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.non_finite.store(0, Ordering::Relaxed);
        self.over_1eps.store(0, Ordering::Relaxed);
        self.over_2eps.store(0, Ordering::Relaxed);
        self.over_4eps.store(0, Ordering::Relaxed);
        self.over_8eps.store(0, Ordering::Relaxed);
        self.over_1e6.store(0, Ordering::Relaxed);
        self.over_1e5.store(0, Ordering::Relaxed);
        self.over_1e4.store(0, Ordering::Relaxed);
        self.max_abs_error_bits.store(0, Ordering::Relaxed);
        self.f32_rule_changed.store(0, Ordering::Relaxed);
        self.f64_rule_changed.store(0, Ordering::Relaxed);
        self.rules_differ.store(0, Ordering::Relaxed);
        self.f32_rule_max_abs_error_bits.store(0, Ordering::Relaxed);
        self.f64_rule_max_abs_error_bits.store(0, Ordering::Relaxed);
    }

    fn record(&self, current: [f32; 3], source: [f64; 3]) {
        self.record_candidates(
            current,
            normalize_f32_then_store(source),
            normalize_f64_then_store(source),
        );
    }

    fn record_f64_canonical(&self, current: [f32; 3]) {
        self.record_candidates(
            current,
            normalize_f32_then_store([current[0] as f64, current[1] as f64, current[2] as f64]),
            current,
        );
    }

    fn record_candidates(&self, current: [f32; 3], f32_rule: [f32; 3], f64_rule: [f32; 3]) {
        self.count.fetch_add(1, Ordering::Relaxed);
        let error = stored_error(current);
        if !error.is_finite() {
            self.non_finite.fetch_add(1, Ordering::Relaxed);
            return;
        }
        self.over_1eps
            .fetch_add(u64::from(error > EPS), Ordering::Relaxed);
        self.over_2eps
            .fetch_add(u64::from(error > 2.0 * EPS), Ordering::Relaxed);
        self.over_4eps
            .fetch_add(u64::from(error > 4.0 * EPS), Ordering::Relaxed);
        self.over_8eps
            .fetch_add(u64::from(error > 8.0 * EPS), Ordering::Relaxed);
        self.over_1e6
            .fetch_add(u64::from(error > 1.0e-6), Ordering::Relaxed);
        self.over_1e5
            .fetch_add(u64::from(error > 1.0e-5), Ordering::Relaxed);
        self.over_1e4
            .fetch_add(u64::from(error > 1.0e-4), Ordering::Relaxed);
        self.max_abs_error_bits
            .fetch_max(error.to_bits(), Ordering::Relaxed);

        self.f32_rule_changed.fetch_add(
            u64::from(stored_bits(f32_rule) != stored_bits(current)),
            Ordering::Relaxed,
        );
        self.f64_rule_changed.fetch_add(
            u64::from(stored_bits(f64_rule) != stored_bits(current)),
            Ordering::Relaxed,
        );
        self.rules_differ.fetch_add(
            u64::from(stored_bits(f32_rule) != stored_bits(f64_rule)),
            Ordering::Relaxed,
        );
        let f32_error = stored_error(f32_rule);
        let f64_error = stored_error(f64_rule);
        if f32_error.is_finite() {
            self.f32_rule_max_abs_error_bits
                .fetch_max(f32_error.to_bits(), Ordering::Relaxed);
        }
        if f64_error.is_finite() {
            self.f64_rule_max_abs_error_bits
                .fetch_max(f64_error.to_bits(), Ordering::Relaxed);
        }
    }
}

#[inline]
fn stored_bits(point: [f32; 3]) -> [u32; 3] {
    [point[0].to_bits(), point[1].to_bits(), point[2].to_bits()]
}

#[inline]
fn stored_error(point: [f32; 3]) -> f64 {
    let x = point[0] as f64;
    let y = point[1] as f64;
    let z = point[2] as f64;
    (x * x + y * y + z * z - 1.0).abs()
}

#[inline]
fn normalize_f32_then_store(source: [f64; 3]) -> [f32; 3] {
    let point = glam::Vec3::new(source[0] as f32, source[1] as f32, source[2] as f32);
    let normalized = point * point.length_squared().sqrt().recip();
    [normalized.x, normalized.y, normalized.z]
}

#[inline]
fn normalize_f64_then_store(source: [f64; 3]) -> [f32; 3] {
    let point = glam::DVec3::from_array(source);
    let normalized = point / point.length_squared().sqrt();
    [
        normalized.x as f32,
        normalized.y as f32,
        normalized.z as f32,
    ]
}

static ENVELOPES: [AtomicEnvelope; PRODUCER_COUNT] =
    [const { AtomicEnvelope::new() }; PRODUCER_COUNT];
static ENABLED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug)]
pub struct PointEnvelopeSummary {
    pub producer: &'static str,
    pub count: u64,
    pub non_finite: u64,
    pub over_1eps: u64,
    pub over_2eps: u64,
    pub over_4eps: u64,
    pub over_8eps: u64,
    pub over_1e6: u64,
    pub over_1e5: u64,
    pub over_1e4: u64,
    pub max_abs_error: f64,
    pub f32_rule_changed: u64,
    pub f64_rule_changed: u64,
    pub rules_differ: u64,
    pub f32_rule_max_abs_error: f64,
    pub f64_rule_max_abs_error: f64,
}

pub(crate) fn reset() {
    for envelope in &ENVELOPES {
        envelope.reset();
    }
    ENABLED.store(true, Ordering::Relaxed);
}

pub(crate) fn record_xyz(producer: PointProducer, x: f32, y: f32, z: f32) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    ENVELOPES[producer as usize].record([x, y, z], [x as f64, y as f64, z as f64]);
}

pub(crate) fn record_vec3(producer: PointProducer, point: glam::Vec3) {
    record_xyz(producer, point.x, point.y, point.z);
}

pub(crate) fn record_vec3_from_dvec3(
    producer: PointProducer,
    current: glam::Vec3,
    source: glam::DVec3,
) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    ENVELOPES[producer as usize].record(current.to_array(), source.to_array());
}

pub(crate) fn record_sphere_point_f64_canonical(
    producer: PointProducer,
    current: crate::SpherePoint,
) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    ENVELOPES[producer as usize].record_f64_canonical(current.to_array());
}

pub(crate) fn record_sphere_point(producer: PointProducer, point: crate::SpherePoint) {
    let [x, y, z] = point.to_array();
    record_xyz(producer, x, y, z);
}

pub(crate) fn snapshot() -> Vec<PointEnvelopeSummary> {
    ENABLED.store(false, Ordering::Relaxed);
    PointProducer::ALL
        .into_iter()
        .map(|producer| {
            let envelope = &ENVELOPES[producer as usize];
            PointEnvelopeSummary {
                producer: producer.name(),
                count: envelope.count.load(Ordering::Relaxed),
                non_finite: envelope.non_finite.load(Ordering::Relaxed),
                over_1eps: envelope.over_1eps.load(Ordering::Relaxed),
                over_2eps: envelope.over_2eps.load(Ordering::Relaxed),
                over_4eps: envelope.over_4eps.load(Ordering::Relaxed),
                over_8eps: envelope.over_8eps.load(Ordering::Relaxed),
                over_1e6: envelope.over_1e6.load(Ordering::Relaxed),
                over_1e5: envelope.over_1e5.load(Ordering::Relaxed),
                over_1e4: envelope.over_1e4.load(Ordering::Relaxed),
                max_abs_error: f64::from_bits(envelope.max_abs_error_bits.load(Ordering::Relaxed)),
                f32_rule_changed: envelope.f32_rule_changed.load(Ordering::Relaxed),
                f64_rule_changed: envelope.f64_rule_changed.load(Ordering::Relaxed),
                rules_differ: envelope.rules_differ.load(Ordering::Relaxed),
                f32_rule_max_abs_error: f64::from_bits(
                    envelope.f32_rule_max_abs_error_bits.load(Ordering::Relaxed),
                ),
                f64_rule_max_abs_error: f64::from_bits(
                    envelope.f64_rule_max_abs_error_bits.load(Ordering::Relaxed),
                ),
            }
        })
        .collect()
}
