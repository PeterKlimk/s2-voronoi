//! Timing hooks for the planar packed stage.
//!
//! Currently no-op placeholders mirroring the sphere's `PackedKnnTimings`
//! call surface, so the hot-path call sites are already in place when the
//! planar TIMING_KV plumbing lands (mechanical swap, like the sphere's
//! real/stub split).

#[derive(Debug, Default, Clone, Copy)]
pub struct PlanePackedTimings;

/// Zero-sized lap measurement; becomes a `Duration` when the real timing
/// plumbing lands.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlaneLap;

#[allow(clippy::unused_self)]
impl PlanePackedTimings {
    #[inline(always)]
    pub(crate) fn clear(&mut self) {}
    #[inline(always)]
    pub(crate) fn add_setup(&mut self, _: PlaneLap) {}
    #[inline(always)]
    pub(crate) fn add_security_thresholds(&mut self, _: PlaneLap) {}
    #[inline(always)]
    pub(crate) fn add_select_prep(&mut self, _: PlaneLap) {}
    #[inline(always)]
    pub(crate) fn add_center_pass(&mut self, _: PlaneLap) {}
    #[inline(always)]
    pub(crate) fn add_ring_thresholds(&mut self, _: PlaneLap) {}
    #[inline(always)]
    pub(crate) fn add_ring_pass(&mut self, _: PlaneLap) {}
    #[inline(always)]
    pub(crate) fn add_ring_fallback(&mut self, _: PlaneLap) {}
    #[inline(always)]
    pub(crate) fn add_select_query_prep(&mut self, _: PlaneLap) {}
    #[inline(always)]
    pub(crate) fn add_select_sort_sized(&mut self, _: PlaneLap, _len: usize) {}
    #[inline(always)]
    pub(crate) fn add_select_partition(&mut self, _: PlaneLap) {}
    #[inline(always)]
    pub(crate) fn add_select_scatter(&mut self, _: PlaneLap) {}
    #[inline(always)]
    pub(crate) fn add_expand_r2_scan(&mut self, _: PlaneLap) {}
    #[inline(always)]
    pub(crate) fn inc_tail_builds(&mut self) {}
    #[inline(always)]
    pub(crate) fn inc_expand_r2_builds(&mut self) {}
    #[inline(always)]
    pub(crate) fn inc_expand_r2_cap_skips(&mut self) {}
}

/// No-op lap timer matching the sphere's `PackedLapTimer` shape.
pub(crate) struct PlanePackedLapTimer;

impl PlanePackedLapTimer {
    #[inline(always)]
    pub(crate) fn start() -> Self {
        Self
    }
    #[inline(always)]
    pub(crate) fn lap(&mut self) -> PlaneLap {
        PlaneLap
    }
}
