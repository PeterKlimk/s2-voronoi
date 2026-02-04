use std::time::Duration;

/// Timer that tracks elapsed time when timing is enabled.
#[cfg(feature = "timing")]
pub(super) struct PackedLapTimer(std::time::Instant);

#[cfg(feature = "timing")]
impl PackedLapTimer {
    #[inline]
    pub fn start() -> Self {
        Self(std::time::Instant::now())
    }

    #[inline]
    pub fn lap(&mut self) -> Duration {
        let now = std::time::Instant::now();
        let d = now.duration_since(self.0);
        self.0 = now;
        d
    }
}

/// Dummy timer when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
pub(super) struct PackedLapTimer;

#[cfg(not(feature = "timing"))]
impl PackedLapTimer {
    #[inline(always)]
    pub fn start() -> Self {
        Self
    }

    #[inline(always)]
    pub fn lap(&mut self) -> Duration {
        Duration::ZERO
    }
}

/// Fine-grained timing breakdown for the packed-kNN per-cell-group flow.
#[cfg(feature = "timing")]
#[derive(Debug, Clone)]
pub struct PackedKnnTimings {
    pub setup: Duration,
    pub query_cache: Duration,
    pub security_thresholds: Duration,
    pub center_pass: Duration,
    pub ring_thresholds: Duration,
    pub ring_pass: Duration,
    pub ring_fallback: Duration,
    pub select_prep: Duration,
    pub select_query_prep: Duration,
    pub select_partition: Duration,
    pub select_sort: Duration,
    pub select_scatter: Duration,
    /// Number of times tail candidates were built (per query, but counted at most once per group).
    pub tail_builds: u64,
}

#[cfg(feature = "timing")]
impl Default for PackedKnnTimings {
    fn default() -> Self {
        Self {
            setup: Duration::ZERO,
            query_cache: Duration::ZERO,
            security_thresholds: Duration::ZERO,
            center_pass: Duration::ZERO,
            ring_thresholds: Duration::ZERO,
            ring_pass: Duration::ZERO,
            ring_fallback: Duration::ZERO,
            select_prep: Duration::ZERO,
            select_query_prep: Duration::ZERO,
            select_partition: Duration::ZERO,
            select_sort: Duration::ZERO,
            select_scatter: Duration::ZERO,
            tail_builds: 0,
        }
    }
}

/// Dummy timings when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct PackedKnnTimings;

#[cfg(feature = "timing")]
impl PackedKnnTimings {
    #[inline]
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    #[inline]
    pub fn add_setup(&mut self, d: Duration) {
        self.setup += d;
    }

    #[inline]
    pub fn add_query_cache(&mut self, d: Duration) {
        self.query_cache += d;
    }

    #[inline]
    pub fn add_security_thresholds(&mut self, d: Duration) {
        self.security_thresholds += d;
    }

    #[inline]
    pub fn add_center_pass(&mut self, d: Duration) {
        self.center_pass += d;
    }

    #[inline]
    pub fn add_ring_thresholds(&mut self, d: Duration) {
        self.ring_thresholds += d;
    }

    #[inline]
    pub fn add_ring_pass(&mut self, d: Duration) {
        self.ring_pass += d;
    }

    #[inline]
    pub fn add_ring_fallback(&mut self, d: Duration) {
        self.ring_fallback += d;
    }

    #[inline]
    pub fn add_select_prep(&mut self, d: Duration) {
        self.select_prep += d;
    }

    #[inline]
    pub fn add_select_query_prep(&mut self, d: Duration) {
        self.select_query_prep += d;
    }

    #[inline]
    pub fn add_select_partition(&mut self, d: Duration) {
        self.select_partition += d;
    }

    #[inline]
    pub fn add_select_sort_sized(&mut self, d: Duration, _n: usize) {
        self.select_sort += d;
    }

    #[inline]
    pub fn add_select_scatter(&mut self, d: Duration) {
        self.select_scatter += d;
    }

    #[inline]
    pub fn inc_tail_builds(&mut self) {
        self.tail_builds += 1;
    }

    #[inline]
    pub fn total(&self) -> Duration {
        self.setup
            + self.query_cache
            + self.security_thresholds
            + self.center_pass
            + self.ring_thresholds
            + self.ring_pass
            + self.ring_fallback
            + self.select_prep
            + self.select_query_prep
            + self.select_partition
            + self.select_sort
            + self.select_scatter
    }
}

#[cfg(not(feature = "timing"))]
impl PackedKnnTimings {
    #[inline(always)]
    pub fn clear(&mut self) {}

    #[inline(always)]
    pub fn add_setup(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_query_cache(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_security_thresholds(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_center_pass(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_ring_thresholds(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_ring_pass(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_ring_fallback(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_select_prep(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_select_query_prep(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_select_partition(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_select_sort_sized(&mut self, _d: Duration, _n: usize) {}
    #[inline(always)]
    pub fn add_select_scatter(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn inc_tail_builds(&mut self) {}
}

