use std::sync::atomic::{AtomicU64, Ordering};

pub struct ClipConvexStatsSnapshot {
    pub calls: u64,
    pub early_unchanged_hits: u64,
    pub early_unchanged_hits_bounded: u64,
    // Buckets are indexed by n, for n in 0..=16. Larger n are accumulated in *_gt_16.
    pub calls_by_n: [u64; 17],
    pub hits_by_n: [u64; 17],
    pub calls_gt_16: u64,
    pub hits_gt_16: u64,
}

static CALLS: AtomicU64 = AtomicU64::new(0);
static EARLY_UNCHANGED_HITS: AtomicU64 = AtomicU64::new(0);
static EARLY_UNCHANGED_HITS_BOUNDED: AtomicU64 = AtomicU64::new(0);

static CALLS_GT_16: AtomicU64 = AtomicU64::new(0);
static HITS_GT_16: AtomicU64 = AtomicU64::new(0);

static CALLS_BY_N: [AtomicU64; 17] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

static HITS_BY_N: [AtomicU64; 17] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

#[inline(always)]
pub fn record_call(n: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    if n <= 16 {
        CALLS_BY_N[n].fetch_add(1, Ordering::Relaxed);
    } else {
        CALLS_GT_16.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline(always)]
pub fn record_early_unchanged(n: usize, bounded: bool) {
    EARLY_UNCHANGED_HITS.fetch_add(1, Ordering::Relaxed);
    if bounded {
        EARLY_UNCHANGED_HITS_BOUNDED.fetch_add(1, Ordering::Relaxed);
    }
    if n <= 16 {
        HITS_BY_N[n].fetch_add(1, Ordering::Relaxed);
    } else {
        HITS_GT_16.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn take() -> ClipConvexStatsSnapshot {
    let calls = CALLS.swap(0, Ordering::Relaxed);
    let early_unchanged_hits = EARLY_UNCHANGED_HITS.swap(0, Ordering::Relaxed);
    let early_unchanged_hits_bounded = EARLY_UNCHANGED_HITS_BOUNDED.swap(0, Ordering::Relaxed);

    let mut calls_by_n = [0u64; 17];
    let mut hits_by_n = [0u64; 17];
    for i in 0..=16 {
        calls_by_n[i] = CALLS_BY_N[i].swap(0, Ordering::Relaxed);
        hits_by_n[i] = HITS_BY_N[i].swap(0, Ordering::Relaxed);
    }
    let calls_gt_16 = CALLS_GT_16.swap(0, Ordering::Relaxed);
    let hits_gt_16 = HITS_GT_16.swap(0, Ordering::Relaxed);

    ClipConvexStatsSnapshot {
        calls,
        early_unchanged_hits,
        early_unchanged_hits_bounded,
        calls_by_n,
        hits_by_n,
        calls_gt_16,
        hits_gt_16,
    }
}
