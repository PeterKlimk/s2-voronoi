//! Profiling-only edge-check queue shape census.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

#[derive(Clone, Copy, Debug, Default)]
#[allow(missing_docs)] // internal profiling record; field names are the machine-readable schema
pub struct EdgeQueueSummary {
    pub queues_taken: u64,
    pub lengths: [u64; 8],
    pub pushes: u64,
    pub fresh_allocations: u64,
    pub pool_reuses: u64,
    pub growth_events: u64,
    pub growth_copied_records: u64,
    pub capacity_records_at_take: u64,
    pub used_records_at_take: u64,
    pub max_queue_len: u64,
    pub max_active_queues: u64,
    pub max_live_records: u64,
    pub max_pool_len: u64,
    pub sum_shard_peak_active: u64,
    pub sum_shard_peak_live: u64,
    pub sum_shard_peak_pool: u64,
}

#[derive(Default)]
pub(crate) struct LocalQueueAudit {
    summary: EdgeQueueSummary,
    active_queues: u64,
    live_records: u64,
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static SUMMARY: Mutex<EdgeQueueSummary> = Mutex::new(EdgeQueueSummary {
    queues_taken: 0,
    lengths: [0; 8],
    pushes: 0,
    fresh_allocations: 0,
    pool_reuses: 0,
    growth_events: 0,
    growth_copied_records: 0,
    capacity_records_at_take: 0,
    used_records_at_take: 0,
    max_queue_len: 0,
    max_active_queues: 0,
    max_live_records: 0,
    max_pool_len: 0,
    sum_shard_peak_active: 0,
    sum_shard_peak_live: 0,
    sum_shard_peak_pool: 0,
});

impl LocalQueueAudit {
    pub(crate) fn record_push(
        &mut self,
        old_len: usize,
        capacity_before_push: usize,
        new_capacity: usize,
        reused_pool: bool,
    ) {
        self.summary.pushes += 1;
        if old_len == 0 {
            self.active_queues += 1;
            self.summary.max_active_queues = self.summary.max_active_queues.max(self.active_queues);
        }
        self.live_records += 1;
        self.summary.max_live_records = self.summary.max_live_records.max(self.live_records);
        if reused_pool {
            self.summary.pool_reuses += 1;
        }
        if new_capacity != capacity_before_push {
            self.summary.growth_events += 1;
            self.summary.growth_copied_records += old_len as u64;
            if capacity_before_push == 0 && !reused_pool {
                self.summary.fresh_allocations += 1;
            }
        }
    }

    pub(crate) fn record_take(&mut self, len: usize, capacity: usize) {
        self.summary.queues_taken += 1;
        self.summary.lengths[length_bucket(len)] += 1;
        self.summary.used_records_at_take += len as u64;
        self.summary.capacity_records_at_take += capacity as u64;
        self.summary.max_queue_len = self.summary.max_queue_len.max(len as u64);
        if len != 0 {
            self.active_queues -= 1;
            self.live_records -= len as u64;
        }
    }

    pub(crate) fn observe_pool(&mut self, len: usize) {
        self.summary.max_pool_len = self.summary.max_pool_len.max(len as u64);
    }
}

impl Drop for LocalQueueAudit {
    fn drop(&mut self) {
        if !ENABLED.load(Ordering::Relaxed) {
            return;
        }
        let mut total = SUMMARY.lock().expect("edge-queue audit poisoned");
        merge(&mut total, self.summary);
    }
}

const fn length_bucket(len: usize) -> usize {
    match len {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5..=8 => 5,
        9..=16 => 6,
        _ => 7,
    }
}

fn merge(total: &mut EdgeQueueSummary, local: EdgeQueueSummary) {
    total.queues_taken += local.queues_taken;
    for (dst, src) in total.lengths.iter_mut().zip(local.lengths) {
        *dst += src;
    }
    total.pushes += local.pushes;
    total.fresh_allocations += local.fresh_allocations;
    total.pool_reuses += local.pool_reuses;
    total.growth_events += local.growth_events;
    total.growth_copied_records += local.growth_copied_records;
    total.capacity_records_at_take += local.capacity_records_at_take;
    total.used_records_at_take += local.used_records_at_take;
    total.max_queue_len = total.max_queue_len.max(local.max_queue_len);
    total.max_active_queues = total.max_active_queues.max(local.max_active_queues);
    total.max_live_records = total.max_live_records.max(local.max_live_records);
    total.max_pool_len = total.max_pool_len.max(local.max_pool_len);
    total.sum_shard_peak_active += local.max_active_queues;
    total.sum_shard_peak_live += local.max_live_records;
    total.sum_shard_peak_pool += local.max_pool_len;
}

pub(crate) fn reset() {
    *SUMMARY.lock().expect("edge-queue audit poisoned") = EdgeQueueSummary::default();
    ENABLED.store(true, Ordering::Relaxed);
}

pub(crate) fn snapshot() -> EdgeQueueSummary {
    ENABLED.store(false, Ordering::Relaxed);
    *SUMMARY.lock().expect("edge-queue audit poisoned")
}
