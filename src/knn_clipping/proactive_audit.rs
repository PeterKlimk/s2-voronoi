//! Probe-only proactive-correctness audit side channel.
//!
//! Enabled by `S2_PROACTIVE_AUDIT=1` under the `escalate_probe` feature. The
//! production path never records these metrics.

use std::sync::{Mutex, OnceLock};

/// Per-cell fast-path audit metrics used by exact-reference probes.
#[derive(Debug, Clone, Copy)]
pub struct CellAudit {
    /// Effective generator id.
    pub generator: u32,
    /// Cell entered the spherical fallback due to projection limit.
    pub fallback_projection: bool,
    /// Cell entered the spherical fallback due to polygon vertex cap.
    pub fallback_polygon_cap: bool,
    /// Cell stopped because the neighbor-frontier termination certificate fired.
    pub terminated: bool,
    /// Smallest positive termination clearance observed for an accepted
    /// termination, in dot-product units (`threshold - unseen_bound`).
    pub termination_clearance: Option<f64>,
    /// Smallest transition denominator `|d0 - d1|` observed at a mixed clip.
    pub transition_delta: Option<f64>,
    /// Smallest positive early-unchanged clearance `c^2 - |ab|^2 r^2`.
    pub early_unchanged_clearance: Option<f64>,
}

fn records() -> &'static Mutex<Vec<CellAudit>> {
    static RECORDS: OnceLock<Mutex<Vec<CellAudit>>> = OnceLock::new();
    RECORDS.get_or_init(|| Mutex::new(Vec::new()))
}

#[inline]
pub(crate) fn enabled() -> bool {
    std::env::var("S2_PROACTIVE_AUDIT").is_ok()
}

/// Clear all accumulated proactive audit records.
pub fn reset() {
    records().lock().expect("proactive audit lock").clear();
}

pub(crate) fn record(record: CellAudit) {
    if enabled() {
        records().lock().expect("proactive audit lock").push(record);
    }
}

/// Take the current audit records, sorted by generator id for deterministic
/// probe output.
pub fn take() -> Vec<CellAudit> {
    let mut out = std::mem::take(&mut *records().lock().expect("proactive audit lock"));
    out.sort_unstable_by_key(|r| r.generator);
    out
}
