//! Probe-only proactive-correctness audit side channel.
//!
//! Enabled by `S2_PROACTIVE_AUDIT=1` under the `escalate_probe` feature. The
//! production path never records these metrics.

use std::collections::BTreeSet;
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
    /// Number of neighbor clips attempted for this cell.
    pub neighbors_processed: usize,
    /// Final emitted edge count.
    pub final_edges: usize,
    /// Whether the directed stream reached shell exhaustion.
    pub knn_exhausted: bool,
    /// Whether the cell entered the shell-expansion takeover.
    pub used_knn: bool,
    /// Whether the packed directed query served this cell.
    pub did_packed: bool,
    /// Whether the packed tail was materialized.
    pub packed_tail_used: bool,
    /// Whether the packed query reported a safe exhaustion before takeover.
    pub packed_safe_exhausted: bool,
    /// Smallest positive termination clearance observed for an accepted
    /// termination, in dot-product units (`threshold - unseen_bound`).
    pub termination_clearance: Option<f64>,
    /// The unseen dot-product bound paired with `termination_clearance`.
    pub termination_bound: Option<f64>,
    /// Smallest transition denominator `|d0 - d1|` observed at a mixed clip.
    pub transition_delta: Option<f64>,
    /// Smallest positive early-unchanged clearance `c^2 - |ab|^2 r^2`.
    pub early_unchanged_clearance: Option<f64>,
}

/// Probe-only result for a watched `(generator, neighbor)` clip attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchedClipResult {
    /// The watched neighbor was tested and left the polygon unchanged.
    Unchanged,
    /// The watched neighbor changed the polygon.
    Changed,
    /// The watched neighbor would exceed the polygon vertex cap.
    TooManyVertices,
    /// The watched neighbor triggered the spherical fallback path.
    NeedsFallback,
    /// The watched neighbor attempt returned a builder error.
    Error,
}

/// Probe-only record that a watched neighbor was attempted by the fast stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WatchedClip {
    /// Effective generator id.
    pub generator: u32,
    /// Watched neighbor id.
    pub neighbor: u32,
    /// Result of the watched clip attempt.
    pub result: WatchedClipResult,
}

fn records() -> &'static Mutex<Vec<CellAudit>> {
    static RECORDS: OnceLock<Mutex<Vec<CellAudit>>> = OnceLock::new();
    RECORDS.get_or_init(|| Mutex::new(Vec::new()))
}

fn watch_pairs() -> &'static Mutex<BTreeSet<(u32, u32)>> {
    static WATCH_PAIRS: OnceLock<Mutex<BTreeSet<(u32, u32)>>> = OnceLock::new();
    WATCH_PAIRS.get_or_init(|| Mutex::new(BTreeSet::new()))
}

fn watched_clips() -> &'static Mutex<Vec<WatchedClip>> {
    static WATCHED_CLIPS: OnceLock<Mutex<Vec<WatchedClip>>> = OnceLock::new();
    WATCHED_CLIPS.get_or_init(|| Mutex::new(Vec::new()))
}

#[inline]
pub(crate) fn enabled() -> bool {
    std::env::var("S2_PROACTIVE_AUDIT").is_ok()
}

/// Clear all accumulated proactive audit records.
pub fn reset() {
    records().lock().expect("proactive audit lock").clear();
    watched_clips()
        .lock()
        .expect("proactive watch lock")
        .clear();
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

/// Replace the watched `(generator, neighbor)` set for targeted clip tracing.
pub fn set_watch_pairs(pairs: &[(u32, u32)]) {
    let mut watch = watch_pairs().lock().expect("proactive watch-pair lock");
    watch.clear();
    watch.extend(pairs.iter().copied());
    watched_clips()
        .lock()
        .expect("proactive watch lock")
        .clear();
}

/// Clear watched clip pairs and any accumulated watched clip records.
pub fn clear_watch_pairs() {
    watch_pairs()
        .lock()
        .expect("proactive watch-pair lock")
        .clear();
    watched_clips()
        .lock()
        .expect("proactive watch lock")
        .clear();
}

pub(crate) fn record_watched_clip(generator: usize, neighbor: usize, result: WatchedClipResult) {
    let pair = (generator as u32, neighbor as u32);
    if !watch_pairs()
        .lock()
        .expect("proactive watch-pair lock")
        .contains(&pair)
    {
        return;
    }
    watched_clips()
        .lock()
        .expect("proactive watch lock")
        .push(WatchedClip {
            generator: pair.0,
            neighbor: pair.1,
            result,
        });
}

/// Take watched clip records sorted by `(generator, neighbor)`.
pub fn take_watched_clips() -> Vec<WatchedClip> {
    let mut out = std::mem::take(&mut *watched_clips().lock().expect("proactive watch lock"));
    out.sort_unstable_by_key(|r| (r.generator, r.neighbor));
    out
}
