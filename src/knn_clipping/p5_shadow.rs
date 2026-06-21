//! P5 stage-1 shadow audit (feature `p5_shadow`).
//!
//! Logs, with zero behavior change, how the chart-local clip decisions
//! compare against the canonical exact in-circle predicate that P5 would
//! escalate to (see `docs/p5-consistency-design.md`):
//!
//! - a margin histogram of every audited vertex-vs-bisector decision
//!   (normalized chart distance |d|/|n|, bucketed by decimal exponent),
//! - for near-margin decisions (below `CANONICAL_CUTOFF`), the exact
//!   canonical answer and whether the local decision disagrees,
//! - exact ties (true cocircularity) counted separately.
//!
//! This measures (a) escalation frequency for the P5 perf model and (b) how
//! often today's path disagrees with canonical — before any behavior changes.
//! Audits cover the gnomonic builder only (the fallback builder is the rare
//! `ProjectionLimit` path and defers clipping entirely).
//!
//! Note on permutation consistency: because the evaluator is *exact*, its
//! signs are automatically permutation-covariant — the sorted-row-order
//! discipline in the design doc only matters for rounded evaluators and for
//! the SoS cascade (stage 2).

use std::sync::atomic::{AtomicU64, Ordering};

use glam::{Vec2, Vec3};

use crate::knn_clipping::canonical::in_circle_sphere_sign;
use crate::knn_clipping::topo2d::types::{HalfPlane, PolyBuffer, INVALID_PLANE_ID};

/// Margin buckets by decimal exponent of the normalized distance:
/// bucket 0: nd >= 1e-1, bucket k: 1e-(k+1) <= nd < 1e-k, bucket 15: nd < 1e-15 (incl. 0).
const BUCKETS: usize = 16;

/// Normalized-margin cutoff below which the canonical predicate is evaluated.
/// Chart units are ~radians; cell sizes are ~2.5e-3 at 2M points, so 1e-4
/// catches the near-tie tail without evaluating everything.
const CANONICAL_CUTOFF: f64 = 1e-4;

static AUDITED: AtomicU64 = AtomicU64::new(0);
static SKIPPED_SYNTHETIC: AtomicU64 = AtomicU64::new(0);
static CANON_EVALS: AtomicU64 = AtomicU64::new(0);
static EXACT_TIES: AtomicU64 = AtomicU64::new(0);
#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static MARGIN_HIST: [AtomicU64; BUCKETS] = [ZERO; BUCKETS];
static DISAGREE_HIST: [AtomicU64; BUCKETS] = [ZERO; BUCKETS];
/// Exact ties (canonical == 0) bucketed by margin — a tie at margin `m` forces
/// the superset BAND >= m just as a disagreement does (§5 of
/// docs/adaptive-canonical-clip-design-2026-06.md).
static TIE_HIST: [AtomicU64; BUCKETS] = [ZERO; BUCKETS];

/// Probe-settable margin below which `audit_clip` evaluates the exact
/// predicate. Default `CANONICAL_CUTOFF`; the superset-BAND measurement raises
/// it (e.g. to 10.0) to evaluate EVERY decision, so a disagreement/tie at a
/// *large* margin — the thing that would break the superset property — cannot
/// be missed. `u64::MAX` bits = use the default. Probe-only.
static AUDIT_CUTOFF_BITS: AtomicU64 = AtomicU64::new(u64::MAX);

/// Set (or clear) the exact-evaluation cutoff (probe API).
pub fn set_audit_cutoff(cutoff: Option<f64>) {
    AUDIT_CUTOFF_BITS.store(cutoff.map_or(u64::MAX, |c| c.to_bits()), Ordering::Relaxed);
}

#[inline]
fn audit_cutoff() -> f64 {
    let bits = AUDIT_CUTOFF_BITS.load(Ordering::Relaxed);
    if bits == u64::MAX {
        CANONICAL_CUTOFF
    } else {
        f64::from_bits(bits)
    }
}

/// Paired-audit collection cutoff (normalized margin); 0.0 disables.
/// f64 bits in an atomic so probes can vary it per run.
static PAIR_CUTOFF_BITS: AtomicU64 = AtomicU64::new(0);

/// Enable/disable paired collection (probe API).
pub fn set_pair_cutoff(cutoff: f64) {
    PAIR_CUTOFF_BITS.store(cutoff.to_bits(), Ordering::Relaxed);
}

#[inline]
fn pair_cutoff() -> f64 {
    f64::from_bits(PAIR_CUTOFF_BITS.load(Ordering::Relaxed))
}

/// Pass-2 key filter: when set, decisions whose question key is in the set
/// are recorded at ANY margin (de-censoring the cutoff-based pass-1
/// collection). Arc-cloned once per audit call; None disables.
#[allow(clippy::type_complexity)]
static PAIR_KEY_FILTER: std::sync::RwLock<
    Option<std::sync::Arc<std::collections::HashSet<[u32; 4]>>>,
> = std::sync::RwLock::new(None);

/// Install (or clear) the pass-2 key filter (probe API).
pub fn set_pair_key_filter(keys: Option<Vec<[u32; 4]>>) {
    *PAIR_KEY_FILTER.write().unwrap() = keys.map(|k| std::sync::Arc::new(k.into_iter().collect()));
}

/// Per-question summary of the current paired collection:
/// (key, distinct answering cells, local answers conflict, min margin).
pub fn paired_question_summaries() -> Vec<([u32; 4], u32, bool, f32)> {
    use std::collections::HashMap;
    let records = PAIRED.lock().unwrap().clone();
    let mut groups: HashMap<[u32; 4], Vec<&PairedRecord>> = HashMap::new();
    for r in &records {
        groups.entry(r.key).or_default().push(r);
    }
    groups
        .into_iter()
        .map(|(key, entries)| {
            let mut cells: Vec<u32> = entries.iter().map(|e| e.cell).collect();
            cells.sort_unstable();
            cells.dedup();
            let conflict =
                entries.iter().any(|e| e.local_keep) && entries.iter().any(|e| !e.local_keep);
            let min_margin = entries
                .iter()
                .map(|e| e.margin)
                .fold(f32::INFINITY, f32::min);
            (key, cells.len() as u32, conflict, min_margin)
        })
        .collect()
}

/// Experimental termination-pad override (radians); NaN bits = disabled.
/// Gate-1 question-set-closure experiments (EPS_CERT sizing) widen the
/// certificate so marginal generators are delivered to every cell whose
/// shared features they could affect. Probe-only.
static TERM_PAD_OVERRIDE_BITS: AtomicU64 = AtomicU64::new(u64::MAX);

/// Set (or clear) the termination angle-pad override (probe API).
pub fn set_term_pad_override(pad: Option<f64>) {
    let bits = match pad {
        Some(p) => p.to_bits(),
        None => u64::MAX,
    };
    TERM_PAD_OVERRIDE_BITS.store(bits, Ordering::Relaxed);
}

/// Current override, if any (read at builder construction — cold).
pub(crate) fn term_pad_override() -> Option<f64> {
    let bits = TERM_PAD_OVERRIDE_BITS.load(Ordering::Relaxed);
    if bits == u64::MAX {
        None
    } else {
        Some(f64::from_bits(bits))
    }
}

/// Experimental escalation-factor override (multiplies hp.eps to form the
/// escalation threshold); NaN bits = use the tolerances default. Probe-only.
static ESC_FACTOR_OVERRIDE_BITS: AtomicU64 = AtomicU64::new(u64::MAX);

/// Set (or clear) the escalation-factor override (probe API).
pub fn set_escalation_factor_override(factor: Option<f64>) {
    let bits = match factor {
        Some(f) => f.to_bits(),
        None => u64::MAX,
    };
    ESC_FACTOR_OVERRIDE_BITS.store(bits, Ordering::Relaxed);
}

/// Current override, if any.
pub(crate) fn escalation_factor_override() -> Option<f64> {
    let bits = ESC_FACTOR_OVERRIDE_BITS.load(Ordering::Relaxed);
    if bits == u64::MAX {
        None
    } else {
        Some(f64::from_bits(bits))
    }
}

/// Experimental clip-eps override (replaces CLIP_EPS_INSIDE at HalfPlane
/// construction); u64::MAX bits = use the tolerances default. 0.0 turns the
/// keep-bias `d >= -eps` into the strict antisymmetric tie rule `d >= 0`
/// (successor candidate 1 in docs/p5-consistency-design.md). The edgecheck
/// eps-reuse path degrades cleanly: a forwarded eps of 0 routes through the
/// ordinary construction path, which reads this override again. Probe-only.
static CLIP_EPS_OVERRIDE_BITS: AtomicU64 = AtomicU64::new(u64::MAX);

/// Set (or clear) the clip-eps override (probe API).
pub fn set_clip_eps_override(eps: Option<f64>) {
    let bits = match eps {
        Some(e) => e.to_bits(),
        None => u64::MAX,
    };
    CLIP_EPS_OVERRIDE_BITS.store(bits, Ordering::Relaxed);
}

/// Current override, if any.
pub(crate) fn clip_eps_override() -> Option<f64> {
    let bits = CLIP_EPS_OVERRIDE_BITS.load(Ordering::Relaxed);
    if bits == u64::MAX {
        None
    } else {
        Some(f64::from_bits(bits))
    }
}

/// Experimental PLANAR clip-eps override (replaces PLANE_CLIP_EPS_INSIDE
/// at the bounded plane builder's construction sites — walls and
/// bisectors; the periodic builder is not covered). u64::MAX bits =
/// default. Probe-only.
static PLANE_CLIP_EPS_OVERRIDE_BITS: AtomicU64 = AtomicU64::new(u64::MAX);

/// Set (or clear) the planar clip-eps override (probe API).
pub fn set_plane_clip_eps_override(eps: Option<f64>) {
    let bits = match eps {
        Some(e) => e.to_bits(),
        None => u64::MAX,
    };
    PLANE_CLIP_EPS_OVERRIDE_BITS.store(bits, Ordering::Relaxed);
}

/// Current planar override, if any.
pub(crate) fn plane_clip_eps_override() -> Option<f64> {
    let bits = PLANE_CLIP_EPS_OVERRIDE_BITS.load(Ordering::Relaxed);
    if bits == u64::MAX {
        None
    } else {
        Some(f64::from_bits(bits))
    }
}

/// Exact planar in-circle, sphere-convention signs: +1 = `x` strictly
/// inside the circumcircle of (g, a, b) (the vertex must be cut), -1 =
/// strictly outside (kept), 0 = exact tie or degenerate triple.
/// Orientation-independent via the orient2d factor; signs exact (Shewchuk
/// adaptive, f32 inputs exactly representable in f64).
pub(crate) fn in_circle_plane_sign(g: Vec2, a: Vec2, b: Vec2, x: Vec2) -> i8 {
    let c = |p: Vec2| robust::Coord {
        x: p.x as f64,
        y: p.y as f64,
    };
    let d1 = robust::incircle(c(g), c(a), c(b), c(x));
    let d2 = robust::orient2d(c(g), c(a), c(b));
    ((d1 > 0.0) as i8 - (d1 < 0.0) as i8) * ((d2 > 0.0) as i8 - (d2 < 0.0) as i8)
}

/// Planar twin of [`audit_clip`]: same paired collection, margin
/// histogram, cutoff and key-filter machinery; canonical answered by the
/// exact planar in-circle. Vertices attributed to rect walls (plane
/// indices 0..4) are skipped as synthetic.
#[allow(clippy::too_many_arguments)] // shadow-only audit seam
pub(crate) fn audit_clip_plane(
    generator_idx: usize,
    generator: Vec2,
    neighbor_idx: usize,
    neighbor: Vec2,
    neighbor_indices: &[usize],
    neighbor_positions: &[Vec2],
    poly: &PolyBuffer,
    hp: &HalfPlane,
) {
    if hp.ab2.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return;
    }
    let inv_norm = 1.0 / hp.ab2.sqrt();
    let pcut = pair_cutoff();
    let key_filter = PAIR_KEY_FILTER.read().unwrap().clone();
    let mut batch: Vec<PairedRecord> = Vec::new();
    for i in 0..poly.len {
        let (pa, pb) = poly.vertex_planes[i];
        if pa == INVALID_PLANE_ID || pb == INVALID_PLANE_ID || pa < 4 || pb < 4 {
            SKIPPED_SYNTHETIC.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let pa = pa as usize;
        let pb = pb as usize;
        debug_assert!(pa < neighbor_positions.len() && pb < neighbor_positions.len());
        let d = hp.signed_dist(poly.us[i], poly.vs[i]);
        let nd = d.abs() * inv_norm;
        AUDITED.fetch_add(1, Ordering::Relaxed);
        MARGIN_HIST[bucket_for(nd)].fetch_add(1, Ordering::Relaxed);

        let below_cutoff = nd < pcut;
        if !(below_cutoff || key_filter.is_some()) {
            continue;
        }
        let (ga, gb) = (neighbor_indices[pa] as u32, neighbor_indices[pb] as u32);
        let g = generator_idx as u32;
        let x = neighbor_idx as u32;
        let mut triple = [g, ga, gb];
        triple.sort_unstable();
        if triple.contains(&x) {
            continue;
        }
        let key = [triple[0], triple[1], triple[2], x];
        let wanted = below_cutoff || key_filter.as_ref().is_some_and(|f| f.contains(&key));
        if wanted {
            batch.push(PairedRecord {
                key,
                cell: g,
                margin: nd as f32,
                local_keep: d >= -hp.eps,
                canonical: in_circle_plane_sign(
                    generator,
                    neighbor_positions[pa],
                    neighbor_positions[pb],
                    neighbor,
                ),
            });
        }
    }
    if !batch.is_empty() {
        PAIRED.lock().unwrap().extend(batch);
    }
}

/// Planar unresolved-edge collector: what detection saw, pre-repair
/// (the sphere exposes this via `ComputeOutput`; the plane does not).
static PLANE_UNRESOLVED: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());

pub(crate) fn record_plane_unresolved(keys: impl Iterator<Item = u64>) {
    PLANE_UNRESOLVED.lock().unwrap().extend(keys);
}

/// Reset the planar unresolved-edge collection (probe API).
pub fn plane_unresolved_reset() {
    PLANE_UNRESOLVED.lock().unwrap().clear();
}

/// Detected unresolved planar edges as (cell, cell) pairs (probe API).
pub fn plane_unresolved() -> Vec<(u32, u32)> {
    PLANE_UNRESOLVED
        .lock()
        .unwrap()
        .iter()
        .map(|&k| (k as u32, (k >> 32) as u32))
        .collect()
}

/// One near-margin decision, keyed by its abstract question: does generator
/// `key[3]` kill the vertex of sorted triple `key[0..3]`? Multiple cells
/// answer the same question (each owner of the triple clips that corner
/// against x); pairing them measures cross-cell conflict directly.
#[derive(Clone, Copy)]
struct PairedRecord {
    key: [u32; 4],
    cell: u32,
    margin: f32,
    local_keep: bool,
    canonical: i8,
}

static PAIRED: std::sync::Mutex<Vec<PairedRecord>> = std::sync::Mutex::new(Vec::new());

#[inline]
fn bucket_for(nd: f64) -> usize {
    if nd >= 1e-1 {
        return 0;
    }
    // nd == 0.0 -> inf -> saturating cast -> clamped to BUCKETS-1.
    let e = (-nd.log10()).floor() as usize;
    e.min(BUCKETS - 1)
}

/// Audit one clip attempt: for every current polygon vertex with real plane
/// attribution, classify its margin against the incoming half-plane and,
/// below the cutoff, compare the local decision with the canonical one.
#[allow(clippy::too_many_arguments)] // shadow-only audit seam
pub(crate) fn audit_clip(
    generator_idx: usize,
    generator_raw: Vec3,
    neighbor_idx: usize,
    neighbor_raw: Vec3,
    neighbor_indices: &[usize],
    neighbor_positions: &[Vec3],
    poly: &PolyBuffer,
    hp: &HalfPlane,
) {
    // NaN or zero-normal planes carry no geometric meaning to audit.
    if hp.ab2.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return;
    }
    let inv_norm = 1.0 / hp.ab2.sqrt();
    let pcut = pair_cutoff();
    let key_filter = PAIR_KEY_FILTER.read().unwrap().clone();
    let mut batch: Vec<PairedRecord> = Vec::new();
    for i in 0..poly.len {
        let (pa, pb) = poly.vertex_planes[i];
        if pa == INVALID_PLANE_ID || pb == INVALID_PLANE_ID {
            SKIPPED_SYNTHETIC.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let pa = pa as usize;
        let pb = pb as usize;
        debug_assert!(pa < neighbor_positions.len() && pb < neighbor_positions.len());
        let d = hp.signed_dist(poly.us[i], poly.vs[i]);
        let nd = d.abs() * inv_norm;
        AUDITED.fetch_add(1, Ordering::Relaxed);
        let bucket = bucket_for(nd);
        MARGIN_HIST[bucket].fetch_add(1, Ordering::Relaxed);

        let a = neighbor_positions[pa];
        let b = neighbor_positions[pb];

        if nd < audit_cutoff() {
            CANON_EVALS.fetch_add(1, Ordering::Relaxed);
            let sign = in_circle_sphere_sign(generator_raw, a, b, neighbor_raw);
            if sign == 0 {
                EXACT_TIES.fetch_add(1, Ordering::Relaxed);
                TIE_HIST[bucket].fetch_add(1, Ordering::Relaxed);
            } else {
                let local_keep = d >= -hp.eps;
                let canonical_keep = sign < 0;
                if local_keep != canonical_keep {
                    DISAGREE_HIST[bucket].fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // Paired collection: record this decision under its abstract
        // question key so cross-cell answers can be matched up. Pass 1
        // collects below the margin cutoff; pass 2 additionally collects
        // any decision whose key is in the installed filter, at any margin
        // (de-censoring).
        let below_cutoff = nd < pcut;
        if below_cutoff || key_filter.is_some() {
            let (ga, gb) = (neighbor_indices[pa] as u32, neighbor_indices[pb] as u32);
            let g = generator_idx as u32;
            let x = neighbor_idx as u32;
            let mut triple = [g, ga, gb];
            triple.sort_unstable();
            // x in the triple is a degenerate re-clip of an existing
            // constraint; not a shared 4-point question.
            if triple.contains(&x) {
                continue;
            }
            let key = [triple[0], triple[1], triple[2], x];
            let wanted = below_cutoff || key_filter.as_ref().is_some_and(|f| f.contains(&key));
            if wanted {
                batch.push(PairedRecord {
                    key,
                    cell: g,
                    margin: nd as f32,
                    local_keep: d >= -hp.eps,
                    canonical: in_circle_sphere_sign(generator_raw, a, b, neighbor_raw),
                });
            }
        }
    }
    if !batch.is_empty() {
        PAIRED.lock().unwrap().extend(batch);
    }
}

/// Reset the paired-audit collection.
pub fn paired_reset() {
    PAIRED.lock().unwrap().clear();
}

/// Group the paired records by question and report cross-cell agreement.
///
/// The headline statistic is the conflict tail: the largest margin at which
/// some cell holds a local answer that conflicts with a peer's answer (or
/// with canonical) for the same shared question — the quantity EPS_FILTER
/// must dominate for P5 stage 2.
pub fn paired_report() -> String {
    use std::collections::HashMap;
    use std::fmt::Write;

    let records = PAIRED.lock().unwrap().clone();
    let mut groups: HashMap<[u32; 4], Vec<&PairedRecord>> = HashMap::new();
    for r in &records {
        groups.entry(r.key).or_default().push(r);
    }

    let mut out = String::new();
    let mut multi_party = 0u64;
    let mut conflict_groups = 0u64;
    let mut canon_inconsistent = 0u64;
    // Bucketed by the margin of the conflicting entry (each side of a
    // conflict counts at its own margin).
    let mut conflict_hist = [0u64; BUCKETS];
    let mut conflict_max_margin = 0.0f32;
    // Property (b) on shared questions only: local vs canonical.
    let mut canon_disagree_hist = [0u64; BUCKETS];
    let mut canon_disagree_max = 0.0f32;

    for entries in groups.values() {
        let mut cells: Vec<u32> = entries.iter().map(|e| e.cell).collect();
        cells.sort_unstable();
        cells.dedup();

        // Canonical self-check: one question, one exact answer.
        if entries.iter().any(|e| e.canonical != entries[0].canonical) {
            canon_inconsistent += 1;
        }

        if cells.len() < 2 {
            continue;
        }
        multi_party += 1;

        let any_keep = entries.iter().any(|e| e.local_keep);
        let any_cut = entries.iter().any(|e| !e.local_keep);
        if any_keep && any_cut {
            conflict_groups += 1;
            for e in entries {
                // Every entry in a split group conflicts with someone.
                conflict_hist[bucket_for(e.margin as f64)] += 1;
                conflict_max_margin = conflict_max_margin.max(e.margin);
            }
        }

        for e in entries {
            if e.canonical != 0 && e.local_keep != (e.canonical < 0) {
                canon_disagree_hist[bucket_for(e.margin as f64)] += 1;
                canon_disagree_max = canon_disagree_max.max(e.margin);
            }
        }
    }

    writeln!(
        out,
        "paired: records={} questions={} multi_party={} conflict_groups={}          canon_self_check_failures={}",
        records.len(),
        groups.len(),
        multi_party,
        conflict_groups,
        canon_inconsistent,
    )
    .unwrap();
    writeln!(
        out,
        "  cross-cell conflict tail: max margin {:.3e}; canonical-disagreement          tail (shared questions): max margin {:.3e}",
        conflict_max_margin, canon_disagree_max
    )
    .unwrap();
    writeln!(out, "  margin bucket        conflicts  canon-disagrees").unwrap();
    for k in 0..BUCKETS {
        if conflict_hist[k] == 0 && canon_disagree_hist[k] == 0 {
            continue;
        }
        let label = if k == 0 {
            ">= 1e-1        ".to_string()
        } else if k == BUCKETS - 1 {
            format!("<  1e-{}        ", BUCKETS - 1)
        } else {
            format!("1e-{:<2} .. 1e-{:<2}", k + 1, k)
        };
        writeln!(
            out,
            "  {label} {:>10}  {:>10}",
            conflict_hist[k], canon_disagree_hist[k]
        )
        .unwrap();
    }
    out
}

/// Dump every collected paired record whose question involves any of the
/// given generator ids (defect-site anatomy; probe use).
pub fn paired_dump_involving(ids: &[u32]) -> String {
    use std::collections::HashMap;
    use std::fmt::Write;
    let records = PAIRED.lock().unwrap().clone();
    let mut groups: HashMap<[u32; 4], Vec<&PairedRecord>> = HashMap::new();
    for r in &records {
        if r.key.iter().any(|k| ids.contains(k)) {
            groups.entry(r.key).or_default().push(r);
        }
    }
    let mut keys: Vec<_> = groups.keys().copied().collect();
    keys.sort_unstable();
    let mut out = String::new();
    for key in keys {
        let entries = &groups[&key];
        writeln!(out, "  q={key:?} canonical={}", entries[0].canonical).unwrap();
        for e in entries {
            writeln!(
                out,
                "    cell={} keep={} margin={:.3e}",
                e.cell, e.local_keep, e.margin
            )
            .unwrap();
        }
    }
    out
}

/// Quad-level coherence report: group records by the sorted 4-point SET
/// (not the (triple, x) phrasing). The two phrasings "(g,h,t1) vs t2" and
/// "(g,h,t2) vs t1" are the same determinant with opposite parity, so a
/// 4-set's records are mutually coherent iff they ALL agree with canonical
/// or ALL disagree; mixed agreement is a genuine contradiction (two corners
/// that cannot canonically coexist) — the thing the (triple, x)-keyed
/// pairing structurally could not see.
pub fn paired_quad_report() -> String {
    use std::collections::HashMap;
    use std::fmt::Write;

    let records = PAIRED.lock().unwrap().clone();
    let mut quads: HashMap<[u32; 4], Vec<&PairedRecord>> = HashMap::new();
    for r in &records {
        if r.canonical == 0 {
            continue; // exact ties: SoS territory, not coherence
        }
        let mut q = r.key;
        q.sort_unstable();
        quads.entry(q).or_default().push(r);
    }

    let mut multi_record = 0u64;
    let mut contradictions = 0u64;
    let mut cross_cell_contradictions = 0u64;
    let mut contra_margin_hist = [0u64; BUCKETS];
    let mut contra_max_margin = 0.0f32;
    let mut contra_dump: Vec<String> = Vec::new();

    for (q, entries) in &quads {
        if entries.len() < 2 {
            continue;
        }
        multi_record += 1;
        let agree = |e: &PairedRecord| e.local_keep == (e.canonical < 0);
        let any_agree = entries.iter().any(|e| agree(e));
        let any_disagree = entries.iter().any(|e| !agree(e));
        if any_agree && any_disagree {
            contradictions += 1;
            let mut cells: Vec<u32> = entries.iter().map(|e| e.cell).collect();
            cells.sort_unstable();
            cells.dedup();
            if cells.len() >= 2 {
                cross_cell_contradictions += 1;
            }
            for e in entries {
                if !agree(e) {
                    contra_margin_hist[bucket_for(e.margin as f64)] += 1;
                    contra_max_margin = contra_max_margin.max(e.margin);
                }
            }
            if contra_dump.len() < 12 {
                let mut line = format!("    quad={q:?}");
                for e in entries {
                    write!(
                        line,
                        " [x={} cell={} keep={} canon={} m={:.2e}]",
                        e.key[3], e.cell, e.local_keep, e.canonical, e.margin
                    )
                    .unwrap();
                }
                contra_dump.push(line);
            }
        }
    }

    let mut out = String::new();
    writeln!(
        out,
        "quads: total={} multi_record={multi_record} CONTRADICTIONS={contradictions}          (cross_cell={cross_cell_contradictions}) max_disagreeing_margin={contra_max_margin:.3e}",
        quads.len()
    )
    .unwrap();
    writeln!(out, "  contradiction margin buckets (disagreeing records):").unwrap();
    for (k, c) in contra_margin_hist.iter().enumerate() {
        if *c == 0 {
            continue;
        }
        let label = if k == 0 {
            ">= 1e-1".to_string()
        } else {
            format!("1e-{:<2}..1e-{:<2}", k + 1, k)
        };
        writeln!(out, "    {label} {c}").unwrap();
    }
    for line in &contra_dump {
        writeln!(out, "{line}").unwrap();
    }
    out
}

/// Reset all shadow counters.
pub fn reset() {
    AUDITED.store(0, Ordering::Relaxed);
    SKIPPED_SYNTHETIC.store(0, Ordering::Relaxed);
    CANON_EVALS.store(0, Ordering::Relaxed);
    EXACT_TIES.store(0, Ordering::Relaxed);
    for b in &MARGIN_HIST {
        b.store(0, Ordering::Relaxed);
    }
    for b in &DISAGREE_HIST {
        b.store(0, Ordering::Relaxed);
    }
    for b in &TIE_HIST {
        b.store(0, Ordering::Relaxed);
    }
}

/// Formatted dump of the shadow counters.
pub fn report() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let audited = AUDITED.load(Ordering::Relaxed);
    writeln!(
        out,
        "p5_shadow: audited={} skipped_synthetic={} canon_evals={} exact_ties={}",
        audited,
        SKIPPED_SYNTHETIC.load(Ordering::Relaxed),
        CANON_EVALS.load(Ordering::Relaxed),
        EXACT_TIES.load(Ordering::Relaxed),
    )
    .unwrap();
    // Snapshot the histograms once (atomics may be live under parallel builds).
    let margin: Vec<u64> = MARGIN_HIST
        .iter()
        .map(|b| b.load(Ordering::Relaxed))
        .collect();
    let disagree: Vec<u64> = DISAGREE_HIST
        .iter()
        .map(|b| b.load(Ordering::Relaxed))
        .collect();
    let tie: Vec<u64> = TIE_HIST.iter().map(|b| b.load(Ordering::Relaxed)).collect();
    let total: u64 = margin.iter().sum();

    // Superset BAND (§5): the smallest bucket index (= LARGEST margin) carrying
    // any disagreement or tie. The BAND must reach that margin (~1e-k), so the
    // trip rate is the cumulative fraction of decisions at margin <= 1e-k, i.e.
    // buckets j >= k. `None` => no leak at any margin => BAND -> 0 (ideal).
    let min_leak = (0..BUCKETS).find(|&k| disagree[k] + tie[k] > 0);
    match min_leak {
        Some(k) => {
            let trip: u64 = margin[k..].iter().sum();
            let trip_pct = if total > 0 {
                100.0 * trip as f64 / total as f64
            } else {
                0.0
            };
            let leak_total: u64 = disagree.iter().chain(tie.iter()).sum();
            writeln!(
                out,
                "  SUPERSET: min-leak band ~1e-{k} (largest disagree/tie margin); \
                 trip rate {trip_pct:.4}% ({trip}/{total}); total leaks={leak_total}"
            )
            .unwrap();
        }
        None => {
            writeln!(
                out,
                "  SUPERSET: no disagreement/tie at any audited margin (BAND -> 0); total={total}"
            )
            .unwrap();
        }
    }

    writeln!(
        out,
        "  margin nd=|d|/|n|      decisions  disagreements         ties   cum_trip%"
    )
    .unwrap();
    // Cumulative from the largest-margin bucket (k=0) downward.
    let mut cum: u64 = 0;
    for k in 0..BUCKETS {
        cum += margin[k];
        let (m, dis, ti) = (margin[k], disagree[k], tie[k]);
        if m == 0 && dis == 0 && ti == 0 {
            continue;
        }
        let cum_pct = if total > 0 {
            100.0 * cum as f64 / total as f64
        } else {
            0.0
        };
        let label = if k == 0 {
            ">= 1e-1        ".to_string()
        } else if k == BUCKETS - 1 {
            format!("<  1e-{}        ", BUCKETS - 1)
        } else {
            format!("1e-{:<2} .. 1e-{:<2}", k + 1, k)
        };
        writeln!(out, "  {label} {m:>12}  {dis:>12} {ti:>12}  {cum_pct:>9.4}").unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_point(colat: f32, lon_deg: f32) -> Vec3 {
        let lon = lon_deg.to_radians();
        Vec3::new(
            colat.sin() * lon.cos(),
            colat.sin() * lon.sin(),
            colat.cos(),
        )
    }

    /// Known geometry: three points on the colatitude-theta circle around
    /// +z; the circumcircle pole is +z. Points at smaller colatitude are
    /// inside the cap, larger are outside.
    #[test]
    fn in_circle_sign_matches_known_geometry() {
        let theta = 0.3f32;
        let g = ring_point(theta, 0.0);
        let a = ring_point(theta, 120.0);
        let b = ring_point(theta, 240.0);

        let h_inside = ring_point(theta * 0.5, 60.0);
        let h_outside = ring_point(theta * 1.5, 60.0);

        assert_eq!(in_circle_sphere_sign(g, a, b, h_inside), 1);
        assert_eq!(in_circle_sphere_sign(g, a, b, h_outside), -1);
        // Swapping a/b (reversing triple orientation) must not change the
        // geometric answer.
        assert_eq!(in_circle_sphere_sign(g, b, a, h_inside), 1);
        assert_eq!(in_circle_sphere_sign(g, b, a, h_outside), -1);
        // The roles of g and the third ring point are symmetric.
        assert_eq!(in_circle_sphere_sign(a, g, b, h_inside), 1);
        // Exactly-on-circle fourth point: f32 ring points are not exactly
        // cocircular in general, so only check the planted exact tie below.
    }

    /// An exactly cocircular quadruple (shared z, symmetric lattice) must
    /// return 0.
    #[test]
    fn in_circle_sign_exact_tie() {
        let z = 0.5f32;
        let r = (1.0f32 - z * z).sqrt();
        let g = Vec3::new(r, 0.0, z);
        let a = Vec3::new(-r, 0.0, z);
        let b = Vec3::new(0.0, r, z);
        let h = Vec3::new(0.0, -r, z);
        assert_eq!(in_circle_sphere_sign(g, a, b, h), 0);
    }

    /// The exact sign must match a naive f64 evaluation on well-separated
    /// random quadruples (where naive arithmetic is reliable).
    #[test]
    fn in_circle_sign_matches_naive_on_clear_cases() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let random_unit = |rng: &mut rand_chacha::ChaCha8Rng| loop {
            let v = Vec3::new(
                rng.gen_range(-1.0f32..1.0),
                rng.gen_range(-1.0f32..1.0),
                rng.gen_range(-1.0f32..1.0),
            );
            let len = v.length();
            if len > 1e-2 && len < 1.0 {
                return v / len;
            }
        };
        let naive = |g: Vec3, a: Vec3, b: Vec3, h: Vec3| -> f64 {
            let gd = g.as_dvec3();
            let det3 = |r0: glam::DVec3, r1: glam::DVec3, r2: glam::DVec3| r0.cross(r1).dot(r2);
            let d1 = det3(a.as_dvec3() - gd, b.as_dvec3() - gd, h.as_dvec3() - gd);
            let d2 = det3(a.as_dvec3(), b.as_dvec3(), gd);
            d1.signum() * d2.signum()
        };
        let mut checked = 0;
        for _ in 0..2000 {
            let (g, a, b, h) = (
                random_unit(&mut rng),
                random_unit(&mut rng),
                random_unit(&mut rng),
                random_unit(&mut rng),
            );
            let gd = g.as_dvec3();
            let det3 = |r0: glam::DVec3, r1: glam::DVec3, r2: glam::DVec3| r0.cross(r1).dot(r2);
            let d1 = det3(a.as_dvec3() - gd, b.as_dvec3() - gd, h.as_dvec3() - gd);
            let d2 = det3(a.as_dvec3(), b.as_dvec3(), gd);
            // Only compare clear cases: naive f64 signs are trustworthy only
            // well away from zero (error ~1e-15 on O(1) inputs).
            if d1.abs() < 1e-9 || d2.abs() < 1e-9 {
                continue;
            }
            let n = naive(g, a, b, h);
            checked += 1;
            assert_eq!(
                in_circle_sphere_sign(g, a, b, h),
                n as i8,
                "exact vs naive mismatch for g={g:?} a={a:?} b={b:?} h={h:?}"
            );
        }
        assert!(checked > 1500, "too few clear cases: {checked}");
    }
}
