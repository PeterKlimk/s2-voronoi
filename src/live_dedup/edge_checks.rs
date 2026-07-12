//! Edge-check bookkeeping helpers for live dedup.

use std::mem;

use super::binning::BinAssignment;
use super::packed::{pack_edge, pack_ref, DEFERRED, INVALID_INDEX};
use super::shard::{ShardDedup, ShardState};
use super::types::{
    BinId, EdgeCheck, EdgeCheckOverflow, EdgeKey, EdgeOverflowLocal, EdgeToLater, LocalId,
    UnresolvedEdgeMismatch, UnresolvedEdgeOrigin,
};
use super::with_two_mut;
use crate::knn_clipping::cell_build::VertexKey;
use crate::packed_layout::PackedSlotLayout;
use crate::timing::Timer;
use std::time::Duration;

#[inline]
pub(crate) fn unpack_edge_key(key: EdgeKey) -> (u32, u32) {
    let v: u64 = key.into();
    (v as u32, (v >> 32) as u32)
}

/// Sentinel "third" for an endpoint whose vertex key does not name both edge
/// endpoints. Never a valid generator id (keys hold real generator indices),
/// and inert in `reconcile_edge_endpoints`: a malformed endpoint neither
/// full-matches nor patches, so bad attribution can only widen the defect
/// record set, never silently share a vertex id.
pub(super) const MALFORMED_THIRD: u32 = u32::MAX;

/// The endpoint's "third" generator — the key entry that is neither edge
/// endpoint. When the key contains `{a, b, third}`, XOR is self-canceling
/// (`x ^ x = 0`), so `key[0] ^ key[1] ^ key[2] ^ a ^ b = third`.
///
/// `None` when the key does NOT contain both endpoints: a malformed triple
/// attribution from a near-degenerate clip. This was long believed
/// unreachable (a debug assert aborted here), but dense near-cocircular
/// `mega` inputs give it a natural trigger — the fallback extract resolves a
/// corner via a split plane within tolerance, position dedup collapses the
/// split micro-edge, and the surviving vertex keeps the split plane in its
/// key while the edge sequence continues with the original neighbor. The
/// wrong attribution is a handled defect (callers record it as
/// `EndpointKeyMismatch`, feeding the reconcile + repair pipeline that
/// provably restores strict validity), not an invariant violation — the same
/// reachable-and-handled family as the four asserts demoted for the strict
/// planar keep rule.
#[inline]
pub(super) fn third_for_edge_endpoint(key: VertexKey, a: u32, b: u32) -> Option<u32> {
    (key.contains(&a) && key.contains(&b)).then(|| key[0] ^ key[1] ^ key[2] ^ a ^ b)
}

/// Unchecked XOR "third" — valid only for cells whose extractor guarantees
/// key/edge consistency (`CellOutputBuffer::edge_keys_verified`); the hot
/// path for every gnomonic-built cell. Measured +1.3% whole-build
/// instructions when the membership checks ran here per edge instead.
#[inline]
fn xor_third(key: VertexKey, a: u32, b: u32) -> u32 {
    key[0] ^ key[1] ^ key[2] ^ a ^ b
}

/// Both endpoint thirds for edge `key` on a VERIFIED cell (unchecked XOR).
#[inline]
fn thirds_verified(key: EdgeKey, endpoint_keys: [VertexKey; 2]) -> [u32; 2] {
    let (a, b) = unpack_edge_key(key);
    [
        xor_third(endpoint_keys[0], a, b),
        xor_third(endpoint_keys[1], a, b),
    ]
}

/// Per-cell dispatcher: unchecked XOR thirds when the extractor verified
/// key/edge consistency, the malformed-endpoint-recording path otherwise.
#[inline]
pub(super) fn thirds_for_emit(
    keys_verified: bool,
    unresolved: &mut Vec<UnresolvedEdgeMismatch>,
    key: EdgeKey,
    endpoint_keys: [VertexKey; 2],
) -> [u32; 2] {
    if keys_verified {
        thirds_verified(key, endpoint_keys)
    } else {
        thirds_or_record(unresolved, key, endpoint_keys)
    }
}

/// Both endpoint thirds for edge `key`, recording an `EndpointKeyMismatch`
/// defect (once per side) when either endpoint's vertex key is malformed;
/// malformed endpoints carry the inert [`MALFORMED_THIRD`] sentinel. Keeping
/// the record at the site that computed the third makes detection
/// deterministic — the alternative (shipping a garbage XOR value and relying
/// on it failing to match the other side) is probabilistic in principle, and
/// XOR of structured ids is not uniform.
#[inline]
pub(super) fn thirds_or_record(
    unresolved: &mut Vec<UnresolvedEdgeMismatch>,
    key: EdgeKey,
    endpoint_keys: [VertexKey; 2],
) -> [u32; 2] {
    let (a, b) = unpack_edge_key(key);
    let t0 = third_for_edge_endpoint(endpoint_keys[0], a, b);
    let t1 = third_for_edge_endpoint(endpoint_keys[1], a, b);
    if t0.is_none() || t1.is_none() {
        unresolved.push(UnresolvedEdgeMismatch {
            key,
            origin: UnresolvedEdgeOrigin::EndpointKeyMismatch,
        });
    }
    [t0.unwrap_or(MALFORMED_THIRD), t1.unwrap_or(MALFORMED_THIRD)]
}

/// Reconcile the two directed sides of a shared edge by their endpoint
/// "third" generators. The sides traverse the edge in opposite directions
/// (reverse winding), so a full match is `other[0] == mine[1] &&
/// other[1] == mine[0]`, patched crosswise; otherwise every shared endpoint
/// is patched individually (cross-slot search) and `false` is returned so
/// the caller records the unresolved mismatch.
///
/// `patch(my_k, other_k)` receives endpoint positions on each side; the
/// in-shard path writes one direction, the cross-bin overflow path patches
/// both sides from the same pairing.
///
/// [`MALFORMED_THIRD`] is inert: a malformed endpoint never full-matches and
/// never patches (two malformed sides agreeing on the sentinel is not
/// endpoint agreement), while the well-formed endpoint of a half-malformed
/// edge still patches individually.
#[inline]
fn reconcile_edge_endpoints(
    my_thirds: [u32; 2],
    other_thirds: [u32; 2],
    mut patch: impl FnMut(usize, usize),
) -> bool {
    if other_thirds[0] == my_thirds[1]
        && other_thirds[1] == my_thirds[0]
        && other_thirds[0] != MALFORMED_THIRD
        && other_thirds[1] != MALFORMED_THIRD
    {
        patch(0, 1);
        patch(1, 0);
        return true;
    }
    for (ok, other) in other_thirds.iter().enumerate() {
        for (mk, mine) in my_thirds.iter().enumerate() {
            if other == mine && *other != MALFORMED_THIRD {
                patch(mk, ok);
            }
        }
    }
    false
}

impl ShardDedup {
    pub(super) fn push_edge_check(&mut self, local: LocalId, check: EdgeCheck) {
        let local_idx = local.as_usize();
        debug_assert!(
            local_idx < self.edge_checks.len(),
            "edge check local out of bounds"
        );

        let slot = &mut self.edge_checks[local_idx];
        if slot.capacity() == 0 {
            if let Some(mut v) = self.edge_check_pool.pop() {
                v.clear();
                *slot = v;
            }
        }
        slot.push(check);
    }

    pub(crate) fn take_edge_checks(&mut self, local: LocalId) -> Vec<EdgeCheck> {
        let local_idx = local.as_usize();
        debug_assert!(
            local_idx < self.edge_checks.len(),
            "edge check local out of bounds"
        );
        mem::take(&mut self.edge_checks[local_idx])
    }

    pub(super) fn recycle_edge_checks(&mut self, v: Vec<EdgeCheck>) {
        self.edge_check_pool.push(v);
    }
}

/// Fused collect + resolve: iterates cell edges once, immediately resolving
/// edges to earlier neighbors against incoming checks while collecting
/// edges_to_later and edges_overflow for the emit phase.
///
/// This eliminates the edges_to_earlier intermediate vec
#[inline(always)]
fn assert_cell_output_lengths<P>(
    output_buffer: &crate::knn_clipping::cell_build::CellOutputBuffer<P>,
    vertex_indices_len: usize,
) -> usize {
    let n = output_buffer.vertices.len();
    assert!(
        n >= 2
            && output_buffer.edge_neighbor_slots.len() == n
            && output_buffer.edge_neighbor_globals.len() == n
            && output_buffer.edge_neighbor_eps.len() == n
            && vertex_indices_len == n,
        "cell output arrays out of sync"
    );
    n
}

#[cfg_attr(feature = "profiling", inline(never))]
// The argument list is the fused collect+resolve data flow; the two
// output vecs are caller-owned scratch reused across cells.
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_and_resolve_cell_edges<P: super::types::VertexPosition>(
    cell_idx: u32,
    shard_ctx: &mut super::emit::ShardContext<'_, P>,
    output_buffer: &crate::knn_clipping::cell_build::CellOutputBuffer<P>,
    assignment: &BinAssignment,
    incoming_checks: Vec<EdgeCheck>,
    vertex_indices: &mut [u32],
    edges_to_later: &mut Vec<EdgeToLater>,
    edges_overflow: &mut Vec<EdgeOverflowLocal>,
) {
    let shard = &mut *shard_ctx.shard;
    let local = shard_ctx.local;
    let bin = shard_ctx.bin;

    let cell_vertices = &output_buffer.vertices;
    let edge_neighbor_slots = &output_buffer.edge_neighbor_slots;
    let edge_neighbor_globals = &output_buffer.edge_neighbor_globals;
    let keys_verified = output_buffer.edge_keys_verified;

    let n = assert_cell_output_lengths(output_buffer, vertex_indices.len());
    edges_to_later.clear();
    edges_overflow.clear();

    #[cfg(debug_assertions)]
    {
        let d_bin = assignment.generator_bin[cell_idx as usize];
        let d_local = assignment.generator_local(cell_idx as usize);
        debug_assert_eq!(d_bin, bin, "bin index mismatch");
        debug_assert_eq!(d_local, local, "local index mismatch");
    }
    let local_u32 = local.as_u32();

    // Incoming checks were taken earlier (e.g. for geometry seeding).
    let incoming_count = incoming_checks.len();

    // Track which incoming checks have been consumed. The common case fits
    // the u64 bitmask; a dense near-cocircular cell can receive more than 64
    // checks (mega inputs produce >64-vertex cells), so the surplus spills to
    // a Vec (no allocation in the common case). Exactness matters: the
    // unmatched loop below is detection-critical, and a wrongly-"consumed"
    // check (as the old release build's masked `1 << idx` shift could produce
    // past 64) is a silently dropped defect record.
    let mut matched: u64 = 0;
    let mut matched_spill: Vec<bool> = vec![false; incoming_count.saturating_sub(64)];

    let layout = PackedSlotLayout::new(
        &assignment.slot_gen_map,
        assignment.local_shift,
        assignment.local_mask,
    );

    // Process all edges
    for i in 0..n {
        let j = if i + 1 == n { 0 } else { i + 1 };
        // Length equality was checked once above; `i` and cyclic `j` are in
        // `0..n`. Keep the hot edge loop free of repeated bounds checks.
        let slot = unsafe { *edge_neighbor_slots.get_unchecked(i) };

        if slot == u32::MAX {
            continue;
        }

        // Derive global index from propagated array (sequential access)
        let neighbor = unsafe { *edge_neighbor_globals.get_unchecked(i) };
        if neighbor == cell_idx {
            continue;
        }

        let locals = [i as u8, j as u8];
        let edge_key = pack_edge(cell_idx, neighbor);
        let (bin_b, local_b) = layout.bin_local(slot);
        let bin_b = BinId::from(bin_b);
        let local_b = LocalId::from(local_b);

        if bin != bin_b {
            // Cross-bin edge → overflow
            let side = if cell_idx <= neighbor { 0 } else { 1 };
            edges_overflow.push(EdgeOverflowLocal {
                key: edge_key,
                locals,
                side,
            });
        } else if local_u32 < local_b.as_u32() {
            // Edge to later neighbor → collect for emit
            edges_to_later.push(EdgeToLater {
                key: edge_key,
                local_b,
                locals,
            });
        } else {
            // Edge to earlier neighbor → resolve immediately.
            // Search Vec for matching check (cache-friendly sequential access)
            let found = incoming_checks
                .iter()
                .position(|check| check.key == edge_key)
                .map(|idx| (idx, incoming_checks[idx]));

            if let Some((found_idx, check)) = found {
                // One incoming check matching two of this cell's edges
                // (duplicate side) was long believed impossible. The strict
                // planar keep rule (PLANE_CLIP_EPS_INSIDE = 0.0) makes it
                // reachable: a sliver can give a cell two edges to one
                // neighbor. Release has no assert here and already produces
                // strictly-valid output on these inputs (the duplicate
                // re-resolves, any displaced check surfaces via the
                // unmatched-check loop below, and repair + the output-
                // invariant scan restore validity — verified by the plane
                // battery and the strict-plane campaign). So this is a
                // handled defect, not an invariant violation; debug builds
                // must not abort (this only makes debug match release).
                if found_idx < 64 {
                    matched |= 1u64 << found_idx;
                } else {
                    matched_spill[found_idx - 64] = true;
                }

                let my_thirds = thirds_for_emit(
                    keys_verified,
                    &mut shard.output.unresolved_edges,
                    edge_key,
                    [unsafe { cell_vertices.get_unchecked(i).0 }, unsafe {
                        cell_vertices.get_unchecked(j).0
                    }],
                );

                let full = reconcile_edge_endpoints(my_thirds, check.thirds, |mk, ck| {
                    let local_idx = locals[mk] as usize;
                    // A local endpoint already carrying a DIFFERENT global id
                    // is a vertex-identity sliver (one corner committed under
                    // two triple attributions) — reachable under the strict
                    // planar keep rule (PLANE_CLIP_EPS_INSIDE = 0.0), not an
                    // invariant violation. Release adopts the incoming id
                    // (last write wins) and the output-invariant scan + repair
                    // restore strict validity (verified by the plane battery
                    // and the strict campaign); the validate / VORONOI_MESH_VERIFY
                    // gates remain the production catch. Debug must not abort
                    // (this only makes debug match release behavior).
                    unsafe {
                        *vertex_indices.get_unchecked_mut(local_idx) = check.indices[ck];
                    }
                });
                // A malformed endpoint on either side already produced an
                // EndpointKeyMismatch record (here or at the emitter), and a
                // sentinel-bearing side can never fully reconcile — recording
                // the inevitable mismatch again would double-report the edge.
                if !full
                    && !my_thirds.contains(&MALFORMED_THIRD)
                    && !check.thirds.contains(&MALFORMED_THIRD)
                {
                    shard.output.unresolved_edges.push(UnresolvedEdgeMismatch {
                        key: edge_key,
                        origin: UnresolvedEdgeOrigin::InBinThirdsMismatch,
                    });
                }
            } else {
                // Missing side - earlier neighbor didn't emit check
                shard.output.unresolved_edges.push(UnresolvedEdgeMismatch {
                    key: edge_key,
                    origin: UnresolvedEdgeOrigin::InBinMissingCheck,
                });
            }
        }
    }

    // Fast path: if all incoming checks matched, skip the unmatched loop
    let all_mask = if incoming_count >= 64 {
        u64::MAX
    } else {
        (1u64 << incoming_count) - 1
    };
    let all_matched =
        incoming_count == 0 || (matched == all_mask && matched_spill.iter().all(|&m| m));

    if !all_matched {
        for (idx, check) in incoming_checks.iter().enumerate() {
            let consumed = if idx < 64 {
                matched & (1u64 << idx) != 0
            } else {
                matched_spill[idx - 64]
            };
            if !consumed {
                shard.output.unresolved_edges.push(UnresolvedEdgeMismatch {
                    key: check.key,
                    origin: UnresolvedEdgeOrigin::InBinUnconsumedCheck,
                });
            }
        }
    }

    if incoming_count > 0 {
        shard.dedup.recycle_edge_checks(incoming_checks);
    }
}

/// Timing breakdown for the overflow resolution phase.
pub(super) struct OverflowResolveTiming {
    pub sort: Duration,
    pub match_: Duration,
}

/// Matches edge checks across shard boundaries.
///
/// Since cross-bin edges are emitted twice (once by each bin), we sort all overflow
/// records and match them by edge key. This allows propagating vertex indices
/// between bins without global communication during the main clipping phase.
///
#[cfg_attr(feature = "profiling", inline(never))]
pub(super) fn resolve_edge_check_overflow<P: super::types::VertexPosition>(
    shards: &mut [ShardState<P>],
    edge_check_overflow: &mut [EdgeCheckOverflow],
    unresolved_edges: &mut Vec<UnresolvedEdgeMismatch>,
) -> OverflowResolveTiming {
    let t_edge_sort = Timer::start();
    // Resolution only requires contiguous equal-key runs. Within a two-record run, side equality
    // and reverse-winding endpoint patching are symmetric; larger runs are deferred as a whole.
    edge_check_overflow.sort_unstable_by_key(|entry| entry.key);
    let edge_checks_overflow_sort_time = t_edge_sort.elapsed();

    let t_edge_match = Timer::start();
    // Returns true when the slot already held a DIFFERENT concrete
    // reference (duplicate same-key vertices reaching this cell through two
    // edges); the caller records the conflict so repair sees the site even
    // when the thirds fully agree.
    let patch_slot = |slot: &mut u64, owner_bin: BinId, idx: u32| -> bool {
        let packed = pack_ref(owner_bin, idx);
        // A slot already holding a DIFFERENT concrete reference is a real,
        // handled defect (recorded by the caller as CrossBinSlotConflict so
        // repair sees the site); not an invariant violation, so no assert.
        let conflict = *slot != DEFERRED && *slot != packed;
        *slot = packed;
        conflict
    };
    let mut i = 0usize;
    while i < edge_check_overflow.len() {
        let key = edge_check_overflow[i].key;
        let mut run_end = i + 1;
        while run_end < edge_check_overflow.len() && edge_check_overflow[run_end].key == key {
            run_end += 1;
        }
        let run = &edge_check_overflow[i..run_end];

        if run.len() == 1 {
            unresolved_edges.push(UnresolvedEdgeMismatch {
                key,
                origin: UnresolvedEdgeOrigin::CrossBinSingleSided,
            });
        } else if run.len() == 2 {
            let a = edge_check_overflow[i];
            let b = edge_check_overflow[i + 1];
            if a.side == b.side {
                // Two same-side overflow checks for one edge key: a
                // duplicate cross-bin attribution (a marginal corner kept by
                // an extra cell). Long believed unreachable, but the strict
                // planar keep rule (PLANE_CLIP_EPS_INSIDE = 0.0) gives it a
                // natural trigger — e.g. the bounded-plane fixture in the
                // `locate` suite. It is recorded as CrossBinDuplicateSide and
                // repaired to strict validity, so it is a handled defect, not
                // an invariant violation (hence no abort).
                unresolved_edges.push(UnresolvedEdgeMismatch {
                    key: a.key,
                    origin: UnresolvedEdgeOrigin::CrossBinDuplicateSide,
                });
            } else {
                let (a_shard, b_shard) =
                    with_two_mut(shards, a.source_bin.as_usize(), b.source_bin.as_usize());

                // The two sides traverse the edge in reverse winding;
                // every endpoint pairing patches both shards symmetrically.
                let mut conflict = false;
                let full = reconcile_edge_endpoints(b.thirds, a.thirds, |bk, ak| {
                    if a.indices[ak] != INVALID_INDEX {
                        conflict |= patch_slot(
                            &mut b_shard.output.cell_indices[b.slots[bk] as usize],
                            a.source_bin,
                            a.indices[ak],
                        );
                    }
                    if b.indices[bk] != INVALID_INDEX {
                        conflict |= patch_slot(
                            &mut a_shard.output.cell_indices[a.slots[ak] as usize],
                            b.source_bin,
                            b.indices[bk],
                        );
                    }
                });
                if !full {
                    // A malformed endpoint (MALFORMED_THIRD) was already
                    // recorded as EndpointKeyMismatch by the emitting side and
                    // can never fully reconcile — don't double-report it as a
                    // thirds mismatch.
                    if !a.thirds.contains(&MALFORMED_THIRD) && !b.thirds.contains(&MALFORMED_THIRD)
                    {
                        unresolved_edges.push(UnresolvedEdgeMismatch {
                            key: a.key,
                            origin: UnresolvedEdgeOrigin::CrossBinThirdsMismatch,
                        });
                    }
                } else if conflict {
                    unresolved_edges.push(UnresolvedEdgeMismatch {
                        key: a.key,
                        origin: UnresolvedEdgeOrigin::CrossBinSlotConflict,
                    });
                }
            }
        } else {
            // A normal cross-bin edge has exactly one record per side. Three
            // or more records mean at least one cell emitted duplicate edges
            // for this key. Pairing an arbitrary opposite-side subset would
            // make patches depend on unstable ordering within a side, so leave
            // the whole run to the deterministic vertex-key fallback.
            unresolved_edges.push(UnresolvedEdgeMismatch {
                key,
                origin: UnresolvedEdgeOrigin::CrossBinDuplicateSide,
            });
            let first_side = run[0].side;
            if run.iter().all(|entry| entry.side == first_side) {
                unresolved_edges.push(UnresolvedEdgeMismatch {
                    key,
                    origin: UnresolvedEdgeOrigin::CrossBinSingleSided,
                });
            }
        }
        i = run_end;
    }

    let edge_checks_overflow_match_time = t_edge_match.elapsed();
    OverflowResolveTiming {
        sort: edge_checks_overflow_sort_time,
        match_: edge_checks_overflow_match_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "cell output arrays out of sync")]
    fn cell_output_length_mismatch_panics_before_collection() {
        let mut output = crate::live_dedup::CellOutputBuffer::default();
        output.vertices.resize(2, ([0, 1, 2], glam::Vec3::ZERO));
        output.edge_neighbor_slots.resize(1, u32::MAX);
        output.edge_neighbor_globals.resize(2, u32::MAX);
        output.edge_neighbor_eps.resize(2, 0.0);
        assert_cell_output_lengths(&output, 2);
    }

    #[test]
    fn third_requires_both_endpoints() {
        assert_eq!(third_for_edge_endpoint([1, 5, 9], 1, 9), Some(5));
        assert_eq!(third_for_edge_endpoint([1, 5, 9], 5, 1), Some(9));
        // Malformed attribution: the key lacks one or both endpoints.
        assert_eq!(third_for_edge_endpoint([1, 5, 9], 1, 7), None);
        assert_eq!(third_for_edge_endpoint([1, 5, 9], 2, 3), None);
    }

    #[test]
    fn malformed_thirds_never_reconcile() {
        // Half-malformed edge: no full match (the sentinel side is inert),
        // but the well-formed endpoint still patches individually.
        let mut patched = Vec::new();
        let full =
            reconcile_edge_endpoints([MALFORMED_THIRD, 7], [7, MALFORMED_THIRD], |mk, ok| {
                patched.push((mk, ok))
            });
        assert!(!full);
        assert_eq!(patched, vec![(1, 0)]);

        // Fully malformed on both sides: sentinel agreement is not endpoint
        // agreement — nothing matches, nothing patches.
        patched.clear();
        let full = reconcile_edge_endpoints(
            [MALFORMED_THIRD, MALFORMED_THIRD],
            [MALFORMED_THIRD, MALFORMED_THIRD],
            |mk, ok| patched.push((mk, ok)),
        );
        assert!(!full);
        assert!(patched.is_empty());
    }

    #[test]
    fn thirds_or_record_flags_malformed_endpoint() {
        let key = pack_edge(10, 20);
        let mut unresolved = Vec::new();
        // Well-formed: both endpoint keys contain both edge endpoints.
        let thirds = thirds_or_record(&mut unresolved, key, [[5, 10, 20], [10, 20, 30]]);
        assert_eq!(thirds, [5, 30]);
        assert!(unresolved.is_empty());

        // One malformed endpoint: sentinel third + one EndpointKeyMismatch.
        let thirds = thirds_or_record(&mut unresolved, key, [[5, 10, 20], [10, 25, 30]]);
        assert_eq!(thirds, [5, MALFORMED_THIRD]);
        assert_eq!(unresolved.len(), 1);
        assert_eq!(
            unresolved[0].origin,
            UnresolvedEdgeOrigin::EndpointKeyMismatch
        );
        assert_eq!(unresolved[0].key, key);
    }
}
