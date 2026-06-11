//! Edge-check bookkeeping helpers for live dedup.

use std::mem;

use super::binning::BinAssignment;
use super::packed::{pack_edge, pack_ref, DEFERRED, INVALID_INDEX};
use super::shard::{ShardDedup, ShardState};
use super::types::{
    BinId, EdgeCheck, EdgeCheckOverflow, EdgeKey, EdgeOverflowLocal, EdgeToLater, LocalId,
    UnresolvedEdgeMismatch,
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

#[inline]
pub(super) fn third_for_edge_endpoint(key: VertexKey, a: u32, b: u32) -> u32 {
    // key contains {a, b, third} in sorted order
    // XOR is self-canceling: x ^ x = 0
    // So: key[0] ^ key[1] ^ key[2] ^ a ^ b = third
    debug_assert!(
        key.contains(&a) && key.contains(&b),
        "vertex key {:?} does not contain edge endpoints ({}, {})",
        key,
        a,
        b
    );
    key[0] ^ key[1] ^ key[2] ^ a ^ b
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
#[inline]
fn reconcile_edge_endpoints(
    my_thirds: [u32; 2],
    other_thirds: [u32; 2],
    mut patch: impl FnMut(usize, usize),
) -> bool {
    if other_thirds[0] == my_thirds[1] && other_thirds[1] == my_thirds[0] {
        patch(0, 1);
        patch(1, 0);
        return true;
    }
    for (ok, other) in other_thirds.iter().enumerate() {
        for (mk, mine) in my_thirds.iter().enumerate() {
            if other == mine {
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
    let edge_neighbor_eps = &output_buffer.edge_neighbor_eps;

    let n = cell_vertices.len();
    debug_assert_eq!(
        edge_neighbor_slots.len(),
        n,
        "edge neighbor slot data out of sync"
    );
    debug_assert_eq!(
        edge_neighbor_globals.len(),
        n,
        "edge neighbor global data out of sync"
    );
    debug_assert_eq!(
        edge_neighbor_eps.len(),
        n,
        "edge neighbor eps data out of sync"
    );
    edges_to_later.clear();
    edges_overflow.clear();

    debug_assert!(n >= 2, "cell has fewer than 2 vertices");

    #[cfg(debug_assertions)]
    {
        let d_bin = assignment.generator_bin[cell_idx as usize];
        let d_local = assignment.global_to_local[cell_idx as usize];
        debug_assert_eq!(d_bin, bin, "bin index mismatch");
        debug_assert_eq!(d_local, local, "local index mismatch");
    }
    let local_u32 = local.as_u32();

    // Incoming checks were taken earlier (e.g. for geometry seeding).
    let incoming_count = incoming_checks.len();

    // Track which incoming checks we've matched (bitmask, supports up to 64)
    let mut matched: u64 = 0;
    debug_assert!(incoming_count <= 64, "more than 64 incoming edge checks");
    debug_assert!(
        n <= 64,
        "cell has more than 64 vertices, matched bitmask unsafe"
    );

    let layout = PackedSlotLayout::new(
        &assignment.slot_gen_map,
        assignment.local_shift,
        assignment.local_mask,
    );

    // Process all edges
    for i in 0..n {
        let j = if i + 1 == n { 0 } else { i + 1 };
        let slot = edge_neighbor_slots[i];

        if slot == u32::MAX {
            continue;
        }

        // Derive global index from propagated array (sequential access)
        let neighbor = edge_neighbor_globals[i];
        if neighbor == cell_idx {
            continue;
        }

        let locals = [i as u8, j as u8];
        let edge_key = pack_edge(cell_idx, neighbor);
        let hp_eps = edge_neighbor_eps[i];

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
                hp_eps,
            });
        } else {
            // Edge to earlier neighbor → resolve immediately.
            // Search Vec for matching check (cache-friendly sequential access)
            let found = incoming_checks
                .iter()
                .position(|check| check.key == edge_key)
                .map(|idx| (idx, incoming_checks[idx]));

            if let Some((found_idx, check)) = found {
                debug_assert!(
                    matched & (1u64 << found_idx) == 0,
                    "edge check duplicate side"
                );
                matched |= 1u64 << found_idx;

                let (a, b) = unpack_edge_key(edge_key);
                let my_thirds = [
                    third_for_edge_endpoint(cell_vertices[locals[0] as usize].0, a, b),
                    third_for_edge_endpoint(cell_vertices[locals[1] as usize].0, a, b),
                ];

                let full = reconcile_edge_endpoints(my_thirds, check.thirds, |mk, ck| {
                    let local_idx = locals[mk] as usize;
                    debug_assert!(
                        vertex_indices[local_idx] == INVALID_INDEX
                            || vertex_indices[local_idx] == check.indices[ck],
                        "edge check index mismatch"
                    );
                    vertex_indices[local_idx] = check.indices[ck];
                });
                if !full {
                    shard
                        .output
                        .unresolved_edges
                        .push(UnresolvedEdgeMismatch { key: edge_key });
                }
            } else {
                // Missing side - earlier neighbor didn't emit check
                shard
                    .output
                    .unresolved_edges
                    .push(UnresolvedEdgeMismatch { key: edge_key });
            }
        }
    }

    // Fast path: if all incoming checks matched, skip the unmatched loop
    let all_mask = if incoming_count >= 64 {
        u64::MAX
    } else {
        (1u64 << incoming_count) - 1
    };
    let all_matched = incoming_count == 0 || matched == all_mask;

    if !all_matched {
        for (idx, check) in incoming_checks.iter().enumerate() {
            if matched & (1u64 << idx) == 0 {
                shard
                    .output
                    .unresolved_edges
                    .push(UnresolvedEdgeMismatch { key: check.key });
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
    edge_check_overflow.sort_unstable_by_key(|entry| (entry.key, entry.side));
    let edge_checks_overflow_sort_time = t_edge_sort.elapsed();

    let t_edge_match = Timer::start();
    let patch_slot = |slot: &mut u64, owner_bin: BinId, idx: u32| {
        let packed = pack_ref(owner_bin, idx);
        debug_assert!(
            *slot == DEFERRED || *slot == packed,
            "edge check overflow slot mismatch"
        );
        *slot = packed;
    };
    let mut i = 0usize;
    while i < edge_check_overflow.len() {
        if i + 1 < edge_check_overflow.len()
            && edge_check_overflow[i].key == edge_check_overflow[i + 1].key
        {
            let a = edge_check_overflow[i];
            let b = edge_check_overflow[i + 1];
            if a.side == b.side {
                debug_assert!(false, "edge check overflow duplicate side");
                unresolved_edges.push(UnresolvedEdgeMismatch { key: a.key });
            } else {
                let (a_shard, b_shard) =
                    with_two_mut(shards, a.source_bin.as_usize(), b.source_bin.as_usize());

                // The two sides traverse the edge in reverse winding;
                // every endpoint pairing patches both shards symmetrically.
                let full = reconcile_edge_endpoints(b.thirds, a.thirds, |bk, ak| {
                    if a.indices[ak] != INVALID_INDEX {
                        patch_slot(
                            &mut b_shard.output.cell_indices[b.slots[bk] as usize],
                            a.source_bin,
                            a.indices[ak],
                        );
                    }
                    if b.indices[bk] != INVALID_INDEX {
                        patch_slot(
                            &mut a_shard.output.cell_indices[a.slots[ak] as usize],
                            b.source_bin,
                            b.indices[bk],
                        );
                    }
                });
                if !full {
                    unresolved_edges.push(UnresolvedEdgeMismatch { key: a.key });
                }
            }
            i += 2;
        } else {
            unresolved_edges.push(UnresolvedEdgeMismatch {
                key: edge_check_overflow[i].key,
            });
            i += 1;
        }
    }

    let edge_checks_overflow_match_time = t_edge_match.elapsed();
    OverflowResolveTiming {
        sort: edge_checks_overflow_sort_time,
        match_: edge_checks_overflow_match_time,
    }
}
