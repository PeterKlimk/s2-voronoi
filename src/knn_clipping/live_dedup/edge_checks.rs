//! Edge-check bookkeeping helpers for live dedup.

use glam::Vec3;
use std::mem;

use super::binning::BinAssignment;
use super::packed::{pack_edge, pack_ref, DEFERRED, INVALID_INDEX};
use super::shard::{ShardDedup, ShardState};
use super::types::{
    BadEdgeRecord, BinId, EdgeCheck, EdgeCheckOverflow, EdgeKey, EdgeOverflowLocal, EdgeToLater,
    LocalId,
};
use super::with_two_mut;
use crate::knn_clipping::cell_builder::VertexKey;
use crate::knn_clipping::timing::Timer;
use std::time::Duration;

#[inline]
pub(super) fn unpack_edge_key(key: EdgeKey) -> (u32, u32) {
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

    pub(super) fn take_edge_checks(&mut self, local: LocalId) -> Vec<EdgeCheck> {
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
/// This eliminates the edges_to_earlier intermediate vec.
pub(super) fn collect_and_resolve_cell_edges(
    cell_idx: u32,
    bin: BinId,
    local: LocalId,
    cell_vertices: &[(VertexKey, Vec3)],
    edge_neighbor_slots: &[u32],
    edge_neighbor_globals: &[u32],
    assignment: &BinAssignment,
    shard: &mut ShardState,
    incoming_checks: Vec<EdgeCheck>,
    vertex_indices: &mut [u32],
    edges_to_later: &mut Vec<EdgeToLater>,
    edges_overflow: &mut Vec<EdgeOverflowLocal>,
) {
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

    // Hoist constants to registers (avoid struct/stack indirection)
    let local_shift = assignment.local_shift;
    let local_mask = assignment.local_mask;

    // Load first key to start the rotation
    let mut curr_key = cell_vertices[0].0;

    // Process all edges
    for i in 0..n {
        let j = if i + 1 == n { 0 } else { i + 1 };
        let next_key = cell_vertices[j].0;
        let key_i = curr_key;
        let key_j = next_key;

        let slot = edge_neighbor_slots[i];

        if slot == u32::MAX {
            curr_key = next_key;
            continue;
        }

        // Derive global index from propagated array (sequential access)
        let neighbor = edge_neighbor_globals[i];
        if neighbor == cell_idx {
            curr_key = next_key;
            continue;
        }

        let locals = [i as u8, j as u8];
        let edge_key = pack_edge(cell_idx, neighbor);

        // Manual unpack with hoisted constants
        let packed = assignment.slot_gen_map[slot as usize];
        let bin_b = BinId::from((packed >> local_shift) as u8);
        let local_b = LocalId::from(packed & local_mask);

        // Prepare for next iteration
        curr_key = next_key;

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
            // Edge to earlier neighbor → resolve immediately
            // Search Vec for matching check (cache-friendly sequential access)
            let mut found_check: Option<EdgeCheck> = None;
            let mut found_idx = 0usize;
            for (idx, check) in incoming_checks.iter().enumerate() {
                if check.key == edge_key {
                    found_check = Some(*check);
                    found_idx = idx;
                    break;
                }
            }

            if let Some(check) = found_check {
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

                // Incoming check is form [OtherStart, OtherEnd]
                // We are traversing in reverse.
                // Match: MyStart (thirds[0]) == OtherEnd (check.thirds[1])
                //        MyEnd   (thirds[1]) == OtherStart (check.thirds[0])
                let endpoints_match =
                    check.thirds[0] == my_thirds[1] && check.thirds[1] == my_thirds[0];

                if endpoints_match {
                    // check.indices[0] is OtherStart -> matches MyEnd (locals[1])
                    // check.indices[1] is OtherEnd   -> matches MyStart (locals[0])

                    // 1. MyStart receives OtherEnd
                    let local_idx_0 = locals[0] as usize;
                    debug_assert!(
                        vertex_indices[local_idx_0] == INVALID_INDEX
                            || vertex_indices[local_idx_0] == check.indices[1],
                        "edge check index mismatch (start)"
                    );
                    vertex_indices[local_idx_0] = check.indices[1];

                    // 2. MyEnd receives OtherStart
                    let local_idx_1 = locals[1] as usize;
                    debug_assert!(
                        vertex_indices[local_idx_1] == INVALID_INDEX
                            || vertex_indices[local_idx_1] == check.indices[0],
                        "edge check index mismatch (end)"
                    );
                    vertex_indices[local_idx_1] = check.indices[0];
                } else {
                    // Partial match: search for any shared endpoint (may be in different slots)
                    for ck in 0..2 {
                        let check_third = check.thirds[ck];
                        for mk in 0..2 {
                            if check_third == my_thirds[mk] {
                                let local_idx = locals[mk] as usize;
                                debug_assert!(
                                    vertex_indices[local_idx] == INVALID_INDEX
                                        || vertex_indices[local_idx] == check.indices[ck],
                                    "edge check index mismatch (cross-slot)"
                                );
                                vertex_indices[local_idx] = check.indices[ck];
                            }
                        }
                    }
                    // Endpoint mismatch
                    shard.output.bad_edges.push(BadEdgeRecord { key: edge_key });
                }
            } else {
                // Missing side - earlier neighbor didn't emit check
                shard.output.bad_edges.push(BadEdgeRecord { key: edge_key });
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
                    .bad_edges
                    .push(BadEdgeRecord { key: check.key });
            }
        }
    }

    if incoming_count > 0 {
        shard.dedup.recycle_edge_checks(incoming_checks);
    }
}

/// Matches edge checks across shard boundaries.
///
/// Since cross-bin edges are emitted twice (once by each bin), we sort all overflow
/// records and match them by edge key. This allows propagating vertex indices
/// between bins without global communication during the main clipping phase.
pub(super) fn resolve_edge_check_overflow(
    shards: &mut [ShardState],
    edge_check_overflow: &mut [EdgeCheckOverflow],
    bad_edges: &mut Vec<BadEdgeRecord>,
) -> (Duration, Duration) {
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
                bad_edges.push(BadEdgeRecord { key: a.key });
            } else {
                let (a_shard, b_shard) =
                    with_two_mut(shards, a.source_bin.as_usize(), b.source_bin.as_usize());

                // Match based on reverse winding:
                // a (Side 0, Creator) traversing Start->End
                // b (Side 1, User)    traversing End->Start (relative to a)
                //
                // So a.Start == b.End and a.End == b.Start
                if a.thirds[0] == b.thirds[1] && a.thirds[1] == b.thirds[0] {
                    // Full match:
                    // a.indices[0] (a.Start) <-> b.indices[1] (b.End)
                    // a.indices[1] (a.End)   <-> b.indices[0] (b.Start)

                    // Patch b's slots (using a's indices)
                    // b.slots[0] (b.Start) gets a.indices[1] (a.End)
                    if a.indices[1] != INVALID_INDEX {
                        patch_slot(
                            &mut b_shard.output.cell_indices[b.slots[0] as usize],
                            a.source_bin,
                            a.indices[1],
                        );
                    }
                    // b.slots[1] (b.End) gets a.indices[0] (a.Start)
                    if a.indices[0] != INVALID_INDEX {
                        patch_slot(
                            &mut b_shard.output.cell_indices[b.slots[1] as usize],
                            a.source_bin,
                            a.indices[0],
                        );
                    }

                    // Patch a's slots (using b's indices)
                    // a.slots[0] (a.Start) gets b.indices[1] (b.End)
                    if b.indices[1] != INVALID_INDEX {
                        patch_slot(
                            &mut a_shard.output.cell_indices[a.slots[0] as usize],
                            b.source_bin,
                            b.indices[1],
                        );
                    }
                    // a.slots[1] (a.End) gets b.indices[0] (b.Start)
                    if b.indices[0] != INVALID_INDEX {
                        patch_slot(
                            &mut a_shard.output.cell_indices[a.slots[1] as usize],
                            b.source_bin,
                            b.indices[0],
                        );
                    }
                } else {
                    // Partial match: cross-slot search for any shared endpoint
                    for ak in 0..2 {
                        for bk in 0..2 {
                            if a.thirds[ak] == b.thirds[bk] {
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
                            }
                        }
                    }
                    // Endpoint mismatch
                    bad_edges.push(BadEdgeRecord { key: a.key });
                }
            }
            i += 2;
        } else {
            bad_edges.push(BadEdgeRecord {
                key: edge_check_overflow[i].key,
            });
            i += 1;
        }
    }

    let edge_checks_overflow_match_time = t_edge_match.elapsed();
    (
        edge_checks_overflow_sort_time,
        edge_checks_overflow_match_time,
    )
}
