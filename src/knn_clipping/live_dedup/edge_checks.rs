//! Edge-check bookkeeping helpers for live dedup.

use glam::Vec3;

use super::binning::BinAssignment;
use super::packed::{pack_edge, pack_ref, DEFERRED, INVALID_INDEX};
use super::shard::{ShardDedup, ShardState};
use super::types::{
    BadEdgeRecord, BinId, EdgeCheck, EdgeCheckNode, EdgeCheckOverflow, EdgeKey, EdgeOverflowLocal,
    EdgeToLater, LocalId,
};
use super::{with_two_mut, EDGE_CHECK_NONE};
use crate::knn_clipping::cell_builder::VertexKey;
use crate::knn_clipping::timing::Timer;
use std::time::Duration;

const INVALID_THIRD: u32 = u32::MAX;

#[inline]
fn unpack_edge_key(key: EdgeKey) -> (u32, u32) {
    let v: u64 = key.into();
    (v as u32, (v >> 32) as u32)
}

#[inline]
fn third_for_edge_endpoint(key: VertexKey, a: u32, b: u32) -> u32 {
    // key contains {a, b, third} in sorted order
    // XOR is self-canceling: x ^ x = 0
    // So: key[0] ^ key[1] ^ key[2] ^ a ^ b = third
    key[0] ^ key[1] ^ key[2] ^ a ^ b
}

impl ShardDedup {
    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn push_edge_check(&mut self, local: LocalId, check: EdgeCheck) {
        let local = local.as_usize();
        debug_assert!(
            local < self.edge_check_heads.len(),
            "edge check local out of bounds"
        );

        let idx = if self.edge_check_free != EDGE_CHECK_NONE {
            let idx = self.edge_check_free;
            let next_free = self.edge_check_nodes[idx as usize].next;
            self.edge_check_free = next_free;
            idx
        } else {
            let idx = u32::try_from(self.edge_check_nodes.len()).expect("edge check pool overflow");
            self.edge_check_nodes.push(EdgeCheckNode {
                check,
                next: EDGE_CHECK_NONE,
            });
            idx
        };

        let head = self.edge_check_heads[local];
        self.edge_check_nodes[idx as usize].check = check;
        self.edge_check_nodes[idx as usize].next = head;
        self.edge_check_heads[local] = idx;
        self.edge_check_counts[local] = self.edge_check_counts[local].saturating_add(1);
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn take_edge_checks(&mut self, local: LocalId) -> (u32, u8) {
        let local = local.as_usize();
        debug_assert!(
            local < self.edge_check_heads.len(),
            "edge check local out of bounds"
        );
        let head = self.edge_check_heads[local];
        let count = self.edge_check_counts[local];
        self.edge_check_heads[local] = EDGE_CHECK_NONE;
        self.edge_check_counts[local] = 0;
        (head, count)
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn recycle_edge_checks(&mut self, mut head: u32) {
        while head != EDGE_CHECK_NONE {
            let node = &mut self.edge_check_nodes[head as usize];
            let next = node.next;
            node.next = self.edge_check_free;
            self.edge_check_free = head;
            head = next;
        }
    }
}

/// Fused collect + resolve: iterates cell edges once, immediately resolving
/// edges to earlier neighbors against incoming checks while collecting
/// edges_to_later and edges_overflow for the emit phase.
///
/// This eliminates the edges_to_earlier intermediate vec.
#[cfg_attr(feature = "profiling", inline(never))]
pub(super) fn collect_and_resolve_cell_edges(
    cell_idx: u32,
    local: LocalId,
    cell_vertices: &[(VertexKey, Vec3)],
    edge_neighbors: &[u32],
    assignment: &BinAssignment,
    shard: &mut ShardState,
    vertex_indices: &mut [u32],
    edges_to_later: &mut Vec<EdgeToLater>,
    edges_overflow: &mut Vec<EdgeOverflowLocal>,
) {
    let n = cell_vertices.len();
    debug_assert_eq!(edge_neighbors.len(), n, "edge neighbor data out of sync");
    edges_to_later.clear();
    edges_overflow.clear();

    debug_assert!(n >= 2, "cell has fewer than 2 vertices");

    let (bin_a, local_a) = assignment.unpack(assignment.gen_map[cell_idx as usize]);
    debug_assert_eq!(local_a, local, "local index mismatch in edge checks");
    let local_u32 = local.as_u32();

    // Take incoming checks from linked list (returns head and count)
    let (assigned_head, incoming_count) = shard.dedup.take_edge_checks(local);

    // Track which incoming checks we've matched (bitmask, supports up to 64)
    let mut matched: u64 = 0;

    // Pre-gather neighbor gen_map entries to improve memory-level parallelism.
    // This separates the random memory accesses from the dependent logic loop.
    let mut neighbor_gen_maps = [0u32; 64];
    debug_assert!(n <= 64, "cell has more than 64 vertices");

    for i in 0..n {
        let neighbor = edge_neighbors[i];
        if neighbor != u32::MAX {
            neighbor_gen_maps[i] = assignment.gen_map[neighbor as usize];
        }
    }

    // Process all edges
    for i in 0..n {
        let j = if i + 1 == n { 0 } else { i + 1 };
        let key_i = cell_vertices[i].0;
        let key_j = cell_vertices[j].0;
        let neighbor = edge_neighbors[i];

        if neighbor == u32::MAX || neighbor == cell_idx {
            continue;
        }

        let locals = if key_i <= key_j {
            [i as u8, j as u8]
        } else {
            [j as u8, i as u8]
        };
        let edge_key = pack_edge(cell_idx, neighbor);

        let (bin_b, local_b) = assignment.unpack(neighbor_gen_maps[i]);

        if bin_a != bin_b {
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
            // Search linked list for matching check
            let mut found_check: Option<EdgeCheck> = None;
            let mut found_idx = 0u32;
            let mut search_cur = assigned_head;
            let mut idx = 0u32;
            while search_cur != EDGE_CHECK_NONE {
                let node = &shard.dedup.edge_check_nodes[search_cur as usize];
                if node.check.key == edge_key {
                    found_check = Some(node.check);
                    found_idx = idx;
                    break;
                }
                search_cur = node.next;
                idx += 1;
            }

            if let Some(check) = found_check {
                if matched & (1u64 << found_idx) != 0 {
                    // Duplicate
                    debug_assert!(false, "edge check duplicate side");
                    shard.output.bad_edges.push(BadEdgeRecord { key: edge_key });
                } else {
                    matched |= 1u64 << found_idx;

                    let (a, b) = unpack_edge_key(edge_key);
                    let my_thirds = [
                        third_for_edge_endpoint(cell_vertices[locals[0] as usize].0, a, b),
                        third_for_edge_endpoint(cell_vertices[locals[1] as usize].0, a, b),
                    ];

                    let mut endpoints_match = true;
                    for k in 0..2 {
                        if check.thirds[k] == INVALID_THIRD
                            || my_thirds[k] == INVALID_THIRD
                            || check.thirds[k] != my_thirds[k]
                        {
                            endpoints_match = false;
                            continue;
                        }

                        let idx = check.indices[k];
                        if idx != INVALID_INDEX {
                            let local_idx = locals[k] as usize;
                            let existing = vertex_indices[local_idx];
                            if existing == INVALID_INDEX {
                                vertex_indices[local_idx] = idx;
                            } else {
                                debug_assert_eq!(existing, idx, "edge check index mismatch");
                            }
                        }
                    }

                    if !endpoints_match {
                        shard.output.bad_edges.push(BadEdgeRecord { key: edge_key });
                    }
                }
            } else {
                // Missing side - earlier neighbor didn't emit check
                shard.output.bad_edges.push(BadEdgeRecord { key: edge_key });
            }
        }
    }

    // Fast path: if all incoming checks matched, skip the unmatched loop
    let all_matched = incoming_count == 0 || matched == (1u64 << incoming_count) - 1;

    if !all_matched {
        let mut cur = assigned_head;
        let mut idx = 0u32;
        while cur != EDGE_CHECK_NONE {
            if matched & (1u64 << idx) == 0 {
                let node = &shard.dedup.edge_check_nodes[cur as usize];
                shard.output.bad_edges.push(BadEdgeRecord {
                    key: node.check.key,
                });
            }
            cur = shard.dedup.edge_check_nodes[cur as usize].next;
            idx += 1;
        }
    }

    shard.dedup.recycle_edge_checks(assigned_head);
}

#[cfg_attr(feature = "profiling", inline(never))]
pub(super) fn resolve_edge_check_overflow(
    shards: &mut [ShardState],
    edge_check_overflow: &mut Vec<EdgeCheckOverflow>,
    bad_edges: &mut Vec<BadEdgeRecord>,
) -> (Duration, Duration) {
    let t_edge_sort = Timer::start();
    edge_check_overflow.sort_unstable_by_key(|entry| (entry.key, entry.side));
    let edge_checks_overflow_sort_time = t_edge_sort.elapsed();

    let t_edge_match = Timer::start();
    let patch_slot = |slot: &mut u64, owner_bin: BinId, idx: u32| {
        let packed = pack_ref(owner_bin, idx);
        if *slot == DEFERRED {
            *slot = packed;
        } else {
            debug_assert_eq!(*slot, packed, "edge check index mismatch");
        }
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
                for k in 0..2 {
                    if a.thirds[k] != INVALID_THIRD && a.thirds[k] == b.thirds[k] {
                        if a.indices[k] != INVALID_INDEX {
                            let slot = &mut b_shard.output.cell_indices[b.slots[k] as usize];
                            patch_slot(slot, a.source_bin, a.indices[k]);
                        }
                        if b.indices[k] != INVALID_INDEX {
                            let slot = &mut a_shard.output.cell_indices[a.slots[k] as usize];
                            patch_slot(slot, b.source_bin, b.indices[k]);
                        }
                        if a.indices[k] != INVALID_INDEX
                            && b.indices[k] != INVALID_INDEX
                            && (a.source_bin, a.indices[k]) != (b.source_bin, b.indices[k])
                        {
                            debug_assert!(false, "edge check overflow index mismatch");
                        }
                    }
                }
                if a.thirds != b.thirds {
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
