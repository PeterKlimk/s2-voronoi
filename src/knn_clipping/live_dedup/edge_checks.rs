//! Edge-check bookkeeping helpers for live dedup.

use glam::Vec3;

use super::binning::{BinAssignment, GenMap};
use super::packed::{pack_edge, pack_ref, DEFERRED, INVALID_INDEX};
use super::shard::{ShardDedup, ShardState};
use super::types::{
    BadEdgeReason, BadEdgeRecord, BinId, EdgeCheck, EdgeCheckNode, EdgeCheckOverflow, EdgeLocal,
    EdgeOverflowLocal, EdgeToLater, LocalId,
};
use super::{with_two_mut, EDGE_CHECK_NONE};
use crate::knn_clipping::cell_builder::VertexKey;
use crate::knn_clipping::timing::Timer;
use std::time::Duration;

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
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn take_edge_checks(&mut self, local: LocalId) -> u32 {
        let local = local.as_usize();
        debug_assert!(
            local < self.edge_check_heads.len(),
            "edge check local out of bounds"
        );
        let head = self.edge_check_heads[local];
        self.edge_check_heads[local] = EDGE_CHECK_NONE;
        head
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

#[cfg_attr(feature = "profiling", inline(never))]
pub(super) fn collect_cell_edges(
    cell_idx: u32,
    local: LocalId,
    cell_vertices: &[(VertexKey, Vec3)],
    edge_neighbors: &[u32],
    assignment: &BinAssignment,
    edges_to_earlier: &mut Vec<EdgeLocal>,
    edges_to_later: &mut Vec<EdgeToLater>,
    edges_overflow: &mut Vec<EdgeOverflowLocal>,
) {
    let n = cell_vertices.len();
    debug_assert_eq!(edge_neighbors.len(), n, "edge neighbor data out of sync");
    edges_to_earlier.clear();
    edges_to_later.clear();
    edges_overflow.clear();
    if n < 2 {
        return;
    }

    let bin_a = assignment.gen_map[cell_idx as usize].bin;
    debug_assert_eq!(
        assignment.gen_map[cell_idx as usize].local, local,
        "local index mismatch in edge checks"
    );
    let local_u32 = local.as_u32();

    for i in 0..n {
        let j = if i + 1 == n { 0 } else { i + 1 };
        let key_i = cell_vertices[i].0;
        let key_j = cell_vertices[j].0;
        let neighbor = edge_neighbors[i];
        if neighbor == u32::MAX {
            continue;
        }
        if neighbor == cell_idx {
            continue;
        }
        let locals = if key_i <= key_j {
            [i as u8, j as u8]
        } else {
            [j as u8, i as u8]
        };
        let edge = EdgeLocal {
            key: pack_edge(cell_idx, neighbor),
            locals,
        };
        let GenMap {
            bin: bin_b,
            local: local_b_u32,
        } = assignment.gen_map[neighbor as usize];
        if bin_a == bin_b {
            let local_b = local_b_u32.as_usize();
            debug_assert_ne!(
                local.as_usize(),
                local_b,
                "edge checks: neighbor mapped to same local index as cell"
            );
            if local_u32 < local_b_u32.as_u32() {
                edges_to_later.push(EdgeToLater {
                    edge,
                    local_b: local_b_u32,
                });
            } else {
                edges_to_earlier.push(edge);
            }
        } else {
            let side = if cell_idx <= neighbor { 0 } else { 1 };
            edges_overflow.push(EdgeOverflowLocal { edge, side });
        }
    }
}

#[cfg_attr(feature = "profiling", inline(never))]
pub(super) fn resolve_cell_edge_checks(
    shard: &mut ShardState,
    local: LocalId,
    edges_to_earlier: &mut Vec<EdgeLocal>,
    cell_vertices: &[(VertexKey, Vec3)],
    vertex_indices: &mut [u32],
    matched: &mut Vec<bool>,
) {
    let assigned_head = shard.dedup.take_edge_checks(local);
    if assigned_head == EDGE_CHECK_NONE && edges_to_earlier.is_empty() {
        return;
    }
    matched.clear();
    matched.resize(edges_to_earlier.len(), false);

    let mut cur = assigned_head;
    while cur != EDGE_CHECK_NONE {
        let node = shard.dedup.edge_check_nodes[cur as usize];
        let assigned_edge = node.check;
        let next = node.next;

        let mut found = None;
        for (idx, edge) in edges_to_earlier.iter().enumerate() {
            if edge.key == assigned_edge.key {
                found = Some(idx);
                break;
            }
        }
        if let Some(edge_idx) = found {
            if matched[edge_idx] {
                debug_assert!(false, "edge check duplicate side");
                shard.output.bad_edges.push(BadEdgeRecord {
                    key: assigned_edge.key,
                    reason: BadEdgeReason::DuplicateSide,
                });
            } else {
                matched[edge_idx] = true;
                let emitted = edges_to_earlier[edge_idx];
                let emitted_endpoints = [
                    cell_vertices[emitted.locals[0] as usize].0,
                    cell_vertices[emitted.locals[1] as usize].0,
                ];
                
                let mut endpoints_match = true;
                for k in 0..2 {
                    if assigned_edge.endpoints[k] != emitted_endpoints[k] {
                        endpoints_match = false;
                        continue;
                    }

                    let idx = assigned_edge.indices[k];
                    if idx == INVALID_INDEX {
                        continue;
                    }
                    let local_idx = emitted.locals[k] as usize;
                    let existing = vertex_indices[local_idx];
                    if existing == INVALID_INDEX {
                        vertex_indices[local_idx] = idx;
                    } else {
                        debug_assert_eq!(existing, idx, "edge check index mismatch");
                    }
                }
                if !endpoints_match {
                    shard.output.bad_edges.push(BadEdgeRecord {
                        key: assigned_edge.key,
                        reason: BadEdgeReason::EndpointMismatch,
                    });
                }
            }
        } else {
            shard.output.bad_edges.push(BadEdgeRecord {
                key: assigned_edge.key,
                reason: BadEdgeReason::MissingSide,
            });
        }

        cur = next;
    }
    shard.dedup.recycle_edge_checks(assigned_head);

    for (idx, edge) in edges_to_earlier.iter().enumerate() {
        if !matched[idx] {
            shard.output.bad_edges.push(BadEdgeRecord {
                key: edge.key,
                reason: BadEdgeReason::MissingSide,
            });
        }
    }
    edges_to_earlier.clear();
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
                bad_edges.push(BadEdgeRecord {
                    key: a.key,
                    reason: BadEdgeReason::DuplicateSide,
                });
            } else {
                let (a_shard, b_shard) =
                    with_two_mut(shards, a.source_bin.as_usize(), b.source_bin.as_usize());
                for k in 0..2 {
                    if a.endpoints[k] == b.endpoints[k] {
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
                if a.endpoints != b.endpoints {
                    bad_edges.push(BadEdgeRecord {
                        key: a.key,
                        reason: BadEdgeReason::EndpointMismatch,
                    });
                }
            }
            i += 2;
        } else {
            bad_edges.push(BadEdgeRecord {
                key: edge_check_overflow[i].key,
                reason: BadEdgeReason::MissingSide,
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
