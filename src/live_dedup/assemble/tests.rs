use super::*;
use crate::knn_clipping::edge_reconcile::{
    edge_segments_for_neighbor, reconcile_edge_mismatches, ReconcileApply, ReconcileOptions,
    VertexKeys,
};
use crate::live_dedup::binning::BinAssignment;
use crate::live_dedup::packed::{pack_edge, INVALID_INDEX};
use crate::live_dedup::shard::ShardState;
use crate::live_dedup::types::{EdgeCheckOverflow, EdgeMismatchOrigin, LocalId};
use crate::live_dedup::{EdgeRecord, ShardedCellsData};
use glam::Vec3;
use std::collections::BTreeSet;

fn bin(value: usize) -> BinId {
    BinId::from_usize(value)
}

#[test]
fn scatter_order_gate_distinguishes_correlated_and_scrambled_ids() {
    let correlated = vec![(0..100).collect(), (100..200).collect()];
    assert!(!prefer_shard_order_scatter(&correlated, 200));

    let scrambled = vec![
        (0..100)
            .map(|i| if i % 2 == 0 { i / 2 } else { 199 - i / 2 })
            .collect(),
        Vec::new(),
    ];
    assert!(prefer_shard_order_scatter(&scrambled, 200));

    let at_threshold = vec![(0..33).map(|i| i * 100).collect()];
    assert!(!prefer_shard_order_scatter(&at_threshold, 10_000));
    let above_threshold = vec![(0..33).map(|i| i * 101).collect()];
    assert!(prefer_shard_order_scatter(&above_threshold, 10_000));
}

#[test]
fn incidence_summary_collects_sparse_low_ids_only_when_requested() {
    let mut shard0 = ShardState::new(0);
    shard0.output.vertices = vec![Vec3::ZERO; 4];
    shard0.output.vertex_incidence = vec![0, 1, 2, 3];
    let mut shard1 = ShardState::new(0);
    shard1.output.vertices = vec![Vec3::ZERO; 2];
    shard1.output.vertex_incidence = vec![2, 3];
    let finals = vec![shard0.into_final(), shard1.into_final()];

    let clean = summarize_incidence(&finals, 11, false);
    assert_eq!(clean.used_vertices, 5);
    assert!(clean.low_incidence);
    assert!(clean.low_incidence_vertices.is_empty());

    let defect = summarize_incidence(&finals, 11, true);
    assert_eq!(defect.used_vertices, 5);
    assert!(defect.low_incidence);
    assert_eq!(defect.low_incidence_vertices, [1, 2, 4]);
}

#[test]
fn exact_zero_hint_confirmation_preserves_count_and_deduplicates_pairs() {
    let mut shard = ShardState::new(2);
    shard.output.exact_zero_edge_hint_cells = vec![0, 1];
    let finals = vec![shard.into_final()];
    let vertices = [Vec3::X, Vec3::X, Vec3::Y, Vec3::Z];
    let cells = [VoronoiCell::new(0, 3), VoronoiCell::new(3, 3)];
    let cell_indices = [0, 1, 2, 1, 0, 3];

    let confirmed = confirm_exact_zero_edge_hints(&finals, &vertices, &cells, &cell_indices);

    assert_eq!(confirmed.hinted_cells, vec![0, 1]);
    assert_eq!(confirmed.candidates, [(0, 1)]);
}

#[test]
fn shard_bookkeeping_collection_reserves_drains_and_preserves_order() {
    let mut shards = vec![ShardState::new(0), ShardState::new(0)];
    for (ordinal, shard) in shards.iter_mut().enumerate() {
        let key = pack_edge(ordinal as u32, ordinal as u32 + 10);
        shard.output.edge_mismatches.push(EdgeMismatch {
            key,
            origin: EdgeMismatchOrigin::InBinMissingCheck,
        });
        shard.output.edge_check_overflow.push(EdgeCheckOverflow {
            key,
            side: ordinal as u8,
            source_bin: bin(ordinal),
            target_bin: bin(1 - ordinal),
            thirds: [1, 2],
            indices: [3, 4],
            slots: [5, 6],
            source_cell: ordinal as u32,
            source_offsets: [0, 1],
        });
        shard.output.deferred_slots.push(DeferredSlot {
            key: [ordinal as u32, 20, 30],
            pos: Vec3::new(ordinal as f32, 0.0, 1.0),
            source_bin: bin(ordinal),
            source_slot: ordinal as u32,
            source_cell: ordinal as u32,
            source_offset: 0,
        });
    }

    let collected = collect_shard_bookkeeping(&mut shards);

    assert_eq!(collected.edge_mismatches.len(), 2);
    assert_eq!(collected.edge_check_overflow.len(), 2);
    assert_eq!(collected.deferred_slots.len(), 2);
    assert!(collected.edge_mismatches.capacity() >= 2);
    assert!(collected.edge_check_overflow.capacity() >= 2);
    assert!(collected.deferred_slots.capacity() >= 2);
    assert_eq!(collected.edge_mismatches[0].key, pack_edge(0, 10));
    assert_eq!(collected.edge_mismatches[1].key, pack_edge(1, 11));
    assert_eq!(collected.edge_check_overflow[0].source_bin, bin(0));
    assert_eq!(collected.edge_check_overflow[1].source_bin, bin(1));
    assert_eq!(collected.deferred_slots[0].key[0], 0);
    assert_eq!(collected.deferred_slots[1].key[0], 1);
    for shard in &shards {
        assert!(shard.output.edge_mismatches.is_empty());
        assert!(shard.output.edge_check_overflow.is_empty());
        assert!(shard.output.deferred_slots.is_empty());
    }

    let empty_unresolved = collect_shard_bookkeeping(&mut shards);
    assert_eq!(empty_unresolved.edge_mismatches.capacity(), 0);
}

#[test]
fn deferred_fallback_allocates_once_per_owner_key() {
    let mut shards = vec![ShardState::new(1), ShardState::new(1)];
    shards[0].output.cell_indices = vec![INVALID_INDEX, INVALID_INDEX];
    let generator_bin = vec![bin(1), bin(0), bin(0)];
    let key = [0, 1, 2];
    let pos = Vec3::new(0.0, 0.0, 1.0);

    let drift_exceeded = patch_deferred_slots_with_fallback(
        &mut shards,
        &generator_bin,
        vec![
            DeferredSlot {
                key,
                pos,
                source_bin: bin(0),
                source_slot: 0,
                source_cell: 0,
                source_offset: 0,
            },
            DeferredSlot {
                key,
                pos,
                source_bin: bin(0),
                source_slot: 1,
                source_cell: 0,
                source_offset: 1,
            },
        ],
    )
    .expect("fallback patching should succeed without capacity overflow");

    assert!(!drift_exceeded);
    assert_eq!(shards[1].output.vertices.len(), 1);
    assert_eq!(shards[1].output.vertex_keys, vec![key]);
    assert_eq!(shards[1].output.vertex_incidence, vec![2]);
    assert_eq!(
        shards[0].output.logical_reference(bin(0), 0),
        Some((bin(1), 0))
    );
    assert_eq!(
        shards[0].output.logical_reference(bin(0), 1),
        Some((bin(1), 0))
    );
}

#[test]
fn deferred_patch_reports_representative_drift_beyond_guard() {
    let mut shards = vec![ShardState::new(1), ShardState::new(1)];
    shards[0].output.cell_indices = vec![INVALID_INDEX];
    shards[0].output.patch_reference(bin(0), 0, 0, 0, bin(1), 0);
    shards[1].output.vertices = vec![Vec3::ZERO];
    shards[1].output.vertex_keys = vec![[0, 1, 2]];
    shards[1].output.vertex_incidence = vec![0];
    let eps = crate::tolerances::OUTPUT_RESOLUTION_REPRESENTATIVE_X_EPS;

    let drift_exceeded = patch_deferred_slots_with_fallback(
        &mut shards,
        &[bin(1), bin(0), bin(0)],
        vec![DeferredSlot {
            key: [0, 1, 2],
            pos: Vec3::new(f32::from_bits(eps.to_bits() + 1), 0.0, 0.0),
            source_bin: bin(0),
            source_slot: 0,
            source_cell: 0,
            source_offset: 0,
        }],
    )
    .expect("prepatched deferred slot should be checked");

    assert!(drift_exceeded);
}

#[test]
fn overflow_matching_patches_cross_bin_slots_before_fallback() {
    let mut shards = vec![ShardState::new(1), ShardState::new(1)];
    shards[0].output.cell_indices = vec![INVALID_INDEX, INVALID_INDEX];
    shards[1].output.cell_indices = vec![INVALID_INDEX, INVALID_INDEX];

    let edge_key = pack_edge(0, 1);
    let mut unresolved = Vec::new();
    let overflow = vec![
        EdgeCheckOverflow {
            key: edge_key,
            side: 0,
            source_bin: bin(0),
            target_bin: bin(1),
            thirds: [2, 3],
            indices: [10, 11],
            slots: [0, 1],
            source_cell: 0,
            source_offsets: [0, 1],
        },
        EdgeCheckOverflow {
            key: edge_key,
            side: 1,
            source_bin: bin(1),
            target_bin: bin(0),
            thirds: [3, 2],
            indices: [20, 21],
            slots: [0, 1],
            source_cell: 1,
            source_offsets: [0, 1],
        },
    ];

    resolve_edge_check_overflow(&mut shards, &overflow, &mut unresolved);

    assert!(
        unresolved.is_empty(),
        "full reverse-winding match should not remain unresolved"
    );
    assert_eq!(
        shards[0].output.logical_reference(bin(0), 0),
        Some((bin(1), 21))
    );
    assert_eq!(
        shards[0].output.logical_reference(bin(0), 1),
        Some((bin(1), 20))
    );
    assert_eq!(
        shards[1].output.logical_reference(bin(1), 0),
        Some((bin(0), 11))
    );
    assert_eq!(
        shards[1].output.logical_reference(bin(1), 1),
        Some((bin(0), 10))
    );
}

#[test]
fn overflow_mismatch_is_reported_unresolved() {
    let mut shards = vec![ShardState::new(1), ShardState::new(1)];
    let edge_key = pack_edge(0, 1);
    let mut unresolved = Vec::new();
    let overflow = vec![
        EdgeCheckOverflow {
            key: edge_key,
            side: 0,
            source_bin: bin(0),
            target_bin: bin(1),
            thirds: [2, 3],
            indices: [10, 11],
            slots: [0, 1],
            source_cell: 0,
            source_offsets: [0, 1],
        },
        EdgeCheckOverflow {
            key: edge_key,
            side: 1,
            source_bin: bin(1),
            target_bin: bin(0),
            thirds: [9, 8],
            indices: [20, 21],
            slots: [0, 1],
            source_cell: 1,
            source_offsets: [0, 1],
        },
    ];

    resolve_edge_check_overflow(&mut shards, &overflow, &mut unresolved);

    assert_eq!(unresolved.len(), 1);
    assert_eq!(unresolved[0].key, edge_key);
}

#[test]
fn overflow_duplicate_runs_do_not_patch_an_arbitrary_pair() {
    for sides in [[0u8, 0, 1], [0u8, 1, 1]] {
        let mut shards = vec![ShardState::new(1), ShardState::new(1)];
        shards[0].output.cell_indices = vec![INVALID_INDEX; 4];
        shards[1].output.cell_indices = vec![INVALID_INDEX; 4];
        let edge_key = pack_edge(0, 1);
        let mut side_counts = [0usize; 2];
        let overflow: Vec<EdgeCheckOverflow> = sides
            .into_iter()
            .map(|side| {
                let ordinal = side_counts[side as usize];
                side_counts[side as usize] += 1;
                EdgeCheckOverflow {
                    key: edge_key,
                    side,
                    source_bin: bin(side as usize),
                    target_bin: bin(1 - side as usize),
                    thirds: if side == 0 { [2, 3] } else { [3, 2] },
                    indices: [10 + ordinal as u32 * 2, 11 + ordinal as u32 * 2],
                    slots: [ordinal as u32 * 2, ordinal as u32 * 2 + 1],
                    source_cell: side as u32,
                    source_offsets: [0, 1],
                }
            })
            .collect();
        let mut unresolved = Vec::new();

        resolve_edge_check_overflow(&mut shards, &overflow, &mut unresolved);

        assert_eq!(unresolved.len(), 1, "sides={sides:?}");
        assert_eq!(
            unresolved[0].origin,
            EdgeMismatchOrigin::CrossBinDuplicateSide,
            "sides={sides:?}"
        );
        assert!(
            shards
                .iter()
                .all(|shard| shard.output.reference_overrides.is_empty()),
            "ambiguous run must be left to vertex-key fallback; sides={sides:?}"
        );
    }
}

#[test]
fn overflow_duplicate_run_without_opposite_side_reports_both_defects() {
    let mut shards = vec![ShardState::new(1), ShardState::new(1)];
    shards[0].output.cell_indices = vec![INVALID_INDEX; 6];
    let edge_key = pack_edge(0, 1);
    let overflow: Vec<EdgeCheckOverflow> = (0..3)
        .map(|ordinal| EdgeCheckOverflow {
            key: edge_key,
            side: 0,
            source_bin: bin(0),
            target_bin: bin(1),
            thirds: [2, 3],
            indices: [10 + ordinal * 2, 11 + ordinal * 2],
            slots: [ordinal * 2, ordinal * 2 + 1],
            source_cell: 0,
            source_offsets: [0, 1],
        })
        .collect();
    let mut unresolved = Vec::new();

    resolve_edge_check_overflow(&mut shards, &overflow, &mut unresolved);

    let origins: BTreeSet<_> = unresolved.iter().map(|entry| entry.origin).collect();
    assert_eq!(
        origins,
        BTreeSet::from([
            EdgeMismatchOrigin::CrossBinDuplicateSide,
            EdgeMismatchOrigin::CrossBinSingleSided,
        ])
    );
}

#[test]
fn assembly_then_reconcile_handles_overflow_fallback_and_unresolved_edge() {
    let mut shard0 = ShardState::new(3);
    let mut shard1 = ShardState::new(3);
    shard0.output.resolution_drift_exceeded = true;

    shard0.output.vertices = vec![
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    shard0.output.vertex_keys = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3]];
    shard0.output.vertex_incidence = vec![0; 3];
    shard0.output.cell_indices = vec![0, 1, 2];
    shard0.output.set_cell_start(LocalId::from_usize(0), 0);
    shard0.output.set_cell_count(LocalId::from_usize(0), 3);

    shard1.output.vertices = vec![
        Vec3::new(1.0 + 2.0e-7, 4.0e-8, 0.0),
        Vec3::new(4.0e-8, 1.0 + 2.0e-7, 0.0),
    ];
    shard1.output.vertex_keys = vec![[0, 1, 4], [0, 1, 5]];
    shard1.output.vertex_incidence = vec![0; 2];
    shard1.output.cell_indices = vec![0, 1, INVALID_INDEX];
    shard1.output.set_cell_start(LocalId::from_usize(0), 0);
    shard1.output.set_cell_count(LocalId::from_usize(0), 3);
    shard1.output.deferred_slots.push(DeferredSlot {
        key: [0, 4, 5],
        pos: Vec3::new(-1.0, 0.0, 0.0),
        source_bin: bin(1),
        source_slot: 2,
        source_cell: 1,
        source_offset: 2,
    });
    let edge_key = pack_edge(0, 1);
    shard0.output.edge_check_overflow.push(EdgeCheckOverflow {
        key: edge_key,
        side: 0,
        source_bin: bin(0),
        target_bin: bin(1),
        thirds: [2, 3],
        indices: [0, 1],
        slots: [0, 1],
        source_cell: 0,
        source_offsets: [0, 1],
    });
    shard1.output.edge_check_overflow.push(EdgeCheckOverflow {
        key: edge_key,
        side: 1,
        source_bin: bin(1),
        target_bin: bin(0),
        thirds: [9, 8],
        indices: [0, 1],
        slots: [0, 1],
        source_cell: 1,
        source_offsets: [0, 1],
    });

    let assignment = BinAssignment {
        generator_bin: vec![bin(0), bin(1), bin(0), bin(0), bin(1), bin(1)],
        generator_layout: vec![0, 1u32 << 31, 1, 2, (1u32 << 31) | 1, (1u32 << 31) | 2],
        slot_gen_map: Vec::new(),
        local_shift: 31,
        local_mask: (1u32 << 31) - 1,
        bin_generators: vec![vec![0, 2, 3], vec![1, 4, 5]],
        bin_cells: Vec::new(),
        bin_cell_offsets: vec![0, 0, 0],
        num_bins: 2,
    };
    let sharded = ShardedCellsData {
        assignment,
        shards: vec![shard0, shard1],
        cell_sub: crate::timing::CellSubAccum::new(),
    };

    let assembled = assemble_sharded_live_dedup(sharded).expect("assembly should succeed");
    assert!(assembled.resolution_drift_exceeded);
    assert_eq!(assembled.edge_mismatches.len(), 1);
    assert_eq!(assembled.edge_mismatches[0].key, edge_key);
    assert_eq!(assembled.cells.len(), 6);
    assert_eq!(assembled.cells[0].vertex_count(), 3);
    assert_eq!(assembled.cells[1].vertex_count(), 3);

    let cell1_start = assembled.cells[1].vertex_start();
    let cell1_indices = &assembled.cell_indices[cell1_start..cell1_start + 3];
    let fallback_global = cell1_indices[2] as usize;
    assert_eq!(
        assembled.vertex_keys.get(fallback_global as u32),
        Some([0, 4, 5]),
        "deferred slot should be patched through fallback ownership before reconciliation"
    );

    let reconcile_input: Vec<EdgeRecord> = assembled
        .edge_mismatches
        .iter()
        .map(|edge| EdgeRecord { key: edge.key })
        .collect();
    let mut cells = assembled.cells.clone();
    let mut cell_indices = assembled.cell_indices.clone();
    let spans_before: Vec<Vec<u32>> = cells
        .iter()
        .map(|c| cell_indices[c.vertex_start()..c.vertex_start() + c.vertex_count()].to_vec())
        .collect();
    let _residual = reconcile_edge_mismatches(
        &reconcile_input,
        &assembled.vertices,
        &mut cells,
        &mut cell_indices,
        VertexKeys::Sharded(&assembled.vertex_keys),
        crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
        ReconcileOptions::with_apply(ReconcileApply::InPlace),
    )
    .expect("reconciliation should succeed without capacity overflow");
    let spans_after: Vec<Vec<u32>> = cells
        .iter()
        .map(|c| cell_indices[c.vertex_start()..c.vertex_start() + c.vertex_count()].to_vec())
        .collect();
    assert_ne!(
        spans_before, spans_after,
        "expected unresolved shared-edge mismatch to be reconciled"
    );

    let seg_a = edge_segments_for_neighbor(
        0,
        1,
        crate::cell_layout::LiveCellLayout::new(&cells, &cell_indices),
        VertexKeys::Sharded(&assembled.vertex_keys),
    )
    .expect("edge segments should resolve after reconciliation");
    let seg_b = edge_segments_for_neighbor(
        1,
        0,
        crate::cell_layout::LiveCellLayout::new(&cells, &cell_indices),
        VertexKeys::Sharded(&assembled.vertex_keys),
    )
    .expect("edge segments should resolve after reconciliation");
    assert_eq!(seg_a.len(), 1);
    assert_eq!(seg_b.len(), 1);
    let set_a = BTreeSet::from([seg_a[0].0, seg_a[0].1]);
    let set_b = BTreeSet::from([seg_b[0].0, seg_b[0].1]);
    assert_eq!(
        set_a, set_b,
        "post-assembly reconciliation should make both cells share the same edge endpoints"
    );
}
