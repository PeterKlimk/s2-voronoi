//! Live vertex deduplication during cell construction using sharded ownership.
//!
//! V1 design:
//! - Parallel cell building by spatial bin
//! - Single-threaded overflow flush (simplifies correctness)
//! - Per-cell duplicate index checks handled by validation (not in hot path)

use glam::Vec3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::cell_builder::{VertexData, VertexKey};
use super::timing::{DedupSubPhases, Timer};
use super::topo2d::Topo2DBuilder;
use super::TerminationConfig;

use crate::cube_grid::{
    cell_to_face_ij,
    packed_knn::{
        packed_knn_cell_stream, PackedKnnCellScratch, PackedKnnCellStatus, PackedKnnTimings,
    },
};
use crate::VoronoiCell;

const DEFERRED: u64 = u64::MAX;
const INVALID_INDEX: u32 = u32::MAX;

#[inline]
fn pack_ref(bin: u32, local: u32) -> u64 {
    ((bin as u64) << 32) | (local as u64)
}

#[inline]
fn unpack_ref(packed: u64) -> (u32, u32) {
    ((packed >> 32) as u32, (packed & 0xFFFF_FFFF) as u32)
}

#[inline]
fn pack_bc(b: u32, c: u32) -> u64 {
    (b as u64) | ((c as u64) << 32)
}

#[inline]
fn pack_edge(a: u32, b: u32) -> u64 {
    let (min, max) = if a <= b { (a, b) } else { (b, a) };
    pack_bc(min, max)
}

#[inline]
fn canonical_endpoints(a: VertexKey, b: VertexKey) -> [VertexKey; 2] {
    if a <= b {
        [a, b]
    } else {
        [b, a]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct EdgeRecord {
    pub(super) key: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum BadEdgeReason {
    MissingSide,
    EndpointMismatch,
    DuplicateSide,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BadEdgeRecord {
    pub(super) key: u64,
    pub(super) reason: BadEdgeReason,
}

#[derive(Clone, Copy)]
struct EdgeCheck {
    key: u64,
    endpoints: [VertexKey; 2],
    indices: [u32; 2],
}

const EDGE_CHECK_NONE: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct EdgeCheckNode {
    check: EdgeCheck,
    next: u32,
}

#[derive(Clone, Copy)]
struct EdgeCheckOverflow {
    key: u64,
    side: u8,
    source_bin: u32,
    endpoints: [VertexKey; 2],
    indices: [u32; 2],
    slots: [u32; 2],
}

#[derive(Clone, Copy)]
struct EdgeLocal {
    key: u64,
    endpoints: [VertexKey; 2],
    locals: [u8; 2],
}

#[derive(Clone, Copy)]
struct EdgeToLater {
    edge: EdgeLocal,
    local_b: u32,
}

#[derive(Clone, Copy)]
struct EdgeOverflowLocal {
    edge: EdgeLocal,
    side: u8,
}

#[derive(Clone, Copy)]
struct DeferredSlot {
    key: VertexKey,
    pos: Vec3,
    source_bin: u32,
    source_slot: u32,
}

struct SupportOverflow {
    source_bin: u32,
    target_bin: u32,
    source_slot: u32,
    support: Vec<u32>,
    pos: Vec3,
}

struct BinAssignment {
    generator_bin: Vec<u32>,
    global_to_local: Vec<u32>,
    bin_generators: Vec<Vec<usize>>,
    num_bins: usize,
}

struct PackedSeed<'a> {
    neighbors: &'a [u32],
    count: usize,
    security: f32,
    k: usize,
}

struct BinLayout {
    bin_res: usize,
    bin_stride: usize,
    num_bins: usize,
}

fn choose_bin_layout(grid_res: usize) -> BinLayout {
    #[cfg(feature = "parallel")]
    let threads = rayon::current_num_threads().max(1);
    #[cfg(not(feature = "parallel"))]
    let threads = 1;
    let target_bins = (threads * 2).clamp(6, 96);
    let target_per_face = (target_bins as f64 / 6.0).max(1.0);
    let mut bin_res = target_per_face.sqrt().ceil() as usize;
    bin_res = bin_res.clamp(1, grid_res.max(1));

    let mut bin_stride = (grid_res + bin_res - 1) / bin_res;
    bin_stride = bin_stride.max(1);
    bin_res = (grid_res + bin_stride - 1) / bin_stride;

    BinLayout {
        bin_res,
        bin_stride,
        num_bins: 6 * bin_res * bin_res,
    }
}

fn assign_bins(points: &[Vec3], grid: &crate::cube_grid::CubeMapGrid) -> BinAssignment {
    let n = points.len();
    let layout = choose_bin_layout(grid.res());
    let num_bins = layout.num_bins;

    let mut generator_bin: Vec<u32> = Vec::with_capacity(n);
    let mut counts: Vec<usize> = vec![0; num_bins];
    for i in 0..n {
        let cell = grid.point_index_to_cell(i);
        let (face, iu, iv) = cell_to_face_ij(cell, grid.res());
        let bu = (iu / layout.bin_stride).min(layout.bin_res - 1);
        let bv = (iv / layout.bin_stride).min(layout.bin_res - 1);
        let b = face * layout.bin_res * layout.bin_res + bv * layout.bin_res + bu;
        generator_bin.push(b as u32);
        counts[b] += 1;
    }

    let mut bin_generators: Vec<Vec<usize>> = (0..num_bins)
        .map(|b| Vec::with_capacity(counts[b]))
        .collect();
    for (i, &b) in generator_bin.iter().enumerate() {
        bin_generators[b as usize].push(i);
    }

    for generators in &mut bin_generators {
        generators.sort_unstable_by_key(|&g| (grid.point_index_to_cell(g), g));
    }

    let mut global_to_local: Vec<u32> = vec![0; n];
    for generators in &bin_generators {
        for (local_idx, &global_idx) in generators.iter().enumerate() {
            global_to_local[global_idx] = local_idx as u32;
        }
    }

    BinAssignment {
        generator_bin,
        global_to_local,
        bin_generators,
        num_bins,
    }
}

/// Data only needed during vertex deduplication (dropped after overflow flush).
struct ShardDedup {
    support_map: FxHashMap<Vec<u32>, u32>,
    support_data: Vec<u32>,
    support_overflow: Vec<SupportOverflow>,
    edge_check_heads: Vec<u32>,
    edge_check_nodes: Vec<EdgeCheckNode>,
    edge_check_free: u32,
}

impl ShardDedup {
    fn new(num_local_generators: usize) -> Self {
        Self {
            support_map: FxHashMap::default(),
            support_data: Vec::new(),
            support_overflow: Vec::new(),
            edge_check_heads: vec![EDGE_CHECK_NONE; num_local_generators],
            edge_check_nodes: Vec::new(),
            edge_check_free: EDGE_CHECK_NONE,
        }
    }

    #[inline(always)]
    fn push_edge_check(&mut self, local: usize, check: EdgeCheck) {
        debug_assert!(local < self.edge_check_heads.len(), "edge check local out of bounds");

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

    #[inline(always)]
    fn take_edge_checks(&mut self, local: usize) -> u32 {
        debug_assert!(local < self.edge_check_heads.len(), "edge check local out of bounds");
        let head = self.edge_check_heads[local];
        self.edge_check_heads[local] = EDGE_CHECK_NONE;
        head
    }

    #[inline(always)]
    fn recycle_edge_checks(&mut self, mut head: u32) {
        while head != EDGE_CHECK_NONE {
            let node = &mut self.edge_check_nodes[head as usize];
            let next = node.next;
            node.next = self.edge_check_free;
            self.edge_check_free = head;
            head = next;
        }
    }
}

/// Output data needed for final assembly.
struct ShardOutput {
    vertices: Vec<Vec3>,
    vertex_keys: Vec<VertexKey>,
    bad_edges: Vec<BadEdgeRecord>,
    edge_check_overflow: Vec<EdgeCheckOverflow>,
    deferred: Vec<DeferredSlot>,
    cell_indices: Vec<u64>,
    cell_starts: Vec<u32>,
    cell_counts: Vec<u8>,
}

impl ShardOutput {
    fn new(num_local_generators: usize) -> Self {
        Self {
            vertices: Vec::new(),
            vertex_keys: Vec::new(),
            bad_edges: Vec::new(),
            edge_check_overflow: Vec::new(),
            deferred: Vec::new(),
            cell_indices: Vec::new(),
            cell_starts: vec![0; num_local_generators],
            cell_counts: vec![0; num_local_generators],
        }
    }
}

/// Per-shard state during cell construction.
struct ShardState {
    dedup: ShardDedup,
    output: ShardOutput,
    #[cfg(feature = "timing")]
    triplet_keys: u64,
    #[cfg(feature = "timing")]
    support_keys: u64,
}

impl ShardState {
    fn new(num_local_generators: usize) -> Self {
        Self {
            dedup: ShardDedup::new(num_local_generators),
            output: ShardOutput::new(num_local_generators),
            #[cfg(feature = "timing")]
            triplet_keys: 0,
            #[cfg(feature = "timing")]
            support_keys: 0,
        }
    }

    #[inline(always)]
    fn dedup_support_owned(&mut self, support: Vec<u32>, pos: Vec3) -> u32 {
        if let Some(&idx) = self.dedup.support_map.get(support.as_slice()) {
            return idx;
        }
        let idx = self.output.vertices.len() as u32;
        self.output.vertices.push(pos);
        self.dedup.support_map.insert(support, idx);
        idx
    }
}

/// Shard state after construction, with dedup dropped.
struct ShardFinal {
    output: ShardOutput,
    #[cfg(feature = "timing")]
    triplet_keys: u64,
    #[cfg(feature = "timing")]
    support_keys: u64,
}

impl ShardState {
    fn into_final(self) -> ShardFinal {
        ShardFinal {
            output: self.output,
            #[cfg(feature = "timing")]
            triplet_keys: self.triplet_keys,
            #[cfg(feature = "timing")]
            support_keys: self.support_keys,
        }
        // self.dedup dropped here automatically
    }
}

pub(super) struct ShardedCellsData {
    assignment: BinAssignment,
    shards: Vec<ShardState>,
    pub(super) cell_sub: super::timing::CellSubAccum,
}

fn with_two_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i != j);
    if i < j {
        let (a, b) = v.split_at_mut(j);
        (&mut a[i], &mut b[0])
    } else {
        let (a, b) = v.split_at_mut(i);
        (&mut b[0], &mut a[j])
    }
}

fn collect_cell_edges(
    cell_idx: u32,
    local: usize,
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

    let bin_a = assignment.generator_bin[cell_idx as usize];
    debug_assert_eq!(
        assignment.global_to_local[cell_idx as usize] as usize, local,
        "local index mismatch in edge checks"
    );
    let local_u32 = u32::try_from(local).expect("local index must fit in u32");

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
        let (endpoints, locals) = if key_i <= key_j {
            ([key_i, key_j], [i as u8, j as u8])
        } else {
            ([key_j, key_i], [j as u8, i as u8])
        };
        let edge = EdgeLocal {
            key: pack_edge(cell_idx, neighbor),
            endpoints,
            locals,
        };
        let bin_b = assignment.generator_bin[neighbor as usize];
        if bin_a == bin_b {
            let local_b = assignment.global_to_local[neighbor as usize] as usize;
            debug_assert_ne!(
                local, local_b,
                "edge checks: neighbor mapped to same local index as cell"
            );
            let local_b_u32 =
                u32::try_from(local_b).expect("local index must fit in u32");
            if local_u32 < local_b_u32 {
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

fn resolve_cell_edge_checks(
    shard: &mut ShardState,
    local: usize,
    edges_to_earlier: &mut Vec<EdgeLocal>,
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
                for k in 0..2 {
                    if assigned_edge.endpoints[k] == emitted.endpoints[k] {
                        let idx = assigned_edge.indices[k];
                        if idx != INVALID_INDEX {
                            let local_idx = emitted.locals[k] as usize;
                            let existing = vertex_indices[local_idx];
                            if existing == INVALID_INDEX {
                                vertex_indices[local_idx] = idx;
                            } else if existing != idx {
                                debug_assert!(false, "edge check index mismatch");
                            }
                        }
                    }
                }
                if assigned_edge.endpoints != emitted.endpoints {
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

pub(super) fn build_cells_sharded_live_dedup(
    points: &[Vec3],
    knn: &super::CubeMapGridKnn,
    termination: TerminationConfig,
) -> ShardedCellsData {
    // If termination is enabled but not proven after the kNN schedule, we keep requesting
    // more neighbors until the termination check succeeds.
    //
    // This makes correctness independent of any fixed k cap. You can cap the extra work
    // via `S2V_TERMINATION_MAX_K` if you prefer (unset/0 = no cap).
    let termination_max_k_cap: Option<usize> = std::env::var("S2V_TERMINATION_MAX_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .and_then(|v| (v != 0).then_some(v));

    let assignment = assign_bins(points, knn.grid());
    let num_bins = assignment.num_bins;
    let packed_k = super::KNN_RESUME_K.min(points.len().saturating_sub(1));

    let per_bin: Vec<(ShardState, super::timing::CellSubAccum)> = maybe_par_into_iter!(0..num_bins)
        .map(|bin_usize| {
            use super::timing::{CellSubAccum, KnnCellStage, Timer};

            let bin = bin_usize as u32;
            let my_generators = &assignment.bin_generators[bin_usize];
            let mut shard = ShardState::new(my_generators.len());

            let mut scratch = knn.make_scratch();
            let mut builder = Topo2DBuilder::new(0, Vec3::ZERO);
            let mut sub_accum = CellSubAccum::new();
            let mut neighbors: Vec<usize> = Vec::with_capacity(super::KNN_RESTART_MAX);
            let mut cell_vertices: Vec<VertexData> = Vec::new();
            let mut edge_neighbors: Vec<u32> = Vec::new();
            let mut edges_to_earlier: Vec<EdgeLocal> = Vec::new();
            let mut edges_to_later: Vec<EdgeToLater> = Vec::new();
            let mut edges_overflow: Vec<EdgeOverflowLocal> = Vec::new();
            let mut edge_matched: Vec<bool> = Vec::new();
            let mut vertex_indices: Vec<u32> = Vec::new();
            shard
                .output
                .vertices
                .reserve(my_generators.len().saturating_mul(4));
            shard
                .output
                .cell_indices
                .reserve(my_generators.len().saturating_mul(6));
            shard
                .dedup
                .support_data
                .reserve(my_generators.len().saturating_mul(2));

            let grid = knn.grid();
            let mut packed_scratch = PackedKnnCellScratch::new();
            let mut packed_timings = PackedKnnTimings::default();

            let packed_queries_all: Vec<u32> = my_generators
                .iter()
                .map(|&i| u32::try_from(i).expect("point index must fit in u32"))
                .collect();

            #[cfg(debug_assertions)]
            {
                for &i in my_generators {
                    debug_assert_eq!(
                        assignment.generator_bin[i],
                        bin,
                        "cell assigned to wrong bin"
                    );
                }
            }

            let mut process_cell =
                |cell_sub: &mut super::timing::CellSubAccum,
                 i: usize,
                 local: u32,
                 packed: Option<PackedSeed>| {
                builder.reset(i, points[i]);
                neighbors.clear();

                let cell_start = shard.output.cell_indices.len() as u32;
                shard.output.cell_starts[local as usize] = cell_start;

                let mut cell_neighbors_processed = 0usize;
                let mut terminated = false;
                let mut knn_exhausted = false;
                let mut full_scan_done = false;
                let mut did_full_scan_fallback = false;
                let mut used_knn = false;

                let mut worst_cos = 1.0f32;
                let max_neighbors = points.len().saturating_sub(1);
                let mut max_k_requested = 0usize;
                let mut knn_stage = KnnCellStage::Resume(super::KNN_RESUME_K);
                let mut reached_schedule_max_k = false;

                let mut did_packed = false;
                let mut packed_count = 0usize;
                let mut packed_security = 0.0f32;
                let mut packed_k_local = 0usize;

                if let Some(seed) = packed {
                    did_packed = true;
                    packed_count = seed.count;
                    packed_security = seed.security;
                    packed_k_local = seed.k;

                    if packed_count > 0 {
                        let t_clip = Timer::start();
                        for (pos, &neighbor_idx) in seed.neighbors.iter().enumerate() {
                            let neighbor_idx = neighbor_idx as usize;
                            if neighbor_idx == i {
                                continue;
                            }
                            #[cfg(debug_assertions)]
                            debug_assert!(
                                !builder.has_neighbor(neighbor_idx),
                                "packed kNN returned duplicate neighbor {} for cell {}",
                                neighbor_idx, i
                            );
                            let neighbor = points[neighbor_idx];
                            if builder.clip(neighbor_idx, neighbor).is_err() {
                                break;
                            }
                            cell_neighbors_processed += 1;
                            let dot = points[i].dot(neighbor);
                            worst_cos = worst_cos.min(dot);

                            if builder.is_bounded() {
                                if termination.should_check(cell_neighbors_processed)
                                    && builder.can_terminate({
                                        let mut bound = worst_cos;
                                        for &next in seed.neighbors.iter().skip(pos + 1) {
                                            let next = next as usize;
                                            if next != i {
                                                bound = points[i].dot(points[next]);
                                                break;
                                            }
                                        }
                                        bound
                                    })
                                {
                                    terminated = true;
                                    break;
                                }
                            }
                        }
                        cell_sub.add_clip(t_clip.elapsed());
                    }

                    if !terminated && builder.is_bounded() {
                        let bound = if packed_count == packed_k_local {
                            worst_cos
                        } else {
                            packed_security
                        };
                        if builder.can_terminate(bound) {
                            terminated = true;
                        }
                    }
                }

                let resume_k = super::KNN_RESUME_K.min(max_neighbors);
                if !terminated && !knn_exhausted && !builder.is_failed() && resume_k > 0 {
                    used_knn = true;
                    max_k_requested = resume_k;
                    neighbors.clear();
                    let t_knn = Timer::start();
                    let status = knn.knn_resumable_into(
                        points[i],
                        i,
                        resume_k,
                        resume_k,
                        &mut scratch,
                        &mut neighbors,
                    );
                    cell_sub.add_knn(t_knn.elapsed());

                    // Track which resume stage we're at
                    knn_stage = KnnCellStage::Resume(resume_k);

                    let t_clip = Timer::start();
                    for (pos, &neighbor_idx) in neighbors.iter().enumerate() {
                        if did_packed && builder.has_neighbor(neighbor_idx) {
                            continue;
                        }
                        #[cfg(debug_assertions)]
                        debug_assert!(
                            !builder.has_neighbor(neighbor_idx),
                            "kNN resume returned duplicate neighbor {} for cell {}",
                            neighbor_idx, i
                        );
                        let neighbor = points[neighbor_idx];
                        if builder.clip(neighbor_idx, neighbor).is_err() {
                            break;
                        }
                        cell_neighbors_processed += 1;
                        let dot = points[i].dot(neighbor);
                        worst_cos = worst_cos.min(dot);

                        if builder.is_bounded() {
                            if termination.should_check(cell_neighbors_processed)
                                && builder.can_terminate({
                                    let mut bound = worst_cos;
                                    for &next in neighbors.iter().skip(pos + 1) {
                                        if next != i {
                                            bound = points[i].dot(points[next]);
                                            break;
                                        }
                                    }
                                    bound
                                })
                            {
                                terminated = true;
                                break;
                            }
                        }
                    }
                    cell_sub.add_clip(t_clip.elapsed());

                    if terminated {
                        knn_exhausted = status == crate::cube_grid::KnnStatus::Exhausted;
                    } else if status == crate::cube_grid::KnnStatus::Exhausted {
                        knn_exhausted = true;
                    }
                }

                if !terminated && !knn_exhausted && !builder.is_failed() {
                    for &k_stage in super::KNN_RESTART_KS.iter() {
                        let k = k_stage.min(max_neighbors);
                        if k == 0 || k <= max_k_requested {
                            continue;
                        }
                        used_knn = true;
                        max_k_requested = k;
                        neighbors.clear();

                        let t_knn = Timer::start();
                        let status = knn.knn_resumable_into(
                            points[i],
                            i,
                            k,
                            k,
                            &mut scratch,
                            &mut neighbors,
                        );
                        cell_sub.add_knn(t_knn.elapsed());

                        // Track which restart stage we're at
                        knn_stage = KnnCellStage::Restart(k_stage);

                        let t_clip = Timer::start();
                        for (pos, &neighbor_idx) in neighbors.iter().enumerate() {
                            if builder.has_neighbor(neighbor_idx) {
                                continue;
                            }
                            let neighbor = points[neighbor_idx];
                            if builder.clip(neighbor_idx, neighbor).is_err() {
                                break;
                            }
                            cell_neighbors_processed += 1;
                            let dot = points[i].dot(neighbor);
                            worst_cos = worst_cos.min(dot);

                            if builder.is_bounded() {
                                if termination.should_check(cell_neighbors_processed)
                                    && builder.can_terminate({
                                        let mut bound = worst_cos;
                                        for &next in neighbors.iter().skip(pos + 1) {
                                            if next != i {
                                                bound = points[i].dot(points[next]);
                                                break;
                                            }
                                        }
                                        bound
                                    })
                                {
                                    terminated = true;
                                    break;
                                }
                            }
                        }
                        cell_sub.add_clip(t_clip.elapsed());

                        if terminated {
                            break;
                        }

                        knn_exhausted =
                            status == crate::cube_grid::KnnStatus::Exhausted;
                        if knn_exhausted {
                            break;
                        }
                    }
                }

                let max_knn_target = super::KNN_RESTART_MAX.min(max_neighbors);
                if max_k_requested >= max_knn_target {
                    reached_schedule_max_k = true;
                }

                // Final termination check at the end
                if !terminated && builder.is_bounded() {
                    let bound = if used_knn {
                        worst_cos
                    } else if did_packed && packed_count < packed_k_local {
                        packed_security
                    } else {
                        worst_cos
                    };
                    if builder.can_terminate(bound) {
                        terminated = true;
                    }
                }

                // If termination is enabled and the cell is bounded but still not proven,
                // keep requesting more neighbors until `can_terminate()` succeeds.
                //
                // Important: without this, the algorithm can accept "bounded but unproven" cells,
                // which causes asymmetric edges at high densities.
                if !terminated
                    && reached_schedule_max_k
                    && !builder.is_failed()
                    && builder.is_bounded()
                {
                    let mut k = max_k_requested.max(super::KNN_RESTART_MAX).min(max_neighbors);
                    let cap = termination_max_k_cap
                        .unwrap_or(max_neighbors)
                        .min(max_neighbors)
                        .max(k);

                    const K_STEP_MIN: usize = 32;

                    while !terminated && !builder.is_failed() && k < cap {
                        let next_k = (k.saturating_mul(2))
                            .max(k + K_STEP_MIN)
                            .min(cap);
                        if next_k <= k {
                            break;
                        }

                        used_knn = true;
                        max_k_requested = next_k;
                        neighbors.clear();

                        let t_knn = Timer::start();
                        let status = knn.knn_resumable_into(
                            points[i],
                            i,
                            next_k,
                            next_k,
                            &mut scratch,
                            &mut neighbors,
                        );
                        cell_sub.add_knn(t_knn.elapsed());
                        knn_stage = KnnCellStage::Restart(next_k);

                        let t_clip = Timer::start();
                        for (pos, &neighbor_idx) in neighbors.iter().enumerate() {
                            if builder.has_neighbor(neighbor_idx) {
                                continue;
                            }
                            let neighbor = points[neighbor_idx];
                            if builder.clip(neighbor_idx, neighbor).is_err() {
                                break;
                            }
                            cell_neighbors_processed += 1;
                            let dot = points[i].dot(neighbor);
                            worst_cos = worst_cos.min(dot);

                            if termination.should_check(cell_neighbors_processed)
                                && builder.can_terminate({
                                    let mut bound = worst_cos;
                                    for &next in neighbors.iter().skip(pos + 1) {
                                        if next != i {
                                            bound = points[i].dot(points[next]);
                                            break;
                                        }
                                    }
                                    bound
                                })
                            {
                                terminated = true;
                                break;
                            }
                        }
                        cell_sub.add_clip(t_clip.elapsed());

                        knn_exhausted = status == crate::cube_grid::KnnStatus::Exhausted;
                        k = next_k;

                        // Conservative bound on unseen: the k-th neighbor dot.
                        // (kNN results are sorted by distance; for unit vectors, distance order
                        // matches dot order.)
                        let kth_dot = neighbors
                            .last()
                            .map(|&j| points[i].dot(points[j]))
                            .unwrap_or(worst_cos);
                        if builder.can_terminate(kth_dot) {
                            terminated = true;
                        }

                        // If we've effectively clipped against all possible neighbors, there's
                        // nothing left unseen.
                        if k >= max_neighbors {
                            terminated = true;
                        }

                        // If we brute-forced and still can't terminate, we'll continue growing k
                        // until cap (possibly all points).
                    }

                }

                // Full scan fallback if cell is not bounded after kNN
                // With Topo2DBuilder, we just need to keep adding neighbors until bounded
                if !builder.is_bounded() && !builder.is_failed() {
                    did_full_scan_fallback = true;
                    let already_clipped: rustc_hash::FxHashSet<usize> =
                        builder.neighbor_indices_iter().collect();
                    let gen = points[i];
                    let mut sorted_indices: Vec<usize> = (0..points.len())
                        .filter(|&j| j != i && !already_clipped.contains(&j))
                        .collect();
                    sorted_indices.sort_by(|&a, &b| {
                        let da = gen.dot(points[a]);
                        let db = gen.dot(points[b]);
                        db.partial_cmp(&da).unwrap()
                    });
                    for p_idx in sorted_indices {
                        if builder.clip(p_idx, points[p_idx]).is_err() {
                            break;
                        }
                        cell_neighbors_processed += 1;
                        // Check if now bounded
                        if builder.is_bounded() {
                            break;
                        }
                    }
                    full_scan_done = true;
                }

                // If still not bounded or failed, panic with diagnostics
                if !builder.is_bounded() || builder.is_failed() {
                    let (active, total) = builder.count_active_planes();
                    let gen = points[i];
                    let neighbor_indices: Vec<usize> = builder.neighbor_indices_iter().collect();

                    panic!(
                        "Cell {} construction failed: bounded={}, failure={:?}, \
                         planes={}, active={}, vertices={}, \
                         did_packed={}, did_knn={}, did_full_scan={}\n\
                         Generator pos: {:?}\n\
                         First 10 neighbor indices: {:?}",
                        i,
                        builder.is_bounded(),
                        builder.failure(),
                        total,
                        active,
                        builder.vertex_count(),
                        did_packed,
                        used_knn,
                        full_scan_done,
                        gen,
                        &neighbor_indices[..neighbor_indices.len().min(10)],
                    );
                }

                let knn_stage = if did_full_scan_fallback {
                    KnnCellStage::FullScanFallback
                } else {
                    knn_stage
                };
                cell_sub.add_cell_stage(knn_stage, knn_exhausted, cell_neighbors_processed);

                // Phase 4: Extract vertices with triplet keys
                let t_cert = Timer::start();
                builder
                    .to_vertex_data_with_edge_neighbors_into(
                        &mut cell_vertices,
                        &mut edge_neighbors,
                    )
                    .expect("to_vertex_data_with_edge_neighbors_into failed after bounded check");
                cell_sub.add_cert(t_cert.elapsed());

                let cell_idx = i as u32;
                let t_edge_collect = Timer::start();
                collect_cell_edges(
                    cell_idx,
                    local as usize,
                    &cell_vertices,
                    &edge_neighbors,
                    &assignment,
                    &mut edges_to_earlier,
                    &mut edges_to_later,
                    &mut edges_overflow,
                );
                vertex_indices.clear();
                vertex_indices.resize(cell_vertices.len(), INVALID_INDEX);
                cell_sub.add_edge_collect(t_edge_collect.elapsed());

                let t_edge_resolve = Timer::start();
                resolve_cell_edge_checks(
                    &mut shard,
                    local as usize,
                    &mut edges_to_earlier,
                    &mut vertex_indices,
                    &mut edge_matched,
                );
                cell_sub.add_edge_resolve(t_edge_resolve.elapsed());

                let count = cell_vertices.len();
                shard.output.cell_counts[local as usize] =
                    u8::try_from(count).expect("cell vertex count exceeds u8 capacity");

                let t_keys = Timer::start();
                for (idx, (key, pos)) in cell_vertices.iter().copied().enumerate() {
                    #[cfg(feature = "timing")]
                    {
                        shard.triplet_keys += 1;
                    }
                    let owner_bin = assignment.generator_bin[key[0] as usize];
                    if owner_bin == bin {
                        if vertex_indices[idx] == INVALID_INDEX {
                            let new_idx = shard.output.vertices.len() as u32;
                            shard.output.vertices.push(pos);
                            shard.output.vertex_keys.push(key);
                            vertex_indices[idx] = new_idx;
                        }
                        let v_idx = vertex_indices[idx];
                        debug_assert!(v_idx != INVALID_INDEX, "missing on-shard vertex index");
                        shard.output.cell_indices.push(pack_ref(bin, v_idx));
                    } else {
                        debug_assert_eq!(
                            vertex_indices[idx],
                            INVALID_INDEX,
                            "received index for off-shard owner"
                        );
                        let source_slot = shard.output.cell_indices.len() as u32;
                        shard.output.cell_indices.push(DEFERRED);
                        shard.output.deferred.push(DeferredSlot {
                            key,
                            pos,
                            source_bin: bin,
                            source_slot,
                        });
                    }
                }
                cell_sub.add_key_dedup(t_keys.elapsed());

                let t_edge_emit = Timer::start();
                for entry in edges_to_later.drain(..) {
                    let locals = entry.edge.locals;
                    shard.dedup.push_edge_check(entry.local_b as usize, EdgeCheck {
                        key: entry.edge.key,
                        endpoints: entry.edge.endpoints,
                        indices: [
                            vertex_indices[locals[0] as usize],
                            vertex_indices[locals[1] as usize],
                        ],
                    });
                }

                for entry in edges_overflow.drain(..) {
                    let locals = entry.edge.locals;
                    shard.output.edge_check_overflow.push(EdgeCheckOverflow {
                        key: entry.edge.key,
                        side: entry.side,
                        source_bin: bin,
                        endpoints: entry.edge.endpoints,
                        indices: [
                            vertex_indices[locals[0] as usize],
                            vertex_indices[locals[1] as usize],
                        ],
                        slots: [
                            cell_start + locals[0] as u32,
                            cell_start + locals[1] as u32,
                        ],
                    });
                }
                cell_sub.add_edge_emit(t_edge_emit.elapsed());

                debug_assert_eq!(
                    shard.output.cell_indices.len() as u32 - cell_start,
                    count as u32,
                    "cell index stream mismatch"
                );

            };

            let mut cursor = 0usize;
            while cursor < my_generators.len() {
                let cell = grid.point_index_to_cell(my_generators[cursor]) as u32;
                let start = cursor;
                while cursor < my_generators.len()
                    && grid.point_index_to_cell(my_generators[cursor]) as u32 == cell
                {
                    cursor += 1;
                }
                let group_start = start;

                if packed_k > 0 {
                    let queries = &packed_queries_all[group_start..cursor];

                    // NOTE: `packed_knn_cell_stream` invokes the callback per query.
                    // The callback builds the Voronoi cell and is separately timed (clipping,
                    // certification, key_dedup, and any fallback knn work). If we time the whole
                    // call naively, we'd double-count that work under `packed_knn`.
                    let t_packed = Timer::start();
                    let status = packed_knn_cell_stream(
                        grid,
                        points,
                        cell as usize,
                        queries,
                        packed_k,
                        &mut packed_scratch,
                        &mut packed_timings,
                        |qi, query_idx, neighbors, count, security| {
                            let local = u32::try_from(group_start + qi)
                                .expect("local index must fit in u32");
                            let seed = PackedSeed {
                                neighbors,
                                count,
                                security,
                                k: packed_k,
                            };
                            process_cell(&mut sub_accum, query_idx as usize, local, Some(seed));
                        },
                    );
                    let packed_elapsed = t_packed.elapsed();

                    if status == PackedKnnCellStatus::SlowPath {
                        for local in group_start..cursor {
                            let global = my_generators[local];
                            let local_u32 =
                                u32::try_from(local).expect("local index must fit in u32");
                            process_cell(&mut sub_accum, global, local_u32, None);
                        }
                    }

                    // Attribute only the packed k-NN overhead to `packed_knn`, excluding the work
                    // done inside `process_cell` (which has its own sub-phase timers).
                    #[cfg(feature = "timing")]
                    {
                        let overhead_total =
                            packed_elapsed.saturating_sub(packed_timings.callback);
                        sub_accum.add_packed_knn(overhead_total);

                        sub_accum.add_packed_knn_setup(packed_timings.setup);
                        sub_accum.add_packed_knn_query_cache(packed_timings.query_cache);
                        sub_accum.add_packed_knn_security_thresholds(
                            packed_timings.security_thresholds,
                        );
                        sub_accum.add_packed_knn_center_pass(packed_timings.center_pass);
                        sub_accum.add_packed_knn_ring_thresholds(packed_timings.ring_thresholds);
                        sub_accum.add_packed_knn_ring_pass(packed_timings.ring_pass);
                        sub_accum.add_packed_knn_ring_fallback(packed_timings.ring_fallback);
                        sub_accum.add_packed_knn_select_sort(packed_timings.select_sort);

                        let measured = packed_timings.total();
                        sub_accum.add_packed_knn_other(overhead_total.saturating_sub(measured));
                    }
                    #[cfg(not(feature = "timing"))]
                    sub_accum.add_packed_knn(packed_elapsed);
                } else {
                    for local in group_start..cursor {
                        let global = my_generators[local];
                        let local_u32 =
                            u32::try_from(local).expect("local index must fit in u32");
                        process_cell(&mut sub_accum, global, local_u32, None);
                    }
                }
            }

            (shard, sub_accum)
        })
        .collect();

    let mut shards: Vec<ShardState> = Vec::with_capacity(num_bins);
    let mut merged_sub = super::timing::CellSubAccum::new();
    for (shard, sub) in per_bin {
        merged_sub.merge(&sub);
        shards.push(shard);
    }

    ShardedCellsData {
        assignment,
        shards,
        cell_sub: merged_sub,
    }
}

pub(super) fn assemble_sharded_live_dedup(
    mut data: ShardedCellsData,
) -> (
    Vec<Vec3>,
    Vec<VertexKey>,
    Vec<BadEdgeRecord>,
    Vec<VoronoiCell>,
    Vec<u32>,
    DedupSubPhases,
) {
    let t0 = Timer::start();

    let num_bins = data.assignment.num_bins;

    // Phase 3: collect overflow by target bin
    let mut support_by_target: Vec<Vec<SupportOverflow>> =
        (0..num_bins).map(|_| Vec::new()).collect();
    for shard in &mut data.shards {
        for entry in shard.dedup.support_overflow.drain(..) {
            support_by_target[entry.target_bin as usize].push(entry);
        }
    }

    #[allow(unused_variables)]
    let overflow_collect_time = t0.elapsed();
    let t1 = Timer::start();

    // Phase 3: overflow flush (V1: single-threaded)
    for target in 0..num_bins {
        // Support sets
        for entry in support_by_target[target].drain(..) {
            let source = entry.source_bin as usize;
            let target = target;
            debug_assert_ne!(source, target, "overflow should not target same bin");
            let (source_shard, target_shard) = with_two_mut(&mut data.shards, source, target);

            let idx = target_shard.dedup_support_owned(entry.support, entry.pos);
            source_shard.output.cell_indices[entry.source_slot as usize] =
                pack_ref(target as u32, idx);
        }
    }

    let mut bad_edges: Vec<BadEdgeRecord> = Vec::new();
    let mut edge_check_overflow: Vec<EdgeCheckOverflow> = Vec::new();
    let mut deferred_slots: Vec<DeferredSlot> = Vec::new();
    for shard in &mut data.shards {
        bad_edges.extend(shard.output.bad_edges.drain(..));
        edge_check_overflow.extend(shard.output.edge_check_overflow.drain(..));
        deferred_slots.extend(shard.output.deferred.drain(..));
    }

    let t_edge_sort = Timer::start();
    edge_check_overflow.sort_unstable_by_key(|entry| (entry.key, entry.side));
    let edge_checks_overflow_sort_time = t_edge_sort.elapsed();
    let t_edge_match = Timer::start();
    let mut patch_slot = |slot: &mut u64, owner_bin: u32, idx: u32| {
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
                let (a_shard, b_shard) = with_two_mut(
                    &mut data.shards,
                    a.source_bin as usize,
                    b.source_bin as usize,
                );
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
    #[allow(unused_variables)]
    let edge_checks_overflow_match_time = t_edge_match.elapsed();
    #[allow(unused_variables)]
    let edge_checks_overflow_time =
        edge_checks_overflow_sort_time + edge_checks_overflow_match_time;

    let t_deferred = Timer::start();
    let mut fallback_map: FxHashMap<VertexKey, (u32, u32)> = FxHashMap::default();
    for entry in deferred_slots {
        let source_bin = entry.source_bin as usize;
        let source_slot = entry.source_slot as usize;
        if data.shards[source_bin].output.cell_indices[source_slot] != DEFERRED {
            continue;
        }

        let owner_bin = data.assignment.generator_bin[entry.key[0] as usize] as u32;
        let idx = if let Some(&(bin, idx)) = fallback_map.get(&entry.key) {
            debug_assert_eq!(bin, owner_bin, "fallback owner bin mismatch");
            idx
        } else {
            let new_idx = {
                let owner_shard = &mut data.shards[owner_bin as usize];
                let new_idx = owner_shard.output.vertices.len() as u32;
                owner_shard.output.vertices.push(entry.pos);
                owner_shard.output.vertex_keys.push(entry.key);
                new_idx
            };
            fallback_map.insert(entry.key, (owner_bin, new_idx));
            new_idx
        };

        let slot = &mut data.shards[source_bin].output.cell_indices[source_slot];
        patch_slot(slot, owner_bin, idx);
    }
    #[allow(unused_variables)]
    let deferred_fallback_time = t_deferred.elapsed();

    #[cfg(debug_assertions)]
    for shard in &data.shards {
        debug_assert!(
            !shard.output.cell_indices.iter().any(|&x| x == DEFERRED),
            "unresolved deferred indices remain after overflow flush"
        );
    }

    #[allow(unused_variables)]
    let overflow_flush_time = t1.elapsed();

    // Convert to ShardFinal, dropping dedup structures to reduce memory pressure
    let finals: Vec<ShardFinal> = std::mem::take(&mut data.shards)
        .into_iter()
        .map(|s| s.into_final())
        .collect();

    let t2 = Timer::start();

    // Phase 4: concatenate vertices
    let mut vertex_offsets: Vec<u32> = vec![0; num_bins];
    let mut total_vertices = 0usize;
    for (bin, shard) in finals.iter().enumerate() {
        vertex_offsets[bin] =
            u32::try_from(total_vertices).expect("total vertex count exceeds u32 capacity");
        total_vertices += shard.output.vertices.len();
    }

    let mut all_vertices: Vec<Vec3> = Vec::with_capacity(total_vertices);
    let mut all_vertex_keys: Vec<VertexKey> = Vec::with_capacity(total_vertices);
    for shard in &finals {
        debug_assert_eq!(
            shard.output.vertices.len(),
            shard.output.vertex_keys.len(),
            "vertex keys out of sync with vertex positions"
        );
        all_vertices.extend_from_slice(&shard.output.vertices);
        all_vertex_keys.extend_from_slice(&shard.output.vertex_keys);
    }

    let num_cells = data.assignment.generator_bin.len();
    #[allow(unused_variables)]
    let concat_vertices_time = t2.elapsed();
    let t3 = Timer::start();

    // Phase 4: emit cells in generator index order (prefix-sum + direct fill).
    let mut cell_starts_global: Vec<u32> = vec![0; num_cells + 1];
    let mut total_cell_indices = 0u32;
    for gen_idx in 0..num_cells {
        let bin = data.assignment.generator_bin[gen_idx] as usize;
        let local = data.assignment.global_to_local[gen_idx] as usize;
        let count = finals[bin].output.cell_counts[local] as u32;
        total_cell_indices = total_cell_indices
            .checked_add(count)
            .expect("cell index buffer exceeds u32 capacity");
        cell_starts_global[gen_idx + 1] = total_cell_indices;
    }

    let mut cells: Vec<VoronoiCell> = vec![VoronoiCell::new(0, 0); num_cells];
    let mut cell_indices: Vec<u32> = vec![0; total_cell_indices as usize];

    #[cfg(debug_assertions)]
    {
        let expected_indices: usize = finals
            .iter()
            .map(|shard| shard.output.cell_indices.len())
            .sum();
        debug_assert_eq!(
            expected_indices,
            cell_indices.len(),
            "cell index count mismatch after prefix sum"
        );
    }

    let cell_indices_ptr = cell_indices.as_mut_ptr() as usize;
    maybe_par_iter_mut!(&mut cells)
        .enumerate()
        .for_each(|(gen_idx, cell)| {
            let bin = data.assignment.generator_bin[gen_idx] as usize;
            let local = data.assignment.global_to_local[gen_idx] as usize;
            let shard = &finals[bin];
            let start = shard.output.cell_starts[local] as usize;
            let count = shard.output.cell_counts[local] as usize;

            let dst_start = cell_starts_global[gen_idx] as usize;
            let src = &shard.output.cell_indices[start..start + count];
            // Safety: pointer is valid for the buffer and each cell writes a disjoint range.
            unsafe {
                let dst = (cell_indices_ptr as *mut u32).add(dst_start);
                for (i, &packed) in src.iter().enumerate() {
                    debug_assert_ne!(packed, DEFERRED, "deferred index leaked to assembly");
                    let (vbin, local) = unpack_ref(packed);
                    dst.add(i)
                        .write(vertex_offsets[vbin as usize] + local);
                }
            }
            let count_u16 = u16::from(shard.output.cell_counts[local]);
            *cell = VoronoiCell::new(cell_starts_global[gen_idx], count_u16);
        });
    #[allow(unused_variables)]
    let emit_cells_time = t3.elapsed();

    #[cfg(feature = "timing")]
    let sub_phases = DedupSubPhases {
        overflow_collect: overflow_collect_time,
        overflow_flush: overflow_flush_time,
        edge_checks_overflow: edge_checks_overflow_time,
        edge_checks_overflow_sort: edge_checks_overflow_sort_time,
        edge_checks_overflow_match: edge_checks_overflow_match_time,
        deferred_fallback: deferred_fallback_time,
        concat_vertices: concat_vertices_time,
        emit_cells: emit_cells_time,
        triplet_keys: finals.iter().map(|s| s.triplet_keys).sum(),
        support_keys: finals.iter().map(|s| s.support_keys).sum(),
    };
    #[cfg(not(feature = "timing"))]
    let sub_phases = DedupSubPhases;

    (
        all_vertices,
        all_vertex_keys,
        bad_edges,
        cells,
        cell_indices,
        sub_phases,
    )
}
