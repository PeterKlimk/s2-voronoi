//! Tier-2 re-clip repair (spherical).
//!
//! Runs only on the residual that the cheap Tier-1 stitch (`edge_reconcile`)
//! could not pair. Each surviving unpaired interior edge marks a contested
//! cluster of cells that disagree on a high-degree degenerate vertex; this
//! pass re-resolves each cluster jointly against a pinned boundary of
//! already-valid cells. See `docs/reclip-repair-design.md`.
//!
//! Opt-in via `S2_RECLIP_REPAIR` while under development; the default path is
//! unchanged (the loud `residual_error` still fires on any survivor).

use crate::diagram::VoronoiCell;
use crate::knn_clipping::edge_reconcile::{self, VertexKeys};
use crate::knn_clipping::union_find::SparseUnionFind;
use crate::live_dedup::ShardedVertexKeys;
use glam::Vec3;
use std::collections::BTreeMap;

/// Whether the Tier-2 re-clip pass is enabled (off by default).
pub(crate) fn enabled() -> bool {
    std::env::var("S2_RECLIP_REPAIR").is_ok_and(|v| !v.is_empty() && v != "0")
}

fn trace() -> bool {
    std::env::var("S2_RECLIP_TRACE").is_ok()
}

/// One contested cluster: the cells (generators) that disagree about a
/// high-degree degenerate vertex, plus the raw unpaired interior edges within
/// it. The boundary (already-paired edges to outside cells) is derived later
/// by the resolver.
pub(crate) struct Component {
    /// Generator indices of the cells in this component (sorted, unique).
    pub(crate) cells: Vec<u32>,
    /// Raw unpaired interior edges `(va, vb, owner)` belonging to this cluster.
    pub(crate) edges: Vec<(u32, u32, u32)>,
}

/// Identify the contested connected components from Tier-1's residual.
///
/// The component is the transitive closure of *contested adjacency*: every
/// generator named in either endpoint key of an unpaired edge is unioned, so a
/// degenerate vertex's whole petal lands in one component (matches the
/// de-risking measurement). By construction the component's outer boundary
/// consists only of already-paired edges (a contested vertex on the boundary
/// would pull its outside cell in), so it is safe to pin.
fn identify_components(
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
    residual: &[(u32, u32)],
) -> Result<Vec<Component>, crate::VoronoiError> {
    // Candidate cells = those named by Tier-1's residual pairs. Tier-1 derived
    // those pairs from the residual edges, so every residual edge's owner is
    // covered; the localized scan re-surfaces the raw `(va, vb, owner)` edges.
    let mut candidates: Vec<u32> = residual.iter().flat_map(|&(a, b)| [a, b]).collect();
    candidates.sort_unstable();
    candidates.dedup();

    let raw = edge_reconcile::scan_unpaired_interior(
        cells,
        cell_indices,
        VertexKeys::Sharded(vertex_keys),
        &candidates,
        &|_, _| false,
    )?;

    // Union every generator incident to a contested edge's two endpoints.
    let mut uf = SparseUnionFind::new();
    for &(va, vb, owner) in &raw {
        let mut gens: Vec<u32> = Vec::with_capacity(6);
        for v in [va, vb] {
            if let Some(k) = vertex_keys.get(v) {
                gens.extend_from_slice(&k);
            }
        }
        if gens.is_empty() {
            gens.push(owner);
        }
        let first = gens[0];
        for &g in &gens[1..] {
            uf.union(first, g);
        }
    }

    // Group edges and cells by component root.
    let mut by_root: BTreeMap<u32, Component> = BTreeMap::new();
    for &(va, vb, owner) in &raw {
        // Attribute the edge to the component of its first incident generator.
        let anchor = vertex_keys
            .get(va)
            .map(|k| k[0])
            .or_else(|| vertex_keys.get(vb).map(|k| k[0]))
            .unwrap_or(owner);
        let r = uf.find(anchor);
        let comp = by_root.entry(r).or_insert_with(|| Component {
            cells: Vec::new(),
            edges: Vec::new(),
        });
        comp.edges.push((va, vb, owner));
        for v in [va, vb] {
            if let Some(k) = vertex_keys.get(v) {
                comp.cells.extend_from_slice(&k);
            }
        }
    }

    let mut comps: Vec<Component> = by_root.into_values().collect();
    for c in &mut comps {
        c.cells.sort_unstable();
        c.cells.dedup();
    }
    Ok(comps)
}

/// Re-resolve contested components that survived Tier-1, returning the residual
/// still unpaired afterward (empty on full success).
///
/// Currently identifies the components (and traces them under `S2_RECLIP_TRACE`)
/// but does not yet re-clip — returns the input residual unchanged. The
/// resolver and re-stitch phases land incrementally.
// `&mut Vec` (not `&mut [_]`): the re-stitch phase appends re-resolved cell
// spans to `cell_indices`, so it needs to grow the buffer, not just mutate it.
#[allow(clippy::too_many_arguments, clippy::ptr_arg)]
pub(crate) fn repair(
    _points: &[Vec3],
    _vertices: &[Vec3],
    cells: &mut Vec<VoronoiCell>,
    cell_indices: &mut Vec<u32>,
    vertex_keys: &ShardedVertexKeys,
    residual: Vec<(u32, u32)>,
) -> Result<Vec<(u32, u32)>, crate::VoronoiError> {
    let comps = identify_components(cells, cell_indices, vertex_keys, &residual)?;
    if trace() {
        eprintln!(
            "[reclip] {} residual pair(s) -> {} component(s)",
            residual.len(),
            comps.len()
        );
        for (i, c) in comps.iter().enumerate() {
            eprintln!(
                "  comp {i}: {} cells, {} contested edges; cells={:?}",
                c.cells.len(),
                c.edges.len(),
                c.cells
            );
        }
    }
    Ok(residual)
}
