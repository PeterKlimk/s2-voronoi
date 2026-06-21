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

use super::boundary;
use crate::cube_grid::CubeMapGrid;
use crate::diagram::VoronoiCell;
use crate::knn_clipping::edge_reconcile::{self, VertexKeys};
use crate::knn_clipping::union_find::SparseUnionFind;
use crate::live_dedup::ShardedVertexKeys;
use glam::{DVec3, Vec3};
use std::collections::{BTreeMap, HashMap, HashSet};

type VertexKey3 = [u32; 3];
type KeyEdge = (VertexKey3, VertexKey3);

/// Cap on a single component's cell count; larger contested regions fail loud
/// (returned as residual) rather than risk an unbounded re-clip. The interior
/// solve is O(|G|^3 * filter), so this also bounds cost; the de-risking
/// measurement put the largest real component at ~50 generators.
const MAX_COMPONENT_CELLS: usize = 128;

/// Whether the Tier-2 re-clip pass is enabled (off by default).
pub(crate) fn enabled() -> bool {
    std::env::var("S2_RECLIP_REPAIR").is_ok_and(|v| !v.is_empty() && v != "0")
}

fn trace() -> bool {
    std::env::var("S2_RECLIP_TRACE").is_ok()
}

/// Observational seam-classification diagnostic (no repair). See `diagnose_seam`.
fn diag() -> bool {
    std::env::var("S2_RECLIP_DIAG").is_ok()
}

/// Rebuild the free local hull per component and classify each contested↔ring
/// seam face WITHOUT stitching, to size the overlay/collapse design: how many
/// seam vertices the exact hull reproduces (`match` — pin would succeed) vs
/// disagrees (`mismatch`), split by C–V–V / C–C–V, and for each mismatch the
/// angular distance to the nearest existing ring-cell vertex (the collapsibility
/// proxy: `coincident` = mergeable/deletable, `far` = genuine disagreement).
fn diagnose_seam(
    points: &[Vec3],
    grid: &CubeMapGrid,
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
    comps: &[Component],
) {
    const SECURED_K: usize = 384;
    const MAX_LOCAL_SET: usize = 3072;

    let (mut n_comp, mut n_bail) = (0usize, 0usize);
    let (mut seam, mut matched) = (0usize, 0usize);
    let (mut mm_cvv, mut mm_ccv) = (0usize, 0usize);
    let (mut g_coinc, mut g_near, mut g_far) = (0usize, 0usize, 0usize);
    let mut comp_sizes: Vec<usize> = Vec::new();

    for comp in comps {
        n_comp += 1;
        comp_sizes.push(comp.cells.len());
        let cset: HashSet<u32> = comp.cells.iter().copied().collect();

        // Local point set S = C ∪ secured(C) (nearest-K of the grid neighborhood).
        let mut sset = cset.clone();
        let mut overflow = false;
        for &g in &comp.cells {
            if (g as usize) >= points.len() {
                overflow = true;
                break;
            }
            let gp = points[g as usize];
            let cell = grid.point_index_to_cell(g as usize);
            let mut cand: HashSet<u32> = HashSet::new();
            for &nc in grid.cell_neighbors(cell).iter() {
                if nc != u32::MAX {
                    cand.extend(grid.cell_points(nc as usize).iter().copied());
                }
            }
            for &nc in grid.cell_ring2(cell) {
                if nc != u32::MAX {
                    cand.extend(grid.cell_points(nc as usize).iter().copied());
                }
            }
            let mut by_dist: Vec<(f32, u32)> = cand
                .into_iter()
                .filter(|&c| c != g && (c as usize) < points.len())
                .map(|c| (gp.dot(points[c as usize]), c))
                .collect();
            by_dist.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            for &(_, id) in by_dist.iter().take(SECURED_K) {
                sset.insert(id);
            }
            if sset.len() > MAX_LOCAL_SET {
                overflow = true;
                break;
            }
        }
        if overflow {
            n_bail += 1;
            continue;
        }

        let mut s_ids: Vec<u32> = sset.into_iter().collect();
        s_ids.sort_unstable();
        let s_pts: Vec<Vec3> = s_ids.iter().map(|&id| points[id as usize]).collect();
        let Some(hull) = crate::knn_clipping::local_hull::LocalHull::build(&s_pts) else {
            n_bail += 1;
            continue;
        };

        // Existing ring-cell vertex keys (match test) + positions (geometry test).
        let mut key_to_vid: HashMap<VertexKey3, u32> = HashMap::new();
        for &id in &s_ids {
            if cset.contains(&id) || (id as usize) >= cells.len() {
                continue;
            }
            if let Some(span) = cell_span(cells, cell_indices, id) {
                for &vid in span {
                    if let Some(k) = vertex_keys.get(vid) {
                        key_to_vid.entry(k).or_insert(vid);
                    }
                }
            }
        }

        // Classify each hull face once: seam = some-but-not-all contested.
        for (fi, f) in hull.faces().iter().enumerate() {
            let (ga, gb, gc) = (s_ids[f[0]], s_ids[f[1]], s_ids[f[2]]);
            let ncon = [ga, gb, gc].iter().filter(|x| cset.contains(x)).count();
            if ncon == 0 || ncon == 3 {
                continue; // pure-ring or interior, not a seam
            }
            seam += 1;
            let key = key3(ga, gb, gc);
            if key_to_vid.contains_key(&key) {
                matched += 1;
                continue;
            }
            if ncon == 1 {
                mm_cvv += 1;
            } else {
                mm_ccv += 1;
            }
            // Nearest existing vertex among the involved ring cells.
            let cc = hull.face_circumcenter(fi);
            let ccf = Vec3::new(cc.x as f32, cc.y as f32, cc.z as f32);
            let mut best = f32::INFINITY;
            for &gen in &[ga, gb, gc] {
                if cset.contains(&gen) || (gen as usize) >= cells.len() {
                    continue;
                }
                if let Some(span) = cell_span(cells, cell_indices, gen) {
                    for &vid in span {
                        if (vid as usize) < vertices.len() {
                            let dot = ccf.dot(vertices[vid as usize]).clamp(-1.0, 1.0);
                            let ang = (2.0 * (1.0 - dot)).max(0.0).sqrt(); // ~chord≈angle
                            best = best.min(ang);
                        }
                    }
                }
            }
            if best < 1e-4 {
                g_coinc += 1;
            } else if best < 1e-2 {
                g_near += 1;
            } else {
                g_far += 1;
            }
        }
    }

    comp_sizes.sort_unstable();
    eprintln!(
        "[diag] comps={n_comp} bailed={n_bail} sizes={comp_sizes:?} | seam={seam} match={matched} \
         mismatch={}  (cvv={mm_cvv} ccv={mm_ccv})  mm_dist[coincident<1e-4={g_coinc} near<1e-2={g_near} far>=1e-2={g_far}]",
        mm_cvv + mm_ccv
    );
}

// ============================================================================
// Canonical topology audit (dev-only, `S2_CANON_AUDIT`).
//
// Answers: does the EXACT local Delaunay (local_hull, exact orient3d) ever
// disagree with the fast clipper's topology for a cell that produced NO
// inter-cell disagreement (no residual)? That set is the "bait": coherent
// wrong-but-valid topology the disagreement-triggered repair can never see.
//
// Method per cell g: build the exact local hull over g + its secured
// neighborhood, take g's fan (cell_faces), compare its neighbor set to the
// fast diagram's. Guarded against the free-hull boundary artifact by requiring
// the secured radius to exceed 2x g's fan radius (the bounded-cell certificate:
// no unseen generator beyond 2*max_r can affect g's cell).
//
// The oracle is validated by a uniform-input control: on non-degenerate input
// the fast clipper is correct, so bait must be ~0 there. If it is not, the
// audit (not the clipper) is wrong.
// ============================================================================

fn audit() -> bool {
    std::env::var("S2_CANON_AUDIT").is_ok()
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[inline]
fn ang(a: DVec3, b: DVec3) -> f64 {
    a.normalize().dot(b.normalize()).clamp(-1.0, 1.0).acos()
}

/// Outcome of auditing one cell.
enum CellAudit {
    SkipNoSpan,
    SkipHullFail,
    SkipUnderSecured,
    Agree,
    /// Topology differs. `displacement` = max angular gap (rad) between an exact
    /// fan circumcenter and the nearest fast vertex of g (how far the diagram's
    /// worst vertex is from the exact answer). `exact_only` = neighbors the exact
    /// hull has but the fast clipper dropped; `fast_only` = the reverse.
    Disagree {
        displacement: f64,
        exact_only: usize,
        fast_only: usize,
    },
}

/// Audit a single cell `g` against the exact local hull. `secured_k` neighbors
/// are gathered from the 3x3 + ring-2 grid neighborhood.
#[allow(clippy::too_many_arguments)]
fn audit_cell(
    g: u32,
    points: &[Vec3],
    grid: &CubeMapGrid,
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
    secured_k: usize,
) -> CellAudit {
    let gi = g as usize;
    if gi >= points.len() {
        return CellAudit::SkipNoSpan;
    }
    let Some(span) = cell_span(cells, cell_indices, g) else {
        return CellAudit::SkipNoSpan;
    };
    if span.len() < 3 {
        return CellAudit::SkipNoSpan;
    }
    let gp = points[gi];
    let gpd = DVec3::new(gp.x as f64, gp.y as f64, gp.z as f64);

    // Fast neighbor set + fast vertex positions for g.
    let mut n_fast: HashSet<u32> = HashSet::new();
    let mut fast_verts: Vec<DVec3> = Vec::with_capacity(span.len());
    for &vid in span {
        if (vid as usize) < vertices.len() {
            let v = vertices[vid as usize];
            fast_verts.push(DVec3::new(v.x as f64, v.y as f64, v.z as f64));
        }
        if let Some(key) = vertex_keys.get(vid) {
            for id in key {
                if id != g && (id as usize) < points.len() {
                    n_fast.insert(id);
                }
            }
        }
    }

    // Secured neighborhood: nearest secured_k of the grid neighborhood.
    let cell = grid.point_index_to_cell(gi);
    let mut cand: HashSet<u32> = HashSet::new();
    for &nc in grid.cell_neighbors(cell).iter() {
        if nc != u32::MAX {
            cand.extend(grid.cell_points(nc as usize).iter().copied());
        }
    }
    for &nc in grid.cell_ring2(cell) {
        if nc != u32::MAX {
            cand.extend(grid.cell_points(nc as usize).iter().copied());
        }
    }
    let mut by_dist: Vec<(f32, u32)> = cand
        .into_iter()
        .filter(|&c| c != g && (c as usize) < points.len())
        .map(|c| (gp.dot(points[c as usize]), c))
        .collect();
    by_dist.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    by_dist.truncate(secured_k);
    if by_dist.len() < 4 {
        return CellAudit::SkipHullFail;
    }
    // Angular radius of the secured set (farthest included neighbor).
    let secured_dot = by_dist.last().map(|&(d, _)| d).unwrap_or(1.0);
    let secured_radius = (secured_dot as f64).clamp(-1.0, 1.0).acos();

    let mut s_ids: Vec<u32> = Vec::with_capacity(by_dist.len() + 1);
    s_ids.push(g);
    s_ids.extend(by_dist.iter().map(|&(_, id)| id));
    let s_pts: Vec<Vec3> = s_ids.iter().map(|&id| points[id as usize]).collect();

    let Some(hull) = crate::knn_clipping::local_hull::LocalHull::build(&s_pts) else {
        return CellAudit::SkipHullFail;
    };
    let g_local = 0usize; // g pushed first.
    let fan = hull.cell_faces(g_local);
    if fan.is_empty() {
        return CellAudit::SkipHullFail;
    }

    // Exact neighbor set + fan radius (farthest circumcenter from g).
    let mut n_exact: HashSet<u32> = HashSet::new();
    let mut fan_radius = 0.0f64;
    let mut ccs: Vec<DVec3> = Vec::with_capacity(fan.len());
    for &fi in &fan {
        let f = hull.faces()[fi];
        for &l in &f {
            if l != g_local {
                n_exact.insert(s_ids[l]);
            }
        }
        let cc = hull.face_circumcenter(fi);
        fan_radius = fan_radius.max(ang(gpd, cc));
        ccs.push(cc);
    }

    // Under-secured: g's fan might be truncated by the secured set's own outer
    // boundary. The certificate guarantees completeness only when the secured
    // radius exceeds 2x the fan radius. Otherwise we cannot trust a "disagree".
    if secured_radius < 2.0 * fan_radius {
        return CellAudit::SkipUnderSecured;
    }

    if n_exact == n_fast {
        return CellAudit::Agree;
    }

    // Geometric significance: how far the exact answer's worst vertex is from
    // any fast vertex of g.
    let mut displacement = 0.0f64;
    if !fast_verts.is_empty() {
        for cc in &ccs {
            let nearest = fast_verts
                .iter()
                .map(|fv| ang(*cc, *fv))
                .fold(f64::INFINITY, f64::min);
            displacement = displacement.max(nearest);
        }
    }
    let exact_only = n_exact.difference(&n_fast).count();
    let fast_only = n_fast.difference(&n_exact).count();
    CellAudit::Disagree {
        displacement,
        exact_only,
        fast_only,
    }
}

/// Run the canonical topology audit (no mutation). Audits two populations:
/// `suspect` cells (grid neighborhood of the contested components — where
/// coherent wrong topology would hide, adjacent to detected disagreement) and a
/// strided uniform `control` sample (oracle sanity, expect ~0 bait).
pub(crate) fn audit_if_enabled(
    points: &[Vec3],
    grid: &CubeMapGrid,
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
    residual: &[(u32, u32)],
) {
    if !audit() {
        return;
    }
    let secured_k = env_usize("S2_CANON_AUDIT_K", 256);
    let sample_target = env_usize("S2_CANON_AUDIT_SAMPLE", 4000);

    // Contested cells (for detected-vs-bait classification).
    let contested: HashSet<u32> =
        match identify_components(cells, cell_indices, vertex_keys, residual) {
            Ok(comps) => comps.iter().flat_map(|c| c.cells.iter().copied()).collect(),
            Err(_) => residual.iter().flat_map(|&(a, b)| [a, b]).collect(),
        };

    // Suspect population: grid neighborhood of every contested cell.
    let mut suspects: HashSet<u32> = HashSet::new();
    for &g in &contested {
        if (g as usize) >= points.len() {
            continue;
        }
        let cell = grid.point_index_to_cell(g as usize);
        for &nc in grid.cell_neighbors(cell).iter() {
            if nc != u32::MAX {
                suspects.extend(grid.cell_points(nc as usize).iter().copied());
            }
        }
        for &nc in grid.cell_ring2(cell) {
            if nc != u32::MAX {
                suspects.extend(grid.cell_points(nc as usize).iter().copied());
            }
        }
    }

    // Control population: deterministic strided sample of all cells.
    let n_cells = cells.len();
    let stride = (n_cells / sample_target.max(1)).max(1);
    let mut control: HashSet<u32> = (0..n_cells as u32).step_by(stride).collect();
    control.retain(|g| !suspects.contains(g));

    // Tally per population.
    struct Tally {
        audited: usize,
        skip_nospan: usize,
        skip_hullfail: usize,
        skip_undersecured: usize,
        agree: usize,
        disagree_detected: usize,
        disagree_bait: usize,
        // Bait displacement decade buckets (rad):
        // [<1e-6, 1e-6, 1e-5, 1e-4, 1e-3, >=1e-2].
        disp_buckets: [usize; 6],
        max_disp: f64,
        bait_exact_only: usize, // fast dropped a neighbor exact keeps
        bait_fast_only: usize,  // fast invented a neighbor exact lacks
        examples: Vec<(u32, f64, usize, usize)>,
    }
    impl Tally {
        fn new() -> Self {
            Tally {
                audited: 0,
                skip_nospan: 0,
                skip_hullfail: 0,
                skip_undersecured: 0,
                agree: 0,
                disagree_detected: 0,
                disagree_bait: 0,
                disp_buckets: [0; 6],
                max_disp: 0.0,
                bait_exact_only: 0,
                bait_fast_only: 0,
                examples: Vec::new(),
            }
        }
        fn record(&mut self, g: u32, detected: bool, a: CellAudit) {
            self.audited += 1;
            match a {
                CellAudit::SkipNoSpan => self.skip_nospan += 1,
                CellAudit::SkipHullFail => self.skip_hullfail += 1,
                CellAudit::SkipUnderSecured => self.skip_undersecured += 1,
                CellAudit::Agree => self.agree += 1,
                CellAudit::Disagree {
                    displacement,
                    exact_only,
                    fast_only,
                } => {
                    if detected {
                        self.disagree_detected += 1;
                    } else {
                        self.disagree_bait += 1;
                        let b = if displacement < 1e-6 {
                            0
                        } else if displacement < 1e-5 {
                            1
                        } else if displacement < 1e-4 {
                            2
                        } else if displacement < 1e-3 {
                            3
                        } else if displacement < 1e-2 {
                            4
                        } else {
                            5
                        };
                        self.disp_buckets[b] += 1;
                        self.max_disp = self.max_disp.max(displacement);
                        self.bait_exact_only += exact_only;
                        self.bait_fast_only += fast_only;
                        if self.examples.len() < 12 {
                            self.examples.push((g, displacement, exact_only, fast_only));
                        }
                    }
                }
            }
        }
        fn report(&self, label: &str) {
            eprintln!(
                "[audit:{label}] audited={} agree={} disagree(detected={} bait={}) \
                 skip(nospan={} hullfail={} undersecured={})",
                self.audited,
                self.agree,
                self.disagree_detected,
                self.disagree_bait,
                self.skip_nospan,
                self.skip_hullfail,
                self.skip_undersecured,
            );
            if self.disagree_bait > 0 {
                eprintln!(
                    "[audit:{label}]   BAIT displacement(rad) [<1e-6={} 1e-6={} 1e-5={} 1e-4={} \
                     1e-3={} >=1e-2={}] max={:.3e} | neighbors: exact_only={} fast_only={}",
                    self.disp_buckets[0],
                    self.disp_buckets[1],
                    self.disp_buckets[2],
                    self.disp_buckets[3],
                    self.disp_buckets[4],
                    self.disp_buckets[5],
                    self.max_disp,
                    self.bait_exact_only,
                    self.bait_fast_only,
                );
                for (g, d, eo, fo) in &self.examples {
                    eprintln!(
                        "[audit:{label}]     cell {g}: displacement={d:.3e} rad exact_only={eo} fast_only={fo}"
                    );
                }
            }
        }
    }

    let mut t_suspect = Tally::new();
    for &g in &suspects {
        let detected = contested.contains(&g);
        let a = audit_cell(
            g,
            points,
            grid,
            vertices,
            cells,
            cell_indices,
            vertex_keys,
            secured_k,
        );
        t_suspect.record(g, detected, a);
    }

    let mut t_control = Tally::new();
    for &g in &control {
        let a = audit_cell(
            g,
            points,
            grid,
            vertices,
            cells,
            cell_indices,
            vertex_keys,
            secured_k,
        );
        t_control.record(g, false, a);
    }

    eprintln!(
        "[audit] contested={} suspects={} control={} secured_k={secured_k}",
        contested.len(),
        suspects.len(),
        control.len(),
    );
    t_suspect.report("suspect");
    t_control.report("control");
}

/// Rim probe (dev-only, `S2_RIM_PROBE`). BFS outward in grid-adjacency from the
/// contested components and report the exact-vs-fast disagree-rate per ring, to
/// test whether the contested region is wrapped by a BOUNDED, clean firewall (a
/// ring where the fast diagram agrees with the exact Delaunay, so a rebuild can
/// pin there cleanly — unlike the inside-cap seam that failed 0/6). A firewall
/// is "clean" when a ring has zero disagreements (agree, or sparse-skip = a
/// non-degenerate background cell the window method can't audit — both clean).
fn rim_probe_components(
    points: &[Vec3],
    grid: &CubeMapGrid,
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
    comps: &[Component],
) {
    let secured_k = env_usize("S2_CANON_AUDIT_K", 256);
    let max_rings = env_usize("S2_RIM_MAX_RINGS", 40);
    let mut visited: HashSet<u32> = comps.iter().flat_map(|c| c.cells.iter().copied()).collect();
    let mut frontier: Vec<u32> = visited.iter().copied().collect();
    eprintln!(
        "[rim] components={} ring0(contested)={}",
        comps.len(),
        frontier.len()
    );
    for ring in 1..=max_rings {
        let mut next: Vec<u32> = Vec::new();
        for &g in &frontier {
            if (g as usize) >= points.len() {
                continue;
            }
            let cell = grid.point_index_to_cell(g as usize);
            for &nc in grid.cell_neighbors(cell).iter() {
                if nc != u32::MAX {
                    for &p in grid.cell_points(nc as usize) {
                        if visited.insert(p) {
                            next.push(p);
                        }
                    }
                }
            }
        }
        if next.is_empty() {
            eprintln!("[rim] ring {ring}: exhausted (no clean firewall before exhaustion)");
            break;
        }
        let (mut agree, mut dis, mut skip) = (0usize, 0usize, 0usize);
        for &g in &next {
            match audit_cell(
                g,
                points,
                grid,
                vertices,
                cells,
                cell_indices,
                vertex_keys,
                secured_k,
            ) {
                CellAudit::Agree => agree += 1,
                CellAudit::Disagree { .. } => dis += 1,
                _ => skip += 1,
            }
        }
        let rate = if agree + dis > 0 {
            100.0 * agree as f64 / (agree + dis) as f64
        } else {
            f64::NAN
        };
        eprintln!(
            "[rim] ring {ring}: n={} agree={agree} disagree={dis} skip={skip} agree_rate={rate:.0}%",
            next.len()
        );
        if dis == 0 {
            eprintln!(
                "[rim] CLEAN FIREWALL at ring {ring} (cumulative cells inside = {})",
                visited.len()
            );
            break;
        }
        frontier = next;
    }
}

fn rim_probe() -> bool {
    std::env::var("S2_RIM_PROBE").is_ok()
}

fn boundary_probe() -> bool {
    std::env::var("S2_BOUNDARY_PROBE").is_ok()
}

/// Gather a (super)set of every non-`C` cell adjacent to component `C` (grid
/// neighborhood of every generator plus every outside generator named in a
/// component cell's vertex keys). Shared by the boundary probe and grow probe.
fn gather_ring(
    gset: &HashSet<u32>,
    points: &[Vec3],
    grid: &CubeMapGrid,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
) -> HashSet<u32> {
    let mut ring: HashSet<u32> = HashSet::new();
    for &g in gset {
        if (g as usize) >= points.len() {
            continue;
        }
        let cell = grid.point_index_to_cell(g as usize);
        for &nc in grid.cell_neighbors(cell).iter() {
            if nc != u32::MAX {
                ring.extend(grid.cell_points(nc as usize).iter().copied());
            }
        }
        for &nc in grid.cell_ring2(cell) {
            if nc != u32::MAX {
                ring.extend(grid.cell_points(nc as usize).iter().copied());
            }
        }
        if let Some(span) = cell_span(cells, cell_indices, g) {
            for &vid in span {
                if let Some(k) = vertex_keys.get(vid) {
                    ring.extend(k.iter().copied());
                }
            }
        }
    }
    ring.retain(|h| !gset.contains(h));
    ring
}

/// Local edge-pairing check over `cellset` (= C ∪ ring): for each target vid,
/// how many incident undirected edges it has and how many of those are UNPAIRED
/// (used once, or twice in the same orientation) in the current diagram. Used to
/// test the hypothesis that a boundary imbalance coincides with a genuine
/// (but Tier-1-undetected) edge mismatch. Target vids are interior to `cellset`
/// (rim of C), so their incident edges have both cells in `cellset` and no
/// outer-border false-positive applies.
fn local_unpaired_incident(
    target_vids: &HashSet<u32>,
    cellset: &HashSet<u32>,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> HashMap<u32, (usize, usize)> {
    // group key (lo,hi) -> (count, forward_count)
    let mut groups: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
    for &g in cellset {
        let Some(span) = cell_span(cells, cell_indices, g) else {
            continue;
        };
        let n = span.len();
        for i in 0..n {
            let a = span[i];
            let b = span[(i + 1) % n];
            if a == b {
                continue;
            }
            let (lo, hi, fwd) = if a < b { (a, b, 1u32) } else { (b, a, 0u32) };
            let e = groups.entry((lo, hi)).or_insert((0, 0));
            e.0 += 1;
            e.1 += fwd;
        }
    }
    let mut out: HashMap<u32, (usize, usize)> = HashMap::new();
    for (&(lo, hi), &(count, fwd)) in &groups {
        // Paired iff exactly two uses, one forward one backward.
        let paired = count == 2 && fwd == 1;
        for v in [lo, hi] {
            if target_vids.contains(&v) {
                let slot = out.entry(v).or_insert((0, 0));
                slot.0 += 1;
                if !paired {
                    slot.1 += 1;
                }
            }
        }
    }
    out
}

/// Grow-until-clean experiment for one initially-unbalanced component. Repeatedly
/// absorbs the background cells at each dangling rim end into `C` and re-extracts,
/// reporting whether the boundary converges to clean oriented loops and at what
/// component size. (Measurement only.)
#[allow(clippy::too_many_arguments)]
fn boundary_grow_probe(
    points: &[Vec3],
    grid: &CubeMapGrid,
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
    gset0: &HashSet<u32>,
    mode: boundary::CollectMode,
) {
    let max_iters = env_usize("S2_BOUNDARY_GROW_ITERS", 64);
    let max_cells = env_usize("S2_BOUNDARY_GROW_MAX", 4096);
    let mut gset = gset0.clone();
    let start = gset.len();

    for iter in 1..=max_iters {
        let ring = gather_ring(&gset, points, grid, cells, cell_indices, vertex_keys);
        let diag =
            match boundary::diagnose_boundary(&gset, &ring, cells, cell_indices, vertex_keys, mode)
            {
                Ok(d) => d,
                Err(e) => {
                    eprintln!(
                        "[grow]   iter={iter} diagnose error {e:?} (size={})",
                        gset.len()
                    );
                    return;
                }
            };
        if diag.imbalances.is_empty() {
            // Confirm a full extraction succeeds.
            let ok = boundary::extract_boundary(
                &gset,
                &ring,
                cells,
                cell_indices,
                vertices,
                vertex_keys,
                mode,
            )
            .is_ok();
            eprintln!(
                "[grow]   CONVERGED iter={iter} start={start} -> size={} (extract_ok={ok})",
                gset.len()
            );
            return;
        }
        // Absorb the background generators at every dangling rim end.
        let mut added = 0usize;
        for im in &diag.imbalances {
            for &x in &im.key {
                if !gset.contains(&x) && (x as usize) < points.len() && gset.insert(x) {
                    added += 1;
                }
            }
        }
        // If key-based absorption stalls (the imbalanced vids are interior splits
        // with all-C keys — a degenerate rim vertex), broaden along the OTHER
        // axis: absorb every ring cell incident to an imbalanced vid (the cells
        // that own the split's clean cut edges), pushing the boundary outward past
        // the split.
        if added == 0 {
            let bad: HashSet<u32> = diag.imbalances.iter().map(|im| im.vid).collect();
            for &h in &ring {
                if gset.contains(&h) {
                    continue;
                }
                if let Some(span) = cell_span(cells, cell_indices, h) {
                    if span.iter().any(|v| bad.contains(v)) && gset.insert(h) {
                        added += 1;
                    }
                }
            }
        }
        if gset.len() > max_cells {
            eprintln!(
                "[grow]   DIVERGED iter={iter} start={start} size={} imbalanced={} (cap {max_cells})",
                gset.len(),
                diag.imbalances.len()
            );
            return;
        }
        if added == 0 {
            eprintln!(
                "[grow]   STUCK iter={iter} start={start} size={} imbalanced={} (no new cells absorbed)",
                gset.len(),
                diag.imbalances.len()
            );
            return;
        }
    }
    eprintln!(
        "[grow]   MAXITER start={start} size={} (still unbalanced after {max_iters})",
        gset.len()
    );
}

/// Boundary-extractor probe (dev-only, `S2_BOUNDARY_PROBE`). Runs the Tier-2
/// patch synthesizer's boundary extractor (`boundary::extract_boundary`) on every
/// real contested component WITHOUT filling, to measure which boundary topology
/// actually occurs on the mega seeds: how many components extract into clean
/// oriented loops, how many are multi-loop / pinched, the loop-size distribution
/// (sizes the fill), and which `BoundaryError` fires when extraction fails. This
/// is the measurement that tells us whether the fill must handle the hard cases
/// Codex flagged (pinch / multi-loop / holes) or whether the floor case covers
/// the real data.
fn boundary_probe_components(
    points: &[Vec3],
    grid: &CubeMapGrid,
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
    comps: &[Component],
) {
    let (mut ok, mut multiloop) = (0usize, 0usize);
    let mut loop_sizes: Vec<usize> = Vec::new();
    // Error tally by variant: [unbalanced, duplicate, missing_pos, missing_key, open_walk].
    let mut errs = [0usize; 5];
    let mut detail_budget = env_usize("S2_BOUNDARY_DETAIL", 3);
    // Collection mode: edge-pairing (default, robust) vs key-domain (legacy).
    let mode = if std::env::var("S2_BOUNDARY_KEYS").is_ok() {
        boundary::CollectMode::Keys
    } else {
        boundary::CollectMode::Paired
    };
    eprintln!("[boundary] collect_mode={mode:?}");

    for comp in comps {
        let gset: HashSet<u32> = comp.cells.iter().copied().collect();

        // Ring candidates: grid neighborhood of every component generator plus
        // every outside generator named in a component cell's vertex keys. A
        // superset is safe (the extractor ignores non-adjacent cells); this
        // mirrors the gathering in `resolve_component_attempt`.
        let mut ring: HashSet<u32> = HashSet::new();
        for &g in &comp.cells {
            if (g as usize) >= points.len() {
                continue;
            }
            let cell = grid.point_index_to_cell(g as usize);
            for &nc in grid.cell_neighbors(cell).iter() {
                if nc != u32::MAX {
                    ring.extend(grid.cell_points(nc as usize).iter().copied());
                }
            }
            for &nc in grid.cell_ring2(cell) {
                if nc != u32::MAX {
                    ring.extend(grid.cell_points(nc as usize).iter().copied());
                }
            }
            if let Some(span) = cell_span(cells, cell_indices, g) {
                for &vid in span {
                    if let Some(k) = vertex_keys.get(vid) {
                        ring.extend(k.iter().copied());
                    }
                }
            }
        }
        ring.retain(|h| !gset.contains(h));

        match boundary::extract_boundary(
            &gset,
            &ring,
            cells,
            cell_indices,
            vertices,
            vertex_keys,
            mode,
        ) {
            Ok(b) => {
                ok += 1;
                if b.loops.len() > 1 {
                    multiloop += 1;
                }
                loop_sizes.extend(b.loops.iter().map(|l| l.len()));
            }
            Err(e) => {
                let i = match e {
                    boundary::BoundaryError::UnbalancedVertex { .. } => 0,
                    boundary::BoundaryError::DuplicateDirectedEdge { .. } => 1,
                    boundary::BoundaryError::MissingPosition { .. } => 2,
                    boundary::BoundaryError::MissingKey { .. } => 3,
                    boundary::BoundaryError::OpenWalk { .. } => 4,
                };
                errs[i] += 1;

                // Detailed dump for the first few unbalanced components: WHY is
                // the boundary not a clean manifold? Cross-check each imbalanced
                // vid against the component's own unpaired edges and classify its
                // key as interior-to-C (all 3 gens in C) vs boundary (mixed).
                if matches!(e, boundary::BoundaryError::UnbalancedVertex { .. })
                    && detail_budget > 0
                {
                    detail_budget -= 1;
                    let resid_vids: HashSet<u32> = comp
                        .edges
                        .iter()
                        .flat_map(|&(va, vb, _)| [va, vb])
                        .collect();
                    match boundary::diagnose_boundary(
                        &gset,
                        &ring,
                        cells,
                        cell_indices,
                        vertex_keys,
                        mode,
                    ) {
                        Ok(d) => {
                            let (mut interior, mut boundary_kind, mut on_resid) = (0, 0, 0);
                            for im in &d.imbalances {
                                let nc = im.key.iter().filter(|x| gset.contains(x)).count();
                                if nc == 3 {
                                    interior += 1;
                                } else {
                                    boundary_kind += 1;
                                }
                                if resid_vids.contains(&im.vid) {
                                    on_resid += 1;
                                }
                            }
                            eprintln!(
                                "[boundary]   DETAIL comp cells={} ring={} edges={} verts={} \
                                 imbalanced={} (interior_key={interior} boundary_key={boundary_kind} \
                                 on_residual_edge={on_resid})",
                                comp.cells.len(),
                                ring.len(),
                                d.num_edges,
                                d.num_verts,
                                d.imbalances.len(),
                            );
                            // HYPOTHESIS TEST (user): does each boundary imbalance
                            // coincide with a genuine (Tier-1-undetected) edge
                            // mismatch? Local edge-pairing over C ∪ ring at the
                            // imbalanced vids.
                            let mut cellset = gset.clone();
                            cellset.extend(ring.iter().copied());
                            let target: HashSet<u32> =
                                d.imbalances.iter().map(|im| im.vid).collect();
                            let unpaired =
                                local_unpaired_incident(&target, &cellset, cells, cell_indices);
                            let on_unpaired = d
                                .imbalances
                                .iter()
                                .filter(|im| unpaired.get(&im.vid).is_some_and(|&(_, u)| u > 0))
                                .count();
                            eprintln!(
                                "[boundary]   HYP imbalanced={} on_unpaired_edge={on_unpaired} \
                                 on_residual={on_resid} (if on_unpaired==imbalanced and on_residual==0 \
                                 => detection gap, not a new repair mechanism)",
                                d.imbalances.len(),
                            );
                            for im in d.imbalances.iter().take(6) {
                                let nc = im.key.iter().filter(|x| gset.contains(x)).count();
                                // For each non-C generator in the key, is it in the
                                // gathered ring set? (distinguishes a gathering gap
                                // from genuine rim non-manifoldness).
                                let bg: Vec<(u32, bool)> = im
                                    .key
                                    .iter()
                                    .filter(|x| !gset.contains(x))
                                    .map(|&x| (x, ring.contains(&x)))
                                    .collect();
                                let (inc, unp) = unpaired.get(&im.vid).copied().unwrap_or((0, 0));
                                eprintln!(
                                    "[boundary]     vid={} key={:?} in={} out={} (gens_in_C={nc}, on_resid={}) \
                                     bg_in_ring={:?} local_edges={inc} local_unpaired={unp}",
                                    im.vid,
                                    im.key,
                                    im.in_deg,
                                    im.out_deg,
                                    resid_vids.contains(&im.vid),
                                    bg,
                                );
                            }
                        }
                        Err(de) => eprintln!("[boundary]   DETAIL diagnose failed: {de:?}"),
                    }

                    // Grow-until-clean experiment: absorb the background cells at
                    // each dangling rim end into C, re-extract, repeat. Measures
                    // whether "expand the component past the non-manifold rim"
                    // converges to a clean boundary at a bounded size (the
                    // principled alternative to bilaterally healing the rim, which
                    // would touch ring cells and reopen the firewall).
                    boundary_grow_probe(
                        points,
                        grid,
                        vertices,
                        cells,
                        cell_indices,
                        vertex_keys,
                        &gset,
                        mode,
                    );
                }
            }
        }
    }

    loop_sizes.sort_unstable();
    let (min, max) = (
        loop_sizes.first().copied().unwrap_or(0),
        loop_sizes.last().copied().unwrap_or(0),
    );
    eprintln!(
        "[boundary] comps={} extracted_ok={ok} multiloop={multiloop} loops={} \
         loop_size[min={min} max={max}] err[unbalanced={} dup={} miss_pos={} miss_key={} open={}]",
        comps.len(),
        loop_sizes.len(),
        errs[0],
        errs[1],
        errs[2],
        errs[3],
        errs[4],
    );
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

    // Residual components can touch at an already-paired high-degree vertex
    // without that vertex itself producing a residual edge. Re-clipping those
    // components separately makes the first component's edits part of the
    // second component's "fixed" boundary. Merge any residual-named cells that
    // co-occur in the current vertex keys so the whole local corner is resolved
    // in one shared pass.
    let mut named: Vec<u32> = Vec::new();
    for &(va, vb, owner) in &raw {
        named.push(owner);
        for v in [va, vb] {
            if let Some(k) = vertex_keys.get(v) {
                named.extend_from_slice(&k);
            }
        }
    }
    named.sort_unstable();
    named.dedup();
    let named_set: HashSet<u32> = named.iter().copied().collect();
    for &g in &named {
        let Some(span) = cell_span(cells, cell_indices, g) else {
            continue;
        };
        for &vid in span {
            if let Some(k) = vertex_keys.get(vid) {
                for &x in &k {
                    if named_set.contains(&x) {
                        uf.union(g, x);
                    }
                }
            }
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

#[inline]
fn key3(a: u32, b: u32, c: u32) -> [u32; 3] {
    let mut k = [a, b, c];
    k.sort_unstable();
    k
}

/// Deterministic pseudo-random value in [-1, 1] keyed by `(gen, axis)`. Used to
/// jitter projected generators so no two are coincident/cocircular; stable per
/// generator id so every component sees the same perturbation.
#[inline]
fn jitter_unit(gen: u32, axis: u32) -> f64 {
    // SplitMix64-style avalanche on a mixed key.
    let mut z = (gen as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((axis as u64) << 1 | 1);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Map to [-1, 1).
    (z as f64 / u64::MAX as f64) * 2.0 - 1.0
}

fn build_cycle_from_edges(
    g: u32,
    gpos: DVec3,
    verts: &[VertexKey3],
    edges: &[KeyEdge],
    pos_of: impl Fn(VertexKey3) -> Option<DVec3>,
) -> Option<Vec<VertexKey3>> {
    let n = verts.len();
    if n < 3 {
        return None;
    }
    let index: HashMap<[u32; 3], usize> = verts.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let mut adj: Vec<[usize; 2]> = vec![[usize::MAX; 2]; n];
    let mut deg = vec![0usize; n];
    for &(ka, kb) in edges {
        let (Some(&a), Some(&b)) = (index.get(&ka), index.get(&kb)) else {
            continue;
        };
        if a == b {
            continue;
        }
        if deg[a] == 2 || deg[b] == 2 {
            if trace() {
                eprintln!("[reclip]       edge-cycle fail cell {g}: vertex degree overflow");
            }
            return None;
        }
        adj[a][deg[a]] = b;
        deg[a] += 1;
        adj[b][deg[b]] = a;
        deg[b] += 1;
    }
    if deg.iter().any(|&d| d != 2) {
        if trace() {
            let bad = deg.iter().filter(|&&d| d != 2).count();
            eprintln!("[reclip]       edge-cycle fail cell {g}: {bad} non-2-degree vertices");
            for (i, &d) in deg.iter().enumerate() {
                if d != 2 {
                    eprintln!("[reclip]         degree {d}: {:?}", verts[i]);
                }
            }
        }
        return None;
    }

    let mut order = Vec::with_capacity(n);
    let (mut prev, mut cur) = (usize::MAX, 0usize);
    for _ in 0..n {
        order.push(cur);
        let nxt = if adj[cur][0] != prev {
            adj[cur][0]
        } else {
            adj[cur][1]
        };
        prev = cur;
        cur = nxt;
    }
    if cur != 0 || order.len() != n {
        if trace() {
            eprintln!("[reclip]       edge-cycle fail cell {g}: disjoint cycle");
        }
        return None;
    }
    let mut poly: Vec<[u32; 3]> = order.into_iter().map(|i| verts[i]).collect();
    let mut signed = 0.0f64;
    for w in 0..poly.len() {
        let a = pos_of(poly[w])?;
        let b = pos_of(poly[(w + 1) % poly.len()])?;
        signed += a.cross(b).dot(gpos);
    }
    if signed < 0.0 {
        poly[1..].reverse();
    }
    Some(poly)
}

#[inline]
pub(crate) fn cell_span<'a>(
    cells: &[VoronoiCell],
    cell_indices: &'a [u32],
    g: u32,
) -> Option<&'a [u32]> {
    let c = cells.get(g as usize)?;
    let start = c.vertex_start();
    let end = start + c.vertex_count();
    cell_indices.get(start..end)
}

pub(crate) fn key_common_pair(a: VertexKey3, b: VertexKey3) -> Option<(u32, u32)> {
    let mut common = [0u32; 3];
    let mut n = 0usize;
    for x in a {
        if b.contains(&x) {
            common[n] = x;
            n += 1;
        }
    }
    (n == 2).then(|| (common[0].min(common[1]), common[0].max(common[1])))
}

/// Per-cell re-resolved polygons (ordered vertex keys), the positions of the new
/// interior vertices, and the pinned vid for each boundary key (recovered from
/// the valid outside cells, so a contested cell's own dropped boundary edges are
/// restored).
struct Resolved {
    polys: Vec<(u32, Vec<VertexKey3>)>,
    interior_pos: HashMap<VertexKey3, Vec3>,
    boundary_pin: HashMap<VertexKey3, u32>,
}

enum ResolveAttempt {
    Done(Resolved),
    Expand(Vec<u32>),
}

/// Re-resolve one component into consistent per-cell key polygons by finding
/// the local Voronoi vertices (empty-circumcircle triples) and assembling them
/// with explicit pinned-boundary/component edges. `None` to bail (component
/// left as residual): horizon/degeneracy, or a cell that does not form a clean
/// cycle.
fn resolve_component_attempt(
    gvec: &[u32],
    points: &[Vec3],
    grid: &CubeMapGrid,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
) -> Option<ResolveAttempt> {
    // Component generators come from vertex keys. In production those are always
    // real ids (`g < points.len() == cells.len() == grid point count`), but
    // synthetic/fixture keys can name out-of-range generators. Guard the
    // unchecked indexing below (`points[g]`, `grid.point_index_to_cell(g)`,
    // `cells[g]`) by bailing to residual rather than panicking.
    if gvec
        .iter()
        .any(|&g| (g as usize) >= points.len() || (g as usize) >= cells.len())
    {
        return None;
    }
    let gset: HashSet<u32> = gvec.iter().copied().collect();
    // Jittered unit positions: a deterministic per-generator perturbation (~1e-9,
    // keyed by global id) so no triple is exactly cocircular/coincident. This
    // makes the degenerate-split choice jitter-determined — but consistently, as
    // the jitter is shared across cells — and keeps positions within ~1e-9 of
    // true (well inside the near-Voronoi tolerance). All geometry uses jittered
    // positions so the emptiness decision and the output vertex agree.
    let gjit = |g: u32| -> DVec3 {
        let p = points[g as usize];
        let base = DVec3::new(p.x as f64, p.y as f64, p.z as f64);
        const J: f64 = 1e-9;
        let jit = DVec3::new(jitter_unit(g, 0), jitter_unit(g, 1), jitter_unit(g, 2)) * J;
        (base + jit).normalize()
    };

    // Filter set: the COMPLETE local neighborhood of every component generator,
    // read from the spatial grid (its cell + the 3x3 + ring-2 cells). An interior
    // vertex is real iff it is closer to its three generators than to any of
    // these. The grid is in the same effective index space as `points`/keys, so
    // its point ids are valid generator ids. Using the grid (not the existing
    // cell keys) is essential: the keys come from the buggy fallback output that
    // dropped neighbors, so a key-derived filter is incomplete and lets spurious
    // triples pass.
    let mut filter: Vec<(u32, DVec3)> = Vec::new();
    let mut filter_seen: HashSet<u32> = HashSet::new();
    let mut add_filter = |x: u32, filter: &mut Vec<(u32, DVec3)>| {
        if (x as usize) < points.len() && filter_seen.insert(x) {
            filter.push((x, gjit(x)));
        }
    };
    for &g in gvec {
        let cell = grid.point_index_to_cell(g as usize);
        for &nc in grid.cell_neighbors(cell).iter() {
            for &pt in grid.cell_points(nc as usize) {
                add_filter(pt, &mut filter);
            }
        }
        for &nc in grid.cell_ring2(cell) {
            for &pt in grid.cell_points(nc as usize) {
                add_filter(pt, &mut filter);
            }
        }
    }

    // Each component cell's boundary edges, recovered from unchanged outside
    // cells and pinned to those cells' existing vids. The component cells' own
    // polygons are exactly the suspect data here: fallback/gnomonic disagreement
    // can drop or over-emit boundary vertices, so using them as constraints can
    // make an otherwise local re-clip impossible to cycle.
    let mut boundary_of: HashMap<u32, Vec<VertexKey3>> = HashMap::new();
    let mut boundary_edges: HashMap<u32, Vec<KeyEdge>> = HashMap::new();
    let mut boundary_pin: HashMap<VertexKey3, u32> = HashMap::new();
    let mut outside_candidates: HashSet<u32> = filter
        .iter()
        .map(|&(g, _)| g)
        .filter(|g| !gset.contains(g))
        .collect();
    for &g in gvec {
        for &vid in cell_span(cells, cell_indices, g)? {
            if let Some(k) = vertex_keys.get(vid) {
                for &x in &k {
                    if !gset.contains(&x) {
                        outside_candidates.insert(x);
                    }
                }
            }
        }
    }
    for h in outside_candidates {
        if (h as usize) >= cells.len() {
            continue;
        }
        let Some(span) = cell_span(cells, cell_indices, h) else {
            continue;
        };
        let n = span.len();
        for i in 0..n {
            let va = span[i];
            let vb = span[(i + 1) % n];
            let Some(ka) = vertex_keys.get(va) else {
                continue;
            };
            let Some(kb) = vertex_keys.get(vb) else {
                continue;
            };
            let Some((a, b)) = key_common_pair(ka, kb) else {
                continue;
            };
            let g = if a == h && gset.contains(&b) {
                b
            } else if b == h && gset.contains(&a) {
                a
            } else {
                continue;
            };
            // The outside cell traverses this edge in its own orientation; the
            // component cell will be wound later, so store it as an undirected
            // adjacency and pin both endpoint keys to the outside vids.
            boundary_pin.entry(ka).or_insert(va);
            boundary_pin.entry(kb).or_insert(vb);
            boundary_of.entry(g).or_default().extend([ka, kb]);
            boundary_edges.entry(g).or_default().push((ka, kb));
        }
    }
    for keys in boundary_of.values_mut() {
        keys.sort_unstable();
        keys.dedup();
    }
    for edges in boundary_edges.values_mut() {
        for (a, b) in edges.iter_mut() {
            if *b < *a {
                std::mem::swap(a, b);
            }
        }
        edges.sort_unstable();
        edges.dedup();
    }

    // Interior Voronoi vertices: every all-G triple whose circumcircle is empty
    // of all filter generators. Brute force over triples — no triangulation, so
    // no generator can be orphaned — and jitter removes the cocircular
    // over-counting that made the per-cell emptiness inconsistent. The triple set
    // is a single deterministic computation, so every component cell inherits the
    // same vertices (consistent by construction).
    const EMPTY_TOL: f64 = 1e-12; // f64 noise floor; jitter (1e-9) dominates ties.
    let gp: Vec<DVec3> = gvec.iter().map(|&g| gjit(g)).collect();
    let mut interior_pos: HashMap<VertexKey3, Vec3> = HashMap::new();
    for i in 0..gvec.len() {
        for j in (i + 1)..gvec.len() {
            for k in (j + 1)..gvec.len() {
                let Some(cc) = delaunay::circumcenter(gp[i], gp[j], gp[k]) else {
                    continue; // collinear triple: not a vertex
                };
                let g3 = [gvec[i], gvec[j], gvec[k]];
                let radius = cc.dot(gp[i]);
                let empty = filter
                    .iter()
                    .all(|&(d, dp)| g3.contains(&d) || cc.dot(dp) <= radius + EMPTY_TOL);
                if !empty {
                    continue;
                }
                let key = key3(g3[0], g3[1], g3[2]);
                interior_pos.insert(key, Vec3::new(cc.x as f32, cc.y as f32, cc.z as f32));
            }
        }
    }
    if trace() {
        eprintln!(
            "[reclip]   diag: {} cells {:?}, {} filter gens, {} interior vertices",
            gvec.len(),
            gvec,
            filter.len(),
            interior_pos.len()
        );
    }

    // A free all-component triple is usable only if every component-component
    // edge it implies has a second endpoint, either another interior triple or
    // a pinned boundary vertex. Otherwise polygon assembly must invent an edge
    // to an unrelated boundary key, which re-detect correctly reports as a
    // one-sided edge. Prune those unsupported local-hull triples to keep the
    // component snapped to its valid outside boundary.
    loop {
        let mut pair_count: HashMap<(u32, u32), u32> = HashMap::new();
        let mut seen_boundary: HashSet<VertexKey3> = HashSet::new();
        for keys in boundary_of.values() {
            for &key in keys {
                if seen_boundary.insert(key) {
                    let members: Vec<u32> = key.into_iter().filter(|g| gset.contains(g)).collect();
                    for i in 0..members.len() {
                        for j in (i + 1)..members.len() {
                            let a = members[i].min(members[j]);
                            let b = members[i].max(members[j]);
                            *pair_count.entry((a, b)).or_default() += 1;
                        }
                    }
                }
            }
        }
        for &key in interior_pos.keys() {
            for i in 0..3 {
                for j in (i + 1)..3 {
                    let a = key[i].min(key[j]);
                    let b = key[i].max(key[j]);
                    *pair_count.entry((a, b)).or_default() += 1;
                }
            }
        }

        let before = interior_pos.len();
        interior_pos.retain(|key, _| {
            (0..3).all(|i| {
                ((i + 1)..3).all(|j| {
                    let a = key[i].min(key[j]);
                    let b = key[i].max(key[j]);
                    pair_count.get(&(a, b)).copied().unwrap_or(0) >= 2
                })
            })
        });
        if interior_pos.len() == before {
            break;
        }
        if trace() {
            eprintln!(
                "[reclip]   pruned {} unsupported interior vertices",
                before - interior_pos.len()
            );
        }
    }
    let mut incident_interior: HashMap<u32, Vec<VertexKey3>> = HashMap::new();
    for &key in interior_pos.keys() {
        for &g in &key {
            incident_interior.entry(g).or_default().push(key);
        }
    }

    let mut cell_edges = boundary_edges;
    let mut pair_keys: HashMap<(u32, u32), Vec<VertexKey3>> = HashMap::new();
    let mut all_keys: HashSet<VertexKey3> = HashSet::new();
    for keys in boundary_of.values() {
        all_keys.extend(keys.iter().copied());
    }
    all_keys.extend(interior_pos.keys().copied());
    for key in all_keys {
        let members: Vec<u32> = key.into_iter().filter(|g| gset.contains(g)).collect();
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let a = members[i].min(members[j]);
                let b = members[i].max(members[j]);
                pair_keys.entry((a, b)).or_default().push(key);
            }
        }
    }
    for ((a, b), mut keys) in pair_keys {
        keys.sort_unstable();
        keys.dedup();
        if keys.len() == 2 {
            cell_edges.entry(a).or_default().push((keys[0], keys[1]));
            cell_edges.entry(b).or_default().push((keys[0], keys[1]));
        } else if trace() && keys.len() > 2 {
            eprintln!(
                "[reclip]       pair ({a},{b}) has {} candidate endpoints",
                keys.len()
            );
        }
    }
    for edges in cell_edges.values_mut() {
        for (a, b) in edges.iter_mut() {
            if *b < *a {
                std::mem::swap(a, b);
            }
        }
        edges.sort_unstable();
        edges.dedup();
    }

    // Build each cell's polygon from explicit edges: pinned outside-boundary
    // edges plus component-component edges with exactly two endpoints. This
    // avoids inferring adjacency from one-off high-degree vertex keys.
    let mut expand: Vec<u32> = Vec::new();
    for &g in gvec {
        let Some(edges) = cell_edges.get(&g) else {
            continue;
        };
        let mut verts: Vec<VertexKey3> = Vec::new();
        for &(a, b) in edges {
            verts.extend([a, b]);
        }
        verts.sort_unstable();
        verts.dedup();
        let index: HashMap<VertexKey3, usize> =
            verts.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        let mut deg = vec![0usize; verts.len()];
        for &(ka, kb) in edges {
            let (Some(&a), Some(&b)) = (index.get(&ka), index.get(&kb)) else {
                continue;
            };
            if a != b {
                deg[a] += 1;
                deg[b] += 1;
            }
        }
        for (i, &d) in deg.iter().enumerate() {
            if d == 1 {
                expand.extend(verts[i].into_iter().filter(|x| !gset.contains(x)));
            }
        }
    }
    expand.sort_unstable();
    expand.dedup();
    if !expand.is_empty() {
        if trace() {
            eprintln!("[reclip]   expanding component by {:?}", expand);
        }
        return Some(ResolveAttempt::Expand(expand));
    }

    let mut polys: Vec<(u32, Vec<VertexKey3>)> = Vec::with_capacity(gvec.len());
    for &g in gvec {
        let mut verts: Vec<VertexKey3> = Vec::new();
        let Some(edges) = cell_edges.get(&g) else {
            if trace() {
                eprintln!("[reclip]     bail cell {g}: no recovered edges");
            }
            return None;
        };
        for &(a, b) in edges {
            verts.extend([a, b]);
        }
        verts.sort_unstable();
        verts.dedup();
        if trace() {
            let boundary_count = boundary_of.get(&g).map_or(0, Vec::len);
            let interior_count = incident_interior.get(&g).map_or(0, Vec::len);
            eprintln!(
                "[reclip]     cell {g}: boundary={} interior={} unique={}",
                boundary_count,
                interior_count,
                verts.len()
            );
        }
        let gc = gjit(g);
        let pos_of =
            |key: [u32; 3]| delaunay::circumcenter(gjit(key[0]), gjit(key[1]), gjit(key[2]));
        let Some(poly) = build_cycle_from_edges(g, gc, &verts, edges, pos_of) else {
            if trace() {
                eprintln!(
                    "[reclip]     bail cell {g}: {} vertices do not form a clean cycle",
                    verts.len()
                );
                for key in &verts {
                    eprintln!("[reclip]       key {key:?}");
                }
            }
            return None;
        };
        polys.push((g, poly));
    }

    Some(ResolveAttempt::Done(Resolved {
        polys,
        interior_pos,
        boundary_pin,
    }))
}

fn resolve_component(
    comp: &Component,
    points: &[Vec3],
    grid: &CubeMapGrid,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
) -> Option<Resolved> {
    let mut gvec = comp.cells.clone();
    for _ in 0..8 {
        match resolve_component_attempt(&gvec, points, grid, cells, cell_indices, vertex_keys)? {
            ResolveAttempt::Done(resolved) => return Some(resolved),
            ResolveAttempt::Expand(add) => {
                if gvec.len() + add.len() > MAX_COMPONENT_CELLS {
                    return None;
                }
                gvec.extend(add);
                gvec.sort_unstable();
                gvec.dedup();
            }
        }
    }
    None
}

/// Whether the experimental hull + snap-to-loop resolver is selected.
fn hull_mode() -> bool {
    std::env::var("S2_RECLIP_HULL").is_ok()
}

/// Outcome of one component under the hull-snap resolver (for instrumentation).
#[derive(Default)]
struct HullStats {
    bail_fan: usize,
    bail_snap: usize,
    bail_hull: usize,
    resolved: usize,
}

/// Experimental Tier-2 resolver: rebuild the contested component `C` from the
/// EXACT local Delaunay (`local_hull`, exact `orient3d`) and attach it to the
/// surrounding diagram by SNAPPING each seam vertex to the nearest existing
/// loop vertex — bypassing the pin-by-key match that bails when the exact hull
/// picks a different third generator at a near-degenerate seam.
///
/// Snap is not strictly safe: merging two near-coincident loop vertices into
/// one can leave a ring cell below 3 vertices (a digon), and keeping both can
/// leave a sub-epsilon edge. Both are caught by the existing validate-or-revert
/// gate (never silent-invalid); this resolver only changes how `Resolved` is
/// produced. See `docs/canonical-certification-design-2026-06.md`.
fn resolve_component_hull_snap(
    comp: &Component,
    points: &[Vec3],
    grid: &CubeMapGrid,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertices: &[Vec3],
    stats: &mut HullStats,
) -> Option<Resolved> {
    const SECURED_K: usize = 256;
    const MAX_LOCAL_SET: usize = 8192;
    let cset: HashSet<u32> = comp.cells.iter().copied().collect();

    // Local point set S = C ∪ secured(C). The dense cap's 3x3 + ring-2 window
    // holds far more than the cell needs, so take the nearest-K per generator
    // (the certificate closes well within K here) and union. A whole-window
    // union would blow past MAX_LOCAL_SET on mega.
    let mut sset = cset.clone();
    for &g in &comp.cells {
        if (g as usize) >= points.len() {
            return None;
        }
        let gp = points[g as usize];
        let cell = grid.point_index_to_cell(g as usize);
        let mut cand: HashSet<u32> = HashSet::new();
        for &nc in grid.cell_neighbors(cell).iter() {
            if nc != u32::MAX {
                cand.extend(grid.cell_points(nc as usize).iter().copied());
            }
        }
        for &nc in grid.cell_ring2(cell) {
            if nc != u32::MAX {
                cand.extend(grid.cell_points(nc as usize).iter().copied());
            }
        }
        let mut by_dist: Vec<(f32, u32)> = cand
            .into_iter()
            .filter(|&c| c != g && (c as usize) < points.len())
            .map(|c| (gp.dot(points[c as usize]), c))
            .collect();
        by_dist.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        sset.extend(by_dist.iter().take(SECURED_K).map(|&(_, id)| id));
        if sset.len() > MAX_LOCAL_SET {
            return None;
        }
    }

    let mut s_ids: Vec<u32> = sset.into_iter().collect();
    s_ids.sort_unstable();
    let s_pts: Vec<Vec3> = s_ids.iter().map(|&id| points[id as usize]).collect();
    let local_of: HashMap<u32, usize> = s_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    let Some(hull) = crate::knn_clipping::local_hull::LocalHull::build(&s_pts) else {
        stats.bail_hull += 1;
        return None;
    };

    let mut interior_pos: HashMap<VertexKey3, Vec3> = HashMap::new();
    let mut boundary_pin: HashMap<VertexKey3, u32> = HashMap::new();
    let mut polys: Vec<(u32, Vec<VertexKey3>)> = Vec::with_capacity(comp.cells.len());

    for &g in &comp.cells {
        let g_local = local_of[&g];
        let fan = hull.cell_faces(g_local);
        if fan.is_empty() {
            stats.bail_fan += 1;
            return None;
        }
        let mut poly: Vec<VertexKey3> = Vec::with_capacity(fan.len());
        // Pinned vid per poly entry (Some for boundary/snapped, None for a fresh
        // interior vertex). Used to collapse consecutive snaps to the same loop
        // vid — the near-cocircular case where the exact hull splits one existing
        // seam vertex into two faces.
        let mut poly_pin: Vec<Option<u32>> = Vec::with_capacity(fan.len());
        for &fi in &fan {
            let f = hull.faces()[fi];
            let gens = [s_ids[f[0]], s_ids[f[1]], s_ids[f[2]]];
            let key = key3(gens[0], gens[1], gens[2]);
            let ncon = gens.iter().filter(|x| cset.contains(x)).count();

            if ncon == 3 {
                let cc = hull.face_circumcenter(fi);
                interior_pos
                    .entry(key)
                    .or_insert_with(|| Vec3::new(cc.x as f32, cc.y as f32, cc.z as f32));
                poly.push(key);
                poly_pin.push(None);
                continue;
            }

            // Boundary face: snap to the nearest existing vertex among the
            // involved ring cells.
            let cc = hull.face_circumcenter(fi);
            let ccf = Vec3::new(cc.x as f32, cc.y as f32, cc.z as f32);
            let mut best: Option<(f32, u32)> = None;
            for &r in &gens {
                if cset.contains(&r) || (r as usize) >= cells.len() {
                    continue;
                }
                if let Some(span) = cell_span(cells, cell_indices, r) {
                    for &vid in span {
                        if (vid as usize) < vertices.len() {
                            let d = ccf.dot(vertices[vid as usize]);
                            if best.is_none_or(|(bd, _)| d > bd) {
                                best = Some((d, vid));
                            }
                        }
                    }
                }
            }
            let Some((_, vid)) = best else {
                stats.bail_snap += 1;
                return None;
            };
            // Collapse a consecutive snap to the same loop vid (merge the split).
            if poly_pin.last() == Some(&Some(vid)) {
                continue;
            }
            boundary_pin.insert(key, vid);
            poly.push(key);
            poly_pin.push(Some(vid));
        }
        // Cyclic collapse: first and last both snap to the same loop vid.
        while poly.len() >= 2
            && poly_pin.first().copied().flatten().is_some()
            && poly_pin.first() == poly_pin.last()
        {
            poly.pop();
            poly_pin.pop();
        }
        if poly.len() < 3 {
            stats.bail_snap += 1;
            return None;
        }
        polys.push((g, poly));
    }

    stats.resolved += 1;
    Some(Resolved {
        polys,
        interior_pos,
        boundary_pin,
    })
}

/// Re-resolve contested components that survived Tier-1, returning the residual
/// still unpaired afterward (empty on full success). See
/// `docs/reclip-repair-design.md`.
// `&mut Vec` (not `&mut [_]`): re-stitch appends re-resolved cell spans to
// `cell_indices` and new interior vertices to `vertices`.
#[allow(clippy::too_many_arguments, clippy::ptr_arg)]
pub(crate) fn repair(
    points: &[Vec3],
    grid: &CubeMapGrid,
    vertices: &mut Vec<Vec3>,
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
    }

    if diag() {
        diagnose_seam(
            points,
            grid,
            vertices.as_slice(),
            cells.as_slice(),
            cell_indices.as_slice(),
            vertex_keys,
            &comps,
        );
        return Ok(residual);
    }

    if rim_probe() {
        rim_probe_components(
            points,
            grid,
            vertices.as_slice(),
            cells.as_slice(),
            cell_indices.as_slice(),
            vertex_keys,
            &comps,
        );
        return Ok(residual);
    }

    if boundary_probe() {
        boundary_probe_components(
            points,
            grid,
            vertices.as_slice(),
            cells.as_slice(),
            cell_indices.as_slice(),
            vertex_keys,
            &comps,
        );
        return Ok(residual);
    }

    let orig_len = vertices.len() as u32;
    let orig_indices_len = cell_indices.len();
    let mut interior_vid: HashMap<[u32; 3], u32> = HashMap::new();
    let mut overlay: Vec<[u32; 3]> = Vec::new();
    let mut touched: Vec<u32> = Vec::new();
    // First-overwrite `g -> old cell` snapshots. `cell_indices`/`vertices` are
    // only appended to, so restoring these cells and truncating both buffers back
    // to their original lengths fully reverts the repair if the validate-or-revert
    // gate below rejects it. Keyed by `g` and keeping the FIRST snapshot only: a
    // generator can be re-stitched by two components (component `Expand` can pull
    // a shared outside cell into both), and replaying intermediate snapshots would
    // restore `g` to a re-stitched state pointing into the now-truncated buffers
    // instead of its true pre-repair cell. A bailed component re-stitches nothing;
    // its original contested edge survives and the gate's whole-diagram validation
    // sees it, so no separate residual bookkeeping is needed here.
    let mut touched_snapshot: HashMap<u32, VoronoiCell> = HashMap::new();
    let use_hull = hull_mode();
    let mut hull_stats = HullStats::default();

    'comp: for comp in &comps {
        if comp.cells.len() > MAX_COMPONENT_CELLS {
            if trace() {
                eprintln!(
                    "[reclip]   bail: component too large ({} cells)",
                    comp.cells.len()
                );
            }
            continue;
        }
        let resolved = if use_hull {
            resolve_component_hull_snap(
                comp,
                points,
                grid,
                cells,
                cell_indices,
                vertices.as_slice(),
                &mut hull_stats,
            )
        } else {
            resolve_component(comp, points, grid, cells, cell_indices, vertex_keys)
        };
        let Some(res) = resolved else {
            if trace() {
                eprintln!(
                    "[reclip]   bail: resolve_component failed ({} cells)",
                    comp.cells.len()
                );
            }
            continue;
        };

        // Validate: every poly key is either a recomputed interior vertex or a
        // boundary key pinned from a valid outside cell. Also bound u32 index
        // capacity before appending (assembly checked pre-repair, but Tier-2
        // appends after).
        let new_interior = res
            .interior_pos
            .keys()
            .filter(|k| !interior_vid.contains_key(*k))
            .count();
        let total_poly: usize = res.polys.iter().map(|(_, p)| p.len()).sum();
        if vertices.len() + new_interior > u32::MAX as usize
            || cell_indices.len() + total_poly > u32::MAX as usize
        {
            return Err(crate::VoronoiError::ComputationFailed(
                "re-clip repair exceeded u32 index capacity".into(),
            ));
        }
        let mut unpinnable = false;
        for (_, poly) in &res.polys {
            for key in poly {
                if !res.interior_pos.contains_key(key) && !res.boundary_pin.contains_key(key) {
                    if trace() {
                        eprintln!("[reclip]   bail: unpinnable boundary key {key:?}");
                    }
                    unpinnable = true;
                }
            }
        }
        if unpinnable {
            continue 'comp;
        }

        // Assign interior vids in a deterministic key order. `res.interior_pos`
        // is a `HashMap` whose iteration order is per-process randomized; using
        // it directly would randomize the output vertex array order and each
        // re-stitched cell's VID references run-to-run (topology stays correct,
        // but the byte layout would not be reproducible). Sorting the keys first
        // makes the repaired diagram deterministic single-threaded.
        let mut interior_keys: Vec<VertexKey3> = res.interior_pos.keys().copied().collect();
        interior_keys.sort_unstable();
        for key in interior_keys {
            interior_vid.entry(key).or_insert_with(|| {
                let v = orig_len + overlay.len() as u32;
                overlay.push(key);
                vertices.push(res.interior_pos[&key]);
                v
            });
        }

        // Re-stitch: append each new polygon and repoint its cell (snapshotting
        // the old cell first so the gate below can revert).
        for (g, poly) in &res.polys {
            let start = cell_indices.len() as u32;
            for key in poly {
                let vid = interior_vid
                    .get(key)
                    .copied()
                    .or_else(|| res.boundary_pin.get(key).copied())
                    .expect("validated above");
                cell_indices.push(vid);
            }
            touched_snapshot.entry(*g).or_insert(cells[*g as usize]);
            cells[*g as usize] = VoronoiCell::new(start, poly.len() as u16);
            touched.push(*g);
        }
    }

    // Hull-snap instrumentation: categorize the predicted local failure modes
    // over the touched cells before the gate decides (digon = cell dropped below
    // 3 vertices by a merge; dup-edge = consecutive duplicate vid = a sub-eps
    // collapse). These are the snap failure modes we set out to measure.
    if use_hull {
        let mut digons = 0usize;
        let mut dup_edges = 0usize;
        for &g in &touched {
            let Some(span) = cell_span(cells, cell_indices, g) else {
                continue;
            };
            if span.len() < 3 {
                digons += 1;
            }
            let n = span.len();
            for i in 0..n {
                if n > 0 && span[i] == span[(i + 1) % n] {
                    dup_edges += 1;
                    break;
                }
            }
        }
        let gate_ok = !touched.is_empty()
            && crate::validation::verify_sphere_effective_strict(
                vertices.as_slice(),
                cells.as_slice(),
                cell_indices.as_slice(),
            )
            .is_ok();
        eprintln!(
            "[hull] comps={} resolved={} bail(hull={} fan={} snap={}) touched={} \
             digons={} dup_edges={} gate={}",
            comps.len(),
            hull_stats.resolved,
            hull_stats.bail_hull,
            hull_stats.bail_fan,
            hull_stats.bail_snap,
            touched.len(),
            digons,
            dup_edges,
            if gate_ok { "clean" } else { "revert" },
        );
    }

    // Validate-or-revert gate.
    //
    // The cheap directed-edge-pairing re-detect this pass used before was a
    // strict SUBSET of `validation::validate` — it never checked vertex degree,
    // antipodal/off-sphere vertices, duplicate cells, or Euler. A re-stitch that
    // abandons a shared boundary vertex (dropping it from degree 3 to 2) passes
    // an edge-only check yet is rejected by the validator, and with
    // `S2_VORONOI_VERIFY` off (the default) `compute` would ship it silently.
    // Instead run the FULL effective-space validator over the result and bind the
    // guarantee to the repair itself, not the env flag — the report path also
    // returns the diagram on a non-empty residual, so an env-gated check is not
    // enough.
    //
    // Whole-diagram validation also fails on any component that BAILED (its
    // original unpaired edge is still present), so a clean validation implies
    // every component resolved AND the result is strictly valid: this commit is
    // all-or-nothing. Per-component commit-or-rollback (to keep the valid
    // components when a sibling bails, recovering partial repairs on the report
    // path) is a documented follow-up — see docs/reclip-repair-design.md.
    if !touched.is_empty()
        && crate::validation::verify_sphere_effective_strict(
            vertices.as_slice(),
            cells.as_slice(),
            cell_indices.as_slice(),
        )
        .is_ok()
    {
        if trace() {
            eprintln!(
                "[reclip] re-stitched {} cell(s), {} new interior vertices; strict validation clean",
                touched.len(),
                overlay.len()
            );
        }
        return Ok(Vec::new());
    }

    // Not strictly valid (or a component bailed): revert every re-stitch so the
    // diagram returns byte-identical to its pre-repair state, and surface the
    // original contested edges as residual so the existing loud-fail backstop
    // fires. Neither the plain nor the report path can then ship invalid topology.
    for (&g, &cell) in &touched_snapshot {
        cells[g as usize] = cell;
    }
    cell_indices.truncate(orig_indices_len);
    vertices.truncate(orig_len as usize);
    // After a correct revert every restored cell points inside the pre-repair
    // index buffer. A stale re-stitched span (the duplicate-touch rollback
    // hazard the first-snapshot-only map guards against) would point past it,
    // which the report path would later read as a corrupted cell.
    debug_assert!(
        touched_snapshot.keys().all(|&g| {
            let c = cells[g as usize];
            c.vertex_start() + c.vertex_count() <= orig_indices_len
        }),
        "reverted cell span points past the pre-repair index buffer"
    );

    let mut residual_out = residual;
    residual_out.sort_unstable();
    residual_out.dedup();
    if trace() {
        eprintln!(
            "[reclip] repair not strictly valid; reverted {} re-stitch(es); {} residual edge(s) remain",
            touched_snapshot.len(),
            residual_out.len()
        );
    }
    Ok(residual_out)
}

/// Spherical circumcenter helper.
mod delaunay {
    use glam::DVec3;

    /// Unit direction equidistant (spherical) from three unit generators — the
    /// Voronoi vertex of triangle `(a,b,c)`. `None` if degenerate (collinear).
    pub(super) fn circumcenter(a: DVec3, b: DVec3, c: DVec3) -> Option<DVec3> {
        let nrm = (a - b).cross(b - c);
        let len2 = nrm.length_squared();
        if !len2.is_finite() || len2 <= 1e-30 {
            return None;
        }
        let dir = nrm / len2.sqrt();
        Some(if dir.dot(a) >= 0.0 { dir } else { -dir })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn circumcenter_is_equidistant() {
            let a = DVec3::new(0.0, 0.0, 1.0);
            let b = DVec3::new(0.1, 0.0, 1.0).normalize();
            let c = DVec3::new(0.0, 0.1, 1.0).normalize();
            let p = circumcenter(a, b, c).unwrap();
            let (da, db, dc) = (p.dot(a), p.dot(b), p.dot(c));
            assert!(
                (da - db).abs() < 1e-9 && (db - dc).abs() < 1e-9,
                "equidistant"
            );
        }
    }
}
