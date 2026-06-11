// The frontier loop has a non-trivial Exhausted arm (mirrors production
// consumers); while-let would bury it.
#![allow(clippy::while_let_loop)]
//! Builder-level planar Voronoi tests: stream-driven cells (with early
//! termination) must equal brute-force cells (clipped against *every*
//! neighbor), plus global invariants the bounded-rect model guarantees.
//!
//! Degenerate-tie inputs (exact lattices) are only checked against
//! order-robust invariants (validity, area), not vertex-key equality —
//! epsilon ties are policy territory, the same as on the sphere.

use glam::Vec2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::builder::PlaneCellBuilder;
use crate::cube_grid::DirectedEligibility;
use crate::knn_clipping::cell_build::CellFailure;
use crate::knn_clipping::cell_build::CellOutputBuffer;
use crate::packed_layout::PackedSlotLayout;
use crate::plane_grid::{PlaneGrid, PlaneNeighborFrontier, PlaneNeighborStream};

const LOCAL_SHIFT: u32 = 24;
const LOCAL_MASK: u32 = (1u32 << LOCAL_SHIFT) - 1;
/// Positions reachable via different clip orders agree to f64 intersection
/// math quantized to f32 output.
const POS_TOL: f32 = 2e-6;

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn uniform(n: usize, seed: u64) -> Vec<Vec2> {
    let mut r = rng(seed);
    (0..n)
        .map(|_| Vec2::new(r.gen_range(0.0f32..1.0), r.gen_range(0.0f32..1.0)))
        .collect()
}

/// Clip against every other generator in index order: the ground truth.
fn brute_force_cell(
    points: &[Vec2],
    gi: usize,
    wall_base: u32,
) -> Result<CellOutputBuffer<Vec2>, CellFailure> {
    let mut builder = PlaneCellBuilder::new(gi, points[gi], wall_base, Vec2::ONE);
    for (j, &p) in points.iter().enumerate() {
        if j != gi {
            builder.clip_with_slot(j, j as u32, p)?;
        }
    }
    let mut buf = CellOutputBuffer::<Vec2>::default();
    builder.to_vertex_data(&mut buf)?;
    Ok(buf)
}

/// Build via the grid stream with early termination on the certificates.
fn stream_cell(
    points: &[Vec2],
    grid: &PlaneGrid,
    slot_gen_map: &[u32],
    gi: usize,
    wall_base: u32,
) -> Result<CellOutputBuffer<Vec2>, CellFailure> {
    let layout = PackedSlotLayout::new(slot_gen_map, LOCAL_SHIFT, LOCAL_MASK);
    // query_bin 1 vs all-zero map: every cell is "other bin" => emit all.
    let ctx = DirectedEligibility::from_layout(1, 0, layout);
    let mut scratch = grid.make_scratch();
    let mut stream = PlaneNeighborStream::new(grid, points, gi, &mut scratch, ctx);
    let mut builder = PlaneCellBuilder::new(gi, points[gi], wall_base, Vec2::ONE);

    let mut batch = Vec::new();
    let mut batch_dists = Vec::new();
    loop {
        match stream.frontier(&mut batch, &mut batch_dists) {
            PlaneNeighborFrontier::ExactBatch(result) => {
                for &slot in &batch[..result.n] {
                    let nidx = grid.point_indices()[slot as usize] as usize;
                    builder.clip_with_slot(nidx, slot, points[nidx])?;
                }
                stream.advance_frontier();
                if builder.can_terminate(result.unseen_bound) {
                    break;
                }
            }
            PlaneNeighborFrontier::Exhausted => break,
        }
    }

    let mut buf = CellOutputBuffer::<Vec2>::default();
    builder.to_vertex_data(&mut buf)?;
    Ok(buf)
}

fn all_zero_slot_map(n: usize) -> Vec<u32> {
    (0..n as u32).collect() // bin 0, local = slot
}

/// Shoelace area of the (ordered) extracted polygon, accumulated in f64.
fn cell_area(buf: &CellOutputBuffer<Vec2>) -> f64 {
    let vs = &buf.vertices;
    let mut acc = 0.0f64;
    for i in 0..vs.len() {
        let (_, a) = vs[i];
        let (_, b) = vs[(i + 1) % vs.len()];
        acc += (a.x as f64) * (b.y as f64) - (b.x as f64) * (a.y as f64);
    }
    0.5 * acc
}

fn assert_cells_equal(
    name: &str,
    gi: usize,
    brute: &CellOutputBuffer<Vec2>,
    stream: &CellOutputBuffer<Vec2>,
) {
    let mut bkeys: Vec<_> = brute.vertices.iter().map(|&(k, _)| k).collect();
    let mut skeys: Vec<_> = stream.vertices.iter().map(|&(k, _)| k).collect();
    bkeys.sort_unstable();
    skeys.sort_unstable();
    assert_eq!(
        bkeys, skeys,
        "{name}: vertex key sets differ for generator {gi}"
    );

    for &(key, spos) in &stream.vertices {
        let (_, bpos) = brute
            .vertices
            .iter()
            .find(|&&(k, _)| k == key)
            .expect("key checked above");
        assert!(
            (spos.x - bpos.x).abs() <= POS_TOL && (spos.y - bpos.y).abs() <= POS_TOL,
            "{name}: vertex {key:?} of generator {gi} moved between brute ({bpos:?}) \
             and stream ({spos:?}) builds"
        );
    }
}

/// Stream-with-termination equals brute-force for every cell, and the global
/// invariants (area partition, corner coverage, edge pairing) hold.
fn check_scenario(name: &str, points: Vec<Vec2>, res: usize) {
    let n = points.len();
    let wall_base = n as u32;
    let grid = PlaneGrid::new(&points, res);
    let slot_map = all_zero_slot_map(n);

    let mut total_area = 0.0f64;
    let mut corner_counts = [0usize; 4];
    let mut edge_counts: std::collections::HashMap<(u32, u32), usize> =
        std::collections::HashMap::new();

    for gi in 0..n {
        let brute = brute_force_cell(&points, gi, wall_base)
            .unwrap_or_else(|f| panic!("{name}: brute build failed for {gi}: {f:?}"));
        let stream = stream_cell(&points, &grid, &slot_map, gi, wall_base)
            .unwrap_or_else(|f| panic!("{name}: stream build failed for {gi}: {f:?}"));
        assert_cells_equal(name, gi, &brute, &stream);

        let area = cell_area(&stream);
        assert!(
            area > 0.0,
            "{name}: non-positive cell area {area} for generator {gi}"
        );
        total_area += area;

        for &(key, _) in &stream.vertices {
            // A rect corner is [gen, wall, wall] with two adjacent walls.
            if key[1] >= wall_base && key[2] >= wall_base {
                let sides = (key[1] - wall_base, key[2] - wall_base);
                let corner = match sides {
                    (0, 3) | (3, 0) => 0, // bottom-left
                    (0, 1) | (1, 0) => 1, // bottom-right
                    (1, 2) | (2, 1) => 2, // top-right
                    (2, 3) | (3, 2) => 3, // top-left
                    other => panic!("{name}: non-adjacent wall pair {other:?} in a vertex key"),
                };
                corner_counts[corner] += 1;
            }
        }

        for &nb in &stream.edge_neighbor_globals {
            if nb < wall_base {
                let key = (gi.min(nb as usize) as u32, gi.max(nb as usize) as u32);
                *edge_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    assert!(
        (total_area - 1.0).abs() < 1e-5,
        "{name}: cell areas sum to {total_area}, expected 1.0"
    );
    for (corner, &count) in corner_counts.iter().enumerate() {
        assert_eq!(
            count, 1,
            "{name}: rect corner {corner} appears in {count} cells, expected exactly 1"
        );
    }
    for (&(i, j), &count) in &edge_counts {
        assert_eq!(
            count, 2,
            "{name}: interior edge ({i},{j}) appears {count} times, expected 2"
        );
    }
}

#[test]
fn plane_cells_uniform() {
    check_scenario("uniform_200", uniform(200, 5), 5);
    check_scenario("uniform_50_coarse", uniform(50, 11), 2);
}

#[test]
fn plane_cells_clustered() {
    let mut r = rng(17);
    let mut points: Vec<Vec2> = (0..150)
        .map(|_| {
            Vec2::new(
                (0.3 + r.gen_range(-0.02f32..0.02)).clamp(0.0, 1.0),
                (0.7 + r.gen_range(-0.02f32..0.02)).clamp(0.0, 1.0),
            )
        })
        .collect();
    points.extend([
        Vec2::new(0.05, 0.05),
        Vec2::new(0.95, 0.1),
        Vec2::new(0.9, 0.9),
    ]);
    check_scenario("clustered", points, 8);
}

#[test]
fn plane_cells_collinear() {
    // Generic positions along a diagonal: cells are perpendicular slabs
    // clipped by the rect.
    let mut r = rng(23);
    let points: Vec<Vec2> = (0..40)
        .map(|_| {
            let t = r.gen_range(0.05f32..0.95);
            Vec2::new(t, (t + r.gen_range(-1e-4f32..1e-4)).clamp(0.0, 1.0))
        })
        .collect();
    check_scenario("collinear", points, 4);
}

#[test]
fn plane_cells_tiny_inputs() {
    // n=1: the cell is the whole rect (no spherical analog!).
    let one = vec![Vec2::new(0.4, 0.6)];
    let buf = brute_force_cell(&one, 0, 1).unwrap();
    assert_eq!(buf.vertices.len(), 4);
    assert!((cell_area(&buf) - 1.0).abs() < 1e-6);
    check_scenario("n1", one, 2);

    check_scenario("n2", vec![Vec2::new(0.2, 0.3), Vec2::new(0.8, 0.6)], 2);
    check_scenario(
        "n3",
        vec![
            Vec2::new(0.1, 0.1),
            Vec2::new(0.9, 0.2),
            Vec2::new(0.5, 0.9),
        ],
        3,
    );
}

#[test]
fn plane_cells_generator_on_boundary() {
    // Generators exactly on walls and corners: the rect seed degenerates
    // gracefully (zero-distance walls), cells stay valid.
    let mut points = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 0.5),
        Vec2::new(1.0, 0.5),
        Vec2::new(0.5, 0.0),
        Vec2::new(0.5, 1.0),
    ];
    points.extend(uniform(30, 31));
    check_scenario("boundary_generators", points, 4);
}

#[test]
fn plane_cells_exact_lattice_order_robust_invariants() {
    // 4x4 exact lattice: every Voronoi vertex is a 4-cocircular tie. Vertex
    // keys are order/epsilon-dependent there (policy, as on the sphere), so
    // only check validity and the area partition.
    let mut points = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            points.push(Vec2::new((i as f32 + 0.5) / 4.0, (j as f32 + 0.5) / 4.0));
        }
    }
    let n = points.len();
    let wall_base = n as u32;
    let grid = PlaneGrid::new(&points, 4);
    let slot_map = all_zero_slot_map(n);

    let mut total_area = 0.0f64;
    for gi in 0..n {
        let buf = stream_cell(&points, &grid, &slot_map, gi, wall_base)
            .unwrap_or_else(|f| panic!("lattice: stream build failed for {gi}: {f:?}"));
        let area = cell_area(&buf);
        assert!(area > 0.0, "lattice: non-positive area for {gi}");
        total_area += area;
    }
    assert!(
        (total_area - 1.0).abs() < 1e-5,
        "lattice: areas sum to {total_area}"
    );
}

#[test]
fn plane_cells_exact_duplicate_is_upstream_policy() {
    // Bisector of a bit-identical twin is the degenerate all-zero half-plane:
    // the clip is a no-op (both twins keep overlapping full cells). The
    // pipeline's exact-duplicate policy handles this before the builder, as
    // on the sphere; here we pin the builder-level behavior.
    let p = Vec2::new(0.4, 0.4);
    let mut builder = PlaneCellBuilder::new(0, p, 2, Vec2::ONE);
    let result = builder.clip_with_slot_result(1, 1, p).unwrap();
    assert_eq!(
        result,
        crate::knn_clipping::topo2d::types::ClipResult::Unchanged
    );
    assert_eq!(builder.vertex_count(), 4);
}

#[test]
fn plane_builder_reset_reuses_cleanly() {
    let points = uniform(60, 41);
    let wall_base = points.len() as u32;
    let mut builder = PlaneCellBuilder::new(0, points[0], wall_base, Vec2::ONE);
    let mut reused = CellOutputBuffer::<Vec2>::default();
    let mut fresh_buf = CellOutputBuffer::<Vec2>::default();

    for gi in 0..points.len() {
        builder.reset(gi, points[gi]);
        for (j, &p) in points.iter().enumerate() {
            if j != gi {
                builder.clip_with_slot(j, j as u32, p).unwrap();
            }
        }
        builder.to_vertex_data(&mut reused).unwrap();

        let fresh = {
            let mut b = PlaneCellBuilder::new(gi, points[gi], wall_base, Vec2::ONE);
            for (j, &p) in points.iter().enumerate() {
                if j != gi {
                    b.clip_with_slot(j, j as u32, p).unwrap();
                }
            }
            b.to_vertex_data(&mut fresh_buf).unwrap();
            &fresh_buf
        };

        assert_eq!(
            reused.vertices, fresh.vertices,
            "reset state leaked into generator {gi}"
        );
    }
}
