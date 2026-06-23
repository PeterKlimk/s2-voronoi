# s2-voronoi

Spherical Voronoi diagrams on the unit sphere, and planar Voronoi diagrams over a bounded
rectangle or torus, built by the same parallel engine.

Most spherical Voronoi code goes through a 3D convex hull (qhull, scipy) and slows down past tens
of thousands of points. This crate builds each cell independently by clipping half-spaces of
nearby points — the construction usually seen on the GPU — and stitches the per-cell results into
one shared, validated graph on the CPU. Per-point cost is near constant, so the gap to hull-based
code grows with n: roughly 4x faster than `voronoice` at 1M planar points, ~330ms for 1M sphere
points multithreaded on a Ryzen 3600.

Status: pre-release (0.1). The API is not yet stable. Stable Rust, MSRV 1.88.

## Quickstart

```rust
use s2_voronoi::{compute, UnitVec3};

let points = vec![
    UnitVec3::new(1.0, 0.0, 0.0),
    UnitVec3::new(0.0, 1.0, 0.0),
    UnitVec3::new(0.0, 0.0, 1.0),
    UnitVec3::new(-1.0, 0.0, 0.0),
    UnitVec3::new(0.0, -1.0, 0.0),
    UnitVec3::new(0.0, 0.0, -1.0),
];

let diagram = compute(&points)?;
for cell in diagram.iter_cells() {
    let generator = diagram.generator(cell.generator_index);
    let boundary = cell.vertex_indices.iter().map(|&i| diagram.vertex(i as usize));
    let _ = (generator, boundary);
}
# Ok::<(), s2_voronoi::VoronoiError>(())
```

Inputs are assumed unit-normalized. They are canonicalized once at entry (renormalized in f64,
rounded back to f32), so `generators()` may differ from the raw input by ~1 ulp.

## Planar and periodic

`compute_plane` tiles a bounded rectangle; the walls act as virtual generators, so hull cells are
clipped to the rect and cell areas partition it exactly. `compute_plane_periodic` identifies
opposite edges into a torus: cells wrap, every edge is shared by two cells, and the diagram has no
boundary.

```rust
use s2_voronoi::{compute_plane, PlaneRect};

let points = vec![[0.25f32, 0.25], [0.75, 0.25], [0.5, 0.8]];
let diagram = compute_plane(&points, PlaneRect::unit())?;
for i in 0..diagram.num_cells() {
    let boundary = diagram.cell(i).iter().map(|&v| diagram.vertex(v as usize));
    let _ = boundary;
}
# Ok::<(), s2_voronoi::VoronoiError>(())
```

## What you get

- `SphericalVoronoi` / `PlanarVoronoi`: shared vertex list, per-cell boundary indices, generators.
- `cell_area(i)`, `cell_centroid(i)` — topology-aware on the sphere, rect, and torus.
- `lloyd_step()` — one centroidal-relaxation iteration; the same loop on all three topologies.
- `build_adjacency()` — per-cell Voronoi neighbors aligned with boundary edges (the Delaunay
  edges of the generator set).
- `delaunay_triangles()` — the dual triangulation as `Vec<[u32; 3]>`.
- `build_locator()` — reusable point-location; `locate(q)` maps a point to its cell in
  near-constant time, `locate_many(&[q])` batches across cores.
- `validation::validate` / `validate_plane` — strict subdivision check.
- `weld_map()` — generators merged as coincident (see Correctness).

Configuration is through `compute_with(points, VoronoiConfig)`; `compute_with_report` additionally
returns what was welded, perturbed, or repaired. Defaults handle coincident and degenerate inputs;
see [docs/correctness.md](docs/correctness.md).

## How it works

Each cell is the intersection of half-spaces, one per neighbor, bounded by the bisector between
the generator and that neighbor. Candidates stream nearest-first from a spatial grid; the
"security radius" certificate stops the stream once no unseen point can reach the current polygon
(typically after clipping ~6-7 neighbors, independent of n). On the sphere a gnomonic projection
turns great circles into straight lines, so cell construction is 2D convex-polygon clipping.

The per-cell construction follows Ray et al., *Meshless Voronoi on the GPU* (2018). The part this
crate adds is stitching the independently-built cells into one consistent graph: vertices are
identified combinatorially (by the triple of generators that meet there, never by position) and
deduplicated shard-locally with no global lock. [docs/architecture.md](docs/architecture.md) has
the full description.

## Correctness

The output is a strictly valid subdivision — Euler characteristic holds, every edge is shared by
exactly two cells, one connected component — checked by `validation::validate` and fuzz-tested at
multi-million point counts. Geometry is accurate to floating-point precision, not exact: no f32
implementation can promise exact positions. Near-coincident generators are welded, degenerate
great-circle inputs are perturbed, and rare topology defects are repaired, all by default and all
reported. [docs/correctness.md](docs/correctness.md) states the guarantees and limits precisely.

## Performance

Multithreaded on a Ryzen 3600 (6 cores), uniform input:

| n  | sphere | plane | voronoice (plane) |
|----|--------|-------|-------------------|
| 1M | ~330ms | ~430ms | ~1.4s |
| 2M | ~720ms | ~1.0s | ~3.5s |

Single-threaded the two geometries are at parity (~1.8s at 1M). Per-build peak memory is roughly
0.65 KB/point. [docs/performance.md](docs/performance.md) covers benchmarking and reproduction.

## Features

- `parallel` (default): rayon parallelism in cell construction.
- `glam`: `UnitVec3Like` impl and conversions for `glam::Vec3`.
- `serde`: `Serialize`/`Deserialize` for the diagram types.
- `qhull`: convex-hull backend, for test/bench comparison only.
- `timing`: phase and sub-phase timing reports.

## License

MIT OR Apache-2.0
