# voronoi-mesh

Fast spherical Voronoi diagrams on the unit sphere.

Most spherical Voronoi code goes through a 3D convex hull (qhull, scipy) and slows down past tens
of thousands of points. This crate builds each cell independently by clipping half-spaces of
nearby points — the construction usually seen on the GPU — and stitches the per-cell results into
one shared, validated graph on the CPU. Per-point cost is near constant, so the gap to hull-based
code grows with n: ~330ms for 1M points multithreaded on a Ryzen 3600.

Status: pre-release (0.1). The API is not yet stable. Stable Rust, MSRV 1.88.

## Quickstart

```rust
use voronoi_mesh::{compute, UnitVec3};

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
# Ok::<(), voronoi_mesh::VoronoiError>(())
```

Inputs are assumed unit-normalized. They are canonicalized once at entry (renormalized in f64,
rounded back to f32), so `generators()` may differ from the raw input by ~1 ulp.

### Spheres in world coordinates

Keep the geometric computation in stable unit-sphere coordinates while embedding the result at
any finite center and positive radius:

```rust
use voronoi_mesh::{compute_on_sphere, SphereEmbedding};

let sphere = SphereEmbedding::new([10.0, -5.0, 2.0], 3.0)?;
let points = [
    [13.0, -5.0, 2.0],
    [7.0, -5.0, 2.0],
    [10.0, -2.0, 2.0],
    [10.0, -8.0, 2.0],
    [10.0, -5.0, 5.0],
    [10.0, -5.0, -1.0],
];
let embedded = compute_on_sphere(&points, sphere)?;
let world_vertex = embedded.vertex_world(0);
# let _ = world_vertex;
# Ok::<(), Box<dyn std::error::Error>>(())
```

World inputs use f64 and are interpreted by their direction from the center: their radial distance
is deliberately discarded. The returned wrapper stores only the canonical unit diagram plus the
embedding. World vertices, generators, spherical centroids, physical areas, Lloyd targets, and
point-location queries are derived without duplicating the topology or geometry buffers.

## What you get

- `SphericalVoronoi`: shared vertex list, per-cell boundary indices, generators.
- `cell_area(i)`, `cell_centroid(i)` — spherical areas and centroids.
- `lloyd_step()` — one centroidal-relaxation iteration.
- `build_adjacency()` — per-cell Voronoi neighbors aligned with boundary edges (the Delaunay
  edges of the generator set).
- `delaunay_triangles()` — the dual triangulation as `Vec<[u32; 3]>`.
- `build_locator()` — reusable point-location; `locate(q)` maps a point to its cell in
  near-constant time, `locate_many(&[q])` batches across cores.
- `SphereEmbedding` / `compute_on_sphere` — translated and uniformly scaled world-space spheres
  backed by the same canonical unit diagram.
- `validation::validate` — strict subdivision check.
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

The per-cell construction follows Ray et al., *Meshless Voronoi on the GPU* (2018); other recent
CPU implementations of that construction (e.g. [vortex](https://github.com/philipclaude/vortex))
emit independent per-cell polygons. The part this crate adds is stitching the independently-built
cells into one consistent graph: vertices are identified combinatorially (by the triple of
generators that meet there, never by position) and deduplicated shard-locally with no global
lock. [docs/architecture.md](docs/architecture.md) has the full description.

## Correctness

Every successfully returned diagram is a strictly valid subdivision — Euler characteristic
holds, every edge is shared by exactly two cells, and there is one connected component — checked
by `validation::validate` and fuzz-tested at multi-million point counts. Inputs outside the
supported numerical/model envelope return a defined error rather than a non-manifold diagram.
Geometry is accurate to floating-point precision, not exact: no f32 implementation can promise
exact positions. Near-coincident generators are welded, degenerate great-circle inputs are
perturbed, and rare topology defects are repaired, all by default and all reported.
[docs/correctness.md](docs/correctness.md) states the guarantees and limits precisely.

## Performance

Multithreaded on a Ryzen 3600 (6 cores), uniform input:

| n  | time |
|----|------|
| 1M | ~330ms |
| 2M | ~720ms |

Single-threaded, ~1.8s at 1M. Per-build peak memory is roughly 0.65 KB/point.
[docs/performance.md](docs/performance.md) covers benchmarking and reproduction.

## Features

- `parallel` (default): rayon parallelism in cell construction.
- `glam`: `UnitVec3Like` impl and conversions for `glam::Vec3`.
- `serde`: `Serialize`/`Deserialize` for the diagram types.
- `qhull`: convex-hull backend, for test/bench comparison only.
- `timing`: phase and sub-phase timing reports.

## License

MIT OR Apache-2.0
