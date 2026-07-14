# voronoi-mesh

Fast spherical Voronoi diagrams on the unit sphere.

Most spherical Voronoi code goes through a global 3D convex hull and slows down past tens
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
any finite center and positive radius whose axis-aligned extent remains representable in f64:

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
point-location queries are derived without duplicating the topology or geometry buffers. With the
default `parallel` feature and a multi-threaded Rayon pool, large world inputs are snapshotted once
in bounded chunks and projected across the pool before the unchanged unit-sphere backend begins.
Smaller inputs and single-threaded pools stay serial.

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
- `ComputeOutput::into_elided_cell_mesh()` — explicit cold conversion that may remove
  unrepresentable zero-geometry cells and returns a separately typed, validated spherical cell
  mesh with original-input provenance.

Configuration is through `compute_with(points, VoronoiConfig)`; `compute_with_report` additionally
returns what was welded, perturbed, or repaired. Defaults handle coincident and degenerate inputs;
`CellKillingPolicy::Error` is available when a consumer requires exact stored-zero resolution to
fail rather than preserve an unrepresentable generator cell. Consumers that instead accept removal
can call `into_elided_cell_mesh` on the successful report-bearing Preserve result. The cell mesh
does not expose locator, Delaunay, or Lloyd methods because simplification can break those Voronoi
interpretations. See
[docs/correctness.md](docs/correctness.md).

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

The fast success path certifies the properties graphical consumers need: every shared edge agrees
exactly in multiplicity and orientation, low-incidence defects are rejected or repaired, and the
spherical Euler characteristic holds. Full strict validation (including connectivity and broader
representation diagnostics) remains available through `validation::validate`, is included in
`compute_with_report`, and can gate plain `compute` with `VORONOI_MESH_VERIFY=1`; it is not imposed
as a second global edge sort on every production build. These checks are fuzz-tested at
multi-million point counts.

Geometry is accurate to floating-point working precision, not an unqualified “exact Voronoi”
claim: inputs are canonicalized, different robust/fallback policies can resolve ambiguity-scale
features, and output vertices are stored as f32. Near-coincident generators are welded,
degenerate great-circle inputs are perturbed, and rare topology defects are repaired, all by
default and all reported.
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
- `timing`: phase and sub-phase timing reports.

## Development documents

- [Active triage and work log](docs/work-log.md)
- [Roadmap](ROADMAP.md)
- [Correctness and safety audit record](docs/audit-triage.md)
- [Output-resolution policy](docs/output-resolution-policy.md)

## License

MIT OR Apache-2.0
