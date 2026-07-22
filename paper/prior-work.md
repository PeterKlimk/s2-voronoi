# Prior work ledger

This file records evidence needed for comparison and novelty analysis. It is not finished related-
work prose.

## Ray et al. (2018)

Nicolas Ray, Dmitry Sokolov, Sylvain Lefebvre, and Bruno Lévy, "Meshless Voronoi on the GPU."

- DOI: <https://doi.org/10.1145/3272127.3275092>
- Relationship: source construction idea—nearest-first half-space clipping with a termination
  certificate, designed for independent cells and geometry queries rather than a shared mesh.
- Needed before publication: reread the algorithm and terminology closely; distinguish inherited
  construction from changes made for the sphere, CPU execution, and shared topology.

## Caplan et al. / Vortex (2025 preprint; 2026 JCP)

Philip Caplan et al., "A Lagrangian method for solving the spherical shallow water equations using
power diagrams."

- DOI: <https://doi.org/10.1016/j.jcp.2026.114833>
- Open preprint: <https://arxiv.org/abs/2508.08129>
- Code: <https://github.com/philipclaude/vortex>
- Source revision inspected: `3d59c666d69dbfb3d72513de19b8aae7ce4a57e0`.

### Confirmed overlap

- CPU-parallel spherical half-space clipping derived from the same general construction.
- Weighted spherical power diagrams, with ordinary Voronoi diagrams as the zero/equal-weight case.
- Nearest-neighbor candidates from a spherical quadtree, with kd-tree fallback.
- Per-cell radius-of-security termination.
- Optional explicit polygon/mesh output and Delaunay extraction.
- Optional vertex merging by the sorted triple of incident generator IDs:
  [`VoronoiDiagram::merge`](https://github.com/philipclaude/vortex/blob/3d59c666d69dbfb3d72513de19b8aae7ce4a57e0/src/voronoi.cpp#L955-L995).

### Confirmed distinction in the inspected implementation

- Initial neighbor lists are produced for every site, and every cell independently consumes its own
  list:
  [`VoronoiPolygon::compute`](https://github.com/philipclaude/vortex/blob/3d59c666d69dbfb3d72513de19b8aae7ce4a57e0/src/voronoi_polygon.hpp#L80-L154).
- Cells are assigned to thread blocks but remain independent during clipping:
  [`VoronoiDiagram::compute`](https://github.com/philipclaude/vortex/blob/3d59c666d69dbfb3d72513de19b8aae7ce4a57e0/src/voronoi.cpp#L700-L754).
- Duplicate interior facets are suppressed for storage by generator-index order, after the cell has
  already been constructed:
  [`VoronoiMesh::add_facet`](https://github.com/philipclaude/vortex/blob/3d59c666d69dbfb3d72513de19b8aae7ce4a57e0/src/voronoi.h#L384-L387).
- No ownership-filtered query or construction-time edge forwarding was found in these paths.

### Benchmark interpretation

The paper's large-scale "total" timing is described as neighbor calculation plus cell clipping. It
does not establish the cost of constructing, reconciling, and validating a conforming indexed mesh.
Its published hardware is also different from the local machine. Reproduce Vortex locally before
making any direct speed claim.

### Consequence for our claims

Do not claim that Vortex can only emit independent polygons: it can optionally merge them. The
potential distinction is avoiding redundant pair discovery/clipping during construction and the
stronger topology/outcome contract, not the bare existence of mesh output or triple-key merging.

## CGAL

Role: established explicit-topology comparator and an important end-to-end performance baseline.

Before publication, pin and record:

- exact CGAL version and package/API;
- construction route used for spherical Voronoi output;
- exact kernel/precision choices;
- whether the selected path is single-threaded or parallel;
- included preprocessing, conversion, topology construction, and validation work; and
- equivalence of the output contract.

An order-of-magnitude result is potentially central, but only after a same-machine, same-input,
same-output-tier reproduction.

## Literature-search leads

- CPU implementations citing Ray et al.
- Spherical Voronoi and power-diagram implementations using half-space clipping.
- Parallel Voronoi construction with half-neighbor lists or pair ownership.
- Constraint/edge forwarding between sequential cells in a parallel shard.
- Distributed or sharded conforming-mesh assembly using combinatorial vertex identities.
- Production topology repair and validation for floating-point Voronoi construction.
