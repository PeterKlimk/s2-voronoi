# AGENTS.md

This repo-root file provides guidance for coding agents working in the `voronoi-mesh` crate.

For user-facing crate docs, see `README.md` and `docs/`.

## Toolchain / Constraints

- Stable Rust (MSRV 1.88); explicit SIMD via the `wide` crate behind the `src/fp.rs` backend seam.
- Run heavy checks in release mode where possible.
- Input points are assumed to be unit-normalized.

## Build & Test

```bash
cargo test --release
cargo clippy --all-targets
cargo fmt
```

Useful targeted checks:

```bash
# Optimized codegen with internal debug assertions enabled
cargo test --profile checked

# API/correctness suites only
cargo test --release --test api --test correctness
```

## Benchmarking

```bash
# Large-scale benchmark driver
cargo run --release --features tools --bin bench_voronoi -- 100k 500k 1m

# Detailed sub-phase timing
VORONOI_MESH_TIMING_KV=1 cargo run --release --features tools,timing --bin bench_voronoi -- 500k --no-preprocess

# Inter-commit perf comparisons
./scripts/bench_build.sh --chain 6
./scripts/bench_run.sh -s 500k -r 20 -m total
```

## Common Environment Knobs

- `RAYON_NUM_THREADS=1`: force single-threaded mode (stable perf comparisons).
- `VORONOI_MESH_BIN_COUNT=<n>`: override the sharded bin target (defaults to 2x threads below
  12 workers and a 96-bin coarse layout at 12+; severely imbalanced default layouts may refine
  to 216 bins).
- `VORONOI_MESH_TIMING_KV=1`: emit machine-readable timing lines (`timing` feature).

The authoritative inventory, including diagnostic/oracle knobs and mutation policy, is
`docs/environment-knobs.md`.

## Crate Overview

`voronoi-mesh` computes spherical Voronoi diagrams on the unit sphere (S2) using kNN-driven half-space clipping.

High-level flow per generator:

1. Find candidate neighbors via cube-map spatial index.
2. Build bisector great-circle constraints.
3. Clip cell in local gnomonic/topological 2D representation.
4. Deduplicate/assemble shared vertices across cells.

## Documentation Map

- `README.md`: user-facing overview and API summary.
- `docs/architecture.md`: the algorithm (per-cell construction + stitching) and module map.
- `docs/correctness.md`: guarantees, outcome classes, and limits.
- `docs/performance.md`: benchmark guidance and perf knobs.
- `docs/environment-knobs.md`: environment-variable ownership and test-mutation policy.
- `docs/work-log.md`: authoritative active triage, dependencies, and backburner.
- `docs/internal/README.md`: index of closed plans, audits, and experiment records.
- `docs/internal/code-quality-closeout.md`: consolidated July 2026 cleanup decisions and evidence.
- `docs/research/README.md`: uncommitted hypotheses and retired design ideas.

## Module Map (Current)

```text
src/
├── lib.rs                         # Public API and feature-gated internal exports
├── types.rs                       # SpherePoint storage + raw input adapters
├── diagram.rs                     # SphericalVoronoi storage
├── cell_layout.rs                 # Internal live cell-span view
├── cell_mesh.rs                   # Explicitly simplified S2 cell meshes + provenance
├── adjacency.rs                   # Generator adjacency derived from diagram/cell mesh
├── delaunay.rs                    # Dual triangulation views
├── validation.rs                  # Topology/consistency checks
├── locate.rs                      # Point-location API
├── measures.rs                    # Area, centroid, and Lloyd geometry
├── spherical_arc.rs               # Owner-conditioned spherical edge geometry
├── embedding.rs                   # World-space sphere projection/wrappers
├── error.rs                       # VoronoiError
├── fp.rs                          # Numeric helper ops
├── tolerances.rs                  # Centralized numerical tolerances
├── policy.rs                      # Grid and query policy
├── packed_layout.rs               # Packed point/layout helpers
├── spatial_order.rs               # Deterministic spherical ordering
├── point_audit.rs                 # Profiling-only point audit surface
├── quality.rs                     # Tools-only diagnostic/reference routines
├── knn_clipping/                  # Main backend
│   ├── compute.rs                 # End-to-end backend orchestration
│   ├── driver.rs                  # Per-bin cell construction
│   ├── preprocess.rs              # Near-coincident merge pass
│   ├── edge_reconcile.rs          # Post-assembly edge reconciliation
│   ├── output_resolution.rs       # Exact stored-zero canonicalization
│   ├── local_rebuild.rs           # Hull3d local-rebuild orchestration
│   ├── local_hull.rs              # Robust local 3D hull
│   ├── union_find.rs              # Deterministic component tracking
│   ├── cell_build/                # Single-cell construction loop
│   └── topo2d/                    # Gnomonic/topological clipping
├── live_dedup/                    # Sharded dedup + assembly
├── timing/                        # Real/zero-sized timing backends
├── cube_grid/                     # Spatial index + query stack
│   ├── build.rs                   # Grid construction
│   ├── dense.rs                   # Dense-cell detection and feedback policy
│   ├── projection.rs              # Face/uv/st conversion helpers
│   ├── weld.rs                    # Near-coincident pair discovery
│   ├── query/                     # Directed resumable kNN query path
│   └── packed_knn/                # Packed batched directed kNN
├── generated/
│   └── sort_nets.rs               # Auto-generated sorting network code
└── sort.rs                        # Internal small-sort utilities (feature/test use)
```

`live_dedup/` and reconciliation are intentionally specialized to spherical `Vec3` positions.
Reintroduce a shared position abstraction only when a second in-repository backend supplies a
current consumer and contract tests. `src/generated/sort_nets.rs` must be changed through
`scripts/gen_sort_nets.py`, not by editing the generated body.

## Supported Features

- `parallel` (default): rayon-based parallel execution across eligible build/query work.
- `glam`: public input support and checked `SpherePoint` conversions for `glam::Vec3`.
- `serde`: checked serialization/deserialization support for diagram types.

## Internal Features

These are repository instrumentation, benchmark, comparison, or probe hooks and are not
semver-covered public features:

- `timing`: detailed timing instrumentation.
- `profiling`: helpers for profiling runs (e.g. inline control).
- `microbench`: internal microbench harnesses.
- `manual_probes`: wholly manual/ignored integration-test targets.
- `simd_scalar`: scalar/autovectorized comparison backend.
- `tools`: benchmark/utility binaries, the fidelity campaign, and quality diagnostics.

## Tests

Default integration-test targets in `tests/`:

- Core contracts: `api`, `correctness`, `validation`, and `backend_fingerprint`.
- Derived/public views: `delaunay`, `embedding`, and `locate`.
- Geometry and representation regimes: `adversarial`, `geometric_regressions`, `high_degree`,
  `output_resolution`, `small_n_geometry`, `weird_geometry`, and `weld_cell_survival`.
- Pipeline regressions: `edge_reconciliation`, `local_rebuild`, and `local_rebuild_contract`.
- Test infrastructure: `env_isolation`; `tests/support/` is shared code, not a test target.

Feature-gated targets declared explicitly in `Cargo.toml`:

- `coincidence_probes` and `robustness_campaign`: `manual_probes`.
- `fidelity_campaign`: `tools`.

## Git Workflow Policy For Agents

- Agents may edit files in workspace as needed for requested tasks.
- Agents may commit without explicit per-turn approval when the requested change is complete, scoped, and validated.
- Do not auto-commit exploratory, partial, or uncertain work; ask before committing in those cases.
- For substantial work, prefer a topic branch like `agent/<short-topic>` unless user says otherwise.
- Keep commits scoped to a single logical change.
- Run relevant validation before commit and report what was run.
- Report the commit hash and message after committing.
- Do not push, force-push, amend, rebase, or reset unless explicitly requested.
- Do not include unrelated file churn just to satisfy formatting/linting unless requested.

## Change Checklist (Recommended)

1. Implement minimal coherent change.
2. Run `cargo fmt`.
3. Run focused tests/checks relevant to changed modules.
4. If broad behavior changed, run `cargo test --release`.
5. Summarize results and residual risks.

## Known Limitations

- Inputs should be unit-normalized.
- Numerical edge cases can still produce degeneracies; use `validation::validate`.
