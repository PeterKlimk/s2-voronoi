# TODO / Roadmap

This is the active working roadmap for `s2-voronoi`, updated after the 2026-06 release-readiness
evaluation. If you want to know "what should we work on next?", start here.

Related documents:

- `docs/correctness-contract.md`: the "essentially Voronoi" contract this roadmap implements,
  including the empirical coincidence/weld evidence.
- `docs/engineering-findings.md`: the findings log (correctness/contract/organization issues).
- `docs/supported-envelope.md`: outcome classification for the current backend.

## Guiding goal

A prestige-quality open-source crate that is the fastest spherical Voronoi implementation
available, with a machine-checked topological guarantee: **valid graph or clean error, at any
scale, for any input inside a precisely documented envelope.**

Headline claims to be able to make honestly at release:

- millions of cells per second on desktop CPUs (2.5M points < 500ms on a Ryzen 3600 class part)
- strict graph validity asserted by CI at multi-million point counts
- a documented coincidence policy with measured safety margins (not folklore epsilons)

## P0: Implement the correctness contract

The empirical groundwork is done (see correctness-contract.md); this is the implementation.

1. ~~**Replace `MergeDensity` with a fixed-radius weld**~~ **Done.** `PreprocessMode::Weld`
   (default) welds at the coincident-distance radius via a parallel quantized-key sort pass
   (~110ms at 2M points); welded twins alias their canonical cell and are exposed via
   `SphericalVoronoi::weld_map`; the validator accounts canonical cells and checks weld-map
   consistency. `Disabled` and `MergeWithin(r)` retained.
2. ~~**Orphan-vertex policy.**~~ **Done.** Unreferenced vertices are a representation note
   (`ValidationReport::representation_notes`), and `SphericalVoronoi::compact_vertices()` removes
   them on demand. (Possible refinement: O(repairs) tracking during edge repair so compaction is
   proportional to defect count.)
3. ~~**`ClippedAway` backstop.**~~ **Done (as classification, not masking).** Emitting a
   degenerate cell turned out unsound — neighbors already clipped against the missing cell's
   bisectors would carry unpaired edges. Instead, a coincidence-driven `ClippedAway` (sub-weld
   neighbors present; only reachable via `Disabled`/undersized `MergeWithin`) returns an
   actionable `DegenerateInput` naming the coincident generators; without coincidence it stays
   `ComputationFailed` (bug class). See correctness-contract.md.
4. ~~**Input validation.**~~ **Done.** O(n) finite check at compute entry returns
   `VoronoiError::InvalidInput { point_index, .. }`.
5. ~~**Promote the evidence to CI.**~~ **Done** (modulo wiring scheduled CI when CI exists): the
   2M/3M/4M fuzz sweeps assert `STRICT_VALID` under the default config (`#[ignore]` for runtime
   only), weld contract tests cover exact duplicates, ulp clusters, and seam-aligned pairs,
   resolvable-regime tests cover above-weld pairs and the rotated-symmetric control with welding
   disabled, and the margin-mapping probes live on as a permanent diagnostic suite in
   `tests/coincidence_probes.rs` (run after numerics changes to detect boundary drift).
6. ~~**Tighten loose assertions**~~ **Done.** Euler exactly 2 (over referenced vertices), zero
   sub-3-vertex cells, zero duplicate boundary vertices in `tests/correctness.rs`.

**P0 is complete.** The correctness contract is implemented and enforced by asserting tests.

## P1: Release engineering

1. ~~**Stable-Rust build path.**~~ **Done — better than planned: nightly is gone entirely.** All
   explicit SIMD flows through the `fp.rs` backend seam (`PointChunk8`/`Dots8`); the default
   backend is the `wide` crate (stable Rust, explicit lanes), with `simd_scalar` as a
   debugging/comparison fallback. Reference-machine (Ryzen 3600, target-cpu=native) benchmarks
   showed wide at parity with the old portable_simd backend (within ~1-2%, winning some runs);
   all backends are bit-identical (tests/backend_fingerprint.rs). portable_simd was deleted
   (recoverable from git history if std::simd stabilizes). MSRV set to 1.88.
2. ~~**Feature consolidation.**~~ **Done.** `simd_clip` and `packed_knn_sort_small`
   merged away (kernel-contest promotions); remaining internal flags (`timing`,
   `profiling`, `microbench`, `simd_scalar`, `fma`, `tools`, `bench_voronoice`)
   documented as non-contractual in README/CLAUDE.md, with docs.rs metadata
   building only the public surface (`parallel`, `glam`, `serde`).
3. ~~**Finish the non-panicking contract**~~ **Done.** The clipper "invariant failure" panic was
   proven unreachable (mixed cyclic masks always carry both transitions) and converted to a
   documented `unreachable!`; the fallback angle sort uses `total_cmp` (the old
   `partial_cmp + unwrap_or` could trip std::sort's total-order check on NaN). Remaining panics
   are genuine bug traps per the supported-envelope contract.
4. ~~Zero-warning builds, `rust-version = "1.88"`, clippy-clean, `#[deny(missing_docs)]`~~
   **All done** (missing_docs now denied across all features).

## P2: API completeness

In order of user value:

1. ~~**Neighbor adjacency**~~ **Done.** `SphericalVoronoi::build_adjacency()` returns a
   `CellAdjacency` built by sort-based edge pairing (O(E log E), on demand); neighbor entry `k`
   of a cell sits across boundary edge `(v[k], v[k+1])`, welded twins alias their canonical
   cell's list, and the neighbor pairs double as Delaunay edges (covers most of item 3).
2. ~~**Cell areas / centroids**~~ **Done.** `cell_area(i)` (signed solid-angle fan, f64; sums
   to 4π over canonical cells, test-pinned) and `cell_centroid(i)` (exact boundary integral,
   orientation-robust) — Lloyd relaxation is a 3-line loop, with a convergence test as the demo.
3. **Delaunay dual access** — mostly covered by `build_adjacency()` (neighbor pairs are the
   Delaunay edges); an explicit triangle-list API remains if users ask for it.
4. ~~**`serde` feature**~~ **Done.** Optional derives on UnitVec3, SphericalVoronoi (weld map included), and CellAdjacency; round-trip test behind the feature.
5. ~~`compact()`~~ **Done** as `SphericalVoronoi::compact_vertices()` (landed with P0.2).

Explicitly deferred: weighted Voronoi, f64 storage, no_std, dynamic insertion/deletion.

## P3: kNN / performance robustness

Design principle (2026-06 discussion): the knot is not the measured fast path (chunk0 / tail /
expand-r2 — which already are Chebyshev shells r=0,1,2 with bespoke tuning, serving ~99.95% of
cells); it is that the fallback beyond r=2 is a *second algorithm* (the dual-heap
`DirectedNoKCursor`, ~0.05% of cells, a large share of the stack's conceptual weight). The end
state is one outward-marching shell frontier where the first three shells stay hand-tuned, the
rest are generated, the certificate is uniform (per-shell annulus bound), eligibility is decided
at cell level everywhere (a grid cell maps to one bin), and there is no second algorithm.

In implementation order:

1. **Standalone NN contract suite (before touching anything).** Test the neighbor-source layer
   in isolation from the Voronoi pipeline: emitted eligible set equals brute force, and every
   frontier certificate conservatively bounds every unseen eligible point. Explicit cube-grid
   edge/corner coverage: queries and points at the 8 corners (7-cell neighborhoods), the 12
   face-edge seams, face centers/poles, exact-seam coordinates (1/sqrt(3), 1/sqrt(2)), all
   points on one face (cross-face shell traversal), one point per face, antipodal pairs (maximal
   shell radius), near-wall points (classification robustness), tiny n, clusters, bimodal
   sparse+dense (forces deep shells). This suite is the strangler-pattern net: the shell rework
   must pass it unchanged. Note: emission *order* is deliberately not part of the contract (the
   current exact-order stream test pins the cursor's implementation and will be rewritten).
2. **Grid resolution policy.** Mostly done: density set to 24 from the 3600 sweep (fastest at
   100k/500k/2M, flat optimum, 4.8-7.1% over the old 16; neighbors-before-termination 8.16->8.44
   over that n range, density-independent), occupancy feedback + memory cap implemented, sweep
   script + counters in place. Remaining: re-sweep beyond 4M and on non-uniform distributions;
   the shells-native big-cell path below. Original text follows for context. Important constraint (2026-06):
   cells need *more* neighbors before terminating at higher densities, so the optimal
   points-per-cell is not a constant — it varies with point count and distribution. Replace
   with: (a) a target-density *curve* fit from a benchmark sweep across n (and sanity-checked on
   non-uniform distributions), not a single re-tuned constant; (b) occupancy feedback — the
   build already histograms occupancy; rebuild at higher resolution when max occupancy exceeds a
   bound, capped by a memory budget (total cells O(n)); (c) for concentration beyond what global
   resolution can fix within memory, a shells-native big-cell path (chunked selection within
   shell 0) instead of today's bail to the cursor (`SlowPath`). The termination-vs-density
   interaction also suggests recording neighbors-before-termination stats in timing runs so the
   curve has an explanatory model, not just fitted numbers.
3. ~~**Shell generalization behind a policy switch.**~~ **Done, flipped, cursor deleted.**
   `ShellFrontier` (BFS layers, per-layer sorted emission, ring certificates) passed the NN
   contract suite on both paths; the side-by-side phase caught two consumer-side bound bugs
   (mid-batch and pre-consumption termination both assumed "next emission bounds all unseen",
   true for cursor order and packed invariants, false for shell layers - fixed by combining
   with the layer certificate). Reference-machine A/B at parity (exact tie at 500k; 2M noise
   exceeded the takeover's 0.05%-of-cells ceiling). The flip removed `DirectedNoKCursor`, both
   scratch heaps, the stream's dual-takeover state, the cursor cadence path, and the
   exact-order stream test; `quality::assess` nearest-generator sampling ported to shells.
   There is no second algorithm.
4. **Policy shrink.** After (2) normalizes occupancy, revisit the remaining constants (chunk
   sizes, termination cadence, count-model flags) — most were compensating for unbounded density
   variance and likely simplify or die. Keep what remains expressed through `src/policy.rs`.
5. Open questions parked from the discussion: whether `termination_max_k` survives (looks
   vestigial once certificates are uniform and occupancy is bounded), and per-shell vs per-cell
   certificate granularity (believed to tax only already-pathological cells; benchmark confirms).

## P4: Code quality (prestige pass)

1. ~~**Centralize tolerances**~~ **Done.** `src/tolerances.rs`: every numerical tolerance with
   its empirical justification, grouped by pipeline stage, with a hierarchy sanity test and the
   re-validation protocol in the module header. Proven a pure relocation (identical diagram
   fingerprint). The inert `termination_max_k` config and the dead cadence policy were removed
   alongside (the parked P3.5 question, closed).
2. ~~**Split the god functions**~~ **Done.** `build_cell_into` decomposed into phase functions
   (seed clipping, stream consumption with per-batch clipping, terminal classification /
   extraction) over two small state structs (`BuildTrace` diagnostics, `BuildCounters`
   stats), which also collapsed `unexpected_failure_error`'s 13 arguments. The duplicated
   edge-endpoint matching is now one `reconcile_edge_endpoints` primitive (reverse-winding
   full match + cross-slot partial search) shared by the in-shard and cross-bin overflow
   paths, verified arm-by-arm. Proven behavior-preserving by identical diagram fingerprint;
   the crate is now clippy-clean (remaining hot-path arg-count/loop-index lints carry
   targeted allows with stated justifications).
3. ~~**Document the stitching invariant.**~~ **Done.** `docs/live_dedup.md` ("The stitching
   invariant"): the total order, the per-pair coverage contract (cross-bin double discovery vs
   same-bin seed forwarding), why the earlier side always delivers when it agrees the edge
   exists, and the epsilon caveat that defines reconciliation's job and motivates P5.
4. Invariant comments on the remaining undocumented unsafe blocks (small clippers; the grid
   scatter is already documented).

## P5: Structural upgrade — consistency by construction

**Designed (2026-06): see `docs/p5-consistency-design.md`** — the authoritative spec; this
section is the summary. Make every shared combinatorial decision canonical via three pillars:
(1) the canonical primitive is the symmetric in-circle predicate on generator 4-tuples (no
frame choice — supersedes this section's earlier "sorted-index frame" sketch); (2) filtered
escalation keeps the chart-local hot path (one margin compare per clip lane; only near-ties
evaluate canonically); (3) question-set closure — certificates conservative by EPS_CERT >
EPS_FILTER so differing termination can never hide a marginal question from one side (the
design's named failure mode B).

Hard exit criteria: the edge-repair net's defects go to ZERO at every bin count (then
edge_reconcile is deleted and detection becomes a bug trap), and paired benches show <=1% ST
regression at 500k/2M — criterion 2 can reject the project; the fallback is the hardened
2026-06 status quo.

**DEFERRED (2026-06-14): removing `edge_reconcile` is rejected for now.** The tie-regime
defect class was eliminated by construction (the strict keep rule shipped); the error-regime
remainder needs a cheap-AND-correct canonical escalation that is still unsolved (margin-gated
escalation was measured unsound). Deferral is safe because every defect is already caught and
repaired to strict validity — finishing P5 buys simplification, not a breakage fix. See the
Status block in `docs/p5-consistency-design.md` for the full rationale and the resume plan
(certificate closure / EPS_CERT).

### Tier-2 re-clip repair (fallback hardening) — correctness-first

Opt-in `S2_RECLIP_REPAIR` on `agent/fallback-incremental-clip`. **Gate now sound
(landed); fuzzed.** Full review + two Codex passes:
`docs/reclip-fallback-review-2026-06.md`; design + roadmap:
`docs/reclip-repair-design.md`. The repair only fires on `mega` (unrepairable
stitch errors are never observed outside it), so it is rare; its cost is an
acceptable stopgap, and runtime always-on validation is rejected (kills perf).

Correctness-first work:

1. ~~**Gate → validator equivalence (CRITICAL).**~~ **Done.** `repair()` runs
   `validation::verify_sphere_effective_strict` over the re-stitched diagram and
   reverts-or-loud-fails; bound to the repair (covers plain + report paths), not
   `S2_VORONOI_VERIFY`. Rollback made byte-identical (first-snapshot-per-`g`,
   Codex-found). Determinism (sorted vids) + OOB guard landed.
2. **Fallthrough fail-loud — TRIED, REVERTED.** Returning `Vec::new()` breaks all
   mega repair (the fallthrough is load-bearing). The kept approximate corner is
   a topology-valid accuracy defect, not invalid topology. Proper fix (synthetic
   detection record forcing the output-invariant scan) deferred.
3. ~~**Cheap robustness.**~~ **Done** (OOB guard, deterministic vids; zero-area
   winding + C5 pin-ordering left as low-pri / not-currently-reachable).
4. ~~**Fuzzing.**~~ **Done.** `robustness_campaign` + `tests/reclip_repair.rs`
   cover the repair path with the contract `no surviving PostRepairUnpaired
   residual ⟹ strictly valid`. Two sweeps: default (detection-completeness) and
   `RECLIP=1` mega (recovery rate, `post_repair` column).

Resolver rework — **MEASURED, direction corrected** (see
`docs/reclip-repair-design.md` roadmap #2). Prototyped the exact-predicate swap
and instrumented mega before investing: exact cocircular ties NEVER fire on mega
(`in_circle==0` count = 0), the exact predicate REGRESSED recovery (jitter 4/6 →
exact 2/6 clean, the jitter approximation was helping the assembly), and the real
bail is **runaway `Expand`** (boundary recovery leaves degree-1 endpoints →
component grows 8→49 cells → budget bail). So the lever is the **assembly, not the
predicate**: pursue **B — local constrained Delaunay** (dual of one local
triangulation incl. boundary generators → no boundary-recovery, no `Expand`, no
degree-1 runaway), which needs a real spherical CDT/hull engine. **C+M
deprioritized** (no mega benefit; structured high-degree already handled by Tier-1,
`tests/high_degree.rs`).

### Structured high-degeneracy inputs — perf weakness (low priority)

A *clean, spread-out* input whose Voronoi has many genuine degree-4+ vertices
(cubed-sphere grid, regular polytopes) stays strictly valid at every scale but
runs **~10× slower than uniform** (linear, no cliff): O(n) coincident-vertex
disagreements → thousands of reconcile defects (uniform sees 1–20). A deliberate
consequence of the hot-path triple-key choice + sparse-defect-tuned reconcile,
not a correctness/scaling break. Recovery ideas exist (likely: position-aware
coincident-vertex merge at assembly time instead of the detect→reconcile
round-trip — TBD). Low priority. Regression-guarded by `tests/high_degree.rs`
(cube/cuboctahedron/cubed-sphere) and the `grid` campaign distribution. Detail:
`docs/optimization-ideas.md` "structured high-degeneracy inputs".

Known pre-existing branch debt: 3 clippy warnings (`timing/stub.rs:88`,
`cell_build/run/frontier.rs:61`, `compute.rs:587` `reconcile_edges` 8-arg) trip
CI's `-D warnings`; clean up before merge.

## P6: New geometries

The engine decomposes into geometry-specific layers (`cube_grid` indexing, `topo2d` chart +
bisectors) and a geometry-agnostic core (directed stitching, sharded dedup, reconciliation). A
`Geometry` abstraction needs: spatial index with conservative cell bounds, bisector constructor,
2D clipping chart, termination certificate.

- **Plane** is mostly simpler (no chart needed — clip directly; hemisphere machinery vanishes)
  with one new problem the sphere never has: unbounded cells need a boundary policy (bbox clip /
  marked-unbounded), which touches the stitching contract.
- Sequence: ship the sphere first; keep geometry-specific code quarantined behind the two module
  boundaries now; do the trait extraction when a second geometry is actually being built, so the
  abstraction is shaped by two instances rather than one.
- *(Done since: the plane shipped as `compute_plane`, then periodic as `compute_plane_periodic`,
  both as sibling drivers over the shared core — the quarantine held.)*

### Power diagrams (weighted Voronoi) — designed, not started

Assessment (June 2026): doable as a planar-arc-sized project; recorded here so the design
thinking isn't lost.

What carries over untouched: triplet vertex keys, sharded dedup, edge forwarding,
reconciliation, assembly. Bisectors stay straight — the planar radical axis is the
perpendicular bisector offset by `(w_i − w_j) / (2 d_ij)`; on the sphere use **Sugihara's
spherical Laguerre diagram**, whose bisectors are still great circles, so gnomonic clipping
survives. Cell construction changes by one line per geometry.

The three real changes:

1. **Termination certificates** (the hard part): `d > 2·max_r` is unsound with weights — a far
   generator with a large weight can still cut. The sound bound picks up a weight-spread term;
   the grid needs per-cell weight maxima and frontiers must order candidates by power distance.
   Every certificate proof in `plane_grid`/`cube_grid` gets revisited; shell-only first, packed
   later (the periodic playbook). Silent-wrong-answer territory → contract-suite treatment.
2. **Empty cells become legal**: a dominated generator owns nothing, but the current contract
   treats `ClippedAway` as an error. Needs an API-visible `hidden_map` (analogous to
   `weld_map`) and ripples into validation, adjacency, and the regular-triangulation dual.
   Design this first.
3. **Torus guard × weights**: the half-period guard check must use the weighted `max_r`
   (fail-loudly mechanism unchanged).

Combinatorial blow-up is real but lives in tests/docs, not engine code: weighted follows the
existing sibling-driver template (additive cost per geometry). Test suites parameterize on
weighted-with-equal-weights ≡ unweighted topology.

Base-case performance must not regress: no runtime `w == 1` checks in hot loops. Preferred:
monomorphize over a weighting policy with a ZST unweighted instantiation (offsets constant-fold
away; the backend fingerprint proves bit-identity). Fallback: a fully separate sibling driver.

Sequence when started: empty-cell contract design → planar bounded power → sphere Laguerre →
torus → packed stages.

## Working rules

(unchanged)

- Do not trade contract clarity for small benchmark wins.
- Heuristics live in policy, not in hot-path cell processing.
- Do not broaden the public promise beyond what tests and docs back.
- One coherent improvement with tests over multiple speculative micro-optimizations.

## Exit criteria for release

- P0 complete: fuzz + coincidence probes green in CI with default config, asserting strict
  validity.
- P1 complete: builds on stable (degraded perf), zero warnings, no input-reachable panics.
- README states the contract, the envelope, and the measured performance honestly.
- At least neighbor adjacency from P2 (areas/centroids strongly preferred — enables the Lloyd
  demo).
