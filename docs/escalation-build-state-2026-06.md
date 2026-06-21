# Escalation build — state & resume (2026-06-22)

Resume anchor for the adaptive-canonical-clip / escalation work. Read this +
`docs/adaptive-canonical-clip-design-2026-06.md` (design + measured GO/NO-GO) to
pick up. Branch: `agent/canonical-predicate-topology`.

## One-line status

The near-cocircular unpaired-edge residual (the "valid-or-error" regime) has a
**proven local fix**: rebuild a defect neighborhood as ONE exact local Delaunay
(`local_hull`) and read each generator's cell off the shared dual — cells pair
by construction. Proven on real mega defects. **Remaining: splice the rebuilt
cells back into the global diagram and run `validate`.**

## Commits (this branch, newest last)

- `8f5392d` GO/NO-GO instrumentation (`p5_shadow` probes) + design doc + verdict
- `c661511` A: `local_hull` validated as the exact engine (tests only)
- `e13724c` C slice: rebuild resolves real defects (a–b consistency)
- `8f2a654` C: defect cells FULLY internally valid (all edges pair)

## The settled design (don't relitigate — see design doc §6/§6b/§6c)

Pipeline: **detect residuals → connected component → rebuild the whole component
as ONE `local_hull` → splice into the diagram → `validate` → grow until clean.**

Why it works where every reverted repair failed: a component is rebuilt as a
SINGLE hull, so all its cells share one triangulation and pair by construction;
the boundary is a WELL-CONDITIONED rim (fast == exact there), so there is no
exact/approximate seam to stitch. The reverted pin-by-key/boundary repairs
stitched AT the degeneracy, which is why they failed.

## What is PROVEN (committed, tested)

Run: `cargo test --release --features escalate_probe --test escalate -- --nocapture`
and `cargo test --release --lib local_hull`.

1. **Dual consistency** (`local_hull::tests::dual_cells_agree_on_shared_edges`):
   every two generators sharing an edge, read from one hull, name the SAME two
   endpoint triples, cyclically adjacent in both fans. THE core invariant.
2. **Rebuild resolves real defects** (`tests/escalate.rs::rebuild_resolves_mega_defects`):
   mega 100k s3's 9 defects → 9/9 rebuild to a consistent shared-edge state
   (5 genuine edges, 4 spurious fast-path edges correctly removed). Stable
   k=48..192 ⇒ coverage (step B) is NOT the bottleneck for mega.
3. **Defect cells fully valid** (`tests/escalate.rs::defect_cells_are_internally_valid_after_rebuild`):
   all 207 edges of the 9 defects' cells pair internally, 0 rim edges.

So: **exact rebuild = fully-valid local subdivision.** UNTESTED: that the
rebuilt rim matches the fast diagram (the splice).

## What is NOT done — the next step (the hard piece)

**Global splice + `validate`.** Obstacle: representation mismatch.
- The diagram (`src/diagram.rs`) stores cells as vertex-INDEX lists over a
  shared `vertices: Vec<UnitVec3>` coord array.
- Rebuilt cells (`escalate::RebuiltCell`) are global-id TRIPLES + circumcenters.
- Splicing needs a triple→vertex-index map at the rim (reuse the fast index
  where the triple already exists, mint a new vertex otherwise). The fast
  diagram does NOT store triples.

Two routes:
- **(a) Re-run assembly with the rebuild integrated** (the production loop). The
  rebuild emits triple-keyed cells; feed them through the SAME live-dedup /
  edge-reconcile assembly the fast path uses (`src/live_dedup/`,
  `src/knn_clipping/edge_reconcile.rs`), which already keys vertices by triple
  (`VertexKey=[u32;3]`) and runs proximity-merge. RECOMMENDED — it reuses the
  existing triple-keyed assembly and sidesteps the index-mapping problem.
- **(b) Triple-indexed splice via reconstructing fast-cell triples from
  `adjacency.rs`** — circular, because fast adjacency is broken AT the defects.
  Avoid.

Recommended first concrete task: take the rebuilt component cells (triple-keyed)
+ the fast cells (already triple-keyed pre-assembly in
`topo2d/builder/extract.rs:79`), replace the component's emitted cells, and run
the existing assembly + `validation::validate` on the result for mega 100k s3 —
target: 9 defects → strictly valid.

## Code map

- `src/knn_clipping/escalate.rs` — `gather_local` (brute-force k-NN; production
  uses the grid / considered-neighbor set), `rebuild_cells` (one hull →
  per-generator ordered global-id triples; skips broken-fan rim generators),
  `RebuiltCell`, `shared_neighbor`, `check_cell_internally_paired`. `pub` via
  the `escalate_probe` feature (`src/lib.rs`), dead-code until the loop lands.
- `src/knn_clipping/local_hull.rs` — the exact engine: `build`, `faces`,
  `face_circumcenter`, `cell_faces` (dual fan). Validated; no prod change needed.
- `src/knn_clipping/p5_shadow.rs` — GO/NO-GO probes (`set_audit_cutoff`,
  cross-cell/superset report). `tests/p5_shadow.rs` drives them.
- `tests/escalate.rs` — the slice integration tests.

## Refuted / settled — do NOT re-tread

- **Exact-everywhere is NOT mandatory** (blind-adjudicated). Partial escalation
  is viable because the fast/exact "seam" is avoidable: on a well-conditioned
  shared edge fast == exact (cross-cell conflict tail ~1e-10). Cost of
  exact-everywhere measured ~+31% uniform / +48% mega anyway — affordable, but
  escalation pays it only on the degenerate component.
- **Coverage / EPS_CERT is NOT the mega lever** (termination-pad ladder: 23×
  more neighbors → byte-identical defects; and the k-invariance above). mega
  defects are DECISION divergence.
- **High-degree merge NOT needed** ([[high-degree-vertices-rationale]]): genuine
  exact-cocircular vertices are ~unconstructable at f32; near-cocircular (mega)
  gets a definite diagonal; the rare exact case is handled by existing
  proximity-merge. Don't build a degree-k vertex representation.
- **No cheap PROACTIVE complete flag exists** (missing-edge detection needs exact
  local topology). Use the post-assembly defect list (`unresolved_edge_pairs`,
  `compute.rs:146-199`) as a perfect-recall REACTIVE trigger.
- **A drift-based / chart-margin flag is unreliable** (lerp drift). The
  defect-driven reactive trigger sidesteps it.

## Backburner (see [[exact-valid-then-simplify]])

Once the exact valid graph is produced, reintroduce **epsilon-edge collapse** as
a FEATURE on the valid structure (`exact valid → inexact valid` is easy;
`inexact invalid → inexact valid` is the trap). The crate had epsilon-collapse
before; resurrect it on the right foundation.

## Relevant memory notes

`adaptive-canonical-clip-direction`, `mega-coverage-ruled-out`,
`high-degree-vertices-rationale`, `exact-valid-then-simplify`,
`mega-regime-concluded`, `reclip-fallback-review-2026-06`.
