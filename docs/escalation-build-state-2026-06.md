# Escalation build — state & resume (updated 2026-06-22)

Resume anchor for the adaptive-canonical-clip / escalation work. Read this +
`docs/adaptive-canonical-clip-design-2026-06.md` (design + measured GO/NO-GO) to
pick up. Branch: `agent/canonical-predicate-topology`.

> **CORRECTION (2026-06-22):** an earlier version of this doc presented
> "rebuild a defect neighborhood as ONE exact `local_hull` and splice it back"
> as the *proven local fix* and recommended re-running assembly with that rebuild
> (route (a), local-hull source). **That conclusion was DISPROVEN.** The
> local-hull splice DIVERGES, and the divergence was an artifact of using
> `local_hull` as the oracle — not a real obstacle to local repair. The current
> state, what is proven vs open, and the corrected recommended path are below.
> Authoritative findings: memory notes `fast-clip-is-projected-delaunay` and
> `route-a-splice-diverges`.

## One-line status

The near-cocircular unpaired-edge residual (the "valid-or-error" regime) is a
**tiny, per-cell-chart f64-rounding error (~tens of cells, 0.01–0.02%, in every
regime)** — NOT a metric difference and NOT an f32 resolution floor on the graph.
Detect + repair with a **consistent exact oracle (raw `orient3d` in-circle)** is
**viable and cheap**: it converges in ~1 round at the small defect set. Measured
to make **4/5 mega-100k seeds strictly valid** (offline, behind a gate). The
recommended path is now **reactive detect+repair (option B)** with a
whole-diagram never-worse gate; clip-time exact-by-construction (option A) is
demoted to a future EXACT/CANONICAL-output feature, not a prerequisite for the
valid-diagram contract. **Nothing is committed to git**; the escalation lives
behind `set_escalation_enabled` (off by default) and the `escalate_probe`
feature.

## The corrected model — what the mega defect actually is

(Authoritative: `fast-clip-is-projected-delaunay`.)

1. **Projected ≡ raw Delaunay in EXACT arithmetic.** Gnomonic (per-cell clip),
   stereographic (delaunator), and raw-3D `orient3d` (local_hull) all compute the
   SAME spherical Voronoi/Delaunay in exact arithmetic; the differences are pure
   f64 precision. Measured (`tests/escalate.rs::projection_theory_3way`, 12k
   interior cells): uniform agrees to a 4-cell precision tail across all three.
2. **The defect is per-cell-gnomonic-CHART f64 rounding.** A vertex (g,a,b) is
   decided by 3 cells, each in its OWN tangent-plane chart (centered on g / a / b).
   Exact ⇒ all three agree; at f64 each chart rounds the same keep/drop decision
   differently ⇒ cross-cell disagreement ⇒ an unpaired edge (~tens of cells per
   100k). A single GLOBAL chart (delaunator) evaluates each decision once ⇒ 0 such
   defects. The true error is TINY in every regime, mega included.
3. **The exact diagram does NOT disagree with fast on ~9% / "the whole cap is
   invalid" — that was WRONG.** That figure was a `local_hull` IMPLEMENTATION
   artifact: the global convex hull of points clustered on a sphere PATCH wraps
   spurious "back faces" around the empty far hemisphere, corrupting
   cluster-boundary cells. It is NOT a real projected-vs-raw metric difference
   (uniform's 4-cell agreement proves it). With a real exact oracle (delaunator,
   or deterministic-tie raw orient3d) the changed set is ~tens of cells everywhere.

## The corrected recommended path

**Option B — reactive detect + repair (fill-with-grow), the recommended path:**

Pipeline: **post-assembly defect list (`unresolved_edge_pairs`,
`compute.rs:146-199`, perfect-recall reactive trigger) → cluster the
unpaired-incident (broken) ∪ malformed cells → replace each with the
**consistent exact oracle's** cell, rim-pinned from trusted non-cluster neighbors
→ grow until the rim is well-conditioned → `validate` with a whole-diagram
never-worse gate (revert if not strictly better).**

- **The consistent oracle = exact raw `orient3d` in-circle**
  (`canonical::in_circle_sphere_sign`, already exists): a function of the 4 raw
  points, identical from every cell, with no projection and no hemisphere issue.
  This is the safest exact decision. (delaunator works as an oracle only because
  it agrees with fast — single global chart, projected like fast — but it is NOT
  canonical: its near-cocircular answer is stereographic-pole-dependent, so a
  local delaunator with a different pole diverges. Use it as a *reference* in
  tests, not as the production oracle.)
- **Why it converges (does NOT explode):** with a consistent oracle the detected
  defect cells, once replaced, pair immediately with their unchanged neighbors
  (which already equal exact). Measured: `detect_fix_expand_delaunator` converges
  in ONE round at repair_set 4–19 (mega 100k), 80 (mega 300k), 0 (uniform/
  clustered). The historical local_hull-splice cascade (grew to ~2900 cells over
  40 rounds) was caused by local_hull being an INCONSISTENT oracle (insertion-
  order ties + per-gather cap artifacts + clustered back-faces) manufacturing
  fresh disagreement at every rebuilt cell — NOT by a real obstacle.
- **Measured repair status:** the `fill_cluster_pass` engine made mega-0.8 100k
  s1/s3/s4/s5 strictly VALID; s2 regressed (3→5); 300k s3 partial (51→45);
  uniform clean no-op. So **4/5 100k seeds fixed**. Holdout (s2) is a
  non-manifold BOUNDARY-EXTRACTION issue (cell 15514: a 27-vertex boundary that
  visits a generator's bisector in disconnected arcs — consecutive verts sharing
  only one generator), NOT a cascade.
- **Remaining work for B:** (1) swap the oracle to raw `orient3d` (consistent,
  hemisphere-safe) instead of delaunator/local_hull; (2) add the whole-diagram
  never-worse gate (validate before/after, revert unless strictly better) so the
  regressing cases (s2, 300k) are safe; (3) robust boundary extraction for the
  non-manifold holdout; (4) cost/scale (1m/2m) + interior-vertex f32-
  realizability at scale are untested.

**Option A — clip-time exact decision by construction ("primary clip"):** NOT
required for the valid-diagram contract. The work is "identify the cells + repair
with the raw-orient3d decision," not exact arithmetic in the hot clipper. A only
matters as a future EXACT/CANONICAL-OUTPUT feature, and even then it is best
framed as "flag + repair," not "exact clip by construction." Because the true
error is ~tens of cells everywhere, A's canonicalization target is tiny and its
cost is cheap everywhere (not the earlier "~+31% uniform / +48% mega" upper
bound, which assumed exact-everywhere). The design doc's §2 "primary clip, NOT a
repair" argument ("a repair can't stitch because exact meets approximate at the
rim") is **UNDERCUT**: a rim-pinned repair with a consistent oracle has no such
seam, and projected ≡ raw means there is barely any disagreement at the
well-conditioned rim anyway.

## What is PROVEN (committed, tested)

Run: `cargo test --release --features escalate_probe --test escalate -- --nocapture`
and `cargo test --release --lib local_hull`.

1. **Dual consistency** (`local_hull::tests::dual_cells_agree_on_shared_edges`):
   any two generators sharing an edge, read from one hull, name the SAME endpoint
   triples, cyclically adjacent in both fans. (Holds within a SINGLE hull; this is
   why a from-scratch local hull is internally consistent but, as a SPLICE oracle
   against the fast diagram, is not — see refuted dead-ends.)
2. **Rebuild resolves defects internally** (`rebuild_resolves_mega_defects`,
   `defect_cells_are_internally_valid_after_rebuild`): a defect neighborhood
   rebuilt as one hull is internally fully valid. This only ever proved
   rebuilt-vs-rebuilt consistency — NEVER the splice against the fast diagram.
3. **Projected ≡ raw, error is tiny** (`projection_theory_3way`,
   `a0_exact_reference_delaunator`): exact diagram differs from fast on ~tens of
   cells (0.01–0.02%) in every regime; mega differs only in that its changed
   cells happen to cause unpaired defects (uniform's changed cells are benign).
4. **Detect+fix+expand converges with a consistent oracle**
   (`detect_fix_expand_delaunator`): 1 round, bounded, valid on mega 100k/300k.
5. **Fill-with-grow makes 4/5 mega-100k seeds valid** (`fill_cluster_pass`,
   wired-but-gated): see repair status above.

## What is NOT done / open

- Swap the repair oracle to raw `orient3d` (consistent + hemisphere-safe).
- Whole-diagram never-worse gate (validate before/after, revert if not strictly
  better) — required before any repair can land safely (it regresses s2/300k).
- Robust boundary extraction for the non-manifold holdout (s2, cell 15514).
- Cost/scale at 1m/2m; interior-vertex f32-realizability at scale.
- The representation/splice plumbing (triple-keyed cells → the shared
  vertex-index diagram) is sound and reusable; the live-dedup / edge-reconcile
  assembly already keys vertices by triple (`VertexKey=[u32;3]`).

## Refuted / settled — do NOT re-tread

- **local_hull as a from-scratch SPLICE oracle is DEAD.** The local-hull splice
  diverges (splice_set 4→…→2917 over 40 rounds, residual climbs monotonically).
  Root cause is local_hull's INCONSISTENCY (insertion-order ties + per-gather cap
  artifacts + clustered back-faces on a sphere patch), NOT a real obstacle. Do
  not retry the local-hull source. (`route-a-splice-diverges`.)
- **"Projected is a different metric" is FALSE.** Gnomonic/stereographic/raw-3D
  are the same Delaunay in exact arithmetic; the mega defect is per-cell-chart
  rounding, not a metric difference. Do not chase "fast is not the Delaunay dual."
- **Chasing a canonical *delaunator-matching* local oracle is the wrong target.**
  delaunator matches fast only because it is single-chart/projected like fast; it
  is not canonical (pole-dependent near-cocircular answer). The canonical oracle
  is raw `orient3d`, which is pole/chart-free.
- **Surgical single-edge flip / whole-component Delaunay rebuild / neighbor
  consensus all cascade or regress on the GENERAL mega defect.** They were tried;
  the working mechanism is rim-pinned fill-with-grow with a consistent oracle.
- **Exact-everywhere is NOT mandatory** and is NOT the cheap framing's premise.
  Partial reactive escalation is viable; cost is paid only on the tiny defect set.
- **Coverage / EPS_CERT is NOT the mega lever** (termination-pad ladder: 23× more
  neighbors → byte-identical defects). mega defects are DECISION divergence.
- **High-degree merge NOT needed** (`high-degree-vertices-rationale`): mega is
  *near*-cocircular ⇒ a definite diagonal; the rare exact-cocircular case is
  handled by existing proximity-merge.
- **No cheap PROACTIVE complete flag exists** (missing-edge detection needs exact
  local topology). Use the post-assembly defect list as a perfect-recall REACTIVE
  trigger.

## Backburner (see `exact-valid-then-simplify`)

Once the exact valid graph is produced, reintroduce **epsilon-edge collapse** as a
FEATURE on the valid structure (`exact valid → inexact valid` is easy;
`inexact invalid → inexact valid` is the trap).

## Code map

- `src/knn_clipping/escalate.rs` — `gather_local`, `rebuild_cells`, `RebuiltCell`,
  `WorkingDiagram` (`malformed_cells`/`consensus_cycle`/`defect_cluster`/
  `fill_cluster_pass`), `escalate_diagram`. `pub` via the `escalate_probe`
  feature; gated by `set_escalation_enabled` (off by default).
- `src/knn_clipping/local_hull.rs` — exact engine (`build`, `faces`,
  `face_circumcenter`, `cell_faces`). Internally consistent; NOT a consistent
  cross-diagram splice oracle on near-cocircular regimes.
- `src/knn_clipping/canonical.rs` — `in_circle_sphere_sign` (the canonical raw
  `orient3d` in-circle: the recommended consistent oracle).
- `tests/escalate.rs` — the probe/slice tests + the exact-reference (delaunator)
  diagnostics. Probes: `S2_ESCALATE_*` (DIST/N/SEED/SOURCE/PROBE_*/CLUSTER_K/
  ROUNDS).
- `src/knn_clipping/p5_shadow.rs` — GO/NO-GO probes.

## Relevant memory notes

`fast-clip-is-projected-delaunay`, `route-a-splice-diverges`,
`adaptive-canonical-clip-direction`, `mega-coverage-ruled-out`,
`high-degree-vertices-rationale`, `exact-valid-then-simplify`,
`mega-regime-concluded` (superseded re: the "resolution floor" framing — see its
UPDATE note).
