# Adaptive canonical clip — design (2026-06-21)

> ## CORRECTION / UPDATE (2026-06-22)
>
> Later investigation (memory notes `fast-clip-is-projected-delaunay` and
> `route-a-splice-diverges`; probes in `tests/escalate.rs`) sharpened and partly
> overturns the framing below. Read this first; the §1–§7 analysis is kept intact
> for the record but should be read through these corrections:
>
> 1. **Projected ≡ raw Delaunay in EXACT arithmetic.** Gnomonic (per-cell clip),
>    stereographic (delaunator), and raw-3D `orient3d` all compute the SAME
>    spherical Delaunay/Voronoi exactly; differences are pure f64 precision
>    (uniform 12k: a 4-cell tail across all three). §3's "no precision floor on
>    the graph" intuition was RIGHT, but the framing changes: the residual defect
>    is **per-cell-gnomonic-CHART f64 rounding** — a vertex (g,a,b) is decided by
>    3 cells, each in its own tangent-plane chart, which round the same keep/drop
>    differently. A single global chart has 0 such defects.
> 2. **The true error is TINY everywhere (~tens of cells, 0.01–0.02%), mega
>    included** (`a0_exact_reference_delaunator`). The earlier "~77% near-
>    cocircular trip ⇒ A ≈ exact-everywhere ⇒ ~+31%/+48%" cost estimate (§6c) is a
>    red herring: near-cocircular ≠ changed; fast gets ~all near-cocircular cells
>    right. A's canonicalization target is small ⇒ A is cheap everywhere.
> 3. **§2's "primary clip, NOT a repair" argument is UNDERCUT.** §2 argues a
>    repair can't stitch because "exact meets approximate at the rim." That seam
>    does not exist for a **rim-pinned repair with a CONSISTENT oracle**: on the
>    well-conditioned rim projected ≡ raw, so fast == exact there and there is
>    nothing to stitch. Detect+repair (`detect_fix_expand_delaunator`) CONVERGES
>    in ~1 round at the tiny defect set — it does not cascade. The historical
>    "repair can't stitch / whole cap invalid" was a `local_hull` INCONSISTENCY
>    artifact (insertion-order ties + cap + clustered back-faces), not a property
>    of repair.
> 4. **A per-decision hot-path exact/flag mechanism is UNNECESSARY for the
>    valid-diagram contract.** The work is "identify cells + repair with the raw
>    `orient3d` decision," reactively, off the post-assembly defect list — not
>    exact arithmetic in the clipper. Option A (clip-time exact by construction)
>    is now demoted to a future EXACT/CANONICAL-OUTPUT feature; the recommended
>    path is reactive detect+repair (option B) with a whole-diagram never-worse
>    gate. See `docs/escalation-build-state-2026-06.md`.
> 5. **The consistent oracle is raw `orient3d` (`canonical::in_circle_sphere_sign`),
>    NOT delaunator and NOT bare `local_hull`.** delaunator matches fast only
>    because it is single-chart/projected like fast — it is pole-dependent on
>    near-cocircular ties, so it is a useful test REFERENCE but not canonical.
>    Bare `local_hull` is inconsistent (the §6c/§2 cascades trace to it).
>
> Below: original DESIGN (2026-06-21), corrected by the above.

Status: DESIGN. Supersedes the valid-or-error contract *for the mechanism it
targets*. Goal: make the per-cell clip produce a **strictly-valid diagram by
construction** across all regimes, with **no Tier-2 repair**, by sourcing the
near-degenerate keep/drop and endpoint decisions from an **exact, question-keyed
predicate** instead of from order-dependent f64 chart geometry. Adaptive:
exact cost is paid only on the <1% of decisions the fast filter cannot certify.

## 1. Root cause this targets (measured, 2026-06-21)

- The dedup key is already combinatorial: `VertexKey = [u32;3]` = sorted
  generator triple (`live_dedup/cell_output.rs:7`, `extract.rs:79`). Coordinate
  drift is NOT the failure mode.
- All residual unpaired-edge defects are **decision divergence**, not coverage:
  - `InBinThirdsMismatch` = both cells keep the shared edge, disagree on the
    endpoint triple (decision on **endpoint identity**).
  - `InBinUnconsumedCheck` = one cell keeps the edge, the other has no such edge
    (decision on **edge existence**); = asymmetric adjacency, same shape as the
    [[fma-bug-rootcause]] desymmetrization.
  - **Coverage refuted** ([[mega-coverage-ruled-out]]): the `S2_TERM_EXTRA_PAD`
    ladder forced 23× more neighbors per cell on mega 200k seed3 and produced
    **byte-identical** defects (51; 19 thirds / 17 unconsumed). Each cell already
    processes its true neighbors; it just *decides* the near-cocircular cut
    inconsistently with its neighbor.
- The decision is exactly an empty-circumcircle test. For existing vertex
  `V = circumcenter(G, M_a, M_b)` and new neighbor `M_c`, "does `M_c` clip `V`"
  ⇔ `M_c` inside circumcircle(G, M_a, M_b) ⇔ sign of
  `robust::orient3d(G, M_a, M_b, M_c)` on the raw f32→f64 generator directions
  (the lifting identity; same predicate `local_hull` already uses).

The exact predicate is a **function of the four raw points**, identical from
every cell that poses it — so it is consistent across cells *by construction*.
The f64 chart `signed_dist` is a function of the *cell's gnomonic chart*: the
same question's computed margin varies ~100× across cells
(`p5-consistency-design.md`), which is why margin-keyed escalation was unsound.
Key the decision on the **question (the 4 raw points)**, never on the chart
margin.

## 2. Why the *primary* clip, not a repair

Every reverted approach (free local hull + pin-by-key; constrained boundary;
snap-to-loop; whole-cap DT; C+M resolver) was a **repair**: an exact patch that
had to meet an **approximate exterior** at a rim. Exact and approximate disagree
*exactly at the degeneracy*, which is where the rim sits — so the patch could
not stitch (`reclip-tier2-state-2026-06.md`, `reclip-hull-snap-experiment`).

An adaptive **primary** clip has **no rim**: there is no exact/approximate
boundary, because the decision rule is the same (filter→exact) in every cell.
Consistency is pairwise-by-construction; locality is the natural Voronoi
neighborhood.

There is no precision floor on the graph. f32-promoted points are ordinary f64
points; their exact Voronoi diagram is well-defined and `orient3d` gives one
definite, consistent sign at every site (mega is *near*- not *exactly*-
cocircular: `insphere==0` never fires on mega, so even the sign is definite).
The old "f32 Voronoi-vertex resolution floor" is a floor on the vertex
**coordinate** (ill-conditioned circumcenter), which is irrelevant because we
key on the triple, not the coordinate.

## 3. The landmine (do not repeat)

A *partial* exact substitution — swapping inside/outside **signs** on the
existing approximate polygon — **erases the representation**
(`canonical-certification-design-2026-06.md:61-81`: observed bounded triangles
with all three approximate vertices "outside" per the exact predicate). The
approximate polygon's coordinates and the exact predicate's identities can
disagree, and mixing them within one cell is incoherent. **Rule: a cell is
either fully fast (f64) or fully exact — never mixed.**

## 4. Architecture: per-decision symmetric flag → per-cell exact rebuild

Fast path (unchanged, ~all cells): the f64 incremental gnomonic clip exactly as
today, including the per-vertex `vertex_planes` triple attribution. This stays
the SIMD hot path; no per-decision exact arithmetic.

Per-decision uncertainty flag (new, cheap): for each keep/drop decision, set a
per-cell `uncertain` bit when the decision is **not certifiable** by the fast
filter. The filter is the adaptive predicate's own error bound, mapped
conservatively into the chart so it is **symmetric per decision** (see §5). Cost
= one compare + OR on the already-computed margin; the bulk clip geometry is
untouched.

Exact path (new, <1% of cells): a flagged cell is rebuilt as the **exact local
Delaunay star** via `local_hull` over `{G} ∪ considered_neighbors(G)`:
`cell_faces(G)` gives the ordered fan of faces (each face = a generator triple =
a Voronoi vertex), `face_circumcenter` gives the f64 coordinate. Exact `orient3d`
⇒ order-free ⇒ every cell that shares a degenerate clique computes the identical
sub-triangulation.

Note `considered_neighbors`, NOT accepted constraints: an *unchanged* clip
returns before the neighbor is recorded (`clip.rs:108`) and termination can stop
after unchanged clips (`run.rs:513`), so a generator just outside the fast cell
but inside a degenerate circumcircle may never be in the accepted half-plane
list yet still belong to the exact clique. The exact path must retain the full
*considered* (delivered-and-tested) neighbor set for flagged cells — which the
fast path currently discards — and may need a small clique-completeness widening
(open item 3).

Output keying is unchanged: triples → `VertexKey`. The fast and exact paths
emit the *same kind* of vertex (a triple + a best-effort f64 coordinate); the
coordinate need not match across cells because pairing is on the triple.

## 5. Correctness condition — the SUPERSET PROPERTY (revised after review)

The first draft rested on "per-decision *symmetric flagging*" — any decision
exact-uncertain from either cell is flagged from both. Review (codex, 2026-06-21)
showed that is the wrong/weak framing: a band on the **chart margin** is a
calibrated proxy, not structurally symmetric (margins vary ~100× across charts,
and the chart margin is evaluated at the *drifted* vertex coordinate, not the
exact circumcenter of its triple). The correct, checkable condition is:

> **Superset property.** For every keep/drop decision, if `|chart_margin| > BAND`
> then the exact in-circle sign is *certain* and *equal to the chart sign*.
> Equivalently: `{ decisions where the fast sign is wrong OR the exact sign is a
> tie }  ⊆  { |chart_margin| ≤ BAND }`.

Two facts make this the right axis:

1. The exact decision is **drift-free and structurally symmetric**: it is
   `canonical::in_circle_sphere_sign(G, P_a, P_b, h)` on the **raw** generator
   directions (`canonical.rs:37`, Shewchuk adaptive `orient3d`, permutation-
   coherent), keyed by the vertex's *triple identity* `(G,P_a,P_b)` — which the
   fast path already tracks in `vertex_planes` — **not** by the drifted `(u,v)`
   coordinate. Every cell posing the same 4-point question gets the same sign.
2. Under the superset property, **asymmetric tiering is safe** — we do *not*
   need both cells to flag. If cell G is fast on a decision (`|margin_G|>BAND`),
   the property says G's sign is *correct*; if neighbor M is exact on the same
   decision, M computes that same correct sign. They agree regardless of which
   tier each took.

Lemma (no unpaired interior edge), per shared edge G–M with endpoints
`{G,M,X},{G,M,Y}`:
- M **unflagged** ⇒ *all* of M's decisions had `|margin|>BAND` ⇒ by the superset
  property M's fast answers are all correct, **including the in-circle tests that
  fix the endpoint thirds X,Y** (not just edge existence). Exact-G recomputes the
  same correct X,Y. Paired.
- M **flagged** ⇒ M takes the exact path ⇒ identical raw `orient3d` triples to
  exact-G. Paired.
This closes the review's "well-conditioned edge ≠ same endpoint triples" gap:
the unit of the property is the **in-circle decision per candidate vertex**
(existence *and* third), not the edge.

**The superset property is the entire GO/NO-GO.** It is a forward-error question
— does a chart-margin BAND exist that provably (or empirically, across regimes)
contains every wrong-fast-sign and every exact-tie, while tripping <1%? It must
absorb chart distortion **and** accumulated lerp/vertex drift. If no such tight
band exists, the SIMD prefilter cannot stand and the decision itself must be the
(more expensive) raw-predicate filter — a real cost change, not "one compare".

## 6. Open items / calibration (in priority order)

1. **Superset BAND + trip rate (GO/NO-GO).** For every keep/drop decision, log
   both the chart margin and the exact `in_circle_sphere_sign` (existing
   predicate, `canonical.rs:37`). Find the smallest BAND with the **superset
   property** (§5): `|margin|>BAND ⇒ chart_sign == exact_sign ≠ 0`. Measure its
   trip rate across all regimes (uniform, mega, cube, grid, cocircular, bimodal,
   gradient). Decision: a BAND with the superset property AND <1% trip ⇒ the
   cheap SIMD prefilter stands, build proceeds. If the minimal superset BAND
   trips ≫1% (chart distortion + lerp drift too large), the SIMD prefilter is
   not viable; fall back to the raw-predicate decision (cost reckoning required).
2. **`local_hull` is not yet an all-regimes oracle.** Beyond the tie policy
   (`orient3d == 0` punted to insertion order, `local_hull.rs:16`; visibility
   treats `orient==0` as non-visible, `:100`; all-coplanar ⇒ `None`, `:55`),
   `cell_faces` assumes a single clean fan and returns empty otherwise (`:185`)
   — a high-degree / non-fan dual would silently drop the cell. All-regimes
   (cube/grid/cocircular carry exact ties) needs either an index-keyed tie rule
   *and* a non-fan `cell_faces`, or an explicit **high-degree-vertex merge**
   (one vid for the clique; the crate already tolerates high-degree vertices,
   `delaunay.rs`). Merge is the safer manifold-preserving choice; index-keyed
   diagonals fabricate arbitrary edges on exact cocircular cliques.
3. **Clique completeness of the considered set.** §4: the exact star needs every
   clique generator in `considered_neighbors(G)`. Audit flagged cells for the
   case where the exact answer needs a generator the fast clip neither accepted
   nor (after termination) tested; widen the retained set if so.
4. **Cost.** Per-decision compare+OR for the flag is negligible **iff** item 1
   succeeds (SIMD prefilter stands). Exact rebuild is `local_hull` O(k²), k≈tens,
   on <1% of cells. If item 1 forces a raw-predicate decision, re-cost the hot
   path (scalar `orient3d` filter + plane gathers, vs SIMD `signed_dist`).
5. **Validation.** Add a flagged-cell parity test (exact rebuild agrees with f64
   on well-conditioned cells) and full `validation::validate` over every regime
   in the robustness campaign.

## 6b. GO/NO-GO measured (2026-06-21) — VERDICT: GO, flag reframed

Instrumented `p5_shadow` (`set_audit_cutoff`, `TIE_HIST`, superset summary;
tests `probe_superset_band` / `probe_superset_paired` / `probe_mega_cutoff_sweep`).
Findings:

- The **single-cell** chart-vs-exact disagreement tail is ~1e-2 and ~99.9%
  **benign**: uniform/cube/bimodal carry thousands of such disagreements and
  ZERO defects. So "superset of all single-cell disagreements" (the original §5
  target) is the WRONG, pessimistic requirement. Drift is real but cross-cell-
  correlated.
- The defect-causing quantity is **cross-cell conflict** (two cells answering
  the same 4-point question differently). Its margin tail is **≤1.3e-10**:
  uniform 200k → 1 contradiction @2.9e-11; grid 150k → 7 @1.3e-10; clustered →
  0. So a chart-margin flag with band ~1e-9 (headroom above the tail) catches
  the defect-causing cells at **~1e-6 trip** — the cheap flag IS viable for the
  random regime.
- **Three mechanisms**, not one:
  1. random/uniform = same-question **sign conflict** at ~1e-10 → tight
     chart-margin flag.
  2. grid/structured (3907 defects) = exact **ties** (canonical==0, cocircular
     cubed-sphere lattice) → needs the tie policy / **high-degree-vertex merge**,
     NOT a margin band.
  3. mega/dense = **question-set divergence** — ZERO same-question contradictions
     even at cutoff 1e-2 (8.86M quads) → the two cells never pose the same
     question. The flag must be a per-cell **near-cocircular detector** (small
     in-circle margin anywhere in the star); mega's degeneracies sit at the f32
     scale (~1e-4), so its band is coarse and fires on much of the dense cap →
     mega rebuilds a large fraction = slow but CORRECT (acceptable on a
     pathological input; normal inputs stay ~1e-6 trip).

Consequences for §5/§4:
- The flag's job is **recall on cross-cell-divergent cells**, not a superset of
  single-cell disagreements. Recall is structural: both cells of a divergent
  pair are incident to the same near-cocircular cluster, so both fire.
- **Correctness is carried by the exact REBUILD, not the flag's precision.** The
  flag only decides *which* cells rebuild; `local_hull` makes them consistent.
- Residual risk (codex): the band's symmetry is **empirical** (8x headroom over
  the observed tail), not proven — firm up across more seeds/sizes before trust.
- Cost: ~1e-6 cell-trip on normal regimes (negligible); high on mega (bounded,
  pathological). The per-decision flag can stay the cheap chart margin; it does
  NOT need a per-decision exact predicate.

## 6c. Escalation viability + cost (2026-06-22)

Two independent adversarial reviews (one arguing partial escalation is seam-prone
NO-GO, one arguing it is viable via a well-conditioned rim) were adjudicated
blind. **Verdict: partial escalation is VIABLE; exact-everywhere is NOT
mandatory.** The "fundamental seam" claim is too strong — on a genuinely
well-conditioned shared edge fast == exact (they answer the same in-circle
questions), and the cross-cell conflict tail is ~1e-10, so escalating one side
creates no new defect at well-conditioned edges.

The condition is STRICT (this is the real spec):
1. Escalate the WHOLE near-degenerate / question-divergent **component**, not
   single cells.
2. With a **complete considered-neighbor set** (unchanged clips return before
   recording the neighbor, `clip.rs:108`; termination can stop after unchanged
   clips, `run.rs:513`) — so the rebuild must retain considered, not accepted,
   neighbors, possibly widened.
3. The rim must certify **question-set CLOSURE** (edge-existence + endpoint-
   thirds), not merely that visible final edges look non-degenerate — because
   the hard class is question-set divergence (one cell never poses the other's
   question). Grow until the rim's fast decisions are certified == exact, OR
   validation proves no residual seam.
4. Handle exact ties / high-degree vertices (`local_hull` punts today).
5. A cheap PROACTIVE complete flag is impossible (missing edges need exact local
   topology), BUT the **post-assembly defect list** (`unresolved_edge_pairs`,
   `compute.rs:146-199`) is a perfect-recall REACTIVE trigger. So the pipeline
   is defect-driven: detect residuals -> escalate component -> rebuild exact ->
   validate -> grow until no residual. This is the reverted grow-until-clean
   pipeline with an EXACT-CLIP interior engine replacing the crude fill that
   failed.

Cost (measured, `probe_predicate_cost`): the dominant overhead is one adaptive
`orient3d` in-circle per decision instead of SIMD `signed_dist`. Isolated cost
on uniform 200k = 228ms for 2 orient3d/decision (~114ms with d2 caching) against
a 363ms production build => **exact-EVERYWHERE ~ +31% (uniform), ~+48% (mega)**
before adding circumcenter-vs-lerp. So even the everywhere upper bound is
~1.3-1.5x, not catastrophic. **Escalation pays this only on the degenerate
component** => negligible on normal inputs, bounded (~whole dense cap) on mega.
Cost does not block either approach; escalation is the cheaper target.

## 7. First step

Item 1 only — the **superset-BAND measurement**: instrument the per-decision
(chart margin, exact in-circle sign) pair, sweep BAND, report the minimal
superset BAND and its trip rate across all regimes. No hot-path change, no exact
path yet. That single measurement decides whether the cheap-prefilter form of
the adaptive primary clip is viable before any build — the same discipline that
refuted EPS_CERT. It also directly answers the review's central doubt (does a
tight symmetric band even exist given ~100× chart variance + lerp drift).
