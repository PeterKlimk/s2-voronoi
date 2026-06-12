# P5: Consistency by Construction — Design

Status: **design accepted, not started** (2026-06-12). This document supersedes the
two-paragraph P5 sketch in `docs/todo.md` and records the design discussion that
produced it. Sequencing note: originally slated post-release; now planned
pre-release at the maintainer's call.

## Goal

Make every shared combinatorial decision — "does the edge between cells g and h
exist?", "which generator triple identifies this corner?" — canonical: decided
once, by a computation every involved cell reproduces bit-for-bit. Graph validity
then no longer depends on epsilon agreement between two charts; one proves
*determinism* instead of error bounds.

What this is **not**: exact arithmetic or a platonic Voronoi diagram. A canonical
decision can still be geometrically wrong at epsilon scale — but it is
*identically* wrong for every cell, which is all the "essentially Voronoi" graph
contract requires. The iffy math survives; only its decisions gain a consistency
guarantee.

### Exit criteria (hard)

1. **Defects vanish**: `tests/edge_repair_net.rs` finds **zero** unresolved edges
   on its fixtures at every bin count (the net's defect-presence assertions get
   inverted — the fixture becomes a regression test that the disagreement class
   stays dead). `edge_reconcile` and both repair backends are then deleted;
   unresolved-edge *detection* remains as a bug trap that fails the computation
   instead of routing to repair.
2. **Throughput holds**: paired `bench_build.sh`/`bench_run.sh` runs show ≤1% ST
   regression at 500k and 2M. This criterion can **reject the whole project**;
   the fallback position is the current (post-2026-06) status quo: tested
   cross-bin detection, O(defects) in-place repair, observable defect origins.
3. The epsilon-caveat paragraph of `docs/live_dedup.md` is deleted, and the
   contract language upgrades from "empirically safe with 8x margin" to
   "consistent by construction".

## Why naive canonicalization fails: two named failure modes

### Failure mode A: frame-dependent evaluation

Each cell clips in its own gnomonic chart (anchored at its generator) with
chart-local f32/f64 arithmetic and incrementally lerped intersection points.
The same geometric predicate evaluated in two charts rounds differently; at
epsilon scale the decisions diverge. This is the textbook failure the original
P5 sketch targeted.

Naive fix — evaluate everything in one frame — destroys the hot path: the
chart-local SoA batches, nearest-first incremental clipping, and the kernel-
contest wins all live on chart-local coordinates, and decisions fed by lerped
intermediates cannot be made canonical without recomputing from generator
coordinates. Estimated cost of the naive form: double-digit percent.

### Failure mode B: question-set divergence (the harder one)

Canonical *answers* are not enough — the cells must agree on which *questions*
get asked. Each cell's combinatorial structure is the outcome of decisions
against **its own delivered generator set**, and the two streams terminate at
different points (the termination certificate is itself an epsilon-influenced
decision). If generator x is marginally relevant to a shared corner and h's
stream delivered x while g's certificate said "nothing unseen can matter", g
never poses the question x answers. No canonical evaluator fixes a question one
side never asked.

This mechanism — not chart rounding — plausibly produces a share of today's
real defects: the edge-repair fixture's defects die under input-order reversal
(which changes seed-forwarding direction and hence effective constraint sets),
pointing at seen-set asymmetry rather than pure arithmetic divergence.

## The design: three pillars

### Pillar 1 — the canonical primitive is the symmetric in-circle predicate

The shared decision about a candidate corner v = vertex (g, h, t) is "does any
generator x kill v". From g's side this is d(v,g) <= d(v,x); from h's side
d(v,h) <= d(v,x); these are the same question because d(v,g) = d(v,h) = d(v,t)
by definition of the circumcenter:

    in_circle(g, h, t; x)  —  is x inside the circumcircle of (g, h, t)?

This is a function of **four generator coordinates only** — no chart, no frame
choice, symmetric in (g,h,t). It dissolves the frame-selection question
entirely (no "lower-index generator's chart", no world-space expression zoo):
all parties evaluate one deterministic function of the same four inputs and get
the same bits. Determinism, not exactness: the evaluator is a fixed f64
expression; its sign can be geometrically wrong at epsilon scale, identically
for everyone.

Edge existence follows: edge (g,h) exists iff some t survives, i.e. the
canonical structure of every cell is determined by the in_circle answers alone.

### Pillar 2 — filtered escalation keeps the hot path

Decisions with large margins are frame-independent for free: if g's chart says
"inside by 1e-3" and the maximum cross-chart deviation for that predicate class
is ~1e-7, every frame agrees, no canonical evaluation needed. So:

- The hot path is unchanged chart-local clipping, plus **one extra compare per
  clip lane**: `|signed distance| > EPS_FILTER`. The signed distance is already
  computed (it drives today's `CLIP_EPS_INSIDE` test), so the compare is
  near-free and SIMD-compatible.
- Lanes below the filter — the same near-tie rarity class that produces today's
  1-20 defects per multi-million points — escalate to the canonical in_circle
  evaluation, expressed on generator tuples (a clip vertex is the intersection
  of bisector(g,a) and bisector(g,b); its sidedness against bisector(g,h) is
  in_circle(g, a, b; h)).
- **Soundness condition**: EPS_FILTER must dominate (cross-chart deviation) +
  (canonical evaluator's own rounding), so that *clearing the filter locally
  implies the local decision equals the canonical one*. Then it does not matter
  that one side escalated and the other did not — they agree regardless. This
  inequality is the load-bearing proof obligation; the coincidence-probe
  infrastructure (`tests/coincidence_probes.rs`) exists to measure exactly
  these deviation scales.

When an escalated decision disagrees with the local f32 coordinates, the
**decision is authoritative over the geometry**: the polygon update follows the
canonical answer (the combinatorial structure is the contract; coordinates are
representation). This touches the clipper's transition/extraction logic
(`topo2d`) and is the main implementation risk besides the error analysis.

### Pillar 3 — question-set closure (resolves failure mode B)

The termination certificate bounds d(v, unseen x) against the polygon's vertex
radius — and the in-circle margin *is* that distance gap. So the closure
condition is quantitative:

    EPS_CERT > EPS_FILTER

where EPS_CERT is the certificate's conservatism guard: every unseen generator
is outside every surviving vertex's circumcircle **by at least EPS_CERT**,
proven against the *computed* (f32) vertex positions, not platonic ones. Then:

    unseen  ⟹  not-in-circle with margin > EPS_CERT > EPS_FILTER
            ⟹  the question is never marginal
            ⟹  provably irrelevant to every shared decision.

The cells still see different generators — that is normal and stays — but every
generator in the symmetric difference has a clear-margin "irrelevant" answer
that both sides would compute identically had they asked. Question-set
divergence becomes harmless by construction.

The qualitative half is already test-pinned: the P3.1 NN contract suite asserts
"every frontier certificate conservatively bounds every unseen eligible point".
P5 adds the quantitative obligation (guard ≥ EPS_CERT including computed-vertex
error) on top of the existing guards (`TERMINATION_ANGLE_PAD`,
`TERMINATION_THRESHOLD_GUARD`, `GRID_SIN_EPS` in `src/tolerances.rs`).

Perf consequence: certificates conservative by EPS_CERT terminate slightly
later — marginal cells see a few extra candidates. Expected cheap (the guard
widens circles by ~1e-6-scale quantities against ~1e-3 cell radii; mean
neighbors-before-termination is ~8.4 today), but it is the third thing the
paired benchmarks gate, alongside the filter compare and the escalation path.

## What P5 does and does not replace

| Component | Fate under P5 |
| --- | --- |
| `edge_reconcile` + both repair backends | **Deleted** (exit criterion 1) |
| `UnresolvedEdgeMismatch` production | Kept as **bug trap**: under P5 an unresolved edge proves broken determinism; it fails the computation (or debug-asserts) instead of routing to repair |
| Edge checks / seed forwarding / overflow matching | **Kept** — this is vertex-index agreement (dedup machinery), not repair |
| The weld (`PreprocessMode::Weld`) | **Kept** — coincident input is a data problem, not a consistency problem |
| `tests/edge_repair_net.rs` | Kept, assertions inverted: fixture must produce **zero** defects at every bin count |
| Epsilon caveat in `docs/live_dedup.md` | Deleted |

## Proof obligations (the error-analysis work)

1. **Cross-chart deviation bound**: for the clip sidedness predicate, bound the
   difference between its value in any cell's chart and the canonical
   in_circle sign, as a function of input magnitudes. Empirical estimation via
   the coincidence probes; the bound feeds EPS_FILTER.
2. **Canonical evaluator error**: rounding of the fixed f64 in_circle
   expression. Standard forward-error analysis.
3. **Certificate guard chain**: certificate distance bounds hold against
   computed vertex positions with slack ≥ EPS_CERT; folds the f32 vertex
   position error into the existing guard constants.
4. **The inequality**: EPS_CERT > EPS_FILTER > (1) + (2). All constants land in
   `src/tolerances.rs` with their justifications, per house style.

## Migration plan

Staged, each stage behind the existing harnesses (NN contract suite, edge-repair
net, fingerprint tests, paired benches):

1. **Canonical evaluator + escalation plumbing, decisions logged not applied.**
   Run the filter in shadow mode: count escalations, compare canonical answers
   against the local decisions actually taken, on the net fixtures and fuzz
   sweeps. This measures (a) escalation frequency (perf model) and (b) how often
   today's path disagrees with canonical (defect-mechanism confirmation) before
   any behavior changes.
2. **Apply canonical decisions in the clipper** (decision-authoritative polygon
   updates). Differential oracle: old pipeline vs new on the net fixtures and
   fuzz sweeps — defects must go to zero; everything else identical within the
   decision changes. Fingerprint moves here; tolerances re-validated
   (coincidence probes re-run, per the `tolerances.rs` protocol).
3. **Tighten certificates to EPS_CERT** and prove/measure the closure
   inequality. NN contract suite extended with the quantitative guard check.
4. **Invert the net's assertions; demote detection to bug trap; delete
   edge_reconcile** after a soak period (fuzz sweeps at 2-4.5M green with
   detection-as-error).
5. Paired benches gate stages 2-4; criterion 2 rejects at any point.

## Assets this design leans on (all 2026-06)

- `tests/edge_repair_net.rs`: deterministic defect fixture + origin-tagged
  detection — the instrument that measures success (defects → 0 at every bin
  count), and the per-origin tags identify *which* disagreement class survives
  an incomplete canonicalization.
- The differential-oracle pattern from the O(defects) repair work (run both
  pipelines, compare per-cell sequences) — reused for stage 2.
- The bin-layout finding (engineering-findings #13): the defect set is not
  bin-invariant under the current two-chart evaluation (seed-replay vs EmitAll
  paths differ at epsilon scale). Under P5 it must be empty — hence trivially
  bin-invariant — *regardless of layout*; the net's bin-count matrix checks
  exactly this.
- The NN contract suite (P3.1) as the carrier for the certificate-guard
  obligations.

## Rejected alternatives

- **Naive full canonicalization** (single frame for all evaluation): destroys
  the chart-local SoA hot path; estimated double-digit regression. Rejected on
  exit criterion 2.
- **Exact predicates** (CGAL-style exact arithmetic fallback): more than the
  contract needs — "essentially Voronoi" requires agreement, not truth; the
  deterministic-shared-evaluator fallback is strictly cheaper with the same
  graph guarantee.
- **Lower-index-generator's-chart as canonical frame**: dominated by the
  symmetric in_circle formulation, which needs no frame choice at all and is
  insensitive to which side evaluates it.
- **Live within-bin edge repair** (ledger): a second repair phase with its own
  composition proof, made moot by removing the mismatches at the source. Already
  marked deprioritized in the ledger.
