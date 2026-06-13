# P5: Consistency by Construction — Design

Status (2026-06-12): **stages 0-1 done; stages 2-4 kept on the roadmap**
(maintainer decision after the post-stage-0 defect survey: the remaining
payload is ~1 repairable site per ~3 multi-million runs at 3M+, and the
contract upgrade is still wanted). Next P5 action: the paired two-cell
shadow audit (compare both cells' local decisions for the same 4-tuple) to
place EPS_FILTER — the existing shadow histogram conflates
platonic-vs-computed deviation with chart-vs-chart divergence. This
document supersedes the two-paragraph P5 sketch in `docs/todo.md`.

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

**Both repair classes are one configuration.** A tiny edge (g,h) runs between
vertices (g,h,a) and (g,h,b) and its length goes to zero exactly as the
quadruple (g,h,a,b) approaches cocircularity — at exact cocircularity the
endpoints merge into a degree-4 vertex. The signed distance the filter tests
*is* the in-circle margin of that quadruple, so degeneracy and small margin are
the same condition: dangerous configurations cannot present with comfortable
margins, and the filter catches the 4+-cocircular vertex and the epsilon edge
as one case.

**Evaluator implementation (canonical ordering, exactness, tie coherence):**

- *Canonical ordering*: different cells phrase the same 4-point question
  differently (cell g asks in_circle(g,h,a;b); cell a asks in_circle(g,a,b;h))
  — all are the same determinant up to row permutation and orientation. The
  evaluator computes **one determinant in sorted-index row order**; call sites
  apply the permutation parity. This gives cross-question consistency within a
  quadruple for free, not just per-question agreement.
- *Exact signs*: on the unit sphere in_circle(g,h,t;x) reduces to
  **orient3d(g,h,t,x)** (the circumcircle is the sphere's intersection with
  the plane through g,h,t); the planar pipelines use the standard 2D incircle.
  The escalation evaluator computes these signs **exactly** via Shewchuk-style
  adaptive arithmetic (the `robust` crate, or vendored — f32 inputs are
  exactly representable in f64, the easiest case). Exactness is confined to
  the cold escalation path; it makes the agreed structure the *true* topology
  whenever the determinant is nonzero, and it zeroes the evaluator-error term
  in the EPS_FILTER inequality.
- *Exact ties (true cocircularity)*: resolved by **simulation of simplicity**
  (Edelsbrunner–Mücke): a fixed-order cascade of exact sub-determinant signs,
  equivalent to an infinitesimal index-ordered perturbation of the input. The
  perturbation order is the **effective (post-weld) generator index** (the
  stable global order all predicates run on). Coherence guarantee: the
  tie-broken structure is the exact Voronoi topology of an actually-perturbed
  configuration — valid for arbitrarily deep ties (5+ cocircular) by
  construction, not by validator luck. Degenerate inputs then yield
  zero-length edges / zero-area slivers, which the "essentially Voronoi"
  contract already tolerates as representation.

Without the exact+SoS evaluator, deterministic-but-rounded f64 signs guarantee
*agreement* but not *coherence* (the agreed tie-breaks need not correspond to
any perturbed configuration). That weaker evaluator is acceptable only as a
stage-1 shadow-mode stopgap; the implemented escalation path is exact+SoS.

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
2. **Canonical evaluator error**: zero — the escalation evaluator is exact
   (adaptive orient3d/incircle + SoS), which loosens the EPS_FILTER bound to
   "dominate cross-chart deviation" alone.
3. **Certificate guard chain**: certificate distance bounds hold against
   computed vertex positions with slack ≥ EPS_CERT; folds the f32 vertex
   position error into the existing guard constants.
4. **The inequality**: EPS_CERT > EPS_FILTER > (1) + (2). All constants land in
   `src/tolerances.rs` with their justifications, per house style.

## Stage-1 shadow-mode findings (2026-06)

The shadow audit (`feature p5_shadow`: exact canonical evaluator +
margin/disagreement histograms in the gnomonic clip path, zero behavior
change; probe in `tests/p5_shadow.rs`) measured on uniform 2M:

- **Escalation frequency** at normalized-margin cutoffs: ~2.4% of decisions
  below 1e-4, ~0.24% below 1e-5, ~0.02% below 1e-6 (roughly linear in the
  threshold). Exact ties: zero on uniform data.
- **A dominant inconsistency source nobody had quantified**: ~8% of
  sub-cutoff decisions disagree with the raw-input canonical answer,
  concentrated at margins 1e-5..1e-4 — far above rounding scale. Cause: the
  builder f64-renormalizes the generator (`reset`) while neighbors enter as
  raw f32, so the pipeline solves a slightly different point set than the
  raw input (f32 points are ~6e-8 off-unit; against |g-h| ~ 2.5e-3 at 2M
  that is exactly the observed ~2e-5 band). Cell errors are highly
  correlated (only 3 cross-cell defects on the same run), but canonical
  decisions must be about the same points the charts consume.

Consequence — **stage 0, input canonicalization**: renormalize all points
once at pipeline entry (f64-normalize, round back to f32) and remove the
per-builder renormalization, so every consumer (charts, canonical
predicates, certificates) sees identical bits. After stage 0 the residual
computed-vs-canonical deviation is genuine chart/lerp rounding; re-measure
the margin histogram to place EPS_FILTER (expected ~1e-6, where escalation
cost is negligible).

## Paired two-cell audit findings (stage-2 prerequisite, 2026-06)

The paired audit (`p5_shadow::paired_report`, probe in `tests/p5_shadow.rs`)
keys every near-margin decision by its abstract question — (sorted triple,
opposing generator x) — and groups records across cells, measuring how often
distinct cells answer the SAME question with conflicting local signs.
Measured at 500k (cutoff 1e-3), 2M, and defect-bearing 3M seed 3 (cutoff
1e-4):

1. **The cross-cell conflict tail is ZERO — now rigorous, not censored.**
   300k-557k multi-party questions per run; not one conflicting pair of
   local answers — including on the run with 4 real defects. The follow-up
   two-pass audit (`probe_two_pass_audit`: pass 1 collects question keys
   below the margin cutoff; pass 2 re-runs the same input recording every
   party's answer to those keys at ANY margin) closed the censoring hole:
   it de-censored 167k-313k question pairs per run whose partner had
   answered above the cutoff, and the conflict count stayed **zero** across
   ~5M questions total. Chart-local errors on shared questions are
   near-perfectly correlated. (The canonical evaluator's permutation
   self-check also held: zero inconsistencies.)

   The two-pass split also quantifies how normal question-set divergence
   is: 60-65% of near-tie questions are *truly one-sided* (only one of the
   triple's three owners ever posed them, at any margin) — divergence is
   the default at epsilon scale and almost always benign; defects are the
   rare cases where it touches a feature two cells must agree on.
2. **Local-vs-canonical disagreements are benign for consistency**: tens of
   thousands per run (the 1e-5..1e-4 band), but they are shared bias, not
   conflict — wrong-but-identical answers yield a consistent
   (epsilon-wrong) structure, which is all the contract needs.
3. **Defects are question-set divergence, period** (failure mode B,
   confirmed as the sole observed mechanism): the defect-site anatomy dump
   (`paired_dump_involving`) shows agreeing, canonical-consistent answers
   everywhere — the mismatch lives in questions only one side ever posed
   (corner never formed in the other cell's polygon evolution, or x never
   delivered) and in divergent triple identities (thirds mismatches are
   different question SETS by definition).

### Correction: the quad-coherence analysis (supersedes the paragraph below)

The (triple, x)-keyed pairing had a canonicalization gap: for a 4-set
{g,h,t1,t2}, the phrasings "(g,h,t1) vs t2" and "(g,h,t2) vs t1" are the
SAME determinant with opposite parity — two cells keeping parity-
incompatible corners is a genuine contradiction the keying could never
pair. Regrouping the same records by sorted 4-set (`paired_quad_report`,
coherence rule: all records in a quad must agree with canonical or all
disagree) found **exactly one contradiction per defect-bearing run, and it
is the defect site's quad** — fixture_2k: [25,63,287,322]; 3M seed 3:
[1790353,2327897,2902347,2992988]; clean 500k: zero. Perfect 1:1.

The contradiction margins are **1e-13..1e-11** — the CLIP_EPS_INSIDE
(1e-12) scale, where the eps-inside tie-break itself straddles — far below
the benign 1e-5..1e-4 shared-bias band.

Meanwhile the gate-1 experiment (termination-pad sweep via
`set_term_pad_override`, pads up to 1e-4 = 100x default, on every known
defect-bearing input) changed **nothing**: identical defect sets at every
pad. Delivery was never the mechanism.

**Net design conclusion (replacing the superseded paragraph below):**

- The observed defect mechanism is **failure mode A** — answer divergence
  on opposite-parity phrasings of near-exactly-cocircular quads — not
  failure mode B. (Mode B remains a designed-for possibility; it is just
  not what produces today's defects.)
- **Pillars 1-2 are the fix**, with economics far better than feared:
  EPS_FILTER ~ 1e-8 normalized (three orders above the observed
  contradiction tail) escalates ~0.003% of decisions. The evaluator's
  parity-canonical form (sorted-row determinant + call-site parity) is
  exactly what makes the two phrasings coherent.
- **Pillar 3 certificate-conservatism work is unnecessary** for the
  observed defects (pad sweep negative). EPS_CERT survives only as a
  small proof obligation: termination must not hide a question at
  EPS_FILTER-marginal scale, which the existing pads already dominate by
  orders of magnitude (filter 1e-8 vs pads ~1e-6).
- Stage 2 is therefore the decision-authoritative clipper escalation as
  originally designed, at EPS_FILTER ~ 1e-8.

*(Superseded paragraph, kept for the record:)* Consequence for the design:
pillar 3 is the load-bearing pillar. Pillars 1-2 (canonical answers behind
a filter) would NOT have fixed the observed 3M defects — those answers
already matched canonical. [...] This conclusion was an artifact of the
under-canonicalized pairing keys described above.

## Stage-2a findings (2026-06): margin-gated escalation is unsound

Stage 2a was implemented in full — `EscalationCtx` threaded through the
clippers, near-margin lanes re-decided by the exact canonical predicate
(`knn_clipping::canonical`, now a production module; `robust` is a hard
dependency), guarded lerps for flipped decisions — and rejected by its own
gates. The escalation-band sweep (probe-overridable factor, band =
factor x CLIP_EPS_INSIDE x |n|):

| factor | fixture_2k | 500k | 3M s3 | 4.5M s2 |
|---|---|---|---|---|
| 0 (off) | 4 | 0 | 4 | 3 |
| 2 | 4 | 0 | 0 | - |
| 32-128 | 0 | 0 | 0 | 6-11 |
| 256-4096 | 0 | 0-3 | 4-64 | - |
| 1e4 (= the planned 1e-8 filter) | 0 | **10** | **200** | - |

The wide filter manufactures defects wholesale; the narrow band fixes the
watched sites but elevates 4.5M at every width. Diagnosis: the gate is
evaluated per cell on **displaced computed margins** — the same question's
margins differ by ~100x across cells (lerp-chain displacement), and local
answers below ~1e-4 are ~50% anti-correlated with canonical while
near-perfectly correlated with each other. Today's consistency is
**algorithmic correlation**, not geometric accuracy; mixing canonical
answers into it converts shared bias into cross-system contradiction at
the band boundary, at a rate that scales with n. No margin threshold is
sound, because the margin is not a function of the question.

Status: the clipper machinery is kept, **disabled in production**
(CLIP_ESCALATION_FACTOR = 0.0), overridable via
`p5_shadow::set_escalation_factor_override` as the test rig for successor
designs. Successor candidates, in current preference order:

1. **Antisymmetric tie rule**: the observed natural contradictions are
   caused specifically by the keep-bias `d >= -eps` (both opposite-parity
   phrasings keep when |d| < eps). A parity-coherent local rule (strict
   `d >= 0`, or an index-deterministic tie-break) would fix the observed
   mechanism with NO canonical/local mixing — the correlation story stays
   intact. Needs: why the eps bias exists (sliver/chatter guards, seed
   replay tolerance), and a fingerprint-moving revalidation.
2. **Question-intrinsic gating** = full adaptive canonicalization: every
   decision evaluated by the (cheap stage A of the) exact predicate on raw
   generator bits — the gate becomes a pure function of the question, so
   no mixing boundary exists. This is the "naive canonicalization" cost
   problem (per-lane gathers, ~10x the flops of the current distance
   test); would need a cost-feasibility spike before being taken
   seriously.
3. **Accept the status quo**: ~1 repairable site per ~3 multi-million
   runs, O(defects) repair, machine-checked validity. The standing
   fallback, now with precise knowledge of the residual mechanism.

## Tie-rule findings (2026-06): candidate 1 measured; strict rule adopted

Candidate 1 was implemented as a probe (`set_clip_eps_override`: the eps
override replaces CLIP_EPS_INSIDE at `HalfPlane` construction; 0.0 =
strict `d >= 0`. The edgecheck eps-reuse path degrades cleanly — a
forwarded eps of 0 routes through ordinary construction) and swept over
the defect battery (`probe_tie_rule_sweep`):

| clip eps | fixture_2k | 500k s2 | 2M s1 | 3M s3 | 4.5M s2 |
|---|---|---|---|---|---|
| 1e-12 (old default) | 4 | 0 | 0 | 4 | 3 |
| 0.0 (strict) | 4 | 0 | 0 | **0** | 3 |
| 1e-14 | 4 | 0 | 0 | 0 | 3 |
| 1e-13 | 4 | 0 | 0 | 0 | 3 |

All runs strictly valid; wall times unchanged. Strict-rule quad reports:
3M s3 shows **0 contradictions across 8.0M multi-record quads** — the
cross-chart error correlation really does deliver antisymmetry at scale —
while fixture_2k's single contradiction survives bit-identically.

The observed contradiction band (1e-13..1e-11) therefore splits into two
regimes:

- **Tie regime** (margins <= eps): both opposite-parity phrasings kept by
  the `d >= -eps` bias. The strict rule eliminates these (3M s3: 4 -> 0;
  any eps below ~1e-13 suffices).
- **Error regime** (margins > eps): the computed d itself carries the
  wrong sign in one chart (fixture_2k: disagreeing record at +1.51e-11
  where canonical says cut; 4.5M s2: the same phrasing answered
  oppositely by two cells at ~4e-11 — the correlation tail itself). No
  local rule can fix these, and stage 2a showed canonical mixing makes
  them worse. Edge repair owns them.

4.5M s2 anatomy (`probe_45m_quad_anatomy`; previously uncharacterized):
its 3 defects trace to exactly one parity-contradiction quad
[749399, 1899262, 3667055, 4269011] at margins 3.5e-11..4.6e-11,
bit-identical under default and strict rules — pure error regime. The
1:1 defect-site:quad-contradiction correspondence now holds on all three
carriers.

Also closed: why the eps bias existed. Git history traces `d >= -eps` to
the pre-repo hex3 import with no recorded rationale beyond biasing
shared-edge agreement on marginal vertices; the sweep found no input
where removing it costs anything (no new defects, no validity loss, no
sliver/chatter symptoms, no wall-time change).

**Decision: CLIP_EPS_INSIDE = 0.0 in production — spherical backend
only.** Expected residual: error-regime contradictions only — the
fixture_2k/4.5M class — with O(defects) edge repair unchanged as the
backstop.

**The planar backends keep the bias** (`PLANE_CLIP_EPS_INSIDE = 1e-12`),
and the full-suite run measured why — the first recorded justification
for the bias existing at all: with strict bisectors,
`plane_larger_uniform_strict` (~50k uniform) yields 3 unpaired interior
edges and `plane_clustered_and_collinear` fails too (strict walls alone
are fine; biased bisectors restore green).

This is NOT because the plane lacks repair machinery — the planar
pipeline shares the full edge-check/detection/reconcile stack (a first
draft of this section claimed otherwise; corrected). The planar anatomy
probe (`probe_plane_strict_anatomy`: plane eps override, plane-side
paired audit with exact planar in-circle, detected-unresolved collector,
external unpaired-edge recomputation) then measured the mechanism, and
it is NOT the sphere's:

- **Zero quad contradictions on both failing fixtures** (4.5k / 130k
  quads audited). The plane's strict failures are not in-circle parity
  contradictions; the tie/error-regime split does not transfer.
- **The defects are vertex-identity slivers**: the same geometric corner
  committed under different triple attributions in different cells —
  duplicate vertex ids at bit-identical coordinates (e.g. two ids both
  at (0.510638, 0.507183)), zero-length sliver edges between them,
  shifted vertex sequences along shared edges. This is the
  thirds-mismatch family — the same class as the sphere's surviving
  error-regime defects — at a density orders beyond the sphere's
  (~1 site per 3 multi-million runs vs 32 sites in 402 cells).
- **The bias is the sliver guard**: keeping marginal corners in BOTH
  cells keeps vertex keys and sequences identical across an edge's two
  cells. That is exactly the "agreement" purpose the constant always
  documented; the strict experiment finally measured the class it
  suppresses.

The strict run also exposed two net weaknesses the bias was masking,
each independently valuable:

1. **Cross-bin detection escape**: every unpaired pair involving the
   fixture's two far points (whose huge cells live in different spatial
   bins than the cluster) was absent from the detected-unresolved set.
   This is the CrossBinThirdsMismatch class whose deterministic pin the
   sphere lost at stage 0 (edge_repair_net coverage gap) and whose
   repair was always "a guess" — the 402-cell fixture under
   `set_plane_clip_eps_override(0.0)` is a cheap deterministic lab
   for it.
2. **Repair fails under defect density**: detected, zero-length,
   bit-identical duplicate-vertex pairs survived reconciliation when
   one cell participates in several overlapping defect pairs — the
   isolated-site assumption of the O(defects) repair breaks.

Net framing: the two backends sit at different points of one
prevention-vs-repair design space — the sphere's prevention (weld,
canonicalization) keeps its sliver rate near zero so strict + sparse
repair holds; the plane's bias IS its prevention. Unifying on strict
requires detection completeness (item 1) and density-robust repair
(item 2) first; both are worth pursuing regardless, because they are
exactly what would make output validity machine-guaranteed independent
of any eps choice.

### Net hardening (2026-06): causal chain closed, three fixes landed

Tracing the cross-bin escape on the 402-cell lab produced the complete
causal chain, each step confirmed by instrumentation:

1. Strict rule -> adjacent cells commit a marginal corner under
   different attributions (sliver/sequence divergence).
2. The in-bin check detects the thirds mismatch but does NOT propagate
   the vertex index across the mismatched endpoint, so a later same-bin
   cell re-creates an already-emitted key: **duplicate ids for one
   abstract vertex** (live dedup's identity invariant, violated
   silently; observed directly — e.g. two ids both keyed [53,266,401]
   at bit-identical positions).
3. Cross-bin overflow matching compares THIRDS, which fully agree (both
   sides honestly name the same triple), so no record fires; the two
   edges adjacent to the duplicated corner then patch the SAME cell
   slot with different concrete references — debug-asserted only,
   silent last-write-wins in release. That was the whole detection
   escape.
4. Repair never merged same-key duplicates globally, and silently
   bailed (`continue`) on defective edges with != 1 segment per side.

Fixes (all defect-gated — clean runs are bit-identical):

- **CrossBinSlotConflict** (new `UnresolvedEdgeOrigin`): a contradictory
  slot patch now records the site; the formerly invisible far-point
  pairs surface.
- **Duplicate-key backstop** in `collect_merges`: same key = same
  abstract vertex by model definition; all duplicates union up front
  (O(V) hash scan, paid only when unresolved edges exist).
- **Multi-segment proximity repair**: the silent bail is replaced by a
  position-based union of all segment endpoints within the degenerate
  length scale, local to the defective edge.

Measured on the lab (strict bisectors, the stress configuration):
uniform 50k repairs fully (3 unpaired -> 0, strictly valid); the
402-cell cluster torture case improves 52 -> 31 — residuals need
iteration-to-fixpoint or sequence-level repair, noting this fixture is
400 generators at ~1e-3 spacing under strict clipping, far outside the
production envelope. Under the production bias both fixtures remain
0-defect, and the full suites (incl. the sphere edge-repair net) are
unchanged. The sphere inherits all three fixes — its own thirds-mismatch
defect class runs through exactly these paths.

## Migration plan

Staged, each stage behind the existing harnesses (NN contract suite, edge-repair
net, fingerprint tests, paired benches):

0. **Input canonicalization** (added after stage-1 findings). ~~Done~~ —
   results below exceeded expectations:
   - One f64 renormalization at entry (band-guarded against
     contract-violating lengths); per-builder renormalization deleted. The
     returned diagram's generators are now the canonicalized points (up to
     ~1 ulp from raw input).
   - **Required a real fix found by the contract tests**: the bisector's
     chord compensation hard-coded |g|^2 = 1; with a raw (≤1-ulp-off)
     generator, the uncompensated c error (~delta_g) amplified by 1/|n|
     misplaced 2e-6-separation twin bisectors by ~3% of the chart. Fixed
     exactly and free: scale = (|g|^2 + |h|^2)/(2|g|^2) with 0.5/|g|^2
     cached per cell (bit-identical to legacy when |g|^2 == 1).
   - **Natural defects at 2M went to ZERO** (uniform seeds 1-10; pre
     stage-0, ~1 site per ~4 seeds). The renormalization asymmetry was the
     dominant source of real cross-cell defects, not just shadow
     disagreements. Fuzz sweeps 2-4.5M strictly valid; coincidence probes
     unmoved. The edge-repair net re-pinned (CrossBinThirdsMismatch lost
     its deterministic fixture — documented coverage gap).
   - Radial-only ulp-distinct inputs now collapse to exact duplicates at
     entry (a radial ulp is the same direction; under default Weld they
     weld; under Disabled they were never supported).
   - **Defect-rate survey at scale (post stage 0)**: 3M x3 / 4M x4 / 4.5M x2
     uniform seeds show ~1 defect site per ~3 seeds (3-4 defects each, all
     in-bin, repair restores strict validity; the 3M/4M seed-3 site is the
     same generator cluster — sites survive n changes). Bimodal 1M and
     250k-group cocircular 1M: zero defects. So stage 0 cleaned 2M entirely
     but the near-tie population still produces occasional sites at 3M+
     (consistent with finding #12's f32 threshold pressure) — stages 2-4
     retain a small, now precisely-sized payload. Side find: the survey's
     clustered-cap 1M input exposed (and got fixed) an n==64 transition-mask
     overflow panic in the bitmask clipper; that input now fails cleanly
     with the vertex-budget error (a cell bordering the cap rim genuinely
     exceeds MAX_POLY_VERTICES — envelope, not defect).
   - Post-stage-0 shadow audit: disagreements at 2M dropped ~25% (129k ->
     98k); the remainder sits at 1e-5..1e-4 margins and is
     computed-vertex conditioning (ill-conditioned lerped intersections),
     not input bits — and produces zero cross-cell defects. NOTE for stage
     2: the shadow histogram conflates platonic-vs-computed deviation with
     chart-vs-chart divergence; placing EPS_FILTER needs a paired audit
     comparing the two cells' local decisions for the same 4-tuple.
1. **Canonical evaluator + escalation plumbing, decisions logged not applied.**
   ~~Done~~ (feature `p5_shadow`; findings above).
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
- **Exact arithmetic on the hot path** (CGAL-style throughout): more than the
  contract needs — "essentially Voronoi" requires agreement, not truth, and
  the filtered design keeps the hot path in f32. Note the *escalation
  evaluator* IS exact (see Pillar 1): there, exactness is not bought for
  geometric truth but because exact signs + SoS are the cheapest way to make
  tie-breaking provably coherent, and the path is cold.
- **Lower-index-generator's-chart as canonical frame**: dominated by the
  symmetric in_circle formulation, which needs no frame choice at all and is
  insensitive to which side evaluates it.
- **Live within-bin edge repair** (ledger): a second repair phase with its own
  composition proof, made moot by removing the mismatches at the source. Already
  marked deprioritized in the ledger.
