# Fallback Extractor + Tier-2 Re-clip Repair — Review (2026-06-20)

**Scope.** Correctness and optimization review of the two recently-built
subsystems on branch `agent/fallback-incremental-clip`:

1. **The fallback extractor** ("fallback system") — `FallbackBuilder`'s
   incremental spherical Sutherland–Hodgman clip + extraction
   (`topo2d/builder/clip.rs`, `topo2d/builder/extract.rs`). **Always live**:
   handles any cell the primary gnomonic clipper overflows (24-vertex /
   projection limit), on every distribution. Defects here ship by default.
2. **The Tier-2 re-clip repair** ("fallback to the fallback") —
   `knn_clipping/reclip_repair.rs`. **Opt-in** via `S2_RECLIP_REPAIR`,
   mega-only in practice, **off by default**. Runs on the residual unpaired
   interior edges Tier-1 (`edge_reconcile`) could not pair; bails → loud
   `residual_error`. See `docs/reclip-repair-design.md`, memory
   `fallback-rewrite-mega-bug`, `reclip-fallback-review-2026-06`.

**Method.** A 5-dimension adversarial review (soundness-gate, reclip-algorithm,
fallback-clip, determinism/termination/panics, optimization); every finding was
independently re-checked by a separate refuter agent (29 findings adjudicated,
27 survived, 2 refuted). The headline soundness gap was reproduced end-to-end,
and the gate-vs-validator narrowness was verified by hand against `validation.rs`.

**Timing caveat.** Wall-clock on the dev box is unreliable (WSL2, shared
machine). All optimization claims below are **prior-based** (complexity,
allocation, cache) — no measured speedups are asserted. Each names the
measurement that would confirm it.

---

## TL;DR

- **The repair is *not* sound as written.** Its re-detect gate
  (`reclip_repair.rs:792-882`) is a hand-rolled **subset** of
  `validation::validate`, not the equivalent it claims to be (`:798-799`). It
  can declare a repaired diagram clean while the validator rejects it →
  **silent-invalid output** when `S2_RECLIP_REPAIR=1` and `S2_VORONOI_VERIFY` is
  unset (the default). **Reproduced** (mega 1m seed 2 → `d2:1` low-incidence
  vertex). This is the one blocker before repair can move toward default or
  before the main clipper is loosened against it.
- **One always-live hot-path hole:** the fallback extractor's fallthrough can
  fabricate a geometrically-wrong vertex key that two adjacent cells agree on,
  which **even the full validator cannot catch**. Rare (needs symmetry) but
  real and on the default path.
- Smaller items: an OOB-id panic, run-to-run vertex-layout nondeterminism, an
  undefined-winding degenerate case, and a component-merge that exceeds the
  design's firewall (bounded by the cap).
- **Optimization:** the hot fallback extractor has a few clean prior-justified
  wins (kill a redundant `pair_vertex`, shrink an O(V·C²) scan, normalize once);
  the cold repair has exactly one cost lever that matters (incremental `Expand`
  instead of re-running the O(|G|³·|filter|) solve up to 8× from scratch). Most
  cold-path micro-opts are explicitly **not worth doing**.

Severity is weighted by the **hot (always live) vs cold (opt-in)** split below.

---

## Decision & roadmap (2026-06-20)

**Context that bounds the blast radius.** The Tier-2 repair only ever runs on
inputs that leave a residual Tier-1 could not stitch, and **those are never
observed outside the `mega` distribution** — `uniform`/`clustered`/`bimodal`/…
produce no unrepairable stitch errors. So the repair path is genuinely rare and
mega-specific; its current performance is acceptable as a stopgap.

**Agreed direction:**

1. **Correctness first (now).** Implement the gate→validator fix (CRITICAL) and
   the fallthrough fail-loud (HIGH). Goal: *valid graph or clean error*, with the
   guarantee bound to the repair itself (not the env-gated verifier), so neither
   the plain (VERIFY-off) nor the report path can expose invalid topology.
   Because it only fires on mega, paying full validation over the touched region
   (or even the whole diagram) is fine for now.

2. **Performance later, preserving correctness.** The resolver's cost driver is
   the tolerance brute force (#O5: all-`|G|³` triples × full-filter emptiness,
   re-run up to 8× per `Expand`), which **explodes as the number of contested
   cells N grows**. Replace it with **concrete/exact predicates** — an adaptive
   in-circle / `orient3d` (Shewchuk-style) local Delaunay with a consistent
   tie-break — which is both correct and well-scaled, *instead of* something that
   blows up with failed-N. Note: exact arithmetic was earlier rejected as
   unnecessary **for consistency** (memory `fallback-rewrite-mega-bug`); it is
   re-motivated here purely as the **performance** path, not a correctness
   prerequisite. The correctness fix must land first and must not regress when
   the resolver is later swapped.

This is recorded canonically in `docs/reclip-repair-design.md` (Status), with the
perf item in `docs/optimization-ideas.md` and actionable steps in `docs/todo.md`.

---

## Correctness

### 🔴 CRITICAL — re-detect gate is a *subset* of `validate`, not an equivalent (reproduced silent-invalid)

**Where.** Gate: `reclip_repair.rs:792-882`. Region build: `:808-817`.
Edge-only check: `:857-882`. Self-doc overclaim: `:797-799`. Contrast
`verify_sphere_fast` in `validation.rs:466-504` and `low_incidence_vertices` in
`validation.rs:489-497, 737-744`; default-off verify gate `validation.rs:326-337`;
sole backstop `compute.rs:65-66`.

**Mechanism.** The gate counts **only** directed edge-pairing
(`fwd==1 && bwd==1`) and **only** for edges flagged `u.touched`. The strict
validator additionally rejects: **low-incidence vertices (degree ≥ 3)**,
antipodal edges, off-sphere vertices, duplicate-vertex-in-cell, duplicate cells,
degenerate cells, and global connectivity / Euler == 2. A re-stitch that
abandons a shared boundary vertex drops it from degree 3 → 2 while *every
remaining edge still pairs cleanly* → the gate passes. With `S2_VORONOI_VERIFY`
off (default) and the residual empty, `compute()` returns the invalid diagram
with no error (the backstop at `compute.rs:65-66` fires only on a *non-empty*
residual).

The comment at `reclip_repair.rs:797-799` — "the SAME invariant
`validation::validate` enforces, so a clean result here means the returned
diagram is a valid subdivision (no silent ship)" — is therefore **false**.

**Three facets, one root cause.**
- (a) **No non-edge invariants.** No vertex-degree / antipodal / off-sphere /
  duplicate / Euler check at all.
- (b) **`region` is not a provable superset.** It is seeded from the
  touched cells' **new** (post-re-stitch) spans (`:808-817`), so an outside cell
  that held a *now-abandoned* vertex — present only in the **old** span — can
  fall outside `region` and never be audited.
- (c) **`u.touched` filter hides one-sided edges.** `:861` only flags an edge
  when a touched cell is incident. If a touched cell drops an edge but an
  *untouched* outside cell still holds it, that edge has one use with
  `u.touched=false` → not reported, though the validator counts it as a
  boundary (unpaired) edge.

**Geometric root.** A degree-≥4 degenerate vertex — exactly what this repair
targets — yields a generator pair with **>2** candidate endpoint keys. The edge
rule at `:537-549` adds an edge only when a pair has **exactly 2** candidates
(the `>2` case is only `trace!`-logged), while the prune fixpoint at `:494-502`
keeps any interior vertex whose pairs have **≥2** support. That asymmetry lets a
vertex survive pruning (get a vid at `:767`) yet contribute no edge to some
incident cell → its degree drops below 3.

**Reproduction.**
```
S2_RECLIP_REPAIR=1 S2_RECLIP_TRACE=1 S2_VORONOI_VERIFY=1 \
  cargo run --release --features tools --bin bench_voronoi -- \
  1m --dist mega --seed 2 --no-preprocess
```
Trace prints `[reclip] re-stitched … cell(s), … new interior vertices; 0
residual edge(s) remain` (gate declares clean), then aborts with
`ComputationFailed("S2_VORONOI_VERIFY: returned diagram failed strict
validation: 1 low-incidence vertices (d2:1)")`. With VERIFY off (default) the
same diagram returns as success.

**Fix (unifies a/b/c).** On a clean re-detect, before returning `Ok`, run
`validation::verify_sphere_fast` (or `validate_impl`) over the affected
sub-region — widened to include holders of the touched cells' **old** vids —
and fold any failure into `residual_out`. Components are ≤ 128 cells, so even a
whole-diagram `validate()` is affordable on this cold path. This makes soundness
**definitional** rather than an argued subset — which is exactly what the
design's endgame (repair → default, then loosen the hot clipper) depends on.
Closing the geometric root (#A1) as well would let more components *succeed*
rather than merely fail loud.

**The validation must attach to the repair, not to `S2_VORONOI_VERIFY`** (Codex).
The plain `compute` path loud-fails on a non-empty residual (`compute.rs:65-66`),
but the **report** path (`compute_voronoi_knn_clipping_report_core`,
`compute.rs:153-197`) folds the residual into `unresolved_edge_pairs` and
**returns the diagram anyway** — so a repaired-but-invalid diagram can be exposed
alongside diagnostics there even today. Inline validation inside `repair()` (fold
failures into residual) is therefore required regardless of caller path; relying
on the env-gated verifier leaves both the plain (VERIFY-off) and report paths
exposed.

### 🟠 HIGH — fallback extractor fallthrough can ship a mutually-consistent *wrong* key (always-live path)

**Where.** `extract.rs:388-399` (fallthrough), keyed at `:427-431`;
`pair_vertex` `:309-331`; dedup `live_dedup/assemble.rs:30,39-55`.

**Mechanism.** When `pair_vertex(plane_a, plane_b)` returns `None` (the exact
f64 plane-pair intersection violates a constraint by > `FALLBACK_PLANE_TOL`
1e-9 — i.e. it is *outside* the cell) and no split `plane_c` is found,
extraction falls through and emits the **raw f32 clip corner** with key
`sort3(gen, n_a, n_b)` — a triple it just proved does not produce an in-cell
vertex. If two adjacent fallback cells symmetrically hit this fallthrough on
their shared edge and pick the **same** triple, they agree on a wrong key.
Because dedup is **key-only**, the two collapse to one index and the cell is
emitted, not failed → a geometrically-wrong vertex ships. The full validator
**cannot** catch it: `validate` pairs edges by index and checks
on-sphere/antipodal/Euler, never equidistance/circumcenter.

**Impact / reachability.** Always-live (no flag). Requires symmetric
fallthrough on a shared edge — any asymmetry yields a one-sided/mismatched edge
that reconcile or the residual guard catches. Plausible only in
cocircular/near-degenerate (mega-like) regimes where `pair_vertex`'s 1e-9 f64
test rejects a corner the f32 clip kept.

**Fix (with a trade-off — corrected per Codex).** Treat a `pair_vertex` failure
with no split as a hard local failure: return `Vec::new()` from
`computed_active_vertices` → `NoValidSeed`. Note the propagation: this surfaces
as a **cell-build error** via `unexpected_failure_error(… "vertex extraction")`
at `cell_build/run.rs:646-655` — *before* assembly — so it is **not** repaired
by Tier-1/Tier-2/residual; it fails the **whole compute** loudly. That is the
soundness-correct outcome, but it trades availability: the *asymmetric* case
already loud-fails later (the mismatched key → unpaired edge → `residual_error`),
so the net availability loss is limited to near-degenerate cells whose fabricated
corner currently happens to pair and pass. If that loss is unacceptable, the
softer alternative is to *flag* any cell that used a fallthrough corner for the
unconditional output-invariant scan (force a synthetic detection record) so the
empty-records fast path can't skip it — keeping availability while removing the
silent-ship.

### 🟡 Smaller confirmed correctness items

| # | Sev | Path | Where | Issue | Fix |
|---|-----|------|-------|-------|-----|
| C1 | low | cold | `reclip_repair.rs:312,335,787` | `points[g]`, `grid.point_index_to_cell(g)`, `cells[g]` are unguarded; a vertex key naming an out-of-range generator (synthetic fixtures tolerate fake ids elsewhere, cf. `edge_reconcile::synthesize_backstop_records`) **panics** instead of bailing. In valid operation keys only name real ids. | Guard `gvec` to `g < points.len()` at the top of `resolve_component_attempt`/`repair`; out-of-range → bail to residual. |
| C2 | low | cold | `reclip_repair.rs:767-774` (`interior_pos` is `std::HashMap`, `:280-284`) | **Determinism:** new interior vids are assigned in `std::HashMap` iteration order (per-process random seed), so output vertex array order **and** per-cell vid references differ run-to-run. Topology / positions / key-set are invariant; the layout is not. The "re-clip is deterministic (ST)" note is false for byte-output. | Collect keys into a `Vec`, `sort_unstable()`, then assign vids in sorted order (or use `BTreeMap`). Negligible cost (≤128 cells). |
| C3 | low | cold | `reclip_repair.rs:244-252` (`build_cycle_from_edges`) | Winding is fixed by `if signed < 0.0 { reverse }`; at `signed == 0.0` (cell + generator coplanar through sphere center — degenerate near-great-circle) the orientation is left arbitrary → possible same-direction shared edge. Currently caught downstream as residual. | Treat `|signed| < tol` as degenerate → bail (`None`), or break the tie against a fixed reference. |
| C4 | low | cold | `reclip_repair.rs:102-127` (working-tree co-occurrence merge) | The new merge unions a generator with any co-occurring key member in `named_set`, i.e. across already-**paired** shared vertices — broader than the design's "transitive closure of *contested* adjacency". On a dense mega blob this can coalesce otherwise-separable components, weakening the firewall. Cannot pull in non-`named_set` cells; hard-capped at `MAX_COMPONENT_CELLS=128` (sound, just more bails). | Keep the cap as backstop. Optionally restrict the union to contested vertices, or document the separability/joint-resolution trade-off + add a dense-mega test that pushes the merged component near 128 and asserts a loud bail. |
| C5 | nit | cold | `reclip_repair.rs:779-784` | **Defensive (not currently reachable).** Re-stitch resolves a poly key by `interior_vid` *first*, `boundary_pin` second. Codex flagged a recomputed interior vertex shadowing a pinned boundary vid. **Verified: the two key sets are disjoint** — every boundary-pin key is a vertex of an outside cell `h` and so names `h ∉ gset`, while interior keys are all-in-`gset` — so the shadow cannot trigger. But the ordering and the `.expect("validated above")` both silently rely on that unstated invariant; if a future boundary-recovery change emits an all-`gset` key, it becomes a silent mis-pin. | Prefer `boundary_pin` for pinned keys (swap the `or_else`), or assert disjointness, to make the invariant explicit rather than load-bearing-by-luck. |

### 🟡 FLAGGED (Codex — review under-covered this) — gnomonic→fallback handoff inherits stale edge-plane ownership

**Where.** `builder.rs:294-341` (`from_gnomonic`); `SphericalPoly::from_gnomonic`
(`builder.rs:362-410`); per-corner plane pair read at `extract.rs:352-354`;
geometric re-derivation at `extract.rs:282-307` (`active_candidate_planes`).

**Mechanism.** On handoff the fallback **rebuilds all constraints** as freshly
normalized 3D spherical bisector planes (`from_neighbor`, `projection.rs:79-90`),
but **inherits the starting polygon from the gnomonic 2D model** (corner
positions projected to 3D, and per-edge `edge_planes` carried over by index),
replaying only the single overflow-causing constraint (`:325-337`). Extraction
then reads each corner's defining pair `(plane_a, plane_b)` directly from those
**inherited** `edge_planes` (`extract.rs:352-354`) and keys the corner
`sort3(gen, n_a, n_b)` from them. Near degeneracy the gnomonic chart may have
rounded a corner's owning planes differently than the 3D model would, so the
inherited pair — and thus the emitted key — can be **stale** relative to the
freshly-normalized 3D constraints. This is the same failure *class* as the
fallthrough (a key inconsistent with the cell's actual 3D geometry): usually a
one-sided edge caught by reconcile/residual, silently wrong only if symmetric.

**Partial mitigation already present.** `active_candidate_planes` re-derives
touching planes geometrically (not just from `edge_planes`), and the split search
re-tests `pair_vertex`, so the staleness is *partly* self-correcting — but the
corner's base pair at `:352-354` is still taken from the inherited metadata.

**Status.** Not proven to produce a concrete defect here; flagged because the
review under-examined the handoff and Codex independently rated it a real risk
surface. Needs a focused audit (medium, hot path): confirm `from_gnomonic`'s
`edge_planes` index-mapping stays consistent with the rebuilt constraints, and
consider re-deriving each corner's owning pair from the 3D model rather than
trusting inherited `edge_planes`.

### Refuted (do not chase)

- **"1000× clip-vs-extract tolerance gap (1e-6 vs 1e-9) drives corners into the
  fallthrough."** Arithmetic is correct (both tolerances inherit the same
  non-unit-normal `1/θ` scaling, constant 1000× gap) but the causal conclusion
  is wrong — the gap is intentional and not the fallthrough trigger. **Refuted.**
- **"`shared_edge_constraint` returns the wrong neighbor for a repeated-neighbor
  sliver (debug panic)."** The shared neighbor is present in both endpoint keys
  by construction. **Refuted.**
- **Positive audit:** the always-live fallback clip/extract path was checked for
  panics/OOB/NaN on adversarial input and came back **clean** — all cross-product
  magnitudes and `recip()`s are guarded (`clip.rs:311,317`; `extract.rs:314`),
  all plane indexing is bounds-checked, and a surviving bounding-box pseudo-edge
  sentinel (`usize::MAX`) fails the whole cell to `NoValidSeed` rather than
  emitting an open chain as closed (`extract.rs:355-362`).

---

## Optimization (prior-based only)

### Hot path — fallback extractor (worth doing)

| # | Where | Opportunity | Prior / magnitude | Confirm with |
|---|-------|-------------|-------------------|--------------|
| O1 | `extract.rs:373-387` | The `find` closure already evaluates both `pair_vertex(plane_a,plane_c)` and `pair_vertex(plane_c,plane_b)` to test `.is_some()`; lines `:382,:385` then recompute the **winning** pair a third time to extract the value. | Use `find_map` returning the resolved `FallbackVertex`s. Drops 3 `pair_vertex` calls on the winner to 2 — removes one O(C) `satisfies_all_constraints` scan + cross + normalize per split corner. Pure restructure, no semantic change. **Cleanest win.** | per-corner `pair_vertex` call count; `perf stat` instr count on a fallback-heavy mega repro. |
| O2 | `extract.rs:343-409,244-248` | `computed_active_vertices` is **O(V·C²)**: per corner the `find` scans ≤C candidates, each calling `pair_vertex` twice, each doing an O(C) `satisfies_all_constraints` flat `.all()` over **all** constraints with no pruning/ordering. C is the *overflow-cell* constraint count — not tiny. | (1) Restrict `satisfies_all_constraints` to `active_candidate_planes()` (boundary directions need only the active planes). (2) Order the scan so planes nearest the test direction are first → `.all()` short-circuits faster on the common reject. | `perf stat` retired instr/branches attributed to `to_vertex_data_full`, fallback-heavy mega. |
| O3 | `projection.rs:83-90` | `from_neighbor` calls `generator.normalize()` on **every** constraint (once per accepted neighbor + once per replayed neighbor on gnomonic→fallback handoff). Generator is constant per cell. | Normalize once at `FallbackBuilder` construction; pass the unit generator in. Removes O(C) redundant sqrts/cell. Micro but free, zero semantic change. | code inspection (value is bit-identical each call). |
| O4 | `clip.rs:349-404,439` | `clip_poly_with_constraint` allocates two fresh `Vec`s (cap n+1) per constraint, then moves into `self.poly`, freeing the prior buffers — ~2·C alloc/free per fallback cell, no reuse. The gnomonic path deliberately ping-pongs two `PolyBuffer`s. | Reuse a scratch `SphericalPoly` via `mem::swap`. **Lowest hot priority** — fallback cells are rare vs total, so absolute volume is small even on mega; pursue last. | count fallback-cell invocations on mega; skip if ≪1% of cells. |

### Cold path — repair (only one item matters)

| # | Where | Opportunity | Prior / magnitude | Confirm with |
|---|-------|-------------|-------------------|--------------|
| O5 | `reclip_repair.rs:650-673,327-449` | `resolve_component` re-runs the **entire** `resolve_component_attempt` from scratch on every `Expand` (up to 8×): rebuilds the grid filter, the boundary recovery, and the **O(\|G\|³·\|filter\|)** interior brute force. On the dense mega clusters it targets, `filter ≈ 8.5k–9.4k` and `\|G\|` grows toward 128 → this is the dominant cost and the one cold lever that can threaten budget/latency. | Make expansion incremental: filter is monotone under `Expand` (accumulate, don't rebuild); prior triples are a subset (only re-test triples touching the added generators, re-checking emptiness against new filter gens only); memoize `gjit(g)`. | the documented mega repro producing the largest contested component; phase timing on `edge_reconcile`. |
| O6 | `reclip_repair.rs:466-512` | Prune fixpoint rebuilds `pair_count` over all interior+boundary keys every iteration. Bounded (`before==after` guarantees progress), but re-derives work. | Fold into O5's incremental rework via a worklist (decrement the two pairs of each removed key, re-queue only affected). **Not worth standalone.** | n/a — bundle with O5. |
| O7 | repair-path `HashMap`s | `std::HashMap` (SipHash) vs the crate's `FxHashMap` elsewhere. | **Explicitly NOT worth doing.** Opt-in, rare; hashing is dwarfed by the O5 cubic solve. (The only marginal reason — determinism — is better fixed by C2's sorted assignment.) | n/a. |

### Suggested order

1. **Gate → validator equivalence** (CRITICAL; unblocks the design's endgame).
2. **Fallthrough → loud `NoValidSeed`** (HIGH; only hot-path silent-invalid hole).
3. **C1 OOB guard + C2 determinism sort** (cheap, low-risk, both clean).
4. **O1 (`find_map`) + O3 (normalize-once)** (clean hot wins).
5. **O5 incremental `Expand`** — only after correctness is locked.
6. Optional deeper fix: disambiguate the degree-≥4 ">2 candidate endpoints"
   geometric root (#A1) so the resolver *succeeds* on those vertices rather than
   relying on `Expand` to remove the ambiguity.

---

## Second opinion (Codex gpt-5.5, read-only)

An independent Codex `exec` review (read-only sandbox, high reasoning) was run
against the live code with this document. Outcome:

- **CONFIRM — CRITICAL** (gate is a subset of `validate`; silent-ship real; all
  three facets correct). Recommends **whole-diagram validation** over regional —
  "regional validation is easier to get subtly wrong." Added the **report-path
  exposure** point (`compute.rs:153-197` returns the diagram even with a residual)
  → validation must attach to `repair()`, not the env-gated verifier.
- **CONFIRM — HIGH**, with a factual correction now folded in: `NoValidSeed`
  becomes a cell-build error at `run.rs:646-655` (pre-assembly), **not**
  Tier-1/Tier-2-repaired — so the fix fails the whole compute loudly (trade-off
  documented above).
- **CONFIRM** — the prune(`≥2`)/edge(`==2`) asymmetry is the mechanism; deeper
  root is "no tie-break for >2 endpoints."
- **RE-RATE / nuance** — agrees boundary/winding are mostly loud-fail (consistent
  with the `low` ratings here); says the review **under-rated the
  gnomonic→fallback constraint-replay staleness** → added as a FLAGGED item.
- **MISSED (Codex), adjudicated here:**
  - *Interior vid shadows boundary pin* (`:779-784`): **not currently reachable**
    — the key sets are disjoint (verified). Recorded as defensive item **C5**.
  - *Validator-as-gate must bind to the repair, not `S2_VORONOI_VERIFY`*:
    **valid** — folded into the CRITICAL fix (report-path note).

Net: no finding was overturned; the critical and high calls are corroborated by a
second model, one propagation detail was corrected, and two items were added
(report-path exposure; gnomonic→fallback staleness).

## Appendix — invariants the strict validator enforces (gate must match)

From `verify_sphere_fast` / `validate_impl` (`validation.rs`), the gate currently
covers only the **bold** one:

- **edge used exactly twice, once per direction** (unpaired / overused /
  misoriented) — `:519-522` — *gate covers this*.
- self-loop edge `a==b` — `:469-470`.
- antipodal edge (`dot ≤ -1+ε`) — `:475-477`.
- off-sphere vertex (`|len²-1| > ε`) — `:500-503`.
- duplicate vertex within a cell — `:439-440`.
- degenerate cell (< 3 distinct vertices) — `:451-452`.
- duplicate cell (identical signature) — `:460-462`.
- **low-incidence vertex (referenced by < 3 cells)** — `:493-494` — *the
  reproduced gap*.
- connectivity / Euler characteristic == 2 — `:529-552`.
