# Re-clip Repair — Design

Status (2026-06-20): **PROPOSED / implementation starting.** Tier-2 repair that
re-resolves the rare local clusters the cheap stitch (`edge_reconcile`) cannot,
by jointly re-clipping a connected component of contested cells against a pinned
boundary of already-valid cells. Grew out of diagnosing the spherical fallback
rewrite's failures on the `mega` distribution; see memory
`fallback-rewrite-mega-bug` and [[fallback_builder.md]].

## Problem

Some inputs produce **high-degree degenerate Voronoi vertices** — 4+ generators
that are cocircular to within the predicate's numerical resolution. Each cell
clips in its **own** generator-centered gnomonic chart (see
[architecture.md](architecture.md), [p5-consistency-design.md](p5-consistency-design.md)),
and the same in-circle predicate rounds differently across charts. At a
degenerate vertex, adjacent cells therefore make **different discrete choices**
about which generator triple forms the corner. They emit different vertex
*keys* for the same point → the shared edge gets mismatched endpoints → an
**unpaired interior edge** → not a valid subdivision.

`mega` (a huge fraction of points in a tiny cap) manufactures these by the
dozen; general-position inputs (`uniform`, `clustered`, …) essentially never
do, which is why they pass. The fallback (3D-sphere kernel) is the dominant
*contributor* on `mega` because it disagrees with its gnomonic neighbors, but
~13% of the residual edges are pure gnomonic↔gnomonic — the mechanism is
whole-pipeline, not fallback-specific.

### Why the current repair can't fix these

`edge_reconcile` (see [live_dedup.md](live_dedup.md)) is deliberately narrow and
**sound** (it never silently returns invalid topology — any surviving unpaired
interior edge becomes a loud `residual_error`, modulo the debug-asserted
detection-completeness fast path). Its repair *power* is limited to:

- dropping spurious collinear (repeated-generator) vertices (exact),
- **1-to-1** segment mismatches → force-merge endpoints (any distance), and
- one-sided ε-edges / proximity merges within `RECONCILE_DEGENERATE_LEN_EPS` (1e-6).

A high-degree degeneracy is neither: the disagreement is **multi-way** (not 1-1)
and **feature-scale** (~6e-4 rad, ~600× the 1e-6 proximity bound — because a
flipped *discrete* decision moves a vertex to the alternative trio's location,
not by float noise). So it falls in the gap and survives → residual → reject.

## Detection is complete and cheap (the gate is reliable)

Confirmed by review: every emitted cell edge with `neighbor != self` has exactly
one directed detection owner (cross-bin overflow from each surviving side, or
same-bin earlier→later forwarded check), so **any** surviving one-sided interior
edge produces ≥1 `UnresolvedEdgeMismatch` record (`edge_checks.rs`). The records
already name the contested cells. Detection is inline in assembly — no extra
pass. The worst case is a *loud reject*, never silent corruption.

## Design: tiered repair, re-clip as Tier 2

- **Tier 0 (assembly):** key-merge — same-triple vertices unify; no defect.
- **Tier 1 (`edge_reconcile`, unchanged):** cheap, conservative, provably-safe
  moves. Absorbs the common ~1–20 defects/run. Kept inside its safe envelope —
  *not* made more aggressive (geometry-free multi-way force-merging is unsafe).
- **Tier 2 (re-clip, NEW):** runs only on Tier-1's residual. Jointly re-resolves
  each contested connected component against a pinned valid boundary.
- **Budget:** if the contested region exceeds a cap, fail loud (preserves
  soundness; bounds cost on pathological all-degenerate input).

### The component and why its boundary is safe to pin

Define a component as the **transitive closure of contested adjacency**: keep
absorbing any cell that shares a *contested* (residual) edge. Two structural
consequences:

1. **Every boundary edge (component↔outside) already paired** — it is
   non-residual, i.e. the two sides already agreed. A *contested* vertex can
   never sit on the boundary: if it did, the outside cell would also have an
   unpaired edge there → it is itself contested → the closure pulls it in →
   the vertex becomes interior. So pinning the boundary only freezes
   already-consistent geometry; it can never be an over-constraint.
2. **All disagreement is strictly interior** to the component.

### The primitive: boundary-constrained joint re-resolution

This is intentionally **not** independent per-cell clipping — independent
clipping *is* the disease (each chart rounds differently → disagreement). The
re-clip does one shared computation for the whole component:

- **Boundary (already-consistent edges/vertices):** fixed anchors. The interior
  re-resolution snaps to them — they are ground truth.
- **Interior (contested region):** free joint clipping. The clipper *may* adjust
  or drop a marginal constraint — that is allowed — but the decision is made
  **once, shared by all component cells** (one frame), so it can never produce a
  cross-cell mismatch. We are not forbidding the clipper's geometric judgment;
  we are making it shared instead of per-cell.

Because the boundary vertices are fixed anchors the interior must terminate at,
the cleanest primitive is a **constrained** local triangulation — Delaunay of
the component's generators with the boundary loop's vertices pinned — rather
than a free local Voronoi whose seam we then hope matches the boundary. (A
`delaunay.rs` dual already exists in the crate; assess for reuse.)

Consistency, not exactness, is what makes the output valid: one shared
computation (any precision, deterministic tie-break) guarantees all component
cells agree. Exact/SoS arithmetic is an **optional later knob** for run-to-run
tie determinism and geometric correctness among near-ties — not a prerequisite,
and never needed globally.

### Cascade is bounded (firewall)

Pinning the valid boundary firewalls propagation: a re-clip can only change
geometry inside the contested closure, and the closure stops at the first
non-degenerate (paired) boundary. Touching components are merged and resolved as
one unit (avoids ping-pong). A fixpoint re-detect after re-clip catches any
under-scoped component; the budget caps the pathological tail.

## De-risking measurements (mega, post dedup-fix `78936f4`)

Connected-component structure of the residuals (component = generators
co-occurring at contested vertices):

| input         | residual edges | components | largest (gens) | % of N |
|---------------|---------------:|-----------:|---------------:|-------:|
| 500k seed 3   | 37             | 3          | 27             | 0.005% |
| 500k seed 1   | 50             | 4          | 20             | 0.004% |
| 500k seed 2   | 37             | 2          | 30             | 0.006% |
| 500k seed 7   | 19             | 4          | 11             | 0.002% |
| 1m  seed 3    | 95             | 8          | 49             | 0.005% |

Components are **small, separable, and scale as more pockets — not bigger ones**
(largest ~tens of generators even at 1m). Every component touched a fallback
cell, so the gnomonic-only residual edges sit *inside* fallback-anchored
components (ripple one ring out), not as independent defects. Implication: a
component-level re-clip would **make `mega` pass**, not merely harden — and the
firewall/closure terminates small in practice.

## Plumbing

`reconcile_unresolved_edges` currently receives only `vertices + cells + keys`.
Re-clip additionally needs `points` (generator positions) and ideally the
spatial grid (for the boundary 1-ring); both are owned by `compute.rs` at the
call site, so this is an interface change, not a new data source.

## Test plan

- Reproduce: `bench_voronoi 500k --dist mega --seed 3 --no-preprocess` (and
  seeds 1/2/7, 1m) must reach a clean `S2_VORONOI_VERIFY=1` pass after re-clip.
- Unit: synthetic cocircular degree-4/5 fixtures (cf. `tests/coincidence_probes.rs`,
  `tests/edge_repair_net.rs`) with a known correct local resolution.
- Invariant guard: pin the two contracts re-clip relies on — every emitted edge
  with `neighbor != self` has a detection owner; endpoint keys contain the
  edge's two generators (debug-asserted in `third_for_edge_endpoint`).
- Differential: re-clipped region must satisfy the output invariant (every
  interior edge used by exactly two cells) and not regress the non-mega soak.
- Budget: a synthetic all-degenerate input must fail loud, not spin.

## Future: weaker main clip (optimistic concurrency)

Once the Tier-1+Tier-2 net is trusted, the hot main clipper (the dominant
profile cost, see memory `clipper-dominates-profile`) can be deliberately
loosened — looser tolerances, skipped escalation, more f32 — trading a higher
(but caught-and-repaired) defect rate for speed. Gated by the defect budget so
correctness never depends on the main clip being conservative. Stage strictly
*after* re-clip is proven; watch the defect-rate ↔ re-clip-cost balance.
