# Proactive Correctness Audit (2026-06-22)

Goal: move from "fast graph plus exact local 3D repair for detected topology
residuals" to a defensible **correct-by-construction** story.

The remaining gap is a coherent but wrong fast graph: every emitted edge pairs
and validation passes, but one or more cells disagree with the normalized 3D S2
truth graph. With the corrected normalized reference, we have not yet observed
this as a meaningful class, but a proof or sound detector still needs to cover it.

## Proposed Contract

For normalized input directions, output is correct if every cell is either:

1. certified by the fast path to match normalized 3D Delaunay locally, or
2. flagged and rebuilt by the normalized local 3D repair oracle.

The flag must be a **cell/escalation trigger**, not an in-place decision override.
Older P5 probes showed per-decision mask overrides can manufacture cross-cell
contradictions at a band boundary. A flag should only decide which region to
rebuild with one coherent normalized 3D oracle.

## Truth Model

The reference graph is the Delaunay graph of f64-renormalized S2 directions:

```text
q_i = normalize(f64(p_i.x), f64(p_i.y), f64(p_i.z))
```

CGAL `Convex_hull_3` and in-crate `LocalHull` are valid truth oracles only under
that normalization. Raw f32 radius drift is not part of the S2 contract.

## Error Sources To Bound

### 1. Input Radius Drift

The fast clipper works from canonicalized f32 directions and its gnomonic
bisector coefficients intentionally preserve tiny f32 radius terms. Normalized
truth removes them. A proactive flag must include a radius-drift term:

```text
delta_norm_i = ||p_i|| - 1
```

Any sign/margin test against normalized truth needs headroom for the difference
between the fast chord-bisector question and the normalized S2 question.

### 2. Projection / Chart Error

Fast clipping evaluates each cell in that generator's gnomonic chart. The exact
S2 decision for a candidate `h` clipping a vertex `(g, a, b)` is a normalized
3D orientation/cocircularity question. The fast chart decision is equivalent in
exact arithmetic only after the chart transform and bisector construction are
exact. A certification margin must therefore account for:

- f64 basis construction for `g`;
- projection coefficient rounding for `h`;
- per-cell chart differences for the same geometric question;
- `CLIP_EPS_INSIDE == 0`, so no intentional inside slack exists.

This argues for a **question-intrinsic normalized 3D margin** as the flag metric,
not the local chart signed distance. The current `probe_cgal_hull3_flag_recall`
is the right empirical starting point.

### 3. Lerp / Vertex Drift

When a clip changes the polygon, the new 2D vertex is created by

```text
t = d0 / (d0 - d1)
u = u0 + t * (u1 - u0)
v = v0 + t * (v1 - v0)
```

This is stable when `|d0 - d1|` is comfortably nonzero. Near a tangency or short
edge, `t` can amplify distance error. A certification flag needs either:

- a lower bound on `|d0 - d1|`, or
- a derived bound on the resulting normalized 3D vertex displacement, or
- a simple policy: if the transition is poorly conditioned, flag the cell.

This is separate from final vertex coordinate accuracy. A wrong `t` can change a
future keep/drop decision even if the emitted graph stays coherent.

### 4. Early-Unchanged / Support Rejection

The fast clipper can skip a clip when the half-plane is certified not to cut the
current polygon. For correctness-by-construction, these skips need the same
margin accounting as explicit vertex decisions. If the support/early-unchanged
clearance is within the projection/radius/lerp error budget, flag instead of
treating the skip as certified.

### 5. kNN / Termination Bounds

Even perfect clipping is only correct if every unprocessed generator is proven
unable to cut the cell. The current termination certificate compares a
`max_unseen_dot_bound: f32` against a gnomonic-derived threshold widened by:

```text
TERMINATION_ANGLE_PAD
TERMINATION_THRESHOLD_GUARD
```

The cube-grid frontier bounds also depend on f32 cell cap bounds:

```text
GRID_PLANE_PAD
GRID_SIN_EPS
```

This needs a separate audit from clip decision error. The proof obligation is:

```text
for every unseen h:
  dot(g, h) <= max_unseen_dot_bound + bound_error
```

and the termination threshold must be low enough that no such `h` can introduce
a normalized 3D Delaunay face for `g`. If this cannot be proven cheaply, the
fallback is a coverage flag: near-threshold termination marks the cell for exact
local 3D rebuild.

### 6. Fallback Spherical Clipper

The fallback path is used near the projection limit and has its own f64/f32
normalization and tolerances (`ON_PLANE_TOL`, spherical edge intersections, f32
vertex storage). It should probably be treated as **uncertified by default** for
correct-by-construction mode: if fallback participates, flag/rebuild the cell.

## Candidate Flag Shape

A first sound-ish flag can be conservative:

```text
flag cell if:
  any normalized-3D quad margin <= B_quad
  OR any transition has |d0 - d1| <= B_lerp
  OR any early-unchanged/support skip clearance <= B_skip
  OR termination margin <= B_term
  OR fallback path was used
  OR topology validation finds an unpaired/low-incidence residual
```

Flagged cells are not individually patched. They seed the existing normalized
local 3D repair closure, which grows until the whole diagram validates.

## Empirical Work Before Hot-Path Implementation

1. Sweep `probe_cgal_hull3_flag_recall` across larger uniform and mega seeds.
   Record changed cells, topology-defect cells, and recall by normalized 3D
   margin band.
2. Add a termination audit probe: for sampled builds, compare each cell's final
   termination point against brute-force all-point exact local 3D truth, and
   report near-threshold terminations.
3. Add a clip trace probe under `p5_shadow` or `escalate_probe` that records, per
   cell, minimum values for:
   - normalized 3D quad margin;
   - `|d0 - d1|` for transitions;
   - early-unchanged clearance;
   - termination clearance.
4. Confirm every CGAL-changed cell is covered by one of the flags above at an
   acceptable trip rate.

## Current Working Hypothesis

With normalized truth, coherent fast-vs-exact disagreements appear absent or
vanishingly rare in current probes. The likely production strategy is therefore:

```text
fast clip
  -> if no flags and no topology residual: accept as certified
  -> else normalized local 3D repair + strict validation gate
```

The hard part is not the repair anymore. It is making the "no flags" condition a
defensible certificate rather than just an empirical observation.
