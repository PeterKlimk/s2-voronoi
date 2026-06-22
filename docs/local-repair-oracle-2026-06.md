# Local Repair Oracle - Current Understanding (2026-06-22)

This note supersedes the earlier "projected oracle, not raw 3D" diagnosis.
The root issue in the old 3D probes was **not projection drift**. It was that
the exact 3D hull oracle was run on raw f32 coordinates with tiny radius drift,
so it solved a Euclidean off-sphere hull problem instead of the crate's S2
problem.

The crate contract is spherical: inputs are meant to be unit-normalized, and the
pipeline canonicalizes directions once at entry. Exact reference construction
must do the same. For exact 3D hulls this is load-bearing: f32 radius error is
small geometrically, but exact predicates preserve it and can flip hull facets.

## Corrected Model

- Exact 3D spherical Delaunay = convex hull of **normalized directions**.
- Gnomonic, stereographic, and 3D hull references agree on the tested S2 inputs
  once the 3D oracle renormalizes before exact predicates.
- The earlier "raw 3D disagrees with fast" and "local 3D repair cascades"
  results were artifacts of comparing against / repairing with an off-sphere
  reference.
- Projection is still part of the fast clipper implementation, but it is no
  longer the active explanation for the observed local-repair failures.

## Current Repair State

Production default repair is now `repair_local_hull`: one normalized local 3D
hull over the implicated closure's gather, followed by grow/splice and a
whole-diagram strict-valid gate. This avoids the single-chart/pole failure mode
of projected repair in extreme closures.

Projected local repair (`repair_local_exact`) remains available as
`RepairMode::LocalProjected` and as a probe/A-B path. On the known mega 100k
defects normalized local 3D repair matches the projected repair's behavior:

| seed | projected repair | normalized local 3D repair |
|---:|---:|---:|
| 1 | valid, 7 spliced, 2 rounds | valid, 7 spliced, 2 rounds |
| 2 | valid, 15 spliced, 2 rounds | valid, 15 spliced, 2 rounds |
| 15 | valid, 9 spliced, 2 rounds | valid, 9 spliced, 2 rounds |

This makes normalized local 3D the right final backstop candidate: it is still
local and dependency-free, but its exact oracle is the normalized S2 3D graph
rather than a particular projected chart.

With the normalized-truth result, there is no reason to broaden local repair.
The right repair shape is still surgical:

- trigger from real topology residuals (`post_repair_unpaired`) plus the existing
  low-incidence backstop;
- rebuild only the implicated closure;
- accept only if whole-diagram strict validation succeeds;
- keep projected repair available for A/B diagnostics, not as the final fallback
  in extreme closures.

## Reference Probes

External CGAL probe:

```bash
g++ -O3 -std=c++17 scripts/cgal_hull3.cpp -lgmp -lmpfr -o /tmp/cgal_hull3
```

Fast vs normalized CGAL:

```bash
S2_CGAL_HULL3_BIN=/tmp/cgal_hull3 S2_ESCALATE_DIST=uniform S2_ESCALATE_N=1000000 \
cargo test --release --features escalate_probe --test escalate \
probe_fast_vs_cgal_hull3_reference -- --ignored --nocapture
```

Measured:

| distribution | n | seed | changed vs normalized CGAL |
|---|---:|---:|---:|
| uniform | 100k | 3 | 0 |
| uniform | 500k | 3 | 0 |
| uniform | 1m | 3 | 0 |
| mega | 12k | 3 | 0 |
| mega | 100k | 3 | 1 |

CGAL vs normalized `LocalHull` also matched at uniform 100k (`changed=0`), though
the in-repo global `LocalHull` implementation is much slower than CGAL at that
scale and should remain a diagnostic/probe path.

## Contract Direction

The current target is **nearly always correct by fast construction, always valid
by validation/repair construction**. We are no longer trying to turn the fast
gnomonic clip loop into a fully exact-predicate construction.

The reason is practical and mathematical: the observed rare disagreements live
in degenerate regimes where f32 input, normalization policy, simulation-of-
simplicity tie choices, chart arithmetic, lerped intersections, and kNN
termination all interact. The fast clipper can be made better, but a complete
symbolic proof for that path would be expensive and would still need a tie
policy story.

Normalized local 3D repair is the better boundary:

1. fast path handles ordinary cells;
2. topology residuals seed a normalized local 3D rebuild;
3. the repaired closure is accepted only if the whole diagram validates;
4. if repair cannot produce a strictly valid diagram, the computation fails
   loudly instead of returning a non-manifold graph.

So far, repaired cases have produced the normalized 3D truth graph in every
observed probe. We have observed fast-path coherent disagreements in mega
inputs, but we have not observed a strictly valid repaired local topology that
coherently agrees on the wrong graph.

## Practical Rules

1. Any exact 3D reference or repair oracle must f64-renormalize the input
   direction before exact predicates.
2. Do not interpret raw f32 convex hull disagreement as spherical Delaunay
   disagreement.
3. Do not treat "local 3D repair cascades" as a settled result unless the probe
   normalizes before exact predicates.
4. Treat exact-predicate fast-path work as research only. It should not block
   the repair-backed contract.

## Open Work

- Keep broad normalized local 3D repair sweeps in the regression suite.
- Use CGAL/local-hull probes to look specifically for repaired-valid-but-wrong
  local topology. None is currently known.
- Keep `RepairMode::LocalProjected` as an A/B diagnostic, not as the final
  fallback in extreme closures.
