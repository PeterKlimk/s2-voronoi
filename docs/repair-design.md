# Repair design — normalized local 3D repair (2026-06-22)

Current-state note for the cold-path topology repair. Supersedes the earlier
"projected oracle, not raw 3D" diagnosis.

## Current status

- Default repair is **normalized local 3D repair** (`RepairMode::Local3d`):
  `repair_local_hull` builds one normalized local 3D hull over the implicated
  closure's gather, splices the repaired closure, grows on residuals, and accepts
  only if whole-diagram strict validation succeeds (valid-or-revert gate).
- `RepairMode::LocalProjected` (the projected single-chart oracle,
  `repair_local_exact`) remains available as an A/B diagnostic path.
- Known mega 100k defects (seeds 1, 2, 15) repair to strict validity; a broad
  sweep repaired 25 defective inputs while clean controls stayed valid.
- Contract target: the fast path is nearly always graph-correct, repair is the
  cold correctness backstop, and returned diagrams are strictly valid or the
  computation fails loudly. We do not pursue a full exact-predicate hot-path
  construction as the active product goal.

## The correction: normalize before exact 3D

The earlier conclusion that local 3D hull repair was the "wrong metric" came from
a bad reference: exact 3D hulls were computed on **raw f32** coordinates. Those
have tiny radius drift, exact predicates preserve it, and the result is a
Euclidean off-sphere hull rather than the S2 Delaunay graph.

Exact 3D spherical Delaunay = convex hull of **f64-renormalized directions**.
Once the 3D oracle renormalizes before exact predicates:

- fast gnomonic output matches normalized CGAL on tested uniform and mega inputs;
- normalized `LocalHull` matches CGAL;
- normalized local 3D repair matches projected repair on the known mega defects.

So the issue was not projection drift — it was failure to normalize. This makes
normalized local 3D repair the right backstop (still local and dependency-free,
but with an on-sphere exact oracle), not a reason to make every fast clip
decision exact.

## Probe results

External CGAL exact-hull reference:

```bash
g++ -O3 -std=c++17 scripts/cgal_hull3.cpp -lgmp -lmpfr -o /tmp/cgal_hull3
S2_CGAL_HULL3_BIN=/tmp/cgal_hull3 S2_ESCALATE_DIST=uniform S2_ESCALATE_N=1000000 \
cargo test --release --features escalate_probe --test escalate \
  probe_fast_vs_cgal_hull3_reference -- --ignored --nocapture
```

Fast vs normalized CGAL (`changed` = differing facets):

| distribution | n | seed | changed |
|---|---:|---:|---:|
| uniform | 100k | 3 | 0 |
| uniform | 500k | 3 | 0 |
| uniform | 1m | 3 | 0 |
| mega | 12k | 3 | 0 |
| mega | 100k | 3 | 1 |

Normalized `LocalHull` vs CGAL also matched at uniform 100k (`changed=0`), though
the in-repo global `LocalHull` is much slower than CGAL at that scale and should
stay a diagnostic/probe path.

Known mega 100k repair comparison:

| seed | projected repair | normalized local 3D repair |
|---:|---:|---:|
| 1 | valid, 7 spliced, 2 rounds | valid, 7 spliced, 2 rounds |
| 2 | valid, 15 spliced, 2 rounds | valid, 15 spliced, 2 rounds |
| 15 | valid, 9 spliced, 2 rounds | valid, 9 spliced, 2 rounds |

Probe switches: `S2_CGAL_HULL3_BIN=/tmp/cgal_hull3` enables the ignored CGAL
reference probes in `tests/escalate.rs`; `S2_ESCALATE_DELAUNATOR=1` routes
`escalate_probe` repair through the older global/projected delaunator oracle.

## Contract direction

Target: **nearly always correct by fast construction, always valid by
validation/repair construction.** The rare disagreements live in degenerate
regimes where f32 input, normalization policy, SoS tie choices, chart arithmetic,
lerped intersections, and kNN termination all interact; a complete symbolic proof
for the fast path would be expensive and would still need a tie-policy story.

Normalized local 3D repair is the better boundary:

1. the fast path handles ordinary cells;
2. topology residuals (`post_repair_unpaired`) plus the low-incidence backstop
   seed a normalized local 3D rebuild of only the implicated closure;
3. the repaired closure is accepted only if the whole diagram validates;
4. if repair cannot produce a strictly valid diagram, the computation fails
   loudly instead of returning a non-manifold graph.

So far every repaired case has produced the normalized 3D truth graph; we have
observed coherent fast-path disagreements in mega inputs but no strictly valid
repaired topology that coherently agrees on the wrong graph.

Practical rules: any exact 3D reference or repair oracle must f64-renormalize
direction before exact predicates; raw f32 hull disagreement is not spherical
Delaunay disagreement; treat exact-predicate fast-path work as research only.

## Code map

- `src/knn_clipping/compute.rs`: repair-mode wiring and valid-or-revert gate.
- `src/knn_clipping/escalate.rs`: normalized local 3D repair, projected A/B
  repair, and grow/splice machinery.
- `src/knn_clipping/local_hull.rs`: local 3D hull; f64-renormalizes points before
  exact predicates.
- `scripts/cgal_hull3.cpp`: external CGAL exact-hull reference utility.
- `tests/escalate.rs`: CGAL/reference/local-repair probes.
- `tests/escalate_local.rs`: default repair broad sweep.

## Open work

- Keep broad normalized local 3D repair sweeps in the regression suite, comparing
  closure sizes / rounds / acceptance / runtime against `RepairMode::LocalProjected`
  when changing repair code.
- Use CGAL/local-hull probes to hunt specifically for repaired-valid-but-wrong
  local topology. None is currently known; finding one would change the contract.
- Keep `RepairMode::LocalProjected` as an A/B diagnostic, not the final fallback
  in extreme closures.
