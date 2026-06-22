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

## Exact-By-Construction Detection

The starting point is not "make every clip exact"; the measured target set is too
small for that to be the first move. The useful probe is:

```bash
S2_CGAL_HULL3_BIN=/tmp/cgal_hull3 S2_ESCALATE_DIST=mega S2_ESCALATE_N=100000 \
cargo test --release --features escalate_probe --test escalate \
probe_cgal_hull3_flag_recall -- --ignored --nocapture
```

This compares fast cells against normalized CGAL truth, then sweeps a cheap
normalized-3D near-cocircularity margin over the fast fan. On mega 100k seed 3:

- changed vs normalized CGAL: 1 cell (0.0010%);
- topology-defect cells from the assembled fast graph: 2;
- `flag ∪ topology_defect` recalled the changed cell even with no margin-band
  trip;
- a `1e-6` normalized 3D margin band also recalled it directly, flagging 4.746%
  of cells on that case.

That suggests the exact-by-construction path should be developed as a detector
and escalation pipeline:

1. keep the reactive topology-defect trigger as a guaranteed validity backstop;
2. add a cheap normalized-3D near-cocircular flag probe for valid-but-wrong cells;
3. rebuild flagged/defective cells with a normalized exact local oracle;
4. measure recall against CGAL before attempting a hot-path exact clipper.

See `docs/proactive-correctness-audit-2026-06.md` for the flag/error-budget
audit plan covering projection error, lerp drift, termination bounds, and KNN
coverage.

## Practical Rules

1. Any exact 3D reference or repair oracle must f64-renormalize the input
   direction before exact predicates.
2. Do not interpret raw f32 convex hull disagreement as spherical Delaunay
   disagreement.
3. Do not treat "local 3D repair cascades" as a settled result unless the probe
   normalizes before exact predicates.
4. The remaining decision is empirical: run the broad sweep with the new
   `RepairMode::Local3d` default and compare closure/runtime against
   `RepairMode::LocalProjected`.

## Open Work

- Broad sweep normalized local 3D repair across the existing mega/clustered/
  bimodal/adversarial cases, with `RepairMode::LocalProjected` as the A/B
  comparator.
- Sweep `probe_cgal_hull3_flag_recall` across larger uniform and mega seeds to
  find the smallest band with zero changed-cell misses.
- If exact-by-construction mode is pursued, start from `flag ∪ topology defect`
  escalation using normalized 3D hull / local hull as the canonical graph
  reference, with CGAL as the external audit oracle.
